//! Pre-launch custody for one original P-C Collection attempt.
//!
//! This module is not yet wired into the resource launcher. In particular,
//! creating a key does not register the resource owner: the launcher must
//! register its actual `NativeResourceCustody` immediately after creation,
//! before admission or worker startup. The process-local registry retains the
//! original slot even if every external key is dropped. Registry removal is
//! intentionally unavailable until same-owner settlement proof is integrated;
//! consequently this WIP must not be enabled for repeated production runs.
//! None of these operations establishes a fence, validates an output handoff,
//! or authorizes settlement or resource release.
use super::pc_lock_v3::{ManagedOutputOwnerV3, PcCollectionOwnersV3};
use crate::rovent_collect::pc_hooks_lock_v3::{
    PcSupervisorStopAuthorizationV3, RoventPcCollectionBridgeV3,
};
use crate::rovent_collect::pc_worker_v3::{
    PcDriverEvidenceErrorV3, PcDriverPollV3, PcSupervisorBeforeAdmissionJoinProofV3,
    PcSupervisorJoinedProofV3, PcSupervisorNoWorkerProofV3, PcWorkerSupervisorV3,
    SupervisorStartCustodyV3,
};
use crate::rovent_resource_scope::launch::{
    NativeResourceCustody, RoventPcCollectionRecoveryKeyV3, RoventPcCollectionResourceOutcomeV3,
    RoventPcCollectionSettledOutcomeV3,
};
use std::any::Any;
use std::sync::{Arc, Mutex, OnceLock, TryLockError};

/// A busy original owner is pending, never a completed stop or join.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PcStopStepV3 {
    Pending,
    Ready,
}

/// Game-join observation is separate from driver and GPU completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PcGameJoinObservationV3 {
    Pending,
    Bound,
    Refused,
    Failed(&'static str),
    Panicked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PcRunRecoveryStateV3 {
    Prepared,
    Executing,
    Returned,
    Panicked,
}

#[derive(Debug)]
pub(crate) enum PcRunRecoveryRecordErrorV3 {
    Busy,
    Poisoned,
    Invalid,
}

struct OriginalPcRunV3 {
    owners: Arc<PcCollectionOwnersV3>,
    output: Arc<ManagedOutputOwnerV3>,
    resource: Option<Arc<Mutex<NativeResourceCustody>>>,
    outcome: Option<RoventPcCollectionResourceOutcomeV3>,
    outcome_key: Option<RoventPcCollectionRecoveryKeyV3>,
    settled: Option<RoventPcCollectionSettledOutcomeV3>,
    panic: Option<Box<dyn Any + Send>>,
    stop_authorization: Option<PcSupervisorStopAuthorizationV3>,
    stop_handed_off: bool,
    joined_proof: Option<PcSupervisorJoinedProofV3>,
    game_join: PcGameJoinObservationV3,
    game_join_panic: Option<Box<dyn Any + Send>>,
    no_worker_proof: Option<PcSupervisorNoWorkerProofV3>,
    before_admission_join_proof: Option<PcSupervisorBeforeAdmissionJoinProofV3>,
    unstarted_released: bool,
    before_admission_released: bool,
    joined_offered: bool,
    state: PcRunRecoveryStateV3,
}

struct PcRunRecoverySlotV3 {
    supervisor: Arc<Mutex<PcWorkerSupervisorV3>>,
    original: Mutex<Option<OriginalPcRunV3>>,
}

fn original_runs_v3() -> &'static Mutex<Vec<Arc<PcRunRecoverySlotV3>>> {
    static REGISTRY: OnceLock<Mutex<Vec<Arc<PcRunRecoverySlotV3>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

/// A key to the same original slot, not a completion or release capability.
#[derive(Clone)]
pub(crate) struct PcRunRecoveryKeyV3(Arc<PcRunRecoverySlotV3>);

impl PcRunRecoveryKeyV3 {
    /// Prepare before any launch callback or native/resource admission.
    /// A mismatched owner is returned untouched to the caller.
    pub(crate) fn prepare(
        owners: Arc<PcCollectionOwnersV3>,
        output: Arc<ManagedOutputOwnerV3>,
        outcome_key: RoventPcCollectionRecoveryKeyV3,
    ) -> Result<
        Self,
        (
            Arc<PcCollectionOwnersV3>,
            Arc<ManagedOutputOwnerV3>,
            RoventPcCollectionRecoveryKeyV3,
        ),
    > {
        if !output.matches_inputs(owners.inputs())
            || !outcome_key.matches_collection(owners.native())
        {
            return Err((owners, output, outcome_key));
        }
        // This lock covers only the in-memory registration, not a launch or
        // recovery operation. Concurrent independent registrations serialize
        // briefly rather than rejecting one merely because another arrived.
        let Ok(mut registry) = original_runs_v3().lock() else {
            return Err((owners, output, outcome_key));
        };
        if registry
            .iter()
            .any(|slot| Arc::ptr_eq(&slot.supervisor, owners.supervisor()))
            || registry.try_reserve(1).is_err()
        {
            return Err((owners, output, outcome_key));
        }
        let slot = Arc::new(PcRunRecoverySlotV3 {
            supervisor: Arc::clone(owners.supervisor()),
            original: Mutex::new(Some(OriginalPcRunV3 {
                owners,
                output,
                resource: None,
                outcome: None,
                outcome_key: Some(outcome_key),
                settled: None,
                panic: None,
                stop_authorization: None,
                stop_handed_off: false,
                joined_proof: None,
                game_join: PcGameJoinObservationV3::Pending,
                game_join_panic: None,
                no_worker_proof: None,
                before_admission_join_proof: None,
                unstarted_released: false,
                before_admission_released: false,
                joined_offered: false,
                state: PcRunRecoveryStateV3::Prepared,
            })),
        });
        registry.push(Arc::clone(&slot));
        Ok(Self(slot))
    }

    /// Enumerate recoverable original slots without interpreting their state.
    /// A busy or poisoned registry is reported, never treated as empty.
    pub(crate) fn registered() -> Result<Vec<Self>, &'static str> {
        let registry = original_runs_v3()
            .try_lock()
            .map_err(|_| "P-C recovery registry unavailable")?;
        let mut keys = Vec::new();
        keys.try_reserve(registry.len())
            .map_err(|_| "P-C recovery key allocation failed")?;
        keys.extend(registry.iter().map(|slot| Self(Arc::clone(slot))));
        Ok(keys)
    }

    /// Find the same Arc owner, never a reconstructed ID or matching hash.
    pub(crate) fn recover_for_supervisor(
        supervisor: &Arc<Mutex<PcWorkerSupervisorV3>>,
    ) -> Result<Option<Self>, &'static str> {
        let registry = original_runs_v3()
            .try_lock()
            .map_err(|_| "P-C recovery registry unavailable")?;
        Ok(registry
            .iter()
            .find(|slot| Arc::ptr_eq(&slot.supervisor, supervisor))
            .map(|slot| Self(Arc::clone(slot))))
    }

    /// Serialize the one transition into execution for this original slot.
    pub(crate) fn begin(&self) -> Result<(), &'static str> {
        // During Prepared no outcome/panic callback can run under this lock;
        // wait only for this run's short registry observations so transient
        // contention cannot strand its already reserved original driver.
        let mut guard = self
            .0
            .original
            .lock()
            .map_err(|_| "P-C recovery slot poisoned")?;
        let original = guard.as_mut().ok_or("P-C original custody missing")?;
        if original.state != PcRunRecoveryStateV3::Prepared {
            return Err("P-C original attempt already started");
        }
        original.state = PcRunRecoveryStateV3::Executing;
        Ok(())
    }

    /// Retain exactly the Arc supplied by the original resource constructor.
    /// The caller is responsible for binding this registration to that
    /// constructor; this method does not itself prove resource identity.
    /// Rejection always returns the supplied owner intact.
    pub(crate) fn register_resource(
        &self,
        resource: Arc<Mutex<NativeResourceCustody>>,
    ) -> Result<(), Arc<Mutex<NativeResourceCustody>>> {
        let Ok(mut guard) = self.0.original.try_lock() else {
            return Err(resource);
        };
        let Some(original) = guard.as_mut() else {
            return Err(resource);
        };
        if original.state != PcRunRecoveryStateV3::Executing || original.resource.is_some() {
            return Err(resource);
        }
        original.resource = Some(resource);
        Ok(())
    }

    /// Preserve the original outcome, rejecting a different Collection owner.
    /// A missing resource registration stays visible as missing, not success.
    pub(crate) fn record_return(
        &self,
        outcome: RoventPcCollectionResourceOutcomeV3,
    ) -> Result<(), RoventPcCollectionResourceOutcomeV3> {
        let Ok(mut guard) = self.0.original.try_lock() else {
            return Err(outcome);
        };
        let Some(original) = guard.as_mut() else {
            return Err(outcome);
        };
        if original.state != PcRunRecoveryStateV3::Executing
            || !Arc::ptr_eq(outcome.original_collection(), original.owners.native())
            || !original
                .outcome_key
                .as_ref()
                .is_some_and(|key| key.same_original(&outcome.recovery_key()))
        {
            return Err(outcome);
        }
        original.outcome = Some(outcome);
        original.state = PcRunRecoveryStateV3::Returned;
        Ok(())
    }

    /// Retain the returned outcome's own recovery key without moving or
    /// duplicating the outcome. The original outcome must still be retained
    /// by its caller; dropping it transfers custody to that same key's slot.
    /// This key is neither a result nor a completion/settlement proof.
    /// Pending stop/join tokens must be retained independently by their owner.
    pub(crate) fn record_outcome_key(
        &self,
        outcome: &RoventPcCollectionResourceOutcomeV3,
    ) -> Result<(), &'static str> {
        let mut guard = self
            .0
            .original
            .try_lock()
            .map_err(|_| "P-C recovery slot unavailable")?;
        let original = guard.as_mut().ok_or("P-C original custody missing")?;
        if original.state != PcRunRecoveryStateV3::Executing
            || !Arc::ptr_eq(outcome.original_collection(), original.owners.native())
            || !original
                .outcome_key
                .as_ref()
                .is_some_and(|key| key.same_original(&outcome.recovery_key()))
        {
            return Err("P-C original outcome differs or was already recorded");
        }
        original.state = PcRunRecoveryStateV3::Returned;
        Ok(())
    }

    /// Nonblocking update for a key that was already bound before execution.
    /// The original outcome stays with the caller on every failure.
    pub(crate) fn try_record_outcome_key(
        &self,
        outcome: &RoventPcCollectionResourceOutcomeV3,
    ) -> Result<(), PcRunRecoveryRecordErrorV3> {
        let mut guard = self.0.original.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => PcRunRecoveryRecordErrorV3::Busy,
            TryLockError::Poisoned(_) => PcRunRecoveryRecordErrorV3::Poisoned,
        })?;
        let original = guard.as_mut().ok_or(PcRunRecoveryRecordErrorV3::Invalid)?;
        if !outcome.matches_original_collection(original.owners.native())
            || !original
                .outcome_key
                .as_ref()
                .is_some_and(|key| key.same_original(&outcome.recovery_key()))
        {
            return Err(PcRunRecoveryRecordErrorV3::Invalid);
        }
        match original.state {
            PcRunRecoveryStateV3::Executing => {
                original.state = PcRunRecoveryStateV3::Returned;
                Ok(())
            }
            // Re-observing the same original outcome must not overwrite a
            // retained panic or reject a concurrent prior observation.
            PcRunRecoveryStateV3::Returned | PcRunRecoveryStateV3::Panicked => Ok(()),
            PcRunRecoveryStateV3::Prepared => Err(PcRunRecoveryRecordErrorV3::Invalid),
        }
    }

    /// Obtain the exact key issued by the original outcome. Recovering through
    /// it returns None while that outcome is still owned by its caller.
    pub(crate) fn original_outcome_key(
        &self,
    ) -> Result<Option<RoventPcCollectionRecoveryKeyV3>, &'static str> {
        let guard = self
            .0
            .original
            .try_lock()
            .map_err(|_| "P-C recovery slot unavailable")?;
        let original = guard.as_ref().ok_or("P-C original custody missing")?;
        Ok(original.outcome_key.clone())
    }

    /// Produce the exact bridge authorization directly into this slot. The
    /// bridge's native fence and original supervisor checks remain mandatory.
    /// Its current issuance path has no irreversible token-consumption step;
    /// a panic before return must not be treated as an issued authorization.
    pub(crate) fn try_prepare_stop(
        &self,
        bridge: &Arc<RoventPcCollectionBridgeV3>,
        fence: &rz_uci::rovent_bootstrap::launch::RoventPcCollectionNativeFenceV3,
    ) -> Result<PcStopStepV3, &'static str> {
        let mut guard = match self.0.original.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(PcStopStepV3::Pending),
            Err(TryLockError::Poisoned(_)) => return Err("P-C recovery slot poisoned"),
        };
        let original = guard.as_mut().ok_or("P-C original custody missing")?;
        if !fence.matches_original_owner(original.owners.native()) {
            return Err("P-C stop fence belongs to another owner");
        }
        if original.stop_handed_off {
            return Ok(PcStopStepV3::Ready);
        }
        if original.stop_authorization.is_some() {
            return Ok(PcStopStepV3::Ready);
        }
        let issued = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bridge.try_authorize_supervisor_stop(fence, original.owners.supervisor())
        }));
        match issued {
            Ok(Ok(Some(token))) => {
                // Bridge checks the original fence, hook and same supervisor
                // before issuing this non-cloneable token.
                original.stop_authorization = Some(token);
                Ok(PcStopStepV3::Ready)
            }
            Ok(Ok(None)) => Ok(PcStopStepV3::Pending),
            Ok(Err(error)) => Err(error),
            Err(payload) => {
                drop(guard);
                std::panic::resume_unwind(payload)
            }
        }
    }

    /// Move the slot's original authorization directly into the same driver
    /// custody. The worker primitive checks identity and performs the final
    /// in-memory move without further fallible or user-provided callbacks.
    /// This is not a joined, native/GPU, teacher, or output completion proof.
    pub(crate) fn try_handoff_stop(&self) -> Result<PcStopStepV3, &'static str> {
        let mut guard = match self.0.original.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(PcStopStepV3::Pending),
            Err(TryLockError::Poisoned(_)) => return Err("P-C recovery slot poisoned"),
        };
        let original = guard.as_mut().ok_or("P-C original custody missing")?;
        if original.stop_handed_off {
            return Ok(PcStopStepV3::Ready);
        }
        if original.stop_authorization.is_none() {
            return Ok(PcStopStepV3::Pending);
        }
        let mut driver = match original.owners.driver_custody().try_lock() {
            Ok(driver) => driver,
            Err(TryLockError::WouldBlock) => return Ok(PcStopStepV3::Pending),
            Err(TryLockError::Poisoned(_)) => return Err("P-C supervisor custody poisoned"),
        };
        let moved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            driver.authorize_from_slot(&mut original.stop_authorization)
        }));
        drop(driver);
        match moved {
            Ok(Ok(())) => {
                original.stop_handed_off = true;
                Ok(PcStopStepV3::Ready)
            }
            Ok(Err(error)) => Err(error),
            Err(payload) => {
                drop(guard);
                std::panic::resume_unwind(payload)
            }
        }
    }

    /// Retire only the same proven unstarted or pre-admission driver. Native,
    /// physical device, teacher, output and resource custody remain separate.
    /// The proofs stay in this registered slot after the driver retirement.
    pub(crate) fn poll_pre_admission_driver(
        &self,
    ) -> Result<PcDriverPollV3<bool>, PcDriverEvidenceErrorV3> {
        let mut guard = match self.0.original.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(PcDriverPollV3::Pending),
            Err(TryLockError::Poisoned(_)) => {
                return Err(PcDriverEvidenceErrorV3::Poisoned("recovery"));
            }
        };
        let original = guard
            .as_mut()
            .ok_or(PcDriverEvidenceErrorV3::Invalid("original custody missing"))?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if original.unstarted_released || original.before_admission_released {
                return Ok(PcDriverPollV3::Ready(true));
            }
            // A joined driver is owned by the outcome's settlement path;
            // its original registry entry may already have been retired.
            if original.joined_offered {
                return Ok(PcDriverPollV3::Ready(false));
            }
            if original.no_worker_proof.is_none() && original.before_admission_join_proof.is_none()
            {
                match SupervisorStartCustodyV3::try_seal_no_worker(
                    original.owners.driver_custody(),
                    original.owners.native(),
                )? {
                    PcDriverPollV3::Pending => return Ok(PcDriverPollV3::Pending),
                    PcDriverPollV3::Ready(proof) => original.no_worker_proof = proof,
                }
            }
            if let Some(proof) = original.no_worker_proof.as_ref() {
                if !proof.matches_original(
                    original.owners.driver_custody(),
                    original.owners.supervisor(),
                    original.owners.native(),
                ) {
                    return Err(PcDriverEvidenceErrorV3::Invalid(
                        "original no-worker proof differs",
                    ));
                }
                return match SupervisorStartCustodyV3::try_release_no_worker(
                    original.owners.driver_custody(),
                )? {
                    PcDriverPollV3::Pending => Ok(PcDriverPollV3::Pending),
                    PcDriverPollV3::Ready(released) => {
                        original.unstarted_released = released;
                        Ok(PcDriverPollV3::Ready(released))
                    }
                };
            }
            if original.before_admission_join_proof.is_none() {
                {
                    let mut driver = match original.owners.driver_custody().try_lock() {
                        Ok(driver) => driver,
                        Err(TryLockError::WouldBlock) => return Ok(PcDriverPollV3::Pending),
                        Err(TryLockError::Poisoned(_)) => {
                            return Err(PcDriverEvidenceErrorV3::Poisoned("driver"));
                        }
                    };
                    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        driver.try_join()
                    }));
                    drop(driver);
                    if let Err(payload) = attempt {
                        std::panic::resume_unwind(payload);
                    }
                }
                match SupervisorStartCustodyV3::try_before_admission_join_proof(
                    original.owners.driver_custody(),
                )? {
                    PcDriverPollV3::Pending => return Ok(PcDriverPollV3::Pending),
                    PcDriverPollV3::Ready(proof) => original.before_admission_join_proof = proof,
                }
            }
            let Some(proof) = original.before_admission_join_proof.as_ref() else {
                return Ok(PcDriverPollV3::Ready(false));
            };
            if !proof.matches_original(
                original.owners.driver_custody(),
                original.owners.supervisor(),
                original.owners.native(),
            ) {
                return Err(PcDriverEvidenceErrorV3::Invalid(
                    "original before-admission proof differs",
                ));
            }
            match SupervisorStartCustodyV3::try_release_before_admission_join(
                original.owners.driver_custody(),
                proof,
            )? {
                PcDriverPollV3::Pending => Ok(PcDriverPollV3::Pending),
                PcDriverPollV3::Ready(released) => {
                    original.before_admission_released = released;
                    Ok(PcDriverPollV3::Ready(released))
                }
            }
        }));
        drop(guard);
        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Directly retain joined evidence in the same original outcome. The
    /// proof remains in its slot if the outcome rejects it or is busy.
    pub(crate) fn poll_joined_driver(
        &self,
        outcome: &mut RoventPcCollectionResourceOutcomeV3,
    ) -> Result<PcDriverPollV3<bool>, PcDriverEvidenceErrorV3> {
        let mut guard = match self.0.original.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(PcDriverPollV3::Pending),
            Err(TryLockError::Poisoned(_)) => {
                return Err(PcDriverEvidenceErrorV3::Poisoned("recovery"));
            }
        };
        let original = guard
            .as_mut()
            .ok_or(PcDriverEvidenceErrorV3::Invalid("original custody missing"))?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if !outcome.matches_original_collection(original.owners.native())
                || !original
                    .outcome_key
                    .as_ref()
                    .is_some_and(|key| key.same_original(&outcome.recovery_key()))
            {
                return Err(PcDriverEvidenceErrorV3::Invalid("original outcome differs"));
            }
            if original.joined_offered {
                return Ok(PcDriverPollV3::Ready(true));
            }
            if original.joined_proof.is_none() {
                {
                    let mut driver = match original.owners.driver_custody().try_lock() {
                        Ok(driver) => driver,
                        Err(TryLockError::WouldBlock) => return Ok(PcDriverPollV3::Pending),
                        Err(TryLockError::Poisoned(_)) => {
                            return Err(PcDriverEvidenceErrorV3::Poisoned("driver"));
                        }
                    };
                    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        driver.try_join()
                    }));
                    drop(driver);
                    if let Err(payload) = attempt {
                        std::panic::resume_unwind(payload);
                    }
                }
                match SupervisorStartCustodyV3::try_joined_proof(original.owners.driver_custody())?
                {
                    PcDriverPollV3::Pending => return Ok(PcDriverPollV3::Pending),
                    PcDriverPollV3::Ready(proof) => original.joined_proof = proof,
                }
            }
            if original.joined_proof.is_none() {
                return Ok(PcDriverPollV3::Ready(false));
            }
            let moved = outcome
                .offer_joined_driver_from_slot(&original.owners, &mut original.joined_proof)
                .map_err(PcDriverEvidenceErrorV3::Invalid)?;
            if moved {
                original.joined_offered = true;
            }
            Ok(PcDriverPollV3::Ready(moved))
        }));
        drop(guard);
        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Observe game join independently, borrowing the exact proof from its
    /// original slot, live outcome, or same settled custody. A pending game
    /// seal never holds back driver/native physical settlement.
    pub(crate) fn poll_game_join(
        &self,
        outcome: Option<&RoventPcCollectionResourceOutcomeV3>,
        bridge: &Arc<RoventPcCollectionBridgeV3>,
    ) -> Result<PcGameJoinObservationV3, PcDriverEvidenceErrorV3> {
        let mut guard = match self.0.original.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(PcGameJoinObservationV3::Pending),
            Err(TryLockError::Poisoned(_)) => {
                return Err(PcDriverEvidenceErrorV3::Poisoned("recovery"));
            }
        };
        let original = guard
            .as_mut()
            .ok_or(PcDriverEvidenceErrorV3::Invalid("original custody missing"))?;
        if let Some(outcome) = outcome {
            if !original
                .outcome_key
                .as_ref()
                .is_some_and(|key| key.same_original(&outcome.recovery_key()))
            {
                return Err(PcDriverEvidenceErrorV3::Invalid("original outcome differs"));
            }
        } else if !original
            .settled
            .as_ref()
            .is_some_and(|settled| Arc::ptr_eq(&settled.collection, original.owners.native()))
        {
            // Without a live outcome, only this slot's same-owner settled
            // custody can supply an original joined proof.
            return Ok(PcGameJoinObservationV3::Pending);
        }
        if original.game_join != PcGameJoinObservationV3::Pending {
            return Ok(original.game_join);
        }
        let proof = original
            .joined_proof
            .as_ref()
            .or_else(|| outcome.and_then(|original| original.joined_driver()))
            .or_else(|| {
                original
                    .settled
                    .as_ref()
                    .and_then(|settled| settled.driver_joined.as_ref())
            });
        let Some(proof) = proof else {
            return Ok(PcGameJoinObservationV3::Pending);
        };
        if !proof.matches_original(
            original.owners.driver_custody(),
            original.owners.supervisor(),
            original.owners.native(),
        ) {
            return Err(PcDriverEvidenceErrorV3::Invalid(
                "original joined proof differs",
            ));
        }
        let observed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bridge.try_bind_game_joined(proof, original.owners.driver_custody())
        }));
        original.game_join = match observed {
            Ok(Ok(None)) => PcGameJoinObservationV3::Pending,
            Ok(Ok(Some(true))) => PcGameJoinObservationV3::Bound,
            Ok(Ok(Some(false))) => PcGameJoinObservationV3::Refused,
            Ok(Err(error)) => PcGameJoinObservationV3::Failed(error),
            Err(payload) => {
                original.game_join_panic = Some(payload);
                PcGameJoinObservationV3::Panicked
            }
        };
        Ok(original.game_join)
    }

    /// Observe only this original run's game-join result, never GPU success.
    pub(crate) fn game_join_observation(&self) -> Result<PcGameJoinObservationV3, &'static str> {
        let guard = self
            .0
            .original
            .try_lock()
            .map_err(|_| "P-C recovery slot unavailable")?;
        let original = guard.as_ref().ok_or("P-C original custody missing")?;
        Ok(original.game_join)
    }

    /// Inspect the original game-join panic without moving its payload.
    /// The observer must be finite and must not re-enter this same slot.
    pub(crate) fn with_game_join_panic<T>(
        &self,
        observe: impl FnOnce(&(dyn Any + Send)) -> T,
    ) -> Result<Option<T>, &'static str> {
        let guard = self
            .0
            .original
            .try_lock()
            .map_err(|_| "P-C recovery slot unavailable")?;
        let original = guard.as_ref().ok_or("P-C original custody missing")?;
        let Some(payload) = original.game_join_panic.as_deref() else {
            return Ok(None);
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| observe(payload)));
        drop(guard);
        match result {
            Ok(value) => Ok(Some(value)),
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Settle only the same original outcome and retain every returned
    /// observer/hook/driver owner in this slot before returning to the caller.
    /// This is not output closure, an independent GPU fence, or registry release.
    pub(crate) fn try_settle_outcome(
        &self,
        outcome: &mut RoventPcCollectionResourceOutcomeV3,
    ) -> Result<PcDriverPollV3<bool>, &'static str> {
        let mut guard = match self.0.original.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(PcDriverPollV3::Pending),
            Err(TryLockError::Poisoned(_)) => return Err("P-C recovery slot poisoned"),
        };
        let original = guard.as_mut().ok_or("P-C original custody missing")?;
        if !original
            .outcome_key
            .as_ref()
            .is_some_and(|key| key.same_original(&outcome.recovery_key()))
        {
            return Err("P-C original outcome key differs");
        }
        if let Some(settled) = original.settled.as_ref() {
            return if Arc::ptr_eq(&settled.collection, original.owners.native()) {
                Ok(PcDriverPollV3::Ready(true))
            } else {
                Err("P-C settled original owner differs")
            };
        }
        if !outcome.matches_original_collection(original.owners.native()) {
            return Err("P-C original outcome owner differs");
        }
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| outcome.try_settle()));
        match result {
            Ok(Ok(Some(settled))) => {
                original.settled = Some(settled);
                Ok(PcDriverPollV3::Ready(true))
            }
            Ok(Ok(None)) => Ok(PcDriverPollV3::Ready(false)),
            Ok(Err(error)) => Err(error),
            Err(payload) => {
                drop(guard);
                std::panic::resume_unwind(payload)
            }
        }
    }

    /// Borrow retained settled custody for a finite, non-reentrant read.
    /// No ownership, physical-completion, or release authority is created.
    pub(crate) fn with_settled<T>(
        &self,
        observe: impl FnOnce(&RoventPcCollectionSettledOutcomeV3) -> T,
    ) -> Result<Option<T>, &'static str> {
        let guard = self
            .0
            .original
            .try_lock()
            .map_err(|_| "P-C recovery slot unavailable")?;
        let original = guard.as_ref().ok_or("P-C original custody missing")?;
        let Some(settled) = original.settled.as_ref() else {
            return Ok(None);
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| observe(settled)));
        drop(guard);
        match result {
            Ok(value) => Ok(Some(value)),
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// The caller catches the launch panic and transfers the original payload.
    /// A panic does not imply that no worker, native work, or GPU work started.
    pub(crate) fn record_panic(
        &self,
        payload: Box<dyn Any + Send>,
    ) -> Result<(), Box<dyn Any + Send>> {
        let Ok(mut guard) = self.0.original.try_lock() else {
            return Err(payload);
        };
        let Some(original) = guard.as_mut() else {
            return Err(payload);
        };
        if original.state != PcRunRecoveryStateV3::Executing {
            return Err(payload);
        }
        original.panic = Some(payload);
        original.state = PcRunRecoveryStateV3::Panicked;
        Ok(())
    }

    /// Read only a custody observation; resource registration is not a fence.
    pub(crate) fn observation(&self) -> Result<(PcRunRecoveryStateV3, bool, bool), &'static str> {
        let guard = self
            .0
            .original
            .try_lock()
            .map_err(|_| "P-C recovery slot unavailable")?;
        let original = guard.as_ref().ok_or("P-C original custody missing")?;
        Ok((
            original.state,
            original.resource.is_some(),
            original.panic.is_some(),
        ))
    }

    /// Clones preserve the original owners, and are not newly issued proofs.
    pub(crate) fn original_owners(
        &self,
    ) -> Result<(Arc<PcCollectionOwnersV3>, Arc<ManagedOutputOwnerV3>), &'static str> {
        let guard = self
            .0
            .original
            .try_lock()
            .map_err(|_| "P-C recovery slot unavailable")?;
        let original = guard.as_ref().ok_or("P-C original custody missing")?;
        Ok((Arc::clone(&original.owners), Arc::clone(&original.output)))
    }

    /// This is the retained Arc, if the actual constructor registered it.
    /// It is not proof that its resource scope is safe to finish.
    pub(crate) fn original_resource(
        &self,
    ) -> Result<Option<Arc<Mutex<NativeResourceCustody>>>, &'static str> {
        let guard = self
            .0
            .original
            .try_lock()
            .map_err(|_| "P-C recovery slot unavailable")?;
        let original = guard.as_ref().ok_or("P-C original custody missing")?;
        Ok(original.resource.as_ref().map(Arc::clone))
    }

    /// Borrow the same retained outcome for a finite, non-reentrant recovery
    /// step. Unwinding keeps the outcome in the slot and does not poison it.
    pub(crate) fn with_outcome<T>(
        &self,
        action: impl FnOnce(&mut RoventPcCollectionResourceOutcomeV3) -> T,
    ) -> Result<Option<T>, &'static str> {
        let mut guard = self
            .0
            .original
            .try_lock()
            .map_err(|_| "P-C recovery slot unavailable")?;
        let original = guard.as_mut().ok_or("P-C original custody missing")?;
        let Some(outcome) = original.outcome.as_mut() else {
            return Ok(None);
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| action(outcome)));
        drop(guard);
        match result {
            Ok(value) => Ok(Some(value)),
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Inspect the retained original payload without transferring ownership.
    /// The callback must be finite and must not re-enter this slot.
    pub(crate) fn with_panic<T>(
        &self,
        action: impl FnOnce(&(dyn Any + Send)) -> T,
    ) -> Result<Option<T>, &'static str> {
        let guard = self
            .0
            .original
            .try_lock()
            .map_err(|_| "P-C recovery slot unavailable")?;
        let original = guard.as_ref().ok_or("P-C original custody missing")?;
        let Some(payload) = original.panic.as_deref() else {
            return Ok(None);
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| action(payload)));
        drop(guard);
        match result {
            Ok(value) => Ok(Some(value)),
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
}
