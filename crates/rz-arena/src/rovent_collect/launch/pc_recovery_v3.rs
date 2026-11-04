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
use super::pc_lock_v3::{ManagedOutputCloseEvidenceV3, ManagedOutputOwnerV3, PcCollectionOwnersV3};
use crate::rovent_collect::game_authority_v3::{GamePublishGuardV3, GameStepV3};
use crate::rovent_collect::pc_hooks_lock_v3::{PcGameClosureProofV3, PcV3LaunchClosure};
use crate::rovent_collect::pc_hooks_lock_v3::{
    PcSupervisorStopAuthorizationV3, RoventPcCollectionBridgeV3,
};
use crate::rovent_collect::pc_worker_v3::{
    PcDriverEvidenceErrorV3, PcDriverPollV3, PcSupervisorBeforeAdmissionJoinProofV3,
    PcSupervisorJoinedProofV3, PcSupervisorNoWorkerProofV3, PcWorkerSupervisorV3,
    SupervisorStartCustodyV3,
};
use crate::rovent_collect::policy_row_pipeline_v3::PolicyFinalizationV3;
use crate::rovent_resource_scope::launch::{
    NativeResourceCustody, RoventPcCollectionRecoveryKeyV3, RoventPcCollectionResourceOutcomeV3,
    RoventPcCollectionSettledOutcomeV3,
};
use std::any::Any;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock, TryLockError};
use std::time::Instant;

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

/// Publication observation only: independent of GPU and parent lease closure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PcOutputClosureV3 {
    Pending,
    Progress,
    Published,
    NoData,
    Aborted,
    Failed,
    Unknown,
    Timeout,
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
    output_game_proof: Option<PcGameClosureProofV3>,
    output_game_closed: bool,
    output_guard: Option<GamePublishGuardV3<PcWorkerSupervisorV3>>,
    output_error: Option<super::ArenaError>,
    output_panic: Option<Box<dyn Any + Send>>,
    output_terminal: Option<PcOutputClosureV3>,
    output_close: Option<ManagedOutputCloseEvidenceV3>,
    no_worker_proof: Option<PcSupervisorNoWorkerProofV3>,
    before_admission_join_proof: Option<PcSupervisorBeforeAdmissionJoinProofV3>,
    unstarted_released: bool,
    before_admission_released: bool,
    joined_offered: bool,
    state: PcRunRecoveryStateV3,
}

/// Close admission only after the caller has established producer shutdown.
/// Preserve the original terminal separately from uncertain close evidence.
fn observe_closed_output_v3(
    original: &mut OriginalPcRunV3,
    terminal: PcOutputClosureV3,
) -> PcOutputClosureV3 {
    if original.output_close.is_none() {
        original.output_close = original.output.try_close();
    }
    let Some(evidence) = original.output_close.as_ref() else {
        return PcOutputClosureV3::Pending;
    };
    if !evidence.matches_original(&original.output) {
        return PcOutputClosureV3::Unknown;
    }
    use super::pc_lock_v3::ManagedOutputCloseStatusV3 as Close;
    match (terminal, evidence.status()) {
        (_, Close::Unknown | Close::UnknownAfterAttempt) => PcOutputClosureV3::Unknown,
        (PcOutputClosureV3::Published, Close::WritesAcknowledged { count }) if count > 0 => {
            terminal
        }
        (PcOutputClosureV3::NoData, Close::NoWrites) => terminal,
        (PcOutputClosureV3::Published | PcOutputClosureV3::NoData, _) => PcOutputClosureV3::Unknown,
        _ => terminal,
    }
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

/// A point-in-time view of this same registered owner, not completion proof.
pub(crate) struct PcRecoverySnapshotV3 {
    pub(crate) recovery: PcRunRecoveryKeyV3,
    pub(crate) owners: Arc<PcCollectionOwnersV3>,
    pub(crate) output: Arc<ManagedOutputOwnerV3>,
    pub(crate) outcome_key: RoventPcCollectionRecoveryKeyV3,
    pub(crate) state: PcRunRecoveryStateV3,
}

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
                output_game_proof: None,
                output_game_closed: false,
                output_guard: None,
                output_error: None,
                output_panic: None,
                output_terminal: None,
                output_close: None,
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

    /// Observe the exact original supervisor and its per-run custody without
    /// conflating contention with missing registration or poisoned state.
    /// No global lock is held while the original slot is accessed.
    pub(crate) fn try_recovery_snapshot_for_supervisor(
        supervisor: &Arc<Mutex<PcWorkerSupervisorV3>>,
    ) -> Result<PcDriverPollV3<Option<PcRecoverySnapshotV3>>, PcDriverEvidenceErrorV3> {
        let slot = {
            let registry = match original_runs_v3().try_lock() {
                Ok(registry) => registry,
                Err(TryLockError::WouldBlock) => return Ok(PcDriverPollV3::Pending),
                Err(TryLockError::Poisoned(_)) => {
                    return Err(PcDriverEvidenceErrorV3::Poisoned("recovery registry"));
                }
            };
            registry
                .iter()
                .find(|entry| Arc::ptr_eq(&entry.supervisor, supervisor))
                .map(Arc::clone)
        };
        let Some(slot) = slot else {
            return Ok(PcDriverPollV3::Ready(None));
        };
        let guard = match slot.original.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(PcDriverPollV3::Pending),
            Err(TryLockError::Poisoned(_)) => {
                return Err(PcDriverEvidenceErrorV3::Poisoned("recovery slot"));
            }
        };
        let original = guard
            .as_ref()
            .ok_or(PcDriverEvidenceErrorV3::Invalid("original custody missing"))?;
        let outcome_key = original
            .outcome_key
            .as_ref()
            .ok_or(PcDriverEvidenceErrorV3::Invalid(
                "original outcome key missing",
            ))?;
        if !Arc::ptr_eq(original.owners.supervisor(), supervisor)
            || !original.output.matches_inputs(original.owners.inputs())
            || !outcome_key.matches_collection(original.owners.native())
        {
            return Err(PcDriverEvidenceErrorV3::Invalid(
                "original recovery identities differ",
            ));
        }
        let owners = Arc::clone(&original.owners);
        let output = Arc::clone(&original.output);
        let outcome_key = outcome_key.clone();
        let state = original.state;
        drop(guard);
        Ok(PcDriverPollV3::Ready(Some(PcRecoverySnapshotV3 {
            recovery: Self(slot),
            owners,
            output,
            outcome_key,
            state,
        })))
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

    /// Advance the exact settled game's output without asserting parent lease
    /// or device completion. The cancellation source is supplied by the caller;
    /// this method does not create or reset one. Finalization may perform
    /// synchronous file I/O; the deadline cannot preempt a blocked syscall.
    /// Call under process supervision, not from a native manager tick.
    /// Original proof, guard, errors,
    /// panic and close evidence remain in the same registered recovery slot.
    pub(crate) fn poll_output_closure(
        &self,
        until: Instant,
        cancel: &AtomicBool,
    ) -> Result<PcOutputClosureV3, PcDriverEvidenceErrorV3> {
        let mut guard = match self.0.original.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(PcOutputClosureV3::Pending),
            Err(TryLockError::Poisoned(_)) => {
                return Err(PcDriverEvidenceErrorV3::Poisoned("recovery"));
            }
        };
        let original = guard
            .as_mut()
            .ok_or(PcDriverEvidenceErrorV3::Invalid("original custody missing"))?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if let Some(terminal) = original.output_terminal {
                return observe_closed_output_v3(original, terminal);
            }
            if original.output_panic.is_some() {
                return PcOutputClosureV3::Unknown;
            }
            if Instant::now() >= until {
                return PcOutputClosureV3::Pending;
            }
            let Some(settled) = original.settled.as_ref() else {
                return PcOutputClosureV3::Pending;
            };
            if !Arc::ptr_eq(&settled.collection, original.owners.native()) {
                return PcOutputClosureV3::Unknown;
            }
            if settled.native.is_err() {
                original.output_terminal = Some(PcOutputClosureV3::Failed);
                return PcOutputClosureV3::Progress;
            }
            let Some(joined) = settled.driver_joined.as_ref() else {
                return PcOutputClosureV3::Pending;
            };
            let Some(closure) = settled
                .hook_custody
                .downcast_ref::<Arc<PcV3LaunchClosure>>()
            else {
                original.output_terminal = Some(PcOutputClosureV3::Unknown);
                return PcOutputClosureV3::Progress;
            };
            if original.game_join == PcGameJoinObservationV3::Pending {
                original.game_join = match original
                    .owners
                    .bridge()
                    .try_bind_game_joined(joined, original.owners.driver_custody())
                {
                    Ok(None) => return PcOutputClosureV3::Pending,
                    Ok(Some(true)) => PcGameJoinObservationV3::Bound,
                    Ok(Some(false)) => PcGameJoinObservationV3::Refused,
                    Err(error) => PcGameJoinObservationV3::Failed(error),
                };
            }
            if original.game_join != PcGameJoinObservationV3::Bound {
                original.output_terminal = Some(PcOutputClosureV3::Failed);
                return PcOutputClosureV3::Progress;
            }
            if original.output_game_proof.is_none() {
                match closure.try_game_proof(joined, original.owners.driver_custody()) {
                    Ok(None) => return PcOutputClosureV3::Pending,
                    Ok(Some(proof)) => original.output_game_proof = Some(proof),
                    Err(error) => {
                        original.output_error = Some(error);
                        original.output_terminal = Some(PcOutputClosureV3::Failed);
                        return PcOutputClosureV3::Progress;
                    }
                }
            }
            let proof = original.output_game_proof.as_ref().expect("proof retained");
            if !original.output_game_closed {
                match proof.game().try_close_pc(proof, &mut original.output_error) {
                    GameStepV3::Ready | GameStepV3::Existing => original.output_game_closed = true,
                    GameStepV3::Busy => return PcOutputClosureV3::Pending,
                    GameStepV3::Aborted => {
                        original.output_terminal = Some(PcOutputClosureV3::Aborted);
                        return PcOutputClosureV3::Progress;
                    }
                    GameStepV3::Unknown => {
                        original.output_terminal = Some(PcOutputClosureV3::Unknown);
                        return PcOutputClosureV3::Progress;
                    }
                    _ => {
                        original.output_terminal = Some(PcOutputClosureV3::Failed);
                        return PcOutputClosureV3::Progress;
                    }
                }
            }
            if original.output_guard.is_none() {
                match proof.game().try_begin_publication_pc(
                    proof,
                    &mut original.output_guard,
                    &mut original.output_error,
                ) {
                    GameStepV3::Ready => {}
                    GameStepV3::Busy => return PcOutputClosureV3::Pending,
                    GameStepV3::Aborted => {
                        original.output_terminal = Some(PcOutputClosureV3::Aborted);
                        return PcOutputClosureV3::Progress;
                    }
                    GameStepV3::Unknown => {
                        original.output_terminal = Some(PcOutputClosureV3::Unknown);
                        return PcOutputClosureV3::Progress;
                    }
                    _ => {
                        original.output_terminal = Some(PcOutputClosureV3::Failed);
                        return PcOutputClosureV3::Progress;
                    }
                }
            }
            let publication_guard = original.output_guard.as_mut().expect("guard retained");
            let step = match proof
                .pipeline()
                .try_finalize(publication_guard, until, cancel)
            {
                PolicyFinalizationV3::Pending => PcOutputClosureV3::Pending,
                PolicyFinalizationV3::Progress => PcOutputClosureV3::Progress,
                PolicyFinalizationV3::Published => PcOutputClosureV3::Published,
                PolicyFinalizationV3::NoData => PcOutputClosureV3::NoData,
                PolicyFinalizationV3::Aborted => PcOutputClosureV3::Aborted,
                PolicyFinalizationV3::Failed => PcOutputClosureV3::Failed,
                PolicyFinalizationV3::Unknown => PcOutputClosureV3::Unknown,
            };
            if !matches!(
                step,
                PcOutputClosureV3::Pending | PcOutputClosureV3::Progress
            ) {
                original.output_terminal = Some(step);
                return PcOutputClosureV3::Progress;
            }
            step
        }));
        match result {
            Ok(step) => Ok(step),
            Err(payload) => {
                original.output_panic = Some(payload);
                original.output_terminal = Some(PcOutputClosureV3::Unknown);
                Ok(PcOutputClosureV3::Progress)
            }
        }
    }

    /// Seal a nonterminal publication timeout without starting any additional
    /// publication. The original guard, pipeline, partial data and errors stay
    /// in this slot. Same-owner settlement and actual driver join are required
    /// before closing output admission; GPU and parent lease remain independent.
    pub(crate) fn poll_output_timeout(&self) -> Result<PcOutputClosureV3, PcDriverEvidenceErrorV3> {
        let mut guard = match self.0.original.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(PcOutputClosureV3::Pending),
            Err(TryLockError::Poisoned(_)) => {
                return Err(PcDriverEvidenceErrorV3::Poisoned("recovery"));
            }
        };
        let original = guard
            .as_mut()
            .ok_or(PcDriverEvidenceErrorV3::Invalid("original custody missing"))?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if let Some(terminal) = original.output_terminal {
                return observe_closed_output_v3(original, terminal);
            }
            if original.output_panic.is_some() {
                return PcOutputClosureV3::Unknown;
            }
            let Some(settled) = original.settled.as_ref() else {
                return PcOutputClosureV3::Pending;
            };
            if !Arc::ptr_eq(&settled.collection, original.owners.native()) {
                return PcOutputClosureV3::Unknown;
            }
            let Some(joined) = settled.driver_joined.as_ref() else {
                return PcOutputClosureV3::Pending;
            };
            if !joined.matches_original(
                original.owners.driver_custody(),
                original.owners.supervisor(),
                original.owners.native(),
            ) || settled
                .hook_custody
                .downcast_ref::<Arc<PcV3LaunchClosure>>()
                .is_none()
            {
                return PcOutputClosureV3::Unknown;
            }
            original.output_terminal = Some(PcOutputClosureV3::Timeout);
            observe_closed_output_v3(original, PcOutputClosureV3::Timeout)
        }));
        match result {
            Ok(step) => Ok(step),
            Err(payload) => {
                original.output_panic = Some(payload);
                Ok(PcOutputClosureV3::Unknown)
            }
        }
    }

    /// Observe the retained finalization result separately from close evidence.
    pub(crate) fn output_terminal_observation(
        &self,
    ) -> Result<PcDriverPollV3<Option<PcOutputClosureV3>>, PcDriverEvidenceErrorV3> {
        let guard = match self.0.original.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(PcDriverPollV3::Pending),
            Err(TryLockError::Poisoned(_)) => {
                return Err(PcDriverEvidenceErrorV3::Poisoned("recovery"));
            }
        };
        let original = guard
            .as_ref()
            .ok_or(PcDriverEvidenceErrorV3::Invalid("original custody missing"))?;
        Ok(PcDriverPollV3::Ready(original.output_terminal))
    }

    /// Observe the same output-close evidence without moving its original owner.
    pub(crate) fn output_close_status(
        &self,
    ) -> Result<
        PcDriverPollV3<Option<super::pc_lock_v3::ManagedOutputCloseStatusV3>>,
        PcDriverEvidenceErrorV3,
    > {
        let guard = match self.0.original.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Ok(PcDriverPollV3::Pending),
            Err(TryLockError::Poisoned(_)) => {
                return Err(PcDriverEvidenceErrorV3::Poisoned("recovery"));
            }
        };
        let original = guard
            .as_ref()
            .ok_or(PcDriverEvidenceErrorV3::Invalid("original custody missing"))?;
        Ok(PcDriverPollV3::Ready(
            original
                .output_close
                .as_ref()
                .map(|evidence| evidence.status()),
        ))
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
