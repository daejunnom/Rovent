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
    PcSupervisorBeforeAdmissionJoinProofV3, PcSupervisorJoinedProofV3, PcSupervisorNoWorkerProofV3,
    PcWorkerSupervisorV3,
};
use crate::rovent_resource_scope::launch::{
    NativeResourceCustody, RoventPcCollectionRecoveryKeyV3, RoventPcCollectionResourceOutcomeV3,
};
use std::any::Any;
use std::sync::{Arc, Mutex, OnceLock, TryLockError};

/// A custody operation does not consume an incoming original on failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PcEvidenceCustodyErrorV3 {
    Busy,
    Poisoned,
    Missing,
    AlreadyOccupied,
    DifferentOwner,
}

/// A busy original owner is pending, never a completed stop or join.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PcStopStepV3 {
    Pending,
    Ready,
}

trait SamePcRunV3 {
    fn matches_run(&self, owners: &PcCollectionOwnersV3) -> bool;
}

impl SamePcRunV3 for PcSupervisorStopAuthorizationV3 {
    fn matches_run(&self, owners: &PcCollectionOwnersV3) -> bool {
        self.matches_original(owners.supervisor(), owners.native())
    }
}

macro_rules! same_driver_evidence {
    ($($evidence:ty),+ $(,)?) => {$(
        impl SamePcRunV3 for $evidence {
            fn matches_run(&self, owners: &PcCollectionOwnersV3) -> bool {
                self.matches_original(owners.driver_custody(), owners.supervisor(), owners.native())
            }
        }
    )+};
}

same_driver_evidence!(
    PcSupervisorJoinedProofV3,
    PcSupervisorNoWorkerProofV3,
    PcSupervisorBeforeAdmissionJoinProofV3,
);

macro_rules! evidence_custody {
    ($put:ident, $take:ident, $field:ident, $evidence:ty) => {
        /// Retain this exact token; rejection returns it unchanged.
        pub(crate) fn $put(
            &self,
            evidence: $evidence,
        ) -> Result<(), ($evidence, PcEvidenceCustodyErrorV3)> {
            let mut guard = match self.0.original.try_lock() {
                Ok(guard) => guard,
                Err(TryLockError::WouldBlock) => {
                    return Err((evidence, PcEvidenceCustodyErrorV3::Busy))
                }
                Err(TryLockError::Poisoned(_)) => {
                    return Err((evidence, PcEvidenceCustodyErrorV3::Poisoned))
                }
            };
            let Some(original) = guard.as_mut() else {
                return Err((evidence, PcEvidenceCustodyErrorV3::Missing));
            };
            if !evidence.matches_run(&original.owners) {
                return Err((evidence, PcEvidenceCustodyErrorV3::DifferentOwner));
            }
            if original.$field.is_some() {
                return Err((evidence, PcEvidenceCustodyErrorV3::AlreadyOccupied));
            }
            original.$field = Some(evidence);
            Ok(())
        }

        /// Transfer the same token to the caller; this does not consume its
        /// meaning or authorize settlement or resource release.
        pub(crate) fn $take(&self) -> Result<Option<$evidence>, PcEvidenceCustodyErrorV3> {
            let mut guard = match self.0.original.try_lock() {
                Ok(guard) => guard,
                Err(TryLockError::WouldBlock) => return Err(PcEvidenceCustodyErrorV3::Busy),
                Err(TryLockError::Poisoned(_)) => return Err(PcEvidenceCustodyErrorV3::Poisoned),
            };
            let original = guard.as_mut().ok_or(PcEvidenceCustodyErrorV3::Missing)?;
            Ok(original.$field.take())
        }
    };
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
    panic: Option<Box<dyn Any + Send>>,
    stop_authorization: Option<PcSupervisorStopAuthorizationV3>,
    stop_handed_off: bool,
    joined_proof: Option<PcSupervisorJoinedProofV3>,
    no_worker_proof: Option<PcSupervisorNoWorkerProofV3>,
    before_admission_join_proof: Option<PcSupervisorBeforeAdmissionJoinProofV3>,
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
                panic: None,
                stop_authorization: None,
                stop_handed_off: false,
                joined_proof: None,
                no_worker_proof: None,
                before_admission_join_proof: None,
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
        if original.state != PcRunRecoveryStateV3::Executing
            || !Arc::ptr_eq(outcome.original_collection(), original.owners.native())
            || !original
                .outcome_key
                .as_ref()
                .is_some_and(|key| key.same_original(&outcome.recovery_key()))
        {
            return Err(PcRunRecoveryRecordErrorV3::Invalid);
        }
        original.state = PcRunRecoveryStateV3::Returned;
        Ok(())
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

    evidence_custody!(
        put_joined_proof,
        take_joined_proof,
        joined_proof,
        PcSupervisorJoinedProofV3
    );
    evidence_custody!(
        put_no_worker_proof,
        take_no_worker_proof,
        no_worker_proof,
        PcSupervisorNoWorkerProofV3
    );
    evidence_custody!(
        put_before_admission_join_proof,
        take_before_admission_join_proof,
        before_admission_join_proof,
        PcSupervisorBeforeAdmissionJoinProofV3
    );

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
