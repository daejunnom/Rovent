//! One-root P-C /3 supervisor and recovery custody.
//!
//! A prepared root is not an execution, teacher label or physical-completion
//! proof. The bridge must retain the same recovery Arc before any claim,
//! source handoff or physical worker spawn.
use super::capture_router_v3::CaptureRouterV3;
use super::game_authority_v3::{PcOperationClaimV3, PcOperationKindV3, PcOperationObservationV3};
use super::launch::ManagedOutputOwnerV3;
use super::native_capture::{
    RoventNativeCaptureRegistration, RoventPreRouteFenceAttemptV3, RoventPreRouteFenceWitnessV3,
    RoventRouteRollbackV3, RoventUnstartedRouteRollbackV3,
};
use super::observer_lock_v3::RoventCollectionOwnerLockV3;
use super::pc_hooks_lock_v3::{
    PcPreparedOutcomeV3, PcPreparedRootV3, PcQuotaEvidenceErrorV3, PcQuotaOutcomeV3,
    PcReceivedRootV3, PcRootAccountingV3, RoventPcCollectionBridgeV3,
};
use super::policy_row_pipeline_v3::{
    PolicyChunkPersistenceV3, PolicyOutputOwnerV3, PolicyRootRetainedOwnerV3,
    PolicyRowPipelineOwnerV3,
};
use super::speculative_route_v3::{SpeculativeReceiverV3, SpeculativeRootV3};
use super::teacher::v3_policy_consumer::{
    TeacherPolicyConsumerOwnerV3, TeacherPolicyConsumerProgressV3, TeacherPolicyConsumerStartV3,
    TeacherProcessedOwnerV3,
};
use super::teacher::v3_pump::{
    TeacherClosedCustodyStartV3, TeacherClosedCustodyV3, TeacherClosedDrainV3,
    TeacherClosedHandoffV3, TeacherPendingDrainV3, TeacherPumpSetupErrorV3, TeacherPumpStepV3,
    TeacherPumpV3,
};
use super::teacher::v3_session::{
    RoventCollectionTeacherSessionV3, TeacherPrepareErrorV3, TeacherPrepareOwnerV3,
    TeacherSessionClaimErrorV3, TeacherSessionStageStateV3, TeacherSessionStagingV3,
    TeacherStartErrorV3, is_teacher_scope_busy,
};
use super::writer_lock_v3::RoventCorpusWriterLockV3;
use rz_search::external_cpu::collection_teacher::TeacherDiagnostics;
use rz_uci::pals_adaptive::{
    AdaptiveCollectionError, RoventCollectionRootAdmissionV3, RoventTeacherScopeCapabilityV3,
    RoventTeacherScopePhysicalAckV3, RoventTeacherScopeWorkerAckV3,
};
use rz_uci::rovent_bootstrap::pc_collection_v3::{
    RootHandoffV3, RoventPcCollectionWorkerKeyV3, RoventPcCollectionWorkerSpawnFailureDetailsV3,
    RoventPcCollectionWorkerSpawnFailureV3, RoventPcCollectionWorkerV3, RoventPcRootWorkerTaskV3,
};
use rz_uci::rovent_external_launch::{
    RoventTeacherScopeAggregateAckV3, RoventTeacherScopeAggregateFailureV3,
    TeacherRecoveryCustodianV3,
};
use std::sync::{
    Arc, Mutex, OnceLock, Weak,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

/// Preserve the first root failure even when a later operation finishes after
/// the recovery guard was released and reacquired.
fn record_first_root_failure(failure: &mut Option<String>, error: impl Into<String>) {
    failure.get_or_insert_with(|| error.into());
}

fn record_teacher_drain_error(
    failure: &mut Option<String>,
    drain_failure: &mut Option<String>,
    error: String,
) {
    failure.get_or_insert_with(|| error.clone());
    drain_failure.get_or_insert(error);
}

fn record_teacher_drain_shutdown(
    failure: &mut Option<String>,
    drain_failure: &mut Option<String>,
    error: &AdaptiveCollectionError,
) {
    if !super::teacher::v3_session::is_teacher_scope_busy(error) {
        record_teacher_drain_error(
            failure,
            drain_failure,
            format!("P-C V3 teacher shutdown: {error}"),
        );
    }
}

fn try_teacher_recovery_lock<'a, T>(
    lock: &'a Mutex<T>,
    failure: &mut Option<String>,
    drain_failure: &mut Option<String>,
    name: &'static str,
) -> Option<std::sync::MutexGuard<'a, T>> {
    match lock.try_lock() {
        Ok(guard) => Some(guard),
        Err(std::sync::TryLockError::WouldBlock) => None,
        Err(std::sync::TryLockError::Poisoned(_)) => {
            record_teacher_drain_error(
                failure,
                drain_failure,
                format!("P-C V3 teacher {name} lock poisoned"),
            );
            None
        }
    }
}

fn retain_teacher_drain_aggregate<T>(
    failure: &mut Option<String>,
    drain_failure: &mut Option<String>,
    result: Result<Option<T>, String>,
) -> Option<T> {
    match result {
        Ok(value) => value,
        Err(error) => {
            record_teacher_drain_error(
                failure,
                drain_failure,
                format!("P-C V3 teacher aggregate: {error}"),
            );
            None
        }
    }
}

fn teacher_drain_worker_finished(
    failure: &mut Option<String>,
    drain_failure: &mut Option<String>,
    result: Result<bool, &'static str>,
) -> bool {
    match result {
        Ok(finished) => finished,
        Err(error) => {
            record_teacher_drain_error(
                failure,
                drain_failure,
                format!("P-C V3 teacher worker finish: {error}"),
            );
            false
        }
    }
}

fn teacher_pending_drain_ready(
    failure: &mut Option<String>,
    drain_failure: &mut Option<String>,
    result: Result<TeacherPendingDrainV3, String>,
    still_pending: bool,
) -> bool {
    match result {
        Ok(TeacherPendingDrainV3::Drained) => !still_pending,
        Ok(
            TeacherPendingDrainV3::Pending
            | TeacherPendingDrainV3::Full
            | TeacherPendingDrainV3::Unknown,
        ) => false,
        Err(error) => {
            record_teacher_drain_error(
                failure,
                drain_failure,
                format!("P-C V3 teacher pending drain: {error}"),
            );
            false
        }
    }
}

/// Record a pump error without treating a backpressure or unknown step as a
/// failure or discarding the pump's retained input, output, or ACK state.
fn record_teacher_pump_result(
    failure: &mut Option<String>,
    drain_failure: &mut Option<String>,
    result: Result<TeacherPumpStepV3, String>,
) {
    if let Err(error) = result {
        record_teacher_drain_error(
            failure,
            drain_failure,
            format!("P-C V3 teacher pump: {error}"),
        );
    }
}

#[cfg(test)]
mod teacher_drain_error_tests {
    use super::*;
    use rz_uci::pals_adaptive::AdaptiveCollectionStage;

    #[test]
    fn drain_preserves_real_errors_and_keeps_busy_pending() {
        let mut failure = None;
        let mut drain_failure = None;
        let busy = AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Settled,
            "TeacherScopeBusy",
            "busy",
        );
        record_teacher_drain_shutdown(&mut failure, &mut drain_failure, &busy);
        assert!(failure.is_none());
        let poison = AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Settled,
            "TeacherScopePoisoned",
            "poisoned",
        );
        record_teacher_drain_shutdown(&mut failure, &mut drain_failure, &poison);
        assert!(failure.as_ref().unwrap().contains("TeacherScopePoisoned"));
        let first = failure.clone();
        assert!(
            retain_teacher_drain_aggregate::<()>(
                &mut failure,
                &mut drain_failure,
                Err("aggregate failed".into())
            )
            .is_none()
        );
        assert_eq!(failure, first);
        failure = None;
        assert!(
            retain_teacher_drain_aggregate::<()>(&mut failure, &mut drain_failure, Ok(None))
                .is_none()
        );
        assert!(failure.is_none());
        assert!(
            retain_teacher_drain_aggregate::<()>(
                &mut failure,
                &mut drain_failure,
                Err("aggregate failed".into())
            )
            .is_none()
        );
        assert!(failure.as_ref().unwrap().contains("aggregate failed"));
        failure = None;
        assert!(!teacher_drain_worker_finished(
            &mut failure,
            &mut drain_failure,
            Ok(false)
        ));
        assert!(failure.is_none());
        assert!(!teacher_drain_worker_finished(
            &mut failure,
            &mut drain_failure,
            Err("actor absent")
        ));
        assert!(failure.as_ref().unwrap().contains("actor absent"));
        failure = None;
        for state in [
            TeacherPendingDrainV3::Pending,
            TeacherPendingDrainV3::Full,
            TeacherPendingDrainV3::Unknown,
        ] {
            assert!(!teacher_pending_drain_ready(
                &mut failure,
                &mut drain_failure,
                Ok(state),
                true
            ));
            assert!(failure.is_none());
        }
        assert!(!teacher_pending_drain_ready(
            &mut failure,
            &mut drain_failure,
            Ok(TeacherPendingDrainV3::Drained),
            true
        ));
        assert!(teacher_pending_drain_ready(
            &mut failure,
            &mut drain_failure,
            Ok(TeacherPendingDrainV3::Drained),
            false
        ));
        assert!(!teacher_pending_drain_ready(
            &mut failure,
            &mut drain_failure,
            Err("pump failed".into()),
            true
        ));
        assert!(failure.as_ref().unwrap().contains("pump failed"));
    }

    #[test]
    fn drain_failure_is_retained_even_if_root_already_failed() {
        let mut failure = Some("original root failure".to_owned());
        let mut drain_failure = None;
        assert!(!teacher_drain_worker_finished(
            &mut failure,
            &mut drain_failure,
            Err("actor absent")
        ));
        assert_eq!(failure.as_deref(), Some("original root failure"));
        assert!(drain_failure.as_deref().unwrap().contains("actor absent"));
    }

    #[test]
    fn pump_error_preserves_first_root_and_secondary_without_promoting_pending() {
        let mut failure = Some("original root failure".to_owned());
        let mut drain_failure = None;
        for step in [
            TeacherPumpStepV3::Idle,
            TeacherPumpStepV3::AuthorityBusy,
            TeacherPumpStepV3::OutputBackpressure,
            TeacherPumpStepV3::SubmissionUnknown,
            TeacherPumpStepV3::ObservationUnknown,
        ] {
            record_teacher_pump_result(&mut failure, &mut drain_failure, Ok(step));
            assert!(drain_failure.is_none());
        }
        record_teacher_pump_result(
            &mut failure,
            &mut drain_failure,
            Err("V3 teacher original receiver poisoned".into()),
        );
        assert_eq!(failure.as_deref(), Some("original root failure"));
        assert_eq!(
            drain_failure.as_deref(),
            Some("P-C V3 teacher pump: V3 teacher original receiver poisoned")
        );
    }

    #[test]
    fn later_consumer_and_publish_failures_preserve_original_root_failure() {
        let mut failure = None;
        record_first_root_failure(&mut failure, "original root failure");
        record_first_root_failure(&mut failure, "consumer lock poisoned");
        record_first_root_failure(&mut failure, "P-C V3 publish outcome unknown");
        assert_eq!(failure.as_deref(), Some("original root failure"));
    }

    #[test]
    fn recovery_lock_contention_is_pending_and_poison_preserves_first_failure() {
        let lock = Arc::new(Mutex::new(()));
        let guard = lock.lock().unwrap();
        let mut failure = None;
        let mut drain_failure = None;
        assert!(
            try_teacher_recovery_lock(&lock, &mut failure, &mut drain_failure, "session").is_none()
        );
        assert!(failure.is_none());
        drop(guard);
        let child = Arc::clone(&lock);
        assert!(
            std::thread::spawn(move || {
                let _held = child.lock().unwrap();
                panic!("intentional fixture poison");
            })
            .join()
            .is_err()
        );
        assert!(
            try_teacher_recovery_lock(&lock, &mut failure, &mut drain_failure, "session").is_none()
        );
        assert_eq!(
            failure.as_deref(),
            Some("P-C V3 teacher session lock poisoned")
        );
        assert!(
            try_teacher_recovery_lock(&lock, &mut failure, &mut drain_failure, "stage").is_none()
        );
        assert_eq!(
            failure.as_deref(),
            Some("P-C V3 teacher session lock poisoned")
        );
        let mut original_failure = Some("original root failure".to_owned());
        let mut pump_failure = None;
        assert!(
            try_teacher_recovery_lock(&lock, &mut original_failure, &mut pump_failure, "pump")
                .is_none()
        );
        assert_eq!(original_failure.as_deref(), Some("original root failure"));
        assert_eq!(
            pump_failure.as_deref(),
            Some("P-C V3 teacher pump lock poisoned")
        );
    }
}

#[path = "pc_worker_operation_v3.rs"]
mod pc_worker_operation_v3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PcUnstartedTaskDispositionV3 {
    Pending,
    Released,
    Unknown,
}

/// The caller must invoke this outside the recovery guard, after observing
/// the original spawn producer's returned error. This does not prove that any
/// other teacher, native or GPU owner has completed.
fn release_unstarted_pc_worker_task_v3(
    task: RoventPcRootWorkerTaskV3,
) -> Result<(), Box<dyn std::any::Any + Send>> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(task)))
}

fn retain_pc_panic_v3(
    destination: &mut Option<Box<dyn std::any::Any + Send>>,
    additional: Box<dyn std::any::Any + Send>,
) {
    // Another owner may have recorded an unrelated panic while the recovery
    // mutex was released. Preserve both original payloads rather than replace
    // and drop the earlier unknown outcome.
    *destination = Some(match destination.take() {
        Some(previous) => Box::new((previous, additional)),
        None => additional,
    });
}

/// Keep the original proof and claim in their original owner across unwind.
/// `access` only projects fields; it must not perform fallible work.
fn with_preserved_pc_route_owners_v3<S, P, C, R>(
    slot: &mut S,
    access: impl for<'a> Fn(&'a mut S) -> (&'a mut Option<P>, &'a mut Option<C>),
    operation: impl FnOnce(&mut S, Option<&P>, &mut Option<C>) -> R,
) -> Result<R, Box<dyn std::any::Any + Send>> {
    let (mut proof, mut claim) = {
        let (proof, claim) = access(slot);
        (proof.take(), claim.take())
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        operation(slot, proof.as_ref(), &mut claim)
    }));
    let (proof_slot, claim_slot) = access(slot);
    *proof_slot = proof.take();
    *claim_slot = claim;
    result
}

/// The task has been returned by the original spawn attempt and detached from
/// its outcome slot. Keep the recovery unlocked while disposing its captures.
fn finish_failed_pc_task_release_v3(
    recovery: &Arc<Mutex<PcRootRecoveryV3>>,
    original_task: RoventPcRootWorkerTaskV3,
) -> Result<(), String> {
    let disposition = release_unstarted_pc_worker_task_v3(original_task);
    let mut state = match recovery.lock() {
        Ok(state) => state,
        Err(poison) => {
            let mut state = poison.into_inner();
            state.worker_failure_task_disposition = Some(PcUnstartedTaskDispositionV3::Unknown);
            state.worker_failure_release_poisoned = true;
            if let Err(payload) = disposition {
                state.worker_failure_task_drop_panicked = true;
                retain_pc_panic_v3(&mut state.panic, payload);
            }
            state.failure.get_or_insert_with(|| {
                "P-C V3 recovery poisoned during unstarted task release".into()
            });
            return Err("P-C V3 recovery poisoned".into());
        }
    };
    match disposition {
        Ok(()) => {
            state.worker_failure_task_disposition = Some(PcUnstartedTaskDispositionV3::Released)
        }
        Err(payload) => {
            state.worker_failure_task_disposition = Some(PcUnstartedTaskDispositionV3::Unknown);
            state.worker_failure_task_drop_panicked = true;
            retain_pc_panic_v3(&mut state.panic, payload);
            state
                .failure
                .get_or_insert_with(|| "P-C V3 unstarted worker task release unknown".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod unstarted_worker_task_tests {
    use super::*;

    #[test]
    fn returned_task_release_breaks_its_original_owner_cycle() {
        let owner = Arc::new(Mutex::new(None::<RoventPcRootWorkerTaskV3>));
        let captured = Arc::clone(&owner);
        *owner.lock().unwrap() = Some(Box::new(move |_| Box::new(captured)));
        let weak = Arc::downgrade(&owner);
        let returned = owner.lock().unwrap().take().unwrap();
        assert!(release_unstarted_pc_worker_task_v3(returned).is_ok());
        drop(owner);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn panicking_task_destructor_is_observed_as_unknown() {
        struct PanicOnDrop;
        impl Drop for PanicOnDrop {
            fn drop(&mut self) {
                panic!("task destructor failed");
            }
        }
        let captured = PanicOnDrop;
        let task: RoventPcRootWorkerTaskV3 = Box::new(move |_| {
            let _ = &captured;
            Box::new(())
        });
        assert!(release_unstarted_pc_worker_task_v3(task).is_err());
    }

    #[test]
    fn task_release_panic_does_not_replace_an_existing_payload() {
        type Payload = Box<dyn std::any::Any + Send>;
        let mut original: Option<Payload> = Some(Box::new("earlier"));
        retain_pc_panic_v3(&mut original, Box::new("release"));
        let pair = original.unwrap().downcast::<(Payload, Payload)>().unwrap();
        let (earlier, release) = *pair;
        assert_eq!(*earlier.downcast::<&str>().unwrap(), "earlier");
        assert_eq!(*release.downcast::<&str>().unwrap(), "release");
    }

    #[test]
    fn route_unwind_retains_original_proof_claim_and_partial_result() {
        struct Slot {
            proof: Option<Arc<()>>,
            claim: Option<Arc<()>>,
            result_retained: bool,
        }
        let proof = Arc::new(());
        let claim = Arc::new(());
        let mut slot = Slot {
            proof: Some(Arc::clone(&proof)),
            claim: Some(Arc::clone(&claim)),
            result_retained: false,
        };
        let result: Result<(), _> = with_preserved_pc_route_owners_v3(
            &mut slot,
            |slot| (&mut slot.proof, &mut slot.claim),
            |slot, original_proof, original_claim| {
                assert!(Arc::ptr_eq(original_proof.unwrap(), &proof));
                assert!(Arc::ptr_eq(original_claim.as_ref().unwrap(), &claim));
                slot.result_retained = true;
                panic!("route operation failed after result custody");
            },
        );
        assert!(result.is_err());
        assert!(Arc::ptr_eq(slot.proof.as_ref().unwrap(), &proof));
        assert!(Arc::ptr_eq(slot.claim.as_ref().unwrap(), &claim));
        assert!(slot.result_retained);
    }
}

/// Pre-owned before a teacher scope can be claimed. The separate slot lock
/// permits revalidation with the original supervisor/recovery guards held.
struct PcScopeEffectSlotsV3 {
    proof: Option<PcReceivedStageProofV3>,
    teacher_source: Option<super::teacher::v3_session::TeacherScopeEffectSourceV3>,
    game_source: Option<super::game_authority_v3::GameEffectSourceV3<PcWorkerSupervisorV3>>,
    claim: Option<super::game_authority_v3::GameEffectClaimV3<PcWorkerSupervisorV3>>,
    outcome: Option<
        Result<
            super::teacher::v3_session::TeacherScopeEffectOutcomeV3,
            super::teacher::v3_session::TeacherScopeEffectClaimFailureV3,
        >,
    >,
    unclaimed_proof: Option<super::teacher::v3_session::TeacherScopeUnclaimedProofV3>,
    before_source_proof: Option<super::teacher::v3_session::TeacherScopeBeforeSourceProofV3>,
    issue_error: Option<TeacherSessionClaimErrorV3>,
    source_error: Option<&'static str>,
    claim_error: Option<&'static str>,
    completion_error: Option<&'static str>,
    source_step: Option<super::game_authority_v3::GameStepV3>,
    claim_step: Option<super::game_authority_v3::GameStepV3>,
    completion_step: Option<super::game_authority_v3::GameStepV3>,
    finished: bool,
}
impl PcScopeEffectSlotsV3 {
    fn new() -> Self {
        Self {
            proof: None,
            teacher_source: None,
            game_source: None,
            claim: None,
            outcome: None,
            unclaimed_proof: None,
            before_source_proof: None,
            issue_error: None,
            source_error: None,
            claim_error: None,
            completion_error: None,
            source_step: None,
            claim_step: None,
            completion_step: None,
            finished: false,
        }
    }
}

/// A separate owner for one original actor's acquisition state. It is
/// registered with the root before source issuance or process creation.
struct PcActorEffectSlotsV3 {
    actor_id: String,
    proof: Option<PcReceivedActorProofV3>,
    teacher_source: Option<super::teacher::v3_session::TeacherActorEffectSourceV3>,
    game_source: Option<super::game_authority_v3::GameEffectSourceV3<PcWorkerSupervisorV3>>,
    claim: Option<super::game_authority_v3::GameEffectClaimV3<PcWorkerSupervisorV3>>,
    outcome: Option<
        Result<
            (
                rz_uci::rovent_external_launch::TeacherRecoveryKeyV3,
                super::teacher::v3_session::TeacherActorEffectOutcomeV3,
            ),
            super::teacher::v3_session::TeacherActorEffectFailureV3,
        >,
    >,
    issue_error: Option<TeacherStartErrorV3>,
    source_error: Option<&'static str>,
    claim_error: Option<&'static str>,
    completion_error: Option<&'static str>,
    source_step: Option<super::game_authority_v3::GameStepV3>,
    claim_step: Option<super::game_authority_v3::GameStepV3>,
    completion_step: Option<super::game_authority_v3::GameStepV3>,
    finished: bool,
}
impl PcActorEffectSlotsV3 {
    fn new(actor_id: String) -> Self {
        Self {
            actor_id,
            proof: None,
            teacher_source: None,
            game_source: None,
            claim: None,
            outcome: None,
            issue_error: None,
            source_error: None,
            claim_error: None,
            completion_error: None,
            source_step: None,
            claim_step: None,
            completion_step: None,
            finished: false,
        }
    }
}

/// One original root's bounded operation slot. The route receiver remains in
/// this slot from installation until its original attempt has settled.
pub(super) struct PcOperationSlotV3 {
    original_self: Weak<Mutex<PcOperationSlotV3>>,
    kind: PcOperationKindV3,
    identity: Arc<()>,
    admission: RoventCollectionRootAdmissionV3,
    recovery: Weak<Mutex<PcRootRecoveryV3>>,
    registration: RoventNativeCaptureRegistration,
    proof: Option<PcReceivedStageProofV3>,
    claim: Option<PcOperationClaimV3>,
    generation: Option<u64>,
    route_destination: Option<(SpeculativeRootV3, Arc<Mutex<SpeculativeReceiverV3>>)>,
    route_result: Option<Result<Option<()>, crate::ArenaError>>,
    route_unknown: bool,
    worker_destination:
        Option<Result<RoventPcCollectionWorkerV3, RoventPcCollectionWorkerSpawnFailureV3>>,
    worker_undispatched: Option<(RoventCollectionRootAdmissionV3, RoventPcRootWorkerTaskV3)>,
    worker_failure_details: Option<RoventPcCollectionWorkerSpawnFailureDetailsV3>,
    worker_unknown: bool,
}

fn pc_operation_index_v3(kind: PcOperationKindV3) -> usize {
    match kind {
        PcOperationKindV3::RouteInstall => 0,
        PcOperationKindV3::NativeWorker => 1,
        PcOperationKindV3::AttachRoute => 2,
        PcOperationKindV3::PrepareAdmission => 3,
        PcOperationKindV3::BindPolicyConsumer => 4,
        PcOperationKindV3::PublishPrepared => 5,
    }
}

impl PcOperationSlotV3 {
    pub(super) fn kind(&self) -> PcOperationKindV3 {
        self.kind
    }
    pub(super) fn identity(&self) -> &Arc<()> {
        &self.identity
    }
    pub(super) fn admission(&self) -> &RoventCollectionRootAdmissionV3 {
        &self.admission
    }
    pub(super) fn abort_source_proof(&self) -> Option<&PcReceivedStageProofV3> {
        self.proof.as_ref()
    }

    /// Reobserve this exact pre-retained slot and received-stage proof
    /// without reacquiring the supervisor/recovery mutexes held by the caller.
    pub(super) fn try_matches_abort_source(
        &self,
        slot_owner: &Arc<Mutex<PcOperationSlotV3>>,
        recovery: &PcRootRecoveryV3,
        registration: &RoventNativeCaptureRegistration,
        index: usize,
    ) -> Option<bool> {
        if pc_operation_index_v3(self.kind) != index
            || index >= recovery.operation_slots.len()
            || !Weak::ptr_eq(&self.original_self, &Arc::downgrade(slot_owner))
            || !Weak::ptr_eq(&self.recovery, &recovery.original_self)
            || !recovery.operation_slots[index]
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, slot_owner))
            || !Arc::ptr_eq(&self.identity, &recovery.operation_identities[index])
            || !self.admission.same_original_admission(&recovery.admission)
            || !self.registration.same_original_owner(registration)
        {
            return Some(false);
        }
        let Some(proof) = self.proof.as_ref() else {
            return None;
        };
        if !proof.matches_admission(&recovery.admission)
            || !Weak::ptr_eq(&proof.recovery, &recovery.original_self)
            || !recovery
                .stage
                .as_ref()
                .is_some_and(|stage| Weak::ptr_eq(&proof.stage, &Arc::downgrade(stage)))
            || !recovery
                .original_scope_identity
                .as_ref()
                .is_some_and(|scope| Weak::ptr_eq(&proof.scope_identity, scope))
        {
            return Some(false);
        }
        // A vanished weak owner is unresolved; never convert it to a safe
        // post-route completion or a reason to release original custody.
        let bridge = proof.bridge.upgrade()?;
        let _supervisor = proof.supervisor.upgrade()?;
        if !bridge
            .capture_registration()
            .same_original_owner(registration)
            || !recovery.matches_received_root(&bridge)
            || proof.matches_game(bridge.game_authority()) != Some(true)
        {
            return Some(false);
        }
        if !bridge.game_authority().abort_requested() {
            use super::game_authority_v3::GameStepV3 as G;
            match bridge
                .game_authority()
                .try_observe_pc_root_cancel_quiescent(&self.admission)
            {
                G::Ready => {}
                G::Busy | G::Unknown => return None,
                _ => return Some(false),
            }
        }
        Some(true)
    }

    /// Exhaustive by design: adding a new producer/result field requires an
    /// explicit decision before an unattempted slot can authorize cleanup.
    pub(super) fn has_no_attempt_data(&self) -> bool {
        let Self {
            original_self: _,
            kind: _,
            identity: _,
            admission: _,
            recovery: _,
            registration: _,
            proof: _,
            claim,
            generation,
            route_destination,
            route_result,
            route_unknown,
            worker_destination,
            worker_undispatched,
            worker_failure_details,
            worker_unknown,
        } = self;
        claim.is_none()
            && generation.is_none()
            && route_destination.is_none()
            && route_result.is_none()
            && !*route_unknown
            && worker_destination.is_none()
            && worker_undispatched.is_none()
            && worker_failure_details.is_none()
            && !*worker_unknown
    }

    /// Reobserve this producer's actual failed, no-install result only after
    /// authority has settled the matching generation. An arbitrary error,
    /// missing destination, or panic alone is not this evidence.
    pub(super) fn has_settled_failed_route_no_effect(&self, original_generation: u64) -> bool {
        let Self {
            original_self: _,
            kind,
            identity: _,
            admission: _,
            recovery: _,
            registration: _,
            proof: _,
            claim,
            generation,
            route_destination,
            route_result,
            route_unknown,
            worker_destination,
            worker_undispatched,
            worker_failure_details,
            worker_unknown,
        } = self;
        *kind == PcOperationKindV3::RouteInstall
            && claim.is_none()
            && *generation == Some(original_generation)
            && route_destination.is_none()
            && matches!(route_result, Some(Err(_)))
            && !*route_unknown
            && worker_destination.is_none()
            && worker_undispatched.is_none()
            && worker_failure_details.is_none()
            && !*worker_unknown
    }

    pub(super) fn try_revalidate_with_guards(
        &self,
        proof: &PcReceivedStageProofV3,
        supervisor: &PcWorkerSupervisorV3,
        recovery: &PcRootRecoveryV3,
    ) -> Result<Option<bool>, &'static str> {
        if !self.admission.same_original_admission(&recovery.admission)
            || !Weak::ptr_eq(&self.recovery, &recovery.original_self)
            || !Arc::ptr_eq(
                &self.identity,
                &recovery.operation_identities[pc_operation_index_v3(self.kind)],
            )
            || !recovery.operation_slots[pc_operation_index_v3(self.kind)]
                .as_ref()
                .is_some_and(|original| {
                    Weak::ptr_eq(&self.original_self, &Arc::downgrade(original))
                })
            || !self
                .registration
                .same_original_owner(supervisor.bridge.capture_registration())
        {
            return Ok(Some(false));
        }
        proof.try_revalidate_with_guards(supervisor, recovery)
    }

    pub(super) fn observation(
        &self,
        generation: u64,
        recovery: &PcRootRecoveryV3,
    ) -> PcOperationObservationV3 {
        if self.generation != Some(generation)
            || !Weak::ptr_eq(&self.recovery, &recovery.original_self)
            || !Arc::ptr_eq(
                &self.identity,
                &recovery.operation_identities[pc_operation_index_v3(self.kind)],
            )
        {
            return PcOperationObservationV3::Unknown;
        }
        if self.kind == PcOperationKindV3::NativeWorker {
            if self.worker_unknown || self.worker_undispatched.is_some() {
                return PcOperationObservationV3::Unknown;
            }
            return match (
                &self.worker_destination,
                &self.worker_failure_details,
                recovery.worker_failure_task_disposition,
            ) {
                (None, None, None) => PcOperationObservationV3::Pending,
                (Some(Ok(worker)), None, None) if worker.matches_admission(&self.admission) => {
                    PcOperationObservationV3::Committed
                }
                (Some(Err(failure)), None, None)
                    if failure.admission.same_original_admission(&self.admission) =>
                {
                    PcOperationObservationV3::Pending
                }
                (None, Some(details), Some(PcUnstartedTaskDispositionV3::Pending))
                    if details.matches_admission(&self.admission) =>
                {
                    PcOperationObservationV3::Pending
                }
                (None, Some(details), Some(PcUnstartedTaskDispositionV3::Released))
                    if details.matches_admission(&self.admission) =>
                {
                    PcOperationObservationV3::FailedNoEffect
                }
                _ => PcOperationObservationV3::Unknown,
            };
        }
        if self.kind != PcOperationKindV3::RouteInstall {
            return PcOperationObservationV3::Unknown;
        }
        if self.route_unknown {
            return PcOperationObservationV3::Unknown;
        }
        match (&self.route_result, &self.route_destination) {
            (None, _) => PcOperationObservationV3::Pending,
            (Some(Ok(None)), None) => PcOperationObservationV3::NoEffect,
            (Some(Err(_)), None) => PcOperationObservationV3::FailedNoEffect,
            (Some(Ok(Some(()))), Some((root, receiver))) => {
                let receiver = match receiver.try_lock() {
                    Ok(receiver) => receiver,
                    Err(std::sync::TryLockError::WouldBlock) => {
                        return PcOperationObservationV3::Pending;
                    }
                    Err(_) => return PcOperationObservationV3::Unknown,
                };
                if root.matches_admission(&self.admission)
                    && receiver.matches_original_route(root)
                    && self.registration.owns_speculative_root_v3(root)
                {
                    PcOperationObservationV3::Committed
                } else {
                    PcOperationObservationV3::Unknown
                }
            }
            _ => PcOperationObservationV3::Unknown,
        }
    }
}

/// Separate historical custody for a route that was installed and then
/// rolled back before an observer or worker existed. Never a native fence.
struct PcPostRouteRollbackV3 {
    route: SpeculativeRootV3,
    receiver: Arc<Mutex<SpeculativeReceiverV3>>,
    witness: RoventUnstartedRouteRollbackV3,
}

/// The original scope, route, session and any uncertain worker remain rooted
/// here even if preparation or a worker task is interrupted.
pub(crate) struct PcRootRecoveryV3 {
    original_self: Weak<Mutex<PcRootRecoveryV3>>,
    operation_identities: [Arc<()>; 6],
    operation_slots: [Option<Arc<Mutex<PcOperationSlotV3>>>; 6],
    scope_effects: Arc<Mutex<PcScopeEffectSlotsV3>>,
    actor_effects: Vec<Arc<Mutex<PcActorEffectSlotsV3>>>,
    admission: RoventCollectionRootAdmissionV3,
    received_root: Option<PcReceivedRootV3>,
    stage: Option<Arc<Mutex<TeacherSessionStagingV3>>>,
    original_scope_identity: Option<Weak<()>>,
    preflight_failure: Option<TeacherPrepareErrorV3>,
    stage_claim_failure: Option<TeacherSessionClaimErrorV3>,
    claimed_unstarted: Option<ClaimedUnstartedV3>,
    claimed_unstarted_release: NoScopeRecoveryV3,
    sealed_claimed_unstarted: bool,
    unmatched_claim_owner: Option<TeacherPrepareOwnerV3>,
    route: Option<SpeculativeRootV3>,
    receiver: Option<Arc<Mutex<SpeculativeReceiverV3>>>,
    owner: Option<RoventCollectionOwnerLockV3>,
    pump: Option<Arc<Mutex<TeacherPumpV3>>>,
    closed_custody: Option<TeacherClosedCustodyV3>,
    closed_custody_setup_unknown: bool,
    abort_targets_revoked: bool,
    abort_input_observation: Option<TeacherClosedDrainV3>,
    abort_handoff_observation: Option<TeacherClosedHandoffV3>,
    abort_terminal_observation: Option<TeacherPendingDrainV3>,
    abort_terminal_unknown: bool,
    pipeline: PolicyRowPipelineOwnerV3,
    output: PolicyOutputOwnerV3,
    processed: Option<TeacherProcessedOwnerV3>,
    retained: Option<PolicyRootRetainedOwnerV3>,
    consumer: Option<TeacherPolicyConsumerOwnerV3>,
    consumer_setup_unknown: bool,
    consumer_phase: ConsumerPhaseV3,
    pump_setup_failure: Option<TeacherPumpSetupErrorV3>,
    pump_setup_unknown: bool,
    actor_start_failure: Option<TeacherStartErrorV3>,
    panic: Option<Box<dyn std::any::Any + Send>>,
    worker: Option<RoventPcCollectionWorkerV3>,
    worker_spawn_failure: Option<RoventPcCollectionWorkerSpawnFailureDetailsV3>,
    worker_failure_task_disposition: Option<PcUnstartedTaskDispositionV3>,
    worker_failure_release_poisoned: bool,
    worker_failure_task_drop_panicked: bool,
    teacher_drain_failure: Option<String>,
    teacher_claim_observation_error: Option<&'static str>,
    worker_key: Option<RoventPcCollectionWorkerKeyV3>,
    handoff: Option<RootHandoffV3>,
    aggregate: Option<RoventTeacherScopeAggregateAckV3>,
    // A game abort is sticky for this original root. This is not a native
    // cancellation, an aggregate acknowledgement, or permission to reuse it.
    abort_before_route: bool,
    sealed_abort_before_route: bool,
    sealed_abort_no_scope: bool,
    aborted_pre_route_release: Option<Result<(), &'static str>>,
    aborted_no_scope_release: Option<Result<(), &'static str>>,
    aborted_unclaimed_stage_release: Option<Result<(), &'static str>>,
    post_route_result: Option<RoventRouteRollbackV3>,
    post_route_custody: Option<PcPostRouteRollbackV3>,
    pre_route_fence: Option<RoventPreRouteFenceWitnessV3>,
    post_route_unknown: bool,
    post_route_error: Option<String>,
    sealed_abort_post_route: bool,
    aborted_post_route_release: Option<Result<(), &'static str>>,
    diagnostics: Vec<(String, TeacherDiagnostics)>,
    prepared: Option<PcPreparedRootV3>,
    custodian: TeacherRecoveryCustodianV3,
    teacher_reserved: bool,
    no_scope_recovery: NoScopeRecoveryV3,
    sealed_unstarted: bool,
    next_actor: usize,
    published: bool,
    failure: Option<String>,
}

impl PcRootRecoveryV3 {
    fn new(
        admission: RoventCollectionRootAdmissionV3,
        pipeline: PolicyRowPipelineOwnerV3,
    ) -> Arc<Mutex<Self>> {
        let output = pipeline.output_owner();
        Arc::new_cyclic(|original_self| {
            Mutex::new(Self {
                original_self: original_self.clone(),
                operation_identities: std::array::from_fn(|_| Arc::new(())),
                operation_slots: std::array::from_fn(|_| None),
                scope_effects: Arc::new(Mutex::new(PcScopeEffectSlotsV3::new())),
                actor_effects: Vec::new(),
                admission,
                received_root: None,
                stage: None,
                original_scope_identity: None,
                preflight_failure: None,
                stage_claim_failure: None,
                claimed_unstarted: None,
                claimed_unstarted_release: NoScopeRecoveryV3::Pending,
                sealed_claimed_unstarted: false,
                unmatched_claim_owner: None,
                route: None,
                receiver: None,
                owner: None,
                pump: None,
                closed_custody: None,
                closed_custody_setup_unknown: false,
                abort_targets_revoked: false,
                abort_input_observation: None,
                abort_handoff_observation: None,
                abort_terminal_observation: None,
                abort_terminal_unknown: false,
                pipeline,
                output,
                processed: None,
                retained: None,
                consumer: None,
                consumer_setup_unknown: false,
                consumer_phase: ConsumerPhaseV3::Receive,
                pump_setup_failure: None,
                pump_setup_unknown: false,
                actor_start_failure: None,
                panic: None,
                worker: None,
                worker_spawn_failure: None,
                worker_failure_task_disposition: None,
                worker_failure_release_poisoned: false,
                worker_failure_task_drop_panicked: false,
                teacher_drain_failure: None,
                teacher_claim_observation_error: None,
                worker_key: None,
                handoff: None,
                aggregate: None,
                abort_before_route: false,
                sealed_abort_before_route: false,
                sealed_abort_no_scope: false,
                aborted_pre_route_release: None,
                aborted_no_scope_release: None,
                aborted_unclaimed_stage_release: None,
                post_route_result: None,
                post_route_custody: None,
                pre_route_fence: None,
                post_route_unknown: false,
                post_route_error: None,
                sealed_abort_post_route: false,
                aborted_post_route_release: None,
                diagnostics: Vec::new(),
                prepared: None,
                custodian: TeacherRecoveryCustodianV3::new(),
                teacher_reserved: false,
                no_scope_recovery: NoScopeRecoveryV3::Pending,
                sealed_unstarted: false,
                next_actor: 0,
                published: false,
                failure: None,
            })
        })
    }
    pub(crate) fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }

    /// Only the retained original receiver witness can establish that this
    /// recovery was actually received by this bridge. It is not a worker,
    /// dispatch-completion or physical-completion witness.
    pub(super) fn matches_received_root(&self, bridge: &Arc<RoventPcCollectionBridgeV3>) -> bool {
        self.received_root
            .as_ref()
            .is_some_and(|received| received.matches_original(bridge, &self.admission))
    }

    /// Retains the real aggregate in this same recovery. The presence of this
    /// evidence does not itself release quota or attest native/GPU completion.
    pub(crate) fn aborted_pre_route_aggregate(&self) -> Option<&RoventTeacherScopeAggregateAckV3> {
        if self.sealed_abort_before_route {
            self.aggregate.as_ref()
        } else {
            None
        }
    }

    /// This distinct receipt is valid only for the original route that was
    /// installed, revoked and rolled back before observer/worker creation.
    pub(crate) fn aborted_post_route_aggregate(&self) -> Option<&RoventTeacherScopeAggregateAckV3> {
        if self.sealed_abort_post_route {
            self.aggregate.as_ref()
        } else {
            None
        }
    }

    /// The recovery guard is held by the caller, so no new slot can be
    /// installed while this checks each later stage. Each authority check
    /// additionally observes sticky abort and the absence of its original
    /// root/kind record. Existing records remain held even after NoEffect.
    fn try_observe_later_operations_never_attempted(
        &self,
        registration: &RoventNativeCaptureRegistration,
    ) -> Result<Option<bool>, &'static str> {
        use super::game_authority_v3::GameStepV3 as G;
        for (index, slot_owner) in self.operation_slots.iter().enumerate().skip(1) {
            let Some(slot_owner) = slot_owner else {
                continue;
            };
            let slot = match slot_owner.try_lock() {
                Ok(slot) => slot,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                Err(_) => return Err("P-C V3 later operation slot poisoned"),
            };
            let Some(proof) = slot.abort_source_proof() else {
                return Ok(None);
            };
            let Some(bridge) = proof.bridge.upgrade() else {
                return Ok(None);
            };
            match bridge
                .game_authority()
                .try_observe_pc_operation_never_claimed(
                    slot_owner,
                    &slot,
                    self,
                    registration,
                    index,
                ) {
                G::Ready => {}
                G::Busy | G::Unknown => return Ok(None),
                _ => return Ok(Some(false)),
            }
        }
        Ok(Some(true))
    }

    /// Publish and retain this original admission's sticky registration fence.
    /// The fence closes route/ticket admission only, not physical work.
    pub(crate) fn try_retain_pre_route_fence(
        &mut self,
        registration: &RoventNativeCaptureRegistration,
        maximum_roots: u16,
    ) -> Result<Option<bool>, &'static str> {
        if let Some(original) = &self.pre_route_fence {
            return original.try_matches_original(registration, &self.admission);
        }
        match registration.try_seal_pre_route_v3(&self.admission, maximum_roots) {
            RoventPreRouteFenceAttemptV3::Sealed(witness)
            | RoventPreRouteFenceAttemptV3::AlreadySealed(witness) => {
                self.pre_route_fence = Some(witness);
                Ok(Some(true))
            }
            RoventPreRouteFenceAttemptV3::Busy => Ok(None),
            RoventPreRouteFenceAttemptV3::Conflict => {
                Err("P-C V3 pre-route registration conflicts with a route or ticket")
            }
            RoventPreRouteFenceAttemptV3::Capacity => {
                Err("P-C V3 pre-route fence capacity unavailable")
            }
            RoventPreRouteFenceAttemptV3::Poisoned => {
                Err("P-C V3 pre-route registration fence poisoned")
            }
        }
    }

    /// Reobserve the same sticky fence without publishing a new one.
    pub(crate) fn try_observe_pre_route_fence(
        &self,
        registration: &RoventNativeCaptureRegistration,
    ) -> Result<Option<bool>, &'static str> {
        match &self.pre_route_fence {
            Some(original) => original.try_matches_original(registration, &self.admission),
            None => Ok(Some(false)),
        }
    }

    /// A scope aggregate does not settle a still-owned authority claim.
    pub(crate) fn try_observe_teacher_claims_settled(&self) -> Result<Option<bool>, &'static str> {
        let scope = match self.scope_effects.try_lock() {
            Ok(scope) => scope,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 scope claim state poisoned"),
        };
        if scope.claim.is_some() && !scope.finished {
            return Ok(Some(false));
        }
        drop(scope);
        for owner in &self.actor_effects {
            let actor = match owner.try_lock() {
                Ok(actor) => actor,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                Err(_) => return Err("P-C V3 actor claim state poisoned"),
            };
            if actor.claim.is_some() && !actor.finished {
                return Ok(Some(false));
            }
        }
        Ok(Some(true))
    }

    /// Local observation only: a clear retained slot can still have an
    /// earlier settled no-effect record in authority. The enclosing authority
    /// and registration fence remain separate checks.
    pub(crate) fn try_observe_failure_operation_slots_clear(
        &self,
        registration: &RoventNativeCaptureRegistration,
    ) -> Result<Option<bool>, &'static str> {
        for (index, owner) in self.operation_slots.iter().enumerate() {
            let Some(owner) = owner else {
                continue;
            };
            let slot = match owner.try_lock() {
                Ok(slot) => slot,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                Err(_) => return Err("P-C V3 original operation slot poisoned"),
            };
            if pc_operation_index_v3(slot.kind) != index
                || !Weak::ptr_eq(&slot.original_self, &Arc::downgrade(owner))
                || !Weak::ptr_eq(&slot.recovery, &self.original_self)
                || !Arc::ptr_eq(&slot.identity, &self.operation_identities[index])
                || !slot.admission.same_original_admission(&self.admission)
                || !slot.registration.same_original_owner(registration)
            {
                return Ok(Some(false));
            }
            if slot.route_unknown || slot.worker_unknown {
                return Err("P-C V3 original operation outcome unknown");
            }
            if slot.claim.is_some() {
                return Ok(None);
            }
            if !slot.has_no_attempt_data() {
                return Ok(Some(false));
            }
        }
        Ok(Some(true))
    }

    /// A route can still be in its original operation slot after the caller
    /// has requested abort. Only the original sticky-abort authority may
    /// establish that this exact route operation never installed anything.
    fn try_observe_pre_route_never_started(
        &self,
        registration: &RoventNativeCaptureRegistration,
    ) -> Result<Option<bool>, &'static str> {
        use super::game_authority_v3::GameStepV3 as G;
        let Some(fence) = self.pre_route_fence.as_ref() else {
            return Ok(None);
        };
        match fence.try_matches_original(registration, &self.admission)? {
            Some(true) => {}
            other => return Ok(other),
        }
        let Some(slot_owner) = self.operation_slots[0].as_ref() else {
            // Slot creation precedes every route claim and slots are retained
            // for the lifetime of this original recovery.
            return Ok(Some(true));
        };
        let slot = match slot_owner.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 route operation slot poisoned"),
        };
        let Some(proof) = slot.abort_source_proof() else {
            return Ok(None);
        };
        let Some(bridge) = proof.bridge.upgrade() else {
            return Ok(None);
        };
        match bridge.game_authority().try_observe_pc_route_never_started(
            slot_owner,
            &slot,
            self,
            registration,
            fence,
        ) {
            G::Ready => Ok(Some(true)),
            G::Busy | G::Unknown => Ok(None),
            _ => Ok(Some(false)),
        }
    }

    fn try_validate_aborted_post_route(
        &self,
        registration: &RoventNativeCaptureRegistration,
    ) -> Result<Option<bool>, &'static str> {
        if self.post_route_unknown
            || self.post_route_error.is_some()
            || self.panic.is_some()
            || !self.abort_targets_revoked
            || !self.teacher_reserved
            || self.owner.is_some()
            || self.pump.is_some()
            || self.pump_setup_unknown
            || self.pump_setup_failure.is_some()
            || self.closed_custody.is_some()
            || self.closed_custody_setup_unknown
            || self.processed.is_some()
            || self.retained.is_some()
            || self.consumer.is_some()
            || self.consumer_setup_unknown
            || self.worker.is_some()
            || self.worker_key.is_some()
            || self.worker_spawn_failure.is_some()
            || self.handoff.is_some()
            || self.prepared.is_some()
            || self.published
            || self.abort_before_route
            || self.sealed_abort_before_route
            || self.sealed_abort_no_scope
        {
            return Ok(Some(false));
        }
        match self.try_observe_later_operations_never_attempted(registration)? {
            Some(true) => {}
            other => return Ok(other),
        }
        let (Some(custody), Some(route), Some(receiver), Some(slot_owner)) = (
            self.post_route_custody.as_ref(),
            self.route.as_ref(),
            self.receiver.as_ref(),
            self.operation_slots[0].as_ref(),
        ) else {
            return Ok(Some(false));
        };
        if !custody.route.same_original_route(route)
            || !Arc::ptr_eq(&custody.receiver, receiver)
            || !custody.witness.matches_original_route(route)
            || !route.matches_admission(&self.admission)
            || !self
                .aggregate
                .as_ref()
                .is_some_and(|ack| ack.matches_admission(&self.admission))
        {
            return Ok(Some(false));
        }
        let slot = match slot_owner.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 post-route operation slot poisoned"),
        };
        match slot.try_matches_abort_source(slot_owner, self, registration, 0) {
            Some(true) => {}
            Some(false) => return Ok(Some(false)),
            None => return Ok(None),
        }
        if slot.claim.is_some()
            || slot.route_unknown
            || slot.route_destination.is_some()
            || !matches!(slot.route_result.as_ref(), Some(Ok(Some(()))))
            || !slot.registration.same_original_owner(registration)
            || !Arc::ptr_eq(&slot.identity, &self.operation_identities[0])
        {
            return Ok(Some(false));
        }
        let Some(proof) = slot.proof.as_ref() else {
            return Ok(Some(false));
        };
        let Some(bridge) = proof.bridge.upgrade() else {
            return Ok(Some(false));
        };
        if !bridge.game_authority().abort_requested()
            || !self.matches_received_root(&bridge)
            || !proof.matches_admission(&self.admission)
        {
            return Ok(Some(false));
        }
        let receiver = match receiver.try_lock() {
            Ok(receiver) => receiver,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 post-route receiver poisoned"),
        };
        if !receiver.matches_original_route(route) {
            return Ok(Some(false));
        }
        match receiver.try_handoffs_accounted(route) {
            Ok(None) => return Ok(None),
            Ok(Some(false)) => return Ok(Some(false)),
            Ok(Some(true)) => {}
            Err(_) => return Err("P-C V3 post-route receiver accounting failed"),
        }
        let scope = match self.scope_effects.try_lock() {
            Ok(scope) => scope,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 post-route scope poisoned"),
        };
        if scope.claim.is_some() && !scope.finished {
            return Ok(Some(false));
        }
        for actor in &self.actor_effects {
            let actor = match actor.try_lock() {
                Ok(actor) => actor,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                Err(_) => return Err("P-C V3 post-route actor poisoned"),
            };
            if actor.claim.is_some() && !actor.finished {
                return Ok(Some(false));
            }
        }
        Ok(Some(true))
    }

    pub(crate) fn try_seal_aborted_post_route(
        &mut self,
        registration: &RoventNativeCaptureRegistration,
    ) -> Result<Option<bool>, &'static str> {
        match self.try_validate_aborted_post_route(registration)? {
            Some(true) => {
                self.sealed_abort_post_route = true;
                Ok(Some(true))
            }
            other => Ok(other),
        }
    }

    pub(crate) fn try_observe_sealed_aborted_post_route(
        &self,
        registration: &RoventNativeCaptureRegistration,
    ) -> Result<Option<bool>, &'static str> {
        if !self.sealed_abort_post_route {
            return Ok(Some(false));
        }
        self.try_validate_aborted_post_route(registration)
    }

    /// Revoke and remove only the exact ticket-free route whose original
    /// receiver is still held by this recovery. Nothing here proves a native
    /// fence or teacher completion; those remain separate obligations.
    fn try_abort_unstarted_route(
        &mut self,
        registration: &RoventNativeCaptureRegistration,
    ) -> Result<Option<bool>, &'static str> {
        if self.post_route_custody.is_some() {
            return Ok(Some(true));
        }
        if self.post_route_unknown
            || self.post_route_error.is_some()
            || self.panic.is_some()
            || self.owner.is_some()
            || self.pump.is_some()
            || self.pump_setup_unknown
            || self.pump_setup_failure.is_some()
            || self.closed_custody.is_some()
            || self.closed_custody_setup_unknown
            || self.processed.is_some()
            || self.retained.is_some()
            || self.consumer.is_some()
            || self.consumer_setup_unknown
            || self.worker.is_some()
            || self.worker_key.is_some()
            || self.worker_spawn_failure.is_some()
            || self.handoff.is_some()
            || self.prepared.is_some()
            || self.published
        {
            return Ok(Some(false));
        }
        match self.try_observe_later_operations_never_attempted(registration)? {
            Some(true) => {}
            other => return Ok(other),
        }
        let (Some(route), Some(receiver), Some(slot_owner)) = (
            self.route.as_ref().cloned(),
            self.receiver.as_ref().cloned(),
            self.operation_slots[0].as_ref().cloned(),
        ) else {
            return Ok(Some(false));
        };
        {
            let slot = match slot_owner.try_lock() {
                Ok(slot) => slot,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                Err(_) => return Err("P-C V3 route operation slot poisoned"),
            };
            match slot.try_matches_abort_source(&slot_owner, self, registration, 0) {
                Some(true) => {}
                Some(false) => return Ok(Some(false)),
                None => return Ok(None),
            }
            if slot.claim.is_some()
                || slot.route_unknown
                || slot.route_destination.is_some()
                || !matches!(slot.route_result.as_ref(), Some(Ok(Some(()))))
                || !slot.registration.same_original_owner(registration)
                || !Arc::ptr_eq(&slot.identity, &self.operation_identities[0])
                || !registration.owns_speculative_root_v3(&route)
            {
                return Ok(Some(false));
            }
        }
        {
            let scope = match self.scope_effects.try_lock() {
                Ok(scope) => scope,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                Err(_) => return Err("P-C V3 abort scope poisoned"),
            };
            if scope.claim.is_some() && !scope.finished {
                return Ok(None);
            }
        }
        for actor in &self.actor_effects {
            let actor = match actor.try_lock() {
                Ok(actor) => actor,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                Err(_) => return Err("P-C V3 abort actor poisoned"),
            };
            if actor.claim.is_some() && !actor.finished {
                return Ok(None);
            }
        }
        {
            let receiver_guard = match receiver.try_lock() {
                Ok(receiver) => receiver,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                Err(_) => return Err("P-C V3 route receiver poisoned"),
            };
            if !receiver_guard.matches_original_route(&route) {
                return Ok(Some(false));
            }
            match receiver_guard.try_handoffs_accounted(&route) {
                Ok(None) => return Ok(None),
                Ok(Some(false)) => return Ok(Some(false)),
                Ok(Some(true)) => {}
                Err(_) => return Err("P-C V3 route receiver accounting failed"),
            }
        }
        route.cancel_all();
        if !self.abort_targets_revoked {
            match route.try_revoke_all_targets() {
                Ok(Some(())) => self.abort_targets_revoked = true,
                Ok(None) => return Ok(None),
                Err(error) => {
                    self.post_route_error = Some(format!("P-C V3 route revoke failed: {error}"));
                    return Ok(Some(false));
                }
            }
        }
        if self.post_route_result.is_none() {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                registration.try_rollback_unstarted_speculative_root_into_v3(
                    &route,
                    &mut self.post_route_result,
                )
            }));
            match result {
                Ok(Ok(None)) => return Ok(None),
                Ok(Ok(Some(()))) => {}
                Ok(Err(error)) => {
                    self.post_route_error = Some(format!("P-C V3 route rollback failed: {error}"));
                    return Ok(Some(false));
                }
                Err(payload) => {
                    self.post_route_unknown = true;
                    self.panic = Some(payload);
                    return Ok(Some(false));
                }
            }
        }
        match self.post_route_result.take() {
            Some(RoventRouteRollbackV3::RolledBack(witness)) => {
                if !witness.matches_original_route(&route) {
                    self.post_route_result = Some(RoventRouteRollbackV3::RolledBack(witness));
                    self.post_route_unknown = true;
                    return Ok(Some(false));
                }
                self.post_route_custody = Some(PcPostRouteRollbackV3 {
                    route,
                    receiver,
                    witness,
                });
                Ok(Some(true))
            }
            Some(other) => {
                self.post_route_result = Some(other);
                self.post_route_error = Some("P-C V3 route was not unstarted".into());
                Ok(Some(false))
            }
            None => {
                self.post_route_unknown = true;
                Ok(Some(false))
            }
        }
    }

    /// Seal this exact aborted recovery before any stage/issuer was created.
    /// Issuing the original no-scope acknowledgement and returning any
    /// reserved quota remain the bridge ledger's separate responsibilities.
    pub(crate) fn seal_aborted_no_scope(&mut self) -> bool {
        if !self.abort_before_route
            || self.operation_slots[0].is_some()
            || self.stage.is_some()
            || self.original_scope_identity.is_some()
            || self.preflight_failure.is_some()
            || self.stage_claim_failure.is_some()
            || self.claimed_unstarted.is_some()
            || self.unmatched_claim_owner.is_some()
            || self.route.is_some()
            || self.receiver.is_some()
            || self.owner.is_some()
            || self.pump.is_some()
            || self.closed_custody.is_some()
            || self.closed_custody_setup_unknown
            || self.pump_setup_unknown
            || self.pump_setup_failure.is_some()
            || self.processed.is_some()
            || self.retained.is_some()
            || self.consumer.is_some()
            || self.consumer_setup_unknown
            || self.worker.is_some()
            || self.worker_key.is_some()
            || self.worker_spawn_failure.is_some()
            || self.handoff.is_some()
            || self.aggregate.is_some()
            || !self.diagnostics.is_empty()
            || self.prepared.is_some()
            || self.published
            || self.next_actor != 0
            || self.actor_start_failure.is_some()
            || !self.custodian.keys().is_empty()
            || self.panic.is_some()
            || self.sealed_abort_before_route
        {
            return false;
        }
        self.sealed_abort_no_scope = true;
        true
    }

    pub(crate) fn is_sealed_abort_no_scope(&self) -> bool {
        self.sealed_abort_no_scope
    }
    pub(crate) fn has_teacher_reservation(&self) -> bool {
        self.teacher_reserved
    }

    /// Seal only the retained stage/source whose scope claim was never entered.
    /// The bridge still owns the original no-scope ACK and quota checks.
    pub(crate) fn try_seal_aborted_unclaimed_stage(
        &mut self,
    ) -> Result<Option<bool>, &'static str> {
        use super::game_authority_v3::GameStepV3 as G;
        if !self.abort_before_route
            || self.operation_slots[0].is_some()
            || self.panic.is_some()
            || self.route.is_some()
            || self.worker.is_some()
            || self.worker_key.is_some()
            || self.prepared.is_some()
            || self.published
            || self.next_actor != 0
            || !self.custodian.keys().is_empty()
        {
            return Ok(Some(false));
        }
        let Some(stage_owner) = self.stage.as_ref().map(Arc::clone) else {
            return Ok(Some(false));
        };
        let mut stage = match stage_owner.try_lock() {
            Ok(stage) => stage,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 unclaimed stage poisoned"),
        };
        let owner = Arc::clone(&self.scope_effects);
        let mut slots = match owner.try_lock() {
            Ok(slots) => slots,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 unclaimed scope slots poisoned"),
        };
        let PcScopeEffectSlotsV3 {
            teacher_source,
            game_source,
            claim,
            outcome,
            claim_step,
            unclaimed_proof,
            before_source_proof,
            ..
        } = &mut *slots;
        if claim.is_some()
            || outcome.is_some()
            || !matches!(
                *claim_step,
                None | Some(G::Busy | G::Aborted | G::Closed | G::RootCanceled)
            )
        {
            return Ok(Some(false));
        }
        let source = teacher_source.as_ref().or_else(|| {
            game_source
                .as_ref()
                .and_then(|owner| owner.pc_scope_source())
        });
        let Some(source) = source else {
            if game_source.is_some() || unclaimed_proof.is_some() {
                return Ok(Some(false));
            }
            if let Some(proof) = before_source_proof.as_ref() {
                return Ok(Some(proof.matches_original(&stage, &self.admission)));
            }
            return Ok(Some(stage.seal_unclaimed_scope_before_source(
                &self.admission,
                before_source_proof,
            )));
        };
        if before_source_proof.is_some() {
            return Ok(Some(false));
        }
        if let Some(proof) = unclaimed_proof.as_ref() {
            return Ok(Some(proof.matches_original(
                &stage,
                source,
                &self.admission,
            )));
        }
        Ok(Some(stage.seal_unclaimed_scope_after_source(
            source,
            unclaimed_proof,
        )))
    }

    /// Reobserve only a proof already sealed in this original recovery. It
    /// never issues a new proof, scope, ACK, or physical completion evidence.
    pub(crate) fn try_observe_sealed_aborted_unclaimed_stage(
        &self,
    ) -> Result<Option<bool>, &'static str> {
        use super::game_authority_v3::GameStepV3 as G;
        if !self.abort_before_route
            || self.operation_slots[0].is_some()
            || self.panic.is_some()
            || self.route.is_some()
            || self.worker.is_some()
            || self.worker_key.is_some()
            || self.prepared.is_some()
            || self.published
            || self.next_actor != 0
            || !self.custodian.keys().is_empty()
        {
            return Ok(Some(false));
        }
        let Some(stage_owner) = self.stage.as_ref().map(Arc::clone) else {
            return Ok(Some(false));
        };
        let stage = match stage_owner.try_lock() {
            Ok(stage) => stage,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 unclaimed stage poisoned"),
        };
        let slots = match self.scope_effects.try_lock() {
            Ok(slots) => slots,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 unclaimed scope slots poisoned"),
        };
        if slots.claim.is_some()
            || slots.outcome.is_some()
            || !matches!(
                slots.claim_step,
                None | Some(G::Busy | G::Aborted | G::Closed | G::RootCanceled)
            )
        {
            return Ok(Some(false));
        }
        let source = slots.teacher_source.as_ref().or_else(|| {
            slots
                .game_source
                .as_ref()
                .and_then(|owner| owner.pc_scope_source())
        });
        match source {
            Some(source) if slots.before_source_proof.is_none() => {
                Ok(Some(slots.unclaimed_proof.as_ref().is_some_and(|proof| {
                    proof.matches_original(&stage, source, &self.admission)
                })))
            }
            None if slots.game_source.is_none() && slots.unclaimed_proof.is_none() => {
                Ok(Some(slots.before_source_proof.as_ref().is_some_and(
                    |proof| proof.matches_original(&stage, &self.admission),
                )))
            }
            _ => Ok(Some(false)),
        }
    }

    /// Seal only a known pre-route root whose own session physically retired.
    /// The ledger must separately validate this same recovery and aggregate.
    pub(crate) fn try_seal_aborted_pre_route(
        &mut self,
        registration: &RoventNativeCaptureRegistration,
    ) -> Result<Option<bool>, &'static str> {
        if !self.abort_before_route
            || !self.teacher_reserved
            || self.panic.is_some()
            || self.stage_claim_failure.is_some()
            || self.worker_spawn_failure.is_some()
            || self.worker.is_some()
            || self.worker_key.is_some()
            || self.handoff.is_some()
            || self.route.is_some()
            || self.receiver.is_some()
            || self.owner.is_some()
            || self.pump.is_some()
            || self.pump_setup_unknown
            || self.pump_setup_failure.is_some()
            || self.closed_custody.is_some()
            || self.closed_custody_setup_unknown
            || self.processed.is_some()
            || self.retained.is_some()
            || self.consumer.is_some()
            || self.consumer_setup_unknown
            || self.prepared.is_some()
            || self.published
            || !self
                .aggregate
                .as_ref()
                .is_some_and(|ack| ack.matches_admission(&self.admission))
        {
            return Ok(Some(false));
        }
        match self.try_observe_pre_route_never_started(registration)? {
            Some(true) => {}
            other => return Ok(other),
        }
        // Acquired effect claims are separate from physical aggregation. An
        // unknown or uncompleted claim may not be erased by a scope ACK.
        let scope = match self.scope_effects.try_lock() {
            Ok(scope) => scope,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 pre-route scope poisoned"),
        };
        if scope.claim.is_some() && !scope.finished {
            return Ok(Some(false));
        }
        for original in &self.actor_effects {
            let actor = match original.try_lock() {
                Ok(actor) => actor,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                Err(_) => return Err("P-C V3 pre-route actor poisoned"),
            };
            if actor.claim.is_some() && !actor.finished {
                return Ok(Some(false));
            }
        }
        self.sealed_abort_before_route = true;
        Ok(Some(true))
    }
    pub(crate) fn matches_route_pump(
        &self,
        route: &SpeculativeRootV3,
        pump: &Arc<Mutex<TeacherPumpV3>>,
    ) -> bool {
        self.route
            .as_ref()
            .is_some_and(|original| original.same_original_route(route))
            && self
                .pump
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, pump))
    }

    /// Recheck only this recovery's custody against its original route/pump.
    /// This does not attest callback fencing, target closure or worker join.
    pub(crate) fn try_matches_closed_custody(
        &self,
        route: &SpeculativeRootV3,
        pump: &Arc<Mutex<TeacherPumpV3>>,
    ) -> Result<Option<bool>, &'static str> {
        if self.closed_custody_setup_unknown || !self.matches_route_pump(route, pump) {
            return Ok(Some(false));
        }
        let Some(custody) = &self.closed_custody else {
            return Ok(Some(false));
        };
        let pump = match pump.try_lock() {
            Ok(pump) => pump,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 closed custody pump poisoned"),
        };
        Ok(Some(pump.owns_closed_custody(custody)))
    }
    /// This root has no in-flight policy stage. Shared output may still be
    /// held by the game pipeline and is not evidence of durable storage.
    pub(crate) fn consumer_route_retirement(&self) -> PcConsumerRetirementV3 {
        if self.consumer_setup_unknown || self.consumer.is_none() {
            return PcConsumerRetirementV3::Unknown;
        }
        match self.consumer_phase {
            ConsumerPhaseV3::Receive => PcConsumerRetirementV3::Ready,
            ConsumerPhaseV3::Rejected => PcConsumerRetirementV3::Rejected,
            ConsumerPhaseV3::Unknown => PcConsumerRetirementV3::Unknown,
            _ => PcConsumerRetirementV3::Pending,
        }
    }
    pub(crate) fn matches_prepared(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
        route: &SpeculativeRootV3,
        stage: &Arc<Mutex<TeacherSessionStagingV3>>,
        pump: &Arc<Mutex<TeacherPumpV3>>,
        owner: &RoventCollectionOwnerLockV3,
        worker: &RoventPcCollectionWorkerV3,
    ) -> bool {
        self.matches_admission(admission)
            && self
                .route
                .as_ref()
                .is_some_and(|original| original.same_original_route(route))
            && self
                .stage
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, stage))
            && self
                .pump
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, pump))
            && self
                .owner
                .as_ref()
                .is_some_and(|original| original.same_original_owner(owner))
            && self
                .worker_key
                .as_ref()
                .is_some_and(|original| original.same_original(&worker.recovery_key()))
    }
    pub(crate) fn definitely_unstarted(&self) -> bool {
        self.teacher_reserved
            && self.stage.is_none()
            && self.route.is_none()
            && self.pump.is_none()
            && self.worker.is_none()
            && self.worker_key.is_none()
            && self.prepared.is_none()
            && !self.published
            && self.next_actor == 0
            && self.panic.is_none()
            && self.worker_spawn_failure.is_none()
            && self.receiver.is_none()
            && self.owner.is_none()
            && self.pump_setup_failure.is_none()
            && !self.pump_setup_unknown
            && self.processed.is_none()
            && self.retained.is_none()
            && self.consumer.is_none()
            && !self.consumer_setup_unknown
            && self.actor_start_failure.is_none()
            && self.custodian.keys().is_empty()
    }
    pub(crate) fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    /// A later teacher drain failure is retained separately from the root's
    /// first failure and is not proof of worker or physical completion.
    pub(crate) fn teacher_drain_failure(&self) -> Option<&str> {
        self.teacher_drain_failure.as_deref()
    }
    /// Irreversibly close preparation before issuing the root's no-scope seal.
    /// This records no physical or worker completion by itself.
    pub(crate) fn seal_definitely_unstarted(&mut self) -> Result<Option<bool>, &'static str> {
        if self.failure.is_none() || !self.definitely_unstarted() {
            return Ok(Some(false));
        }
        match self.try_observe_teacher_claims_settled()? {
            None => return Ok(None),
            Some(false) => return Ok(Some(false)),
            Some(true) => {}
        }
        self.sealed_unstarted = true;
        Ok(Some(true))
    }
    pub(crate) fn is_sealed_unstarted(&self) -> bool {
        self.sealed_unstarted
    }
    pub(crate) fn no_scope_recovery(&self) -> NoScopeRecoveryV3 {
        self.no_scope_recovery
    }
}

#[cfg(test)]
mod teacher_reservation_tests {
    use super::*;

    #[test]
    fn contended_reservation_retries_the_same_root_without_marking_failure() {
        // Exercise an actual try-lock contention and the production outcome
        // handler. The test ledger models a single original root; it is not a
        // native collection admission or physical worker proof.
        let root = Arc::new(());
        let ledger = Mutex::new((Arc::clone(&root), false));
        let reserve = |requested: &Arc<()>| {
            let mut entry = match ledger.try_lock() {
                Ok(entry) => entry,
                Err(std::sync::TryLockError::WouldBlock) => {
                    return Err(PcQuotaEvidenceErrorV3::Busy);
                }
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Err(PcQuotaEvidenceErrorV3::Poisoned);
                }
            };
            if !Arc::ptr_eq(requested, &entry.0) {
                return Err(PcQuotaEvidenceErrorV3::WrongRoot);
            }
            if entry.1 {
                return Ok(PcQuotaOutcomeV3::AlreadyReserved);
            }
            entry.1 = true;
            Ok(PcQuotaOutcomeV3::Reserved)
        };
        let mut reserved = false;
        let mut failure = None;
        let held = ledger.lock().unwrap();
        assert!(matches!(
            apply_teacher_reservation_v3(reserve(&root), &mut reserved, &mut failure),
            Some(PcPrepareStepV3::Pending)
        ));
        assert!(!reserved && failure.is_none() && !held.1);
        drop(held);
        assert!(
            apply_teacher_reservation_v3(reserve(&root), &mut reserved, &mut failure).is_none()
        );
        assert!(reserved && failure.is_none() && ledger.lock().unwrap().1);
    }

    #[test]
    fn reservation_ownership_and_poison_errors_remain_failures() {
        for error in [
            PcQuotaEvidenceErrorV3::Poisoned,
            PcQuotaEvidenceErrorV3::WrongRoot,
            PcQuotaEvidenceErrorV3::AlreadyRetired,
            PcQuotaEvidenceErrorV3::MissingTeacherAggregate,
        ] {
            let mut reserved = false;
            let mut failure = None;
            assert!(matches!(
                apply_teacher_reservation_v3(Err(error), &mut reserved, &mut failure),
                Some(PcPrepareStepV3::Failed)
            ));
            assert!(!reserved && failure.is_some());
        }
    }
}

pub(crate) enum PcPrepareStepV3 {
    Pending,
    Published,
    Failed,
}

// Keep both the ownership failure and the reservation bit unchanged while a
// contended ledger is retried for the same received root.
fn apply_teacher_reservation_v3(
    result: Result<PcQuotaOutcomeV3, PcQuotaEvidenceErrorV3>,
    reserved: &mut bool,
    failure: &mut Option<String>,
) -> Option<PcPrepareStepV3> {
    match result {
        Ok(PcQuotaOutcomeV3::Reserved) => {
            *reserved = true;
            None
        }
        Ok(PcQuotaOutcomeV3::Busy | PcQuotaOutcomeV3::Exhausted)
        | Err(PcQuotaEvidenceErrorV3::Busy) => Some(PcPrepareStepV3::Pending),
        _ => {
            *failure = Some("teacher quota is not owned by this recovery".into());
            Some(PcPrepareStepV3::Failed)
        }
    }
}

#[derive(Clone, Copy)]
enum ConsumerPhaseV3 {
    Receive,
    Seal,
    TransferRow,
    Prepare,
    Candidate,
    Capability,
    Append,
    RetainAppend,
    RetainCompletion,
    Rejected,
    Unknown,
}

pub(crate) enum PcConsumerRetirementV3 {
    Ready,
    Pending,
    Rejected,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NoScopeRecoveryV3 {
    Pending,
    Complete,
    Unknown,
}

struct ClaimedUnstartedV3 {
    capability: RoventTeacherScopeCapabilityV3,
    physical: Option<RoventTeacherScopePhysicalAckV3>,
    workers: Option<RoventTeacherScopeWorkerAckV3>,
    failure: Option<RoventTeacherScopeAggregateFailureV3>,
    aggregate: Option<RoventTeacherScopeAggregateAckV3>,
    closed: bool,
    unknown: bool,
    scope_error: Option<AdaptiveCollectionError>,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ClaimedUnstartedPollV3 {
    Pending,
    Confirmed,
    Ineligible,
    Unknown,
}

impl PcRootRecoveryV3 {
    /// Same-root, same-issuer no-child/no-thread evidence; not a no-scope
    /// seal, native completion, corpus acceptance or a teacher result.
    pub(crate) fn claimed_unstarted_aggregate(&self) -> Option<&RoventTeacherScopeAggregateAckV3> {
        self.claimed_unstarted.as_ref()?.aggregate.as_ref()
    }

    pub(crate) fn seal_claimed_unstarted(&mut self) -> Result<Option<bool>, &'static str> {
        match self.try_observe_teacher_claims_settled()? {
            None => return Ok(None),
            Some(false) => return Ok(Some(false)),
            Some(true) => {}
        }
        if !self
            .claimed_unstarted
            .as_ref()
            .is_some_and(|owner| owner.aggregate.is_some())
        {
            return Ok(Some(false));
        }
        match self.try_claimed_unstarted() {
            ClaimedUnstartedPollV3::Confirmed => {}
            ClaimedUnstartedPollV3::Pending => return Ok(None),
            ClaimedUnstartedPollV3::Ineligible => return Ok(Some(false)),
            ClaimedUnstartedPollV3::Unknown => {
                return Err("P-C V3 claimed teacher scope completion unknown");
            }
        }
        self.sealed_claimed_unstarted = true;
        Ok(Some(true))
    }

    pub(crate) fn is_sealed_claimed_unstarted(&self) -> bool {
        self.sealed_claimed_unstarted
    }

    pub(crate) fn try_claimed_unstarted(&mut self) -> ClaimedUnstartedPollV3 {
        use rz_search::external_cpu::process_scope::{
            CollectionMemberPhysicalState as P, CollectionMemberWorkerState as W,
        };
        match self.try_observe_teacher_claims_settled() {
            Ok(Some(true)) => {}
            Ok(None | Some(false)) => return ClaimedUnstartedPollV3::Pending,
            Err(error) => {
                self.teacher_claim_observation_error.get_or_insert(error);
                return ClaimedUnstartedPollV3::Unknown;
            }
        }
        if self.failure.is_none()
            || !self.teacher_reserved
            || self.next_actor != 0
            || self.actor_start_failure.is_some()
            || self.preflight_failure.is_some()
            || !self.custodian.keys().is_empty()
            || self.route.is_some()
            || self.receiver.is_some()
            || self.owner.is_some()
            || self.pump.is_some()
            || self.pump_setup_failure.is_some()
            || self.pump_setup_unknown
            || self.processed.is_some()
            || self.retained.is_some()
            || self.consumer.is_some()
            || self.consumer_setup_unknown
            || self.worker.is_some()
            || self.worker_spawn_failure.is_some()
            || self.worker_key.is_some()
            || self.handoff.is_some()
            || self.aggregate.is_some()
            || !self.diagnostics.is_empty()
            || self.prepared.is_some()
            || self.published
            || self.panic.is_some()
            || self.sealed_unstarted
            || self.unmatched_claim_owner.is_some()
        {
            return ClaimedUnstartedPollV3::Ineligible;
        }
        if self.claimed_unstarted.is_none() {
            let Some(stage_owner) = self.stage.as_ref().map(Arc::clone) else {
                return ClaimedUnstartedPollV3::Ineligible;
            };
            let mut stage = match stage_owner.try_lock() {
                Ok(stage) => stage,
                Err(std::sync::TryLockError::WouldBlock) => return ClaimedUnstartedPollV3::Pending,
                Err(_) => return ClaimedUnstartedPollV3::Unknown,
            };
            if stage.state() != TeacherSessionStageStateV3::AfterClaimRejected {
                return ClaimedUnstartedPollV3::Ineligible;
            }
            let capability = match stage.take_rejected_owner() {
                Some(TeacherPrepareOwnerV3::Claimed(capability)) => capability,
                Some(original) => {
                    self.unmatched_claim_owner = Some(original);
                    return ClaimedUnstartedPollV3::Unknown;
                }
                None => return ClaimedUnstartedPollV3::Unknown,
            };
            self.claimed_unstarted = Some(ClaimedUnstartedV3 {
                capability,
                physical: None,
                workers: None,
                failure: None,
                aggregate: None,
                closed: false,
                unknown: false,
                scope_error: None,
                panic: None,
            });
        }
        let recovery = self
            .claimed_unstarted
            .as_mut()
            .expect("original claimed scope retained");
        if recovery.unknown {
            return ClaimedUnstartedPollV3::Unknown;
        }
        if recovery.aggregate.is_some() {
            return ClaimedUnstartedPollV3::Confirmed;
        }
        if !recovery.closed {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                recovery.capability.close()
            }));
            match result {
                Ok(Ok(())) => recovery.closed = true,
                Ok(Err(error)) if super::teacher::v3_session::is_teacher_scope_busy(&error) => {
                    return ClaimedUnstartedPollV3::Pending;
                }
                Ok(Err(error)) => {
                    if recovery.scope_error.is_none() {
                        recovery.scope_error = Some(error);
                    }
                    recovery.unknown = true;
                    return ClaimedUnstartedPollV3::Unknown;
                }
                Err(payload) => {
                    recovery.panic = Some(payload);
                    recovery.unknown = true;
                    return ClaimedUnstartedPollV3::Unknown;
                }
            }
        }
        if recovery.physical.is_none() {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                recovery.capability.try_physical_retirement()
            }));
            match result {
                Ok(Ok(Some(ack))) => recovery.physical = Some(ack),
                Ok(Ok(None)) => return ClaimedUnstartedPollV3::Pending,
                Ok(Err(error)) if super::teacher::v3_session::is_teacher_scope_busy(&error) => {
                    return ClaimedUnstartedPollV3::Pending;
                }
                Ok(Err(error)) => {
                    if recovery.scope_error.is_none() {
                        recovery.scope_error = Some(error);
                    }
                    recovery.unknown = true;
                    return ClaimedUnstartedPollV3::Unknown;
                }
                Err(payload) => {
                    recovery.panic = Some(payload);
                    recovery.unknown = true;
                    return ClaimedUnstartedPollV3::Unknown;
                }
            }
        }
        if recovery.workers.is_none() {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                recovery.capability.try_worker_retirement()
            }));
            match result {
                Ok(Ok(Some(ack))) => recovery.workers = Some(ack),
                Ok(Ok(None)) => return ClaimedUnstartedPollV3::Pending,
                Ok(Err(error)) if super::teacher::v3_session::is_teacher_scope_busy(&error) => {
                    return ClaimedUnstartedPollV3::Pending;
                }
                Ok(Err(error)) => {
                    if recovery.scope_error.is_none() {
                        recovery.scope_error = Some(error);
                    }
                    recovery.unknown = true;
                    return ClaimedUnstartedPollV3::Unknown;
                }
                Err(payload) => {
                    recovery.panic = Some(payload);
                    recovery.unknown = true;
                    return ClaimedUnstartedPollV3::Unknown;
                }
            }
        }
        let physical = recovery
            .physical
            .as_ref()
            .expect("physical evidence retained");
        let workers = recovery.workers.as_ref().expect("worker evidence retained");
        let valid = physical.matches_admission(&self.admission)
            && workers.matches_admission(&self.admission)
            && physical.scope_outcome().is_some_and(|p| {
                let w = workers.outcome();
                p.same_issuer_as_worker(w)
                    && !p.members().is_empty()
                    && p.members().len() == w.members().len()
                    && p.members()
                        .iter()
                        .zip(w.members())
                        .all(|((pr, ps), (wr, ws))| {
                            pr == wr && *ps == P::NeverIssuedNoChild && *ws == W::NoThread
                        })
            });
        if !valid {
            recovery.unknown = true;
            return ClaimedUnstartedPollV3::Unknown;
        }
        let physical = recovery
            .physical
            .take()
            .expect("physical evidence retained");
        let workers = recovery.workers.take().expect("worker evidence retained");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            RoventTeacherScopeAggregateAckV3::try_build(physical, Some(workers), Vec::new())
        }));
        match result {
            Ok(Ok(aggregate)) => {
                recovery.aggregate = Some(aggregate);
                ClaimedUnstartedPollV3::Confirmed
            }
            Ok(Err(failure)) => {
                recovery.failure = Some(failure);
                recovery.unknown = true;
                ClaimedUnstartedPollV3::Unknown
            }
            Err(payload) => {
                recovery.panic = Some(payload);
                recovery.unknown = true;
                ClaimedUnstartedPollV3::Unknown
            }
        }
    }
}

/// Bounded supervisor; each original root remains owned even if a different
/// root fails or needs additional recovery. Caller sets finite deadlines.
pub(crate) struct PcWorkerSupervisorV3 {
    original_self: OnceLock<Weak<Mutex<PcWorkerSupervisorV3>>>,
    bridge: Arc<RoventPcCollectionBridgeV3>,
    router: Arc<CaptureRouterV3>,
    roots: Vec<Arc<Mutex<PcRootRecoveryV3>>>,
    first_root_error: Option<String>,
    pending_root: Option<PcReceivedRootV3>,
    pipeline: Option<PolicyRowPipelineOwnerV3>,
    managed_output: Option<Arc<ManagedOutputOwnerV3>>,
    output_installed: bool,
    driver_sealed: bool,
    startup_deadline: Instant,
    cleanup_deadline: Instant,
}

/// Actual same-channel receive, same-supervisor membership, bridge slot and
/// original stage identity. This is not a teacher or physical completion.
/// It is deliberately nonclone and must remain with its caller on Busy.
pub(super) struct PcReceivedStageProofV3 {
    supervisor: Weak<Mutex<PcWorkerSupervisorV3>>,
    bridge: Weak<RoventPcCollectionBridgeV3>,
    recovery: Weak<Mutex<PcRootRecoveryV3>>,
    stage: Weak<Mutex<TeacherSessionStagingV3>>,
    scope_identity: Weak<()>,
    admission: RoventCollectionRootAdmissionV3,
}

impl PcReceivedStageProofV3 {
    pub(super) fn matches_game(
        &self,
        game: &Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>,
    ) -> Option<bool> {
        self.bridge
            .upgrade()
            .map(|bridge| Arc::ptr_eq(bridge.game_authority(), game))
    }
    pub(super) fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }
    pub(super) fn matches_recovery(&self, recovery: &Arc<Mutex<PcRootRecoveryV3>>) -> Option<bool> {
        self.recovery
            .upgrade()
            .map(|original| Arc::ptr_eq(&original, recovery))
    }
    pub(super) fn stage(&self) -> Option<Arc<Mutex<TeacherSessionStagingV3>>> {
        self.stage.upgrade()
    }
    pub(super) fn matches_scope_source(
        &self,
        source: &super::teacher::v3_session::TeacherScopeEffectSourceV3,
    ) -> bool {
        Weak::ptr_eq(&self.scope_identity, &source.identity_weak())
            && source.matches_admission(&self.admission)
    }

    /// The caller holds the exact supervisor, recovery and stage guards, and
    /// supplies the same stage Arc from which its guard was obtained.
    pub(super) fn try_from_guards(
        supervisor: &PcWorkerSupervisorV3,
        recovery: &PcRootRecoveryV3,
        stage_owner: &Arc<Mutex<TeacherSessionStagingV3>>,
        stage: &TeacherSessionStagingV3,
    ) -> Result<Option<Self>, &'static str> {
        let Some(supervisor_identity) = supervisor.original_self.get() else {
            return Err("P-C V3 supervisor identity is not bound");
        };
        if !recovery
            .stage
            .as_ref()
            .is_some_and(|original| Arc::ptr_eq(original, stage_owner))
        {
            return Err("P-C V3 stage owner differs");
        }
        let Some(scope_identity) = &recovery.original_scope_identity else {
            return Err("P-C V3 original scope identity absent");
        };
        if !Weak::ptr_eq(scope_identity, &stage.scope_effect_identity_weak()) {
            return Err("P-C V3 guarded stage identity differs");
        }
        let proof = Self {
            supervisor: supervisor_identity.clone(),
            bridge: Arc::downgrade(&supervisor.bridge),
            recovery: recovery.original_self.clone(),
            stage: Arc::downgrade(stage_owner),
            scope_identity: scope_identity.clone(),
            admission: recovery.admission.clone(),
        };
        match proof.try_revalidate_with_guards(supervisor, recovery)? {
            None => Ok(None),
            Some(true) => Ok(Some(proof)),
            Some(false) => Err("P-C V3 original guarded stage differs"),
        }
    }

    /// Revalidate while the caller already holds both original guards. The
    /// two Weak identities were sealed at construction/binding; no guard is
    /// reacquired and no raw address or copied identifier is authority.
    pub(super) fn try_revalidate_with_guards(
        &self,
        supervisor: &PcWorkerSupervisorV3,
        recovery: &PcRootRecoveryV3,
    ) -> Result<Option<bool>, &'static str> {
        let Some(supervisor_identity) = supervisor.original_self.get() else {
            return Err("P-C V3 supervisor identity is not bound");
        };
        if !Weak::ptr_eq(supervisor_identity, &self.supervisor)
            || !Weak::ptr_eq(&recovery.original_self, &self.recovery)
        {
            return Ok(Some(false));
        }
        let bridge = self
            .bridge
            .upgrade()
            .ok_or("P-C V3 bridge owner unavailable")?;
        let recovery_owner = self
            .recovery
            .upgrade()
            .ok_or("P-C V3 recovery owner unavailable")?;
        let stage = self
            .stage
            .upgrade()
            .ok_or("P-C V3 stage owner unavailable")?;
        if !supervisor.matches_original_bridge(&bridge)
            || !supervisor.owns_recovery(&recovery_owner)
            || !recovery.matches_admission(&self.admission)
            || !recovery.matches_received_root(&bridge)
            || !recovery
                .stage
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, &stage))
            || !recovery
                .original_scope_identity
                .as_ref()
                .is_some_and(|original| Weak::ptr_eq(original, &self.scope_identity))
        {
            return Ok(Some(false));
        }
        bridge.try_matches_registered_recovery(&self.admission, &recovery_owner)
    }

    /// Reobserve the original owners before issuing an effect. This never
    /// locks the supervisor across a blocking call or fabricates a receipt.
    pub(super) fn try_revalidate(&self) -> Result<Option<bool>, &'static str> {
        let supervisor_owner = self
            .supervisor
            .upgrade()
            .ok_or("P-C V3 supervisor owner unavailable")?;
        let bridge = self
            .bridge
            .upgrade()
            .ok_or("P-C V3 bridge owner unavailable")?;
        let recovery_owner = self
            .recovery
            .upgrade()
            .ok_or("P-C V3 recovery owner unavailable")?;
        let stage = self
            .stage
            .upgrade()
            .ok_or("P-C V3 stage owner unavailable")?;
        let supervisor = match supervisor_owner.try_lock() {
            Ok(owner) => owner,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 supervisor poisoned"),
        };
        if !supervisor.matches_original_bridge(&bridge)
            || !supervisor.owns_recovery(&recovery_owner)
        {
            return Ok(Some(false));
        }
        let recovery = match recovery_owner.try_lock() {
            Ok(owner) => owner,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 recovery poisoned"),
        };
        if !recovery.matches_admission(&self.admission)
            || !recovery.matches_received_root(&bridge)
            || !recovery
                .stage
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, &stage))
            || !recovery
                .original_scope_identity
                .as_ref()
                .is_some_and(|original| Weak::ptr_eq(original, &self.scope_identity))
        {
            return Ok(Some(false));
        }
        drop(recovery);
        bridge.try_matches_registered_recovery(&self.admission, &recovery_owner)
    }
}

/// Same received root, original ready session and exact actor slot. A source
/// identity is not an actor start, a terminal result or a physical fence.
pub(super) struct PcReceivedActorProofV3 {
    stage_proof: PcReceivedStageProofV3,
    session: Weak<Mutex<RoventCollectionTeacherSessionV3>>,
    session_identity: Weak<()>,
    actor_identity: Weak<()>,
}

impl PcReceivedActorProofV3 {
    pub(super) fn matches_game(
        &self,
        game: &Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>,
    ) -> Option<bool> {
        self.stage_proof.matches_game(game)
    }
    pub(super) fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.stage_proof.matches_admission(admission)
    }
    pub(super) fn matches_actor_source(
        &self,
        source: &super::teacher::v3_session::TeacherActorEffectSourceV3,
    ) -> bool {
        Weak::ptr_eq(&self.session_identity, &source.session_identity_weak())
            && Weak::ptr_eq(
                &self.actor_identity,
                &Arc::downgrade(source.actor_identity()),
            )
            && source.matches_admission(&self.stage_proof.admission)
    }
    pub(super) fn try_from_guards(
        supervisor: &PcWorkerSupervisorV3,
        recovery: &PcRootRecoveryV3,
        stage_owner: &Arc<Mutex<TeacherSessionStagingV3>>,
        stage: &TeacherSessionStagingV3,
        session_owner: &Arc<Mutex<RoventCollectionTeacherSessionV3>>,
        session: &RoventCollectionTeacherSessionV3,
        source: &super::teacher::v3_session::TeacherActorEffectSourceV3,
    ) -> Result<Option<Self>, &'static str> {
        let Some(stage_proof) =
            PcReceivedStageProofV3::try_from_guards(supervisor, recovery, stage_owner, stage)?
        else {
            return Ok(None);
        };
        let Some(original_session) = stage.session() else {
            return Ok(None);
        };
        if !Arc::ptr_eq(&original_session, session_owner)
            || !stage
                .ready_session_effect_identity_weak()
                .is_some_and(|id| Weak::ptr_eq(&id, &source.session_identity_weak()))
            || !session.matches_actor_effect_source(source)
        {
            return Err("P-C V3 original actor session differs");
        }
        Ok(Some(Self {
            stage_proof,
            session: Arc::downgrade(session_owner),
            session_identity: source.session_identity_weak(),
            actor_identity: Arc::downgrade(source.actor_identity()),
        }))
    }
    pub(super) fn try_revalidate_with_guards(
        &self,
        supervisor: &PcWorkerSupervisorV3,
        recovery: &PcRootRecoveryV3,
        stage: &TeacherSessionStagingV3,
        session: &RoventCollectionTeacherSessionV3,
        source: &super::teacher::v3_session::TeacherActorEffectSourceV3,
    ) -> Result<Option<bool>, &'static str> {
        match self
            .stage_proof
            .try_revalidate_with_guards(supervisor, recovery)?
        {
            Some(true) => {}
            other => return Ok(other),
        }
        let original_session = self
            .session
            .upgrade()
            .ok_or("P-C V3 actor session owner unavailable")?;
        if !Weak::ptr_eq(
            &self.stage_proof.scope_identity,
            &stage.scope_effect_identity_weak(),
        ) || !stage
            .session()
            .is_some_and(|original| Arc::ptr_eq(&original, &original_session))
            || !stage
                .ready_session_effect_identity_weak()
                .is_some_and(|id| Weak::ptr_eq(&id, &self.session_identity))
            || !self.matches_actor_source(source)
            || !session.matches_actor_effect_source(source)
        {
            return Ok(Some(false));
        }
        Ok(Some(true))
    }
}

/// A registered supervisor task is retained before any thread can start.
/// Join of this driver is independent of teacher/native/GPU completion.
pub(crate) struct SupervisorStartCustodyV3 {
    supervisor: Arc<Mutex<PcWorkerSupervisorV3>>,
    native: Option<Arc<rz_uci::rovent_bootstrap::launch::RoventPcCollectionLaunchOwnerV3>>,
    stop: Arc<AtomicBool>,
    admitted: Arc<AtomicBool>,
    stop_authorization: Option<super::pc_hooks_lock_v3::PcSupervisorStopAuthorizationV3>,
    until: Instant,
    state: SupervisorThreadStateV3,
    thread: Option<std::thread::JoinHandle<Result<(), String>>>,
    result: Option<Result<(), String>>,
    first_error: Option<String>,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SupervisorThreadStateV3 {
    Prepared,
    StartingUnknown,
    Running,
    NoWorker,
    Joined,
    JoinPanicked,
}

/// Nonblocking observation of the same original driver. Pending means only
/// lock contention; Ready(None/false) is a completed negative observation.
pub(crate) enum PcDriverPollV3<T> {
    Pending,
    Ready(T),
}

#[derive(Debug)]
pub(crate) enum PcDriverEvidenceErrorV3 {
    Poisoned(&'static str),
    Invalid(&'static str),
}

macro_rules! driver_evidence_lock {
    ($mutex:expr, $name:literal) => {
        match $mutex.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(PcDriverPollV3::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(PcDriverEvidenceErrorV3::Poisoned($name));
            }
        }
    };
}

/// Evidence only for this driver's joined thread, not GPU or output success.
pub(crate) struct PcSupervisorJoinedProofV3 {
    custody: Arc<Mutex<SupervisorStartCustodyV3>>,
    supervisor: Arc<Mutex<PcWorkerSupervisorV3>>,
    native: Arc<rz_uci::rovent_bootstrap::launch::RoventPcCollectionLaunchOwnerV3>,
    panicked: bool,
}

/// Driver execution status only; this is not a game, output, or GPU result.
/// A panic payload remains in the original custody retained by the proof.
#[derive(Debug)]
pub(crate) enum PcSupervisorDriverObservationV3 {
    Completed,
    Failed {
        first_error: Option<String>,
        run_error: Option<String>,
        root_error: Option<String>,
        root_error_unknown: bool,
        root_error_pending: bool,
    },
    Panicked {
        first_error: Option<String>,
    },
    /// A root could not be inspected; this is not evidence of failure or success.
    Unknown,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PcRootFailureObservationV3 {
    NoFailureObserved,
    Failed {
        error: String,
        unknown: bool,
        pending: bool,
    },
    Unknown,
}

fn observe_root_failures<T>(
    roots: &[Arc<Mutex<T>>],
    first_error: Option<&str>,
    failure: impl for<'a> Fn(&'a T) -> Option<&'a str>,
) -> PcDriverPollV3<PcRootFailureObservationV3> {
    let mut unknown = false;
    let mut busy = false;
    let mut first_error = first_error.map(str::to_owned);
    for original in roots {
        match original.try_lock() {
            Ok(root) => {
                if first_error.is_none() {
                    first_error = failure(&root).map(str::to_owned);
                }
            }
            Err(std::sync::TryLockError::WouldBlock) => busy = true,
            Err(std::sync::TryLockError::Poisoned(_)) => unknown = true,
        }
    }
    if let Some(error) = first_error {
        PcDriverPollV3::Ready(PcRootFailureObservationV3::Failed {
            error,
            unknown,
            pending: busy,
        })
    } else if unknown {
        PcDriverPollV3::Ready(PcRootFailureObservationV3::Unknown)
    } else if busy {
        PcDriverPollV3::Pending
    } else {
        PcDriverPollV3::Ready(PcRootFailureObservationV3::NoFailureObserved)
    }
}

fn classify_joined_driver(
    first_error: Option<&String>,
    result: &Result<(), String>,
    root: PcRootFailureObservationV3,
) -> PcSupervisorDriverObservationV3 {
    let (root_error, root_error_unknown, root_error_pending) = match root {
        PcRootFailureObservationV3::Failed {
            error,
            unknown,
            pending,
        } => (Some(error), unknown, pending),
        PcRootFailureObservationV3::NoFailureObserved => (None, false, false),
        PcRootFailureObservationV3::Unknown => (None, true, false),
    };
    match (
        first_error,
        result,
        root_error,
        root_error_unknown,
        root_error_pending,
    ) {
        (None, Ok(()), None, false, false) => PcSupervisorDriverObservationV3::Completed,
        (None, Ok(()), None, true, false) => PcSupervisorDriverObservationV3::Unknown,
        (None, Ok(()), None, _, true) => PcSupervisorDriverObservationV3::Unknown,
        (first_error, result, root_error, root_error_unknown, root_error_pending) => {
            PcSupervisorDriverObservationV3::Failed {
                first_error: first_error.cloned(),
                run_error: result.as_ref().err().cloned(),
                root_error,
                root_error_unknown,
                root_error_pending,
            }
        }
    }
}

#[cfg(test)]
mod root_failure_observation_tests {
    use super::*;

    #[test]
    fn first_root_and_secondary_drain_observations_preserve_unknown_and_pending() {
        struct RootEvidence {
            root: Option<String>,
            drain: Option<String>,
        }
        let known = Arc::new(Mutex::new(RootEvidence {
            root: Some("first root failure".into()),
            drain: Some("later drain failure".into()),
        }));
        let busy = Arc::new(Mutex::new(RootEvidence {
            root: None,
            drain: None,
        }));
        let held = busy.lock().unwrap();
        let poisoned = Arc::new(Mutex::new(RootEvidence {
            root: None,
            drain: None,
        }));
        let worker = Arc::clone(&poisoned);
        assert!(
            std::thread::spawn(move || {
                let _guard = worker.lock().unwrap();
                panic!("intentional fixture poison");
            })
            .join()
            .is_err()
        );
        let roots = [known, Arc::clone(&busy), poisoned];
        let root = observe_root_failures(&roots, None, |r| r.root.as_deref());
        let drain = observe_root_failures(&roots, None, |r| r.drain.as_deref());
        assert!(
            matches!(root, PcDriverPollV3::Ready(PcRootFailureObservationV3::Failed {
            error, unknown: true, pending: true,
        }) if error == "first root failure")
        );
        assert!(
            matches!(drain, PcDriverPollV3::Ready(PcRootFailureObservationV3::Failed {
            error, unknown: true, pending: true,
        }) if error == "later drain failure")
        );
        drop(held);
    }

    #[test]
    fn joined_driver_keeps_original_root_and_run_errors_distinct() {
        let startup = "startup error".to_owned();
        let result = Err("driver error".to_owned());
        let observed = classify_joined_driver(
            Some(&startup),
            &result,
            PcRootFailureObservationV3::Failed {
                error: "teacher drain failed".into(),
                unknown: false,
                pending: false,
            },
        );
        assert!(matches!(observed, PcSupervisorDriverObservationV3::Failed {
            first_error: Some(first), run_error: Some(run), root_error: Some(root), root_error_unknown: false, root_error_pending: false,
        } if first == "startup error" && run == "driver error" && root == "teacher drain failed"));
    }

    #[test]
    fn unknown_root_is_not_success_or_failure_but_does_not_hide_known_driver_failure() {
        assert!(matches!(
            classify_joined_driver(None, &Ok(()), PcRootFailureObservationV3::Unknown),
            PcSupervisorDriverObservationV3::Unknown
        ));
        assert!(matches!(
            classify_joined_driver(
                None,
                &Ok(()),
                PcRootFailureObservationV3::Failed {
                    error: "root".into(),
                    unknown: false,
                    pending: false
                }
            ),
            PcSupervisorDriverObservationV3::Failed {
                root_error: Some(_),
                ..
            }
        ));
        assert!(matches!(
            classify_joined_driver(
                None,
                &Err("driver".into()),
                PcRootFailureObservationV3::Unknown
            ),
            PcSupervisorDriverObservationV3::Failed {
                run_error: Some(_),
                root_error_unknown: true,
                ..
            }
        ));
    }

    #[test]
    fn root_snapshot_keeps_busy_and_poison_distinct_from_recorded_failure() {
        let root = Arc::new(Mutex::new(None::<String>));
        let held = root.lock().unwrap();
        assert!(matches!(
            observe_root_failures(&[Arc::clone(&root)], None, |r| r.as_deref()),
            PcDriverPollV3::Pending
        ));
        drop(held);
        *root.lock().unwrap() = Some("original drain error".into());
        assert!(
            matches!(observe_root_failures(&[Arc::clone(&root)], None, |r| r.as_deref()), PcDriverPollV3::Ready(PcRootFailureObservationV3::Failed { error, unknown: false, pending: false }) if error == "original drain error")
        );
        let poison = Arc::new(Mutex::new(None::<String>));
        let poison_worker = Arc::clone(&poison);
        assert!(
            std::thread::spawn(move || {
                let _guard = poison_worker.lock().unwrap();
                panic!("poison");
            })
            .join()
            .is_err()
        );
        assert!(matches!(
            observe_root_failures(&[poison], None, |r| r.as_deref()),
            PcDriverPollV3::Ready(PcRootFailureObservationV3::Unknown)
        ));
    }

    #[test]
    fn retained_first_error_preserves_other_roots_unknown_and_busy() {
        let failed = Arc::new(Mutex::new(Some("later failure".to_owned())));
        let poisoned = Arc::new(Mutex::new(None::<String>));
        let worker = Arc::clone(&poisoned);
        assert!(
            std::thread::spawn(move || {
                let _guard = worker.lock().unwrap();
                panic!("poison");
            })
            .join()
            .is_err()
        );
        let busy = Arc::new(Mutex::new(None::<String>));
        let held = busy.lock().unwrap();
        let observed = observe_root_failures(
            &[failed, poisoned, Arc::clone(&busy)],
            Some("first failure"),
            |r| r.as_deref(),
        );
        let PcDriverPollV3::Ready(PcRootFailureObservationV3::Failed {
            error,
            unknown,
            pending,
        }) = observed
        else {
            panic!("known root error must remain visible")
        };
        assert_eq!(error, "first failure");
        assert!(unknown && pending);
        assert!(
            matches!(classify_joined_driver(None, &Ok(()), PcRootFailureObservationV3::Failed { error, unknown, pending }),
            PcSupervisorDriverObservationV3::Failed { root_error: Some(root), root_error_unknown: true, root_error_pending: true, .. } if root == "first failure")
        );
        drop(held);
    }
}

impl PcSupervisorJoinedProofV3 {
    pub(crate) fn panicked(&self) -> bool {
        self.panicked
    }

    /// Read the same joined driver's actual outcome without waiting. A join
    /// alone, or the absence of a panic, does not establish driver success.
    pub(crate) fn try_driver_observation(
        &self,
    ) -> Result<PcDriverPollV3<PcSupervisorDriverObservationV3>, PcDriverEvidenceErrorV3> {
        let custody = driver_evidence_lock!(self.custody, "custody");
        if !Arc::ptr_eq(&custody.supervisor, &self.supervisor)
            || !custody
                .native
                .as_ref()
                .is_some_and(|native| Arc::ptr_eq(native, &self.native))
            || custody.thread.is_some()
            || !custody
                .stop_authorization
                .as_ref()
                .is_some_and(|authorization| {
                    authorization.matches_original(&self.supervisor, &self.native)
                })
        {
            return Err(PcDriverEvidenceErrorV3::Invalid(
                "P-C joined original custody differs",
            ));
        }
        let observation = match custody.state {
            SupervisorThreadStateV3::Joined if !self.panicked && custody.panic.is_none() => {
                let Some(result) = custody.result.as_ref() else {
                    return Err(PcDriverEvidenceErrorV3::Invalid(
                        "P-C joined driver result absent",
                    ));
                };
                let root_observation = {
                    let supervisor = driver_evidence_lock!(self.supervisor, "supervisor");
                    match supervisor.try_root_failure_observation() {
                        PcDriverPollV3::Pending => return Ok(PcDriverPollV3::Pending),
                        PcDriverPollV3::Ready(observation) => observation,
                    }
                };
                classify_joined_driver(custody.first_error.as_ref(), result, root_observation)
            }
            SupervisorThreadStateV3::JoinPanicked if self.panicked && custody.panic.is_some() => {
                PcSupervisorDriverObservationV3::Panicked {
                    first_error: custody.first_error.clone(),
                }
            }
            _ => {
                return Err(PcDriverEvidenceErrorV3::Invalid(
                    "P-C joined driver state differs",
                ));
            }
        };
        Ok(PcDriverPollV3::Ready(observation))
    }

    /// Nonblocking retirement of only this joined driver's registry owner.
    /// Both proofs remain borrowed on contention and errors.
    pub(crate) fn try_release_after_settlement(
        &self,
        permit: &rz_uci::rovent_bootstrap::launch::RoventPcCollectionSettlementPermitV3,
    ) -> Result<PcDriverPollV3<bool>, PcDriverEvidenceErrorV3> {
        if !permit.matches_original_owner(&self.native) {
            return Err(PcDriverEvidenceErrorV3::Invalid(
                "P-C supervisor settlement belongs to another native owner",
            ));
        }
        let mut registry = driver_evidence_lock!(supervisor_registry_v3(), "registry");
        let Some(index) = registry
            .iter()
            .position(|entry| Arc::ptr_eq(&entry.custody, &self.custody))
        else {
            return Ok(PcDriverPollV3::Ready(false));
        };
        let custody = driver_evidence_lock!(self.custody, "custody");
        if !Arc::ptr_eq(&custody.supervisor, &self.supervisor)
            || custody.state != SupervisorThreadStateV3::Joined
            || custody.thread.is_some()
            || !custody
                .native
                .as_ref()
                .is_some_and(|native| permit.matches_original_owner(native))
        {
            return Ok(PcDriverPollV3::Ready(false));
        }
        let mut supervisor = driver_evidence_lock!(custody.supervisor, "supervisor");
        supervisor.driver_sealed = true;
        drop(supervisor);
        drop(custody);
        registry.swap_remove(index);
        Ok(PcDriverPollV3::Ready(true))
    }

    /// Keep both original proofs with the caller on pending or failure.
    /// This retires only the supervisor registry owner, not the observer,
    /// output, GPU work, or the collection's semantic result.
    pub(crate) fn release_after_settlement(
        &self,
        permit: &rz_uci::rovent_bootstrap::launch::RoventPcCollectionSettlementPermitV3,
    ) -> Result<bool, &'static str> {
        if !permit.matches_original_owner(&self.native) {
            return Err("P-C supervisor settlement belongs to another native owner");
        }
        SupervisorStartCustodyV3::release_after_settlement(&self.custody, permit)
    }
    pub(crate) fn matches_original(
        &self,
        custody: &Arc<Mutex<SupervisorStartCustodyV3>>,
        supervisor: &Arc<Mutex<PcWorkerSupervisorV3>>,
        native: &Arc<rz_uci::rovent_bootstrap::launch::RoventPcCollectionLaunchOwnerV3>,
    ) -> bool {
        Arc::ptr_eq(&self.custody, custody)
            && Arc::ptr_eq(&self.supervisor, supervisor)
            && Arc::ptr_eq(&self.native, native)
    }
}

/// A sealed original driver which never created a supervisor thread.
pub(crate) struct PcSupervisorNoWorkerProofV3 {
    custody: Arc<Mutex<SupervisorStartCustodyV3>>,
    supervisor: Arc<Mutex<PcWorkerSupervisorV3>>,
    native: Arc<rz_uci::rovent_bootstrap::launch::RoventPcCollectionLaunchOwnerV3>,
}

impl PcSupervisorNoWorkerProofV3 {
    pub(crate) fn matches_original(
        &self,
        custody: &Arc<Mutex<SupervisorStartCustodyV3>>,
        supervisor: &Arc<Mutex<PcWorkerSupervisorV3>>,
        native: &Arc<rz_uci::rovent_bootstrap::launch::RoventPcCollectionLaunchOwnerV3>,
    ) -> bool {
        Arc::ptr_eq(&self.custody, custody)
            && Arc::ptr_eq(&self.supervisor, supervisor)
            && Arc::ptr_eq(&self.native, native)
    }
}

/// A created driver joined before it was admitted to process any root.
pub(crate) struct PcSupervisorBeforeAdmissionJoinProofV3 {
    custody: Arc<Mutex<SupervisorStartCustodyV3>>,
    supervisor: Arc<Mutex<PcWorkerSupervisorV3>>,
    native: Arc<rz_uci::rovent_bootstrap::launch::RoventPcCollectionLaunchOwnerV3>,
    panicked: bool,
}

impl PcSupervisorBeforeAdmissionJoinProofV3 {
    pub(crate) fn panicked(&self) -> bool {
        self.panicked
    }
    pub(crate) fn matches_original(
        &self,
        custody: &Arc<Mutex<SupervisorStartCustodyV3>>,
        supervisor: &Arc<Mutex<PcWorkerSupervisorV3>>,
        native: &Arc<rz_uci::rovent_bootstrap::launch::RoventPcCollectionLaunchOwnerV3>,
    ) -> bool {
        Arc::ptr_eq(&self.custody, custody)
            && Arc::ptr_eq(&self.supervisor, supervisor)
            && Arc::ptr_eq(&self.native, native)
    }
}

struct SupervisorRegistryEntryV3 {
    supervisor: Arc<Mutex<PcWorkerSupervisorV3>>,
    custody: Arc<Mutex<SupervisorStartCustodyV3>>,
}

fn supervisor_registry_v3() -> &'static Mutex<Vec<SupervisorRegistryEntryV3>> {
    static REGISTRY: OnceLock<Mutex<Vec<SupervisorRegistryEntryV3>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

impl SupervisorStartCustodyV3 {
    /// Nonblocking versions retain the original owners on contention/error.
    pub(crate) fn try_joined_proof(
        owner: &Arc<Mutex<Self>>,
    ) -> Result<PcDriverPollV3<Option<PcSupervisorJoinedProofV3>>, PcDriverEvidenceErrorV3> {
        let registry = driver_evidence_lock!(supervisor_registry_v3(), "registry");
        if !registry
            .iter()
            .any(|entry| Arc::ptr_eq(&entry.custody, owner))
        {
            return Err(PcDriverEvidenceErrorV3::Invalid(
                "original owner is not registered",
            ));
        }
        let custody = driver_evidence_lock!(owner, "custody");
        let Some(native) = custody.native.as_ref() else {
            return Ok(PcDriverPollV3::Ready(None));
        };
        if !matches!(
            custody.state,
            SupervisorThreadStateV3::Joined | SupervisorThreadStateV3::JoinPanicked
        ) || custody.thread.is_some()
            || custody.stop_authorization.is_none()
        {
            return Ok(PcDriverPollV3::Ready(None));
        }
        Ok(PcDriverPollV3::Ready(Some(PcSupervisorJoinedProofV3 {
            custody: Arc::clone(owner),
            supervisor: Arc::clone(&custody.supervisor),
            native: Arc::clone(native),
            panicked: custody.state == SupervisorThreadStateV3::JoinPanicked,
        })))
    }

    pub(crate) fn try_before_admission_join_proof(
        owner: &Arc<Mutex<Self>>,
    ) -> Result<
        PcDriverPollV3<Option<PcSupervisorBeforeAdmissionJoinProofV3>>,
        PcDriverEvidenceErrorV3,
    > {
        let registry = driver_evidence_lock!(supervisor_registry_v3(), "registry");
        if !registry
            .iter()
            .any(|entry| Arc::ptr_eq(&entry.custody, owner))
        {
            return Err(PcDriverEvidenceErrorV3::Invalid(
                "original owner is not registered",
            ));
        }
        let custody = driver_evidence_lock!(owner, "custody");
        let Some(native) = custody.native.as_ref() else {
            return Ok(PcDriverPollV3::Ready(None));
        };
        if !matches!(
            custody.state,
            SupervisorThreadStateV3::Joined | SupervisorThreadStateV3::JoinPanicked
        ) || custody.thread.is_some()
            || custody.admitted.load(Ordering::Acquire)
            || !custody.stop.load(Ordering::Acquire)
            || custody.stop_authorization.is_some()
            || custody.first_error.is_none()
        {
            return Ok(PcDriverPollV3::Ready(None));
        }
        Ok(PcDriverPollV3::Ready(Some(
            PcSupervisorBeforeAdmissionJoinProofV3 {
                custody: Arc::clone(owner),
                supervisor: Arc::clone(&custody.supervisor),
                native: Arc::clone(native),
                panicked: custody.state == SupervisorThreadStateV3::JoinPanicked,
            },
        )))
    }

    pub(crate) fn try_seal_no_worker(
        owner: &Arc<Mutex<Self>>,
        native: &Arc<rz_uci::rovent_bootstrap::launch::RoventPcCollectionLaunchOwnerV3>,
    ) -> Result<PcDriverPollV3<Option<PcSupervisorNoWorkerProofV3>>, PcDriverEvidenceErrorV3> {
        let registry = driver_evidence_lock!(supervisor_registry_v3(), "registry");
        if !registry
            .iter()
            .any(|entry| Arc::ptr_eq(&entry.custody, owner))
        {
            return Err(PcDriverEvidenceErrorV3::Invalid(
                "original owner is not registered",
            ));
        }
        let mut custody = driver_evidence_lock!(owner, "custody");
        if !matches!(
            custody.state,
            SupervisorThreadStateV3::Prepared | SupervisorThreadStateV3::NoWorker
        ) || custody.thread.is_some()
            || custody.admitted.load(Ordering::Acquire)
        {
            return Ok(PcDriverPollV3::Ready(None));
        }
        let original_supervisor = Arc::clone(&custody.supervisor);
        let mut original = driver_evidence_lock!(original_supervisor, "supervisor");
        original
            .bridge
            .bind_native_owner(native)
            .map_err(PcDriverEvidenceErrorV3::Invalid)?;
        original.driver_sealed = true;
        custody.state = SupervisorThreadStateV3::NoWorker;
        drop(original);
        Ok(PcDriverPollV3::Ready(Some(PcSupervisorNoWorkerProofV3 {
            custody: Arc::clone(owner),
            supervisor: original_supervisor,
            native: Arc::clone(native),
        })))
    }

    pub(crate) fn try_release_before_admission_join(
        owner: &Arc<Mutex<Self>>,
        proof: &PcSupervisorBeforeAdmissionJoinProofV3,
    ) -> Result<PcDriverPollV3<bool>, PcDriverEvidenceErrorV3> {
        let mut registry = driver_evidence_lock!(supervisor_registry_v3(), "registry");
        let Some(index) = registry
            .iter()
            .position(|entry| Arc::ptr_eq(&entry.custody, owner))
        else {
            return Ok(PcDriverPollV3::Ready(false));
        };
        let custody = driver_evidence_lock!(owner, "custody");
        let Some(native) = custody.native.as_ref() else {
            return Ok(PcDriverPollV3::Ready(false));
        };
        if !proof.matches_original(owner, &custody.supervisor, native)
            || !matches!(
                custody.state,
                SupervisorThreadStateV3::Joined | SupervisorThreadStateV3::JoinPanicked
            )
            || custody.thread.is_some()
            || custody.admitted.load(Ordering::Acquire)
            || !custody.stop.load(Ordering::Acquire)
            || custody.stop_authorization.is_some()
        {
            return Ok(PcDriverPollV3::Ready(false));
        }
        let mut original = driver_evidence_lock!(custody.supervisor, "supervisor");
        original.driver_sealed = true;
        drop(original);
        drop(custody);
        registry.swap_remove(index);
        Ok(PcDriverPollV3::Ready(true))
    }

    pub(crate) fn try_release_no_worker(
        owner: &Arc<Mutex<Self>>,
    ) -> Result<PcDriverPollV3<bool>, PcDriverEvidenceErrorV3> {
        let mut registry = driver_evidence_lock!(supervisor_registry_v3(), "registry");
        let Some(index) = registry
            .iter()
            .position(|entry| Arc::ptr_eq(&entry.custody, owner))
        else {
            return Ok(PcDriverPollV3::Ready(false));
        };
        let custody = driver_evidence_lock!(owner, "custody");
        if custody.state != SupervisorThreadStateV3::NoWorker || custody.thread.is_some() {
            return Ok(PcDriverPollV3::Ready(false));
        }
        let original = driver_evidence_lock!(custody.supervisor, "supervisor");
        if !original.driver_sealed {
            return Ok(PcDriverPollV3::Ready(false));
        }
        drop(original);
        drop(custody);
        registry.swap_remove(index);
        Ok(PcDriverPollV3::Ready(true))
    }

    /// Reserve exactly one retained owner per original supervisor before
    /// entering the post-auth callback. This creates no thread or teacher.
    pub(crate) fn register(
        supervisor: &Arc<Mutex<PcWorkerSupervisorV3>>,
        until: Instant,
    ) -> Result<Arc<Mutex<Self>>, &'static str> {
        {
            let original = supervisor
                .try_lock()
                .map_err(|_| "P-C supervisor unavailable")?;
            if original.driver_sealed
                || until <= Instant::now()
                || until > original.cleanup_deadline
            {
                return Err("P-C supervisor driver deadline invalid");
            }
        }
        let mut registry = supervisor_registry_v3()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if registry
            .iter()
            .any(|entry| Arc::ptr_eq(&entry.supervisor, supervisor))
        {
            return Err("P-C original supervisor already registered");
        }
        if supervisor
            .try_lock()
            .map_err(|_| "P-C supervisor unavailable")?
            .driver_sealed
        {
            return Err("P-C supervisor driver is sealed");
        }
        registry
            .try_reserve(1)
            .map_err(|_| "P-C supervisor registry allocation")?;
        let custody = Arc::new(Mutex::new(Self {
            supervisor: Arc::clone(supervisor),
            native: None,
            stop: Arc::new(AtomicBool::new(false)),
            admitted: Arc::new(AtomicBool::new(false)),
            stop_authorization: None,
            until,
            state: SupervisorThreadStateV3::Prepared,
            thread: None,
            result: None,
            first_error: None,
            panic: None,
        }));
        registry.push(SupervisorRegistryEntryV3 {
            supervisor: Arc::clone(supervisor),
            custody: Arc::clone(&custody),
        });
        Ok(custody)
    }

    /// Recover the same original owner; this key does not authorize starting.
    pub(crate) fn recover(
        supervisor: &Arc<Mutex<PcWorkerSupervisorV3>>,
    ) -> Result<Option<Arc<Mutex<Self>>>, &'static str> {
        let registry = supervisor_registry_v3()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        Ok(registry
            .iter()
            .find(|entry| Arc::ptr_eq(&entry.supervisor, supervisor))
            .map(|entry| Arc::clone(&entry.custody)))
    }

    /// Capture actual join of the same quiescence-authorized driver. This
    /// establishes no GPU, semantic, or durable-output result.
    pub(crate) fn joined_proof(
        owner: &Arc<Mutex<Self>>,
    ) -> Result<Option<PcSupervisorJoinedProofV3>, &'static str> {
        let registry = supervisor_registry_v3()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !registry
            .iter()
            .any(|entry| Arc::ptr_eq(&entry.custody, owner))
        {
            return Err("P-C supervisor original owner is not registered");
        }
        let custody = owner
            .try_lock()
            .map_err(|_| "P-C supervisor custody unavailable")?;
        let Some(native) = custody.native.as_ref() else {
            return Ok(None);
        };
        if !matches!(
            custody.state,
            SupervisorThreadStateV3::Joined | SupervisorThreadStateV3::JoinPanicked
        ) || custody.thread.is_some()
            || custody.stop_authorization.is_none()
        {
            return Ok(None);
        }
        Ok(Some(PcSupervisorJoinedProofV3 {
            custody: Arc::clone(owner),
            supervisor: Arc::clone(&custody.supervisor),
            native: Arc::clone(native),
            panicked: custody.state == SupervisorThreadStateV3::JoinPanicked,
        }))
    }

    pub(crate) fn before_admission_join_proof(
        owner: &Arc<Mutex<Self>>,
    ) -> Result<Option<PcSupervisorBeforeAdmissionJoinProofV3>, &'static str> {
        let registry = supervisor_registry_v3()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !registry
            .iter()
            .any(|entry| Arc::ptr_eq(&entry.custody, owner))
        {
            return Err("P-C supervisor original owner is not registered");
        }
        let custody = owner
            .try_lock()
            .map_err(|_| "P-C supervisor custody unavailable")?;
        let Some(native) = custody.native.as_ref() else {
            return Ok(None);
        };
        if !matches!(
            custody.state,
            SupervisorThreadStateV3::Joined | SupervisorThreadStateV3::JoinPanicked
        ) || custody.thread.is_some()
            || custody.admitted.load(Ordering::Acquire)
            || !custody.stop.load(Ordering::Acquire)
            || custody.stop_authorization.is_some()
            || custody.first_error.is_none()
        {
            return Ok(None);
        }
        Ok(Some(PcSupervisorBeforeAdmissionJoinProofV3 {
            custody: Arc::clone(owner),
            supervisor: Arc::clone(&custody.supervisor),
            native: Arc::clone(native),
            panicked: custody.state == SupervisorThreadStateV3::JoinPanicked,
        }))
    }

    /// The joined-before-admission case has no native root work; only the
    /// exact original proof can retire this driver's registry entry.
    pub(crate) fn release_before_admission_join(
        owner: &Arc<Mutex<Self>>,
        proof: &PcSupervisorBeforeAdmissionJoinProofV3,
    ) -> Result<bool, &'static str> {
        let mut registry = supervisor_registry_v3()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let Some(index) = registry
            .iter()
            .position(|entry| Arc::ptr_eq(&entry.custody, owner))
        else {
            return Ok(false);
        };
        let custody = owner
            .try_lock()
            .map_err(|_| "P-C supervisor custody unavailable")?;
        let Some(native) = custody.native.as_ref() else {
            return Ok(false);
        };
        if !proof.matches_original(owner, &custody.supervisor, native)
            || !matches!(
                custody.state,
                SupervisorThreadStateV3::Joined | SupervisorThreadStateV3::JoinPanicked
            )
            || custody.thread.is_some()
            || custody.admitted.load(Ordering::Acquire)
            || !custody.stop.load(Ordering::Acquire)
            || custody.stop_authorization.is_some()
        {
            return Ok(false);
        }
        custody
            .supervisor
            .try_lock()
            .map_err(|_| "P-C supervisor unavailable")?
            .driver_sealed = true;
        drop(custody);
        registry.swap_remove(index);
        Ok(true)
    }

    /// Irreversibly seal a registered driver which never spawned a thread.
    /// A started or unknown worker cannot be converted to this evidence.
    pub(crate) fn seal_no_worker(
        owner: &Arc<Mutex<Self>>,
        native: &Arc<rz_uci::rovent_bootstrap::launch::RoventPcCollectionLaunchOwnerV3>,
    ) -> Result<Option<PcSupervisorNoWorkerProofV3>, &'static str> {
        let registry = supervisor_registry_v3()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !registry
            .iter()
            .any(|entry| Arc::ptr_eq(&entry.custody, owner))
        {
            return Err("P-C supervisor original owner is not registered");
        }
        let mut custody = owner
            .try_lock()
            .map_err(|_| "P-C supervisor custody unavailable")?;
        if !matches!(
            custody.state,
            SupervisorThreadStateV3::Prepared | SupervisorThreadStateV3::NoWorker
        ) || custody.thread.is_some()
            || custody.admitted.load(Ordering::Acquire)
        {
            return Ok(None);
        }
        let original_supervisor = Arc::clone(&custody.supervisor);
        let mut original = original_supervisor
            .try_lock()
            .map_err(|_| "P-C supervisor unavailable")?;
        original.bridge.bind_native_owner(native)?;
        original.driver_sealed = true;
        custody.state = SupervisorThreadStateV3::NoWorker;
        drop(original);
        Ok(Some(PcSupervisorNoWorkerProofV3 {
            custody: Arc::clone(owner),
            supervisor: original_supervisor,
            native: Arc::clone(native),
        }))
    }

    /// Only the one registered original owner may start, and only inside the
    /// same post-auth gate after the arena's original resource/control proof.
    /// The registry lock also serializes this transition with no-worker release.
    pub(crate) fn start_verified(
        owner: &Arc<Mutex<Self>>,
        lease: &crate::rovent_resource_scope::launch::PcCollectionResourceStartLeaseV3<'_>,
        gate: &rz_uci::rovent_bootstrap::launch::RoventPcCollectionStartGateV3,
        native: &Arc<rz_uci::rovent_bootstrap::launch::RoventPcCollectionLaunchOwnerV3>,
        control: &rz_uci::rovent_bootstrap::RoventStartupControl,
    ) -> Result<(), &'static str> {
        let registry = match supervisor_registry_v3().try_lock() {
            Ok(registry) => registry,
            Err(std::sync::TryLockError::Poisoned(error)) => error.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return Err("P-C supervisor registry busy"),
        };
        if !registry
            .iter()
            .any(|entry| Arc::ptr_eq(&entry.custody, owner))
        {
            return Err("P-C supervisor original owner is not registered");
        }
        let mut custody = owner
            .try_lock()
            .map_err(|_| "P-C supervisor custody unavailable")?;
        // The exact owner lock prevents its removal; unrelated registry
        // entries need not wait through lease validation or thread creation.
        drop(registry);
        let original_supervisor = Arc::clone(&custody.supervisor);
        let verified = (|| -> Result<(), &'static str> {
            if !lease.matches_original(gate, native, control) {
                return Err("P-C supervisor original resource lease differs");
            }
            let supervisor = original_supervisor
                .try_lock()
                .map_err(|_| "P-C supervisor unavailable")?;
            supervisor.bridge.bind_native_owner(native)?;
            let original_control = supervisor
                .bridge
                .startup()
                .ok_or("P-C original startup control absent")?;
            if !gate.matches_startup_control(original_control) {
                return Err("P-C supervisor startup control differs from original gate");
            }
            Ok(())
        })();
        if let Err(error) = verified {
            custody.first_error.get_or_insert_with(|| error.to_owned());
            return Err(error);
        }
        custody.native = Some(Arc::clone(native));
        let result = custody.spawn_after_authorization(control);
        if let Err(error) = result {
            custody.first_error.get_or_insert_with(|| error.to_owned());
            return Err(error);
        }
        // Cancellation can race the OS thread creation.  Preserve the exact
        // handle and Running state even when this post-check rejects startup.
        if control.check().is_err() {
            let error = "P-C supervisor startup canceled or expired after spawn";
            custody.first_error.get_or_insert_with(|| error.to_owned());
            // The driver has not been admitted to step any root yet.
            custody.stop.store(true, Ordering::Release);
            return Err(error);
        }
        custody.admitted.store(true, Ordering::Release);
        Ok(())
    }

    /// A joined driver may be removed only with this exact launch owner's
    /// original settlement permit. Merely joining or timing out is not proof.
    pub(crate) fn release_after_settlement(
        owner: &Arc<Mutex<Self>>,
        permit: &rz_uci::rovent_bootstrap::launch::RoventPcCollectionSettlementPermitV3,
    ) -> Result<bool, &'static str> {
        let mut registry = match supervisor_registry_v3().try_lock() {
            Ok(registry) => registry,
            Err(std::sync::TryLockError::Poisoned(error)) => error.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return Err("P-C supervisor registry busy"),
        };
        let Some(index) = registry
            .iter()
            .position(|entry| Arc::ptr_eq(&entry.custody, owner))
        else {
            return Ok(false);
        };
        let custody = owner
            .try_lock()
            .map_err(|_| "P-C supervisor custody unavailable")?;
        if custody.state != SupervisorThreadStateV3::Joined
            || custody.thread.is_some()
            || !custody
                .native
                .as_ref()
                .is_some_and(|native| permit.matches_original_owner(native))
        {
            return Ok(false);
        }
        custody
            .supervisor
            .try_lock()
            .map_err(|_| "P-C supervisor unavailable")?
            .driver_sealed = true;
        drop(custody);
        registry.swap_remove(index);
        Ok(true)
    }

    /// A started or unknown driver cannot be released by this path.
    pub(crate) fn release_no_worker(owner: &Arc<Mutex<Self>>) -> Result<bool, &'static str> {
        let mut registry = supervisor_registry_v3()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let Some(index) = registry
            .iter()
            .position(|entry| Arc::ptr_eq(&entry.custody, owner))
        else {
            return Ok(false);
        };
        let custody = owner
            .try_lock()
            .map_err(|_| "P-C supervisor custody unavailable")?;
        if custody.state != SupervisorThreadStateV3::NoWorker || custody.thread.is_some() {
            return Ok(false);
        }
        if !custody
            .supervisor
            .try_lock()
            .map_err(|_| "P-C supervisor unavailable")?
            .driver_sealed
        {
            return Ok(false);
        }
        drop(custody);
        registry.swap_remove(index);
        Ok(true)
    }

    pub(crate) fn state(&self) -> SupervisorThreadStateV3 {
        self.state
    }
    pub(crate) fn matches_supervisor(&self, supervisor: &Arc<Mutex<PcWorkerSupervisorV3>>) -> bool {
        Arc::ptr_eq(&self.supervisor, supervisor)
    }
    /// Stop only after the original bridge has sealed root quiescence. On a
    /// mismatch the caller receives the very same authorization back.
    pub(crate) fn authorize_stop(
        &mut self,
        authorization: super::pc_hooks_lock_v3::PcSupervisorStopAuthorizationV3,
    ) -> Result<
        (),
        (
            super::pc_hooks_lock_v3::PcSupervisorStopAuthorizationV3,
            &'static str,
        ),
    > {
        let Some(native) = self.native.as_ref() else {
            return Err((authorization, "P-C supervisor native owner absent"));
        };
        if self.state != SupervisorThreadStateV3::Running {
            return Err((authorization, "P-C supervisor driver is not running"));
        }
        if self.stop_authorization.is_some() {
            return Err((authorization, "P-C supervisor stop already authorized"));
        }
        if !authorization.matches_original(&self.supervisor, native) {
            return Err((authorization, "P-C supervisor stop authorization differs"));
        }
        self.stop_authorization = Some(authorization);
        self.stop.store(true, Ordering::Release);
        Ok(())
    }

    /// Validate the borrowed original proof before moving it directly from
    /// its retained slot into this custody. No fallible operation follows
    /// the move; an error leaves the original slot untouched.
    pub(crate) fn authorize_from_slot(
        &mut self,
        slot: &mut Option<super::pc_hooks_lock_v3::PcSupervisorStopAuthorizationV3>,
    ) -> Result<(), &'static str> {
        let authorization = slot.as_ref().ok_or("P-C stop authorization absent")?;
        let native = self
            .native
            .as_ref()
            .ok_or("P-C supervisor native owner absent")?;
        if self.stop_authorization.is_some() {
            return Err("P-C supervisor stop already authorized");
        }
        if !authorization.matches_original(&self.supervisor, native) {
            return Err("P-C supervisor stop authorization differs");
        }
        let running = self.state == SupervisorThreadStateV3::Running;
        if !running {
            let abnormal_join = match self.state {
                SupervisorThreadStateV3::Joined => self.result.as_ref().is_some_and(Result::is_err),
                SupervisorThreadStateV3::JoinPanicked => self.panic.is_some(),
                _ => false,
            };
            if !abnormal_join || self.thread.is_some() || !self.admitted.load(Ordering::Acquire) {
                return Err("P-C supervisor abnormal join not established");
            }
        }
        self.stop_authorization = slot.take();
        if running {
            self.stop.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// Associate the same quiescence proof with a driver which actually
    /// joined after an error or panic before a stop request could be sent.
    /// This retains its original result and does not assert collection, GPU,
    /// or output success. A rejected authorization is returned unchanged.
    pub(crate) fn authorize_already_joined(
        &mut self,
        authorization: super::pc_hooks_lock_v3::PcSupervisorStopAuthorizationV3,
    ) -> Result<
        (),
        (
            super::pc_hooks_lock_v3::PcSupervisorStopAuthorizationV3,
            &'static str,
        ),
    > {
        let Some(native) = self.native.as_ref() else {
            return Err((authorization, "P-C supervisor native owner absent"));
        };
        let abnormal_join = match self.state {
            SupervisorThreadStateV3::Joined => self.result.as_ref().is_some_and(Result::is_err),
            SupervisorThreadStateV3::JoinPanicked => self.panic.is_some(),
            _ => false,
        };
        if !abnormal_join || self.thread.is_some() || !self.admitted.load(Ordering::Acquire) {
            return Err((
                authorization,
                "P-C supervisor abnormal join not established",
            ));
        }
        if self.stop_authorization.is_some() {
            return Err((authorization, "P-C supervisor stop already authorized"));
        }
        if !authorization.matches_original(&self.supervisor, native) {
            return Err((authorization, "P-C supervisor stop authorization differs"));
        }
        self.stop_authorization = Some(authorization);
        Ok(())
    }
    pub(crate) fn result(&self) -> Option<&Result<(), String>> {
        self.result.as_ref()
    }
    pub(crate) fn first_error(&self) -> Option<&str> {
        self.first_error.as_deref()
    }
    pub(crate) fn panic_payload(&self) -> Option<&(dyn std::any::Any + Send)> {
        self.panic.as_deref()
    }

    /// Never wait on a live thread. A driver join says nothing about its
    /// root workers, physical devices, output durability or collection success.
    pub(crate) fn try_join(&mut self) -> SupervisorThreadStateV3 {
        if self.state != SupervisorThreadStateV3::Running
            || !self
                .thread
                .as_ref()
                .is_some_and(std::thread::JoinHandle::is_finished)
        {
            return self.state;
        }
        let Some(handle) = self.thread.take() else {
            return self.state;
        };
        match handle.join() {
            Ok(result) => {
                if let Err(error) = &result {
                    self.first_error.get_or_insert_with(|| error.clone());
                }
                self.result = Some(result);
                self.state = SupervisorThreadStateV3::Joined;
            }
            Err(payload) => {
                self.panic = Some(payload);
                self.state = SupervisorThreadStateV3::JoinPanicked;
            }
        }
        self.state
    }

    /// Internal primitive; an externally callable start must first validate
    /// the original post-auth gate and same resource/control/parent custody.
    /// Until that verifier exists this method has no production caller.
    fn spawn_after_authorization(
        &mut self,
        control: &rz_uci::rovent_bootstrap::RoventStartupControl,
    ) -> Result<(), &'static str> {
        if self.state != SupervisorThreadStateV3::Prepared || Instant::now() >= self.until {
            return Err("P-C supervisor start state or deadline invalid");
        }
        control
            .check()
            .map_err(|_| "P-C supervisor startup canceled or expired before spawn")?;
        self.state = SupervisorThreadStateV3::StartingUnknown;
        let supervisor = Arc::clone(&self.supervisor);
        let stop = Arc::clone(&self.stop);
        let admitted = Arc::clone(&self.admitted);
        let until = self.until;
        let started = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            std::thread::Builder::new()
                .name("rovent-pc-supervisor-v3".into())
                .spawn(move || {
                    loop {
                        if stop.load(Ordering::Acquire) {
                            return Ok(());
                        }
                        if Instant::now() >= until {
                            return Err("P-C supervisor cleanup deadline expired".into());
                        }
                        if !admitted.load(Ordering::Acquire) {
                            std::thread::sleep(Duration::from_millis(10));
                            continue;
                        }
                        match supervisor.try_lock() {
                            Ok(mut owner) => owner.step()?,
                            Err(std::sync::TryLockError::WouldBlock) => {}
                            Err(std::sync::TryLockError::Poisoned(_)) => {
                                return Err("P-C supervisor owner poisoned".into());
                            }
                        }
                        std::thread::sleep(Duration::from_millis(10));
                    }
                })
        }));
        match started {
            Ok(Ok(handle)) => {
                self.thread = Some(handle);
                self.state = SupervisorThreadStateV3::Running;
                Ok(())
            }
            Ok(Err(error)) => {
                self.result = Some(Err(error.to_string()));
                self.state = SupervisorThreadStateV3::NoWorker;
                Err("P-C supervisor thread creation failed")
            }
            Err(payload) => {
                self.panic = Some(payload);
                Err("P-C supervisor thread creation unknown")
            }
        }
    }
}

impl PcWorkerSupervisorV3 {
    /// Observe retained root failures without terminating the driver, altering
    /// recovery, or promoting an unknown lock state to a failed root.
    pub(crate) fn try_root_failure_observation(
        &self,
    ) -> PcDriverPollV3<PcRootFailureObservationV3> {
        observe_root_failures(
            &self.roots,
            self.first_root_error.as_deref(),
            PcRootRecoveryV3::failure,
        )
    }

    /// Observe secondary teacher drain failures on the original roots. This
    /// is a separate snapshot from `try_root_failure_observation`: lock
    /// contention or poison in either snapshot must not be inferred away.
    /// No root, ACK, worker, or physical-completion state is changed.
    pub(crate) fn try_teacher_drain_failure_observation(
        &self,
    ) -> PcDriverPollV3<PcRootFailureObservationV3> {
        observe_root_failures(&self.roots, None, PcRootRecoveryV3::teacher_drain_failure)
    }
    /// Call only while holding the guard obtained from `original` itself.
    /// The bridge performs this binding before any worker or admission.
    pub(super) fn bind_original_identity(
        &mut self,
        original: &Arc<Mutex<Self>>,
    ) -> Result<(), &'static str> {
        let candidate = Arc::downgrade(original);
        if let Some(bound) = self.original_self.get() {
            return if Weak::ptr_eq(bound, &candidate) {
                Ok(())
            } else {
                Err("P-C V3 supervisor identity changed")
            };
        }
        self.original_self
            .set(candidate)
            .map_err(|_| "P-C V3 supervisor identity raced")
    }

    /// Construct only from the same original supervisor and a recovery already
    /// registered in its root list and in the bridge's exact slot.
    pub(super) fn try_received_stage_proof(
        supervisor: &Arc<Mutex<Self>>,
        recovery: &Arc<Mutex<PcRootRecoveryV3>>,
    ) -> Result<Option<PcReceivedStageProofV3>, &'static str> {
        let owner = match supervisor.try_lock() {
            Ok(owner) => owner,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 supervisor poisoned"),
        };
        if !owner.owns_recovery(recovery) {
            return Err("P-C V3 recovery not owned by supervisor");
        }
        let bridge = Arc::clone(&owner.bridge);
        let root = match recovery.try_lock() {
            Ok(root) => root,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 recovery poisoned"),
        };
        if !root.matches_received_root(&bridge) {
            return Err("P-C V3 received root owner differs");
        }
        let Some(stage) = root.stage.as_ref().map(Arc::clone) else {
            return Ok(None);
        };
        let stage_guard = match stage.try_lock() {
            Ok(stage) => stage,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 stage poisoned"),
        };
        let scope_identity = stage_guard.scope_effect_identity_weak();
        if !root
            .original_scope_identity
            .as_ref()
            .is_some_and(|original| Weak::ptr_eq(original, &scope_identity))
        {
            return Err("P-C V3 original scope identity differs");
        }
        drop(stage_guard);
        let proof = PcReceivedStageProofV3 {
            supervisor: Arc::downgrade(supervisor),
            bridge: Arc::downgrade(&bridge),
            recovery: Arc::downgrade(recovery),
            stage: Arc::downgrade(&stage),
            scope_identity,
            admission: root.admission.clone(),
        };
        drop(root);
        drop(owner);
        match proof.try_revalidate()? {
            None => Ok(None),
            Some(true) => Ok(Some(proof)),
            Some(false) => Err("P-C V3 registered stage owner differs"),
        }
    }

    /// These are only identity and in-memory handoff observations. The bridge
    /// must separately close admission and verify its queue and physical proof.
    pub(crate) fn matches_original_bridge(&self, bridge: &Arc<RoventPcCollectionBridgeV3>) -> bool {
        Arc::ptr_eq(&self.bridge, bridge)
    }

    pub(crate) fn pending_root_absent(&self) -> bool {
        self.pending_root.is_none()
    }

    pub(crate) fn owns_recovery(&self, recovery: &Arc<Mutex<PcRootRecoveryV3>>) -> bool {
        self.roots
            .iter()
            .any(|original| Arc::ptr_eq(original, recovery))
    }

    /// Retain the game's original output owner before the first root exists.
    /// Actual pipeline installation occurs only after that root creates it.
    pub(crate) fn register_output(
        &mut self,
        output: &Arc<ManagedOutputOwnerV3>,
    ) -> Result<(), &'static str> {
        if let Some(original) = &self.managed_output {
            return if Arc::ptr_eq(original, output) {
                Ok(())
            } else {
                Err("P-C V3 original output owner changed")
            };
        }
        if self.pipeline.is_some() || self.pending_root.is_some() || !self.roots.is_empty() {
            return Err("P-C V3 output owner must be registered before roots");
        }
        self.managed_output = Some(Arc::clone(output));
        Ok(())
    }

    /// Same in-memory game output owner; no durable sink is implied.
    pub(super) fn output_owner(&self) -> Option<PolicyOutputOwnerV3> {
        self.pipeline
            .as_ref()
            .map(PolicyRowPipelineOwnerV3::output_owner)
    }

    /// Retain the original game pipeline; this is not a finish or output
    /// receipt. Callers must still verify same-root observer custody.
    pub(super) fn game_pipeline(&self) -> Option<PolicyRowPipelineOwnerV3> {
        self.pipeline.clone()
    }

    /// The caller retains its original managed output capability; path names
    /// or a second copy of input bytes cannot manufacture this ownership.
    fn try_install_output(&self, output: &Arc<ManagedOutputOwnerV3>) -> Result<bool, String> {
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(false);
        };
        let Some(mut pipeline) = pipeline.try_lock().map_err(str::to_owned)? else {
            return Ok(false);
        };
        pipeline
            .install_output_directory(self.bridge.inputs(), output)
            .map_err(|e| e.to_string())
    }
    pub(super) fn new(
        bridge: Arc<RoventPcCollectionBridgeV3>,
        router: Arc<CaptureRouterV3>,
        startup_deadline: Instant,
        cleanup_deadline: Instant,
    ) -> Result<Self, &'static str> {
        if startup_deadline > cleanup_deadline {
            return Err("P-C V3 invalid supervisor deadlines");
        }
        let mut roots = Vec::new();
        roots
            .try_reserve_exact(usize::from(bridge.inputs().plan().maximum_roots))
            .map_err(|_| "P-C V3 supervisor allocation failed")?;
        Ok(Self {
            original_self: OnceLock::new(),
            bridge,
            router,
            roots,
            first_root_error: None,
            pending_root: None,
            pipeline: None,
            managed_output: None,
            output_installed: false,
            driver_sealed: false,
            startup_deadline,
            cleanup_deadline,
        })
    }

    /// One finite poll; unresolved evidence stays in its original root owner.
    fn step(&mut self) -> Result<(), String> {
        if self.pending_root.is_none()
            && self.roots.len() < usize::from(self.bridge.inputs().plan().maximum_roots)
        {
            let _ = self
                .bridge
                .try_receive_root_into(&mut self.pending_root)
                .map_err(str::to_owned)?;
        }
        if let Some(received) = self.pending_root.as_ref() {
            let admission = received.admission();
            if !received.matches_original(&self.bridge, admission) {
                return Err("P-C V3 received root owner differs".into());
            }
            if self.pipeline.is_none() {
                let writer = RoventCorpusWriterLockV3::new_shared(
                    self.bridge.inputs().manifest(),
                    admission,
                )
                .map_err(|error| error.to_string())?;
                self.pipeline = Some(PolicyRowPipelineOwnerV3::new(writer));
            }
            match self
                .bridge
                .game_authority()
                .try_bind_pipeline(self.pipeline.as_ref().expect("pipeline retained"))
            {
                super::game_authority_v3::GameStepV3::Ready
                | super::game_authority_v3::GameStepV3::Existing => {}
                super::game_authority_v3::GameStepV3::Busy => return Ok(()),
                _ => return Err("P-C V3 game pipeline binding rejected".into()),
            }
            if !self.output_installed {
                if let Some(output) = self.managed_output.as_ref() {
                    match self.bridge.game_authority().try_bind_output(output) {
                        super::game_authority_v3::GameStepV3::Ready
                        | super::game_authority_v3::GameStepV3::Existing => {}
                        super::game_authority_v3::GameStepV3::Busy => return Ok(()),
                        _ => return Err("P-C V3 game output binding rejected".into()),
                    }
                    if !self.try_install_output(output)? {
                        return Ok(());
                    }
                    self.output_installed = true;
                }
            }
            if self.roots.iter().any(|root| {
                root.lock()
                    .ok()
                    .is_some_and(|root| root.matches_admission(admission))
            }) {
                return Err("P-C V3 duplicate original root".into());
            }
            self.roots.push(PcRootRecoveryV3::new(
                admission.clone(),
                self.pipeline.as_ref().expect("pipeline retained").clone(),
            ));
            // Allocation and vector insertion precede moving the actual
            // receiver-issued owner out of the supervisor's pending slot.
            let original = self.roots.last().expect("root inserted");
            let mut original = original.lock().map_err(|_| "P-C V3 new root poisoned")?;
            original.received_root = self.pending_root.take();
        }
        for original in &self.roots {
            let outcome = prepare_root_step_v3(
                self,
                &self.bridge,
                &self.router,
                original,
                self.startup_deadline,
                self.cleanup_deadline,
            );
            let published = match outcome {
                Ok(PcPrepareStepV3::Published) => true,
                Ok(PcPrepareStepV3::Pending | PcPrepareStepV3::Failed) => false,
                Err(error) => {
                    if let Ok(mut root) = original.lock() {
                        root.failure.get_or_insert(error);
                    }
                    false
                }
            };
            if self.first_root_error.is_none() {
                if let Ok(root) = original.try_lock() {
                    if let Some(error) = root.failure() {
                        self.first_root_error = Some(error.to_owned());
                    }
                }
            }
            if !published {
                continue;
            }
            let admission = original
                .lock()
                .map_err(|_| "P-C V3 recovery poisoned")?
                .admission
                .clone();
            if !self.bridge.try_join_root(&admission).unwrap_or(false) {
                continue;
            }
            if matches!(
                self.bridge.try_account_root(&admission),
                Ok(PcRootAccountingV3::PhysicallyRetired)
            ) {
                let _ = self.bridge.try_retire_route(&admission);
            }
        }
        Ok(())
    }
}

enum PcScopeDriveV3 {
    Pending,
    Finished,
    Failed,
    Aborted,
    RootCanceled,
}

/// All mutable effect slots were installed with this recovery before entry.
/// The caller holds the original supervisor/recovery/stage guards already.
fn drive_pc_scope_effect_v3(
    supervisor: &PcWorkerSupervisorV3,
    recovery: &mut PcRootRecoveryV3,
    stage_owner: &Arc<Mutex<TeacherSessionStagingV3>>,
    stage: &mut TeacherSessionStagingV3,
) -> PcScopeDriveV3 {
    use super::game_authority_v3::GameStepV3 as G;
    use super::teacher::v3_session::TeacherScopeEffectClaimFailureV3 as F;
    let owner = Arc::clone(&recovery.scope_effects);
    let mut slots = match owner.try_lock() {
        Ok(slots) => slots,
        Err(std::sync::TryLockError::WouldBlock) => return PcScopeDriveV3::Pending,
        Err(_) => {
            recovery
                .failure
                .get_or_insert_with(|| "P-C V3 scope slots poisoned".into());
            return PcScopeDriveV3::Failed;
        }
    };
    let PcScopeEffectSlotsV3 {
        proof,
        teacher_source,
        game_source,
        claim,
        outcome,
        issue_error,
        source_error,
        claim_error,
        completion_error,
        source_step,
        claim_step,
        completion_step,
        finished,
        ..
    } = &mut *slots;
    if *finished {
        // Effect settlement and a successful scope claim are separate. Keep
        // a known rejected original outcome rejected on every re-observation.
        return if matches!(outcome.as_ref(), Some(Ok(_))) {
            PcScopeDriveV3::Finished
        } else {
            PcScopeDriveV3::Failed
        };
    }
    if issue_error.is_some()
        || source_error.is_some()
        || claim_error.is_some()
        || completion_error.is_some()
    {
        return PcScopeDriveV3::Failed;
    }
    let game = supervisor.bridge.game_authority();
    if game.abort_requested() && claim.is_none() {
        return PcScopeDriveV3::Aborted;
    }
    if game_source.is_none() {
        if proof.is_none() {
            match PcReceivedStageProofV3::try_from_guards(supervisor, recovery, stage_owner, stage)
            {
                Ok(Some(original)) => *proof = Some(original),
                Ok(None) => return PcScopeDriveV3::Pending,
                Err(error) => {
                    *source_error = Some(error);
                    return PcScopeDriveV3::Failed;
                }
            }
        }
        if teacher_source.is_none() {
            if let Err(error) = stage.issue_scope_effect_source(&recovery.admission, teacher_source)
            {
                *issue_error = Some(error);
                return PcScopeDriveV3::Failed;
            }
        }
        let step = game.try_issue_pc_scope_source_in_slot(
            proof,
            teacher_source,
            supervisor,
            recovery,
            &recovery.admission,
            game_source,
            source_error,
        );
        *source_step = Some(step);
        match step {
            G::Ready => {}
            G::Busy => return PcScopeDriveV3::Pending,
            G::Aborted | G::Closed => return PcScopeDriveV3::Aborted,
            G::RootCanceled => return PcScopeDriveV3::RootCanceled,
            _ => return PcScopeDriveV3::Failed,
        }
    }
    let Some(source) = game_source.as_ref() else {
        return PcScopeDriveV3::Failed;
    };
    if claim.is_none() {
        let step = game.try_claim_pc_scope(source, supervisor, recovery, claim, claim_error);
        *claim_step = Some(step);
        match step {
            G::Ready => {}
            G::Busy => return PcScopeDriveV3::Pending,
            G::Aborted | G::Closed => return PcScopeDriveV3::Aborted,
            G::RootCanceled => return PcScopeDriveV3::RootCanceled,
            _ => return PcScopeDriveV3::Failed,
        }
    }
    let Some(original_claim) = claim.as_mut() else {
        return PcScopeDriveV3::Failed;
    };
    if outcome.is_none() {
        let Some(teacher) = source.pc_scope_source() else {
            return PcScopeDriveV3::Failed;
        };
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stage.claim_with_effect_source(teacher, original_claim, outcome)
        }));
        match attempted {
            Ok(true) => {}
            Ok(false) => {
                *completion_error = Some("P-C V3 scope claim was not entered");
                return PcScopeDriveV3::Failed;
            }
            Err(payload) => {
                recovery.panic = Some(payload);
                return PcScopeDriveV3::Failed;
            }
        }
    }
    let (original_outcome, succeeded) = match outcome.as_ref() {
        Some(Ok(value)) => (value, true),
        Some(Err(F::Known { outcome, .. })) => (outcome, false),
        Some(Err(F::RejectedBeforeEntry(_))) | Some(Err(F::Unknown(_))) | None => {
            return PcScopeDriveV3::Failed;
        }
    };
    let step = original_claim.try_complete_pc_scope(
        source,
        original_outcome,
        supervisor,
        recovery,
        completion_error,
    );
    *completion_step = Some(step);
    match step {
        G::Ready | G::Existing => {
            *finished = true;
            if succeeded {
                PcScopeDriveV3::Finished
            } else {
                PcScopeDriveV3::Failed
            }
        }
        G::Busy => PcScopeDriveV3::Pending,
        _ => PcScopeDriveV3::Failed,
    }
}

/// One already-registered actor acquisition, separate from physical shutdown.
/// All four original guards are held by the caller; none is reacquired here.
fn drive_pc_actor_effect_v3(
    supervisor: &PcWorkerSupervisorV3,
    recovery: &mut PcRootRecoveryV3,
    stage_owner: &Arc<Mutex<TeacherSessionStagingV3>>,
    stage: &TeacherSessionStagingV3,
    session_owner: &Arc<Mutex<RoventCollectionTeacherSessionV3>>,
    session: &mut RoventCollectionTeacherSessionV3,
    actor_slot: &Arc<Mutex<PcActorEffectSlotsV3>>,
    startup_deadline: Instant,
    cleanup_deadline: Instant,
) -> PcScopeDriveV3 {
    use super::game_authority_v3::GameStepV3 as G;
    use super::teacher::v3_session::TeacherActorEffectFailureV3 as F;
    let mut slots = match actor_slot.try_lock() {
        Ok(slots) => slots,
        Err(std::sync::TryLockError::WouldBlock) => return PcScopeDriveV3::Pending,
        Err(_) => {
            recovery
                .failure
                .get_or_insert_with(|| "P-C V3 actor slots poisoned".into());
            return PcScopeDriveV3::Failed;
        }
    };
    let PcActorEffectSlotsV3 {
        actor_id,
        proof,
        teacher_source,
        game_source,
        claim,
        outcome,
        issue_error,
        source_error,
        claim_error,
        completion_error,
        source_step,
        claim_step,
        completion_step,
        finished,
    } = &mut *slots;
    if *finished {
        return if matches!(outcome.as_ref(), Some(Ok(_))) {
            PcScopeDriveV3::Finished
        } else {
            PcScopeDriveV3::Failed
        };
    }
    if issue_error.is_some()
        || source_error.is_some()
        || claim_error.is_some()
        || completion_error.is_some()
    {
        return PcScopeDriveV3::Failed;
    }
    let game = supervisor.bridge.game_authority();
    if game.abort_requested() && claim.is_none() {
        return PcScopeDriveV3::Aborted;
    }
    if game_source.is_none() {
        if teacher_source.is_none() {
            if let Err(error) =
                session.issue_actor_effect_source(&recovery.admission, actor_id, teacher_source)
            {
                *issue_error = Some(error);
                return PcScopeDriveV3::Failed;
            }
        }
        if proof.is_none() {
            let Some(source) = teacher_source.as_ref() else {
                return PcScopeDriveV3::Failed;
            };
            match PcReceivedActorProofV3::try_from_guards(
                supervisor,
                recovery,
                stage_owner,
                stage,
                session_owner,
                session,
                source,
            ) {
                Ok(Some(original)) => *proof = Some(original),
                Ok(None) => return PcScopeDriveV3::Pending,
                Err(error) => {
                    *source_error = Some(error);
                    return PcScopeDriveV3::Failed;
                }
            }
        }
        let step = game.try_issue_pc_actor_source_in_slot(
            proof,
            teacher_source,
            supervisor,
            recovery,
            stage,
            session,
            &recovery.admission,
            game_source,
            source_error,
        );
        *source_step = Some(step);
        match step {
            G::Ready => {}
            G::Busy => return PcScopeDriveV3::Pending,
            G::Aborted | G::Closed => return PcScopeDriveV3::Aborted,
            G::RootCanceled => return PcScopeDriveV3::RootCanceled,
            _ => return PcScopeDriveV3::Failed,
        }
    }
    let Some(source) = game_source.as_ref() else {
        return PcScopeDriveV3::Failed;
    };
    if claim.is_none() {
        let step = game.try_claim_pc_actor(
            source,
            supervisor,
            recovery,
            stage,
            session,
            claim,
            claim_error,
        );
        *claim_step = Some(step);
        match step {
            G::Ready => {}
            G::Busy => return PcScopeDriveV3::Pending,
            G::Aborted | G::Closed => return PcScopeDriveV3::Aborted,
            G::RootCanceled => return PcScopeDriveV3::RootCanceled,
            _ => return PcScopeDriveV3::Failed,
        }
    }
    let Some(original_claim) = claim.as_mut() else {
        return PcScopeDriveV3::Failed;
    };
    if outcome.is_none() {
        let Some(teacher) = source.pc_actor_source() else {
            return PcScopeDriveV3::Failed;
        };
        if !session.start_actor_with_effect_source(
            teacher,
            original_claim,
            startup_deadline,
            cleanup_deadline,
            &mut recovery.custodian,
            outcome,
        ) {
            *completion_error = Some("P-C V3 actor start was not entered");
            return PcScopeDriveV3::Failed;
        }
    }
    let (original_outcome, succeeded) = match outcome.as_ref() {
        Some(Ok((_, value))) => (value, true),
        Some(Err(F::Known { outcome, .. })) => (outcome, false),
        Some(Err(F::RejectedBeforeEntry(_))) | Some(Err(F::Unknown(_))) | None => {
            return PcScopeDriveV3::Failed;
        }
    };
    let step = original_claim.try_complete_pc_actor(
        source,
        original_outcome,
        supervisor,
        recovery,
        stage,
        session,
        completion_error,
    );
    *completion_step = Some(step);
    match step {
        G::Ready | G::Existing => {
            *finished = true;
            if succeeded {
                PcScopeDriveV3::Finished
            } else {
                PcScopeDriveV3::Failed
            }
        }
        G::Busy => PcScopeDriveV3::Pending,
        _ => PcScopeDriveV3::Failed,
    }
}

/// Drive exactly one original route installation attempt. The result slot is
/// retained in the recovery before the authority claim or producer entry.
fn drive_pc_route_effect_v3(
    supervisor: &PcWorkerSupervisorV3,
    recovery: &mut PcRootRecoveryV3,
    slot_owner: &Arc<Mutex<PcOperationSlotV3>>,
) -> PcScopeDriveV3 {
    use super::game_authority_v3::GameStepV3 as G;
    let mut slot = match slot_owner.try_lock() {
        Ok(slot) => slot,
        Err(std::sync::TryLockError::WouldBlock) => return PcScopeDriveV3::Pending,
        Err(_) => return PcScopeDriveV3::Failed,
    };
    let game = supervisor.bridge.game_authority();
    let result = with_preserved_pc_route_owners_v3(
        &mut *slot,
        |slot| (&mut slot.proof, &mut slot.claim),
        |slot, proof, claim| {
            let Some(proof) = proof else {
                return PcScopeDriveV3::Failed;
            };
            let mut newly_claimed = false;
            if claim.is_none() {
                match game.try_claim_pc_operation(proof, slot, supervisor, recovery, claim) {
                    G::Ready => newly_claimed = true,
                    G::Busy => return PcScopeDriveV3::Pending,
                    G::Aborted | G::Closed => return PcScopeDriveV3::Aborted,
                    G::RootCanceled => return PcScopeDriveV3::RootCanceled,
                    _ => return PcScopeDriveV3::Failed,
                }
            }
            let Some(original) = claim.as_mut() else {
                return PcScopeDriveV3::Failed;
            };
            if newly_claimed {
                if slot.route_destination.is_some()
                    || slot.route_result.is_some()
                    || slot.route_unknown
                {
                    return PcScopeDriveV3::Failed;
                }
                slot.generation = Some(original.generation());
                let registration = slot.registration.clone();
                let admission = slot.admission.clone();
                let mailbox = supervisor.bridge.inputs().plan().teacher_mailbox_capacity as usize;
                let tickets = supervisor.bridge.inputs().plan().teacher_tickets_per_root as usize;
                let producer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    registration.try_install_speculative_root_into_v3(
                        &admission,
                        mailbox,
                        tickets,
                        Arc::new(AtomicBool::new(false)),
                        &mut slot.route_destination,
                    )
                }));
                match producer {
                    Ok(result) => slot.route_result = Some(result),
                    Err(payload) => {
                        slot.route_unknown = true;
                        retain_pc_panic_v3(&mut recovery.panic, payload);
                        recovery.failure.get_or_insert_with(|| {
                            "P-C V3 route installation outcome unknown".into()
                        });
                        return PcScopeDriveV3::Pending;
                    }
                }
            }
            match original.try_complete_pc_operation(proof, slot, supervisor, recovery) {
                G::Ready => {}
                G::Busy | G::Unknown => return PcScopeDriveV3::Pending,
                _ => return PcScopeDriveV3::Failed,
            }
            // Drop only an already settled claim. Keep the producer result until
            // it is transferred or explicitly recorded as a failure.
            *claim = None;
            match slot.route_result.as_ref() {
                Some(Ok(None)) => {
                    slot.route_result = None;
                    slot.generation = None;
                    PcScopeDriveV3::Pending
                }
                Some(Err(error)) => {
                    recovery
                        .failure
                        .get_or_insert_with(|| format!("P-C V3 route installation: {error}"));
                    PcScopeDriveV3::Failed
                }
                Some(Ok(Some(()))) => {
                    let Some((route, receiver)) = slot.route_destination.as_ref() else {
                        return PcScopeDriveV3::Failed;
                    };
                    if recovery.route.is_some() || recovery.receiver.is_some() {
                        slot.route_unknown = true;
                        recovery.failure.get_or_insert_with(|| {
                            "P-C V3 route recovery destination already occupied".into()
                        });
                        return PcScopeDriveV3::Pending;
                    }
                    // Clone the original owners while the producer slot still
                    // retains them. If anything unwinds before both assignments,
                    // the original receiver remains accessible in this slot.
                    let route = route.clone();
                    let receiver = Arc::clone(receiver);
                    recovery.route = Some(route);
                    recovery.receiver = Some(receiver);
                    slot.route_destination = None;
                    PcScopeDriveV3::Finished
                }
                None => PcScopeDriveV3::Failed,
            }
        },
    );
    match result {
        Ok(step) => step,
        Err(payload) => {
            slot.route_unknown = true;
            retain_pc_panic_v3(&mut recovery.panic, payload);
            recovery
                .failure
                .get_or_insert_with(|| "P-C V3 route effect outcome unknown".into());
            PcScopeDriveV3::Pending
        }
    }
}

/// Called outside the native manager callback. Busy leaves the same recovery
/// owner in place; no new actor or claim is issued for another root.
fn prepare_root_step_v3(
    supervisor: &PcWorkerSupervisorV3,
    bridge: &Arc<RoventPcCollectionBridgeV3>,
    router: &Arc<CaptureRouterV3>,
    recovery: &Arc<Mutex<PcRootRecoveryV3>>,
    startup_deadline: Instant,
    cleanup_deadline: Instant,
) -> Result<PcPrepareStepV3, String> {
    let admission = recovery
        .lock()
        .map_err(|_| "P-C V3 recovery poisoned")?
        .admission
        .clone();
    // Startup windows are relative to each original root clock, not to the
    // first root or a previously observed global duration.
    let root = admission.root_admission();
    let origin = root.runtime_clock().origin();
    let root_startup = origin
        .checked_add(Duration::from_nanos(root.admission_deadline().at.0))
        .ok_or("P-C V3 root startup deadline overflow")?;
    let root_cleanup = origin
        .checked_add(Duration::from_nanos(root.cleanup_deadline().at.0))
        .ok_or("P-C V3 root cleanup deadline overflow")?;
    let startup_deadline = startup_deadline.min(root_startup);
    let cleanup_deadline = cleanup_deadline.min(root_cleanup);
    if startup_deadline > cleanup_deadline {
        return Err("P-C V3 root deadline ordering invalid".into());
    }
    if !bridge
        .try_publish_recovery(&admission, recovery)
        .map_err(str::to_owned)?
    {
        return Ok(PcPrepareStepV3::Pending);
    }
    let mut state = recovery.lock().map_err(|_| "P-C V3 recovery poisoned")?;
    if !state.matches_received_root(bridge) {
        return Err("P-C V3 original receiver witness missing or changed".into());
    }
    // Root cancellation is not a game abort and is only a request. Original
    // in-flight effects still have to be settled before any root cleanup.
    let game = bridge.game_authority();
    let root_cancel_requested = if game.abort_requested() {
        // Game-wide abort has its own authority and cleanup proofs; an
        // unbound root gate must not conceal that separate request.
        false
    } else {
        match game.try_observe_pc_root_cancel(&state.admission) {
            super::game_authority_v3::GameStepV3::Ready => false,
            super::game_authority_v3::GameStepV3::RootCanceled => true,
            super::game_authority_v3::GameStepV3::Busy if game.abort_requested() => false,
            super::game_authority_v3::GameStepV3::Busy => return Ok(PcPrepareStepV3::Pending),
            _ => return Err("P-C V3 original root cancel observation unresolved".into()),
        }
    };
    // Settle an already-issued worker operation before failure or abort can
    // decide this root. No new task factory is available on reobservation.
    if let Some(worker_slot) = state.operation_slots[1].as_ref().cloned() {
        let active = match worker_slot.try_lock() {
            Ok(slot) => slot.claim.is_some(),
            Err(std::sync::TryLockError::WouldBlock) => return Ok(PcPrepareStepV3::Pending),
            Err(_) => return Err("P-C V3 worker operation slot poisoned".into()),
        };
        if active {
            use pc_worker_operation_v3::PcWorkerEffectDriveV3 as W;
            let result = pc_worker_operation_v3::drive_pc_worker_effect_v3(
                supervisor,
                &mut state,
                &worker_slot,
                None::<fn() -> RoventPcRootWorkerTaskV3>,
            );
            return match result {
                W::DropUnstartedTask(task) => {
                    drop(state);
                    finish_failed_pc_task_release_v3(recovery, task)?;
                    Ok(PcPrepareStepV3::Pending)
                }
                W::Failed => {
                    state
                        .failure
                        .get_or_insert_with(|| "P-C V3 worker acquisition unresolved".into());
                    Ok(PcPrepareStepV3::Failed)
                }
                W::Pending | W::Finished | W::Aborted => Ok(PcPrepareStepV3::Pending),
            };
        }
    }
    // Settle an existing route claim even if abort has not yet been observed.
    // Otherwise abort may arrive between two observations while the original
    // route and receiver are still only in the claim's result slot, causing
    // this root to be incorrectly classified as pre-route.
    if let Some(route_slot) = state.operation_slots[0].as_ref().cloned() {
        let active = match route_slot.try_lock() {
            Ok(slot) => slot.claim.is_some(),
            Err(std::sync::TryLockError::WouldBlock) => return Ok(PcPrepareStepV3::Pending),
            Err(_) => return Err("P-C V3 route operation slot poisoned".into()),
        };
        if active {
            if matches!(
                drive_pc_route_effect_v3(supervisor, &mut state, &route_slot),
                PcScopeDriveV3::Failed
            ) {
                state
                    .failure
                    .get_or_insert_with(|| "P-C V3 active route outcome unresolved".into());
            }
            return Ok(PcPrepareStepV3::Pending);
        }
    }
    // Claims acquired before either abort or a preparation failure retain
    // their original authority until their own result has been settled.
    if bridge.game_authority().abort_requested() || root_cancel_requested || state.failure.is_some()
    {
        let active = match state.scope_effects.try_lock() {
            Ok(slots) => slots.claim.is_some() && !slots.finished,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(PcPrepareStepV3::Pending),
            Err(_) => return Err("P-C V3 scope slots poisoned".into()),
        };
        if active {
            let Some(stage_owner) = state.stage.as_ref().map(Arc::clone) else {
                return Err("P-C V3 active scope claim lost stage owner".into());
            };
            let mut stage = match stage_owner.try_lock() {
                Ok(stage) => stage,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(PcPrepareStepV3::Pending),
                Err(_) => return Err("P-C V3 stage poisoned".into()),
            };
            if matches!(
                drive_pc_scope_effect_v3(supervisor, &mut state, &stage_owner, &mut stage),
                PcScopeDriveV3::Failed
            ) {
                state
                    .failure
                    .get_or_insert_with(|| "P-C V3 active scope outcome unresolved".into());
            }
            return Ok(PcPrepareStepV3::Pending);
        }
        if let Some(actor_slot) = state.actor_effects.get(state.next_actor).cloned() {
            let active = match actor_slot.try_lock() {
                Ok(slots) => slots.claim.is_some() && !slots.finished,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(PcPrepareStepV3::Pending),
                Err(_) => return Err("P-C V3 actor slots poisoned".into()),
            };
            if active {
                let Some(stage_owner) = state.stage.as_ref().map(Arc::clone) else {
                    return Err("P-C V3 active actor lost stage owner".into());
                };
                let stage = match stage_owner.try_lock() {
                    Ok(stage) => stage,
                    Err(std::sync::TryLockError::WouldBlock) => {
                        return Ok(PcPrepareStepV3::Pending);
                    }
                    Err(_) => return Err("P-C V3 stage poisoned".into()),
                };
                let Some(session_owner) = stage.session() else {
                    return Err("P-C V3 active actor lost session owner".into());
                };
                let mut session = match session_owner.try_lock() {
                    Ok(session) => session,
                    Err(std::sync::TryLockError::WouldBlock) => {
                        return Ok(PcPrepareStepV3::Pending);
                    }
                    Err(_) => return Err("P-C V3 session poisoned".into()),
                };
                match drive_pc_actor_effect_v3(
                    supervisor,
                    &mut state,
                    &stage_owner,
                    &stage,
                    &session_owner,
                    &mut session,
                    &actor_slot,
                    startup_deadline,
                    cleanup_deadline,
                ) {
                    PcScopeDriveV3::Finished => state.next_actor += 1,
                    PcScopeDriveV3::Failed => {
                        state
                            .failure
                            .get_or_insert_with(|| "P-C V3 active actor outcome unresolved".into());
                    }
                    PcScopeDriveV3::Pending
                    | PcScopeDriveV3::Aborted
                    | PcScopeDriveV3::RootCanceled => {}
                }
                return Ok(PcPrepareStepV3::Pending);
            }
        }
    }
    if root_cancel_requested && !bridge.game_authority().abort_requested() {
        if let Some(route) = state.route.as_ref() {
            route.cancel_all();
            match route.try_revoke_all_targets() {
                Ok(Some(())) => {}
                Ok(None) => return Ok(PcPrepareStepV3::Pending),
                Err(error) => return Err(format!("P-C V3 root target revocation failed: {error}")),
            }
        }
        // Only a never-published pre-route root may use the separate
        // root-local pre-route retirement. Live route/worker custody needs its
        // own physical drain and cannot be converted to a no-scope result.
        if state.route.is_some()
            || state.worker.is_some()
            || state.worker_key.is_some()
            || state.prepared.is_some()
            || state.published
        {
            return Ok(PcPrepareStepV3::Pending);
        }
        match bridge
            .game_authority()
            .try_observe_pc_root_cancel_quiescent(&state.admission)
        {
            super::game_authority_v3::GameStepV3::Ready => {}
            super::game_authority_v3::GameStepV3::Busy => return Ok(PcPrepareStepV3::Pending),
            _ => return Err("P-C V3 original root cancel quiescence unresolved".into()),
        }
    }
    if state.published {
        return Ok(PcPrepareStepV3::Published);
    }
    if (bridge.game_authority().abort_requested() || root_cancel_requested)
        && state.route.is_none()
        && state.worker.is_none()
        && state.worker_key.is_none()
        && state.prepared.is_none()
    {
        state.abort_before_route = true;
    }
    if state.abort_before_route {
        if state.aborted_unclaimed_stage_release.is_some() {
            return Ok(PcPrepareStepV3::Pending);
        }
        if state.stage.is_none()
            && state.panic.is_none()
            && state.aborted_no_scope_release.is_none()
        {
            // No stage is invented here. The bridge separately proves the
            // original one-shot no-scope receipt and exact quota ownership.
            drop(state);
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                bridge.try_release_aborted_no_scope(&admission, recovery)
            }));
            let mut state = recovery.lock().map_err(|_| "P-C V3 recovery poisoned")?;
            match outcome {
                Ok(Ok(Some(true))) => state.aborted_no_scope_release = Some(Ok(())),
                Ok(Ok(None)) => {}
                Ok(Ok(Some(false))) => {
                    let error = "P-C V3 no-scope abort release rejected";
                    state.aborted_no_scope_release = Some(Err(error));
                    state.failure.get_or_insert_with(|| error.into());
                }
                Ok(Err(error)) => {
                    state.aborted_no_scope_release = Some(Err(error));
                    state.failure.get_or_insert_with(|| error.into());
                }
                Err(payload) => {
                    state.panic = Some(payload);
                    state
                        .failure
                        .get_or_insert_with(|| "P-C V3 no-scope abort release panicked".into());
                    state.aborted_no_scope_release =
                        Some(Err("P-C V3 no-scope abort release panicked"));
                }
            }
            return Ok(PcPrepareStepV3::Pending);
        }
        // A published scope source is distinct from an entered scope claim.
        // Only its original sealed proof permits the bridge's no-scope ACK.
        if state.stage.is_some() && state.panic.is_none() {
            let sealed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                state.try_seal_aborted_unclaimed_stage()
            }));
            match sealed {
                Ok(Ok(None)) => return Ok(PcPrepareStepV3::Pending),
                Ok(Ok(Some(false))) => {}
                Ok(Err(error)) => {
                    state.aborted_unclaimed_stage_release = Some(Err(error));
                    state.failure.get_or_insert_with(|| error.into());
                    return Ok(PcPrepareStepV3::Pending);
                }
                Err(payload) => {
                    let error = "P-C V3 unclaimed-stage proof panicked";
                    state.panic = Some(payload);
                    state.aborted_unclaimed_stage_release = Some(Err(error));
                    state.failure.get_or_insert_with(|| error.into());
                    return Ok(PcPrepareStepV3::Pending);
                }
                Ok(Ok(Some(true))) => {
                    // The bridge re-locks this exact recovery and retains its
                    // own one-shot ACK; do not hold the recovery guard here.
                    drop(state);
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        bridge.try_release_aborted_unclaimed_stage(&admission, recovery)
                    }));
                    let mut state = recovery.lock().map_err(|_| "P-C V3 recovery poisoned")?;
                    match outcome {
                        Ok(Ok(Some(true))) => state.aborted_unclaimed_stage_release = Some(Ok(())),
                        Ok(Ok(None)) => {}
                        Ok(Ok(Some(false))) => {
                            let error = "P-C V3 unclaimed-stage abort release rejected";
                            state.aborted_unclaimed_stage_release = Some(Err(error));
                            state.failure.get_or_insert_with(|| error.into());
                        }
                        Ok(Err(error)) => {
                            state.aborted_unclaimed_stage_release = Some(Err(error));
                            state.failure.get_or_insert_with(|| error.into());
                        }
                        Err(payload) => {
                            let error = "P-C V3 unclaimed-stage abort release panicked";
                            state.panic = Some(payload);
                            state.aborted_unclaimed_stage_release = Some(Err(error));
                            state.failure.get_or_insert_with(|| error.into());
                        }
                    }
                    return Ok(PcPrepareStepV3::Pending);
                }
            }
        }
        // Preserve the exact session and recovery custodian. A claim or spawn
        // with an unknown outcome is never promoted to a no-worker result.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.try_drain_teacher_without_live_route(cleanup_deadline)
        }));
        let release = match result {
            Ok(ready) => {
                ready && state.panic.is_none() && state.aborted_pre_route_release.is_none()
            }
            Err(payload) => {
                state.panic = Some(payload);
                state
                    .failure
                    .get_or_insert_with(|| "P-C V3 pre-route abort drain panicked".into());
                false
            }
        };
        // The hook reacquires this exact recovery. Never call it while the
        // original guard is held; Busy keeps both the aggregate and owner.
        drop(state);
        if release {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                bridge.try_release_aborted_pre_route(&admission, recovery)
            }));
            let mut state = recovery.lock().map_err(|_| "P-C V3 recovery poisoned")?;
            match outcome {
                Ok(Ok(Some(true))) => state.aborted_pre_route_release = Some(Ok(())),
                Ok(Ok(None)) => {}
                Ok(Ok(Some(false))) => {
                    let error = "P-C V3 pre-route release rejected";
                    state.aborted_pre_route_release = Some(Err(error));
                    state.failure.get_or_insert_with(|| error.into());
                }
                Ok(Err(error)) => {
                    state.aborted_pre_route_release = Some(Err(error));
                    state.failure.get_or_insert_with(|| error.into());
                }
                Err(payload) => {
                    state.panic = Some(payload);
                    state
                        .failure
                        .get_or_insert_with(|| "P-C V3 pre-route release panicked".into());
                    state.aborted_pre_route_release =
                        Some(Err("P-C V3 pre-route release panicked"));
                }
            }
        }
        return Ok(PcPrepareStepV3::Pending);
    }
    if bridge.game_authority().abort_requested()
        && state.route.is_some()
        && state.receiver.is_some()
        && state.owner.is_none()
        && state.pump.is_none()
        && state.consumer.is_none()
        && state.worker.is_none()
        && state.worker_key.is_none()
        && state.prepared.is_none()
        && !state.published
    {
        if state.aborted_post_route_release.is_some() {
            return Ok(PcPrepareStepV3::Pending);
        }
        match state.try_abort_unstarted_route(bridge.capture_registration()) {
            Ok(Some(true)) => {}
            Ok(None) => return Ok(PcPrepareStepV3::Pending),
            Ok(Some(false)) => {
                state
                    .failure
                    .get_or_insert_with(|| "P-C V3 post-route abort custody unresolved".into());
                return Ok(PcPrepareStepV3::Pending);
            }
            Err(error) => {
                state.failure.get_or_insert_with(|| error.into());
                return Ok(PcPrepareStepV3::Pending);
            }
        }
        if !state.try_drain_teacher_without_live_route(cleanup_deadline) {
            return Ok(PcPrepareStepV3::Pending);
        }
        // The bridge verifies separate post-route custody and the real
        // teacher aggregate before releasing only this root's quota.
        drop(state);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bridge.try_release_aborted_post_route(&admission, recovery)
        }));
        let mut state = recovery.lock().map_err(|_| "P-C V3 recovery poisoned")?;
        match outcome {
            Ok(Ok(Some(true))) => state.aborted_post_route_release = Some(Ok(())),
            Ok(Ok(None)) => {}
            Ok(Ok(Some(false))) => {
                let error = "P-C V3 post-route release rejected";
                state.aborted_post_route_release = Some(Err(error));
                state.failure.get_or_insert_with(|| error.into());
            }
            Ok(Err(error)) => {
                state.aborted_post_route_release = Some(Err(error));
                state.failure.get_or_insert_with(|| error.into());
            }
            Err(payload) => {
                state.panic = Some(payload);
                state.aborted_post_route_release = Some(Err("P-C V3 post-route release panicked"));
            }
        }
        return Ok(PcPrepareStepV3::Pending);
    }
    if bridge.game_authority().abort_requested() {
        // A route or worker may have appeared while abort raced preparation.
        // Do not start another one or invent the missing native handoff.
        if state.panic.is_some() {
            return Ok(PcPrepareStepV3::Pending);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.request_route_abort(cleanup_deadline)
        }));
        if let Err(payload) = result {
            state.panic = Some(payload);
            state
                .failure
                .get_or_insert_with(|| "P-C V3 route abort request panicked".into());
        }
        return Ok(PcPrepareStepV3::Pending);
    }
    if state.failure.is_some() {
        let unstarted =
            state.definitely_unstarted() && state.no_scope_recovery == NoScopeRecoveryV3::Pending;
        let claimed = if !unstarted && state.claimed_unstarted_release == NoScopeRecoveryV3::Pending
        {
            match state.try_claimed_unstarted() {
                ClaimedUnstartedPollV3::Confirmed | ClaimedUnstartedPollV3::Pending => true,
                ClaimedUnstartedPollV3::Unknown => {
                    state.claimed_unstarted_release = NoScopeRecoveryV3::Unknown;
                    false
                }
                ClaimedUnstartedPollV3::Ineligible => false,
            }
        } else {
            false
        };
        drop(state);
        if unstarted {
            let result = bridge.try_release_unstarted_teacher(&admission, recovery);
            let mut state = recovery.lock().map_err(|_| "P-C V3 recovery poisoned")?;
            match result {
                Ok(Some(true)) => state.no_scope_recovery = NoScopeRecoveryV3::Complete,
                Ok(None) => {}
                Ok(Some(false)) | Err(_) => state.no_scope_recovery = NoScopeRecoveryV3::Unknown,
            }
        }
        if claimed {
            let result = bridge.try_release_claimed_unstarted_teacher(&admission, recovery);
            let mut state = recovery.lock().map_err(|_| "P-C V3 recovery poisoned")?;
            match result {
                Ok(Some(true)) => state.claimed_unstarted_release = NoScopeRecoveryV3::Complete,
                Ok(None) => {}
                Ok(Some(false)) | Err(_) => {
                    state.claimed_unstarted_release = NoScopeRecoveryV3::Unknown
                }
            }
        }
        return Ok(PcPrepareStepV3::Failed);
    }
    if state.sealed_unstarted || state.sealed_claimed_unstarted {
        record_first_root_failure(&mut state.failure, "P-C V3 unstarted root was sealed");
        return Ok(PcPrepareStepV3::Failed);
    }
    if !router
        .registration()
        .same_original_owner(&bridge.capture_registration())
    {
        record_first_root_failure(
            &mut state.failure,
            "P-C V3 capture router differs from bridge",
        );
        return Ok(PcPrepareStepV3::Failed);
    }
    if !state.teacher_reserved {
        let result = bridge.try_reserve_teacher(&admission);
        // Split mutable borrows of fields through the original recovery guard.
        let state = &mut *state;
        if let Some(step) =
            apply_teacher_reservation_v3(result, &mut state.teacher_reserved, &mut state.failure)
        {
            return Ok(step);
        }
    }
    let inputs = bridge.inputs();
    let Some(startup) = bridge.startup() else {
        return Ok(PcPrepareStepV3::Pending);
    };
    if state.stage.is_none() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            RoventCollectionTeacherSessionV3::prepare_staged(
                &admission,
                startup,
                inputs.manifest(),
                inputs.roster(),
            )
        }));
        let stage = match result {
            Ok(Ok(stage)) => stage,
            Ok(Err(error)) => {
                state.preflight_failure = Some(error);
                record_first_root_failure(
                    &mut state.failure,
                    "P-C V3 teacher stage preparation failed before claim",
                );
                return Ok(PcPrepareStepV3::Failed);
            }
            Err(payload) => {
                state.panic = Some(payload);
                record_first_root_failure(
                    &mut state.failure,
                    "P-C V3 teacher preflight panicked; ownership unknown",
                );
                return Ok(PcPrepareStepV3::Failed);
            }
        };
        state.original_scope_identity = Some(stage.scope_effect_identity_weak());
        state.stage = Some(Arc::new(Mutex::new(stage)));
    }
    let stage = Arc::clone(state.stage.as_ref().expect("stage retained"));
    {
        let mut staged = stage.lock().map_err(|_| "P-C V3 stage poisoned")?;
        use super::teacher::v3_session::TeacherSessionStageStateV3 as S;
        let scope_pending = match state.scope_effects.try_lock() {
            Ok(slots) => slots.claim.is_some() && !slots.finished,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(PcPrepareStepV3::Pending),
            Err(_) => return Err("P-C V3 scope slots poisoned".into()),
        };
        if staged.state() == S::Prepared || scope_pending {
            match drive_pc_scope_effect_v3(supervisor, &mut state, &stage, &mut staged) {
                PcScopeDriveV3::Finished => {}
                PcScopeDriveV3::Pending
                | PcScopeDriveV3::Aborted
                | PcScopeDriveV3::RootCanceled => {
                    return Ok(PcPrepareStepV3::Pending);
                }
                PcScopeDriveV3::Failed => {
                    state.failure.get_or_insert_with(|| {
                        "P-C V3 teacher scope effect failed or unresolved".into()
                    });
                    return Ok(PcPrepareStepV3::Failed);
                }
            }
        }
        match staged.state() {
            S::Prepared => return Ok(PcPrepareStepV3::Pending),
            S::Claimed => {}
            S::Ready => {}
            _ => {
                record_first_root_failure(&mut state.failure, "teacher stage is unresolved");
                return Ok(PcPrepareStepV3::Failed);
            }
        }
        if staged.state() == S::Claimed && staged.finish().is_err() {
            record_first_root_failure(&mut state.failure, "teacher session assembly failed");
            return Ok(PcPrepareStepV3::Failed);
        }
    }
    let session = stage
        .lock()
        .map_err(|_| "P-C V3 stage poisoned")?
        .session()
        .ok_or("P-C V3 staged session absent")?;
    let roster = &inputs.plan().teacher_slots;
    if state.next_actor < roster.len() {
        if state.actor_effects.len() == state.next_actor {
            if state.actor_effects.try_reserve(1).is_err() {
                return Ok(PcPrepareStepV3::Pending);
            }
            let actor = roster[state.next_actor].actor_id.clone();
            state
                .actor_effects
                .push(Arc::new(Mutex::new(PcActorEffectSlotsV3::new(actor))));
        }
        let Some(actor_slot) = state.actor_effects.get(state.next_actor).cloned() else {
            return Err("P-C V3 original actor slot absent".into());
        };
        let stage_guard = match stage.try_lock() {
            Ok(stage) => stage,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(PcPrepareStepV3::Pending),
            Err(_) => return Err("P-C V3 stage poisoned".into()),
        };
        let mut session_guard = match session.try_lock() {
            Ok(session) => session,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(PcPrepareStepV3::Pending),
            Err(_) => return Err("P-C V3 session poisoned".into()),
        };
        match drive_pc_actor_effect_v3(
            supervisor,
            &mut state,
            &stage,
            &stage_guard,
            &session,
            &mut session_guard,
            &actor_slot,
            startup_deadline,
            cleanup_deadline,
        ) {
            PcScopeDriveV3::Finished => state.next_actor += 1,
            PcScopeDriveV3::Pending | PcScopeDriveV3::Aborted | PcScopeDriveV3::RootCanceled => {}
            PcScopeDriveV3::Failed => {
                state
                    .failure
                    .get_or_insert_with(|| "P-C V3 actor effect failed or unresolved".into());
                return Ok(PcPrepareStepV3::Failed);
            }
        }
        return Ok(PcPrepareStepV3::Pending);
    }
    if state.route.is_none() {
        if state.operation_slots[0].is_none() {
            let stage_guard = match stage.try_lock() {
                Ok(stage) => stage,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(PcPrepareStepV3::Pending),
                Err(_) => return Err("P-C V3 stage poisoned".into()),
            };
            let proof = match PcReceivedStageProofV3::try_from_guards(
                supervisor,
                &state,
                &stage,
                &stage_guard,
            ) {
                Ok(Some(proof)) => proof,
                Ok(None) => return Ok(PcPrepareStepV3::Pending),
                Err(error) => return Err(error.into()),
            };
            let identity = Arc::clone(&state.operation_identities[0]);
            let admission = admission.clone();
            let recovery = state.original_self.clone();
            let registration = bridge.capture_registration().clone();
            state.operation_slots[0] = Some(Arc::new_cyclic(|original_self| {
                Mutex::new(PcOperationSlotV3 {
                    original_self: original_self.clone(),
                    kind: PcOperationKindV3::RouteInstall,
                    identity,
                    admission,
                    recovery,
                    registration,
                    proof: Some(proof),
                    claim: None,
                    generation: None,
                    route_destination: None,
                    route_result: None,
                    route_unknown: false,
                    worker_destination: None,
                    worker_undispatched: None,
                    worker_failure_details: None,
                    worker_unknown: false,
                })
            }));
        }
        let route_slot = Arc::clone(
            state.operation_slots[0]
                .as_ref()
                .expect("original route slot retained"),
        );
        return match drive_pc_route_effect_v3(supervisor, &mut state, &route_slot) {
            PcScopeDriveV3::Finished
            | PcScopeDriveV3::Pending
            | PcScopeDriveV3::Aborted
            | PcScopeDriveV3::RootCanceled => Ok(PcPrepareStepV3::Pending),
            PcScopeDriveV3::Failed => {
                state
                    .failure
                    .get_or_insert_with(|| "P-C V3 route effect failed or unresolved".into());
                Ok(PcPrepareStepV3::Failed)
            }
        };
    }
    if state.owner.is_none() {
        state.owner = Some(
            RoventCollectionOwnerLockV3::new_shared(
                Arc::clone(inputs.manifest()),
                inputs.plan().metadata.clone(),
                Arc::clone(router),
            )
            .map_err(|e| e.to_string())?,
        );
    }
    if state.pump.is_none() {
        let receiver = Arc::clone(state.receiver.as_ref().expect("receiver retained"));
        state.pump_setup_unknown = true;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            TeacherPumpV3::new(
                Arc::clone(&session),
                receiver,
                inputs.plan().teacher_constraint,
                Duration::from_nanos(inputs.plan().teacher_soft_quantum_ns),
            )
        }));
        match result {
            Ok(Ok(pump)) => {
                state.pump = Some(pump);
                state.pump_setup_unknown = false;
            }
            Ok(Err(error)) => {
                state.pump_setup_failure = Some(error);
                state.pump_setup_unknown = false;
                record_first_root_failure(
                    &mut state.failure,
                    "teacher pump setup failed; original owners retained",
                );
                return Ok(PcPrepareStepV3::Failed);
            }
            Err(payload) => {
                state.panic = Some(payload);
                record_first_root_failure(
                    &mut state.failure,
                    "teacher pump setup unknown after panic",
                );
                return Ok(PcPrepareStepV3::Failed);
            }
        }
    }
    if state.closed_custody.is_none() {
        if state.closed_custody_setup_unknown {
            return Ok(PcPrepareStepV3::Pending);
        }
        let pump = Arc::clone(state.pump.as_ref().expect("pump retained"));
        let route = state.route.as_ref().expect("route retained").clone();
        state.closed_custody_setup_unknown = true;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            TeacherClosedCustodyV3::try_new(
                &pump,
                &route,
                inputs.plan().teacher_tickets_per_root as usize,
            )
        }));
        match result {
            Ok(Ok(TeacherClosedCustodyStartV3::Ready(custody))) => {
                state.closed_custody = Some(custody);
                state.closed_custody_setup_unknown = false;
            }
            Ok(Ok(TeacherClosedCustodyStartV3::Busy)) => {
                state.closed_custody_setup_unknown = false;
                return Ok(PcPrepareStepV3::Pending);
            }
            Ok(Err(error)) => {
                state.closed_custody_setup_unknown = false;
                record_first_root_failure(
                    &mut state.failure,
                    format!("P-C V3 closed custody setup: {error}"),
                );
                return Ok(PcPrepareStepV3::Failed);
            }
            Err(payload) => {
                state.panic = Some(payload);
                record_first_root_failure(
                    &mut state.failure,
                    "P-C V3 closed custody setup outcome unknown",
                );
                return Ok(PcPrepareStepV3::Failed);
            }
        }
    }
    if state.processed.is_none() {
        state.processed = Some(
            TeacherProcessedOwnerV3::new(
                admission.clone(),
                inputs.plan().teacher_tickets_per_root as usize,
            )
            .map_err(|e| e.to_string())?,
        );
    }
    if state.retained.is_none() {
        state.retained = match PolicyRootRetainedOwnerV3::try_new(
            &state.pipeline,
            admission.clone(),
            inputs.plan().teacher_tickets_per_root as usize,
        )
        .map_err(str::to_owned)?
        {
            Some(owner) => Some(owner),
            None => return Ok(PcPrepareStepV3::Pending),
        };
    }
    if state.consumer.is_none() {
        let owner = state.owner.as_ref().expect("owner retained").clone();
        let route = state.route.as_ref().expect("route retained").clone();
        match owner
            .try_attach_speculative_root(&admission, &route)
            .map_err(|e| e.to_string())?
        {
            Some(()) => {}
            None => return Ok(PcPrepareStepV3::Pending),
        }
        let pump = Arc::clone(state.pump.as_ref().expect("pump retained"));
        let pipeline = state.pipeline.clone();
        let processed = state
            .processed
            .as_ref()
            .expect("processed owner retained")
            .clone();
        let retained = state
            .retained
            .as_ref()
            .expect("root policy owner retained")
            .clone();
        if !state.output.same_owner(&pipeline) {
            record_first_root_failure(&mut state.failure, "P-C V3 output/pipeline owner differs");
            return Ok(PcPrepareStepV3::Failed);
        }
        state.consumer_setup_unknown = true;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            TeacherPolicyConsumerOwnerV3::try_new(
                &session, &pump, &route, &owner, &pipeline, &processed, &retained,
            )
        }));
        match result {
            Ok(Ok(TeacherPolicyConsumerStartV3::Ready(consumer))) => {
                state.consumer = Some(consumer);
                state.consumer_setup_unknown = false;
            }
            Ok(Ok(TeacherPolicyConsumerStartV3::Busy | TeacherPolicyConsumerStartV3::NotReady)) => {
                state.consumer_setup_unknown = false;
                return Ok(PcPrepareStepV3::Pending);
            }
            Ok(Err(error)) => {
                state.consumer_setup_unknown = false;
                record_first_root_failure(
                    &mut state.failure,
                    format!("P-C V3 consumer setup: {error}"),
                );
                return Ok(PcPrepareStepV3::Failed);
            }
            Err(payload) => {
                state.panic = Some(payload);
                record_first_root_failure(&mut state.failure, "P-C V3 consumer setup unknown");
                return Ok(PcPrepareStepV3::Failed);
            }
        }
    }
    {
        let consumer = state.consumer.as_ref().expect("consumer retained");
        match consumer.try_matches_registered_owners(
            &session,
            state.pump.as_ref().expect("pump retained"),
            state.route.as_ref().expect("route retained"),
            state.owner.as_ref().expect("owner retained"),
            &state.pipeline,
            state.processed.as_ref().expect("processed owner retained"),
            state.retained.as_ref().expect("root policy owner retained"),
        ) {
            Ok(Some(true)) => {}
            Ok(None) => return Ok(PcPrepareStepV3::Pending),
            _ => {
                record_first_root_failure(
                    &mut state.failure,
                    "P-C V3 consumer original owners differ",
                );
                return Ok(PcPrepareStepV3::Failed);
            }
        }
    }
    // Root worker construction and publication are deliberately separate from
    // claiming the teacher scope; failure must keep the original owner.
    if state.worker.is_none() {
        let Some(worker_slot) = pc_worker_operation_v3::ensure_pc_worker_operation_slot_v3(
            supervisor, &mut state, &stage,
        )
        .map_err(str::to_owned)?
        else {
            return Ok(PcPrepareStepV3::Pending);
        };
        let original = Arc::clone(recovery);
        let game = Arc::clone(bridge.game_authority());
        // This factory captures only reconstructible owner Arcs. The actual
        // one-shot task is not constructed until the abort claim succeeds.
        let task_factory = move || -> RoventPcRootWorkerTaskV3 {
            Box::new(move |receiver: rz_uci::rovent_bootstrap::pc_collection_v3::RoventPcRootHandoffReceiverV3| {
            let mut handoff = None;
            while Instant::now() < cleanup_deadline {
                if let Ok(mut owner) = original.try_lock() {
                    if game.abort_requested() && owner.panic.is_none() {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            owner.request_route_abort(cleanup_deadline)
                        }));
                        if let Err(payload) = result {
                            owner.panic = Some(payload);
                            owner.failure.get_or_insert_with(|| "P-C V3 route abort request panicked".into());
                        }
                    }
                    if let Some(pump) = owner.pump.as_ref().map(Arc::clone) {
                        let owner = &mut *owner;
                        if let Some(mut pump) = try_teacher_recovery_lock(
                            &pump,
                            &mut owner.failure,
                            &mut owner.teacher_drain_failure,
                            "pump",
                        ) {
                            let result = pump.try_pump();
                            record_teacher_pump_result(&mut owner.failure, &mut owner.teacher_drain_failure, result);
                        };
                    }
                    owner.drive_consumer_step();
                }
                match receiver.recv_timeout(Duration::from_millis(10)) {
                    Ok(Some(value)) => { handoff = Some(value); break; }
                    Ok(None) => {},
                    Err(_) => break,
                }
            }
            if let Some(handoff) = handoff {
                if let Ok(mut owner) = original.lock() {
                    owner.handoff = Some(Arc::clone(&handoff));
                    if let Some(route) = &owner.route { route.cancel_all(); }
                    if let Some(pump) = &owner.pump { if let Ok(mut pump) = pump.lock() { pump.close_admission(); } }
                }
                while Instant::now() < cleanup_deadline {
                    let completed = match original.try_lock() {
                        Ok(mut owner) => {
                            if game.abort_requested() && owner.panic.is_none() {
                                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    owner.request_route_abort(cleanup_deadline)
                                }));
                                if let Err(payload) = result {
                                    owner.panic = Some(payload);
                                    owner.failure.get_or_insert_with(|| "P-C V3 route abort request panicked".into());
                                }
                            }
                            owner.try_drain_teacher(&handoff, cleanup_deadline)
                        },
                        Err(_) => false,
                    };
                    if completed { break; }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            Box::new(original) as Box<dyn std::any::Any + Send>
        })
        };
        use pc_worker_operation_v3::PcWorkerEffectDriveV3 as W;
        let result = pc_worker_operation_v3::drive_pc_worker_effect_v3(
            supervisor,
            &mut state,
            &worker_slot,
            Some(task_factory),
        );
        match result {
            W::Finished => {}
            W::Pending | W::Aborted => return Ok(PcPrepareStepV3::Pending),
            W::Failed => {
                state
                    .failure
                    .get_or_insert_with(|| "P-C V3 worker acquisition unresolved".into());
                return Ok(PcPrepareStepV3::Failed);
            }
            W::DropUnstartedTask(task) => {
                drop(state);
                finish_failed_pc_task_release_v3(recovery, task)?;
                return Ok(PcPrepareStepV3::Pending);
            }
        }
    }
    if state.prepared.is_none() {
        state.prepared = Some(PcPreparedRootV3 {
            admission: admission.clone(),
            owner: state.owner.as_ref().expect("owner retained").clone(),
            route: state.route.as_ref().expect("route retained").clone(),
            stage,
            pump: Arc::clone(state.pump.as_ref().expect("pump retained")),
            worker: state.worker.take().expect("worker retained"),
            recovery: Arc::clone(recovery),
        });
    }
    let prepared = state.prepared.take().expect("prepared owner retained");
    drop(state);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bridge.try_publish_prepared(prepared)
    }));
    let mut state = recovery.lock().map_err(|_| "P-C V3 recovery poisoned")?;
    match outcome {
        Ok(PcPreparedOutcomeV3::Accepted) => {
            state.published = true;
            Ok(PcPrepareStepV3::Published)
        }
        Ok(PcPreparedOutcomeV3::Busy(prepared)) => {
            state.prepared = Some(prepared);
            Ok(PcPrepareStepV3::Pending)
        }
        Ok(PcPreparedOutcomeV3::Rejected(reason, prepared)) => {
            state.prepared = Some(prepared);
            record_first_root_failure(&mut state.failure, reason);
            Ok(PcPrepareStepV3::Failed)
        }
        Err(payload) => {
            state.panic = Some(payload);
            record_first_root_failure(&mut state.failure, "P-C V3 publish outcome unknown");
            Ok(PcPrepareStepV3::Failed)
        }
    }
}

impl PcRootRecoveryV3 {
    /// One stage per worker tick; Busy/NotReady and an unacknowledged output
    /// remain pending in the original consumer/pipeline. No durable sink is
    /// assumed merely because an in-memory append or chunk succeeded.
    fn drive_consumer_step(&mut self) {
        use TeacherPolicyConsumerProgressV3 as P;
        let Some(consumer) = self.consumer.as_ref().cloned() else {
            return;
        };
        let mut guard = match consumer.try_lock() {
            Ok(Some(guard)) => guard,
            Ok(None) => return,
            Err(error) => {
                record_first_root_failure(&mut self.failure, error);
                self.consumer_phase = ConsumerPhaseV3::Unknown;
                return;
            }
        };
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match self.consumer_phase {
                ConsumerPhaseV3::Receive => guard.try_transfer_completion(),
                ConsumerPhaseV3::Seal => guard.try_seal(),
                ConsumerPhaseV3::TransferRow => guard.try_transfer_retired(),
                ConsumerPhaseV3::Prepare => guard.try_prepare_sealed(),
                ConsumerPhaseV3::Candidate => guard.try_prepare_candidate(),
                ConsumerPhaseV3::Capability => guard.try_transfer_capability(),
                ConsumerPhaseV3::Append => guard.try_append_policy(),
                ConsumerPhaseV3::RetainAppend => guard.try_retain_appended(),
                ConsumerPhaseV3::RetainCompletion => guard.try_retain_processed_completion(),
                ConsumerPhaseV3::Rejected => return Ok(P::Rejected),
                ConsumerPhaseV3::Unknown => return Ok(P::Unknown),
            }));
        match result {
            Ok(Ok(P::CompletionStored))
                if matches!(self.consumer_phase, ConsumerPhaseV3::Receive) =>
            {
                self.consumer_phase = ConsumerPhaseV3::Seal
            }
            Ok(Ok(P::Sealed)) if matches!(self.consumer_phase, ConsumerPhaseV3::Seal) => {
                self.consumer_phase = ConsumerPhaseV3::TransferRow
            }
            Ok(Ok(P::RowTransferred))
                if matches!(self.consumer_phase, ConsumerPhaseV3::TransferRow) =>
            {
                self.consumer_phase = ConsumerPhaseV3::Prepare
            }
            Ok(Ok(P::Prepared)) if matches!(self.consumer_phase, ConsumerPhaseV3::Prepare) => {
                self.consumer_phase = ConsumerPhaseV3::Candidate
            }
            Ok(Ok(P::CandidateReady))
                if matches!(self.consumer_phase, ConsumerPhaseV3::Candidate) =>
            {
                self.consumer_phase = ConsumerPhaseV3::Capability
            }
            Ok(Ok(P::CapabilityStored))
                if matches!(self.consumer_phase, ConsumerPhaseV3::Capability) =>
            {
                self.consumer_phase = ConsumerPhaseV3::Append
            }
            Ok(Ok(P::Appended)) if matches!(self.consumer_phase, ConsumerPhaseV3::Append) => {
                self.consumer_phase = ConsumerPhaseV3::RetainAppend
            }
            Ok(Ok(P::AppendedEvidenceStored))
                if matches!(self.consumer_phase, ConsumerPhaseV3::RetainAppend) =>
            {
                self.consumer_phase = ConsumerPhaseV3::RetainCompletion
            }
            Ok(Ok(P::CompletionStored))
                if matches!(self.consumer_phase, ConsumerPhaseV3::RetainCompletion) =>
            {
                self.consumer_phase = ConsumerPhaseV3::Receive
            }
            Ok(Ok(P::Busy | P::NotReady | P::OutputHeld)) => {}
            Ok(Ok(P::ChunkReady)) => {
                let persistence = match self.pipeline.try_lock() {
                    Ok(Some(mut pipeline)) => {
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            pipeline.persist_chunk()
                        })) {
                            Ok(result) => Some(result),
                            Err(payload) => {
                                self.panic = Some(payload);
                                self.consumer_phase = ConsumerPhaseV3::Unknown;
                                self.failure.get_or_insert_with(|| {
                                    "P-C V3 chunk persistence panicked".into()
                                });
                                None
                            }
                        }
                    }
                    Ok(None) => None,
                    Err(error) => {
                        self.consumer_phase = ConsumerPhaseV3::Unknown;
                        self.failure.get_or_insert_with(|| error.into());
                        None
                    }
                };
                match persistence {
                    Some(PolicyChunkPersistenceV3::Failed) => {
                        self.consumer_phase = ConsumerPhaseV3::Rejected;
                        self.failure
                            .get_or_insert_with(|| "P-C V3 output chunk persistence failed".into());
                    }
                    Some(PolicyChunkPersistenceV3::Unknown) => {
                        self.consumer_phase = ConsumerPhaseV3::Unknown;
                        self.failure.get_or_insert_with(|| {
                            "P-C V3 output chunk persistence unknown".into()
                        });
                    }
                    Some(
                        PolicyChunkPersistenceV3::Persisted | PolicyChunkPersistenceV3::NotReady,
                    )
                    | None => {}
                }
            }
            Ok(Ok(P::Rejected)) => {
                self.consumer_phase = ConsumerPhaseV3::Rejected;
                self.failure
                    .get_or_insert_with(|| "P-C V3 policy consumer rejected".into());
            }
            Ok(Ok(P::Unknown)) => {
                self.consumer_phase = ConsumerPhaseV3::Unknown;
                self.failure
                    .get_or_insert_with(|| "P-C V3 policy consumer outcome unknown".into());
            }
            Ok(Ok(_)) => {
                self.consumer_phase = ConsumerPhaseV3::Unknown;
                self.failure
                    .get_or_insert_with(|| "P-C V3 policy phase mismatch".into());
            }
            Ok(Err(error)) => {
                self.consumer_phase = ConsumerPhaseV3::Unknown;
                self.failure.get_or_insert_with(|| error.to_string());
            }
            Err(payload) => {
                self.consumer_phase = ConsumerPhaseV3::Unknown;
                self.panic = Some(payload);
                self.failure
                    .get_or_insert_with(|| "P-C V3 policy step panicked".into());
            }
        }
    }

    /// Signal only this root's existing route and session. These requests do
    /// not attest delivery, terminal drain, worker join or native completion.
    fn request_route_abort(&mut self, deadline: Instant) {
        if let Some(route) = &self.route {
            route.cancel_all();
            match route.try_revoke_all_targets() {
                Ok(Some(())) => self.abort_targets_revoked = true,
                Ok(None) => {}
                Err(error) => {
                    self.failure
                        .get_or_insert_with(|| format!("P-C V3 abort target revoke: {error}"));
                }
            }
        }
        if let Some(pump) = self.pump.as_ref().map(Arc::clone) {
            match pump.try_lock() {
                Ok(mut pump) => {
                    pump.close_admission();
                    match pump.try_closed_drain() {
                        Ok(observation) => self.abort_input_observation = Some(observation),
                        Err(error) => {
                            self.failure.get_or_insert_with(|| {
                                format!("P-C V3 abort input drain: {error}")
                            });
                        }
                    }
                    if let (Some(route), Some(custody)) = (&self.route, &self.closed_custody) {
                        match pump.try_handoff_closed(route, custody) {
                            Ok(observation) => self.abort_handoff_observation = Some(observation),
                            Err(error) => {
                                self.failure.get_or_insert_with(|| {
                                    format!("P-C V3 abort custody handoff: {error}")
                                });
                            }
                        }
                    }
                    match pump.try_drain_pending() {
                        Ok(observation) => {
                            if matches!(observation, TeacherPendingDrainV3::Unknown) {
                                self.abort_terminal_unknown = true;
                            }
                            self.abort_terminal_observation = Some(observation);
                        }
                        Err(error) => {
                            self.abort_terminal_unknown = true;
                            self.failure.get_or_insert_with(|| {
                                format!("P-C V3 abort terminal drain: {error}")
                            });
                        }
                    }
                }
                Err(std::sync::TryLockError::WouldBlock) => {}
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    self.abort_terminal_unknown = true;
                    self.record_teacher_drain_failure("P-C V3 teacher pump lock poisoned".into());
                }
            };
        }
        let Some(stage) = self.stage.as_ref().map(Arc::clone) else {
            return;
        };
        let session = match stage.try_lock() {
            Ok(stage) => stage.session(),
            Err(std::sync::TryLockError::WouldBlock) => return,
            Err(std::sync::TryLockError::Poisoned(_)) => {
                self.record_teacher_drain_failure("P-C V3 teacher stage lock poisoned".into());
                return;
            }
        };
        let Some(session) = session else {
            return;
        };
        let Some(mut session) = try_teacher_recovery_lock(
            &session,
            &mut self.failure,
            &mut self.teacher_drain_failure,
            "session",
        ) else {
            return;
        };
        if !session
            .original_admission()
            .same_original_admission(&self.admission)
        {
            self.record_teacher_drain_failure("P-C V3 abort session admission differs".into());
            return;
        }
        if let Err(error) = session.request_shutdown(deadline) {
            // A nonblocking conflict retains the same original scope and can
            // be retried; it is not a shutdown or physical completion proof.
            if super::teacher::v3_session::is_teacher_scope_busy(&error) {
                return;
            }
            self.record_teacher_drain_failure(format!("P-C V3 abort shutdown: {error}"));
        }
    }

    /// Drain the same session with no live route or root worker. A route that
    /// existed may only use this after exact rollback custody was retained.
    /// The aggregate is not a handoff, fence, success, or quota receipt.
    fn try_drain_teacher_without_live_route(&mut self, deadline: Instant) -> bool {
        if self.aggregate.is_some() {
            return true;
        }
        let post_route_held = match (&self.post_route_custody, &self.route, &self.receiver) {
            (Some(custody), Some(route), Some(receiver)) => {
                custody.route.same_original_route(route)
                    && Arc::ptr_eq(&custody.receiver, receiver)
                    && custody.witness.matches_original_route(route)
            }
            _ => false,
        };
        if self.panic.is_some()
            || ((self.route.is_some() || self.receiver.is_some()) && !post_route_held)
            || self.owner.is_some()
            || self.pump.is_some()
            || self.worker.is_some()
            || self.worker_key.is_some()
            || self.prepared.is_some()
            || self.published
        {
            return false;
        }
        let Some(stage) = self.stage.as_ref().map(Arc::clone) else {
            return false;
        };
        let session = match stage.try_lock() {
            Ok(stage) => stage.session(),
            Err(std::sync::TryLockError::WouldBlock) => return false,
            Err(_) => {
                self.record_teacher_drain_failure("P-C V3 abort stage poisoned".into());
                return false;
            }
        };
        let Some(session) = session else {
            // Prepared/claimed/unknown stages retain their original owner.
            return false;
        };
        let mut session = match session.try_lock() {
            Ok(session) => session,
            Err(std::sync::TryLockError::WouldBlock) => return false,
            Err(_) => {
                self.record_teacher_drain_failure("P-C V3 abort session poisoned".into());
                return false;
            }
        };
        if !session
            .original_admission()
            .same_original_admission(&self.admission)
        {
            self.record_teacher_drain_failure("P-C V3 abort session admission differs".into());
            return false;
        }
        if let Err(error) = session.request_shutdown(deadline) {
            // This is the producer's documented nonblocking lock conflict;
            // a later tick may close the same original scope successfully.
            if is_teacher_scope_busy(&error) {
                return false;
            }
            self.record_teacher_drain_failure(format!("P-C V3 abort shutdown: {error}"));
            return false;
        }
        let actors: Vec<String> = session.registration().teachers.keys().cloned().collect();
        for actor in actors {
            if self.diagnostics.try_reserve(1).is_err() {
                return false;
            }
            let has_lifecycle = if let Some(lifecycle) = session.lifecycle_mut(&actor) {
                let _ = lifecycle.poll();
                if let Some(diagnostics) = lifecycle.take_diagnostics_after_join() {
                    self.diagnostics.push((actor.clone(), diagnostics));
                }
                true
            } else {
                false
            };
            if has_lifecycle {
                match session.finish_worker(&actor) {
                    Ok(true) => {}
                    Ok(false) => return false,
                    Err(error) => {
                        self.record_teacher_drain_failure(format!(
                            "P-C V3 abort worker {actor}: {error}"
                        ));
                        return false;
                    }
                }
            }
        }
        match session.try_aggregate() {
            Ok(Some(aggregate)) => {
                self.aggregate = Some(aggregate);
                true
            }
            Ok(None) => false,
            Err(error) => {
                self.record_teacher_drain_failure(format!("P-C V3 abort aggregate: {error}"));
                false
            }
        }
    }

    /// Preserve an earlier root failure and the distinct drain failure.
    fn record_teacher_drain_failure(&mut self, error: String) {
        record_teacher_drain_error(&mut self.failure, &mut self.teacher_drain_failure, error);
    }

    /// Only actual drained terminals, joined teacher workers and the original
    /// scope aggregate can reach the handoff. Output backpressure remains in
    /// the pump until the target consumer takes it; it is never discarded.
    fn try_drain_teacher(&mut self, handoff: &RootHandoffV3, deadline: Instant) -> bool {
        self.drive_consumer_step();
        let Some(stage) = self.stage.as_ref().map(Arc::clone) else {
            return false;
        };
        let session = match stage.try_lock() {
            Ok(stage) => stage.session(),
            Err(std::sync::TryLockError::WouldBlock) => return false,
            Err(std::sync::TryLockError::Poisoned(_)) => {
                self.record_teacher_drain_failure("P-C V3 teacher stage lock poisoned".into());
                return false;
            }
        };
        let Some(session) = session else {
            return false;
        };
        {
            let mut session = match session.try_lock() {
                Ok(session) => session,
                Err(std::sync::TryLockError::WouldBlock) => return false,
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    self.record_teacher_drain_failure(
                        "P-C V3 teacher session lock poisoned".into(),
                    );
                    return false;
                }
            };
            if let Err(error) = session.request_shutdown(deadline) {
                record_teacher_drain_shutdown(
                    &mut self.failure,
                    &mut self.teacher_drain_failure,
                    &error,
                );
                return false;
            }
        }
        let Some(pump_owner) = self.pump.as_ref().map(Arc::clone) else {
            return false;
        };
        {
            let mut pump = match pump_owner.try_lock() {
                Ok(pump) => pump,
                Err(std::sync::TryLockError::WouldBlock) => return false,
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    self.record_teacher_drain_failure("P-C V3 teacher pump lock poisoned".into());
                    return false;
                }
            };
            // Output and semantic failure remain in the retained pump or
            // consumer; neither alone prevents physical worker retirement.
            // An actual pending dispatch is still a separate drain barrier.
            let result = pump.try_pump();
            record_teacher_pump_result(&mut self.failure, &mut self.teacher_drain_failure, result);
            if pump.has_pending() {
                let result = pump.try_drain_pending();
                if !teacher_pending_drain_ready(
                    &mut self.failure,
                    &mut self.teacher_drain_failure,
                    result,
                    pump.has_pending(),
                ) {
                    return false;
                }
            }
        }
        let mut session = match session.try_lock() {
            Ok(session) => session,
            Err(std::sync::TryLockError::WouldBlock) => return false,
            Err(std::sync::TryLockError::Poisoned(_)) => {
                self.record_teacher_drain_failure("P-C V3 teacher session lock poisoned".into());
                return false;
            }
        };
        let actors: Vec<String> = session.registration().teachers.keys().cloned().collect();
        for actor in actors {
            if self.diagnostics.try_reserve(1).is_err() {
                return false;
            }
            if let Some(lifecycle) = session.lifecycle_mut(&actor) {
                let _ = lifecycle.poll();
                if let Some(diagnostics) = lifecycle.take_diagnostics_after_join() {
                    self.diagnostics.push((actor.clone(), diagnostics));
                }
            }
            if !teacher_drain_worker_finished(
                &mut self.failure,
                &mut self.teacher_drain_failure,
                session.finish_worker(&actor),
            ) {
                return false;
            }
        }
        if self.aggregate.is_none() {
            self.aggregate = retain_teacher_drain_aggregate(
                &mut self.failure,
                &mut self.teacher_drain_failure,
                session.try_aggregate(),
            );
            if self.aggregate.is_none() {
                return false;
            }
        }
        let mut original_handoff = match handoff.try_lock() {
            Ok(handoff) => handoff,
            Err(std::sync::TryLockError::WouldBlock) => return false,
            Err(std::sync::TryLockError::Poisoned(_)) => {
                self.record_teacher_drain_failure("P-C V3 teacher handoff lock poisoned".into());
                return false;
            }
        };
        let Some(aggregate) = self.aggregate.take() else {
            return false;
        };
        match original_handoff.submit_teacher_aggregate_ack(aggregate) {
            Ok(()) => true,
            Err(aggregate) => {
                self.aggregate = Some(aggregate);
                false
            }
        }
    }
}
