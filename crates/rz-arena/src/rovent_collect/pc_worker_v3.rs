//! One-root P-C /3 supervisor and recovery custody.
//!
//! A prepared root is not an execution, teacher label or physical-completion
//! proof. The bridge must retain the same recovery Arc before any claim,
//! source handoff or physical worker spawn.
use super::capture_router_v3::CaptureRouterV3;
use super::launch::ManagedOutputOwnerV3;
use super::observer_lock_v3::RoventCollectionOwnerLockV3;
use super::pc_hooks_lock_v3::{
    PcPreparedOutcomeV3, PcPreparedRootV3, PcQuotaOutcomeV3, PcRootAccountingV3,
    RoventPcCollectionBridgeV3,
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
    TeacherPendingDrainV3, TeacherPumpSetupErrorV3, TeacherPumpStepV3, TeacherPumpV3,
};
use super::teacher::v3_session::{
    RoventCollectionTeacherSessionV3, TeacherPrepareErrorV3, TeacherPrepareOwnerV3,
    TeacherSessionClaimErrorV3, TeacherSessionStageStateV3, TeacherSessionStagingV3,
    TeacherStartErrorV3,
};
use super::writer_lock_v3::RoventCorpusWriterLockV3;
use rz_search::external_cpu::collection_teacher::TeacherDiagnostics;
use rz_uci::pals_adaptive::{
    AdaptiveCollectionError, RoventCollectionRootAdmissionV3, RoventTeacherScopeCapabilityV3,
    RoventTeacherScopePhysicalAckV3, RoventTeacherScopeWorkerAckV3,
};
use rz_uci::rovent_bootstrap::pc_collection_v3::{
    RootHandoffV3, RoventPcCollectionWorkerKeyV3, RoventPcCollectionWorkerSpawnFailureV3,
    RoventPcCollectionWorkerV3,
};
use rz_uci::rovent_external_launch::{
    RoventTeacherScopeAggregateAckV3, RoventTeacherScopeAggregateFailureV3,
    TeacherRecoveryCustodianV3,
};
use std::sync::{Arc, Mutex, atomic::AtomicBool};
use std::time::{Duration, Instant};

/// The original scope, route, session and any uncertain worker remain rooted
/// here even if preparation or a worker task is interrupted.
pub(crate) struct PcRootRecoveryV3 {
    admission: RoventCollectionRootAdmissionV3,
    stage: Option<Arc<Mutex<TeacherSessionStagingV3>>>,
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
    worker_spawn_failure: Option<RoventPcCollectionWorkerSpawnFailureV3>,
    worker_key: Option<RoventPcCollectionWorkerKeyV3>,
    handoff: Option<RootHandoffV3>,
    aggregate: Option<RoventTeacherScopeAggregateAckV3>,
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
    pub(crate) fn new(
        admission: RoventCollectionRootAdmissionV3,
        pipeline: PolicyRowPipelineOwnerV3,
    ) -> Arc<Mutex<Self>> {
        let output = pipeline.output_owner();
        Arc::new(Mutex::new(Self {
            admission,
            stage: None,
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
            worker_key: None,
            handoff: None,
            aggregate: None,
            diagnostics: Vec::new(),
            prepared: None,
            custodian: TeacherRecoveryCustodianV3::new(),
            teacher_reserved: false,
            no_scope_recovery: NoScopeRecoveryV3::Pending,
            sealed_unstarted: false,
            next_actor: 0,
            published: false,
            failure: None,
        }))
    }
    pub(crate) fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
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
    /// Irreversibly close preparation before issuing the root's no-scope seal.
    /// This records no physical or worker completion by itself.
    pub(crate) fn seal_definitely_unstarted(&mut self) -> bool {
        if self.failure.is_none() || !self.definitely_unstarted() {
            return false;
        }
        self.sealed_unstarted = true;
        true
    }
    pub(crate) fn is_sealed_unstarted(&self) -> bool {
        self.sealed_unstarted
    }
    pub(crate) fn no_scope_recovery(&self) -> NoScopeRecoveryV3 {
        self.no_scope_recovery
    }
}

pub(crate) enum PcPrepareStepV3 {
    Pending,
    Published,
    Failed,
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

    pub(crate) fn seal_claimed_unstarted(&mut self) -> bool {
        if !self
            .claimed_unstarted
            .as_ref()
            .is_some_and(|owner| owner.aggregate.is_some())
            || self.try_claimed_unstarted() != ClaimedUnstartedPollV3::Confirmed
        {
            return false;
        }
        self.sealed_claimed_unstarted = true;
        true
    }

    pub(crate) fn is_sealed_claimed_unstarted(&self) -> bool {
        self.sealed_claimed_unstarted
    }

    pub(crate) fn try_claimed_unstarted(&mut self) -> ClaimedUnstartedPollV3 {
        use rz_search::external_cpu::process_scope::{
            CollectionMemberPhysicalState as P, CollectionMemberWorkerState as W,
        };
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
                Ok(Err(error)) => {
                    if recovery.scope_error.is_none() {
                        recovery.scope_error = Some(error);
                    }
                    return ClaimedUnstartedPollV3::Pending;
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
    bridge: Arc<RoventPcCollectionBridgeV3>,
    router: Arc<CaptureRouterV3>,
    roots: Vec<Arc<Mutex<PcRootRecoveryV3>>>,
    pending_root: Option<RoventCollectionRootAdmissionV3>,
    pipeline: Option<PolicyRowPipelineOwnerV3>,
    startup_deadline: Instant,
    cleanup_deadline: Instant,
}

impl PcWorkerSupervisorV3 {
    /// Same in-memory game output owner; no durable sink is implied.
    pub(crate) fn output_owner(&self) -> Option<PolicyOutputOwnerV3> {
        self.pipeline
            .as_ref()
            .map(PolicyRowPipelineOwnerV3::output_owner)
    }

    /// The caller retains its original managed output capability; path names
    /// or a second copy of input bytes cannot manufacture this ownership.
    pub(crate) fn try_install_output(
        &self,
        output: &Arc<ManagedOutputOwnerV3>,
    ) -> Result<bool, String> {
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Ok(false);
        };
        let Some(mut pipeline) = pipeline.try_lock().map_err(str::to_owned)? else {
            return Ok(false);
        };
        pipeline
            .install_output_directory(self.bridge.inputs(), output)
            .map_err(|e| e.to_string())?;
        Ok(true)
    }
    pub(crate) fn new(
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
            bridge,
            router,
            roots,
            pending_root: None,
            pipeline: None,
            startup_deadline,
            cleanup_deadline,
        })
    }

    /// One finite poll; unresolved evidence stays in its original root owner.
    pub(crate) fn step(&mut self) -> Result<(), String> {
        if self.pending_root.is_none()
            && self.roots.len() < usize::from(self.bridge.inputs().plan().maximum_roots)
        {
            self.pending_root = self.bridge.try_next_root().map_err(str::to_owned)?;
        }
        if let Some(admission) = self.pending_root.as_ref() {
            if self.pipeline.is_none() {
                let writer = RoventCorpusWriterLockV3::new_shared(
                    self.bridge.inputs().manifest(),
                    admission,
                )
                .map_err(|error| error.to_string())?;
                self.pipeline = Some(PolicyRowPipelineOwnerV3::new(writer));
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
            self.pending_root.take();
        }
        for original in &self.roots {
            let outcome = prepare_root_step_v3(
                &self.bridge,
                &self.router,
                original,
                self.startup_deadline,
                self.cleanup_deadline,
            );
            match outcome {
                Ok(PcPrepareStepV3::Published) => {}
                Ok(PcPrepareStepV3::Pending | PcPrepareStepV3::Failed) => continue,
                Err(error) => {
                    if let Ok(mut root) = original.lock() {
                        root.failure = Some(error);
                    }
                    continue;
                }
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

/// Called outside the native manager callback. Busy leaves the same recovery
/// owner in place; no new actor or claim is issued for another root.
pub(crate) fn prepare_root_step_v3(
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
    if state.published {
        return Ok(PcPrepareStepV3::Published);
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
        state.failure = Some("P-C V3 unstarted root was sealed".into());
        return Ok(PcPrepareStepV3::Failed);
    }
    if !router
        .registration()
        .same_original_owner(&bridge.capture_registration())
    {
        state.failure = Some("P-C V3 capture router differs from bridge".into());
        return Ok(PcPrepareStepV3::Failed);
    }
    if !state.teacher_reserved {
        match bridge.try_reserve_teacher(&admission) {
            Ok(PcQuotaOutcomeV3::Reserved) => state.teacher_reserved = true,
            Ok(PcQuotaOutcomeV3::Busy | PcQuotaOutcomeV3::Exhausted) => {
                return Ok(PcPrepareStepV3::Pending);
            }
            _ => {
                state.failure = Some("teacher quota is not owned by this recovery".into());
                return Ok(PcPrepareStepV3::Failed);
            }
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
                state.failure = Some("P-C V3 teacher stage preparation failed before claim".into());
                return Ok(PcPrepareStepV3::Failed);
            }
            Err(payload) => {
                state.panic = Some(payload);
                state.failure = Some("P-C V3 teacher preflight panicked; ownership unknown".into());
                return Ok(PcPrepareStepV3::Failed);
            }
        };
        state.stage = Some(Arc::new(Mutex::new(stage)));
    }
    let stage = Arc::clone(state.stage.as_ref().expect("stage retained"));
    {
        let mut staged = stage.lock().map_err(|_| "P-C V3 stage poisoned")?;
        use super::teacher::v3_session::TeacherSessionStageStateV3 as S;
        match staged.state() {
            S::Prepared => {
                if let Err(error) = staged.claim() {
                    state.stage_claim_failure = Some(error);
                    state.failure = Some("teacher scope claim unresolved or rejected".into());
                    return Ok(PcPrepareStepV3::Failed);
                }
            }
            S::Claimed => {}
            S::Ready => {}
            _ => {
                state.failure = Some("teacher stage is unresolved".into());
                return Ok(PcPrepareStepV3::Failed);
            }
        }
        if staged.state() == S::Claimed && staged.finish().is_err() {
            state.failure = Some("teacher session assembly failed".into());
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
        let actor_id = roster[state.next_actor].actor_id.clone();
        let mut owner = session.lock().map_err(|_| "P-C V3 session poisoned")?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            owner.start_actor(
                &actor_id,
                startup_deadline,
                cleanup_deadline,
                &mut state.custodian,
            )
        }));
        drop(owner);
        match result {
            Ok(Ok(_)) => {}
            Ok(Err(error)) => {
                state.actor_start_failure = Some(error);
                state.failure =
                    Some("teacher actor start failed; original custody retained".into());
                return Ok(PcPrepareStepV3::Failed);
            }
            Err(payload) => {
                state.panic = Some(payload);
                state.failure =
                    Some("teacher actor start panicked; original outcome unknown".into());
                return Ok(PcPrepareStepV3::Failed);
            }
        }
        state.next_actor += 1;
        return Ok(PcPrepareStepV3::Pending);
    }
    if state.route.is_none() {
        let installed = bridge
            .capture_registration()
            .try_install_speculative_root_v3(
                &admission,
                inputs.plan().teacher_mailbox_capacity as usize,
                inputs.plan().teacher_tickets_per_root as usize,
                Arc::new(AtomicBool::new(false)),
            )
            .map_err(|e| e.to_string())?;
        let Some((route, receiver)) = installed else {
            return Ok(PcPrepareStepV3::Pending);
        };
        state.route = Some(route);
        state.receiver = Some(Arc::new(Mutex::new(receiver)));
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
                state.failure = Some("teacher pump setup failed; original owners retained".into());
                return Ok(PcPrepareStepV3::Failed);
            }
            Err(payload) => {
                state.panic = Some(payload);
                state.failure = Some("teacher pump setup unknown after panic".into());
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
            state.failure = Some("P-C V3 output/pipeline owner differs".into());
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
                state.failure = Some(format!("P-C V3 consumer setup: {error}"));
                return Ok(PcPrepareStepV3::Failed);
            }
            Err(payload) => {
                state.panic = Some(payload);
                state.failure = Some("P-C V3 consumer setup unknown".into());
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
                state.failure = Some("P-C V3 consumer original owners differ".into());
                return Ok(PcPrepareStepV3::Failed);
            }
        }
    }
    // Root worker construction and publication are deliberately separate from
    // claiming the teacher scope; failure must keep the original owner.
    if state.worker.is_none() {
        let original = Arc::clone(recovery);
        let task = Box::new(move |receiver: rz_uci::rovent_bootstrap::pc_collection_v3::RoventPcRootHandoffReceiverV3| {
            let mut handoff = None;
            while Instant::now() < cleanup_deadline {
                if let Ok(mut owner) = original.try_lock() {
                    if let Some(pump) = &owner.pump {
                        if let Ok(mut pump) = pump.try_lock() {
                            if matches!(pump.try_pump(), Ok(TeacherPumpStepV3::OutputBackpressure)) { /* retained by pump */ }
                        }
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
                        Ok(mut owner) => owner.try_drain_teacher(&handoff, cleanup_deadline),
                        Err(_) => false,
                    };
                    if completed { break; }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            Box::new(original) as Box<dyn std::any::Any + Send>
        });
        match RoventPcCollectionWorkerV3::spawn(admission.clone(), task) {
            Ok(worker) => {
                state.worker_key = Some(worker.recovery_key());
                state.worker = Some(worker);
            }
            Err(error) => {
                state.worker_spawn_failure = Some(error);
                state.failure = Some("P-C V3 worker spawn failed".into());
                return Ok(PcPrepareStepV3::Failed);
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
            state.failure = Some(reason.into());
            Ok(PcPrepareStepV3::Failed)
        }
        Err(payload) => {
            state.panic = Some(payload);
            state.failure = Some("P-C V3 publish outcome unknown".into());
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
                self.failure = Some(error.into());
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
            Err(_) => None,
        };
        let Some(session) = session else {
            return false;
        };
        {
            let Ok(mut session) = session.try_lock() else {
                return false;
            };
            if session.request_shutdown(deadline).is_err() {
                return false;
            }
        }
        let Some(pump_owner) = self.pump.as_ref().map(Arc::clone) else {
            return false;
        };
        {
            let Ok(mut pump) = pump_owner.try_lock() else {
                return false;
            };
            // Output and semantic failure remain in the retained pump or
            // consumer; neither alone prevents physical worker retirement.
            // An actual pending dispatch is still a separate drain barrier.
            let _ = pump.try_pump();
            if pump.has_pending() {
                match pump.try_drain_pending() {
                    Ok(TeacherPendingDrainV3::Drained) if !pump.has_pending() => {}
                    Ok(
                        TeacherPendingDrainV3::Drained
                        | TeacherPendingDrainV3::Pending
                        | TeacherPendingDrainV3::Full
                        | TeacherPendingDrainV3::Unknown,
                    )
                    | Err(_) => return false,
                }
            }
        }
        let Ok(mut session) = session.try_lock() else {
            return false;
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
            if !matches!(session.finish_worker(&actor), Ok(true)) {
                return false;
            }
        }
        if self.aggregate.is_none() {
            match session.try_aggregate() {
                Ok(Some(aggregate)) => self.aggregate = Some(aggregate),
                _ => return false,
            }
        }
        let Ok(mut original_handoff) = handoff.try_lock() else {
            return false;
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
