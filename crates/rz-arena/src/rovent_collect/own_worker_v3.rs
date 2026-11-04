//! Original Own /3 root preparation custody, shared with its supervisor.
//! This is not a worker completion, teacher result, or publication proof.

use super::capture_router_v3::CaptureRouterV3;
use super::launch::PreparedPcCollectionInputsV3;
use super::observer_lock_v3::RoventCollectionOwnerLockV3;
use super::own_hooks_lock_v3::{
    OwnPrepareOutcomeV3, OwnPreparedRootV3, OwnReserveOutcomeV3, OwnRootRequestV3,
    RoventOwnCollectionBridgeV3,
};
use super::policy_row_pipeline_v3::PolicyRowPipelineOwnerV3;
use super::speculative_route_v3::{SpeculativeReceiverV3, SpeculativeRootV3};
use super::teacher::v3_pump::{TeacherPumpSetupErrorV3, TeacherPumpStepV3, TeacherPumpV3};
use super::teacher::v3_session::{
    RoventCollectionTeacherSessionV3, TeacherPrepareErrorV3, TeacherSessionStageStateV3,
    TeacherSessionStagingV3, TeacherStartErrorV3,
};
use super::writer_lock_v3::RoventCorpusWriterLockV3;
use rz_uci::pals_adaptive::{
    AdaptiveCollectionError, RoventBoardCollectionWorkerKeyV3,
    RoventBoardCollectionWorkerSpawnFailureV3, RoventBoardCollectionWorkerV3,
    RoventBoardRootHandoffReceiverV3, RoventBoardRootHandoffV3, RoventCollectionRootAdmissionV3,
};
use rz_uci::rovent_external_launch::{
    RoventTeacherScopeAggregateAckV3, TeacherRecoveryCustodianV3,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Publish this Arc to the bridge before claiming a teacher scope, opening a
/// route, or starting a worker. All uncertain original owners stay together.
pub(crate) struct OwnRootRecoveryV3 {
    pub(super) request: OwnRootRequestV3,
    pub(super) inputs: Arc<PreparedPcCollectionInputsV3>,
    pub(super) stage: Option<Arc<Mutex<TeacherSessionStagingV3>>>,
    pub(super) route: Option<SpeculativeRootV3>,
    pub(super) receiver: Option<Arc<Mutex<SpeculativeReceiverV3>>>,
    pub(super) owner: Option<RoventCollectionOwnerLockV3>,
    pub(super) pump: Option<Arc<Mutex<TeacherPumpV3>>>,
    pub(super) pipeline: PolicyRowPipelineOwnerV3,
    pub(super) pump_error: Option<TeacherPumpSetupErrorV3>,
    pub(super) pump_unknown: bool,
    pub(super) worker_key: Option<RoventBoardCollectionWorkerKeyV3>,
    pub(super) worker_failure: Option<RoventBoardCollectionWorkerSpawnFailureV3>,
    pub(super) custodian: TeacherRecoveryCustodianV3,
    pub(super) panic: Option<Box<dyn std::any::Any + Send>>,
    pub(super) additional_panics: Vec<Box<dyn std::any::Any + Send>>,
    pub(super) preflight_failure: Option<TeacherPrepareErrorV3>,
    pub(super) actor_failure: Option<TeacherStartErrorV3>,
    pub(super) worker: Option<RoventBoardCollectionWorkerV3>,
    pub(super) prepared: Option<OwnPreparedRootV3>,
    pub(super) aggregate: Option<RoventTeacherScopeAggregateAckV3>,
    pub(super) diagnostics: Vec<(
        String,
        rz_search::external_cpu::collection_teacher::TeacherDiagnostics,
    )>,
    pub(super) next_actor: usize,
    pub(super) published: bool,
    pub(super) failure: Option<String>,
    pub(super) shutdown_failure: Option<AdaptiveCollectionError>,
    pub(super) shutdown_failure_count: u64,
    pub(super) pump_failure: Option<String>,
    pub(super) pump_failure_count: u64,
}

impl OwnRootRecoveryV3 {
    pub(crate) fn new(
        request: OwnRootRequestV3,
        inputs: Arc<PreparedPcCollectionInputsV3>,
        pipeline: PolicyRowPipelineOwnerV3,
    ) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            request,
            inputs,
            pipeline,
            stage: None,
            route: None,
            receiver: None,
            owner: None,
            pump: None,
            pump_error: None,
            pump_unknown: false,
            worker_key: None,
            worker_failure: None,
            custodian: TeacherRecoveryCustodianV3::new(),
            panic: None,
            additional_panics: Vec::new(),
            preflight_failure: None,
            actor_failure: None,
            worker: None,
            prepared: None,
            aggregate: None,
            diagnostics: Vec::new(),
            next_actor: 0,
            published: false,
            failure: None,
            shutdown_failure: None,
            shutdown_failure_count: 0,
            pump_failure: None,
            pump_failure_count: 0,
        }))
    }

    pub(crate) fn matches_admission(&self, original: &RoventCollectionRootAdmissionV3) -> bool {
        self.request.admission.same_original_admission(original)
    }

    pub(crate) fn try_matches_prepared(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
        route: &SpeculativeRootV3,
        owner: &RoventCollectionOwnerLockV3,
        stage: &Arc<Mutex<TeacherSessionStagingV3>>,
        pump: &Arc<Mutex<TeacherPumpV3>>,
        pipeline: &PolicyRowPipelineOwnerV3,
        key: &RoventBoardCollectionWorkerKeyV3,
    ) -> Result<Option<bool>, &'static str> {
        let basic = !self.request.is_revoked()
            && self.pipeline.same_owner(pipeline)
            && self.matches_admission(admission)
            && self
                .route
                .as_ref()
                .is_some_and(|original| original.same_original_route(route))
            && self
                .owner
                .as_ref()
                .is_some_and(|original| original.same_original_owner(owner))
            && self
                .stage
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, stage))
            && self
                .pump
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, pump))
            && self
                .worker_key
                .as_ref()
                .is_some_and(|original| original.same_original(key));
        if !basic {
            return Ok(Some(false));
        }
        let stage = match stage.try_lock() {
            Ok(value) => value,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err("Own /3 staged session poisoned");
            }
        };
        let Some(session) = stage.session() else {
            return Ok(Some(false));
        };
        let pump = match pump.try_lock() {
            Ok(value) => value,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err("Own /3 teacher pump poisoned");
            }
        };
        if !Arc::ptr_eq(pump.session(), &session)
            || !self
                .receiver
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, pump.receiver()))
        {
            return Ok(Some(false));
        }
        let receiver = match pump.receiver().try_lock() {
            Ok(value) => value,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err("Own /3 route receiver poisoned");
            }
        };
        if !receiver.matches_original_route(route) {
            return Ok(Some(false));
        }
        let session = match session.try_lock() {
            Ok(value) => value,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err("Own /3 teacher session poisoned");
            }
        };
        Ok(Some(
            session
                .original_admission()
                .same_original_admission(admission)
                && session.all_actor_owners_started(),
        ))
    }
}

pub(crate) enum OwnPrepareStepV3 {
    Pending,
    Published,
    Failed,
}

/// One supervisor owns the pinned input and every root recovery. Its caller
/// chooses finite deadlines and must retain this owner across any failure.
pub(crate) struct OwnWorkerSupervisorV3 {
    bridge: Arc<RoventOwnCollectionBridgeV3>,
    router: Arc<CaptureRouterV3>,
    pipeline: Option<PolicyRowPipelineOwnerV3>,
    pending_request: Option<OwnRootRequestV3>,
    roots: Vec<Arc<Mutex<OwnRootRecoveryV3>>>,
    startup_deadline: Instant,
    cleanup_deadline: Instant,
}

impl OwnWorkerSupervisorV3 {
    pub(crate) fn new(
        bridge: Arc<RoventOwnCollectionBridgeV3>,
        router: Arc<CaptureRouterV3>,
        startup_deadline: Instant,
        cleanup_deadline: Instant,
    ) -> Result<Self, &'static str> {
        if startup_deadline > cleanup_deadline
            || !router
                .registration()
                .same_original_owner(bridge.capture_registration())
        {
            return Err("Own /3 supervisor deadline or capture owner mismatch");
        }
        let mut roots = Vec::new();
        roots
            .try_reserve_exact(usize::from(bridge.inputs().plan().maximum_roots))
            .map_err(|_| "Own /3 supervisor root allocation")?;
        Ok(Self {
            bridge,
            router,
            pipeline: None,
            pending_request: None,
            roots,
            startup_deadline,
            cleanup_deadline,
        })
    }

    /// One bounded supervisor pass, never invoked from the native manager.
    /// Failures retain every original root and do not skip its cleanup.
    pub(crate) fn step(&mut self) -> Result<(), String> {
        if self.pending_request.is_none()
            && self.roots.len() < usize::from(self.bridge.inputs().plan().maximum_roots)
        {
            self.pending_request = self.bridge.try_next_root().map_err(str::to_owned)?;
        }
        if let Some(request) = self.pending_request.as_ref() {
            if self.pipeline.is_none() {
                let writer = RoventCorpusWriterLockV3::new(
                    self.bridge.inputs().manifest(),
                    &request.admission,
                )
                .map_err(|error| error.to_string())?;
                let pipeline = PolicyRowPipelineOwnerV3::new(writer);
                self.bridge
                    .register_pipeline(&pipeline)
                    .map_err(str::to_owned)?;
                self.pipeline = Some(pipeline);
            }
            for root in &self.roots {
                match root.try_lock() {
                    Ok(root) if root.matches_admission(&request.admission) => {
                        return Err("Own /3 duplicate original root".into());
                    }
                    Ok(_) => {}
                    Err(std::sync::TryLockError::WouldBlock) => return Ok(()),
                    Err(std::sync::TryLockError::Poisoned(_)) => {
                        return Err("Own /3 recovery poisoned".into());
                    }
                }
            }
            let original = request.clone();
            let recovery = OwnRootRecoveryV3::new(
                original,
                Arc::clone(self.bridge.inputs()),
                self.pipeline.as_ref().expect("pipeline retained").clone(),
            );
            self.roots.push(recovery);
            self.pending_request.take();
        }
        for original in &self.roots {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                prepare_own_root_step_v3(
                    &self.bridge,
                    &self.router,
                    original,
                    self.startup_deadline,
                    self.cleanup_deadline,
                )
            }));
            match outcome {
                Ok(Ok(OwnPrepareStepV3::Published)) => {}
                Ok(Ok(OwnPrepareStepV3::Pending)) => continue,
                Ok(Ok(OwnPrepareStepV3::Failed)) => {
                    if let Ok(mut root) = original.try_lock() {
                        root.try_request_teacher_shutdown(self.cleanup_deadline);
                    }
                    continue;
                }
                Ok(Err(error)) => {
                    let mut root = original.lock().map_err(|_| "Own /3 recovery poisoned")?;
                    if root.failure.is_none() {
                        root.failure = Some(error);
                    }
                    root.try_request_teacher_shutdown(self.cleanup_deadline);
                    continue;
                }
                Err(payload) => {
                    let mut root = original.lock().unwrap_or_else(|poison| poison.into_inner());
                    root.retain_panic(payload);
                    root.failure =
                        Some("Own /3 supervisor step panicked; original state unknown".into());
                    root.try_request_teacher_shutdown(self.cleanup_deadline);
                    continue;
                }
            }
            let admission = original
                .lock()
                .map_err(|_| "Own /3 recovery poisoned")?
                .request
                .admission
                .clone();
            // Joining this one original worker is observed separately from
            // teacher aggregate, target publication and native closure.
            if let Err(error) = self.bridge.try_join_root(&admission) {
                let mut root = original.lock().map_err(|_| "Own /3 recovery poisoned")?;
                if root.failure.is_none() {
                    root.failure = Some(error.into());
                }
            }
        }
        Ok(())
    }
}

/// Resume the same root's staged acquisition. No stage is manufactured from
/// a digest, and every failed/uncertain owner remains in the same recovery.
pub(crate) fn prepare_own_root_step_v3(
    bridge: &Arc<RoventOwnCollectionBridgeV3>,
    router: &Arc<CaptureRouterV3>,
    recovery: &Arc<Mutex<OwnRootRecoveryV3>>,
    startup_deadline: Instant,
    cleanup_deadline: Instant,
) -> Result<OwnPrepareStepV3, String> {
    let admission = recovery
        .lock()
        .map_err(|_| "Own /3 recovery poisoned")?
        .request
        .admission
        .clone();
    if !bridge
        .try_publish_recovery(&admission, recovery)
        .map_err(str::to_owned)?
    {
        return Ok(OwnPrepareStepV3::Pending);
    }
    let mut state = recovery.lock().map_err(|_| "Own /3 recovery poisoned")?;
    if state.published {
        return Ok(OwnPrepareStepV3::Published);
    }
    if state.failure.is_some() {
        return Ok(OwnPrepareStepV3::Failed);
    }
    if state.request.is_revoked() {
        state.failure = Some("Own /3 original request revoked; retain staged owners".into());
        return Ok(OwnPrepareStepV3::Failed);
    }
    match admission.root_admission().try_accepts_original() {
        Ok(true) => {}
        Ok(false) => {
            state.failure = Some("Own /3 original root expired or was revoked".into());
            return Ok(OwnPrepareStepV3::Failed);
        }
        Err(error) if error.code == rz_contracts::ErrorCode::ResourceExhausted => {
            return Ok(OwnPrepareStepV3::Pending);
        }
        Err(error) => {
            state.failure = Some(format!("Own /3 root observation failed: {error}"));
            return Ok(OwnPrepareStepV3::Failed);
        }
    }
    if !router
        .registration()
        .same_original_owner(bridge.capture_registration())
    {
        state.failure = Some("Own /3 capture owner changed".into());
        return Ok(OwnPrepareStepV3::Failed);
    }
    for teacher in [false, true] {
        match bridge.try_reserve_root(&admission, recovery, teacher) {
            Ok(OwnReserveOutcomeV3::Reserved | OwnReserveOutcomeV3::AlreadyReserved) => {}
            Ok(OwnReserveOutcomeV3::Busy | OwnReserveOutcomeV3::Exhausted) => {
                return Ok(OwnPrepareStepV3::Pending);
            }
            Err(error) => {
                state.failure = Some(error.into());
                return Ok(OwnPrepareStepV3::Failed);
            }
        }
    }
    let Some(startup) = bridge.startup() else {
        return Ok(OwnPrepareStepV3::Pending);
    };
    if state.stage.is_none() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            RoventCollectionTeacherSessionV3::prepare_staged(
                &admission,
                startup,
                state.inputs.manifest(),
                state.inputs.roster(),
            )
        }));
        match result {
            Ok(Ok(stage)) => state.stage = Some(Arc::new(Mutex::new(stage))),
            Ok(Err(error)) => {
                state.preflight_failure = Some(error);
                state.failure = Some("Own /3 teacher preflight rejected".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
            Err(payload) => {
                state.retain_panic(payload);
                state.failure = Some("Own /3 teacher preflight panicked; state unknown".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
        }
    }
    let stage = Arc::clone(state.stage.as_ref().expect("original stage retained"));
    {
        let mut staged = stage.lock().map_err(|_| "Own /3 teacher stage poisoned")?;
        match staged.state() {
            TeacherSessionStageStateV3::Prepared => {
                if staged.claim().is_err() {
                    state.failure = Some("Own /3 teacher claim unresolved".into());
                    return Ok(OwnPrepareStepV3::Failed);
                }
            }
            TeacherSessionStageStateV3::Claimed | TeacherSessionStageStateV3::Ready => {}
            _ => {
                state.failure = Some("Own /3 teacher scope unresolved".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
        }
        if staged.state() == TeacherSessionStageStateV3::Claimed && staged.finish().is_err() {
            state.failure = Some("Own /3 teacher session assembly failed".into());
            return Ok(OwnPrepareStepV3::Failed);
        }
    }
    let session = stage
        .lock()
        .map_err(|_| "Own /3 stage poisoned")?
        .session()
        .ok_or("Own /3 session absent")?;
    if state.next_actor < state.inputs.plan().teacher_slots.len() {
        if state.request.is_revoked() {
            state.failure = Some("Own /3 root revoked before teacher start".into());
            return Ok(OwnPrepareStepV3::Failed);
        }
        let actor = state.inputs.plan().teacher_slots[state.next_actor]
            .actor_id
            .clone();
        let mut owner = session.lock().map_err(|_| "Own /3 session poisoned")?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            owner.start_actor(
                &actor,
                startup_deadline,
                cleanup_deadline,
                &mut state.custodian,
            )
        }));
        drop(owner);
        match result {
            Ok(Ok(_)) => state.next_actor += 1,
            Ok(Err(error)) => {
                state.actor_failure = Some(error);
                state.failure = Some("Own /3 teacher start failed".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
            Err(payload) => {
                state.retain_panic(payload);
                state.failure = Some("Own /3 teacher start outcome unknown".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
        }
        return Ok(OwnPrepareStepV3::Pending);
    }
    if state.request.is_revoked() {
        state.failure = Some("Own /3 root revoked during teacher preparation".into());
        return Ok(OwnPrepareStepV3::Failed);
    }
    if state.route.is_none() {
        let installed = bridge
            .capture_registration()
            .try_install_speculative_root_v3(
                &admission,
                state.inputs.plan().teacher_mailbox_capacity as usize,
                state.inputs.plan().teacher_tickets_per_root as usize,
                state.request.revocation_flag(),
            )
            .map_err(|error| error.to_string())?;
        let Some((route, receiver)) = installed else {
            return Ok(OwnPrepareStepV3::Pending);
        };
        state.route = Some(route);
        state.receiver = Some(Arc::new(Mutex::new(receiver)));
    }
    if state.owner.is_none() {
        state.owner = Some(
            RoventCollectionOwnerLockV3::new_shared(
                Arc::clone(state.inputs.manifest()),
                state.inputs.plan().metadata.clone(),
                Arc::clone(router),
            )
            .map_err(|error| error.to_string())?,
        );
    }
    if state.pump.is_none() {
        let receiver = Arc::clone(state.receiver.as_ref().expect("original receiver retained"));
        state.pump_unknown = true;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            TeacherPumpV3::new(
                Arc::clone(&session),
                receiver,
                state.inputs.plan().teacher_constraint,
                Duration::from_nanos(state.inputs.plan().teacher_soft_quantum_ns),
            )
        }));
        match result {
            Ok(Ok(pump)) => {
                state.pump = Some(pump);
                state.pump_unknown = false;
            }
            Ok(Err(error)) => {
                state.pump_error = Some(error);
                state.pump_unknown = false;
                state.failure = Some("Own /3 teacher pump setup rejected".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
            Err(payload) => {
                state.retain_panic(payload);
                state.failure = Some("Own /3 teacher pump setup unknown".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
        }
    }
    if state.worker.is_none() && state.prepared.is_none() {
        // A failed spawn retains its closure in this same recovery. Use a
        // Weak back-reference so that retained task cannot form an Arc cycle.
        let root = Arc::downgrade(recovery);
        let task = Box::new(move |receiver: RoventBoardRootHandoffReceiverV3| {
            let mut handoff = None;
            while Instant::now() < cleanup_deadline {
                let Some(original_root) = root.upgrade() else {
                    break;
                };
                if let Ok(mut original) = original_root.try_lock() {
                    let pump = original.pump.as_ref().map(Arc::clone);
                    let failure = if let Some(pump) = pump {
                        let value = match pump.try_lock() {
                            Ok(mut owner) => owner.try_pump().err(),
                            Err(_) => None,
                        };
                        value
                    } else {
                        None
                    };
                    if let Some(error) = failure {
                        original.pump_failure_count = original.pump_failure_count.saturating_add(1);
                        if original.pump_failure.is_none() {
                            original.pump_failure = Some(error);
                        }
                    }
                }
                match receiver.recv_timeout(Duration::from_millis(10)) {
                    Ok(Some(value)) => {
                        handoff = Some(value);
                        break;
                    }
                    Ok(None) => {}
                    Err(_) => break,
                }
            }
            if let Some(handoff) = handoff {
                while Instant::now() < cleanup_deadline {
                    let Some(original_root) = root.upgrade() else {
                        break;
                    };
                    let complete = match original_root.try_lock() {
                        Ok(mut original) => original.try_drain_teacher(&handoff, cleanup_deadline),
                        Err(_) => false,
                    };
                    if complete {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            Box::new(root) as Box<dyn std::any::Any + Send>
        });
        match RoventBoardCollectionWorkerV3::spawn(admission.clone(), task) {
            Ok(worker) => {
                state.worker_key = Some(worker.recovery_key());
                state.worker = Some(worker);
            }
            Err(error) => {
                state.worker_failure = Some(error);
                state.failure = Some("Own /3 root worker spawn failed".into());
                return Ok(OwnPrepareStepV3::Failed);
            }
        }
    }
    if state.prepared.is_none() {
        state.prepared = Some(OwnPreparedRootV3 {
            admission: admission.clone(),
            owner: state
                .owner
                .as_ref()
                .expect("original observer owner")
                .clone(),
            route: state.route.as_ref().expect("original route").clone(),
            worker: state.worker.take().expect("original worker"),
            stage,
            pump: Arc::clone(state.pump.as_ref().expect("original pump")),
            pipeline: state.pipeline.clone(),
            recovery: Arc::downgrade(recovery),
        });
    }
    let prepared = state.prepared.take().expect("original prepared root");
    drop(state);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bridge.try_publish_prepared(prepared)
    }));
    let mut state = recovery.lock().map_err(|_| "Own /3 recovery poisoned")?;
    match outcome {
        Ok(OwnPrepareOutcomeV3::Prepared) => {
            state.published = true;
            Ok(OwnPrepareStepV3::Published)
        }
        Ok(OwnPrepareOutcomeV3::Busy(original)) => {
            state.prepared = Some(original);
            Ok(OwnPrepareStepV3::Pending)
        }
        Ok(OwnPrepareOutcomeV3::Rejected { reason, original }) => {
            state.prepared = Some(original);
            state.failure = Some(reason);
            Ok(OwnPrepareStepV3::Failed)
        }
        Ok(OwnPrepareOutcomeV3::Panicked { original, payload }) => {
            state.prepared = Some(original);
            state.retain_panic(payload);
            state.failure = Some("Own /3 root preparation panicked".into());
            Ok(OwnPrepareStepV3::Failed)
        }
        Err(payload) => {
            // Worker key and all pre-registered owners stay in the recovery;
            // an unwind can never be reinterpreted as no physical start.
            state.retain_panic(payload);
            state.failure = Some("Own /3 prepared publication outcome unknown".into());
            Ok(OwnPrepareStepV3::Failed)
        }
    }
}

impl OwnRootRecoveryV3 {
    fn retain_panic(&mut self, payload: Box<dyn std::any::Any + Send>) {
        if self.panic.is_none() {
            self.panic = Some(payload);
        } else {
            self.additional_panics.push(payload);
        }
    }
    /// Best-effort cancellation of this same root after preparation failure.
    /// This is only a request: owner, workers and unresolved results remain
    /// retained until their own physical and handoff evidence is collected.
    fn try_request_teacher_shutdown(&mut self, deadline: Instant) {
        if let Some(route) = &self.route {
            route.cancel_all();
        }
        if let Some(pump) = &self.pump {
            if let Ok(mut pump) = pump.try_lock() {
                pump.close_admission();
            }
        }
        let Some(stage) = self.stage.as_ref() else {
            return;
        };
        let session = match stage.try_lock() {
            Ok(stage) => stage.session(),
            Err(_) => None,
        };
        if let Some(session) = session {
            if let Ok(mut session) = session.try_lock() {
                if let Err(error) = session.request_shutdown(deadline) {
                    self.shutdown_failure_count = self.shutdown_failure_count.saturating_add(1);
                    if self.shutdown_failure.is_none() {
                        self.shutdown_failure = Some(error);
                    }
                }
            }
        }
    }

    /// Retire the actual teacher scope only after all physical workers join;
    /// a pending query, unconsumed completion or missing aggregate stays held.
    fn try_drain_teacher(&mut self, handoff: &RoventBoardRootHandoffV3, deadline: Instant) -> bool {
        self.try_request_teacher_shutdown(deadline);
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
        if let Some(route) = &self.route {
            route.cancel_all();
        }
        let Some(pump) = self.pump.as_ref().map(Arc::clone) else {
            return false;
        };
        {
            let Ok(mut pump) = pump.try_lock() else {
                return false;
            };
            pump.close_admission();
            match pump.try_pump() {
                Err(error) => {
                    self.pump_failure_count = self.pump_failure_count.saturating_add(1);
                    if self.pump_failure.is_none() {
                        self.pump_failure = Some(error);
                    }
                    return false;
                }
                Ok(
                    TeacherPumpStepV3::OutputBackpressure
                    | TeacherPumpStepV3::Completed
                    | TeacherPumpStepV3::Rejected,
                ) => return false,
                Ok(_) => {}
            }
            if pump.has_pending() {
                return false;
            }
        }
        let Ok(mut session) = session.try_lock() else {
            return false;
        };
        if let Err(error) = session.request_shutdown(deadline) {
            self.shutdown_failure_count = self.shutdown_failure_count.saturating_add(1);
            if self.shutdown_failure.is_none() {
                self.shutdown_failure = Some(error);
            }
            return false;
        }
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
                Ok(Some(value)) => self.aggregate = Some(value),
                _ => return false,
            }
        }
        let Ok(mut original) = handoff.try_lock() else {
            return false;
        };
        let Some(aggregate) = self.aggregate.take() else {
            return false;
        };
        match original.submit_teacher_aggregate_ack(aggregate) {
            Ok(()) => true,
            Err(aggregate) => {
                self.aggregate = Some(aggregate);
                false
            }
        }
    }
}
