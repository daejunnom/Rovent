//! One-root P-C /3 supervisor and recovery custody.
//!
//! A prepared root is not an execution, teacher label or physical-completion
//! proof. The bridge must retain the same recovery Arc before any claim,
//! source handoff or physical worker spawn.
use super::capture_router_v3::CaptureRouterV3;
use super::observer_lock_v3::RoventCollectionOwnerLockV3;
use super::pc_hooks_lock_v3::{
    PcPreparedOutcomeV3, PcPreparedRootV3, PcQuotaOutcomeV3, PcRootAccountingV3,
    RoventPcCollectionBridgeV3,
};
use super::speculative_route_v3::{SpeculativeReceiverV3, SpeculativeRootV3};
use super::teacher::v3_pump::{TeacherPumpSetupErrorV3, TeacherPumpStepV3, TeacherPumpV3};
use super::teacher::v3_session::{
    RoventCollectionTeacherSessionV3, TeacherPrepareErrorV3, TeacherSessionStagingV3,
    TeacherStartErrorV3,
};
use rz_search::external_cpu::collection_teacher::TeacherDiagnostics;
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3;
use rz_uci::rovent_bootstrap::pc_collection_v3::{
    RootHandoffV3, RoventPcCollectionWorkerKeyV3, RoventPcCollectionWorkerSpawnFailureV3,
    RoventPcCollectionWorkerV3,
};
use rz_uci::rovent_external_launch::{
    RoventTeacherScopeAggregateAckV3, TeacherRecoveryCustodianV3,
};
use std::sync::{Arc, Mutex, atomic::AtomicBool};
use std::time::{Duration, Instant};

/// The original scope, route, session and any uncertain worker remain rooted
/// here even if preparation or a worker task is interrupted.
pub(crate) struct PcRootRecoveryV3 {
    admission: RoventCollectionRootAdmissionV3,
    stage: Option<Arc<Mutex<TeacherSessionStagingV3>>>,
    preflight_failure: Option<TeacherPrepareErrorV3>,
    route: Option<SpeculativeRootV3>,
    receiver: Option<Arc<Mutex<SpeculativeReceiverV3>>>,
    owner: Option<RoventCollectionOwnerLockV3>,
    pump: Option<Arc<Mutex<TeacherPumpV3>>>,
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
    pub(crate) fn new(admission: RoventCollectionRootAdmissionV3) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            admission,
            stage: None,
            preflight_failure: None,
            route: None,
            receiver: None,
            owner: None,
            pump: None,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NoScopeRecoveryV3 {
    Pending,
    Complete,
    Unknown,
}

/// Bounded supervisor; each original root remains owned even if a different
/// root fails or needs additional recovery. Caller sets finite deadlines.
pub(crate) struct PcWorkerSupervisorV3 {
    bridge: Arc<RoventPcCollectionBridgeV3>,
    router: Arc<CaptureRouterV3>,
    roots: Vec<Arc<Mutex<PcRootRecoveryV3>>>,
    startup_deadline: Instant,
    cleanup_deadline: Instant,
}

impl PcWorkerSupervisorV3 {
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
            startup_deadline,
            cleanup_deadline,
        })
    }

    /// One finite poll; unresolved evidence stays in its original root owner.
    pub(crate) fn step(&mut self) -> Result<(), String> {
        if self.roots.len() < usize::from(self.bridge.inputs().plan().maximum_roots) {
            if let Some(admission) = self.bridge.try_next_root().map_err(str::to_owned)? {
                if self.roots.iter().any(|root| {
                    root.lock()
                        .ok()
                        .is_some_and(|root| root.matches_admission(&admission))
                }) {
                    return Err("P-C V3 duplicate original root".into());
                }
                self.roots.push(PcRootRecoveryV3::new(admission));
            }
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
        return Ok(PcPrepareStepV3::Failed);
    }
    if state.sealed_unstarted {
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
                if staged.claim().is_err() {
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
    // Root worker construction and publication are deliberately separate from
    // claiming the teacher scope; failure must keep the original owner.
    if state.worker.is_none() {
        let original = Arc::clone(recovery);
        let task = Box::new(move |receiver: rz_uci::rovent_bootstrap::pc_collection_v3::RoventPcRootHandoffReceiverV3| {
            let mut handoff = None;
            while Instant::now() < cleanup_deadline {
                if let Ok(owner) = original.try_lock() {
                    if let Some(pump) = &owner.pump {
                        if let Ok(mut pump) = pump.try_lock() {
                            if matches!(pump.try_pump(), Ok(TeacherPumpStepV3::OutputBackpressure)) { /* retained by pump */ }
                        }
                    }
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
    /// Only actual drained terminals, joined teacher workers and the original
    /// scope aggregate can reach the handoff. Output backpressure remains in
    /// the pump until the target consumer takes it; it is never discarded.
    fn try_drain_teacher(&mut self, handoff: &RootHandoffV3, deadline: Instant) -> bool {
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
            if matches!(
                pump.try_pump(),
                Err(_)
                    | Ok(TeacherPumpStepV3::OutputBackpressure)
                    | Ok(TeacherPumpStepV3::Completed)
                    | Ok(TeacherPumpStepV3::Rejected)
            ) {
                return false;
            }
            if pump.has_pending() {
                return false;
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
