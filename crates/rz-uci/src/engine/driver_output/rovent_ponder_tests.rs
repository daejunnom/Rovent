//! Actual Owner event/dispatch/Start/SourceSeal/output-envelope ingress fixtures.
//! The probe admits Rules/source custody and delegates the existing CPU fixture;
//! its declared model artifacts are fixture metadata, never loaded capabilities.
//! This does not exercise native bootstrap, a neural backend, GPU or root credit.

#![cfg(feature = "pals-adaptive")]

use super::super::tests::search_driver_output_fixture;
use super::super::{
    Owner, ProcessClock, RulesSearchPosition, RulesUciPort, append_result, event_channel,
};
use crate::bridge::NodeBudgetSemantics;
use crate::rovent_root_admission::{RoventRootAdmission, RoventRootAdmissionPolicy};
use crate::search_driver::{
    DriverPhysicalCompletion, SearchKind, SearchSessionContext, SearchSessionDriver,
    SearchSessionFailure, SearchSessionReport,
};
use crate::{Effect, EngineIdentity, Event, GoLimits, Session, SessionResult};
use rz_contracts::rovent::ROVENT_SEARCH_REVISION;
use rz_contracts::{ContractError, Digest};
use rz_experiments::{
    ROVENT_EXECUTION_DOMAIN, ROVENT_POSITION_VISIT_UNIT, RoventExecutionManifest, RoventInputLock,
};
use rz_position::{BoardMove, Position};
use serde_json::{Value, json};
use std::{
    num::NonZeroU16,
    sync::{Arc, Condvar, Mutex, mpsc::Receiver},
    time::{Duration, Instant},
};

const GO: &str = "go ponder wtime 5000 btime 5000 movestogo 1 nodes 7 searchmoves e2e4 d2d4";

fn hex(digest: Digest) -> String {
    digest.0.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn artifact(name: &str) -> Value {
    json!({"path":format!("fixture/{name}"),"sha256":"a".repeat(64),"bytes":32,
        "source":"https://example.org/owner-source-only-fixture","license":"fixture-only"})
}

/// Same source-only manifest shape as root-admission tests. Original go fields
/// come from the actual context; artifact strings do not attest a real backend.
fn source_lock(position: &Position, context: &SearchSessionContext) -> RoventInputLock {
    let go = context.original_go.as_ref().unwrap();
    let model = json!({"semantics":"fixture-pc","encoding":"fixture-encoding",
        "policy_head":"legal-candidates/1","value_head":"side-to-move-wdl/1",
        "implementation_sha256":"b".repeat(64),"weights":artifact("weights"),
        "provenance":artifact("model-provenance"),"diagnostic":false});
    let cleanup = context
        .shutdown_deadline
        .duration_since(context.control.deadline);
    let cleanup_ms = u64::try_from(cleanup.as_millis()).unwrap();
    assert_eq!(Duration::from_millis(cleanup_ms), cleanup);
    let wire = json!({
        "domain":ROVENT_EXECUTION_DOMAIN,"contract_revision":ROVENT_SEARCH_REVISION,
        "run_id":"actual-owner-ponder-source-fixture","purpose":"validation",
        "engine":{"source_commit":"a".repeat(40),"source_components":{"fixture":"c".repeat(64)},
            "binary":artifact("binary"),"mode":"cpu_pc",
            "pc":{"proposer":model,"critic":model,"validator":null,
                "backend":"rust_cpu","device":{"device":"cpu"},
                "backend_configuration_sha256":"d".repeat(64),"runtime_artifacts":[],
                "fp32":true,"tf32":false,"batch_width":1,
                "ort_intra_threads":null,"ort_inter_threads":null,"ort_spinning":null},
            "researcher":null,"teachers":[],"nnuec":null,
            "search":{"node_unit":ROVENT_POSITION_VISIT_UNIT,"requested_nodes":context.requested_nodes,
                "depth_plies":go.depth_plies.map(NonZeroU16::get),
                "mate_moves":go.mate_moves.map(NonZeroU16::get),"registered_max_chess_plies":64,
                "searchmoves":go.searchmoves,"search_configuration_sha256":"1".repeat(64),
                "root_state_and_history_sha256":hex(rz_position::contracts::rules_snapshot_semantic_digest(&position.snapshot())),
                "rules_profile_sha256":hex(rz_position::contracts::profile_digest())}},
        "resources":{"cpu_quota_millicpus":2000,
            "pools":[{"actors":["manager","pc_caller"],"max_threads":1,"cpu_charge_millicpus":1000}],
            "memory_high_bytes":6_u64<<30,"memory_max_bytes":12_u64<<30,
            "gpu_devices":0,"physical_nn_concurrency":1,"topology_policy":"shared-control/1"},
        "coordinator":{"policy_sha256":"2".repeat(64),"questions_max":64,
            "ancestor_depth_max":16,"waiting_jobs_max":16,"role_tickets_max":16,
            "cpu_credit_limit":0,"original_root_budget_sha256":"3".repeat(64),
            "original_deadline_seal_sha256":"4".repeat(64)},
        "wall_time_max_ms":60000,"cleanup_max_ms":cleanup_ms,"output_bytes_max":1048576,
        "agent_renewable":true,"provenance":[artifact("source-provenance")]});
    RoventExecutionManifest::from_json(&wire.to_string())
        .unwrap()
        .lock()
        .unwrap()
}

#[derive(Clone)]
struct Delivered {
    position: Position,
    context: SearchSessionContext,
    root: RoventRootAdmission,
}
#[derive(Default)]
struct ProbeState {
    runs: usize,
    delivered: Option<Delivered>,
    failure: Option<SearchSessionFailure>,
    release: bool,
    observed_hit: bool,
    returned: bool,
}
#[derive(Default)]
struct ProbeGate {
    state: Mutex<ProbeState>,
    changed: Condvar,
}
impl ProbeGate {
    fn wait_for(&self, predicate: impl Fn(&ProbeState) -> bool) {
        let state = self.state.lock().unwrap();
        let (state, _) = self
            .changed
            .wait_timeout_while(state, Duration::from_secs(5), |state| {
                !predicate(state) && state.failure.is_none()
            })
            .unwrap();
        assert!(
            state.failure.is_none(),
            "actual probe worker failed before the requested observation: {:?}",
            state.failure
        );
        assert!(predicate(&state), "bounded actual worker observation");
    }
    fn retain_failure(&self, failure: SearchSessionFailure) -> SearchSessionFailure {
        self.state.lock().unwrap().failure = Some(failure.clone());
        self.changed.notify_all();
        failure
    }
    fn delivered(&self) -> Delivered {
        self.wait_for(|state| state.delivered.is_some());
        self.state
            .lock()
            .unwrap()
            .delivered
            .as_ref()
            .unwrap()
            .clone()
    }
    fn release(&self) {
        let mut state = self.state.lock().unwrap();
        state.release = true;
        if let Some(delivered) = &state.delivered {
            delivered
                .context
                .effective_clock
                .as_ref()
                .unwrap()
                .signal()
                .notify();
        }
        self.changed.notify_all();
    }
}

struct SourceAdmissionProbe {
    inner: Arc<dyn SearchSessionDriver>,
    clock: ProcessClock,
    gate: Arc<ProbeGate>,
}
impl SearchSessionDriver for SourceAdmissionProbe {
    fn supports_ponder_clock(&self) -> bool {
        true
    }
    fn node_budget_semantics(&self) -> NodeBudgetSemantics {
        NodeBudgetSemantics::RoventPositionVisits
    }
    fn validate_go(&self, position: &Position, go: &GoLimits) -> Result<(), ContractError> {
        self.inner.validate_go(position, go)
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        self.inner.search_goal_capability()
    }
    fn kind(&self) -> SearchKind {
        self.inner.kind()
    }
    fn implementation(&self) -> Digest {
        self.inner.implementation()
    }
    fn identity(&self) -> EngineIdentity {
        self.inner.identity()
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        self.inner.reset_game()
    }
    fn finish_session(
        &self,
        until: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        self.inner.finish_session(until)
    }
    fn accept_uci_output(
        &self,
        authority: rz_contracts::pals::SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        self.inner
            .accept_uci_output(authority, bestmove, from_report)
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        let source = context
            .original_start
            .as_ref()
            .expect("actual dispatch must issue private Start custody");
        source
            .verify(position, context, &self.clock)
            .map_err(|error| {
                self.gate
                    .retain_failure(SearchSessionFailure::debug("PonderStartSource", &error))
            })?;
        let root = RoventRootAdmission::admit(
            &source_lock(position, context),
            RoventRootAdmissionPolicy::DynamicGo,
            position,
            context,
            &self.clock,
        )
        .map_err(|error| {
            self.gate
                .retain_failure(SearchSessionFailure::debug("PonderSourceAdmission", &error))
        })?;
        // The report is produced by the pre-existing CPU fixture. No synthetic
        // report or report Arc is injected into the actual output envelope.
        let report = self
            .inner
            .run(position, context, progress)
            .map_err(|failure| self.gate.retain_failure(failure))?;
        let clock = context
            .effective_clock
            .as_ref()
            .expect("actual ponder clock");
        {
            let mut state = self.gate.state.lock().unwrap();
            state.runs += 1;
            assert_eq!(state.runs, 1, "ponder hit must not dispatch another Start");
            state.delivered = Some(Delivered {
                position: position.clone(),
                context: context.clone(),
                root,
            });
            self.gate.changed.notify_all();
        }
        let until = Instant::now() + Duration::from_secs(5);
        loop {
            // Version is read before the snapshot/release/cancellation checks.
            let observed = clock.version();
            let snapshot = clock.snapshot(Instant::now());
            let mut state = self.gate.state.lock().unwrap();
            if snapshot.has_hit() {
                state.observed_hit = true;
                self.gate.changed.notify_all();
            }
            if state.release || context.cancellation.is_canceled() || snapshot.is_closed() {
                state.returned = true;
                self.gate.changed.notify_all();
                return Ok(report);
            }
            if Instant::now() >= until {
                return Err(SearchSessionFailure::debug(
                    "PonderFixtureLatch",
                    &"bounded report latch expired",
                ));
            }
            drop(state);
            clock.wait_changed(observed, Duration::from_millis(10));
        }
    }
}

fn install_probe(owner: &mut Owner, session: &mut Session<RulesUciPort>) -> Arc<ProbeGate> {
    assert!(session.active_ticket().is_none());
    assert!(owner.active.is_none() && owner.workers.is_empty());
    // This idle registration matches the wrapped CPU fixture's existing limit
    // of 8. It is not a root replacement: scope/current Arc/implementation stay
    // unchanged, and no Go, binding or private Start has been issued yet.
    let scope = owner.session_owner.lifecycle_scope();
    owner.settings.search.max_simulations = 8;
    owner.session_owner = crate::contracts::ContractSessionOwner::new_scope(
        scope,
        owner.clock.epoch(),
        owner.clock.b_clock(),
        owner.settings.search,
    )
    .unwrap();
    assert_eq!(*owner.scope.lock().unwrap(), scope);
    session
        .set_node_budget_semantics(NodeBudgetSemantics::RoventPositionVisits)
        .unwrap();
    let gate = Arc::new(ProbeGate::default());
    owner.driver = Some(Arc::new(SourceAdmissionProbe {
        inner: owner.driver.take().unwrap(),
        clock: owner.clock.clone(),
        gate: Arc::clone(&gate),
    }));
    assert!(
        owner
            .handle(
                session,
                Event::Line("setoption name Ponder value true".into())
            )
            .accepted
    );
    gate
}

fn dispatch_all(
    owner: &mut Owner,
    sender: &std::sync::mpsc::SyncSender<Event>,
    result: &mut SessionResult<RulesSearchPosition>,
) {
    for effect in std::mem::take(&mut result.effects) {
        owner.dispatch(effect, sender).unwrap();
    }
}
fn finish_worker(
    owner: &mut Owner,
    session: &mut Session<RulesUciPort>,
    events: &Receiver<Event>,
) -> SessionResult<RulesSearchPosition> {
    let until = Instant::now() + Duration::from_secs(5);
    let mut out = SessionResult::default();
    loop {
        let event = events
            .recv_timeout(until.saturating_duration_since(Instant::now()))
            .expect("bounded actual completion event");
        let complete = matches!(&event, Event::Complete { .. });
        append_result(&mut out, owner.handle(session, event));
        if complete {
            break;
        }
    }
    while owner.workers.iter().any(|worker| !worker.is_finished()) {
        assert!(Instant::now() < until, "bounded physical worker return");
        std::thread::yield_now();
    }
    owner.reap_workers(true).unwrap();
    append_result(
        &mut out,
        owner.handle(
            session,
            Event::RejectedInput {
                code: "OwnerWake",
                message: String::new(),
            },
        ),
    );
    assert!(owner.workers.is_empty(), "actual physical worker join");
    out
}
fn bestmoves(result: &SessionResult<RulesSearchPosition>) -> Vec<&str> {
    result
        .protocol
        .iter()
        .filter_map(|line| line.strip_prefix("bestmove "))
        .map(|line| line.split_ascii_whitespace().next().unwrap())
        .collect()
}
fn no_restart_or_cancel(result: &SessionResult<RulesSearchPosition>) {
    assert!(
        result
            .effects
            .iter()
            .all(|effect| !matches!(effect, Effect::Start { .. } | Effect::Cancel { .. }))
    );
}
fn original_custody_unchanged(original: &Delivered, observed: &Delivered, owner: &Owner) {
    let a = &original.context;
    let b = &observed.context;
    assert_eq!(a.original_time_budget, b.original_time_budget);
    assert_eq!(a.control.deadline, b.control.deadline);
    assert_eq!(a.control.soft_deadline, b.control.soft_deadline);
    assert_eq!(a.control.admission_deadline, b.control.admission_deadline);
    assert!(Arc::ptr_eq(
        &a.control.cancellation,
        &b.control.cancellation
    ));
    assert!(Arc::ptr_eq(
        a.original_go.as_ref().unwrap(),
        b.original_go.as_ref().unwrap()
    ));
    assert!(Arc::ptr_eq(&a.current, &b.current));
    assert!(Arc::ptr_eq(
        a.original_start.as_ref().unwrap(),
        b.original_start.as_ref().unwrap()
    ));
    assert_eq!(
        owner.session_owner.control().unwrap().deadline,
        a.control.deadline
    );
    assert_eq!(original.root.digest(), observed.root.digest());
    assert_eq!(
        original.root.root_budget_digest(),
        observed.root.root_budget_digest()
    );
    assert_eq!(
        original.root.deadline_seal_digest(),
        observed.root.deadline_seal_digest()
    );
    assert_eq!(original.root.hard_deadline(), observed.root.hard_deadline());
    assert_eq!(
        original.root.cleanup_deadline(),
        observed.root.cleanup_deadline()
    );
    a.original_start
        .as_ref()
        .unwrap()
        .verify(&original.position, a, &owner.clock)
        .unwrap();
}

#[test]
fn actual_start_hit_keeps_worker_and_root_caps_then_publishes_its_report_once() {
    let (mut owner, mut session, acknowledgments) = search_driver_output_fixture(false);
    let gate = install_probe(&mut owner, &mut session);
    let (sender, events) = event_channel();
    let mut start = owner.handle(&mut session, Event::Line(GO.into()));
    assert!(start.accepted && start.protocol.is_empty());
    let ticket = session.active_ticket().unwrap();
    let scope = owner.session_owner.lifecycle_scope();
    dispatch_all(&mut owner, &sender, &mut start);
    let delivered = gate.delivered();
    let envelope = Arc::clone(
        owner
            .active
            .as_ref()
            .unwrap()
            .driver_output
            .as_ref()
            .unwrap(),
    );
    assert!(envelope.report_for(&ticket, scope).unwrap().is_none());
    assert_eq!(delivered.root.requested_nodes(), Some(7));
    assert_eq!(delivered.context.control.max_simulations, 8);
    assert_eq!(
        delivered.context.original_go.as_ref().unwrap().nodes,
        Some(7)
    );
    assert_eq!(delivered.root.allowed_root_moves().len(), 2);
    assert_eq!(delivered.root.declared_ceilings().cpu_credit_limit, 0);
    let original = delivered.context.original_time_budget.unwrap();
    assert!(original.allocation <= Duration::from_secs(60));
    // Pure preadmission rejects an oversized future hit before replacing the
    // running Start, allocating another cap or canceling its original work.
    let refused = owner.handle(
        &mut session,
        Event::Line("go ponder movetime 60001 nodes 7".into()),
    );
    assert!(!refused.accepted && bestmoves(&refused).is_empty());
    no_restart_or_cancel(&refused);
    assert_eq!(session.active_ticket(), Some(ticket.clone()));
    assert!(!delivered.context.cancellation.is_canceled());
    let hit = owner.handle(&mut session, Event::Line("ponderhit".into()));
    assert!(hit.accepted && bestmoves(&hit).is_empty());
    no_restart_or_cancel(&hit);
    assert_eq!(session.active_ticket(), Some(ticket.clone()));
    assert_eq!(owner.session_owner.lifecycle_scope(), scope);
    gate.wait_for(|state| state.observed_hit);
    assert!(!gate.state.lock().unwrap().returned);
    assert_eq!(owner.workers.len(), 1);
    assert!(!owner.workers[0].is_finished());
    assert!(!delivered.context.cancellation.is_canceled());
    assert!(
        !delivered
            .context
            .control
            .cancellation
            .load(std::sync::atomic::Ordering::Acquire)
    );
    let effective = delivered
        .context
        .effective_clock
        .as_ref()
        .unwrap()
        .snapshot(Instant::now());
    assert!(effective.has_hit() && effective.can_admit(Instant::now()));
    assert!(effective.budget().soft_deadline < original.soft_deadline);
    assert!(effective.budget().admission_deadline < original.admission_deadline);
    assert!(effective.budget().hard_deadline < original.hard_deadline);
    assert!(effective.budget().output_deadline < original.output_deadline);
    original_custody_unchanged(&delivered, &gate.delivered(), &owner);
    gate.release();
    let completed = finish_worker(&mut owner, &mut session, &events);
    assert_eq!(bestmoves(&completed).len(), 1);
    let report = envelope.report_for(&ticket, scope).unwrap().unwrap();
    assert_eq!(
        Some(bestmoves(&completed)[0].to_owned()),
        report.best_move.map(|movement| movement.to_string())
    );
    assert_eq!(acknowledgments.lock().unwrap().len(), 1);
    assert!(acknowledgments.lock().unwrap()[0].2);
    assert_eq!(gate.state.lock().unwrap().runs, 1);
    for repeated in ["ponderhit", "stop"] {
        let repeated = owner.handle(&mut session, Event::Line(repeated.into()));
        no_restart_or_cancel(&repeated);
        assert!(bestmoves(&repeated).is_empty());
        assert!(Arc::ptr_eq(
            &report,
            &envelope.report_for(&ticket, scope).unwrap().unwrap()
        ));
        assert_eq!(acknowledgments.lock().unwrap().len(), 1);
    }
}

#[test]
fn actual_worker_empty_held_report_hit_and_stop_release_the_same_arc_once() {
    for release in ["ponderhit", "stop"] {
        let (mut owner, mut session, acknowledgments) = search_driver_output_fixture(false);
        let gate = install_probe(&mut owner, &mut session);
        let (sender, events) = event_channel();
        let mut start = owner.handle(&mut session, Event::Line(GO.into()));
        assert!(start.accepted);
        let ticket = session.active_ticket().unwrap();
        let scope = owner.session_owner.lifecycle_scope();
        dispatch_all(&mut owner, &sender, &mut start);
        let delivered = gate.delivered();
        gate.release();
        let completed = finish_worker(&mut owner, &mut session, &events);
        assert!(bestmoves(&completed).is_empty());
        let (envelope, report) = owner.admitted_ponder_output_for_tests().unwrap();
        let envelope = Arc::clone(envelope);
        let report = Arc::clone(report);
        assert!(Arc::ptr_eq(
            &report,
            &envelope.report_for(&ticket, scope).unwrap().unwrap()
        ));
        assert!(owner.workers.is_empty());
        assert!(acknowledgments.lock().unwrap().is_empty());
        original_custody_unchanged(&delivered, &gate.delivered(), &owner);
        let released = owner.handle(&mut session, Event::Line(release.into()));
        assert!(
            released
                .effects
                .iter()
                .all(|effect| !matches!(effect, Effect::Start { .. }))
        );
        assert_eq!(bestmoves(&released).len(), 1);
        assert_eq!(
            Some(bestmoves(&released)[0].to_owned()),
            report.best_move.map(|movement| movement.to_string())
        );
        assert!(Arc::ptr_eq(
            &report,
            &envelope.report_for(&ticket, scope).unwrap().unwrap()
        ));
        assert!(owner.admitted_ponder_output_for_tests().is_none());
        assert_eq!(acknowledgments.lock().unwrap().len(), 1);
        assert!(acknowledgments.lock().unwrap()[0].2);
        for repeat in [release, "ponderhit", "stop"] {
            assert!(bestmoves(&owner.handle(&mut session, Event::Line(repeat.into()))).is_empty());
            assert_eq!(acknowledgments.lock().unwrap().len(), 1);
        }
        assert_eq!(gate.state.lock().unwrap().runs, 1);
    }
}

#[test]
fn actual_miss_stop_and_quit_cancel_join_without_cross_root_publication() {
    for command in ["position startpos moves e2e4", "ucinewgame", "stop", "quit"] {
        let (mut owner, mut session, acknowledgments) = search_driver_output_fixture(false);
        let gate = install_probe(&mut owner, &mut session);
        let (sender, events) = event_channel();
        let mut start = owner.handle(&mut session, Event::Line(GO.into()));
        assert!(start.accepted);
        let ticket = session.active_ticket().unwrap();
        let scope = owner.session_owner.lifecycle_scope();
        dispatch_all(&mut owner, &sender, &mut start);
        let delivered = gate.delivered();
        let envelope = Arc::clone(
            owner
                .active
                .as_ref()
                .unwrap()
                .driver_output
                .as_ref()
                .unwrap(),
        );
        let mut ended = owner.handle(&mut session, Event::Line(command.into()));
        assert!(
            ended
                .effects
                .iter()
                .all(|effect| !matches!(effect, Effect::Start { .. }))
        );
        dispatch_all(&mut owner, &sender, &mut ended);
        assert!(delivered.context.cancellation.is_canceled());
        append_result(&mut ended, finish_worker(&mut owner, &mut session, &events));
        assert!(gate.state.lock().unwrap().returned);
        assert_eq!(gate.state.lock().unwrap().runs, 1);
        assert!(session.active_ticket().is_none());
        assert!(owner.admitted_ponder_output_for_tests().is_none());
        if command == "stop" {
            assert_eq!(bestmoves(&ended).len(), 1);
            assert_eq!(acknowledgments.lock().unwrap().len(), 1);
            assert!(!acknowledgments.lock().unwrap()[0].2);
        } else {
            assert!(bestmoves(&ended).is_empty());
            assert!(acknowledgments.lock().unwrap().is_empty());
        }
        if matches!(command, "position startpos moves e2e4" | "ucinewgame") {
            assert_ne!(scope, owner.session_owner.lifecycle_scope());
            assert!(
                envelope
                    .report_for(&ticket, owner.session_owner.lifecycle_scope())
                    .is_err()
            );
            let stale_hit = owner.handle(&mut session, Event::Line("ponderhit".into()));
            assert!(!stale_hit.accepted && bestmoves(&stale_hit).is_empty());
        }
    }
}
