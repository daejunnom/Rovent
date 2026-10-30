//! Unregistered Owner publication draft, not native Rovent startup acceptance.
//!
//! Registration prerequisites (owned by the parent source window):
//! - expose engine::tests::{search_driver_output_fixture,
//!   completed_driver_search} as pub(super);
//! - wire each actual Start's DriverOutputEnvelope and immutable report Arc
//!   through ActiveBinding and held output, after the actual worker fence;
//! - add the read-only cfg(test) Owner::admitted_ponder_output_for_tests getter
//!   returning references to those actual held envelope/report Arc fields;
//! - register this file below engine::driver_output, without synthetic reports.
//!
//! These cases use the existing bounded CPU/mock fixtures. They do not attest
//! the new native startup/root-admission path. Its initially expired admission
//! must remain StoppedInitial; neither these fixtures nor ponderhit may issue a
//! replacement live cap or extend the original question deadline.

use super::super::tests::{completed_driver_search, search_driver_output_fixture};
use super::super::{Owner, RulesSearchPosition, RulesUciPort, append_result, event_channel};
use crate::contracts::SessionScope;
use crate::search_driver::{
    DriverPhysicalCompletion, SearchKind, SearchSessionContext, SearchSessionDriver,
    SearchSessionFailure, SearchSessionReport,
};
use crate::{Effect, EngineIdentity, Event, Session, SessionResult};
use rz_contracts::{ContractError, Digest, ErrorCode};
use rz_position::{BoardMove, Position};
use std::{
    num::NonZeroU16,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

/// Observes the context actually delivered to run. It delegates all selected
/// driver admission/output behavior; it is not a second ponder publisher.
struct DeliveredPonderContext {
    inner: Arc<dyn SearchSessionDriver>,
    observed: Arc<Mutex<Vec<SearchSessionContext>>>,
}
impl SearchSessionDriver for DeliveredPonderContext {
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        self.inner.node_budget_semantics()
    }
    fn validate_go(&self, position: &Position, go: &crate::GoLimits) -> Result<(), ContractError> {
        self.inner.validate_go(position, go)
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        self.inner.search_goal_capability()
    }
    fn finish_session(
        &self,
        until: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        self.inner.finish_session(until)
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
        {
            let mut observed = self
                .observed
                .lock()
                .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
            let repeated = !observed.is_empty();
            // Preserve evidence of a forbidden second invocation before
            // rejecting it. Two slots suffice to falsify the one-run contract.
            if observed.len() < 2 {
                observed.push(context.clone());
            }
            if repeated {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PonderRunRestarted",
                    detail: "publication fixture must not restart an original ponder question"
                        .into(),
                });
            }
        }
        self.inner.run(position, context, progress)
    }
}

fn observe_original_context(owner: &mut Owner) -> Arc<Mutex<Vec<SearchSessionContext>>> {
    let observed = Arc::new(Mutex::new(Vec::with_capacity(1)));
    let inner = owner
        .driver
        .take()
        .expect("actual startup-selected fixture driver");
    owner.driver = Some(Arc::new(DeliveredPonderContext {
        inner,
        observed: Arc::clone(&observed),
    }));
    observed
}

fn one_context(observed: &Arc<Mutex<Vec<SearchSessionContext>>>) -> SearchSessionContext {
    let observed = observed.lock().unwrap();
    assert_eq!(
        observed.len(),
        1,
        "hit/stop must not dispatch another question"
    );
    observed[0].clone()
}

/// Drives only the actual Owner/event loop. This small fixture separates Start
/// from dispatch so the original binding can be inspected before run receives
/// it; it does not reimplement held report state or publication decisions.
fn complete_original_start(
    owner: &mut Owner,
    session: &mut Session<RulesUciPort>,
    started: SessionResult<RulesSearchPosition>,
) -> SessionResult<RulesSearchPosition> {
    assert!(started.accepted && started.protocol.is_empty());
    assert!(matches!(started.effects.as_slice(), [Effect::Start { .. }]));
    let (sender, events) = event_channel();
    for effect in started.effects {
        owner.dispatch(effect, &sender).unwrap();
    }
    let until = Instant::now() + Duration::from_secs(5);
    let mut completed = SessionResult::default();
    loop {
        let event = events
            .recv_timeout(until.saturating_duration_since(Instant::now()))
            .expect("bounded actual CPU/mock fixture completion");
        let terminal = matches!(&event, Event::Complete { .. });
        append_result(&mut completed, owner.handle(session, event));
        if terminal {
            break;
        }
        assert!(Instant::now() < until);
    }
    while owner.workers.iter().any(|worker| !worker.is_finished()) {
        assert!(Instant::now() < until, "bounded original worker completion");
        thread::yield_now();
    }
    completed
}

fn bestmoves(output: &SessionResult<RulesSearchPosition>) -> Vec<&str> {
    output
        .protocol
        .iter()
        .filter_map(|line| line.strip_prefix("bestmove "))
        .map(|line| line.split_ascii_whitespace().next().unwrap())
        .collect()
}

fn assert_no_new_start(output: &SessionResult<RulesSearchPosition>) {
    assert!(
        output
            .effects
            .iter()
            .all(|effect| !matches!(effect, Effect::Start { .. })),
        "held publication must not request another worker: {:?}",
        output.effects
    );
}

#[test]
fn held_report_hit_and_stop_publish_same_start_once_without_renewing_question() {
    for pals in [false, true] {
        for release in ["ponderhit", "stop"] {
            let (mut owner, mut session, acknowledgments) = search_driver_output_fixture(pals);
            let observed = observe_original_context(&mut owner);
            assert!(
                owner
                    .handle(
                        &mut session,
                        Event::Line("setoption name Ponder value true".into()),
                    )
                    .accepted
            );
            let started = owner.handle(&mut session, Event::Line("go ponder nodes 1".into()));
            let ticket = session.active_ticket().expect("actual original Start");
            let scope = owner.session_owner.lifecycle_scope();
            let original_control = owner.session_owner.control().unwrap().clone();
            let completed = complete_original_start(&mut owner, &mut session, started);
            assert!(bestmoves(&completed).is_empty());
            let delivered = one_context(&observed);
            assert_eq!(SessionScope::Search(delivered.authority), scope);
            assert_eq!(delivered.control.deadline, original_control.deadline);
            assert_eq!(
                delivered.control.soft_deadline,
                original_control.soft_deadline
            );
            assert_eq!(
                delivered.control.admission_deadline,
                original_control.admission_deadline
            );
            assert!(Arc::ptr_eq(
                &delivered.control.cancellation,
                &original_control.cancellation
            ));
            assert!(delivered.original_go.as_ref().unwrap().ponder);
            assert!(!delivered.original_go.as_ref().unwrap().infinite);
            assert_eq!(delivered.requested_nodes, Some(1));
            assert_eq!(session.active_ticket(), Some(ticket.clone()));
            let admitted = owner.admitted_ponder_report.as_ref().unwrap();
            assert_eq!(admitted.ticket, ticket);
            assert_eq!(admitted.scope, scope);
            assert!(acknowledgments.lock().unwrap().is_empty());

            // TODO(parent wiring): this getter must borrow the actual held
            // envelope/report fields. It must not build or re-record a report.
            let (envelope, held) = owner.admitted_ponder_output_for_tests().unwrap();
            let envelope = Arc::clone(envelope);
            let held = Arc::clone(held);
            let original_report = envelope.report_for(&ticket, scope).unwrap().unwrap();
            assert!(Arc::ptr_eq(&held, &original_report));

            let released = owner.handle(&mut session, Event::Line(release.into()));
            assert_no_new_start(&released);
            let printed = bestmoves(&released);
            assert_eq!(printed.len(), 1);
            assert_eq!(
                Some(printed[0]),
                held.best_move
                    .map(|movement| movement.to_string())
                    .as_deref()
            );
            assert_eq!(owner.session_owner.lifecycle_scope(), scope);
            assert!(owner.workers.is_empty());
            assert!(session.active_ticket().is_none());
            assert!(owner.admitted_ponder_output_for_tests().is_none());
            assert!(Arc::ptr_eq(
                &held,
                &envelope.report_for(&ticket, scope).unwrap().unwrap()
            ));
            let SessionScope::Search(authority) = scope else {
                panic!("search scope");
            };
            assert_eq!(
                *acknowledgments.lock().unwrap(),
                vec![(authority, printed[0].to_owned(), true)]
            );
            assert_eq!(
                one_context(&observed).control.deadline,
                original_control.deadline
            );
            for repeated in [release, "ponderhit", "stop"] {
                let repeated = owner.handle(&mut session, Event::Line(repeated.into()));
                assert_no_new_start(&repeated);
                assert!(bestmoves(&repeated).is_empty());
                assert_eq!(acknowledgments.lock().unwrap().len(), 1);
                one_context(&observed);
            }
        }
    }
}

#[test]
fn already_admitted_passive_report_retains_original_deadline_after_clock_expiry() {
    for pals in [false, true] {
        for release in ["ponderhit", "stop"] {
            let (mut owner, mut session, acknowledgments) = search_driver_output_fixture(pals);
            let observed = observe_original_context(&mut owner);
            let completed = completed_driver_search(&mut owner, &mut session, true);
            assert!(bestmoves(&completed).is_empty());
            let ticket = session.active_ticket().unwrap();
            let scope = owner.session_owner.lifecycle_scope();
            let original = one_context(&observed);
            let (_, held) = owner.admitted_ponder_output_for_tests().unwrap();
            let held = Arc::clone(held);
            let expired = owner.session_owner.handle_event_at(
                &mut session,
                Event::Deadline(ticket.clone()),
                original.control.deadline + Duration::from_nanos(1),
            );
            assert!(expired.errors.is_empty() && expired.session.protocol.is_empty());
            assert_no_new_start(&expired.session);
            assert!(original.cancellation.is_canceled());
            assert_eq!(
                owner.session_owner.control().unwrap().deadline,
                original.control.deadline
            );
            let (_, still_held) = owner.admitted_ponder_output_for_tests().unwrap();
            assert!(Arc::ptr_eq(&held, still_held));
            let released = owner.handle(&mut session, Event::Line(release.into()));
            assert_no_new_start(&released);
            assert_eq!(bestmoves(&released).len(), 1);
            assert_eq!(owner.session_owner.lifecycle_scope(), scope);
            assert!(owner.workers.is_empty());
            assert_eq!(acknowledgments.lock().unwrap().len(), 1);
            assert!(acknowledgments.lock().unwrap()[0].2);
            assert_eq!(
                one_context(&observed).control.deadline,
                original.control.deadline
            );
            assert!(owner.admitted_ponder_output_for_tests().is_none());
        }
    }
}

#[test]
fn root_replacement_revokes_held_custody_without_publishing_an_old_report() {
    for pals in [false, true] {
        for replacement in ["position startpos moves e2e4", "ucinewgame"] {
            let (mut owner, mut session, acknowledgments) = search_driver_output_fixture(pals);
            let observed = observe_original_context(&mut owner);
            completed_driver_search(&mut owner, &mut session, true);
            let ticket = session.active_ticket().unwrap();
            let prior_scope = owner.session_owner.lifecycle_scope();
            let (envelope, held) = owner.admitted_ponder_output_for_tests().unwrap();
            let envelope = Arc::clone(envelope);
            let held = Arc::clone(held);
            assert!(Arc::ptr_eq(
                &held,
                &envelope.report_for(&ticket, prior_scope).unwrap().unwrap()
            ));
            let replaced = owner.handle(&mut session, Event::Line(replacement.into()));
            assert_no_new_start(&replaced);
            assert!(replaced.accepted && bestmoves(&replaced).is_empty());
            assert_ne!(owner.session_owner.lifecycle_scope(), prior_scope);
            assert!(owner.admitted_ponder_output_for_tests().is_none());
            let stale = envelope
                .report_for(&ticket, owner.session_owner.lifecycle_scope())
                .unwrap_err();
            assert_eq!(stale.code, ErrorCode::Stale);
            let (sender, _events) = event_channel();
            for effect in replaced.effects {
                owner.dispatch(effect, &sender).unwrap();
            }
            let hit = owner.handle(&mut session, Event::Line("ponderhit".into()));
            assert_no_new_start(&hit);
            assert!(!hit.accepted && bestmoves(&hit).is_empty());
            assert!(acknowledgments.lock().unwrap().is_empty());
            owner.reap_workers(true).unwrap();
            assert!(owner.workers.is_empty());
            one_context(&observed);
        }
    }
}
