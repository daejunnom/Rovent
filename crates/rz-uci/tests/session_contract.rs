use rz_uci::bridge::NodeBudgetSemantics;
use rz_uci::{
    CancelReason, Effect, EngineIdentity, OptionKind, OptionSpec, OptionValue, ParserLimits,
    PositionBase, PositionPort, PositionSpec, PreparedPosition, SearchCompletion, SearchTicket,
    Session, SessionResult,
};
use std::cell::{Cell, RefCell};
use std::num::NonZeroU16;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};

const MATE_FEN: &str = "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1";
const STALEMATE_FEN: &str = "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1";
const PROMOTION_FEN: &str = "7k/P7/7K/8/8/8/8/8 w - - 0 1";

type FixtureView = (Vec<String>, bool);

/// Fixed, hand-authored views test the UCI boundary. This fixture does not
/// implement or claim to independently validate chess Rules.
#[derive(Clone, Default)]
struct FixturePort {
    requests: Rc<RefCell<Vec<PositionSpec>>>,
    next_view: Rc<RefCell<Option<FixtureView>>>,
}

impl PositionPort for FixturePort {
    type Snapshot = PositionSpec;
    type Error = &'static str;

    fn prepare(
        &self,
        spec: &PositionSpec,
    ) -> Result<PreparedPosition<Self::Snapshot>, Self::Error> {
        self.requests.borrow_mut().push(spec.clone());
        if let Some((legal_moves, exact_terminal)) = self.next_view.borrow_mut().take() {
            return Ok(PreparedPosition {
                snapshot: spec.clone(),
                legal_moves,
                exact_terminal,
            });
        }
        let (legal, exact_terminal): (&[&str], bool) = match &spec.base {
            PositionBase::StartPos => match spec
                .moves
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .as_slice()
            {
                [] => (&["d2d4", "e2e4", "g1f3"], false),
                ["e2e4"] => (&["e7e5", "e7e6"], false),
                ["e2e4", "e7e5"] => (&["g1f3", "f1c4"], false),
                ["e2e4", "e7e5", "g1f3"] => (&["b8c6", "g8f6"], false),
                _ => return Err("illegal transition in complete fixture trace"),
            },
            PositionBase::Fen(fen) if spec.moves.is_empty() => match fen.as_str() {
                MATE_FEN | STALEMATE_FEN => (&[], true),
                PROMOTION_FEN => (&["a7a8q", "a7a8r", "a7a8b", "a7a8n"], false),
                _ => return Err("semantically invalid or unregistered fixture FEN"),
            },
            _ => return Err("unregistered FEN trace"),
        };
        Ok(PreparedPosition {
            snapshot: spec.clone(),
            legal_moves: legal.iter().map(|m| (*m).into()).collect(),
            exact_terminal,
        })
    }
}

fn identity() -> EngineIdentity {
    EngineIdentity {
        name: "RoveZero fixture".into(),
        author: "RoveZero".into(),
    }
}

fn options() -> Vec<OptionSpec> {
    vec![
        OptionSpec {
            name: "Hash".into(),
            kind: OptionKind::Spin {
                default: 16,
                min: 1,
                max: 64,
            },
        },
        OptionSpec {
            name: "Use Cache".into(),
            kind: OptionKind::Check { default: false },
        },
        OptionSpec {
            name: "Weights File".into(),
            kind: OptionKind::String {
                default: String::new(),
                max_bytes: 20,
            },
        },
        OptionSpec {
            name: "Search Policy".into(),
            kind: OptionKind::Combo {
                default: "PUCT".into(),
                choices: vec!["PUCT".into(), "Fixture".into()],
            },
        },
        OptionSpec {
            name: "Clear Hash".into(),
            kind: OptionKind::Button,
        },
    ]
}

fn session(port: FixturePort) -> Session<FixturePort> {
    Session::new(port, identity(), options(), ParserLimits::default()).unwrap()
}

fn start(session: &mut Session<FixturePort>) -> SearchTicket {
    let out = session.handle_line("go nodes 64");
    assert!(out.accepted);
    assert!(out.protocol.is_empty());
    match out.effects.as_slice() {
        [
            Effect::Start {
                ticket,
                snapshot,
                limits,
                ..
            },
        ] => {
            assert_eq!(snapshot, session.snapshot());
            assert_eq!(limits.nodes, Some(64));
            assert!(!limits.infinite);
            ticket.clone()
        }
        effects => panic!("expected one Start, got {effects:?}"),
    }
}

fn has_code(out: &SessionResult<PositionSpec>, code: &str) -> bool {
    out.diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == code)
}

fn start_line(session: &mut Session<FixturePort>, line: &str) -> (SearchTicket, rz_uci::GoLimits) {
    let out = session.handle_line(line);
    assert!(out.accepted, "{:?}", out.diagnostics);
    match out.effects.as_slice() {
        [Effect::Start { ticket, limits, .. }] => (ticket.clone(), limits.clone()),
        other => panic!("expected one Start, got {other:?}"),
    }
}

#[test]
fn root_restriction_is_deduplicated_once_in_rules_order() {
    let mut session = session(FixturePort::default());
    let (ticket, limits) = start_line(&mut session, "go nodes 8 searchmoves e2e4 d2d4 e2e4");
    assert_eq!(limits.searchmoves, Some(vec!["d2d4".into(), "e2e4".into()]));
    assert_eq!(session.legal_moves(), ["d2d4", "e2e4", "g1f3"]);
    let outside = session.progress(&ticket, "g1f3");
    assert!(!outside.accepted && has_code(&outside, "IllegalBestMove"));
    assert_eq!(session.handle_line("stop").protocol, ["bestmove d2d4"]);
}

#[test]
fn node_budget_unit_registration_is_idle_only_and_legacy_remains_the_default() {
    let mut session = session(FixturePort::default());
    assert_eq!(
        session.node_budget_semantics(),
        NodeBudgetSemantics::LegacySimulations
    );
    let legacy_zero = session.handle_line("go nodes 0");
    assert!(!legacy_zero.accepted && has_code(&legacy_zero, "InvalidCommand"));
    session
        .set_node_budget_semantics(NodeBudgetSemantics::RoventPositionVisits)
        .unwrap();
    let (ticket, limits) = start_line(&mut session, "go nodes 0 searchmoves e2e4");
    assert_eq!(limits.nodes, Some(0));
    assert!(
        session
            .set_node_budget_semantics(NodeBudgetSemantics::LegacySimulations)
            .is_err()
    );
    assert_eq!(session.active_ticket(), Some(ticket));
    assert_eq!(
        session.node_budget_semantics(),
        NodeBudgetSemantics::RoventPositionVisits
    );
    assert_eq!(session.handle_line("stop").protocol, ["bestmove e2e4"]);
    session
        .set_node_budget_semantics(NodeBudgetSemantics::LegacySimulations)
        .unwrap();
    assert!(!session.handle_line("go nodes 0").accepted);
    assert!(session.handle_line("quit").accepted);
    assert!(
        session
            .set_node_budget_semantics(NodeBudgetSemantics::RoventPositionVisits)
            .is_err()
    );
}

#[test]
fn illegal_root_restriction_preserves_prior_work_and_output_authority() {
    let mut session = session(FixturePort::default());
    let ticket = start(&mut session);
    assert!(session.progress(&ticket, "g1f3").accepted);
    let rejected = session.handle_line("go nodes 1 searchmoves e2e4 e2e5");
    assert!(!rejected.accepted && has_code(&rejected, "InvalidSearchMoves"));
    assert!(rejected.effects.is_empty() && rejected.protocol.is_empty());
    assert_eq!(session.active_ticket(), Some(ticket.clone()));
    assert_eq!(
        session
            .complete(&ticket, SearchCompletion::Completed { bestmove: None })
            .protocol,
        ["bestmove g1f3"]
    );
}

#[test]
fn restricted_stop_expiry_failure_and_invalid_completion_use_only_the_allowed_fallback() {
    for end in ["stop", "expire", "failure", "illegal", "empty"] {
        let mut session = session(FixturePort::default());
        let (ticket, limits) = start_line(&mut session, "go nodes 8 searchmoves g1f3 e2e4");
        assert_eq!(limits.searchmoves, Some(vec!["e2e4".into(), "g1f3".into()]));
        assert!(!session.progress(&ticket, "d2d4").accepted);
        let out = match end {
            "stop" => session.handle_line("stop"),
            "expire" => session.expire(&ticket),
            "failure" => session.complete(
                &ticket,
                SearchCompletion::Failed {
                    code: "fixture failure".into(),
                    message: "fixture diagnostic".into(),
                },
            ),
            "illegal" => session.complete(
                &ticket,
                SearchCompletion::Completed {
                    bestmove: Some("d2d4".into()),
                },
            ),
            _ => session.complete(&ticket, SearchCompletion::Completed { bestmove: None }),
        };
        assert_eq!(out.protocol, ["bestmove e2e4"], "{end}");
        assert!(has_code(&out, "LegalFallback"), "{end}");
        if end == "failure" {
            assert!(!out.accepted && has_code(&out, "SearchFailed"));
        } else if end == "illegal" {
            assert!(!out.accepted && has_code(&out, "IllegalBestMove"));
        }
        assert!(session.handle_line("stop").protocol.is_empty());
        assert_stale(session.complete(
            &ticket,
            SearchCompletion::Completed {
                bestmove: Some("d2d4".into()),
            },
        ));
    }
}

#[test]
fn changing_only_root_restriction_revokes_old_progress_and_completion() {
    let mut session = session(FixturePort::default());
    let (old, _) = start_line(&mut session, "go nodes 8 searchmoves e2e4");
    let out = session.handle_line("go nodes 8 searchmoves g1f3");
    assert!(out.accepted);
    let [
        Effect::Cancel {
            ticket,
            reason: CancelReason::ReplacedSearch,
        },
        Effect::Start {
            ticket: new,
            limits,
            ..
        },
    ] = out.effects.as_slice()
    else {
        panic!(
            "expected old cancellation and new constrained search: {:?}",
            out.effects
        );
    };
    assert_eq!(ticket, &old);
    assert_eq!(limits.searchmoves, Some(vec!["g1f3".into()]));
    let new = new.clone();
    assert_stale(session.progress(&old, "e2e4"));
    assert_stale(session.complete(
        &old,
        SearchCompletion::Completed {
            bestmove: Some("e2e4".into()),
        },
    ));
    assert!(!session.progress(&new, "e2e4").accepted);
    assert_eq!(session.handle_line("stop").protocol, ["bestmove g1f3"]);
}

#[test]
fn root_restrictions_use_full_promotion_castling_and_en_passant_move_text() {
    let mut session = session(FixturePort::default());
    assert!(
        session
            .handle_line(&format!("position fen {PROMOTION_FEN}"))
            .accepted
    );
    let (ticket, limits) = start_line(&mut session, "go nodes 1 searchmoves a7a8n a7a8r a7a8n");
    assert_eq!(
        limits.searchmoves,
        Some(vec!["a7a8r".into(), "a7a8n".into()])
    );
    assert!(!session.progress(&ticket, "a7a8q").accepted);
    assert_eq!(session.handle_line("stop").protocol, ["bestmove a7a8r"]);

    // These hand-authored Rules-port views isolate UCI handling. Actual chess
    // legality belongs to the independent product Rules integration checks.
    let port = FixturePort::default();
    *port.next_view.borrow_mut() = Some((vec!["e1g1".into(), "e5d6".into()], false));
    let mut special = self::session(port);
    let (_, limits) = start_line(&mut special, "go nodes 1 searchmoves e5d6 e1g1");
    assert_eq!(limits.searchmoves, Some(vec!["e1g1".into(), "e5d6".into()]));
    assert_eq!(special.handle_line("stop").protocol, ["bestmove e1g1"]);
}

#[test]
fn actual_rules_terminal_with_legal_moves_is_independent_of_a_valid_restriction() {
    let port = FixturePort::default();
    *port.next_view.borrow_mut() = Some((vec!["d2d4".into(), "e2e4".into()], true));
    let mut session = session(port);
    let out = session.handle_line("go nodes 1 searchmoves e2e4");
    assert!(out.accepted);
    assert_eq!(out.protocol, ["bestmove 0000"]);
    assert!(out.effects.is_empty() && session.active_ticket().is_none());
}

#[test]
fn unsupported_or_over_capability_goals_preserve_the_active_search() {
    let mut session = session(FixturePort::default());
    let ticket = start(&mut session);
    for line in ["go depth 1", "go mate 1"] {
        let out = session.handle_line(line);
        assert!(!out.accepted && has_code(&out, "UnsupportedSearchGoal"));
        assert!(out.effects.is_empty() && out.protocol.is_empty());
        assert_eq!(session.active_ticket(), Some(ticket.clone()));
    }
    assert!(
        session
            .set_search_goal_capability(NonZeroU16::new(8))
            .is_err()
    );
    assert_eq!(session.search_goal_capability(), None);
    session.handle_line("stop");
    session
        .set_search_goal_capability(NonZeroU16::new(8))
        .unwrap();
    let (ticket, limits) = start_line(&mut session, "go depth 8 mate 4 searchmoves e2e4");
    assert_eq!(limits.depth_plies, NonZeroU16::new(8));
    assert_eq!(limits.mate_moves, NonZeroU16::new(4));
    for line in ["go depth 9", "go mate 5"] {
        let out = session.handle_line(line);
        assert!(!out.accepted && has_code(&out, "InvalidSearchGoal"));
        assert!(out.effects.is_empty() && out.protocol.is_empty());
        assert_eq!(session.active_ticket(), Some(ticket.clone()));
    }
    assert_eq!(session.handle_line("stop").protocol, ["bestmove e2e4"]);
    session.handle_line("quit");
    assert!(session.set_search_goal_capability(None).is_err());
}

#[test]
fn constrained_infinite_completion_holds_only_an_allowed_move() {
    let mut session = session(FixturePort::default());
    session
        .set_search_goal_capability(NonZeroU16::new(8))
        .unwrap();
    let (ticket, _) = start_line(
        &mut session,
        "go infinite depth 3 mate 2 nodes 8 searchmoves g1f3",
    );
    let held = session.complete(
        &ticket,
        SearchCompletion::Completed {
            bestmove: Some("g1f3".into()),
        },
    );
    assert!(held.accepted && held.protocol.is_empty());
    assert_stale(session.progress(&ticket, "d2d4"));
    assert_eq!(session.handle_line("stop").protocol, ["bestmove g1f3"]);
    assert!(session.handle_line("stop").protocol.is_empty());
}

fn assert_stale(out: SessionResult<PositionSpec>) {
    assert!(!out.accepted);
    assert!(has_code(&out, "StaleSearch"));
    assert!(out.protocol.is_empty());
    assert!(out.effects.is_empty());
}

#[test]
fn handshake_readiness_and_diagnostics_use_separate_outputs() {
    let mut session = session(FixturePort::default());
    let out = session.handle_line("uci");
    assert_eq!(
        out.protocol,
        vec![
            "id name RoveZero fixture",
            "id author RoveZero",
            "option name Hash type spin default 16 min 1 max 64",
            "option name Use Cache type check default false",
            "option name Weights File type string default <empty>",
            "option name Search Policy type combo default PUCT var PUCT var Fixture",
            "option name Clear Hash type button",
            "uciok",
        ]
    );
    assert!(out.diagnostics.is_empty());
    assert!(out.effects.is_empty());
    let ticket = start(&mut session);
    assert_eq!(session.handle_line("isready").protocol, vec!["readyok"]);
    let unknown = session.handle_line("debug on");
    assert!(has_code(&unknown, "UnknownCommand"));
    assert!(unknown.protocol.is_empty());
    assert_eq!(session.active_ticket(), Some(ticket));
}

#[test]
fn whole_trace_semantic_rejection_preserves_old_snapshot_and_active_search() {
    let port = FixturePort::default();
    let requests = port.requests.clone();
    let mut session = session(port);
    assert!(session.handle_line("position startpos moves e2e4").accepted);
    let old_snapshot = session.snapshot().clone();
    let old_legal = session.legal_moves().to_vec();
    let ticket = start(&mut session);
    let rejected = session.handle_line("position startpos moves e2e4 e7e5 b1b3");
    assert!(!rejected.accepted);
    assert!(has_code(&rejected, "PositionRejected"));
    assert!(rejected.effects.is_empty());
    assert!(rejected.protocol.is_empty());
    assert_eq!(
        requests.borrow().last().unwrap().moves,
        vec!["e2e4", "e7e5", "b1b3"]
    );
    assert_eq!(session.snapshot(), &old_snapshot);
    assert_eq!(session.legal_moves(), old_legal);
    assert_eq!(session.active_ticket(), Some(ticket.clone()));
    assert_eq!(
        session
            .complete(
                &ticket,
                SearchCompletion::Completed {
                    bestmove: Some("e7e6".into())
                }
            )
            .protocol,
        vec!["bestmove e7e6"]
    );
}

#[test]
fn malformed_and_semantically_invalid_fen_preserve_original_search() {
    let port = FixturePort::default();
    let requests = port.requests.clone();
    let mut session = session(port);
    let ticket = start(&mut session);
    let original = session.snapshot().clone();
    let initial_requests = requests.borrow().len();
    let malformed = session.handle_line("position fen invalid");
    assert!(has_code(&malformed, "InvalidCommand"));
    assert_eq!(requests.borrow().len(), initial_requests);
    let kingless = session.handle_line("position fen 8/8/8/8/8/8/8/8 w - - 0 1");
    assert!(has_code(&kingless, "PositionRejected"));
    assert_eq!(requests.borrow().len(), initial_requests + 1);
    assert_eq!(session.snapshot(), &original);
    assert_eq!(session.active_ticket(), Some(ticket));
}

#[test]
fn illegal_progress_cannot_poison_legal_completion_or_stop() {
    let mut session = session(FixturePort::default());
    let ticket = start(&mut session);
    assert!(session.progress(&ticket, "e2e4").accepted);
    for movement in ["e2e5", "0000", "e2e4\nbestmove d2d4", "e2e4q"] {
        let out = session.progress(&ticket, movement);
        assert!(!out.accepted);
        assert!(has_code(&out, "IllegalBestMove"));
        assert!(out.protocol.is_empty());
    }
    let stopped = session.handle_line("stop");
    assert_eq!(stopped.protocol, vec!["bestmove e2e4"]);
    assert!(!has_code(&stopped, "LegalFallback"));
    assert!(
        matches!(stopped.effects.as_slice(), [Effect::Cancel { ticket: cancelled, reason: CancelReason::Stop }] if cancelled == &ticket)
    );
    assert!(session.handle_line("stop").protocol.is_empty());
    assert_stale(session.complete(
        &ticket,
        SearchCompletion::Completed {
            bestmove: Some("d2d4".into()),
        },
    ));
}

#[test]
fn rejected_progress_guard_preserves_ticket_and_prior_candidate() {
    let mut session = session(FixturePort::default());
    let ticket = start(&mut session);
    assert!(session.progress(&ticket, "g1f3").accepted);
    let invocations = Cell::new(0);
    let rejected = session.progress_with_guard(&ticket, "e2e4", || {
        invocations.set(invocations.get() + 1);
        false
    });
    assert_eq!(invocations.get(), 1);
    assert!(!rejected.accepted);
    assert!(has_code(&rejected, "AcceptanceClosed"));
    assert!(rejected.effects.is_empty());
    assert!(rejected.protocol.is_empty());
    assert_eq!(session.active_ticket(), Some(ticket));
    assert_eq!(session.handle_line("stop").protocol, vec!["bestmove g1f3"]);
}

#[test]
fn rejected_completion_guard_does_not_consume_ticket_before_next_valid_completion() {
    let mut session = session(FixturePort::default());
    let ticket = start(&mut session);
    assert!(session.progress(&ticket, "g1f3").accepted);
    let invocations = Cell::new(0);
    let rejected = session.complete_with_guard(
        &ticket,
        SearchCompletion::Completed {
            bestmove: Some("e2e4".into()),
        },
        || {
            invocations.set(invocations.get() + 1);
            false
        },
    );
    assert_eq!(invocations.get(), 1);
    assert!(!rejected.accepted);
    assert!(has_code(&rejected, "AcceptanceClosed"));
    assert!(rejected.protocol.is_empty());
    assert!(rejected.effects.is_empty());
    assert_eq!(session.active_ticket(), Some(ticket.clone()));
    // Empty accepted completion emits the previously admitted candidate. A
    // premature mutation from the rejected completion would emit e2e4 instead.
    let completed = session.complete_with_guard(
        &ticket,
        SearchCompletion::Completed { bestmove: None },
        || {
            invocations.set(invocations.get() + 1);
            true
        },
    );
    assert_eq!(invocations.get(), 2);
    assert!(completed.accepted);
    assert_eq!(completed.protocol, vec!["bestmove g1f3"]);
}

#[test]
fn rejected_failed_or_illegal_completion_preserves_cause_and_output_ownership() {
    for completion in [
        SearchCompletion::Failed {
            code: "EvaluatorNaN".into(),
            message: "non-finite WDL".into(),
        },
        SearchCompletion::Completed {
            bestmove: Some("g1f5".into()),
        },
    ] {
        let failed = matches!(completion, SearchCompletion::Failed { .. });
        let mut session = session(FixturePort::default());
        let ticket = start(&mut session);
        assert!(session.progress(&ticket, "g1f3").accepted);
        let invocations = Cell::new(0);
        let out = session.complete_with_guard(&ticket, completion, || {
            invocations.set(invocations.get() + 1);
            false
        });
        assert_eq!(invocations.get(), 1);
        assert!(!out.accepted);
        assert!(has_code(&out, "AcceptanceClosed"));
        assert!(has_code(
            &out,
            if failed {
                "SearchFailed"
            } else {
                "IllegalBestMove"
            }
        ));
        if failed {
            assert!(
                out.diagnostics
                    .iter()
                    .any(|d| d.message == "EvaluatorNaN: non-finite WDL")
            );
        }
        assert!(out.protocol.is_empty());
        assert!(out.effects.is_empty());
        assert_eq!(session.active_ticket(), Some(ticket));
        assert_eq!(session.handle_line("stop").protocol, vec!["bestmove g1f3"]);
    }
}

#[test]
fn invalid_progress_returns_before_guard_and_accepted_guards_run_once() {
    let mut session = session(FixturePort::default());
    let ticket = start(&mut session);
    let invocations = Cell::new(0);
    let invalid = session.progress_with_guard(&ticket, "g1f5", || {
        invocations.set(invocations.get() + 1);
        true
    });
    assert!(has_code(&invalid, "IllegalBestMove"));
    assert_eq!(invocations.get(), 0);
    assert!(
        session
            .progress_with_guard(&ticket, "e2e4", || {
                invocations.set(invocations.get() + 1);
                true
            })
            .accepted
    );
    assert_eq!(invocations.get(), 1);
    let completed = session.complete_with_guard(
        &ticket,
        SearchCompletion::Completed {
            bestmove: Some("g1f3".into()),
        },
        || {
            invocations.set(invocations.get() + 1);
            true
        },
    );
    assert_eq!(invocations.get(), 2);
    assert_eq!(completed.protocol, vec!["bestmove g1f3"]);
}

#[test]
fn cancellation_during_commit_guard_prevents_new_candidate_and_ticket_consumption() {
    let mut session = session(FixturePort::default());
    let ticket = start(&mut session);
    assert!(session.progress(&ticket, "g1f3").accepted);
    let canceled = AtomicBool::new(false);
    let out = session.complete_with_guard(
        &ticket,
        SearchCompletion::Completed {
            bestmove: Some("e2e4".into()),
        },
        || {
            // Deterministic boundary observer models cancellation after candidate
            // validation but before its acceptance and completion ownership commit.
            canceled.store(true, Ordering::Release);
            !canceled.load(Ordering::Acquire)
        },
    );
    assert!(!out.accepted);
    assert!(has_code(&out, "AcceptanceClosed"));
    assert!(out.protocol.is_empty());
    assert_eq!(session.active_ticket(), Some(ticket.clone()));
    assert_eq!(session.expire(&ticket).protocol, vec!["bestmove g1f3"]);
}

#[test]
fn normal_empty_completion_falls_back_in_rules_order_once() {
    let mut session = session(FixturePort::default());
    let ticket = start(&mut session);
    let out = session.complete(&ticket, SearchCompletion::Completed { bestmove: None });
    assert!(out.accepted);
    assert_eq!(out.protocol, vec!["bestmove d2d4"]);
    assert!(has_code(&out, "LegalFallback"));
    assert!(out.effects.is_empty());
    assert_stale(session.complete(&ticket, SearchCompletion::Completed { bestmove: None }));
    assert_stale(session.expire(&ticket));
}

#[test]
fn failed_search_retains_error_and_returns_only_a_valid_fallback() {
    let mut session = session(FixturePort::default());
    let ticket = start(&mut session);
    let out = session.complete(
        &ticket,
        SearchCompletion::Failed {
            code: "EvaluatorNaN".into(),
            message: "non-finite WDL".into(),
        },
    );
    assert!(!out.accepted);
    assert!(has_code(&out, "SearchFailed"));
    assert!(
        out.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("EvaluatorNaN: non-finite WDL"))
    );
    assert!(has_code(&out, "LegalFallback"));
    assert_eq!(out.protocol, vec!["bestmove d2d4"]);
}

#[test]
fn illegal_completed_move_retains_last_valid_progress_and_failure_diagnostic() {
    let mut session = session(FixturePort::default());
    let ticket = start(&mut session);
    assert!(session.progress(&ticket, "g1f3").accepted);
    let out = session.complete(
        &ticket,
        SearchCompletion::Completed {
            bestmove: Some("g1f5".into()),
        },
    );
    assert!(!out.accepted);
    assert!(has_code(&out, "IllegalBestMove"));
    assert_eq!(out.protocol, vec!["bestmove g1f3"]);
    assert!(!has_code(&out, "LegalFallback"));
}

#[test]
fn promotion_choice_is_checked_against_full_move_text() {
    let mut session = session(FixturePort::default());
    assert!(
        session
            .handle_line(&format!("position fen {PROMOTION_FEN}"))
            .accepted
    );
    let ticket = start(&mut session);
    assert!(!session.progress(&ticket, "a7a8").accepted);
    assert!(!session.progress(&ticket, "a7a8k").accepted);
    assert_eq!(
        session
            .complete(
                &ticket,
                SearchCompletion::Completed {
                    bestmove: Some("a7a8n".into())
                }
            )
            .protocol,
        vec!["bestmove a7a8n"]
    );
}

#[test]
fn exact_rules_terminals_emit_null_move_without_starting_evaluation() {
    for fen in [MATE_FEN, STALEMATE_FEN] {
        let mut session = session(FixturePort::default());
        assert!(session.handle_line(&format!("position fen {fen}")).accepted);
        let out = session.handle_line("go movetime 0");
        assert!(out.accepted);
        assert_eq!(out.protocol, vec!["bestmove 0000"]);
        assert!(out.effects.is_empty());
        assert!(out.diagnostics.is_empty());
        assert!(session.active_ticket().is_none());
    }
    // Automatic draws may still have legal moves; emptiness is not the
    // authority for the exact Rules terminal classification.
    let port = FixturePort::default();
    *port.next_view.borrow_mut() = Some((vec!["e2e4".into()], true));
    let mut session = session(port);
    let out = session.handle_line("go nodes 1");
    assert_eq!(out.protocol, vec!["bestmove 0000"]);
    assert!(out.effects.is_empty());
}

#[test]
fn invalid_rules_views_preserve_active_search_and_do_not_invent_a_terminal() {
    let port = FixturePort::default();
    let next_view = port.next_view.clone();
    let mut session = session(port);
    let ticket = start(&mut session);
    let original = session.snapshot().clone();
    for legal in [
        vec![],
        vec!["e2e4".into(), "e2e4".into()],
        vec!["0000".into()],
    ] {
        *next_view.borrow_mut() = Some((legal, false));
        let out = session.handle_line("position startpos moves e2e4");
        assert!(!out.accepted);
        assert!(has_code(&out, "InvalidPositionView"));
        assert!(out.effects.is_empty());
        assert!(out.protocol.is_empty());
        assert_eq!(session.snapshot(), &original);
        assert_eq!(session.active_ticket(), Some(ticket.clone()));
    }
    assert_eq!(session.handle_line("stop").protocol, vec!["bestmove d2d4"]);
}

#[test]
fn configurable_legal_view_budget_rejects_distinct_moves_before_replacement() {
    let port = FixturePort::default();
    let next_view = port.next_view.clone();
    let limits = ParserLimits {
        max_legal_moves: 3,
        ..ParserLimits::default()
    };
    let mut session = Session::new(port, identity(), options(), limits).unwrap();
    let ticket = start(&mut session);
    *next_view.borrow_mut() = Some((
        vec![
            "a7a8q".into(),
            "a7a8r".into(),
            "a7a8b".into(),
            "a7a8n".into(),
        ],
        false,
    ));
    let out = session.handle_line(&format!("position fen {PROMOTION_FEN}"));
    assert!(!out.accepted);
    assert!(has_code(&out, "InvalidPositionView"));
    assert!(out.effects.is_empty());
    assert_eq!(session.snapshot(), &PositionSpec::default());
    assert_eq!(session.active_ticket(), Some(ticket));
}

#[test]
fn invalid_initial_view_is_rejected_before_a_session_exists() {
    let port = FixturePort::default();
    *port.next_view.borrow_mut() = Some((vec![], false));
    assert!(Session::new(port, identity(), options(), ParserLimits::default()).is_err());
}

#[test]
fn opaque_tickets_reject_cross_session_stale_and_duplicate_results() {
    let mut first = session(FixturePort::default());
    let mut second = session(FixturePort::default());
    let first_ticket = start(&mut first);
    let second_ticket = start(&mut second);
    assert_ne!(first_ticket, second_ticket);
    assert_stale(second.progress(&first_ticket, "e2e4"));
    assert_stale(second.complete(
        &first_ticket,
        SearchCompletion::Completed {
            bestmove: Some("e2e4".into()),
        },
    ));
    assert_stale(second.expire(&first_ticket));
    assert_eq!(second.active_ticket(), Some(second_ticket.clone()));
    assert_eq!(
        second
            .complete(
                &second_ticket,
                SearchCompletion::Completed {
                    bestmove: Some("e2e4".into())
                }
            )
            .protocol,
        vec!["bestmove e2e4"]
    );
    assert_stale(second.progress(&second_ticket, "d2d4"));
    assert_stale(second.complete(
        &second_ticket,
        SearchCompletion::Completed { bestmove: None },
    ));
    assert_eq!(first.active_ticket(), Some(first_ticket));
}

#[test]
fn replacing_search_cancels_old_ticket_before_start_and_rejects_late_completion() {
    let mut session = session(FixturePort::default());
    let old_ticket = start(&mut session);
    let out = session.handle_line("go nodes 2");
    let new_ticket = match out.effects.as_slice() {
        [
            Effect::Cancel {
                ticket,
                reason: CancelReason::ReplacedSearch,
            },
            Effect::Start {
                ticket: new_ticket, ..
            },
        ] => {
            assert_eq!(ticket, &old_ticket);
            assert_ne!(new_ticket, &old_ticket);
            new_ticket.clone()
        }
        effects => panic!("unexpected replacement effects: {effects:?}"),
    };
    assert!(out.protocol.is_empty());
    assert_stale(session.complete(
        &old_ticket,
        SearchCompletion::Completed {
            bestmove: Some("e2e4".into()),
        },
    ));
    assert_eq!(session.active_ticket(), Some(new_ticket));
}

#[test]
fn valid_position_replacement_pins_old_snapshot_and_invalidates_old_ticket() {
    let mut session = session(FixturePort::default());
    let out = session.handle_line("go infinite");
    let (ticket, pinned_old) = match out.effects.as_slice() {
        [
            Effect::Start {
                ticket, snapshot, ..
            },
        ] => (ticket.clone(), snapshot.clone()),
        effects => panic!("unexpected effects: {effects:?}"),
    };
    let replacement = session.handle_line("position startpos moves e2e4 e7e5");
    assert!(
        matches!(replacement.effects.as_slice(), [Effect::Cancel { ticket: old, reason: CancelReason::ReplacedPosition }] if old == &ticket)
    );
    assert_eq!(pinned_old, PositionSpec::default());
    assert_eq!(session.snapshot().moves, vec!["e2e4", "e7e5"]);
    assert_stale(session.complete(
        &ticket,
        SearchCompletion::Completed {
            bestmove: Some("e2e4".into()),
        },
    ));
    let current = start(&mut session);
    assert_eq!(
        session
            .complete(
                &current,
                SearchCompletion::Completed {
                    bestmove: Some("f1c4".into())
                }
            )
            .protocol,
        vec!["bestmove f1c4"]
    );
}

#[test]
fn new_game_resets_position_cancels_worker_and_requests_game_state_clear() {
    let mut session = session(FixturePort::default());
    assert!(session.handle_line("position startpos moves e2e4").accepted);
    let ticket = start(&mut session);
    let out = session.handle_line("ucinewgame");
    assert!(
        matches!(out.effects.as_slice(), [Effect::Cancel { ticket: cancelled, reason: CancelReason::NewGame }, Effect::NewGame] if cancelled == &ticket)
    );
    assert_eq!(session.snapshot(), &PositionSpec::default());
    assert!(out.protocol.is_empty());
    assert_stale(session.progress(&ticket, "e7e5"));
    assert_stale(session.complete(&ticket, SearchCompletion::Completed { bestmove: None }));
}

#[test]
fn malformed_go_does_not_cancel_a_running_search() {
    let mut session = session(FixturePort::default());
    let ticket = start(&mut session);
    for (line, code) in [
        ("go movetime -1", "InvalidCommand"),
        ("go ponder", "PonderNeedsOwner"),
        ("ponderhit", "PonderNeedsOwner"),
        ("position startpos moves e2e9", "InvalidCommand"),
    ] {
        let out = session.handle_line(line);
        assert!(!out.accepted);
        assert!(has_code(&out, code));
        assert!(out.effects.is_empty());
        assert_eq!(session.active_ticket(), Some(ticket.clone()));
    }
}

#[test]
fn deadline_closes_admission_once_with_the_latest_valid_move() {
    let mut session = session(FixturePort::default());
    let ticket = start(&mut session);
    assert!(session.progress(&ticket, "g1f3").accepted);
    let out = session.expire(&ticket);
    assert_eq!(out.protocol, vec!["bestmove g1f3"]);
    assert!(
        matches!(out.effects.as_slice(), [Effect::Cancel { ticket: cancelled, reason: CancelReason::Deadline }] if cancelled == &ticket)
    );
    assert_stale(session.expire(&ticket));
    assert_stale(session.complete(
        &ticket,
        SearchCompletion::Completed {
            bestmove: Some("e2e4".into()),
        },
    ));
}

#[test]
fn infinite_completion_holds_candidate_until_stop_and_consumes_completion_once() {
    let mut session = session(FixturePort::default());
    let start = session.handle_line("go infinite");
    let ticket = match start.effects.as_slice() {
        [Effect::Start { ticket, limits, .. }] => {
            assert!(limits.infinite);
            ticket.clone()
        }
        effects => panic!("unexpected start effects: {effects:?}"),
    };
    let completed = session.complete(
        &ticket,
        SearchCompletion::Completed {
            bestmove: Some("g1f3".into()),
        },
    );
    assert!(completed.accepted);
    assert!(completed.protocol.is_empty());
    assert!(completed.effects.is_empty());
    assert_eq!(session.active_ticket(), Some(ticket.clone()));
    assert_eq!(session.handle_line("isready").protocol, vec!["readyok"]);
    assert_stale(session.complete(
        &ticket,
        SearchCompletion::Completed {
            bestmove: Some("e2e4".into()),
        },
    ));
    assert_stale(session.progress(&ticket, "d2d4"));
    assert_eq!(session.handle_line("stop").protocol, vec!["bestmove g1f3"]);
    assert!(session.handle_line("stop").protocol.is_empty());
    assert_stale(session.complete(&ticket, SearchCompletion::Completed { bestmove: None }));
}

#[test]
fn infinite_exact_terminal_waits_for_stop_without_starting_a_worker() {
    let mut session = session(FixturePort::default());
    assert!(
        session
            .handle_line(&format!("position fen {MATE_FEN}"))
            .accepted
    );
    let out = session.handle_line("go infinite");
    assert!(out.protocol.is_empty());
    assert!(out.effects.is_empty());
    let ticket = session
        .active_ticket()
        .expect("terminal search keeps output ownership");
    assert_eq!(session.handle_line("isready").protocol, vec!["readyok"]);
    assert_stale(session.complete(&ticket, SearchCompletion::Completed { bestmove: None }));
    let stopped = session.handle_line("stop");
    assert_eq!(stopped.protocol, vec!["bestmove 0000"]);
    assert!(stopped.effects.is_empty());
    assert!(session.handle_line("stop").protocol.is_empty());
}

#[test]
fn infinite_failure_preserves_error_and_exits_with_legal_fallback() {
    let mut session = session(FixturePort::default());
    session.handle_line("go infinite");
    let ticket = session.active_ticket().unwrap();
    let failed = session.complete(
        &ticket,
        SearchCompletion::Failed {
            code: "MockFailure".into(),
            message: "fixture failure".into(),
        },
    );
    assert!(!failed.accepted);
    assert!(has_code(&failed, "SearchFailed"));
    assert!(has_code(&failed, "LegalFallback"));
    assert_eq!(failed.protocol, vec!["bestmove d2d4"]);
    assert!(session.active_ticket().is_none());
    assert!(session.handle_line("stop").protocol.is_empty());
}

#[test]
fn infinite_empty_completion_defers_prior_candidate_or_fallback_until_stop() {
    for prior in [None, Some("g1f3")] {
        let mut session = session(FixturePort::default());
        session.handle_line("go infinite");
        let ticket = session.active_ticket().unwrap();
        if let Some(prior) = prior {
            assert!(session.progress(&ticket, prior).accepted);
        }
        let completed = session.complete(&ticket, SearchCompletion::Completed { bestmove: None });
        assert!(completed.accepted);
        assert!(completed.protocol.is_empty());
        assert!(completed.diagnostics.is_empty());
        let stopped = session.handle_line("stop");
        assert_eq!(
            stopped.protocol,
            vec![format!("bestmove {}", prior.unwrap_or("d2d4"))]
        );
        assert_eq!(has_code(&stopped, "LegalFallback"), prior.is_none());
    }
}

#[test]
fn held_infinite_completion_is_suppressed_on_replacement_newgame_or_quit() {
    for command in ["position startpos moves e2e4", "ucinewgame", "quit"] {
        let mut session = session(FixturePort::default());
        session.handle_line("go infinite");
        let ticket = session.active_ticket().unwrap();
        assert!(
            session
                .complete(
                    &ticket,
                    SearchCompletion::Completed {
                        bestmove: Some("g1f3".into())
                    }
                )
                .accepted
        );
        let out = session.handle_line(command);
        assert!(out.accepted);
        assert!(out.protocol.is_empty());
        assert!(
            matches!(out.effects.first(), Some(Effect::Cancel { ticket: old, .. }) if old == &ticket)
        );
        assert_stale(session.complete(&ticket, SearchCompletion::Completed { bestmove: None }));
    }
}

#[test]
fn terminal_infinite_replacement_newgame_and_quit_never_cancel_nonexistent_work() {
    for command in ["position startpos", "ucinewgame", "quit"] {
        let mut session = session(FixturePort::default());
        session.handle_line(&format!("position fen {MATE_FEN}"));
        session.handle_line("go infinite");
        let ticket = session.active_ticket().unwrap();
        let out = session.handle_line(command);
        assert!(out.accepted);
        assert!(out.protocol.is_empty());
        assert!(
            out.effects
                .iter()
                .all(|effect| !matches!(effect, Effect::Cancel { .. }))
        );
        assert_stale(session.expire(&ticket));
    }
}

#[test]
fn held_infinite_completion_at_deadline_outputs_candidate_once() {
    let mut session = session(FixturePort::default());
    session.handle_line("go infinite");
    let ticket = session.active_ticket().unwrap();
    assert!(
        session
            .complete(
                &ticket,
                SearchCompletion::Completed {
                    bestmove: Some("g1f3".into())
                }
            )
            .protocol
            .is_empty()
    );
    assert_eq!(session.expire(&ticket).protocol, vec!["bestmove g1f3"]);
    assert_stale(session.expire(&ticket));
    assert!(session.handle_line("stop").protocol.is_empty());
}

#[test]
fn quit_and_eof_close_once_and_reject_all_late_work() {
    for eof in [false, true] {
        let mut session = session(FixturePort::default());
        let ticket = start(&mut session);
        let out = if eof {
            session.end_of_input()
        } else {
            session.handle_line("quit")
        };
        let expected_reason = if eof {
            CancelReason::EndOfInput
        } else {
            CancelReason::Quit
        };
        assert!(
            matches!(out.effects.as_slice(), [Effect::Cancel { ticket: cancelled, reason }, Effect::Shutdown] if cancelled == &ticket && *reason == expected_reason)
        );
        assert!(session.is_closed());
        assert!(out.protocol.is_empty());
        assert!(session.end_of_input().effects.is_empty());
        for line in ["uci", "isready", "position startpos", "go nodes 1", "quit"] {
            let out = session.handle_line(line);
            assert!(!out.accepted);
            assert!(has_code(&out, "SessionClosed"));
            assert!(out.protocol.is_empty());
            assert!(out.effects.is_empty());
        }
        assert_stale(session.complete(
            &ticket,
            SearchCompletion::Completed {
                bestmove: Some("e2e4".into()),
            },
        ));
    }
}

#[test]
fn options_are_validated_transactionally_and_copied_into_search_start() {
    let mut session = session(FixturePort::default());
    for (line, name, expected) in [
        (
            "setoption name hAsH value 32",
            "Hash",
            OptionValue::Spin(32),
        ),
        (
            "setoption name Use Cache value true",
            "Use Cache",
            OptionValue::Check(true),
        ),
        (
            "setoption name Weights File value one two",
            "Weights File",
            OptionValue::String("one two".into()),
        ),
        (
            "setoption name Search Policy value Fixture",
            "Search Policy",
            OptionValue::Combo("Fixture".into()),
        ),
    ] {
        let out = session.handle_line(line);
        assert!(out.accepted, "{line}");
        assert_eq!(session.option_values().get(name), Some(&expected));
        assert!(
            matches!(out.effects.as_slice(), [Effect::OptionChanged { name: changed, value }] if changed == name && value == &expected)
        );
    }
    let before = session.option_values().clone();
    for line in [
        "setoption name Hash value 0",
        "setoption name Hash value 65",
        "setoption name Hash value NaN",
        "setoption name Hash",
        "setoption name Use Cache value yes",
        "setoption name Use Cache value TRUE",
        "setoption name Search Policy value Unknown",
        "setoption name Weights File value 123456789012345678901",
        "setoption name Clear Hash value 1",
        "setoption name Unknown value 1",
    ] {
        let out = session.handle_line(line);
        assert!(!out.accepted, "{line}");
        assert!(out.effects.is_empty());
        assert!(out.protocol.is_empty());
        assert_eq!(session.option_values(), &before);
    }
    let button = session.handle_line("setoption name Clear Hash");
    assert!(matches!(button.effects.as_slice(), [Effect::Button { name }] if name == "Clear Hash"));
    assert_eq!(session.option_values(), &before);
    let started = session.handle_line("go nodes 2");
    assert!(
        matches!(started.effects.as_slice(), [Effect::Start { options, .. }] if options == &before)
    );
    let active_change = session.handle_line("setoption name Hash value 16");
    assert!(has_code(&active_change, "SearchActive"));
    assert_eq!(session.option_values(), &before);
    assert!(active_change.effects.is_empty());
    assert!(session.handle_line("stop").accepted);
    assert!(
        session
            .handle_line("setoption name Weights File value <empty>")
            .accepted
    );
    assert_eq!(
        session.option_values().get("Weights File"),
        Some(&OptionValue::String(String::new()))
    );
}

#[test]
fn option_and_identity_registration_cannot_inject_protocol_lines() {
    for bad_identity in [
        EngineIdentity {
            name: "name\nuciok".into(),
            author: "author".into(),
        },
        EngineIdentity {
            name: "name".into(),
            author: "".into(),
        },
        EngineIdentity {
            name: "name".into(),
            author: "한글".into(),
        },
    ] {
        assert!(
            Session::new(
                FixturePort::default(),
                bad_identity,
                vec![],
                ParserLimits::default()
            )
            .is_err()
        );
    }
    let invalid_specs = vec![
        vec![OptionSpec {
            name: "Hash".into(),
            kind: OptionKind::Spin {
                default: 0,
                min: 1,
                max: 2,
            },
        }],
        vec![OptionSpec {
            name: "Has\nh".into(),
            kind: OptionKind::Button,
        }],
        vec![OptionSpec {
            name: "name Hash".into(),
            kind: OptionKind::Button,
        }],
        vec![OptionSpec {
            name: "Search type Policy".into(),
            kind: OptionKind::Button,
        }],
        vec![
            OptionSpec {
                name: "Hash".into(),
                kind: OptionKind::Button,
            },
            OptionSpec {
                name: "hAsH".into(),
                kind: OptionKind::Button,
            },
        ],
        vec![OptionSpec {
            name: "Policy".into(),
            kind: OptionKind::Combo {
                default: "Missing".into(),
                choices: vec!["PUCT".into()],
            },
        }],
        vec![OptionSpec {
            name: "Policy".into(),
            kind: OptionKind::Combo {
                default: "PUCT".into(),
                choices: vec!["PUCT".into(), "PUCT".into()],
            },
        }],
        vec![OptionSpec {
            name: "Policy".into(),
            kind: OptionKind::Combo {
                default: "var".into(),
                choices: vec!["var".into()],
            },
        }],
        vec![OptionSpec {
            name: "File".into(),
            kind: OptionKind::String {
                default: "1234".into(),
                max_bytes: 3,
            },
        }],
        vec![OptionSpec {
            name: "File".into(),
            kind: OptionKind::String {
                default: String::new(),
                max_bytes: 1025,
            },
        }],
    ];
    for specs in invalid_specs {
        assert!(
            Session::new(
                FixturePort::default(),
                identity(),
                specs,
                ParserLimits::default()
            )
            .is_err()
        );
    }
}

#[test]
fn full_session_transcript_remains_responsive_across_errors_and_cancellation() {
    let mut session = session(FixturePort::default());
    let mut protocol = session.handle_line("uci").protocol;
    assert!(session.handle_line("setoption name Hash value 24").accepted);
    protocol.extend(session.handle_line("isready").protocol);
    assert!(
        session
            .handle_line("position startpos moves e2e4 e7e5")
            .accepted
    );
    let old = start(&mut session);
    protocol.extend(session.handle_line("isready").protocol);
    assert!(
        !session
            .handle_line("position startpos moves e2e4 e7e5 b1b3")
            .accepted
    );
    assert!(session.progress(&old, "g1f3").accepted);
    protocol.extend(session.handle_line("stop").protocol);
    assert_stale(session.complete(
        &old,
        SearchCompletion::Completed {
            bestmove: Some("f1c4".into()),
        },
    ));
    assert!(session.handle_line("ucinewgame").accepted);
    let current = start(&mut session);
    protocol.extend(
        session
            .complete(
                &current,
                SearchCompletion::Failed {
                    code: "MockFailure".into(),
                    message: "fixture failure".into(),
                },
            )
            .protocol,
    );
    assert!(session.handle_line("quit").accepted);
    assert_eq!(
        &protocol[8..],
        ["readyok", "readyok", "bestmove g1f3", "bestmove d2d4"]
    );
    assert!(session.is_closed());
}
