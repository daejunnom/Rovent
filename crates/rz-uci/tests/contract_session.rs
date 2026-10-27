use rz_contracts::{
    AcceptanceScope, ClockDomain, Color, ContractError, Digest, EncodingHandle, ErrorCode,
    GameGeneration, ModelHandle, OwnerId, ProcessEpoch, RootGeneration, SlotGeneration, Stage,
};
use rz_search::contract_time::InstantClock;
use rz_search::driver::StopReason;
use rz_search::time::TimeBudgetConfig;
use rz_uci::bridge::{BuildSearchSettings, NodeBudgetSemantics};
use rz_uci::contracts::{ContractSessionOutcome, ContractSessionOwner};
use rz_uci::{
    CancelReason, Effect, EngineIdentity, Event, ParserLimits, PositionBase, PositionPort,
    PositionSpec, PreparedPosition, SearchCompletion, SearchTicket, Session,
};
use std::num::NonZeroU16;
use std::time::{Duration, Instant};

const EPOCH: ProcessEpoch = ProcessEpoch(7000);
const MATE_FEN: &str = "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1";

#[derive(Clone, Debug, Eq, PartialEq)]
struct Snapshot {
    specification: PositionSpec,
    side: Color,
}

/// A small hand-authored Rules port isolates lifecycle/contract checks. These
/// fixture states and handles are not evidence of product Rules or a registry.
struct FixtureRules;

impl PositionPort for FixtureRules {
    type Snapshot = Snapshot;
    type Error = &'static str;

    fn prepare(&self, spec: &PositionSpec) -> Result<PreparedPosition<Snapshot>, Self::Error> {
        let (side, moves, exact_terminal): (Color, &[&str], bool) = match &spec.base {
            PositionBase::StartPos => {
                match spec
                    .moves
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .as_slice()
                {
                    [] => (Color::White, &["e2e4", "d2d4"], false),
                    ["e2e4"] => (Color::Black, &["e7e5", "c7c5"], false),
                    ["e2e4", "e7e5"] => (Color::White, &["g1f3", "f1c4"], false),
                    _ => return Err("complete fixture trace contains an illegal transition"),
                }
            }
            PositionBase::Fen(fen) if fen == MATE_FEN && spec.moves.is_empty() => {
                (Color::Black, &[], true)
            }
            _ => return Err("fixture FEN semantics rejected"),
        };
        Ok(PreparedPosition {
            snapshot: Snapshot {
                specification: spec.clone(),
                side,
            },
            legal_moves: moves.iter().map(|movement| (*movement).into()).collect(),
            exact_terminal,
        })
    }
}

fn session() -> Session<FixtureRules> {
    Session::new(
        FixtureRules,
        EngineIdentity {
            name: "Contract fixture".into(),
            author: "B".into(),
        },
        vec![],
        ParserLimits::default(),
    )
    .unwrap()
}

fn fixture_scope() -> AcceptanceScope {
    AcceptanceScope {
        game: GameGeneration(17),
        root: RootGeneration(31),
        model: ModelHandle {
            owner: OwnerId(1),
            slot: 2,
            generation: SlotGeneration(3),
            manifest: Digest([4; 32]),
        },
        encoding: EncodingHandle {
            owner: OwnerId(5),
            slot: 6,
            generation: SlotGeneration(7),
            manifest: Digest([8; 32]),
        },
        backend: Digest([9; 32]),
    }
}

fn settings() -> BuildSearchSettings {
    BuildSearchSettings {
        time_config: TimeBudgetConfig::default(),
        max_simulations: 128,
        untimed_limit: Duration::from_secs(30),
    }
}

fn owner(scope: AcceptanceScope) -> ContractSessionOwner {
    ContractSessionOwner::new(
        scope,
        EPOCH,
        InstantClock::new(ClockDomain(EPOCH), Instant::now()),
        settings(),
    )
    .unwrap()
}

fn checked_side(snapshot: &Snapshot) -> Result<Color, ContractError> {
    Ok(snapshot.side)
}

fn line(
    owner: &mut ContractSessionOwner,
    session: &mut Session<FixtureRules>,
    text: &str,
) -> ContractSessionOutcome<Snapshot> {
    owner.handle_line(session, text, checked_side)
}

fn begin(
    owner: &mut ContractSessionOwner,
    session: &mut Session<FixtureRules>,
    command: &str,
) -> SearchTicket {
    let out = line(owner, session, command);
    assert!(out.session.accepted, "{out:?}");
    assert!(out.errors.is_empty(), "{out:?}");
    assert!(out.session.protocol.is_empty());
    match out.session.effects.as_slice() {
        [Effect::Start { ticket, .. }] => {
            assert_eq!(owner.active_ticket(), Some(ticket));
            assert_eq!(session.active_ticket().as_ref(), Some(ticket));
            ticket.clone()
        }
        effects => panic!("expected one common-backed Start, got {effects:?}"),
    }
}

fn assert_scope(actual: AcceptanceScope, expected: AcceptanceScope) {
    assert_eq!(actual.game, expected.game);
    assert_eq!(actual.root, expected.root);
    assert_eq!(actual.model, expected.model);
    assert_eq!(actual.encoding, expected.encoding);
    assert_eq!(actual.backend, expected.backend);
}

fn assert_typed_error(out: &ContractSessionOutcome<Snapshot>, code: ErrorCode) {
    assert!(!out.session.accepted);
    assert!(
        out.errors
            .iter()
            .any(|error| error.code == code && error.stage == Stage::Admission),
        "{out:?}"
    );
    assert!(
        out.session
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "ContractFailure")
    );
}

#[test]
fn default_go_announces_registered_limits_and_stop_uses_restricted_root() {
    for (command, expected_move) in [("go", "e2e4"), ("go searchmoves d2d4", "d2d4")] {
        let mut owner = owner(fixture_scope());
        let mut session = session();
        let started = line(&mut owner, &mut session, command);
        assert!(started.session.accepted, "{started:?}");
        assert_eq!(
            started.session.protocol,
            [
                "info string go uses configured resource wall at most 30000 ms and simulation limit 128"
            ]
        );
        assert!(matches!(
            started.session.effects.as_slice(),
            [Effect::Start { .. }]
        ));
        let stopped = line(&mut owner, &mut session, "stop");
        assert_eq!(
            stopped.session.protocol,
            [format!("bestmove {expected_move}")]
        );
        assert!(owner.active_ticket().is_none());
        assert!(
            line(&mut owner, &mut session, "stop")
                .session
                .protocol
                .is_empty()
        );
    }

    let mut owner = owner(fixture_scope());
    let mut session = session();
    session
        .set_node_budget_semantics(NodeBudgetSemantics::RoventPositionVisits)
        .unwrap();
    let started = line(&mut owner, &mut session, "go searchmoves d2d4");
    assert!(started.session.accepted, "{started:?}");
    assert_eq!(
        started.session.protocol,
        [
            "info string go uses configured resource wall at most 30000 ms; other search limits follow the registered engine profile"
        ]
    );
    assert_eq!(
        line(&mut owner, &mut session, "stop").session.protocol,
        ["bestmove d2d4"]
    );
}

#[test]
fn rejected_restrictions_and_unregistered_goals_preserve_common_authority_and_budget() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let ticket = begin(&mut owner, &mut session, "go nodes 8 searchmoves e2e4");
    let scope = owner.scope();
    let deadline = owner.control().unwrap().deadline;
    for (command, code) in [
        ("go nodes 1 searchmoves e2e4 e2e5", "InvalidSearchMoves"),
        ("go depth 1", "UnsupportedSearchGoal"),
        ("go mate 1", "UnsupportedSearchGoal"),
    ] {
        let out = line(&mut owner, &mut session, command);
        assert!(!out.session.accepted);
        assert!(
            out.session
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == code)
        );
        assert!(out.session.effects.is_empty() && out.session.protocol.is_empty());
        assert!(out.errors.is_empty());
        assert_scope(owner.scope(), scope);
        assert_eq!(owner.active_ticket(), Some(&ticket));
        assert_eq!(owner.control().unwrap().deadline, deadline);
        assert!(
            !owner
                .control()
                .unwrap()
                .cancellation
                .load(std::sync::atomic::Ordering::Acquire)
        );
    }
    assert_eq!(
        line(&mut owner, &mut session, "stop").session.protocol,
        ["bestmove e2e4"]
    );
}

#[test]
fn registered_goal_binding_and_canonical_root_restriction_reach_the_start_effect() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    session
        .set_search_goal_capability(NonZeroU16::new(8))
        .unwrap();
    let out = line(
        &mut owner,
        &mut session,
        "go mate 3 depth 6 searchmoves d2d4 e2e4 d2d4",
    );
    assert!(out.session.accepted && out.errors.is_empty(), "{out:?}");
    let [Effect::Start { limits, .. }] = out.session.effects.as_slice() else {
        panic!(
            "expected goal-aware Start effect: {:?}",
            out.session.effects
        );
    };
    assert_eq!(limits.searchmoves, Some(vec!["e2e4".into(), "d2d4".into()]));
    let binding = owner.active_binding().unwrap();
    assert_eq!(
        binding.goal(),
        rz_contracts::rovent::SearchGoal {
            depth_plies: NonZeroU16::new(6),
            mate_moves: NonZeroU16::new(3),
        }
    );
    assert_eq!(
        binding.budget_kind(),
        rz_uci::bridge::BudgetKind::ResourceWall
    );
    assert_eq!(
        binding.control().max_simulations,
        settings().max_simulations
    );
    assert_eq!(
        line(&mut owner, &mut session, "stop").session.protocol,
        ["bestmove e2e4"]
    );
}

#[test]
fn rovent_position_nodes_preserve_original_values_without_becoming_simulations() {
    for nodes in [0, 1, u64::MAX] {
        let mut owner = owner(fixture_scope());
        let mut session = session();
        session
            .set_node_budget_semantics(NodeBudgetSemantics::RoventPositionVisits)
            .unwrap();
        session
            .set_search_goal_capability(NonZeroU16::new(8))
            .unwrap();
        let out = line(
            &mut owner,
            &mut session,
            &format!("go nodes {nodes} depth 4 searchmoves d2d4 e2e4 d2d4"),
        );
        assert!(out.session.accepted && out.errors.is_empty(), "{out:?}");
        let [Effect::Start { limits, .. }] = out.session.effects.as_slice() else {
            panic!(
                "expected exactly one original Start: {:?}",
                out.session.effects
            );
        };
        assert_eq!(limits.nodes, Some(nodes));
        assert_eq!(limits.searchmoves, Some(vec!["e2e4".into(), "d2d4".into()]));
        let binding = owner.active_binding().unwrap();
        assert_eq!(
            binding.control().max_simulations,
            settings().max_simulations
        );
        assert_eq!(binding.goal().depth_plies, NonZeroU16::new(4));
        assert_eq!(
            binding.node_budget_semantics(),
            NodeBudgetSemantics::RoventPositionVisits
        );
    }
}

#[test]
fn rovent_failed_pure_admission_keeps_prior_ticket_clock_scope_and_cancellation() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    session
        .set_node_budget_semantics(NodeBudgetSemantics::RoventPositionVisits)
        .unwrap();
    session
        .set_search_goal_capability(NonZeroU16::new(8))
        .unwrap();
    let ticket = begin(&mut owner, &mut session, "go nodes 1 searchmoves e2e4");
    let scope = owner.scope();
    let deadline = owner.control().unwrap().deadline;
    let token = owner.cancellation().unwrap().token();
    for command in [
        "go movetime 0",
        "go depth 9",
        "go movetime 18446744073709551615",
    ] {
        let out = line(&mut owner, &mut session, command);
        assert!(!out.session.accepted, "{command}: {out:?}");
        assert!(!out.errors.is_empty(), "{out:?}");
        assert!(out.session.protocol.is_empty() && out.session.effects.is_empty());
        assert_eq!(owner.active_ticket(), Some(&ticket));
        assert_eq!(session.active_ticket(), Some(ticket.clone()));
        assert_eq!(owner.control().unwrap().deadline, deadline);
        assert_scope(owner.scope(), scope);
        assert!(!token.is_canceled());
    }
    let failure = ContractError::new(
        ErrorCode::IdentityMismatch,
        Stage::Admission,
        "checked fixture side is unavailable before admission",
    );
    let out = owner.handle_line(&mut session, "go nodes 2", |_| Err(failure));
    assert_typed_error(&out, ErrorCode::IdentityMismatch);
    assert!(out.session.effects.is_empty() && out.session.protocol.is_empty());
    assert_eq!(owner.active_ticket(), Some(&ticket));
    assert_eq!(owner.control().unwrap().deadline, deadline);
    assert_scope(owner.scope(), scope);
    assert!(!token.is_canceled());
}

#[test]
fn rovent_prepared_time_binding_reads_the_original_snapshot_side_once() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    session
        .set_node_budget_semantics(NodeBudgetSemantics::RoventPositionVisits)
        .unwrap();
    let calls = std::cell::Cell::new(0);
    let out = owner.handle_line(
        &mut session,
        "go wtime 10000 btime 20000 winc 3 binc 7 nodes 1",
        |snapshot| {
            calls.set(calls.get() + 1);
            checked_side(snapshot)
        },
    );
    assert!(out.session.accepted && out.errors.is_empty(), "{out:?}");
    assert_eq!(calls.get(), 1);
    assert!(matches!(
        out.session.effects.as_slice(),
        [Effect::Start { .. }]
    ));
    assert_eq!(
        owner.control().unwrap().max_simulations,
        settings().max_simulations
    );
}

#[test]
fn accepted_root_commands_and_new_game_advance_only_their_owned_generations() {
    let initial = fixture_scope();
    let mut owner = owner(initial);
    let mut session = session();
    for command in ["uci", "isready", "stop"] {
        assert!(line(&mut owner, &mut session, command).session.accepted);
        assert_scope(owner.scope(), initial);
    }
    assert!(
        line(&mut owner, &mut session, "position startpos moves e2e4")
            .session
            .accepted
    );
    let after_position = AcceptanceScope {
        root: RootGeneration(32),
        ..initial
    };
    assert_scope(owner.scope(), after_position);
    let ticket = begin(&mut owner, &mut session, "go nodes 64");
    let after_go = AcceptanceScope {
        root: RootGeneration(33),
        ..initial
    };
    assert_scope(owner.scope(), after_go);
    assert_scope(owner.active_scope().unwrap(), after_go);
    let token = owner.cancellation().unwrap().token();
    let out = line(&mut owner, &mut session, "ucinewgame");
    assert!(out.session.accepted);
    assert!(
        matches!(out.session.effects.as_slice(), [Effect::Cancel { ticket: old, reason: CancelReason::NewGame }, Effect::NewGame] if old == &ticket)
    );
    assert!(token.is_canceled());
    assert_eq!(session.snapshot().specification, PositionSpec::default());
    assert_scope(
        owner.scope(),
        AcceptanceScope {
            game: GameGeneration(18),
            root: RootGeneration(34),
            ..initial
        },
    );
    assert!(owner.active_ticket().is_none());
    assert!(owner.active_binding().is_none());
}

#[test]
fn malformed_and_rejected_whole_trace_preserve_live_scope_state_and_token() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    assert!(
        line(&mut owner, &mut session, "position startpos moves e2e4")
            .session
            .accepted
    );
    let ticket = begin(&mut owner, &mut session, "go nodes 64");
    let original_scope = owner.scope();
    let original_snapshot = session.snapshot().clone();
    let token = owner.cancellation().unwrap().token();
    for command in [
        "position fen malformed",
        "position startpos moves e2e9",
        "position startpos moves e2e4 e7e5 b1b3",
        "go nodes 0",
        "go ponder",
        "stop extra",
    ] {
        let out = line(&mut owner, &mut session, command);
        assert!(!out.session.accepted, "{command}");
        assert!(out.session.protocol.is_empty());
        assert!(out.session.effects.is_empty());
        assert!(out.errors.is_empty());
        assert_scope(owner.scope(), original_scope);
        assert_eq!(session.snapshot(), &original_snapshot);
        assert_eq!(owner.active_ticket(), Some(&ticket));
        assert!(!token.is_canceled());
    }
    let completed = owner.handle_event(
        &mut session,
        Event::Complete {
            ticket,
            completion: SearchCompletion::Completed {
                bestmove: Some("c7c5".into()),
            },
        },
    );
    assert_eq!(completed.session.protocol, vec!["bestmove c7c5"]);
}

#[test]
fn root_overflow_rejects_before_canceling_or_replacing_an_active_search() {
    let initial = AcceptanceScope {
        root: RootGeneration(u64::MAX - 1),
        ..fixture_scope()
    };
    let mut owner = owner(initial);
    let mut session = session();
    let ticket = begin(&mut owner, &mut session, "go nodes 64");
    let current_scope = owner.scope();
    let token = owner.cancellation().unwrap().token();
    for command in ["position startpos moves e2e4", "go nodes 2", "ucinewgame"] {
        let out = line(&mut owner, &mut session, command);
        assert_typed_error(&out, ErrorCode::ResourceExhausted);
        assert!(out.session.protocol.is_empty());
        assert!(out.session.effects.is_empty());
        assert_scope(owner.scope(), current_scope);
        assert_eq!(session.snapshot().specification, PositionSpec::default());
        assert_eq!(owner.active_ticket(), Some(&ticket));
        assert!(!token.is_canceled());
    }
    assert_eq!(
        line(&mut owner, &mut session, "stop").session.protocol,
        vec!["bestmove e2e4"]
    );
    assert!(token.is_canceled());
}

#[test]
fn game_overflow_preserves_root_and_active_authority() {
    let initial = AcceptanceScope {
        game: GameGeneration(u64::MAX),
        ..fixture_scope()
    };
    let mut owner = owner(initial);
    let mut session = session();
    assert!(
        line(&mut owner, &mut session, "position startpos moves e2e4")
            .session
            .accepted
    );
    let ticket = begin(&mut owner, &mut session, "go nodes 64");
    let before = owner.scope();
    let token = owner.cancellation().unwrap().token();
    let snapshot = session.snapshot().clone();
    let out = line(&mut owner, &mut session, "ucinewgame");
    assert_typed_error(&out, ErrorCode::ResourceExhausted);
    assert!(out.session.effects.is_empty());
    assert_scope(owner.scope(), before);
    assert_eq!(session.snapshot(), &snapshot);
    assert_eq!(owner.active_ticket(), Some(&ticket));
    assert!(!token.is_canceled());
}

#[test]
fn one_epoch_origin_maps_all_owned_deadlines_without_utc_or_reset() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    begin(&mut owner, &mut session, "go movetime 10000");
    assert_eq!(owner.process_epoch(), EPOCH);
    let budget = owner.active_binding().unwrap().budget();
    let deadlines = owner.deadlines().unwrap();
    for (deadline, instant) in [
        (deadlines.soft, budget.soft_deadline),
        (deadlines.admission, budget.admission_deadline),
        (deadlines.hard, budget.hard_deadline),
        (deadlines.output, budget.output_deadline),
    ] {
        assert_eq!(deadline.clock, ClockDomain(EPOCH));
        assert_eq!(deadline.at, owner.clock().tick_at(instant).unwrap());
        assert_eq!(owner.clock().instant_at(deadline.at).unwrap(), instant);
    }
    assert_eq!(
        budget.hard_deadline.duration_since(budget.start),
        Duration::from_millis(9990)
    );
    assert_eq!(
        budget.admission_deadline.duration_since(budget.start),
        Duration::from_millis(9980)
    );
}

#[test]
fn checked_snapshot_color_selects_the_clock_after_the_complete_trace() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    assert!(
        line(&mut owner, &mut session, "position startpos moves e2e4")
            .session
            .accepted
    );
    begin(
        &mut owner,
        &mut session,
        "go wtime 60000 btime 30000 winc 1000 binc 2000 movestogo 10",
    );
    // Black: soft=30000/10+2000*0.8=4600ms, hard=13800ms.
    assert_eq!(
        owner.active_binding().unwrap().budget().allocation,
        Duration::from_millis(13800)
    );
}

#[test]
fn result_at_the_exact_hard_tick_preserves_prior_candidate_and_closes_both_tokens() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let ticket = begin(&mut owner, &mut session, "go movetime 10000");
    let previous = owner.handle_event(
        &mut session,
        Event::Progress {
            ticket: ticket.clone(),
            bestmove: "d2d4".into(),
        },
    );
    assert!(previous.session.accepted);
    let hard = owner.active_binding().unwrap().budget().hard_deadline;
    let token = owner.cancellation().unwrap().token();
    let control = owner.control().unwrap().clone();
    let out = owner.handle_event_at(
        &mut session,
        Event::Complete {
            ticket: ticket.clone(),
            completion: SearchCompletion::Completed {
                bestmove: Some("e2e4".into()),
            },
        },
        hard,
    );
    assert_typed_error(&out, ErrorCode::Expired);
    assert_eq!(out.session.protocol, vec!["bestmove d2d4"]);
    assert!(token.is_canceled());
    assert_eq!(
        control.stop_reason(Instant::now()),
        Some(StopReason::Canceled)
    );
    assert!(owner.active_ticket().is_none());
    let duplicate = owner.handle_event(
        &mut session,
        Event::Complete {
            ticket,
            completion: SearchCompletion::Completed {
                bestmove: Some("e2e4".into()),
            },
        },
    );
    assert_typed_error(&duplicate, ErrorCode::Stale);
    assert!(duplicate.session.protocol.is_empty());
}

#[test]
fn early_timer_is_typed_rejection_and_cannot_close_live_admission() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let ticket = begin(&mut owner, &mut session, "go movetime 10000");
    let start = owner.active_binding().unwrap().budget().start;
    let token = owner.cancellation().unwrap().token();
    let out = owner.handle_event_at(&mut session, Event::Deadline(ticket.clone()), start);
    assert_typed_error(&out, ErrorCode::InvalidInput);
    assert!(out.session.protocol.is_empty());
    assert!(out.session.effects.is_empty());
    assert_eq!(owner.active_ticket(), Some(&ticket));
    assert!(!token.is_canceled());
}

#[test]
fn canceled_finite_worker_still_finishes_owned_hard_deadline_once() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let ticket = begin(&mut owner, &mut session, "go movetime 10000");
    assert!(
        owner
            .handle_event(
                &mut session,
                Event::Progress {
                    ticket: ticket.clone(),
                    bestmove: "d2d4".into()
                }
            )
            .session
            .accepted
    );
    let hard = owner.active_binding().unwrap().budget().hard_deadline;
    let token = owner.cancellation().unwrap().token();
    token.cancel();
    let canceled = owner.handle_event(
        &mut session,
        Event::Complete {
            ticket: ticket.clone(),
            completion: SearchCompletion::Failed {
                code: "CanceledProviderSource".into(),
                message: "original source survives".into(),
            },
        },
    );
    assert_typed_error(&canceled, ErrorCode::Canceled);
    assert!(canceled.session.protocol.is_empty());
    assert!(
        canceled
            .session
            .diagnostics
            .iter()
            .any(|d| d.code == "SearchFailed"
                && d.message == "CanceledProviderSource: original source survives")
    );
    let expired = owner.handle_event_at(&mut session, Event::Deadline(ticket.clone()), hard);
    assert_typed_error(&expired, ErrorCode::Expired);
    assert_eq!(expired.session.protocol, vec!["bestmove d2d4"]);
    assert!(owner.active_ticket().is_none());
    assert!(token.is_canceled());
    assert!(
        owner
            .handle_event_at(&mut session, Event::Deadline(ticket), hard)
            .session
            .protocol
            .is_empty()
    );
    assert!(
        line(&mut owner, &mut session, "stop")
            .session
            .protocol
            .is_empty()
    );
}

#[test]
fn early_timer_after_completed_infinite_worker_keeps_passive_authority_open() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let ticket = begin(&mut owner, &mut session, "go infinite");
    let start = owner.active_binding().unwrap().budget().start;
    let token = owner.cancellation().unwrap().token();
    assert!(
        owner
            .handle_event(
                &mut session,
                Event::Complete {
                    ticket: ticket.clone(),
                    completion: SearchCompletion::Completed {
                        bestmove: Some("d2d4".into())
                    }
                }
            )
            .session
            .accepted
    );
    let early = owner.handle_event_at(&mut session, Event::Deadline(ticket.clone()), start);
    assert_typed_error(&early, ErrorCode::InvalidInput);
    assert!(early.session.protocol.is_empty());
    assert!(early.session.effects.is_empty());
    assert!(!token.is_canceled());
    assert_eq!(owner.active_ticket(), Some(&ticket));
    assert_eq!(
        line(&mut owner, &mut session, "stop").session.protocol,
        vec!["bestmove d2d4"]
    );
}

#[test]
fn externally_canceled_common_token_blocks_callback_and_mirrors_legacy_control() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let ticket = begin(&mut owner, &mut session, "go nodes 64");
    let token = owner.cancellation().unwrap().token();
    let control = owner.control().unwrap().clone();
    token.cancel();
    let out = owner.handle_event(
        &mut session,
        Event::Complete {
            ticket: ticket.clone(),
            completion: SearchCompletion::Completed {
                bestmove: Some("d2d4".into()),
            },
        },
    );
    assert_typed_error(&out, ErrorCode::Canceled);
    assert!(out.session.protocol.is_empty());
    assert_eq!(
        control.stop_reason(Instant::now()),
        Some(StopReason::Canceled)
    );
    assert_eq!(owner.active_ticket(), Some(&ticket));
    assert_eq!(
        line(&mut owner, &mut session, "stop").session.protocol,
        vec!["bestmove e2e4"]
    );
}

#[test]
fn cancellation_at_session_commit_guard_preserves_candidate_and_output_owner() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let ticket = begin(&mut owner, &mut session, "go nodes 64");
    let progress = owner.handle_event(
        &mut session,
        Event::Progress {
            ticket: ticket.clone(),
            bestmove: "d2d4".into(),
        },
    );
    assert!(progress.session.accepted);
    let token = owner.cancellation().unwrap().token();
    let rejected = session.complete_with_guard(
        &ticket,
        SearchCompletion::Completed {
            bestmove: Some("e2e4".into()),
        },
        || {
            // Cancel after candidate preparation, at the final commit guard.
            token.cancel();
            !token.is_canceled()
        },
    );
    assert!(!rejected.accepted);
    assert!(rejected.protocol.is_empty());
    assert_eq!(session.active_ticket(), Some(ticket.clone()));
    let canceled = owner.handle_event(
        &mut session,
        Event::Complete {
            ticket,
            completion: SearchCompletion::Completed {
                bestmove: Some("e2e4".into()),
            },
        },
    );
    assert_typed_error(&canceled, ErrorCode::Canceled);
    assert_eq!(
        line(&mut owner, &mut session, "stop").session.protocol,
        vec!["bestmove d2d4"]
    );
}

#[test]
fn dropping_owner_revokes_retained_common_and_worker_control_clones() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    begin(&mut owner, &mut session, "go nodes 64");
    let token = owner.cancellation().unwrap().token();
    let control = owner.control().unwrap().clone();
    drop(owner);
    assert!(token.is_canceled());
    assert_eq!(
        control.stop_reason(Instant::now()),
        Some(StopReason::Canceled)
    );
    // Dropping logical acceptance does not pretend that Session or physical
    // Runtime shutdown/drain has completed.
    assert!(!session.is_closed());
}

#[test]
fn older_dispatch_tick_cannot_reopen_the_owned_result_boundary() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let ticket = begin(&mut owner, &mut session, "go movetime 10000");
    let budget = *owner.active_binding().unwrap().budget();
    let marker = owner.handle_event_at(
        &mut session,
        Event::RejectedInput {
            code: "FixtureMarker",
            message: "owner observed hard boundary".into(),
        },
        budget.hard_deadline,
    );
    assert!(!marker.session.accepted);
    assert_eq!(owner.active_ticket(), Some(&ticket));
    let old_tick = owner.handle_event_at(
        &mut session,
        Event::Complete {
            ticket,
            completion: SearchCompletion::Completed {
                bestmove: Some("d2d4".into()),
            },
        },
        budget.start,
    );
    assert_typed_error(&old_tick, ErrorCode::Expired);
    assert_eq!(old_tick.session.protocol, vec!["bestmove e2e4"]);
}

#[test]
fn stop_replacement_position_newgame_quit_and_eof_revoke_common_clones() {
    for command in [
        "stop",
        "go nodes 2",
        "position startpos moves e2e4",
        "ucinewgame",
        "quit",
        "EOF",
    ] {
        let mut owner = owner(fixture_scope());
        let mut session = session();
        let ticket = begin(&mut owner, &mut session, "go nodes 64");
        let token = owner.cancellation().unwrap().token();
        let control = owner.control().unwrap().clone();
        let out = if command == "EOF" {
            owner.handle_event(&mut session, Event::EndOfInput)
        } else {
            line(&mut owner, &mut session, command)
        };
        assert!(out.session.accepted, "{command}: {out:?}");
        assert!(token.is_canceled(), "{command}");
        assert_eq!(
            control.stop_reason(Instant::now()),
            Some(StopReason::Canceled)
        );
        let stale = owner.handle_event(
            &mut session,
            Event::Progress {
                ticket,
                bestmove: "d2d4".into(),
            },
        );
        assert_typed_error(&stale, ErrorCode::Stale);
        assert!(stale.session.protocol.is_empty());
        if command != "stop" {
            assert!(out.session.protocol.is_empty(), "{command}");
        }
    }
}

#[test]
fn finite_completion_is_consumed_once_and_closes_scope_cancellation() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let ticket = begin(&mut owner, &mut session, "go nodes 64");
    let token = owner.cancellation().unwrap().token();
    let out = owner.handle_event(
        &mut session,
        Event::Complete {
            ticket: ticket.clone(),
            completion: SearchCompletion::Completed {
                bestmove: Some("d2d4".into()),
            },
        },
    );
    assert!(out.session.accepted);
    assert!(out.errors.is_empty());
    assert_eq!(out.session.protocol, vec!["bestmove d2d4"]);
    assert!(token.is_canceled());
    assert!(owner.active_ticket().is_none());
    assert!(owner.active_binding().is_none());
    let duplicate = owner.handle_event(
        &mut session,
        Event::Complete {
            ticket,
            completion: SearchCompletion::Failed {
                code: "LateBackendFailure".into(),
                message: "late device callback".into(),
            },
        },
    );
    assert_typed_error(&duplicate, ErrorCode::Stale);
    assert!(duplicate.session.protocol.is_empty());
    assert!(
        duplicate
            .session
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "SearchFailed"
                && diagnostic.message.contains("LateBackendFailure"))
    );
}

#[test]
fn infinite_success_waits_for_stop_and_passive_old_timer_cannot_invent_a_failure() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let ticket = begin(&mut owner, &mut session, "go infinite");
    let hard = owner.active_binding().unwrap().budget().hard_deadline;
    let token = owner.cancellation().unwrap().token();
    let out = owner.handle_event(
        &mut session,
        Event::Complete {
            ticket: ticket.clone(),
            completion: SearchCompletion::Completed {
                bestmove: Some("d2d4".into()),
            },
        },
    );
    assert!(out.session.accepted);
    assert!(out.errors.is_empty());
    assert!(out.session.protocol.is_empty());
    assert_eq!(owner.active_ticket(), Some(&ticket));
    let timer = owner.handle_event_at(&mut session, Event::Deadline(ticket.clone()), hard);
    assert!(timer.session.accepted);
    assert!(timer.errors.is_empty());
    assert!(timer.session.protocol.is_empty());
    assert!(token.is_canceled());
    let duplicate = owner.handle_event(
        &mut session,
        Event::Progress {
            ticket,
            bestmove: "e2e4".into(),
        },
    );
    assert_typed_error(&duplicate, ErrorCode::Stale);
    assert_eq!(
        line(&mut owner, &mut session, "stop").session.protocol,
        vec!["bestmove d2d4"]
    );
    assert!(
        line(&mut owner, &mut session, "stop")
            .session
            .protocol
            .is_empty()
    );
}

#[test]
fn infinite_resource_wall_closes_work_and_holds_fallback_until_stop() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let ticket = begin(&mut owner, &mut session, "go infinite");
    let hard = owner.active_binding().unwrap().budget().hard_deadline;
    let token = owner.cancellation().unwrap().token();
    let out = owner.handle_event_at(&mut session, Event::Deadline(ticket.clone()), hard);
    assert_typed_error(&out, ErrorCode::Expired);
    assert!(out.session.protocol.is_empty());
    assert!(
        out.session
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "ResourceWallLimit")
    );
    assert!(token.is_canceled());
    assert_eq!(owner.active_ticket(), Some(&ticket));
    let late = owner.handle_event(
        &mut session,
        Event::Complete {
            ticket,
            completion: SearchCompletion::Completed {
                bestmove: Some("d2d4".into()),
            },
        },
    );
    assert_typed_error(&late, ErrorCode::Stale);
    assert_eq!(
        line(&mut owner, &mut session, "stop").session.protocol,
        vec!["bestmove e2e4"]
    );
}

#[test]
fn exact_terminal_finite_and_infinite_never_construct_evaluator_bindings() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    assert!(
        line(
            &mut owner,
            &mut session,
            &format!("position fen {MATE_FEN}")
        )
        .session
        .accepted
    );
    let finite = line(&mut owner, &mut session, "go movetime 0");
    assert_eq!(finite.session.protocol, vec!["bestmove 0000"]);
    assert!(finite.errors.is_empty());
    assert!(finite.session.effects.is_empty());
    assert!(owner.active_binding().is_none());
    let infinite = line(&mut owner, &mut session, "go infinite");
    assert!(infinite.session.protocol.is_empty());
    assert!(infinite.session.effects.is_empty());
    assert!(owner.active_ticket().is_some());
    assert!(owner.active_binding().is_none());
    assert!(owner.cancellation().is_none());
    let stopped = line(&mut owner, &mut session, "stop");
    assert_eq!(stopped.session.protocol, vec!["bestmove 0000"]);
    assert!(stopped.session.effects.is_empty());
    assert!(owner.active_ticket().is_none());
}

#[test]
fn invalid_time_and_resource_limits_preserve_typed_cause_with_legal_failure_output() {
    for (command, expected) in [
        ("go movetime 0", ErrorCode::Expired),
        ("go nodes 129", ErrorCode::ResourceExhausted),
    ] {
        let mut owner = owner(fixture_scope());
        let mut session = session();
        let out = line(&mut owner, &mut session, command);
        assert_typed_error(&out, expected);
        assert_eq!(out.session.protocol, vec!["bestmove e2e4"]);
        assert!(
            !out.session
                .effects
                .iter()
                .any(|effect| matches!(effect, Effect::Start { .. }))
        );
        assert!(
            out.session
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "SearchFailed")
        );
        assert!(owner.active_ticket().is_none());
    }
}

#[test]
fn side_adapter_failure_is_reported_without_dispatching_a_worker() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let failure = ContractError::new(
        ErrorCode::IdentityMismatch,
        Stage::Admission,
        "checked fixture snapshot is stale",
    );
    let out = owner.handle_line(&mut session, "go nodes 64", |_| Err(failure));
    assert_typed_error(&out, ErrorCode::IdentityMismatch);
    assert_eq!(out.errors, vec![failure]);
    assert_eq!(out.session.protocol, vec!["bestmove e2e4"]);
    assert!(
        !out.session
            .effects
            .iter()
            .any(|effect| matches!(effect, Effect::Start { .. }))
    );
    assert!(owner.active_ticket().is_none());
}

#[test]
fn independently_active_or_externally_replaced_sessions_cannot_bypass_owner() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let foreign = session.handle_line("go nodes 4");
    assert!(foreign.accepted);
    let original = owner.scope();
    let ticket = session.active_ticket().unwrap();
    let out = line(&mut owner, &mut session, "stop");
    assert_typed_error(&out, ErrorCode::Stale);
    assert!(out.session.effects.is_empty());
    assert_scope(owner.scope(), original);
    assert_eq!(session.active_ticket(), Some(ticket));
    session.handle_line("stop");
    let tracked = begin(&mut owner, &mut session, "go nodes 64");
    session.handle_line("position startpos moves e2e4");
    let out = owner.handle_event(
        &mut session,
        Event::Complete {
            ticket: tracked,
            completion: SearchCompletion::Completed {
                bestmove: Some("d2d4".into()),
            },
        },
    );
    assert_typed_error(&out, ErrorCode::Stale);
    assert!(out.session.protocol.is_empty());
}

#[test]
fn line_events_require_checked_rules_side_and_do_not_use_untimed_reducer() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let original = owner.scope();
    let out = owner.handle_event(&mut session, Event::Line("go nodes 64".into()));
    assert_typed_error(&out, ErrorCode::InvalidInput);
    assert!(out.session.effects.is_empty());
    assert_scope(owner.scope(), original);
    assert!(session.active_ticket().is_none());
}

#[test]
fn registry_updates_are_explicit_idle_only_and_preserve_game_root_counters() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    let previous = owner.scope();
    let mut changed = previous;
    changed.model.generation = SlotGeneration(4);
    changed.encoding.generation = SlotGeneration(8);
    changed.backend = Digest([42; 32]);
    owner.set_registry(&session, changed).unwrap();
    assert_scope(owner.scope(), changed);
    let wrong_game = AcceptanceScope {
        game: GameGeneration(99),
        ..changed
    };
    assert_eq!(
        owner.set_registry(&session, wrong_game).unwrap_err().code,
        ErrorCode::IdentityMismatch
    );
    begin(&mut owner, &mut session, "go nodes 64");
    assert_scope(
        owner.active_scope().unwrap(),
        AcceptanceScope {
            root: RootGeneration(32),
            ..changed
        },
    );
    let token = owner.cancellation().unwrap().token();
    assert_eq!(
        owner
            .set_registry(&session, owner.scope())
            .unwrap_err()
            .code,
        ErrorCode::InvalidInput
    );
    assert!(!token.is_canceled());
}

#[test]
fn constructor_rejects_clock_epoch_mismatch_and_future_origin() {
    let mismatch = ContractSessionOwner::new(
        fixture_scope(),
        EPOCH,
        InstantClock::new(ClockDomain(ProcessEpoch(EPOCH.0 + 1)), Instant::now()),
        settings(),
    )
    .err()
    .unwrap();
    assert_eq!(mismatch.code, ErrorCode::IdentityMismatch);
    assert_eq!(mismatch.stage, Stage::Admission);
    let future_origin = Instant::now().checked_add(Duration::from_secs(30)).unwrap();
    let future = ContractSessionOwner::new(
        fixture_scope(),
        EPOCH,
        InstantClock::new(ClockDomain(EPOCH), future_origin),
        settings(),
    )
    .err()
    .unwrap();
    assert_eq!(future.code, ErrorCode::InvalidInput);
    assert_eq!(future.stage, Stage::Admission);
}

#[test]
fn closed_session_commands_do_not_advance_any_scope_counter() {
    let mut owner = owner(fixture_scope());
    let mut session = session();
    assert!(line(&mut owner, &mut session, "quit").session.accepted);
    let before = owner.scope();
    for command in ["go nodes 2", "position startpos moves e2e4", "ucinewgame"] {
        let out = line(&mut owner, &mut session, command);
        assert!(!out.session.accepted);
        assert_scope(owner.scope(), before);
        assert!(out.session.effects.is_empty());
    }
}
