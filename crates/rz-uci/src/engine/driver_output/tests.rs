//! Publication-only Rules fixtures. No search engine, NN, worker or GPU runs.
use super::*;
use crate::engine::{OwnerRegistry, RulesUciPort};
use crate::search_driver::DriverWork;
use crate::{Effect, EngineIdentity, ParserLimits, Session};
use rz_contracts::rovent::ChessPly;
use rz_contracts::{
    AcceptanceScope, Digest, EncodingHandle, GameGeneration, ModelHandle, OwnerId, ProcessEpoch,
    RootGeneration, SlotGeneration,
};
use rz_position::PositionLimits;
use rz_search::pals::rovent_solved::MateProofStore;

const WHITE_MATE: &str = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1";
const BLACK_MATE: &str = "8/8/8/8/8/6k1/5q2/7K b - - 0 1";

struct StartFixture {
    _session: Session<RulesUciPort>,
    ticket: SearchTicket,
    root: Position,
}

fn start(position_command: &str) -> StartFixture {
    let owners = Arc::new(OwnerRegistry::default());
    let mut session = Session::new(
        RulesUciPort::new(owners, PositionLimits::default()),
        EngineIdentity {
            name: "publication-rules-fixture".into(),
            author: "fixture".into(),
        },
        vec![],
        ParserLimits::default(),
    )
    .unwrap();
    let positioned = session.handle_line(position_command);
    assert!(positioned.accepted, "{:?}", positioned.diagnostics);
    let started = session.handle_line("go nodes 1");
    assert!(started.accepted, "{:?}", started.diagnostics);
    let (ticket, root) = started
        .effects
        .into_iter()
        .find_map(|effect| match effect {
            Effect::Start {
                ticket, snapshot, ..
            } => Some((ticket, snapshot.rules_position().clone())),
            _ => None,
        })
        .expect("actual Session-issued Start ticket and Rules snapshot");
    StartFixture {
        _session: session,
        ticket,
        root,
    }
}

fn fen_start(fen: &str) -> StartFixture {
    start(&format!("position fen {fen}"))
}

fn authority() -> SearchAuthority {
    SearchAuthority {
        epoch: ProcessEpoch(23),
        game: GameGeneration(1),
        root: RootGeneration(2),
        implementation: Digest([7; 32]),
    }
}

fn scope() -> SessionScope {
    SessionScope::Search(authority())
}

/// Rejected scope metadata only; no model descriptor or neural receipt exists.
fn evaluation_scope() -> SessionScope {
    SessionScope::Evaluation(AcceptanceScope {
        game: authority().game,
        root: authority().root,
        model: ModelHandle {
            owner: OwnerId(1),
            slot: 0,
            generation: SlotGeneration(1),
            manifest: Digest([1; 32]),
        },
        encoding: EncodingHandle {
            owner: OwnerId(2),
            slot: 0,
            generation: SlotGeneration(1),
            manifest: Digest([2; 32]),
        },
        backend: Digest([3; 32]),
    })
}

fn mv(text: &str) -> BoardMove {
    BoardMove::from_uci(text).unwrap()
}

fn report(
    best_move: Option<BoardMove>,
    mate_strategy: Option<FrontierMateStrategy>,
) -> SearchSessionReport {
    SearchSessionReport {
        best_move,
        work: DriverWork::Cpu {
            nodes: 0,
            completed_depth: 0,
        },
        mate_strategy,
    }
}

fn proof(
    position: &Position,
    movement: BoardMove,
    root_moves: Option<&[BoardMove]>,
) -> FrontierMateStrategy {
    let mut store = MateProofStore::new(
        position,
        position.side_to_move(),
        ChessPly(1),
        root_moves,
        8,
    )
    .unwrap();
    let root = store.root();
    store.open_child(root, movement).unwrap();
    store
        .root_strategy()
        .expect("actual Rules mate-in-one proof")
}

fn assert_error<T>(result: Result<T, ContractError>, code: ErrorCode, stage: Stage) {
    match result {
        Err(error) => {
            assert_eq!(error.code, code);
            assert_eq!(error.stage, stage);
        }
        Ok(_) => panic!("expected typed publication rejection"),
    }
}

#[test]
fn both_colors_publish_actual_rules_mate_only_for_their_recorded_bestmove() {
    for (fen, text) in [(WHITE_MATE, "f7g7"), (BLACK_MATE, "f2g2")] {
        let fixture = fen_start(fen);
        let movement = mv(text);
        let strategy = proof(&fixture.root, movement, None);
        let envelope =
            DriverOutputEnvelope::new(fixture.ticket.clone(), authority(), &fixture.root, None)
                .unwrap();
        let recorded = envelope
            .record(report(Some(movement), Some(strategy)))
            .unwrap();
        let issued = envelope
            .mate_for(&recorded, scope(), text, true)
            .unwrap()
            .unwrap();
        assert_eq!(issued.best_move(), movement);
        assert_eq!(issued.conclusion().exact_uci_mate(), Some(1));
        assert!(issued.validates_root(
            &fixture.root.snapshot(),
            fixture.root.ordered_legal_moves().moves()
        ));
    }
}

#[test]
fn root_subset_must_already_be_legal_distinct_and_in_rules_order() {
    let fixture = start("position startpos");
    let legal = fixture.root.ordered_legal_moves().moves().to_vec();
    for invalid in [
        vec![],
        vec![legal[0], legal[0]],
        vec![mv("e2e5")],
        vec![legal[1], legal[0]],
    ] {
        assert_error(
            DriverOutputEnvelope::new(
                fixture.ticket.clone(),
                authority(),
                &fixture.root,
                Some(&invalid),
            ),
            ErrorCode::InvalidInput,
            Stage::Admission,
        );
    }
    DriverOutputEnvelope::new(
        fixture.ticket.clone(),
        authority(),
        &fixture.root,
        Some(&legal[..2]),
    )
    .unwrap();
}

#[test]
fn report_lookup_requires_the_exact_start_ticket_and_every_search_scope_field() {
    let fixture = start("position startpos");
    let other_start = start("position startpos");
    let envelope =
        DriverOutputEnvelope::new(fixture.ticket.clone(), authority(), &fixture.root, None)
            .unwrap();
    assert!(
        envelope
            .report_for(&fixture.ticket, scope())
            .unwrap()
            .is_none()
    );
    let movement = fixture.root.ordered_legal_moves().moves()[0];
    let recorded = envelope.record(report(Some(movement), None)).unwrap();
    let found = envelope
        .report_for(&fixture.ticket, scope())
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&recorded, &found));
    assert_error(
        envelope.report_for(&other_start.ticket, scope()),
        ErrorCode::Stale,
        Stage::Output,
    );
    for field in 0..4 {
        let mut changed = authority();
        match field {
            0 => changed.epoch = ProcessEpoch(changed.epoch.0 + 1),
            1 => changed.game = GameGeneration(changed.game.0 + 1),
            2 => changed.root = RootGeneration(changed.root.0 + 1),
            _ => changed.implementation = Digest([8; 32]),
        }
        assert_error(
            envelope.report_for(&fixture.ticket, SessionScope::Search(changed)),
            ErrorCode::Stale,
            Stage::Output,
        );
    }
    assert_error(
        envelope.report_for(&fixture.ticket, evaluation_scope()),
        ErrorCode::Stale,
        Stage::Output,
    );
}

#[test]
fn rejected_illegal_or_excluded_move_does_not_fill_slot_and_duplicate_cannot_replace_it() {
    let fixture = start("position startpos");
    let legal = fixture.root.ordered_legal_moves().moves().to_vec();
    let envelope = DriverOutputEnvelope::new(
        fixture.ticket.clone(),
        authority(),
        &fixture.root,
        Some(&legal[..1]),
    )
    .unwrap();
    for rejected in [mv("e2e5"), legal[1]] {
        assert_error(
            envelope.record(report(Some(rejected), None)),
            ErrorCode::InvalidInput,
            Stage::Output,
        );
        assert!(
            envelope
                .report_for(&fixture.ticket, scope())
                .unwrap()
                .is_none()
        );
    }
    let recorded = envelope.record(report(Some(legal[0]), None)).unwrap();
    assert_error(
        envelope.record(report(None, None)),
        ErrorCode::Stale,
        Stage::Output,
    );
    let found = envelope
        .report_for(&fixture.ticket, scope())
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&recorded, &found));
    assert_eq!(found.best_move, Some(legal[0]));
}

#[test]
fn rejection_retains_the_actual_final_report_and_nonzero_work_without_filling_the_slot() {
    let fixture = start("position startpos");
    let envelope =
        DriverOutputEnvelope::new(fixture.ticket.clone(), authority(), &fixture.root, None)
            .unwrap();
    let mut original = report(Some(mv("e2e5")), None);
    original.work = DriverWork::Cpu {
        nodes: 739,
        completed_depth: 3,
    };
    let (rejected, error) = envelope.record_retaining(original).unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(rejected.best_move, Some(mv("e2e5")));
    assert_eq!(
        rejected.work,
        DriverWork::Cpu {
            nodes: 739,
            completed_depth: 3,
        }
    );
    assert!(
        envelope
            .report_for(&fixture.ticket, scope())
            .unwrap()
            .is_none()
    );
}

#[test]
fn proof_for_another_final_move_or_no_final_move_cannot_consume_the_slot() {
    let fixture = fen_start(WHITE_MATE);
    let movement = mv("f7g7");
    let strategy = proof(&fixture.root, movement, None);
    let other = fixture
        .root
        .ordered_legal_moves()
        .moves()
        .iter()
        .copied()
        .find(|candidate| *candidate != movement)
        .unwrap();
    let envelope =
        DriverOutputEnvelope::new(fixture.ticket.clone(), authority(), &fixture.root, None)
            .unwrap();
    for best in [Some(other), None] {
        assert_error(
            envelope.record(report(best, Some(strategy.clone()))),
            ErrorCode::IdentityMismatch,
            Stage::Output,
        );
        assert!(
            envelope
                .report_for(&fixture.ticket, scope())
                .unwrap()
                .is_none()
        );
    }
    envelope
        .record(report(Some(movement), Some(strategy)))
        .unwrap();
}

#[test]
fn same_move_proof_from_another_root_or_same_fen_with_different_history_is_rejected() {
    let fixture = fen_start(WHITE_MATE);
    let foreign = Position::from_fen("7k/5Q2/5K2/8/8/8/8/8 w - - 0 1").unwrap();
    let movement = mv("f7g7");
    let envelope =
        DriverOutputEnvelope::new(fixture.ticket.clone(), authority(), &fixture.root, None)
            .unwrap();
    assert_error(
        envelope.record(report(
            Some(movement),
            Some(proof(&foreign, movement, None)),
        )),
        ErrorCode::IdentityMismatch,
        Stage::Output,
    );

    let history = start(&format!(
        "position fen {WHITE_MATE} moves f7f6 h8g8 f6f7 g8h8"
    ));
    let imported = Position::from_fen(&history.root.to_fen()).unwrap();
    assert!(!history.root.snapshot().same_state(&imported.snapshot()));
    let envelope =
        DriverOutputEnvelope::new(history.ticket.clone(), authority(), &history.root, None)
            .unwrap();
    assert_error(
        envelope.record(report(
            Some(movement),
            Some(proof(&imported, movement, None)),
        )),
        ErrorCode::IdentityMismatch,
        Stage::Output,
    );
    envelope
        .record(report(
            Some(movement),
            Some(proof(&history.root, movement, None)),
        ))
        .unwrap();
}

#[test]
fn full_and_restricted_mate_domains_are_not_interchangeable_in_either_direction() {
    let fixture = fen_start(WHITE_MATE);
    let movement = mv("f7g7");
    let restriction = [movement];
    let full = proof(&fixture.root, movement, None);
    let restricted = proof(&fixture.root, movement, Some(&restriction));
    for (domain, rejected, accepted) in [
        (None, restricted.clone(), full.clone()),
        (Some(restriction.as_slice()), full, restricted),
    ] {
        let envelope =
            DriverOutputEnvelope::new(fixture.ticket.clone(), authority(), &fixture.root, domain)
                .unwrap();
        assert_error(
            envelope.record(report(Some(movement), Some(rejected))),
            ErrorCode::IdentityMismatch,
            Stage::Output,
        );
        envelope
            .record(report(Some(movement), Some(accepted)))
            .unwrap();
    }
}

#[test]
fn proof_publication_requires_slot_arc_identity_scope_and_exact_report_move_text() {
    let fixture = fen_start(WHITE_MATE);
    let movement = mv("f7g7");
    let envelope =
        DriverOutputEnvelope::new(fixture.ticket.clone(), authority(), &fixture.root, None)
            .unwrap();
    let recorded = envelope
        .record(report(
            Some(movement),
            Some(proof(&fixture.root, movement, None)),
        ))
        .unwrap();
    let clone = Arc::clone(&recorded);
    assert!(
        envelope
            .mate_for(&clone, scope(), "f7g7", true)
            .unwrap()
            .is_some()
    );
    let same_contents_other_owner = Arc::new(recorded.as_ref().clone());
    assert_error(
        envelope.mate_for(&same_contents_other_owner, scope(), "f7g7", true),
        ErrorCode::Stale,
        Stage::Output,
    );
    assert_error(
        envelope.mate_for(&recorded, evaluation_scope(), "f7g7", true),
        ErrorCode::Stale,
        Stage::Output,
    );
    assert_error(
        envelope.mate_for(&recorded, scope(), "f7f6", true),
        ErrorCode::IdentityMismatch,
        Stage::Output,
    );
}

#[test]
fn fallback_and_none_bestmove_never_inherit_an_older_proof_or_another_start_slot() {
    let fixture = fen_start(WHITE_MATE);
    let movement = mv("f7g7");
    let old = DriverOutputEnvelope::new(fixture.ticket.clone(), authority(), &fixture.root, None)
        .unwrap();
    let old_report = old
        .record(report(
            Some(movement),
            Some(proof(&fixture.root, movement, None)),
        ))
        .unwrap();
    let next_start = fen_start(WHITE_MATE);
    let next = DriverOutputEnvelope::new(
        next_start.ticket.clone(),
        authority(),
        &next_start.root,
        None,
    )
    .unwrap();
    assert!(
        next.report_for(&next_start.ticket, scope())
            .unwrap()
            .is_none()
    );
    assert!(
        next.mate_for(&old_report, evaluation_scope(), "0000", false)
            .unwrap()
            .is_none()
    );
    assert_error(
        next.mate_for(&old_report, scope(), "f7g7", true),
        ErrorCode::Stale,
        Stage::Output,
    );
    let empty = next.record(report(None, None)).unwrap();
    assert!(
        next.mate_for(&empty, scope(), "0000", true)
            .unwrap()
            .is_none()
    );
    assert!(
        next.mate_for(&empty, scope(), "f7g7", true)
            .unwrap()
            .is_none()
    );
}

#[test]
fn poisoned_publication_slot_returns_typed_failure_and_fallback_remains_proof_free() {
    let fixture = start("position startpos");
    let envelope =
        DriverOutputEnvelope::new(fixture.ticket.clone(), authority(), &fixture.root, None)
            .unwrap();
    let unrelated = Arc::new(report(None, None));
    let poison = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _held = envelope.report.lock().unwrap();
        panic!("fixture: poison only this publication slot");
    }));
    assert!(poison.is_err());
    assert_error(
        envelope.record(report(None, None)),
        ErrorCode::BackendFailure,
        Stage::Output,
    );
    assert_error(
        envelope.report_for(&fixture.ticket, scope()),
        ErrorCode::BackendFailure,
        Stage::Output,
    );
    assert_error(
        envelope.mate_for(&unrelated, scope(), "0000", true),
        ErrorCode::BackendFailure,
        Stage::Output,
    );
    assert!(
        envelope
            .mate_for(&unrelated, evaluation_scope(), "0000", false)
            .unwrap()
            .is_none()
    );
}
