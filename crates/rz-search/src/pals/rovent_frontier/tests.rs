//! Authored contract evidence, not actual model/GPU or speed acceptance.
use super::*;
use crate::contract_time::{ContractClock, InstantClock};
use crate::cpu::{CpuError, CpuWork};
use crate::cpu_worker::CpuCreditLedger;
use crate::pals::store::{SituationId, StoreLimits};
use crate::pals::value::{MODEL_WDL_VALUE_SEMANTICS, ModelValueIdentity};
use rz_contracts::pals_adaptive::{CpuJobId, CpuWorkerId};
use rz_contracts::{
    ExecutionId, GameGeneration, OwnerId, PrecisionProfile, ProcessEpoch, RootGeneration,
};
use rz_position::contracts::ContractPosition;
use std::time::{Duration, Instant};

fn mv(text: &str) -> BoardMove {
    BoardMove::from_uci(text).unwrap()
}

#[test]
fn unrelated_capacity_stop_does_not_certify_public_record_exhaustion() {
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), None, false);
    assert!(!fixture.frontier.public_record_capacity_refused());
    assert!(!fixture.frontier.public_review_index_capacity_refused());
    // An actual controller can stop for another bounded resource. Its equal
    // public Capacity snapshot must not authorize the record-specific path.
    fixture.frontier.interrupt(FrontierStop::Capacity);
    assert_eq!(
        fixture.frontier.snapshot().goal,
        FrontierGoalStatus::Stopped(FrontierStop::Capacity),
    );
    assert!(!fixture.frontier.public_record_capacity_refused());
    assert!(!fixture.frontier.public_review_index_capacity_refused());
}

#[test]
fn sealing_an_empty_policy_does_not_complete_time_only_or_depth_work() {
    for goal in [
        SearchGoal::default(),
        SearchGoal {
            depth_plies: NonZeroU16::new(3),
            mate_moves: None,
        },
    ] {
        let mut fixture = Fixture::new(Position::startpos(), goal, None, false);
        fixture
            .frontier
            .seal_selected_policy(fixture.clock.domain(), fixture.clock.now().unwrap())
            .unwrap();
        let snapshot = fixture.frontier.snapshot();
        assert!(snapshot.selection_sealed);
        assert_eq!(snapshot.registered_frames, 0);
        assert_eq!(snapshot.goal, FrontierGoalStatus::Pending);
        assert_eq!(snapshot.horizon.completed_depth, None);
    }
}

#[test]
fn paid_cpu_pv_adopts_root_and_nonroot_suffix_without_another_frontier_visit() {
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), None, false);
    let moves = [mv("e2e4"), mv("e7e5")];
    let path = fixture.observe(&moves).unwrap();
    let checked = fixture
        .frontier
        .checked_line(path, &fixture.stores)
        .unwrap();
    let root = Position::from_snapshot(checked.root_snapshot(), Default::default()).unwrap();
    let before = fixture.frontier.snapshot().work;
    let line = fixture
        .stores
        .hot_append_checked_cpu_pv(&root, &moves, &checked)
        .unwrap();
    assert_eq!(fixture.stores.lines.moves(line).unwrap(), moves);
    let child = Position::from_snapshot(checked.steps()[0].snapshot(), Default::default()).unwrap();
    fixture
        .stores
        .hot_insert_situation(child.snapshot())
        .unwrap();
    let suffix = fixture
        .stores
        .hot_append_checked_cpu_pv(&child, &moves[1..], &checked)
        .unwrap();
    assert_eq!(fixture.stores.lines.moves(suffix).unwrap(), moves[1..]);
    let after = fixture.frontier.snapshot().work;
    assert_eq!(after.checked_transitions, before.checked_transitions);
    assert_eq!(after.total_node_visits(), before.total_node_visits());
    let copied_fen = Position::from_fen(&root.to_fen()).unwrap();
    assert!(!copied_fen.snapshot().same_state(checked.root_snapshot()));
    assert!(
        fixture
            .stores
            .hot_append_checked_cpu_pv(&copied_fen, &moves, &checked)
            .is_err()
    );
    assert!(
        fixture
            .stores
            .hot_append_checked_cpu_pv(&root, &[mv("d2d4")], &checked)
            .is_err()
    );
    let mut foreign = Fixture::new(Position::startpos(), SearchGoal::default(), None, false);
    assert!(!checked.matches_store(&foreign.stores));
    assert!(
        foreign
            .stores
            .hot_append_checked_cpu_pv(&root, &moves, &checked)
            .is_err()
    );
}

#[test]
fn paid_source_tracks_replaced_pin_without_adding_or_extending_its_lease() {
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), None, false);
    let path = fixture.observe(&[mv("e2e4")]).unwrap();
    let checked = fixture
        .frontier
        .checked_line(path, &fixture.stores)
        .unwrap();
    let old_pin = Arc::downgrade(&fixture.frontier.pin);
    let leases = fixture.stores.review_pin_count().unwrap();
    fixture.observe(&[mv("d2d4")]).unwrap();
    assert!(old_pin.upgrade().is_none());
    assert!(checked.matches_store(&fixture.stores));
    assert_eq!(fixture.stores.review_pin_count().unwrap(), leases);
    let Fixture {
        mut stores,
        frontier,
        ..
    } = fixture;
    let root = Position::from_snapshot(checked.root_snapshot(), Default::default()).unwrap();
    drop(frontier);
    assert_eq!(stores.review_pin_count().unwrap(), 0);
    assert!(!checked.matches_store(&stores));
    assert!(
        stores
            .hot_append_checked_cpu_pv(&root, &[mv("e2e4")], &checked)
            .is_err()
    );
}

#[test]
fn paid_store_source_rejects_revocation_and_generation_but_not_spent_budget_alone() {
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), Some(3), false);
    let moves = [mv("e2e4"), mv("e7e5")];
    let path = fixture.observe(&moves).unwrap();
    let checked = fixture
        .frontier
        .checked_line(path, &fixture.stores)
        .unwrap();
    assert_eq!(
        fixture.frontier.snapshot().work.total_node_visits(),
        Some(3)
    );
    assert_eq!(
        fixture
            .frontier
            .admit_role_input(fixture.clock.domain(), fixture.clock.now().unwrap()),
        Err(FrontierError::NodeBudget)
    );
    assert!(checked.matches_store(&fixture.stores));
    let root = Position::from_snapshot(checked.root_snapshot(), Default::default()).unwrap();
    fixture
        .stores
        .hot_append_checked_cpu_pv(&root, &moves, &checked)
        .unwrap();
    fixture.frontier.cancel();
    // The old matcher reports historical owner/path identity. Store-only
    // admission separately rejects that same source after cancellation.
    assert!(checked.matches(&fixture.frontier, &fixture.stores));
    assert!(!checked.matches_store(&fixture.stores));
    assert!(
        fixture
            .stores
            .hot_append_checked_cpu_pv(&root, &moves, &checked)
            .is_err()
    );
    let mut newer = Fixture::new(Position::startpos(), SearchGoal::default(), None, false);
    let path = newer.observe(&[mv("e2e4")]).unwrap();
    let checked = newer.frontier.checked_line(path, &newer.stores).unwrap();
    newer
        .stores
        .hot_focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    assert!(!checked.matches_store(&newer.stores));
}

#[test]
fn ordinary_constructor_still_requires_own_ledger_for_pcr_and_no_r_source_for_pc() {
    let fixture = Fixture::new(Position::startpos(), SearchGoal::default(), None, false);
    let copied = |mode, native_ledger| FrontierRegistration {
        authority: fixture.frontier.registration.authority,
        mode,
        original_deadline: fixture.frontier.registration.original_deadline,
        root_budget: fixture.frontier.registration.root_budget,
        goal: fixture.frontier.registration.goal,
        nodes: fixture.frontier.registration.nodes,
        selective_policy: fixture.frontier.registration.selective_policy,
        root_rules: Arc::clone(&fixture.frontier.registration.root_rules),
        root_moves: Arc::clone(&fixture.frontier.registration.root_moves),
        root_state: fixture.frontier.registration.root_state,
        model_namespace: Arc::clone(&fixture.frontier.registration.model_namespace),
        native_ledger,
    };
    let ledger =
        CpuCreditLedger::new(fixture.frontier.registration.authority, 100, fixture.clock).unwrap();
    for registration in [
        copied(ExecutionMode::CpuPcr, None),
        copied(ExecutionMode::CpuPc, Some(ledger.identity())),
    ] {
        assert!(matches!(
            RoventFrontier::new(
                registration,
                fixture.frontier.limits.clone(),
                &fixture.stores,
                fixture.clock.domain(),
                fixture.clock.now().unwrap(),
            ),
            Err(FrontierError::ForeignRoot)
        ));
    }
    assert!(fixture.frontier.external_source().is_none());
    assert!(fixture.frontier.snapshot().external_work.is_none());
}

struct Fixture {
    stores: PalsStores,
    clock: InstantClock,
    frontier: RoventFrontier,
    root_line: LineId,
    root_situation: SituationId,
}

impl Fixture {
    fn new(position: Position, goal: SearchGoal, nodes: Option<u64>, native: bool) -> Self {
        Self::with_maximum_plies(position, goal, nodes, native, 32)
    }

    fn with_maximum_plies(
        position: Position,
        goal: SearchGoal,
        nodes: Option<u64>,
        native: bool,
        maximum_plies: u16,
    ) -> Self {
        let mut stores = PalsStores::new(StoreLimits {
            states: 128,
            situations: 128,
            ..Default::default()
        });
        let root_situation = stores.hot_focus_actual_moves(position.snapshot()).unwrap();
        let root_state = stores.situations.get(root_situation).unwrap().state;
        let root_line = stores.lines.root(root_state).unwrap();
        let clock = InstantClock::new(ClockDomain(ProcessEpoch(71)), Instant::now());
        let authority = SearchAuthority {
            epoch: clock.domain().0,
            game: GameGeneration(1),
            root: RootGeneration(stores.generation()),
            implementation: Digest([7; 32]),
        };
        let rules = Arc::new(
            ContractPosition::new(OwnerId(71), position)
                .export()
                .unwrap(),
        );
        let moves = Arc::new(RootMoveDomain::new(&rules, None).unwrap());
        let namespace = Arc::new(
            AdaptiveModelNamespace::new(
                ModelValueIdentity {
                    semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
                    model: "authored-frontier-fixture".into(),
                    encoding: "authored-selected-review".into(),
                    precision: "fp32".into(),
                    model_epoch: [8; 32],
                },
                Digest([5; 32]),
                Digest([6; 32]),
                PrecisionProfile::Fp32,
                1,
            )
            .unwrap(),
        );
        let ledger = native.then(|| CpuCreditLedger::new(authority, 100, clock).unwrap());
        let frontier = RoventFrontier::new(
            FrontierRegistration {
                authority,
                mode: if native {
                    ExecutionMode::CpuPcr
                } else {
                    ExecutionMode::CpuPc
                },
                original_deadline: clock
                    .deadline_at(clock.origin() + Duration::from_secs(30))
                    .unwrap(),
                root_budget: Digest([9; 32]),
                goal,
                nodes,
                selective_policy: Digest([10; 32]),
                root_rules: rules,
                root_moves: moves,
                root_state,
                model_namespace: namespace,
                native_ledger: ledger.as_ref().map(CpuCreditLedger::identity),
            },
            FrontierLimits {
                maximum_goal_plies: NonZeroU16::new(maximum_plies).unwrap(),
                maximum_frames: 64,
                maximum_paths: 128,
                maximum_mate_nodes: 128,
                maximum_native_receipts: 16,
            },
            &stores,
            clock.domain(),
            clock.now().unwrap(),
        )
        .unwrap();
        Self {
            stores,
            clock,
            frontier,
            root_line,
            root_situation,
        }
    }

    fn line(&mut self, moves: &[BoardMove]) -> LineId {
        self.stores.hot_append_line(self.root_line, moves).unwrap()
    }

    fn observe(&mut self, moves: &[BoardMove]) -> Result<FrontierPath, FrontierError> {
        let line = self.line(moves);
        self.frontier.observe_stored_line(
            &self.stores,
            line,
            self.clock.domain(),
            self.clock.now().unwrap(),
        )
    }
}

#[test]
fn passive_mate_uses_only_paid_rules_edges_and_never_issues_missing_questions() {
    for (fen, movement) in [
        ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", "f7g7"),
        ("8/8/8/8/8/6k1/5q2/7K b - - 0 1", "f2g2"),
    ] {
        let mut fixture = Fixture::new(
            Position::from_fen(fen).unwrap(),
            SearchGoal::default(),
            None,
            false,
        );
        fixture.observe(&[mv(movement)]).unwrap();
        let snapshot = fixture.frontier.snapshot();
        assert_eq!(snapshot.work.checked_transitions, 1);
        assert_eq!(snapshot.work.native_nodes, 0);
        assert_eq!(snapshot.work.total_node_visits(), Some(2));
        let proof = snapshot.mate[0].unwrap();
        assert_eq!(proof.exact_uci_mate(), Some(1));
        let strategy = snapshot.mate_strategy.unwrap();
        assert_eq!(strategy.best_move(), mv(movement));
        assert_eq!(strategy.conclusion(), proof);
        assert!(strategy.validates_root(
            fixture.frontier.registration.root_rules.rules().snapshot(),
            fixture.frontier.root_moves().allowed(),
        ));
        assert_eq!(snapshot.goal, FrontierGoalStatus::MateSatisfied);
        assert!(
            fixture
                .frontier
                .missing_mate_questions(64, fixture.clock.domain(), fixture.clock.now().unwrap())
                .unwrap()
                .is_empty()
        );
        fixture.observe(&[mv(movement)]).unwrap();
        assert_eq!(fixture.frontier.snapshot().work.checked_transitions, 1);
    }
}

#[test]
fn passive_cooperative_terminal_does_not_close_unvisited_defenses() {
    let mut root = Position::from_fen("6k1/p7/5K2/8/8/8/P7/5Q2 w - - 0 1").unwrap();
    for movement in ["a2a3", "a7a6", "f1g1"] {
        root.make_uci(movement).unwrap();
    }
    let mut fixture = Fixture::new(root, SearchGoal::default(), None, false);
    fixture.observe(&[mv("g8h8"), mv("g1g7")]).unwrap();
    let snapshot = fixture.frontier.snapshot();
    assert_eq!(snapshot.work.checked_transitions, 2);
    assert_eq!(snapshot.work.total_node_visits(), Some(3));
    assert_eq!(
        snapshot.mate[1].unwrap().conclusion,
        MateConclusion::Unknown
    );
    assert_eq!(snapshot.goal, FrontierGoalStatus::Pending);
    assert!(
        fixture
            .frontier
            .missing_mate_questions(64, fixture.clock.domain(), fixture.clock.now().unwrap())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn passive_nonterminal_cutoff_stays_unknown_and_one_edge_cannot_exceed_root_cap() {
    let mut fixture =
        Fixture::with_maximum_plies(Position::startpos(), SearchGoal::default(), None, false, 2);
    let checked = fixture
        .frontier
        .observe_rules_prefix(
            &[mv("e2e4"), mv("e7e5")],
            &fixture.stores,
            fixture.clock.domain(),
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    for lane in fixture.frontier.mate.iter().flatten() {
        assert_eq!(
            lane.proof.root_conclusion().conclusion,
            MateConclusion::Unknown
        );
        let node = lane
            .paths
            .iter()
            .find_map(|(node, path)| (*path == checked.path()).then_some(*node))
            .unwrap();
        assert_eq!(lane.proof.remaining(node).unwrap(), ChessPly(1));
        assert_eq!(
            lane.proof.conclusion(node).unwrap(),
            MateConclusion::Unknown
        );
    }
    assert_eq!(
        fixture.frontier.advance_path(
            checked.path(),
            mv("g1f3"),
            fixture.clock.domain(),
            fixture.clock.now().unwrap(),
        ),
        Err(FrontierError::DepthHorizon)
    );
    assert_eq!(fixture.frontier.snapshot().work.checked_transitions, 2);
    assert_eq!(
        fixture.frontier.snapshot().horizon.maximum_examined_ply,
        ChessPly(2)
    );
}

#[test]
fn readonly_prefix_lookup_requires_original_paid_path_without_creating_work() {
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), None, false);
    assert!(
        fixture
            .frontier
            .existing_rules_prefix(&[mv("e2e4")], &fixture.stores)
            .unwrap()
            .is_none()
    );
    assert_eq!(fixture.frontier.snapshot().work.checked_transitions, 0);
    fixture.observe(&[mv("e2e4")]).unwrap();
    let checked = fixture
        .frontier
        .existing_rules_prefix(&[mv("e2e4")], &fixture.stores)
        .unwrap()
        .unwrap();
    assert!(checked.matches(&fixture.frontier, &fixture.stores));
    assert!(
        fixture
            .frontier
            .existing_rules_prefix(&[mv("e2e4"), mv("e7e5")], &fixture.stores)
            .unwrap()
            .is_none()
    );
    let other = Fixture::new(Position::startpos(), SearchGoal::default(), None, false);
    assert!(matches!(
        fixture
            .frontier
            .existing_rules_prefix(&[mv("e2e4")], &other.stores),
        Err(FrontierError::StoreUnavailable)
    ));
    assert_eq!(fixture.frontier.snapshot().work.checked_transitions, 1);
}

#[test]
fn checked_snapshot_reducer_rejects_foreign_root_and_wrong_step_without_advancing_rules() {
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), None, false);
    fixture.observe(&[mv("e2e4")]).unwrap();
    let checked = fixture
        .frontier
        .existing_rules_prefix(&[mv("e2e4")], &fixture.stores)
        .unwrap()
        .unwrap();
    let mut foreign_position = Position::startpos();
    foreign_position.make_uci("d2d4").unwrap();
    let mut foreign =
        MateProofStore::new(&foreign_position, Color::White, ChessPly(3), None, 16).unwrap();
    let foreign_root = foreign.root();
    assert_eq!(
        foreign.observe_checked_child(foreign_root, &checked, 0),
        Err(MateProofError::ForeignHandle)
    );
    assert_eq!(foreign.len(), 1);
    let root = Position::startpos();
    let mut proof = MateProofStore::new(&root, Color::White, ChessPly(3), None, 16).unwrap();
    let proof_root = proof.root();
    assert_eq!(
        proof.observe_checked_child(proof_root, &checked, 1),
        Err(MateProofError::ForeignHandle)
    );
    let child = proof
        .observe_checked_child(proof_root, &checked, 0)
        .unwrap();
    assert!(
        proof
            .snapshot(child)
            .unwrap()
            .same_state(checked.endpoint_snapshot())
    );
    assert_eq!(
        proof
            .observe_checked_child(proof_root, &checked, 0)
            .unwrap(),
        child
    );
    assert_eq!(proof.len(), 2);
    assert_eq!(fixture.frontier.snapshot().work.checked_transitions, 1);
}

#[test]
fn zero_one_nodes_and_cached_edges_preserve_the_registered_unit() {
    for nodes in [0, 1] {
        let mut fixture = Fixture::new(
            Position::startpos(),
            SearchGoal::default(),
            Some(nodes),
            false,
        );
        assert!(matches!(
            fixture.frontier.observe_root_child(
                mv("e2e4"),
                fixture.clock.domain(),
                fixture.clock.now().unwrap()
            ),
            Err(FrontierError::NodeBudget)
        ));
        let snapshot = fixture.frontier.snapshot();
        assert_eq!(snapshot.work.total_node_visits(), Some(nodes));
        assert_eq!(snapshot.work.checked_transitions, 0);
        assert_eq!(
            snapshot.goal,
            FrontierGoalStatus::Stopped(FrontierStop::WorkBudget)
        );
    }
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), Some(10), false);
    let root_child = fixture
        .frontier
        .observe_root_child(
            mv("e2e4"),
            fixture.clock.domain(),
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    assert_eq!(fixture.frontier.path_ply(root_child).unwrap(), ChessPly(1));
    let first = fixture.observe(&[mv("e2e4"), mv("e7e5")]).unwrap();
    let second = fixture.observe(&[mv("e2e4"), mv("e7e5")]).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        fixture.frontier.snapshot().work.total_node_visits(),
        Some(3)
    );
    assert_eq!(fixture.frontier.remaining_depth(first).unwrap(), None);
    assert_eq!(fixture.frontier.snapshot().horizon.declared_obligations, 0);
}

#[test]
fn checked_line_keeps_paid_snapshot_after_budget_closes_and_cannot_change_owner() {
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), Some(2), false);
    let path = fixture
        .frontier
        .observe_root_child(
            mv("e2e4"),
            fixture.clock.domain(),
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    let line = fixture
        .frontier
        .checked_line(path, &fixture.stores)
        .unwrap();
    assert!(line.matches(&fixture.frontier, &fixture.stores));
    assert_eq!(line.path(), path);
    assert_eq!(line.prefix(), [mv("e2e4")]);
    assert_eq!(line.steps().len(), 1);
    assert_eq!(line.steps()[0].movement(), mv("e2e4"));
    assert!(
        line.root_snapshot()
            .same_state(fixture.frontier.registration.root_rules.rules().snapshot())
    );
    assert!(
        line.steps()[0]
            .snapshot()
            .same_state(line.endpoint_snapshot())
    );
    assert_eq!(
        fixture
            .frontier
            .admit_role_input(fixture.clock.domain(), fixture.clock.now().unwrap()),
        Err(FrontierError::NodeBudget)
    );
    assert!(line.matches(&fixture.frontier, &fixture.stores));
    assert_eq!(
        fixture.frontier.snapshot().work.total_node_visits(),
        Some(2)
    );
    let other = Fixture::new(Position::startpos(), SearchGoal::default(), Some(2), false);
    assert!(!line.matches(&other.frontier, &other.stores));
    assert!(matches!(
        other.frontier.checked_line(path, &other.stores),
        Err(FrontierError::ForeignHandle)
    ));
}

#[test]
fn original_clock_bridge_keeps_constructor_entry_across_owner_move() {
    let fixture = Fixture::new(Position::startpos(), SearchGoal::default(), None, false);
    let entry = fixture.frontier.original_entry_instant;
    let original_tick = fixture.frontier.original_entry_tick;
    let before = fixture.frontier.observed_original_tick().unwrap();
    assert_eq!(before.0, fixture.clock.domain());
    assert!(before.1.0 >= original_tick.0);
    let moved = Box::new(fixture.frontier);
    assert_eq!(moved.original_entry_instant, entry);
    let after = moved.observed_original_tick().unwrap();
    assert_eq!(before.0, after.0);
    assert!(after.1.0 >= before.1.0);
    assert_eq!(moved.snapshot().work.total_node_visits(), Some(1));
}

#[test]
fn missing_child_admission_charges_before_returning_its_unforgeable_snapshot_once() {
    let mut fixture = Fixture::new(
        Position::startpos(),
        SearchGoal {
            depth_plies: NonZeroU16::new(3),
            mate_moves: NonZeroU16::new(2),
        },
        Some(20),
        false,
    );
    let question = fixture
        .frontier
        .missing_mate_questions(1, fixture.clock.domain(), fixture.clock.now().unwrap())
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(
        fixture.frontier.snapshot().work.total_node_visits(),
        Some(1)
    );
    let line = fixture
        .frontier
        .observe_missing_child(
            &question,
            fixture.clock.domain(),
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    assert!(line.matches(&fixture.frontier, &fixture.stores));
    assert_eq!(line.prefix(), [question.movement()]);
    // One actual horizon transition and the two independently checked mate
    // reducer transitions; no model call or raw value proves either lane.
    assert_eq!(fixture.frontier.snapshot().work.checked_transitions, 3);
    assert!(matches!(
        fixture.frontier.observe_missing_child(
            &question,
            fixture.clock.domain(),
            fixture.clock.now().unwrap()
        ),
        Err(FrontierError::ForeignHandle)
    ));
    assert_eq!(
        fixture.frontier.snapshot().work.total_node_visits(),
        Some(4)
    );
}

#[test]
fn unexamined_mate_defenses_are_returned_as_questions_and_depth_does_not_shrink_mate() {
    let mut root = Position::from_fen("6k1/p7/5K2/8/8/8/P7/5Q2 w - - 0 1").unwrap();
    for movement in ["a2a3", "a7a6", "f1g1"] {
        root.make_uci(movement).unwrap();
    }
    let mut fixture = Fixture::new(
        root,
        SearchGoal {
            depth_plies: NonZeroU16::new(1),
            mate_moves: NonZeroU16::new(2),
        },
        None,
        false,
    );
    fixture.observe(&[mv("g8h8")]).unwrap();
    let questions = fixture
        .frontier
        .missing_mate_questions(64, fixture.clock.domain(), fixture.clock.now().unwrap())
        .unwrap();
    assert!(
        questions
            .iter()
            .all(|q| fixture.frontier.path_ply(q.anchor()).unwrap() == ChessPly(0))
    );
    assert!(
        questions
            .iter()
            .any(|q| q.kind() == MateQuestionKind::RequiredDefense)
    );
    for lane in fixture.frontier.mate.iter().flatten() {
        assert_eq!(
            lane.proof.root_conclusion().conclusion,
            MateConclusion::Unknown
        );
    }
    assert_eq!(
        fixture.frontier.observe_prefix(
            &[mv("g8h8"), mv("g1g7")],
            fixture.clock.domain(),
            fixture.clock.now().unwrap()
        ),
        Err(FrontierError::DepthHorizon)
    );
    assert_eq!(
        fixture.frontier.snapshot().horizon.maximum_examined_ply,
        ChessPly(1)
    );
    assert_eq!(
        fixture.frontier.snapshot().mate[0].unwrap().conclusion,
        MateConclusion::Unknown
    );
}

#[test]
fn missing_question_key_and_branch_descriptor_keep_original_owner_first_and_horizons() {
    let goal = SearchGoal {
        depth_plies: NonZeroU16::new(3),
        mate_moves: NonZeroU16::new(2),
    };
    let mut fixture = Fixture::new(Position::startpos(), goal, None, false);
    let mut questions = fixture
        .frontier
        .missing_mate_questions(1, fixture.clock.domain(), fixture.clock.now().unwrap())
        .unwrap();
    let question = questions.pop().unwrap();
    assert_eq!(question.kind(), MateQuestionKind::AttackCandidate);
    assert_eq!(question.remaining(), ChessPly(3));
    let repeated = fixture
        .frontier
        .missing_mate_questions(1, fixture.clock.domain(), fixture.clock.now().unwrap())
        .unwrap();
    assert_eq!(question.key(), repeated[0].key());
    assert_eq!(
        fixture.frontier.missing_question_prefix(&question).unwrap(),
        vec![question.movement()]
    );
    assert_eq!(
        fixture
            .frontier
            .missing_question_first_move(&question)
            .unwrap(),
        question.movement()
    );
    assert_eq!(
        fixture
            .frontier
            .missing_question_chunk_limit(&question, 16)
            .unwrap(),
        2
    );
    // A descriptor is not a visit or a completed defense.
    assert_eq!(
        fixture.frontier.snapshot().work.total_node_visits(),
        Some(1)
    );
    let other = Fixture::new(Position::startpos(), goal, None, false);
    assert_eq!(
        other.frontier.validate_missing_question(&question),
        Err(FrontierError::ForeignHandle)
    );
    fixture.observe(&[question.movement()]).unwrap();
    assert_eq!(
        fixture.frontier.validate_missing_question(&question),
        Err(FrontierError::ForeignHandle)
    );
}

fn native_job(fixture: &Fixture) -> CpuJobEnvelope {
    CpuJobEnvelope {
        id: CpuJobId {
            authority: fixture.frontier.registration.authority,
            sequence: 1,
        },
        state_identity: fixture
            .frontier
            .registration
            .root_rules
            .snapshot()
            .identity(),
        question: Digest([30; 32]),
        root_restriction: Digest([31; 32]),
        evaluation_namespace: Digest([32; 32]),
        search_conditions: Digest([33; 32]),
        total_native_node_credit: 100,
        hard_deadline: fixture.frontier.original_deadline(),
    }
}
fn native_grant(job: CpuJobEnvelope, sequence: u64, credit: u64) -> SliceGrant {
    SliceGrant {
        job: job.id,
        execution: ExecutionId {
            epoch: job.id.authority.epoch,
            sequence,
        },
        owner: CpuWorkerId(0),
        node_credit: credit,
        soft_lease: Duration::from_millis(8),
        control_sequence: sequence,
    }
}
fn failed_native_report(grant: SliceGrant, nodes: u64, qnodes: u64) -> CpuSliceReport {
    CpuSliceReport {
        grant,
        reciprocal_request: None,
        actual_work: Some(CpuWork {
            nodes,
            quiescence_nodes: qnodes,
            tt_hits: 0,
        }),
        result: Err(CpuError::Allocation),
        cumulative_nodes: nodes,
        progress_sequence: 1,
        continuation_retained: false,
        terminal: true,
        admission_rejected: false,
        controls: None,
    }
}

#[test]
fn native_accounting_requires_the_original_ledger_receipt_and_charges_actual_once() {
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), Some(10), true);
    let job = native_job(&fixture);
    let mut ledger = CpuCreditLedger::new(job.id.authority, 100, fixture.clock).unwrap();
    fixture.frontier.registration.native_ledger = Some(ledger.identity());
    ledger.register(job).unwrap();
    assert_eq!(
        fixture
            .frontier
            .native_capacity(100, fixture.clock.domain(), fixture.clock.now().unwrap())
            .unwrap(),
        9
    );
    let grant = native_grant(job, 1, 9);
    ledger.reserve(grant).unwrap();
    let reservation = fixture
        .frontier
        .reserve_native(
            &job,
            grant,
            fixture.clock.domain(),
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    assert_eq!(
        fixture.observe(&[mv("e2e4")]),
        Err(FrontierError::NodesReserved)
    );
    let report = failed_native_report(grant, 5, 2);
    let settled = ledger.settle_attested(&report).unwrap();
    assert_eq!(
        fixture
            .frontier
            .settle_native(&reservation, &report, &settled)
            .unwrap(),
        NativeCloseReason::Executed
    );
    let snapshot = fixture.frontier.snapshot();
    assert_eq!(snapshot.work.total_node_visits(), Some(6));
    assert_eq!(snapshot.work.native_qsearch_nodes, 2);
    assert_eq!(snapshot.work.reserved_native_nodes, 0);
    assert_eq!(
        fixture
            .frontier
            .settle_native(&reservation, &report, &settled),
        Err(FrontierError::Duplicate)
    );
    assert_eq!(
        fixture
            .frontier
            .native_capacity(100, fixture.clock.domain(), fixture.clock.now().unwrap())
            .unwrap(),
        4
    );
}

#[test]
fn foreign_and_unknown_native_work_quarantine_without_refunding() {
    for foreign in [false, true] {
        let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), Some(10), true);
        let job = native_job(&fixture);
        let mut ledger = CpuCreditLedger::new(job.id.authority, 100, fixture.clock).unwrap();
        ledger.register(job).unwrap();
        let grant = native_grant(job, 1, 8);
        ledger.reserve(grant).unwrap();
        let reservation = fixture
            .frontier
            .reserve_native(
                &job,
                grant,
                fixture.clock.domain(),
                fixture.clock.now().unwrap(),
            )
            .unwrap();
        if foreign {
            let report = failed_native_report(grant, 3, 1);
            let settled = ledger.settle_attested(&report).unwrap();
            assert_eq!(
                fixture
                    .frontier
                    .settle_native(&reservation, &report, &settled),
                Err(FrontierError::NativeUnknown)
            );
        } else {
            fixture.frontier.native_usage_unknown(&reservation).unwrap();
        }
        fixture.frontier.close_role_call_budget();
        let snapshot = fixture.frontier.snapshot();
        assert_eq!(snapshot.work.reserved_native_nodes, 8);
        assert_eq!(snapshot.work.native_nodes, 0);
        assert!(!snapshot.work.native_usage_known);
        assert_eq!(
            snapshot.goal,
            FrontierGoalStatus::Stopped(FrontierStop::NativeUnknown)
        );
    }
}

#[test]
fn attested_owner_zero_work_rejection_is_distinct_and_does_not_count_as_a_visit() {
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), Some(10), true);
    let job = native_job(&fixture);
    let mut ledger = CpuCreditLedger::new(job.id.authority, 100, fixture.clock).unwrap();
    fixture.frontier.registration.native_ledger = Some(ledger.identity());
    ledger.register(job).unwrap();
    let grant = native_grant(job, 1, 8);
    ledger.reserve(grant).unwrap();
    let reservation = fixture
        .frontier
        .reserve_native(
            &job,
            grant,
            fixture.clock.domain(),
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    let mut report = failed_native_report(grant, 0, 0);
    report.admission_rejected = true;
    report.terminal = false;
    let settled = ledger.settle_attested(&report).unwrap();
    assert_eq!(
        fixture
            .frontier
            .settle_native(&reservation, &report, &settled)
            .unwrap(),
        NativeCloseReason::OwnerRejectedBeforeWork
    );
    assert_eq!(
        fixture.frontier.snapshot().work.total_node_visits(),
        Some(1)
    );
    assert_eq!(fixture.frontier.snapshot().work.reserved_native_nodes, 0);
}

#[test]
fn actual_store_generation_expiry_and_foreign_root_assembly_fail_closed() {
    let mut fixture = Fixture::new(Position::startpos(), SearchGoal::default(), None, false);
    let root_rules = Arc::clone(&fixture.frontier.registration.root_rules);
    let root_moves = Arc::clone(fixture.frontier.root_moves());
    let namespace = Arc::clone(&fixture.frontier.registration.model_namespace);
    let equivalent_domain = Arc::new(RootMoveDomain::new(&root_rules, None).unwrap());
    let attachment = FrontierAttachment {
        authority: fixture.frontier.registration.authority,
        mode: ExecutionMode::CpuPc,
        original_deadline: fixture.frontier.original_deadline(),
        root_budget: fixture.frontier.registration.root_budget,
        root_rules: &root_rules,
        root_moves: &root_moves,
        root_state: fixture.frontier.registration.root_state,
        model_namespace: &namespace,
        native_ledger: None,
    };
    fixture
        .frontier
        .validate_attachment(attachment, &fixture.stores)
        .unwrap();
    assert_eq!(
        fixture.frontier.validate_attachment(
            FrontierAttachment {
                root_moves: &equivalent_domain,
                ..attachment
            },
            &fixture.stores
        ),
        Err(FrontierError::ForeignRoot)
    );
    assert_eq!(fixture.stores.root(), Some(fixture.root_situation));
    fixture
        .stores
        .hot_focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let line = fixture.line(&[mv("e2e4")]);
    assert_eq!(
        fixture.frontier.observe_stored_line(
            &fixture.stores,
            line,
            fixture.clock.domain(),
            fixture.clock.now().unwrap()
        ),
        Err(FrontierError::StoreUnavailable)
    );
    assert_eq!(
        fixture.frontier.snapshot().goal,
        FrontierGoalStatus::Stopped(FrontierStop::EvidenceReclaimed)
    );
}
