//! 독립적인 인공 Rules tree와 수동 runtime script로 공통 경계를 검증한다.
//! 실제 chess Rules/Encoder/NN/물리 backend를 구현·검증한 결과가 아니다.

use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use rz_contracts::*;
use rz_search::contract_time::{ContractClock, ContractDeadlines};
use rz_search::contracts::*;
use rz_search::{EdgeStats, TreeLimits};

const EPOCH: ProcessEpoch = ProcessEpoch(7);

fn digest(byte: u8) -> Digest {
    Digest([byte; 32])
}
fn error(code: ErrorCode, detail: &'static str) -> ContractError {
    ContractError::new(code, Stage::Backend, detail)
}
fn chess_move(from: u8, to: u8) -> Move {
    Move::new(
        Square::try_new(from).unwrap(),
        Square::try_new(to).unwrap(),
        None,
    )
    .unwrap()
}

#[derive(Clone)]
struct ManualClock {
    tick: Arc<AtomicU64>,
}
impl ManualClock {
    fn new(tick: u64) -> Self {
        Self {
            tick: Arc::new(AtomicU64::new(tick)),
        }
    }
    fn set(&self, tick: u64) {
        self.tick.store(tick, Ordering::Release);
    }
}
impl ContractClock for ManualClock {
    fn domain(&self) -> ClockDomain {
        ClockDomain(EPOCH)
    }
    fn now(&self) -> Result<MonotonicTick, ContractError> {
        Ok(MonotonicTick(self.tick.load(Ordering::Acquire)))
    }
}

struct FixtureNode {
    side: Color,
    status: PlayStatus,
    edges: Vec<(Move, usize)>,
}
struct FixtureState {
    node: usize,
}
type Authority = Arc<dyn Fn() -> Result<(), ContractError> + Send + Sync>;

#[derive(Clone)]
struct Position {
    nodes: Arc<Vec<FixtureNode>>,
    index: usize,
    snapshot: PositionSnapshot<FixtureState>,
    legal: LegalMoveView,
    authority: Authority,
}

impl Position {
    fn from_nodes(nodes: Vec<FixtureNode>) -> Self {
        Self::at(Arc::new(nodes), 0, Arc::new(|| Ok(())))
    }
    fn at(nodes: Arc<Vec<FixtureNode>>, index: usize, authority: Authority) -> Self {
        let node = &nodes[index];
        let identity = StateIdentity {
            owner: OwnerId(99),
            revision: StateRevision(index as u64 + 1),
            semantic: digest(index as u8 + 1),
        };
        let classification = PositionClassification {
            play_status: node.status,
            claims: Arc::from([]),
            history: HistoryCompleteness::Complete,
            rules_profile: digest(70),
        };
        let snapshot = PositionSnapshot::try_new(
            identity,
            Arc::new(FixtureState { node: index }),
            node.side,
            classification,
        )
        .unwrap();
        let legal = LegalMoveView::try_new(
            identity,
            LegalOrderIdentity(digest(index as u8 + 30)),
            node.edges.iter().map(|(movement, _)| *movement).collect(),
            4096,
        )
        .unwrap();
        Self {
            nodes,
            index,
            snapshot,
            legal,
            authority,
        }
    }
    fn with_authority(mut self, authority: Authority) -> Self {
        self.authority = authority;
        self
    }
}

impl ContractPosition for Position {
    type State = FixtureState;
    fn snapshot(&self) -> &PositionSnapshot<Self::State> {
        &self.snapshot
    }
    fn legal(&self) -> &LegalMoveView {
        &self.legal
    }
    fn play(&self, movement: &Move) -> Result<Self, ContractError> {
        let child = self.nodes[self.index]
            .edges
            .iter()
            .find(|(candidate, _)| candidate == movement)
            .map(|(_, child)| *child)
            .ok_or_else(|| error(ErrorCode::InvalidInput, "illegal artificial move"))?;
        Ok(Self::at(
            Arc::clone(&self.nodes),
            child,
            Arc::clone(&self.authority),
        ))
    }
    fn validate_authority(&self) -> Result<(), ContractError> {
        (self.authority)()?;
        if self.snapshot.state().node != self.index
            || self.legal.state() != self.snapshot.identity()
        {
            return Err(error(
                ErrorCode::IdentityMismatch,
                "artificial Rules authority mismatch",
            ));
        }
        let expected: Vec<_> = self.nodes[self.index]
            .edges
            .iter()
            .map(|(movement, _)| *movement)
            .collect();
        if self.legal.moves() != expected {
            return Err(error(
                ErrorCode::IdentityMismatch,
                "artificial legal view mismatch",
            ));
        }
        Ok(())
    }
}

fn ongoing(side: Color, edges: &[(Move, usize)]) -> FixtureNode {
    FixtureNode {
        side,
        status: PlayStatus::Ongoing,
        edges: edges.to_vec(),
    }
}
fn terminal(side: Color, winner: Option<Color>) -> FixtureNode {
    FixtureNode {
        side,
        status: PlayStatus::Terminal {
            reason: if winner.is_some() {
                TerminalReason::Checkmate
            } else {
                TerminalReason::Stalemate
            },
            winner,
        },
        edges: Vec::new(),
    }
}
fn two_terminal_children() -> Position {
    Position::from_nodes(vec![
        ongoing(
            Color::White,
            &[(chess_move(12, 28), 1), (chess_move(11, 27), 2)],
        ),
        terminal(Color::Black, None),
        terminal(Color::Black, Some(Color::White)),
    ])
}
fn ongoing_child() -> Position {
    Position::from_nodes(vec![
        ongoing(Color::White, &[(chess_move(12, 28), 1)]),
        ongoing(Color::Black, &[(chess_move(52, 36), 2)]),
        terminal(Color::White, None),
    ])
}
fn promotion_position() -> Position {
    let promotions = [
        Promotion::Queen,
        Promotion::Rook,
        Promotion::Bishop,
        Promotion::Knight,
    ];
    Position::from_nodes(vec![
        ongoing(
            Color::White,
            &promotions.map(|promotion| {
                (
                    Move::new(
                        Square::try_new(48).unwrap(),
                        Square::try_new(56).unwrap(),
                        Some(promotion),
                    )
                    .unwrap(),
                    1,
                )
            }),
        ),
        terminal(Color::Black, None),
    ])
}

fn model() -> Arc<ModelDescriptor> {
    Arc::new(
        ModelDescriptor::try_new(
            ModelHandle {
                owner: OwnerId(30),
                slot: 1,
                generation: SlotGeneration(1),
                manifest: digest(9),
            },
            EncodingDescriptor {
                handle: EncodingHandle {
                    owner: OwnerId(31),
                    slot: 2,
                    generation: SlotGeneration(1),
                    manifest: digest(10),
                },
                history_length: 8,
                action_map: digest(12),
                history_policy: digest(13),
            },
            vec![PrecisionProfile::Fp32],
            2,
            4,
        )
        .unwrap(),
    )
}
fn scope() -> AcceptanceScope {
    let model = model();
    AcceptanceScope {
        game: GameGeneration(1),
        root: RootGeneration(1),
        model: model.handle(),
        encoding: model.encoding().handle,
        backend: digest(11),
    }
}
fn config(ids: Arc<IdAllocator>, max_simulations: u64) -> ContractSearchConfig {
    let deadline = |tick| Deadline {
        clock: ClockDomain(EPOCH),
        at: MonotonicTick(tick),
    };
    ContractSearchConfig {
        scope: scope(),
        model: model(),
        precision: PrecisionProfile::Fp32,
        compute: ComputeBudget {
            min_steps: 2,
            max_steps: 2,
            require_full: true,
        },
        bytes: ByteBudget {
            host: 4096,
            device: 0,
            pinned: 0,
        },
        policy_tolerance: 1e-6,
        wdl_tolerance: 1e-6,
        cancellation: CancelToken::new(),
        deadlines: ContractDeadlines {
            soft: deadline(80),
            admission: deadline(90),
            hard: deadline(100),
            output: deadline(110),
        },
        ids,
        max_simulations,
        tree_limits: TreeLimits {
            max_nodes: 128,
            max_edges: 512,
            max_depth: 32,
            max_legal_moves: 256,
            ..TreeLimits::default()
        },
    }
}
fn input_key(
    position: &Position,
    encoding: &EncodingDescriptor,
) -> Result<EvalInputKey, ContractError> {
    assert_eq!(encoding.action_map, digest(12));
    // Explicit synthetic encoder attestation of the owned fixture state, not a product board hash.
    Ok(EvalInputKey(digest(
        position.snapshot.state().node as u8 + 100,
    )))
}

#[derive(Default)]
struct ScriptedEvaluator {
    submissions: Vec<Arc<EvalRequest<FixtureState>>>,
    results: VecDeque<EvalResult>,
    canceled: Vec<RequestId>,
    submit_error: Option<ContractError>,
    cancel_error: Option<ContractError>,
    polls: usize,
}
impl Evaluator<FixtureState> for ScriptedEvaluator {
    fn submit(&mut self, request: Arc<EvalRequest<FixtureState>>) -> Result<(), ContractError> {
        if let Some(error) = self.submit_error {
            return Err(error);
        }
        self.submissions.push(request);
        Ok(())
    }
    fn poll(&mut self) -> Option<EvalResult> {
        self.polls += 1;
        self.results.pop_front()
    }
    fn cancel(&mut self, request: RequestId) -> Result<(), ContractError> {
        self.canceled.push(request);
        self.cancel_error.map_or(Ok(()), Err)
    }
}

fn completed(request: &EvalRequest<FixtureState>, policy: &[f64], wdl: [f32; 3]) -> EvalOutput {
    let context = request.context();
    EvalOutput {
        context,
        legal: request.legal().clone(),
        policy: LegalPolicy::try_new(policy.to_vec(), 0.01).unwrap(),
        wdl: Wdl::try_new(wdl[0], wdl[1], wdl[2], 0.01).unwrap(),
        viewpoint: Viewpoint::SideToMove,
        actual: ActualCompute {
            precision: context.precision,
            steps: 2,
            full: true,
            backend: context.backend,
            execution: Some(ExecutionId::new(EPOCH, context.request.sequence)),
            provenance: CacheProvenance::Computed,
        },
    }
}
fn pump(
    search: &mut ContractSearch<Position>,
    runtime: &mut ScriptedEvaluator,
    clock: &ManualClock,
) -> ContractPumpEvent {
    search.pump(runtime, clock, scope, input_key)
}
fn new_search(position: Position, cap: u64) -> ContractSearch<Position> {
    ContractSearch::new(position, config(Arc::new(IdAllocator::new(EPOCH)), cap)).unwrap()
}

#[test]
fn root_restriction_keeps_full_nn_order_and_excludes_the_better_outside_move() {
    let allowed = chess_move(12, 28);
    let excluded = chess_move(11, 27);
    let mut search = new_search(two_terminal_children(), 3);
    search.set_root_moves(&[allowed]).unwrap();
    assert_eq!(search.best_move(), Some(allowed));
    let mut runtime = ScriptedEvaluator::default();
    let clock = ManualClock::new(1);
    pump(&mut search, &mut runtime, &clock);
    let request = runtime.submissions[0].clone();
    assert_eq!(request.legal().moves(), [allowed, excluded]);
    // All raw prior mass, as well as the winning child, is outside the domain.
    // The declared zero-subset-mass policy uses uniform only within the domain.
    let output = completed(&request, &[0.0, 1.0], [0.0, 1.0, 0.0]);
    runtime
        .results
        .push_back(EvalResult::Completed(output.clone()));
    assert!(matches!(
        pump(&mut search, &mut runtime, &clock),
        ContractPumpEvent::Accepted { .. }
    ));
    assert_eq!(output.policy.probabilities(), [0.0, 1.0]);
    for _ in 0..3 {
        assert!(matches!(
            pump(&mut search, &mut runtime, &clock),
            ContractPumpEvent::Accepted {
                traversed_edges: 1,
                ..
            }
        ));
        assert_eq!(search.best_move(), Some(allowed));
    }
    pump(&mut search, &mut runtime, &clock);
    let outcome = search.outcome();
    assert!(matches!(outcome.status, ContractSearchStatus::Completed));
    assert_eq!(outcome.best_move, Some(allowed));
    assert!(!outcome.fallback_used);
    assert_eq!(
        outcome.root_stats,
        [(
            allowed,
            EdgeStats {
                prior: 1.0,
                visits: 3,
                value_sum: 0.0
            }
        )]
    );
    assert_eq!(runtime.submissions.len(), 1);
}

#[test]
fn root_restriction_never_projects_the_opponent_child_legal_array() {
    let excluded = chess_move(12, 28);
    let allowed = chess_move(11, 27);
    let response = chess_move(52, 36);
    let position = Position::from_nodes(vec![
        ongoing(Color::White, &[(excluded, 1), (allowed, 2)]),
        terminal(Color::Black, Some(Color::White)),
        ongoing(Color::Black, &[(response, 3)]),
        terminal(Color::White, None),
    ]);
    let mut search = new_search(position, 2);
    // Index 1 exists at the root and does not exist in its one-move child.
    search.set_root_moves(&[allowed]).unwrap();
    let mut runtime = ScriptedEvaluator::default();
    let clock = ManualClock::new(1);
    pump(&mut search, &mut runtime, &clock);
    let root = runtime.submissions[0].clone();
    runtime.results.push_back(EvalResult::Completed(completed(
        &root,
        &[0.9, 0.1],
        [0.0, 1.0, 0.0],
    )));
    pump(&mut search, &mut runtime, &clock);
    pump(&mut search, &mut runtime, &clock);
    let child = runtime.submissions[1].clone();
    assert_eq!(child.legal().moves(), [response]);
    assert_eq!(child.position().side_to_move(), Color::Black);
    runtime.results.push_back(EvalResult::Completed(completed(
        &child,
        &[1.0],
        [0.0, 1.0, 0.0],
    )));
    assert!(matches!(
        pump(&mut search, &mut runtime, &clock),
        ContractPumpEvent::Accepted {
            traversed_edges: 1,
            ..
        }
    ));
    assert!(matches!(
        pump(&mut search, &mut runtime, &clock),
        ContractPumpEvent::Accepted {
            traversed_edges: 2,
            ..
        }
    ));
    pump(&mut search, &mut runtime, &clock);
    assert_eq!(search.outcome().best_move, Some(allowed));
    assert_eq!(search.outcome().root_stats[0].1.visits, 2);
}

#[test]
fn root_restriction_preserves_rules_order_promotions_and_zero_work_fallback() {
    let position = promotion_position();
    let legal = position.legal().moves().to_vec();
    let mut search = new_search(position, 0);
    search.set_root_moves(&[legal[3], legal[1]]).unwrap();
    let mut runtime = ScriptedEvaluator::default();
    pump(&mut search, &mut runtime, &ManualClock::new(1));
    assert_eq!(search.outcome().best_move, Some(legal[1]));
    assert!(search.outcome().fallback_used);
    assert!(runtime.submissions.is_empty());
    assert_eq!(runtime.polls, 0);
}

#[test]
fn root_subset_priors_are_conditioned_without_changing_raw_head_order() {
    let position = promotion_position();
    let legal = position.legal().moves().to_vec();
    let mut search = new_search(position, 1);
    search.set_root_moves(&[legal[3], legal[1]]).unwrap();
    let mut runtime = ScriptedEvaluator::default();
    let clock = ManualClock::new(1);
    pump(&mut search, &mut runtime, &clock);
    let request = runtime.submissions[0].clone();
    assert_eq!(request.legal().moves(), legal);
    runtime.results.push_back(EvalResult::Completed(completed(
        &request,
        &[0.1, 0.2, 0.3, 0.4],
        [0.0, 1.0, 0.0],
    )));
    pump(&mut search, &mut runtime, &clock);
    let edges = search.outcome().root_stats;
    assert_eq!(
        edges
            .iter()
            .map(|(movement, _)| *movement)
            .collect::<Vec<_>>(),
        [legal[1], legal[3]]
    );
    assert!((edges[0].1.prior - 1.0 / 3.0).abs() < 1e-12);
    assert!((edges[1].1.prior - 2.0 / 3.0).abs() < 1e-12);
    assert_eq!(search.best_move(), Some(legal[1]));
}

#[test]
fn rejected_or_late_root_restrictions_do_not_change_the_admitted_domain() {
    let first = chess_move(12, 28);
    let second = chess_move(11, 27);
    for requested in [vec![], vec![first, first], vec![chess_move(0, 8)]] {
        let mut search = new_search(two_terminal_children(), 1);
        let rejected = search.set_root_moves(&requested).unwrap_err();
        assert_eq!(rejected.code, ErrorCode::InvalidInput);
        assert_eq!(rejected.stage, Stage::Admission);
        assert_eq!(search.best_move(), Some(first));
        search.set_root_moves(&[second]).unwrap();
        assert_eq!(search.best_move(), Some(second));
    }
    let mut search = new_search(two_terminal_children(), 1);
    search.set_root_moves(&[second]).unwrap();
    let mut runtime = ScriptedEvaluator::default();
    pump(&mut search, &mut runtime, &ManualClock::new(1));
    let original = search.pending_request();
    let rejected = search.set_root_moves(&[first]).unwrap_err();
    assert_eq!(rejected.code, ErrorCode::UnsupportedContract);
    assert_eq!(rejected.stage, Stage::Admission);
    assert_eq!(search.pending_request(), original);
    assert_eq!(search.best_move(), Some(second));
}

#[test]
fn root_restriction_does_not_relax_the_original_full_legal_view_resource_bound() {
    let mut settings = config(Arc::new(IdAllocator::new(EPOCH)), 1);
    settings.tree_limits.max_legal_moves = 1;
    let mut search = ContractSearch::new(two_terminal_children(), settings).unwrap();
    let allowed = chess_move(11, 27);
    search.set_root_moves(&[allowed]).unwrap();
    let mut runtime = ScriptedEvaluator::default();
    pump(&mut search, &mut runtime, &ManualClock::new(1));
    assert!(matches!(
        search.outcome().status,
        ContractSearchStatus::Failed(ContractSearchFailure::Tree(
            rz_search::SearchError::EdgeLimit
        ))
    ));
    assert_eq!(search.outcome().best_move, Some(allowed));
    assert!(search.outcome().fallback_used);
    assert!(runtime.submissions.is_empty());
}

#[cfg(feature = "experimental-batch")]
#[test]
fn restricted_root_survives_parallel_child_completion_order_and_late_cancellation_results() {
    let excluded = chess_move(12, 28);
    let allowed = [chess_move(11, 27), chess_move(6, 21)];
    let replies = [chess_move(52, 36), chess_move(51, 35)];
    for order in [Some([1, 2]), Some([2, 1]), None] {
        let position = Position::from_nodes(vec![
            ongoing(
                Color::White,
                &[(excluded, 1), (allowed[0], 2), (allowed[1], 3)],
            ),
            terminal(Color::Black, Some(Color::White)),
            ongoing(Color::Black, &[(replies[0], 4)]),
            ongoing(Color::Black, &[(replies[1], 4)]),
            terminal(Color::White, None),
        ]);
        let mut search = new_search(position, 2);
        search.set_root_moves(&allowed).unwrap();
        search.set_parallelism(2).unwrap();
        let mut runtime = ScriptedEvaluator::default();
        let clock = ManualClock::new(1);
        pump(&mut search, &mut runtime, &clock);
        let root = runtime.submissions[0].clone();
        assert_eq!(root.legal().moves(), [excluded, allowed[0], allowed[1]]);
        runtime.results.push_back(EvalResult::Completed(completed(
            &root,
            &[0.98, 0.01, 0.01],
            [0.0, 1.0, 0.0],
        )));
        pump(&mut search, &mut runtime, &clock);
        assert!(matches!(
            pump(&mut search, &mut runtime, &clock),
            ContractPumpEvent::Submitted { .. }
        ));
        assert!(matches!(
            pump(&mut search, &mut runtime, &clock),
            ContractPumpEvent::Submitted { .. }
        ));
        assert_eq!(runtime.submissions.len(), 3);
        for child in &runtime.submissions[1..] {
            let node = child.position().state().node;
            assert!((2..=3).contains(&node));
            assert_eq!(child.legal().moves(), [replies[node - 2]]);
        }
        if let Some(order) = order {
            for index in order {
                let child = runtime.submissions[index].clone();
                runtime.results.push_back(EvalResult::Completed(completed(
                    &child,
                    &[1.0],
                    [0.0, 1.0, 0.0],
                )));
                assert!(matches!(
                    pump(&mut search, &mut runtime, &clock),
                    ContractPumpEvent::Accepted {
                        traversed_edges: 1,
                        ..
                    }
                ));
            }
            pump(&mut search, &mut runtime, &clock);
            let outcome = search.outcome();
            assert_eq!(outcome.root_stats.len(), 2);
            assert!(
                outcome
                    .root_stats
                    .iter()
                    .all(|(movement, edge)| allowed.contains(movement) && edge.visits == 1)
            );
            assert!(allowed.contains(&outcome.best_move.unwrap()));
        } else {
            let late = completed(&runtime.submissions[1], &[1.0], [0.0, 1.0, 0.0]);
            clock.set(100);
            pump(&mut search, &mut runtime, &clock);
            assert_eq!(runtime.canceled.len(), 2);
            runtime.results.push_back(EvalResult::Completed(late));
            pump(&mut search, &mut runtime, &clock);
            assert_eq!(search.outcome().counters.completed_visits, 0);
            assert_eq!(search.outcome().best_move, Some(allowed[0]));
        }
    }
}

#[test]
fn artificial_terminal_tree_has_independent_visit_and_value_expectations() {
    let mut search = new_search(two_terminal_children(), 3);
    let mut runtime = ScriptedEvaluator::default();
    let clock = ManualClock::new(1);
    assert!(matches!(
        pump(&mut search, &mut runtime, &clock),
        ContractPumpEvent::Submitted { .. }
    ));
    let request = runtime.submissions[0].clone();
    runtime.results.push_back(EvalResult::Completed(completed(
        &request,
        &[0.5, 0.5],
        [0.2, 0.6, 0.2],
    )));
    let event = pump(&mut search, &mut runtime, &clock);
    let ContractPumpEvent::Accepted {
        traversed_edges: 0,
        evaluation: Some(metadata),
        ..
    } = event
    else {
        panic!("a validated root initialization must retain passive evaluation metadata");
    };
    assert_eq!(metadata.context.request, request.context());
    assert_eq!(metadata.context.execution, metadata.actual.execution);
    assert_eq!(metadata.actual.provenance, CacheProvenance::Computed);
    assert_eq!(search.outcome().counters.accepted_backups, 0);
    for _ in 0..3 {
        assert!(matches!(
            pump(&mut search, &mut runtime, &clock),
            ContractPumpEvent::Accepted {
                request: None,
                traversed_edges: 1,
                evaluation: None,
                ..
            }
        ));
    }
    assert!(matches!(
        pump(&mut search, &mut runtime, &clock),
        ContractPumpEvent::Finished
    ));
    let outcome = search.outcome();
    assert!(matches!(outcome.status, ContractSearchStatus::Completed));
    assert_eq!(outcome.best_move, Some(chess_move(11, 27)));
    assert!(!outcome.fallback_used);
    assert_eq!(
        outcome.root_stats,
        vec![
            (
                chess_move(12, 28),
                EdgeStats {
                    prior: 0.5,
                    visits: 1,
                    value_sum: 0.0
                }
            ),
            (
                chess_move(11, 27),
                EdgeStats {
                    prior: 0.5,
                    visits: 2,
                    value_sum: 2.0
                }
            ),
        ]
    );
    assert_eq!(outcome.counters.root_initializations, 1);
    assert_eq!(outcome.counters.accepted_backups, 3);
    assert_eq!(outcome.counters.reservations_released, 4);
    assert_eq!(outcome.metrics.submissions, 1);
    assert_eq!(outcome.metrics.accepted_execution_ids, 1);
    assert_eq!(runtime.submissions.len(), 1);
}

#[test]
fn poll_none_is_one_prompt_action_and_keeps_reservation() {
    let mut search = new_search(ongoing_child(), 1);
    let mut runtime = ScriptedEvaluator::default();
    let clock = ManualClock::new(1);
    pump(&mut search, &mut runtime, &clock);
    let request = search.pending_request();
    assert!(matches!(
        pump(&mut search, &mut runtime, &clock),
        ContractPumpEvent::Waiting
    ));
    assert_eq!(runtime.polls, 1);
    assert_eq!(runtime.submissions.len(), 1);
    assert_eq!(search.pending_request(), request);
    assert_eq!(search.outcome().counters.reservations_released, 0);
    assert_eq!(search.outcome().best_move, Some(chess_move(12, 28)));
    assert!(search.outcome().fallback_used);
}

#[test]
fn exact_terminal_and_zero_budget_bypass_runtime_and_preserve_meaning() {
    let mut terminal = new_search(
        Position::from_nodes(vec![terminal(Color::White, Some(Color::Black))]),
        5,
    );
    let mut runtime = ScriptedEvaluator::default();
    assert!(matches!(
        pump(&mut terminal, &mut runtime, &ManualClock::new(1)),
        ContractPumpEvent::Finished
    ));
    assert!(matches!(
        terminal.outcome().status,
        ContractSearchStatus::Terminal {
            reason: TerminalReason::Checkmate,
            winner: Some(Color::Black),
            value: -1.0,
        }
    ));
    assert_eq!(terminal.outcome().best_move, None);
    let mut zero = new_search(ongoing_child(), 0);
    pump(&mut zero, &mut runtime, &ManualClock::new(1));
    assert!(matches!(
        zero.outcome().status,
        ContractSearchStatus::Completed
    ));
    assert!(zero.outcome().fallback_used);
    assert_eq!(runtime.submissions.len(), 0);
    assert_eq!(runtime.polls, 0);
}

#[test]
fn valid_f32_heads_use_explicit_normalized_f64_before_tree() {
    let mut search = new_search(ongoing_child(), 1);
    let mut runtime = ScriptedEvaluator::default();
    let clock = ManualClock::new(1);
    pump(&mut search, &mut runtime, &clock);
    let root = runtime.submissions[0].clone();
    runtime.results.push_back(EvalResult::Completed(completed(
        &root,
        &[1.0],
        [0.0, 1.0, 0.0],
    )));
    pump(&mut search, &mut runtime, &clock);
    pump(&mut search, &mut runtime, &clock);
    let child = runtime.submissions[1].clone();
    let raw = [0.8_f32, 0.15_f32, 0.05_f32];
    runtime
        .results
        .push_back(EvalResult::Completed(completed(&child, &[1.0], raw)));
    assert!(matches!(
        pump(&mut search, &mut runtime, &clock),
        ContractPumpEvent::Accepted {
            traversed_edges: 1,
            ..
        }
    ));
    let sum = raw.iter().map(|value| f64::from(*value)).sum::<f64>();
    let expected = -(f64::from(raw[0]) - f64::from(raw[2])) / sum;
    assert_ne!(sum, 1.0);
    assert!((search.outcome().root_stats[0].1.value_sum - expected).abs() < 1e-12);
    assert_eq!(search.outcome().root_stats[0].1.visits, 1);

    let mut promotions = new_search(promotion_position(), 1);
    let mut runtime = ScriptedEvaluator::default();
    pump(&mut promotions, &mut runtime, &clock);
    let request = runtime.submissions[0].clone();
    let raw = [0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32].map(f64::from);
    runtime.results.push_back(EvalResult::Completed(completed(
        &request,
        &raw,
        [0.0, 1.0, 0.0],
    )));
    pump(&mut promotions, &mut runtime, &clock);
    let sum = raw.iter().sum::<f64>();
    for ((movement, edge), (raw, promotion)) in
        promotions
            .outcome()
            .root_stats
            .iter()
            .zip(raw.into_iter().zip([
                Promotion::Queen,
                Promotion::Rook,
                Promotion::Bishop,
                Promotion::Knight,
            ]))
    {
        assert_eq!(movement.promotion, Some(promotion));
        assert!((edge.prior - raw / sum).abs() < 1e-12);
        assert_eq!(edge.visits, 0);
    }
}

#[test]
fn partial_heads_wrong_backend_and_permuted_four_promotions_fail_without_visit() {
    for case in 0..3 {
        let mut search = new_search(promotion_position(), 1);
        let mut runtime = ScriptedEvaluator::default();
        let clock = ManualClock::new(1);
        pump(&mut search, &mut runtime, &clock);
        let request = runtime.submissions[0].clone();
        let mut output = completed(&request, &[0.25; 4], [0.0, 1.0, 0.0]);
        match case {
            0 => output.policy = LegalPolicy::try_new(vec![0.5, 0.25, 0.25], 0.0).unwrap(),
            1 => output.actual.backend = digest(200),
            _ => {
                let mut moves = request.legal().moves().to_vec();
                moves.swap(0, 3);
                output.legal = LegalMoveView::try_new(
                    request.legal().state(),
                    request.legal().order(),
                    moves,
                    4096,
                )
                .unwrap();
            }
        }
        runtime.results.push_back(EvalResult::Completed(output));
        assert!(matches!(
            pump(&mut search, &mut runtime, &clock),
            ContractPumpEvent::Finished
        ));
        assert!(matches!(
            search.outcome().status,
            ContractSearchStatus::Failed(ContractSearchFailure::Boundary(_))
        ));
        assert_eq!(search.outcome().counters.accepted_backups, 0);
        assert_eq!(search.outcome().counters.root_initializations, 0);
        assert_eq!(search.outcome().counters.reservations_released, 1);
        assert_eq!(search.outcome().metrics.accepted_outputs, 0);
        assert!(search.outcome().fallback_used);
    }
}

#[test]
fn declared_strict_raw_tolerances_cannot_be_bypassed_by_loose_common_construction() {
    for wdl_case in [false, true] {
        let mut search = new_search(ongoing_child(), 1);
        let mut runtime = ScriptedEvaluator::default();
        let clock = ManualClock::new(1);
        pump(&mut search, &mut runtime, &clock);
        let request = runtime.submissions[0].clone();
        let output = if wdl_case {
            completed(&request, &[1.0], [0.5, 0.25, 0.249])
        } else {
            completed(&request, &[0.999], [0.0, 1.0, 0.0])
        };
        runtime.results.push_back(EvalResult::Completed(output));
        pump(&mut search, &mut runtime, &clock);
        assert!(matches!(
            search.outcome().status,
            ContractSearchStatus::Failed(ContractSearchFailure::Boundary(ContractError {
                code: ErrorCode::NumericalFailure,
                ..
            }))
        ));
        assert_eq!(search.outcome().metrics.accepted_outputs, 0);
        assert_eq!(search.outcome().counters.reservations_released, 1);
        assert!(search.outcome().root_stats.is_empty());
    }
}

#[test]
fn foreign_and_malformed_terminal_contexts_keep_active_request_reserved() {
    let mut search = new_search(ongoing_child(), 1);
    let mut runtime = ScriptedEvaluator::default();
    let clock = ManualClock::new(1);
    pump(&mut search, &mut runtime, &clock);
    let request = runtime.submissions[0].clone();
    for case in 0..4 {
        let mut context = request.context();
        if case % 2 == 0 {
            context.request.sequence += 100;
        } else {
            context.selection.sequence += 100;
        }
        let context = CompletionContext {
            request: context,
            execution: None,
        };
        let result = match case {
            0 => EvalResult::Canceled(context),
            1 => EvalResult::Expired(context),
            2 => EvalResult::Stale(context),
            _ => EvalResult::Failed(EvalFailure {
                context,
                error: error(ErrorCode::BackendFailure, "foreign failure"),
                recovery: RecoveryOutcome::Failed,
            }),
        };
        runtime.results.push_back(result);
        match pump(&mut search, &mut runtime, &clock) {
            ContractPumpEvent::RejectedResult {
                error: diagnostic,
                result,
            } => {
                assert_eq!(diagnostic.code, ErrorCode::IdentityMismatch);
                if case == 3 {
                    match *result {
                        EvalResult::Failed(failure) => {
                            assert_eq!(
                                failure.error,
                                error(ErrorCode::BackendFailure, "foreign failure")
                            );
                            assert_eq!(failure.recovery, RecoveryOutcome::Failed);
                            assert_eq!(failure.context, context);
                        }
                        other => panic!("foreign failure cause lost: {other:?}"),
                    }
                }
            }
            other => panic!("foreign result consumed reservation: {other:?}"),
        }
        assert_eq!(search.pending_request(), Some(request.context().request));
        assert_eq!(search.outcome().counters.reservations_released, 0);
        assert!(!search.is_finished());
    }
    runtime.results.push_back(EvalResult::Completed(completed(
        &request,
        &[1.0],
        [0.0, 1.0, 0.0],
    )));
    assert!(matches!(
        pump(&mut search, &mut runtime, &clock),
        ContractPumpEvent::Accepted { .. }
    ));
    assert_eq!(search.outcome().metrics.diagnostics, 4);
    assert_eq!(search.outcome().counters.reservations_released, 1);
    assert_eq!(runtime.canceled.len(), 0);
}

#[test]
fn prior_duplicate_response_does_not_consume_new_selection_then_failure_keeps_recovery() {
    let mut search = new_search(ongoing_child(), 1);
    let mut runtime = ScriptedEvaluator::default();
    let clock = ManualClock::new(1);
    pump(&mut search, &mut runtime, &clock);
    let root = runtime.submissions[0].clone();
    let root_output = completed(&root, &[1.0], [0.0, 1.0, 0.0]);
    runtime
        .results
        .push_back(EvalResult::Completed(root_output.clone()));
    pump(&mut search, &mut runtime, &clock);
    pump(&mut search, &mut runtime, &clock);
    let child = runtime.submissions[1].clone();
    runtime
        .results
        .push_back(EvalResult::Completed(root_output));
    assert!(matches!(
        pump(&mut search, &mut runtime, &clock),
        ContractPumpEvent::RejectedResult { .. }
    ));
    assert_eq!(search.pending_request(), Some(child.context().request));
    assert_eq!(search.outcome().counters.reservations_released, 1);
    let failure = EvalFailure {
        context: CompletionContext {
            request: child.context(),
            execution: Some(ExecutionId::new(EPOCH, 2)),
        },
        error: error(ErrorCode::BackendFailure, "actual evaluator failure"),
        recovery: RecoveryOutcome::Completed,
    };
    runtime
        .results
        .push_back(EvalResult::Failed(failure.clone()));
    pump(&mut search, &mut runtime, &clock);
    match search.outcome().status {
        ContractSearchStatus::Failed(ContractSearchFailure::Evaluation(actual)) => {
            assert_eq!(actual.error, failure.error);
            assert_eq!(actual.context, failure.context);
            assert_eq!(actual.recovery, RecoveryOutcome::Completed);
        }
        other => panic!("failure was hidden: {other:?}"),
    }
    assert_eq!(search.outcome().counters.accepted_backups, 0);
    assert_eq!(search.outcome().counters.reservations_released, 2);
    assert!(search.outcome().fallback_used);
}

#[test]
fn matching_canceled_expired_and_stale_release_once_without_root_expansion() {
    for expected in [
        ContractStopReason::Canceled,
        ContractStopReason::Expired,
        ContractStopReason::Stale,
    ] {
        let mut search = new_search(ongoing_child(), 1);
        let mut runtime = ScriptedEvaluator::default();
        let clock = ManualClock::new(1);
        pump(&mut search, &mut runtime, &clock);
        let context = CompletionContext {
            request: runtime.submissions[0].context(),
            execution: None,
        };
        runtime.results.push_back(match expected {
            ContractStopReason::Canceled => EvalResult::Canceled(context),
            ContractStopReason::Expired => EvalResult::Expired(context),
            _ => EvalResult::Stale(context),
        });
        pump(&mut search, &mut runtime, &clock);
        pump(&mut search, &mut runtime, &clock);
        assert!(
            matches!(search.outcome().status, ContractSearchStatus::Stopped { reason, .. } if reason == expected)
        );
        assert_eq!(search.outcome().counters.reservations_released, 1);
        assert_eq!(search.outcome().counters.accepted_backups, 0);
        assert_eq!(runtime.polls, 1);
        assert!(search.outcome().root_stats.is_empty());
    }
}

#[test]
fn live_scope_changes_at_final_guard_reject_prepared_output_without_mutation() {
    for replacement in 0..5 {
        let mut search = new_search(ongoing_child(), 1);
        let mut runtime = ScriptedEvaluator::default();
        let clock = ManualClock::new(1);
        pump(&mut search, &mut runtime, &clock);
        let request = runtime.submissions[0].clone();
        runtime.results.push_back(EvalResult::Completed(completed(
            &request,
            &[1.0],
            [0.0, 1.0, 0.0],
        )));
        let mut observations = 0;
        let event = search.pump(
            &mut runtime,
            &clock,
            || {
                observations += 1;
                let mut current = scope();
                if observations >= 3 {
                    match replacement {
                        0 => current.root = RootGeneration(2),
                        1 => current.game = GameGeneration(2),
                        2 => current.model.generation = SlotGeneration(2),
                        3 => current.encoding.generation = SlotGeneration(2),
                        _ => current.backend = digest(200),
                    }
                }
                current
            },
            input_key,
        );
        assert!(matches!(event, ContractPumpEvent::Finished));
        assert_eq!(observations, 3);
        assert!(matches!(
            search.outcome().status,
            ContractSearchStatus::Stopped {
                reason: ContractStopReason::Stale,
                ..
            }
        ));
        assert_eq!(search.outcome().counters.root_initializations, 0);
        assert_eq!(search.outcome().counters.reservations_released, 1);
        assert!(search.outcome().root_stats.is_empty());
        assert_eq!(search.outcome().metrics.accepted_execution_ids, 0);
    }
}

#[test]
fn final_rules_attestation_crossing_exact_deadline_or_cancel_blocks_commit() {
    for cancel_case in [false, true] {
        let clock = ManualClock::new(1);
        let calls = Arc::new(AtomicU64::new(0));
        let trigger = Arc::new(AtomicU64::new(u64::MAX));
        let mut config = config(Arc::new(IdAllocator::new(EPOCH)), 1);
        let cancellation = config.cancellation.clone();
        let authority: Authority = {
            let calls = Arc::clone(&calls);
            let trigger = Arc::clone(&trigger);
            let clock = clock.clone();
            Arc::new(move || {
                let call = calls.fetch_add(1, Ordering::AcqRel) + 1;
                if call == trigger.load(Ordering::Acquire) {
                    if cancel_case {
                        cancellation.cancel();
                    } else {
                        clock.set(100);
                    }
                }
                Ok(())
            })
        };
        config.tree_limits.probability_tolerance = 1e-9;
        let mut search =
            ContractSearch::new(ongoing_child().with_authority(authority), config).unwrap();
        let mut runtime = ScriptedEvaluator::default();
        pump(&mut search, &mut runtime, &clock);
        let request = runtime.submissions[0].clone();
        runtime.results.push_back(EvalResult::Completed(completed(
            &request,
            &[1.0],
            [0.0, 1.0, 0.0],
        )));
        calls.store(0, Ordering::Release);
        trigger.store(3, Ordering::Release); // pending guard, metadata authority, final Rules authority
        let rejected = pump(&mut search, &mut runtime, &clock);
        assert!(!matches!(
            rejected,
            ContractPumpEvent::Accepted {
                evaluation: Some(_),
                ..
            }
        ));
        assert!(
            matches!(search.outcome().status, ContractSearchStatus::Stopped { reason, .. }
            if reason == if cancel_case { ContractStopReason::Canceled } else { ContractStopReason::Expired })
        );
        assert_eq!(search.outcome().counters.root_initializations, 0);
        assert_eq!(search.outcome().counters.reservations_released, 1);
        assert_eq!(search.outcome().metrics.accepted_outputs, 0);
        assert!(search.outcome().root_stats.is_empty());
    }
}

#[test]
fn in_flight_soft_and_admission_expiry_do_not_reject_before_hard() {
    for (tick, expected) in [
        (80, ContractStopReason::SoftBudget),
        (90, ContractStopReason::AdmissionClosed),
    ] {
        let mut search = new_search(ongoing_child(), 1);
        let mut runtime = ScriptedEvaluator::default();
        let clock = ManualClock::new(1);
        pump(&mut search, &mut runtime, &clock);
        let request = runtime.submissions[0].clone();
        runtime.results.push_back(EvalResult::Completed(completed(
            &request,
            &[1.0],
            [0.0, 1.0, 0.0],
        )));
        clock.set(tick);
        assert!(matches!(
            pump(&mut search, &mut runtime, &clock),
            ContractPumpEvent::Accepted {
                traversed_edges: 0,
                ..
            }
        ));
        assert_eq!(search.outcome().counters.root_initializations, 1);
        pump(&mut search, &mut runtime, &clock);
        assert!(
            matches!(search.outcome().status, ContractSearchStatus::Stopped { reason, .. } if reason == expected)
        );
        assert_eq!(runtime.submissions.len(), 1);
    }
}

#[test]
fn hard_expiry_cancels_one_logical_request_and_cleanup_error_keeps_primary() {
    let mut search = new_search(ongoing_child(), 1);
    let mut runtime = ScriptedEvaluator::default();
    let clock = ManualClock::new(1);
    pump(&mut search, &mut runtime, &clock);
    let id = search.pending_request().unwrap();
    let cleanup = error(ErrorCode::BackendFailure, "logical cancellation failed");
    runtime.cancel_error = Some(cleanup);
    clock.set(100);
    pump(&mut search, &mut runtime, &clock);
    pump(&mut search, &mut runtime, &clock);
    assert_eq!(runtime.canceled, vec![id]);
    assert_eq!(runtime.polls, 0);
    assert_eq!(search.outcome().counters.reservations_released, 1);
    match search.outcome().status {
        ContractSearchStatus::Failed(ContractSearchFailure::Cleanup {
            primary,
            runtime: Some(actual),
            ..
        }) => {
            assert_eq!(actual, cleanup);
            assert!(matches!(
                *primary,
                ContractSearchFailure::Boundary(ContractError {
                    code: ErrorCode::Expired,
                    ..
                })
            ));
        }
        other => panic!("cleanup hid primary cause: {other:?}"),
    }
    assert!(search.outcome().fallback_used);
}

#[test]
fn submit_error_releases_once_and_never_turns_into_uniform_policy_or_success() {
    let mut search = new_search(ongoing_child(), 1);
    let failure = error(ErrorCode::BackendUnavailable, "CPU mock unavailable");
    let mut runtime = ScriptedEvaluator {
        submit_error: Some(failure),
        ..ScriptedEvaluator::default()
    };
    pump(&mut search, &mut runtime, &ManualClock::new(1));
    assert!(
        matches!(search.outcome().status, ContractSearchStatus::Failed(ContractSearchFailure::Boundary(actual)) if actual == failure)
    );
    assert!(search.outcome().fallback_used);
    assert_eq!(search.outcome().metrics.submissions, 0);
    assert_eq!(search.outcome().metrics.submission_attempts, 1);
    assert_eq!(search.outcome().counters.reservations_released, 1);
    assert_eq!(search.outcome().counters.accepted_backups, 0);
}

#[test]
fn shared_allocator_sequences_continue_across_new_root_and_game() {
    let ids = Arc::new(IdAllocator::new(EPOCH));
    let mut runtime = ScriptedEvaluator::default();
    let clock = ManualClock::new(1);
    let mut first = ContractSearch::new(ongoing_child(), config(Arc::clone(&ids), 1)).unwrap();
    pump(&mut first, &mut runtime, &clock);
    // One active consumer per poll stream: close the old root before the next.
    runtime.submissions[0].cancel_token().cancel();
    pump(&mut first, &mut runtime, &clock);
    assert!(first.is_finished());
    let mut next_config = config(ids, 1);
    next_config.scope.game = GameGeneration(2);
    next_config.scope.root = RootGeneration(2);
    let next_scope = next_config.scope;
    let mut second = ContractSearch::new(ongoing_child(), next_config).unwrap();
    second.pump(&mut runtime, &clock, || next_scope, input_key);
    let first_context = runtime.submissions[0].context();
    let second_context = runtime.submissions[1].context();
    assert_eq!(first_context.request, RequestId::new(EPOCH, 1));
    assert_eq!(second_context.request, RequestId::new(EPOCH, 2));
    assert_eq!(first_context.selection, SelectionId::new(EPOCH, 1));
    assert_eq!(second_context.selection, SelectionId::new(EPOCH, 2));
}

#[test]
fn cache_provenance_is_recorded_separately_from_submission_and_execution_ids() {
    let mut search = new_search(ongoing_child(), 1);
    let mut runtime = ScriptedEvaluator::default();
    let clock = ManualClock::new(1);
    pump(&mut search, &mut runtime, &clock);
    let request = runtime.submissions[0].clone();
    let mut output = completed(&request, &[1.0], [0.0, 1.0, 0.0]);
    output.actual.execution = None;
    output.actual.provenance = CacheProvenance::RawEvalHit {
        source_execution: Some(ExecutionId::new(EPOCH, 777)),
    };
    runtime.results.push_back(EvalResult::Completed(output));
    pump(&mut search, &mut runtime, &clock);
    let metrics = search.outcome().metrics;
    assert_eq!(metrics.submissions, 1);
    assert_eq!(metrics.accepted_outputs, 1);
    assert_eq!(metrics.accepted_execution_ids, 0);
    assert_eq!(metrics.accepted_raw_cache_hits, 1);
    assert_eq!(search.outcome().counters.accepted_backups, 0);
}

#[test]
fn final_submission_guard_closes_admission_after_request_preparation() {
    for (tick, expected) in [
        (80, ContractStopReason::SoftBudget),
        (90, ContractStopReason::AdmissionClosed),
        (100, ContractStopReason::Expired),
    ] {
        let clock = ManualClock::new(1);
        let calls = Arc::new(AtomicU64::new(0));
        let trigger = Arc::new(AtomicU64::new(u64::MAX));
        let authority: Authority = {
            let calls = Arc::clone(&calls);
            let trigger = Arc::clone(&trigger);
            let clock = clock.clone();
            Arc::new(move || {
                if calls.fetch_add(1, Ordering::AcqRel) + 1 == trigger.load(Ordering::Acquire) {
                    clock.set(tick);
                }
                Ok(())
            })
        };
        let mut search = new_search(ongoing_child().with_authority(authority), 1);
        calls.store(0, Ordering::Release);
        // root admission, checked leaf, encoder admission, final submission authority
        trigger.store(4, Ordering::Release);
        let mut runtime = ScriptedEvaluator::default();
        assert!(matches!(
            pump(&mut search, &mut runtime, &clock),
            ContractPumpEvent::Finished
        ));
        assert!(
            matches!(search.outcome().status, ContractSearchStatus::Stopped { reason, .. } if reason == expected)
        );
        assert_eq!(runtime.submissions.len(), 0);
        assert_eq!(runtime.polls, 0);
        assert_eq!(runtime.canceled.len(), 0);
        assert_eq!(search.outcome().metrics.submission_attempts, 0);
        assert_eq!(search.outcome().counters.reservations_released, 1);
        assert!(search.outcome().fallback_used);
    }
}

#[test]
fn resource_ceiling_and_encoder_failure_preserve_fallback_and_typed_failure() {
    let mut settings = config(Arc::new(IdAllocator::new(EPOCH)), 1);
    settings.tree_limits.max_legal_moves = 1;
    let mut search = ContractSearch::new(promotion_position(), settings).unwrap();
    let mut runtime = ScriptedEvaluator::default();
    let clock = ManualClock::new(1);
    pump(&mut search, &mut runtime, &clock);
    assert!(matches!(
        search.outcome().status,
        ContractSearchStatus::Failed(ContractSearchFailure::Tree(
            rz_search::SearchError::EdgeLimit
        ))
    ));
    assert!(search.outcome().fallback_used);
    assert_eq!(search.outcome().counters.reservations_released, 1);
    assert_eq!(runtime.submissions.len(), 0);
    let mut search = new_search(ongoing_child(), 1);
    let failure = error(
        ErrorCode::UnsupportedContract,
        "encoder attestation unavailable",
    );
    search.pump(&mut runtime, &clock, scope, |_, _| Err(failure));
    assert!(
        matches!(search.outcome().status, ContractSearchStatus::Failed(ContractSearchFailure::Boundary(actual)) if actual == failure)
    );
    assert!(search.outcome().fallback_used);
    assert_eq!(search.outcome().counters.reservations_released, 1);
    assert_eq!(runtime.submissions.len(), 0);
}

#[test]
fn accepted_shared_execution_id_is_counted_once_for_distinct_requests() {
    let mut search = new_search(ongoing_child(), 1);
    let mut runtime = ScriptedEvaluator::default();
    let clock = ManualClock::new(1);
    for index in 0..2 {
        pump(&mut search, &mut runtime, &clock);
        let request = runtime.submissions[index].clone();
        let mut output = completed(&request, &[1.0], [0.0, 1.0, 0.0]);
        output.actual.execution = Some(ExecutionId::new(EPOCH, 42));
        runtime.results.push_back(EvalResult::Completed(output));
        pump(&mut search, &mut runtime, &clock);
    }
    assert_eq!(search.outcome().metrics.submissions, 2);
    assert_eq!(search.outcome().metrics.accepted_outputs, 2);
    assert_eq!(search.outcome().metrics.accepted_execution_ids, 1);
    assert_eq!(search.outcome().counters.accepted_backups, 1);
}

#[test]
fn dropping_active_search_revokes_logical_authority_without_physical_cleanup() {
    let mut search = new_search(ongoing_child(), 1);
    let mut runtime = ScriptedEvaluator::default();
    pump(&mut search, &mut runtime, &ManualClock::new(1));
    let request = runtime.submissions[0].clone();
    assert!(!request.cancel_token().is_canceled());
    drop(search);
    assert!(request.cancel_token().is_canceled());
    assert_eq!(runtime.canceled.len(), 0);
    assert_eq!(runtime.polls, 0);
    assert_eq!(request.context().request, RequestId::new(EPOCH, 1));
}

#[test]
fn completed_search_drop_keeps_caller_output_authority() {
    let settings = config(Arc::new(IdAllocator::new(EPOCH)), 0);
    let cancellation = settings.cancellation.clone();
    let mut search = ContractSearch::new(ongoing_child(), settings).unwrap();
    pump(
        &mut search,
        &mut ScriptedEvaluator::default(),
        &ManualClock::new(1),
    );
    assert!(search.is_finished());
    drop(search);
    assert!(!cancellation.is_canceled());
}

#[test]
fn final_submission_guard_rechecks_cancellation_and_current_scope() {
    for cancel_case in [false, true] {
        let calls = Arc::new(AtomicU64::new(0));
        let trigger = Arc::new(AtomicU64::new(u64::MAX));
        let registry_root = Arc::new(AtomicU64::new(1));
        let settings = config(Arc::new(IdAllocator::new(EPOCH)), 1);
        let cancellation = settings.cancellation.clone();
        let authority: Authority = {
            let calls = Arc::clone(&calls);
            let trigger = Arc::clone(&trigger);
            let registry_root = Arc::clone(&registry_root);
            Arc::new(move || {
                if calls.fetch_add(1, Ordering::AcqRel) + 1 == trigger.load(Ordering::Acquire) {
                    if cancel_case {
                        cancellation.cancel();
                    } else {
                        registry_root.store(2, Ordering::Release);
                    }
                }
                Ok(())
            })
        };
        let mut search =
            ContractSearch::new(ongoing_child().with_authority(authority), settings).unwrap();
        calls.store(0, Ordering::Release);
        trigger.store(4, Ordering::Release);
        let mut runtime = ScriptedEvaluator::default();
        search.pump(
            &mut runtime,
            &ManualClock::new(1),
            || {
                let mut current = scope();
                current.root = RootGeneration(registry_root.load(Ordering::Acquire));
                current
            },
            input_key,
        );
        assert!(
            matches!(search.outcome().status, ContractSearchStatus::Stopped { reason, .. }
            if reason == if cancel_case { ContractStopReason::Canceled } else { ContractStopReason::Stale })
        );
        assert_eq!(runtime.submissions.len(), 0);
        assert_eq!(search.outcome().metrics.submission_attempts, 0);
        assert_eq!(search.outcome().counters.reservations_released, 1);
        assert!(search.outcome().fallback_used);
    }
}

#[test]
fn unwind_inside_submit_or_poll_revokes_retained_request_authority() {
    struct PanickingRuntime {
        request: Option<Arc<EvalRequest<FixtureState>>>,
        submit: bool,
    }
    impl Evaluator<FixtureState> for PanickingRuntime {
        fn submit(&mut self, request: Arc<EvalRequest<FixtureState>>) -> Result<(), ContractError> {
            self.request = Some(request);
            if self.submit {
                panic!("injected submit panic");
            }
            Ok(())
        }
        fn poll(&mut self) -> Option<EvalResult> {
            panic!("injected poll panic");
        }
        fn cancel(&mut self, _: RequestId) -> Result<(), ContractError> {
            panic!("Drop must not invoke Runtime");
        }
    }
    for submit in [false, true] {
        let mut runtime = PanickingRuntime {
            request: None,
            submit,
        };
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut search = new_search(ongoing_child(), 1);
            search.pump(&mut runtime, &ManualClock::new(1), scope, input_key);
            search.pump(&mut runtime, &ManualClock::new(1), scope, input_key);
        }));
        assert!(unwound.is_err());
        assert!(
            runtime
                .request
                .as_ref()
                .unwrap()
                .cancel_token()
                .is_canceled()
        );
    }
}

#[cfg(feature = "experimental-batch")]
#[test]
fn parallel_requests_accept_reverse_order_once_and_cancel_all_virtual_paths() {
    for cancel in [false, true] {
        let position = Position::from_nodes(vec![
            ongoing(
                Color::White,
                &[
                    (chess_move(12, 28), 1),
                    (chess_move(11, 27), 2),
                    (chess_move(10, 26), 3),
                ],
            ),
            ongoing(Color::Black, &[(chess_move(52, 36), 4)]),
            ongoing(Color::Black, &[(chess_move(51, 35), 4)]),
            ongoing(Color::Black, &[(chess_move(50, 34), 4)]),
            terminal(Color::White, None),
        ]);
        let mut search = new_search(position, 2);
        search.set_parallelism(2).unwrap();
        let mut runtime = ScriptedEvaluator::default();
        let clock = ManualClock::new(1);
        assert!(matches!(
            pump(&mut search, &mut runtime, &clock),
            ContractPumpEvent::Submitted { .. }
        ));
        runtime.results.push_back(EvalResult::Completed(completed(
            &runtime.submissions[0],
            &[0.5, 0.3, 0.2],
            [0.3, 0.4, 0.3],
        )));
        assert!(matches!(
            pump(&mut search, &mut runtime, &clock),
            ContractPumpEvent::Accepted {
                traversed_edges: 0,
                ..
            }
        ));
        assert!(matches!(
            pump(&mut search, &mut runtime, &clock),
            ContractPumpEvent::Submitted { .. }
        ));
        assert!(matches!(
            pump(&mut search, &mut runtime, &clock),
            ContractPumpEvent::Submitted { .. }
        ));
        assert_eq!(runtime.submissions.len(), 3);
        assert_ne!(
            runtime.submissions[1].context().state,
            runtime.submissions[2].context().state
        );
        assert_eq!(search.outcome().counters.completed_visits, 0);
        assert!(
            search
                .outcome()
                .root_stats
                .iter()
                .all(|(_, s)| s.visits == 0 && s.value_sum == 0.0)
        );
        assert!(search.set_parallelism(3).is_err());
        if cancel {
            clock.set(100);
            assert!(matches!(
                pump(&mut search, &mut runtime, &clock),
                ContractPumpEvent::Finished
            ));
            assert_eq!(runtime.canceled.len(), 2);
            assert_ne!(runtime.canceled[0], runtime.canceled[1]);
            assert_eq!(search.outcome().counters.completed_visits, 0);
            assert_eq!(search.outcome().counters.reservations_released, 3);
        } else {
            let reply = completed(&runtime.submissions[2], &[1.0], [0.2, 0.5, 0.3]);
            runtime
                .results
                .push_back(EvalResult::Completed(reply.clone()));
            assert!(matches!(
                pump(&mut search, &mut runtime, &clock),
                ContractPumpEvent::Accepted {
                    traversed_edges: 1,
                    ..
                }
            ));
            runtime.results.push_back(EvalResult::Completed(reply));
            assert!(matches!(
                pump(&mut search, &mut runtime, &clock),
                ContractPumpEvent::RejectedResult { .. }
            ));
            assert_eq!(search.outcome().counters.completed_visits, 1);
            runtime.results.push_back(EvalResult::Completed(completed(
                &runtime.submissions[1],
                &[1.0],
                [0.4, 0.3, 0.3],
            )));
            assert!(matches!(
                pump(&mut search, &mut runtime, &clock),
                ContractPumpEvent::Accepted {
                    traversed_edges: 1,
                    ..
                }
            ));
            assert!(matches!(
                pump(&mut search, &mut runtime, &clock),
                ContractPumpEvent::Finished
            ));
            assert_eq!(search.outcome().counters.completed_visits, 2);
            assert_eq!(search.outcome().counters.reservations_released, 3);
            assert_eq!(runtime.submissions.len(), 3);
        }
    }
}

#[cfg(feature = "experimental-state-cache")]
#[test]
fn state_cache_zero_tiny_and_eviction_preserve_exact_visit_results() {
    #[derive(Clone)]
    struct Estimated(Position);
    impl ContractPosition for Estimated {
        type State = FixtureState;
        fn snapshot(&self) -> &PositionSnapshot<Self::State> {
            self.0.snapshot()
        }
        fn legal(&self) -> &LegalMoveView {
            self.0.legal()
        }
        fn play(&self, movement: &Move) -> Result<Self, ContractError> {
            self.0.play(movement).map(Self)
        }
        fn validate_authority(&self) -> Result<(), ContractError> {
            self.0.validate_authority()
        }
        fn retained_bytes(&self) -> Option<usize> {
            Some(512)
        }
    }
    let mut golden = None;
    for limits in [
        StateCacheLimits {
            max_entries: 0,
            max_bytes: 0,
        },
        StateCacheLimits {
            max_entries: 2,
            max_bytes: 1,
        },
        StateCacheLimits {
            max_entries: 1,
            max_bytes: 8192,
        },
        StateCacheLimits::default(),
    ] {
        let mut search = ContractSearch::new(
            Estimated(ongoing_child()),
            config(Arc::new(IdAllocator::new(EPOCH)), 3),
        )
        .unwrap();
        search.set_state_cache_limits(limits).unwrap();
        assert!(
            search
                .set_state_cache_limits(StateCacheLimits {
                    max_entries: 1025,
                    max_bytes: 1
                })
                .is_err()
        );
        let mut runtime = ScriptedEvaluator::default();
        let clock = ManualClock::new(1);
        for _ in 0..24 {
            let event = search.pump(&mut runtime, &clock, scope, |p, e| input_key(&p.0, e));
            if let ContractPumpEvent::Submitted { .. } = event {
                let request = runtime.submissions.last().unwrap();
                runtime.results.push_back(EvalResult::Completed(completed(
                    request,
                    &[1.0],
                    [0.4, 0.3, 0.3],
                )));
            }
            let (entries, bytes, _) = search.state_cache_storage();
            assert!(entries <= limits.max_entries);
            assert!(bytes <= limits.max_bytes);
            if search.is_finished() {
                break;
            }
        }
        let outcome = search.outcome();
        assert!(matches!(outcome.status, ContractSearchStatus::Completed));
        assert_eq!(outcome.counters.completed_visits, 3);
        let actual = (outcome.counters, outcome.metrics, outcome.root_stats);
        if let Some(expected) = &golden {
            assert_eq!(&actual, expected);
        } else {
            golden = Some(actual);
        }
        let (_, _, hits) = search.state_cache_storage();
        if limits.max_bytes >= 8192 {
            assert!(hits > 0);
        } else {
            assert_eq!(hits, 0);
        }
    }
}
