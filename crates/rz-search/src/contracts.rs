//! 공통 revision 0.1을 사용하는 B의 비동기 탐색 연결부.
//!
//! Rules와 Encoder가 검증한 상태·합법 수·입력 키를 주입한다. 각 pump는 유한한
//! 상태 전이 한 번만 수행하고 기다림·재시도·thread join을 하지 않는다. Runtime의
//! cancel은 논리 요청만 닫으며 물리 실행과 buffer 수명은 Runtime 소유로 남는다.

use std::collections::HashSet;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

use rz_contracts::*;
use rz_telemetry::source::{SourceJournal, SourceStage};

use crate::contract_time::{ContractClock, ContractDeadlines};
use crate::policy::{EdgeStats, PolicyIdentity, Puct, SelectionPolicy};
use crate::tree::{
    Completion, Leaf, SearchCounters, SearchError, SelectionTicket, Tree, TreeLimits,
};

/// A의 checked Rules adapter. snapshot과 legal은 같은 불변 상태에 귀속되어야 한다.
pub trait ContractPosition: Clone {
    type State: Send + Sync + 'static;

    fn snapshot(&self) -> &PositionSnapshot<Self::State>;
    fn legal(&self) -> &LegalMoveView;
    fn play(&self, chess_move: &Move) -> Result<Self, ContractError>;
    /// 실제 Rules 소유권·state revision·합법 수 의미·이력을 다시 검증한다.
    fn validate_authority(&self) -> Result<(), ContractError>;
    /// Conservative retained-storage charge. Unknown implementations miss the
    /// optional cache rather than pretending an opaque state costs zero bytes.
    fn retained_bytes(&self) -> Option<usize> {
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StateCacheLimits {
    pub max_entries: usize,
    pub max_bytes: usize,
}
impl Default for StateCacheLimits {
    fn default() -> Self {
        Self {
            max_entries: 64,
            max_bytes: 8 * 1024 * 1024,
        }
    }
}

#[cfg(feature = "experimental-state-cache")]
struct CachedState<P> {
    path: Vec<Move>,
    state: P,
    charge: usize,
}

#[cfg(feature = "experimental-state-cache")]
struct StateCache<P> {
    limits: StateCacheLimits,
    entries: std::collections::VecDeque<CachedState<P>>,
    bytes: usize,
    hits: u64,
}

#[cfg(feature = "experimental-state-cache")]
impl<P: ContractPosition> StateCache<P> {
    fn lookup(&mut self, path: &[Move]) -> Option<(usize, P)> {
        let best = self
            .entries
            .iter()
            .filter(|entry| path.starts_with(&entry.path))
            .max_by_key(|entry| entry.path.len())?;
        self.hits = self.hits.saturating_add(1);
        Some((best.path.len(), best.state.clone()))
    }
    fn insert(&mut self, path: &[Move], state: &P) -> Result<(), ContractError> {
        let Some(state_bytes) = state.retained_bytes() else {
            return Ok(());
        };
        let charge = path
            .len()
            .checked_mul(std::mem::size_of::<Move>())
            .and_then(|bytes| bytes.checked_add(state_bytes))
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<CachedState<P>>()));
        let Some(charge) = charge.filter(|&bytes| bytes <= self.limits.max_bytes) else {
            return Ok(());
        };
        if self.limits.max_entries == 0 || path.is_empty() {
            return Ok(());
        }
        if self.entries.iter().any(|entry| entry.path == path) {
            return Ok(());
        }
        while self.entries.len() >= self.limits.max_entries
            || self.bytes > self.limits.max_bytes - charge
        {
            if let Some(old) = self.entries.pop_front() {
                self.bytes -= old.charge;
            }
        }
        let failure = || {
            boundary(
                ErrorCode::ResourceExhausted,
                Stage::Contract,
                "state cache allocation failed",
            )
        };
        let mut owned_path = Vec::new();
        owned_path
            .try_reserve_exact(path.len())
            .map_err(|_| failure())?;
        owned_path.extend_from_slice(path);
        self.entries.try_reserve(1).map_err(|_| failure())?;
        self.entries.push_back(CachedState {
            path: owned_path,
            state: state.clone(),
            charge,
        });
        self.bytes += charge;
        Ok(())
    }
}

/// 프로세스 epoch 전체에서 같은 Arc를 공유한다. root/newgame에서 초기화하지 않는다.
/// RequestId와 SelectionId는 독립 sequence이며 발급 실패 시 wrap/reuse하지 않는다.
pub struct IdAllocator {
    epoch: ProcessEpoch,
    requests: AtomicU64,
    selections: AtomicU64,
}

impl IdAllocator {
    pub fn new(epoch: ProcessEpoch) -> Self {
        Self {
            epoch,
            requests: AtomicU64::new(0),
            selections: AtomicU64::new(0),
        }
    }

    pub fn epoch(&self) -> ProcessEpoch {
        self.epoch
    }

    pub fn request(&self) -> Result<RequestId, ContractError> {
        next_sequence(&self.requests).map(|sequence| RequestId::new(self.epoch, sequence))
    }

    pub fn selection(&self) -> Result<SelectionId, ContractError> {
        next_sequence(&self.selections).map(|sequence| SelectionId::new(self.epoch, sequence))
    }
}

fn next_sequence(sequence: &AtomicU64) -> Result<u64, ContractError> {
    sequence
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            current.checked_add(1)
        })
        .map(|previous| previous + 1)
        .map_err(|_| {
            boundary(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "ID sequence overflow",
            )
        })
}

pub struct ContractSearchConfig {
    pub scope: AcceptanceScope,
    pub model: Arc<ModelDescriptor>,
    pub precision: PrecisionProfile,
    pub compute: ComputeBudget,
    pub bytes: ByteBudget,
    pub policy_tolerance: f64,
    pub wdl_tolerance: f32,
    pub cancellation: CancelToken,
    pub deadlines: ContractDeadlines,
    pub ids: Arc<IdAllocator>,
    /// 0은 평가 제출 없이 검증한 합법 fallback으로 끝나는 명시적 예산이다.
    pub max_simulations: u64,
    pub tree_limits: TreeLimits,
}

#[derive(Clone, Debug)]
pub enum ContractSearchFailure {
    Boundary(ContractError),
    Evaluation(Box<EvalFailure>),
    Tree(SearchError),
    Cleanup {
        primary: Box<Self>,
        runtime: Option<ContractError>,
        tree: Option<SearchError>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContractStopReason {
    Canceled,
    Expired,
    Stale,
    SoftBudget,
    AdmissionClosed,
}

#[derive(Clone, Debug)]
pub enum ContractSearchStatus {
    Running,
    Completed,
    Terminal {
        reason: TerminalReason,
        winner: Option<Color>,
        value: f64,
    },
    Stopped {
        reason: ContractStopReason,
        source: Option<ContractError>,
    },
    Failed(ContractSearchFailure),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ContractSearchMetrics {
    pub submission_attempts: u64,
    /// Runtime이 논리적으로 수락한 제출 수. submit 호출 시도와 구별한다.
    pub submissions: u64,
    pub accepted_outputs: u64,
    /// 서로 다른 execution ID를 포함한 수락 결과 수다. 물리 완료 계측이 아니다.
    pub accepted_execution_ids: u64,
    pub accepted_raw_cache_hits: u64,
    pub diagnostics: u64,
}

#[derive(Clone, Debug)]
pub struct ContractSearchOutcome {
    pub best_move: Option<Move>,
    pub fallback_used: bool,
    pub status: ContractSearchStatus,
    pub counters: SearchCounters,
    pub metrics: ContractSearchMetrics,
    pub root_stats: Vec<(Move, EdgeStats)>,
    pub policy_identity: PolicyIdentity,
}

/// Passive metadata copied only after the final tree acceptance guard commits.
/// It cannot authorize evaluation acceptance, another visit, or another backup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcceptedEvaluation {
    pub context: CompletionContext,
    pub actual: ActualCompute,
}

/// Optional, single-slot observation of an exact Rules terminal after commit.
/// The observer cannot mark nodes solved or change selection/backup authority.
#[derive(Clone, Debug)]
pub struct TerminalBackupObservation {
    pub selection: SelectionId,
    pub path: Vec<Move>,
    pub classification: PlayStatus,
    pub leaf_side: Color,
    pub leaf_value: f64,
}

#[derive(Clone, Debug)]
pub enum ContractPumpEvent {
    Submitted {
        request: RequestId,
        selection: SelectionId,
    },
    Waiting,
    Accepted {
        request: Option<RequestId>,
        selection: SelectionId,
        traversed_edges: usize,
        /// Exact Rules terminals have no evaluator output and leave this empty.
        evaluation: Option<Box<AcceptedEvaluation>>,
    },
    /// 다른 요청 또는 잘못 echo된 context는 현재 reservation을 소비하지 않는다.
    Diagnostic(ContractError),
    /// 원래 결과·실패 원인·recovery를 보존한다. Dispatcher는 이 소유한 결과를
    /// 실제 subscriber에게 전달하거나 이전 요청의 진단으로 기록할 수 있다.
    RejectedResult {
        error: ContractError,
        result: Box<EvalResult>,
    },
    Finished,
}

struct Pending<P: ContractPosition> {
    ticket: SelectionTicket,
    selection: SelectionId,
    request: Arc<EvalRequest<P::State>>,
    leaf: P,
    root_leaf: bool,
}

pub struct ContractSearch<P: ContractPosition, S: SelectionPolicy = Puct> {
    root: P,
    config: ContractSearchConfig,
    tree: Tree<Move, S>,
    fallback: Option<Move>,
    /// Indices into the unchanged full Rules/NN root legal array.
    root_projection: Option<Vec<usize>>,
    status: ContractSearchStatus,
    pending: Option<Pending<P>>,
    #[cfg(feature = "experimental-batch")]
    other_pending: Vec<Pending<P>>,
    #[cfg(feature = "experimental-batch")]
    max_pending: usize,
    simulations: u64,
    metrics: ContractSearchMetrics,
    accepted_executions: HashSet<ExecutionId>,
    #[cfg(feature = "experimental-state-cache")]
    state_cache: StateCache<P>,
    source_trace: Option<SourceJournal<CompletionContext>>,
    observe_terminals: bool,
    terminal_observation: Option<TerminalBackupObservation>,
}

impl<P: ContractPosition, S: SelectionPolicy> Drop for ContractSearch<P, S> {
    fn drop(&mut self) {
        // Covers unwind while submit/poll owns a temporarily detached Pending as well.
        // This revokes logical acceptance only; Runtime retains every physical lease.
        if matches!(self.status, ContractSearchStatus::Running) || self.pending_count() != 0 {
            self.config.cancellation.cancel();
        }
    }
}

impl<P: ContractPosition> ContractSearch<P, Puct> {
    pub fn new(root: P, config: ContractSearchConfig) -> Result<Self, ContractError> {
        Self::with_policy(root, config, Puct::default())
    }
}

impl<P: ContractPosition, S: SelectionPolicy> ContractSearch<P, S> {
    pub fn with_policy(
        root: P,
        config: ContractSearchConfig,
        policy: S,
    ) -> Result<Self, ContractError> {
        validate_config(&config)?;
        validate_position(&root)?;
        let (fallback, status) = match terminal_status(&root) {
            Some(status) => (None, status),
            None => {
                let chess_move = *root.legal().moves().first().ok_or_else(|| {
                    boundary(
                        ErrorCode::InvalidInput,
                        Stage::Contract,
                        "ongoing Rules state has no legal moves",
                    )
                })?;
                // Confirm fallback through the immutable checked Rules transition as well.
                root.play(&chess_move)?.validate_authority()?;
                (Some(chess_move), ContractSearchStatus::Running)
            }
        };
        let tree = Tree::new(policy, config.tree_limits).map_err(tree_boundary)?;
        Ok(Self {
            root,
            config,
            tree,
            fallback,
            root_projection: None,
            status,
            pending: None,
            #[cfg(feature = "experimental-batch")]
            other_pending: Vec::new(),
            #[cfg(feature = "experimental-batch")]
            max_pending: 1,
            simulations: 0,
            metrics: ContractSearchMetrics::default(),
            accepted_executions: HashSet::new(),
            #[cfg(feature = "experimental-state-cache")]
            state_cache: StateCache {
                limits: StateCacheLimits::default(),
                entries: std::collections::VecDeque::new(),
                bytes: 0,
                hits: 0,
            },
            source_trace: None,
            observe_terminals: false,
            terminal_observation: None,
        })
    }

    pub fn is_finished(&self) -> bool {
        !matches!(self.status, ContractSearchStatus::Running)
    }

    pub fn pending_requests(&self) -> usize {
        self.pending_count()
    }
    /// Optional passive source timestamps. Journal loss never changes acceptance.
    pub fn set_source_trace(&mut self, trace: Option<SourceJournal<CompletionContext>>) {
        self.source_trace = trace;
    }

    pub fn observe_terminal_backups(&mut self, enabled: bool) {
        self.observe_terminals = enabled;
        self.terminal_observation = None;
    }
    pub fn set_final_move_policy(
        &mut self,
        policy: crate::tree::FinalMovePolicy,
    ) -> Result<(), ContractError> {
        self.tree
            .set_final_move_policy(policy)
            .map_err(tree_boundary)
    }

    /// Seal root choices before any selection. Requests, input identity and raw
    /// NN policy retain the complete Rules legal order. Only root tree edges
    /// are projected and renormalized. Zero mass on the allowed subset uses a
    /// uniform prior on that subset; it never restores an excluded root move.
    pub fn set_root_moves(&mut self, requested: &[Move]) -> Result<(), ContractError> {
        if self.tree.counters().selections != 0
            || self.pending_count() != 0
            || self.root_projection.is_some()
        {
            return Err(boundary(
                ErrorCode::UnsupportedContract,
                Stage::Admission,
                "root restriction is immutable and must precede first selection",
            ));
        }
        if requested.is_empty() || requested.len() > self.config.tree_limits.max_legal_moves {
            return Err(boundary(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "root restriction must be nonempty and within the tree move bound",
            ));
        }
        let requested_set: HashSet<_> = requested.iter().copied().collect();
        if requested_set.len() != requested.len()
            || requested
                .iter()
                .any(|movement| !self.root.legal().moves().contains(movement))
        {
            return Err(boundary(
                ErrorCode::InvalidInput,
                Stage::Admission,
                "root restriction contains a duplicate or nonlegal move",
            ));
        }
        let mut projection = Vec::new();
        projection.try_reserve_exact(requested.len()).map_err(|_| {
            boundary(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "root restriction allocation failed",
            )
        })?;
        projection.extend(
            self.root
                .legal()
                .moves()
                .iter()
                .enumerate()
                .filter_map(|(index, movement)| requested_set.contains(movement).then_some(index)),
        );
        let fallback = self.root.legal().moves()[projection[0]];
        self.root.play(&fallback)?.validate_authority()?;
        if matches!(self.status, ContractSearchStatus::Running) {
            self.fallback = Some(fallback);
        }
        self.root_projection = Some(projection);
        Ok(())
    }

    pub fn take_terminal_observation(&mut self) -> Option<TerminalBackupObservation> {
        self.terminal_observation.take()
    }

    pub fn pending_request(&self) -> Option<RequestId> {
        self.pending
            .as_ref()
            .map(|pending| pending.request.context().request)
    }

    pub fn outcome(&self) -> ContractSearchOutcome {
        let started = self.source_trace.as_ref().map(|_| Instant::now());
        let root_stats = self.tree.root_stats();
        let visited = root_stats.iter().any(|(_, edge)| edge.visits > 0);
        let best = if visited {
            self.tree.best_move().copied()
        } else {
            None
        };
        let outcome = ContractSearchOutcome {
            best_move: best.or(self.fallback),
            fallback_used: best.is_none() && self.fallback.is_some(),
            status: self.status.clone(),
            counters: self.tree.counters(),
            metrics: self.metrics,
            root_stats,
            policy_identity: self.tree.policy_identity(),
        };
        if let (Some(trace), Some(start)) = (&self.source_trace, started) {
            trace.record(
                None,
                SourceStage::FinalSelection,
                start,
                Instant::now(),
                true,
            );
        }
        outcome
    }

    /// Allocation-free progress query, with the same zero-visit fallback rule.
    pub fn best_move(&self) -> Option<Move> {
        let best = if self.tree.has_root_visits() {
            self.tree.best_move().copied()
        } else {
            None
        };
        best.or(self.fallback)
    }

    pub fn counters(&self) -> SearchCounters {
        self.tree.counters()
    }

    pub fn set_state_cache_limits(
        &mut self,
        limits: StateCacheLimits,
    ) -> Result<(), ContractError> {
        if limits.max_entries > 1024 || limits.max_bytes > 64 * 1024 * 1024 {
            return Err(boundary(
                ErrorCode::ResourceExhausted,
                Stage::Contract,
                "state cache limit exceeds hard ceiling",
            ));
        }
        #[cfg(feature = "experimental-state-cache")]
        {
            self.state_cache = StateCache {
                limits,
                entries: std::collections::VecDeque::new(),
                bytes: 0,
                hits: 0,
            };
            Ok(())
        }
        #[cfg(not(feature = "experimental-state-cache"))]
        {
            if limits.max_entries != 0 && limits.max_bytes != 0 {
                Err(boundary(
                    ErrorCode::UnsupportedContract,
                    Stage::Contract,
                    "state cache experiment is disabled",
                ))
            } else {
                Ok(())
            }
        }
    }

    pub fn state_cache_storage(&self) -> (usize, usize, u64) {
        #[cfg(feature = "experimental-state-cache")]
        {
            (
                self.state_cache.entries.len(),
                self.state_cache.bytes,
                self.state_cache.hits,
            )
        }
        #[cfg(not(feature = "experimental-state-cache"))]
        {
            (0, 0, 0)
        }
    }

    fn pending_count(&self) -> usize {
        let count = usize::from(self.pending.is_some());
        #[cfg(feature = "experimental-batch")]
        let count = count + self.other_pending.len();
        count
    }
    #[cfg(feature = "experimental-batch")]
    pub fn set_parallelism(&mut self, max_pending: usize) -> Result<(), ContractError> {
        if self.tree.counters().selections != 0 || self.pending_count() != 0 {
            return Err(boundary(
                ErrorCode::UnsupportedContract,
                Stage::Contract,
                "parallelism is fixed before first selection",
            ));
        }
        self.tree
            .set_parallelism(max_pending)
            .map_err(tree_boundary)?;
        self.other_pending.try_reserve(max_pending).map_err(|_| {
            boundary(
                ErrorCode::ResourceExhausted,
                Stage::Contract,
                "pending allocation failed",
            )
        })?;
        self.max_pending = max_pending;
        Ok(())
    }
    #[cfg(feature = "experimental-batch")]
    fn cancel_pending<R: Evaluator<P::State>>(&mut self, runtime: &mut R) {
        while let Some(pending) = self.pending.take().or_else(|| self.other_pending.pop()) {
            if let Err(error) = runtime.cancel(pending.request.context().request) {
                self.record_runtime_cleanup(error);
            }
            if let Err(error) = self.tree.cancel(&pending.ticket) {
                self.status = ContractSearchStatus::Failed(ContractSearchFailure::Tree(error));
            }
        }
    }
    pub fn pump<R, C, L, K>(
        &mut self,
        runtime: &mut R,
        clock: &C,
        live_scope: L,
        input_key: K,
    ) -> ContractPumpEvent
    where
        R: Evaluator<P::State>,
        C: ContractClock,
        L: FnMut() -> AcceptanceScope,
        K: FnMut(&P, &EncodingDescriptor) -> Result<EvalInputKey, ContractError>,
    {
        let event = self.pump_inner(runtime, clock, live_scope, input_key);
        #[cfg(feature = "experimental-batch")]
        if self.is_finished() {
            self.cancel_pending(runtime);
        }
        event
    }

    /// 기다리지 않는 논리 상태 전이. Runtime의 각 메서드도 blocking하면 안 된다.
    /// 이 탐색은 해당 poll stream의 단일 활성 consumer다. 이전 탐색을 논리적으로
    /// 닫은 뒤 stream을 재사용한다. 동시 탐색은 Runtime dispatcher가 RequestId별로
    /// 분리한 evaluator stream을 제공해야 하며 전역 poll stream을 함께 읽으면 안 된다.
    fn pump_inner<R, C, L, K>(
        &mut self,
        runtime: &mut R,
        clock: &C,
        mut live_scope: L,
        mut input_key: K,
    ) -> ContractPumpEvent
    where
        R: Evaluator<P::State>,
        C: ContractClock,
        L: FnMut() -> AcceptanceScope,
        K: FnMut(&P, &EncodingDescriptor) -> Result<EvalInputKey, ContractError>,
    {
        if self.is_finished() {
            return ContractPumpEvent::Finished;
        }
        #[cfg(feature = "experimental-batch")]
        if self.pending.is_none() {
            self.pending = self.other_pending.pop();
        }
        if self.pending.is_some() {
            let event = self.poll_pending(runtime, clock, &mut live_scope);
            #[cfg(feature = "experimental-batch")]
            if matches!(event, ContractPumpEvent::Waiting)
                && self.pending_count() < self.max_pending
            {
                let now = match clock.now() {
                    Ok(now) => now,
                    Err(_) => return event,
                };
                if now >= self.config.deadlines.admission.at
                    || now >= self.config.deadlines.soft.at
                    || self.simulations.saturating_add(self.pending_count() as u64)
                        >= self.config.max_simulations
                {
                    return event;
                }
                // Continue to select another independent leaf; its virtual path
                // reserves capacity but does not create a completed visit.
            } else {
                return event;
            }
            #[cfg(not(feature = "experimental-batch"))]
            return event;
        }
        if let Err(error) = live_acceptance(&self.config, clock, &mut live_scope, &self.root) {
            self.status = status_for_boundary(error);
            return ContractPumpEvent::Finished;
        }
        let now = match clock.now() {
            Ok(now) => now,
            Err(error) => {
                self.status = ContractSearchStatus::Failed(ContractSearchFailure::Boundary(error));
                return ContractPumpEvent::Finished;
            }
        };
        if now >= self.config.deadlines.admission.at {
            self.status = ContractSearchStatus::Stopped {
                reason: ContractStopReason::AdmissionClosed,
                source: None,
            };
            return ContractPumpEvent::Finished;
        }
        if now >= self.config.deadlines.soft.at {
            self.status = ContractSearchStatus::Stopped {
                reason: ContractStopReason::SoftBudget,
                source: None,
            };
            return ContractPumpEvent::Finished;
        }
        if self.simulations >= self.config.max_simulations {
            self.status = ContractSearchStatus::Completed;
            return ContractPumpEvent::Finished;
        }
        let preparation_started = self.source_trace.as_ref().map(|_| Instant::now());
        let selection = match self.tree.begin_selection(Instant::now()) {
            Ok(selection) => selection,
            #[cfg(feature = "experimental-batch")]
            Err(SearchError::Busy) if self.pending_count() != 0 => {
                return ContractPumpEvent::Waiting;
            }
            Err(error) => {
                self.status = ContractSearchStatus::Failed(ContractSearchFailure::Tree(error));
                return ContractPumpEvent::Finished;
            }
        };
        let selection_id = match self.config.ids.selection() {
            Ok(id) => id,
            Err(error) => {
                return self.fail_ticket(&selection.ticket, ContractSearchFailure::Boundary(error));
            }
        };
        let replay_started = self.source_trace.as_ref().map(|_| Instant::now());
        #[cfg(feature = "experimental-state-cache")]
        let (replayed, mut leaf) = self
            .state_cache
            .lookup(&selection.moves)
            .unwrap_or_else(|| (0, self.root.clone()));
        #[cfg(not(feature = "experimental-state-cache"))]
        let (replayed, mut leaf) = (0, self.root.clone());
        for chess_move in &selection.moves[replayed..] {
            leaf = match leaf.play(chess_move) {
                Ok(next) => next,
                Err(error) => {
                    return self
                        .fail_ticket(&selection.ticket, ContractSearchFailure::Boundary(error));
                }
            };
        }
        let replay_finished = self.source_trace.as_ref().map(|_| Instant::now());
        if let Err(error) = validate_position(&leaf) {
            return self.fail_ticket(&selection.ticket, ContractSearchFailure::Boundary(error));
        }
        #[cfg(feature = "experimental-state-cache")]
        if let Err(error) = self.state_cache.insert(&selection.moves, &leaf) {
            return self.fail_ticket(&selection.ticket, ContractSearchFailure::Boundary(error));
        }
        let terminal = terminal_utility(&leaf);
        let validation_finished = self.source_trace.as_ref().map(|_| Instant::now());
        if let Leaf::Terminal(expected) = selection.leaf {
            if terminal != Some(expected) {
                return self.fail_ticket(
                    &selection.ticket,
                    ContractSearchFailure::Boundary(boundary(
                        ErrorCode::IdentityMismatch,
                        Stage::Backup,
                        "Rules terminal classification changed",
                    )),
                );
            }
        }
        if let Some(value) = terminal {
            if let (Some(trace), Some(start), Some(replay), Some(replayed), Some(validated)) = (
                &self.source_trace,
                preparation_started,
                replay_started,
                replay_finished,
                validation_finished,
            ) {
                trace.record(None, SourceStage::SearchPreparation, start, validated, true);
                trace.record(None, SourceStage::SearchSelection, start, replay, true);
                trace.record(None, SourceStage::StateReplay, replay, replayed, true);
                trace.record(
                    None,
                    SourceStage::LegalValidation,
                    replayed,
                    validated,
                    true,
                );
            }
            let backup_started = self.source_trace.as_ref().map(|_| Instant::now());
            let mut guard_error = None;
            let completion = self.tree.accept_terminal_with_guard(
                &selection.ticket,
                value,
                Instant::now(),
                || match live_acceptance(&self.config, clock, &mut live_scope, &leaf) {
                    Ok(()) => true,
                    Err(error) => {
                        guard_error = Some(error);
                        false
                    }
                },
            );
            let event = self.commit_result(
                completion,
                &selection.ticket,
                None,
                selection_id,
                guard_error,
            );
            let accepted = matches!(event, ContractPumpEvent::Accepted { .. });
            if let (Some(trace), Some(start)) = (&self.source_trace, backup_started) {
                trace.record(
                    None,
                    SourceStage::TerminalBackup,
                    start,
                    Instant::now(),
                    accepted,
                );
            }
            if accepted && self.observe_terminals {
                self.terminal_observation = Some(TerminalBackupObservation {
                    selection: selection_id,
                    path: selection.moves,
                    classification: leaf.snapshot().classification().play_status,
                    leaf_side: leaf.snapshot().side_to_move(),
                    leaf_value: value,
                });
            }
            return event;
        }
        if leaf.legal().moves().len() > self.config.tree_limits.max_legal_moves {
            return self.fail_ticket(
                &selection.ticket,
                ContractSearchFailure::Tree(SearchError::EdgeLimit),
            );
        }
        let input_started = self.source_trace.as_ref().map(|_| Instant::now());
        let input = match input_key(&leaf, self.config.model.encoding()) {
            Ok(input) => input,
            Err(error) => {
                return self.fail_ticket(&selection.ticket, ContractSearchFailure::Boundary(error));
            }
        };
        let input_finished = self.source_trace.as_ref().map(|_| Instant::now());
        if let Err(error) = live_acceptance(&self.config, clock, &mut live_scope, &leaf) {
            self.release_ticket(&selection.ticket, status_for_boundary(error));
            return ContractPumpEvent::Finished;
        }
        let now = match clock.now() {
            Ok(now) => now,
            Err(error) => {
                return self.fail_ticket(&selection.ticket, ContractSearchFailure::Boundary(error));
            }
        };
        if now >= self.config.deadlines.admission.at || now >= self.config.deadlines.soft.at {
            let reason = if now >= self.config.deadlines.admission.at {
                ContractStopReason::AdmissionClosed
            } else {
                ContractStopReason::SoftBudget
            };
            self.release_ticket(
                &selection.ticket,
                ContractSearchStatus::Stopped {
                    reason,
                    source: None,
                },
            );
            return ContractPumpEvent::Finished;
        }
        let request_id = match self.config.ids.request() {
            Ok(id) => id,
            Err(error) => {
                return self.fail_ticket(&selection.ticket, ContractSearchFailure::Boundary(error));
            }
        };
        let context = EvalContext {
            revision: CONTRACT_REVISION,
            request: request_id,
            selection: selection_id,
            game: self.config.scope.game,
            root: self.config.scope.root,
            state: leaf.snapshot().identity(),
            legal_order: leaf.legal().order(),
            input,
            model: self.config.model.handle(),
            encoding: self.config.model.encoding().handle,
            precision: self.config.precision,
            compute: self.config.compute,
            backend: self.config.scope.backend,
        };
        let request = match EvalRequest::try_new(
            context,
            leaf.snapshot().clone(),
            leaf.legal().clone(),
            Arc::clone(&self.config.model),
            self.config.deadlines.hard,
            self.config.cancellation.clone(),
            self.config.bytes,
        ) {
            Ok(request) => Arc::new(request),
            Err(error) => {
                return self.fail_ticket(&selection.ticket, ContractSearchFailure::Boundary(error));
            }
        };
        let submissions = match self.metrics.submissions.checked_add(1) {
            Some(count) => count,
            None => {
                return self.fail_ticket(
                    &selection.ticket,
                    ContractSearchFailure::Tree(SearchError::CounterOverflow),
                );
            }
        };
        let attempts = match self.metrics.submission_attempts.checked_add(1) {
            Some(count) => count,
            None => {
                return self.fail_ticket(
                    &selection.ticket,
                    ContractSearchFailure::Tree(SearchError::CounterOverflow),
                );
            }
        };
        // Request construction and attestation are part of the move's own budget.
        // Read live authority and all admission boundaries after that preparation.
        let final_admission = leaf.validate_authority().and_then(|()| {
            let scope = live_scope();
            let now = clock.now()?;
            request.validate_acceptance(scope, clock.domain(), now)?;
            Ok(now)
        });
        let final_now = match final_admission {
            Ok(now) => now,
            Err(error) => {
                self.release_ticket(&selection.ticket, status_for_boundary(error));
                return ContractPumpEvent::Finished;
            }
        };
        if final_now >= self.config.deadlines.admission.at
            || final_now >= self.config.deadlines.soft.at
        {
            let reason = if final_now >= self.config.deadlines.admission.at {
                ContractStopReason::AdmissionClosed
            } else {
                ContractStopReason::SoftBudget
            };
            self.release_ticket(
                &selection.ticket,
                ContractSearchStatus::Stopped {
                    reason,
                    source: None,
                },
            );
            return ContractPumpEvent::Finished;
        }
        self.metrics.submission_attempts = attempts;
        if let (Some(trace), Some(start)) = (&self.source_trace, preparation_started) {
            let key = Some(CompletionContext {
                request: context,
                execution: None,
            });
            if let (
                Some(replay),
                Some(replayed),
                Some(validated),
                Some(input_start),
                Some(input_end),
            ) = (
                replay_started,
                replay_finished,
                validation_finished,
                input_started,
                input_finished,
            ) {
                trace.record(key, SourceStage::SearchSelection, start, replay, true);
                trace.record(key, SourceStage::StateReplay, replay, replayed, true);
                trace.record(key, SourceStage::LegalValidation, replayed, validated, true);
                trace.record(key, SourceStage::InputKey, input_start, input_end, true);
            }
            trace.record(
                Some(CompletionContext {
                    request: context,
                    execution: None,
                }),
                SourceStage::SearchPreparation,
                start,
                Instant::now(),
                true,
            );
        }
        if let Err(error) = runtime.submit(Arc::clone(&request)) {
            return self.fail_ticket(&selection.ticket, ContractSearchFailure::Boundary(error));
        }
        self.metrics.submissions = submissions;
        #[cfg(feature = "experimental-batch")]
        if let Some(previous) = self.pending.take() {
            self.other_pending.push(previous);
        }
        self.pending = Some(Pending {
            ticket: selection.ticket,
            selection: selection_id,
            request,
            leaf,
            root_leaf: selection.moves.is_empty(),
        });
        ContractPumpEvent::Submitted {
            request: request_id,
            selection: selection_id,
        }
    }

    fn poll_pending<R, C, L>(
        &mut self,
        runtime: &mut R,
        clock: &C,
        live_scope: &mut L,
    ) -> ContractPumpEvent
    where
        R: Evaluator<P::State>,
        C: ContractClock,
        L: FnMut() -> AcceptanceScope,
    {
        #[allow(unused_mut)]
        let mut pending = self
            .pending
            .take()
            .expect("pump checks pending before polling");
        if let Err(error) = live_acceptance(&self.config, clock, live_scope, &pending.leaf) {
            let cancel_error = runtime.cancel(pending.request.context().request).err();
            self.release_ticket(&pending.ticket, status_for_boundary(error));
            if let Some(cleanup) = cancel_error {
                self.record_runtime_cleanup(cleanup);
            }
            return ContractPumpEvent::Finished;
        }
        let Some(result) = runtime.poll() else {
            self.pending = Some(pending);
            return ContractPumpEvent::Waiting;
        };
        let context = match &result {
            EvalResult::Completed(output) => output.context,
            EvalResult::Canceled(completion)
            | EvalResult::Expired(completion)
            | EvalResult::Stale(completion) => completion.request,
            EvalResult::Failed(failure) => failure.context.request,
        };
        #[cfg(feature = "experimental-batch")]
        if context != pending.request.context() {
            if let Some(index) = self
                .other_pending
                .iter()
                .position(|p| p.request.context() == context)
            {
                let matched = self.other_pending.swap_remove(index);
                self.other_pending.push(pending);
                pending = matched;
            }
        }
        if context != pending.request.context() {
            self.pending = Some(pending);
            return self.rejected_result(
                runtime,
                boundary(
                    ErrorCode::IdentityMismatch,
                    Stage::Output,
                    "foreign or malformed response context; active request remains pending",
                ),
                result,
            );
        }
        match result {
            EvalResult::Completed(output) => self.accept_output(pending, output, clock, live_scope),
            EvalResult::Failed(failure) => {
                self.release_ticket(
                    &pending.ticket,
                    ContractSearchStatus::Failed(ContractSearchFailure::Evaluation(Box::new(
                        failure,
                    ))),
                );
                ContractPumpEvent::Finished
            }
            EvalResult::Canceled(_) | EvalResult::Expired(_) | EvalResult::Stale(_) => {
                let reason = match result {
                    EvalResult::Canceled(_) => ContractStopReason::Canceled,
                    EvalResult::Expired(_) => ContractStopReason::Expired,
                    _ => ContractStopReason::Stale,
                };
                self.release_ticket(
                    &pending.ticket,
                    ContractSearchStatus::Stopped {
                        reason,
                        source: None,
                    },
                );
                ContractPumpEvent::Finished
            }
        }
    }

    fn accept_output<C, L>(
        &mut self,
        pending: Pending<P>,
        output: EvalOutput,
        clock: &C,
        live_scope: &mut L,
    ) -> ContractPumpEvent
    where
        C: ContractClock,
        L: FnMut() -> AcceptanceScope,
    {
        let validation = (|| {
            output.validate_for(&pending.request, live_scope(), clock.domain(), clock.now()?)?;
            LegalPolicy::try_new(
                output.policy.probabilities().to_vec(),
                self.config.policy_tolerance,
            )?;
            let [win, draw, loss] = output.wdl.probabilities();
            Wdl::try_new(win, draw, loss, self.config.wdl_tolerance)?;
            validate_position(&pending.leaf)
        })();
        if let Err(error) = validation {
            self.release_ticket(&pending.ticket, status_for_boundary(error));
            return ContractPumpEvent::Finished;
        }
        let prepared_metrics = match self.prepare_metrics(&output) {
            Ok(metrics) => metrics,
            Err(error) => {
                return self.fail_ticket(&pending.ticket, ContractSearchFailure::Tree(error));
            }
        };
        // Explicit common-contract conversion; validated raw heads remain in output.
        let mut priors = output.policy.normalized();
        let moves =
            if let Some(projection) = self.root_projection.as_ref().filter(|_| pending.root_leaf) {
                priors = projection.iter().map(|&index| priors[index]).collect();
                let mass: f64 = priors.iter().sum();
                if mass > 0.0 {
                    priors.iter_mut().for_each(|prior| *prior /= mass);
                } else {
                    let uniform = 1.0 / priors.len() as f64;
                    priors.fill(uniform);
                }
                projection
                    .iter()
                    .map(|&index| pending.request.legal().moves()[index])
                    .collect()
            } else {
                pending.request.legal().moves().to_vec()
            };
        let wdl = output.wdl.normalized();
        let value = wdl[0] - wdl[2];
        let mut guard_error = None;
        let backup_started = self.source_trace.as_ref().map(|_| Instant::now());
        let completion = self.tree.accept_evaluation_with_guard(
            &pending.ticket,
            moves,
            &priors,
            value,
            Instant::now(),
            || {
                // Shape/head preparation finished. Rules authority can be expensive; obtain
                // the live scope and common tick again after it, immediately before commit.
                let valid = pending.leaf.validate_authority().and_then(|()| {
                    let scope = live_scope();
                    let now = clock.now()?;
                    pending
                        .request
                        .validate_acceptance(scope, clock.domain(), now)
                });
                match valid {
                    Ok(()) => true,
                    Err(error) => {
                        guard_error = Some(error);
                        false
                    }
                }
            },
        );
        if let (Some(trace), Some(start)) = (&self.source_trace, backup_started) {
            trace.record(
                Some(CompletionContext {
                    request: output.context,
                    execution: output.actual.execution,
                }),
                SourceStage::SearchBackup,
                start,
                Instant::now(),
                matches!(completion, Ok(Completion::Accepted { .. })),
            );
        }
        let mut event = self.commit_result(
            completion,
            &pending.ticket,
            Some(pending.request.context().request),
            pending.selection,
            guard_error,
        );
        if matches!(event, ContractPumpEvent::Accepted { .. }) {
            self.metrics = prepared_metrics;
            if let Some(execution) = output.actual.execution {
                self.accepted_executions.insert(execution);
            }
            if let ContractPumpEvent::Accepted { evaluation, .. } = &mut event {
                *evaluation = Some(Box::new(AcceptedEvaluation {
                    context: CompletionContext {
                        request: output.context,
                        execution: output.actual.execution,
                    },
                    actual: output.actual,
                }));
            }
        }
        event
    }

    fn prepare_metrics(
        &mut self,
        output: &EvalOutput,
    ) -> Result<ContractSearchMetrics, SearchError> {
        let mut metrics = self.metrics;
        metrics.accepted_outputs = metrics
            .accepted_outputs
            .checked_add(1)
            .ok_or(SearchError::CounterOverflow)?;
        if let Some(execution) = output.actual.execution {
            if !self.accepted_executions.contains(&execution) {
                metrics.accepted_execution_ids = metrics
                    .accepted_execution_ids
                    .checked_add(1)
                    .ok_or(SearchError::CounterOverflow)?;
                self.accepted_executions
                    .try_reserve(1)
                    .map_err(|_| SearchError::AllocationFailed)?;
            }
        }
        if matches!(output.actual.provenance, CacheProvenance::RawEvalHit { .. }) {
            metrics.accepted_raw_cache_hits = metrics
                .accepted_raw_cache_hits
                .checked_add(1)
                .ok_or(SearchError::CounterOverflow)?;
        }
        Ok(metrics)
    }

    fn commit_result(
        &mut self,
        completion: Result<Completion, SearchError>,
        ticket: &SelectionTicket,
        request: Option<RequestId>,
        selection: SelectionId,
        guard_error: Option<ContractError>,
    ) -> ContractPumpEvent {
        match completion {
            Ok(Completion::Accepted { traversed_edges }) => {
                if traversed_edges > 0 {
                    self.simulations = match self.simulations.checked_add(1) {
                        Some(count) => count,
                        None => {
                            self.status = ContractSearchStatus::Failed(
                                ContractSearchFailure::Tree(SearchError::CounterOverflow),
                            );
                            return ContractPumpEvent::Finished;
                        }
                    };
                }
                ContractPumpEvent::Accepted {
                    request,
                    selection,
                    traversed_edges,
                    evaluation: None,
                }
            }
            Ok(Completion::Rejected(_)) => {
                self.status = guard_error.map(status_for_boundary).unwrap_or_else(|| {
                    ContractSearchStatus::Failed(ContractSearchFailure::Boundary(boundary(
                        ErrorCode::IdentityMismatch,
                        Stage::Backup,
                        "tree rejected live selection without matching authority error",
                    )))
                });
                ContractPumpEvent::Finished
            }
            Err(error) => self.fail_ticket(ticket, ContractSearchFailure::Tree(error)),
        }
    }

    fn rejected_result<R: Evaluator<P::State>>(
        &mut self,
        runtime: &mut R,
        error: ContractError,
        result: EvalResult,
    ) -> ContractPumpEvent {
        match self.metrics.diagnostics.checked_add(1) {
            Some(count) => self.metrics.diagnostics = count,
            None => {
                let pending = self
                    .pending
                    .take()
                    .expect("foreign result keeps active request");
                let cleanup = runtime.cancel(pending.request.context().request).err();
                self.release_ticket(
                    &pending.ticket,
                    ContractSearchStatus::Failed(ContractSearchFailure::Tree(
                        SearchError::CounterOverflow,
                    )),
                );
                if let Some(error) = cleanup {
                    self.record_runtime_cleanup(error);
                }
            }
        }
        ContractPumpEvent::RejectedResult {
            error,
            result: Box::new(result),
        }
    }

    fn fail_ticket(
        &mut self,
        ticket: &SelectionTicket,
        failure: ContractSearchFailure,
    ) -> ContractPumpEvent {
        self.release_ticket(ticket, ContractSearchStatus::Failed(failure));
        ContractPumpEvent::Finished
    }

    fn release_ticket(&mut self, ticket: &SelectionTicket, status: ContractSearchStatus) {
        self.status = match self.tree.cancel(ticket) {
            Ok(_) => status,
            Err(error) => ContractSearchStatus::Failed(ContractSearchFailure::Cleanup {
                primary: Box::new(failure_from_status(status)),
                runtime: None,
                tree: Some(error),
            }),
        };
    }

    fn record_runtime_cleanup(&mut self, error: ContractError) {
        let primary = failure_from_status(self.status.clone());
        self.status = ContractSearchStatus::Failed(ContractSearchFailure::Cleanup {
            primary: Box::new(primary),
            runtime: Some(error),
            tree: None,
        });
    }
}

fn validate_config(config: &ContractSearchConfig) -> Result<(), ContractError> {
    config.deadlines.validate_metadata()?;
    if config.ids.epoch() != config.deadlines.hard.clock.0 {
        return Err(boundary(
            ErrorCode::IdentityMismatch,
            Stage::Contract,
            "allocator and deadline epochs differ",
        ));
    }
    if config.scope.model != config.model.handle()
        || config.scope.encoding != config.model.encoding().handle
    {
        return Err(boundary(
            ErrorCode::IdentityMismatch,
            Stage::Contract,
            "initial scope and model/encoding differ",
        ));
    }
    if !config.policy_tolerance.is_finite()
        || !(0.0..=0.01).contains(&config.policy_tolerance)
        || !config.wdl_tolerance.is_finite()
        || !(0.0..=0.01).contains(&config.wdl_tolerance)
    {
        return Err(boundary(
            ErrorCode::InvalidInput,
            Stage::Contract,
            "invalid declared raw-head tolerance",
        ));
    }
    config.compute.validate()?;
    if !config.model.supports(config.precision)
        || config.compute.max_steps > config.model.full_steps()
        || (config.compute.require_full && config.compute.max_steps != config.model.full_steps())
    {
        return Err(boundary(
            ErrorCode::UnsupportedContract,
            Stage::Contract,
            "model precision/compute does not match search",
        ));
    }
    Ok(())
}

fn validate_position<P: ContractPosition>(position: &P) -> Result<(), ContractError> {
    position.validate_authority()?;
    position.snapshot().classification().validate()?;
    if position.snapshot().identity() != position.legal().state() {
        return Err(boundary(
            ErrorCode::IdentityMismatch,
            Stage::Contract,
            "Rules state and legal view identity differ",
        ));
    }
    if position.snapshot().classification().play_status == PlayStatus::Ongoing
        && position.legal().moves().is_empty()
    {
        return Err(boundary(
            ErrorCode::InvalidInput,
            Stage::Contract,
            "ongoing Rules state has no legal moves",
        ));
    }
    Ok(())
}

fn terminal_utility<P: ContractPosition>(position: &P) -> Option<f64> {
    match position.snapshot().classification().play_status {
        PlayStatus::Ongoing => None,
        PlayStatus::Terminal { winner, .. } => Some(match winner {
            None => 0.0,
            Some(winner) if winner == position.snapshot().side_to_move() => 1.0,
            Some(_) => -1.0,
        }),
    }
}

fn terminal_status<P: ContractPosition>(position: &P) -> Option<ContractSearchStatus> {
    let PlayStatus::Terminal { reason, winner } = position.snapshot().classification().play_status
    else {
        return None;
    };
    Some(ContractSearchStatus::Terminal {
        reason,
        winner,
        value: terminal_utility(position).expect("matched terminal"),
    })
}

fn live_acceptance<P, C, L>(
    config: &ContractSearchConfig,
    clock: &C,
    live_scope: &mut L,
    position: &P,
) -> Result<(), ContractError>
where
    P: ContractPosition,
    C: ContractClock,
    L: FnMut() -> AcceptanceScope,
{
    position.validate_authority()?;
    let scope = live_scope();
    let now = clock.now()?;
    if scope.game != config.scope.game
        || scope.root != config.scope.root
        || scope.model != config.scope.model
        || scope.encoding != config.scope.encoding
        || scope.backend != config.scope.backend
    {
        return Err(boundary(
            ErrorCode::Stale,
            Stage::Backup,
            "current game/root/model/encoding/backend replaced",
        ));
    }
    if config.cancellation.is_canceled() {
        return Err(boundary(
            ErrorCode::Canceled,
            Stage::Backup,
            "search owner canceled",
        ));
    }
    config.deadlines.hard.accepts(clock.domain(), now)
}

fn status_for_boundary(error: ContractError) -> ContractSearchStatus {
    let reason = match error.code {
        ErrorCode::Canceled => Some(ContractStopReason::Canceled),
        ErrorCode::Expired => Some(ContractStopReason::Expired),
        ErrorCode::Stale => Some(ContractStopReason::Stale),
        _ => None,
    };
    match reason {
        Some(reason) => ContractSearchStatus::Stopped {
            reason,
            source: Some(error),
        },
        None => ContractSearchStatus::Failed(ContractSearchFailure::Boundary(error)),
    }
}

fn failure_from_status(status: ContractSearchStatus) -> ContractSearchFailure {
    match status {
        ContractSearchStatus::Failed(failure) => failure,
        ContractSearchStatus::Stopped {
            source: Some(error),
            ..
        } => ContractSearchFailure::Boundary(error),
        ContractSearchStatus::Stopped { reason, .. } => ContractSearchFailure::Boundary(boundary(
            match reason {
                ContractStopReason::Canceled => ErrorCode::Canceled,
                ContractStopReason::Expired
                | ContractStopReason::SoftBudget
                | ContractStopReason::AdmissionClosed => ErrorCode::Expired,
                ContractStopReason::Stale => ErrorCode::Stale,
            },
            Stage::Backup,
            "logical selection closed during cleanup",
        )),
        _ => ContractSearchFailure::Boundary(boundary(
            ErrorCode::InvalidInput,
            Stage::Backup,
            "unexpected cleanup status",
        )),
    }
}

fn tree_boundary(error: SearchError) -> ContractError {
    match error {
        SearchError::AllocationFailed
        | SearchError::CounterOverflow
        | SearchError::DepthLimit
        | SearchError::NodeLimit
        | SearchError::EdgeLimit => boundary(
            ErrorCode::ResourceExhausted,
            Stage::Contract,
            "search tree resource limit",
        ),
        _ => boundary(
            ErrorCode::InvalidInput,
            Stage::Contract,
            "invalid search tree configuration",
        ),
    }
}

fn boundary(code: ErrorCode, stage: Stage, detail: &'static str) -> ContractError {
    ContractError::new(code, stage, detail)
}

#[cfg(test)]
mod allocator_tests {
    use super::*;

    #[test]
    fn independent_checked_counters_do_not_wrap_or_reset() {
        let ids = IdAllocator::new(ProcessEpoch(5));
        assert_eq!(ids.request().unwrap().sequence, 1);
        assert_eq!(ids.selection().unwrap().sequence, 1);
        assert_eq!(ids.request().unwrap().sequence, 2);
        ids.requests.store(u64::MAX, Ordering::Relaxed);
        assert_eq!(
            ids.request().unwrap_err().code,
            ErrorCode::ResourceExhausted
        );
        assert_eq!(ids.selection().unwrap().sequence, 2);
        assert_eq!(ids.requests.load(Ordering::Relaxed), u64::MAX);
    }
}
