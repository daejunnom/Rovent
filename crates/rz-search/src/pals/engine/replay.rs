//! One-shot, caller-declared frozen-parent replay. This is an additive logical
//! engine lane, not the legacy CPU dispatcher and not a Query/2 admission proof.
//!
//! Historical digests and hypotheses never become accepted records or imported
//! task/observation IDs. A fresh Rules owner and empty engine stores first obtain
//! an actual Proposer line. Two independent PlanAssisted CPU/TT owners then check
//! the same Rules-ordered response restriction. Only the completed after check
//! may supply the Counterexample consumed by a new Reply context.
//!
//! The caller-provided RoleModel is not evidence of native worker/model-epoch
//! freshness. Constructor allocation and Rules reconstruction precede `run`;
//! its elapsed time therefore is not whole invocation cost. This module grants
//! no native causality/physical closure, strategic refutation, utility, target,
//! training, successful strategic Repair, or whole-line authority. The optional
//! Repair tail records an actual accepted model continuation and a separate fresh
//! unrestricted endpoint check; it issues no supported-repair conclusion and
//! does not enter the native post-Repair recheck phase.

use super::*;
use crate::cpu::{CpuCompletion, CpuConfig, CpuOrderingPolicy, CpuProfile};
use rz_position::{HistoryCompleteness, HistoryOrigin};
use std::fmt::Write;

mod opponent_recheck;
use opponent_recheck::OpponentRecheckState;
pub use opponent_recheck::{
    FRESH_REPLAY_OPPONENT_SCOPE, RepairOpponentReplayRequirements, ReplayOpponentEndpoint,
    ReplayOpponentOutcome, ReplayOpponentRepairOrigin, ReplayOpponentRoleStep,
    repair_opponent_replay_requirements,
};

pub const FRESH_REPLAY_SCOPE: &str = "rz-pals-frozen-parent-defend-response-replay/1";
pub const FRESH_REPLAY_REPAIR_SCOPE: &str = "rz-pals-frozen-parent-repair-endpoint-replay/1";
const MAX_REPLAY_HISTORY_PLIES: usize = 4096;
const MAX_REPLAY_CONDITION_BYTES: usize = 4096;

/// Pure finite upper-bound declarations for the optional Repair endpoint lane.
/// These are not observed allocation peaks, StoreLimits, execution support,
/// Query admission or native/physical authority. Product configuration/plan
/// bounds are validated separately by the existing owner constructor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepairReplayRequirements {
    cpu_nodes: u64,
    role_calls: u64,
    nodes: usize,
    observations: usize,
    line_chunks: usize,
    records: usize,
    stages: usize,
}
impl RepairReplayRequirements {
    pub const fn cpu_nodes(&self) -> u64 {
        self.cpu_nodes
    }
    pub const fn role_calls(&self) -> u64 {
        self.role_calls
    }
    pub const fn nodes(&self) -> usize {
        self.nodes
    }
    pub const fn observations(&self) -> usize {
        self.observations
    }
    pub const fn line_chunks(&self) -> usize {
        self.line_chunks
    }
    pub const fn records(&self) -> usize {
        self.records
    }
    pub const fn stages(&self) -> usize {
        self.stages
    }
}

/// Computes 3N and the source-owned worst-case Repair reservations without I/O,
/// allocation, model/CPU calls or any private StoreLimits copy. Invalid scalar
/// extents and every intermediate arithmetic/conversion overflow are rejected;
/// even error construction uses only static InvalidPlan details (no Box).
pub fn repair_replay_requirements(
    line_plies: usize,
    prefix_plies: usize,
    nodes_per_check: u64,
) -> Result<RepairReplayRequirements, ReplayError> {
    if line_plies == 0 || prefix_plies >= line_plies || nodes_per_check == 0 {
        return Err(ReplayError::InvalidPlan("invalid Repair L/P/N"));
    }
    let cpu_nodes = nodes_per_check
        .checked_mul(3)
        .ok_or(ReplayError::InvalidPlan("3N overflow"))?;
    let tail = line_plies
        .checked_sub(prefix_plies)
        .and_then(|remaining| remaining.checked_sub(1))
        .ok_or(ReplayError::InvalidPlan("Repair response extent"))?;
    // L Proposal calls, one initial Reply, then at most L-P-1 calls per tail.
    let role_calls = line_plies
        .checked_add(1)
        .and_then(|calls| {
            tail.checked_mul(2)
                .and_then(|extra| calls.checked_add(extra))
        })
        .and_then(|calls| u64::try_from(calls).ok())
        .ok_or(ReplayError::InvalidPlan("Repair role bound overflow"))?;
    // Root+Proposal, bounded CPU candidate, selected response and two tails.
    let nodes = line_plies
        .checked_mul(4)
        .and_then(|count| {
            prefix_plies
                .checked_mul(3)
                .and_then(|shared| count.checked_sub(shared))
        })
        .ok_or(ReplayError::InvalidPlan("Repair node bound overflow"))?;
    let observations = nodes.checked_add(6).ok_or(ReplayError::InvalidPlan(
        "Repair observation bound overflow",
    ))?;
    let line_chunks = nodes
        .checked_mul(
            line_plies
                .checked_add(1)
                .ok_or(ReplayError::InvalidPlan("Repair line extent overflow"))?,
        )
        .and_then(|count| {
            line_plies
                .checked_mul(5)
                .and_then(|extra| count.checked_add(extra))
        })
        .ok_or(ReplayError::InvalidPlan("Repair line chunk bound overflow"))?;
    Ok(RepairReplayRequirements {
        cpu_nodes,
        role_calls,
        nodes,
        observations,
        line_chunks,
        records: 4,
        stages: 3,
    })
}

/// Binding declarations only. The independently registered frozen parent,
/// Query/2 bytes, catalogue and before-result ordering are checked by the caller.
/// No historical RequestId, ExecutionId, record or model epoch is imported.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayHistoricalPins {
    pub parent_input_sha256: [u8; 32],
    pub current_view_sha256: [u8; 32],
    pub frozen_admission_sha256: [u8; 32],
    pub query_sha256: [u8; 32],
    pub catalogue_sha256: [u8; 32],
    pub before_result_sha256: [u8; 32],
    pub semantic_input_sha256: [u8; 32],
    pub cpu_request_sha256: [u8; 32],
}

/// Owned declarations are checked against Rules before any role or CPU call.
/// `claimed_line` is a legal historical hypothesis, not a CPU-verified full line.
/// The original response order is retained; effective order is the target's full
/// Rules legal order filtered by membership, matching the legacy question.
#[derive(Debug)]
pub struct DefendResponseReplayPlan {
    pub historical: ReplayHistoricalPins,
    pub expected_root: PositionSnapshot,
    pub expected_target: PositionSnapshot,
    pub root_legal_order: Vec<BoardMove>,
    pub target_legal_order: Vec<BoardMove>,
    pub prefix: Vec<BoardMove>,
    pub response_restriction: Vec<BoardMove>,
    pub claimed_line: Vec<BoardMove>,
    pub baseline_depth: u16,
    pub requested_depth: u16,
    /// Exact budget of each check. It is never replaced by a remaining/minimum.
    pub nodes_per_check: u64,
    /// Both owners have this identical H1 capability; CpuLimits select H0/H1.
    pub cpu: CpuConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayCpuPhase {
    Baseline,
    After,
    RepairEndpoint,
    RepairOpponentEndpoint,
}
impl ReplayCpuPhase {
    fn label(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::After => "after",
            Self::RepairEndpoint => "repair-endpoint",
            Self::RepairOpponentEndpoint => "repair-opponent-endpoint",
        }
    }
}

/// Read-only actual stage ledger. A report can be partial or rejected; presence
/// alone is never completion. Task conditions have a replay phase suffix while
/// registered checker conditions remain the actual unchanged H1 configuration.
#[derive(Debug)]
pub struct ReplayStageReport {
    phase: ReplayCpuPhase,
    requested_depth: u16,
    node_budget: u64,
    execution: ExecutionId,
    task_condition: String,
    registered_condition: String,
    report: Option<CpuReport>,
    attempt: Option<CheckerAttempt>,
    observation: Option<ObservationId>,
    exact_completed: bool,
    cleanup_error: Option<Box<StoreError>>,
}
impl ReplayStageReport {
    pub fn phase(&self) -> ReplayCpuPhase {
        self.phase
    }
    pub fn requested_depth(&self) -> u16 {
        self.requested_depth
    }
    pub fn node_budget(&self) -> u64 {
        self.node_budget
    }
    pub fn execution(&self) -> ExecutionId {
        self.execution
    }
    pub fn task_condition(&self) -> &str {
        &self.task_condition
    }
    pub fn registered_condition(&self) -> &str {
        &self.registered_condition
    }
    pub fn report(&self) -> Option<&CpuReport> {
        self.report.as_ref()
    }
    pub fn attempt(&self) -> Option<&CheckerAttempt> {
        self.attempt.as_ref()
    }
    pub fn observation(&self) -> Option<ObservationId> {
        self.observation
    }
    pub fn exact_completed(&self) -> bool {
        self.exact_completed
    }
    /// Secondary task cleanup never replaces the original operation error.
    pub fn cleanup_error(&self) -> Option<&StoreError> {
        self.cleanup_error.as_deref()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayOutcome {
    RulesTerminal(TerminalReason),
    /// The actual newly generated Proposal did not match the historical prefix.
    SeedNotApplied,
    /// No completed H0/H1 claim or Reply publication follows a partial check.
    Partial(ReplayCpuPhase),
    /// The actual after response equals Proposal's response; no replacement made.
    SameProposalResponse,
    ReplyAccepted {
        selected_response: BoardMove,
        after_observation: ObservationId,
        publication_revision: u64,
    },
}

/// Search-local factual outcomes only. A completed endpoint is an unrestricted
/// side-to-move CPU observation, not a repaired-line/root value, successful
/// strategic repair, conditional refutation or native/utility authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayRepairOutcome {
    ReplyNotApplied(ReplayOutcome),
    /// No actual Repair output was accepted (for example, a terminal response
    /// or an already full prefix). No Repair record or endpoint claim is made.
    NoAcceptedRepair,
    PartialRepairEndpoint {
        execution: ExecutionId,
        observation: Option<ObservationId>,
    },
    CompletedRepairEndpoint {
        repaired_line: LineId,
        repair_record_revision: u64,
        endpoint_state: StateId,
        endpoint_situation: SituationId,
        execution: ExecutionId,
        observation: ObservationId,
    },
    RulesTerminalRepairEndpoint {
        repaired_line: LineId,
        repair_record_revision: u64,
        endpoint_state: StateId,
        endpoint_situation: SituationId,
        reason: TerminalReason,
    },
}

#[derive(Debug)]
pub enum ReplayError {
    InvalidPlan(&'static str),
    AlreadyUsed,
    Search(Box<PalsError>),
}
impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPlan(reason) => write!(f, "fresh replay declaration: {reason}"),
            Self::AlreadyUsed => f.write_str("fresh replay owner already used"),
            Self::Search(error) => std::fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for ReplayError {}
impl From<PalsError> for ReplayError {
    fn from(error: PalsError) -> Self {
        Self::Search(Box::new(error))
    }
}
impl From<PositionError> for ReplayError {
    fn from(error: PositionError) -> Self {
        PalsError::from(error).into()
    }
}
impl From<StoreError> for ReplayError {
    fn from(error: StoreError) -> Self {
        PalsError::from(error).into()
    }
}
impl From<CpuError> for ReplayError {
    fn from(error: CpuError) -> Self {
        PalsError::from(error).into()
    }
}
impl From<CheckerError> for ReplayError {
    fn from(error: CheckerError) -> Self {
        PalsError::from(error).into()
    }
}

/// Owns a newly reconstructed Rules root, empty logical stores and internally
/// constructed own CPU checkers. There is no engine/store mutable accessor or
/// reset/retry method. Even an invalid or failed `run` consumes this owner.
pub struct FreshReplayOwner<M: RoleModel> {
    engine: PalsEngine<M>,
    root: Position,
    target: PositionSnapshot,
    plan: DefendResponseReplayPlan,
    effective_order: Vec<BoardMove>,
    stages: Vec<ReplayStageReport>,
    used: bool,
    counters: PalsCounters,
    elapsed: Option<Duration>,
    proposal: Vec<BoardMove>,
    candidate_line: Vec<BoardMove>,
    reply_line: Vec<BoardMove>,
    reply_prepared_context: Option<RoleLogicalContext>,
    reply_accepted_context: Option<RoleLogicalContext>,
    publication_revision: Option<u64>,
    model_counterline: Vec<BoardMove>,
    repaired_line: Vec<BoardMove>,
    repair_record_revision: Option<u64>,
    repair_line_id: Option<LineId>,
    repair_endpoint_node: Option<usize>,
    repair_endpoint_snapshot: Option<PositionSnapshot>,
    opponent_recheck: Option<Box<OpponentRecheckState>>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum EndpointKind {
    Repair,
    Opponent,
}
impl EndpointKind {
    fn phase(self) -> ReplayCpuPhase {
        match self {
            Self::Repair => ReplayCpuPhase::RepairEndpoint,
            Self::Opponent => ReplayCpuPhase::RepairOpponentEndpoint,
        }
    }
    fn prior_stages(self) -> usize {
        match self {
            Self::Repair => 2,
            Self::Opponent => 3,
        }
    }
    fn scope(self) -> &'static str {
        match self {
            Self::Repair => FRESH_REPLAY_REPAIR_SCOPE,
            Self::Opponent => FRESH_REPLAY_OPPONENT_SCOPE,
        }
    }
}

struct StageRequest<'a> {
    root: usize,
    target: usize,
    phase: ReplayCpuPhase,
    depth: u16,
    limits: PalsLimits,
    cancel: &'a AtomicBool,
}

struct StartedStage<'a> {
    index: usize,
    root: usize,
    target: usize,
    consumer: TaskConsumer,
    active_root: SituationId,
    root_revision: u64,
    limits: PalsLimits,
    cancel: &'a AtomicBool,
}

impl<M: RoleModel> FreshReplayOwner<M> {
    pub fn new(
        config: PalsConfig,
        model: M,
        historical_root: Position,
        plan: DefendResponseReplayPlan,
    ) -> Result<Self, ReplayError> {
        config.validate()?;
        validate_extents(&config, &plan)?;
        // Check the input's O(1) extent before snapshot/identity or replay work.
        if historical_root.known_history_len() == 0
            || historical_root.known_history_len() > MAX_REPLAY_HISTORY_PLIES + 1
            || historical_root.history_completeness() != HistoryCompleteness::Complete
            || historical_root.history_origin() != HistoryOrigin::StartPosition
        {
            return Err(ReplayError::InvalidPlan(
                "first lane requires bounded complete startpos history",
            ));
        }
        let old_snapshot = historical_root.snapshot();
        if !old_snapshot.same_state(&plan.expected_root)
            || old_snapshot.history_completeness() != HistoryCompleteness::Complete
        {
            return Err(ReplayError::InvalidPlan(
                "root requires exact complete Rules history",
            ));
        }
        let replay = old_snapshot.uci_replay(MAX_REPLAY_HISTORY_PLIES)?;
        let mut root = Position::startpos();
        if replay.origin != HistoryOrigin::StartPosition
            || replay.completeness != HistoryCompleteness::Complete
            || replay.start_fen != root.to_fen()
        {
            return Err(ReplayError::InvalidPlan(
                "UCI replay origin/start FEN differs from startpos",
            ));
        }
        for movement in replay.moves {
            root.make_move(movement)?;
        }
        if !root.snapshot().same_state(&old_snapshot) || root.legal_moves() != plan.root_legal_order
        {
            return Err(ReplayError::InvalidPlan(
                "fresh Rules reconstruction/root legal order",
            ));
        }
        let mut target = root.clone();
        for &movement in &plan.prefix {
            if !matches!(target.classify_position()?.play_status, PlayStatus::Ongoing) {
                return Err(ReplayError::InvalidPlan(
                    "prefix continues after Rules terminal",
                ));
            }
            target.make_move(movement)?;
        }
        if !target.snapshot().same_state(&plan.expected_target)
            || target.legal_moves() != plan.target_legal_order
        {
            return Err(ReplayError::InvalidPlan(
                "target full history/legal order differs",
            ));
        }
        let terminal_root = matches!(
            root.classify_position()?.play_status,
            PlayStatus::Terminal { .. }
        );
        if terminal_root {
            if !plan.prefix.is_empty()
                || !plan.response_restriction.is_empty()
                || !plan.claimed_line.is_empty()
            {
                return Err(ReplayError::InvalidPlan(
                    "terminal root has a response hypothesis",
                ));
            }
        } else if plan.prefix.is_empty()
            || target.side_to_move() == root.side_to_move()
            || !matches!(target.classify_position()?.play_status, PlayStatus::Ongoing)
            || plan.response_restriction.is_empty()
        {
            return Err(ReplayError::InvalidPlan(
                "defend_response needs an ongoing opponent target",
            ));
        }
        for (at, movement) in plan.response_restriction.iter().enumerate() {
            if !plan.target_legal_order.contains(movement)
                || plan.response_restriction[..at].contains(movement)
            {
                return Err(ReplayError::InvalidPlan(
                    "response restriction is illegal or duplicate",
                ));
            }
        }
        let mut effective_order = bounded_moves(&[], 256)?;
        for &movement in &plan.target_legal_order {
            if plan.response_restriction.contains(&movement) {
                effective_order.push(movement);
            }
        }
        let mut claim = root.clone();
        for &movement in &plan.claimed_line {
            if !matches!(claim.classify_position()?.play_status, PlayStatus::Ongoing) {
                return Err(ReplayError::InvalidPlan(
                    "claimed hypothesis follows terminal",
                ));
            }
            claim.make_move(movement)?;
        }
        let cpu = CpuEngine::new(plan.cpu.clone())?;
        if cpu.ordering_policy() != CpuOrderingPolicy::LegacyMvvLvaV1 {
            return Err(ReplayError::InvalidPlan(
                "first replay requires legacy CPU ordering",
            ));
        }
        let engine = PalsEngine::new(config, model, cpu)?;
        let mut stages = Vec::new();
        stages
            .try_reserve_exact(2)
            .map_err(|_| PalsError::Capacity)?;
        let extent = engine.config.line_plies;
        Ok(Self {
            engine,
            root,
            target: target.snapshot(),
            plan,
            effective_order,
            stages,
            used: false,
            counters: PalsCounters::default(),
            elapsed: None,
            proposal: bounded_moves(&[], extent)?,
            candidate_line: bounded_moves(&[], extent)?,
            reply_line: bounded_moves(&[], extent)?,
            reply_prepared_context: None,
            reply_accepted_context: None,
            publication_revision: None,
            // The legacy Reply-only lane has no additional heap reservation.
            model_counterline: Vec::new(),
            repaired_line: Vec::new(),
            repair_record_revision: None,
            repair_line_id: None,
            repair_endpoint_node: None,
            repair_endpoint_snapshot: None,
            opponent_recheck: None,
        })
    }

    pub fn historical_pins(&self) -> &ReplayHistoricalPins {
        &self.plan.historical
    }
    pub fn root_snapshot(&self) -> PositionSnapshot {
        self.root.snapshot()
    }
    pub fn target_snapshot(&self) -> &PositionSnapshot {
        &self.target
    }
    pub fn original_restriction(&self) -> &[BoardMove] {
        &self.plan.response_restriction
    }
    pub fn effective_order(&self) -> &[BoardMove] {
        &self.effective_order
    }
    pub fn claimed_hypothesis(&self) -> &[BoardMove] {
        &self.plan.claimed_line
    }
    pub fn stages(&self) -> &[ReplayStageReport] {
        &self.stages
    }
    pub fn counters(&self) -> PalsCounters {
        self.counters
    }
    pub fn elapsed(&self) -> Option<Duration> {
        self.elapsed
    }
    pub fn proposal(&self) -> &[BoardMove] {
        &self.proposal
    }
    pub fn candidate_line(&self) -> &[BoardMove] {
        &self.candidate_line
    }
    /// Only the selected Reply prefix+move, not a completed counterline proof.
    pub fn reply_line(&self) -> &[BoardMove] {
        &self.reply_line
    }
    pub fn records(&self) -> &[RoleRecord] {
        &self.engine.semantic.records
    }
    pub fn reply_prepared_context(&self) -> Option<&RoleLogicalContext> {
        self.reply_prepared_context.as_ref()
    }
    /// Actual successful acceptance callback context, even if a later control
    /// check rejects the invocation. This is not a native RequestId receipt.
    pub fn reply_accepted_context(&self) -> Option<&RoleLogicalContext> {
        self.reply_accepted_context.as_ref()
    }
    pub fn publication_revision(&self) -> Option<u64> {
        self.publication_revision
    }
    /// Actual model-generated continuation from the selected Reply response.
    /// It does not claim that the restricted CPU discovered this full line.
    pub fn model_counterline(&self) -> &[BoardMove] {
        &self.model_counterline
    }
    /// Actual accepted/partial model Repair prefix, including retained failure
    /// work. Presence alone is not a Repair acceptance or endpoint completion.
    pub fn repaired_line(&self) -> &[BoardMove] {
        &self.repaired_line
    }
    pub fn repair_record_revision(&self) -> Option<u64> {
        self.repair_record_revision
    }
    pub fn repair_line_id(&self) -> Option<LineId> {
        self.repair_line_id
    }
    /// Reuses the parent's actual store provenance validator. An unobserved or
    /// invalid endpoint stays explicit; this getter cannot mint completion.
    pub fn repair_endpoint(&self) -> Option<RecheckEndpoint<'_>> {
        let node = self.engine.semantic.nodes.get(self.repair_endpoint_node?)?;
        let snapshot = self.repair_endpoint_snapshot.as_ref()?;
        Some(PalsEngine::<M>::recheck_endpoint(
            node,
            &self.engine.semantic.stores,
            snapshot,
        ))
    }
    pub fn task(&self, execution: ExecutionId) -> Result<&TaskRecord, StoreError> {
        self.engine.semantic.stores.tasks.get(execution)
    }
    pub fn observation(&self, id: ObservationId) -> Result<&Observation, StoreError> {
        self.engine.semantic.stores.observations.get(id)
    }
    pub fn strict_query_admission_verified(&self) -> bool {
        false
    }
    pub fn native_freshness_verified(&self) -> bool {
        false
    }
    pub fn whole_cost_verified(&self) -> bool {
        false
    }
    pub fn physical_closure_verified(&self) -> bool {
        false
    }
    pub fn utility_authority(&self) -> bool {
        false
    }
    pub fn training_target_created(&self) -> bool {
        false
    }

    pub fn run(
        &mut self,
        limits: PalsLimits,
        cancel: &AtomicBool,
    ) -> Result<ReplayOutcome, ReplayError> {
        self.run_once(limits, cancel, |owner, counters| {
            owner.run_inner(limits, cancel, counters)
        })
    }

    /// Opt-in 3N lane. The old two-check Reply result is not reinterpreted: only
    /// an accepted Reply can enter a new model continuation/Repair tail. No
    /// recheck callbacks, supported-repair publication or value propagation run.
    pub fn run_with_repair(
        &mut self,
        limits: PalsLimits,
        cancel: &AtomicBool,
    ) -> Result<ReplayRepairOutcome, ReplayError> {
        self.run_once(limits, cancel, |owner, counters| {
            owner.prepare_repair(limits, cancel)?;
            let reply = owner.run_inner(limits, cancel, counters)?;
            owner.repair_tail(reply, limits, cancel, counters)
        })
    }

    fn run_once<T>(
        &mut self,
        limits: PalsLimits,
        cancel: &AtomicBool,
        operation: impl FnOnce(&mut Self, &mut PalsCounters) -> Result<T, ReplayError>,
    ) -> Result<T, ReplayError> {
        if self.used {
            return Err(ReplayError::AlreadyUsed);
        }
        self.used = true;
        let started = Instant::now();
        let mut counters = PalsCounters::default();
        let mut result = operation(self, &mut counters);
        counters.retained_situations = self.engine.semantic.nodes.len();
        self.counters = counters;
        self.engine.semantic.last_search_counters = Some(counters);
        let closure = match &result {
            Err(ReplayError::Search(error)) => match error.as_ref() {
                PalsError::Role(RoleError::Canceled) => RoleSearchClosure::Canceled,
                PalsError::Role(RoleError::Deadline) => RoleSearchClosure::Deadline,
                PalsError::Role(RoleError::PhysicalCompletionUnknown) => {
                    RoleSearchClosure::PhysicalCompletionUnknown
                }
                _ => RoleSearchClosure::Failed,
            },
            Err(_) => RoleSearchClosure::Failed,
            Ok(_) => RoleSearchClosure::Completed,
        };
        self.engine.semantic.model.finish_search(closure);
        // A secondary accounting callback must not hide a primary failure.
        if result.is_ok()
            && let Err(error) = self.check_control(limits, cancel)
        {
            result = Err(error);
        }
        self.elapsed = Some(started.elapsed());
        result
    }

    fn check_control(&self, limits: PalsLimits, cancel: &AtomicBool) -> Result<(), ReplayError> {
        check_role_control(limits.deadline, cancel).map_err(|error| PalsError::from(error).into())
    }

    fn run_inner(
        &mut self,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<ReplayOutcome, ReplayError> {
        let total = self
            .plan
            .nodes_per_check
            .checked_mul(2)
            .ok_or(ReplayError::InvalidPlan("2N overflow"))?;
        if limits.max_rounds == 0
            || limits.cpu_depth != self.plan.requested_depth
            || limits.max_cpu_nodes < total
            || self.engine.config.cpu_nodes_per_task != self.plan.nodes_per_check
        {
            return Err(ReplayError::InvalidPlan(
                "limits do not support the fixed declared H1/2N",
            ));
        }
        self.check_control(limits, cancel)?;
        self.engine.start_checker(limits.deadline, cancel)?;
        self.check_control(limits, cancel)?;
        self.engine
            .semantic
            .stores
            .focus_actual_moves(self.root.snapshot())?;
        let root = self
            .engine
            .intern_checked(self.root.clone(), limits, cancel)?;
        if let Some((reason, _)) = self.engine.semantic.nodes[root].terminal {
            return Ok(ReplayOutcome::RulesTerminal(reason));
        }
        let ranked = self.engine.ranked(
            root,
            Call::Propose,
            &[],
            &[],
            None,
            limits,
            cancel,
            counters,
        )?;
        self.check_control(limits, cancel)?;
        let first_move = *ranked
            .first()
            .ok_or(RoleError::InvalidOutput)
            .map_err(PalsError::from)?;
        let mut next = self.root.clone();
        next.make_move(first_move)?;
        let first = self
            .engine
            .connect_checked(root, first_move, next, counters, limits, cancel)?;
        let mut proposal = bounded_moves(&[first_move], self.engine.config.line_plies)?;
        let followed = self.engine.follow(
            first,
            &mut proposal,
            &[],
            None,
            Call::Propose,
            limits,
            cancel,
            counters,
        );
        self.proposal = proposal;
        // Preserve the actual accepted prefix even if a later role call fails.
        followed?;
        self.check_control(limits, cancel)?;
        counters.proposals += 1;
        self.engine.record_checked(
            RecordKind::Proposal,
            &self.proposal,
            None,
            0,
            None,
            None,
            limits,
            cancel,
        )?;
        if !self.proposal.starts_with(&self.plan.prefix)
            || self.proposal.len() <= self.plan.prefix.len()
        {
            return Ok(ReplayOutcome::SeedNotApplied);
        }
        let mut target = root;
        for &movement in &self.plan.prefix {
            target = self.engine.semantic.nodes[target]
                .edges
                .iter()
                .find(|edge| edge.movement == movement)
                .ok_or(StoreError::InvalidEvidence(
                    "actual Proposal lacks checked prefix edge",
                ))?
                .child;
        }
        if !self.engine.semantic.nodes[target]
            .position
            .snapshot()
            .same_state(&self.target)
        {
            return Err(ReplayError::InvalidPlan(
                "actual Proposal target differs from declared Rules target",
            ));
        }
        self.check_control(limits, cancel)?;
        let baseline = self.stage(
            StageRequest {
                root,
                target,
                phase: ReplayCpuPhase::Baseline,
                depth: self.plan.baseline_depth,
                limits,
                cancel,
            },
            counters,
        )?;
        if baseline.is_none() {
            return Ok(ReplayOutcome::Partial(ReplayCpuPhase::Baseline));
        }
        self.check_control(limits, cancel)?;
        // Replace the baseline with a newly constructed independent CPU. Their
        // allocations briefly coexist; this is not a peak-memory observation.
        // No previous token, report, history table or TT is supplied to `new`.
        let fresh = OwnedCpuChecker::new(CpuEngine::new(self.plan.cpu.clone())?)?;
        self.engine.researcher.cpu = Box::new(fresh);
        self.engine.start_checker(limits.deadline, cancel)?;
        self.check_control(limits, cancel)?;
        let after = self.stage(
            StageRequest {
                root,
                target,
                phase: ReplayCpuPhase::After,
                depth: self.plan.requested_depth,
                limits,
                cancel,
            },
            counters,
        )?;
        let Some(candidate) = after else {
            return Ok(ReplayOutcome::Partial(ReplayCpuPhase::After));
        };
        self.check_control(limits, cancel)?;
        let inserted = self.engine.insert_cpu_candidate(
            root,
            target,
            &self.plan.prefix,
            &self.proposal,
            &candidate,
            limits,
            cancel,
            counters,
        )?;
        self.check_control(limits, cancel)?;
        let Some((_, _, line)) = inserted else {
            return Ok(ReplayOutcome::SameProposalResponse);
        };
        self.candidate_line = line;
        // Recheck actual store/root bindings after insertion and immediately
        // before publication, without fabricating a leaf value or model result.
        if candidate.generation != self.engine.semantic.stores.generation()
            || self.engine.semantic.stores.root() != Some(candidate.root)
            || self
                .engine
                .semantic
                .stores
                .situations
                .get(candidate.root)?
                .revision
                != candidate.root_revision
            || self
                .engine
                .semantic
                .stores
                .observations
                .get(candidate.observation)?
                .state
                != self.engine.semantic.nodes[target].state
            || !self.engine.semantic.nodes[target]
                .position
                .snapshot()
                .same_state(&self.target)
        {
            return Err(StoreError::StaleConsumer.into());
        }
        self.check_control(limits, cancel)?;
        self.engine.record_with_cpu_checked(
            RecordKind::Counterexample,
            &self.candidate_line,
            None,
            0,
            None,
            None,
            Some(candidate.observation),
            limits,
            cancel,
        )?;
        self.publication_revision = Some(self.engine.semantic.revision);
        let question = RoleQuestion {
            purpose: RoleQueryPurpose::ReplyPolicy,
            prefix: &self.plan.prefix,
            proposal: &self.proposal,
            refutation: Some(&self.candidate_line),
            divergences: &[],
        };
        self.reply_prepared_context = Some(self.engine.role_context(target, question)?);
        self.check_control(limits, cancel)?;
        let reply = self.engine.ranked(
            target,
            Call::Reply,
            &self.plan.prefix,
            &self.proposal,
            Some(&self.candidate_line),
            limits,
            cancel,
            counters,
        )?;
        self.reply_accepted_context = self.reply_prepared_context.clone();
        self.check_control(limits, cancel)?;
        let selected = *reply
            .first()
            .ok_or(RoleError::InvalidOutput)
            .map_err(PalsError::from)?;
        let mut checked = self.engine.semantic.nodes[target].position.clone();
        checked.make_move(selected)?;
        self.engine
            .connect_checked(target, selected, checked, counters, limits, cancel)?;
        self.reply_line = bounded_moves(&self.plan.prefix, self.engine.config.line_plies)?;
        self.reply_line.push(selected);
        self.check_control(limits, cancel)?;
        Ok(ReplayOutcome::ReplyAccepted {
            selected_response: selected,
            after_observation: candidate.observation,
            publication_revision: self.engine.semantic.revision,
        })
    }

    fn prepare_repair(
        &mut self,
        limits: PalsLimits,
        cancel: &AtomicBool,
    ) -> Result<(), ReplayError> {
        let extent = self.engine.config.line_plies;
        let required =
            repair_replay_requirements(extent, self.plan.prefix.len(), self.plan.nodes_per_check)?;
        // Store limits remain constructor-owned and are not exported in the
        // pure declaration. Comparisons use the same calculated requirements.
        let reserved = PalsEngine::<M>::store_limits(&self.engine.config);
        if limits.max_rounds == 0
            || limits.cpu_depth != self.plan.requested_depth
            || limits.max_cpu_nodes < required.cpu_nodes()
            || self.engine.config.cpu_nodes_per_task != self.plan.nodes_per_check
            || self.engine.config.max_role_calls < required.role_calls()
            || self.engine.config.max_records < required.records()
            || self.engine.config.max_nodes < required.nodes()
            || reserved.observations < required.observations()
            || reserved.line_chunks < required.line_chunks()
            || reserved.executions < required.stages()
            || reserved.consumers < required.stages()
            || !self.stages.is_empty()
        {
            return Err(ReplayError::InvalidPlan(
                "limits do not support fixed H1/3N and Repair reservations",
            ));
        }
        self.check_control(limits, cancel)?;
        self.stages
            .try_reserve_exact(required.stages())
            .map_err(|_| PalsError::Capacity)?;
        self.model_counterline = bounded_moves(&[], extent)?;
        self.repaired_line = bounded_moves(&[], extent)?;
        self.check_control(limits, cancel)
    }

    fn repair_tail(
        &mut self,
        reply: ReplayOutcome,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<ReplayRepairOutcome, ReplayError> {
        let ReplayOutcome::ReplyAccepted {
            selected_response, ..
        } = reply
        else {
            return Ok(ReplayRepairOutcome::ReplyNotApplied(reply));
        };
        self.check_control(limits, cancel)?;
        let context = self
            .reply_accepted_context
            .as_ref()
            .ok_or(StoreError::InvalidEvidence(
                "Repair lacks actual accepted Reply context",
            ))?;
        let root_situation = self
            .engine
            .semantic
            .stores
            .root()
            .ok_or(StoreError::StaleConsumer)?;
        let root = self
            .engine
            .semantic
            .nodes
            .iter()
            .position(|node| node.situation == root_situation)
            .ok_or(StoreError::StaleConsumer)?;
        let target = self
            .engine
            .semantic
            .nodes
            .iter()
            .position(|node| node.situation == context.situation)
            .ok_or(StoreError::StaleConsumer)?;
        if context.purpose != RoleQueryPurpose::ReplyPolicy
            || context.search_generation != self.engine.semantic.stores.generation()
            || Some(context.game_generation) != self.engine.semantic.game_generation
            || context.public_revision != self.engine.semantic.revision
            || !self.engine.semantic.nodes[root]
                .position
                .snapshot()
                .same_state(&self.root.snapshot())
            || !self.engine.semantic.nodes[target]
                .position
                .snapshot()
                .same_state(&self.target)
        {
            return Err(StoreError::StaleConsumer.into());
        }
        let response = self.engine.semantic.nodes[target]
            .edges
            .iter()
            .find(|edge| edge.movement == selected_response)
            .ok_or(StoreError::InvalidEvidence(
                "Repair lacks actual selected Reply edge",
            ))?
            .child;
        let mut counterline = std::mem::take(&mut self.model_counterline);
        counterline.extend_from_slice(&self.reply_line);
        // Always generate from the actual response's Rules state. In particular,
        // an H1 PV whose first move differs is never spliced into this branch.
        let followed = self.engine.follow(
            response,
            &mut counterline,
            &self.proposal,
            Some(&self.candidate_line),
            Call::Reply,
            limits,
            cancel,
            counters,
        );
        self.model_counterline = counterline;
        followed?;
        self.check_control(limits, cancel)?;
        counters.refutations += 1;
        // This full line is model-generated. The initial H1 observation remains
        // on its own initial Counterexample and accepted Reply context only.
        self.engine.record_checked(
            RecordKind::Counterexample,
            &self.model_counterline,
            None,
            0,
            None,
            None,
            limits,
            cancel,
        )?;
        self.check_control(limits, cancel)?;
        let accepted_before = counters.accepted_repair_outputs;
        let mut repaired = std::mem::take(&mut self.repaired_line);
        repaired.extend_from_slice(&self.reply_line);
        let followed = self.engine.follow(
            response,
            &mut repaired,
            &self.proposal,
            Some(&self.model_counterline),
            Call::Repair,
            limits,
            cancel,
            counters,
        );
        self.repaired_line = repaired;
        let endpoint = followed?;
        self.check_control(limits, cancel)?;
        if counters.accepted_repair_outputs == accepted_before {
            return Ok(ReplayRepairOutcome::NoAcceptedRepair);
        }
        // Re-run the actual complete ordered line with Rules, independently of
        // the path's node index. No history is manufactured from FEN or digests.
        let mut checked = self.root.clone();
        for &movement in &self.repaired_line {
            self.check_control(limits, cancel)?;
            if !matches!(
                checked.classify_position()?.play_status,
                PlayStatus::Ongoing
            ) {
                return Err(
                    StoreError::InvalidEvidence("Repair continues after Rules terminal").into(),
                );
            }
            checked.make_move(movement)?;
        }
        let snapshot = checked.snapshot();
        if !self.engine.semantic.nodes[endpoint]
            .position
            .snapshot()
            .same_state(&snapshot)
            || self.engine.semantic.nodes[endpoint].position.legal_moves() != checked.legal_moves()
            || !self
                .engine
                .semantic
                .stores
                .states
                .get(self.engine.semantic.nodes[endpoint].state)?
                .same_state(&snapshot)
        {
            return Err(
                StoreError::InvalidEvidence("Repair endpoint full Rules history/order").into(),
            );
        }
        self.check_control(limits, cancel)?;
        let (line_id, _) = self
            .engine
            .record_checked(
                RecordKind::Repair,
                &self.repaired_line,
                None,
                0,
                None,
                None,
                limits,
                cancel,
            )?
            .ok_or(StoreError::InvalidEvidence(
                "accepted Repair model publication absent",
            ))?;
        counters.repairs += 1;
        self.repair_line_id = Some(line_id);
        self.repair_record_revision = Some(self.engine.semantic.revision);
        self.repair_endpoint_node = Some(endpoint);
        self.repair_endpoint_snapshot = Some(snapshot);
        self.check_control(limits, cancel)?;
        let state = self.engine.semantic.nodes[endpoint].state;
        let situation = self.engine.semantic.nodes[endpoint].situation;
        if let Some((reason, _)) = self.engine.semantic.nodes[endpoint].terminal {
            let facts = self.repair_endpoint().ok_or(StoreError::InvalidEvidence(
                "Repair terminal snapshot absent",
            ))?;
            if !matches!(facts.evidence, RecheckEndpointEvidence::RulesTerminal { reason: actual, .. } if actual == reason)
            {
                return Err(StoreError::InvalidEvidence("Repair terminal Rules provenance").into());
            }
            self.check_control(limits, cancel)?;
            return Ok(ReplayRepairOutcome::RulesTerminalRepairEndpoint {
                repaired_line: line_id,
                repair_record_revision: self.engine.semantic.revision,
                endpoint_state: state,
                endpoint_situation: situation,
                reason,
            });
        }
        // Independent third TT/checker with exactly the same registered H1
        // capability. No prior report/token/cache is passed to its constructor.
        let fresh = OwnedCpuChecker::new(CpuEngine::new(self.plan.cpu.clone())?)?;
        self.engine.researcher.cpu = Box::new(fresh);
        self.engine.start_checker(limits.deadline, cancel)?;
        self.check_control(limits, cancel)?;
        let completed = self.repair_endpoint_stage(
            StageRequest {
                root,
                target: endpoint,
                phase: ReplayCpuPhase::RepairEndpoint,
                depth: self.plan.requested_depth,
                limits,
                cancel,
            },
            counters,
        )?;
        self.check_control(limits, cancel)?;
        let stage = self.stages.last().ok_or(PalsError::Capacity)?;
        let Some(observation) = completed else {
            return Ok(ReplayRepairOutcome::PartialRepairEndpoint {
                execution: stage.execution,
                observation: stage.observation,
            });
        };
        let execution = stage.execution;
        let facts = self.repair_endpoint().ok_or(StoreError::InvalidEvidence(
            "Repair endpoint snapshot absent",
        ))?;
        match facts.evidence {
            RecheckEndpointEvidence::OwnCpu {
                observation_id,
                execution_id,
                task,
                admitted_scope,
                ..
            } if observation_id == observation
                && execution_id == execution
                && task.status == TaskStatus::Completed(observation)
                && admitted_scope == CpuScoreScope::CompletedIteration
                && stage.exact_completed => {}
            RecheckEndpointEvidence::InvalidProvenance { error } => return Err(error.into()),
            _ => {
                return Err(StoreError::InvalidEvidence(
                    "Repair completed endpoint provenance absent",
                )
                .into());
            }
        }
        self.check_control(limits, cancel)?;
        Ok(ReplayRepairOutcome::CompletedRepairEndpoint {
            repaired_line: line_id,
            repair_record_revision: self.engine.semantic.revision,
            endpoint_state: state,
            endpoint_situation: situation,
            execution,
            observation,
        })
    }

    /// A separate unrestricted endpoint namespace. The two legacy restricted
    /// stage helpers below are deliberately unchanged and cannot mint this fact.
    fn repair_endpoint_stage(
        &mut self,
        request: StageRequest<'_>,
        counters: &mut PalsCounters,
    ) -> Result<Option<ObservationId>, ReplayError> {
        self.unrestricted_endpoint_stage(request, counters, EndpointKind::Repair)
    }

    fn unrestricted_endpoint_stage(
        &mut self,
        request: StageRequest<'_>,
        counters: &mut PalsCounters,
        kind: EndpointKind,
    ) -> Result<Option<ObservationId>, ReplayError> {
        let StageRequest {
            root,
            target,
            phase,
            depth,
            limits,
            cancel,
        } = request;
        self.check_control(limits, cancel)?;
        self.engine.validate_checker_namespace()?;
        if phase != kind.phase()
            || depth != self.plan.requested_depth
            || self.engine.semantic.nodes[target].terminal.is_some()
            || (kind == EndpointKind::Repair
                && self.engine.semantic.nodes[target].evidence.is_some())
            || self.engine.researcher.cpu.last_attempt().is_some()
            || self.engine.cpu_condition() != self.engine.researcher.cpu_registered_condition
            || self.engine.researcher.cpu.identity()
                != &self.engine.researcher.checker_registered_identity
            || self.stages.len() != kind.prior_stages()
            || self.stages.capacity() < kind.prior_stages() + 1
        {
            return Err(ReplayError::InvalidPlan(match kind {
                EndpointKind::Repair => "endpoint is not a fresh unrestricted third stage",
                EndpointKind::Opponent => "endpoint is not a fresh unrestricted fourth stage",
            }));
        }
        let descriptor = self.engine.owned_descriptor()?;
        let value_identity = self.engine.owned_identity()?.clone();
        let registered_condition = self.engine.cpu_condition();
        let mut task_condition = String::new();
        task_condition
            .try_reserve_exact(MAX_REPLAY_CONDITION_BYTES)
            .map_err(|_| PalsError::Capacity)?;
        write!(
            &mut task_condition,
            "{};replay={};phase={};question=AnalyzePosition;input-revision=0;root-moves=empty",
            registered_condition,
            kind.scope(),
            kind.phase().label(),
        )
        .map_err(|_| PalsError::Capacity)?;
        if task_condition.len() > MAX_REPLAY_CONDITION_BYTES {
            return Err(PalsError::Capacity.into());
        }
        let profile = stable_id(descriptor.config.profile.identity());
        let condition = stable_id(&task_condition);
        let generation = self.engine.semantic.stores.generation();
        let active_root = self
            .engine
            .semantic
            .stores
            .root()
            .ok_or(StoreError::StaleConsumer)?;
        let root_revision = self
            .engine
            .semantic
            .stores
            .situations
            .get(active_root)?
            .revision;
        self.engine.semantic.consumer_id = self
            .engine
            .semantic
            .consumer_id
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        let consumer = TaskConsumer {
            id: self.engine.semantic.consumer_id,
            situation: self.engine.semantic.nodes[target].situation,
            revision: self
                .engine
                .semantic
                .stores
                .situations
                .get(self.engine.semantic.nodes[target].situation)?
                .revision,
            generation,
            deadline_tick: self.engine.tick_at(limits.deadline),
        };
        let admission = self.engine.semantic.stores.request_task(
            TaskKey {
                state: self.engine.semantic.nodes[target].state,
                line: None,
                question: TaskQuestion::AnalyzePosition,
                root_moves: Vec::new(),
                model: 0,
                epoch: 0,
                value_identity: Some(value_identity),
                checker_identity: None,
                cpu_condition: Some(task_condition.clone()),
                profile,
                condition,
                input_revision: 0,
                requested_depth: depth,
                node_budget: self.plan.nodes_per_check,
            },
            consumer,
            self.engine.tick_at(Instant::now()),
        )?;
        let TaskAdmission::Start(execution) = admission else {
            return Err(StoreError::InvalidConditions(
                "fresh endpoint requires Start, never reuse/join/resume",
            )
            .into());
        };
        let index = self.stages.len();
        self.stages.push(ReplayStageReport {
            phase,
            requested_depth: depth,
            node_budget: self.plan.nodes_per_check,
            execution,
            task_condition,
            registered_condition,
            report: None,
            attempt: None,
            observation: None,
            exact_completed: false,
            cleanup_error: None,
        });
        let result = self.repair_endpoint_started(
            StartedStage {
                index,
                root,
                target,
                consumer,
                active_root,
                root_revision,
                limits,
                cancel,
            },
            counters,
            kind,
        );
        if result.is_err()
            && let Err(error) = self.engine.semantic.stores.tasks.fail(execution)
        {
            self.stages[index].cleanup_error = Some(Box::new(error));
        }
        result
    }

    fn repair_endpoint_started(
        &mut self,
        context: StartedStage<'_>,
        counters: &mut PalsCounters,
        kind: EndpointKind,
    ) -> Result<Option<ObservationId>, ReplayError> {
        let StartedStage {
            index,
            root,
            target,
            consumer,
            active_root,
            root_revision,
            limits,
            cancel,
        } = context;
        self.check_control(limits, cancel)?;
        let cpu_limits = CpuLimits {
            max_depth: self.plan.requested_depth,
            max_nodes: self.plan.nodes_per_check,
            deadline: Some(limits.deadline),
        };
        self.engine.researcher.cpu.preflight_task(cpu_limits)?;
        self.check_control(limits, cancel)?;
        counters.cpu_tasks_requested += 1;
        let result = self.engine.researcher.cpu.analyze(
            &self.engine.semantic.nodes[target].position,
            cpu_limits,
            cancel,
        );
        self.stages[index].attempt = self.engine.researcher.cpu.last_attempt().cloned();
        let report = match result {
            Ok(report) => match PalsEngine::<M>::own_report(report) {
                Ok(report) => report,
                Err(error) => {
                    self.engine
                        .observe_cpu_failure(counters, cpu_limits.max_nodes);
                    return Err(error.into());
                }
            },
            Err(error) => {
                if matches!(error, CheckerError::OwnedReportRejected { .. }) {
                    counters.cpu_tasks += 1;
                }
                self.engine
                    .observe_cpu_failure(counters, cpu_limits.max_nodes);
                return Err(error.into());
            }
        };
        self.stages[index].report = Some(report);
        let stage = &self.stages[index];
        let report = stage.report.as_ref().ok_or(PalsError::Capacity)?;
        let validation = self.engine.validate_cpu_report(
            &self.engine.semantic.nodes[target].position,
            report,
            self.engine.owned_identity()?,
            &stage.registered_condition,
            false,
            cpu_limits,
        );
        if validation.is_ok() {
            counters
                .cpu_nodes
                .checked_add(report.nodes)
                .ok_or(PalsError::Capacity)?;
            counters
                .cpu_quiescence_nodes
                .checked_add(report.quiescence_nodes)
                .ok_or(PalsError::Capacity)?;
            counters
                .cpu_tt_hits
                .checked_add(report.tt_hits)
                .ok_or(PalsError::Capacity)?;
        }
        let exact = validation.is_ok() && exact_coverage(report, cpu_limits.max_depth);
        PalsEngine::<M>::observe_cpu_report(
            counters,
            report,
            cpu_limits.max_depth,
            validation.is_ok(),
        );
        if validation.is_ok()
            && !exact
            && report.score_scope == CpuScoreScope::CompletedIteration
            && report.completed_depth >= cpu_limits.max_depth
        {
            counters.completed_cpu_tasks -= 1;
        }
        validation?;
        let observation = self
            .engine
            .semantic
            .stores
            .append_observation(Observation {
                state: self.engine.semantic.nodes[target].state,
                line: None,
                source: stable_id(report.score_provenance),
                epoch: 0,
                scope: EvidenceScope::DepthLimited {
                    depth: report.completed_depth,
                    profile: stable_id(self.plan.cpu.profile.identity()),
                    condition: stable_id(&stage.task_condition),
                },
                value_identity: Some(report.value_identity.clone()),
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: None,
                model_value_input: None,
                cpu_condition: Some(stage.task_condition.clone()),
                cpu_pv: None,
                score: if report.score_scope == CpuScoreScope::FrontierOnly {
                    RawScore::Estimate {
                        value: report.score as f32,
                        perspective: self.engine.semantic.nodes[target].position.side_to_move(),
                    }
                } else {
                    RawScore::Cpu {
                        value: report.score,
                        perspective: self.engine.semantic.nodes[target].position.side_to_move(),
                        bound: BoundKind::ExactWithinSearch,
                    }
                },
                budget: report.nodes,
                kind: ObservationKind::CpuAnalysis,
                supersedes: None,
                execution: Some(stage.execution),
            })?;
        self.stages[index].observation = Some(observation);
        self.stages[index].exact_completed = exact;
        if !exact {
            self.engine
                .semantic
                .stores
                .tasks
                .fail(self.stages[index].execution)?;
            self.check_control(limits, cancel)?;
            return Ok(None);
        }
        self.engine
            .semantic
            .stores
            .complete_task(self.stages[index].execution, observation)?;
        self.check_control(limits, cancel)?;
        self.engine.semantic.stores.consume_task(
            self.stages[index].execution,
            consumer.id,
            self.engine.tick_at(Instant::now()),
        )?;
        self.check_control(limits, cancel)?;
        counters.consumed_cpu_tasks += 1;
        self.engine
            .semantic
            .stores
            .dependencies
            .add(observation, self.engine.semantic.nodes[target].situation)?;
        self.engine
            .semantic
            .stores
            .dependencies
            .add(observation, self.engine.semantic.nodes[root].situation)?;
        if self.engine.semantic.stores.generation() != consumer.generation
            || self.engine.semantic.stores.root() != Some(active_root)
            || self
                .engine
                .semantic
                .stores
                .situations
                .get(active_root)?
                .revision
                != root_revision
            || self.endpoint_snapshot(kind).is_none_or(|snapshot| {
                !self.engine.semantic.nodes[target]
                    .position
                    .snapshot()
                    .same_state(snapshot)
            })
        {
            return Err(StoreError::StaleConsumer.into());
        }
        self.check_control(limits, cancel)?;
        let report = self.stages[index]
            .report
            .as_ref()
            .ok_or(PalsError::Capacity)?;
        // Only this completed unrestricted actual observation enters endpoint
        // evidence. Both restricted root observations remain sources only.
        if kind == EndpointKind::Repair {
            self.engine.semantic.nodes[target].evidence = Some(CpuEvidence {
                context: None,
                score: report.score,
                depth: report.completed_depth,
                scope: report.score_scope,
                value_identity: report.value_identity.clone(),
                provenance: Some((observation, self.stages[index].execution)),
            });
        }
        // A fourth observation remains in its own stage/task ledger. In
        // particular, a transposition cannot overwrite the third-stage evidence
        // exposed by repair_endpoint(), or masquerade as an old cached value.
        Ok(Some(observation))
    }

    fn stage(
        &mut self,
        request: StageRequest<'_>,
        counters: &mut PalsCounters,
    ) -> Result<Option<CpuCandidate>, ReplayError> {
        let StageRequest {
            root,
            target,
            phase,
            depth,
            limits,
            cancel,
        } = request;
        self.check_control(limits, cancel)?;
        self.engine.validate_checker_namespace()?;
        if self.engine.researcher.cpu.last_attempt().is_some()
            || self.engine.cpu_condition() != self.engine.researcher.cpu_registered_condition
            || self.engine.researcher.cpu.identity()
                != &self.engine.researcher.checker_registered_identity
        {
            return Err(ReplayError::InvalidPlan(
                "CPU stage is not a fresh registered checker",
            ));
        }
        let descriptor = self.engine.owned_descriptor()?;
        let value_identity = self.engine.owned_identity()?.clone();
        let registered_condition = self.engine.cpu_condition();
        let mut task_condition = String::new();
        task_condition
            .try_reserve_exact(MAX_REPLAY_CONDITION_BYTES)
            .map_err(|_| PalsError::Capacity)?;
        write!(
            &mut task_condition,
            "{};replay={};phase={};question=AnalyzeRootMoves;prefix-plies={};ordered-root-mask=",
            registered_condition,
            FRESH_REPLAY_SCOPE,
            phase.label(),
            self.plan.prefix.len()
        )
        .map_err(|_| PalsError::Capacity)?;
        let mut root_moves = Vec::new();
        root_moves
            .try_reserve_exact(self.effective_order.len())
            .map_err(|_| PalsError::Capacity)?;
        for &movement in &self.effective_order {
            let packed = Move16::pack(movement)?;
            write!(&mut task_condition, "{:04x}", packed.bits())
                .map_err(|_| PalsError::Capacity)?;
            root_moves.push(packed);
        }
        if task_condition.len() > MAX_REPLAY_CONDITION_BYTES {
            return Err(PalsError::Capacity.into());
        }
        let profile = stable_id(descriptor.config.profile.identity());
        let condition = stable_id(&task_condition);
        let generation = self.engine.semantic.stores.generation();
        let active_root = self
            .engine
            .semantic
            .stores
            .root()
            .ok_or(StoreError::StaleConsumer)?;
        let root_revision = self
            .engine
            .semantic
            .stores
            .situations
            .get(active_root)?
            .revision;
        self.engine.semantic.consumer_id = self
            .engine
            .semantic
            .consumer_id
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        let consumer = TaskConsumer {
            id: self.engine.semantic.consumer_id,
            situation: self.engine.semantic.nodes[target].situation,
            revision: self
                .engine
                .semantic
                .stores
                .situations
                .get(self.engine.semantic.nodes[target].situation)?
                .revision,
            generation,
            deadline_tick: self.engine.tick_at(limits.deadline),
        };
        let admission = self.engine.semantic.stores.request_task(
            TaskKey {
                state: self.engine.semantic.nodes[target].state,
                line: None,
                question: TaskQuestion::AnalyzeRootMoves,
                root_moves,
                model: 0,
                epoch: 0,
                value_identity: Some(value_identity.clone()),
                checker_identity: None,
                cpu_condition: Some(task_condition.clone()),
                profile,
                condition,
                input_revision: self.plan.prefix.len() as u64,
                requested_depth: depth,
                node_budget: self.plan.nodes_per_check,
            },
            consumer,
            self.engine.tick_at(Instant::now()),
        )?;
        let TaskAdmission::Start(execution) = admission else {
            return Err(StoreError::InvalidConditions(
                "fresh replay CPU requires Start, never reuse/join/resume",
            )
            .into());
        };
        let index = self.stages.len();
        self.stages.push(ReplayStageReport {
            phase,
            requested_depth: depth,
            node_budget: self.plan.nodes_per_check,
            execution,
            task_condition,
            registered_condition,
            report: None,
            attempt: None,
            observation: None,
            exact_completed: false,
            cleanup_error: None,
        });
        let result = self.stage_started(
            StartedStage {
                index,
                root,
                target,
                consumer,
                active_root,
                root_revision,
                limits,
                cancel,
            },
            counters,
        );
        // Preserve the primary CPU/Rules/control failure even if fail() fails.
        if result.is_err()
            && let Err(error) = self.engine.semantic.stores.tasks.fail(execution)
        {
            self.stages[index].cleanup_error = Some(Box::new(error));
        }
        result
    }

    fn stage_started(
        &mut self,
        context: StartedStage<'_>,
        counters: &mut PalsCounters,
    ) -> Result<Option<CpuCandidate>, ReplayError> {
        let StartedStage {
            index,
            root,
            target,
            consumer,
            active_root,
            root_revision,
            limits,
            cancel,
        } = context;
        self.check_control(limits, cancel)?;
        let cpu_limits = CpuLimits {
            max_depth: self.stages[index].requested_depth,
            max_nodes: self.stages[index].node_budget,
            deadline: Some(limits.deadline),
        };
        self.engine.researcher.cpu.preflight_task(cpu_limits)?;
        self.check_control(limits, cancel)?;
        counters.cpu_tasks_requested += 1;
        let result = self.engine.researcher.cpu.analyze_root_moves(
            &self.engine.semantic.nodes[target].position,
            &self.effective_order,
            cpu_limits,
            cancel,
        );
        // Actual work survives errors and is captured before replacing the CPU.
        self.stages[index].attempt = self.engine.researcher.cpu.last_attempt().cloned();
        let report = match result {
            Ok(report) => match PalsEngine::<M>::own_report(report) {
                Ok(report) => report,
                Err(error) => {
                    self.engine
                        .observe_cpu_failure(counters, cpu_limits.max_nodes);
                    return Err(error.into());
                }
            },
            Err(error) => {
                if matches!(error, CheckerError::OwnedReportRejected { .. }) {
                    counters.cpu_tasks += 1;
                }
                self.engine
                    .observe_cpu_failure(counters, cpu_limits.max_nodes);
                return Err(error.into());
            }
        };
        self.stages[index].report = Some(report);
        let stage = &self.stages[index];
        let report = stage.report.as_ref().ok_or(PalsError::Capacity)?;
        let validation = self
            .engine
            .validate_cpu_report(
                &self.engine.semantic.nodes[target].position,
                report,
                self.engine.owned_identity()?,
                &stage.registered_condition,
                true,
                cpu_limits,
            )
            .and_then(|()| {
                if report
                    .best_move
                    .is_none_or(|movement| !self.effective_order.contains(&movement))
                {
                    Err(StoreError::InvalidEvidence(
                        "replay response outside exact effective order",
                    )
                    .into())
                } else {
                    Ok(())
                }
            });
        // Checked sums precede the existing accounting helper. Rejected raw
        // reports remain accessible even when generic counters are lower bounds.
        if validation.is_ok() {
            counters
                .cpu_nodes
                .checked_add(report.nodes)
                .ok_or(PalsError::Capacity)?;
            counters
                .cpu_quiescence_nodes
                .checked_add(report.quiescence_nodes)
                .ok_or(PalsError::Capacity)?;
            counters
                .cpu_tt_hits
                .checked_add(report.tt_hits)
                .ok_or(PalsError::Capacity)?;
        }
        let exact = validation.is_ok() && exact_coverage(report, cpu_limits.max_depth);
        PalsEngine::<M>::observe_cpu_report(
            counters,
            report,
            cpu_limits.max_depth,
            validation.is_ok(),
        );
        // Generic >=H accounting cannot certify this exact fresh-stage lane.
        if validation.is_ok()
            && !exact
            && report.score_scope == CpuScoreScope::CompletedIteration
            && report.completed_depth >= cpu_limits.max_depth
        {
            counters.completed_cpu_tasks -= 1;
        }
        validation?;
        let pv = bounded_moves(
            &report.pv[..report
                .pv
                .len()
                .min(self.engine.config.line_plies - self.plan.prefix.len())],
            self.engine.config.line_plies - self.plan.prefix.len(),
        )?;
        let pv_line = self
            .engine
            .semantic
            .stores
            .append_cpu_pv(&self.engine.semantic.nodes[target].position, &pv)?;
        let observation = self
            .engine
            .semantic
            .stores
            .append_observation(Observation {
                state: self.engine.semantic.nodes[target].state,
                line: None,
                source: stable_id(report.score_provenance),
                epoch: 0,
                scope: EvidenceScope::DepthLimited {
                    depth: report.completed_depth,
                    profile: stable_id(self.plan.cpu.profile.identity()),
                    condition: stable_id(&stage.task_condition),
                },
                value_identity: Some(report.value_identity.clone()),
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: None,
                model_value_input: None,
                cpu_condition: Some(stage.task_condition.clone()),
                cpu_pv: Some(pv_line),
                score: if report.score_scope == CpuScoreScope::FrontierOnly {
                    RawScore::Estimate {
                        value: report.score as f32,
                        perspective: self.engine.semantic.nodes[target].position.side_to_move(),
                    }
                } else {
                    RawScore::Cpu {
                        value: report.score,
                        perspective: self.engine.semantic.nodes[target].position.side_to_move(),
                        bound: BoundKind::ExactWithinSearch,
                    }
                },
                budget: report.nodes,
                kind: ObservationKind::CpuAnalysis,
                supersedes: None,
                execution: Some(stage.execution),
            })?;
        self.stages[index].observation = Some(observation);
        self.stages[index].exact_completed = exact;
        if !exact {
            self.engine
                .semantic
                .stores
                .tasks
                .fail(self.stages[index].execution)?;
            self.check_control(limits, cancel)?;
            return Ok(None);
        }
        self.engine
            .semantic
            .stores
            .complete_task(self.stages[index].execution, observation)?;
        self.check_control(limits, cancel)?;
        self.engine.semantic.stores.consume_task(
            self.stages[index].execution,
            consumer.id,
            self.engine.tick_at(Instant::now()),
        )?;
        self.check_control(limits, cancel)?;
        counters.consumed_cpu_tasks += 1;
        self.engine
            .semantic
            .stores
            .dependencies
            .add(observation, self.engine.semantic.nodes[target].situation)?;
        self.engine
            .semantic
            .stores
            .dependencies
            .add(observation, self.engine.semantic.nodes[root].situation)?;
        Ok(Some(CpuCandidate {
            pv,
            observation,
            generation: consumer.generation,
            root: active_root,
            root_revision,
            deadline: limits.deadline,
        }))
    }
}

fn validate_extents(
    config: &PalsConfig,
    plan: &DefendResponseReplayPlan,
) -> Result<(), ReplayError> {
    if plan.cpu.profile != CpuProfile::PlanAssisted
        || plan.baseline_depth == 0
        || plan.requested_depth <= plan.baseline_depth
        || plan.requested_depth > 64
        || plan.cpu.max_depth != plan.requested_depth
        || plan.cpu.quiescence_ply > 32
        || plan.cpu.tt_entries > 1_048_576
        || plan.nodes_per_check == 0
        || plan.nodes_per_check > 10_000_000
        || config.cpu_nodes_per_task != plan.nodes_per_check
        || config.max_role_calls < config.line_plies as u64 + 1
        || plan.prefix.len() >= config.line_plies
        || plan.claimed_line.len() > config.line_plies
        || plan.prefix.capacity() > config.line_plies
        || plan.claimed_line.capacity() > config.line_plies
    {
        return Err(ReplayError::InvalidPlan(
            "unsupported profile/horizons/budget/line extent",
        ));
    }
    for moves in [
        &plan.root_legal_order,
        &plan.target_legal_order,
        &plan.response_restriction,
    ] {
        if moves.len() > 256 || moves.capacity() > 256 {
            return Err(ReplayError::InvalidPlan(
                "legal/restriction retained extent exceeds 256",
            ));
        }
    }
    Ok(())
}

fn bounded_moves(input: &[BoardMove], extent: usize) -> Result<Vec<BoardMove>, ReplayError> {
    if input.len() > extent {
        return Err(ReplayError::InvalidPlan("move extent exceeded"));
    }
    let mut moves = Vec::new();
    moves
        .try_reserve_exact(extent)
        .map_err(|_| PalsError::Capacity)?;
    moves.extend_from_slice(input);
    Ok(moves)
}

fn exact_coverage(report: &CpuReport, depth: u16) -> bool {
    report.completion == CpuCompletion::DepthLimit
        && report.score_scope == CpuScoreScope::CompletedIteration
        && report.completed_depth == depth
        && report.reused_completed_depth == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals::store::TaskStatus;

    struct ScriptedRoles {
        proposer_line: Vec<BoardMove>,
        accepted: usize,
        cancel_on_accept: Option<usize>,
        fail_reply: bool,
        saw_after: Option<ObservationId>,
        closure: Option<RoleSearchClosure>,
        finish_calls: usize,
    }
    impl ScriptedRoles {
        fn evaluation(
            query: &RoleQuery<'_>,
            selected: BoardMove,
        ) -> Result<RoleEvaluation, RoleError> {
            let index = query
                .legal
                .iter()
                .position(|movement| *movement == selected)
                .ok_or(RoleError::InvalidOutput)?;
            let mut logits = vec![0.0; query.legal.len()];
            logits[index] = 1.0;
            Ok(RoleEvaluation {
                logits,
                wdl: [0.25, 0.5, 0.25],
            })
        }
    }
    impl RoleModel for ScriptedRoles {
        fn identity(&self) -> &str {
            "synthetic-replay-role-model/1"
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            let movement = *self
                .proposer_line
                .get(query.prefix.len())
                .ok_or(RoleError::InvalidOutput)?;
            Self::evaluation(&query, movement)
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            let record = query
                .records
                .iter()
                .rev()
                .find(|record| record.kind == RecordKind::Counterexample)
                .ok_or(RoleError::InvalidOutput)?;
            if record.value.is_some() || record.completed_depth != 0 || record.score_scope.is_some()
            {
                return Err(RoleError::InvalidOutput);
            }
            self.saw_after = record.cpu_observation;
            if self.fail_reply {
                return Err(RoleError::Backend("synthetic primary Reply failure".into()));
            }
            let movement = *record
                .line
                .get(query.prefix.len())
                .ok_or(RoleError::InvalidOutput)?;
            Self::evaluation(&query, movement)
        }
        fn repair(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            Err(RoleError::Unavailable)
        }
        fn divergences(&mut self, _: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            Err(RoleError::Unavailable)
        }
        fn accepted_output_checked(
            &mut self,
            acceptance: RoleAcceptance<'_>,
        ) -> Result<(), RoleError> {
            acceptance.check_control()?;
            self.accepted += 1;
            if self.cancel_on_accept == Some(self.accepted) {
                acceptance.cancel.store(true, Ordering::Release);
            }
            Ok(())
        }
        fn finish_search(&mut self, reason: RoleSearchClosure) {
            self.finish_calls += 1;
            self.closure = Some(reason);
        }
    }
    fn mv(text: &str) -> BoardMove {
        BoardMove::from_uci(text).unwrap()
    }
    fn fixture(nodes: u64, response: &str) -> FreshReplayOwner<ScriptedRoles> {
        fixture_with_restriction(nodes, vec![mv(response)])
    }
    fn fixture_with_restriction(
        nodes: u64,
        restriction: Vec<BoardMove>,
    ) -> FreshReplayOwner<ScriptedRoles> {
        let root = Position::startpos();
        let mut target = root.clone();
        target.make_move(mv("e2e4")).unwrap();
        let roles = ScriptedRoles {
            proposer_line: vec![mv("e2e4"), mv("e7e5"), mv("g1f3")],
            accepted: 0,
            cancel_on_accept: None,
            fail_reply: false,
            saw_after: None,
            closure: None,
            finish_calls: 0,
        };
        let pins = ReplayHistoricalPins {
            parent_input_sha256: [1; 32],
            current_view_sha256: [2; 32],
            frozen_admission_sha256: [3; 32],
            query_sha256: [4; 32],
            catalogue_sha256: [5; 32],
            before_result_sha256: [6; 32],
            semantic_input_sha256: [7; 32],
            cpu_request_sha256: [8; 32],
        };
        let plan = DefendResponseReplayPlan {
            historical: pins,
            expected_root: root.snapshot(),
            expected_target: target.snapshot(),
            root_legal_order: root.legal_moves(),
            target_legal_order: target.legal_moves(),
            prefix: vec![mv("e2e4")],
            response_restriction: restriction,
            claimed_line: vec![mv("e2e4"), mv("e7e5"), mv("g1f3")],
            baseline_depth: 1,
            requested_depth: 2,
            nodes_per_check: nodes,
            cpu: CpuConfig {
                profile: CpuProfile::PlanAssisted,
                tt_entries: 16,
                max_depth: 2,
                quiescence_ply: 4,
            },
        };
        FreshReplayOwner::new(
            PalsConfig {
                line_plies: 3,
                cpu_nodes_per_task: nodes,
                ..PalsConfig::default()
            },
            roles,
            root,
            plan,
        )
        .unwrap()
    }
    fn limits(nodes: u64) -> PalsLimits {
        PalsLimits {
            deadline: Instant::now() + Duration::from_secs(10),
            max_rounds: 1,
            max_cpu_nodes: nodes * 2,
            cpu_depth: 2,
        }
    }

    #[test]
    fn fresh_two_starts_after_observation_enters_actual_reply() {
        let mut owner = fixture(100_000, "c7c5");
        let result = owner.run(limits(100_000), &AtomicBool::new(false)).unwrap();
        let ReplayOutcome::ReplyAccepted {
            after_observation,
            publication_revision,
            ..
        } = result
        else {
            panic!("{result:?}");
        };
        assert_eq!(owner.stages.len(), 2);
        assert_ne!(owner.stages[0].execution, owner.stages[1].execution);
        assert_ne!(
            owner.stages[0].task_condition,
            owner.stages[1].task_condition
        );
        assert_eq!(
            owner.stages[0].registered_condition,
            owner.stages[1].registered_condition
        );
        for stage in &owner.stages {
            assert!(stage.exact_completed);
            assert_eq!(stage.report.as_ref().unwrap().reused_completed_depth, 0);
            assert_eq!(owner.task(stage.execution).unwrap().resumed_from, None);
            assert_eq!(
                owner.task(stage.execution).unwrap().status,
                TaskStatus::Completed(stage.observation.unwrap())
            );
        }
        assert_eq!(
            owner.engine.semantic.model.saw_after,
            Some(after_observation)
        );
        assert_ne!(Some(after_observation), owner.stages[0].observation);
        let context = owner.reply_accepted_context().unwrap();
        assert_eq!(context.public_revision, publication_revision);
        assert_eq!(context.purpose, RoleQueryPurpose::ReplyPolicy);
        assert_eq!(owner.counters.cpu_tasks_requested, 2);
        assert_eq!(owner.counters.reused_completed_cpu_tasks_consumed, 0);
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
        assert_eq!(
            owner.engine.semantic.model.closure,
            Some(RoleSearchClosure::Completed)
        );
        assert!(!owner.strict_query_admission_verified() && !owner.native_freshness_verified());
        assert!(!owner.whole_cost_verified() && !owner.physical_closure_verified());
        assert!(!owner.utility_authority() && !owner.training_target_created());
    }
    #[test]
    fn duplicate_owner_does_not_dispatch_again() {
        let mut owner = fixture(100_000, "c7c5");
        let cancel = AtomicBool::new(false);
        owner.run(limits(100_000), &cancel).unwrap();
        let before = owner.counters;
        assert!(matches!(
            owner.run(limits(100_000), &cancel),
            Err(ReplayError::AlreadyUsed)
        ));
        assert_eq!(owner.counters, before);
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
    }
    #[test]
    fn fixed_two_n_refused_before_roles_and_cpu() {
        let mut owner = fixture(100_000, "c7c5");
        let mut bound = limits(100_000);
        bound.max_cpu_nodes -= 1;
        assert!(matches!(
            owner.run(bound, &AtomicBool::new(false)),
            Err(ReplayError::InvalidPlan(_))
        ));
        assert_eq!(owner.counters.role_calls, 0);
        assert!(owner.stages.is_empty());
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
        assert_eq!(
            owner.engine.semantic.model.closure,
            Some(RoleSearchClosure::Failed)
        );
    }
    #[test]
    fn partial_report_is_retained_without_reply_or_counterexample() {
        let mut owner = fixture(1, "c7c5");
        assert_eq!(
            owner.run(limits(1), &AtomicBool::new(false)).unwrap(),
            ReplayOutcome::Partial(ReplayCpuPhase::Baseline)
        );
        assert_eq!(owner.stages.len(), 1);
        assert!(owner.stages[0].report.is_some() && owner.stages[0].attempt.is_some());
        assert!(!owner.stages[0].exact_completed);
        assert_eq!(owner.counters.cpu_tasks_requested, 1);
        assert_eq!(owner.counters.critic_calls, 0);
        assert!(
            !owner
                .records()
                .iter()
                .any(|record| record.kind == RecordKind::Counterexample)
        );
    }
    #[test]
    fn after_partial_retains_the_completed_baseline_report_and_work() {
        // One quiet restricted response completes H0, while H1 must examine
        // several legal White continuations within the same exact small N.
        let mut owner = fixture(8, "c7c5");
        let result = owner.run(limits(8), &AtomicBool::new(false)).unwrap();
        assert_eq!(result, ReplayOutcome::Partial(ReplayCpuPhase::After));
        assert_eq!(owner.stages.len(), 2);
        assert!(owner.stages[0].exact_completed);
        assert!(owner.stages[0].report.is_some() && owner.stages[0].attempt.is_some());
        assert_eq!(owner.stages[0].report.as_ref().unwrap().completed_depth, 1);
        assert_eq!(
            owner.task(owner.stages[0].execution).unwrap().status,
            TaskStatus::Completed(owner.stages[0].observation.unwrap())
        );
        assert!(owner.stages[1].report.is_some() && owner.stages[1].attempt.is_some());
        assert!(!owner.stages[1].exact_completed);
        assert_eq!(
            owner.task(owner.stages[1].execution).unwrap().status,
            TaskStatus::Failed
        );
        assert_eq!(owner.counters.cpu_tasks_requested, 2);
        assert_eq!(owner.publication_revision(), None);
        assert_eq!(owner.counters.critic_calls, 0);
    }
    #[test]
    fn same_proposal_response_is_unapplied() {
        let mut owner = fixture(100_000, "e7e5");
        assert_eq!(
            owner.run(limits(100_000), &AtomicBool::new(false)).unwrap(),
            ReplayOutcome::SameProposalResponse
        );
        assert_eq!(owner.stages.len(), 2);
        assert_eq!(owner.counters.critic_calls, 0);
        assert_eq!(owner.publication_revision(), None);
    }
    #[test]
    fn late_proposer_acceptance_cancellation_prevents_cpu() {
        let mut owner = fixture(100_000, "c7c5");
        owner.engine.semantic.model.cancel_on_accept = Some(1);
        let result = owner.run(limits(100_000), &AtomicBool::new(false));
        assert!(
            matches!(result, Err(ReplayError::Search(error)) if matches!(*error, PalsError::Role(RoleError::Canceled)))
        );
        assert_eq!(owner.counters.cpu_tasks_requested, 0);
        assert!(owner.stages.is_empty());
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
        assert_eq!(
            owner.engine.semantic.model.closure,
            Some(RoleSearchClosure::Canceled)
        );
    }
    #[test]
    fn late_reply_acceptance_cancellation_preserves_actual_work_but_refuses_success() {
        let mut owner = fixture(100_000, "c7c5");
        owner.engine.semantic.model.cancel_on_accept = Some(4);
        let result = owner.run(limits(100_000), &AtomicBool::new(false));
        assert!(
            matches!(result, Err(ReplayError::Search(error)) if matches!(*error, PalsError::Role(RoleError::Canceled)))
        );
        assert_eq!(owner.stages.len(), 2);
        assert!(owner.reply_accepted_context().is_some());
        assert!(owner.publication_revision().is_some());
        assert!(owner.reply_line().is_empty());
    }
    #[test]
    fn primary_reply_failure_and_raw_cpu_work_survive() {
        let mut owner = fixture(100_000, "c7c5");
        owner.engine.semantic.model.fail_reply = true;
        let result = owner.run(limits(100_000), &AtomicBool::new(false));
        assert!(
            matches!(result, Err(ReplayError::Search(error)) if matches!(*error, PalsError::Role(RoleError::Backend(_))))
        );
        assert_eq!(owner.stages.len(), 2);
        assert!(
            owner
                .stages
                .iter()
                .all(|stage| stage.report.is_some() && stage.attempt.is_some())
        );
        assert!(owner.reply_prepared_context().is_some());
        assert!(owner.reply_accepted_context().is_none());
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
        assert_eq!(
            owner.engine.semantic.model.closure,
            Some(RoleSearchClosure::Failed)
        );
    }
    #[test]
    fn expired_original_deadline_has_no_dispatch() {
        let mut owner = fixture(100_000, "c7c5");
        let mut bound = limits(100_000);
        bound.deadline = Instant::now();
        assert!(
            matches!(owner.run(bound, &AtomicBool::new(false)), Err(ReplayError::Search(error)) if matches!(*error, PalsError::Role(RoleError::Deadline)))
        );
        assert_eq!(owner.counters.role_calls, 0);
        assert!(owner.stages.is_empty());
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
        assert_eq!(
            owner.engine.semantic.model.closure,
            Some(RoleSearchClosure::Deadline)
        );
    }
    #[test]
    fn actual_proposal_mismatch_keeps_seed_out_of_publication() {
        let mut owner = fixture(100_000, "c7c5");
        owner.engine.semantic.model.proposer_line = vec![mv("d2d4"), mv("d7d5"), mv("c1f4")];
        assert_eq!(
            owner.run(limits(100_000), &AtomicBool::new(false)).unwrap(),
            ReplayOutcome::SeedNotApplied
        );
        assert_eq!(owner.proposal()[0], mv("d2d4"));
        assert_eq!(owner.counters.cpu_tasks_requested, 0);
        assert_eq!(owner.records().len(), 1);
        assert_eq!(owner.records()[0].line[0], mv("d2d4"));
    }
    #[test]
    fn original_restriction_order_is_not_effective_rules_order() {
        let mut target = Position::startpos();
        target.make_move(mv("e2e4")).unwrap();
        let order = target.legal_moves();
        let owner = fixture_with_restriction(100_000, vec![order[1], order[0]]);
        assert_eq!(owner.original_restriction(), &[order[1], order[0]]);
        assert_eq!(owner.effective_order(), &[order[0], order[1]]);
    }
    #[test]
    fn even_a_fresh_tt_cannot_reuse_a_completed_same_phase_task() {
        let mut owner = fixture(100_000, "c7c5");
        let cancel = AtomicBool::new(false);
        let bound = limits(100_000);
        owner.run(bound, &cancel).unwrap();
        // Internal guard fixture: replace only the checker, retaining the real
        // completed baseline key. Public callers cannot mutate the owner.
        owner.engine.researcher.cpu = Box::new(
            OwnedCpuChecker::new(CpuEngine::new(owner.plan.cpu.clone()).unwrap()).unwrap(),
        );
        let root = owner
            .engine
            .semantic
            .nodes
            .iter()
            .position(|node| node.position.snapshot().same_state(&owner.root.snapshot()))
            .unwrap();
        let target = owner
            .engine
            .semantic
            .nodes
            .iter()
            .position(|node| node.position.snapshot().same_state(&owner.target))
            .unwrap();
        let mut counters = owner.counters;
        let before = counters.cpu_tasks_requested;
        let result = owner.stage(
            StageRequest {
                root,
                target,
                phase: ReplayCpuPhase::Baseline,
                depth: 1,
                limits: bound,
                cancel: &cancel,
            },
            &mut counters,
        );
        assert!(
            matches!(result, Err(ReplayError::Search(error)) if matches!(*error, PalsError::Store(StoreError::InvalidConditions(_))))
        );
        assert_eq!(counters.cpu_tasks_requested, before);
        assert_eq!(owner.stages.len(), 2);
    }
}

#[cfg(test)]
mod repair_tests {
    use super::*;
    use crate::pals::store::TaskStatus;

    /// Synthetic RoleModel source fixture only. It does not attest native NN
    /// submission, Query admission, physical cleanup or whole invocation cost.
    struct RepairRoles {
        proposal: Vec<BoardMove>,
        response: BoardMove,
        counter_suffix: BoardMove,
        repair_suffix: BoardMove,
        accepted: usize,
        cancel_on_accept: Option<usize>,
        reject_repair_acceptance: bool,
        fail_repair: bool,
        repair_physical_unknown: bool,
        initial_cpu_source: Option<ObservationId>,
        saw_model_counterline: bool,
        finish_calls: usize,
        closure: Option<RoleSearchClosure>,
    }
    impl RepairRoles {
        fn evaluate(
            query: &RoleQuery<'_>,
            selected: BoardMove,
        ) -> Result<RoleEvaluation, RoleError> {
            let index = query
                .legal
                .iter()
                .position(|movement| *movement == selected)
                .ok_or(RoleError::InvalidOutput)?;
            let mut logits = vec![0.0; query.legal.len()];
            logits[index] = 1.0;
            Ok(RoleEvaluation {
                logits,
                wdl: [0.25, 0.5, 0.25],
            })
        }
    }
    impl RoleModel for RepairRoles {
        fn identity(&self) -> &str {
            "synthetic-replay-repair-model/1"
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            let movement = *self
                .proposal
                .get(query.prefix.len())
                .ok_or(RoleError::InvalidOutput)?;
            Self::evaluate(&query, movement)
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            let initial = query
                .records
                .iter()
                .rev()
                .find(|record| record.kind == RecordKind::Counterexample)
                .ok_or(RoleError::InvalidOutput)?;
            if initial.value.is_some()
                || initial.completed_depth != 0
                || initial.score_scope.is_some()
            {
                return Err(RoleError::InvalidOutput);
            }
            self.initial_cpu_source = initial.cpu_observation;
            Self::evaluate(
                &query,
                if query.prefix.len() == 1 {
                    self.response
                } else {
                    self.counter_suffix
                },
            )
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            let record = query
                .records
                .iter()
                .rev()
                .find(|record| record.kind == RecordKind::Counterexample)
                .ok_or(RoleError::InvalidOutput)?;
            if record.cpu_observation.is_some()
                || record.value.is_some()
                || record.completed_depth != 0
                || record.score_scope.is_some()
                || query.counterexample != Some(record.line.as_slice())
                || record.line.get(1) != Some(&self.response)
                || record.line.get(2) != Some(&self.counter_suffix)
            {
                return Err(RoleError::InvalidOutput);
            }
            self.saw_model_counterline = true;
            if self.repair_physical_unknown {
                return Err(RoleError::PhysicalCompletionUnknown);
            }
            if self.fail_repair {
                return Err(RoleError::Backend(
                    "synthetic primary Repair failure".into(),
                ));
            }
            Self::evaluate(&query, self.repair_suffix)
        }
        fn divergences(&mut self, _: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            Err(RoleError::Unavailable)
        }
        fn accepted_output_checked(
            &mut self,
            acceptance: RoleAcceptance<'_>,
        ) -> Result<(), RoleError> {
            acceptance.check_control()?;
            if self.reject_repair_acceptance
                && acceptance.context.purpose == RoleQueryPurpose::RepairPolicy
            {
                return Err(RoleError::Backend(
                    "synthetic Repair acceptance rejected".into(),
                ));
            }
            self.accepted += 1;
            if self.cancel_on_accept == Some(self.accepted) {
                acceptance.cancel.store(true, Ordering::Release);
            }
            Ok(())
        }
        fn finish_search(&mut self, reason: RoleSearchClosure) {
            self.finish_calls += 1;
            self.closure = Some(reason);
        }
    }
    fn mv(text: &str) -> BoardMove {
        BoardMove::from_uci(text).unwrap()
    }
    fn fixture(nodes: u64) -> FreshReplayOwner<RepairRoles> {
        from_root(Position::startpos(), nodes, false)
    }
    fn from_root(
        root: Position,
        nodes: u64,
        terminal_repair: bool,
    ) -> FreshReplayOwner<RepairRoles> {
        from_root_with_extent(root, nodes, terminal_repair, 3)
    }
    fn from_root_with_extent(
        root: Position,
        nodes: u64,
        terminal_repair: bool,
        extent: usize,
    ) -> FreshReplayOwner<RepairRoles> {
        let (proposal, restriction, response, counter_suffix, repair_suffix) = if terminal_repair {
            (
                vec![mv("e7e5"), mv("e2e4"), mv("b8c6")],
                mv("g2g4"),
                mv("g2g4"),
                mv("a7a6"),
                mv("d8h4"),
            )
        } else {
            (
                vec![mv("e2e4"), mv("e7e5"), mv("g1f3")],
                mv("c7c5"),
                mv("e7e6"),
                mv("d2d4"),
                mv("d2d3"),
            )
        };
        let proposal = bounded_moves(&proposal[..extent], extent).unwrap();
        let prefix = vec![proposal[0]];
        let mut target = root.clone();
        target.make_move(prefix[0]).unwrap();
        let roles = RepairRoles {
            proposal: proposal.clone(),
            response,
            counter_suffix,
            repair_suffix,
            accepted: 0,
            cancel_on_accept: None,
            reject_repair_acceptance: false,
            fail_repair: false,
            repair_physical_unknown: false,
            initial_cpu_source: None,
            saw_model_counterline: false,
            finish_calls: 0,
            closure: None,
        };
        let plan = DefendResponseReplayPlan {
            historical: ReplayHistoricalPins {
                parent_input_sha256: [1; 32],
                current_view_sha256: [2; 32],
                frozen_admission_sha256: [3; 32],
                query_sha256: [4; 32],
                catalogue_sha256: [5; 32],
                before_result_sha256: [6; 32],
                semantic_input_sha256: [7; 32],
                cpu_request_sha256: [8; 32],
            },
            expected_root: root.snapshot(),
            expected_target: target.snapshot(),
            root_legal_order: root.legal_moves(),
            target_legal_order: target.legal_moves(),
            prefix,
            response_restriction: vec![restriction],
            claimed_line: proposal,
            baseline_depth: 1,
            requested_depth: 2,
            nodes_per_check: nodes,
            cpu: CpuConfig {
                profile: CpuProfile::PlanAssisted,
                tt_entries: 16,
                max_depth: 2,
                quiescence_ply: 4,
            },
        };
        FreshReplayOwner::new(
            PalsConfig {
                line_plies: extent,
                max_records: 4,
                cpu_nodes_per_task: nodes,
                ..PalsConfig::default()
            },
            roles,
            root,
            plan,
        )
        .unwrap()
    }
    fn limits(nodes: u64) -> PalsLimits {
        PalsLimits {
            deadline: Instant::now() + Duration::from_secs(10),
            max_rounds: 1,
            max_cpu_nodes: nodes * 3,
            cpu_depth: 2,
        }
    }

    #[test]
    fn actual_repair_endpoint_is_third_fresh_unrestricted_completed_task() {
        let mut owner = fixture(100_000);
        let result = owner
            .run_with_repair(limits(100_000), &AtomicBool::new(false))
            .unwrap();
        let ReplayRepairOutcome::CompletedRepairEndpoint {
            repaired_line,
            repair_record_revision,
            endpoint_state,
            execution,
            observation,
            ..
        } = result
        else {
            panic!("{result:?}");
        };
        assert_eq!(owner.repair_line_id(), Some(repaired_line));
        assert_eq!(owner.repair_record_revision(), Some(repair_record_revision));
        assert_eq!(owner.stages.len(), 3);
        let endpoint = &owner.stages[2];
        assert_eq!(endpoint.phase, ReplayCpuPhase::RepairEndpoint);
        assert!(endpoint.exact_completed && endpoint.attempt.is_some());
        assert_eq!(endpoint.execution, execution);
        assert_eq!(endpoint.observation, Some(observation));
        assert_eq!(endpoint.report.as_ref().unwrap().reused_completed_depth, 0);
        assert_ne!(execution, owner.stages[0].execution);
        assert_ne!(execution, owner.stages[1].execution);
        assert_eq!(
            endpoint.registered_condition,
            owner.stages[1].registered_condition
        );
        assert!(endpoint.task_condition.contains(FRESH_REPLAY_REPAIR_SCOPE));
        let task = owner.task(execution).unwrap();
        assert_eq!(task.key.question, TaskQuestion::AnalyzePosition);
        assert!(task.key.root_moves.is_empty());
        assert_eq!(task.key.input_revision, 0);
        assert_eq!(task.key.state, endpoint_state);
        assert_eq!(task.status, TaskStatus::Completed(observation));
        assert_eq!(task.resumed_from, None);
        let facts = owner.repair_endpoint().unwrap();
        assert_eq!(facts.snapshot.side_to_move(), Color::Black);
        assert!(matches!(facts.evidence, RecheckEndpointEvidence::OwnCpu {
            observation_id, execution_id, ..
        } if observation_id == observation && execution_id == execution));
        let raw = owner.observation(observation).unwrap();
        assert_eq!(
            raw.score,
            RawScore::Cpu {
                value: endpoint.report.as_ref().unwrap().score,
                perspective: Color::Black,
                bound: BoundKind::ExactWithinSearch,
            }
        );
        assert_eq!(owner.counters.cpu_tasks_requested, 3);
        assert_eq!(owner.counters.supported_repairs, 0);
        assert_eq!(owner.counters.supported_refutations, 0);
        assert!(owner.engine.semantic.nodes.iter().all(|node| {
            owner
                .engine
                .semantic
                .stores
                .situations
                .get(node.situation)
                .unwrap()
                .conclusions
                .iter()
                .next()
                .is_none()
        }));
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
        assert!(!owner.strict_query_admission_verified() && !owner.native_freshness_verified());
        assert!(!owner.whole_cost_verified() && !owner.physical_closure_verified());
        assert!(!owner.utility_authority() && !owner.training_target_created());
    }

    #[test]
    fn selected_reply_differs_from_cpu_pv_and_model_line_has_no_cpu_source_or_score() {
        let mut owner = fixture(100_000);
        owner
            .run_with_repair(limits(100_000), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(owner.stages[1].report.as_ref().unwrap().pv[0], mv("c7c5"));
        assert_eq!(owner.reply_line(), &[mv("e2e4"), mv("e7e6")]);
        assert_eq!(
            owner.model_counterline(),
            &[mv("e2e4"), mv("e7e6"), mv("d2d4")]
        );
        assert_eq!(owner.repaired_line(), &[mv("e2e4"), mv("e7e6"), mv("d2d3")]);
        assert!(owner.engine.semantic.model.saw_model_counterline);
        assert_eq!(
            owner.engine.semantic.model.initial_cpu_source,
            owner.stages[1].observation
        );
        assert_eq!(owner.records().len(), 4);
        let model_counter = owner
            .records()
            .iter()
            .rev()
            .find(|record| record.kind == RecordKind::Counterexample)
            .unwrap();
        assert_eq!(model_counter.line, owner.model_counterline());
        let repair = owner
            .records()
            .iter()
            .find(|record| record.kind == RecordKind::Repair)
            .unwrap();
        for record in [model_counter, repair] {
            assert_eq!(record.cpu_observation, None);
            assert_eq!(record.value, None);
            assert_eq!(record.completed_depth, 0);
            assert_eq!(record.score_scope, None);
        }
        let target = owner
            .engine
            .semantic
            .nodes
            .iter()
            .find(|node| node.position.snapshot().same_state(&owner.target))
            .unwrap();
        assert!(target.evidence.is_none());
    }

    #[test]
    fn fixed_three_n_and_role_reservations_refuse_before_dispatch() {
        let mut owner = fixture(100_000);
        let mut bound = limits(100_000);
        bound.max_cpu_nodes -= 1;
        assert!(matches!(
            owner.run_with_repair(bound, &AtomicBool::new(false)),
            Err(ReplayError::InvalidPlan(_))
        ));
        assert_eq!(owner.counters.role_calls, 0);
        assert!(owner.stages.is_empty());
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
        let mut owner = fixture(100_000);
        owner.engine.config.max_role_calls = 5; // L+1+2*(L-P-1) is six.
        assert!(matches!(
            owner.run_with_repair(limits(100_000), &AtomicBool::new(false)),
            Err(ReplayError::InvalidPlan(_))
        ));
        assert_eq!(owner.counters.role_calls, 0);
        assert!(owner.stages.is_empty());
    }

    #[test]
    fn reply_only_stays_two_checks_and_duplicate_other_mode_has_no_finish_hook() {
        let mut owner = fixture(100_000);
        let cancel = AtomicBool::new(false);
        assert!(matches!(
            owner.run(limits(100_000), &cancel).unwrap(),
            ReplayOutcome::ReplyAccepted { .. }
        ));
        assert_eq!(owner.stages.len(), 2);
        assert_eq!(owner.counters.repair_calls, 0);
        assert!(owner.repaired_line().is_empty() && owner.model_counterline().is_empty());
        assert!(matches!(
            owner.run_with_repair(limits(100_000), &cancel),
            Err(ReplayError::AlreadyUsed)
        ));
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
        let mut owner = fixture(100_000);
        owner.run_with_repair(limits(100_000), &cancel).unwrap();
        assert!(matches!(
            owner.run(limits(100_000), &cancel),
            Err(ReplayError::AlreadyUsed)
        ));
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
    }

    #[test]
    fn failed_or_rejected_repair_keeps_h0_h1_and_never_dispatches_endpoint() {
        for reject_acceptance in [false, true] {
            let mut owner = fixture(100_000);
            owner.engine.semantic.model.fail_repair = !reject_acceptance;
            owner.engine.semantic.model.reject_repair_acceptance = reject_acceptance;
            let error = owner.run_with_repair(limits(100_000), &AtomicBool::new(false));
            assert!(
                matches!(error, Err(ReplayError::Search(error)) if matches!(*error, PalsError::Role(RoleError::Backend(_))))
            );
            assert_eq!(owner.stages.len(), 2);
            assert!(owner.stages.iter().all(|stage| stage.report.is_some()
                && stage.attempt.is_some()
                && stage.exact_completed));
            assert_eq!(owner.counters.accepted_repair_outputs, 0);
            assert!(
                !owner
                    .records()
                    .iter()
                    .any(|record| record.kind == RecordKind::Repair)
            );
            assert_eq!(owner.repair_record_revision(), None);
            assert_eq!(owner.engine.semantic.model.finish_calls, 1);
            assert_eq!(
                owner.engine.semantic.model.closure,
                Some(RoleSearchClosure::Failed)
            );
        }
    }

    #[test]
    fn late_repair_acceptance_cancel_preserves_partial_line_but_no_record_or_endpoint() {
        let mut owner = fixture(100_000);
        owner.engine.semantic.model.cancel_on_accept = Some(6);
        let error = owner.run_with_repair(limits(100_000), &AtomicBool::new(false));
        assert!(
            matches!(error, Err(ReplayError::Search(error)) if matches!(*error, PalsError::Role(RoleError::Canceled)))
        );
        assert_eq!(owner.counters.accepted_repair_outputs, 1);
        assert_eq!(owner.counters.completed_repair_calls, 1);
        assert_eq!(owner.repaired_line().len(), 3);
        assert_eq!(owner.stages.len(), 2);
        // The completed legal movement remains a local raw prefix. It grants
        // no graph edge, Repair record/revision or endpoint investigation.
        let mut checked = owner.root.clone();
        for &movement in owner.repaired_line() {
            checked.make_move(movement).unwrap();
        }
        assert!(
            !owner
                .engine
                .semantic
                .nodes
                .iter()
                .any(|node| node.position.snapshot().same_state(&checked.snapshot()))
        );
        assert!(
            !owner
                .records()
                .iter()
                .any(|record| record.kind == RecordKind::Repair)
        );
        assert_eq!(owner.repair_endpoint_node, None);
        assert!(owner.repair_endpoint().is_none());
        for stage in owner.stages() {
            let observation = stage.observation().unwrap();
            assert!(stage.report().is_some());
            assert!(stage.exact_completed());
            assert_eq!(
                owner
                    .engine
                    .semantic
                    .stores
                    .tasks
                    .get(stage.execution())
                    .unwrap()
                    .status,
                TaskStatus::Completed(observation)
            );
            assert_eq!(
                owner
                    .engine
                    .semantic
                    .stores
                    .observations
                    .get(observation)
                    .unwrap()
                    .execution,
                Some(stage.execution())
            );
        }
        assert_eq!(owner.repair_record_revision(), None);
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
        assert_eq!(
            owner.engine.semantic.model.closure,
            Some(RoleSearchClosure::Canceled)
        );
    }

    #[test]
    fn fresh_endpoint_partial_retains_raw_work_without_completed_node_evidence() {
        // Quiet e4/c5 has one restricted Black root for H0/H1; the repaired
        // e4/e6/d3 endpoint must examine the full Black root order at H1. The
        // finite budgets locate that difference with real work under identical
        // TT16/q4/H1 configuration, not an injected report or budget downgrade.
        let mut observed_partial = false;
        for nodes in [32, 64, 128, 256] {
            let mut owner = fixture(nodes);
            let result = owner
                .run_with_repair(limits(nodes), &AtomicBool::new(false))
                .unwrap();
            if let ReplayRepairOutcome::PartialRepairEndpoint {
                execution,
                observation,
            } = result
            {
                observed_partial = true;
                let stage = &owner.stages[2];
                assert!(stage.report.is_some() && stage.attempt.is_some());
                assert!(!stage.exact_completed);
                assert_eq!(stage.node_budget, nodes);
                assert_eq!(stage.observation, observation);
                assert_eq!(owner.task(execution).unwrap().status, TaskStatus::Failed);
                assert!(owner.stages[..2].iter().all(|stage| stage.exact_completed));
                assert!(
                    owner.stages[..2].iter().all(|stage| stage
                        .report
                        .as_ref()
                        .unwrap()
                        .root_restricted)
                );
                assert!(!stage.report.as_ref().unwrap().root_restricted);
                assert!(matches!(
                    owner.repair_endpoint().unwrap().evidence,
                    RecheckEndpointEvidence::Unobserved
                ));
                assert!(
                    owner.engine.semantic.nodes[owner.repair_endpoint_node.unwrap()]
                        .evidence
                        .is_none()
                );
                assert_eq!(owner.engine.semantic.model.finish_calls, 1);
                break;
            }
        }
        assert!(
            observed_partial,
            "no actual endpoint-partial boundary in finite fixture budgets"
        );
    }

    #[test]
    fn restricted_root_observation_cannot_be_promoted_to_endpoint_provenance() {
        let mut owner = fixture(100_000);
        owner.run(limits(100_000), &AtomicBool::new(false)).unwrap();
        let target = owner
            .engine
            .semantic
            .nodes
            .iter()
            .position(|node| node.position.snapshot().same_state(&owner.target))
            .unwrap();
        let stage = &owner.stages[1];
        let report = stage.report.as_ref().unwrap();
        // Internal adversarial fixture only: a matching actual raw score/state
        // still has AnalyzeRootMoves/restricted namespace and must be refused.
        owner.engine.semantic.nodes[target].evidence = Some(CpuEvidence {
            context: None,
            score: report.score,
            depth: report.completed_depth,
            scope: report.score_scope,
            value_identity: report.value_identity.clone(),
            provenance: Some((stage.observation.unwrap(), stage.execution)),
        });
        let facts = PalsEngine::<RepairRoles>::recheck_endpoint(
            &owner.engine.semantic.nodes[target],
            &owner.engine.semantic.stores,
            &owner.target,
        );
        assert!(matches!(
            facts.evidence,
            RecheckEndpointEvidence::InvalidProvenance { .. }
        ));
    }

    #[test]
    fn terminal_repair_endpoint_is_rules_only_and_has_two_cpu_checks() {
        let mut root = Position::startpos();
        root.make_move(mv("f2f3")).unwrap();
        let mut owner = from_root(root, 100_000, true);
        let result = owner
            .run_with_repair(limits(100_000), &AtomicBool::new(false))
            .unwrap();
        assert!(matches!(
            result,
            ReplayRepairOutcome::RulesTerminalRepairEndpoint { .. }
        ));
        assert_eq!(owner.repaired_line(), &[mv("e7e5"), mv("g2g4"), mv("d8h4")]);
        assert_eq!(owner.stages.len(), 2);
        assert_eq!(owner.counters.cpu_tasks_requested, 2);
        assert!(matches!(
            owner.repair_endpoint().unwrap().evidence,
            RecheckEndpointEvidence::RulesTerminal { .. }
        ));
        assert_eq!(owner.counters.supported_repairs, 0);
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
    }

    #[test]
    fn full_reply_prefix_without_any_repair_acceptance_is_not_completed_repair() {
        let mut owner = from_root_with_extent(Position::startpos(), 100_000, false, 2);
        let result = owner
            .run_with_repair(limits(100_000), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(result, ReplayRepairOutcome::NoAcceptedRepair);
        assert_eq!(owner.counters.accepted_repair_outputs, 0);
        assert_eq!(owner.repair_record_revision(), None);
        assert_eq!(owner.stages.len(), 2);
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
    }

    #[test]
    fn repair_physical_unknown_keeps_primary_reason_and_stops_before_endpoint() {
        let mut owner = fixture(100_000);
        owner.engine.semantic.model.repair_physical_unknown = true;
        let result = owner.run_with_repair(limits(100_000), &AtomicBool::new(false));
        assert!(
            matches!(result, Err(ReplayError::Search(error)) if matches!(*error, PalsError::Role(RoleError::PhysicalCompletionUnknown)))
        );
        assert_eq!(owner.stages.len(), 2);
        assert_eq!(owner.counters.accepted_repair_outputs, 0);
        assert_eq!(owner.counters.repair_calls, 1);
        assert_eq!(owner.counters.cpu_tasks_requested, 2);
        assert_eq!(owner.repair_record_revision(), None);
        assert_eq!(owner.engine.semantic.model.finish_calls, 1);
        assert_eq!(
            owner.engine.semantic.model.closure,
            Some(RoleSearchClosure::PhysicalCompletionUnknown)
        );
    }

    #[test]
    fn fresh_endpoint_tt_cannot_reuse_the_completed_endpoint_task() {
        let mut owner = fixture(100_000);
        let cancel = AtomicBool::new(false);
        let bound = limits(100_000);
        owner.run_with_repair(bound, &cancel).unwrap();
        // Internal guard fixture keeps the immutable completed task while
        // temporarily detaching only the public stage ledger/Node projection.
        let retained_stage = owner.stages.pop().unwrap();
        let endpoint = owner.repair_endpoint_node.unwrap();
        let root_situation = owner.engine.semantic.stores.root().unwrap();
        let root = owner
            .engine
            .semantic
            .nodes
            .iter()
            .position(|node| node.situation == root_situation)
            .unwrap();
        owner.engine.semantic.nodes[endpoint].evidence = None;
        owner.engine.researcher.cpu = Box::new(
            OwnedCpuChecker::new(CpuEngine::new(owner.plan.cpu.clone()).unwrap()).unwrap(),
        );
        let mut counters = owner.counters;
        let before = counters.cpu_tasks_requested;
        let result = owner.repair_endpoint_stage(
            StageRequest {
                root,
                target: endpoint,
                phase: ReplayCpuPhase::RepairEndpoint,
                depth: 2,
                limits: bound,
                cancel: &cancel,
            },
            &mut counters,
        );
        assert!(
            matches!(result, Err(ReplayError::Search(error)) if matches!(*error, PalsError::Store(StoreError::InvalidConditions(_))))
        );
        assert_eq!(counters.cpu_tasks_requested, before);
        assert_eq!(owner.stages.len(), 2);
        assert_eq!(
            owner.task(retained_stage.execution).unwrap().status,
            TaskStatus::Completed(retained_stage.observation.unwrap())
        );
    }

    #[test]
    fn repair_requirements_exact_formulas_and_short_tail_boundaries() {
        let declared = repair_replay_requirements(3, 1, 100_000).unwrap();
        assert_eq!(declared.cpu_nodes(), 300_000);
        assert_eq!(declared.role_calls(), 6);
        assert_eq!(declared.nodes(), 9);
        assert_eq!(declared.observations(), 15);
        assert_eq!(declared.line_chunks(), 51);
        assert_eq!(declared.records(), 4);
        assert_eq!(declared.stages(), 3);
        let shortest = repair_replay_requirements(1, 0, 1).unwrap();
        assert_eq!(shortest.cpu_nodes(), 3);
        assert_eq!(shortest.role_calls(), 2);
        assert_eq!(shortest.nodes(), 4);
        assert_eq!(shortest.observations(), 10);
        assert_eq!(shortest.line_chunks(), 13);
        let no_tail = repair_replay_requirements(16, 15, 1).unwrap();
        assert_eq!(no_tail.role_calls(), 17);
        assert_eq!(no_tail.nodes(), 19);
        assert_eq!(no_tail.observations(), 25);
        assert_eq!(no_tail.line_chunks(), 403);
        // Pure declaration only: product's configured N ceiling is a separate
        // constructor gate. This is the exact non-overflowing u64 3N boundary.
        assert_eq!(
            repair_replay_requirements(1, 0, u64::MAX / 3)
                .unwrap()
                .cpu_nodes(),
            u64::MAX
        );
    }

    #[test]
    fn repair_requirements_invalid_scalars_are_static_errors() {
        for (line, prefix, nodes) in [(0, 0, 1), (3, 3, 1), (3, 4, 1), (3, 1, 0)] {
            assert!(matches!(
                repair_replay_requirements(line, prefix, nodes),
                Err(ReplayError::InvalidPlan("invalid Repair L/P/N"))
            ));
        }
    }

    #[test]
    fn repair_requirements_checked_overflow_does_not_return_wrapped_or_allocating_error() {
        assert!(matches!(
            repair_replay_requirements(3, 1, u64::MAX / 3 + 1),
            Err(ReplayError::InvalidPlan("3N overflow"))
        ));
        assert!(matches!(
            repair_replay_requirements(usize::MAX, 0, 1),
            Err(ReplayError::InvalidPlan("Repair role bound overflow"))
        ));
        assert!(matches!(
            repair_replay_requirements(usize::MAX / 4 + 1, 0, 1),
            Err(ReplayError::InvalidPlan("Repair node bound overflow"))
        ));
        assert!(matches!(
            repair_replay_requirements(usize::MAX / 4, 0, 1),
            Err(ReplayError::InvalidPlan(
                "Repair observation bound overflow"
            ))
        ));
        assert!(matches!(
            repair_replay_requirements(usize::MAX / 8 + 1, 0, 1),
            Err(ReplayError::InvalidPlan("Repair line chunk bound overflow"))
        ));
    }

    #[test]
    fn actual_repair_preflight_uses_public_requirement_thresholds_and_stage_reservation() {
        let required = repair_replay_requirements(3, 1, 100_000).unwrap();
        let cancel = AtomicBool::new(false);
        let mut owner = fixture(100_000);
        let mut bound = limits(100_000);
        bound.max_cpu_nodes = required.cpu_nodes();
        owner.engine.config.max_role_calls = required.role_calls();
        owner.prepare_repair(bound, &cancel).unwrap();
        assert!(owner.stages.capacity() >= required.stages());
        assert!(owner.stages.is_empty());
        assert_eq!(owner.counters.role_calls, 0);
        assert_eq!(owner.counters.cpu_tasks_requested, 0);
        let mut owner = fixture(100_000);
        bound.max_cpu_nodes = required.cpu_nodes() - 1;
        assert!(matches!(
            owner.prepare_repair(bound, &cancel),
            Err(ReplayError::InvalidPlan(_))
        ));
        assert!(owner.stages.is_empty());
        let mut owner = fixture(100_000);
        bound.max_cpu_nodes = required.cpu_nodes();
        owner.engine.config.max_role_calls = required.role_calls() - 1;
        assert!(matches!(
            owner.prepare_repair(bound, &cancel),
            Err(ReplayError::InvalidPlan(_))
        ));
        assert!(owner.stages.is_empty());
        assert_eq!(owner.counters.role_calls, 0);
        assert_eq!(owner.counters.cpu_tasks_requested, 0);
    }
}
