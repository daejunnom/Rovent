use super::{CoordinatorError, FreshPairSeal, ProposerRankingPublished, ProposerRankingSource};
use rz_contracts::pals::{Move16, Role, SearchAuthority};
use rz_contracts::pals_adaptive::{
    CpuJobEnvelope, CpuJobId, CpuWorkerId, ExternalCpuJobEnvelope, ExternalSliceGrant,
    ReviewViewSeal, RoleTicketEnvelope, SliceGrant,
};
use rz_contracts::rovent::{
    ExecutionMode, QuestionEnvelope, ReciprocalQuestionId, SearchRecipient,
};
use rz_contracts::{ClockDomain, Deadline, Digest, MonotonicTick, RequestId};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub const MAX_COORDINATOR_FRAMES: usize = 64;
pub const MAX_REPAIR_PER_FIRST_MOVE: u8 = 3;
const MAX_REPAIR_QUESTION_HISTORY: usize = 256 * MAX_REPAIR_PER_FIRST_MOVE as usize;
pub const MAX_FRAME_GENEALOGY: usize = 1024;
pub const MAX_STEP_BINDING_HISTORY: usize = 65_536;
pub const MAX_RECIPROCAL_QUESTION_HISTORY: usize = MAX_FRAME_GENEALOGY;
pub const MAX_RECIPROCAL_QUESTION_DEPTH: usize = 16;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CoordinatorHistoryUsage {
    pub role_requests: usize,
    pub cpu_executions: usize,
    /// Observed sum only. Each replay namespace has its own registered bound.
    pub replay_bindings: usize,
    /// All issued frame IDs, including retired candidate and Repair frames.
    pub frame_genealogy: usize,
    /// A subset of frame genealogy. Do not add this again to its byte/work count.
    pub ranking_fanouts: usize,
    pub repair_questions: usize,
}

/// The two non-default entry modes require private Search-port proofs. They
/// cannot be selected through a caller-supplied success/completion flag.
#[derive(Clone, Copy, Eq, PartialEq)]
enum CandidateAdmissionMode {
    InitialView,
    CheckedOnePly,
    DeferredView,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FrameId {
    pub authority: SearchAuthority,
    pub sequence: u64,
}

/// Admission carries ancestry as one value while keeping parent and supersedes
/// distinct. This is an inline private context, not a new runtime allocation.
#[derive(Clone, Copy, Default)]
struct FrameLineage {
    parent: Option<FrameId>,
    supersedes: Option<FrameId>,
}

/// These are actual existing refinement responsibilities, rather than a new
/// CPU algorithm. A step advances only after its sealed completion is admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RefinementStage {
    ProposerRanking,
    /// Standalone collection-only task ranking. It is never a candidate-frame
    /// entry stage and never selects the final chess move.
    ValidatorTaskRanking,
    ProposalContinuation,
    CpuValidation,
    /// Model-WDL resolver observation after proposal CPU validation. Own-raw
    /// profiles retain their direct CPU-to-C transition.
    ProposalValueFresh,
    CriticDivergence,
    CpuDiscovery,
    CriticReply,
    CounterContinuation,
    CounterValidation,
    CounterValueFresh,
    Repair,
    RepairValidation,
    RepairValueFresh,
    CriticRecheck,
    FreshRepairedEndpoint,
    FreshCounterEndpoint,
    /// The legacy Repair recheck verifies the continued counter endpoint on
    /// CPU before preparing its frozen comparison. Keep this distinct from
    /// CounterValidation, which would incorrectly enter another Repair.
    RecheckCounterValidation,
    /// Iterative Repair evaluates the repaired endpoint again when its queued
    /// recheck actually starts. This is distinct from the resolver's earlier
    /// RepairValueFresh observation and precedes the C recheck.
    RecheckRepairedValueFresh,
}
impl RefinementStage {
    pub fn role(self) -> Option<Role> {
        match self {
            Self::ProposerRanking | Self::ProposalContinuation | Self::Repair => {
                Some(Role::Proposer)
            }
            Self::ValidatorTaskRanking => Some(Role::Validator),
            Self::CriticDivergence
            | Self::CriticReply
            | Self::CounterContinuation
            | Self::CriticRecheck => Some(Role::Critic),
            Self::ProposalValueFresh
            | Self::CounterValueFresh
            | Self::RepairValueFresh
            | Self::FreshRepairedEndpoint
            | Self::FreshCounterEndpoint
            | Self::RecheckRepairedValueFresh => Some(Role::Proposer),
            Self::CpuValidation
            | Self::CpuDiscovery
            | Self::CounterValidation
            | Self::RepairValidation
            | Self::RecheckCounterValidation => None,
        }
    }
    fn permits(self, next: Self) -> bool {
        match self {
            Self::ProposerRanking => next == Self::ProposalContinuation,
            Self::ValidatorTaskRanking => false,
            Self::ProposalContinuation => {
                matches!(next, Self::ProposalContinuation | Self::CpuValidation)
            }
            Self::CpuValidation => {
                matches!(next, Self::ProposalValueFresh | Self::CriticDivergence)
            }
            Self::ProposalValueFresh => next == Self::CriticDivergence,
            Self::CriticDivergence => next == Self::CpuDiscovery,
            Self::CpuDiscovery => matches!(next, Self::CpuDiscovery | Self::CriticReply),
            Self::CriticReply => {
                // A reply can already complete the bounded counter line, or
                // reach a Rules terminal; it then proceeds directly to CPU
                // validation without an extra continuation inference.
                matches!(next, Self::CounterContinuation | Self::CounterValidation)
            }
            Self::CounterContinuation => {
                matches!(next, Self::CounterContinuation | Self::CounterValidation)
            }
            Self::CounterValidation => matches!(
                next,
                Self::CounterValueFresh
                    | Self::Repair
                    | Self::RepairValidation
                    | Self::RepairValueFresh
                    | Self::CounterContinuation
                    | Self::CounterValidation
            ),
            Self::CounterValueFresh => matches!(
                next,
                Self::Repair
                    | Self::RepairValidation
                    | Self::RepairValueFresh
                    | Self::CounterContinuation
                    | Self::CounterValidation
            ),
            Self::Repair => matches!(next, Self::Repair | Self::RepairValidation),
            Self::RepairValidation => matches!(
                next,
                Self::RepairValueFresh
                    | Self::CriticRecheck
                    | Self::RecheckRepairedValueFresh
                    | Self::CounterContinuation
                    | Self::CounterValidation
                    | Self::RepairValidation
            ),
            Self::RepairValueFresh => matches!(
                next,
                Self::CriticRecheck
                    | Self::RecheckRepairedValueFresh
                    | Self::CounterContinuation
                    | Self::CounterValidation
                    | Self::RepairValidation
                    | Self::RepairValueFresh
            ),
            Self::CriticRecheck => {
                matches!(
                    next,
                    Self::CriticRecheck
                        | Self::RecheckCounterValidation
                        | Self::CounterContinuation
                        | Self::CounterValidation
                        | Self::Repair
                        | Self::RepairValidation
                        | Self::RepairValueFresh
                )
            }
            Self::RecheckCounterValidation => matches!(
                next,
                Self::FreshRepairedEndpoint
                    | Self::CounterContinuation
                    | Self::CounterValidation
                    | Self::Repair
                    | Self::RepairValidation
                    | Self::RepairValueFresh
            ),
            Self::RecheckRepairedValueFresh => next == Self::CriticRecheck,
            Self::FreshRepairedEndpoint => next == Self::FreshCounterEndpoint,
            Self::FreshCounterEndpoint => matches!(
                next,
                Self::CounterContinuation
                    | Self::CounterValidation
                    | Self::Repair
                    | Self::RepairValidation
                    | Self::RepairValueFresh
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameBudget {
    pub role_submissions_remaining: u64,
    /// Original credit, not a new credit on every continuation; the CPU ledger
    /// is the only owner allowed to charge/reserve this value.
    pub original_cpu_credit: u64,
    pub original_store_byte_limit: u64,
}

/// Query and immutable view are owned until the step is resolved. The manager
/// constructs the query from checked Rules/store facts outside this module.
pub struct FrameInput<V, Q> {
    pub stage: RefinementStage,
    pub question_digest: Digest,
    pub question: Arc<Q>,
    pub view_seal: ReviewViewSeal,
    pub public_context: Digest,
    pub view: Arc<V>,
}
impl<V, Q> Clone for FrameInput<V, Q> {
    fn clone(&self) -> Self {
        Self {
            stage: self.stage,
            question_digest: self.question_digest,
            question: Arc::clone(&self.question),
            view_seal: self.view_seal,
            public_context: self.public_context,
            view: Arc::clone(&self.view),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum StepExecution {
    Role(RequestId),
    Cpu {
        job: CpuJobId,
        execution: rz_contracts::ExecutionId,
    },
    /// UCI reported nodes never enter native credit accounting.
    ExternalCpu {
        job: CpuJobId,
        execution: rz_contracts::ExecutionId,
    },
    ProfiledExternal {
        question: ReciprocalQuestionId,
        execution: rz_contracts::ExecutionId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FramePhase {
    Ready,
    Waiting(StepExecution),
    Suspended,
}

/// The final pair can use two exact Rules terminals without evaluating NN.
/// This seal covers the frozen public input; endpoint queries remain distinct.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RulesTerminalPairSeal {
    root_state: rz_contracts::StateIdentity,
    public_context: Digest,
    context_policy: Digest,
    selected_records: u16,
    anchors: u8,
    store_root: crate::pals::store::StateId,
    store_generation: u64,
}

/// A subset of bounded frame genealogy. The original closed question's
/// authority is retained while the owner queue materializes its later view.
#[derive(Clone, Copy)]
struct QueuedRecheckSeal {
    first_move: Move16,
    question: Digest,
    root_state: rz_contracts::StateIdentity,
    context_policy: Digest,
    original_deadline: Deadline,
    budget: FrameBudget,
    ranking_source: Option<ProposerRankingSource>,
    queued_evidence: Digest,
    original_view: ReviewViewSeal,
    public_context: Digest,
    /// Generic completion scopes remain diagnostic; only a port-issued proof
    /// grants deferred-question admission after its exact transition commits.
    producer_checked: bool,
    committed: bool,
    execution: Option<StepExecution>,
    issued: bool,
}

pub struct CoordinatorFrame<V, Q> {
    pub id: FrameId,
    pub original_first_move: Move16,
    pub original_deadline: Deadline,
    pub budget: FrameBudget,
    pub input: FrameInput<V, Q>,
    pub phase: FramePhase,
    pub evidence: Vec<Digest>,
    pub parent: Option<FrameId>,
    pub supersedes: Option<FrameId>,
    ranking_source: Option<ProposerRankingSource>,
    progress_sequence: u64,
    yield_sequence: u64,
    zero_work_yields: u8,
    fresh_pair: Option<FreshPairSeal>,
    rules_terminal_pair: Option<RulesTerminalPairSeal>,
    cpu_owner: Option<(CpuJobId, CpuWorkerId)>,
    last_execution: Option<StepExecution>,
}
impl<V, Q> CoordinatorFrame<V, Q> {
    pub fn ranking_source(&self) -> Option<ProposerRankingSource> {
        self.ranking_source
    }
}

/// Effect, not execution: callers route it to a private worker/native facade.
pub struct CoordinatorWork<V, Q> {
    pub frame: FrameId,
    pub original_first_move: Move16,
    pub original_deadline: Deadline,
    /// Actual manager-owned remaining budget at this snapshot. This is not a
    /// fresh allowance and cloning the work never refills it.
    pub budget: FrameBudget,
    pub input: FrameInput<V, Q>,
}

/// The original machine's move-only Ready custody for an opt-in intervention.
/// Public Work metadata, equal authority numbers or cloned view Arcs cannot
/// recreate the private machine owner. This cap alone grants no execution.
pub struct UtilityInterventionReadySource<Q> {
    owner: Arc<()>,
    work: CoordinatorWork<crate::pals::review_context::ReviewView, Q>,
}
impl<Q> UtilityInterventionReadySource<Q> {
    pub fn work(&self) -> &CoordinatorWork<crate::pals::review_context::ReviewView, Q> {
        &self.work
    }
    pub(crate) fn matches_owner(&self, owner: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.owner, owner)
    }
}
impl<V, Q> Clone for CoordinatorWork<V, Q> {
    fn clone(&self) -> Self {
        Self {
            frame: self.frame,
            original_first_move: self.original_first_move,
            original_deadline: self.original_deadline,
            budget: self.budget,
            input: self.input.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ValidatorActionHistoryUsage {
    pub source_contexts: usize,
    pub action_keys: usize,
}

/// Proof of the exact submitted step. The manager supplies this after actual
/// shape, Rules, store and worker report validation; no CP/NN number is promoted
/// to a Rules terminal fact by this state machine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepEvidence {
    pub frame: FrameId,
    pub stage: RefinementStage,
    pub execution: StepExecution,
    pub submitted_question: Digest,
    pub submitted_view: ReviewViewSeal,
    pub completed_evidence: Digest,
}

// Continue owns the next immutable envelope on the ordinary refinement path.
// Boxing it would add an allocation per continuation just to shrink the two
// terminal variants; retained frame capacity is independently bounded to 64.
#[expect(
    clippy::large_enum_variant,
    reason = "keep the bounded hot continuation envelope inline without per-step allocation"
)]
pub enum FrameTransition<V, Q> {
    Continue(FrameInput<V, Q>),
    Complete { scope: FrameCompletionScope },
    Unknown { reason: &'static str },
}

/// Finite completion scopes are explicit. A completed policy ranking by itself
/// never finishes the entire proposal/counter/repair workflow.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameCompletionScope {
    RulesTerminal,
    NoDivergence,
    NoAlternative,
    RepairNotAdmitted,
    /// The admitted Repair prefix already filled its bounded line or reached
    /// a Rules terminal. No Repair policy output or new witness was produced.
    NoRepairContinuation,
    /// Actual C continuation ends in a pair of Rules terminal endpoints. The
    /// consumer retains both exact endpoint proofs; no Fresh NN was executed.
    CompletedRulesPair,
    CompletedFreshPair,
    /// Actual owner queue registration, not a completed recheck, proof of no
    /// alternative, or a completed Fresh pair. A private producer proof is
    /// required to start the deferred recheck frame.
    QueuedRecheck,
}
impl FrameCompletionScope {
    fn permits(self, stage: RefinementStage) -> bool {
        match self {
            Self::RulesTerminal => true,
            Self::NoDivergence => matches!(
                stage,
                RefinementStage::CpuValidation
                    | RefinementStage::ProposalValueFresh
                    | RefinementStage::CriticDivergence
            ),
            Self::NoAlternative => matches!(
                stage,
                RefinementStage::CpuDiscovery
                    | RefinementStage::CriticReply
                    | RefinementStage::RepairValidation
                    | RefinementStage::RepairValueFresh
                    | RefinementStage::CriticRecheck
                    | RefinementStage::RecheckRepairedValueFresh
            ),
            Self::RepairNotAdmitted => matches!(
                stage,
                RefinementStage::CounterValidation | RefinementStage::CounterValueFresh
            ),
            Self::NoRepairContinuation => matches!(
                stage,
                RefinementStage::CounterValidation
                    | RefinementStage::CounterValueFresh
                    | RefinementStage::RepairValidation
                    | RefinementStage::RepairValueFresh
                    | RefinementStage::CriticRecheck
                    | RefinementStage::RecheckCounterValidation
                    | RefinementStage::FreshCounterEndpoint
            ),
            Self::CompletedFreshPair => stage == RefinementStage::FreshCounterEndpoint,
            Self::CompletedRulesPair => matches!(
                stage,
                RefinementStage::CriticRecheck
                    | RefinementStage::RecheckCounterValidation
                    | RefinementStage::FreshCounterEndpoint
            ),
            Self::QueuedRecheck => matches!(
                stage,
                RefinementStage::RepairValidation | RefinementStage::RepairValueFresh
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClosedFrame {
    pub id: FrameId,
    pub original_first_move: Move16,
    pub evidence: Vec<Digest>,
    pub unknown: Option<&'static str>,
    pub completion_scope: Option<FrameCompletionScope>,
}

/// A dependent question reuses the real frame/ticket/CPU owner path. Joining
/// existing work does not submit another role call or allocate more credit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DependentFrameAdmission {
    Admitted(FrameId),
    Joined(FrameId),
}

/// Completion of one dependent question, not a terminal chess proof or a
/// completed legacy PALS refinement workflow.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependentFrameResponse {
    pub frame: FrameId,
    pub question: Digest,
    pub view: ReviewViewSeal,
    pub evidence: Digest,
    pub execution: StepExecution,
}

#[derive(Clone, Copy)]
struct FrameDependency {
    waiter: FrameId,
    dependency: FrameId,
    question: ReciprocalQuestionId,
    origin_seal: Option<Digest>,
}

struct ReciprocalRoot {
    mode: ExecutionMode,
    root_budget: Digest,
    // Tombstones remain until root retirement: closing a question is not
    // permission to resubmit the same unchanged work under a fresh sequence.
    questions: Vec<(QuestionEnvelope, FrameId)>,
    accepted_origins: HashMap<(FrameId, Digest), super::AcceptedRoleOrigin>,
}

/// Finite continuation state with no worker calls and no hidden blocking.
/// Frames can progress independently; only this owner mutates their authority.
pub struct RefinementMachine<V, Q> {
    authority: SearchAuthority,
    hard_deadline: Deadline,
    token_budget: u16,
    next_sequence: u64,
    admission_open: bool,
    frames: VecDeque<CoordinatorFrame<V, Q>>,
    // One opt-in intervention may replace an exact unexecuted source frame.
    // Its original evidence and view owners stay here until actual worker join.
    intervention_sources: Vec<CoordinatorFrame<V, Q>>,
    intervention_owner: Option<Arc<()>>,
    repair_counts: Vec<(Move16, u8)>,
    repair_questions: VecDeque<(Move16, Digest, Digest)>,
    frame_genealogy: Vec<(FrameId, Move16)>,
    queued_rechecks: HashMap<(FrameId, Digest), QueuedRecheckSeal>,
    used_role_requests: HashSet<RequestId>,
    used_cpu_executions: HashSet<rz_contracts::ExecutionId>,
    ranking_fanouts: HashSet<(RequestId, Move16)>,
    ranking_request_sources: HashMap<RequestId, ProposerRankingSource>,
    validator_request_sources: HashMap<RequestId, super::AcceptedValidatorSource>,
    validator_actions: HashSet<(RequestId, u8)>,
    validator_history_peak: ValidatorActionHistoryUsage,
    history_peak: CoordinatorHistoryUsage,
    // Each live frame may await at most one other frame. This is a logical
    // wait; a suspended R frame retains its original private CPU owner.
    dependencies: Vec<FrameDependency>,
    // Only the allocating parent receives unused role credit. Joined waiters
    // paid no second reservation and therefore receive no refund.
    dependent_credit_owners: Vec<(FrameId, FrameId)>,
    reciprocal: Option<ReciprocalRoot>,
}
impl<V, Q> RefinementMachine<V, Q> {
    pub fn new(
        authority: SearchAuthority,
        hard_deadline: Deadline,
        token_budget: u16,
    ) -> Result<Self, CoordinatorError> {
        if hard_deadline.clock.0 != authority.epoch || ![128, 256, 512].contains(&token_budget) {
            return Err(CoordinatorError::IdentityMismatch(
                "coordinator clock/context profile",
            ));
        }
        Ok(Self {
            authority,
            hard_deadline,
            token_budget,
            next_sequence: 1,
            admission_open: true,
            frames: VecDeque::with_capacity(MAX_COORDINATOR_FRAMES),
            intervention_sources: Vec::new(),
            intervention_owner: None,
            repair_counts: Vec::with_capacity(256),
            repair_questions: VecDeque::with_capacity(MAX_COORDINATOR_FRAMES),
            frame_genealogy: Vec::with_capacity(MAX_FRAME_GENEALOGY),
            queued_rechecks: HashMap::new(),
            used_role_requests: HashSet::new(),
            used_cpu_executions: HashSet::new(),
            ranking_fanouts: HashSet::new(),
            ranking_request_sources: HashMap::new(),
            validator_request_sources: HashMap::new(),
            validator_actions: HashSet::new(),
            validator_history_peak: ValidatorActionHistoryUsage::default(),
            history_peak: CoordinatorHistoryUsage::default(),
            dependencies: Vec::with_capacity(MAX_COORDINATOR_FRAMES),
            dependent_credit_owners: Vec::with_capacity(MAX_COORDINATOR_FRAMES),
            reciprocal: None,
        })
    }
    pub fn frames(&self) -> impl Iterator<Item = &CoordinatorFrame<V, Q>> {
        self.frames.iter()
    }
    pub fn frame(&self, id: FrameId) -> Option<&CoordinatorFrame<V, Q>> {
        self.frames.iter().find(|frame| frame.id == id)
    }
    pub fn intervention_source_count(&self) -> usize {
        self.intervention_sources.len()
    }
    pub fn history_usage(&self) -> CoordinatorHistoryUsage {
        CoordinatorHistoryUsage {
            role_requests: self.used_role_requests.len(),
            cpu_executions: self.used_cpu_executions.len(),
            replay_bindings: self.used_role_requests.len() + self.used_cpu_executions.len(),
            frame_genealogy: self.frame_genealogy.len(),
            ranking_fanouts: self.ranking_fanouts.len(),
            repair_questions: self.repair_questions.len(),
        }
    }
    /// Small maxima survive final worker-join retirement for receipt emission.
    pub fn history_peak(&self) -> CoordinatorHistoryUsage {
        self.history_peak
    }
    pub fn validator_action_history_usage(&self) -> ValidatorActionHistoryUsage {
        ValidatorActionHistoryUsage {
            source_contexts: self.validator_request_sources.len(),
            action_keys: self.validator_actions.len(),
        }
    }
    pub fn validator_action_history_peak(&self) -> ValidatorActionHistoryUsage {
        self.validator_history_peak
    }
    fn observe_history_peak(&mut self) {
        let current = self.history_usage();
        let validator = self.validator_action_history_usage();
        self.validator_history_peak.source_contexts = self
            .validator_history_peak
            .source_contexts
            .max(validator.source_contexts);
        self.validator_history_peak.action_keys = self
            .validator_history_peak
            .action_keys
            .max(validator.action_keys);
        self.history_peak.role_requests =
            self.history_peak.role_requests.max(current.role_requests);
        self.history_peak.cpu_executions =
            self.history_peak.cpu_executions.max(current.cpu_executions);
        self.history_peak.replay_bindings = self
            .history_peak
            .replay_bindings
            .max(current.replay_bindings);
        self.history_peak.frame_genealogy = self
            .history_peak
            .frame_genealogy
            .max(current.frame_genealogy);
        self.history_peak.ranking_fanouts = self
            .history_peak
            .ranking_fanouts
            .max(current.ranking_fanouts);
        self.history_peak.repair_questions = self
            .history_peak
            .repair_questions
            .max(current.repair_questions);
    }
    pub fn admit(
        &mut self,
        first_move: Move16,
        input: FrameInput<V, Q>,
        budget: FrameBudget,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrameId, CoordinatorError> {
        if input.stage != RefinementStage::ProposerRanking {
            return Err(CoordinatorError::InvalidTransition(
                "new workflow starts with P ranking",
            ));
        }
        self.admit_internal(
            first_move,
            input,
            budget,
            FrameLineage::default(),
            clock,
            now,
        )
    }
    fn admit_internal(
        &mut self,
        first_move: Move16,
        input: FrameInput<V, Q>,
        budget: FrameBudget,
        lineage: FrameLineage,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrameId, CoordinatorError> {
        self.check_control(clock, now)?;
        input
            .view_seal
            .validate(self.authority, self.token_budget)?;
        first_move.decode()?;
        if self.frames.len() + self.intervention_sources.len() >= MAX_COORDINATOR_FRAMES {
            return Err(CoordinatorError::Capacity("pending coordinator frames"));
        }
        if self.frame_genealogy.len() == MAX_FRAME_GENEALOGY {
            return Err(CoordinatorError::Capacity("root frame genealogy"));
        }
        let next = self
            .next_sequence
            .checked_add(1)
            .ok_or(CoordinatorError::SequenceExhausted)?;
        let id = FrameId {
            authority: self.authority,
            sequence: self.next_sequence,
        };
        self.frames.push_back(CoordinatorFrame {
            id,
            original_first_move: first_move,
            original_deadline: self.hard_deadline,
            budget,
            input,
            phase: FramePhase::Ready,
            evidence: Vec::with_capacity(16),
            parent: lineage.parent,
            supersedes: lineage.supersedes,
            ranking_source: None,
            progress_sequence: 0,
            yield_sequence: 0,
            zero_work_yields: 0,
            fresh_pair: None,
            rules_terminal_pair: None,
            cpu_owner: None,
            last_execution: None,
        });
        self.frame_genealogy.push((id, first_move));
        self.next_sequence = next;
        self.observe_history_peak();
        Ok(id)
    }

    /// An immutable ready step may be prepared, then bound to one accepted
    /// worker submission. Returning work does not consume a role or CPU credit.
    pub fn ready_work(
        &self,
        id: FrameId,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<CoordinatorWork<V, Q>, CoordinatorError> {
        self.check_control(clock, now)?;
        self.check_not_waiting(id)?;
        let frame = self.frame(id).ok_or(CoordinatorError::UnknownRequest)?;
        if frame.phase != FramePhase::Ready {
            return Err(CoordinatorError::InvalidTransition("frame is not ready"));
        }
        Ok(CoordinatorWork {
            frame: id,
            original_first_move: frame.original_first_move,
            original_deadline: frame.original_deadline,
            budget: frame.budget,
            input: frame.input.clone(),
        })
    }

    /// Read the exact bound step after its real admission charged the role
    /// budget. The original immutable input/view/question remain unchanged;
    /// this grants neither execution nor renewed root/deadline authority.
    pub fn bound_work(
        &self,
        id: FrameId,
        execution: StepExecution,
    ) -> Result<CoordinatorWork<V, Q>, CoordinatorError> {
        let frame = self.frame(id).ok_or(CoordinatorError::UnknownRequest)?;
        if frame.phase != FramePhase::Waiting(execution) || frame.last_execution != Some(execution)
        {
            return Err(CoordinatorError::IdentityMismatch(
                "bound work original execution",
            ));
        }
        Ok(CoordinatorWork {
            frame: id,
            original_first_move: frame.original_first_move,
            original_deadline: frame.original_deadline,
            budget: frame.budget,
            input: frame.input.clone(),
        })
    }

    pub fn bind_role(
        &mut self,
        id: FrameId,
        ticket: &RoleTicketEnvelope,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), CoordinatorError> {
        self.check_control(clock, now)?;
        self.check_not_waiting(id)?;
        ticket.validate(clock, now, self.token_budget)?;
        if self.used_role_requests.contains(&ticket.id) {
            return Err(CoordinatorError::Duplicate);
        }
        if self.used_role_requests.len() == MAX_STEP_BINDING_HISTORY {
            return Err(CoordinatorError::Capacity("role binding history"));
        }
        let frame = self.match_frame_mut(id)?;
        if frame.phase != FramePhase::Ready
            || frame.input.stage.role() != Some(ticket.role)
            || ticket.authority != id.authority
            || ticket.question != frame.input.question_digest
            || ticket.view != frame.input.view_seal
            || ticket.original_deadline != frame.original_deadline
        {
            return Err(CoordinatorError::IdentityMismatch(
                "role question/frame/view/deadline",
            ));
        }
        if frame.budget.role_submissions_remaining == 0 {
            return Err(CoordinatorError::Capacity("frame role submission budget"));
        }
        if let Some(source) = frame.ranking_source {
            if ticket.role == Role::Proposer
                && (ticket.representation.model != source.ticket.representation.model
                    || ticket.representation.encoding != source.ticket.representation.encoding
                    || ticket.representation.precision != source.ticket.representation.precision
                    || ticket.representation.frozen_epoch
                        != source.ticket.representation.frozen_epoch)
            {
                return Err(CoordinatorError::IdentityMismatch(
                    "ranking proposer model namespace",
                ));
            }
        }
        if matches!(
            frame.input.stage,
            RefinementStage::FreshRepairedEndpoint | RefinementStage::FreshCounterEndpoint
        ) {
            if frame.rules_terminal_pair.is_some() {
                return Err(CoordinatorError::InvalidTransition(
                    "mixed Rules-terminal/neural Fresh pair is unresolved",
                ));
            }
            let seal = frame.fresh_pair.ok_or(CoordinatorError::MissingEvidence)?;
            seal.validate_pair(&FreshPairSeal::from_ticket(
                ticket,
                frame.input.public_context,
                seal.perspective,
            ))?;
        }
        frame.budget.role_submissions_remaining -= 1;
        frame.phase = FramePhase::Waiting(StepExecution::Role(ticket.id));
        frame.last_execution = Some(StepExecution::Role(ticket.id));
        self.used_role_requests.insert(ticket.id);
        self.observe_history_peak();
        Ok(())
    }

    /// Only a reservation already admitted by CpuCreditLedger may be supplied.
    /// CPU work/credit is not charged by this machine.
    pub fn bind_cpu(
        &mut self,
        id: FrameId,
        job: &CpuJobEnvelope,
        grant: &SliceGrant,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), CoordinatorError> {
        self.check_control(clock, now)?;
        self.check_not_waiting(id)?;
        grant.validate(job, clock, now)?;
        if self.used_cpu_executions.contains(&grant.execution) {
            return Err(CoordinatorError::Duplicate);
        }
        if self.used_cpu_executions.len() == MAX_STEP_BINDING_HISTORY {
            return Err(CoordinatorError::Capacity("CPU execution binding history"));
        }
        let frame = self.match_frame_mut(id)?;
        if frame.phase != FramePhase::Ready
            || frame.input.stage.role().is_some()
            || grant.job.authority != id.authority
            || grant.execution.epoch != id.authority.epoch
            || grant.node_credit > frame.budget.original_cpu_credit
            || job.question != frame.input.question_digest
            || job.hard_deadline != frame.original_deadline
            || frame
                .cpu_owner
                .is_some_and(|(previous_job, owner)| previous_job != job.id || owner != grant.owner)
        {
            return Err(CoordinatorError::IdentityMismatch(
                "CPU frame/reservation authority",
            ));
        }
        frame.cpu_owner = Some((job.id, grant.owner));
        frame.phase = FramePhase::Waiting(StepExecution::Cpu {
            job: grant.job,
            execution: grant.execution,
        });
        frame.last_execution = Some(StepExecution::Cpu {
            job: grant.job,
            execution: grant.execution,
        });
        self.used_cpu_executions.insert(grant.execution);
        self.observe_history_peak();
        Ok(())
    }

    /// Foreign wall-time/stop-reserve admission. No CpuCreditLedger operation or
    /// fabricated native node credit is performed for external UCI observations.
    pub fn bind_external_cpu(
        &mut self,
        id: FrameId,
        job: &ExternalCpuJobEnvelope,
        grant: &ExternalSliceGrant,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), CoordinatorError> {
        self.check_control(clock, now)?;
        self.check_not_waiting(id)?;
        grant.validate(job, clock, now)?;
        if self.used_cpu_executions.contains(&grant.execution) {
            return Err(CoordinatorError::Duplicate);
        }
        if self.used_cpu_executions.len() == MAX_STEP_BINDING_HISTORY {
            return Err(CoordinatorError::Capacity("CPU execution binding history"));
        }
        let frame = self.match_frame_mut(id)?;
        if frame.phase != FramePhase::Ready
            || frame.input.stage.role().is_some()
            || job.id.authority != id.authority
            || job.question != frame.input.question_digest
            || job.hard_deadline != frame.original_deadline
            || frame.cpu_owner.is_some()
        {
            return Err(CoordinatorError::IdentityMismatch(
                "external CPU question/frame/deadline",
            ));
        }
        frame.phase = FramePhase::Waiting(StepExecution::ExternalCpu {
            job: job.id,
            execution: grant.execution,
        });
        frame.last_execution = Some(StepExecution::ExternalCpu {
            job: job.id,
            execution: grant.execution,
        });
        self.used_cpu_executions.insert(grant.execution);
        self.observe_history_peak();
        Ok(())
    }

    /// Both endpoint admissions must carry the identical presealed Fresh scope.
    pub fn seal_fresh_pair(
        &mut self,
        id: FrameId,
        seal: FreshPairSeal,
    ) -> Result<(), CoordinatorError> {
        if !self.admission_open {
            return Err(CoordinatorError::AdmissionClosed);
        }
        let frame = self.match_frame_mut(id)?;
        if frame.rules_terminal_pair.is_some() {
            return Err(CoordinatorError::InvalidTransition(
                "cannot replace a Rules-terminal pair with neural Fresh evidence",
            ));
        }
        if !matches!(
            frame.input.stage,
            RefinementStage::FreshRepairedEndpoint | RefinementStage::FreshCounterEndpoint
        ) || !seal.validates_public_view(&frame.input.view_seal, frame.input.public_context)
        {
            return Err(CoordinatorError::IdentityMismatch("Fresh endpoint view"));
        }
        if let Some(original) = frame.fresh_pair {
            original.validate_pair(&seal)?;
        } else {
            frame.fresh_pair = Some(seal);
        }
        Ok(())
    }

    pub fn resolve(
        &mut self,
        evidence: StepEvidence,
        transition: FrameTransition<V, Q>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Option<ClosedFrame>, CoordinatorError> {
        self.check_control(clock, now)?;
        if matches!(
            transition,
            FrameTransition::Complete {
                scope: FrameCompletionScope::CompletedRulesPair
            }
        ) {
            return Err(CoordinatorError::MissingEvidence);
        }
        let frame = self.match_frame_mut(evidence.frame)?;
        if frame.phase != FramePhase::Waiting(evidence.execution)
            || frame.input.stage != evidence.stage
            || frame.input.question_digest != evidence.submitted_question
            || frame.input.view_seal != evidence.submitted_view
        {
            return Err(CoordinatorError::IdentityMismatch(
                "completed step does not match submitted frame",
            ));
        }
        self.commit_transition(
            evidence.frame,
            evidence.completed_evidence,
            transition,
            None,
        )
    }

    /// Only an attested worker completion, checked Rules endpoint, or opaque
    /// same-owner goal endpoint may enter this commit boundary. The endpoint
    /// paths never invent a StepExecution; Fresh still needs a real ticket.
    fn commit_transition(
        &mut self,
        frame_id: FrameId,
        completed_evidence: Digest,
        transition: FrameTransition<V, Q>,
        rules_pair: Option<RulesTerminalPairSeal>,
    ) -> Result<Option<ClosedFrame>, CoordinatorError> {
        self.check_not_waiting(frame_id)?;
        let is_dependent = self
            .dependent_credit_owners
            .iter()
            .any(|(child, _)| *child == frame_id);
        if is_dependent && !matches!(&transition, FrameTransition::Unknown { .. }) {
            return Err(CoordinatorError::InvalidTransition(
                "dependent question requires its exact reciprocal response",
            ));
        }
        if is_dependent
            && matches!(&transition, FrameTransition::Unknown { .. })
            && self.frame(frame_id).is_some_and(|frame| {
                matches!(
                    frame.last_execution,
                    Some(
                        StepExecution::Cpu { .. }
                            | StepExecution::ExternalCpu { .. }
                            | StepExecution::ProfiledExternal { .. }
                    )
                )
            })
        {
            return Err(CoordinatorError::MissingEvidence);
        }
        if !matches!(&transition, FrameTransition::Continue(_)) {
            self.check_dependent_retirement(frame_id)?;
        }
        let token_budget = self.token_budget;
        let authority = self.authority;
        let current = self
            .frame(frame_id)
            .ok_or(CoordinatorError::UnknownRequest)?;
        let queue_key = (frame_id, current.input.question_digest);
        if matches!(
            &transition,
            FrameTransition::Complete {
                scope: FrameCompletionScope::QueuedRecheck
            }
        ) && !self.queued_rechecks.contains_key(&queue_key)
            && self.queued_rechecks.len() >= MAX_FRAME_GENEALOGY
        {
            return Err(CoordinatorError::Capacity("queued Repair question history"));
        }
        if let Some(seal) = self
            .queued_rechecks
            .get(&queue_key)
            .filter(|seal| seal.producer_checked)
        {
            if seal.committed
                || (seal.execution.is_some() && seal.queued_evidence != completed_evidence)
                || current.phase
                    != seal
                        .execution
                        .map_or(FramePhase::Ready, FramePhase::Waiting)
                || seal.original_view != current.input.view_seal
                || seal.public_context != current.input.public_context
                || seal.budget != current.budget
            {
                return Err(CoordinatorError::IdentityMismatch(
                    "queued recheck exact committed question",
                ));
            }
            if !matches!(&transition,
                FrameTransition::Continue(next) if matches!(next.stage, RefinementStage::CounterContinuation | RefinementStage::CounterValidation)
            ) && !matches!(
                &transition,
                FrameTransition::Complete {
                    scope: FrameCompletionScope::QueuedRecheck
                }
            ) {
                return Err(CoordinatorError::InvalidTransition(
                    "queued Repair must commit its continuation or queue closure",
                ));
            }
        }
        let repair_key = match &transition {
            FrameTransition::Continue(next)
                if next.stage == RefinementStage::Repair
                    || (matches!(
                        next.stage,
                        RefinementStage::RepairValidation | RefinementStage::RepairValueFresh
                    ) && matches!(
                        current.input.stage,
                        RefinementStage::CounterValidation
                            | RefinementStage::CounterValueFresh
                            | RefinementStage::CriticRecheck
                            | RefinementStage::RecheckCounterValidation
                            | RefinementStage::FreshCounterEndpoint
                    )) =>
            {
                let current = self
                    .frame(frame_id)
                    .ok_or(CoordinatorError::UnknownRequest)?;
                if current.input.stage != RefinementStage::Repair {
                    let key = (
                        current.original_first_move,
                        next.question_digest,
                        next.view_seal.view_digest,
                    );
                    self.check_repair_key(key)?;
                    Some(key)
                } else {
                    None
                }
            }
            FrameTransition::Complete {
                scope: FrameCompletionScope::NoRepairContinuation,
            } => {
                let key = (
                    current.original_first_move,
                    current.input.question_digest,
                    current.input.view_seal.view_digest,
                );
                self.check_repair_key(key)?;
                Some(key)
            }
            _ => None,
        };
        let frame = self.match_frame_mut(frame_id)?;
        if frame.evidence.len() == 256 {
            return Err(CoordinatorError::Capacity("frame evidence links"));
        }
        match transition {
            FrameTransition::Continue(next) => {
                next.view_seal.validate(authority, token_budget)?;
                if !frame.input.stage.permits(next.stage) {
                    return Err(CoordinatorError::InvalidTransition("PALS refinement order"));
                }
                if next.view_seal.root_state != frame.input.view_seal.root_state {
                    return Err(CoordinatorError::IdentityMismatch(
                        "continuation exact original root",
                    ));
                }
                let next_branch = matches!(
                    next.stage,
                    RefinementStage::CounterContinuation | RefinementStage::CounterValidation
                ) && matches!(
                    frame.input.stage,
                    RefinementStage::CounterValidation
                        | RefinementStage::CounterValueFresh
                        | RefinementStage::RepairValidation
                        | RefinementStage::RepairValueFresh
                        | RefinementStage::CriticRecheck
                        | RefinementStage::RecheckCounterValidation
                        | RefinementStage::FreshCounterEndpoint
                );
                let next_repair = matches!(
                    next.stage,
                    RefinementStage::Repair
                        | RefinementStage::RepairValidation
                        | RefinementStage::RepairValueFresh
                ) && matches!(
                    frame.input.stage,
                    RefinementStage::CriticRecheck
                        | RefinementStage::RecheckCounterValidation
                        | RefinementStage::FreshCounterEndpoint
                );
                if (next_branch || next_repair)
                    && next.question_digest == frame.input.question_digest
                {
                    return Err(CoordinatorError::Duplicate);
                }
                if next.stage == RefinementStage::FreshCounterEndpoint
                    && !frame.fresh_pair.is_some_and(|seal| {
                        seal.validates_public_view(&next.view_seal, next.public_context)
                    })
                    && !rules_pair.is_some_and(|seal| {
                        seal.root_state == next.view_seal.root_state
                            && seal.public_context == next.public_context
                            && seal.context_policy == next.view_seal.context_policy
                            && seal.selected_records == next.view_seal.selected_records
                            && seal.anchors == next.view_seal.anchors
                    })
                {
                    return Err(CoordinatorError::IdentityMismatch(
                        "Fresh pair must retain frozen selected view",
                    ));
                }
                frame.evidence.push(completed_evidence);
                if next_branch || next_repair {
                    // The completed branch keeps its own evidence. Its frozen
                    // comparison may not be relabelled as a new counter/Repair
                    // question's pair. Actual port owns the bounded pending
                    // branch and Rules/genealogy checks before this commit.
                    frame.fresh_pair = None;
                    frame.rules_terminal_pair = None;
                }
                if let Some(pair) = rules_pair {
                    frame.rules_terminal_pair = Some(pair);
                }
                frame.input = next;
                frame.phase = FramePhase::Ready;
                frame.zero_work_yields = 0;
                frame.cpu_owner = None;
                frame.last_execution = None;
                frame.progress_sequence = 0;
                frame.yield_sequence = 0;
                if let Some(seal) = self.queued_rechecks.get_mut(&queue_key) {
                    seal.committed = true;
                }
                if let Some(key) = repair_key {
                    self.remember_repair_key(key);
                }
                Ok(None)
            }
            FrameTransition::Complete { scope } if !scope.permits(frame.input.stage) => Err(
                CoordinatorError::InvalidTransition("completion scope does not finish this stage"),
            ),
            FrameTransition::Complete {
                scope: FrameCompletionScope::CompletedFreshPair,
            } if frame.rules_terminal_pair.is_some() => Err(CoordinatorError::InvalidTransition(
                "Rules terminal pair did not execute a neural Fresh pair",
            )),
            FrameTransition::Complete {
                scope: FrameCompletionScope::CompletedRulesPair,
            } if (rules_pair.is_none() && frame.rules_terminal_pair.is_none())
                || frame.fresh_pair.is_some() =>
            {
                Err(CoordinatorError::MissingEvidence)
            }
            FrameTransition::Complete { .. } | FrameTransition::Unknown { .. } => {
                // Manager explicitly selects the completion scope. This is not
                // an automatic all-defenses or learned-utility proof.
                frame.evidence.push(completed_evidence);
                let unknown = match transition {
                    FrameTransition::Unknown { reason } => Some(reason),
                    _ => None,
                };
                let completion_scope = match transition {
                    FrameTransition::Complete { scope } => Some(scope),
                    _ => None,
                };
                let queued_recheck = (completion_scope
                    == Some(FrameCompletionScope::QueuedRecheck))
                .then_some(QueuedRecheckSeal {
                    first_move: frame.original_first_move,
                    question: frame.input.question_digest,
                    root_state: frame.input.view_seal.root_state,
                    context_policy: frame.input.view_seal.context_policy,
                    original_deadline: frame.original_deadline,
                    budget: frame.budget,
                    ranking_source: frame.ranking_source,
                    queued_evidence: completed_evidence,
                    original_view: frame.input.view_seal,
                    public_context: frame.input.public_context,
                    producer_checked: false,
                    committed: true,
                    execution: frame.last_execution,
                    issued: false,
                });
                let closed = ClosedFrame {
                    id: frame.id,
                    original_first_move: frame.original_first_move,
                    evidence: std::mem::take(&mut frame.evidence),
                    unknown,
                    completion_scope,
                };
                let remaining_roles = frame.budget.role_submissions_remaining;
                self.frames.retain(|held| held.id != closed.id);
                self.retire_dependencies(closed.id, remaining_roles);
                if let Some(seal) = queued_recheck {
                    self.queued_rechecks
                        .entry(queue_key)
                        .or_insert(seal)
                        .committed = true;
                }
                if let Some(key) = repair_key {
                    self.remember_repair_key(key);
                }
                Ok(Some(closed))
            }
        }
    }

    /// Enqueue only a distinct question under a new public revision/evidence.
    /// Counts and genealogy persist after older frames close; no recursive call.
    #[expect(
        clippy::too_many_arguments,
        reason = "preserve the public Repair boundary's distinct parent/supersedes and explicit original clock observation"
    )]
    pub fn enqueue_repair(
        &mut self,
        parent: FrameId,
        first_move: Move16,
        input: FrameInput<V, Q>,
        budget: FrameBudget,
        supersedes: Option<FrameId>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrameId, CoordinatorError> {
        self.check_control(clock, now)?;
        if parent.authority != self.authority
            || supersedes.is_some_and(|id| id.authority != self.authority)
            || input.stage != RefinementStage::Repair
            || parent.sequence == 0
            || parent.sequence >= self.next_sequence
            || supersedes.is_some_and(|id| id.sequence == 0 || id.sequence >= self.next_sequence)
            || self
                .frame(parent)
                .is_some_and(|frame| frame.original_first_move != first_move)
            || supersedes
                .and_then(|id| self.frame(id))
                .is_some_and(|frame| frame.original_first_move != first_move)
            || !self.frame_genealogy.contains(&(parent, first_move))
            || supersedes.is_some_and(|id| !self.frame_genealogy.contains(&(id, first_move)))
        {
            return Err(CoordinatorError::IdentityMismatch(
                "Repair genealogy/root/stage",
            ));
        }
        let key = (
            first_move,
            input.question_digest,
            input.view_seal.view_digest,
        );
        self.check_repair_key(key)?;
        let id = self.admit_internal(
            first_move,
            input,
            budget,
            FrameLineage {
                parent: Some(parent),
                supersedes,
            },
            clock,
            now,
        )?;
        self.remember_repair_key(key);
        Ok(id)
    }

    fn check_repair_key(&self, key: (Move16, Digest, Digest)) -> Result<(), CoordinatorError> {
        if self.repair_questions.contains(&key) {
            return Err(CoordinatorError::Duplicate);
        }
        let count = self
            .repair_counts
            .iter()
            .find(|(movement, _)| *movement == key.0)
            .map_or(0, |(_, count)| *count);
        if count >= MAX_REPAIR_PER_FIRST_MOVE
            || self.repair_questions.len() == MAX_REPAIR_QUESTION_HISTORY
        {
            return Err(CoordinatorError::Capacity(
                "Repair per-first-move/history limit",
            ));
        }
        if count == 0 && self.repair_counts.len() == 256 {
            return Err(CoordinatorError::Capacity("Repair first-move index"));
        }
        Ok(())
    }
    fn remember_repair_key(&mut self, key: (Move16, Digest, Digest)) {
        if let Some((_, count)) = self
            .repair_counts
            .iter_mut()
            .find(|(movement, _)| *movement == key.0)
        {
            *count += 1;
        } else {
            self.repair_counts.push((key.0, 1));
        }
        self.repair_questions.push_back(key);
        self.observe_history_peak();
    }

    /// Publication does not settle a reservation or move a paused token.
    pub fn publish_progress(
        &mut self,
        id: FrameId,
        execution: StepExecution,
        sequence: u64,
    ) -> Result<(), CoordinatorError> {
        if !self.admission_open {
            return Err(CoordinatorError::AdmissionClosed);
        }
        let frame = self.match_frame_mut(id)?;
        if frame.phase != FramePhase::Waiting(execution) {
            return Err(CoordinatorError::IdentityMismatch(
                "progress active execution",
            ));
        }
        if sequence == 0 || sequence <= frame.progress_sequence {
            return Err(CoordinatorError::Duplicate);
        }
        frame.progress_sequence = sequence;
        Ok(())
    }
    /// A bounded guard against issuing the same empty Yield forever. The worker
    /// ledger owns actual node charges; this only observes its validated delta.
    pub fn observe_yield(
        &mut self,
        id: FrameId,
        execution: StepExecution,
        control_ack: Option<u64>,
        actual_nodes: u64,
        new_witness: bool,
    ) -> Result<bool, CoordinatorError> {
        if !self.admission_open {
            return Err(CoordinatorError::AdmissionClosed);
        }
        let frame = self.match_frame_mut(id)?;
        if frame.phase != FramePhase::Waiting(execution)
            || !matches!(execution, StepExecution::Cpu { .. })
        {
            return Err(CoordinatorError::InvalidTransition(
                "Yield requires CPU step",
            ));
        }
        if let Some(sequence) = control_ack {
            if sequence == 0 || sequence <= frame.yield_sequence {
                return Err(CoordinatorError::Duplicate);
            }
            frame.yield_sequence = sequence;
        }
        if actual_nodes == 0 && !new_witness {
            frame.zero_work_yields = frame.zero_work_yields.saturating_add(1);
        } else {
            frame.zero_work_yields = 0;
        }
        frame.phase = FramePhase::Suspended;
        Ok(frame.zero_work_yields < 3)
    }
    /// CPU ledger/worker must first validate the same-owner continuation. A new
    /// execution is bound later; the original question/view/deadline stay intact.
    pub fn make_resumed_ready(&mut self, id: FrameId) -> Result<(), CoordinatorError> {
        if !self.admission_open {
            return Err(CoordinatorError::AdmissionClosed);
        }
        self.check_not_waiting(id)?;
        let frame = self.match_frame_mut(id)?;
        if frame.phase != FramePhase::Suspended || frame.zero_work_yields >= 3 {
            return Err(CoordinatorError::InvalidTransition(
                "resume requires progressing suspended frame",
            ));
        }
        frame.phase = FramePhase::Ready;
        Ok(())
    }
    /// After the credit owner accepts the real same-owner token-discard receipt,
    /// a suspended or closing CPU step may close as Unknown even after revoke
    /// or hard expiry. A late Yield need not restore revoked progress authority
    /// before its subsequent actual token-discard receipt can retire the step.
    /// This never settles credit, accepts a value, or restores search authority.
    pub fn retire_suspended_cpu(
        &mut self,
        id: FrameId,
        termination: &crate::cpu_worker::CpuJobTermination,
        evidence: Digest,
        reason: &'static str,
    ) -> Result<ClosedFrame, CoordinatorError> {
        self.retire_cpu_owner(id, termination, None, evidence, reason)
    }
    /// A canceled paused owner can retire before its already-queued Resume
    /// returns a definite no-work rejection. Only the ledger's opaque settled
    /// receipt for that newer bound grant permits closing the waiting frame.
    /// Neither receipt alone refunds credit or accepts a search value.
    pub fn retire_rejected_cpu_resume(
        &mut self,
        id: FrameId,
        termination: &crate::cpu_worker::CpuJobTermination,
        rejected: &crate::cpu_worker::SettledCpuSlice,
        evidence: Digest,
        reason: &'static str,
    ) -> Result<ClosedFrame, CoordinatorError> {
        self.retire_cpu_owner(id, termination, Some(rejected), evidence, reason)
    }
    fn retire_cpu_owner(
        &mut self,
        id: FrameId,
        termination: &crate::cpu_worker::CpuJobTermination,
        rejected: Option<&crate::cpu_worker::SettledCpuSlice>,
        evidence: Digest,
        reason: &'static str,
    ) -> Result<ClosedFrame, CoordinatorError> {
        let execution = match rejected {
            Some(settled) => {
                let grant = settled.grant();
                if settled.terminated_owner_receipt() != Some(termination)
                    || !settled.admission_rejected()
                    || settled.terminal()
                    || settled.actual_work() != crate::cpu::CpuWork::default()
                    || grant.job != termination.job
                    || grant.owner != termination.owner
                    || grant.execution.epoch != termination.last_execution.epoch
                    || grant.execution.sequence <= termination.last_execution.sequence
                {
                    return Err(CoordinatorError::IdentityMismatch(
                        "terminated CPU owner and settled rejected Resume",
                    ));
                }
                grant.execution
            }
            None => termination.last_execution,
        };
        // A real terminal owner receipt may retire a cancelled parent even
        // while its child is still running. Joined waiters retain that child;
        // an orphan remains visible for manager cancellation/physical drain.
        self.check_dependent_credit_return(id)?;
        let frame = self.match_frame_mut(id)?;
        let expected = StepExecution::Cpu {
            job: termination.job,
            execution,
        };
        if !(matches!(frame.phase, FramePhase::Suspended | FramePhase::Ready)
            || frame.phase == FramePhase::Waiting(expected))
            || frame.last_execution != Some(expected)
            || frame.cpu_owner != Some((termination.job, termination.owner))
            || termination.retained_bytes != 0
            || !termination.controls.terminal
            || !matches!(
                termination.completion,
                crate::cpu::CpuCompletion::Canceled | crate::cpu::CpuCompletion::Deadline
            )
            || (termination.completion == crate::cpu::CpuCompletion::Canceled
                && termination.controls.cancel_ack == 0
                && !termination.shutdown)
        {
            return Err(CoordinatorError::IdentityMismatch(
                "paused CPU terminal owner/token receipt",
            ));
        }
        if frame.evidence.len() == 256 {
            return Err(CoordinatorError::Capacity("frame evidence links"));
        }
        frame.evidence.push(evidence);
        let closed = ClosedFrame {
            id,
            original_first_move: frame.original_first_move,
            evidence: std::mem::take(&mut frame.evidence),
            unknown: Some(reason),
            completion_scope: None,
        };
        let remaining_roles = frame.budget.role_submissions_remaining;
        self.frames.retain(|frame| frame.id != id);
        self.retire_dependencies(id, remaining_roles);
        Ok(closed)
    }
    pub fn revoke(&mut self) -> Vec<StepExecution> {
        if !self.admission_open {
            return Vec::new();
        }
        self.admission_open = false;
        self.frames
            .iter()
            .filter_map(|frame| match frame.phase {
                FramePhase::Waiting(work) => Some(work),
                FramePhase::Suspended | FramePhase::Ready if frame.cpu_owner.is_some() => {
                    frame.last_execution
                }
                _ => None,
            })
            .collect()
    }
    pub fn retired_after_worker_join(&mut self) {
        // The caller must finish physical fences, worker joins and archive close
        // before dropping these query/view owners and issuing a receipt.
        self.admission_open = false;
        self.frames.clear();
        self.intervention_sources.clear();
        self.intervention_owner = None;
        self.frame_genealogy.clear();
        self.queued_rechecks.clear();
        self.ranking_fanouts.clear();
        self.ranking_request_sources.clear();
        self.validator_request_sources.clear();
        self.validator_actions.clear();
        self.dependencies.clear();
        self.dependent_credit_owners.clear();
        if let Some(root) = self.reciprocal.as_mut() {
            root.questions.clear();
            root.accepted_origins.clear();
        }
    }
    fn check_control(
        &self,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), CoordinatorError> {
        if !self.admission_open {
            return Err(CoordinatorError::AdmissionClosed);
        }
        self.hard_deadline.accepts(clock, now)?;
        Ok(())
    }
    fn match_frame_mut(
        &mut self,
        id: FrameId,
    ) -> Result<&mut CoordinatorFrame<V, Q>, CoordinatorError> {
        if id.authority != self.authority {
            return Err(CoordinatorError::IdentityMismatch("frame root authority"));
        }
        self.frames
            .iter_mut()
            .find(|frame| frame.id == id)
            .ok_or(CoordinatorError::UnknownRequest)
    }
}

/// A task action's admission is distinct from its later worker execution,
/// witness commit and utility label. Defer creates no work or positive label.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidatorActionAdmission {
    Frame(FrameId),
    Resumed(FrameId),
    Defer,
}

impl<Q> RefinementMachine<crate::pals::review_context::ReviewView, Q> {
    pub fn capture_intervention_source(
        &mut self,
        id: FrameId,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<UtilityInterventionReadySource<Q>, CoordinatorError> {
        let work = self.ready_work(id, clock, now)?;
        let source = self.frame(id).ok_or(CoordinatorError::UnknownRequest)?;
        if !self.intervention_sources.is_empty()
            || source.cpu_owner.is_some()
            || source.fresh_pair.is_some()
            || source.rules_terminal_pair.is_some()
            || self
                .dependencies
                .iter()
                .any(|edge| edge.waiter == id || edge.dependency == id)
            || self
                .dependent_credit_owners
                .iter()
                .any(|(child, parent)| *child == id || *parent == id)
            || self.queued_rechecks.keys().any(|(parent, _)| *parent == id)
        {
            return Err(CoordinatorError::InvalidTransition(
                "independent unstarted intervention source",
            ));
        }
        let owner = Arc::clone(self.intervention_owner.get_or_insert_with(|| Arc::new(())));
        Ok(UtilityInterventionReadySource { owner, work })
    }
    /// A separately declared experimental arm receives a real fresh frame.
    /// It replaces, rather than dispatches, its exact Ready neutral source.
    /// Credit is transferred unchanged; the source evidence is retained until
    /// the root's existing physical fence/worker-join retirement.
    #[allow(clippy::result_large_err)]
    pub fn admit_intervention(
        &mut self,
        candidate: crate::pals::engine::adaptive_port::utility_route::UtilityInterventionCandidate<
            Q,
        >,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<
        (
            FrameId,
            crate::pals::engine::adaptive_port::utility_route::UtilityInterventionAdmission<Q>,
        ),
        (
            CoordinatorError,
            crate::pals::engine::adaptive_port::utility_route::UtilityInterventionCandidate<Q>,
        ),
    > {
        let checked = (|| {
            self.check_control(clock, now)?;
            if !self.intervention_sources.is_empty() {
                return Err(CoordinatorError::Capacity("one intervention per root"));
            }
            let source_id = candidate.source_frame();
            let work = self.ready_work(source_id, clock, now)?;
            let source = self
                .frame(source_id)
                .ok_or(CoordinatorError::UnknownRequest)?;
            if !candidate.validates_source(&work)
                || self
                    .intervention_owner
                    .as_ref()
                    .is_none_or(|owner| !candidate.matches_source_owner(owner))
                || candidate.original_deadline() != self.hard_deadline
                || source.cpu_owner.is_some()
                || source.fresh_pair.is_some()
                || source.rules_terminal_pair.is_some()
                || self
                    .dependencies
                    .iter()
                    .any(|edge| edge.waiter == source_id || edge.dependency == source_id)
                || self
                    .dependent_credit_owners
                    .iter()
                    .any(|(child, parent)| *child == source_id || *parent == source_id)
                || self
                    .queued_rechecks
                    .keys()
                    .any(|(parent, _)| *parent == source_id)
            {
                return Err(CoordinatorError::IdentityMismatch(
                    "original independent Ready intervention source",
                ));
            }
            candidate
                .input()
                .view_seal
                .validate(self.authority, self.token_budget)?;
            super::ranking::validate_actual_view(
                &candidate.input().view,
                candidate.input().view_seal,
            )?;
            candidate.first_move().decode()?;
            let ranking_source = source.ranking_source;
            self.intervention_sources
                .try_reserve_exact(1)
                .map_err(|_| CoordinatorError::Capacity("retained intervention source"))?;
            Ok(ranking_source)
        })();
        let ranking_source = match checked {
            Ok(source) => source,
            Err(error) => return Err((error, candidate)),
        };
        let frame = match self.admit_internal(
            candidate.first_move(),
            candidate.input().clone(),
            candidate.budget(),
            FrameLineage::default(),
            clock,
            now,
        ) {
            Ok(frame) => frame,
            Err(error) => return Err((error, candidate)),
        };
        self.match_frame_mut(frame)
            .expect("new intervention frame")
            .ranking_source = ranking_source;
        let source_index = self
            .frames
            .iter()
            .position(|source| source.id == candidate.source_frame())
            .expect("checked original source remains until this single-writer transaction");
        let source = self
            .frames
            .remove(source_index)
            .expect("checked intervention source");
        self.intervention_sources.push(source);
        Ok((frame, candidate.admitted(frame)))
    }

    /// The opaque port binding retains the actual single-limit question and
    /// source. It never constructs a CpuJobId or reserves native credit.
    pub fn bind_profiled_external_cpu(
        &mut self,
        id: FrameId,
        proof: &super::CheckedProfiledExternalBinding,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), CoordinatorError> {
        self.check_control(clock, now)?;
        self.check_not_waiting(id)?;
        if self.used_cpu_executions.contains(&proof.execution) {
            return Err(CoordinatorError::Duplicate);
        }
        if self.used_cpu_executions.len() == MAX_STEP_BINDING_HISTORY {
            return Err(CoordinatorError::Capacity(
                "external execution binding history",
            ));
        }
        let frame = self.match_frame_mut(id)?;
        super::ranking::validate_actual_view(&frame.input.view, frame.input.view_seal)?;
        if proof.frame != id
            || proof.question.question_digest != frame.input.question_digest
            || proof.view != frame.input.view_seal
            || proof.public_context != frame.input.public_context
            || proof.input_digest != Digest(frame.input.view.digest())
            || frame.phase != FramePhase::Ready
            || frame.input.stage.role().is_some()
            || proof.question.id.authority != id.authority
            || proof.question.original_deadline != frame.original_deadline
            || frame.cpu_owner.is_some()
        {
            return Err(CoordinatorError::IdentityMismatch(
                "profiled external original frame binding",
            ));
        }
        let execution = StepExecution::ProfiledExternal {
            question: proof.question.id,
            execution: proof.execution,
        };
        frame.phase = FramePhase::Waiting(execution);
        frame.last_execution = Some(execution);
        self.used_cpu_executions.insert(proof.execution);
        self.observe_history_peak();
        Ok(())
    }

    /// Register an actual Repair's queued question while its original bound
    /// step, or unsubmitted exact Rules-terminal question, still exists. A
    /// parent may then continue several other counters;
    /// each question retains its own view and already-charged remaining budget.
    pub fn register_checked_recheck_queue(
        &mut self,
        proof: crate::pals::engine::adaptive_port::AdaptiveQueuedRecheck,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), CoordinatorError> {
        self.check_control(clock, now)?;
        let frame = self
            .frame(proof.parent())
            .ok_or(CoordinatorError::UnknownRequest)?;
        let execution = match frame.phase {
            FramePhase::Waiting(execution) if frame.last_execution == Some(execution) => {
                Some(execution)
            }
            FramePhase::Ready if frame.last_execution.is_none() && frame.cpu_owner.is_none() => {
                if frame.input.stage != RefinementStage::RepairValidation {
                    return Err(CoordinatorError::InvalidTransition(
                        "only Rules-terminal Repair validation queues without execution",
                    ));
                }
                let anchor = frame
                    .input
                    .view
                    .anchors()
                    .first()
                    .ok_or(CoordinatorError::MissingEvidence)?;
                let mut replay = rz_position::Position::from_snapshot(
                    &anchor.snapshot,
                    rz_position::PositionLimits::default(),
                )
                .map_err(|_| CoordinatorError::IdentityMismatch("queued Rules terminal anchor"))?;
                for movement in &frame.input.view.query().prefix {
                    replay.make_move(*movement).map_err(|_| {
                        CoordinatorError::IdentityMismatch("queued Rules terminal exact prefix")
                    })?;
                }
                let classification = replay.classify_position().map_err(|_| {
                    CoordinatorError::IdentityMismatch("queued Rules terminal classification")
                })?;
                if !matches!(
                    classification.play_status,
                    rz_position::PlayStatus::Terminal { .. }
                ) {
                    return Err(CoordinatorError::MissingEvidence);
                }
                None
            }
            _ => {
                return Err(CoordinatorError::InvalidTransition(
                    "queued Repair requires original completion or exact Rules terminal",
                ));
            }
        };
        let work = CoordinatorWork {
            frame: frame.id,
            original_first_move: frame.original_first_move,
            original_deadline: frame.original_deadline,
            budget: frame.budget,
            input: frame.input.clone(),
        };
        proof.validate_work(&work).map_err(|_| {
            CoordinatorError::IdentityMismatch(
                "actual queued Repair question/view/post-bind budget",
            )
        })?;
        super::ranking::validate_actual_view(&work.input.view, work.input.view_seal)?;
        work.input
            .view_seal
            .validate(self.authority, self.token_budget)?;
        if proof.parent().authority != self.authority
            || proof.original_deadline() != self.hard_deadline
            || !self
                .frame_genealogy
                .contains(&(proof.parent(), proof.first_move()))
        {
            return Err(CoordinatorError::IdentityMismatch(
                "queued Repair root/genealogy/deadline",
            ));
        }
        let key = (proof.parent(), proof.original_question());
        if self.queued_rechecks.contains_key(&key) {
            return Err(CoordinatorError::Duplicate);
        }
        if self.queued_rechecks.len() >= MAX_FRAME_GENEALOGY
            || self
                .queued_rechecks
                .values()
                .filter(|seal| !seal.issued)
                .count()
                >= MAX_COORDINATOR_FRAMES
        {
            return Err(CoordinatorError::Capacity(
                "queued Repair question history/pending frontier",
            ));
        }
        let seal = QueuedRecheckSeal {
            first_move: proof.first_move(),
            question: proof.original_question(),
            root_state: proof.root_state(),
            context_policy: proof.original_view().context_policy,
            original_deadline: proof.original_deadline(),
            budget: proof.budget(),
            ranking_source: frame.ranking_source,
            queued_evidence: proof.evidence(),
            original_view: proof.original_view(),
            public_context: proof.public_context(),
            producer_checked: true,
            committed: false,
            execution,
            issued: false,
        };
        self.queued_rechecks.insert(key, seal);
        Ok(())
    }

    /// Starts the actual queued iterative recheck only after its original
    /// parent closed and its particular question committed. Parent closure
    /// alone grants no deferred admission. This is not another Repair admission:
    /// Repair-per-first-move credit is charged only by a later real Repair.
    pub fn enqueue_checked_recheck(
        &mut self,
        proof: crate::pals::engine::adaptive_port::AdaptiveDeferredRepair<Q>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrameId, CoordinatorError> {
        self.check_control(clock, now)?;
        proof
            .input()
            .view_seal
            .validate(self.authority, self.token_budget)?;
        super::ranking::validate_actual_view(&proof.input().view, proof.input().view_seal)?;
        proof.validate().map_err(|_| {
            CoordinatorError::IdentityMismatch("actual queued iterative recheck port proof")
        })?;
        let parent = proof.parent();
        let first = proof.first_move();
        let queue_key = (parent, proof.original_question());
        let seal = *self
            .queued_rechecks
            .get(&queue_key)
            .ok_or(CoordinatorError::MissingEvidence)?;
        if seal.issued {
            return Err(CoordinatorError::Duplicate);
        }
        if parent.authority != self.authority
            || parent.sequence == 0
            || parent.sequence >= self.next_sequence
            || self.frame(parent).is_some()
            || !seal.producer_checked
            || !seal.committed
            || !self.frame_genealogy.contains(&(parent, first))
            || first != seal.first_move
            || proof.original_deadline() != self.hard_deadline
            || proof.original_deadline() != seal.original_deadline
            || proof.original_question() != seal.question
            || proof.original_view() != seal.original_view
            || proof.original_public_context() != seal.public_context
            || proof.input().question_digest == seal.question
            || proof.input().stage != RefinementStage::RecheckRepairedValueFresh
            || proof.input().view_seal.root_state != seal.root_state
            || proof.input().view_seal.context_policy != seal.context_policy
            || proof.budget().original_cpu_credit != seal.budget.original_cpu_credit
            || proof.budget().original_store_byte_limit != seal.budget.original_store_byte_limit
            || proof.budget().role_submissions_remaining > seal.budget.role_submissions_remaining
        {
            return Err(CoordinatorError::IdentityMismatch(
                "queued recheck original parent/first/question/resource scope",
            ));
        }
        let (original_parent, original_first, input, budget) = proof.into_parts();
        let id = self.admit_internal(
            original_first,
            input,
            budget,
            FrameLineage {
                parent: Some(original_parent),
                supersedes: Some(original_parent),
            },
            clock,
            now,
        )?;
        let frame = self.match_frame_mut(id).expect("new checked recheck frame");
        frame.ranking_source = seal.ranking_source;
        frame.evidence.push(seal.queued_evidence);
        self.queued_rechecks
            .get_mut(&queue_key)
            .expect("checked original queue parent")
            .issued = true;
        Ok(id)
    }

    /// A completed native/foreign counter check may close a Rules pair only
    /// with the actual Search port's opaque two-endpoint proof. Public generic
    /// StepEvidence alone is not enough to manufacture this scope.
    pub fn resolve_checked_rules_pair(
        &mut self,
        evidence: StepEvidence,
        proof: &crate::pals::engine::adaptive_port::AdaptiveRulesPairProof,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Option<ClosedFrame>, CoordinatorError> {
        self.check_control(clock, now)?;
        let frame = self
            .frame(evidence.frame)
            .ok_or(CoordinatorError::UnknownRequest)?;
        if frame.phase != FramePhase::Waiting(evidence.execution)
            || frame.input.stage != evidence.stage
            || frame.input.question_digest != evidence.submitted_question
            || frame.input.view_seal != evidence.submitted_view
        {
            return Err(CoordinatorError::IdentityMismatch(
                "Rules pair completion does not match submitted frame",
            ));
        }
        let seal = self.checked_rules_pair_seal(evidence.frame, proof)?;
        self.commit_transition(
            evidence.frame,
            evidence.completed_evidence,
            FrameTransition::Complete {
                scope: FrameCompletionScope::CompletedRulesPair,
            },
            Some(seal),
        )
    }

    /// Both exact Rules endpoints are already terminal; no CPU or NN work is
    /// submitted and no physical execution or evaluation count is invented.
    pub fn resolve_ready_rules_pair(
        &mut self,
        id: FrameId,
        proof: &crate::pals::engine::adaptive_port::AdaptiveRulesPairProof,
        evidence: Digest,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<ClosedFrame, CoordinatorError> {
        self.check_control(clock, now)?;
        let frame = self.frame(id).ok_or(CoordinatorError::UnknownRequest)?;
        if frame.phase != FramePhase::Ready
            || frame.cpu_owner.is_some()
            || frame.last_execution.is_some()
        {
            return Err(CoordinatorError::InvalidTransition(
                "ready Rules pair cannot retire active or paused work",
            ));
        }
        let seal = self.checked_rules_pair_seal(id, proof)?;
        self.commit_transition(
            id,
            evidence,
            FrameTransition::Complete {
                scope: FrameCompletionScope::CompletedRulesPair,
            },
            Some(seal),
        )?
        .ok_or(CoordinatorError::MissingEvidence)
    }

    fn checked_rules_pair_seal(
        &self,
        id: FrameId,
        proof: &crate::pals::engine::adaptive_port::AdaptiveRulesPairProof,
    ) -> Result<RulesTerminalPairSeal, CoordinatorError> {
        let frame = self.frame(id).ok_or(CoordinatorError::UnknownRequest)?;
        if !FrameCompletionScope::CompletedRulesPair.permits(frame.input.stage)
            || frame.fresh_pair.is_some()
        {
            return Err(CoordinatorError::InvalidTransition(
                "Rules pair completion stage",
            ));
        }
        super::ranking::validate_actual_view(&frame.input.view, frame.input.view_seal)?;
        if Digest(frame.input.view.record_scope_digest()) != frame.input.public_context {
            return Err(CoordinatorError::IdentityMismatch(
                "Rules pair frozen public view",
            ));
        }
        let work = CoordinatorWork {
            frame: id,
            original_first_move: frame.original_first_move,
            original_deadline: frame.original_deadline,
            budget: frame.budget,
            input: frame.input.clone(),
        };
        proof.validate_frozen_scope(&work).map_err(|_| {
            CoordinatorError::IdentityMismatch(
                "opaque Rules pair proof does not match the frozen frame",
            )
        })?;
        Ok(RulesTerminalPairSeal {
            root_state: frame.input.view_seal.root_state,
            public_context: frame.input.public_context,
            context_policy: frame.input.view_seal.context_policy,
            selected_records: frame.input.view_seal.selected_records,
            anchors: frame.input.view_seal.anchors,
            store_root: frame.input.view.root_state(),
            store_generation: frame.input.view.generation(),
        })
    }

    /// Consume a plan privately issued by the real Search port from an actual
    /// published V head and supported action. Parent authority and exact input
    /// remain live. A child receives role credit transferred from its parent;
    /// own CPU nodes remain under the single root-owned native credit ledger.
    pub fn admit_validator_action(
        &mut self,
        plan: crate::pals::engine::adaptive_port::validator_port::AdaptiveValidatorActionPlan<Q>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<ValidatorActionAdmission, CoordinatorError> {
        use crate::pals::engine::adaptive_port::validator_port::{
            AdaptiveValidatorAction as Action, AdaptiveValidatorEffect as Effect,
            MAX_VALIDATOR_CONTEXTS_PER_ROOT,
        };
        self.check_control(clock, now)?;
        let source_ticket = *plan.source();
        source_ticket.validate(clock, now, self.token_budget)?;
        let action = plan.action();
        let action_key = (source_ticket.id, action.index() as u8);
        let source = super::AcceptedValidatorSource {
            ticket: source_ticket,
            public_context: plan.source_public_context(),
            view_input_digest: plan.source_input(),
            output_digest: plan.source_output(),
            consumption_digest: plan.source_consumption(),
        };
        self.check_not_waiting(plan.parent())?;
        let parent = self
            .frame(plan.parent())
            .ok_or(CoordinatorError::UnknownRequest)?;
        if plan.id().authority != self.authority
            || plan.id().validator_request != source_ticket.id
            || source_ticket.role != Role::Validator
            || source_ticket.authority != self.authority
            || source_ticket.original_deadline != self.hard_deadline
            || plan.original_deadline() != parent.original_deadline
            || plan.original_first_move() != parent.original_first_move
            || source_ticket.view.root_state != parent.input.view_seal.root_state
            || plan.parent_question() != parent.input.question_digest
            || plan.parent_view() != parent.input.view_seal
            || plan.parent_input() != Digest(parent.input.view.digest())
            || plan.parent_public_context() != parent.input.public_context
            || !plan.action_support_mask()[action.index()]
            || !self
                .frame_genealogy
                .contains(&(parent.id, parent.original_first_move))
        {
            return Err(CoordinatorError::IdentityMismatch(
                "actual V action original parent/input/authority",
            ));
        }
        super::ranking::validate_actual_view(&parent.input.view, parent.input.view_seal)?;
        if self.validator_actions.contains(&action_key)
            || self
                .validator_request_sources
                .get(&source_ticket.id)
                .is_some_and(|prior| *prior != source)
            || (self.used_role_requests.contains(&source_ticket.id)
                && !self
                    .validator_request_sources
                    .contains_key(&source_ticket.id))
        {
            return Err(CoordinatorError::Duplicate);
        }
        if !self
            .validator_request_sources
            .contains_key(&source_ticket.id)
            && (self.validator_request_sources.len() == MAX_VALIDATOR_CONTEXTS_PER_ROOT
                || self.used_role_requests.len() == MAX_STEP_BINDING_HISTORY)
        {
            return Err(CoordinatorError::Capacity(
                "validator action source history",
            ));
        }
        let parent_id = parent.id;
        let first = parent.original_first_move;
        let inherited_ranking = parent.ranking_source;
        let role_transfer = match plan.effect() {
            Effect::Frame { input, budget } => {
                if parent.phase != FramePhase::Ready && parent.phase != FramePhase::Suspended {
                    return Err(CoordinatorError::InvalidTransition(
                        "V child requires stable parent checkpoint",
                    ));
                }
                let allowed = match action {
                    Action::DefendResponse | Action::WidenResponses => matches!(
                        input.stage,
                        RefinementStage::CpuDiscovery | RefinementStage::CriticReply
                    ),
                    Action::AttackRepair => input.stage == RefinementStage::CriticRecheck,
                    Action::CrossProfileRecheck | Action::LowerSelectivity => matches!(
                        input.stage,
                        RefinementStage::CpuValidation
                            | RefinementStage::CounterValidation
                            | RefinementStage::RepairValidation
                            | RefinementStage::CpuDiscovery
                    ),
                    Action::ResumeTask | Action::Defer => false,
                };
                if !allowed
                    || budget.role_submissions_remaining == 0
                    || budget.role_submissions_remaining > parent.budget.role_submissions_remaining
                    || budget.original_cpu_credit != parent.budget.original_cpu_credit
                    || budget.original_store_byte_limit != parent.budget.original_store_byte_limit
                    || input.view_seal.root_state != parent.input.view_seal.root_state
                    || input.view.root_state() != parent.input.view.root_state()
                    || input.view.generation() != parent.input.view.generation()
                    || Digest(input.view.record_scope_digest()) != input.public_context
                {
                    return Err(CoordinatorError::IdentityMismatch(
                        "V action stage/shared budget/root scope",
                    ));
                }
                input
                    .view_seal
                    .validate(self.authority, self.token_budget)?;
                super::ranking::validate_actual_view(&input.view, input.view_seal)?;
                let decoded = rz_position::BoardMove::try_from(first.decode()?)?;
                let anchor = input
                    .view
                    .anchors()
                    .first()
                    .ok_or(CoordinatorError::MissingEvidence)?;
                let original_anchor = parent
                    .input
                    .view
                    .anchors()
                    .first()
                    .ok_or(CoordinatorError::MissingEvidence)?;
                if !anchor.snapshot.same_state(&original_anchor.snapshot)
                    || input.view.query().prefix.first() != Some(&decoded)
                {
                    return Err(CoordinatorError::IdentityMismatch(
                        "V action original first/history",
                    ));
                }
                let mut endpoint = rz_position::Position::from_snapshot(
                    &anchor.snapshot,
                    rz_position::PositionLimits::default(),
                )
                .map_err(|_| CoordinatorError::IdentityMismatch("V action Rules anchor"))?;
                for movement in &input.view.query().prefix {
                    endpoint
                        .make_move(*movement)
                        .map_err(|_| CoordinatorError::IdentityMismatch("V action Rules prefix"))?;
                }
                let legal = endpoint.legal_moves();
                if input.view.query().legal.len() != legal.len()
                    || input
                        .view
                        .query()
                        .legal
                        .iter()
                        .collect::<HashSet<_>>()
                        .len()
                        != legal.len()
                    || input
                        .view
                        .query()
                        .legal
                        .iter()
                        .any(|movement| !legal.contains(movement))
                {
                    return Err(CoordinatorError::IdentityMismatch(
                        "V action Rules legal list",
                    ));
                }
                budget.role_submissions_remaining
            }
            Effect::Resume { frame, job, owner } => {
                if action != Action::ResumeTask
                    || *frame != parent_id
                    || parent.phase != FramePhase::Suspended
                    || parent.zero_work_yields >= 3
                    || parent.cpu_owner != Some((*job, *owner))
                    || !matches!(parent.last_execution,Some(StepExecution::Cpu { job: actual,.. }) if actual == *job)
                {
                    return Err(CoordinatorError::IdentityMismatch(
                        "V action same-owner paused native Job",
                    ));
                }
                0
            }
            Effect::Defer if action == Action::Defer => 0,
            Effect::Defer => {
                return Err(CoordinatorError::InvalidTransition(
                    "V action/effect mismatch",
                ));
            }
        };
        // All fallible guards run before this transaction. admit_internal's
        // bounded capacity/sequence check occurs before insertion; the parent
        // still exists, so its credit transfer cannot fail after insertion.
        let result = match plan.into_effect() {
            Effect::Frame { input, budget } => {
                let id = if input.stage == RefinementStage::Repair {
                    self.enqueue_repair(parent_id, first, input, budget, None, clock, now)?
                } else {
                    self.admit_internal(
                        first,
                        input,
                        budget,
                        FrameLineage {
                            parent: Some(parent_id),
                            supersedes: None,
                        },
                        clock,
                        now,
                    )?
                };
                self.match_frame_mut(parent_id)
                    .expect("checked live parent")
                    .budget
                    .role_submissions_remaining -= role_transfer;
                let child = self.match_frame_mut(id).expect("new action frame");
                child.ranking_source = inherited_ranking;
                child.evidence.push(source.consumption_digest);
                ValidatorActionAdmission::Frame(id)
            }
            Effect::Resume { frame, .. } => {
                self.make_resumed_ready(frame)?;
                ValidatorActionAdmission::Resumed(frame)
            }
            Effect::Defer => ValidatorActionAdmission::Defer,
        };
        self.validator_request_sources
            .insert(source_ticket.id, source);
        self.validator_actions.insert(action_key);
        self.used_role_requests.insert(source_ticket.id);
        self.observe_history_peak();
        Ok(result)
    }
}

impl<Q> RefinementMachine<crate::pals::review_context::ReviewView, Q> {
    /// Admit an actual engine-port mapping without treating the external UCI
    /// question digest as the semantic PALS view/question digest.
    pub fn bind_checked_external_cpu(
        &mut self,
        id: FrameId,
        job: &ExternalCpuJobEnvelope,
        grant: &ExternalSliceGrant,
        proof: &super::CheckedExternalCpuBinding,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), CoordinatorError> {
        self.check_control(clock, now)?;
        self.check_not_waiting(id)?;
        grant.validate(job, clock, now)?;
        if self.used_cpu_executions.contains(&grant.execution) {
            return Err(CoordinatorError::Duplicate);
        }
        if self.used_cpu_executions.len() == MAX_STEP_BINDING_HISTORY {
            return Err(CoordinatorError::Capacity("CPU execution binding history"));
        }
        let frame = self.match_frame_mut(id)?;
        super::ranking::validate_actual_view(&frame.input.view, frame.input.view_seal)?;
        if proof.frame != id
            || proof.job != *job
            || proof.grant != *grant
            || proof.question != frame.input.question_digest
            || proof.view != frame.input.view_seal
            || proof.public_context != frame.input.public_context
            || proof.input_digest != Digest(frame.input.view.digest())
            || frame.phase != FramePhase::Ready
            || frame.input.stage.role().is_some()
            || job.id.authority != id.authority
            || job.hard_deadline != frame.original_deadline
            || frame.cpu_owner.is_some()
        {
            return Err(CoordinatorError::IdentityMismatch(
                "checked external CPU original binding",
            ));
        }
        let execution = StepExecution::ExternalCpu {
            job: job.id,
            execution: grant.execution,
        };
        frame.phase = FramePhase::Waiting(execution);
        frame.last_execution = Some(execution);
        self.used_cpu_executions.insert(grant.execution);
        self.observe_history_peak();
        Ok(())
    }

    /// The actual Search owner replaced an unsubmitted validation exactly at
    /// its original go horizon with a Fresh question. This creates no R job,
    /// completed role value or physical execution evidence. A public input
    /// cannot authorize this path without the move-only original owner cap.
    pub fn advance_original_goal_boundary(
        &mut self,
        id: FrameId,
        boundary: crate::pals::engine::adaptive_port::AdaptiveGoalBoundary<Q>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), CoordinatorError> {
        self.check_control(clock, now)?;
        let frame = self.frame(id).ok_or(CoordinatorError::UnknownRequest)?;
        if frame.phase != FramePhase::Ready
            || frame.cpu_owner.is_some()
            || frame.last_execution.is_some()
        {
            return Err(CoordinatorError::InvalidTransition(
                "original goal boundary cannot replace active or paused work",
            ));
        }
        let work = self.ready_work(id, clock, now)?;
        boundary.validate(&work).map_err(|_| {
            CoordinatorError::IdentityMismatch("actual original goal boundary owner")
        })?;
        let (evidence, next) = boundary.into_parts();
        super::ranking::validate_actual_view(&next.view, next.view_seal)?;
        if self
            .commit_transition(id, evidence, FrameTransition::Continue(next), None)?
            .is_some()
        {
            return Err(CoordinatorError::InvalidTransition(
                "original goal boundary cannot close an unevaluated frame",
            ));
        }
        Ok(())
    }

    /// Decline a ready question without submitting work. Unknown never creates
    /// a success, evaluation, CPU credit delta, or a physical completion proof.
    /// Active and paused jobs must instead return their actual terminal ack.
    pub fn resolve_ready_unknown(
        &mut self,
        id: FrameId,
        evidence: Digest,
        reason: &'static str,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<ClosedFrame, CoordinatorError> {
        self.check_control(clock, now)?;
        let frame = self.frame(id).ok_or(CoordinatorError::UnknownRequest)?;
        if frame.phase != FramePhase::Ready
            || frame.cpu_owner.is_some()
            || frame.last_execution.is_some()
        {
            return Err(CoordinatorError::InvalidTransition(
                "unsubmitted Unknown cannot retire active or paused work",
            ));
        }
        super::ranking::validate_actual_view(&frame.input.view, frame.input.view_seal)?;
        if Digest(frame.input.view.record_scope_digest()) != frame.input.public_context {
            return Err(CoordinatorError::IdentityMismatch(
                "ready Unknown public view",
            ));
        }
        self.commit_transition(id, evidence, FrameTransition::Unknown { reason }, None)?
            .ok_or(CoordinatorError::MissingEvidence)
    }

    /// Resolve an exact terminal endpoint without creating a CPU/NN execution.
    /// This is a conditional line observation, not a proof covering all replies.
    /// It therefore preserves continuation/validation/C/Repair order instead of
    /// automatically closing the whole candidate with RulesTerminal scope.
    pub fn resolve_rules_terminal(
        &mut self,
        id: FrameId,
        endpoint: &rz_position::contracts::ContractState,
        evidence: Digest,
        transition: FrameTransition<crate::pals::review_context::ReviewView, Q>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Option<ClosedFrame>, CoordinatorError> {
        use sha2::{Digest as _, Sha256};
        self.check_control(clock, now)?;
        let frame = self.frame(id).ok_or(CoordinatorError::UnknownRequest)?;
        if frame.phase != FramePhase::Ready
            || frame.cpu_owner.is_some()
            || frame.last_execution.is_some()
        {
            return Err(CoordinatorError::InvalidTransition(
                "Rules terminal requires unsubmitted, unpaused ready step",
            ));
        }
        super::ranking::validate_actual_view(&frame.input.view, frame.input.view_seal)?;
        if Digest(frame.input.view.record_scope_digest()) != frame.input.public_context {
            return Err(CoordinatorError::IdentityMismatch(
                "Rules terminal public view",
            ));
        }
        let anchor = frame
            .input
            .view
            .anchors()
            .first()
            .ok_or(CoordinatorError::MissingEvidence)?;
        let mut replay = rz_position::Position::from_snapshot(
            &anchor.snapshot,
            rz_position::PositionLimits::default(),
        )
        .map_err(|_| CoordinatorError::IdentityMismatch("Rules terminal anchor restoration"))?;
        for movement in &frame.input.view.query().prefix {
            replay.make_move(*movement).map_err(|_| {
                CoordinatorError::IdentityMismatch("Rules terminal sealed prefix replay")
            })?;
        }
        let Some(wdl) = endpoint.terminal_wdl() else {
            return Err(CoordinatorError::MissingEvidence);
        };
        if !replay.snapshot().same_state(endpoint.rules().snapshot()) {
            return Err(CoordinatorError::IdentityMismatch(
                "Rules terminal exact endpoint history",
            ));
        }
        let legal = replay.legal_moves();
        let query = frame.input.view.query();
        if legal.len() != query.legal.len()
            || query.legal.iter().collect::<HashSet<_>>().len() != query.legal.len()
            || query.legal.iter().any(|movement| !legal.contains(movement))
        {
            return Err(CoordinatorError::IdentityMismatch(
                "Rules terminal endpoint legal order",
            ));
        }
        let stage = frame.input.stage;
        let allowed = match &transition {
            FrameTransition::Continue(next) => {
                matches!(
                    (stage, next.stage),
                    (
                        RefinementStage::ProposalContinuation,
                        RefinementStage::CpuValidation
                    ) | (
                        RefinementStage::CounterContinuation,
                        RefinementStage::CounterValidation
                    ) | (RefinementStage::Repair, RefinementStage::RepairValidation)
                        | (
                            RefinementStage::CpuValidation,
                            RefinementStage::CriticDivergence
                        )
                        | (RefinementStage::CounterValidation, RefinementStage::Repair)
                        | (
                            RefinementStage::CounterValidation,
                            RefinementStage::RepairValidation
                        )
                        | (
                            RefinementStage::RepairValidation,
                            RefinementStage::CriticRecheck
                        )
                        | (
                            RefinementStage::RecheckCounterValidation,
                            RefinementStage::FreshRepairedEndpoint
                        )
                        | (
                            RefinementStage::RecheckRepairedValueFresh,
                            RefinementStage::CriticRecheck
                        )
                        | (
                            RefinementStage::FreshRepairedEndpoint,
                            RefinementStage::FreshCounterEndpoint
                        )
                ) || (stage == RefinementStage::RepairValidation
                    && matches!(
                        next.stage,
                        RefinementStage::CounterContinuation | RefinementStage::CounterValidation
                    )
                    && self
                        .queued_rechecks
                        .get(&(id, frame.input.question_digest))
                        .is_some_and(|seal| {
                            seal.producer_checked && !seal.committed && seal.execution.is_none()
                        }))
            }
            FrameTransition::Complete { scope } => matches!(
                (stage, scope),
                (
                    RefinementStage::CpuValidation,
                    FrameCompletionScope::NoDivergence
                ) | (
                    RefinementStage::CounterValidation,
                    FrameCompletionScope::RepairNotAdmitted
                ) | (
                    RefinementStage::CounterValidation,
                    FrameCompletionScope::NoRepairContinuation
                ) | (
                    RefinementStage::RepairValidation,
                    FrameCompletionScope::NoAlternative
                ) | (
                    RefinementStage::FreshCounterEndpoint,
                    FrameCompletionScope::CompletedRulesPair
                ) | (
                    RefinementStage::RecheckRepairedValueFresh,
                    FrameCompletionScope::NoAlternative
                ) | (
                    RefinementStage::RepairValidation,
                    FrameCompletionScope::QueuedRecheck
                )
            ),
            FrameTransition::Unknown { .. } => matches!(
                stage,
                RefinementStage::ProposalContinuation
                    | RefinementStage::CounterContinuation
                    | RefinementStage::Repair
                    | RefinementStage::CpuValidation
                    | RefinementStage::CounterValidation
                    | RefinementStage::RepairValidation
                    | RefinementStage::RecheckCounterValidation
                    | RefinementStage::RecheckRepairedValueFresh
                    | RefinementStage::FreshRepairedEndpoint
                    | RefinementStage::FreshCounterEndpoint
            ),
        };
        if !allowed {
            return Err(CoordinatorError::InvalidTransition(
                "conditional Rules-terminal stage scope",
            ));
        }
        if let FrameTransition::Continue(next) = &transition {
            super::ranking::validate_actual_view(&next.view, next.view_seal)?;
            if Digest(next.view.record_scope_digest()) != next.public_context {
                return Err(CoordinatorError::IdentityMismatch(
                    "terminal next public view",
                ));
            }
            // These three edges replace only work on the same endpoint. A
            // different valid terminal line cannot lend its result to a new
            // CPU validation question. C/Repair edges intentionally change
            // question and are validated by their actual consumer instead.
            if matches!(
                stage,
                RefinementStage::ProposalContinuation
                    | RefinementStage::CounterContinuation
                    | RefinementStage::Repair
            ) && (next.view.query().prefix != query.prefix
                || next.view.query().legal != query.legal
                || next.view_seal.root_state != frame.input.view_seal.root_state
                || next.view.root_state() != frame.input.view.root_state()
                || next.view.generation() != frame.input.view.generation()
                || !next
                    .view
                    .anchors()
                    .first()
                    .is_some_and(|next_anchor| next_anchor.snapshot.same_state(&anchor.snapshot)))
            {
                return Err(CoordinatorError::IdentityMismatch(
                    "terminal continuation must validate the same exact line",
                ));
            }
        }
        let pair = RulesTerminalPairSeal {
            root_state: frame.input.view_seal.root_state,
            public_context: frame.input.public_context,
            context_policy: frame.input.view_seal.context_policy,
            selected_records: frame.input.view_seal.selected_records,
            anchors: frame.input.view_seal.anchors,
            store_root: frame.input.view.root_state(),
            store_generation: frame.input.view.generation(),
        };
        let mut pending_pair = None;
        if stage == RefinementStage::FreshRepairedEndpoint {
            if frame.fresh_pair.is_some() || frame.rules_terminal_pair.is_some() {
                return Err(CoordinatorError::InvalidTransition(
                    "mixed or duplicate final Fresh evidence",
                ));
            }
            if let FrameTransition::Continue(next) = &transition {
                super::ranking::validate_actual_view(&next.view, next.view_seal)?;
                if Digest(next.view.record_scope_digest()) != pair.public_context
                    || next.view.root_state() != pair.store_root
                    || next.view.generation() != pair.store_generation
                {
                    return Err(CoordinatorError::IdentityMismatch(
                        "Rules pair frozen public context",
                    ));
                }
                pending_pair = Some(pair);
            }
        }
        if stage == RefinementStage::FreshCounterEndpoint
            && (frame.fresh_pair.is_some() || frame.rules_terminal_pair != Some(pair))
        {
            return Err(CoordinatorError::InvalidTransition(
                "mixed or unfrozen terminal endpoint pair",
            ));
        }
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-checked-rules-terminal-step/1\0");
        hash.update(evidence.0);
        hash.update(frame.input.view_seal.view_digest.0);
        hash.update(endpoint.snapshot().identity().semantic.0);
        for probability in wdl.probabilities() {
            hash.update(probability.to_bits().to_le_bytes());
        }
        let checked_evidence = Digest(hash.finalize().into());
        let queue_key = (id, frame.input.question_digest);
        if self.queued_rechecks.get(&queue_key).is_some_and(|seal| {
            seal.producer_checked && seal.execution.is_none() && seal.queued_evidence != evidence
        }) {
            return Err(CoordinatorError::IdentityMismatch(
                "queued Rules terminal original evidence",
            ));
        }
        let closed = self.commit_transition(id, checked_evidence, transition, pending_pair)?;
        if let Some(seal) = self
            .queued_rechecks
            .get_mut(&queue_key)
            .filter(|seal| seal.producer_checked && seal.execution.is_none() && seal.committed)
        {
            seal.queued_evidence = checked_evidence;
        }
        Ok(closed)
    }

    /// Start candidate continuation from the one actually consumed root P
    /// ranking. The original forward is provenance, not a new worker execution
    /// or another consumed NN output. Its first move and the exact Rules replay
    /// are read from the immutable continuation view rather than a boolean.
    pub fn admit_candidate(
        &mut self,
        ranking: &ProposerRankingPublished,
        first_move: Move16,
        input: FrameInput<crate::pals::review_context::ReviewView, Q>,
        budget: FrameBudget,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrameId, CoordinatorError> {
        self.admit_published_candidate(
            ranking,
            first_move,
            input,
            budget,
            CandidateAdmissionMode::InitialView,
            clock,
            now,
        )
    }

    /// The actual port may register a one-ply proposal profile. Its accepted
    /// root ranking already supplies that proposal's sole move, so no second
    /// policy forward is invented before the real CPU validation. Arbitrary
    /// public FrameInput cannot select this profile shortcut.
    pub fn admit_checked_candidate(
        &mut self,
        ranking: &ProposerRankingPublished,
        candidate: crate::pals::engine::adaptive_port::CheckedRootCandidate<Q>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrameId, CoordinatorError> {
        self.check_control(clock, now)?;
        candidate
            .input()
            .view_seal
            .validate(self.authority, self.token_budget)?;
        super::ranking::validate_actual_view(&candidate.input().view, candidate.input().view_seal)?;
        candidate.validate(ranking).map_err(|_| {
            CoordinatorError::IdentityMismatch("actual registered one-ply candidate port proof")
        })?;
        if candidate.source() != ranking.source() {
            return Err(CoordinatorError::IdentityMismatch(
                "one-ply original ranking source",
            ));
        }
        let (first, input, budget) = candidate.into_candidate();
        self.admit_published_candidate(
            ranking,
            first,
            input,
            budget,
            CandidateAdmissionMode::CheckedOnePly,
            clock,
            now,
        )
    }

    /// SequentialFixed creates this new view only after the previous original
    /// candidate actually closes. The private Search port proof attests that
    /// it was not a submitted frame/ticket refresh. Its initial ranking remains
    /// provenance from the older frozen context, never a new policy evaluation.
    pub fn admit_deferred_candidate(
        &mut self,
        ranking: &ProposerRankingPublished,
        candidate: crate::pals::engine::adaptive_port::DeferredRootCandidate<Q>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrameId, CoordinatorError> {
        self.check_control(clock, now)?;
        candidate
            .input()
            .view_seal
            .validate(self.authority, self.token_budget)?;
        super::ranking::validate_actual_view(&candidate.input().view, candidate.input().view_seal)?;
        candidate.validate(ranking).map_err(|_| {
            CoordinatorError::IdentityMismatch("actual deferred root candidate port proof")
        })?;
        let previous = candidate.previous_frame();
        if previous.authority != self.authority
            || self.frame(previous).is_some()
            || !self.frame_genealogy.iter().any(|(id, _)| *id == previous)
            || candidate.source() != ranking.source()
        {
            return Err(CoordinatorError::IdentityMismatch(
                "previous deferred candidate actual closure/source",
            ));
        }
        let (first, input, budget) = candidate.into_candidate();
        self.admit_published_candidate(
            ranking,
            first,
            input,
            budget,
            CandidateAdmissionMode::DeferredView,
            clock,
            now,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn admit_published_candidate(
        &mut self,
        ranking: &ProposerRankingPublished,
        first_move: Move16,
        input: FrameInput<crate::pals::review_context::ReviewView, Q>,
        budget: FrameBudget,
        mode: CandidateAdmissionMode,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrameId, CoordinatorError> {
        self.check_control(clock, now)?;
        let source = ranking.source();
        source.ticket.validate(clock, now, self.token_budget)?;
        input
            .view_seal
            .validate(self.authority, self.token_budget)?;
        super::ranking::validate_actual_view(&input.view, input.view_seal)?;
        let query = input.view.query();
        let decoded = rz_position::BoardMove::try_from(first_move.decode()?)?;
        if !matches!(
            input.stage,
            RefinementStage::ProposalContinuation | RefinementStage::CpuValidation
        ) || source.ticket.authority != self.authority
            || source.ticket.original_deadline != self.hard_deadline
            || input.view_seal.root_state != source.ticket.view.root_state
            || input.view.root_state() != ranking.view().root_state()
            || input.view.generation() != ranking.view().generation()
            || (mode == CandidateAdmissionMode::InitialView
                && input.public_context != source.public_context)
            || Digest(input.view.record_scope_digest()) != input.public_context
            || (mode != CandidateAdmissionMode::InitialView
                && (input.view_seal.context_policy != source.ticket.view.context_policy
                    || input.view.sequence() <= ranking.view().sequence()))
            || query.purpose != crate::pals::engine::RoleQueryPurpose::ProposePolicy
            || query.prefix.first() != Some(&decoded)
            || query.prefix.len() > 256
            || query.counterexample.is_some()
            || !query.divergences.is_empty()
            || query
                .proposal
                .first()
                .is_some_and(|movement| *movement != decoded)
            || !ranking.candidates().contains(&first_move)
        {
            return Err(CoordinatorError::IdentityMismatch(
                "published root ranking/candidate question scope",
            ));
        }
        let anchor = input
            .view
            .anchors()
            .first()
            .ok_or(CoordinatorError::MissingEvidence)?;
        if !anchor
            .snapshot
            .same_state(ranking.rules().rules().snapshot())
        {
            return Err(CoordinatorError::IdentityMismatch(
                "candidate exact Rules root history",
            ));
        }
        let mut endpoint = rz_position::Position::from_snapshot(
            ranking.rules().rules().snapshot(),
            rz_position::PositionLimits::default(),
        )
        .map_err(|_| CoordinatorError::IdentityMismatch("candidate Rules root restoration"))?;
        for movement in &query.prefix {
            endpoint.make_move(*movement).map_err(|_| {
                CoordinatorError::IdentityMismatch("candidate checked continuation prefix")
            })?;
        }
        if input.stage == RefinementStage::CpuValidation
            && (query.prefix.len() != 1
                || query.proposal != query.prefix
                || (mode == CandidateAdmissionMode::InitialView
                    && endpoint
                        .classify_position()
                        .map_err(|_| {
                            CoordinatorError::IdentityMismatch("candidate Rules terminal")
                        })?
                        .play_status
                        == rz_position::PlayStatus::Ongoing))
        {
            return Err(CoordinatorError::InvalidTransition(
                "candidate CPU entry requires checked completed proposal",
            ));
        }
        let legal = endpoint.legal_moves();
        if query.legal.len() != legal.len()
            || query.legal.iter().collect::<HashSet<_>>().len() != query.legal.len()
            || query.legal.iter().any(|movement| !legal.contains(movement))
        {
            return Err(CoordinatorError::IdentityMismatch(
                "candidate endpoint ordered legal list",
            ));
        }
        if self
            .ranking_fanouts
            .contains(&(source.ticket.id, first_move))
        {
            return Err(CoordinatorError::Duplicate);
        }
        if self.used_role_requests.contains(&source.ticket.id)
            && self.ranking_request_sources.get(&source.ticket.id) != Some(&source)
        {
            // A request previously bound as an ordinary step is not a second
            // root fanout source. Nor may a colliding RequestId replace an
            // accepted source's model/view/output seal with a different proof.
            return Err(CoordinatorError::Duplicate);
        }
        if self.ranking_fanouts.len() == MAX_FRAME_GENEALOGY {
            return Err(CoordinatorError::Capacity("root ranking fanout history"));
        }
        if !self.used_role_requests.contains(&source.ticket.id)
            && self.used_role_requests.len() == MAX_STEP_BINDING_HISTORY
        {
            return Err(CoordinatorError::Capacity("role binding history"));
        }
        let id = self.admit_internal(
            first_move,
            input,
            budget,
            FrameLineage::default(),
            clock,
            now,
        )?;
        let frame = self.match_frame_mut(id).expect("new candidate frame");
        frame.ranking_source = Some(source);
        frame.evidence.push(source.consumption_digest);
        self.ranking_fanouts.insert((source.ticket.id, first_move));
        self.ranking_request_sources
            .insert(source.ticket.id, source);
        self.used_role_requests.insert(source.ticket.id);
        self.observe_history_peak();
        Ok(id)
    }
}

#[cfg(test)]
#[path = "machine_tests.rs"]
mod tests;

#[path = "reciprocal.rs"]
mod reciprocal;
pub use reciprocal::ReciprocalDependencyProof;
