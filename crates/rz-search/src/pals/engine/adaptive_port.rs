//! Single-writer semantic port for worker-owned adaptive execution.
//! No method in this module forwards a model, searches a CPU, drains a pipe,
//! or writes an archive. Only bounded Rules replay and checked hot commits run
//! on the manager. A hot-capacity error is an archive-worker admission barrier.
use super::*;
use crate::pals::coordinator::{
    ClosedFrame, CoordinatorWork, FrameBudget, FrameCompletionScope, FrameId, FrameInput,
    FrameTransition, FreshPairSeal, ProposerRankingPublished, ProposerRankingSource,
    RefinementStage,
};
use rz_contracts::pals::{Role, RoleOutput, RolePayload, SearchAuthority};
use rz_contracts::pals_adaptive::{ReviewViewSeal, RoleTicketEnvelope};
use rz_contracts::{Deadline, Digest as ContractDigest, OwnerId, RequestId};
use rz_position::contracts::{ContractPosition, ContractState};
use std::collections::{HashMap, HashSet, VecDeque};
mod branch_queue;
mod checked_replay;
mod conditional_evidence;
mod consumption_observation;
mod continuation_blueprint;
mod cpu_commit;
mod cpu_evidence_context;
pub(in crate::pals::engine) use cpu_evidence_context::CpuEvidenceContextSeal;
mod cpu_semantic_witness;
pub use cpu_semantic_witness::{CpuSemanticStage, CpuSemanticWitness};
mod external_commit;
#[cfg(test)]
mod fixed_trace_tests;
mod frontier_port;
mod goal_hooks;
pub(in crate::pals::engine) mod goal_local;
mod legacy_full_context;
mod mate_strategy;
mod model_namespace;
mod nnuec_installation;
mod nnuec_port;
#[cfg(test)]
mod port_tests;
mod preparing_nnuec;
mod preparing_port;
mod profiled_external_port;
mod reciprocal_port;
mod role_commit;
mod utility_conditional_basis;
mod utility_dependencies;
pub use reciprocal_port::{
    AdaptiveAcceptedReciprocalAnswer, AdaptiveCpuReciprocalQuestion, AdaptivePeerQuestion,
    AdaptiveSourceOpinion,
};
pub use utility_dependencies::{UtilityCohortOfferSemanticSource, UtilityContinuationSource};
pub mod utility_route;
pub mod validator_port;
pub use branch_queue::{AdaptiveDeferredRepair, AdaptiveQueuedRecheck};
pub use consumption_observation::{
    AdaptivePublishedRepairLine, AdaptiveRepairLineInput, AdaptiveRepairLineProvenance,
    AdaptiveRoleConsumptionEvent, AdaptiveRoleConsumptionObservation,
    AdaptiveRoleConsumptionOutcome, AdaptiveRoleConsumptionRejection, AdaptiveRoleConsumptionStage,
    AdaptiveRoleConsumptionUnavailable, MAX_ROLE_CONSUMPTION_EVENTS,
};
pub use continuation_blueprint::{
    AdaptiveContinuationBlueprint, AdaptivePortMaterializedStep, AdaptivePortTransition,
};
pub use external_commit::{AdaptiveForeignPublication, AdaptivePortExternalQuestion};
pub use frontier_port::{
    AdaptiveFrontierReceipt, AdaptiveFrontierStartup, AdaptiveReviewLeaseAdmission,
};
pub use goal_hooks::AdaptiveGoalBoundary;
pub(crate) use goal_hooks::AdaptiveLaneSelection;
pub use goal_local::{
    AdaptiveGoalLocalAdmission, GoalLocalBlockReason, GoalLocalInput, GoalLocalLeaf,
    GoalLocalLeafKind, GoalLocalQuestionKey, GoalLocalStep,
};
pub use legacy_full_context::LegacyFullRoleRecords;
pub use model_namespace::AdaptiveModelNamespace;
pub use nnuec_port::{
    AdaptiveNnuecCaptureError, AdaptiveNnuecPreparationError, AdaptiveNnuecReviewRequest,
    AdaptiveNnuecReviewSeal,
};
pub use preparing_nnuec::{PreparingRootNnuecRequest, PreparingRootViewSeal};
pub use preparing_port::{AdaptivePreparingRootAdmission, AdaptivePreparingRootCustody};
pub(in crate::pals::engine) use profiled_external_port::ProfiledExternalRetiredCustody;
pub use profiled_external_port::{
    AdaptiveProfiledExternalAdmission, AdaptiveProfiledExternalCommit,
    AdaptiveProfiledExternalPreparation, AdaptiveProfiledForeignPublication,
    CheckedProfiledExternalResponse, ProfiledExternalRetiredPacket,
    ProfiledExternalRetirementIdentity, ProfiledExternalRootRetirement,
    ProfiledExternalUnsupported, ProfiledFeedbackConsumedChain,
    ProfiledFeedbackConsumptionObservation, ProfiledFeedbackObservationUnavailable,
};
pub mod archive_worker;
pub mod profiled_retirement_archive;
/// The process registry remains owned by UCI. Search cannot invent an owner
/// from a root generation or reuse one for a separately owned Rules instance.
pub type AdaptiveRulesOwnerIssuer =
    Arc<dyn Fn() -> Result<OwnerId, rz_contracts::ContractError> + Send + Sync>;

/// Metadata mirror for the search owner; physical methods are deliberately
/// unsupported. The actual backend identity is registered before construction.
pub struct AdaptivePortModel {
    identity: String,
    value: Option<ModelValueIdentity>,
    namespace: Option<AdaptiveModelNamespace>,
}
impl AdaptivePortModel {
    pub fn new(identity: String, value: Option<ModelValueIdentity>) -> Result<Self, PalsError> {
        if identity.is_empty() {
            return Err(PalsError::InvalidConfig);
        }
        if let Some(value) = &value {
            value.validate()?;
        }
        Ok(Self {
            identity,
            value,
            namespace: None,
        })
    }
    /// The caller obtains the explicit mapping from the actual native startup
    /// snapshot, rather than deriving representation keys from display IDs.
    pub fn new_with_namespace(
        identity: String,
        value: Option<ModelValueIdentity>,
        namespace: AdaptiveModelNamespace,
    ) -> Result<Self, PalsError> {
        namespace.verify_identity(&identity, value.as_ref())?;
        let mut model = Self::new(identity, value)?;
        model.namespace = Some(namespace);
        Ok(model)
    }
}
impl RoleModel for AdaptivePortModel {
    fn identity(&self) -> &str {
        &self.identity
    }
    fn supports_selected_review_view(&self) -> bool {
        true
    }
    fn requires_registered_representation_namespace(&self) -> bool {
        true
    }
    fn registered_representation_namespace(&self) -> Option<&AdaptiveModelNamespace> {
        self.namespace.as_ref()
    }
    fn value_identity(&self) -> Option<&ModelValueIdentity> {
        self.value.as_ref()
    }
    fn propose(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        Err(RoleError::Unsupported("adaptive role owner is external"))
    }
    fn reply(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        Err(RoleError::Unsupported("adaptive role owner is external"))
    }
    fn repair(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        Err(RoleError::Unsupported("adaptive role owner is external"))
    }
    fn divergences(&mut self, _: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
        Err(RoleError::Unsupported("adaptive role owner is external"))
    }
}

/// Search owns the opaque continuation fields. Native consumes the submitted
/// Rules/context/deadline and an optional legacy raw snapshot; it cannot
/// substitute a newer line, raw projection, or revision on delivery.
#[derive(Clone)]
pub struct AdaptivePortQuestion {
    pub position: Position,
    pub context: RoleLogicalContext,
    pub deadline: Instant,
    /// Present only for the explicitly registered V2Full native baseline.
    /// Search captured every raw row before dispatch; the native worker never
    /// reads or scans mutable engine memory to reconstruct this input.
    pub legacy_full_records: Option<Arc<LegacyFullRoleRecords>>,
    node: usize,
    stage: RefinementStage,
    first: Option<BoardMove>,
    proposal: Vec<BoardMove>,
    counter: Option<Vec<BoardMove>>,
    repaired: Option<Vec<BoardMove>>,
    attack_ply: Option<usize>,
    discovery_restricted: bool,
    counter_source: Option<ObservationId>,
    validator_root_moves: Option<Vec<BoardMove>>,
    recheck: bool,
    frozen_counter: Option<Arc<ReviewView>>,
    paired_repaired: Option<ObservationId>,
    paired_fresh_seal: Option<FreshPairSeal>,
    question_sequence: u64,
    question_digest: ContractDigest,
    frontier_source: frontier_port::FrontierQuestionSource,
    goal_local: Option<Arc<goal_local::GoalLocalOrigin>>,
    // Observer-only inline Copy metadata. Never hashed, encoded, selected or
    // consulted by Rules, a cache, acceptance, scheduling or the model.
    diagnostic_repair_line: Option<AdaptiveRepairLineProvenance>,
}
/// Read-only hash preimage. Equal bytes neither recreate the original Question
/// Arc nor grant a frame, native request, or acceptance authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveQuestionDigestPreimage {
    pub sequence: u64,
    pub stage_code: u8,
    pub view_digest: ContractDigest,
    pub view_seal_digest: ContractDigest,
    pub focus_prefix_sha256: ContractDigest,
    pub goal_local_sha256: Option<ContractDigest>,
    pub legacy_full_records_sha256: Option<ContractDigest>,
}
impl AdaptiveQuestionDigestPreimage {
    pub fn digest(&self) -> ContractDigest {
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-adaptive-checked-question/1\0");
        hash.update(self.sequence.to_le_bytes());
        hash.update([self.stage_code]);
        hash.update(self.view_digest.0);
        hash.update(self.view_seal_digest.0);
        hash.update(self.focus_prefix_sha256.0);
        if let Some(origin) = self.goal_local_sha256 {
            hash.update(b"goal-local-rules-origin\0");
            hash.update(origin.0);
        }
        if let Some(records) = self.legacy_full_records_sha256 {
            hash.update(b"legacy-full-raw-snapshot\0");
            hash.update(records.0);
        }
        ContractDigest(hash.finalize().into())
    }
}
impl AdaptivePortQuestion {
    /// Conservative known owned host backing, never an RSS/physical peak.
    pub fn owned_host_bytes(&self) -> Option<u64> {
        let mut bytes = std::mem::size_of::<Self>();
        bytes = bytes.checked_add(self.position.snapshot().retained_history_bytes()?)?;
        for capacity in [
            self.context.prefix.capacity(),
            self.proposal.capacity(),
            self.counter.as_ref().map_or(0, Vec::capacity),
            self.repaired.as_ref().map_or(0, Vec::capacity),
            self.validator_root_moves.as_ref().map_or(0, Vec::capacity),
        ] {
            bytes = bytes.checked_add(capacity.checked_mul(std::mem::size_of::<BoardMove>())?)?;
        }
        if let Some(view) = &self.frozen_counter {
            bytes = bytes.checked_add(usize::try_from(view.owned_host_bytes()?).ok()?)?;
        }
        if let Some(records) = &self.legacy_full_records {
            bytes = bytes.checked_add(usize::try_from(records.owned_host_bytes()?).ok()?)?;
        }
        if let Some(origin) = &self.goal_local {
            bytes = bytes.checked_add(usize::try_from(origin.owned_host_bytes()?).ok()?)?;
        }
        u64::try_from(bytes).ok()
    }
}
pub struct AdaptivePortCandidate {
    pub first_move: rz_contracts::pals::Move16,
    pub input: FrameInput<ReviewView, AdaptivePortQuestion>,
    pub budget: FrameBudget,
    pub deferred: Option<DeferredRootCandidate<AdaptivePortQuestion>>,
    pub initialized: Option<CheckedRootCandidate<AdaptivePortQuestion>>,
}
/// The actual port alone attests a one-ply policy shortcut within its
/// registered chunk and the admitted original go horizon.
/// General public FrameInput cannot claim that a second forward was omitted.
pub struct CheckedRootCandidate<Q> {
    source: ProposerRankingSource,
    first: rz_contracts::pals::Move16,
    input: FrameInput<ReviewView, Q>,
    budget: FrameBudget,
    line_plies: usize,
}
impl<Q> CheckedRootCandidate<Q> {
    pub fn source(&self) -> ProposerRankingSource {
        self.source
    }
    pub fn first_move(&self) -> rz_contracts::pals::Move16 {
        self.first
    }
    pub fn input(&self) -> &FrameInput<ReviewView, Q> {
        &self.input
    }
    pub fn budget(&self) -> FrameBudget {
        self.budget
    }
    pub fn into_candidate(
        self,
    ) -> (
        rz_contracts::pals::Move16,
        FrameInput<ReviewView, Q>,
        FrameBudget,
    ) {
        (self.first, self.input, self.budget)
    }
    pub fn validate(&self, ranking: &ProposerRankingPublished) -> Result<(), PalsError> {
        let movement =
            BoardMove::try_from(self.first.decode().map_err(|_| RoleError::InvalidOutput)?)
                .map_err(|_| RoleError::InvalidOutput)?;
        if self.line_plies != 1
            || self.source != ranking.source()
            || !ranking.candidates().contains(&self.first)
            || self.input.stage != RefinementStage::CpuValidation
            || self.input.view.query().prefix != [movement]
            || self.input.view.query().proposal != [movement]
            || self.input.view_seal.authority != self.source.ticket.authority
            || self.input.view_seal.root_state != self.source.ticket.view.root_state
            || self.input.view_seal.view_digest.0 != self.input.view.seal_digest()
            || self.input.public_context.0 != self.input.view.record_scope_digest()
            || !self.input.view.anchors()[0]
                .snapshot
                .same_state(ranking.rules().rules().snapshot())
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }
}
/// Issued only by the sequential port after a prior actual main frame closes.
/// The original root-head source remains separate from the new candidate view.
pub struct DeferredRootCandidate<Q> {
    source: ProposerRankingSource,
    first: rz_contracts::pals::Move16,
    input: FrameInput<ReviewView, Q>,
    budget: FrameBudget,
    previous: crate::pals::coordinator::FrameId,
}
impl<Q> DeferredRootCandidate<Q> {
    pub fn source(&self) -> ProposerRankingSource {
        self.source
    }
    pub fn first_move(&self) -> rz_contracts::pals::Move16 {
        self.first
    }
    pub fn input(&self) -> &FrameInput<ReviewView, Q> {
        &self.input
    }
    pub fn budget(&self) -> FrameBudget {
        self.budget
    }
    pub fn previous_frame(&self) -> crate::pals::coordinator::FrameId {
        self.previous
    }
    pub fn into_candidate(
        self,
    ) -> (
        rz_contracts::pals::Move16,
        FrameInput<ReviewView, Q>,
        FrameBudget,
    ) {
        (self.first, self.input, self.budget)
    }
    pub fn validate(&self, ranking: &ProposerRankingPublished) -> Result<(), PalsError> {
        let movement =
            BoardMove::try_from(self.first.decode().map_err(|_| RoleError::InvalidOutput)?)
                .map_err(|_| RoleError::InvalidOutput)?;
        if self.source != ranking.source()
            || !ranking.candidates().contains(&self.first)
            || self.previous.authority != self.source.ticket.authority
            || self.input.view_seal.authority != self.source.ticket.authority
            || self.input.view_seal.root_state != self.source.ticket.view.root_state
            || self.input.public_context.0 != self.input.view.record_scope_digest()
            || self.input.view_seal.view_digest.0 != self.input.view.seal_digest()
            || !matches!(
                self.input.stage,
                RefinementStage::ProposalContinuation | RefinementStage::CpuValidation
            )
            || self.input.view.query().prefix != [movement]
            || !self.input.view.anchors()[0]
                .snapshot
                .same_state(ranking.rules().rules().snapshot())
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AdaptiveRootFanoutPolicy {
    SequentialFixed,
    #[default]
    SimultaneousAdaptive,
}
pub struct AdaptivePortStep {
    pub evidence: ContractDigest,
    pub transition: AdaptivePortTransition,
    consumption_gate: Option<(RequestId, ContractDigest, ContractDigest)>,
    feedback_gate: Option<(ContractDigest, ObservationId)>,
    queue_question: Option<(FrameId, ContractDigest)>,
    pub fresh_pair: Option<FreshPairSeal>,
    /// Present only when both exact Rules endpoints were compared. This is
    /// never a substitute for physical NN completion or a Fresh pair seal.
    pub rules_pair: Option<AdaptiveRulesPairProof>,
}
/// Search-owned proof of two exact terminal endpoints under one frozen public
/// review scope. Construction remains private to the checked semantic port.
#[derive(Clone)]
pub struct AdaptiveRulesPairProof {
    authority: SearchAuthority,
    root_state: rz_contracts::StateIdentity,
    public_context: ContractDigest,
    submitted_question: ContractDigest,
    repaired_line: Vec<BoardMove>,
    counter_line: Vec<BoardMove>,
    repaired: Arc<ContractState>,
    counter: Arc<ContractState>,
    perspective: rz_contracts::Color,
}
impl AdaptiveRulesPairProof {
    pub fn validate(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), PalsError> {
        self.validate_frozen_scope(work)?;
        if work.input.question.repaired.as_ref() != Some(&self.repaired_line)
            || (work.input.stage == RefinementStage::RecheckCounterValidation
                && work.input.question.counter.as_ref() != Some(&self.counter_line))
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }

    /// The coordinator may validate its opaque Q without depending on native
    /// or search-private continuation fields. No terminal flag is trusted.
    pub fn validate_frozen_scope<Q>(
        &self,
        work: &CoordinatorWork<ReviewView, Q>,
    ) -> Result<(), PalsError> {
        let original_first = BoardMove::try_from(
            work.original_first_move
                .decode()
                .map_err(|_| RoleError::InvalidOutput)?,
        )
        .map_err(|_| RoleError::InvalidOutput)?;
        if work.frame.authority != self.authority
            || !matches!(
                work.input.stage,
                RefinementStage::CriticRecheck | RefinementStage::RecheckCounterValidation
            )
            || work.input.public_context != self.public_context
            || work.input.question_digest != self.submitted_question
            || work.input.public_context.0 != work.input.view.record_scope_digest()
            || work.input.view_seal.view_digest.0 != work.input.view.seal_digest()
            || work.input.view_seal.authority != self.authority
            || work.input.view_seal.root_state != self.root_state
            || self.repaired_line.first() != Some(&original_first)
            || self.repaired_line.first() != self.counter_line.first()
            || self.repaired_line.len() != self.counter_line.len()
            || self.repaired.terminal_wdl().is_none()
            || self.counter.terminal_wdl().is_none()
            || rz_contracts::Color::from(work.input.view.anchors()[0].snapshot.side_to_move())
                != self.perspective
        {
            return Err(RoleError::InvalidOutput.into());
        }
        for (line, endpoint) in [
            (&self.repaired_line, &self.repaired),
            (&self.counter_line, &self.counter),
        ] {
            let mut replay = Position::from_snapshot(
                &work.input.view.anchors()[0].snapshot,
                rz_position::PositionLimits::default(),
            )?;
            for movement in line {
                replay.make_move(*movement)?;
            }
            if !replay.snapshot().same_state(endpoint.rules().snapshot()) {
                return Err(ReviewError::AnchorMismatch.into());
            }
        }
        Ok(())
    }
    pub fn endpoints(&self) -> (&Arc<ContractState>, &Arc<ContractState>) {
        (&self.repaired, &self.counter)
    }
    pub fn perspective(&self) -> rz_contracts::Color {
        self.perspective
    }
}
#[derive(Clone, Copy, Debug)]
pub struct AdaptivePortRoleEvidence {
    pub execution: Option<rz_contracts::ExecutionId>,
    /// Actual completed physical NN inputs, not semantic value/output count.
    /// A Fresh role may execute both a public graph and a private role graph.
    pub completed_nn_inputs: Option<u64>,
    pub measurement_complete: bool,
    pub physical_fresh: Option<bool>,
}
pub struct AdaptivePortRulesStep {
    pub endpoint: Arc<ContractState>,
    pub checked: AdaptivePortStep,
}
pub struct AdaptivePortCpuQuestion {
    pub position: ContractPosition,
    pub root_moves: Option<Vec<BoardMove>>,
    pub target_depth: u16,
    pub total_native_node_credit: u64,
}

/// Keeps existing Rules graph, observations, resolver and bounded review index
/// as the authoritative implementation; workers own all physical execution.
pub struct AdaptiveEnginePort<M: RoleModel> {
    engine: PalsEngine<M>,
    root: usize,
    rules: Arc<ContractState>,
    authority: SearchAuthority,
    deadline: Deadline,
    limits: PalsLimits,
    started: Instant,
    counters: PalsCounters,
    next_question: u64,
    root_rules_scanned: bool,
    root_round: u64,
    root_ranking_pending: bool,
    root_ranking_question: Option<ContractDigest>,
    root_fanout: AdaptiveRootFanoutPolicy,
    lazy_frontier: VecDeque<BoardMove>,
    sequential_ranking_source: Option<ProposerRankingSource>,
    last_main_closed: Option<crate::pals::coordinator::FrameId>,
    round_candidates:
        HashMap<ContractDigest, (BoardMove, Option<crate::pals::coordinator::FrameId>)>,
    accepted: HashSet<RequestId>,
    attempted_role_commits: HashSet<RequestId>,
    consumed_role_views: HashMap<RequestId, ContractDigest>,
    role_consumption_observation: Option<AdaptiveRoleConsumptionObservation>,
    role_consumption_observation_started: bool,
    committed_role_questions: HashMap<RequestId, ContractDigest>,
    last_role_commit: Option<(ContractDigest, RequestId, usize, usize)>,
    closed: HashSet<crate::pals::coordinator::FrameId>,
    cpu_admissions: HashMap<ContractDigest, CpuPortAdmission>,
    fresh_pending: HashMap<ContractDigest, FrameInput<ReviewView, AdaptivePortQuestion>>,
    rules_owner_issuer: AdaptiveRulesOwnerIssuer,
    external_admissions: HashMap<ContractDigest, external_commit::ExternalPortAdmission>,
    profiled_external: profiled_external_port::ProfiledExternalPortState,
    supported_counters: HashMap<ContractDigest, conditional_evidence::SupportedCounter>,
    conditional_witnesses: HashMap<
        ObservationId,
        (
            ObservationKind,
            Vec<BoardMove>,
            Option<utility_conditional_basis::ConditionalComparisonSource>,
        ),
    >,
    pending_counters: HashMap<BoardMove, VecDeque<branch_queue::PendingCounter>>,
    deferred_rechecks: VecDeque<branch_queue::DeferredRecheck>,
    /// Exact consumed queue admissions, not a certificate that their parent
    /// already closed. Deferred CPU use separately requires actual closure.
    queued_recheck_closures:
        HashMap<(crate::pals::coordinator::FrameId, ContractDigest), BoardMove>,
    branch_issues: HashMap<BoardMove, &'static str>,
    branch_receipts: Vec<branch_queue::AdaptiveBranchReceipt>,
    queued_questions: HashSet<ContractDigest>,
    validator_state: validator_port::ValidatorPortState,
    /// A weak cache shares an identical raw snapshot without extending a
    /// completed ticket's Store lease. Only the opt-in legacy route uses it.
    last_legacy_full_snapshot: Option<std::sync::Weak<LegacyFullRoleRecords>>,
    reciprocal: reciprocal_port::ReciprocalPortState,
    nnuec_installations: nnuec_installation::NnuecQuestionInstallations,
    goal_local: goal_local::GoalLocalPortState,
    goal_lanes: goal_hooks::AdaptiveGoalLanes,
    utility_continuations: utility_dependencies::UtilityContinuationRegistry,
}
#[derive(Clone)]
struct CpuPortAdmission {
    node: usize,
    frame: crate::pals::coordinator::FrameId,
    stage: RefinementStage,
    view_seal: ReviewViewSeal,
    context: RoleLogicalContext,
    first_move: rz_contracts::pals::Move16,
    execution: Option<ExecutionId>,
    dispatch: Option<crate::pals::store::CpuTaskDispatch>,
    consumer: Option<u64>,
    original_consumer: Option<TaskConsumer>,
    target_depth: u16,
    credit: u64,
    restriction: Option<Vec<BoardMove>>,
    generation: u64,
    root: SituationId,
    root_revision: u64,
    root_snapshot: PositionSnapshot,
    position: Position,
    rules_identity: rz_contracts::StateIdentity,
    situation: SituationId,
    situation_revision: u64,
    task_key: TaskKey,
    value_identity: CpuValueIdentity,
    base_condition: String,
    profiles: Vec<crate::cpu_worker::CpuWorkerProfile>,
    evaluation_namespace: ContractDigest,
    search_conditions: ContractDigest,
    job: Option<rz_contracts::pals_adaptive::CpuJobEnvelope>,
    grant: Option<rz_contracts::pals_adaptive::SliceGrant>,
    owner: Option<rz_contracts::pals_adaptive::CpuWorkerId>,
    cumulative_nodes: u64,
    progress_sequence: u64,
    control_sequence: u64,
    execution_sequence: u64,
    last_started_sequence: u64,
    committed_execution: Option<(rz_contracts::ExecutionId, ExecutionId, ObservationId)>,
    consumed_observation: Option<ObservationId>,
    settled_receipt: Option<(rz_contracts::ExecutionId, ExecutionId, ContractDigest)>,
    continuation_retained: bool,
    terminal: bool,
    quarantined: bool,
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    pub fn new(
        engine: PalsEngine<M>,
        position: Position,
        authority: SearchAuthority,
        deadline: Deadline,
        limits: PalsLimits,
        rules_owner_issuer: AdaptiveRulesOwnerIssuer,
    ) -> Result<Self, PalsError> {
        Self::new_inner(
            engine,
            position,
            authority,
            deadline,
            limits,
            rules_owner_issuer,
            None,
        )
        .map_err(|(_, error)| error)
    }

    /// Consume a one-use registration for the actual global UCI root. The
    /// caller has already fenced and joined physical owners before transfer.
    #[allow(clippy::too_many_arguments)]
    pub fn new_registered(
        engine: PalsEngine<M>,
        position: Position,
        authority: SearchAuthority,
        deadline: Deadline,
        limits: PalsLimits,
        rules_owner_issuer: AdaptiveRulesOwnerIssuer,
        registration: super::adaptive_root::AdaptiveRootRegistration,
    ) -> Result<Self, PalsError> {
        Self::new_inner(
            engine,
            position,
            authority,
            deadline,
            limits,
            rules_owner_issuer,
            Some(registration),
        )
        .map_err(|(_, error)| error)
    }

    /// Lifecycle owners can retain every Store/index allocation on a failed
    /// registration or bounded hot admission. This does not assert a physical
    /// worker fence; the caller must supply its actual quiescence evidence.
    // The rejected engine must return intact even under allocation pressure;
    // allocating a boxed error would add a new failure point to owner recovery.
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn new_registered_retaining(
        engine: PalsEngine<M>,
        position: Position,
        authority: SearchAuthority,
        deadline: Deadline,
        limits: PalsLimits,
        rules_owner_issuer: AdaptiveRulesOwnerIssuer,
        registration: super::adaptive_root::AdaptiveRootRegistration,
    ) -> Result<Self, (PalsEngine<M>, PalsError)> {
        Self::new_inner(
            engine,
            position,
            authority,
            deadline,
            limits,
            rules_owner_issuer,
            Some(registration),
        )
    }

    // Both callers recover the original engine by value on failed admission;
    // the hot path must not allocate an error box to retain that ownership.
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn new_inner(
        mut engine: PalsEngine<M>,
        position: Position,
        authority: SearchAuthority,
        deadline: Deadline,
        limits: PalsLimits,
        rules_owner_issuer: AdaptiveRulesOwnerIssuer,
        registration: Option<super::adaptive_root::AdaptiveRootRegistration>,
    ) -> Result<Self, (PalsEngine<M>, PalsError)> {
        let prepared = (|| -> Result<(usize, Arc<ContractState>), PalsError> {
            if engine.semantic.review_index.is_none()
                || engine.semantic.review_policy.is_none()
                || limits.max_rounds == 0
                || limits.max_cpu_nodes == 0
                || limits.cpu_depth == 0
                || deadline.clock.0 != authority.epoch
                || engine.semantic.game_generation != Some(authority.game.0)
                || Instant::now() >= limits.deadline
            {
                return Err(PalsError::InvalidLimits);
            }
            if engine
                .semantic
                .model
                .requires_registered_representation_namespace()
                && engine
                    .semantic
                    .model
                    .registered_representation_namespace()
                    .is_none()
            {
                return Err(RoleError::Unsupported(
                    "native representation namespace is not registered",
                )
                .into());
            }
            engine.validate_checker_namespace()?;
            engine.ensure_archive_admission_open()?;
            match registration {
                Some(registration) => {
                    engine.consume_adaptive_root_registration(
                        registration,
                        &position,
                        authority,
                    )?;
                }
                None => {
                    engine
                        .semantic
                        .stores
                        .hot_focus_actual_moves(position.snapshot())?;
                }
            }
            if engine.semantic.stores.generation() != authority.root.0 {
                return Err(StoreError::StaleConsumer.into());
            }
            let root = Self::intern_hot(&mut engine, position.clone())?;
            for node in &mut engine.semantic.nodes {
                node.model_value = None;
                node.model_observation = None;
                node.model_value_revision = None;
            }
            engine.refresh_projection(engine.semantic.nodes[root].state);
            let rules = Arc::new(
                ContractPosition::new(
                    rules_owner_issuer().map_err(|_| RoleError::InvalidOutput)?,
                    position,
                )
                .export()
                .map_err(|_| RoleError::InvalidOutput)?,
            );
            Ok((root, rules))
        })();
        let (root, rules) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return Err((engine, error)),
        };
        Ok(Self::from_admitted_root(
            engine,
            root,
            rules,
            authority,
            deadline,
            limits,
            rules_owner_issuer,
            Instant::now(),
            false,
        ))
    }

    /// Only existing successful admission or an opaque PreparingRoot handoff
    /// reaches this initializer. It issues no Rules owner or root nonce.
    #[allow(clippy::too_many_arguments)]
    fn from_admitted_root(
        engine: PalsEngine<M>,
        root: usize,
        rules: Arc<ContractState>,
        authority: SearchAuthority,
        deadline: Deadline,
        limits: PalsLimits,
        rules_owner_issuer: AdaptiveRulesOwnerIssuer,
        started: Instant,
        root_rules_scanned: bool,
    ) -> Self {
        Self {
            engine,
            root,
            rules,
            authority,
            deadline,
            limits,
            started,
            counters: PalsCounters::default(),
            next_question: 1,
            root_rules_scanned,
            root_round: 0,
            root_ranking_pending: false,
            root_ranking_question: None,
            root_fanout: AdaptiveRootFanoutPolicy::default(),
            lazy_frontier: VecDeque::new(),
            sequential_ranking_source: None,
            last_main_closed: None,
            round_candidates: HashMap::new(),
            accepted: HashSet::new(),
            attempted_role_commits: HashSet::new(),
            consumed_role_views: HashMap::new(),
            role_consumption_observation: None,
            role_consumption_observation_started: false,
            committed_role_questions: HashMap::new(),
            last_role_commit: None,
            closed: HashSet::new(),
            cpu_admissions: HashMap::new(),
            fresh_pending: HashMap::new(),
            rules_owner_issuer,
            external_admissions: HashMap::new(),
            profiled_external: profiled_external_port::ProfiledExternalPortState::default(),
            supported_counters: HashMap::new(),
            conditional_witnesses: HashMap::new(),
            pending_counters: HashMap::new(),
            deferred_rechecks: VecDeque::new(),
            queued_recheck_closures: HashMap::new(),
            branch_issues: HashMap::new(),
            branch_receipts: Vec::new(),
            queued_questions: HashSet::new(),
            validator_state: validator_port::ValidatorPortState::default(),
            last_legacy_full_snapshot: None,
            reciprocal: reciprocal_port::ReciprocalPortState::default(),
            nnuec_installations: nnuec_installation::NnuecQuestionInstallations::default(),
            goal_local: goal_local::GoalLocalPortState::default(),
            goal_lanes: goal_hooks::AdaptiveGoalLanes::default(),
            utility_continuations: utility_dependencies::UtilityContinuationRegistry::default(),
        }
    }
    /// Explicit root-local diagnostic opt-in before the first observed call.
    /// No callback, worker, acceptance or coordinator-transition authority.
    /// False prevents resetting history or silently starting a partial journal.
    pub fn enable_role_consumption_observation(&mut self) -> bool {
        if self.role_consumption_observation_started || self.role_consumption_observation.is_some()
        {
            return false;
        }
        self.role_consumption_observation =
            Some(AdaptiveRoleConsumptionObservation::new(self.authority));
        true
    }
    /// Borrow only the bounded copied facts. None means not opted in; an
    /// unavailable journal never certifies complete or exactly-once coverage.
    pub fn role_consumption_observation(&self) -> Option<&AdaptiveRoleConsumptionObservation> {
        self.role_consumption_observation.as_ref()
    }
    pub fn root_rules(&self) -> Arc<ContractState> {
        Arc::clone(&self.rules)
    }
    pub fn root_state_id(&self) -> StateId {
        self.engine.semantic.nodes[self.root].state
    }
    pub fn model_namespace(&self) -> Result<&AdaptiveModelNamespace, PalsError> {
        let namespace = self
            .engine
            .semantic
            .model
            .registered_representation_namespace()
            .ok_or(RoleError::Unsupported(
                "native representation namespace is not registered",
            ))?;
        namespace.verify_identity(
            self.engine.semantic.model.identity(),
            self.engine.semantic.model.value_identity(),
        )?;
        if self
            .engine
            .semantic
            .model_registered_value
            .as_ref()
            .is_some_and(|value| value != namespace.value_identity())
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(namespace)
    }
    pub fn root_is_terminal(&self) -> bool {
        self.engine.semantic.nodes[self.root].terminal.is_some()
    }

    /// Rules covers every immediate legal child before any model beam is used.
    /// This bounded replay neither dispatches a worker nor invents NN work.
    pub fn prepare_root_rules(&mut self) -> Result<(), PalsError> {
        self.check_live()?;
        if self.root_rules_scanned {
            return Ok(());
        }
        if self.engine.semantic.rovent_frontier.is_some() {
            return Err(StoreError::InvalidConditions(
                "Rovent root Rules inspection requires its original clock",
            )
            .into());
        }
        let position = self.engine.semantic.nodes[self.root].position.clone();
        for movement in position.legal_moves() {
            self.check_live()?;
            let mut child = position.clone();
            child.make_move(movement)?;
            if matches!(
                child.classify_position()?.play_status,
                PlayStatus::Terminal { .. }
            ) {
                self.connect_hot(self.root, movement)?;
            }
        }
        self.root_rules_scanned = true;
        Ok(())
    }
    pub fn engine(&self) -> &PalsEngine<M> {
        &self.engine
    }
    pub fn counters(&self) -> PalsCounters {
        self.counters
    }
    /// Root-wide logical role admission ceiling, including root rankings.
    /// The physical manager reserves this before preparing a question; it is
    /// independent of NN input counts and successful semantic consumption.
    pub fn role_call_limit(&self) -> u64 {
        self.engine.config.max_role_calls
    }
    pub(super) fn role_action_committed(
        &self,
        question: ContractDigest,
        request: RequestId,
        observation_start: usize,
        observation_end: usize,
    ) -> bool {
        self.accepted.contains(&request)
            && self.last_role_commit
                == Some((question, request, observation_start, observation_end))
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    /// Pure ownership transfer. This method attests no drain, worker join,
    /// archive install or physical fence; the concrete owner must verify them.
    pub fn into_engine(mut self) -> PalsEngine<M> {
        // The admitted-root constructors reject a previously retired owner.
        // Any unfinished logical source and even a failed pin projection must
        // move with this same Store instead of disappearing with the facade.
        let pins = self.goal_local_protected_pins();
        let state = std::mem::take(&mut self.goal_local);
        self.engine.semantic.retired_goal_local = goal_local::GoalLocalRetiredCustody::preserve(
            state,
            pins,
            self.engine.semantic.stores.read_owner(),
            self.engine.semantic.stores.generation(),
            self.authority,
        );
        let pins = self.profiled_external_protected_pins();
        let state = std::mem::take(&mut self.profiled_external);
        self.engine.semantic.retired_profiled_external = ProfiledExternalRetiredCustody::preserve(
            state,
            pins,
            self.engine.semantic.stores.read_owner(),
            self.engine.semantic.stores.generation(),
            self.authority,
        );
        self.engine
    }
    /// Register scheduling before accepting any role. A previously prepared
    /// root-ranking view is unaffected; candidate views have not been created.
    pub fn configure_root_fanout(
        &mut self,
        policy: AdaptiveRootFanoutPolicy,
    ) -> Result<(), PalsError> {
        if !self.accepted.is_empty()
            || !self.round_candidates.is_empty()
            || !self.lazy_frontier.is_empty()
            || !self.deferred_rechecks.is_empty()
        {
            return Err(StoreError::InvalidConditions("root fanout already started").into());
        }
        self.root_fanout = policy;
        Ok(())
    }
    fn issue_rules_position(&self, position: Position) -> Result<ContractPosition, PalsError> {
        Ok(ContractPosition::new(
            (self.rules_owner_issuer)().map_err(|_| RoleError::InvalidOutput)?,
            position,
        ))
    }

    pub fn root_input(
        &mut self,
    ) -> Result<FrameInput<ReviewView, AdaptivePortQuestion>, PalsError> {
        self.prepare_root_rules()?;
        if self.root_is_terminal() {
            return Err(StoreError::InvalidConditions(
                "Rules-terminal root has no policy question",
            )
            .into());
        }
        if self.root_ranking_pending
            || !self.round_candidates.is_empty()
            || !self.lazy_frontier.is_empty()
        {
            return Err(StoreError::InvalidConditions("root round is already active").into());
        }
        let input = self.input(
            self.root,
            RefinementStage::ProposerRanking,
            None,
            Vec::new(),
            Vec::new(),
            None,
            None,
            None,
            false,
            None,
            None,
        )?;
        self.root_ranking_pending = true;
        self.root_ranking_question = Some(input.question_digest);
        Ok(input)
    }

    /// Bind only a generated main candidate to the actual machine-issued ID.
    /// V child frames and conditional branches do not count as root iterations.
    pub fn root_candidate_admitted(
        &mut self,
        frame: crate::pals::coordinator::FrameId,
        first: rz_contracts::pals::Move16,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), PalsError> {
        let movement = BoardMove::try_from(first.decode().map_err(|_| RoleError::InvalidOutput)?)
            .map_err(|_| RoleError::InvalidOutput)?;
        let expected = self
            .round_candidates
            .get(&input.question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        if expected.0 != movement || expected.1.is_some() || frame.authority != self.authority {
            return Err(RoleError::InvalidOutput.into());
        }
        // This admission only validates identity and Rules inputs. It must not
        // fabricate a CoordinatorWork with freshly issued resource credit.
        self.check_work_input(frame, self.deadline, input)?;
        self.round_candidates
            .get_mut(&input.question_digest)
            .expect("checked root candidate")
            .1 = Some(frame);
        Ok(())
    }

    /// Called at the manager's quiescent boundary. Completed main frames finish
    /// one root round together; frame count is never renamed as iteration count.
    pub fn next_root_round_input(
        &mut self,
    ) -> Result<Option<FrameInput<ReviewView, AdaptivePortQuestion>>, PalsError> {
        self.check_live()?;
        if self.root_ranking_pending
            || !self.lazy_frontier.is_empty()
            || self.round_candidates.is_empty()
            || self
                .round_candidates
                .values()
                .any(|(_, frame)| frame.is_none_or(|frame| !self.closed.contains(&frame)))
        {
            return Ok(None);
        }
        self.root_round = self.root_round.checked_add(1).ok_or(PalsError::Capacity)?;
        self.counters.rounds = self
            .counters
            .rounds
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        self.round_candidates.clear();
        self.sequential_ranking_source = None;
        self.last_main_closed = None;
        if self.rovent_frontier().is_some_and(|frontier| {
            let goal = frontier.registration().goal;
            goal.depth_plies.is_some() || goal.mate_moves.is_some()
        }) || self.root_round >= self.limits.max_rounds
            || self.engine.cpu_budget_used(&self.counters) >= self.limits.max_cpu_nodes
            || self.counters.role_calls >= self.engine.config.max_role_calls
        {
            return Ok(None);
        }
        self.root_input().map(Some)
    }
    /// Fixed mode retains only unsubmitted move identities. Construct the next
    /// view after earlier candidate publications; no existing frame or ticket
    /// is refreshed and no second root forward is claimed.
    pub fn next_fixed_root_candidate(
        &mut self,
    ) -> Result<Option<AdaptivePortCandidate>, PalsError> {
        self.check_live()?;
        if self.root_fanout != AdaptiveRootFanoutPolicy::SequentialFixed
            || self.root_ranking_pending
            || self.counters.role_calls >= self.engine.config.max_role_calls
            || self
                .round_candidates
                .values()
                .any(|(_, frame)| frame.is_none_or(|frame| !self.closed.contains(&frame)))
        {
            return Ok(None);
        }
        let Some(movement) = self.lazy_frontier.front().copied() else {
            return Ok(None);
        };
        let source = self
            .sequential_ranking_source
            .ok_or(RoleError::InvalidOutput)?;
        let previous = self.last_main_closed.ok_or(RoleError::InvalidOutput)?;
        if !self.closed.contains(&previous)
            || previous.authority != self.authority
            || source.ticket.authority != self.authority
            || source.ticket.view.root_state != self.rules.snapshot().identity()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let mut candidate = self.make_root_candidate(movement)?;
        candidate.deferred = Some(DeferredRootCandidate {
            source,
            first: candidate.first_move,
            input: candidate.input.clone(),
            budget: candidate.budget,
            previous,
        });
        self.round_candidates
            .insert(candidate.input.question_digest, (movement, None));
        self.lazy_frontier.pop_front();
        Ok(Some(candidate))
    }

    /// Rules-terminal endpoints skip physical CPU/NN dispatch while any
    /// earlier opponent divergence remains eligible for conditional review.
    pub fn short_circuit(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<Option<AdaptivePortRulesStep>, PalsError> {
        self.check_work(work)?;
        let query = &work.input.question;
        if self.engine.semantic.nodes[query.node].terminal.is_none()
            || !matches!(
                query.stage,
                RefinementStage::CpuValidation
                    | RefinementStage::CounterValidation
                    | RefinementStage::RepairValidation
                    | RefinementStage::RecheckCounterValidation
                    | RefinementStage::RecheckRepairedValueFresh
            )
        {
            return Ok(None);
        }
        let endpoint = Arc::new(
            self.issue_rules_position(query.position.clone())?
                .export()
                .map_err(|_| RoleError::InvalidOutput)?,
        );
        if endpoint.terminal_wdl().is_none() {
            return Err(RoleError::InvalidOutput.into());
        }
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-ready-Rules-terminal/1\0");
        hash.update(work.input.question_digest.0);
        hash.update(endpoint.snapshot().identity().semantic.0);
        let (transition, rules_pair) = if query.stage == RefinementStage::RecheckCounterValidation {
            self.prepare_final_pair(
                work,
                query.repaired.clone().ok_or(RoleError::InvalidOutput)?,
                query.counter.clone().ok_or(RoleError::InvalidOutput)?,
            )?
        } else if query.stage == RefinementStage::RecheckRepairedValueFresh {
            (self.critic_recheck_after_repaired_fresh(query)?, None)
        } else {
            (self.after_validation(query, Some(work))?, None)
        };
        let transition = self.finish_branch(work, transition)?;
        Ok(Some(AdaptivePortRulesStep {
            endpoint,
            checked: AdaptivePortStep {
                evidence: ContractDigest(hash.finalize().into()),
                transition,
                consumption_gate: None,
                feedback_gate: None,
                queue_question: Some((work.frame, work.input.question_digest)),
                fresh_pair: None,
                rules_pair,
            },
        }))
    }

    pub fn ranking_published(
        &mut self,
        ranking: &ProposerRankingPublished,
    ) -> Result<Vec<AdaptivePortCandidate>, PalsError> {
        self.check_live()?;
        let observations_before = self
            .engine
            .semantic
            .stores
            .observations
            .next_observation_sequence();
        if ranking.source().ticket.authority != self.authority
            || !self.root_ranking_pending
            || self.root_ranking_question != Some(ranking.source().ticket.question)
            || !self.round_candidates.is_empty()
            || !ranking.view().validates(&self.engine.semantic.stores)
            || ranking.source().view_input_digest.0 != ranking.view().digest()
            || !ranking
                .rules()
                .rules()
                .snapshot()
                .same_state(self.rules.rules().snapshot())
        {
            return Err(RoleError::InvalidOutput.into());
        }
        self.verify_registered_ticket_namespace(&ranking.source().ticket)?;
        if self.counters.role_calls >= self.engine.config.max_role_calls {
            return Err(PalsError::RoleCallLimit);
        }
        let mut ranked = Vec::new();
        for movement in ranking.ranked_candidates() {
            let movement =
                BoardMove::try_from(movement.decode().map_err(|_| RoleError::InvalidOutput)?)
                    .map_err(|_| RoleError::InvalidOutput)?;
            if self.root_move_allowed(movement) {
                ranked.push(movement);
            }
        }
        // Preserve root candidate selection and deterministic ties. Independent
        // frames carry the same already-consumed ranking source, never new NN work.
        let width = self
            .engine
            .config
            .beam_width
            .saturating_add(usize::try_from(self.root_round).map_err(|_| PalsError::InvalidLimits)?)
            .min(ranked.len());
        let frontier = self.root_frontier(&ranked, width)?;
        if self.accepted.contains(&ranking.source().ticket.id) {
            return Err(RoleError::InvalidOutput.into());
        }
        // The root ranking was linearly accepted by its physical owner before
        // this publication. Consume that original immutable view before any
        // candidate's new public record/view can observe the protection state.
        self.accepted.insert(ranking.source().ticket.id);
        self.committed_role_questions
            .insert(ranking.source().ticket.id, ranking.source().ticket.question);
        self.role_consumed(&ranking.source().ticket, ranking.view())?;
        // Consumption is already real even if a later hot allocation or view
        // materialization fails. Do not omit it from the failure receipt.
        self.counters.proposer_calls += 1;
        self.counters.completed_proposer_calls += 1;
        self.counters.accepted_proposer_outputs += 1;
        self.counters.role_calls += 1;
        self.counters.consumed_role_outputs += 1;
        // The root ranking itself spends one call. Do not issue zero-credit
        // candidate frames when it consumes the final root-wide admission.
        let mut remaining: VecDeque<_> =
            if self.counters.role_calls >= self.engine.config.max_role_calls {
                VecDeque::new()
            } else {
                frontier.into()
            };
        let issue_count = if self.root_fanout == AdaptiveRootFanoutPolicy::SequentialFixed {
            1.min(remaining.len())
        } else {
            remaining.len()
        };
        let mut out = Vec::with_capacity(issue_count);
        for _ in 0..issue_count {
            let mut candidate = self.make_root_candidate_with_frozen_scope(
                remaining.pop_front().ok_or(PalsError::InvalidConfig)?,
                Some(ranking.view().as_ref()),
            )?;
            if self.continuation_limit() == 1
                && self.engine.semantic.nodes[candidate.input.question.node]
                    .terminal
                    .is_none()
            {
                candidate.initialized = Some(CheckedRootCandidate {
                    source: ranking.source(),
                    first: candidate.first_move,
                    input: candidate.input.clone(),
                    budget: candidate.budget,
                    line_plies: self.continuation_limit(),
                });
            }
            out.push(candidate);
        }
        self.lazy_frontier = remaining;
        self.sequential_ranking_source =
            if self.root_fanout == AdaptiveRootFanoutPolicy::SequentialFixed {
                Some(ranking.source())
            } else {
                None
            };
        for candidate in &out {
            self.round_candidates.insert(
                candidate.input.question_digest,
                (
                    candidate
                        .input
                        .question
                        .first
                        .ok_or(RoleError::InvalidOutput)?,
                    None,
                ),
            );
        }
        self.root_ranking_pending = false;
        self.root_ranking_question = None;
        self.last_role_commit = Some((
            ranking.source().ticket.question,
            ranking.source().ticket.id,
            observations_before,
            self.engine
                .semantic
                .stores
                .observations
                .next_observation_sequence(),
        ));
        Ok(out)
    }

    fn make_root_candidate(
        &mut self,
        movement: BoardMove,
    ) -> Result<AdaptivePortCandidate, PalsError> {
        // A deferred candidate is selected only after the previous actual
        // candidate closes. Its separate port proof permits that new scope.
        self.make_root_candidate_with_frozen_scope(movement, None)
    }

    fn make_root_candidate_with_frozen_scope(
        &mut self,
        movement: BoardMove,
        frozen_scope: Option<&ReviewView>,
    ) -> Result<AdaptivePortCandidate, PalsError> {
        if !self.root_move_allowed(movement) {
            return Err(RoleError::InvalidOutput.into());
        }
        let prefix = vec![movement];
        let child = self.replay_node(&prefix)?;
        let stage = if self.continuation_limit() <= 1
            || self.engine.semantic.nodes[child].terminal.is_some()
        {
            RefinementStage::CpuValidation
        } else {
            RefinementStage::ProposalContinuation
        };
        if stage == RefinementStage::CpuValidation {
            self.record_hot(RecordKind::Proposal, &prefix, None)?;
            self.counters.proposals += 1;
        }
        let proposal = if stage == RefinementStage::CpuValidation {
            prefix.clone()
        } else {
            Vec::new()
        };
        let frozen = match frozen_scope {
            Some(source) => {
                // The initial ranking has already been consumed, which may
                // change bundle protection and selection order. Keep its exact
                // records/relations/anchors and fork only this checked query.
                // The additional query tokens must fit that original scope;
                // fork_query returns ViewCapacity rather than replacing rows.
                self.ensure_new_review_leases(true)?;
                let legal = self.engine.semantic.nodes[child].position.legal_moves();
                let view = self
                    .engine
                    .semantic
                    .review_index
                    .as_mut()
                    .ok_or(PalsError::InvalidConfig)?
                    .fork_query(
                        &self.engine.semantic.stores,
                        source,
                        crate::pals::review_context::ReviewQuerySequences {
                            purpose: purpose_for_stage(stage),
                            prefix: &prefix,
                            proposal: &proposal,
                            counterexample: None,
                            legal: &legal,
                            divergences: &[],
                        },
                    )?;
                Some(Arc::new(view))
            }
            None => None,
        };
        let input = self.input(
            child,
            stage,
            Some(movement),
            prefix,
            proposal,
            None,
            None,
            None,
            false,
            frozen,
            None,
        )?;
        Ok(AdaptivePortCandidate {
            first_move: rz_contracts::pals::Move16::encode(
                rz_contracts::Move::try_from(movement).map_err(|_| RoleError::InvalidOutput)?,
            ),
            input,
            budget: FrameBudget {
                original_cpu_credit: self.limits.max_cpu_nodes,
                // A frame ceiling never replenishes the root ranking or an
                // earlier candidate's consumed calls. Concurrent frames still
                // share the manager's single admission/reservation ceiling.
                role_submissions_remaining: self
                    .engine
                    .config
                    .max_role_calls
                    .saturating_sub(self.counters.role_calls),
                original_store_byte_limit: 8 * 1024 * 1024,
            },
            deferred: None,
            initialized: None,
        })
    }

    fn root_frontier(
        &self,
        ranked: &[BoardMove],
        width: usize,
    ) -> Result<Vec<BoardMove>, PalsError> {
        let mut frontier: Vec<_> = ranked.iter().take(width).copied().collect();
        frontier.sort_by(|a, b| {
            let rank = || {
                ranked
                    .iter()
                    .position(|movement| movement == a)
                    .cmp(&ranked.iter().position(|movement| movement == b))
            };
            if self.engine.uses_model_resolver() {
                let value = |movement| {
                    self.engine.semantic.nodes[self.root]
                        .edges
                        .iter()
                        .find(|edge| edge.movement == movement)
                        .and_then(|edge| {
                            model_order_value(self.engine.root_model_value(edge.child).0)
                        })
                };
                value(*b)
                    .partial_cmp(&value(*a))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(rank)
            } else {
                let value = |movement| {
                    self.engine.semantic.nodes[self.root]
                        .edges
                        .iter()
                        .find(|edge| edge.movement == movement)
                        .and_then(|edge| {
                            self.engine
                                .value(edge.child, self.engine.config.line_plies.saturating_sub(1))
                        })
                        .map(|score| -score)
                };
                value(*b).cmp(&value(*a)).then_with(rank)
            }
        });
        let new_root = ranked.iter().take(width).copied().find(|movement| {
            !self.engine.semantic.nodes[self.root]
                .edges
                .iter()
                .any(|edge| edge.movement == *movement)
        });
        frontier.truncate(self.engine.config.beam_width);
        if let Some(movement) = new_root
            && !frontier.contains(&movement)
        {
            if frontier.len() == self.engine.config.beam_width {
                frontier.pop();
            }
            frontier.push(movement);
        }
        Ok(frontier)
    }

    /// Permit another frame's publication. The submitted context is never
    /// recomputed against latest public_revision or latest situation revision.
    pub fn validate_role(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), PalsError> {
        self.check_work(work)?;
        self.verify_registered_model_namespace(ticket, output)?;
        let query = &work.input.question;
        let view = &work.input.view;
        if (self.accepted.contains(&ticket.id)
            && !self.reciprocal_source_consumed(work, ticket, output))
            || ticket.authority != self.authority
            || ticket.question != work.input.question_digest
            || ticket.original_deadline != work.original_deadline
            || ticket.view != work.input.view_seal
            || output.id != ticket.id
            || output.authority != ticket.authority
            || output.key != ticket.representation
            || ticket.role != role_for_purpose(query.context.purpose)
            || view.query().purpose != query.context.purpose
            || view.query().prefix != query.context.prefix
            || query.context.public_revision > self.engine.semantic.revision
            || query.context.game_generation != self.authority.game.0
            || query.context.search_generation != self.authority.root.0
            || query.context.proposal_sha256
                != role_line_sha256(b"proposal", &view.query().proposal)?
            || query.context.refutation_sha256
                != view
                    .query()
                    .counterexample
                    .as_deref()
                    .map(|line| role_line_sha256(b"refutation", line))
                    .transpose()?
            || query.context.divergence_sha256 != role_divergence_sha256(&view.query().divergences)?
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let node = &self.engine.semantic.nodes[query.node];
        let situation = self
            .engine
            .semantic
            .stores
            .situations
            .get(query.context.situation)?;
        if node.state != query.context.state
            || node.situation != query.context.situation
            || situation.state != query.context.state
            || situation.focus != query.context.focus
            || self
                .engine
                .semantic
                .stores
                .lines
                .get(query.context.focus)?
                .start_state
                != query.context.state
            || query.context.focus_sha256
                != role_line_sha256(
                    b"focus",
                    &self
                        .engine
                        .semantic
                        .stores
                        .lines
                        .moves(query.context.focus)?,
                )?
            || query.context.prefix_sha256 != role_line_sha256(b"prefix", &query.context.prefix)?
            || !node
                .position
                .snapshot()
                .same_state(&query.position.snapshot())
        {
            return Err(RoleError::InvalidOutput.into());
        }
        validate_payload(query.context.purpose, view, output)?;
        if query.stage == RefinementStage::FreshCounterEndpoint {
            let original = query.paired_fresh_seal.ok_or(RoleError::InvalidOutput)?;
            let current = FreshPairSeal::from_ticket(
                ticket,
                work.input.public_context,
                self.engine.semantic.nodes[self.root]
                    .position
                    .side_to_move()
                    .into(),
            );
            original
                .validate_pair(&current)
                .map_err(|_| RoleError::InvalidOutput)?;
            if query.paired_repaired.is_none() {
                return Err(RoleError::InvalidOutput.into());
            }
        }
        Ok(())
    }

    fn verify_registered_model_namespace(
        &self,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), PalsError> {
        match self
            .engine
            .semantic
            .model
            .registered_representation_namespace()
        {
            Some(namespace) => {
                namespace.verify_identity(
                    self.engine.semantic.model.identity(),
                    self.engine.semantic.model.value_identity(),
                )?;
                namespace.verify(ticket, output, self.engine.semantic.model.value_identity())?;
                if self
                    .engine
                    .semantic
                    .model_registered_value
                    .as_ref()
                    .is_some_and(|value| value != namespace.value_identity())
                {
                    return Err(RoleError::InvalidOutput.into());
                }
                Ok(())
            }
            None if self
                .engine
                .semantic
                .model
                .requires_registered_representation_namespace() =>
            {
                Err(
                    RoleError::Unsupported("native representation namespace is not registered")
                        .into(),
                )
            }
            None => Ok(()),
        }
    }

    fn verify_registered_ticket_namespace(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), PalsError> {
        match self
            .engine
            .semantic
            .model
            .registered_representation_namespace()
        {
            Some(namespace) => {
                namespace.verify_identity(
                    self.engine.semantic.model.identity(),
                    self.engine.semantic.model.value_identity(),
                )?;
                namespace.verify_ticket(ticket, self.engine.semantic.model.value_identity())?;
                if self
                    .engine
                    .semantic
                    .model_registered_value
                    .as_ref()
                    .is_some_and(|value| value != namespace.value_identity())
                {
                    return Err(RoleError::InvalidOutput.into());
                }
                Ok(())
            }
            None if self
                .engine
                .semantic
                .model
                .requires_registered_representation_namespace() =>
            {
                Err(
                    RoleError::Unsupported("native representation namespace is not registered")
                        .into(),
                )
            }
            None => Ok(()),
        }
    }

    pub fn role_consumed(
        &mut self,
        ticket: &RoleTicketEnvelope,
        view: &Arc<ReviewView>,
    ) -> Result<(), PalsError> {
        self.role_consumption_observation_started = true;
        let returned = self.role_consumed_unobserved(ticket, view);
        if let Err(error) = &returned {
            self.observe_role_view(
                ticket,
                view,
                AdaptiveRoleConsumptionOutcome::Rejected(
                    AdaptiveRoleConsumptionRejection::from_error(error),
                ),
            );
        }
        returned
    }

    fn role_consumed_unobserved(
        &mut self,
        ticket: &RoleTicketEnvelope,
        view: &Arc<ReviewView>,
    ) -> Result<(), PalsError> {
        if ticket.authority != self.authority
            || ticket.view.view_digest.0 != view.seal_digest()
            || ticket.view != self.seal(view)
            || !view.validates(&self.engine.semantic.stores)
            || !self.accepted.contains(&ticket.id)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if let Some(original) = self.consumed_role_views.get(&ticket.id) {
            if *original == ticket.view.view_digest {
                self.observe_role_view(
                    ticket,
                    view,
                    AdaptiveRoleConsumptionOutcome::DuplicateNoNewConsumption,
                );
                return Ok(());
            }
            return Err(RoleError::InvalidOutput.into());
        }
        self.engine
            .semantic
            .review_index
            .as_mut()
            .ok_or(PalsError::InvalidConfig)?
            .consume(view, &self.engine.semantic.stores)?;
        self.consumed_role_views
            .insert(ticket.id, ticket.view.view_digest);
        // This is the original first successful consume+insert, not a native
        // acceptance or a count inferred from a later completion callback.
        self.observe_role_view(ticket, view, AdaptiveRoleConsumptionOutcome::ConsumedView);
        Ok(())
    }

    fn observe_role_view(
        &mut self,
        ticket: &RoleTicketEnvelope,
        view: &ReviewView,
        outcome: AdaptiveRoleConsumptionOutcome,
    ) {
        if let Some(journal) = &mut self.role_consumption_observation {
            journal.record(AdaptiveRoleConsumptionEvent {
                authority: ticket.authority,
                request: ticket.id,
                role: ticket.role,
                question: ticket.question,
                submitted_question: self.committed_role_questions.get(&ticket.id).copied(),
                ticket_view: ticket.view.view_digest,
                submitted_view: ContractDigest(view.seal_digest()),
                frame: None,
                stage: AdaptiveRoleConsumptionStage::ViewConsumption,
                outcome,
                completed_evidence: None,
                prior_semantic_attempt: None,
                published_repair_line: None,
                repair_line_input: None,
            });
        }
    }

    /// Observe an already recorded semantic consumption after a later
    /// continuation/materialization failure. This does not consume again,
    /// mutate protection, or certify that the next search stage succeeded.
    pub fn role_consumption_recorded(
        &self,
        ticket: &RoleTicketEnvelope,
        view: &ReviewView,
    ) -> Result<bool, PalsError> {
        if ticket.authority != self.authority
            || ticket.view != self.seal(view)
            || self
                .committed_role_questions
                .get(&ticket.id)
                .is_some_and(|question| *question != ticket.question)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        match self.consumed_role_views.get(&ticket.id) {
            Some(original)
                if *original == ticket.view.view_digest
                    && self.accepted.contains(&ticket.id)
                    && self.committed_role_questions.get(&ticket.id) == Some(&ticket.question) =>
            {
                Ok(true)
            }
            Some(_) => Err(RoleError::InvalidOutput.into()),
            None => Ok(self.reciprocal_head_use_recorded(ticket, view)
                || self.rovent_goal_consumption_recorded(ticket, view)),
        }
    }

    fn check_live(&self) -> Result<(), PalsError> {
        if Instant::now() >= self.limits.deadline {
            return Err(RoleError::Deadline.into());
        }
        if self.engine.semantic.game_generation != Some(self.authority.game.0)
            || self.engine.semantic.stores.generation() != self.authority.root.0
            || self.engine.semantic.stores.root()
                != Some(self.engine.semantic.nodes[self.root].situation)
            || !self.engine.semantic.nodes[self.root]
                .position
                .snapshot()
                .same_state(self.rules.rules().snapshot())
        {
            return Err(StoreError::StaleConsumer.into());
        }
        Ok(())
    }
    fn check_work(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), PalsError> {
        self.check_work_input(work.frame, work.original_deadline, &work.input)?;
        if let Some(first) = work.input.question.first
            && rz_contracts::pals::Move16::encode(
                rz_contracts::Move::try_from(first).map_err(|_| RoleError::InvalidOutput)?,
            ) != work.original_first_move
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }

    fn check_work_input(
        &self,
        frame: crate::pals::coordinator::FrameId,
        original_deadline: Deadline,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), PalsError> {
        self.check_live()?;
        let query = &input.question;
        if query.goal_local.is_some()
            || frame.authority != self.authority
            || self.closed.contains(&frame)
            || original_deadline != self.deadline
            || input.stage != query.stage
            || input.question_digest != query.question_digest
            || input.view_seal != self.seal(&input.view)
            || input.public_context.0 != input.view.record_scope_digest()
            || !input.view.validates(&self.engine.semantic.stores)
            || query.node >= self.engine.semantic.nodes.len()
            || query.first.is_some_and(|first| {
                Some(first) != query.context.prefix.first().copied()
                    && !query.context.prefix.is_empty()
            })
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let exact = if self.rovent_frontier().is_some() {
            self.existing_checked_frontier_line(&query.context.prefix)?
                .ok_or(crate::pals::rovent_frontier::FrontierError::Invalid(
                    "question prefix has no original charged Rules path",
                ))?
                .endpoint_snapshot()
                .clone()
        } else {
            let mut replay = self.engine.semantic.nodes[self.root].position.clone();
            for movement in &query.context.prefix {
                replay.make_move(*movement)?;
            }
            replay.snapshot()
        };
        if !exact.same_state(&query.position.snapshot()) {
            return Err(ReviewError::AnchorMismatch.into());
        }
        Ok(())
    }

    fn seal(&self, view: &ReviewView) -> ReviewViewSeal {
        ReviewViewSeal {
            authority: self.authority,
            view_sequence: view.sequence(),
            context_policy: ContractDigest(view.policy_digest()),
            view_digest: ContractDigest(view.seal_digest()),
            root_state: self.rules.snapshot().identity(),
            actual_tokens: view.token_count(),
            selected_records: view.records().len() as u16,
            anchors: view.anchors().len() as u8,
            length_bucket: view.length_bucket(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn input(
        &mut self,
        node: usize,
        stage: RefinementStage,
        first: Option<BoardMove>,
        prefix: Vec<BoardMove>,
        proposal: Vec<BoardMove>,
        counter: Option<Vec<BoardMove>>,
        repaired: Option<Vec<BoardMove>>,
        attack_ply: Option<usize>,
        recheck: bool,
        frozen: Option<Arc<ReviewView>>,
        paired_repaired: Option<ObservationId>,
    ) -> Result<FrameInput<ReviewView, AdaptivePortQuestion>, PalsError> {
        self.input_with_purpose(
            node,
            stage,
            first,
            prefix,
            proposal,
            counter,
            repaired,
            attack_ply,
            recheck,
            frozen,
            paired_repaired,
            purpose_for_stage(stage),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn input_with_purpose(
        &mut self,
        node: usize,
        stage: RefinementStage,
        first: Option<BoardMove>,
        prefix: Vec<BoardMove>,
        proposal: Vec<BoardMove>,
        counter: Option<Vec<BoardMove>>,
        repaired: Option<Vec<BoardMove>>,
        attack_ply: Option<usize>,
        recheck: bool,
        frozen: Option<Arc<ReviewView>>,
        paired_repaired: Option<ObservationId>,
        purpose: RoleQueryPurpose,
    ) -> Result<FrameInput<ReviewView, AdaptivePortQuestion>, PalsError> {
        self.input_with_source(
            node,
            stage,
            first,
            prefix,
            proposal,
            counter,
            repaired,
            attack_ply,
            recheck,
            frozen,
            paired_repaired,
            purpose,
            None,
        )
    }

    fn input_goal_local(
        &mut self,
        node: usize,
        stage: RefinementStage,
        first: BoardMove,
        prefix: Vec<BoardMove>,
        origin: Arc<goal_local::GoalLocalOrigin>,
    ) -> Result<FrameInput<ReviewView, AdaptivePortQuestion>, PalsError> {
        origin.validate_prepared_node(self, node, first, &prefix)?;
        if stage != origin.stage() {
            return Err(RoleError::InvalidOutput.into());
        }
        let purpose = origin.purpose();
        let input = self.input_with_source(
            node,
            stage,
            Some(first),
            prefix,
            Vec::new(),
            None,
            None,
            None,
            false,
            None,
            None,
            purpose,
            Some(Arc::clone(&origin)),
        )?;
        origin.validates_frame_input(&input)?;
        Ok(input)
    }

    #[allow(clippy::too_many_arguments)]
    fn input_with_source(
        &mut self,
        node: usize,
        stage: RefinementStage,
        first: Option<BoardMove>,
        prefix: Vec<BoardMove>,
        proposal: Vec<BoardMove>,
        counter: Option<Vec<BoardMove>>,
        repaired: Option<Vec<BoardMove>>,
        attack_ply: Option<usize>,
        recheck: bool,
        frozen: Option<Arc<ReviewView>>,
        paired_repaired: Option<ObservationId>,
        purpose: RoleQueryPurpose,
        goal_local: Option<Arc<goal_local::GoalLocalOrigin>>,
    ) -> Result<FrameInput<ReviewView, AdaptivePortQuestion>, PalsError> {
        self.check_live()?;
        let divergences = if stage == RefinementStage::CriticDivergence {
            self.divergence_indices(&proposal)?
        } else {
            Vec::new()
        };
        // The branch continuation is owner-private state, not an extra model
        // refutation. Keep the synchronous Reply/ValueFresh input semantics.
        let model_proposal = if goal_local.is_some() {
            &[][..]
        } else if purpose == RoleQueryPurpose::ValueFresh {
            prefix.as_slice()
        } else {
            proposal.as_slice()
        };
        let model_counter = if purpose == RoleQueryPurpose::ValueFresh
            || matches!(
                stage,
                RefinementStage::CriticReply | RefinementStage::CounterContinuation
            ) {
            None
        } else {
            counter.as_deref()
        };
        let question = RoleQuestion {
            purpose,
            prefix: &prefix,
            proposal: model_proposal,
            refutation: model_counter,
            divergences: &divergences,
        };
        let context = self.engine.role_context(node, question)?;
        let legal = if matches!(
            purpose,
            RoleQueryPurpose::DivergencePolicy | RoleQueryPurpose::TaskPolicy
        ) {
            Vec::new()
        } else {
            self.engine.semantic.nodes[node].position.legal_moves()
        };
        self.ensure_new_review_leases(frozen.is_none())?;
        let view = match frozen {
            Some(view) => {
                if view.query().purpose != purpose
                    || view.query().prefix != prefix
                    || view.query().proposal != model_proposal
                    || view.query().counterexample.as_deref() != model_counter
                    || view.query().legal != legal
                    || !view.validates(&self.engine.semantic.stores)
                {
                    return Err(RoleError::InvalidOutput.into());
                }
                view
            }
            None => self
                .engine
                .prepare_review_view(node, question, &legal)?
                .ok_or(PalsError::InvalidConfig)?,
        };
        let sequence = self.next_question;
        self.next_question = sequence.checked_add(1).ok_or(PalsError::Capacity)?;
        let legacy_full_records = self.prepare_legacy_full_records(&context, &view)?;
        let question_digest = AdaptiveQuestionDigestPreimage {
            sequence,
            stage_code: stage as u8,
            view_digest: ContractDigest(view.digest()),
            view_seal_digest: ContractDigest(view.seal_digest()),
            focus_prefix_sha256: ContractDigest(context.focus_prefix_sha256()),
            goal_local_sha256: goal_local.as_ref().map(|origin| origin.digest()),
            legacy_full_records_sha256: legacy_full_records
                .as_ref()
                .map(|records| ContractDigest(records.digest())),
        }
        .digest();
        let frontier_source = frontier_port::FrontierQuestionSource::new(
            self.authority,
            self.deadline,
            self.limits.deadline,
            self.engine.semantic.nodes[node].position.snapshot(),
            &context,
            self.seal(&view),
            legacy_full_records.as_ref(),
            goal_local.as_ref().map(|origin| origin.digest()),
        )?;
        let query = AdaptivePortQuestion {
            position: self.engine.semantic.nodes[node].position.clone(),
            context,
            deadline: self.limits.deadline,
            legacy_full_records,
            node,
            stage,
            first,
            proposal,
            counter,
            repaired,
            attack_ply,
            discovery_restricted: false,
            counter_source: None,
            validator_root_moves: None,
            recheck,
            frozen_counter: None,
            paired_repaired,
            paired_fresh_seal: None,
            question_sequence: sequence,
            question_digest,
            frontier_source,
            goal_local,
            diagnostic_repair_line: None,
        };
        Ok(FrameInput {
            stage,
            question_digest,
            question: Arc::new(query),
            view_seal: self.seal(&view),
            public_context: ContractDigest(view.record_scope_digest()),
            view,
        })
    }

    fn divergence_indices(&self, line: &[BoardMove]) -> Result<Vec<usize>, PalsError> {
        let checked = if self.rovent_frontier().is_some() {
            Some(self.existing_checked_frontier_line(line)?.ok_or(
                crate::pals::rovent_frontier::FrontierError::Invalid(
                    "divergence line has no original charged Rules path",
                ),
            )?)
        } else {
            None
        };
        let mut replay = self.engine.semantic.nodes[self.root].position.clone();
        let color = replay.side_to_move();
        let mut indices = Vec::new();
        for (ply, movement) in line.iter().enumerate() {
            if replay.side_to_move() != color
                && matches!(replay.classify_position()?.play_status, PlayStatus::Ongoing)
            {
                indices.push(ply);
            }
            if let Some(checked) = &checked {
                replay =
                    Position::from_snapshot(checked.steps()[ply].snapshot(), Default::default())?;
            } else {
                replay.make_move(*movement)?;
            }
        }
        Ok(indices)
    }
    fn intern_hot(engine: &mut PalsEngine<M>, position: Position) -> Result<usize, PalsError> {
        let snapshot = position.snapshot();
        if let Some(state) = engine.semantic.stores.states.find(&snapshot)
            && let Some(index) = engine
                .semantic
                .nodes
                .iter()
                .position(|node| node.state == state)
        {
            return Ok(index);
        }
        if engine.semantic.nodes.len() >= engine.config.max_nodes {
            return Err(PalsError::Capacity);
        }
        let situation = if let Some(root) = engine.semantic.stores.root().filter(|root| {
            engine
                .semantic
                .stores
                .situations
                .get(*root)
                .is_ok_and(|situation| {
                    engine
                        .semantic
                        .stores
                        .states
                        .get(situation.state)
                        .is_ok_and(|state| state.same_state(&snapshot))
                })
        }) {
            root
        } else {
            engine.semantic.stores.hot_insert_situation(snapshot)?
        };
        let state = engine.semantic.stores.situations.get(situation)?.state;
        let terminal = match position.classify_position()?.play_status {
            PlayStatus::Ongoing => None,
            PlayStatus::Terminal { reason, winner } => Some((
                reason,
                match winner {
                    Some(color) if color == position.side_to_move() => CPU_MATE_SCORE,
                    Some(_) => -CPU_MATE_SCORE,
                    None => 0,
                },
            )),
        };
        if terminal.is_some() {
            engine
                .semantic
                .stores
                .hot_append_rules_terminal(&position, 0, 0)?;
        }
        engine.semantic.nodes.push(Node {
            position,
            state,
            situation,
            terminal,
            edges: Vec::new(),
            evidence: None,
            model_value: None,
            model_observation: None,
            model_value_revision: None,
            resume: None,
        });
        Ok(engine.semantic.nodes.len() - 1)
    }
    fn connect_hot(&mut self, parent: usize, movement: BoardMove) -> Result<usize, PalsError> {
        if self.rovent_frontier().is_some() {
            return Err(crate::pals::rovent_frontier::FrontierError::Invalid(
                "uncharged hot connection under original frontier",
            )
            .into());
        }
        if let Some(edge) = self.engine.semantic.nodes[parent]
            .edges
            .iter()
            .find(|edge| edge.movement == movement)
        {
            return Ok(edge.child);
        }
        let mut position = self.engine.semantic.nodes[parent].position.clone();
        position.make_move(movement)?;
        let child = Self::intern_hot(&mut self.engine, position)?;
        self.engine.semantic.nodes[parent]
            .edges
            .push(Edge { movement, child });
        self.counters.examined_edges += 1;
        Ok(child)
    }
    fn replay_node(&mut self, prefix: &[BoardMove]) -> Result<usize, PalsError> {
        if self.rovent_frontier().is_some() {
            return self.replay_checked_frontier(prefix);
        }
        let mut node = self.root;
        for movement in prefix {
            node = self.connect_hot(node, *movement)?;
        }
        Ok(node)
    }
}

fn role_for_purpose(purpose: RoleQueryPurpose) -> Role {
    match purpose {
        RoleQueryPurpose::ReplyPolicy | RoleQueryPurpose::DivergencePolicy => Role::Critic,
        RoleQueryPurpose::TaskPolicy => Role::Validator,
        _ => Role::Proposer,
    }
}
fn purpose_for_stage(stage: RefinementStage) -> RoleQueryPurpose {
    match stage {
        RefinementStage::CriticDivergence => RoleQueryPurpose::DivergencePolicy,
        RefinementStage::CriticReply
        | RefinementStage::CounterContinuation
        | RefinementStage::CriticRecheck => RoleQueryPurpose::ReplyPolicy,
        RefinementStage::Repair => RoleQueryPurpose::RepairPolicy,
        RefinementStage::FreshRepairedEndpoint
        | RefinementStage::FreshCounterEndpoint
        | RefinementStage::ProposalValueFresh
        | RefinementStage::CounterValueFresh
        | RefinementStage::RepairValueFresh
        | RefinementStage::RecheckRepairedValueFresh => RoleQueryPurpose::ValueFresh,
        _ => RoleQueryPurpose::ProposePolicy,
    }
}
pub(super) fn validate_payload(
    purpose: RoleQueryPurpose,
    view: &ReviewView,
    output: &RoleOutput,
) -> Result<(), PalsError> {
    if output.private_latent.len() != 16 * 384
        || output.private_latent.iter().any(|value| !value.is_finite())
    {
        return Err(RoleError::InvalidOutput.into());
    }
    let (candidates, divergences, wdl) = match (&output.payload, role_for_purpose(purpose)) {
        (
            RolePayload::Proposal {
                candidate_logits,
                wdl,
            },
            Role::Proposer,
        ) => (candidate_logits.as_slice(), &[][..], Some(*wdl)),
        (
            RolePayload::Counterexample {
                candidate_logits,
                divergence_logits,
                wdl,
            },
            Role::Critic,
        ) => (
            candidate_logits.as_slice(),
            divergence_logits.as_slice(),
            Some(*wdl),
        ),
        (RolePayload::TaskRanking { task_logits }, Role::Validator)
            if task_logits.len() == 7 && task_logits.iter().all(|x| x.is_finite()) =>
        {
            return Ok(());
        }
        _ => return Err(RoleError::InvalidOutput.into()),
    };
    if candidates.len() != view.query().legal.len()
        || candidates.iter().any(|x| !x.is_finite())
        || divergences.len() != view.query().divergences.len()
        || divergences.iter().any(|x| !x.is_finite())
    {
        return Err(RoleError::InvalidOutput.into());
    }
    if let Some(wdl) = wdl {
        let [w, d, l] = wdl.probabilities();
        rz_contracts::Wdl::try_new(w, d, l, 1e-4).map_err(|_| RoleError::InvalidOutput)?;
    }
    Ok(())
}
