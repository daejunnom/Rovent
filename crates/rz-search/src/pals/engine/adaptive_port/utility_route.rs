//! Actual route-utility actions are separate from V task actions and from the
//! seven numerical shadow hints. These owners never turn a CP, mate, depth or
//! model value into utility. Only new conditional Store witnesses are counted.
use super::*;
use crate::cpu_value::nnuec::v2::Model;
use crate::cpu_value::nnuec::v2::review::{Request, RequestIdentity};
use crate::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture;
use crate::pals::coordinator::{AcceptedRoleOrigin, AcceptedRoleStep, FrameBudget, StepExecution};
use crate::pals::store::StoreReadOwner;
use rz_contracts::rovent::{QuestionEnvelope, SearchRecipient};
use std::collections::{BTreeSet, HashSet};
use std::sync::Weak;
use validator_port::{ActionWitnessKind, MAX_ACTION_WITNESSES};

mod conditional_witness;
mod intervention;
pub use super::utility_conditional_basis::{
    UtilityConditionalWitnessBasis, UtilityConditionalWitnessBasisUnavailable,
    UtilityWitnessBasisError, UtilityWitnessBasisExpectation, UtilityWitnessBasisMetadata,
    UtilityWitnessCpuMetadata, UtilityWitnessEndpointMetadata, UtilityWitnessPcMetadata,
    validate_utility_witness_basis,
};
pub(super) use conditional_witness::MAX_UTILITY_WITNESS_BODY_BYTES;
pub(in crate::pals::engine::adaptive_port) use conditional_witness::{
    HistoricalWitnessObservation, HistoricalWitnessRawScore,
    parse_validated_utility_witness_preimage,
};
pub use conditional_witness::{UtilityConditionalWitnessBodies, UtilityConditionalWitnessBody};
pub use conditional_witness::{
    UtilityWitnessPreimageError, UtilityWitnessPreimageExpectation,
    validate_utility_witness_preimage,
};
pub use intervention::{
    UTILITY_INTERVENTION_TARGET_DOMAIN, UtilityInterventionAdmission,
    UtilityInterventionBindRejection, UtilityInterventionCandidate,
    UtilityInterventionHistoricalTargetContent, UtilityInterventionTarget,
    UtilityInterventionTargetBindRejection, UtilityInterventionTargetDigestPreimage,
    UtilityInterventionUnavailable, utility_intervention_arm_question_preimage,
    utility_intervention_historical_snapshot_digest, utility_intervention_historical_target_digest,
    utility_intervention_target_digest, utility_intervention_target_digest_preimage,
};

/// Array indices belong to NNUEC utility, not the legacy seven V tasks.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum UtilityRouteRecipient {
    Proposer,
    Critic,
    Researcher,
}
impl UtilityRouteRecipient {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Proposer => "route_p",
            Self::Critic => "route_c",
            Self::Researcher => "route_r",
        }
    }
    pub fn output_index(self) -> usize {
        match self {
            Self::Proposer => 4,
            Self::Critic => 5,
            Self::Researcher => 6,
        }
    }
    pub fn search_recipient(self) -> SearchRecipient {
        match self {
            Self::Proposer => SearchRecipient::Proposer,
            Self::Critic => SearchRecipient::Critic,
            Self::Researcher => SearchRecipient::Researcher,
        }
    }
}

/// Cloning this name preserves identity; it cannot create another action.
#[derive(Clone, Debug)]
pub struct UtilityRouteActionId {
    owner: Arc<()>,
    authority: SearchAuthority,
    frame: FrameId,
    question: ContractDigest,
    recipient: UtilityRouteRecipient,
}
impl PartialEq for UtilityRouteActionId {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
            && self.authority == other.authority
            && self.frame == other.frame
            && self.question == other.question
            && self.recipient == other.recipient
    }
}
impl Eq for UtilityRouteActionId {}
impl UtilityRouteActionId {
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn frame(&self) -> FrameId {
        self.frame
    }
    pub fn question(&self) -> ContractDigest {
        self.question
    }
    pub fn recipient(&self) -> UtilityRouteRecipient {
        self.recipient
    }
}

/// The manager owns this before submitting the original preparation Request.
/// A failure keeps it as unknown action custody; no cost is refunded here.
pub struct UtilityRoutePreparation {
    id: UtilityRouteActionId,
    store: StoreReadOwner,
    generation: u64,
    root: StateId,
    root_budget: ContractDigest,
    mode: rz_contracts::rovent::ExecutionMode,
    deadline: Deadline,
    hard_deadline: Instant,
    first: BoardMove,
    budget: FrameBudget,
    target_record: u16,
    target_records: Vec<u64>,
    source_view: Weak<ReviewView>,
    source_query: Weak<AdaptivePortQuestion>,
    view_seal: ReviewViewSeal,
    full_view: ContractDigest,
    first_observation: usize,
    baseline: HashSet<ContractDigest>,
    search_scope: ContractDigest,
    started: Instant,
    request: Option<RequestIdentity>,
}
impl UtilityRoutePreparation {
    pub fn id(&self) -> &UtilityRouteActionId {
        &self.id
    }
    pub fn frame(&self) -> FrameId {
        self.id.frame
    }
    pub fn recipient(&self) -> UtilityRouteRecipient {
        self.id.recipient
    }
    pub fn target_record(&self) -> u16 {
        self.target_record
    }
    pub fn target_record_ids(&self) -> &[u64] {
        &self.target_records
    }
    pub fn original_deadline(&self) -> Deadline {
        self.deadline
    }
    pub fn hard_deadline(&self) -> Instant {
        self.hard_deadline
    }
    pub fn root_budget(&self) -> ContractDigest {
        self.root_budget
    }
    pub fn started(&self) -> Instant {
        self.started
    }
    pub fn budget(&self) -> FrameBudget {
        self.budget
    }
    /// Attach only to the exact original global-view Request. Retaining error
    /// returns that same Request; metadata equality is not a new admission.
    // Rejection returns the same move-only Request/preparation/action owner;
    // boxing here would require a new allocation while preserving old pins.
    #[allow(clippy::result_large_err)]
    pub fn attach_input_capture(
        &mut self,
        request: Request,
    ) -> Result<Request, (AdaptiveNnuecPreparationError, Request)> {
        let p = request.provenance();
        let same_view = self
            .source_view
            .upgrade()
            .is_some_and(|view| request.matches_view(&view));
        if self.request.is_some()
            || !same_view
            || p.root != self.root
            || p.generation != self.generation
            || p.view_seal != self.view_seal.view_digest.0
            || p.full_view_digest != self.full_view.0
        {
            return Err((PalsError::from(RoleError::InvalidOutput).into(), request));
        }
        let identity = request.identity();
        match request.with_utility_input_capture(self.target_record) {
            Ok(request) => {
                self.request = Some(identity);
                Ok(request)
            }
            Err((error, request)) => {
                Err((AdaptiveNnuecPreparationError::Preparation(error), request))
            }
        }
    }
}

pub struct UtilityRouteAction {
    preparation: UtilityRoutePreparation,
    input: Arc<NnuecUtilityInputCapture>,
    frozen_context: ContractDigest,
    committed: HashSet<StepExecution>,
    committed_sources: HashMap<StepExecution, (FrameId, ContractDigest)>,
    witness_watermarks: HashMap<StepExecution, usize>,
    authorized: Vec<UtilityRouteDependencyWitness>,
    r_parents: Vec<(FrameId, QuestionEnvelope)>,
    pc_consumed: BTreeSet<rz_contracts::RequestId>,
    pc_origins: Vec<UtilityRoutePcConsumption>,
    native_executions: BTreeSet<rz_contracts::ExecutionId>,
    witnesses: Vec<UtilityRouteWitness>,
    witnessed: HashSet<ContractDigest>,
    r_pc_consumption: Vec<CpuSemanticWitness>,
    failed: bool,
    intervention: Option<intervention::UtilityInterventionSource>,
}
/// An original accepted P/C head that this exact source owner consumed.
/// It retains no ReviewView, output tensor or private model latent state.
#[derive(Clone)]
pub struct UtilityRoutePcConsumption {
    origin: AcceptedRoleOrigin,
    question: ContractDigest,
    view: ReviewViewSeal,
    full_view: ContractDigest,
}
impl UtilityRoutePcConsumption {
    pub fn origin(&self) -> &AcceptedRoleOrigin {
        &self.origin
    }
    pub fn question(&self) -> ContractDigest {
        self.question
    }
    pub fn view(&self) -> ReviewViewSeal {
        self.view
    }
    pub fn full_view(&self) -> ContractDigest {
        self.full_view
    }
}
/// No public constructor or deserialization path supplies a utility witness.
pub struct UtilityRouteWitness {
    source_question: ContractDigest,
    source_frame: FrameId,
    kind: ActionWitnessKind,
    observation: ObservationId,
    identity: ContractDigest,
    raw_evidence: ContractDigest,
    line: Vec<BoardMove>,
}
impl UtilityRouteWitness {
    pub fn source_question(&self) -> ContractDigest {
        self.source_question
    }
    pub fn source_frame(&self) -> FrameId {
        self.source_frame
    }

    pub fn kind(&self) -> ActionWitnessKind {
        self.kind
    }
    pub fn observation(&self) -> ObservationId {
        self.observation
    }
    pub fn identity(&self) -> ContractDigest {
        self.identity
    }
    pub fn raw_evidence(&self) -> ContractDigest {
        self.raw_evidence
    }
    pub fn line(&self) -> &[BoardMove] {
        &self.line
    }
}
/// Immutable actual dependency, retained without pinning its old input.
/// Public getters never provide a constructor or new admission authority.
pub struct UtilityRouteDependencyWitness {
    frame: FrameId,
    question: ContractDigest,
    view: ReviewViewSeal,
    full_view: ContractDigest,
    source_query: Weak<AdaptivePortQuestion>,
    source_view: Weak<ReviewView>,
    peer: Option<(QuestionEnvelope, Option<AcceptedRoleOrigin>)>,
    canonical_peer: Option<QuestionEnvelope>,
    coordinator_parent: Option<FrameId>,
    continuation: Option<super::utility_dependencies::UtilityContinuationSource>,
}
impl UtilityRouteDependencyWitness {
    pub fn frame(&self) -> FrameId {
        self.frame
    }
    pub fn question(&self) -> ContractDigest {
        self.question
    }
    pub fn view(&self) -> ReviewViewSeal {
        self.view
    }
    pub fn full_view(&self) -> ContractDigest {
        self.full_view
    }
    pub fn peer(&self) -> Option<(QuestionEnvelope, Option<AcceptedRoleOrigin>)> {
        self.peer
    }
    pub fn canonical_peer(&self) -> Option<QuestionEnvelope> {
        self.canonical_peer
    }
    pub fn coordinator_parent(&self) -> Option<FrameId> {
        self.coordinator_parent
    }
    pub fn continuation(&self) -> Option<&super::utility_dependencies::UtilityContinuationSource> {
        self.continuation.as_ref()
    }
    fn matches(&self, work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>) -> bool {
        self.frame == work.frame
            && self.question == work.input.question_digest
            && self
                .source_query
                .upgrade()
                .is_some_and(|q| Arc::ptr_eq(&q, &work.input.question))
            && self
                .source_view
                .upgrade()
                .is_some_and(|v| Arc::ptr_eq(&v, &work.input.view))
    }
}
pub struct UtilityRouteCommitGuard {
    action: UtilityRouteActionId,
    frame: FrameId,
    question: ContractDigest,
    execution: StepExecution,
    first_observation: usize,
    generation: u64,
    r_parent: Option<QuestionEnvelope>,
}
/// Search semantic closure is not a manager physical fence or whole cost.
pub struct UtilityRouteSemanticOutcome {
    action: UtilityRouteAction,
}
/// Partial/failing actions retain their actual input and all already used
/// evidence. This type grants no semantic completion or known target.
pub struct UtilityRouteUnresolvedAction {
    action: UtilityRouteAction,
    reason: UtilityRouteUnresolvedReason,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UtilityRouteUnresolvedReason {
    Cancelled,
    Deadline,
    ReplacedRoot,
    SemanticFailure,
    PhysicalUnknown,
    MissingConsumption,
    MissingWholeCost,
}
impl UtilityRouteUnresolvedReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::Deadline => "deadline",
            Self::ReplacedRoot => "replaced_root",
            Self::SemanticFailure => "semantic_failure",
            Self::PhysicalUnknown => "physical_unknown",
            Self::MissingConsumption => "missing_actual_consumption",
            Self::MissingWholeCost => "missing_exclusive_whole_cost",
        }
    }
}
impl UtilityRouteUnresolvedAction {
    pub fn id(&self) -> &UtilityRouteActionId {
        self.action.id()
    }
    pub fn input(&self) -> &Arc<NnuecUtilityInputCapture> {
        &self.action.input
    }
    pub fn root_budget(&self) -> ContractDigest {
        self.action.preparation.root_budget
    }
    pub fn mode(&self) -> rz_contracts::rovent::ExecutionMode {
        self.action.preparation.mode
    }

    pub fn frozen_context_digest(&self) -> ContractDigest {
        self.action.frozen_context
    }
    pub fn witnesses(&self) -> &[UtilityRouteWitness] {
        &self.action.witnesses
    }
    pub fn pc_consumption(&self) -> &[UtilityRoutePcConsumption] {
        &self.action.pc_origins
    }
    pub fn dependencies(&self) -> &[UtilityRouteDependencyWitness] {
        &self.action.authorized
    }

    pub fn r_pc_consumption(&self) -> &[CpuSemanticWitness] {
        &self.action.r_pc_consumption
    }
    pub fn reason(&self) -> UtilityRouteUnresolvedReason {
        self.reason
    }
    pub fn target_known_mask(&self) -> [bool; 7] {
        [false; 7]
    }
}
impl UtilityRouteSemanticOutcome {
    pub fn into_unresolved(
        self,
        reason: UtilityRouteUnresolvedReason,
    ) -> UtilityRouteUnresolvedAction {
        self.action.into_unresolved(reason)
    }
    pub fn id(&self) -> &UtilityRouteActionId {
        &self.action.preparation.id
    }
    pub fn input(&self) -> &Arc<NnuecUtilityInputCapture> {
        &self.action.input
    }
    pub fn root_budget(&self) -> ContractDigest {
        self.action.preparation.root_budget
    }
    pub fn mode(&self) -> rz_contracts::rovent::ExecutionMode {
        self.action.preparation.mode
    }

    /// Comparison content, excluding process-local request and owner numbers.
    /// Actual authority, target submission and execution seals remain separate.
    pub fn frozen_context_digest(&self) -> ContractDigest {
        self.action.frozen_context
    }
    pub fn baseline_witnesses(&self) -> impl Iterator<Item = &ContractDigest> {
        self.action.preparation.baseline.iter()
    }
    pub fn original_first_move(&self) -> BoardMove {
        self.action.preparation.first
    }
    pub fn original_deadline(&self) -> Deadline {
        self.action.preparation.deadline
    }
    pub fn started(&self) -> Instant {
        self.action.preparation.started
    }
    pub fn witnesses(&self) -> &[UtilityRouteWitness] {
        &self.action.witnesses
    }
    pub fn consumed_pc_requests(&self) -> impl Iterator<Item = &rz_contracts::RequestId> {
        self.action.pc_consumed.iter()
    }
    pub fn pc_consumption(&self) -> &[UtilityRoutePcConsumption] {
        &self.action.pc_origins
    }
    pub fn dependencies(&self) -> &[UtilityRouteDependencyWitness] {
        &self.action.authorized
    }

    pub fn native_executions(&self) -> impl Iterator<Item = &rz_contracts::ExecutionId> {
        self.action.native_executions.iter()
    }
    pub fn original_budget(&self) -> FrameBudget {
        self.action.preparation.budget
    }
    pub fn r_pc_consumption(&self) -> &[CpuSemanticWitness] {
        &self.action.r_pc_consumption
    }
    pub fn semantic_consumption_complete(&self) -> bool {
        self.action.semantic_consumption_complete()
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    pub fn prepare_utility_route(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        target_record: u16,
    ) -> Result<UtilityRoutePreparation, PalsError> {
        let started = Instant::now();
        self.check_work(work)?;
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        let recipient = match work.input.stage.role() {
            Some(Role::Proposer) => UtilityRouteRecipient::Proposer,
            Some(Role::Critic) => UtilityRouteRecipient::Critic,
            Some(Role::Validator) => {
                return Err(RoleError::Unsupported("utility is not a V action").into());
            }
            None => UtilityRouteRecipient::Researcher,
        };
        let frontier = self.rovent_frontier().ok_or(RoleError::Unavailable)?;
        if !frontier
            .registration()
            .mode
            .supports(recipient.search_recipient())
        {
            return Err(RoleError::Unsupported("utility recipient unavailable").into());
        }
        let metadata = work.input.view.records_metadata();
        let target = metadata
            .get(usize::from(target_record))
            .ok_or(RoleError::InvalidOutput)?;
        let first = work.input.question.first.ok_or(RoleError::InvalidOutput)?;
        if target.original_first_move != Some(first) {
            return Err(RoleError::InvalidOutput.into());
        }
        let target_records = metadata
            .iter()
            .enumerate()
            .filter_map(|(index, row)| {
                (row.source_state == target.source_state && row.original_first_move == Some(first))
                    .then_some(index as u64)
            })
            .collect();
        let preparation = UtilityRoutePreparation {
            id: UtilityRouteActionId {
                owner: Arc::new(()),
                authority: self.authority,
                frame: work.frame,
                question: work.input.question_digest,
                recipient,
            },
            store: self.engine.semantic.stores.read_owner(),
            generation: self.authority.root.0,
            root: self.root_state_id(),
            root_budget: frontier.registration().root_budget,
            mode: frontier.registration().mode,
            deadline: work.original_deadline,
            hard_deadline: work.input.question.deadline,
            first,
            budget: work.budget,
            target_record,
            target_records,
            source_view: Arc::downgrade(&work.input.view),
            source_query: Arc::downgrade(&work.input.question),
            view_seal: work.input.view_seal,
            full_view: ContractDigest(work.input.view.digest()),
            first_observation: self
                .engine
                .semantic
                .stores
                .observations
                .next_observation_sequence(),
            baseline: self.validator_existing_witnesses()?,
            search_scope: self.utility_route_search_scope(work)?,
            started,
            request: None,
        };
        self.utility_continuations
            .enable_bounded_payload(work.budget.original_store_byte_limit);
        Ok(preparation)
    }
}

pub struct UtilityRouteBindRejection {
    pub preparation: UtilityRoutePreparation,
    pub input: Arc<NnuecUtilityInputCapture>,
    pub error: PalsError,
}
impl UtilityRouteAction {
    pub fn id(&self) -> &UtilityRouteActionId {
        &self.preparation.id
    }
    pub fn input(&self) -> &Arc<NnuecUtilityInputCapture> {
        &self.input
    }
    pub fn started(&self) -> Instant {
        self.preparation.started
    }
    pub fn original_deadline(&self) -> Deadline {
        self.preparation.deadline
    }
    pub fn root_budget(&self) -> ContractDigest {
        self.preparation.root_budget
    }
    pub fn failed(&self) -> bool {
        self.failed
    }
    pub fn has_committed(&self) -> bool {
        !self.committed.is_empty()
    }
    pub fn semantic_consumption_complete(&self) -> bool {
        match self.id().recipient() {
            UtilityRouteRecipient::Proposer | UtilityRouteRecipient::Critic => {
                !self.pc_origins.is_empty()
            }
            UtilityRouteRecipient::Researcher => !self.r_pc_consumption.is_empty(),
        }
    }
    pub fn mark_failed(&mut self) {
        self.failed = true;
    }
    pub fn into_unresolved(
        self,
        reason: UtilityRouteUnresolvedReason,
    ) -> UtilityRouteUnresolvedAction {
        UtilityRouteUnresolvedAction {
            action: self,
            reason,
        }
    }
}
impl<M: RoleModel> AdaptiveEnginePort<M> {
    // Rejection returns the same move-only Request/preparation/action owner;
    // boxing here would require a new allocation while preserving old pins.
    #[allow(clippy::result_large_err)]
    pub fn bind_utility_route_input(
        &self,
        preparation: UtilityRoutePreparation,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        input: Arc<NnuecUtilityInputCapture>,
    ) -> Result<UtilityRouteAction, UtilityRouteBindRejection> {
        let checked = (|| -> Result<(), PalsError> {
            self.check_work(work)?;
            AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
            let original = preparation
                .source_view
                .upgrade()
                .ok_or(RoleError::InvalidOutput)?;
            let original_query = preparation
                .source_query
                .upgrade()
                .ok_or(RoleError::InvalidOutput)?;
            if preparation.id.authority != self.authority
                || preparation.id.frame != work.frame
                || preparation.id.question != work.input.question_digest
                || preparation.deadline != work.original_deadline
                || preparation.hard_deadline != work.input.question.deadline
                || preparation.budget != work.budget
                || preparation.generation != self.authority.root.0
                || preparation.root != self.root_state_id()
                || !preparation.store.matches(&self.engine.semantic.stores)
                || !Arc::ptr_eq(&original, &work.input.view)
                || !Arc::ptr_eq(&original_query, &work.input.question)
                || self
                    .engine
                    .semantic
                    .stores
                    .observations
                    .next_observation_sequence()
                    != preparation.first_observation
                || self.validator_existing_witnesses()? != preparation.baseline
                || !input.validates(
                    &self.engine.semantic.stores,
                    preparation.root,
                    preparation.generation,
                    &original,
                )
                || preparation
                    .request
                    .as_ref()
                    .is_none_or(|request| !input.matches_request(request))
                || input.target_record_ids() != preparation.target_records
                || input.original_first_move() != preparation.first
            {
                return Err(RoleError::InvalidOutput.into());
            }
            Ok(())
        })();
        if let Err(error) = checked {
            return Err(UtilityRouteBindRejection {
                preparation,
                input,
                error,
            });
        }
        Ok(new_utility_route_action(preparation, input))
    }
}

fn new_utility_route_action(
    preparation: UtilityRoutePreparation,
    input: Arc<NnuecUtilityInputCapture>,
) -> UtilityRouteAction {
    let frozen_context = utility_route_frozen_context(&preparation, &input);
    UtilityRouteAction {
        preparation,
        input,
        frozen_context,
        committed: HashSet::new(),
        committed_sources: HashMap::new(),
        witness_watermarks: HashMap::new(),
        authorized: Vec::new(),
        r_parents: Vec::new(),
        pc_consumed: BTreeSet::new(),
        pc_origins: Vec::new(),
        native_executions: BTreeSet::new(),
        witnesses: Vec::new(),
        witnessed: HashSet::new(),
        r_pc_consumption: Vec::new(),
        failed: false,
        intervention: None,
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Read-only gate for the original actual work. Additional questions need
    /// a separately validated private dependency, never frame equality alone.
    pub fn validate_utility_route_work(
        &self,
        action: &UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), PalsError> {
        self.check_work(work)?;
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        if action.failed
            || !action
                .preparation
                .store
                .matches(&self.engine.semantic.stores)
            || action.preparation.id.authority != self.authority
            || action.preparation.generation != self.authority.root.0
            || action.preparation.deadline != work.original_deadline
            || work.input.question.first != Some(action.preparation.first)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let original = action.preparation.source_query.upgrade();
        let view = action.preparation.source_view.upgrade();
        let original_matches = action.preparation.id.frame == work.frame
            && action.preparation.id.question == work.input.question_digest
            && original
                .as_ref()
                .is_some_and(|q| Arc::ptr_eq(q, &work.input.question))
            && view
                .as_ref()
                .is_some_and(|v| Arc::ptr_eq(v, &work.input.view));
        if !original_matches && !action.authorized.iter().any(|held| held.matches(work)) {
            return Err(RoleError::Unsupported(
                "utility descendant requires original dependency proof",
            )
            .into());
        }
        Ok(())
    }

    /// Enroll only the exact child retained by the actual reciprocal owner.
    /// A copied envelope, matching FrameId or a caller boolean is insufficient.
    pub fn register_utility_route_descendant(
        &self,
        action: &mut UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), PalsError> {
        self.check_work(work)?;
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        if action.failed
            || !action
                .preparation
                .store
                .matches(&self.engine.semantic.stores)
            || action.preparation.id.authority != self.authority
            || action.preparation.generation != self.authority.root.0
            || work.original_deadline != action.preparation.deadline
            || work.input.question.first != Some(action.preparation.first)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if let Some(held) = action.authorized.iter().find(|held| {
            held.question == work.input.question_digest
                || held.frame == work.frame
                    && held
                        .source_query
                        .upgrade()
                        .is_some_and(|q| Arc::ptr_eq(&q, &work.input.question))
        }) {
            return if held.matches(work) {
                Ok(())
            } else {
                Err(RoleError::InvalidOutput.into())
            };
        }
        let continuation = self.validated_utility_continuation(work)?;
        let linked = if let Some(source) = &continuation {
            if let Some((request, frame, question, view)) = source.role_source() {
                action.pc_origins.iter().any(|source| {
                    source.origin.ticket().id == request
                        && source.origin.frame() == frame
                        && source.question == question
                        && source.view.view_digest == view
                })
            } else if let Some((job, execution, frame, question, _observation)) =
                source.cpu_source()
            {
                action
                    .committed_sources
                    .get(&StepExecution::Cpu { job, execution })
                    == Some(&(frame, question))
            } else {
                false
            }
        } else {
            false
        };
        if !linked {
            return Err(RoleError::InvalidOutput.into());
        }
        if action.authorized.len() >= MAX_ACTION_WITNESSES {
            return Err(PalsError::Capacity);
        }
        action
            .authorized
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        action.authorized.push(UtilityRouteDependencyWitness {
            frame: work.frame,
            question: work.input.question_digest,
            view: work.input.view_seal,
            full_view: ContractDigest(work.input.view.digest()),
            source_query: Arc::downgrade(&work.input.question),
            source_view: Arc::downgrade(&work.input.view),
            peer: None,
            canonical_peer: None,
            coordinator_parent: None,
            continuation,
        });
        Ok(())
    }
    /// A native peer needs the coordinator's actual admitted/joined edge,
    /// plus the port's original pending Q/view/source validation.
    pub fn register_utility_route_peer(
        &self,
        action: &mut UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        proof: &crate::pals::coordinator::ReciprocalDependencyProof<
            ReviewView,
            AdaptivePortQuestion,
        >,
    ) -> Result<(), PalsError> {
        self.check_work(work)?;
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        if action.failed
            || !action
                .preparation
                .store
                .matches(&self.engine.semantic.stores)
            || action.preparation.id.authority != self.authority
            || action.preparation.generation != self.authority.root.0
            || work.original_deadline != action.preparation.deadline
            || work.input.question.first != Some(action.preparation.first)
            || !proof.validates_child(work)
            || proof.child() != work.frame
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let (canonical, origin) = self.validated_utility_peer_dependency(work, proof)?;
        let alias = proof.alias_question();
        let linked = if let Some(origin) = origin {
            origin.frame() == proof.parent()
                && action
                    .pc_origins
                    .iter()
                    .any(|source| source.origin == origin)
        } else {
            alias.parent.is_some_and(|parent| {
                action.r_parents.iter().any(|(frame, source)| {
                    *frame == proof.parent()
                        && source.id == parent
                        && source.root_budget == alias.root_budget
                        && proof.parent_question() == Some(*source)
                })
            })
        };
        if !linked
            || alias.root_budget != action.preparation.root_budget
            || canonical.question_digest != work.input.question_digest
            || canonical.view != work.input.view_seal
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if let Some(held) = action
            .authorized
            .iter()
            .find(|held| held.frame == work.frame && held.question == work.input.question_digest)
        {
            return if held.matches(work)
                && held.peer == Some((alias, origin))
                && held.canonical_peer == Some(canonical)
                && held.coordinator_parent == Some(proof.parent())
            {
                Ok(())
            } else {
                Err(RoleError::InvalidOutput.into())
            };
        }
        if action.authorized.len() >= MAX_ACTION_WITNESSES {
            return Err(PalsError::Capacity);
        }
        action
            .authorized
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        action.authorized.push(UtilityRouteDependencyWitness {
            frame: work.frame,
            question: work.input.question_digest,
            view: work.input.view_seal,
            full_view: ContractDigest(work.input.view.digest()),
            source_query: Arc::downgrade(&work.input.question),
            source_view: Arc::downgrade(&work.input.view),
            peer: Some((alias, origin)),
            canonical_peer: Some(canonical),
            coordinator_parent: Some(proof.parent()),
            continuation: None,
        });
        Ok(())
    }

    pub fn begin_utility_route_commit(
        &self,
        action: &UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        execution: StepExecution,
    ) -> Result<UtilityRouteCommitGuard, PalsError> {
        self.validate_utility_route_work(action, work)?;
        if action.failed
            || !action
                .preparation
                .store
                .matches(&self.engine.semantic.stores)
            || action.preparation.id.authority != self.authority
            || action.preparation.generation != self.authority.root.0
            || action.preparation.deadline != work.original_deadline
            || work.input.question.first != Some(action.preparation.first)
            || action.committed.contains(&execution)
            || action.committed.len() >= MAX_ACTION_WITNESSES
            || matches!(
                execution,
                StepExecution::ExternalCpu { .. } | StepExecution::ProfiledExternal { .. }
            )
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(UtilityRouteCommitGuard {
            action: action.preparation.id.clone(),
            frame: work.frame,
            question: work.input.question_digest,
            execution,
            first_observation: self
                .engine
                .semantic
                .stores
                .observations
                .next_observation_sequence(),
            generation: self.authority.root.0,
            r_parent: if matches!(execution, StepExecution::Cpu { .. }) {
                self.validated_utility_r_parent(work)?
            } else {
                None
            },
        })
    }

    /// The actual settled Await slice is executed work and ancestry only.
    /// It is never labelled RawPublished, a completed depth, or a search bound.
    pub fn commit_utility_route_await(
        &self,
        action: &mut UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &crate::cpu_worker::CpuSliceReport,
    ) -> Result<(), PalsError> {
        self.validate_utility_route_work(action, work)?;
        let source = self.validated_utility_await(work, receipt)?;
        let step = StepExecution::Cpu {
            job: source.job().id,
            execution: source.execution(),
        };
        if action.committed.contains(&step) {
            return if action.committed_sources.get(&step)
                == Some(&(work.frame, work.input.question_digest))
                && action.r_parents.contains(&(work.frame, source.parent()))
            {
                Ok(())
            } else {
                Err(RoleError::InvalidOutput.into())
            };
        }
        if action.committed.len() >= MAX_ACTION_WITNESSES {
            return Err(PalsError::Capacity);
        }
        action
            .committed
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        action
            .committed_sources
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        action
            .witness_watermarks
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        action
            .r_parents
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        action.committed.insert(step);
        action
            .committed_sources
            .insert(step, (work.frame, work.input.question_digest));
        action.witness_watermarks.insert(
            step,
            self.engine
                .semantic
                .stores
                .observations
                .next_observation_sequence(),
        );
        if !action.r_parents.contains(&(work.frame, source.parent())) {
            action.r_parents.push((work.frame, source.parent()));
        }
        action.native_executions.insert(source.execution());
        Ok(())
    }

    /// Only the exact original hot commit may contribute a witness. Repeating
    /// this same guard after deferred semantic materialization refreshes only
    /// the unread observation suffix, without recording another CPU action.
    /// Raw PV, depth or another frame's head does not qualify.
    pub fn commit_utility_route_step(
        &self,
        action: &mut UtilityRouteAction,
        guard: &UtilityRouteCommitGuard,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), PalsError> {
        self.check_work(work)?;
        let end = self
            .engine
            .semantic
            .stores
            .observations
            .next_observation_sequence();
        if guard.action != action.preparation.id
            || guard.frame != work.frame
            || guard.question != work.input.question_digest
            || guard.generation != self.authority.root.0
            || action.failed
            || !action
                .preparation
                .store
                .matches(&self.engine.semantic.stores)
            || end < guard.first_observation
            || end - guard.first_observation > MAX_ACTION_WITNESSES
            || action.committed.contains(&guard.execution)
                && action.committed_sources.get(&guard.execution)
                    != Some(&(guard.frame, guard.question))
        {
            return Err(RoleError::InvalidOutput.into());
        }
        match guard.execution {
            StepExecution::Cpu { job, execution }
                if self.cpu_action_committed(guard.question, job, execution) => {}
            _ => return Err(RoleError::InvalidOutput.into()),
        }
        self.commit_utility_route_observations(action, guard, work, None)
    }

    /// A RequestId and a successful public output are not a native acceptance.
    /// The exact actual accepted proof and already consumed original source
    /// are checked before any witness or request enters this action.
    pub fn commit_utility_route_role(
        &self,
        action: &mut UtilityRouteAction,
        guard: &UtilityRouteCommitGuard,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
    ) -> Result<(), PalsError> {
        self.check_work(work)?;
        let ticket = accepted.ticket();
        let previous = action
            .pc_origins
            .iter()
            .find(|source| source.origin.ticket().id == ticket.id)
            .cloned();
        let end = self
            .engine
            .semantic
            .stores
            .observations
            .next_observation_sequence();
        if guard.execution != StepExecution::Role(ticket.id)
            || guard.action != action.preparation.id
            || guard.frame != work.frame
            || guard.question != work.input.question_digest
            || guard.generation != self.authority.root.0
            || action.failed
            || ticket.authority != self.authority
            || ticket.question != work.input.question_digest
            || ticket.original_deadline != work.original_deadline
            || ticket.view != work.input.view_seal
            || !matches!(ticket.role, Role::Proposer | Role::Critic)
            || ticket.role != role_for_purpose(work.input.question.context.purpose)
            || !accepted.validates_output(output)
            || !self.role_consumption_recorded(ticket, &work.input.view)?
            || !(self.role_action_committed(
                guard.question,
                ticket.id,
                guard.first_observation,
                end,
            ) || self.reciprocal_source_consumed(work, ticket, output)
                || previous.as_ref().is_some_and(|source| {
                    source.origin.matches(accepted)
                        && source.question == work.input.question_digest
                        && source.view == work.input.view_seal
                        && source.full_view == ContractDigest(work.input.view.digest())
                        && self.accepted.contains(&ticket.id)
                        && self.last_role_commit.is_some_and(
                            |(question, request, start, finish)| {
                                question == guard.question
                                    && request == ticket.id
                                    && start >= guard.first_observation
                                    && finish == end
                            },
                        )
                }))
        {
            return Err(RoleError::InvalidOutput.into());
        }
        self.verify_registered_model_namespace(ticket, output)?;
        let consumption = if let Some(original) = previous {
            if !original.origin.matches(accepted)
                || original.question != work.input.question_digest
                || original.view != work.input.view_seal
                || original.full_view != ContractDigest(work.input.view.digest())
            {
                return Err(RoleError::InvalidOutput.into());
            }
            original
        } else {
            // Preserve the original private peer if this was an actual message.
            let peer = match action
                .authorized
                .iter()
                .find(|source| source.matches(work))
                .and_then(|source| source.canonical_peer)
            {
                Some(peer) => Some(peer),
                None => self.validated_utility_peer(work)?.map(|(peer, _)| peer),
            };
            let origin = accepted
                .origin(work.frame, peer)
                .map_err(|_| RoleError::InvalidOutput)?;
            UtilityRoutePcConsumption {
                origin,
                question: work.input.question_digest,
                view: work.input.view_seal,
                full_view: ContractDigest(work.input.view.digest()),
            }
        };
        action
            .pc_origins
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        self.commit_utility_route_observations(action, guard, work, Some(consumption))
    }

    fn commit_utility_route_observations(
        &self,
        action: &mut UtilityRouteAction,
        guard: &UtilityRouteCommitGuard,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        consumption: Option<UtilityRoutePcConsumption>,
    ) -> Result<(), PalsError> {
        let end = self
            .engine
            .semantic
            .stores
            .observations
            .next_observation_sequence();
        let refresh = action.committed.contains(&guard.execution);
        let first_observation = action
            .witness_watermarks
            .get(&guard.execution)
            .copied()
            .unwrap_or(guard.first_observation);
        if guard.action != action.preparation.id
            || guard.frame != work.frame
            || guard.question != work.input.question_digest
            || guard.generation != self.authority.root.0
            || action.failed
            || !action
                .preparation
                .store
                .matches(&self.engine.semantic.stores)
            || end < first_observation
            || end - first_observation > MAX_ACTION_WITNESSES
            || refresh
                && action.committed_sources.get(&guard.execution)
                    != Some(&(guard.frame, guard.question))
            || matches!(guard.execution, StepExecution::Role(_)) != consumption.is_some()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let mut fresh = Vec::new();
        for sequence in first_observation..end {
            let observation = ObservationId(sequence);
            let Some(kind) = self.validator_committed_witness_kind(observation)? else {
                continue;
            };
            let line = self
                .conditional_witness_line(observation)
                .ok_or(RoleError::InvalidOutput)?;
            if line.first() != Some(&action.preparation.first) {
                return Err(RoleError::InvalidOutput.into());
            }
            let identity = self.validator_witness_identity(kind, line)?;
            if action.preparation.baseline.contains(&identity)
                || action.witnessed.contains(&identity)
            {
                continue;
            }
            let mut raw = Sha256::new();
            raw.update(b"rz-rovent-utility-actual-conditional-witness/1\0");
            raw.update(guard.question.0);
            raw.update(guard.frame.sequence.to_le_bytes());
            raw.update(identity.0);
            self.validator_write_observation(&mut raw, observation, 0)?;
            fresh.push(UtilityRouteWitness {
                source_question: guard.question,
                source_frame: guard.frame,
                kind,
                observation,
                identity,
                raw_evidence: ContractDigest(raw.finalize().into()),
                line: line.to_vec(),
            });
        }
        let mut seen = HashSet::new();
        fresh.retain(|witness| seen.insert(witness.identity));
        if action
            .witnesses
            .len()
            .checked_add(fresh.len())
            .is_none_or(|n| n > MAX_ACTION_WITNESSES)
        {
            return Err(PalsError::Capacity);
        }
        action
            .witnesses
            .try_reserve(fresh.len())
            .map_err(|_| PalsError::Capacity)?;
        action
            .committed
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        action
            .witnessed
            .try_reserve(fresh.len())
            .map_err(|_| PalsError::Capacity)?;
        action
            .committed_sources
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        action
            .witness_watermarks
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        if guard.r_parent.is_some() {
            action
                .r_parents
                .try_reserve(1)
                .map_err(|_| PalsError::Capacity)?;
        }
        for witness in fresh {
            action.witnessed.insert(witness.identity);
            action.witnesses.push(witness);
        }
        if !refresh {
            if let Some(consumption) = consumption {
                action.pc_origins.push(consumption);
            }
        }
        action.committed.insert(guard.execution);
        action
            .committed_sources
            .insert(guard.execution, (guard.frame, guard.question));
        action.witness_watermarks.insert(guard.execution, end);
        if let Some(parent) = guard.r_parent {
            if !action.r_parents.contains(&(guard.frame, parent)) {
                action.r_parents.push((guard.frame, parent));
            }
        }
        match guard.execution {
            StepExecution::Role(request) => {
                action.pc_consumed.insert(request);
            }
            StepExecution::Cpu { execution, .. } => {
                action.native_executions.insert(execution);
            }
            _ => return Err(RoleError::InvalidOutput.into()),
        }
        Ok(())
    }

    // Rejection returns the same move-only Request/preparation/action owner;
    // boxing here would require a new allocation while preserving old pins.
    #[allow(clippy::result_large_err)]
    pub fn finish_utility_route_semantics(
        &self,
        action: UtilityRouteAction,
    ) -> Result<UtilityRouteSemanticOutcome, (PalsError, UtilityRouteAction)> {
        let valid = !action.failed
            && action
                .preparation
                .store
                .matches(&self.engine.semantic.stores)
            && action.preparation.id.authority == self.authority
            && action.preparation.generation == self.authority.root.0
            && !action.committed.is_empty();
        if !valid {
            return Err((RoleError::InvalidOutput.into(), action));
        }
        Ok(UtilityRouteSemanticOutcome { action })
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Original global utility input preparation, available for actual P/C/R
    /// route work without constructing another researcher or utility forward.
    pub fn prepare_utility_route_input_request(
        &self,
        preparation: &mut UtilityRoutePreparation,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        model: &Model,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<Request, AdaptiveNnuecPreparationError> {
        self.check_work(work)?;
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        let original = preparation
            .source_view
            .upgrade()
            .ok_or_else(|| PalsError::from(RoleError::InvalidOutput))?;
        let original_query = preparation
            .source_query
            .upgrade()
            .ok_or_else(|| PalsError::from(RoleError::InvalidOutput))?;
        if preparation.id.authority != self.authority
            || preparation.id.frame != work.frame
            || preparation.id.question != work.input.question_digest
            || !preparation.store.matches(&self.engine.semantic.stores)
            || !Arc::ptr_eq(&original, &work.input.view)
            || !Arc::ptr_eq(&original_query, &work.input.question)
            || self.engine.researcher.cpu_registered_value.as_ref() != Some(model.identity())
        {
            return Err(AdaptiveNnuecPreparationError::StateMismatch);
        }
        let request = Request::admit(
            model,
            Arc::clone(&original),
            &self.engine.semantic.stores,
            preparation.root,
            preparation.generation,
            preparation.hard_deadline,
            cancel,
        )?;
        preparation
            .attach_input_capture(request)
            .map_err(|(error, _request)| error)
    }
    /// Only the Core owner can have issued this already selected/consumed
    /// R→P/C proof. It adds no readout and performs no numerical evaluation.
    pub fn record_utility_route_r_pc_consumption(
        &self,
        action: &mut UtilityRouteAction,
        witness: &CpuSemanticWitness,
    ) -> Result<(), PalsError> {
        let ticket = witness
            .selected_pc_ticket()
            .ok_or(RoleError::InvalidOutput)?;
        if action.failed
            || witness.stage() != CpuSemanticStage::SelectedPcViewConsumed
            || witness.authority() != action.preparation.id.authority
            || !action
                .preparation
                .store
                .matches(&self.engine.semantic.stores)
            || !(action
                .input
                .matches_capture_lineage(witness.capture_lineage())
                || action.intervention.as_ref().is_some_and(|source| {
                    source.matches_arm_capture_lineage(witness.capture_lineage())
                }))
            || !action
                .native_executions
                .contains(&witness.native_execution())
            || !matches!(ticket.role, Role::Proposer | Role::Critic)
            || !action.pc_origins.iter().any(|source| {
                source.origin.ticket() == ticket
                    && Some(source.origin.output_digest()) == witness.selected_pc_output_digest()
            })
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if action.r_pc_consumption.iter().any(|existing| {
            existing.native_execution() == witness.native_execution()
                && existing.selected_pc_ticket() == witness.selected_pc_ticket()
        }) {
            return Ok(());
        }
        if action.r_pc_consumption.len() >= MAX_ACTION_WITNESSES {
            return Err(PalsError::Capacity);
        }
        action
            .r_pc_consumption
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        action.r_pc_consumption.push(witness.clone());
        action.pc_consumed.insert(ticket.id);
        Ok(())
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    fn utility_route_search_scope(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<ContractDigest, PalsError> {
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-utility-route-search-scope/1\0");
        hash.update(self.rules.snapshot().identity().semantic.0);
        let frontier = self.rovent_frontier().ok_or(RoleError::Unavailable)?;
        hash.update([match frontier.registration().mode {
            rz_contracts::rovent::ExecutionMode::HybridPcr => 0,
            rz_contracts::rovent::ExecutionMode::HybridPc => 1,
            rz_contracts::rovent::ExecutionMode::CpuPcr => 2,
            rz_contracts::rovent::ExecutionMode::CpuPc => 3,
        }]);
        hash.update((frontier.root_moves().allowed().len() as u64).to_le_bytes());
        for movement in frontier.root_moves().allowed().iter() {
            hash.update(Move16::pack(*movement)?.bits().to_le_bytes());
        }
        let sequence = self.engine.semantic_sequence_config();
        hash.update((sequence.line_plies as u64).to_le_bytes());
        hash.update((sequence.max_nodes as u64).to_le_bytes());
        hash.update((sequence.max_records as u64).to_le_bytes());
        hash.update(sequence.max_role_calls.to_le_bytes());
        hash.update(work.original_first_move.bits().to_le_bytes());
        hash.update(work.budget.role_submissions_remaining.to_le_bytes());
        hash.update(work.budget.original_cpu_credit.to_le_bytes());
        hash.update(work.budget.original_store_byte_limit.to_le_bytes());
        hash.update(self.limits.max_rounds.to_le_bytes());
        hash.update(self.limits.max_cpu_nodes.to_le_bytes());
        hash.update(self.limits.cpu_depth.to_le_bytes());
        hash.update(self.counters.cpu_nodes.to_le_bytes());
        hash.update(self.counters.role_calls.to_le_bytes());
        hash.update((self.engine.config.line_plies as u64).to_le_bytes());
        hash.update((self.engine.config.beam_width as u64).to_le_bytes());
        hash.update(self.engine.config.cpu_nodes_per_task.to_le_bytes());
        for bytes in [
            self.engine.resolver.version().as_bytes(),
            self.engine.post_repair_recheck.search_identity().as_bytes(),
            self.engine.semantic.model.identity().as_bytes(),
            self.engine.researcher.cpu_registered_condition.as_bytes(),
        ] {
            utility_route_bytes(&mut hash, bytes);
        }
        let value = self
            .engine
            .semantic
            .model
            .value_identity()
            .ok_or(RoleError::Unavailable)?;
        value.validate()?;
        for text in [
            &value.semantics,
            &value.model,
            &value.encoding,
            &value.precision,
        ] {
            utility_route_bytes(&mut hash, text.as_bytes());
        }
        hash.update(value.model_epoch);
        Ok(ContractDigest(hash.finalize().into()))
    }
}
fn utility_route_bytes(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}
fn utility_route_frozen_context(
    preparation: &UtilityRoutePreparation,
    input: &NnuecUtilityInputCapture,
) -> ContractDigest {
    let mut hash = Sha256::new();
    hash.update(b"rz-rovent-utility-frozen-global-target-context/1\0");
    hash.update(preparation.search_scope.0);
    utility_route_bytes(&mut hash, input.input_sha256().as_bytes());
    utility_route_bytes(&mut hash, input.target_sha256().as_bytes());
    // Whole target membership and order were frozen by the actual actor.
    hash.update((preparation.target_records.len() as u32).to_le_bytes());
    for record in &preparation.target_records {
        hash.update(record.to_le_bytes());
    }
    let mut baseline: Vec<_> = preparation.baseline.iter().copied().collect();
    baseline.sort_unstable_by_key(|digest| digest.0);
    hash.update((baseline.len() as u32).to_le_bytes());
    for witness in baseline {
        hash.update(witness.0);
    }
    ContractDigest(hash.finalize().into())
}
