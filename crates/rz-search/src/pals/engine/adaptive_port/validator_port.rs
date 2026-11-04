//! Collection-only V authority over actual, bounded Search actions.
//!
//! A task head cannot certify a witness. The registry first accepts its
//! original physical output; this owner then checks the frozen question and
//! actual capability. Witnesses are read from the hot Store after an executed
//! action's commit, never manufactured from a coordinator completion flag.
use super::*;
use crate::cpu_worker::CpuWorkerProfile;
use crate::external_cpu::{CpuEndpoint, CpuEndpointSelection, CpuExecutionUse};
use crate::pals::coordinator::{AcceptedValidatorPublished, FrameId, StepExecution};
use rz_contracts::ExecutionId as NativeExecutionId;
use rz_contracts::pals_adaptive::{CpuJobId, CpuWorkerId};
use std::collections::{BTreeSet, HashMap};

mod observation_writer;
use observation_writer::ObservationShaSink;
pub(super) use observation_writer::{BoundedObservationPreimage, ObservationSink};

pub const MAX_VALIDATOR_CONTEXTS_PER_ROOT: usize = 128;
pub const MAX_VALIDATOR_ACTIONS_PENDING: usize = 64;
pub const MAX_ACTION_WITNESSES: usize = 64;
pub const MAX_ACTION_EXECUTIONS: usize = 4096;
pub const MAX_ACTION_ROLE_SUBMISSIONS: u64 = 64;

fn validator_write_bytes(hash: &mut Sha256, bytes: &[u8]) {
    observation_writer::write_bytes(&mut ObservationShaSink(hash), bytes);
}
fn validator_write_moves(hash: &mut Sha256, moves: &[BoardMove]) -> Result<(), PalsError> {
    observation_writer::write_moves(&mut ObservationShaSink(hash), moves)
}
fn validator_write_optional_moves(
    hash: &mut Sha256,
    moves: Option<&[BoardMove]>,
) -> Result<(), PalsError> {
    hash.update([u8::from(moves.is_some())]);
    if let Some(moves) = moves {
        validator_write_moves(hash, moves)?;
    }
    Ok(())
}
/// A normalized selected view, retaining ordered local relations and exact
/// anchor histories while excluding source store handles/revision counters.
fn validator_write_review(hash: &mut Sha256, view: &ReviewView) -> Result<(), PalsError> {
    hash.update(view.policy_digest());
    hash.update(view.token_count().to_le_bytes());
    hash.update(view.length_bucket().to_le_bytes());
    hash.update((view.anchors().len() as u64).to_le_bytes());
    for anchor in view.anchors() {
        hash.update(rz_position::contracts::rules_snapshot_semantic_digest(&anchor.snapshot).0);
    }
    if view.records().len() != view.records_metadata().len() {
        return Err(RoleError::InvalidOutput.into());
    }
    hash.update((view.records().len() as u64).to_le_bytes());
    for (record, meta) in view.records().iter().zip(view.records_metadata()) {
        hash.update([
            record.kind as u8,
            u8::from(record.perspective == Color::Black),
            u8::from(record.critical),
            meta.anchor,
            u8::from(record.value.is_some()),
        ]);
        hash.update(record.value.unwrap_or(0).to_le_bytes());
        hash.update(record.completed_depth.to_le_bytes());
        hash.update([match record.score_scope {
            None => 0,
            Some(crate::cpu::CpuScoreScope::FrontierOnly) => 1,
            Some(crate::cpu::CpuScoreScope::CompletedIteration) => 2,
            Some(crate::cpu::CpuScoreScope::RulesTerminal) => 3,
        }]);
        for relation in [meta.parent, meta.supersedes] {
            hash.update([u8::from(relation.is_some())]);
            hash.update(relation.unwrap_or(0).to_le_bytes());
        }
        hash.update(meta.source_line_offset.to_le_bytes());
        validator_write_optional_moves(
            hash,
            meta.original_first_move.as_ref().map(std::slice::from_ref),
        )?;
        validator_write_moves(hash, &record.line)?;
    }
    let q = view.query();
    hash.update([q.purpose as u8]);
    validator_write_moves(hash, &q.prefix)?;
    validator_write_moves(hash, &q.proposal)?;
    validator_write_optional_moves(hash, q.counterexample.as_deref())?;
    validator_write_moves(hash, &q.legal)?;
    hash.update((q.divergences.len() as u64).to_le_bytes());
    for ply in &q.divergences {
        hash.update((*ply as u64).to_le_bytes());
    }
    Ok(())
}

/// Append-only order shared by the actual V head and generation V3.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AdaptiveValidatorAction {
    DefendResponse,
    AttackRepair,
    WidenResponses,
    LowerSelectivity,
    ResumeTask,
    CrossProfileRecheck,
    Defer,
}

#[cfg(test)]
#[path = "validator_port_tests.rs"]
mod tests;
impl AdaptiveValidatorAction {
    pub const ALL: [Self; 7] = [
        Self::DefendResponse,
        Self::AttackRepair,
        Self::WidenResponses,
        Self::LowerSelectivity,
        Self::ResumeTask,
        Self::CrossProfileRecheck,
        Self::Defer,
    ];
    pub const fn index(self) -> usize {
        self as usize
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::DefendResponse => "defend_response",
            Self::AttackRepair => "attack_repair",
            Self::WidenResponses => "widen_responses",
            Self::LowerSelectivity => "lower_selectivity",
            Self::ResumeTask => "resume_task",
            Self::CrossProfileRecheck => "cross_profile_recheck",
            Self::Defer => "defer",
        }
    }
}
/// Selection policy is provenance, not an invented task head. Both policies
/// retain the original accepted V logits and actual immutable support mask.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidatorActionSelection {
    ValidatorArgmax,
    RegisteredSupportedEnumerationV1,
}
impl ValidatorActionSelection {
    pub const fn name(self) -> &'static str {
        match self {
            Self::ValidatorArgmax => "validator-supported-argmax-v1",
            Self::RegisteredSupportedEnumerationV1 => "registered-supported-enumeration-v1",
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidatorUnsupported {
    NoEligibleQuestion,
    NoUnexaminedResponse,
    NoSameOwnerPausedJob,
    NoRemainingBudget,
    NoSelectivityControl,
    NoRegisteredIndependentProfile,
    AlreadyOwnedByAction,
}

#[derive(Clone)]
enum ActionOffer {
    Responses {
        node: usize,
        prefix: Vec<BoardMove>,
        moves: Vec<BoardMove>,
    },
    Attack {
        node: usize,
        prefix: Vec<BoardMove>,
    },
    Resume {
        job: CpuJobId,
        owner: CpuWorkerId,
    },
    Defer,
}

/// Issued by the Search owner before V submission. The support mask has no
/// public boolean setter and stays bound to this actual immutable Task view.
#[derive(Clone)]
pub struct AdaptiveValidatorQuestion {
    frozen_context: ContractDigest,
    input: FrameInput<ReviewView, AdaptivePortQuestion>,
    target: CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    budget: FrameBudget,
    offers: [Option<ActionOffer>; 7],
    // Action inputs are sealed before V is submitted. Explicit outcome
    // comparisons cannot add later Store publications to an earlier input.
    prepared_inputs: [Option<FrameInput<ReviewView, AdaptivePortQuestion>>; 7],
    unsupported: [Option<ValidatorUnsupported>; 7],
    prepared_at: Instant,
    observation_watermark: usize,
    baseline_witnesses: HashSet<ContractDigest>,
}
impl AdaptiveValidatorQuestion {
    pub fn frozen_context_digest(&self) -> ContractDigest {
        self.frozen_context
    }
    pub fn input(&self) -> &FrameInput<ReviewView, AdaptivePortQuestion> {
        &self.input
    }
    pub fn parent(&self) -> FrameId {
        self.target.frame
    }
    pub fn original_first_move(&self) -> rz_contracts::pals::Move16 {
        self.target.original_first_move
    }
    pub fn original_deadline(&self) -> Deadline {
        self.target.original_deadline
    }
    pub fn support_mask(&self) -> [bool; 7] {
        std::array::from_fn(|i| self.offers[i].is_some())
    }
    /// Store append ordinal at the actual immutable input capture. This is a
    /// Store namespace, not the generation collector's chronological sequence.
    pub fn input_observation_sequence(&self) -> u64 {
        self.observation_watermark as u64
    }
    pub fn unsupported_reason(
        &self,
        action: AdaptiveValidatorAction,
    ) -> Option<ValidatorUnsupported> {
        self.unsupported[action.index()]
    }
}

/// A frame plan is an additional bounded question under the parent's original
/// budget. Resume retains its existing frame/worker/Job and never grants fresh
/// native credit. Defer submits no work and grants no positive utility label.
pub enum AdaptiveValidatorEffect<Q = AdaptivePortQuestion> {
    Frame {
        input: FrameInput<ReviewView, Q>,
        budget: FrameBudget,
    },
    Resume {
        frame: FrameId,
        job: CpuJobId,
        owner: CpuWorkerId,
    },
    Defer,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ValidatorActionId {
    pub authority: SearchAuthority,
    pub validator_request: RequestId,
    pub action: AdaptiveValidatorAction,
}
/// Constructor is private to checked actual Search state + accepted V proof.
pub struct AdaptiveValidatorActionPlan<Q = AdaptivePortQuestion> {
    id: ValidatorActionId,
    frozen_context: ContractDigest,
    selection: ValidatorActionSelection,
    task_logits: [f32; 7],
    source: RoleTicketEnvelope,
    source_input: ContractDigest,
    source_output: ContractDigest,
    source_public_context: ContractDigest,
    source_consumption: ContractDigest,
    parent_question: ContractDigest,
    parent_view: ReviewViewSeal,
    parent_input: ContractDigest,
    parent_public_context: ContractDigest,
    parent: FrameId,
    first: rz_contracts::pals::Move16,
    deadline: Deadline,
    support: [bool; 7],
    effect: AdaptiveValidatorEffect<Q>,
}
impl<Q> AdaptiveValidatorActionPlan<Q> {
    pub fn frozen_context_digest(&self) -> ContractDigest {
        self.frozen_context
    }
    pub fn id(&self) -> ValidatorActionId {
        self.id
    }
    pub fn action(&self) -> AdaptiveValidatorAction {
        self.id.action
    }
    pub fn selection_provenance(&self) -> ValidatorActionSelection {
        self.selection
    }
    pub fn accepted_task_logits(&self) -> &[f32; 7] {
        &self.task_logits
    }
    pub fn source(&self) -> &RoleTicketEnvelope {
        &self.source
    }
    pub fn source_input(&self) -> ContractDigest {
        self.source_input
    }
    pub fn source_output(&self) -> ContractDigest {
        self.source_output
    }
    pub fn source_public_context(&self) -> ContractDigest {
        self.source_public_context
    }
    pub fn source_consumption(&self) -> ContractDigest {
        self.source_consumption
    }
    pub fn parent_question(&self) -> ContractDigest {
        self.parent_question
    }
    pub fn parent_view(&self) -> ReviewViewSeal {
        self.parent_view
    }
    pub fn parent_input(&self) -> ContractDigest {
        self.parent_input
    }
    pub fn parent_public_context(&self) -> ContractDigest {
        self.parent_public_context
    }
    pub fn parent(&self) -> FrameId {
        self.parent
    }
    pub fn original_first_move(&self) -> rz_contracts::pals::Move16 {
        self.first
    }
    pub fn original_deadline(&self) -> Deadline {
        self.deadline
    }
    pub fn action_support_mask(&self) -> [bool; 7] {
        self.support
    }
    pub fn effect(&self) -> &AdaptiveValidatorEffect<Q> {
        &self.effect
    }
    pub fn into_effect(self) -> AdaptiveValidatorEffect<Q> {
        self.effect
    }
}

#[derive(Default)]
pub(super) struct ValidatorPortState {
    contexts: HashMap<RequestId, (ContractDigest, u8)>,
    actions: HashMap<ValidatorActionId, ActionTracking>,
}
struct ActionTracking {
    source: RoleTicketEnvelope,
    frozen_context: ContractDigest,
    selection: ValidatorActionSelection,
    task_logits: [f32; 7],
    source_output: ContractDigest,
    source_public_context: ContractDigest,
    source_consumption: ContractDigest,
    input: ContractDigest,
    parent_question: ContractDigest,
    parent_view: ReviewViewSeal,
    parent_input: ContractDigest,
    parent_public_context: ContractDigest,
    parent: FrameId,
    first: rz_contracts::pals::Move16,
    support: [bool; 7],
    input_observation_sequence: usize,
    started: Instant,
    frame: Option<FrameId>,
    executions: BTreeSet<NativeExecutionId>,
    role_requests: HashSet<RequestId>,
    witnessed: HashSet<ContractDigest>,
    witnesses: Vec<ActionWitnessCommit>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionWitnessKind {
    Counter,
    Repair,
}
/// Opaque Store-backed evidence; numeric handles are provenance, not NN input.
#[derive(Clone)]
pub struct ActionWitnessCommit {
    action: ValidatorActionId,
    kind: ActionWitnessKind,
    observation: ObservationId,
    generation: u64,
    actual_line: Vec<BoardMove>,
    identity: ContractDigest,
    raw_evidence: ContractDigest,
}
impl ActionWitnessCommit {
    pub fn action(&self) -> ValidatorActionId {
        self.action
    }
    pub fn kind(&self) -> ActionWitnessKind {
        self.kind
    }
    pub fn observation(&self) -> ObservationId {
        self.observation
    }
    /// Actual Store append ordinal; never synthesized from a dataset row ID.
    pub fn observation_sequence(&self) -> u64 {
        self.observation.0 as u64
    }
    pub fn store_generation(&self) -> u64 {
        self.generation
    }
    pub fn line(&self) -> &[BoardMove] {
        &self.actual_line
    }
    pub fn witness_identity(&self) -> ContractDigest {
        self.identity
    }
    pub fn raw_evidence(&self) -> ContractDigest {
        self.raw_evidence
    }
}
/// Issued before, and consumed only after, one exact actual commit call.
pub struct ActionCommitGuard {
    action: ValidatorActionId,
    frame: FrameId,
    question: ContractDigest,
    execution: StepExecution,
    first_observation: usize,
    generation: u64,
}
/// Entire prepare/queue/execute/commit/drain wall cost. Unknown completion does
/// not create this receipt or release its action's tracking state.
pub struct ValidatorActionOutcome {
    id: ValidatorActionId,
    frozen_context: ContractDigest,
    selection: ValidatorActionSelection,
    task_logits: [f32; 7],
    source_output: ContractDigest,
    source_public_context: ContractDigest,
    source_consumption: ContractDigest,
    source: RoleTicketEnvelope,
    input: ContractDigest,
    parent_question: ContractDigest,
    parent_view: ReviewViewSeal,
    parent_input: ContractDigest,
    parent_public_context: ContractDigest,
    first: rz_contracts::pals::Move16,
    support: [bool; 7],
    input_observation_sequence: usize,
    wall_cost: Duration,
    witnesses: Vec<ActionWitnessCommit>,
    executions: Vec<NativeExecutionId>,
    role_requests: Vec<RequestId>,
    started: Instant,
}
impl ValidatorActionOutcome {
    pub fn frozen_context_digest(&self) -> ContractDigest {
        self.frozen_context
    }
    pub fn id(&self) -> ValidatorActionId {
        self.id
    }
    pub fn selection_provenance(&self) -> ValidatorActionSelection {
        self.selection
    }
    pub fn accepted_task_logits(&self) -> &[f32; 7] {
        &self.task_logits
    }
    pub fn source_output(&self) -> ContractDigest {
        self.source_output
    }
    pub fn source_public_context(&self) -> ContractDigest {
        self.source_public_context
    }
    pub fn source_consumption(&self) -> ContractDigest {
        self.source_consumption
    }
    pub fn source(&self) -> &RoleTicketEnvelope {
        &self.source
    }
    pub fn input(&self) -> ContractDigest {
        self.input
    }
    pub fn parent_question(&self) -> ContractDigest {
        self.parent_question
    }
    pub fn parent_view(&self) -> ReviewViewSeal {
        self.parent_view
    }
    pub fn parent_input(&self) -> ContractDigest {
        self.parent_input
    }
    pub fn parent_public_context(&self) -> ContractDigest {
        self.parent_public_context
    }
    pub fn original_first_move(&self) -> rz_contracts::pals::Move16 {
        self.first
    }
    pub fn action_support_mask(&self) -> [bool; 7] {
        self.support
    }
    pub fn input_observation_sequence(&self) -> u64 {
        self.input_observation_sequence as u64
    }
    pub fn wall_cost(&self) -> Duration {
        self.wall_cost
    }
    pub fn witnesses(&self) -> &[ActionWitnessCommit] {
        &self.witnesses
    }
    pub fn executions(&self) -> &[NativeExecutionId] {
        &self.executions
    }
    pub fn role_requests(&self) -> &[RequestId] {
        &self.role_requests
    }
    /// Additional wall delay until the manager's actual physical fence. The
    /// caller records this as drain/wait, never service or kernel throughput.
    pub fn physical_close_wall_cost(&self) -> Duration {
        self.started.elapsed()
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Read-only eligibility for an S-policy prediction. This does not submit
    /// V, accept CPU feedback, or attest a new witness. A real paused receipt
    /// must still pass the original validator admission after Yield.
    pub fn published_cpu_feedback_eligible(
        &self,
        target: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        progress: &crate::cpu_worker::CpuWorkerProgress,
    ) -> bool {
        if self.check_work(target).is_err()
            || target.budget.role_submissions_remaining == 0
            || self.validator_state.contexts.len() >= MAX_VALIDATOR_CONTEXTS_PER_ROOT
            || self
                .validator_state
                .actions
                .values()
                .any(|a| a.frame == Some(target.frame))
        {
            return false;
        }
        let Some(admission) = self.cpu_admissions.get(&target.input.question_digest) else {
            return false;
        };
        let (Some(job), Some(grant)) = (admission.job, admission.grant) else {
            return false;
        };
        let Some(best) = progress.iteration.best_move else {
            return false;
        };
        !admission.terminal
            && !admission.quarantined
            && self.cpu_context_current(target, admission)
            && job.id == progress.job
            && job.question == target.input.question_digest
            && job.hard_deadline == target.original_deadline
            && grant.job == job.id
            && grant.execution == progress.execution
            && grant.owner == progress.owner
            && admission.owner == Some(progress.owner)
            && progress.iteration.completed_depth > 0
            && progress.iteration.completed_depth <= admission.target_depth
            && progress.iteration.nodes <= grant.node_credit
            && admission.position.legal_moves().contains(&best)
            && admission
                .restriction
                .as_ref()
                .is_none_or(|moves| moves.contains(&best))
    }

    /// Actual eligibility revision for deciding whether a question deserves a
    /// new V review. This reads owner state and never increments RequestId or
    /// search root generation. The digest is scheduler identity, not an NN
    /// numeric feature or caller-supplied continuation claim.
    pub fn validator_review_key(
        &self,
        target: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<ContractDigest, PalsError> {
        self.check_work(target)?;
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-validator-eligibility-revision/1");
        hash.update(self.authority.epoch.0.to_le_bytes());
        hash.update(self.authority.game.0.to_le_bytes());
        hash.update(self.authority.root.0.to_le_bytes());
        hash.update(self.authority.implementation.0);
        hash.update(target.frame.sequence.to_le_bytes());
        hash.update(target.input.question_digest.0);
        hash.update(target.input.public_context.0);
        hash.update(target.input.view.digest());
        hash.update(self.engine.semantic.revision.to_le_bytes());
        hash.update(
            (self
                .engine
                .semantic
                .stores
                .observations
                .next_observation_sequence() as u64)
                .to_le_bytes(),
        );
        if let Some(admission) = self
            .cpu_admissions
            .get(&target.input.question_digest)
            .filter(|admission| self.cpu_context_current(target, admission))
        {
            hash.update([1]);
            hash.update([
                u8::from(admission.continuation_retained),
                u8::from(admission.terminal),
                u8::from(admission.quarantined),
            ]);
            hash.update(admission.cumulative_nodes.to_le_bytes());
            hash.update(admission.progress_sequence.to_le_bytes());
            hash.update(admission.control_sequence.to_le_bytes());
            hash.update(admission.execution_sequence.to_le_bytes());
            if let (Some(job), Some(owner)) = (admission.job, admission.owner) {
                hash.update([1]);
                hash.update(job.id.sequence.to_le_bytes());
                hash.update(job.evaluation_namespace.0);
                hash.update(job.search_conditions.0);
                hash.update(owner.0.to_le_bytes());
            } else {
                hash.update([0]);
            }
            if let Some(grant) = admission.grant {
                hash.update([1]);
                hash.update(grant.execution.epoch.0.to_le_bytes());
                hash.update(grant.execution.sequence.to_le_bytes());
            } else {
                hash.update([0]);
            }
            if let Some((native, store_execution, observation)) = admission.committed_execution {
                hash.update([1]);
                hash.update(native.epoch.0.to_le_bytes());
                hash.update(native.sequence.to_le_bytes());
                hash.update((store_execution.0 as u64).to_le_bytes());
                hash.update((observation.0 as u64).to_le_bytes());
            } else {
                hash.update([0]);
            }
        } else {
            hash.update([0]);
        }
        Ok(ContractDigest(hash.finalize().into()))
    }

    pub fn prepare_validator_question(
        &mut self,
        target: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        budget: FrameBudget,
        profiles: &[CpuWorkerProfile],
        endpoints: CpuEndpointSelection,
    ) -> Result<AdaptiveValidatorQuestion, PalsError> {
        let prepared_at = Instant::now();
        let observation_watermark = self
            .engine
            .semantic
            .stores
            .observations
            .next_observation_sequence();
        endpoints
            .validate_generation_collection()
            .map_err(|_| PalsError::InvalidConfig)?;
        if endpoints.execution_use != CpuExecutionUse::Collection || !endpoints.load_v {
            return Err(RoleError::Unsupported("V is collection-only").into());
        }
        self.check_work(target)?;
        if self.closed.contains(&target.frame)
            || self.validator_state.contexts.len() >= MAX_VALIDATOR_CONTEXTS_PER_ROOT
            || profiles.len() > 4
        {
            return Err(PalsError::Capacity);
        }
        let q = &target.input.question;
        let mut offers: [Option<ActionOffer>; 7] = std::array::from_fn(|_| None);
        let mut unsupported = [Some(ValidatorUnsupported::NoEligibleQuestion); 7];
        // These engines have no reductions/null-move in any current own
        // profile; the external adapter also advertises no selectivity knob.
        unsupported[AdaptiveValidatorAction::LowerSelectivity.index()] =
            Some(ValidatorUnsupported::NoSelectivityControl);
        // Changing a profile requires a separately registered owner/namespace
        // and actual observation commit support. A distinct enum is not that.
        unsupported[AdaptiveValidatorAction::CrossProfileRecheck.index()] =
            Some(ValidatorUnsupported::NoRegisteredIndependentProfile);
        let budget_available = budget.original_cpu_credit > 0
            && budget.role_submissions_remaining > 0
            && self.counters.cpu_nodes < self.limits.max_cpu_nodes;
        if !budget_available {
            unsupported[AdaptiveValidatorAction::DefendResponse.index()] =
                Some(ValidatorUnsupported::NoRemainingBudget);
            unsupported[AdaptiveValidatorAction::WidenResponses.index()] =
                Some(ValidatorUnsupported::NoRemainingBudget);
        } else if let Some(ply) = q.attack_ply {
            if let Some(response) = q.proposal.get(ply).copied() {
                let prefix = q
                    .proposal
                    .get(..ply)
                    .ok_or(RoleError::InvalidOutput)?
                    .to_vec();
                let node = self.replay_node(&prefix)?;
                let position = &self.engine.semantic.nodes[node].position;
                // Response search is meaningful at the opponent's exact
                // ongoing anchor; there is no board-only rebase.
                if self.engine.semantic.nodes[node].terminal.is_none()
                    && position.side_to_move()
                        != self.engine.semantic.nodes[self.root]
                            .position
                            .side_to_move()
                    && position.legal_moves().contains(&response)
                {
                    offers[AdaptiveValidatorAction::DefendResponse.index()] =
                        Some(ActionOffer::Responses {
                            node,
                            prefix: prefix.clone(),
                            moves: vec![response],
                        });
                    unsupported[AdaptiveValidatorAction::DefendResponse.index()] = None;
                    let examined = &self.engine.semantic.nodes[node].edges;
                    let moves: Vec<_> = position
                        .legal_moves()
                        .into_iter()
                        .filter(|movement| {
                            *movement != response
                                && !examined.iter().any(|edge| edge.movement == *movement)
                        })
                        .collect();
                    if !moves.is_empty() {
                        offers[AdaptiveValidatorAction::WidenResponses.index()] =
                            Some(ActionOffer::Responses {
                                node,
                                prefix,
                                moves,
                            });
                        unsupported[AdaptiveValidatorAction::WidenResponses.index()] = None;
                    } else {
                        unsupported[AdaptiveValidatorAction::WidenResponses.index()] =
                            Some(ValidatorUnsupported::NoUnexaminedResponse);
                    }
                }
            }
        }
        if budget.role_submissions_remaining > 0 {
            if let Some((node, prefix)) = self.validator_attack_anchor(q)? {
                offers[AdaptiveValidatorAction::AttackRepair.index()] =
                    Some(ActionOffer::Attack { node, prefix });
                unsupported[AdaptiveValidatorAction::AttackRepair.index()] = None;
            }
        } else {
            unsupported[AdaptiveValidatorAction::AttackRepair.index()] =
                Some(ValidatorUnsupported::NoRemainingBudget);
        }
        if endpoints.cpu_r == CpuEndpoint::Own {
            if let Some(admission) = self
                .cpu_admissions
                .get(&target.input.question_digest)
                .filter(|a| {
                    a.frame == target.frame
                        && a.continuation_retained
                        && !a.terminal
                        && !a.quarantined
                        && self.cpu_context_current(target, a)
                })
            {
                if let (Some(job), Some(owner)) = (admission.job, admission.owner) {
                    if profiles.iter().any(|profile| {
                        profile.owner == owner
                            && profile.evaluation_namespace == job.evaluation_namespace
                            && profile.search_conditions == job.search_conditions
                    }) {
                        offers[AdaptiveValidatorAction::ResumeTask.index()] =
                            Some(ActionOffer::Resume { job: job.id, owner });
                        unsupported[AdaptiveValidatorAction::ResumeTask.index()] = None;
                    }
                }
            }
        }
        if offers[AdaptiveValidatorAction::ResumeTask.index()].is_none() {
            unsupported[AdaptiveValidatorAction::ResumeTask.index()] =
                Some(ValidatorUnsupported::NoSameOwnerPausedJob);
        }
        if self
            .validator_state
            .actions
            .values()
            .any(|action| action.frame == Some(target.frame))
        {
            offers[AdaptiveValidatorAction::ResumeTask.index()] = None;
            unsupported[AdaptiveValidatorAction::ResumeTask.index()] =
                Some(ValidatorUnsupported::AlreadyOwnedByAction);
        }
        offers[AdaptiveValidatorAction::Defer.index()] = Some(ActionOffer::Defer);
        unsupported[AdaptiveValidatorAction::Defer.index()] = None;
        let mut prepared_inputs = std::array::from_fn(|_| None);
        for (index, offer) in offers.iter().enumerate() {
            prepared_inputs[index] = match offer {
                Some(ActionOffer::Responses {
                    node,
                    prefix,
                    moves,
                }) => {
                    let mut input = self.input(
                        *node,
                        RefinementStage::CpuDiscovery,
                        q.first,
                        prefix.clone(),
                        q.proposal.clone(),
                        q.counter.clone(),
                        q.repaired.clone(),
                        q.attack_ply,
                        false,
                        None,
                        None,
                    )?;
                    Arc::get_mut(&mut input.question)
                        .ok_or(RoleError::InvalidOutput)?
                        .validator_root_moves = Some(moves.clone());
                    Some(input)
                }
                Some(ActionOffer::Attack { node, prefix }) => {
                    let repaired = q.repaired.as_ref().ok_or(RoleError::InvalidOutput)?.clone();
                    Some(self.input(
                        *node,
                        RefinementStage::CriticRecheck,
                        q.first,
                        prefix.clone(),
                        repaired.clone(),
                        q.counter.clone(),
                        Some(repaired),
                        q.attack_ply,
                        false,
                        None,
                        None,
                    )?)
                }
                _ => None,
            };
        }
        let baseline_witnesses = self.validator_existing_witnesses()?;
        let input = self.input_with_purpose(
            q.node,
            RefinementStage::ValidatorTaskRanking,
            q.first,
            q.context.prefix.clone(),
            q.proposal.clone(),
            q.counter.clone(),
            q.repaired.clone(),
            q.attack_ply,
            q.recheck,
            None,
            None,
            RoleQueryPurpose::TaskPolicy,
        )?;
        let frozen_context = self.validator_frozen_context(
            target,
            budget,
            endpoints,
            &offers,
            &prepared_inputs,
            &baseline_witnesses,
        )?;
        Ok(AdaptiveValidatorQuestion {
            frozen_context,
            input,
            target: target.clone(),
            budget,
            offers,
            prepared_inputs,
            unsupported,
            prepared_at,
            observation_watermark,
            baseline_witnesses,
        })
    }

    pub(super) fn validator_witness_identity(
        &self,
        kind: ActionWitnessKind,
        line: &[BoardMove],
    ) -> Result<ContractDigest, PalsError> {
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-action-unique-witness/1\0");
        hash.update(self.rules.snapshot().identity().semantic.0);
        hash.update([kind as u8]);
        for movement in line {
            hash.update(Move16::pack(*movement)?.bits().to_le_bytes());
        }
        Ok(ContractDigest(hash.finalize().into()))
    }

    /// Cross-root comparison identity of the actual frozen private question.
    /// Original authority/view seals remain separate provenance. This digest
    /// excludes numeric owners/handles/request IDs and absolute deadlines.
    fn validator_frozen_context(
        &self,
        target: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        budget: FrameBudget,
        endpoints: CpuEndpointSelection,
        offers: &[Option<ActionOffer>; 7],
        prepared_inputs: &[Option<FrameInput<ReviewView, AdaptivePortQuestion>>; 7],
        baseline: &HashSet<ContractDigest>,
    ) -> Result<ContractDigest, PalsError> {
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-frozen-validator-action-context/1\0");
        hash.update(self.rules.snapshot().identity().semantic.0);
        hash.update(target.original_first_move.bits().to_le_bytes());
        self.validator_write_question(&mut hash, &target.input.question)?;
        validator_write_review(&mut hash, &target.input.view)?;
        hash.update(budget.role_submissions_remaining.to_le_bytes());
        hash.update(budget.original_cpu_credit.to_le_bytes());
        hash.update(budget.original_store_byte_limit.to_le_bytes());
        hash.update(self.limits.max_rounds.to_le_bytes());
        hash.update(self.limits.max_cpu_nodes.to_le_bytes());
        hash.update(self.limits.cpu_depth.to_le_bytes());
        hash.update(self.counters.cpu_nodes.to_le_bytes());
        hash.update(self.counters.role_calls.to_le_bytes());
        hash.update([
            endpoints.execution_use as u8,
            endpoints.cpu_t as u8,
            endpoints.cpu_r as u8,
            u8::from(endpoints.load_v),
        ]);
        hash.update((self.engine.config.line_plies as u64).to_le_bytes());
        hash.update((self.engine.config.beam_width as u64).to_le_bytes());
        hash.update(self.engine.config.cpu_nodes_per_task.to_le_bytes());
        validator_write_bytes(
            &mut hash,
            self.engine.researcher.cpu_registered_condition.as_bytes(),
        );
        for (index, offer) in offers.iter().enumerate() {
            hash.update([index as u8, u8::from(offer.is_some())]);
            match offer {
                None => {}
                Some(ActionOffer::Responses {
                    node,
                    prefix,
                    moves,
                }) => {
                    hash.update([0]);
                    hash.update(
                        rz_position::contracts::rules_snapshot_semantic_digest(
                            &self.engine.semantic.nodes[*node].position.snapshot(),
                        )
                        .0,
                    );
                    validator_write_moves(&mut hash, prefix)?;
                    validator_write_moves(&mut hash, moves)?;
                }
                Some(ActionOffer::Attack { node, prefix }) => {
                    hash.update([1]);
                    hash.update(
                        rz_position::contracts::rules_snapshot_semantic_digest(
                            &self.engine.semantic.nodes[*node].position.snapshot(),
                        )
                        .0,
                    );
                    validator_write_moves(&mut hash, prefix)?;
                }
                Some(ActionOffer::Resume { job, owner }) => {
                    hash.update([2]);
                    let admission = self
                        .cpu_admissions
                        .get(&target.input.question_digest)
                        .filter(|a| {
                            a.job.map(|j| j.id) == Some(*job)
                                && a.owner == Some(*owner)
                                && self.cpu_context_current(target, a)
                        })
                        .ok_or(RoleError::InvalidOutput)?;
                    let admitted = admission.job.ok_or(RoleError::InvalidOutput)?;
                    hash.update(admitted.state_identity.semantic.0);
                    hash.update(admitted.root_restriction.0);
                    hash.update(admitted.evaluation_namespace.0);
                    hash.update(admitted.search_conditions.0);
                    hash.update(admitted.total_native_node_credit.to_le_bytes());
                    hash.update(admission.cumulative_nodes.to_le_bytes());
                    hash.update(admission.target_depth.to_le_bytes());
                    hash.update([
                        u8::from(admission.continuation_retained),
                        u8::from(admission.terminal),
                        u8::from(admission.quarantined),
                        u8::from(admission.grant.is_some()),
                    ]);
                    validator_write_optional_moves(&mut hash, admission.restriction.as_deref())?;
                }
                Some(ActionOffer::Defer) => hash.update([3]),
            }
            if let Some(input) = &prepared_inputs[index] {
                hash.update([1]);
                self.validator_write_question(&mut hash, &input.question)?;
                validator_write_review(&mut hash, &input.view)?;
            } else {
                hash.update([0]);
            }
        }
        // Utility newness depends on the full owner-issued history, including
        // superseded and unselected witnesses. HashSet iteration order is not
        // a semantic feature, so canonicalize the bounded history by bytes.
        let mut historical: Vec<_> = baseline.iter().map(|id| id.0).collect();
        historical.sort_unstable();
        hash.update((historical.len() as u64).to_le_bytes());
        for identity in historical {
            hash.update(identity);
        }
        Ok(ContractDigest(hash.finalize().into()))
    }

    fn validator_write_question(
        &self,
        hash: &mut Sha256,
        q: &AdaptivePortQuestion,
    ) -> Result<(), PalsError> {
        hash.update([q.stage as u8, q.context.purpose as u8]);
        hash.update(
            rz_position::contracts::rules_snapshot_semantic_digest(&q.position.snapshot()).0,
        );
        validator_write_optional_moves(hash, q.first.as_ref().map(std::slice::from_ref))?;
        validator_write_moves(hash, &q.context.prefix)?;
        validator_write_moves(hash, &q.proposal)?;
        validator_write_optional_moves(hash, q.counter.as_deref())?;
        validator_write_optional_moves(hash, q.repaired.as_deref())?;
        validator_write_optional_moves(hash, q.validator_root_moves.as_deref())?;
        hash.update([u8::from(q.discovery_restricted), u8::from(q.recheck)]);
        hash.update([u8::from(q.attack_ply.is_some())]);
        hash.update((q.attack_ply.unwrap_or(0) as u64).to_le_bytes());
        hash.update(q.context.focus_sha256);
        hash.update(q.context.prefix_sha256);
        hash.update(q.context.proposal_sha256);
        hash.update(q.context.divergence_sha256);
        hash.update([u8::from(q.context.refutation_sha256.is_some())]);
        if let Some(digest) = q.context.refutation_sha256 {
            hash.update(digest);
        }
        if let Some(view) = &q.frozen_counter {
            hash.update([1]);
            validator_write_review(hash, view)?;
        } else {
            hash.update([0]);
        }
        for observation in [q.counter_source, q.paired_repaired] {
            hash.update([u8::from(observation.is_some())]);
            if let Some(id) = observation {
                self.validator_write_observation(hash, id, 0)?;
            }
        }
        if let Some(seal) = q.paired_fresh_seal {
            hash.update([1]);
            hash.update(seal.root_state.semantic.0);
            hash.update(seal.context_policy.0);
            hash.update(seal.model.0);
            hash.update(seal.encoding.0);
            hash.update(seal.frozen_epoch.to_le_bytes());
            hash.update([
                seal.role as u8,
                u8::from(seal.perspective == rz_contracts::Color::Black),
            ]);
            validator_write_bytes(hash, format!("{:?}", seal.precision).as_bytes());
        } else {
            hash.update([0]);
        }
        Ok(())
    }

    /// Follow only already-resident dependencies, with a bounded recursion.
    /// Missing/cold evidence is an explicit error; the manager does no I/O.
    pub(super) fn validator_write_observation(
        &self,
        hash: &mut Sha256,
        id: ObservationId,
        depth: u8,
    ) -> Result<(), PalsError> {
        self.validator_write_observation_into(&mut ObservationShaSink(hash), id, depth)
    }
    pub(super) fn validator_write_observation_into<S: ObservationSink>(
        &self,
        hash: &mut S,
        id: ObservationId,
        depth: u8,
    ) -> Result<(), PalsError> {
        hash.check()?;
        if depth >= 8 {
            return Err(PalsError::Capacity);
        }
        let obs = self.engine.semantic.stores.observations.get(id)?;
        // Legacy witness bytes stay unchanged. This new domain includes both
        // the submitted single-limit meaning and the actual issued clamp;
        // neither may be reconstructed by relabeling an existing TaskKey.
        if let EvidenceScope::ProfiledExternalUci {
            requested,
            issued,
            reported_depth,
            seldepth,
            bound,
        } = obs.scope
        {
            hash.write(b"rz-validator-profiled-external-observation/1\0");
            observation_writer::write_profiled_constraint(hash, requested);
            observation_writer::write_profiled_constraint(hash, issued);
            for depth in [reported_depth, seldepth] {
                hash.write([u8::from(depth.is_some())]);
                if let Some(depth) = depth {
                    hash.write(depth.to_le_bytes());
                }
            }
            hash.write([bound as u8]);
        }
        hash.write([obs.kind as u8]);
        hash.write(
            rz_position::contracts::rules_snapshot_semantic_digest(
                self.engine.semantic.stores.states.get(obs.state)?,
            )
            .0,
        );
        for line in [obs.line, obs.cpu_pv] {
            hash.write([u8::from(line.is_some())]);
            if let Some(line) = line {
                observation_writer::write_moves(
                    hash,
                    &self.engine.semantic.stores.lines.moves(line)?,
                )?;
            }
        }
        hash.write(obs.budget.to_le_bytes());
        observation_writer::write_bytes(
            hash,
            obs.cpu_condition.as_deref().unwrap_or("").as_bytes(),
        );
        if let Some(identity) = &obs.value_identity {
            hash.write([1]);
            observation_writer::write_bytes(hash, identity.semantics.as_bytes());
            observation_writer::write_bytes(
                hash,
                identity.weights_sha256.as_deref().unwrap_or("").as_bytes(),
            );
        } else {
            hash.write([0]);
        }
        if let Some(identity) = &obs.model_value_identity {
            hash.write([1]);
            for field in [
                &identity.semantics,
                &identity.model,
                &identity.encoding,
                &identity.precision,
            ] {
                observation_writer::write_bytes(hash, field.as_bytes());
            }
            hash.write(identity.model_epoch);
        } else {
            hash.write([0]);
        }
        match obs.score {
            RawScore::Unknown => hash.write([0]),
            RawScore::Estimate { value, perspective } => {
                hash.write([1, u8::from(perspective == Color::Black)]);
                hash.write(value.to_bits().to_le_bytes());
            }
            RawScore::Cpu {
                value,
                perspective,
                bound,
            } => {
                hash.write([2, u8::from(perspective == Color::Black), bound as u8]);
                hash.write(value.to_le_bytes());
            }
            RawScore::ExternalUci {
                value,
                bound,
                perspective,
                wdl_per_mille,
            } => {
                hash.write([3, bound as u8, u8::from(perspective == Color::Black)]);
                match value {
                    crate::cpu_checker::ExternalRawScore::Centipawns(value) => {
                        hash.write([0]);
                        hash.write(value.to_le_bytes());
                    }
                    crate::cpu_checker::ExternalRawScore::MateMoves(value) => {
                        hash.write([1]);
                        hash.write(value.to_le_bytes());
                    }
                    crate::cpu_checker::ExternalRawScore::Unknown => hash.write([2]),
                }
                hash.write([u8::from(wdl_per_mille.is_some())]);
                if let Some(wdl) = wdl_per_mille {
                    for value in wdl {
                        hash.write(value.to_le_bytes());
                    }
                }
            }
            RawScore::Wdl {
                win,
                draw,
                loss,
                perspective,
            }
            | RawScore::ContextWdl {
                win,
                draw,
                loss,
                perspective,
                ..
            } => {
                hash.write([4, u8::from(perspective == Color::Black)]);
                for value in [win, draw, loss] {
                    hash.write(value.to_bits().to_le_bytes());
                }
            }
            RawScore::ConditionalWdl {
                expectation,
                perspective,
                repaired,
                counter,
                ..
            } => {
                hash.write([5, u8::from(perspective == Color::Black)]);
                hash.write(expectation.to_bits().to_le_bytes());
                self.validator_write_observation_into(hash, repaired, depth + 1)?;
                self.validator_write_observation_into(hash, counter, depth + 1)?;
            }
            RawScore::ConditionalRepairWdl {
                expectation,
                perspective,
                before,
                after,
                ..
            } => {
                hash.write([6, u8::from(perspective == Color::Black)]);
                hash.write(expectation.to_bits().to_le_bytes());
                self.validator_write_observation_into(hash, before, depth + 1)?;
                self.validator_write_observation_into(hash, after, depth + 1)?;
            }
            RawScore::Terminal { winner } => hash.write([
                7,
                match winner {
                    None => 0,
                    Some(Color::White) => 1,
                    Some(Color::Black) => 2,
                },
            ]),
        }
        hash.check()
    }
    pub(super) fn validator_existing_witnesses(
        &self,
    ) -> Result<HashSet<ContractDigest>, PalsError> {
        let mut existing = HashSet::new();
        // Keep the bounded owner's publication history, including superseded
        // counters. Rediscovering a formerly supported line is not a new
        // witness merely because a later Repair changed its active status.
        for (kind, moves, _) in self.conditional_witnesses.values() {
            // This private owner registry was populated only after the hot
            // commit. Its checked line survives archival; baseline selection
            // never tries to load a now-cold observation on the manager.
            let kind = match kind {
                ObservationKind::Refutation => ActionWitnessKind::Counter,
                ObservationKind::Repair => ActionWitnessKind::Repair,
                _ => return Err(RoleError::InvalidOutput.into()),
            };
            existing.insert(self.validator_witness_identity(kind, moves)?);
        }
        Ok(existing)
    }

    /// Only an actual owner's accepted conditional publication has utility
    /// authority. A raw Estimate/CP/terminal label cannot mint this proof.
    pub(super) fn validator_committed_witness_kind(
        &self,
        id: ObservationId,
    ) -> Result<Option<ActionWitnessKind>, PalsError> {
        let observation = self.engine.semantic.stores.observations.get(id)?;
        let Some(line) = observation.line else {
            return Ok(None);
        };
        let kind = match (observation.kind, self.conditional_witness_kind(id)) {
            (ObservationKind::Refutation, Some(ObservationKind::Refutation)) => {
                ActionWitnessKind::Counter
            }
            (ObservationKind::Repair, Some(ObservationKind::Repair)) => ActionWitnessKind::Repair,
            _ => return Ok(None),
        };
        let root = self
            .engine
            .semantic
            .stores
            .situations
            .get(self.engine.semantic.nodes[self.root].situation)?;
        let Some(conclusion) = root.conclusions.get(line) else {
            return Ok(None);
        };
        let expected_status = match kind {
            ActionWitnessKind::Counter => conclusion.status == ContinuationStatus::Refuted,
            ActionWitnessKind::Repair => conclusion.status == ContinuationStatus::Supported,
        };
        if !expected_status
            || conclusion.evidence != Some(id)
            || observation.state != self.engine.semantic.nodes[self.root].state
        {
            return Ok(None);
        }
        Ok(Some(kind))
    }

    fn validator_attack_anchor(
        &mut self,
        q: &AdaptivePortQuestion,
    ) -> Result<Option<(usize, Vec<BoardMove>)>, PalsError> {
        let (Some(repaired), Some(counter), Some(attack)) = (&q.repaired, &q.counter, q.attack_ply)
        else {
            return Ok(None);
        };
        if repaired.first() != counter.first() || repaired.first().copied() != q.first {
            return Ok(None);
        }
        let root_color = self.engine.semantic.nodes[self.root]
            .position
            .side_to_move();
        let mut position = self.engine.semantic.nodes[self.root].position.clone();
        let mut changed = false;
        for (ply, movement) in repaired.iter().enumerate() {
            if ply > attack
                && position.side_to_move() == root_color
                && counter.get(ply) != Some(movement)
            {
                changed = true;
            }
            if changed
                && position.side_to_move() != root_color
                && matches!(
                    position.classify_position()?.play_status,
                    PlayStatus::Ongoing
                )
            {
                let prefix = repaired[..ply].to_vec();
                let node = self.replay_node(&prefix)?;
                return Ok(Some((node, prefix)));
            }
            position.make_move(*movement)?;
        }
        Ok(None)
    }

    /// Rank only actually supported actions. Equal logits retain vocabulary
    /// order, and Defer never acquires a synthetic positive training utility.
    pub fn select_validator_action(
        &mut self,
        question: &AdaptiveValidatorQuestion,
        accepted: &AcceptedValidatorPublished,
    ) -> Result<AdaptiveValidatorActionPlan, PalsError> {
        let action = AdaptiveValidatorAction::ALL
            .into_iter()
            .filter(|action| question.offers[action.index()].is_some())
            .max_by(|a, b| {
                accepted.task_logits()[a.index()]
                    .total_cmp(&accepted.task_logits()[b.index()])
                    .then_with(|| b.index().cmp(&a.index()))
            })
            .ok_or(RoleError::Unsupported("no eligible V action"))?;
        self.plan_validator_action_with_selection(
            question,
            accepted,
            action,
            ValidatorActionSelection::ValidatorArgmax,
        )
    }

    /// The explicit action-outcomes collector may run each supported action
    /// once from this frozen input. All plans keep original deadline/budget;
    /// the manager's existing credit ledger decides actual admission.
    pub fn plan_validator_action(
        &mut self,
        question: &AdaptiveValidatorQuestion,
        accepted: &AcceptedValidatorPublished,
        action: AdaptiveValidatorAction,
    ) -> Result<AdaptiveValidatorActionPlan, PalsError> {
        self.plan_validator_action_with_selection(
            question,
            accepted,
            action,
            ValidatorActionSelection::RegisteredSupportedEnumerationV1,
        )
    }

    fn plan_validator_action_with_selection(
        &mut self,
        question: &AdaptiveValidatorQuestion,
        accepted: &AcceptedValidatorPublished,
        action: AdaptiveValidatorAction,
        selection: ValidatorActionSelection,
    ) -> Result<AdaptiveValidatorActionPlan, PalsError> {
        self.check_live()?;
        self.check_work(&question.target)?;
        let ticket = accepted.ticket();
        if ticket.authority != self.authority
            || ticket.role != Role::Validator
            || ticket.question != question.input.question_digest
            || ticket.original_deadline != question.target.original_deadline
            || ticket.view != question.input.view_seal
            || accepted.view().digest() != question.input.view.digest()
            || !Arc::ptr_eq(accepted.view(), &question.input.view)
            || !accepted.view().validates(&self.engine.semantic.stores)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let offer = question.offers[action.index()]
            .clone()
            .ok_or(RoleError::Unsupported(
                "V action is not supported by this question",
            ))?;
        if self.validator_state.actions.len() >= MAX_VALIDATOR_ACTIONS_PENDING {
            return Err(PalsError::Capacity);
        }
        let bit = 1u8 << action.index();
        if let Some((digest, used)) = self.validator_state.contexts.get(&ticket.id) {
            if *digest != question.input.question_digest || used & bit != 0 {
                return Err(RoleError::InvalidOutput.into());
            }
        } else if self.validator_state.contexts.len() >= MAX_VALIDATOR_CONTEXTS_PER_ROOT {
            return Err(PalsError::Capacity);
        }
        let first_consumption = !self.validator_state.contexts.contains_key(&ticket.id);
        if first_consumption
            && (self.accepted.contains(&ticket.id)
                || self.counters.role_calls >= self.engine.config.max_role_calls)
        {
            return Err(PalsError::RoleCallLimit);
        }
        let action_budget = FrameBudget {
            role_submissions_remaining: question
                .budget
                .role_submissions_remaining
                .min(MAX_ACTION_ROLE_SUBMISSIONS),
            ..question.budget
        };
        let effect = match offer {
            ActionOffer::Responses { node, moves, .. } => {
                let actual_legal = self.engine.semantic.nodes[node].position.legal_moves();
                if moves.is_empty() || moves.iter().any(|mv| !actual_legal.contains(mv)) {
                    return Err(RoleError::InvalidOutput.into());
                }
                let input = question.prepared_inputs[action.index()]
                    .clone()
                    .ok_or(RoleError::InvalidOutput)?;
                if input.question.validator_root_moves.as_ref() != Some(&moves)
                    || !input.view.validates(&self.engine.semantic.stores)
                {
                    return Err(RoleError::InvalidOutput.into());
                }
                AdaptiveValidatorEffect::Frame {
                    input,
                    budget: action_budget,
                }
            }
            ActionOffer::Attack { .. } => {
                let input = question.prepared_inputs[action.index()]
                    .clone()
                    .ok_or(RoleError::InvalidOutput)?;
                if input.stage != RefinementStage::CriticRecheck
                    || !input.view.validates(&self.engine.semantic.stores)
                {
                    return Err(RoleError::InvalidOutput.into());
                }
                AdaptiveValidatorEffect::Frame {
                    input,
                    budget: action_budget,
                }
            }
            ActionOffer::Resume { job, owner } => {
                let a = self
                    .cpu_admissions
                    .get(&question.target.input.question_digest)
                    .ok_or(RoleError::InvalidOutput)?;
                if !a.continuation_retained
                    || a.terminal
                    || a.quarantined
                    || a.job.map(|j| j.id) != Some(job)
                    || a.owner != Some(owner)
                    || !self.cpu_context_current(&question.target, a)
                {
                    return Err(RoleError::InvalidOutput.into());
                }
                AdaptiveValidatorEffect::Resume {
                    frame: question.parent(),
                    job,
                    owner,
                }
            }
            ActionOffer::Defer => AdaptiveValidatorEffect::Defer,
        };
        let id = ValidatorActionId {
            authority: self.authority,
            validator_request: ticket.id,
            action,
        };
        if first_consumption {
            self.accepted.insert(ticket.id);
            self.counters.role_calls += 1;
            self.counters.consumed_role_outputs += 1;
        }
        self.validator_state
            .contexts
            .entry(ticket.id)
            .or_insert((question.input.question_digest, 0))
            .1 |= bit;
        self.validator_state.actions.insert(
            id,
            ActionTracking {
                source: *ticket,
                frozen_context: question.frozen_context,
                selection,
                task_logits: *accepted.task_logits(),
                source_output: accepted.output_digest(),
                source_public_context: accepted.source().public_context,
                source_consumption: accepted.source().consumption_digest,
                input: ContractDigest(question.input.view.digest()),
                parent_question: question.target.input.question_digest,
                parent_view: question.target.input.view_seal,
                parent_input: ContractDigest(question.target.input.view.digest()),
                parent_public_context: question.target.input.public_context,
                parent: question.parent(),
                first: question.original_first_move(),
                support: question.support_mask(),
                input_observation_sequence: question.observation_watermark,
                started: question.prepared_at,
                frame: None,
                executions: BTreeSet::new(),
                role_requests: HashSet::new(),
                witnessed: question.baseline_witnesses.clone(),
                witnesses: Vec::new(),
            },
        );
        Ok(AdaptiveValidatorActionPlan {
            id,
            frozen_context: question.frozen_context,
            selection,
            task_logits: *accepted.task_logits(),
            source: *ticket,
            source_input: ContractDigest(question.input.view.digest()),
            source_output: accepted.output_digest(),
            source_public_context: accepted.source().public_context,
            source_consumption: accepted.source().consumption_digest,
            parent_question: question.target.input.question_digest,
            parent_view: question.target.input.view_seal,
            parent_input: ContractDigest(question.target.input.view.digest()),
            parent_public_context: question.target.input.public_context,
            parent: question.parent(),
            first: question.original_first_move(),
            deadline: question.original_deadline(),
            support: question.support_mask(),
            effect,
        })
    }

    pub fn bind_validator_action(
        &mut self,
        id: ValidatorActionId,
        frame: FrameId,
    ) -> Result<(), PalsError> {
        self.check_live()?;
        let tracking = self
            .validator_state
            .actions
            .get_mut(&id)
            .ok_or(RoleError::InvalidOutput)?;
        if frame.authority != self.authority
            || frame.sequence == 0
            || tracking.frame.is_some()
            || (id.action == AdaptiveValidatorAction::ResumeTask && frame != tracking.parent)
            || id.action == AdaptiveValidatorAction::Defer
        {
            return Err(RoleError::InvalidOutput.into());
        }
        tracking.frame = Some(frame);
        Ok(())
    }

    pub fn begin_validator_action_commit(
        &self,
        id: ValidatorActionId,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        execution: StepExecution,
    ) -> Result<ActionCommitGuard, PalsError> {
        self.check_work(work)?;
        let tracking = self
            .validator_state
            .actions
            .get(&id)
            .ok_or(RoleError::InvalidOutput)?;
        let same_authority = match execution {
            StepExecution::Role(request) => {
                request.epoch == self.authority.epoch && !self.accepted.contains(&request)
            }
            StepExecution::Cpu { job, execution } => {
                job.authority == self.authority
                    && execution.epoch == self.authority.epoch
                    && self
                        .cpu_admissions
                        .get(&work.input.question_digest)
                        .is_some_and(|actual| {
                            !actual.quarantined
                                && actual.job.map(|j| j.id) == Some(job)
                                && actual.grant.map(|grant| grant.execution) == Some(execution)
                        })
            }
            StepExecution::ExternalCpu { job, execution } => {
                job.authority == self.authority
                    && execution.epoch == self.authority.epoch
                    && !self.external_action_committed(work.input.question_digest, job, execution)
            }
            StepExecution::ProfiledExternal { .. } => {
                return Err(
                    RoleError::Unsupported("profiled R has no legacy V action witness").into(),
                );
            }
        };
        if tracking.frame != Some(work.frame)
            || tracking.first != work.original_first_move
            || !same_authority
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(ActionCommitGuard {
            action: id,
            frame: work.frame,
            question: work.input.question_digest,
            execution,
            first_observation: self
                .engine
                .semantic
                .stores
                .observations
                .next_observation_sequence(),
            generation: self.engine.semantic.stores.generation(),
        })
    }

    pub fn finish_validator_action_commit(
        &mut self,
        guard: ActionCommitGuard,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), PalsError> {
        self.check_live()?;
        if guard.frame != work.frame
            || guard.question != work.input.question_digest
            || guard.generation != self.engine.semantic.stores.generation()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let end = self
            .engine
            .semantic
            .stores
            .observations
            .next_observation_sequence();
        // Actual semantic consumption populated the owner's exact commit
        // receipt. A different frame's accepted ID or public StepEvidence
        // cannot attach that frame's publications to this action.
        match guard.execution {
            StepExecution::Role(id)
                if self.role_action_committed(guard.question, id, guard.first_observation, end) => {
            }
            StepExecution::Cpu { job, execution } => {
                if !self.cpu_action_committed(guard.question, job, execution) {
                    return Err(RoleError::InvalidOutput.into());
                }
            }
            StepExecution::ExternalCpu { job, execution } => {
                if !self.external_action_committed(guard.question, job, execution) {
                    return Err(RoleError::InvalidOutput.into());
                }
            }
            _ => return Err(RoleError::InvalidOutput.into()),
        }
        if end < guard.first_observation || end - guard.first_observation > MAX_ACTION_WITNESSES {
            return Err(PalsError::Capacity);
        }
        let mut witnesses = Vec::new();
        for sequence in guard.first_observation..end {
            let observation_id = ObservationId(sequence);
            let observation = self
                .engine
                .semantic
                .stores
                .observations
                .get(observation_id)?;
            if observation.line.is_none() {
                continue;
            }
            let Some(kind) = self.validator_committed_witness_kind(observation_id)? else {
                continue;
            };
            let actual_line = self
                .conditional_witness_line(observation_id)
                .ok_or(RoleError::InvalidOutput)?
                .to_vec();
            if actual_line.is_empty()
                || Some(Move16::pack(actual_line[0])?.bits())
                    != Some(work.original_first_move.bits())
            {
                return Err(RoleError::InvalidOutput.into());
            }
            let mut replay = self.engine.semantic.nodes[self.root].position.clone();
            for movement in &actual_line {
                replay.make_move(*movement)?;
            }
            let identity = self.validator_witness_identity(kind, &actual_line)?;
            let mut raw = Sha256::new();
            raw.update(b"rz-pals-action-store-commit/1\0");
            raw.update(guard.question.0);
            raw.update(guard.action.authority.epoch.0.to_le_bytes());
            raw.update(guard.action.authority.game.0.to_le_bytes());
            raw.update(guard.action.authority.root.0.to_le_bytes());
            raw.update(guard.action.authority.implementation.0);
            raw.update(guard.action.validator_request.sequence.to_le_bytes());
            raw.update([guard.action.action.index() as u8]);
            raw.update(guard.frame.sequence.to_le_bytes());
            match guard.execution {
                StepExecution::Role(id) => {
                    raw.update([0]);
                    raw.update(id.sequence.to_le_bytes());
                }
                StepExecution::Cpu { job, execution } => {
                    raw.update([1]);
                    raw.update(job.sequence.to_le_bytes());
                    raw.update(execution.sequence.to_le_bytes());
                }
                StepExecution::ExternalCpu { job, execution } => {
                    raw.update([2]);
                    raw.update(job.sequence.to_le_bytes());
                    raw.update(execution.sequence.to_le_bytes());
                }
                StepExecution::ProfiledExternal { .. } => {
                    return Err(RoleError::Unsupported(
                        "profiled R has no legacy V action witness",
                    )
                    .into());
                }
            }
            raw.update(identity.0);
            raw.update((sequence as u64).to_le_bytes());
            raw.update(guard.generation.to_le_bytes());
            witnesses.push(ActionWitnessCommit {
                action: guard.action,
                kind,
                observation: observation_id,
                generation: guard.generation,
                actual_line,
                identity,
                raw_evidence: ContractDigest(raw.finalize().into()),
            });
        }
        // Validate the entire bounded commit before mutating dedupe/history.
        // Capacity failure cannot consume an execution and half its witnesses.
        let tracking = self
            .validator_state
            .actions
            .get(&guard.action)
            .ok_or(RoleError::InvalidOutput)?;
        let repeated = match guard.execution {
            StepExecution::Role(id) => tracking.role_requests.contains(&id),
            StepExecution::Cpu { execution, .. } | StepExecution::ExternalCpu { execution, .. } => {
                tracking.executions.contains(&execution)
            }
            StepExecution::ProfiledExternal { .. } => {
                return Err(
                    RoleError::Unsupported("profiled R has no legacy V action witness").into(),
                );
            }
        };
        let new_count = witnesses
            .iter()
            .map(|w| w.identity)
            .filter(|id| !tracking.witnessed.contains(id))
            .collect::<HashSet<_>>()
            .len();
        if repeated {
            return Err(RoleError::InvalidOutput.into());
        }
        if tracking
            .witnesses
            .len()
            .checked_add(new_count)
            .is_none_or(|n| n > MAX_ACTION_WITNESSES)
            || tracking.executions.len() + tracking.role_requests.len() >= MAX_ACTION_EXECUTIONS
            || witnesses
                .iter()
                .any(|w| w.observation.0 < tracking.input_observation_sequence)
        {
            return Err(PalsError::Capacity);
        }
        let tracking = self
            .validator_state
            .actions
            .get_mut(&guard.action)
            .ok_or(RoleError::InvalidOutput)?;
        match guard.execution {
            StepExecution::Role(id) => {
                if !tracking.role_requests.insert(id) {
                    return Err(RoleError::InvalidOutput.into());
                }
            }
            StepExecution::Cpu { execution, .. } | StepExecution::ExternalCpu { execution, .. } => {
                if !tracking.executions.insert(execution) {
                    return Err(RoleError::InvalidOutput.into());
                }
            }
            StepExecution::ProfiledExternal { .. } => {
                return Err(
                    RoleError::Unsupported("profiled R has no legacy V action witness").into(),
                );
            }
        }
        for witness in witnesses {
            if tracking.witnessed.insert(witness.identity) {
                tracking.witnesses.push(witness);
            }
        }
        Ok(())
    }

    /// Semantic outcome only. The manager still owns the independent physical
    /// fence/drain proof before an arena adapter may serialize `completed`.
    pub fn finish_validator_action_semantics(
        &mut self,
        id: ValidatorActionId,
        closed: Option<&ClosedFrame>,
    ) -> Result<ValidatorActionOutcome, PalsError> {
        self.check_live()?;
        let tracking = self
            .validator_state
            .actions
            .get(&id)
            .ok_or(RoleError::InvalidOutput)?;
        if id.action == AdaptiveValidatorAction::Defer {
            if closed.is_some() || tracking.frame.is_some() {
                return Err(RoleError::InvalidOutput.into());
            }
        } else if closed.is_none_or(|closed| {
            Some(closed.id) != tracking.frame
                || closed.original_first_move != tracking.first
                || closed.unknown.is_some()
        }) || (tracking.executions.is_empty() && tracking.role_requests.is_empty())
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let tracking = self
            .validator_state
            .actions
            .remove(&id)
            .ok_or(RoleError::InvalidOutput)?;
        Ok(ValidatorActionOutcome {
            id,
            frozen_context: tracking.frozen_context,
            selection: tracking.selection,
            task_logits: tracking.task_logits,
            source_output: tracking.source_output,
            source_public_context: tracking.source_public_context,
            source_consumption: tracking.source_consumption,
            source: tracking.source,
            input: tracking.input,
            parent_question: tracking.parent_question,
            parent_view: tracking.parent_view,
            parent_input: tracking.parent_input,
            parent_public_context: tracking.parent_public_context,
            first: tracking.first,
            support: tracking.support,
            input_observation_sequence: tracking.input_observation_sequence,
            wall_cost: tracking.started.elapsed(),
            witnesses: tracking.witnesses,
            executions: tracking.executions.into_iter().collect(),
            role_requests: tracking.role_requests.into_iter().collect(),
            started: tracking.started,
        })
    }
}
