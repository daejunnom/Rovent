//! Selected-view P/C semantic owner, structurally independent of native R.
//!
//! The role-only manager performs actual ticket/native acceptance. This port
//! only prepares exact immutable questions and consumes their linear accepted
//! heads. No checker, CPU worker, TT, native credit ledger or CPU-stage success
//! is constructed to make the P/C path resemble the registered-R assembly.

use super::adaptive_port::{AdaptivePortRoleEvidence, AdaptiveReviewLeaseAdmission};
use super::*;
use crate::pals::coordinator::{
    AcceptedRoleOrigin, AcceptedRoleStep, CoordinatorWork, FrameId, FrameInput, RefinementStage,
};
use crate::pals::rovent_engine::RootMoveDomain;
use crate::pals::rovent_frontier::FrontierFrameKind;
use rz_contracts::pals::{Role, RoleOutput, RolePayload, SearchAuthority};
use rz_contracts::pals_adaptive::{ReviewViewSeal, RoleTicketEnvelope};
use rz_contracts::rovent::ExecutionMode;
use rz_contracts::{Deadline, Digest as ContractDigest};
use rz_position::contracts::{ContractPosition, ContractState};
use std::collections::{HashMap, VecDeque};

mod attack_source;
mod frontier_port;
mod goal_local;
mod peer_questions;
mod repair_cycle;
mod repair_report;
mod repair_witness;
mod shared_nnuec;
pub use attack_source::PcAttackRankedSource;
pub use frontier_port::PcFrontierStartup;
use peer_questions::{PcPeerQuestion, PcPeerSource};
pub use repair_cycle::PcRepairPhase;
use repair_cycle::{PcRepairCycles, PcRepairStep, PublishedPcLine};
pub use repair_report::{PcRepairProgress, PcRepairReport};
pub use repair_witness::{
    PcAcceptedFreshEndpoint, PcDivergenceResolution, PcDivergenceSource, PcDivergenceWitness,
    PcRepairAcceptedLine,
};
use shared_nnuec::SharedNnuecState;
pub use shared_nnuec::{
    PcNnuecReadoutError, PcNnuecReadoutObservation, PcNnuecReadoutRequest, PcNnuecReadoutSeal,
    PcNnuecReadoutTerminal, PcSharedReadoutRootClosure,
};

const MAX_PC_PENDING: usize = 64;
const MAX_PC_CONSUMED: usize = 128;

/// Immutable submitted input. Only this source owner can construct one.
#[derive(Clone)]
pub struct PcPortQuestion {
    position: Position,
    context: RoleLogicalContext,
    deadline: Instant,
    node: usize,
    first: Option<BoardMove>,
    frame: u64,
    continuation_end: usize,
    responsibilities: Vec<PcResponsibilitySource>,
    goal_local: Option<Arc<goal_local::PcGoalLocalRules>>,
    peer: Option<Arc<PcPeerQuestion>>,
    repair: Option<Arc<PcRepairStep>>,
    source: Arc<PcQuestionSource>,
}

struct PcQuestionSource {
    owner: Arc<()>,
    authority: SearchAuthority,
    original_deadline: Deadline,
    sequence: u64,
    frame: u64,
    continuation_end: usize,
    responsibilities: ContractDigest,
    goal_local: Option<ContractDigest>,
    peer: Option<rz_contracts::rovent::QuestionEnvelope>,
    repair: Option<ContractDigest>,
    root_budget: Option<ContractDigest>,
    mode: ExecutionMode,
    stage: RefinementStage,
    first: Option<BoardMove>,
    position: PositionSnapshot,
    logical: ContractDigest,
    view: ReviewViewSeal,
    question: ContractDigest,
}

/// A frozen actual accepted action plus its checked child. Only the P/C
/// semantic owner can construct it; public stage/kind metadata grants nothing.
/// It retains no selected-view pin, so a long continuation does not pin every
/// historical tensor while preserving its opaque acceptance provenance.
#[derive(Clone)]
pub(crate) struct PcResponsibilitySource {
    kind: FrontierFrameKind,
    origin: AcceptedRoleOrigin,
    source: Arc<PcQuestionSource>,
    purpose: RoleQueryPurpose,
    movement: BoardMove,
    parent: PositionSnapshot,
    snapshot: PositionSnapshot,
    prefix: Vec<BoardMove>,
}
impl PcResponsibilitySource {
    pub(crate) fn kind(&self) -> FrontierFrameKind {
        self.kind
    }
    pub(crate) fn question_digest(&self) -> ContractDigest {
        self.origin.ticket().question
    }
    pub(crate) fn prefix(&self) -> &[BoardMove] {
        &self.prefix
    }
    pub(crate) fn snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }
    fn from_selection(
        kind: FrontierFrameKind,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        accepted: &AcceptedRoleStep,
        movement: BoardMove,
        child: &Position,
        prefix: Vec<BoardMove>,
    ) -> Result<Self, PalsError> {
        let query = &input.question;
        let origin = accepted
            .origin(query.frontier_frame(), query.reciprocal_question())
            .map_err(|_| RoleError::InvalidOutput)?;
        let source = Self {
            kind,
            origin,
            source: Arc::clone(&query.source),
            purpose: query.context.purpose,
            movement,
            parent: query.position.snapshot(),
            snapshot: child.snapshot(),
            prefix,
        };
        source.validate(&query.source.owner, query.source.authority, &source.prefix)?;
        Ok(source)
    }
    fn validate(
        &self,
        owner: &Arc<()>,
        authority: SearchAuthority,
        descendant: &[BoardMove],
    ) -> Result<(), PalsError> {
        let valid_kind = matches!(
            (self.kind, self.purpose),
            (
                FrontierFrameKind::Candidate,
                RoleQueryPurpose::ProposePolicy
            ) | (FrontierFrameKind::Counter, RoleQueryPurpose::ReplyPolicy)
                | (FrontierFrameKind::Repair, RoleQueryPurpose::RepairPolicy)
        );
        let ticket = self.origin.ticket();
        if !valid_kind
            || !Arc::ptr_eq(&self.source.owner, owner)
            || self.source.authority != authority
            || ticket.authority != authority
            || ticket.question != self.source.question
            || ticket.view != self.source.view
            || ticket.original_deadline != self.source.original_deadline
            || !self.source.position.same_state(&self.parent)
            || self.prefix.is_empty()
            || !descendant.starts_with(&self.prefix)
            || self.prefix.last() != Some(&self.movement)
            || (self.kind == FrontierFrameKind::Candidate && self.prefix.len() != 1)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let expected_role = if self.kind == FrontierFrameKind::Counter {
            Role::Critic
        } else {
            Role::Proposer
        };
        if ticket.role != expected_role {
            return Err(RoleError::InvalidOutput.into());
        }
        // The private constructor already received this exact quota-paid
        // checked child. Revalidating immutable custody performs no Rules work.
        Ok(())
    }
}
fn responsibility_digest(sources: &[PcResponsibilitySource]) -> Result<ContractDigest, PalsError> {
    let mut hash = Sha256::new();
    hash.update(b"rz-rovent-pc-accepted-responsibilities/1\0");
    hash.update((sources.len() as u64).to_le_bytes());
    for source in sources {
        hash.update([source.kind as u8]);
        hash.update(source.origin.seal_digest().0);
        hash.update(role_line_sha256(b"responsibility-prefix", &source.prefix)?);
    }
    Ok(ContractDigest(hash.finalize().into()))
}

impl PcQuestionSource {
    fn owned_host_bytes(&self) -> Option<u64> {
        u64::try_from(
            std::mem::size_of::<Self>().checked_add(self.position.retained_history_bytes()?)?,
        )
        .ok()
    }
}

impl PcPortQuestion {
    pub fn position(&self) -> &Position {
        &self.position
    }
    pub fn context(&self) -> &RoleLogicalContext {
        &self.context
    }
    pub fn deadline(&self) -> Instant {
        self.deadline
    }
    pub fn reciprocal_question(&self) -> Option<rz_contracts::rovent::QuestionEnvelope> {
        self.peer.as_ref().map(|peer| peer.envelope())
    }
    pub fn goal_local_origin_digest(&self) -> Option<ContractDigest> {
        self.goal_local
            .as_ref()
            .map(|origin| origin.origin_digest())
    }
    pub(crate) fn validated_horizon_source<'a>(
        input: &'a FrameInput<ReviewView, Self>,
        stores: &PalsStores,
    ) -> Result<&'a crate::pals::rovent_frontier::UnmetHorizonQuestion, PalsError> {
        Self::validate_input_source(input, stores)?;
        input
            .question
            .goal_local
            .as_ref()
            .and_then(|origin| origin.horizon())
            .ok_or(RoleError::InvalidOutput.into())
    }
    pub(crate) fn original_deadline(&self) -> Deadline {
        self.source.original_deadline
    }
    pub(crate) fn validate_frontier_input(
        input: &FrameInput<ReviewView, Self>,
        stores: &PalsStores,
    ) -> Result<(), PalsError> {
        Self::validate_input_source(input, stores)?;
        if input.question.frame == 0 || input.question.first.is_none() {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }
    /// Original accepted policy head for a single selected legal child. Root
    /// ranking has no first move yet, so it cannot borrow a fictitious frame
    /// responsibility merely to obtain Rules admission.
    pub(crate) fn validate_accepted_frontier_child(
        input: &FrameInput<ReviewView, Self>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        movement: BoardMove,
        stores: &PalsStores,
    ) -> Result<(), PalsError> {
        Self::validate_input_source(input, stores)?;
        let query = &input.question;
        let ticket = accepted.ticket();
        if query.frame == 0
            || query.context.purpose == RoleQueryPurpose::ValueFresh
            || ticket.authority != query.source.authority
            || ticket.question != input.question_digest
            || ticket.view != input.view_seal
            || ticket.original_deadline != query.original_deadline()
            || ticket.role != role_for_purpose(query.context.purpose)?
            || ticket.representation.record_revision != query.context.public_revision
            || !accepted.validates_output(output)
            || !input.view.query().legal.contains(&movement)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }
    pub(crate) fn frontier_responsibilities(&self) -> &[PcResponsibilitySource] {
        &self.responsibilities
    }
    pub fn frontier_frame(&self) -> FrameId {
        FrameId {
            authority: self.source.authority,
            sequence: self.frame,
        }
    }
    pub fn first_move(&self) -> Option<BoardMove> {
        self.first
    }
    pub fn owned_host_bytes(&self) -> Option<u64> {
        let bytes = std::mem::size_of::<Self>()
            .checked_add(std::mem::size_of::<PcQuestionSource>())?
            .checked_add(self.position.snapshot().retained_history_bytes()?)?
            .checked_add(
                self.context
                    .prefix
                    .capacity()
                    .checked_mul(std::mem::size_of::<BoardMove>())?,
            )?;
        let mut bytes = u64::try_from(bytes).ok()?;
        if let Some(step) = &self.repair {
            bytes = bytes.checked_add(step.owned_host_bytes()?)?;
        }
        Some(bytes)
    }

    /// The controller consumes the real source question and original view.
    /// This check grants no model execution, current-root or acceptance proof.
    pub fn validate_frontier_source(
        work: &CoordinatorWork<ReviewView, Self>,
        stores: &PalsStores,
    ) -> Result<(), PalsError> {
        Self::validate_input_source(&work.input, stores)?;
        let question = &work.input.question;
        let source = &question.source;
        let first = question.first.ok_or(RoleError::InvalidOutput)?;
        let first = rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(first).map_err(|_| RoleError::InvalidOutput)?,
        );
        if work.frame.authority != source.authority
            || work.original_deadline != source.original_deadline
            || work.original_first_move != first
            || work.frame != question.frontier_frame()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }

    fn validate_input_source(
        input: &FrameInput<ReviewView, Self>,
        stores: &PalsStores,
    ) -> Result<(), PalsError> {
        let question = &input.question;
        let source = &question.source;
        let view = &input.view;
        if question.frame != source.frame
            || question.continuation_end != source.continuation_end
            || source.goal_local != question.goal_local.as_ref().map(|origin| origin.digest())
            || source.peer != question.reciprocal_question()
            || source.repair
                != question
                    .repair
                    .as_ref()
                    .map(|step| step.digest())
                    .transpose()?
            || source.responsibilities != responsibility_digest(&question.responsibilities)?
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if let Some(peer) = &question.peer {
            peer.validate(question, view, source.mode)?;
        }
        if let Some(step) = &question.repair {
            step.validates(input, &source.owner, source.authority, source.root_budget)?;
        }
        if let Some(origin) = &question.goal_local {
            if source.root_budget != Some(origin.root_budget()) {
                return Err(RoleError::InvalidOutput.into());
            }
            origin.validates_input(input, &source.owner, source.authority, origin.root_budget())?;
        }
        for responsibility in &question.responsibilities {
            responsibility.validate(&source.owner, source.authority, &question.context.prefix)?;
        }
        let actual_view = ReviewViewSeal {
            authority: source.authority,
            view_sequence: view.sequence(),
            context_policy: ContractDigest(view.policy_digest()),
            view_digest: ContractDigest(view.seal_digest()),
            root_state: source.view.root_state,
            actual_tokens: view.token_count(),
            selected_records: view.records().len() as u16,
            anchors: view.anchors().len() as u8,
            length_bucket: view.length_bucket(),
        };
        if input.stage != source.stage
            || input.question_digest != source.question
            || input.view_seal != source.view
            || actual_view != source.view
            || input.public_context.0 != view.record_scope_digest()
            || question.first != source.first
            || (question.repair.is_none()
                && question.first != question.context.prefix.first().copied())
            || source.logical != logical_digest(&question.context)?
            || !source.position.same_state(&question.position.snapshot())
            || !view.validates(stores)
            || stores.generation() != source.authority.root.0
            || question.context.game_generation != source.authority.game.0
            || question.context.search_generation != source.authority.root.0
            || view.query().purpose != question.context.purpose
            || view.query().prefix != question.context.prefix
            || question.context.prefix_sha256 != role_line_sha256(b"prefix", &view.query().prefix)?
            || question.context.proposal_sha256
                != role_line_sha256(b"proposal", &view.query().proposal)?
            || question.context.refutation_sha256
                != view
                    .query()
                    .counterexample
                    .as_deref()
                    .map(|line| role_line_sha256(b"refutation", line))
                    .transpose()?
            || question.context.divergence_sha256
                != role_divergence_sha256(&view.query().divergences)?
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let situation = stores.situations.get(question.context.situation)?;
        if situation.state != question.context.state
            || situation.focus != question.context.focus
            || situation.revision < question.context.situation_revision
            || !stores
                .states
                .get(question.context.state)?
                .same_state(&question.position.snapshot())
            || stores.lines.get(question.context.focus)?.start_state != question.context.state
            || question.context.focus_sha256
                != role_line_sha256(b"focus", &stores.lines.moves(question.context.focus)?)?
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }
}

#[derive(Clone)]
struct PcPlan {
    node: usize,
    frame: u64,
    responsibilities: Vec<PcResponsibilitySource>,
    goal_local: Option<Arc<goal_local::PcGoalLocalRules>>,
    peer_source: Option<PcPeerSource>,
    repair: Option<Arc<PcRepairStep>>,
    continuation_end: usize,
    purpose: RoleQueryPurpose,
    prefix: Vec<BoardMove>,
    proposal: Vec<BoardMove>,
    counter: Option<Vec<BoardMove>>,
}
struct PcPending {
    input: FrameInput<ReviewView, PcPortQuestion>,
    ticket: Option<RoleTicketEnvelope>,
    failure: Option<PalsError>,
}
struct PcConsumed {
    ticket: RoleTicketEnvelope,
    evidence: ContractDigest,
    question: std::sync::Weak<PcPortQuestion>,
    view: std::sync::Weak<ReviewView>,
    semantic_committed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcPortCompletion {
    Active,
    RulesTerminal,
    SelectiveFrontier,
    Canceled,
    Deadline,
    RoleBudget,
    WorkBudget,
    Failed,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PcPortCounters {
    pub submitted_role_calls: u64,
    pub consumed_role_outputs: u64,
    pub accepted_proposer: u64,
    pub accepted_critic: u64,
    pub accepted_fresh: u64,
    pub checked_transitions: u64,
    pub completed_nn_inputs: Option<u64>,
}

pub struct PcPortResult {
    pub best_move: Option<BoardMove>,
    pub completion: PcPortCompletion,
    pub value: PalsResolvedValue,
    pub scope: PalsValueScope,
    pub counters: PcPortCounters,
    pub root_restricted: bool,
}

/// A pure semantic port. Its only researcher is the zero-sized NoResearcher.
pub struct PcSemanticPort<M: RoleModel> {
    owner: PcSemanticOwner<M>,
    identity: Arc<()>,
    root: usize,
    rules: Arc<ContractState>,
    root_moves: Arc<RootMoveDomain>,
    rules_owner_issuer: adaptive_port::AdaptiveRulesOwnerIssuer,
    root_ranking: Vec<BoardMove>,
    authority: SearchAuthority,
    deadline: Deadline,
    hard_deadline: Instant,
    mode: ExecutionMode,
    next_question: u64,
    next_frame: u64,
    ready: VecDeque<PcPlan>,
    pending: HashMap<ContractDigest, PcPending>,
    consumed: VecDeque<PcConsumed>,
    counters: PcPortCounters,
    completion: PcPortCompletion,
    frontier_state: Option<frontier_port::PcFrontierState>,
    retired_strategy: Option<crate::pals::rovent_frontier::FrontierMateStrategy>,
    repair_cycles: Option<PcRepairCycles>,
    post_repair_recheck: PostRepairRecheckPolicy,
    shared_nnuec: Option<SharedNnuecState>,
}

impl<M: RoleModel> PcSemanticPort<M> {
    /// Admission returns the exact original semantic owner on failure. The
    /// caller supplies its real Rules-owner issuer; no researcher is admitted.
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn new_registered_retaining(
        mut owner: PcSemanticOwner<M>,
        position: Position,
        authority: SearchAuthority,
        deadline: Deadline,
        hard_deadline: Instant,
        mode: ExecutionMode,
        rules_owner_issuer: adaptive_port::AdaptiveRulesOwnerIssuer,
        registration: adaptive_root::AdaptiveRootRegistration,
        searchmoves: Option<&[BoardMove]>,
    ) -> Result<Self, (PcSemanticOwner<M>, PalsError)> {
        let prepared = (|| -> Result<_, PalsError> {
            if mode.has_native_r()
                || deadline.clock.0 != authority.epoch
                || Instant::now() >= hard_deadline
            {
                return Err(PalsError::InvalidLimits);
            }
            owner.consume_adaptive_root_registration(registration, &position, authority)?;
            let rules = Arc::new(
                ContractPosition::new(
                    rules_owner_issuer().map_err(|_| RoleError::InvalidOutput)?,
                    position.clone(),
                )
                .export()
                .map_err(|_| RoleError::InvalidOutput)?,
            );
            let domain = Arc::new(
                RootMoveDomain::new(&rules, searchmoves).map_err(|_| RoleError::InvalidOutput)?,
            );
            let root = owner.intern_hot(position)?;
            for node in &mut owner.semantic.nodes {
                node.model_value = None;
                node.model_observation = None;
                node.model_value_revision = None;
            }
            owner.refresh_projection(owner.semantic.nodes[root].state);
            Ok((root, rules, domain))
        })();
        let (root, rules, root_moves) = match prepared {
            Ok(value) => value,
            Err(error) => return Err((owner, error)),
        };
        let terminal = owner.semantic.nodes[root].terminal.is_some();
        let mut ready = VecDeque::new();
        if !terminal {
            ready.push_back(PcPlan {
                node: root,
                frame: 1,
                responsibilities: Vec::new(),
                goal_local: None,
                peer_source: None,
                repair: None,
                continuation_end: 0,
                purpose: RoleQueryPurpose::ProposePolicy,
                prefix: Vec::new(),
                proposal: Vec::new(),
                counter: None,
            });
        }
        Ok(Self {
            owner,
            identity: Arc::new(()),
            root,
            rules,
            root_moves,
            rules_owner_issuer,
            root_ranking: Vec::new(),
            authority,
            deadline,
            hard_deadline,
            mode,
            next_question: 1,
            next_frame: 2,
            ready,
            pending: HashMap::new(),
            consumed: VecDeque::new(),
            frontier_state: None,
            retired_strategy: None,
            repair_cycles: None,
            post_repair_recheck: PostRepairRecheckPolicy::Disabled,
            shared_nnuec: None,
            counters: PcPortCounters {
                completed_nn_inputs: Some(0),
                ..PcPortCounters::default()
            },
            completion: if terminal {
                PcPortCompletion::RulesTerminal
            } else {
                PcPortCompletion::Active
            },
        })
    }

    pub fn researcher(&self) -> &NoResearcher {
        self.owner.researcher()
    }
    pub fn owner(&self) -> &PcSemanticOwner<M> {
        &self.owner
    }
    pub fn into_owner(mut self) -> PcSemanticOwner<M> {
        // Admission required an empty retired slot, and no mutable owner
        // accessor exists while this port is alive. Move unclaimed history
        // into that SAME owner instead of dropping it at the port boundary.
        if let Some(report) = self.detach_repair_report() {
            self.owner.semantic.retired_pc_repair = Some(report);
        }
        self.owner
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }
    pub fn hard_deadline(&self) -> Instant {
        self.hard_deadline
    }
    pub fn root_rules(&self) -> Arc<ContractState> {
        Arc::clone(&self.rules)
    }
    pub fn root_move_domain(&self) -> &Arc<RootMoveDomain> {
        &self.root_moves
    }
    pub fn root_state_id(&self) -> StateId {
        self.owner.semantic.nodes[self.root].state
    }
    pub fn model_namespace(&self) -> Result<&adaptive_port::AdaptiveModelNamespace, PalsError> {
        self.owner
            .semantic
            .model
            .registered_representation_namespace()
            .ok_or(PalsError::InvalidConfig)
    }
    pub fn is_complete(&self) -> bool {
        self.completion != PcPortCompletion::Active
            || (self.ready.is_empty() && self.pending.is_empty())
    }
    pub fn revoke(&mut self) {
        self.completion = PcPortCompletion::Canceled;
        self.ready.clear();
        if let Some(frontier) = &mut self.owner.semantic.rovent_frontier {
            frontier.cancel();
        }
    }

    /// Manager admission and semantic acceptance are distinct counters. The
    /// original registered limit and empty pending custody are checked here;
    /// only the manager can supply its settled registry admission count.
    pub fn close_role_call_budget(&mut self, admitted: u64, limit: u64) -> Result<(), PalsError> {
        if limit != self.owner.config.max_role_calls
            || admitted < limit
            || self.counters.submitted_role_calls > admitted
            || !self.pending.is_empty()
        {
            return Err(PalsError::InvalidConfig);
        }
        if !matches!(
            self.completion,
            PcPortCompletion::Active | PcPortCompletion::RoleBudget
        ) {
            return Ok(());
        }
        if Instant::now() >= self.hard_deadline {
            return self.close_original_deadline();
        }
        if let Some(frontier) = self.owner.semantic.rovent_frontier.as_deref_mut() {
            frontier.close_role_call_budget();
        }
        if self.completion == PcPortCompletion::Active {
            self.completion = PcPortCompletion::RoleBudget;
            self.ready.clear();
        }
        Ok(())
    }

    pub fn review_lease_admission(&self) -> Result<AdaptiveReviewLeaseAdmission, PalsError> {
        let stores = &self.owner.semantic.stores;
        let current = stores.review_pin_count()?;
        let capacity = stores.review_pin_capacity();
        let frontier_owner_present = self.owner.semantic.rovent_frontier.is_some();
        let frontier_replacement_reserve = usize::from(frontier_owner_present);
        Ok(AdaptiveReviewLeaseAdmission {
            current,
            capacity,
            required_new_leases: 1,
            frontier_replacement_reserve,
            frontier_owner_present,
            may_prepare: current
                .checked_add(1)
                .and_then(|n| n.checked_add(frontier_replacement_reserve))
                .is_some_and(|n| n <= capacity),
        })
    }

    /// New views are materialized only for work actually admitted now. Plans
    /// retain their exact lines under backpressure; they do not reserve pins.
    pub fn next_role(
        &mut self,
    ) -> Result<Option<FrameInput<ReviewView, PcPortQuestion>>, PalsError> {
        if self.completion != PcPortCompletion::Active {
            return Ok(None);
        }
        if Instant::now() >= self.hard_deadline {
            self.close_original_deadline()?;
            return Ok(None);
        }
        self.prepare_frontier_root_if_attached()?;
        if self.completion != PcPortCompletion::Active {
            return Ok(None);
        }
        if self.ready.is_empty() {
            return Ok(None);
        }
        if self.pending.len() >= MAX_PC_PENDING {
            return Err(PalsError::Capacity);
        }
        let lease = self.review_lease_admission()?;
        if !lease.may_prepare {
            return Err(StoreError::ReviewLeaseBackpressure {
                current: lease.current,
                capacity: lease.capacity,
                required_new: 1,
                replacement_reserve: lease.frontier_replacement_reserve,
            }
            .into());
        }
        let plan = self
            .ready
            .front()
            .cloned()
            .ok_or(RoleError::InvalidOutput)?;
        let node = plan.node;
        self.owner
            .semantic
            .nodes
            .get(node)
            .ok_or(StoreError::InvalidHandle("original P/C plan node"))?;
        if self.owner.semantic.nodes[node].terminal.is_none()
            && self.counters.submitted_role_calls.saturating_add(
                self.pending
                    .values()
                    .filter(|held| held.ticket.is_none())
                    .count() as u64,
            ) >= self.owner.config.max_role_calls
        {
            if self.pending.is_empty() {
                self.close_role_call_budget(
                    self.counters.submitted_role_calls,
                    self.owner.config.max_role_calls,
                )?;
            }
            return Ok(None);
        }
        // Materializing a frozen pair is a local recipe, not an NN input.
        // Each actual endpoint is admitted separately below on its own turn.
        if plan.repair.as_ref().is_some_and(|step| step.is_recipe()) {
            self.materialize_repair_recipe(&plan)?;
            return self.next_role();
        }
        if self.owner.semantic.nodes[node].terminal.is_none() && !self.admit_frontier_role()? {
            return Ok(None);
        }
        let legal = if plan.purpose == RoleQueryPurpose::DivergencePolicy {
            Vec::new()
        } else {
            self.owner.semantic.nodes[node].position.legal_moves()
        };
        let question = RoleQuestion {
            purpose: plan.purpose,
            prefix: &plan.prefix,
            proposal: &plan.proposal,
            refutation: plan.counter.as_deref(),
            divergences: plan.repair.as_ref().map_or(&[], |step| step.divergences()),
        };
        let context =
            if let Some(context) = plan.repair.as_ref().and_then(|step| step.frozen_context()) {
                context.clone()
            } else {
                self.owner.role_context(node, question)?
            };
        let view = if let Some(frozen) = plan.repair.as_ref().and_then(|step| step.frozen()) {
            if !frozen.validates(&self.owner.semantic.stores) {
                return Err(StoreError::StaleConsumer.into());
            }
            Arc::clone(frozen)
        } else {
            self.owner
                .prepare_review_view(node, question, &legal)?
                .ok_or(RoleError::Unavailable)?
        };
        let seal = ReviewViewSeal {
            authority: self.authority,
            view_sequence: view.sequence(),
            context_policy: ContractDigest(view.policy_digest()),
            view_digest: ContractDigest(view.seal_digest()),
            root_state: self.rules.snapshot().identity(),
            actual_tokens: view.token_count(),
            selected_records: view.records().len() as u16,
            anchors: view.anchors().len() as u8,
            length_bucket: view.length_bucket(),
        };
        let sequence = self.next_question;
        let next = sequence.checked_add(1).ok_or(PalsError::Capacity)?;
        let stage = stage_for_purpose(plan.purpose)?;
        let position = self.owner.semantic.nodes[node].position.clone();
        let first = plan
            .repair
            .as_ref()
            .map(|step| step.first())
            .or_else(|| plan.prefix.first().copied());
        let peer = self
            .prepare_peer_question(
                plan.peer_source.as_ref(),
                plan.purpose,
                &context,
                &position,
                &view,
                seal,
                sequence,
                stage,
                first,
            )?
            .map(Arc::new);
        let mut digest = question_digest(
            self.authority,
            sequence,
            &context,
            seal,
            self.owner.model_identity(),
        )?;
        if let Some(origin) = &plan.goal_local {
            let mut hash = Sha256::new();
            hash.update(b"rz-rovent-pc-goal-local-question/1\0");
            hash.update(digest.0);
            hash.update(origin.digest().0);
            digest = ContractDigest(hash.finalize().into());
        }
        if let Some(step) = &plan.repair {
            let mut hash = Sha256::new();
            hash.update(b"rz-rovent-pc-repair-invocation/1\0");
            hash.update(digest.0);
            hash.update(step.digest()?.0);
            digest = ContractDigest(hash.finalize().into());
        }
        if let Some(peer) = &peer {
            digest = peer.envelope().question_digest;
        }
        let continuation_end = if plan.continuation_end == 0 {
            self.continuation_limit()
        } else {
            plan.continuation_end
        };
        let source = Arc::new(PcQuestionSource {
            owner: Arc::clone(&self.identity),
            authority: self.authority,
            original_deadline: self.deadline,
            sequence,
            frame: plan.frame,
            continuation_end,
            responsibilities: responsibility_digest(&plan.responsibilities)?,
            goal_local: plan.goal_local.as_ref().map(|origin| origin.digest()),
            peer: peer.as_ref().map(|peer| peer.envelope()),
            repair: plan.repair.as_ref().map(|step| step.digest()).transpose()?,
            root_budget: self.original_root_budget(),
            mode: self.mode,
            stage,
            first,
            position: position.snapshot(),
            logical: logical_digest(&context)?,
            view: seal,
            question: digest,
        });
        let input = FrameInput {
            stage,
            question_digest: digest,
            question: Arc::new(PcPortQuestion {
                position,
                context,
                deadline: self.hard_deadline,
                node,
                first,
                frame: plan.frame,
                continuation_end,
                responsibilities: plan.responsibilities.clone(),
                goal_local: plan.goal_local.clone(),
                peer,
                repair: plan.repair.clone(),
                source,
            }),
            view_seal: seal,
            public_context: ContractDigest(view.record_scope_digest()),
            view,
        };
        if let Err(error) = self.observe_frontier_input(&input) {
            self.frontier_boundary_or_failure(error)?;
            return Ok(None);
        }
        if self.owner.semantic.nodes[node].terminal.is_some() {
            if let Err(error) = self.accept_frontier_endpoint(&input, None) {
                self.frontier_boundary_or_failure(error)?;
                return Ok(None);
            }
            self.next_question = next;
            self.ready.pop_front();
            self.complete_role_queue()?;
            // A terminal Rules endpoint has no role ticket or NN forward.
            return self.next_role();
        }
        self.pending.insert(
            digest,
            PcPending {
                input: input.clone(),
                ticket: None,
                failure: None,
            },
        );
        self.next_question = next;
        self.ready.pop_front();
        Ok(Some(input))
    }

    pub fn role_submitted(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), PalsError> {
        self.check_input(input)?;
        if self.requires_shared_nnuec_readout()
            && !self
                .shared_nnuec_readout_consumed(input)
                .map_err(|_| RoleError::InvalidOutput)?
        {
            return Err(
                RoleError::Unsupported("registered shared NNUEC readout not consumed").into(),
            );
        }
        self.check_ticket(input, ticket)?;
        let held = self
            .pending
            .get_mut(&input.question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        if let Some(previous) = held.ticket {
            return if previous == *ticket {
                Ok(())
            } else {
                Err(RoleError::InvalidOutput.into())
            };
        }
        if self.counters.submitted_role_calls >= self.owner.config.max_role_calls {
            return Err(PalsError::RoleCallLimit);
        }
        held.ticket = Some(*ticket);
        self.counters.submitted_role_calls += 1;
        Ok(())
    }

    pub fn validate_role(
        &self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), PalsError> {
        self.check_input(input)?;
        self.check_ticket(input, ticket)?;
        let held = self
            .pending
            .get(&input.question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        if held.ticket != Some(*ticket)
            || held.failure.is_some()
            || output.id != ticket.id
            || output.authority != ticket.authority
            || output.key != ticket.representation
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if let Some(namespace) = self
            .owner
            .semantic
            .model
            .registered_representation_namespace()
        {
            namespace.verify(
                ticket,
                output,
                self.owner.semantic.model_registered_value.as_ref(),
            )?;
        } else if self
            .owner
            .semantic
            .model
            .requires_registered_representation_namespace()
        {
            return Err(RoleError::Unavailable.into());
        }
        super::adaptive_port::validate_payload(input.question.context.purpose, &input.view, output)
    }

    /// A retry of the exact successful commit is observational only. A foreign
    /// source/output or a failed prior commit cannot become a success retry.
    pub fn consume_accepted(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: Option<AdaptivePortRoleEvidence>,
    ) -> Result<(), PalsError> {
        if Instant::now() >= self.hard_deadline {
            return Err(RoleError::Deadline.into());
        }
        if matches!(
            self.completion,
            PcPortCompletion::Canceled
                | PcPortCompletion::Deadline
                | PcPortCompletion::WorkBudget
                | PcPortCompletion::Failed
        ) {
            return Err(RoleError::Canceled.into());
        }
        if let Some(old) = self
            .consumed
            .iter()
            .find(|old| old.ticket.id == accepted.ticket().id)
        {
            PcPortQuestion::validate_input_source(input, &self.owner.semantic.stores)?;
            let same = input.question_digest == accepted.ticket().question
                && input.view_seal == accepted.ticket().view
                && input.question.source.original_deadline == accepted.ticket().original_deadline
                && Arc::ptr_eq(&input.question.source.owner, &self.identity)
                && old.ticket == *accepted.ticket()
                && old.evidence == accepted.output_digest()
                && accepted.validates_output(output)
                && old
                    .question
                    .upgrade()
                    .is_some_and(|q| Arc::ptr_eq(&q, &input.question))
                && old
                    .view
                    .upgrade()
                    .is_some_and(|v| Arc::ptr_eq(&v, &input.view));
            return if same && old.semantic_committed {
                Ok(())
            } else {
                Err(RoleError::InvalidOutput.into())
            };
        }
        self.validate_role(input, accepted.ticket(), output)?;
        if !accepted.validates_output(output) {
            return Err(RoleError::InvalidOutput.into());
        }
        self.consumed
            .retain(|entry| entry.question.strong_count() != 0 || entry.view.strong_count() != 0);
        if self.consumed.len() >= MAX_PC_CONSUMED {
            return Err(PalsError::Capacity);
        }
        self.consumed
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        let result = self.consume_once(input, accepted, output, physical);
        if let Err(error) = &result {
            if let Some(held) = self.pending.get_mut(&input.question_digest) {
                held.failure = Some(error.clone());
            }
            self.completion = PcPortCompletion::Failed;
            return result;
        }
        self.pending.remove(&input.question_digest);
        self.consumed
            .iter_mut()
            .find(|entry| entry.ticket == *accepted.ticket())
            .ok_or(RoleError::InvalidOutput)?
            .semantic_committed = true;
        self.complete_role_queue()?;
        Ok(())
    }

    /// An observation of the exact head already consumed by this owner. It
    /// grants no acceptance and remains true if later Store publication failed.
    pub fn role_consumption_recorded(
        &self,
        ticket: &RoleTicketEnvelope,
        input: &FrameInput<ReviewView, PcPortQuestion>,
    ) -> Result<bool, PalsError> {
        PcPortQuestion::validate_input_source(input, &self.owner.semantic.stores)?;
        if !Arc::ptr_eq(&input.question.source.owner, &self.identity)
            || input.question.source.authority != self.authority
            || input.question.source.original_deadline != self.deadline
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(self.consumed.iter().any(|entry| {
            entry.ticket == *ticket
                && ticket.question == input.question_digest
                && ticket.view == input.view_seal
                && entry
                    .question
                    .upgrade()
                    .is_some_and(|q| Arc::ptr_eq(&q, &input.question))
                && entry
                    .view
                    .upgrade()
                    .is_some_and(|v| Arc::ptr_eq(&v, &input.view))
        }))
    }

    pub fn role_discarded(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        ticket: Option<&RoleTicketEnvelope>,
    ) -> Result<(), PalsError> {
        let held = self
            .pending
            .get(&input.question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        if held.ticket.as_ref() != ticket
            || !Arc::ptr_eq(&held.input.question, &input.question)
            || !Arc::ptr_eq(&held.input.view, &input.view)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        self.pending.remove(&input.question_digest);
        if self.completion == PcPortCompletion::Active {
            self.completion = if Instant::now() >= self.hard_deadline {
                PcPortCompletion::Deadline
            } else {
                PcPortCompletion::Failed
            };
        }
        Ok(())
    }

    pub fn finish_materialized(&self) -> PcPortResult {
        let mut best_move = self
            .root_ranking
            .first()
            .copied()
            .or_else(|| self.root_moves.allowed().first().copied());
        let mut best = PalsResolvedValue::Unknown;
        let mut scope = PalsValueScope::Unknown;
        for movement in self.root_moves.allowed().iter() {
            if let Some(edge) = self.owner.semantic.nodes[self.root]
                .edges
                .iter()
                .find(|edge| edge.movement == *movement)
            {
                let (value, candidate_scope) = self.owner.root_model_value(edge.child);
                if better_model_choice(
                    value,
                    best,
                    candidate_scope == PalsValueScope::RulesTerminal,
                    scope == PalsValueScope::RulesTerminal,
                ) {
                    best_move = Some(*movement);
                    best = value;
                    scope = candidate_scope;
                }
            }
        }
        if self.owner.semantic.nodes[self.root].terminal.is_some() {
            best_move = None;
            best = self.owner.semantic.rules_value(self.root);
            scope = PalsValueScope::RulesTerminal;
        } else if let Some(strategy) = self
            .rovent_frontier()
            .and_then(|frontier| frontier.snapshot().mate_strategy)
            .or_else(|| self.retired_strategy.clone())
            .filter(|strategy| {
                strategy.validates_root(
                    &self.owner.semantic.nodes[self.root].position.snapshot(),
                    self.root_moves.allowed(),
                )
            })
        {
            best_move = Some(strategy.best_move());
            // A proof move must not inherit a different model-selected move's
            // estimate. The mate proof itself stays typed in Frontier receipt.
            let selected = self.owner.semantic.nodes[self.root]
                .edges
                .iter()
                .find(|edge| edge.movement == strategy.best_move());
            (best, scope) = selected
                .map(|edge| self.owner.root_model_value(edge.child))
                .unwrap_or((PalsResolvedValue::Unknown, PalsValueScope::Unknown));
        }
        PcPortResult {
            best_move,
            completion: self.completion,
            value: best,
            scope,
            counters: self.counters,
            root_restricted: self.root_moves.allowed().len()
                != self.owner.semantic.nodes[self.root]
                    .position
                    .legal_moves()
                    .len(),
        }
    }

    fn check_input(&self, input: &FrameInput<ReviewView, PcPortQuestion>) -> Result<(), PalsError> {
        if self.completion != PcPortCompletion::Active {
            return Err(RoleError::Canceled.into());
        }
        if Instant::now() >= self.hard_deadline {
            return Err(RoleError::Deadline.into());
        }
        PcPortQuestion::validate_input_source(input, &self.owner.semantic.stores)?;
        let source = &input.question.source;
        let held = self
            .pending
            .get(&input.question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        if !Arc::ptr_eq(&source.owner, &self.identity)
            || source.authority != self.authority
            || source.original_deadline != self.deadline
            || source.sequence == 0
            || source.sequence >= self.next_question
            || input.question.deadline != self.hard_deadline
            || source.mode != self.mode
            || source.root_budget != self.original_root_budget()
            || !Arc::ptr_eq(&held.input.question, &input.question)
            || !Arc::ptr_eq(&held.input.view, &input.view)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }
    fn check_ticket(
        &self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), PalsError> {
        if ticket.authority != self.authority
            || ticket.question != input.question_digest
            || ticket.view != input.view_seal
            || ticket.original_deadline != self.deadline
            || ticket.role != role_for_purpose(input.question.context.purpose)?
            || ticket.representation.record_revision != input.question.context.public_revision
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }

    fn consume_once(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: Option<AdaptivePortRoleEvidence>,
    ) -> Result<(), PalsError> {
        let query = &input.question;
        let fresh = query.context.purpose == RoleQueryPurpose::ValueFresh;
        if fresh
            && physical.is_none_or(|e| {
                e.physical_fresh != Some(true)
                    || !e.measurement_complete
                    || e.completed_nn_inputs.is_none_or(|n| n == 0)
                    || e.execution.is_none_or(|e| e.epoch != self.authority.epoch)
            })
        {
            return Err(StoreError::InvalidEvidence(
                "PC Fresh endpoint needs original physical evidence",
            )
            .into());
        }
        if query.repair.is_some() {
            return self.consume_repair_once(input, accepted, output, physical);
        }
        let ranked = rank_moves(&input.view.query().legal, output)?;
        let root_ranking = query.context.prefix.is_empty().then(|| {
            ranked
                .iter()
                .copied()
                .filter(|movement| self.root_moves.allowed().contains(movement))
                .collect::<Vec<_>>()
        });
        let mut continuations = Vec::new();
        if !fresh {
            let width = if query.context.prefix.is_empty() {
                self.owner.config.beam_width
            } else {
                1
            };
            for movement in ranked
                .into_iter()
                .filter(|movement| {
                    !query.context.prefix.is_empty() || self.root_moves.allowed().contains(movement)
                })
                .take(width)
            {
                let mut prefix = query.context.prefix.clone();
                prefix.push(movement);
                continuations.push((movement, prefix));
            }
            if continuations.is_empty() {
                return Err(RoleError::InvalidOutput.into());
            }
            if self
                .ready
                .len()
                .checked_add(continuations.len())
                .is_none_or(|n| n > MAX_PC_PENDING)
            {
                return Err(PalsError::Capacity);
            }
            if self
                .owner
                .semantic
                .records
                .len()
                .checked_add(continuations.len())
                .is_none_or(|n| n > self.owner.config.max_records)
            {
                return Err(PalsError::Capacity);
            }
        }
        self.consume_pc_head_once(input, accepted, physical)?;
        if let Some(ranking) = root_ranking {
            self.root_ranking = ranking;
        }
        let peer_source = if !fresh && query.goal_local.is_none() && self.frontier_state.is_some() {
            Some(PcPeerSource::freeze(input, accepted, output)?)
        } else {
            None
        };
        if fresh {
            self.commit_value(input, output)?;
            if self.frontier_state.is_some() {
                if let Err(error) = self.accept_frontier_endpoint(
                    input,
                    Some((accepted, output, physical.ok_or(RoleError::InvalidOutput)?)),
                ) {
                    self.frontier_boundary_or_failure(error)?;
                    return Ok(());
                }
            }
        }
        for (movement, prefix) in continuations {
            let Some(admitted) = self.admit_selected_child(input, accepted, output, movement)?
            else {
                return Ok(());
            };
            let child = admitted.node;
            if let Some(origin) = &query.goal_local {
                let checked = admitted.checked.ok_or(RoleError::InvalidOutput)?;
                let next =
                    Arc::new(origin.successor(input, accepted, output, movement, checked, child)?);
                self.ready.push_back(PcPlan {
                    node: child,
                    frame: query.frame,
                    responsibilities: Vec::new(),
                    goal_local: Some(Arc::clone(&next)),
                    peer_source: None,
                    repair: None,
                    continuation_end: next.continuation_end(),
                    purpose: next.purpose(),
                    prefix,
                    proposal: Vec::new(),
                    counter: None,
                });
                continue;
            }
            let kind = match query.context.purpose {
                RoleQueryPurpose::ReplyPolicy => RecordKind::Counterexample,
                RoleQueryPurpose::RepairPolicy => RecordKind::Repair,
                _ => RecordKind::Proposal,
            };
            self.publish_line(kind, &prefix)?;
            let terminal = self.owner.semantic.nodes[child].terminal.is_some();
            let continuation_end = query.continuation_end;
            let end = prefix.len() >= continuation_end;
            let frame = if query.context.prefix.is_empty() {
                let frame = self.next_frame;
                self.next_frame = frame.checked_add(1).ok_or(PalsError::Capacity)?;
                frame
            } else {
                query.frame
            };
            let mut responsibilities = query.responsibilities.clone();
            let responsibility = match query.context.purpose {
                RoleQueryPurpose::ProposePolicy if query.context.prefix.is_empty() => {
                    Some(FrontierFrameKind::Candidate)
                }
                RoleQueryPurpose::ReplyPolicy => Some(FrontierFrameKind::Counter),
                RoleQueryPurpose::RepairPolicy => Some(FrontierFrameKind::Repair),
                _ => None,
            };
            if let Some(kind) = responsibility
                && !responsibilities.iter().any(|source| source.kind == kind)
            {
                responsibilities.push(PcResponsibilitySource::from_selection(
                    kind,
                    input,
                    accepted,
                    movement,
                    &self.owner.semantic.nodes[child].position,
                    prefix.clone(),
                )?);
            }
            if query.context.prefix.is_empty()
                && query.context.purpose == RoleQueryPurpose::ProposePolicy
            {
                self.enqueue_actual_repair_cycle(
                    input,
                    accepted,
                    output,
                    child,
                    &prefix,
                    frame,
                    &responsibilities,
                )?;
            }
            self.ready.push_back(if end || terminal {
                PcPlan {
                    node: child,
                    frame,
                    responsibilities: responsibilities.clone(),
                    goal_local: None,
                    peer_source: None,
                    repair: None,
                    continuation_end,
                    purpose: RoleQueryPurpose::ValueFresh,
                    proposal: prefix.clone(),
                    prefix,
                    counter: None,
                }
            } else if query.context.purpose == RoleQueryPurpose::ReplyPolicy {
                PcPlan {
                    node: child,
                    frame,
                    responsibilities: responsibilities.clone(),
                    goal_local: None,
                    peer_source: peer_source.clone(),
                    repair: None,
                    continuation_end,
                    purpose: RoleQueryPurpose::RepairPolicy,
                    proposal: input.view.query().proposal.clone(),
                    counter: Some(prefix.clone()),
                    prefix,
                }
            } else {
                PcPlan {
                    node: child,
                    frame,
                    responsibilities: responsibilities.clone(),
                    goal_local: None,
                    peer_source: peer_source.clone(),
                    repair: None,
                    continuation_end,
                    purpose: RoleQueryPurpose::ReplyPolicy,
                    proposal: prefix.clone(),
                    prefix,
                    counter: None,
                }
            });
        }
        Ok(())
    }

    fn consume_pc_head_once(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        accepted: &AcceptedRoleStep,
        physical: Option<AdaptivePortRoleEvidence>,
    ) -> Result<(), PalsError> {
        let query = &input.question;
        let fresh = query.context.purpose == RoleQueryPurpose::ValueFresh;
        self.owner
            .semantic
            .review_index
            .as_mut()
            .ok_or(PalsError::InvalidConfig)?
            .consume(&input.view, &self.owner.semantic.stores)?;
        self.consumed.push_back(PcConsumed {
            ticket: *accepted.ticket(),
            evidence: accepted.output_digest(),
            question: Arc::downgrade(&input.question),
            view: Arc::downgrade(&input.view),
            semantic_committed: false,
        });
        self.counters.consumed_role_outputs += 1;
        match role_for_purpose(query.context.purpose)? {
            Role::Proposer => self.counters.accepted_proposer += 1,
            Role::Critic => self.counters.accepted_critic += 1,
            Role::Validator => unreachable!("PC never admits V"),
        }
        if fresh {
            self.counters.accepted_fresh += 1;
        }
        self.counters.completed_nn_inputs = match (
            self.counters.completed_nn_inputs,
            physical.and_then(|e| e.completed_nn_inputs),
        ) {
            (Some(total), Some(count)) => total.checked_add(count),
            _ => None,
        };
        Ok(())
    }

    fn publish_line(&mut self, kind: RecordKind, moves: &[BoardMove]) -> Result<(), PalsError> {
        self.publish_pc_repair_line(kind, moves).map(|_| ())
    }
    fn publish_pc_repair_line(
        &mut self,
        kind: RecordKind,
        moves: &[BoardMove],
    ) -> Result<PublishedPcLine, PalsError> {
        self.publish_pc_repair_line_inner(kind, moves, None)
    }
    fn publish_pc_repair_line_with_relations(
        &mut self,
        kind: RecordKind,
        moves: &[BoardMove],
        relations: repair_cycle::PcLineRelations,
    ) -> Result<PublishedPcLine, PalsError> {
        self.publish_pc_repair_line_inner(kind, moves, Some(relations))
    }
    fn publish_pc_repair_line_inner(
        &mut self,
        kind: RecordKind,
        moves: &[BoardMove],
        relations: Option<repair_cycle::PcLineRelations>,
    ) -> Result<PublishedPcLine, PalsError> {
        let root = &self.owner.semantic.nodes[self.root];
        let root_state = root.state;
        let root_situation = root.situation;
        let revision = self
            .owner
            .semantic
            .revision
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        let parent_revision = self
            .owner
            .semantic
            .records
            .iter()
            .rev()
            .find(|r| {
                r.origin_state == root_state
                    && r.line.first() == moves.first()
                    && match kind {
                        RecordKind::Counterexample => {
                            matches!(r.kind, RecordKind::Proposal | RecordKind::Repair)
                        }
                        RecordKind::Repair => r.kind == RecordKind::Counterexample,
                        _ => false,
                    }
            })
            .map(|r| r.revision);
        let supersedes_revision = self
            .owner
            .semantic
            .records
            .iter()
            .rev()
            .find(|r| {
                r.origin_state == root_state && r.line.first() == moves.first() && r.kind == kind
            })
            .map(|r| r.revision);
        // Actual cycle publication names its own accepted lineage. Ordinary
        // lanes retain the legacy latest-record selection above.
        let (parent_revision, supersedes_revision, supersedes_observation) =
            if let Some(relations) = relations {
                for (relation, parent) in [(relations.parent, true), (relations.supersedes, false)]
                {
                    let Some(relation) = relation else { continue };
                    let record = self
                        .owner
                        .semantic
                        .records
                        .iter()
                        .find(|record| record.revision == relation.revision)
                        .ok_or(StoreError::StaleConsumer)?;
                    let observation = self
                        .owner
                        .semantic
                        .stores
                        .observations
                        .get(relation.observation)?;
                    if record.origin_state != root_state
                        || record.kind != relation.kind
                        || record.line.first() != moves.first()
                        || self.owner.semantic.stores.lines.moves(relation.line)? != record.line
                        || observation.state != root_state
                        || observation.line != Some(relation.line)
                        || if parent {
                            match kind {
                                RecordKind::Counterexample => !matches!(
                                    record.kind,
                                    RecordKind::Proposal | RecordKind::Repair
                                ),
                                RecordKind::Repair => record.kind != RecordKind::Counterexample,
                                _ => true,
                            }
                        } else {
                            record.kind != kind
                        }
                    {
                        return Err(StoreError::InvalidEvidence("actual PC Repair lineage").into());
                    }
                }
                (
                    relations.parent.map(|line| line.revision),
                    relations.supersedes.map(|line| line.revision),
                    relations.supersedes.map(|line| line.observation),
                )
            } else {
                (parent_revision, supersedes_revision, None)
            };
        let focus = self
            .owner
            .semantic
            .stores
            .situations
            .get(root_situation)?
            .focus;
        let line = self.owner.semantic.stores.hot_append_line(focus, moves)?;
        let observation = self
            .owner
            .semantic
            .stores
            .hot_append_observation(Observation {
                state: root_state,
                line: Some(line),
                source: stable_id(self.owner.model_identity()),
                epoch: 0,
                scope: EvidenceScope::Model {
                    model: stable_id(self.owner.model_identity()),
                    encoding: stable_id("pals-role-query-v1"),
                    input: revision,
                },
                value_identity: None,
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: None,
                model_value_input: None,
                cpu_condition: None,
                cpu_pv: None,
                score: RawScore::Unknown,
                budget: 0,
                kind: match kind {
                    RecordKind::Proposal => ObservationKind::Proposal,
                    RecordKind::Counterexample => ObservationKind::Refutation,
                    RecordKind::Repair => ObservationKind::Repair,
                    RecordKind::CpuVerification => return Err(RoleError::InvalidOutput.into()),
                },
                supersedes: supersedes_observation,
                execution: None,
            })?;
        self.owner
            .semantic
            .pin_archive_active_record(line, observation)?;
        self.owner.publish_record(
            RoleRecord {
                revision,
                parent_revision,
                supersedes_revision,
                origin_state: root_state,
                kind,
                line: moves.to_vec(),
                value: None,
                completed_depth: 0,
                score_scope: None,
                cpu_observation: None,
                perspective: self.owner.semantic.nodes[self.root].position.side_to_move(),
                critical: true,
            },
            None,
            Some(observation),
        )?;
        Ok(PublishedPcLine {
            line,
            observation,
            revision,
            kind,
        })
    }

    fn commit_value(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        output: &RoleOutput,
    ) -> Result<(), PalsError> {
        let RolePayload::Proposal { wdl, .. } = output.payload else {
            return Err(RoleError::InvalidOutput.into());
        };
        let query = &input.question;
        let identity = self
            .owner
            .semantic
            .model_registered_value
            .clone()
            .ok_or(RoleError::Unavailable)?;
        let value = ModelValueOutput {
            identity: identity.clone(),
            input_sha256: output.key.input.0,
            state: query.position.position_identity(),
            perspective: query.position.side_to_move(),
            wdl: wdl.probabilities(),
        };
        value.validate(&query.position, &identity)?;
        let observation = self
            .owner
            .semantic
            .stores
            .hot_append_observation(Observation {
                state: query.context.state,
                line: None,
                source: stable_id(&identity.semantics),
                epoch: 0,
                scope: EvidenceScope::Model {
                    model: stable_id(&identity.model),
                    encoding: stable_id(&identity.encoding),
                    input: u64::from_le_bytes(
                        output.key.input.0[..8]
                            .try_into()
                            .map_err(|_| RoleError::InvalidOutput)?,
                    ),
                },
                value_identity: None,
                checker_identity: None,
                checker_work: None,
                external_report: None,
                model_value_identity: Some(identity),
                model_value_input: Some(output.key.input.0),
                cpu_condition: None,
                cpu_pv: None,
                score: RawScore::ContextWdl {
                    win: value.wdl[0],
                    draw: value.wdl[1],
                    loss: value.wdl[2],
                    perspective: value.perspective,
                    context_revision: query.context.public_revision,
                },
                budget: 1,
                kind: ObservationKind::Proposal,
                supersedes: None,
                execution: None,
            })?;
        self.owner
            .semantic
            .stores
            .hot_add_dependency(observation, query.context.situation)?;
        self.owner
            .semantic
            .stores
            .hot_add_dependency(observation, self.owner.semantic.nodes[self.root].situation)?;
        let node = &mut self.owner.semantic.nodes[query.node];
        node.model_value = Some(value);
        node.model_observation = Some(observation);
        node.model_value_revision = Some(query.context.public_revision);
        Ok(())
    }
}

fn role_for_purpose(purpose: RoleQueryPurpose) -> Result<Role, PalsError> {
    match purpose {
        RoleQueryPurpose::ProposePolicy
        | RoleQueryPurpose::RepairPolicy
        | RoleQueryPurpose::ValueFresh => Ok(Role::Proposer),
        RoleQueryPurpose::ReplyPolicy | RoleQueryPurpose::DivergencePolicy => Ok(Role::Critic),
        _ => Err(RoleError::Unsupported("PC port does not admit V or an undeclared head").into()),
    }
}
fn stage_for_purpose(purpose: RoleQueryPurpose) -> Result<RefinementStage, PalsError> {
    match purpose {
        RoleQueryPurpose::ProposePolicy => Ok(RefinementStage::ProposerRanking),
        RoleQueryPurpose::ReplyPolicy => Ok(RefinementStage::CriticReply),
        RoleQueryPurpose::RepairPolicy => Ok(RefinementStage::Repair),
        RoleQueryPurpose::ValueFresh => Ok(RefinementStage::ProposalValueFresh),
        RoleQueryPurpose::DivergencePolicy => Ok(RefinementStage::CriticDivergence),
        _ => Err(RoleError::InvalidOutput.into()),
    }
}
fn rank_moves(legal: &[BoardMove], output: &RoleOutput) -> Result<Vec<BoardMove>, PalsError> {
    let logits = match &output.payload {
        RolePayload::Proposal {
            candidate_logits, ..
        }
        | RolePayload::Counterexample {
            candidate_logits, ..
        } => candidate_logits,
        _ => return Err(RoleError::InvalidOutput.into()),
    };
    if logits.len() != legal.len() || logits.iter().any(|value| !value.is_finite()) {
        return Err(RoleError::InvalidOutput.into());
    }
    let mut indices: Vec<_> = (0..legal.len()).collect();
    indices.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    Ok(indices.into_iter().map(|index| legal[index]).collect())
}
fn logical_digest(context: &RoleLogicalContext) -> Result<ContractDigest, PalsError> {
    let mut hash = Sha256::new();
    hash.update(b"rz-rovent-pc-private-logical-source/1\0");
    for value in [
        context.game_generation,
        context.search_generation,
        context.situation.slot as u64,
        context.situation.generation,
        context.state.0 as u64,
        context.focus.0 as u64,
        context.public_revision,
        context.situation_revision,
    ] {
        hash.update(value.to_le_bytes());
    }
    hash.update([context.purpose as u8]);
    hash.update(role_line_sha256(b"prefix", &context.prefix)?);
    hash.update(context.focus_sha256);
    hash.update(context.prefix_sha256);
    hash.update(context.proposal_sha256);
    hash.update([u8::from(context.refutation_sha256.is_some())]);
    if let Some(value) = context.refutation_sha256 {
        hash.update(value);
    }
    hash.update(context.divergence_sha256);
    Ok(ContractDigest(hash.finalize().into()))
}
fn question_digest(
    authority: SearchAuthority,
    sequence: u64,
    context: &RoleLogicalContext,
    view: ReviewViewSeal,
    model: &str,
) -> Result<ContractDigest, PalsError> {
    let mut hash = Sha256::new();
    hash.update(b"rz-rovent-pc-exact-role-question/1\0");
    hash.update(authority.epoch.0.to_le_bytes());
    hash.update(authority.game.0.to_le_bytes());
    hash.update(authority.root.0.to_le_bytes());
    hash.update(authority.implementation.0);
    hash.update(sequence.to_le_bytes());
    hash.update(logical_digest(context)?.0);
    hash.update(view.view_digest.0);
    hash.update(view.context_policy.0);
    hash.update((model.len() as u64).to_le_bytes());
    hash.update(model.as_bytes());
    Ok(ContractDigest(hash.finalize().into()))
}

#[cfg(test)]
mod tests;
