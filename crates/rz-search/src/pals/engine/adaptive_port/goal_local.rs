//! Goal-local work over an actual controller-issued Rules/depth question.
//!
//! This owner creates neither an ordinary refinement frame, a proposer opinion,
//! a reciprocal envelope nor a CPU search. The manager retains the opaque input
//! beside its real native ticket and returns its physically retired acceptance.
//! One preparation turn advances at most one imposed edge.
use super::*;
use crate::pals::coordinator::AcceptedRoleStep;
use crate::pals::rovent_frontier::{
    CheckedFrontierLine, FrontierError, FrontierHorizonAcceptance, MissingMateQuestion,
    MissingMateQuestionKey, UnmetHorizonQuestion, UnmetHorizonQuestionKey,
};
use crate::pals::store::{StorePins, StoreReadOwner};
use rz_contracts::{ClockDomain, MonotonicTick};
use std::sync::Weak;

const MAX_GOAL_LOCAL_PENDING: usize = 16;
const MAX_GOAL_LOCAL_ATTEMPTS: usize = 64;

/// Read-only metadata is not an admission: only the private origin constructor
/// below can retain the original Missing capability and charged exact path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GoalLocalBlockReason {
    ViewLeaseCapacity,
    PendingCapacity,
    PreviousAttemptFailed,
    Frontier(FrontierError),
}

/// Manager custody identity only. Its variants do not construct a question,
/// manufacture a frame or grant any Rules transition or native submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GoalLocalQuestionKey {
    Mate(MissingMateQuestionKey),
    Horizon(UnmetHorizonQuestionKey),
}

pub enum AdaptiveGoalLocalAdmission<Q = AdaptivePortQuestion> {
    RoleInput(GoalLocalInput<Q>),
    RulesAdvanced { key: GoalLocalQuestionKey },
    NoQuestion,
    Blocked { reason: GoalLocalBlockReason },
}

/// A real immutable Q/view allocation, rather than caller-written authority or
/// an invented CoordinatorWork. Its constructor is local to this semantic owner.
pub struct GoalLocalInput<Q = AdaptivePortQuestion> {
    seal: Arc<GoalLocalInputSeal<Q>>,
}
impl<Q> Clone for GoalLocalInput<Q> {
    fn clone(&self) -> Self {
        Self {
            seal: Arc::clone(&self.seal),
        }
    }
}
struct GoalLocalInputSeal<Q> {
    input: FrameInput<ReviewView, Q>,
    origin: Arc<GoalLocalOrigin>,
}
impl<Q> GoalLocalInput<Q> {
    pub fn input(&self) -> &FrameInput<ReviewView, Q> {
        &self.seal.input
    }
    pub fn key(&self) -> GoalLocalQuestionKey {
        self.source_key()
    }
    pub fn source_key(&self) -> GoalLocalQuestionKey {
        self.seal.origin.source_key()
    }
    pub fn original_deadline(&self) -> Deadline {
        self.seal.origin.source.original_deadline
    }
    pub fn root_budget(&self) -> ContractDigest {
        self.seal.origin.source.root_budget
    }
    pub fn original_first_move(&self) -> BoardMove {
        self.seal.origin.source.first
    }
    pub fn authority(&self) -> SearchAuthority {
        self.seal.origin.source.authority
    }
}

impl GoalLocalInput<AdaptivePortQuestion> {
    /// Concrete producer validation, not a caller-declared frame/source.
    /// Frontier rechecks its own original duty/root/budget after this gate.
    pub(crate) fn validated_horizon_source(
        &self,
        stores: &PalsStores,
    ) -> Result<&UnmetHorizonQuestion, FrontierError> {
        AdaptivePortQuestion::validate_frontier_input(self.input(), stores)
            .map_err(|_| FrontierError::MissingAcceptance)?;
        self.seal
            .origin
            .validates_frame_input(self.input())
            .map_err(|_| FrontierError::MissingAcceptance)?;
        match &self.seal.origin.source.question {
            GoalLocalRootQuestion::Horizon(question) => Ok(question),
            GoalLocalRootQuestion::Mate(_) => Err(FrontierError::MissingAcceptance),
        }
    }
}

pub enum GoalLocalStep {
    ContinuePrepared { key: GoalLocalQuestionKey },
    Leaf(GoalLocalLeaf),
}

/// Closure of this exact local action, not a forced-mate certificate. A Fresh
/// endpoint can finish its action while the mate coverage remains Unknown.
#[derive(Clone)]
pub struct GoalLocalLeaf {
    origin: Arc<GoalLocalOrigin>,
    snapshot: PositionSnapshot,
    kind: GoalLocalLeafKind,
    horizon_acceptance: Option<FrontierHorizonAcceptance>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GoalLocalLeafKind {
    RulesTerminal,
    FreshBoundary,
}
impl GoalLocalLeaf {
    pub fn source_key(&self) -> GoalLocalQuestionKey {
        self.origin.source_key()
    }
    pub fn snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }
    pub fn kind(&self) -> GoalLocalLeafKind {
        self.kind
    }
    pub fn horizon_acceptance(&self) -> Option<FrontierHorizonAcceptance> {
        self.horizon_acceptance
    }
}

enum GoalLocalRootQuestion {
    Mate(Arc<MissingMateQuestion>),
    Horizon(Arc<UnmetHorizonQuestion>),
}
impl GoalLocalRootQuestion {
    fn key(&self) -> GoalLocalQuestionKey {
        match self {
            Self::Mate(question) => GoalLocalQuestionKey::Mate(question.key()),
            Self::Horizon(question) => GoalLocalQuestionKey::Horizon(question.key()),
        }
    }
    fn original_deadline(&self) -> Deadline {
        match self {
            Self::Mate(question) => question.original_deadline(),
            Self::Horizon(question) => question.original_deadline(),
        }
    }
    fn declared_backing(&self) -> usize {
        match self {
            Self::Mate(_) => std::mem::size_of::<MissingMateQuestion>(),
            Self::Horizon(_) => std::mem::size_of::<UnmetHorizonQuestion>(),
        }
    }
}

struct GoalLocalRootSource {
    question: GoalLocalRootQuestion,
    owner: Arc<()>,
    authority: SearchAuthority,
    root_budget: ContractDigest,
    original_deadline: Deadline,
    hard_deadline: Instant,
    first: BoardMove,
    imposed_prefix: Vec<BoardMove>,
    imposed_snapshot: PositionSnapshot,
    root_side: Color,
}

/// No public constructor/fields and no original Q/view pin chain. Descendants
/// keep this exact Rules origin; they cannot turn it into a model proposal.
pub struct GoalLocalOrigin {
    source: Arc<GoalLocalRootSource>,
    prefix: Vec<BoardMove>,
    snapshot: PositionSnapshot,
    continuation_end: usize,
    digest: ContractDigest,
}
impl GoalLocalOrigin {
    /// Before preparing a view, match the concrete semantic node to this
    /// immutable paid path. This gate performs neither Rules replay nor NN
    /// admission and cannot be called to construct a new origin.
    pub(super) fn validate_prepared_node<M: RoleModel>(
        &self,
        port: &AdaptiveEnginePort<M>,
        node: usize,
        first: BoardMove,
        prefix: &[BoardMove],
    ) -> Result<(), PalsError> {
        let semantic = port
            .engine
            .semantic
            .nodes
            .get(node)
            .ok_or(RoleError::InvalidOutput)?;
        let frontier = port.rovent_frontier().ok_or(FrontierError::ForeignRoot)?;
        if !Arc::ptr_eq(&self.source.owner, &port.goal_local.owner)
            || self.source.authority != port.authority
            || self.source.original_deadline != port.deadline
            || self.source.hard_deadline != port.limits.deadline
            || self.source.root_budget != frontier.registration().root_budget
            || first != self.source.first
            || prefix != self.prefix
            || prefix.first().copied() != Some(first)
            || !frontier.root_moves().allowed().contains(&first)
            || !semantic.position.snapshot().same_state(&self.snapshot)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let checked = port
            .existing_checked_frontier_line(prefix)?
            .ok_or(FrontierError::ForeignHandle)?;
        if !checked.endpoint_snapshot().same_state(&self.snapshot)
            || !self.source.imposed_snapshot.same_state(
                checked
                    .steps()
                    .get(self.source.imposed_prefix.len() - 1)
                    .ok_or(RoleError::InvalidOutput)?
                    .snapshot(),
            )
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }

    pub(super) fn source_key(&self) -> GoalLocalQuestionKey {
        self.source.question.key()
    }
    pub(super) fn prefix(&self) -> &[BoardMove] {
        &self.prefix
    }
    pub(super) fn snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }
    pub(super) fn digest(&self) -> ContractDigest {
        self.digest
    }
    /// Conservative declared backing for the Q's shared origin, not an RSS
    /// peak or proof of physical sharing with its current Rules history.
    pub(super) fn owned_host_bytes(&self) -> Option<u64> {
        let bytes = std::mem::size_of::<Self>()
            .checked_add(std::mem::size_of::<GoalLocalRootSource>())?
            .checked_add(self.source.question.declared_backing())?
            .checked_add(
                self.prefix
                    .capacity()
                    .checked_mul(std::mem::size_of::<BoardMove>())?,
            )?
            .checked_add(
                self.source
                    .imposed_prefix
                    .capacity()
                    .checked_mul(std::mem::size_of::<BoardMove>())?,
            )?
            .checked_add(self.snapshot.retained_history_bytes()?)?
            .checked_add(self.source.imposed_snapshot.retained_history_bytes()?)?;
        u64::try_from(bytes).ok()
    }
    pub(super) fn first_move(&self) -> BoardMove {
        self.source.first
    }
    pub(super) fn original_deadline(&self) -> Deadline {
        self.source.original_deadline
    }
    pub(super) fn purpose(&self) -> RoleQueryPurpose {
        if self.prefix.len() >= self.continuation_end {
            RoleQueryPurpose::ValueFresh
        } else if self.snapshot.side_to_move() == self.source.root_side {
            RoleQueryPurpose::ProposePolicy
        } else {
            RoleQueryPurpose::ReplyPolicy
        }
    }
    pub(super) fn stage(&self) -> RefinementStage {
        match self.purpose() {
            RoleQueryPurpose::ValueFresh => RefinementStage::ProposalValueFresh,
            RoleQueryPurpose::ReplyPolicy => RefinementStage::CounterContinuation,
            _ => RefinementStage::ProposalContinuation,
        }
    }

    fn from_source(
        source: Arc<GoalLocalRootSource>,
        prefix: Vec<BoardMove>,
        snapshot: PositionSnapshot,
        continuation_end: usize,
    ) -> Result<Self, PalsError> {
        if prefix.is_empty()
            || !prefix.starts_with(&source.imposed_prefix)
            || prefix.len() > continuation_end
            || prefix.first() != Some(&source.first)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-adaptive-goal-local-origin/1\0");
        hash.update(source.root_budget.0);
        hash.update(source.original_deadline.at.0.to_le_bytes());
        match &source.question {
            GoalLocalRootQuestion::Mate(question) => {
                hash.update([0, question.attacker() as u8]);
                hash.update([match question.kind() {
                    crate::pals::rovent_frontier::MateQuestionKind::AttackCandidate => 0,
                    crate::pals::rovent_frontier::MateQuestionKind::RequiredDefense => 1,
                }]);
                hash.update(question.remaining().0.to_le_bytes());
            }
            GoalLocalRootQuestion::Horizon(question) => {
                hash.update([1]);
                hash.update(question.source_question().0);
                hash.update(question.frame().sequence.to_le_bytes());
                hash.update([match question.kind() {
                    crate::pals::rovent_frontier::FrontierFrameKind::Candidate => 0,
                    crate::pals::rovent_frontier::FrontierFrameKind::Counter => 1,
                    crate::pals::rovent_frontier::FrontierFrameKind::Repair => 2,
                }]);
                hash.update(question.target().0.to_le_bytes());
            }
        }
        hash.update(role_line_sha256(
            b"imposed-root-prefix",
            &source.imposed_prefix,
        )?);
        hash.update(role_line_sha256(b"goal-local-root-prefix", &prefix)?);
        hash.update(
            u64::try_from(continuation_end)
                .map_err(|_| PalsError::Capacity)?
                .to_le_bytes(),
        );
        Ok(Self {
            source,
            prefix,
            snapshot,
            continuation_end,
            digest: ContractDigest(hash.finalize().into()),
        })
    }

    /// Recheck the frozen producer origin. Its imposed edge no longer needs
    /// to be currently missing after it has actually been charged/observed.
    pub(super) fn validates_input(
        &self,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
        owner: &Arc<()>,
        authority: SearchAuthority,
        root_budget: ContractDigest,
    ) -> Result<(), PalsError> {
        let query = &input.question;
        if !Arc::ptr_eq(&self.source.owner, owner)
            || self.source.authority != authority
            || self.source.root_budget != root_budget
            || self.source.hard_deadline != query.deadline
            || query.first != Some(self.source.first)
            || query.context.prefix != self.prefix
            || query.context.purpose != self.purpose()
            || query.stage != self.stage()
            || !query
                .goal_local
                .as_deref()
                .is_some_and(|origin| std::ptr::eq(origin, self))
            || !query.position.snapshot().same_state(&self.snapshot)
            || !input.view.query().proposal.is_empty()
            || input.view.query().counterexample.is_some()
            || !input.view.query().divergences.is_empty()
            || query.context.proposal_sha256 != role_line_sha256(b"proposal", &[])?
            || query.context.refutation_sha256.is_some()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }

    pub(super) fn validates_frame_input(
        &self,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), PalsError> {
        self.validates_input(
            input,
            &self.source.owner,
            self.source.authority,
            self.source.root_budget,
        )
    }
}

struct PreparedGoalLocal {
    origin: Arc<GoalLocalOrigin>,
    checked: CheckedFrontierLine,
    node: Option<usize>,
}
/// The selector runs once for this exact physically accepted head. A native
/// reservation may temporarily prevent the paid child transition, without
/// consuming the head/view or losing its immutable selected movement.
struct GoalLocalSelection {
    input: Weak<GoalLocalInputSeal<AdaptivePortQuestion>>,
    ticket: RoleTicketEnvelope,
    output: ContractDigest,
    physical: GoalLocalPhysicalSeal,
    prefix: Vec<BoardMove>,
    prepared: Option<PreparedGoalLocal>,
}
struct GoalLocalAttempt {
    ticket: RoleTicketEnvelope,
    output: ContractDigest,
    physical: GoalLocalPhysicalSeal,
    question: Weak<AdaptivePortQuestion>,
    view: Weak<ReviewView>,
    head_used: bool,
    view_consumed: bool,
    failure: Option<PalsError>,
    completed: Option<CachedGoalLocalStep>,
}
#[derive(Clone, Copy, Eq, PartialEq)]
struct GoalLocalPhysicalSeal {
    execution: Option<rz_contracts::ExecutionId>,
    completed_nn_inputs: Option<u64>,
    measurement_complete: bool,
    physical_fresh: Option<bool>,
}
impl From<AdaptivePortRoleEvidence> for GoalLocalPhysicalSeal {
    fn from(value: AdaptivePortRoleEvidence) -> Self {
        Self {
            execution: value.execution,
            completed_nn_inputs: value.completed_nn_inputs,
            measurement_complete: value.measurement_complete,
            physical_fresh: value.physical_fresh,
        }
    }
}
enum CachedGoalLocalStep {
    ContinuePrepared { key: GoalLocalQuestionKey },
    Leaf(GoalLocalLeaf),
}
impl CachedGoalLocalStep {
    fn read(&self) -> Result<GoalLocalStep, PalsError> {
        match self {
            Self::ContinuePrepared { key } => Ok(GoalLocalStep::ContinuePrepared { key: *key }),
            Self::Leaf(leaf) => Ok(GoalLocalStep::Leaf(leaf.clone())),
        }
    }
}

pub(in crate::pals::engine) struct GoalLocalPortState {
    owner: Arc<()>,
    admission_closed: bool,
    prepared: Option<PreparedGoalLocal>,
    selection: Option<GoalLocalSelection>,
    retired_prepared: Option<PreparedGoalLocal>,
    retired_selection: Option<GoalLocalSelection>,
    pending: HashMap<ContractDigest, GoalLocalInput>,
    attempts: HashMap<RequestId, GoalLocalAttempt>,
}
impl Default for GoalLocalPortState {
    fn default() -> Self {
        Self {
            owner: Arc::new(()),
            admission_closed: false,
            prepared: None,
            selection: None,
            retired_prepared: None,
            retired_selection: None,
            pending: HashMap::new(),
            attempts: HashMap::new(),
        }
    }
}

impl GoalLocalPortState {
    fn has_retained_state(&self) -> bool {
        self.prepared.is_some()
            || self.selection.is_some()
            || self.retired_prepared.is_some()
            || self.retired_selection.is_some()
            || !self.pending.is_empty()
            || !self.attempts.is_empty()
    }
    fn owned_host_bytes(&self) -> Option<u64> {
        let mut bytes = u64::try_from(std::mem::size_of::<Self>()).ok()?;
        for prepared in [self.prepared.as_ref(), self.retired_prepared.as_ref()]
            .into_iter()
            .flatten()
        {
            bytes = bytes
                .checked_add(prepared.origin.owned_host_bytes()?)?
                .checked_add(prepared.checked.owned_host_bytes()?)?;
        }
        for selected in [self.selection.as_ref(), self.retired_selection.as_ref()]
            .into_iter()
            .flatten()
        {
            bytes = bytes
                .checked_add(u64::try_from(std::mem::size_of::<GoalLocalSelection>()).ok()?)?
                .checked_add(
                    u64::try_from(
                        selected
                            .prefix
                            .capacity()
                            .checked_mul(std::mem::size_of::<BoardMove>())?,
                    )
                    .ok()?,
                )?;
            if let Some(prepared) = &selected.prepared {
                bytes = bytes
                    .checked_add(prepared.origin.owned_host_bytes()?)?
                    .checked_add(prepared.checked.owned_host_bytes()?)?;
            }
        }
        // Count the declared hash payload backing. Hash control bytes,
        // allocator overhead and actual shared allocations are not observed.
        bytes = bytes.checked_add(
            u64::try_from(
                self.pending
                    .capacity()
                    .checked_mul(std::mem::size_of::<(ContractDigest, GoalLocalInput)>())?,
            )
            .ok()?,
        )?;
        for input in self.pending.values() {
            bytes = bytes
                .checked_add(input.input().question.owned_host_bytes()?)?
                .checked_add(input.input().view.owned_host_bytes()?)?;
        }
        bytes = bytes.checked_add(
            u64::try_from(
                self.attempts
                    .capacity()
                    .checked_mul(std::mem::size_of::<(RequestId, GoalLocalAttempt)>())?,
            )
            .ok()?,
        )?;
        for attempt in self.attempts.values() {
            if let Some(CachedGoalLocalStep::Leaf(leaf)) = &attempt.completed {
                bytes = bytes
                    .checked_add(leaf.origin.owned_host_bytes()?)?
                    .checked_add(u64::try_from(leaf.snapshot.retained_history_bytes()?).ok()?)?;
            }
            if attempt.failure.is_some() {
                // Dynamic backend/archive error backing is not fully known.
                return None;
            }
        }
        Some(bytes)
    }
}

fn pin_backing_bytes(pins: &StorePins) -> Option<u64> {
    let payload = pins
        .states
        .len()
        .checked_mul(std::mem::size_of::<crate::pals::store::StateId>())?
        .checked_add(
            pins.lines
                .len()
                .checked_mul(std::mem::size_of::<crate::pals::store::LineId>())?,
        )?
        .checked_add(
            pins.situations
                .len()
                .checked_mul(std::mem::size_of::<crate::pals::store::SituationId>())?,
        )?
        .checked_add(
            pins.observations
                .len()
                .checked_mul(std::mem::size_of::<crate::pals::store::ObservationId>())?,
        )?
        .checked_add(
            pins.executions
                .len()
                .checked_mul(std::mem::size_of::<crate::pals::store::ExecutionId>())?,
        )?;
    u64::try_from(payload).ok()
}

/// The same semantic state moved out of a port. This is retention only: it
/// issues no new-root, resume, native completion or archive-load authority.
/// Invalid pin evidence remains with the original state as an explicit error.
pub(in crate::pals::engine) struct GoalLocalRetiredCustody {
    state: GoalLocalPortState,
    pins: Result<StorePins, PalsError>,
    owner: StoreReadOwner,
    generation: u64,
    authority: SearchAuthority,
}
impl GoalLocalRetiredCustody {
    pub(in crate::pals::engine) fn preserve(
        state: GoalLocalPortState,
        pins: Result<StorePins, PalsError>,
        owner: StoreReadOwner,
        generation: u64,
        authority: SearchAuthority,
    ) -> Option<Box<Self>> {
        state.has_retained_state().then(|| {
            Box::new(Self {
                state,
                pins,
                owner,
                generation,
                authority,
            })
        })
    }
    pub(in crate::pals::engine) fn validate_owner(
        &self,
        stores: &PalsStores,
    ) -> Result<(), PalsError> {
        if !self.owner.matches(stores)
            || stores.generation() != self.generation
            || self.authority.root.0 != self.generation
        {
            return Err(StoreError::ArchiveIntegrity("retired GoalLocal Store/root owner").into());
        }
        Ok(())
    }
    pub(in crate::pals::engine) fn protected_pins(&self) -> Result<&StorePins, PalsError> {
        self.pins.as_ref().map_err(Clone::clone)
    }
    pub(in crate::pals::engine) fn owned_host_bytes(&self) -> Option<u64> {
        let mut bytes = u64::try_from(std::mem::size_of::<Self>()).ok()?;
        bytes = bytes.checked_add(self.state.owned_host_bytes()?)?;
        // Failed projection may retain dynamic backend/archive diagnostics.
        // Preserve it, but do not invent a complete backing count for it.
        let pins = self.pins.as_ref().ok()?;
        bytes = bytes.checked_add(pin_backing_bytes(pins)?)?;
        Some(bytes)
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Pre-permit check of the real pending allocation and submitted output.
    /// This creates no acceptance, physical fence, visit, view consumption or
    /// head-use receipt. The native registry remains the permit issuer.
    pub fn validate_rovent_goal_input(
        &self,
        input: &GoalLocalInput,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), PalsError> {
        if self.goal_local.admission_closed {
            return Err(RoleError::Canceled.into());
        }
        self.check_live()?;
        self.validate_goal_local_input(input)?;
        let actual = input.input();
        self.verify_registered_model_namespace(ticket, output)?;
        if ticket.authority != self.authority
            || ticket.question != actual.question_digest
            || ticket.original_deadline != self.deadline
            || ticket.view != actual.view_seal
            || ticket.representation != output.key
            || ticket.role != role_for_purpose(actual.question.context.purpose)
            || output.id != ticket.id
            || output.authority != ticket.authority
            || output.key.record_revision != actual.question.context.public_revision
        {
            return Err(RoleError::InvalidOutput.into());
        }
        validate_payload(actual.question.context.purpose, &actual.view, output)
    }

    /// Close logical admission only. Active native inputs remain pending until
    /// the actual manager returns their original physical-retirement custody.
    pub fn close_rovent_goal_admission(&mut self) -> Result<(), PalsError> {
        if self.goal_local.admission_closed {
            return Ok(());
        }
        self.goal_local.admission_closed = true;
        self.goal_local.retired_prepared = self.goal_local.prepared.take();
        self.goal_local.retired_selection = self.goal_local.selection.take();
        Ok(())
    }

    /// The manager calls this after its real native fence and exact input
    /// retirement. Remaining pending/active preparation is never discarded by
    /// this release. Closed-only is empty for into_engine preservation.
    pub fn release_rovent_goal_state(&mut self) -> Result<(), PalsError> {
        if !self.goal_local.admission_closed
            || self.goal_local.prepared.is_some()
            || self.goal_local.selection.is_some()
            || !self.goal_local.pending.is_empty()
        {
            return Err(
                StoreError::InvalidConditions("GoalLocal admission/inputs not retired").into(),
            );
        }
        self.goal_local.retired_prepared = None;
        self.goal_local.retired_selection = None;
        self.goal_local.attempts.clear();
        Ok(())
    }

    /// Declared retained backing, not an RSS/physical-sharing observation.
    pub fn rovent_goal_owned_host_bytes(&self) -> Option<u64> {
        self.goal_local.owned_host_bytes()
    }

    /// No current Missing descriptor is selected twice after a paid edge. A
    /// hot allocation/view failure retains its immutable preparation first.
    pub fn next_rovent_goal_input(
        &mut self,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<AdaptiveGoalLocalAdmission, PalsError> {
        if self.goal_local.admission_closed {
            return Err(RoleError::Canceled.into());
        }
        self.check_rovent_scan_live()?;
        if self
            .goal_local
            .attempts
            .values()
            .any(|attempt| attempt.failure.is_some())
        {
            return Ok(AdaptiveGoalLocalAdmission::Blocked {
                reason: GoalLocalBlockReason::PreviousAttemptFailed,
            });
        }
        let Some(frontier) = self.rovent_frontier() else {
            return Ok(AdaptiveGoalLocalAdmission::NoQuestion);
        };
        if matches!(
            frontier.snapshot().goal,
            crate::pals::rovent_frontier::FrontierGoalStatus::MateSatisfied
                | crate::pals::rovent_frontier::FrontierGoalStatus::DepthSatisfied
        ) {
            return Ok(AdaptiveGoalLocalAdmission::NoQuestion);
        }
        if !self.goal_local.pending.is_empty() {
            return Ok(AdaptiveGoalLocalAdmission::Blocked {
                reason: GoalLocalBlockReason::PendingCapacity,
            });
        }
        if self.goal_local.prepared.is_none() {
            let configured = u16::try_from(self.engine.config.line_plies)
                .map_err(|_| PalsError::InvalidConfig)?;
            let prepared = self.with_rovent_frontier(|frontier, stores| {
                if let Some(question) = frontier.unmet_horizon_questions(1, clock, now)?.pop() {
                    let first = question.original_first_move();
                    let prefix = frontier.unmet_horizon_question_prefix(&question)?;
                    let chunk =
                        frontier.unmet_horizon_question_chunk_limit(&question, configured)?;
                    let checked =
                        frontier.observe_unmet_horizon_anchor(&question, stores, clock, now)?;
                    return Ok(Some((
                        GoalLocalRootQuestion::Horizon(Arc::new(question)),
                        first,
                        prefix,
                        chunk,
                        checked,
                    )));
                }
                let Some(question) = frontier.missing_mate_questions(1, clock, now)?.pop() else {
                    return Ok(None);
                };
                let first = frontier.missing_question_first_move(&question)?;
                let prefix = frontier.missing_question_prefix(&question)?;
                let chunk = frontier.missing_question_chunk_limit(&question, configured)?;
                let checked = frontier.observe_missing_child(&question, clock, now)?;
                Ok(Some((
                    GoalLocalRootQuestion::Mate(Arc::new(question)),
                    first,
                    prefix,
                    chunk,
                    checked,
                )))
            })?;
            let Some(prepared) = prepared else {
                return Ok(AdaptiveGoalLocalAdmission::Blocked {
                    reason: GoalLocalBlockReason::Frontier(FrontierError::StoreUnavailable),
                });
            };
            let Some((question, first, prefix, chunk, checked)) = prepared else {
                return Ok(AdaptiveGoalLocalAdmission::NoQuestion);
            };
            let continuation_end = prefix
                .len()
                .checked_add(usize::from(chunk))
                .ok_or(PalsError::Capacity)?;
            let root_budget = self
                .rovent_frontier()
                .ok_or(FrontierError::ForeignRoot)?
                .registration()
                .root_budget;
            if checked.prefix() != prefix {
                return Err(FrontierError::ForeignHandle.into());
            }
            let snapshot = checked.endpoint_snapshot().clone();
            let source = Arc::new(GoalLocalRootSource {
                original_deadline: question.original_deadline(),
                question,
                owner: Arc::clone(&self.goal_local.owner),
                authority: self.authority,
                root_budget,
                hard_deadline: self.limits.deadline,
                first,
                imposed_prefix: prefix.clone(),
                imposed_snapshot: snapshot.clone(),
                root_side: self.engine.semantic.nodes[self.root]
                    .position
                    .side_to_move(),
            });
            self.goal_local.prepared = Some(PreparedGoalLocal {
                origin: Arc::new(GoalLocalOrigin::from_source(
                    source,
                    prefix,
                    snapshot,
                    continuation_end,
                )?),
                checked,
                node: None,
            });
        }
        self.prepare_goal_local_input(clock, now)
    }

    fn prepare_goal_local_input(
        &mut self,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<AdaptiveGoalLocalAdmission, PalsError> {
        if self.goal_local.pending.len() >= MAX_GOAL_LOCAL_PENDING {
            return Ok(AdaptiveGoalLocalAdmission::Blocked {
                reason: GoalLocalBlockReason::PendingCapacity,
            });
        }
        self.goal_local
            .pending
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        let mut held = self
            .goal_local
            .prepared
            .take()
            .ok_or(RoleError::InvalidOutput)?;
        if held.node.is_none() {
            match self.adopt_checked_frontier_line(&held.checked) {
                Ok(node) => held.node = Some(node),
                Err(error) => {
                    self.goal_local.prepared = Some(held);
                    return Err(error);
                }
            }
        }
        let node = held.node.ok_or(RoleError::InvalidOutput)?;
        if self.engine.semantic.nodes[node].terminal.is_some() {
            if let GoalLocalRootQuestion::Horizon(source) = &held.origin.source.question {
                self.with_rovent_frontier(|frontier, stores| {
                    frontier.close_goal_horizon_rules_origin(
                        source,
                        stores,
                        held.checked.path(),
                        clock,
                        now,
                    )
                })?
                .ok_or(FrontierError::ForeignRoot)?;
            }
            return Ok(AdaptiveGoalLocalAdmission::RulesAdvanced {
                key: held.origin.source_key(),
            });
        }
        if let Err(error) =
            self.with_rovent_frontier(|frontier, _| frontier.admit_role_input(clock, now))
        {
            self.goal_local.prepared = Some(held);
            return Ok(AdaptiveGoalLocalAdmission::Blocked {
                reason: GoalLocalBlockReason::Frontier(error),
            });
        }
        let lease = match self.review_lease_admission() {
            Ok(lease) => lease,
            Err(error) => {
                self.goal_local.prepared = Some(held);
                return Err(error);
            }
        };
        if !lease.may_prepare {
            self.goal_local.prepared = Some(held);
            return Ok(AdaptiveGoalLocalAdmission::Blocked {
                reason: GoalLocalBlockReason::ViewLeaseCapacity,
            });
        }
        let input = match self.input_goal_local(
            node,
            held.origin.stage(),
            held.origin.first_move(),
            held.origin.prefix().to_vec(),
            Arc::clone(&held.origin),
        ) {
            Ok(input) => input,
            Err(error) => {
                self.goal_local.prepared = Some(held);
                return Err(error);
            }
        };
        let input = GoalLocalInput {
            seal: Arc::new(GoalLocalInputSeal {
                input,
                origin: held.origin,
            }),
        };
        if self
            .goal_local
            .pending
            .insert(input.input().question_digest, input.clone())
            .is_some()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(AdaptiveGoalLocalAdmission::RoleInput(input))
    }

    fn validate_goal_local_input(&self, input: &GoalLocalInput) -> Result<(), PalsError> {
        let actual = input.input();
        let pending = self
            .goal_local
            .pending
            .get(&actual.question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        let frontier = self.rovent_frontier().ok_or(FrontierError::ForeignRoot)?;
        if !Arc::ptr_eq(&pending.seal, &input.seal)
            || !actual
                .question
                .goal_local
                .as_ref()
                .is_some_and(|origin| Arc::ptr_eq(origin, &input.seal.origin))
            || actual.view_seal.authority != self.authority
            || input.seal.origin.original_deadline() != self.deadline
        {
            return Err(RoleError::InvalidOutput.into());
        }
        AdaptivePortQuestion::validate_frontier_input(actual, &self.engine.semantic.stores)?;
        input.seal.origin.validates_input(
            actual,
            &self.goal_local.owner,
            self.authority,
            frontier.registration().root_budget,
        )?;
        let checked = self
            .existing_checked_frontier_line(&actual.question.context.prefix)?
            .ok_or(FrontierError::ForeignHandle)?;
        if !checked
            .endpoint_snapshot()
            .same_state(&actual.question.position.snapshot())
            || !input.seal.origin.source.imposed_snapshot.same_state(
                checked
                    .steps()
                    .get(input.seal.origin.source.imposed_prefix.len() - 1)
                    .ok_or(RoleError::InvalidOutput)?
                    .snapshot(),
            )
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }

    pub fn rovent_goal_consumption_recorded(
        &self,
        ticket: &RoleTicketEnvelope,
        view: &ReviewView,
    ) -> bool {
        self.goal_local
            .attempts
            .get(&ticket.id)
            .is_some_and(|attempt| {
                attempt.ticket == *ticket
                    && attempt.head_used
                    && attempt
                        .view
                        .upgrade()
                        .is_some_and(|original| std::ptr::eq(view, original.as_ref()))
            })
    }

    pub fn rovent_goal_has_pending(&self) -> bool {
        self.goal_local.prepared.is_some()
            || self.goal_local.selection.is_some()
            || !self.goal_local.pending.is_empty()
    }

    /// Merge these real semantic owners into the archive owner's existing
    /// protected-pin request. This creates no extra review lease and performs
    /// no archive I/O. A prepared snapshot with no hot node needs no invented
    /// Store id; its immutable checked path remains with the Frontier owner.
    pub(super) fn goal_local_protected_pins(&self) -> Result<StorePins, PalsError> {
        let mut pins = StorePins::default();
        let stores = &self.engine.semantic.stores;
        for prepared in [
            self.goal_local.prepared.as_ref(),
            self.goal_local.retired_prepared.as_ref(),
            self.goal_local
                .selection
                .as_ref()
                .and_then(|selection| selection.prepared.as_ref()),
            self.goal_local
                .retired_selection
                .as_ref()
                .and_then(|selection| selection.prepared.as_ref()),
        ]
        .into_iter()
        .flatten()
        {
            let Some(node) = prepared.node else {
                continue;
            };
            let node = self
                .engine
                .semantic
                .nodes
                .get(node)
                .ok_or(PalsError::InvalidConfig)?;
            let situation = stores.situations.get(node.situation)?;
            if situation.state != node.state
                || !stores
                    .states
                    .get(node.state)?
                    .same_state(&node.position.snapshot())
            {
                return Err(RoleError::InvalidOutput.into());
            }
            pins.states.insert(node.state);
            pins.situations.insert(node.situation);
            pins.lines.insert(situation.focus);
        }
        for input in self.goal_local.pending.values() {
            let context = &input.input().question.context;
            pins.states.insert(context.state);
            pins.situations.insert(context.situation);
            pins.lines.insert(context.focus);
        }
        Ok(pins)
    }

    /// Only discard the exact original logical input. Native output/workspace
    /// custody is still retained by the real manager/service until its fence.
    /// The already paid Rules edge is neither revoked nor refunded here.
    pub fn discard_rovent_goal_input(&mut self, input: &GoalLocalInput) -> Result<(), PalsError> {
        let original = self
            .goal_local
            .pending
            .get(&input.input().question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        if !Arc::ptr_eq(&original.seal, &input.seal) {
            return Err(RoleError::InvalidOutput.into());
        }
        self.goal_local
            .pending
            .remove(&input.input().question_digest);
        Ok(())
    }

    fn validate_goal_local_output(
        &self,
        input: &GoalLocalInput,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
    ) -> Result<(), PalsError> {
        let actual = input.input();
        let ticket = accepted.ticket();
        self.verify_registered_model_namespace(ticket, output)?;
        if !accepted.validates_output(output)
            || ticket.authority != self.authority
            || ticket.question != actual.question_digest
            || ticket.original_deadline != self.deadline
            || ticket.view != actual.view_seal
            || ticket.representation != output.key
            || ticket.role != role_for_purpose(actual.question.context.purpose)
            || output.id != ticket.id
            || output.authority != ticket.authority
            || output.key.record_revision != actual.question.context.public_revision
        {
            return Err(RoleError::InvalidOutput.into());
        }
        validate_payload(actual.question.context.purpose, &actual.view, output)
    }

    /// Preserve the exact selected move before requesting common credit. A
    /// NodesReserved wait leaves this original retired head unconsumed, and a
    /// retry uses the same choice and paid line without replaying Rules.
    #[allow(clippy::too_many_arguments)]
    fn prepare_goal_local_selection(
        &mut self,
        input: &GoalLocalInput,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), PalsError> {
        if input.input().question.context.purpose == RoleQueryPurpose::ValueFresh {
            return Ok(());
        }
        if let Some(selected) = &self.goal_local.selection {
            if !selected.input.ptr_eq(&Arc::downgrade(&input.seal))
                || selected.ticket != *accepted.ticket()
                || selected.output != accepted.output_digest()
                || selected.physical != physical.into()
            {
                return Err(RoleError::InvalidOutput.into());
            }
            if selected.prepared.is_some() {
                return Ok(());
            }
        } else {
            let movement = *role_commit::rank_moves(&input.input().view.query().legal, output)?
                .first()
                .ok_or(RoleError::InvalidOutput)?;
            let mut prefix = input.seal.origin.prefix.clone();
            prefix.try_reserve(1).map_err(|_| PalsError::Capacity)?;
            prefix.push(movement);
            self.goal_local.selection = Some(GoalLocalSelection {
                input: Arc::downgrade(&input.seal),
                ticket: *accepted.ticket(),
                output: accepted.output_digest(),
                physical: physical.into(),
                prefix,
                prepared: None,
            });
        }
        let mut selection = self
            .goal_local
            .selection
            .take()
            .ok_or(RoleError::InvalidOutput)?;
        let prepared = (|| {
            let checked = self
                .with_rovent_frontier(|frontier, stores| {
                    // Already paid prefixes are read before new-work admission.
                    // Exact budget fill cannot make that lookup a fresh visit.
                    let tick = frontier.original_tick_after_elapsed(clock, now, Duration::ZERO)?;
                    if frontier
                        .registration()
                        .original_deadline
                        .accepts(clock, tick)
                        .is_err()
                    {
                        return Err(FrontierError::Deadline);
                    }
                    match frontier.existing_rules_prefix(&selection.prefix, stores)? {
                        Some(checked) => Ok(checked),
                        None => {
                            frontier.observe_rules_prefix(&selection.prefix, stores, clock, tick)
                        }
                    }
                })?
                .ok_or(FrontierError::StoreUnavailable)?;
            let origin = Arc::new(GoalLocalOrigin::from_source(
                Arc::clone(&input.seal.origin.source),
                selection.prefix.clone(),
                checked.endpoint_snapshot().clone(),
                input.seal.origin.continuation_end,
            )?);
            Ok::<_, PalsError>(PreparedGoalLocal {
                origin,
                checked,
                node: None,
            })
        })();
        match prepared {
            Ok(prepared) => {
                selection.prepared = Some(prepared);
                self.goal_local.selection = Some(selection);
                Ok(())
            }
            Err(error) => {
                self.goal_local.selection = Some(selection);
                Err(error)
            }
        }
    }

    /// The actual native owner/registry issued AcceptedRoleStep after its
    /// permit/physical retirement. No fake refinement/peer frame is required.
    #[allow(clippy::too_many_arguments)]
    pub fn consume_rovent_goal_input(
        &mut self,
        input: &GoalLocalInput,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<GoalLocalStep, PalsError> {
        let actual = input.input();
        self.validate_goal_local_output(input, accepted, output)?;
        if let Some(attempt) = self.goal_local.attempts.get(&accepted.ticket().id) {
            AdaptivePortQuestion::validate_frontier_input(actual, &self.engine.semantic.stores)?;
            if attempt.ticket != *accepted.ticket()
                || attempt.output != accepted.output_digest()
                || attempt.physical != GoalLocalPhysicalSeal::from(physical)
                || !attempt
                    .question
                    .upgrade()
                    .is_some_and(|original| Arc::ptr_eq(&original, &actual.question))
                || !attempt
                    .view
                    .upgrade()
                    .is_some_and(|original| Arc::ptr_eq(&original, &actual.view))
            {
                return Err(RoleError::InvalidOutput.into());
            }
            if let Some(error) = &attempt.failure {
                return Err(error.clone());
            }
            return attempt
                .completed
                .as_ref()
                .ok_or(RoleError::InvalidOutput)?
                .read();
        }
        self.check_rovent_scan_live()?;
        self.validate_goal_local_input(input)?;
        if self.accepted.contains(&accepted.ticket().id)
            || self
                .committed_role_questions
                .contains_key(&accepted.ticket().id)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if self.counters.role_calls >= self.engine.config.max_role_calls {
            return Err(PalsError::RoleCallLimit);
        }
        // Check Fresh numerical/physical namespace before reserving one
        // semantic attempt. A value is not a Rules mate proof.
        if actual.question.context.purpose == RoleQueryPurpose::ValueFresh {
            self.checked_value_input(actual, accepted.ticket(), output, Some(physical))?;
        }
        if self.goal_local.admission_closed {
            return Err(RoleError::Canceled.into());
        }
        self.with_rovent_frontier(|frontier, stores| {
            frontier.admit_existing_consumption(stores, clock, now)
        })?
        .ok_or(FrontierError::StoreUnavailable)?;
        self.prepare_goal_local_selection(input, accepted, output, physical, clock, now)?;
        self.goal_local.attempts.retain(|_, entry| {
            entry.question.strong_count() != 0
                || entry.view.strong_count() != 0
                || entry.failure.is_some()
        });
        if self.goal_local.attempts.len() >= MAX_GOAL_LOCAL_ATTEMPTS {
            return Err(PalsError::Capacity);
        }
        self.goal_local
            .attempts
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        self.goal_local.attempts.insert(
            accepted.ticket().id,
            GoalLocalAttempt {
                ticket: *accepted.ticket(),
                output: accepted.output_digest(),
                physical: physical.into(),
                question: Arc::downgrade(&actual.question),
                view: Arc::downgrade(&actual.view),
                head_used: false,
                view_consumed: false,
                failure: None,
                completed: None,
            },
        );
        let returned = self.consume_goal_local_once(input, accepted, output, physical, clock, now);
        match &returned {
            Ok(step) => {
                let cached = match step {
                    GoalLocalStep::ContinuePrepared { key } => {
                        CachedGoalLocalStep::ContinuePrepared { key: *key }
                    }
                    GoalLocalStep::Leaf(leaf) => CachedGoalLocalStep::Leaf(leaf.clone()),
                };
                self.goal_local
                    .attempts
                    .get_mut(&accepted.ticket().id)
                    .ok_or(RoleError::InvalidOutput)?
                    .completed = Some(cached);
                self.goal_local.pending.remove(&actual.question_digest);
            }
            Err(error) => {
                // Preserve the exact failed attempt. A partial hot commit or
                // a used head is never retried as a new free computation.
                self.goal_local
                    .attempts
                    .get_mut(&accepted.ticket().id)
                    .ok_or(RoleError::InvalidOutput)?
                    .failure = Some(error.clone());
            }
        }
        returned
    }

    #[allow(clippy::too_many_arguments)]
    fn consume_goal_local_once(
        &mut self,
        input: &GoalLocalInput,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<GoalLocalStep, PalsError> {
        let started = Instant::now();
        let actual = input.input();
        let ticket = accepted.ticket();
        self.with_rovent_frontier(|frontier, stores| {
            let tick = frontier.original_tick_after_elapsed(clock, now, started.elapsed())?;
            frontier.admit_existing_consumption(stores, clock, tick)
        })?
        .ok_or(FrontierError::StoreUnavailable)?;
        self.accepted.insert(ticket.id);
        let old_question = self
            .committed_role_questions
            .insert(ticket.id, ticket.question);
        if let Err(error) = self.role_consumed(ticket, &actual.view) {
            self.accepted.remove(&ticket.id);
            match old_question {
                Some(old) => {
                    self.committed_role_questions.insert(ticket.id, old);
                }
                None => {
                    self.committed_role_questions.remove(&ticket.id);
                }
            }
            return Err(error);
        }
        let attempt = self
            .goal_local
            .attempts
            .get_mut(&ticket.id)
            .ok_or(RoleError::InvalidOutput)?;
        attempt.view_consumed = true;
        attempt.head_used = true;
        self.counters.role_calls += 1;
        self.counters.consumed_role_outputs += 1;
        match actual.question.context.purpose {
            RoleQueryPurpose::ValueFresh => {
                self.counters.value_calls += 1;
                self.counters.completed_value_calls += 1;
                self.counters.accepted_value_outputs += 1;
            }
            RoleQueryPurpose::ReplyPolicy => {
                self.counters.critic_calls += 1;
                self.counters.completed_critic_calls += 1;
                self.counters.accepted_critic_outputs += 1;
            }
            _ => {
                self.counters.proposer_calls += 1;
                self.counters.completed_proposer_calls += 1;
                self.counters.accepted_proposer_outputs += 1;
            }
        }
        if actual.question.context.purpose == RoleQueryPurpose::ValueFresh {
            self.commit_value_input(actual, ticket, output, Some(physical))?;
            let horizon_acceptance = if matches!(
                &input.seal.origin.source.question,
                GoalLocalRootQuestion::Horizon(_)
            ) {
                Some(
                    self.with_rovent_frontier(|frontier, stores| {
                        let checked = frontier
                            .existing_rules_prefix(&actual.question.context.prefix, stores)?
                            .ok_or(FrontierError::ForeignHandle)?;
                        let tick =
                            frontier.original_tick_after_elapsed(clock, now, started.elapsed())?;
                        frontier.accept_goal_horizon_fresh(
                            input,
                            stores,
                            checked.path(),
                            accepted,
                            output,
                            physical,
                            clock,
                            tick,
                        )
                    })?
                    .ok_or(FrontierError::ForeignRoot)?,
                )
            } else {
                None
            };
            return Ok(GoalLocalStep::Leaf(GoalLocalLeaf {
                origin: Arc::clone(&input.seal.origin),
                snapshot: actual.question.position.snapshot(),
                kind: GoalLocalLeafKind::FreshBoundary,
                horizon_acceptance,
            }));
        }
        let mut selection = self
            .goal_local
            .selection
            .take()
            .ok_or(RoleError::InvalidOutput)?;
        let prepared = selection.prepared.take().ok_or(RoleError::InvalidOutput)?;
        let origin = Arc::clone(&prepared.origin);
        let snapshot = prepared.checked.endpoint_snapshot().clone();
        // Selection and common quota were resolved before head/view use.
        // The exact paid path now moves into continuation preparation.
        self.goal_local.prepared = Some(prepared);
        let held = self
            .goal_local
            .prepared
            .take()
            .ok_or(RoleError::InvalidOutput)?;
        let node = match self.adopt_checked_frontier_line(&held.checked) {
            Ok(node) => node,
            Err(error) => {
                self.goal_local.prepared = Some(held);
                return Err(error);
            }
        };
        if self.engine.semantic.nodes[node].terminal.is_some() {
            let horizon_acceptance = if let GoalLocalRootQuestion::Horizon(source) =
                &origin.source.question
            {
                Some(
                    self.with_rovent_frontier(|frontier, stores| {
                        let tick =
                            frontier.original_tick_after_elapsed(clock, now, started.elapsed())?;
                        frontier.close_goal_horizon_rules_origin(
                            source,
                            stores,
                            held.checked.path(),
                            clock,
                            tick,
                        )
                    })?
                    .ok_or(FrontierError::ForeignRoot)?,
                )
            } else {
                None
            };
            return Ok(GoalLocalStep::Leaf(GoalLocalLeaf {
                origin,
                snapshot,
                kind: GoalLocalLeafKind::RulesTerminal,
                horizon_acceptance,
            }));
        }
        self.goal_local.prepared = Some(PreparedGoalLocal {
            node: Some(node),
            ..held
        });
        // The manager may retire the original view before preparing the next
        // one, so the global Store lease16 bound remains sufficient.
        Ok(GoalLocalStep::ContinuePrepared {
            key: origin.source_key(),
        })
    }
}

#[cfg(test)]
mod tests;
