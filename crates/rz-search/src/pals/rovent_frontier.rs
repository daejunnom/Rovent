//! Goal consumption for the real asynchronous single-writer manager.
//!
//! This module runs no role model, DFS, CPU search, archive I/O or external
//! engine. The manager declares actual frame responsibilities, commits checked
//! Store lines, and supplies its original native-owner acceptance/accounting
//! receipts. Missing mate defenses are questions for that same manager queue.

use super::coordinator::{AcceptedRoleStep, CoordinatorWork, FrameId, FrameInput, RefinementStage};
use super::engine::RoleQueryPurpose;
use super::engine::adaptive_port::{
    AdaptiveModelNamespace, AdaptivePortQuestion, AdaptivePortRoleEvidence,
};
use super::engine::pc_port::{PcPortQuestion, PcResponsibilitySource};
use super::review_context::ReviewView;
use super::rovent_engine::{AdmittedRoleQuestion, ROVENT_NODE_UNIT, RootMoveDomain};
use super::rovent_goals::{
    HorizonBoundary, HorizonError, HorizonInterruption, HorizonObligationId, HorizonObligationKind,
    HorizonSnapshot, RootPlyAnchor, SelectiveHorizon,
};
use super::rovent_solved::{
    MateConclusion, MateNodeId, MateProofError, MateProofStore, RootMateConclusion,
};
use super::store::{
    LineId, PalsStores, StateId, StoreError, StorePins, StoreReadOwner, StoreReadPin,
};
use super::value::ModelValueOutput;
use crate::cpu_worker::{
    CpuLedgerIdentity, CpuSliceReport, ReleasedCpuReservation, SettledCpuSlice,
};
use rz_contracts::pals::{Role, RoleOutput, RolePayload, SearchAuthority};
use rz_contracts::pals_adaptive::{CpuJobEnvelope, SliceGrant};
use rz_contracts::pals_adaptive::{ReviewViewSeal, RoleTicketEnvelope};
use rz_contracts::rovent::{ChessPly, ExecutionMode, SearchGoal};
use rz_contracts::{ClockDomain, ContractError, Deadline, Digest, MonotonicTick};
use rz_position::contracts::ContractState;
use rz_position::{BoardMove, Color, PlayStatus, Position, PositionError, PositionSnapshot};
use std::collections::HashMap;
use std::num::NonZeroU16;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

mod external;
mod horizon_questions;
pub use external::{FrontierExternalAcceptance, FrontierExternalReservation, FrontierExternalWork};
mod question;
pub use super::rovent_solved::RootMateStrategy as FrontierMateStrategy;
pub use horizon_questions::{
    FrontierHorizonAcceptance, UnmetHorizonQuestion, UnmetHorizonQuestionKey,
};
pub(crate) use question::VerifiedFrontierQuestion;
use question::WeakQuestionOwner;

static NEXT_FRONTIER_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
pub struct FrontierLimits {
    pub maximum_goal_plies: NonZeroU16,
    pub maximum_frames: usize,
    pub maximum_paths: usize,
    pub maximum_mate_nodes: usize,
    pub maximum_native_receipts: usize,
}

pub struct FrontierRegistration {
    pub authority: SearchAuthority,
    pub mode: ExecutionMode,
    pub original_deadline: Deadline,
    pub root_budget: Digest,
    pub goal: SearchGoal,
    pub nodes: Option<u64>,
    pub selective_policy: Digest,
    pub root_rules: Arc<ContractState>,
    pub root_moves: Arc<RootMoveDomain>,
    pub root_state: StateId,
    pub model_namespace: Arc<AdaptiveModelNamespace>,
    /// The actual native ledger instance. Absent in P/C-only assembly, or in
    /// the separate opaque-source constructor for profiled external R.
    pub native_ledger: Option<CpuLedgerIdentity>,
}

/// Borrowed actual port assembly. Matching a digest alone cannot replace the
/// original Rules, root restriction, Store lease or native ledger owners.
#[derive(Clone, Copy)]
pub struct FrontierAttachment<'a> {
    pub authority: SearchAuthority,
    pub mode: ExecutionMode,
    pub original_deadline: Deadline,
    pub root_budget: Digest,
    pub root_rules: &'a Arc<ContractState>,
    pub root_moves: &'a Arc<RootMoveDomain>,
    pub root_state: StateId,
    pub model_namespace: &'a AdaptiveModelNamespace,
    pub native_ledger: Option<&'a CpuLedgerIdentity>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontierError {
    Invalid(&'static str),
    Capacity(&'static str),
    /// The original semantic Store refused another retained public record.
    /// This does not cover pin, archive, allocation or physical-owner failures.
    PublicRecordCapacity,
    /// The original selected-review producer refused its per-bundle record cap.
    /// Generic index, view, sequence, owner and Store errors remain distinct.
    PublicReviewIndexCapacity,
    ForeignRoot,
    ForeignHandle,
    Duplicate,
    PolicySealed,
    DepthHorizon,
    NodeBudget,
    /// Capacity is currently reserved by admitted native work. Wait for its
    /// real receipt; this is not a spent-budget stop or permission to refund.
    NodesReserved,
    NativeUnknown,
    MissingAcceptance,
    Deadline,
    Revoked,
    StoreUnavailable,
    Horizon(HorizonError),
    Mate(MateProofError),
}
impl std::fmt::Display for FrontierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rovent frontier: {self:?}")
    }
}
impl std::error::Error for FrontierError {}
impl From<HorizonError> for FrontierError {
    fn from(error: HorizonError) -> Self {
        Self::Horizon(error)
    }
}
impl From<MateProofError> for FrontierError {
    fn from(error: MateProofError) -> Self {
        Self::Mate(error)
    }
}
impl From<StoreError> for FrontierError {
    fn from(_: StoreError) -> Self {
        Self::StoreUnavailable
    }
}
impl From<PositionError> for FrontierError {
    fn from(_: PositionError) -> Self {
        Self::Invalid("Rules path")
    }
}
impl From<ContractError> for FrontierError {
    fn from(_: ContractError) -> Self {
        Self::Invalid("clock or shared contract")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FrontierPath {
    owner: u64,
    slot: usize,
}

/// One immutable root-relative path checked and charged by this controller.
/// This is a process-local authority, never a serialized snapshot declaration.
pub(crate) struct CheckedFrontierLine {
    owner: u64,
    source: Weak<CheckedFrontierSource>,
    store_owner: StoreReadOwner,
    authority: SearchAuthority,
    root_budget: Digest,
    root_snapshot: PositionSnapshot,
    path: FrontierPath,
    prefix: Vec<BoardMove>,
    steps: Vec<CheckedFrontierStep>,
}

/// One live controller owns this source and its current pin. Paid snapshots
/// retain only Weak references, so they neither extend root lifetime nor add
/// a review lease. Replacing the pin updates this same source identity.
struct CheckedFrontierSource {
    owner: u64,
    authority: SearchAuthority,
    root_budget: Digest,
    store_owner: StoreReadOwner,
    pin: Mutex<Weak<StoreReadPin>>,
    closed: AtomicBool,
}

pub(crate) struct CheckedFrontierStep {
    movement: BoardMove,
    snapshot: PositionSnapshot,
}
impl CheckedFrontierStep {
    pub(crate) fn movement(&self) -> BoardMove {
        self.movement
    }
    pub(crate) fn snapshot(&self) -> &PositionSnapshot {
        &self.snapshot
    }
}
impl CheckedFrontierLine {
    pub(crate) fn matches_store(&self, stores: &PalsStores) -> bool {
        let Some(source) = self.source.upgrade() else {
            return false;
        };
        source.owner == self.owner
            && source.authority == self.authority
            && source.root_budget == self.root_budget
            && !source.closed.load(Ordering::Acquire)
            && source.store_owner.matches(stores)
            && self.store_owner.matches(stores)
            && self.authority.root.0 == stores.generation()
            && source
                .pin
                .lock()
                .is_ok_and(|pin| pin.upgrade().is_some_and(|pin| pin.validates(stores)))
            && !source.closed.load(Ordering::Acquire)
    }
    /// Declared owned backing excluding allocator overhead and physical
    /// sharing. Used by retained semantic custody, never a measured peak.
    pub(crate) fn owned_host_bytes(&self) -> Option<u64> {
        let mut bytes = std::mem::size_of::<Self>()
            .checked_add(
                self.prefix
                    .capacity()
                    .checked_mul(std::mem::size_of::<BoardMove>())?,
            )?
            .checked_add(
                self.steps
                    .capacity()
                    .checked_mul(std::mem::size_of::<CheckedFrontierStep>())?,
            )?
            .checked_add(self.root_snapshot.retained_history_bytes()?)?;
        for step in &self.steps {
            bytes = bytes.checked_add(step.snapshot.retained_history_bytes()?)?;
        }
        u64::try_from(bytes).ok()
    }
    pub(crate) fn matches(&self, frontier: &RoventFrontier, stores: &PalsStores) -> bool {
        self.owner == frontier.owner
            && self
                .source
                .ptr_eq(&Arc::downgrade(&frontier.checked_source))
            && self.authority == frontier.registration.authority
            && self.root_budget == frontier.registration.root_budget
            && self.store_owner.matches(stores)
            && frontier.store_owner.matches(stores)
            && frontier.pin.validates(stores)
            && self.authority.root.0 == stores.generation()
            && self
                .root_snapshot
                .same_state(frontier.registration.root_rules.rules().snapshot())
            && frontier
                .path_snapshot(self.path)
                .is_ok_and(|snapshot| snapshot.same_state(self.endpoint_snapshot()))
    }
    pub(crate) fn root_snapshot(&self) -> &PositionSnapshot {
        &self.root_snapshot
    }
    pub(crate) fn prefix(&self) -> &[BoardMove] {
        &self.prefix
    }
    pub(crate) fn steps(&self) -> &[CheckedFrontierStep] {
        &self.steps
    }
    pub(crate) fn path(&self) -> FrontierPath {
        self.path
    }
    pub(crate) fn endpoint_snapshot(&self) -> &PositionSnapshot {
        self.steps
            .last()
            .map_or(&self.root_snapshot, |step| &step.snapshot)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrontierObligation {
    owner: u64,
    slot: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontierEndpointAcceptance {
    Accepted,
    AlreadyAcceptedExact,
}

struct CompletionScope {
    question: WeakQuestionOwner,
    view: Weak<ReviewView>,
    frame: FrameId,
    stage: RefinementStage,
    first: rz_contracts::pals::Move16,
    original_deadline: Deadline,
    question_digest: Digest,
    view_seal: ReviewViewSeal,
    public_context: Digest,
    endpoint: FrontierPath,
}
impl CompletionScope {
    fn new(question: &VerifiedFrontierQuestion<'_>, endpoint: FrontierPath) -> Self {
        Self {
            question: question.weak_owner(),
            view: Arc::downgrade(question.view),
            frame: question.frame,
            stage: question.stage,
            first: question.original_first_move,
            original_deadline: question.original_deadline,
            question_digest: question.question_digest,
            view_seal: question.view_seal,
            public_context: question.public_context,
            endpoint,
        }
    }
    fn matches(&self, question: &VerifiedFrontierQuestion<'_>, endpoint: FrontierPath) -> bool {
        self.question.matches(question)
            && self.view.ptr_eq(&Arc::downgrade(question.view))
            && self.frame == question.frame
            && self.stage == question.stage
            && self.first == question.original_first_move
            && self.original_deadline == question.original_deadline
            && self.question_digest == question.question_digest
            && self.view_seal == question.view_seal
            && self.public_context == question.public_context
            && self.endpoint == endpoint
    }
}
// This bounded completion seal retains the exact ticket and physical evidence
// inline; sealing completion cannot introduce an unbudgeted heap allocation.
#[allow(clippy::large_enum_variant)]
enum FrontierCompletionSeal {
    Rules {
        scope: CompletionScope,
    },
    Fresh {
        scope: CompletionScope,
        ticket: RoleTicketEnvelope,
        output: Digest,
        physical: FreshPhysicalSeal,
    },
}
#[derive(Clone, Copy, Eq, PartialEq)]
struct FreshPhysicalSeal {
    execution: Option<rz_contracts::ExecutionId>,
    completed_nn_inputs: Option<u64>,
    measurement_complete: bool,
    physical_fresh: Option<bool>,
}
impl From<AdaptivePortRoleEvidence> for FreshPhysicalSeal {
    fn from(value: AdaptivePortRoleEvidence) -> Self {
        Self {
            execution: value.execution,
            completed_nn_inputs: value.completed_nn_inputs,
            measurement_complete: value.measurement_complete,
            physical_fresh: value.physical_fresh,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FrontierFrameKind {
    Candidate,
    Counter,
    Repair,
}
impl FrontierFrameKind {
    fn horizon_kind(self) -> HorizonObligationKind {
        match self {
            Self::Candidate => HorizonObligationKind::Candidate,
            Self::Counter => HorizonObligationKind::Counter,
            Self::Repair => HorizonObligationKind::Repair,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeCloseReason {
    Executed,
    OwnerRejectedBeforeWork,
    RejectedBeforeEnqueue,
}

/// Linear manager custody of one original SliceGrant. The authoritative
/// reservation remains in this controller even when a worker fails or drops.
#[derive(Debug)]
pub struct NativeReservation {
    owner: u64,
    slot: usize,
    grant: SliceGrant,
}
impl NativeReservation {
    pub fn grant(&self) -> SliceGrant {
        self.grant
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativePhase {
    Reserved,
    Settled(NativeCloseReason),
    Quarantined,
}
struct NativeAccount {
    grant: SliceGrant,
    phase: NativePhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontierStop {
    Canceled,
    Deadline,
    WorkBudget,
    /// Registered logical role-call admissions, independent of `go nodes`.
    RoleBudget,
    Capacity,
    NativeUnknown,
    EvidenceReclaimed,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontierGoalStatus {
    Pending,
    DepthSatisfied,
    MateSatisfied,
    /// No depth/mate was requested. The explicitly sealed selective policy's
    /// nonempty responsibility journal is complete; this proves neither a
    /// depth/mate goal nor the native owners' separate physical fence.
    SelectiveFrontier,
    GoalUnresolved,
    Stopped(FrontierStop),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FrontierNodeWork {
    pub root_entries: u64,
    pub checked_transitions: u64,
    pub native_nodes: u64,
    pub native_qsearch_nodes: u64,
    pub reserved_native_nodes: u64,
    pub outstanding_native_executions: usize,
    pub native_usage_known: bool,
}
impl FrontierNodeWork {
    pub fn total_node_visits(self) -> Option<u64> {
        self.root_entries
            .checked_add(self.checked_transitions)?
            .checked_add(self.native_nodes)
    }
}

#[derive(Clone, Debug)]
pub struct FrontierSnapshot {
    pub authority: SearchAuthority,
    pub original_deadline: Deadline,
    pub node_unit: &'static str,
    pub requested_nodes: Option<u64>,
    pub work: FrontierNodeWork,
    /// Engine-reported foreign work, never calibrated to position visits.
    pub external_work: Option<FrontierExternalWork>,
    pub horizon: HorizonSnapshot,
    pub mate: [Option<RootMateConclusion>; 2],
    pub mate_strategy: Option<FrontierMateStrategy>,
    pub goal: FrontierGoalStatus,
    pub registered_frames: usize,
    pub completed_frames: usize,
    pub selection_sealed: bool,
}

struct PathEntry {
    parent: Option<usize>,
    movement: Option<BoardMove>,
    anchor: RootPlyAnchor,
}
struct FrameEntry {
    frame: FrameId,
    kind: FrontierFrameKind,
    original_first: BoardMove,
    question: Digest,
    source_path: FrontierPath,
    pc_source: bool,
    horizon: Option<HorizonObligationId>,
    finished: bool,
    completion: Option<FrontierCompletionSeal>,
}
struct MateLane {
    proof: MateProofStore,
    paths: HashMap<MateNodeId, FrontierPath>,
    ordered: Vec<(MateNodeId, FrontierPath)>,
}

/// Read-only request descriptor. Dispatch and duplicate-question ownership stay
/// in the manager's existing queue. This descriptor grants no new node budget.
pub struct MissingMateQuestion {
    owner: u64,
    lane: usize,
    node: MateNodeId,
    anchor: FrontierPath,
    movement: BoardMove,
    attacker: Color,
    remaining: ChessPly,
    original_deadline: Deadline,
    kind: MateQuestionKind,
}
/// Exact owner/action tuple for the existing manager's bounded pending set.
/// A copied key grants no dispatch, credit, state transition or completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct MissingMateQuestionKey {
    owner: u64,
    lane: usize,
    node: MateNodeId,
    movement: BoardMove,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MateQuestionKind {
    AttackCandidate,
    RequiredDefense,
}
impl MissingMateQuestion {
    pub fn key(&self) -> MissingMateQuestionKey {
        MissingMateQuestionKey {
            owner: self.owner,
            lane: self.lane,
            node: self.node,
            movement: self.movement,
        }
    }
    pub fn kind(&self) -> MateQuestionKind {
        self.kind
    }
    pub fn anchor(&self) -> FrontierPath {
        self.anchor
    }
    pub fn movement(&self) -> BoardMove {
        self.movement
    }
    pub fn attacker(&self) -> Color {
        self.attacker
    }
    pub fn remaining(&self) -> ChessPly {
        self.remaining
    }
    pub fn original_deadline(&self) -> Deadline {
        self.original_deadline
    }
}

pub struct RoventFrontier {
    owner: u64,
    registration: FrontierRegistration,
    limits: FrontierLimits,
    horizon: SelectiveHorizon,
    paths: Vec<PathEntry>,
    path_index: HashMap<(usize, BoardMove), usize>,
    frames: Vec<FrameEntry>,
    mate: [Option<MateLane>; 2],
    native: Vec<NativeAccount>,
    external_source: Option<crate::external_cpu::ProfiledExternalRootSource>,
    external: external::ExternalFrontierJournal,
    work: FrontierNodeWork,
    selection_sealed: bool,
    stop: Option<FrontierStop>,
    // Internal cause only; absent from FrontierSnapshot and historical schemas.
    public_record_capacity_refused: bool,
    // Only the actual per-bundle review publication bound sets this cause.
    public_review_index_capacity_refused: bool,
    // WorkBudget remains the first reported stop, but later revoke/unknown
    // still closes consumption of an already paid in-flight selection.
    consumption_stop: Option<FrontierStop>,
    pins: StorePins,
    pin: Arc<StoreReadPin>,
    checked_source: Arc<CheckedFrontierSource>,
    store_owner: StoreReadOwner,
    original_clock: ClockDomain,
    original_entry_tick: MonotonicTick,
    original_entry_instant: Instant,
}

impl RoventFrontier {
    pub fn new(
        registration: FrontierRegistration,
        limits: FrontierLimits,
        stores: &PalsStores,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Self, FrontierError> {
        Self::new_inner(registration, limits, stores, None, clock, now)
    }

    /// Separate assembly for an actual profiled external ResearchChecker.
    /// The existing own-R and R-free constructor keeps its original guard.
    pub fn new_external(
        registration: FrontierRegistration,
        limits: FrontierLimits,
        stores: &PalsStores,
        source: &crate::external_cpu::ProfiledExternalRootSource,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Self, FrontierError> {
        Self::new_inner(
            registration,
            limits,
            stores,
            Some(source.clone()),
            clock,
            now,
        )
    }

    fn new_inner(
        registration: FrontierRegistration,
        limits: FrontierLimits,
        stores: &PalsStores,
        external_source: Option<crate::external_cpu::ProfiledExternalRootSource>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Self, FrontierError> {
        let original_entry_instant = Instant::now();
        if limits.maximum_goal_plies.get() > 256
            || limits.maximum_frames == 0
            || limits.maximum_frames > 1024
            || limits.maximum_paths == 0
            || limits.maximum_paths > 65_536
            || limits.maximum_mate_nodes == 0
            || limits.maximum_mate_nodes > 65_536
            || limits.maximum_native_receipts == 0
            || limits.maximum_native_receipts > 65_536
        {
            return Err(FrontierError::Invalid("finite frontier limits"));
        }
        registration.goal.validate(limits.maximum_goal_plies)?;
        registration.original_deadline.accepts(clock, now)?;
        if registration.original_deadline.clock.0 != registration.authority.epoch
            || registration.root_moves.root_identity()
                != registration.root_rules.snapshot().identity()
            || !stores
                .states
                .get(registration.root_state)?
                .same_state(registration.root_rules.rules().snapshot())
            || stores.generation() != registration.authority.root.0
            || (external_source.is_none()
                && registration.mode.has_native_r() != registration.native_ledger.is_some())
        {
            return Err(FrontierError::ForeignRoot);
        }
        if let Some(source) = &external_source {
            external::validate_external_registration(&registration, source, clock)?;
        }
        let root = Position::from_snapshot(
            registration.root_rules.rules().snapshot(),
            Default::default(),
        )?;
        let mut horizon = SelectiveHorizon::new(
            &root,
            registration.goal,
            registration.selective_policy,
            limits.maximum_frames,
        )?;
        let owner = NEXT_FRONTIER_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| FrontierError::Capacity("frontier owner"))?;
        let mut pins = StorePins::default();
        pins.states.insert(registration.root_state);
        let pin = Arc::new(stores.pin_review(pins.clone())?);
        let checked_source = Arc::new(CheckedFrontierSource {
            owner,
            authority: registration.authority,
            root_budget: registration.root_budget,
            store_owner: stores.read_owner(),
            pin: Mutex::new(Arc::downgrade(&pin)),
            closed: AtomicBool::new(false),
        });
        let mut mate: [Option<MateLane>; 2] = [None, None];
        for (lane, attacker) in [root.side_to_move(), root.side_to_move().opposite()]
            .into_iter()
            .enumerate()
        {
            // Passive coverage never reaches a synthetic horizon boundary:
            // every admitted Rules path is shorter than this sentinel. Only
            // actual Rules terminals/covered choices can close a passive node.
            let remaining = registration
                .goal
                .mate_horizon(lane == 0)?
                .unwrap_or(ChessPly(
                    limits
                        .maximum_goal_plies
                        .get()
                        .checked_add(1)
                        .ok_or(FrontierError::DepthHorizon)?,
                ));
            {
                let root_moves = if registration.root_moves.allowed().is_empty() {
                    None
                } else {
                    Some(registration.root_moves.allowed().as_ref())
                };
                let proof = MateProofStore::new(
                    &root,
                    attacker,
                    remaining,
                    root_moves,
                    limits.maximum_mate_nodes,
                )?;
                let mut paths = HashMap::new();
                let root_node = proof.root();
                let root_path = FrontierPath { owner, slot: 0 };
                paths.insert(root_node, root_path);
                mate[lane] = Some(MateLane {
                    proof,
                    paths,
                    ordered: vec![(root_node, root_path)],
                });
            }
        }
        let stop = (registration.nodes == Some(0)).then_some(FrontierStop::WorkBudget);
        if stop.is_some() {
            horizon.interrupt(HorizonInterruption::WorkBudget);
        }
        let root_entry = PathEntry {
            parent: None,
            movement: None,
            anchor: horizon.root_anchor(),
        };
        Ok(Self {
            owner,
            registration,
            limits,
            horizon,
            paths: vec![root_entry],
            path_index: HashMap::new(),
            frames: Vec::new(),
            mate,
            native: Vec::new(),
            external_source,
            external: external::ExternalFrontierJournal::default(),
            work: FrontierNodeWork {
                root_entries: u64::from(stop.is_none()),
                native_usage_known: true,
                ..Default::default()
            },
            selection_sealed: false,
            stop,
            public_record_capacity_refused: false,
            public_review_index_capacity_refused: false,
            consumption_stop: None,
            pins,
            pin,
            checked_source,
            store_owner: stores.read_owner(),
            original_clock: clock,
            original_entry_tick: now,
            original_entry_instant,
        })
    }

    pub fn root_path(&self) -> FrontierPath {
        FrontierPath {
            owner: self.owner,
            slot: 0,
        }
    }
    pub fn root_moves(&self) -> &Arc<RootMoveDomain> {
        &self.registration.root_moves
    }
    pub fn original_deadline(&self) -> Deadline {
        self.registration.original_deadline
    }
    /// Preserve the original constructor entry tick while this owner moves
    /// through startup/Ready/Port. Validation and loading time are included.
    pub(crate) fn observed_original_tick(
        &self,
    ) -> Result<(ClockDomain, MonotonicTick), FrontierError> {
        let elapsed = Instant::now()
            .checked_duration_since(self.original_entry_instant)
            .ok_or(FrontierError::Invalid("observed monotonic clock regressed"))?;
        Ok((
            self.original_clock,
            self.original_tick_after_elapsed(
                self.original_clock,
                self.original_entry_tick,
                elapsed,
            )?,
        ))
    }
    /// The PCR manager retains one responsibility per frame/kind. This reads
    /// its existing owner handle; it creates no completion or new work.
    pub(crate) fn adaptive_obligation(
        &self,
        frame: FrameId,
        kind: FrontierFrameKind,
    ) -> Result<Option<FrontierObligation>, FrontierError> {
        if frame.authority != self.registration.authority {
            return Err(FrontierError::ForeignRoot);
        }
        let mut matches = self
            .frames
            .iter()
            .enumerate()
            .filter(|(_, held)| held.frame == frame && held.kind == kind);
        let Some((slot, held)) = matches.next() else {
            return Ok(None);
        };
        if held.pc_source || matches.next().is_some() {
            return Err(FrontierError::Invalid(
                "ambiguous adaptive responsibility owner",
            ));
        }
        Ok(Some(FrontierObligation {
            owner: self.owner,
            slot,
        }))
    }

    pub(crate) fn adaptive_obligation_source_matches(
        &mut self,
        obligation: FrontierObligation,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        stores: &PalsStores,
    ) -> Result<bool, FrontierError> {
        let source = VerifiedFrontierQuestion::from_adaptive(work, stores)?;
        self.check_verified_work(&source, stores)?;
        if obligation.owner != self.owner {
            return Err(FrontierError::ForeignHandle);
        }
        let held = self
            .frames
            .get(obligation.slot)
            .ok_or(FrontierError::ForeignHandle)?;
        Ok(!held.pc_source
            && held.frame == source.frame
            && held.question == source.question_digest
            && held.original_first == BoardMove::try_from(source.original_first_move.decode()?)?)
    }
    pub fn registration(&self) -> &FrontierRegistration {
        &self.registration
    }
    pub fn validate_attachment(
        &self,
        actual: FrontierAttachment<'_>,
        stores: &PalsStores,
    ) -> Result<(), FrontierError> {
        if self.external_source.is_some() {
            return Err(FrontierError::MissingAcceptance);
        }
        self.validate_attachment_fields(actual, stores)
    }

    pub fn validate_external_attachment(
        &self,
        actual: FrontierAttachment<'_>,
        stores: &PalsStores,
        source: &crate::external_cpu::ProfiledExternalRootSource,
    ) -> Result<(), FrontierError> {
        external::validate_external_registration(&self.registration, source, self.original_clock)?;
        if !self
            .external_source
            .as_ref()
            .is_some_and(|original| original.same_source(source))
        {
            return Err(FrontierError::ForeignRoot);
        }
        self.validate_attachment_fields(actual, stores)
    }

    fn validate_attachment_fields(
        &self,
        actual: FrontierAttachment<'_>,
        stores: &PalsStores,
    ) -> Result<(), FrontierError> {
        let registered = &self.registration;
        if actual.authority != registered.authority
            || actual.mode != registered.mode
            || actual.original_deadline != registered.original_deadline
            || actual.root_budget != registered.root_budget
            || !Arc::ptr_eq(actual.root_rules, &registered.root_rules)
            || !Arc::ptr_eq(actual.root_moves, &registered.root_moves)
            || actual.root_state != registered.root_state
            || actual.model_namespace != registered.model_namespace.as_ref()
            || actual.native_ledger != registered.native_ledger.as_ref()
            || !self.pin.validates(stores)
        {
            return Err(FrontierError::ForeignRoot);
        }
        Ok(())
    }
    pub fn path_snapshot(&self, path: FrontierPath) -> Result<&PositionSnapshot, FrontierError> {
        Ok(self.path(path)?.anchor.snapshot())
    }
    pub fn path_ply(&self, path: FrontierPath) -> Result<ChessPly, FrontierError> {
        Ok(self.path(path)?.anchor.ply())
    }
    pub fn path_prefix(&self, path: FrontierPath) -> Result<Vec<BoardMove>, FrontierError> {
        self.path(path)?;
        let mut sequence = Vec::with_capacity(usize::from(self.paths[path.slot].anchor.ply().0));
        let mut at = Some(path.slot);
        while let Some(index) = at {
            let entry = &self.paths[index];
            if let Some(movement) = entry.movement {
                sequence.push(movement);
            }
            at = entry.parent;
        }
        sequence.reverse();
        Ok(sequence)
    }

    pub fn remaining_depth(&self, path: FrontierPath) -> Result<Option<ChessPly>, FrontierError> {
        Ok(self.horizon.remaining(&self.path(path)?.anchor)?)
    }
    pub fn chunk_limit(&self, path: FrontierPath, configured: u16) -> Result<u16, FrontierError> {
        Ok(self
            .horizon
            .chunk_limit(&self.path(path)?.anchor, configured)?)
    }

    /// Actual startup Rules inspection under the original UCI controls. This
    /// checks one admitted child only; it neither scans the root nor dispatches
    /// a model/R task. The caller retains the live original Store/root owner.
    pub(crate) fn observe_root_child(
        &mut self,
        movement: BoardMove,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierPath, FrontierError> {
        let prefix = [movement];
        let child = self.observe_prefix(&prefix, clock, now)?;
        for lane in 0..2 {
            self.observe_mate_path(lane, &prefix, clock, now)?;
        }
        Ok(child)
    }

    /// Control-only gate, including the root P ranking which has no first move
    /// and therefore cannot mint a candidate question capability.
    pub(crate) fn admit_role_input(
        &mut self,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        self.check_control(clock, now)
    }

    /// Recheck a physically retired question's use of existing paid Rules
    /// work. This never admits a model call or node and never refunds credit.
    /// Exact budget fill can preserve consumption, while cancellation, source
    /// reclamation, deadline and unknown native custody always reject it.
    pub(crate) fn admit_existing_consumption(
        &mut self,
        stores: &PalsStores,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        self.check_store(stores)?;
        if clock != self.registration.original_deadline.clock {
            return Err(FrontierError::ForeignRoot);
        }
        if self
            .registration
            .original_deadline
            .accepts(clock, now)
            .is_err()
        {
            self.interrupt(FrontierStop::Deadline);
        }
        if !self.work.native_usage_known {
            return Err(FrontierError::NativeUnknown);
        }
        if let Some(stop) = self.consumption_stop {
            return Err(match stop {
                FrontierStop::NativeUnknown => FrontierError::NativeUnknown,
                FrontierStop::Deadline => FrontierError::Deadline,
                _ => FrontierError::Revoked,
            });
        }
        Ok(())
    }

    /// The actual port calls this only after observing its original hard
    /// Instant deadline. It never manufactures a later monotonic tick.
    pub(crate) fn observe_original_hard_deadline_expiry(
        &mut self,
        authority: SearchAuthority,
        original_deadline: Deadline,
    ) -> Result<(), FrontierError> {
        if authority != self.registration.authority
            || original_deadline != self.registration.original_deadline
        {
            return Err(FrontierError::ForeignRoot);
        }
        self.interrupt(FrontierStop::Deadline);
        Err(match self.stop {
            Some(FrontierStop::WorkBudget) => FrontierError::NodeBudget,
            Some(FrontierStop::NativeUnknown) => FrontierError::NativeUnknown,
            Some(FrontierStop::Deadline) => FrontierError::Deadline,
            _ => FrontierError::Revoked,
        })
    }

    /// Add only measured elapsed time to an original-domain entry tick. This
    /// is not a new clock origin, a reset budget or a claimed future deadline.
    pub(crate) fn original_tick_after_elapsed(
        &self,
        clock: ClockDomain,
        entry: MonotonicTick,
        elapsed: Duration,
    ) -> Result<MonotonicTick, FrontierError> {
        if clock != self.registration.original_deadline.clock {
            return Err(FrontierError::ForeignRoot);
        }
        let elapsed = u64::try_from(elapsed.as_nanos())
            .map_err(|_| FrontierError::Invalid("elapsed monotonic tick overflow"))?;
        entry
            .0
            .checked_add(elapsed)
            .map(MonotonicTick)
            .ok_or(FrontierError::Invalid("original monotonic tick overflow"))
    }

    /// The descriptor itself did no work. Admit and charge its exact one-edge
    /// transition before giving a semantic owner a snapshot it may adopt.
    pub(crate) fn observe_missing_child(
        &mut self,
        question: &MissingMateQuestion,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<CheckedFrontierLine, FrontierError> {
        let started = Instant::now();
        self.check_control(clock, now)?;
        self.validate_missing_question(question)?;
        let child = self.advance_path(question.anchor, question.movement, clock, now)?;
        let prefix = self.path_prefix(child)?;
        for lane in 0..2 {
            let tick = self.original_tick_after_elapsed(clock, now, started.elapsed())?;
            self.observe_mate_path(lane, &prefix, clock, tick)?;
        }
        self.seal_checked_line(child)
    }

    /// The actual private port consumer validates its role/CPU provenance
    /// separately. This boundary grants only original-root Rules inspection,
    /// charging each new checked transition before mutation. It neither
    /// attests NN acceptance nor charges a settled native CPU report again.
    pub(crate) fn observe_rules_prefix(
        &mut self,
        prefix: &[BoardMove],
        stores: &PalsStores,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<CheckedFrontierLine, FrontierError> {
        let started = Instant::now();
        self.check_control(clock, now)?;
        self.check_store(stores)?;
        let path = self.observe_prefix(prefix, clock, now)?;
        for lane in 0..2 {
            let tick = self.original_tick_after_elapsed(clock, now, started.elapsed())?;
            self.observe_mate_path(lane, prefix, clock, tick)?;
        }
        self.seal_checked_line(path)
    }

    /// Native acceptance and selection remain with the real R-free input
    /// owner. Root ranking with first=None is accepted through that original
    /// source, without inventing a first move or CoordinatorWork.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn observe_accepted_pc_child(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        movement: BoardMove,
        stores: &PalsStores,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<CheckedFrontierLine, FrontierError> {
        let started = Instant::now();
        self.check_control(clock, now)?;
        self.check_store(stores)?;
        PcPortQuestion::validate_accepted_frontier_child(input, accepted, output, movement, stores)
            .map_err(|_| FrontierError::MissingAcceptance)?;
        let question = input.question.as_ref();
        if input.view_seal.authority != self.registration.authority
            || accepted.ticket().authority != self.registration.authority
            || question.original_deadline() != self.registration.original_deadline
            || input.view_seal.root_state != self.registration.root_rules.snapshot().identity()
            || input.view.root_state() != self.registration.root_state
            || !input.view.validates(stores)
        {
            return Err(FrontierError::ForeignRoot);
        }
        self.registration
            .model_namespace
            .verify(
                accepted.ticket(),
                output,
                Some(self.registration.model_namespace.value_identity()),
            )
            .map_err(|_| FrontierError::MissingAcceptance)?;
        let tick = self.original_tick_after_elapsed(clock, now, started.elapsed())?;
        let parent = self.observe_prefix(&question.context().prefix, clock, tick)?;
        if !self
            .path_snapshot(parent)?
            .same_state(&question.position().snapshot())
        {
            return Err(FrontierError::ForeignRoot);
        }
        let tick = self.original_tick_after_elapsed(clock, now, started.elapsed())?;
        let child = self.advance_path(parent, movement, clock, tick)?;
        let prefix = self.path_prefix(child)?;
        for lane in 0..2 {
            let tick = self.original_tick_after_elapsed(clock, now, started.elapsed())?;
            self.observe_mate_path(lane, &prefix, clock, tick)?;
        }
        self.seal_checked_line(child)
    }

    /// Read an already charged exact path. No new visit, Rules replay or role
    /// submission is permitted by this snapshot factory.
    pub(crate) fn checked_line(
        &self,
        path: FrontierPath,
        stores: &PalsStores,
    ) -> Result<CheckedFrontierLine, FrontierError> {
        if !self.pin.validates(stores) || !self.store_owner.matches(stores) {
            return Err(FrontierError::StoreUnavailable);
        }
        self.seal_checked_line(path)
    }

    /// Inspect only edges this exact owner already checked and charged.
    /// Absence grants no validity and performs no Rules replay or admission.
    pub(crate) fn existing_rules_prefix(
        &self,
        prefix: &[BoardMove],
        stores: &PalsStores,
    ) -> Result<Option<CheckedFrontierLine>, FrontierError> {
        if !self.pin.validates(stores) || !self.store_owner.matches(stores) {
            return Err(FrontierError::StoreUnavailable);
        }
        self.existing_path(prefix)?
            .map(|path| self.seal_checked_line(path))
            .transpose()
    }

    fn existing_path(&self, prefix: &[BoardMove]) -> Result<Option<FrontierPath>, FrontierError> {
        if prefix.len() > usize::from(self.limits.maximum_goal_plies.get()) {
            return Err(FrontierError::DepthHorizon);
        }
        let mut path = self.root_path();
        for movement in prefix {
            let Some(slot) = self.path_index.get(&(path.slot, *movement)) else {
                return Ok(None);
            };
            path = FrontierPath {
                owner: self.owner,
                slot: *slot,
            };
        }
        Ok(Some(path))
    }

    fn seal_checked_line(&self, path: FrontierPath) -> Result<CheckedFrontierLine, FrontierError> {
        self.path(path)?;
        let prefix = self.path_prefix(path)?;
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(prefix.len())
            .map_err(|_| FrontierError::Capacity("checked line path"))?;
        let mut at = path.slot;
        while let Some(parent) = self.paths[at].parent {
            indices.push(at);
            at = parent;
        }
        indices.reverse();
        let mut steps = Vec::new();
        steps
            .try_reserve_exact(prefix.len())
            .map_err(|_| FrontierError::Capacity("checked line snapshots"))?;
        for slot in indices {
            let entry = &self.paths[slot];
            steps.push(CheckedFrontierStep {
                movement: entry.movement.ok_or(FrontierError::ForeignHandle)?,
                snapshot: entry.anchor.snapshot().clone(),
            });
        }
        Ok(CheckedFrontierLine {
            owner: self.owner,
            source: Arc::downgrade(&self.checked_source),
            store_owner: self.store_owner.clone(),
            authority: self.registration.authority,
            root_budget: self.registration.root_budget,
            root_snapshot: self.registration.root_rules.rules().snapshot().clone(),
            path,
            prefix,
            steps,
        })
    }

    /// This is actual Store evidence, never a generated string or raw value.
    /// New prefixes are Rules checked, while cached edge lookup adds no visit.
    pub fn observe_stored_line(
        &mut self,
        stores: &PalsStores,
        line: LineId,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierPath, FrontierError> {
        self.check_store(stores)?;
        self.check_control(clock, now)?;
        let start = stores.lines.get(line)?.start_state;
        if !stores
            .states
            .get(start)?
            .same_state(self.registration.root_rules.rules().snapshot())
        {
            return Err(FrontierError::ForeignRoot);
        }
        let moves = stores.lines.moves(line)?;
        self.hold_line(stores, start, line)?;
        let endpoint = self.observe_prefix(&moves, clock, now)?;
        for lane in 0..2 {
            self.observe_mate_path(lane, &moves, clock, now)?;
        }
        Ok(endpoint)
    }

    /// Port focus lines may start at a child state. Their real immutable Q
    /// still carries the full original-root prefix; use it without fabricating
    /// a new root-relative Store line, view or peer message.
    pub fn observe_local_question(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        stores: &PalsStores,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierPath, FrontierError> {
        let question = VerifiedFrontierQuestion::from_adaptive(work, stores)?;
        self.observe_verified_question(&question, stores, clock, now)
    }

    /// This source capability can only come from an actual port's typed smart
    /// admission. It is not a frame-liveness or physical completion receipt.
    pub(crate) fn observe_verified_question(
        &mut self,
        question: &VerifiedFrontierQuestion<'_>,
        stores: &PalsStores,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierPath, FrontierError> {
        self.check_control(clock, now)?;
        self.check_verified_work(question, stores)?;
        self.hold_verified_source(stores, question)?;
        let prefix = &question.context.prefix;
        let exact = self.observe_prefix(prefix, clock, now)?;
        self.check_verified_endpoint(exact, question)?;
        for lane in 0..2 {
            self.observe_mate_path(lane, prefix, clock, now)?;
        }
        Ok(exact)
    }

    /// Register only a real admitted manager frame. The immutable submitted
    /// view remains in its existing ticket owner; no new view is synthesized.
    pub fn register_frame(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        admitted: &AdmittedRoleQuestion,
        stores: &PalsStores,
        kind: FrontierFrameKind,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierObligation, FrontierError> {
        self.check_control(clock, now)?;
        self.check_peer_store(admitted, stores)?;
        self.check_work(work, admitted)?;
        let question = VerifiedFrontierQuestion::from_adaptive(work, stores)?;
        self.register_verified_frame(&question, stores, kind, clock, now)
    }

    /// A normal local continuation is not a reciprocal peer envelope. Its
    /// actual private-construction port question and selected view are checked
    /// directly, then bound to the same root-relative responsibility ledger.
    pub fn register_local_frame(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        stores: &PalsStores,
        kind: FrontierFrameKind,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierObligation, FrontierError> {
        let question = VerifiedFrontierQuestion::from_adaptive(work, stores)?;
        self.register_verified_frame(&question, stores, kind, clock, now)
    }

    /// The live Search producer selected another original local lane under the
    /// same coordinator frame. Its opaque materialized-selection source is
    /// mandatory; the old duty stays unchanged, including unmet goal progress.
    /// Ordinary registration and its strict retry checks are not widened.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn register_local_lane_successor(
        &mut self,
        previous: FrontierObligation,
        source: &super::engine::adaptive_port::AdaptiveLaneSelection,
        original: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        stores: &PalsStores,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierObligation, FrontierError> {
        self.check_control(clock, now)?;
        if !source.validates_transition(original, work) {
            return Err(FrontierError::MissingAcceptance);
        }
        let kind = source
            .obligation_kind()
            .ok_or(FrontierError::MissingAcceptance)?;
        if kind == FrontierFrameKind::Candidate {
            return Err(FrontierError::MissingAcceptance);
        }
        let question = VerifiedFrontierQuestion::from_adaptive(work, stores)?;
        let at = self.obligation_slot(previous, work.frame)?;
        let held = &self.frames[at];
        let first = BoardMove::try_from(work.original_first_move.decode()?)?;
        if held.pc_source
            || held.kind != kind
            || held.original_first != first
            || !admission_stage_matches(kind, question.stage)
        {
            return Err(FrontierError::MissingAcceptance);
        }
        if self.selection_sealed {
            return Err(FrontierError::PolicySealed);
        }
        if self.frames.len() >= self.limits.maximum_frames {
            return Err(FrontierError::Capacity("frame obligations"));
        }
        let anchor = self.observe_verified_question(&question, stores, clock, now)?;
        if self.frames.iter().any(|held| {
            held.frame == work.frame
                && held.kind == kind
                && (held.source_path == anchor || held.question == question.question_digest)
        }) {
            return Err(FrontierError::Duplicate);
        }
        self.append_checked_frame(
            work.frame,
            first,
            question.question_digest,
            anchor,
            kind,
            false,
        )
    }

    pub(crate) fn register_verified_frame(
        &mut self,
        question: &VerifiedFrontierQuestion<'_>,
        stores: &PalsStores,
        kind: FrontierFrameKind,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierObligation, FrontierError> {
        if question.pc_question.is_some() {
            return Err(FrontierError::Invalid(
                "R-free goal requires its original accepted responsibility",
            ));
        }
        if !admission_stage_matches(kind, question.stage) {
            return Err(FrontierError::Invalid(
                "responsibility differs from actual question stage",
            ));
        }
        self.observe_verified_question(question, stores, clock, now)?;
        self.register_checked_frame(
            question.frame,
            question.original_first_move,
            question.question_digest,
            &question.context.prefix,
            kind,
            false,
            clock,
            now,
        )
    }

    /// The R-free producer seals this source only after actual native role
    /// acceptance, exact legal selection and Rules-child validation. The
    /// current C/Repair/Fresh stage does not invent the original responsibility.
    pub(crate) fn register_pc_responsibility(
        &mut self,
        question: &VerifiedFrontierQuestion<'_>,
        source: &PcResponsibilitySource,
        stores: &PalsStores,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierObligation, FrontierError> {
        self.check_control(clock, now)?;
        self.check_verified_work(question, stores)?;
        let port_question = question
            .pc_question
            .ok_or(FrontierError::MissingAcceptance)?;
        if !port_question
            .frontier_responsibilities()
            .iter()
            .any(|held| std::ptr::eq(held, source))
        {
            return Err(FrontierError::MissingAcceptance);
        }
        self.observe_verified_question(question, stores, clock, now)?;
        let exact = self.observe_prefix(source.prefix(), clock, now)?;
        if !self.path_snapshot(exact)?.same_state(source.snapshot()) {
            return Err(FrontierError::Invalid(
                "responsibility differs from original checked Rules child",
            ));
        }
        let first = BoardMove::try_from(question.original_first_move.decode()?)?;
        if source.prefix().first() != Some(&first) {
            return Err(FrontierError::ForeignRoot);
        }
        for lane in 0..2 {
            self.observe_mate_path(lane, source.prefix(), clock, now)?;
        }
        self.register_checked_frame(
            question.frame,
            question.original_first_move,
            source.question_digest(),
            source.prefix(),
            source.kind(),
            true,
            clock,
            now,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn register_checked_frame(
        &mut self,
        frame: FrameId,
        first: rz_contracts::pals::Move16,
        question: Digest,
        prefix: &[BoardMove],
        kind: FrontierFrameKind,
        pc_source: bool,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierObligation, FrontierError> {
        if self.selection_sealed {
            return Err(FrontierError::PolicySealed);
        }
        if self.frames.len() >= self.limits.maximum_frames {
            return Err(FrontierError::Capacity("frame obligations"));
        }
        let original_first = BoardMove::try_from(first.decode()?)?;
        let mut anchor = self.observe_prefix(prefix, clock, now)?;
        // A root candidate's actual first move scopes its responsibility. The
        // submitted root view stays root-anchored; this is a Rules transition,
        // not an invented child tensor/view or a mandatory time-only pass.
        if self.registration.goal.depth_plies.is_some() && self.path_ply(anchor)? == ChessPly(0) {
            anchor = self.advance_path(anchor, original_first, clock, now)?;
        }
        for held in self
            .frames
            .iter()
            .filter(|held| held.frame == frame && held.kind == kind)
        {
            if held.question == question
                && held.source_path == anchor
                && held.pc_source == pc_source
            {
                return Err(FrontierError::Duplicate);
            }
            if !pc_source || !held.pc_source {
                return Err(FrontierError::Invalid(
                    "responsibility already has an original question",
                ));
            }
        }
        self.append_checked_frame(frame, original_first, question, anchor, kind, pc_source)
    }

    fn append_checked_frame(
        &mut self,
        frame: FrameId,
        original_first: BoardMove,
        question: Digest,
        anchor: FrontierPath,
        kind: FrontierFrameKind,
        pc_source: bool,
    ) -> Result<FrontierObligation, FrontierError> {
        self.frames
            .try_reserve(1)
            .map_err(|_| FrontierError::Capacity("frame obligation allocation"))?;
        let obligation = if self.registration.goal.depth_plies.is_some() {
            let exact_anchor = self.path(anchor)?.anchor.clone();
            Some(self.horizon.declare(&exact_anchor, kind.horizon_kind())?)
        } else {
            None
        };
        let handle = FrontierObligation {
            owner: self.owner,
            slot: self.frames.len(),
        };
        self.frames.push(FrameEntry {
            frame,
            kind,
            original_first,
            question,
            source_path: anchor,
            pc_source,
            horizon: obligation,
            finished: false,
            completion: None,
        });
        Ok(handle)
    }

    /// Explicit manager selection boundary. An empty queue is not this proof.
    pub fn seal_selected_policy(
        &mut self,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        self.check_control(clock, now)?;
        if self.selection_sealed {
            return Err(FrontierError::PolicySealed);
        }
        if self.registration.goal.depth_plies.is_some() {
            self.horizon.seal_policy()?;
        }
        self.selection_sealed = true;
        Ok(())
    }

    // Check the actual admitted question, Store owner, endpoint and original
    // clock separately before consuming their Rules completion evidence.
    #[allow(clippy::too_many_arguments)]
    pub fn accept_rules_endpoint(
        &mut self,
        obligation: FrontierObligation,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        admitted: &AdmittedRoleQuestion,
        stores: &PalsStores,
        endpoint: FrontierPath,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        self.check_control(clock, now)?;
        self.check_peer_store(admitted, stores)?;
        self.check_work(work, admitted)?;
        let question = VerifiedFrontierQuestion::from_adaptive(work, stores)?;
        self.accept_verified_rules_endpoint(obligation, &question, stores, endpoint, clock, now)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn accept_local_rules_endpoint(
        &mut self,
        obligation: FrontierObligation,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        stores: &PalsStores,
        endpoint: FrontierPath,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        let question = VerifiedFrontierQuestion::from_adaptive(work, stores)?;
        self.accept_verified_rules_endpoint(obligation, &question, stores, endpoint, clock, now)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn accept_local_rules_endpoint_once(
        &mut self,
        obligation: FrontierObligation,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        stores: &PalsStores,
        endpoint: FrontierPath,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierEndpointAcceptance, FrontierError> {
        let question = VerifiedFrontierQuestion::from_adaptive(work, stores)?;
        self.accept_rules_inner(obligation, &question, stores, endpoint, clock, now, true)
    }

    pub(crate) fn accept_verified_rules_endpoint(
        &mut self,
        obligation: FrontierObligation,
        question: &VerifiedFrontierQuestion<'_>,
        stores: &PalsStores,
        endpoint: FrontierPath,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        self.accept_rules_inner(obligation, question, stores, endpoint, clock, now, false)
            .map(|_| ())
    }

    #[allow(clippy::too_many_arguments)]
    fn accept_rules_inner(
        &mut self,
        obligation: FrontierObligation,
        question: &VerifiedFrontierQuestion<'_>,
        stores: &PalsStores,
        endpoint: FrontierPath,
        clock: ClockDomain,
        now: MonotonicTick,
        allow_retry: bool,
    ) -> Result<FrontierEndpointAcceptance, FrontierError> {
        let started = Instant::now();
        if clock != self.registration.original_deadline.clock {
            return Err(FrontierError::ForeignRoot);
        }
        self.check_verified_work(question, stores)?;
        let at = self.obligation_slot(obligation, question.frame)?;
        if !allow_retry || !self.frames[at].finished {
            self.check_control(clock, now)?;
        }
        let pc_source = self.check_completion_source(at, question)?;
        if !pc_source && !rules_stage_matches(self.frames[at].kind, question.stage) {
            return Err(FrontierError::MissingAcceptance);
        }
        self.check_verified_endpoint(endpoint, question)?;
        let position = Position::from_snapshot(self.path_snapshot(endpoint)?, Default::default())?;
        let boundary = HorizonBoundary::rules_terminal(&position)?;
        if self.frames[at].finished {
            if allow_retry {
                return match &self.frames[at].completion {
                    Some(FrontierCompletionSeal::Rules { scope })
                        if scope.matches(question, endpoint) =>
                    {
                        Ok(FrontierEndpointAcceptance::AlreadyAcceptedExact)
                    }
                    _ => Err(FrontierError::MissingAcceptance),
                };
            }
            return Err(FrontierError::Duplicate);
        }
        let scope = CompletionScope::new(question, endpoint);
        self.check_control(
            clock,
            self.original_tick_after_elapsed(clock, now, started.elapsed())?,
        )?;
        self.finish_obligation(at, endpoint, boundary)?;
        self.frames[at].completion = Some(FrontierCompletionSeal::Rules { scope });
        Ok(FrontierEndpointAcceptance::Accepted)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn accept_fresh_endpoint(
        &mut self,
        obligation: FrontierObligation,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        admitted: &AdmittedRoleQuestion,
        stores: &PalsStores,
        endpoint: FrontierPath,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        self.check_control(clock, now)?;
        self.check_peer_store(admitted, stores)?;
        self.check_work(work, admitted)?;
        let question = VerifiedFrontierQuestion::from_adaptive(work, stores)?;
        self.accept_verified_fresh_endpoint(
            obligation, &question, stores, endpoint, accepted, output, physical, clock, now,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn accept_local_fresh_endpoint(
        &mut self,
        obligation: FrontierObligation,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        stores: &PalsStores,
        endpoint: FrontierPath,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        let question = VerifiedFrontierQuestion::from_adaptive(work, stores)?;
        self.accept_verified_fresh_endpoint(
            obligation, &question, stores, endpoint, accepted, output, physical, clock, now,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn accept_local_fresh_endpoint_once(
        &mut self,
        obligation: FrontierObligation,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        stores: &PalsStores,
        endpoint: FrontierPath,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierEndpointAcceptance, FrontierError> {
        let question = VerifiedFrontierQuestion::from_adaptive(work, stores)?;
        self.accept_fresh_inner(
            obligation, &question, stores, endpoint, accepted, output, physical, clock, now, true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accept_verified_fresh_endpoint(
        &mut self,
        obligation: FrontierObligation,
        question: &VerifiedFrontierQuestion<'_>,
        stores: &PalsStores,
        endpoint: FrontierPath,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        self.accept_fresh_inner(
            obligation, question, stores, endpoint, accepted, output, physical, clock, now, false,
        )
        .map(|_| ())
    }

    #[allow(clippy::too_many_arguments)]
    fn accept_fresh_inner(
        &mut self,
        obligation: FrontierObligation,
        question: &VerifiedFrontierQuestion<'_>,
        stores: &PalsStores,
        endpoint: FrontierPath,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
        allow_retry: bool,
    ) -> Result<FrontierEndpointAcceptance, FrontierError> {
        let started = Instant::now();
        if clock != self.registration.original_deadline.clock {
            return Err(FrontierError::ForeignRoot);
        }
        self.check_verified_work(question, stores)?;
        let at = self.obligation_slot(obligation, question.frame)?;
        if !allow_retry || !self.frames[at].finished {
            self.check_control(clock, now)?;
        }
        let pc_source = self.check_completion_source(at, question)?;
        if !(if pc_source {
            any_fresh_stage(question.stage)
        } else {
            fresh_stage_matches(self.frames[at].kind, question.stage)
        }) {
            return Err(FrontierError::MissingAcceptance);
        }
        self.check_verified_endpoint(endpoint, question)?;
        let boundary =
            self.checked_fresh_boundary(question, endpoint, accepted, output, physical)?;
        if self.frames[at].finished {
            if allow_retry {
                return match &self.frames[at].completion {
                    Some(FrontierCompletionSeal::Fresh {
                        scope,
                        ticket,
                        output: digest,
                        physical: measured,
                    }) if scope.matches(question, endpoint)
                        && ticket == accepted.ticket()
                        && *digest == accepted.output_digest()
                        && *measured == FreshPhysicalSeal::from(physical) =>
                    {
                        Ok(FrontierEndpointAcceptance::AlreadyAcceptedExact)
                    }
                    _ => Err(FrontierError::MissingAcceptance),
                };
            }
            return Err(FrontierError::Duplicate);
        }
        let completion = FrontierCompletionSeal::Fresh {
            scope: CompletionScope::new(question, endpoint),
            ticket: *accepted.ticket(),
            output: accepted.output_digest(),
            physical: physical.into(),
        };
        self.check_control(
            clock,
            self.original_tick_after_elapsed(clock, now, started.elapsed())?,
        )?;
        self.finish_obligation(at, endpoint, boundary)?;
        self.frames[at].completion = Some(completion);
        Ok(FrontierEndpointAcceptance::Accepted)
    }

    #[allow(clippy::too_many_arguments)]
    fn checked_fresh_boundary(
        &self,
        question: &VerifiedFrontierQuestion<'_>,
        endpoint: FrontierPath,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
    ) -> Result<HorizonBoundary, FrontierError> {
        let ticket = accepted.ticket();
        if ticket.authority != question.frame.authority
            || ticket.question != question.question_digest
            || ticket.view != question.view_seal
            || ticket.original_deadline != self.registration.original_deadline
            || ticket.role != Role::Proposer
            || ticket.representation.record_revision != question.context.public_revision
            || !accepted.validates_output(output)
            || question.context.purpose != RoleQueryPurpose::ValueFresh
            || physical.physical_fresh != Some(true)
            || !physical.measurement_complete
            || physical.completed_nn_inputs.is_none_or(|count| count == 0)
            || physical
                .execution
                .is_none_or(|execution| execution.epoch != self.registration.authority.epoch)
        {
            return Err(FrontierError::MissingAcceptance);
        }
        self.registration
            .model_namespace
            .verify(
                ticket,
                output,
                Some(self.registration.model_namespace.value_identity()),
            )
            .map_err(|_| FrontierError::MissingAcceptance)?;
        let RolePayload::Proposal { wdl, .. } = output.payload else {
            return Err(FrontierError::MissingAcceptance);
        };
        let position = Position::from_snapshot(self.path_snapshot(endpoint)?, Default::default())?;
        let value = ModelValueOutput {
            identity: self.registration.model_namespace.value_identity().clone(),
            input_sha256: output.key.input.0,
            state: position.position_identity(),
            perspective: position.side_to_move(),
            wdl: wdl.probabilities(),
        };
        Ok(HorizonBoundary::fresh_value(
            &position,
            &value,
            self.registration.model_namespace.value_identity(),
        )?)
    }

    pub fn native_capacity(
        &mut self,
        requested: u64,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<u64, FrontierError> {
        self.check_control(clock, now)?;
        Ok(requested.min(self.available_nodes()?))
    }

    /// Call after the actual native ledger reserved this exact capped grant,
    /// before enqueueing its owner command. This is a second-purpose ledger of
    /// the same work, not a second charge added to native ledger spent.
    pub fn reserve_native(
        &mut self,
        job: &CpuJobEnvelope,
        grant: SliceGrant,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<NativeReservation, FrontierError> {
        self.check_control(clock, now)?;
        grant.validate(job, clock, now)?;
        if !self.registration.mode.has_native_r()
            || self.external_source.is_some()
            || job.id.authority != self.registration.authority
            || job.hard_deadline != self.registration.original_deadline
        {
            return Err(FrontierError::ForeignRoot);
        }
        if self
            .native
            .iter()
            .any(|held| held.grant.execution == grant.execution)
        {
            return Err(FrontierError::Duplicate);
        }
        if self.native.len() >= self.limits.maximum_native_receipts {
            return Err(FrontierError::Capacity("native receipt journal"));
        }
        if grant.node_credit > self.available_nodes()? {
            return Err(if self.work.reserved_native_nodes != 0 {
                FrontierError::NodesReserved
            } else {
                FrontierError::Invalid("native grant exceeds remaining original node quota")
            });
        }
        self.native
            .try_reserve(1)
            .map_err(|_| FrontierError::Capacity("native reservation allocation"))?;
        let slot = self.native.len();
        self.native.push(NativeAccount {
            grant,
            phase: NativePhase::Reserved,
        });
        self.work.reserved_native_nodes = self
            .work
            .reserved_native_nodes
            .checked_add(grant.node_credit)
            .ok_or(FrontierError::Invalid("native reservation overflow"))?;
        self.work.outstanding_native_executions += 1;
        Ok(NativeReservation {
            owner: self.owner,
            slot,
            grant,
        })
    }

    /// The real owner channel/report and this ledger-issued success receipt
    /// are both required. Late settlement still charges known work after stop;
    /// it does not reopen admission or accept an expired chess result.
    pub fn settle_native(
        &mut self,
        reservation: &NativeReservation,
        report: &CpuSliceReport,
        settled: &SettledCpuSlice,
    ) -> Result<NativeCloseReason, FrontierError> {
        let at = self.native_account(reservation)?;
        if self.native[at].phase != NativePhase::Reserved {
            return Err(FrontierError::Duplicate);
        }
        let valid = self.registration.native_ledger.as_ref() == Some(settled.ledger_identity())
            && settled.grant() == reservation.grant
            && report.grant == reservation.grant
            && report.actual_work == Some(settled.actual_work())
            && report.admission_rejected == settled.admission_rejected()
            && report.terminal == settled.terminal()
            && settled.actual_work().nodes <= reservation.grant.node_credit;
        if !valid {
            self.quarantine_native_at(at);
            return Err(FrontierError::NativeUnknown);
        }
        let actual = settled.actual_work();
        if actual.quiescence_nodes > actual.nodes {
            self.quarantine_native_at(at);
            return Err(FrontierError::NativeUnknown);
        }
        if settled.admission_rejected() && (actual != Default::default() || report.result.is_ok()) {
            self.quarantine_native_at(at);
            return Err(FrontierError::NativeUnknown);
        }
        let native_nodes = self
            .work
            .native_nodes
            .checked_add(actual.nodes)
            .ok_or(FrontierError::Invalid("native spent overflow"))?;
        let qnodes = self
            .work
            .native_qsearch_nodes
            .checked_add(actual.quiescence_nodes)
            .ok_or(FrontierError::Invalid("native qsearch overflow"))?;
        let total = self
            .work
            .root_entries
            .checked_add(self.work.checked_transitions)
            .and_then(|spent| spent.checked_add(native_nodes))
            .ok_or(FrontierError::Invalid("common spent overflow"))?;
        if self.registration.nodes.is_some_and(|limit| total > limit) {
            self.quarantine_native_at(at);
            return Err(FrontierError::NativeUnknown);
        }
        let reason = if settled.admission_rejected() {
            NativeCloseReason::OwnerRejectedBeforeWork
        } else {
            NativeCloseReason::Executed
        };
        self.work.native_nodes = native_nodes;
        self.work.native_qsearch_nodes = qnodes;
        self.close_native_at(at, reason)?;
        Ok(reason)
    }

    /// A definite pre-enqueue rejection remains distinguishable from an
    /// attested worker response whose actual delta happened to be zero.
    pub fn release_unqueued(
        &mut self,
        reservation: &NativeReservation,
        released: &ReleasedCpuReservation,
    ) -> Result<NativeCloseReason, FrontierError> {
        let at = self.native_account(reservation)?;
        if self.native[at].phase != NativePhase::Reserved {
            return Err(FrontierError::Duplicate);
        }
        if self.registration.native_ledger.as_ref() != Some(released.ledger_identity())
            || released.grant() != reservation.grant
        {
            self.quarantine_native_at(at);
            return Err(FrontierError::NativeUnknown);
        }
        self.close_native_at(at, NativeCloseReason::RejectedBeforeEnqueue)?;
        Ok(NativeCloseReason::RejectedBeforeEnqueue)
    }

    /// Main-ledger rejection, disconnected owner or unknown actual usage does
    /// not mint a zero-work receipt and never releases this reservation.
    pub fn native_usage_unknown(
        &mut self,
        reservation: &NativeReservation,
    ) -> Result<(), FrontierError> {
        let at = self.native_account(reservation)?;
        if self.native[at].phase != NativePhase::Reserved {
            return Err(FrontierError::Duplicate);
        }
        self.quarantine_native_at(at);
        Ok(())
    }

    pub fn missing_mate_questions(
        &mut self,
        maximum: usize,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Vec<MissingMateQuestion>, FrontierError> {
        self.check_control(clock, now)?;
        if maximum > self.limits.maximum_frames {
            return Err(FrontierError::Capacity("missing-question result bound"));
        }
        let mut questions = Vec::with_capacity(maximum);
        if maximum == 0 || self.registration.goal.mate_moves.is_none() {
            return Ok(questions);
        }
        for (lane_index, lane) in self.mate.iter().enumerate() {
            let Some(lane) = lane else {
                continue;
            };
            // The owner maintains admission order incrementally; no complete
            // per-tick clone, sort or quadratic path matching is required.
            for &(node, path) in &lane.ordered {
                if lane.proof.conclusion(node)? != MateConclusion::Unknown
                    || lane.proof.remaining(node)?.0 == 0
                    || self
                        .registration
                        .goal
                        .depth_plies
                        .is_some_and(|depth| self.paths[path.slot].anchor.ply().0 >= depth.get())
                {
                    continue;
                }
                for movement in lane.proof.missing_moves(node)? {
                    questions.push(MissingMateQuestion {
                        owner: self.owner,
                        lane: lane_index,
                        node,
                        anchor: path,
                        movement,
                        attacker: lane.proof.root_conclusion().attacker,
                        remaining: lane.proof.remaining(node)?,
                        original_deadline: self.registration.original_deadline,
                        kind: if self.paths[path.slot].anchor.snapshot().side_to_move()
                            == lane.proof.root_conclusion().attacker
                        {
                            MateQuestionKind::AttackCandidate
                        } else {
                            MateQuestionKind::RequiredDefense
                        },
                    });
                    if questions.len() == maximum {
                        return Ok(questions);
                    }
                }
            }
        }
        Ok(questions)
    }

    pub fn validate_missing_question(
        &self,
        question: &MissingMateQuestion,
    ) -> Result<(), FrontierError> {
        if question.owner != self.owner
            || question.original_deadline != self.registration.original_deadline
        {
            return Err(FrontierError::ForeignHandle);
        }
        let lane = self
            .mate
            .get(question.lane)
            .and_then(Option::as_ref)
            .ok_or(FrontierError::ForeignHandle)?;
        if lane.paths.get(&question.node) != Some(&question.anchor)
            || !lane
                .proof
                .missing_moves(question.node)?
                .contains(&question.movement)
            || lane.proof.conclusion(question.node)? != MateConclusion::Unknown
        {
            return Err(FrontierError::ForeignHandle);
        }
        Ok(())
    }

    /// Build only the one requested branch descriptor. Its final transition
    /// must still be checked by the actual port; this function executes none.
    pub fn missing_question_prefix(
        &self,
        question: &MissingMateQuestion,
    ) -> Result<Vec<BoardMove>, FrontierError> {
        self.validate_missing_question(question)?;
        let mut prefix = self.path_prefix(question.anchor)?;
        prefix
            .try_reserve(1)
            .map_err(|_| FrontierError::Capacity("missing question prefix"))?;
        prefix.push(question.movement);
        Ok(prefix)
    }

    pub fn missing_question_first_move(
        &self,
        question: &MissingMateQuestion,
    ) -> Result<BoardMove, FrontierError> {
        self.validate_missing_question(question)?;
        Ok(self
            .path_prefix(question.anchor)?
            .first()
            .copied()
            .unwrap_or(question.movement))
    }

    /// Remaining generation chunk after the imposed missing move. Zero
    /// requests only a boundary/Rules observation, never an invented extra ply.
    pub fn missing_question_chunk_limit(
        &self,
        question: &MissingMateQuestion,
        configured: u16,
    ) -> Result<u16, FrontierError> {
        self.validate_missing_question(question)?;
        let proof = question.remaining.0.saturating_sub(1);
        let depth = self
            .remaining_depth(question.anchor)?
            .map(|plies| plies.0.saturating_sub(1));
        Ok(configured.min(proof).min(depth.unwrap_or(u16::MAX)))
    }

    pub fn revoke_obligation(
        &mut self,
        obligation: FrontierObligation,
    ) -> Result<(), FrontierError> {
        if obligation.owner != self.owner {
            return Err(FrontierError::ForeignHandle);
        }
        let held = self
            .frames
            .get_mut(obligation.slot)
            .ok_or(FrontierError::ForeignHandle)?;
        if let Some(horizon) = held.horizon {
            self.horizon.revoke(horizon)?;
        }
        held.finished = false;
        held.completion = None;
        self.interrupt(FrontierStop::EvidenceReclaimed);
        Ok(())
    }

    pub fn cancel(&mut self) {
        self.interrupt(FrontierStop::Canceled);
    }

    /// The same semantic owner calls this only after its role-call admission
    /// limit is exhausted and all admitted role heads have settled. It does
    /// not attest a worker fence or turn a satisfied/failed goal into a stop.
    pub(crate) fn close_role_call_budget(&mut self) {
        if self.work.native_usage_known
            && matches!(self.snapshot().goal, FrontierGoalStatus::Pending)
        {
            self.interrupt(FrontierStop::RoleBudget);
        }
    }

    /// Read-only cause evidence from this same controller. A generic capacity
    /// stop, copied snapshot or error value cannot set the private marker.
    pub fn public_record_capacity_refused(&self) -> bool {
        self.public_record_capacity_refused && self.stop == Some(FrontierStop::Capacity)
    }

    /// Read-only cause evidence from this same controller's original bundle cap.
    /// A generic IndexCapacity, copied snapshot or error value cannot set it.
    pub fn public_review_index_capacity_refused(&self) -> bool {
        self.public_review_index_capacity_refused && self.stop == Some(FrontierStop::Capacity)
    }

    /// Called only by the original public-record producer at its registered
    /// bound. Closing admission neither reclaims evidence nor completes depth.
    pub(super) fn refuse_public_record_capacity(&mut self) -> FrontierError {
        self.public_record_capacity_refused = true;
        self.interrupt(FrontierStop::Capacity);
        FrontierError::PublicRecordCapacity
    }

    /// Called only after the original selected-review publication returned
    /// BundleRecordCapacity. Other review/index failures cannot call this path.
    /// Closing admission neither removes evidence nor completes a goal/depth.
    pub(super) fn refuse_public_review_index_capacity(&mut self) -> FrontierError {
        self.public_review_index_capacity_refused = true;
        self.interrupt(FrontierStop::Capacity);
        FrontierError::PublicReviewIndexCapacity
    }

    /// A typed observation, not an output/fence authority. The real consumer
    /// must still join its original root/domain and physical result custody.
    /// A proved own win is preferred; otherwise the opponent lane supplies
    /// its proved longest-delay defense within the original root domain.
    pub fn mate_strategy(&self) -> Option<FrontierMateStrategy> {
        self.mate
            .iter()
            .flatten()
            .find_map(|lane| lane.proof.root_strategy())
    }

    pub fn snapshot(&self) -> FrontierSnapshot {
        let horizon = self.horizon.snapshot();
        let mate = std::array::from_fn(|lane| {
            self.mate[lane]
                .as_ref()
                .map(|lane| lane.proof.root_conclusion())
        });
        let completed_frames = self.frames.iter().filter(|frame| frame.finished).count();
        let exhausted = self.registration.nodes.is_some_and(|limit| {
            self.work
                .total_node_visits()
                .is_none_or(|spent| spent >= limit)
        });
        let goal = if let Some(stop) = self.stop {
            FrontierGoalStatus::Stopped(stop)
        } else if exhausted {
            FrontierGoalStatus::Stopped(FrontierStop::WorkBudget)
        } else if self.work.outstanding_native_executions != 0
            || self.external.snapshot().outstanding_questions != 0
        {
            FrontierGoalStatus::Pending
        } else if mate.iter().flatten().any(|proof| proof.goal_satisfied()) {
            FrontierGoalStatus::MateSatisfied
        } else if self.registration.goal.mate_moves.is_some()
            && self.selection_sealed
            && completed_frames == self.frames.len()
        {
            FrontierGoalStatus::GoalUnresolved
        } else if horizon.target_satisfied {
            FrontierGoalStatus::DepthSatisfied
        } else if self.registration.goal.depth_plies.is_none()
            && self.registration.goal.mate_moves.is_none()
            && self.selection_sealed
            && !self.frames.is_empty()
            && completed_frames == self.frames.len()
            && self.work.native_usage_known
            && self.work.reserved_native_nodes == 0
        {
            FrontierGoalStatus::SelectiveFrontier
        } else {
            FrontierGoalStatus::Pending
        };
        FrontierSnapshot {
            authority: self.registration.authority,
            original_deadline: self.registration.original_deadline,
            node_unit: ROVENT_NODE_UNIT,
            requested_nodes: self.registration.nodes,
            work: self.work,
            external_work: self
                .external_source
                .as_ref()
                .map(|_| self.external.snapshot()),
            horizon,
            mate,
            mate_strategy: self.mate_strategy(),
            goal,
            registered_frames: self.frames.len(),
            completed_frames,
            selection_sealed: self.selection_sealed,
        }
    }

    fn path(&self, path: FrontierPath) -> Result<&PathEntry, FrontierError> {
        if path.owner != self.owner {
            return Err(FrontierError::ForeignHandle);
        }
        self.paths
            .get(path.slot)
            .ok_or(FrontierError::ForeignHandle)
    }
    fn check_store(&mut self, stores: &PalsStores) -> Result<(), FrontierError> {
        if !self.pin.validates(stores) {
            self.interrupt(FrontierStop::EvidenceReclaimed);
            return Err(FrontierError::StoreUnavailable);
        }
        Ok(())
    }
    fn check_control(
        &mut self,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        if let Some(stop) = self.stop {
            return Err(match stop {
                FrontierStop::WorkBudget => FrontierError::NodeBudget,
                FrontierStop::NativeUnknown => FrontierError::NativeUnknown,
                FrontierStop::Deadline => FrontierError::Deadline,
                _ => FrontierError::Revoked,
            });
        }
        if clock != self.registration.original_deadline.clock {
            return Err(FrontierError::ForeignRoot);
        }
        if self
            .registration
            .original_deadline
            .accepts(clock, now)
            .is_err()
        {
            self.interrupt(FrontierStop::Deadline);
            return Err(FrontierError::Deadline);
        }
        if self.registration.nodes.is_some_and(|limit| {
            self.work
                .total_node_visits()
                .is_none_or(|spent| spent >= limit)
        }) {
            self.interrupt(FrontierStop::WorkBudget);
            return Err(FrontierError::NodeBudget);
        }
        Ok(())
    }
    fn available_nodes(&self) -> Result<u64, FrontierError> {
        let spent = self
            .work
            .total_node_visits()
            .and_then(|spent| spent.checked_add(self.work.reserved_native_nodes))
            .ok_or(FrontierError::Invalid(
                "common reserved accounting overflow",
            ))?;
        Ok(self
            .registration
            .nodes
            .unwrap_or(u64::MAX)
            .saturating_sub(spent))
    }
    fn spend_transition(&mut self) -> Result<(), FrontierError> {
        if self.available_nodes()? == 0 {
            return Err(if self.work.reserved_native_nodes != 0 {
                FrontierError::NodesReserved
            } else {
                FrontierError::NodeBudget
            });
        }
        self.work.checked_transitions = self
            .work
            .checked_transitions
            .checked_add(1)
            .ok_or(FrontierError::Invalid("checked transition overflow"))?;
        Ok(())
    }
    fn interrupt(&mut self, stop: FrontierStop) {
        if stop != FrontierStop::WorkBudget {
            self.checked_source.closed.store(true, Ordering::Release);
        }
        if stop != FrontierStop::WorkBudget && self.consumption_stop.is_none() {
            self.consumption_stop = Some(stop);
        }
        if self.stop.is_some() {
            return;
        }
        self.stop = Some(stop);
        self.horizon.interrupt(match stop {
            FrontierStop::Canceled => HorizonInterruption::Stop,
            FrontierStop::Deadline => HorizonInterruption::Deadline,
            FrontierStop::WorkBudget => HorizonInterruption::WorkBudget,
            FrontierStop::RoleBudget => HorizonInterruption::RoleBudget,
            FrontierStop::Capacity => HorizonInterruption::Capacity,
            FrontierStop::EvidenceReclaimed => HorizonInterruption::EvidenceReclaimed,
            _ => HorizonInterruption::Failed,
        });
    }
    fn hold_line(
        &mut self,
        stores: &PalsStores,
        state: StateId,
        line: LineId,
    ) -> Result<(), FrontierError> {
        if self.pins.lines.contains(&line) {
            return Ok(());
        }
        if self.pins.lines.len() >= self.limits.maximum_paths {
            return Err(FrontierError::Capacity("retained line pins"));
        }
        let mut pins = self.pins.clone();
        pins.states.insert(state);
        pins.lines.insert(line);
        let pin = Arc::new(stores.pin_review(pins.clone())?);
        self.replace_checked_source_pin(pin)?;
        self.pins = pins;
        Ok(())
    }
    fn hold_verified_source(
        &mut self,
        stores: &PalsStores,
        question: &VerifiedFrontierQuestion<'_>,
    ) -> Result<(), FrontierError> {
        let source = question.context;
        if self.pins.states.contains(&source.state)
            && self.pins.lines.contains(&source.focus)
            && self.pins.situations.contains(&source.situation)
        {
            return Ok(());
        }
        if self.pins.lines.len() >= self.limits.maximum_paths
            || self.pins.states.len() > self.limits.maximum_paths
            || self.pins.situations.len() >= self.limits.maximum_paths
        {
            return Err(FrontierError::Capacity("retained local source pins"));
        }
        let mut pins = self.pins.clone();
        pins.states.insert(source.state);
        pins.lines.insert(source.focus);
        pins.situations.insert(source.situation);
        let pin = Arc::new(stores.pin_review(pins.clone())?);
        self.replace_checked_source_pin(pin)?;
        self.pins = pins;
        Ok(())
    }
    fn replace_checked_source_pin(&mut self, pin: Arc<StoreReadPin>) -> Result<(), FrontierError> {
        let mut current = self
            .checked_source
            .pin
            .lock()
            .map_err(|_| FrontierError::StoreUnavailable)?;
        *current = Arc::downgrade(&pin);
        self.pin = pin;
        Ok(())
    }
    fn observe_prefix(
        &mut self,
        prefix: &[BoardMove],
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierPath, FrontierError> {
        let started = Instant::now();
        if prefix.len() > usize::from(self.limits.maximum_goal_plies.get()) {
            return Err(FrontierError::DepthHorizon);
        }
        if self
            .registration
            .goal
            .depth_plies
            .is_some_and(|depth| prefix.len() > usize::from(depth.get()))
        {
            return Err(FrontierError::DepthHorizon);
        }
        let mut path = self.root_path();
        for movement in prefix {
            let tick = self.original_tick_after_elapsed(clock, now, started.elapsed())?;
            path = self.advance_path(path, *movement, clock, tick)?;
        }
        Ok(path)
    }
    fn advance_path(
        &mut self,
        parent: FrontierPath,
        movement: BoardMove,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierPath, FrontierError> {
        self.check_control(clock, now)?;
        self.path(parent)?;
        if let Some(slot) = self.path_index.get(&(parent.slot, movement)) {
            return Ok(FrontierPath {
                owner: self.owner,
                slot: *slot,
            });
        }
        if parent.slot == 0 && !self.registration.root_moves.allowed().contains(&movement) {
            return Err(FrontierError::Invalid(
                "first move is outside original searchmoves",
            ));
        }
        let old_anchor = self.path(parent)?.anchor.clone();
        if old_anchor.ply().0 >= self.limits.maximum_goal_plies.get()
            || self
                .registration
                .goal
                .depth_plies
                .is_some_and(|depth| old_anchor.ply().0 >= depth.get())
        {
            return Err(FrontierError::DepthHorizon);
        }
        if self.paths.len() >= self.limits.maximum_paths {
            return Err(FrontierError::Capacity("exact frontier paths"));
        }
        let position = Position::from_snapshot(old_anchor.snapshot(), Default::default())?;
        let legal = position.ordered_legal_moves();
        if position.play_status_from_view(&legal)? != PlayStatus::Ongoing
            || !legal.moves().contains(&movement)
        {
            return Err(FrontierError::Invalid(
                "non-legal or post-terminal frontier transition",
            ));
        }
        self.paths
            .try_reserve(1)
            .map_err(|_| FrontierError::Capacity("frontier path allocation"))?;
        self.path_index
            .try_reserve(1)
            .map_err(|_| FrontierError::Capacity("frontier edge index"))?;
        // Exclusive single-writer custody reserves this credit before Rules
        // mutation; no worker submission/await may intervene before its charge.
        self.spend_transition()?;
        let anchor = self.horizon.advance(&old_anchor, &position, movement)?;
        let slot = self.paths.len();
        self.paths.push(PathEntry {
            parent: Some(parent.slot),
            movement: Some(movement),
            anchor,
        });
        self.path_index.insert((parent.slot, movement), slot);
        Ok(FrontierPath {
            owner: self.owner,
            slot,
        })
    }
    fn observe_mate_path(
        &mut self,
        lane_index: usize,
        prefix: &[BoardMove],
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        let started = Instant::now();
        let Some(lane) = self.mate[lane_index].as_ref() else {
            return Ok(());
        };
        let passive = self.registration.goal.mate_moves.is_none();
        let checked = if passive {
            if clock != self.registration.original_deadline.clock {
                return Err(FrontierError::ForeignRoot);
            }
            let endpoint = self.existing_path(prefix)?.ok_or(FrontierError::Invalid(
                "passive mate coverage lacks prior checked Rules path",
            ))?;
            Some(self.seal_checked_line(endpoint)?)
        } else {
            None
        };
        let mut node = lane.proof.root();
        let mut path = self.root_path();
        for (step_index, movement) in prefix.iter().enumerate() {
            let tick = self.original_tick_after_elapsed(clock, now, started.elapsed())?;
            // Passive reduction consumes immutable evidence already admitted
            // under this root. It creates no new work/admission and may finish
            // projecting the paid final edge when that edge filled the quota.
            if !passive {
                self.check_control(clock, tick)?;
            }
            let lane = self.mate[lane_index].as_ref().expect("registered lane");
            if lane.proof.remaining(node)?.0 == 0
                || lane.proof.conclusion(node)? != MateConclusion::Unknown
            {
                break;
            }
            let existing = lane.proof.child(node, *movement)?;
            let slot =
                *self
                    .path_index
                    .get(&(path.slot, *movement))
                    .ok_or(FrontierError::Invalid(
                        "mate edge lacks prior exact Rules path",
                    ))?;
            path = FrontierPath {
                owner: self.owner,
                slot,
            };
            node = match existing {
                Some(child) => child,
                None if passive => self.mate[lane_index]
                    .as_mut()
                    .expect("registered lane")
                    .proof
                    .observe_checked_child(
                        node,
                        checked.as_ref().expect("passive checked source"),
                        step_index,
                    )?,
                None => {
                    self.spend_transition()?;
                    self.mate[lane_index]
                        .as_mut()
                        .expect("registered lane")
                        .proof
                        .open_child(node, *movement)?
                }
            };
            let lane = self.mate[lane_index].as_mut().expect("registered lane");
            if !lane.paths.contains_key(&node) {
                lane.paths
                    .try_reserve(1)
                    .map_err(|_| FrontierError::Capacity("mate path index"))?;
                lane.ordered
                    .try_reserve(1)
                    .map_err(|_| FrontierError::Capacity("mate path order"))?;
                lane.paths.insert(node, path);
                lane.ordered.push((node, path));
            }
        }
        Ok(())
    }
    fn check_work(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        admitted: &AdmittedRoleQuestion,
    ) -> Result<(), FrontierError> {
        let question = admitted.envelope();
        if work.frame.authority != self.registration.authority
            || work.frame.sequence == 0
            || work.original_deadline != self.registration.original_deadline
            || question.id.authority != self.registration.authority
            || question.root_budget != self.registration.root_budget
            || question.original_deadline != self.registration.original_deadline
            || question.original_first_move != Some(work.original_first_move)
            || question.question_digest != work.input.question_digest
            || question.view != work.input.view_seal
            || !Arc::ptr_eq(&work.input.view, admitted.view())
            || &work.input.question.context != admitted.context()
            || !work
                .input
                .question
                .position
                .snapshot()
                .same_state(admitted.rules().rules().snapshot())
            || !Arc::ptr_eq(admitted.root_moves(), &self.registration.root_moves)
            || admitted.root_rules().snapshot().identity()
                != self.registration.root_rules.snapshot().identity()
            || work.input.public_context.0 != work.input.view.record_scope_digest()
        {
            return Err(FrontierError::ForeignRoot);
        }
        Ok(())
    }

    fn check_verified_work(
        &mut self,
        question: &VerifiedFrontierQuestion<'_>,
        stores: &PalsStores,
    ) -> Result<(), FrontierError> {
        self.check_store(stores)?;
        let first = BoardMove::try_from(question.original_first_move.decode()?)?;
        let view = question.view;
        if question.frame.authority != self.registration.authority
            || question.frame.sequence == 0
            || question.original_deadline != self.registration.original_deadline
            || question.view_seal.authority != self.registration.authority
            || question.view_seal.root_state != self.registration.root_rules.snapshot().identity()
            || question.public_context.0 != view.record_scope_digest()
            || view.root_state() != self.registration.root_state
            || view.generation() != self.registration.authority.root.0
            || !view.validates(stores)
            || !self.registration.root_moves.allowed().contains(&first)
            || question
                .context
                .prefix
                .first()
                .is_some_and(|movement| *movement != first)
        {
            return Err(FrontierError::ForeignRoot);
        }
        Ok(())
    }

    fn check_peer_store(
        &mut self,
        admitted: &AdmittedRoleQuestion,
        stores: &PalsStores,
    ) -> Result<(), FrontierError> {
        self.check_store(stores)?;
        if !admitted.view().validates(stores) {
            return Err(FrontierError::StoreUnavailable);
        }
        Ok(())
    }
    fn obligation_slot(
        &self,
        obligation: FrontierObligation,
        frame: FrameId,
    ) -> Result<usize, FrontierError> {
        if obligation.owner != self.owner {
            return Err(FrontierError::ForeignHandle);
        }
        let held = self
            .frames
            .get(obligation.slot)
            .ok_or(FrontierError::ForeignHandle)?;
        if held.frame != frame {
            return Err(FrontierError::ForeignHandle);
        }
        Ok(obligation.slot)
    }
    fn check_completion_source(
        &self,
        at: usize,
        question: &VerifiedFrontierQuestion<'_>,
    ) -> Result<bool, FrontierError> {
        let held = &self.frames[at];
        if !held.pc_source {
            return if question.pc_question.is_none() {
                Ok(false)
            } else {
                Err(FrontierError::MissingAcceptance)
            };
        }
        let current = question
            .pc_question
            .ok_or(FrontierError::MissingAcceptance)?;
        let prefix = self.path_prefix(held.source_path)?;
        let snapshot = self.path_snapshot(held.source_path)?;
        if !current.frontier_responsibilities().iter().any(|source| {
            source.kind() == held.kind
                && source.question_digest() == held.question
                && source.prefix() == prefix.as_slice()
                && source.snapshot().same_state(snapshot)
        }) {
            return Err(FrontierError::MissingAcceptance);
        }
        Ok(true)
    }
    fn check_verified_endpoint(
        &self,
        endpoint: FrontierPath,
        question: &VerifiedFrontierQuestion<'_>,
    ) -> Result<(), FrontierError> {
        if !self
            .path_snapshot(endpoint)?
            .same_state(&question.position.snapshot())
            || self.path_prefix(endpoint)? != question.context.prefix
        {
            return Err(FrontierError::Invalid(
                "local endpoint differs from exact root replay",
            ));
        }
        Ok(())
    }
    fn finish_obligation(
        &mut self,
        at: usize,
        endpoint: FrontierPath,
        boundary: HorizonBoundary,
    ) -> Result<(), FrontierError> {
        let first = self.path_prefix(endpoint)?.first().copied();
        if first != Some(self.frames[at].original_first) {
            return Err(FrontierError::Invalid(
                "endpoint escaped original first move",
            ));
        }
        if let Some(obligation) = self.frames[at].horizon {
            let anchor = self.path(endpoint)?.anchor.clone();
            self.horizon.finish(obligation, &anchor, &boundary)?;
        }
        self.frames[at].finished = true;
        Ok(())
    }
    fn native_account(&self, token: &NativeReservation) -> Result<usize, FrontierError> {
        if token.owner != self.owner
            || self
                .native
                .get(token.slot)
                .is_none_or(|held| held.grant != token.grant)
        {
            return Err(FrontierError::ForeignHandle);
        }
        Ok(token.slot)
    }
    fn quarantine_native_at(&mut self, at: usize) {
        self.native[at].phase = NativePhase::Quarantined;
        self.work.native_usage_known = false;
        self.interrupt(FrontierStop::NativeUnknown);
    }
    fn close_native_at(
        &mut self,
        at: usize,
        reason: NativeCloseReason,
    ) -> Result<(), FrontierError> {
        self.work.reserved_native_nodes = self
            .work
            .reserved_native_nodes
            .checked_sub(self.native[at].grant.node_credit)
            .ok_or(FrontierError::Invalid("native reservation underflow"))?;
        self.work.outstanding_native_executions = self
            .work
            .outstanding_native_executions
            .checked_sub(1)
            .ok_or(FrontierError::Invalid("native execution underflow"))?;
        self.native[at].phase = NativePhase::Settled(reason);
        Ok(())
    }
}

fn fresh_stage_matches(kind: FrontierFrameKind, stage: RefinementStage) -> bool {
    match kind {
        FrontierFrameKind::Candidate => stage == RefinementStage::ProposalValueFresh,
        FrontierFrameKind::Counter => matches!(
            stage,
            RefinementStage::CounterValueFresh | RefinementStage::FreshCounterEndpoint
        ),
        FrontierFrameKind::Repair => matches!(
            stage,
            RefinementStage::RepairValueFresh
                | RefinementStage::FreshRepairedEndpoint
                | RefinementStage::RecheckRepairedValueFresh
        ),
    }
}

fn any_fresh_stage(stage: RefinementStage) -> bool {
    [
        FrontierFrameKind::Candidate,
        FrontierFrameKind::Counter,
        FrontierFrameKind::Repair,
    ]
    .into_iter()
    .any(|kind| fresh_stage_matches(kind, stage))
}

fn admission_stage_matches(kind: FrontierFrameKind, stage: RefinementStage) -> bool {
    rules_stage_matches(kind, stage)
        || matches!(
            (kind, stage),
            (FrontierFrameKind::Counter, RefinementStage::CpuDiscovery)
        )
}

fn rules_stage_matches(kind: FrontierFrameKind, stage: RefinementStage) -> bool {
    fresh_stage_matches(kind, stage)
        || match kind {
            FrontierFrameKind::Candidate => matches!(
                stage,
                RefinementStage::ProposalContinuation | RefinementStage::CpuValidation
            ),
            FrontierFrameKind::Counter => matches!(
                stage,
                RefinementStage::CriticReply
                    | RefinementStage::CounterContinuation
                    | RefinementStage::CounterValidation
                    | RefinementStage::CriticRecheck
                    | RefinementStage::RecheckCounterValidation
            ),
            FrontierFrameKind::Repair => matches!(
                stage,
                RefinementStage::Repair | RefinementStage::RepairValidation
            ),
        }
}

#[cfg(test)]
#[path = "rovent_frontier/tests.rs"]
mod tests;
