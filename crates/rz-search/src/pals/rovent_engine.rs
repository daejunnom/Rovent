//! Explicit P/C and P/C/R synchronous compatibility owner.
//!
//! This consumes the existing `RoleModel` and Rules interfaces. It is not the
//! asynchronous coordinator and does not fabricate selected views for native
//! providers. P/C constructors have no CPU searcher/checker argument or owner.
//! A caller runs this compatibility owner on its search worker, never on the
//! coordinator's event loop. The original provider retains physical drain and
//! quarantine responsibility under the existing RoleModel contract.

use super::engine::{
    RoleAcceptance, RoleError, RoleEvaluation, RoleLogicalContext, RoleModel, RoleQuery,
    RoleQueryPurpose, RoleSearchClosure,
};
use super::review_context::ReviewView;
use super::rovent_goals::{
    HorizonBoundary, HorizonError, HorizonInterruption, HorizonObligationKind, HorizonSnapshot,
    SelectiveHorizon,
};
use super::rovent_solved::{
    MateConclusion, MateNodeId, MateProofError, MateProofStore, RootMateConclusion,
};
use super::store::{PalsStores, StoreError, StoreLimits};
use super::value::{ModelValueIdentity, ModelValueOutput};
use crate::cpu::{CpuLimits, CpuSearcher};
use crate::cpu_checker::{
    CheckerError, CheckerIdentity, CheckerReport, CheckerShutdown, CpuChecker, OwnedCpuChecker,
};
use rz_contracts::rovent::{ChessPly, ExecutionMode, PcCompute, QuestionEnvelope, SearchGoal};
use rz_contracts::{ClockDomain, Digest, MonotonicTick, StateIdentity};
use rz_position::contracts::ContractState;
use rz_position::{
    BoardMove, Color, PlayStatus, Position, PositionError, PositionSnapshot, TerminalReason,
};
use sha2::{Digest as _, Sha256};
use std::num::NonZeroU16;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

pub const ROVENT_SYNC_SEARCH_VERSION: &str = "rovent-selective-role-frontier/0.1";
/// Root admission + actual checked P/C/mate transitions + actual native R
/// nodes. Model calls, NN inputs and PUCT simulations are different units.
pub const ROVENT_NODE_UNIT: &str = "rovent-position-visits/1";
pub const ROVENT_PREFIX_DIGEST_DOMAIN: &str = "rovent-rules-prefix/1";
pub const ROVENT_LINE_DIGEST_DOMAIN: &str = "rovent-rules-complete-line/1";

/// A loaded owner supplies the actual startup device registration. This
/// declaration is checked against mode/model identity; it is not GPU evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PcRegistration {
    pub compute: PcCompute,
    pub model_identity: String,
}

#[derive(Clone, Debug)]
pub struct RoventConfig {
    pub mode: ExecutionMode,
    pub maximum_goal_plies: NonZeroU16,
    pub candidate_width: usize,
    pub normal_line_plies: NonZeroU16,
    pub maximum_positions: usize,
    pub maximum_role_calls: u64,
    pub r_depth: NonZeroU16,
    pub r_nodes_per_call: u64,
    pub selective_policy: Digest,
}

impl RoventConfig {
    fn validate(&self) -> Result<(), RoventError> {
        if self.maximum_goal_plies.get() > 256
            || self.candidate_width == 0
            || self.candidate_width > 16
            || self.normal_line_plies > self.maximum_goal_plies
            || self.maximum_positions == 0
            || self.maximum_positions > 65_536
            || self.maximum_role_calls == 0
            || self.maximum_role_calls > 1_000_000
            || self.r_nodes_per_call == 0
        {
            return Err(RoventError::Invalid("invalid finite Rovent configuration"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RoventLimits {
    pub goal: SearchGoal,
    pub deadline: Instant,
    pub maximum_role_calls: u64,
    pub maximum_native_nodes: u64,
    /// Original `go nodes`, independent of provider and native owner ceilings.
    pub maximum_node_visits: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RoventWork {
    pub root_entries: u64,
    pub role_calls: u64,
    pub accepted_policy: u64,
    pub accepted_fresh_values: u64,
    pub checked_transitions: u64,
    pub native_r_calls: u64,
    pub native_pvs_entries: Option<u64>,
    pub native_nodes: u64,
    pub native_qsearch_nodes: u64,
    /// PC has no R owner, hence observed zero. PCR does not infer internal SEE
    /// work from a checker report that does not expose that counter.
    pub native_see_nodes: Option<u64>,
    pub external_teacher_calls: u64,
}

impl RoventWork {
    /// An overflow is unavailable accounting, never a fabricated zero count.
    pub fn total_node_visits(&self) -> Option<u64> {
        self.root_entries
            .checked_add(self.checked_transitions)?
            .checked_add(self.native_nodes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventCompletion {
    RulesTerminal,
    SelectiveFrontier,
    DepthGoal,
    MateGoal,
    GoalUnresolved,
    Canceled,
    Deadline,
    WorkBudget,
    Capacity,
}

#[derive(Clone, Debug)]
pub struct RoventCandidate {
    pub movement: BoardMove,
    pub continuation: Vec<BoardMove>,
    /// A line estimate, never a forced outcome. None is unknown, not draw.
    pub root_wdl: Option<[f32; 3]>,
    pub terminal: Option<(TerminalReason, Option<Color>)>,
}

#[derive(Clone, Debug)]
pub struct RoventResult {
    pub best_move: Option<BoardMove>,
    pub completion: RoventCompletion,
    pub terminal: Option<TerminalReason>,
    pub horizon: HorizonSnapshot,
    pub mate: Option<RootMateConclusion>,
    pub mate_strategy: Option<super::rovent_solved::RootMateStrategy>,
    pub candidates: Vec<RoventCandidate>,
    pub work: RoventWork,
    pub mode: ExecutionMode,
    pub root_restricted: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct RoventProgress {
    pub best_move: BoardMove,
    pub work: RoventWork,
    pub horizon: HorizonSnapshot,
}

#[derive(Clone, Debug)]
pub struct RoventShutdown {
    /// A successful synchronous RoleModel return carries its existing physical
    /// completion contract. An unknown completion permanently closes this owner.
    pub model_completion_known: bool,
    pub researcher: Option<CheckerShutdown>,
}

#[derive(Debug)]
pub enum RoventError {
    Invalid(&'static str),
    Unsupported(&'static str),
    Contract(rz_contracts::ContractError),
    Rules(PositionError),
    Role(RoleError),
    Checker(CheckerError),
    Store(StoreError),
    Horizon(HorizonError),
    Mate(MateProofError),
}

impl std::fmt::Display for RoventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rovent: {self:?}")
    }
}
impl std::error::Error for RoventError {}

macro_rules! error_from {
    ($variant:ident, $kind:ty) => {
        impl From<$kind> for RoventError {
            fn from(error: $kind) -> Self {
                Self::$variant(error)
            }
        }
    };
}
error_from!(Rules, PositionError);
error_from!(Role, RoleError);
error_from!(Checker, CheckerError);
error_from!(Store, StoreError);
error_from!(Horizon, HorizonError);
error_from!(Mate, MateProofError);
error_from!(Contract, rz_contracts::ContractError);

enum Researcher {
    Absent,
    Native(Box<dyn CpuChecker>),
}

/// Immutable legal domain issued at the original root admission and retained
/// by every child question. Its Arc must come from the manager's original go,
/// rather than a later widened domain. Neither a digest nor an old LineId can
/// reconstruct that original authority for a consumer.
pub struct RootMoveDomain {
    root: StateIdentity,
    snapshot: PositionSnapshot,
    allowed: Arc<[BoardMove]>,
}

impl RootMoveDomain {
    pub fn new(root: &ContractState, requested: Option<&[BoardMove]>) -> Result<Self, RoventError> {
        let position = Position::from_snapshot(root.rules().snapshot(), Default::default())?;
        let legal = position.ordered_legal_moves();
        let allowed = canonical_root_moves(legal.moves(), requested)?;
        Ok(Self {
            root: root.snapshot().identity(),
            snapshot: position.snapshot(),
            allowed: allowed.into(),
        })
    }
    pub fn root_identity(&self) -> StateIdentity {
        self.root
    }
    pub fn allowed(&self) -> &Arc<[BoardMove]> {
        &self.allowed
    }
}

/// Retains the exact caller-issued question/view rather than substituting the
/// newest public context. The immutable view retains its existing Store pin.
pub struct AdmittedRoleQuestion {
    envelope: QuestionEnvelope,
    root_moves: Arc<RootMoveDomain>,
    root_rules: Arc<ContractState>,
    rules: Arc<ContractState>,
    view: Arc<ReviewView>,
    context: RoleLogicalContext,
}

impl AdmittedRoleQuestion {
    pub fn envelope(&self) -> &QuestionEnvelope {
        &self.envelope
    }
    pub fn root_moves(&self) -> &Arc<RootMoveDomain> {
        &self.root_moves
    }
    pub fn root_rules(&self) -> &Arc<ContractState> {
        &self.root_rules
    }
    pub fn rules(&self) -> &Arc<ContractState> {
        &self.rules
    }
    pub fn view(&self) -> &Arc<ReviewView> {
        &self.view
    }
    pub fn context(&self) -> &RoleLogicalContext {
        &self.context
    }
    /// This is an admission capability for the native ticket facade to consume;
    /// this compatibility engine does not turn it into a fabricated model input.
    #[allow(clippy::too_many_arguments)]
    pub fn validate(
        envelope: QuestionEnvelope,
        root_moves: Arc<RootMoveDomain>,
        root_rules: Arc<ContractState>,
        rules: Arc<ContractState>,
        view: Arc<ReviewView>,
        context: RoleLogicalContext,
        stores: &PalsStores,
        mode: ExecutionMode,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Self, RoventError> {
        envelope.validate(mode, clock, now, view.token_budget())?;
        let state = stores.states.get(context.state)?;
        let root_state = stores.states.get(view.root_state())?;
        let situation = stores.situations.get(context.situation)?;
        if root_moves.root != root_rules.snapshot().identity()
            || !root_moves
                .snapshot
                .same_state(root_rules.rules().snapshot())
        {
            return Err(RoventError::Invalid(
                "question root move domain belongs to another exact root",
            ));
        }
        if envelope.anchor != rules.snapshot().identity()
            || envelope.view.root_state != root_rules.snapshot().identity()
            || !root_state.same_state(root_rules.rules().snapshot())
            || !state.same_state(rules.rules().snapshot())
            || situation.state != context.state
            || situation.focus != context.focus
            || !view.validates(stores)
            || view.sequence() != envelope.view.view_sequence
            || view.seal_digest() != envelope.view.view_digest.0
            || view.policy_digest() != envelope.view.context_policy.0
            || view.token_count() != envelope.view.actual_tokens
            || view.records().len() != usize::from(envelope.view.selected_records)
            || view.anchors().len() != usize::from(envelope.view.anchors)
            || view.length_bucket() != envelope.view.length_bucket
            || context.game_generation != envelope.id.authority.game.0
            || context.search_generation != envelope.id.authority.root.0
            || view.generation() != envelope.id.authority.root.0
            || context.public_revision != envelope.evidence_revision
            || context.purpose != view.query().purpose
            || context.prefix != view.query().prefix
        {
            return Err(RoventError::Invalid(
                "question, Rules, Store and immutable view differ",
            ));
        }
        validate_question_path(
            &envelope,
            &root_moves,
            rules.rules().snapshot(),
            &context.prefix,
        )?;
        if let Some(expected) = envelope.line_digest {
            let line = stores.lines.moves(context.focus)?;
            let line_start = stores.lines.get(context.focus)?.start_state;
            let offset = usize::from(envelope.line_offset.0);
            if !stores
                .states
                .get(line_start)?
                .same_state(root_rules.rules().snapshot())
                || rovent_line_digest(&line)? != expected
                || line.get(..offset) != Some(context.prefix.as_slice())
            {
                return Err(RoventError::Invalid(
                    "question line digest, root anchor or offset differs from the retained Store line",
                ));
            }
        }
        Ok(Self {
            envelope,
            root_moves,
            root_rules,
            rules,
            view,
            context,
        })
    }
}

/// New Rovent domain only. Existing PALS context/digest profiles are unchanged.
pub fn rovent_prefix_digest(prefix: &[BoardMove]) -> Result<Digest, RoventError> {
    rovent_rules_sequence_digest(ROVENT_PREFIX_DIGEST_DOMAIN, prefix)
}

pub fn rovent_line_digest(line: &[BoardMove]) -> Result<Digest, RoventError> {
    rovent_rules_sequence_digest(ROVENT_LINE_DIGEST_DOMAIN, line)
}

fn rovent_rules_sequence_digest(domain: &str, moves: &[BoardMove]) -> Result<Digest, RoventError> {
    if moves.len() > 256 {
        return Err(RoventError::Invalid(
            "Rovent Rules sequence exceeds the registered 256-ply bound",
        ));
    }
    let mut hash = Sha256::new();
    hash.update((domain.len() as u64).to_le_bytes());
    hash.update(domain.as_bytes());
    hash.update((moves.len() as u64).to_le_bytes());
    for movement in moves {
        let movement = rz_contracts::Move::try_from(*movement)?;
        hash.update(
            rz_contracts::pals::Move16::encode(movement)
                .bits()
                .to_le_bytes(),
        );
    }
    Ok(Digest(hash.finalize().into()))
}

fn validate_question_path(
    envelope: &QuestionEnvelope,
    root_moves: &RootMoveDomain,
    anchor: &PositionSnapshot,
    prefix: &[BoardMove],
) -> Result<(), RoventError> {
    if prefix.len() != usize::from(envelope.anchor_root_ply.0)
        || rovent_prefix_digest(prefix)? != envelope.prefix_digest
    {
        return Err(RoventError::Invalid(
            "question prefix digest or root-relative chess ply differs from its actual Rules path",
        ));
    }
    let original_first = envelope
        .original_first_move
        .map(|movement| movement.decode().and_then(BoardMove::try_from))
        .transpose()?;
    if original_first.is_some_and(|first| !root_moves.allowed.contains(&first))
        || prefix
            .first()
            .is_some_and(|first| original_first != Some(*first))
    {
        return Err(RoventError::Invalid(
            "question original first move differs from its root restriction or Rules prefix",
        ));
    }
    let mut position = Position::from_snapshot(&root_moves.snapshot, Default::default())?;
    for movement in prefix {
        let legal = position.ordered_legal_moves();
        if position.play_status_from_view(&legal)? != PlayStatus::Ongoing {
            return Err(RoventError::Invalid(
                "question prefix continues after a Rules terminal",
            ));
        }
        position.make_move(*movement)?;
    }
    if !position.snapshot().same_state(anchor) {
        return Err(RoventError::Invalid(
            "question Rules prefix does not reach its exact anchor and history",
        ));
    }
    Ok(())
}

pub struct RoventEngine<M: RoleModel> {
    config: RoventConfig,
    model: M,
    researcher: Researcher,
    game_generation: u64,
    search_generation: u64,
    physical_completion_unknown: bool,
    stores: Option<PalsStores>,
    registered_model: String,
    registered_value: Option<ModelValueIdentity>,
}

impl<M: RoleModel> RoventEngine<M> {
    /// No CPU searcher/checker or external process can be supplied on this path.
    pub fn pc(config: RoventConfig, model: M, actual: PcRegistration) -> Result<Self, RoventError> {
        Self::assemble(config, model, actual, Researcher::Absent)
    }

    pub fn pcr<R: CpuSearcher + 'static>(
        config: RoventConfig,
        model: M,
        actual: PcRegistration,
        native_r: R,
    ) -> Result<Self, RoventError> {
        Self::assemble(
            config,
            model,
            actual,
            Researcher::Native(Box::new(OwnedCpuChecker::new(native_r)?)),
        )
    }

    fn assemble(
        config: RoventConfig,
        model: M,
        actual: PcRegistration,
        researcher: Researcher,
    ) -> Result<Self, RoventError> {
        config.validate()?;
        config
            .mode
            .validate_assembly(actual.compute, matches!(researcher, Researcher::Native(_)))?;
        if actual.model_identity != model.identity() || actual.model_identity.trim().is_empty() {
            return Err(RoventError::Invalid(
                "P/C startup identity differs from loaded provider",
            ));
        }
        if let Researcher::Native(owner) = &researcher {
            owner.validate_namespace()?;
            if !matches!(owner.identity(), CheckerIdentity::Owned(_))
                || owner.owned_descriptor().is_none()
            {
                return Err(RoventError::Invalid(
                    "P/C/R requires the actual own R owner",
                ));
            }
        }
        let registered_value = model.value_identity().cloned();
        if let Some(identity) = &registered_value {
            identity.validate()?;
        }
        let stores = Some(new_stores(config.maximum_positions));
        Ok(Self {
            config,
            model,
            researcher,
            game_generation: 1,
            search_generation: 0,
            physical_completion_unknown: false,
            stores,
            registered_model: actual.model_identity,
            registered_value,
        })
    }

    pub fn mode(&self) -> ExecutionMode {
        self.config.mode
    }

    pub fn goal_capability(&self) -> NonZeroU16 {
        self.config.maximum_goal_plies
    }

    pub fn model_identity(&self) -> &str {
        self.model.identity()
    }

    /// Configuration identity comes from this assembled owner, not a caller
    /// label. The source pin is separate from model/input semantic identities.
    pub fn implementation_identity(&self) -> Result<Digest, RoventError> {
        let mut hash = Sha256::new();
        let mut part = |bytes: &[u8]| {
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        };
        part(ROVENT_SYNC_SEARCH_VERSION.as_bytes());
        part(ROVENT_NODE_UNIT.as_bytes());
        part(include_str!("rovent_engine.rs").as_bytes());
        part(&[match self.config.mode {
            ExecutionMode::HybridPcr => 0,
            ExecutionMode::HybridPc => 1,
            ExecutionMode::CpuPcr => 2,
            ExecutionMode::CpuPc => 3,
        }]);
        for value in [
            u64::from(self.config.maximum_goal_plies.get()),
            self.config.candidate_width as u64,
            u64::from(self.config.normal_line_plies.get()),
            self.config.maximum_positions as u64,
            self.config.maximum_role_calls,
            u64::from(self.config.r_depth.get()),
            self.config.r_nodes_per_call,
        ] {
            part(&value.to_le_bytes());
        }
        part(&self.config.selective_policy.0);
        part(self.registered_model.as_bytes());
        match &self.registered_value {
            None => part(&[0]),
            Some(identity) => {
                part(&[1]);
                for text in [
                    &identity.semantics,
                    &identity.model,
                    &identity.encoding,
                    &identity.precision,
                ] {
                    part(text.as_bytes());
                }
                part(&identity.model_epoch);
            }
        }
        match &self.researcher {
            Researcher::Absent => part(&[0]),
            Researcher::Native(owner) => {
                part(&[1]);
                let identity = serde_json::to_vec(owner.identity())
                    .map_err(|_| RoventError::Invalid("own R identity serialization failed"))?;
                part(&identity);
                part(owner.conditions().as_bytes());
            }
        }
        Ok(Digest(hash.finalize().into()))
    }

    pub fn requires_admitted_ticket_facade(&self) -> bool {
        self.model.requires_registered_representation_namespace()
    }

    pub fn new_game(&mut self, deadline: Instant, cancel: &AtomicBool) -> Result<(), RoventError> {
        if self.physical_completion_unknown {
            return Err(RoleError::PhysicalCompletionUnknown.into());
        }
        if let Err(error) = self.model.preflight_new_game(deadline, cancel) {
            if error == RoleError::PhysicalCompletionUnknown {
                self.physical_completion_unknown = true;
            }
            return Err(error.into());
        }
        if let Researcher::Native(owner) = &mut self.researcher {
            owner.new_game(deadline, cancel)?;
        }
        let next = self
            .game_generation
            .checked_add(1)
            .ok_or(RoventError::Invalid("game generation exhausted"))?;
        self.model.new_game_with_generation(Some(next));
        self.game_generation = next;
        self.search_generation = 0;
        self.stores = Some(new_stores(self.config.maximum_positions));
        Ok(())
    }

    pub fn shutdown(&mut self, deadline: Instant) -> Result<RoventShutdown, RoventError> {
        let researcher = match &mut self.researcher {
            Researcher::Absent => None,
            Researcher::Native(owner) => Some(owner.shutdown(deadline)?),
        };
        Ok(RoventShutdown {
            model_completion_known: !self.physical_completion_unknown,
            researcher,
        })
    }

    /// Synchronous compatibility search. Native selected-view providers use the
    /// admitted ticket path; no view, StateIdentity or digest is invented here.
    pub fn search(
        &mut self,
        root: &Position,
        root_moves: Option<&[BoardMove]>,
        limits: RoventLimits,
        cancel: &AtomicBool,
    ) -> Result<RoventResult, RoventError> {
        self.search_with_progress(root, root_moves, limits, cancel, |_| {})
    }

    pub fn search_with_progress(
        &mut self,
        root: &Position,
        root_moves: Option<&[BoardMove]>,
        limits: RoventLimits,
        cancel: &AtomicBool,
        mut progress: impl FnMut(RoventProgress),
    ) -> Result<RoventResult, RoventError> {
        if self.physical_completion_unknown {
            return Err(RoleError::PhysicalCompletionUnknown.into());
        }
        if self.model.identity() != self.registered_model
            || self.model.value_identity() != self.registered_value.as_ref()
        {
            return Err(RoventError::Invalid(
                "P/C identity changed after startup registration",
            ));
        }
        limits.goal.validate(self.config.maximum_goal_plies)?;
        if limits.maximum_role_calls == 0 {
            return Err(RoventError::Invalid("zero root role budget"));
        }
        if self.model.requires_registered_representation_namespace()
            || self.model.supports_selected_review_view()
        {
            // A selected-view capable fixture can still implement plain calls,
            // but silently routing a native consumer through those is unsafe.
            if self.model.requires_registered_representation_namespace() {
                return Err(RoventError::Unsupported(
                    "native selected-view provider requires the admitted async ticket facade",
                ));
            }
        }
        self.search_generation = self
            .search_generation
            .checked_add(1)
            .ok_or(RoventError::Invalid("search generation exhausted"))?;
        let stores = self
            .stores
            .take()
            .ok_or(RoventError::Invalid("Store owner is already borrowed"))?;
        let mut state = SearchState {
            limits,
            cancel,
            work: RoventWork {
                native_see_nodes: Some(0),
                native_pvs_entries: Some(0),
                ..Default::default()
            },
            stores,
            positions: 0,
            maximum_positions: self.config.maximum_positions,
            maximum_examined_ply: ChessPly(0),
        };
        let outcome = self.search_inner(root, root_moves, &mut state, &mut progress);
        // Even a failed search returns the same exact Store owner and evidence.
        self.stores = Some(state.stores);
        let closure = match &outcome {
            Ok(result) => match result.completion {
                RoventCompletion::Canceled => RoleSearchClosure::Canceled,
                RoventCompletion::Deadline => RoleSearchClosure::Deadline,
                _ => RoleSearchClosure::Completed,
            },
            Err(RoventError::Role(RoleError::PhysicalCompletionUnknown)) => {
                RoleSearchClosure::PhysicalCompletionUnknown
            }
            Err(_) => RoleSearchClosure::Failed,
        };
        if closure == RoleSearchClosure::PhysicalCompletionUnknown {
            self.physical_completion_unknown = true;
        }
        self.model.finish_search(closure);
        outcome
    }

    fn search_inner(
        &mut self,
        root: &Position,
        root_moves: Option<&[BoardMove]>,
        state: &mut SearchState<'_>,
        progress: &mut dyn FnMut(RoventProgress),
    ) -> Result<RoventResult, RoventError> {
        let limits = state.limits;
        let legal_view = root.ordered_legal_moves();
        let legal = legal_view.moves();
        let selected = canonical_root_moves(legal, root_moves)?;
        let mut horizon = SelectiveHorizon::new(
            root,
            limits.goal,
            self.config.selective_policy,
            self.config.maximum_positions,
        )?;
        let status = root.play_status_from_view(&legal_view)?;
        if let PlayStatus::Terminal { reason, .. } = status {
            let completion = match state.enter_root() {
                Ok(()) => RoventCompletion::RulesTerminal,
                Err(RunError::Interrupted(completion)) => completion,
                Err(RunError::Engine(error)) => return Err(error),
            };
            return Ok(RoventResult {
                best_move: None,
                completion,
                terminal: Some(reason),
                horizon: horizon.snapshot(),
                mate: None,
                mate_strategy: None,
                candidates: Vec::new(),
                work: state.work,
                mode: self.config.mode,
                root_restricted: false,
            });
        }
        let fallback = selected.first().copied();
        let mut result = RoventResult {
            best_move: fallback,
            completion: RoventCompletion::SelectiveFrontier,
            terminal: None,
            horizon: horizon.snapshot(),
            mate: None,
            mate_strategy: None,
            candidates: Vec::new(),
            work: state.work,
            mode: self.config.mode,
            root_restricted: selected.len() != legal.len(),
        };
        let inner = (|| -> Result<(), RunError> {
            state.enter_root()?;
            let ranked: Vec<_> = self
                .rank(root, legal, &[], root.side_to_move(), false, state)?
                .into_iter()
                .filter(|movement| selected.contains(movement))
                .collect();
            let root_anchor = horizon.root_anchor();
            for movement in ranked.into_iter().take(self.config.candidate_width) {
                state.control()?;
                state.transition(ChessPly(1))?;
                let child_anchor = horizon.advance(&root_anchor, root, movement)?;
                let mut position =
                    Position::from_snapshot(child_anchor.snapshot(), Default::default())?;
                let obligation = if limits.goal.depth_plies.is_some() {
                    Some(horizon.declare(&child_anchor, HorizonObligationKind::Candidate)?)
                } else {
                    None
                };
                let mut anchor = child_anchor;
                let mut prefix = vec![movement];
                let line_limit = limits
                    .goal
                    .depth_plies
                    .map_or(self.config.normal_line_plies.get(), NonZeroU16::get);
                let mut terminal = None;
                loop {
                    state.control()?;
                    let view = position.ordered_legal_moves();
                    if let PlayStatus::Terminal { reason, winner } =
                        position.play_status_from_view(&view)?
                    {
                        terminal = Some((reason, winner));
                        if let Some(id) = obligation {
                            horizon.finish(
                                id,
                                &anchor,
                                &HorizonBoundary::rules_terminal(&position)?,
                            )?;
                        }
                        break;
                    }
                    if anchor.ply().0 >= line_limit {
                        break;
                    }
                    state.control()?;
                    let mut next = self
                        .rank(
                            &position,
                            view.moves(),
                            &prefix,
                            root.side_to_move(),
                            false,
                            state,
                        )?
                        .into_iter()
                        .next()
                        .ok_or(RoventError::Invalid(
                            "ongoing Rules state has no legal policy move",
                        ))?;
                    if let Some(research) =
                        self.r_move(&position, view.moves(), anchor.ply(), limits.goal, state)?
                    {
                        next = research;
                    }
                    state.transition(anchor.ply().advance(1)?)?;
                    let new_anchor = horizon.advance(&anchor, &position, next)?;
                    // `advance` already executed this exact Rules transition;
                    // retain its snapshot instead of applying the move twice.
                    position = Position::from_snapshot(new_anchor.snapshot(), Default::default())?;
                    anchor = new_anchor;
                    prefix.push(next);
                }
                let wdl = if terminal.is_none() {
                    let output = self.fresh_value(&position, &prefix, state)?;
                    if let Some(id) = obligation {
                        let identity = self
                            .registered_value
                            .as_ref()
                            .ok_or(RoleError::Unavailable)?;
                        horizon.finish(
                            id,
                            &anchor,
                            &HorizonBoundary::fresh_value(&position, &output, identity)?,
                        )?;
                    }
                    Some(if position.side_to_move() == root.side_to_move() {
                        output.wdl
                    } else {
                        [output.wdl[2], output.wdl[1], output.wdl[0]]
                    })
                } else {
                    None
                };
                result.candidates.push(RoventCandidate {
                    movement,
                    continuation: prefix,
                    root_wdl: wdl,
                    terminal,
                });
                result.best_move =
                    choose_candidate(&result.candidates, root.side_to_move()).or(fallback);
                if let Some(best_move) = result.best_move {
                    state.control()?;
                    progress(RoventProgress {
                        best_move,
                        work: state.work,
                        horizon: horizon.snapshot(),
                    });
                }
            }
            if limits.goal.depth_plies.is_some() {
                state.control()?;
                horizon.seal_policy()?;
                if horizon.snapshot().target_satisfied {
                    result.completion = RoventCompletion::DepthGoal;
                }
            }
            if limits.goal.mate_moves.is_some() {
                for attacker in [root.side_to_move(), root.side_to_move().opposite()] {
                    state.control()?;
                    let remaining = limits
                        .goal
                        .mate_horizon(attacker == root.side_to_move())?
                        .ok_or(RoventError::Invalid("missing mate horizon"))?;
                    let mut proof = MateProofStore::new(
                        root,
                        attacker,
                        remaining,
                        root_moves,
                        self.config.maximum_positions,
                    )?;
                    let proof_root = proof.root();
                    self.prove_mate(
                        &mut proof,
                        proof_root,
                        &mut Vec::new(),
                        root.side_to_move(),
                        state,
                    )?;
                    state.control()?;
                    let conclusion = proof.root_conclusion();
                    if conclusion.goal_satisfied() {
                        let strategy = proof.root_strategy();
                        if let Some(strategy) = &strategy {
                            result.best_move = Some(strategy.best_move());
                        }
                        result.mate_strategy = strategy;
                        result.mate = Some(conclusion);
                        result.completion = RoventCompletion::MateGoal;
                        break;
                    }
                    result.mate = Some(conclusion);
                }
                if !result.mate.is_some_and(RootMateConclusion::goal_satisfied) {
                    result.completion = RoventCompletion::GoalUnresolved;
                }
            }
            Ok(())
        })();
        if let Err(error) = inner {
            let interruption = match error {
                RunError::Interrupted(completion) => completion,
                RunError::Engine(RoventError::Role(RoleError::Canceled)) => {
                    RoventCompletion::Canceled
                }
                RunError::Engine(RoventError::Role(RoleError::Deadline)) => {
                    RoventCompletion::Deadline
                }
                RunError::Engine(RoventError::Store(StoreError::Capacity(_)))
                | RunError::Engine(RoventError::Mate(MateProofError::Capacity))
                | RunError::Engine(RoventError::Horizon(HorizonError::Capacity)) => {
                    RoventCompletion::Capacity
                }
                RunError::Engine(error) => return Err(error),
            };
            horizon.interrupt(match interruption {
                RoventCompletion::Canceled => HorizonInterruption::Stop,
                RoventCompletion::Deadline => HorizonInterruption::Deadline,
                RoventCompletion::Capacity => HorizonInterruption::Capacity,
                _ => HorizonInterruption::WorkBudget,
            });
            result.completion = interruption;
        }
        result.horizon = horizon.snapshot();
        result.horizon.maximum_examined_ply = result
            .horizon
            .maximum_examined_ply
            .max(state.maximum_examined_ply);
        result.work = state.work;
        Ok(result)
    }

    fn rank(
        &mut self,
        position: &Position,
        legal: &[BoardMove],
        prefix: &[BoardMove],
        root_side: Color,
        repair: bool,
        state: &mut SearchState<'_>,
    ) -> Result<Vec<BoardMove>, RunError> {
        state.role(self.config.maximum_role_calls)?;
        let purpose = if position.side_to_move() != root_side {
            RoleQueryPurpose::ReplyPolicy
        } else if repair {
            RoleQueryPurpose::RepairPolicy
        } else {
            RoleQueryPurpose::ProposePolicy
        };
        let context = state.context(
            position,
            prefix,
            purpose,
            self.game_generation,
            self.search_generation,
        )?;
        let query = RoleQuery {
            position,
            legal,
            prefix,
            proposal: prefix,
            counterexample: None,
            records: &[],
            revision: 0,
            deadline: state.limits.deadline,
            cancel: state.cancel,
        };
        let output = match purpose {
            RoleQueryPurpose::ReplyPolicy => self.model.reply_with_context(query, &context),
            RoleQueryPurpose::RepairPolicy => self.model.repair_with_context(query, &context),
            _ => self.model.propose_with_context(query, &context),
        }?;
        validate_policy(&output, legal.len())?;
        state.control()?;
        self.model.accepted_output_checked(RoleAcceptance {
            snapshot: &position.snapshot(),
            context: &context,
            deadline: state.limits.deadline,
            cancel: state.cancel,
        })?;
        state.work.accepted_policy += 1;
        let mut indices: Vec<usize> = (0..legal.len()).collect();
        indices.sort_by(|left, right| {
            output.logits[*right]
                .total_cmp(&output.logits[*left])
                .then(left.cmp(right))
        });
        Ok(indices.into_iter().map(|index| legal[index]).collect())
    }

    fn fresh_value(
        &mut self,
        position: &Position,
        prefix: &[BoardMove],
        state: &mut SearchState<'_>,
    ) -> Result<ModelValueOutput, RunError> {
        state.role(self.config.maximum_role_calls)?;
        let identity = self
            .registered_value
            .as_ref()
            .ok_or(RoleError::Unavailable)?
            .clone();
        let context = state.context(
            position,
            prefix,
            RoleQueryPurpose::ValueFresh,
            self.game_generation,
            self.search_generation,
        )?;
        let legal = position.ordered_legal_moves();
        let output = self.model.evaluate_value_with_context(
            RoleQuery {
                position,
                legal: legal.moves(),
                prefix,
                proposal: prefix,
                counterexample: None,
                records: &[],
                revision: 0,
                deadline: state.limits.deadline,
                cancel: state.cancel,
            },
            &context,
        )?;
        output.validate(position, &identity)?;
        state.control()?;
        self.model.accepted_output_checked(RoleAcceptance {
            snapshot: &position.snapshot(),
            context: &context,
            deadline: state.limits.deadline,
            cancel: state.cancel,
        })?;
        state.work.accepted_fresh_values += 1;
        Ok(output)
    }

    fn r_move(
        &mut self,
        position: &Position,
        legal: &[BoardMove],
        ply: ChessPly,
        goal: SearchGoal,
        state: &mut SearchState<'_>,
    ) -> Result<Option<BoardMove>, RunError> {
        let Researcher::Native(owner) = &mut self.researcher else {
            return Ok(None);
        };
        state.control()?;
        let native_remaining = state
            .limits
            .maximum_native_nodes
            .saturating_sub(state.work.native_nodes);
        let remaining = native_remaining.min(state.remaining_nodes()?);
        if remaining == 0 {
            return Err(RunError::Interrupted(RoventCompletion::WorkBudget));
        }
        let local_depth = goal
            .normal_remaining(ply)
            .map_or(self.config.r_depth.get(), |left| {
                left.0.min(self.config.r_depth.get())
            });
        if local_depth == 0 {
            return Ok(None);
        }
        owner.reset_attempt();
        state.work.native_r_calls += 1;
        state.work.native_see_nodes = None;
        state.work.native_pvs_entries = None;
        let granted = remaining.min(self.config.r_nodes_per_call);
        let report = owner.analyze_root_moves(
            position,
            legal,
            CpuLimits {
                max_depth: local_depth,
                max_nodes: granted,
                deadline: Some(state.limits.deadline),
            },
            state.cancel,
        );
        // Failed/partial owner work is charged too. Unknown cannot be refunded.
        let observed = owner
            .last_attempt()
            .and_then(|attempt| attempt.work.nodes)
            .ok_or(RoventError::Invalid(
                "native R work is unknown after dispatch",
            ))?;
        state.work.native_nodes = state
            .work
            .native_nodes
            .checked_add(observed)
            .ok_or(RoventError::Invalid("native node counter overflow"))?;
        let qnodes = owner
            .last_attempt()
            .and_then(|attempt| attempt.work.qnodes)
            .ok_or(RoventError::Invalid(
                "native R qsearch work is unknown after dispatch",
            ))?;
        state.work.native_qsearch_nodes = state
            .work
            .native_qsearch_nodes
            .checked_add(qnodes)
            .ok_or(RoventError::Invalid("native qsearch counter overflow"))?;
        if observed > granted {
            return Err(RoventError::Invalid(
                "native R exceeded its original common-node reservation",
            )
            .into());
        }
        let CheckerReport::Owned(report) = report? else {
            return Err(RoventError::Invalid("native R returned external evidence").into());
        };
        state.control()?;
        if report.nodes != observed
            || report.quiescence_nodes != qnodes
            || report
                .best_move
                .is_some_and(|movement| !legal.contains(&movement))
        {
            return Err(RoventError::Invalid(
                "native R report differs from observed work or Rules domain",
            )
            .into());
        }
        Ok(report.best_move)
    }

    fn prove_mate(
        &mut self,
        proof: &mut MateProofStore,
        node: MateNodeId,
        prefix: &mut Vec<BoardMove>,
        root_side: Color,
        state: &mut SearchState<'_>,
    ) -> Result<(), RunError> {
        state.control()?;
        if proof.conclusion(node)? != MateConclusion::Unknown {
            return Ok(());
        }
        // The original mate horizon stays registered in the proof owner. A
        // simultaneous depth cap only stops new work; it cannot manufacture
        // NoMateWithin for a smaller, substituted mate question.
        let root_ply = u16::try_from(prefix.len()).map(ChessPly).map_err(|_| {
            RoventError::Invalid("mate Rules prefix exceeds chess-ply representation")
        })?;
        if state
            .limits
            .goal
            .depth_plies
            .is_some_and(|depth| root_ply.0 >= depth.get())
        {
            return Ok(());
        }
        let position = Position::from_snapshot(proof.snapshot(node)?, Default::default())?;
        let missing = proof.missing_moves(node)?;
        if missing.is_empty() {
            return Ok(());
        }
        let legal = position.ordered_legal_moves();
        let ranked = self.rank(&position, legal.moves(), prefix, root_side, false, state)?;
        for movement in ranked
            .into_iter()
            .filter(|movement| missing.contains(movement))
        {
            state.control()?;
            let child = match proof.child(node, movement)? {
                Some(child) => child,
                None => {
                    state.transition(root_ply.advance(1)?)?;
                    proof.open_child(node, movement)?
                }
            };
            prefix.push(movement);
            let result = self.prove_mate(proof, child, prefix, root_side, state);
            prefix.pop();
            result?;
            if proof.conclusion(node)? != MateConclusion::Unknown {
                break;
            }
        }
        Ok(())
    }
}

struct SearchState<'a> {
    limits: RoventLimits,
    cancel: &'a AtomicBool,
    work: RoventWork,
    stores: PalsStores,
    positions: usize,
    maximum_positions: usize,
    maximum_examined_ply: ChessPly,
}

impl SearchState<'_> {
    fn control(&self) -> Result<(), RunError> {
        if self.cancel.load(Ordering::Acquire) {
            return Err(RunError::Interrupted(RoventCompletion::Canceled));
        }
        if Instant::now() >= self.limits.deadline {
            return Err(RunError::Interrupted(RoventCompletion::Deadline));
        }
        if self.remaining_nodes()? == 0 {
            return Err(RunError::Interrupted(RoventCompletion::WorkBudget));
        }
        Ok(())
    }
    fn remaining_nodes(&self) -> Result<u64, RunError> {
        let spent = self
            .work
            .total_node_visits()
            .ok_or(RoventError::Invalid("common node accounting overflow"))?;
        Ok(self
            .limits
            .maximum_node_visits
            .map_or(u64::MAX.saturating_sub(spent), |limit| {
                limit.saturating_sub(spent)
            }))
    }
    fn enter_root(&mut self) -> Result<(), RunError> {
        self.control()?;
        if self.work.root_entries != 0 {
            return Err(RoventError::Invalid("root visit already admitted").into());
        }
        self.work.root_entries = 1;
        Ok(())
    }
    fn role(&mut self, configured: u64) -> Result<(), RunError> {
        self.control()?;
        if self.work.role_calls >= self.limits.maximum_role_calls.min(configured) {
            return Err(RunError::Interrupted(RoventCompletion::WorkBudget));
        }
        self.work.role_calls += 1;
        Ok(())
    }
    fn transition(&mut self, root_ply: ChessPly) -> Result<(), RunError> {
        self.control()?;
        if self.positions >= self.maximum_positions {
            return Err(RunError::Interrupted(RoventCompletion::Capacity));
        }
        self.positions = self
            .positions
            .checked_add(1)
            .ok_or(RoventError::Invalid("position counter overflow"))?;
        self.work.checked_transitions = self
            .work
            .checked_transitions
            .checked_add(1)
            .ok_or(RoventError::Invalid("Rules transition counter overflow"))?;
        self.maximum_examined_ply = self.maximum_examined_ply.max(root_ply);
        Ok(())
    }
    fn context(
        &mut self,
        position: &Position,
        prefix: &[BoardMove],
        purpose: RoleQueryPurpose,
        game: u64,
        search: u64,
    ) -> Result<RoleLogicalContext, RoventError> {
        let situation = self.stores.insert_situation(position.snapshot())?;
        let stored = self.stores.situations.get(situation)?;
        let digest = line_digest(prefix);
        Ok(RoleLogicalContext {
            game_generation: game,
            search_generation: search,
            situation,
            state: stored.state,
            focus: stored.focus,
            purpose,
            prefix: prefix.to_vec(),
            focus_sha256: line_digest(&[]),
            prefix_sha256: digest,
            proposal_sha256: digest,
            refutation_sha256: None,
            divergence_sha256: line_digest(&[]),
            public_revision: 0,
            situation_revision: stored.revision,
        })
    }
}

#[derive(Debug)]
enum RunError {
    Engine(RoventError),
    Interrupted(RoventCompletion),
}
impl From<RoventError> for RunError {
    fn from(error: RoventError) -> Self {
        Self::Engine(error)
    }
}
macro_rules! run_from {
    ($kind:ty) => {
        impl From<$kind> for RunError {
            fn from(error: $kind) -> Self {
                Self::Engine(error.into())
            }
        }
    };
}
run_from!(PositionError);
run_from!(RoleError);
run_from!(CheckerError);
run_from!(HorizonError);
run_from!(MateProofError);
run_from!(rz_contracts::ContractError);

fn canonical_root_moves(
    legal: &[BoardMove],
    requested: Option<&[BoardMove]>,
) -> Result<Vec<BoardMove>, RoventError> {
    let Some(requested) = requested else {
        return Ok(legal.to_vec());
    };
    if requested.is_empty()
        || requested
            .iter()
            .enumerate()
            .any(|(at, movement)| !legal.contains(movement) || requested[..at].contains(movement))
    {
        return Err(RoventError::Invalid(
            "root restriction is empty, duplicate or illegal",
        ));
    }
    Ok(legal
        .iter()
        .copied()
        .filter(|movement| requested.contains(movement))
        .collect())
}

fn new_stores(maximum_positions: usize) -> PalsStores {
    PalsStores::new(StoreLimits {
        states: maximum_positions,
        situations: maximum_positions,
        ..Default::default()
    })
}

fn validate_policy(output: &RoleEvaluation, candidates: usize) -> Result<(), RoleError> {
    if output.logits.len() != candidates
        || output.logits.iter().any(|value| !value.is_finite())
        || output
            .wdl
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || (output.wdl.iter().sum::<f32>() - 1.0).abs() > 1e-4
    {
        return Err(RoleError::InvalidOutput);
    }
    Ok(())
}

fn line_digest(line: &[BoardMove]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"rovent-rules-line/1\0");
    digest.update((line.len() as u64).to_le_bytes());
    for movement in line {
        digest.update(movement.to_string().as_bytes());
    }
    digest.finalize().into()
}

fn choose_candidate(candidates: &[RoventCandidate], root_side: Color) -> Option<BoardMove> {
    let value = |candidate: &RoventCandidate| -> Option<f32> {
        if let Some((_, winner)) = candidate.terminal {
            // The estimate applies to this examined line, not all replies.
            Some(match winner {
                Some(winner) if winner == root_side => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            })
        } else {
            candidate.root_wdl.map(|wdl| wdl[0] - wdl[2])
        }
    };
    let mut best: Option<(&RoventCandidate, f32)> = None;
    for candidate in candidates {
        if let Some(score) = value(candidate)
            && best.is_none_or(|(_, held)| score > held)
        {
            best = Some((candidate, score));
        }
    }
    best.map(|(candidate, _)| candidate.movement)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{CpuConfig, CpuEngine};
    use crate::pals::engine::{DivergenceQuery, LegalOrderRoleMock};
    use std::sync::Mutex;
    use std::time::Duration;

    fn config(mode: ExecutionMode) -> RoventConfig {
        RoventConfig {
            mode,
            maximum_goal_plies: NonZeroU16::new(32).unwrap(),
            candidate_width: 2,
            normal_line_plies: NonZeroU16::new(2).unwrap(),
            maximum_positions: 2048,
            maximum_role_calls: 2048,
            r_depth: NonZeroU16::new(1).unwrap(),
            r_nodes_per_call: 256,
            selective_policy: Digest([17; 32]),
        }
    }

    fn registration(mode: ExecutionMode) -> PcRegistration {
        PcRegistration {
            compute: mode.pc_compute(),
            model_identity: LegalOrderRoleMock.identity().into(),
        }
    }

    fn limits(depth: Option<u16>, mate: Option<u16>) -> RoventLimits {
        RoventLimits {
            goal: SearchGoal {
                depth_plies: depth.and_then(NonZeroU16::new),
                mate_moves: mate.and_then(NonZeroU16::new),
            },
            deadline: Instant::now() + Duration::from_secs(10),
            maximum_role_calls: 2048,
            maximum_native_nodes: 4096,
            maximum_node_visits: None,
        }
    }

    fn mv(text: &str) -> BoardMove {
        BoardMove::from_uci(text).unwrap()
    }

    #[test]
    fn both_pc_modes_complete_real_ply_without_constructing_or_calling_r() {
        for mode in [ExecutionMode::HybridPc, ExecutionMode::CpuPc] {
            // Accelerator here is a mock startup declaration, not GPU proof.
            let mut owner =
                RoventEngine::pc(config(mode), LegalOrderRoleMock, registration(mode)).unwrap();
            let restricted = [mv("e2e4"), mv("d2d4")];
            let mut published = Vec::new();
            let result = owner
                .search_with_progress(
                    &Position::startpos(),
                    Some(&restricted),
                    limits(Some(3), None),
                    &AtomicBool::new(false),
                    |progress| published.push(progress.best_move),
                )
                .unwrap();
            assert_eq!(result.completion, RoventCompletion::DepthGoal);
            assert_eq!(result.horizon.completed_depth, Some(ChessPly(3)));
            assert_eq!(result.horizon.declared_obligations, 2);
            assert_eq!(result.horizon.completed_obligations, 2);
            assert!(result.root_restricted);
            assert!(restricted.contains(&result.best_move.unwrap()));
            assert!(
                published
                    .iter()
                    .all(|movement| restricted.contains(movement))
            );
            assert_eq!(result.work.native_r_calls, 0);
            assert_eq!(result.work.native_nodes, 0);
            assert_eq!(result.work.native_pvs_entries, Some(0));
            assert_eq!(result.work.native_qsearch_nodes, 0);
            assert_eq!(result.work.native_see_nodes, Some(0));
            assert_eq!(result.work.external_teacher_calls, 0);
            assert_eq!(result.work.accepted_fresh_values, 2);
            assert!(
                owner
                    .shutdown(Instant::now() + Duration::from_secs(1))
                    .unwrap()
                    .researcher
                    .is_none()
            );
        }
    }

    #[test]
    fn pcr_modes_use_actual_owned_searcher_and_preserve_the_same_root_budget() {
        for mode in [ExecutionMode::HybridPcr, ExecutionMode::CpuPcr] {
            let native = CpuEngine::new(CpuConfig {
                max_depth: 2,
                tt_entries: 32,
                quiescence_ply: 2,
                ..Default::default()
            })
            .unwrap();
            let mut owner =
                RoventEngine::pcr(config(mode), LegalOrderRoleMock, registration(mode), native)
                    .unwrap();
            let result = owner
                .search(
                    &Position::startpos(),
                    Some(&[mv("e2e4")]),
                    limits(Some(2), None),
                    &AtomicBool::new(false),
                )
                .unwrap();
            assert_eq!(result.completion, RoventCompletion::DepthGoal);
            assert_eq!(result.best_move, Some(mv("e2e4")));
            assert_eq!(result.work.native_r_calls, 1);
            assert!(result.work.native_nodes > 0);
            assert!(result.work.native_nodes <= 4096);
            assert_eq!(result.work.native_pvs_entries, None);
            assert_eq!(result.work.native_see_nodes, None);
            assert_eq!(result.work.external_teacher_calls, 0);
        }
    }

    #[test]
    fn budget_cancel_and_time_only_never_fabricate_completed_depth() {
        let mode = ExecutionMode::CpuPc;
        let mut owner =
            RoventEngine::pc(config(mode), LegalOrderRoleMock, registration(mode)).unwrap();
        let canceled = owner
            .search(
                &Position::startpos(),
                Some(&[mv("g1f3")]),
                limits(Some(3), None),
                &AtomicBool::new(true),
            )
            .unwrap();
        assert_eq!(canceled.completion, RoventCompletion::Canceled);
        assert_eq!(canceled.best_move, Some(mv("g1f3")));
        assert_eq!(canceled.work.role_calls, 0);
        assert!(!canceled.horizon.target_satisfied);
        let mut small = limits(Some(3), None);
        small.maximum_role_calls = 1;
        let partial = owner
            .search(&Position::startpos(), None, small, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(partial.completion, RoventCompletion::WorkBudget);
        assert!(!partial.horizon.target_satisfied);
        let ordinary = owner
            .search(
                &Position::startpos(),
                Some(&[mv("e2e4")]),
                limits(None, None),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(ordinary.horizon.nominal_target, None);
        assert_eq!(ordinary.horizon.completed_depth, None);
        assert_eq!(ordinary.horizon.declared_obligations, 0);
    }

    #[test]
    fn mate_goal_consumes_rules_proof_for_both_colors_and_stalemate_stays_draw() {
        for fen in [
            "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
            "8/8/8/8/8/6k1/5q2/7K b - - 0 1",
        ] {
            let root = Position::from_fen(fen).unwrap();
            let mode = ExecutionMode::CpuPc;
            let mut owner =
                RoventEngine::pc(config(mode), LegalOrderRoleMock, registration(mode)).unwrap();
            let result = owner
                .search(&root, None, limits(None, Some(1)), &AtomicBool::new(false))
                .unwrap();
            assert_eq!(result.completion, RoventCompletion::MateGoal);
            assert!(result.mate.unwrap().goal_satisfied());
            let mut child = Position::from_snapshot(&root.snapshot(), Default::default()).unwrap();
            child.make_move(result.best_move.unwrap()).unwrap();
            let legal = child.ordered_legal_moves();
            assert!(matches!(
                child.play_status_from_view(&legal).unwrap(),
                PlayStatus::Terminal {
                    reason: TerminalReason::Checkmate,
                    ..
                }
            ));
            assert_eq!(result.work.native_r_calls, 0);
        }
        let mode = ExecutionMode::CpuPc;
        let mut owner =
            RoventEngine::pc(config(mode), LegalOrderRoleMock, registration(mode)).unwrap();
        let stale = owner
            .search(
                &Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1").unwrap(),
                None,
                limits(None, Some(1)),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(stale.completion, RoventCompletion::RulesTerminal);
        assert_eq!(stale.terminal, Some(TerminalReason::Stalemate));
        assert_eq!(stale.best_move, None);
        assert_eq!(stale.work.role_calls, 0);
    }

    #[test]
    fn mode_device_root_and_goal_admission_fail_before_neural_calls() {
        let mode = ExecutionMode::CpuPc;
        assert!(
            RoventEngine::pc(
                config(ExecutionMode::CpuPcr),
                LegalOrderRoleMock,
                registration(ExecutionMode::CpuPcr)
            )
            .is_err()
        );
        let mut wrong = registration(mode);
        wrong.compute = PcCompute::Accelerator;
        assert!(RoventEngine::pc(config(mode), LegalOrderRoleMock, wrong).is_err());
        let mut owner =
            RoventEngine::pc(config(mode), LegalOrderRoleMock, registration(mode)).unwrap();
        assert!(
            owner
                .search(
                    &Position::startpos(),
                    Some(&[mv("e2e5")]),
                    limits(Some(2), None),
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        assert!(
            owner
                .search(
                    &Position::startpos(),
                    Some(&[]),
                    limits(Some(2), None),
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        assert!(
            owner
                .search(
                    &Position::startpos(),
                    None,
                    limits(Some(33), None),
                    &AtomicBool::new(false)
                )
                .is_err()
        );
    }

    struct CompletionUnknownModel {
        closures: Arc<Mutex<Vec<RoleSearchClosure>>>,
    }
    impl RoleModel for CompletionUnknownModel {
        fn identity(&self) -> &str {
            "unknown-physical-owner"
        }
        fn propose(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            Err(RoleError::PhysicalCompletionUnknown)
        }
        fn reply(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            panic!("no call after unknown completion")
        }
        fn repair(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            panic!("no call after unknown completion")
        }
        fn divergences(&mut self, _: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            panic!("no V or divergence call")
        }
        fn finish_search(&mut self, closure: RoleSearchClosure) {
            self.closures.lock().unwrap().push(closure);
        }
    }

    #[test]
    fn physical_unknown_quarantines_owner_instead_of_reporting_normal_shutdown() {
        let mode = ExecutionMode::CpuPc;
        let closures = Arc::new(Mutex::new(Vec::new()));
        let mut owner = RoventEngine::pc(
            config(mode),
            CompletionUnknownModel {
                closures: Arc::clone(&closures),
            },
            PcRegistration {
                compute: PcCompute::Cpu,
                model_identity: "unknown-physical-owner".into(),
            },
        )
        .unwrap();
        assert!(matches!(
            owner.search(
                &Position::startpos(),
                None,
                limits(Some(1), None),
                &AtomicBool::new(false)
            ),
            Err(RoventError::Role(RoleError::PhysicalCompletionUnknown))
        ));
        assert_eq!(
            *closures.lock().unwrap(),
            vec![RoleSearchClosure::PhysicalCompletionUnknown]
        );
        assert!(
            owner
                .new_game(
                    Instant::now() + Duration::from_secs(1),
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        assert!(
            !owner
                .shutdown(Instant::now() + Duration::from_secs(1))
                .unwrap()
                .model_completion_known
        );
    }

    #[test]
    fn zero_and_one_node_budgets_keep_fallback_without_a_neural_or_r_call() {
        let mode = ExecutionMode::CpuPc;
        for (quota, root_entries) in [(0, 0), (1, 1)] {
            let mut owner =
                RoventEngine::pc(config(mode), LegalOrderRoleMock, registration(mode)).unwrap();
            let mut bounded = limits(Some(2), None);
            bounded.maximum_node_visits = Some(quota);
            let result = owner
                .search(
                    &Position::startpos(),
                    Some(&[mv("g1f3")]),
                    bounded,
                    &AtomicBool::new(false),
                )
                .unwrap();
            assert_eq!(result.completion, RoventCompletion::WorkBudget);
            assert_eq!(result.best_move, Some(mv("g1f3")));
            assert_eq!(result.work.root_entries, root_entries);
            assert_eq!(result.work.total_node_visits(), Some(root_entries));
            assert_eq!(result.work.role_calls, 0);
            assert_eq!(result.work.checked_transitions, 0);
            assert_eq!(result.work.native_nodes, 0);
            assert!(!result.horizon.target_satisfied);
        }
    }

    #[test]
    fn node_exhaustion_precedes_boundary_acceptance_and_neural_calls_are_not_nodes() {
        let mode = ExecutionMode::CpuPc;
        let mut owner =
            RoventEngine::pc(config(mode), LegalOrderRoleMock, registration(mode)).unwrap();
        let mut bounded = limits(Some(1), None);
        bounded.maximum_node_visits = Some(2);
        let stopped = owner
            .search(
                &Position::startpos(),
                Some(&[mv("e2e4")]),
                bounded,
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(stopped.completion, RoventCompletion::WorkBudget);
        assert_eq!(stopped.work.total_node_visits(), Some(2));
        assert_eq!(stopped.work.checked_transitions, 1);
        assert_eq!(stopped.work.accepted_fresh_values, 0);
        assert!(!stopped.horizon.target_satisfied);
        bounded.maximum_node_visits = Some(3);
        let completed = owner
            .search(
                &Position::startpos(),
                Some(&[mv("e2e4")]),
                bounded,
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(completed.completion, RoventCompletion::DepthGoal);
        assert_eq!(completed.work.total_node_visits(), Some(2));
        assert_eq!(completed.work.role_calls, 2);
        assert_eq!(completed.work.accepted_fresh_values, 1);
        assert!(completed.horizon.target_satisfied);
    }

    #[test]
    fn native_r_receives_only_remaining_common_quota_and_all_actual_work_is_charged() {
        let mode = ExecutionMode::CpuPcr;
        let native = CpuEngine::new(CpuConfig {
            max_depth: 2,
            tt_entries: 32,
            quiescence_ply: 2,
            ..Default::default()
        })
        .unwrap();
        let mut owner =
            RoventEngine::pcr(config(mode), LegalOrderRoleMock, registration(mode), native)
                .unwrap();
        let mut bounded = limits(Some(2), None);
        bounded.maximum_node_visits = Some(3);
        let result = owner
            .search(
                &Position::startpos(),
                Some(&[mv("e2e4")]),
                bounded,
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(result.completion, RoventCompletion::WorkBudget);
        assert_eq!(result.work.root_entries, 1);
        assert_eq!(result.work.checked_transitions, 1);
        assert_eq!(result.work.native_r_calls, 1);
        assert_eq!(result.work.native_nodes, 1);
        assert_eq!(result.work.total_node_visits(), Some(3));
        assert_eq!(result.work.accepted_fresh_values, 0);
        assert!(!result.horizon.target_satisfied);
    }

    #[test]
    fn repeated_mate_transition_is_new_work_but_existing_proof_lookup_is_not() {
        let mode = ExecutionMode::CpuPc;
        let root = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
        let restricted = [mv("f7g7")];
        let mut owner =
            RoventEngine::pc(config(mode), LegalOrderRoleMock, registration(mode)).unwrap();
        let mut bounded = limits(None, Some(1));
        bounded.maximum_node_visits = Some(3);
        let stopped = owner
            .search(&root, Some(&restricted), bounded, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(stopped.completion, RoventCompletion::WorkBudget);
        assert_eq!(stopped.work.root_entries, 1);
        // The normal candidate and the independent proof each execute the edge.
        assert_eq!(stopped.work.checked_transitions, 2);
        assert_eq!(stopped.work.total_node_visits(), Some(3));
        assert!(stopped.mate.is_none());
        bounded.maximum_node_visits = Some(4);
        let completed = owner
            .search(&root, Some(&restricted), bounded, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(completed.completion, RoventCompletion::MateGoal);
        assert_eq!(completed.work.total_node_visits(), Some(3));
        assert!(completed.mate.unwrap().goal_satisfied());
        let mut proof = MateProofStore::new(
            &root,
            root.side_to_move(),
            ChessPly(1),
            Some(&restricted),
            16,
        )
        .unwrap();
        let proof_root = proof.root();
        proof.open_child(proof_root, restricted[0]).unwrap();
        let cancel = AtomicBool::new(false);
        let mut state = SearchState {
            limits: limits(None, Some(1)),
            cancel: &cancel,
            work: RoventWork {
                root_entries: 1,
                ..Default::default()
            },
            stores: new_stores(16),
            positions: 0,
            maximum_positions: 16,
            maximum_examined_ply: ChessPly(0),
        };
        owner
            .prove_mate(
                &mut proof,
                proof_root,
                &mut Vec::new(),
                root.side_to_move(),
                &mut state,
            )
            .unwrap();
        assert_eq!(state.work.total_node_visits(), Some(1));
        assert_eq!(state.work.role_calls, 0);
        assert_eq!(state.work.checked_transitions, 0);
        assert_eq!(
            RoventWork {
                root_entries: 1,
                native_nodes: u64::MAX,
                ..Default::default()
            }
            .total_node_visits(),
            None
        );
    }

    #[test]
    fn combined_depth_caps_mate_work_without_changing_its_original_horizon() {
        let mode = ExecutionMode::CpuPc;
        let mut root = Position::from_fen("6k1/p7/5K2/8/8/8/P7/5Q2 w - - 0 1").unwrap();
        for movement in ["a2a3", "a7a6", "f1g1"] {
            root.make_uci(movement).unwrap();
        }
        let restricted = [mv("g8h8")];
        for (depth, expected, examined) in [
            (Some(1), RoventCompletion::GoalUnresolved, ChessPly(1)),
            (None, RoventCompletion::MateGoal, ChessPly(2)),
        ] {
            let mut selected = config(mode);
            selected.candidate_width = 1;
            selected.normal_line_plies = NonZeroU16::new(1).unwrap();
            let mut owner =
                RoventEngine::pc(selected, LegalOrderRoleMock, registration(mode)).unwrap();
            let result = owner
                .search(
                    &root,
                    Some(&restricted),
                    limits(depth, Some(1)),
                    &AtomicBool::new(false),
                )
                .unwrap();
            assert_eq!(result.completion, expected);
            assert_eq!(result.horizon.maximum_examined_ply, examined);
            let proof = result.mate.unwrap();
            if depth.is_some() {
                assert_eq!(proof.conclusion, MateConclusion::Unknown);
                assert!(!proof.goal_satisfied());
                assert!(result.horizon.target_satisfied);
            } else {
                assert!(proof.goal_satisfied());
                assert_eq!(
                    proof.scope,
                    super::super::rovent_solved::MateScope::RestrictedRoot
                );
            }
        }
    }

    #[test]
    fn admitted_question_replays_exact_prefix_and_keeps_original_move_domain() {
        use super::super::review_context::{
            ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy,
        };
        use rz_contracts::pals::SearchAuthority;
        use rz_contracts::pals_adaptive::ReviewViewSeal;
        use rz_contracts::rovent::{QuestionPurpose, ReciprocalQuestionId, SearchRecipient};
        use rz_contracts::{Deadline, GameGeneration, OwnerId, ProcessEpoch, RootGeneration};
        use rz_position::contracts::ContractPosition;

        let root = Position::startpos();
        let root_rules = Arc::new(
            ContractPosition::new(OwnerId(101), Position::startpos())
                .export()
                .unwrap(),
        );
        let mut target = Position::startpos();
        target.make_uci("e2e4").unwrap();
        let rules = Arc::new(
            ContractPosition::new(OwnerId(102), target)
                .export()
                .unwrap(),
        );
        let root_moves = Arc::new(RootMoveDomain::new(&root_rules, Some(&[mv("e2e4")])).unwrap());
        let prefix = [mv("e2e4")];
        let line = [mv("e2e4"), mv("e7e5")];
        let mut stores = new_stores(16);
        let root_situation = stores.hot_focus_actual_moves(root.snapshot()).unwrap();
        let root_state = stores.situations.get(root_situation).unwrap().state;
        let root_line = stores.lines.root(root_state).unwrap();
        let focus = stores.hot_append_line(root_line, &line).unwrap();
        let state = stores
            .states
            .insert(rules.rules().snapshot().clone())
            .unwrap();
        let situation = stores.situations.insert(state, focus).unwrap();
        let mut index = ReviewContextIndex::default();
        let view = Arc::new(
            index
                .select(
                    &stores,
                    root_state,
                    ReviewQuerySequences {
                        purpose: RoleQueryPurpose::ReplyPolicy,
                        prefix: &prefix,
                        proposal: &line,
                        counterexample: None,
                        legal: &[],
                        divergences: &[],
                    },
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let context = RoleLogicalContext {
            game_generation: 1,
            search_generation: stores.generation(),
            situation,
            state,
            focus,
            purpose: RoleQueryPurpose::ReplyPolicy,
            prefix: prefix.to_vec(),
            focus_sha256: rovent_line_digest(&line).unwrap().0,
            prefix_sha256: rovent_prefix_digest(&prefix).unwrap().0,
            proposal_sha256: rovent_line_digest(&line).unwrap().0,
            refutation_sha256: None,
            divergence_sha256: line_digest(&[]),
            public_revision: 0,
            situation_revision: 0,
        };
        let authority = SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(1),
            root: RootGeneration(stores.generation()),
            implementation: Digest([11; 32]),
        };
        let envelope = QuestionEnvelope {
            id: ReciprocalQuestionId {
                authority,
                sequence: 1,
            },
            parent: None,
            sender: SearchRecipient::Proposer,
            recipient: SearchRecipient::Critic,
            purpose: QuestionPurpose::Counter,
            anchor: rules.snapshot().identity(),
            anchor_root_ply: ChessPly(1),
            original_first_move: Some(rz_contracts::pals::Move16::encode(
                rz_contracts::Move::try_from(prefix[0]).unwrap(),
            )),
            prefix_digest: rovent_prefix_digest(&prefix).unwrap(),
            line_digest: Some(rovent_line_digest(&line).unwrap()),
            line_offset: ChessPly(1),
            question_digest: Digest([12; 32]),
            view: ReviewViewSeal {
                authority,
                view_sequence: view.sequence(),
                context_policy: Digest(view.policy_digest()),
                view_digest: Digest(view.seal_digest()),
                root_state: root_rules.snapshot().identity(),
                actual_tokens: view.token_count(),
                selected_records: view.records().len() as u16,
                anchors: view.anchors().len() as u8,
                length_bucket: view.length_bucket(),
            },
            evidence_revision: 0,
            root_budget: Digest([13; 32]),
            original_deadline: Deadline {
                clock: ClockDomain(ProcessEpoch(1)),
                at: MonotonicTick(100),
            },
        };
        let admit = |envelope, root_moves, rules| {
            AdmittedRoleQuestion::validate(
                envelope,
                root_moves,
                Arc::clone(&root_rules),
                rules,
                Arc::clone(&view),
                context.clone(),
                &stores,
                ExecutionMode::CpuPc,
                ClockDomain(ProcessEpoch(1)),
                MonotonicTick(1),
            )
        };
        let accepted = admit(envelope, Arc::clone(&root_moves), Arc::clone(&rules)).unwrap();
        assert!(Arc::ptr_eq(accepted.root_moves(), &root_moves));
        for changed in [
            QuestionEnvelope {
                anchor_root_ply: ChessPly(200),
                ..envelope
            },
            QuestionEnvelope {
                prefix_digest: Digest([99; 32]),
                ..envelope
            },
            QuestionEnvelope {
                line_offset: ChessPly(2),
                ..envelope
            },
            QuestionEnvelope {
                line_digest: Some(rovent_prefix_digest(&line).unwrap()),
                ..envelope
            },
            QuestionEnvelope {
                original_first_move: Some(rz_contracts::pals::Move16::encode(
                    rz_contracts::Move::try_from(mv("d2d4")).unwrap(),
                )),
                ..envelope
            },
        ] {
            assert!(admit(changed, Arc::clone(&root_moves), Arc::clone(&rules)).is_err());
        }
        let wrong_root = ContractPosition::new(OwnerId(103), Position::startpos())
            .export()
            .unwrap();
        let wrong_domain = Arc::new(RootMoveDomain::new(&wrong_root, None).unwrap());
        assert!(admit(envelope, wrong_domain, Arc::clone(&rules)).is_err());
        let incomplete_history = Position::from_fen(
            &Position::from_snapshot(rules.rules().snapshot(), Default::default())
                .unwrap()
                .to_fen(),
        )
        .unwrap();
        let wrong_rules = Arc::new(
            ContractPosition::new(OwnerId(104), incomplete_history)
                .export()
                .unwrap(),
        );
        let changed = QuestionEnvelope {
            anchor: wrong_rules.snapshot().identity(),
            ..envelope
        };
        assert!(admit(changed, root_moves, wrong_rules).is_err());
    }
}
