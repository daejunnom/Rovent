//! Common logical ownership, independent of an R searcher or worker.
//!
//! Legacy engines retain this semantic state beside their actual registered
//! researcher. A future NoResearcher port can own the same semantic type
//! without constructing an R owner; the separate port/driver must still prove
//! its supported execution path. No model, Store, or researcher is cloned here.

use super::*;

/// P/C graph, public-memory and role limits. There is no native-R task budget.
/// The caller retains its authoritative configuration; state does not keep a
/// second mutable copy of these fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PalsSemanticConfig {
    pub beam_width: usize,
    pub line_plies: usize,
    pub max_nodes: usize,
    pub max_records: usize,
    pub max_role_calls: u64,
}

impl Default for PalsSemanticConfig {
    fn default() -> Self {
        Self::from(&PalsConfig::default())
    }
}

impl From<&PalsConfig> for PalsSemanticConfig {
    fn from(config: &PalsConfig) -> Self {
        Self {
            beam_width: config.beam_width,
            line_plies: config.line_plies,
            max_nodes: config.max_nodes,
            max_records: config.max_records,
            max_role_calls: config.max_role_calls,
        }
    }
}

impl PalsSemanticConfig {
    pub fn validate(&self) -> Result<(), PalsError> {
        if self.beam_width == 0
            || self.beam_width > 16
            || self.line_plies == 0
            || self.line_plies > 16
            || self.max_nodes < 257
            || self.max_nodes > 65_536
            || self.max_records < 4
            || self.max_records > 128
            || self.max_role_calls == 0
            || self.max_role_calls > 1_000_000
        {
            return Err(PalsError::InvalidConfig);
        }
        Ok(())
    }

    pub(super) fn store_limits(&self) -> StoreLimits {
        StoreLimits {
            states: self.max_nodes,
            situations: self.max_nodes,
            line_chunks: self.max_nodes * 16,
            line_plies: self.line_plies,
            observations: self.max_nodes * 16,
            dependency_edges: self.max_nodes * 16,
            executions: self.max_nodes * 8,
            consumers: self.max_nodes * 16,
            ..StoreLimits::default()
        }
    }
}

/// Separate Rovent admission: bounded generation chunks keep the legacy V5
/// maximum, while an exact original-root query prefix may span 256 plies.
/// This profile never changes legacy PalsSemanticConfig validation or bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoventSemanticProfile {
    pub semantic: PalsSemanticConfig,
    pub maximum_root_prefix_plies: u16,
}
impl RoventSemanticProfile {
    pub fn validate(&self) -> Result<(), PalsError> {
        self.semantic.validate()?;
        if self.maximum_root_prefix_plies == 0
            || self.maximum_root_prefix_plies > 256
            || usize::from(self.maximum_root_prefix_plies) < self.semantic.line_plies
        {
            return Err(PalsError::InvalidConfig);
        }
        Ok(())
    }
    pub(super) fn sequence_config(self) -> PalsSemanticConfig {
        PalsSemanticConfig {
            line_plies: usize::from(self.maximum_root_prefix_plies),
            ..self.semantic
        }
    }
}

/// Source compatibility name for the original R-free constructor.
pub type PcSemanticProfile = RoventSemanticProfile;

/// Exact graph/public-memory ownership shared by future typed assemblies.
/// A state is not itself a search driver, researcher registration or receipt.
pub struct PalsSemanticState<M: RoleModel> {
    pub(super) model: M,
    pub(super) model_registered_value: Option<ModelValueIdentity>,
    pub(super) nodes: Vec<Node>,
    pub(super) records: Vec<RoleRecord>,
    pub(super) review_index: Option<ReviewContextIndex>,
    pub(super) review_policy: Option<ReviewSelectionPolicy>,
    /// Selected/compact consumers never allocate or scan a separate immutable
    /// legacy full-record input unless that baseline was explicitly selected.
    pub(super) legacy_full_native_context: bool,
    pub(super) last_review_issue: Option<ReviewError>,
    pub(super) revision: u64,
    /// Store handles may restart after reset; this independent namespace must
    /// never let an old-game question acquire a successor game's authority.
    pub(super) game_generation: Option<u64>,
    pub(super) adaptive_root_registry: adaptive_root::AdaptiveRootRegistry,
    /// Process-local owner follows the exact Store through port/archive/engine
    /// moves. Only explicit terminal retirement may reclaim it. Declare before
    /// Store so its pins are released before that Store on ordinary destruction.
    pub(super) rovent_frontier: Option<Box<crate::pals::rovent_frontier::RoventFrontier>>,
    pub(super) retired_goal_local: Option<Box<adaptive_port::goal_local::GoalLocalRetiredCustody>>,
    pub(super) retired_profiled_external:
        Option<Box<adaptive_port::ProfiledExternalRetiredCustody>>,
    pub(super) retired_pc_repair: Option<pc_port::PcRepairReport>,
    pub(super) stores: PalsStores,
    pub(super) clock_origin: Instant,
    pub(super) consumer_id: u64,
    pub(super) last_search_counters: Option<PalsCounters>,
    pub(super) last_recheck_observer_error: Option<RoleError>,
    pub(super) repair_queue: std::collections::VecDeque<frozen_recheck::RepairWork>,
    pub(super) repair_questions: HashSet<(BoardMove, u64, Vec<BoardMove>)>,
    pub(super) repair_counts: std::collections::HashMap<BoardMove, u8>,
    pub(super) archive_state: archive_lifecycle::EngineArchiveLifecycle,
    pub(super) new_game_transition_error: Option<PalsError>,
}

impl<M: RoleModel> PalsSemanticState<M> {
    /// Caller already admitted its model namespace and execution policy. This
    /// initializer allocates only the existing bounded semantic structures.
    pub(super) fn new_admitted(
        config: &PalsSemanticConfig,
        model: M,
        model_registered_value: Option<ModelValueIdentity>,
    ) -> Result<Self, PalsError> {
        config.validate()?;
        Self::allocate_admitted(config, model, model_registered_value)
    }

    pub(super) fn new_rovent_admitted(
        profile: RoventSemanticProfile,
        model: M,
        model_registered_value: Option<ModelValueIdentity>,
    ) -> Result<Self, PalsError> {
        profile.validate()?;
        Self::allocate_admitted(&profile.sequence_config(), model, model_registered_value)
    }

    pub(super) fn new_researcher_admitted(
        profile: &ResearcherSemanticProfile,
        model: M,
        model_registered_value: Option<ModelValueIdentity>,
    ) -> Result<Self, PalsError> {
        profile.validate()?;
        Self::allocate_admitted(&profile.sequence_config(), model, model_registered_value)
    }

    fn allocate_admitted(
        config: &PalsSemanticConfig,
        model: M,
        model_registered_value: Option<ModelValueIdentity>,
    ) -> Result<Self, PalsError> {
        let mut nodes = Vec::new();
        nodes
            .try_reserve(config.max_nodes)
            .map_err(|_| PalsError::Capacity)?;
        let mut records = Vec::new();
        records
            .try_reserve(config.max_records)
            .map_err(|_| PalsError::Capacity)?;
        let stores = PalsStores::new(config.store_limits());
        Ok(Self {
            model,
            model_registered_value,
            nodes,
            records,
            review_index: None,
            review_policy: None,
            legacy_full_native_context: false,
            last_review_issue: None,
            revision: 0,
            game_generation: Some(0),
            adaptive_root_registry: adaptive_root::AdaptiveRootRegistry::default(),
            rovent_frontier: None,
            retired_goal_local: None,
            retired_profiled_external: None,
            retired_pc_repair: None,
            stores,
            clock_origin: Instant::now(),
            consumer_id: 0,
            last_search_counters: None,
            last_recheck_observer_error: None,
            repair_queue: std::collections::VecDeque::new(),
            repair_questions: HashSet::new(),
            repair_counts: std::collections::HashMap::new(),
            archive_state: archive_lifecycle::EngineArchiveLifecycle::default(),
            new_game_transition_error: None,
        })
    }

    pub(super) fn select_review_context(
        &mut self,
        policy: ReviewSelectionPolicy,
        limits: ReviewIndexLimits,
    ) -> Result<(), PalsError> {
        policy.validate()?;
        if !self.model.supports_selected_review_view() {
            return Err(RoleError::Unavailable.into());
        }
        if self.stores.generation() != 0
            || !self.nodes.is_empty()
            || !self.records.is_empty()
            || self.review_policy.is_some()
        {
            return Err(StoreError::InvalidConditions(
                "review context is selected once before search",
            )
            .into());
        }
        self.review_index = Some(ReviewContextIndex::new(limits)?);
        self.review_policy = Some(policy);
        Ok(())
    }

    /// Exact hot-only Rules graph admission, shared by R-registered and R-free
    /// ports. It never searches or loads/archives a cold record on the manager.
    pub(super) fn intern_hot(
        &mut self,
        config: &PalsSemanticConfig,
        position: Position,
    ) -> Result<usize, PalsError> {
        let snapshot = position.snapshot();
        if let Some(state) = self.stores.states.find(&snapshot)
            && let Some(index) = self.nodes.iter().position(|node| node.state == state)
        {
            return Ok(index);
        }
        if self.nodes.len() >= config.max_nodes {
            return Err(PalsError::Capacity);
        }
        let situation = if let Some(root) = self.stores.root().filter(|root| {
            self.stores.situations.get(*root).is_ok_and(|situation| {
                self.stores
                    .states
                    .get(situation.state)
                    .is_ok_and(|state| state.same_state(&snapshot))
            })
        }) {
            root
        } else {
            self.stores.hot_insert_situation(snapshot)?
        };
        let state = self.stores.situations.get(situation)?.state;
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
            self.stores.hot_append_rules_terminal(&position, 0, 0)?;
        }
        self.nodes.push(Node {
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
        Ok(self.nodes.len() - 1)
    }

    /// Returns whether a new edge was examined, so caller-owned work counters
    /// charge exactly once. Reusing an edge does not create native-R work.
    pub(super) fn connect_hot(
        &mut self,
        config: &PalsSemanticConfig,
        parent: usize,
        movement: BoardMove,
    ) -> Result<(usize, bool), PalsError> {
        let node = self
            .nodes
            .get(parent)
            .ok_or(StoreError::InvalidHandle("semantic graph parent"))?;
        if let Some(edge) = node.edges.iter().find(|edge| edge.movement == movement) {
            return Ok((edge.child, false));
        }
        let mut position = node.position.clone();
        position.make_move(movement)?;
        let child = self.intern_hot(config, position)?;
        self.nodes[parent].edges.push(Edge { movement, child });
        Ok((child, true))
    }

    /// Adopt only Rules snapshots already checked and charged by this exact
    /// controller. No second make_move or reconstructed quota is performed.
    pub(super) fn adopt_checked_frontier_line(
        &mut self,
        config: &PalsSemanticConfig,
        root: usize,
        line: &crate::pals::rovent_frontier::CheckedFrontierLine,
    ) -> Result<(usize, u64), PalsError> {
        let frontier = self
            .rovent_frontier
            .as_deref()
            .ok_or(StoreError::InvalidConditions(
                "checked Rules line lost its original frontier",
            ))?;
        if !line.matches(frontier, &self.stores)
            || line.prefix().len() != line.steps().len()
            || line.prefix().len() > config.line_plies
            || !self
                .nodes
                .get(root)
                .is_some_and(|node| node.position.snapshot().same_state(line.root_snapshot()))
        {
            return Err(StoreError::InvalidEvidence("foreign checked frontier line").into());
        }
        let mut parent = root;
        let mut new_edges = 0u64;
        for (movement, step) in line.prefix().iter().zip(line.steps()) {
            if *movement != step.movement() {
                return Err(StoreError::InvalidEvidence("checked Rules step order").into());
            }
            if let Some(edge) = self.nodes[parent]
                .edges
                .iter()
                .find(|edge| edge.movement == *movement)
            {
                if !self.nodes[edge.child]
                    .position
                    .snapshot()
                    .same_state(step.snapshot())
                {
                    return Err(StoreError::InvalidEvidence(
                        "semantic edge differs from checked Rules snapshot",
                    )
                    .into());
                }
                parent = edge.child;
                continue;
            }
            self.nodes[parent]
                .edges
                .try_reserve(1)
                .map_err(|_| PalsError::Capacity)?;
            let child = self.intern_hot(
                config,
                Position::from_snapshot(step.snapshot(), Default::default())?,
            )?;
            self.nodes[parent].edges.push(Edge {
                movement: *movement,
                child,
            });
            parent = child;
            new_edges = new_edges.checked_add(1).ok_or(PalsError::Capacity)?;
        }
        Ok((parent, new_edges))
    }

    pub(super) fn ensure_frontier_retired(&self) -> Result<(), PalsError> {
        if self.rovent_frontier.is_some()
            || self.retired_goal_local.is_some()
            || self.retired_profiled_external.is_some()
            || self.retired_pc_repair.is_some()
        {
            return Err(StoreError::InvalidConditions(
                "Rovent frontier owner must be explicitly retired before root or game reset",
            )
            .into());
        }
        Ok(())
    }
}

impl<M: RoleModel> PalsSemanticState<M> {
    pub(super) fn role_context(
        &self,
        config: &PalsSemanticConfig,
        node: usize,
        question: RoleQuestion<'_>,
    ) -> Result<RoleLogicalContext, PalsError> {
        let game_generation = self.game_generation.ok_or(PalsError::Capacity)?;
        let node = self
            .nodes
            .get(node)
            .ok_or(StoreError::InvalidHandle("role node"))?;
        let situation = self.stores.situations.get(node.situation)?;
        if situation.state != node.state
            || !self
                .stores
                .states
                .get(node.state)?
                .same_state(&node.position.snapshot())
            || self.stores.lines.get(situation.focus)?.start_state != node.state
        {
            return Err(StoreError::InvalidHandle("role checked situation/state/focus").into());
        }
        if question.prefix.len() > config.line_plies
            || question.proposal.len() > config.line_plies
            || question
                .refutation
                .is_some_and(|line| line.len() > config.line_plies)
            || question.divergences.len() > config.line_plies
        {
            return Err(PalsError::Capacity);
        }
        let focus = self.stores.lines.moves(situation.focus)?;
        let mut prefix = Vec::new();
        prefix
            .try_reserve_exact(question.prefix.len())
            .map_err(|_| PalsError::Capacity)?;
        prefix.extend_from_slice(question.prefix);
        Ok(RoleLogicalContext {
            game_generation,
            search_generation: self.stores.generation(),
            situation: node.situation,
            state: node.state,
            focus: situation.focus,
            purpose: question.purpose,
            prefix,
            focus_sha256: role_line_sha256(b"focus", &focus)?,
            prefix_sha256: role_line_sha256(b"prefix", question.prefix)?,
            proposal_sha256: role_line_sha256(b"proposal", question.proposal)?,
            refutation_sha256: question
                .refutation
                .map(|line| role_line_sha256(b"refutation", line))
                .transpose()?,
            divergence_sha256: role_divergence_sha256(question.divergences)?,
            public_revision: self.revision,
            situation_revision: situation.revision,
        })
    }

    pub(super) fn publish_record(
        &mut self,
        config: &PalsSemanticConfig,
        record: RoleRecord,
        obsolete: Option<usize>,
        evidence: Option<ObservationId>,
    ) -> Result<(), PalsError> {
        self.publish_review_record(config, &record, evidence)?;
        for previous in &mut self.records {
            if previous.origin_state == record.origin_state
                && previous.kind == record.kind
                && previous.line.first() == record.line.first()
            {
                previous.critical = false;
            }
        }
        if let Some(index) = obsolete {
            self.records.remove(index);
        }
        self.revision = record.revision;
        self.records.push(record);
        Ok(())
    }

    pub(super) fn refresh_projection(&mut self, root_state: StateId) {
        if let Some(index) = &mut self.review_index {
            index.refresh_root(root_state);
        }
        let mut newest = HashSet::new();
        for record in self.records.iter_mut().rev() {
            record.critical = record.origin_state == root_state
                && newest.insert((record.kind, record.line.first().copied()));
        }
    }

    pub(super) fn rules_value(&self, node: usize) -> PalsResolvedValue {
        let perspective = self.nodes[node].position.side_to_move();
        match self.nodes[node].terminal {
            Some((_, score)) => PalsResolvedValue::RulesTerminal {
                winner: if score > 0 {
                    Some(perspective)
                } else if score < 0 {
                    Some(perspective.opposite())
                } else {
                    None
                },
                perspective,
            },
            None => PalsResolvedValue::Unknown,
        }
    }

    pub(super) fn model_value(&self, index: usize, remaining: usize) -> PalsResolvedValue {
        let node = &self.nodes[index];
        if node.terminal.is_some() {
            return self.rules_value(index);
        }
        if remaining > 0 {
            let mut best = PalsResolvedValue::Unknown;
            // Equal W-L distributions/conditional outcomes preserve Rules order.
            for movement in node.position.legal_moves() {
                if let Some(edge) = node.edges.iter().find(|edge| edge.movement == movement) {
                    let value = flip_resolved_value(self.model_value(edge.child, remaining - 1));
                    if better_model_choice(value, best, false, false) {
                        best = value;
                    }
                }
            }
            if !matches!(best, PalsResolvedValue::Unknown) {
                return best;
            }
        }
        node.model_value
            .as_ref()
            .filter(|output| {
                Some(&output.identity) == self.model_registered_value.as_ref()
                    && output.state == node.position.position_identity()
                    && output.perspective == node.position.side_to_move()
            })
            .map_or(PalsResolvedValue::Unknown, |output| {
                PalsResolvedValue::ModelWdl {
                    wdl: output.wdl,
                    perspective: output.perspective,
                }
            })
    }

    pub(super) fn root_model_value(
        &self,
        config: &PalsSemanticConfig,
        child: usize,
    ) -> (PalsResolvedValue, PalsValueScope) {
        let terminal = self.nodes[child].terminal.is_some();
        let value = self.model_value(child, config.line_plies.saturating_sub(1));
        let mut flipped = flip_resolved_value(value);
        if terminal {
            // This is the directly selected child's current-state fact. Deeper
            // terminal lines remain RestrictedRulesLine with estimate scope.
            if let PalsResolvedValue::RestrictedRulesLine {
                winner,
                perspective,
            } = flipped
            {
                flipped = PalsResolvedValue::RulesTerminal {
                    winner,
                    perspective,
                };
            }
        }
        let scope = if terminal {
            PalsValueScope::RulesTerminal
        } else if matches!(flipped, PalsResolvedValue::Unknown) {
            PalsValueScope::Unknown
        } else {
            PalsValueScope::RestrictedEstimate
        };
        (flipped, scope)
    }
}

/// P/C-only semantic owner. This type cannot contain a checker, native worker,
/// TT or R credit ledger. Its port drives selected role tickets and uses the
/// same exact Rules/public-memory implementation as the registered-R assembly.
pub struct PcSemanticOwner<M: RoleModel> {
    pub(super) config: PalsSemanticConfig,
    pub(super) profile: PcSemanticProfile,
    pub(super) semantic: PalsSemanticState<M>,
    researcher: NoResearcher,
}

impl<M: RoleModel> PcSemanticOwner<M> {
    pub fn new_selected(
        config: PalsSemanticConfig,
        model: M,
        policy: ReviewSelectionPolicy,
        limits: ReviewIndexLimits,
    ) -> Result<Self, PalsError> {
        let maximum_root_prefix_plies =
            u16::try_from(config.line_plies).map_err(|_| PalsError::InvalidConfig)?;
        Self::new_selected_rovent(
            PcSemanticProfile {
                semantic: config,
                maximum_root_prefix_plies,
            },
            model,
            policy,
            limits,
        )
    }

    pub fn new_selected_rovent(
        profile: PcSemanticProfile,
        model: M,
        policy: ReviewSelectionPolicy,
        limits: ReviewIndexLimits,
    ) -> Result<Self, PalsError> {
        profile.validate()?;
        let config = profile.semantic;
        let identity = model
            .value_identity()
            .ok_or(RoleError::Unavailable)?
            .clone();
        identity.validate()?;
        if identity.model != model.identity() || identity.semantics != MODEL_WDL_VALUE_SEMANTICS {
            return Err(RoleError::InvalidOutput.into());
        }
        if model.requires_registered_representation_namespace()
            && model.registered_representation_namespace().is_none()
        {
            return Err(RoleError::Unsupported(
                "native representation namespace is not registered",
            )
            .into());
        }
        if let Some(namespace) = model.registered_representation_namespace() {
            namespace.verify_identity(model.identity(), Some(&identity))?;
        }
        let mut semantic = PalsSemanticState::new_rovent_admitted(profile, model, Some(identity))?;
        semantic.select_review_context(policy, limits)?;
        Ok(Self {
            config,
            profile,
            semantic,
            researcher: NoResearcher,
        })
    }

    pub fn config(&self) -> PalsSemanticConfig {
        self.config
    }
    pub fn profile(&self) -> PcSemanticProfile {
        self.profile
    }
    pub fn researcher(&self) -> &NoResearcher {
        &self.researcher
    }
    pub fn stores(&self) -> &PalsStores {
        &self.semantic.stores
    }
    pub fn records(&self) -> &[RoleRecord] {
        &self.semantic.records
    }
    pub fn model_identity(&self) -> &str {
        self.semantic.model.identity()
    }
    pub fn model_value_identity(&self) -> Option<&ModelValueIdentity> {
        self.semantic.model_registered_value.as_ref()
    }
    pub fn review_policy(&self) -> Option<ReviewSelectionPolicy> {
        self.semantic.review_policy
    }
    pub fn last_review_issue(&self) -> Option<&ReviewError> {
        self.semantic.last_review_issue.as_ref()
    }

    /// Explicit capability error for assembly validation. No success-valued
    /// zero-node answer or fabricated checker identity is provided.
    pub fn require_researcher(&self) -> Result<(), PalsError> {
        Err(RoleError::Unsupported("R is structurally unavailable in the P-C owner").into())
    }

    pub(super) fn role_context(
        &self,
        node: usize,
        question: RoleQuestion<'_>,
    ) -> Result<RoleLogicalContext, PalsError> {
        self.semantic
            .role_context(&self.profile.sequence_config(), node, question)
    }
    pub(super) fn prepare_review_view(
        &mut self,
        node: usize,
        question: RoleQuestion<'_>,
        legal: &[BoardMove],
    ) -> Result<Option<Arc<ReviewView>>, PalsError> {
        self.semantic
            .prepare_review_view(&self.profile.sequence_config(), node, question, legal)
    }
    pub(super) fn publish_record(
        &mut self,
        record: RoleRecord,
        obsolete: Option<usize>,
        evidence: Option<ObservationId>,
    ) -> Result<(), PalsError> {
        self.semantic
            .publish_record(&self.config, record, obsolete, evidence)
    }
    pub(super) fn refresh_projection(&mut self, state: StateId) {
        self.semantic.refresh_projection(state);
    }
    pub(super) fn root_model_value(&self, child: usize) -> (PalsResolvedValue, PalsValueScope) {
        self.semantic
            .root_model_value(&self.profile.sequence_config(), child)
    }
    pub(super) fn intern_hot(&mut self, position: Position) -> Result<usize, PalsError> {
        self.semantic.intern_hot(&self.config, position)
    }
    pub(super) fn connect_hot(
        &mut self,
        parent: usize,
        movement: BoardMove,
    ) -> Result<(usize, bool), PalsError> {
        self.semantic.connect_hot(&self.config, parent, movement)
    }
}

/// Actual registered legacy R owner. Construction requires the already
/// admitted checker; it never fabricates an absent checker identity.
pub struct RegisteredResearcher {
    pub(super) cpu: Box<dyn CpuChecker>,
    pub(super) checker_registered_identity: CheckerIdentity,
    pub(super) cpu_registered_value: Option<CpuValueIdentity>,
    pub(super) cpu_registered_condition: String,
    pub(super) checker_new_game_pending: bool,
    pub(super) last_external_attempt: Option<u64>,
    pub(super) external_attempts: Vec<CheckerAttempt>,
}

impl RegisteredResearcher {
    pub(super) fn from_admitted(
        cpu: Box<dyn CpuChecker>,
        checker_registered_identity: CheckerIdentity,
        cpu_registered_value: Option<CpuValueIdentity>,
        cpu_registered_condition: String,
    ) -> Self {
        Self {
            cpu,
            checker_registered_identity,
            cpu_registered_value,
            cpu_registered_condition,
            checker_new_game_pending: false,
            last_external_attempt: None,
            external_attempts: Vec::new(),
        }
    }
}

/// Typed absence, deliberately implementing neither CpuChecker nor CpuSearcher.
/// It owns no checker, TT, continuation, worker, identity or native-R budget.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NoResearcher;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct MoveOnlyModel(Arc<AtomicUsize>);
    impl Drop for MoveOnlyModel {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl RoleModel for MoveOnlyModel {
        fn identity(&self) -> &str {
            "semantic-owner-no-forward"
        }
        fn propose(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            panic!("admitted initialization must not run P");
        }
        fn reply(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            panic!("admitted initialization must not run C");
        }
        fn repair(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            panic!("admitted initialization must not run Repair");
        }
        fn divergences(&mut self, _: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            panic!("admitted initialization must not run divergence");
        }
    }

    fn pc_owner() -> PcSemanticOwner<LegalOrderRoleMock> {
        PcSemanticOwner::new_selected(
            PalsSemanticConfig::default(),
            LegalOrderRoleMock,
            ReviewSelectionPolicy::default(),
            ReviewIndexLimits::default(),
        )
        .unwrap()
    }
    fn authority(root: u64) -> rz_contracts::pals::SearchAuthority {
        rz_contracts::pals::SearchAuthority {
            epoch: rz_contracts::ProcessEpoch(2),
            game: rz_contracts::GameGeneration(0),
            root: rz_contracts::RootGeneration(root),
            implementation: rz_contracts::Digest([3; 32]),
        }
    }

    #[test]
    fn pc_owner_uses_exact_registered_root_view_and_graph_without_researcher() {
        // No CpuEngine/OwnedCpuChecker/CpuWorker or credit ledger is constructed.
        let mut owner = pc_owner();
        assert_eq!(std::mem::size_of_val(owner.researcher()), 0);
        assert!(matches!(
            owner.require_researcher(),
            Err(PalsError::Role(RoleError::Unsupported(_)))
        ));
        let position = Position::startpos();
        let scope = authority(7);
        let registration = owner.register_adaptive_root(&position, scope).unwrap();
        owner
            .consume_adaptive_root_registration(registration, &position, scope)
            .unwrap();
        let root = owner.intern_hot(position).unwrap();
        let movement = BoardMove::from_uci("e2e4").unwrap();
        let (child, created) = owner.connect_hot(root, movement).unwrap();
        assert!(created);
        assert_eq!(owner.connect_hot(root, movement).unwrap(), (child, false));
        assert_eq!(owner.semantic.nodes[root].edges.len(), 1);
        assert!(
            owner
                .semantic
                .nodes
                .iter()
                .all(|node| node.evidence.is_none() && node.resume.is_none())
        );
        let line = [movement];
        let question = RoleQuestion {
            purpose: RoleQueryPurpose::ReplyPolicy,
            prefix: &line,
            proposal: &[],
            refutation: None,
            divergences: &[],
        };
        let logical = owner.role_context(child, question).unwrap();
        let legal = owner.semantic.nodes[child].position.legal_moves();
        let view = owner
            .prepare_review_view(child, question, &legal)
            .unwrap()
            .unwrap();
        assert!(view.validates(owner.stores()));
        assert_eq!(view.query().prefix, line);
        assert_eq!(logical.prefix, line);
        assert_eq!(logical.search_generation, scope.root.0);
        assert_eq!(
            owner
                .semantic
                .stores
                .situations
                .get(logical.situation)
                .unwrap()
                .state,
            logical.state
        );
        assert!(matches!(
            owner.root_model_value(child).0,
            PalsResolvedValue::Unknown
        ));
        drop(view);
        assert_eq!(owner.stores().review_pin_count().unwrap(), 0);
    }

    #[test]
    fn pc_root_registration_rejects_foreign_semantic_owner_without_mutation() {
        let mut first = pc_owner();
        let mut other = pc_owner();
        let position = Position::startpos();
        let scope = authority(11);
        let proof = first.register_adaptive_root(&position, scope).unwrap();
        assert!(
            other
                .consume_adaptive_root_registration(proof, &position, scope)
                .is_err()
        );
        assert_eq!(other.stores().generation(), 0);
        assert!(other.stores().root().is_none());
        assert!(other.semantic.nodes.is_empty());
        first.abandon_unconsumed_adaptive_root(scope).unwrap();
    }

    #[test]
    fn admitted_semantic_owner_moves_nonclone_model_without_researcher_or_forward() {
        let legacy = PalsConfig {
            cpu_nodes_per_task: 0,
            ..PalsConfig::default()
        };
        assert!(legacy.validate().is_err());
        let common = PalsSemanticConfig::from(&legacy);
        common.validate().unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let state =
            PalsSemanticState::new_admitted(&common, MoveOnlyModel(drops.clone()), None).unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        let PalsSemanticState { model, .. } = state;
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(model);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(std::mem::size_of::<NoResearcher>(), 0);
    }
}
