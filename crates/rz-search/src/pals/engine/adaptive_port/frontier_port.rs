//! Root goal ownership beside the actual semantic Store. This bridge performs
//! no model call, native search, archive I/O or physical completion check.
use super::*;
use crate::cpu_worker::CpuLedgerIdentity;
use crate::pals::rovent_engine::RootMoveDomain;
use crate::pals::rovent_frontier::{
    FrontierAttachment, FrontierError, FrontierLimits, FrontierRegistration, FrontierSnapshot,
    RoventFrontier,
};
use rz_contracts::rovent::ExecutionMode;
use rz_contracts::{ClockDomain, MonotonicTick};

/// Independent startup owners, supplied by the concrete manager assembly.
/// A registration's own copies are not used to certify its root domain/ledger.
pub struct AdaptiveFrontierStartup<'a> {
    pub mode: ExecutionMode,
    pub root_budget: ContractDigest,
    pub root_moves: &'a Arc<RootMoveDomain>,
    pub native_ledger: Option<&'a CpuLedgerIdentity>,
}

/// A copied semantic report. It attests neither GPU drain nor worker join.
pub struct AdaptiveFrontierReceipt {
    pub root_budget: ContractDigest,
    pub snapshot: FrontierSnapshot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveReviewLeaseAdmission {
    pub current: usize,
    pub capacity: usize,
    pub required_new_leases: usize,
    /// The permanent frontier lease is already part of current. Only its
    /// allocate-before-release replacement is reserved in addition here.
    pub frontier_replacement_reserve: usize,
    pub frontier_owner_present: bool,
    pub may_prepare: bool,
}

/// Private original submission. PositionSnapshot shares immutable Rules
/// history; no second tensor, policy input or Store pin is allocated here.
#[derive(Clone)]
pub(super) struct FrontierQuestionSource {
    authority: SearchAuthority,
    original_deadline: Deadline,
    deadline: Instant,
    position: PositionSnapshot,
    context: ContractDigest,
    view: ReviewViewSeal,
    legacy: Option<std::sync::Weak<LegacyFullRoleRecords>>,
    goal_origin: Option<ContractDigest>,
}

/// Read-only historical projection for the bounded retirement writer. This
/// cannot reconstruct the original private source or mint a selected view.
pub(crate) struct AdaptiveRetirementFrontierSource<'a> {
    pub authority: SearchAuthority,
    pub original_deadline: Deadline,
    pub deadline: Instant,
    pub position: &'a PositionSnapshot,
    pub context: ContractDigest,
    pub view: ReviewViewSeal,
    /// None: absent; Some(Err): original Weak expired, never a fabricated digest.
    pub legacy_digest_or_expired: Option<Result<ContractDigest, ()>>,
    pub goal_origin: Option<ContractDigest>,
}

impl FrontierQuestionSource {
    // Retain the original authority, deadlines, Rules, view and lineage as
    // distinct sources instead of replacing them with an unchecked context.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        authority: SearchAuthority,
        original_deadline: Deadline,
        deadline: Instant,
        position: PositionSnapshot,
        context: &RoleLogicalContext,
        view: ReviewViewSeal,
        legacy: Option<&Arc<LegacyFullRoleRecords>>,
        goal_origin: Option<ContractDigest>,
    ) -> Result<Self, PalsError> {
        Ok(Self {
            authority,
            original_deadline,
            deadline,
            position,
            context: source_context_digest(context)?,
            view,
            legacy: legacy.map(Arc::downgrade),
            goal_origin,
        })
    }
}

impl AdaptivePortQuestion {
    pub(crate) fn retirement_frontier_source(&self) -> AdaptiveRetirementFrontierSource<'_> {
        let source = &self.frontier_source;
        AdaptiveRetirementFrontierSource {
            authority: source.authority,
            original_deadline: source.original_deadline,
            deadline: source.deadline,
            position: &source.position,
            context: source.context,
            view: source.view,
            legacy_digest_or_expired: source.legacy.as_ref().map(|weak| {
                weak.upgrade()
                    .map(|live| ContractDigest(live.digest()))
                    .ok_or(())
            }),
            goal_origin: source.goal_origin,
        }
    }
    /// Check the actual private producer seal. Frame liveness/role acceptance
    /// remains with the machine; root-to-prefix replay stays with Frontier.
    pub fn validate_frontier_source(
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        stores: &PalsStores,
    ) -> Result<(), PalsError> {
        let input = &work.input;
        let query = &input.question;
        if query.goal_local.is_some() {
            return Err(RoleError::Unsupported("goal-local input has no ordinary frame").into());
        }
        let source = &query.frontier_source;
        let first = query.first.ok_or(RoleError::InvalidOutput)?;
        let first = rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(first).map_err(|_| RoleError::InvalidOutput)?,
        );
        if work.frame.authority != source.authority
            || work.original_deadline != source.original_deadline
            || work.original_first_move != first
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Self::validate_frontier_input(input, stores)
    }

    pub(super) fn validate_frontier_input(
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
        stores: &PalsStores,
    ) -> Result<(), PalsError> {
        let query = &input.question;
        let source = &query.frontier_source;
        let view = &input.view;
        let actual_seal = ReviewViewSeal {
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
        if query.deadline != source.deadline
            || input.stage != query.stage
            || input.question_digest != query.question_digest
            || actual_seal != source.view
            || input.view_seal != source.view
            || input.public_context.0 != view.record_scope_digest()
            || source.context != source_context_digest(&query.context)?
            || !source.position.same_state(&query.position.snapshot())
            || !view.validates(stores)
            || query.context.game_generation != source.authority.game.0
            || query.context.search_generation != source.authority.root.0
            || stores.generation() != source.authority.root.0
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let legacy_matches = match (&source.legacy, &query.legacy_full_records) {
            (None, None) => true,
            (Some(original), Some(records)) => original.upgrade().is_some_and(|original| {
                Arc::ptr_eq(&original, records) && records.validates(stores)
            }),
            _ => false,
        };
        if !legacy_matches {
            return Err(RoleError::InvalidOutput.into());
        }
        if query.goal_local.as_ref().map(|origin| origin.digest()) != source.goal_origin {
            return Err(RoleError::InvalidOutput.into());
        }
        if let Some(origin) = &query.goal_local {
            origin.validates_frame_input(input)?;
        }
        let visible = view.query();
        let proposal = if query.goal_local.is_some() {
            &[][..]
        } else if query.context.purpose == RoleQueryPurpose::ValueFresh {
            query.context.prefix.as_slice()
        } else {
            query.proposal.as_slice()
        };
        let counter = if query.context.purpose == RoleQueryPurpose::ValueFresh
            || matches!(
                query.stage,
                RefinementStage::CriticReply | RefinementStage::CounterContinuation
            ) {
            None
        } else {
            query.counter.as_deref()
        };
        let legal = if matches!(
            query.context.purpose,
            RoleQueryPurpose::DivergencePolicy | RoleQueryPurpose::TaskPolicy
        ) {
            Vec::new()
        } else {
            query.position.legal_moves()
        };
        if visible.purpose != query.context.purpose
            || visible.prefix != query.context.prefix
            || visible.proposal != proposal
            || visible.counterexample.as_deref() != counter
            || visible.legal != legal
            || query.context.prefix_sha256 != role_line_sha256(b"prefix", &visible.prefix)?
            || query.context.proposal_sha256 != role_line_sha256(b"proposal", &visible.proposal)?
            || query.context.refutation_sha256
                != visible
                    .counterexample
                    .as_deref()
                    .map(|line| role_line_sha256(b"refutation", line))
                    .transpose()?
            || query.context.divergence_sha256 != role_divergence_sha256(&visible.divergences)?
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let situation = stores.situations.get(query.context.situation)?;
        if situation.state != query.context.state
            || situation.focus != query.context.focus
            || situation.revision < query.context.situation_revision
            || !stores
                .states
                .get(query.context.state)?
                .same_state(&query.position.snapshot())
            || stores.lines.get(query.context.focus)?.start_state != query.context.state
            || query.context.focus_sha256
                != role_line_sha256(b"focus", &stores.lines.moves(query.context.focus)?)?
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }
}

fn source_context_digest(context: &RoleLogicalContext) -> Result<ContractDigest, PalsError> {
    let mut hash = Sha256::new();
    hash.update(b"rz-rovent-private-role-source-context/1\0");
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
    if let Some(refutation) = context.refutation_sha256 {
        hash.update(refutation);
    }
    hash.update(context.divergence_sha256);
    Ok(ContractDigest(hash.finalize().into()))
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Preflight for one genuinely new selected view. Cached input/alias
    /// return paths retain their original lease and need no new admission.
    pub fn review_lease_admission(&self) -> Result<AdaptiveReviewLeaseAdmission, PalsError> {
        self.review_lease_admission_for(true)
    }

    fn review_lease_admission_for(
        &self,
        new_selected_view: bool,
    ) -> Result<AdaptiveReviewLeaseAdmission, PalsError> {
        let stores = &self.engine.semantic.stores;
        let current = stores.review_pin_count()?;
        let capacity = stores.review_pin_capacity();
        let frontier_owner_present = self.engine.semantic.rovent_frontier.is_some();
        // The legacy full-input branch may also acquire an independently
        // pinned raw snapshot. Conservatively reserve it before capture; a
        // reused existing snapshot does not turn this declaration into work.
        let required_new_leases = usize::from(new_selected_view)
            + usize::from(self.engine.semantic.legacy_full_native_context);
        let frontier_replacement_reserve = usize::from(frontier_owner_present);
        let may_prepare = current
            .checked_add(required_new_leases)
            .and_then(|count| count.checked_add(frontier_replacement_reserve))
            .is_some_and(|count| count <= capacity);
        Ok(AdaptiveReviewLeaseAdmission {
            current,
            capacity,
            required_new_leases,
            frontier_replacement_reserve,
            frontier_owner_present,
            may_prepare,
        })
    }

    pub(super) fn ensure_new_review_leases(
        &self,
        new_selected_view: bool,
    ) -> Result<(), PalsError> {
        // Existing V5 admission remains unchanged when no Rovent frontier is
        // registered. pin_review still enforces its exact hard ceiling.
        if self.engine.semantic.rovent_frontier.is_none() {
            return Ok(());
        }
        let observed = self.review_lease_admission_for(new_selected_view)?;
        if !observed.may_prepare {
            return Err(StoreError::ReviewLeaseBackpressure {
                current: observed.current,
                capacity: observed.capacity,
                required_new: observed.required_new_leases,
                replacement_reserve: observed.frontier_replacement_reserve,
            }
            .into());
        }
        Ok(())
    }

    /// Startup-only attachment. No caller-supplied root Rules/Store/model
    /// mirror can replace the owners already held by this port.
    pub fn attach_rovent_frontier(
        &mut self,
        registration: FrontierRegistration,
        limits: FrontierLimits,
        actual: AdaptiveFrontierStartup<'_>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        self.attach_rovent_frontier_inner(registration, limits, actual, None, clock, now)
    }

    /// An actual current-root external owner is required independently of
    /// registration metadata. Its reported work never acquires a native ledger.
    pub fn attach_rovent_external_frontier(
        &mut self,
        registration: FrontierRegistration,
        limits: FrontierLimits,
        actual: AdaptiveFrontierStartup<'_>,
        source: &crate::external_cpu::ProfiledExternalRootSource,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        if !self.engine.is_external() || actual.native_ledger.is_some() {
            return Err(FrontierError::ForeignRoot);
        }
        self.attach_rovent_frontier_inner(registration, limits, actual, Some(source), clock, now)
    }

    fn attach_rovent_frontier_inner(
        &mut self,
        registration: FrontierRegistration,
        limits: FrontierLimits,
        actual: AdaptiveFrontierStartup<'_>,
        external: Option<&crate::external_cpu::ProfiledExternalRootSource>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        self.check_live().map_err(|_| FrontierError::ForeignRoot)?;
        if self.engine.semantic.rovent_frontier.is_some()
            || self.root_rules_scanned
            || self.next_question != 1
            || !self.accepted.is_empty()
            || !self.cpu_admissions.is_empty()
            || !self.profiled_external.is_empty()
        {
            return Err(FrontierError::Invalid(
                "frontier attachment after root work",
            ));
        }
        let namespace = self
            .engine
            .semantic
            .model
            .registered_representation_namespace()
            .ok_or(FrontierError::Invalid(
                "actual native model namespace absent",
            ))?;
        namespace
            .verify_identity(
                self.engine.semantic.model.identity(),
                self.engine.semantic.model.value_identity(),
            )
            .map_err(|_| FrontierError::ForeignRoot)?;
        let frontier = match external {
            Some(source) => RoventFrontier::new_external(
                registration,
                limits,
                &self.engine.semantic.stores,
                source,
                clock,
                now,
            )?,
            None => RoventFrontier::new(
                registration,
                limits,
                &self.engine.semantic.stores,
                clock,
                now,
            )?,
        };
        let attachment = FrontierAttachment {
            authority: self.authority,
            mode: actual.mode,
            original_deadline: self.deadline,
            root_budget: actual.root_budget,
            root_rules: &self.rules,
            root_moves: actual.root_moves,
            root_state: self.engine.semantic.nodes[self.root].state,
            model_namespace: namespace,
            native_ledger: actual.native_ledger,
        };
        match external {
            Some(source) => frontier.validate_external_attachment(
                attachment,
                &self.engine.semantic.stores,
                source,
            )?,
            None => frontier.validate_attachment(attachment, &self.engine.semantic.stores)?,
        }
        self.engine.semantic.rovent_frontier = Some(Box::new(frontier));
        Ok(())
    }

    pub fn rovent_frontier(&self) -> Option<&RoventFrontier> {
        self.engine.semantic.rovent_frontier.as_deref()
    }

    /// Inspect only the originally admitted root moves. The original clock
    /// and controller charge every actual checked Rules transition; this is
    /// neither a model opinion nor an extra search outside the root quota.
    pub fn prepare_rovent_root_rules(
        &mut self,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), FrontierError> {
        let scan_started = Instant::now();
        let frontier = self
            .rovent_frontier()
            .ok_or(FrontierError::Invalid("Rovent frontier absent"))?;
        if frontier.registration().authority != self.authority
            || frontier.registration().root_state != self.engine.semantic.nodes[self.root].state
            || !Arc::ptr_eq(&frontier.registration().root_rules, &self.rules)
        {
            return Err(FrontierError::ForeignRoot);
        }
        let attachment = FrontierAttachment {
            authority: self.authority,
            mode: frontier.registration().mode,
            original_deadline: self.deadline,
            root_budget: frontier.registration().root_budget,
            root_rules: &self.rules,
            root_moves: frontier.root_moves(),
            root_state: self.engine.semantic.nodes[self.root].state,
            model_namespace: self
                .model_namespace()
                .map_err(|_| FrontierError::ForeignRoot)?,
            native_ledger: frontier.registration().native_ledger.as_ref(),
        };
        if let Some(source) = frontier.external_source() {
            frontier.validate_external_attachment(
                attachment,
                &self.engine.semantic.stores,
                source,
            )?;
        } else {
            frontier.validate_attachment(attachment, &self.engine.semantic.stores)?;
        }
        if clock != self.deadline.clock {
            return Err(FrontierError::ForeignRoot);
        }
        self.check_rovent_scan_live()?;
        let observed_now = self
            .rovent_frontier()
            .unwrap()
            .original_tick_after_elapsed(clock, now, scan_started.elapsed())?;
        self.engine
            .semantic
            .rovent_frontier
            .as_mut()
            .unwrap()
            .admit_role_input(clock, observed_now)?;
        if self.root_rules_scanned {
            return Ok(());
        }
        let moves = self
            .rovent_frontier()
            .unwrap()
            .root_moves()
            .allowed()
            .to_vec();
        for movement in moves {
            self.check_rovent_scan_live()?;
            let observed_now = self
                .rovent_frontier()
                .unwrap()
                .original_tick_after_elapsed(clock, now, scan_started.elapsed())?;
            let child = {
                let frontier = self.engine.semantic.rovent_frontier.as_mut().unwrap();
                let path = frontier.observe_root_child(movement, clock, observed_now)?;
                let child =
                    Position::from_snapshot(frontier.path_snapshot(path)?, Default::default())?;
                matches!(
                    child.classify_position()?.play_status,
                    PlayStatus::Terminal { .. }
                )
                .then_some(child)
            };
            if let Some(position) = child {
                // The controller already checked and charged this exact edge.
                // Adopt its immutable result instead of replaying make_move.
                if let Some(edge) = self.engine.semantic.nodes[self.root]
                    .edges
                    .iter()
                    .find(|edge| edge.movement == movement)
                {
                    if !self.engine.semantic.nodes[edge.child]
                        .position
                        .snapshot()
                        .same_state(&position.snapshot())
                    {
                        return Err(FrontierError::ForeignRoot);
                    }
                } else {
                    let child = Self::intern_hot(&mut self.engine, position)
                        .map_err(|_| FrontierError::StoreUnavailable)?;
                    self.engine.semantic.nodes[self.root]
                        .edges
                        .push(Edge { movement, child });
                    self.counters.examined_edges += 1;
                }
            }
        }
        self.check_rovent_scan_live()?;
        self.root_rules_scanned = true;
        Ok(())
    }

    pub(super) fn check_rovent_scan_live(&mut self) -> Result<(), FrontierError> {
        match self.check_live() {
            Ok(()) => Ok(()),
            Err(PalsError::Role(RoleError::Deadline)) => self
                .engine
                .semantic
                .rovent_frontier
                .as_mut()
                .ok_or(FrontierError::ForeignRoot)?
                .observe_original_hard_deadline_expiry(self.authority, self.deadline),
            Err(_) => Err(FrontierError::ForeignRoot),
        }
    }

    /// Disjoint borrowing of the real controller and its Store. Controller
    /// APIs retain their own original deadline, receipt and pin validation.
    /// Late known-work settlement remains possible after admission closes.
    pub fn with_rovent_frontier<T>(
        &mut self,
        apply: impl FnOnce(&mut RoventFrontier, &PalsStores) -> Result<T, FrontierError>,
    ) -> Result<Option<T>, FrontierError> {
        let semantic = &mut self.engine.semantic;
        semantic
            .rovent_frontier
            .as_deref_mut()
            .map(|frontier| apply(frontier, &semantic.stores))
            .transpose()
    }

    /// Always available for evidence recovery, including stopped/unknown work.
    /// This copy neither releases pins nor refunds outstanding reservations.
    pub fn rovent_frontier_receipt(&self) -> Option<AdaptiveFrontierReceipt> {
        self.rovent_frontier()
            .map(|frontier| AdaptiveFrontierReceipt {
                root_budget: frontier.registration().root_budget,
                snapshot: frontier.snapshot(),
            })
    }

    /// The concrete manager invokes this at its settled role-admission bound,
    /// before normal quiescence. Legacy owners have no Frontier to change.
    pub fn close_rovent_role_call_budget(
        &mut self,
        admitted: u64,
        limit: u64,
    ) -> Result<(), PalsError> {
        if self.rovent_frontier().is_none() {
            return Ok(());
        }
        if limit != self.engine.config.max_role_calls
            || admitted < limit
            || self.counters.role_calls > admitted
        {
            return Err(PalsError::InvalidConfig);
        }
        self.check_rovent_scan_live()?;
        self.engine
            .semantic
            .rovent_frontier
            .as_deref_mut()
            .ok_or(PalsError::InvalidConfig)?
            .close_role_call_budget();
        Ok(())
    }

    /// The concrete owner must first drain role/CPU workers and recover this
    /// receipt. Unknown/outstanding native usage keeps the opaque owner beside
    /// its Store, including across legacy into_engine ownership transfers.
    /// A successful retirement makes no goal-success or physical-fence claim.
    pub fn retire_rovent_frontier(
        &mut self,
    ) -> Result<Option<AdaptiveFrontierReceipt>, FrontierError> {
        let Some(receipt) = self.rovent_frontier_receipt() else {
            return Ok(None);
        };
        if receipt
            .snapshot
            .external_work
            .is_some_and(|work| work.outstanding_questions != 0)
        {
            // The profiled worker's original terminal packets are separate
            // from own-R native credit. Keep this controller/Store owner until
            // they are recovered; even then the caller needs its real fence.
            return Err(FrontierError::MissingAcceptance);
        }
        if !receipt.snapshot.work.native_usage_known
            || receipt.snapshot.work.outstanding_native_executions != 0
            || receipt.snapshot.work.reserved_native_nodes != 0
            || self
                .cpu_admissions
                .values()
                .any(|admission| admission.grant.is_some() || admission.continuation_retained)
        {
            return Err(FrontierError::NativeUnknown);
        }
        self.engine.semantic.rovent_frontier.take();
        Ok(Some(receipt))
    }

    pub(super) fn root_move_allowed(&self, movement: BoardMove) -> bool {
        self.rovent_frontier()
            .is_none_or(|frontier| frontier.root_moves().allowed().contains(&movement))
    }
}

#[cfg(test)]
pub(in crate::pals::engine::adaptive_port) mod tests {
    // Authored CPU/mock native receipts exercise contract custody only. None
    // of these fixtures runs a role model or establishes GPU/Fresh execution.
    use super::*;
    use crate::contract_time::InstantClock;
    use crate::cpu_worker::CpuCreditLedger;
    use crate::pals::coordinator::{
        AcceptedRoleStep, NativeAcceptanceIssuer, PhysicalRoleCompletion, RoleCompletion,
        RoleSubmission, RoleTicketRegistry,
    };
    use crate::pals::rovent_frontier::{FrontierFrameKind, FrontierGoalStatus};
    use rz_contracts::ExecutionId;
    use rz_contracts::rovent::SearchGoal;
    use rz_runtime::adaptive::CoordinatorPolicy;
    use std::num::NonZeroU16;

    type Port = AdaptiveEnginePort<LegalOrderRoleMock>;
    type Work = CoordinatorWork<ReviewView, AdaptivePortQuestion>;

    fn copied_registration(source: &FrontierRegistration) -> FrontierRegistration {
        FrontierRegistration {
            authority: source.authority,
            mode: source.mode,
            original_deadline: source.original_deadline,
            root_budget: source.root_budget,
            goal: source.goal,
            nodes: source.nodes,
            selective_policy: source.selective_policy,
            root_rules: Arc::clone(&source.root_rules),
            root_moves: Arc::clone(&source.root_moves),
            root_state: source.root_state,
            model_namespace: Arc::clone(&source.model_namespace),
            native_ledger: source.native_ledger.clone(),
        }
    }

    fn limits() -> FrontierLimits {
        FrontierLimits {
            maximum_goal_plies: NonZeroU16::new(16).unwrap(),
            maximum_frames: 16,
            maximum_paths: 64,
            maximum_mate_nodes: 64,
            maximum_native_receipts: 16,
        }
    }

    pub(in crate::pals::engine::adaptive_port) fn controller(port: &Port) -> RoventFrontier {
        let domain = Arc::new(RootMoveDomain::new(&port.rules, None).unwrap());
        let value = port.engine.semantic.model.value_identity().unwrap().clone();
        let namespace = Arc::new(
            AdaptiveModelNamespace::new(
                value,
                ContractDigest([5; 32]),
                ContractDigest([6; 32]),
                rz_contracts::PrecisionProfile::Fp32,
                1,
            )
            .unwrap(),
        );
        let ledger = CpuCreditLedger::new(
            port.authority,
            2000,
            InstantClock::new(port.deadline.clock, Instant::now()),
        )
        .unwrap();
        RoventFrontier::new(
            FrontierRegistration {
                authority: port.authority,
                mode: ExecutionMode::CpuPcr,
                original_deadline: port.deadline,
                root_budget: ContractDigest([10; 32]),
                goal: SearchGoal {
                    depth_plies: NonZeroU16::new(2),
                    mate_moves: None,
                },
                nodes: Some(2000),
                selective_policy: ContractDigest([11; 32]),
                root_rules: Arc::clone(&port.rules),
                root_moves: domain,
                root_state: port.engine.semantic.nodes[port.root].state,
                model_namespace: namespace,
                native_ledger: Some(ledger.identity()),
            },
            FrontierLimits {
                maximum_goal_plies: NonZeroU16::new(16).unwrap(),
                maximum_frames: 16,
                maximum_paths: 64,
                maximum_mate_nodes: 64,
                maximum_native_receipts: 16,
            },
            &port.engine.semantic.stores,
            port.deadline.clock,
            MonotonicTick(0),
        )
        .unwrap()
    }

    fn accepted(work: &Work) -> (AcceptedRoleStep, RoleOutput) {
        let (_, output) = super::super::port_tests::output(work, 1);
        let accepted = accepted_output(work, &output);
        (accepted, output)
    }

    fn accepted_output(work: &Work, output: &RoleOutput) -> AcceptedRoleStep {
        let (expected, _) = super::super::port_tests::output(work, output.id.sequence);
        let mut registry = RoleTicketRegistry::new(
            work.frame.authority,
            512,
            CoordinatorPolicy::Fixed,
            output.id.sequence,
        )
        .unwrap();
        let ticket = registry
            .admit(
                RoleSubmission {
                    question: expected.question,
                    view: expected.view,
                    representation: expected.representation,
                    role: expected.role,
                    original_deadline: expected.original_deadline,
                },
                Arc::clone(&work.input.view),
                work.original_deadline.clock,
                MonotonicTick(0),
            )
            .unwrap();
        assert_eq!(ticket, expected);
        registry.take_dispatch().unwrap();
        registry
            .complete_reciprocal_role(RoleCompletion {
                ticket,
                physical: PhysicalRoleCompletion::Confirmed,
                result: Ok(output.clone()),
            })
            .unwrap();
        registry
            .take_ready(work.original_deadline.clock, MonotonicTick(0))
            .unwrap();
        let (issuer, verifier) = NativeAcceptanceIssuer::new_pair();
        registry
            .register_native_acceptance_verifier(verifier)
            .unwrap();
        let permit = registry
            .issue_reciprocal_native_acceptance(
                &ticket,
                work.original_deadline.clock,
                MonotonicTick(0),
            )
            .unwrap();
        registry
            .accept_reciprocal_native(
                issuer.issue_success(permit),
                work.original_deadline.clock,
                MonotonicTick(0),
            )
            .unwrap()
    }

    fn fresh_descendant(port: &mut Port, source: &Work) -> Work {
        let prefix = vec![
            BoardMove::from_uci("e2e4").unwrap(),
            BoardMove::from_uci("e7e5").unwrap(),
        ];
        let node = port.replay_node(&prefix).unwrap();
        let input = port
            .input(
                node,
                RefinementStage::ProposalValueFresh,
                Some(prefix[0]),
                prefix.clone(),
                prefix,
                None,
                None,
                None,
                false,
                None,
                None,
            )
            .unwrap();
        Work {
            input,
            ..source.clone()
        }
    }

    #[test]
    fn private_frontier_source_rejects_cloned_public_field_changes() {
        let mut port = super::super::port_tests::setup(Position::startpos(), 4);
        let work = super::super::port_tests::work(
            &mut port,
            "e2e4",
            RefinementStage::ProposalContinuation,
            1,
        );
        AdaptivePortQuestion::validate_frontier_source(&work, &port.engine.semantic.stores)
            .unwrap();
        let mut changed = work.clone();
        Arc::make_mut(&mut changed.input.question)
            .context
            .public_revision += 1;
        assert!(
            AdaptivePortQuestion::validate_frontier_source(&changed, &port.engine.semantic.stores)
                .is_err()
        );
        let mut changed = work.clone();
        Arc::make_mut(&mut changed.input.question).deadline += Duration::from_millis(1);
        assert!(
            AdaptivePortQuestion::validate_frontier_source(&changed, &port.engine.semantic.stores)
                .is_err()
        );
        let mut changed = work.clone();
        let fen = changed.input.question.position.to_fen();
        Arc::make_mut(&mut changed.input.question).position = Position::from_fen(&fen).unwrap();
        assert!(
            AdaptivePortQuestion::validate_frontier_source(&changed, &port.engine.semantic.stores)
                .is_err()
        );
        let mut changed = work.clone();
        changed.original_deadline.at.0 += 1;
        assert!(
            AdaptivePortQuestion::validate_frontier_source(&changed, &port.engine.semantic.stores)
                .is_err()
        );
        let replacement = super::super::port_tests::work(
            &mut port,
            "d2d4",
            RefinementStage::ProposalContinuation,
            2,
        );
        let mut changed = work.clone();
        changed.input.view = replacement.input.view;
        changed.input.view_seal = replacement.input.view_seal;
        changed.input.public_context = replacement.input.public_context;
        assert!(
            AdaptivePortQuestion::validate_frontier_source(&changed, &port.engine.semantic.stores)
                .is_err()
        );
    }

    #[test]
    fn actual_local_fresh_descendant_closes_only_its_own_responsibility() {
        let mut port = super::super::port_tests::setup(Position::startpos(), 4);
        let work = super::super::port_tests::work(
            &mut port,
            "e2e4",
            RefinementStage::ProposalContinuation,
            1,
        );
        let mut frontier = controller(&port);
        let mut obligations = Vec::new();
        let counter =
            super::super::port_tests::work(&mut port, "e2e4", RefinementStage::CriticReply, 1);
        let repair = super::super::port_tests::work(&mut port, "e2e4", RefinementStage::Repair, 1);
        for (kind, source) in [
            (FrontierFrameKind::Candidate, &work),
            (FrontierFrameKind::Counter, &counter),
            (FrontierFrameKind::Repair, &repair),
        ] {
            obligations.push(
                frontier
                    .register_local_frame(
                        source,
                        &port.engine.semantic.stores,
                        kind,
                        port.deadline.clock,
                        MonotonicTick(0),
                    )
                    .unwrap(),
            );
        }
        let endpoint = fresh_descendant(&mut port, &work);
        let path = frontier
            .observe_local_question(
                &endpoint,
                &port.engine.semantic.stores,
                port.deadline.clock,
                MonotonicTick(0),
            )
            .unwrap();
        let (accepted, output) = accepted(&endpoint);
        let physical = AdaptivePortRoleEvidence {
            execution: Some(ExecutionId::new(port.authority.epoch, 1)),
            completed_nn_inputs: Some(1),
            measurement_complete: true,
            physical_fresh: Some(true),
        };
        let mut incomplete = physical;
        incomplete.physical_fresh = Some(false);
        assert_eq!(
            frontier.accept_local_fresh_endpoint(
                obligations[0],
                &endpoint,
                &port.engine.semantic.stores,
                path,
                &accepted,
                &output,
                incomplete,
                port.deadline.clock,
                MonotonicTick(0),
            ),
            Err(FrontierError::MissingAcceptance)
        );
        let mut changed = output.clone();
        let RolePayload::Proposal {
            candidate_logits, ..
        } = &mut changed.payload
        else {
            unreachable!("Fresh fixture returns the P head");
        };
        candidate_logits[0] += 1.0;
        assert_eq!(
            frontier.accept_local_fresh_endpoint(
                obligations[0],
                &endpoint,
                &port.engine.semantic.stores,
                path,
                &accepted,
                &changed,
                physical,
                port.deadline.clock,
                MonotonicTick(0),
            ),
            Err(FrontierError::MissingAcceptance)
        );
        frontier
            .accept_local_fresh_endpoint(
                obligations[0],
                &endpoint,
                &port.engine.semantic.stores,
                path,
                &accepted,
                &output,
                physical,
                port.deadline.clock,
                MonotonicTick(0),
            )
            .unwrap();
        frontier
            .seal_selected_policy(port.deadline.clock, MonotonicTick(0))
            .unwrap();
        let snapshot = frontier.snapshot();
        assert_eq!(snapshot.completed_frames, 1);
        assert_eq!(snapshot.horizon.declared_obligations, 3);
        assert_eq!(snapshot.horizon.completed_obligations, 1);
        assert_eq!(snapshot.goal, FrontierGoalStatus::Pending);
    }

    #[test]
    fn actual_local_rules_terminal_closes_early_without_inventing_depth() {
        let mut port = super::super::port_tests::setup(
            Position::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").unwrap(),
            4,
        );
        let work =
            super::super::port_tests::work(&mut port, "g6g7", RefinementStage::CpuValidation, 1);
        let mut frontier = controller(&port);
        let obligation = frontier
            .register_local_frame(
                &work,
                &port.engine.semantic.stores,
                FrontierFrameKind::Candidate,
                port.deadline.clock,
                MonotonicTick(0),
            )
            .unwrap();
        let endpoint = frontier
            .observe_local_question(
                &work,
                &port.engine.semantic.stores,
                port.deadline.clock,
                MonotonicTick(0),
            )
            .unwrap();
        frontier
            .accept_local_rules_endpoint(
                obligation,
                &work,
                &port.engine.semantic.stores,
                endpoint,
                port.deadline.clock,
                MonotonicTick(0),
            )
            .unwrap();
        frontier
            .seal_selected_policy(port.deadline.clock, MonotonicTick(0))
            .unwrap();
        let snapshot = frontier.snapshot();
        assert!(snapshot.horizon.target_satisfied);
        assert_eq!(snapshot.horizon.completed_depth.unwrap().0, 1);
        assert_eq!(snapshot.horizon.maximum_examined_ply.0, 1);
        assert!(frontier.registration().goal.mate_moves.is_none());
        // The paid terminal edge also passively proves an exact mate-in-one.
        // Depth still closes at the actual one ply; it is not fabricated up
        // to the requested depth and no extra mate question is generated.
        assert_eq!(snapshot.goal, FrontierGoalStatus::MateSatisfied);
        let strategy = snapshot.mate_strategy.as_ref().unwrap();
        assert_eq!(strategy.best_move(), BoardMove::from_uci("g6g7").unwrap());
        assert!(strategy.validates_root(
            port.rules.rules().snapshot(),
            frontier.root_moves().allowed(),
        ));
        assert_eq!(strategy.conclusion().exact_uci_mate(), Some(1));
        assert_eq!(snapshot.work.checked_transitions, 1);
        assert_eq!(snapshot.work.native_nodes, 0);
    }

    #[test]
    fn frontier_lease_gate_preserves_replacement_without_changing_the_hard_ceiling() {
        let mut port = super::super::port_tests::setup(Position::startpos(), 4);
        let frontier = controller(&port);
        port.engine.semantic.rovent_frontier = Some(Box::new(frontier));
        let stores = &port.engine.semantic.stores;
        let capacity = stores.review_pin_capacity();
        assert_eq!(capacity, 16);
        let mut held = Vec::new();
        while stores.review_pin_count().unwrap() < capacity - 1 {
            held.push(stores.pin_review(Default::default()).unwrap());
        }
        let observed = port.review_lease_admission().unwrap();
        assert_eq!(observed.current, capacity - 1);
        assert_eq!(observed.frontier_replacement_reserve, 1);
        assert!(!observed.may_prepare);
        assert!(matches!(
            port.ensure_new_review_leases(true),
            Err(PalsError::Store(StoreError::ReviewLeaseBackpressure { .. }))
        ));
        // A frozen existing view needs no new selected lease. It does not
        // spend the replacement slot or fabricate another ticket/NN input.
        port.ensure_new_review_leases(false).unwrap();
        let last = port
            .engine
            .semantic
            .stores
            .pin_review(Default::default())
            .unwrap();
        assert_eq!(
            port.engine.semantic.stores.review_pin_count().unwrap(),
            capacity
        );
        drop(last);
        held.pop();
        assert!(port.review_lease_admission().unwrap().may_prepare);
        port.ensure_new_review_leases(true).unwrap();
    }

    #[test]
    fn legacy_engine_transfer_retains_the_frontier_and_original_budget() {
        let mut port = super::super::port_tests::setup(Position::startpos(), 4);
        let frontier = controller(&port);
        let budget = frontier.registration().root_budget;
        port.engine.semantic.rovent_frontier = Some(Box::new(frontier));
        let engine = port.into_engine();
        let retained = engine.semantic.rovent_frontier.as_ref().unwrap();
        assert_eq!(retained.registration().root_budget, budget);
        assert_eq!(retained.snapshot().work.root_entries, 1);
        assert!(engine.semantic.ensure_frontier_retired().is_err());
    }

    #[test]
    fn legacy_root_rules_scan_cannot_be_relabelled_as_frontier_startup() {
        let mut port = super::super::port_tests::setup(Position::startpos(), 4);
        let frontier = controller(&port);
        let registration = copied_registration(frontier.registration());
        let domain = Arc::clone(&registration.root_moves);
        let ledger = registration.native_ledger.clone();
        drop(frontier);
        port.prepare_root_rules().unwrap();
        assert!(port.root_rules_scanned);
        let result = port.attach_rovent_frontier(
            registration,
            limits(),
            AdaptiveFrontierStartup {
                mode: ExecutionMode::CpuPcr,
                root_budget: ContractDigest([10; 32]),
                root_moves: &domain,
                native_ledger: ledger.as_ref(),
            },
            port.deadline.clock,
            MonotonicTick(0),
        );
        assert_eq!(
            result,
            Err(FrontierError::Invalid(
                "frontier attachment after root work"
            ))
        );
        assert!(port.rovent_frontier().is_none());
    }

    #[test]
    fn actual_hard_deadline_records_the_original_frontier_stop() {
        let mut port = super::super::port_tests::setup(Position::startpos(), 4);
        port.engine.semantic.rovent_frontier = Some(Box::new(controller(&port)));
        let original = port.deadline;
        let before = port.rovent_frontier().unwrap().snapshot().work;
        port.limits.deadline = Instant::now();
        assert_eq!(port.check_rovent_scan_live(), Err(FrontierError::Deadline));
        let snapshot = port.rovent_frontier().unwrap().snapshot();
        assert_eq!(snapshot.original_deadline, original);
        assert_eq!(
            snapshot.goal,
            FrontierGoalStatus::Stopped(crate::pals::rovent_frontier::FrontierStop::Deadline,)
        );
        assert_eq!(snapshot.work.root_entries, before.root_entries);
        assert_eq!(
            snapshot.work.checked_transitions,
            before.checked_transitions
        );
        assert_eq!(snapshot.work.native_nodes, before.native_nodes);
        assert_eq!(
            snapshot.work.reserved_native_nodes,
            before.reserved_native_nodes
        );
        assert!(!port.root_rules_scanned);
    }

    #[test]
    fn scan_tick_preserves_the_supplied_origin_and_adds_observed_elapsed_time() {
        let port = super::super::port_tests::setup(Position::startpos(), 4);
        let frontier = controller(&port);
        let earlier = Instant::now()
            .checked_sub(Duration::from_millis(2))
            .unwrap();
        let tick = frontier
            .original_tick_after_elapsed(port.deadline.clock, MonotonicTick(100), earlier.elapsed())
            .unwrap();
        assert!(tick.0 >= 2_000_100);
        assert!(
            frontier
                .original_tick_after_elapsed(
                    port.deadline.clock,
                    MonotonicTick(u64::MAX),
                    earlier.elapsed()
                )
                .is_err()
        );
        assert_eq!(
            frontier.original_tick_after_elapsed(
                ClockDomain(rz_contracts::ProcessEpoch(99)),
                MonotonicTick(100),
                earlier.elapsed()
            ),
            Err(FrontierError::ForeignRoot)
        );
    }

    #[test]
    fn manager_hooks_keep_original_lane_and_exact_fresh_completion_on_retry() {
        let mut port = super::super::port_tests::setup_with_resolver(
            Position::startpos(),
            4,
            ResolverPolicy::ModelWdlRestricted,
        );
        let work = super::super::port_tests::work(
            &mut port,
            "e2e4",
            RefinementStage::ProposalContinuation,
            1,
        );
        let frontier = controller(&port);
        port.engine.semantic.rovent_frontier = Some(Box::new(frontier));
        let clock = port.deadline.clock;
        for _ in 0..2 {
            assert!(
                port.rovent_frame_admitted(
                    &work,
                    FrontierFrameKind::Candidate,
                    clock,
                    MonotonicTick(0),
                )
                .unwrap()
            );
        }
        assert_eq!(
            port.rovent_frontier().unwrap().snapshot().registered_frames,
            1
        );
        // This models the actual selected continuation producer: charge the
        // descendant through the attached controller before submitting Fresh.
        let endpoint = fresh_descendant(&mut port, &work);
        let mut changed = work.clone();
        Arc::make_mut(&mut changed.input.question)
            .context
            .public_revision += 1;
        assert!(
            port.rovent_frame_admitted(
                &changed,
                FrontierFrameKind::Candidate,
                clock,
                MonotonicTick(0),
            )
            .is_err()
        );
        let (accepted, output) = accepted(&endpoint);
        let physical = AdaptivePortRoleEvidence {
            execution: Some(ExecutionId::new(port.authority.epoch, 1)),
            completed_nn_inputs: Some(1),
            measurement_complete: true,
            physical_fresh: Some(true),
        };
        for _ in 0..2 {
            assert!(
                port.rovent_fresh_accepted(
                    &endpoint,
                    &accepted,
                    &output,
                    physical,
                    clock,
                    MonotonicTick(0),
                )
                .unwrap()
            );
        }
        assert_eq!(port.counters.role_calls, 1);
        assert_eq!(port.counters.value_calls, 1);
        assert_eq!(port.counters.consumed_role_outputs, 1);
        assert!(
            port.role_consumption_recorded(accepted.ticket(), &endpoint.input.view)
                .unwrap()
        );
        let mut changed_output = output.clone();
        let RolePayload::Proposal {
            candidate_logits, ..
        } = &mut changed_output.payload
        else {
            unreachable!()
        };
        candidate_logits[0] += 1.0;
        assert!(
            port.rovent_fresh_accepted(
                &endpoint,
                &accepted,
                &changed_output,
                physical,
                clock,
                MonotonicTick(0),
            )
            .is_err()
        );
        let budget = port.rovent_frontier().unwrap().registration().root_budget;
        assert_eq!(
            port.rovent_root_round_sealed(ContractDigest([99; 32]), clock, MonotonicTick(0)),
            Err(FrontierError::ForeignRoot),
        );
        for _ in 0..2 {
            assert!(
                port.rovent_root_round_sealed(budget, clock, MonotonicTick(0))
                    .unwrap()
            );
        }
        let snapshot = port.rovent_frontier().unwrap().snapshot();
        assert_eq!(snapshot.registered_frames, 1);
        assert_eq!(snapshot.completed_frames, 1);
        assert_eq!(snapshot.goal, FrontierGoalStatus::DepthSatisfied);
    }

    #[test]
    fn zero_step_repair_observes_new_source_without_reusing_prior_unmet_repair_duty() {
        let mut port = super::super::port_tests::setup_with_resolver(
            Position::startpos(),
            4,
            ResolverPolicy::ModelWdlRestricted,
        );
        let original = super::super::port_tests::work(
            &mut port,
            "e2e4",
            RefinementStage::ProposalContinuation,
            1,
        );
        let base = controller(&port);
        let mut registration = copied_registration(base.registration());
        registration.goal.depth_plies = NonZeroU16::new(4);
        drop(base);
        let clock = port.deadline.clock;
        port.engine.semantic.rovent_frontier = Some(Box::new(
            RoventFrontier::new(
                registration,
                limits(),
                &port.engine.semantic.stores,
                clock,
                MonotonicTick(0),
            )
            .unwrap(),
        ));
        let proposal = ["e2e4", "c7c5", "g1f3", "d7d6"]
            .map(|movement| BoardMove::from_uci(movement).unwrap())
            .to_vec();
        let counter = ["e2e4", "e7e5", "g1f3", "b8c6"]
            .map(|movement| BoardMove::from_uci(movement).unwrap())
            .to_vec();
        port.record_hot(RecordKind::Proposal, &proposal, None)
            .unwrap();
        port.record_hot(RecordKind::Counterexample, &counter, None)
            .unwrap();
        let prefix = counter[..2].to_vec();
        let node = port.replay_node(&prefix).unwrap();
        // An authored, Rules-checked starting question supplies no CPU proof.
        // The duty itself must be issued by two real registry-accepted P moves,
        // their original submitted-view consumption and the normal producer.
        let input = port
            .input(
                node,
                RefinementStage::Repair,
                Some(proposal[0]),
                prefix,
                proposal,
                Some(counter),
                None,
                Some(1),
                false,
                None,
                None,
            )
            .unwrap();
        let mut current = Work { input, ..original };
        for (sequence, chosen) in [(1, "b1c3"), (2, "b8c6")] {
            current.budget.role_submissions_remaining -= 1;
            let (ticket, mut output) = super::super::port_tests::output(&current, sequence);
            let RolePayload::Proposal {
                candidate_logits, ..
            } = &mut output.payload
            else {
                panic!("Repair must consume the actual P policy output")
            };
            let chosen = BoardMove::from_uci(chosen).unwrap();
            let index = current
                .input
                .view
                .query()
                .legal
                .iter()
                .position(|movement| *movement == chosen)
                .unwrap();
            candidate_logits[index] = 1.0;
            let accepted = accepted_output(&current, &output);
            assert_eq!(accepted.ticket(), &ticket);
            let step = port.commit_role(&current, &ticket, &output, None).unwrap();
            port.role_consumed(&ticket, &current.input.view).unwrap();
            let FrameTransition::Continue(input) = port.materialize_step(step).unwrap().transition
            else {
                panic!("actual P Repair lost its selected continuation")
            };
            let selected = Work {
                input,
                ..current.clone()
            };
            assert_eq!(
                port.rovent_lane_selected(&current, &selected, clock, MonotonicTick(0))
                    .unwrap(),
                sequence == 2,
            );
            current = selected;
        }
        assert_eq!(current.input.stage, RefinementStage::RepairValidation);
        let prior = port.rovent_frontier().unwrap().snapshot();
        assert_eq!(prior.registered_frames, 1);
        assert_eq!(prior.completed_frames, 0);
        assert_eq!(prior.horizon.declared_obligations, 1);
        assert_eq!(prior.horizon.completed_obligations, 0);
        assert_eq!(port.counters.repair_calls, 2);

        // A later real Counter Fresh selects a different final opponent move.
        // Its unchanged full prefix creates a Repair record, but no P move.
        let proposal = current.input.question.repaired.as_ref().unwrap().clone();
        let counter = ["e2e4", "e7e5", "b1c3", "d7d6"]
            .map(|movement| BoardMove::from_uci(movement).unwrap())
            .to_vec();
        port.record_hot(RecordKind::Counterexample, &counter, None)
            .unwrap();
        let node = port.replay_node(&counter).unwrap();
        let input = port
            .input(
                node,
                RefinementStage::CounterValueFresh,
                Some(counter[0]),
                counter.clone(),
                proposal,
                Some(counter),
                None,
                Some(3),
                false,
                None,
                None,
            )
            .unwrap();
        let mut counter_work = Work { input, ..current };
        counter_work.budget.role_submissions_remaining -= 1;
        let (ticket, output) = super::super::port_tests::output(&counter_work, 3);
        let accepted = accepted_output(&counter_work, &output);
        assert_eq!(accepted.ticket(), &ticket);
        let physical = AdaptivePortRoleEvidence {
            execution: Some(ExecutionId::new(port.authority.epoch, 3)),
            completed_nn_inputs: Some(1),
            measurement_complete: true,
            physical_fresh: Some(true),
        };
        let step = port
            .commit_role(&counter_work, &ticket, &output, Some(physical))
            .unwrap();
        port.role_consumed(&ticket, &counter_work.input.view)
            .unwrap();
        let FrameTransition::Continue(input) = port.materialize_step(step).unwrap().transition
        else {
            panic!("zero-step record lost its original observation question")
        };
        let observation = Work {
            input,
            ..counter_work.clone()
        };
        assert_eq!(observation.input.stage, RefinementStage::RepairValidation);
        assert!(
            port.rovent_lane_selected(&counter_work, &observation, clock, MonotonicTick(0))
                .unwrap()
        );
        assert!(
            !port
                .rovent_frame_admitted(
                    &observation,
                    FrontierFrameKind::Repair,
                    clock,
                    MonotonicTick(0),
                )
                .unwrap()
        );
        let mut changed = observation.clone();
        changed.budget.role_submissions_remaining -= 1;
        assert!(
            port.rovent_frame_admitted(
                &changed,
                FrontierFrameKind::Repair,
                clock,
                MonotonicTick(0),
            )
            .is_err()
        );
        let boundary = port
            .prepare_rovent_goal_boundary(&observation)
            .unwrap()
            .unwrap();
        boundary.validate(&observation).unwrap();
        let (_, input) = boundary.into_parts();
        let mut fresh = Work {
            input,
            ..observation
        };
        fresh.budget.role_submissions_remaining -= 1;
        let (ticket, output) = super::super::port_tests::output(&fresh, 4);
        let accepted = accepted_output(&fresh, &output);
        let physical = AdaptivePortRoleEvidence {
            execution: Some(ExecutionId::new(port.authority.epoch, 4)),
            ..physical
        };
        let mut changed = fresh.clone();
        changed.budget.role_submissions_remaining -= 1;
        assert!(
            port.rovent_fresh_accepted(
                &changed,
                &accepted,
                &output,
                physical,
                clock,
                MonotonicTick(0),
            )
            .is_err()
        );
        assert!(
            !port
                .rovent_fresh_accepted(
                    &fresh,
                    &accepted,
                    &output,
                    physical,
                    clock,
                    MonotonicTick(0),
                )
                .unwrap()
        );
        let step = port
            .commit_role(&fresh, &ticket, &output, Some(physical))
            .unwrap();
        port.role_consumed(&ticket, &fresh.input.view).unwrap();
        assert!(matches!(
            port.materialize_step(step).unwrap().transition,
            FrameTransition::Complete { .. },
        ));
        assert_eq!(port.counters.repair_calls, 2);
        assert_eq!(port.counters.repairs, 2);
        let observed = port.rovent_frontier().unwrap().snapshot();
        assert_eq!(observed.registered_frames, prior.registered_frames);
        assert_eq!(observed.completed_frames, prior.completed_frames);
        assert_eq!(
            observed.horizon.declared_obligations,
            prior.horizon.declared_obligations
        );
        assert_eq!(
            observed.horizon.completed_obligations,
            prior.horizon.completed_obligations
        );
    }

    #[test]
    fn manager_rules_endpoint_hook_closes_exact_terminal_without_native_result() {
        let mut port = super::super::port_tests::setup(
            Position::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").unwrap(),
            4,
        );
        let work =
            super::super::port_tests::work(&mut port, "g6g7", RefinementStage::CpuValidation, 1);
        let frontier = controller(&port);
        port.engine.semantic.rovent_frontier = Some(Box::new(frontier));
        let clock = port.deadline.clock;
        port.rovent_frame_admitted(&work, FrontierFrameKind::Candidate, clock, MonotonicTick(0))
            .unwrap();
        for _ in 0..2 {
            assert!(
                port.rovent_rules_endpoint(&work, clock, MonotonicTick(0))
                    .unwrap()
            );
        }
        let snapshot = port.rovent_frontier().unwrap().snapshot();
        assert_eq!(snapshot.completed_frames, 1);
        assert!(!snapshot.selection_sealed);
        assert_eq!(snapshot.horizon.completed_depth, None);
        let budget = port.rovent_frontier().unwrap().registration().root_budget;
        port.rovent_root_round_sealed(budget, clock, MonotonicTick(0))
            .unwrap();
        let snapshot = port.rovent_frontier().unwrap().snapshot();
        assert_eq!(snapshot.horizon.completed_depth.unwrap().0, 1);
        assert_eq!(port.counters.role_calls, 0);
    }

    include!("checked_replay_tests.rs");
    include!("fresh_usage_tests.rs");
}
