//! Actual manager-to-goal observations and same-owner endpoint transitions.
//! An endpoint transition only issues the next real Fresh question; it never
//! invents a physical completion, a value, or a selected move.
use super::*;
use crate::pals::coordinator::{AcceptedRoleStep, MAX_COORDINATOR_FRAMES};
use crate::pals::rovent_frontier::{FrontierError, FrontierFrameKind, FrontierObligation};
use rz_contracts::{ClockDomain, MonotonicTick};

#[derive(Default)]
pub(super) struct AdaptiveGoalLanes {
    active: HashMap<(FrameId, FrontierFrameKind), ActiveGoalLane>,
    selected: HashMap<ContractDigest, SelectedGoalLane>,
}

enum ActiveGoalLane {
    Obligation(FrontierObligation),
    ZeroStepRepair {
        source: Arc<AdaptiveLaneSelection>,
        previous: Option<FrontierObligation>,
    },
}
impl ActiveGoalLane {
    fn previous_obligation(&self) -> Option<FrontierObligation> {
        match self {
            Self::Obligation(handle) => Some(*handle),
            Self::ZeroStepRepair { previous, .. } => *previous,
        }
    }
}

#[derive(Clone, Copy)]
enum GoalLaneEvidence {
    // Counter may be CPU-discovered or C-selected. Only the completed Repair
    // producer below requires an actual accepted P move before issuing this.
    ActualSelectedLine,
    ZeroStepRepairRecord {
        revision: u64,
        counter_revision: u64,
        root: StateId,
    },
}

enum GoalLaneBudgetPhase<'a> {
    Ready,
    AcceptedRole(&'a AcceptedRoleStep),
}

/// Only actual counter selection or a completed Repair step can put a duty
/// source into a continuation recipe. Zero-step Repair has a separate record
/// observation source and never issues an accepted P line or a new duty. The
/// materializer binds it to the actual new question/view after native source
/// consumption; an arbitrary public work descriptor cannot issue this source.
pub(crate) struct AdaptiveLaneSelection {
    kind: FrontierFrameKind,
    evidence: GoalLaneEvidence,
    frame: FrameId,
    first: rz_contracts::pals::Move16,
    deadline: Deadline,
    budget: FrameBudget,
    parent_question: std::sync::Weak<AdaptivePortQuestion>,
    parent_view: std::sync::Weak<ReviewView>,
    parent_digest: ContractDigest,
    selected_question: Option<std::sync::Weak<AdaptivePortQuestion>>,
    selected_view: Option<std::sync::Weak<ReviewView>>,
    prefix: Vec<BoardMove>,
    counter: Option<Vec<BoardMove>>,
    proposal: Vec<BoardMove>,
    repaired: Option<Vec<BoardMove>>,
    attack: Option<usize>,
    source: Option<ObservationId>,
}
struct SelectedGoalLane {
    source: Arc<AdaptiveLaneSelection>,
}

impl AdaptiveLaneSelection {
    pub(crate) fn obligation_kind(&self) -> Option<FrontierFrameKind> {
        match self.evidence {
            GoalLaneEvidence::ActualSelectedLine => Some(self.kind),
            GoalLaneEvidence::ZeroStepRepairRecord { .. } => None,
        }
    }
    pub(crate) fn validates_transition(
        &self,
        original: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        selected: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> bool {
        self.matches_parent(original)
            && self.matches_selected(&selected.input)
            && self
                .selected_question
                .as_ref()
                .is_some_and(|question| question.ptr_eq(&Arc::downgrade(&selected.input.question)))
            && self
                .selected_view
                .as_ref()
                .is_some_and(|view| view.ptr_eq(&Arc::downgrade(&selected.input.view)))
            && selected.frame == original.frame
            && selected.original_first_move == original.original_first_move
            && selected.budget == original.budget
            && selected.original_deadline == original.original_deadline
    }
    fn matches_parent(&self, work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>) -> bool {
        self.frame == work.frame
            && self.first == work.original_first_move
            && self.deadline == work.original_deadline
            && self.budget == work.budget
            && self.parent_digest == work.input.question_digest
            && self
                .parent_question
                .ptr_eq(&Arc::downgrade(&work.input.question))
            && self.parent_view.ptr_eq(&Arc::downgrade(&work.input.view))
    }
    fn matches_selected(&self, input: &FrameInput<ReviewView, AdaptivePortQuestion>) -> bool {
        let stage_matches = match self.kind {
            FrontierFrameKind::Counter => matches!(
                input.stage,
                RefinementStage::CounterContinuation | RefinementStage::CounterValidation
            ),
            FrontierFrameKind::Repair => match self.evidence {
                GoalLaneEvidence::ActualSelectedLine => {
                    input.stage == RefinementStage::RepairValidation
                }
                GoalLaneEvidence::ZeroStepRepairRecord { .. } => matches!(
                    input.stage,
                    RefinementStage::RepairValidation | RefinementStage::RepairValueFresh
                ),
            },
            FrontierFrameKind::Candidate => false,
        };
        stage_matches && self.matches_line(input)
    }
    fn matches_line(&self, input: &FrameInput<ReviewView, AdaptivePortQuestion>) -> bool {
        input.question.context.prefix == self.prefix
            && input.question.counter == self.counter
            && input.question.proposal == self.proposal
            && input.question.repaired == self.repaired
            && input.question.attack_ply == self.attack
            && input.question.counter_source == self.source
            && input.question.first.is_some_and(|first| {
                rz_contracts::Move::try_from(first).is_ok_and(|movement| {
                    rz_contracts::pals::Move16::encode(movement) == self.first
                })
            })
    }
    fn permits_zero_step_observation(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        phase: GoalLaneBudgetPhase<'_>,
    ) -> bool {
        let mut expected = self.budget;
        if let GoalLaneBudgetPhase::AcceptedRole(accepted) = phase {
            let ticket = accepted.ticket();
            let Some(remaining) = expected.role_submissions_remaining.checked_sub(1) else {
                return false;
            };
            if ticket.question != work.input.question_digest
                || ticket.view != work.input.view_seal
                || ticket.authority != self.frame.authority
                || ticket.original_deadline != self.deadline
            {
                return false;
            }
            expected.role_submissions_remaining = remaining;
        }
        matches!(self.evidence, GoalLaneEvidence::ZeroStepRepairRecord { .. })
            && self.kind == FrontierFrameKind::Repair
            && work.frame == self.frame
            && work.original_first_move == self.first
            && work.original_deadline == self.deadline
            && work.budget == expected
            && self.matches_line(&work.input)
    }
}

/// An unsubmitted same-owner validation at its original go horizon. Only the
/// live Search port can issue this move-only transition. It is neither a CPU
/// completion nor a Fresh value; the next real role ticket still supplies the
/// physically retired Fresh evidence. Keep its immutable Arc custody on heap.
pub struct AdaptiveGoalBoundary<Q> {
    seal: Box<GoalBoundarySeal<Q>>,
}
struct GoalBoundarySeal<Q> {
    frame: FrameId,
    first: rz_contracts::pals::Move16,
    deadline: Deadline,
    budget: FrameBudget,
    original: FrameInput<ReviewView, Q>,
    root_budget: ContractDigest,
    target: usize,
    evidence: ContractDigest,
    next: FrameInput<ReviewView, Q>,
}
impl<Q> AdaptiveGoalBoundary<Q> {
    pub(crate) fn validate(&self, work: &CoordinatorWork<ReviewView, Q>) -> Result<(), PalsError> {
        let seal = &self.seal;
        let next = &seal.next;
        let original = &seal.original;
        let stage_matches = matches!(
            (original.stage, next.stage),
            (
                RefinementStage::CpuValidation,
                RefinementStage::ProposalValueFresh
            ) | (
                RefinementStage::CounterValidation,
                RefinementStage::CounterValueFresh
            ) | (
                RefinementStage::RepairValidation,
                RefinementStage::RepairValueFresh
            ) | (
                RefinementStage::RecheckCounterValidation,
                RefinementStage::FreshRepairedEndpoint
            )
        );
        if work.frame != seal.frame
            || work.original_first_move != seal.first
            || work.original_deadline != seal.deadline
            || work.budget != seal.budget
            || work.input.stage != original.stage
            || work.input.question_digest != original.question_digest
            || work.input.view_seal != original.view_seal
            || work.input.public_context != original.public_context
            || !Arc::ptr_eq(&work.input.question, &original.question)
            || !Arc::ptr_eq(&work.input.view, &original.view)
            || seal.root_budget == ContractDigest([0; 32])
            || seal.target == 0
            || original.view.query().prefix.len() != seal.target
            || !stage_matches
            || next.view.query().purpose != RoleQueryPurpose::ValueFresh
            || next.view.query().prefix.len() > seal.target
            || next.view_seal.authority != seal.frame.authority
            || next.view_seal.root_state != original.view_seal.root_state
            || next.view.root_state() != original.view.root_state()
            || next.view.generation() != original.view.generation()
            || next.public_context.0 != next.view.record_scope_digest()
            || (original.stage != RefinementStage::RecheckCounterValidation
                && next.view.query().prefix != original.view.query().prefix)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }
    pub(crate) fn into_parts(self) -> (ContractDigest, FrameInput<ReviewView, Q>) {
        (self.seal.evidence, self.seal.next)
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    fn lane_selection_origin(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        kind: FrontierFrameKind,
        evidence: GoalLaneEvidence,
    ) -> Result<Box<AdaptiveLaneSelection>, PalsError> {
        self.check_work(work)?;
        let query = &work.input.question;
        Ok(Box::new(AdaptiveLaneSelection {
            kind,
            evidence,
            frame: work.frame,
            first: work.original_first_move,
            deadline: work.original_deadline,
            budget: work.budget,
            parent_question: Arc::downgrade(&work.input.question),
            parent_view: Arc::downgrade(&work.input.view),
            parent_digest: work.input.question_digest,
            selected_question: None,
            selected_view: None,
            prefix: query.context.prefix.clone(),
            counter: query.counter.clone(),
            proposal: query.proposal.clone(),
            repaired: query.repaired.clone(),
            attack: query.attack_ply,
            source: query.counter_source,
        }))
    }

    pub(super) fn prepare_counter_goal_selection(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        counter: &[BoardMove],
        proposal: &[BoardMove],
        attack: usize,
        source: Option<ObservationId>,
    ) -> Result<Option<Box<AdaptiveLaneSelection>>, PalsError> {
        if self.rovent_frontier().is_none() {
            return Ok(None);
        }
        let mut selection = self.lane_selection_origin(
            work,
            FrontierFrameKind::Counter,
            GoalLaneEvidence::ActualSelectedLine,
        )?;
        selection.prefix = counter.to_vec();
        selection.counter = Some(counter.to_vec());
        selection.proposal = proposal.to_vec();
        selection.repaired = None;
        selection.attack = Some(attack);
        selection.source = source;
        Ok(Some(selection))
    }

    pub(super) fn prepare_completed_repair_goal_selection(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        prefix: &[BoardMove],
    ) -> Result<Option<Box<AdaptiveLaneSelection>>, PalsError> {
        if self.rovent_frontier().is_none() {
            return Ok(None);
        }
        if work.input.stage != RefinementStage::Repair
            || prefix.len() != work.input.question.context.prefix.len() + 1
            || !prefix.starts_with(&work.input.question.context.prefix)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let mut selection = self.lane_selection_origin(
            work,
            FrontierFrameKind::Repair,
            GoalLaneEvidence::ActualSelectedLine,
        )?;
        selection.prefix = prefix.to_vec();
        selection.repaired = Some(prefix.to_vec());
        Ok(Some(selection))
    }

    pub(super) fn prepare_zero_step_repair_goal_observation(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        planned: RoleQuestion<'_>,
        attack: usize,
        counter_source: Option<ObservationId>,
    ) -> Result<Option<Box<AdaptiveLaneSelection>>, PalsError> {
        if self.rovent_frontier().is_none() {
            return Ok(None);
        }
        if planned.prefix.len() != self.continuation_limit() || planned.refutation.is_none() {
            return Err(RoleError::InvalidOutput.into());
        }
        let root = self.engine.semantic.nodes[self.root].state;
        let record = self
            .engine
            .semantic
            .records
            .iter()
            .rev()
            .find(|record| {
                record.origin_state == root
                    && record.kind == RecordKind::Repair
                    && record.line == planned.prefix
            })
            .ok_or(StoreError::InvalidEvidence(
                "zero-step Repair lacks its original record",
            ))?;
        let counter = self
            .engine
            .semantic
            .records
            .iter()
            .rev()
            .find(|record| {
                record.origin_state == root
                    && record.kind == RecordKind::Counterexample
                    && Some(record.line.as_slice()) == planned.refutation
            })
            .ok_or(StoreError::InvalidEvidence(
                "zero-step Repair lacks its original Counter record",
            ))?;
        let mut selection = self.lane_selection_origin(
            work,
            FrontierFrameKind::Repair,
            GoalLaneEvidence::ZeroStepRepairRecord {
                revision: record.revision,
                counter_revision: counter.revision,
                root,
            },
        )?;
        selection.prefix = planned.prefix.to_vec();
        selection.proposal = planned.proposal.to_vec();
        selection.counter = planned.refutation.map(<[BoardMove]>::to_vec);
        selection.repaired = Some(planned.prefix.to_vec());
        selection.attack = Some(attack);
        selection.source = counter_source;
        // begin_repair_continuation carries the original Counter line and
        // proposal, not a fabricated accepted P output.
        Ok(Some(selection))
    }

    /// Called only by after_cpu after its original observation was consumed.
    /// A reciprocal child can spend transferred role credit before that CPU
    /// result returns. Re-seal the actual parent budget, not an arbitrary
    /// permitted decrease: materialization still requires the CPU feedback
    /// gate, and the manager must claim this exact original/selected Work.
    pub(super) fn prepare_zero_step_repair_feedback_selection(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<Option<Box<AdaptiveLaneSelection>>, PalsError> {
        if work.input.stage != RefinementStage::RepairValidation {
            return Ok(None);
        }
        let Some(ActiveGoalLane::ZeroStepRepair { source, .. }) = self
            .goal_lanes
            .active
            .get(&(work.frame, FrontierFrameKind::Repair))
        else {
            return Ok(None);
        };
        let GoalLaneEvidence::ZeroStepRepairRecord {
            root,
            revision,
            counter_revision,
        } = source.evidence
        else {
            return Err(RoleError::InvalidOutput.into());
        };
        if source.frame != work.frame
            || source.first != work.original_first_move
            || source.deadline != work.original_deadline
            || source.budget.original_cpu_credit != work.budget.original_cpu_credit
            || source.budget.original_store_byte_limit != work.budget.original_store_byte_limit
            || work.budget.role_submissions_remaining > source.budget.role_submissions_remaining
            || !source.matches_line(&work.input)
            || root != self.engine.semantic.nodes[self.root].state
            || revision > self.engine.semantic.revision
            || counter_revision > self.engine.semantic.revision
        {
            return Err(RoleError::InvalidOutput.into());
        }
        self.lane_selection_origin(work, FrontierFrameKind::Repair, source.evidence)
            .map(Some)
    }

    pub(super) fn bind_goal_lane_selection(
        &mut self,
        mut selection: Box<AdaptiveLaneSelection>,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), PalsError> {
        if !selection.matches_selected(input)
            || self
                .goal_lanes
                .selected
                .contains_key(&input.question_digest)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if self.goal_lanes.selected.len() >= MAX_COORDINATOR_FRAMES {
            return Err(PalsError::Capacity);
        }
        self.goal_lanes
            .selected
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        selection.selected_question = Some(Arc::downgrade(&input.question));
        selection.selected_view = Some(Arc::downgrade(&input.view));
        self.goal_lanes.selected.insert(
            input.question_digest,
            SelectedGoalLane {
                source: Arc::from(selection),
            },
        );
        Ok(())
    }

    /// Called only after the manager committed this real selected continuation.
    /// The original immutable source in the private materialized recipe is
    /// required even when a sibling shares the same coordinator frame/kind.
    pub fn rovent_lane_selected(
        &mut self,
        original: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        selected: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<bool, PalsError> {
        let Some(held) = self
            .goal_lanes
            .selected
            .get(&selected.input.question_digest)
        else {
            return Ok(false);
        };
        if !held.source.validates_transition(original, selected) {
            return Err(RoleError::InvalidOutput.into());
        }
        let source = Arc::clone(&held.source);
        self.check_work(original)?;
        self.check_work(selected)?;
        let key = (selected.frame, source.kind);
        let previous = self
            .goal_lanes
            .active
            .get(&key)
            .and_then(ActiveGoalLane::previous_obligation);
        self.goal_lanes
            .active
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        let active = if let Some(kind) = source.obligation_kind() {
            let handle = self
                .with_rovent_frontier(|frontier, stores| match previous {
                    Some(previous) => frontier.register_local_lane_successor(
                        previous, &source, original, selected, stores, clock, now,
                    ),
                    None => frontier.register_local_frame(selected, stores, kind, clock, now),
                })?
                .ok_or(FrontierError::ForeignRoot)?;
            ActiveGoalLane::Obligation(handle)
        } else {
            self.with_rovent_frontier(|frontier, stores| {
                frontier
                    .observe_local_question(selected, stores, clock, now)
                    .map(|_| ())
            })?
            .ok_or(FrontierError::ForeignRoot)?;
            ActiveGoalLane::ZeroStepRepair { source, previous }
        };
        self.goal_lanes.active.insert(key, active);
        self.goal_lanes
            .selected
            .remove(&selected.input.question_digest);
        Ok(true)
    }

    fn current_goal_lane_obligation(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        kind: FrontierFrameKind,
        phase: GoalLaneBudgetPhase<'_>,
    ) -> Result<Option<FrontierObligation>, FrontierError> {
        match self.goal_lanes.active.get(&(work.frame, kind)) {
            Some(ActiveGoalLane::Obligation(handle)) => Ok(Some(*handle)),
            Some(ActiveGoalLane::ZeroStepRepair { source, .. }) => {
                let GoalLaneEvidence::ZeroStepRepairRecord {
                    revision,
                    counter_revision,
                    root,
                } = source.evidence
                else {
                    return Err(FrontierError::MissingAcceptance);
                };
                if root != self.engine.semantic.nodes[self.root].state
                    || revision > self.engine.semantic.revision
                    || counter_revision > self.engine.semantic.revision
                    || !source.permits_zero_step_observation(work, phase)
                {
                    return Err(FrontierError::MissingAcceptance);
                }
                // This typed state is different from a never-registered lane.
                // Do not look up an older Repair obligation by frame/kind.
                Ok(None)
            }
            None => self
                .rovent_frontier()
                .ok_or(FrontierError::ForeignRoot)?
                .adaptive_obligation(work.frame, kind),
        }
    }

    /// The configured generation chunk remains unchanged. A go's own depth
    /// or mate horizon can only shorten it; goal-local work keeps its separate
    /// controller-issued continuation beyond a shorter configured chunk.
    pub(super) fn original_goal_horizon(&self) -> Option<usize> {
        let goal = self.rovent_frontier()?.registration().goal;
        // The admitted goal contains both mate lanes: root win at 2N-1 and
        // root loss at 2N. Keep their maximum endpoint here; the original
        // Frontier still proves each lane and checks Rules terminals itself.
        // SearchGoal::validate already checked the conversion and capability.
        goal.depth_plies
            .map(|depth| usize::from(depth.get()))
            .into_iter()
            .chain(goal.mate_moves.map(|moves| usize::from(moves.get()) * 2))
            .min()
    }
    pub(super) fn continuation_limit(&self) -> usize {
        self.original_goal_horizon()
            .map_or(self.engine.config.line_plies, |horizon| {
                horizon.min(self.engine.config.line_plies)
            })
    }

    /// Replace only an unsubmitted validation exactly at the original goal
    /// endpoint with its real Fresh question. No zero-depth R job is issued.
    pub fn prepare_rovent_goal_boundary(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<Option<AdaptiveGoalBoundary<AdaptivePortQuestion>>, PalsError> {
        self.check_work(work)?;
        let query = &work.input.question;
        if !matches!(
            query.stage,
            RefinementStage::CpuValidation
                | RefinementStage::CounterValidation
                | RefinementStage::RepairValidation
                | RefinementStage::RecheckCounterValidation
        ) {
            return Ok(None);
        }
        let Some(target) = self.original_goal_horizon() else {
            return Ok(None);
        };
        if query.context.prefix.len() < target {
            return Ok(None);
        }
        if query.context.prefix.len() != target {
            return Err(FrontierError::DepthHorizon.into());
        }
        if self.engine.semantic.nodes[query.node].terminal.is_some() {
            // Existing Rules-terminal handling owns this endpoint instead.
            return Ok(None);
        }
        let checked = self.checked_frontier_line(&query.context.prefix)?;
        if !checked
            .endpoint_snapshot()
            .same_state(&query.position.snapshot())
        {
            return Err(FrontierError::ForeignRoot.into());
        }
        let root_budget = self
            .rovent_frontier()
            .ok_or(FrontierError::ForeignRoot)?
            .registration()
            .root_budget;
        let next = if query.stage == RefinementStage::RecheckCounterValidation {
            let (transition, rules_pair) = self.prepare_final_pair(
                work,
                query.repaired.clone().ok_or(RoleError::InvalidOutput)?,
                query.counter.clone().ok_or(RoleError::InvalidOutput)?,
            )?;
            if rules_pair.is_some() {
                return Err(RoleError::InvalidOutput.into());
            }
            let materialized = self.materialize_step(AdaptivePortStep {
                evidence: work.input.question_digest,
                transition,
                consumption_gate: None,
                feedback_gate: None,
                queue_question: None,
                fresh_pair: None,
                rules_pair: None,
            })?;
            let FrameTransition::Continue(next) = materialized.transition else {
                return Err(RoleError::InvalidOutput.into());
            };
            next
        } else {
            let stage = match query.stage {
                RefinementStage::CpuValidation => RefinementStage::ProposalValueFresh,
                RefinementStage::CounterValidation => RefinementStage::CounterValueFresh,
                RefinementStage::RepairValidation => RefinementStage::RepairValueFresh,
                _ => return Err(RoleError::InvalidOutput.into()),
            };
            let mut next = self.input(
                query.node,
                stage,
                query.first,
                query.context.prefix.clone(),
                query.proposal.clone(),
                query.counter.clone(),
                query.repaired.clone(),
                query.attack_ply,
                query.recheck,
                None,
                None,
            )?;
            Arc::make_mut(&mut next.question).counter_source = query.counter_source;
            next
        };
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-original-goal-boundary/1\0");
        hash.update(root_budget.0);
        hash.update(work.input.question_digest.0);
        hash.update(work.input.view_seal.view_digest.0);
        hash.update((target as u64).to_le_bytes());
        hash.update(next.question_digest.0);
        let boundary = AdaptiveGoalBoundary {
            seal: Box::new(GoalBoundarySeal {
                frame: work.frame,
                first: work.original_first_move,
                deadline: work.original_deadline,
                budget: work.budget,
                original: work.input.clone(),
                root_budget,
                target,
                evidence: ContractDigest(hash.finalize().into()),
                next,
            }),
        };
        boundary.validate(work)?;
        Ok(Some(boundary))
    }

    /// A concrete manager invokes this at the first actual admission of a
    /// selected lane. A later continuation cannot replace its original Q.
    pub fn rovent_frame_admitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        kind: FrontierFrameKind,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<bool, FrontierError> {
        if !self.rovent_goal_work_live(work)? {
            return Ok(false);
        }
        let key = (work.frame, kind);
        if matches!(
            self.goal_lanes.active.get(&key),
            Some(ActiveGoalLane::ZeroStepRepair { .. })
        ) {
            self.current_goal_lane_obligation(work, kind, GoalLaneBudgetPhase::Ready)?;
            self.with_rovent_frontier(|frontier, _| frontier.admit_role_input(clock, now))?
                .ok_or(FrontierError::ForeignRoot)?;
            return Ok(false);
        }
        let active = self
            .goal_lanes
            .active
            .get(&key)
            .and_then(ActiveGoalLane::previous_obligation);
        // Every active key is a subset of the Frontier's bounded original
        // responsibility journal. Reserve before that journal can mutate.
        self.goal_lanes
            .active
            .try_reserve(1)
            .map_err(|_| FrontierError::Capacity("active adaptive responsibility allocation"))?;
        let handle = self
            .with_rovent_frontier(|frontier, stores| {
                frontier.admit_role_input(clock, now)?;
                let original = match active {
                    Some(handle) => Some(handle),
                    None => frontier.adaptive_obligation(work.frame, kind)?,
                };
                if let Some(original) = original {
                    return if frontier.adaptive_obligation_source_matches(original, work, stores)? {
                        Ok(original)
                    } else {
                        Err(FrontierError::Invalid(
                            "responsibility retry differs from original question",
                        ))
                    };
                }
                frontier.register_local_frame(work, stores, kind, clock, now)
            })?
            .ok_or(FrontierError::ForeignRoot)?;
        self.goal_lanes
            .active
            .insert(key, ActiveGoalLane::Obligation(handle));
        Ok(true)
    }

    /// Only an actual retired native Fresh response can close its lane. A
    /// queued Repair may compare a counter endpoint without owning a new
    /// Counter responsibility; that observation does not invent one.
    #[allow(clippy::too_many_arguments)]
    pub fn rovent_fresh_accepted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<bool, PalsError> {
        if !self.rovent_goal_work_live(work)? {
            return Ok(false);
        }
        let kind = fresh_kind(work.input.stage).ok_or(FrontierError::MissingAcceptance)?;
        // Capacity and original physical/source validation precede head use.
        // The controller is the only place that may close its responsibility.
        let obligation = self.current_goal_lane_obligation(
            work,
            kind,
            GoalLaneBudgetPhase::AcceptedRole(accepted),
        )?;
        let Some(obligation) = obligation else {
            self.with_rovent_frontier(|frontier, stores| {
                frontier.observe_local_question(work, stores, clock, now)?;
                Ok(())
            })?;
            return Ok(false);
        };
        let permit = self.prepare_fresh_source_use(work, accepted, output, physical)?;
        self.with_rovent_frontier(|frontier, stores| {
            let endpoint = frontier.observe_local_question(work, stores, clock, now)?;
            // Shallow Fresh is real progress and consumption, while only the
            // controller's Closed disposition finishes the original duty.
            frontier.accept_local_fresh_horizon_once(
                obligation, work, stores, endpoint, accepted, output, physical, clock, now,
            )?;
            Ok(())
        })?
        .ok_or(FrontierError::ForeignRoot)?;
        if let Some(permit) = permit {
            self.record_used_fresh_source(permit)?;
        }
        Ok(true)
    }

    /// Rules terminal is exact evidence for this endpoint. CPU/NN values,
    /// missing alternatives and unfinished lines cannot enter this path.
    pub fn rovent_rules_endpoint(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<bool, FrontierError> {
        if !self.rovent_goal_work_live(work)? {
            return Ok(false);
        }
        if !matches!(
            work.input
                .question
                .position
                .classify_position()?
                .play_status,
            PlayStatus::Terminal { .. }
        ) {
            return Ok(false);
        }
        let kind = endpoint_kind(work.input.stage).ok_or(FrontierError::MissingAcceptance)?;
        let obligation =
            self.current_goal_lane_obligation(work, kind, GoalLaneBudgetPhase::Ready)?;
        self.with_rovent_frontier(|frontier, stores| {
            let endpoint = frontier.observe_local_question(work, stores, clock, now)?;
            let Some(obligation) = obligation else {
                return Ok(false);
            };
            frontier
                .accept_local_rules_endpoint_once(obligation, work, stores, endpoint, clock, now)?;
            Ok(true)
        })?
        .ok_or(FrontierError::ForeignRoot)
    }

    /// Called at the manager's explicit end of selected branch/recheck work,
    /// rather than inferred from a momentarily empty queue.
    pub fn rovent_root_round_sealed(
        &mut self,
        root_budget: ContractDigest,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<bool, FrontierError> {
        if self.rovent_frontier().is_none() {
            return Ok(false);
        }
        self.check_rovent_scan_live()?;
        self.with_rovent_frontier(|frontier, _| {
            if frontier.registration().root_budget != root_budget {
                return Err(FrontierError::ForeignRoot);
            }
            frontier.admit_role_input(clock, now)?;
            if !frontier.snapshot().selection_sealed {
                frontier.seal_selected_policy(clock, now)?;
            }
            Ok(true)
        })?
        .ok_or(FrontierError::ForeignRoot)
    }

    fn rovent_goal_work_live(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<bool, FrontierError> {
        if self.rovent_frontier().is_none() {
            return Ok(false);
        }
        self.check_rovent_scan_live()?;
        if work.frame.authority != self.authority
            || work.original_deadline != self.deadline
            || self.closed.contains(&work.frame)
        {
            return Err(FrontierError::ForeignRoot);
        }
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)
            .map_err(|_| FrontierError::MissingAcceptance)?;
        Ok(true)
    }
}

fn fresh_kind(stage: RefinementStage) -> Option<FrontierFrameKind> {
    match stage {
        RefinementStage::ProposalValueFresh => Some(FrontierFrameKind::Candidate),
        RefinementStage::CounterValueFresh | RefinementStage::FreshCounterEndpoint => {
            Some(FrontierFrameKind::Counter)
        }
        RefinementStage::RepairValueFresh
        | RefinementStage::FreshRepairedEndpoint
        | RefinementStage::RecheckRepairedValueFresh => Some(FrontierFrameKind::Repair),
        _ => None,
    }
}

fn endpoint_kind(stage: RefinementStage) -> Option<FrontierFrameKind> {
    fresh_kind(stage).or(match stage {
        RefinementStage::ProposalContinuation | RefinementStage::CpuValidation => {
            Some(FrontierFrameKind::Candidate)
        }
        RefinementStage::CriticReply
        | RefinementStage::CounterContinuation
        | RefinementStage::CounterValidation
        | RefinementStage::CriticRecheck
        | RefinementStage::RecheckCounterValidation => Some(FrontierFrameKind::Counter),
        RefinementStage::Repair | RefinementStage::RepairValidation => {
            Some(FrontierFrameKind::Repair)
        }
        _ => None,
    })
}
