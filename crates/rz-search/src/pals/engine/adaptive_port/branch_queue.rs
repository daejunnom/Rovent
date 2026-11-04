//! Bounded independent CPU/C branches and iterative Repair frontiers.
//! Queue entries are exact Rules lines, never completed scores or new credit.
use super::*;

#[cfg(test)]
mod queue_boundary_tests;

const MAX_PENDING_COUNTERS: usize = 64;
pub(super) struct DeferredRecheck {
    parent: crate::pals::coordinator::FrameId,
    original_question: ContractDigest,
    first: BoardMove,
    repaired: Vec<BoardMove>,
    blueprint: AdaptiveContinuationBlueprint,
    parent_budget: FrameBudget,
    original_view: ReviewViewSeal,
    public_context: ContractDigest,
    feedback: Option<(ContractDigest, ObservationId)>,
    sealed: bool,
}
/// Issued only after an actual queued Repair's semantic commit and original
/// consumption gates succeed. It is separate from parent closure: a parent
/// can continue another counter before its round frontier closes.
pub struct AdaptiveQueuedRecheck {
    parent: crate::pals::coordinator::FrameId,
    question: ContractDigest,
    first: rz_contracts::pals::Move16,
    budget: FrameBudget,
    deadline: Deadline,
    original_view: ReviewViewSeal,
    public_context: ContractDigest,
    root: Arc<ContractState>,
    evidence: ContractDigest,
}
impl AdaptiveQueuedRecheck {
    pub fn parent(&self) -> crate::pals::coordinator::FrameId {
        self.parent
    }
    pub fn original_question(&self) -> ContractDigest {
        self.question
    }
    pub fn first_move(&self) -> rz_contracts::pals::Move16 {
        self.first
    }
    pub fn budget(&self) -> FrameBudget {
        self.budget
    }
    pub fn original_deadline(&self) -> Deadline {
        self.deadline
    }
    pub fn original_view(&self) -> ReviewViewSeal {
        self.original_view
    }
    pub fn public_context(&self) -> ContractDigest {
        self.public_context
    }
    pub fn root_state(&self) -> rz_contracts::StateIdentity {
        self.root.snapshot().identity()
    }
    pub fn evidence(&self) -> ContractDigest {
        self.evidence
    }
    pub fn validate_work<Q>(&self, work: &CoordinatorWork<ReviewView, Q>) -> Result<(), PalsError> {
        if work.frame != self.parent
            || work.input.question_digest != self.question
            || work.original_first_move != self.first
            || work.original_deadline != self.deadline
            || work.budget != self.budget
            || work.input.view_seal != self.original_view
            || work.input.public_context != self.public_context
            || self.original_view.authority != self.parent.authority
            || self.original_view.root_state != self.root.snapshot().identity()
            || self.original_view.view_digest.0 != work.input.view.seal_digest()
            || self.public_context.0 != work.input.view.record_scope_digest()
            || work.input.view.anchors().is_empty()
            || !work.input.view.anchors()[0]
                .snapshot
                .same_state(self.root.rules().snapshot())
            || !matches!(
                work.input.stage,
                RefinementStage::RepairValidation | RefinementStage::RepairValueFresh
            )
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }
}
/// A private port-issued queue admission. The coordinator must verify this
/// producer proof; public FrameInput cannot claim deferred Repair authority.
pub struct AdaptiveDeferredRepair<Q> {
    parent: crate::pals::coordinator::FrameId,
    original_question: ContractDigest,
    original_view: ReviewViewSeal,
    original_public_context: ContractDigest,
    first: rz_contracts::pals::Move16,
    input: FrameInput<ReviewView, Q>,
    budget: FrameBudget,
    deadline: Deadline,
    root: Arc<ContractState>,
    repaired: Vec<BoardMove>,
}
impl<Q> AdaptiveDeferredRepair<Q> {
    pub fn parent(&self) -> crate::pals::coordinator::FrameId {
        self.parent
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
    pub fn original_deadline(&self) -> Deadline {
        self.deadline
    }
    pub fn original_question(&self) -> ContractDigest {
        self.original_question
    }
    pub fn original_view(&self) -> ReviewViewSeal {
        self.original_view
    }
    pub fn original_public_context(&self) -> ContractDigest {
        self.original_public_context
    }
    pub fn validate(&self) -> Result<(), PalsError> {
        let first = BoardMove::try_from(self.first.decode().map_err(|_| RoleError::InvalidOutput)?)
            .map_err(|_| RoleError::InvalidOutput)?;
        let query = self.input.view.query();
        if self.parent.authority != self.input.view_seal.authority
            || self.parent.sequence == 0
            || self.original_view.authority != self.parent.authority
            || self.original_view.root_state != self.root.snapshot().identity()
            || self.original_view.context_policy != self.input.view_seal.context_policy
            || self.deadline.clock.0 != self.parent.authority.epoch
            || self.input.stage != RefinementStage::RecheckRepairedValueFresh
            || query.purpose != RoleQueryPurpose::ValueFresh
            || query.prefix != self.repaired
            || query.proposal != self.repaired
            || query.counterexample.is_some()
            || self.repaired.first() != Some(&first)
            || self.input.view.anchors().is_empty()
            || !self.input.view.anchors()[0]
                .snapshot
                .same_state(self.root.rules().snapshot())
            || self.input.view_seal.root_state != self.root.snapshot().identity()
            || self.input.view_seal.view_digest.0 != self.input.view.seal_digest()
            || self.input.public_context.0 != self.input.view.record_scope_digest()
            || self.input.view_seal.context_policy.0 != self.input.view.policy_digest()
            || self.input.view_seal.view_sequence != self.input.view.sequence()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let root = Position::from_snapshot(
            self.root.rules().snapshot(),
            rz_position::PositionLimits::default(),
        )?;
        if !root.legal_moves().contains(&first) {
            return Err(ReviewError::AnchorMismatch.into());
        }
        let mut endpoint = root;
        for movement in &self.repaired {
            endpoint.make_move(*movement)?;
        }
        if query.legal != endpoint.legal_moves() {
            return Err(ReviewError::AnchorMismatch.into());
        }
        Ok(())
    }
    pub fn into_parts(
        self,
    ) -> (
        crate::pals::coordinator::FrameId,
        rz_contracts::pals::Move16,
        FrameInput<ReviewView, Q>,
        FrameBudget,
    ) {
        (self.parent, self.first, self.input, self.budget)
    }
}
#[derive(Clone)]
pub(super) struct PendingCounter {
    first: BoardMove,
    proposal: Vec<BoardMove>,
    counter: Vec<BoardMove>,
    attack: usize,
    source: Option<ObservationId>,
    repair_direct: bool,
}
#[cfg(test)]
impl PendingCounter {
    /// Authored queue metadata only; this grants no accepted observation.
    pub(super) fn authored_without_observation_for_test(
        first: BoardMove,
        proposal: Vec<BoardMove>,
        counter: Vec<BoardMove>,
        attack: usize,
    ) -> Self {
        Self {
            first,
            proposal,
            counter,
            attack,
            source: None,
            repair_direct: false,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub enum AdaptiveBranchOutcome {
    Complete(FrameCompletionScope),
    Unknown(&'static str),
}
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveBranchReceipt {
    pub first_move: BoardMove,
    pub submitted_question: ContractDigest,
    pub public_context: ContractDigest,
    pub outcome: AdaptiveBranchOutcome,
}
impl<M: RoleModel> AdaptiveEnginePort<M> {
    pub fn branch_receipts(&self) -> &[AdaptiveBranchReceipt] {
        &self.branch_receipts
    }

    pub(super) fn defer_fixed_recheck(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        mut blueprint: AdaptiveContinuationBlueprint,
    ) -> Result<AdaptivePortTransition, PalsError> {
        self.check_work(work)?;
        if self.root_fanout != AdaptiveRootFanoutPolicy::SequentialFixed
            || self.engine.post_repair_recheck != PostRepairRecheckPolicy::IterativeFrozenModelWdlV2
            || work.frame.authority != self.authority
            || !matches!(
                work.input.stage,
                RefinementStage::RepairValidation | RefinementStage::RepairValueFresh
            )
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if self.deferred_rechecks.len()
            + self
                .pending_counters
                .values()
                .map(VecDeque::len)
                .sum::<usize>()
            >= MAX_PENDING_COUNTERS
        {
            return Err(PalsError::Capacity);
        }
        let (first, repaired) = blueprint.queued_recheck_identity()?;
        if rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(first).map_err(|_| RoleError::InvalidOutput)?,
        ) != work.original_first_move
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if !self.engine.semantic.records.iter().any(|record| {
            record.origin_state == self.engine.semantic.nodes[self.root].state
                && record.kind == RecordKind::Repair
                && record.line == repaired
        }) {
            return Err(
                StoreError::InvalidEvidence("deferred recheck lacks actual Repair record").into(),
            );
        }
        if self.deferred_rechecks.iter().any(|entry| {
            entry.parent == work.frame && entry.original_question == work.input.question_digest
        }) {
            return Err(StoreError::InvalidEvidence("duplicate deferred recheck").into());
        }
        let queue_enqueued = self
            .counters
            .repair_queue_enqueued
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        blueprint.carry_diagnostic_repair_line(work.input.question.diagnostic_repair_line);
        self.deferred_rechecks.push_back(DeferredRecheck {
            parent: work.frame,
            original_question: work.input.question_digest,
            first,
            repaired,
            blueprint,
            parent_budget: work.budget,
            original_view: work.input.view_seal,
            public_context: work.input.public_context,
            feedback: None,
            sealed: false,
        });
        self.counters.repair_queue_enqueued = queue_enqueued;
        self.counters.repair_queue_peak = self
            .counters
            .repair_queue_peak
            .max(self.deferred_rechecks.len());
        Ok(AdaptivePortTransition::Complete {
            scope: FrameCompletionScope::QueuedRecheck,
        })
    }

    pub(super) fn bind_deferred_recheck_feedback_cause(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        observation: ObservationId,
    ) -> Result<(), PalsError> {
        if !self.cpu_observation_consumed(work, observation) {
            return Err(StoreError::StaleConsumer.into());
        }
        for entry in &mut self.deferred_rechecks {
            if entry.parent == work.frame && entry.original_question == work.input.question_digest {
                if entry.feedback.is_some() || entry.sealed {
                    return Err(RoleError::InvalidOutput.into());
                }
                entry.feedback = Some((work.input.question_digest, observation));
            }
        }
        Ok(())
    }

    pub(super) fn seal_queued_recheck(
        &mut self,
        queued_question: Option<(crate::pals::coordinator::FrameId, ContractDigest)>,
        evidence: ContractDigest,
    ) -> Result<Option<AdaptiveQueuedRecheck>, PalsError> {
        let Some((parent, question)) = queued_question else {
            return Ok(None);
        };
        let Some(index) = self
            .deferred_rechecks
            .iter()
            .position(|entry| entry.parent == parent && entry.original_question == question)
        else {
            return Ok(None);
        };
        self.check_live()?;
        let entry = &self.deferred_rechecks[index];
        if entry.sealed {
            return Err(RoleError::InvalidOutput.into());
        }
        self.validate_blueprint(&entry.blueprint)?;
        if entry.feedback.is_some_and(|(original, observation)| {
            !self.cpu_feedback_still_authorized(original, observation)
        }) {
            return Err(StoreError::StaleConsumer.into());
        }
        let proof = AdaptiveQueuedRecheck {
            parent: entry.parent,
            question,
            first: rz_contracts::pals::Move16::encode(
                rz_contracts::Move::try_from(entry.first).map_err(|_| RoleError::InvalidOutput)?,
            ),
            budget: entry.parent_budget,
            deadline: self.deadline,
            original_view: entry.original_view,
            public_context: entry.public_context,
            root: Arc::clone(&self.rules),
            evidence,
        };
        self.queued_recheck_closures
            .insert((entry.parent, question), entry.first);
        self.deferred_rechecks[index].sealed = true;
        Ok(Some(proof))
    }

    pub(super) fn bind_deferred_recheck_role_cause(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), PalsError> {
        for entry in &mut self.deferred_rechecks {
            if entry.parent == work.frame && entry.original_question == work.input.question_digest {
                entry.blueprint.bind_role_cause(work, ticket)?;
            }
        }
        Ok(())
    }

    /// Called only after the main round frontier is closed. Selecting the view
    /// here gives queued rechecks their actual latest public context without
    /// refreshing any previously submitted role ticket.
    pub fn next_fixed_iterative_repair(
        &mut self,
    ) -> Result<Option<AdaptiveDeferredRepair<AdaptivePortQuestion>>, PalsError> {
        self.check_live()?;
        if self.root_fanout != AdaptiveRootFanoutPolicy::SequentialFixed
            || self.root_ranking_pending
            || !self.lazy_frontier.is_empty()
            || self.round_candidates.is_empty()
            || self
                .round_candidates
                .values()
                .any(|(_, frame)| frame.is_none_or(|id| !self.closed.contains(&id)))
        {
            return Ok(None);
        }
        let Some(entry) = self.deferred_rechecks.pop_front() else {
            return Ok(None);
        };
        if !entry.sealed
            || !self.closed.contains(&entry.parent)
            || self
                .queued_recheck_closures
                .get(&(entry.parent, entry.original_question))
                != Some(&entry.first)
            || entry.feedback.is_some_and(|(question, observation)| {
                !self.cpu_deferred_feedback_still_authorized(entry.parent, question, observation)
            })
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let step = self.materialize_step(AdaptivePortStep {
            evidence: entry.original_question,
            transition: AdaptivePortTransition::Continue(entry.blueprint),
            consumption_gate: None,
            feedback_gate: None,
            queue_question: None,
            fresh_pair: None,
            rules_pair: None,
        })?;
        let FrameTransition::Continue(input) = step.transition else {
            return Err(RoleError::InvalidOutput.into());
        };
        let proof = AdaptiveDeferredRepair {
            parent: entry.parent,
            original_question: entry.original_question,
            original_view: entry.original_view,
            original_public_context: entry.public_context,
            first: rz_contracts::pals::Move16::encode(
                rz_contracts::Move::try_from(entry.first).map_err(|_| RoleError::InvalidOutput)?,
            ),
            input,
            budget: FrameBudget {
                original_cpu_credit: entry.parent_budget.original_cpu_credit,
                role_submissions_remaining: entry.parent_budget.role_submissions_remaining.min(
                    self.engine
                        .config
                        .max_role_calls
                        .saturating_sub(self.counters.role_calls),
                ),
                original_store_byte_limit: entry.parent_budget.original_store_byte_limit,
            },
            deadline: self.deadline,
            root: Arc::clone(&self.rules),
            repaired: entry.repaired,
        };
        proof.validate()?;
        Ok(Some(proof))
    }

    pub(super) fn pending_counter(
        query: &AdaptivePortQuestion,
        counter: Vec<BoardMove>,
        source: Option<ObservationId>,
    ) -> Result<PendingCounter, PalsError> {
        Ok(PendingCounter {
            first: query.first.ok_or(RoleError::InvalidOutput)?,
            proposal: query.proposal.clone(),
            counter,
            attack: query.attack_ply.ok_or(RoleError::InvalidOutput)?,
            source,
            repair_direct: false,
        })
    }
    pub(super) fn enqueue_counter(&mut self, pending: PendingCounter) -> Result<bool, PalsError> {
        if pending.proposal.first() != Some(&pending.first)
            || pending.counter.first() != Some(&pending.first)
            || pending.counter.len() > self.continuation_limit()
            || pending.proposal.len() > self.continuation_limit()
            || pending.attack >= pending.counter.len()
            || pending.attack >= pending.proposal.len()
            || pending.counter.get(..pending.attack) != pending.proposal.get(..pending.attack)
            || pending.counter[pending.attack] == pending.proposal[pending.attack]
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if self.rovent_frontier().is_some() {
            // A queued alternative has not been selected for examination yet.
            // Reserve/check its Rules path without publishing hot graph edges;
            // begin_counter adopts those snapshots at the actual selection.
            if self
                .existing_checked_frontier_line(&pending.counter)?
                .is_none()
            {
                self.checked_frontier_line(&pending.counter)?;
            }
        } else {
            let mut checked = self.engine.semantic.nodes[self.root].position.clone();
            for movement in &pending.counter {
                checked.make_move(*movement)?;
            }
        }
        let mut hash = Sha256::new();
        hash.update(b"rz-pals-pending-counter/1\0");
        hash.update(self.authority.epoch.0.to_le_bytes());
        hash.update(self.authority.game.0.to_le_bytes());
        hash.update(self.authority.root.0.to_le_bytes());
        hash.update(self.engine.semantic.revision.to_le_bytes());
        hash.update(role_line_sha256(b"proposal", &pending.proposal)?);
        hash.update(role_line_sha256(b"counter", &pending.counter)?);
        hash.update([u8::from(pending.repair_direct)]);
        let digest = ContractDigest(hash.finalize().into());
        if self.queued_questions.contains(&digest) {
            return Ok(false);
        }
        if self
            .pending_counters
            .values()
            .map(VecDeque::len)
            .sum::<usize>()
            >= MAX_PENDING_COUNTERS
            || self.deferred_rechecks.len()
                + self
                    .pending_counters
                    .values()
                    .map(VecDeque::len)
                    .sum::<usize>()
                >= MAX_PENDING_COUNTERS
            || self.queued_questions.len() >= MAX_PENDING_COUNTERS
        {
            return Err(PalsError::Capacity);
        }
        self.queued_questions.insert(digest);
        self.pending_counters
            .entry(pending.first)
            .or_default()
            .push_back(pending);
        Ok(true)
    }
    pub(super) fn begin_counter(
        &mut self,
        original: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        pending: PendingCounter,
    ) -> Result<AdaptivePortTransition, PalsError> {
        let node = self.replay_node(&pending.counter)?;
        if pending.repair_direct {
            if !self.engine.admit_iterative_repair(pending.first) {
                return Ok(AdaptivePortTransition::Complete {
                    scope: FrameCompletionScope::RepairNotAdmitted,
                });
            }
            let prefix = pending
                .counter
                .get(..=pending.attack)
                .ok_or(RoleError::InvalidOutput)?
                .to_vec();
            let anchor = self.replay_node(&prefix)?;
            let mut transition = self.begin_repair_continuation(
                anchor,
                Some(pending.first),
                prefix,
                pending.proposal,
                pending.counter,
                pending.attack,
                pending.source,
            )?;
            self.bind_zero_step_repair_goal_transition(original, &mut transition)?;
            return Ok(transition);
        }
        let goal_selection = self.prepare_counter_goal_selection(
            original,
            &pending.counter,
            &pending.proposal,
            pending.attack,
            pending.source,
        )?;
        let done = pending.counter.len() >= self.continuation_limit()
            || self.engine.semantic.nodes[node].terminal.is_some();
        let stage = if done {
            self.record_hot(RecordKind::Counterexample, &pending.counter, pending.source)?;
            self.counters.refutations += 1;
            RefinementStage::CounterValidation
        } else {
            RefinementStage::CounterContinuation
        };
        let next = self.blueprint(
            node,
            stage,
            Some(pending.first),
            pending.counter.clone(),
            pending.proposal,
            Some(pending.counter),
            None,
            Some(pending.attack),
            false,
            None,
            None,
        )?;
        Ok(AdaptivePortTransition::Continue(
            next.with_counter_source(pending.source)
                .with_goal_lane_selection(goal_selection)?,
        ))
    }
    /// Preserve each branch outcome even when a sibling continues under the
    /// same first move. A later success cannot erase an earlier unresolved line.
    pub(super) fn finish_branch(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        mut transition: AdaptivePortTransition,
    ) -> Result<AdaptivePortTransition, PalsError> {
        // Only the same continuing branch inherits the immutable provenance.
        // Completed branches selecting a queued sibling below do not transfer it.
        // A new Repair (including a zero-step one) must not inherit the
        // preceding Repair's identity from CounterValidation/CounterValueFresh.
        if matches!(
            work.input.stage,
            RefinementStage::RepairValidation
                | RefinementStage::RepairValueFresh
                | RefinementStage::RecheckRepairedValueFresh
                | RefinementStage::CriticRecheck
                | RefinementStage::RecheckCounterValidation
                | RefinementStage::FreshRepairedEndpoint
                | RefinementStage::FreshCounterEndpoint
        ) {
            transition.carry_diagnostic_repair_line(work.input.question.diagnostic_repair_line);
        }
        let outcome = match &transition {
            AdaptivePortTransition::Continue(_) => return Ok(transition),
            AdaptivePortTransition::Complete { scope } => AdaptiveBranchOutcome::Complete(*scope),
            AdaptivePortTransition::Unknown { reason } => AdaptiveBranchOutcome::Unknown(reason),
        };
        let first = work.input.question.first.ok_or(RoleError::InvalidOutput)?;
        let queued_recheck = matches!(
            transition,
            AdaptivePortTransition::Complete {
                scope: FrameCompletionScope::QueuedRecheck
            }
        ) && self.root_fanout == AdaptiveRootFanoutPolicy::SequentialFixed
            && self.engine.post_repair_recheck
                == PostRepairRecheckPolicy::IterativeFrozenModelWdlV2
            && matches!(
                work.input.stage,
                RefinementStage::RepairValidation | RefinementStage::RepairValueFresh
            )
            && work.original_deadline == self.deadline
            && self.deferred_rechecks.iter().any(|entry| {
                !entry.sealed
                    && entry.parent == work.frame
                    && entry.original_question == work.input.question_digest
                    && entry.first == first
                    && rz_contracts::Move::try_from(entry.first).is_ok_and(|movement| {
                        rz_contracts::pals::Move16::encode(movement) == work.original_first_move
                    })
                    && entry.parent_budget == work.budget
                    && entry.original_view == work.input.view_seal
                    && entry.public_context == work.input.public_context
            });
        if queued_recheck {
            self.check_work(work)?;
        }
        if self.branch_receipts.len() >= MAX_PENDING_COUNTERS {
            return Err(PalsError::Capacity);
        }
        self.branch_receipts.push(AdaptiveBranchReceipt {
            first_move: first,
            submitted_question: work.input.question_digest,
            public_context: work.input.public_context,
            outcome,
        });
        if let AdaptiveBranchOutcome::Unknown(reason) = outcome {
            self.branch_issues.entry(first).or_insert(reason);
        }
        let next = self
            .pending_counters
            .get_mut(&first)
            .and_then(VecDeque::pop_front);
        if let Some(next) = next {
            return self.begin_counter(work, next);
        }
        // Queuing the exact accepted Repair is an administrative frontier
        // closure. Earlier Unknown receipts still describe unresolved sibling
        // lines, but must not replace the actual private queue transition.
        // A public completion scope alone grants no such exception.
        if queued_recheck {
            return Ok(transition);
        }
        if let Some(reason) = self.branch_issues.get(&first) {
            return Ok(AdaptivePortTransition::Unknown { reason });
        }
        Ok(transition)
    }
    /// Only a newly accepted strict conditional refutation can schedule another
    /// Repair. The existing per-first-move cap is charged at actual admission.
    pub(super) fn queue_strict_repair(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        counter: Vec<BoardMove>,
    ) -> Result<(), PalsError> {
        if self.engine.post_repair_recheck != PostRepairRecheckPolicy::IterativeFrozenModelWdlV2 {
            return Ok(());
        }
        let query = &work.input.question;
        let proposal = query
            .repaired
            .as_ref()
            .ok_or(RoleError::InvalidOutput)?
            .clone();
        if proposal == counter {
            return Ok(());
        }
        let attack = proposal
            .iter()
            .zip(&counter)
            .position(|(a, b)| a != b)
            .ok_or(RoleError::InvalidOutput)?;
        let position = if self.rovent_frontier().is_some() {
            let node = self.replay_node(&proposal[..attack])?;
            self.engine.semantic.nodes[node].position.clone()
        } else {
            let mut position = self.engine.semantic.nodes[self.root].position.clone();
            for movement in &proposal[..attack] {
                position.make_move(*movement)?;
            }
            position
        };
        if position.side_to_move()
            == self.engine.semantic.nodes[self.root]
                .position
                .side_to_move()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        self.enqueue_counter(PendingCounter {
            first: query.first.ok_or(RoleError::InvalidOutput)?,
            proposal,
            counter,
            attack,
            source: None,
            repair_direct: true,
        })?;
        Ok(())
    }
}
