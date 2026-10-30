//! Owner-private materialized causes. These attest a semantic successor only;
//! no CPU grant, role submission, deadline or completion is issued here.
use super::*;
use crate::pals::store::StoreReadOwner;
use rz_contracts::pals_adaptive::CpuJobEnvelope;
use rz_contracts::pals_adaptive::CpuJobId;
use rz_contracts::rovent::QuestionEnvelope;
use std::sync::Weak;

const MAX_UTILITY_CONTINUATIONS: usize = crate::pals::coordinator::MAX_COORDINATOR_FRAMES;
const COHORT_LINE_DOMAIN: &[u8] = b"rz-rovent-utility-cohort-ordered-line/1";

/// A comparison-only source minted from this port's checked immutable input.
/// The manager separately attests that this exact source is currently Ready.
/// Matching content cannot admit a child, refill a budget or revive an offer.
/// Weak input references add no selected-view pin or model/CPU owner lifetime.
pub struct UtilityCohortOfferSemanticSource<Q = AdaptivePortQuestion> {
    semantic: ContractDigest,
    owner: StoreReadOwner,
    generation: u64,
    root: StateId,
    frame: FrameId,
    first: rz_contracts::pals::Move16,
    deadline: Deadline,
    budget: FrameBudget,
    stage: RefinementStage,
    question: ContractDigest,
    view: ReviewViewSeal,
    full_view: ContractDigest,
    public_context: ContractDigest,
    query: Weak<Q>,
    selected: Weak<ReviewView>,
}
impl<Q> Clone for UtilityCohortOfferSemanticSource<Q> {
    fn clone(&self) -> Self {
        Self {
            semantic: self.semantic,
            owner: self.owner.clone(),
            generation: self.generation,
            root: self.root,
            frame: self.frame,
            first: self.first,
            deadline: self.deadline,
            budget: self.budget,
            stage: self.stage,
            question: self.question,
            view: self.view,
            full_view: self.full_view,
            public_context: self.public_context,
            query: self.query.clone(),
            selected: self.selected.clone(),
        }
    }
}
impl<Q> UtilityCohortOfferSemanticSource<Q> {
    pub fn digest(&self) -> ContractDigest {
        self.semantic
    }

    /// Original identity only; current scheduling/admission is still checked
    /// by the owner. A cloned equal Q or a rewritten budget is not this offer.
    pub fn matches_work(&self, work: &CoordinatorWork<ReviewView, Q>) -> bool {
        self.frame == work.frame
            && self.first == work.original_first_move
            && self.deadline == work.original_deadline
            && self.budget == work.budget
            && self.stage == work.input.stage
            && self.question == work.input.question_digest
            && self.view == work.input.view_seal
            && self.full_view == ContractDigest(work.input.view.digest())
            && self.public_context == work.input.public_context
            && self.query.ptr_eq(&Arc::downgrade(&work.input.question))
            && self.selected.ptr_eq(&Arc::downgrade(&work.input.view))
    }

    /// Cleanup identity check, including after deadline/cancel. This neither
    /// claims pin-zero/physical retirement nor reopens logical authority.
    pub fn matches_store(&self, stores: &PalsStores) -> bool {
        self.owner.matches(stores)
            && self.generation == stores.generation()
            && self.generation == self.frame.authority.root.0
            && stores.root().is_some_and(|root| {
                stores
                    .situations
                    .get(root)
                    .is_ok_and(|situation| situation.state == self.root)
            })
    }
}

#[derive(Clone, Copy)]
pub(super) enum UtilityContinuationCause {
    Role {
        request: RequestId,
        frame: FrameId,
        question: ContractDigest,
        view: ContractDigest,
    },
    Cpu {
        job: CpuJobId,
        execution: rz_contracts::ExecutionId,
        frame: FrameId,
        question: ContractDigest,
        observation: ObservationId,
    },
}

impl UtilityContinuationCause {
    fn frame(self) -> FrameId {
        match self {
            Self::Role { frame, .. } | Self::Cpu { frame, .. } => frame,
        }
    }
}

/// Only materialization of a checked private blueprint can mint this source.
/// Weak references do not keep an old query, output tensor or Store lease live.
#[derive(Clone)]
pub struct UtilityContinuationSource {
    owner: StoreReadOwner,
    generation: u64,
    root: StateId,
    authority: SearchAuthority,
    deadline: Deadline,
    hard_deadline: Instant,
    cause: UtilityContinuationCause,
    target_query: Weak<AdaptivePortQuestion>,
    target_view: Weak<ReviewView>,
    stage: RefinementStage,
    question: ContractDigest,
    view: ReviewViewSeal,
    full_view: ContractDigest,
    public_context: ContractDigest,
    first: rz_contracts::pals::Move16,
}
impl UtilityContinuationSource {
    pub fn role_source(&self) -> Option<(RequestId, FrameId, ContractDigest, ContractDigest)> {
        match self.cause {
            UtilityContinuationCause::Role {
                request,
                frame,
                question,
                view,
            } => Some((request, frame, question, view)),
            _ => None,
        }
    }
    pub fn cpu_source(
        &self,
    ) -> Option<(
        CpuJobId,
        rz_contracts::ExecutionId,
        FrameId,
        ContractDigest,
        ObservationId,
    )> {
        match self.cause {
            UtilityContinuationCause::Cpu {
                job,
                execution,
                frame,
                question,
                observation,
            } => Some((job, execution, frame, question, observation)),
            _ => None,
        }
    }
}

/// An actually settled suspension, not a raw publication or completed search.
/// It exists only while the original Store task and same CPU owner are paused.
pub(super) struct UtilityReciprocalAwaitSource {
    job: CpuJobEnvelope,
    execution: rz_contracts::ExecutionId,
    parent: QuestionEnvelope,
}
impl UtilityReciprocalAwaitSource {
    pub(super) fn job(&self) -> CpuJobEnvelope {
        self.job
    }
    pub(super) fn execution(&self) -> rz_contracts::ExecutionId {
        self.execution
    }
    pub(super) fn parent(&self) -> QuestionEnvelope {
        self.parent
    }
}

#[derive(Default)]
pub(super) struct UtilityContinuationRegistry {
    enabled: bool,
    witness_payload_limit: Option<usize>,
    entries: HashMap<ContractDigest, UtilityContinuationSource>,
}
impl UtilityContinuationRegistry {
    #[cfg(test)]
    pub(super) fn enable(&mut self) {
        self.enable_bounded_payload(super::utility_route::MAX_UTILITY_WITNESS_BODY_BYTES as u64);
    }
    pub(super) fn enable_bounded_payload(&mut self, original_store_limit: u64) {
        let requested = usize::try_from(original_store_limit)
            .unwrap_or(usize::MAX)
            .min(super::utility_route::MAX_UTILITY_WITNESS_BODY_BYTES);
        self.witness_payload_limit = Some(
            self.witness_payload_limit
                .map_or(requested, |original| original.min(requested)),
        );
        self.enabled = true;
    }
    pub(super) fn witness_payload_limit(&self) -> usize {
        self.witness_payload_limit.unwrap_or(0)
    }
    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }
    pub(super) fn reserve(&mut self) -> Result<(), PalsError> {
        // Expired weak metadata has no historical packet, view lease or live
        // continuation authority. Actual action history belongs to its owner.
        self.entries.retain(|_, source| {
            source.target_query.strong_count() != 0 && source.target_view.strong_count() != 0
        });
        if self.entries.len() >= MAX_UTILITY_CONTINUATIONS {
            return Err(PalsError::Capacity);
        }
        self.entries.try_reserve(1).map_err(|_| PalsError::Capacity)
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Freeze comparison semantics only after validating this owner's actual
    /// immutable submission. Ready/idle membership and the complete offer set
    /// are attested separately by the manager; no raw digest can mint this cap.
    pub fn utility_cohort_offer_semantics(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<UtilityCohortOfferSemanticSource, PalsError> {
        self.check_work(work)?;
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        if work.input.stage.role() == Some(Role::Validator) {
            return Err(RoleError::Unsupported("utility cohort has no validator offer").into());
        }
        let frontier = self.rovent_frontier().ok_or(RoleError::Unavailable)?;
        let registration = frontier.registration();
        let namespace = self.model_namespace()?;
        frontier.validate_attachment(
            crate::pals::rovent_frontier::FrontierAttachment {
                authority: self.authority,
                mode: registration.mode,
                original_deadline: self.deadline,
                root_budget: registration.root_budget,
                root_rules: &self.rules,
                root_moves: frontier.root_moves(),
                root_state: self.root_state_id(),
                model_namespace: namespace,
                native_ledger: registration.native_ledger.as_ref(),
            },
            &self.engine.semantic.stores,
        )?;
        if registration.native_ledger.is_none() {
            return Err(RoleError::Unsupported(
                "utility cohort requires the original native ledger",
            )
            .into());
        }
        let q = &work.input.question;
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-utility-cohort-ready-offer/2\0");
        hash.update(self.rules.snapshot().identity().semantic.0);
        hash.update(
            rz_position::contracts::rules_snapshot_semantic_digest(&q.position.snapshot()).0,
        );
        hash.update([work.input.stage as u8, q.context.purpose as u8]);
        hash.update(work.original_first_move.bits().to_le_bytes());
        hash.update(work.budget.role_submissions_remaining.to_le_bytes());
        hash.update(work.budget.original_cpu_credit.to_le_bytes());
        hash.update(work.budget.original_store_byte_limit.to_le_bytes());
        // Input digest excludes view issuance. The existing V2 canonical
        // input conservatively retains its record metadata; no old bytes or
        // model feature semantics are normalized by this comparison path.
        self.utility_cohort_write_view(&mut hash, &work.input.view)?;
        for line in [&q.context.prefix, &q.proposal] {
            hash.update(role_line_sha256(COHORT_LINE_DOMAIN, line)?);
        }
        for line in [&q.counter, &q.repaired, &q.validator_root_moves] {
            hash.update([u8::from(line.is_some())]);
            if let Some(line) = line {
                hash.update(role_line_sha256(COHORT_LINE_DOMAIN, line)?);
            }
        }
        hash.update([
            u8::from(q.attack_ply.is_some()),
            u8::from(q.discovery_restricted),
            u8::from(q.recheck),
        ]);
        if let Some(ply) = q.attack_ply {
            hash.update((ply as u64).to_le_bytes());
        }
        for observation in [q.counter_source, q.paired_repaired] {
            hash.update([u8::from(observation.is_some())]);
            if let Some(observation) = observation {
                self.validator_write_observation(&mut hash, observation, 0)?;
            }
        }
        hash.update([u8::from(q.frozen_counter.is_some())]);
        if let Some(view) = &q.frozen_counter {
            self.utility_cohort_write_view(&mut hash, view)?;
        }
        hash.update([u8::from(q.paired_fresh_seal.is_some())]);
        if let Some(pair) = &q.paired_fresh_seal {
            hash.update(pair.public_context.0);
            hash.update(pair.root_state.semantic.0);
            hash.update(pair.context_policy.0);
            hash.update(pair.selected_records.to_le_bytes());
            hash.update([
                pair.anchors,
                pair.role as u8,
                u8::from(pair.perspective == rz_contracts::Color::Black),
            ]);
            hash.update(pair.model.0);
            hash.update(pair.encoding.0);
            utility_cohort_precision(&mut hash, pair.precision);
            hash.update(pair.frozen_epoch.to_le_bytes());
            hash.update(pair.record_revision.to_le_bytes());
        }
        hash.update([u8::from(q.legacy_full_records.is_some())]);
        if let Some(records) = &q.legacy_full_records {
            hash.update(records.root_state());
            hash.update((records.records().len() as u64).to_le_bytes());
            for record in records.records() {
                hash.update(
                    rz_position::contracts::rules_snapshot_semantic_digest(
                        self.engine
                            .semantic
                            .stores
                            .states
                            .get(record.origin_state)?,
                    )
                    .0,
                );
                for relation in [record.parent_revision, record.supersedes_revision] {
                    hash.update([u8::from(relation.is_some())]);
                    if let Some(revision) = relation {
                        let local = records
                            .records()
                            .iter()
                            .position(|candidate| candidate.revision == revision)
                            .ok_or(RoleError::InvalidOutput)?;
                        hash.update((local as u64).to_le_bytes());
                    }
                }
                self.utility_cohort_write_record(&mut hash, record)?;
            }
        }
        hash.update(namespace.utility_cohort_digest().0);
        hash.update([registration.mode as u8]);
        hash.update(registration.selective_policy.0);
        for goal in [registration.goal.depth_plies, registration.goal.mate_moves] {
            hash.update([u8::from(goal.is_some())]);
            if let Some(goal) = goal {
                hash.update(goal.get().to_le_bytes());
            }
        }
        hash.update([u8::from(registration.nodes.is_some())]);
        if let Some(nodes) = registration.nodes {
            hash.update(nodes.to_le_bytes());
        }
        hash.update(role_line_sha256(
            COHORT_LINE_DOMAIN,
            frontier.root_moves().allowed(),
        )?);
        let config = self.engine.semantic_sequence_config();
        for value in [
            self.engine.config.line_plies,
            config.line_plies,
            config.beam_width,
            config.max_nodes,
            config.max_records,
        ] {
            hash.update((value as u64).to_le_bytes());
        }
        hash.update(config.max_role_calls.to_le_bytes());
        hash.update(self.engine.config.cpu_nodes_per_task.to_le_bytes());
        hash.update(self.limits.max_rounds.to_le_bytes());
        hash.update(self.limits.max_cpu_nodes.to_le_bytes());
        hash.update(self.limits.cpu_depth.to_le_bytes());
        for text in [
            self.engine.resolver.version(),
            self.engine.post_repair_recheck.search_identity(),
            self.engine.semantic.model.identity(),
            self.engine.researcher.cpu_registered_condition.as_str(),
        ] {
            hash.update((text.len() as u64).to_le_bytes());
            hash.update(text.as_bytes());
        }
        // An empty selected context still depends on the actual registered R
        // weights and training identity, not only its condition namespace.
        let cpu_identity = serde_json::to_vec(self.engine.owned_identity()?)
            .map_err(|_| PalsError::InvalidConfig)?;
        hash.update((cpu_identity.len() as u64).to_le_bytes());
        hash.update(cpu_identity);
        self.check_live()?;
        Ok(UtilityCohortOfferSemanticSource {
            semantic: ContractDigest(hash.finalize().into()),
            owner: self.engine.semantic.stores.read_owner(),
            generation: self.engine.semantic.stores.generation(),
            root: self.root_state_id(),
            frame: work.frame,
            first: work.original_first_move,
            deadline: work.original_deadline,
            budget: work.budget,
            stage: work.input.stage,
            question: work.input.question_digest,
            view: work.input.view_seal,
            full_view: ContractDigest(work.input.view.digest()),
            public_context: work.input.public_context,
            query: Arc::downgrade(&work.input.question),
            selected: Arc::downgrade(&work.input.view),
        })
    }

    fn utility_cohort_write_view(
        &self,
        hash: &mut Sha256,
        view: &ReviewView,
    ) -> Result<(), PalsError> {
        if !view.validates(&self.engine.semantic.stores)
            || view.records().len() != view.records_metadata().len()
            || view.records().len() != view.source_snapshots().len()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        hash.update(view.digest());
        for ((record, metadata), source) in view
            .records()
            .iter()
            .zip(view.records_metadata())
            .zip(view.source_snapshots().iter())
        {
            self.utility_cohort_write_record(hash, record)?;
            hash.update(rz_position::contracts::rules_snapshot_semantic_digest(source).0);
            hash.update(metadata.source_line_offset.to_le_bytes());
            hash.update([u8::from(metadata.original_first_move.is_some())]);
            if let Some(first) = metadata.original_first_move {
                hash.update(Move16::pack(first)?.bits().to_le_bytes());
            }
        }
        Ok(())
    }

    fn utility_cohort_write_record(
        &self,
        hash: &mut Sha256,
        record: &RoleRecord,
    ) -> Result<(), PalsError> {
        hash.update([
            record.kind as u8,
            u8::from(record.perspective == Color::Black),
            u8::from(record.critical),
            u8::from(record.value.is_some()),
        ]);
        if let Some(value) = record.value {
            hash.update(value.to_le_bytes());
        }
        hash.update(record.completed_depth.to_le_bytes());
        hash.update([match record.score_scope {
            None => 0,
            Some(CpuScoreScope::FrontierOnly) => 1,
            Some(CpuScoreScope::CompletedIteration) => 2,
            Some(CpuScoreScope::RulesTerminal) => 3,
        }]);
        hash.update(role_line_sha256(COHORT_LINE_DOMAIN, &record.line)?);
        hash.update([u8::from(record.cpu_observation.is_some())]);
        if let Some(observation) = record.cpu_observation {
            self.validator_write_observation(hash, observation, 0)?;
        }
        Ok(())
    }

    pub(super) fn validated_utility_await(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &crate::cpu_worker::CpuSliceReport,
    ) -> Result<UtilityReciprocalAwaitSource, PalsError> {
        self.check_work(work)?;
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        let admission = self
            .cpu_admissions
            .get(&work.input.question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        let job = admission.job.ok_or(RoleError::InvalidOutput)?;
        let execution = admission.execution.ok_or(RoleError::InvalidOutput)?;
        let parent = self
            .validated_utility_r_parent(work)?
            .ok_or(RoleError::InvalidOutput)?;
        let request = receipt
            .reciprocal_request
            .as_ref()
            .ok_or(RoleError::InvalidOutput)?;
        let task = self.engine.semantic.stores.tasks.get(execution)?;
        if admission.frame != work.frame
            || admission.stage != work.input.stage
            || admission.view_seal != work.input.view_seal
            || admission.context != work.input.question.context
            || admission.first_move != work.original_first_move
            || admission.grant.is_some()
            || admission.owner != Some(receipt.grant.owner)
            || admission.terminal
            || admission.quarantined
            || !admission.continuation_retained
            || receipt.grant.job != job.id
            || receipt.terminal
            || receipt.admission_rejected
            || !receipt.continuation_retained
            || !receipt.result.as_ref().is_ok_and(|actual| {
                actual.report.completion == crate::cpu::CpuCompletion::AwaitReciprocal
            })
            || receipt.actual_work.is_none_or(|actual| actual.nodes == 0)
            || receipt.cumulative_nodes != admission.cumulative_nodes
            || receipt.progress_sequence != admission.progress_sequence
            || request.scope().job != job
            || request.scope().owner != receipt.grant.owner
            || request.scope().parent != parent
            || admission.settled_receipt
                != Some((
                    receipt.grant.execution,
                    execution,
                    cpu_commit::cpu_receipt_digest(receipt),
                ))
            || task.key != admission.task_key
            || task.status
                != (TaskStatus::Paused {
                    checkpoint: job.id.sequence,
                    evidence: None,
                })
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(UtilityReciprocalAwaitSource {
            job,
            execution: receipt.grant.execution,
            parent,
        })
    }

    pub(super) fn utility_cpu_continuation_cause(
        &self,
        question: ContractDigest,
        observation: ObservationId,
    ) -> Result<Option<UtilityContinuationCause>, PalsError> {
        let Some(admission) = self.cpu_admissions.get(&question) else {
            // Legacy/profiled external feedback has a different authority.
            return Ok(None);
        };
        let job = admission.job.ok_or(RoleError::InvalidOutput)?;
        let (execution, _, actual) = admission
            .committed_execution
            .ok_or(RoleError::InvalidOutput)?;
        if actual != observation
            || !self.cpu_feedback_still_authorized(question, observation)
            || !self.cpu_action_committed(question, job.id, execution)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(Some(UtilityContinuationCause::Cpu {
            job: job.id,
            execution,
            frame: admission.frame,
            question,
            observation,
        }))
    }

    pub(super) fn record_utility_continuation(
        &mut self,
        cause: UtilityContinuationCause,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), PalsError> {
        AdaptivePortQuestion::validate_frontier_input(input, &self.engine.semantic.stores)?;
        let first = input.question.first.ok_or(RoleError::InvalidOutput)?;
        let first = rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(first).map_err(|_| RoleError::InvalidOutput)?,
        );
        if cause.frame().authority != self.authority
            || self
                .utility_continuations
                .entries
                .contains_key(&input.question_digest)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        self.utility_continuations.entries.insert(
            input.question_digest,
            UtilityContinuationSource {
                owner: self.engine.semantic.stores.read_owner(),
                generation: self.engine.semantic.stores.generation(),
                root: self.engine.semantic.nodes[self.root].state,
                authority: self.authority,
                deadline: self.deadline,
                hard_deadline: self.limits.deadline,
                cause,
                target_query: Arc::downgrade(&input.question),
                target_view: Arc::downgrade(&input.view),
                stage: input.stage,
                question: input.question_digest,
                view: input.view_seal,
                full_view: ContractDigest(input.view.digest()),
                public_context: input.public_context,
                first,
            },
        );
        Ok(())
    }

    /// The caller supplies actual ready_work after a successful machine
    /// transition. Public frame equality or a reconstructed question cannot
    /// substitute for the exact input this owner materialized.
    pub(super) fn validated_utility_continuation(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<Option<UtilityContinuationSource>, PalsError> {
        self.check_work(work)?;
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        let Some(source) = self
            .utility_continuations
            .entries
            .get(&work.input.question_digest)
        else {
            return Ok(None);
        };
        if !source.owner.matches(&self.engine.semantic.stores)
            || source.generation != self.engine.semantic.stores.generation()
            || source.root != self.engine.semantic.nodes[self.root].state
            || source.authority != self.authority
            || source.deadline != work.original_deadline
            || source.hard_deadline != work.input.question.deadline
            || source.cause.frame() != work.frame
            || source.first != work.original_first_move
            || source.stage != work.input.stage
            || source.question != work.input.question_digest
            || source.view != work.input.view_seal
            || source.full_view != ContractDigest(work.input.view.digest())
            || source.public_context != work.input.public_context
            || !source
                .target_query
                .upgrade()
                .is_some_and(|query| Arc::ptr_eq(&query, &work.input.question))
            || !source
                .target_view
                .upgrade()
                .is_some_and(|view| Arc::ptr_eq(&view, &work.input.view))
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(Some(source.clone()))
    }
}

fn utility_cohort_precision(hash: &mut Sha256, precision: rz_contracts::PrecisionProfile) {
    match precision {
        rz_contracts::PrecisionProfile::Fp32 => hash.update([0]),
        rz_contracts::PrecisionProfile::Fp16 => hash.update([1]),
        rz_contracts::PrecisionProfile::Bf16 => hash.update([2]),
        rz_contracts::PrecisionProfile::Quantized(key) => {
            hash.update([3]);
            hash.update(key.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::port_tests::{output, setup, work};
    use super::*;

    fn cohort_port() -> AdaptiveEnginePort<AdaptivePortModel> {
        use crate::contract_time::InstantClock;
        use crate::cpu_worker::CpuCreditLedger;
        use crate::pals::rovent_engine::RootMoveDomain;
        use crate::pals::rovent_frontier::{FrontierLimits, FrontierRegistration};
        use rz_contracts::MonotonicTick;
        use rz_contracts::rovent::{ExecutionMode, SearchGoal};
        use std::num::NonZeroU16;

        // Reuse the existing semantic fixture, with the product metadata
        // mirror and actual namespace admission. This runs no role backend.
        let value = LegalOrderRoleMock.value_identity().unwrap().clone();
        let namespace = AdaptiveModelNamespace::new(
            value.clone(),
            ContractDigest([5; 32]),
            ContractDigest([6; 32]),
            rz_contracts::PrecisionProfile::Fp32,
            1,
        )
        .unwrap();
        let model = AdaptivePortModel::new_with_namespace(
            value.model.clone(),
            Some(value),
            namespace.clone(),
        )
        .unwrap();
        let mut port = super::super::port_tests::setup_with_model(
            Position::startpos(),
            3,
            ResolverPolicy::ModelWdlRestricted,
            model,
        );
        let allowed = [
            BoardMove::from_uci("e2e4").unwrap(),
            BoardMove::from_uci("d2d4").unwrap(),
        ];
        let domain = Arc::new(RootMoveDomain::new(&port.rules, Some(&allowed)).unwrap());
        let ledger = CpuCreditLedger::new(
            port.authority,
            2000,
            InstantClock::new(port.deadline.clock, Instant::now()),
        )
        .unwrap();
        let identity = ledger.identity();
        let registration = FrontierRegistration {
            authority: port.authority,
            mode: ExecutionMode::CpuPcr,
            original_deadline: port.deadline,
            root_budget: ContractDigest([88; 32]),
            goal: SearchGoal {
                depth_plies: NonZeroU16::new(3),
                mate_moves: None,
            },
            nodes: Some(2000),
            selective_policy: ContractDigest([89; 32]),
            root_rules: Arc::clone(&port.rules),
            root_moves: Arc::clone(&domain),
            root_state: port.root_state_id(),
            model_namespace: Arc::new(namespace),
            native_ledger: Some(identity.clone()),
        };
        port.attach_rovent_frontier(
            registration,
            FrontierLimits {
                maximum_goal_plies: NonZeroU16::new(16).unwrap(),
                maximum_frames: 16,
                maximum_paths: 64,
                maximum_mate_nodes: 64,
                maximum_native_receipts: 16,
            },
            AdaptiveFrontierStartup {
                mode: ExecutionMode::CpuPcr,
                root_budget: ContractDigest([88; 32]),
                root_moves: &domain,
                native_ledger: Some(&identity),
            },
            port.deadline.clock,
            MonotonicTick(0),
        )
        .unwrap();
        port.prepare_rovent_root_rules(port.deadline.clock, MonotonicTick(0))
            .unwrap();
        port
    }

    fn cohort_work(
        port: &mut AdaptiveEnginePort<AdaptivePortModel>,
        first: &str,
        sequence: u64,
    ) -> CoordinatorWork<ReviewView, AdaptivePortQuestion> {
        let first = BoardMove::from_uci(first).unwrap();
        let node = port.replay_node(&[first]).unwrap();
        let input = port
            .input(
                node,
                RefinementStage::ProposalContinuation,
                Some(first),
                vec![first],
                vec![first],
                None,
                None,
                None,
                false,
                None,
                None,
            )
            .unwrap();
        CoordinatorWork {
            frame: FrameId {
                authority: port.authority,
                sequence,
            },
            original_first_move: rz_contracts::pals::Move16::encode(
                rz_contracts::Move::try_from(first).unwrap(),
            ),
            original_deadline: port.deadline,
            budget: FrameBudget {
                role_submissions_remaining: port.engine.config.max_role_calls,
                original_cpu_credit: port.limits.max_cpu_nodes,
                original_store_byte_limit: 8 * 1024 * 1024,
            },
            input,
        }
    }

    #[test]
    fn cohort_content_can_match_reissuance_without_transferring_the_actual_offer() {
        let mut port = cohort_port();
        let first = cohort_work(&mut port, "e2e4", 1);
        let lease_count = port.engine.semantic.stores.review_pin_count().unwrap();
        let strong_query = Arc::strong_count(&first.input.question);
        let strong_view = Arc::strong_count(&first.input.view);
        let source = port.utility_cohort_offer_semantics(&first).unwrap();
        assert!(source.matches_work(&first));
        assert!(source.matches_store(&port.engine.semantic.stores));
        assert_eq!(Arc::strong_count(&first.input.question), strong_query);
        assert_eq!(Arc::strong_count(&first.input.view), strong_view);
        assert_eq!(
            port.engine.semantic.stores.review_pin_count().unwrap(),
            lease_count
        );

        let second = cohort_work(&mut port, "e2e4", 2);
        assert_ne!(first.input.question_digest, second.input.question_digest);
        assert_ne!(first.input.view.sequence(), second.input.view.sequence());
        let reissued = port.utility_cohort_offer_semantics(&second).unwrap();
        assert_eq!(source.digest(), reissued.digest());
        assert!(!source.matches_work(&second));
        assert!(reissued.matches_work(&second));

        let mut copied = first.clone();
        copied.input.question = Arc::new(first.input.question.as_ref().clone());
        assert!(!source.matches_work(&copied));
        let mut changed_budget = first.clone();
        changed_budget.budget.role_submissions_remaining -= 1;
        assert!(!source.matches_work(&changed_budget));
        let mut changed_deadline = first.clone();
        changed_deadline.original_deadline.at.0 += 1;
        assert!(!source.matches_work(&changed_deadline));
        assert!(
            port.utility_cohort_offer_semantics(&changed_deadline)
                .is_err()
        );
    }

    #[test]
    fn cohort_cap_keeps_store_identity_separate_from_contents_and_query_choice() {
        let mut first_port = cohort_port();
        let first_work = cohort_work(&mut first_port, "e2e4", 1);
        let source = first_port
            .utility_cohort_offer_semantics(&first_work)
            .unwrap();
        let mut second_port = cohort_port();
        let second_work = cohort_work(&mut second_port, "e2e4", 9);
        let same_contents = second_port
            .utility_cohort_offer_semantics(&second_work)
            .unwrap();
        assert_eq!(source.digest(), same_contents.digest());
        assert!(!source.matches_store(&second_port.engine.semantic.stores));
        assert!(!same_contents.matches_store(&first_port.engine.semantic.stores));
        assert!(!source.matches_work(&second_work));
        let other_first = cohort_work(&mut first_port, "d2d4", 2);
        let different = first_port
            .utility_cohort_offer_semantics(&other_first)
            .unwrap();
        assert_ne!(source.digest(), different.digest());
        assert!(source.matches_store(&first_port.engine.semantic.stores));
    }

    #[test]
    fn ordinary_utility_cause_requires_consumed_source_and_exact_materialized_target() {
        let mut port = setup(Position::startpos(), 3);
        let original = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
        let (ticket, response) = output(&original, 1);
        port.utility_continuations.enable();
        let step = port
            .commit_role(&original, &ticket, &response, None)
            .unwrap();
        port.role_consumed(&ticket, &original.input.view).unwrap();
        let FrameTransition::Continue(input) = port.materialize_step(step).unwrap().transition
        else {
            panic!("actual semantic continuation")
        };
        let target = CoordinatorWork {
            input,
            ..original.clone()
        };
        let leases = port.engine.semantic.stores.review_pin_count().unwrap();
        let view_owners = Arc::strong_count(&target.input.view);
        let source = port
            .validated_utility_continuation(&target)
            .unwrap()
            .unwrap();
        assert_eq!(
            source.role_source(),
            Some((
                ticket.id,
                original.frame,
                original.input.question_digest,
                original.input.view_seal.view_digest
            ))
        );
        assert!(source.cpu_source().is_none());
        assert_eq!(Arc::strong_count(&target.input.view), view_owners);
        assert_eq!(
            port.engine.semantic.stores.review_pin_count().unwrap(),
            leases
        );
        assert!(
            port.validated_utility_continuation(&original)
                .unwrap()
                .is_none()
        );

        let mut copied_query = target.clone();
        copied_query.input.question = Arc::new(target.input.question.as_ref().clone());
        assert!(port.validated_utility_continuation(&copied_query).is_err());
        let mut other_frame = target.clone();
        other_frame.frame.sequence += 1;
        assert!(port.validated_utility_continuation(&other_frame).is_err());
        let mut changed_scope = target.clone();
        changed_scope.input.public_context = ContractDigest([243; 32]);
        assert!(port.validated_utility_continuation(&changed_scope).is_err());
        assert!(
            port.validated_utility_continuation(&target)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn unconsumed_or_default_off_materialization_grants_no_utility_successor() {
        let mut port = setup(Position::startpos(), 3);
        let original = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
        let (ticket, response) = output(&original, 1);
        let step = port
            .commit_role(&original, &ticket, &response, None)
            .unwrap();
        port.role_consumed(&ticket, &original.input.view).unwrap();
        let FrameTransition::Continue(input) = port.materialize_step(step).unwrap().transition
        else {
            panic!("actual semantic continuation")
        };
        let target = CoordinatorWork {
            input,
            ..original.clone()
        };
        assert!(port.utility_continuations.entries.is_empty());
        assert!(
            port.validated_utility_continuation(&target)
                .unwrap()
                .is_none()
        );

        let mut port = setup(Position::startpos(), 3);
        let original = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
        let (ticket, response) = output(&original, 1);
        port.utility_continuations.enable();
        let step = port
            .commit_role(&original, &ticket, &response, None)
            .unwrap();
        assert!(port.materialize_step(step).is_err());
        assert!(port.utility_continuations.entries.is_empty());
    }
}
