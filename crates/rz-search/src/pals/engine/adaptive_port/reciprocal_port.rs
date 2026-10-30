//! Exact selected-view bridge for an owner-paused native R question. The
//! existing Store, native ticket service and CPU worker keep their authority.
use super::*;
mod fresh_usage;
use crate::cpu::{CpuCompletion, CpuReciprocalReply, CpuReciprocalRequest};
use crate::cpu_worker::CpuSliceReport;
use crate::pals::coordinator::{
    AcceptedPeerSourceSeal, AcceptedRoleOrigin, AcceptedRoleStep, MAX_COORDINATOR_FRAMES,
    MAX_RECIPROCAL_QUESTION_HISTORY, accepted_peer_semantic_digest,
};
use rz_contracts::pals_adaptive::CpuJobEnvelope;
use rz_contracts::rovent::{
    ChessPly, ExecutionMode, QuestionEnvelope, QuestionPurpose, ReciprocalQuestionId,
    SearchRecipient,
};

#[derive(Default)]
pub(super) struct ReciprocalPortState {
    parents: HashMap<(FrameId, ContractDigest), QuestionEnvelope>,
    pending: HashMap<ReciprocalQuestionId, PendingCpuQuestion>,
    answers: VecDeque<Arc<AdaptiveAcceptedReciprocalAnswer>>,
    consumed_sources: HashMap<RequestId, ConsumedRoleSource>,
    peer_publications: HashMap<RequestId, Result<ContractDigest, PalsError>>,
    source_publications: HashMap<RequestId, Result<AdaptiveSourceOpinion, PalsError>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pals::coordinator::{
        NativeAcceptanceIssuer, PhysicalRoleCompletion, RoleCompletion, RoleSubmission,
        RoleTicketRegistry,
    };
    use crate::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
    use rz_contracts::pals::RepresentationKey;
    use rz_contracts::{
        ClockDomain, GameGeneration, MonotonicTick, PrecisionProfile, ProcessEpoch, RootGeneration,
    };

    type Port = AdaptiveEnginePort<LegalOrderRoleMock>;
    type Work = CoordinatorWork<ReviewView, AdaptivePortQuestion>;

    fn source() -> (Port, Work, AcceptedRoleStep, RoleOutput) {
        let cpu = CpuEngine::new(crate::cpu::CpuConfig {
            tt_entries: 16,
            max_depth: 8,
            ..crate::cpu::CpuConfig::default()
        })
        .unwrap();
        let engine = PalsEngine::new(
            PalsConfig {
                line_plies: 4,
                ..PalsConfig::default()
            },
            LegalOrderRoleMock,
            cpu,
        )
        .unwrap()
        .with_selected_review_context(
            ReviewSelectionPolicy::default(),
            ReviewIndexLimits::default(),
        )
        .unwrap();
        let authority = SearchAuthority {
            epoch: ProcessEpoch(37),
            game: GameGeneration(0),
            root: RootGeneration(1),
            implementation: ContractDigest([17; 32]),
        };
        let deadline = Deadline {
            clock: ClockDomain(authority.epoch),
            at: MonotonicTick(20_000_000_000),
        };
        let sequence = Arc::new(std::sync::atomic::AtomicU64::new(1));
        let issue: AdaptiveRulesOwnerIssuer = Arc::new(move || {
            Ok(OwnerId(
                sequence.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            ))
        });
        let mut port = AdaptiveEnginePort::new(
            engine,
            Position::startpos(),
            authority,
            deadline,
            PalsLimits {
                deadline: Instant::now() + Duration::from_secs(20),
                max_rounds: 1,
                max_cpu_nodes: 1000,
                cpu_depth: 2,
            },
            issue,
        )
        .unwrap();
        port.prepare_root_rules().unwrap();
        let candidate = port
            .make_root_candidate(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        let work = CoordinatorWork {
            frame: FrameId {
                authority,
                sequence: 1,
            },
            original_first_move: candidate.first_move,
            original_deadline: deadline,
            budget: candidate.budget,
            input: candidate.input,
        };
        let representation = RepresentationKey {
            input: ContractDigest([1; 32]),
            history: ContractDigest([2; 32]),
            candidates: ContractDigest([3; 32]),
            mask_positions: ContractDigest([4; 32]),
            model: ContractDigest([5; 32]),
            encoding: ContractDigest([6; 32]),
            precision: PrecisionProfile::Fp32,
            frozen_epoch: 1,
            record_revision: work.input.question.context.public_revision,
            role: Role::Proposer,
        };
        let mut registry = RoleTicketRegistry::<ReviewView, RoleOutput>::new(
            authority,
            512,
            rz_runtime::adaptive::CoordinatorPolicy::Fixed,
            1,
        )
        .unwrap();
        let ticket = registry
            .admit(
                RoleSubmission {
                    question: work.input.question_digest,
                    view: work.input.view_seal,
                    representation,
                    role: Role::Proposer,
                    original_deadline: deadline,
                },
                Arc::clone(&work.input.view),
                deadline.clock,
                MonotonicTick(0),
            )
            .unwrap();
        registry.take_dispatch().unwrap();
        let output = RoleOutput {
            id: ticket.id,
            authority,
            key: representation,
            payload: RolePayload::Proposal {
                candidate_logits: vec![0.0; work.input.view.query().legal.len()],
                wdl: rz_contracts::Wdl::try_new(0.25, 0.5, 0.25, 0.0).unwrap(),
            },
            private_latent: vec![0.0; 16 * 384],
        };
        registry
            .complete_reciprocal_role(RoleCompletion {
                ticket,
                physical: PhysicalRoleCompletion::Confirmed,
                result: Ok(output.clone()),
            })
            .unwrap();
        registry
            .take_ready(deadline.clock, MonotonicTick(0))
            .unwrap();
        let (issuer, verifier) = NativeAcceptanceIssuer::new_pair();
        registry
            .register_native_acceptance_verifier(verifier)
            .unwrap();
        let permit = registry
            .issue_reciprocal_native_acceptance(&ticket, deadline.clock, MonotonicTick(0))
            .unwrap();
        let accepted = registry
            .accept_reciprocal_native(
                issuer.issue_success(permit),
                deadline.clock,
                MonotonicTick(0),
            )
            .unwrap();
        (port, work, accepted, output)
    }

    fn pending(port: &mut Port, work: &Work) {
        let anchor = Arc::new(
            port.issue_rules_position(work.input.question.position.clone())
                .unwrap()
                .export()
                .unwrap(),
        );
        let id = ReciprocalQuestionId {
            authority: work.frame.authority,
            sequence: 1,
        };
        let envelope = QuestionEnvelope {
            id,
            parent: None,
            sender: SearchRecipient::Researcher,
            recipient: SearchRecipient::Proposer,
            purpose: QuestionPurpose::Propose,
            anchor: anchor.snapshot().identity(),
            anchor_root_ply: ChessPly(1),
            original_first_move: Some(work.original_first_move),
            prefix_digest: ContractDigest(work.input.question.context.prefix_sha256),
            line_digest: None,
            line_offset: ChessPly(0),
            question_digest: work.input.question_digest,
            view: work.input.view_seal,
            evidence_revision: work.input.question.context.public_revision,
            root_budget: ContractDigest([8; 32]),
            original_deadline: work.original_deadline,
        };
        port.reciprocal.pending.insert(
            id,
            PendingCpuQuestion {
                envelope,
                input: work.input.clone(),
                request: None,
                anchor,
                origin: None,
            },
        );
    }

    #[test]
    fn failed_view_consumption_stays_failed_without_publishing_an_answer_or_counter() {
        let (mut port, work, accepted, output) = source();
        pending(&mut port, &work);
        port.engine
            .semantic
            .review_index
            .as_mut()
            .unwrap()
            .consume(&work.input.view, &port.engine.semantic.stores)
            .unwrap();
        for _ in 0..2 {
            assert!(matches!(
                port.consume_cpu_reciprocal(&work, &accepted, &output, None),
                Err(PalsError::Review(ReviewError::AlreadyConsumed))
            ));
            assert!(!port.accepted.contains(&accepted.ticket().id));
            assert!(
                !port
                    .committed_role_questions
                    .contains_key(&accepted.ticket().id)
            );
            assert_eq!(port.counters.consumed_role_outputs, 0);
            assert_eq!(port.counters.role_calls, 0);
            assert!(port.reciprocal.answers.is_empty());
        }
    }

    #[test]
    fn cached_answer_requires_the_original_question_arc_and_frame() {
        let (mut port, work, accepted, output) = source();
        pending(&mut port, &work);
        let answer = port
            .consume_cpu_reciprocal(&work, &accepted, &output, None)
            .unwrap();
        let repeated = port
            .consume_cpu_reciprocal(&work, &accepted, &output, None)
            .unwrap();
        assert!(Arc::ptr_eq(&answer, &repeated));
        let mut changed = work.clone();
        let mut query = changed.input.question.as_ref().clone();
        query.context.public_revision += 1;
        changed.input.question = Arc::new(query);
        assert!(matches!(
            port.consume_cpu_reciprocal(&changed, &accepted, &output, None),
            Err(PalsError::Role(RoleError::InvalidOutput))
        ));
        let mut changed = work.clone();
        changed.frame.sequence += 1;
        assert!(matches!(
            port.consume_cpu_reciprocal(&changed, &accepted, &output, None),
            Err(PalsError::Role(RoleError::InvalidOutput))
        ));
        assert_eq!(port.counters.consumed_role_outputs, 1);
    }

    #[test]
    fn accepted_source_consumption_is_charged_once_before_final_legacy_commit() {
        let (mut port, work, accepted, output) = source();
        port.accept_accepted_role_source(&work, &accepted, &output)
            .unwrap();
        port.accept_accepted_role_source(&work, &accepted, &output)
            .unwrap();
        assert!(
            port.role_consumption_recorded(accepted.ticket(), &work.input.view)
                .unwrap()
        );
        assert_eq!(port.counters.role_calls, 1);
        port.commit_role(&work, accepted.ticket(), &output, None)
            .unwrap();
        assert_eq!(port.counters.role_calls, 1);
        assert_eq!(port.counters.consumed_role_outputs, 1);
        assert!(
            !port
                .reciprocal
                .consumed_sources
                .contains_key(&accepted.ticket().id)
        );
    }

    #[test]
    fn preconsumed_peer_finishes_at_the_role_cap_without_a_second_view_or_role_charge() {
        let (mut port, work, accepted, output) = source();
        pending(&mut port, &work);
        let pending = port.reciprocal.pending.values_mut().next().unwrap();
        pending.origin = Some(accepted.origin(work.frame, None).unwrap());
        port.engine.config.max_role_calls = 1;
        assert!(port.enable_role_consumption_observation());

        // This is the same source head already read to select a nested peer.
        port.accept_accepted_role_source(&work, &accepted, &output)
            .unwrap();
        assert_eq!(port.counters.role_calls, 1);
        let evidence = port
            .consume_accepted_role_reciprocal(&work, &accepted, &output, None)
            .unwrap();
        let revision = port.engine.semantic.revision;
        assert_eq!(
            port.consume_accepted_role_reciprocal(&work, &accepted, &output, None)
                .unwrap(),
            evidence
        );
        assert_eq!(port.engine.semantic.revision, revision);
        assert_eq!(port.counters.role_calls, 1);
        assert_eq!(port.counters.consumed_role_outputs, 1);
        assert_eq!(port.counters.accepted_proposer_outputs, 1);
        assert_eq!(port.consumed_role_views.len(), 1);
        assert!(
            !port
                .reciprocal
                .consumed_sources
                .contains_key(&accepted.ticket().id)
        );
        assert_eq!(port.reciprocal.peer_publications.len(), 1);
        assert_eq!(
            port.role_consumption_observation()
                .unwrap()
                .events()
                .iter()
                .filter(|event| {
                    event.request == accepted.ticket().id
                        && event.outcome == AdaptiveRoleConsumptionOutcome::ConsumedView
                })
                .count(),
            1
        );
    }

    #[test]
    fn peer_preparation_retry_rechecks_the_original_native_output_identity() {
        let (mut port, work, accepted, output) = source();
        // This fixture already has the public proposal required by C. Missing
        // dependent context is a different boundary from output retry identity.
        port.record_hot(
            RecordKind::Proposal,
            &work.input.question.context.prefix,
            None,
        )
        .unwrap();
        let id = ReciprocalQuestionId {
            authority: work.frame.authority,
            sequence: 1,
        };
        let prepare = |port: &mut Port, output: &RoleOutput| {
            port.prepare_accepted_role_reciprocal(
                &work,
                &accepted,
                output,
                id,
                ExecutionMode::HybridPcr,
                ContractDigest([8; 32]),
                SearchRecipient::Critic,
            )
        };
        assert!(prepare(&mut port, &output).unwrap().is_some());
        let mut changed = output.clone();
        changed.id.sequence += 1;
        assert!(matches!(
            prepare(&mut port, &changed),
            Err(PalsError::Role(RoleError::InvalidOutput))
        ));
        let mut changed = output.clone();
        changed.key.model = ContractDigest([99; 32]);
        assert!(matches!(
            prepare(&mut port, &changed),
            Err(PalsError::Role(RoleError::InvalidOutput))
        ));
        assert!(prepare(&mut port, &output).unwrap().is_some());
        assert_eq!(port.counters.role_calls, 1);
        assert_eq!(port.reciprocal.pending.len(), 1);
    }

    #[test]
    fn first_accepted_policy_publishes_its_raw_opinion_before_the_dependent_view() {
        let (mut port, work, accepted, output) = source();
        assert_eq!(port.diagnostic_accepted_source_revision(&accepted), None);
        assert!(port.engine.semantic.records.is_empty());
        assert!(work.input.view.records().is_empty());
        let prepare = |port: &mut Port, sequence, recipient| {
            port.prepare_accepted_role_reciprocal(
                &work,
                &accepted,
                &output,
                ReciprocalQuestionId {
                    authority: work.frame.authority,
                    sequence,
                },
                ExecutionMode::HybridPcr,
                ContractDigest([8; 32]),
                recipient,
            )
            .unwrap()
            .unwrap()
        };
        let peer = prepare(&mut port, 1, SearchRecipient::Critic);
        let opinion = port
            .accepted_source_opinion(accepted.ticket().id)
            .unwrap()
            .as_ref()
            .unwrap();
        assert_eq!(opinion.frame, work.frame);
        assert_eq!(opinion.ticket, *accepted.ticket());
        assert_eq!(opinion.output, accepted.output_digest());
        assert_eq!(opinion.original_first_move, work.original_first_move);
        assert_eq!(
            port.engine
                .semantic
                .stores
                .lines
                .moves(opinion.line)
                .unwrap(),
            peer.input.question.context.prefix
        );
        assert!(
            peer.input
                .view
                .records()
                .iter()
                .any(|record| record.line == peer.input.question.context.prefix)
        );
        let raw = port
            .engine
            .semantic
            .stores
            .observations
            .get(opinion.observation)
            .unwrap();
        assert_eq!(raw.score, RawScore::Unknown);
        let revision = port.engine.semantic.revision;
        assert_eq!(
            port.diagnostic_accepted_source_revision(&accepted),
            Some(revision)
        );
        assert!(
            peer.input
                .view
                .records_metadata()
                .iter()
                .any(|record| record.source_revision == revision)
        );
        let _researcher = prepare(&mut port, 2, SearchRecipient::Researcher);
        assert_eq!(port.engine.semantic.revision, revision);
        assert_eq!(port.engine.semantic.records.len(), 1);
        assert_eq!(port.reciprocal.source_publications.len(), 1);
        assert_eq!(port.counters.consumed_role_outputs, 1);
        assert!(work.input.view.records().is_empty());
        assert_eq!(
            port.diagnostic_accepted_source_revision(&accepted),
            Some(revision)
        );
    }

    #[test]
    fn false_original_first_move_cannot_consume_or_publish_an_accepted_source() {
        let (mut port, work, accepted, output) = source();
        let mut changed = work.clone();
        changed.original_first_move = rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(BoardMove::from_uci("d2d4").unwrap()).unwrap(),
        );
        assert!(matches!(
            port.prepare_accepted_role_reciprocal(
                &changed,
                &accepted,
                &output,
                ReciprocalQuestionId {
                    authority: work.frame.authority,
                    sequence: 1,
                },
                ExecutionMode::HybridPcr,
                ContractDigest([8; 32]),
                SearchRecipient::Critic,
            ),
            Err(PalsError::Role(RoleError::InvalidOutput))
        ));
        assert_eq!(port.counters.role_calls, 0);
        assert!(port.accepted.is_empty());
        assert!(port.reciprocal.pending.is_empty());
        assert!(port.reciprocal.source_publications.is_empty());
        assert!(port.engine.semantic.records.is_empty());
        port.accept_accepted_role_source(&work, &accepted, &output)
            .unwrap();
        assert!(matches!(
            port.accept_accepted_role_source(&changed, &accepted, &output),
            Err(PalsError::Role(RoleError::InvalidOutput))
        ));
        assert_eq!(port.counters.role_calls, 1);
        assert_eq!(port.reciprocal.consumed_sources.len(), 1);
    }
}

struct PendingCpuQuestion {
    envelope: QuestionEnvelope,
    input: FrameInput<ReviewView, AdaptivePortQuestion>,
    request: Option<CpuReciprocalRequest>,
    anchor: Arc<ContractState>,
    origin: Option<AcceptedRoleOrigin>,
}

struct ConsumedRoleSource {
    frame: FrameId,
    original_first_move: rz_contracts::pals::Move16,
    ticket: RoleTicketEnvelope,
    output: ContractDigest,
    question: Arc<AdaptivePortQuestion>,
    view: Arc<ReviewView>,
    failure: Option<PalsError>,
}

pub struct AdaptiveCpuReciprocalQuestion {
    pub question: QuestionEnvelope,
    pub input: FrameInput<ReviewView, AdaptivePortQuestion>,
}

pub struct AdaptivePeerQuestion {
    pub origin: AcceptedRoleOrigin,
    pub question: QuestionEnvelope,
    pub input: FrameInput<ReviewView, AdaptivePortQuestion>,
}

/// Actual accepted head -> Rules-checked raw line -> public record provenance.
/// This is a model opinion, never a CPU/Rules refutation or a search bound.
#[derive(Clone, Debug)]
pub struct AdaptiveSourceOpinion {
    pub frame: FrameId,
    pub ticket: RoleTicketEnvelope,
    pub output: ContractDigest,
    pub original_first_move: rz_contracts::pals::Move16,
    pub kind: RecordKind,
    pub line: LineId,
    pub observation: ObservationId,
    pub source_root: StateId,
}

/// One consumed canonical ticket, shared only with aliases of its exact view.
/// It is a policy answer, never a completed minimax or Rules terminal proof.
pub struct AdaptiveAcceptedReciprocalAnswer {
    question: QuestionEnvelope,
    ticket: RoleTicketEnvelope,
    evidence: ContractDigest,
    ranked: Vec<BoardMove>,
    view: Arc<ReviewView>,
    frame: FrameId,
    query: Arc<AdaptivePortQuestion>,
}
impl AdaptiveAcceptedReciprocalAnswer {
    pub fn question(&self) -> QuestionEnvelope {
        self.question
    }
    pub fn ticket(&self) -> &RoleTicketEnvelope {
        &self.ticket
    }
    pub fn evidence(&self) -> ContractDigest {
        self.evidence
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Read only the original actual peer input retained by this owner. Alias
    /// ambiguity cannot select a new/latest envelope for a utility action.
    pub(super) fn validated_utility_peer(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<Option<(QuestionEnvelope, Option<AcceptedRoleOrigin>)>, PalsError> {
        self.check_work(work)?;
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        let mut matching = self
            .reciprocal
            .pending
            .values()
            .filter(|pending| pending.envelope.question_digest == work.input.question_digest);
        let Some(pending) = matching.next() else {
            return Ok(None);
        };
        if matching.next().is_some()
            || pending.envelope.original_first_move != Some(work.original_first_move)
            || pending.envelope.original_deadline != work.original_deadline
            || pending.envelope.view != work.input.view_seal
            || pending.input.stage != work.input.stage
            || pending.input.question_digest != work.input.question_digest
            || pending.input.public_context != work.input.public_context
            || !Arc::ptr_eq(&pending.input.question, &work.input.question)
            || !Arc::ptr_eq(&pending.input.view, &work.input.view)
            || !pending.input.view.validates(&self.engine.semantic.stores)
            || !pending
                .anchor
                .rules()
                .snapshot()
                .same_state(&work.input.question.position.snapshot())
            || self.rovent_frontier().is_none_or(|frontier| {
                frontier.registration().root_budget != pending.envelope.root_budget
            })
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(Some((pending.envelope, pending.origin)))
    }

    /// Join ancestry uses its actual edge alias; the head still belongs to the
    /// canonical child input. Neither identity is selected from a latest slot.
    pub(super) fn validated_utility_peer_dependency(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        proof: &crate::pals::coordinator::ReciprocalDependencyProof<
            ReviewView,
            AdaptivePortQuestion,
        >,
    ) -> Result<(QuestionEnvelope, Option<AcceptedRoleOrigin>), PalsError> {
        self.check_work(work)?;
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        if !proof.validates_child(work) {
            return Err(RoleError::InvalidOutput.into());
        }
        let canonical = proof.question();
        let alias = proof.alias_question();
        let input = self
            .reciprocal
            .pending
            .get(&canonical.id)
            .ok_or(RoleError::InvalidOutput)?;
        let original_alias = self
            .reciprocal
            .pending
            .get(&alias.id)
            .ok_or(RoleError::InvalidOutput)?;
        AdaptivePortQuestion::validate_frontier_input(
            &original_alias.input,
            &self.engine.semantic.stores,
        )?;
        if input.envelope != canonical
            || original_alias.envelope != alias
            || original_alias.origin != proof.origin()
            || input.input.stage != work.input.stage
            || input.input.question_digest != work.input.question_digest
            || input.input.view_seal != work.input.view_seal
            || input.input.public_context != work.input.public_context
            || !Arc::ptr_eq(&input.input.question, &work.input.question)
            || !Arc::ptr_eq(&input.input.view, &work.input.view)
            || canonical.original_first_move != Some(work.original_first_move)
            || canonical.original_deadline != work.original_deadline
            || !input
                .anchor
                .rules()
                .snapshot()
                .same_state(&work.input.question.position.snapshot())
            || self.rovent_frontier().is_none_or(|frontier| {
                frontier.registration().root_budget != canonical.root_budget
                    || frontier.registration().root_budget != alias.root_budget
            })
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok((canonical, proof.origin()))
    }

    /// Existing R descriptors are issued at actual job admission. This does
    /// not create a parent question from a public frame or a root snapshot.
    pub(super) fn validated_utility_r_parent(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<Option<QuestionEnvelope>, PalsError> {
        self.check_work(work)?;
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        let Some(parent) = self
            .reciprocal
            .parents
            .get(&(work.frame, work.input.question_digest))
        else {
            return Ok(None);
        };
        if parent.recipient != SearchRecipient::Researcher
            || parent.original_first_move != Some(work.original_first_move)
            || parent.original_deadline != work.original_deadline
            || parent.view != work.input.view_seal
            || parent.question_digest != work.input.question_digest
            || self
                .rovent_frontier()
                .is_none_or(|frontier| frontier.registration().root_budget != parent.root_budget)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(Some(*parent))
    }

    /// Preserve an already-issued peer R question instead of replacing its
    /// parent/origin with an unrelated local descriptor for an external job.
    pub(super) fn profiled_reciprocal_question(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<Option<QuestionEnvelope>, PalsError> {
        let Some(pending) = self
            .reciprocal
            .pending
            .values()
            .find(|pending| pending.envelope.question_digest == work.input.question_digest)
        else {
            return Ok(None);
        };
        if pending.request.is_some()
            || pending.origin.is_none()
            || pending.envelope.recipient != SearchRecipient::Researcher
            || pending.envelope.original_first_move != Some(work.original_first_move)
            || pending.envelope.original_deadline != work.original_deadline
            || pending.input.view_seal != work.input.view_seal
            || !Arc::ptr_eq(&pending.input.question, &work.input.question)
            || !Arc::ptr_eq(&pending.input.view, &work.input.view)
            || !pending
                .anchor
                .rules()
                .snapshot()
                .same_state(&work.input.question.position.snapshot())
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(Some(pending.envelope))
    }
    pub fn accepted_source_opinion(
        &self,
        request: RequestId,
    ) -> Option<&Result<AdaptiveSourceOpinion, PalsError>> {
        self.reciprocal.source_publications.get(&request)
    }

    /// Read-only diagnostic projection of the original successful publication.
    /// The public record revision already lives in that exact observation's
    /// Model scope. No latest-record scan, replay, new publication or authority.
    pub fn diagnostic_accepted_source_revision(&self, accepted: &AcceptedRoleStep) -> Option<u64> {
        let source = self
            .reciprocal
            .source_publications
            .get(&accepted.ticket().id)?
            .as_ref()
            .ok()?;
        if source.ticket != *accepted.ticket() || source.output != accepted.output_digest() {
            return None;
        }
        let original = self
            .engine
            .semantic
            .stores
            .observations
            .get(source.observation)
            .ok()?;
        if original.state != source.source_root || original.line != Some(source.line) {
            return None;
        }
        match original.scope {
            EvidenceScope::Model { input, .. } => Some(input),
            _ => None,
        }
    }

    fn publish_accepted_source_opinion(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        kind: RecordKind,
        line: &[BoardMove],
    ) -> Result<AdaptiveSourceOpinion, PalsError> {
        if !accepted.validates_output(output)
            || !self.reciprocal_source_consumed(work, accepted.ticket(), output)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if let Some(previous) = self
            .reciprocal
            .source_publications
            .get(&accepted.ticket().id)
        {
            return previous.clone();
        }
        if self.reciprocal.source_publications.len() >= MAX_RECIPROCAL_QUESTION_HISTORY {
            return Err(PalsError::Capacity);
        }
        let published = (|| {
            let (line, observation) =
                self.record_hot(kind, line, None)?
                    .ok_or(StoreError::InvalidEvidence(
                        "accepted source opinion was not published",
                    ))?;
            Ok(AdaptiveSourceOpinion {
                frame: work.frame,
                ticket: *accepted.ticket(),
                output: accepted.output_digest(),
                original_first_move: work.original_first_move,
                kind,
                line,
                observation,
                source_root: self.engine.semantic.nodes[self.root].state,
            })
        })();
        // Preserve partial Store failure; a retry may not append an unsealed
        // second raw opinion or relabel failure as a completed publication.
        self.reciprocal
            .source_publications
            .insert(accepted.ticket().id, published.clone());
        published
    }

    pub(super) fn reciprocal_source_consumed(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> bool {
        self.reciprocal
            .consumed_sources
            .get(&ticket.id)
            .is_some_and(|source| {
                source.frame == work.frame
                    && source.original_first_move == work.original_first_move
                    && source.ticket == *ticket
                    && Arc::ptr_eq(&source.question, &work.input.question)
                    && Arc::ptr_eq(&source.view, &work.input.view)
                    && crate::pals::coordinator::native_acceptance_output_digest(output).ok()
                        == Some(source.output)
                    && self
                        .role_consumption_recorded(ticket, &work.input.view)
                        .unwrap_or(false)
            })
    }

    pub(super) fn retire_reciprocal_source(&mut self, ticket: RequestId) {
        self.reciprocal.consumed_sources.remove(&ticket);
    }

    fn account_reciprocal_role(&mut self, purpose: RoleQueryPurpose) {
        self.counters.role_calls += 1;
        self.counters.consumed_role_outputs += 1;
        match purpose {
            RoleQueryPurpose::ValueFresh => {
                self.counters.value_calls += 1;
                self.counters.completed_value_calls += 1;
                self.counters.accepted_value_outputs += 1;
            }
            RoleQueryPurpose::RepairPolicy => {
                self.counters.repair_calls += 1;
                self.counters.completed_repair_calls += 1;
                self.counters.accepted_repair_outputs += 1;
            }
            RoleQueryPurpose::ReplyPolicy | RoleQueryPurpose::DivergencePolicy => {
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
    }

    /// Reading a source head for a follow-up is already search consumption.
    /// Final legacy evidence commit remains a separate one-use operation.
    pub fn accept_accepted_role_source(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
    ) -> Result<(), PalsError> {
        self.check_work(work)?;
        if work.input.question.first.is_none() {
            return Err(RoleError::InvalidOutput.into());
        }
        if self.reciprocal_source_consumed(work, accepted.ticket(), output) {
            if accepted.validates_output(output) {
                return Ok(());
            }
            return Err(RoleError::InvalidOutput.into());
        }
        self.validate_role(work, accepted.ticket(), output)?;
        if !accepted.validates_output(output)
            || !matches!(
                work.input.question.context.purpose,
                RoleQueryPurpose::ProposePolicy
                    | RoleQueryPurpose::ReplyPolicy
                    | RoleQueryPurpose::RepairPolicy
                    | RoleQueryPurpose::DivergencePolicy
            )
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if self.counters.role_calls >= self.engine.config.max_role_calls {
            return Err(PalsError::RoleCallLimit);
        }
        if self.reciprocal.consumed_sources.len() >= MAX_COORDINATOR_FRAMES {
            return Err(PalsError::Capacity);
        }
        self.accepted.insert(accepted.ticket().id);
        let previous = self
            .committed_role_questions
            .insert(accepted.ticket().id, accepted.ticket().question);
        if let Err(error) = self.role_consumed(accepted.ticket(), &work.input.view) {
            self.accepted.remove(&accepted.ticket().id);
            if let Some(previous) = previous {
                self.committed_role_questions
                    .insert(accepted.ticket().id, previous);
            } else {
                self.committed_role_questions.remove(&accepted.ticket().id);
            }
            return Err(error);
        }
        self.reciprocal.consumed_sources.insert(
            accepted.ticket().id,
            ConsumedRoleSource {
                frame: work.frame,
                original_first_move: work.original_first_move,
                ticket: *accepted.ticket(),
                output: accepted.output_digest(),
                question: Arc::clone(&work.input.question),
                view: Arc::clone(&work.input.view),
                failure: None,
            },
        );
        self.account_reciprocal_role(work.input.question.context.purpose);
        Ok(())
    }

    /// The peer observes the exact entered continuation of the accepted head.
    /// Local source provenance is not disguised as a made-up peer question.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_accepted_role_reciprocal(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        id: ReciprocalQuestionId,
        mode: ExecutionMode,
        root_budget: ContractDigest,
        recipient: SearchRecipient,
    ) -> Result<Option<AdaptivePeerQuestion>, PalsError> {
        self.check_work(work)?;
        if work.input.question.first.is_none() {
            return Err(RoleError::InvalidOutput.into());
        }
        self.verify_registered_model_namespace(accepted.ticket(), output)?;
        if !accepted.validates_output(output) {
            return Err(RoleError::InvalidOutput.into());
        }
        let sender = match accepted.ticket().role {
            Role::Proposer => SearchRecipient::Proposer,
            Role::Critic => SearchRecipient::Critic,
            Role::Validator => return Err(RoleError::InvalidOutput.into()),
        };
        if sender == recipient
            || !mode.supports(recipient)
            || id.authority != self.authority
            || id.sequence == 0
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if let Some(pending) = self.reciprocal.pending.get(&id) {
            let origin = pending.origin.ok_or(RoleError::InvalidOutput)?;
            if !origin.matches(accepted)
                || origin.frame() != work.frame
                || pending.envelope.sender != sender
                || pending.envelope.recipient != recipient
                || pending.envelope.root_budget != root_budget
                || !self.reciprocal_source_consumed(work, accepted.ticket(), output)
            {
                return Err(RoleError::InvalidOutput.into());
            }
            return Ok(Some(AdaptivePeerQuestion {
                origin,
                question: pending.envelope,
                input: pending.input.clone(),
            }));
        }
        self.accept_accepted_role_source(work, accepted, output)?;
        let divergence = work.input.question.context.purpose == RoleQueryPurpose::DivergencePolicy;
        let mut attack_ply = None;
        let prefix = if divergence {
            let RolePayload::Counterexample {
                divergence_logits, ..
            } = &output.payload
            else {
                return Err(RoleError::InvalidOutput.into());
            };
            let selected = divergence_logits
                .iter()
                .enumerate()
                .max_by(|(a, left), (b, right)| left.total_cmp(right).then_with(|| b.cmp(a)));
            let Some((selected, _)) = selected else {
                return Ok(None);
            };
            let attack = *work
                .input
                .view
                .query()
                .divergences
                .get(selected)
                .ok_or(RoleError::InvalidOutput)?;
            if attack >= work.input.question.proposal.len() {
                return Err(RoleError::InvalidOutput.into());
            }
            attack_ply = Some(attack);
            work.input.question.proposal[..attack].to_vec()
        } else {
            let ranked = role_commit::rank_moves(&work.input.view.query().legal, output)?;
            let Some(movement) = ranked.first().copied() else {
                return Ok(None);
            };
            if work.input.question.context.prefix.len() >= self.continuation_limit().min(256) {
                return Ok(None);
            }
            let mut prefix = work.input.question.context.prefix.clone();
            prefix.push(movement);
            prefix
        };
        let node = self.replay_node(&prefix)?;
        if self.engine.semantic.nodes[node].terminal.is_some() {
            return Ok(None);
        }
        if self
            .original_goal_horizon()
            .is_some_and(|horizon| prefix.len() >= horizon)
        {
            // The original accepted source was consumed above. At a go's
            // endpoint there is no remaining ply for a new optional policy
            // or R peer. Its ordinary continuation still supplies the real
            // endpoint Fresh ticket; no dependent completion is fabricated.
            return Ok(None);
        }
        if self.reciprocal.pending.len() >= MAX_COORDINATOR_FRAMES {
            return Err(PalsError::Capacity);
        }
        let source_peer = self
            .reciprocal
            .pending
            .values()
            .filter(|held| {
                held.envelope.question_digest == work.input.question_digest
                    && held.input.view_seal == work.input.view_seal
            })
            .min_by_key(|held| held.envelope.id.sequence)
            .map(|held| held.envelope);
        let origin = accepted
            .origin(work.frame, source_peer)
            .map_err(|_| RoleError::InvalidOutput)?;
        let source = AcceptedPeerSourceSeal::freeze(origin, &work.input.view, output)
            .map_err(|_| RoleError::InvalidOutput)?;
        let parent = source_peer.map(|peer| peer.id);
        if parent.is_some_and(|parent| parent.sequence >= id.sequence) {
            return Err(RoleError::InvalidOutput.into());
        }
        let (stage, purpose, question_purpose) = match recipient {
            SearchRecipient::Proposer => {
                if divergence {
                    (
                        RefinementStage::ProposalContinuation,
                        RoleQueryPurpose::ProposePolicy,
                        QuestionPurpose::Propose,
                    )
                } else {
                    (
                        RefinementStage::Repair,
                        RoleQueryPurpose::RepairPolicy,
                        QuestionPurpose::Repair,
                    )
                }
            }
            SearchRecipient::Critic => (
                RefinementStage::CriticReply,
                RoleQueryPurpose::ReplyPolicy,
                QuestionPurpose::Counter,
            ),
            SearchRecipient::Researcher => (
                if divergence {
                    RefinementStage::CpuDiscovery
                } else if sender == SearchRecipient::Proposer {
                    RefinementStage::CpuValidation
                } else {
                    RefinementStage::CounterValidation
                },
                RoleQueryPurpose::ProposePolicy,
                QuestionPurpose::TacticalReview,
            ),
        };
        let proposal = if sender == SearchRecipient::Proposer {
            prefix.clone()
        } else {
            work.input.question.proposal.clone()
        };
        let counter = (sender == SearchRecipient::Critic && !divergence).then(|| prefix.clone());
        if !divergence {
            let kind = match sender {
                SearchRecipient::Proposer
                    if work.input.question.context.purpose == RoleQueryPurpose::RepairPolicy =>
                {
                    RecordKind::Repair
                }
                SearchRecipient::Proposer => RecordKind::Proposal,
                SearchRecipient::Critic => RecordKind::Counterexample,
                SearchRecipient::Researcher => return Err(RoleError::InvalidOutput.into()),
            };
            self.publish_accepted_source_opinion(work, accepted, output, kind, &prefix)?;
        }
        let mut input = self.input_with_purpose(
            node,
            stage,
            work.input.question.first,
            prefix.clone(),
            proposal,
            counter,
            None,
            attack_ply,
            false,
            None,
            None,
            purpose,
        )?;
        let semantic = accepted_peer_semantic_digest(
            self.reciprocal_semantic_digest(&input, root_budget)?,
            &source,
        );
        let canonical = self
            .reciprocal
            .pending
            .values()
            .find(|held| held.envelope.question_digest == semantic);
        let anchor;
        if let Some(canonical) = canonical {
            input = canonical.input.clone();
            anchor = Arc::clone(&canonical.anchor);
        } else {
            let mut query = input.question.as_ref().clone();
            query.question_digest = semantic;
            input.question_digest = semantic;
            input.question = Arc::new(query);
            anchor = Arc::new(
                self.issue_rules_position(self.engine.semantic.nodes[node].position.clone())?
                    .export()
                    .map_err(|_| RoleError::InvalidOutput)?,
            );
        }
        let envelope = QuestionEnvelope {
            id,
            parent,
            sender,
            recipient,
            purpose: question_purpose,
            anchor: anchor.snapshot().identity(),
            anchor_root_ply: ChessPly(
                u16::try_from(prefix.len()).map_err(|_| PalsError::Capacity)?,
            ),
            original_first_move: Some(work.original_first_move),
            prefix_digest: ContractDigest(role_line_sha256(b"prefix", &prefix)?),
            line_digest: None,
            line_offset: ChessPly(0),
            question_digest: semantic,
            view: input.view_seal,
            evidence_revision: input.question.context.public_revision,
            root_budget,
            original_deadline: work.original_deadline,
        };
        self.reciprocal.pending.insert(
            id,
            PendingCpuQuestion {
                envelope,
                input: input.clone(),
                request: None,
                anchor,
                origin: Some(origin),
            },
        );
        Ok(Some(AdaptivePeerQuestion {
            origin,
            question: envelope,
            input,
        }))
    }

    pub fn register_cpu_reciprocal_scope(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &CpuJobEnvelope,
        question: &QuestionEnvelope,
    ) -> Result<(), PalsError> {
        self.check_work(work)?;
        let pending = self
            .reciprocal
            .pending
            .values()
            .find(|pending| pending.envelope == *question)
            .ok_or(RoleError::InvalidOutput)?;
        let admission = self
            .cpu_admissions
            .get(&work.input.question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        if question.recipient != SearchRecipient::Researcher
            || question.question_digest != work.input.question_digest
            || pending.input.view_seal != work.input.view_seal
            || job.state_identity != admission.rules_identity
            || question.anchor != job.state_identity
            || job.question != work.input.question_digest
            || job.hard_deadline != work.original_deadline
            || job.id.authority != self.authority
            || admission.frame != work.frame
        {
            return Err(RoleError::InvalidOutput.into());
        }
        self.reciprocal
            .parents
            .insert((work.frame, work.input.question_digest), *question);
        Ok(())
    }

    /// Publish the actual peer policy as an unverified model opinion. It does
    /// not certify a refutation, Rules result, completed CPU task or value.
    pub fn consume_accepted_role_reciprocal(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: Option<AdaptivePortRoleEvidence>,
    ) -> Result<ContractDigest, PalsError> {
        // The first accepted head must keep its original custody if native
        // reservations currently prevent checking its chosen child. Nothing
        // has been consumed or cached as a failed publication at this point.
        if !self
            .reciprocal
            .answers
            .iter()
            .any(|answer| answer.ticket == *accepted.ticket())
        {
            self.validate_role(work, accepted.ticket(), output)?;
            if !accepted.validates_output(output) {
                return Err(RoleError::InvalidOutput.into());
            }
            let pending = self
                .reciprocal
                .pending
                .values()
                .find(|pending| {
                    pending.envelope.question_digest == work.input.question_digest
                        && pending.input.view_seal == work.input.view_seal
                        && pending.origin.is_some()
                })
                .ok_or(RoleError::InvalidOutput)?;
            if pending.request.is_some()
                || !matches!(
                    work.input.question.context.purpose,
                    RoleQueryPurpose::ProposePolicy
                        | RoleQueryPurpose::ReplyPolicy
                        | RoleQueryPurpose::RepairPolicy
                )
            {
                return Err(RoleError::InvalidOutput.into());
            }
            if !self.reciprocal_source_consumed(work, accepted.ticket(), output)
                && self.counters.role_calls >= self.engine.config.max_role_calls
            {
                return Err(PalsError::RoleCallLimit);
            }
            if self.reciprocal.peer_publications.len() >= MAX_RECIPROCAL_QUESTION_HISTORY {
                return Err(PalsError::Capacity);
            }
            let movement = role_commit::rank_moves(&work.input.view.query().legal, output)?
                .first()
                .copied()
                .ok_or(RoleError::InvalidOutput)?;
            if self.rovent_frontier().is_some() {
                let mut line = work.input.question.context.prefix.clone();
                line.push(movement);
                self.checked_frontier_line(&line)?;
            }
        }
        let answer = self.consume_cpu_reciprocal(work, accepted, output, physical)?;
        if let Some(original) = self.reciprocal.peer_publications.get(&accepted.ticket().id) {
            return original.clone();
        }
        if self.reciprocal.peer_publications.len() >= MAX_RECIPROCAL_QUESTION_HISTORY {
            return Err(PalsError::Capacity);
        }
        let published = (|| {
            let pending = self
                .reciprocal
                .pending
                .values()
                .find(|pending| {
                    pending.envelope.question_digest == work.input.question_digest
                        && pending.input.view_seal == work.input.view_seal
                        && pending.origin.is_some()
                })
                .ok_or(RoleError::InvalidOutput)?;
            if pending.request.is_some() {
                return Err(RoleError::InvalidOutput.into());
            }
            let Some(movement) = answer.ranked.first().copied() else {
                return Err(RoleError::InvalidOutput.into());
            };
            let mut line = work.input.question.context.prefix.clone();
            line.push(movement);
            let kind = match work.input.question.context.purpose {
                RoleQueryPurpose::ReplyPolicy => RecordKind::Counterexample,
                RoleQueryPurpose::RepairPolicy => RecordKind::Repair,
                RoleQueryPurpose::ProposePolicy => RecordKind::Proposal,
                _ => return Err(RoleError::InvalidOutput.into()),
            };
            let publication = self.record_hot(kind, &line, None)?;
            let mut digest = Sha256::new();
            digest.update(b"rz-rovent-consumed-peer-policy-opinion/1\0");
            digest.update(answer.evidence.0);
            digest.update(work.input.question_digest.0);
            if let Some((_, observation)) = publication {
                digest.update((observation.0 as u64).to_le_bytes());
            }
            Ok(ContractDigest(digest.finalize().into()))
        })();
        // A failed partial Store publication is retained as failure, never
        // replayed as an ordinary success or silently appended twice.
        self.reciprocal
            .peer_publications
            .insert(accepted.ticket().id, published.clone());
        published
    }

    /// Consume only the exact terminal CPU observation. Store publication and
    /// scope validation reuse the existing CPU owner; no legacy stage follows.
    pub fn consume_reciprocal_cpu(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &CpuSliceReport,
    ) -> Result<ContractDigest, PalsError> {
        self.check_work(work)?;
        if !receipt.terminal
            || receipt.continuation_retained
            || receipt.admission_rejected
            || receipt.actual_work.is_none()
            || receipt.reciprocal_request.is_some()
            || !receipt.result.as_ref().is_ok_and(|actual| {
                !matches!(
                    actual.report.completion,
                    CpuCompletion::AwaitReciprocal
                        | CpuCompletion::Yielded
                        | CpuCompletion::Canceled
                        | CpuCompletion::Deadline
                ) && actual.report.resume.is_none()
            })
        {
            return Err(StoreError::InvalidEvidence(
                "peer R answer needs a completed original owner receipt",
            )
            .into());
        }
        if !self.reciprocal.pending.values().any(|pending| {
            pending.envelope.question_digest == work.input.question_digest
                && pending.input.view_seal == work.input.view_seal
                && pending.envelope.recipient == SearchRecipient::Researcher
                && pending.origin.is_some()
        }) {
            return Err(RoleError::InvalidOutput.into());
        }
        let already = self
            .cpu_admissions
            .get(&work.input.question_digest)
            .is_some_and(|admission| {
                admission
                    .settled_receipt
                    .is_some_and(|(physical, _, digest)| {
                        physical == receipt.grant.execution
                            && digest == cpu_commit::cpu_receipt_digest(receipt)
                    })
                    && admission
                        .consumed_observation
                        .is_some_and(|observation| self.cpu_observation_consumed(work, observation))
            });
        if !already {
            self.commit_cpu_report(work, receipt)?;
        }
        let admission = self
            .cpu_admissions
            .get(&work.input.question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        let (_, _, observation) = admission
            .committed_execution
            .ok_or(RoleError::InvalidOutput)?;
        if !self.cpu_observation_consumed(work, observation) {
            return Err(StoreError::InvalidEvidence("peer R raw result was not consumed").into());
        }
        let mut digest = Sha256::new();
        digest.update(b"rz-rovent-consumed-peer-R-observation/1\0");
        digest.update(work.input.question_digest.0);
        digest.update(cpu_commit::cpu_receipt_digest(receipt).0);
        digest.update((observation.0 as u64).to_le_bytes());
        Ok(ContractDigest(digest.finalize().into()))
    }

    pub fn discard_accepted_role_reciprocal(&mut self, id: ReciprocalQuestionId) -> bool {
        self.reciprocal.pending.remove(&id).is_some()
    }
    /// The CPU preparation already issued this Rules owner. Keep its identity
    /// instead of exporting a new root owner or guessing an anchor from a hash.
    pub fn cpu_reciprocal_parent(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &CpuJobEnvelope,
        id: ReciprocalQuestionId,
        mode: ExecutionMode,
        root_budget: ContractDigest,
    ) -> Result<QuestionEnvelope, PalsError> {
        self.check_work(work)?;
        let sender = match work.input.stage {
            RefinementStage::CpuValidation | RefinementStage::RepairValidation => {
                SearchRecipient::Proposer
            }
            RefinementStage::CpuDiscovery
            | RefinementStage::CounterValidation
            | RefinementStage::RecheckCounterValidation => SearchRecipient::Critic,
            _ => return Err(RoleError::Unsupported("question is not a native R stage").into()),
        };
        let admission = self
            .cpu_admissions
            .get(&work.input.question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        if !mode.has_native_r()
            || id.authority != self.authority
            || id.sequence == 0
            || job.id.authority != self.authority
            || job.question != work.input.question_digest
            || job.state_identity != admission.rules_identity
            || job.hard_deadline != self.deadline
            || admission.frame != work.frame
            || admission.context != work.input.question.context
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let key = (work.frame, work.input.question_digest);
        if let Some(original) = self.reciprocal.parents.get(&key) {
            return if original.id == id && original.root_budget == root_budget {
                Ok(*original)
            } else {
                Err(RoleError::InvalidOutput.into())
            };
        }
        if self.reciprocal.parents.len() >= MAX_RECIPROCAL_QUESTION_HISTORY {
            return Err(PalsError::Capacity);
        }
        let question = QuestionEnvelope {
            id,
            parent: None,
            sender,
            recipient: SearchRecipient::Researcher,
            purpose: QuestionPurpose::TacticalReview,
            anchor: job.state_identity,
            anchor_root_ply: ChessPly(
                u16::try_from(work.input.question.context.prefix.len())
                    .map_err(|_| PalsError::Capacity)?,
            ),
            original_first_move: Some(work.original_first_move),
            prefix_digest: ContractDigest(role_line_sha256(
                b"prefix",
                &work.input.question.context.prefix,
            )?),
            line_digest: None,
            line_offset: ChessPly(0),
            question_digest: work.input.question_digest,
            view: work.input.view_seal,
            evidence_revision: work.input.question.context.public_revision,
            root_budget,
            original_deadline: work.original_deadline,
        };
        question
            .view
            .validate(self.authority, work.input.view.token_budget())
            .map_err(|_| RoleError::InvalidOutput)?;
        self.reciprocal.parents.insert(key, question);
        Ok(question)
    }

    /// Reuse exact Rules replay, hot graph insertion and Store-pinned selection.
    /// No played history is appended to the model-facing virtual prefix.
    pub fn prepare_cpu_reciprocal(
        &mut self,
        parent: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        request: &CpuReciprocalRequest,
        id: ReciprocalQuestionId,
        mode: ExecutionMode,
        root_budget: ContractDigest,
    ) -> Result<AdaptiveCpuReciprocalQuestion, PalsError> {
        self.check_work(parent)?;
        let scope = request.scope();
        if !mode.has_native_r()
            || self
                .reciprocal
                .parents
                .get(&(parent.frame, parent.input.question_digest))
                != Some(&scope.parent)
            || id.authority != self.authority
            || id.sequence <= scope.parent.id.sequence
            || root_budget != scope.parent.root_budget
            || scope.job.question != parent.input.question_digest
            || scope.parent.view != parent.input.view_seal
            || scope.parent.original_deadline != parent.original_deadline
            || request.relative_prefix().len() != usize::from(request.ply())
            || self.reciprocal.pending.contains_key(&id)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if self.reciprocal.pending.len() >= MAX_COORDINATOR_FRAMES {
            return Err(PalsError::Capacity);
        }
        let (stage, purpose, envelope_purpose) = match scope.recipient {
            SearchRecipient::Proposer => (
                RefinementStage::ProposalContinuation,
                RoleQueryPurpose::ProposePolicy,
                QuestionPurpose::Propose,
            ),
            SearchRecipient::Critic => (
                RefinementStage::CriticReply,
                RoleQueryPurpose::ReplyPolicy,
                QuestionPurpose::Counter,
            ),
            SearchRecipient::Researcher => return Err(RoleError::InvalidOutput.into()),
        };
        let mut prefix = parent.input.question.context.prefix.clone();
        prefix.extend_from_slice(request.relative_prefix());
        if prefix.len() > 256 {
            return Err(PalsError::Capacity);
        }
        if self.rovent_frontier().is_none() {
            let mut replay = parent.input.question.position.clone();
            for movement in request.relative_prefix() {
                replay.make_move(*movement)?;
            }
            if !replay.snapshot().same_state(&request.position().snapshot()) {
                return Err(ReviewError::AnchorMismatch.into());
            }
        }
        let node = self.replay_node(&prefix)?;
        if !self.engine.semantic.nodes[node]
            .position
            .snapshot()
            .same_state(&request.position().snapshot())
        {
            return Err(ReviewError::AnchorMismatch.into());
        }
        let mut input = self.input_with_purpose(
            node,
            stage,
            parent.input.question.first,
            prefix.clone(),
            Vec::new(),
            None,
            None,
            None,
            false,
            None,
            None,
            purpose,
        )?;
        // The selected input digest covers model-facing contents, not issuance
        // sequence. Keep the old V5 question digest path byte-for-byte intact.
        let semantic = self.reciprocal_semantic_digest(&input, root_budget)?;
        let canonical = self
            .reciprocal
            .pending
            .values()
            .find(|held| held.input.question_digest == semantic);
        let anchor;
        if let Some(canonical) = canonical {
            input = canonical.input.clone();
            anchor = Arc::clone(&canonical.anchor);
        } else {
            let mut query = input.question.as_ref().clone();
            query.question_digest = semantic;
            input.question_digest = semantic;
            input.question = Arc::new(query);
            anchor = Arc::new(
                self.issue_rules_position(request.position().clone())?
                    .export()
                    .map_err(|_| RoleError::InvalidOutput)?,
            );
        }
        let envelope = QuestionEnvelope {
            id,
            parent: Some(scope.parent.id),
            sender: SearchRecipient::Researcher,
            recipient: scope.recipient,
            purpose: envelope_purpose,
            anchor: anchor.snapshot().identity(),
            anchor_root_ply: ChessPly(
                u16::try_from(prefix.len()).map_err(|_| PalsError::Capacity)?,
            ),
            original_first_move: Some(parent.original_first_move),
            prefix_digest: ContractDigest(role_line_sha256(b"prefix", &prefix)?),
            line_digest: None,
            line_offset: ChessPly(0),
            question_digest: semantic,
            view: input.view_seal,
            evidence_revision: input.question.context.public_revision,
            root_budget,
            original_deadline: parent.original_deadline,
        };
        self.reciprocal.pending.insert(
            id,
            PendingCpuQuestion {
                envelope,
                input: input.clone(),
                request: Some(request.clone()),
                anchor,
                origin: None,
            },
        );
        Ok(AdaptiveCpuReciprocalQuestion {
            question: envelope,
            input,
        })
    }

    fn reciprocal_semantic_digest(
        &self,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
        budget: ContractDigest,
    ) -> Result<ContractDigest, PalsError> {
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-exact-reciprocal-question/1\0");
        hash.update([input.stage as u8]);
        hash.update(input.view.digest());
        hash.update(input.view.policy_digest());
        hash.update(input.public_context.0);
        hash.update(input.question.context.focus_prefix_sha256());
        hash.update(self.rules.snapshot().identity().semantic.0);
        hash.update(budget.0);
        let model = self.engine.semantic.model.identity().as_bytes();
        hash.update((model.len() as u64).to_le_bytes());
        hash.update(model);
        if let Some(value) = self.engine.semantic.model.value_identity() {
            for text in [
                &value.semantics,
                &value.model,
                &value.encoding,
                &value.precision,
            ] {
                hash.update((text.len() as u64).to_le_bytes());
                hash.update(text.as_bytes());
            }
            hash.update(value.model_epoch);
        }
        if let Some(records) = &input.question.legacy_full_records {
            hash.update(records.digest());
        }
        Ok(ContractDigest(hash.finalize().into()))
    }

    /// Admission/acceptance run on original ticket data. The legacy fixed
    /// PALS transition is intentionally not invoked for a one-question answer.
    pub fn consume_cpu_reciprocal(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        _physical: Option<AdaptivePortRoleEvidence>,
    ) -> Result<Arc<AdaptiveAcceptedReciprocalAnswer>, PalsError> {
        self.check_work(work)?;
        if let Some(answer) = self
            .reciprocal
            .answers
            .iter()
            .find(|answer| answer.ticket == *accepted.ticket())
        {
            self.verify_registered_model_namespace(accepted.ticket(), output)?;
            if answer.evidence != accepted.output_digest()
                || !accepted.validates_output(output)
                || accepted.ticket().question != work.input.question_digest
                || accepted.ticket().view != work.input.view_seal
                || !Arc::ptr_eq(&answer.view, &work.input.view)
                || answer.frame != work.frame
                || !Arc::ptr_eq(&answer.query, &work.input.question)
                || !self.role_consumption_recorded(accepted.ticket(), &work.input.view)?
            {
                return Err(RoleError::InvalidOutput.into());
            }
            return Ok(Arc::clone(answer));
        }
        self.validate_role(work, accepted.ticket(), output)?;
        let source_preconsumed = self.reciprocal_source_consumed(work, accepted.ticket(), output);
        if !accepted.validates_output(output) {
            return Err(RoleError::InvalidOutput.into());
        }
        let pending = self
            .reciprocal
            .pending
            .values()
            .find(|held| {
                held.envelope.question_digest == work.input.question_digest
                    && held.input.view_seal == work.input.view_seal
            })
            .ok_or(RoleError::InvalidOutput)?;
        if !matches!(
            work.input.question.context.purpose,
            RoleQueryPurpose::ProposePolicy
                | RoleQueryPurpose::ReplyPolicy
                | RoleQueryPurpose::RepairPolicy
        ) {
            return Err(RoleError::InvalidOutput.into());
        }
        if !source_preconsumed && self.counters.role_calls >= self.engine.config.max_role_calls {
            return Err(PalsError::RoleCallLimit);
        }
        let ranked = role_commit::rank_moves(&work.input.view.query().legal, output)?;
        let answer = Arc::new(AdaptiveAcceptedReciprocalAnswer {
            question: pending.envelope,
            ticket: *accepted.ticket(),
            evidence: accepted.output_digest(),
            ranked,
            view: Arc::clone(&work.input.view),
            frame: work.frame,
            query: Arc::clone(&work.input.question),
        });
        self.reciprocal
            .answers
            .retain(|answer| Arc::strong_count(answer) > 1);
        if self.reciprocal.answers.len() >= MAX_COORDINATOR_FRAMES {
            return Err(PalsError::Capacity);
        }
        let inserted_acceptance = self.accepted.insert(accepted.ticket().id);
        let previous = self
            .committed_role_questions
            .insert(accepted.ticket().id, accepted.ticket().question);
        if let Err(error) = self.role_consumed(accepted.ticket(), &work.input.view) {
            if inserted_acceptance {
                self.accepted.remove(&accepted.ticket().id);
            }
            if let Some(previous) = previous {
                self.committed_role_questions
                    .insert(accepted.ticket().id, previous);
            } else {
                self.committed_role_questions.remove(&accepted.ticket().id);
            }
            return Err(error);
        }
        if !source_preconsumed {
            self.account_reciprocal_role(work.input.question.context.purpose);
        }
        self.retire_reciprocal_source(accepted.ticket().id);
        self.reciprocal.answers.push_back(Arc::clone(&answer));
        Ok(answer)
    }

    /// A joined caller receives a reply for its own original opaque request;
    /// the shared output remains labelled with the canonical submitted ticket.
    pub fn cpu_reciprocal_reply(
        &mut self,
        answer: &AdaptiveAcceptedReciprocalAnswer,
        request: &CpuReciprocalRequest,
        alias: ReciprocalQuestionId,
    ) -> Result<CpuReciprocalReply, PalsError> {
        self.check_live()?;
        let pending = self
            .reciprocal
            .pending
            .get(&alias)
            .ok_or(RoleError::InvalidOutput)?;
        let original = pending.request.as_ref().ok_or(RoleError::InvalidOutput)?;
        if pending.envelope.question_digest != answer.question.question_digest
            || pending.input.view_seal != answer.ticket.view
            || !Arc::ptr_eq(&pending.input.view, &answer.view)
            || !original.same_request(request)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let preferred = answer
            .ranked
            .iter()
            .copied()
            .filter(|movement| original.candidates().contains(movement))
            .collect();
        let reply =
            CpuReciprocalReply::new(original, pending.envelope, &pending.anchor, preferred)?;
        Ok(reply)
    }

    /// Drop only a logical pin/admission. The manager still retains and drains
    /// every native ticket and the R worker's private continuation separately.
    pub fn discard_cpu_reciprocal(&mut self, id: ReciprocalQuestionId) -> bool {
        self.reciprocal.pending.remove(&id).is_some()
    }

    /// Preserve actual CPU usage and the original Store execution as Paused.
    /// No frontier score, incomplete frame or model output is published here.
    pub fn cpu_reciprocal_wait(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &CpuSliceReport,
    ) -> Result<(), PalsError> {
        let admission = self
            .cpu_admissions
            .get(&work.input.question_digest)
            .ok_or(RoleError::InvalidOutput)?
            .clone();
        let job = admission.job.ok_or(RoleError::InvalidOutput)?;
        let execution = admission.execution.ok_or(RoleError::InvalidOutput)?;
        let request = receipt
            .reciprocal_request
            .as_ref()
            .ok_or(RoleError::InvalidOutput)?;
        let result = receipt
            .result
            .as_ref()
            .map_err(|error| PalsError::Cpu(error.clone()))?;
        let actual = receipt
            .actual_work
            .ok_or(StoreError::InvalidEvidence("reciprocal CPU work unknown"))?;
        let cumulative = admission
            .cumulative_nodes
            .checked_add(actual.nodes)
            .ok_or(PalsError::Capacity)?;
        if admission.terminal
            || admission.quarantined
            || admission.frame != work.frame
            || work.frame.authority != self.authority
            || work.original_deadline != self.deadline
            || admission.grant != Some(receipt.grant)
            || admission.owner != Some(receipt.grant.owner)
            || job.id != receipt.grant.job
            || request.scope().job != job
            || request.scope().owner != receipt.grant.owner
            || self
                .reciprocal
                .parents
                .get(&(work.frame, work.input.question_digest))
                != Some(&request.scope().parent)
            || result.report.completion != CpuCompletion::AwaitReciprocal
            || result.report.resume.is_some()
            || receipt.terminal
            || receipt.admission_rejected
            || !receipt.continuation_retained
            || result.report.nodes != actual.nodes
            || result.report.quiescence_nodes != actual.quiescence_nodes
            || result.report.tt_hits != actual.tt_hits
            || actual.nodes > receipt.grant.node_credit
            || actual.nodes == 0
            || actual.quiescence_nodes > actual.nodes
            || cumulative != receipt.cumulative_nodes
            || cumulative > admission.credit
            || receipt.progress_sequence < admission.progress_sequence
            || admission.context != work.input.question.context
            || admission.view_seal != work.input.view_seal
            || self.seal(&work.input.view) != admission.view_seal
            || work.input.question.node != admission.node
            || work.input.question.question_digest != work.input.question_digest
            || !work
                .input
                .question
                .position
                .snapshot()
                .same_state(&admission.position.snapshot())
            || admission.first_move != work.original_first_move
            || admission.stage != work.input.stage
            || job.hard_deadline != work.original_deadline
            || self.engine.semantic.stores.tasks.get(execution)?.key != admission.task_key
            || self.engine.semantic.stores.tasks.get(execution)?.status != TaskStatus::InFlight
        {
            return Err(StoreError::InvalidEvidence(
                "reciprocal CPU owner/usage/Store receipt mismatch",
            )
            .into());
        }
        let nodes = self
            .counters
            .cpu_nodes
            .checked_add(actual.nodes)
            .ok_or(PalsError::Capacity)?;
        let qnodes = self
            .counters
            .cpu_quiescence_nodes
            .checked_add(actual.quiescence_nodes)
            .ok_or(PalsError::Capacity)?;
        let hits = self
            .counters
            .cpu_tt_hits
            .checked_add(actual.tt_hits)
            .ok_or(PalsError::Capacity)?;
        let tasks = self
            .counters
            .cpu_tasks
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        if nodes > self.limits.max_cpu_nodes {
            return Err(
                StoreError::InvalidEvidence("reciprocal wait exceeds root CPU credit").into(),
            );
        }
        self.engine
            .semantic
            .stores
            .pause_task(execution, job.id.sequence, None)?;
        self.counters.cpu_nodes = nodes;
        self.counters.cpu_quiescence_nodes = qnodes;
        self.counters.cpu_tt_hits = hits;
        self.counters.cpu_tasks = tasks;
        let stored = self
            .cpu_admissions
            .get_mut(&work.input.question_digest)
            .expect("checked native R admission");
        stored.grant = None;
        stored.cumulative_nodes = cumulative;
        stored.progress_sequence = receipt.progress_sequence;
        stored.last_started_sequence = receipt.grant.execution.sequence;
        stored.continuation_retained = true;
        stored.settled_receipt = Some((
            receipt.grant.execution,
            execution,
            cpu_commit::cpu_receipt_digest(receipt),
        ));
        Ok(())
    }
}
