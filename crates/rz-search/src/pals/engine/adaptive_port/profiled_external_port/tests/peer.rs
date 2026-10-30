//! Optional P -> external-R round trip through the actual registry and worker.
//! P heads are authored CPU/mock data, not NN execution evidence. The existing
//! ELF fixture is likewise a protocol fixture, not an official engine pin.
use super::*;
use crate::pals::coordinator::{
    DependentFrameAdmission, FramePhase, NativeAcceptanceIssuer, PhysicalRoleCompletion,
    ProposerRankingPublished, RefinementMachine, RoleCompletion, RoleSubmission,
    RoleTicketRegistry, StepEvidence, StepExecution, TicketPhase,
};
use rz_contracts::pals::RepresentationKey;
use rz_runtime::adaptive::CoordinatorPolicy;

type Registry = RoleTicketRegistry<ReviewView, RoleOutput>;

/// The same real registry producer protocol used by port_tests' ranking
/// fixture, adapted only to this fixture's explicitly registered namespace.
fn authored_proposer_submission(
    registry: &mut Registry,
    input: &FrameInput<ReviewView, AdaptivePortQuestion>,
    deadline: Deadline,
    selected: &str,
    clock: ClockDomain,
    now: MonotonicTick,
) -> (RoleTicketEnvelope, RoleOutput) {
    let mut candidates = Sha256::new();
    for movement in &input.view.query().legal {
        let packed =
            rz_contracts::pals::Move16::encode(rz_contracts::Move::try_from(*movement).unwrap());
        candidates.update(packed.bits().to_le_bytes());
    }
    let representation = RepresentationKey {
        input: ContractDigest(input.view.digest()),
        history: input.view_seal.root_state.semantic,
        candidates: ContractDigest(candidates.finalize().into()),
        mask_positions: ContractDigest(
            Sha256::digest(vec![1_u8; input.view.query().legal.len()]).into(),
        ),
        // These are the existing AuthoredWorkerFixture namespace declaration,
        // not digests inferred from a display name or a native loaded model.
        model: ContractDigest([5; 32]),
        encoding: ContractDigest([6; 32]),
        precision: PrecisionProfile::Fp32,
        frozen_epoch: 1,
        record_revision: input.question.context.public_revision,
        role: Role::Proposer,
    };
    let ticket = registry
        .admit(
            RoleSubmission {
                question: input.question_digest,
                view: input.view_seal,
                representation,
                role: Role::Proposer,
                original_deadline: deadline,
            },
            Arc::clone(&input.view),
            clock,
            now,
        )
        .unwrap();
    assert_eq!(registry.phase(ticket.id), Some(TicketPhase::Prepared));
    let dispatched = registry.take_dispatch().unwrap();
    assert_eq!(dispatched.ticket, ticket);
    assert_eq!(registry.phase(ticket.id), Some(TicketPhase::Submitted));
    let selected = BoardMove::from_uci(selected).unwrap();
    assert!(input.view.query().legal.contains(&selected));
    let output = RoleOutput {
        id: ticket.id,
        authority: ticket.authority,
        key: representation,
        payload: RolePayload::Proposal {
            candidate_logits: input
                .view
                .query()
                .legal
                .iter()
                .map(|movement| if *movement == selected { 1.0 } else { -1.0 })
                .collect(),
            wdl: rz_contracts::Wdl::try_new(0.25, 0.5, 0.25, 0.0).unwrap(),
        },
        private_latent: vec![0.0; 16 * 384],
    };
    (ticket, output)
}

fn authored_published_ranking(
    fixture: &mut AuthoredWorkerFixture,
) -> (Registry, NativeAcceptanceIssuer, ProposerRankingPublished) {
    let input = fixture.port.root_input().unwrap();
    assert_eq!(input.stage, RefinementStage::ProposerRanking);
    let mut registry = Registry::new(
        fixture.port.authority,
        input.view.token_budget(),
        CoordinatorPolicy::Fixed,
        1,
    )
    .unwrap();
    let (issuer, verifier) = NativeAcceptanceIssuer::new_pair();
    registry
        .register_native_acceptance_verifier(verifier)
        .unwrap();
    let clock = fixture.clock.domain();
    let (ticket, output) = authored_proposer_submission(
        &mut registry,
        &input,
        fixture.port.deadline,
        "e2e4",
        clock,
        fixture.clock.now().unwrap(),
    );
    registry
        .complete_proposer_ranking(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap();
    let delivered = registry
        .take_ready(clock, fixture.clock.now().unwrap())
        .unwrap();
    assert!(delivered.logically_eligible && delivered.result.is_ok());
    assert_eq!(delivered.ticket, ticket);
    let permit = registry
        .issue_native_acceptance(&ticket, clock, fixture.clock.now().unwrap())
        .unwrap();
    registry
        .confirm_native_acceptance(
            issuer.issue_success(permit),
            clock,
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    let ranking = registry
        .accept_proposer_ranking(
            &ticket,
            fixture.port.root_rules(),
            clock,
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    assert_eq!(registry.phase(ticket.id), None);
    (registry, issuer, ranking)
}

#[test]
fn authored_worker_optional_peer_r_returns_to_original_accepted_parent() {
    let _serial = EXTERNAL_FIXTURE_LOCK.lock().unwrap();
    // P publishes the two-ply prefix; R's actual one-ply raw PV needs one
    // further paid Rules transition inside this original startup horizon.
    let mut fixture =
        AuthoredWorkerFixture::with_goal_and_pv(Duration::from_secs(20), false, 3, false);
    let clock = fixture.clock.domain();
    let root_budget = fixture
        .port
        .rovent_frontier()
        .unwrap()
        .registration()
        .root_budget;
    let (mut registry, issuer, ranking) = authored_published_ranking(&mut fixture);
    let mut candidates = fixture.port.ranking_published(&ranking).unwrap();
    assert_eq!(candidates.len(), 1);
    let candidate = candidates.pop().unwrap();
    assert_eq!(candidate.input.stage, RefinementStage::ProposalContinuation);
    assert!(candidate.initialized.is_none());
    let mut machine =
        RefinementMachine::new(fixture.port.authority, fixture.port.deadline, 512).unwrap();
    machine
        .enable_reciprocal_questions(ExecutionMode::CpuPcr, root_budget)
        .unwrap();
    let parent = machine
        .admit_candidate(
            &ranking,
            candidate.first_move,
            candidate.input.clone(),
            candidate.budget,
            clock,
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    fixture
        .port
        .root_candidate_admitted(parent, candidate.first_move, &candidate.input)
        .unwrap();
    let source_work = machine
        .ready_work(parent, clock, fixture.clock.now().unwrap())
        .unwrap();
    assert_eq!(
        source_work.input.question.context.prefix,
        [BoardMove::from_uci("e2e4").unwrap()]
    );
    let (ticket, output) = authored_proposer_submission(
        &mut registry,
        &source_work.input,
        source_work.original_deadline,
        "e7e5",
        clock,
        fixture.clock.now().unwrap(),
    );
    machine
        .bind_role(parent, &ticket, clock, fixture.clock.now().unwrap())
        .unwrap();
    registry
        .complete_reciprocal_role(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output.clone()),
        })
        .unwrap();
    let delivered = registry
        .take_ready(clock, fixture.clock.now().unwrap())
        .unwrap();
    assert!(delivered.logically_eligible && delivered.result.is_ok());
    assert_eq!(registry.phase(ticket.id), Some(TicketPhase::Delivered));
    let permit = registry
        .issue_reciprocal_native_acceptance(&ticket, clock, fixture.clock.now().unwrap())
        .unwrap();
    let accepted = registry
        .accept_reciprocal_native(
            issuer.issue_success(permit),
            clock,
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    assert!(accepted.validates_output(&output));
    assert_eq!(registry.phase(ticket.id), None);
    let parent_evidence = StepEvidence {
        frame: parent,
        stage: source_work.input.stage,
        execution: StepExecution::Role(ticket.id),
        submitted_question: source_work.input.question_digest,
        submitted_view: source_work.input.view_seal,
        completed_evidence: accepted.output_digest(),
    };
    let original_budget = machine.frame(parent).unwrap().budget;
    let prepared = fixture
        .port
        .prepare_accepted_role_reciprocal(
            &source_work,
            &accepted,
            &output,
            ReciprocalQuestionId {
                authority: fixture.port.authority,
                sequence: 1,
            },
            ExecutionMode::CpuPcr,
            root_budget,
            SearchRecipient::Researcher,
        )
        .unwrap()
        .unwrap();
    assert!(prepared.origin.matches(&accepted));
    assert_eq!(prepared.question.sender, SearchRecipient::Proposer);
    assert_eq!(prepared.question.recipient, SearchRecipient::Researcher);
    assert!(prepared.question.parent.is_none());
    assert_eq!(prepared.input.stage, RefinementStage::CpuValidation);
    assert_eq!(
        prepared.input.question.context.prefix,
        [
            BoardMove::from_uci("e2e4").unwrap(),
            BoardMove::from_uci("e7e5").unwrap(),
        ]
    );
    let peer_question = prepared.question;
    let admission = machine
        .admit_after_accepted_role(
            parent_evidence,
            &accepted,
            peer_question,
            prepared.input,
            0,
            clock,
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    let DependentFrameAdmission::Admitted(child) = admission else {
        panic!("first optional peer must receive one original child frame")
    };
    assert_eq!(machine.waiting_on(parent), Some(child));
    assert_eq!(
        machine.frame(parent).unwrap().phase,
        FramePhase::Waiting(parent_evidence.execution)
    );
    assert!(
        machine
            .ready_work(parent, clock, fixture.clock.now().unwrap())
            .is_err()
    );
    let child_work = machine
        .ready_work(child, clock, fixture.clock.now().unwrap())
        .unwrap();
    let (external, packet) = fixture.dispatch(&child_work, 2);
    assert_eq!(external.binding.question(), peer_question);
    assert_eq!(packet.question(), peer_question);
    assert_eq!(
        packet.result().as_ref().unwrap().report.raw.best_move,
        Some(BoardMove::from_uci("g1f3").unwrap())
    );
    machine
        .bind_profiled_external_cpu(
            child,
            &external.binding,
            clock,
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    let execution = StepExecution::ProfiledExternal {
        question: peer_question.id,
        execution: external.binding.execution(),
    };
    let bound = machine.bound_work(child, execution).unwrap();
    assert!(Arc::ptr_eq(
        &bound.input.question,
        &child_work.input.question
    ));
    assert!(Arc::ptr_eq(&bound.input.view, &child_work.input.view));
    let committed = fixture
        .port
        .commit_profiled_external(
            &child_work,
            &fixture.source,
            external.reservation,
            &packet,
            true,
        )
        .unwrap();
    assert!(committed.consumed);
    let proof = fixture
        .port
        .after_profiled_reciprocal_cpu(
            &child_work,
            &fixture.source,
            external.reservation,
            &packet,
            &committed,
        )
        .unwrap();
    let evidence = StepEvidence {
        frame: child,
        stage: child_work.input.stage,
        execution,
        submitted_question: child_work.input.question_digest,
        submitted_view: child_work.input.view_seal,
        completed_evidence: proof.evidence(),
    };

    let other_work = fixture.work(99);
    let (other_external, other_packet) = fixture.dispatch(&other_work, 9);
    assert!(!Arc::ptr_eq(&packet, &other_packet));
    assert!(
        fixture
            .port
            .after_profiled_reciprocal_cpu(
                &child_work,
                &fixture.source,
                external.reservation,
                &other_packet,
                &committed,
            )
            .is_err()
    );
    assert!(
        fixture
            .port
            .after_profiled_reciprocal_cpu(
                &other_work,
                &fixture.source,
                other_external.reservation,
                &packet,
                &committed,
            )
            .is_err()
    );
    let mut changed_evidence = evidence;
    changed_evidence.frame = parent;
    assert!(
        machine
            .resolve_reciprocal_profiled_external_response(
                changed_evidence,
                &proof,
                clock,
                fixture.clock.now().unwrap(),
            )
            .is_err()
    );
    assert_eq!(machine.waiting_on(parent), Some(child));
    fixture
        .port
        .commit_profiled_external(
            &other_work,
            &fixture.source,
            other_external.reservation,
            &other_packet,
            false,
        )
        .unwrap();

    let response = machine
        .resolve_reciprocal_profiled_external_response(
            evidence,
            &proof,
            clock,
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    assert_eq!(response.frame, child);
    assert_eq!(response.question, child_work.input.question_digest);
    assert_eq!(response.execution, execution);
    assert_eq!(machine.waiting_on(parent), None);
    assert!(machine.frame(child).is_none());
    assert_eq!(
        machine.frame(parent).unwrap().phase,
        FramePhase::Waiting(parent_evidence.execution)
    );
    let resumed = machine
        .bound_work(parent, parent_evidence.execution)
        .unwrap();
    assert_eq!(resumed.frame, source_work.frame);
    assert_eq!(resumed.original_deadline, source_work.original_deadline);
    assert_eq!(resumed.original_first_move, source_work.original_first_move);
    assert_eq!(resumed.budget, original_budget);
    assert!(Arc::ptr_eq(
        &resumed.input.question,
        &source_work.input.question
    ));
    assert!(Arc::ptr_eq(&resumed.input.view, &source_work.input.view));
    assert!(
        machine
            .resolve_reciprocal_profiled_external_response(
                evidence,
                &proof,
                clock,
                fixture.clock.now().unwrap(),
            )
            .is_err()
    );
    assert_eq!(machine.frame(parent).unwrap().budget, original_budget);
    assert_eq!(registry.phase(ticket.id), None);
    assert_eq!(fixture.port.counters.role_calls, 2); // Root ranking plus original P, once each.
    assert_eq!(fixture.port.counters.consumed_external_checker_tasks, 1);
    assert_eq!(fixture.port.counters.cpu_nodes, 0);
    assert_eq!(fixture.port.counters.completed_cpu_tasks, 0);
    let raw = fixture
        .port
        .engine
        .semantic
        .stores
        .observations
        .get(committed.observation)
        .unwrap();
    assert_eq!(
        fixture
            .port
            .engine
            .semantic
            .stores
            .tasks
            .get(raw.execution.unwrap())
            .unwrap()
            .status,
        TaskStatus::Failed
    );
    assert!(matches!(
        raw.scope,
        EvidenceScope::ProfiledExternalUci { .. }
    ));
    assert!(matches!(raw.score, RawScore::ExternalUci { .. }));
    assert_eq!(
        fixture
            .port
            .engine
            .semantic
            .stores
            .lines
            .moves(raw.cpu_pv.expect("peer R has its original paid PV"))
            .unwrap(),
        [BoardMove::from_uci("g1f3").unwrap()]
    );
    let fence = fixture.quiesce();
    let retirement = fixture
        .port
        .release_profiled_external_root(&fixture.source, &fence)
        .unwrap();
    assert_eq!(retirement.retained_questions(), 2);
}

#[test]
fn authored_worker_published_closed_source_survives_next_preparation_until_retirement() {
    let _serial = EXTERNAL_FIXTURE_LOCK.lock().unwrap();
    let mut fixture = AuthoredWorkerFixture::new(Duration::from_secs(20), false);
    let original = fixture.work(1);
    let (admitted, packet) = fixture.dispatch(&original, 1);
    let published = fixture
        .port
        .commit_profiled_external(
            &original,
            &fixture.source,
            admitted.reservation,
            &packet,
            false,
        )
        .unwrap();
    assert!(!published.consumed);
    let question = original.input.question_digest;
    let retained = &fixture.port.profiled_external.entries[&question];
    assert!(retained.terminal && retained.accounted);
    assert_eq!(retained.published, Some(published.observation));
    assert!(retained.consumed.is_none());
    assert!(Arc::ptr_eq(retained.completion.as_ref().unwrap(), &packet));
    let execution = retained.execution;
    let original_dispatch = retained.dispatch.as_ref().unwrap().clone();

    // Semantic branch closure is not a durable packet/view retirement ACK.
    // An explicit Unknown closes the branch without inventing P/C acceptance,
    // completed native depth, a mate proof, or a refinement completion scope.
    fixture
        .port
        .publish_closed(&crate::pals::coordinator::ClosedFrame {
            id: original.frame,
            original_first_move: original.original_first_move,
            evidence: Vec::new(),
            unknown: Some("authored fixture branch closure without refinement proof"),
            completion_scope: None,
        })
        .unwrap();
    assert!(fixture.port.closed.contains(&original.frame));
    assert!(!fixture.port.profiled_external.retirement_issued);

    // One actual worker packet suffices: the next question is prepared only,
    // not submitted. No additional worker run or synthetic admission rows are
    // needed to expose the old terminal+published+closed retain predicate.
    let next = fixture.work(2);
    assert_ne!(next.input.question_digest, question);
    let prepared = fixture
        .port
        .prepare_profiled_external(
            &next,
            &fixture.config,
            &fixture.source,
            ReciprocalQuestionId {
                authority: fixture.port.authority,
                sequence: 2,
            },
            rz_contracts::ExecutionId::new(fixture.port.authority.epoch, 2),
            ExternalGoConstraint::Nodes { reported_nodes: 16 },
            Duration::from_millis(100),
            CompletionSignal::default(),
            fixture.clock.domain(),
            fixture.clock.now().unwrap(),
        )
        .unwrap();
    assert!(matches!(
        prepared,
        AdaptiveProfiledExternalPreparation::Ready(_)
    ));
    assert_eq!(fixture.port.profiled_external.entries.len(), 2);
    let retained = &fixture.port.profiled_external.entries[&question];
    assert!(retained.matches(&original));
    assert!(retained.source.same_source(&fixture.source));
    assert!(Arc::ptr_eq(
        &retained.work.input.question,
        &original.input.question
    ));
    assert!(Arc::ptr_eq(&retained.work.input.view, &original.input.view));
    assert!(Arc::ptr_eq(retained.completion.as_ref().unwrap(), &packet));
    assert_eq!(retained.execution, execution);
    assert_eq!(retained.reservation, Some(admitted.reservation));
    assert_eq!(retained.published, Some(published.observation));
    let dispatch = retained.dispatch.as_ref().unwrap();
    assert!(packet.matches_submission(dispatch));
    assert_eq!(dispatch.question(), original_dispatch.question());
    assert_eq!(dispatch.scope(), original_dispatch.scope());
    assert_eq!(
        dispatch.process_identity(),
        original_dispatch.process_identity()
    );
    assert_eq!(dispatch.constraint(), original_dispatch.constraint());
    assert_eq!(
        dispatch.root_restriction(),
        original_dispatch.root_restriction()
    );
    assert_eq!(
        fixture
            .port
            .engine
            .semantic
            .stores
            .tasks
            .get(execution)
            .unwrap()
            .status,
        TaskStatus::Failed
    );
    assert!(
        fixture
            .port
            .engine
            .semantic
            .stores
            .observations
            .get(published.observation)
            .is_ok()
    );
    assert_eq!(fixture.port.counters.external_checker_tasks, 1);
    assert_eq!(fixture.port.counters.external_checker_reports, 1);
    assert_eq!(fixture.port.counters.consumed_external_checker_tasks, 0);
    assert_eq!(fixture.port.counters.cpu_nodes, 0);
    assert_eq!(fixture.port.counters.role_calls, 0);

    // A definite rejection settles the unsubmitted task without fabricating a
    // second completion. The actual root drain then transfers both original
    // rows into move-only retirement custody rather than dropping the first.
    fixture.port.reject_profiled_external(&next).unwrap();
    let fence = fixture.quiesce();
    let retirement = fixture
        .port
        .release_profiled_external_root(&fixture.source, &fence)
        .unwrap();
    assert_eq!(retirement.retained_questions(), 2);
    let retired = retirement
        .packets()
        .find(|retired| retired.work().frame == original.frame)
        .unwrap();
    assert!(Arc::ptr_eq(retired.completion().unwrap(), &packet));
    assert!(Arc::ptr_eq(
        &retired.work().input.question,
        &original.input.question
    ));
    assert!(Arc::ptr_eq(
        &retired.work().input.view,
        &original.input.view
    ));
    assert_eq!(retired.observation(), Some(published.observation));
    assert!(!retired.search_consumed());
    assert!(
        retirement
            .protected_pins()
            .unwrap()
            .executions
            .contains(&execution)
    );
}
