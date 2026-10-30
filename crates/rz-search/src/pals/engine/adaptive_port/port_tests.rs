//! Contract fixtures exercise the actual semantic port. No physical native NN
//! completion or engine performance is inferred from these authored outputs.
use super::*;
use crate::pals::coordinator::{
    CoordinatorError, FrameId, FramePhase, NativeAcceptanceIssuer, PhysicalRoleCompletion,
    ProposerRankingPublished, RefinementMachine, RoleCompletion, RoleSubmission,
    RoleTicketRegistry,
};
use crate::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
use rz_contracts::pals::RepresentationKey;
use rz_contracts::{ClockDomain, GameGeneration, MonotonicTick, ProcessEpoch, RootGeneration};
use rz_runtime::adaptive::CoordinatorPolicy;

type Port = AdaptiveEnginePort<LegalOrderRoleMock>;
type Work = CoordinatorWork<ReviewView, AdaptivePortQuestion>;
pub(super) fn setup(position: Position, line_plies: usize) -> Port {
    setup_with_resolver(position, line_plies, ResolverPolicy::OwnRawRestricted)
}
pub(super) fn setup_with_resolver(
    position: Position,
    line_plies: usize,
    resolver: ResolverPolicy,
) -> Port {
    setup_with_model(position, line_plies, resolver, LegalOrderRoleMock)
}
pub(super) fn setup_with_model<M: RoleModel>(
    position: Position,
    line_plies: usize,
    resolver: ResolverPolicy,
    model: M,
) -> AdaptiveEnginePort<M> {
    let cpu = CpuEngine::new(crate::cpu::CpuConfig {
        tt_entries: 16,
        max_depth: 8,
        ..Default::default()
    })
    .unwrap();
    let config = PalsConfig {
        line_plies,
        ..Default::default()
    };
    let engine = match resolver {
        // Keep the original fixture's legacy followup-lane/default identity.
        ResolverPolicy::OwnRawRestricted => PalsEngine::new(config, model, cpu),
        ResolverPolicy::ModelWdlRestricted => PalsEngine::new_with_boxed_checker_and_policies(
            config,
            model,
            Box::new(OwnedCpuChecker::new(Box::new(cpu)).unwrap()),
            resolver,
            PostRepairRecheckPolicy::Disabled,
        ),
    }
    .unwrap()
    .with_selected_review_context(
        ReviewSelectionPolicy::default(),
        ReviewIndexLimits::default(),
    )
    .unwrap();
    let authority = SearchAuthority {
        epoch: ProcessEpoch(23),
        game: GameGeneration(0),
        root: RootGeneration(1),
        implementation: ContractDigest([7; 32]),
    };
    let next = Arc::new(std::sync::atomic::AtomicU64::new(1));
    let issuer: AdaptiveRulesOwnerIssuer =
        Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::Relaxed))));
    AdaptiveEnginePort::new(
        engine,
        position,
        authority,
        Deadline {
            clock: ClockDomain(authority.epoch),
            at: MonotonicTick(20_000_000_000),
        },
        PalsLimits {
            deadline: Instant::now() + Duration::from_secs(20),
            max_rounds: 2,
            max_cpu_nodes: 2000,
            cpu_depth: 1,
        },
        issuer,
    )
    .unwrap()
}
pub(super) fn work(port: &mut Port, first: &str, stage: RefinementStage, sequence: u64) -> Work {
    let first = BoardMove::from_uci(first).unwrap();
    let node = port.connect_hot(port.root, first).unwrap();
    if !matches!(
        purpose_for_stage(stage),
        RoleQueryPurpose::ProposePolicy | RoleQueryPurpose::ValueFresh
    ) {
        // These authored dependent questions review a Rules-checked proposal
        // already in public memory. This publication is not a CPU/NN receipt.
        port.record_hot(RecordKind::Proposal, &[first], None)
            .unwrap();
    }
    let input = port
        .input(
            node,
            stage,
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
    Work {
        frame: FrameId {
            authority: port.authority,
            sequence,
        },
        original_first_move: rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(first).unwrap(),
        ),
        original_deadline: port.deadline,
        budget: FrameBudget {
            original_cpu_credit: port.limits.max_cpu_nodes,
            role_submissions_remaining: port.engine.config.max_role_calls,
            original_store_byte_limit: 8 * 1024 * 1024,
        },
        input,
    }
}
pub(super) fn output(work: &Work, id: u64) -> (RoleTicketEnvelope, RoleOutput) {
    let purpose = work.input.question.context.purpose;
    let role = role_for_purpose(purpose);
    let key = RepresentationKey {
        input: ContractDigest(work.input.view.digest()),
        history: ContractDigest([2; 32]),
        candidates: ContractDigest([3; 32]),
        mask_positions: ContractDigest([4; 32]),
        model: ContractDigest([5; 32]),
        encoding: ContractDigest([6; 32]),
        precision: rz_contracts::PrecisionProfile::Fp32,
        frozen_epoch: 1,
        record_revision: work.input.question.context.public_revision,
        role,
    };
    let ticket = RoleTicketEnvelope {
        id: RequestId::new(work.frame.authority.epoch, id),
        authority: work.frame.authority,
        question: work.input.question_digest,
        view: work.input.view_seal,
        representation: key,
        role,
        original_deadline: work.original_deadline,
    };
    let logits = vec![0.0; work.input.view.query().legal.len()];
    let wdl = rz_contracts::Wdl::try_new(0.25, 0.5, 0.25, 0.0).unwrap();
    let payload = match role {
        Role::Proposer => RolePayload::Proposal {
            candidate_logits: logits,
            wdl,
        },
        Role::Critic => RolePayload::Counterexample {
            candidate_logits: logits,
            divergence_logits: vec![0.0; work.input.view.query().divergences.len()],
            wdl,
        },
        Role::Validator => RolePayload::TaskRanking {
            task_logits: vec![0.0; 7],
        },
    };
    let output = RoleOutput {
        id: ticket.id,
        authority: ticket.authority,
        key,
        payload,
        private_latent: vec![0.0; 16 * 384],
    };
    (ticket, output)
}

/// This CPU/mock owner crosses the real linear registry acceptance boundary.
/// Its authored output is not native neural execution or GPU acceptance proof.
pub(in crate::pals::engine::adaptive_port) fn accepted_root_ranking<M: RoleModel>(
    port: &mut AdaptiveEnginePort<M>,
) -> (
    ProposerRankingPublished,
    RoleTicketRegistry<ReviewView, RoleOutput>,
) {
    let (ranking, registry, _) = accepted_root_ranking_with_input(port);
    (ranking, registry)
}

fn accepted_root_ranking_with_input<M: RoleModel>(
    port: &mut AdaptiveEnginePort<M>,
) -> (
    ProposerRankingPublished,
    RoleTicketRegistry<ReviewView, RoleOutput>,
    FrameInput<ReviewView, AdaptivePortQuestion>,
) {
    let input = port.root_input().unwrap();
    assert_eq!(input.stage, RefinementStage::ProposerRanking);
    let mut ordered_candidates = Sha256::new();
    for movement in &input.view.query().legal {
        let encoded =
            rz_contracts::pals::Move16::encode(rz_contracts::Move::try_from(*movement).unwrap());
        ordered_candidates.update(encoded.bits().to_le_bytes());
    }
    let identity = port.engine.semantic.model.value_identity().unwrap();
    let key = RepresentationKey {
        input: ContractDigest(input.view.digest()),
        history: input.view_seal.root_state.semantic,
        candidates: ContractDigest(ordered_candidates.finalize().into()),
        mask_positions: ContractDigest(
            Sha256::digest(vec![1u8; input.view.query().legal.len()]).into(),
        ),
        model: ContractDigest(Sha256::digest(identity.model.as_bytes()).into()),
        encoding: ContractDigest(Sha256::digest(identity.encoding.as_bytes()).into()),
        precision: rz_contracts::PrecisionProfile::Fp32,
        frozen_epoch: 0,
        record_revision: input.question.context.public_revision,
        role: Role::Proposer,
    };
    let mut registry =
        RoleTicketRegistry::new(port.authority, 512, CoordinatorPolicy::Fixed, 1).unwrap();
    let (issuer, verifier) = NativeAcceptanceIssuer::new_pair();
    registry
        .register_native_acceptance_verifier(verifier)
        .unwrap();
    let ticket = registry
        .admit(
            RoleSubmission {
                question: input.question_digest,
                view: input.view_seal,
                representation: key,
                role: Role::Proposer,
                original_deadline: port.deadline,
            },
            Arc::clone(&input.view),
            port.deadline.clock,
            MonotonicTick(0),
        )
        .unwrap();
    let dispatched = registry.take_dispatch().unwrap();
    assert_eq!(dispatched.ticket, ticket);
    registry
        .complete_proposer_ranking(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(RoleOutput {
                id: ticket.id,
                authority: ticket.authority,
                key,
                payload: RolePayload::Proposal {
                    candidate_logits: (0..input.view.query().legal.len())
                        .map(|index| -(index as f32))
                        .collect(),
                    wdl: rz_contracts::Wdl::try_new(0.25, 0.5, 0.25, 0.0).unwrap(),
                },
                private_latent: vec![0.0; 16 * 384],
            }),
        })
        .unwrap();
    let delivered = registry
        .take_ready(port.deadline.clock, MonotonicTick(1))
        .unwrap();
    assert!(delivered.logically_eligible);
    assert_eq!(delivered.ticket, ticket);
    assert!(delivered.result.is_ok());
    let permit = registry
        .issue_native_acceptance(&ticket, port.deadline.clock, MonotonicTick(2))
        .unwrap();
    registry
        .confirm_native_acceptance(
            issuer.issue_success(permit),
            port.deadline.clock,
            MonotonicTick(2),
        )
        .unwrap();
    let ranking = registry
        .accept_proposer_ranking(
            &ticket,
            port.root_rules(),
            port.deadline.clock,
            MonotonicTick(3),
        )
        .unwrap();
    assert_eq!(ranking.source().ticket.representation, key);
    assert_eq!(ranking.source().ordered_candidate_digest, key.candidates);
    (ranking, registry, input)
}

#[test]
fn root_ranking_initial_fanout_keeps_consumed_scope_when_protection_reorders_selection() {
    for legacy_full in [false, true] {
        let mut port = setup(Position::startpos(), 3);
        port.engine.semantic.legacy_full_native_context = legacy_full;
        let proposal = vec![
            BoardMove::from_uci("e2e4").unwrap(),
            BoardMove::from_uci("e7e5").unwrap(),
        ];
        let counter = vec![proposal[0], BoardMove::from_uci("c7c5").unwrap()];
        port.record_hot(RecordKind::Proposal, &proposal, None)
            .unwrap();
        port.record_hot(RecordKind::Counterexample, &counter, None)
            .unwrap();
        let bundle = port
            .engine
            .semantic
            .review_index
            .as_ref()
            .unwrap()
            .bundle_for_revision(port.engine.semantic.revision)
            .unwrap();
        let input = port
            .input(
                port.root,
                RefinementStage::CriticDivergence,
                Some(proposal[0]),
                Vec::new(),
                proposal,
                None,
                None,
                None,
                false,
                None,
                None,
            )
            .unwrap();
        let prior = Work {
            frame: FrameId {
                authority: port.authority,
                sequence: 100,
            },
            original_first_move: rz_contracts::pals::Move16::encode(
                rz_contracts::Move::try_from(counter[0]).unwrap(),
            ),
            original_deadline: port.deadline,
            budget: FrameBudget {
                original_cpu_credit: port.limits.max_cpu_nodes,
                role_submissions_remaining: port.engine.config.max_role_calls,
                original_store_byte_limit: 8 * 1024 * 1024,
            },
            input,
        };
        let (ticket, value) = output(&prior, 100);
        port.commit_role(&prior, &ticket, &value, None).unwrap();
        port.role_consumed(&ticket, &prior.input.view).unwrap();
        let protected = |port: &Port| {
            port.engine
                .semantic
                .review_index
                .as_ref()
                .unwrap()
                .bundle(bundle)
                .unwrap()
                .protected_consumptions()
        };
        assert_eq!(protected(&port), 1);
        // This newer unprotected bundle follows the protected old bundle in
        // the root input. The actual ranking consume expires that protection.
        port.record_hot(
            RecordKind::Proposal,
            &[BoardMove::from_uci("d2d4").unwrap()],
            None,
        )
        .unwrap();
        let consumed_before = port.counters.consumed_role_outputs;
        let calls_before = port.counters.role_calls;
        let (ranking, mut registry, ranking_input) = accepted_root_ranking_with_input(&mut port);
        let source_scope = ContractDigest(ranking.view().record_scope_digest());
        let source_metadata = ranking.view().records_metadata().to_vec();
        let source_lines: Vec<_> = ranking
            .view()
            .records()
            .iter()
            .map(|record| (record.revision, record.line.clone()))
            .collect();
        let candidates = port.ranking_published(&ranking).unwrap();
        assert!(!candidates.is_empty());
        assert_eq!(protected(&port), 0);
        let first_input = &candidates[0].input;
        let query = first_input.view.query();
        let selected_again = port
            .engine
            .prepare_review_view(
                first_input.question.node,
                RoleQuestion {
                    purpose: query.purpose,
                    prefix: &query.prefix,
                    proposal: &query.proposal,
                    refutation: query.counterexample.as_deref(),
                    divergences: &query.divergences,
                },
                &query.legal,
            )
            .unwrap()
            .unwrap();
        let changed_scope = ContractDigest(selected_again.record_scope_digest());
        assert_ne!(
            changed_scope, source_scope,
            "fixture must cross actual selector ordering"
        );
        let mut machine = RefinementMachine::new(port.authority, port.deadline, 512).unwrap();
        let candidate_count = candidates.len();
        for candidate in candidates {
            assert_eq!(candidate.input.public_context, source_scope);
            assert_eq!(candidate.input.view.record_scope_digest(), source_scope.0);
            assert_eq!(candidate.input.view.records_metadata(), source_metadata);
            assert_eq!(
                candidate
                    .input
                    .view
                    .records()
                    .iter()
                    .map(|record| (record.revision, record.line.clone()))
                    .collect::<Vec<_>>(),
                source_lines,
            );
            assert!(candidate.input.view.validates(&port.engine.semantic.stores));
            assert_eq!(
                candidate.input.view.generation(),
                ranking.view().generation()
            );
            assert_eq!(
                candidate.input.view.root_state(),
                ranking.view().root_state()
            );
            assert_ne!(
                candidate.input.question_digest,
                ranking_input.question_digest
            );
            assert_ne!(candidate.input.view.digest(), ranking.view().digest());
            assert_eq!(candidate.input.view.query().prefix.len(), 1);
            assert_eq!(
                candidate.budget.role_submissions_remaining,
                port.engine.config.max_role_calls - port.counters.role_calls
            );
            match (
                &ranking_input.question.legacy_full_records,
                &candidate.input.question.legacy_full_records,
            ) {
                (Some(original), Some(current)) => {
                    assert!(Arc::ptr_eq(original, current));
                    assert!(current.validates(&port.engine.semantic.stores));
                }
                (None, None) => assert!(!legacy_full),
                _ => panic!("initial query changed the independent legacy raw namespace"),
            }
            let mut wrong_scope = candidate.input.clone();
            wrong_scope.public_context = changed_scope;
            let before = machine.history_usage();
            assert_eq!(
                machine.admit_candidate(
                    &ranking,
                    candidate.first_move,
                    wrong_scope,
                    candidate.budget,
                    port.deadline.clock,
                    MonotonicTick(4),
                ),
                Err(CoordinatorError::IdentityMismatch(
                    "published root ranking/candidate question scope",
                )),
            );
            assert_eq!(machine.history_usage(), before);
            let frame = machine
                .admit_candidate(
                    &ranking,
                    candidate.first_move,
                    candidate.input.clone(),
                    candidate.budget,
                    port.deadline.clock,
                    MonotonicTick(4),
                )
                .unwrap();
            port.root_candidate_admitted(frame, candidate.first_move, &candidate.input)
                .unwrap();
            assert_eq!(
                machine.frame(frame).unwrap().ranking_source(),
                Some(ranking.source())
            );
        }
        assert_eq!(machine.history_usage().ranking_fanouts, candidate_count);
        assert_eq!(machine.history_usage().role_requests, 1);
        assert_eq!(port.counters.role_calls, calls_before + 1);
        assert_eq!(port.counters.consumed_role_outputs, consumed_before + 1);
        assert_eq!(registry.next_request_sequence(), 2);
        assert!(
            registry.take_dispatch().is_none(),
            "forking invented a forward"
        );
    }
}

#[test]
fn root_ranking_full_frozen_scope_rejects_extra_query_tokens_without_reselection() {
    let mut port = setup(Position::startpos(), 4);
    port.engine
        .semantic
        .review_policy
        .as_mut()
        .unwrap()
        .token_budget = 128;
    let proposal = [
        BoardMove::from_uci("e2e4").unwrap(),
        BoardMove::from_uci("e7e5").unwrap(),
        BoardMove::from_uci("g1f3").unwrap(),
        BoardMove::from_uci("b8c6").unwrap(),
    ];
    // Fill one original coherent bundle with actual Rules-checked revisions.
    // No raw evidence is truncated to make the candidate's extra token fit.
    for _ in 0..32 {
        port.record_hot(RecordKind::Proposal, &proposal, None)
            .unwrap();
    }
    let revisions: Vec<_> = port
        .engine
        .semantic
        .records
        .iter()
        .map(|record| record.revision)
        .collect();
    let (ranking, mut registry, _input) = accepted_root_ranking_with_input(&mut port);
    assert_eq!(ranking.view().records().len(), 32);
    assert_eq!(ranking.view().token_count(), 128);
    let source_scope = ranking.view().record_scope_digest();
    assert!(matches!(
        port.ranking_published(&ranking),
        Err(PalsError::Review(ReviewError::ViewCapacity))
    ));
    assert_eq!(port.counters.role_calls, 1);
    assert_eq!(port.counters.consumed_role_outputs, 1);
    assert!(
        port.role_consumption_recorded(&ranking.source().ticket, ranking.view())
            .unwrap()
    );
    assert!(port.round_candidates.is_empty());
    assert!(port.lazy_frontier.is_empty());
    assert_eq!(
        port.engine
            .semantic
            .records
            .iter()
            .map(|record| record.revision)
            .collect::<Vec<_>>(),
        revisions
    );
    assert_eq!(ranking.view().record_scope_digest(), source_scope);
    assert!(ranking.view().validates(&port.engine.semantic.stores));
    assert_eq!(registry.next_request_sequence(), 2);
    assert!(registry.take_dispatch().is_none());
}

#[test]
fn root_ranking_spends_the_only_call_without_issuing_zero_credit_candidates() {
    let mut port = setup(Position::startpos(), 3);
    port.engine.config.max_role_calls = 1;
    port.configure_root_fanout(AdaptiveRootFanoutPolicy::SequentialFixed)
        .unwrap();
    let (ranking, _) = accepted_root_ranking(&mut port);
    assert!(port.ranking_published(&ranking).unwrap().is_empty());
    assert_eq!(port.counters.role_calls, 1);
    assert_eq!(port.counters.consumed_role_outputs, 1);
    assert!(port.next_fixed_root_candidate().unwrap().is_none());
    assert!(port.next_root_round_input().unwrap().is_none());
    assert!(port.lazy_frontier.is_empty());
    assert!(port.round_candidates.is_empty());
}

#[test]
fn root_frame_ceiling_excludes_ranking_and_final_reserved_role_still_commits() {
    let mut port = setup(Position::startpos(), 3);
    port.engine.config.max_role_calls = 2;
    port.engine.config.beam_width = 2;
    let (ranking, _) = accepted_root_ranking(&mut port);
    let candidates = port.ranking_published(&ranking).unwrap();
    assert_eq!(candidates.len(), 2);
    let works = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| {
            assert_eq!(candidate.budget.role_submissions_remaining, 1);
            Work {
                frame: FrameId {
                    authority: port.authority,
                    sequence: index as u64 + 1,
                },
                original_first_move: candidate.first_move,
                original_deadline: port.deadline,
                input: candidate.input,
                budget: candidate.budget,
            }
        })
        .collect::<Vec<_>>();
    let (ticket, result) = output(&works[0], 2);
    port.commit_role(&works[0], &ticket, &result, None).unwrap();
    assert_eq!(port.counters.role_calls, 2);
    assert_eq!(port.counters.consumed_role_outputs, 2);
    // This is a genuine unreserved overrun, retained as an error. The manager
    // regression separately verifies it cannot be submitted in production.
    let (overrun, result) = output(&works[1], 3);
    assert!(matches!(
        port.commit_role(&works[1], &overrun, &result, None),
        Err(PalsError::RoleCallLimit)
    ));
    assert_eq!(port.counters.role_calls, 2);
    assert_eq!(port.counters.consumed_role_outputs, 2);
    assert!(!port.accepted.contains(&overrun.id));
}

#[test]
fn actual_one_ply_ranking_enters_cpu_without_another_policy_forward() {
    let mut port = setup(Position::startpos(), 1);
    let (ranking, mut registry) = accepted_root_ranking(&mut port);
    let mut candidates = port.ranking_published(&ranking).unwrap();
    assert_eq!(candidates.len(), port.engine.config.beam_width);
    let mut machine = RefinementMachine::new(port.authority, port.deadline, 512).unwrap();
    for candidate in &mut candidates {
        assert_eq!(candidate.input.stage, RefinementStage::CpuValidation);
        assert_eq!(candidate.input.view.query().prefix.len(), 1);
        assert_eq!(
            candidate.input.view.query().proposal,
            candidate.input.view.query().prefix
        );
        let proof = candidate
            .initialized
            .take()
            .expect("registered one-ply proof");
        proof.validate(&ranking).unwrap();
        let frame = machine
            .admit_checked_candidate(&ranking, proof, port.deadline.clock, MonotonicTick(4))
            .unwrap();
        port.root_candidate_admitted(frame, candidate.first_move, &candidate.input)
            .unwrap();
        let admitted = machine.frame(frame).unwrap();
        assert_eq!(admitted.phase, FramePhase::Ready);
        assert_eq!(admitted.input.stage, RefinementStage::CpuValidation);
        assert_eq!(admitted.ranking_source(), Some(ranking.source()));
    }
    assert_eq!(machine.history_usage().role_requests, 1);
    assert_eq!(machine.history_usage().cpu_executions, 0);
    assert_eq!(machine.history_usage().ranking_fanouts, candidates.len());
    assert_eq!(port.counters.role_calls, 1);
    assert_eq!(port.counters.proposer_calls, 1);
    assert_eq!(port.counters.consumed_role_outputs, 1);
    assert_eq!(port.counters.cpu_tasks_requested, 0);
    assert_eq!(registry.next_request_sequence(), 2);
    assert!(
        registry.take_dispatch().is_none(),
        "candidate fanout invented another forward"
    );
}

#[test]
fn longer_registered_proposals_keep_actual_policy_continuation_entry() {
    let mut port = setup(Position::startpos(), 3);
    let (ranking, mut registry) = accepted_root_ranking(&mut port);
    let candidates = port.ranking_published(&ranking).unwrap();
    let mut machine = RefinementMachine::new(port.authority, port.deadline, 512).unwrap();
    for candidate in candidates {
        assert!(candidate.initialized.is_none());
        assert_eq!(candidate.input.stage, RefinementStage::ProposalContinuation);
        assert!(candidate.input.view.query().proposal.is_empty());
        let frame = machine
            .admit_candidate(
                &ranking,
                candidate.first_move,
                candidate.input.clone(),
                candidate.budget,
                port.deadline.clock,
                MonotonicTick(4),
            )
            .unwrap();
        port.root_candidate_admitted(frame, candidate.first_move, &candidate.input)
            .unwrap();
        assert_eq!(
            machine.frame(frame).unwrap().input.stage,
            RefinementStage::ProposalContinuation
        );
    }
    assert_eq!(machine.history_usage().role_requests, 1);
    assert_eq!(port.counters.role_calls, 1);
    assert_eq!(port.counters.proposals, 0);
    assert_eq!(registry.next_request_sequence(), 2);
    assert!(registry.take_dispatch().is_none());
}

#[test]
fn same_actual_one_ply_source_and_first_move_cannot_be_admitted_twice() {
    let mut port = setup(Position::startpos(), 1);
    let (ranking, registry) = accepted_root_ranking(&mut port);
    let mut candidates = port.ranking_published(&ranking).unwrap();
    let candidate = candidates.first_mut().unwrap();
    let proof = candidate.initialized.take().unwrap();
    // Replay only the fields of the actual issued proof; no changed source,
    // view, profile or first move is manufactured for a positive assertion.
    let replay = CheckedRootCandidate {
        source: proof.source,
        first: proof.first,
        input: proof.input.clone(),
        budget: proof.budget,
        line_plies: proof.line_plies,
    };
    let mut machine = RefinementMachine::new(port.authority, port.deadline, 512).unwrap();
    machine
        .admit_checked_candidate(&ranking, proof, port.deadline.clock, MonotonicTick(4))
        .unwrap();
    let before = machine.history_usage();
    assert_eq!(
        machine.admit_checked_candidate(&ranking, replay, port.deadline.clock, MonotonicTick(4)),
        Err(CoordinatorError::Duplicate)
    );
    assert_eq!(machine.history_usage(), before);
    assert_eq!(machine.frames().count(), 1);
    assert_eq!(port.counters.consumed_role_outputs, 1);
    assert_eq!(registry.next_request_sequence(), 2);
    assert!(
        port.ranking_published(&ranking).is_err(),
        "accepted root ranking was consumed twice"
    );
    assert_eq!(port.counters.role_calls, 1);
}

#[test]
fn independently_pending_roles_keep_their_original_view_after_publication() {
    let mut port = setup(Position::startpos(), 2);
    let first = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
    let second = work(&mut port, "d2d4", RefinementStage::ProposalContinuation, 2);
    let second_digest = second.input.view.digest();
    let second_revision = second.input.question.context.public_revision;
    let (a, out_a) = output(&first, 1);
    let (b, out_b) = output(&second, 2);
    port.commit_role(&first, &a, &out_a, None).unwrap();
    port.role_consumed(&a, &first.input.view).unwrap();
    assert!(port.engine.semantic.revision > second_revision);
    port.validate_role(&second, &b, &out_b).unwrap();
    port.commit_role(&second, &b, &out_b, None).unwrap();
    port.role_consumed(&b, &second.input.view).unwrap();
    assert_eq!(second.input.view.digest(), second_digest);
    assert_eq!(
        second.input.question.context.public_revision,
        second_revision
    );
    assert_eq!(port.counters.consumed_role_outputs, 2);
    assert!(port.commit_role(&first, &a, &out_a, None).is_err());
}

#[test]
fn recorded_role_consumption_is_read_only_and_survives_later_progress_or_deadline() {
    let mut port = setup(Position::startpos(), 2);
    let proposal = vec![
        BoardMove::from_uci("e2e4").unwrap(),
        BoardMove::from_uci("e7e5").unwrap(),
    ];
    let counter = vec![proposal[0], BoardMove::from_uci("c7c5").unwrap()];
    port.record_hot(RecordKind::Proposal, &proposal, None)
        .unwrap();
    port.record_hot(RecordKind::Counterexample, &counter, None)
        .unwrap();
    let witness_revision = port.engine.semantic.revision;
    let bundle = port
        .engine
        .semantic
        .review_index
        .as_ref()
        .unwrap()
        .bundle_for_revision(witness_revision)
        .unwrap();
    let protected = |port: &Port| {
        port.engine
            .semantic
            .review_index
            .as_ref()
            .unwrap()
            .bundle(bundle)
            .unwrap()
            .protected_consumptions()
    };
    assert_eq!(protected(&port), 2);
    let input = port
        .input(
            port.root,
            RefinementStage::CriticDivergence,
            Some(proposal[0]),
            Vec::new(),
            proposal.clone(),
            None,
            None,
            None,
            false,
            None,
            None,
        )
        .unwrap();
    let original = Work {
        frame: FrameId {
            authority: port.authority,
            sequence: 1,
        },
        original_first_move: rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(proposal[0]).unwrap(),
        ),
        original_deadline: port.deadline,
        budget: FrameBudget {
            original_cpu_credit: port.limits.max_cpu_nodes,
            role_submissions_remaining: port.engine.config.max_role_calls,
            original_store_byte_limit: 8 * 1024 * 1024,
        },
        input,
    };
    let (ticket, out) = output(&original, 1);
    assert!(
        !port
            .role_consumption_recorded(&ticket, &original.input.view)
            .unwrap()
    );
    port.commit_role(&original, &ticket, &out, None).unwrap();
    assert!(
        !port
            .role_consumption_recorded(&ticket, &original.input.view)
            .unwrap()
    );
    port.role_consumed(&ticket, &original.input.view).unwrap();
    assert_eq!(protected(&port), 1);
    assert!(
        port.role_consumption_recorded(&ticket, &original.input.view)
            .unwrap()
    );

    let mut wrong_question = ticket;
    wrong_question.question = ContractDigest([99; 32]);
    assert!(matches!(
        port.role_consumption_recorded(&wrong_question, &original.input.view),
        Err(PalsError::Role(RoleError::InvalidOutput))
    ));
    let other = work(&mut port, "d2d4", RefinementStage::ProposalContinuation, 2);
    assert!(matches!(
        port.role_consumption_recorded(&ticket, &other.input.view),
        Err(PalsError::Role(RoleError::InvalidOutput))
    ));
    let revision_before = port.engine.semantic.revision;
    let (other_ticket, other_out) = output(&other, 2);
    port.commit_role(&other, &other_ticket, &other_out, None)
        .unwrap();
    port.role_consumed(&other_ticket, &other.input.view)
        .unwrap();
    assert!(port.engine.semantic.revision > revision_before);
    assert!(
        port.role_consumption_recorded(&ticket, &original.input.view)
            .unwrap()
    );

    // An expired current search cannot erase historical recorded consumption.
    // This authored clock condition is not wall-clock performance evidence.
    port.limits.deadline = Instant::now();
    assert!(matches!(
        port.check_live(),
        Err(PalsError::Role(RoleError::Deadline))
    ));
    let counters = port.counters;
    let observations = port.engine.semantic.stores.observations.len();
    let pins = port.engine.semantic.stores.review_pin_count().unwrap();
    let remaining = protected(&port);
    let revision = port.engine.semantic.revision;
    for _ in 0..3 {
        assert!(
            port.role_consumption_recorded(&ticket, &original.input.view)
                .unwrap()
        );
        assert_eq!(protected(&port), remaining);
        assert_eq!(port.counters, counters);
        assert_eq!(port.engine.semantic.stores.observations.len(), observations);
        assert_eq!(
            port.engine.semantic.stores.review_pin_count().unwrap(),
            pins
        );
        assert_eq!(port.engine.semantic.revision, revision);
    }
}
#[test]
fn wrong_question_and_mutated_focus_are_rejected_before_commit() {
    let mut port = setup(Position::startpos(), 2);
    let original = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
    let (mut ticket, out) = output(&original, 1);
    ticket.question = ContractDigest([99; 32]);
    assert!(port.validate_role(&original, &ticket, &out).is_err());
    let (ticket, out) = output(&original, 1);
    let mut altered = Work {
        frame: original.frame,
        original_first_move: original.original_first_move,
        original_deadline: original.original_deadline,
        budget: original.budget,
        input: original.input.clone(),
    };
    Arc::make_mut(&mut altered.input.question)
        .context
        .focus_sha256 = [99; 32];
    assert!(port.validate_role(&altered, &ticket, &out).is_err());
    assert_eq!(port.counters.consumed_role_outputs, 0);
}
#[test]
fn fresh_counter_rejects_changed_namespace_before_any_hot_publication() {
    let mut port = setup(Position::startpos(), 2);
    let mut pending = work(&mut port, "e2e4", RefinementStage::FreshCounterEndpoint, 1);
    let (mut ticket, mut out) = output(&pending, 1);
    let original = FreshPairSeal::from_ticket(
        &ticket,
        pending.input.public_context,
        port.engine.semantic.nodes[port.root]
            .position
            .side_to_move()
            .into(),
    );
    let query = Arc::make_mut(&mut pending.input.question);
    query.paired_fresh_seal = Some(original);
    // This test exercises the pre-publication guard, not NN completion or the
    // existence of a comparison source observation.
    query.paired_repaired = Some(ObservationId(0));
    port.validate_role(&pending, &ticket, &out).unwrap();
    let observations = port.engine.semantic.stores.observations.len();
    for changed in ["model", "encoding", "epoch", "revision"] {
        ticket.representation = out.key;
        match changed {
            "model" => ticket.representation.model = ContractDigest([99; 32]),
            "encoding" => ticket.representation.encoding = ContractDigest([99; 32]),
            "epoch" => ticket.representation.frozen_epoch += 1,
            _ => ticket.representation.record_revision += 1,
        }
        let saved = out.key;
        out.key = ticket.representation;
        assert!(
            port.validate_role(&pending, &ticket, &out).is_err(),
            "{changed}"
        );
        out.key = saved;
    }
    assert_eq!(port.engine.semantic.stores.observations.len(), observations);
    assert_eq!(port.counters.consumed_role_outputs, 0);
}
#[test]
fn mate_endpoint_uses_actual_rules_without_cpu_or_nn_receipt() {
    let mut port = setup(
        Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap(),
        2,
    );
    let work = work(&mut port, "f7g7", RefinementStage::CpuValidation, 1);
    let terminal = port.short_circuit(&work).unwrap().unwrap();
    assert!(terminal.endpoint.terminal_wdl().is_some());
    assert!(matches!(
        terminal.checked.transition,
        AdaptivePortTransition::Complete {
            scope: FrameCompletionScope::NoDivergence
        }
    ));
    assert!(terminal.checked.fresh_pair.is_none());
    assert_eq!(port.counters.cpu_nodes, 0);
    assert_eq!(port.counters.role_calls, 0);
    assert_eq!(
        port.result(PalsCompletion::RoundLimit).unwrap().best_move,
        Some(BoardMove::from_uci("f7g7").unwrap())
    );
}
#[test]
fn distinct_cpu_and_critic_lines_remain_two_checked_branches() {
    let mut port = setup(Position::startpos(), 4);
    let first = BoardMove::from_uci("e2e4").unwrap();
    let original = BoardMove::from_uci("c7c5").unwrap();
    let cpu = BoardMove::from_uci("e7e5").unwrap();
    let critic = BoardMove::from_uci("e7e6").unwrap();
    let node = port.connect_hot(port.root, first).unwrap();
    // Coherent authored branch genealogy; no physical CPU/NN evidence is
    // attached to either line before this contract fixture submits its head.
    port.record_hot(RecordKind::Proposal, &[first, original], None)
        .unwrap();
    port.record_hot(RecordKind::Counterexample, &[first, cpu], None)
        .unwrap();
    let mut input = port
        .input(
            node,
            RefinementStage::CriticReply,
            Some(first),
            vec![first],
            vec![first, original],
            Some(vec![first, cpu]),
            None,
            Some(1),
            false,
            None,
            None,
        )
        .unwrap();
    Arc::make_mut(&mut input.question).counter_source = None; // Authored line fixture, not physical CPU evidence.
    let work = Work {
        frame: FrameId {
            authority: port.authority,
            sequence: 1,
        },
        original_first_move: rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(first).unwrap(),
        ),
        original_deadline: port.deadline,
        budget: FrameBudget {
            original_cpu_credit: port.limits.max_cpu_nodes,
            role_submissions_remaining: port.engine.config.max_role_calls,
            original_store_byte_limit: 8 * 1024 * 1024,
        },
        input,
    };
    let (ticket, mut out) = output(&work, 1);
    if let RolePayload::Counterexample {
        candidate_logits, ..
    } = &mut out.payload
    {
        candidate_logits[work
            .input
            .view
            .query()
            .legal
            .iter()
            .position(|mv| *mv == critic)
            .unwrap()] = 1.0;
    }
    let first_step = port.commit_role(&work, &ticket, &out, None).unwrap();
    port.role_consumed(&ticket, &work.input.view).unwrap();
    let first_step = port.materialize_step(first_step).unwrap();
    let FrameTransition::Continue(cpu_input) = first_step.transition else {
        panic!("CPU line lost")
    };
    assert_eq!(cpu_input.question.context.prefix, vec![first, cpu]);
    assert_eq!(port.pending_counters[&first].len(), 1);
    let cpu_work = Work {
        frame: work.frame,
        original_first_move: work.original_first_move,
        original_deadline: work.original_deadline,
        budget: work.budget,
        input: cpu_input,
    };
    let next = port
        .finish_branch(
            &cpu_work,
            AdaptivePortTransition::Complete {
                scope: FrameCompletionScope::NoAlternative,
            },
        )
        .unwrap();
    // This authored branch fixture checks sibling preservation, not a
    // fabricated CPU receipt. Selection uses the already consumed role cause.
    let next = port
        .materialize_step(AdaptivePortStep {
            evidence: first_step.evidence,
            transition: next,
            fresh_pair: None,
            rules_pair: None,
            consumption_gate: None,
            feedback_gate: None,
            queue_question: None,
        })
        .unwrap();
    let FrameTransition::Continue(critic_input) = next.transition else {
        panic!("critic line discarded")
    };
    assert_eq!(critic_input.question.context.prefix, vec![first, critic]);
    assert_eq!(critic_input.question.first, Some(first));
    assert_eq!(critic_input.question.deadline, port.limits.deadline);
}

#[test]
fn main_frames_finish_one_root_round_together() {
    let mut port = setup(Position::startpos(), 1);
    let first = work(&mut port, "e2e4", RefinementStage::CpuValidation, 1);
    let second = work(&mut port, "d2d4", RefinementStage::CpuValidation, 2);
    for pending in [&first, &second] {
        port.round_candidates.insert(
            pending.input.question_digest,
            (pending.input.question.first.unwrap(), None),
        );
        port.root_candidate_admitted(pending.frame, pending.original_first_move, &pending.input)
            .unwrap();
        assert!(
            port.root_candidate_admitted(
                pending.frame,
                pending.original_first_move,
                &pending.input
            )
            .is_err()
        );
    }
    let close = |pending: &Work| ClosedFrame {
        id: pending.frame,
        original_first_move: pending.original_first_move,
        evidence: vec![ContractDigest([33; 32])],
        unknown: None,
        completion_scope: Some(FrameCompletionScope::NoDivergence),
    };
    port.publish_closed(&close(&first)).unwrap();
    assert!(port.next_root_round_input().unwrap().is_none());
    assert_eq!(port.counters.rounds, 0);
    port.publish_closed(&close(&second)).unwrap();
    assert_eq!(port.counters.rounds, 0);
    let next = port.next_root_round_input().unwrap().unwrap();
    assert_eq!(next.stage, RefinementStage::ProposerRanking);
    assert_eq!(port.counters.rounds, 1);
    assert_eq!(port.root_round, 1);
    assert!(port.next_root_round_input().unwrap().is_none());
    assert_eq!(port.counters.rounds, 1);
}

#[test]
fn widening_frontier_preserves_one_unexamined_first_move() {
    let mut port = setup(Position::startpos(), 2);
    port.engine.config.beam_width = 2;
    let ranked: Vec<_> = ["e2e4", "d2d4", "g1f3", "c2c4"]
        .into_iter()
        .map(|text| BoardMove::from_uci(text).unwrap())
        .collect();
    port.connect_hot(port.root, ranked[0]).unwrap();
    port.connect_hot(port.root, ranked[1]).unwrap();
    assert_eq!(
        port.root_frontier(&ranked, 3).unwrap(),
        vec![ranked[0], ranked[2]]
    );
}

#[test]
fn divergence_aging_uses_the_registered_root_round() {
    let mut port = setup(Position::startpos(), 6);
    port.root_round = 1;
    let proposal: Vec<_> = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
        .into_iter()
        .map(|text| BoardMove::from_uci(text).unwrap())
        .collect();
    let first = proposal[0];
    port.record_hot(RecordKind::Proposal, &proposal, None)
        .unwrap();
    let input = port
        .input(
            port.root,
            RefinementStage::CriticDivergence,
            Some(first),
            vec![],
            proposal,
            None,
            None,
            None,
            false,
            None,
            None,
        )
        .unwrap();
    let work = Work {
        frame: FrameId {
            authority: port.authority,
            sequence: 1,
        },
        original_first_move: rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(first).unwrap(),
        ),
        original_deadline: port.deadline,
        budget: FrameBudget {
            original_cpu_credit: port.limits.max_cpu_nodes,
            role_submissions_remaining: port.engine.config.max_role_calls,
            original_store_byte_limit: 8 * 1024 * 1024,
        },
        input,
    };
    let (ticket, mut out) = output(&work, 1);
    if let RolePayload::Counterexample {
        divergence_logits, ..
    } = &mut out.payload
    {
        *divergence_logits = vec![3.0, 2.0, 1.0];
    }
    let next = port.commit_role(&work, &ticket, &out, None).unwrap();
    port.role_consumed(&ticket, &work.input.view).unwrap();
    let next = port.materialize_step(next).unwrap();
    let FrameTransition::Continue(next) = next.transition else {
        panic!("no discovery")
    };
    assert_eq!(next.question.attack_ply, Some(3));
    assert_eq!(next.question.context.prefix.len(), 3);
}

#[test]
fn immediate_rules_terminal_scan_precedes_every_model_beam() {
    let position = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
    let mates: Vec<_> = position
        .legal_moves()
        .into_iter()
        .filter(|movement| {
            let mut child = position.clone();
            child.make_move(*movement).unwrap();
            matches!(
                child.classify_position().unwrap().play_status,
                PlayStatus::Terminal {
                    reason: TerminalReason::Checkmate,
                    ..
                }
            )
        })
        .collect();
    assert!(!mates.is_empty());
    let mut port = setup(position, 4);
    port.engine.config.beam_width = 1;
    port.prepare_root_rules().unwrap();
    for movement in mates {
        assert!(
            port.engine.semantic.nodes[port.root]
                .edges
                .iter()
                .any(|edge| edge.movement == movement
                    && port.engine.semantic.nodes[edge.child].terminal.is_some())
        );
    }
    let result = port.result(PalsCompletion::RoundLimit).unwrap();
    let mut child = port.engine.semantic.nodes[port.root].position.clone();
    child.make_move(result.best_move.unwrap()).unwrap();
    assert!(matches!(
        child.classify_position().unwrap().play_status,
        PlayStatus::Terminal {
            reason: TerminalReason::Checkmate,
            ..
        }
    ));
    assert_eq!(port.counters.role_calls, 0);
    assert_eq!(port.counters.cpu_nodes, 0);
    let first_edges = port.counters.examined_edges;
    port.prepare_root_rules().unwrap();
    assert_eq!(port.counters.examined_edges, first_edges);
}

#[test]
fn opt_in_role_consumption_copies_semantic_result_and_first_consume_separately() {
    let mut port = setup(Position::startpos(), 2);
    assert!(port.role_consumption_observation().is_none());
    assert!(port.enable_role_consumption_observation());
    assert!(!port.enable_role_consumption_observation());
    let original = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
    let (ticket, out) = output(&original, 1);
    let step = port.commit_role(&original, &ticket, &out, None).unwrap();
    let observed = port.role_consumption_observation().unwrap();
    assert_eq!(observed.authority(), port.authority);
    assert_eq!(observed.events().len(), 1);
    let semantic = observed.events()[0];
    assert_eq!(semantic.request, ticket.id);
    assert_eq!(semantic.authority, ticket.authority);
    assert_eq!(semantic.question, ticket.question);
    assert_eq!(
        semantic.submitted_question,
        Some(original.input.question_digest)
    );
    assert_eq!(semantic.ticket_view, ticket.view.view_digest);
    assert_eq!(
        semantic.submitted_view,
        original.input.view_seal.view_digest
    );
    assert_eq!(semantic.frame, Some(original.frame));
    assert_eq!(
        semantic.outcome,
        AdaptiveRoleConsumptionOutcome::SemanticResult
    );
    assert_eq!(semantic.completed_evidence, Some(step.evidence));
    assert_eq!(semantic.prior_semantic_attempt, Some(false));
    assert!(
        !port
            .role_consumption_recorded(&ticket, &original.input.view)
            .unwrap()
    );

    port.role_consumed(&ticket, &original.input.view).unwrap();
    port.role_consumed(&ticket, &original.input.view).unwrap();
    let observed = port.role_consumption_observation().unwrap();
    assert_eq!(observed.events().len(), 3);
    assert_eq!(
        observed.events()[1].outcome,
        AdaptiveRoleConsumptionOutcome::ConsumedView
    );
    assert_eq!(
        observed.events()[1].stage,
        AdaptiveRoleConsumptionStage::ViewConsumption
    );
    assert_eq!(observed.events()[1].completed_evidence, None);
    assert_eq!(
        observed.events()[1].submitted_question,
        Some(ticket.question)
    );
    assert_eq!(
        observed.events()[2].outcome,
        AdaptiveRoleConsumptionOutcome::DuplicateNoNewConsumption
    );
    assert!(
        port.role_consumption_recorded(&ticket, &original.input.view)
            .unwrap()
    );
    assert!(matches!(
        port.commit_role(&original, &ticket, &out, None),
        Err(PalsError::Role(RoleError::InvalidOutput))
    ));
    let observed = port.role_consumption_observation().unwrap();
    assert_eq!(observed.events()[3].prior_semantic_attempt, Some(true));
    assert_eq!(
        observed.events()[3].outcome,
        AdaptiveRoleConsumptionOutcome::Rejected(AdaptiveRoleConsumptionRejection::InvalidOutput)
    );
    assert_eq!(
        observed
            .events()
            .iter()
            .filter(|event| event.outcome == AdaptiveRoleConsumptionOutcome::ConsumedView)
            .count(),
        1
    );
}

#[test]
fn role_consumption_observation_preserves_rejections_and_refuses_late_enable() {
    let mut port = setup(Position::startpos(), 2);
    assert!(port.enable_role_consumption_observation());
    let original = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
    let (ticket, out) = output(&original, 1);
    let mut foreign = ticket;
    foreign.authority.root.0 += 1;
    assert!(matches!(
        port.commit_role(&original, &foreign, &out, None),
        Err(PalsError::Role(RoleError::InvalidOutput))
    ));
    let event = port.role_consumption_observation().unwrap().events()[0];
    assert_eq!(event.authority, foreign.authority);
    assert_eq!(
        event.outcome,
        AdaptiveRoleConsumptionOutcome::Rejected(AdaptiveRoleConsumptionRejection::InvalidOutput)
    );
    port.engine.semantic.game_generation = Some(port.authority.game.0 + 1);
    assert!(matches!(
        port.commit_role(&original, &ticket, &out, None),
        Err(PalsError::Store(StoreError::StaleConsumer))
    ));
    assert_eq!(
        port.role_consumption_observation().unwrap().events()[1].outcome,
        AdaptiveRoleConsumptionOutcome::Rejected(AdaptiveRoleConsumptionRejection::StaleConsumer)
    );
    port.engine.semantic.game_generation = Some(port.authority.game.0);
    port.limits.deadline = Instant::now();
    assert!(matches!(
        port.commit_role(&original, &ticket, &out, None),
        Err(PalsError::Role(RoleError::Deadline))
    ));
    assert_eq!(
        port.role_consumption_observation().unwrap().events()[2].outcome,
        AdaptiveRoleConsumptionOutcome::Rejected(AdaptiveRoleConsumptionRejection::Deadline)
    );
    assert!(port.consumed_role_views.is_empty());

    let mut unobserved = setup(Position::startpos(), 2);
    let original = work(
        &mut unobserved,
        "e2e4",
        RefinementStage::ProposalContinuation,
        1,
    );
    let (mut ticket, out) = output(&original, 1);
    ticket.authority.root.0 += 1;
    assert!(
        unobserved
            .commit_role(&original, &ticket, &out, None)
            .is_err()
    );
    assert!(!unobserved.enable_role_consumption_observation());
    assert!(unobserved.role_consumption_observation().is_none());
}

#[test]
fn role_consumption_observation_overflow_cannot_change_idempotent_search_behavior() {
    let mut port = setup(Position::startpos(), 2);
    assert!(port.enable_role_consumption_observation());
    let original = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
    let (ticket, out) = output(&original, 1);
    port.commit_role(&original, &ticket, &out, None).unwrap();
    port.role_consumed(&ticket, &original.input.view).unwrap();
    let counters = port.counters;
    let revision = port.engine.semantic.revision;
    let observations = port.engine.semantic.stores.observations.len();
    for _ in 0..MAX_ROLE_CONSUMPTION_EVENTS + 3 {
        port.role_consumed(&ticket, &original.input.view).unwrap();
    }
    let observed = port.role_consumption_observation().unwrap();
    assert_eq!(observed.events().len(), MAX_ROLE_CONSUMPTION_EVENTS);
    assert_eq!(
        observed.unavailable(),
        Some(AdaptiveRoleConsumptionUnavailable::EventCapacity)
    );
    assert_eq!(
        observed
            .events()
            .iter()
            .filter(|event| event.outcome == AdaptiveRoleConsumptionOutcome::ConsumedView)
            .count(),
        1
    );
    assert_eq!(port.counters, counters);
    assert_eq!(port.engine.semantic.revision, revision);
    assert_eq!(port.engine.semantic.stores.observations.len(), observations);
    assert!(
        port.role_consumption_recorded(&ticket, &original.input.view)
            .unwrap()
    );
}

// Authored source fixtures: original semantic/Store path, no native forward.
fn repair_publication_work(line_plies: usize, observe: bool) -> (Port, Work) {
    let mut port = setup(Position::startpos(), line_plies);
    if observe {
        assert!(port.enable_role_consumption_observation());
    }
    let first = BoardMove::from_uci("e2e4").unwrap();
    let proposal = ["e2e4", "c7c5", "g1f3"]
        .map(|m| BoardMove::from_uci(m).unwrap())
        .to_vec();
    let counter = ["e2e4", "e7e5"]
        .map(|m| BoardMove::from_uci(m).unwrap())
        .to_vec();
    port.record_hot(RecordKind::Proposal, &proposal, None)
        .unwrap();
    port.record_hot(RecordKind::Counterexample, &counter, None)
        .unwrap();
    let node = port.replay_node(&counter).unwrap();
    let input = port
        .input(
            node,
            RefinementStage::Repair,
            Some(first),
            counter.clone(),
            proposal,
            Some(counter),
            None,
            Some(1),
            false,
            None,
            None,
        )
        .unwrap();
    let work = Work {
        frame: FrameId {
            authority: port.authority,
            sequence: 1,
        },
        original_first_move: rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(first).unwrap(),
        ),
        original_deadline: port.deadline,
        budget: FrameBudget {
            original_cpu_credit: port.limits.max_cpu_nodes,
            role_submissions_remaining: port.engine.config.max_role_calls,
            original_store_byte_limit: 8 * 1024 * 1024,
        },
        input,
    };
    (port, work)
}
#[test]
fn repair_publication_metadata_is_original_and_does_not_change_semantic_result() {
    let mut results = Vec::new();
    for observe in [false, true] {
        let (mut port, work) = repair_publication_work(3, observe);
        let prior_revision = port.engine.semantic.revision;
        let (ticket, out) = output(&work, 1);
        let step = port.commit_role(&work, &ticket, &out, None).unwrap();
        let record = port.engine.semantic.records.last().unwrap();
        assert_eq!(record.kind, RecordKind::Repair);
        assert_eq!(record.parent_revision, Some(prior_revision));
        assert_eq!(record.line.len(), 3);
        results.push((
            step.evidence,
            record.revision,
            record.line.len(),
            record.line.first().copied(),
            port.counters.role_calls,
            port.counters.accepted_repair_outputs,
        ));
        if observe {
            let event = port.role_consumption_observation().unwrap().events()[0];
            assert_eq!(
                event.outcome,
                AdaptiveRoleConsumptionOutcome::SemanticResult
            );
            assert_eq!(event.completed_evidence, Some(step.evidence));
            assert_eq!(event.frame, Some(work.frame));
            assert_eq!(event.submitted_question, Some(work.input.question_digest));
            let published = event.published_repair_line.unwrap();
            assert_eq!(published.record_revision, record.revision);
            assert_eq!(published.parent_revision, record.parent_revision);
            assert_eq!(published.supersedes_revision, record.supersedes_revision);
            assert_eq!(published.origin_state, record.origin_state);
            assert_eq!(published.committed_plies, record.line.len());
            assert_eq!(published.first_move, record.line.first().copied());
            let original_observation = port
                .engine
                .semantic
                .stores
                .observations
                .get(published.observation)
                .unwrap();
            assert_eq!(original_observation.line, Some(published.line));
            assert_eq!(original_observation.kind, ObservationKind::Repair);
            assert_eq!(
                port.engine
                    .semantic
                    .stores
                    .lines
                    .moves(published.line)
                    .unwrap(),
                record.line
            );
            port.role_consumed(&ticket, &work.input.view).unwrap();
            let consumed = port.role_consumption_observation().unwrap().events()[1];
            assert_eq!(
                consumed.outcome,
                AdaptiveRoleConsumptionOutcome::ConsumedView
            );
            assert!(consumed.published_repair_line.is_none());
            assert!(port.commit_role(&work, &ticket, &out, None).is_err());
            let rejected = *port
                .role_consumption_observation()
                .unwrap()
                .events()
                .last()
                .unwrap();
            assert!(matches!(
                rejected.outcome,
                AdaptiveRoleConsumptionOutcome::Rejected(_)
            ));
            assert!(rejected.published_repair_line.is_none());
        } else {
            assert!(port.role_consumption_observation().is_none());
        }
    }
    assert_eq!(results[0], results[1]);
}
#[test]
fn unfinished_repair_semantic_result_is_not_a_published_repair_line() {
    let (mut port, work) = repair_publication_work(4, true);
    let (ticket, out) = output(&work, 1);
    let step = port.commit_role(&work, &ticket, &out, None).unwrap();
    assert!(matches!(
        step.transition,
        AdaptivePortTransition::Continue(_)
    ));
    let event = port.role_consumption_observation().unwrap().events()[0];
    assert_eq!(
        event.outcome,
        AdaptiveRoleConsumptionOutcome::SemanticResult
    );
    assert!(event.published_repair_line.is_none());
    assert!(
        !port
            .engine
            .semantic
            .records
            .iter()
            .any(|r| r.kind == RecordKind::Repair)
    );
}

// Exercise original owner continuation builders and selected query construction.
// These authored outputs do not stand for CPU validation, machine acceptance,
// live pending concurrency, an actual model graph or case acceptance.
fn repair_provenance_source_fixture(offset_store: bool, frame_sequence: u64) -> (Port, Work) {
    let mut port = setup(Position::startpos(), 4);
    assert!(port.enable_role_consumption_observation());
    if offset_store {
        port.record_hot(
            RecordKind::Proposal,
            &[BoardMove::from_uci("a2a3").unwrap()],
            None,
        )
        .unwrap();
    }
    let proposal = ["e2e4", "c7c5", "g1f3", "d7d6"]
        .map(|m| BoardMove::from_uci(m).unwrap())
        .to_vec();
    let counter = ["e2e4", "e7e5", "g1f3", "b8c6"]
        .map(|m| BoardMove::from_uci(m).unwrap())
        .to_vec();
    port.record_hot(RecordKind::Proposal, &proposal, None)
        .unwrap();
    port.record_hot(RecordKind::Counterexample, &counter, None)
        .unwrap();
    let prefix = counter[..2].to_vec();
    let node = port.replay_node(&prefix).unwrap();
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
    let mut current = Work {
        frame: FrameId {
            authority: port.authority,
            sequence: frame_sequence,
        },
        original_first_move: rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(input.question.first.unwrap()).unwrap(),
        ),
        original_deadline: port.deadline,
        budget: FrameBudget {
            original_cpu_credit: port.limits.max_cpu_nodes,
            role_submissions_remaining: port.engine.config.max_role_calls,
            original_store_byte_limit: 8 * 1024 * 1024,
        },
        input,
    };
    for (sequence, chosen) in [(1, "b1c3"), (2, "b8c6")] {
        let (ticket, mut out) = output(&current, sequence);
        if let RolePayload::Proposal {
            candidate_logits, ..
        } = &mut out.payload
        {
            let chosen = BoardMove::from_uci(chosen).unwrap();
            let index = current
                .input
                .view
                .query()
                .legal
                .iter()
                .position(|m| *m == chosen)
                .unwrap();
            candidate_logits[index] = 1.0;
        } else {
            panic!("authored Repair must use P output")
        }
        let step = port.commit_role(&current, &ticket, &out, None).unwrap();
        port.role_consumed(&ticket, &current.input.view).unwrap();
        let step = port.materialize_step(step).unwrap();
        let FrameTransition::Continue(input) = step.transition else {
            panic!("Repair lost original continuation")
        };
        current = Work { input, ..current };
    }
    assert_eq!(current.input.stage, RefinementStage::RepairValidation);
    assert!(current.input.question.diagnostic_repair_line.is_some());
    (port, current)
}
fn materialize_authored_source_transition(
    port: &mut Port,
    work: &Work,
    transition: AdaptivePortTransition,
) -> Work {
    let transition = port.finish_branch(work, transition).unwrap();
    let step = port
        .materialize_step(AdaptivePortStep {
            evidence: ContractDigest([0x55; 32]),
            transition,
            fresh_pair: None,
            rules_pair: None,
            consumption_gate: None,
            feedback_gate: None,
            queue_question: None,
        })
        .unwrap();
    let FrameTransition::Continue(input) = step.transition else {
        panic!("authored source continuation missing")
    };
    Work {
        input,
        frame: work.frame,
        original_first_move: work.original_first_move,
        original_deadline: work.original_deadline,
        budget: work.budget,
    }
}
#[test]
fn repair_provenance_reaches_original_critic_query_and_first_consume_without_handle_lookup() {
    let mut observed = Vec::new();
    for (offset, frame_sequence) in [(false, 1), (true, 2)] {
        let (mut port, repaired) = repair_provenance_source_fixture(offset, frame_sequence);
        let original = repaired.input.question.diagnostic_repair_line.unwrap();
        assert_eq!(original.producer_frame, repaired.frame);
        assert_eq!(original.producer_request.sequence, 2);
        let repaired_moves = repaired.input.question.repaired.as_ref().unwrap().clone();
        assert_eq!(
            port.engine
                .semantic
                .stores
                .lines
                .moves(original.published_line.line)
                .unwrap(),
            repaired_moves
        );
        // The actual owner helper builds C from its repaired line. Calling the
        // helper is an authored source fixture, not a CPU/Fresh admission proof.
        let next = port
            .critic_recheck_after_repaired_fresh(&repaired.input.question)
            .unwrap();
        let critic = materialize_authored_source_transition(&mut port, &repaired, next);
        assert_eq!(critic.input.stage, RefinementStage::CriticRecheck);
        assert_eq!(critic.input.question.diagnostic_repair_line, Some(original));
        assert_eq!(
            critic.input.question.repaired.as_deref(),
            Some(repaired_moves.as_slice())
        );
        assert_eq!(critic.input.question.proposal, repaired_moves);
        assert_eq!(critic.input.view.query().proposal, repaired_moves);
        assert_eq!(
            critic.input.question.context.proposal_sha256,
            role_line_sha256(b"proposal", &repaired_moves).unwrap()
        );
        let (ticket, out) = output(&critic, 3);
        port.commit_role(&critic, &ticket, &out, None).unwrap();
        let event = *port
            .role_consumption_observation()
            .unwrap()
            .events()
            .last()
            .unwrap();
        assert_eq!(
            event.outcome,
            AdaptiveRoleConsumptionOutcome::SemanticResult
        );
        let source = event.repair_line_input.unwrap();
        assert_eq!(source.refinement_stage, RefinementStage::CriticRecheck);
        assert_eq!(source.provenance, original);
        assert_eq!(
            source.model_proposal_sha256,
            ContractDigest(critic.input.question.context.proposal_sha256)
        );
        assert_eq!(source.model_proposal_plies, 4);
        assert_eq!(source.repaired_plies, Some(4));
        assert_eq!(event.submitted_question, Some(critic.input.question_digest));
        assert_eq!(event.submitted_view, critic.input.view_seal.view_digest);
        port.role_consumed(&ticket, &critic.input.view).unwrap();
        let consumed = *port
            .role_consumption_observation()
            .unwrap()
            .events()
            .last()
            .unwrap();
        assert_eq!(
            consumed.outcome,
            AdaptiveRoleConsumptionOutcome::ConsumedView
        );
        assert_eq!(consumed.request, event.request);
        assert_eq!(consumed.submitted_question, event.submitted_question);
        assert_eq!(consumed.submitted_view, event.submitted_view);
        assert!(consumed.repair_line_input.is_none());
        observed.push((original, repaired_moves));
    }
    // Independent owned Store fixtures deliberately allocate equal move values
    // under different original handles/producer frames; values are no join key.
    assert_eq!(observed[0].1, observed[1].1);
    assert_ne!(
        observed[0].0.published_line.line,
        observed[1].0.published_line.line
    );
    assert_ne!(observed[0].0.producer_frame, observed[1].0.producer_frame);
}
#[test]
fn original_critic_recheck_opens_native_cpu_question_but_closed_frame_stays_rejected() {
    // Reuse the original accepted-output/Rules producer fixture. No worker,
    // physical NN receipt or completed-depth claim is authored by this test.
    let (mut port, repaired) = repair_provenance_source_fixture(false, 1);
    let next = port
        .critic_recheck_after_repaired_fresh(&repaired.input.question)
        .unwrap();
    let critic = materialize_authored_source_transition(&mut port, &repaired, next);
    assert_eq!(critic.input.stage, RefinementStage::CriticRecheck);
    let (evaluation_namespace, search_conditions) = port.cpu_namespace_digests().unwrap();
    let profile = crate::cpu_worker::CpuWorkerProfile {
        owner: rz_contracts::pals_adaptive::CpuWorkerId(1),
        max_depth: port.engine.owned_descriptor().unwrap().config.max_depth,
        evaluation_namespace,
        search_conditions,
    };
    assert!(matches!(
        port.prepare_cpu_question(&critic, &[profile], false),
        Err(PalsError::Store(StoreError::InvalidConditions(
            "not an open CPU frame"
        )))
    ));
    assert!(port.cpu_admissions.is_empty());

    let (ticket, mut output) = output(&critic, 3);
    let chosen = BoardMove::from_uci("c7c5").unwrap();
    let index = critic
        .input
        .view
        .query()
        .legal
        .iter()
        .position(|movement| *movement == chosen)
        .expect("original repaired endpoint has the legal C alternative");
    let RolePayload::Counterexample {
        candidate_logits, ..
    } = &mut output.payload
    else {
        panic!("the original CriticRecheck must consume C output");
    };
    candidate_logits[index] = 1.0;
    let step = port.commit_role(&critic, &ticket, &output, None).unwrap();
    port.role_consumed(&ticket, &critic.input.view).unwrap();
    let next = port.materialize_step(step).unwrap();
    let FrameTransition::Continue(input) = next.transition else {
        panic!("the actual C reply must issue its native recheck validation");
    };
    let validation = Work { input, ..critic };
    assert_eq!(
        validation.input.stage,
        RefinementStage::RecheckCounterValidation
    );
    assert_eq!(
        validation.input.question.context.prefix.last(),
        Some(&chosen)
    );
    assert_eq!(
        validation.input.question.counter.as_deref(),
        Some(validation.input.question.context.prefix.as_slice())
    );
    assert!(!port.closed.contains(&validation.frame));
    let before = port.counters();
    let prepared = port
        .prepare_cpu_question(&validation, &[profile], false)
        .unwrap()
        .expect("the actual open C recheck has its original native CPU credit");
    assert_eq!(prepared.target_depth, port.limits.cpu_depth);
    assert_eq!(
        prepared.total_native_node_credit,
        port.limits
            .max_cpu_nodes
            .min(port.engine.config.cpu_nodes_per_task)
    );
    assert!(prepared.root_moves.is_none());
    let admission = port
        .cpu_admissions
        .get(&validation.input.question_digest)
        .unwrap();
    assert_eq!(admission.frame, validation.frame);
    assert_eq!(admission.stage, RefinementStage::RecheckCounterValidation);
    assert_eq!(admission.first_move, validation.original_first_move);
    assert!(
        admission
            .position
            .snapshot()
            .same_state(&validation.input.question.position.snapshot())
    );
    assert_eq!(
        prepared.position.export().unwrap().snapshot().identity(),
        admission.rules_identity
    );
    assert_eq!(port.counters(), before);
    assert_eq!(port.cpu_admissions.len(), 1);

    // Only logical closure is authored here. This does not claim a worker
    // report, fence, root goal completion or original Store evidence deletion.
    port.publish_closed(&ClosedFrame {
        id: validation.frame,
        original_first_move: validation.original_first_move,
        evidence: vec![validation.input.question_digest],
        unknown: Some("authored closed-frame admission guard"),
        completion_scope: None,
    })
    .unwrap();
    // The common sealed-work check rejects the closed frame before the CPU
    // stage-specific check. Preserve that original typed rejection.
    assert!(matches!(
        port.prepare_cpu_question(&validation, &[profile], false),
        Err(PalsError::Role(RoleError::InvalidOutput))
    ));
    assert_eq!(port.cpu_admissions.len(), 1);
    assert_eq!(port.counters(), before);
}

#[test]
fn zero_step_repair_does_not_inherit_an_earlier_p_line_identity() {
    let (mut port, prior) = repair_provenance_source_fixture(false, 1);
    let provenance = prior.input.question.diagnostic_repair_line.unwrap();
    let proposal = ["e2e4", "e7e5", "g1f3", "b8c6"]
        .map(|m| BoardMove::from_uci(m).unwrap())
        .to_vec();
    let counter = ["e2e4", "e7e5", "g1f3", "d7d6"]
        .map(|m| BoardMove::from_uci(m).unwrap())
        .to_vec();
    port.record_hot(RecordKind::Counterexample, &counter, None)
        .unwrap();
    let node = port.replay_node(&counter).unwrap();
    let mut input = port
        .input(
            node,
            RefinementStage::CounterValidation,
            Some(proposal[0]),
            counter.clone(),
            proposal.clone(),
            Some(counter.clone()),
            Some(proposal.clone()),
            Some(3),
            false,
            None,
            None,
        )
        .unwrap();
    // Authored previous-line metadata on a valid newly submitted counter.
    Arc::make_mut(&mut input.question).diagnostic_repair_line = Some(provenance);
    let counter_work = Work { input, ..prior };
    let zero = port
        .begin_repair_continuation(
            node,
            Some(proposal[0]),
            counter.clone(),
            proposal,
            counter,
            3,
            None,
        )
        .unwrap();
    let repaired = materialize_authored_source_transition(&mut port, &counter_work, zero);
    assert_eq!(repaired.input.stage, RefinementStage::RepairValidation);
    assert!(repaired.input.question.repaired.is_some());
    assert!(repaired.input.question.diagnostic_repair_line.is_none());
}
