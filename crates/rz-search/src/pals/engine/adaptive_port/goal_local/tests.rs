//! Authored semantic/registry fixtures. They use the actual private port and
//! opaque registry permit path, but run no backend, native worker or model.
//! Startup, actual manager dispatch and GPU acceptance remain separate checks.
use super::*;
use crate::contract_time::InstantClock;
use crate::cpu_worker::CpuCreditLedger;
use crate::pals::coordinator::{
    NativeAcceptanceIssuer, PhysicalRoleCompletion, RoleCompletion, RoleSubmission,
    RoleTicketRegistry,
};
use crate::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
use crate::pals::rovent_engine::RootMoveDomain;
use crate::pals::rovent_frontier::{
    FrontierFrameKind, FrontierHorizonAcceptance, FrontierLimits, FrontierRegistration,
    RoventFrontier,
};
use crate::pals::store::StorePins;
use rz_contracts::pals::RepresentationKey;
use rz_contracts::rovent::{ExecutionMode, SearchGoal};
use rz_runtime::adaptive::CoordinatorPolicy;
use std::num::NonZeroU16;

type Port = AdaptiveEnginePort<LegalOrderRoleMock>;

fn source_port() -> Port {
    source_port_at(Position::startpos(), None)
}

fn source_port_at(position: Position, root_moves: Option<&[BoardMove]>) -> Port {
    // GoalLocal reaches the actual Fresh WDL commit boundary. Use the admitted
    // model resolver so its value identity is registered by the real engine
    // constructor; attaching a Frontier alone grants no value registration.
    let mut port = super::super::port_tests::setup_with_resolver(
        position,
        4,
        ResolverPolicy::ModelWdlRestricted,
    );
    attach_source_frontier(
        &mut port,
        root_moves,
        SearchGoal {
            depth_plies: None,
            mate_moves: NonZeroU16::new(2),
        },
    );
    port
}

fn attach_source_frontier(
    port: &mut Port,
    root_moves: Option<&[BoardMove]>,
    goal: SearchGoal,
) -> CpuCreditLedger {
    let namespace = AdaptiveModelNamespace::new(
        port.engine.semantic.model.value_identity().unwrap().clone(),
        ContractDigest([5; 32]),
        ContractDigest([6; 32]),
        rz_contracts::PrecisionProfile::Fp32,
        1,
    )
    .unwrap();
    let ledger = CpuCreditLedger::new(
        port.authority,
        1000,
        InstantClock::new(port.deadline.clock, Instant::now()),
    )
    .unwrap();
    // This isolates the real semantic/controller boundary, as existing port
    // fixtures do. It does not claim native startup namespace acceptance.
    let frontier = RoventFrontier::new(
        FrontierRegistration {
            authority: port.authority,
            mode: ExecutionMode::CpuPcr,
            original_deadline: port.deadline,
            root_budget: ContractDigest([19; 32]),
            goal,
            nodes: Some(1000),
            selective_policy: ContractDigest([20; 32]),
            root_rules: Arc::clone(&port.rules),
            root_moves: Arc::new(RootMoveDomain::new(&port.rules, root_moves).unwrap()),
            root_state: port.engine.semantic.nodes[port.root].state,
            model_namespace: Arc::new(namespace),
            native_ledger: Some(ledger.identity()),
        },
        FrontierLimits {
            maximum_goal_plies: NonZeroU16::new(32).unwrap(),
            maximum_frames: 64,
            maximum_paths: 256,
            maximum_mate_nodes: 256,
            maximum_native_receipts: 16,
        },
        &port.engine.semantic.stores,
        port.deadline.clock,
        MonotonicTick(0),
    )
    .unwrap();
    port.engine.semantic.rovent_frontier = Some(Box::new(frontier));
    ledger
}

include!("lifecycle_tests.rs");

fn input(port: &mut Port) -> GoalLocalInput {
    match port
        .next_rovent_goal_input(port.deadline.clock, MonotonicTick(0))
        .unwrap()
    {
        AdaptiveGoalLocalAdmission::RoleInput(input) => input,
        _ => panic!("expected actual GoalLocal input"),
    }
}

fn accepted(input: &GoalLocalInput, sequence: u64) -> (AcceptedRoleStep, RoleOutput) {
    accepted_frame(input.input(), input.original_deadline(), sequence)
}

fn accepted_frame(
    original: &FrameInput<ReviewView, AdaptivePortQuestion>,
    deadline: Deadline,
    sequence: u64,
) -> (AcceptedRoleStep, RoleOutput) {
    accepted_frame_select(original, deadline, sequence, None)
}

fn accepted_frame_select(
    original: &FrameInput<ReviewView, AdaptivePortQuestion>,
    deadline: Deadline,
    sequence: u64,
    selected: Option<BoardMove>,
) -> (AcceptedRoleStep, RoleOutput) {
    let role = role_for_purpose(original.question.context.purpose);
    let key = RepresentationKey {
        input: ContractDigest(original.view.digest()),
        history: ContractDigest([2; 32]),
        candidates: ContractDigest([3; 32]),
        mask_positions: ContractDigest([4; 32]),
        model: ContractDigest([5; 32]),
        encoding: ContractDigest([6; 32]),
        precision: rz_contracts::PrecisionProfile::Fp32,
        frozen_epoch: 1,
        record_revision: original.question.context.public_revision,
        role,
    };
    let mut registry = RoleTicketRegistry::new(
        original.view_seal.authority,
        512,
        CoordinatorPolicy::Fixed,
        sequence,
    )
    .unwrap();
    let (issuer, verifier) = NativeAcceptanceIssuer::new_pair();
    registry
        .register_native_acceptance_verifier(verifier)
        .unwrap();
    let ticket = registry
        .admit(
            RoleSubmission {
                question: original.question_digest,
                view: original.view_seal,
                representation: key,
                role,
                original_deadline: deadline,
            },
            Arc::clone(&original.view),
            deadline.clock,
            MonotonicTick(0),
        )
        .unwrap();
    registry.take_dispatch().unwrap();
    if let Some(selected) = selected {
        assert!(original.view.query().legal.contains(&selected));
    }
    let candidate_logits = original
        .view
        .query()
        .legal
        .iter()
        .enumerate()
        .map(|(index, movement)| {
            if Some(*movement) == selected {
                100.0
            } else {
                -(index as f32)
            }
        })
        .collect();
    let wdl = rz_contracts::Wdl::try_new(0.25, 0.5, 0.25, 0.0).unwrap();
    let payload = match role {
        Role::Proposer => RolePayload::Proposal {
            candidate_logits,
            wdl,
        },
        Role::Critic => RolePayload::Counterexample {
            candidate_logits,
            divergence_logits: Vec::new(),
            wdl,
        },
        _ => unreachable!("GoalLocal admits only P/C"),
    };
    let output = RoleOutput {
        id: ticket.id,
        authority: ticket.authority,
        key,
        payload,
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
    let permit = registry
        .issue_reciprocal_native_acceptance(&ticket, deadline.clock, MonotonicTick(0))
        .unwrap();
    let proof = registry
        .accept_reciprocal_native(
            issuer.issue_success(permit),
            deadline.clock,
            MonotonicTick(0),
        )
        .unwrap();
    (proof, output)
}

fn physical(port: &Port, sequence: u64) -> AdaptivePortRoleEvidence {
    AdaptivePortRoleEvidence {
        execution: Some(rz_contracts::ExecutionId {
            epoch: port.authority.epoch,
            sequence,
        }),
        completed_nn_inputs: Some(1),
        measurement_complete: true,
        physical_fresh: Some(true),
    }
}

fn depth_profile_port(line_plies: usize) -> Port {
    let config = PalsConfig {
        line_plies,
        ..PalsConfig::default()
    };
    let cpu = OwnedCpuChecker::new(Box::new(
        CpuEngine::new(crate::cpu::CpuConfig {
            tt_entries: 0,
            max_depth: 2,
            quiescence_ply: 0,
            ..Default::default()
        })
        .unwrap(),
    ))
    .unwrap();
    let engine = PalsEngine::new_with_boxed_checker_and_rovent_profile(
        ResearcherSemanticProfile {
            config,
            maximum_root_prefix_plies: 32,
        },
        LegalOrderRoleMock,
        Box::new(cpu),
        ResolverPolicy::ModelWdlRestricted,
        PostRepairRecheckPolicy::Disabled,
    )
    .unwrap()
    .with_selected_review_context(
        ReviewSelectionPolicy::default(),
        ReviewIndexLimits::default(),
    )
    .unwrap();
    let authority = SearchAuthority {
        epoch: rz_contracts::ProcessEpoch(23),
        game: rz_contracts::GameGeneration(0),
        root: rz_contracts::RootGeneration(1),
        implementation: ContractDigest([7; 32]),
    };
    let next = Arc::new(std::sync::atomic::AtomicU64::new(1));
    let issuer: AdaptiveRulesOwnerIssuer =
        Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::Relaxed))));
    Port::new(
        engine,
        Position::startpos(),
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

fn shallow_depth_port(
    depth: u16,
) -> (
    Port,
    CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    AcceptedRoleStep,
    RoleOutput,
    AdaptivePortRoleEvidence,
) {
    shallow_depth_port_with_chunk(depth, 2)
}

fn shallow_depth_port_with_chunk(
    depth: u16,
    chunk: usize,
) -> (
    Port,
    CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    AcceptedRoleStep,
    RoleOutput,
    AdaptivePortRoleEvidence,
) {
    shallow_depth_port_with_kind(depth, chunk, FrontierFrameKind::Candidate)
}

fn shallow_depth_port_with_kind(
    depth: u16,
    chunk: usize,
    kind: FrontierFrameKind,
) -> (
    Port,
    CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    AcceptedRoleStep,
    RoleOutput,
    AdaptivePortRoleEvidence,
) {
    let mut port = depth_profile_port(chunk);
    let (source_stage, endpoint_stage) = match kind {
        FrontierFrameKind::Candidate => (
            RefinementStage::ProposalContinuation,
            RefinementStage::ProposalValueFresh,
        ),
        FrontierFrameKind::Counter => (
            RefinementStage::CriticReply,
            RefinementStage::CounterValueFresh,
        ),
        FrontierFrameKind::Repair => (RefinementStage::Repair, RefinementStage::RepairValueFresh),
    };
    let candidate = super::super::port_tests::work(&mut port, "e2e4", source_stage, 1);
    let first = BoardMove::from_uci("e2e4").unwrap();
    let mut endpoint = candidate.clone();
    endpoint.input = port
        .input(
            candidate.input.question.node,
            endpoint_stage,
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
    attach_source_frontier(
        &mut port,
        Some(&[first]),
        SearchGoal {
            depth_plies: NonZeroU16::new(depth),
            mate_moves: None,
        },
    );
    let clock = port.deadline.clock;
    port.rovent_frame_admitted(&candidate, kind, clock, MonotonicTick(0))
        .unwrap();
    let (accepted, output) = accepted_frame(&endpoint.input, port.deadline, 100);
    let evidence = physical(&port, 100);
    let observed = port
        .with_rovent_frontier(|frontier, stores| {
            let obligation = frontier
                .adaptive_obligation(candidate.frame, kind)?
                .unwrap();
            let path =
                frontier.observe_local_question(&endpoint, stores, clock, MonotonicTick(0))?;
            frontier.accept_local_fresh_horizon_once(
                obligation,
                &endpoint,
                stores,
                path,
                &accepted,
                &output,
                evidence,
                clock,
                MonotonicTick(0),
            )
        })
        .unwrap()
        .unwrap();
    assert_eq!(observed, FrontierHorizonAcceptance::ProgressAccepted);
    let budget = port.rovent_frontier().unwrap().registration().root_budget;
    port.rovent_root_round_sealed(budget, clock, MonotonicTick(0))
        .unwrap();
    (port, endpoint, accepted, output, evidence)
}

#[test]
fn every_original_kind_issues_only_its_own_shallow_progress_continuation() {
    for kind in [
        FrontierFrameKind::Candidate,
        FrontierFrameKind::Counter,
        FrontierFrameKind::Repair,
    ] {
        let (mut port, original, _, _, _) = shallow_depth_port_with_kind(5, 2, kind);
        let clock = port.deadline.clock;
        let source = port
            .with_rovent_frontier(|frontier, _| {
                Ok(frontier
                    .unmet_horizon_questions(1, clock, MonotonicTick(0))?
                    .pop()
                    .unwrap())
            })
            .unwrap()
            .unwrap();
        assert_eq!(source.kind(), kind);
        assert_eq!(source.frame(), original.frame);
        assert_eq!(
            source.original_first_move(),
            BoardMove::from_uci("e2e4").unwrap()
        );
        assert_eq!(source.original_deadline(), original.original_deadline);
        assert_eq!(source.target().0, 5);
        let next = input(&mut port);
        assert_eq!(
            next.source_key(),
            GoalLocalQuestionKey::Horizon(source.key())
        );
        assert_eq!(
            port.rovent_frontier().unwrap().snapshot().registered_frames,
            1
        );
        assert_eq!(
            port.rovent_frontier().unwrap().snapshot().completed_frames,
            0
        );
    }
}

#[test]
fn shallow_accepted_depth_remains_one_original_duty_and_anchor_lookup_adds_no_visit() {
    let (mut port, endpoint, accepted, output, evidence) = shallow_depth_port(5);
    let clock = port.deadline.clock;
    let before = port.rovent_frontier().unwrap().snapshot();
    assert_eq!(before.completed_frames, 0);
    assert_eq!(before.horizon.declared_obligations, 1);
    assert!(!before.horizon.target_satisfied);
    let repeated = port
        .with_rovent_frontier(|frontier, stores| {
            let obligation = frontier
                .adaptive_obligation(endpoint.frame, FrontierFrameKind::Candidate)?
                .unwrap();
            let path =
                frontier.observe_local_question(&endpoint, stores, clock, MonotonicTick(0))?;
            frontier.accept_local_fresh_horizon_once(
                obligation,
                &endpoint,
                stores,
                path,
                &accepted,
                &output,
                evidence,
                clock,
                MonotonicTick(0),
            )
        })
        .unwrap()
        .unwrap();
    assert_eq!(
        repeated,
        FrontierHorizonAcceptance::AlreadyAcceptedProgressExact
    );
    let original = input(&mut port);
    assert!(matches!(
        original.source_key(),
        GoalLocalQuestionKey::Horizon(_)
    ));
    assert_eq!(
        original.original_first_move(),
        BoardMove::from_uci("e2e4").unwrap()
    );
    assert_eq!(
        original.input().question.context.prefix,
        endpoint.input.question.context.prefix
    );
    assert!(original.input().view.query().proposal.is_empty());
    assert!(original.input().view.query().counterexample.is_none());
    assert_eq!(
        port.rovent_frontier()
            .unwrap()
            .snapshot()
            .work
            .checked_transitions,
        before.work.checked_transitions
    );
    assert_eq!(port.counters.role_calls, 0);
    assert_eq!(port.counters.cpu_tasks_requested, 0);
}

#[test]
fn actual_goal_local_chunks_reach_original_depth_without_new_frames_or_r_work() {
    let (mut port, _, _, _, _) = shallow_depth_port(6);
    let clock = port.deadline.clock;
    let old_source = port
        .with_rovent_frontier(|frontier, _| {
            Ok(frontier
                .unmet_horizon_questions(1, clock, MonotonicTick(0))?
                .pop()
                .unwrap())
        })
        .unwrap()
        .unwrap();
    let mut progress_leaf = false;
    let mut closed_leaf = false;
    for sequence in 1..=16 {
        let original = input(&mut port);
        let (accepted, output) = accepted(&original, sequence);
        let evidence = physical(&port, sequence);
        let step = port
            .consume_rovent_goal_input(
                &original,
                &accepted,
                &output,
                evidence,
                port.deadline.clock,
                MonotonicTick(0),
            )
            .unwrap();
        if let GoalLocalStep::Leaf(leaf) = step {
            match leaf.horizon_acceptance() {
                Some(FrontierHorizonAcceptance::ProgressAccepted) => {
                    progress_leaf = true;
                    let snapshot = port.rovent_frontier().unwrap().snapshot();
                    assert_eq!(snapshot.completed_frames, 0);
                    assert!(!snapshot.horizon.target_satisfied);
                }
                Some(FrontierHorizonAcceptance::Closed) => {
                    closed_leaf = true;
                    break;
                }
                other => panic!("unexpected new local action disposition: {other:?}"),
            }
        }
    }
    assert!(progress_leaf);
    assert!(closed_leaf);
    let snapshot = port.rovent_frontier().unwrap().snapshot();
    assert_eq!(snapshot.registered_frames, 1);
    assert_eq!(snapshot.completed_frames, 1);
    assert!(snapshot.horizon.target_satisfied);
    assert_eq!(snapshot.horizon.completed_depth.unwrap().0, 6);
    assert_eq!(snapshot.work.native_nodes, 0);
    assert_eq!(port.counters.cpu_tasks_requested, 0);
    assert_eq!(port.counters.external_checker_tasks, 0);
    assert!(
        port.rovent_frontier()
            .unwrap()
            .validate_unmet_horizon_question(&old_source)
            .is_err()
    );
    assert!(matches!(
        port.next_rovent_goal_input(port.deadline.clock, MonotonicTick(0))
            .unwrap(),
        AdaptiveGoalLocalAdmission::NoQuestion
    ));
}

#[test]
fn sixteen_ply_chunk_retains_the_same_original_depth_duty_beyond_ply_seventeen() {
    let moves: Vec<_> = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1",
        "b7b5", "a4b3", "d7d6", "c2c3", "e8g8", "h2h3", "c6b8", "d2d4", "b8d7",
    ]
    .into_iter()
    .map(|movement| BoardMove::from_uci(movement).unwrap())
    .collect();
    let (mut port, _, _, _, _) = shallow_depth_port_with_chunk(20, 16);
    assert_eq!(port.engine.config.line_plies, 16);
    assert_eq!(port.engine.maximum_root_prefix_plies(), 32);
    let mut at_seventeen = false;
    let mut closed = false;
    for sequence in 1..=24 {
        let original = input(&mut port);
        let prefix = &original.input().question.context.prefix;
        assert!(moves.starts_with(prefix));
        let preferred = (original.input().question.context.purpose != RoleQueryPurpose::ValueFresh)
            .then(|| moves[prefix.len()]);
        let (accepted, output) = accepted_frame_select(
            original.input(),
            original.original_deadline(),
            sequence,
            preferred,
        );
        let evidence = physical(&port, sequence);
        let step = port
            .consume_rovent_goal_input(
                &original,
                &accepted,
                &output,
                evidence,
                port.deadline.clock,
                MonotonicTick(0),
            )
            .unwrap();
        if let GoalLocalStep::Leaf(leaf) = step {
            match leaf.horizon_acceptance().unwrap() {
                FrontierHorizonAcceptance::ProgressAccepted => {
                    assert_eq!(prefix.len(), 17);
                    assert_eq!(
                        port.rovent_frontier().unwrap().snapshot().completed_frames,
                        0
                    );
                    at_seventeen = true;
                }
                FrontierHorizonAcceptance::Closed => {
                    assert_eq!(prefix.len(), 20);
                    closed = true;
                    break;
                }
                other => panic!("new output was not a retry: {other:?}"),
            }
        }
    }
    assert!(at_seventeen && closed);
    let snapshot = port.rovent_frontier().unwrap().snapshot();
    assert_eq!(snapshot.registered_frames, 1);
    assert_eq!(snapshot.completed_frames, 1);
    assert_eq!(snapshot.horizon.completed_depth.unwrap().0, 20);
    assert_eq!(snapshot.horizon.maximum_examined_ply.0, 20);
    assert_eq!(snapshot.work.checked_transitions, 20);
    assert_eq!(snapshot.work.native_nodes, 0);
    assert_eq!(port.counters.cpu_tasks_requested, 0);
}

#[test]
fn missing_origin_is_not_a_model_opinion_and_pending_input_is_not_dispatched_twice() {
    let mut port = source_port();
    let input = input(&mut port);
    assert!(!input.input().question.context.prefix.is_empty());
    assert!(input.input().question.goal_local.is_some());
    assert!(input.input().view.query().proposal.is_empty());
    assert!(input.input().view.query().counterexample.is_none());
    assert!(input.input().view.query().divergences.is_empty());
    assert!(port.engine.semantic.records.is_empty());
    assert_eq!(port.counters.consumed_role_outputs, 0);
    assert!(port.rovent_goal_has_pending());
    assert!(matches!(
        port.next_rovent_goal_input(port.deadline.clock, MonotonicTick(0))
            .unwrap(),
        AdaptiveGoalLocalAdmission::Blocked {
            reason: GoalLocalBlockReason::PendingCapacity
        }
    ));
    port.validate_goal_local_input(&input).unwrap();
}

#[test]
fn view_capacity_retains_the_same_paid_missing_origin_for_a_later_turn() {
    let mut port = source_port();
    let mut leases = Vec::new();
    while port.engine.semantic.stores.review_pin_count().unwrap()
        < port.engine.semantic.stores.review_pin_capacity()
    {
        leases.push(
            port.engine
                .semantic
                .stores
                .pin_review(StorePins::default())
                .unwrap(),
        );
    }
    assert!(matches!(
        port.next_rovent_goal_input(port.deadline.clock, MonotonicTick(0))
            .unwrap(),
        AdaptiveGoalLocalAdmission::Blocked {
            reason: GoalLocalBlockReason::ViewLeaseCapacity
        }
    ));
    let key = port
        .goal_local
        .prepared
        .as_ref()
        .unwrap()
        .origin
        .source_key();
    let work = port.rovent_frontier().unwrap().snapshot().work;
    drop(leases);
    let original = input(&mut port);
    assert_eq!(original.source_key(), key);
    assert_eq!(
        port.rovent_frontier()
            .unwrap()
            .snapshot()
            .work
            .checked_transitions,
        work.checked_transitions
    );
    assert_eq!(port.counters.role_calls, 0);
}

#[test]
fn original_registry_head_advances_once_and_rejects_changed_output_or_physical_seal() {
    let mut port = source_port();
    let original = input(&mut port);
    let (proof, output) = accepted(&original, 1);
    let evidence = physical(&port, 1);
    assert!(matches!(
        port.consume_rovent_goal_input(
            &original,
            &proof,
            &output,
            evidence,
            port.deadline.clock,
            MonotonicTick(0)
        )
        .unwrap(),
        GoalLocalStep::ContinuePrepared { .. }
    ));
    let work = port.rovent_frontier().unwrap().snapshot().work;
    assert_eq!(port.counters.consumed_role_outputs, 1);
    assert!(port.rovent_goal_consumption_recorded(proof.ticket(), &original.input().view));
    assert!(matches!(
        port.consume_rovent_goal_input(
            &original,
            &proof,
            &output,
            evidence,
            port.deadline.clock,
            MonotonicTick(0)
        )
        .unwrap(),
        GoalLocalStep::ContinuePrepared { .. }
    ));
    assert_eq!(port.counters.consumed_role_outputs, 1);
    assert_eq!(
        port.rovent_frontier()
            .unwrap()
            .snapshot()
            .work
            .checked_transitions,
        work.checked_transitions
    );
    let mut altered = output.clone();
    // AcceptedRoleStep seals the consumed public head. Private latent has its
    // own shape/numerical validation and is not that head's digest payload.
    match &mut altered.payload {
        RolePayload::Proposal {
            candidate_logits, ..
        }
        | RolePayload::Counterexample {
            candidate_logits, ..
        } => candidate_logits[0] += 1.0,
        RolePayload::TaskRanking { .. } => unreachable!("GoalLocal admits only P/C"),
    }
    assert!(proof.validates_output(&output));
    assert!(!proof.validates_output(&altered));
    assert!(
        port.consume_rovent_goal_input(
            &original,
            &proof,
            &altered,
            evidence,
            port.deadline.clock,
            MonotonicTick(0)
        )
        .is_err()
    );
    let changed = AdaptivePortRoleEvidence {
        completed_nn_inputs: Some(2),
        ..evidence
    };
    assert!(
        port.consume_rovent_goal_input(
            &original,
            &proof,
            &output,
            changed,
            port.deadline.clock,
            MonotonicTick(0)
        )
        .is_err()
    );
    assert!(port.engine.semantic.records.is_empty());
    assert_eq!(port.counters.cpu_tasks_requested, 0);
    assert_eq!(port.counters.cpu_nodes, 0);
    assert_eq!(port.counters.external_checker_tasks, 0);
}

#[test]
fn one_original_goal_reaches_actual_fresh_leaf_without_promoting_it_to_mate_proof() {
    let mut port = source_port();
    let mut source = None;
    let mut leaf_seen = false;
    for sequence in 1..=4 {
        let original = input(&mut port);
        match source {
            Some(key) => assert_eq!(original.source_key(), key),
            None => source = Some(original.source_key()),
        }
        let fresh = original.input().question.context.purpose == RoleQueryPurpose::ValueFresh;
        let (proof, output) = accepted(&original, sequence);
        let evidence = physical(&port, sequence);
        let step = port
            .consume_rovent_goal_input(
                &original,
                &proof,
                &output,
                evidence,
                port.deadline.clock,
                MonotonicTick(0),
            )
            .unwrap();
        match step {
            GoalLocalStep::ContinuePrepared { key } => {
                assert_eq!(Some(key), source);
                assert!(!fresh);
                assert!(port.rovent_goal_has_pending());
            }
            GoalLocalStep::Leaf(leaf) => {
                assert!(fresh);
                assert_eq!(leaf.kind(), GoalLocalLeafKind::FreshBoundary);
                assert_eq!(Some(leaf.source_key()), source);
                assert!(
                    leaf.snapshot()
                        .same_state(&original.input().question.position.snapshot())
                );
                assert!(!port.rovent_goal_has_pending());
                leaf_seen = true;
                break;
            }
        }
        // The next turn may drop the retired parent's view before preparing
        // its continuation. The leaf/cached step retains no old view pin.
    }
    assert!(leaf_seen);
    assert_eq!(port.counters.value_calls, 1);
    assert_eq!(port.counters.accepted_value_outputs, 1);
    assert_eq!(port.counters.cpu_tasks_requested, 0);
    assert!(port.engine.semantic.records.is_empty());
    assert!(
        port.rovent_frontier()
            .unwrap()
            .snapshot()
            .mate_strategy
            .is_none()
    );
}

#[test]
fn copied_public_input_and_foreign_semantic_owner_cannot_manufacture_an_admission() {
    let mut port = source_port();
    let original = input(&mut port);
    let forged = GoalLocalInput {
        seal: Arc::new(GoalLocalInputSeal {
            input: original.input().clone(),
            origin: Arc::clone(&original.seal.origin),
        }),
    };
    assert!(port.validate_goal_local_input(&forged).is_err());
    let other = source_port();
    assert!(other.validate_goal_local_input(&original).is_err());
    assert_eq!(port.counters.consumed_role_outputs, 0);
}

#[test]
fn imposed_rules_terminal_is_progress_without_any_role_ticket_or_nn_claim() {
    let movement = BoardMove::from_uci("f7g7").unwrap();
    let mut port = source_port_at(
        Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap(),
        Some(&[movement]),
    );
    assert!(matches!(
        port.next_rovent_goal_input(port.deadline.clock, MonotonicTick(0))
            .unwrap(),
        AdaptiveGoalLocalAdmission::RulesAdvanced { .. },
    ));
    assert_eq!(port.counters.role_calls, 0);
    assert_eq!(port.counters.consumed_role_outputs, 0);
    assert_eq!(port.counters.cpu_tasks_requested, 0);
    assert!(port.goal_local.pending.is_empty());
    assert!(!port.rovent_goal_has_pending());
    let strategy = port.rovent_frontier().unwrap().mate_strategy().unwrap();
    assert_eq!(strategy.best_move(), movement);
    assert_eq!(strategy.conclusion().exact_uci_mate(), Some(1));
    assert!(matches!(
        port.next_rovent_goal_input(port.deadline.clock, MonotonicTick(0))
            .unwrap(),
        AdaptiveGoalLocalAdmission::NoQuestion,
    ));
}
