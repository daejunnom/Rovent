//! Authored native receipts exercise only source custody and goal semantics.
//! No neural model, real R worker or GPU runs in these fixtures.
use super::*;
use crate::contract_time::InstantClock;
use crate::pals::engine::adaptive_port::{AdaptiveModelNamespace, AdaptivePortModel};
use crate::pals::rovent_frontier::{FrontierGoalStatus, FrontierLimits, FrontierRegistration};
use rz_contracts::rovent::SearchGoal;
use std::num::NonZeroU16;

#[path = "peer_tests.rs"]
mod peer_tests;
#[path = "repair_cycle_tests.rs"]
mod repair_cycle_tests;

type GoalPort = PcSemanticPort<AdaptivePortModel>;

fn goal_port(position: Position, goal: SearchGoal, roots: Option<&[BoardMove]>) -> GoalPort {
    goal_port_with_nodes(position, goal, roots, 2000)
}

fn goal_port_with_nodes(
    position: Position,
    goal: SearchGoal,
    roots: Option<&[BoardMove]>,
    nodes: u64,
) -> GoalPort {
    let value = LegalOrderRoleMock.value_identity().unwrap().clone();
    let namespace = AdaptiveModelNamespace::new(
        value.clone(),
        ContractDigest([5; 32]),
        ContractDigest([6; 32]),
        PrecisionProfile::Fp32,
        1,
    )
    .unwrap();
    let model =
        AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value), namespace.clone())
            .unwrap();
    let mut owner = PcSemanticOwner::new_selected_rovent(
        PcSemanticProfile {
            semantic: PalsSemanticConfig {
                beam_width: 1,
                line_plies: 2,
                max_nodes: 257,
                max_records: 128,
                max_role_calls: 128,
            },
            maximum_root_prefix_plies: 16,
        },
        model,
        ReviewSelectionPolicy::default(),
        ReviewIndexLimits::default(),
    )
    .unwrap();
    let authority = SearchAuthority {
        epoch: ProcessEpoch(23),
        game: GameGeneration(0),
        root: RootGeneration(7),
        implementation: ContractDigest([7; 32]),
    };
    let clock = InstantClock::new(ClockDomain(authority.epoch), Instant::now());
    let deadline = Deadline {
        clock: clock.domain(),
        at: MonotonicTick(20_000_000_000),
    };
    let registration = owner.register_adaptive_root(&position, authority).unwrap();
    let next = Arc::new(std::sync::atomic::AtomicU64::new(1));
    let issuer: adaptive_port::AdaptiveRulesOwnerIssuer =
        Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::Relaxed))));
    let mut port = GoalPort::new_registered_retaining(
        owner,
        position,
        authority,
        deadline,
        clock.instant_at(deadline.at).unwrap(),
        ExecutionMode::CpuPc,
        issuer,
        registration,
        roots,
    )
    .map_err(|(_, error)| error)
    .unwrap();
    port.attach_rovent_frontier(
        FrontierRegistration {
            authority,
            mode: ExecutionMode::CpuPc,
            original_deadline: deadline,
            root_budget: ContractDigest([11; 32]),
            goal,
            nodes: Some(nodes),
            selective_policy: ContractDigest([12; 32]),
            root_rules: Arc::clone(&port.rules),
            root_moves: Arc::clone(&port.root_moves),
            root_state: port.owner.semantic.nodes[port.root].state,
            model_namespace: Arc::new(namespace),
            native_ledger: None,
        },
        FrontierLimits {
            maximum_goal_plies: NonZeroU16::new(16).unwrap(),
            maximum_frames: 64,
            maximum_paths: 128,
            maximum_mate_nodes: 256,
            maximum_native_receipts: 1,
        },
        PcFrontierStartup {
            root_budget: ContractDigest([11; 32]),
            clock,
        },
    )
    .unwrap();
    port
}

#[test]
fn zero_one_and_exact_root_child_quota_admit_no_role_input() {
    use crate::pals::rovent_frontier::FrontierStop;
    let roots = [BoardMove::from_uci("e2e4").unwrap()];
    for nodes in [0, 1, 2] {
        let mut port = goal_port_with_nodes(
            Position::startpos(),
            SearchGoal {
                depth_plies: NonZeroU16::new(5),
                mate_moves: None,
            },
            Some(&roots),
            nodes,
        );
        assert!(port.next_role().unwrap().is_none());
        assert!(port.next_role().unwrap().is_none());
        let receipt = port.rovent_frontier_receipt().unwrap();
        assert_eq!(
            receipt.snapshot.goal,
            FrontierGoalStatus::Stopped(FrontierStop::WorkBudget)
        );
        assert_eq!(receipt.snapshot.work.total_node_visits(), Some(nodes));
        assert_eq!(
            receipt.snapshot.work.checked_transitions,
            nodes.saturating_sub(1)
        );
        assert_eq!(port.completion, PcPortCompletion::WorkBudget);
        assert_eq!(port.counters.submitted_role_calls, 0);
        assert_eq!(port.counters.consumed_role_outputs, 0);
        assert_eq!(port.counters.completed_nn_inputs, Some(0));
        assert_eq!(port.next_question, 1);
        assert!(port.pending.is_empty());
        // Root inspection never creates an uncharged semantic child or tensor.
        assert_eq!(port.owner.semantic.nodes.len(), 1);
        assert!(port.owner.records().is_empty());
    }
}

#[test]
fn consumed_policy_heads_remain_charged_when_the_last_child_exhausts_quota() {
    use crate::pals::rovent_frontier::FrontierStop;
    let roots = [BoardMove::from_uci("e2e4").unwrap()];
    let mut port = goal_port_with_nodes(
        Position::startpos(),
        SearchGoal {
            depth_plies: NonZeroU16::new(5),
            mate_moves: None,
        },
        Some(&roots),
        3,
    );
    let root = port.next_role().unwrap().unwrap();
    let (proof, output) = accepted(&mut port, &root, 1);
    port.consume_accepted(&root, &proof, &output, Some(evidence(&port, &root, 1)))
        .unwrap();
    let child = port.next_role().unwrap().unwrap();
    assert_eq!(child.question.context.prefix, roots);
    let (proof, output) = accepted(&mut port, &child, 2);
    port.consume_accepted(&child, &proof, &output, Some(evidence(&port, &child, 2)))
        .unwrap();
    assert!(port.next_role().unwrap().is_none());
    let receipt = port.rovent_frontier_receipt().unwrap();
    assert_eq!(
        receipt.snapshot.goal,
        FrontierGoalStatus::Stopped(FrontierStop::WorkBudget)
    );
    assert_eq!(receipt.snapshot.work.total_node_visits(), Some(3));
    assert_eq!(receipt.snapshot.work.checked_transitions, 2);
    assert_eq!(port.counters.submitted_role_calls, 2);
    assert_eq!(port.counters.consumed_role_outputs, 2);
    assert_eq!(port.counters.completed_nn_inputs, Some(2));
    assert_eq!(port.counters.checked_transitions, 2);
    assert_eq!(port.owner.semantic.nodes.len(), 3);
    assert!(port.pending.is_empty());
}

#[test]
fn time_only_selective_completion_requires_accepted_nonempty_sealed_responsibilities() {
    use crate::pals::rovent_frontier::FrontierStop;
    let roots = [BoardMove::from_uci("e2e4").unwrap()];
    let mut port = goal_port(Position::startpos(), SearchGoal::default(), Some(&roots));
    assert_eq!(
        port.rovent_frontier_receipt().unwrap().snapshot.goal,
        FrontierGoalStatus::Pending
    );
    let mut calls = 0;
    let mut saw_unfinished_responsibility = false;
    while let Some(input) = port.next_role().unwrap() {
        calls += 1;
        assert!(
            calls <= 32,
            "finite selected policy must retire its actual work"
        );
        let before = port.rovent_frontier_receipt().unwrap().snapshot;
        assert_ne!(before.goal, FrontierGoalStatus::SelectiveFrontier);
        saw_unfinished_responsibility |= before.completed_frames < before.registered_frames;
        let (proof, output) = accepted(&mut port, &input, calls);
        port.consume_accepted(
            &input,
            &proof,
            &output,
            Some(evidence(&port, &input, calls)),
        )
        .unwrap();
    }
    let snapshot = port.rovent_frontier_receipt().unwrap().snapshot;
    assert!(saw_unfinished_responsibility);
    assert_eq!(port.completion, PcPortCompletion::SelectiveFrontier);
    assert_eq!(snapshot.goal, FrontierGoalStatus::SelectiveFrontier);
    assert!(snapshot.selection_sealed && snapshot.registered_frames > 0);
    assert_eq!(snapshot.completed_frames, snapshot.registered_frames);
    assert_eq!(snapshot.horizon.completed_depth, None);
    assert!(!snapshot.horizon.target_satisfied);
    assert_eq!(snapshot.work.native_nodes, 0);
    assert_eq!(snapshot.work.reserved_native_nodes, 0);
    assert_eq!(snapshot.work.outstanding_native_executions, 0);
    assert_eq!(port.counters.submitted_role_calls, calls);
    assert_eq!(port.counters.consumed_role_outputs, calls);
    assert!(port.counters.accepted_fresh > 0);
    // A later explicit revoke remains a control stop, never a successful
    // selected frontier reinterpreted as depth/mate completion.
    port.revoke();
    assert_eq!(
        port.rovent_frontier_receipt().unwrap().snapshot.goal,
        FrontierGoalStatus::Stopped(FrontierStop::Canceled)
    );
}

#[test]
fn time_only_role_budget_does_not_complete_the_selected_frontier() {
    use crate::pals::rovent_frontier::FrontierStop;
    let roots = [BoardMove::from_uci("e2e4").unwrap()];
    let mut port = goal_port(Position::startpos(), SearchGoal::default(), Some(&roots));
    port.owner.config.max_role_calls = 1;
    let input = port.next_role().unwrap().unwrap();
    let (proof, output) = accepted(&mut port, &input, 1);
    port.consume_accepted(&input, &proof, &output, Some(evidence(&port, &input, 1)))
        .unwrap();
    assert!(port.next_role().unwrap().is_none());
    assert_eq!(port.completion, PcPortCompletion::RoleBudget);
    let snapshot = port.rovent_frontier_receipt().unwrap().snapshot;
    assert_eq!(
        snapshot.goal,
        FrontierGoalStatus::Stopped(FrontierStop::RoleBudget)
    );
    assert!(!snapshot.selection_sealed);
    assert_eq!(snapshot.horizon.completed_depth, None);
    assert_eq!(
        snapshot.horizon.interruption,
        Some(crate::pals::rovent_goals::HorizonInterruption::RoleBudget)
    );
}

#[test]
fn exact_role_limit_keeps_the_actual_completed_goal_instead_of_a_budget_stop() {
    let roots = [BoardMove::from_uci("e2e4").unwrap()];
    for goal in [
        SearchGoal::default(),
        SearchGoal {
            depth_plies: NonZeroU16::new(2),
            mate_moves: None,
        },
    ] {
        let mut expected_calls = 0;
        let mut reference = goal_port(Position::startpos(), goal, Some(&roots));
        while let Some(input) = reference.next_role().unwrap() {
            expected_calls += 1;
            assert!(expected_calls <= 32);
            let (proof, output) = accepted(&mut reference, &input, expected_calls);
            reference
                .consume_accepted(
                    &input,
                    &proof,
                    &output,
                    Some(evidence(&reference, &input, expected_calls)),
                )
                .unwrap();
        }
        let expected_goal = reference.rovent_frontier_receipt().unwrap().snapshot.goal;
        assert!(matches!(
            expected_goal,
            FrontierGoalStatus::SelectiveFrontier | FrontierGoalStatus::DepthSatisfied
        ));
        let mut exact = goal_port(Position::startpos(), goal, Some(&roots));
        exact.owner.config.max_role_calls = expected_calls;
        let mut calls = 0;
        while let Some(input) = exact.next_role().unwrap() {
            calls += 1;
            assert!(calls <= expected_calls);
            let (proof, output) = accepted(&mut exact, &input, calls);
            exact
                .consume_accepted(
                    &input,
                    &proof,
                    &output,
                    Some(evidence(&exact, &input, calls)),
                )
                .unwrap();
        }
        assert_eq!(calls, expected_calls);
        exact.close_role_call_budget(calls, expected_calls).unwrap();
        assert_eq!(
            exact.rovent_frontier_receipt().unwrap().snapshot.goal,
            expected_goal
        );
        assert_eq!(exact.completion, PcPortCompletion::SelectiveFrontier);
    }
}

#[test]
fn role_limit_cannot_override_inflight_custody_or_a_prior_port_failure() {
    let roots = [BoardMove::from_uci("e2e4").unwrap()];
    let mut port = goal_port(Position::startpos(), SearchGoal::default(), Some(&roots));
    port.owner.config.max_role_calls = 1;
    let input = port.next_role().unwrap().unwrap();
    let (proof, output) = accepted(&mut port, &input, 1);
    assert!(port.close_role_call_budget(1, 1).is_err());
    assert_eq!(port.completion, PcPortCompletion::Active);
    assert_eq!(
        port.rovent_frontier_receipt().unwrap().snapshot.goal,
        FrontierGoalStatus::Pending
    );
    port.consume_accepted(&input, &proof, &output, Some(evidence(&port, &input, 1)))
        .unwrap();
    port.completion = PcPortCompletion::Failed;
    port.close_role_call_budget(1, 1).unwrap();
    assert_eq!(port.completion, PcPortCompletion::Failed);
    assert_eq!(
        port.rovent_frontier_receipt().unwrap().snapshot.goal,
        FrontierGoalStatus::Pending
    );
}

#[test]
fn original_pc_goal_reaches_depth_beyond_its_generation_chunk() {
    let roots = [BoardMove::from_uci("e2e4").unwrap()];
    let mut port = goal_port(
        Position::startpos(),
        SearchGoal {
            depth_plies: NonZeroU16::new(5),
            mate_moves: None,
        },
        Some(&roots),
    );
    let mut calls = 0;
    while let Some(input) = port.next_role().unwrap() {
        calls += 1;
        assert!(calls <= 16, "a bounded five-ply lane must make progress");
        let (proof, output) = accepted(&mut port, &input, calls);
        let physical = evidence(&port, &input, calls);
        port.consume_accepted(&input, &proof, &output, Some(physical))
            .unwrap();
    }
    let receipt = port.rovent_frontier_receipt().unwrap();
    assert_eq!(receipt.snapshot.goal, FrontierGoalStatus::DepthSatisfied);
    assert_eq!(
        receipt.snapshot.completed_frames,
        receipt.snapshot.registered_frames
    );
    assert!(receipt.snapshot.registered_frames >= 3);
    assert!(port.counters.accepted_fresh > 0);
    assert!(!matches!(
        port.finish_materialized().value,
        PalsResolvedValue::Unknown
    ));
    assert!(
        port.owner
            .semantic
            .nodes
            .iter()
            .all(|node| node.evidence.is_none() && node.resume.is_none())
    );
    assert_eq!(std::mem::size_of_val(port.researcher()), 0);
}

#[test]
fn terminal_selected_child_closes_rules_duty_without_an_extra_role_call() {
    let position = Position::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").unwrap();
    let roots = [BoardMove::from_uci("g6g7").unwrap()];
    let mut port = goal_port(
        position,
        SearchGoal {
            depth_plies: NonZeroU16::new(3),
            mate_moves: None,
        },
        Some(&roots),
    );
    let input = port.next_role().unwrap().unwrap();
    let (proof, output) = accepted(&mut port, &input, 1);
    port.consume_accepted(&input, &proof, &output, Some(evidence(&port, &input, 1)))
        .unwrap();
    assert!(port.next_role().unwrap().is_none());
    let receipt = port.rovent_frontier_receipt().unwrap();
    assert_eq!(receipt.snapshot.goal, FrontierGoalStatus::MateSatisfied);
    assert!(receipt.snapshot.horizon.target_satisfied);
    assert_eq!(receipt.snapshot.horizon.completed_depth.unwrap().0, 1);
    assert_eq!(receipt.snapshot.horizon.maximum_examined_ply.0, 1);
    let strategy = receipt.snapshot.mate_strategy.as_ref().unwrap();
    assert_eq!(strategy.best_move(), roots[0]);
    assert!(strategy.root_restricted());
    assert!(strategy.validates_root(
        &port.owner.semantic.nodes[port.root].position.snapshot(),
        &roots,
    ));
    assert_eq!(strategy.conclusion().exact_uci_mate(), Some(1));
    assert_eq!(receipt.snapshot.work.checked_transitions, 1);
    assert_eq!(receipt.snapshot.work.native_nodes, 0);
    assert_eq!(receipt.snapshot.registered_frames, 1);
    assert_eq!(receipt.snapshot.completed_frames, 1);
    assert_eq!(port.counters.submitted_role_calls, 1);
    assert_eq!(port.counters.consumed_role_outputs, 1);
    assert_eq!(port.counters.accepted_fresh, 0);
    port.quiesce_satisfied_goal().unwrap();
    assert_eq!(port.completion, PcPortCompletion::SelectiveFrontier);
    port.completion = PcPortCompletion::Failed;
    assert!(port.quiesce_satisfied_goal().is_err());
    assert_eq!(port.completion, PcPortCompletion::Failed);
    assert_eq!(
        port.rovent_frontier_receipt().unwrap().snapshot.goal,
        FrontierGoalStatus::MateSatisfied
    );
}

#[test]
fn real_missing_mate_descriptor_does_not_forge_a_policy_opinion_or_peer() {
    let mut port = goal_port(
        Position::startpos(),
        SearchGoal {
            depth_plies: None,
            mate_moves: NonZeroU16::new(2),
        },
        None,
    );
    port.prepare_rovent_root_rules().unwrap();
    // Isolate the controller's actual missing branch producer; this fixture
    // neither accepts nor simulates a proposer result for the imposed move.
    port.ready.clear();
    port.complete_role_queue().unwrap();
    let input = port.next_role().unwrap().unwrap();
    assert!(input.question.goal_local.is_some());
    assert!(input.view.query().proposal.is_empty());
    assert!(input.view.query().counterexample.is_none());
    assert!(input.view.query().divergences.is_empty());
    assert!(input.question.responsibilities.is_empty());
    assert_eq!(port.counters.consumed_role_outputs, 0);
    assert_eq!(port.owner.records().len(), 0);
    PcPortQuestion::validate_frontier_input(&input, port.owner.stores()).unwrap();
    let exact = input.question.position.side_to_move();
    let root = port.owner.semantic.nodes[port.root].position.side_to_move();
    if input.question.context.prefix.len() < input.question.continuation_end {
        assert_eq!(
            input.question.context.purpose,
            if exact == root {
                RoleQueryPurpose::ProposePolicy
            } else {
                RoleQueryPurpose::ReplyPolicy
            }
        );
    } else {
        assert_eq!(input.question.context.purpose, RoleQueryPurpose::ValueFresh);
    }
}

#[test]
fn missing_origin_survives_actual_policy_consumption_and_fresh_without_new_duties() {
    let mut port = goal_port(
        Position::startpos(),
        SearchGoal {
            depth_plies: None,
            mate_moves: NonZeroU16::new(2),
        },
        None,
    );
    port.prepare_rovent_root_rules().unwrap();
    port.ready.clear();
    port.complete_role_queue().unwrap();
    let first = port.next_role().unwrap().unwrap();
    let origin = first.question.goal_local_origin_digest().unwrap();
    let end = first.question.continuation_end;
    let mut input = first;
    let before_frames = port
        .rovent_frontier_receipt()
        .unwrap()
        .snapshot
        .registered_frames;
    let mut request = 1;
    loop {
        assert_eq!(input.question.goal_local_origin_digest(), Some(origin));
        assert!(input.question.reciprocal_question().is_none());
        assert!(input.question.responsibilities.is_empty());
        assert!(input.view.query().proposal.is_empty());
        assert!(input.view.query().counterexample.is_none());
        let is_fresh = input.question.context.purpose == RoleQueryPurpose::ValueFresh;
        let (proof, output) = accepted(&mut port, &input, request);
        port.consume_accepted(
            &input,
            &proof,
            &output,
            Some(evidence(&port, &input, request)),
        )
        .unwrap();
        assert!(
            port.role_consumption_recorded(proof.ticket(), &input)
                .unwrap()
        );
        assert_eq!(
            port.rovent_frontier_receipt()
                .unwrap()
                .snapshot
                .registered_frames,
            before_frames
        );
        assert!(port.owner.records().is_empty());
        if is_fresh {
            break;
        }
        input = port.next_role().unwrap().unwrap();
        assert!(input.question.context.prefix.len() <= end);
        request += 1;
        assert!(request <= 4);
    }
    assert!(port.counters.accepted_fresh > 0);
    assert_eq!(std::mem::size_of_val(port.researcher()), 0);
}

#[test]
fn shallow_fresh_keeps_original_duty_until_paid_goal_local_continuation_closes_it() {
    let roots = [BoardMove::from_uci("e2e4").unwrap()];
    let mut port = goal_port(
        Position::startpos(),
        SearchGoal {
            depth_plies: NonZeroU16::new(5),
            mate_moves: None,
        },
        Some(&roots),
    );
    // A real shallow Fresh checkpoint is allowed as progress. This bounded
    // fixture deliberately ends its first selected segment at two plies.
    port.ready.front_mut().unwrap().continuation_end = 2;
    let mut request = 1;
    let mut saw_shallow = false;
    let mut saw_horizon_source = false;
    while let Some(input) = port.next_role().unwrap() {
        let shallow = input.question.context.purpose == RoleQueryPurpose::ValueFresh
            && input.question.context.prefix.len() == 2;
        if input.question.goal_local_origin_digest().is_some() {
            assert!(input.question.reciprocal_question().is_none());
            assert!(input.question.responsibilities.is_empty());
            assert!(PcPortQuestion::validated_horizon_source(&input, port.owner.stores()).is_ok());
            saw_horizon_source = true;
        }
        let (proof, output) = accepted(&mut port, &input, request);
        port.consume_accepted(
            &input,
            &proof,
            &output,
            Some(evidence(&port, &input, request)),
        )
        .unwrap();
        assert!(
            port.role_consumption_recorded(proof.ticket(), &input)
                .unwrap()
        );
        if shallow {
            let receipt = port.rovent_frontier_receipt().unwrap();
            assert!(receipt.snapshot.completed_frames < receipt.snapshot.registered_frames);
            assert_ne!(receipt.snapshot.goal, FrontierGoalStatus::DepthSatisfied);
            saw_shallow = true;
        }
        request += 1;
        assert!(request <= 24);
    }
    assert!(saw_shallow && saw_horizon_source);
    let receipt = port.rovent_frontier_receipt().unwrap();
    assert_eq!(receipt.snapshot.goal, FrontierGoalStatus::DepthSatisfied);
    assert_eq!(
        receipt.snapshot.completed_frames,
        receipt.snapshot.registered_frames
    );
    assert!(port.counters.accepted_fresh >= 3);
    assert_eq!(receipt.snapshot.work.native_nodes, 0);
}

#[test]
fn final_pc_selection_uses_the_actual_mate_strategy_even_after_frontier_retirement() {
    let position = Position::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").unwrap();
    let mating = BoardMove::from_uci("g6g7").unwrap();
    let other = BoardMove::from_uci("g6f5").unwrap();
    let roots = [mating, other];
    let mut port = goal_port(position, SearchGoal::default(), Some(&roots));
    port.prepare_rovent_root_rules().unwrap();
    port.root_ranking = vec![other, mating];
    let receipt = port.rovent_frontier_receipt().unwrap();
    assert_eq!(
        receipt.snapshot.mate_strategy.as_ref().unwrap().best_move(),
        mating
    );
    assert_eq!(port.finish_materialized().best_move, Some(mating));
    port.retire_rovent_frontier().unwrap();
    assert_eq!(port.finish_materialized().best_move, Some(mating));
    assert_eq!(port.counters.submitted_role_calls, 0);
}
