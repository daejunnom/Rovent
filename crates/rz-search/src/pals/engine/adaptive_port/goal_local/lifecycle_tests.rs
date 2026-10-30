// Included by the actual semantic/registry fixture module. The native usage
// packets below are authored ledger fixtures, not execution/fence evidence.
// The manager's real private-worker fence tests remain a separate boundary.

fn next_test_authority(authority: SearchAuthority) -> SearchAuthority {
    SearchAuthority {
        root: rz_contracts::RootGeneration(authority.root.0 + 1),
        ..authority
    }
}

#[test]
fn pending_goal_input_moves_with_the_same_store_and_blocks_a_new_root() {
    let mut port = source_port();
    let original = input(&mut port);
    let owner = port.engine.semantic.stores.read_owner();
    let authority = port.authority;
    let digest = original.input().question_digest;
    let state = original.input().question.context.state;
    let focus = original.input().question.context.focus;
    let weak_view = Arc::downgrade(&original.input().view);
    // Isolate GoalLocal's own lifetime guard from the independent Frontier.
    port.retire_rovent_frontier().unwrap().unwrap();
    let mut engine = port.into_engine();
    let custody = engine.semantic.retired_goal_local.as_ref().unwrap();
    custody.validate_owner(&engine.semantic.stores).unwrap();
    assert!(owner.matches(&engine.semantic.stores));
    assert!(Arc::ptr_eq(
        &custody.state.pending[&digest].seal,
        &original.seal
    ));
    assert!(custody.protected_pins().unwrap().states.contains(&state));
    assert!(custody.protected_pins().unwrap().lines.contains(&focus));
    assert!(custody.owned_host_bytes().unwrap() > 0);
    drop(original);
    assert!(weak_view.upgrade().is_some());
    assert_eq!(engine.semantic.stores.review_pin_count().unwrap(), 1);
    assert!(engine
        .register_adaptive_root(&Position::startpos(), next_test_authority(authority))
        .is_err());
    let generation = engine.semantic.stores.generation();
    engine.new_game();
    assert_eq!(engine.semantic.stores.generation(), generation);
    assert!(engine.semantic.new_game_transition_error.is_some());
    assert!(engine.semantic.retired_goal_local.is_some());
}

#[test]
fn prepared_and_failed_attempts_survive_into_engine_without_new_root_authority() {
    for inject_failure in [false, true] {
        let mut port = source_port();
        let original = input(&mut port);
        let (proof, output) = accepted(&original, 1);
        let evidence = physical(&port, 1);
        if inject_failure {
            // The real view-consumer failure happens after the paid preflight.
            // No fabricated role/ticket or completion authority is installed.
            port.engine.semantic.review_index = None;
        }
        let returned = port.consume_rovent_goal_input(
            &original,
            &proof,
            &output,
            evidence,
            port.deadline.clock,
            MonotonicTick(0),
        );
        if inject_failure {
            assert!(returned.is_err());
            assert!(!port.goal_local.attempts[&proof.ticket().id].head_used);
            assert!(port
                .goal_local
                .selection
                .as_ref()
                .unwrap()
                .prepared
                .is_some());
        } else {
            assert!(matches!(
                returned.unwrap(),
                GoalLocalStep::ContinuePrepared { .. }
            ));
            assert!(port.goal_local.prepared.as_ref().unwrap().node.is_some());
        }
        let source = Arc::clone(&original.seal.origin.source);
        let authority = port.authority;
        port.retire_rovent_frontier().unwrap().unwrap();
        let mut engine = port.into_engine();
        let custody = engine.semantic.retired_goal_local.as_ref().unwrap();
        custody.validate_owner(&engine.semantic.stores).unwrap();
        let attempt = &custody.state.attempts[&proof.ticket().id];
        assert_eq!(attempt.ticket, *proof.ticket());
        assert_eq!(attempt.failure.is_some(), inject_failure);
        if inject_failure {
            assert!(Arc::ptr_eq(
                &custody.state.pending[&original.input().question_digest].seal,
                &original.seal
            ));
            assert!(Arc::ptr_eq(
                &custody
                    .state
                    .selection
                    .as_ref()
                    .unwrap()
                    .prepared
                    .as_ref()
                    .unwrap()
                    .origin
                    .source,
                &source
            ));
            assert!(custody.owned_host_bytes().is_none());
        } else {
            assert!(Arc::ptr_eq(
                &custody.state.prepared.as_ref().unwrap().origin.source,
                &source
            ));
        }
        assert!(engine
            .register_adaptive_root(&Position::startpos(), next_test_authority(authority))
            .is_err());
    }
}

#[test]
fn failed_pin_projection_preserves_its_error_and_original_preparation() {
    let mut port = source_port();
    let original = input(&mut port);
    let (proof, output) = accepted(&original, 1);
    let evidence = physical(&port, 1);
    port.consume_rovent_goal_input(
        &original,
        &proof,
        &output,
        evidence,
        port.deadline.clock,
        MonotonicTick(0),
    )
    .unwrap();
    let key = port
        .goal_local
        .prepared
        .as_ref()
        .unwrap()
        .origin
        .source_key();
    // Fault injection into the resident projection only. The real paid Rules
    // origin is unchanged, and into_engine must not swallow its pin error.
    port.goal_local.prepared.as_mut().unwrap().node = Some(usize::MAX);
    assert!(matches!(
        port.goal_local_protected_pins(),
        Err(PalsError::InvalidConfig)
    ));
    port.retire_rovent_frontier().unwrap().unwrap();
    let engine = port.into_engine();
    let custody = engine.semantic.retired_goal_local.as_ref().unwrap();
    custody.validate_owner(&engine.semantic.stores).unwrap();
    assert_eq!(
        custody.state.prepared.as_ref().unwrap().origin.source_key(),
        key
    );
    assert!(matches!(
        custody.protected_pins(),
        Err(PalsError::InvalidConfig)
    ));
    assert!(custody.owned_host_bytes().is_none());
    assert!(engine.semantic.ensure_frontier_retired().is_err());
}

#[test]
fn explicit_retirement_releases_views_but_close_alone_cannot_release_pending_input() {
    let mut port = source_port();
    let original = input(&mut port);
    let (proof, output) = accepted(&original, 1);
    port.validate_rovent_goal_input(&original, proof.ticket(), &output)
        .unwrap();
    assert_eq!(port.counters.consumed_role_outputs, 0);
    let weak_view = Arc::downgrade(&original.input().view);
    let authority = port.authority;
    port.close_rovent_goal_admission().unwrap();
    assert!(port.release_rovent_goal_state().is_err());
    assert!(port
        .validate_rovent_goal_input(&original, proof.ticket(), &output)
        .is_err());
    assert!(port
        .goal_local
        .pending
        .contains_key(&original.input().question_digest));
    // This exercises the logical API after the original registry retired its
    // fixture packet. It does not claim an actual worker/GPU fence was run.
    port.discard_rovent_goal_input(&original).unwrap();
    port.release_rovent_goal_state().unwrap();
    assert!(!port.rovent_goal_has_pending());
    assert!(!port.goal_local.has_retained_state());
    drop(original);
    assert!(weak_view.upgrade().is_none());
    port.retire_rovent_frontier().unwrap().unwrap();
    assert_eq!(port.engine.semantic.stores.review_pin_count().unwrap(), 0);
    let mut engine = port.into_engine();
    assert!(engine.semantic.retired_goal_local.is_none());
    let authority = next_test_authority(authority);
    let registration = engine
        .register_adaptive_root(&Position::startpos(), authority)
        .unwrap();
    engine
        .consume_adaptive_root_registration(registration, &Position::startpos(), authority)
        .unwrap();
    assert_eq!(engine.semantic.stores.generation(), authority.root.0);
}

fn reserve_remaining_goal_quota(
    port: &mut Port,
    ledger: &mut CpuCreditLedger,
) -> crate::pals::rovent_frontier::NativeReservation {
    use rz_contracts::pals_adaptive::{CpuJobEnvelope, CpuJobId, CpuWorkerId, SliceGrant};
    let frontier = port.rovent_frontier().unwrap();
    let job = CpuJobEnvelope {
        id: CpuJobId {
            authority: port.authority,
            sequence: 1,
        },
        state_identity: frontier.registration().root_rules.snapshot().identity(),
        question: ContractDigest([30; 32]),
        root_restriction: ContractDigest([31; 32]),
        evaluation_namespace: ContractDigest([32; 32]),
        search_conditions: ContractDigest([33; 32]),
        total_native_node_credit: 1000,
        hard_deadline: port.deadline,
    };
    ledger.register(job).unwrap();
    let clock = port.deadline.clock;
    let credit = port
        .with_rovent_frontier(|frontier, _| frontier.native_capacity(1000, clock, MonotonicTick(0)))
        .unwrap()
        .unwrap();
    let grant = SliceGrant {
        job: job.id,
        execution: rz_contracts::ExecutionId::new(port.authority.epoch, 90),
        owner: CpuWorkerId(0),
        node_credit: credit,
        soft_lease: Duration::from_millis(8),
        control_sequence: 1,
    };
    ledger.reserve(grant).unwrap();
    port.with_rovent_frontier(|frontier, _| {
        frontier.reserve_native(&job, grant, clock, MonotonicTick(0))
    })
    .unwrap()
    .unwrap()
}

#[test]
fn nodes_reserved_preserves_one_unconsumed_choice_and_retry_reuses_the_paid_child() {
    let mut port = super::super::port_tests::setup(Position::startpos(), 4);
    let mut ledger = attach_source_frontier(
        &mut port,
        None,
        SearchGoal {
            depth_plies: None,
            mate_moves: NonZeroU16::new(2),
        },
    );
    let original = input(&mut port);
    let (proof, output) = accepted(&original, 1);
    let evidence = physical(&port, 1);
    let reservation = reserve_remaining_goal_quota(&mut port, &mut ledger);
    let before = port.rovent_frontier().unwrap().snapshot().work;
    assert!(matches!(
        port.consume_rovent_goal_input(
            &original,
            &proof,
            &output,
            evidence,
            port.deadline.clock,
            MonotonicTick(0)
        ),
        Err(PalsError::Rovent(FrontierError::NodesReserved))
    ));
    let selected = port.goal_local.selection.as_ref().unwrap().prefix.clone();
    assert!(port.goal_local.attempts.is_empty());
    assert_eq!(port.counters.consumed_role_outputs, 0);
    assert!(!port.rovent_goal_consumption_recorded(proof.ticket(), &original.input().view));
    assert_eq!(
        port.rovent_frontier()
            .unwrap()
            .snapshot()
            .work
            .checked_transitions,
        before.checked_transitions
    );
    let changed = AdaptivePortRoleEvidence {
        completed_nn_inputs: Some(2),
        ..evidence
    };
    assert!(port
        .consume_rovent_goal_input(
            &original,
            &proof,
            &output,
            changed,
            port.deadline.clock,
            MonotonicTick(0)
        )
        .is_err());
    let (different, different_output) = accepted(&original, 2);
    assert!(port
        .consume_rovent_goal_input(
            &original,
            &different,
            &different_output,
            evidence,
            port.deadline.clock,
            MonotonicTick(0)
        )
        .is_err());
    assert_eq!(port.goal_local.selection.as_ref().unwrap().prefix, selected);
    let report = crate::cpu_worker::CpuSliceReport {
        grant: reservation.grant(),
        reciprocal_request: None,
        actual_work: Some(crate::cpu::CpuWork::default()),
        result: Err(crate::cpu::CpuError::Allocation),
        cumulative_nodes: 0,
        progress_sequence: 1,
        continuation_retained: false,
        terminal: true,
        admission_rejected: false,
        controls: None,
    };
    let attested = ledger.settle_attested(&report).unwrap();
    port.with_rovent_frontier(|frontier, _| {
        frontier.settle_native(&reservation, &report, &attested)
    })
    .unwrap()
    .unwrap();
    port.prepare_goal_local_selection(
        &original,
        &proof,
        &output,
        evidence,
        port.deadline.clock,
        MonotonicTick(0),
    )
    .unwrap();
    let paid = port
        .rovent_frontier()
        .unwrap()
        .snapshot()
        .work
        .checked_transitions;
    // This explicit mate query performs one new checked path transition and
    // one real open_child transition in each attacker reducer. These are
    // three actual Rules investigations, not a charge for retry/NN use.
    assert_eq!(paid, before.checked_transitions + 3);
    assert_eq!(port.counters.consumed_role_outputs, 0);
    port.prepare_goal_local_selection(
        &original,
        &proof,
        &output,
        evidence,
        port.deadline.clock,
        MonotonicTick(0),
    )
    .unwrap();
    assert_eq!(
        port.rovent_frontier()
            .unwrap()
            .snapshot()
            .work
            .checked_transitions,
        paid
    );
    for _ in 0..2 {
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
        assert_eq!(
            port.rovent_frontier()
                .unwrap()
                .snapshot()
                .work
                .checked_transitions,
            paid
        );
    }
    assert_eq!(port.counters.consumed_role_outputs, 1);
}

#[test]
fn paid_choice_rechecks_cancel_and_unknown_before_any_head_use() {
    for unknown in [false, true] {
        let mut port = super::super::port_tests::setup(Position::startpos(), 4);
        let mut ledger = attach_source_frontier(
            &mut port,
            None,
            SearchGoal {
                depth_plies: None,
                mate_moves: NonZeroU16::new(2),
            },
        );
        let original = input(&mut port);
        let (proof, output) = accepted(&original, 1);
        let evidence = physical(&port, 1);
        port.prepare_goal_local_selection(
            &original,
            &proof,
            &output,
            evidence,
            port.deadline.clock,
            MonotonicTick(0),
        )
        .unwrap();
        let prefix = port.goal_local.selection.as_ref().unwrap().prefix.clone();
        if unknown {
            let reservation = reserve_remaining_goal_quota(&mut port, &mut ledger);
            port.with_rovent_frontier(|frontier, _| frontier.native_usage_unknown(&reservation))
                .unwrap()
                .unwrap();
        } else {
            port.with_rovent_frontier(|frontier, _| {
                frontier.cancel();
                Ok(())
            })
            .unwrap();
        }
        let work = port.rovent_frontier().unwrap().snapshot().work;
        let returned = port.consume_rovent_goal_input(
            &original,
            &proof,
            &output,
            evidence,
            port.deadline.clock,
            MonotonicTick(0),
        );
        assert!(matches!(
            (unknown, returned),
            (false, Err(PalsError::Rovent(FrontierError::Revoked)))
                | (true, Err(PalsError::Rovent(FrontierError::NativeUnknown)))
        ));
        assert_eq!(port.counters.consumed_role_outputs, 0);
        assert!(!port.rovent_goal_consumption_recorded(proof.ticket(), &original.input().view));
        assert!(port.goal_local.attempts.is_empty());
        assert_eq!(port.goal_local.selection.as_ref().unwrap().prefix, prefix);
        assert_eq!(
            port.rovent_frontier()
                .unwrap()
                .snapshot()
                .work
                .checked_transitions,
            work.checked_transitions
        );
        assert!(port
            .goal_local
            .pending
            .contains_key(&original.input().question_digest));
    }
}
