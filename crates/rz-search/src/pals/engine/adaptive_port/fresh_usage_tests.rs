// Actual existing semantic/native registry fixtures; no role model execution.

#[test]
fn fresh_goal_use_preserves_once_consumption_and_failed_protection_tombstone() {
    let mut port = super::super::port_tests::setup_with_resolver(
        Position::startpos(),
        4,
        ResolverPolicy::ModelWdlRestricted,
    );
    let source =
        super::super::port_tests::work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
    port.engine.semantic.rovent_frontier = Some(Box::new(controller(&port)));
    let clock = port.deadline.clock;
    port.rovent_frame_admitted(
        &source,
        FrontierFrameKind::Candidate,
        clock,
        MonotonicTick(0),
    )
    .unwrap();
    // Like the real producer, create the descendant only after the original
    // controller is attached, so replay_node first pays its Rules path.
    let endpoint = fresh_descendant(&mut port, &source);
    let (accepted, output) = accepted(&endpoint);
    let physical = AdaptivePortRoleEvidence {
        execution: Some(ExecutionId::new(port.authority.epoch, 1)),
        completed_nn_inputs: Some(1),
        measurement_complete: true,
        physical_fresh: Some(true),
    };
    // A real protection failure is retained independently of the WDL head
    // already used by the goal owner; it is not converted into Store success.
    port.engine
        .semantic
        .review_index
        .as_mut()
        .unwrap()
        .consume(&endpoint.input.view, &port.engine.semantic.stores)
        .unwrap();
    for _ in 0..2 {
        let result = port.rovent_fresh_accepted(
            &endpoint,
            &accepted,
            &output,
            physical,
            clock,
            MonotonicTick(0),
        );
        assert!(
            matches!(
                &result,
                Err(PalsError::Review(ReviewError::AlreadyConsumed)),
            ),
            "actual Fresh protection failure changed: {result:?}"
        );
        assert!(
            port.role_consumption_recorded(accepted.ticket(), &endpoint.input.view)
                .unwrap()
        );
        assert_eq!(port.counters.role_calls, 1);
        assert_eq!(port.counters.value_calls, 1);
        assert_eq!(port.counters.consumed_role_outputs, 1);
    }
    assert!(matches!(
        port.commit_role(&endpoint, accepted.ticket(), &output, Some(physical)),
        Err(PalsError::Review(ReviewError::AlreadyConsumed)),
    ));
    assert_eq!(port.counters.role_calls, 1);
    assert_eq!(
        port.rovent_frontier().unwrap().snapshot().completed_frames,
        1
    );
}

#[test]
fn actual_rules_strategy_selects_its_move_for_both_resolvers_without_numeric_relabelling() {
    for resolver in [
        ResolverPolicy::OwnRawRestricted,
        ResolverPolicy::ModelWdlRestricted,
    ] {
        let mut port = super::super::port_tests::setup_with_resolver(
            Position::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").unwrap(),
            4,
            resolver,
        );
        port.engine.semantic.rovent_frontier = Some(Box::new(controller(&port)));
        let movement = BoardMove::from_uci("g6g7").unwrap();
        port.replay_node(&[movement]).unwrap();
        let strategy = port.rovent_frontier().unwrap().mate_strategy().unwrap();
        assert_eq!(strategy.best_move(), movement);
        let result = port.result(PalsCompletion::RoundLimit).unwrap();
        assert_eq!(result.best_move, Some(movement));
        assert_eq!(result.score, None);
        assert!(matches!(result.resolved_value, PalsResolvedValue::Unknown));
        assert_eq!(result.value_scope, PalsValueScope::Unknown);
        assert_eq!(port.counters.role_calls, 0);
        assert_eq!(
            port.rovent_frontier().unwrap().snapshot().work.native_nodes,
            0
        );
    }
}
