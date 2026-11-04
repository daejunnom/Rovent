// Included by the existing frontier_port fixture module. No new model,
// platform, worker, acceptance shortcut or test-only question factory.
use crate::pals::engine::adaptive_port::branch_queue::PendingCounter;

#[test]
fn checked_replay_reuses_charged_snapshots_and_rejects_unchecked_connect() {
    let mut port = super::super::port_tests::setup(Position::startpos(), 4);
    let frontier = controller(&port);
    port.engine.semantic.rovent_frontier = Some(Box::new(frontier));
    let prefix = [
        BoardMove::from_uci("e2e4").unwrap(),
        BoardMove::from_uci("e7e5").unwrap(),
    ];
    let before = port.rovent_frontier().unwrap().snapshot().work;
    let node = port.replay_node(&prefix).unwrap();
    let first = port.rovent_frontier().unwrap().snapshot().work;
    assert_eq!(first.checked_transitions, before.checked_transitions + 2);
    let edges = port.counters.examined_edges;
    assert_eq!(port.replay_node(&prefix).unwrap(), node);
    assert_eq!(
        port.rovent_frontier()
            .unwrap()
            .snapshot()
            .work
            .checked_transitions,
        first.checked_transitions,
    );
    assert_eq!(port.counters.examined_edges, edges);
    assert!(matches!(
        port.connect_hot(port.root, BoardMove::from_uci("d2d4").unwrap()),
        Err(PalsError::Rovent(FrontierError::Invalid(_))),
    ));
}

#[test]
fn checked_replay_rejects_budget_before_hot_graph_or_record_mutation() {
    let mut port = super::super::port_tests::setup(Position::startpos(), 4);
    let original = controller(&port);
    let mut registration = copied_registration(original.registration());
    registration.nodes = Some(1);
    let ledger = CpuCreditLedger::new(
        port.authority,
        1,
        InstantClock::new(port.deadline.clock, Instant::now()),
    )
    .unwrap();
    registration.native_ledger = Some(ledger.identity());
    let frontier = RoventFrontier::new(
        registration,
        limits(),
        &port.engine.semantic.stores,
        port.deadline.clock,
        MonotonicTick(0),
    )
    .unwrap();
    drop(original);
    port.engine.semantic.rovent_frontier = Some(Box::new(frontier));
    let nodes = port.engine.semantic.nodes.len();
    let records = port.engine.semantic.records.len();
    let edges = port.counters.examined_edges;
    assert!(matches!(
        port.replay_node(&[BoardMove::from_uci("e2e4").unwrap()]),
        Err(PalsError::Rovent(FrontierError::NodeBudget)),
    ));
    assert_eq!(port.engine.semantic.nodes.len(), nodes);
    assert_eq!(port.engine.semantic.records.len(), records);
    assert_eq!(port.counters.examined_edges, edges);
    let snapshot = port.rovent_frontier().unwrap().snapshot();
    assert_eq!(snapshot.work.root_entries, 1);
    assert_eq!(snapshot.work.checked_transitions, 0);
}

#[test]
fn queued_counter_checks_quota_without_publishing_examined_graph_edges() {
    let mut port = super::super::port_tests::setup(Position::startpos(), 4);
    let original = controller(&port);
    let mut registration = copied_registration(original.registration());
    registration.nodes = Some(3);
    let ledger = CpuCreditLedger::new(
        port.authority,
        3,
        InstantClock::new(port.deadline.clock, Instant::now()),
    )
    .unwrap();
    registration.native_ledger = Some(ledger.identity());
    let frontier = RoventFrontier::new(
        registration,
        limits(),
        &port.engine.semantic.stores,
        port.deadline.clock,
        MonotonicTick(0),
    )
    .unwrap();
    drop(original);
    port.engine.semantic.rovent_frontier = Some(Box::new(frontier));
    let first = BoardMove::from_uci("e2e4").unwrap();
    let counter = vec![first, BoardMove::from_uci("c7c5").unwrap()];
    let nodes = port.engine.semantic.nodes.len();
    let edges = port.counters.examined_edges;
    // A selected critic head may already have paid this exact path, filling
    // the common budget before queue bookkeeping runs after its attempt seal.
    port.checked_frontier_line(&counter).unwrap();
    assert!(
        port.enqueue_counter(PendingCounter::authored_without_observation_for_test(
            first,
            vec![first, BoardMove::from_uci("e7e5").unwrap()],
            counter.clone(),
            1,
        ))
        .unwrap()
    );
    assert_eq!(port.engine.semantic.nodes.len(), nodes);
    assert_eq!(port.counters.examined_edges, edges);
    let checked = port
        .rovent_frontier()
        .unwrap()
        .snapshot()
        .work
        .checked_transitions;
    assert_eq!(checked, 2);
    // The actual branch selection can adopt this already paid path without
    // another Rules visit; queue validation itself did not choose the branch.
    port.replay_node(&counter).unwrap();
    assert_eq!(port.counters.examined_edges, edges + 2);
    assert_eq!(
        port.rovent_frontier()
            .unwrap()
            .snapshot()
            .work
            .checked_transitions,
        checked,
    );
}
