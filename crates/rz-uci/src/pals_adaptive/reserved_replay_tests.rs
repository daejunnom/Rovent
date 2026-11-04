//! Authored semantic backpressure, with the real native service, worker,
//! common controller and opaque native-ledger settlement. This tests manager
//! custody/event ordering; Core tests the actual quota-first Rules producer.
use super::*;
use rz_contracts::rovent::{ExecutionMode, SearchGoal, SearchRecipient};
use rz_runtime::Clock;
use rz_runtime::contracts::ContractClock;
use rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace;
use rz_search::pals::rovent_engine::RootMoveDomain;
use rz_search::pals::rovent_frontier::{FrontierLimits, FrontierRegistration, RoventFrontier};
use rz_search::pals::value::{MODEL_WDL_VALUE_SEMANTICS, ModelValueIdentity};
use std::num::NonZeroU16;

fn install_fixture_frontier(manager: &mut Manager) {
    let rules = manager.driver().root_rules();
    let root_moves = Arc::new(RootMoveDomain::new(&rules, None).unwrap());
    let namespace = AdaptiveModelNamespace::new(
        ModelValueIdentity {
            semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
            model: "authored-reserved-replay".into(),
            encoding: "authored-selected-view".into(),
            precision: "fp32".into(),
            model_epoch: [5; 32],
        },
        Digest([5; 32]),
        Digest([6; 32]),
        PrecisionProfile::Fp32,
        1,
    )
    .unwrap();
    let frontier = RoventFrontier::new(
        FrontierRegistration {
            authority: manager.registration.authority,
            mode: ExecutionMode::CpuPcr,
            original_deadline: manager.deadline,
            root_budget: Digest([92; 32]),
            goal: SearchGoal::default(),
            nodes: Some(TOTAL_NODES + 1),
            selective_policy: Digest([93; 32]),
            root_rules: rules,
            root_moves,
            root_state: manager.driver().state,
            model_namespace: Arc::new(namespace),
            native_ledger: Some(manager.cpu_ledger_identity()),
        },
        FrontierLimits {
            maximum_goal_plies: NonZeroU16::new(256).unwrap(),
            maximum_frames: 64,
            maximum_paths: 128,
            maximum_mate_nodes: 128,
            maximum_native_receipts: 16,
        },
        &manager.driver().stores,
        manager.clock.domain(),
        manager.clock.now(),
    )
    .unwrap();
    manager.driver_mut().replay_frontier = Some(frontier);
}

fn accept_second_source_without_draining_cpu(
    manager: &mut Manager,
    until: Instant,
    peer: bool,
) -> RoleTicketEnvelope {
    let ranking = manager.root_ranking_proof.clone().unwrap();
    let first = ranking.ranked_candidates()[1];
    let movement = BoardMove::try_from(first.decode().unwrap()).unwrap();
    let input = manager
        .driver_mut()
        .input(RefinementStage::ProposalContinuation, Some(movement));
    let now = manager.check_running().unwrap();
    let clock = manager.clock.domain();
    let frame = manager
        .machine
        .admit_candidate(
            &ranking,
            first,
            input.clone(),
            FrameBudget {
                role_submissions_remaining: 4,
                original_cpu_credit: TOTAL_NODES,
                original_store_byte_limit: 1 << 20,
            },
            clock,
            now,
        )
        .unwrap();
    let work = manager
        .machine
        .ready_work(
            frame,
            manager.clock.domain(),
            manager.check_running().unwrap(),
        )
        .unwrap();
    manager
        .prepare_role(
            Some(work),
            input,
            if peer {
                RolePurpose::RoventRefinement
            } else {
                RolePurpose::Refinement
            },
        )
        .unwrap();
    loop {
        assert!(Instant::now() < until, "native source retirement timed out");
        manager.dispatch_roles().unwrap();
        manager.deliver_roles().unwrap();
        manager.flush_acks().unwrap();
        while let Some(event) = manager.native.poll() {
            manager.native_event(event).unwrap();
        }
        if let Some(ticket) = manager.roles.iter().find_map(|role| {
            role.work.as_ref().filter(|work| work.frame == frame).and(role.ticket)
        }) && manager.deferred.iter().any(|event| {
            matches!(event, DeferredCommit::Role { ticket: pending, accepted: true } if *pending == ticket)
        }) {
            return ticket;
        }
        thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn original_retired_heads_wait_for_real_native_settlement_without_reexecution_or_double_charge() {
    for peer in [false, true] {
        let (mut manager, facts, root, until) = fixture(FixtureScope::OwnCpuResume, true, 4096);
        assert!(
            !manager
                .semantic_replay_can_wait(&AdaptiveManagerError::Frontier(
                    FrontierError::NodesReserved
                ))
                .unwrap()
        );
        install_fixture_frontier(&mut manager);
        if peer {
            manager
                .configure_rovent_reciprocal(RoventReciprocalPolicy {
                    mode: ExecutionMode::CpuPcr,
                    root_budget: Digest([92; 32]),
                    recipient: SearchRecipient::Proposer,
                    max_questions_per_job: 1,
                    node_interval: TOTAL_NODES,
                    first_question_sequence: 1,
                })
                .unwrap();
            manager
                .configure_rovent_peer_followups(RoventPeerFollowupPolicy {
                    pc_peer: false,
                    researcher: true,
                    max_sources_per_root: 4,
                })
                .unwrap();
        }
        facts.child_gate.store(true, Ordering::Release);
        manager.prepare_root_ranking(root).unwrap();
        pump(&mut manager, until, |manager| {
            manager.own[0].binding.is_some()
                && manager.native_ledger().unwrap().global_usage().unwrap().1 != 0
        });
        manager.driver_mut().block_replay_while_reserved = true;
        let original_job = manager.own[0].binding.as_ref().unwrap().envelope.id;
        let source = accept_second_source_without_draining_cpu(&mut manager, until, peer);
        let nn_before = facts.started.lock().unwrap().len();
        manager.flush_deferred().unwrap();
        if peer {
            manager.drive_accepted_role_parents().unwrap();
            assert!(!manager.drive_accepted_role_parents().unwrap());
        }
        assert!(manager.driver().blocked_replays > 0);
        let consumed = manager.counters.consumed_role_outputs;
        let quiet = if peer {
            manager.drive_accepted_role_parents()
        } else {
            manager.flush_deferred()
        };
        assert!(
            !quiet.unwrap(),
            "a repeated reserved head is not new progress"
        );
        assert_eq!(manager.counters.consumed_role_outputs, consumed);
        assert_eq!(facts.started.lock().unwrap().len(), nn_before);
        assert_eq!(
            facts
                .accepted
                .lock()
                .unwrap()
                .iter()
                .filter(|ticket| **ticket == source)
                .count(),
            1
        );
        assert!(manager.failures().next().is_none());
        assert_eq!(manager.phase(), ManagerPhase::Running);
        assert_eq!(
            manager.own[0].binding.as_ref().unwrap().envelope.id,
            original_job
        );
        while manager
            .native_ledger()
            .unwrap()
            .usage(original_job)
            .unwrap()
            .1
            != 0
        {
            assert!(
                Instant::now() < until,
                "actual CPU owner terminal timed out"
            );
            if let Some(event) = manager.own[0]
                .worker
                .as_ref()
                .unwrap()
                .try_event_observed()
                .unwrap()
            {
                manager.own_event_observed(0, event).unwrap();
            } else {
                thread::sleep(Duration::from_millis(1));
            }
        }
        manager.flush_deferred().unwrap();
        if peer {
            manager.drive_accepted_role_parents().unwrap();
        }
        assert_eq!(
            manager
                .driver()
                .consumed
                .iter()
                .filter(|ticket| **ticket == source)
                .count(),
            1
        );
        assert_eq!(facts.started.lock().unwrap().len(), nn_before);
        let (spent, reserved) = manager.native_ledger().unwrap().global_usage().unwrap();
        let work = manager
            .driver()
            .replay_frontier
            .as_ref()
            .unwrap()
            .snapshot()
            .work;
        assert_eq!(reserved, 0);
        assert_eq!(work.reserved_native_nodes, 0);
        assert_eq!(work.native_nodes, spent);
        assert!(
            !manager
                .semantic_replay_can_wait(&AdaptiveManagerError::Frontier(
                    FrontierError::NodesReserved
                ))
                .unwrap()
        );
        assert!(shutdown(&mut manager, until).permits_archive_close());
    }
}
