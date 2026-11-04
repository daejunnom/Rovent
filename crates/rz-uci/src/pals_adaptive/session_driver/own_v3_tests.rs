//! Own-R V3 binding to actual, independently admitted go capabilities.
//! The CPU/native owner is the existing mock lifecycle owner; no model runs.
use super::*;
use crate::parser::GoLimits;
use crate::rovent_root_admission::{
    RoventRootAdmissionPolicy,
    source_seal::{
        PreparedSourceV3,
        pure_tests::{context_for_go, own_input_lock_v3},
    },
};
use rz_contracts::rovent::ExecutionMode;

#[test]
fn own_v3_two_original_go_budgets_bind_without_replacing_cpu_owner() {
    let (driver, _, clock, actual_owner) =
        fixture_with_rovent_mode(None, None, Some(ExecutionMode::CpuPcr));
    let fixed = driver.rovent.as_ref().unwrap().fixed().unwrap().clone();
    let position = Position::startpos();
    let original_owner = own_input_lock_v3(&position);
    let issue = |nodes, root| {
        let context = context_for_go(
            &clock,
            GoLimits {
                nodes: Some(nodes),
                depth_plies: std::num::NonZeroU16::new(2),
                ..GoLimits::default()
            },
            root,
        );
        let admission = PreparedSourceV3::prepare(
            &original_owner,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &context,
            &clock,
        )
        .unwrap()
        .try_into_live()
        .unwrap();
        assert!(admission.owner().same_original_owner(&original_owner));
        let budget = admission.descriptor().root_budget_digest().unwrap();
        let startup_lock = admission.startup_lock_digest();
        let wrapped = RoventPcrRootAdmission {
            budget,
            original: None,
            original_v2: None,
            original_v3: Some(Arc::new(admission)),
        };
        (context, wrapped, startup_lock)
    };
    let (first_context, first, startup_lock) = issue(7, 1);
    let (second_context, second, second_lock) = issue(11, 2);
    assert_eq!(startup_lock, second_lock);
    assert_ne!(first.budget, second.budget);
    let profile = RoventOwnPcrSessionProfileV3 {
        common: RoventPcrSessionProfileV3 {
            mode: fixed.mode,
            maximum_goal_plies: fixed.maximum_goal_plies,
            input_lock: startup_lock,
            selective_policy: fixed.selective_policy,
            model_namespace: Arc::clone(&fixed.model_namespace),
            frontier_limits: fixed.frontier_limits.clone(),
            peers: fixed.peers,
        },
        reciprocal: RoventOwnReciprocalV3 {
            recipient: fixed.reciprocal.recipient,
            max_questions_per_job: fixed.reciprocal.max_questions_per_job,
            node_interval: fixed.reciprocal.node_interval,
            first_question_sequence: fixed.reciprocal.first_question_sequence,
        },
    };
    for (context, admission) in [(&first_context, &first), (&second_context, &second)] {
        let bound = profile
            .bind_original_root(context, &clock, admission)
            .unwrap();
        assert_eq!(bound.root_budget, admission.budget);
        assert_eq!(bound.reciprocal.root_budget, admission.budget);
    }
    assert!(
        profile
            .bind_original_root(&first_context, &clock, &second)
            .is_err()
    );
    let mut changed = first_context.clone();
    changed.requested_nodes = Some(12);
    assert!(
        profile
            .bind_original_root(&changed, &clock, &first)
            .is_err()
    );
    let wrong_clock = ProcessClock::new(ProcessEpoch(clock.epoch().0 + 1));
    assert!(
        profile
            .bind_original_root(&first_context, &wrong_clock, &first)
            .is_err()
    );
    {
        let state = driver.state.lock().unwrap();
        let SessionState::Idle(owners) = &*state else {
            panic!("original owner must remain idle")
        };
        assert_eq!(
            owners.cpu[0]
                .engine
                .paused_stack_snapshot()
                .unwrap()
                .owner_id,
            actual_owner
        );
    }
    assert!(
        driver
            .finish_process(Instant::now() + Duration::from_secs(5))
            .unwrap()
            .permits_process_close()
    );
}
