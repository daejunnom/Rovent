//! Pure Rules/context fixtures; no model, engine, worker or evaluator starts.
use super::*;
use crate::bridge::BuildSearchSettings;
use crate::contracts::SessionScope;
use crate::engine::driver_start_fixture::{FixtureStart, issue_start};
use rz_contracts::rovent::ROVENT_SEARCH_REVISION;
use rz_contracts::{GameGeneration, ProcessEpoch, RootGeneration};
use rz_experiments::{
    ROVENT_EXECUTION_DOMAIN, ROVENT_POSITION_VISIT_UNIT, RoventExecutionManifest,
};
use rz_runtime::contracts::ContractClock;
use rz_search::time::TimeBudgetConfig;
use serde_json::{Value, json};
use std::{
    num::NonZeroU16,
    time::{Duration, Instant},
};

fn artifact(name: &str) -> Value {
    json!({"path":format!("fixture/{name}"),"sha256":"a".repeat(64),"bytes":32,
        "source":"https://example.org/source-only-root-fixture","license":"fixture-only"})
}
fn wire(position: &Position) -> Value {
    let model = json!({"semantics":"fixture-pc","encoding":"fixture-encoding",
        "policy_head":"legal-candidates/1","value_head":"side-to-move-wdl/1",
        "implementation_sha256":"b".repeat(64),"weights":artifact("weights"),
        "provenance":artifact("model-provenance"),"diagnostic":false});
    json!({
        "domain":ROVENT_EXECUTION_DOMAIN,"contract_revision":ROVENT_SEARCH_REVISION,
        "run_id":"root-admission-source-fixture","purpose":"validation",
        "engine":{"source_commit":"a".repeat(40),"source_components":{"fixture":"c".repeat(64)},
            "binary":artifact("binary"),"mode":"cpu_pc",
            "pc":{"proposer":model,"critic":model,"validator":null,
                "backend":"rust_cpu","device":{"device":"cpu"},
                "backend_configuration_sha256":"d".repeat(64),"runtime_artifacts":[],
                "fp32":true,"tf32":false,"batch_width":1,
                "ort_intra_threads":null,"ort_inter_threads":null,"ort_spinning":null},
            "researcher":null,"teachers":[],"nnuec":null,
            "search":{"node_unit":ROVENT_POSITION_VISIT_UNIT,"requested_nodes":7,
                "depth_plies":2,"mate_moves":null,"registered_max_chess_plies":64,
                "searchmoves":["e2e4","d2d4"],"search_configuration_sha256":"1".repeat(64),
                "root_state_and_history_sha256":hex(rz_position::contracts::rules_snapshot_semantic_digest(&position.snapshot())),
                "rules_profile_sha256":hex(rz_position::contracts::profile_digest())}},
        "resources":{"cpu_quota_millicpus":2000,
            "pools":[{"actors":["manager","pc_caller"],"max_threads":1,"cpu_charge_millicpus":1000}],
            "memory_high_bytes":6_u64<<30,"memory_max_bytes":12_u64<<30,
            "gpu_devices":0,"physical_nn_concurrency":1,"topology_policy":"shared-control/1"},
        "coordinator":{"policy_sha256":"2".repeat(64),"questions_max":64,
            "ancestor_depth_max":16,"waiting_jobs_max":16,"role_tickets_max":16,
            "cpu_credit_limit":0,"original_root_budget_sha256":"3".repeat(64),
            "original_deadline_seal_sha256":"4".repeat(64)},
        "wall_time_max_ms":60000,"cleanup_max_ms":1000,"output_bytes_max":1048576,
        "agent_renewable":true,"provenance":[artifact("source-provenance")]})
}
fn lock(value: Value) -> RoventInputLock {
    RoventExecutionManifest::from_json(&value.to_string())
        .unwrap()
        .lock()
        .unwrap()
}
fn context(clock: &ProcessClock) -> SearchSessionContext {
    let authority = SearchAuthority {
        epoch: clock.epoch(),
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest([9; 32]),
    };
    let go = GoLimits {
        searchmoves: Some(vec!["e2e4".into(), "d2d4".into()]),
        depth_plies: NonZeroU16::new(2),
        nodes: Some(7),
        white_time_ms: Some(120000),
        black_time_ms: Some(120000),
        white_increment_ms: Some(1000),
        black_increment_ms: Some(1000),
        ..GoLimits::default()
    };
    match issue_start(
        &Position::startpos(),
        go,
        authority,
        clock,
        clock.runtime_clock().origin(),
        start_settings(),
        Duration::from_millis(500),
    )
    .unwrap()
    {
        FixtureStart::Started(context) => context,
        FixtureStart::RulesOnly(_) => panic!("ongoing Rules fixture did not issue actual Start"),
    }
}
fn start_settings() -> BuildSearchSettings {
    BuildSearchSettings {
        time_config: TimeBudgetConfig::default(),
        max_simulations: 4096,
        untimed_limit: Duration::from_secs(60),
    }
}
fn refresh_start(position: &Position, clock: &ProcessClock, context: &mut SearchSessionContext) {
    let start = context.original_time_budget.as_ref().unwrap().start;
    refresh_start_at(position, clock, context, start);
}
fn refresh_start_at(
    position: &Position,
    clock: &ProcessClock,
    context: &mut SearchSessionContext,
    start: Instant,
) {
    let go = context.original_go.as_deref().unwrap().clone();
    *context = match issue_start(
        position,
        go,
        context.authority,
        clock,
        start,
        start_settings(),
        Duration::from_millis(500),
    )
    .unwrap()
    {
        FixtureStart::Started(context) => context,
        FixtureStart::RulesOnly(_) => panic!("successful source fixture requires an actual Start"),
    };
}
fn sync_original_go(context: &mut SearchSessionContext) {
    let mut go = context.original_go.as_deref().cloned().unwrap_or_default();
    go.nodes = context.requested_nodes;
    go.depth_plies = context.goal.depth_plies;
    go.mate_moves = context.goal.mate_moves;
    go.searchmoves = context.root_moves.as_ref().map(|moves| move_texts(moves));
    context.original_go = Some(Arc::new(go));
}
fn admission(
    lock: &RoventInputLock,
    policy: RoventRootAdmissionPolicy,
    position: &Position,
    context: &SearchSessionContext,
    clock: &ProcessClock,
) -> RoventRootAdmission {
    RoventRootAdmission::admit(lock, policy, position, context, clock).unwrap()
}

#[test]
fn exact_registered_admission_preserves_actual_rules_history_and_original_clocks() {
    let position = Position::startpos();
    let lock = lock(wire(&position));
    let clock = ProcessClock::new(ProcessEpoch(7101));
    let context = context(&clock);
    let admitted = admission(
        &lock,
        RoventRootAdmissionPolicy::ExactRegistered,
        &position,
        &context,
        &clock,
    );
    assert!(admitted.root_snapshot().same_state(&position.snapshot()));
    assert_eq!(
        admitted.state_and_history_digest(),
        rz_position::contracts::rules_snapshot_semantic_digest(&position.snapshot())
    );
    assert_eq!(
        admitted.rules_profile_digest(),
        rz_position::contracts::profile_digest()
    );
    assert_eq!(
        admitted.hard_deadline(),
        clock.deadline(context.control.deadline).unwrap()
    );
    assert_eq!(
        admitted.soft_deadline(),
        clock.deadline(context.control.soft_deadline).unwrap()
    );
    assert_eq!(
        admitted.admission_deadline(),
        clock.deadline(context.control.admission_deadline).unwrap()
    );
    assert_eq!(
        admitted.cleanup_deadline(),
        clock.deadline(context.shutdown_deadline).unwrap()
    );
    assert_eq!(
        admitted.runtime_clock().origin(),
        clock.runtime_clock().origin()
    );
    assert_eq!(admitted.runtime_clock().domain(), clock.domain());
    assert_eq!(
        admitted.original_go_coverage(),
        RoventOriginalGoCoverage::FullOriginalGo
    );
    let legal = position.ordered_legal_moves();
    let expected: Vec<_> = legal
        .moves()
        .iter()
        .copied()
        .filter(|movement| context.root_moves.as_ref().unwrap().contains(movement))
        .collect();
    assert_eq!(admitted.allowed_root_moves().as_ref(), expected.as_slice());
}

#[test]
fn stable_same_go_seals_do_not_depend_on_observation_now_or_raw_instant_formatting() {
    let position = Position::startpos();
    let lock = lock(wire(&position));
    let clock = ProcessClock::new(ProcessEpoch(7102));
    let context = context(&clock);
    let first = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &context,
        &clock,
    );
    let second = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &context,
        &clock,
    );
    assert_eq!(first.digest(), second.digest());
    assert_eq!(first.root_budget_digest(), second.root_budget_digest());
    assert_eq!(first.deadline_seal_digest(), second.deadline_seal_digest());
    assert_ne!(first.root_budget_digest(), first.deadline_seal_digest());
    assert_eq!(first.canonical_bytes(), second.canonical_bytes());
    let value: Value = serde_json::from_slice(first.canonical_bytes()).unwrap();
    assert_eq!(value["input"]["policy"], "dynamic_go");
    assert_eq!(
        value["input"]["clock"]["hard"]["at_ns"],
        clock.deadline(context.control.deadline).unwrap().at.0
    );
    assert!(!String::from_utf8_lossy(first.canonical_bytes()).contains("Instant"));
}

#[test]
fn dynamic_nodes_zero_is_distinct_without_relaxing_legacy_manifest_validation() {
    let position = Position::startpos();
    let registered = wire(&position);
    let lock = lock(registered.clone());
    let clock = ProcessClock::new(ProcessEpoch(7103));
    let mut context = context(&clock);
    context.requested_nodes = Some(0);
    sync_original_go(&mut context);
    refresh_start(&position, &clock, &mut context);
    let zero = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &context,
        &clock,
    );
    assert_eq!(zero.requested_nodes(), Some(0));
    assert_eq!(
        serde_json::from_slice::<Value>(zero.canonical_bytes()).unwrap()["input"]["original_go"]["requested_nodes"],
        0
    );
    assert_eq!(
        RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::ExactRegistered,
            &position,
            &context,
            &clock
        )
        .unwrap_err()
        .kind,
        RoventRootAdmissionFailure::RegisteredWorkload
    );
    context.requested_nodes = None;
    sync_original_go(&mut context);
    refresh_start(&position, &clock, &mut context);
    let unspecified = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &context,
        &clock,
    );
    assert_ne!(zero.root_budget_digest(), unspecified.root_budget_digest());
    let mut legacy_zero = registered;
    legacy_zero["engine"]["search"]["requested_nodes"] = json!(0);
    assert!(RoventExecutionManifest::from_json(&legacy_zero.to_string()).is_err());
}

#[test]
fn exact_workload_checks_go_goal_history_and_root_subset() {
    let position = Position::startpos();
    let lock = lock(wire(&position));
    let clock = ProcessClock::new(ProcessEpoch(7104));
    let original = context(&clock);
    let mut other = original.clone();
    other.goal.depth_plies = NonZeroU16::new(3);
    sync_original_go(&mut other);
    assert!(
        RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::ExactRegistered,
            &position,
            &other,
            &clock
        )
        .is_err()
    );
    other = original.clone();
    other.root_moves = Some(Arc::from([BoardMove::from_uci("a2a4").unwrap()]));
    sync_original_go(&mut other);
    assert!(
        RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::ExactRegistered,
            &position,
            &other,
            &clock
        )
        .is_err()
    );
    let fen_only = Position::from_fen(&position.to_fen()).unwrap();
    assert_eq!(position.to_fen(), fen_only.to_fen());
    assert!(
        RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::ExactRegistered,
            &fen_only,
            &original,
            &clock
        )
        .is_err()
    );
    let mut fen_context = original.clone();
    refresh_start(&fen_only, &clock, &mut fen_context);
    let fresh = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &fen_only,
        &fen_context,
        &clock,
    );
    assert_ne!(
        fresh.state_and_history_digest(),
        rz_position::contracts::rules_snapshot_semantic_digest(&position.snapshot())
    );
}

#[test]
fn root_subset_rejects_empty_duplicates_and_illegal_moves_and_preserves_explicit_full() {
    let position = Position::startpos();
    let lock = lock(wire(&position));
    let clock = ProcessClock::new(ProcessEpoch(7105));
    let mut context = context(&clock);
    for invalid in [
        vec![],
        vec![BoardMove::from_uci("e2e4").unwrap(); 2],
        vec![BoardMove::from_uci("e2e5").unwrap()],
    ] {
        context.root_moves = Some(invalid.into());
        sync_original_go(&mut context);
        assert_eq!(
            RoventRootAdmission::admit(
                &lock,
                RoventRootAdmissionPolicy::DynamicGo,
                &position,
                &context,
                &clock
            )
            .unwrap_err()
            .kind,
            RoventRootAdmissionFailure::RootMoves
        );
    }
    context.root_moves = None;
    sync_original_go(&mut context);
    refresh_start(&position, &clock, &mut context);
    let unrestricted = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &context,
        &clock,
    );
    context.root_moves = Some(Arc::from(position.ordered_legal_moves().moves()));
    sync_original_go(&mut context);
    refresh_start(&position, &clock, &mut context);
    let explicit = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &context,
        &clock,
    );
    assert_eq!(
        unrestricted.allowed_root_moves(),
        explicit.allowed_root_moves()
    );
    assert_ne!(
        unrestricted.root_budget_digest(),
        explicit.root_budget_digest()
    );
}

#[test]
fn authority_revoke_cancel_and_foreign_clock_cannot_mint_an_admission() {
    let position = Position::startpos();
    let lock = lock(wire(&position));
    let clock = ProcessClock::new(ProcessEpoch(7106));
    let original = context(&clock);
    let foreign = ProcessClock::new(ProcessEpoch(7107));
    assert_eq!(
        RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &original,
            &foreign
        )
        .unwrap_err()
        .kind,
        RoventRootAdmissionFailure::Clock
    );
    let admitted = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &original,
        &clock,
    );
    original.cancellation.cancel();
    assert!(!admitted.original_context().accepts().unwrap());
    assert_eq!(
        RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &original,
            &clock
        )
        .unwrap_err()
        .kind,
        RoventRootAdmissionFailure::RevokedAuthority
    );
    let fresh = context(&clock);
    let mut next = fresh.authority;
    next.root = RootGeneration(next.root.0 + 1);
    *fresh.current.lock().unwrap() = SessionScope::Search(next);
    assert!(
        RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &fresh,
            &clock
        )
        .is_err()
    );
}

#[test]
fn original_deadline_order_origin_and_cleanup_ceiling_are_enforced() {
    let position = Position::startpos();
    let lock = lock(wire(&position));
    let clock = ProcessClock::new(ProcessEpoch(7108));
    let original = context(&clock);
    let mut wrong = original.clone();
    wrong.control.soft_deadline = wrong.control.deadline + Duration::from_nanos(1);
    assert_eq!(
        RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &wrong,
            &clock
        )
        .unwrap_err()
        .kind,
        RoventRootAdmissionFailure::DeadlineOrder
    );
    wrong = original.clone();
    wrong.control.soft_deadline = clock.runtime_clock().origin() - Duration::from_nanos(1);
    assert_eq!(
        RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &wrong,
            &clock
        )
        .unwrap_err()
        .kind,
        RoventRootAdmissionFailure::Clock
    );
    wrong = original.clone();
    wrong.shutdown_deadline = wrong.control.deadline + Duration::from_millis(1001);
    assert_eq!(
        RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &wrong,
            &clock
        )
        .unwrap_err()
        .kind,
        RoventRootAdmissionFailure::DeclaredCeiling
    );
    wrong = original.clone();
    let later_start =
        original.original_time_budget.as_ref().unwrap().start + Duration::from_nanos(1);
    refresh_start_at(&position, &clock, &mut wrong, later_start);
    let changed = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &wrong,
        &clock,
    );
    let unchanged = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &original,
        &clock,
    );
    assert_ne!(
        changed.deadline_seal_digest(),
        unchanged.deadline_seal_digest()
    );
}

#[test]
fn actual_rules_profile_goal_ceiling_and_tampered_lock_are_rejected() {
    let position = Position::startpos();
    let original_wire = wire(&position);
    let lock = lock(original_wire.clone());
    let clock = ProcessClock::new(ProcessEpoch(7109));
    let mut context = context(&clock);
    context.goal.depth_plies = NonZeroU16::new(65);
    sync_original_go(&mut context);
    assert_eq!(
        RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &context,
            &clock
        )
        .unwrap_err()
        .kind,
        RoventRootAdmissionFailure::Goal
    );
    context.goal.depth_plies = NonZeroU16::new(2);
    sync_original_go(&mut context);
    refresh_start(&position, &clock, &mut context);
    let mut wrong_profile = original_wire;
    // The outer manifest rejects a missing/all-zero declaration before root
    // admission. Use a valid nonzero declaration to exercise actual mismatch.
    wrong_profile["engine"]["search"]["rules_profile_sha256"] = json!("f".repeat(64));
    let wrong_profile = super::tests::lock(wrong_profile);
    assert_eq!(
        RoventRootAdmission::admit(
            &wrong_profile,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &context,
            &clock
        )
        .unwrap_err()
        .kind,
        RoventRootAdmissionFailure::RulesProfile
    );
    let mut tampered = lock;
    tampered.manifest.coordinator.questions_max += 1;
    assert_eq!(
        RoventRootAdmission::admit(
            &tampered,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &context,
            &clock
        )
        .unwrap_err()
        .kind,
        RoventRootAdmissionFailure::InvalidLock
    );
}

#[test]
fn actual_product_requires_original_go_and_rejects_mismatched_resolved_fields() {
    let position = Position::startpos();
    let lock = lock(wire(&position));
    let clock = ProcessClock::new(ProcessEpoch(7110));
    let original = context(&clock);
    let mut missing = original.clone();
    missing.original_go = None;
    assert_eq!(
        RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &missing,
            &clock
        )
        .unwrap_err()
        .kind,
        RoventRootAdmissionFailure::OriginalGoMissing
    );
    for changed in 0..3 {
        let mut mismatch = original.clone();
        match changed {
            0 => mismatch.requested_nodes = Some(0),
            1 => mismatch.goal.mate_moves = NonZeroU16::new(1),
            _ => mismatch.root_moves = None,
        }
        assert_eq!(
            RoventRootAdmission::admit(
                &lock,
                RoventRootAdmissionPolicy::DynamicGo,
                &position,
                &mismatch,
                &clock
            )
            .unwrap_err()
            .kind,
            RoventRootAdmissionFailure::OriginalGoMismatch
        );
    }
}

#[test]
fn actual_start_source_rejects_missing_or_substituted_same_value_input_owners() {
    let position = Position::startpos();
    let lock = lock(wire(&position));
    let clock = ProcessClock::new(ProcessEpoch(7115));
    let original = context(&clock);
    let mut missing = original.clone();
    missing.original_start = None;
    let error = RoventRootAdmission::admit(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &missing,
        &clock,
    )
    .unwrap_err();
    assert_eq!(error.kind, RoventRootAdmissionFailure::Clock);
    assert_eq!(error.detail, "actual Start source missing");
    for changed in 0..5 {
        let mut mismatch = original.clone();
        match changed {
            0 => {
                mismatch.original_go =
                    Some(Arc::new(original.original_go.as_deref().unwrap().clone()))
            }
            1 => {
                mismatch.root_moves =
                    Some(Arc::from(original.root_moves.as_ref().unwrap().as_ref()))
            }
            2 => {
                mismatch.current = Arc::new(std::sync::Mutex::new(SessionScope::Search(
                    original.authority,
                )))
            }
            3 => mismatch.cancellation = rz_contracts::CancelToken::new(),
            4 => {
                mismatch.control.cancellation = Arc::new(std::sync::atomic::AtomicBool::new(false))
            }
            _ => unreachable!(),
        }
        let error = RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &mismatch,
            &clock,
        )
        .unwrap_err();
        assert_eq!(error.kind, RoventRootAdmissionFailure::Clock);
        assert_eq!(error.detail, "actual Start source mismatch");
        assert!(error.metadata_error.is_some());
    }
}

#[test]
fn raw_uci_clocks_and_output_ownership_are_preserved_without_reconstructing_them() {
    let position = Position::startpos();
    let lock = lock(wire(&position));
    let clock = ProcessClock::new(ProcessEpoch(7111));
    let original = context(&clock);
    let first = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &original,
        &clock,
    );
    let mut other = original.clone();
    let raw = Arc::make_mut(other.original_go.as_mut().unwrap());
    raw.white_time_ms = Some(123456);
    raw.black_time_ms = Some(234567);
    raw.white_increment_ms = Some(321);
    raw.black_increment_ms = Some(654);
    raw.moves_to_go = Some(17);
    raw.ponder = true;
    refresh_start(&position, &clock, &mut other);
    let changed = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &other,
        &clock,
    );
    let sealed: Value = serde_json::from_slice(changed.canonical_bytes()).unwrap();
    assert_eq!(
        sealed["input"]["original_go"]["original_parsed_limits"]["white_time_ms"],
        123456
    );
    assert_eq!(
        sealed["input"]["original_go"]["original_parsed_limits"]["black_time_ms"],
        234567
    );
    assert_eq!(
        sealed["input"]["original_go"]["original_parsed_limits"]["moves_to_go"],
        17
    );
    assert_eq!(
        sealed["input"]["original_go"]["original_parsed_limits"]["ponder"],
        true
    );
    assert_eq!(changed.original_go(), other.original_go.as_deref().unwrap());
    assert_eq!(
        changed.hard_deadline(),
        clock
            .deadline(other.original_time_budget.as_ref().unwrap().hard_deadline)
            .unwrap()
    );
    assert_ne!(changed.root_budget_digest(), first.root_budget_digest());
}

#[test]
fn raw_searchmoves_repetition_and_order_join_the_session_canonical_subset() {
    let position = Position::startpos();
    let lock = lock(wire(&position));
    let clock = ProcessClock::new(ProcessEpoch(7112));
    let original = context(&clock);
    let first = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &original,
        &clock,
    );
    let mut repeated = original.clone();
    Arc::make_mut(repeated.original_go.as_mut().unwrap()).searchmoves =
        Some(vec!["d2d4".into(), "e2e4".into(), "e2e4".into()]);
    refresh_start(&position, &clock, &mut repeated);
    let admitted = admission(
        &lock,
        RoventRootAdmissionPolicy::ExactRegistered,
        &position,
        &repeated,
        &clock,
    );
    assert_eq!(first.allowed_root_moves(), admitted.allowed_root_moves());
    assert_eq!(
        admitted.original_go().searchmoves.as_ref().unwrap().len(),
        3
    );
    let repeated_dynamic = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &repeated,
        &clock,
    );
    assert_ne!(
        first.root_budget_digest(),
        repeated_dynamic.root_budget_digest()
    );
    let mut many = original;
    many.root_moves = Some(Arc::from([BoardMove::from_uci("e2e4").unwrap()]));
    Arc::make_mut(many.original_go.as_mut().unwrap()).searchmoves = Some(vec!["e2e4".into(); 219]);
    refresh_start(&position, &clock, &mut many);
    let admitted = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &many,
        &clock,
    );
    assert_eq!(admitted.allowed_root_moves().len(), 1);
    assert_eq!(
        admitted.original_go().searchmoves.as_ref().unwrap().len(),
        219
    );
}

#[test]
fn pure_preadmission_checks_workload_without_creating_or_reviving_root_authority() {
    let position = Position::startpos();
    let lock = lock(wire(&position));
    let clock = ProcessClock::new(ProcessEpoch(7114));
    let context = context(&clock);
    let mut go = context.original_go.as_deref().unwrap().clone();
    let max = NonZeroU16::new(64).unwrap();
    let observed = preadmission::validate_go(
        &position,
        &go,
        &lock,
        RoventRootAdmissionPolicy::ExactRegistered,
        max,
    )
    .unwrap();
    assert_eq!(observed.requested_nodes(), Some(7));
    assert!(observed.root_snapshot().same_state(&position.snapshot()));
    go.nodes = Some(0);
    assert_eq!(
        preadmission::validate_go(
            &position,
            &go,
            &lock,
            RoventRootAdmissionPolicy::ExactRegistered,
            max
        )
        .unwrap_err()
        .kind,
        RoventRootAdmissionFailure::RegisteredWorkload
    );
    let dynamic = preadmission::validate_go(
        &position,
        &go,
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        max,
    )
    .unwrap();
    assert_eq!(dynamic.requested_nodes(), Some(0));
    context.cancellation.cancel();
    // Pure declarations stay inspectable, but do not revive this root.
    assert!(
        preadmission::validate_go(
            &position,
            &go,
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            max
        )
        .is_ok()
    );
    assert!(!context.accepts().unwrap());
}

#[test]
fn v2_preadmission_uses_original_lock_and_the_same_rules_and_go_checks() {
    let position = Position::startpos();
    let original = lock(wire(&position));
    let mut value = serde_json::to_value(&original.manifest).unwrap();
    let fields = value.as_object_mut().unwrap();
    fields.insert(
        "domain".into(),
        json!(rz_experiments::ROVENT_EXECUTION_V2_DOMAIN),
    );
    fields.insert("startup_document".into(), artifact("startup-v4"));
    fields.insert("startup_configuration_sha256".into(), json!("4".repeat(64)));
    fields.insert(
        "startup_search_configuration_sha256".into(),
        json!(
            original
                .manifest
                .engine
                .search
                .search_configuration_sha256
                .clone()
        ),
    );
    fields.insert(
        "own_r_root_tie".into(),
        json!({"kind":"not_applicable", "reason":"pc_only"}),
    );
    let v2 = rz_experiments::RoventExecutionManifestV2::from_json(&value.to_string())
        .unwrap()
        .lock()
        .unwrap();
    let clock = ProcessClock::new(ProcessEpoch(7115));
    let context = context(&clock);
    let mut go = context.original_go.as_deref().unwrap().clone();
    let max = NonZeroU16::new(64).unwrap();
    let policy = RoventRootAdmissionPolicy::ExactRegistered;
    let v1_result = preadmission::validate_go(&position, &go, &original, policy, max).unwrap();
    let v2_result = preadmission::validate_go_v2(&position, &go, &v2, policy, max).unwrap();
    assert_eq!(
        v2_result.startup_lock_digest(),
        parse_digest(&v2.canonical_sha256).unwrap()
    );
    assert_ne!(
        v2_result.startup_lock_digest(),
        v1_result.startup_lock_digest()
    );
    assert_eq!(
        v2_result.state_and_history_digest(),
        v1_result.state_and_history_digest()
    );
    assert_eq!(
        v2_result.rules_profile_digest(),
        v1_result.rules_profile_digest()
    );
    assert_eq!(v2_result.goal(), v1_result.goal());
    assert_eq!(
        v2_result.allowed_root_moves(),
        v1_result.allowed_root_moves()
    );
    go.nodes = Some(0);
    assert_eq!(
        preadmission::validate_go_v2(&position, &go, &v2, policy, max)
            .unwrap_err()
            .kind,
        RoventRootAdmissionFailure::RegisteredWorkload
    );
    assert_eq!(
        preadmission::validate_go_v2(
            &position,
            &go,
            &v2,
            RoventRootAdmissionPolicy::DynamicGo,
            max
        )
        .unwrap()
        .requested_nodes(),
        Some(0)
    );
    let mut invalid = v2;
    invalid.canonical_sha256 = "0".repeat(64);
    assert_eq!(
        preadmission::validate_go_v2(&position, &go, &invalid, policy, max)
            .unwrap_err()
            .kind,
        RoventRootAdmissionFailure::InvalidLock
    );
}

#[test]
fn persisted_metadata_verification_cannot_revive_a_revoked_live_root() {
    let position = Position::startpos();
    let lock = lock(wire(&position));
    let clock = ProcessClock::new(ProcessEpoch(7113));
    let context = context(&clock);
    let live = admission(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &context,
        &clock,
    );
    let persisted =
        RoventRootDescriptor::from_json(std::str::from_utf8(live.canonical_bytes()).unwrap())
            .unwrap();
    assert_eq!(persisted.digest().unwrap(), live.digest());
    persisted.verify_against(&lock).unwrap();
    context.cancellation.cancel();
    // Immutable historical metadata remains valid; the original shared root
    // authority is still revoked and no JSON-to-live constructor is available.
    persisted.verify_against(&lock).unwrap();
    assert!(!live.original_context().accepts().unwrap());
    assert_eq!(
        RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &context,
            &clock
        )
        .unwrap_err()
        .kind,
        RoventRootAdmissionFailure::RevokedAuthority
    );
}
