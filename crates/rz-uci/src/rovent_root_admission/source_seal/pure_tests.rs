//! Source/Rules/control fixtures only. No engine, provider or worker starts.

#[path = "shared_teacher_scope_tests.rs"]
mod shared_teacher_scope_tests;

use super::*;
use crate::bridge::BuildSearchSettings;
use crate::engine::driver_start_fixture::{FixtureStart, issue_start};
use crate::parser::GoLimits;
use rz_contracts::{GameGeneration, ProcessEpoch, RootGeneration, pals::SearchAuthority};
use rz_experiments::{
    ROVENT_EXECUTION_DOMAIN, ROVENT_POSITION_VISIT_UNIT, RoventExecutionManifest,
};
use rz_search::driver::SearchControl;
use rz_search::time::{TimeBudget, TimeBudgetConfig, TimeControl};
use serde_json::{Value, json};
use std::{
    num::NonZeroU16,
    time::{Duration, Instant},
};

fn artifact(name: &str) -> Value {
    json!({"path":format!("fixture/{name}"),"sha256":"a".repeat(64),"bytes":32,
        "source":"https://example.org/source-only-root-fixture","license":"fixture-only"})
}

fn input_lock(position: &Position) -> RoventInputLock {
    let model = json!({"semantics":"fixture-pc","encoding":"fixture-encoding",
        "policy_head":"legal-candidates/1","value_head":"side-to-move-wdl/1",
        "implementation_sha256":"b".repeat(64),"weights":artifact("weights"),
        "provenance":artifact("model-provenance"),"diagnostic":false});
    let value = json!({
        "domain":ROVENT_EXECUTION_DOMAIN,
        "contract_revision":rz_contracts::rovent::ROVENT_SEARCH_REVISION,
        "run_id":"initial-source-seal-fixture","purpose":"validation",
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
                "searchmoves":null,"search_configuration_sha256":"1".repeat(64),
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
        "agent_renewable":true,"provenance":[artifact("source-provenance")]});
    RoventExecutionManifest::from_json(&value.to_string())
        .unwrap()
        .lock()
        .unwrap()
}

fn input_lock_v2(position: &Position) -> rz_experiments::RoventOriginalInputLock {
    let base = input_lock(position);
    let mut wire = serde_json::to_value(&base.manifest).unwrap();
    let object = wire.as_object_mut().unwrap();
    object.insert(
        "domain".into(),
        json!(rz_experiments::ROVENT_EXECUTION_V2_DOMAIN),
    );
    object.insert("startup_document".into(), artifact("startup-v4"));
    object.insert("startup_configuration_sha256".into(), json!("4".repeat(64)));
    object.insert(
        "startup_search_configuration_sha256".into(),
        json!(base.manifest.engine.search.search_configuration_sha256),
    );
    object.insert(
        "own_r_root_tie".into(),
        json!({"kind":"not_applicable", "reason":"pc_only"}),
    );
    let lock = rz_experiments::RoventExecutionManifestV2::from_json(&wire.to_string())
        .unwrap()
        .lock()
        .unwrap();
    rz_experiments::RoventOriginalInputLock::from_v2(Arc::new(lock)).unwrap()
}

#[test]
fn v2_prepared_source_uses_actual_start_and_preserves_original_owner_without_live_authority() {
    let position = Position::startpos();
    let clock = ProcessClock::new(ProcessEpoch(7500));
    let context = context(&clock, false);
    let owner = input_lock_v2(&position);
    let prepared = PreparedSourceV2::prepare(
        &owner,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &context,
        &clock,
    )
    .unwrap();
    assert!(prepared.owner().same_original_owner(&owner));
    assert_eq!(
        prepared.descriptor().input.startup_lock_sha256,
        owner.v2().unwrap().canonical_sha256
    );
    assert_eq!(
        prepared.descriptor().input.allowed_root_moves,
        position
            .ordered_legal_moves()
            .moves()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
    prepared
        .allocation_clock()
        .verify_against(prepared.descriptor(), owner.v2().unwrap())
        .unwrap();
    let mut invalid = context.clone();
    Arc::make_mut(invalid.original_go.as_mut().unwrap()).nodes = Some(8);
    let error = PreparedSourceV2::prepare(
        &owner,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &invalid,
        &clock,
    )
    .unwrap_err();
    assert_eq!(error.kind, RoventRootAdmissionFailure::OriginalGoMismatch);
    let v1 =
        rz_experiments::RoventOriginalInputLock::from_v1(Arc::new(input_lock(&position))).unwrap();
    assert_eq!(
        PreparedSourceV2::prepare(
            &v1,
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
fn v2_final_gate_retains_exact_owner_and_rejects_replaced_or_canceled_sources() {
    let position = Position::startpos();
    let clock = ProcessClock::new(ProcessEpoch(7509));
    let context = context(&clock, false);
    let owner = input_lock_v2(&position);
    let prepare_v2 = || {
        PreparedSourceV2::prepare(
            &owner,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &context,
            &clock,
        )
        .unwrap()
    };

    let admitted = prepare_v2().try_into_live().unwrap();
    assert!(admitted.owner().same_original_owner(&owner));
    assert_eq!(admitted.original_context().authority, context.authority);
    assert!(admitted.root_snapshot().same_state(&position.snapshot()));
    admitted
        .allocation_clock()
        .verify_against(admitted.descriptor(), owner.v2().unwrap())
        .unwrap();

    let canceled = prepare_v2();
    context.control.cancel();
    let (canceled, error) = canceled.try_into_live().unwrap_err();
    assert_eq!(error.kind, RoventRootAdmissionFailure::RevokedAuthority);
    context.control.cancellation.store(false, Ordering::Release);
    match canceled.into_initial().unwrap() {
        InitialAdmissionOutcomeV2::StoppedInitial(stopped) => {
            assert_eq!(stopped.stop_reason(), InitialStopReason::Canceled);
            assert!(stopped.owner().same_original_owner(&owner));
            assert_eq!(
                stopped.fallback(),
                stopped.allowed_root_moves().first().copied()
            );
        }
        InitialAdmissionOutcomeV2::Admitted(_) => panic!("canceled /2 source became live"),
    }

    let replaced = prepare_v2();
    let mut other = context.authority;
    other.implementation = Digest([8; 32]);
    *context.current.lock().unwrap() = SessionScope::Search(other);
    context.cancellation.cancel();
    match replaced.into_initial().unwrap() {
        InitialAdmissionOutcomeV2::StoppedInitial(stopped) => {
            assert_eq!(stopped.stop_reason(), InitialStopReason::RootReplaced);
        }
        InitialAdmissionOutcomeV2::Admitted(_) => panic!("replaced /2 source became live"),
    }
}

#[test]
fn v2_expired_source_is_stopped_before_live_admission() {
    let position = Position::startpos();
    let clock = ProcessClock::new(ProcessEpoch(7510));
    let context = context(&clock, true);
    let owner = input_lock_v2(&position);
    let source = PreparedSourceV2::prepare(
        &owner,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &context,
        &clock,
    )
    .unwrap();
    match source.into_initial().unwrap() {
        InitialAdmissionOutcomeV2::StoppedInitial(stopped) => {
            assert_eq!(stopped.stop_reason(), InitialStopReason::Deadline);
            assert!(stopped.owner().same_original_owner(&owner));
        }
        InitialAdmissionOutcomeV2::Admitted(_) => panic!("expired /2 source became live"),
    }
}

fn context(clock: &ProcessClock, expired: bool) -> SearchSessionContext {
    let authority = SearchAuthority {
        epoch: clock.epoch(),
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest([9; 32]),
    };
    let go = GoLimits {
        nodes: Some(7),
        depth_plies: NonZeroU16::new(2),
        movetime_ms: expired.then_some(0),
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
fn refresh_start(
    position: &Position,
    clock: &ProcessClock,
    context: &mut SearchSessionContext,
    start: Instant,
    settings: BuildSearchSettings,
) {
    let go = context.original_go.as_deref().unwrap().clone();
    *context = match issue_start(
        position,
        go,
        context.authority,
        clock,
        start,
        settings,
        Duration::from_millis(500),
    )
    .unwrap()
    {
        FixtureStart::Started(context) => context,
        FixtureStart::RulesOnly(_) => {
            panic!("successful source fixture requires actual Session Start")
        }
    };
}
fn prepare(
    position: &Position,
    context: &SearchSessionContext,
    clock: &ProcessClock,
) -> SourceSeal {
    SourceSeal::prepare(
        &input_lock(position),
        RoventRootAdmissionPolicy::DynamicGo,
        position,
        context,
        clock,
    )
    .unwrap()
}

fn stopped(source: SourceSeal) -> StoppedInitial {
    match source.into_initial().unwrap() {
        InitialAdmissionOutcome::StoppedInitial(stopped) => stopped,
        InitialAdmissionOutcome::Admitted(_) => panic!("stopped original source became live"),
    }
}

#[test]
fn stopped_source_preserves_original_slash_one_bytes_and_rules_ordered_fallback() {
    let position = Position::startpos();
    let clock = ProcessClock::new(ProcessEpoch(7501));
    let mut context = context(&clock, false);
    let raw = vec!["e2e4".into(), "d2d4".into(), "e2e4".into()];
    Arc::make_mut(context.original_go.as_mut().unwrap()).searchmoves = Some(raw);
    context.root_moves = Some(Arc::from([
        BoardMove::from_uci("e2e4").unwrap(),
        BoardMove::from_uci("d2d4").unwrap(),
    ]));
    refresh_start(
        &position,
        &clock,
        &mut context,
        clock.runtime_clock().origin(),
        start_settings(),
    );
    let lock = input_lock(&position);
    let live = RoventRootAdmission::admit(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &context,
        &clock,
    )
    .unwrap();
    let sealed_live = prepare(&position, &context, &clock)
        .try_into_live()
        .unwrap();
    assert_eq!(sealed_live.canonical_bytes(), live.canonical_bytes());
    assert_eq!(sealed_live.authority(), context.authority);
    assert!(sealed_live.root_snapshot().same_state(&position.snapshot()));
    assert_eq!(sealed_live.hard_deadline(), live.hard_deadline());
    assert_eq!(
        sealed_live.runtime_clock().origin(),
        live.runtime_clock().origin()
    );
    let source = prepare(&position, &context, &clock);
    assert_eq!(
        source.allocation_clock_digest(),
        sealed_live.allocation_clock_digest()
    );
    assert_eq!(
        source.allocation_clock_canonical_bytes(),
        sealed_live.allocation_clock_canonical_bytes()
    );
    source
        .allocation_clock()
        .verify_against(live.descriptor(), &lock)
        .unwrap();
    // The original control can change after metadata is sealed. The source
    // must retain its source bytes while refusing to issue live authority.
    context.control.cancel();
    let stopped = stopped(source);
    assert_eq!(stopped.stop_reason(), InitialStopReason::Canceled);
    assert_eq!(stopped.canonical_bytes(), live.canonical_bytes());
    assert_eq!(stopped.digest(), live.digest());
    assert_eq!(stopped.root_budget_digest(), live.root_budget_digest());
    assert_eq!(stopped.deadline_seal_digest(), live.deadline_seal_digest());
    assert_eq!(stopped.allowed_root_moves(), live.allowed_root_moves());
    assert_eq!(stopped.allocation_clock(), sealed_live.allocation_clock());
    assert_eq!(
        stopped.allocation_clock_digest(),
        sealed_live.allocation_clock_digest()
    );
    assert_eq!(
        stopped.allocation_clock_canonical_bytes(),
        sealed_live.allocation_clock_canonical_bytes()
    );
    let allocation = &stopped.allocation_clock().input;
    let original_budget = context.original_time_budget.as_ref().unwrap();
    assert_eq!(
        allocation.start.at_ns,
        clock.tick_at(original_budget.start).unwrap().0
    );
    assert_eq!(
        allocation.output.at_ns,
        clock.tick_at(original_budget.output_deadline).unwrap().0
    );
    assert_eq!(
        allocation.allocation_ns,
        original_budget.allocation.as_nanos() as u64
    );
    assert_eq!(
        stopped.fallback(),
        live.allowed_root_moves().first().copied()
    );
}

#[test]
fn expired_original_source_is_metadata_only_and_reason_priority_is_exact_scope_first() {
    let position = Position::startpos();
    let clock = ProcessClock::new(ProcessEpoch(7502));
    let context = context(&clock, true);
    assert_eq!(
        stopped(prepare(&position, &context, &clock)).stop_reason(),
        InitialStopReason::Deadline
    );
    context.cancellation.cancel();
    assert_eq!(
        stopped(prepare(&position, &context, &clock)).stop_reason(),
        InitialStopReason::Canceled
    );
    let mut replaced = context.authority;
    replaced.implementation = Digest([8; 32]);
    *context.current.lock().unwrap() = SessionScope::Search(replaced);
    assert_eq!(
        stopped(prepare(&position, &context, &clock)).stop_reason(),
        InitialStopReason::RootReplaced
    );
}

#[test]
fn rejected_source_cannot_be_retried_into_live_after_control_is_restored() {
    let position = Position::startpos();
    let clock = ProcessClock::new(ProcessEpoch(7503));
    let context = context(&clock, false);
    let source = prepare(&position, &context, &clock);
    context.control.cancel();
    let (source, error) = source.try_into_live().unwrap_err();
    assert_eq!(error.kind, RoventRootAdmissionFailure::RevokedAuthority);
    context.control.cancellation.store(false, Ordering::Release);
    assert!(context.accepts().unwrap());
    assert_eq!(stopped(source).stop_reason(), InitialStopReason::Canceled);
}

#[test]
fn terminal_rules_publish_without_start_and_missing_actual_source_is_explicitly_rejected() {
    let position = Position::from_fen("7k/8/8/8/8/8/8/K7 w - - 0 1").unwrap();
    let clock = ProcessClock::new(ProcessEpoch(7504));
    let mut context = context(&clock, true);
    let go = context.original_go.as_deref().unwrap().clone();
    match issue_start(
        &position,
        go,
        context.authority,
        &clock,
        clock.runtime_clock().origin(),
        start_settings(),
        Duration::from_millis(500),
    )
    .unwrap()
    {
        FixtureStart::RulesOnly(out) => {
            assert!(out.accepted);
            assert_eq!(out.protocol, vec!["bestmove 0000".to_owned()]);
            assert!(
                !out.effects
                    .iter()
                    .any(|effect| matches!(effect, crate::Effect::Start { .. }))
            );
        }
        FixtureStart::Started(_) => panic!("terminal Rules invented an actual Start"),
    }
    assert!(!position.ordered_legal_moves().moves().is_empty());
    context.original_start = None;
    let error = SourceSeal::prepare(
        &input_lock(&position),
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &context,
        &clock,
    )
    .unwrap_err();
    assert_eq!(error.kind, RoventRootAdmissionFailure::Clock);
    assert_eq!(error.detail, "actual Start source missing");
}

#[test]
fn poisoned_authority_and_invalid_source_do_not_become_normal_stops() {
    let position = Position::startpos();
    let clock = ProcessClock::new(ProcessEpoch(7505));
    let mut context = context(&clock, true);
    let source = prepare(&position, &context, &clock);
    let current = Arc::clone(&context.current);
    let _panic = std::panic::catch_unwind(move || {
        let _owner = current.lock().unwrap();
        panic!("source-only mutex poison fixture");
    });
    let error = source.into_initial().unwrap_err();
    assert_eq!(error.kind, RoventRootAdmissionFailure::RevokedAuthority);
    assert!(error.metadata_error.is_some());
    context.original_go = None;
    let error = SourceSeal::prepare(
        &input_lock(&position),
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &context,
        &clock,
    )
    .unwrap_err();
    assert_eq!(error.kind, RoventRootAdmissionFailure::OriginalGoMissing);
}

#[test]
fn allocation_start_and_whole_duration_are_separate_from_unchanged_slash_one_metadata() {
    let position = Position::startpos();
    let clock = ProcessClock::new(ProcessEpoch(7506));
    let first_context = context(&clock, false);
    let mut second_context = first_context.clone();
    // Both original allocations end at exactly the same boundaries. Only their
    // actual start and whole allocation differ; `/1` carries neither field.
    let start = clock.runtime_clock().origin() + Duration::from_millis(1);
    let mut second_settings = start_settings();
    second_settings.untimed_limit = Duration::from_millis(59_999);
    refresh_start(
        &position,
        &clock,
        &mut second_context,
        start,
        second_settings,
    );
    let second_budget = second_context.original_time_budget.as_ref().unwrap();
    assert_eq!(
        second_budget.soft_deadline,
        first_context.control.soft_deadline
    );
    assert_eq!(
        second_budget.admission_deadline,
        first_context.control.admission_deadline
    );
    assert_eq!(second_budget.hard_deadline, first_context.control.deadline);
    let first = prepare(&position, &first_context, &clock);
    let second = prepare(&position, &second_context, &clock);
    assert_eq!(first.canonical, second.canonical);
    assert_eq!(first.digest, second.digest);
    assert_eq!(first.root_budget, second.root_budget);
    assert_eq!(first.deadline_seal, second.deadline_seal);
    assert_ne!(
        first.allocation_clock_digest(),
        second.allocation_clock_digest()
    );
    assert_eq!(first.allocation_clock().input.start.at_ns, 0);
    assert_eq!(second.allocation_clock().input.start.at_ns, 1_000_000);
    assert_eq!(
        first.allocation_clock().input.output,
        second.allocation_clock().input.output
    );
}

#[test]
fn original_allocation_must_join_every_actual_control_boundary_exactly() {
    let position = Position::startpos();
    let clock = ProcessClock::new(ProcessEpoch(7507));
    let lock = input_lock(&position);
    for boundary in 0..3 {
        let mut context = context(&clock, false);
        match boundary {
            0 => context.control.soft_deadline -= Duration::from_nanos(1),
            1 => context.control.admission_deadline += Duration::from_nanos(1),
            2 => context.control.deadline -= Duration::from_nanos(1),
            _ => unreachable!(),
        }
        let error = SourceSeal::prepare(
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &context,
            &clock,
        )
        .unwrap_err();
        assert_eq!(error.kind, RoventRootAdmissionFailure::Clock);
        assert_eq!(
            error.detail,
            "original TimeBudget boundaries differ from the actual search control"
        );
    }
}

#[test]
fn missing_inconsistent_or_out_of_origin_allocation_is_not_a_normal_stop() {
    let position = Position::startpos();
    let clock = ProcessClock::new(ProcessEpoch(7508));
    let lock = input_lock(&position);
    let mut missing = context(&clock, true);
    missing.original_time_budget = None;
    let error = SourceSeal::prepare(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &missing,
        &clock,
    )
    .unwrap_err();
    assert_eq!(error.kind, RoventRootAdmissionFailure::Clock);
    assert_eq!(
        error.detail,
        "actual admission requires the original allocated TimeBudget"
    );

    for change_output in [false, true] {
        let mut inconsistent = context(&clock, false);
        let budget = inconsistent.original_time_budget.as_mut().unwrap();
        if change_output {
            budget.output_deadline += Duration::from_nanos(1);
        } else {
            budget.allocation -= Duration::from_nanos(1);
        }
        let error = SourceSeal::prepare(
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &inconsistent,
            &clock,
        )
        .unwrap_err();
        assert_eq!(error.kind, RoventRootAdmissionFailure::Clock);
        assert_eq!(
            error.detail,
            "original TimeBudget allocation differs from output minus actual start"
        );
    }

    let origin = clock.runtime_clock().origin();
    let mut outside_origin = context(&clock, false);
    let mut budget = TimeBudget::new(
        origin,
        TimeControl::MoveTime(Duration::from_secs(1)),
        TimeBudgetConfig::default(),
    )
    .unwrap();
    budget.start = origin.checked_sub(Duration::from_nanos(1)).unwrap();
    budget.allocation = budget
        .output_deadline
        .checked_duration_since(budget.start)
        .unwrap();
    outside_origin.control = SearchControl::from_budget(&budget, 4096);
    outside_origin.shutdown_deadline = budget.hard_deadline + Duration::from_millis(500);
    outside_origin.original_time_budget = Some(budget);
    let error = SourceSeal::prepare(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &outside_origin,
        &clock,
    )
    .unwrap_err();
    assert_eq!(error.kind, RoventRootAdmissionFailure::Clock);
    assert_eq!(
        error.detail,
        "deadline predates or overflows the original process origin"
    );
}

#[test]
fn allocation_wall_ceiling_covers_output_reserve_as_well_as_hard_work() {
    let position = Position::startpos();
    let clock = ProcessClock::new(ProcessEpoch(7509));
    let lock = input_lock(&position);
    let mut context = context(&clock, false);
    // hard is still below 60s, but the actual whole allocation exceeds 60s.
    let budget = TimeBudget::new(
        clock.runtime_clock().origin(),
        TimeControl::MoveTime(Duration::from_millis(60_001)),
        TimeBudgetConfig::default(),
    )
    .unwrap();
    context.control = SearchControl::from_budget(&budget, 4096);
    context.shutdown_deadline = budget.hard_deadline + Duration::from_millis(500);
    context.original_time_budget = Some(budget);
    let error = SourceSeal::prepare(
        &lock,
        RoventRootAdmissionPolicy::DynamicGo,
        &position,
        &context,
        &clock,
    )
    .unwrap_err();
    assert_eq!(error.kind, RoventRootAdmissionFailure::DeclaredCeiling);
    assert_eq!(
        error.detail,
        "original whole allocation exceeds the declared wall ceiling"
    );
}
