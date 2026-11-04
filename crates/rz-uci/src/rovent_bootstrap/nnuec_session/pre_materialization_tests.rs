//! Actual Rules retirement -> private UCI producer -> original diagnostics.
//! No NN Request, C view, TT, R worker or forward is created. The initial-cleanup
//! negative uses one real actor cache-release ACK, without a submitted Request.
use super::*;
use crate::bridge::BuildSearchSettings;
use crate::engine::driver_start_fixture::{FixtureStart, issue_start};
use crate::parser::GoLimits;
use rz_contracts::rovent::ExecutionMode;
use rz_contracts::{GameGeneration, OwnerId, PrecisionProfile, ProcessEpoch, RootGeneration};
use rz_experiments::RoventExecutionManifest;
use rz_search::cpu::{CpuConfig, CpuOrderingPolicy};
use rz_search::cpu_value::CpuTrainingState;
use rz_search::cpu_value::nnuec::v2::{Checkpoint, Model, RelationParameters, SCHEMA, SEMANTICS};
use rz_search::cpu_value::nnuec::{
    NNUEC_SCHEMA, NNUEC_SEMANTICS, NnuecCheckpoint, NnuecConfig, NnuecParameters,
};
use rz_search::pals::engine::PostRepairRecheckPolicy;
use rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace;
use rz_search::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
use rz_search::pals::value::{MODEL_WDL_VALUE_SEMANTICS, ModelValueIdentity};
use rz_search::time::TimeBudgetConfig;
use serde_json::{Value, json};
use std::sync::atomic::AtomicU64;

pub(in crate::rovent_bootstrap) fn model_source() -> (DeferredNnuecRSource, Arc<Model>) {
    let checkpoint = Checkpoint {
        schema: SCHEMA.into(),
        semantics: SEMANTICS.into(),
        base: NnuecCheckpoint {
            schema: NNUEC_SCHEMA.into(),
            semantics: NNUEC_SEMANTICS.into(),
            training: CpuTrainingState::Untrained,
            output_scale: 1.0,
            config: NnuecConfig {
                board_width: 1,
                line_width: 1,
                context_width: 1,
                local_width: 1,
                hidden_width: 1,
                causal_window: 3,
                max_lines: 8,
                max_plies: 16,
            },
            parameters: NnuecParameters {
                board_weights: vec![0.0; 800],
                board_bias: vec![0.0],
                anchor_weights: vec![0.0; 800],
                anchor_bias: vec![0.0],
                from_weights: vec![0.0; 64],
                to_weights: vec![0.0; 64],
                promotion_weights: vec![0.0; 5],
                ply_weights: vec![0.0; 16],
                token_bias: vec![0.0],
                causal_weights: vec![0.0; 3],
                causal_bias: vec![0.0],
                context_projection: vec![0.0],
                context_bias: vec![0.0],
                local_projection: vec![0.0],
                local_bias: vec![0.0],
                hidden_weights: vec![0.0; 4],
                hidden_bias: vec![0.0],
                output_weights: vec![0.0; 11],
                output_bias: vec![0.0; 11],
            },
        },
        parameters: RelationParameters {
            kind_embeddings: vec![0.0; 4],
            pair_parent: vec![0.0; 2],
            pair_child: vec![0.0; 2],
            pair_bias: vec![0.0; 2],
            pair_projection: vec![0.0],
        },
    };
    let plan = DeferredNnuecCpuPlan::new(
        CpuConfig {
            tt_entries: 16,
            max_depth: 2,
            quiescence_ply: 0,
            ..CpuConfig::default()
        },
        CpuOrderingPolicy::LegacyMvvLvaV1,
        128 * 1024,
        false,
        3,
        None,
    )
    .unwrap();
    let model = Arc::new(Model::new(checkpoint).unwrap());
    (DeferredNnuecRSource::new(Arc::clone(&model), plan), model)
}

struct Fixture {
    retired: RetiredPreparingRoot<AdaptivePortModel>,
    admission: Arc<RoventRootAdmission>,
    context: SearchSessionContext,
    clock: ProcessClock,
    model: Arc<Model>,
}

fn retired_fixture() -> Fixture {
    let position = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
    let movement: BoardMove = "f7g7".parse().unwrap();
    let clock = ProcessClock::new(ProcessEpoch(34881));
    let authority = SearchAuthority {
        epoch: clock.epoch(),
        game: GameGeneration(1),
        root: RootGeneration(6),
        implementation: Digest([81; 32]),
    };
    let go = GoLimits {
        searchmoves: Some(vec!["f7g7".into()]),
        mate_moves: NonZeroU16::new(1),
        nodes: Some(16),
        ..GoLimits::default()
    };
    let context = match issue_start(
        &position,
        go,
        authority,
        &clock,
        clock.runtime_clock().origin(),
        BuildSearchSettings {
            time_config: TimeBudgetConfig::default(),
            max_simulations: 4096,
            untimed_limit: Duration::from_secs(20),
        },
        Duration::from_secs(1),
    )
    .unwrap()
    {
        FixtureStart::Started(context) => context,
        FixtureStart::RulesOnly(_) => panic!("ongoing mate fixture did not issue original Start"),
    };
    let mut wire = super::super::tests::wire(ExecutionMode::CpuPcr);
    wire["engine"]["search"]["root_state_and_history_sha256"] = json!(
        rz_position::contracts::rules_snapshot_semantic_digest(&position.snapshot())
            .0
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    wire["engine"]["search"]["rules_profile_sha256"] = json!(
        rz_position::contracts::profile_digest()
            .0
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    let lock = RoventExecutionManifest::from_json(&wire.to_string())
        .unwrap()
        .lock()
        .unwrap();
    let admission = Arc::new(
        RoventRootAdmission::admit(
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
            &position,
            &context,
            &clock,
        )
        .unwrap(),
    );
    let value = ModelValueIdentity {
        semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
        model: "retirement-role-fixture".into(),
        encoding: "retirement-encoding-fixture".into(),
        precision: "fp32".into(),
        model_epoch: [7; 32],
    };
    let namespace = AdaptiveModelNamespace::new(
        value.clone(),
        Digest([5; 32]),
        Digest([6; 32]),
        PrecisionProfile::Fp32,
        1,
    )
    .unwrap();
    let role_model =
        AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value), namespace).unwrap();
    let (source, model) = model_source();
    let owner = ResearcherPreparingOwner::new_selected(
        ResearcherSemanticProfile {
            config: PalsConfig {
                line_plies: 2,
                max_nodes: 257,
                max_records: 32,
                max_role_calls: 32,
                beam_width: 1,
                cpu_nodes_per_task: 256,
            },
            maximum_root_prefix_plies: 16,
        },
        role_model,
        ResolverPolicy::ModelWdlRestricted,
        PostRepairRecheckPolicy::Disabled,
        ReviewSelectionPolicy::default(),
        ReviewIndexLimits::default(),
        source,
    )
    .unwrap();
    let store = owner.stores().read_owner();
    let mut root = owner
        .bind_root(
            position,
            authority,
            admission.hard_deadline(),
            context.control.deadline,
            Arc::clone(&context.control.cancellation),
        )
        .map_err(|(_, error)| error)
        .unwrap();
    let next = Arc::new(AtomicU64::new(1));
    let issuer: AdaptiveRulesOwnerIssuer =
        Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::Relaxed))));
    root.attach_original_controls(
        ResearcherRootStartup {
            mode: ExecutionMode::CpuPcr,
            root_budget: admission.root_budget_digest(),
            goal: context.goal,
            nodes: context.requested_nodes,
            selective_policy: Digest([83; 32]),
            frontier_limits: FrontierLimits {
                maximum_goal_plies: NonZeroU16::new(16).unwrap(),
                maximum_frames: 16,
                maximum_paths: 64,
                maximum_mate_nodes: 64,
                maximum_native_receipts: 8,
            },
            clock: clock.b_clock(),
            limits: PalsLimits {
                deadline: context.control.deadline,
                max_rounds: 4,
                max_cpu_nodes: 4096,
                cpu_depth: 2,
            },
            cleanup_deadline: context.shutdown_deadline,
        },
        issuer,
        Some(&[movement]),
    )
    .unwrap();
    root.prepare_original_root_rules().unwrap();
    let retired = root
        .retire_before_materialization()
        .map_err(|(_, error)| error)
        .unwrap();
    assert!(store.matches(retired.owner.stores()));
    assert_eq!(retired.owner.stores().review_pin_count().unwrap(), 0);
    let original = retired.frontier.as_ref().unwrap();
    assert_eq!(original.snapshot.goal, FrontierGoalStatus::MateSatisfied);
    assert_eq!(original.snapshot.work.checked_transitions, 3);
    assert_eq!(
        original
            .snapshot
            .mate_strategy
            .as_ref()
            .unwrap()
            .best_move(),
        movement
    );
    Fixture {
        retired,
        admission,
        context,
        clock,
        model,
    }
}

fn observe_fixture(
    fixture: Fixture,
    initial: Option<Arc<RoventNnuecInitialRetirementReceipt>>,
) -> (RoventNnuecRootReceipt, SearchSessionContext) {
    let store = fixture.retired.owner.stores().read_owner();
    let input_lock = fixture.admission.startup_lock_digest();
    let (retired, receipt) = preparing_retirement_receipt(
        fixture.retired,
        &fixture.context,
        fixture.admission,
        input_lock,
        &fixture.clock,
        4096,
        initial,
    )
    .unwrap_or_else(|_| panic!("actual original Rules retirement was rejected"));
    assert!(store.matches(retired.owner.stores()));
    assert_eq!(retired.owner.stores().review_pin_count().unwrap(), 0);
    ((*receipt).clone(), fixture.context)
}

fn cleanup_marker_without_a_request(model: Arc<Model>) -> Arc<RoventNnuecInitialRetirementReceipt> {
    let mut actor = PreparationActor::spawn(model).unwrap();
    let until = Instant::now() + Duration::from_secs(3);
    let invalidation = actor.invalidate_context_cache().unwrap();
    let cache_release = loop {
        if let Some(ack) = actor.try_context_cache_release(&invalidation).unwrap() {
            break ack;
        }
        if Instant::now() >= until {
            actor.request_shutdown();
            std::mem::forget(actor);
            panic!("bounded no-Request cache ACK did not arrive");
        }
        std::thread::yield_now();
    };
    actor.request_shutdown();
    loop {
        if actor.try_join().unwrap().is_some() {
            break;
        }
        if Instant::now() >= until {
            std::mem::forget(actor);
            panic!("bounded no-Request actor join did not finish");
        }
        std::thread::yield_now();
    }
    // A negative input only: actual cache ACK, no Request or claimed C success.
    // Every Some(initial_retirement), including the real submitted-C cleanup
    // producer, must stay outside the pre-materialization source.
    Arc::new(RoventNnuecInitialRetirementReceipt {
        original_phase: InitialNnuecPhase::PrepareReview,
        original_failure: None,
        release: None,
        cache_release,
        observations: NnuecInitialCleanupObservations::default(),
    })
}

fn assert_not_projected<F>(
    label: &str,
    receipt: &RoventNnuecRootReceipt,
    context: &SearchSessionContext,
    project: &F,
) where
    F: Fn(&RoventNnuecRootReceipt, Digest, &SearchSessionContext) -> Value,
{
    assert!(!receipt.pre_materialization_retirement_matches(), "{label}");
    let row = project(
        receipt,
        receipt.root_admission.startup_lock_digest(),
        context,
    );
    assert!(
        row["root"].is_null() || row["root"]["physical"].is_null(),
        "{label}: {row}"
    );
}

pub(in crate::rovent_bootstrap) fn check_projection<F>(project: F)
where
    F: Fn(&RoventNnuecRootReceipt, Digest, &SearchSessionContext) -> Value,
{
    let (receipt, context) = observe_fixture(retired_fixture(), None);
    assert!(receipt.pre_materialization_retirement_matches());
    assert_eq!(receipt.counters, AdaptiveManagerCounters::default());
    let row = project(&receipt, receipt.input_lock, &context);
    assert_eq!(row["observation_status"], "original_current_root");
    assert_eq!(row["root"]["goal_status"], "mate_satisfied");
    let physical = &row["root"]["physical"];
    assert_eq!(
        physical["source"],
        "original_pre_materialization_retirement"
    );
    assert_eq!(physical["semantic_closure_confirmed"], true);
    for field in [
        "native_root_fence_present",
        "native_root_fence_reusable",
        "native_process_joined",
        "native_physical_shutdown_confirmed",
    ] {
        assert_eq!(physical[field], false, "{field}");
    }
    let mut changed = receipt.clone();
    changed.pre_materialization_retirement = None;
    assert_eq!(changed.counters, AdaptiveManagerCounters::default());
    assert_not_projected("absent witness plus counter0", &changed, &context, &project);
    let mut changed = receipt.clone();
    changed.counters.submitted_role_tickets = 1;
    assert_not_projected("nonzero role work", &changed, &context, &project);
    let mut changed = receipt.clone();
    changed.root_admission = Arc::new((*receipt.root_admission).clone());
    assert_eq!(
        changed.root_admission.digest(),
        receipt.root_admission.digest()
    );
    assert!(!Arc::ptr_eq(
        &changed.root_admission,
        &receipt.root_admission
    ));
    assert_not_projected(
        "equal admission bytes in different Arc",
        &changed,
        &context,
        &project,
    );
    let mut changed = receipt.clone();
    changed.authority.root = RootGeneration(changed.authority.root.0 + 1);
    assert_not_projected("changed receipt authority", &changed, &context, &project);
    let mut changed = receipt.clone();
    changed.frontier.as_mut().unwrap().authority.root =
        RootGeneration(context.authority.root.0 + 1);
    assert_not_projected("changed frontier authority", &changed, &context, &project);
    let mut changed = receipt.clone();
    changed.frontier.as_mut().unwrap().original_deadline.at.0 += 1;
    assert_not_projected("changed frontier deadline", &changed, &context, &project);
    let mut changed = receipt.clone();
    Arc::make_mut(changed.pre_materialization_retirement.as_mut().unwrap()).root_budget =
        Digest([99; 32]);
    assert_not_projected("changed original budget", &changed, &context, &project);
    let mut changed = receipt.clone();
    Arc::make_mut(changed.pre_materialization_retirement.as_mut().unwrap()).cleanup_deadline +=
        Duration::from_nanos(1);
    assert_not_projected(
        "changed original cleanup deadline",
        &changed,
        &context,
        &project,
    );
    let mut changed = receipt.clone();
    changed.frontier.as_mut().unwrap().work.checked_transitions += 1;
    assert_not_projected("changed original Rules work", &changed, &context, &project);
    let mut changed = receipt.clone();
    Arc::make_mut(changed.pre_materialization_retirement.as_mut().unwrap())
        .ledger
        .highest_job_sequence = 1;
    assert_not_projected(
        "original ledger has issued a job",
        &changed,
        &context,
        &project,
    );
    let mut changed = receipt.clone();
    Arc::make_mut(changed.pre_materialization_retirement.as_mut().unwrap())
        .ledger
        .clock_origin += Duration::from_nanos(1);
    assert_not_projected(
        "changed original ledger clock",
        &changed,
        &context,
        &project,
    );
    let mut changed = receipt.clone();
    changed.frontier.as_mut().unwrap().goal = FrontierGoalStatus::Pending;
    assert_not_projected("nonterminal copied frontier", &changed, &context, &project);
    let mut changed = receipt.clone();
    changed.deadline_before_frontier = true;
    assert_not_projected("pre-frontier receipt flag", &changed, &context, &project);
    for alter in [false, true] {
        let mut fixture = retired_fixture();
        if alter {
            fixture.retired.cleanup_deadline += Duration::from_nanos(1);
        } else {
            fixture.retired.root_budget = Digest([100; 32]);
        }
        let store = fixture.retired.owner.stores().read_owner();
        let rejected = preparing_retirement_receipt(
            fixture.retired,
            &fixture.context,
            Arc::clone(&fixture.admission),
            fixture.admission.startup_lock_digest(),
            &fixture.clock,
            4096,
            None,
        );
        let (retired, initial, original_ledger) = match rejected {
            Ok(_) => panic!("producer accepted changed original budget/cleanup deadline"),
            Err(rejected) => rejected,
        };
        assert!(store.matches(retired.owner.stores()));
        assert!(initial.is_none());
        assert!(original_ledger.is_ok());
    }
    let fixture = retired_fixture();
    let marker = cleanup_marker_without_a_request(Arc::clone(&fixture.model));
    let (initial, initial_context) = observe_fixture(fixture, Some(marker));
    assert!(initial.initial_retirement.is_some());
    assert!(initial.pre_materialization_retirement.is_none());
    assert_not_projected(
        "initial cleanup marker is a separate source",
        &initial,
        &initial_context,
        &project,
    );
    // Also retain the marker with a previously genuine witness: the matcher
    // itself rejects it, independently of the producer's None guard.
    let mut changed = receipt.clone();
    changed.initial_retirement = initial.initial_retirement.clone();
    assert_not_projected(
        "initial cleanup cannot reuse a genuine witness",
        &changed,
        &context,
        &project,
    );
}
