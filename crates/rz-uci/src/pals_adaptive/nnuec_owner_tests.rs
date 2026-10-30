//! Actual tiny, explicitly untrained NNUEC CPU ownership integration.
//!
//! P/C outputs use the existing authored NativeFixture. Only NNUEC C
//! preparation and native CPU value/search execute the real FP32 reference
//! path. These checks provide no GPU, trained-model, speed, or strength claim.

use super::*;
use crate::rovent_bootstrap::nnuec_preparing::{InitialNnuecPreparing, InitialNnuecStep};
use rz_contracts::MonotonicTick;
use rz_contracts::rovent::{ExecutionMode, SearchGoal};
use rz_runtime::contracts::ContractClock as _;
use rz_search::contract_time::InstantClock;
use rz_search::cpu_value::CpuTrainingState;
use rz_search::cpu_value::nnuec::v2::review::PreparationActor;
use rz_search::cpu_value::nnuec::v2::{Checkpoint, Model, RelationParameters, SCHEMA, SEMANTICS};
use rz_search::cpu_value::nnuec::{
    NNUEC_SCHEMA, NNUEC_SEMANTICS, NnuecCheckpoint, NnuecConfig, NnuecParameters,
};
use rz_search::cpu_worker::{DeferredNnuecCpuPlan, DeferredNnuecRSource};
use rz_search::pals::engine::adaptive_port::{
    AdaptiveModelNamespace, AdaptivePortModel, AdaptivePortQuestion, AdaptivePreparingRootCustody,
};
use rz_search::pals::engine::preparing_root::{
    ReleasedResearcherReadback, ResearcherPreparingOwner, ResearcherRootStartup,
};
use rz_search::pals::engine::{
    PalsCompletion, PalsConfig, PalsLimits, PostRepairRecheckPolicy, ResearcherSemanticProfile,
    ResolverPolicy,
};
use rz_search::pals::review_context::ReviewIndexLimits;
use rz_search::pals::rovent_frontier::FrontierLimits;
use rz_search::pals::store::{ObservationId, ObservationKind};
use rz_search::pals::value::{MODEL_WDL_VALUE_SEMANTICS, ModelValueIdentity};
use std::num::NonZeroU16;
use std::sync::atomic::AtomicU64;

const OWNER_WINDOW: Duration = Duration::from_secs(20);
const OWNER_CLEANUP: Duration = Duration::from_secs(2);
const OWNER_NODES: u64 = 100_000;

/// Layout-only diagnostic: this does not construct a model, actor, engine or
/// worker. Keep product owner transport sizes visible without changing thread
/// stack limits or entering the neural/native execution fixture.
#[test]
fn nnuec_owner_inline_layout_observation() {
    use crate::rovent_bootstrap::nnuec_preparing::{
        InitialNnuecCleanup, InitialNnuecCleanupOwners, InitialNnuecReady,
        InitialNnuecRetainedFailure,
    };
    use rz_search::pals::engine::PalsEngine;
    use rz_search::pals::engine::adaptive_port::AdaptiveEnginePort;
    use rz_search::pals::engine::preparing_root::{
        PreparingRootFinishRejection, PreparingRootHandoff, ResearcherPreparingRoot,
    };
    use std::mem::size_of;
    for (name, bytes) in [
        (
            "PalsStores",
            size_of::<rz_search::pals::store::PalsStores>(),
        ),
        ("PalsEngine", size_of::<PalsEngine<AdaptivePortModel>>()),
        (
            "ResearcherPreparingOwner",
            size_of::<ResearcherPreparingOwner<AdaptivePortModel>>(),
        ),
        (
            "ResearcherPreparingRoot",
            size_of::<ResearcherPreparingRoot<AdaptivePortModel>>(),
        ),
        (
            "PreparingRootHandoff",
            size_of::<PreparingRootHandoff<AdaptivePortModel>>(),
        ),
        (
            "PreparingRootFinishRejection",
            size_of::<PreparingRootFinishRejection<AdaptivePortModel>>(),
        ),
        (
            "AdaptiveEnginePort",
            size_of::<AdaptiveEnginePort<AdaptivePortModel>>(),
        ),
        (
            "InitialNnuecCleanupOwners",
            size_of::<InitialNnuecCleanupOwners>(),
        ),
        ("InitialNnuecPreparing", size_of::<InitialNnuecPreparing>()),
        ("InitialNnuecCleanup", size_of::<InitialNnuecCleanup>()),
        (
            "InitialNnuecRetainedFailure",
            size_of::<InitialNnuecRetainedFailure>(),
        ),
        ("InitialNnuecReady", size_of::<InitialNnuecReady>()),
        ("InitialNnuecStep", size_of::<InitialNnuecStep>()),
        (
            "AdaptivePortDriver",
            size_of::<AdaptivePortDriver<AdaptivePortModel>>(),
        ),
        (
            "AdaptiveExecutionManager",
            size_of::<
                AdaptiveExecutionManager<
                    AdaptivePortQuestion,
                    NativeFixture,
                    AdaptivePortDriver<AdaptivePortModel>,
                >,
            >(),
        ),
    ] {
        eprintln!("NNUEC_OWNER_INLINE {name} {bytes}");
    }
    // The actual bundle and final Ready owner may be large on the heap; each
    // repeated step/failure/cleanup transport must stay small on the stack.
    for (name, bytes) in [
        ("Preparing", size_of::<InitialNnuecPreparing>()),
        ("Cleanup", size_of::<InitialNnuecCleanup>()),
        ("RetainedFailure", size_of::<InitialNnuecRetainedFailure>()),
        ("Step", size_of::<InitialNnuecStep>()),
    ] {
        assert!(
            bytes <= 1024,
            "{name} embeds {bytes} bytes of lifecycle custody inline"
        );
    }
}

/// Required numerical configuration and authored parameter data only. This
/// uses the same tiny shapes as native-owner regressions, through production
/// Model validation; no alternate evaluator or test-only execution is added.
fn authored_untrained_model() -> Arc<Model> {
    Arc::new(
        Model::new(Checkpoint {
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
                    max_lines: 64,
                    max_plies: 32,
                },
                parameters: NnuecParameters {
                    board_weights: vec![0.0; 800],
                    board_bias: vec![0.0],
                    anchor_weights: vec![0.0; 800],
                    anchor_bias: vec![0.0],
                    from_weights: vec![0.0; 64],
                    to_weights: vec![0.0; 64],
                    promotion_weights: vec![0.0; 5],
                    ply_weights: vec![0.0; 32],
                    token_bias: vec![1.0],
                    causal_weights: vec![1.0, 0.5, 0.25],
                    causal_bias: vec![0.0],
                    context_projection: vec![1.0],
                    context_bias: vec![0.125],
                    local_projection: vec![1.0],
                    local_bias: vec![0.0],
                    hidden_weights: vec![0.0, 0.0, 1.0, 1.0],
                    hidden_bias: vec![0.0],
                    output_weights: vec![1.0; 11],
                    output_bias: vec![0.0; 11],
                },
            },
            parameters: RelationParameters {
                kind_embeddings: vec![0.0, 1.0, 2.0, 3.0],
                pair_parent: vec![1.0; 2],
                pair_child: vec![1.0; 2],
                pair_bias: vec![0.0; 2],
                pair_projection: vec![1.0],
            },
        })
        .unwrap(),
    )
}

#[test]
fn same_preparing_owner_installs_each_question_and_releases_all_view_pins() {
    run_same_preparing_owner(CpuEndpointSelection::deployment());
}

#[test]
fn collection_native_r_uses_original_idle_owner_independently_of_teacher() {
    let mut endpoints = CpuEndpointSelection::deployment();
    endpoints.execution_use = rz_search::external_cpu::CpuExecutionUse::Collection;
    // This is the actual NNUEC Product selection. Arena owns its teacher;
    // there is no synthetic product CPU_T or external-worker Ready here.
    run_same_preparing_owner(endpoints);
}

fn run_same_preparing_owner(endpoints: CpuEndpointSelection) {
    eprintln!("NNUEC_OWNER_STAGE fixture-entered");
    let clock = ContractSystemClock::new(ProcessEpoch(321));
    let until = clock.origin() + OWNER_WINDOW;
    let cleanup_until = until + OWNER_CLEANUP;
    let authority = SearchAuthority {
        epoch: ProcessEpoch(321),
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest([121; 32]),
    };
    let deadline = rz_contracts::Deadline {
        clock: clock.domain(),
        at: MonotonicTick(OWNER_WINDOW.as_nanos() as u64),
    };
    let model = authored_untrained_model();
    assert_eq!(model.identity().training, CpuTrainingState::Untrained);
    let resources = AdaptiveExecutionLimits {
        paused_bytes_per_worker_max: 128 * 1024,
        ..AdaptiveExecutionLimits::default()
    };
    let source = DeferredNnuecRSource::new(
        Arc::clone(&model),
        DeferredNnuecCpuPlan::new(
            CpuConfig {
                profile: CpuProfile::PlanAssisted,
                tt_entries: 16,
                max_depth: 2,
                quiescence_ply: 0,
            },
            CpuOrderingPolicy::LegacyMvvLvaV1,
            resources.paused_bytes_per_worker_max,
            false,
            resources.worker_commands_max,
            None,
        )
        .unwrap(),
    );
    let value = ModelValueIdentity {
        semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
        model: "authored-native-fixture".into(),
        encoding: "authored-fixture-encoding".into(),
        precision: "fp32".into(),
        model_epoch: [5; 32],
    };
    let namespace = AdaptiveModelNamespace::new(
        value.clone(),
        Digest([5; 32]),
        Digest([6; 32]),
        PrecisionProfile::Fp32,
        1,
    )
    .unwrap();
    let roles =
        AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value), namespace).unwrap();
    let owner = ResearcherPreparingOwner::new_selected(
        ResearcherSemanticProfile {
            config: PalsConfig {
                line_plies: 2,
                max_nodes: 1024,
                max_records: 128,
                max_role_calls: 64,
                beam_width: 1,
                cpu_nodes_per_task: 4096,
            },
            maximum_root_prefix_plies: 16,
        },
        roles,
        ResolverPolicy::OwnRawRestricted,
        PostRepairRecheckPolicy::Disabled,
        ReviewSelectionPolicy::default(),
        ReviewIndexLimits::default(),
        source.clone(),
    )
    .unwrap();
    let original_store = owner.stores().read_owner();
    let mut preparing = owner
        .bind_root(
            Position::startpos(),
            authority,
            deadline,
            until,
            Arc::new(AtomicBool::new(false)),
        )
        .map_err(|(_, error)| error)
        .unwrap();
    let next_owner = Arc::new(AtomicU64::new(200));
    preparing
        .attach_original_controls(
            ResearcherRootStartup {
                mode: ExecutionMode::CpuPcr,
                root_budget: Digest([122; 32]),
                goal: SearchGoal::default(),
                nodes: Some(OWNER_NODES),
                selective_policy: Digest([123; 32]),
                frontier_limits: FrontierLimits {
                    maximum_goal_plies: NonZeroU16::new(16).unwrap(),
                    maximum_frames: 128,
                    maximum_paths: 1024,
                    maximum_mate_nodes: 128,
                    maximum_native_receipts: 128,
                },
                clock: InstantClock::new(clock.domain(), clock.origin()),
                limits: PalsLimits {
                    deadline: until,
                    max_rounds: 1,
                    max_cpu_nodes: OWNER_NODES,
                    cpu_depth: 1,
                },
                cleanup_deadline: cleanup_until,
            },
            Arc::new(move || Ok(OwnerId(next_owner.fetch_add(1, Ordering::Relaxed)))),
            None,
        )
        .unwrap();
    preparing.prepare_original_root_rules().unwrap();
    eprintln!("NNUEC_OWNER_STAGE original-root-prepared");
    assert!(original_store.matches(preparing.stores()));
    assert_eq!(
        preparing
            .original_ledger()
            .unwrap()
            .registration_observation()
            .unwrap()
            .active_jobs,
        0
    );

    let facts = Arc::new(NativeFacts::default());
    facts.child_gate.store(true, Ordering::Release);
    let owner_facts = Arc::clone(&facts);
    let owner_clock = clock.clone();
    let native = TicketedNativeService::<ReviewView, AdaptivePortQuestion, NativeFixture>::spawn_with_capacity(
        move || {
            Ok(NativeFixture {
                clock: owner_clock,
                facts: owner_facts,
                finish_ok: true,
            })
        },
        TicketServiceLimits::default(),
        resources.role_tickets_max,
        resources.worker_commands_max,
    )
    .unwrap();
    let actor = PreparationActor::spawn_notified(Arc::clone(&model), native.signal()).unwrap();
    let mut initial = InitialNnuecPreparing::new(preparing, actor, CpuWorkerId(0), native.signal());
    let ready = loop {
        assert!(Instant::now() < until, "initial actual C/TT/Ready timeout");
        match initial.step() {
            InitialNnuecStep::Pending(pending) => {
                initial = pending;
                thread::yield_now();
            }
            InitialNnuecStep::Ready(ready) => break ready,
            InitialNnuecStep::RetainedFailure(failed) => panic!(
                "actual startup failed at {:?}: {:?}",
                failed.phase(),
                failed.error(),
            ),
        }
    };
    let crate::rovent_bootstrap::nnuec_preparing::InitialNnuecReady {
        port,
        custody,
        worker,
        actor,
        metadata,
    } = *ready;
    eprintln!("NNUEC_OWNER_STAGE actual-initial-ready");
    let AdaptivePreparingRootCustody {
        ledger,
        root_budget,
        cleanup_deadline,
        original_view,
        original_review_seal,
        ready: original_ready,
        root_moves,
        source: ready_source,
        cancellation,
        ..
    } = custody;
    assert!(ready_source.same_authority(&source));
    assert!(original_store.matches(port.engine().stores()));
    assert_eq!(worker.profile(), original_ready.profile());
    let initial_stack = original_ready.actual_stack_owner_id();
    let initial_binding = original_ready.binding_digest();
    let initial_view = Arc::downgrade(&original_view);
    let driver = AdaptivePortDriver::new(
        port,
        until,
        Arc::new(native.signal()),
        rz_search::external_cpu::ExternalSlicePolicy::default(),
    )
    .unwrap();
    let mut manager = AdaptiveExecutionManager::new_retaining_with_registered_ledger(
        AdaptiveManagerRegistration {
            authority,
            hard_deadline: until,
            token_budget: 512,
            policy: CoordinatorPolicy::Fixed,
            first_request_sequence: 1,
            native_cpu_node_limit: OWNER_NODES,
            endpoints,
            controller: ControllerConfig::default(),
            device_digest: Digest([124; 32]),
            cpu_device_digest: Digest([125; 32]),
        },
        clock,
        resources,
        AdaptiveRegisteredLedgerSources {
            sources: AdaptiveManagerSources {
                native,
                own_workers: vec![worker],
                external: None,
                driver,
            },
            ledger,
            root_budget,
            cleanup_deadline,
        },
    )
    .unwrap_or_else(|rejected| panic!("actual original-ledger admission: {:?}", rejected.error));
    if endpoints.execution_use == rz_search::external_cpu::CpuExecutionUse::Collection {
        use rz_search::external_cpu::CpuEndpoint;
        let original_budget = manager
            .driver()
            .port()
            .and_then(|port| port.rovent_frontier())
            .map(|frontier| frontier.registration().root_budget);
        assert_eq!(original_budget, Some(root_budget));
        assert_eq!(manager.registration.authority, authority);
        assert_eq!(manager.registration.hard_deadline, until);
        assert_eq!(manager.clock.domain().0, authority.epoch);
        assert!(!manager.root_started);
        assert!(manager.nnuec.is_none() && manager.nnuec_pending.is_none());

        // Exercise only this startup method's endpoint guard on the actual
        // original manager. An Own teacher cannot stand in for a missing or
        // external R. No root input, actor request, installation or work runs.
        for wrong_r in [CpuEndpoint::Disabled, CpuEndpoint::ExternalUci] {
            manager.registration.endpoints.cpu_t = CpuEndpoint::Own;
            manager.registration.endpoints.cpu_r = wrong_r;
            assert!(matches!(
                manager.enable_nnuec_preparation(Arc::clone(&model)),
                Err(AdaptiveManagerError::Invalid(
                    "NNUEC preparation idle root registration"
                ))
            ));
            assert!(manager.nnuec.is_none() && manager.nnuec_pending.is_none());
            assert!(!manager.root_started);
        }

        // This is a method-local declaration test, not an admission of an
        // external worker. The manager constructor still independently
        // requires the real owner whenever product CPU_T is ExternalUci.
        // Recover each unsubmitted model declaration before restoring the
        // validated Product policy and entering the existing execution below.
        for teacher in [CpuEndpoint::Disabled, CpuEndpoint::ExternalUci] {
            manager.registration.endpoints = endpoints;
            manager.registration.endpoints.cpu_t = teacher;
            manager
                .enable_nnuec_preparation(Arc::clone(&model))
                .unwrap();
            assert!(matches!(
                manager.enable_nnuec_preparation(Arc::clone(&model)),
                Err(AdaptiveManagerError::Invalid(
                    "NNUEC preparation idle root registration"
                ))
            ));
            assert!(!manager.root_started && manager.nnuec_pending.is_none());
            let unsubmitted = manager
                .nnuec
                .take()
                .expect("original idle model declaration");
            assert!(Arc::ptr_eq(&unsubmitted.model, &model));
            assert!(!unsubmitted.canceled.load(Ordering::Acquire));
            drop(unsubmitted);
        }
        manager.registration.endpoints = endpoints;
        assert_eq!(manager.registration.authority, authority);
        assert_eq!(manager.registration.hard_deadline, until);
        assert_eq!(
            manager
                .driver()
                .port()
                .and_then(|port| port.rovent_frontier())
                .map(|frontier| frontier.registration().root_budget),
            original_budget,
        );
    }
    manager
        .enable_nnuec_preparation(Arc::clone(&model))
        .unwrap();
    let root = manager.driver_mut().root_input().unwrap();
    manager.prepare_root_ranking(root).unwrap();
    let collection_prelude =
        endpoints.execution_use == rz_search::external_cpu::CpuExecutionUse::Collection;
    let mut prelude_bound_checked = false;
    let receipt = manager
        .run_root_until_observed(cleanup_until, &AtomicBool::new(false), &mut |manager| {
            if collection_prelude {
                // Exercise the actual actor -> prepared_bound -> manager bind
                // path. Private state and the strict route-negative probe stay
                // inside the manager's cfg(test)-only helper.
                manager.pump_nnuec_with_unselected_intervention_for_test(
                    &actor,
                    &mut prelude_bound_checked,
                )?;
            } else {
                manager.pump_nnuec_preparation(&actor)?;
            }
            Ok(())
        })
        .unwrap();
    assert!(
        !collection_prelude || prelude_bound_checked,
        "Collection regression must observe an actual installed R Bound"
    );
    eprintln!("NNUEC_OWNER_STAGE actual-root-completed");
    assert_eq!(manager.phase(), ManagerPhase::Finished);
    assert!(receipt.permits_root_reuse(), "{receipt:?}");
    assert!(
        manager.failures().next().is_none(),
        "{:?}",
        manager.failures().collect::<Vec<_>>()
    );
    let result = manager.driver().result(PalsCompletion::RoundLimit).unwrap();
    assert!(
        result.counters.completed_cpu_tasks >= 2,
        "{:?}",
        result.counters
    );
    assert!(
        result.counters.consumed_cpu_tasks >= 2,
        "{:?}",
        result.counters
    );
    assert!(
        result.counters.refutations > 0,
        "the actual counter path must execute"
    );
    // A new counter CPU admission prunes terminal proposal admission metadata.
    // Reaching zero-step Repair crosses publish_supported_counter's retained
    // proposal/counter Node evidence validation after that pruning, then the
    // real completed-evidence reuse check; result() alone would not prove it.
    assert!(result.counters.repairs > 0);
    assert!(result.counters.consumed_cached_cpu_values > 0);
    assert_eq!(
        result.counters.supported_refutations, 0,
        "different contextual Q values cannot certify a conditional refutation"
    );
    assert_eq!(
        manager.native_cpu_usage().unwrap(),
        (receipt.native_cpu_spent, 0)
    );
    assert!(receipt.native_cpu_spent > 0);
    let installations = manager.nnuec_installation_receipts();
    assert_eq!(installations.len(), 1);
    let last_install = &installations[0].1;
    assert!(last_install.is_installed());
    #[cfg(feature = "pals-adaptive-diagnostics")]
    {
        let observations: Vec<_> = manager.nnuec_preparation_observations().collect();
        let observed = observations
            .iter()
            .find(|observation| {
                observation.original_ack.installation_receipt_sequence
                    == Some(last_install.installation_id().sequence())
            })
            .expect("same actual owner installation is available to opt-in diagnostics");
        assert!(observed.original_ack.actor_terminal_ack_observed);
        assert!(observed.original_ack.driver_attested);
        assert_eq!(observed.original_ack.installation_succeeded, Some(true));
        assert_eq!(
            observed.original_ack.binding,
            Some(last_install.binding_digest())
        );
        assert_eq!(
            observed.original_ack.actual_stack_owner_id,
            Some(last_install.result().unwrap().actual_stack_owner_id())
        );
    }
    assert_eq!(
        last_install.installed_profile().unwrap().owner,
        CpuWorkerId(0)
    );
    assert_ne!(last_install.binding_digest(), initial_binding);
    assert_ne!(
        last_install.result().unwrap().actual_stack_owner_id(),
        initial_stack
    );
    let stores = manager.driver().port().unwrap().engine().stores();
    let cpu_observations = (0..stores.observations.next_observation_sequence())
        .map(|sequence| stores.observations.get(ObservationId(sequence)).unwrap())
        .filter(|observation| observation.kind == ObservationKind::CpuAnalysis)
        .collect::<Vec<_>>();
    assert!(cpu_observations.len() >= 2);
    let conditions = cpu_observations
        .iter()
        .map(|observation| observation.cpu_condition.as_ref().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    assert!(
        conditions.len() >= 2,
        "proposal/counter retain distinct Q namespaces"
    );
    assert!(
        cpu_observations
            .iter()
            .all(|observation| observation.value_identity.as_ref() == Some(model.identity()))
    );
    assert!(
        !facts.accepted.lock().unwrap().is_empty(),
        "P/C are authored, accepted native-service outputs"
    );
    for role in [
        rz_contracts::pals::Role::Proposer,
        rz_contracts::pals::Role::Critic,
    ] {
        assert!(
            facts
                .accepted
                .lock()
                .unwrap()
                .iter()
                .any(|ticket| ticket.role == role)
        );
    }
    manager
        .driver_mut()
        .retire_rovent_frontier()
        .unwrap()
        .unwrap();
    let mut parts = manager
        .into_nnuec_root_parts()
        .unwrap_or_else(|rejected| panic!("actual owner transfer: {:?}", rejected.error));
    assert_eq!(parts.cpu.len(), 1);
    let retired = parts.cpu.pop().unwrap();
    let last = retired.last_installation.unwrap();
    let id = actor
        .try_retire_engine(retired.engine)
        .unwrap_or_else(|(error, _)| panic!("same actor retirement: {error:?}"));
    let release = loop {
        assert!(Instant::now() < cleanup_until);
        if let Some(completion) = actor.try_retired_engine().unwrap() {
            assert_eq!(completion.request_id(), &id);
            break completion
                .into_result()
                .unwrap_or_else(|(_, error)| panic!("actual release: {error:?}"));
        }
        thread::yield_now();
    };
    assert!(release.matches_source(&source));
    assert_eq!(
        release.actual_stack_owner_id(),
        last.result().unwrap().actual_stack_owner_id()
    );
    let invalidation = actor.invalidate_context_cache().unwrap();
    let cache = loop {
        assert!(Instant::now() < cleanup_until);
        if let Some(cache) = actor.try_context_cache_release(&invalidation).unwrap() {
            break cache;
        }
        thread::yield_now();
    };
    assert!(release.matches_cache_release(&cache));
    let port = parts
        .driver
        .into_port(&parts.receipt)
        .unwrap_or_else(|(_, error)| panic!("actual physical root fence: {error:?}"));
    drop(original_ready);
    drop(original_review_seal);
    drop(original_view);
    drop(root_moves);
    drop(metadata);
    drop(cancellation);
    assert!(initial_view.upgrade().is_none());
    assert_eq!(
        port.engine().stores().review_pin_count().unwrap(),
        0,
        "actual engine/cache release and original custody release reclaim every view pin"
    );
    let owner = port
        .into_engine()
        .into_preparing_owner(release, cache, ReleasedResearcherReadback::Installed(&last))
        .unwrap_or_else(|rejected| panic!("same semantic retirement: {:?}", rejected.error));
    assert!(original_store.matches(owner.owner.stores()));
    assert_eq!(owner.owner.stores().review_pin_count().unwrap(), 0);
    actor.shutdown().unwrap();
    parts.native.begin_shutdown(cleanup_until);
    while !parts.native.try_join().unwrap() {
        assert!(Instant::now() < cleanup_until);
        thread::yield_now();
    }
    assert!(facts.finish_called.load(Ordering::Acquire));
    eprintln!("NNUEC_OWNER_STAGE actual-owner-released");
}
