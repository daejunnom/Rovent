//! Source regressions for deferred native-R construction. These do not run
//! training or provide throughput/strength evidence. This file is initially
//! unregistered while the current source snapshot is compiled by the owner.

use super::*;
use crate::cpu::CpuProfile;
use crate::cpu_value::CpuTrainingState;
use crate::cpu_value::nnuec::v2::review::{NnuecPreparationTerminalAck, PreparationActor, Request};
use crate::cpu_value::nnuec::v2::{Checkpoint, RelationParameters, SCHEMA, SEMANTICS};
use crate::cpu_value::nnuec::{
    NNUEC_SCHEMA, NNUEC_SEMANTICS, NnuecCheckpoint, NnuecConfig, NnuecParameters,
};
use crate::cpu_worker::NnuecAuthorityRevocation;
use crate::pals::engine::{RecordKind, RoleQueryPurpose, RoleRecord};
use crate::pals::review_context::{
    ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy, ReviewView,
};
use crate::pals::store::StoreLimits;
use rz_position::{Color, Position};
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

const WINDOW: Duration = Duration::from_secs(20);

#[path = "retired_value_tests.rs"]
mod retirement;

struct Fixture {
    model: Arc<Model>,
    bound: Arc<BoundNnuecReviewContext>,
    stores: PalsStores,
    view: Arc<ReviewView>,
    cancel: Arc<AtomicBool>,
    actor: PreparationActor,
    terminal: NnuecPreparationTerminalAck,
}

fn checkpoint() -> Checkpoint {
    Checkpoint {
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
    }
}

fn fixture() -> Fixture {
    let model = Arc::new(Model::new(checkpoint()).unwrap());
    let mut stores = PalsStores::new(StoreLimits::default());
    let situation = stores
        .focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let state = stores.situations.get(situation).unwrap().state;
    let mut index = ReviewContextIndex::default();
    index
        .publish(
            &stores,
            &RoleRecord {
                revision: 1,
                parent_revision: None,
                supersedes_revision: None,
                origin_state: state,
                kind: RecordKind::Proposal,
                line: vec!["e2e4".parse().unwrap()],
                value: None,
                completed_depth: 0,
                score_scope: None,
                cpu_observation: None,
                perspective: Color::White,
                critical: true,
            },
            None,
        )
        .unwrap();
    let view = Arc::new(
        index
            .select(
                &stores,
                state,
                ReviewQuerySequences {
                    purpose: RoleQueryPurpose::ProposePolicy,
                    prefix: &[],
                    proposal: &[],
                    counterexample: None,
                    legal: &[],
                    divergences: &[],
                },
                ReviewSelectionPolicy::default(),
            )
            .unwrap(),
    );
    let cancel = Arc::new(AtomicBool::new(false));
    let request = Request::admit(
        &model,
        Arc::clone(&view),
        &stores,
        state,
        stores.generation(),
        Instant::now() + WINDOW,
        Arc::clone(&cancel),
    )
    .unwrap();
    let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
    actor
        .try_submit(request)
        .unwrap_or_else(|_| panic!("view preparation admission"));
    let until = Instant::now() + WINDOW;
    let (bound, terminal) = loop {
        if let Some(completion) = actor.try_complete().unwrap() {
            break (
                Arc::new(completion.result.unwrap()),
                completion.terminal_ack,
            );
        }
        assert!(Instant::now() < until, "view preparation did not complete");
        std::thread::yield_now();
    };
    Fixture {
        model,
        bound,
        stores,
        view,
        cancel,
        actor,
        terminal,
    }
}

fn config() -> CpuConfig {
    CpuConfig {
        profile: CpuProfile::PlanAssisted,
        tt_entries: 16,
        max_depth: 2,
        quiescence_ply: 0,
    }
}

fn plan(side_scope: bool) -> DeferredNnuecCpuPlan {
    DeferredNnuecCpuPlan::new(
        config(),
        CpuOrderingPolicy::LegacyMvvLvaV1,
        128 * 1024,
        side_scope,
        3,
        None,
    )
    .unwrap()
}

fn source(fixture: &Fixture, side_scope: bool) -> DeferredNnuecRSource {
    DeferredNnuecRSource::new(Arc::clone(&fixture.model), plan(side_scope))
}

fn request(source: &DeferredNnuecRSource, fixture: &Fixture) -> MaterializationRequest {
    source
        .prepare_materialization(
            Arc::clone(&fixture.bound),
            &fixture.stores,
            fixture.view.seal_digest(),
        )
        .unwrap()
}

fn materialize(request: MaterializationRequest) -> MaterializedNnuecEngine {
    request
        .materialize()
        .unwrap_or_else(|(_, error)| panic!("materialization failed: {error:?}"))
}

fn start(
    source: &DeferredNnuecRSource,
    materialized: MaterializedNnuecEngine,
) -> StartingNnuecWorker {
    source
        .try_spawn_worker(materialized, CpuWorkerId(0), CompletionSignal::default())
        .unwrap_or_else(|(_, error)| panic!("deferred worker did not start: {error:?}"))
}

fn wait_ready(starting: &StartingNnuecWorker) -> CpuWorkerProfile {
    let until = Instant::now() + WINDOW;
    loop {
        let observed = starting.worker.signal.version();
        if let Some(profile) = starting.try_ready().unwrap() {
            return profile;
        }
        assert!(
            Instant::now() < until,
            "actual ready observation did not arrive"
        );
        starting
            .worker
            .signal
            .wait_changed(observed, Duration::from_millis(10));
    }
}

fn wait_actor_materialized(actor: &PreparationActor) -> MaterializationCompletion {
    let until = Instant::now() + WINDOW;
    loop {
        if let Some(completion) = actor.try_materialized().unwrap() {
            return completion;
        }
        assert!(
            Instant::now() < until,
            "materialization actor did not publish its terminal"
        );
        std::thread::yield_now();
    }
}

fn join_for_shutdown(starting: &mut StartingNnuecWorker) -> super::super::CpuWorkerShutdown {
    starting.request_shutdown();
    let until = Instant::now() + WINDOW;
    loop {
        let observed = starting.worker.signal.version();
        if let Some(receipt) = starting.try_join().unwrap() {
            return receipt;
        }
        assert!(Instant::now() < until, "deferred worker did not join");
        starting
            .worker
            .signal
            .wait_changed(observed, Duration::from_millis(10));
    }
}

#[test]
fn plan_rejects_unsupported_bounds_and_keeps_affinity_as_a_declaration() {
    for (paused, commands, affinity) in [
        (0, 3, None),
        (CPU_PAUSED_STACK_MAX_BYTES + 1, 3, None),
        (128 * 1024, 0, None),
        (128 * 1024, CPU_WORKER_COMMAND_CAPACITY + 1, None),
        (128 * 1024, 3, Some(vec![])),
        (
            128 * 1024,
            3,
            Some(vec![LogicalCpu { group: 0, index: 0 }; 2]),
        ),
        (
            128 * 1024,
            3,
            Some(
                (0..1025)
                    .map(|index| LogicalCpu { group: 0, index })
                    .collect(),
            ),
        ),
    ] {
        assert_eq!(
            DeferredNnuecCpuPlan::new(
                config(),
                CpuOrderingPolicy::LegacyMvvLvaV1,
                paused,
                false,
                commands,
                affinity,
            )
            .err(),
            Some(CpuWorkerError::Invalid("deferred NNUEC CPU declaration"))
        );
    }
    for (depth, qplies) in [(0, 0), (65, 0), (2, 33)] {
        let mut invalid = config();
        invalid.max_depth = depth;
        invalid.quiescence_ply = qplies;
        assert_eq!(
            DeferredNnuecCpuPlan::new(
                invalid,
                CpuOrderingPolicy::LegacyMvvLvaV1,
                128 * 1024,
                false,
                3,
                None,
            )
            .err(),
            Some(CpuWorkerError::Invalid("deferred NNUEC CPU declaration"))
        );
    }
    let cpus = vec![LogicalCpu { group: 0, index: 0 }];
    let declared = DeferredNnuecCpuPlan::new(
        config(),
        CpuOrderingPolicy::LegalSeeV1,
        64 * 1024,
        true,
        2,
        Some(cpus.clone()),
    )
    .unwrap();
    assert_eq!(declared.ordering_policy(), CpuOrderingPolicy::LegalSeeV1);
    assert_eq!(declared.paused_bytes_limit(), 64 * 1024);
    assert!(declared.preserves_paused_side_scope());
    assert_eq!(declared.command_capacity(), 2);
    assert_eq!(declared.requested_affinity(), Some(cpus.as_slice()));
}

#[test]
fn materialization_preserves_required_plan_and_has_no_searched_work() {
    let fixture = fixture();
    let source = source(&fixture, true);
    assert!(Arc::ptr_eq(source.model(), &fixture.model));
    assert_eq!(source.cpu_plan().config().tt_entries, 16);
    let original = request(&source, &fixture);
    assert!(original.model_matches(&fixture.model));
    let foreign_equal_model = Arc::new(Model::new(checkpoint()).unwrap());
    assert_eq!(foreign_equal_model.identity(), fixture.model.identity());
    assert!(!original.model_matches(&foreign_equal_model));
    let binding = original.binding_digest();
    let provenance = original.provenance().clone();
    let materialized = materialize(original);
    assert_eq!(materialized.binding_digest(), binding);
    assert_eq!(materialized.provenance(), &provenance);
    assert_eq!(
        materialized.engine.value_identity(),
        fixture.model.identity()
    );
    assert_eq!(
        materialized.engine.job_binding_digest().unwrap(),
        Some(binding)
    );
    assert_eq!(materialized.engine.config().tt_entries, 16);
    assert_eq!(materialized.engine.config().max_depth, 2);
    assert_eq!(materialized.engine.config().quiescence_ply, 0);
    assert_eq!(
        materialized.engine.resume_policy(),
        CpuResumePolicy::PausedStack
    );
    assert_eq!(
        materialized.engine.ordering_policy(),
        source.cpu_plan().ordering_policy()
    );
    assert!(materialized.engine.preserves_paused_on_fresh_tasks());
    assert_eq!(materialized.engine.paused_stack_max_bytes(), 128 * 1024);
    assert_eq!(materialized.engine.last_attempt_work(), None);
    let paused = materialized.engine.paused_stack_snapshot().unwrap();
    assert!(paused.complete && paused.tokens_retained == 0 && paused.bytes_current == 0);
}

#[test]
fn deferred_materialization_preserves_explicit_root_tie_policy_and_rejects_mismatch() {
    assert_eq!(
        plan(false).root_tie_policy(),
        CpuRootTiePolicy::LegacyFirstSearched
    );
    for side_scope in [false, true] {
        let fixture = fixture();
        let explicit = DeferredNnuecCpuPlan::new_with_root_tie_policy(
            config(),
            CpuOrderingPolicy::LegacyMvvLvaV1,
            128 * 1024,
            side_scope,
            3,
            None,
            CpuRootTiePolicy::RulesOrderConfirmed,
        )
        .unwrap();
        let source = DeferredNnuecRSource::new(Arc::clone(&fixture.model), explicit);
        let original = request(&source, &fixture);
        let materialized = materialize(original);
        assert_eq!(
            materialized.engine.root_tie_policy(),
            CpuRootTiePolicy::RulesOrderConfirmed
        );
        assert!(
            materialized
                .engine
                .search_identity()
                .contains("root-tie-confirmed")
        );
        assert_eq!(
            materialized.engine.preserves_paused_on_fresh_tasks(),
            side_scope
        );
        assert_eq!(
            validate_materialized_readback(
                &materialized.engine,
                &fixture.model,
                &materialized.bound,
                materialized.binding,
                CpuRootTiePolicy::LegacyFirstSearched,
                "policy mismatch",
            ),
            Err(CpuWorkerError::Invalid("policy mismatch")),
        );
        assert_eq!(materialized.engine.last_attempt_work(), None);
        let expected_conditions =
            crate::cpu_worker::cpu_search_conditions_digest(&materialized.engine);
        let starting = start(&source, materialized);
        assert_eq!(wait_ready(&starting).search_conditions, expected_conditions);
        let (worker, registration) = starting
            .accept_ready()
            .unwrap_or_else(|(_, error)| panic!("ready refused: {error:?}"));
        assert_eq!(
            registration.profile().search_conditions,
            expected_conditions
        );
        assert!(worker.shutdown().unwrap().thread_joined);
    }
}

#[test]
fn deferred_policy_mismatch_returns_the_same_materialized_owner_before_spawn() {
    let fixture = fixture();
    let plan = DeferredNnuecCpuPlan::new_with_root_tie_policy(
        config(),
        CpuOrderingPolicy::LegacyMvvLvaV1,
        128 * 1024,
        false,
        3,
        None,
        CpuRootTiePolicy::RulesOrderConfirmed,
    )
    .unwrap();
    let source = DeferredNnuecRSource::new(Arc::clone(&fixture.model), plan);
    let original = request(&source, &fixture);
    let evaluator: Arc<dyn CpuValueEvaluator> = original.value.clone();
    let mut materialized = materialize(original);
    // Simulate a mismatched engine before the original source is allowed to
    // start a worker, while retaining the original model and Bound binding.
    materialized.engine = CpuEngine::with_evaluator_ordering_and_resume_policy(
        config(),
        evaluator,
        CpuOrderingPolicy::LegacyMvvLvaV1,
        CpuResumePolicy::PausedStack,
    )
    .unwrap();
    let owner = materialized
        .engine
        .paused_stack_snapshot()
        .unwrap()
        .owner_id;
    let binding = materialized.binding_digest();
    let returned =
        match source.try_spawn_worker(materialized, CpuWorkerId(0), CompletionSignal::default()) {
            Ok(_) => panic!("mismatched policy must not start a worker"),
            Err((returned, reason)) => {
                assert_eq!(
                    reason,
                    CpuWorkerError::Invalid("deferred NNUEC source/binding revoked")
                );
                returned
            }
        };
    assert_eq!(
        returned.engine.paused_stack_snapshot().unwrap().owner_id,
        owner
    );
    assert_eq!(
        returned.engine.root_tie_policy(),
        CpuRootTiePolicy::LegacyFirstSearched
    );
    assert_eq!(returned.binding_digest(), binding);
    assert_eq!(returned.engine.job_binding_digest().unwrap(), Some(binding));
    assert_eq!(returned.engine.last_attempt_work(), None);
}

#[test]
fn an_equal_model_and_plan_cannot_substitute_for_the_original_source_owner() {
    let fixture = fixture();
    let original = source(&fixture, false);
    let foreign = source(&fixture, false);
    let materialized = materialize(request(&original, &fixture));
    let binding = materialized.binding_digest();
    let provenance = materialized.provenance().clone();
    let returned =
        match foreign.try_spawn_worker(materialized, CpuWorkerId(0), CompletionSignal::default()) {
            Ok(_) => panic!("equal public model/plan replaced the actual source owner"),
            Err((returned, reason)) => {
                assert_eq!(
                    reason,
                    CpuWorkerError::Invalid("deferred NNUEC source/binding revoked")
                );
                returned
            }
        };
    assert_eq!(returned.binding_digest(), binding);
    assert_eq!(returned.provenance(), &provenance);
    let original_clone = original.clone();
    let starting = start(&original_clone, returned);
    assert_eq!(wait_ready(&starting).evaluation_namespace, binding);
    let (worker, registration) = starting
        .accept_ready()
        .unwrap_or_else(|(_, error)| panic!("ready refused: {error:?}"));
    assert_eq!(registration.binding_digest(), binding);
    assert_eq!(registration.profile(), worker.profile());
    assert!(registration.validates(
        &fixture.stores,
        fixture.bound.provenance().root,
        fixture.stores.generation(),
        fixture.view.seal_digest(),
    ));
    assert!(worker.shutdown().unwrap().thread_joined);
}

#[test]
fn revocation_before_materialization_returns_the_same_unmaterialized_request() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let original = request(&source, &fixture);
    let binding = original.binding_digest();
    let provenance = original.provenance().clone();
    fixture.cancel.store(true, Ordering::Release);
    let returned = match original.materialize() {
        Ok(_) => panic!("revoked bound allocated a usable native owner"),
        Err((returned, reason)) => {
            assert_eq!(
                reason,
                CpuWorkerError::NnuecAuthorityRevoked(NnuecAuthorityRevocation::Cancelled)
            );
            returned
        }
    };
    assert_eq!(returned.binding_digest(), binding);
    assert_eq!(returned.provenance(), &provenance);
    assert!(Arc::ptr_eq(&returned.bound, &fixture.bound));
    assert!(matches!(
        source.prepare_materialization(
            Arc::clone(&fixture.bound),
            &fixture.stores,
            fixture.view.seal_digest(),
        ),
        Err(NnuecError::StateMismatch)
    ));
}

#[test]
fn revocation_after_materialization_returns_the_unsearched_engine_without_a_thread() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let materialized = materialize(request(&source, &fixture));
    let binding = materialized.binding_digest();
    fixture.cancel.store(true, Ordering::Release);
    match source.try_spawn_worker(materialized, CpuWorkerId(0), CompletionSignal::default()) {
        Ok(_) => panic!("revoked materialized value started a usable worker"),
        Err((returned, reason)) => {
            assert_eq!(
                reason,
                CpuWorkerError::NnuecAuthorityRevoked(NnuecAuthorityRevocation::Cancelled)
            );
            assert_eq!(returned.binding_digest(), binding);
            assert_eq!(returned.engine.last_attempt_work(), None);
            assert_eq!(returned.engine.value_identity(), fixture.model.identity());
            assert!(Arc::ptr_eq(&returned.bound, &fixture.bound));
        }
    }
}

#[test]
fn missing_ready_observation_cannot_publish_an_accepted_worker() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let original = request(&source, &fixture);
    let binding = original.binding_digest();
    let materialized = materialize(original);
    let stack_owner_id = materialized
        .engine_for_registration()
        .paused_stack_snapshot()
        .unwrap()
        .owner_id;
    let mut starting = start(&source, materialized);
    // Hold back only the manager's observation. The original shared slot is
    // restored before acceptance, so a fabricated Ready never authorizes work.
    let actual_readback = std::mem::replace(
        &mut starting.worker.ready_readback,
        Arc::new(OnceLock::new()),
    );
    assert_eq!(starting.try_ready(), Ok(None));
    let mut starting = match starting.accept_ready() {
        Ok(_) => panic!("a missing actual readback was accepted as ready"),
        Err((retained, reason)) => {
            assert_eq!(reason, CpuWorkerError::AlreadyReserved);
            retained
        }
    };
    starting.worker.ready_readback = actual_readback;
    let observed = wait_ready(&starting);
    assert_eq!(observed.evaluation_namespace, binding);
    assert_eq!(observed.max_depth, 2);
    let (worker, registration) = starting
        .accept_ready()
        .unwrap_or_else(|(_, error)| panic!("actual ready refused: {error:?}"));
    assert_eq!(worker.profile(), observed);
    assert_eq!(worker.command_capacity(), 3);
    assert_eq!(worker.paused_bytes_limit(), 128 * 1024);
    assert_eq!(registration.actual_stack_owner_id(), stack_owner_id);
    assert_eq!(registration.value_identity(), fixture.model.identity());
    assert_eq!(registration.cpu_plan().command_capacity(), 3);
    assert_eq!(registration.provenance(), fixture.bound.provenance());
    assert!(registration.validates(
        &fixture.stores,
        fixture.bound.provenance().root,
        fixture.stores.generation(),
        fixture.view.seal_digest(),
    ));
    let mut wrong_seal = fixture.view.seal_digest();
    wrong_seal[0] ^= 1;
    assert!(!registration.validates(
        &fixture.stores,
        fixture.bound.provenance().root,
        fixture.stores.generation(),
        wrong_seal,
    ));
    assert!(!registration.validates(
        &fixture.stores,
        fixture.bound.provenance().root,
        fixture.stores.generation() + 1,
        fixture.view.seal_digest(),
    ));
    let foreign = PalsStores::new(StoreLimits::default());
    assert!(!registration.validates(
        &foreign,
        fixture.bound.provenance().root,
        fixture.stores.generation(),
        fixture.view.seal_digest(),
    ));
    fixture.cancel.store(true, Ordering::Release);
    assert!(!registration.validates(
        &fixture.stores,
        fixture.bound.provenance().root,
        fixture.stores.generation(),
        fixture.view.seal_digest(),
    ));
    assert!(worker.shutdown().unwrap().engine_recoverable);
}

#[test]
fn cancelled_starting_owner_retains_original_bound_and_can_only_be_joined_for_cleanup() {
    let Fixture {
        model,
        bound,
        stores,
        view,
        cancel,
        actor,
        terminal,
    } = fixture();
    assert!(terminal.matches_original(&model, &view));
    assert!(terminal.allows_c_only_cleanup());
    let source = DeferredNnuecRSource::new(Arc::clone(&model), plan(false));
    let weak_bound = Arc::downgrade(&bound);
    let weak_view = Arc::downgrade(&view);
    let request = source
        .prepare_materialization(Arc::clone(&bound), &stores, view.seal_digest())
        .unwrap();
    drop(bound);
    drop(view);
    let starting = start(&source, materialize(request));
    assert!(!terminal.allows_c_only_cleanup());
    cancel.store(true, Ordering::Release);
    assert_eq!(
        starting.try_ready(),
        Err(CpuWorkerError::NnuecAuthorityRevoked(
            NnuecAuthorityRevocation::Cancelled
        ))
    );
    actor.shutdown().unwrap();
    assert!(weak_bound.upgrade().is_some());
    assert!(weak_view.upgrade().is_some());
    assert_eq!(
        starting.try_ready(),
        Err(CpuWorkerError::NnuecAuthorityRevoked(
            NnuecAuthorityRevocation::ActorStopped
        ))
    );
    let mut retained = match starting.accept_ready() {
        Ok(_) => panic!("revoked starting owner was published as accepted"),
        Err((retained, reason)) => {
            assert_eq!(
                reason,
                CpuWorkerError::NnuecAuthorityRevoked(NnuecAuthorityRevocation::ActorStopped)
            );
            retained
        }
    };
    let receipt = join_for_shutdown(&mut retained);
    assert!(receipt.thread_joined && receipt.engine_recoverable);
    assert!(receipt.events.is_empty());
    assert!(weak_bound.upgrade().is_some());
    let recovered = retained.take_joined_engine().unwrap();
    assert_eq!(recovered.value_identity(), model.identity());
    drop(recovered);
    drop(retained);
    assert!(weak_bound.upgrade().is_none());
    assert!(weak_view.upgrade().is_none());
    drop(terminal);
}

#[test]
fn shutting_down_the_original_preparation_actor_revokes_its_pending_materialization() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let request = request(&source, &fixture);
    let provenance = request.provenance().clone();
    fixture.actor.shutdown().unwrap();
    match request.materialize() {
        Ok(_) => panic!("an actor-revoked request was materialized on another authority"),
        Err((returned, reason)) => {
            assert_eq!(
                reason,
                CpuWorkerError::NnuecAuthorityRevoked(NnuecAuthorityRevocation::ActorStopped)
            );
            assert_eq!(returned.provenance(), &provenance);
            assert!(Arc::ptr_eq(&returned.bound, &fixture.bound));
        }
    }
}

#[test]
fn materialization_handoff_claims_one_unique_engine_and_releases_its_gate() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let materialized = materialize(request(&source, &fixture));
    let binding = materialized.binding_digest();
    let gate = Arc::new(std::sync::atomic::AtomicUsize::new(2));
    let (owner, handoff) = MaterializationOwner::handoff(materialized, Arc::clone(&gate));
    let claimed = handoff.claim().unwrap();
    assert_eq!(claimed.binding_digest(), binding);
    assert_eq!(claimed.engine.last_attempt_work(), None);
    assert_eq!(gate.load(Ordering::Acquire), 0);
    drop(owner);
    assert_eq!(gate.load(Ordering::Acquire), 0);
    let starting = start(&source, claimed);
    let _ = wait_ready(&starting);
    assert!(starting.shutdown().unwrap().thread_joined);
}

#[test]
fn unclaimed_handoff_stays_in_preparation_owner_custody_until_release() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let materialized = materialize(request(&source, &fixture));
    let gate = Arc::new(std::sync::atomic::AtomicUsize::new(2));
    let (owner, handoff) = MaterializationOwner::handoff(materialized, Arc::clone(&gate));
    assert_eq!(gate.load(Ordering::Acquire), 2);
    drop(owner);
    assert_eq!(gate.load(Ordering::Acquire), 0);
    assert!(matches!(handoff.claim(), Err(CpuWorkerError::UnknownJob)));
}

#[test]
fn another_preparation_actor_cannot_materialize_the_original_bound_even_with_the_same_model() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let original = request(&source, &fixture);
    let provenance = original.provenance().clone();
    let foreign = PreparationActor::spawn(Arc::clone(&fixture.model)).unwrap();
    foreign
        .try_materialize(original)
        .unwrap_or_else(|(reason, _)| panic!("foreign test admission failed: {reason:?}"));
    let completion = wait_actor_materialized(&foreign);
    assert_eq!(completion.provenance, provenance);
    let returned = match completion.result {
        Ok(_) => panic!("equal model bytes authorized a foreign preparation actor"),
        Err((returned, reason)) => {
            assert_eq!(
                reason,
                CpuWorkerError::Invalid("foreign or revoked NNUEC preparation actor")
            );
            assert_eq!(returned.provenance(), &provenance);
            assert!(Arc::ptr_eq(&returned.bound, &fixture.bound));
            returned
        }
    };
    foreign.shutdown().unwrap();
    assert!(fixture.bound.authority_is_live());
    fixture
        .actor
        .try_materialize(returned)
        .unwrap_or_else(|(reason, _)| panic!("original actor refused its request: {reason:?}"));
    let completion = wait_actor_materialized(&fixture.actor);
    assert_eq!(completion.provenance, provenance);
    let handoff = completion
        .result
        .unwrap_or_else(|(_, error)| panic!("original actor failed: {error:?}"));
    let materialized = handoff.claim().unwrap();
    let starting = start(&source, materialized);
    let _ = wait_ready(&starting);
    let (worker, registration) = starting
        .accept_ready()
        .unwrap_or_else(|(_, error)| panic!("original actor ready refused: {error:?}"));
    assert_eq!(registration.provenance(), &provenance);
    assert!(Arc::ptr_eq(&registration.bound, &fixture.bound));
    worker.shutdown().unwrap();
}

#[test]
fn matching_actor_reports_actual_cancel_without_materializing_or_losing_the_original_request() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let original = request(&source, &fixture);
    let identity = fixture.bound.request_identity();
    let binding = original.binding_digest();
    fixture.cancel.store(true, Ordering::Release);
    fixture
        .actor
        .try_materialize(original)
        .unwrap_or_else(|_| panic!("cancelled materialization admission"));
    let completion = wait_actor_materialized(&fixture.actor);
    let (returned, error) = completion
        .result
        .err()
        .expect("actual cancellation terminal");
    assert_eq!(
        error,
        CpuWorkerError::NnuecAuthorityRevoked(NnuecAuthorityRevocation::Cancelled)
    );
    assert_eq!(returned.binding_digest(), binding);
    assert!(returned.bound.matches_request(&identity));
    assert!(fixture.terminal.matches_request(&identity));
    assert!(fixture.terminal.allows_c_only_cleanup());
    assert!(Arc::ptr_eq(&returned.bound, &fixture.bound));
}

#[test]
fn foreign_actor_rejection_is_not_reclassified_by_cancelled_original_control() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let original = request(&source, &fixture);
    let foreign = PreparationActor::spawn(Arc::clone(&fixture.model)).unwrap();
    fixture.cancel.store(true, Ordering::Release);
    foreign
        .try_materialize(original)
        .unwrap_or_else(|_| panic!("foreign materialization admission"));
    let completion = wait_actor_materialized(&foreign);
    let (returned, error) = completion.result.err().expect("foreign request rejection");
    assert_eq!(
        error,
        CpuWorkerError::Invalid("foreign or revoked NNUEC preparation actor")
    );
    assert!(Arc::ptr_eq(&returned.bound, &fixture.bound));
    assert!(fixture.terminal.allows_c_only_cleanup());
    foreign.shutdown().unwrap();
}

#[test]
fn cancelled_original_control_cannot_hide_a_foreign_engine_readback() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let mut materialized = materialize(request(&source, &fixture));
    // Deliberately substitute a real foreign evaluator owner as an invalid
    // input. It is never accepted as NNUEC or used as a fallback searcher.
    materialized.engine = CpuEngine::new(config()).unwrap();
    fixture.cancel.store(true, Ordering::Release);
    let (returned, error) =
        match source.try_spawn_worker(materialized, CpuWorkerId(0), CompletionSignal::default()) {
            Err(rejected) => rejected,
            Ok(_) => panic!("foreign engine must remain rejected"),
        };
    assert_eq!(
        error,
        CpuWorkerError::Invalid("deferred NNUEC source/binding revoked")
    );
    assert_eq!(returned.engine.last_attempt_work(), None);
}

#[test]
fn native_constructor_configuration_failure_remains_a_typed_failure() {
    let fixture = fixture();
    let mut source = source(&fixture, false);
    Arc::get_mut(&mut source.declaration)
        .unwrap()
        .plan
        .config
        .max_depth = 0;
    let original = request(&source, &fixture);
    let (returned, error) = match original.materialize() {
        Err(rejected) => rejected,
        Ok(_) => panic!("invalid native declaration"),
    };
    assert_eq!(
        error,
        CpuWorkerError::Cpu(CpuError::InvalidConfig("depth must be in 1..=64"))
    );
    assert!(fixture.bound.authority_is_live());
    assert!(Arc::ptr_eq(&returned.bound, &fixture.bound));
    assert!(fixture.terminal.allows_c_only_cleanup());
}

#[test]
fn actual_actor_handoff_keeps_its_shared_seat_until_claim_and_preserves_the_original_bound() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let original = request(&source, &fixture);
    let provenance = original.provenance().clone();
    let binding = original.binding_digest();
    fixture
        .actor
        .try_materialize(original)
        .unwrap_or_else(|(reason, _)| panic!("materialization admission failed: {reason:?}"));
    let completion = wait_actor_materialized(&fixture.actor);
    assert_eq!(completion.provenance, provenance);
    let handoff = completion
        .result
        .unwrap_or_else(|(_, error)| panic!("materialization failed: {error:?}"));
    match fixture.actor.try_materialize(request(&source, &fixture)) {
        Ok(()) => panic!("polling an unclaimed handoff released its bounded seat"),
        Err((reason, returned)) => {
            assert_eq!(
                reason,
                crate::cpu_value::nnuec::v2::review::PreparationError::Busy
            );
            assert_eq!(returned.provenance(), &provenance);
        }
    }
    let make_neural = || {
        Request::admit(
            &fixture.model,
            Arc::clone(&fixture.view),
            &fixture.stores,
            fixture.bound.provenance().root,
            fixture.stores.generation(),
            Instant::now() + WINDOW,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap()
    };
    assert!(matches!(
        fixture.actor.try_submit(make_neural()),
        Err((
            crate::cpu_value::nnuec::v2::review::PreparationError::Busy,
            _
        ))
    ));
    let materialized = handoff.claim().unwrap();
    assert_eq!(materialized.binding_digest(), binding);
    assert_eq!(materialized.engine.last_attempt_work(), None);
    // A later preparation may reuse the actor cache, but cannot relabel the
    // original materialized engine's Bound as its newest completion.
    fixture
        .actor
        .try_submit(make_neural())
        .unwrap_or_else(|(reason, _)| panic!("claim did not release the seat: {reason:?}"));
    let until = Instant::now() + WINDOW;
    let later = loop {
        if let Some(completion) = fixture.actor.try_complete().unwrap() {
            break completion.result.unwrap();
        }
        assert!(
            Instant::now() < until,
            "later neural preparation did not complete"
        );
        std::thread::yield_now();
    };
    assert_eq!(later.provenance(), &provenance);
    assert!(later.authority_is_live());
    let starting = start(&source, materialized);
    let observed = wait_ready(&starting);
    assert_eq!(observed.evaluation_namespace, binding);
    let (worker, registration) = starting
        .accept_ready()
        .unwrap_or_else(|(_, error)| panic!("actual actor ready refused: {error:?}"));
    assert_eq!(registration.provenance(), &provenance);
    assert!(Arc::ptr_eq(&registration.bound, &fixture.bound));
    fixture.cancel.store(true, Ordering::Release);
    assert!(!registration.bound.authority_is_live());
    assert!(later.authority_is_live());
    worker.shutdown().unwrap();
}

#[test]
fn actual_actor_join_reclaims_an_unclaimed_materialized_engine() {
    let fixture = fixture();
    let source = source(&fixture, false);
    fixture
        .actor
        .try_materialize(request(&source, &fixture))
        .unwrap_or_else(|(reason, _)| panic!("materialization admission failed: {reason:?}"));
    let completion = wait_actor_materialized(&fixture.actor);
    let handoff = completion
        .result
        .unwrap_or_else(|(_, error)| panic!("materialization failed: {error:?}"));
    assert!(matches!(
        fixture.actor.try_materialize(request(&source, &fixture)),
        Err((
            crate::cpu_value::nnuec::v2::review::PreparationError::Busy,
            _
        ))
    ));
    fixture.actor.shutdown().unwrap();
    assert!(!fixture.bound.authority_is_live());
    assert!(matches!(handoff.claim(), Err(CpuWorkerError::UnknownJob)));
}
