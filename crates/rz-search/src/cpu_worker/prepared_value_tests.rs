//! Source regressions for the prepared-value channel and real CPU owner loop.
//! These checks do not constitute throughput, strength, or training evidence.

use super::super::{
    CpuJob, CpuSliceReport, CpuWorker, CpuWorkerEvent, CpuWorkerShutdown,
    cpu_root_restriction_digest,
};
use super::*;
use crate::contract_time::InstantClock;
use crate::cpu::{
    CpuConfig, CpuOrderingPolicy, CpuProfile, CpuResumePolicy, CpuRootTiePolicy, CpuWork,
};
use crate::cpu_value::nnuec::v2::review::{PreparationActor, Request as ReviewRequest};
use crate::cpu_value::nnuec::v2::{Checkpoint, RelationParameters, SCHEMA, SEMANTICS};
use crate::cpu_value::nnuec::{
    NNUEC_SCHEMA, NNUEC_SEMANTICS, NnuecCheckpoint, NnuecConfig, NnuecParameters,
};
use crate::cpu_value::{BootstrapCpuValue, CpuTrainingState};
use crate::pals::engine::{RecordKind, RoleQueryPurpose, RoleRecord};
use crate::pals::review_context::{
    ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy, ReviewView,
};
use crate::pals::store::StoreLimits;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::{CpuJobEnvelope, CpuJobId, CpuWorkerId, SliceGrant};
use rz_contracts::{
    ClockDomain, ExecutionId, GameGeneration, OwnerId, ProcessEpoch, RootGeneration,
};
use rz_position::contracts::ContractPosition;
use rz_position::{Color, Position};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const WINDOW: Duration = Duration::from_secs(20);

struct Fixture {
    model: Arc<Model>,
    bound: Arc<BoundNnuecReviewContext>,
    stores: PalsStores,
    view: Arc<ReviewView>,
    cancel: Arc<AtomicBool>,
    _actor: PreparationActor,
}

fn fixture(first_move: &str) -> Fixture {
    let config = NnuecConfig {
        board_width: 1,
        line_width: 1,
        context_width: 1,
        local_width: 1,
        hidden_width: 1,
        causal_window: 3,
        max_lines: 8,
        max_plies: 16,
    };
    let model = Arc::new(
        Model::new(Checkpoint {
            schema: SCHEMA.into(),
            semantics: SEMANTICS.into(),
            base: NnuecCheckpoint {
                schema: NNUEC_SCHEMA.into(),
                semantics: NNUEC_SEMANTICS.into(),
                training: CpuTrainingState::Untrained,
                output_scale: 1.0,
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
                config,
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
    );
    let mut stores = PalsStores::new(StoreLimits::default());
    let situation = stores
        .focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let root = stores.situations.get(situation).unwrap().state;
    let mut index = ReviewContextIndex::default();
    index
        .publish(
            &stores,
            &RoleRecord {
                revision: 1,
                parent_revision: None,
                supersedes_revision: None,
                origin_state: root,
                kind: RecordKind::Proposal,
                line: vec![first_move.parse().unwrap()],
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
                root,
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
    let request = ReviewRequest::admit(
        &model,
        Arc::clone(&view),
        &stores,
        root,
        stores.generation(),
        Instant::now() + WINDOW,
        Arc::clone(&cancel),
    )
    .unwrap();
    let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
    actor
        .try_submit(request)
        .unwrap_or_else(|_| panic!("preparation admission"));
    let until = Instant::now() + WINDOW;
    let bound = loop {
        if let Some(completion) = actor.try_complete().unwrap() {
            break Arc::new(completion.result.unwrap());
        }
        assert!(
            Instant::now() < until,
            "preparation did not publish completion"
        );
        std::thread::yield_now();
    };
    Fixture {
        model,
        bound,
        stores,
        view,
        cancel,
        _actor: actor,
    }
}

fn payload(fixture: &Fixture) -> PreparedValueInstallation {
    PreparedValueInstallation::from_bound(
        Arc::clone(&fixture.model),
        Arc::clone(&fixture.bound),
        &fixture.stores,
        fixture.view.seal_digest(),
    )
    .unwrap()
}

fn engine(side_scope: bool) -> CpuEngine {
    engine_with_tie(side_scope, CpuRootTiePolicy::LegacyFirstSearched)
}

fn engine_with_tie(side_scope: bool, tie: CpuRootTiePolicy) -> CpuEngine {
    let config = CpuConfig {
        profile: CpuProfile::PlanAssisted,
        tt_entries: 16,
        max_depth: 2,
        quiescence_ply: 0,
    };
    if side_scope {
        CpuEngine::with_paused_side_scope_and_tie_policy(
            config,
            Arc::new(BootstrapCpuValue::default()),
            CpuOrderingPolicy::LegacyMvvLvaV1,
            tie,
        )
        .unwrap()
    } else {
        CpuEngine::with_evaluator_ordering_resume_and_tie_policy(
            config,
            Arc::new(BootstrapCpuValue::default()),
            CpuOrderingPolicy::LegacyMvvLvaV1,
            CpuResumePolicy::PausedStack,
            tie,
        )
        .unwrap()
    }
}

#[test]
fn actual_prepared_replacement_keeps_selected_root_tie_policy_and_owner_profile() {
    for side_scope in [false, true] {
        let fixture = fixture("e2e4");
        let cpu = engine_with_tie(side_scope, CpuRootTiePolicy::RulesOrderConfirmed);
        let identity = cpu.search_identity();
        let conditions = cpu.search_conditions();
        let mut worker = CpuWorker::spawn(CpuWorkerId(0), cpu).unwrap();
        let before = worker.profile();
        let value = payload(&fixture);
        let binding = value.binding_digest();
        worker.try_install_prepared_value(value).unwrap();
        let receipt = wait_install(&mut worker);
        let readback = receipt.result().unwrap();
        assert_eq!(
            readback.profile().search_conditions,
            before.search_conditions
        );
        assert_eq!(readback.profile().evaluation_namespace, binding);
        assert_eq!(readback.side_scope_preserved(), side_scope);
        worker.request_shutdown();
        assert!(wait_join(&mut worker).engine_recoverable);
        let cpu = worker.take_engine().unwrap();
        assert_eq!(cpu.root_tie_policy(), CpuRootTiePolicy::RulesOrderConfirmed);
        assert_eq!(cpu.search_identity(), identity);
        assert_eq!(cpu.search_conditions(), conditions);
        assert_eq!(cpu.job_binding_digest().unwrap(), Some(binding));
    }
}

#[test]
fn rejected_prepared_replacement_keeps_selected_tie_owner_and_binding() {
    for side_scope in [false, true] {
        let fixture = fixture("e2e4");
        let (mut endpoint, mut owner) = channel();
        let mut cpu = engine_with_tie(side_scope, CpuRootTiePolicy::RulesOrderConfirmed);
        let previous = profile(&cpu);
        let old_identity = cpu.search_identity();
        let old_stack_owner = cpu.paused_stack_snapshot().unwrap().owner_id;
        let mut binding = None;
        endpoint.try_submit(payload(&fixture), previous).unwrap();
        fixture.cancel.store(true, Ordering::Release);
        owner
            .service(
                &mut cpu,
                &mut binding,
                true,
                true,
                empty_occupancy(),
                false,
                &CompletionSignal::default(),
            )
            .unwrap();
        let receipt = endpoint.try_ack().unwrap().unwrap();
        assert_eq!(
            receipt.result().err(),
            Some(&CpuWorkerError::Invalid("prepared value authority revoked"))
        );
        assert_eq!(profile(&cpu), previous);
        assert_eq!(cpu.root_tie_policy(), CpuRootTiePolicy::RulesOrderConfirmed);
        assert_eq!(cpu.search_identity(), old_identity);
        assert_eq!(
            cpu.paused_stack_snapshot().unwrap().owner_id,
            old_stack_owner
        );
        assert_eq!(binding, None);
    }
}

#[test]
fn mismatched_search_conditions_do_not_replace_existing_bound_owner() {
    for side_scope in [false, true] {
        let first = fixture("e2e4");
        let second = fixture("d2d4");
        let (mut endpoint, mut owner) = channel();
        let mut cpu = engine_with_tie(side_scope, CpuRootTiePolicy::RulesOrderConfirmed);
        let mut binding = None;
        endpoint.try_submit(payload(&first), profile(&cpu)).unwrap();
        owner
            .service(
                &mut cpu,
                &mut binding,
                true,
                true,
                empty_occupancy(),
                false,
                &CompletionSignal::default(),
            )
            .unwrap();
        let first_receipt = endpoint.try_ack().unwrap().unwrap();
        let actual = first_receipt.result().unwrap().profile();
        let old_owner = cpu.paused_stack_snapshot().unwrap().owner_id;
        let old_search = cpu.search_identity();
        let old_value = cpu.value_identity().clone();
        let old_binding = binding;
        assert_eq!(cpu.job_binding_digest().unwrap(), old_binding);
        let mut mismatched = actual;
        mismatched.search_conditions.0[0] ^= 1;
        endpoint.try_submit(payload(&second), mismatched).unwrap();
        owner
            .service(
                &mut cpu,
                &mut binding,
                true,
                true,
                empty_occupancy(),
                false,
                &CompletionSignal::default(),
            )
            .unwrap();
        let rejected = endpoint.try_ack().unwrap().unwrap();
        assert_eq!(
            rejected.result().err(),
            Some(&CpuWorkerError::Invalid(
                "prepared value previous profile mismatch"
            ))
        );
        assert_eq!(cpu.root_tie_policy(), CpuRootTiePolicy::RulesOrderConfirmed);
        assert_eq!(cpu.search_identity(), old_search);
        assert_eq!(cpu.value_identity(), &old_value);
        assert_eq!(cpu.paused_stack_snapshot().unwrap().owner_id, old_owner);
        assert_eq!(cpu_search_conditions_digest(&cpu), actual.search_conditions);
        assert_eq!(binding, old_binding);
        assert_eq!(cpu.job_binding_digest().unwrap(), old_binding);
    }
}

fn profile(engine: &CpuEngine) -> CpuWorkerProfile {
    CpuWorkerProfile {
        owner: CpuWorkerId(0),
        max_depth: engine.config().max_depth,
        evaluation_namespace: cpu_evaluation_namespace_digest(engine).unwrap(),
        search_conditions: cpu_search_conditions_digest(engine),
    }
}

fn empty_occupancy() -> MailboxOccupancy {
    MailboxOccupancy {
        commands: 0,
        reliable_queued: 0,
        reliable_reserved: 0,
        progress_present: false,
        closed: false,
    }
}

fn wait_install(worker: &mut CpuWorker) -> ValueInstallReceipt {
    let until = Instant::now() + WINDOW;
    loop {
        let observed = worker.signal.version();
        if let Some(receipt) = worker.try_install_readback().unwrap() {
            return receipt;
        }
        assert!(
            Instant::now() < until,
            "owner did not publish install readback"
        );
        worker
            .signal
            .wait_changed(observed, Duration::from_millis(10));
    }
}

fn wait_report(worker: &CpuWorker) -> CpuSliceReport {
    let until = Instant::now() + WINDOW;
    loop {
        let observed = worker.signal.version();
        if let Some(event) = worker.try_event().unwrap() {
            return match event {
                CpuWorkerEvent::Slice(report) => report,
                CpuWorkerEvent::JobTerminated(value) => panic!("unexpected termination: {value:?}"),
            };
        }
        assert!(Instant::now() < until, "CPU slice did not publish a report");
        worker
            .signal
            .wait_changed(observed, Duration::from_millis(10));
    }
}

fn wait_join(worker: &mut CpuWorker) -> CpuWorkerShutdown {
    let until = Instant::now() + WINDOW;
    loop {
        let observed = worker.signal.version();
        if let Some(receipt) = worker.try_join().unwrap() {
            return receipt;
        }
        assert!(Instant::now() < until, "CPU owner did not join");
        worker
            .signal
            .wait_changed(observed, Duration::from_millis(10));
    }
}

fn job(worker: &CpuWorker, clock: &InstantClock, sequence: u64) -> CpuJob {
    let position = ContractPosition::new(OwnerId(71), Position::startpos());
    let profile = worker.acknowledged_profile().unwrap();
    let envelope = CpuJobEnvelope {
        id: CpuJobId {
            authority: SearchAuthority {
                epoch: clock.domain().0,
                game: GameGeneration(1),
                root: RootGeneration(1),
                implementation: Digest([9; 32]),
            },
            sequence,
        },
        state_identity: position.export().unwrap().snapshot().identity(),
        question: Digest([7; 32]),
        root_restriction: cpu_root_restriction_digest(None),
        evaluation_namespace: profile.evaluation_namespace,
        search_conditions: profile.search_conditions,
        total_native_node_credit: 4096,
        hard_deadline: clock.deadline_at(clock.origin() + WINDOW).unwrap(),
    };
    worker
        .prepare_job(envelope, position, None, 2, clock)
        .unwrap()
}

fn grant(job: &CpuJob, sequence: u64) -> SliceGrant {
    let envelope = job.envelope();
    SliceGrant {
        job: envelope.id,
        execution: ExecutionId::new(envelope.id.authority.epoch, sequence),
        owner: CpuWorkerId(0),
        node_credit: 1,
        soft_lease: Duration::from_millis(100),
        control_sequence: sequence,
    }
}

#[test]
fn actual_idle_owner_installs_only_after_ack_and_preserves_search_profile() {
    let fixture = fixture("e2e4");
    let mut cpu = engine(true);
    cpu.set_paused_stack_max_bytes(128 * 1024).unwrap();
    let mut worker = CpuWorker::spawn(CpuWorkerId(0), cpu).unwrap();
    let old = worker.profile();
    let value = payload(&fixture);
    let binding = value.binding_digest();
    let id = worker.try_install_prepared_value(value).unwrap();
    assert!(worker.value_install_pending().unwrap());
    assert_eq!(worker.profile(), old);
    assert_eq!(
        worker.acknowledged_profile(),
        Err(CpuWorkerError::AlreadyReserved)
    );
    let rejected = worker
        .try_install_prepared_value(payload(&fixture))
        .unwrap_err();
    assert_eq!(rejected.reason(), &CpuWorkerError::Capacity);
    assert_eq!(rejected.into_parts().1.binding_digest(), binding);
    let receipt = wait_install(&mut worker);
    let readback = receipt.result().unwrap();
    assert_eq!(receipt.installation_id(), id);
    assert_eq!(receipt.binding_digest(), binding);
    assert!(receipt.matches_bound(&fixture.bound));
    assert_ne!(readback.actual_stack_owner_id(), 0);
    assert_eq!(readback.profile().owner, old.owner);
    assert_eq!(readback.profile().max_depth, old.max_depth);
    assert_eq!(readback.profile().search_conditions, old.search_conditions);
    assert_eq!(readback.profile().evaluation_namespace, binding);
    assert!(readback.tt_reset() && readback.ordering_history_reset());
    assert!(readback.side_scope_preserved());
    assert_eq!(readback.paused_bytes_limit(), 128 * 1024);
    assert_eq!(worker.acknowledged_profile().unwrap(), readback.profile());
    assert!(worker.validates_installation_receipt(&receipt));
    let stopped = worker.shutdown().unwrap();
    assert!(stopped.value_installation_receipt.is_none());
    assert!(stopped.value_installation_error.is_none());
}

#[test]
fn equal_question_view_and_binding_do_not_attest_another_preparation_instance() {
    let fixture = fixture("e2e4");
    let request = ReviewRequest::admit(
        &fixture.model,
        Arc::clone(&fixture.view),
        &fixture.stores,
        fixture.bound.provenance().root,
        fixture.bound.provenance().generation,
        Instant::now() + WINDOW,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    fixture
        ._actor
        .try_submit(request)
        .unwrap_or_else(|_| panic!("prepare"));
    let until = Instant::now() + WINDOW;
    let separate = loop {
        if let Some(completion) = fixture._actor.try_complete().unwrap() {
            break Arc::new(completion.result.unwrap());
        }
        assert!(Instant::now() < until);
        std::thread::yield_now();
    };
    assert_eq!(separate.provenance(), fixture.bound.provenance());
    let other = PreparedValueInstallation::from_bound(
        Arc::clone(&fixture.model),
        Arc::clone(&separate),
        &fixture.stores,
        separate.provenance().view_seal,
    )
    .unwrap();
    let mut worker = CpuWorker::spawn(CpuWorkerId(0), engine(false)).unwrap();
    worker
        .try_install_prepared_value(payload(&fixture))
        .unwrap();
    let receipt = wait_install(&mut worker);
    assert_eq!(receipt.binding_digest(), other.binding_digest());
    assert!(receipt.matches_bound(&fixture.bound));
    assert!(!receipt.matches_bound(&separate));
    worker.shutdown().unwrap();
}

#[test]
fn actual_paused_owner_rejects_installation_without_replacing_its_job() {
    let fixture = fixture("e2e4");
    let mut worker = CpuWorker::spawn(CpuWorkerId(0), engine(false)).unwrap();
    let old = worker.profile();
    let clock = InstantClock::new(ClockDomain(ProcessEpoch(41)), Instant::now());
    let original = job(&worker, &clock, 1);
    let first = grant(&original, 1);
    worker.begin(original, first).unwrap();
    let report = wait_report(&worker);
    assert!(report.continuation_retained && !report.terminal);
    assert_eq!(report.actual_work.unwrap().nodes, 1);
    worker
        .try_install_prepared_value(payload(&fixture))
        .unwrap();
    let receipt = wait_install(&mut worker);
    assert_eq!(receipt.result().err(), Some(&CpuWorkerError::Capacity));
    assert!(!receipt.is_installed());
    assert_eq!(worker.acknowledged_profile().unwrap(), old);
    let stopped = worker.shutdown().unwrap();
    assert!(
        stopped
            .events
            .iter()
            .any(|event| matches!(event, CpuWorkerEvent::JobTerminated(value) if value.shutdown))
    );
    assert!(stopped.engine_recoverable);
}

#[test]
fn a_preinstall_job_cannot_bypass_the_new_namespace_with_a_fresh_grant() {
    let fixture = fixture("e2e4");
    let mut worker = CpuWorker::spawn(CpuWorkerId(0), engine(false)).unwrap();
    let clock = InstantClock::new(ClockDomain(ProcessEpoch(41)), Instant::now());
    let stale = job(&worker, &clock, 1);
    let request = grant(&stale, 1);
    worker
        .try_install_prepared_value(payload(&fixture))
        .unwrap();
    let receipt = wait_install(&mut worker);
    assert!(receipt.is_installed());
    assert_ne!(
        stale.envelope().evaluation_namespace,
        worker.profile().evaluation_namespace
    );
    worker.begin(stale, request).unwrap();
    let rejected = wait_report(&worker);
    assert!(rejected.admission_rejected);
    assert_eq!(rejected.actual_work, Some(CpuWork::default()));
    assert!(matches!(
        rejected.result,
        Err(crate::cpu::CpuError::InvalidLimits(
            "slice does not name the live private job"
        ))
    ));
    assert!(!rejected.continuation_retained);
    worker.shutdown().unwrap();
}

#[test]
fn expired_idle_value_can_be_replaced_and_recovered_binding_survives_respawn() {
    let old = fixture("e2e4");
    let new = fixture("d2d4");
    let mut worker = CpuWorker::spawn(CpuWorkerId(0), engine(false)).unwrap();
    worker.try_install_prepared_value(payload(&old)).unwrap();
    let first = wait_install(&mut worker);
    assert!(first.is_installed());
    old.cancel.store(true, Ordering::Release);
    let value = payload(&new);
    let binding = value.binding_digest();
    assert_ne!(binding, first.binding_digest());
    worker.try_install_prepared_value(value).unwrap();
    let latest = wait_install(&mut worker);
    assert!(latest.is_installed());
    assert!(!worker.validates_installation_receipt(&first));
    assert!(worker.validates_installation_receipt(&latest));
    worker.request_shutdown();
    let stopped = wait_join(&mut worker);
    assert!(stopped.engine_recoverable && stopped.value_installation_error.is_none());
    let cpu = worker.take_engine().unwrap();
    assert_eq!(cpu.job_binding_digest().unwrap(), Some(binding));
    let mut restarted = CpuWorker::spawn(CpuWorkerId(0), cpu).unwrap();
    assert_eq!(
        restarted
            .acknowledged_profile()
            .unwrap()
            .evaluation_namespace,
        binding
    );
    assert!(!restarted.validates_installation_receipt(&latest));
    restarted.request_shutdown();
    assert!(wait_join(&mut restarted).engine_recoverable);
    let recovered = restarted.take_engine().unwrap();
    new.cancel.store(true, Ordering::Release);
    match CpuWorker::spawn_notified_retaining(
        CpuWorkerId(0),
        recovered,
        CompletionSignal::default(),
    ) {
        Ok(_) => panic!("revoked recovered value was admitted as a live worker"),
        Err((returned, reason)) => {
            assert_eq!(
                reason,
                CpuWorkerError::Cpu(crate::cpu::CpuError::Value(
                    crate::cpu_value::CpuValueError::AuthorityRevoked(
                        crate::cpu_value::CpuValueAuthorityRevocation::Cancelled,
                    )
                ))
            );
            assert_eq!(returned.value_identity(), new.model.identity());
        }
    }
}

#[test]
fn one_terminal_seat_remains_reserved_until_the_private_ack_is_checked() {
    let fixture = fixture("e2e4");
    let (mut endpoint, mut owner) = channel();
    let mut cpu = engine(false);
    let old = profile(&cpu);
    let mut binding = None;
    let signal = CompletionSignal::default();
    let value = payload(&fixture);
    let expected = value.binding_digest();
    endpoint.try_submit(value, old).unwrap();
    owner
        .service(
            &mut cpu,
            &mut binding,
            true,
            true,
            empty_occupancy(),
            false,
            &signal,
        )
        .unwrap();
    assert!(endpoint.pending());
    assert_eq!(
        endpoint
            .try_submit(payload(&fixture), old)
            .unwrap_err()
            .reason(),
        &CpuWorkerError::Capacity
    );
    let first = endpoint.try_ack().unwrap().unwrap();
    assert!(!endpoint.pending());
    assert_eq!(binding, Some(expected));
    assert!(endpoint.validates_receipt(&first));
    endpoint
        .try_submit(payload(&fixture), first.installed_profile().unwrap())
        .unwrap();
    assert!(!endpoint.validates_receipt(&first));
    owner
        .service(
            &mut cpu,
            &mut binding,
            true,
            true,
            empty_occupancy(),
            false,
            &signal,
        )
        .unwrap();
    let second = endpoint.try_ack().unwrap().unwrap();
    assert!(!endpoint.validates_receipt(&first));
    assert!(endpoint.validates_receipt(&second));
    let mut forged = second.clone();
    forged.authority = Arc::new(());
    assert!(!endpoint.validates_receipt(&forged));
}

#[test]
fn soft_terminal_recovery_requires_the_same_actual_bound_and_installed_binding() {
    use crate::cpu_value::{CpuValueAuthorityRevocation, CpuValueError};
    let fixture = fixture("e2e4");
    let value = crate::cpu_value::nnuec::v2::cpu::V2CpuValue::from_bound(
        Arc::clone(&fixture.model),
        Arc::clone(&fixture.bound),
        &fixture.stores,
        fixture.view.seal_digest(),
    )
    .unwrap();
    let cpu = CpuEngine::with_evaluator_ordering_and_resume_policy(
        engine(false).config().clone(),
        Arc::new(value),
        CpuOrderingPolicy::LegacyMvvLvaV1,
        CpuResumePolicy::PausedStack,
    )
    .unwrap();
    let binding = cpu.job_binding_digest().unwrap();
    let cancelled = crate::cpu::CpuError::Value(CpuValueError::AuthorityRevoked(
        CpuValueAuthorityRevocation::Cancelled,
    ));
    let known_work = Some(CpuWork::default());
    assert_eq!(
        super::super::owned_soft_value_terminal(&cpu, binding, &cancelled, known_work),
        None
    );
    fixture.cancel.store(true, Ordering::Release);
    assert_eq!(
        super::super::owned_soft_value_terminal(&cpu, binding, &cancelled, known_work),
        Some(crate::cpu::CpuCompletion::Canceled),
    );
    assert_eq!(
        super::super::owned_soft_value_terminal(&cpu, None, &cancelled, known_work),
        None
    );
    assert_eq!(
        super::super::owned_soft_value_terminal(&cpu, binding, &cancelled, None),
        None
    );
    let mut foreign_binding = binding.unwrap();
    foreign_binding.0[0] ^= 1;
    assert_eq!(
        super::super::owned_soft_value_terminal(
            &cpu,
            Some(foreign_binding),
            &cancelled,
            known_work
        ),
        None
    );
    assert_eq!(
        super::super::owned_soft_value_terminal(&engine(false), binding, &cancelled, known_work),
        None
    );
    for error in [
        CpuValueError::AuthorityRevoked(CpuValueAuthorityRevocation::Expired),
        CpuValueError::AuthorityRevoked(CpuValueAuthorityRevocation::ActorStopped),
        CpuValueError::AuthorityRevoked(CpuValueAuthorityRevocation::Invalidated),
        CpuValueError::StateMismatch,
        CpuValueError::NonFiniteForward,
    ] {
        assert_eq!(
            super::super::owned_soft_value_terminal(
                &cpu,
                binding,
                &crate::cpu::CpuError::Value(error),
                known_work,
            ),
            None
        );
    }
    fixture._actor.invalidate_cache().unwrap();
    assert_eq!(
        super::super::owned_soft_value_terminal(&cpu, binding, &cancelled, known_work),
        None
    );
}

#[test]
fn original_expiry_then_cancellation_preserves_deadline_without_accepting_hard_revocation() {
    use crate::cpu_value::{CpuValueAuthorityRevocation, CpuValueError};
    use crate::cpu_worker::NnuecAuthorityRevocation;
    let original_deadline = Instant::now();
    let before = original_deadline
        .checked_sub(Duration::from_nanos(1))
        .unwrap();
    let after = original_deadline
        .checked_add(Duration::from_nanos(1))
        .unwrap();
    let error = crate::cpu::CpuError::Value(CpuValueError::AuthorityRevoked(
        CpuValueAuthorityRevocation::Expired,
    ));
    let original_error = error.clone();
    assert!(super::super::original_soft_revocation_matches(
        CpuValueAuthorityRevocation::Expired,
        Some(NnuecAuthorityRevocation::Expired),
        original_deadline,
        original_deadline,
    ));
    assert!(super::super::original_soft_revocation_matches(
        CpuValueAuthorityRevocation::Expired,
        Some(NnuecAuthorityRevocation::Cancelled),
        original_deadline,
        after,
    ));
    assert!(!super::super::original_soft_revocation_matches(
        CpuValueAuthorityRevocation::Expired,
        Some(NnuecAuthorityRevocation::Cancelled),
        original_deadline,
        before,
    ));
    for current in [
        None,
        Some(NnuecAuthorityRevocation::ActorStopped),
        Some(NnuecAuthorityRevocation::Invalidated),
    ] {
        assert!(!super::super::original_soft_revocation_matches(
            CpuValueAuthorityRevocation::Expired,
            current,
            original_deadline,
            after,
        ));
    }
    for hard in [
        CpuValueAuthorityRevocation::ActorStopped,
        CpuValueAuthorityRevocation::Invalidated,
    ] {
        assert!(!super::super::original_soft_revocation_matches(
            hard,
            Some(NnuecAuthorityRevocation::Cancelled),
            original_deadline,
            after,
        ));
    }
    assert!(!super::super::original_soft_revocation_matches(
        CpuValueAuthorityRevocation::Cancelled,
        Some(NnuecAuthorityRevocation::Expired),
        original_deadline,
        after,
    ));
    // Only the comparison helper consumed deterministic clock observations;
    // no real Bound, registered deadline or original error was changed.
    assert_eq!(error, original_error);
    assert_eq!(
        error.value_authority_stop(),
        Some(crate::cpu::CpuCompletion::Deadline)
    );
}

#[test]
fn every_busy_mailbox_scope_preserves_the_original_engine_and_namespace() {
    let fixture = fixture("e2e4");
    let cases = [
        (false, empty_occupancy()),
        (
            true,
            MailboxOccupancy {
                commands: 1,
                ..empty_occupancy()
            },
        ),
        (
            true,
            MailboxOccupancy {
                reliable_queued: 1,
                ..empty_occupancy()
            },
        ),
        (
            true,
            MailboxOccupancy {
                reliable_reserved: 1,
                ..empty_occupancy()
            },
        ),
        (
            true,
            MailboxOccupancy {
                progress_present: true,
                ..empty_occupancy()
            },
        ),
    ];
    for (idle, occupancy) in cases {
        let (mut endpoint, mut owner) = channel();
        let mut cpu = engine(false);
        let old = profile(&cpu);
        let identity = cpu.value_identity().clone();
        let mut binding = None;
        endpoint.try_submit(payload(&fixture), old).unwrap();
        owner
            .service(
                &mut cpu,
                &mut binding,
                idle,
                true,
                occupancy,
                false,
                &CompletionSignal::default(),
            )
            .unwrap();
        let receipt = endpoint.try_ack().unwrap().unwrap();
        assert_eq!(receipt.result().err(), Some(&CpuWorkerError::Capacity));
        assert_eq!(receipt.previous_profile(), old);
        assert_eq!(profile(&cpu), old);
        assert_eq!(cpu.value_identity(), &identity);
        assert_eq!(binding, None);
    }
}

#[test]
fn cancellation_before_enqueue_or_owner_acceptance_preserves_the_old_engine() {
    let first = fixture("e2e4");
    let value = payload(&first);
    first.cancel.store(true, Ordering::Release);
    let (mut endpoint, _) = channel();
    let cpu = engine(false);
    let error = endpoint.try_submit(value, profile(&cpu)).unwrap_err();
    assert_eq!(
        error.reason(),
        &CpuWorkerError::Invalid("prepared value authority revoked")
    );
    assert!(!endpoint.pending());
    let second = fixture("d2d4");
    let (mut endpoint, mut owner) = channel();
    let mut cpu = engine(false);
    let previous = profile(&cpu);
    let mut binding = None;
    endpoint.try_submit(payload(&second), previous).unwrap();
    second.cancel.store(true, Ordering::Release);
    owner
        .service(
            &mut cpu,
            &mut binding,
            true,
            true,
            empty_occupancy(),
            false,
            &CompletionSignal::default(),
        )
        .unwrap();
    let receipt = endpoint.try_ack().unwrap().unwrap();
    assert_eq!(
        receipt.result().err(),
        Some(&CpuWorkerError::Invalid("prepared value authority revoked"))
    );
    assert_eq!(profile(&cpu), previous);
    assert_eq!(binding, None);
}

#[test]
fn uncollected_installation_terminal_is_preserved_by_actual_worker_shutdown() {
    let fixture = fixture("e2e4");
    let worker = CpuWorker::spawn(CpuWorkerId(0), engine(false)).unwrap();
    let id = worker
        .try_install_prepared_value(payload(&fixture))
        .unwrap();
    let stopped = worker.shutdown().unwrap();
    assert!(stopped.value_installation_error.is_none());
    let receipt = stopped
        .value_installation_receipt
        .expect("admitted installation lost its terminal");
    assert_eq!(receipt.installation_id(), id);
    match receipt.result() {
        Ok(readback) => assert_eq!(
            readback.profile().evaluation_namespace,
            receipt.binding_digest()
        ),
        Err(error) => assert_eq!(error, &CpuWorkerError::Disconnected),
    }
}

#[test]
fn private_ack_identity_mismatch_quarantines_instead_of_releasing_pending() {
    let fixture = fixture("e2e4");
    let (mut endpoint, owner) = channel();
    let cpu = engine(false);
    let previous = profile(&cpu);
    endpoint.try_submit(payload(&fixture), previous).unwrap();
    let request = owner.receiver.try_recv().unwrap();
    owner
        .sender
        .try_send(ValueInstallReceipt {
            authority: Arc::new(()),
            id: request.id,
            previous_profile: previous,
            binding: request.payload.binding,
            bound_authority: request.payload.bound.instance_authority().clone(),
            result: Err(CpuWorkerError::Capacity),
        })
        .unwrap();
    assert!(matches!(
        endpoint.try_ack(),
        Err(CpuWorkerError::Quarantined)
    ));
    assert!(endpoint.pending());
}

#[test]
fn unsupported_or_unknown_paused_owner_cannot_claim_idle_installation() {
    let fixture = fixture("e2e4");
    let mut cpu = CpuEngine::new(CpuConfig {
        profile: CpuProfile::PlanAssisted,
        tt_entries: 16,
        max_depth: 2,
        quiescence_ply: 0,
    })
    .unwrap();
    assert_eq!(cpu.paused_stack_snapshot(), None);
    let old = profile(&cpu);
    let mut binding = None;
    let (mut endpoint, mut owner) = channel();
    endpoint.try_submit(payload(&fixture), old).unwrap();
    assert_eq!(
        owner.service(
            &mut cpu,
            &mut binding,
            true,
            true,
            empty_occupancy(),
            false,
            &CompletionSignal::default(),
        ),
        Err(CpuWorkerError::Quarantined),
    );
    let receipt = endpoint.try_ack().unwrap().unwrap();
    assert_eq!(receipt.result().err(), Some(&CpuWorkerError::Quarantined));
    assert!(endpoint.pending());
    assert_eq!(profile(&cpu), old);
    assert_eq!(binding, None);
}
