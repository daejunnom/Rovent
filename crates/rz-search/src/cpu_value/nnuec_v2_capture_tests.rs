//! Causal capture regressions. The private reservation fixture supplies a
//! synthetic original submission; actual worker/score/retirement paths remain
//! real. This does not supply a production semantic acceptance or WDL target.

use super::*;
use crate::cpu::{CpuConfig, CpuOrderingPolicy, CpuProfile, CpuResumePolicy};
use crate::cpu_value::nnuec::v2::cpu::V2CpuValue;
use crate::cpu_value::nnuec::v2::review::{NnuecPreparationTerminalAck, PreparationActor, Request};
use crate::cpu_worker::{
    CpuJob, CpuSliceReport, CpuWorker, CpuWorkerError, CpuWorkerEvent, cpu_root_restriction_digest,
};
use crate::pals::engine::{RecordKind, RoleQueryPurpose, RoleRecord};
use crate::pals::review_context::{
    ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy,
};
use crate::pals::store::StoreLimits;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::{CpuJobId, SliceGrant};
use rz_contracts::{
    ClockDomain, ExecutionId, GameGeneration, OwnerId, ProcessEpoch, RootGeneration,
};
use rz_position::contracts::ContractPosition;
use std::sync::atomic::{AtomicBool, Ordering};

const WINDOW: Duration = Duration::from_secs(20);

struct Fixture {
    model: Arc<Model>,
    bound: Arc<BoundNnuecReviewContext>,
    stores: PalsStores,
    clock: InstantClock,
    cancel: Arc<AtomicBool>,
    terminal: NnuecPreparationTerminalAck,
    _actor: PreparationActor,
}

fn fixture() -> Fixture {
    fixture_with_utility_target(false)
}

fn fixture_with_utility_target(capture_utility: bool) -> Fixture {
    let clock = InstantClock::new(ClockDomain(ProcessEpoch(301)), Instant::now());
    let model = Arc::new(Model::new(super::super::tests::checkpoint()).unwrap());
    let mut stores = PalsStores::new(StoreLimits::default());
    let situation = stores
        .focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let root = stores.situations.get(situation).unwrap().state;
    let mut index = ReviewContextIndex::default();
    if capture_utility {
        index
            .publish(
                &stores,
                &RoleRecord {
                    revision: 1,
                    parent_revision: None,
                    supersedes_revision: None,
                    origin_state: root,
                    kind: RecordKind::Proposal,
                    line: vec!["e2e4".parse().unwrap()],
                    value: None,
                    completed_depth: 0,
                    score_scope: None,
                    cpu_observation: None,
                    perspective: rz_position::Color::White,
                    critical: true,
                },
                None,
            )
            .unwrap();
    }
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
    let request = Request::admit(
        &model,
        view,
        &stores,
        root,
        stores.generation(),
        clock.origin() + WINDOW,
        Arc::clone(&cancel),
    )
    .unwrap();
    let request = if capture_utility {
        request
            .with_utility_input_capture(0)
            .unwrap_or_else(|_| panic!("utility source selection"))
    } else {
        request
    };
    let actor = PreparationActor::spawn(Arc::clone(&model)).unwrap();
    actor
        .try_submit(request)
        .unwrap_or_else(|_| panic!("prepare admission"));
    let (bound, terminal) = loop {
        if let Some(completion) = actor.try_complete().unwrap() {
            break (
                Arc::new(completion.result.unwrap()),
                completion.terminal_ack,
            );
        }
        assert!(Instant::now() < clock.origin() + WINDOW);
        std::thread::yield_now();
    };
    Fixture {
        model,
        bound,
        stores,
        clock,
        cancel,
        terminal,
        _actor: actor,
    }
}

fn worker(fixture: &Fixture) -> CpuWorker {
    let value = V2CpuValue::from_bound(
        Arc::clone(&fixture.model),
        Arc::clone(&fixture.bound),
        &fixture.stores,
        fixture.bound.provenance().view_seal,
    )
    .unwrap();
    let engine = crate::cpu::CpuEngine::with_evaluator_ordering_and_resume_policy(
        CpuConfig {
            profile: CpuProfile::PlanAssisted,
            tt_entries: 16,
            max_depth: 2,
            quiescence_ply: 0,
        },
        Arc::new(value),
        CpuOrderingPolicy::LegacyMvvLvaV1,
        CpuResumePolicy::PausedStack,
    )
    .unwrap();
    CpuWorker::spawn(CpuWorkerId(0), engine).unwrap()
}

fn job(fixture: &Fixture, worker: &CpuWorker, sequence: u64, depth: u16) -> CpuJob {
    let position = ContractPosition::new(OwnerId(701), Position::startpos());
    let profile = worker.acknowledged_profile().unwrap();
    let envelope = CpuJobEnvelope {
        id: CpuJobId {
            authority: SearchAuthority {
                epoch: fixture.clock.domain().0,
                game: GameGeneration(1),
                root: RootGeneration(fixture.bound.provenance().generation),
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
        hard_deadline: fixture
            .clock
            .deadline_at(fixture.clock.origin() + WINDOW)
            .unwrap(),
    };
    worker
        .prepare_job(envelope, position, None, depth, &fixture.clock)
        .unwrap()
}

fn reservation(
    fixture: &Fixture,
    job: &CpuJob,
    limits: NnuecCaptureLimits,
) -> NnuecCaptureReservation {
    NnuecCaptureReservation {
        submission: Arc::new(Submission {
            lineage: NnuecCaptureLineage::new(
                fixture.bound.instance_authority().clone(),
                fixture.bound.request_identity(),
                FrameId {
                    authority: job.envelope().id.authority,
                    sequence: 1,
                },
                job.envelope().question,
                1,
                job.envelope().hard_deadline,
                job.envelope().evaluation_namespace,
                Digest(fixture.bound.view().digest()),
                Digest(fixture.bound.view().seal_digest()),
            ),
            model: Arc::clone(&fixture.model),
            bound: Arc::clone(&fixture.bound),
            frame: FrameId {
                authority: job.envelope().id.authority,
                sequence: 1,
            },
            question: job.envelope().question,
            capture_sequence: 1,
            limits,
            binding: job.envelope().evaluation_namespace,
            original_deadline: job.envelope().hard_deadline,
            context_sha256: fixture.bound.context().sha256().unwrap(),
            question_anchor: Position::startpos().snapshot(),
        }),
        state: Arc::new(Mutex::new(State::default())),
        retirement: Arc::new(OnceLock::new()),
    }
}

fn grant(job: &CpuJob, sequence: u64, credit: u64) -> SliceGrant {
    let envelope = job.envelope();
    SliceGrant {
        job: envelope.id,
        execution: ExecutionId::new(envelope.id.authority.epoch, sequence),
        owner: CpuWorkerId(0),
        node_credit: credit,
        soft_lease: Duration::from_secs(5),
        control_sequence: sequence,
    }
}

fn report(worker: &CpuWorker) -> CpuSliceReport {
    let until = Instant::now() + WINDOW;
    loop {
        let observed = worker.signal.version();
        if let Some(event) = worker.try_event().unwrap() {
            return match event {
                CpuWorkerEvent::Slice(report) => report,
                CpuWorkerEvent::JobTerminated(_) => panic!("unexpected idle termination"),
            };
        }
        assert!(Instant::now() < until);
        worker
            .signal
            .wait_changed(observed, Duration::from_millis(10));
    }
}

fn assert_same_report(a: &CpuSliceReport, b: &CpuSliceReport) {
    assert_eq!(a.actual_work, b.actual_work);
    assert_eq!(a.cumulative_nodes, b.cumulative_nodes);
    let a = &a.result.as_ref().unwrap().report;
    let b = &b.result.as_ref().unwrap().report;
    assert_eq!(a.best_move, b.best_move);
    assert_eq!(a.score, b.score);
    assert_eq!(a.pv, b.pv);
    assert_eq!(a.completed_depth, b.completed_depth);
    assert_eq!(a.nodes, b.nodes);
    assert_eq!(a.quiescence_nodes, b.quiescence_nodes);
    assert_eq!(a.tt_hits, b.tt_hits);
    assert_eq!(a.completion, b.completion);
}

#[test]
fn capture_settings_require_explicit_bounded_nonzero_limits() {
    assert!(NnuecCaptureLimits::new(0, 1, 1).is_err());
    assert!(NnuecCaptureLimits::new(65, 1, 1).is_err());
    assert!(NnuecCaptureLimits::new(1, 0, 1).is_err());
    assert!(NnuecCaptureLimits::new(1, 1, MAX_CAPTURE_BYTES + 1).is_err());
}

#[test]
fn utility_source_matches_only_the_same_original_bound_and_request_r_lineage() {
    let fixture = fixture_with_utility_target(true);
    let foreign = fixture_with_utility_target(true);
    let utility = Arc::clone(fixture.bound.utility_input_capture().unwrap());
    let worker = worker(&fixture);
    let mut original = job(&fixture, &worker, 1, 1);
    let capture = reservation(
        &fixture,
        &original,
        NnuecCaptureLimits::new(4, 1, 1 << 20).unwrap(),
    );
    let foreign_capture = reservation(
        &foreign,
        &original,
        NnuecCaptureLimits::new(4, 1, 1 << 20).unwrap(),
    );
    assert!(utility.matches_capture_lineage(&capture.lineage()));
    assert!(!utility.matches_capture_lineage(&foreign_capture.lineage()));
    assert!(
        !capture
            .lineage()
            .matches_activated_job(original.envelope(), CpuWorkerId(0)),
        "shared C source alone proves no execution"
    );
    original
        .enable_nnuec_capture(capture.clone())
        .unwrap_or_else(|_| panic!("actual capture attachment"));
    let first = grant(&original, 1, 4096);
    worker.begin(original, first).unwrap();
    assert!(report(&worker).terminal);
    let retired = capture.retirement().unwrap();
    assert!(retired.matches_lineage(&capture.lineage()));
    assert!(utility.matches_capture_lineage(&capture.lineage()));
    assert!(utility.reference().base_input.local_line.moves.is_empty());
    fixture.cancel.store(true, Ordering::Release);
    assert!(!utility.validates(
        &fixture.stores,
        fixture.bound.provenance().root,
        fixture.stores.generation(),
        fixture.bound.view()
    ));
    assert!(
        utility.matches_capture_lineage(&capture.lineage()),
        "historical matching never revives action authority"
    );
    worker.shutdown().unwrap();
}

#[test]
fn utility_c_only_closure_matchers_require_the_exact_request_and_actual_actor_cache_ack() {
    let original = fixture_with_utility_target(true);
    let foreign = fixture_with_utility_target(true);
    let cap = original.bound.utility_input_capture().unwrap();
    assert!(cap.matches_preparation_terminal(&original.terminal));
    assert!(!cap.matches_preparation_terminal(&foreign.terminal));
    assert!(original.terminal.allows_c_only_cleanup());
    let own_ticket = original._actor.invalidate_context_cache().unwrap();
    let own_ack = loop {
        if let Some(ack) = original
            ._actor
            .try_context_cache_release(&own_ticket)
            .unwrap()
        {
            break ack;
        }
        assert!(Instant::now() < original.clock.origin() + WINDOW);
        std::thread::yield_now();
    };
    let foreign_ticket = foreign._actor.invalidate_context_cache().unwrap();
    let foreign_ack = loop {
        if let Some(ack) = foreign
            ._actor
            .try_context_cache_release(&foreign_ticket)
            .unwrap()
        {
            break ack;
        }
        assert!(Instant::now() < foreign.clock.origin() + WINDOW);
        std::thread::yield_now();
    };
    assert!(cap.matches_context_cache_release(&own_ack));
    assert!(!cap.matches_context_cache_release(&foreign_ack));
    assert!(original.terminal.matches_cache_release(&own_ack));
    assert!(!original.bound.authority_is_live());
    assert!(cap.reference().base_input.local_line.moves.is_empty());
}

#[test]
fn lineage_requires_actual_begin_and_rejects_an_equal_metadata_reservation() {
    let fixture = fixture();
    let worker = worker(&fixture);
    let mut original = job(&fixture, &worker, 1, 1);
    let envelope = original.envelope();
    let capture = reservation(
        &fixture,
        &original,
        NnuecCaptureLimits::new(4, 1, 1024 * 1024).unwrap(),
    );
    let foreign = reservation(
        &fixture,
        &original,
        NnuecCaptureLimits::new(4, 1, 1024 * 1024).unwrap(),
    );
    let lineage = capture.lineage();
    let foreign_lineage = foreign.lineage();
    assert!(lineage.matches_reservation(&capture));
    assert!(!lineage.matches_reservation(&foreign));
    assert_eq!(lineage.frame(), foreign_lineage.frame());
    assert_eq!(lineage.question_digest(), foreign_lineage.question_digest());
    assert_eq!(lineage.binding_digest(), foreign_lineage.binding_digest());
    assert_eq!(
        lineage.capture_sequence(),
        foreign_lineage.capture_sequence()
    );
    assert_eq!(
        lineage.full_view_digest(),
        Digest(fixture.bound.view().digest())
    );
    assert_eq!(
        lineage.view_seal_digest(),
        Digest(fixture.bound.view().seal_digest())
    );
    assert_eq!(lineage.original_deadline(), envelope.hard_deadline);
    assert!(!lineage.matches_activated_job(envelope, CpuWorkerId(0)));
    assert!(!lineage.activation_observed());
    original
        .enable_nnuec_capture(capture.clone())
        .unwrap_or_else(|_| panic!("attach"));
    assert!(
        !lineage.matches_activated_job(envelope, CpuWorkerId(0)),
        "prepared attachment has not entered its actual worker"
    );
    let first = grant(&original, 1, 4096);
    worker.begin(original, first).unwrap();
    assert!(report(&worker).terminal);
    assert!(lineage.matches_activated_job(envelope, CpuWorkerId(0)));
    assert!(lineage.activation_observed());
    assert!(!lineage.matches_activated_job(envelope, CpuWorkerId(1)));
    let mut other = envelope;
    other.id.sequence += 1;
    assert!(!lineage.matches_activated_job(other, CpuWorkerId(0)));
    assert!(!foreign_lineage.matches_activated_job(envelope, CpuWorkerId(0)));
    let retirement = capture.retirement().unwrap();
    assert!(retirement.matches_lineage(&lineage));
    assert!(lineage.matches_retirement(retirement));
    assert!(capture.matches_retirement(retirement));
    assert!(capture.clone().matches_retirement(retirement));
    assert!(!foreign.matches_retirement(retirement));
    assert!(!retirement.matches_lineage(&foreign_lineage));
    worker.shutdown().unwrap();
}

#[test]
fn historical_lineage_keeps_no_model_bound_or_review_pin_after_actual_cleanup() {
    let fixture = fixture();
    let worker = worker(&fixture);
    let mut original = job(&fixture, &worker, 1, 1);
    let envelope = original.envelope();
    let capture = reservation(
        &fixture,
        &original,
        NnuecCaptureLimits::new(4, 1, 1024 * 1024).unwrap(),
    );
    let before = Arc::strong_count(&fixture.bound);
    let lineage = capture.lineage();
    let cloned = lineage.clone();
    assert_eq!(Arc::strong_count(&fixture.bound), before);
    let model = Arc::downgrade(&fixture.model);
    let bound = Arc::downgrade(&fixture.bound);
    original
        .enable_nnuec_capture(capture.clone())
        .unwrap_or_else(|_| panic!("attach"));
    let first = grant(&original, 1, 4096);
    worker.begin(original, first).unwrap();
    assert!(report(&worker).terminal);
    worker.shutdown().unwrap();
    drop(capture);
    fixture._actor.shutdown().unwrap();
    drop(fixture.bound);
    drop(fixture.model);
    assert!(bound.upgrade().is_none());
    assert!(model.upgrade().is_none());
    assert_eq!(fixture.stores.review_pin_count().unwrap(), 0);
    assert!(lineage.matches_activated_job(envelope, CpuWorkerId(0)));
    assert!(
        cloned.matches_activated_job(envelope, CpuWorkerId(0)),
        "historical equality remains without reviving any execution authority"
    );
}

#[test]
fn foreign_job_question_owner_and_rules_cannot_rebind_original_capture() {
    let fixture = fixture();
    let worker = worker(&fixture);
    let mut original = job(&fixture, &worker, 1, 1);
    let capture = reservation(
        &fixture,
        &original,
        NnuecCaptureLimits::new(4, 1, 1024 * 1024).unwrap(),
    );
    original
        .enable_nnuec_capture(capture.clone())
        .unwrap_or_else(|_| panic!("attach"));
    let mut foreign = job(&fixture, &worker, 2, 1);
    assert!(foreign.enable_nnuec_capture(capture.clone()).is_err());
    assert_eq!(
        capture.error().unwrap(),
        None,
        "foreign attachment cannot poison original"
    );
    let mut envelope = original.envelope();
    envelope.question = Digest([8; 32]);
    assert!(
        capture
            .bind_job(
                envelope,
                CpuWorkerId(0),
                &Position::startpos(),
                fixture.clock,
                capture.submission.binding
            )
            .is_err()
    );
    assert!(
        capture
            .bind_job(
                original.envelope(),
                CpuWorkerId(1),
                &Position::startpos(),
                fixture.clock,
                capture.submission.binding
            )
            .is_err()
    );
    let mut other = Position::startpos();
    other.make_move("e2e4".parse().unwrap()).unwrap();
    assert!(
        capture
            .bind_job(
                original.envelope(),
                CpuWorkerId(0),
                &other,
                fixture.clock,
                capture.submission.binding
            )
            .is_err()
    );
    assert!(
        capture
            .bind_job(
                original.envelope(),
                CpuWorkerId(0),
                &Position::startpos(),
                fixture.clock,
                capture.submission.binding
            )
            .is_ok()
    );
    worker.shutdown().unwrap();
}

#[test]
fn actual_capture_and_capture_off_preserve_score_pv_depth_and_native_work() {
    let fixture = fixture();
    let plain = worker(&fixture);
    let observed = worker(&fixture);
    let plain_job = job(&fixture, &plain, 1, 1);
    let mut observed_job = job(&fixture, &observed, 1, 1);
    let capture = reservation(
        &fixture,
        &observed_job,
        NnuecCaptureLimits::new(64, 1, 1024 * 1024).unwrap(),
    );
    observed_job
        .enable_nnuec_capture(capture.clone())
        .unwrap_or_else(|_| panic!("attach"));
    let a = grant(&plain_job, 1, 4096);
    let b = grant(&observed_job, 1, 4096);
    plain.begin(plain_job, a).unwrap();
    observed.begin(observed_job, b).unwrap();
    let a = report(&plain);
    let b = report(&observed);
    assert!(a.terminal && b.terminal);
    assert_same_report(&a, &b);
    let retirement = capture.retirement().expect("actual terminal retirement");
    assert!(retirement.error().is_none());
    assert!(Arc::ptr_eq(retirement.context_view(), fixture.bound.view()));
    assert_eq!(
        retirement.context_sha256(),
        fixture.bound.numeric_context_sha256()
    );
    assert!(!retirement.readouts().is_empty());
    assert_eq!(
        retirement.ending(),
        NnuecCaptureEnding::CpuTerminal(b.result.as_ref().unwrap().report.completion)
    );
    for readout in retirement.readouts() {
        assert_eq!(readout.job(), retirement.job());
        assert_eq!(readout.question_digest(), retirement.question_digest());
        assert_eq!(readout.numeric().schema, super::super::OUTPUT_SCHEMA);
        assert_eq!(
            Some(&readout.numeric().weights_sha256),
            fixture.model.identity().weights_sha256.as_ref()
        );
        assert!(readout.observed_at().0 <= readout.job().hard_deadline.at.0);
    }
    plain.shutdown().unwrap();
    observed.shutdown().unwrap();
}

#[test]
fn bounded_sample_overflow_is_recorded_without_fabricating_more_readouts() {
    let fixture = fixture();
    let worker = worker(&fixture);
    let mut job = job(&fixture, &worker, 1, 1);
    let capture = reservation(
        &fixture,
        &job,
        NnuecCaptureLimits::new(1, 1, 1024 * 1024).unwrap(),
    );
    job.enable_nnuec_capture(capture.clone())
        .unwrap_or_else(|_| panic!("attach"));
    let grant = grant(&job, 1, 4096);
    worker.begin(job, grant).unwrap();
    assert!(report(&worker).terminal);
    let retirement = capture.retirement().unwrap();
    assert_eq!(retirement.readouts().len(), 1);
    assert!(retirement.attempted() > 1 && retirement.overflow() > 0);
    assert!(retirement.input_bytes() > 0);
    worker.shutdown().unwrap();
}

#[test]
fn insufficient_input_byte_budget_keeps_cpu_result_and_explicit_capture_error() {
    let fixture = fixture();
    let plain = worker(&fixture);
    let observed = worker(&fixture);
    let a = job(&fixture, &plain, 1, 1);
    let mut b = job(&fixture, &observed, 1, 1);
    let capture = reservation(&fixture, &b, NnuecCaptureLimits::new(1, 1, 1).unwrap());
    b.enable_nnuec_capture(capture.clone())
        .unwrap_or_else(|_| panic!("attach"));
    let ga = grant(&a, 1, 4096);
    let gb = grant(&b, 1, 4096);
    plain.begin(a, ga).unwrap();
    observed.begin(b, gb).unwrap();
    assert_same_report(&report(&plain), &report(&observed));
    let retirement = capture.retirement().unwrap();
    assert_eq!(retirement.error(), Some(&CaptureError::Capacity));
    assert!(retirement.readouts().is_empty());
    plain.shutdown().unwrap();
    observed.shutdown().unwrap();
}

#[test]
fn yield_is_not_retirement_and_owner_cancel_retires_once_without_refill() {
    let fixture = fixture();
    let worker = worker(&fixture);
    let mut job = job(&fixture, &worker, 1, 2);
    let envelope = job.envelope();
    let mailbox = Arc::clone(&job.mailbox);
    let capture = reservation(
        &fixture,
        &job,
        NnuecCaptureLimits::new(4, 1, 1024 * 1024).unwrap(),
    );
    job.enable_nnuec_capture(capture.clone())
        .unwrap_or_else(|_| panic!("attach"));
    let grant = grant(&job, 1, 1);
    worker.begin(job, grant).unwrap();
    let first = report(&worker);
    assert!(!first.terminal && first.continuation_retained);
    assert!(capture.retirement().is_none());
    assert_eq!(mailbox.scope().job, envelope.id);
    assert_eq!(mailbox.scope().owner, CpuWorkerId(0));
    let cancel = mailbox.request_cancel(2);
    // The original owner may acknowledge this exact request and seal the
    // mailbox after request.store/notify but before request's final read.
    // Closed alone proves nothing: require that owner's real terminal below.
    assert!(matches!(
        cancel,
        Ok(()) | Err(rz_runtime::adaptive::ControlError::Closed)
    ));
    let until = Instant::now() + WINDOW;
    let termination = loop {
        let observed = worker.signal.version();
        if let Some(event) = worker.try_event().unwrap() {
            match event {
                CpuWorkerEvent::JobTerminated(termination) => break termination,
                CpuWorkerEvent::Slice(_) => {
                    panic!("cancel must not execute or refill another slice")
                }
            }
        }
        assert!(
            Instant::now() < until,
            "original cancel terminal missing; request={cancel:?}"
        );
        worker
            .signal
            .wait_changed(observed, Duration::from_millis(10));
    };
    assert_eq!(termination.job, envelope.id);
    assert_eq!(termination.owner, CpuWorkerId(0));
    assert_eq!(termination.last_execution, first.grant.execution);
    assert_eq!(termination.completion, crate::cpu::CpuCompletion::Canceled);
    assert_eq!(termination.controls.cancel_sequence, 2);
    assert_eq!(termination.controls.cancel_ack, 2);
    assert!(termination.controls.terminal);
    assert_eq!(termination.retained_bytes, 0);
    assert!(
        !termination.shutdown,
        "the owner canceled before process shutdown"
    );
    let controls = mailbox.snapshot();
    assert_eq!(controls.cancel_sequence, 2);
    assert_eq!(controls.cancel_ack, 2);
    assert!(controls.terminal);
    let retirement = capture.retirement().unwrap();
    assert_eq!(
        retirement.ending(),
        NnuecCaptureEnding::CpuTerminal(crate::cpu::CpuCompletion::Canceled)
    );
    let attempted = retirement.attempted();
    assert!(
        capture
            .retire(envelope, CpuWorkerId(0), NnuecCaptureEnding::Failed)
            .is_err()
    );
    assert_eq!(capture.retirement().unwrap().attempted(), attempted);
    let stopped = worker.shutdown().unwrap();
    assert!(stopped.thread_joined && !stopped.thread_panicked && stopped.engine_recoverable);
    assert!(
        stopped.events.is_empty(),
        "original terminal was already collected once"
    );
}

#[test]
fn revoked_original_bound_cannot_bind_and_still_allows_known_unstarted_retirement() {
    let fixture = fixture();
    let worker = worker(&fixture);
    let mut job = job(&fixture, &worker, 1, 1);
    let capture = reservation(&fixture, &job, NnuecCaptureLimits::new(1, 1, 1024).unwrap());
    let lineage = capture.lineage();
    job.enable_nnuec_capture(capture.clone())
        .unwrap_or_else(|_| panic!("attach"));
    fixture.cancel.store(true, Ordering::Release);
    assert!(
        capture
            .bind_job(
                job.envelope(),
                CpuWorkerId(0),
                &Position::startpos(),
                fixture.clock,
                capture.submission.binding
            )
            .is_err()
    );
    capture
        .retire(
            job.envelope(),
            CpuWorkerId(0),
            NnuecCaptureEnding::NotStarted,
        )
        .unwrap();
    assert!(capture.retirement().unwrap().readouts().is_empty());
    assert!(!lineage.matches_activated_job(job.envelope(), CpuWorkerId(0)));
    assert!(!capture.retirement().unwrap().matches_lineage(&lineage));
    let stopped = worker.shutdown().unwrap();
    assert!(stopped.engine_recoverable);
    assert!(
        !fixture.bound.authority_is_live(),
        "known engine recovery does not restore a revoked Bound"
    );
}

#[test]
fn first_begin_cannot_reset_the_prepared_clock_origin_in_the_same_domain() {
    let fixture = fixture();
    let worker = worker(&fixture);
    let mut job = job(&fixture, &worker, 1, 1);
    let capture = reservation(&fixture, &job, NnuecCaptureLimits::new(1, 1, 1024).unwrap());
    job.enable_nnuec_capture(capture.clone())
        .unwrap_or_else(|_| panic!("attach"));
    let other_clock = InstantClock::new(
        fixture.clock.domain(),
        fixture.clock.origin() + Duration::from_millis(1),
    );
    assert!(
        capture
            .bind_job(
                job.envelope(),
                CpuWorkerId(0),
                &Position::startpos(),
                other_clock,
                capture.submission.binding,
            )
            .is_err()
    );
    assert!(
        capture
            .bind_job(
                job.envelope(),
                CpuWorkerId(0),
                &Position::startpos(),
                fixture.clock,
                capture.submission.binding,
            )
            .is_ok()
    );
    worker.shutdown().unwrap();
}

#[test]
fn definite_unqueued_rejection_retires_original_without_claiming_cpu_execution() {
    let fixture = fixture();
    let worker = worker(&fixture);
    let mut job = job(&fixture, &worker, 1, 1);
    let capture = reservation(&fixture, &job, NnuecCaptureLimits::new(1, 1, 1024).unwrap());
    let equal_metadata = reservation(&fixture, &job, NnuecCaptureLimits::new(1, 1, 1024).unwrap());
    let lineage = capture.lineage();
    job.enable_nnuec_capture(capture.clone())
        .unwrap_or_else(|_| panic!("attach"));
    let grant = grant(&job, 1, 1);
    worker.request_shutdown();
    let rejected = worker.begin_retaining(job, grant).unwrap_err();
    assert!(rejected.proof().is_verified_unqueued());
    let retirement = capture.retirement().expect("known unqueued retirement");
    assert_eq!(retirement.ending(), NnuecCaptureEnding::NotStarted);
    assert!(capture.matches_retirement(retirement));
    assert!(!equal_metadata.matches_retirement(retirement));
    assert!(!retirement.matches_lineage(&lineage));
    assert!(!lineage.matches_activated_job(retirement.job(), retirement.owner()));
    assert!(!lineage.activation_observed());
    assert_eq!(retirement.attempted(), 0);
    assert!(retirement.readouts().is_empty());
    let stopped = worker.shutdown().unwrap();
    assert!(stopped.events.is_empty());
}

#[test]
fn prepared_rejection_retains_the_same_job_until_actual_fence_attests_not_started() {
    let fixture = fixture();
    let worker = worker(&fixture);
    let mut original = job(&fixture, &worker, 1, 1);
    let capture = reservation(
        &fixture,
        &original,
        NnuecCaptureLimits::new(2, 1, 1 << 20).unwrap(),
    );
    let lineage = capture.lineage();
    original
        .enable_nnuec_capture(capture.clone())
        .unwrap_or_else(|_| panic!("capture attach"));
    let mailbox = Arc::clone(&original.mailbox);
    let valid = grant(&original, 1, 1);
    let mut foreign = valid;
    foreign.owner = CpuWorkerId(1);
    let (error, original) = match worker.reject_prepared_retaining(original, foreign) {
        Err(rejected) => rejected,
        Ok(_) => panic!("foreign scope rejection"),
    };
    assert_eq!(
        error,
        CpuWorkerError::Invalid("CPU prepared rejection scope mismatch")
    );
    assert!(Arc::ptr_eq(&mailbox, &original.mailbox));
    assert!(capture.retirement().is_none());
    let mut invalid = valid;
    invalid.node_credit = 0;
    let (error, original) = match worker.reject_prepared_retaining(original, invalid) {
        Err(rejected) => rejected,
        Ok(_) => panic!("invalid grant rejection"),
    };
    assert_eq!(
        error,
        CpuWorkerError::Invalid("CPU submission owner or grant mismatch")
    );
    assert!(Arc::ptr_eq(&mailbox, &original.mailbox));
    assert!(capture.retirement().is_none());
    let rejected = worker
        .reject_prepared_retaining(original, valid)
        .unwrap_or_else(|_| panic!("actual unqueued fence"));
    assert!(rejected.proof().is_verified_unqueued());
    let retirement = capture.retirement().expect("original capture retired");
    assert_eq!(retirement.ending(), NnuecCaptureEnding::NotStarted);
    assert_eq!(retirement.attempted(), 0);
    assert!(retirement.readouts().is_empty());
    assert!(!retirement.matches_lineage(&lineage));
    let (_, _, original) = rejected.into_parts();
    let original = original.expect("same rejected Job");
    assert!(Arc::ptr_eq(&mailbox, &original.mailbox));
    let (error, original) = match worker.reject_prepared_retaining(original, valid) {
        Err(rejected) => rejected,
        Ok(_) => panic!("same execution cannot issue another proof"),
    };
    assert_eq!(error, CpuWorkerError::Duplicate);
    assert!(Arc::ptr_eq(&mailbox, &original.mailbox));
    assert!(capture.error().unwrap().is_none());
    assert!(worker.shutdown().unwrap().thread_joined);
}

#[test]
fn an_enqueued_execution_cannot_attest_not_started_for_another_prepared_capture() {
    let fixture = fixture();
    let worker = worker(&fixture);
    let first = job(&fixture, &worker, 1, 1);
    let first_grant = grant(&first, 1, 4096);
    worker.begin(first, first_grant).unwrap();
    let completed = report(&worker);
    assert!(!completed.admission_rejected);
    let mut original = job(&fixture, &worker, 2, 1);
    let capture = reservation(
        &fixture,
        &original,
        NnuecCaptureLimits::new(2, 1, 1 << 20).unwrap(),
    );
    original
        .enable_nnuec_capture(capture.clone())
        .unwrap_or_else(|_| panic!("capture attach"));
    let mailbox = Arc::clone(&original.mailbox);
    let repeated_execution = grant(&original, first_grant.execution.sequence, 1);
    let (error, original) = match worker.reject_prepared_retaining(original, repeated_execution) {
        Err(rejected) => rejected,
        Ok(_) => panic!("actual queue already owns execution"),
    };
    assert_eq!(error, CpuWorkerError::Duplicate);
    assert!(Arc::ptr_eq(&mailbox, &original.mailbox));
    assert!(capture.retirement().is_none());
    let next = grant(&original, first_grant.execution.sequence + 1, 1);
    assert!(
        worker
            .reject_prepared_retaining(original, next)
            .unwrap_or_else(|_| panic!("fresh actual unqueued rejection"))
            .proof()
            .is_verified_unqueued()
    );
    assert_eq!(
        capture.retirement().unwrap().ending(),
        NnuecCaptureEnding::NotStarted
    );
    assert!(worker.shutdown().unwrap().thread_joined);
}
