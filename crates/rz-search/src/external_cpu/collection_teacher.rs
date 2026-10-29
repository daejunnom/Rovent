//! Collection 전용 CPU_T 물리 worker. 검색의 R/reciprocal 권한과 독립이다.
//!
//! 여기의 input scope는 원 입력을 찾는 metadata다. 실제 SubmittedInputSeal,
//! FrozenTeacherActor, 동일 RootAdmission은 상위 collector의 opaque binding이
//! 보유하고 private dispatch/completion과 대조한다. 이 worker는 teacher label,
//! Rules 결과, native node credit 또는 exact resume 권한을 발급하지 않는다.
//! 기존 profiled R worker의 mode/recipient guard는 이 경로에서 변경하지 않는다.

use super::process_scope::CollectionProcessPermit;
use super::profiled_worker::{ProfiledExternalOutcome, ProfiledExternalTiming};
use super::worker::{
    ExternalWorkerClock, ExternalWorkerStartup, external_evaluation_namespace,
    external_root_restriction_digest,
};
use super::*;
use rz_contracts::pals::{Role, SearchAuthority};
use rz_contracts::{Deadline, Digest as ContractDigest};
use rz_runtime::CompletionSignal;
use rz_runtime::adaptive::{
    ControlMailbox, ControlSnapshot, MailboxError, MailboxOccupancy, WorkerMailbox,
};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

pub const COLLECTION_TEACHER_DOMAIN: &str = "rovent-collection-teacher-job/1";

/// Collection owns this namespace. It is neither a CPU R Job nor a request ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TeacherJobId {
    pub collection_owner: u32,
    pub sequence: u64,
}

/// Read-only projections of the actual upper-layer custody, never custody itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TeacherInputScope {
    pub authority: SearchAuthority,
    pub input_sha256: ContractDigest,
    pub view_sha256: ContractDigest,
    pub source_sha256: ContractDigest,
    pub role: Role,
    pub rules_semantic: ContractDigest,
    pub input_lock: ContractDigest,
    pub root_admission: ContractDigest,
    pub root_budget: ContractDigest,
    pub frozen_actor: ContractDigest,
}

/// One same-root collection owner. Root replacement requires retiring this worker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TeacherWorkerRegistration {
    pub collection_owner: u32,
    pub authority: SearchAuthority,
    pub original_deadline: Deadline,
    pub input_lock: ContractDigest,
    pub root_admission: ContractDigest,
    pub root_budget: ContractDigest,
    pub frozen_actor: ContractDigest,
}

#[derive(Clone, Debug)]
pub struct TeacherJob {
    pub id: TeacherJobId,
    pub input_scope: TeacherInputScope,
    pub position: Position,
    pub root_moves: Option<Vec<BoardMove>>,
    pub constraint: ExternalGoConstraint,
    pub original_deadline: Deadline,
    pub soft_quantum: Duration,
    pub control: ControlMailbox<TeacherJobId>,
}

/// Issued only by successful bounded admission after this physical startup.
#[derive(Clone, Debug)]
pub struct TeacherDispatch {
    id: TeacherJobId,
    input_scope: TeacherInputScope,
    requested: ExternalGoConstraint,
    original_deadline: Deadline,
    profile: ExternalEngineProfile,
    process: ExternalProcessIdentity,
    evaluation_namespace: ContractDigest,
    root_restriction: ContractDigest,
    admission: Arc<()>,
}

impl TeacherDispatch {
    pub fn id(&self) -> TeacherJobId {
        self.id
    }
    pub fn input_scope(&self) -> TeacherInputScope {
        self.input_scope
    }
    pub fn constraint(&self) -> ExternalGoConstraint {
        self.requested
    }
    pub fn original_deadline(&self) -> Deadline {
        self.original_deadline
    }
    pub fn profile(&self) -> ExternalEngineProfile {
        self.profile
    }
    pub fn process_identity(&self) -> ExternalProcessIdentity {
        self.process
    }
    pub fn evaluation_namespace(&self) -> ContractDigest {
        self.evaluation_namespace
    }
    pub fn root_restriction(&self) -> ContractDigest {
        self.root_restriction
    }
}

/// A private terminal receipt remains mandatory even when the raw report is public.
#[derive(Clone, Debug)]
pub struct TeacherCompletion {
    dispatch: TeacherDispatch,
    process: Option<ExternalProcessIdentity>,
    result: Result<ProfiledExternalOutcome, CheckerError>,
    attempt: Option<CheckerAttempt>,
    controls: ControlSnapshot,
    timing: ProfiledExternalTiming,
    go_issue: ExternalGoIssue,
    issued_constraint: Option<ExternalGoConstraint>,
    issued_go: Option<String>,
}

impl TeacherCompletion {
    pub fn id(&self) -> TeacherJobId {
        self.dispatch.id
    }
    pub fn input_scope(&self) -> TeacherInputScope {
        self.dispatch.input_scope
    }
    pub fn result(&self) -> &Result<ProfiledExternalOutcome, CheckerError> {
        &self.result
    }
    pub fn attempt(&self) -> Option<&CheckerAttempt> {
        self.attempt.as_ref()
    }
    pub fn controls(&self) -> ControlSnapshot {
        self.controls
    }
    pub fn timing(&self) -> ProfiledExternalTiming {
        self.timing
    }
    pub fn go_issue(&self) -> ExternalGoIssue {
        self.go_issue
    }
    pub fn issued_constraint(&self) -> Option<ExternalGoConstraint> {
        self.issued_constraint
    }
    pub fn issued_go(&self) -> Option<&str> {
        self.issued_go.as_deref()
    }
    pub fn process_identity(&self) -> Option<ExternalProcessIdentity> {
        self.process
    }
    /// Matches the original private dispatch even if the terminal cannot
    /// provide a process observation. This establishes the terminal's
    /// submission identity only: it does not validate the process, outcome,
    /// Rules state, or authorization to use the result as a label.
    pub fn matches_dispatch_identity(&self, submitted: &TeacherDispatch) -> bool {
        Arc::ptr_eq(&self.dispatch.admission, &submitted.admission)
            && self.dispatch.id == submitted.id
            && self.dispatch.input_scope == submitted.input_scope
            && self.dispatch.requested == submitted.requested
            && self.dispatch.original_deadline == submitted.original_deadline
            && self.dispatch.profile == submitted.profile
            && self.dispatch.process == submitted.process
            && self.dispatch.evaluation_namespace == submitted.evaluation_namespace
            && self.dispatch.root_restriction == submitted.root_restriction
    }

    /// Strict submission match including the terminal's process evidence.
    pub fn matches_submission(&self, submitted: &TeacherDispatch) -> bool {
        self.matches_dispatch_identity(submitted) && self.process == Some(submitted.process)
    }
}

#[derive(Clone, Debug)]
pub struct TeacherProgress {
    pub id: TeacherJobId,
    pub opinion: ExternalPolicyOpinion,
    pub reported_nodes: Option<u64>,
    /// External reported nodes are never native credit.
    pub strict_native_nodes: Option<u64>,
    pub control_sequence: u64,
}

#[derive(Clone, Debug)]
pub struct TeacherDiagnostics {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}
impl TeacherDiagnostics {
    pub fn stdout(&self) -> &[u8] {
        &self.stdout
    }
    pub fn stderr(&self) -> &[u8] {
        &self.stderr
    }
}

struct TimedTeacherJob {
    original: TeacherJob,
    dispatch: TeacherDispatch,
    admitted_at: Instant,
}
type Mailbox = WorkerMailbox<TimedTeacherJob, TeacherCompletion, TeacherProgress>;

/// One physical owner, one admitted Job at a time, with reserved terminal space.
pub struct CollectionTeacherWorker {
    registration: TeacherWorkerRegistration,
    config: ExternalCpuConfig,
    profile: ExternalEngineProfile,
    clock: ExternalWorkerClock,
    cleanup_ceiling: Instant,
    mailbox: Mailbox,
    shutdown: Arc<AtomicBool>,
    shutdown_deadline: Arc<Mutex<Option<Instant>>>,
    startup: Arc<Mutex<Option<ExternalWorkerStartup>>>,
    diagnostics: Arc<Mutex<Option<TeacherDiagnostics>>>,
    process_after_shutdown: Arc<Mutex<Option<CheckerShutdown>>>,
    thread: Option<JoinHandle<Result<CheckerShutdown, CheckerError>>>,
    joined: Option<Result<CheckerShutdown, CheckerError>>,
    high_water: u64,
}

impl CollectionTeacherWorker {
    /// The owner has already selected a frozen Teacher profile and pinned files.
    /// No file reads, Rules replay, handshake or pipe waits run on this caller.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        config: ExternalCpuConfig,
        profile: ExternalEngineProfile,
        registration: TeacherWorkerRegistration,
        clock: ExternalWorkerClock,
        startup_deadline: Instant,
        cleanup_ceiling: Instant,
        launch_pin: ExternalLaunchPin,
    ) -> Result<Self, CheckerError> {
        config.validate()?;
        profile.validate_identity(
            ExternalEnginePurpose::Teacher,
            &config.identity,
            config.expected_uci_name.as_deref().unwrap_or_default(),
        )?;
        Self::spawn_checked(
            config,
            profile,
            registration,
            clock,
            startup_deadline,
            cleanup_ceiling,
            launch_pin,
            None,
            create_profiled_teacher,
        )
    }

    /// Registered collection sibling. The opaque permit belongs to the exact
    /// original root/actor and survives until the lower process owner closes.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_in_process_scope(
        config: ExternalCpuConfig,
        profile: ExternalEngineProfile,
        registration: TeacherWorkerRegistration,
        clock: ExternalWorkerClock,
        startup_deadline: Instant,
        cleanup_ceiling: Instant,
        launch_pin: ExternalLaunchPin,
        permit: CollectionProcessPermit,
    ) -> Result<Self, CheckerError> {
        config.validate()?;
        profile.validate_identity(
            ExternalEnginePurpose::Teacher,
            &config.identity,
            config.expected_uci_name.as_deref().unwrap_or_default(),
        )?;
        Self::spawn_checked(
            config,
            profile,
            registration,
            clock,
            startup_deadline,
            cleanup_ceiling,
            launch_pin,
            Some(permit),
            create_profiled_teacher,
        )
    }

    /// The public path can only supply the strict profiled constructor above.
    /// A private factory seam lets native test-ELF fixtures exercise identical
    /// physical ownership without claiming to be an official engine profile.
    #[allow(clippy::too_many_arguments)]
    fn spawn_checked(
        config: ExternalCpuConfig,
        profile: ExternalEngineProfile,
        registration: TeacherWorkerRegistration,
        clock: ExternalWorkerClock,
        startup_deadline: Instant,
        cleanup_ceiling: Instant,
        launch_pin: ExternalLaunchPin,
        process_permit: Option<CollectionProcessPermit>,
        create_checker: fn(
            ExternalCpuConfig,
            ExternalEngineProfile,
        ) -> Result<ExternalUciCpuChecker, CheckerError>,
    ) -> Result<Self, CheckerError> {
        config.validate()?;
        if cfg!(not(target_os = "linux")) {
            return Err(CheckerError::Unsupported(
                "collection CPU_T requires Linux native ELF",
            ));
        }
        let hard = registration_deadline(registration, clock)?;
        if registration.collection_owner == 0
            || registration.authority.epoch.0 == 0
            || registration.authority.epoch != clock.domain.0
            || registration.authority.game.0 == 0
            || registration.authority.root.0 == 0
            || registration.authority.implementation == ContractDigest([0; 32])
            || [
                registration.input_lock,
                registration.root_admission,
                registration.root_budget,
                registration.frozen_actor,
            ]
            .contains(&ContractDigest([0; 32]))
            || startup_deadline <= Instant::now()
            || startup_deadline > hard
            || cleanup_ceiling < hard
            || cleanup_ceiling
                > hard
                    .checked_add(config.shutdown_grace)
                    .ok_or(CheckerError::Invalid("teacher cleanup ceiling overflow"))?
        {
            return Err(CheckerError::Invalid(
                "teacher original registration/clock/deadline",
            ));
        }
        let launch_pin = match process_permit {
            Some(permit) => {
                permit.validate(registration)?;
                launch_pin.with_collection_permit(permit)
            }
            None => launch_pin,
        };
        let mailbox = Mailbox::new(1, 2, CompletionSignal::default())
            .map_err(|_| error("teacher_worker", "mailbox_capacity"))?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_deadline = Arc::new(Mutex::new(None));
        let startup = Arc::new(Mutex::new(None));
        let diagnostics = Arc::new(Mutex::new(None));
        let process_after_shutdown = Arc::new(Mutex::new(None));
        let (
            worker_config,
            worker_mailbox,
            worker_shutdown,
            worker_deadline,
            worker_startup,
            worker_diagnostics,
        ) = (
            config.clone(),
            mailbox.clone(),
            shutdown.clone(),
            shutdown_deadline.clone(),
            startup.clone(),
            diagnostics.clone(),
        );
        let worker_process = process_after_shutdown.clone();
        let thread = std::thread::Builder::new()
            .name(format!(
                "rz-collection-teacher-{}",
                registration.collection_owner
            ))
            .spawn(move || {
                run_worker(
                    worker_config,
                    profile,
                    registration,
                    clock,
                    startup_deadline,
                    cleanup_ceiling,
                    launch_pin,
                    worker_mailbox,
                    worker_shutdown,
                    worker_deadline,
                    worker_startup,
                    worker_diagnostics,
                    worker_process,
                    create_checker,
                )
            })
            .map_err(|_| error("teacher_worker", "thread_spawn"))?;
        Ok(Self {
            registration,
            config,
            profile,
            clock,
            cleanup_ceiling,
            mailbox,
            shutdown,
            shutdown_deadline,
            startup,
            diagnostics,
            process_after_shutdown,
            thread: Some(thread),
            joined: None,
            high_water: 0,
        })
    }
    pub fn registration(&self) -> TeacherWorkerRegistration {
        self.registration
    }
    pub fn profile(&self) -> ExternalEngineProfile {
        self.profile
    }
    pub fn configuration_matches(&self, config: &ExternalCpuConfig) -> bool {
        self.config == *config
    }
    pub fn signal(&self) -> CompletionSignal {
        self.mailbox.signal()
    }
    /// A control channel alone grants neither admission nor teacher authority.
    /// Returning it here keeps upper collectors independent of runtime details.
    pub fn control_mailbox(&self, id: TeacherJobId) -> ControlMailbox<TeacherJobId> {
        ControlMailbox::new(id, self.signal())
    }
    pub fn occupancy(&self) -> MailboxOccupancy {
        self.mailbox.occupancy()
    }
    pub fn startup_report(&self) -> Option<ExternalWorkerStartup> {
        self.startup
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }
    pub fn submit(&mut self, job: TeacherJob) -> Result<TeacherDispatch, CheckerError> {
        validate_metadata(
            &job,
            self.registration,
            self.clock,
            &self.config,
            self.profile,
        )?;
        if self.shutdown.load(Ordering::Acquire)
            || self.thread.as_ref().is_none_or(JoinHandle::is_finished)
            || job.id.sequence <= self.high_water
        {
            return Err(error("teacher_admission", "closed_duplicate_or_old_job"));
        }
        // Bound retained Position/line ownership to one Job, including its terminal.
        let occupancy = self.mailbox.occupancy();
        if occupancy.commands != 0
            || occupancy.reliable_reserved != 0
            || occupancy.reliable_queued != 0
        {
            return Err(error("teacher_admission", "prior_terminal_not_consumed"));
        }
        let startup = self
            .startup_report()
            .ok_or(error("teacher_admission", "startup_pending"))?;
        startup.result?;
        let process = startup
            .attempt
            .as_ref()
            .and_then(|attempt| attempt.external.as_ref())
            .and_then(|attempt| attempt.process.process_identity)
            .ok_or(error(
                "teacher_admission",
                "actual_process_identity_missing",
            ))?;
        let dispatch = TeacherDispatch {
            id: job.id,
            input_scope: job.input_scope,
            requested: job.constraint,
            original_deadline: job.original_deadline,
            profile: self.profile,
            process,
            evaluation_namespace: external_evaluation_namespace(&self.config.identity)?,
            root_restriction: external_root_restriction_digest(job.root_moves.as_deref()),
            admission: Arc::new(()),
        };
        self.mailbox
            .admit(
                TimedTeacherJob {
                    original: job,
                    dispatch: dispatch.clone(),
                    admitted_at: Instant::now(),
                },
                2,
            )
            .map_err(|rejected| match rejected.reason {
                MailboxError::QueueFull | MailboxError::ReliableFull => {
                    error("teacher_admission", "capacity")
                }
                _ => error("teacher_admission", "closed"),
            })?;
        self.high_water = dispatch.id.sequence;
        Ok(dispatch)
    }
    pub fn take_completion(&self) -> Option<TeacherCompletion> {
        let result = self.mailbox.take_reliable();
        if result.is_some() {
            self.mailbox.signal().notify();
        }
        result
    }
    pub fn take_progress(&self) -> Option<(u64, TeacherProgress)> {
        self.mailbox.take_progress()
    }
    /// The ordinary queue may be full. Shutdown/control still travels atomically.
    pub fn request_shutdown(&self, until: Instant) {
        self.mailbox.close();
        let mut deadline = self
            .shutdown_deadline
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let until = until.min(self.cleanup_ceiling);
        *deadline = Some(deadline.map_or(until, |old| old.min(until)));
        self.shutdown.store(true, Ordering::Release);
        self.mailbox.signal().notify();
    }
    pub fn try_join(&mut self) -> Option<Result<CheckerShutdown, CheckerError>> {
        if self.thread.as_ref().is_some_and(JoinHandle::is_finished) {
            let result = self
                .thread
                .take()
                .expect("checked teacher worker")
                .join()
                .unwrap_or_else(|_| Err(error("teacher_worker", "thread_panicked")));
            self.joined = Some(result.clone());
            Some(result)
        } else {
            None
        }
    }
    pub fn shutdown_snapshot(&self) -> Option<Result<CheckerShutdown, CheckerError>> {
        self.joined.clone()
    }
    /// An Err retirement can still leave actual quarantine/pipe/process evidence.
    /// Joining alone is not evidence that the process group and pipes closed.
    pub fn process_after_join(&self) -> Option<CheckerShutdown> {
        self.joined.as_ref()?;
        *self
            .process_after_shutdown
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }
    pub fn take_diagnostics_after_join(&mut self) -> Option<TeacherDiagnostics> {
        self.joined.as_ref()?;
        self.diagnostics
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
    }
}

fn create_profiled_teacher(
    config: ExternalCpuConfig,
    profile: ExternalEngineProfile,
) -> Result<ExternalUciCpuChecker, CheckerError> {
    ExternalUciCpuChecker::create_profiled(config, profile, ExternalEnginePurpose::Teacher)
}
impl Drop for CollectionTeacherWorker {
    fn drop(&mut self) {
        self.request_shutdown(Instant::now() + self.config.shutdown_grace);
    }
}

fn registration_deadline(
    registration: TeacherWorkerRegistration,
    clock: ExternalWorkerClock,
) -> Result<Instant, CheckerError> {
    if registration.original_deadline.clock != clock.domain {
        return Err(CheckerError::Invalid("teacher monotonic clock differs"));
    }
    clock
        .origin
        .checked_add(Duration::from_nanos(registration.original_deadline.at.0))
        .ok_or(CheckerError::Invalid("teacher deadline overflow"))
}

fn tighten_cleanup_fence(shared: &Mutex<Option<Instant>>, requested: Instant, ceiling: Instant) {
    let mut held = shared.lock().unwrap_or_else(|p| p.into_inner());
    let requested = requested.min(ceiling);
    *held = Some(held.map_or(requested, |old| old.min(requested)));
}

fn validate_metadata(
    job: &TeacherJob,
    registration: TeacherWorkerRegistration,
    clock: ExternalWorkerClock,
    config: &ExternalCpuConfig,
    profile: ExternalEngineProfile,
) -> Result<Instant, CheckerError> {
    profile.require_query(job.constraint, job.root_moves.is_some())?;
    let input = job.input_scope;
    let deadline = clock
        .origin
        .checked_add(Duration::from_nanos(job.original_deadline.at.0))
        .ok_or(CheckerError::Invalid("teacher job deadline overflow"))?;
    let remaining = job
        .soft_quantum
        .checked_add(config.stop_grace)
        .ok_or(CheckerError::Invalid("teacher quantum overflow"))?;
    if job.id.collection_owner != registration.collection_owner
        || job.id.sequence == 0
        || input.authority != registration.authority
        || input.input_lock != registration.input_lock
        || input.root_admission != registration.root_admission
        || input.root_budget != registration.root_budget
        || input.frozen_actor != registration.frozen_actor
        || !matches!(input.role, Role::Proposer | Role::Critic)
        || [
            input.input_sha256,
            input.view_sha256,
            input.source_sha256,
            input.rules_semantic,
        ]
        .contains(&ContractDigest([0; 32]))
        || job.original_deadline.clock != clock.domain
        || job.original_deadline.at > registration.original_deadline.at
        || *job.control.scope() != job.id
        || job.control.snapshot().terminal
        || job
            .constraint
            .requested_depth()
            .is_some_and(|depth| depth == 0 || depth > config.max_depth)
        || job
            .constraint
            .requested_nodes()
            .is_some_and(|nodes| nodes == 0 || nodes > EXTERNAL_MAX_REPORTED_NODE_REQUEST)
        || job.soft_quantum < EXTERNAL_MIN_QUANTUM
        || job.soft_quantum > Duration::from_secs(180)
        || job
            .root_moves
            .as_ref()
            .is_some_and(|roots| roots.is_empty() || roots.len() > 256 || roots.capacity() > 256)
        || config.stop_grace < EXTERNAL_STOP_DRAIN_RESERVE
        || config.max_task_wall_time < remaining
        || deadline.saturating_duration_since(Instant::now()) < remaining
    {
        return Err(CheckerError::Invalid(
            "teacher input/owner/control/resource/deadline seal",
        ));
    }
    registration_deadline(registration, clock)?;
    Ok(deadline)
}

fn validate_rules(job: &TeacherJob) -> Result<(), CheckerError> {
    if rz_position::contracts::rules_snapshot_semantic_digest(&job.position.snapshot())
        != job.input_scope.rules_semantic
    {
        return Err(CheckerError::Invalid(
            "teacher exact Rules state/history mismatch",
        ));
    }
    // The same Rules replay and all legal PV checks are also performed by the pipe owner.
    let legal = job.position.ordered_legal_moves();
    if let Some(roots) = &job.root_moves {
        if roots.iter().enumerate().any(|(at, movement)| {
            !legal.moves().contains(movement) || roots[..at].contains(movement)
        }) {
            return Err(CheckerError::Invalid(
                "teacher illegal/duplicate root restriction",
            ));
        }
    }
    Ok(())
}

struct JobControl {
    id: TeacherJobId,
    control: ControlMailbox<TeacherJobId>,
    mailbox: Mailbox,
    shutdown: Arc<AtomicBool>,
    shutdown_deadline: Arc<Mutex<Option<Instant>>>,
    cleanup_ceiling: Instant,
    timing: Arc<Mutex<ProfiledExternalTiming>>,
    progress_sequence: Arc<AtomicU64>,
}
impl ExternalControl for JobControl {
    fn observe(&self) -> ExternalControlObservation {
        let state = self.control.snapshot();
        let (sequence, action) = if state.terminal || self.shutdown.load(Ordering::Acquire) {
            (0, ExternalControlAction::Revoked)
        } else if state.is_cancelled() {
            (state.cancel_sequence, ExternalControlAction::Cancel)
        } else if let Some(sequence) = state.pending_yield() {
            (sequence, ExternalControlAction::Yield)
        } else {
            (0, ExternalControlAction::Continue)
        };
        ExternalControlObservation { sequence, action }
    }
    fn physical_fence_deadline(&self) -> Option<Instant> {
        Some(
            self.shutdown_deadline
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .map_or(self.cleanup_ceiling, |until| {
                    until.min(self.cleanup_ceiling)
                }),
        )
    }
    fn observe_owner_phase(&self, phase: ExternalOwnerPhase) {
        let mut timing = self.timing.lock().unwrap_or_else(|p| p.into_inner());
        match phase {
            ExternalOwnerPhase::GoIssued {
                at,
                quantum_deadline,
            } => {
                timing.go_issued_at.get_or_insert(at);
                timing.quantum_deadline.get_or_insert(quantum_deadline);
            }
            ExternalOwnerPhase::StopSent { at } => {
                timing.stop_sent_at.get_or_insert(at);
            }
            ExternalOwnerPhase::BestMoveDrained { at } => {
                timing.bestmove_drained_at.get_or_insert(at);
            }
            ExternalOwnerPhase::ReadyBarrierCompleted { at } => {
                timing.ready_completed_at.get_or_insert(at);
            }
        }
    }
    fn publish_progress(&self, opinion: &ExternalPolicyOpinion, reported_nodes: Option<u64>) {
        let state = self.control.snapshot();
        let Ok(previous) =
            self.progress_sequence
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                    old.checked_add(1)
                })
        else {
            return;
        };
        if self
            .mailbox
            .publish_progress(
                previous + 1,
                TeacherProgress {
                    id: self.id,
                    opinion: opinion.clone(),
                    reported_nodes,
                    strict_native_nodes: None,
                    control_sequence: state.pending_publish().unwrap_or(0),
                },
            )
            .is_ok()
        {
            if let Some(sequence) = state.pending_publish() {
                let _ = self.control.acknowledge_publish_progress(sequence);
            }
        }
    }
}

fn coherent_policy(
    report: &ExternalCheckerReport,
    opinions: &[ExternalPolicyOpinion],
    multipv: u16,
) -> bool {
    report.completion == ExternalCompletion::BestMove
        && opinions.len() == usize::from(multipv)
        && opinions.iter().enumerate().all(|(at, opinion)| {
            opinion.rank == at as u16 + 1
                && Some(opinion.depth) == opinions.first().map(|first| first.depth)
                && opinion.bound == ExternalBound::ExactReported
                && !opinions[..at]
                    .iter()
                    .any(|prior| prior.pv.first() == opinion.pv.first())
        })
        && opinions
            .first()
            .and_then(|opinion| opinion.pv.first())
            .copied()
            == report.best_move
}

#[allow(clippy::too_many_arguments)]
fn run_worker(
    config: ExternalCpuConfig,
    profile: ExternalEngineProfile,
    registration: TeacherWorkerRegistration,
    clock: ExternalWorkerClock,
    startup_deadline: Instant,
    cleanup_ceiling: Instant,
    launch_pin: ExternalLaunchPin,
    mailbox: Mailbox,
    shutdown: Arc<AtomicBool>,
    shutdown_deadline: Arc<Mutex<Option<Instant>>>,
    startup_report: Arc<Mutex<Option<ExternalWorkerStartup>>>,
    diagnostics: Arc<Mutex<Option<TeacherDiagnostics>>>,
    process_after_shutdown: Arc<Mutex<Option<CheckerShutdown>>>,
    create_checker: fn(
        ExternalCpuConfig,
        ExternalEngineProfile,
    ) -> Result<ExternalUciCpuChecker, CheckerError>,
) -> Result<CheckerShutdown, CheckerError> {
    let mut checker = create_checker(config.clone(), profile)?;
    checker.launch_pin = Some(launch_pin);
    let read_fence = || {
        Some(
            shutdown_deadline
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .map_or(cleanup_ceiling, |until| until.min(cleanup_ceiling)),
        )
    };
    // The lower owner also clamps to handshake_timeout. Register that exact
    // conservative fence here so a failed short handshake cannot reopen the
    // caller's longer startup window during retirement.
    let effective_startup_deadline = bounded_deadline(
        Instant::now(),
        config.handshake_timeout,
        Some(startup_deadline),
    )?;
    let startup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        checker.start_with_cleanup_fence(effective_startup_deadline, &shutdown, Some(&read_fence))
    }))
    .unwrap_or_else(|_| {
        checker.failed = true;
        Err(error("teacher_startup", "panicked"))
    });
    *startup_report.lock().unwrap_or_else(|p| p.into_inner()) = Some(ExternalWorkerStartup {
        result: startup.clone().and_then(|()| {
            checker
                .startup_uci()
                .ok_or(error("teacher_startup", "uci_identity_missing"))
        }),
        attempt: checker.last_attempt().cloned(),
    });
    if startup.is_err() {
        // A failed startup may already have unknown cleanup at this fence.
        // Never reopen it with the later whole-root cleanup ceiling.
        tighten_cleanup_fence(
            &shutdown_deadline,
            effective_startup_deadline,
            cleanup_ceiling,
        );
        shutdown.store(true, Ordering::Release);
        mailbox.close();
    }
    mailbox.signal().notify();
    let hard_deadline = registration_deadline(registration, clock)?;
    let progress_sequence = Arc::new(AtomicU64::new(0));
    loop {
        let observed = mailbox.signal().version();
        if Instant::now() >= hard_deadline {
            shutdown.store(true, Ordering::Release);
            mailbox.close();
        }
        if let Some(mut admitted) = mailbox.take_command() {
            let mut supervisor = admitted
                .reliable
                .split_off(1)
                .map_err(|_| error("teacher_worker", "terminal_permit_split"))?;
            let TimedTeacherJob {
                original: job,
                dispatch,
                admitted_at,
            } = admitted.command;
            let timing = Arc::new(Mutex::new(ProfiledExternalTiming {
                admitted_at: Some(admitted_at),
                run_started_at: Some(Instant::now()),
                ..Default::default()
            }));
            checker.reset_attempt();
            checker.go_issue = ExternalGoIssue::NotIssued;
            checker.last_issued_constraint = None;
            checker.last_issued_go = None;
            let mut issued_task_fence = None;
            let operation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || -> Result<ProfiledExternalOutcome, CheckerError> {
                    if shutdown.load(Ordering::Acquire) {
                        return Err(error("teacher_admission", "revoked_before_go"));
                    }
                    startup.clone()?;
                    let deadline = validate_metadata(&job, registration, clock, &config, profile)?;
                    validate_rules(&job)?;
                    let multipv = config
                        .identity
                        .options
                        .get("MultiPV")
                        .and_then(|v| v.parse::<u16>().ok())
                        .ok_or(CheckerError::Invalid("teacher MultiPV declaration missing"))?;
                    let candidate_count = job.root_moves.as_ref().map_or_else(
                        || job.position.ordered_legal_moves().moves().len(),
                        Vec::len,
                    );
                    if !(1..=4).contains(&multipv) || usize::from(multipv) > candidate_count {
                        return Err(CheckerError::Invalid(
                            "teacher MultiPV exceeds legal candidates",
                        ));
                    }
                    let control = JobControl {
                        id: job.id,
                        control: job.control.clone(),
                        mailbox: mailbox.clone(),
                        shutdown: shutdown.clone(),
                        shutdown_deadline: shutdown_deadline.clone(),
                        cleanup_ceiling,
                        timing: timing.clone(),
                        progress_sequence: progress_sequence.clone(),
                    };
                    checker.cooperative_opinions.clear();
                    checker.stop_control = None;
                    let limits = CpuLimits {
                        max_depth: job.constraint.requested_depth().unwrap_or(0),
                        max_nodes: job.constraint.requested_nodes().unwrap_or(0),
                        deadline: Some(deadline),
                    };
                    // Preserve Nodes/Depth/MoveTime exactly. This never supplies a fictitious Combined cap.
                    let policy = ExternalSlicePolicy {
                        quantum: job.soft_quantum,
                        stop_drain_reserve: config.stop_grace,
                        requested_reported_nodes: job.constraint.requested_nodes().unwrap_or(0),
                        multipv,
                    };
                    // Record a conservative actual task fence before entering
                    // the lower owner. A failed owner cannot gain a new grace
                    // period beyond its original finite task wall.
                    issued_task_fence = Some(
                        Instant::now()
                            .checked_add(config.max_task_wall_time)
                            .ok_or(CheckerError::Invalid("teacher task fence overflow"))?
                            .min(deadline),
                    );
                    let CheckerReport::ExternalUci(raw) = checker.run_with_control(
                        &job.position,
                        job.root_moves.as_deref(),
                        limits,
                        &shutdown,
                        Some((&control, policy)),
                        job.constraint,
                        false,
                    )?
                    else {
                        return Err(error("teacher_analysis", "foreign_report_type"));
                    };
                    checker
                        .cooperative_opinions
                        .sort_by_key(|opinion| opinion.rank);
                    let opinions = checker.cooperative_opinions.clone();
                    let known = coherent_policy(&raw, &opinions, multipv);
                    Ok(ProfiledExternalOutcome {
                        report: ProfiledExternalReport {
                            profile,
                            purpose: ExternalEnginePurpose::Teacher,
                            requested: job.constraint,
                            issued: checker.last_issued_constraint.ok_or(CheckerError::Invalid(
                                "teacher issued constraint missing",
                            ))?,
                            issued_go: checker
                                .last_issued_go
                                .clone()
                                .ok_or(CheckerError::Invalid("teacher issued go missing"))?,
                            reported_node_overshoot: job.constraint.requested_nodes().is_some_and(
                                |limit| raw.work.nodes.is_some_and(|nodes| nodes > limit),
                            ),
                            raw,
                            strict_native_nodes: None,
                            option_application_observed: None,
                            readiness_barrier_completed: true,
                        },
                        opinions,
                        requested_multipv: multipv,
                        conditional_policy_target_known: known,
                        stop_control: checker.stop_control,
                    })
                },
            ));
            let (result, panicked) = match operation {
                Ok(result) => (result, false),
                Err(_) => {
                    checker.failed = true;
                    (Err(error("teacher_worker", "task_panicked")), true)
                }
            };
            if let Ok(outcome) = &result {
                if let Some(control) = outcome.stop_control {
                    match control.action {
                        ExternalControlAction::Cancel => {
                            let _ = job.control.acknowledge_cancel(control.sequence);
                        }
                        ExternalControlAction::Yield => {
                            let _ = job.control.acknowledge_yield(control.sequence);
                        }
                        ExternalControlAction::Continue | ExternalControlAction::Revoked => {}
                    }
                }
            }
            let controls = job.control.seal_terminal();
            let mut final_timing = *timing.lock().unwrap_or_else(|p| p.into_inner());
            final_timing.returned_at = Some(Instant::now());
            let completion = TeacherCompletion {
                dispatch,
                process: checker
                    .owner
                    .as_ref()
                    .and_then(|owner| owner.evidence().process_identity),
                result,
                attempt: checker.last_attempt().cloned(),
                controls,
                timing: final_timing,
                go_issue: checker.go_issue,
                issued_constraint: checker.last_issued_constraint,
                issued_go: checker.last_issued_go.clone(),
            };
            let publication = if panicked {
                supervisor.publish(completion)
            } else {
                admitted.reliable.publish(completion)
            };
            if publication.is_err() {
                return Err(error(
                    "teacher_worker",
                    "reserved_terminal_publication_failed",
                ));
            }
            if panicked || checker.failed {
                let original = issued_task_fence.unwrap_or_else(|| {
                    clock
                        .origin
                        .checked_add(Duration::from_nanos(job.original_deadline.at.0))
                        .unwrap_or(cleanup_ceiling)
                });
                let prompt_cleanup = Instant::now()
                    .checked_add(config.shutdown_grace)
                    .unwrap_or(cleanup_ceiling);
                tighten_cleanup_fence(
                    &shutdown_deadline,
                    original.min(prompt_cleanup),
                    cleanup_ceiling,
                );
                shutdown.store(true, Ordering::Release);
                mailbox.close();
            }
            continue;
        }
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        mailbox.wait_changed(
            observed,
            hard_deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(20)),
        );
    }
    let until = shutdown_deadline
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .map_or(cleanup_ceiling, |until| until.min(cleanup_ceiling));
    let result = checker.shutdown(until);
    *process_after_shutdown
        .lock()
        .unwrap_or_else(|p| p.into_inner()) = checker
        .owner
        .as_ref()
        .map(|owner| owner.evidence())
        .or_else(|| {
            checker
                .last_attempt()
                .and_then(|attempt| attempt.external.as_ref().map(|external| external.process))
        });
    let (stdout, stderr) = checker.diagnostic_output();
    *diagnostics.lock().unwrap_or_else(|p| p.into_inner()) = Some(TeacherDiagnostics {
        stdout: stdout.to_vec(),
        stderr: stderr.to_vec(),
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::{ClockDomain, GameGeneration, MonotonicTick, ProcessEpoch, RootGeneration};

    fn fixture() -> (
        TeacherJob,
        TeacherWorkerRegistration,
        ExternalCpuConfig,
        ExternalWorkerClock,
    ) {
        let clock = ExternalWorkerClock {
            domain: ClockDomain(ProcessEpoch(1)),
            origin: Instant::now(),
        };
        let authority = SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: ContractDigest([1; 32]),
        };
        let deadline = Deadline {
            clock: clock.domain,
            at: MonotonicTick(10_000_000_000),
        };
        let registration = TeacherWorkerRegistration {
            collection_owner: 7,
            authority,
            original_deadline: deadline,
            input_lock: ContractDigest([2; 32]),
            root_admission: ContractDigest([3; 32]),
            root_budget: ContractDigest([4; 32]),
            frozen_actor: ContractDigest([5; 32]),
        };
        let position = Position::startpos();
        let input_scope = TeacherInputScope {
            authority,
            input_sha256: ContractDigest([6; 32]),
            view_sha256: ContractDigest([7; 32]),
            source_sha256: ContractDigest([8; 32]),
            role: Role::Proposer,
            rules_semantic: rz_position::contracts::rules_snapshot_semantic_digest(
                &position.snapshot(),
            ),
            input_lock: registration.input_lock,
            root_admission: registration.root_admission,
            root_budget: registration.root_budget,
            frozen_actor: registration.frozen_actor,
        };
        let id = TeacherJobId {
            collection_owner: 7,
            sequence: 1,
        };
        let job = TeacherJob {
            id,
            input_scope,
            position,
            root_moves: None,
            constraint: ExternalGoConstraint::Nodes {
                reported_nodes: 4096,
            },
            original_deadline: deadline,
            soft_quantum: EXTERNAL_MIN_QUANTUM,
            control: ControlMailbox::new(id, CompletionSignal::default()),
        };
        let profile = ExternalEngineProfile::Reckless090EmbeddedNnue;
        let config = ExternalCpuConfig {
            program: std::env::temp_dir().join("rz-absent-teacher-fixture"),
            arguments: Vec::new(),
            working_directory: std::env::temp_dir(),
            identity: ExternalCheckerIdentity {
                adapter_semantics: EXTERNAL_UCI_SCORE_SEMANTICS.into(),
                binary_sha256: "a".repeat(64),
                launch_arguments_sha256: arguments_sha256(&[]),
                declared_name: profile.expected_uci_name().into(),
                declared_version: "0.9.0".into(),
                declared_source: format!(
                    "{}/tree/{}",
                    profile.source_repository(),
                    profile.source_commit()
                ),
                declared_license: profile.license().into(),
                options: profile
                    .options(
                        ExternalEnginePurpose::Teacher,
                        ExternalEngineResources::default(),
                    )
                    .unwrap(),
                assets: Vec::new(),
                model_metadata: ExternalModelMetadata {
                    weights_sha256: None,
                    training: ExternalTrainingKnowledge::Unknown,
                    declared_rights: None,
                    precision: None,
                },
            },
            expected_uci_name: Some(profile.expected_uci_name().into()),
            max_depth: 64,
            max_prefix_plies: 4096,
            handshake_timeout: Duration::from_secs(1),
            max_task_wall_time: Duration::from_secs(2),
            stop_grace: EXTERNAL_STOP_DRAIN_RESERVE,
            shutdown_grace: Duration::from_millis(500),
            max_output_bytes: 65536,
            max_line_bytes: 4096,
        };
        (job, registration, config, clock)
    }

    #[test]
    fn collection_pc_input_has_no_r_recipient_or_student_mode_requirement() {
        let (mut job, registration, config, clock) = fixture();
        let profile = ExternalEngineProfile::Reckless090EmbeddedNnue;
        assert!(validate_metadata(&job, registration, clock, &config, profile).is_ok());
        job.input_scope.role = Role::Critic;
        assert!(validate_metadata(&job, registration, clock, &config, profile).is_ok());
        job.input_scope.role = Role::Validator;
        assert!(validate_metadata(&job, registration, clock, &config, profile).is_err());
    }

    #[test]
    fn original_owner_actor_root_context_and_deadline_cannot_be_rebound() {
        let (job, registration, config, clock) = fixture();
        let profile = ExternalEngineProfile::Reckless090EmbeddedNnue;
        for field in 0..8 {
            let mut changed = job.clone();
            match field {
                0 => changed.id.collection_owner += 1,
                1 => changed.input_scope.authority.root.0 += 1,
                2 => changed.input_scope.frozen_actor = ContractDigest([22; 32]),
                3 => changed.input_scope.root_admission = ContractDigest([23; 32]),
                4 => changed.input_scope.input_lock = ContractDigest([24; 32]),
                5 => changed.input_scope.root_budget = ContractDigest([25; 32]),
                6 => changed.original_deadline.at.0 += 1,
                7 => changed.original_deadline.clock = ClockDomain(ProcessEpoch(2)),
                _ => unreachable!(),
            }
            assert!(
                validate_metadata(&changed, registration, clock, &config, profile).is_err(),
                "field {field}"
            );
        }
    }

    #[test]
    fn reckless_single_limit_is_preserved_without_hidden_combined_limits() {
        let (mut job, registration, config, clock) = fixture();
        let profile = ExternalEngineProfile::Reckless090EmbeddedNnue;
        assert!(validate_metadata(&job, registration, clock, &config, profile).is_ok());
        job.constraint = ExternalGoConstraint::Depth { depth: 4 };
        assert!(validate_metadata(&job, registration, clock, &config, profile).is_ok());
        job.constraint = ExternalGoConstraint::Combined {
            depth: 4,
            reported_nodes: 4096,
        };
        assert!(validate_metadata(&job, registration, clock, &config, profile).is_err());
        job.constraint = ExternalGoConstraint::Nodes {
            reported_nodes: 4096,
        };
        job.root_moves = Some(vec![BoardMove::from_uci("e2e4").unwrap()]);
        assert!(validate_metadata(&job, registration, clock, &config, profile).is_err());
    }

    #[test]
    fn worker_rules_validation_rejects_other_state_and_duplicate_or_illegal_roots() {
        let (job, _, _, _) = fixture();
        assert!(validate_rules(&job).is_ok());
        let mut other = job.clone();
        other
            .position
            .make_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        assert!(validate_rules(&other).is_err());
        let mut same_board_other_history = job.clone();
        for movement in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            same_board_other_history
                .position
                .make_move(BoardMove::from_uci(movement).unwrap())
                .unwrap();
        }
        assert!(validate_rules(&same_board_other_history).is_err());
        for roots in [
            vec![BoardMove::from_uci("e7e5").unwrap()],
            vec![BoardMove::from_uci("e2e4").unwrap(); 2],
        ] {
            let mut changed = job.clone();
            changed.root_moves = Some(roots);
            assert!(validate_rules(&changed).is_err());
        }
    }

    #[test]
    fn full_queue_preserves_atomic_cancel_and_reserved_terminal_after_close() {
        let (job, _, config, _) = fixture();
        let mailbox = Mailbox::new(1, 2, CompletionSignal::default()).unwrap();
        let dispatch = TeacherDispatch {
            id: job.id,
            input_scope: job.input_scope,
            requested: job.constraint,
            original_deadline: job.original_deadline,
            profile: ExternalEngineProfile::Reckless090EmbeddedNnue,
            process: ExternalProcessIdentity::default(),
            evaluation_namespace: external_evaluation_namespace(&config.identity).unwrap(),
            root_restriction: external_root_restriction_digest(None),
            admission: Arc::new(()),
        };
        let controls = job.control.clone();
        assert!(
            mailbox
                .admit(
                    TimedTeacherJob {
                        original: job.clone(),
                        dispatch: dispatch.clone(),
                        admitted_at: Instant::now()
                    },
                    2
                )
                .is_ok()
        );
        assert!(
            mailbox
                .admit(
                    TimedTeacherJob {
                        original: job,
                        dispatch,
                        admitted_at: Instant::now()
                    },
                    2
                )
                .is_err()
        );
        controls.request_cancel(1).unwrap();
        mailbox.close();
        assert!(controls.snapshot().is_cancelled());
        let mut command = mailbox.take_command().unwrap();
        let mut supervisor = command.reliable.split_off(1).unwrap();
        let result = TeacherCompletion {
            dispatch: command.command.dispatch,
            process: None,
            result: Err(error("teacher_admission", "canceled_before_go")),
            attempt: None,
            controls: controls.seal_terminal(),
            timing: ProfiledExternalTiming::default(),
            go_issue: ExternalGoIssue::NotIssued,
            issued_constraint: None,
            issued_go: None,
        };
        assert!(supervisor.publish(result).is_ok());
        let retained = mailbox.take_reliable().unwrap();
        assert!(retained.controls().is_cancelled());
        assert_eq!(retained.go_issue(), ExternalGoIssue::NotIssued);
        assert!(retained.attempt().is_none());
        assert!(mailbox.take_reliable().is_none());
    }

    #[test]
    fn early_failure_never_reopens_startup_or_task_cleanup_fence() {
        let now = Instant::now();
        let root_ceiling = now + Duration::from_secs(3600);
        let startup_fence = now + Duration::from_secs(1);
        let shared = Mutex::new(None);
        tighten_cleanup_fence(&shared, startup_fence, root_ceiling);
        tighten_cleanup_fence(&shared, root_ceiling, root_ceiling);
        assert_eq!(*shared.lock().unwrap(), Some(startup_fence));
        let original_task_fence = now + Duration::from_millis(20);
        tighten_cleanup_fence(&shared, original_task_fence, root_ceiling);
        tighten_cleanup_fence(&shared, now + Duration::from_secs(3), root_ceiling);
        assert_eq!(*shared.lock().unwrap(), Some(original_task_fence));
        let long_caller_startup = now + Duration::from_secs(60);
        let effective =
            bounded_deadline(now, Duration::from_secs(1), Some(long_caller_startup)).unwrap();
        assert_eq!(effective, startup_fence);
        let actual_lower = bounded_deadline(
            now + Duration::from_millis(1),
            Duration::from_secs(1),
            Some(effective),
        )
        .unwrap();
        assert_eq!(effective, actual_lower);
        let handshake = Mutex::new(None);
        tighten_cleanup_fence(&handshake, effective, root_ceiling);
        tighten_cleanup_fence(&handshake, long_caller_startup, root_ceiling);
        assert_eq!(*handshake.lock().unwrap(), Some(effective));
    }

    /// Native test-ELF seam only: a fixture is never declared as an official
    /// Stockfish/Reckless binary. Public spawn still validates the real profile
    /// and empty launch arguments before it can reach spawn_checked.
    #[cfg(target_os = "linux")]
    fn create_fixture_checker(
        config: ExternalCpuConfig,
        profile: ExternalEngineProfile,
    ) -> Result<ExternalUciCpuChecker, CheckerError> {
        let mut checker = ExternalUciCpuChecker::create(config)?;
        checker.engine_profile = Some((profile, ExternalEnginePurpose::Teacher));
        Ok(checker)
    }

    #[cfg(target_os = "linux")]
    fn native_fixture(mode: &str) -> (CollectionTeacherWorker, TeacherJob, Instant) {
        // Expected identity preparation reads this large test ELF once. Do
        // that before issuing the fixture's original root/startup clocks;
        // the real process owner still independently rehashes the pinned ELF
        // inside the unchanged common fixture handshake window below.
        let mut config = super::super::tests::fixture_config(mode);
        config.identity.options.insert("MultiPV".into(), "1".into());
        config.stop_grace = EXTERNAL_STOP_DRAIN_RESERVE;
        config.shutdown_grace = Duration::from_secs(1);
        let pin = ExternalLaunchPin::from_owned_descriptors(
            File::open(&config.program).unwrap(),
            File::open(&config.working_directory).unwrap(),
        )
        .unwrap();
        let (mut job, mut registration, _, clock) = fixture();
        // This fixture runs startup followed by two finite go/stop/ready
        // phases. Startup time must not consume either original task's wall
        // allocation, and cleanup remains outside the original root limit.
        let startup = clock.origin + config.handshake_timeout;
        let root_wall = config
            .handshake_timeout
            .checked_add(config.max_task_wall_time.checked_mul(2).unwrap())
            .unwrap();
        job.original_deadline.at = MonotonicTick(u64::try_from(root_wall.as_nanos()).unwrap());
        registration.original_deadline = job.original_deadline;
        let hard = registration_deadline(registration, clock).unwrap();
        let cleanup = hard + config.shutdown_grace;
        let profile = ExternalEngineProfile::Reckless090EmbeddedNnue;
        // This actual test executable requires --exact arguments and its own
        // MIT fixture identity. It cannot pass as an official engine artifact.
        assert!(
            CollectionTeacherWorker::spawn(
                config.clone(),
                profile,
                registration,
                clock,
                startup,
                cleanup,
                pin.clone()
            )
            .is_err()
        );
        let worker = CollectionTeacherWorker::spawn_checked(
            config,
            profile,
            registration,
            clock,
            startup,
            cleanup,
            pin,
            None,
            create_fixture_checker,
        )
        .unwrap();
        wait_startup(&worker, startup);
        job.soft_quantum = Duration::from_secs(1);
        job.control = ControlMailbox::new(job.id, worker.signal());
        (worker, job, hard)
    }

    #[cfg(target_os = "linux")]
    fn wait_startup(worker: &CollectionTeacherWorker, until: Instant) {
        loop {
            let observed = worker.signal().version();
            if let Some(startup) = worker.startup_report() {
                let observed_uci = startup.result.unwrap();
                assert_eq!(observed_uci.name, "RZ external checker fixture");
                assert!(
                    startup
                        .attempt
                        .unwrap()
                        .external
                        .unwrap()
                        .process
                        .process_identity
                        .is_some()
                );
                return;
            }
            assert!(
                Instant::now() < until,
                "teacher fixture startup did not finish"
            );
            worker
                .signal()
                .wait_changed(observed, Duration::from_millis(20));
        }
    }

    #[cfg(target_os = "linux")]
    fn wait_progress(worker: &CollectionTeacherWorker, id: TeacherJobId, until: Instant) {
        loop {
            let observed = worker.signal().version();
            if let Some((_, progress)) = worker.take_progress() {
                assert_eq!(progress.id, id);
                assert_eq!(progress.strict_native_nodes, None);
                assert!(!progress.opinion.pv.is_empty());
                return;
            }
            assert!(
                Instant::now() < until,
                "teacher fixture produced no actual go progress"
            );
            worker
                .signal()
                .wait_changed(observed, Duration::from_millis(20));
        }
    }

    #[cfg(target_os = "linux")]
    fn wait_completion(worker: &CollectionTeacherWorker, until: Instant) -> TeacherCompletion {
        loop {
            let observed = worker.signal().version();
            if let Some(completion) = worker.take_completion() {
                return completion;
            }
            assert!(
                Instant::now() < until,
                "teacher fixture terminal did not finish"
            );
            worker
                .signal()
                .wait_changed(observed, Duration::from_millis(20));
        }
    }

    #[cfg(target_os = "linux")]
    fn close_fixture(worker: &mut CollectionTeacherWorker, until: Instant) -> CheckerShutdown {
        worker.request_shutdown(until);
        loop {
            let observed = worker.signal().version();
            if let Some(closed) = worker.try_join() {
                let closed = closed.unwrap();
                assert!(closed.cleanup_complete && closed.exit_observed && closed.quit_sent);
                assert!(closed.stdout_drained && closed.stderr_drained);
                assert!(!closed.quarantined && !closed.ownership_lost);
                assert_eq!(worker.process_after_join(), Some(closed));
                return closed;
            }
            assert!(
                Instant::now() < until,
                "teacher fixture process retirement did not finish"
            );
            worker
                .signal()
                .wait_changed(observed, Duration::from_millis(20));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn registered_scope_keeps_two_teacher_processes_ready_and_fully_retires_both() {
        use super::super::process_scope::CollectionProcessScope;

        let _guard = super::super::tests::FIXTURE_LOCK.lock().unwrap();
        // Hash the fixture ELF before issuing the shared original clock. The
        // actual lower owners independently revalidate it during startup.
        let mut config = super::super::tests::fixture_config("fake_uci_normal");
        config.identity.options.insert("MultiPV".into(), "1".into());
        config.stop_grace = EXTERNAL_STOP_DRAIN_RESERVE;
        config.shutdown_grace = Duration::from_secs(1);
        let pin = ExternalLaunchPin::from_owned_descriptors(
            File::open(&config.program).unwrap(),
            File::open(&config.working_directory).unwrap(),
        )
        .unwrap();
        let (_, mut first_registration, _, clock) = fixture();
        let root_wall = config
            .handshake_timeout
            .checked_add(config.max_task_wall_time.checked_mul(2).unwrap())
            .unwrap();
        first_registration.original_deadline.at =
            MonotonicTick(u64::try_from(root_wall.as_nanos()).unwrap());
        let mut second_registration = first_registration;
        second_registration.collection_owner += 1;
        second_registration.frozen_actor = ContractDigest([55; 32]);
        let startup = clock.origin + config.handshake_timeout;
        let hard = registration_deadline(first_registration, clock).unwrap();
        let cleanup = hard + config.shutdown_grace;
        let manifest = ContractDigest([91; 32]);
        let scope =
            CollectionProcessScope::new(manifest, vec![first_registration, second_registration])
                .unwrap();
        let first_permit = scope.permit(manifest, first_registration).unwrap();
        let second_permit = scope.permit(manifest, second_registration).unwrap();
        let profile = ExternalEngineProfile::Reckless090EmbeddedNnue;
        // A valid seat for another member must fail before a worker is started.
        assert!(
            CollectionTeacherWorker::spawn_checked(
                config.clone(),
                profile,
                second_registration,
                clock,
                startup,
                cleanup,
                pin.clone(),
                Some(first_permit.clone()),
                create_fixture_checker,
            )
            .is_err()
        );
        let mut first = CollectionTeacherWorker::spawn_checked(
            config.clone(),
            profile,
            first_registration,
            clock,
            startup,
            cleanup,
            pin.clone(),
            Some(first_permit),
            create_fixture_checker,
        )
        .unwrap();
        let mut second = CollectionTeacherWorker::spawn_checked(
            config.clone(),
            profile,
            second_registration,
            clock,
            startup,
            cleanup,
            pin,
            Some(second_permit),
            create_fixture_checker,
        )
        .unwrap();
        wait_startup(&first, startup);
        wait_startup(&second, startup);
        let first_process = first
            .startup_report()
            .unwrap()
            .attempt
            .unwrap()
            .external
            .unwrap()
            .process
            .process_identity
            .unwrap();
        let second_process = second
            .startup_report()
            .unwrap()
            .attempt
            .unwrap()
            .external
            .unwrap()
            .process
            .process_identity
            .unwrap();
        assert_ne!(first_process, second_process);

        let mut legacy = ExternalUciCpuChecker::create(config.clone()).unwrap();
        assert!(matches!(
            legacy.start(startup, &AtomicBool::new(false)),
            Err(CheckerError::External {
                stage: "spawn",
                code: "process_slot_unavailable"
            })
        ));
        scope.close_admission();
        assert!(scope.permit(manifest, first_registration).is_err());
        drop(scope);
        first.request_shutdown(cleanup);
        second.request_shutdown(cleanup);
        let first_closed = close_fixture(&mut first, cleanup);
        let second_closed = close_fixture(&mut second, cleanup);
        assert_eq!(first_closed.process_identity, Some(first_process));
        assert_eq!(second_closed.process_identity, Some(second_process));

        // Both actual process leases have retired; a retained closed worker
        // receipt alone must not keep the whole group exclusively claimed.
        let until = Instant::now() + config.handshake_timeout;
        let mut replacement = ExternalUciCpuChecker::create(config).unwrap();
        replacement.start(until, &AtomicBool::new(false)).unwrap();
        let closed = replacement
            .shutdown(until + Duration::from_secs(1))
            .unwrap();
        assert!(closed.exit_observed && closed.cleanup_complete);
        assert!(closed.stdout_drained && closed.stderr_drained);
        assert!(!closed.quarantined && !closed.ownership_lost);
    }

    #[cfg(target_os = "linux")]
    fn assert_native_stop_ready(
        completion: &TeacherCompletion,
        submitted: &TeacherDispatch,
        expected: ExternalCompletion,
    ) {
        assert!(completion.matches_submission(submitted));
        assert_eq!(completion.go_issue(), ExternalGoIssue::Issued);
        assert_eq!(
            completion.issued_constraint(),
            Some(ExternalGoConstraint::Nodes {
                reported_nodes: 4096
            })
        );
        assert_eq!(completion.issued_go(), Some("go nodes 4096"));
        assert!(completion.controls().terminal);
        let outcome = completion.result().as_ref().unwrap();
        assert_eq!(
            outcome.report.profile,
            ExternalEngineProfile::Reckless090EmbeddedNnue
        );
        assert_eq!(outcome.report.purpose, ExternalEnginePurpose::Teacher);
        assert_eq!(
            outcome.report.raw.identity.declared_name,
            "RZ external checker fixture"
        );
        assert_eq!(outcome.report.raw.identity.declared_license, "MIT");
        assert_eq!(outcome.report.raw.completion, expected);
        assert_eq!(outcome.report.strict_native_nodes, None);
        assert!(outcome.report.readiness_barrier_completed);
        assert!(!outcome.conditional_policy_target_known);
        let attempt = completion.attempt().unwrap();
        assert_eq!(attempt.elapsed, outcome.report.raw.elapsed);
        let external = attempt.external.as_ref().unwrap();
        assert_eq!(external.partial_report.as_ref(), Some(&outcome.report.raw));
        assert_eq!(
            external.process.process_identity,
            Some(submitted.process_identity())
        );
        let timing = completion.timing();
        let go = timing.go_issued_at.unwrap();
        let stop = timing.stop_sent_at.unwrap();
        let bestmove = timing.bestmove_drained_at.unwrap();
        let ready = timing.ready_completed_at.unwrap();
        assert!(
            go <= stop
                && stop <= bestmove
                && bestmove <= ready
                && ready <= timing.returned_at.unwrap()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn native_pipe_cancel_yield_ready_and_second_go_preserve_teacher_provenance() {
        let _guard = super::super::tests::FIXTURE_LOCK.lock().unwrap();
        let (mut worker, job, hard) = native_fixture("fake_uci_waits_for_stop");
        let first = worker.submit(job.clone()).unwrap();
        wait_progress(&worker, job.id, hard);
        job.control.request_cancel(7).unwrap();
        // A second logical job cannot bypass the first terminal reservation.
        let mut second = job.clone();
        second.id.sequence += 1;
        second.control = ControlMailbox::new(second.id, worker.signal());
        assert!(worker.submit(second.clone()).is_err());
        let completion = wait_completion(&worker, hard);
        assert_native_stop_ready(&completion, &first, ExternalCompletion::StoppedCanceled);
        assert_eq!(completion.controls().cancel_ack, 7);
        // Delivery identity survives a missing process observation; the
        // strict submission/label check must still reject that terminal.
        let mut missing_process = completion.clone();
        missing_process.process = None;
        assert!(missing_process.matches_dispatch_identity(&first));
        assert!(!missing_process.matches_submission(&first));
        assert!(!missing_process.matches_dispatch_identity(&TeacherDispatch {
            admission: Arc::new(()),
            ..first.clone()
        }));
        assert!(!completion.matches_submission(&TeacherDispatch {
            admission: Arc::new(()),
            ..first.clone()
        }));
        let next = worker.submit(second.clone()).unwrap();
        wait_progress(&worker, second.id, hard);
        second.control.request_yield(9).unwrap();
        let completion = wait_completion(&worker, hard);
        assert_native_stop_ready(&completion, &next, ExternalCompletion::StoppedYield);
        assert_eq!(completion.controls().yield_ack, 9);
        assert!(!completion.matches_submission(&first));
        assert_eq!(next.process_identity(), first.process_identity());
        let closed = close_fixture(&mut worker, Instant::now() + Duration::from_secs(2));
        assert_eq!(closed.process_identity, Some(first.process_identity()));
        let diagnostics = worker.take_diagnostics_after_join().unwrap();
        let output = String::from_utf8_lossy(diagnostics.stdout());
        assert!(output.matches("bestmove ").count() >= 2);
        assert!(output.matches("readyok").count() >= 3);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn native_pipe_failed_stop_preserves_issued_attempt_and_quiesces_owned_process() {
        let _guard = super::super::tests::FIXTURE_LOCK.lock().unwrap();
        let (mut worker, job, hard) = native_fixture("fake_uci_ignores_stop");
        let submitted = worker.submit(job.clone()).unwrap();
        wait_progress(&worker, job.id, hard);
        job.control.request_cancel(3).unwrap();
        let completion = wait_completion(&worker, hard);
        assert!(completion.matches_submission(&submitted));
        assert_eq!(completion.go_issue(), ExternalGoIssue::Issued);
        assert_eq!(completion.issued_go(), Some("go nodes 4096"));
        assert!(completion.result().is_err());
        assert!(
            completion
                .attempt()
                .unwrap()
                .external
                .as_ref()
                .unwrap()
                .partial_report
                .is_some()
        );
        // Failed stop cannot mint cancel ack or readiness from a queued request.
        assert_eq!(completion.controls().cancel_ack, 0);
        assert!(completion.timing().stop_sent_at.is_some());
        assert!(completion.timing().bestmove_drained_at.is_none());
        assert!(completion.timing().ready_completed_at.is_none());
        assert!(worker.submit(job).is_err());
        let closed = close_fixture(&mut worker, Instant::now() + Duration::from_secs(2));
        assert_eq!(closed.process_identity, Some(submitted.process_identity()));
        assert!(worker.take_diagnostics_after_join().is_some());
    }
}
