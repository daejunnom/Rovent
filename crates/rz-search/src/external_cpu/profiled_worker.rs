//! Rovent의 명시적 UCI 제한을 private 물리 소유자에게 비동기로 전달한다.
//!
//! 기존 Combined 전용 worker와 codec은 변경하지 않는다. manager의 제출·poll은
//! bounded mailbox와 보존된 증거만 다루며, Rules 재생·pipe·stop/drain은 이
//! worker에서 수행한다. 외부 reported depth/nodes/CP/mate는 Rules 증명이 아니다.

use super::worker::{
    ExternalWorkerClock, ExternalWorkerStartup, external_evaluation_namespace,
    external_root_restriction_digest,
};
use super::*;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_contracts::rovent::{
    ExecutionMode, QuestionEnvelope, ReciprocalQuestionId, SearchRecipient,
};
use rz_contracts::{Digest, ExecutionId, MonotonicTick};
use rz_runtime::CompletionSignal;
use rz_runtime::adaptive::{
    ControlMailbox, ControlSnapshot, MailboxError, MailboxOccupancy, WorkerMailbox,
};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

mod current_root;
pub use current_root::{
    ExternalRootSource, ProfiledExternalCurrentRootFence, ProfiledExternalProcessCloseProof,
    ProfiledExternalResetProof, ProfiledExternalRootSource,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProfiledExternalScope {
    pub question: ReciprocalQuestionId,
    pub execution: ExecutionId,
    pub owner: CpuWorkerId,
}

#[derive(Clone, Debug)]
pub struct ProfiledExternalJob {
    pub question: QuestionEnvelope,
    pub mode: ExecutionMode,
    pub execution: ExecutionId,
    pub owner: CpuWorkerId,
    pub position: Position,
    pub root_moves: Option<Vec<BoardMove>>,
    pub constraint: ExternalGoConstraint,
    pub soft_quantum: Duration,
    pub context_token_budget: u16,
    pub control: ControlMailbox<ProfiledExternalScope>,
}

impl ProfiledExternalJob {
    pub fn scope(&self) -> ProfiledExternalScope {
        ProfiledExternalScope {
            question: self.question.id,
            execution: self.execution,
            owner: self.owner,
        }
    }
}

/// Minted only after actual startup and successful bounded mailbox admission.
/// It certifies submission, never go, a teacher target, or a Rules result.
#[derive(Clone, Debug)]
pub struct ProfiledExternalDispatch {
    question: QuestionEnvelope,
    scope: ProfiledExternalScope,
    constraint: ExternalGoConstraint,
    profile: ExternalEngineProfile,
    purpose: ExternalEnginePurpose,
    evaluation_namespace: Digest,
    process: ExternalProcessIdentity,
    root_restriction: Digest,
}

impl ProfiledExternalDispatch {
    pub fn scope(&self) -> ProfiledExternalScope {
        self.scope
    }
    pub fn question(&self) -> QuestionEnvelope {
        self.question
    }
    pub fn constraint(&self) -> ExternalGoConstraint {
        self.constraint
    }
    pub fn process_identity(&self) -> ExternalProcessIdentity {
        self.process
    }
    pub fn root_restriction(&self) -> Digest {
        self.root_restriction
    }
    pub(crate) fn profile(&self) -> ExternalEngineProfile {
        self.profile
    }
    pub(crate) fn purpose(&self) -> ExternalEnginePurpose {
        self.purpose
    }
    pub(crate) fn evaluation_namespace(&self) -> Digest {
        self.evaluation_namespace
    }
}

#[derive(Clone, Debug)]
pub struct ProfiledExternalOutcome {
    pub report: ProfiledExternalReport,
    pub opinions: Vec<ExternalPolicyOpinion>,
    pub requested_multipv: u16,
    pub conditional_policy_target_known: bool,
    pub stop_control: Option<ExternalControlObservation>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ProfiledExternalTiming {
    pub admitted_at: Option<Instant>,
    pub run_started_at: Option<Instant>,
    pub go_issued_at: Option<Instant>,
    pub quantum_deadline: Option<Instant>,
    pub stop_sent_at: Option<Instant>,
    pub bestmove_drained_at: Option<Instant>,
    pub ready_completed_at: Option<Instant>,
    pub returned_at: Option<Instant>,
}

/// Private fields preserve the original worker-issued provenance. A caller's
/// public ProfiledExternalReport cannot be rewrapped into this completion.
#[derive(Clone, Debug)]
pub struct ProfiledExternalCompletion {
    question: QuestionEnvelope,
    scope: ProfiledExternalScope,
    requested: ExternalGoConstraint,
    profile: ExternalEngineProfile,
    purpose: ExternalEnginePurpose,
    evaluation_namespace: Digest,
    process: Option<ExternalProcessIdentity>,
    result: Result<ProfiledExternalOutcome, CheckerError>,
    attempt: Option<CheckerAttempt>,
    controls: ControlSnapshot,
    timing: ProfiledExternalTiming,
    go_issue: ExternalGoIssue,
    root_restriction: Digest,
}

impl ProfiledExternalCompletion {
    pub fn question(&self) -> QuestionEnvelope {
        self.question
    }
    pub fn scope(&self) -> ProfiledExternalScope {
        self.scope
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
    pub fn process_identity(&self) -> Option<ExternalProcessIdentity> {
        self.process
    }
    pub fn root_restriction(&self) -> Digest {
        self.root_restriction
    }
    pub(crate) fn profile(&self) -> ExternalEngineProfile {
        self.profile
    }
    pub(crate) fn purpose(&self) -> ExternalEnginePurpose {
        self.purpose
    }
    pub(crate) fn evaluation_namespace(&self) -> Digest {
        self.evaluation_namespace
    }
    pub(crate) fn requested(&self) -> ExternalGoConstraint {
        self.requested
    }
    pub fn matches_submission(&self, dispatch: &ProfiledExternalDispatch) -> bool {
        self.question == dispatch.question
            && self.scope == dispatch.scope
            && self.requested == dispatch.constraint
            && self.profile == dispatch.profile
            && self.purpose == dispatch.purpose
            && self.evaluation_namespace == dispatch.evaluation_namespace
            && self.process == Some(dispatch.process)
            && self.root_restriction == dispatch.root_restriction
    }
}

#[derive(Clone, Debug)]
pub struct ProfiledExternalProgress {
    pub scope: ProfiledExternalScope,
    pub opinion: ExternalPolicyOpinion,
    pub reported_nodes: Option<u64>,
    pub strict_native_nodes: Option<u64>,
    pub control_sequence: u64,
}

/// Root/new-game reuse is granted only after accepted old terminals have been
/// consumed and this same physical owner completed its ready/reset barrier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProfiledExternalRootFence {
    previous: SearchAuthority,
    next: SearchAuthority,
    sequence: u64,
    execution_high_water: u64,
    native_request_high_water: u64,
    process: CheckerShutdown,
    new_game_reset: bool,
}

impl ProfiledExternalRootFence {
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn previous(&self) -> SearchAuthority {
        self.previous
    }
    pub fn next(&self) -> SearchAuthority {
        self.next
    }
    pub fn new_game_reset(&self) -> bool {
        self.new_game_reset
    }
    pub fn execution_high_water(&self) -> u64 {
        self.execution_high_water
    }
    pub fn native_request_high_water(&self) -> u64 {
        self.native_request_high_water
    }
    pub fn permits_reuse(&self) -> bool {
        reusable_process(self.process)
    }
    pub fn process_identity(&self) -> Option<ExternalProcessIdentity> {
        self.process.process_identity
    }
}

#[derive(Clone, Copy, Debug)]
struct Transition {
    previous: SearchAuthority,
    next: SearchAuthority,
    sequence: u64,
    until: Instant,
}

struct RootState {
    authority: SearchAuthority,
    transition: Option<Transition>,
    result: Option<Result<ProfiledExternalRootFence, CheckerError>>,
    fence_sequence: u64,
    execution_high_water: u64,
    last_question: Option<QuestionEnvelope>,
    source_sequence: u64,
    process: Option<ExternalProcessIdentity>,
    accepted_jobs: u64,
    terminal_controls: u64,
    drained_terminals: u64,
    lifecycle_failure: Option<CheckerError>,
    quiesce: Option<current_root::QuiesceRequest>,
    quiesce_result: Option<Result<ProfiledExternalCurrentRootFence, CheckerError>>,
    initial_reset: Option<ProfiledExternalResetProof>,
    current_reset: Option<ProfiledExternalResetProof>,
    reset_sequence: u64,
}

#[derive(Debug)]
struct TimedJob {
    original: ProfiledExternalJob,
    admitted_at: Instant,
}

/// Original bounded process output, retained only after the owner finished its
/// shutdown attempt. Absence is unknown, not evidence of an empty transcript.
#[derive(Clone, Debug)]
pub struct ProfiledExternalDiagnostics {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}
impl ProfiledExternalDiagnostics {
    pub fn stdout(&self) -> &[u8] {
        &self.stdout
    }
    pub fn stderr(&self) -> &[u8] {
        &self.stderr
    }
}
type Mailbox = WorkerMailbox<TimedJob, ProfiledExternalCompletion, ProfiledExternalProgress>;

pub struct ProfiledExternalWorker {
    config: ExternalCpuConfig,
    profile: ExternalEngineProfile,
    purpose: ExternalEnginePurpose,
    owner: CpuWorkerId,
    clock: ExternalWorkerClock,
    mailbox: Mailbox,
    shutdown: Arc<AtomicBool>,
    shutdown_deadline: Arc<Mutex<Option<Instant>>>,
    startup: Arc<Mutex<Option<ExternalWorkerStartup>>>,
    roots: Arc<Mutex<RootState>>,
    thread: Option<JoinHandle<Result<CheckerShutdown, CheckerError>>>,
    joined: Option<Result<CheckerShutdown, CheckerError>>,
    diagnostics: Arc<Mutex<Option<ProfiledExternalDiagnostics>>>,
}

impl ProfiledExternalWorker {
    /// No native file/pipe operation is performed on the caller. The optional
    /// launch capability is retained by the one physical owner in its thread.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        config: ExternalCpuConfig,
        profile: ExternalEngineProfile,
        purpose: ExternalEnginePurpose,
        owner: CpuWorkerId,
        clock: ExternalWorkerClock,
        initial_root: SearchAuthority,
        startup_deadline: Instant,
        launch_pin: Option<ExternalLaunchPin>,
    ) -> Result<Self, CheckerError> {
        config.validate()?;
        profile.validate_identity(
            purpose,
            &config.identity,
            config.expected_uci_name.as_deref().unwrap_or_default(),
        )?;
        if purpose == ExternalEnginePurpose::ArenaOpponent {
            return Err(CheckerError::Unsupported(
                "arena opponent must use the arena process owner",
            ));
        }
        if cfg!(not(target_os = "linux")) {
            return Err(CheckerError::Unsupported(
                "profiled external worker requires Linux native ELF",
            ));
        }
        now(clock)?;
        if initial_root.epoch != clock.domain.0
            || initial_root.game.0 == 0
            || initial_root.root.0 == 0
            || Instant::now() >= startup_deadline
        {
            return Err(CheckerError::Invalid(
                "profiled worker startup authority/deadline",
            ));
        }
        let mailbox = Mailbox::new(8, 16, CompletionSignal::default())
            .map_err(|_| error("profiled_worker", "mailbox_capacity"))?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_deadline = Arc::new(Mutex::new(None));
        let startup = Arc::new(Mutex::new(None));
        let roots = Arc::new(Mutex::new(RootState::new(initial_root)));
        let diagnostics = Arc::new(Mutex::new(None));
        let worker_diagnostics = diagnostics.clone();
        let (
            worker_config,
            worker_mailbox,
            worker_shutdown,
            worker_deadline,
            worker_startup,
            worker_roots,
        ) = (
            config.clone(),
            mailbox.clone(),
            shutdown.clone(),
            shutdown_deadline.clone(),
            startup.clone(),
            roots.clone(),
        );
        let thread = std::thread::Builder::new()
            .name(format!("rz-profiled-external-{}", owner.0))
            .spawn(move || {
                let _admission = current_root::WorkerAdmissionGuard {
                    shutdown: Arc::clone(&worker_shutdown),
                    mailbox: worker_mailbox.clone(),
                };
                run_worker(
                    worker_config,
                    profile,
                    purpose,
                    owner,
                    clock,
                    startup_deadline,
                    worker_mailbox,
                    worker_shutdown,
                    worker_deadline,
                    worker_startup,
                    worker_roots,
                    launch_pin,
                    worker_diagnostics,
                )
            })
            .map_err(|_| error("profiled_worker", "thread_spawn"))?;
        Ok(Self {
            config,
            profile,
            purpose,
            owner,
            clock,
            mailbox,
            shutdown,
            shutdown_deadline,
            startup,
            roots,
            thread: Some(thread),
            joined: None,
            diagnostics,
        })
    }

    pub fn signal(&self) -> CompletionSignal {
        self.mailbox.signal()
    }
    pub fn occupancy(&self) -> MailboxOccupancy {
        self.mailbox.occupancy()
    }
    pub fn configuration_matches(
        &self,
        config: &ExternalCpuConfig,
        profile: ExternalEngineProfile,
        purpose: ExternalEnginePurpose,
        owner: CpuWorkerId,
    ) -> bool {
        self.config == *config
            && self.profile == profile
            && self.purpose == purpose
            && self.owner == owner
    }
    pub fn clock_matches(&self, clock: ExternalWorkerClock) -> bool {
        self.clock.domain == clock.domain && self.clock.origin == clock.origin
    }
    pub fn startup_report(&self) -> Option<ExternalWorkerStartup> {
        self.startup
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Capability rejection occurs before mailbox admission. Legal move and
    /// exact Rules/history checks are deliberately deferred to the worker.
    pub fn submit(
        &mut self,
        job: ProfiledExternalJob,
    ) -> Result<ProfiledExternalDispatch, CheckerError> {
        validate_metadata(
            &job,
            self.owner,
            self.clock,
            &self.config,
            self.profile,
            self.purpose,
        )?;
        if self.shutdown.load(Ordering::Acquire)
            || self.thread.as_ref().is_none_or(JoinHandle::is_finished)
        {
            return Err(error("profiled_admission", "worker_closed"));
        }
        let startup = self
            .startup_report()
            .ok_or_else(|| error("profiled_admission", "actual_startup_missing"))?;
        startup.result?;
        let process = startup
            .attempt
            .and_then(|a| a.external)
            .map(|a| a.process)
            .filter(|p| reusable_process(*p))
            .and_then(|p| p.process_identity)
            .ok_or_else(|| error("profiled_admission", "actual_startup_owner_missing"))?;
        let mut roots = self
            .roots
            .lock()
            .map_err(|_| error("profiled_admission", "root_state_poisoned"))?;
        validate_root_admission(&roots, &job)?;
        let dispatch = ProfiledExternalDispatch {
            question: job.question,
            scope: job.scope(),
            constraint: job.constraint,
            profile: self.profile,
            purpose: self.purpose,
            evaluation_namespace: external_evaluation_namespace(&self.config.identity)?,
            process,
            root_restriction: external_root_restriction_digest(job.root_moves.as_deref()),
        };
        let accepted_jobs = roots
            .accepted_jobs
            .checked_add(1)
            .ok_or(CheckerError::Invalid(
                "profiled accepted job count overflow",
            ))?;
        self.mailbox
            .admit(
                TimedJob {
                    original: job,
                    admitted_at: Instant::now(),
                },
                2,
            )
            .map_err(|rejected| match rejected.reason {
                MailboxError::QueueFull | MailboxError::ReliableFull => {
                    error("profiled_admission", "capacity")
                }
                _ => error("profiled_admission", "closed"),
            })?;
        roots.execution_high_water = dispatch.scope.execution.sequence;
        roots.last_question = Some(dispatch.question);
        roots.accepted_jobs = accepted_jobs;
        Ok(dispatch)
    }

    pub fn take_completion(&self) -> Option<ProfiledExternalCompletion> {
        let result = self.mailbox.take_reliable();
        if result.is_some() {
            let mut roots = self.roots.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(next) = roots.drained_terminals.checked_add(1) {
                roots.drained_terminals = next;
            } else {
                roots.lifecycle_failure = Some(error("profiled_root", "terminal_count_overflow"));
            }
            drop(roots);
            self.mailbox.signal().notify();
        }
        result
    }
    pub fn take_progress(&self) -> Option<(u64, ProfiledExternalProgress)> {
        self.mailbox.take_progress()
    }

    /// Bypasses the full command queue. The active job observes revocation,
    /// and every accepted old job keeps its reserved terminal obligation.
    pub fn request_root_change(
        &self,
        next: SearchAuthority,
        until: Instant,
    ) -> Result<(), CheckerError> {
        if self.shutdown.load(Ordering::Acquire)
            || self.thread.as_ref().is_none_or(JoinHandle::is_finished)
            || Instant::now() >= until
        {
            return Err(error("profiled_root", "closed_or_expired"));
        }
        let mut roots = self
            .roots
            .lock()
            .map_err(|_| error("profiled_root", "state_poisoned"))?;
        if roots.lifecycle_failure.is_some() {
            return Err(error("profiled_root", "lifecycle_accounting_failed"));
        }
        if roots.quiesce.is_some() && !matches!(&roots.quiesce_result, Some(Ok(_))) {
            return Err(error("profiled_root", "current_quiesce_not_completed"));
        }
        if next.epoch != roots.authority.epoch
            || next.implementation != roots.authority.implementation
            || next.game.0 < roots.authority.game.0
            || next.root.0 <= roots.authority.root.0
        {
            return Err(error("profiled_root", "authority_not_forward"));
        }
        if let Some(old) = roots.transition {
            if old.next != next {
                return Err(error("profiled_root", "different_transition_pending"));
            }
            if roots.result.is_none() {
                roots.transition = Some(Transition {
                    until: old.until.min(until),
                    ..old
                });
            }
        } else {
            roots.fence_sequence = roots
                .fence_sequence
                .checked_add(1)
                .ok_or(CheckerError::Invalid("profiled root sequence overflow"))?;
            roots.transition = Some(Transition {
                previous: roots.authority,
                next,
                sequence: roots.fence_sequence,
                until,
            });
            roots.result = None;
            roots.quiesce = None;
            roots.quiesce_result = None;
        }
        drop(roots);
        self.mailbox.signal().notify();
        Ok(())
    }

    pub fn root_change_result(&self) -> Option<Result<ProfiledExternalRootFence, CheckerError>> {
        self.roots
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .result
            .clone()
    }

    pub fn request_shutdown(&self, until: Instant) {
        self.mailbox.close();
        let mut deadline = self
            .shutdown_deadline
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        *deadline = Some(deadline.map_or(until, |old| old.min(until)));
        self.shutdown.store(true, Ordering::Release);
        self.mailbox.signal().notify();
    }
    pub fn try_join(&mut self) -> Option<Result<CheckerShutdown, CheckerError>> {
        if self.thread.as_ref().is_some_and(JoinHandle::is_finished) {
            let result = self
                .thread
                .take()
                .expect("checked physical worker handle")
                .join()
                .unwrap_or_else(|_| Err(error("profiled_worker", "thread_panicked")));
            self.joined = Some(result.clone());
            Some(result)
        } else {
            None
        }
    }
    pub fn shutdown_snapshot(&self) -> Option<Result<CheckerShutdown, CheckerError>> {
        self.joined.clone()
    }
    pub fn take_diagnostics_after_join(&mut self) -> Option<ProfiledExternalDiagnostics> {
        self.joined.as_ref()?;
        self.diagnostics
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
    }
}

impl Drop for ProfiledExternalWorker {
    fn drop(&mut self) {
        self.request_shutdown(Instant::now() + self.config.shutdown_grace);
    }
}

fn now(clock: ExternalWorkerClock) -> Result<MonotonicTick, CheckerError> {
    let elapsed =
        Instant::now()
            .checked_duration_since(clock.origin)
            .ok_or(CheckerError::Invalid(
                "profiled clock origin is in the future",
            ))?;
    Ok(MonotonicTick(u64::try_from(elapsed.as_nanos()).map_err(
        |_| CheckerError::Invalid("profiled clock overflow"),
    )?))
}

fn validate_metadata(
    job: &ProfiledExternalJob,
    owner: CpuWorkerId,
    clock: ExternalWorkerClock,
    config: &ExternalCpuConfig,
    profile: ExternalEngineProfile,
    purpose: ExternalEnginePurpose,
) -> Result<Instant, CheckerError> {
    profile.require_query(job.constraint, job.root_moves.is_some())?;
    if !job.mode.has_native_r() {
        return Err(CheckerError::Unsupported(
            "P-C mode cannot admit external research/teacher work",
        ));
    }
    if purpose == ExternalEnginePurpose::ArenaOpponent
        || (purpose == ExternalEnginePurpose::ResearchChecker
            && job.question.recipient != SearchRecipient::Researcher)
    {
        return Err(CheckerError::Invalid(
            "profiled query purpose/recipient mismatch",
        ));
    }
    job.question
        .validate(
            job.mode,
            clock.domain,
            now(clock)?,
            job.context_token_budget,
        )
        .map_err(|_| CheckerError::Invalid("profiled question clock/view/mode seal"))?;
    let deadline = clock
        .origin
        .checked_add(Duration::from_nanos(job.question.original_deadline.at.0))
        .ok_or(CheckerError::Invalid("profiled deadline overflow"))?;
    if job.owner != owner
        || job.execution.epoch != job.question.id.authority.epoch
        || job.execution.sequence == 0
        || *job.control.scope() != job.scope()
        || job.control.snapshot().terminal
        || job.context_token_budget == 0
        || job.context_token_budget > 512
        || job
            .constraint
            .requested_depth()
            .is_some_and(|depth| depth > config.max_depth)
        || job.soft_quantum < EXTERNAL_MIN_QUANTUM
        || job.soft_quantum > Duration::from_secs(180)
        || job
            .root_moves
            .as_ref()
            .is_some_and(|roots| roots.is_empty() || roots.len() > 256 || roots.capacity() > 256)
        || config.stop_grace < EXTERNAL_STOP_DRAIN_RESERVE
        || config.max_task_wall_time < job.soft_quantum + config.stop_grace
        || deadline.saturating_duration_since(Instant::now()) < job.soft_quantum + config.stop_grace
    {
        return Err(CheckerError::Invalid(
            "profiled owner/control/resource/deadline seal",
        ));
    }
    Ok(deadline)
}

fn validate_root_admission(
    roots: &RootState,
    job: &ProfiledExternalJob,
) -> Result<(), CheckerError> {
    if roots.transition.is_some()
        || roots.quiesce.is_some()
        || roots.lifecycle_failure.is_some()
        || roots.authority != job.question.id.authority
        || job.execution.sequence <= roots.execution_high_water
        || roots.last_question.is_some_and(|previous| {
            previous.id.sequence > job.question.id.sequence
                || (previous.id.sequence == job.question.id.sequence && previous != job.question)
        })
    {
        return Err(error(
            "profiled_admission",
            "revoked_duplicate_or_old_question",
        ));
    }
    Ok(())
}

fn validate_rules(job: &ProfiledExternalJob) -> Result<(), CheckerError> {
    let exported = rz_position::contracts::ContractPosition::new(
        job.question.anchor.owner,
        job.position.clone(),
    )
    .export()
    .map_err(|_| CheckerError::Invalid("profiled Rules export failed"))?;
    if exported.snapshot().identity() != job.question.anchor {
        return Err(CheckerError::Invalid(
            "profiled exact anchor/history mismatch",
        ));
    }
    let legal = job.position.ordered_legal_moves();
    if let Some(roots) = &job.root_moves {
        if roots.iter().enumerate().any(|(at, movement)| {
            !legal.moves().contains(movement) || roots[..at].contains(movement)
        }) {
            return Err(CheckerError::Invalid(
                "profiled illegal or duplicate root restriction",
            ));
        }
    }
    Ok(())
}

fn reusable_process(process: CheckerShutdown) -> bool {
    process
        .process_identity
        .is_some_and(|id| id.pid != 0 && id.process_group != 0 && id.proc_start_ticks != 0)
        && !process.quit_sent
        && !process.exit_observed
        && !process.cleanup_complete
        && !process.quarantined
        && !process.ownership_lost
}

fn physical_fence(
    shutdown_deadline: &Mutex<Option<Instant>>,
    roots: &Mutex<RootState>,
) -> Option<Instant> {
    let shutdown = *shutdown_deadline.lock().unwrap_or_else(|p| p.into_inner());
    let root = roots
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .physical_deadline();
    match (shutdown, root) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

struct JobControl {
    scope: ProfiledExternalScope,
    controls: ControlMailbox<ProfiledExternalScope>,
    mailbox: Mailbox,
    shutdown_deadline: Arc<Mutex<Option<Instant>>>,
    roots: Arc<Mutex<RootState>>,
    progress_sequence: Arc<AtomicU64>,
    timing: Arc<Mutex<ProfiledExternalTiming>>,
}

impl ExternalControl for JobControl {
    fn observe(&self) -> ExternalControlObservation {
        let snapshot = self.controls.snapshot();
        let (sequence, action) = if snapshot.terminal {
            (0, ExternalControlAction::Revoked)
        } else if snapshot.is_cancelled() {
            (snapshot.cancel_sequence, ExternalControlAction::Cancel)
        } else if self
            .roots
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_revoked()
        {
            (0, ExternalControlAction::Revoked)
        } else if let Some(sequence) = snapshot.pending_yield() {
            (sequence, ExternalControlAction::Yield)
        } else {
            (0, ExternalControlAction::Continue)
        };
        ExternalControlObservation { sequence, action }
    }
    fn physical_fence_deadline(&self) -> Option<Instant> {
        physical_fence(&self.shutdown_deadline, &self.roots)
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
        let pending = self.controls.snapshot().pending_publish();
        let Ok(previous) =
            self.progress_sequence
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |previous| {
                    previous.checked_add(1)
                })
        else {
            return;
        };
        if self
            .mailbox
            .publish_progress(
                previous + 1,
                ProfiledExternalProgress {
                    scope: self.scope,
                    opinion: opinion.clone(),
                    reported_nodes,
                    strict_native_nodes: None,
                    control_sequence: pending.unwrap_or(0),
                },
            )
            .is_ok()
        {
            if let Some(sequence) = pending {
                let _ = self.controls.acknowledge_publish_progress(sequence);
            }
        }
    }
}

fn policy_target_known(
    report: &ExternalCheckerReport,
    opinions: &[ExternalPolicyOpinion],
    multipv: u16,
) -> bool {
    report.completion == ExternalCompletion::BestMove
        && opinions.len() == usize::from(multipv)
        && opinions.iter().enumerate().all(|(index, opinion)| {
            opinion.rank == index as u16 + 1
                && Some(opinion.depth) == opinions.first().map(|first| first.depth)
                && opinion.bound == ExternalBound::ExactReported
                && !opinions[..index]
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
    purpose: ExternalEnginePurpose,
    owner: CpuWorkerId,
    clock: ExternalWorkerClock,
    startup_deadline: Instant,
    mailbox: Mailbox,
    shutdown: Arc<AtomicBool>,
    shutdown_deadline: Arc<Mutex<Option<Instant>>>,
    startup_report: Arc<Mutex<Option<ExternalWorkerStartup>>>,
    roots: Arc<Mutex<RootState>>,
    launch_pin: Option<ExternalLaunchPin>,
    diagnostics: Arc<Mutex<Option<ProfiledExternalDiagnostics>>>,
) -> Result<CheckerShutdown, CheckerError> {
    let mut checker = ExternalUciCpuChecker::create_profiled(config.clone(), profile, purpose)?;
    checker.launch_pin = launch_pin;
    let evaluation_namespace = external_evaluation_namespace(&config.identity)?;
    let startup_started = Instant::now();
    let startup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let read_fence = || physical_fence(&shutdown_deadline, &roots);
        checker.start_with_cleanup_fence(startup_deadline, &shutdown, Some(&read_fence))?;
        let mut attempt = checker.last_attempt().cloned();
        let result = checker
            .reset_game_with_fence(startup_deadline, &shutdown, Some(&read_fence))
            .and_then(|()| {
                current_root::register_initial_reset(
                    &mut checker,
                    &roots,
                    &shutdown,
                    owner,
                    clock,
                    profile,
                    purpose,
                )
            });
        if let Some(attempt) = attempt.as_mut() {
            attempt.elapsed = startup_started.elapsed();
            if let Some(external) = attempt.external.as_mut() {
                external.process = checker
                    .owner
                    .as_ref()
                    .map_or(external.process, |process| process.evidence());
            }
        }
        checker.last_attempt = attempt;
        result
    }))
    .unwrap_or_else(|_| {
        checker.failed = true;
        Err(error("profiled_startup", "panicked"))
    });
    *startup_report.lock().unwrap_or_else(|p| p.into_inner()) = Some(ExternalWorkerStartup {
        result: startup.clone().and_then(|()| {
            checker
                .startup_uci()
                .ok_or_else(|| error("profiled_startup", "identity_missing"))
        }),
        attempt: checker.last_attempt().cloned(),
    });
    mailbox.signal().notify();
    if startup.is_err() {
        shutdown.store(true, Ordering::Release);
        mailbox.close();
        let mut fence = shutdown_deadline.lock().unwrap_or_else(|p| p.into_inner());
        *fence = Some(fence.map_or(startup_deadline, |old| old.min(startup_deadline)));
    }
    let progress_sequence = Arc::new(AtomicU64::new(0));
    loop {
        let observed = mailbox.signal().version();
        if let Some(mut admitted) = mailbox.take_command() {
            let mut supervisor = admitted
                .reliable
                .split_off(1)
                .map_err(|_| error("profiled_worker", "terminal_permit_split"))?;
            let job = admitted.command.original;
            let scope = job.scope();
            let timing = Arc::new(Mutex::new(ProfiledExternalTiming {
                admitted_at: Some(admitted.command.admitted_at),
                run_started_at: Some(Instant::now()),
                ..Default::default()
            }));
            checker.reset_attempt();
            checker.go_issue = ExternalGoIssue::NotIssued;
            let operation = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || -> Result<ProfiledExternalOutcome, CheckerError> {
                    if shutdown.load(Ordering::Acquire) {
                        return Err(error("profiled_admission", "shutdown_before_go"));
                    }
                    let root = roots
                        .lock()
                        .map_err(|_| error("profiled_admission", "root_state_poisoned"))?;
                    if root.is_revoked() || root.authority != job.question.id.authority {
                        return Err(error("profiled_admission", "root_revoked_before_go"));
                    }
                    drop(root);
                    startup.clone()?;
                    let deadline =
                        validate_metadata(&job, owner, clock, &config, profile, purpose)?;
                    validate_rules(&job)?;
                    let multipv = config
                        .identity
                        .options
                        .get("MultiPV")
                        .and_then(|value| value.parse::<u16>().ok())
                        .ok_or(CheckerError::Invalid(
                            "profiled MultiPV declaration missing",
                        ))?;
                    if usize::from(multipv)
                        > job.root_moves.as_ref().map_or_else(
                            || job.position.ordered_legal_moves().moves().len(),
                            Vec::len,
                        )
                    {
                        return Err(CheckerError::Invalid(
                            "profiled MultiPV exceeds legal candidates",
                        ));
                    }
                    let control = JobControl {
                        scope,
                        controls: job.control.clone(),
                        mailbox: mailbox.clone(),
                        shutdown_deadline: shutdown_deadline.clone(),
                        roots: roots.clone(),
                        progress_sequence: progress_sequence.clone(),
                        timing: timing.clone(),
                    };
                    checker.cooperative_opinions.clear();
                    checker.stop_control = None;
                    let limits = CpuLimits {
                        max_depth: job.constraint.requested_depth().unwrap_or(0),
                        max_nodes: job.constraint.requested_nodes().unwrap_or(0),
                        deadline: Some(deadline),
                    };
                    // This private policy contributes only the quantum/reserve and
                    // MultiPV. An absent node request remains zero/None and is
                    // never validated/advertised as a native credit or a UCI cap.
                    let policy = ExternalSlicePolicy {
                        quantum: job.soft_quantum,
                        stop_drain_reserve: config.stop_grace,
                        requested_reported_nodes: job.constraint.requested_nodes().unwrap_or(0),
                        multipv,
                    };
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
                        return Err(error("profiled_analysis", "foreign_report_type"));
                    };
                    checker
                        .cooperative_opinions
                        .sort_by_key(|opinion| opinion.rank);
                    let opinions = checker.cooperative_opinions.clone();
                    let known = policy_target_known(&raw, &opinions, multipv);
                    Ok(ProfiledExternalOutcome {
                        report: ProfiledExternalReport {
                            profile,
                            purpose,
                            requested: job.constraint,
                            issued: checker.last_issued_constraint.ok_or(CheckerError::Invalid(
                                "profiled issued constraint missing",
                            ))?,
                            issued_go: checker
                                .last_issued_go
                                .take()
                                .ok_or(CheckerError::Invalid("profiled issued command missing"))?,
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
                    (Err(error("profiled_worker", "task_panicked")), true)
                }
            };
            if let Ok(outcome) = &result {
                if let Some(observed) = outcome.stop_control {
                    match observed.action {
                        ExternalControlAction::Cancel => {
                            let _ = job.control.acknowledge_cancel(observed.sequence);
                        }
                        ExternalControlAction::Yield => {
                            let _ = job.control.acknowledge_yield(observed.sequence);
                        }
                        ExternalControlAction::Continue | ExternalControlAction::Revoked => {}
                    }
                }
            }
            let controls = job.control.seal_terminal();
            {
                let mut root = roots.lock().unwrap_or_else(|p| p.into_inner());
                root.terminal_controls = root
                    .terminal_controls
                    .checked_add(1)
                    .ok_or_else(|| error("profiled_worker", "terminal_control_count_overflow"))?;
            }
            let mut retained_timing = *timing.lock().unwrap_or_else(|p| p.into_inner());
            retained_timing.returned_at = Some(Instant::now());
            let completion = ProfiledExternalCompletion {
                question: job.question,
                scope,
                requested: job.constraint,
                profile,
                purpose,
                evaluation_namespace,
                process: checker
                    .owner
                    .as_ref()
                    .and_then(|p| p.evidence().process_identity),
                result,
                attempt: checker.last_attempt().cloned(),
                controls,
                timing: retained_timing,
                go_issue: checker.go_issue,
                root_restriction: external_root_restriction_digest(job.root_moves.as_deref()),
            };
            let published = if panicked {
                supervisor.publish(completion)
            } else {
                admitted.reliable.publish(completion)
            };
            if published.is_err() {
                return Err(error(
                    "profiled_worker",
                    "reserved_terminal_publication_failed",
                ));
            }
            if panicked || checker.failed {
                let until = clock
                    .origin
                    .checked_add(Duration::from_nanos(job.question.original_deadline.at.0))
                    .unwrap_or_else(Instant::now)
                    .min(Instant::now() + config.shutdown_grace);
                let mut fence = shutdown_deadline.lock().unwrap_or_else(|p| p.into_inner());
                *fence = Some(fence.map_or(until, |old| old.min(until)));
                shutdown.store(true, Ordering::Release);
                mailbox.close();
            }
            continue;
        }
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        if current_root::poll_quiesce(
            &mut checker,
            &roots,
            &shutdown,
            &shutdown_deadline,
            &mailbox,
            owner,
            clock,
            profile,
            purpose,
        ) {
            continue;
        }
        let transition = {
            let root = roots.lock().unwrap_or_else(|p| p.into_inner());
            root.result.is_none().then_some(root.transition).flatten()
        };
        if let Some(transition) = transition {
            if Instant::now() >= transition.until {
                roots.lock().unwrap_or_else(|p| p.into_inner()).result = Some(Err(error(
                    "profiled_root",
                    "deadline_before_terminal_drain",
                )));
                mailbox.signal().notify();
                continue;
            }
            let occupancy = mailbox.occupancy();
            if occupancy.commands == 0
                && occupancy.reliable_queued == 0
                && occupancy.reliable_reserved == 0
                && roots
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .terminals_drained()
            {
                mailbox.take_progress();
                let read_fence = || physical_fence(&shutdown_deadline, &roots);
                let reset = transition.previous.game != transition.next.game;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if reset {
                        checker.reset_game_with_fence(
                            transition.until,
                            &shutdown,
                            Some(&read_fence),
                        )?;
                    } else {
                        checker.ready_with_fence(transition.until, &shutdown, Some(&read_fence))?;
                    }
                    let process = checker.owner()?.evidence();
                    if !reusable_process(process)
                        || process.process_identity
                            != roots.lock().unwrap_or_else(|p| p.into_inner()).process
                    {
                        return Err(error("profiled_root", "owner_not_reusable"));
                    }
                    Ok(process)
                }))
                .unwrap_or_else(|_| {
                    checker.failed = true;
                    Err(error("profiled_root", "barrier_panicked"))
                });
                let mut root = roots.lock().unwrap_or_else(|p| p.into_inner());
                let result = result.and_then(|process| {
                    if root.transition.is_none_or(|pending| {
                        pending.sequence != transition.sequence
                            || pending.next != transition.next
                            || Instant::now() >= pending.until
                    }) {
                        return Err(error("profiled_root", "late_or_replaced_ready"));
                    }
                    if root.lifecycle_failure.is_some()
                        || root.accepted_jobs != root.terminal_controls
                        || root.accepted_jobs != root.drained_terminals
                    {
                        return Err(error("profiled_root", "terminal_accounting_incomplete"));
                    }
                    Ok(ProfiledExternalRootFence {
                        previous: transition.previous,
                        next: transition.next,
                        sequence: transition.sequence,
                        execution_high_water: root.execution_high_water,
                        native_request_high_water: checker.request_id,
                        process,
                        new_game_reset: reset,
                    })
                });
                if result.is_ok() {
                    root.authority = transition.next;
                    root.source_sequence = transition.sequence;
                    root.transition = None;
                    root.last_question = None;
                    root.current_reset = None;
                    if reset {
                        if let Err(failure) = current_root::register_transition_reset(
                            &roots,
                            &shutdown,
                            &mut root,
                            transition.previous,
                            owner,
                            clock,
                            profile,
                            purpose,
                        ) {
                            root.lifecycle_failure = Some(failure.clone());
                            root.result = Some(Err(failure));
                            drop(root);
                            mailbox.signal().notify();
                            continue;
                        }
                    }
                }
                root.result = Some(result);
                drop(root);
                mailbox.signal().notify();
                continue;
            }
        }
        mailbox.wait_changed(observed, Duration::from_millis(20));
    }
    let until = shutdown_deadline
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .unwrap_or_else(|| Instant::now() + config.shutdown_grace);
    let result = checker.shutdown(until);
    let (stdout, stderr) = checker.diagnostic_output();
    *diagnostics.lock().unwrap_or_else(|p| p.into_inner()) = Some(ProfiledExternalDiagnostics {
        stdout: stdout.to_vec(),
        stderr: stderr.to_vec(),
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::pals_adaptive::ReviewViewSeal;
    use rz_contracts::rovent::{ChessPly, QuestionPurpose};
    use rz_contracts::{
        ClockDomain, Deadline, GameGeneration, OwnerId, ProcessEpoch, RootGeneration,
    };

    fn fixture() -> (ProfiledExternalJob, ExternalCpuConfig, ExternalWorkerClock) {
        let clock = ExternalWorkerClock {
            domain: ClockDomain(ProcessEpoch(1)),
            origin: Instant::now(),
        };
        let authority = SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: Digest([1; 32]),
        };
        let position = Position::startpos();
        let exported = rz_position::contracts::ContractPosition::new(OwnerId(1), position.clone())
            .export()
            .unwrap();
        let anchor = exported.snapshot().identity();
        let question = QuestionEnvelope {
            id: ReciprocalQuestionId {
                authority,
                sequence: 1,
            },
            parent: None,
            sender: SearchRecipient::Proposer,
            recipient: SearchRecipient::Researcher,
            purpose: QuestionPurpose::TacticalReview,
            anchor,
            anchor_root_ply: ChessPly(0),
            original_first_move: None,
            prefix_digest: Digest([2; 32]),
            line_digest: None,
            line_offset: ChessPly(0),
            question_digest: Digest([3; 32]),
            view: ReviewViewSeal {
                authority,
                view_sequence: 1,
                context_policy: Digest([4; 32]),
                view_digest: Digest([5; 32]),
                root_state: anchor,
                actual_tokens: 1,
                selected_records: 0,
                anchors: 1,
                length_bucket: 8,
            },
            evidence_revision: 0,
            root_budget: Digest([6; 32]),
            original_deadline: Deadline {
                clock: clock.domain,
                at: MonotonicTick(10_000_000_000),
            },
        };
        let scope = ProfiledExternalScope {
            question: question.id,
            execution: ExecutionId {
                epoch: authority.epoch,
                sequence: 1,
            },
            owner: CpuWorkerId(1),
        };
        let job = ProfiledExternalJob {
            question,
            mode: ExecutionMode::CpuPcr,
            execution: scope.execution,
            owner: scope.owner,
            position,
            root_moves: None,
            constraint: ExternalGoConstraint::Nodes {
                reported_nodes: 4096,
            },
            soft_quantum: Duration::from_millis(100),
            context_token_budget: 512,
            control: ControlMailbox::new(scope, CompletionSignal::default()),
        };
        let profile = ExternalEngineProfile::Reckless090EmbeddedNnue;
        let config = ExternalCpuConfig {
            program: std::env::temp_dir().join("rz-absent-profiled-worker-fixture"),
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
                        ExternalEnginePurpose::ResearchChecker,
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
            stop_grace: Duration::from_millis(500),
            shutdown_grace: Duration::from_millis(500),
            max_output_bytes: 65536,
            max_line_bytes: 4096,
        };
        (job, config, clock)
    }

    #[test]
    fn admission_checks_exact_capabilities_mode_owner_and_clock_without_a_process() {
        let (mut job, config, clock) = fixture();
        let profile = ExternalEngineProfile::Reckless090EmbeddedNnue;
        let purpose = ExternalEnginePurpose::ResearchChecker;
        assert!(validate_metadata(&job, CpuWorkerId(1), clock, &config, profile, purpose).is_ok());
        job.root_moves = Some(vec![BoardMove::from_uci("e2e4").unwrap()]);
        assert_eq!(
            validate_metadata(&job, CpuWorkerId(1), clock, &config, profile, purpose),
            Err(CheckerError::Unsupported(
                "engine does not support searchmoves"
            ))
        );
        job.root_moves = None;
        job.mode = ExecutionMode::CpuPc;
        assert_eq!(
            validate_metadata(&job, CpuWorkerId(1), clock, &config, profile, purpose),
            Err(CheckerError::Unsupported(
                "P-C mode cannot admit external research/teacher work"
            ))
        );
        job.mode = ExecutionMode::CpuPcr;
        assert!(validate_metadata(&job, CpuWorkerId(2), clock, &config, profile, purpose).is_err());
        job.question.original_deadline.clock = ClockDomain(ProcessEpoch(2));
        assert!(validate_metadata(&job, CpuWorkerId(1), clock, &config, profile, purpose).is_err());
    }

    #[test]
    fn worker_checks_rules_history_and_root_restriction_instead_of_trusting_metadata() {
        let (mut job, _, _) = fixture();
        assert!(validate_rules(&job).is_ok());
        job.position
            .make_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        assert_eq!(
            validate_rules(&job),
            Err(CheckerError::Invalid(
                "profiled exact anchor/history mismatch"
            ))
        );
        let (mut job, _, _) = fixture();
        job.root_moves = Some(vec![BoardMove::from_uci("a1a8").unwrap()]);
        assert_eq!(
            validate_rules(&job),
            Err(CheckerError::Invalid(
                "profiled illegal or duplicate root restriction"
            ))
        );
    }

    #[test]
    fn old_go_and_mutated_same_question_cannot_reuse_admission() {
        let (mut job, _, _) = fixture();
        let mut roots = RootState::new(job.question.id.authority);
        assert!(validate_root_admission(&roots, &job).is_ok());
        roots.execution_high_water = job.execution.sequence;
        roots.last_question = Some(job.question);
        assert!(validate_root_admission(&roots, &job).is_err());
        job.execution.sequence += 1;
        assert!(validate_root_admission(&roots, &job).is_ok());
        job.question.question_digest = Digest([99; 32]);
        assert!(validate_root_admission(&roots, &job).is_err());
    }

    #[test]
    fn full_command_queue_retains_cancel_and_terminal_reservations() {
        let mailbox = Mailbox::new(1, 2, CompletionSignal::default()).unwrap();
        let (job, _, _) = fixture();
        let control = job.control.clone();
        mailbox
            .admit(
                TimedJob {
                    original: job.clone(),
                    admitted_at: Instant::now(),
                },
                2,
            )
            .unwrap();
        assert_eq!(
            mailbox
                .admit(
                    TimedJob {
                        original: job,
                        admitted_at: Instant::now()
                    },
                    2
                )
                .unwrap_err()
                .reason,
            MailboxError::QueueFull
        );
        control.request_cancel(1).unwrap();
        assert!(control.snapshot().is_cancelled());
        assert_eq!(mailbox.occupancy().reliable_reserved, 2);
        let command = mailbox.take_command().unwrap();
        assert!(command.command.original.control.snapshot().is_cancelled());
        mailbox.close();
        assert_eq!(mailbox.occupancy().reliable_reserved, 2);
    }
}
