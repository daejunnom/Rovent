//! Bounded Linux pipe relay for a directly owned PALS child.
//!
//! The arena owns the chess clock. This relay never fabricates a UCI response
//! or interprets an I/O result as GPU completion. The original pipe owners
//! remain in their slots on stop, timeout, error and caught thread panic.

use super::{PalsDirectChildInputPipeV6, PalsDirectChildOutputPipesV6};
use nix::{
    errno::Errno,
    poll::{PollFd, PollFlags, poll},
    sys::signal::{SigSet, SigmaskHow, Signal, pthread_sigmask},
    sys::stat::{SFlag, fstat},
    unistd::{read, write},
};
use std::{
    io,
    os::fd::{AsFd, OwnedFd},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

// POSIX's minimum PIPE_BUF. A write no larger than this after POLLOUT has
// bounded pipe capacity under the exclusive-writer contract below.
const WRITE_CHUNK: usize = 512;
const READ_CHUNK: usize = 4096;
const MAX_LINE_LIMIT: usize = 1024 * 1024;
const POLL_MS: u16 = 50;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RelayStopV6 {
    Eof,
    Cancelled,
    Deadline,
    LineTooLong,
    IncompleteLine,
    Io(String),
    Panicked,
}

/// Counts are actual bytes successfully written, not search work or a fence.
#[derive(Clone, Debug)]
pub struct RelayPipeObservationV6 {
    pub bytes_forwarded: u64,
    pub stop: RelayStopV6,
    /// False/unknown when the source has not reached EOF after a sink error.
    pub source_drained_after_error: Option<bool>,
    /// Separate cancellation/deadline/read failure following a sink error.
    pub drain_stop: Option<RelayStopV6>,
    pub active: bool,
}

/// A single cleanup attempt on the original input writer. The counts are
/// bytes actually written; neither a complete quit nor a pipe close is a
/// native-worker or GPU completion observation.
pub struct RelayInputCleanupReportV6 {
    pub stop: RelayInputCleanupStopV6,
    pub original_quit_forwarded: bool,
    pub pending_line_bytes_forwarded: usize,
    pub stop_bytes_forwarded: usize,
    pub quit_bytes_forwarded: usize,
    /// A write returned after the original cleanup deadline or cancellation;
    /// the bytes still occurred and are never silently retried.
    pub write_observed_after_deadline: bool,
    pub write_observed_after_cancel: bool,
    pub panic_payload: Option<Box<dyn std::any::Any + Send>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RelayInputCleanupStopV6 {
    QuitAlreadyForwarded,
    StopAndQuitForwarded,
    InconsistentPendingLine,
    InputOwnershipUnknown,
    Stopped(RelayStopV6),
    Panicked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelayInputCleanupStartV6 {
    WaitingForInputJoin,
    Started,
    AlreadyStarted,
    NoInput,
    InvalidControl,
    Expired,
    Cancelled,
    SpawnFailed,
    InputOwnershipUnknown,
    QuitAlreadyForwarded,
}

#[derive(Clone, Copy)]
enum Framing {
    UciLines,
    UciInput,
    RawDiagnostics,
}

struct RelayJob {
    source: OwnedFd,
    destination: OwnedFd,
    framing: Framing,
    drain_after_error: bool,
    // These buffers and their cursors stay in the same owner after a stop.
    read_buffer: [u8; READ_CHUNK],
    read_len: usize,
    read_pos: usize,
    line: Vec<u8>,
    line_ready: bool,
    write_pos: usize,
    forwarded: u64,
    /// Set only after the original complete UCI quit line is fully written.
    quit_forwarded: bool,
    quit_observation: Arc<AtomicBool>,
}

pub(super) fn fifo(fd: &OwnedFd) -> io::Result<()> {
    let stat = fstat(fd).map_err(io::Error::from)?;
    if !SFlag::from_bits_truncate(stat.st_mode).contains(SFlag::S_IFIFO) {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "relay requires a Linux pipe",
        ));
    }
    Ok(())
}

pub(super) fn validate_line_bound(max_line: usize) -> io::Result<()> {
    if (1..=MAX_LINE_LIMIT).contains(&max_line) {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "relay line bound invalid",
        ))
    }
}

/// Pre-spawn check for the primary UCI pipes. Diagnostic stderr keeps its
/// independent discard-drain fallback and is intentionally not checked here.
pub(super) fn validate_proxy_descriptors(
    incoming: &OwnedFd,
    outgoing: &OwnedFd,
    max_line: usize,
) -> io::Result<()> {
    validate_line_bound(max_line)?;
    fifo(incoming)?;
    fifo(outgoing)
}

fn is_quit_line(line: &[u8]) -> bool {
    let Some(line) = line.strip_suffix(b"\n") else {
        return false;
    };
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    if line
        .iter()
        .any(|byte| !byte.is_ascii() || (*byte < b' ' && *byte != b'\t') || *byte == 0x7f)
    {
        return false;
    }
    let mut fields = line
        .split(|byte| *byte == b' ' || *byte == b'\t')
        .filter(|field| !field.is_empty());
    fields.next() == Some(&b"quit"[..]) && fields.next().is_none()
}

fn check_control(until: Instant, cancelled: &AtomicBool) -> Result<(), RelayStopV6> {
    if cancelled.load(Ordering::Acquire) {
        return Err(RelayStopV6::Cancelled);
    }
    if Instant::now() >= until {
        return Err(RelayStopV6::Deadline);
    }
    Ok(())
}

fn wait_ready(
    fd: &OwnedFd,
    flags: PollFlags,
    until: Instant,
    cancelled: &AtomicBool,
) -> Result<(), RelayStopV6> {
    loop {
        check_control(until, cancelled)?;
        let now = Instant::now();
        if now >= until {
            return Err(RelayStopV6::Deadline);
        }
        let mut descriptor = [PollFd::new(fd.as_fd(), flags)];
        let wait_ms = until
            .saturating_duration_since(now)
            .as_millis()
            .min(u128::from(POLL_MS)) as u16;
        match poll(&mut descriptor, wait_ms) {
            Ok(0) => continue,
            Ok(_) => {
                let events = descriptor[0].revents().unwrap_or(PollFlags::empty());
                if events.contains(PollFlags::POLLNVAL) {
                    return Err(RelayStopV6::Io("relay descriptor invalid".into()));
                }
                if flags == PollFlags::POLLOUT
                    && events.intersects(PollFlags::POLLERR | PollFlags::POLLHUP)
                {
                    return Err(RelayStopV6::Io("relay destination closed".into()));
                }
                if events.intersects(flags | PollFlags::POLLERR | PollFlags::POLLHUP) {
                    return check_control(until, cancelled);
                }
            }
            Err(Errno::EINTR) => continue,
            Err(error) => return Err(RelayStopV6::Io(error.to_string())),
        }
    }
}

fn fill(job: &mut RelayJob, until: Instant, cancelled: &AtomicBool) -> Result<bool, RelayStopV6> {
    loop {
        wait_ready(&job.source, PollFlags::POLLIN, until, cancelled)?;
        check_control(until, cancelled)?;
        match read(&job.source, &mut job.read_buffer) {
            Ok(0) => return Ok(false),
            Ok(n) => {
                job.read_pos = 0;
                job.read_len = n;
                return Ok(true);
            }
            Err(Errno::EINTR | Errno::EAGAIN) => continue,
            Err(error) => return Err(RelayStopV6::Io(error.to_string())),
        }
    }
}

fn flush_line(
    job: &mut RelayJob,
    until: Instant,
    cancelled: &AtomicBool,
) -> Result<(), RelayStopV6> {
    while job.write_pos < job.line.len() {
        wait_ready(&job.destination, PollFlags::POLLOUT, until, cancelled)?;
        check_control(until, cancelled)?;
        let end = job
            .line
            .len()
            .min(job.write_pos.saturating_add(WRITE_CHUNK));
        match write(&job.destination, &job.line[job.write_pos..end]) {
            Ok(0) => return Err(RelayStopV6::Io("relay write returned zero".into())),
            Ok(n) => {
                job.write_pos += n;
                job.forwarded = job.forwarded.saturating_add(n as u64);
            }
            Err(Errno::EINTR | Errno::EAGAIN) => continue,
            Err(error) => return Err(RelayStopV6::Io(error.to_string())),
        }
    }
    if matches!(job.framing, Framing::UciInput) && is_quit_line(&job.line) {
        job.quit_forwarded = true;
        job.quit_observation.store(true, Ordering::Release);
    }
    job.line.clear();
    job.write_pos = 0;
    job.line_ready = false;
    Ok(())
}

fn write_cleanup_bytes(
    destination: &OwnedFd,
    bytes: &[u8],
    written: &mut usize,
    until: Instant,
    cancelled: &AtomicBool,
    late: &mut bool,
    cancelled_after_write: &mut bool,
) -> Result<(), RelayStopV6> {
    while *written < bytes.len() {
        wait_ready(destination, PollFlags::POLLOUT, until, cancelled)?;
        check_control(until, cancelled)?;
        match write(destination, &bytes[*written..]) {
            Ok(0) => return Err(RelayStopV6::Io("cleanup write returned zero".into())),
            Ok(count) => {
                *written += count;
                *late |= Instant::now() >= until;
                *cancelled_after_write |= cancelled.load(Ordering::Acquire);
            }
            Err(Errno::EINTR | Errno::EAGAIN) => continue,
            Err(error) => return Err(RelayStopV6::Io(error.to_string())),
        }
    }
    Ok(())
}

fn run_input_cleanup(
    job: &mut RelayJob,
    report: &mut RelayInputCleanupReportV6,
    until: Instant,
    cancelled: &AtomicBool,
) {
    report.original_quit_forwarded = job.quit_forwarded;
    if job.quit_forwarded {
        report.stop = RelayInputCleanupStopV6::QuitAlreadyForwarded;
        return;
    }
    if job.write_pos > 0 {
        if !job.line_ready || job.write_pos > job.line.len() {
            report.stop = RelayInputCleanupStopV6::InconsistentPendingLine;
            return;
        }
        let before = job.write_pos;
        let remaining = job.line.len() - before;
        match flush_line(job, until, cancelled) {
            Ok(()) => report.pending_line_bytes_forwarded = remaining,
            Err(error) => {
                report.pending_line_bytes_forwarded = job.write_pos.saturating_sub(before);
                if report.pending_line_bytes_forwarded > 0 {
                    report.write_observed_after_deadline |= Instant::now() >= until;
                    report.write_observed_after_cancel |= cancelled.load(Ordering::Acquire);
                }
                report.stop = RelayInputCleanupStopV6::Stopped(error);
                return;
            }
        }
        if report.pending_line_bytes_forwarded > 0 {
            report.write_observed_after_deadline |= Instant::now() >= until;
            report.write_observed_after_cancel |= cancelled.load(Ordering::Acquire);
        }
        // A quit whose first bytes were already delivered must not be sent a
        // second time after finishing that exact original line.
        if job.quit_forwarded {
            report.original_quit_forwarded = true;
            report.stop = RelayInputCleanupStopV6::QuitAlreadyForwarded;
            return;
        }
    }
    // A line not yet written, including a partly read line, stays in the
    // original buffer for the caller; no bytes are spliced into that line.
    if let Err(error) = write_cleanup_bytes(
        &job.destination,
        b"stop\n",
        &mut report.stop_bytes_forwarded,
        until,
        cancelled,
        &mut report.write_observed_after_deadline,
        &mut report.write_observed_after_cancel,
    ) {
        report.stop = RelayInputCleanupStopV6::Stopped(error);
        return;
    }
    if let Err(error) = write_cleanup_bytes(
        &job.destination,
        b"quit\n",
        &mut report.quit_bytes_forwarded,
        until,
        cancelled,
        &mut report.write_observed_after_deadline,
        &mut report.write_observed_after_cancel,
    ) {
        report.stop = RelayInputCleanupStopV6::Stopped(error);
        return;
    }
    job.quit_forwarded = true;
    job.quit_observation.store(true, Ordering::Release);
    report.stop = RelayInputCleanupStopV6::StopAndQuitForwarded;
}

fn observation(
    job: &RelayJob,
    stop: RelayStopV6,
    active: bool,
    drained: Option<bool>,
) -> RelayPipeObservationV6 {
    RelayPipeObservationV6 {
        bytes_forwarded: job.forwarded,
        stop,
        source_drained_after_error: drained,
        drain_stop: None,
        active,
    }
}

fn run_job(
    job: &mut RelayJob,
    max_line: usize,
    until: Instant,
    cancelled: &AtomicBool,
    issue: &Mutex<Option<RelayPipeObservationV6>>,
    initial_failure: Option<RelayStopV6>,
) -> RelayPipeObservationV6 {
    let mut output_failure = initial_failure;
    if let Some(failure) = &output_failure {
        if let Ok(mut value) = issue.lock() {
            *value = Some(observation(job, failure.clone(), true, Some(false)));
        }
    }
    loop {
        if let Some(failure) = output_failure.take() {
            // Continue draining the still-live child after a consumer failure.
            // The original buffers and output pipe stay owned even on expiry.
            job.read_pos = job.read_len;
            match fill(job, until, cancelled) {
                Ok(true) => {
                    output_failure = Some(failure);
                    continue;
                }
                Ok(false) => {
                    return observation(job, failure, false, Some(true));
                }
                Err(stop) => {
                    let mut result = observation(job, failure, false, Some(false));
                    result.drain_stop = Some(stop);
                    return result;
                }
            }
        }
        if job.line_ready {
            if let Err(error) = flush_line(job, until, cancelled) {
                if job.drain_after_error && matches!(error, RelayStopV6::Io(_)) {
                    if let Ok(mut value) = issue.lock() {
                        *value = Some(observation(job, error.clone(), true, Some(false)));
                    }
                    output_failure = Some(error);
                    continue;
                }
                return observation(job, error, false, None);
            }
            continue;
        }
        if job.read_pos == job.read_len {
            match fill(job, until, cancelled) {
                Ok(true) => {}
                Ok(false) if job.line.is_empty() => {
                    return observation(job, RelayStopV6::Eof, false, None);
                }
                Ok(false) => return observation(job, RelayStopV6::IncompleteLine, false, None),
                Err(error) => return observation(job, error, false, None),
            }
        }
        let rest = &job.read_buffer[job.read_pos..job.read_len];
        let length = match job.framing {
            Framing::UciLines | Framing::UciInput => rest
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(rest.len(), |i| i + 1),
            Framing::RawDiagnostics => rest.len(),
        };
        if matches!(job.framing, Framing::UciLines | Framing::UciInput)
            && job.line.len().saturating_add(length) > max_line
        {
            let error = RelayStopV6::LineTooLong;
            if job.drain_after_error {
                if let Ok(mut value) = issue.lock() {
                    *value = Some(observation(job, error.clone(), true, Some(false)));
                }
                output_failure = Some(error);
                continue;
            }
            return observation(job, error, false, None);
        }
        job.line.extend_from_slice(&rest[..length]);
        job.read_pos += length;
        job.line_ready =
            matches!(job.framing, Framing::RawDiagnostics) || job.line.last() == Some(&b'\n');
    }
}

struct RelayLeg {
    job: Arc<Mutex<Option<RelayJob>>>,
    cancelled: Arc<AtomicBool>,
    issue: Arc<Mutex<Option<RelayPipeObservationV6>>>,
    thread: Option<JoinHandle<RelayPipeObservationV6>>,
    observation: Option<RelayPipeObservationV6>,
    quit_observation: Arc<AtomicBool>,
}

impl RelayLeg {
    fn new(
        source: OwnedFd,
        destination: OwnedFd,
        framing: Framing,
        drain_after_error: bool,
    ) -> Self {
        let quit_observation = Arc::new(AtomicBool::new(false));
        Self {
            job: Arc::new(Mutex::new(Some(RelayJob {
                source,
                destination,
                framing,
                drain_after_error,
                read_buffer: [0; READ_CHUNK],
                read_len: 0,
                read_pos: 0,
                line: Vec::new(),
                line_ready: false,
                write_pos: 0,
                forwarded: 0,
                quit_forwarded: false,
                quit_observation: Arc::clone(&quit_observation),
            }))),
            cancelled: Arc::new(AtomicBool::new(false)),
            issue: Arc::new(Mutex::new(None)),
            thread: None,
            observation: None,
            quit_observation,
        }
    }

    fn start(
        &mut self,
        name: &'static str,
        max_line: usize,
        until: Instant,
        discard_reason: Option<String>,
    ) -> io::Result<()> {
        if self.thread.is_some() || self.observation.is_some() {
            return Ok(());
        }
        {
            let slot = self
                .job
                .lock()
                .map_err(|_| io::Error::other("relay owner poisoned"))?;
            let job = slot
                .as_ref()
                .ok_or_else(|| io::Error::other("relay owner unavailable"))?;
            fifo(&job.source)?;
            if discard_reason.is_none() {
                fifo(&job.destination)?;
            }
        }
        validate_line_bound(max_line)?;
        if Instant::now() >= until {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "relay line or deadline bound invalid",
            ));
        }
        let job = Arc::clone(&self.job);
        let cancelled = Arc::clone(&self.cancelled);
        let issue = Arc::clone(&self.issue);
        self.thread = Some(thread::Builder::new().name(name.into()).spawn(move || {
            let mut slot = match job.lock() {
                Ok(slot) => slot,
                Err(_) => {
                    return RelayPipeObservationV6 {
                        bytes_forwarded: 0,
                        stop: RelayStopV6::Io("relay owner poisoned".into()),
                        source_drained_after_error: None,
                        drain_stop: None,
                        active: false,
                    };
                }
            };
            let Some(mut owned) = slot.take() else {
                return RelayPipeObservationV6 {
                    bytes_forwarded: 0,
                    stop: RelayStopV6::Io("relay owner unavailable".into()),
                    source_drained_after_error: None,
                    drain_stop: None,
                    active: false,
                };
            };
            drop(slot);
            // Block SIGPIPE on this dedicated thread only; a reader may close
            // after POLLOUT and before write. EPIPE remains an I/O result and
            // cannot terminate the whole process or discard native custody.
            let mut blocked = SigSet::empty();
            blocked.add(Signal::SIGPIPE);
            let result = match pthread_sigmask(SigmaskHow::SIG_BLOCK, Some(&blocked), None) {
                Err(error) => observation(&owned, RelayStopV6::Io(error.to_string()), false, None),
                Ok(()) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_job(
                        &mut owned,
                        max_line,
                        until,
                        &cancelled,
                        &issue,
                        discard_reason.map(RelayStopV6::Io),
                    )
                }))
                .unwrap_or_else(|_| observation(&owned, RelayStopV6::Panicked, false, None)),
            };
            // No original FD or partial data is dropped when the relay stops.
            // A poisoned mutex is recovered to retain the same original job.
            *job.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(owned);
            result
        })?);
        Ok(())
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    fn poll(&mut self) -> Option<RelayPipeObservationV6> {
        if self.thread.as_ref().is_some_and(JoinHandle::is_finished) {
            let worker = self.thread.take().expect("checked finished relay");
            self.observation = Some(worker.join().unwrap_or(RelayPipeObservationV6 {
                bytes_forwarded: 0,
                stop: RelayStopV6::Panicked,
                source_drained_after_error: None,
                drain_stop: None,
                active: false,
            }));
        }
        self.observation
            .clone()
            .or_else(|| self.issue.lock().ok().and_then(|issue| issue.clone()))
    }

    fn finished(&self) -> bool {
        self.thread.is_none()
    }
}

/// Relay FDs must be original dedicated pipes: one reader per input and one
/// writer per output, with no competing aliases. `fstat` can check FIFO type,
/// but does not prove an anonymous pipe or the absence of competing aliases;
/// those remain the original supervisor's obligation. This contract lets poll plus
/// bounded PIPE_BUF writes avoid changing shared file status flags. The caller
/// must also avoid writing through buffered stdout/stderr aliases. Direct FD
/// writes have no user-space buffer to flush; each complete UCI line is sent
/// immediately and byte-for-byte, including its original CR/LF bytes.
#[must_use = "retain original pipes and threads until their lifecycle is resolved"]
pub struct PalsUciRelayV6 {
    input: Option<RelayLeg>,
    output: RelayLeg,
    stderr: RelayLeg,
    max_line: usize,
    until: Instant,
    owner_scope: Arc<()>,
    input_cleanup_control: Option<(Instant, Arc<AtomicBool>)>,
    input_cleanup_thread: Option<JoinHandle<RelayInputCleanupReportV6>>,
    input_cleanup_report: Option<RelayInputCleanupReportV6>,
    input_cleanup_spawn_error: Option<io::Error>,
}

/// Starting stderr diagnostics cannot turn a successful primary UCI pipe
/// start into a failure. `stderr` records the actual sink problem even when
/// a separate bounded discard drain could be started to protect the child.
pub struct RelayOutputsStartV6 {
    pub stdout: io::Result<()>,
    pub stderr: io::Result<()>,
    pub stderr_discard_drain_started: bool,
}

impl PalsUciRelayV6 {
    pub fn new(
        outputs: PalsDirectChildOutputPipesV6,
        outgoing_uci: OwnedFd,
        diagnostic_stderr: OwnedFd,
        max_line: usize,
        until: Instant,
    ) -> Self {
        Self {
            input: None,
            output: RelayLeg::new(outputs.stdout.into(), outgoing_uci, Framing::UciLines, true),
            stderr: RelayLeg::new(
                outputs.stderr.into(),
                diagnostic_stderr,
                Framing::RawDiagnostics,
                true,
            ),
            max_line,
            until,
            owner_scope: outputs.owner_scope,
            input_cleanup_control: None,
            input_cleanup_thread: None,
            input_cleanup_report: None,
            input_cleanup_spawn_error: None,
        }
    }

    /// May start before authentication so the child cannot deadlock on output.
    pub fn start_outputs(&mut self) -> RelayOutputsStartV6 {
        let stdout = self
            .output
            .start("pals-uci-output", self.max_line, self.until, None);
        let stderr = self
            .stderr
            .start("pals-uci-stderr", self.max_line, self.until, None);
        let mut stderr_discard_drain_started = false;
        if let Err(error) = &stderr {
            if error.kind() == io::ErrorKind::Unsupported {
                stderr_discard_drain_started = self
                    .stderr
                    .start(
                        "pals-uci-stderr-drain",
                        self.max_line,
                        self.until,
                        Some(format!("stderr sink unavailable: {error}")),
                    )
                    .is_ok();
            }
        }
        RelayOutputsStartV6 {
            stdout,
            stderr,
            stderr_discard_drain_started,
        }
    }

    /// The executor only obtains this original stdin after child authentication.
    /// If already attached, return both supplied handles rather than dropping
    /// either one and potentially causing an unintended child EOF.
    pub fn attach_input(
        &mut self,
        stdin: PalsDirectChildInputPipeV6,
        incoming_uci: OwnedFd,
    ) -> Result<(), (PalsDirectChildInputPipeV6, OwnedFd)> {
        if self.input.is_some() || !Arc::ptr_eq(&self.owner_scope, &stdin.owner_scope) {
            return Err((stdin, incoming_uci));
        }
        self.input = Some(RelayLeg::new(
            incoming_uci,
            stdin.stdin.into(),
            Framing::UciInput,
            false,
        ));
        Ok(())
    }

    pub fn start_input(&mut self) -> io::Result<()> {
        self.start_input_until(self.until)
    }

    /// Input stops on the original run deadline even while output drains
    /// continue under the separate cleanup deadline.
    pub fn start_input_until(&mut self, until: Instant) -> io::Result<()> {
        if self.input_cleanup_control.is_some() {
            return Err(io::Error::other("input cleanup already registered"));
        }
        if until > self.until {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "input deadline exceeds relay deadline",
            ));
        }
        self.input
            .as_mut()
            .ok_or_else(|| io::Error::other("authenticated child input not attached"))?
            .start("pals-uci-input", self.max_line, until, None)
    }

    /// Stop forwarding input without closing its original pipe. Output drains
    /// stay independent so native cleanup can continue reporting safely.
    pub fn cancel_input(&self) {
        if let Some(input) = &self.input {
            input.cancel();
        }
    }

    /// Observes only a complete quit line written to this exact stdin. It is
    /// not evidence of search, process, native-worker or GPU completion.
    pub fn input_quit_forwarded(&self) -> bool {
        self.input
            .as_ref()
            .is_some_and(|input| input.quit_observation.load(Ordering::Acquire))
    }

    /// Stop the original input worker, then use its same writer for one
    /// cleanup attempt. This never waits for the input worker to join; retry
    /// only while `WaitingForInputJoin` with this exact deadline/token.
    pub fn request_input_cleanup(
        &mut self,
        until: Instant,
        cleanup_cancelled: Arc<AtomicBool>,
    ) -> RelayInputCleanupStartV6 {
        if self.input_cleanup_thread.is_some()
            || self.input_cleanup_report.is_some()
            || self.input_cleanup_spawn_error.is_some()
        {
            return RelayInputCleanupStartV6::AlreadyStarted;
        }
        let Some(input) = self.input.as_mut() else {
            return RelayInputCleanupStartV6::NoInput;
        };
        if until > self.until || Arc::ptr_eq(&cleanup_cancelled, &input.cancelled) {
            return RelayInputCleanupStartV6::InvalidControl;
        }
        if let Some((original_until, original_cancelled)) = &self.input_cleanup_control {
            if *original_until != until || !Arc::ptr_eq(original_cancelled, &cleanup_cancelled) {
                return RelayInputCleanupStartV6::InvalidControl;
            }
        } else {
            self.input_cleanup_control = Some((until, Arc::clone(&cleanup_cancelled)));
        }
        input.cancel();
        let _ = input.poll();
        if input.thread.is_some() {
            if cleanup_cancelled.load(Ordering::Acquire) {
                return RelayInputCleanupStartV6::Cancelled;
            }
            if Instant::now() >= until {
                return RelayInputCleanupStartV6::Expired;
            }
            return RelayInputCleanupStartV6::WaitingForInputJoin;
        }
        let quit_forwarded = match input.job.try_lock() {
            Ok(slot) => match slot.as_ref() {
                Some(job) => job.quit_forwarded,
                None => return RelayInputCleanupStartV6::InputOwnershipUnknown,
            },
            Err(_) => return RelayInputCleanupStartV6::InputOwnershipUnknown,
        };
        if quit_forwarded {
            self.input_cleanup_report = Some(RelayInputCleanupReportV6 {
                stop: RelayInputCleanupStopV6::QuitAlreadyForwarded,
                original_quit_forwarded: true,
                pending_line_bytes_forwarded: 0,
                stop_bytes_forwarded: 0,
                quit_bytes_forwarded: 0,
                write_observed_after_deadline: false,
                write_observed_after_cancel: false,
                panic_payload: None,
            });
            return RelayInputCleanupStartV6::QuitAlreadyForwarded;
        }
        if cleanup_cancelled.load(Ordering::Acquire) {
            return RelayInputCleanupStartV6::Cancelled;
        }
        if Instant::now() >= until {
            return RelayInputCleanupStartV6::Expired;
        }
        if input
            .observation
            .as_ref()
            .is_some_and(|item| matches!(&item.stop, RelayStopV6::Panicked))
        {
            self.input_cleanup_report = Some(RelayInputCleanupReportV6 {
                stop: RelayInputCleanupStopV6::InputOwnershipUnknown,
                original_quit_forwarded: false,
                pending_line_bytes_forwarded: 0,
                stop_bytes_forwarded: 0,
                quit_bytes_forwarded: 0,
                write_observed_after_deadline: false,
                write_observed_after_cancel: false,
                panic_payload: None,
            });
            return RelayInputCleanupStartV6::InputOwnershipUnknown;
        }
        let job = Arc::clone(&input.job);
        let started = thread::Builder::new()
            .name("pals-uci-input-cleanup".into())
            .spawn(move || {
                let mut report = RelayInputCleanupReportV6 {
                    stop: RelayInputCleanupStopV6::InputOwnershipUnknown,
                    original_quit_forwarded: false,
                    pending_line_bytes_forwarded: 0,
                    stop_bytes_forwarded: 0,
                    quit_bytes_forwarded: 0,
                    write_observed_after_deadline: false,
                    write_observed_after_cancel: false,
                    panic_payload: None,
                };
                let mut slot = match job.lock() {
                    Ok(slot) => slot,
                    Err(_) => return report,
                };
                let Some(job) = slot.as_mut() else {
                    return report;
                };
                // Keep SIGPIPE blocked for the lifetime of this dedicated thread;
                // restoring the mask after EPIPE could deliver a pending SIGPIPE.
                let mut blocked = SigSet::empty();
                blocked.add(Signal::SIGPIPE);
                if let Err(error) = pthread_sigmask(SigmaskHow::SIG_BLOCK, Some(&blocked), None) {
                    report.stop =
                        RelayInputCleanupStopV6::Stopped(RelayStopV6::Io(error.to_string()));
                    return report;
                }
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_input_cleanup(job, &mut report, until, &cleanup_cancelled)
                })) {
                    Ok(()) => report,
                    Err(payload) => {
                        report.stop = RelayInputCleanupStopV6::Panicked;
                        report.panic_payload = Some(payload);
                        report
                    }
                }
            });
        match started {
            Ok(thread) => {
                self.input_cleanup_thread = Some(thread);
                RelayInputCleanupStartV6::Started
            }
            Err(error) => {
                self.input_cleanup_spawn_error = Some(error);
                RelayInputCleanupStartV6::SpawnFailed
            }
        }
    }

    /// Nonblocking observation; a still-running cleanup worker and its
    /// original writer remain held by this relay.
    pub fn poll_input_cleanup(&mut self) -> Option<&RelayInputCleanupReportV6> {
        if self
            .input_cleanup_thread
            .as_ref()
            .is_some_and(JoinHandle::is_finished)
        {
            let thread = self
                .input_cleanup_thread
                .take()
                .expect("checked cleanup thread");
            self.input_cleanup_report = Some(match thread.join() {
                Ok(report) => report,
                Err(payload) => RelayInputCleanupReportV6 {
                    stop: RelayInputCleanupStopV6::Panicked,
                    original_quit_forwarded: false,
                    pending_line_bytes_forwarded: 0,
                    stop_bytes_forwarded: 0,
                    quit_bytes_forwarded: 0,
                    write_observed_after_deadline: false,
                    write_observed_after_cancel: false,
                    panic_payload: Some(payload),
                },
            });
        }
        self.input_cleanup_report.as_ref()
    }

    pub fn input_cleanup_spawn_error(&self) -> Option<&io::Error> {
        self.input_cleanup_spawn_error.as_ref()
    }

    /// Call only when the original supervisor has decided to stop output
    /// drains. This operation never claims that the child or GPU has finished.
    pub fn cancel_output_drains(&self) {
        self.output.cancel();
        self.stderr.cancel();
    }

    pub fn poll(&mut self) -> [Option<RelayPipeObservationV6>; 3] {
        [
            self.input.as_mut().and_then(RelayLeg::poll),
            self.output.poll(),
            self.stderr.poll(),
        ]
    }

    /// Bounded join: true means no relay thread is currently running, not
    /// that every leg started or that the child/GPU completed. False leaves
    /// every still-running handle and FD owned by this value.
    pub fn join_until(&mut self, until: Instant) -> bool {
        loop {
            let _ = self.poll();
            let _ = self.poll_input_cleanup();
            if self.input.as_ref().is_none_or(RelayLeg::finished)
                && self.input_cleanup_thread.is_none()
                && self.output.finished()
                && self.stderr.finished()
            {
                return true;
            }
            if Instant::now() >= until {
                return false;
            }
            thread::sleep(
                Duration::from_millis(5).min(until.saturating_duration_since(Instant::now())),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nix::unistd::pipe;

    fn read_exact_pipe(fd: &OwnedFd, length: usize) -> Vec<u8> {
        let until = Instant::now() + Duration::from_secs(2);
        let mut bytes = vec![0; length];
        let mut offset = 0;
        while offset < length {
            wait_ready(fd, PollFlags::POLLIN, until, &AtomicBool::new(false)).unwrap();
            let count = read(fd, &mut bytes[offset..]).unwrap();
            assert!(count > 0);
            offset += count;
        }
        bytes
    }

    fn cleanup_report() -> RelayInputCleanupReportV6 {
        RelayInputCleanupReportV6 {
            stop: RelayInputCleanupStopV6::InputOwnershipUnknown,
            original_quit_forwarded: false,
            pending_line_bytes_forwarded: 0,
            stop_bytes_forwarded: 0,
            quit_bytes_forwarded: 0,
            write_observed_after_deadline: false,
            write_observed_after_cancel: false,
            panic_payload: None,
        }
    }

    fn input_relay_fixture() -> (PalsUciRelayV6, OwnedFd, OwnedFd) {
        let (incoming, incoming_writer) = pipe().unwrap();
        let (child_reader, child_input) = pipe().unwrap();
        let (output_source, _output_source_writer) = pipe().unwrap();
        let (_output_reader, output_sink) = pipe().unwrap();
        let (stderr_source, _stderr_source_writer) = pipe().unwrap();
        let (_stderr_reader, stderr_sink) = pipe().unwrap();
        let relay = PalsUciRelayV6 {
            input: Some(RelayLeg::new(
                incoming,
                child_input,
                Framing::UciInput,
                false,
            )),
            output: RelayLeg::new(output_source, output_sink, Framing::UciLines, true),
            stderr: RelayLeg::new(stderr_source, stderr_sink, Framing::RawDiagnostics, true),
            max_line: 1024,
            until: Instant::now() + Duration::from_secs(3),
            owner_scope: Arc::new(()),
            input_cleanup_control: None,
            input_cleanup_thread: None,
            input_cleanup_report: None,
            input_cleanup_spawn_error: None,
        };
        (relay, incoming_writer, child_reader)
    }

    #[test]
    fn input_worker_hands_same_pipe_to_bounded_cleanup_after_join() {
        let (mut relay, incoming_writer, child_reader) = input_relay_fixture();
        let run_until = relay.until;
        relay.start_input_until(run_until).unwrap();
        let original = b"uci\nstop\nucinewgame\n";
        write(&incoming_writer, original).unwrap();
        assert_eq!(
            read_exact_pipe(&child_reader, original.len()).as_slice(),
            original
        );
        let cleanup_until = relay.until;
        let cleanup_cancelled = Arc::new(AtomicBool::new(false));
        loop {
            match relay.request_input_cleanup(cleanup_until, Arc::clone(&cleanup_cancelled)) {
                RelayInputCleanupStartV6::WaitingForInputJoin => {
                    assert!(Instant::now() < cleanup_until);
                    thread::sleep(Duration::from_millis(1));
                }
                RelayInputCleanupStartV6::Started => break,
                other => panic!("cleanup did not start: {other:?}"),
            }
        }
        loop {
            if let Some(report) = relay.poll_input_cleanup() {
                assert_eq!(report.stop, RelayInputCleanupStopV6::StopAndQuitForwarded);
                assert_eq!(report.stop_bytes_forwarded, b"stop\n".len());
                assert_eq!(report.quit_bytes_forwarded, b"quit\n".len());
                break;
            }
            assert!(Instant::now() < cleanup_until);
            thread::sleep(Duration::from_millis(1));
        }
        assert!(relay.input_quit_forwarded());
        assert_eq!(
            read_exact_pipe(&child_reader, b"stop\nquit\n".len()).as_slice(),
            b"stop\nquit\n"
        );
        assert_eq!(
            relay.request_input_cleanup(cleanup_until, cleanup_cancelled),
            RelayInputCleanupStartV6::AlreadyStarted
        );
    }

    #[test]
    fn original_quit_is_observed_before_input_worker_ends() {
        let (mut relay, incoming_writer, child_reader) = input_relay_fixture();
        relay.start_input_until(relay.until).unwrap();
        write(&incoming_writer, b" \tqu").unwrap();
        assert!(!relay.input_quit_forwarded());
        write(&incoming_writer, b"it \r\n").unwrap();
        assert_eq!(
            read_exact_pipe(&child_reader, b" \tquit \r\n".len()).as_slice(),
            b" \tquit \r\n"
        );
        let until = relay.until;
        while !relay.input_quit_forwarded() {
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
        assert!(
            relay
                .input
                .as_ref()
                .unwrap()
                .thread
                .as_ref()
                .is_some_and(|thread| !thread.is_finished())
        );
        let cancelled = Arc::new(AtomicBool::new(false));
        loop {
            match relay.request_input_cleanup(until, Arc::clone(&cancelled)) {
                RelayInputCleanupStartV6::WaitingForInputJoin => {
                    assert!(Instant::now() < until);
                    thread::sleep(Duration::from_millis(1));
                }
                RelayInputCleanupStartV6::QuitAlreadyForwarded => break,
                other => panic!("original quit not retained: {other:?}"),
            }
        }
        let report = relay.poll_input_cleanup().unwrap();
        assert_eq!(report.stop, RelayInputCleanupStopV6::QuitAlreadyForwarded);
        assert_eq!(report.quit_bytes_forwarded, 0);
        let mut descriptor = [PollFd::new(child_reader.as_fd(), PollFlags::POLLIN)];
        assert_eq!(poll(&mut descriptor, 0_u16).unwrap(), 0);
    }

    #[test]
    fn registered_cleanup_prevents_later_input_restart() {
        let (mut relay, _incoming_writer, child_reader) = input_relay_fixture();
        let until = relay.until;
        let cancelled = Arc::new(AtomicBool::new(false));
        assert_eq!(
            relay.request_input_cleanup(until, Arc::clone(&cancelled)),
            RelayInputCleanupStartV6::Started
        );
        assert!(relay.start_input_until(until).is_err());
        loop {
            if let Some(report) = relay.poll_input_cleanup() {
                assert_eq!(report.stop, RelayInputCleanupStopV6::StopAndQuitForwarded);
                break;
            }
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(
            read_exact_pipe(&child_reader, b"stop\nquit\n".len()).as_slice(),
            b"stop\nquit\n"
        );
    }

    #[test]
    fn cleanup_closed_child_pipe_returns_error_without_process_exit() {
        let (mut relay, _incoming_writer, child_reader) = input_relay_fixture();
        drop(child_reader);
        let until = relay.until;
        assert_eq!(
            relay.request_input_cleanup(until, Arc::new(AtomicBool::new(false))),
            RelayInputCleanupStartV6::Started
        );
        loop {
            if let Some(report) = relay.poll_input_cleanup() {
                assert!(matches!(
                    &report.stop,
                    RelayInputCleanupStopV6::Stopped(RelayStopV6::Io(_))
                ));
                assert_eq!(report.quit_bytes_forwarded, 0);
                break;
            }
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
        assert!(!relay.input_quit_forwarded());
    }

    #[test]
    fn cleanup_reports_cancellation_and_deadline_while_input_still_owned() {
        let (mut cancelled_relay, _writer, _reader) = input_relay_fixture();
        cancelled_relay
            .start_input_until(Instant::now() + Duration::from_secs(2))
            .unwrap();
        let cancelled = Arc::new(AtomicBool::new(true));
        assert_eq!(
            cancelled_relay
                .request_input_cleanup(Instant::now() + Duration::from_secs(1), cancelled),
            RelayInputCleanupStartV6::Cancelled
        );
        assert!(cancelled_relay.join_until(Instant::now() + Duration::from_secs(2)));

        let (mut expired_relay, _writer, _reader) = input_relay_fixture();
        expired_relay
            .start_input_until(Instant::now() + Duration::from_secs(2))
            .unwrap();
        assert_eq!(
            expired_relay.request_input_cleanup(Instant::now(), Arc::new(AtomicBool::new(false))),
            RelayInputCleanupStartV6::Expired
        );
        assert!(expired_relay.join_until(Instant::now() + Duration::from_secs(2)));
    }

    #[test]
    fn cleanup_finishes_partially_forwarded_line_before_control() {
        let (source, _source_writer) = pipe().unwrap();
        let (sink_reader, sink) = pipe().unwrap();
        write(&sink, b"go ").unwrap();
        let leg = RelayLeg::new(source, sink, Framing::UciInput, false);
        let mut slot = leg.job.lock().unwrap();
        let job = slot.as_mut().unwrap();
        job.line = b"go nodes 2\n".to_vec();
        job.line_ready = true;
        job.write_pos = 3;
        let mut report = cleanup_report();
        run_input_cleanup(
            job,
            &mut report,
            Instant::now() + Duration::from_secs(2),
            &AtomicBool::new(false),
        );
        assert_eq!(report.stop, RelayInputCleanupStopV6::StopAndQuitForwarded);
        assert_eq!(report.pending_line_bytes_forwarded, b"nodes 2\n".len());
        assert_eq!(
            read_exact_pipe(&sink_reader, b"go nodes 2\nstop\nquit\n".len()).as_slice(),
            b"go nodes 2\nstop\nquit\n"
        );
    }

    #[test]
    fn cleanup_cancel_keeps_unforwarded_input_and_sends_no_control() {
        let (source, _source_writer) = pipe().unwrap();
        let (sink_reader, sink) = pipe().unwrap();
        let leg = RelayLeg::new(source, sink, Framing::UciInput, false);
        let mut slot = leg.job.lock().unwrap();
        let job = slot.as_mut().unwrap();
        job.line = b"go nodes 7\n".to_vec();
        job.line_ready = true;
        let mut report = cleanup_report();
        run_input_cleanup(
            job,
            &mut report,
            Instant::now() + Duration::from_secs(2),
            &AtomicBool::new(true),
        );
        assert_eq!(
            report.stop,
            RelayInputCleanupStopV6::Stopped(RelayStopV6::Cancelled)
        );
        assert_eq!(job.line.as_slice(), b"go nodes 7\n");
        let mut descriptor = [PollFd::new(sink_reader.as_fd(), PollFlags::POLLIN)];
        assert_eq!(poll(&mut descriptor, 0_u16).unwrap(), 0);
    }

    fn finished(leg: &mut RelayLeg) -> RelayPipeObservationV6 {
        let until = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(observation) = leg.poll() {
                if !observation.active {
                    return observation;
                }
            }
            assert!(Instant::now() < until, "relay did not finish in time");
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn complete_uci_lines_are_forwarded_byte_for_byte() {
        let (source, source_writer) = pipe().unwrap();
        let (destination_reader, destination) = pipe().unwrap();
        let mut leg = RelayLeg::new(source, destination, Framing::UciLines, false);
        leg.start(
            "relay-test-bytes",
            256,
            Instant::now() + Duration::from_secs(2),
            None,
        )
        .unwrap();
        let input = b"uci\r\nisready\n";
        assert_eq!(write(&source_writer, input).unwrap(), input.len());
        drop(source_writer);
        let mut received = Vec::new();
        let until = Instant::now() + Duration::from_secs(2);
        while received.len() < input.len() {
            wait_ready(
                &destination_reader,
                PollFlags::POLLIN,
                until,
                &AtomicBool::new(false),
            )
            .unwrap();
            let mut bytes = [0; 64];
            let n = read(&destination_reader, &mut bytes).unwrap();
            assert!(n > 0);
            received.extend_from_slice(&bytes[..n]);
        }
        assert_eq!(received, input);
        let report = finished(&mut leg);
        assert_eq!(report.stop, RelayStopV6::Eof);
        assert_eq!(report.bytes_forwarded, input.len() as u64);
    }

    #[test]
    fn idle_input_cancel_retains_original_pipes_and_finishes() {
        let (source, source_writer) = pipe().unwrap();
        let (_destination_reader, destination) = pipe().unwrap();
        let mut leg = RelayLeg::new(source, destination, Framing::UciLines, false);
        leg.start(
            "relay-test-cancel",
            128,
            Instant::now() + Duration::from_secs(2),
            None,
        )
        .unwrap();
        leg.cancel();
        assert_eq!(finished(&mut leg).stop, RelayStopV6::Cancelled);
        let slot = leg.job.lock().unwrap();
        let retained = slot.as_ref().expect("original pipe owner retained");
        assert!(fstat(&retained.source).is_ok());
        assert!(fstat(&retained.destination).is_ok());
        assert_eq!(write(&source_writer, b"still open\n").unwrap(), 11);
    }

    #[test]
    fn oversized_input_is_not_forwarded_and_pipe_stays_owned() {
        let (source, source_writer) = pipe().unwrap();
        let (destination_reader, destination) = pipe().unwrap();
        let mut leg = RelayLeg::new(source, destination, Framing::UciLines, false);
        leg.start(
            "relay-test-bound",
            4,
            Instant::now() + Duration::from_secs(2),
            None,
        )
        .unwrap();
        write(&source_writer, b"12345\n").unwrap();
        let report = finished(&mut leg);
        assert_eq!(report.stop, RelayStopV6::LineTooLong);
        assert_eq!(report.bytes_forwarded, 0);
        let mut descriptor = [PollFd::new(destination_reader.as_fd(), PollFlags::POLLIN)];
        assert_eq!(poll(&mut descriptor, 0_u16).unwrap(), 0);
        assert!(leg.job.lock().unwrap().is_some());
    }

    #[test]
    fn closed_output_is_recorded_and_child_output_is_drained() {
        let (source, source_writer) = pipe().unwrap();
        let (destination_reader, destination) = pipe().unwrap();
        drop(destination_reader);
        let mut leg = RelayLeg::new(source, destination, Framing::UciLines, true);
        leg.start(
            "relay-test-drain",
            128,
            Instant::now() + Duration::from_secs(2),
            None,
        )
        .unwrap();
        write(&source_writer, b"info test\n").unwrap();
        drop(source_writer);
        let report = finished(&mut leg);
        assert!(matches!(report.stop, RelayStopV6::Io(_)));
        assert_eq!(report.source_drained_after_error, Some(true));
        assert!(leg.job.lock().unwrap().is_some());
    }

    #[test]
    fn unsupported_diagnostic_sink_can_be_recorded_and_drained() {
        let (source, source_writer) = pipe().unwrap();
        let unsupported: OwnedFd = std::fs::File::open("/dev/null").unwrap().into();
        let mut leg = RelayLeg::new(source, unsupported, Framing::RawDiagnostics, true);
        let until = Instant::now() + Duration::from_secs(2);
        assert_eq!(
            leg.start("relay-test-stderr", 128, until, None)
                .unwrap_err()
                .kind(),
            io::ErrorKind::Unsupported,
        );
        leg.start(
            "relay-test-stderr-drain",
            128,
            until,
            Some("stderr unavailable".into()),
        )
        .unwrap();
        write(&source_writer, b"diagnostic\n").unwrap();
        drop(source_writer);
        let report = finished(&mut leg);
        assert_eq!(report.stop, RelayStopV6::Io("stderr unavailable".into()));
        assert_eq!(report.source_drained_after_error, Some(true));
        assert!(leg.job.lock().unwrap().is_some());
    }
}
