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

#[derive(Clone, Copy)]
enum Framing {
    UciLines,
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
}

fn fifo(fd: &OwnedFd) -> io::Result<()> {
    let stat = fstat(fd).map_err(io::Error::from)?;
    if !SFlag::from_bits_truncate(stat.st_mode).contains(SFlag::S_IFIFO) {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "relay requires a Linux pipe",
        ));
    }
    Ok(())
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
    job.line.clear();
    job.write_pos = 0;
    job.line_ready = false;
    Ok(())
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
        if output_failure.is_some() {
            // Continue draining the still-live child after a consumer failure.
            // The original buffers and output pipe stay owned even on expiry.
            job.read_pos = job.read_len;
            match fill(job, until, cancelled) {
                Ok(true) => continue,
                Ok(false) => {
                    return observation(
                        job,
                        output_failure.expect("checked failure"),
                        false,
                        Some(true),
                    );
                }
                Err(stop) => {
                    let mut result = observation(
                        job,
                        output_failure.expect("checked failure"),
                        false,
                        Some(false),
                    );
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
            Framing::UciLines => rest
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(rest.len(), |i| i + 1),
            Framing::RawDiagnostics => rest.len(),
        };
        if matches!(job.framing, Framing::UciLines)
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
}

impl RelayLeg {
    fn new(
        source: OwnedFd,
        destination: OwnedFd,
        framing: Framing,
        drain_after_error: bool,
    ) -> Self {
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
            }))),
            cancelled: Arc::new(AtomicBool::new(false)),
            issue: Arc::new(Mutex::new(None)),
            thread: None,
            observation: None,
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
        if !(1..=MAX_LINE_LIMIT).contains(&max_line) || Instant::now() >= until {
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
            Framing::UciLines,
            false,
        ));
        Ok(())
    }

    pub fn start_input(&mut self) -> io::Result<()> {
        self.input
            .as_mut()
            .ok_or_else(|| io::Error::other("authenticated child input not attached"))?
            .start("pals-uci-input", self.max_line, self.until, None)
    }

    /// Stop forwarding input without closing its original pipe. Output drains
    /// stay independent so native cleanup can continue reporting safely.
    pub fn cancel_input(&self) {
        if let Some(input) = &self.input {
            input.cancel();
        }
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
            if self.input.as_ref().is_none_or(RelayLeg::finished)
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
