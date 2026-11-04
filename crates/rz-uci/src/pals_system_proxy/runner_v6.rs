//! Bounded supervisor composition for the original direct PALS child.
//!
//! This internal building block is not registered as an executable entry.
//! Its owner must remain alive when child or physical completion is unknown.

use super::{
    DirectPalsChildV6, PalsDirectChildInputPipeV6, PalsDirectChildObservationV6,
    PalsImageGrantObservationV6, PalsProxyBinaryPinV6, PalsProxyLaunchV6,
    cancel_control_v1::{CancelFailureObservation, CancelReceiverV1},
    finish::session_shutdown_report_v1::{self, ObservedSessionShutdownV1},
    finish::{self, PalsRuntimeFinishClaimV6, PalsSessionReleaseClaimV6},
    proxy_entry_smoke_cancel_v1::{NoGameCancelFailure, NoGameCancelReceiverV1},
    relay::{
        PalsUciRelayV6, RelayInputCleanupReportV6, RelayInputCleanupStartV6,
        RelayInputCleanupStopV6, RelayPipeObservationV6,
    },
};
use std::{
    any::Any,
    io,
    os::{fd::OwnedFd, unix::net::UnixStream},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ProxyStageV6 {
    Configured,
    Spawning,
    Spawned,
    StartingOutputs,
    OutputsStarted,
    Authenticating,
    Authenticated,
    GrantingImage,
    ImageAcknowledged,
    AttachingInput,
    Serving,
    Failed,
    Panicked,
}

/// The recorded claim remains separate from physical completion and may be
/// invalid, incomplete, or carry a failed acknowledgement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SessionReleaseObservationV6 {
    NotReady,
    ClaimRecorded,
    TransportUnknown,
    Panicked,
    AlreadyAttempted,
}

/// A same-child reported observation, never a device fence or release token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReportedSessionShutdownObservationV1 {
    NotReady,
    ObservationRecorded,
    TransportUnknown,
    Panicked,
    AlreadyAttempted,
}

/// A supervisor observation is never a device or session release proof.
pub(super) struct ProxySupervisorObservationV6 {
    pub stage: ProxyStageV6,
    pub child: Option<PalsDirectChildObservationV6>,
    pub relay: Option<[Option<RelayPipeObservationV6>; 3]>,
    pub input_cleanup_start: Option<RelayInputCleanupStartV6>,
    pub input_cleanup_stop: Option<RelayInputCleanupStopV6>,
    pub physical_completion: &'static str,
    /// Transport failure only; not a cancellation or physical completion.
    pub cancel_control_failure: Option<CancelFailureObservation>,
    pub no_game_cancel_failure: Option<NoGameCancelFailure>,
    pub cancel_service_failure: Option<String>,
}

/// Run the existing cancellation poll on the original borrowed receivers
/// during one bounded blocking exchange; the caller keeps their ownership.
pub(super) fn poll_cancel_during<T>(
    game: &mut Option<CancelReceiverV1>,
    no_game: &mut Option<NoGameCancelReceiverV1>,
    operation: impl FnOnce() -> T,
) -> io::Result<(std::thread::Result<T>, bool)> {
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("pals-cancel-drain".into())
            .spawn_scoped(scope, || {
                while !done.load(Ordering::Acquire) {
                    if let Some(receiver) = game.as_mut() {
                        let _ = receiver.poll();
                    }
                    if let Some(receiver) = no_game.as_mut() {
                        let _ = receiver.poll();
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
            })?;
        let result = catch_unwind(AssertUnwindSafe(operation));
        done.store(true, Ordering::Release);
        let worker_panicked = worker.join().is_err();
        Ok((result, worker_panicked))
    })
}

#[must_use = "retain the original child, relay and process while physical completion is unknown"]
pub(super) struct PalsProxySupervisorV6 {
    launch: Option<PalsProxyLaunchV6>,
    // Preserve the original binary FD even when launch fails.
    _binary: Arc<PalsProxyBinaryPinV6>,
    child: Option<DirectPalsChildV6>,
    relay: Option<PalsUciRelayV6>,
    incoming: Option<OwnedFd>,
    outgoing: Option<OwnedFd>,
    diagnostics: Option<OwnedFd>,
    unattached_input: Option<PalsDirectChildInputPipeV6>,
    run_until: Instant,
    finish_until: Instant,
    bound_reference: Instant,
    wall_time_max_ms: u64,
    cleanup_max_ms: u64,
    cancelled: Arc<AtomicBool>,
    finish_cancelled: Arc<AtomicBool>,
    // The exact channel remains with this original supervisor even on error.
    cancel_receiver: Option<CancelReceiverV1>,
    no_game_cancel_receiver: Option<NoGameCancelReceiverV1>,
    cancel_service_failure: Option<String>,
    max_line: usize,
    pub stage: ProxyStageV6,
    pub failed_stage: Option<ProxyStageV6>,
    pub failure: Option<io::Error>,
    pub panic_payload: Option<Box<dyn Any + Send>>,
    pub diagnostic_start_failure: Option<io::Error>,
    pub diagnostic_discard_started: bool,
    pub runtime_finish: Option<PalsRuntimeFinishClaimV6>,
    pub session_release: Option<PalsSessionReleaseClaimV6>,
    pub reported_session_shutdown: Option<ObservedSessionShutdownV1>,
    pub runtime_finish_error: Option<io::Error>,
    pub session_release_error: Option<io::Error>,
    pub runtime_finish_panic: Option<Box<dyn Any + Send>>,
    pub session_release_panic: Option<Box<dyn Any + Send>>,
    runtime_finish_stream: Option<UnixStream>,
    session_release_stream: Option<UnixStream>,
    runtime_finish_attempted: bool,
    session_release_attempted: bool,
    input_cleanup_start: Option<RelayInputCleanupStartV6>,
    input_cleanup_stop: Option<RelayInputCleanupStopV6>,
}

impl PalsProxySupervisorV6 {
    /// The three descriptors must be dedicated FIFO endpoints, with no other
    /// reader/writer aliases. Relay validation does not prove exclusivity.
    /// Deadlines are the caller's original registered bounds, not renewed.
    pub(super) fn new(
        launch: PalsProxyLaunchV6,
        incoming: OwnedFd,
        outgoing: OwnedFd,
        diagnostics: OwnedFd,
        max_line: usize,
        bound_reference: Instant,
        run_until: Instant,
        finish_until: Instant,
        cancelled: Arc<AtomicBool>,
        finish_cancelled: Arc<AtomicBool>,
    ) -> Self {
        Self {
            _binary: Arc::clone(&launch.binary),
            wall_time_max_ms: launch.lock.config.window.wall_time_max_ms,
            cleanup_max_ms: launch.lock.config.window.cleanup_max_ms,
            launch: Some(launch),
            child: None,
            relay: None,
            incoming: Some(incoming),
            outgoing: Some(outgoing),
            diagnostics: Some(diagnostics),
            unattached_input: None,
            run_until,
            finish_until,
            bound_reference,
            cancelled,
            finish_cancelled,
            cancel_receiver: None,
            no_game_cancel_receiver: None,
            cancel_service_failure: None,
            max_line,
            stage: ProxyStageV6::Configured,
            failed_stage: None,
            failure: None,
            panic_payload: None,
            diagnostic_start_failure: None,
            diagnostic_discard_started: false,
            runtime_finish: None,
            session_release: None,
            reported_session_shutdown: None,
            runtime_finish_error: None,
            session_release_error: None,
            runtime_finish_panic: None,
            session_release_panic: None,
            runtime_finish_stream: None,
            session_release_stream: None,
            runtime_finish_attempted: false,
            session_release_attempted: false,
            input_cleanup_start: None,
            input_cleanup_stop: None,
        }
    }

    /// The caller must first authenticate this dedicated channel to the
    /// original arena owner and role. Window identity alone is not that proof.
    /// On mismatch return the same receiver and stream without dropping them.
    pub(super) fn attach_caller_authenticated_cancel(
        &mut self,
        receiver: CancelReceiverV1,
    ) -> Result<(), CancelReceiverV1> {
        if self.stage != ProxyStageV6::Configured
            || self.cancel_receiver.is_some()
            || self.no_game_cancel_receiver.is_some()
            || !receiver.matches_window(&self.cancelled, self.finish_until)
        {
            return Err(receiver);
        }
        self.cancel_receiver = Some(receiver);
        Ok(())
    }

    /// A NoGame receiver must have been constructed from this same bundle's
    /// original Run grant. A separate caller owns listener provenance.
    pub(super) fn attach_no_game_cancel(
        &mut self,
        receiver: NoGameCancelReceiverV1,
    ) -> Result<(), NoGameCancelReceiverV1> {
        if self.stage != ProxyStageV6::Configured
            || self.cancel_receiver.is_some()
            || self.no_game_cancel_receiver.is_some()
            || !receiver.matches_window(&self.cancelled, self.finish_until)
        {
            return Err(receiver);
        }
        self.no_game_cancel_receiver = Some(receiver);
        Ok(())
    }

    fn poll_cancel_control(&mut self) {
        if let Some(receiver) = self.cancel_receiver.as_mut() {
            // A transport failure is retained on this receiver and reported
            // separately; output drain and child supervision must continue.
            let _ = receiver.poll();
        }
        if let Some(receiver) = self.no_game_cancel_receiver.as_mut() {
            // Preserve failure independently while continuing the same drain.
            let _ = receiver.poll();
        }
    }

    /// Continue polling the original receiver while a bounded startup or
    /// finish exchange waits. No descriptor is cloned: both receivers are
    /// returned to their original slots even when the operation unwinds.
    fn with_cancel_service<T>(&mut self, operation: impl FnOnce(&mut Self) -> T) -> io::Result<T> {
        if self.cancel_receiver.is_none() && self.no_game_cancel_receiver.is_none() {
            return Ok(operation(self));
        }
        let mut game = self.cancel_receiver.take();
        let mut no_game = self.no_game_cancel_receiver.take();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            poll_cancel_during(&mut game, &mut no_game, || operation(self))
        }));
        self.cancel_receiver = game;
        self.no_game_cancel_receiver = no_game;
        match outcome {
            Ok(Ok((result, worker_panicked))) => {
                if worker_panicked && self.cancel_service_failure.is_none() {
                    self.cancel_service_failure = Some("original cancel service panicked".into());
                }
                match result {
                    Ok(value) => Ok(value),
                    Err(payload) => std::panic::resume_unwind(payload),
                }
            }
            Ok(Err(error)) => Err(error),
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    fn check_run(&self) -> io::Result<()> {
        let registered_run_end = self
            .bound_reference
            .checked_add(Duration::from_millis(self.wall_time_max_ms));
        let registered_cleanup_end = self
            .run_until
            .checked_add(Duration::from_millis(self.cleanup_max_ms));
        if self.cancelled.load(Ordering::Acquire) {
            Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "PALS proxy cancelled",
            ))
        } else if self.bound_reference > Instant::now()
            || registered_run_end.is_none_or(|end| self.run_until > end)
            || registered_cleanup_end.is_none_or(|end| self.finish_until > end)
            || Instant::now() >= self.run_until
            || self.finish_until < self.run_until
        {
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "PALS proxy window expired or invalid",
            ))
        } else {
            Ok(())
        }
    }

    /// Run exactly one bounded startup attempt. Original child/relay and
    /// partial observations remain in this same object on error or unwind.
    /// Panic at a transfer boundary is still physical-custody unknown.
    pub(super) fn start_once(&mut self) -> ProxyStageV6 {
        if self.stage != ProxyStageV6::Configured {
            return self.stage;
        }
        self.poll_cancel_control();
        match self
            .with_cancel_service(|owner| catch_unwind(AssertUnwindSafe(|| owner.start_inner())))
        {
            Ok(Ok(Ok(()))) => self.stage,
            Ok(Ok(Err(error))) | Err(error) => {
                self.failed_stage = Some(self.stage);
                self.failure = Some(error);
                self.stage = ProxyStageV6::Failed;
                self.stage
            }
            Ok(Err(payload)) => {
                self.failed_stage = Some(self.stage);
                self.panic_payload = Some(payload);
                self.stage = ProxyStageV6::Panicked;
                self.stage
            }
        }
    }

    fn start_inner(&mut self) -> io::Result<()> {
        self.check_run()?;
        self.stage = ProxyStageV6::Spawning;
        let launch = self
            .launch
            .as_ref()
            .ok_or_else(|| io::Error::other("original launch unavailable"))?;
        if !Arc::ptr_eq(&launch.startup_cancelled, &self.cancelled) {
            return Err(io::Error::other(
                "proxy cancellation differs from original launch",
            ));
        }
        let mut launch = self.launch.take().expect("checked original launch");
        launch.startup_until = launch.startup_until.min(self.run_until);
        self.child = Some(DirectPalsChildV6::spawn(launch)?);
        self.stage = ProxyStageV6::Spawned;
        self.check_run()?;

        // Start output drains before waiting for authentication or readiness.
        let outputs = self
            .child
            .as_mut()
            .expect("spawned child")
            .take_output_pipes()?;
        let outgoing = self.outgoing.take().expect("original UCI output");
        let diagnostics = self.diagnostics.take().expect("original diagnostic output");
        self.relay = Some(PalsUciRelayV6::new(
            outputs,
            outgoing,
            diagnostics,
            self.max_line,
            self.finish_until,
        ));
        self.stage = ProxyStageV6::StartingOutputs;
        let started = self.relay.as_mut().expect("original relay").start_outputs();
        self.diagnostic_discard_started = started.stderr_discard_drain_started;
        if let Err(error) = started.stderr {
            self.diagnostic_start_failure = Some(error);
            if !self.diagnostic_discard_started {
                return Err(io::Error::other("diagnostic output drain could not start"));
            }
        }
        started.stdout?;
        self.stage = ProxyStageV6::OutputsStarted;
        self.check_run()?;

        self.stage = ProxyStageV6::Authenticating;
        self.child.as_mut().expect("spawned child").authenticate()?;
        self.stage = ProxyStageV6::Authenticated;
        self.check_run()?;
        self.stage = ProxyStageV6::GrantingImage;
        self.child
            .as_mut()
            .expect("spawned child")
            .grant_original_image()?;
        self.stage = ProxyStageV6::ImageAcknowledged;
        self.check_run()?;
        self.stage = ProxyStageV6::AttachingInput;
        let input = self.child.as_mut().expect("spawned child").take_stdin()?;
        let incoming = self.incoming.take().expect("original UCI input");
        match self
            .relay
            .as_mut()
            .expect("original relay")
            .attach_input(input, incoming)
        {
            Ok(()) => {}
            Err((input, incoming)) => {
                self.unattached_input = Some(input);
                self.incoming = Some(incoming);
                return Err(io::Error::other(
                    "original child input could not be attached",
                ));
            }
        }
        self.relay
            .as_mut()
            .expect("original relay")
            .start_input_until(self.run_until)?;
        self.stage = ProxyStageV6::Serving;
        Ok(())
    }

    /// One nonblocking observation. Cancellation stops new UCI input but
    /// leaves output draining to its original finite deadline.
    pub(super) fn poll(&mut self) -> io::Result<ProxySupervisorObservationV6> {
        self.poll_cancel_control();
        let cancel_control_failure = self
            .cancel_receiver
            .as_ref()
            .and_then(CancelReceiverV1::first_failure)
            .cloned();
        let no_game_cancel_failure = self
            .no_game_cancel_receiver
            .as_ref()
            .and_then(NoGameCancelReceiverV1::first_failure)
            .cloned();
        let mut cleanup_stop = None;
        let relay = self.relay.as_mut().map(|relay| {
            let observation = relay.poll();
            let input_ended = observation[0].as_ref().is_some_and(|value| !value.active);
            let should_cleanup = self.cancelled.load(Ordering::Acquire)
                || Instant::now() >= self.run_until
                || relay.input_quit_forwarded()
                || input_ended;
            if should_cleanup
                && matches!(
                    self.input_cleanup_start,
                    None | Some(RelayInputCleanupStartV6::WaitingForInputJoin)
                )
            {
                self.input_cleanup_start =
                    Some(relay.request_input_cleanup(
                        self.finish_until,
                        Arc::clone(&self.finish_cancelled),
                    ));
            }
            cleanup_stop = relay.poll_input_cleanup().map(|value| value.stop.clone());
            self.input_cleanup_stop = cleanup_stop.clone();
            observation
        });
        Ok(ProxySupervisorObservationV6 {
            stage: self.stage,
            // Child::try_wait can reap the process before a pending finish
            // claim passes the existing live-peer check. Reaping is explicit.
            child: None,
            relay,
            input_cleanup_start: self.input_cleanup_start,
            input_cleanup_stop: cleanup_stop,
            physical_completion: "unknown",
            cancel_control_failure,
            no_game_cancel_failure,
            cancel_service_failure: self.cancel_service_failure.clone(),
        })
    }

    /// Explicit process-exit observation after finish-claim collection has
    /// been attempted or intentionally abandoned by the original supervisor.
    /// `try_wait` can reap the child; exit is never physical completion.
    pub(super) fn observe_child_exit(
        &mut self,
    ) -> io::Result<Option<PalsDirectChildObservationV6>> {
        match self.child.as_mut() {
            Some(child) => child.observe().map(Some),
            None => Ok(None),
        }
    }

    /// Joining relay threads is not child or GPU completion. A timeout keeps
    /// every unjoined thread and original pipe in the same relay owner.
    pub(super) fn join_relay_until(&mut self, until: Instant) -> bool {
        let until = until.min(self.finish_until);
        loop {
            if self.poll().is_err() {
                return false;
            }
            let now = Instant::now();
            let Some(relay) = self.relay.as_mut() else {
                return true;
            };
            let step = now
                .checked_add(Duration::from_millis(5))
                .unwrap_or(until)
                .min(until);
            if relay.join_until(step) {
                // A worker can finish during join after the previous poll
                // reported WaitingForInputJoin. Poll again so its original
                // writer gets a cleanup opportunity before declaring relay
                // threads settled.
                if self.poll().is_err() {
                    return false;
                }
                let waiting_for_writer = matches!(
                    self.input_cleanup_start,
                    Some(RelayInputCleanupStartV6::WaitingForInputJoin)
                );
                let settled = self
                    .relay
                    .as_mut()
                    .is_none_or(|relay| relay.join_until(Instant::now()));
                if !waiting_for_writer && settled {
                    return true;
                }
            }
            if Instant::now() >= until {
                return false;
            }
        }
    }

    fn finish_read_allowed(&self) -> bool {
        let input_retired = matches!(
            self.input_cleanup_stop.as_ref(),
            Some(
                RelayInputCleanupStopV6::QuitAlreadyForwarded
                    | RelayInputCleanupStopV6::StopAndQuitForwarded
            )
        );
        let still_needs_poll = self.relay.is_some()
            && matches!(
                self.input_cleanup_start,
                None | Some(RelayInputCleanupStartV6::WaitingForInputJoin)
            );
        !still_needs_poll && (input_retired || Instant::now() >= self.run_until)
    }

    /// The exact report retains partial byte counts, late-write facts and
    /// panic payload. It is not evidence of child or GPU completion.
    pub(super) fn input_cleanup_report(&mut self) -> Option<&RelayInputCleanupReportV6> {
        self.relay
            .as_mut()
            .and_then(PalsUciRelayV6::poll_input_cleanup)
    }

    /// The stream must be an exclusive, already authenticated same-child
    /// runtime-report finish stream. The direct-child accept path below is
    /// for the separate session-release event, not this report transport.
    pub(super) fn observe_runtime_finish_once(
        &mut self,
        stream: UnixStream,
    ) -> Result<(), UnixStream> {
        if self.runtime_finish_attempted || !self.finish_read_allowed() {
            return Err(stream);
        }
        self.runtime_finish_attempted = true;
        self.runtime_finish_stream = Some(stream);
        let result = self.with_cancel_service(|owner| {
            catch_unwind(AssertUnwindSafe(|| {
                match (&mut owner.child, &mut owner.runtime_finish_stream) {
                    (Some(child), Some(stream)) => finish::observe_finish_claim(
                        child,
                        stream,
                        owner.finish_until,
                        owner.finish_cancelled.as_ref(),
                    ),
                    _ => Err(io::Error::other(
                        "original child or finish stream unavailable",
                    )),
                }
            }))
        });
        match result {
            Ok(Ok(Ok(claim))) => self.runtime_finish = Some(claim),
            Ok(Ok(Err(error))) | Err(error) => self.runtime_finish_error = Some(error),
            Ok(Err(payload)) => self.runtime_finish_panic = Some(payload),
        }
        Ok(())
    }

    /// Accept and acknowledge the original child's one session claim through
    /// a distinct authenticated finish stream. A claim is not a GPU fence.
    /// Partial handshake/frame/ACK failure is sticky and is not retried.
    /// This blocks until the cleanup deadline: its caller must first retire
    /// input (including delivering `quit`) or independently enforce the run
    /// deadline while the finish channel is being serviced.
    pub(super) fn observe_session_release_once(&mut self) -> SessionReleaseObservationV6 {
        if self.session_release_attempted {
            return SessionReleaseObservationV6::AlreadyAttempted;
        }
        // A synchronous finish read must not prevent the run deadline from
        // being supervised. Before that deadline, wait until the single
        // input writer actually forwarded quit; afterwards continue trying
        // to preserve any independent release evidence even on input error.
        if !self.finish_read_allowed() {
            // Continue polling until the single input writer is recovered or
            // a terminal cleanup state is known before blocking on a claim.
            return SessionReleaseObservationV6::NotReady;
        }
        match self
            .child
            .as_ref()
            .map(DirectPalsChildV6::image_grant_observation)
        {
            None | Some(PalsImageGrantObservationV6::NotAttempted) => {
                return SessionReleaseObservationV6::NotReady;
            }
            Some(PalsImageGrantObservationV6::Failed { .. }) => {
                self.session_release_attempted = true;
                self.session_release_error = Some(io::Error::other(
                    "original image grant failed; physical delivery unknown",
                ));
                return SessionReleaseObservationV6::TransportUnknown;
            }
            Some(PalsImageGrantObservationV6::Acknowledged) => {}
        }
        self.session_release_attempted = true;
        let result = self.with_cancel_service(|owner| {
            catch_unwind(AssertUnwindSafe(|| {
                let child = owner
                    .child
                    .as_mut()
                    .ok_or_else(|| io::Error::other("original child unavailable"))?;
                let stream = child.accept_original_finish_stream(
                    owner.finish_until,
                    owner.finish_cancelled.as_ref(),
                )?;
                owner.session_release_stream = Some(stream);
                let stream = owner
                    .session_release_stream
                    .as_mut()
                    .expect("original finish stream retained");
                finish::observe_session_release_claim(
                    child,
                    stream,
                    owner.finish_until,
                    owner.finish_cancelled.as_ref(),
                )
            }))
        });
        match result {
            Ok(Ok(Ok(claim))) => {
                self.session_release = Some(claim);
                SessionReleaseObservationV6::ClaimRecorded
            }
            Ok(Ok(Err(error))) | Err(error) => {
                self.session_release_error = Some(error);
                SessionReleaseObservationV6::TransportUnknown
            }
            Ok(Err(payload)) => {
                self.session_release_panic = Some(payload);
                SessionReleaseObservationV6::Panicked
            }
        }
    }

    /// Select the separate report protocol for this same one-shot finish
    /// channel. v1 and this protocol cannot both consume the original stream.
    /// ACK or a reported observation is not a physical release capability.
    pub(super) fn observe_reported_session_shutdown_once(
        &mut self,
    ) -> ReportedSessionShutdownObservationV1 {
        if self.session_release_attempted {
            return ReportedSessionShutdownObservationV1::AlreadyAttempted;
        }
        if !self.finish_read_allowed() {
            return ReportedSessionShutdownObservationV1::NotReady;
        }
        match self
            .child
            .as_ref()
            .map(DirectPalsChildV6::image_grant_observation)
        {
            None | Some(PalsImageGrantObservationV6::NotAttempted) => {
                return ReportedSessionShutdownObservationV1::NotReady;
            }
            Some(PalsImageGrantObservationV6::Failed { .. }) => {
                self.session_release_attempted = true;
                self.session_release_error = Some(io::Error::other(
                    "original image grant failed; physical delivery unknown",
                ));
                return ReportedSessionShutdownObservationV1::TransportUnknown;
            }
            Some(PalsImageGrantObservationV6::Acknowledged) => {}
        }
        self.session_release_attempted = true;
        let result = self.with_cancel_service(|owner| {
            catch_unwind(AssertUnwindSafe(|| {
                let child = owner
                    .child
                    .as_mut()
                    .ok_or_else(|| io::Error::other("original child unavailable"))?;
                let stream = child.accept_original_finish_stream(
                    owner.finish_until,
                    owner.finish_cancelled.as_ref(),
                )?;
                owner.session_release_stream = Some(stream);
                let stream = owner
                    .session_release_stream
                    .as_mut()
                    .expect("original finish stream retained");
                session_shutdown_report_v1::observe_report(
                    child,
                    stream,
                    owner.finish_until,
                    owner.finish_cancelled.as_ref(),
                )
            }))
        });
        match result {
            Ok(Ok(Ok(observation))) => {
                self.reported_session_shutdown = Some(observation);
                ReportedSessionShutdownObservationV1::ObservationRecorded
            }
            Ok(Ok(Err(error))) | Err(error) => {
                self.session_release_error = Some(error);
                ReportedSessionShutdownObservationV1::TransportUnknown
            }
            Ok(Err(payload)) => {
                self.session_release_panic = Some(payload);
                ReportedSessionShutdownObservationV1::Panicked
            }
        }
    }
}
