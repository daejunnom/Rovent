//! Opt-in UCI reducer with bounded asynchronous output admission.
//!
//! FIFO output is retained while the writer queue is full. Normal events are
//! not dequeued in that state; a separate caller-owned control closure remains
//! observable at a caller-selected finite tick. This closure is not itself an
//! engine admission latch, a physical cancellation acknowledgment, or a GPU
//! fence. Production wiring must supply the original session-close authority
//! and retain the same driver and output owner on every unfinished exit.

use super::async_output::{
    AsyncOutputOwner, OutputBatch, OutputEncodingError, OutputRecovery, OutputSnapshot,
    SubmitError, encode_event_output,
};
use super::{Event, handle_event};
use crate::{Effect, PositionPort, Session, SessionResult};
use std::any::Any;
use std::collections::VecDeque;
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::thread;
use std::time::Duration;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsyncControl {
    Continue,
    /// Yield the same live driver without closing its session or admission.
    Yield,
    /// Ask the original reducer to close; queued output still remains ordered.
    Shutdown,
    /// Close the reducer and return any still-pending output to its caller.
    ReturnPending,
    /// The original control/owner may have failed. Return without invoking
    /// any reducer or dispatch again; session and physical state stay unknown.
    RetainWithoutClose,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsyncServeStatus {
    /// All event outputs were queued and ordered drain was requested. This is
    /// not writer completion, physical completion, or delivery evidence.
    OutputQueued,
    /// The caller requested return with original pending evidence retained.
    Yielded,
    /// Caller retains the still-open or otherwise unknown original session.
    RetainedWithoutClose,
    Failed,
}

#[derive(Debug)]
pub enum AsyncDispatchFailure<E> {
    Error(E),
    /// The effect was handed to the callback; its physical outcome is unknown.
    Panic(Box<dyn Any + Send>),
}

#[derive(Debug)]
pub enum AsyncServeFailure<E> {
    InvalidPollInterval,
    OriginalStateSealed,
    ControlPanic(Box<dyn Any + Send>),
    HandlerPanic(Box<dyn Any + Send>),
    Dispatch(AsyncDispatchFailure<E>),
    Encoding(OutputEncodingError),
    OutputTooLarge,
    OutputClosed,
    OutputWorkerFailed,
}

/// The original reducer result and, once prepared, the same unsent byte batch.
#[derive(Debug)]
pub struct PendingEvent<S> {
    pub original: SessionResult<S>,
    pub encoded: Option<OutputBatch>,
}

#[must_use]
pub struct AsyncServeExit<E> {
    pub status: AsyncServeStatus,
    pub failure: Option<AsyncServeFailure<E>>,
    pub cleanup_failures: Vec<AsyncServeFailure<E>>,
}

/// This describes only the original output thread, not engine/GPU completion.
pub enum OutputRecoveryPoll<O, D> {
    DriverNotSealed,
    Pending,
    Recovered(OutputRecovery<O, D>),
    WorkerPanicked(Box<dyn Any + Send>),
    AlreadyRecovered,
}

/// Pins one original session, event receiver, output owner and reducer pair
/// across yields. The pending FIFO and close-issued state never move to a new
/// invocation or a caller-supplied replacement owner.
#[must_use]
pub struct AsyncServeDriver<P, O, D, H, F>
where
    P: PositionPort,
    O: Write + Send + 'static,
    D: Write + Send + 'static,
{
    session: Session<P>,
    events: Receiver<Event>,
    output: Option<AsyncOutputOwner<O, D>>,
    max_output_bytes: usize,
    poll_interval: Duration,
    handle: H,
    dispatch: F,
    pending: VecDeque<PendingEvent<P::Snapshot>>,
    shutdown_requested: bool,
    input_disconnected: bool,
    sealed: bool,
    final_output_observation: Option<OutputSnapshot>,
}

/// Explicit recovery of the same original resources; `sealed` does not grant
/// permission to restart or infer physical completion.
pub struct AsyncServeParts<P, O, D, H, F>
where
    P: PositionPort,
    O: Write + Send + 'static,
    D: Write + Send + 'static,
{
    pub session: Session<P>,
    pub events: Receiver<Event>,
    pub output: Option<AsyncOutputOwner<O, D>>,
    pub handle: H,
    pub dispatch: F,
    pub pending: VecDeque<PendingEvent<P::Snapshot>>,
    pub shutdown_requested: bool,
    pub input_disconnected: bool,
    pub sealed: bool,
    pub final_output_observation: Option<OutputSnapshot>,
}

impl<P, O, D, H, F> AsyncServeDriver<P, O, D, H, F>
where
    P: PositionPort,
    O: Write + Send + 'static,
    D: Write + Send + 'static,
    H: FnMut(&mut Session<P>, Event) -> SessionResult<P::Snapshot>,
{
    pub fn new(
        session: Session<P>,
        events: Receiver<Event>,
        output: AsyncOutputOwner<O, D>,
        max_output_bytes: usize,
        poll_interval: Duration,
        handle: H,
        dispatch: F,
    ) -> Self {
        Self {
            session,
            events,
            output: Some(output),
            max_output_bytes,
            poll_interval,
            handle,
            dispatch,
            pending: VecDeque::new(),
            shutdown_requested: false,
            input_disconnected: false,
            sealed: false,
            final_output_observation: None,
        }
    }

    /// Borrow original pending records without transferring or reissuing them.
    pub fn pending(&self) -> &VecDeque<PendingEvent<P::Snapshot>> {
        &self.pending
    }
    pub fn session_closed(&self) -> bool {
        self.session.is_closed()
    }
    pub fn shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }
    pub fn output(&self) -> Option<&AsyncOutputOwner<O, D>> {
        self.output.as_ref()
    }

    /// Original shared snapshot captured after the writer was actually joined.
    /// It remains distinct from the join result and physical/GPU completion.
    pub fn final_output_observation(&self) -> Option<OutputSnapshot> {
        self.final_output_observation
    }

    /// Never waits. A still-running output owner is restored into this same
    /// driver unchanged; session, pending FIFO and close state never move.
    pub fn try_finish_output(&mut self) -> OutputRecoveryPoll<O, D> {
        if !self.sealed {
            return OutputRecoveryPoll::DriverNotSealed;
        }
        let Some(owner) = self.output.take() else {
            return OutputRecoveryPoll::AlreadyRecovered;
        };
        match owner.try_recover_with_final_snapshot() {
            Err(original) => {
                self.output = Some(original);
                OutputRecoveryPoll::Pending
            }
            Ok((snapshot, Ok(recovered))) => {
                self.final_output_observation = Some(snapshot);
                OutputRecoveryPoll::Recovered(recovered)
            }
            Ok((snapshot, Err(payload))) => {
                self.final_output_observation = Some(snapshot);
                OutputRecoveryPoll::WorkerPanicked(payload)
            }
        }
    }

    /// Explicitly hand back all original owners and pending records. This is
    /// neither output delivery nor permission to release an active writer.
    pub fn into_parts(self) -> AsyncServeParts<P, O, D, H, F> {
        AsyncServeParts {
            session: self.session,
            events: self.events,
            output: self.output,
            handle: self.handle,
            dispatch: self.dispatch,
            pending: self.pending,
            shutdown_requested: self.shutdown_requested,
            input_disconnected: self.input_disconnected,
            sealed: self.sealed,
            final_output_observation: self.final_output_observation,
        }
    }

    /// A normal Yielded result may resume only through this same driver.
    /// Failed/unknown results seal it against implicit retries.
    pub fn drive<C, E>(&mut self, poll: C) -> AsyncServeExit<E>
    where
        F: FnMut(Effect<P::Snapshot>) -> Result<(), E>,
        C: FnMut() -> AsyncControl,
    {
        if self.sealed {
            return AsyncServeExit {
                status: AsyncServeStatus::Failed,
                failure: Some(AsyncServeFailure::OriginalStateSealed),
                cleanup_failures: Vec::new(),
            };
        }
        let Some(output) = self.output.as_ref() else {
            self.sealed = true;
            return AsyncServeExit {
                status: AsyncServeStatus::Failed,
                failure: Some(AsyncServeFailure::OriginalStateSealed),
                cleanup_failures: Vec::new(),
            };
        };
        let exit = serve_events_with_async_handler(
            &mut self.session,
            &self.events,
            output,
            self.max_output_bytes,
            self.poll_interval,
            poll,
            &mut self.handle,
            &mut self.dispatch,
            &mut self.pending,
            &mut self.shutdown_requested,
            &mut self.input_disconnected,
        );
        if exit.status != AsyncServeStatus::Yielded {
            self.sealed = true;
        }
        exit
    }
}

fn dispatch_result<S, F, E>(
    mut result: SessionResult<S>,
    dispatch: &mut F,
) -> (
    PendingEvent<S>,
    Option<AsyncDispatchFailure<E>>,
    Vec<AsyncDispatchFailure<E>>,
)
where
    F: FnMut(Effect<S>) -> Result<(), E>,
{
    let mut failure = None;
    let mut cleanup = Vec::new();
    let mut dispatch_panicked = false;
    for effect in std::mem::take(&mut result.effects) {
        if failure.is_some() && (dispatch_panicked || !matches!(effect, Effect::Shutdown)) {
            // The original effect was not dispatched after a failure.
            result.effects.push(effect);
            continue;
        }
        let dispatched = catch_unwind(AssertUnwindSafe(|| dispatch(effect)));
        let error = match dispatched {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(AsyncDispatchFailure::Error(error)),
            Err(payload) => {
                dispatch_panicked = true;
                Some(AsyncDispatchFailure::Panic(payload))
            }
        };
        if let Some(error) = error {
            if failure.is_none() {
                failure = Some(error);
            } else {
                cleanup.push(error);
            }
        }
    }
    (
        PendingEvent {
            original: result,
            encoded: None,
        },
        failure,
        cleanup,
    )
}

fn close_original<P, H, F, E>(
    session: &mut Session<P>,
    handle: &mut H,
    dispatch: &mut F,
    pending: &mut VecDeque<PendingEvent<P::Snapshot>>,
    cleanup: &mut Vec<AsyncServeFailure<E>>,
) -> Option<AsyncServeFailure<E>>
where
    P: PositionPort,
    H: FnMut(&mut Session<P>, Event) -> SessionResult<P::Snapshot>,
    F: FnMut(Effect<P::Snapshot>) -> Result<(), E>,
{
    if session.is_closed() {
        return None;
    }
    let handled = catch_unwind(AssertUnwindSafe(|| handle(session, Event::EndOfInput)));
    let result = match handled {
        Ok(result) => result,
        Err(payload) => return Some(AsyncServeFailure::HandlerPanic(payload)),
    };
    let mut failure = None;
    // Respect the original Owner if it deferred EOF for an outstanding
    // physical result. Unlike the legacy synchronous error fallback, this
    // opt-in path retains that Owner and does not force the Session closed.
    let (event, dispatch_failure, extra) = dispatch_result(result, dispatch);
    pending.push_back(event);
    cleanup.extend(extra.into_iter().map(AsyncServeFailure::Dispatch));
    if let Some(error) = dispatch_failure {
        if failure.is_none() {
            failure = Some(AsyncServeFailure::Dispatch(error));
        } else {
            cleanup.push(AsyncServeFailure::Dispatch(error));
        }
    }
    failure
}

fn close_once<P, H, F, E>(
    session: &mut Session<P>,
    handle: &mut H,
    dispatch: &mut F,
    pending: &mut VecDeque<PendingEvent<P::Snapshot>>,
    cleanup: &mut Vec<AsyncServeFailure<E>>,
    issued: &mut bool,
) -> Option<AsyncServeFailure<E>>
where
    P: PositionPort,
    H: FnMut(&mut Session<P>, Event) -> SessionResult<P::Snapshot>,
    F: FnMut(Effect<P::Snapshot>) -> Result<(), E>,
{
    if *issued || session.is_closed() {
        return None;
    }
    // A panicking handler remains an issued but unknown attempt, not a
    // reason to issue EOF a second time.
    *issued = true;
    close_original(session, handle, dispatch, pending, cleanup)
}

/// Internal driver step: the caller retains the same borrowed sources and FIFO.
/// `poll` must be nonblocking; no output or physical completion is implied.
fn serve_events_with_async_handler<P, O, D, H, F, C, E>(
    session: &mut Session<P>,
    events: &Receiver<Event>,
    output: &AsyncOutputOwner<O, D>,
    max_output_bytes: usize,
    poll_interval: Duration,
    mut poll: C,
    mut handle: &mut H,
    mut dispatch: &mut F,
    mut pending: &mut VecDeque<PendingEvent<P::Snapshot>>,
    shutdown_requested: &mut bool,
    input_disconnected: &mut bool,
) -> AsyncServeExit<E>
where
    P: PositionPort,
    O: Write + Send + 'static,
    D: Write + Send + 'static,
    H: FnMut(&mut Session<P>, Event) -> SessionResult<P::Snapshot>,
    F: FnMut(Effect<P::Snapshot>) -> Result<(), E>,
    C: FnMut() -> AsyncControl,
{
    let mut cleanup = Vec::new();
    if poll_interval.is_zero() {
        return AsyncServeExit {
            status: AsyncServeStatus::Failed,
            failure: Some(AsyncServeFailure::InvalidPollInterval),
            cleanup_failures: cleanup,
        };
    }
    loop {
        let control = match catch_unwind(AssertUnwindSafe(&mut poll)) {
            Ok(control) => control,
            Err(payload) => {
                return AsyncServeExit {
                    status: AsyncServeStatus::Failed,
                    failure: Some(AsyncServeFailure::ControlPanic(payload)),
                    cleanup_failures: cleanup,
                };
            }
        };
        if control == AsyncControl::Yield {
            return AsyncServeExit {
                status: AsyncServeStatus::Yielded,
                failure: None,
                cleanup_failures: cleanup,
            };
        }
        if control == AsyncControl::RetainWithoutClose {
            return AsyncServeExit {
                status: AsyncServeStatus::RetainedWithoutClose,
                failure: None,
                cleanup_failures: cleanup,
            };
        }
        if control != AsyncControl::Continue {
            if let Some(error) = close_once(
                session,
                &mut handle,
                &mut dispatch,
                &mut pending,
                &mut cleanup,
                shutdown_requested,
            ) {
                return AsyncServeExit {
                    status: AsyncServeStatus::Failed,
                    failure: Some(error),
                    cleanup_failures: cleanup,
                };
            }
        }
        if control == AsyncControl::ReturnPending {
            return AsyncServeExit {
                status: AsyncServeStatus::Yielded,
                failure: None,
                cleanup_failures: cleanup,
            };
        }
        if output.observation().failed {
            if let Some(error) = close_once(
                session,
                &mut handle,
                &mut dispatch,
                &mut pending,
                &mut cleanup,
                shutdown_requested,
            ) {
                cleanup.push(error);
            }
            return AsyncServeExit {
                status: AsyncServeStatus::Failed,
                failure: Some(AsyncServeFailure::OutputWorkerFailed),
                cleanup_failures: cleanup,
            };
        }
        if let Some(front) = pending.front_mut() {
            if front.encoded.is_none() {
                match encode_event_output(&front.original, max_output_bytes) {
                    Ok(batch) => front.encoded = Some(batch),
                    Err(error) => {
                        if let Some(error) = close_once(
                            session,
                            &mut handle,
                            &mut dispatch,
                            &mut pending,
                            &mut cleanup,
                            shutdown_requested,
                        ) {
                            cleanup.push(error);
                        }
                        return AsyncServeExit {
                            status: AsyncServeStatus::Failed,
                            failure: Some(AsyncServeFailure::Encoding(error)),
                            cleanup_failures: cleanup,
                        };
                    }
                }
            }
            let batch = front
                .encoded
                .take()
                .expect("encoded output is installed above");
            match output.try_submit(batch) {
                Ok(()) => {
                    pending.pop_front();
                    continue;
                }
                Err(SubmitError::Full(batch)) => {
                    pending.front_mut().expect("pending front retained").encoded = Some(batch);
                    thread::park_timeout(poll_interval);
                    continue;
                }
                Err(SubmitError::TooLarge(batch)) => {
                    pending.front_mut().expect("pending front retained").encoded = Some(batch);
                    if let Some(error) = close_once(
                        session,
                        &mut handle,
                        &mut dispatch,
                        &mut pending,
                        &mut cleanup,
                        shutdown_requested,
                    ) {
                        cleanup.push(error);
                    }
                    return AsyncServeExit {
                        status: AsyncServeStatus::Failed,
                        failure: Some(AsyncServeFailure::OutputTooLarge),
                        cleanup_failures: cleanup,
                    };
                }
                Err(SubmitError::Closed(batch)) => {
                    pending.front_mut().expect("pending front retained").encoded = Some(batch);
                    if let Some(error) = close_once(
                        session,
                        &mut handle,
                        &mut dispatch,
                        &mut pending,
                        &mut cleanup,
                        shutdown_requested,
                    ) {
                        cleanup.push(error);
                    }
                    return AsyncServeExit {
                        status: AsyncServeStatus::Failed,
                        failure: Some(AsyncServeFailure::OutputClosed),
                        cleanup_failures: cleanup,
                    };
                }
            }
        }
        if session.is_closed() {
            output.request_close();
            return AsyncServeExit {
                status: AsyncServeStatus::OutputQueued,
                failure: None,
                cleanup_failures: cleanup,
            };
        }
        if *input_disconnected {
            thread::park_timeout(poll_interval);
            continue;
        }
        let event = match events.recv_timeout(poll_interval) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                *input_disconnected = true;
                Event::EndOfInput
            }
        };
        if matches!(&event, Event::EndOfInput)
            || matches!(&event, Event::Line(line)
                if matches!(crate::parse(line, session.parser_limits()), Ok(crate::Command::Quit)))
        {
            *shutdown_requested = true;
        }
        let result = match catch_unwind(AssertUnwindSafe(|| handle(session, event))) {
            Ok(result) => result,
            Err(payload) => {
                return AsyncServeExit {
                    status: AsyncServeStatus::Failed,
                    failure: Some(AsyncServeFailure::HandlerPanic(payload)),
                    cleanup_failures: cleanup,
                };
            }
        };
        let (event, failure, extra) = dispatch_result(result, &mut dispatch);
        pending.push_back(event);
        cleanup.extend(extra.into_iter().map(AsyncServeFailure::Dispatch));
        if let Some(error) = failure {
            if !matches!(&error, AsyncDispatchFailure::Panic(_)) {
                if let Some(error) = close_once(
                    session,
                    &mut handle,
                    &mut dispatch,
                    &mut pending,
                    &mut cleanup,
                    shutdown_requested,
                ) {
                    cleanup.push(error);
                }
            }
            return AsyncServeExit {
                status: AsyncServeStatus::Failed,
                failure: Some(AsyncServeFailure::Dispatch(error)),
                cleanup_failures: cleanup,
            };
        }
    }
}

/// Convenience adapter for the original untimed Session reducer.
pub fn default_handler<P: PositionPort>(
    session: &mut Session<P>,
    event: Event,
) -> SessionResult<P::Snapshot> {
    handle_event(session, event)
}
