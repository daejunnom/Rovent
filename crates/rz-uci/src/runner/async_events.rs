//! Opt-in UCI reducer with bounded asynchronous output admission.
//!
//! FIFO output is retained while the writer queue is full. Normal events are
//! not dequeued in that state; a separate caller-owned control closure remains
//! observable at a caller-selected finite tick. This closure is not itself an
//! engine admission latch, a physical cancellation acknowledgment, or a GPU
//! fence. Production wiring must supply the original session-close authority
//! and retain the returned pending events and output owner on every exit.

use super::async_output::{
    AsyncOutputOwner, OutputBatch, OutputEncodingError, SubmitError, encode_event_output,
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
    /// Ask the original reducer to close; queued output still remains ordered.
    Shutdown,
    /// Close the reducer and return any still-pending output to its caller.
    ReturnPending,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsyncServeStatus {
    /// All event outputs were queued and ordered drain was requested. This is
    /// not writer completion, physical completion, or delivery evidence.
    OutputQueued,
    /// The caller requested return with original pending evidence retained.
    Yielded,
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
pub struct AsyncServeExit<S, E> {
    pub status: AsyncServeStatus,
    /// FIFO: current event first, followed by any original close output.
    pub pending: VecDeque<PendingEvent<S>>,
    pub failure: Option<AsyncServeFailure<E>>,
    pub cleanup_failures: Vec<AsyncServeFailure<E>>,
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
    for effect in std::mem::take(&mut result.effects) {
        if failure.is_some() && !matches!(effect, Effect::Shutdown) {
            // The original effect was not dispatched after a failure.
            result.effects.push(effect);
            continue;
        }
        let dispatched = catch_unwind(AssertUnwindSafe(|| dispatch(effect)));
        let error = match dispatched {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(AsyncDispatchFailure::Error(error)),
            Err(payload) => Some(AsyncDispatchFailure::Panic(payload)),
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
    let (mut result, handler_failure) = match handled {
        Ok(result) => (result, None),
        Err(payload) => (
            SessionResult::default(),
            Some(AsyncServeFailure::HandlerPanic(payload)),
        ),
    };
    // Preserve the normal runner's EOF fallback, including its evidence.
    let fallback = catch_unwind(AssertUnwindSafe(|| session.end_of_input()));
    let mut failure = handler_failure;
    match fallback {
        Ok(fallback) => {
            result.protocol.extend(fallback.protocol);
            result.diagnostics.extend(fallback.diagnostics);
            result.effects.extend(fallback.effects);
        }
        Err(payload) if failure.is_none() => {
            failure = Some(AsyncServeFailure::HandlerPanic(payload))
        }
        Err(payload) => cleanup.push(AsyncServeFailure::HandlerPanic(payload)),
    }
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

/// Serve using the original reducer and an externally owned output worker.
/// `poll` must be nonblocking. An external session-close request must be tied
/// to the same engine's admission latch before it is used as shutdown control.
/// On `Yielded`/`Failed`, retain this return value and `output` together; no
/// automatic retry, output drop, thread join, or physical completion is implied.
pub fn serve_events_with_async_handler<P, O, D, H, F, C, E>(
    session: &mut Session<P>,
    events: &Receiver<Event>,
    output: &AsyncOutputOwner<O, D>,
    max_output_bytes: usize,
    poll_interval: Duration,
    mut poll: C,
    mut handle: H,
    mut dispatch: F,
) -> AsyncServeExit<P::Snapshot, E>
where
    P: PositionPort,
    O: Write + Send + 'static,
    D: Write + Send + 'static,
    H: FnMut(&mut Session<P>, Event) -> SessionResult<P::Snapshot>,
    F: FnMut(Effect<P::Snapshot>) -> Result<(), E>,
    C: FnMut() -> AsyncControl,
{
    let mut pending = VecDeque::new();
    let mut cleanup = Vec::new();
    if poll_interval.is_zero() {
        return AsyncServeExit {
            status: AsyncServeStatus::Failed,
            pending,
            failure: Some(AsyncServeFailure::InvalidPollInterval),
            cleanup_failures: cleanup,
        };
    }
    loop {
        let control = match catch_unwind(AssertUnwindSafe(&mut poll)) {
            Ok(control) => control,
            Err(payload) => {
                if let Some(error) = close_original(
                    session,
                    &mut handle,
                    &mut dispatch,
                    &mut pending,
                    &mut cleanup,
                ) {
                    cleanup.push(error);
                }
                return AsyncServeExit {
                    status: AsyncServeStatus::Failed,
                    pending,
                    failure: Some(AsyncServeFailure::ControlPanic(payload)),
                    cleanup_failures: cleanup,
                };
            }
        };
        if control != AsyncControl::Continue {
            if let Some(error) = close_original(
                session,
                &mut handle,
                &mut dispatch,
                &mut pending,
                &mut cleanup,
            ) {
                return AsyncServeExit {
                    status: AsyncServeStatus::Failed,
                    pending,
                    failure: Some(error),
                    cleanup_failures: cleanup,
                };
            }
            if control == AsyncControl::ReturnPending {
                return AsyncServeExit {
                    status: AsyncServeStatus::Yielded,
                    pending,
                    failure: None,
                    cleanup_failures: cleanup,
                };
            }
        }
        if output.observation().failed {
            if let Some(error) = close_original(
                session,
                &mut handle,
                &mut dispatch,
                &mut pending,
                &mut cleanup,
            ) {
                cleanup.push(error);
            }
            return AsyncServeExit {
                status: AsyncServeStatus::Failed,
                pending,
                failure: Some(AsyncServeFailure::OutputWorkerFailed),
                cleanup_failures: cleanup,
            };
        }
        if let Some(front) = pending.front_mut() {
            if front.encoded.is_none() {
                match encode_event_output(&front.original, max_output_bytes) {
                    Ok(batch) => front.encoded = Some(batch),
                    Err(error) => {
                        if let Some(error) = close_original(
                            session,
                            &mut handle,
                            &mut dispatch,
                            &mut pending,
                            &mut cleanup,
                        ) {
                            cleanup.push(error);
                        }
                        return AsyncServeExit {
                            status: AsyncServeStatus::Failed,
                            pending,
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
                    if let Some(error) = close_original(
                        session,
                        &mut handle,
                        &mut dispatch,
                        &mut pending,
                        &mut cleanup,
                    ) {
                        cleanup.push(error);
                    }
                    return AsyncServeExit {
                        status: AsyncServeStatus::Failed,
                        pending,
                        failure: Some(AsyncServeFailure::OutputTooLarge),
                        cleanup_failures: cleanup,
                    };
                }
                Err(SubmitError::Closed(batch)) => {
                    pending.front_mut().expect("pending front retained").encoded = Some(batch);
                    if let Some(error) = close_original(
                        session,
                        &mut handle,
                        &mut dispatch,
                        &mut pending,
                        &mut cleanup,
                    ) {
                        cleanup.push(error);
                    }
                    return AsyncServeExit {
                        status: AsyncServeStatus::Failed,
                        pending,
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
                pending,
                failure: None,
                cleanup_failures: cleanup,
            };
        }
        let event = match events.recv_timeout(poll_interval) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => Event::EndOfInput,
        };
        let result = match catch_unwind(AssertUnwindSafe(|| handle(session, event))) {
            Ok(result) => result,
            Err(payload) => {
                if let Some(error) = close_original(
                    session,
                    &mut handle,
                    &mut dispatch,
                    &mut pending,
                    &mut cleanup,
                ) {
                    cleanup.push(error);
                }
                return AsyncServeExit {
                    status: AsyncServeStatus::Failed,
                    pending,
                    failure: Some(AsyncServeFailure::HandlerPanic(payload)),
                    cleanup_failures: cleanup,
                };
            }
        };
        let (event, failure, extra) = dispatch_result(result, &mut dispatch);
        pending.push_back(event);
        cleanup.extend(extra.into_iter().map(AsyncServeFailure::Dispatch));
        if let Some(error) = failure {
            if let Some(error) = close_original(
                session,
                &mut handle,
                &mut dispatch,
                &mut pending,
                &mut cleanup,
            ) {
                cleanup.push(error);
            }
            return AsyncServeExit {
                status: AsyncServeStatus::Failed,
                pending,
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
