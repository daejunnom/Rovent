//! Event loop with independent command/completion producers. This module never
//! waits for a search worker or takes ownership of a blocking input thread.

use crate::{
    Diagnostic, Effect, PositionPort, SearchCompletion, SearchTicket, Session, SessionResult,
};
use std::fmt;
use std::io::{self, BufRead, Write};
use std::sync::mpsc::{Receiver, SyncSender};

/// Separate opt-in reducer loop; it never replaces the default transport.
pub mod async_events;
/// Opt-in bounded output custody. The existing synchronous runner below keeps
/// its original behavior; consumers must explicitly select and own this path.
pub mod async_output;

#[derive(Clone, Debug)]
pub enum Event {
    Line(String),
    Progress {
        ticket: SearchTicket,
        bestmove: String,
    },
    Complete {
        ticket: SearchTicket,
        completion: SearchCompletion,
    },
    Deadline(SearchTicket),
    RejectedInput {
        code: &'static str,
        message: String,
    },
    EndOfInput,
}

#[derive(Debug)]
pub enum ServeFailure<E> {
    Io(io::Error),
    Handler(E),
}

/// Original failure, prepared event diagnostics, and explicit shutdown failures
/// are kept independently, even when effect dispatch or output prevents logging.
#[derive(Debug)]
pub struct ServeError<E> {
    pub failure: ServeFailure<E>,
    pub cleanup_failures: Vec<E>,
    /// All diagnostics prepared by the failed event and cleanup. Some may already
    /// have reached the diagnostic writer before its write/flush failed.
    pub diagnostics: Vec<Diagnostic>,
}

impl<E: fmt::Display> fmt::Display for ServeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.failure {
            ServeFailure::Io(err) => write!(f, "UCI transport failed: {err}")?,
            ServeFailure::Handler(err) => write!(f, "UCI effect dispatch failed: {err}")?,
        }
        for err in &self.cleanup_failures {
            write!(f, "; shutdown dispatch failed: {err}")?;
        }
        for diagnostic in &self.diagnostics {
            write!(f, "; {}: {}", diagnostic.code, diagnostic.message)?;
        }
        Ok(())
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ServeError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.failure {
            ServeFailure::Io(err) => Some(err),
            ServeFailure::Handler(err) => Some(err),
        }
    }
}

/// Untimed lifecycle transport using [`handle_event`]. This wrapper provides
/// Session ownership and output checks, but does not revalidate clock deadlines.
/// Timed integrations must use [`serve_events_with_handler`] with a binding-aware
/// handler that rechecks admission when each event is dequeued. Producer-only
/// checks cannot prevent a queued completion from expiring before dispatch.
///
/// Prefer `sync_channel` for bounded producers. Receiver disconnection is EOF.
/// The caller owns transport interruption, worker joining, and runtime buffers.
pub fn serve_events<P, O, D, F, E>(
    session: &mut Session<P>,
    events: &Receiver<Event>,
    protocol: &mut O,
    diagnostics: &mut D,
    dispatch: F,
) -> Result<(), ServeError<E>>
where
    P: PositionPort,
    O: Write,
    D: Write,
    F: FnMut(Effect<P::Snapshot>) -> Result<(), E>,
{
    serve_events_with_handler(
        session,
        events,
        protocol,
        diagnostics,
        handle_event,
        dispatch,
    )
}

/// Default untimed Session mapping, reusable for non-search events in a custom
/// binding-aware handler. It does not perform any monotonic deadline checks.
pub fn handle_event<P: PositionPort>(
    session: &mut Session<P>,
    event: Event,
) -> SessionResult<P::Snapshot> {
    match event {
        Event::Line(line) => session.handle_line(&line),
        Event::Progress { ticket, bestmove } => session.progress(&ticket, &bestmove),
        Event::Complete { ticket, completion } => session.complete(&ticket, completion),
        Event::Deadline(ticket) => session.expire(&ticket),
        Event::EndOfInput => session.end_of_input(),
        Event::RejectedInput { code, message } => SessionResult {
            protocol: Vec::new(),
            diagnostics: vec![Diagnostic { code, message }],
            effects: Vec::new(),
            accepted: false,
        },
    }
}

/// Event loop with admission validation at the Session owner's dequeue boundary.
/// A timed `handle` checks its binding's current monotonic clock/generations on
/// Progress, Complete, and Deadline before mutating Session or accepting backup.
/// It may delegate command/EOF/rejected-input events to [`handle_event`]. EOF and
/// quit must still close the session; channel disconnection supplies EOF.
///
/// `dispatch` must enqueue/cancel promptly and initiate finite runtime drain on
/// Shutdown. It must not perform full searches inline or join blocked stdin.
/// The caller owns bounded producer queues, worker joining and runtime buffers.
pub fn serve_events_with_handler<P, O, D, H, F, E>(
    session: &mut Session<P>,
    events: &Receiver<Event>,
    protocol: &mut O,
    diagnostics: &mut D,
    mut handle: H,
    mut dispatch: F,
) -> Result<(), ServeError<E>>
where
    P: PositionPort,
    O: Write,
    D: Write,
    H: FnMut(&mut Session<P>, Event) -> SessionResult<P::Snapshot>,
    F: FnMut(Effect<P::Snapshot>) -> Result<(), E>,
{
    while !session.is_closed() {
        let event = events.recv().unwrap_or(Event::EndOfInput);
        let out = handle(session, event);
        let result = emit(out, protocol, diagnostics, &mut dispatch);
        if let Err(mut error) = result {
            // The injected owner also holds common cancellation/generation rights.
            // Close those through the same reducer before trying runtime cleanup.
            let mut shutdown = handle(session, Event::EndOfInput);
            // Preserve transport shutdown even if a custom reducer ignored EOF.
            let fallback = session.end_of_input();
            shutdown.effects.extend(fallback.effects);
            error.diagnostics.extend(shutdown.diagnostics);
            error.diagnostics.extend(fallback.diagnostics);
            for effect in shutdown.effects {
                if let Err(err) = dispatch(effect) {
                    error.cleanup_failures.push(err);
                }
            }
            return Err(error);
        }
    }
    Ok(())
}

fn emit<S, O: Write, D: Write, F, E>(
    out: SessionResult<S>,
    protocol: &mut O,
    diagnostics: &mut D,
    dispatch: &mut F,
) -> Result<(), ServeError<E>>
where
    F: FnMut(Effect<S>) -> Result<(), E>,
{
    // Revoke result/backup admission before writing a stop/deadline bestmove.
    let mut failed = None;
    let mut cleanup_failures = Vec::new();
    for effect in out.effects {
        // A failing Cancel must still permit the following Shutdown on quit.
        // Do not launch new work after any dispatch failure.
        if failed.is_some() && !matches!(effect, Effect::Shutdown) {
            continue;
        }
        if let Err(err) = dispatch(effect) {
            if failed.is_none() {
                failed = Some(err);
            } else {
                cleanup_failures.push(err);
            }
        }
    }
    if let Some(error) = failed {
        return Err(ServeError {
            failure: ServeFailure::Handler(error),
            cleanup_failures,
            diagnostics: out.diagnostics,
        });
    }
    // Borrow the prepared diagnostics while writing so a protocol/diagnostic
    // write or flush failure can return the complete original event evidence.
    let written = (|| -> io::Result<()> {
        for line in out.protocol {
            writeln!(protocol, "{line}")?;
        }
        protocol.flush()?;
        for diagnostic in &out.diagnostics {
            writeln!(diagnostics, "{}: {}", diagnostic.code, diagnostic.message)?;
        }
        diagnostics.flush()
    })();
    written.map_err(|error| ServeError {
        failure: ServeFailure::Io(error),
        cleanup_failures: Vec::new(),
        diagnostics: out.diagnostics,
    })
}

/// Optional blocking input producer, normally run by the transport on its own
/// task. No line buffers more than `max_line_bytes`; an oversized physical
/// line is fully discarded before the next command. Sender backpressure bounds
/// queued input. A disconnected consumer ends this producer promptly once the
/// current read finishes. OS-level blocked reads remain the caller's concern.
/// Read failure emits a diagnostic and EndOfInput before returning the original
/// I/O error, so worker-held sender clones cannot keep the session open.
pub fn forward_lines<R: BufRead>(
    reader: &mut R,
    events: &SyncSender<Event>,
    max_line_bytes: usize,
) -> io::Result<()> {
    loop {
        let mut line = Vec::new();
        let mut oversized = false;
        let mut any = false;
        loop {
            let buffer = match reader.fill_buf() {
                Ok(buffer) => buffer,
                Err(error) => {
                    let _ = events.send(Event::RejectedInput {
                        code: "InputIoError",
                        message: error.to_string(),
                    });
                    let _ = events.send(Event::EndOfInput);
                    return Err(error);
                }
            };
            if buffer.is_empty() {
                break;
            }
            any = true;
            let newline = buffer.iter().position(|b| *b == b'\n');
            let consumed = newline.map_or(buffer.len(), |at| at + 1);
            if !oversized {
                if consumed > max_line_bytes.saturating_sub(line.len()) {
                    oversized = true;
                    line.clear();
                } else {
                    line.extend_from_slice(&buffer[..consumed]);
                }
            }
            reader.consume(consumed);
            if newline.is_some() {
                break;
            }
        }
        if !any {
            let _ = events.send(Event::EndOfInput);
            return Ok(());
        }
        let event = if oversized {
            Event::RejectedInput {
                code: "LineTooLong",
                message: "discarded oversized physical UCI line".into(),
            }
        } else {
            match String::from_utf8(line) {
                Ok(line) => Event::Line(line),
                Err(_) => Event::RejectedInput {
                    code: "InvalidText",
                    message: "discarded non-UTF-8 UCI line".into(),
                },
            }
        };
        if events.send(event).is_err() {
            return Ok(());
        }
    }
}
