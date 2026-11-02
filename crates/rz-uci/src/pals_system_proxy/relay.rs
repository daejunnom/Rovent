//! Byte-transparent, bounded-buffer relay for the original PALS child pipes.
//!
//! This does not implement a chess clock or invent any UCI response. The
//! outer arena measures its existing position-to-bestmove wall clock, which
//! naturally includes relay overhead. EOF and relay errors remain separate
//! from the original session's physical completion.

use super::PalsDirectChildPipesV6;
use std::{
    io::{self, Read, Write},
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
};

pub type Reader = Box<dyn Read + Send + 'static>;
pub type Writer = Box<dyn Write + Send + 'static>;

struct RelayJob {
    source: Reader,
    destination: Writer,
    drain_after_write_failure: bool,
}

/// A pipe observation only. Partial byte counts after a write failure are
/// unknown because the standard copy loop does not report them.
#[derive(Clone, Debug)]
pub struct RelayPipeObservationV6 {
    pub bytes: Option<u64>,
    pub error: Option<String>,
    pub source_drained_after_error: Option<bool>,
}

struct RelayLeg {
    job: Arc<Mutex<Option<RelayJob>>>,
    thread: Option<JoinHandle<RelayPipeObservationV6>>,
    observation: Option<RelayPipeObservationV6>,
}

impl RelayLeg {
    fn new(source: Reader, destination: Writer, drain_after_write_failure: bool) -> Self {
        Self {
            job: Arc::new(Mutex::new(Some(RelayJob {
                source,
                destination,
                drain_after_write_failure,
            }))),
            thread: None,
            observation: None,
        }
    }

    fn start(&mut self, name: &'static str) -> io::Result<()> {
        if self.thread.is_some() || self.observation.is_some() {
            return Ok(());
        }
        let job = Arc::clone(&self.job);
        // If thread creation fails, the original streams stay in `job`; a
        // retry or explicit recovery can use this exact owner without a pipe
        // being silently closed during partial setup.
        self.thread = Some(thread::Builder::new().name(name.into()).spawn(move || {
            let Ok(mut slot) = job.lock() else {
                return RelayPipeObservationV6 {
                    bytes: None,
                    error: Some("relay owner poisoned".into()),
                    source_drained_after_error: None,
                };
            };
            let Some(mut job) = slot.take() else {
                return RelayPipeObservationV6 {
                    bytes: None,
                    error: Some("relay owner unavailable".into()),
                    source_drained_after_error: None,
                };
            };
            drop(slot);
            match io::copy(&mut job.source, &mut job.destination) {
                Ok(bytes) => RelayPipeObservationV6 {
                    bytes: Some(bytes),
                    error: None,
                    source_drained_after_error: None,
                },
                Err(error) => {
                    // If a consumer goes away, keep draining output so a
                    // still-live GPU owner cannot block solely on this pipe.
                    let drained = job
                        .drain_after_write_failure
                        .then(|| io::copy(&mut job.source, &mut io::sink()).is_ok());
                    RelayPipeObservationV6 {
                        bytes: None,
                        error: Some(error.to_string()),
                        source_drained_after_error: drained,
                    }
                }
            }
        })?);
        Ok(())
    }

    fn poll(&mut self) -> Option<RelayPipeObservationV6> {
        if self.thread.as_ref().is_some_and(JoinHandle::is_finished) {
            let thread = self.thread.take().expect("checked finished thread");
            self.observation = Some(thread.join().unwrap_or_else(|_| RelayPipeObservationV6 {
                bytes: None,
                error: Some("relay thread panicked".into()),
                source_drained_after_error: None,
            }));
        }
        self.observation.clone()
    }
}

/// Exactly one owner for each original pipe and the relay threads. A failed
/// `start` keeps this value and any already-started legs intact; callers must
/// retain it alongside the direct child rather than dropping it on error.
#[must_use = "retain the original pipe and worker owners until shutdown is resolved"]
pub struct PalsUciRelayV6 {
    input: RelayLeg,
    output: RelayLeg,
    stderr: RelayLeg,
}

impl PalsUciRelayV6 {
    pub fn new(
        pipes: PalsDirectChildPipesV6,
        incoming_uci: Reader,
        outgoing_uci: Writer,
        diagnostic_stderr: Writer,
    ) -> Self {
        Self {
            input: RelayLeg::new(incoming_uci, Box::new(pipes.stdin), false),
            output: RelayLeg::new(Box::new(pipes.stdout), outgoing_uci, true),
            stderr: RelayLeg::new(Box::new(pipes.stderr), diagnostic_stderr, true),
        }
    }

    /// Start each leg once; each copy uses Rust's bounded standard I/O buffer
    /// and preserves byte order. Any error leaves all previous legs owned.
    pub fn start(&mut self) -> io::Result<()> {
        self.output.start("pals-uci-output")?;
        self.stderr.start("pals-uci-stderr")?;
        self.input.start("pals-uci-input")
    }

    /// Nonblocking per-leg observations. An incomplete thread is `None` and
    /// none of these observations establish child or GPU completion.
    pub fn poll(&mut self) -> [Option<RelayPipeObservationV6>; 3] {
        [self.input.poll(), self.output.poll(), self.stderr.poll()]
    }
}
