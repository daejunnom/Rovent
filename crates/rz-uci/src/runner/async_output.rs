//! Bounded opt-in output custody for a UCI event loop.
//!
//! Submission only means the original bytes entered this owner's queue; it is
//! not proof that either stream was written or flushed. A full queue returns
//! the same batch to its caller. The writer runs on a separate thread and never
//! holds the queue-state mutex during `Write` or `flush`. A blocked call keeps
//! its writer and in-flight bytes; cancellation or a deadline cannot release it.
//!
//! This component is not wired into the default synchronous runner. Its caller
//! must own the thread, handle priority cancellation and retain the process when
//! output completion is unknown. Dropping this owner quarantines the join
//! handle within the still-running process; it is not a process-lifetime fence.

use std::any::Any;
use std::collections::TryReserveError;
use std::io::{self, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

#[derive(Debug, Eq, PartialEq)]
pub struct OutputBatch {
    pub protocol: Vec<u8>,
    pub diagnostics: Vec<u8>,
}

#[derive(Debug)]
pub enum OutputEncodingError {
    LengthOverflow,
    Limit { required: usize, limit: usize },
    Allocation(TryReserveError),
}

/// Prepare exactly the same stream bytes as the synchronous runner without
/// consuming the original event result. Allocation or a declared byte bound
/// failure leaves all original protocol and diagnostic values with the caller.
pub fn encode_event_output<S>(
    result: &crate::SessionResult<S>,
    limit: usize,
) -> Result<OutputBatch, OutputEncodingError> {
    let protocol_len = result
        .protocol
        .iter()
        .try_fold(0usize, |sum, line| {
            sum.checked_add(line.len())?.checked_add(1)
        })
        .ok_or(OutputEncodingError::LengthOverflow)?;
    let diagnostics_len = result
        .diagnostics
        .iter()
        .try_fold(0usize, |sum, item| {
            sum.checked_add(item.code.len())?
                .checked_add(2)?
                .checked_add(item.message.len())?
                .checked_add(1)
        })
        .ok_or(OutputEncodingError::LengthOverflow)?;
    let required = protocol_len
        .checked_add(diagnostics_len)
        .ok_or(OutputEncodingError::LengthOverflow)?;
    if required > limit {
        return Err(OutputEncodingError::Limit { required, limit });
    }
    let mut protocol = Vec::new();
    protocol
        .try_reserve_exact(protocol_len)
        .map_err(OutputEncodingError::Allocation)?;
    let mut diagnostics = Vec::new();
    diagnostics
        .try_reserve_exact(diagnostics_len)
        .map_err(OutputEncodingError::Allocation)?;
    for line in &result.protocol {
        protocol.extend_from_slice(line.as_bytes());
        protocol.push(b'\n');
    }
    for item in &result.diagnostics {
        diagnostics.extend_from_slice(item.code.as_bytes());
        diagnostics.extend_from_slice(b": ");
        diagnostics.extend_from_slice(item.message.as_bytes());
        diagnostics.push(b'\n');
    }
    Ok(OutputBatch {
        protocol,
        diagnostics,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputPhase {
    ProtocolWrite,
    ProtocolFlush,
    DiagnosticWrite,
    DiagnosticFlush,
}

#[derive(Debug)]
pub struct PartialOutput {
    pub batch: OutputBatch,
    pub phase: OutputPhase,
    /// Bytes accepted by completed calls in this phase. It does not include a
    /// currently blocked call or a call that panicked before returning.
    pub offset: usize,
    pub call_completion_unknown: bool,
}

#[derive(Debug)]
pub enum OutputFailure {
    Io(io::Error),
    Panic(Box<dyn Any + Send>),
}

pub struct OutputRecovery<O, D> {
    pub protocol: O,
    pub diagnostics: D,
    pub partial: Option<PartialOutput>,
    /// Original bounded receiver; draining it does not allocate a replacement
    /// queue or discard batches during failure recovery.
    pub queued: mpsc::Receiver<OutputBatch>,
    pub failure: Option<OutputFailure>,
    /// The owner requested drain; this does not by itself mean it succeeded.
    pub close_requested: bool,
}

pub enum StartError<O, D> {
    Invalid {
        protocol: O,
        diagnostics: D,
    },
    Spawn {
        error: io::Error,
        protocol: O,
        diagnostics: D,
    },
}

#[derive(Debug)]
pub enum SubmitError {
    Full(OutputBatch),
    TooLarge(OutputBatch),
    Closed(OutputBatch),
}

#[derive(Clone, Copy, Debug)]
pub struct OutputSnapshot {
    pub accepted_batches: u64,
    pub completed_batches: u64,
    pub phase: Option<OutputPhase>,
    pub offset: usize,
    pub call_completion_unknown: bool,
    pub close_requested: bool,
    pub failed: bool,
    pub worker_finished: bool,
}

struct Shared {
    accepted: u64,
    completed: u64,
    phase: Option<OutputPhase>,
    offset: usize,
    call_unknown: bool,
    close: bool,
    failed: bool,
    finished: bool,
}

type Exit<O, D> = OutputRecovery<O, D>;

/// The caller retains this owner, including after an event-loop deadline.
#[must_use]
pub struct AsyncOutputOwner<O: Write + Send + 'static, D: Write + Send + 'static> {
    shared: Arc<Mutex<Shared>>,
    sender: mpsc::SyncSender<OutputBatch>,
    worker: Option<JoinHandle<Exit<O, D>>>,
    max_batch_bytes: usize,
}

impl<O: Write + Send + 'static, D: Write + Send + 'static> AsyncOutputOwner<O, D> {
    pub fn start(
        protocol: O,
        diagnostics: D,
        queue_batches: usize,
        max_batch_bytes: usize,
    ) -> Result<Self, StartError<O, D>> {
        if queue_batches == 0 || max_batch_bytes == 0 {
            return Err(StartError::Invalid {
                protocol,
                diagnostics,
            });
        }
        let (sender, receiver) = mpsc::sync_channel(queue_batches);
        let shared = Arc::new(Mutex::new(Shared {
            accepted: 0,
            completed: 0,
            phase: None,
            offset: 0,
            call_unknown: false,
            close: false,
            failed: false,
            finished: false,
        }));
        // Retain both original writers if the OS rejects the thread spawn.
        let originals = Arc::new(Mutex::new(Some((protocol, diagnostics))));
        let worker_originals = Arc::clone(&originals);
        let worker_shared = Arc::clone(&shared);
        let worker = match thread::Builder::new()
            .name("rz-uci-output".into())
            .spawn(move || {
                let (mut protocol, mut diagnostics) = worker_originals
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .take()
                    .expect("original output writers are moved once");
                let mut current = None;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    run_writer(
                        &mut protocol,
                        &mut diagnostics,
                        &receiver,
                        &worker_shared,
                        &mut current,
                    )
                }));
                let failure = match result {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) => Some(OutputFailure::Io(error)),
                    Err(payload) => Some(OutputFailure::Panic(payload)),
                };
                let close_requested = {
                    let mut state = worker_shared
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    state.failed = failure.is_some();
                    state.finished = true;
                    state.close
                };
                OutputRecovery {
                    protocol,
                    diagnostics,
                    partial: current,
                    queued: receiver,
                    failure,
                    close_requested,
                }
            }) {
            Ok(worker) => worker,
            Err(error) => {
                let (protocol, diagnostics) = originals
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .take()
                    .expect("spawn failure retains original output writers");
                return Err(StartError::Spawn {
                    error,
                    protocol,
                    diagnostics,
                });
            }
        };
        Ok(Self {
            shared,
            sender,
            worker: Some(worker),
            max_batch_bytes,
        })
    }

    /// A successful result is queue admission only. On all failures the exact
    /// batch remains in the returned error for caller-owned pending output.
    pub fn try_submit(&self, batch: OutputBatch) -> Result<(), SubmitError> {
        // Allocation capacity, not only logical length, determines retained
        // queue memory; rejected batches remain with their original owner.
        if batch
            .protocol
            .capacity()
            .checked_add(batch.diagnostics.capacity())
            .is_none_or(|bytes| bytes > self.max_batch_bytes)
        {
            return Err(SubmitError::TooLarge(batch));
        }
        let mut state = self
            .shared
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if state.close || state.failed || state.finished || state.accepted == u64::MAX {
            return Err(SubmitError::Closed(batch));
        }
        match self.sender.try_send(batch) {
            Ok(()) => {
                state.accepted += 1;
                if let Some(worker) = &self.worker {
                    worker.thread().unpark();
                }
                Ok(())
            }
            Err(mpsc::TrySendError::Full(batch)) => Err(SubmitError::Full(batch)),
            Err(mpsc::TrySendError::Disconnected(batch)) => Err(SubmitError::Closed(batch)),
        }
    }

    /// Close admission and request ordered draining. This does not join or
    /// interrupt an in-flight write and is not an output-completion witness.
    pub fn request_close(&self) {
        let mut state = self
            .shared
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.close = true;
        if let Some(worker) = &self.worker {
            worker.thread().unpark();
        }
    }

    pub fn observation(&self) -> OutputSnapshot {
        let state = self
            .shared
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        OutputSnapshot {
            accepted_batches: state.accepted,
            completed_batches: state.completed,
            phase: state.phase,
            offset: state.offset,
            call_completion_unknown: state.call_unknown,
            close_requested: state.close,
            failed: state.failed,
            worker_finished: state.finished,
        }
    }

    /// Never waits. If still active, return the original owner unchanged.
    pub fn try_recover(self) -> Result<Result<OutputRecovery<O, D>, Box<dyn Any + Send>>, Self> {
        self.try_recover_with_final_snapshot()
            .map(|(_, result)| result)
    }

    /// Join only an already-finished original writer and then observe its same
    /// shared state. An outer thread panic may leave `worker_finished=false`;
    /// the returned snapshot never substitutes for the join result or a GPU fence.
    pub fn try_recover_with_final_snapshot(
        mut self,
    ) -> Result<
        (
            OutputSnapshot,
            Result<OutputRecovery<O, D>, Box<dyn Any + Send>>,
        ),
        Self,
    > {
        if !self.worker.as_ref().is_some_and(JoinHandle::is_finished) {
            return Err(self);
        }
        let handle = self.worker.take().expect("finished worker handle retained");
        let result = handle.join();
        let snapshot = self.observation();
        Ok((snapshot, result))
    }
}

impl<O: Write + Send + 'static, D: Write + Send + 'static> Drop for AsyncOutputOwner<O, D> {
    fn drop(&mut self) {
        if let Some(handle) = self.worker.take() {
            // Stop accepting new batches and wake an idle worker. An in-flight
            // blocking write is neither interrupted nor declared complete.
            self.shared
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .close = true;
            handle.thread().unpark();
            // Keep the original return value/writers reachable within this
            // process even if an erroneous caller drops the active owner.
            let _quarantined = Box::leak(Box::new(handle));
        }
    }
}

fn publish(shared: &Mutex<Shared>, current: &PartialOutput) {
    let mut state = shared.lock().unwrap_or_else(|error| error.into_inner());
    state.phase = Some(current.phase);
    state.offset = current.offset;
    state.call_unknown = current.call_completion_unknown;
}

fn run_writer<O: Write, D: Write>(
    protocol: &mut O,
    diagnostics: &mut D,
    receiver: &mpsc::Receiver<OutputBatch>,
    shared: &Mutex<Shared>,
    current: &mut Option<PartialOutput>,
) -> io::Result<()> {
    loop {
        let next = {
            let state = shared.lock().unwrap_or_else(|error| error.into_inner());
            match receiver.try_recv() {
                Ok(batch) => Some(batch),
                Err(mpsc::TryRecvError::Empty) if state.close => return Ok(()),
                Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
                Err(mpsc::TryRecvError::Empty) => None,
            }
        };
        let Some(batch) = next else {
            // A submit/close unparks after publishing under the same mutex.
            // The stored unpark token also covers the interval before park.
            thread::park();
            continue;
        };
        *current = Some(PartialOutput {
            batch,
            phase: OutputPhase::ProtocolWrite,
            offset: 0,
            call_completion_unknown: false,
        });
        let active = current.as_mut().expect("active batch was just installed");
        for phase in [
            OutputPhase::ProtocolWrite,
            OutputPhase::ProtocolFlush,
            OutputPhase::DiagnosticWrite,
            OutputPhase::DiagnosticFlush,
        ] {
            active.phase = phase;
            active.offset = 0;
            publish(shared, active);
            match phase {
                OutputPhase::ProtocolWrite | OutputPhase::DiagnosticWrite => loop {
                    let total = if phase == OutputPhase::ProtocolWrite {
                        active.batch.protocol.len()
                    } else {
                        active.batch.diagnostics.len()
                    };
                    if active.offset == total {
                        break;
                    }
                    active.call_completion_unknown = true;
                    publish(shared, active);
                    let result = if phase == OutputPhase::ProtocolWrite {
                        protocol.write(&active.batch.protocol[active.offset..])
                    } else {
                        diagnostics.write(&active.batch.diagnostics[active.offset..])
                    };
                    active.call_completion_unknown = false;
                    match result {
                        Ok(0) => {
                            publish(shared, active);
                            return Err(io::ErrorKind::WriteZero.into());
                        }
                        Ok(count) if count <= total - active.offset => active.offset += count,
                        Ok(_) => {
                            active.call_completion_unknown = true;
                            publish(shared, active);
                            return Err(io::Error::other("writer reported excess bytes"));
                        }
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => {
                            // A caller's close request stops a persistent stream
                            // of interruptions without discarding this batch.
                            if shared
                                .lock()
                                .unwrap_or_else(|poison| poison.into_inner())
                                .close
                            {
                                publish(shared, active);
                                return Err(error);
                            }
                            thread::yield_now();
                            continue;
                        }
                        Err(error) => {
                            publish(shared, active);
                            return Err(error);
                        }
                    }
                    publish(shared, active);
                },
                OutputPhase::ProtocolFlush | OutputPhase::DiagnosticFlush => {
                    active.call_completion_unknown = true;
                    publish(shared, active);
                    let result = if phase == OutputPhase::ProtocolFlush {
                        protocol.flush()
                    } else {
                        diagnostics.flush()
                    };
                    active.call_completion_unknown = false;
                    publish(shared, active);
                    result?;
                }
            }
        }
        *current = None;
        let mut state = shared.lock().unwrap_or_else(|error| error.into_inner());
        state.completed += 1;
        state.phase = None;
        state.offset = 0;
        state.call_unknown = false;
    }
}
