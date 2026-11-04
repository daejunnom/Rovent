//! Opt-in original engine custody for resumable, nonblocking UCI output.
//! A returned status is neither physical completion nor a writer join.
use super::*;
use crate::runner::{
    async_events::{
        AsyncControl, AsyncServeDriver, AsyncServeExit, AsyncServeStatus, OutputRecoveryPoll,
    },
    async_output::{AsyncOutputOwner, OutputRecovery, OutputSnapshot},
};

type Handler =
    Box<dyn FnMut(&mut Session<RulesUciPort>, Event) -> SessionResult<RulesSearchPosition> + Send>;
type Dispatch = Box<dyn FnMut(Effect<RulesSearchPosition>) -> Result<(), EngineError> + Send>;
type Driver<O, D> = AsyncServeDriver<RulesUciPort, O, D, Handler, Dispatch>;

/// Original inputs or the already-started timer/session remain together with
/// their error. Prepared custody does not claim the timer or device is done.
pub enum AsyncEngineStartFailure<O: Write + Send + 'static, D: Write + Send + 'static> {
    Unstarted {
        error: EngineError,
        process: EngineProcess,
        events: Receiver<Event>,
        sender: SyncSender<Event>,
        output: Option<AsyncOutputOwner<O, D>>,
        output_close_requested: bool,
        output_recovery: Option<Result<OutputRecovery<O, D>, Box<dyn std::any::Any + Send>>>,
        close: PriorityCloseHandle,
    },
    Prepared {
        error: EngineError,
        custody: AsyncPreparedStartFailure<O, D>,
    },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnstartedWriterRecoveryStatus {
    NotUnstarted,
    CloseNotRequested,
    Pending,
    Recovered,
    WorkerPanicked,
    OriginalOwnerMissing,
}
impl<O: Write + Send + 'static, D: Write + Send + 'static> AsyncEngineStartFailure<O, D> {
    /// Explicitly request a drain only for the unstarted path. This is not a
    /// writer join or a statement about GPU/native completion.
    pub fn request_unstarted_writer_close(&mut self) -> bool {
        match self {
            Self::Unstarted {
                output: Some(output),
                output_close_requested,
                ..
            } => {
                output.request_close();
                *output_close_requested = true;
                true
            }
            _ => false,
        }
    }
    /// Preserve the same output owner on Pending and the exact recovery or
    /// original worker panic after a nonblocking join. Prepared is excluded.
    pub fn try_recover_unstarted_writer(&mut self) -> UnstartedWriterRecoveryStatus {
        let Self::Unstarted {
            output,
            output_close_requested,
            output_recovery,
            ..
        } = self
        else {
            return UnstartedWriterRecoveryStatus::NotUnstarted;
        };
        if let Some(recovered) = output_recovery {
            return if recovered.is_ok() {
                UnstartedWriterRecoveryStatus::Recovered
            } else {
                UnstartedWriterRecoveryStatus::WorkerPanicked
            };
        }
        if !*output_close_requested {
            return UnstartedWriterRecoveryStatus::CloseNotRequested;
        }
        let Some(original) = output.take() else {
            return UnstartedWriterRecoveryStatus::OriginalOwnerMissing;
        };
        match original.try_recover() {
            Err(original) => {
                *output = Some(original);
                UnstartedWriterRecoveryStatus::Pending
            }
            Ok(recovered) => {
                let status = if recovered.is_ok() {
                    UnstartedWriterRecoveryStatus::Recovered
                } else {
                    UnstartedWriterRecoveryStatus::WorkerPanicked
                };
                *output_recovery = Some(recovered);
                status
            }
        }
    }
    pub fn unstarted_writer_recovery(
        &self,
    ) -> Option<&Result<OutputRecovery<O, D>, Box<dyn std::any::Any + Send>>> {
        match self {
            Self::Unstarted {
                output_recovery, ..
            } => output_recovery.as_ref(),
            _ => None,
        }
    }
}

/// The original prepared owner and all still-owned output inputs. This is a
/// retained failure, not a release or re-admission authority.
pub struct AsyncPreparedStartFailure<O: Write + Send + 'static, D: Write + Send + 'static> {
    _prepared: PreparedEngine,
    _output: AsyncOutputOwner<O, D>,
    _close: PriorityCloseHandle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineTimerJoinStatus {
    NotReady,
    WaitingWorkers,
    WaitingTimer,
    UnsupportedDriver,
    RetainedFailure,
    Failed,
    /// CPU worker handles and original timer were joined. Writer and native
    /// physical completion are not inferred by this observation.
    Joined,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineWriterRecoveryStatus {
    DriverNotSealed,
    Pending,
    Recovered,
    WorkerPanicked,
    OriginalRecoveryMissing,
}

/// Owns the *same* UCI session, pending output, timer and process owner across
/// time slices. A terminal/unknown result stays here for inspection. Dropping
/// this owner is not a shutdown or physical-completion operation.
#[must_use]
pub struct AsyncEngineRun<O: Write + Send + 'static, D: Write + Send + 'static> {
    driver: Driver<O, D>,
    _owner: Arc<Mutex<Owner>>,
    timer_stop: Arc<std::sync::atomic::AtomicBool>,
    timer: Option<JoinHandle<()>>,
    timer_joined: bool,
    cleanup_failure: Option<EngineError>,
    timer_panic: Option<Box<dyn std::any::Any + Send>>,
    output_recovery: Option<Result<OutputRecovery<O, D>, Box<dyn std::any::Any + Send>>>,
    _priority_close_owner: priority_close::PriorityCloseOwner,
    close: PriorityCloseHandle,
    last_exit: Option<AsyncServeExit<EngineError>>,
}

pub fn start_async<O: Write + Send + 'static, D: Write + Send + 'static>(
    events: Receiver<Event>,
    sender: SyncSender<Event>,
    process: EngineProcess,
    settings: EngineSettings,
    output: AsyncOutputOwner<O, D>,
    close: PriorityCloseHandle,
    max_output_bytes: usize,
    poll_interval: Duration,
) -> Result<AsyncEngineRun<O, D>, AsyncEngineStartFailure<O, D>> {
    // Until the original run's pending output, writer and timer survive an
    // unexpected caller drop in a recoverable owner, this opt-in path is not
    // admitted. Return every original input intact; the synchronous CPU engine
    // and other independent paths do not consult this guard.
    const ORIGINAL_CUSTODY_DROP_RECOVERY_IMPLEMENTED: bool = false;
    if !ORIGINAL_CUSTODY_DROP_RECOVERY_IMPLEMENTED {
        return Err(AsyncEngineStartFailure::Unstarted {
            error: failure(
                contract::ErrorCode::UnsupportedContract,
                "async engine original custody recovery unavailable",
            )
            .into(),
            process,
            events,
            sender,
            output: Some(output),
            output_close_requested: false,
            output_recovery: None,
            close,
        });
    }
    if max_output_bytes == 0
        || poll_interval.is_zero()
        || !process
            .priority_close
            .as_ref()
            .is_some_and(|owner| owner.matches(&close))
    {
        return Err(AsyncEngineStartFailure::Unstarted {
            error: failure(
                contract::ErrorCode::InvalidInput,
                "async output bounds or original close owner invalid",
            )
            .into(),
            process,
            events,
            sender,
            output: Some(output),
            output_close_requested: false,
            output_recovery: None,
            close,
        });
    }
    let mut prepared = match prepare_engine(events, sender, process, settings) {
        Ok(prepared) => prepared,
        Err(rejected) => {
            return Err(AsyncEngineStartFailure::Unstarted {
                error: rejected.error,
                process: rejected.process,
                events: rejected.events,
                sender: rejected.sender,
                output: Some(output),
                output_close_requested: false,
                output_recovery: None,
                close,
            });
        }
    };
    // Recoverable invariant failure: the timer may already exist. Return its
    // same prepared owner, session, endpoints and output, without stopping it
    // or asserting that a device or writer has completed.
    let Some(close_owner) = prepared.priority_close_owner.take() else {
        return Err(AsyncEngineStartFailure::Prepared {
            error: failure(
                contract::ErrorCode::IdentityMismatch,
                "prepared close owner missing",
            )
            .into(),
            custody: AsyncPreparedStartFailure {
                _prepared: prepared,
                _output: output,
                _close: close,
            },
        });
    };
    let Some(observer) = prepared.dispatch_close_observer.take() else {
        prepared.priority_close_owner = Some(close_owner);
        return Err(AsyncEngineStartFailure::Prepared {
            error: failure(
                contract::ErrorCode::IdentityMismatch,
                "prepared close observer missing",
            )
            .into(),
            custody: AsyncPreparedStartFailure {
                _prepared: prepared,
                _output: output,
                _close: close,
            },
        });
    };
    let PreparedEngine {
        owner,
        session,
        events,
        sender,
        settings: _,
        timer_stop,
        timer,
        priority_owner: _,
        priority_close_owner: _,
        dispatch_close_observer: _,
    } = prepared;
    let handler_owner = Arc::clone(&owner);
    let handler: Handler = Box::new(move |session, event| {
        handler_owner
            .lock()
            .expect("UCI owner")
            .handle(session, event)
    });
    let dispatch_owner = Arc::clone(&owner);
    let dispatch: Dispatch = Box::new(move |effect| {
        let mut original = dispatch_owner
            .lock()
            .map_err(|_| failure(contract::ErrorCode::BackendFailure, "UCI owner poisoned"))?;
        if matches!(&effect, Effect::Start { .. }) && observer.closed_under_owner_lock() {
            return Err(failure(
                contract::ErrorCode::Canceled,
                "priority admission is closed",
            )
            .into());
        }
        original.dispatch(effect, &sender)
    });
    let driver = AsyncServeDriver::new(
        session,
        events,
        output,
        max_output_bytes,
        poll_interval,
        handler,
        dispatch,
    );
    Ok(AsyncEngineRun {
        driver,
        _owner: owner,
        timer_stop,
        timer: Some(timer),
        timer_joined: false,
        cleanup_failure: None,
        timer_panic: None,
        output_recovery: None,
        _priority_close_owner: close_owner,
        close,
        last_exit: None,
    })
}

impl<O: Write + Send + 'static, D: Write + Send + 'static> AsyncEngineRun<O, D> {
    /// No new drive is allowed after failure, unknown or output close. A clean
    /// yield alone may resume the same driver. The previous failure is retained.
    pub fn drive_until(&mut self, until: Instant) -> Option<AsyncServeStatus> {
        if self.last_exit.as_ref().is_some_and(|exit| {
            exit.status != AsyncServeStatus::Yielded
                || exit.failure.is_some()
                || !exit.cleanup_failures.is_empty()
        }) {
            return None;
        }
        let close = self.close.clone();
        let exit = self.driver.drive(|| {
            let observed = close.observe();
            if observed.application_failure.is_some() || observed.owner_closed {
                AsyncControl::RetainWithoutClose
            } else if observed.admission_closed && observed.cancel_acknowledgment.is_some() {
                AsyncControl::Shutdown
            } else if Instant::now() >= until {
                AsyncControl::Yield
            } else {
                AsyncControl::Continue
            }
        });
        let status = exit.status;
        self.last_exit = Some(exit);
        Some(status)
    }
    /// Advance only the original CPU worker/timer cleanup; never waits for a
    /// writer or declares native/GPU completion. On failure the original error
    /// or panic payload remains in this run and this method does not retry.
    pub fn try_join_timer_after_workers(&mut self) -> EngineTimerJoinStatus {
        if self.cleanup_failure.is_some() || self.timer_panic.is_some() {
            return EngineTimerJoinStatus::Failed;
        }
        if self.timer_joined {
            return EngineTimerJoinStatus::Joined;
        }
        if !self.last_exit.as_ref().is_some_and(|exit| {
            exit.status == AsyncServeStatus::OutputQueued
                && exit.failure.is_none()
                && exit.cleanup_failures.is_empty()
        }) || !self.driver.session_closed()
            || !self.driver.pending().is_empty()
        {
            return EngineTimerJoinStatus::NotReady;
        }
        {
            let mut owner = match self._owner.try_lock() {
                Ok(owner) => owner,
                Err(std::sync::TryLockError::WouldBlock) => {
                    return EngineTimerJoinStatus::WaitingWorkers;
                }
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    self.cleanup_failure = Some(
                        failure(
                            contract::ErrorCode::BackendFailure,
                            "UCI owner poisoned during cleanup",
                        )
                        .into(),
                    );
                    return EngineTimerJoinStatus::Failed;
                }
            };
            if owner.driver.is_some() {
                // The independent search-driver finish_session fence is not
                // implemented by this LC0-only cleanup path.
                return EngineTimerJoinStatus::UnsupportedDriver;
            }
            if let Err(error) = owner.reap_workers(true) {
                self.cleanup_failure = Some(error);
                return EngineTimerJoinStatus::Failed;
            }
            // The legacy owner retains the last ActiveBinding as historical
            // cancellation authority even after workers join; it is not a
            // live-worker counter and must not block timer join forever.
            if !owner.workers.is_empty() {
                return EngineTimerJoinStatus::WaitingWorkers;
            }
            if owner.fence_failure.is_some() || !owner.pending_failures.is_empty() {
                return EngineTimerJoinStatus::RetainedFailure;
            }
            match owner.diagnostics.try_lock() {
                Ok(receipts) if !receipts.is_empty() => {
                    return EngineTimerJoinStatus::RetainedFailure;
                }
                Ok(_) => {}
                Err(std::sync::TryLockError::WouldBlock) => {
                    return EngineTimerJoinStatus::WaitingWorkers;
                }
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    self.cleanup_failure = Some(
                        failure(
                            contract::ErrorCode::BackendFailure,
                            "worker diagnostics owner poisoned during cleanup",
                        )
                        .into(),
                    );
                    return EngineTimerJoinStatus::Failed;
                }
            }
        }
        self.timer_stop.store(true, Ordering::Release);
        if !self.timer.as_ref().is_some_and(JoinHandle::is_finished) {
            return EngineTimerJoinStatus::WaitingTimer;
        }
        match self
            .timer
            .take()
            .expect("finished original timer retained")
            .join()
        {
            Ok(()) => {
                self.timer_joined = true;
                EngineTimerJoinStatus::Joined
            }
            Err(payload) => {
                self.timer_panic = Some(payload);
                EngineTimerJoinStatus::Failed
            }
        }
    }
    pub fn cleanup_failure(&self) -> Option<&EngineError> {
        self.cleanup_failure.as_ref()
    }
    pub fn has_original_timer_panic(&self) -> bool {
        self.timer_panic.is_some()
    }

    pub fn last_exit(&self) -> Option<&AsyncServeExit<EngineError>> {
        self.last_exit.as_ref()
    }
    pub fn close_observation(&self) -> PriorityCloseObservation {
        self.close.observe()
    }
    pub fn output_observation(&self) -> Option<OutputSnapshot> {
        self.driver.output().map(AsyncOutputOwner::observation)
    }
    /// Keep the exact writer recovery or panic with this same engine run.
    /// A joined writer is not proof of delivery or native physical completion.
    pub fn try_recover_writer(&mut self) -> EngineWriterRecoveryStatus {
        if let Some(recovered) = &self.output_recovery {
            return if recovered.is_ok() {
                EngineWriterRecoveryStatus::Recovered
            } else {
                EngineWriterRecoveryStatus::WorkerPanicked
            };
        }
        match self.driver.try_finish_output() {
            OutputRecoveryPoll::DriverNotSealed => EngineWriterRecoveryStatus::DriverNotSealed,
            OutputRecoveryPoll::Pending => EngineWriterRecoveryStatus::Pending,
            OutputRecoveryPoll::Recovered(original) => {
                self.output_recovery = Some(Ok(original));
                EngineWriterRecoveryStatus::Recovered
            }
            OutputRecoveryPoll::WorkerPanicked(payload) => {
                self.output_recovery = Some(Err(payload));
                EngineWriterRecoveryStatus::WorkerPanicked
            }
            OutputRecoveryPoll::AlreadyRecovered => {
                EngineWriterRecoveryStatus::OriginalRecoveryMissing
            }
        }
    }
    pub fn output_recovery(
        &self,
    ) -> Option<&Result<OutputRecovery<O, D>, Box<dyn std::any::Any + Send>>> {
        self.output_recovery.as_ref()
    }
    pub fn session_closed(&self) -> bool {
        self.driver.session_closed()
    }
    pub fn pending_count(&self) -> usize {
        self.driver.pending().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refused_start_recovers_original_unstarted_writer() {
        struct NeverFactory;
        impl EvaluatorFactory for NeverFactory {
            fn profile(&self) -> EvaluatorProfile {
                panic!("guard must reject before factory use")
            }
            fn create(
                &self,
                _: ProcessClock,
                _: SearchAuthority,
            ) -> Result<Box<dyn ManagedEvaluator>, contract::ContractError> {
                panic!("guard must reject before runtime creation")
            }
            fn input_key(
                &self,
                _: &RulesState,
                _: &[contract::Move],
            ) -> Result<contract::EvalInputKey, contract::ContractError> {
                panic!("guard must reject before encoding")
            }
        }
        let factory: Arc<dyn EvaluatorFactory> = Arc::new(NeverFactory);
        let mut process = EngineProcess::new(
            factory.clone(),
            Arc::new(OwnerRegistry::default()),
            ProcessClock::new(contract::ProcessEpoch(81)),
        );
        let close = process.install_priority_close().unwrap();
        let (sender, events) = event_channel();
        let output = AsyncOutputOwner::start(Vec::<u8>::new(), Vec::<u8>::new(), 1, 256)
            .ok()
            .unwrap();
        let mut rejected = match start_async(
            events,
            sender,
            process,
            EngineSettings::default(),
            output,
            close,
            256,
            Duration::from_millis(1),
        ) {
            Err(failure) => failure,
            Ok(_) => panic!("unsettled custody guard must reject"),
        };
        assert_eq!(
            rejected.try_recover_unstarted_writer(),
            UnstartedWriterRecoveryStatus::CloseNotRequested
        );
        assert!(rejected.request_unstarted_writer_close());
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match rejected.try_recover_unstarted_writer() {
                UnstartedWriterRecoveryStatus::Recovered => break,
                UnstartedWriterRecoveryStatus::Pending if Instant::now() < deadline => {
                    std::thread::yield_now()
                }
                _ => panic!("original output worker failed to join"),
            }
        }
        let original = rejected
            .unstarted_writer_recovery()
            .unwrap()
            .as_ref()
            .ok()
            .unwrap();
        assert!(original.protocol.is_empty());
        assert!(original.diagnostics.is_empty());
        assert!(original.failure.is_none());
        let AsyncEngineStartFailure::Unstarted { process, .. } = &rejected else {
            panic!("wrong owner")
        };
        assert!(Arc::ptr_eq(process.factory.as_ref().unwrap(), &factory));
    }
}
