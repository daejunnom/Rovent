//! Opt-in original engine custody for resumable, nonblocking UCI output.
//! A returned status is neither physical completion nor a writer join.
use super::*;
use crate::runner::{
    async_events::{AsyncControl, AsyncServeDriver, AsyncServeExit, AsyncServeStatus},
    async_output::{AsyncOutputOwner, OutputSnapshot},
};

type Handler =
    Box<dyn FnMut(&mut Session<RulesUciPort>, Event) -> SessionResult<RulesSearchPosition> + Send>;
type Dispatch = Box<dyn FnMut(Effect<RulesSearchPosition>) -> Result<(), EngineError> + Send>;
type Driver<O, D> = AsyncServeDriver<RulesUciPort, O, D, Handler, Dispatch>;

/// All unstarted original inputs are returned on an ordinary setup rejection.
pub struct AsyncEngineStartFailure<O: Write + Send + 'static, D: Write + Send + 'static> {
    pub error: EngineError,
    pub process: EngineProcess,
    pub events: Receiver<Event>,
    pub sender: SyncSender<Event>,
    pub output: AsyncOutputOwner<O, D>,
    pub close: PriorityCloseHandle,
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
    if max_output_bytes == 0
        || poll_interval.is_zero()
        || !process
            .priority_close
            .as_ref()
            .is_some_and(|owner| owner.matches(&close))
    {
        return Err(AsyncEngineStartFailure {
            error: failure(
                contract::ErrorCode::InvalidInput,
                "async output bounds or original close owner invalid",
            )
            .into(),
            process,
            events,
            sender,
            output,
            close,
        });
    }
    let prepared = match prepare_engine(events, sender, process, settings) {
        Ok(prepared) => prepared,
        Err(rejected) => {
            return Err(AsyncEngineStartFailure {
                error: rejected.error,
                process: rejected.process,
                events: rejected.events,
                sender: rejected.sender,
                output,
                close,
            });
        }
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
        priority_close_owner,
        dispatch_close_observer,
    } = prepared;
    // Validated against this process before preparation. The owner is moved,
    // never replaced with another handle that happens to share an epoch.
    let close_owner = priority_close_owner.expect("validated original close owner");
    let observer = dispatch_close_observer.expect("validated original close observer");
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
    pub fn output_observation(&self) -> OutputSnapshot {
        self.driver.output().observation()
    }
    pub fn session_closed(&self) -> bool {
        self.driver.session_closed()
    }
    pub fn pending_count(&self) -> usize {
        self.driver.pending().len()
    }
}
