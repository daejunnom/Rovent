//! Bounded drain of the original native service after a V6 startup failure.
//!
//! This is deliberately unconnected. The caller must retain this object and
//! its enclosing pipeline while completion is unknown. It never closes an
//! external CPU worker or archive, ends the process, or fences an entire GPU.
use super::super::{ActualBackend, ActualService};
use crate::pals_native::ticketed::{
    TicketNativeBackend, TicketNativeEvent, TicketNativeFinishReport, TicketNativeRecoveryKey,
    TicketServiceError,
};
use rz_runtime::contracts::ContractSystemClock;
use rz_search::pals::coordinator::RoleFailure;
use rz_search::pals::engine::adaptive_port::AdaptivePortQuestion;
use rz_search::pals::review_context::ReviewView;
use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind},
    time::Instant,
};

type OriginalEvent = TicketNativeEvent<
    <ActualBackend as TicketNativeBackend<ReviewView, AdaptivePortQuestion>>::Output,
>;

/// Observation of this one native worker only. A late observation does not
/// satisfy the original shutdown deadline or erase the startup failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FailedNativeDrainObservation {
    Pending,
    DeadlineElapsed,
    NativePhysicalConfirmed,
    NativePhysicalConfirmedObservedLate,
    PhysicalUnknown,
    UnexpectedOrDuplicateEvent,
    JoinFailed,
    Panicked,
    OwnerUnavailable,
    OwnerMismatch,
}

/// Owned evidence can remain in the pipeline beside its original failure.
/// The original service's process-local recovery key binds every later call;
/// a missing or different key cannot authorize event consumption.
#[must_use = "retain the original failed startup owner while completion is unknown"]
pub(super) struct V6FailedStartupDrain {
    original_owner_key: TicketNativeRecoveryKey,
    requires_report: Option<bool>,
    until: Instant,
    started: bool,
    ready: Option<ContractSystemClock>,
    startup_failure: Option<RoleFailure>,
    finished: Option<bool>,
    finished_observed_late: bool,
    joined: bool,
    join_observed_late: bool,
    queue_empty_after_join: bool,
    empty_observed_late: bool,
    join_failure: Option<TicketServiceError>,
    report: Option<TicketNativeFinishReport>,
    report_snapshot: Option<TicketNativeFinishReport>,
    /// Retains the actual original event, including any non-clone permit.
    unexpected: Option<OriginalEvent>,
    panic_payload: Option<Box<dyn Any + Send>>,
}

impl V6FailedStartupDrain {
    pub(super) fn new(
        service: &ActualService,
        until: Instant,
    ) -> Result<Self, FailedNativeDrainObservation> {
        let original_owner_key = service
            .recovery_key()
            .ok_or(FailedNativeDrainObservation::OwnerUnavailable)?;
        Ok(Self {
            original_owner_key,
            requires_report: None,
            until,
            started: false,
            ready: None,
            startup_failure: None,
            finished: None,
            finished_observed_late: false,
            joined: false,
            join_observed_late: false,
            queue_empty_after_join: false,
            empty_observed_late: false,
            join_failure: None,
            report: None,
            report_snapshot: None,
            unexpected: None,
            panic_payload: None,
        })
    }

    fn check_owner(&self, service: &ActualService) -> Result<(), FailedNativeDrainObservation> {
        let key = service
            .recovery_key()
            .ok_or(FailedNativeDrainObservation::OwnerUnavailable)?;
        if self.original_owner_key.same_owner(&key) {
            Ok(())
        } else {
            Err(FailedNativeDrainObservation::OwnerMismatch)
        }
    }

    pub(super) fn original_startup_failure(&self) -> Option<&RoleFailure> {
        self.startup_failure.as_ref()
    }
    pub(super) fn original_ready_clock(&self) -> Option<&ContractSystemClock> {
        self.ready.as_ref()
    }
    pub(super) fn original_finished(&self) -> Option<bool> {
        self.finished
    }
    pub(super) fn original_worker_joined(&self) -> bool {
        self.joined
    }
    pub(super) fn observed_after_deadline(&self) -> bool {
        self.finished_observed_late || self.join_observed_late || self.empty_observed_late
    }
    pub(super) fn original_report(&self) -> Option<&TicketNativeFinishReport> {
        self.report.as_ref()
    }
    pub(super) fn report_snapshot(&self) -> Option<&TicketNativeFinishReport> {
        self.report_snapshot.as_ref()
    }
    pub(super) fn unexpected_event(&self) -> Option<&OriginalEvent> {
        self.unexpected.as_ref()
    }
    pub(super) fn join_error(&self) -> Option<&TicketServiceError> {
        self.join_failure.as_ref()
    }
    pub(super) fn panic_payload(&self) -> Option<&(dyn Any + Send)> {
        self.panic_payload.as_deref()
    }

    /// Issue shutdown once using the original deadline; do not renew it.
    /// An expired deadline causes no new request and gives no physical proof.
    pub(super) fn begin(&mut self, service: &mut ActualService) -> FailedNativeDrainObservation {
        if let Err(status) = self.check_owner(service) {
            return status;
        }
        if self.started {
            return self.observation();
        }
        if self.panic_payload.is_some() {
            return FailedNativeDrainObservation::Panicked;
        }
        if Instant::now() >= self.until {
            return FailedNativeDrainObservation::DeadlineElapsed;
        }
        self.started = true;
        self.requires_report = Some(service.requires_finish_report());
        match catch_unwind(AssertUnwindSafe(|| service.begin_shutdown(self.until))) {
            Ok(()) => FailedNativeDrainObservation::Pending,
            Err(payload) => {
                self.panic_payload = Some(payload);
                FailedNativeDrainObservation::Panicked
            }
        }
    }

    /// Inspect at most one queued event and attempt at most one nonblocking
    /// join per call. Observations after the original deadline remain late;
    /// a failed join is sticky and is never retried as a later success.
    pub(super) fn poll_once(
        &mut self,
        service: &mut ActualService,
    ) -> FailedNativeDrainObservation {
        if let Err(status) = self.check_owner(service) {
            return status;
        }
        if self.panic_payload.is_some() {
            return FailedNativeDrainObservation::Panicked;
        }
        if !self.started {
            return self.observation();
        }
        if self.unexpected.is_some() {
            return FailedNativeDrainObservation::UnexpectedOrDuplicateEvent;
        }
        // An empty poll before a concurrent join is not an empty poll after
        // join: the worker could still publish between the two operations.
        let was_joined = self.joined;
        let event = match catch_unwind(AssertUnwindSafe(|| service.poll())) {
            Ok(value) => value,
            Err(payload) => {
                self.panic_payload = Some(payload);
                return FailedNativeDrainObservation::Panicked;
            }
        };
        let queue_was_empty = event.is_none();
        match event {
            Some(TicketNativeEvent::Ready { clock }) if self.ready.is_none() => {
                self.ready = Some(clock)
            }
            Some(TicketNativeEvent::StartupFailed(error)) if self.startup_failure.is_none() => {
                self.startup_failure = Some(error)
            }
            Some(TicketNativeEvent::Finished {
                physical_shutdown_confirmed,
            }) if self.finished.is_none() => {
                self.finished = Some(physical_shutdown_confirmed);
                self.finished_observed_late = Instant::now() >= self.until;
                let result = catch_unwind(AssertUnwindSafe(|| {
                    (
                        service.finish_report_snapshot(),
                        service.take_finish_report(),
                    )
                }));
                match result {
                    Ok((snapshot, report)) => {
                        self.report_snapshot = snapshot;
                        self.report = report;
                    }
                    Err(payload) => {
                        self.panic_payload = Some(payload);
                        return FailedNativeDrainObservation::Panicked;
                    }
                }
            }
            Some(other) => {
                self.unexpected = Some(other);
                return FailedNativeDrainObservation::UnexpectedOrDuplicateEvent;
            }
            None => {}
        }
        if !self.joined && self.join_failure.is_none() {
            match catch_unwind(AssertUnwindSafe(|| service.try_join())) {
                Ok(Ok(true)) => {
                    self.joined = true;
                    self.join_observed_late = Instant::now() >= self.until;
                }
                Ok(Ok(false)) => {}
                Ok(Err(error)) => self.join_failure = Some(error),
                Err(payload) => {
                    self.panic_payload = Some(payload);
                    return FailedNativeDrainObservation::Panicked;
                }
            }
        }
        if was_joined && queue_was_empty {
            self.queue_empty_after_join = true;
            self.empty_observed_late = Instant::now() >= self.until;
        }
        self.observation()
    }

    pub(super) fn observation(&self) -> FailedNativeDrainObservation {
        if self.panic_payload.is_some() {
            return FailedNativeDrainObservation::Panicked;
        }
        if self.unexpected.is_some() {
            return FailedNativeDrainObservation::UnexpectedOrDuplicateEvent;
        }
        if self.join_failure.is_some() {
            return FailedNativeDrainObservation::JoinFailed;
        }
        if self.finished == Some(false) {
            return FailedNativeDrainObservation::PhysicalUnknown;
        }
        if self.finished == Some(true) && self.joined && self.queue_empty_after_join {
            let snapshot = self.report_snapshot.as_ref();
            if (self.requires_report == Some(true) && (snapshot.is_none() || self.report.is_none()))
                || (self.requires_report == Some(false) && snapshot.is_some())
                || self.requires_report.is_none()
                || snapshot.is_some_and(|report| {
                    !report.physical_shutdown_confirmed || report.physical_failure.is_some()
                })
                || snapshot
                    .zip(self.report.as_ref())
                    .is_some_and(|(a, b)| a != b)
                || (snapshot.is_none() && self.report.is_some())
            {
                return FailedNativeDrainObservation::PhysicalUnknown;
            }
            return if self.finished_observed_late
                || self.join_observed_late
                || self.empty_observed_late
            {
                FailedNativeDrainObservation::NativePhysicalConfirmedObservedLate
            } else {
                FailedNativeDrainObservation::NativePhysicalConfirmed
            };
        }
        if Instant::now() >= self.until {
            FailedNativeDrainObservation::DeadlineElapsed
        } else {
            FailedNativeDrainObservation::Pending
        }
    }
}
