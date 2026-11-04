//! Recover the same factory custody after an explicitly before-native failure.
//!
//! This is an unconnected building block. The enclosing supervisor must retain
//! this object and its original error. Recovery is neither a GPU-wide fence nor
//! evidence that external CPU, archive, or process owners have closed.

use super::super::{ActualBackend, ActualService};
use crate::pals_native::{
    startup_system_v6::PalsSystemFactoryCustodyV6,
    ticketed::{
        NativeFactoryStartupStatus, TicketNativeBackend, TicketNativeEvent,
        TicketNativeRecoveryKey, TicketRecoveryError, TicketRecoveryStatus, TicketUndeliveredEvent,
    },
};
use rz_search::pals::{engine::adaptive_port::AdaptivePortQuestion, review_context::ReviewView};
use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind},
    time::Instant,
};

pub(super) type OriginalBeforeNativeEvent = TicketNativeEvent<
    <ActualBackend as TicketNativeBackend<ReviewView, AdaptivePortQuestion>>::Output,
>;

/// One finite observation, not permission to destroy the surrounding owners.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BeforeNativeRecoveryObservation {
    Pending,
    EventRetained,
    EventRetainedLate,
    PanicPayloadRetained,
    PanicPayloadRetainedLate,
    Recovered,
    RecoveredLate,
    Busy,
    DeadlineExpired,
    NoRecoveryKey,
    FactoryNotBeforeNative,
    RegistryNotBeforeNative,
    ResourceUnavailable,
    EventAlreadyConsumed,
    Blocked(TicketRecoveryError),
    Panicked,
}

/// Owns exactly the passed service until both the factory and its recovery
/// slot report BeforeNative. Afterwards it keeps the same key, opaque event
/// handle, recovered events and the original factory custody.
#[must_use = "retain the original recovery owner while its outcome is unresolved"]
pub(super) struct BeforeNativeRecoveryV6 {
    service: Option<ActualService>,
    completion_signal: rz_runtime::CompletionSignal,
    original_key: Option<TicketNativeRecoveryKey>,
    until: Instant,
    handover_attempted: bool,
    event_observed_late: bool,
    recovery_observed_late: bool,
    original_status: Option<NativeFactoryStartupStatus>,
    registry_status: Option<TicketRecoveryStatus>,
    pending_event: Option<TicketUndeliveredEvent>,
    events: Vec<OriginalBeforeNativeEvent>,
    factory_panics: Vec<Box<dyn Any + Send>>,
    recovered: Option<PalsSystemFactoryCustodyV6>,
    last_recovery_error: Option<TicketRecoveryError>,
    blocked: Option<BeforeNativeRecoveryObservation>,
    recovery_panic: Option<Box<dyn Any + Send>>,
}

impl BeforeNativeRecoveryV6 {
    /// Accept the service exactly once, retaining it even when no key exists.
    /// `until` is the original registered cleanup deadline and is never reset.
    pub(super) fn new(service: ActualService, until: Instant) -> Self {
        let original_key = service.recovery_key();
        let completion_signal = service.completion_signal();
        Self {
            service: Some(service),
            completion_signal,
            original_key,
            until,
            handover_attempted: false,
            event_observed_late: false,
            recovery_observed_late: false,
            original_status: None,
            registry_status: None,
            pending_event: None,
            events: Vec::new(),
            factory_panics: Vec::new(),
            recovered: None,
            last_recovery_error: None,
            blocked: None,
            recovery_panic: None,
        }
    }

    pub(super) fn original_key(&self) -> Option<&TicketNativeRecoveryKey> {
        self.original_key.as_ref()
    }
    /// Read only this same original registry slot, even after the original
    /// cleanup deadline. This does not take an event, move the factory owner,
    /// submit recovery work, or turn a BeforeNative observation into release.
    pub(super) fn passive_original_status(
        &self,
    ) -> Result<TicketRecoveryStatus, TicketRecoveryError> {
        let key = self
            .original_key
            .as_ref()
            .ok_or(TicketRecoveryError::InvalidKey)?;
        if let Some(service) = self.service.as_ref() {
            let current = service
                .recovery_key()
                .ok_or(TicketRecoveryError::InvalidKey)?;
            if !key.same_owner(&current) {
                return Err(TicketRecoveryError::InvalidKey);
            }
        }
        key.status()
    }
    /// Same worker signal retained before the receiver is handed to the
    /// registry. It carries no event, completion proof, or authority; some
    /// registry transitions do not notify, so callers must recheck with a
    /// finite timeout and the original deadline.
    pub(super) fn completion_signal(&self) -> rz_runtime::CompletionSignal {
        self.completion_signal.clone()
    }
    pub(super) fn service_retained(&self) -> bool {
        self.service.is_some()
    }
    pub(super) fn handover_attempted(&self) -> bool {
        self.handover_attempted
    }
    pub(super) fn event_observed_late(&self) -> bool {
        self.event_observed_late
    }
    pub(super) fn recovery_observed_late(&self) -> bool {
        self.recovery_observed_late
    }
    pub(super) fn original_status(&self) -> Option<NativeFactoryStartupStatus> {
        self.original_status
    }
    pub(super) fn registry_status(&self) -> Option<TicketRecoveryStatus> {
        self.registry_status
    }
    pub(super) fn events(&self) -> &[OriginalBeforeNativeEvent] {
        &self.events
    }
    pub(super) fn factory_panics(&self) -> &[Box<dyn Any + Send>] {
        &self.factory_panics
    }
    pub(super) fn recovered(&self) -> Option<&PalsSystemFactoryCustodyV6> {
        self.recovered.as_ref()
    }
    pub(super) fn last_recovery_error(&self) -> Option<TicketRecoveryError> {
        self.last_recovery_error
    }
    pub(super) fn recovery_panic(&self) -> Option<&(dyn Any + Send)> {
        self.recovery_panic.as_deref()
    }
    pub(super) fn pending_event(&self) -> Option<&TicketUndeliveredEvent> {
        self.pending_event.as_ref()
    }

    /// One caller-driven step; only Busy is retryable. No event, key, owner,
    /// deadline or physical observation is recreated on a later call.
    pub(super) fn poll_once(&mut self) -> BeforeNativeRecoveryObservation {
        if self.recovered.is_some() {
            return if self.recovery_observed_late {
                BeforeNativeRecoveryObservation::RecoveredLate
            } else {
                BeforeNativeRecoveryObservation::Recovered
            };
        }
        if let Some(blocked) = self.blocked {
            return blocked;
        }
        if Instant::now() >= self.until {
            return BeforeNativeRecoveryObservation::DeadlineExpired;
        }
        match catch_unwind(AssertUnwindSafe(|| self.poll_inner())) {
            Ok(observation) => observation,
            Err(payload) => {
                self.recovery_panic = Some(payload);
                self.blocked = Some(BeforeNativeRecoveryObservation::Panicked);
                BeforeNativeRecoveryObservation::Panicked
            }
        }
    }

    fn record_error(&mut self, error: TicketRecoveryError) -> BeforeNativeRecoveryObservation {
        self.last_recovery_error = Some(error);
        match error {
            TicketRecoveryError::Busy => BeforeNativeRecoveryObservation::Busy,
            TicketRecoveryError::TimedOut => BeforeNativeRecoveryObservation::DeadlineExpired,
            other => {
                let blocked = BeforeNativeRecoveryObservation::Blocked(other);
                self.blocked = Some(blocked);
                blocked
            }
        }
    }

    fn poll_inner(&mut self) -> BeforeNativeRecoveryObservation {
        let Some(key) = self.original_key.as_ref() else {
            self.blocked = Some(BeforeNativeRecoveryObservation::NoRecoveryKey);
            return BeforeNativeRecoveryObservation::NoRecoveryKey;
        };
        if let Some(service) = self.service.as_ref() {
            let status = service.startup_status();
            self.original_status = Some(status);
            match status {
                NativeFactoryStartupStatus::Pending => {
                    return BeforeNativeRecoveryObservation::Pending;
                }
                NativeFactoryStartupStatus::BeforeNative => {}
                _ => {
                    self.blocked = Some(BeforeNativeRecoveryObservation::FactoryNotBeforeNative);
                    return BeforeNativeRecoveryObservation::FactoryNotBeforeNative;
                }
            }
            let Some(current_key) = service.recovery_key() else {
                self.blocked = Some(BeforeNativeRecoveryObservation::NoRecoveryKey);
                return BeforeNativeRecoveryObservation::NoRecoveryKey;
            };
            if !key.same_owner(&current_key) {
                let blocked =
                    BeforeNativeRecoveryObservation::Blocked(TicketRecoveryError::InvalidKey);
                self.blocked = Some(blocked);
                return blocked;
            }
            match key.status() {
                Ok(status) => {
                    self.registry_status = Some(status);
                    match status {
                        TicketRecoveryStatus::Reserved | TicketRecoveryStatus::Running => {
                            return BeforeNativeRecoveryObservation::Pending;
                        }
                        TicketRecoveryStatus::BeforeNative => {}
                        _ => {
                            self.blocked =
                                Some(BeforeNativeRecoveryObservation::RegistryNotBeforeNative);
                            return BeforeNativeRecoveryObservation::RegistryNotBeforeNative;
                        }
                    }
                }
                Err(error) => return self.record_error(error),
            }
            // The registry already owns the same receiver; this releases its
            // reading authority without creating a replacement receiver.
            self.handover_attempted = true;
            drop(self.service.take().expect("checked original service"));
            return BeforeNativeRecoveryObservation::Pending;
        }

        if self.pending_event.is_none() {
            match key.take_event(self.until) {
                Ok(Some(event)) => self.pending_event = Some(event),
                Ok(None) => {
                    return match key.take_before_native::<PalsSystemFactoryCustodyV6>(self.until) {
                        Ok(owner) => {
                            self.recovered = Some(owner);
                            if Instant::now() >= self.until {
                                self.recovery_observed_late = true;
                                BeforeNativeRecoveryObservation::RecoveredLate
                            } else {
                                BeforeNativeRecoveryObservation::Recovered
                            }
                        }
                        Err(error) => self.record_error(error),
                    };
                }
                Err(error) => return self.record_error(error),
            }
        }
        // Reserve before consuming an opaque handle so allocation failure
        // leaves the exact original event or factory panic in the registry.
        if self.events.try_reserve(1).is_err() || self.factory_panics.try_reserve(1).is_err() {
            self.blocked = Some(BeforeNativeRecoveryObservation::ResourceUnavailable);
            return BeforeNativeRecoveryObservation::ResourceUnavailable;
        }
        let handle = self.pending_event.as_ref().expect("retained event handle");
        match handle.take::<OriginalBeforeNativeEvent>() {
            Ok(Some(event)) => {
                self.events.push(event);
                self.pending_event = None;
                if Instant::now() >= self.until {
                    self.event_observed_late = true;
                    BeforeNativeRecoveryObservation::EventRetainedLate
                } else {
                    BeforeNativeRecoveryObservation::EventRetained
                }
            }
            Ok(None) => {
                self.blocked = Some(BeforeNativeRecoveryObservation::EventAlreadyConsumed);
                BeforeNativeRecoveryObservation::EventAlreadyConsumed
            }
            Err(TicketRecoveryError::WrongOwnerType) => {
                match handle.take::<Box<dyn Any + Send>>() {
                    Ok(Some(payload)) => {
                        self.factory_panics.push(payload);
                        self.pending_event = None;
                        if Instant::now() >= self.until {
                            self.event_observed_late = true;
                            BeforeNativeRecoveryObservation::PanicPayloadRetainedLate
                        } else {
                            BeforeNativeRecoveryObservation::PanicPayloadRetained
                        }
                    }
                    Ok(None) => {
                        self.blocked = Some(BeforeNativeRecoveryObservation::EventAlreadyConsumed);
                        BeforeNativeRecoveryObservation::EventAlreadyConsumed
                    }
                    Err(error) => self.record_error(error),
                }
            }
            Err(error) => self.record_error(error),
        }
    }
}
