//! Explicit /3 system-CUDA child entry. Historical /1 and /2 inputs are never
//! projected into this path. Runtime and process owners remain distinct.

use super::*;
#[cfg(target_os = "linux")]
use crate::rovent_bootstrap::process_boundary::{
    RoventProcessBoundaryInputsV3, SourceEvidencePolicy, VerifiedOwnedRoventProcessBoundaryV3,
    verify_actual_owned_process_boundary_v3_with_policy,
};
#[cfg(target_os = "linux")]
use rz_experiments::NativeRuntimeSourceV6;
use rz_experiments::RoventInputLockV3;
#[cfg(target_os = "linux")]
#[path = "v3_scope.rs"]
pub(super) mod v3_scope;

// WIP: the supervisor has not yet connected its exclusive inherited-FD
// ownership, per-packet credentials, and single-stdin serialization. Keep
// this optional channel disabled until those actual boundaries are validated.
#[cfg(target_os = "linux")]
const GO_OBSERVER_PARENT_HANDOFF_CONNECTED: bool = false;

/// V3 launch errors can retain the exact unstarted factory owner. Absence of
/// native entry is scoped to this attempt; it is not a GPU completion proof.
#[cfg(target_os = "linux")]
pub(super) enum RoventV3LaunchFailure {
    Ordinary(RoventLaunchError),
    Unstarted {
        error: RoventLaunchError,
        owner: super::super::RoventUnstartedNativeCustodyV3,
    },
    /// Physical native completion was observed on the original owner. The
    /// failure and original events remain caller-owned; this is not success.
    Settled {
        error: RoventLaunchError,
        owner: V3SettledFactoryOwner,
    },
    /// The original launch/product scope is returned to a process owner. This
    /// is not a physical fence, a cross-process lease, or permission to exit.
    Quarantined {
        error: RoventLaunchError,
        owner: RoventV3QuarantinedOwner,
    },
}

#[cfg(target_os = "linux")]
pub(super) struct RoventV3QuarantinedOwner {
    scope_key: v3_scope::RoventV3ScopeKey,
    recovery_key: Option<crate::pals_native::ticketed::TicketNativeRecoveryKey>,
    cleanup_until: Instant,
}

#[cfg(target_os = "linux")]
pub enum V3ControlDirectory {
    Unavailable,
    Busy,
    CloneFailed(std::io::Error),
    Ready(Dir),
}

#[cfg(target_os = "linux")]
impl v3_scope::RoventV3ScopeKey {
    /// Clone the exact originally opened writer directory capability. This is
    /// neither peer authentication nor a device or physical completion proof.
    pub fn clone_original_control_directory(&self) -> V3ControlDirectory {
        match self.try_cloned_pin::<SystemReceiptWriter>() {
            v3_scope::V3ScopeAccess::Missing => V3ControlDirectory::Unavailable,
            v3_scope::V3ScopeAccess::Busy => V3ControlDirectory::Busy,
            v3_scope::V3ScopeAccess::Present(writer) => match writer.directory.try_clone() {
                Ok(directory) => V3ControlDirectory::Ready(directory),
                Err(error) => V3ControlDirectory::CloneFailed(error),
            },
        }
    }
    /// The original registered cleanup deadline; None means missing or busy.
    pub fn cleanup_until(&self) -> Option<Instant> {
        self.cloned_pin::<RoventStartupControl>()
            .map(|control| control.cleanup_until)
    }
    /// Ask only the same original native-ticket owner for a bounded
    /// reobservation. This cannot close or release the surrounding product,
    /// external process, input descriptors, or scope registry entry.
    pub fn reobserve_native(&self, until: Instant) -> v3_scope::V3NativeReobservation {
        use v3_scope::{V3NativeReobservation as Report, V3ScopeAccess};
        let observation = match self.try_observe() {
            V3ScopeAccess::Missing => return Report::ScopeMissing,
            V3ScopeAccess::Busy => return Report::ScopeBusy,
            V3ScopeAccess::Present(observation) => observation,
        };
        let state = observation.state;
        let mut native_key = observation.recovery_key;
        macro_rules! find_custody {
            ($ty:ty, $get:expr) => {
                if native_key.is_none() {
                    match self.try_cloned_pin::<std::sync::Mutex<$ty>>() {
                        V3ScopeAccess::Busy => return Report::ScopeBusy,
                        V3ScopeAccess::Missing => {}
                        V3ScopeAccess::Present(pin) => {
                            let mut owner = match pin.try_lock() {
                                Ok(guard) => guard,
                                Err(std::sync::TryLockError::Poisoned(error)) => error.into_inner(),
                                Err(std::sync::TryLockError::WouldBlock) => {
                                    return Report::CustodyBusy { scope_state: state };
                                }
                            };
                            native_key = ($get)(&mut owner);
                        }
                    }
                }
            };
        }
        find_custody!(
            super::super::product::RoventOwnNnuecAssemblyV3,
            |owner: &mut super::super::product::RoventOwnNnuecAssemblyV3| owner
                .prepared_service()
                .and_then(|service| service.recovery_key())
        );
        find_custody!(
            super::super::product::RoventOwnBoardAssemblyV3,
            |owner: &mut super::super::product::RoventOwnBoardAssemblyV3| owner.recovery_key()
        );
        find_custody!(
            super::super::product::RoventPcAssemblyV3,
            |owner: &mut super::super::product::RoventPcAssemblyV3| owner
                .prepared_service()
                .and_then(|service| service.recovery_key())
        );
        find_custody!(
            super::super::product::RoventExternalAssemblyV3,
            |owner: &mut super::super::product::RoventExternalAssemblyV3| owner.recovery_key()
        );
        match native_key {
            Some(key) => match key.reobserve(until) {
                Ok(native_status) => Report::NativeObserved {
                    scope_state: state,
                    native_status,
                },
                Err(error) => Report::NativeError {
                    scope_state: state,
                    error,
                },
            },
            None => Report::NativeKeyUnavailable { scope_state: state },
        }
    }
}

#[cfg(target_os = "linux")]
impl RoventV3QuarantinedOwner {
    pub(super) fn cleanup_until(&self) -> Instant {
        self.cleanup_until
    }
    pub(super) fn recovery_key(
        &self,
    ) -> Option<&crate::pals_native::ticketed::TicketNativeRecoveryKey> {
        self.recovery_key.as_ref()
    }
    /// The original scope remains owned by this value. Reading a key does not
    /// establish physical completion or authorize dropping the scope.
    pub(super) fn retains_original(&self) -> Option<bool> {
        self.scope_key.observe().map(|scope| scope.parked_original)
    }

    pub(super) fn scope_key(&self) -> &v3_scope::RoventV3ScopeKey {
        &self.scope_key
    }
}

#[cfg(target_os = "linux")]
impl std::fmt::Debug for RoventV3LaunchFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ordinary(error) => f.debug_tuple("Ordinary").field(error).finish(),
            Self::Unstarted { error, .. } => f
                .debug_struct("Unstarted")
                .field("error", error)
                .field("original_owner_retained", &true)
                .finish(),
            Self::Settled { error, .. } => f
                .debug_struct("Settled")
                .field("error", error)
                .field("native_physical_confirmed", &true)
                .finish(),
            Self::Quarantined { error, owner } => f
                .debug_struct("Quarantined")
                .field("error", error)
                .field("original_owner_retained", &owner.retains_original())
                .field("recovery_key_available", &owner.recovery_key().is_some())
                .finish(),
        }
    }
}

#[cfg(target_os = "linux")]
impl From<RoventLaunchError> for RoventV3LaunchFailure {
    fn from(error: RoventLaunchError) -> Self {
        Self::Ordinary(error)
    }
}

#[cfg(target_os = "linux")]
impl RoventV3LaunchFailure {
    /// Historical caller compatibility only. Retaining process owners must
    /// use the typed route and must not use this lossy conversion.
    pub(super) fn into_legacy(self) -> RoventLaunchError {
        match self {
            Self::Ordinary(error)
            | Self::Unstarted { error, .. }
            | Self::Settled { error, .. }
            | Self::Quarantined { error, .. } => error,
        }
    }
}

#[cfg(target_os = "linux")]
enum V3FactoryRejection<Q: NativeQuestionProjection> {
    Ordinary(RoventLaunchError),
    Unstarted {
        error: RoventLaunchError,
        owner: super::super::RoventUnstartedNativeCustodyV3,
    },
    Settled {
        error: RoventLaunchError,
        owner: V3SettledFactoryOwner,
    },
    BeforeNativePhysicalPending {
        error: RoventLaunchError,
        recovered: V3BeforeNativeRecovered,
        recovery_key: Option<crate::pals_native::ticketed::TicketNativeRecoveryKey>,
        first_recovery_error: V3BeforeNativeAttemptError,
    },
    Unknown {
        error: RoventLaunchError,
        original: RoventNativeStartupRejection<Q>,
        recovery_key: Option<crate::pals_native::ticketed::TicketNativeRecoveryKey>,
        events: Vec<V3FactoryOriginalEvent>,
        before_native: bool,
        first_recovery_error: Option<V3BeforeNativeAttemptError>,
    },
}

#[cfg(target_os = "linux")]
pub enum V3FactoryOriginalEvent {
    Native(TicketNativeEvent<crate::pals_native::ticketed::VirtualReviewTicketOutput>),
    FinishReport(crate::pals_native::ticketed::TicketNativeFinishReport),
    /// Diagnostic copy when the same worker's original report was already
    /// moved to another consumer or disagrees with this observation.
    FinishReportSnapshot(crate::pals_native::ticketed::TicketNativeFinishReport),
    Panic(Box<dyn std::any::Any + Send>),
    /// The exact still-pending cell remains owned when its type is unknown.
    Opaque(crate::pals_native::ticketed::TicketUndeliveredEvent),
}

/// Original factory and terminal events returned on an observed pre-native
/// failure. This is not a statement about any unrelated device work.
#[cfg(target_os = "linux")]
pub struct V3BeforeNativeRecovered {
    pub factory: super::super::RoventSystemFactoryCustodyV3,
    pub events: Vec<V3FactoryOriginalEvent>,
    original_failure: Box<dyn std::any::Any + Send>,
}

#[cfg(target_os = "linux")]
impl V3BeforeNativeRecovered {
    /// The enclosing type is created only after the same ticket owner proves
    /// BeforeNative. A published/different physical owner is never canceled;
    /// the original reservation is put back on failure for later recovery.
    pub fn try_cancel_unpublished_physical(&mut self) -> bool {
        cancel_before_native_physical(&mut self.factory)
    }

    pub fn physical_reservation_pending(&self) -> bool {
        self.factory.physical_reservation.is_some()
    }

    pub fn downcast_original_failure<T: std::any::Any + Send>(
        self,
    ) -> Result<
        (
            Box<T>,
            super::super::RoventSystemFactoryCustodyV3,
            Vec<V3FactoryOriginalEvent>,
        ),
        Self,
    > {
        match self.original_failure.downcast::<T>() {
            Ok(original) => Ok((original, self.factory, self.events)),
            Err(original_failure) => Err(Self {
                original_failure,
                factory: self.factory,
                events: self.events,
            }),
        }
    }
}

#[cfg(target_os = "linux")]
fn cancel_before_native_physical(factory: &mut super::super::RoventSystemFactoryCustodyV3) -> bool {
    try_cancel_before_native_physical(factory).is_ok()
}

#[cfg(target_os = "linux")]
fn try_cancel_before_native_physical(
    factory: &mut super::super::RoventSystemFactoryCustodyV3,
) -> Result<(), rz_eval::worker::PhysicalReservationCancelError> {
    let Some(reservation) = factory.physical_reservation.take() else {
        return Ok(());
    };
    match reservation.try_cancel_unpublished() {
        Ok(()) => Ok(()),
        Err((original, reason)) => {
            factory.physical_reservation = Some(original);
            Err(reason)
        }
    }
}

/// Common representation across P-C, own and external startup failures.
/// `context` contains the exact original launch/product owners.
#[cfg(target_os = "linux")]
pub struct V3FactoryPending {
    original_failure: Option<Box<dyn std::any::Any + Send>>,
    context: Option<Box<dyn std::any::Any + Send>>,
    events: Vec<V3FactoryOriginalEvent>,
    key: Option<crate::pals_native::ticketed::TicketNativeRecoveryKey>,
    before_native: bool,
    recovered_factory: Option<super::super::RoventSystemFactoryCustodyV3>,
    first_recovery_error: Option<V3BeforeNativeAttemptError>,
}

#[cfg(target_os = "linux")]
impl V3FactoryPending {
    /// Inspect the exact retained events without consuming their owners.
    /// Opaque cells and panic payloads do not become serializable by being
    /// observed; the caller must preserve this pending value.
    pub fn events(&self) -> &[V3FactoryOriginalEvent] {
        &self.events
    }

    /// First observed failure while recovering this same owner, if any.
    pub fn first_recovery_error(&self) -> Option<V3BeforeNativeAttemptError> {
        self.first_recovery_error
    }

    pub fn original_failure(&self) -> Option<&(dyn std::any::Any + Send)> {
        self.original_failure.as_deref()
    }

    pub fn has_original_context(&self) -> bool {
        self.context.is_some()
    }

    pub fn has_recovered_factory(&self) -> bool {
        self.recovered_factory.is_some()
    }

    /// None means the factory has not been recovered into this pending value;
    /// it must not be interpreted as a completed reservation.
    pub fn physical_reservation_pending(&self) -> Option<bool> {
        self.recovered_factory
            .as_ref()
            .map(|factory| factory.physical_reservation.is_some())
    }

    pub fn before_native_proven(&self) -> bool {
        self.before_native
    }

    pub fn into_parts(
        self,
    ) -> (
        Option<Box<dyn std::any::Any + Send>>,
        Option<Box<dyn std::any::Any + Send>>,
        Vec<V3FactoryOriginalEvent>,
        Option<super::super::RoventSystemFactoryCustodyV3>,
    ) {
        (
            self.original_failure,
            self.context,
            self.events,
            self.recovered_factory,
        )
    }
}

/// Result of recovering this exact pre-native factory scope. Pending never
/// implies GPU entry or GPU completion. Recovered carries every original pin.
#[cfg(target_os = "linux")]
pub enum V3BeforeNativeScopeRecovery {
    Missing,
    Busy,
    Poisoned,
    Expired,
    WrongState,
    WrongType,
    Pending(crate::pals_native::ticketed::TicketRecoveryError),
    PhysicalPending(rz_eval::worker::PhysicalReservationCancelError),
    Panic(Box<dyn std::any::Any + Send>),
    LostRegistration,
    InvariantViolation {
        original: Box<dyn std::any::Any + Send>,
        pins: Vec<Box<dyn std::any::Any + Send>>,
    },
    Recovered {
        pending: Box<V3FactoryPending>,
        pins: Vec<Box<dyn std::any::Any + Send>>,
    },
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum V3BeforeNativeAttemptError {
    Ticket(crate::pals_native::ticketed::TicketRecoveryError),
    Physical(rz_eval::worker::PhysicalReservationCancelError),
}

#[cfg(target_os = "linux")]
impl From<crate::pals_native::ticketed::TicketRecoveryError> for V3BeforeNativeAttemptError {
    fn from(error: crate::pals_native::ticketed::TicketRecoveryError) -> Self {
        Self::Ticket(error)
    }
}

#[cfg(target_os = "linux")]
impl v3_scope::RoventV3ScopeKey {
    pub fn recover_before_native(&self, until: Instant) -> V3BeforeNativeScopeRecovery {
        use crate::pals_native::ticketed::{TicketRecoveryError, TicketRecoveryStatus};
        use v3_scope::V3ScopeRecovery;
        let result = self
            .try_recover_original_until::<V3FactoryPending, (), V3BeforeNativeAttemptError>(
                until,
                |pending| {
                    let attempt = (|| {
                        if let Some(factory) = pending.recovered_factory.as_mut() {
                            return try_cancel_before_native_physical(factory)
                                .map_err(V3BeforeNativeAttemptError::Physical);
                        }
                        let key = pending.key.clone().ok_or(TicketRecoveryError::InvalidKey)?;
                        if !pending.before_native {
                            // A timeout can precede the worker's final pre-native
                            // result. Only its original key can later establish it.
                            if key.status()? != TicketRecoveryStatus::BeforeNative {
                                return Err(TicketRecoveryError::WrongState.into());
                            }
                            pending.before_native = true;
                        }
                        let original = pending
                            .original_failure
                            .as_mut()
                            .ok_or(TicketRecoveryError::WrongOwnerType)?;
                        if let Some(failure) = original
                            .downcast_mut::<RoventNativeStartupRejection<AdaptivePortQuestion>>()
                        {
                            finish_original_before_native(failure, until, &mut pending.events)?;
                        } else if let Some(failure) =
                            original.downcast_mut::<RoventNativeStartupRejection<PcPortQuestion>>()
                        {
                            finish_original_before_native(failure, until, &mut pending.events)?;
                        } else {
                            return Err(TicketRecoveryError::WrongOwnerType.into());
                        }
                        collect_before_native_events(&key, until, &mut pending.events)?;
                        let factory = key
                            .take_before_native::<super::super::RoventSystemFactoryCustodyV3>(
                                until,
                            )?;
                        // No fallible work after receiving the original factory: put it in
                        // the still-registered slot before issuing the recovery proof.
                        pending.recovered_factory = Some(factory);
                        try_cancel_before_native_physical(
                            pending
                                .recovered_factory
                                .as_mut()
                                .expect("original factory just stored"),
                        )
                        .map_err(V3BeforeNativeAttemptError::Physical)
                    })();
                    if let Err(reason) = attempt {
                        pending.first_recovery_error.get_or_insert(reason);
                    }
                    attempt
                },
            );
        match result {
            V3ScopeRecovery::Missing => V3BeforeNativeScopeRecovery::Missing,
            V3ScopeRecovery::Busy => V3BeforeNativeScopeRecovery::Busy,
            V3ScopeRecovery::Poisoned => V3BeforeNativeScopeRecovery::Poisoned,
            V3ScopeRecovery::Expired => V3BeforeNativeScopeRecovery::Expired,
            V3ScopeRecovery::WrongState => V3BeforeNativeScopeRecovery::WrongState,
            V3ScopeRecovery::WrongType => V3BeforeNativeScopeRecovery::WrongType,
            V3ScopeRecovery::Pending(V3BeforeNativeAttemptError::Ticket(error)) => {
                V3BeforeNativeScopeRecovery::Pending(error)
            }
            V3ScopeRecovery::Pending(V3BeforeNativeAttemptError::Physical(error)) => {
                V3BeforeNativeScopeRecovery::PhysicalPending(error)
            }
            V3ScopeRecovery::Panic(payload) => V3BeforeNativeScopeRecovery::Panic(payload),
            V3ScopeRecovery::LostRegistration(()) => V3BeforeNativeScopeRecovery::LostRegistration,
            V3ScopeRecovery::Recovered { original, pins, .. } => {
                match original.downcast::<V3FactoryPending>() {
                    Ok(pending) => V3BeforeNativeScopeRecovery::Recovered { pending, pins },
                    // Type was verified under the same slot lock. Still return
                    // an opaque owner if an invariant ever becomes invalid.
                    Err(original) => {
                        V3BeforeNativeScopeRecovery::InvariantViolation { original, pins }
                    }
                }
            }
        }
    }
}

/// Exact failed factory or assembly scope and its events after that same
/// native service confirmed its own physical completion. An external scope
/// also requires its own explicit no-process proof.
#[cfg(target_os = "linux")]
pub struct V3SettledFactoryOwner {
    original: Box<dyn std::any::Any + Send>,
    pub events: Vec<V3FactoryOriginalEvent>,
}

#[cfg(target_os = "linux")]
impl V3SettledFactoryOwner {
    pub fn downcast_original<T: std::any::Any + Send>(
        self,
    ) -> Result<(Box<T>, Vec<V3FactoryOriginalEvent>), Self> {
        match self.original.downcast::<T>() {
            Ok(original) => Ok((original, self.events)),
            Err(original) => Err(Self {
                original,
                events: self.events,
            }),
        }
    }
}

#[cfg(target_os = "linux")]
struct V3FactoryDrain {
    physical_confirmed: bool,
    joined: bool,
    join_failed: bool,
    before_native: bool,
    events: Vec<V3FactoryOriginalEvent>,
}

#[cfg(target_os = "linux")]
fn collect_before_native_events(
    key: &crate::pals_native::ticketed::TicketNativeRecoveryKey,
    until: Instant,
    events: &mut Vec<V3FactoryOriginalEvent>,
) -> Result<(), crate::pals_native::ticketed::TicketRecoveryError> {
    use crate::pals_native::ticketed::{TicketRecoveryError, VirtualReviewTicketOutput};
    loop {
        let Some(cell) = key.take_event(until)? else {
            return Ok(());
        };
        match cell.take::<TicketNativeEvent<VirtualReviewTicketOutput>>() {
            Ok(Some(event)) => events.push(V3FactoryOriginalEvent::Native(event)),
            Ok(None) => continue,
            Err(TicketRecoveryError::WrongOwnerType) => {
                match cell.take::<Box<dyn std::any::Any + Send>>() {
                    Ok(Some(payload)) => events.push(V3FactoryOriginalEvent::Panic(payload)),
                    Ok(None) => continue,
                    Err(error) => return Err(error),
                }
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(target_os = "linux")]
fn drain_factory_v3<Q: NativeQuestionProjection>(
    native: &mut RoventNativeService<Q>,
    until: Instant,
) -> V3FactoryDrain {
    native.begin_shutdown(until);
    let mut finished = None;
    let mut original_report = None;
    let recoverable = native.requires_finish_report();
    let mut events = Vec::new();
    loop {
        let signal = native.signal();
        let observed = signal.version();
        while let Some(event) = native.poll() {
            if let TicketNativeEvent::Finished {
                physical_shutdown_confirmed,
                ..
            } = &event
            {
                finished = Some(finished.unwrap_or(true) && *physical_shutdown_confirmed);
            }
            events.push(V3FactoryOriginalEvent::Native(event));
        }
        if let Some(report) = native.take_finish_report() {
            original_report = Some(report.clone());
            events.push(V3FactoryOriginalEvent::FinishReport(report));
        }
        let joined = native.try_join();
        if matches!(joined, Ok(true)) {
            while let Some(event) = native.poll() {
                if let TicketNativeEvent::Finished {
                    physical_shutdown_confirmed,
                    ..
                } = &event
                {
                    finished = Some(finished.unwrap_or(true) && *physical_shutdown_confirmed);
                }
                events.push(V3FactoryOriginalEvent::Native(event));
            }
            if let Some(report) = native.take_finish_report() {
                original_report = Some(report.clone());
                events.push(V3FactoryOriginalEvent::FinishReport(report));
            }
            let snapshot = native.finish_report_snapshot();
            let report_matches = original_report
                .as_ref()
                .is_none_or(|original| snapshot.as_ref() == Some(original));
            let report_physical = snapshot
                .as_ref()
                .map(|report| report.physical_shutdown_confirmed);
            if let Some(copy) = snapshot.filter(|_| original_report.is_none() || !report_matches) {
                events.push(V3FactoryOriginalEvent::FinishReportSnapshot(copy));
            }
            return V3FactoryDrain {
                physical_confirmed: report_matches
                    && consistent_finished_observation(finished, report_physical, recoverable),
                joined: true,
                join_failed: false,
                before_native: native.startup_status()
                    == crate::pals_native::ticketed::NativeFactoryStartupStatus::BeforeNative
                    && !native.is_quarantined(),
                events,
            };
        }
        if joined.is_err() || finished == Some(false) || Instant::now() >= until {
            return V3FactoryDrain {
                physical_confirmed: false,
                joined: false,
                join_failed: joined.is_err(),
                before_native: native.startup_status()
                    == crate::pals_native::ticketed::NativeFactoryStartupStatus::BeforeNative
                    && !native.is_quarantined(),
                events,
            };
        }
        signal.wait_changed(
            observed,
            until
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(10)),
        );
    }
}

#[cfg(target_os = "linux")]
fn consistent_finished_observation(
    event: Option<bool>,
    report: Option<bool>,
    recoverable: bool,
) -> bool {
    event == Some(true)
        && match report {
            Some(value) => value,
            None => !recoverable,
        }
}

#[cfg(target_os = "linux")]
fn finish_original_before_native<Q: NativeQuestionProjection>(
    failure: &mut RoventNativeStartupRejection<Q>,
    until: Instant,
    events: &mut Vec<V3FactoryOriginalEvent>,
) -> Result<(), crate::pals_native::ticketed::TicketRecoveryError> {
    use crate::pals_native::ticketed::TicketRecoveryError;
    if let Some(native) = failure.native.as_mut() {
        // Never infer pre-native from an earlier timeout alone. Recheck this
        // exact owner, then join and hand its receiver back to its own key.
        if native.startup_status()
            != crate::pals_native::ticketed::NativeFactoryStartupStatus::BeforeNative
            || native.is_quarantined()
        {
            return Err(TicketRecoveryError::WrongState);
        }
        let drained = drain_factory_v3(native, until);
        events.extend(drained.events);
        if drained.join_failed {
            return Err(TicketRecoveryError::Disconnected);
        }
        if !drained.joined {
            return Err(TicketRecoveryError::TimedOut);
        }
        if !drained.before_native {
            return Err(TicketRecoveryError::WrongState);
        }
        drop(failure.native.take());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn reject_factory_v3<Q: NativeQuestionProjection>(
    failure: super::super::RoventNativeStartupRejectionV3<Q>,
    until: Instant,
    diagnostic: &NativeStartupDiagnostic,
) -> V3FactoryRejection<Q> {
    match failure {
        super::super::RoventNativeStartupRejectionV3::Started(mut failure) => {
            let detail = error("NativeStartup", &failure.error);
            let recovery_key = failure
                .native
                .as_ref()
                .and_then(|native| native.recovery_key());
            let drain = failure.native.as_mut().map(|native| {
                diagnostic.emit(
                    NativeStartupDiagnosticStage::FactoryDrain,
                    NativeStartupDiagnosticPhase::Entry,
                );
                let drain = drain_factory_v3(native, until);
                diagnostic.emit(
                    NativeStartupDiagnosticStage::FactoryDrain,
                    if drain.physical_confirmed || drain.before_native {
                        NativeStartupDiagnosticPhase::Exit
                    } else {
                        NativeStartupDiagnosticPhase::Error
                    },
                );
                drain
            });
            let Some(mut drain) = drain else {
                return V3FactoryRejection::Ordinary(detail);
            };
            let mut first_recovery_error = None;
            if drain.joined && drain.before_native {
                // Drop only the already-joined service to hand the original
                // receiver back to the same pre-reserved recovery key.
                drop(failure.native.take());
                if let Some(key) = recovery_key.as_ref() {
                    let recovered_factory = collect_before_native_events(
                        key,
                        until,
                        &mut drain.events,
                    )
                    .and_then(|()| {
                        key.take_before_native::<super::super::RoventSystemFactoryCustodyV3>(until)
                    });
                    match recovered_factory {
                        Ok(factory) => {
                            let mut recovered = V3BeforeNativeRecovered {
                                factory,
                                events: drain.events,
                                original_failure: Box::new(failure),
                            };
                            if let Err(reason) =
                                try_cancel_before_native_physical(&mut recovered.factory)
                            {
                                return V3FactoryRejection::BeforeNativePhysicalPending {
                                    error: error("BeforeNativePhysicalReservationPending", detail),
                                    recovered,
                                    recovery_key,
                                    first_recovery_error: V3BeforeNativeAttemptError::Physical(
                                        reason,
                                    ),
                                };
                            }
                            return V3FactoryRejection::Unstarted {
                                error: detail,
                                owner: super::super::RoventUnstartedNativeCustodyV3 {
                                    original: Box::new(recovered),
                                },
                            };
                        }
                        Err(reason) => {
                            first_recovery_error = Some(V3BeforeNativeAttemptError::Ticket(reason))
                        }
                    }
                }
            }
            if drain.physical_confirmed && drain.joined {
                return V3FactoryRejection::Settled {
                    error: detail,
                    owner: V3SettledFactoryOwner {
                        original: Box::new(failure),
                        events: drain.events,
                    },
                };
            }
            let stage = if drain.before_native {
                "BeforeNativeCustodyPending"
            } else if drain.physical_confirmed {
                "NativeEventsPending"
            } else {
                "PhysicalCompletionUnknown"
            };
            V3FactoryRejection::Unknown {
                error: error(stage, detail),
                original: failure,
                recovery_key,
                events: drain.events,
                before_native: drain.before_native,
                first_recovery_error,
            }
        }
        super::super::RoventNativeStartupRejectionV3::Unstarted {
            error: cause,
            owner,
        } => V3FactoryRejection::Unstarted {
            error: error("NativeStartup", cause),
            owner,
        },
    }
}

#[cfg(all(test, target_os = "linux"))]
mod retaining_failure_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct OriginalOwner(Arc<AtomicUsize>);

    #[test]
    fn typed_before_native_pending_keeps_original_event_and_pin_on_repeated_invalid_key() {
        use crate::pals_native::ticketed::TicketRecoveryError;
        use rz_search::pals::coordinator::RoleFailure;

        let pin = Arc::new(AtomicUsize::new(37));
        let original_failure = Arc::new(41_u32);
        let cause = String::from("original pre-native mock failure");
        let original_cause_ptr = cause.as_ptr() as usize;
        let pending = V3FactoryPending {
            original_failure: Some(Box::new(Arc::clone(&original_failure))),
            context: None,
            events: vec![V3FactoryOriginalEvent::Native(
                TicketNativeEvent::StartupFailed(RoleFailure::Backend(cause)),
            )],
            key: None,
            before_native: true,
            recovered_factory: None,
            first_recovery_error: None,
        };
        let mut scope = v3_scope::ScopeReservation::reserve(
            v3_scope::V3ScopeDomain::Cpu,
            Box::new(Arc::clone(&pin)),
        )
        .ok()
        .unwrap();
        scope.arm();
        let key = scope.park(Box::new(pending), None);
        let until = Instant::now() + Duration::from_secs(5);

        for _ in 0..2 {
            assert!(matches!(
                key.recover_before_native(until),
                V3BeforeNativeScopeRecovery::Pending(TicketRecoveryError::InvalidKey)
            ));
            let observed = key.with_original::<V3FactoryPending, _>(|pending| {
                assert_eq!(
                    pending.first_recovery_error(),
                    Some(V3BeforeNativeAttemptError::Ticket(
                        TicketRecoveryError::InvalidKey
                    ))
                );
                let failure = pending
                    .original_failure()
                    .unwrap()
                    .downcast_ref::<Arc<u32>>()
                    .unwrap();
                assert!(Arc::ptr_eq(failure, &original_failure));
                match pending.events() {
                    [
                        V3FactoryOriginalEvent::Native(TicketNativeEvent::StartupFailed(
                            RoleFailure::Backend(cause),
                        )),
                    ] => assert_eq!(cause.as_ptr() as usize, original_cause_ptr),
                    _ => panic!("original startup failure was not retained"),
                }
            });
            assert!(observed.is_some());
            let retained_pin = key.cloned_pin::<AtomicUsize>().unwrap();
            assert!(Arc::ptr_eq(&retained_pin, &pin));
        }
        // Retire only this synthetic, never-started test fixture. This helper
        // does not claim production factory recovery or physical completion.
        assert!(matches!(
            key.try_recover_original_until::<V3FactoryPending, (), ()>(until, |_| Ok(())),
            v3_scope::V3ScopeRecovery::Recovered { .. }
        ));
    }

    #[test]
    fn recoverable_finish_requires_matching_original_report() {
        assert!(consistent_finished_observation(
            Some(true),
            Some(true),
            true
        ));
        assert!(!consistent_finished_observation(
            Some(true),
            Some(false),
            true
        ));
        assert!(!consistent_finished_observation(
            Some(false),
            Some(true),
            true
        ));
        assert!(!consistent_finished_observation(Some(true), None, true));
        // V2-compatible services do not have this extra report channel.
        assert!(consistent_finished_observation(Some(true), None, false));
    }
    impl Drop for OriginalOwner {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn unstarted_factory_failure_returns_original_owner_to_launch_caller() {
        use rz_eval::error::{BackendError, FailureKind, FailureStage};
        let drops = Arc::new(AtomicUsize::new(0));
        let original = OriginalOwner(Arc::clone(&drops));
        let failure =
            super::super::super::RoventNativeStartupRejectionV3::<PcPortQuestion>::Unstarted {
                error: BackendError::new(
                    FailureKind::InvalidInput,
                    FailureStage::Admission,
                    "test rejection",
                ),
                owner: super::super::super::RoventUnstartedNativeCustodyV3 {
                    original: Box::new(original),
                },
            };
        let returned =
            reject_factory_v3(failure, Instant::now(), &NativeStartupDiagnostic::default());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        let owner = match returned {
            V3FactoryRejection::Unstarted { error, owner } => {
                assert_eq!(error.stage, "NativeStartup");
                owner
            }
            _ => panic!("unstarted original owner was not returned"),
        };
        let original = owner
            .downcast::<OriginalOwner>()
            .unwrap_or_else(|_| panic!("wrong original owner"));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(original);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn actual_uci_entry_panic_retains_same_driver_and_payload() {
        use crate::search_driver::{
            SearchKind, SearchSessionContext, SearchSessionFailure, SearchSessionReport,
        };
        use rz_contracts::Digest;
        use rz_position::{BoardMove, Position};
        struct PanicDriver(Arc<AtomicUsize>);
        impl Drop for PanicDriver {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        struct PanicPayload(Arc<AtomicUsize>);
        impl Drop for PanicPayload {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        impl SearchSessionDriver for PanicDriver {
            fn kind(&self) -> SearchKind {
                SearchKind::Pals
            }
            fn implementation(&self) -> Digest {
                Digest([0; 32])
            }
            fn identity(&self) -> EngineIdentity {
                std::panic::panic_any(PanicPayload(Arc::clone(&self.0)))
            }
            fn run(
                &self,
                _: &Position,
                _: &SearchSessionContext,
                _: &mut dyn FnMut(BoardMove),
            ) -> Result<SearchSessionReport, SearchSessionFailure> {
                unreachable!()
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let driver: Arc<dyn SearchSessionDriver> = Arc::new(PanicDriver(Arc::clone(&drops)));
        let before = Arc::clone(&driver);
        let (result, payload) = serve_driver_retaining(
            &driver,
            ProcessClock::new(rz_contracts::ProcessEpoch(1)),
            Arc::new(OwnerRegistry::default()),
            EngineSettings::default(),
            None,
        );
        assert!(result.is_err());
        assert!(payload.as_ref().is_some_and(|p| p.is::<PanicPayload>()));
        assert!(Arc::ptr_eq(&before, &driver));
        let mut scope =
            v3_scope::ScopeReservation::reserve(v3_scope::V3ScopeDomain::Cpu, Box::new(()))
                .ok()
                .unwrap();
        scope.arm();
        let key = scope.park(Box::new((Arc::clone(&driver), payload)), None);
        drop(before);
        drop(driver);
        assert!(key.observe().is_some());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }
}

/// Explicit, asset-backed integration only. The caller supplies an isolated
/// registered V3 input root, template lock and normal launch arguments. This
/// runs serially in the real test process; a captured identity mismatch stops
/// the actual factory before ORT/driver entry independently of any guard.
#[cfg(all(test, target_os = "linux"))]
mod before_native_asset_integration {
    use super::*;
    use sha2::{Digest as _, Sha256};
    use std::io::Read;

    struct StageGuard {
        old_cwd: PathBuf,
        staged: PathBuf,
        completed: bool,
    }
    impl StageGuard {
        fn finish(&mut self) -> std::io::Result<()> {
            std::env::set_current_dir(&self.old_cwd)?;
            std::fs::remove_file(self.staged.join("input-lock.json"))?;
            std::fs::remove_file(self.staged.join("startup.json"))?;
            std::fs::remove_file(self.staged.join("test-executable"))?;
            std::fs::remove_dir(&self.staged)?;
            self.completed = true;
            Ok(())
        }
    }
    impl Drop for StageGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.old_cwd);
            if !self.completed {
                eprintln!(
                    "V3 pre-native integration evidence retained: {}",
                    self.staged.display()
                );
            }
        }
    }

    #[test]
    #[ignore = "disabled: requires a live original-owner supervisor before native execution"]
    fn registered_v3_factory_pre_native_reclaims_seventeen_in_one_process() {
        // An identity mismatch is expected before ORT entry, but a timeout or
        // panic can still produce a quarantined owner. This plain test process
        // has no supervisor that can retain and reobserve that same owner.
        // Do not bypass this gate via an environment variable or test flag.
        fn original_owner_supervisor_connected() -> bool {
            false
        }
        assert!(
            original_owner_supervisor_connected(),
            "fixture disabled: same-owner supervision and quarantine recovery are not connected"
        );
        let root = std::fs::canonicalize(
            std::env::var_os("ROVENT_PRENATIVE_ROOT").expect("registered root required"),
        )
        .unwrap();
        let template =
            std::fs::canonicalize(root.join(
                std::env::var_os("ROVENT_PRENATIVE_TEMPLATE").expect("template lock required"),
            ))
            .unwrap();
        assert!(template.starts_with(&root));
        let mut args: Vec<String> = serde_json::from_str(
            &std::env::var("ROVENT_PRENATIVE_ARGS_JSON")
                .expect("registered launch arguments required"),
        )
        .unwrap();
        let mut lock =
            RoventInputLockV3::from_json(&std::fs::read_to_string(&template).unwrap()).unwrap();
        assert!(matches!(
            &lock.manifest.system_runtime_lock.runtime,
            NativeRuntimeSourceV6::SystemCuda(_)
        ));
        let parsed = Arguments::parse(&args).unwrap();
        assert!(
            parsed.direct_native_assets.is_some(),
            "direct registered assets required"
        );
        assert!(
            !parsed.strict_source.unwrap_or(false),
            "test image has no falsely asserted commit"
        );
        assert!(
            parsed.system_runtime_receipt_root.is_none(),
            "each attempt must avoid reusing a receipt directory"
        );

        let test_exe = std::env::current_exe().unwrap();
        let mut input = File::open(&test_exe).unwrap();
        let mut hash = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let n = input.read(&mut buffer).unwrap();
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        let digest = hash
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let relative_dir = format!("rovent-pre-native-integration-{}", std::process::id());
        let staged = root.join(&relative_dir);
        std::fs::create_dir(&staged).expect("exclusive owned staging directory");
        let mut stage = StageGuard {
            old_cwd: std::env::current_dir().unwrap(),
            staged: staged.clone(),
            completed: false,
        };
        std::fs::hard_link(&test_exe, staged.join("test-executable"))
            .expect("test image must share the staging filesystem");
        // Capture opens/hashes ELF files only; it does not load ORT or CUDA.
        // The system loader holds this same owner process-wide. Derive a
        // different valid identity from the actually captured one, rather
        // than hard-coding an environment-specific digest.
        let mut runtime = lock.manifest.system_runtime_lock.runtime.clone();
        let NativeRuntimeSourceV6::SystemCuda(spec) = &mut runtime else {
            unreachable!()
        };
        let requests = spec
            .libraries
            .iter()
            .map(|item| rz_eval::onnx::SystemLibraryRequest {
                role: item.role.clone(),
                path: PathBuf::from(&item.requested_path),
                required_mapping: item.required_mapping,
            })
            .collect();
        let limits = rz_eval::onnx::SystemCaptureLimits {
            max_libraries: usize::try_from(spec.capture_max_libraries).unwrap(),
            max_file_bytes: spec.capture_max_file_bytes,
            max_total_bytes: spec.capture_max_total_bytes,
            max_elapsed: Duration::from_millis(spec.capture_timeout_ms),
        };
        let actual = rz_eval::onnx::SystemRuntime::capture(requests, limits)
            .expect("registered system files must be capturable without native entry");
        let actual_identity = actual
            .identity_sha256()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let mut mismatch = actual_identity.clone();
        mismatch.replace_range(
            ..1,
            if actual_identity.starts_with('0') {
                "1"
            } else {
                "0"
            },
        );
        assert_ne!(mismatch, actual_identity);
        spec.system_identity_sha256 = mismatch;
        lock.manifest.system_runtime_lock =
            rz_experiments::SystemRuntimeLockV6::new(runtime).unwrap();
        let startup_path =
            std::fs::canonicalize(root.join(&lock.manifest.startup_document.path)).unwrap();
        assert!(startup_path.starts_with(&root));
        let mut startup = rz_experiments::RoventStartupDocumentV5::from_json(
            &std::fs::read_to_string(startup_path).unwrap(),
        )
        .unwrap();
        startup.system_runtime_lock_sha256 =
            lock.manifest.system_runtime_lock.canonical_sha256.clone();
        let startup_bytes = serde_json::to_vec(&startup).unwrap();
        std::fs::write(staged.join("startup.json"), &startup_bytes).unwrap();
        lock.manifest.startup_document.path = format!("{relative_dir}/startup.json");
        lock.manifest.startup_document.bytes = startup_bytes.len() as u64;
        lock.manifest.startup_document.sha256 = Sha256::digest(&startup_bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        // Retain the registered public provenance. The new staging path,
        // length and digest identify this locally derived test document.
        lock.manifest.engine.binary.path = format!("{relative_dir}/test-executable");
        lock.manifest.engine.binary.sha256 = digest;
        lock.manifest.engine.binary.bytes = input.metadata().unwrap().len();
        // Preserve the template's registered source reference. The current
        // local test image has its own path, digest and source components;
        // no exact source commit is asserted for it.
        lock.manifest.engine.source_commit = None;
        lock.manifest.engine.source_components =
            crate::rovent_bootstrap::process_boundary::compiled_process_source_components_v3()
                .unwrap();
        println!(
            "test provenance: startup document locally derived; engine is the current test executable; template source references retained; exact engine commit unasserted"
        );
        let lock = lock.manifest.lock().unwrap();
        let lock_path = format!("{relative_dir}/input-lock.json");
        std::fs::write(
            staged.join("input-lock.json"),
            serde_json::to_vec(&lock).unwrap(),
        )
        .unwrap();
        let arg = args
            .iter_mut()
            .find(|arg| arg.starts_with("--input-lock="))
            .expect("actual launch arguments require input lock");
        *arg = format!("--input-lock={lock_path}");

        std::env::set_current_dir(&root).unwrap();
        let mut recovered = Vec::new();
        let overall_deadline = Instant::now() + Duration::from_secs(30 * 60);
        for attempt in 1..=17 {
            assert!(
                Instant::now() < overall_deadline,
                "pre-native integration budget exhausted"
            );
            let failure = super::super::run_retaining(args.clone())
                .err()
                .expect("native-preentry identity mismatch must reject");
            let owner = match failure {
                super::super::RoventRetainingLaunchError::Unstarted { owner, error } => {
                    eprintln!("attempt {attempt}: Unstarted: {error}");
                    owner
                }
                other => {
                    let (variant, error) = match &other {
                        super::super::RoventRetainingLaunchError::Ordinary(error) => {
                            ("Ordinary", error)
                        }
                        super::super::RoventRetainingLaunchError::Settled { error, .. } => {
                            ("Settled", error)
                        }
                        super::super::RoventRetainingLaunchError::Quarantined { error, .. } => {
                            ("Quarantined", error)
                        }
                        super::super::RoventRetainingLaunchError::Unstarted { error, .. } => {
                            ("Unstarted", error)
                        }
                    };
                    panic!(
                        "attempt {attempt}: actual V3 factory returned {variant}: {error}; pre-native owner required"
                    );
                }
            };
            let mut owner = owner
                .downcast::<V3BeforeNativeRecovered>()
                .unwrap_or_else(|_| panic!("actual factory owner type differs"));
            assert!(
                owner.factory.system_owner.is_none(),
                "native loader callback must not have been entered"
            );
            assert!(owner.factory.ticket.is_none());
            let original_error = owner
                .original_failure
                .downcast_ref::<RoventNativeStartupRejection<AdaptivePortQuestion>>()
                .map(|failure| &failure.error)
                .or_else(|| {
                    owner
                        .original_failure
                        .downcast_ref::<RoventNativeStartupRejection<PcPortQuestion>>()
                        .map(|failure| &failure.error)
                })
                .expect("original V3 factory failure required");
            assert_eq!(
                original_error.kind,
                rz_eval::error::FailureKind::IdentityMismatch
            );
            assert_eq!(
                original_error.stage,
                rz_eval::error::FailureStage::Admission
            );
            assert!(owner.events.iter().any(|event| matches!(
                event,
                V3FactoryOriginalEvent::Native(TicketNativeEvent::StartupFailed(_))
            )));
            assert!(owner.try_cancel_unpublished_physical());
            assert!(!owner.physical_reservation_pending());
            println!(
                "attempt {attempt}: original pre-native event and owner recovered; unpublished physical reservation retired"
            );
            recovered.push(owner);
        }
        drop(recovered);
        stage
            .finish()
            .expect("owned integration inputs must be cleaned");
    }
}

/// Select only original lock entries. No independent mutable index is needed;
/// `open_v3` still opens and verifies every original asset and validates the
/// export/Rules/graph/weights relationship before starting the native owner.
#[cfg(target_os = "linux")]
fn direct_selection(
    lock: &RoventInputLockV3,
    selection: &DirectNativeAssets,
) -> Result<RoventNativeAssetSelection, RoventLaunchError> {
    if selection.cuda_control_inventory.is_some()
        && !matches!(
            &lock.manifest.system_runtime_lock.runtime,
            NativeRuntimeSourceV6::SystemCuda(_)
        )
    {
        return Err(error(
            "Assets",
            "CUDA control inventory requires system CUDA",
        ));
    }
    let original = |path: &str| -> Result<ArtifactRef, RoventLaunchError> {
        let mut matches = lock
            .manifest
            .provenance
            .iter()
            .filter(|item| item.path == path);
        let artifact = matches
            .next()
            .ok_or_else(|| error("Assets", "direct path is absent from original provenance"))?;
        if matches.next().is_some() {
            return Err(error(
                "Assets",
                "direct path is ambiguous in original provenance",
            ));
        }
        Ok(artifact.clone())
    };
    let graph = |role: &str,
                 path: &str,
                 original_name: &Option<String>|
     -> Result<RoventNativeGraphAsset, RoventLaunchError> {
        let registered_file = match original_name {
            Some(name) => name.clone(),
            None => path
                .rsplit('/')
                .next()
                .ok_or_else(|| error("Assets", "graph basename is missing"))?
                .to_owned(),
        };
        Ok(RoventNativeGraphAsset {
            role: role.to_owned(),
            registered_file,
            artifact: original(path)?,
        })
    };
    let mut graphs = vec![
        graph(
            "public",
            &selection.public_graph,
            &selection.public_graph_name,
        )?,
        graph(
            "shared_pc",
            &selection.shared_pc_graph,
            &selection.shared_pc_graph_name,
        )?,
    ];
    match (
        lock.manifest.purpose == rz_experiments::RoventPurpose::Collection,
        &selection.validator_graph,
    ) {
        (true, Some(path)) => {
            graphs.push(graph("validator", path, &selection.validator_graph_name)?)
        }
        (false, None) => {}
        (true, None) => return Err(error("Assets", "V3 Collection requires a validator graph")),
        (false, Some(_)) => {
            return Err(error(
                "Assets",
                "validator graph is restricted to V3 Collection",
            ));
        }
    }
    Ok(RoventNativeAssetSelection {
        startup_configuration: lock.manifest.startup_document.clone(),
        export_manifest: original(&selection.export)?,
        rules_descriptor: original(&selection.rules)?,
        graphs,
        cuda_control_inventory: selection
            .cuda_control_inventory
            .as_deref()
            .map(original)
            .transpose()?,
        runtime_bundle_manifest: None,
    })
}

/// Pin the declared original executable beneath the already-owned root.
/// The shared process boundary hashes this same descriptor and the mapped
/// executable under the original cancellation and startup deadline.
#[cfg(target_os = "linux")]
pub(super) fn open_original_binary_v3(
    root: &Dir,
    artifact: &ArtifactRef,
) -> Result<File, RoventLaunchError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    artifact
        .validate()
        .map_err(|failure| error("Binary", failure))?;
    let parts: Vec<_> = artifact.path.split('/').collect();
    let (filename, parents) = parts
        .split_last()
        .ok_or_else(|| error("Binary", "empty binary path"))?;
    let mut scoped = root
        .try_clone()
        .map_err(|failure| error("Binary", failure))?;
    for parent in parents {
        scoped = scoped
            .open_dir_nofollow(parent)
            .map_err(|failure| error("Binary", failure))?;
    }
    let before = scoped
        .symlink_metadata(filename)
        .map_err(|failure| error("Binary", failure))?;
    if !before.is_file() {
        return Err(error(
            "Binary",
            "original executable must be a regular file",
        ));
    }
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(nix::libc::O_NONBLOCK | nix::libc::O_NOFOLLOW);
    let file = scoped
        .open_with(filename, &options)
        .map_err(|failure| error("Binary", failure))?;
    let metadata = file
        .metadata()
        .map_err(|failure| error("Binary", failure))?;
    if !metadata.is_file() || metadata.len() != artifact.bytes || artifact.bytes == 0 {
        return Err(error(
            "Binary",
            "original executable length or file type differs from declaration",
        ));
    }
    Ok(file.into_std())
}

#[cfg(target_os = "linux")]
pub(crate) struct SystemReceiptWriter {
    directory: Dir,
    relative_path: PathBuf,
}

#[cfg(target_os = "linux")]
impl SystemReceiptWriter {
    #[cfg(feature = "onnx-cuda")]
    pub(crate) fn original_directory(&self) -> &Dir {
        &self.directory
    }

    pub(crate) fn open(root: &Dir, path: &std::path::Path) -> std::io::Result<Self> {
        // Walk only the original private-root capability. The resource owner
        // creates the parent; this process exclusively creates its own leaf.
        let mut dir = root.try_clone()?;
        for part in path.components() {
            let Component::Normal(name) = part else {
                return Err(std::io::Error::other("invalid receipt path"));
            };
            dir = dir.open_dir_nofollow(name)?;
        }
        let leaf = format!("native-process-{}", std::process::id());
        dir.create_dir(&leaf)?;
        Ok(Self {
            directory: dir.open_dir_nofollow(&leaf)?,
            relative_path: path.join(&leaf),
        })
    }

    pub(crate) fn write(
        &self,
        name: &str,
        receipt: &rz_experiments::SystemRuntimeReceiptV6,
        lock: &rz_experiments::SystemRuntimeLockV6,
    ) -> Result<(), String> {
        let bytes = receipt
            .canonical_bytes(lock, receipt.process_id, receipt.process_generation)
            .map_err(|failure| failure.to_string())?;
        self.write_bytes(name, &bytes)
    }

    pub(crate) fn write_bytes(&self, name: &str, bytes: &[u8]) -> Result<(), String> {
        use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
        use std::io::Write;
        let mut parts = std::path::Path::new(name).components();
        if !matches!(parts.next(), Some(Component::Normal(_))) || parts.next().is_some() {
            return Err("invalid receipt filename".into());
        }
        let mut options = cap_std::fs::OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let mut file = self
            .directory
            .open_with(name, &options)
            .map_err(|e| e.to_string())?;
        file.write_all(bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())
    }
}

#[cfg(target_os = "linux")]
fn write_receipt(
    writer: Option<&SystemReceiptWriter>,
    name: &str,
    lock: &RoventInputLockV3,
    receipt: &rz_experiments::SystemRuntimeReceiptV6,
) {
    if let Some(writer) = writer {
        if let Err(failure) = writer.write(name, receipt, &lock.manifest.system_runtime_lock) {
            // Recording failures do not imply a GPU failure or permit early
            // owner release. A supervisor must mark missing evidence unknown.
            eprintln!("system runtime receipt warning: {failure}");
        }
    }
}

/// A session-only observation from the same product owner. This is not the
/// strict V6 physical receipt and grants no release or evaluation authority.
#[cfg(target_os = "linux")]
fn session_release_observation_json(
    lock: &RoventInputLockV3,
    process_epoch: u64,
    release: Option<crate::pals_adaptive::NativeSessionReleaseObservation>,
) -> serde_json::Value {
    let provider = match &lock.manifest.system_runtime_lock.runtime {
        NativeRuntimeSourceV6::Cpu => "CPUExecutionProvider",
        NativeRuntimeSourceV6::SystemCuda(_) => "CUDAExecutionProvider",
    };
    let session = match release {
        Some(observation) => serde_json::json!({
            "status": "released",
            "completed_synchronous_runs": observation.release().completed_synchronous_runs,
            "zero_run": observation.release().completed_synchronous_runs == 0,
            "observation_source": format!("{:?}", observation.source()),
        }),
        None => serde_json::json!({
            "status": "unknown", "completed_synchronous_runs": null,
            "zero_run": null, "observation_source": null,
        }),
    };
    serde_json::json!({
        "domain": "rz-rovent-session-release-observation/1",
        "process_id": std::process::id(),
        "process_epoch": process_epoch,
        "lock_sha256": lock.canonical_sha256,
        "configured_provider": provider,
        "actual_execution_provider": { "status": "unknown" },
        "actual_device_placement": { "status": "unknown" },
        "session_release": session,
        "gpu_fence_or_nn_acceptance": false,
        "completed_nn_inputs": "unknown",
        "scope": "same process-local product session owner; not device-wide completion",
    })
}

/// Optional mapping and placement observations from this original session's
/// finish report. Neither is physical completion or numerical acceptance.
#[cfg(target_os = "linux")]
fn system_runtime_finish_observation_json(
    lock: &RoventInputLockV3,
    process_epoch: u64,
    release: Option<crate::pals_adaptive::NativeSessionReleaseObservation>,
) -> serde_json::Value {
    use crate::pals_native::ticketed::{
        TicketRuntimeMappingStatus as M, TicketSystemPlacementStatus as P,
    };
    let mapping = match release.map(|value| value.runtime_mapping_status()) {
        Some(M::NonCuda) => "non_cuda",
        Some(M::Observed) => "observed",
        Some(M::Unknown) => "unknown",
        Some(M::NotObserved) | None => "not_observed",
    };
    let placement = match release.map(|value| value.system_placement_status()) {
        Some(P::Incomplete) => "incomplete",
        Some(P::UnregisteredOrPreexistingMappings) => "unregistered_or_preexisting_mappings",
        Some(P::Confirmed) => "confirmed",
        Some(P::NotObserved) | None => "not_observed",
    };
    serde_json::json!({
        "domain": "rz-rovent-system-runtime-finish-observation/1",
        "process_id": std::process::id(), "process_epoch": process_epoch,
        "lock_sha256": lock.canonical_sha256,
        "runtime_mapping_observation": mapping,
        "system_placement_observation": placement,
        "scope": "same original session finish report; not a GPU fence or numerical acceptance",
    })
}

#[cfg(target_os = "linux")]
fn runtime_receipt(
    lock: &RoventInputLockV3,
    generation: u64,
    observation: Option<&super::super::RoventSystemReadyObservationV3>,
    cpu_provider_ready: bool,
) -> rz_experiments::SystemRuntimeReceiptV6 {
    use rz_experiments::{PhysicalCompletionV6, ProviderReadinessV6, SystemRuntimeReceiptV6};
    SystemRuntimeReceiptV6 {
        domain: rz_experiments::SYSTEM_RUNTIME_RECEIPT_V6_DOMAIN.into(),
        lock_sha256: lock.manifest.system_runtime_lock.canonical_sha256.clone(),
        process_id: std::process::id(),
        process_generation: generation,
        observed_identity_sha256: observation
            .map(|value| super::super::hex(value.runtime_identity.0)),
        observed_libraries: observation
            .map(|value| value.libraries.clone())
            .unwrap_or_default(),
        unregistered_native_mappings: observation
            .and_then(|value| value.unregistered_native_mappings.clone()),
        // Provider setup is observed independently of mapping and inference.
        provider_ready: if observation.is_some_and(|value| value.provider_ready)
            || (cpu_provider_ready
                && matches!(
                    &lock.manifest.system_runtime_lock.runtime,
                    NativeRuntimeSourceV6::Cpu
                )) {
            ProviderReadinessV6::Ready
        } else {
            ProviderReadinessV6::Unknown
        },
        physical_completion: PhysicalCompletionV6::Unknown,
    }
}

#[cfg(target_os = "linux")]
pub(super) struct OriginalV3Inputs<'a> {
    lock: Arc<RoventInputLockV3>,
    boundary: Arc<VerifiedOwnedRoventProcessBoundaryV3>,
    sources: Arc<RoventPinnedNativeSources<RoventInputLockV3>>,
    control: RoventStartupControl,
    clock: ProcessClock,
    collection: Option<Arc<super::RoventPcCollectionLaunchOwnerV3>>,
    // Original open descriptors only; no archive worker is started here.
    storage: Option<storage::RoventStorageByteCustody<RoventInputLockV3>>,
    // A legacy selection index is retained through physical completion;
    // direct lock-backed selection has no separate index descriptor.
    selection_file: Option<Arc<File>>,
    lock_file: Arc<File>,
    receipt_writer: Option<Arc<SystemReceiptWriter>>,
    control_scope_pin: Option<Box<dyn std::any::Any + Send>>,
    connected_parent: Option<Arc<super::ConnectedParent>>,
    go_observer: Option<crate::engine::GoAdmissionObserver>,
    scope_reserved:
        Option<Box<dyn FnOnce(&v3_scope::RoventV3ScopeKey) -> Result<(), RoventLaunchError> + 'a>>,
}

#[cfg(target_os = "linux")]
pub(super) fn admit_original<'a>(
    arguments: &Arguments,
    lock: Arc<RoventInputLockV3>,
    root: Dir,
    cwd: PathBuf,
    lock_file: File,
    resource: Option<ResourceAdmissionV3<'a>>,
) -> Result<OriginalV3Inputs<'a>, RoventLaunchError> {
    admit_original_inner(arguments, lock, root, cwd, lock_file, resource, None)
}

/// Admit the explicit /3 P/C Collection using the same original process,
/// resource control and parent stream as ordinary /3 launch.
#[cfg(target_os = "linux")]
pub(super) fn admit_original_collection<'a>(
    arguments: &Arguments,
    lock: Arc<RoventInputLockV3>,
    root: Dir,
    cwd: PathBuf,
    lock_file: File,
    resource: Option<ResourceAdmissionV3<'a>>,
    collection: Arc<super::RoventPcCollectionLaunchOwnerV3>,
) -> Result<OriginalV3Inputs<'a>, RoventLaunchError> {
    admit_original_inner(
        arguments,
        lock,
        root,
        cwd,
        lock_file,
        resource,
        Some(collection),
    )
}

#[cfg(target_os = "linux")]
fn admit_original_inner<'a>(
    arguments: &Arguments,
    lock: Arc<RoventInputLockV3>,
    root: Dir,
    cwd: PathBuf,
    lock_file: File,
    resource: Option<ResourceAdmissionV3<'a>>,
    collection: Option<Arc<super::RoventPcCollectionLaunchOwnerV3>>,
) -> Result<OriginalV3Inputs<'a>, RoventLaunchError> {
    if arguments.storage.is_some()
        || arguments.startup_diagnostics
        || arguments.lifecycle_diagnostics
        || arguments.lifecycle_diagnostics_v2
        || arguments.initial_stop_source_diagnostics
    {
        return Err(error(
            "InputLockUnsupportedV3",
            "selected legacy storage or diagnostic owner is not connected to /3",
        ));
    }
    if !matches!(
        (lock.manifest.purpose, collection.is_some()),
        (
            rz_experiments::RoventPurpose::Validation | rz_experiments::RoventPurpose::Deployment,
            false
        ) | (rz_experiments::RoventPurpose::Collection, true)
    ) {
        return Err(error(
            "InputLockUnsupportedV3",
            "Collection requires its original /3 owner and is not a Validation or Deployment observer",
        ));
    }
    if collection.is_some()
        && (lock.manifest.engine.mode.has_native_r()
            || !matches!(
                &lock.manifest.system_runtime_lock.runtime,
                NativeRuntimeSourceV6::SystemCuda(_)
            ))
    {
        return Err(error(
            "InputLockUnsupportedV3",
            "this /3 Collection entry supports system CUDA P/C only",
        ));
    }
    if arguments.startup_ms > lock.manifest.wall_time_max_ms
        || matches!(
            &lock.manifest.system_runtime_lock.runtime,
            NativeRuntimeSourceV6::SystemCuda(system) if arguments.startup_ms > system.startup_ms
        )
    {
        return Err(error(
            "StartupWindow",
            "startup exceeds the original /3 or system runtime window",
        ));
    }
    let until = Instant::now()
        .checked_add(Duration::from_millis(arguments.startup_ms))
        .ok_or_else(|| error("StartupWindow", "startup deadline overflow"))?;
    let cleanup_until = until
        .checked_add(Duration::from_millis(lock.manifest.cleanup_max_ms))
        .ok_or_else(|| error("StartupWindow", "cleanup deadline overflow"))?;
    let control = RoventStartupControl {
        until,
        cleanup_until,
        cancellation: Arc::new(AtomicBool::new(false)),
    };
    if let Some(owner) = collection.as_ref() {
        owner
            .hooks()
            .register_startup_control(&control)
            .map_err(|failure| error("CollectionAdmission", failure))?;
    }
    let clock = ProcessClock::new(ProcessEpoch(arguments.process_epoch));
    control
        .check()
        .map_err(|failure| error("Binary", failure))?;
    let binary = open_original_binary_v3(&root, &lock.manifest.engine.binary)?;
    control
        .check()
        .map_err(|failure| error("Binary", failure))?;
    let source_policy = if arguments.strict_source.unwrap_or(false) {
        SourceEvidencePolicy::StrictReproducibility
    } else {
        SourceEvidencePolicy::FunctionalWithSourceObservation
    };
    let boundary = verify_actual_owned_process_boundary_v3_with_policy(
        RoventProcessBoundaryInputsV3 {
            lock: Arc::clone(&lock),
            original_binary: binary,
            original_root: root,
            private_cwd: cwd.clone(),
            control: control.clone(),
        },
        source_policy,
    )
    .map_err(|rejected| error("ProcessBoundary", rejected.error))?;
    let root = boundary.original_root();
    let view = boundary
        .startup_view()
        .map_err(|failure| error("ProcessBoundary", failure))?;
    let early_writer = if resource.as_ref().is_some_and(|r| r.require_control_dir) {
        let path = arguments
            .system_runtime_receipt_root
            .as_ref()
            .ok_or_else(|| {
                error(
                    "ControlAdmission",
                    "explicit control requires a verified receipt root",
                )
            })?;
        control
            .check()
            .map_err(|failure| error("ControlAdmission", failure))?;
        Some(
            SystemReceiptWriter::open(root, path)
                .map_err(|failure| error("ControlAdmission", failure))?,
        )
    } else {
        None
    };
    let mut control_scope_pin = None;
    let mut scope_reserved = None;
    let mut connected_parent = None;
    if let Some(admit_resource) = resource {
        let ResourceAdmissionV3 {
            require_control_dir,
            scope_pin,
            is_installed,
            scope_reserved: reserved_callback,
            parent_stream,
            callback,
        } = admit_resource;
        control
            .check()
            .map_err(|failure| error("ResourceAdmission", failure))?;
        let control_route = early_writer
            .as_ref()
            .map(|writer| cwd.join(&writer.relative_path));
        // Capture before callback publishes the control endpoint. A later
        // reparented process cannot become the original supervisor.
        let original_parent_pid = if parent_stream.is_some()
            && matches!(
                &lock.manifest.system_runtime_lock.runtime,
                NativeRuntimeSourceV6::SystemCuda(_)
            ) {
            Some(
                super::ConnectedParent::capture_original_parent()
                    .map_err(|failure| error("ControlAdmission", failure))?,
            )
        } else {
            None
        };
        callback(
            &lock,
            root,
            &lock_file,
            control.until,
            control.cleanup_until,
            early_writer.as_ref().map(|writer| &writer.directory),
            control_route.as_deref(),
        )?;
        if require_control_dir
            && (scope_pin.is_none() || !is_installed.as_ref().is_some_and(|gate| gate()))
        {
            return Err(error(
                "ControlAdmission",
                "original control owner was not installed",
            ));
        }
        if let Some(parent_stream) = parent_stream {
            if let NativeRuntimeSourceV6::SystemCuda(_) = &lock.manifest.system_runtime_lock.runtime
            {
                let ordinal = match &lock.manifest.engine.pc.device {
                    RoventDevice::Cuda { ordinal, .. } => i32::try_from(*ordinal)
                        .map_err(|_| error("ControlAdmission", "CUDA ordinal is out of range"))?,
                    _ => return Err(error("ControlAdmission", "CUDA device declaration missing")),
                };
                let stream = parent_stream
                    .lock()
                    .map_err(|_| error("ControlAdmission", "parent stream owner poisoned"))?
                    .take()
                    .ok_or_else(|| error("ControlAdmission", "original parent stream missing"))?;
                connected_parent = Some(Arc::new(
                    super::ConnectedParent::authenticate(
                        stream,
                        std::process::id(),
                        original_parent_pid
                            .ok_or_else(|| error("ControlAdmission", "original parent missing"))?,
                        arguments.process_epoch,
                        &lock.canonical_sha256,
                        ordinal,
                        control.until,
                    )
                    .map_err(|failure| error("ControlAdmission", failure))?,
                ));
            }
        }
        control_scope_pin = scope_pin;
        scope_reserved = reserved_callback;
        control
            .check()
            .map_err(|failure| error("ResourceAdmission", failure))?;
    }
    let receipt_writer = early_writer.or_else(|| {
        arguments
            .system_runtime_receipt_root
            .as_ref()
            .and_then(|path| match SystemReceiptWriter::open(root, path) {
                Ok(writer) => Some(writer),
                Err(failure) => {
                    eprintln!("system runtime receipt warning: {failure}");
                    None
                }
            })
    });
    let (selection, selection_file) = if let Some(direct) = arguments.direct_native_assets.as_ref()
    {
        (direct_selection(&lock, direct)?, None)
    } else {
        let selected: Vec<_> = lock
            .manifest
            .provenance
            .iter()
            .filter(|artifact| artifact.path == arguments.native_assets)
            .collect();
        if selected.len() != 1 {
            return Err(error(
                "Assets",
                "selection is not a unique original provenance artifact",
            ));
        }
        let selection_file = selected[0]
            .open_verified_at(root, MAX_SELECTION_BYTES)
            .map_err(|failure| error("Assets", failure))?;
        let selection = SelectionWire::parse_with_graph_count(
            &read_bounded(&selection_file, MAX_SELECTION_BYTES)?,
            if lock.manifest.purpose == rz_experiments::RoventPurpose::Collection {
                3
            } else {
                2
            },
        )?;
        (selection, Some(selection_file))
    };
    let sources = if let Some(owner) = collection.as_ref() {
        super::collection_v3::open_collection_sources(
            Arc::clone(&lock),
            root,
            cwd,
            selection,
            &control,
            owner.teachers().to_vec(),
            owner.sealed_teacher_indices(),
            arguments.strict_source.unwrap_or(false),
        )?
    } else {
        let source_root = root.try_clone().map_err(|failure| error("Cwd", failure))?;
        Arc::new(
            if arguments.strict_source.unwrap_or(false) {
                RoventPinnedNativeSources::<RoventInputLockV3>::open_v3(
                    Arc::clone(&lock),
                    source_root,
                    cwd,
                    selection,
                    &control,
                )
            } else {
                RoventPinnedNativeSources::<RoventInputLockV3>::open_v3_observational(
                    Arc::clone(&lock),
                    source_root,
                    cwd,
                    selection,
                    &control,
                )
            }
            .map_err(|failure| error("Assets", failure))?,
        )
    };
    sources
        .configuration()
        .require_supported_native_capabilities()
        .map_err(|failure| error("Capability", failure))?;
    let evidence = boundary.source_evidence();
    if evidence.commit_status != super::super::process_boundary::SourceEvidenceStatus::Verified
        || evidence.component_status
            != super::super::process_boundary::SourceEvidenceStatus::Verified
        || sources.source_evidence() != super::super::pinned::NativeSourceEvidence::Verified
    {
        eprintln!(
            "source provenance unverified: commit={:?} components={:?} native={:?}",
            evidence.commit_status,
            evidence.component_status,
            sources.source_evidence()
        );
    }
    if let Some(writer) = receipt_writer.as_ref() {
        // These fields describe source provenance only. The declared aggregate
        // is never represented as a digest of actual compiled bytes.
        let json = serde_json::json!({
            "domain": "rz-rovent-source-observation-v3/1",
            "process_id": std::process::id(),
            "process_generation": arguments.process_epoch,
            "policy": format!("{:?}", evidence.policy),
            "commit_status": format!("{:?}", evidence.commit_status),
            "component_status": format!("{:?}", evidence.component_status),
            "native_source_status": format!("{:?}", sources.source_evidence()),
            "declared_commit": evidence.declared_commit,
            "compiled_commit": evidence.compiled_commit,
            "declared_components": evidence.declared_components,
            "compiled_components": evidence.compiled_components,
            "declared_aggregate_sha256": evidence.declared_aggregate.map(|value| super::super::hex(value.0)),
        });
        if let Err(failure) =
            writer.write_bytes("source-observation.v3.json", json.to_string().as_bytes())
        {
            eprintln!("source observation warning: {failure}");
        }
    }
    boundary
        .revalidate_startup()
        .map_err(|failure| error("ProcessBoundary", failure))?;
    drop(view);
    // Optional diagnostics are bound to the authenticated original parent;
    // their absence or failure never rejects the product launch.
    let go_observer = arguments.go_observer_fd.and_then(|fd| {
        if !GO_OBSERVER_PARENT_HANDOFF_CONNECTED {
            super::go_observer_warning("supervisor handoff is not connected");
            return None;
        }
        if !arguments.go_observer_fd_exclusive {
            super::go_observer_warning("child-exclusive descriptor ownership not declared");
            return None;
        }
        if !super::go_observer::early_seal_matches(fd) {
            super::go_observer_warning("early inherited descriptor seal unavailable");
            return None;
        }
        let Some(parent) = connected_parent.as_ref() else {
            super::go_observer_warning("authenticated parent unavailable");
            return None;
        };
        let ordinal = match &lock.manifest.engine.pc.device {
            RoventDevice::Cuda { ordinal, .. } => match i32::try_from(*ordinal) {
                Ok(value) => value,
                Err(_) => {
                    super::go_observer_warning("CUDA ordinal unavailable");
                    return None;
                }
            },
            _ => {
                super::go_observer_warning("system CUDA parent required");
                return None;
            }
        };
        match rz_platform::duplicate_inherited_unix_datagram(fd).and_then(|socket| {
            super::go_observer::from_original_parent(
                socket,
                fd,
                Arc::clone(parent),
                arguments.process_epoch,
                &lock.canonical_sha256,
                ordinal,
            )
        }) {
            Ok(observer) => Some(observer),
            Err(_) => {
                super::go_observer_warning("descriptor or peer validation failed");
                None
            }
        }
    });
    let storage = if let Some(selection) = arguments.storage.as_ref() {
        let custody = storage::open_rovent_storage_byte_custody_v3(
            Arc::clone(&lock),
            boundary.original_root(),
            storage::RoventStorageAdmissionRequest {
                profile_path: &selection.profile,
                profile_sha256: &selection.profile_sha256,
                profile_bytes: selection.profile_bytes,
                archive_root: &selection.archive_root,
                checkout_root: &selection.checkout_root,
                private_snapshot_root: boundary.private_cwd(),
                protected_execution_root: &selection.protected_root,
            },
            control.clone(),
        )
        .map_err(|failure| {
            let cause = error("StorageAdmissionV3", &failure.error);
            // No native service or archive worker has started at admission.
            drop(failure.owners);
            cause
        })?;
        let storage_view = boundary
            .startup_view()
            .map_err(|failure| error("StorageProcessBoundaryV3", failure))?;
        let compiled = storage::check_rovent_storage_compiled_custody_v3(custody, &storage_view)
            .map_err(|failure| {
                let cause = error("StorageCompiledCustodyV3", &failure.error);
                drop(failure.custody);
                cause
            })?;
        let observation = compiled.source_roles_observation();
        if let Some(writer) = receipt_writer.as_ref() {
            let record = serde_json::json!({
                "schema": "rz-rovent-storage-source-observation/1",
                "input_lock_sha256": lock.canonical_sha256,
                "source_roles": format!("{observation:?}"),
            });
            if let Err(failure) = writer.write_bytes(
                "storage-source-observation.v3.json",
                record.to_string().as_bytes(),
            ) {
                eprintln!("storage source observation warning: {failure}");
            }
        } else {
            eprintln!(
                "storage source observation unavailable: no original receipt writer; status={observation:?}"
            );
        }
        Some(compiled.into_byte_custody())
    } else {
        None
    };
    Ok(OriginalV3Inputs {
        lock,
        boundary,
        sources,
        control,
        clock,
        collection,
        storage,
        selection_file: selection_file.map(Arc::new),
        lock_file: Arc::new(lock_file),
        receipt_writer: receipt_writer.map(Arc::new),
        control_scope_pin,
        connected_parent,
        go_observer,
        scope_reserved,
    })
}

#[cfg(target_os = "linux")]
pub(super) fn run_native(
    arguments: Arguments,
    lock: Arc<RoventInputLockV3>,
    root: Dir,
    cwd: PathBuf,
    lock_file: File,
) -> Result<(), RoventLaunchError> {
    run_native_retaining(arguments, lock, root, cwd, lock_file)
        .map_err(RoventV3LaunchFailure::into_legacy)
}

#[cfg(target_os = "linux")]
pub(super) fn run_native_retaining(
    arguments: Arguments,
    lock: Arc<RoventInputLockV3>,
    root: Dir,
    cwd: PathBuf,
    lock_file: File,
) -> Result<(), RoventV3LaunchFailure> {
    run_native_inner(arguments, lock, root, cwd, lock_file, None)
}

#[cfg(target_os = "linux")]
pub(super) fn run_native_with_resource<'a>(
    arguments: Arguments,
    lock: Arc<RoventInputLockV3>,
    root: Dir,
    cwd: PathBuf,
    lock_file: File,
    resource: ResourceAdmissionV3<'a>,
) -> Result<(), RoventLaunchError> {
    run_native_with_resource_retaining(arguments, lock, root, cwd, lock_file, resource)
        .map_err(RoventV3LaunchFailure::into_legacy)
}

#[cfg(target_os = "linux")]
pub(super) fn run_native_with_resource_retaining<'a>(
    arguments: Arguments,
    lock: Arc<RoventInputLockV3>,
    root: Dir,
    cwd: PathBuf,
    lock_file: File,
    resource: ResourceAdmissionV3<'a>,
) -> Result<(), RoventV3LaunchFailure> {
    run_native_inner(arguments, lock, root, cwd, lock_file, Some(resource))
}

#[cfg(target_os = "linux")]
fn serve_driver_retaining(
    driver: &Arc<dyn SearchSessionDriver>,
    clock: ProcessClock,
    owners: Arc<OwnerRegistry>,
    settings: EngineSettings,
    go_observer: Option<crate::engine::GoAdmissionObserver>,
) -> (
    Result<(), RoventLaunchError>,
    Option<Box<dyn std::any::Any + Send>>,
) {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::pals_adaptive_bootstrap::protocol::serve_driver_with_go_observer(
            Arc::clone(driver),
            clock,
            owners,
            settings,
            go_observer,
        )
    })) {
        Ok(result) => (result.map_err(|failure| error("UciSession", failure)), None),
        Err(payload) => (
            Err(error("UciPanic", "UCI session panicked")),
            Some(payload),
        ),
    }
}

#[cfg(target_os = "linux")]
fn run_native_inner<'a>(
    arguments: Arguments,
    lock: Arc<RoventInputLockV3>,
    root: Dir,
    cwd: PathBuf,
    lock_file: File,
    resource: Option<ResourceAdmissionV3<'a>>,
) -> Result<(), RoventV3LaunchFailure> {
    let original = admit_original(&arguments, lock, root, cwd, lock_file, resource)?;
    run_native_admitted(arguments, original)
}

/// Continue only after the caller has retained its original admission owner.
/// Admission itself has not entered the native runtime.
#[cfg(target_os = "linux")]
pub(super) fn run_native_admitted(
    arguments: Arguments,
    original: OriginalV3Inputs<'_>,
) -> Result<(), RoventV3LaunchFailure> {
    let OriginalV3Inputs {
        lock,
        boundary,
        sources,
        control,
        clock,
        collection,
        storage,
        selection_file,
        lock_file,
        receipt_writer,
        control_scope_pin,
        connected_parent,
        go_observer,
        scope_reserved,
    } = original;
    let own_board = matches!(
        sources.configuration().fields().execution,
        ExecutionConfiguration::OwnResearcher {
            value: ResearchValueConfiguration::BaselineBoard,
            ..
        }
    );
    if own_board && (storage.is_some() || collection.is_some()) {
        return Err(error(
            "InputLockUnsupportedV3",
            "own-board V3 archive and Collection owners are not connected",
        )
        .into());
    }
    if storage.is_some()
        && matches!(
            sources.topology(),
            super::super::RoventStartupTopology::ExternalResearcher
        )
    {
        return Err(error(
            "InputLockUnsupportedV3",
            "archive-on requires a native PC or own-researcher session",
        )
        .into());
    }
    // Each selected product receives its own original V3 owner. Collection
    // still requires its own V3 owner and remains separate.
    let nnuec_selection = match (
        sources.topology(),
        sources.configuration().fields().execution,
        sources.configuration().fields().nnuec_selection(),
    ) {
        (
            super::super::RoventStartupTopology::RolesOnly,
            ExecutionConfiguration::RolesOnly {
                shared_nnuec_readout: None,
                ..
            },
            NativeNnuecSelection::Absent,
        ) => NativeNnuecSelection::Absent,
        (
            super::super::RoventStartupTopology::RolesOnly,
            ExecutionConfiguration::RolesOnly {
                shared_nnuec_readout: Some(_),
                ..
            },
            NativeNnuecSelection::SharedPc,
        ) => NativeNnuecSelection::SharedPc,
        (
            super::super::RoventStartupTopology::OwnResearcher,
            ExecutionConfiguration::OwnResearcher {
                value: ResearchValueConfiguration::BaselineBoard,
                ..
            },
            NativeNnuecSelection::Absent,
        ) => NativeNnuecSelection::Absent,
        (
            super::super::RoventStartupTopology::OwnResearcher,
            ExecutionConfiguration::OwnResearcher {
                value: ResearchValueConfiguration::NnuecV2 { .. },
                ..
            },
            NativeNnuecSelection::OwnResearcher,
        ) => NativeNnuecSelection::OwnResearcher,
        (
            super::super::RoventStartupTopology::ExternalResearcher,
            ExecutionConfiguration::ExternalResearcher { .. },
            NativeNnuecSelection::Absent,
        ) => NativeNnuecSelection::Absent,
        _ => {
            return Err(error(
                "InputLockUnsupportedV3",
                "selected V3 product owner is not connected",
            )
            .into());
        }
    };
    // Choose the physical owner before starting the native worker, so no
    // missing-helper branch can drop an already-started GPU owner.
    // One startup owner is moved exactly once; do not add a heap allocation
    // solely to shrink this short-lived choice.
    #[allow(clippy::large_enum_variant)]
    enum ProductChoice {
        Pc,
        PcShared(LoadedRoventNnuecSource<RoventInputLockV3>),
        OwnBoard,
        OwnPcr(LoadedRoventNnuecSource<RoventInputLockV3>),
        External(
            RoventPinnedExternalRSource<RoventInputLockV3>,
            LaunchRetirementEvidence,
        ),
    }
    let (external_source, external_selection_file, retirement_evidence) = match (
        sources.configuration().fields().execution,
        arguments.external_r_sources.as_deref(),
        arguments.retirement_sink.as_deref(),
        arguments.evidence_root.as_deref(),
    ) {
        (
            ExecutionConfiguration::ExternalResearcher { .. },
            Some(path),
            Some(registration),
            Some(root),
        ) => {
            sources
                .configuration()
                .external_question_limit()
                .map_err(|failure| error("ExternalQuestionPolicy", failure))?
                .ok_or_else(|| {
                    error("ExternalQuestionPolicy", "explicit external query required")
                })?;
            let mut matching = lock
                .manifest
                .provenance
                .iter()
                .filter(|artifact| artifact.path == path);
            let artifact = matching.next().ok_or_else(|| {
                error(
                    "ExternalSources",
                    "selection is absent from original provenance",
                )
            })?;
            if matching.next().is_some() {
                return Err(error(
                    "ExternalSources",
                    "selection is not a unique original provenance artifact",
                )
                .into());
            }
            let mapping = artifact
                .open_verified_at(boundary.original_root(), MAX_SELECTION_BYTES)
                .map_err(|failure| error("ExternalSources", failure))?;
            let selected =
                ExternalSelectionWire::parse(&read_bounded(&mapping, MAX_SELECTION_BYTES)?)?;
            let source = RoventPinnedExternalRSource::open(&sources, &selected, &control)
                .map_err(|failure| error("ExternalSources", failure))?;
            let evidence = LaunchRetirementEvidence::open_v3(
                Arc::clone(&boundary),
                registration,
                root,
                &control,
            )
            .map_err(|failure| error("RetirementEvidence", failure))?;
            (Some(source), Some(mapping), Some(evidence))
        }
        (ExecutionConfiguration::ExternalResearcher { .. }, _, _, _) => {
            return Err(error(
                "ExternalSources",
                "external researcher requires --external-r-sources, --retirement-sink and --evidence-root",
            ).into());
        }
        (_, None, None, None) => (None, None, None),
        _ => {
            return Err(error(
                "ExternalSources",
                "external source and retirement arguments require the external researcher",
            )
            .into());
        }
    };
    let external_selection_file = external_selection_file.map(Arc::new);
    let product = match (nnuec_selection, external_source, retirement_evidence) {
        (NativeNnuecSelection::Absent, Some(external), Some(evidence)) => {
            ProductChoice::External(external, evidence)
        }
        (NativeNnuecSelection::Absent, None, None) if own_board => ProductChoice::OwnBoard,
        (NativeNnuecSelection::Absent, None, None) => ProductChoice::Pc,
        (NativeNnuecSelection::SharedPc, None, None) => ProductChoice::PcShared(
            LoadedRoventNnuecSource::load(Arc::clone(&sources), &control)
                .map_err(|failure| error("NnuecSource", failure))?,
        ),
        (NativeNnuecSelection::OwnResearcher, None, None) => ProductChoice::OwnPcr(
            LoadedRoventNnuecSource::load(Arc::clone(&sources), &control)
                .map_err(|failure| error("NnuecSource", failure))?,
        ),
        _ => {
            return Err(error(
                "ExternalSources",
                "incompatible product and external owners",
            )
            .into());
        }
    };
    let native = sources.configuration().fields().native;
    let tickets = native
        .tickets(sources.configuration().fields().review.token_budget)
        .map_err(|failure| error("Configuration", failure))?;
    let resources = match &lock.manifest.system_runtime_lock.runtime {
        NativeRuntimeSourceV6::SystemCuda(system) => {
            let cuda_profile = if sources.cuda_inventory_digest().is_some() {
                control
                    .check()
                    .map_err(|failure| error("ProfileDirectory", failure))?;
                let name = format!(
                    "rovent-cuda-profile-{}-{}",
                    std::process::id(),
                    arguments.process_epoch
                );
                let root = boundary.original_root();
                root.create_dir(&name)
                    .map_err(|failure| error("ProfileDirectory", failure))?;
                let directory = root
                    .open_dir_nofollow(&name)
                    .map_err(|failure| error("ProfileDirectory", failure))?;
                Some(
                    VirtualReviewProfileDirectory::from_owned_directory(directory.into_std_file())
                        .map_err(|failure| error("ProfileDirectory", failure))?,
                )
            } else {
                None
            };
            super::super::RoventNativeResourcesV3::SystemCuda(
                super::super::RoventSystemNativeResourcesV3 {
                    configuration: native
                        .onnx_system(&lock.manifest.engine.pc.device, system)
                        .map_err(|failure| error("Configuration", failure))?,
                    cuda_profile,
                    drain_limit: Duration::from_millis(native.drain_ms),
                    tickets,
                    command_capacity: native.command_capacity,
                },
            )
        }
        NativeRuntimeSourceV6::Cpu => {
            super::super::RoventNativeResourcesV3::Cpu(super::super::RoventNativeResources {
                runtime: sources
                    .pin_runtime()
                    .map_err(|failure| error("Runtime", failure))?,
                configuration: native
                    .onnx(&lock.manifest.engine.pc.device, false)
                    .map_err(|failure| error("Configuration", failure))?,
                loading_profile: native.loading_profile.map(|profile| profile.actual()),
                cuda_profile: None,
                drain_limit: Duration::from_millis(native.drain_ms),
                tickets,
                command_capacity: native.command_capacity,
            })
        }
    };
    let diagnostic = NativeStartupDiagnostic::default();
    let owners = Arc::new(OwnerRegistry::default());
    let identity = EngineIdentity {
        name: format!("Rovent {:?}", lock.manifest.engine.mode),
        author: "Rovent contributors".into(),
    };
    // Reserve before the native worker can be started. The registry owns the
    // same open handles and source objects even if the caller drops its token.
    let pins: Box<dyn std::any::Any + Send> = Box::new((
        Arc::clone(&lock),
        Arc::clone(&boundary),
        Arc::clone(&sources),
        selection_file.clone(),
        external_selection_file.clone(),
        Arc::clone(&lock_file),
        receipt_writer.clone(),
        Arc::clone(&owners),
    ));
    let scope_domain = match &lock.manifest.system_runtime_lock.runtime {
        NativeRuntimeSourceV6::Cpu => v3_scope::V3ScopeDomain::Cpu,
        NativeRuntimeSourceV6::SystemCuda(_) => v3_scope::V3ScopeDomain::CudaUnidentified,
    };
    let mut scope = v3_scope::ScopeReservation::reserve(scope_domain, pins).map_err(|_pins| {
        error(
            "NativeScopeCapacity",
            "no custody slot available before native start",
        )
    })?;
    if let Some(mark_reserved) = scope_reserved {
        mark_reserved(scope.key())?;
    }
    scope.attach_pin(Box::new(Arc::new(control.clone())));
    if let Some(writer) = &receipt_writer {
        scope.attach_pin(Box::new(Arc::clone(writer)));
    }
    if let Some(pin) = control_scope_pin {
        scope.attach_pin(pin);
    }
    if let Some(parent) = connected_parent.as_ref() {
        scope.attach_pin(Box::new(Arc::clone(parent)));
    }
    if let Some(owner) = collection.as_ref() {
        scope.attach_pin(Box::new(Arc::clone(owner)));
    }
    // Before native entry, reserve both ends in the original scope. The only
    // owner transfer is a no-allocation move between these retained slots.
    let storage_slot = if let Some(original) = storage {
        let original = Arc::new(std::sync::Mutex::new(Some(original)));
        scope.attach_pin(Box::new(Arc::clone(&original)));
        let slot = storage::RoventStorageTransferSlotV3::new_empty();
        scope.attach_pin(Box::new(Arc::clone(&slot)));
        let mut retained = original
            .lock()
            .map_err(|_| error("StorageCustodyV3", "original storage slot poisoned"))?;
        slot.adopt_from(&mut retained)
            .map_err(|failure| error("StorageCustodyV3", failure))?;
        drop(retained);
        Some(slot)
    } else {
        None
    };
    scope.arm();
    macro_rules! attach_storage {
        ($driver:ident, $custody:ident) => {
            if let Some(slot) = storage_slot.as_ref() {
                // Preserve this exact product even if attach panics before it
                // returns. Neither a failed attach nor a panic is a fence.
                scope.attach_pin(Box::new(Arc::clone(&$driver)));
                let attached = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    $driver.attach_staged_storage_runtime_registration_v3(slot, &boundary, &clock)
                }));
                let failure: Option<(_, Box<dyn std::any::Any + Send>)> = match attached {
                    Ok(Ok(())) => None,
                    Ok(Err(failure)) => {
                        Some((error("StorageAttachV3", &failure.detail), Box::new(failure)))
                    }
                    Err(payload) => Some((
                        error("StorageAttachV3", "storage attachment panicked"),
                        payload,
                    )),
                };
                if let Some((detail, attach_failure)) = failure {
                    // Attempt bounded closure on this exact session. An error,
                    // unavailable result or panic is retained, never a fence.
                    let close = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        $driver.finish_session(control.cleanup_until)
                    }));
                    let confirmed = matches!(
                        &close,
                        Ok(Ok(Some(
                            crate::search_driver::DriverPhysicalCompletion::Confirmed
                        )))
                    ) && $driver.physical_close_observation().is_confirmed()
                        // Native completion alone does not settle the separate
                        // Collection teacher and observer custody.
                        && collection.is_none();
                    if confirmed {
                        scope.settled_without_quarantine();
                        return Err(RoventV3LaunchFailure::Settled {
                            error: detail,
                            owner: V3SettledFactoryOwner {
                                original: Box::new((
                                    Arc::clone(&$driver),
                                    Arc::clone(&$custody),
                                    attach_failure,
                                    close,
                                )),
                                events: Vec::new(),
                            },
                        });
                    }
                    return Err(RoventV3LaunchFailure::Quarantined {
                        error: detail,
                        owner: RoventV3QuarantinedOwner {
                            cleanup_until: control.cleanup_until,
                            scope_key: scope.park(
                                Box::new((
                                    Arc::clone(&$driver),
                                    Arc::clone(&$custody),
                                    attach_failure,
                                    close,
                                )),
                                None,
                            ),
                            recovery_key: None,
                        },
                    });
                }
            }
        };
    }
    type SessionReleaseProbe =
        Box<dyn Fn() -> Option<crate::pals_adaptive::NativeSessionReleaseObservation>>;
    type PcCleanupProbe =
        Box<dyn Fn() -> Option<super::super::pc_collection_v3::RoventPcCleanupObservationV3>>;
    let mut observe_pc_cleanup: Option<PcCleanupProbe> = None;
    // Research-only observation uses the very same model owner and physical
    // ticket path. It is never enabled by the normal engine configuration.
    let mut numeric_observer: Option<super::super::V3SystemNumericObserver> = None;
    #[cfg(feature = "native-validation-diagnostics")]
    let numeric_handle = if std::env::var_os("RZ_V3_SYSTEM_NUMERIC_CAPTURE").as_deref()
        == Some(std::ffi::OsStr::new("1"))
        && lock.manifest.purpose == rz_experiments::RoventPurpose::Validation
        && matches!(
            &lock.manifest.system_runtime_lock.runtime,
            NativeRuntimeSourceV6::SystemCuda(_)
        ) {
        let (observer, handle) = super::system_numeric_observation::attach();
        numeric_observer = Some(super::super::V3SystemNumericObserver(observer));
        Some(handle)
    } else {
        None
    };
    // The same launch/custody and rejection path serves both own CPU
    // implementations; archive remains available only to a supporting owner.
    macro_rules! attach_own_storage {
        (enabled, $driver:ident, $custody:ident) => {
            attach_storage!($driver, $custody)
        };
        (disabled, $driver:ident, $custody:ident) => {};
    }
    macro_rules! launch_own {
        ($assembly:expr, $storage:ident) => {{
            // The same assembly slot owns loaded inputs before the worker can
            // enter native code. It stays in this scope through errors/panics.
            let custody = Arc::new(std::sync::Mutex::new($assembly));
            scope.attach_pin(Box::new(Arc::clone(&custody)));
            let prepared = match super::super::prepare_native_v3_retaining_with_parent::<
                AdaptivePortQuestion,
            >(
                Arc::clone(&sources),
                resources,
                clock.runtime_clock(),
                control.clone(),
                connected_parent.clone(),
                numeric_observer.take(),
            ) {
                Ok(prepared) => prepared,
                Err(failure) => {
                    let unknown = runtime_receipt(&lock, arguments.process_epoch, None, false);
                    write_receipt(
                        receipt_writer.as_deref(),
                        "system-runtime-startup.v6.json",
                        &lock,
                        &unknown,
                    );
                    let failure = reject_factory_v3(failure, control.cleanup_until, &diagnostic);
                    if matches!(&failure, V3FactoryRejection::Unstarted { .. }) {
                        scope.mark_before_native();
                    }
                    write_receipt(
                        receipt_writer.as_deref(),
                        "system-runtime-termination.v6.json",
                        &lock,
                        &unknown,
                    );
                    if matches!(
                        failure,
                        V3FactoryRejection::Ordinary(_)
                            | V3FactoryRejection::Unstarted { .. }
                            | V3FactoryRejection::Settled { .. }
                    ) {
                        scope.settled_without_quarantine();
                    }
                    return Err(match failure {
                        V3FactoryRejection::Ordinary(error) => error.into(),
                        V3FactoryRejection::Unstarted { error, owner } => {
                            RoventV3LaunchFailure::Unstarted { error, owner }
                        }
                        V3FactoryRejection::Settled { error, owner } => {
                            RoventV3LaunchFailure::Settled { error, owner }
                        }
                        V3FactoryRejection::BeforeNativePhysicalPending {
                            error,
                            recovered,
                            recovery_key,
                            first_recovery_error,
                        } => RoventV3LaunchFailure::Quarantined {
                            error,
                            owner: RoventV3QuarantinedOwner {
                                cleanup_until: control.cleanup_until,
                                scope_key: scope.park(
                                    Box::new(V3FactoryPending {
                                        original_failure: Some(recovered.original_failure),
                                        context: Some(Box::new((
                                            boundary,
                                            sources,
                                            selection_file,
                                            lock_file,
                                            custody,
                                        ))),
                                        events: recovered.events,
                                        key: recovery_key.clone(),
                                        before_native: true,
                                        recovered_factory: Some(recovered.factory),
                                        first_recovery_error: Some(first_recovery_error),
                                    }),
                                    recovery_key.clone(),
                                ),
                                recovery_key,
                            },
                        },
                        V3FactoryRejection::Unknown {
                            error,
                            original,
                            recovery_key,
                            events,
                            before_native,
                            first_recovery_error,
                        } => RoventV3LaunchFailure::Quarantined {
                            error,
                            owner: RoventV3QuarantinedOwner {
                                cleanup_until: control.cleanup_until,
                                scope_key: scope.park(
                                    Box::new(V3FactoryPending {
                                        original_failure: Some(Box::new(original)),
                                        context: Some(Box::new((
                                            boundary,
                                            sources,
                                            selection_file,
                                            lock_file,
                                            custody,
                                        ))),
                                        events,
                                        key: recovery_key.clone(),
                                        before_native,
                                        recovered_factory: None,
                                        first_recovery_error,
                                    }),
                                    recovery_key.clone(),
                                ),
                                recovery_key,
                            },
                        },
                    });
                }
            };
            if let Err(prepared) = custody
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .install_prepared(prepared)
            {
                // This branch should be unreachable for a fresh custody slot;
                // nevertheless the returned original must not be dropped.
                let recovery_key = prepared.service.recovery_key();
                return Err(RoventV3LaunchFailure::Quarantined {
                    error: error(
                        "PhysicalCompletionUnknown",
                        "assembly slot rejected original prepared owner",
                    ),
                    owner: RoventV3QuarantinedOwner {
                        cleanup_until: control.cleanup_until,
                        scope_key: scope.park(
                            Box::new((prepared, Arc::clone(&custody))),
                            recovery_key.clone(),
                        ),
                        recovery_key,
                    },
                });
            }
            let observation = custody
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .system_runtime_observation();
            let startup =
                runtime_receipt(&lock, arguments.process_epoch, observation.as_ref(), true);
            write_receipt(
                receipt_writer.as_deref(),
                "system-runtime-startup.v6.json",
                &lock,
                &startup,
            );
            let assembled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                custody.lock().unwrap_or_else(|e| e.into_inner()).assemble(
                    clock.clone(),
                    Arc::clone(&owners),
                    identity,
                    arguments.root_policy,
                )
            }));
            let driver = match assembled {
                Ok(Ok(driver)) => driver,
                Ok(Err(failure)) => {
                    let detail = error("ProductAssembly", &failure);
                    let unknown =
                        runtime_receipt(&lock, arguments.process_epoch, observation.as_ref(), true);
                    write_receipt(
                        receipt_writer.as_deref(),
                        "system-runtime-termination.v6.json",
                        &lock,
                        &unknown,
                    );
                    let (drained, recovery_key) = {
                        let mut owner = custody.lock().unwrap_or_else(|e| e.into_inner());
                        match owner.prepared_service() {
                            Some(service) => {
                                let key = service.recovery_key();
                                (Some(drain_factory_v3(service, control.cleanup_until)), key)
                            }
                            None => (None, None),
                        }
                    };
                    let confirmed = drained
                        .as_ref()
                        .is_some_and(|d| d.joined && d.physical_confirmed);
                    let events = drained.map_or_else(Vec::new, |d| d.events);
                    if confirmed {
                        scope.settled_without_quarantine();
                        return Err(RoventV3LaunchFailure::Settled {
                            error: detail,
                            owner: V3SettledFactoryOwner {
                                original: Box::new((failure, Arc::clone(&custody))),
                                events,
                            },
                        });
                    }
                    return Err(RoventV3LaunchFailure::Quarantined {
                        error: error("PhysicalCompletionUnknown", detail),
                        owner: RoventV3QuarantinedOwner {
                            cleanup_until: control.cleanup_until,
                            scope_key: scope.park(
                                Box::new((failure, events, Arc::clone(&custody))),
                                recovery_key.clone(),
                            ),
                            recovery_key,
                        },
                    });
                }
                Err(payload) => {
                    let recovery_key = custody
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .prepared_service()
                        .and_then(|service| service.recovery_key());
                    return Err(RoventV3LaunchFailure::Quarantined {
                        error: error("PhysicalCompletionUnknown", "product assembly panicked"),
                        owner: RoventV3QuarantinedOwner {
                            cleanup_until: control.cleanup_until,
                            scope_key: scope.park(
                                Box::new((payload, Arc::clone(&custody))),
                                recovery_key.clone(),
                            ),
                            recovery_key,
                        },
                    });
                }
            };
            attach_own_storage!($storage, driver, custody);
            let completion = driver.physical_close_observation();
            let original_driver = Arc::clone(&driver);
            let observe_release: SessionReleaseProbe =
                Box::new(move || original_driver.session_release_observation());
            (driver, completion, observation, Some(observe_release))
        }};
    }
    let (driver, completion, observation, observe_release): (
        Arc<dyn SearchSessionDriver>,
        _,
        _,
        Option<SessionReleaseProbe>,
    ) = match product {
        ProductChoice::OwnPcr(loaded) => launch_own!(
            super::super::product::RoventOwnNnuecAssemblyV3::new(loaded, Arc::clone(&boundary)),
            enabled
        ),
        ProductChoice::OwnBoard => launch_own!(
            super::super::product::RoventOwnBoardAssemblyV3::new(Arc::clone(&boundary)),
            disabled
        ),
        ProductChoice::External(external, evidence) => {
            // Preserve the original external source, retirement authority and
            // process before any native startup; all live in the same slot.
            let custody = Arc::new(std::sync::Mutex::new(
                super::super::product::RoventExternalAssemblyV3::new(
                    external,
                    evidence,
                    Arc::clone(&boundary),
                ),
            ));
            scope.attach_pin(Box::new(Arc::clone(&custody)));
            let prepared = match super::super::prepare_native_v3_retaining_with_parent::<
                AdaptivePortQuestion,
            >(
                Arc::clone(&sources),
                resources,
                clock.runtime_clock(),
                control.clone(),
                connected_parent.clone(),
                numeric_observer.take(),
            ) {
                Ok(prepared) => prepared,
                Err(failure) => {
                    let unknown = runtime_receipt(&lock, arguments.process_epoch, None, false);
                    write_receipt(
                        receipt_writer.as_deref(),
                        "system-runtime-startup.v6.json",
                        &lock,
                        &unknown,
                    );
                    let failure = reject_factory_v3(failure, control.cleanup_until, &diagnostic);
                    if matches!(&failure, V3FactoryRejection::Unstarted { .. }) {
                        scope.mark_before_native();
                    }
                    write_receipt(
                        receipt_writer.as_deref(),
                        "system-runtime-termination.v6.json",
                        &lock,
                        &unknown,
                    );
                    if matches!(
                        failure,
                        V3FactoryRejection::Ordinary(_)
                            | V3FactoryRejection::Unstarted { .. }
                            | V3FactoryRejection::Settled { .. }
                    ) {
                        scope.settled_without_quarantine();
                    }
                    return Err(match failure {
                        V3FactoryRejection::Ordinary(error) => error.into(),
                        V3FactoryRejection::Unstarted { error, owner } => {
                            RoventV3LaunchFailure::Unstarted { error, owner }
                        }
                        V3FactoryRejection::Settled { error, owner } => {
                            RoventV3LaunchFailure::Settled { error, owner }
                        }
                        V3FactoryRejection::BeforeNativePhysicalPending {
                            error,
                            recovered,
                            recovery_key,
                            first_recovery_error,
                        } => RoventV3LaunchFailure::Quarantined {
                            error,
                            owner: RoventV3QuarantinedOwner {
                                cleanup_until: control.cleanup_until,
                                scope_key: scope.park(
                                    Box::new(V3FactoryPending {
                                        original_failure: Some(recovered.original_failure),
                                        context: Some(Box::new((
                                            boundary,
                                            sources,
                                            selection_file,
                                            external_selection_file,
                                            lock_file,
                                            custody,
                                        ))),
                                        events: recovered.events,
                                        key: recovery_key.clone(),
                                        before_native: true,
                                        recovered_factory: Some(recovered.factory),
                                        first_recovery_error: Some(first_recovery_error),
                                    }),
                                    recovery_key.clone(),
                                ),
                                recovery_key,
                            },
                        },
                        V3FactoryRejection::Unknown {
                            error,
                            original,
                            recovery_key,
                            events,
                            before_native,
                            first_recovery_error,
                        } => RoventV3LaunchFailure::Quarantined {
                            error,
                            owner: RoventV3QuarantinedOwner {
                                cleanup_until: control.cleanup_until,
                                scope_key: scope.park(
                                    Box::new(V3FactoryPending {
                                        original_failure: Some(Box::new(original)),
                                        context: Some(Box::new((
                                            boundary,
                                            sources,
                                            selection_file,
                                            external_selection_file,
                                            lock_file,
                                            custody,
                                        ))),
                                        events,
                                        key: recovery_key.clone(),
                                        before_native,
                                        recovered_factory: None,
                                        first_recovery_error,
                                    }),
                                    recovery_key.clone(),
                                ),
                                recovery_key,
                            },
                        },
                    });
                }
            };
            if let Err(prepared) = custody
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .install_prepared(prepared)
            {
                // This branch should be unreachable for a fresh custody slot;
                // nevertheless the returned original must not be dropped.
                let recovery_key = prepared.service.recovery_key();
                return Err(RoventV3LaunchFailure::Quarantined {
                    error: error(
                        "PhysicalCompletionUnknown",
                        "assembly slot rejected original prepared owner",
                    ),
                    owner: RoventV3QuarantinedOwner {
                        cleanup_until: control.cleanup_until,
                        scope_key: scope.park(
                            Box::new((prepared, Arc::clone(&custody))),
                            recovery_key.clone(),
                        ),
                        recovery_key,
                    },
                });
            }
            let observation = custody
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .system_runtime_observation();
            let startup =
                runtime_receipt(&lock, arguments.process_epoch, observation.as_ref(), true);
            write_receipt(
                receipt_writer.as_deref(),
                "system-runtime-startup.v6.json",
                &lock,
                &startup,
            );
            let assembled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                custody.lock().unwrap_or_else(|e| e.into_inner()).assemble(
                    clock.clone(),
                    Arc::clone(&owners),
                    identity,
                    arguments.root_policy,
                )
            }));
            let driver = match assembled {
                Ok(Ok(driver)) => driver,
                Ok(Err(failure)) => {
                    let detail = error("ProductAssembly", &failure);
                    let unknown =
                        runtime_receipt(&lock, arguments.process_epoch, observation.as_ref(), true);
                    write_receipt(
                        receipt_writer.as_deref(),
                        "system-runtime-termination.v6.json",
                        &lock,
                        &unknown,
                    );
                    // A native fence alone cannot retire an external process.
                    // Only the same custody's explicit NeverStarted state plus
                    // its own native completion permits this scope to close.
                    let (drained, recovery_key) = {
                        let mut owner = custody.lock().unwrap_or_else(|e| e.into_inner());
                        let key = owner.recovery_key();
                        let external_absent = matches!(
                            owner.external_start_observation(),
                            crate::pals_adaptive::RoventExternalV3StartObservation::NeverStarted
                        );
                        let drained = if external_absent {
                            owner
                                .prepared_service()
                                .map(|service| drain_factory_v3(service, control.cleanup_until))
                        } else {
                            None
                        };
                        (drained, key)
                    };
                    let confirmed = drained
                        .as_ref()
                        .is_some_and(|d| d.joined && d.physical_confirmed);
                    let events = drained.map_or_else(Vec::new, |d| d.events);
                    if confirmed {
                        scope.settled_without_quarantine();
                        return Err(RoventV3LaunchFailure::Settled {
                            error: detail,
                            owner: V3SettledFactoryOwner {
                                original: Box::new((failure, Arc::clone(&custody))),
                                events,
                            },
                        });
                    }
                    return Err(RoventV3LaunchFailure::Quarantined {
                        error: error("PhysicalCompletionUnknown", detail),
                        owner: RoventV3QuarantinedOwner {
                            cleanup_until: control.cleanup_until,
                            scope_key: scope.park(
                                Box::new((failure, events, Arc::clone(&custody))),
                                recovery_key.clone(),
                            ),
                            recovery_key,
                        },
                    });
                }
                Err(payload) => {
                    let recovery_key = custody
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .recovery_key();
                    return Err(RoventV3LaunchFailure::Quarantined {
                        error: error("PhysicalCompletionUnknown", "product assembly panicked"),
                        owner: RoventV3QuarantinedOwner {
                            cleanup_until: control.cleanup_until,
                            scope_key: scope.park(
                                Box::new((payload, Arc::clone(&custody))),
                                recovery_key.clone(),
                            ),
                            recovery_key,
                        },
                    });
                }
            };
            let completion = driver.physical_close_observation();
            // The external product has no same-owner release accessor yet.
            (driver, completion, observation, None)
        }
        pc_product @ (ProductChoice::Pc | ProductChoice::PcShared(_)) => {
            let shared_loaded = match pc_product {
                ProductChoice::Pc => None,
                ProductChoice::PcShared(loaded) => Some(loaded),
                ProductChoice::OwnBoard
                | ProductChoice::OwnPcr(_)
                | ProductChoice::External(_, _) => {
                    unreachable!("matched P/C product")
                }
            };
            // The optional shared NNUEC and later native service remain in
            // one registered assembly slot from before native startup.
            let assembly = match collection.as_ref() {
                Some(owner) => super::super::product::RoventPcAssemblyV3::new_collection(
                    shared_loaded,
                    Arc::clone(&boundary),
                    Arc::clone(owner.hooks()),
                ),
                None => super::super::product::RoventPcAssemblyV3::new(
                    shared_loaded,
                    Arc::clone(&boundary),
                ),
            };
            let custody = Arc::new(std::sync::Mutex::new(assembly));
            scope.attach_pin(Box::new(Arc::clone(&custody)));
            let prepared_result = if let Some(owner) = collection.as_ref() {
                let observer = match owner.observer_proxy() {
                    Ok(observer) => observer,
                    Err(failure) => {
                        scope.mark_before_native();
                        scope.settled_without_quarantine();
                        return Err(failure.into());
                    }
                };
                super::collection_v3::prepare_collection_with_parent::<PcPortQuestion>(
                    Arc::clone(&sources),
                    resources,
                    clock.runtime_clock(),
                    control.clone(),
                    observer,
                    connected_parent.clone(),
                )
            } else {
                super::super::prepare_native_v3_retaining_with_parent::<PcPortQuestion>(
                    Arc::clone(&sources),
                    resources,
                    clock.runtime_clock(),
                    control.clone(),
                    connected_parent.clone(),
                    numeric_observer.take(),
                )
            };
            let prepared = match prepared_result {
                Ok(prepared) => prepared,
                Err(failure) => {
                    let unknown = runtime_receipt(&lock, arguments.process_epoch, None, false);
                    write_receipt(
                        receipt_writer.as_deref(),
                        "system-runtime-startup.v6.json",
                        &lock,
                        &unknown,
                    );
                    let failure = reject_factory_v3(failure, control.cleanup_until, &diagnostic);
                    if matches!(&failure, V3FactoryRejection::Unstarted { .. }) {
                        scope.mark_before_native();
                    }
                    write_receipt(
                        receipt_writer.as_deref(),
                        "system-runtime-termination.v6.json",
                        &lock,
                        &unknown,
                    );
                    if matches!(
                        failure,
                        V3FactoryRejection::Ordinary(_)
                            | V3FactoryRejection::Unstarted { .. }
                            | V3FactoryRejection::Settled { .. }
                    ) {
                        scope.settled_without_quarantine();
                    }
                    return Err(match failure {
                        V3FactoryRejection::Ordinary(error) => error.into(),
                        V3FactoryRejection::Unstarted { error, owner } => {
                            RoventV3LaunchFailure::Unstarted { error, owner }
                        }
                        V3FactoryRejection::Settled { error, owner } => {
                            RoventV3LaunchFailure::Settled { error, owner }
                        }
                        V3FactoryRejection::BeforeNativePhysicalPending {
                            error,
                            recovered,
                            recovery_key,
                            first_recovery_error,
                        } => RoventV3LaunchFailure::Quarantined {
                            error,
                            owner: RoventV3QuarantinedOwner {
                                cleanup_until: control.cleanup_until,
                                scope_key: scope.park(
                                    Box::new(V3FactoryPending {
                                        original_failure: Some(recovered.original_failure),
                                        context: Some(Box::new((
                                            boundary,
                                            sources,
                                            selection_file,
                                            lock_file,
                                            custody,
                                        ))),
                                        events: recovered.events,
                                        key: recovery_key.clone(),
                                        before_native: true,
                                        recovered_factory: Some(recovered.factory),
                                        first_recovery_error: Some(first_recovery_error),
                                    }),
                                    recovery_key.clone(),
                                ),
                                recovery_key,
                            },
                        },
                        V3FactoryRejection::Unknown {
                            error,
                            original,
                            recovery_key,
                            events,
                            before_native,
                            first_recovery_error,
                        } => RoventV3LaunchFailure::Quarantined {
                            error,
                            owner: RoventV3QuarantinedOwner {
                                cleanup_until: control.cleanup_until,
                                scope_key: scope.park(
                                    Box::new(V3FactoryPending {
                                        original_failure: Some(Box::new(original)),
                                        context: Some(Box::new((
                                            boundary,
                                            sources,
                                            selection_file,
                                            lock_file,
                                            custody,
                                        ))),
                                        events,
                                        key: recovery_key.clone(),
                                        before_native,
                                        recovered_factory: None,
                                        first_recovery_error,
                                    }),
                                    recovery_key.clone(),
                                ),
                                recovery_key,
                            },
                        },
                    });
                }
            };
            if let Err(prepared) = custody
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .install_prepared(prepared)
            {
                // This branch should be unreachable for a fresh custody slot;
                // nevertheless the returned original must not be dropped.
                let recovery_key = prepared.service.recovery_key();
                return Err(RoventV3LaunchFailure::Quarantined {
                    error: error(
                        "PhysicalCompletionUnknown",
                        "assembly slot rejected original prepared owner",
                    ),
                    owner: RoventV3QuarantinedOwner {
                        cleanup_until: control.cleanup_until,
                        scope_key: scope.park(
                            Box::new((prepared, Arc::clone(&custody))),
                            recovery_key.clone(),
                        ),
                        recovery_key,
                    },
                });
            }
            let observation = custody
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .system_runtime_observation();
            let startup =
                runtime_receipt(&lock, arguments.process_epoch, observation.as_ref(), true);
            write_receipt(
                receipt_writer.as_deref(),
                "system-runtime-startup.v6.json",
                &lock,
                &startup,
            );
            let assembled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                custody.lock().unwrap_or_else(|e| e.into_inner()).assemble(
                    clock.clone(),
                    Arc::clone(&owners),
                    identity,
                    arguments.root_policy,
                )
            }));
            let driver = match assembled {
                Ok(Ok(driver)) => driver,
                Ok(Err(failure)) => {
                    let detail = error("ProductAssembly", &failure);
                    let unknown =
                        runtime_receipt(&lock, arguments.process_epoch, observation.as_ref(), true);
                    write_receipt(
                        receipt_writer.as_deref(),
                        "system-runtime-termination.v6.json",
                        &lock,
                        &unknown,
                    );
                    let (drained, recovery_key) = {
                        let mut owner = custody.lock().unwrap_or_else(|e| e.into_inner());
                        match owner.prepared_service() {
                            Some(service) => {
                                let key = service.recovery_key();
                                (Some(drain_factory_v3(service, control.cleanup_until)), key)
                            }
                            None => (None, None),
                        }
                    };
                    let confirmed = drained
                        .as_ref()
                        .is_some_and(|d| d.joined && d.physical_confirmed);
                    let events = drained.map_or_else(Vec::new, |d| d.events);
                    if confirmed {
                        scope.settled_without_quarantine();
                        return Err(RoventV3LaunchFailure::Settled {
                            error: detail,
                            owner: V3SettledFactoryOwner {
                                original: Box::new((failure, Arc::clone(&custody))),
                                events,
                            },
                        });
                    }
                    return Err(RoventV3LaunchFailure::Quarantined {
                        error: error("PhysicalCompletionUnknown", detail),
                        owner: RoventV3QuarantinedOwner {
                            cleanup_until: control.cleanup_until,
                            scope_key: scope.park(
                                Box::new((failure, events, Arc::clone(&custody))),
                                recovery_key.clone(),
                            ),
                            recovery_key,
                        },
                    });
                }
                Err(payload) => {
                    let recovery_key = custody
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .prepared_service()
                        .and_then(|service| service.recovery_key());
                    return Err(RoventV3LaunchFailure::Quarantined {
                        error: error("PhysicalCompletionUnknown", "product assembly panicked"),
                        owner: RoventV3QuarantinedOwner {
                            cleanup_until: control.cleanup_until,
                            scope_key: scope.park(
                                Box::new((payload, Arc::clone(&custody))),
                                recovery_key.clone(),
                            ),
                            recovery_key,
                        },
                    });
                }
            };
            attach_storage!(driver, custody);
            let completion = driver.physical_close_observation();
            if collection.is_some() {
                let original_driver = Arc::clone(&driver);
                observe_pc_cleanup = Some(Box::new(move || {
                    original_driver.collection_cleanup_observation_v3()
                }));
            }
            let original_driver = Arc::clone(&driver);
            let observe_release: SessionReleaseProbe =
                Box::new(move || original_driver.session_release_observation());
            (driver, completion, observation, Some(observe_release))
        }
    };
    let mut settings = EngineSettings {
        shutdown_limit: Duration::from_millis(lock.manifest.cleanup_max_ms),
        ..EngineSettings::default()
    };
    settings.search.untimed_limit = Duration::from_millis(lock.manifest.wall_time_max_ms);
    // Retain the same driver and process owner across UCI return until this
    // exact session's physical closure is known or transferred to a caller.
    // Keep the original driver in this frame across a UCI unwind. A caught
    // panic is not physical completion, and its original payload is retained
    // with the same owner below instead of being reduced to a string.
    let (result, panic_payload) =
        serve_driver_retaining(&driver, clock, Arc::clone(&owners), settings, go_observer);
    let physical_confirmed = completion.is_confirmed();
    #[cfg(feature = "native-validation-diagnostics")]
    if let Some(handle) = numeric_handle {
        match (
            receipt_writer.as_deref(),
            handle.to_json(
                &lock.canonical_sha256,
                arguments.process_epoch,
                physical_confirmed,
            ),
        ) {
            (Some(writer), Ok(bytes)) => {
                if let Err(failure) =
                    writer.write_bytes("system-numeric-observation.v1.json", &bytes)
                {
                    eprintln!("system numeric observation warning: {failure}");
                }
            }
            (None, _) => {
                eprintln!("system numeric observation unavailable: no original receipt writer")
            }
            (_, Err(failure)) => eprintln!("system numeric observation warning: {failure}"),
        }
    }
    let session_release = observe_release.and_then(|probe| probe());
    if let Some(probe) = observe_pc_cleanup {
        use super::super::pc_collection_v3::RoventPcCleanupObservationV3 as Cleanup;
        let failure = match probe() {
            Some(Cleanup::NoFailureObserved) => serde_json::json!({"status": "not_observed"}),
            Some(Cleanup::FirstFailure {
                kind,
                detail,
                recovered,
            }) => serde_json::json!({
                "status": "observed", "kind": kind, "detail": detail,
                "same_marker_cleanup_later_confirmed": recovered,
            }),
            Some(Cleanup::Unknown) | None => serde_json::json!({"status": "unknown"}),
        };
        let value = serde_json::json!({
            "domain": "rz-rovent-pc-collection-cleanup-observation/1",
            "process_id": std::process::id(), "process_epoch": arguments.process_epoch,
            "lock_sha256": lock.canonical_sha256,
            "first_failure": failure,
            "scope": "same process-local P/C collection session; diagnostic only",
            "physical_completion_or_release_authority": false,
        });
        match receipt_writer.as_deref() {
            Some(writer) => {
                if let Err(failure) = writer.write_bytes(
                    "collection-cleanup-observation.v1.json",
                    value.to_string().as_bytes(),
                ) {
                    eprintln!("collection cleanup observation warning: {failure}");
                }
            }
            None => {
                eprintln!("collection cleanup observation unavailable: no original receipt writer")
            }
        }
    }
    if session_release.is_some() {
        scope.mark_session_released();
    }
    if let Some(writer) = receipt_writer.as_deref() {
        let value =
            session_release_observation_json(&lock, arguments.process_epoch, session_release);
        if let Err(failure) = writer.write_bytes(
            "session-release-observation.v1.json",
            value.to_string().as_bytes(),
        ) {
            eprintln!("session release observation warning: {failure}");
        }
        let value =
            system_runtime_finish_observation_json(&lock, arguments.process_epoch, session_release);
        if let Err(failure) = writer.write_bytes(
            "system-runtime-finish-observation.v1.json",
            value.to_string().as_bytes(),
        ) {
            eprintln!("system runtime finish observation warning: {failure}");
        }
    }
    let mut termination =
        runtime_receipt(&lock, arguments.process_epoch, observation.as_ref(), true);
    if physical_confirmed {
        // This is an observation of this product's confirmed finish_session,
        // not a CUDA fence token or an independent supervisor proof.
        termination.physical_completion = rz_experiments::PhysicalCompletionV6::Confirmed {
            completion_id: format!(
                "session_finish_confirmed/1:{}:{}",
                std::process::id(),
                arguments.process_epoch
            ),
        };
    }
    write_receipt(
        receipt_writer.as_deref(),
        "system-runtime-termination.v6.json",
        &lock,
        &termination,
    );
    if !physical_confirmed || panic_payload.is_some() {
        return Err(RoventV3LaunchFailure::Quarantined {
            error: error(
                if physical_confirmed {
                    "UciPanic"
                } else {
                    "PhysicalCompletionUnknown"
                },
                match result {
                    Ok(()) => "UCI returned before actual physical finish".into(),
                    Err(failure) => format!("UCI failed before actual physical finish: {failure}"),
                },
            ),
            // A product may also own external CPU/NNUEC resources. The
            // available native key alone does not attest that whole scope.
            owner: RoventV3QuarantinedOwner {
                cleanup_until: control.cleanup_until,
                scope_key: scope.park(
                    Box::new((
                        driver,
                        owners,
                        boundary,
                        sources,
                        selection_file,
                        external_selection_file,
                        lock_file,
                        panic_payload,
                    )),
                    None,
                ),
                recovery_key: None,
            },
        });
    }
    scope.mark_product_finished(arguments.process_epoch, &lock.canonical_sha256);
    scope.settled_without_quarantine();
    result.map_err(Into::into)
}

#[cfg(not(target_os = "linux"))]
pub(super) fn run_native(
    _arguments: Arguments,
    _lock: Arc<RoventInputLockV3>,
    _root: Dir,
    _cwd: PathBuf,
    _lock_file: File,
) -> Result<(), RoventLaunchError> {
    Err(error(
        "InputLockUnsupportedV3",
        "system CUDA native owner requires Linux",
    ))
}

#[cfg(not(target_os = "linux"))]
pub(super) fn run_native_with_resource<'a>(
    _arguments: Arguments,
    _lock: Arc<RoventInputLockV3>,
    _root: Dir,
    _cwd: PathBuf,
    _lock_file: File,
    _resource: ResourceAdmissionV3<'a>,
) -> Result<(), RoventLaunchError> {
    Err(error(
        "InputLockUnsupportedV3",
        "system CUDA native owner requires Linux",
    ))
}
