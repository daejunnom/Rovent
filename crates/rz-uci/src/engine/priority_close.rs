//! Opt-in, same-Owner search admission closure, distinct from priority cancel.
//!
//! A request is only recorded until the Owner lock is acquired. Closing the
//! latch and checking `Effect::Start` both happen under that same lock. A Start
//! that acquired the lock first may already have begun and is then canceled;
//! after `admission_closed` no new Start may be dispatched by this engine.
//! Session exit, output preservation, and physical completion are separate.

use super::{
    Owner, PriorityCancelAcknowledgment, PriorityCancelApplicationFailure,
    PriorityCancelPanicAccessError, PriorityCancelRequest,
};
use rz_contracts::ProcessEpoch;
use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex, TryLockError,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PriorityCloseObservation {
    pub process_epoch: ProcessEpoch,
    pub requested: bool,
    pub admission_closed: bool,
    pub cancel_acknowledgment: Option<PriorityCancelAcknowledgment>,
    pub application_failure: Option<PriorityCancelApplicationFailure>,
    /// The opt-in engine owner ended or was dropped, not necessarily the OS process.
    pub owner_closed: bool,
    pub physical_completion_observed: bool,
}

struct Shared {
    epoch: ProcessEpoch,
    requested: AtomicBool,
    admission_closed: AtomicBool,
    acknowledgment: AtomicU8,
    failure: AtomicU8,
    owner_closed: AtomicBool,
    original_panic: Mutex<Option<Box<dyn Any + Send>>>,
}

#[derive(Clone)]
pub struct PriorityCloseHandle(Arc<Shared>);

impl PriorityCloseHandle {
    pub fn request(&self) -> PriorityCancelRequest {
        if self.0.owner_closed.load(Ordering::Acquire) {
            return PriorityCancelRequest::OwnerClosed;
        }
        if self.0.requested.swap(true, Ordering::AcqRel) {
            PriorityCancelRequest::AlreadyRecorded
        } else {
            PriorityCancelRequest::Recorded
        }
    }

    pub fn observe(&self) -> PriorityCloseObservation {
        let acknowledgment = match self.0.acknowledgment.load(Ordering::Acquire) {
            1 => Some(PriorityCancelAcknowledgment::ActiveSearch),
            2 => Some(PriorityCancelAcknowledgment::NoActiveSearch),
            _ => None,
        };
        let failure = match self.0.failure.load(Ordering::Acquire) {
            1 => Some(PriorityCancelApplicationFailure::OwnerPoisoned),
            2 => Some(PriorityCancelApplicationFailure::CancelPanicked),
            _ => None,
        };
        PriorityCloseObservation {
            process_epoch: self.0.epoch,
            requested: self.0.requested.load(Ordering::Acquire),
            admission_closed: self.0.admission_closed.load(Ordering::Acquire),
            cancel_acknowledgment: acknowledgment,
            application_failure: failure,
            owner_closed: self.0.owner_closed.load(Ordering::Acquire),
            physical_completion_observed: false,
        }
    }

    /// Inspect, without taking or printing, the original cancel panic.
    pub fn with_original_cancel_panic<R>(
        &self,
        inspect: impl FnOnce(&(dyn Any + Send)) -> R,
    ) -> Result<Option<R>, PriorityCancelPanicAccessError> {
        let guard = match self.0.original_panic.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return Err(PriorityCancelPanicAccessError::Busy),
            Err(TryLockError::Poisoned(_)) => return Err(PriorityCancelPanicAccessError::Poisoned),
        };
        let inspected = catch_unwind(AssertUnwindSafe(|| guard.as_deref().map(inspect)));
        drop(guard);
        match inspected {
            Ok(value) => Ok(value),
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
}

pub(super) struct PriorityCloseOwner(Arc<Shared>);
impl PriorityCloseOwner {
    pub(super) fn new(epoch: ProcessEpoch) -> (Self, PriorityCloseHandle) {
        let shared = Arc::new(Shared {
            epoch,
            requested: AtomicBool::new(false),
            admission_closed: AtomicBool::new(false),
            acknowledgment: AtomicU8::new(0),
            failure: AtomicU8::new(0),
            owner_closed: AtomicBool::new(false),
            original_panic: Mutex::new(None),
        });
        (Self(Arc::clone(&shared)), PriorityCloseHandle(shared))
    }

    pub(super) fn observer(&self) -> PriorityCloseObserver {
        PriorityCloseObserver(Arc::clone(&self.0))
    }

    pub(super) fn matches(&self, handle: &PriorityCloseHandle) -> bool {
        Arc::ptr_eq(&self.0, &handle.0)
    }
}
impl Drop for PriorityCloseOwner {
    fn drop(&mut self) {
        self.0.owner_closed.store(true, Ordering::Release);
    }
}

#[derive(Clone)]
pub(super) struct PriorityCloseObserver(Arc<Shared>);
impl PriorityCloseObserver {
    /// The caller must hold the exact Owner mutex also used by Start dispatch.
    pub(super) fn apply_under_owner_lock(&self, owner: &Owner) -> bool {
        if !self.0.requested.load(Ordering::Acquire)
            || self.0.admission_closed.load(Ordering::Acquire)
            || self.0.failure.load(Ordering::Acquire) != 0
        {
            return false;
        }
        self.0.admission_closed.store(true, Ordering::Release);
        let active = owner.active.is_some();
        match catch_unwind(AssertUnwindSafe(|| owner.cancel())) {
            Ok(()) => self
                .0
                .acknowledgment
                .store(if active { 1 } else { 2 }, Ordering::Release),
            Err(payload) => {
                *self
                    .0
                    .original_panic
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Some(payload);
                self.0.failure.store(2, Ordering::Release);
            }
        }
        true
    }

    /// Called only while holding the same Owner mutex as `apply`.
    pub(super) fn closed_under_owner_lock(&self) -> bool {
        self.0.admission_closed.load(Ordering::Acquire)
    }

    pub(super) fn owner_poisoned(&self) {
        if self.0.requested.load(Ordering::Acquire)
            && !self.0.admission_closed.load(Ordering::Acquire)
        {
            self.0.failure.store(1, Ordering::Release);
        }
    }
}
