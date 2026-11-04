//! Optional one-shot cancellation mailbox for one original EngineProcess.
//!
//! This control is independent of the bounded UCI event queue. Acknowledgment
//! means the same Owner invoked its ordinary logical cancellation for the
//! active search, or observed that none was active. It does not close future
//! admission, terminate UCI, or establish physical/GPU completion.

use super::Owner;
use rz_contracts::ProcessEpoch;
use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PriorityCancelRequest {
    Recorded,
    AlreadyRecorded,
    OwnerClosed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PriorityCancelAcknowledgment {
    ActiveSearch,
    NoActiveSearch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PriorityCancelApplicationFailure {
    OwnerPoisoned,
    CancelPanicked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PriorityCancelObservation {
    pub process_epoch: ProcessEpoch,
    pub requested: bool,
    pub acknowledgment: Option<PriorityCancelAcknowledgment>,
    pub application_failed: bool,
    pub application_failure: Option<PriorityCancelApplicationFailure>,
    pub owner_closed: bool,
    /// This mailbox does not observe any physical execution fence.
    pub physical_completion_observed: bool,
}

struct Shared {
    epoch: ProcessEpoch,
    requested: AtomicBool,
    acknowledgment: AtomicU8,
    failure: AtomicU8,
    owner_closed: AtomicBool,
    original_panic: Mutex<Option<Box<dyn Any + Send>>>,
}

/// Cloneable request endpoint. It carries no right to replace an owner, close
/// a process, or infer completion. No queue slot is needed to record a request.
#[derive(Clone)]
pub struct PriorityCancelHandle(Arc<Shared>);

impl PriorityCancelHandle {
    pub fn request(&self) -> PriorityCancelRequest {
        if self.0.owner_closed.load(Ordering::Acquire) {
            return PriorityCancelRequest::OwnerClosed;
        }
        if self.0.requested.swap(true, Ordering::AcqRel) {
            PriorityCancelRequest::AlreadyRecorded
        } else {
            // Owner closure can race this operation. A recorded request does
            // not assert delivery; `observe` reports closure independently.
            PriorityCancelRequest::Recorded
        }
    }

    pub fn observe(&self) -> PriorityCancelObservation {
        let acknowledgment = match self.0.acknowledgment.load(Ordering::Acquire) {
            1 => Some(PriorityCancelAcknowledgment::ActiveSearch),
            2 => Some(PriorityCancelAcknowledgment::NoActiveSearch),
            _ => None,
        };
        let failure = self.0.failure.load(Ordering::Acquire);
        PriorityCancelObservation {
            process_epoch: self.0.epoch,
            requested: self.0.requested.load(Ordering::Acquire),
            acknowledgment,
            application_failed: failure != 0,
            application_failure: match failure {
                1 => Some(PriorityCancelApplicationFailure::OwnerPoisoned),
                2 => Some(PriorityCancelApplicationFailure::CancelPanicked),
                _ => None,
            },
            owner_closed: self.0.owner_closed.load(Ordering::Acquire),
            physical_completion_observed: false,
        }
    }
}

/// Dropping the unique owner marks the mailbox closed, including startup
/// failure or dropping EngineProcess before `serve` begins.
pub(super) struct PriorityCancelOwner(Arc<Shared>);

impl PriorityCancelOwner {
    pub(super) fn new(epoch: ProcessEpoch) -> (Self, PriorityCancelHandle) {
        let shared = Arc::new(Shared {
            epoch,
            requested: AtomicBool::new(false),
            acknowledgment: AtomicU8::new(0),
            failure: AtomicU8::new(0),
            owner_closed: AtomicBool::new(false),
            original_panic: Mutex::new(None),
        });
        (Self(Arc::clone(&shared)), PriorityCancelHandle(shared))
    }

    pub(super) fn observer(&self) -> PriorityCancelObserver {
        PriorityCancelObserver(Arc::clone(&self.0))
    }
}

impl Drop for PriorityCancelOwner {
    fn drop(&mut self) {
        self.0.owner_closed.store(true, Ordering::Release);
    }
}

pub(super) struct PriorityCancelObserver(Arc<Shared>);

impl PriorityCancelObserver {
    pub(super) fn apply(&self, owner: &Owner) {
        if !self.0.requested.load(Ordering::Acquire)
            || self.0.acknowledgment.load(Ordering::Acquire) != 0
            || self.0.failure.load(Ordering::Acquire) != 0
        {
            return;
        }
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
    }

    pub(super) fn owner_poisoned(&self) {
        if self.0.requested.load(Ordering::Acquire)
            && self.0.acknowledgment.load(Ordering::Acquire) == 0
        {
            self.0.failure.store(1, Ordering::Release);
        }
    }
}
