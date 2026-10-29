//! Bounded recovery custody for original NNUEC collection handoffs.
//!
//! An escrow conveys ownership only. It never certifies native completion,
//! teacher retirement, worker join, or corpus publication. Those facts stay
//! with their original owners and must be checked by the outer consumer.

use super::{RoventCollectionRootAdmissionV2, RoventNnuecCollectionHandoffV2};
use std::sync::{Arc, Mutex, OnceLock, TryLockError};
use std::sync::atomic::{AtomicBool, Ordering};

/// Process-local maximum, not a limit on games or searches. Full capacity can
/// be reported separately from NN execution by the caller.
pub const HANDOFF_ESCROW_LIMIT_V2: usize = 32;

type OriginalHandoff = Arc<Mutex<RoventNnuecCollectionHandoffV2>>;

struct Entry {
    admission: RoventCollectionRootAdmissionV2,
    original: Mutex<Option<OriginalHandoff>>,
    leased: AtomicBool,
    session_detached: AtomicBool,
}

struct Registry {
    slots: [Option<Arc<Entry>>; HANDOFF_ESCROW_LIMIT_V2],
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry {
        slots: std::array::from_fn(|_| None),
    }))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandoffEscrowErrorV2 {
    Busy,
    Poisoned,
    Capacity,
    Stale,
    DifferentAdmission,
    DifferentHandoff,
    AlreadyInstalled,
    MissingHandoff,
    AlreadyLeased,
    LiveReservation,
    NotEmpty,
}

/// Opaque original process-local slot identity. Cloning a key creates no new
/// slot or right to publish and cannot recreate an admission.
#[derive(Clone)]
pub struct HandoffEscrowKeyV2 {
    index: usize,
    entry: Arc<Entry>,
}

/// The session retains this reservation; the registry owns a second Arc
/// before any handoff is installed. Dropping it never destroys that entry.
pub struct HandoffEscrowReservationV2 {
    key: HandoffEscrowKeyV2,
}

/// Move-only lease over the same original handoff. Dropping it is nonblocking:
/// the registry retains the original and a later caller can lease it again.
pub struct HandoffEscrowLeaseV2 {
    key: HandoffEscrowKeyV2,
    original: OriginalHandoff,
    armed: bool,
}

fn lock_registry() -> Result<std::sync::MutexGuard<'static, Registry>, HandoffEscrowErrorV2> {
    registry().try_lock().map_err(|error| match error {
        TryLockError::WouldBlock => HandoffEscrowErrorV2::Busy,
        TryLockError::Poisoned(_) => HandoffEscrowErrorV2::Poisoned,
    })
}

fn entry_is_original(registry: &Registry, key: &HandoffEscrowKeyV2) -> bool {
    registry.slots.get(key.index).and_then(Option::as_ref)
        .is_some_and(|entry| Arc::ptr_eq(entry, &key.entry))
}

impl HandoffEscrowReservationV2 {
    /// Reserve before ownership may need recovery. A full or busy registry
    /// neither starts a worker nor changes the original admission.
    pub fn try_reserve(admission: &RoventCollectionRootAdmissionV2)
        -> Result<Self, HandoffEscrowErrorV2>
    {
        let entry = Arc::new(Entry {
            admission: admission.clone(),
            original: Mutex::new(None),
            leased: AtomicBool::new(false),
            session_detached: AtomicBool::new(false),
        });
        let mut guard = lock_registry()?;
        let index = guard.slots.iter().position(Option::is_none)
            .ok_or(HandoffEscrowErrorV2::Capacity)?;
        guard.slots[index] = Some(Arc::clone(&entry));
        Ok(Self { key: HandoffEscrowKeyV2 { index, entry } })
    }

    pub fn key(&self) -> HandoffEscrowKeyV2 { self.key.clone() }

    /// Install exactly the supplied original. Failure returns that original;
    /// this check does not imply a native fence or worker termination.
    pub fn try_install(&self, original: OriginalHandoff)
        -> Result<(), (HandoffEscrowErrorV2, OriginalHandoff)>
    {
        let same_admission = original.try_lock()
            .map(|handoff| handoff.admission.same_original_admission(&self.key.entry.admission))
            .map_err(|error| match error {
                TryLockError::WouldBlock => HandoffEscrowErrorV2::Busy,
                TryLockError::Poisoned(_) => HandoffEscrowErrorV2::Poisoned,
            });
        let same_admission = match same_admission {
            Ok(same) => same,
            Err(error) => return Err((error, original)),
        };
        if !same_admission { return Err((HandoffEscrowErrorV2::DifferentAdmission, original)); }
        let registry = match lock_registry() {
            Ok(registry) => registry,
            Err(error) => return Err((error, original)),
        };
        if !entry_is_original(&registry, &self.key) {
            return Err((HandoffEscrowErrorV2::Stale, original));
        }
        let mut slot = match self.key.entry.original.try_lock() {
            Ok(slot) => slot,
            Err(TryLockError::WouldBlock) => return Err((HandoffEscrowErrorV2::Busy, original)),
            Err(TryLockError::Poisoned(_)) => return Err((HandoffEscrowErrorV2::Poisoned, original)),
        };
        if slot.is_some() { return Err((HandoffEscrowErrorV2::AlreadyInstalled, original)); }
        *slot = Some(original);
        Ok(())
    }

    /// Remove an empty reservation only. It says nothing about any worker or
    /// model owner, which must be accounted for by its own lifecycle.
    pub fn try_cancel_empty(&self) -> Result<(), HandoffEscrowErrorV2> {
        self.key.try_cancel_empty_inner(false)
    }
}

impl Drop for HandoffEscrowReservationV2 {
    fn drop(&mut self) { self.key.entry.session_detached.store(true, Ordering::Release); }
}

impl HandoffEscrowKeyV2 {
    /// Recovery may also return an empty original reservation after its
    /// session disappeared. This never attests worker or model shutdown.
    pub fn try_cancel_empty(&self) -> Result<(), HandoffEscrowErrorV2> {
        self.try_cancel_empty_inner(true)
    }

    fn try_cancel_empty_inner(&self, detached_required: bool) -> Result<(), HandoffEscrowErrorV2> {
        let mut registry = lock_registry()?;
        if !entry_is_original(&registry, self) { return Err(HandoffEscrowErrorV2::Stale); }
        if detached_required && !self.entry.session_detached.load(Ordering::Acquire) {
            return Err(HandoffEscrowErrorV2::LiveReservation);
        }
        let slot = self.entry.original.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => HandoffEscrowErrorV2::Busy,
            TryLockError::Poisoned(_) => HandoffEscrowErrorV2::Poisoned,
        })?;
        if slot.is_some() || self.entry.leased.load(Ordering::Acquire) {
            return Err(HandoffEscrowErrorV2::NotEmpty);
        }
        registry.slots[self.index] = None;
        Ok(())
    }
    /// List registered identities. These keys provide recovery custody only,
    /// and require matching admission and handoff before leasing.
    pub fn try_recoverable_keys() -> Result<Vec<Self>, HandoffEscrowErrorV2> {
        let registry = lock_registry()?;
        Ok(registry.slots.iter().enumerate().filter_map(|(index, slot)|
            slot.as_ref().map(|entry| Self { index, entry: Arc::clone(entry) })
        ).collect())
    }

    pub fn session_detached(&self) -> bool { self.entry.session_detached.load(Ordering::Acquire) }

    /// Recover the same registered original by its opaque key after an owner
    /// was dropped. This grants custody only, not authority or completion.
    pub fn try_recover_lease(&self) -> Result<HandoffEscrowLeaseV2, HandoffEscrowErrorV2> {
        let registry = lock_registry()?;
        if !entry_is_original(&registry, self) { return Err(HandoffEscrowErrorV2::Stale); }
        let slot = self.entry.original.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => HandoffEscrowErrorV2::Busy,
            TryLockError::Poisoned(_) => HandoffEscrowErrorV2::Poisoned,
        })?;
        let held = slot.as_ref().ok_or(HandoffEscrowErrorV2::MissingHandoff)?;
        self.entry.leased.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| HandoffEscrowErrorV2::AlreadyLeased)?;
        Ok(HandoffEscrowLeaseV2 { key: self.clone(), original: Arc::clone(held), armed: true })
    }

    /// Both the source admission and original Arc must match. The registry
    /// keeps the same Arc throughout the lease and after an abandoned lease.
    pub fn try_lease(&self, admission: &RoventCollectionRootAdmissionV2, original: &OriginalHandoff)
        -> Result<HandoffEscrowLeaseV2, HandoffEscrowErrorV2>
    {
        if !self.entry.admission.same_original_admission(admission) {
            return Err(HandoffEscrowErrorV2::DifferentAdmission);
        }
        let registry = lock_registry()?;
        if !entry_is_original(&registry, self) { return Err(HandoffEscrowErrorV2::Stale); }
        let slot = self.entry.original.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => HandoffEscrowErrorV2::Busy,
            TryLockError::Poisoned(_) => HandoffEscrowErrorV2::Poisoned,
        })?;
        let held = slot.as_ref().ok_or(HandoffEscrowErrorV2::MissingHandoff)?;
        if !Arc::ptr_eq(held, original) { return Err(HandoffEscrowErrorV2::DifferentHandoff); }
        self.entry.leased.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| HandoffEscrowErrorV2::AlreadyLeased)?;
        Ok(HandoffEscrowLeaseV2 { key: self.clone(), original: Arc::clone(held), armed: true })
    }
}

impl HandoffEscrowLeaseV2 {
    pub fn original(&self) -> &OriginalHandoff { &self.original }

    /// Release this escrow's ownership only after the outer owner has taken
    /// responsibility for the original handoff and its dependent evidence.
    /// This method does not report or validate worker, physical or corpus
    /// completion; those remain the Arena owner's separate responsibility.
    /// On error the identical lease is returned and the registry keeps the
    /// original, so lock contention is retryable without losing custody.
    pub fn try_release_custody(mut self, admission: &RoventCollectionRootAdmissionV2, outer_original: &OriginalHandoff)
        -> Result<(), (HandoffEscrowErrorV2, Self)>
    {
        if !self.key.entry.admission.same_original_admission(admission) {
            return Err((HandoffEscrowErrorV2::DifferentAdmission, self));
        }
        if !Arc::ptr_eq(&self.original, outer_original) {
            return Err((HandoffEscrowErrorV2::DifferentHandoff, self));
        }
        let mut registry = match lock_registry() {
            Ok(registry) => registry,
            Err(error) => return Err((error, self)),
        };
        if !entry_is_original(&registry, &self.key) {
            return Err((HandoffEscrowErrorV2::Stale, self));
        }
        // Clone only the stable entry, so any failure can return the original
        // lease even while acquiring the entry's mutex.
        let entry = Arc::clone(&self.key.entry);
        let mut slot = match entry.original.try_lock() {
            Ok(slot) => slot,
            Err(TryLockError::WouldBlock) => return Err((HandoffEscrowErrorV2::Busy, self)),
            Err(TryLockError::Poisoned(_)) => return Err((HandoffEscrowErrorV2::Poisoned, self)),
        };
        if !slot.as_ref().is_some_and(|held| Arc::ptr_eq(held, &self.original)) {
            return Err((HandoffEscrowErrorV2::DifferentHandoff, self));
        }
        slot.take();
        registry.slots[self.key.index] = None;
        self.armed = false;
        self.key.entry.leased.store(false, Ordering::Release);
        Ok(())
    }
}

impl Drop for HandoffEscrowLeaseV2 {
    fn drop(&mut self) {
        if self.armed { self.key.entry.leased.store(false, Ordering::Release); }
    }
}
