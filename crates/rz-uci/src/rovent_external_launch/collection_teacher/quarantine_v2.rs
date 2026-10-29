//! Process-local recovery slots for original Collection teacher owners.
//!
//! A slot is reserved before spawning. Its opaque key is kept by the caller;
//! it cannot authorize a teacher job or recreate a root. No worker operation,
//! join, root observation, or owner destruction is performed under this lock.

use super::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

const SLOT_LIMIT: usize = 16;
static CUSTODY_VIOLATION: AtomicBool = AtomicBool::new(false);

struct Slot {
    identity: Arc<()>,
    parked: Option<ParkedTeacherV2>,
}

struct Registry {
    slots: [Option<Slot>; SLOT_LIMIT],
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        Mutex::new(Registry {
            slots: std::array::from_fn(|_| None),
        })
    })
}

/// Opaque capability to recover only the same process-local reservation.
/// A cloned key refers to the same single slot, not another owner or permit.
#[derive(Clone)]
pub struct TeacherRecoveryKeyV2 {
    index: usize,
    identity: Arc<()>,
}

/// Caller-owned, bounded custody for keys registered before worker startup.
/// Keeping this outside a root session allows a dropped session's parked
/// original owner to be recovered without granting new teacher authority.
#[derive(Default)]
pub struct TeacherRecoveryCustodianV2 {
    keys: Vec<TeacherRecoveryKeyV2>,
}

impl TeacherRecoveryCustodianV2 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn keys(&self) -> &[TeacherRecoveryKeyV2] {
        &self.keys
    }

    /// Discover only already parked original owners in the bounded registry.
    /// This remains available if the original caller dropped its custodian;
    /// claiming a key still transfers only that owner's recovery custody.
    pub fn recoverable_keys() -> Result<Vec<TeacherRecoveryKeyV2>, CheckerError> {
        let guard = registry()
            .lock()
            .map_err(|_| invalid("teacher /2 quarantine registry poisoned"))?;
        Ok(guard
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| {
                let slot = slot.as_ref()?;
                slot.parked.as_ref()?;
                Some(TeacherRecoveryKeyV2 {
                    index,
                    identity: Arc::clone(&slot.identity),
                })
            })
            .collect())
    }

    pub(super) fn register(&mut self, key: &TeacherRecoveryKeyV2) -> Result<(), CheckerError> {
        let guard = registry()
            .lock()
            .map_err(|_| invalid("teacher /2 quarantine registry poisoned"))?;
        self.keys.retain(|existing| {
            guard
                .slots
                .get(existing.index)
                .and_then(Option::as_ref)
                .is_some_and(|slot| Arc::ptr_eq(&slot.identity, &existing.identity))
        });
        if self.keys.len() >= SLOT_LIMIT {
            return Err(invalid("teacher /2 recovery custodian capacity"));
        }
        if !guard
            .slots
            .get(key.index)
            .and_then(Option::as_ref)
            .is_some_and(|slot| Arc::ptr_eq(&slot.identity, &key.identity) && slot.parked.is_none())
        {
            return Err(invalid("teacher /2 recovery key reservation changed"));
        }
        self.keys
            .try_reserve(1)
            .map_err(|_| invalid("teacher /2 recovery custodian allocation"))?;
        self.keys.push(key.clone());
        Ok(())
    }
}

/// This linear reservation must be obtained before starting the worker.
pub(crate) struct TeacherRecoveryReservationV2 {
    key: TeacherRecoveryKeyV2,
    armed: bool,
}

pub(super) struct ParkedTeacherV2 {
    pub(super) owner: RoventCollectionTeacherOwnerV2,
    pub(super) shutdown_requested: bool,
    pub(super) root_observation_error: Option<rz_contracts::ContractError>,
    pub(super) diagnostics: RoventTeacherDiagnosticsDispositionV2,
    pub(super) cleanup_deadline: Instant,
    pub(super) registry_poisoned: bool,
    pub(super) terminal_ledger: TeacherTerminalLedgerV2,
}

impl TeacherRecoveryReservationV2 {
    pub(crate) fn reserve() -> Result<(Self, TeacherRecoveryKeyV2), CheckerError> {
        if CUSTODY_VIOLATION.load(Ordering::Acquire) {
            return Err(invalid("teacher /2 quarantine custody violation"));
        }
        let identity = Arc::new(());
        let mut guard = registry()
            .lock()
            .map_err(|_| invalid("teacher /2 quarantine registry poisoned"))?;
        if CUSTODY_VIOLATION.load(Ordering::Acquire) {
            return Err(invalid("teacher /2 quarantine custody violation"));
        }
        let index = guard
            .slots
            .iter()
            .position(Option::is_none)
            .ok_or(invalid("teacher /2 quarantine capacity"))?;
        guard.slots[index] = Some(Slot {
            identity: Arc::clone(&identity),
            parked: None,
        });
        let key = TeacherRecoveryKeyV2 { index, identity };
        Ok((
            Self {
                key: key.clone(),
                armed: true,
            },
            key,
        ))
    }

    /// Recheck immediately before worker startup. This observation cannot
    /// freeze another unrelated thread or grant process/teacher authority.
    pub(super) fn validate_before_spawn(&self) -> Result<(), CheckerError> {
        if CUSTODY_VIOLATION.load(Ordering::Acquire) {
            return Err(invalid("teacher /2 quarantine custody violation"));
        }
        let guard = registry()
            .lock()
            .map_err(|_| invalid("teacher /2 quarantine registry poisoned"))?;
        if !self.armed
            || !guard
                .slots
                .get(self.key.index)
                .and_then(Option::as_ref)
                .is_some_and(|slot| {
                    Arc::ptr_eq(&slot.identity, &self.key.identity) && slot.parked.is_none()
                })
            || CUSTODY_VIOLATION.load(Ordering::Acquire)
        {
            return Err(invalid("teacher /2 quarantine reservation changed"));
        }
        Ok(())
    }

    /// A reserved slot cannot be used by another owner. No allocation and no
    /// worker operation occurs here. An invariant failure returns the owner.
    pub(super) fn park(&mut self, mut owner: ParkedTeacherV2) -> Result<(), ParkedTeacherV2> {
        let (mut guard, poisoned) = match registry().lock() {
            Ok(guard) => (guard, false),
            Err(error) => (error.into_inner(), true),
        };
        owner.registry_poisoned |= poisoned;
        let slot = guard.slots.get_mut(self.key.index).and_then(Option::as_mut);
        match slot {
            Some(slot)
                if self.armed
                    && Arc::ptr_eq(&slot.identity, &self.key.identity)
                    && slot.parked.is_none() =>
            {
                slot.parked = Some(owner);
                self.armed = false;
                Ok(())
            }
            _ => Err(owner),
        }
    }

    /// Release only an empty, still-identical reservation. The returned flag
    /// records that a poisoned registry was encountered during recovery.
    pub(super) fn release(&mut self) -> Result<bool, ()> {
        let (mut guard, poisoned) = match registry().lock() {
            Ok(guard) => (guard, false),
            Err(error) => (error.into_inner(), true),
        };
        let slot = guard.slots.get_mut(self.key.index).ok_or(())?;
        if !self.armed
            || !slot.as_ref().is_some_and(|value| {
                Arc::ptr_eq(&value.identity, &self.key.identity) && value.parked.is_none()
            })
        {
            return Err(());
        }
        *slot = None;
        self.armed = false;
        Ok(poisoned)
    }
}

/// Fail closed if an internal invariant prevents the pre-reserved handoff.
/// The caller still retains the original worker by quarantining it separately.
pub(super) fn mark_custody_violation() {
    CUSTODY_VIOLATION.store(true, Ordering::Release);
}

impl TeacherRecoveryKeyV2 {
    /// Atomically claim the original parked owner once. The root may already
    /// be revoked: this provides recovery custody only, never new admission.
    pub fn claim(&self) -> Option<RoventCollectionTeacherLifecycleV2> {
        let (mut guard, poisoned) = match registry().lock() {
            Ok(guard) => (guard, false),
            Err(error) => (error.into_inner(), true),
        };
        let slot = guard.slots.get_mut(self.index)?.as_mut()?;
        if !Arc::ptr_eq(&slot.identity, &self.identity) {
            return None;
        }
        let mut parked = slot.parked.take()?;
        parked.registry_poisoned |= poisoned;
        drop(guard);
        Some(RoventCollectionTeacherLifecycleV2::from_parked(
            TeacherRecoveryReservationV2 {
                key: self.clone(),
                armed: true,
            },
            parked,
        ))
    }
}

impl Drop for TeacherRecoveryReservationV2 {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut guard = registry()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(slot) = guard.slots.get_mut(self.key.index) {
            if slot.as_ref().is_some_and(|value| {
                Arc::ptr_eq(&value.identity, &self.key.identity) && value.parked.is_none()
            }) {
                *slot = None;
            }
        }
    }
}
