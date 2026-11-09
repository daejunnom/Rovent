//! Original same-invocation Own /3 recovery owners.
//!
//! This entry is WIP: no safe registry removal or managed CLI activation is
//! provided yet. The first successful registration fixes the process-local
//! profile. An unstarted reservation may be reclaimed from its original
//! registered entry, but the entry itself is not removed. Until per-profile
//! namespaces and original no-start/settlement removal exist,
//! repeated production use can exhaust its explicit capacity. A native guard,
//! process return, or this registry alone does not prove teacher, output,
//! control, or GPU physical completion.

use super::own_prepared_v3::OwnCollectionPreparedV3;
use super::pc_lock_v3::{ManagedOutputOwnerV3, PreparedPcCollectionInputsV3};
use super::pc_managed_prepare_v3::PreparedManagedOwnV3;
use crate::rovent_resource_scope::launch::{
    NativeResourceCustody, OwnCollectionFactoryV3, OwnCollectionRetainResourceV3,
    RoventControlledRunFailure, RoventNativeResourcePlanPin,
    RoventNativeSystemResourceRunReceiptV3,
};
use cap_std::fs::Dir;
use rz_experiments::RoventInputLockV3;
use rz_uci::rovent_bootstrap::RoventStartupControl;
use rz_uci::rovent_bootstrap::launch::{
    RoventLaunchError, RoventOwnFactoryProductV3, RoventOwnUciNoDispatchV3,
};
use std::any::Any;
use std::fs::File;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Instant;

// This gate controls only the right to invoke this entry's controlled action;
// it is not proof about unrelated processes or physical GPU completion.
#[repr(u8)]
enum ReservationPhaseV3 {
    Available = 0,
    Held = 1,
    Started = 2,
}

struct ReservationGateV3(AtomicU8);

impl ReservationGateV3 {
    fn held() -> Self {
        Self(AtomicU8::new(ReservationPhaseV3::Held as u8))
    }

    fn reclaim(&self) -> bool {
        self.0
            .compare_exchange(
                ReservationPhaseV3::Available as u8,
                ReservationPhaseV3::Held as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    fn abandon_unstarted(&self) {
        // Started is irreversible: dropping the token after execute must
        // never make an in-flight or unresolved entry available again.
        let _ = self.0.compare_exchange(
            ReservationPhaseV3::Held as u8,
            ReservationPhaseV3::Available as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    fn start(&self) -> bool {
        self.0
            .compare_exchange(
                ReservationPhaseV3::Held as u8,
                ReservationPhaseV3::Started as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }
}

/// Pins only this invocation's original identity and input/output owners.
/// In particular it cannot reach the external entry, its outcome or scope
/// key. The native scope may retain this leaf without forming that cycle.
struct OwnScopeLeafV3 {
    _inputs: Arc<PreparedPcCollectionInputsV3>,
    _output: Arc<ManagedOutputOwnerV3>,
    _input_lock: Arc<RoventInputLockV3>,
    _lock_file: Arc<File>,
}

struct OwnRunStateV3 {
    in_flight: bool,
    resource: Option<Arc<Mutex<NativeResourceCustody>>>,
    factory_entered: bool,
    input_lock: Option<Arc<RoventInputLockV3>>,
    lock_file: Option<Arc<File>>,
    startup: Option<RoventStartupControl>,
    prepared: Option<Arc<OwnCollectionPreparedV3>>,
    no_uci_dispatch: Option<RoventOwnUciNoDispatchV3>,
}

pub(crate) enum OwnRunOutcomeV3 {
    Returned(Result<RoventNativeSystemResourceRunReceiptV3, RoventControlledRunFailure>),
    Panicked(Box<dyn Any + Send>),
}

struct OwnRunEntryV3 {
    identity: Arc<()>,
    reservation: ReservationGateV3,
    inputs: Arc<PreparedPcCollectionInputsV3>,
    output: Arc<ManagedOutputOwnerV3>,
    arguments: Vec<String>,
    work_until: Instant,
    cleanup_until: Instant,
    state: Mutex<OwnRunStateV3>,
    // Held by execute across action. Its callbacks only acquire state, never
    // this mutex. Outcome observers must use try_lock and never invert order.
    outcome: Mutex<Option<OwnRunOutcomeV3>>,
}

struct OwnRunRegistryV3 {
    maximum_live_attempts: NonZeroUsize,
    entries: Vec<Arc<OwnRunEntryV3>>,
}

fn registry() -> &'static Mutex<Option<OwnRunRegistryV3>> {
    static REGISTRY: OnceLock<Mutex<Option<OwnRunRegistryV3>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(None))
}

/// A single non-cloneable reservation for the original registered entry.
/// Capacity must come from an explicit execution profile, not a root or ticket
/// count. Registration is not authority to start native or teacher work.
pub(crate) struct OwnRunReservationV3 {
    entry: Arc<OwnRunEntryV3>,
}

impl Drop for OwnRunReservationV3 {
    fn drop(&mut self) {
        self.entry.reservation.abandon_unstarted();
    }
}

/// Observation/recovery identity only; it does not grant release authority.
pub(crate) struct OwnRunHandleV3 {
    entry: Arc<OwnRunEntryV3>,
}

impl OwnRunHandleV3 {
    /// Enumerate the same process-local entries, including runs whose caller
    /// dropped its handle. This is not proof of termination or permission to
    /// remove any entry. A poisoned registry is never reported as empty.
    pub(crate) fn registered() -> Result<Vec<Self>, &'static str> {
        let registry = registry()
            .try_lock()
            .map_err(|_| "Own /3 registry unavailable")?;
        let Some(registry) = registry.as_ref() else {
            return Ok(Vec::new());
        };
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(registry.entries.len())
            .map_err(|_| "Own /3 recovery handle allocation failed")?;
        handles.extend(registry.entries.iter().map(|entry| Self {
            entry: Arc::clone(entry),
        }));
        Ok(handles)
    }

    /// Whether an original return or panic is retained. A busy outcome slot
    /// is not treated as a missing outcome or as physical completion.
    pub(crate) fn outcome_recorded(&self) -> Result<bool, &'static str> {
        self.entry
            .outcome
            .try_lock()
            .map(|outcome| outcome.is_some())
            .map_err(|_| "Own /3 outcome slot unavailable")
    }

    /// Reclaim only this still-registered entry's original unstarted action
    /// reservation. An outstanding token or an already-started action cannot
    /// be duplicated. This does not settle the output, control, parent lease,
    /// other processes, or any GPU work.
    pub(crate) fn try_reacquire_reservation(
        &self,
    ) -> Result<Option<OwnRunReservationV3>, &'static str> {
        let registry = registry()
            .try_lock()
            .map_err(|_| "Own /3 registry unavailable")?;
        let Some(registry) = registry.as_ref() else {
            return Err("Own /3 original recovery entry missing");
        };
        if !registry
            .entries
            .iter()
            .any(|entry| Arc::ptr_eq(entry, &self.entry))
        {
            return Err("Own /3 original recovery entry missing");
        }
        if !self.entry.reservation.reclaim() {
            return Ok(None);
        }
        Ok(Some(OwnRunReservationV3 {
            entry: Arc::clone(&self.entry),
        }))
    }

    /// Observe the original outcome without transferring its custody. A
    /// caller panic is resumed only after the original guard is released
    /// normally, so it does not discard or poison the retained result.
    pub(crate) fn try_with_original_outcome<R>(
        &self,
        observe: impl for<'a> FnOnce(&'a OwnRunOutcomeV3) -> R,
    ) -> Result<Option<R>, &'static str> {
        let outcome = self
            .entry
            .outcome
            .try_lock()
            .map_err(|_| "Own /3 outcome slot unavailable")?;
        let Some(original) = outcome.as_ref() else {
            return Ok(None);
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| observe(original)));
        drop(outcome);
        match result {
            Ok(value) => Ok(Some(value)),
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
}

impl OwnRunReservationV3 {
    pub(crate) fn register(
        prepared: PreparedManagedOwnV3,
        maximum_live_attempts: NonZeroUsize,
    ) -> Result<Self, (PreparedManagedOwnV3, &'static str)> {
        if !prepared.output.matches_inputs(&prepared.inputs) {
            return Err((prepared, "Own /3 managed input/output owners differ"));
        }
        let mut registry = match registry().lock() {
            Ok(registry) => registry,
            Err(_) => return Err((prepared, "Own /3 registry poisoned")),
        };
        if let Some(existing) = registry.as_ref() {
            if existing.maximum_live_attempts != maximum_live_attempts {
                return Err((prepared, "Own /3 registry profile differs"));
            }
            if existing.entries.len() >= maximum_live_attempts.get() {
                return Err((
                    prepared,
                    "Own /3 registry capacity reached; original entries retained",
                ));
            }
        }
        // Do not pin an empty registry profile when allocation fails before
        // the first entry is actually registered.
        let mut first_entries = Vec::new();
        let entries = match registry.as_mut() {
            Some(existing) => &mut existing.entries,
            None => &mut first_entries,
        };
        if entries.try_reserve(1).is_err() {
            return Err((prepared, "Own /3 registry allocation failed"));
        }
        let entry = Arc::new(OwnRunEntryV3 {
            identity: Arc::new(()),
            reservation: ReservationGateV3::held(),
            inputs: prepared.inputs,
            output: prepared.output,
            arguments: prepared.arguments,
            work_until: prepared.work_until,
            cleanup_until: prepared.cleanup_until,
            state: Mutex::new(OwnRunStateV3 {
                in_flight: false,
                resource: None,
                factory_entered: false,
                input_lock: None,
                lock_file: None,
                startup: None,
                prepared: None,
                no_uci_dispatch: None,
            }),
            outcome: Mutex::new(None),
        });
        entries.push(Arc::clone(&entry));
        if registry.is_none() {
            *registry = Some(OwnRunRegistryV3 {
                maximum_live_attempts,
                entries: first_entries,
            });
        }
        Ok(Self { entry })
    }

    /// Derive the invocation only from this entry's original input and
    /// arguments, never from an independently supplied pin or command.
    pub(crate) fn original_invocation(
        &self,
    ) -> Result<(RoventNativeResourcePlanPin, Vec<String>), &'static str> {
        let pin = self
            .entry
            .inputs
            .resource_plan_pin()
            .map_err(|_| "Own /3 original resource plan pin unavailable")?;
        Ok((pin, self.entry.arguments.clone()))
    }

    /// Register before invoking any controlled resource/native callback.
    /// The original returned outcome or panic is retained in the entry before
    /// a handle is returned. Failure before begin returns the original token.
    pub(crate) fn execute<F>(self, action: F) -> Result<OwnRunHandleV3, (Self, &'static str)>
    where
        F: FnOnce(
            OwnCollectionFactoryV3,
            OwnCollectionRetainResourceV3,
        )
            -> Result<RoventNativeSystemResourceRunReceiptV3, RoventControlledRunFailure>,
    {
        let entry = Arc::clone(&self.entry);
        let mut outcome = match entry.outcome.try_lock() {
            Ok(outcome) => outcome,
            Err(_) => return Err((self, "Own /3 outcome slot unavailable")),
        };
        if outcome.is_some() {
            return Err((self, "Own /3 original outcome already recorded"));
        }
        {
            let mut state = match entry.state.try_lock() {
                Ok(state) => state,
                Err(_) => return Err((self, "Own /3 entry unavailable")),
            };
            if state.in_flight {
                return Err((self, "Own /3 original run already started"));
            }
            if !entry.reservation.start() {
                return Err((self, "Own /3 original reservation unavailable"));
            }
            state.in_flight = true;
        }
        let factory_entry = Arc::downgrade(&entry);
        let resource_entry = Arc::downgrade(&entry);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let factory: OwnCollectionFactoryV3 = Box::new(move |lock, root, file, startup| {
                own_factory(factory_entry, lock, root, file, startup)
            });
            let retain: OwnCollectionRetainResourceV3 =
                Box::new(move |resource| retain_resource(resource_entry, resource));
            action(factory, retain)
        }));
        // No callback, allocation, or other fallible operation occurs between
        // the returned original value and assignment into this reserved slot.
        *outcome = Some(match result {
            Ok(result) => OwnRunOutcomeV3::Returned(result),
            Err(payload) => OwnRunOutcomeV3::Panicked(payload),
        });
        drop(outcome);
        Ok(OwnRunHandleV3 { entry })
    }
}

fn retain_resource(
    entry: Weak<OwnRunEntryV3>,
    resource: &Arc<Mutex<NativeResourceCustody>>,
) -> Result<(), &'static str> {
    let entry = entry
        .upgrade()
        .ok_or("Own /3 original recovery entry missing")?;
    let mut state = entry
        .state
        .try_lock()
        .map_err(|_| "Own /3 original entry unavailable")?;
    if !state.in_flight || state.resource.is_some() {
        return Err("Own /3 original resource owner already supplied or not in flight");
    }
    state.resource = Some(Arc::clone(resource));
    Ok(())
}

fn own_factory(
    entry: Weak<OwnRunEntryV3>,
    lock: &Arc<RoventInputLockV3>,
    root: &Dir,
    lock_file: &Arc<File>,
    startup: &RoventStartupControl,
) -> Result<RoventOwnFactoryProductV3, RoventLaunchError> {
    let error = |detail: String| RoventLaunchError {
        stage: "OwnCollectionAdmission",
        detail,
    };
    let entry = entry
        .upgrade()
        .ok_or_else(|| error("original recovery entry missing".into()))?;
    let native = {
        // Keep this run's original state slot reserved while preparing. The
        // constructor starts no worker and does not call this entry; the
        // resource callback completed earlier and the proof sink runs only
        // after this factory returns. Other observers use try_lock and remain
        // pending. This avoids dropping a successfully prepared original
        // owner if a second lock acquisition is busy after construction.
        let mut state = entry
            .state
            .try_lock()
            .map_err(|_| error("original entry unavailable".into()))?;
        if !state.in_flight
            || state.resource.is_none()
            || state.factory_entered
            || state.prepared.is_some()
        {
            return Err(error("original factory is not available".into()));
        }
        state.factory_entered = true;
        state.input_lock = Some(Arc::clone(lock));
        state.lock_file = Some(Arc::clone(lock_file));
        state.startup = Some(startup.clone());
        startup
            .check()
            .map_err(|failure| error(failure.to_string()))?;
        let prepared = Arc::new(
            OwnCollectionPreparedV3::prepare(
                &entry.inputs,
                &entry.output,
                lock,
                root,
                lock_file,
                startup.until.min(entry.work_until),
                entry.work_until,
                entry.cleanup_until,
            )
            .map_err(|failure| error(failure.to_string()))?,
        );
        let native = Arc::clone(prepared.native_owner());
        state.prepared = Some(prepared);
        native
    };
    let leaf = OwnScopeLeafV3 {
        _inputs: Arc::clone(&entry.inputs),
        _output: Arc::clone(&entry.output),
        _input_lock: Arc::clone(lock),
        _lock_file: Arc::clone(lock_file),
    };
    let sink_entry = Arc::downgrade(&entry);
    let sink = Box::new(move |proof: RoventOwnUciNoDispatchV3| {
        let entry = sink_entry
            .upgrade()
            .ok_or_else(|| error("original recovery entry missing".into()))?;
        let mut state = entry
            .state
            .try_lock()
            .map_err(|_| error("original entry unavailable".into()))?;
        let matches = state
            .prepared
            .as_ref()
            .zip(state.input_lock.as_ref())
            .zip(state.lock_file.as_ref())
            .is_some_and(|((prepared, input), file)| {
                proof.matches_original(prepared.native_owner(), input, file, &entry.identity)
            });
        if !matches || state.no_uci_dispatch.is_some() {
            return Err(error(
                "original no-dispatch evidence differs or already recorded".into(),
            ));
        }
        state.no_uci_dispatch = Some(proof);
        Ok(())
    });
    Ok(RoventOwnFactoryProductV3::new(
        native,
        Arc::clone(lock),
        Arc::clone(lock_file),
        Box::new(leaf),
        Arc::clone(&entry.identity),
        sink,
    ))
}

#[cfg(test)]
mod reservation_tests {
    use super::ReservationGateV3;
    use std::sync::{Arc, Barrier};

    #[test]
    fn original_unstarted_reservation_has_one_reclaimer_and_started_is_not_reissued() {
        let gate = Arc::new(ReservationGateV3::held());
        assert!(
            !gate.reclaim(),
            "a live original reservation cannot be duplicated"
        );
        gate.abandon_unstarted();

        let rendezvous = Arc::new(Barrier::new(3));
        let successes = std::thread::scope(|threads| {
            let workers: Vec<_> = (0..2)
                .map(|_| {
                    let gate = Arc::clone(&gate);
                    let rendezvous = Arc::clone(&rendezvous);
                    threads.spawn(move || {
                        rendezvous.wait();
                        gate.reclaim()
                    })
                })
                .collect();
            rendezvous.wait();
            workers
                .into_iter()
                .map(|worker| if worker.join().unwrap() { 1 } else { 0 })
                .sum::<usize>()
        });
        assert_eq!(successes, 1, "the same original reservation has one owner");
        assert!(gate.start());
        gate.abandon_unstarted();
        assert!(!gate.reclaim(), "started or uncertain work is not reissued");
    }
}
