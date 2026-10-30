//! Routes Own NNUEC Collection operations to their original root bridge.
//!
//! Each slot owns a distinct bridge. Preparing a slot is not a claim of
//! native, teacher or publisher completion; the previous root's original
//! worker, fence and outbox must be separated before the shared game gate is
//! released. This router does not itself create workers or run the model.
use super::{ArenaError, invalid, own_hooks_lock_v2::{RoventNnuecCollectionBridgeV2, OwnRootSeparationV2}};
use rz_uci::pals_adaptive::{AdaptiveCollectionBoundaryV2, AdaptiveCollectionError,
    AdaptiveCollectionStage, AdaptiveManagerError, AdaptiveShutdownReceipt,
    RoventCollectionRootAdmissionV2, RoventNoCaptureRootCapabilityV2};
use rz_uci::rovent_bootstrap::{RoventStartupControl, nnuec_session::{
    RoventNnuecCollectionAdmissionFailureV2, RoventNnuecCollectionAdmissionOutcomeV2,
    RoventNnuecCollectionAdmissionPortV2, RoventNnuecCollectionHandoffV2,
    RoventNnuecCollectionHooksV2, RoventNnuecPendingCollectionV2,
    RoventNnuecRootHookFailureV2, RoventPendingEvidenceDispositionV2,
}};
use std::sync::{Arc, Mutex, OnceLock, mpsc, atomic::{AtomicU8, Ordering}};

const EMPTY: u8 = 0;
const READY: u8 = 1;
const RESERVING: u8 = 2;
const INSTALLED: u8 = 3;
const FAILED: u8 = 4;
const DEFERRED: u8 = 5;
const PREPARING: u8 = 6;
const PREINSTALL_FAILED: u8 = 7;
const UNCERTAIN: u8 = 8;

struct PreparationGuard<'a> { state: &'a AtomicU8, published: bool }
impl Drop for PreparationGuard<'_> {
    fn drop(&mut self) {
        if !self.published {
            let _ = self.state.compare_exchange(PREPARING, EMPTY, Ordering::AcqRel, Ordering::Acquire);
        }
    }
}

/// A panic while a slot is reserved cannot prove the install did not start.
struct AdmissionGuard<'a> { state: &'a AtomicU8, settled: bool }
impl Drop for AdmissionGuard<'_> {
    fn drop(&mut self) {
        if !self.settled {
            let _ = self.state.compare_exchange(RESERVING, FAILED, Ordering::AcqRel, Ordering::Acquire);
        }
    }
}

pub(crate) enum OwnPreparedSlotOutcomeV2 { Pending, Published { retained_rows: usize } }

pub(crate) struct RoventOwnRootRouterV2 {
    slots: Vec<OnceLock<Arc<RoventNnuecCollectionBridgeV2>>>,
    states: Vec<AtomicU8>,
    attempts: Vec<OnceLock<RoventCollectionRootAdmissionV2>>,
    stopped: Vec<OnceLock<Arc<RoventNnuecPendingCollectionV2>>>,
    last_deferred: Mutex<Option<RoventCollectionRootAdmissionV2>>,
    startup: OnceLock<RoventStartupControl>,
}

impl RoventOwnRootRouterV2 {
    /// Called before native manager entry. Does not start any worker.
    pub(crate) fn new(primary: Arc<RoventNnuecCollectionBridgeV2>, maximum_roots: usize) -> Result<Self, ArenaError> {
        if !(1..=64).contains(&maximum_roots) { return Err(invalid("Own root router capacity")); }
        let mut slots = Vec::new();
        slots.try_reserve_exact(maximum_roots).map_err(|_| invalid("Own root slots allocation"))?;
        slots.resize_with(maximum_roots, OnceLock::new);
        slots[0].set(primary).map_err(|_| invalid("Own primary root slot already occupied"))?;
        let mut states = Vec::new();
        states.try_reserve_exact(maximum_roots).map_err(|_| invalid("Own root state allocation"))?;
        states.resize_with(maximum_roots, || AtomicU8::new(EMPTY));
        states[0].store(READY, Ordering::Release);
        let mut attempts = Vec::new();
        attempts.try_reserve_exact(maximum_roots).map_err(|_| invalid("Own root attempt allocation"))?;
        attempts.resize_with(maximum_roots, OnceLock::new);
        let mut stopped = Vec::new();
        stopped.try_reserve_exact(maximum_roots).map_err(|_| invalid("Own stopped root allocation"))?;
        stopped.resize_with(maximum_roots, OnceLock::new);
        Ok(Self { slots, states, attempts, stopped, last_deferred: Mutex::new(None), startup: OnceLock::new() })
    }
    fn primary(&self) -> &Arc<RoventNnuecCollectionBridgeV2> { self.slots[0].get().expect("primary root owner") }
    pub(crate) fn original_startup(&self) -> Option<&RoventStartupControl> { self.startup.get() }
    pub(crate) fn latest_installed(&self) -> Option<(Arc<RoventNnuecCollectionBridgeV2>, RoventCollectionRootAdmissionV2)> {
        self.slots.iter().zip(&self.states).rev().find_map(|(slot, state)| {
            if state.load(Ordering::Acquire) != INSTALLED { return None; }
            let bridge = slot.get()?;
            Some((Arc::clone(bridge), bridge.installed_admission_for_recovery()?.clone()))
        })
    }
    pub(crate) fn next_slot_unprepared(&self) -> bool {
        self.states.iter().rposition(|state| state.load(Ordering::Acquire) == INSTALLED)
            .and_then(|index| self.states.get(index + 1))
            .is_some_and(|state| state.load(Ordering::Acquire) == EMPTY)
    }
    fn follows(previous: &RoventNnuecCollectionBridgeV2, next: &RoventCollectionRootAdmissionV2) -> bool {
        let Some(original) = previous.installed_admission_for_recovery() else { return false; };
        let old = original.root_admission().authority();
        let new = next.root_admission().authority();
        !original.same_original_admission(next)
            && original.native_origin().same_original_origin(next.native_origin())
            && old.epoch == new.epoch && old.implementation == new.implementation
            && new.game == old.game && new.root > old.root
    }
    fn follows_admission(previous: &RoventCollectionRootAdmissionV2, next: &RoventCollectionRootAdmissionV2) -> bool {
        let old = previous.root_admission().authority();
        let new = next.root_admission().authority();
        !previous.same_original_admission(next)
            && previous.native_origin().same_original_origin(next.native_origin())
            && old.epoch == new.epoch && old.implementation == new.implementation
            && (new.game > old.game || (new.game == old.game && new.root > old.root))
    }
    fn original_slot(&self, admission: &RoventCollectionRootAdmissionV2) -> Result<&Arc<RoventNnuecCollectionBridgeV2>, RoventNnuecRootHookFailureV2> {
        let mut found = None;
        for bridge in self.slots.iter().filter_map(OnceLock::get) {
            if bridge.matches_original_attempt(admission) {
                if found.is_some() {
                    return Err(RoventNnuecRootHookFailureV2 { admission: admission.clone(), error: AdaptiveManagerError::Invalid("Own original root appears in multiple slots") });
                }
                found = Some(bridge);
            }
        }
        found.ok_or_else(|| RoventNnuecRootHookFailureV2 { admission: admission.clone(), error: AdaptiveManagerError::Invalid("Own original root slot unavailable") })
    }
    fn defer_if_closed(&self, port: &mut RoventNnuecCollectionAdmissionPortV2<'_>, admission: RoventCollectionRootAdmissionV2)
        -> Result<RoventNnuecCollectionAdmissionOutcomeV2, RoventNnuecCollectionAdmissionFailureV2> {
        let previous = self.slots.iter().zip(&self.states).rev()
            .find_map(|(slot, state)| (state.load(Ordering::Acquire) == INSTALLED).then(|| slot.get()).flatten());
        let Some(previous) = previous else {
            if self.states.iter().any(|state| state.load(Ordering::Acquire) == DEFERRED) {
                return self.defer_serialized(port, admission, AdaptiveCollectionError::new(
                    AdaptiveCollectionStage::Dispatch, "OwnCollectionPreviousDeferred", "the earlier root did not start collection"));
            }
            return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission });
        };
        if !previous.installed_admission_for_recovery().is_some_and(|original| Self::follows_admission(original, &admission)) {
            return Err(RoventNnuecCollectionAdmissionFailureV2 { error: AdaptiveManagerError::Invalid("Own next root has a different or stale origin"), admission, owners: Box::new(()) });
        }
        if !previous.closed_before_next_admission(&admission) { return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission }); }
        self.defer_serialized(port, admission, AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Dispatch, "OwnCollectionWorkerUnavailable", "previous root is retained and the next collector is not ready"))
    }

    /// Caller either owns the slot reservation or has checked that this root
    /// has no in-progress installation. The mutex serializes marker order.
    fn defer_serialized(&self, port: &mut RoventNnuecCollectionAdmissionPortV2<'_>, admission: RoventCollectionRootAdmissionV2,
        reason: AdaptiveCollectionError) -> Result<RoventNnuecCollectionAdmissionOutcomeV2, RoventNnuecCollectionAdmissionFailureV2> {
        let mut previous = match self.last_deferred.try_lock() {
            Ok(previous) => previous,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission }),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(RoventNnuecCollectionAdmissionFailureV2 {
                error: AdaptiveManagerError::Invalid("Own deferred root custody poisoned"), admission, owners: Box::new(()),
            }),
        };
        if previous.as_ref().is_some_and(|original| !original.same_original_admission(&admission) && !Self::follows_admission(original, &admission)) {
            return Err(RoventNnuecCollectionAdmissionFailureV2 {
                error: AdaptiveManagerError::Invalid("Own deferred root has different or stale origin"), admission, owners: Box::new(()),
            });
        }
        if previous.is_none() {
            if let Some((bridge, _)) = self.latest_installed() {
                if !bridge.installed_admission_for_recovery().is_some_and(|original| Self::follows_admission(original, &admission)) {
                    return Err(RoventNnuecCollectionAdmissionFailureV2 {
                        error: AdaptiveManagerError::Invalid("Own deferred root does not follow the original installed root"), admission, owners: Box::new(bridge),
                    });
                }
            }
        }
        let original = admission.clone();
        let result = self.primary().defer_no_capture(port, admission, reason);
        if matches!(&result, Ok(RoventNnuecCollectionAdmissionOutcomeV2::DeferredNoCapture { .. })) {
            *previous = Some(original);
        }
        result
    }

    /// Supervisor calls this only with actual same-game worker owners. The
    /// previous bridge separately verifies worker join, native fence, teacher
    /// aggregate and root-specific outbox custody before releasing its gate.
    pub(crate) fn publish_next_prepared(&self, previous: &Arc<RoventNnuecCollectionBridgeV2>,
        previous_admission: &RoventCollectionRootAdmissionV2, next: &Arc<RoventNnuecCollectionBridgeV2>,
        retained_row_limit: usize) -> Result<OwnPreparedSlotOutcomeV2, ArenaError> {
        let index = self.slots.iter().position(|slot| slot.get().is_some_and(|original| Arc::ptr_eq(original, previous)))
            .ok_or_else(|| invalid("Own previous bridge is not the original slot"))?;
        if self.states[index].load(Ordering::Acquire) != INSTALLED { return Ok(OwnPreparedSlotOutcomeV2::Pending); }
        let next_index = index.checked_add(1).filter(|index| *index < self.slots.len()).ok_or_else(|| invalid("Own root slot capacity"))?;
        match self.states[next_index].compare_exchange(EMPTY, PREPARING, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {}, Err(PREPARING) => return Ok(OwnPreparedSlotOutcomeV2::Pending),
            Err(_) => return Err(invalid("Own next root slot already prepared")),
        }
        let mut guard = PreparationGuard { state: &self.states[next_index], published: false };
        if self.slots[next_index].get().is_some() { return Err(invalid("Own root slot unexpectedly occupied")); }
        match next.try_prepared_for_same_owner(previous)? {
            None => return Ok(OwnPreparedSlotOutcomeV2::Pending),
            Some(false) => return Err(invalid("Own next worker is not the original prepared same-owner worker")),
            Some(true) => {},
        }
        let retained_rows = match previous.try_separate_root_for_next(previous_admission, retained_row_limit)? {
            OwnRootSeparationV2::Pending => return Ok(OwnPreparedSlotOutcomeV2::Pending),
            OwnRootSeparationV2::Released { retained_rows } => retained_rows,
        };
        self.slots[next_index].set(Arc::clone(next)).map_err(|_| invalid("Own concurrent root slot publication"))?;
        self.states[next_index].store(READY, Ordering::Release);
        guard.published = true;
        Ok(OwnPreparedSlotOutcomeV2::Published { retained_rows })
    }
}

impl RoventNnuecCollectionHooksV2 for RoventOwnRootRouterV2 {
    fn register_startup_control(&self, control: &RoventStartupControl) -> Result<(), AdaptiveManagerError> {
        self.startup.set(control.clone()).map_err(|_| AdaptiveManagerError::Invalid("Own router startup already registered"))?;
        self.primary().register_startup_control(control)
    }

    fn admit(&self, port: &mut RoventNnuecCollectionAdmissionPortV2<'_>, admission: RoventCollectionRootAdmissionV2)
        -> Result<RoventNnuecCollectionAdmissionOutcomeV2, RoventNnuecCollectionAdmissionFailureV2> {
        for (index, (slot, state)) in self.slots.iter().zip(&self.states).enumerate() {
            let uncertain = if state.compare_exchange(UNCERTAIN, RESERVING, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                true
            } else if state.compare_exchange(READY, RESERVING, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                false
            } else { continue; };
            let mut reservation = AdmissionGuard { state, settled: false };
            if uncertain {
                let Some(bridge) = slot.get() else {
                    state.store(FAILED, Ordering::Release); reservation.settled = true;
                    return Err(RoventNnuecCollectionAdmissionFailureV2 {
                        error: AdaptiveManagerError::Invalid("Own uncertain root owner unavailable"), admission, owners: Box::new(()),
                    });
                };
                if !self.attempts[index].get().is_some_and(|original| original.same_original_admission(&admission)) {
                    state.store(UNCERTAIN, Ordering::Release); reservation.settled = true;
                    return Err(RoventNnuecCollectionAdmissionFailureV2 {
                        error: AdaptiveManagerError::Invalid("Own uncertain root admission mismatch"), admission, owners: Box::new(Arc::clone(bridge)),
                    });
                }
                let mut last = match self.last_deferred.try_lock() {
                    Ok(last) => last,
                    Err(std::sync::TryLockError::WouldBlock) => {
                        state.store(UNCERTAIN, Ordering::Release); reservation.settled = true;
                        return Ok(RoventNnuecCollectionAdmissionOutcomeV2::PendingUncertain { admission });
                    }
                    Err(std::sync::TryLockError::Poisoned(_)) => {
                        state.store(FAILED, Ordering::Release); reservation.settled = true;
                        return Err(RoventNnuecCollectionAdmissionFailureV2 {
                            error: AdaptiveManagerError::Invalid("Own deferred root custody poisoned"), admission, owners: Box::new(Arc::clone(bridge)),
                        });
                    }
                };
                let original = admission.clone();
                return match bridge.admit(port, admission) {
                    Ok(RoventNnuecCollectionAdmissionOutcomeV2::Installed) => {
                        state.store(INSTALLED, Ordering::Release); reservation.settled = true;
                        Ok(RoventNnuecCollectionAdmissionOutcomeV2::Installed)
                    }
                    Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission } | RoventNnuecCollectionAdmissionOutcomeV2::PendingUncertain { admission }) => {
                        state.store(UNCERTAIN, Ordering::Release); reservation.settled = true;
                        Ok(RoventNnuecCollectionAdmissionOutcomeV2::PendingUncertain { admission })
                    }
                    Ok(other @ RoventNnuecCollectionAdmissionOutcomeV2::DeferredNoCapture { .. }) => {
                        *last = Some(original);
                        state.store(DEFERRED, Ordering::Release); reservation.settled = true;
                        Ok(other)
                    }
                    Err(failure) => {
                        state.store(FAILED, Ordering::Release); reservation.settled = true;
                        Err(RoventNnuecCollectionAdmissionFailureV2 { error: failure.error, admission: failure.admission, owners: Box::new((Arc::clone(bridge), failure.owners)) })
                    }
                };
            }
            let _ = self.attempts[index].set(admission.clone());
            if !self.attempts[index].get().is_some_and(|original| original.same_original_admission(&admission)) {
                let original = self.attempts[index].get().expect("reserved original root");
                let Some(event) = self.stopped[index].get().filter(|event| event.admission.same_original_admission(original)) else {
                    state.store(READY, Ordering::Release); reservation.settled = true;
                    return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission });
                };
                if !Self::follows_admission(original, &admission) {
                    state.store(READY, Ordering::Release); reservation.settled = true;
                    return Err(RoventNnuecCollectionAdmissionFailureV2 {
                        error: AdaptiveManagerError::Invalid("Own root after stopped attempt has different or stale origin"), admission, owners: Box::new(Arc::clone(event)),
                    });
                }
                let Some(bridge) = slot.get() else {
                    state.store(PREINSTALL_FAILED, Ordering::Release); reservation.settled = true;
                    return Err(RoventNnuecCollectionAdmissionFailureV2 {
                        error: AdaptiveManagerError::Invalid("Own stopped attempt owner unavailable"), admission, owners: Box::new(Arc::clone(event)),
                    });
                };
                if bridge.matches_original_attempt(original) {
                    match bridge.try_seal_unstarted(original) {
                        Ok(super::collection_hooks_core_v2::CoreUnstartedSealV2::Sealed) => {},
                        Ok(super::collection_hooks_core_v2::CoreUnstartedSealV2::Busy) => {
                            state.store(READY, Ordering::Release); reservation.settled = true;
                            return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission });
                        }
                        Ok(super::collection_hooks_core_v2::CoreUnstartedSealV2::Unknown) => {
                            state.store(FAILED, Ordering::Release); reservation.settled = true;
                            return Err(RoventNnuecCollectionAdmissionFailureV2 {
                                error: AdaptiveManagerError::Invalid("Own stopped original installation is unknown"), admission, owners: Box::new(Arc::clone(bridge)),
                            });
                        }
                        Err(error) => {
                            state.store(FAILED, Ordering::Release); reservation.settled = true;
                            return Err(RoventNnuecCollectionAdmissionFailureV2 {
                                error: AdaptiveManagerError::Collection(AdaptiveCollectionError::new(AdaptiveCollectionStage::Settled, "OwnUnstartedSeal", error.to_string())), admission, owners: Box::new((Arc::clone(bridge), error)),
                            });
                        }
                    }
                }
                let outcome = self.defer_serialized(port, admission, event.reason.clone());
                match &outcome {
                    Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { .. }) => state.store(READY, Ordering::Release),
                    Ok(RoventNnuecCollectionAdmissionOutcomeV2::DeferredNoCapture { .. }) => state.store(DEFERRED, Ordering::Release),
                    _ => state.store(FAILED, Ordering::Release),
                }
                reservation.settled = true;
                return outcome;
            }
            let Some(bridge) = slot.get() else {
                state.store(PREINSTALL_FAILED, Ordering::Release); reservation.settled = true;
                return Err(RoventNnuecCollectionAdmissionFailureV2 {
                    error: AdaptiveManagerError::Invalid("Own prepared root owner disappeared"), admission, owners: Box::new(()),
                });
            };
            let previous = if index == 0 { bridge } else {
                let Some(previous) = self.slots[index - 1].get() else {
                    state.store(READY, Ordering::Release); reservation.settled = true;
                    return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission });
                };
                if !Self::follows(previous, &admission) {
                    if previous.installed_admission_for_recovery().is_some_and(|original| Self::follows_admission(original, &admission))
                        && previous.closed_before_next_admission(&admission) {
                        // The owner/writer is scoped to the previous game.
                        // A later game's root may run only with an explicit
                        // no-capture marker, never through this slot.
                        let outcome = self.defer_serialized(port, admission, AdaptiveCollectionError::new(
                            AdaptiveCollectionStage::Dispatch, "OwnCollectionDifferentGame",
                            "the original collection writer is scoped to an earlier game"));
                        match &outcome {
                            Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { .. }) => state.store(READY, Ordering::Release),
                            Ok(RoventNnuecCollectionAdmissionOutcomeV2::DeferredNoCapture { .. }) => state.store(DEFERRED, Ordering::Release),
                            _ => state.store(FAILED, Ordering::Release),
                        }
                        reservation.settled = true;
                        return outcome;
                    }
                    state.store(PREINSTALL_FAILED, Ordering::Release); reservation.settled = true;
                    return Err(RoventNnuecCollectionAdmissionFailureV2 {
                        error: AdaptiveManagerError::Invalid("Own prepared root has different or stale origin"), admission, owners: Box::new(Arc::clone(bridge)),
                    });
                }
                if !previous.closed_before_next_admission(&admission) {
                    state.store(READY, Ordering::Release); reservation.settled = true;
                    return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission });
                }
                previous
            };
            match bridge.try_prepared_for_same_owner(previous) {
                Ok(Some(true)) => {},
                Ok(None) => {
                    state.store(READY, Ordering::Release); reservation.settled = true;
                    return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission });
                }
                Ok(Some(false)) => {
                    // This router owns the reservation and has not called
                    // bridge.admit; the native marker must still be installed
                    // before independent NN execution can proceed.
                    let outcome = self.defer_serialized(port, admission, AdaptiveCollectionError::new(
                        AdaptiveCollectionStage::Dispatch, "OwnCollectionWorkerUnavailable",
                        "the original root worker is unavailable before observer installation"));
                    match &outcome {
                        Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { .. }) => state.store(READY, Ordering::Release),
                        Ok(RoventNnuecCollectionAdmissionOutcomeV2::DeferredNoCapture { .. }) => state.store(DEFERRED, Ordering::Release),
                        _ => state.store(FAILED, Ordering::Release),
                    }
                    reservation.settled = true;
                    return outcome;
                }
                Err(error) => {
                    state.store(PREINSTALL_FAILED, Ordering::Release); reservation.settled = true;
                    return Err(RoventNnuecCollectionAdmissionFailureV2 {
                        error: AdaptiveManagerError::Collection(AdaptiveCollectionError::new(AdaptiveCollectionStage::Dispatch, "OwnPreparedOwner", error.to_string())),
                        admission, owners: Box::new((Arc::clone(bridge), error)),
                    });
                }
            }
            let mut last = match self.last_deferred.try_lock() {
                Ok(last) => last,
                Err(std::sync::TryLockError::WouldBlock) => {
                    state.store(READY, Ordering::Release); reservation.settled = true;
                    return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission });
                }
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    state.store(PREINSTALL_FAILED, Ordering::Release); reservation.settled = true;
                    return Err(RoventNnuecCollectionAdmissionFailureV2 {
                        error: AdaptiveManagerError::Invalid("Own deferred root custody poisoned"), admission, owners: Box::new(Arc::clone(bridge)),
                    });
                }
            };
            if last.as_ref().is_some_and(|original| original.same_original_admission(&admission)) {
                // Another entry point already installed the no-capture marker
                // for this very root; do not start an observer afterwards.
                state.store(DEFERRED, Ordering::Release); reservation.settled = true;
                return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission });
            }
            if last.as_ref().is_some_and(|original| !original.same_original_admission(&admission) && !Self::follows_admission(original, &admission)) {
                state.store(PREINSTALL_FAILED, Ordering::Release); reservation.settled = true;
                return Err(RoventNnuecCollectionAdmissionFailureV2 {
                    error: AdaptiveManagerError::Invalid("Own root precedes the last deferred root"), admission, owners: Box::new(Arc::clone(bridge)),
                });
            }
            let original = admission.clone();
            return match bridge.admit(port, admission) {
                Ok(RoventNnuecCollectionAdmissionOutcomeV2::Installed) => {
                    state.store(INSTALLED, Ordering::Release); reservation.settled = true;
                    Ok(RoventNnuecCollectionAdmissionOutcomeV2::Installed)
                }
                Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission }) => {
                    state.store(READY, Ordering::Release); reservation.settled = true;
                    Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission })
                }
                Ok(RoventNnuecCollectionAdmissionOutcomeV2::PendingUncertain { admission }) => {
                    // Preserve the same attempt for another poll, and do not
                    // turn an uncertain install into evidence of no start.
                    state.store(UNCERTAIN, Ordering::Release); reservation.settled = true;
                    Ok(RoventNnuecCollectionAdmissionOutcomeV2::PendingUncertain { admission })
                }
                Ok(other @ RoventNnuecCollectionAdmissionOutcomeV2::DeferredNoCapture { .. }) => {
                    *last = Some(original);
                    state.store(DEFERRED, Ordering::Release); reservation.settled = true;
                    Ok(other)
                }
                Err(failure) => {
                    state.store(FAILED, Ordering::Release); reservation.settled = true;
                    Err(RoventNnuecCollectionAdmissionFailureV2 {
                        error: failure.error, admission: failure.admission,
                        owners: Box::new((Arc::clone(bridge), failure.owners)),
                    })
                }
            };
        }
        if self.states.iter().any(|state| matches!(state.load(Ordering::Acquire), READY | RESERVING | UNCERTAIN)) {
            return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission });
        }
        if self.states.iter().any(|state| state.load(Ordering::Acquire) == FAILED) {
            return Err(RoventNnuecCollectionAdmissionFailureV2 {
                error: AdaptiveManagerError::Invalid("Own earlier root installation remains unknown"), admission, owners: Box::new(()),
            });
        }
        if let Some(original) = self.attempts.iter().zip(&self.states).rev().find_map(|(attempt, state)|
            (state.load(Ordering::Acquire) == PREINSTALL_FAILED).then(|| attempt.get()).flatten()) {
            if !Self::follows_admission(original, &admission) {
                return Err(RoventNnuecCollectionAdmissionFailureV2 {
                    error: AdaptiveManagerError::Invalid("Own root after pre-install failure has different or stale origin"), admission, owners: Box::new(()),
                });
            }
            return self.defer_serialized(port, admission, AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Dispatch, "OwnCollectionPriorPreinstallFailure", "earlier collection did not enter observer installation"));
        }
        self.defer_if_closed(port, admission)
    }

    fn defer_no_capture(&self, port: &mut RoventNnuecCollectionAdmissionPortV2<'_>, admission: RoventCollectionRootAdmissionV2,
        reason: AdaptiveCollectionError) -> Result<RoventNnuecCollectionAdmissionOutcomeV2, RoventNnuecCollectionAdmissionFailureV2> {
        if self.states.iter().any(|state| state.load(Ordering::Acquire) == RESERVING) {
            return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission });
        }
        for (attempt, state) in self.attempts.iter().zip(&self.states) {
            if attempt.get().is_some_and(|original| original.same_original_admission(&admission)) {
                match state.load(Ordering::Acquire) {
                    READY | RESERVING | PREPARING => return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission }),
                    UNCERTAIN | INSTALLED | FAILED => return Err(RoventNnuecCollectionAdmissionFailureV2 {
                        error: AdaptiveManagerError::Invalid("Own root may have an installed collection owner"), admission, owners: Box::new(()),
                    }),
                    _ => {},
                }
            }
        }
        self.defer_serialized(port, admission, reason)
    }

    fn finish_deferred(&self, capability: &RoventNoCaptureRootCapabilityV2, fence: &AdaptiveShutdownReceipt)
        -> Result<AdaptiveCollectionBoundaryV2, AdaptiveManagerError> {
        // Every registered slot was checked against the same original game
        // owner. That owner verifies and removes the exact capability only.
        self.primary().finish_deferred(capability, fence)
    }

    fn try_record_pending_stopped(&self, event: &Arc<RoventNnuecPendingCollectionV2>) -> Result<RoventPendingEvidenceDispositionV2, AdaptiveManagerError> {
        for (index, attempt) in self.attempts.iter().enumerate() {
            if attempt.get().is_some_and(|original| original.same_original_admission(&event.admission)) {
                let _ = self.stopped[index].set(Arc::clone(event));
                if let Some(bridge) = self.slots[index].get() { return bridge.try_record_pending_stopped(event); }
            }
        }
        let mut found = None;
        for bridge in self.slots.iter().filter_map(OnceLock::get) {
            if bridge.matches_original_attempt(&event.admission) {
                if found.is_some() { return Err(AdaptiveManagerError::Invalid("Own pending root has multiple owners")); }
                found = Some(bridge);
            }
        }
        found.unwrap_or_else(|| self.primary()).try_record_pending_stopped(event)
    }

    fn pump(&self) -> Result<(), AdaptiveManagerError> { self.primary().pump() }
    fn revoke(&self) {
        for bridge in self.slots.iter().filter_map(OnceLock::get) { bridge.revoke(); }
    }
    fn handoff_sender(&self) -> mpsc::SyncSender<Arc<Mutex<RoventNnuecCollectionHandoffV2>>> {
        self.primary().handoff_sender()
    }
    fn pump_root(&self, admission: &RoventCollectionRootAdmissionV2) -> Result<(), RoventNnuecRootHookFailureV2> {
        self.original_slot(admission)?.pump_root(admission)
    }
    fn revoke_root(&self, admission: &RoventCollectionRootAdmissionV2) -> Result<(), RoventNnuecRootHookFailureV2> {
        for (attempt, state) in self.attempts.iter().zip(&self.states) {
            if state.load(Ordering::Acquire) == PREINSTALL_FAILED
                && attempt.get().is_some_and(|original| original.same_original_admission(admission)) {
                // This router itself proved that bridge.admit was never
                // entered for the original root. The separately prepared
                // worker's own lifetime is still supervised independently.
                return Ok(());
            }
        }
        self.original_slot(admission)?.revoke_root(admission)
    }
    fn handoff_sender_root(&self, admission: &RoventCollectionRootAdmissionV2)
        -> Result<mpsc::SyncSender<Arc<Mutex<RoventNnuecCollectionHandoffV2>>>, RoventNnuecRootHookFailureV2> {
        self.original_slot(admission)?.handoff_sender_root(admission)
    }
}
