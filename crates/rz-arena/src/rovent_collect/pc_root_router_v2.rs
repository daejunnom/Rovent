//! Routes each P-C Collection callback to its original root owner.
//!
//! The first slot wraps the existing single-root bridge. Extra slots are
//! deliberately inert until the supervisor can prove the shared game owner,
//! previous root outbox and admission/commit gate are ready. A slot index or
//! a copied digest never grants that authority.
use super::pc_hooks_lock_v2::{RoventPcCollectionBridgeV2, PcRootSeparationV2};
use rz_uci::pals_adaptive::{AdaptiveCollectionBoundaryV2, AdaptiveCollectionError, AdaptiveCollectionStage, AdaptiveManagerError, RoleOnlyShutdownReceipt, RoventCollectionRootAdmissionV2, RoventNoCaptureRootCapabilityV2};
use rz_uci::rovent_bootstrap::{RoventStartupControl, pc_session::{
    RoventPcCollectionAdmissionFailureV2, RoventPcCollectionAdmissionOutcomeV2,
    RoventPcCollectionAdmissionPortV2, RoventPcCollectionHandoffV2,
    RoventPcCollectionHooksV2, RoventPcCollectionRootCustodyV2, RoventPcCustodyReturnV2,
    RoventPcDeferredCleanupV2, RoventPcRevokeObservationV2,
}};
use std::sync::{Arc, Mutex, OnceLock, mpsc, atomic::{AtomicU8, Ordering}};

const SLOT_UNPREPARED: u8 = 0;
const SLOT_READY: u8 = 1;
const SLOT_RESERVED: u8 = 2;
const SLOT_INSTALLED: u8 = 3;
const SLOT_FAILED: u8 = 4;
// This slot was proven not to have installed an observer for the root.
const SLOT_UNAVAILABLE_BEFORE_INSTALL: u8 = 5;
const SLOT_PREPARING: u8 = 6;

struct SlotPreparationGuard<'a> {
    state: &'a AtomicU8,
    published: bool,
}
impl Drop for SlotPreparationGuard<'_> {
    fn drop(&mut self) {
        if !self.published {
            let _ = self.state.compare_exchange(SLOT_PREPARING, SLOT_UNPREPARED, Ordering::AcqRel, Ordering::Acquire);
        }
    }
}

pub(crate) enum PcPreparedSlotOutcomeV2 {
    Pending,
    Published { retained_rows: usize },
}

pub(crate) struct RoventPcRootRouterV2 {
    slots: Vec<OnceLock<Arc<RoventPcCollectionBridgeV2>>>,
    states: Vec<AtomicU8>,
    startup: OnceLock<RoventStartupControl>,
}

impl RoventPcRootRouterV2 {
    fn follows(previous: &RoventPcCollectionBridgeV2, next: &RoventCollectionRootAdmissionV2) -> bool {
        let Some(original) = previous.installed_admission_for_recovery() else { return false; };
        let old = original.root_admission().authority();
        let new = next.root_admission().authority();
        !original.same_original_admission(next)
            && original.native_origin().same_original_origin(next.native_origin())
            && old.epoch == new.epoch && old.implementation == new.implementation
            && (new.game > old.game || (new.game == old.game && new.root > old.root))
    }
    /// Call before entering the native manager. No worker is spawned here.
    pub(crate) fn new(primary: Arc<RoventPcCollectionBridgeV2>, maximum_roots: usize)
        -> Result<Self, super::ArenaError> {
        if !(1..=64).contains(&maximum_roots) {
            return Err(super::invalid("P/C root router capacity"));
        }
        let mut slots = Vec::new();
        slots.try_reserve_exact(maximum_roots).map_err(|_| super::invalid("P/C root router allocation"))?;
        slots.resize_with(maximum_roots, OnceLock::new);
        slots[0].set(primary).map_err(|_| super::invalid("P/C primary root slot already assigned"))?;
        let mut states = Vec::new();
        states.try_reserve_exact(maximum_roots).map_err(|_| super::invalid("P/C root router state allocation"))?;
        states.resize_with(maximum_roots, || AtomicU8::new(SLOT_UNPREPARED));
        states[0].store(SLOT_READY, Ordering::Release);
        Ok(Self { slots, states, startup: OnceLock::new() })
    }

    fn primary(&self) -> &Arc<RoventPcCollectionBridgeV2> {
        self.slots[0].get().expect("primary root slot set at construction")
    }

    /// The original control is registered by UCI before resource admission.
    /// Looking at it or cloning it does not start a worker or authorize a root.
    pub(crate) fn original_startup(&self) -> Option<&RoventStartupControl> {
        self.startup.get()
    }

    /// Nonblocking view of the last actually installed original root. The
    /// returned Arc and admission remain owned by the router as well.
    pub(crate) fn latest_installed(&self) -> Option<(Arc<RoventPcCollectionBridgeV2>, RoventCollectionRootAdmissionV2)> {
        self.slots.iter().zip(&self.states).rev().find_map(|(slot, state)| {
            if state.load(Ordering::Acquire) != SLOT_INSTALLED { return None; }
            let bridge = slot.get()?;
            Some((Arc::clone(bridge), bridge.installed_admission_for_recovery()?.clone()))
        })
    }

    /// A preparation hint only; publication still reserves the slot and
    /// validates the previous original root and the new worker itself.
    pub(crate) fn next_slot_unprepared(&self) -> bool {
        let Some(index) = self.states.iter().rposition(|state| state.load(Ordering::Acquire) == SLOT_INSTALLED) else { return false; };
        self.states.get(index + 1).is_some_and(|state| state.load(Ordering::Acquire) == SLOT_UNPREPARED)
    }

    fn defer_if_closed(&self, port: &mut RoventPcCollectionAdmissionPortV2<'_>, admission: RoventCollectionRootAdmissionV2)
        -> Result<RoventPcCollectionAdmissionOutcomeV2, RoventPcCollectionAdmissionFailureV2> {
        let previous = self.slots.iter().zip(&self.states).rev()
            .find_map(|(slot, state)| (state.load(Ordering::Acquire) == SLOT_INSTALLED).then(|| slot.get()).flatten());
        let Some(previous) = previous else { return Ok(RoventPcCollectionAdmissionOutcomeV2::Pending { admission }); };
        if !Self::follows(previous, &admission) {
            return Err(RoventPcCollectionAdmissionFailureV2 {
                error: AdaptiveManagerError::Invalid("P/C next root does not follow the original root"),
                admission, owners: Box::new(()),
            });
        }
        if !previous.closed_before_next_admission(&admission) {
            return Ok(RoventPcCollectionAdmissionOutcomeV2::Pending { admission });
        }
        self.primary().defer_before_start(port, admission, AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Dispatch, "PcCollectionCapacity",
            "The next root worker or the shared game owner is not ready; this root is not captured",
        ))
    }

    fn defer_unstarted(&self, port: &mut RoventPcCollectionAdmissionPortV2<'_>, admission: RoventCollectionRootAdmissionV2)
        -> Result<RoventPcCollectionAdmissionOutcomeV2, RoventPcCollectionAdmissionFailureV2> {
        // An uninstalled first slot has not handed the current root to an
        // observer. Later roots additionally require original-order closure.
        if self.states.iter().any(|state| state.load(Ordering::Acquire) == SLOT_INSTALLED) {
            return self.defer_if_closed(port, admission);
        }
        self.primary().defer_before_start(port, admission, AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Dispatch, "PcCollectionWorkerUnavailable",
            "The original collection worker is unavailable before this root's observer installation",
        ))
    }

    fn original_slot(&self, root: &RoventCollectionRootAdmissionV2)
        -> Result<&Arc<RoventPcCollectionBridgeV2>, AdaptiveManagerError> {
        let mut matched = None;
        for slot in &self.slots {
            let Some(bridge) = slot.get() else { continue; };
            if bridge.installed_admission_for_recovery()
                .is_some_and(|original| original.same_original_admission(root)) {
                if matched.is_some() {
                    return Err(AdaptiveManagerError::Invalid("P/C root assigned to multiple slots"));
                }
                matched = Some(bridge);
            }
        }
        matched.ok_or(AdaptiveManagerError::Invalid("P/C original root slot unavailable"))
    }

    fn deferred_owner(&self, capability: &RoventNoCaptureRootCapabilityV2)
        -> Result<Option<&Arc<RoventPcCollectionBridgeV2>>, AdaptiveManagerError> {
        let mut matched = None;
        for slot in &self.slots {
            let Some(bridge) = slot.get() else { continue; };
            match bridge.try_owns_deferred(capability)? {
                None => return Ok(None),
                Some(false) => {}
                Some(true) if matched.is_some() => return Err(AdaptiveManagerError::Invalid("P/C deferred root has multiple original owners")),
                Some(true) => matched = Some(bridge),
            }
        }
        matched.map(Some).ok_or(AdaptiveManagerError::Invalid("P/C deferred root original owner unavailable"))
    }

    /// Supervisor-only transition, performed outside the manager. The original
    /// previous bridge proves handoff, outbox separation and game-gate
    /// release itself; the next bridge must already own a live same-owner
    /// worker. Both input Arcs remain with the caller on every failure.
    pub(crate) fn publish_next_prepared(
        &self,
        previous: &Arc<RoventPcCollectionBridgeV2>,
        previous_admission: &RoventCollectionRootAdmissionV2,
        next: &Arc<RoventPcCollectionBridgeV2>,
        retained_row_limit: usize,
    ) -> Result<PcPreparedSlotOutcomeV2, super::ArenaError> {
        let previous_index = self.slots.iter().position(|slot| slot.get().is_some_and(|value| Arc::ptr_eq(value, previous)))
            .ok_or_else(|| super::invalid("P/C previous prepared slot is not the original bridge"))?;
        if self.states[previous_index].load(Ordering::Acquire) != SLOT_INSTALLED {
            return Ok(PcPreparedSlotOutcomeV2::Pending);
        }
        let index = previous_index.checked_add(1).filter(|index| *index < self.slots.len())
            .ok_or_else(|| super::invalid("P/C root slot capacity"))?;
        match self.states[index].compare_exchange(SLOT_UNPREPARED, SLOT_PREPARING, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {}
            Err(SLOT_PREPARING) => return Ok(PcPreparedSlotOutcomeV2::Pending),
            Err(_) => return Err(super::invalid("P/C next root slot already prepared")),
        }
        let mut reservation = SlotPreparationGuard { state: &self.states[index], published: false };
        if self.slots[index].get().is_some() { return Err(super::invalid("P/C next root slot unexpectedly occupied")); }
        match next.try_prepared_for_same_owner(previous)? {
            None => return Ok(PcPreparedSlotOutcomeV2::Pending),
            Some(false) => return Err(super::invalid("P/C next worker is not the original prepared same-owner worker")),
            Some(true) => {}
        }
        let retained_rows = match previous.try_separate_root_for_next(previous_admission, retained_row_limit)? {
            PcRootSeparationV2::Pending => return Ok(PcPreparedSlotOutcomeV2::Pending),
            PcRootSeparationV2::Released { retained_rows } => retained_rows,
        };
        self.slots[index].set(Arc::clone(next)).map_err(|_| super::invalid("P/C concurrent next slot publication"))?;
        self.states[index].store(SLOT_READY, Ordering::Release);
        reservation.published = true;
        Ok(PcPreparedSlotOutcomeV2::Published { retained_rows })
    }
}

impl RoventPcCollectionHooksV2 for RoventPcRootRouterV2 {
    fn register_startup_control(&self, control: &RoventStartupControl) -> Result<(), AdaptiveManagerError> {
        self.startup.set(control.clone()).map_err(|_| AdaptiveManagerError::Invalid("P/C router startup already registered"))?;
        self.primary().register_startup_control(control)
    }

    fn admit(&self, port: &mut RoventPcCollectionAdmissionPortV2<'_>, admission: RoventCollectionRootAdmissionV2)
        -> Result<RoventPcCollectionAdmissionOutcomeV2, RoventPcCollectionAdmissionFailureV2> {
        for (index, (slot, state)) in self.slots.iter().zip(&self.states).enumerate() {
            if state.compare_exchange(SLOT_READY, SLOT_RESERVED, Ordering::AcqRel, Ordering::Acquire).is_err() { continue; }
            let Some(bridge) = slot.get() else {
                state.store(SLOT_FAILED, Ordering::Release);
                return Err(RoventPcCollectionAdmissionFailureV2 {
                    error: AdaptiveManagerError::Invalid("P/C prepared root slot disappeared"), admission, owners: Box::new(()),
                });
            };
            if index > 0 {
                let Some(previous) = self.slots[index - 1].get() else {
                    state.store(SLOT_READY, Ordering::Release);
                    return Ok(RoventPcCollectionAdmissionOutcomeV2::Pending { admission });
                };
                if !Self::follows(previous, &admission) {
                    state.store(SLOT_READY, Ordering::Release);
                    return Err(RoventPcCollectionAdmissionFailureV2 {
                        error: AdaptiveManagerError::Invalid("P/C prepared root has a different or stale origin"), admission, owners: Box::new(()),
                    });
                }
                if !previous.closed_before_next_admission(&admission) {
                    state.store(SLOT_READY, Ordering::Release);
                    return Ok(RoventPcCollectionAdmissionOutcomeV2::Pending { admission });
                }
            }
            let previous = if index == 0 { bridge } else { self.slots[index - 1].get().expect("previous slot checked") };
            match bridge.try_prepared_for_same_owner(previous) {
                Ok(Some(true)) => {}
                Ok(Some(false)) => {
                    state.store(SLOT_UNAVAILABLE_BEFORE_INSTALL, Ordering::Release);
                    return self.defer_unstarted(port, admission);
                }
                Ok(None) => {
                    state.store(SLOT_READY, Ordering::Release);
                    return self.defer_unstarted(port, admission);
                }
                Err(error) => {
                    state.store(SLOT_FAILED, Ordering::Release);
                    return Err(RoventPcCollectionAdmissionFailureV2 {
                        error: AdaptiveManagerError::Collection(AdaptiveCollectionError::new(
                            AdaptiveCollectionStage::Dispatch, "PcPreparedOwner", error.to_string())),
                        admission, owners: Box::new(Arc::clone(bridge)),
                    });
                }
            }
            return match bridge.admit(port, admission) {
                Ok(RoventPcCollectionAdmissionOutcomeV2::Installed) => {
                    state.store(SLOT_INSTALLED, Ordering::Release);
                    Ok(RoventPcCollectionAdmissionOutcomeV2::Installed)
                }
                Ok(RoventPcCollectionAdmissionOutcomeV2::Pending { admission }) => {
                    state.store(SLOT_READY, Ordering::Release);
                    // The bridge may be waiting on the original attempt lock;
                    // Pending does not establish that installation did not start.
                    Ok(RoventPcCollectionAdmissionOutcomeV2::Pending { admission })
                }
                Ok(other @ RoventPcCollectionAdmissionOutcomeV2::DeferredNoCapture { .. }) => {
                    state.store(SLOT_UNAVAILABLE_BEFORE_INSTALL, Ordering::Release);
                    Ok(other)
                }
                Err(failure) => {
                    state.store(SLOT_FAILED, Ordering::Release);
                    Err(failure)
                }
            };
        }
        // A first pre-install worker may have ended, leaving no installed
        // predecessor. The original marker path can still explicitly defer
        // later roots. A concurrently reserved slot is not proof of no-start.
        if self.states.iter().any(|state| matches!(state.load(Ordering::Acquire), SLOT_READY | SLOT_RESERVED)) {
            return Ok(RoventPcCollectionAdmissionOutcomeV2::Pending { admission });
        }
        if self.states.iter().any(|state| state.load(Ordering::Acquire) == SLOT_FAILED) {
            return Err(RoventPcCollectionAdmissionFailureV2 {
                error: AdaptiveManagerError::Invalid("P/C earlier root slot failed after installation became unknown"),
                admission, owners: Box::new(()),
            });
        }
        if !self.states.iter().any(|state| state.load(Ordering::Acquire) == SLOT_INSTALLED)
            && self.states.iter().any(|state| state.load(Ordering::Acquire) == SLOT_UNAVAILABLE_BEFORE_INSTALL) {
            return self.defer_unstarted(port, admission);
        }
        self.defer_if_closed(port, admission)
    }

    fn finish_deferred(&self, capability: &RoventNoCaptureRootCapabilityV2, fence: &RoleOnlyShutdownReceipt)
        -> Result<AdaptiveCollectionBoundaryV2, AdaptiveManagerError> {
        let Some(owner) = self.deferred_owner(capability)? else { return Ok(AdaptiveCollectionBoundaryV2::Busy); };
        owner.finish_deferred(capability, fence)
    }

    fn retain_deferred_cleanup(&self, cleanup: Arc<Mutex<RoventPcDeferredCleanupV2>>)
        -> Result<RoventPcCustodyReturnV2, AdaptiveManagerError> {
        let capability = match cleanup.try_lock() {
            Ok(original) => original.capability().clone(),
            Err(std::sync::TryLockError::WouldBlock) => return Ok(RoventPcCustodyReturnV2::Busy),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("P/C deferred cleanup owner poisoned")),
        };
        let Some(owner) = self.deferred_owner(&capability)? else { return Ok(RoventPcCustodyReturnV2::Busy); };
        owner.retain_deferred_cleanup(cleanup)
    }

    fn pump(&self) -> Result<(), AdaptiveManagerError> { self.primary().pump() }
    fn revoke(&self) { self.primary().revoke() }
    fn handoff_sender(&self) -> mpsc::SyncSender<Arc<Mutex<RoventPcCollectionHandoffV2>>> {
        self.primary().handoff_sender()
    }

    fn pump_root(&self, root: &RoventCollectionRootAdmissionV2) -> Result<(), AdaptiveManagerError> {
        self.original_slot(root)?.pump_root(root)
    }
    fn revoke_root(&self, root: &RoventCollectionRootAdmissionV2) -> Result<(), AdaptiveManagerError> {
        self.original_slot(root)?.revoke_root(root)
    }
    fn revoke_root_observed(&self, root: &RoventCollectionRootAdmissionV2, observation: RoventPcRevokeObservationV2)
        -> Result<(), AdaptiveManagerError> {
        self.original_slot(root)?.revoke_root_observed(root, observation)
    }
    fn handoff_sender_root(&self, root: &RoventCollectionRootAdmissionV2)
        -> Result<mpsc::SyncSender<Arc<Mutex<RoventPcCollectionHandoffV2>>>, AdaptiveManagerError> {
        self.original_slot(root)?.handoff_sender_root(root)
    }

    fn try_take_root_custody(&self, handoff: &Arc<Mutex<RoventPcCollectionHandoffV2>>)
        -> Result<Option<RoventPcCollectionRootCustodyV2>, AdaptiveManagerError> {
        let root = match handoff.try_lock() {
            Ok(mut original) => original.parts_mut().0.clone(),
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("P/C root handoff poisoned")),
        };
        self.original_slot(&root)?.try_take_root_custody(handoff)
    }

    fn return_root_custody(&self, custody: Arc<Mutex<RoventPcCollectionRootCustodyV2>>)
        -> Result<RoventPcCustodyReturnV2, AdaptiveManagerError> {
        let handoff = match custody.try_lock() {
            Ok(original) => Arc::clone(original.handoff()),
            Err(std::sync::TryLockError::WouldBlock) => return Ok(RoventPcCustodyReturnV2::Busy),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("P/C root custody poisoned")),
        };
        let root = match handoff.try_lock() {
            Ok(mut original) => original.parts_mut().0.clone(),
            Err(std::sync::TryLockError::WouldBlock) => return Ok(RoventPcCustodyReturnV2::Busy),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("P/C root handoff poisoned")),
        };
        self.original_slot(&root)?.return_root_custody(custody)
    }
}
