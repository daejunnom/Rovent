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
    RoventPcDeferredCleanupV2,
}};
use std::sync::{Arc, Mutex, OnceLock, mpsc, atomic::{AtomicU8, Ordering}};

const SLOT_UNPREPARED: u8 = 0;
const SLOT_READY: u8 = 1;
const SLOT_RESERVED: u8 = 2;
const SLOT_INSTALLED: u8 = 3;
const SLOT_FAILED: u8 = 4;

pub(crate) enum PcPreparedSlotOutcomeV2 {
    Pending,
    Published { retained_rows: usize },
}

pub(crate) struct RoventPcRootRouterV2 {
    slots: Vec<OnceLock<Arc<RoventPcCollectionBridgeV2>>>,
    states: Vec<AtomicU8>,
}

impl RoventPcRootRouterV2 {
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
        Ok(Self { slots, states })
    }

    fn primary(&self) -> &Arc<RoventPcCollectionBridgeV2> {
        self.slots[0].get().expect("primary root slot set at construction")
    }

    fn defer_if_closed(&self, port: &mut RoventPcCollectionAdmissionPortV2<'_>, admission: RoventCollectionRootAdmissionV2)
        -> Result<RoventPcCollectionAdmissionOutcomeV2, RoventPcCollectionAdmissionFailureV2> {
        let previous = self.slots.iter().zip(&self.states).rev()
            .find_map(|(slot, state)| (state.load(Ordering::Acquire) == SLOT_INSTALLED).then(|| slot.get()).flatten());
        let Some(previous) = previous else { return Ok(RoventPcCollectionAdmissionOutcomeV2::Pending { admission }); };
        if !previous.closed_before_next_admission(&admission) {
            return Err(RoventPcCollectionAdmissionFailureV2 {
                error: AdaptiveManagerError::Invalid("P/C next root does not follow the original closed root"),
                admission, owners: Box::new(()),
            });
        }
        self.primary().defer_before_start(port, admission, AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Dispatch, "PcCollectionCapacity",
            "The next root worker or the shared game owner is not ready; this root is not captured",
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

    /// Supervisor-only transition, performed outside the manager. The original
    /// previous bridge proves handoff, outbox separation and game-gate
    /// release itself; the next bridge must already own a live same-owner
    /// worker. Both input Arcs remain with the caller on every failure.
    pub(crate) fn publish_next_prepared(
        &self,
        previous: &Arc<RoventPcCollectionBridgeV2>,
        previous_admission: &RoventCollectionRootAdmissionV2,
        next: &Arc<RoventPcCollectionBridgeV2>,
        game_row_capacity: usize,
    ) -> Result<PcPreparedSlotOutcomeV2, super::ArenaError> {
        let previous_index = self.slots.iter().position(|slot| slot.get().is_some_and(|value| Arc::ptr_eq(value, previous)))
            .ok_or_else(|| super::invalid("P/C previous prepared slot is not the original bridge"))?;
        if self.states[previous_index].load(Ordering::Acquire) != SLOT_INSTALLED {
            return Ok(PcPreparedSlotOutcomeV2::Pending);
        }
        let index = previous_index.checked_add(1).filter(|index| *index < self.slots.len())
            .ok_or_else(|| super::invalid("P/C root slot capacity"))?;
        if self.slots[index].get().is_some() || self.states[index].load(Ordering::Acquire) != SLOT_UNPREPARED {
            return Err(super::invalid("P/C next root slot already prepared"));
        }
        match next.try_prepared_for_same_owner(previous)? {
            None => return Ok(PcPreparedSlotOutcomeV2::Pending),
            Some(false) => return Err(super::invalid("P/C next worker is not the original prepared same-owner worker")),
            Some(true) => {}
        }
        let retained_rows = match previous.try_separate_root_for_next(previous_admission, game_row_capacity)? {
            PcRootSeparationV2::Pending => return Ok(PcPreparedSlotOutcomeV2::Pending),
            PcRootSeparationV2::Released { retained_rows } => retained_rows,
        };
        self.slots[index].set(Arc::clone(next)).map_err(|_| super::invalid("P/C concurrent next slot publication"))?;
        self.states[index].store(SLOT_READY, Ordering::Release);
        Ok(PcPreparedSlotOutcomeV2::Published { retained_rows })
    }
}

impl RoventPcCollectionHooksV2 for RoventPcRootRouterV2 {
    fn register_startup_control(&self, control: &RoventStartupControl) -> Result<(), AdaptiveManagerError> {
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
            if index > 0 && !self.slots[index - 1].get()
                .is_some_and(|previous| previous.closed_before_next_admission(&admission)) {
                state.store(SLOT_READY, Ordering::Release);
                return Err(RoventPcCollectionAdmissionFailureV2 {
                    error: AdaptiveManagerError::Invalid("P/C prepared root has a different or stale origin"), admission, owners: Box::new(()),
                });
            }
            return match bridge.admit(port, admission) {
                Ok(RoventPcCollectionAdmissionOutcomeV2::Installed) => {
                    state.store(SLOT_INSTALLED, Ordering::Release);
                    Ok(RoventPcCollectionAdmissionOutcomeV2::Installed)
                }
                Ok(RoventPcCollectionAdmissionOutcomeV2::Pending { admission }) => {
                    state.store(SLOT_READY, Ordering::Release);
                    self.defer_if_closed(port, admission)
                }
                Ok(other @ RoventPcCollectionAdmissionOutcomeV2::DeferredNoCapture { .. }) => {
                    state.store(SLOT_FAILED, Ordering::Release);
                    Ok(other)
                }
                Err(failure) => {
                    state.store(SLOT_FAILED, Ordering::Release);
                    Err(failure)
                }
            };
        }
        self.defer_if_closed(port, admission)
    }

    fn finish_deferred(&self, capability: &RoventNoCaptureRootCapabilityV2, fence: &RoleOnlyShutdownReceipt)
        -> Result<AdaptiveCollectionBoundaryV2, AdaptiveManagerError> {
        self.primary().finish_deferred(capability, fence)
    }

    fn retain_deferred_cleanup(&self, cleanup: Arc<Mutex<RoventPcDeferredCleanupV2>>)
        -> Result<RoventPcCustodyReturnV2, AdaptiveManagerError> {
        self.primary().retain_deferred_cleanup(cleanup)
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
