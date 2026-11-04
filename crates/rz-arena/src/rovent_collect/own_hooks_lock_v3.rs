//! Own-board /3 collection bridge. Manager callbacks only publish bounded
//! requests, install a prepared observer, or observe retained custody.
//!
//! An external supervisor owns preparation, teacher processes, worker joins,
//! and publication. This bridge does not attest their successful completion.

use super::native_capture::RoventNativeCaptureRegistration;
use super::observer_lock_v3::RoventCollectionOwnerLockV3;
use super::speculative_route_v3::SpeculativeRootV3;
use rz_search::pals::engine::adaptive_port::AdaptivePortQuestion;
use rz_uci::pals_adaptive::{
    AdaptiveCollectionBoundaryV2, AdaptiveCollectionObserverBoxV3,
    AdaptiveCollectionV2ObserverStatus, AdaptiveManagerError,
    RoventBoardCollectionAdmissionFailureV3, RoventBoardCollectionAdmissionOutcomeV3,
    RoventBoardCollectionAdmissionPortV3, RoventBoardCollectionHandoffV3,
    RoventBoardCollectionHooksV3, RoventBoardCollectionRootCustodyV3,
    RoventBoardCollectionWorkerJoinFailureV3, RoventBoardCollectionWorkerKeyV3,
    RoventBoardCollectionWorkerV3, RoventBoardCustodyReturnV3, RoventBoardDeferredCleanupV3,
    RoventCollectionRootAdmissionV3, RoventNoCaptureRootCapabilityV3, RoventOwnNoCaptureFenceV3,
};
use rz_uci::pals_native::ticketed::VirtualReviewTicketOutput;
use rz_uci::rovent_bootstrap::{RoventNativeBackend, RoventStartupControl};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, TryLockError, mpsc};

type Output = VirtualReviewTicketOutput;
type Backend = RoventNativeBackend<AdaptivePortQuestion>;
type Observer = AdaptiveCollectionObserverBoxV3<AdaptivePortQuestion, Output>;
type Handoff = Arc<Mutex<RoventBoardCollectionHandoffV3<Output>>>;
type Custody = RoventBoardCollectionRootCustodyV3<Output>;

struct RootSlot {
    admission: RoventCollectionRootAdmissionV3,
    requested: bool,
    revoked: Arc<AtomicBool>,
    installed: bool,
    owner: Option<RoventCollectionOwnerLockV3>,
    observer: Option<Observer>,
    route: Option<SpeculativeRootV3>,
    worker: Option<RoventBoardCollectionWorkerV3>,
    worker_key: Option<RoventBoardCollectionWorkerKeyV3>,
    handoff: Option<Handoff>,
    joined: Option<Custody>,
    returned: Option<Arc<Mutex<Custody>>>,
    join_failure: Option<RoventBoardCollectionWorkerJoinFailureV3>,
    join_panic: Option<Box<dyn std::any::Any + Send>>,
    join_unknown: bool,
}

struct State {
    slots: Vec<RootSlot>,
    deferred: Vec<Arc<Mutex<RoventBoardDeferredCleanupV3>>>,
    maximum_roots: usize,
}

/// Both this hook and its external supervisor retain this same Arc. The
/// supervisor is responsible for settling or retaining every root worker.
pub(crate) struct RoventOwnCollectionBridgeV3 {
    registration: RoventNativeCaptureRegistration,
    startup: OnceLock<RoventStartupControl>,
    state: Mutex<State>,
    request_sender: mpsc::SyncSender<OwnRootRequestV3>,
    request_receiver: Mutex<mpsc::Receiver<OwnRootRequestV3>>,
}

/// Carries the same root's irrevocable cancellation observation to the
/// supervisor. It must be checked before each acquisition or worker start.
pub(crate) struct OwnRootRequestV3 {
    pub admission: RoventCollectionRootAdmissionV3,
    revoked: Arc<AtomicBool>,
}
impl OwnRootRequestV3 {
    pub(crate) fn is_revoked(&self) -> bool {
        self.revoked.load(Ordering::Acquire)
    }
}

/// A rejected preparation returns every original object to its supervisor.
pub(crate) struct OwnPreparedRootV3 {
    pub admission: RoventCollectionRootAdmissionV3,
    pub owner: RoventCollectionOwnerLockV3,
    pub route: SpeculativeRootV3,
    pub worker: RoventBoardCollectionWorkerV3,
}
pub(crate) enum OwnPrepareOutcomeV3 {
    Prepared,
    Busy(OwnPreparedRootV3),
    Rejected {
        reason: String,
        original: OwnPreparedRootV3,
    },
    Panicked {
        original: OwnPreparedRootV3,
        payload: Box<dyn std::any::Any + Send>,
    },
}

impl RoventOwnCollectionBridgeV3 {
    /// Called before native startup. No worker or teacher is started here.
    pub(crate) fn new(
        registration: RoventNativeCaptureRegistration,
        maximum_roots: usize,
    ) -> Result<Arc<Self>, &'static str> {
        if !(1..=64).contains(&maximum_roots) {
            return Err("Own /3 root capacity");
        }
        let mut slots = Vec::new();
        let mut deferred = Vec::new();
        slots
            .try_reserve_exact(maximum_roots)
            .map_err(|_| "Own /3 root allocation")?;
        deferred
            .try_reserve_exact(maximum_roots)
            .map_err(|_| "Own /3 cleanup allocation")?;
        let (request_sender, request_receiver) = mpsc::sync_channel(maximum_roots);
        Ok(Arc::new(Self {
            registration,
            startup: OnceLock::new(),
            state: Mutex::new(State {
                slots,
                deferred,
                maximum_roots,
            }),
            request_sender,
            request_receiver: Mutex::new(request_receiver),
        }))
    }

    /// Only the supervisor calls this; a request is the exact original root,
    /// not permission to renew its deadline or synthesize a replacement.
    pub(crate) fn try_next_root(&self) -> Result<Option<OwnRootRequestV3>, &'static str> {
        let receiver = self
            .request_receiver
            .try_lock()
            .map_err(|_| "Own /3 request receiver unavailable")?;
        match receiver.try_recv() {
            Ok(value) => Ok(Some(value)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err("Own /3 request channel disconnected"),
        }
    }

    pub(crate) fn startup(&self) -> Option<&RoventStartupControl> {
        self.startup.get()
    }

    /// The caller creates the actual worker and teacher owner outside the
    /// manager. Rejection keeps the whole prepared owner with the caller.
    pub(crate) fn try_publish_prepared(&self, original: OwnPreparedRootV3) -> OwnPrepareOutcomeV3 {
        match original.admission.root_admission().try_accepts_original() {
            Ok(true) => {}
            Ok(false) => {
                return OwnPrepareOutcomeV3::Rejected {
                    reason: "Own /3 original root already expired or revoked".into(),
                    original,
                };
            }
            Err(error) if error.code == rz_contracts::ErrorCode::ResourceExhausted => {
                return OwnPrepareOutcomeV3::Busy(original);
            }
            Err(error) => {
                return OwnPrepareOutcomeV3::Rejected {
                    reason: error.to_string(),
                    original,
                };
            }
        }
        if !original.worker.matches_admission(&original.admission)
            || !original.route.matches_admission(&original.admission)
            || !self.registration.owns_speculative_root_v3(&original.route)
            || !original
                .owner
                .capture_registration()
                .same_original_owner(&self.registration)
        {
            return OwnPrepareOutcomeV3::Rejected {
                reason: "Own /3 prepared original owner mismatch".into(),
                original,
            };
        }
        let attached = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            original
                .owner
                .try_attach_speculative_root(&original.admission, &original.route)
        }));
        match attached {
            Ok(Ok(Some(()))) => {}
            Ok(Ok(None)) => return OwnPrepareOutcomeV3::Busy(original),
            Ok(Err(error)) => {
                return OwnPrepareOutcomeV3::Rejected {
                    reason: error.to_string(),
                    original,
                };
            }
            Err(payload) => return OwnPrepareOutcomeV3::Panicked { original, payload },
        }
        // Replay, Rules reconstruction and producer allocation happen in the
        // supervisor, never in the manager's observer admission callback.
        let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            original
                .owner
                .try_prepare_original_admission(&original.admission)
        }));
        match prepared {
            Ok(Ok(Some(()))) => {}
            Ok(Ok(None)) => return OwnPrepareOutcomeV3::Busy(original),
            Ok(Err(error)) => {
                return OwnPrepareOutcomeV3::Rejected {
                    reason: error.to_string(),
                    original,
                };
            }
            Err(payload) => return OwnPrepareOutcomeV3::Panicked { original, payload },
        }
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return OwnPrepareOutcomeV3::Busy(original),
            Err(TryLockError::Poisoned(_)) => {
                return OwnPrepareOutcomeV3::Rejected {
                    reason: "Own /3 root custody poisoned".into(),
                    original,
                };
            }
        };
        let Some(slot) = state
            .slots
            .iter_mut()
            .find(|slot| slot.admission.same_original_admission(&original.admission))
        else {
            return OwnPrepareOutcomeV3::Rejected {
                reason: "Own /3 original request absent".into(),
                original,
            };
        };
        if !slot.requested
            || slot.revoked.load(Ordering::Acquire)
            || slot.installed
            || slot.worker.is_some()
            || slot.observer.is_some()
        {
            return OwnPrepareOutcomeV3::Rejected {
                reason: "Own /3 root already prepared".into(),
                original,
            };
        }
        let observer = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            original.owner.observer::<AdaptivePortQuestion, Output>()
        })) {
            Ok(observer) => observer,
            Err(payload) => return OwnPrepareOutcomeV3::Panicked { original, payload },
        };
        let OwnPreparedRootV3 {
            admission: _,
            owner,
            route,
            worker,
        } = original;
        slot.observer = Some(observer);
        slot.owner = Some(owner);
        slot.route = Some(route);
        slot.worker_key = Some(worker.recovery_key());
        slot.worker = Some(worker);
        OwnPrepareOutcomeV3::Prepared
    }

    /// Called outside the manager: a successful OS join remains separate from
    /// the native root fence and from the teacher aggregate in the handoff.
    pub(crate) fn try_join_root(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<bool, &'static str> {
        let mut state = self
            .state
            .try_lock()
            .map_err(|_| "Own /3 root custody busy")?;
        let slot = state
            .slots
            .iter_mut()
            .find(|slot| slot.admission.same_original_admission(admission))
            .ok_or("Own /3 original root missing")?;
        if slot.joined.is_some() {
            return Ok(true);
        }
        if slot.join_unknown {
            return Err("Own /3 original worker join failed");
        }
        let Some(handoff) = slot.handoff.as_ref() else {
            return Ok(false);
        };
        let worker = slot.worker.as_mut().ok_or("Own /3 worker absent")?;
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker.try_join(handoff)));
        match result {
            Ok(Ok(Some(custody))) => {
                slot.joined = Some(custody);
                Ok(true)
            }
            Ok(Ok(None)) | Ok(Err(RoventBoardCollectionWorkerJoinFailureV3::Busy)) => Ok(false),
            Ok(Err(error)) => {
                slot.join_unknown = true;
                slot.join_failure = Some(error);
                Err("Own /3 worker join rejected")
            }
            Err(payload) => {
                slot.join_unknown = true;
                slot.join_panic = Some(payload);
                Err("Own /3 worker join panicked")
            }
        }
    }

    /// Take the same custody returned by the manager; this is not a teacher
    /// aggregate or permission to discard rows and native evidence.
    pub(crate) fn try_take_returned(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Option<Arc<Mutex<Custody>>> {
        self.state
            .try_lock()
            .ok()?
            .slots
            .iter_mut()
            .find(|slot| slot.admission.same_original_admission(admission))?
            .returned
            .take()
    }

    /// Transfer one exact failed join for diagnosis. Removing the diagnostic
    /// does not clear the root's unknown state or authorize a new worker.
    pub(crate) fn try_take_join_failure(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Option<RoventBoardCollectionWorkerJoinFailureV3> {
        self.state
            .try_lock()
            .ok()?
            .slots
            .iter_mut()
            .find(|slot| slot.admission.same_original_admission(admission))?
            .join_failure
            .take()
    }

    pub(crate) fn try_take_join_panic(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Option<Box<dyn std::any::Any + Send>> {
        self.state
            .try_lock()
            .ok()?
            .slots
            .iter_mut()
            .find(|slot| slot.admission.same_original_admission(admission))?
            .join_panic
            .take()
    }

    /// Transfer the original deferred owner for continued cleanup. The
    /// consumer must retain it and use its same capability and fence.
    pub(crate) fn try_take_deferred(&self) -> Option<Arc<Mutex<RoventBoardDeferredCleanupV3>>> {
        self.state.try_lock().ok()?.deferred.pop()
    }
}

fn manager_lock<'a>(
    state: &'a Mutex<State>,
) -> Result<std::sync::MutexGuard<'a, State>, AdaptiveManagerError> {
    match state.try_lock() {
        Ok(value) => Ok(value),
        Err(TryLockError::WouldBlock) => Err(AdaptiveManagerError::AuxiliaryPending),
        Err(TryLockError::Poisoned(_)) => Err(AdaptiveManagerError::Invalid(
            "Own /3 root custody poisoned",
        )),
    }
}

impl RoventBoardCollectionHooksV3<Backend> for RoventOwnCollectionBridgeV3 {
    fn register_startup_control(
        &self,
        control: &RoventStartupControl,
    ) -> Result<(), AdaptiveManagerError> {
        if let Some(existing) = self.startup.get() {
            if existing.until == control.until
                && existing.cleanup_until == control.cleanup_until
                && Arc::ptr_eq(&existing.cancellation, &control.cancellation)
            {
                return Ok(());
            }
            return Err(AdaptiveManagerError::Invalid(
                "Own /3 startup owner changed",
            ));
        }
        self.startup
            .set(control.clone())
            .map_err(|_| AdaptiveManagerError::AuxiliaryPending)
    }

    fn admit(
        &self,
        port: &mut RoventBoardCollectionAdmissionPortV3<'_, Backend>,
        admission: RoventCollectionRootAdmissionV3,
    ) -> Result<RoventBoardCollectionAdmissionOutcomeV3, RoventBoardCollectionAdmissionFailureV3>
    {
        let mut state = match manager_lock(&self.state) {
            Ok(state) => state,
            Err(AdaptiveManagerError::AuxiliaryPending) => {
                return Ok(RoventBoardCollectionAdmissionOutcomeV3::Pending { admission });
            }
            Err(error) => {
                return Err(RoventBoardCollectionAdmissionFailureV3 {
                    error,
                    admission,
                    owners: Box::new(()),
                });
            }
        };
        let index = match state
            .slots
            .iter()
            .position(|slot| slot.admission.same_original_admission(&admission))
        {
            Some(index) => index,
            None => {
                if state.slots.len() == state.maximum_roots
                    || state.slots.iter().any(|slot| {
                        slot.admission.root_admission().authority()
                            == admission.root_admission().authority()
                    })
                {
                    return Err(RoventBoardCollectionAdmissionFailureV3 {
                        error: AdaptiveManagerError::Invalid(
                            "Own /3 root capacity or duplicate authority",
                        ),
                        admission,
                        owners: Box::new(()),
                    });
                }
                state.slots.push(RootSlot {
                    admission: admission.clone(),
                    requested: false,
                    revoked: Arc::new(AtomicBool::new(false)),
                    installed: false,
                    owner: None,
                    observer: None,
                    route: None,
                    worker: None,
                    worker_key: None,
                    handoff: None,
                    joined: None,
                    returned: None,
                    join_failure: None,
                    join_panic: None,
                    join_unknown: false,
                });
                state.slots.len() - 1
            }
        };
        let slot = &mut state.slots[index];
        if !slot.requested {
            match self.request_sender.try_send(OwnRootRequestV3 {
                admission: admission.clone(),
                revoked: Arc::clone(&slot.revoked),
            }) {
                Ok(()) => slot.requested = true,
                Err(mpsc::TrySendError::Full(_)) => {
                    return Ok(RoventBoardCollectionAdmissionOutcomeV3::Pending { admission });
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    return Err(RoventBoardCollectionAdmissionFailureV3 {
                        error: AdaptiveManagerError::Invalid(
                            "Own /3 supervisor request channel disconnected",
                        ),
                        admission,
                        owners: Box::new(()),
                    });
                }
            }
        }
        if slot.installed {
            return Ok(RoventBoardCollectionAdmissionOutcomeV3::Installed);
        }
        let Some(observer) = slot.observer.take() else {
            // Once queued, the supervisor may have acquired original route or
            // worker custody even before the observer is installed.
            return Ok(RoventBoardCollectionAdmissionOutcomeV3::PendingUncertain { admission });
        };
        match port.install(admission, observer) {
            Ok(()) => {
                slot.installed = true;
                Ok(RoventBoardCollectionAdmissionOutcomeV3::Installed)
            }
            Err(original) => {
                if matches!(original.error, AdaptiveManagerError::AuxiliaryPending)
                    && original.observer_status
                        == AdaptiveCollectionV2ObserverStatus::AdmissionNotCalled
                    && original.panic_payload.is_none()
                    && original.admission.same_original_admission(&slot.admission)
                {
                    // Nothing entered the observer. Preserve and retry this
                    // same observer rather than inventing a fresh owner.
                    slot.observer = Some(original.observer);
                    return Ok(RoventBoardCollectionAdmissionOutcomeV3::PendingUncertain {
                        admission: original.admission,
                    });
                }
                let error = original.error.clone();
                let admission = original.admission.clone();
                // Even an admission callback that failed or panicked may
                // have acquired state; preserve the exact rejection object.
                Err(RoventBoardCollectionAdmissionFailureV3 {
                    error,
                    admission,
                    owners: Box::new(original),
                })
            }
        }
    }

    fn pump_root(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<(), AdaptiveManagerError> {
        let state = manager_lock(&self.state)?;
        if state
            .slots
            .iter()
            .any(|slot| slot.installed && slot.admission.same_original_admission(admission))
        {
            Ok(())
        } else {
            Err(AdaptiveManagerError::Invalid("Own /3 pump root absent"))
        }
    }

    fn revoke_root(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<(), AdaptiveManagerError> {
        let state = manager_lock(&self.state)?;
        let slot = state
            .slots
            .iter()
            .find(|slot| slot.admission.same_original_admission(admission))
            .ok_or(AdaptiveManagerError::Invalid("Own /3 revoke root absent"))?;
        slot.revoked.store(true, Ordering::Release);
        // Cancel admission to new teacher work. This does not prove that any
        // previously admitted physical work or native callback has stopped.
        if let Some(route) = &slot.route {
            route.cancel_all();
        }
        Ok(())
    }

    fn finish_deferred(
        &self,
        capability: &RoventNoCaptureRootCapabilityV3,
        fence: &RoventOwnNoCaptureFenceV3,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveManagerError> {
        match self
            .registration
            .try_finish_whole_root_no_capture_v3_own(capability, fence)
        {
            Ok(true) => Ok(AdaptiveCollectionBoundaryV2::Ready),
            Ok(false) => Ok(AdaptiveCollectionBoundaryV2::Busy),
            Err(error) => Err(AdaptiveManagerError::Semantic(error.to_string())),
        }
    }

    fn retain_deferred_cleanup(
        &self,
        cleanup: Arc<Mutex<RoventBoardDeferredCleanupV3>>,
    ) -> Result<RoventBoardCustodyReturnV3, AdaptiveManagerError> {
        let mut state = match manager_lock(&self.state) {
            Ok(state) => state,
            Err(AdaptiveManagerError::AuxiliaryPending) => {
                return Ok(RoventBoardCustodyReturnV3::Busy);
            }
            Err(error) => return Err(error),
        };
        if state
            .deferred
            .iter()
            .any(|existing| Arc::ptr_eq(existing, &cleanup))
        {
            return Ok(RoventBoardCustodyReturnV3::Accepted);
        }
        if state.deferred.len() == state.maximum_roots {
            return Ok(RoventBoardCustodyReturnV3::Busy);
        }
        state.deferred.push(cleanup);
        Ok(RoventBoardCustodyReturnV3::Accepted)
    }

    fn handoff_sender_root(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<mpsc::SyncSender<Handoff>, AdaptiveManagerError> {
        let state = manager_lock(&self.state)?;
        let slot = state
            .slots
            .iter()
            .find(|slot| slot.admission.same_original_admission(admission))
            .ok_or(AdaptiveManagerError::Invalid("Own /3 handoff root absent"))?;
        Ok(slot
            .worker
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .sender())
    }

    fn try_take_root_custody(
        &self,
        handoff: &Handoff,
    ) -> Result<Option<Custody>, AdaptiveManagerError> {
        let admission = match handoff.try_lock() {
            Ok(mut original) => original.parts_mut().0.clone(),
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => {
                return Err(AdaptiveManagerError::Invalid("Own /3 handoff poisoned"));
            }
        };
        let mut state = match manager_lock(&self.state) {
            Ok(state) => state,
            Err(AdaptiveManagerError::AuxiliaryPending) => return Ok(None),
            Err(error) => return Err(error),
        };
        let slot = state
            .slots
            .iter_mut()
            .find(|slot| slot.admission.same_original_admission(&admission))
            .ok_or(AdaptiveManagerError::Invalid(
                "Own /3 original handoff root absent",
            ))?;
        if slot
            .handoff
            .as_ref()
            .is_some_and(|existing| !Arc::ptr_eq(existing, handoff))
        {
            return Err(AdaptiveManagerError::Invalid(
                "Own /3 handoff owner changed",
            ));
        }
        if slot.handoff.is_none() {
            slot.handoff = Some(Arc::clone(handoff));
        }
        if slot.join_unknown {
            return Err(AdaptiveManagerError::Invalid(
                "Own /3 worker join unresolved",
            ));
        }
        if slot.joined.as_ref().is_some_and(|custody| {
            !slot
                .worker_key
                .as_ref()
                .is_some_and(|key| custody.matches_worker(key))
                || !custody.matches_handoff(handoff)
        }) {
            return Err(AdaptiveManagerError::Invalid(
                "Own /3 joined worker owner differs",
            ));
        }
        Ok(slot.joined.take())
    }

    fn return_root_custody(
        &self,
        custody: Arc<Mutex<Custody>>,
    ) -> Result<RoventBoardCustodyReturnV3, AdaptiveManagerError> {
        let (handoff, key_matches) = match custody.try_lock() {
            Ok(value) => (Arc::clone(value.handoff()), {
                let state = match manager_lock(&self.state) {
                    Ok(state) => state,
                    Err(AdaptiveManagerError::AuxiliaryPending) => {
                        return Ok(RoventBoardCustodyReturnV3::Busy);
                    }
                    Err(error) => return Err(error),
                };
                state.slots.iter().any(|slot| {
                    slot.worker_key
                        .as_ref()
                        .is_some_and(|key| value.matches_worker(key))
                        && slot
                            .handoff
                            .as_ref()
                            .is_some_and(|original| Arc::ptr_eq(original, value.handoff()))
                })
            }),
            Err(TryLockError::WouldBlock) => return Ok(RoventBoardCustodyReturnV3::Busy),
            Err(TryLockError::Poisoned(_)) => {
                return Err(AdaptiveManagerError::Invalid(
                    "Own /3 returned custody poisoned",
                ));
            }
        };
        if !key_matches {
            return Err(AdaptiveManagerError::Invalid(
                "Own /3 returned worker owner differs",
            ));
        }
        let mut state = match manager_lock(&self.state) {
            Ok(state) => state,
            Err(AdaptiveManagerError::AuxiliaryPending) => {
                return Ok(RoventBoardCustodyReturnV3::Busy);
            }
            Err(error) => return Err(error),
        };
        let slot = state
            .slots
            .iter_mut()
            .find(|slot| {
                slot.handoff
                    .as_ref()
                    .is_some_and(|original| Arc::ptr_eq(original, &handoff))
            })
            .ok_or(AdaptiveManagerError::Invalid(
                "Own /3 returned handoff owner differs",
            ))?;
        match &slot.returned {
            Some(original) if Arc::ptr_eq(original, &custody) => {
                Ok(RoventBoardCustodyReturnV3::Accepted)
            }
            Some(_) => Err(AdaptiveManagerError::Invalid(
                "Own /3 returned custody changed",
            )),
            None => {
                slot.returned = Some(custody);
                Ok(RoventBoardCustodyReturnV3::Accepted)
            }
        }
    }
}
