//! Own-board /3 collection bridge. Manager callbacks only publish bounded
//! requests, install a prepared observer, or observe retained custody.
//!
//! An external supervisor owns preparation, teacher processes, worker joins,
//! and publication. This bridge does not attest their successful completion.

use super::game_authority_v3::{
    GameAuthorityV3, GameDispatchChannelV3, GameDispatchClaimV3, GameDispatchFullSummaryV3,
    GameDispatchOutcomeV3, GameDispatchReceiptV3, GameRootAbortReceiptV3, GameRootReservationV3,
    GameStepV3, RootReservationV3,
};
use super::launch::PreparedPcCollectionInputsV3;
use super::native_capture::RoventNativeCaptureRegistration;
use super::observer_lock_v3::RoventCollectionOwnerLockV3;
use super::own_worker_v3::OwnRootRecoveryV3;
use super::policy_row_pipeline_v3::PolicyRowPipelineOwnerV3;
use super::speculative_route_v3::SpeculativeRootV3;
use super::teacher::v3_pump::TeacherPumpV3;
use super::teacher::v3_session::TeacherSessionStagingV3;
use super::{invalid, ArenaError};
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
use std::sync::{mpsc, Arc, Mutex, OnceLock, TryLockError, Weak};

type Output = VirtualReviewTicketOutput;
type Backend = RoventNativeBackend<AdaptivePortQuestion>;
type Observer = AdaptiveCollectionObserverBoxV3<AdaptivePortQuestion, Output>;
type Handoff = Arc<Mutex<RoventBoardCollectionHandoffV3<Output>>>;
type Custody = RoventBoardCollectionRootCustodyV3<Output>;

struct RootSlot {
    admission: RoventCollectionRootAdmissionV3,
    reservation: Option<GameRootReservationV3<super::own_worker_v3::OwnWorkerSupervisorV3>>,
    abort_receipt: Option<GameRootAbortReceiptV3<super::own_worker_v3::OwnWorkerSupervisorV3>>,
    dispatch: Option<GameDispatchClaimV3<super::own_worker_v3::OwnWorkerSupervisorV3>>,
    dispatch_receipt: Option<GameDispatchReceiptV3<super::own_worker_v3::OwnWorkerSupervisorV3>>,
    dispatch_full: Option<GameDispatchFullSummaryV3<super::own_worker_v3::OwnWorkerSupervisorV3>>,
    requested: bool,
    revoked: Arc<AtomicBool>,
    worker_reserved: bool,
    teacher_reserved: bool,
    installed: bool,
    owner: Option<RoventCollectionOwnerLockV3>,
    observer: Option<Observer>,
    route: Option<SpeculativeRootV3>,
    worker: Option<RoventBoardCollectionWorkerV3>,
    worker_key: Option<RoventBoardCollectionWorkerKeyV3>,
    recovery: Option<Arc<Mutex<OwnRootRecoveryV3>>>,
    handoff: Option<Handoff>,
    joined: Option<Custody>,
    returned: Option<Arc<Mutex<Custody>>>,
    join_failure: Option<RoventBoardCollectionWorkerJoinFailureV3>,
    join_panic: Option<Box<dyn std::any::Any + Send>>,
    join_unknown: bool,
    join_complete: bool,
}

struct State {
    slots: Vec<RootSlot>,
    deferred: Vec<Arc<Mutex<RoventBoardDeferredCleanupV3>>>,
    maximum_roots: usize,
    maximum_workers: usize,
    maximum_teacher_roots: usize,
}

/// Own-specific retirement proof. It retains the same bridge, recovery and
/// pump and rechecks their original owners immediately before route unlink.
pub(crate) struct RoventOwnTeacherRouteRetirementV3 {
    bridge: Arc<RoventOwnCollectionBridgeV3>,
    route: SpeculativeRootV3,
    recovery: Arc<Mutex<OwnRootRecoveryV3>>,
    pump: Arc<Mutex<TeacherPumpV3>>,
}

impl RoventOwnTeacherRouteRetirementV3 {
    pub(crate) fn with_original_route(
        &self,
        requested: &SpeculativeRootV3,
        unlink: impl FnOnce(),
    ) -> Result<Option<bool>, ArenaError> {
        if !self.route.same_original_route(requested) {
            return Ok(Some(false));
        }
        let state = match self.bridge.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => return Err(invalid("Own /3 root custody poisoned")),
        };
        let Some(slot) = state
            .slots
            .iter()
            .find(|slot| self.route.matches_admission(&slot.admission))
        else {
            return Ok(Some(false));
        };
        if !slot.join_complete
            || slot.join_unknown
            || slot.join_failure.is_some()
            || slot.join_panic.is_some()
            || !slot
                .route
                .as_ref()
                .is_some_and(|route| route.same_original_route(&self.route))
            || !slot
                .recovery
                .as_ref()
                .is_some_and(|root| Arc::ptr_eq(root, &self.recovery))
        {
            return Ok(Some(false));
        }
        let mut recovery = match self.recovery.try_lock() {
            Ok(root) => root,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => return Err(invalid("Own /3 recovery poisoned")),
        };
        if !recovery.matches_admission(&slot.admission)
            || !recovery
                .route
                .as_ref()
                .is_some_and(|route| route.same_original_route(&self.route))
            || !recovery
                .pump
                .as_ref()
                .is_some_and(|pump| Arc::ptr_eq(pump, &self.pump))
            || recovery.consumer_phase != 0
            || recovery.consumer_failure.is_some()
            || recovery.persistence_failure.is_some()
        {
            return Ok(Some(false));
        }
        let _consumer = match &recovery.consumer {
            Some(owner) => match owner.try_lock() {
                Ok(Some(guard)) => Some(guard),
                Ok(None) => return Ok(None),
                Err(error) => return Err(invalid(error)),
            },
            None => None,
        };
        let Some(returned) = &slot.returned else {
            return Ok(Some(false));
        };
        let custody = match returned.try_lock() {
            Ok(custody) => custody,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => {
                return Err(invalid("Own /3 returned custody poisoned"))
            }
        };
        if !slot
            .worker_key
            .as_ref()
            .is_some_and(|key| custody.matches_worker(key))
            || !slot
                .handoff
                .as_ref()
                .is_some_and(|handoff| custody.matches_handoff(handoff))
        {
            return Ok(Some(false));
        }
        let mut handoff = match custody.handoff().try_lock() {
            Ok(handoff) => handoff,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => return Err(invalid("Own /3 handoff poisoned")),
        };
        let (admission, _, settlement, fence) = handoff.parts_mut();
        if !admission.same_original_admission(&slot.admission)
            || settlement.is_err()
            || fence.authority != admission.root_admission().authority()
            || !fence.permits_root_reuse()
            || !handoff
                .teacher_aggregate_ack()
                .is_some_and(|ack| ack.matches_admission(&slot.admission))
        {
            return Ok(Some(false));
        }
        let pump = match self.pump.try_lock() {
            Ok(pump) => pump,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => return Err(invalid("Own /3 teacher pump poisoned")),
        };
        match pump.try_retirement_ready(&self.route)? {
            Some(true) => {
                // This Own bridge has the concrete RoventNativeBackend. Its
                // observer send is synchronous within run_owned, while the
                // same worker issues this original root-quiesce fence only
                // after run_owned returns. Unknown/panic cannot issue it.
                // Guards remain held through this exact route unlink.
                unlink();
                drop(_consumer);
                recovery.route_unlinked = true;
                Ok(Some(true))
            }
            other => Ok(other),
        }
    }
}

/// Both this hook and its external supervisor retain this same Arc. The
/// supervisor is responsible for settling or retaining every root worker.
pub(crate) struct RoventOwnCollectionBridgeV3 {
    inputs: Arc<PreparedPcCollectionInputsV3>,
    game: Arc<GameAuthorityV3<super::own_worker_v3::OwnWorkerSupervisorV3>>,
    registration: RoventNativeCaptureRegistration,
    startup: OnceLock<RoventStartupControl>,
    pipeline: OnceLock<PolicyRowPipelineOwnerV3>,
    state: Mutex<State>,
    request_sender: GameDispatchChannelV3<OwnRootRequestV3>,
    request_receiver: Mutex<mpsc::Receiver<OwnRootRequestV3>>,
}

/// Carries the same root's irrevocable cancellation observation to the
/// supervisor. It must be checked before each acquisition or worker start.
#[derive(Clone)]
pub(crate) struct OwnRootRequestV3 {
    pub admission: RoventCollectionRootAdmissionV3,
    revoked: Arc<AtomicBool>,
}
impl OwnRootRequestV3 {
    pub(crate) fn is_revoked(&self) -> bool {
        self.revoked.load(Ordering::Acquire)
    }
    pub(crate) fn revocation_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.revoked)
    }
}

/// A rejected preparation returns every original object to its supervisor.
pub(crate) struct OwnPreparedRootV3 {
    pub admission: RoventCollectionRootAdmissionV3,
    pub owner: RoventCollectionOwnerLockV3,
    pub route: SpeculativeRootV3,
    pub worker: RoventBoardCollectionWorkerV3,
    pub stage: Arc<Mutex<TeacherSessionStagingV3>>,
    pub pump: Arc<Mutex<TeacherPumpV3>>,
    pub pipeline: PolicyRowPipelineOwnerV3,
    pub recovery: Weak<Mutex<OwnRootRecoveryV3>>,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OwnReserveOutcomeV3 {
    Reserved,
    AlreadyReserved,
    Busy,
    Exhausted,
}

impl RoventOwnCollectionBridgeV3 {
    /// This removes only the exact completed native capture route. All root,
    /// returned custody and retained output owners stay in their original
    /// slots; unlinking is not launch settlement or permission to delete.
    pub(crate) fn try_retire_route(
        self: &Arc<Self>,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<Option<bool>, ArenaError> {
        let (route, recovery) = {
            let state = match self.state.try_lock() {
                Ok(state) => state,
                Err(TryLockError::WouldBlock) => return Ok(None),
                Err(TryLockError::Poisoned(_)) => {
                    return Err(invalid("Own /3 root custody poisoned"))
                }
            };
            let Some(slot) = state
                .slots
                .iter()
                .find(|slot| slot.admission.same_original_admission(admission))
            else {
                return Ok(Some(false));
            };
            let (Some(route), Some(recovery)) = (&slot.route, &slot.recovery) else {
                return Ok(Some(false));
            };
            (route.clone(), Arc::clone(recovery))
        };
        let pump = {
            let original = match recovery.try_lock() {
                Ok(original) => original,
                Err(TryLockError::WouldBlock) => return Ok(None),
                Err(TryLockError::Poisoned(_)) => return Err(invalid("Own /3 recovery poisoned")),
            };
            let Some(pump) = &original.pump else {
                return Ok(Some(false));
            };
            Arc::clone(pump)
        };
        let proof = RoventOwnTeacherRouteRetirementV3 {
            bridge: Arc::clone(self),
            route: route.clone(),
            recovery,
            pump,
        };
        self.registration
            .try_unlink_finished_speculative_root_v3_own(&route, &proof)
    }
    /// Called before native startup. No worker or teacher is started here.
    pub(crate) fn new(
        registration: RoventNativeCaptureRegistration,
        inputs: Arc<PreparedPcCollectionInputsV3>,
        game: Arc<GameAuthorityV3<super::own_worker_v3::OwnWorkerSupervisorV3>>,
        lock: &rz_experiments::RoventInputLockV3,
        directory: &cap_std::fs::Dir,
        lock_file: &std::fs::File,
    ) -> Result<Arc<Self>, ArenaError> {
        inputs.validate_original_own_boundary(lock, directory, lock_file)?;
        if !game.matches_inputs(&inputs) {
            return Err(invalid("Own /3 game original inputs differ"));
        }
        let maximum_roots = usize::from(inputs.plan().maximum_roots);
        let maximum_workers = usize::from(inputs.plan().maximum_root_workers);
        let maximum_teacher_roots = usize::from(inputs.plan().maximum_simultaneous_teacher_roots);
        if !(1..=64).contains(&maximum_roots)
            || !(1..=maximum_roots).contains(&maximum_workers)
            || !(1..=maximum_workers).contains(&maximum_teacher_roots)
        {
            return Err(invalid("Own /3 root capacity"));
        }
        let mut slots = Vec::new();
        let mut deferred = Vec::new();
        slots
            .try_reserve_exact(maximum_roots)
            .map_err(|_| invalid("Own /3 root allocation"))?;
        deferred
            .try_reserve_exact(maximum_roots)
            .map_err(|_| invalid("Own /3 cleanup allocation"))?;
        let (request_sender, request_receiver) = mpsc::sync_channel(maximum_roots);
        Ok(Arc::new(Self {
            inputs,
            game,
            registration,
            startup: OnceLock::new(),
            pipeline: OnceLock::new(),
            state: Mutex::new(State {
                slots,
                deferred,
                maximum_roots,
                maximum_workers,
                maximum_teacher_roots,
            }),
            request_sender: GameDispatchChannelV3::new(request_sender),
            request_receiver: Mutex::new(request_receiver),
        }))
    }

    /// Only the supervisor calls this; a request is the exact original root,
    /// not permission to renew its deadline or synthesize a replacement.
    pub(crate) fn try_next_root(&self) -> Result<Option<OwnRootRequestV3>, &'static str> {
        let receiver = match self.request_receiver.try_lock() {
            Ok(receiver) => receiver,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => return Err("Own /3 request receiver poisoned"),
        };
        match receiver.try_recv() {
            Ok(value) => Ok(Some(value)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err("Own /3 request channel disconnected"),
        }
    }

    pub(crate) fn startup(&self) -> Option<&RoventStartupControl> {
        self.startup.get()
    }
    /// Register one actual game writer pipeline, retained before any root
    /// receives it. The same opaque owner must be used by every root.
    pub(crate) fn register_pipeline(
        &self,
        pipeline: &PolicyRowPipelineOwnerV3,
    ) -> Result<(), &'static str> {
        if let Some(existing) = self.pipeline.get() {
            return if existing.same_owner(pipeline) {
                Ok(())
            } else {
                Err("Own /3 game pipeline changed")
            };
        }
        self.pipeline
            .set(pipeline.clone())
            .map_err(|_| "Own /3 pipeline registration raced")
    }
    pub(crate) fn inputs(&self) -> &Arc<PreparedPcCollectionInputsV3> {
        &self.inputs
    }

    pub(crate) fn game(
        &self,
    ) -> &Arc<GameAuthorityV3<super::own_worker_v3::OwnWorkerSupervisorV3>> {
        &self.game
    }

    /// Settle this root's original queue claim before reporting any send.
    /// A known Full is summarized in the same slot and retried later; other
    /// results and all uncertain owners remain in that slot.
    fn try_progress_dispatch(&self, slot: &mut RootSlot) -> Result<bool, &'static str> {
        let envelope = OwnRootRequestV3 {
            admission: slot.admission.clone(),
            revoked: Arc::clone(&slot.revoked),
        };
        if let Some(receipt) = slot.dispatch_receipt.as_ref() {
            if !receipt.matches_game(&self.game)
                || !receipt.matches_root(&slot.admission)
                || !receipt.matches_envelope(&envelope)
            {
                return Err("Own /3 dispatch receipt owner differs");
            }
            match receipt.outcome() {
                GameDispatchOutcomeV3::Sent => {
                    slot.requested = true;
                    return Ok(true);
                }
                GameDispatchOutcomeV3::Full if !receipt.abort_observed() => {
                    return match GameDispatchFullSummaryV3::try_merge_in_slots(
                        &mut slot.dispatch_full,
                        &mut slot.dispatch_receipt,
                    ) {
                        GameStepV3::Ready => Ok(false),
                        _ => Err("Own /3 Full dispatch summary unresolved"),
                    };
                }
                GameDispatchOutcomeV3::Disconnected => {
                    return Err("Own /3 supervisor request channel disconnected")
                }
                GameDispatchOutcomeV3::Full
                | GameDispatchOutcomeV3::Aborted
                | GameDispatchOutcomeV3::AdmissionSealed => return Ok(false),
            }
        }
        if slot.dispatch.is_none() {
            if self.game.abort_requested() {
                return Ok(false);
            }
            match self
                .game
                .try_claim_dispatch(&slot.admission, &envelope, &mut slot.dispatch)
            {
                GameStepV3::Ready => {}
                GameStepV3::Busy | GameStepV3::Aborted | GameStepV3::Closed => return Ok(false),
                _ => return Err("Own /3 dispatch claim rejected"),
            }
        }
        let claim = slot
            .dispatch
            .as_mut()
            .ok_or("Own /3 dispatch claim absent")?;
        match claim.try_send(&self.request_sender, &envelope) {
            Ok(_) => {}
            Err(GameStepV3::Busy) => return Ok(false),
            Err(_) => return Err("Own /3 dispatch send unresolved"),
        }
        match claim.try_settle() {
            GameStepV3::Ready | GameStepV3::Existing => {}
            GameStepV3::Busy => return Ok(false),
            _ => return Err("Own /3 dispatch settlement unresolved"),
        }
        if !claim.transfer_receipt(&mut slot.dispatch_receipt) {
            return Err("Own /3 dispatch receipt retention failed");
        }
        slot.dispatch.take();
        // Revalidate the retained receipt on the next pass.
        Ok(false)
    }

    /// Drain already-issued claims even after native admission is sealed.
    /// This never starts a fresh claim for a root without one.
    pub(crate) fn try_drain_existing_dispatch(&self) -> Result<bool, String> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(false),
            Err(TryLockError::Poisoned(_)) => return Err("Own /3 root custody poisoned".into()),
        };
        let mut pending = false;
        let mut first_error = None;
        for slot in &mut state.slots {
            if slot.dispatch.is_some()
                || (!slot.requested
                    && slot
                        .dispatch_receipt
                        .as_ref()
                        .is_some_and(|receipt| receipt.outcome() == GameDispatchOutcomeV3::Sent))
            {
                match self.try_progress_dispatch(slot) {
                    Ok(true) => {}
                    Ok(false) => pending = true,
                    Err(error) => {
                        if first_error.is_none() {
                            first_error = Some(error.to_owned());
                        }
                    }
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(!pending),
        }
    }

    /// Sticky game abort closes new admission and requests cancellation for
    /// each original root. It is not a physical-completion observation.
    pub(crate) fn try_request_game_abort(&self) -> Result<bool, String> {
        self.game.request_abort();
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(false),
            Err(TryLockError::Poisoned(_)) => return Err("Own /3 root custody poisoned".into()),
        };
        let mut pending = false;
        let mut first_error = None;
        for slot in &mut state.slots {
            slot.revoked.store(true, Ordering::Release);
            if let Some(route) = &slot.route {
                route.cancel_all();
                match route.try_revoke_all_targets() {
                    Ok(Some(())) => {}
                    Ok(None) => pending = true,
                    Err(error) => {
                        if first_error.is_none() {
                            first_error = Some(error.to_string());
                        }
                    }
                }
            }
            if slot.dispatch.is_some()
                || (!slot.requested
                    && slot
                        .dispatch_receipt
                        .as_ref()
                        .is_some_and(|receipt| receipt.outcome() == GameDispatchOutcomeV3::Sent))
            {
                match self.try_progress_dispatch(slot) {
                    Ok(true) => {}
                    Ok(false) => pending = true,
                    Err(error) => {
                        if first_error.is_none() {
                            first_error = Some(error.to_owned());
                        }
                    }
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(!pending),
        }
    }

    pub(crate) fn capture_registration(&self) -> &RoventNativeCaptureRegistration {
        &self.registration
    }

    /// Reserve independent root-worker and teacher-process capacity before
    /// starting either. Unresolved originals continue consuming their slots.
    pub(crate) fn try_reserve_root(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
        recovery: &Arc<Mutex<OwnRootRecoveryV3>>,
        teacher: bool,
    ) -> Result<OwnReserveOutcomeV3, &'static str> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(OwnReserveOutcomeV3::Busy),
            Err(TryLockError::Poisoned(_)) => return Err("Own /3 root custody poisoned"),
        };
        let Some(index) = state
            .slots
            .iter()
            .position(|slot| slot.admission.same_original_admission(admission))
        else {
            return Err("Own /3 original request absent");
        };
        let slot = &state.slots[index];
        if slot.revoked.load(Ordering::Acquire)
            || !slot
                .recovery
                .as_ref()
                .is_some_and(|original| Arc::ptr_eq(original, recovery))
        {
            return Err("Own /3 recovery owner is revoked or changed");
        }
        if if teacher {
            slot.teacher_reserved
        } else {
            slot.worker_reserved
        } {
            return Ok(OwnReserveOutcomeV3::AlreadyReserved);
        }
        if teacher && !slot.worker_reserved {
            return Err("Own /3 worker quota must be reserved first");
        }
        let in_use = state
            .slots
            .iter()
            .filter(|slot| {
                if teacher {
                    slot.teacher_reserved
                } else {
                    slot.worker_reserved
                }
            })
            .count();
        let limit = if teacher {
            state.maximum_teacher_roots
        } else {
            state.maximum_workers
        };
        if in_use >= limit {
            return Ok(OwnReserveOutcomeV3::Exhausted);
        }
        if teacher {
            state.slots[index].teacher_reserved = true;
        } else {
            state.slots[index].worker_reserved = true;
        }
        Ok(OwnReserveOutcomeV3::Reserved)
    }

    /// Install the original recovery owner before it starts any acquisition.
    /// Busy or mismatch never transfers or destroys the caller's Arc.
    pub(crate) fn try_publish_recovery(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
        recovery: &Arc<Mutex<OwnRootRecoveryV3>>,
    ) -> Result<bool, &'static str> {
        let original = match recovery.try_lock() {
            Ok(value) => value,
            Err(TryLockError::WouldBlock) => return Ok(false),
            Err(TryLockError::Poisoned(_)) => return Err("Own /3 recovery poisoned"),
        };
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(false),
            Err(TryLockError::Poisoned(_)) => return Err("Own /3 root custody poisoned"),
        };
        let Some(slot) = state
            .slots
            .iter_mut()
            .find(|slot| slot.admission.same_original_admission(admission))
        else {
            return Err("Own /3 original request absent");
        };
        if !original.matches_admission(admission)
            || !Arc::ptr_eq(&original.inputs, &self.inputs)
            || !Arc::ptr_eq(&original.request.revoked, &slot.revoked)
            || slot.revoked.load(Ordering::Acquire)
        {
            return Err("Own /3 request or revocation owner differs");
        }
        if let Some(existing) = &slot.recovery {
            return Ok(Arc::ptr_eq(existing, recovery));
        }
        slot.recovery = Some(Arc::clone(recovery));
        Ok(true)
    }

    /// The caller creates the actual worker and teacher owner outside the
    /// manager. Rejection keeps the whole prepared owner with the caller.
    pub(crate) fn try_publish_prepared(&self, original: OwnPreparedRootV3) -> OwnPrepareOutcomeV3 {
        let Some(original_recovery) = original.recovery.upgrade() else {
            return OwnPrepareOutcomeV3::Rejected {
                reason: "Own /3 original recovery absent".into(),
                original,
            };
        };
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
            || !self
                .pipeline
                .get()
                .is_some_and(|pipeline| pipeline.same_owner(&original.pipeline))
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
        // Verify the registered request and recovery before mutating the
        // observer or route. Recheck after preparation to catch revocation.
        {
            let state = match self.state.try_lock() {
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
                .iter()
                .find(|slot| slot.admission.same_original_admission(&original.admission))
            else {
                return OwnPrepareOutcomeV3::Rejected {
                    reason: "Own /3 original request absent".into(),
                    original,
                };
            };
            if !slot.requested
                || !slot.worker_reserved
                || !slot.teacher_reserved
                || slot.revoked.load(Ordering::Acquire)
                || slot.installed
                || slot.worker.is_some()
                || slot.observer.is_some()
                || !slot
                    .recovery
                    .as_ref()
                    .is_some_and(|owner| Arc::ptr_eq(owner, &original_recovery))
            {
                return OwnPrepareOutcomeV3::Rejected {
                    reason: "Own /3 root or recovery unavailable".into(),
                    original,
                };
            }
            let recovery_owner = Arc::clone(&original_recovery);
            let matches = match recovery_owner.try_lock() {
                Ok(recovery) => match recovery.try_matches_prepared(
                    &original.admission,
                    &original.route,
                    &original.owner,
                    &original.stage,
                    &original.pump,
                    &original.pipeline,
                    &original.worker.recovery_key(),
                ) {
                    Ok(Some(value)) => value,
                    Ok(None) => return OwnPrepareOutcomeV3::Busy(original),
                    Err(reason) => {
                        return OwnPrepareOutcomeV3::Rejected {
                            reason: reason.into(),
                            original,
                        };
                    }
                },
                Err(TryLockError::WouldBlock) => return OwnPrepareOutcomeV3::Busy(original),
                Err(TryLockError::Poisoned(_)) => {
                    return OwnPrepareOutcomeV3::Rejected {
                        reason: "Own /3 recovery poisoned".into(),
                        original,
                    };
                }
            };
            if !matches {
                return OwnPrepareOutcomeV3::Rejected {
                    reason: "Own /3 staged teacher owner differs".into(),
                    original,
                };
            }
        }
        let bound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            original.owner.try_bind_policy_pipeline(&original.pipeline)
        }));
        match bound {
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
            || !slot.worker_reserved
            || !slot.teacher_reserved
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
        if !slot
            .recovery
            .as_ref()
            .is_some_and(|owner| Arc::ptr_eq(owner, &original_recovery))
        {
            return OwnPrepareOutcomeV3::Rejected {
                reason: "Own /3 original recovery not registered".into(),
                original,
            };
        }
        let recovery_owner = Arc::clone(&original_recovery);
        let matches_original = match recovery_owner.try_lock() {
            Ok(recovery) => match recovery.try_matches_prepared(
                &original.admission,
                &original.route,
                &original.owner,
                &original.stage,
                &original.pump,
                &original.pipeline,
                &original.worker.recovery_key(),
            ) {
                Ok(Some(value)) => value,
                Ok(None) => return OwnPrepareOutcomeV3::Busy(original),
                Err(reason) => {
                    return OwnPrepareOutcomeV3::Rejected {
                        reason: reason.into(),
                        original,
                    };
                }
            },
            Err(TryLockError::WouldBlock) => return OwnPrepareOutcomeV3::Busy(original),
            Err(TryLockError::Poisoned(_)) => {
                return OwnPrepareOutcomeV3::Rejected {
                    reason: "Own /3 recovery poisoned".into(),
                    original,
                };
            }
        };
        if !matches_original {
            return OwnPrepareOutcomeV3::Rejected {
                reason: "Own /3 staged teacher owner differs".into(),
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
            stage: _,
            pump: _,
            pipeline: _,
            recovery: _,
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
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(false),
            Err(TryLockError::Poisoned(_)) => return Err("Own /3 root custody poisoned"),
        };
        let slot = state
            .slots
            .iter_mut()
            .find(|slot| slot.admission.same_original_admission(admission))
            .ok_or("Own /3 original root missing")?;
        if slot.join_complete {
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
                let same_recovery = slot.recovery.as_ref().is_some_and(|original| {
                    custody
                        .original_worker()
                        .downcast_ref::<Weak<Mutex<OwnRootRecoveryV3>>>()
                        .is_some_and(|returned| Weak::ptr_eq(returned, &Arc::downgrade(original)))
                });
                if !same_recovery {
                    // The joined owner is still the original worker outcome;
                    // retain it but never promote it to matching recovery.
                    slot.joined = Some(custody);
                    slot.join_unknown = true;
                    return Err("Own /3 joined worker recovery differs");
                }
                slot.join_complete = true;
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
        match self.game.try_bind_dispatch_channel(&self.request_sender) {
            GameStepV3::Ready | GameStepV3::Existing => {}
            GameStepV3::Busy => {
                return if state
                    .slots
                    .iter()
                    .any(|slot| slot.admission.same_original_admission(&admission))
                {
                    Ok(RoventBoardCollectionAdmissionOutcomeV3::PendingUncertain { admission })
                } else {
                    Ok(RoventBoardCollectionAdmissionOutcomeV3::Pending { admission })
                };
            }
            _ => {
                return Err(RoventBoardCollectionAdmissionFailureV3 {
                    error: AdaptiveManagerError::Invalid("Own /3 game dispatch channel rejected"),
                    admission,
                    owners: Box::new(()),
                })
            }
        }
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
                let reservation = match self.game.try_reserve_root(&admission) {
                    RootReservationV3::Reserved(reservation) => reservation,
                    RootReservationV3::Rejected(GameStepV3::Busy) => {
                        return Ok(RoventBoardCollectionAdmissionOutcomeV3::Pending { admission });
                    }
                    RootReservationV3::Existing => {
                        return Err(RoventBoardCollectionAdmissionFailureV3 {
                            error: AdaptiveManagerError::Invalid(
                                "Own /3 game admission exists without original slot",
                            ),
                            admission,
                            owners: Box::new(()),
                        });
                    }
                    RootReservationV3::Rejected(reason) => {
                        let detail = match reason {
                            GameStepV3::Aborted => "Own /3 game admission aborted",
                            GameStepV3::Closed => "Own /3 game admission closed",
                            GameStepV3::Capacity => "Own /3 game admission capacity",
                            GameStepV3::Foreign => "Own /3 game admission owner differs",
                            GameStepV3::Unknown => "Own /3 game admission unknown",
                            _ => "Own /3 game admission unexpected state",
                        };
                        return Err(RoventBoardCollectionAdmissionFailureV3 {
                            error: AdaptiveManagerError::Invalid(detail),
                            admission,
                            owners: Box::new(()),
                        });
                    }
                };
                state.slots.push(RootSlot {
                    admission: admission.clone(),
                    reservation: Some(reservation),
                    abort_receipt: None,
                    dispatch: None,
                    dispatch_receipt: None,
                    dispatch_full: None,
                    requested: false,
                    revoked: Arc::new(AtomicBool::new(false)),
                    worker_reserved: false,
                    teacher_reserved: false,
                    installed: false,
                    owner: None,
                    observer: None,
                    route: None,
                    worker: None,
                    worker_key: None,
                    recovery: None,
                    handoff: None,
                    joined: None,
                    returned: None,
                    join_failure: None,
                    join_panic: None,
                    join_unknown: false,
                    join_complete: false,
                });
                state.slots.len() - 1
            }
        };
        let slot = &mut state.slots[index];
        if let Some(reservation) = slot.reservation.as_mut() {
            match reservation.commit_registered() {
                GameStepV3::Ready => {
                    slot.reservation.take();
                }
                GameStepV3::Aborted => {
                    if !reservation.transfer_abort_receipt(&mut slot.abort_receipt)
                        || !slot.abort_receipt.as_ref().is_some_and(|receipt| {
                            receipt.matches_game(&self.game)
                                && receipt.matches_original(&slot.admission)
                        })
                    {
                        return Err(RoventBoardCollectionAdmissionFailureV3 {
                            error: AdaptiveManagerError::Invalid(
                                "Own /3 root abort receipt unresolved",
                            ),
                            admission,
                            owners: Box::new(()),
                        });
                    }
                    slot.reservation.take();
                    slot.revoked.store(true, Ordering::Release);
                    return Ok(RoventBoardCollectionAdmissionOutcomeV3::PendingUncertain {
                        admission,
                    });
                }
                GameStepV3::Busy => {
                    return Ok(RoventBoardCollectionAdmissionOutcomeV3::PendingUncertain {
                        admission,
                    });
                }
                _ => {
                    return Err(RoventBoardCollectionAdmissionFailureV3 {
                        error: AdaptiveManagerError::Invalid("Own /3 game reservation unresolved"),
                        admission,
                        owners: Box::new(()),
                    });
                }
            }
        }
        if !slot.requested {
            match self.try_progress_dispatch(slot) {
                Ok(true) => {}
                Ok(false) => {
                    return Ok(RoventBoardCollectionAdmissionOutcomeV3::PendingUncertain {
                        admission,
                    })
                }
                Err(error) => {
                    return Err(RoventBoardCollectionAdmissionFailureV3 {
                        error: AdaptiveManagerError::Invalid(error),
                        admission,
                        owners: Box::new(()),
                    })
                }
            }
        }
        if self.game.abort_requested() {
            slot.revoked.store(true, Ordering::Release);
            if let Some(route) = &slot.route {
                route.cancel_all();
            }
            return Ok(RoventBoardCollectionAdmissionOutcomeV3::PendingUncertain { admission });
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
