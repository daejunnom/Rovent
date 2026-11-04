//! Original V3 board-backed collection boundaries.
//!
//! These types never reconstruct a V2 root or promote an observer handoff
//! into teacher retirement. Callbacks must not block the search manager.

use super::*;
use crate::pals_adaptive::{
    AdaptiveCollectionError, AdaptiveCollectionObserverBoxV3, AdaptiveCollectionV2ObserverStatus,
    AdaptiveManagerError, RoventCollectionRootAdmissionV3, RoventNativeCollectionOriginV3,
    RoventNoCaptureRootCapabilityV3, RoventOwnNoCaptureFenceV3, RoventTeacherScopePhysicalAckV3,
};
use crate::pals_native::ticketed::VirtualReviewTicketOutput;
use crate::rovent_external_launch::RoventTeacherScopeAggregateAckV3;
use std::any::Any;
use std::sync::{OnceLock, mpsc};

/// A pending admission belongs to the same root. `PendingUncertain` means a
/// callback may already have acquired an owner and must not be discarded.
pub enum RoventBoardCollectionAdmissionOutcomeV3 {
    Installed,
    Pending {
        admission: RoventCollectionRootAdmissionV3,
    },
    PendingUncertain {
        admission: RoventCollectionRootAdmissionV3,
    },
    DeferredNoCapture {
        capability: RoventNoCaptureRootCapabilityV3,
        reason: AdaptiveCollectionError,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventBoardCustodyReturnV3 {
    Accepted,
    Busy,
}

pub struct RoventBoardCollectionAdmissionFailureV3 {
    pub error: AdaptiveManagerError,
    pub admission: RoventCollectionRootAdmissionV3,
    pub owners: Box<dyn Any + Send>,
}

/// Return the original observer and admission on a rejected installation.
pub struct RoventBoardCollectionObserverRejectionV3<O> {
    pub error: AdaptiveManagerError,
    pub admission: RoventCollectionRootAdmissionV3,
    pub observer: AdaptiveCollectionObserverBoxV3<AdaptivePortQuestion, O>,
    pub observer_status: AdaptiveCollectionV2ObserverStatus,
    pub panic_payload: Option<Box<dyn Any + Send>>,
}

/// Session-owned original origin and hooks. An outstanding root must remain
/// retained until native, observer and separate teacher custody are settled.
pub(super) struct RoventBoardCollectionStateV3<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub(super) origin: RoventNativeCollectionOriginV3,
    pub(super) hooks: Arc<dyn RoventBoardCollectionHooksV3<B>>,
    pub(super) outstanding: AtomicBool,
    pub(super) failure: Mutex<Option<RoventBoardCollectionAdmissionFailureV3>>,
    pub(super) panics: Mutex<Vec<Box<dyn Any + Send>>>,
    pub(super) handoff: Mutex<Option<Arc<Mutex<RoventBoardCollectionHandoffV3<B::Output>>>>>,
    pub(super) no_capture:
        Mutex<Option<(RoventNoCaptureRootCapabilityV3, AdaptiveCollectionError)>>,
}

/// Borrows the actual adaptive manager; callers cannot manufacture a root.
pub struct RoventBoardCollectionAdmissionPortV3<'a, B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub(super) manager: &'a mut AdaptiveSessionManager<B>,
}

impl<B> RoventBoardCollectionAdmissionPortV3<'_, B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub fn no_capture_capability(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<RoventNoCaptureRootCapabilityV3, AdaptiveManagerError> {
        self.manager.no_capture_root_v3(admission)
    }

    pub fn install(
        &mut self,
        admission: RoventCollectionRootAdmissionV3,
        observer: AdaptiveCollectionObserverBoxV3<AdaptivePortQuestion, B::Output>,
    ) -> Result<(), RoventBoardCollectionObserverRejectionV3<B::Output>> {
        self.manager
            .install_collection_observer_v3(admission, observer)
            .map_err(|failure| RoventBoardCollectionObserverRejectionV3 {
                error: failure.error,
                admission: failure.admission,
                observer: failure.observer,
                observer_status: failure.observer_status,
                panic_payload: failure.panic_payload,
            })
    }
}

/// Handoff preserves the exact observer, root and native root receipt.
/// Neither delivery nor this receipt certifies teacher retirement.
pub struct RoventBoardCollectionHandoffV3<O> {
    pub(super) admission: RoventCollectionRootAdmissionV3,
    pub(super) observer: AdaptiveCollectionObserverBoxV3<AdaptivePortQuestion, O>,
    pub(super) settlement: Result<(), AdaptiveCollectionError>,
    pub(super) native_fence: AdaptiveShutdownReceipt,
    pub(super) teacher_physical_ack: Option<RoventTeacherScopePhysicalAckV3>,
    pub(super) teacher_aggregate_ack: Option<RoventTeacherScopeAggregateAckV3>,
}

impl<O> RoventBoardCollectionHandoffV3<O> {
    pub fn parts_mut(
        &mut self,
    ) -> (
        &RoventCollectionRootAdmissionV3,
        &mut AdaptiveCollectionObserverBoxV3<AdaptivePortQuestion, O>,
        &Result<(), AdaptiveCollectionError>,
        &AdaptiveShutdownReceipt,
    ) {
        (
            &self.admission,
            &mut self.observer,
            &self.settlement,
            &self.native_fence,
        )
    }

    pub fn submit_teacher_physical_ack(
        &mut self,
        ack: RoventTeacherScopePhysicalAckV3,
    ) -> Result<(), RoventTeacherScopePhysicalAckV3> {
        if self.teacher_physical_ack.is_some()
            || self.teacher_aggregate_ack.is_some()
            || !ack.matches_admission(&self.admission)
        {
            return Err(ack);
        }
        self.teacher_physical_ack = Some(ack);
        Ok(())
    }

    pub fn take_teacher_physical_ack(&mut self) -> Option<RoventTeacherScopePhysicalAckV3> {
        self.teacher_physical_ack.take()
    }

    pub fn submit_teacher_aggregate_ack(
        &mut self,
        ack: RoventTeacherScopeAggregateAckV3,
    ) -> Result<(), RoventTeacherScopeAggregateAckV3> {
        if self.teacher_physical_ack.is_some()
            || self.teacher_aggregate_ack.is_some()
            || !ack.matches_admission(&self.admission)
        {
            return Err(ack);
        }
        self.teacher_aggregate_ack = Some(ack);
        Ok(())
    }

    pub fn teacher_aggregate_ack(&self) -> Option<&RoventTeacherScopeAggregateAckV3> {
        self.teacher_aggregate_ack.as_ref()
    }

    pub fn take_teacher_aggregate_ack(&mut self) -> Option<RoventTeacherScopeAggregateAckV3> {
        self.teacher_aggregate_ack.take()
    }
}

pub struct RoventBoardCollectionRootCustodyV3<O> {
    handoff: Arc<Mutex<RoventBoardCollectionHandoffV3<O>>>,
    original_worker: Box<dyn Any + Send>,
}

/// An unresolved marker cleanup retains the same capability and manager
/// fence. This value does not establish removal or teacher retirement.
pub struct RoventBoardDeferredCleanupV3 {
    pub(super) capability: RoventNoCaptureRootCapabilityV3,
    pub(super) reason: AdaptiveCollectionError,
    pub(super) fence: RoventOwnNoCaptureFenceV3,
    pub(super) failure: RoventBoardDeferredCleanupFailureV3,
}

pub enum RoventBoardDeferredCleanupFailureV3 {
    Busy,
    Error(AdaptiveManagerError),
    Panic(Box<dyn Any + Send>),
}

impl RoventBoardDeferredCleanupV3 {
    pub fn capability(&self) -> &RoventNoCaptureRootCapabilityV3 {
        &self.capability
    }
    pub fn reason(&self) -> &AdaptiveCollectionError {
        &self.reason
    }
    pub fn fence(&self) -> &RoventOwnNoCaptureFenceV3 {
        &self.fence
    }
    pub fn failure(&self) -> &RoventBoardDeferredCleanupFailureV3 {
        &self.failure
    }
}

impl<O> RoventBoardCollectionRootCustodyV3<O> {
    fn from_joined_worker(
        handoff: Arc<Mutex<RoventBoardCollectionHandoffV3<O>>>,
        original_worker: Box<dyn Any + Send>,
    ) -> Self {
        Self {
            handoff,
            original_worker,
        }
    }
    pub fn matches_handoff(&self, handoff: &Arc<Mutex<RoventBoardCollectionHandoffV3<O>>>) -> bool {
        Arc::ptr_eq(&self.handoff, handoff)
    }
    pub fn handoff(&self) -> &Arc<Mutex<RoventBoardCollectionHandoffV3<O>>> {
        &self.handoff
    }
    pub fn original_worker(&self) -> &(dyn Any + Send) {
        self.original_worker.as_ref()
    }
    pub fn into_parts(
        self,
    ) -> (
        Arc<Mutex<RoventBoardCollectionHandoffV3<O>>>,
        Box<dyn Any + Send>,
    ) {
        (self.handoff, self.original_worker)
    }
}

pub type RoventBoardRootHandoffV3 =
    Arc<Mutex<RoventBoardCollectionHandoffV3<VirtualReviewTicketOutput>>>;
/// Receiver records the exact same-root handoff actually delivered to worker.
pub struct RoventBoardRootHandoffReceiverV3 {
    receiver: mpsc::Receiver<RoventBoardRootHandoffV3>,
    admission: RoventCollectionRootAdmissionV3,
    received: Arc<Mutex<Option<RoventBoardRootHandoffV3>>>,
}
impl RoventBoardRootHandoffReceiverV3 {
    pub fn recv_timeout(
        &self,
        timeout: std::time::Duration,
    ) -> Result<Option<RoventBoardRootHandoffV3>, AdaptiveManagerError> {
        let handoff = match self.receiver.recv_timeout(timeout) {
            Ok(value) => value,
            Err(mpsc::RecvTimeoutError::Timeout) => return Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(AdaptiveManagerError::Invalid(
                    "V3 root handoff channel closed",
                ));
            }
        };
        if !handoff
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .admission
            .same_original_admission(&self.admission)
        {
            return Err(AdaptiveManagerError::Invalid(
                "V3 worker received another root",
            ));
        }
        let mut received = self.received.lock().unwrap_or_else(|p| p.into_inner());
        if received.is_some() {
            return Err(AdaptiveManagerError::Invalid(
                "V3 worker received duplicate handoff",
            ));
        }
        *received = Some(Arc::clone(&handoff));
        Ok(Some(handoff))
    }
}
pub type RoventBoardRootWorkerTaskV3 =
    Box<dyn FnOnce(RoventBoardRootHandoffReceiverV3) -> Box<dyn Any + Send> + Send + 'static>;

/// Owns the exact orchestration thread and admission. A worker result is not
/// evidence of teacher physical completion or corpus publication.
#[must_use]
pub struct RoventBoardCollectionWorkerV3 {
    key: RoventBoardCollectionWorkerKeyV3,
    admission: RoventCollectionRootAdmissionV3,
    sender: mpsc::SyncSender<RoventBoardRootHandoffV3>,
    received: Arc<Mutex<Option<RoventBoardRootHandoffV3>>>,
    thread: Option<std::thread::JoinHandle<Box<dyn Any + Send>>>,
}

/// The owner retains this token to recover an unresolved worker from the
/// process-local quarantine. It does not imply completion or authorization.
#[derive(Clone)]
pub struct RoventBoardCollectionWorkerKeyV3(Arc<()>);
struct QuarantinedBoardWorkerV3 {
    key: RoventBoardCollectionWorkerKeyV3,
    admission: RoventCollectionRootAdmissionV3,
    sender: mpsc::SyncSender<RoventBoardRootHandoffV3>,
    received: Arc<Mutex<Option<RoventBoardRootHandoffV3>>>,
    thread: std::thread::JoinHandle<Box<dyn Any + Send>>,
}
enum BoardWorkerSlotV3 {
    Reserved(RoventBoardCollectionWorkerKeyV3),
    Quarantined(QuarantinedBoardWorkerV3),
}
impl BoardWorkerSlotV3 {
    fn matches(&self, key: &RoventBoardCollectionWorkerKeyV3) -> bool {
        let owner = match self {
            Self::Reserved(owner) => owner,
            Self::Quarantined(owner) => &owner.key,
        };
        Arc::ptr_eq(&owner.0, &key.0)
    }
}
fn quarantined_board_workers_v3() -> &'static Mutex<Vec<BoardWorkerSlotV3>> {
    static WORKERS: OnceLock<Mutex<Vec<BoardWorkerSlotV3>>> = OnceLock::new();
    WORKERS.get_or_init(|| Mutex::new(Vec::new()))
}
fn release_board_worker_slot_v3(key: &RoventBoardCollectionWorkerKeyV3) {
    let mut slots = quarantined_board_workers_v3()
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    if let Some(index) = slots.iter().position(|slot| slot.matches(key)) {
        slots.swap_remove(index);
    }
}

pub struct RoventBoardCollectionWorkerSpawnFailureV3 {
    pub admission: RoventCollectionRootAdmissionV3,
    pub error: std::io::Error,
    pub task: RoventBoardRootWorkerTaskV3,
}

pub enum RoventBoardCollectionWorkerJoinFailureV3 {
    Busy,
    WrongRoot,
    AlreadyJoined,
    Panicked(Box<dyn Any + Send>),
    CompletedWithoutHandoff(Box<dyn Any + Send>),
}

impl RoventBoardCollectionWorkerV3 {
    pub fn spawn(
        admission: RoventCollectionRootAdmissionV3,
        task: RoventBoardRootWorkerTaskV3,
    ) -> Result<Self, RoventBoardCollectionWorkerSpawnFailureV3> {
        let key = RoventBoardCollectionWorkerKeyV3(Arc::new(()));
        {
            let mut slots = quarantined_board_workers_v3()
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if let Err(error) = slots.try_reserve(1) {
                return Err(RoventBoardCollectionWorkerSpawnFailureV3 {
                    admission,
                    error: std::io::Error::other(error),
                    task,
                });
            }
            // Reserve before spawning, so quarantine on Drop does not allocate.
            slots.push(BoardWorkerSlotV3::Reserved(key.clone()));
        }
        let (sender, receiver) = mpsc::sync_channel(1);
        let received = Arc::new(Mutex::new(None));
        let worker_receiver = RoventBoardRootHandoffReceiverV3 {
            receiver,
            admission: admission.clone(),
            received: Arc::clone(&received),
        };
        // A failed thread creation must return the original task and its owners.
        let slot = Arc::new(Mutex::new(Some(task)));
        let worker_slot = Arc::clone(&slot);
        let thread = std::thread::Builder::new()
            .name("rovent-own-collection-v3".to_owned())
            .spawn(move || {
                let task = worker_slot
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .take()
                    .expect("original V3 worker task");
                task(worker_receiver)
            });
        match thread {
            Ok(thread) => Ok(Self {
                key,
                admission,
                sender,
                received,
                thread: Some(thread),
            }),
            Err(error) => {
                release_board_worker_slot_v3(&key);
                let task = slot
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .take()
                    .expect("failed spawn retains V3 worker task");
                Err(RoventBoardCollectionWorkerSpawnFailureV3 {
                    admission,
                    error,
                    task,
                })
            }
        }
    }

    pub fn sender(&self) -> mpsc::SyncSender<RoventBoardRootHandoffV3> {
        self.sender.clone()
    }

    pub fn recovery_key(&self) -> RoventBoardCollectionWorkerKeyV3 {
        self.key.clone()
    }

    /// Recover only the original process-local worker matching this token.
    pub fn recover(key: &RoventBoardCollectionWorkerKeyV3) -> Option<Self> {
        let mut workers = quarantined_board_workers_v3()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let index = workers.iter().position(|slot| slot.matches(key))?;
        if !matches!(&workers[index], BoardWorkerSlotV3::Quarantined(_)) {
            return None;
        }
        let BoardWorkerSlotV3::Quarantined(owner) = std::mem::replace(
            &mut workers[index],
            BoardWorkerSlotV3::Reserved(key.clone()),
        ) else {
            unreachable!()
        };
        Some(Self {
            key: owner.key,
            admission: owner.admission,
            sender: owner.sender,
            received: owner.received,
            thread: Some(owner.thread),
        })
    }

    pub fn try_join(
        &mut self,
        handoff: &RoventBoardRootHandoffV3,
    ) -> Result<
        Option<RoventBoardCollectionRootCustodyV3<VirtualReviewTicketOutput>>,
        RoventBoardCollectionWorkerJoinFailureV3,
    > {
        let same_admission = match handoff.try_lock() {
            Ok(guard) => guard.admission.same_original_admission(&self.admission),
            Err(std::sync::TryLockError::Poisoned(error)) => error
                .into_inner()
                .admission
                .same_original_admission(&self.admission),
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(RoventBoardCollectionWorkerJoinFailureV3::Busy);
            }
        };
        if !same_admission {
            return Err(RoventBoardCollectionWorkerJoinFailureV3::WrongRoot);
        }
        let thread = self
            .thread
            .as_ref()
            .ok_or(RoventBoardCollectionWorkerJoinFailureV3::AlreadyJoined)?;
        if !thread.is_finished() {
            return Ok(None);
        }
        let received_handoff = match self.received.try_lock() {
            Ok(guard) => guard
                .as_ref()
                .is_some_and(|received| Arc::ptr_eq(received, handoff)),
            Err(std::sync::TryLockError::Poisoned(error)) => error
                .into_inner()
                .as_ref()
                .is_some_and(|received| Arc::ptr_eq(received, handoff)),
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(RoventBoardCollectionWorkerJoinFailureV3::Busy);
            }
        };
        let mut slots = match quarantined_board_workers_v3().try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(error)) => error.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(RoventBoardCollectionWorkerJoinFailureV3::Busy);
            }
        };
        let result = self.thread.take().expect("checked V3 worker handle").join();
        if let Some(index) = slots.iter().position(|slot| slot.matches(&self.key)) {
            slots.swap_remove(index);
        }
        drop(slots);
        match result {
            Ok(original_worker) => {
                if !received_handoff {
                    return Err(
                        RoventBoardCollectionWorkerJoinFailureV3::CompletedWithoutHandoff(
                            original_worker,
                        ),
                    );
                }
                Ok(Some(
                    RoventBoardCollectionRootCustodyV3::from_joined_worker(
                        Arc::clone(handoff),
                        original_worker,
                    ),
                ))
            }
            Err(payload) => Err(RoventBoardCollectionWorkerJoinFailureV3::Panicked(payload)),
        }
    }
}

impl Drop for RoventBoardCollectionWorkerV3 {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            let mut slots = quarantined_board_workers_v3()
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let owner = QuarantinedBoardWorkerV3 {
                key: self.key.clone(),
                admission: self.admission.clone(),
                sender: self.sender.clone(),
                received: Arc::clone(&self.received),
                thread,
            };
            if let Some(slot) = slots.iter_mut().find(|slot| slot.matches(&self.key)) {
                if matches!(slot, BoardWorkerSlotV3::Reserved(_)) {
                    *slot = BoardWorkerSlotV3::Quarantined(owner);
                } else {
                    // Never overwrite another retained original on an invariant violation.
                    std::mem::forget(owner);
                }
            } else {
                // Invariant violation: never detach the original physical owner.
                std::mem::forget(owner);
            }
        }
    }
}

pub trait RoventBoardCollectionHooksV3<B>: Send + Sync
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    fn register_startup_control(
        &self,
        control: &crate::rovent_bootstrap::RoventStartupControl,
    ) -> Result<(), AdaptiveManagerError>;
    fn admit(
        &self,
        port: &mut RoventBoardCollectionAdmissionPortV3<'_, B>,
        admission: RoventCollectionRootAdmissionV3,
    ) -> Result<RoventBoardCollectionAdmissionOutcomeV3, RoventBoardCollectionAdmissionFailureV3>;
    fn pump_root(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<(), AdaptiveManagerError>;
    fn revoke_root(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<(), AdaptiveManagerError>;
    fn finish_deferred(
        &self,
        capability: &RoventNoCaptureRootCapabilityV3,
        fence: &RoventOwnNoCaptureFenceV3,
    ) -> Result<crate::pals_adaptive::AdaptiveCollectionBoundaryV2, AdaptiveManagerError>;
    fn retain_deferred_cleanup(
        &self,
        cleanup: Arc<Mutex<RoventBoardDeferredCleanupV3>>,
    ) -> Result<RoventBoardCustodyReturnV3, AdaptiveManagerError>;
    fn handoff_sender_root(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<
        std::sync::mpsc::SyncSender<Arc<Mutex<RoventBoardCollectionHandoffV3<B::Output>>>>,
        AdaptiveManagerError,
    >;
    /// Only the original orchestration worker's successful join can create
    /// this custody. It does not itself prove teacher retirement.
    fn try_take_root_custody(
        &self,
        handoff: &Arc<Mutex<RoventBoardCollectionHandoffV3<B::Output>>>,
    ) -> Result<Option<RoventBoardCollectionRootCustodyV3<B::Output>>, AdaptiveManagerError>;
    fn return_root_custody(
        &self,
        custody: Arc<Mutex<RoventBoardCollectionRootCustodyV3<B::Output>>>,
    ) -> Result<RoventBoardCustodyReturnV3, AdaptiveManagerError>;
}

/// Rejection retains the actual loaded origin and hook owner.
pub struct RoventBoardCollectionRegistrationRejectionV3<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub error: SearchSessionFailure,
    pub origin: RoventNativeCollectionOriginV3,
    pub hooks: Arc<dyn RoventBoardCollectionHooksV3<B>>,
    pub panic_payload: Option<Box<dyn Any + Send>>,
}

/// Retains the same root across nonblocking admission ticks.
pub(super) struct RoventBoardRootAdmissionAttemptV3 {
    pub(super) admission: RoventCollectionRootAdmissionV3,
    uncertain: bool,
}

pub(super) enum RoventBoardAdmissionStepV3 {
    Installed,
    Pending,
    DeferredNoCapture,
}

impl<B> RoventBoardCollectionStateV3<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub(super) fn prepare_root(
        &self,
        manager: &AdaptiveSessionManager<B>,
        root: Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV3>,
        budget: Digest,
        selective_policy: Digest,
    ) -> Result<RoventBoardRootAdmissionAttemptV3, AdaptiveManagerError> {
        let frontier = manager
            .driver()
            .port()
            .and_then(|port| port.rovent_frontier())
            .ok_or(AdaptiveManagerError::Invalid(
                "V3 own collection frontier absent",
            ))?;
        let admission = RoventCollectionRootAdmissionV3::from_attached_frontier(
            &self.origin,
            frontier,
            root,
            budget,
            selective_policy,
        )?;
        Ok(RoventBoardRootAdmissionAttemptV3 {
            admission,
            uncertain: false,
        })
    }

    /// One nonblocking attempt. The caller retains `attempt` on Pending and
    /// retries it from a later coordinator tick under the original deadline.
    pub(super) fn try_admit_root(
        &self,
        manager: &mut AdaptiveSessionManager<B>,
        attempt: &mut RoventBoardRootAdmissionAttemptV3,
    ) -> Result<RoventBoardAdmissionStepV3, AdaptiveManagerError> {
        self.outstanding.store(true, Ordering::Release);
        let admission = &attempt.admission;
        let mut port = RoventBoardCollectionAdmissionPortV3 { manager };
        let outcome = match catch_unwind(AssertUnwindSafe(|| {
            self.hooks.admit(&mut port, admission.clone())
        })) {
            Ok(outcome) => outcome,
            Err(payload) => {
                self.panics
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(payload);
                return Err(AdaptiveManagerError::Invalid(
                    "V3 own collection admission panicked",
                ));
            }
        };
        match outcome {
            Ok(RoventBoardCollectionAdmissionOutcomeV3::DeferredNoCapture {
                capability,
                reason,
            }) => {
                if !capability.same_original_admission(admission)
                    || !port
                        .manager
                        .original_no_capture_v3()
                        .is_some_and(|original| original.same_original_capability(&capability))
                    || port.manager.collection_root_admission_v3().is_some()
                    || attempt.uncertain
                {
                    *self.failure.lock().unwrap_or_else(|p| p.into_inner()) =
                        Some(RoventBoardCollectionAdmissionFailureV3 {
                            error: AdaptiveManagerError::Invalid("V3 own no-capture root mismatch"),
                            admission: admission.clone(),
                            owners: Box::new((capability, reason)),
                        });
                    return Err(AdaptiveManagerError::Invalid(
                        "V3 own no-capture root mismatch",
                    ));
                }
                *self.no_capture.lock().unwrap_or_else(|p| p.into_inner()) =
                    Some((capability, reason));
                Ok(RoventBoardAdmissionStepV3::DeferredNoCapture)
            }
            Ok(RoventBoardCollectionAdmissionOutcomeV3::Installed) => {
                if !port
                    .manager
                    .collection_root_admission_v3()
                    .is_some_and(|actual| actual.same_original_admission(admission))
                {
                    return Err(AdaptiveManagerError::Invalid(
                        "V3 own installed root differs",
                    ));
                }
                Ok(RoventBoardAdmissionStepV3::Installed)
            }
            Ok(RoventBoardCollectionAdmissionOutcomeV3::Pending {
                admission: returned,
            }) => {
                if !returned.same_original_admission(admission)
                    || port.manager.collection_root_admission_v3().is_some()
                {
                    *self.failure.lock().unwrap_or_else(|p| p.into_inner()) =
                        Some(RoventBoardCollectionAdmissionFailureV3 {
                            error: AdaptiveManagerError::Invalid("V3 own pending root differs"),
                            admission: returned,
                            owners: Box::new(()),
                        });
                    return Err(AdaptiveManagerError::Invalid("V3 own pending root differs"));
                }
                if !attempt.uncertain && port.manager.original_no_capture_v3().is_none() {
                    self.outstanding.store(false, Ordering::Release);
                }
                Ok(RoventBoardAdmissionStepV3::Pending)
            }
            Ok(RoventBoardCollectionAdmissionOutcomeV3::PendingUncertain {
                admission: returned,
            }) => {
                if !returned.same_original_admission(admission) {
                    *self.failure.lock().unwrap_or_else(|p| p.into_inner()) =
                        Some(RoventBoardCollectionAdmissionFailureV3 {
                            error: AdaptiveManagerError::Invalid("V3 own uncertain root differs"),
                            admission: returned,
                            owners: Box::new(()),
                        });
                    return Err(AdaptiveManagerError::Invalid(
                        "V3 own uncertain root differs",
                    ));
                }
                attempt.uncertain = true;
                Ok(RoventBoardAdmissionStepV3::Pending)
            }
            Err(failure) => {
                let error = if failure.admission.same_original_admission(admission) {
                    failure.error.clone()
                } else {
                    AdaptiveManagerError::Invalid("V3 own rejected another root")
                };
                *self.failure.lock().unwrap_or_else(|p| p.into_inner()) = Some(failure);
                Err(error)
            }
        }
    }
}

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    /// Install only the original V3 loaded origin on its idle native owner.
    /// This registers hooks; it does not attest a root, teacher or completion.
    pub fn register_rovent_collection_v3(
        &mut self,
        origin: RoventNativeCollectionOriginV3,
        hooks: Arc<dyn RoventBoardCollectionHooksV3<B>>,
        control: &crate::rovent_bootstrap::RoventStartupControl,
    ) -> Result<(), Box<RoventBoardCollectionRegistrationRejectionV3<B>>> {
        let valid = origin.sources().matches_startup_control(control)
            && self.rovent_root_policy_v3.as_ref().is_some_and(|policy| {
                policy.owner.v3().is_some_and(|lock| {
                    Arc::ptr_eq(lock, origin.sources().lock_owner())
                        && matches!(
                            lock.manifest.engine.researcher.as_ref(),
                            Some(rz_experiments::RoventResearcher::Own { .. })
                        )
                })
            })
            && matches!(
                self.rovent.as_ref(),
                Some(SessionPcrProfile::LateBoundOwn(_))
            )
            && self.rovent_collection_origin.is_none()
            && self.rovent_collection_origin_v2.is_none()
            && self.rovent_collection_v3.is_none()
            && !self.closed.load(Ordering::Acquire)
            && self.state.try_lock().is_ok_and(|state| match &*state {
                SessionState::Idle(owners)
                    if owners.continuation.is_none()
                        && owners.native_finished.is_none()
                        && !owners.native.is_quarantined() =>
                {
                    match (
                        owners.native.native_acceptance_verifier(),
                        owners.native.ready_clock(),
                    ) {
                        (Some(verifier), Some(clock)) => {
                            origin.matches_native_owner(&verifier, &clock)
                        }
                        _ => false,
                    }
                }
                _ => false,
            });
        if !valid {
            return Err(Box::new(RoventBoardCollectionRegistrationRejectionV3 {
                error: failure(
                    "RoventBoardCollectionRegistrationV3",
                    "original V3 lock, own researcher or idle native owner differs",
                ),
                origin,
                hooks,
                panic_payload: None,
            }));
        }
        match catch_unwind(AssertUnwindSafe(|| hooks.register_startup_control(control))) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return Err(Box::new(RoventBoardCollectionRegistrationRejectionV3 {
                    error: SearchSessionFailure::debug("RoventBoardCollectionControlV3", &error)
                        .unknown_if_unclosed(),
                    origin,
                    hooks,
                    panic_payload: None,
                }));
            }
            Err(payload) => {
                return Err(Box::new(RoventBoardCollectionRegistrationRejectionV3 {
                    error: SearchSessionFailure::physical_completion_unknown(),
                    origin,
                    hooks,
                    panic_payload: Some(payload),
                }));
            }
        }
        self.rovent_collection_v3 = Some(RoventBoardCollectionStateV3 {
            origin,
            hooks,
            outstanding: AtomicBool::new(false),
            failure: Mutex::new(None),
            panics: Mutex::new(Vec::new()),
            handoff: Mutex::new(None),
            no_capture: Mutex::new(None),
        });
        Ok(())
    }
}
