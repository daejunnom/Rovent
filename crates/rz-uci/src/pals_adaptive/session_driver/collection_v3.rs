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
    pub(super) first_execution_error: Mutex<Option<AdaptiveManagerError>>,
    pub(super) first_session_failure: Mutex<Option<SearchSessionFailure>>,
    pub(super) first_collection_error: Mutex<Option<AdaptiveCollectionError>>,
    pub(super) panics: Mutex<Vec<Box<dyn Any + Send>>>,
    pub(super) handoff: Mutex<Option<Arc<Mutex<RoventBoardCollectionHandoffV3<B::Output>>>>>,
    pub(super) custody: Mutex<Option<Arc<Mutex<RoventBoardCollectionRootCustodyV3<B::Output>>>>>,
    pub(super) no_capture:
        Mutex<Option<(RoventNoCaptureRootCapabilityV3, AdaptiveCollectionError)>>,
    pub(super) deferred_cleanup: Mutex<Option<Arc<Mutex<RoventBoardDeferredCleanupV3>>>>,
    pub(super) prior_cleanup_failures: Mutex<Vec<Arc<Mutex<RoventBoardDeferredCleanupV3>>>>,
    pub(super) recovery_root: Mutex<Option<super::collection_v3_root::OwnCollectionRootRunV3>>,
    pub(super) cleanup_context: Mutex<Option<SearchSessionContext>>,
    pub(super) failed_evidence: Mutex<
        Option<
            Vec<
                super::super::manager::RoventFailedCollectionEvidence<
                    AdaptivePortQuestion,
                    B::Output,
                >,
            >,
        >,
    >,
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
    worker_key: RoventBoardCollectionWorkerKeyV3,
}

/// An unresolved marker cleanup retains the same capability and manager
/// fence. This value does not establish removal or teacher retirement.
pub struct RoventBoardDeferredCleanupV3 {
    pub(super) capability: RoventNoCaptureRootCapabilityV3,
    pub(super) reason: AdaptiveCollectionError,
    pub(super) fence: RoventOwnNoCaptureFenceV3,
    pub(super) failure: RoventBoardDeferredCleanupFailureV3,
    pub(super) retention_failure: Option<AdaptiveManagerError>,
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
    pub fn retention_failure(&self) -> Option<&AdaptiveManagerError> {
        self.retention_failure.as_ref()
    }
}

impl<O> RoventBoardCollectionRootCustodyV3<O> {
    fn from_joined_worker(
        handoff: Arc<Mutex<RoventBoardCollectionHandoffV3<O>>>,
        original_worker: Box<dyn Any + Send>,
        worker_key: RoventBoardCollectionWorkerKeyV3,
    ) -> Self {
        Self {
            handoff,
            original_worker,
            worker_key,
        }
    }
    pub fn matches_worker(&self, key: &RoventBoardCollectionWorkerKeyV3) -> bool {
        Arc::ptr_eq(&self.worker_key.0, &key.0)
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
        RoventBoardCollectionWorkerKeyV3,
    ) {
        (self.handoff, self.original_worker, self.worker_key)
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
    task_custody: Arc<Mutex<Option<RoventBoardRootWorkerTaskV3>>>,
    thread: Option<std::thread::JoinHandle<Box<dyn Any + Send>>>,
}

/// The owner retains this token to recover an unresolved worker from the
/// process-local quarantine. It does not imply completion or authorization.
#[derive(Clone)]
pub struct RoventBoardCollectionWorkerKeyV3(Arc<()>);
impl RoventBoardCollectionWorkerKeyV3 {
    /// Same worker identity only; this is not a completion or join proof.
    pub fn same_original(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    /// Registry observation only. Missing or Reserved never establishes
    /// that a thread did not start or that physical work has completed.
    pub fn observe(&self) -> RoventBoardWorkerRecoveryObservationV3 {
        let slots = match quarantined_board_workers_v3().try_lock() {
            Ok(slots) => slots,
            Err(std::sync::TryLockError::WouldBlock) => {
                return RoventBoardWorkerRecoveryObservationV3::Busy;
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return RoventBoardWorkerRecoveryObservationV3::Poisoned;
            }
        };
        match slots.iter().find(|slot| slot.matches(self)) {
            Some(BoardWorkerSlotV3::Reserved(_)) => {
                RoventBoardWorkerRecoveryObservationV3::Reserved
            }
            Some(BoardWorkerSlotV3::Staged(stage)) => {
                RoventBoardWorkerRecoveryObservationV3::Staged(stage.phase)
            }
            Some(BoardWorkerSlotV3::Quarantined(_)) => {
                RoventBoardWorkerRecoveryObservationV3::Worker
            }
            None => RoventBoardWorkerRecoveryObservationV3::Missing,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventBoardWorkerRecoveryObservationV3 {
    Reserved,
    Staged(RoventBoardWorkerStagePhaseV3),
    Worker,
    Missing,
    Busy,
    Poisoned,
}
struct QuarantinedBoardWorkerV3 {
    key: RoventBoardCollectionWorkerKeyV3,
    admission: RoventCollectionRootAdmissionV3,
    sender: mpsc::SyncSender<RoventBoardRootHandoffV3>,
    received: Arc<Mutex<Option<RoventBoardRootHandoffV3>>>,
    task_custody: Arc<Mutex<Option<RoventBoardRootWorkerTaskV3>>>,
    thread: std::thread::JoinHandle<Box<dyn Any + Send>>,
}
struct QuarantinedBoardWorkerStageV3 {
    key: RoventBoardCollectionWorkerKeyV3,
    admission: RoventCollectionRootAdmissionV3,
    sender: mpsc::SyncSender<RoventBoardRootHandoffV3>,
    received: Arc<Mutex<Option<RoventBoardRootHandoffV3>>>,
    receiver: Option<RoventBoardRootHandoffReceiverV3>,
    task: Arc<Mutex<Option<RoventBoardRootWorkerTaskV3>>>,
    phase: RoventBoardWorkerStagePhaseV3,
    failure: Option<std::io::Error>,
    start_seal: Option<Arc<()>>,
}
enum BoardWorkerSlotV3 {
    Reserved(RoventBoardCollectionWorkerKeyV3),
    Quarantined(QuarantinedBoardWorkerV3),
    Staged(QuarantinedBoardWorkerStageV3),
}
impl BoardWorkerSlotV3 {
    fn matches(&self, key: &RoventBoardCollectionWorkerKeyV3) -> bool {
        let owner = match self {
            Self::Reserved(owner) => owner,
            Self::Quarantined(owner) => &owner.key,
            Self::Staged(owner) => &owner.key,
        };
        Arc::ptr_eq(&owner.0, &key.0)
    }
}
fn quarantined_board_workers_v3() -> &'static Mutex<Vec<BoardWorkerSlotV3>> {
    static WORKERS: OnceLock<Mutex<Vec<BoardWorkerSlotV3>>> = OnceLock::new();
    WORKERS.get_or_init(|| Mutex::new(Vec::new()))
}
fn release_board_worker_slot_v3(key: &RoventBoardCollectionWorkerKeyV3) {
    let Ok(mut slots) = quarantined_board_workers_v3().lock() else {
        return;
    };
    if let Some(index) = slots.iter().position(|slot| slot.matches(key)) {
        slots.swap_remove(index);
    }
}

pub struct RoventBoardCollectionWorkerSpawnFailureV3 {
    pub admission: RoventCollectionRootAdmissionV3,
    pub error: std::io::Error,
    pub task: RoventBoardRootWorkerTaskV3,
}

/// StartingUnknown never authorizes a retry or claims that no thread exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventBoardWorkerStagePhaseV3 {
    Ready,
    StartingUnknown,
    Running,
    Failed,
    FailedTaken,
    Transferred,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventBoardWorkerStageStartV3 {
    Started,
    Failed,
}

pub struct RoventBoardWorkerStagePrepareFailureV3 {
    pub admission: RoventCollectionRootAdmissionV3,
    pub error: std::io::Error,
}

pub struct RoventBoardWorkerStageStartRejectionV3 {
    pub reason: &'static str,
    pub task: RoventBoardRootWorkerTaskV3,
}

/// Own this stage and its key in the supervisor before starting a thread.
/// Drop moves unresolved task/handle custody to the existing worker registry.
#[must_use]
pub struct RoventBoardCollectionWorkerStageV3 {
    key: RoventBoardCollectionWorkerKeyV3,
    admission: RoventCollectionRootAdmissionV3,
    sender: mpsc::SyncSender<RoventBoardRootHandoffV3>,
    received: Arc<Mutex<Option<RoventBoardRootHandoffV3>>>,
    receiver: Option<RoventBoardRootHandoffReceiverV3>,
    task: Arc<Mutex<Option<RoventBoardRootWorkerTaskV3>>>,
    thread: Option<std::thread::JoinHandle<Box<dyn Any + Send>>>,
    phase: RoventBoardWorkerStagePhaseV3,
    failure: Option<std::io::Error>,
    start_seal: Option<Arc<()>>,
}

/// Unique, non-clone witness for one Ready stage and its original receiver.
/// It asserts neither task submission nor physical execution.
pub struct RoventBoardWorkerStartSourceV3 {
    seal: Arc<()>,
    key: RoventBoardCollectionWorkerKeyV3,
    admission: RoventCollectionRootAdmissionV3,
    received: Arc<Mutex<Option<RoventBoardRootHandoffV3>>>,
}

/// Minted only by `start_with_source` after that same stage's spawn returns.
/// A start result is not a worker join or physical completion proof.
pub struct RoventBoardWorkerStartOutcomeV3 {
    seal: Arc<()>,
    key: RoventBoardCollectionWorkerKeyV3,
    admission: RoventCollectionRootAdmissionV3,
    received: Arc<Mutex<Option<RoventBoardRootHandoffV3>>>,
    result: RoventBoardWorkerStageStartV3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventBoardWorkerStartSourceErrorV3 {
    NotReady,
    AlreadyIssued,
    Busy,
    Poisoned,
    RegistryUnknown,
}

pub struct RoventBoardWorkerStartSourceCancelFailureV3 {
    pub error: RoventBoardWorkerStartSourceErrorV3,
    pub source: RoventBoardWorkerStartSourceV3,
}

impl RoventBoardWorkerStartSourceV3 {
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }

    pub fn same_original_outcome(&self, outcome: &RoventBoardWorkerStartOutcomeV3) -> bool {
        Arc::ptr_eq(&self.seal, &outcome.seal)
            && self.key.same_original(&outcome.key)
            && self.admission.same_original_admission(&outcome.admission)
            && Arc::ptr_eq(&self.received, &outcome.received)
    }

    pub fn recovery_key(&self) -> RoventBoardCollectionWorkerKeyV3 {
        self.key.clone()
    }
}

impl RoventBoardWorkerStartOutcomeV3 {
    pub fn result(&self) -> RoventBoardWorkerStageStartV3 {
        self.result
    }
    pub fn recovery_key(&self) -> RoventBoardCollectionWorkerKeyV3 {
        self.key.clone()
    }
}

pub enum RoventBoardCollectionWorkerJoinFailureV3 {
    Busy,
    TaskCustodyUnknown,
    CustodyPoisoned,
    RegistryPoisoned,
    RegistryUnknown,
    WrongRoot,
    AlreadyJoined,
    Panicked(Box<dyn Any + Send>),
    CompletedWithoutHandoff(Box<dyn Any + Send>),
}

impl RoventBoardCollectionWorkerStageV3 {
    /// Issue exactly one witness from this actual Ready stage. The caller
    /// must separately establish the original game/dispatch authority.
    pub fn issue_start_source(
        &mut self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<RoventBoardWorkerStartSourceV3, RoventBoardWorkerStartSourceErrorV3> {
        if self.start_seal.is_some() {
            return Err(RoventBoardWorkerStartSourceErrorV3::AlreadyIssued);
        }
        if self.phase != RoventBoardWorkerStagePhaseV3::Ready
            || !self.matches_admission(admission)
            || self.receiver.is_none()
            || self.thread.is_some()
            || self.failure.is_some()
        {
            return Err(RoventBoardWorkerStartSourceErrorV3::NotReady);
        }
        {
            let slot = self.task.try_lock().map_err(|error| match error {
                std::sync::TryLockError::WouldBlock => RoventBoardWorkerStartSourceErrorV3::Busy,
                std::sync::TryLockError::Poisoned(_) => {
                    RoventBoardWorkerStartSourceErrorV3::Poisoned
                }
            })?;
            if slot.is_some() {
                return Err(RoventBoardWorkerStartSourceErrorV3::NotReady);
            }
        }
        match self.key.observe() {
            RoventBoardWorkerRecoveryObservationV3::Reserved => {}
            RoventBoardWorkerRecoveryObservationV3::Busy => {
                return Err(RoventBoardWorkerStartSourceErrorV3::Busy);
            }
            RoventBoardWorkerRecoveryObservationV3::Poisoned => {
                return Err(RoventBoardWorkerStartSourceErrorV3::Poisoned);
            }
            _ => return Err(RoventBoardWorkerStartSourceErrorV3::RegistryUnknown),
        }
        let seal = Arc::new(());
        self.start_seal = Some(Arc::clone(&seal));
        Ok(RoventBoardWorkerStartSourceV3 {
            seal,
            key: self.key.clone(),
            admission: self.admission.clone(),
            received: Arc::clone(&self.received),
        })
    }

    pub fn matches_start_source(&self, source: &RoventBoardWorkerStartSourceV3) -> bool {
        self.start_seal
            .as_ref()
            .is_some_and(|seal| Arc::ptr_eq(seal, &source.seal))
            && self.key.same_original(&source.key)
            && self.admission.same_original_admission(&source.admission)
            && Arc::ptr_eq(&self.received, &source.received)
    }

    /// Revoke an unused witness only while the same stage is still Ready and
    /// its original task cell is empty. Rejection retains the witness.
    pub fn cancel_start_source(
        &mut self,
        source: RoventBoardWorkerStartSourceV3,
    ) -> Result<(), RoventBoardWorkerStartSourceCancelFailureV3> {
        if !self.matches_start_source(&source)
            || self.phase != RoventBoardWorkerStagePhaseV3::Ready
            || self.receiver.is_none()
        {
            return Err(RoventBoardWorkerStartSourceCancelFailureV3 {
                error: RoventBoardWorkerStartSourceErrorV3::NotReady,
                source,
            });
        }
        let state = match self.task.try_lock() {
            Ok(slot) => {
                if slot.is_none() {
                    None
                } else {
                    Some(RoventBoardWorkerStartSourceErrorV3::NotReady)
                }
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                Some(RoventBoardWorkerStartSourceErrorV3::Busy)
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                Some(RoventBoardWorkerStartSourceErrorV3::Poisoned)
            }
        };
        if let Some(error) = state {
            return Err(RoventBoardWorkerStartSourceCancelFailureV3 { error, source });
        }
        let registry_error = match self.key.observe() {
            RoventBoardWorkerRecoveryObservationV3::Reserved => None,
            RoventBoardWorkerRecoveryObservationV3::Busy => {
                Some(RoventBoardWorkerStartSourceErrorV3::Busy)
            }
            RoventBoardWorkerRecoveryObservationV3::Poisoned => {
                Some(RoventBoardWorkerStartSourceErrorV3::Poisoned)
            }
            _ => Some(RoventBoardWorkerStartSourceErrorV3::RegistryUnknown),
        };
        if let Some(error) = registry_error {
            return Err(RoventBoardWorkerStartSourceCancelFailureV3 { error, source });
        }
        self.start_seal = None;
        Ok(())
    }

    fn retain_unregistered(&mut self) {
        if self.phase == RoventBoardWorkerStagePhaseV3::Running && self.thread.is_none() {
            self.phase = RoventBoardWorkerStagePhaseV3::StartingUnknown;
        }
        let original = QuarantinedBoardWorkerStageV3 {
            key: self.key.clone(),
            admission: self.admission.clone(),
            sender: self.sender.clone(),
            received: Arc::clone(&self.received),
            receiver: self.receiver.take(),
            task: Arc::clone(&self.task),
            phase: self.phase,
            failure: self.failure.take(),
            start_seal: self.start_seal.take(),
        };
        // Registry loss/poison prevents keyed recovery. Preserve every
        // original component and report invariant/physical unknown instead.
        std::mem::forget((original, self.thread.take()));
    }

    /// No thread is started. The caller must retain this original stage and
    /// its key in supervisor custody before calling `start`.
    pub fn prepare(
        admission: RoventCollectionRootAdmissionV3,
    ) -> Result<Self, RoventBoardWorkerStagePrepareFailureV3> {
        let key = RoventBoardCollectionWorkerKeyV3(Arc::new(()));
        let (sender, receiver) = mpsc::sync_channel(1);
        let received = Arc::new(Mutex::new(None));
        let worker_receiver = RoventBoardRootHandoffReceiverV3 {
            receiver,
            admission: admission.clone(),
            received: Arc::clone(&received),
        };
        let task = Arc::new(Mutex::new(None));
        {
            let mut slots = match quarantined_board_workers_v3().lock() {
                Ok(slots) => slots,
                Err(_) => {
                    return Err(RoventBoardWorkerStagePrepareFailureV3 {
                        admission,
                        error: std::io::Error::other("original worker registry poisoned"),
                    });
                }
            };
            if let Err(error) = slots.try_reserve(1) {
                return Err(RoventBoardWorkerStagePrepareFailureV3 {
                    admission,
                    error: std::io::Error::other(error),
                });
            }
            slots.push(BoardWorkerSlotV3::Reserved(key.clone()));
        }
        Ok(Self {
            key,
            admission,
            sender,
            received,
            receiver: Some(worker_receiver),
            task,
            thread: None,
            phase: RoventBoardWorkerStagePhaseV3::Ready,
            failure: None,
            start_seal: None,
        })
    }

    pub fn recovery_key(&self) -> RoventBoardCollectionWorkerKeyV3 {
        self.key.clone()
    }
    pub fn phase(&self) -> RoventBoardWorkerStagePhaseV3 {
        self.phase
    }
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }

    /// Reclaim only the same dropped stage. StartingUnknown is deliberately
    /// not restartable: the platform may have started an unobserved thread.
    pub fn recover(key: &RoventBoardCollectionWorkerKeyV3) -> Option<Self> {
        let mut slots = quarantined_board_workers_v3().lock().ok()?;
        let index = slots.iter().position(|slot| slot.matches(key))?;
        if !matches!(&slots[index], BoardWorkerSlotV3::Staged(_)) {
            return None;
        }
        let BoardWorkerSlotV3::Staged(owner) =
            std::mem::replace(&mut slots[index], BoardWorkerSlotV3::Reserved(key.clone()))
        else {
            unreachable!()
        };
        Some(Self {
            key: owner.key,
            admission: owner.admission,
            sender: owner.sender,
            received: owner.received,
            receiver: owner.receiver,
            task: owner.task,
            thread: None,
            phase: owner.phase,
            failure: owner.failure,
            start_seal: owner.start_seal,
        })
    }

    /// Store the original task and mark StartingUnknown before invoking the
    /// platform. Only a returned Err proves no handle was returned; an unwind
    /// during spawn is retained as unknown. Once Ok returns, the very next
    /// operation installs the returned handle in this same stage.
    pub fn start(
        &mut self,
        task: RoventBoardRootWorkerTaskV3,
    ) -> Result<RoventBoardWorkerStageStartV3, RoventBoardWorkerStageStartRejectionV3> {
        if self.start_seal.is_some() {
            return Err(RoventBoardWorkerStageStartRejectionV3 {
                reason: "sealed stage requires its original start source",
                task,
            });
        }
        self.start_inner(task)
    }

    /// A result witness exists only when the actual platform spawn returned
    /// on this exact stage. Unwind leaves no witness and no retry authority.
    pub fn start_with_source(
        &mut self,
        source: &RoventBoardWorkerStartSourceV3,
        task: RoventBoardRootWorkerTaskV3,
    ) -> Result<RoventBoardWorkerStartOutcomeV3, RoventBoardWorkerStageStartRejectionV3> {
        if !self.matches_start_source(source) {
            return Err(RoventBoardWorkerStageStartRejectionV3 {
                reason: "different original start source",
                task,
            });
        }
        let result = self.start_inner(task)?;
        Ok(RoventBoardWorkerStartOutcomeV3 {
            seal: Arc::clone(&source.seal),
            key: source.key.clone(),
            admission: source.admission.clone(),
            received: Arc::clone(&source.received),
            result,
        })
    }

    fn start_inner(
        &mut self,
        task: RoventBoardRootWorkerTaskV3,
    ) -> Result<RoventBoardWorkerStageStartV3, RoventBoardWorkerStageStartRejectionV3> {
        if self.phase != RoventBoardWorkerStagePhaseV3::Ready || self.receiver.is_none() {
            return Err(RoventBoardWorkerStageStartRejectionV3 {
                reason: "original stage is not ready",
                task,
            });
        }
        match self.key.observe() {
            RoventBoardWorkerRecoveryObservationV3::Reserved => {}
            RoventBoardWorkerRecoveryObservationV3::Busy => {
                return Err(RoventBoardWorkerStageStartRejectionV3 {
                    reason: "original worker registry busy",
                    task,
                });
            }
            _ => {
                return Err(RoventBoardWorkerStageStartRejectionV3 {
                    reason: "original worker registry unknown",
                    task,
                });
            }
        }
        {
            let mut slot = match self.task.lock() {
                Ok(slot) => slot,
                Err(_) => {
                    self.phase = RoventBoardWorkerStagePhaseV3::StartingUnknown;
                    return Err(RoventBoardWorkerStageStartRejectionV3 {
                        reason: "original task custody poisoned",
                        task,
                    });
                }
            };
            if slot.is_some() {
                return Err(RoventBoardWorkerStageStartRejectionV3 {
                    reason: "original task already retained",
                    task,
                });
            }
            *slot = Some(task);
        }
        let worker_task = Arc::clone(&self.task);
        self.phase = RoventBoardWorkerStagePhaseV3::StartingUnknown;
        let receiver = self.receiver.take().expect("checked original receiver");
        let started = std::thread::Builder::new()
            .name("rovent-own-collection-v3".to_owned())
            .spawn(move || {
                let task = match worker_task.lock() {
                    Ok(mut slot) => match slot.take() {
                        Some(task) => task,
                        None => {
                            return Box::new("original staged task unavailable")
                                as Box<dyn Any + Send>;
                        }
                    },
                    Err(_) => {
                        return Box::new("original staged task custody poisoned")
                            as Box<dyn Any + Send>;
                    }
                };
                task(receiver)
            });
        match started {
            Ok(thread) => {
                self.thread = Some(thread);
                self.phase = RoventBoardWorkerStagePhaseV3::Running;
                Ok(RoventBoardWorkerStageStartV3::Started)
            }
            Err(error) => {
                self.failure = Some(error);
                self.phase = RoventBoardWorkerStagePhaseV3::Failed;
                Ok(RoventBoardWorkerStageStartV3::Failed)
            }
        }
    }

    /// An actual spawn Err preserves both the error and the original task.
    pub fn take_failed_start(&mut self) -> Option<(std::io::Error, RoventBoardRootWorkerTaskV3)> {
        if self.phase != RoventBoardWorkerStagePhaseV3::Failed {
            return None;
        }
        let mut slot = self.task.lock().ok()?;
        if slot.is_none() || self.failure.is_none() {
            return None;
        }
        let task = slot.take().expect("checked failed task");
        let failure = self.failure.take().expect("checked failed spawn error");
        self.phase = RoventBoardWorkerStagePhaseV3::FailedTaken;
        Some((failure, task))
    }

    /// A task stored before entering spawn can be reclaimed only while the
    /// exact stage remains Ready with its original receiver.
    pub fn take_unstarted_task(&mut self) -> Option<RoventBoardRootWorkerTaskV3> {
        if self.phase != RoventBoardWorkerStagePhaseV3::Ready || self.receiver.is_none() {
            return None;
        }
        self.task.lock().ok()?.take()
    }

    /// Move the original installed handle to the legacy worker owner. The
    /// worker's Drop quarantines it by the already published key.
    pub fn take_worker(&mut self) -> Option<RoventBoardCollectionWorkerV3> {
        if self.phase != RoventBoardWorkerStagePhaseV3::Running {
            return None;
        }
        let key = self.key.clone();
        let admission = self.admission.clone();
        let sender = self.sender.clone();
        let received = Arc::clone(&self.received);
        let task_custody = Arc::clone(&self.task);
        let thread = self.thread.take()?;
        let worker = RoventBoardCollectionWorkerV3 {
            key,
            admission,
            sender,
            received,
            task_custody,
            thread: Some(thread),
        };
        self.phase = RoventBoardWorkerStagePhaseV3::Transferred;
        Some(worker)
    }
}

impl Drop for RoventBoardCollectionWorkerStageV3 {
    fn drop(&mut self) {
        if self.phase == RoventBoardWorkerStagePhaseV3::Transferred {
            return;
        }
        let has_task = match self.task.lock() {
            Ok(slot) => slot.is_some(),
            Err(_) => {
                self.phase = RoventBoardWorkerStagePhaseV3::StartingUnknown;
                true
            }
        };
        let mut slots = match quarantined_board_workers_v3().lock() {
            Ok(slots) => slots,
            Err(_) => {
                self.retain_unregistered();
                return;
            }
        };
        let Some(index) = slots.iter().position(|slot| slot.matches(&self.key)) else {
            self.retain_unregistered();
            return;
        };
        if !matches!(&slots[index], BoardWorkerSlotV3::Reserved(_)) {
            self.retain_unregistered();
            return;
        }
        if let Some(thread) = self.thread.take() {
            slots[index] = BoardWorkerSlotV3::Quarantined(QuarantinedBoardWorkerV3 {
                key: self.key.clone(),
                admission: self.admission.clone(),
                sender: self.sender.clone(),
                received: Arc::clone(&self.received),
                task_custody: Arc::clone(&self.task),
                thread,
            });
        } else if has_task
            || self.start_seal.is_some()
            || matches!(
                self.phase,
                RoventBoardWorkerStagePhaseV3::StartingUnknown
                    | RoventBoardWorkerStagePhaseV3::Failed
                    | RoventBoardWorkerStagePhaseV3::Running
            )
            || (self.phase == RoventBoardWorkerStagePhaseV3::Ready && self.receiver.is_none())
        {
            if self.phase == RoventBoardWorkerStagePhaseV3::Running
                || (self.phase == RoventBoardWorkerStagePhaseV3::Ready && self.receiver.is_none())
            {
                self.phase = RoventBoardWorkerStagePhaseV3::StartingUnknown;
            }
            slots[index] = BoardWorkerSlotV3::Staged(QuarantinedBoardWorkerStageV3 {
                key: self.key.clone(),
                admission: self.admission.clone(),
                sender: self.sender.clone(),
                received: Arc::clone(&self.received),
                receiver: self.receiver.take(),
                task: Arc::clone(&self.task),
                phase: self.phase,
                failure: self.failure.take(),
                start_seal: self.start_seal.take(),
            });
        } else {
            slots.swap_remove(index);
        }
    }
}

impl RoventBoardCollectionWorkerV3 {
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }

    /// Historical direct path. It does not publish a recovery key before
    /// spawn; new supervisor integrations must use the staged owner instead.
    pub fn spawn(
        admission: RoventCollectionRootAdmissionV3,
        task: RoventBoardRootWorkerTaskV3,
    ) -> Result<Self, RoventBoardCollectionWorkerSpawnFailureV3> {
        let key = RoventBoardCollectionWorkerKeyV3(Arc::new(()));
        {
            let mut slots = match quarantined_board_workers_v3().lock() {
                Ok(slots) => slots,
                Err(_) => {
                    return Err(RoventBoardCollectionWorkerSpawnFailureV3 {
                        admission,
                        error: std::io::Error::other("original worker registry poisoned"),
                        task,
                    });
                }
            };
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
                task_custody: slot,
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
        let mut workers = quarantined_board_workers_v3().lock().ok()?;
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
            task_custody: owner.task_custody,
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
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(RoventBoardCollectionWorkerJoinFailureV3::CustodyPoisoned);
            }
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
        match self.task_custody.try_lock() {
            Ok(task) if task.is_none() => {}
            Ok(_) | Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(RoventBoardCollectionWorkerJoinFailureV3::TaskCustodyUnknown);
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(RoventBoardCollectionWorkerJoinFailureV3::Busy);
            }
        }
        let received_handoff = match self.received.try_lock() {
            Ok(guard) => guard
                .as_ref()
                .is_some_and(|received| Arc::ptr_eq(received, handoff)),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(RoventBoardCollectionWorkerJoinFailureV3::CustodyPoisoned);
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(RoventBoardCollectionWorkerJoinFailureV3::Busy);
            }
        };
        let mut slots = match quarantined_board_workers_v3().try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(RoventBoardCollectionWorkerJoinFailureV3::RegistryPoisoned);
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(RoventBoardCollectionWorkerJoinFailureV3::Busy);
            }
        };
        let Some(index) = slots.iter().position(|slot| slot.matches(&self.key)) else {
            return Err(RoventBoardCollectionWorkerJoinFailureV3::RegistryUnknown);
        };
        if !matches!(&slots[index], BoardWorkerSlotV3::Reserved(_)) {
            return Err(RoventBoardCollectionWorkerJoinFailureV3::RegistryUnknown);
        }
        slots.swap_remove(index);
        drop(slots);
        // Do not hold the shared worker registry during OS thread join; even
        // a finished thread can still be completing thread-local teardown.
        let result = self.thread.take().expect("checked V3 worker handle").join();
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
                        self.key.clone(),
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
            let owner = QuarantinedBoardWorkerV3 {
                key: self.key.clone(),
                admission: self.admission.clone(),
                sender: self.sender.clone(),
                received: Arc::clone(&self.received),
                task_custody: Arc::clone(&self.task_custody),
                thread,
            };
            let mut slots = match quarantined_board_workers_v3().lock() {
                Ok(slots) => slots,
                Err(_) => {
                    std::mem::forget(owner);
                    return;
                }
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
    /// Native physical finish is distinct from teacher/capture settlement.
    /// None retains all original hook custody for a later retry.
    fn try_finish_launch(
        &self,
        _native: &crate::rovent_bootstrap::launch::RoventOwnCollectionNativeFenceV3,
    ) -> Result<Option<Box<dyn Any + Send>>, AdaptiveManagerError> {
        Ok(None)
    }
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
    /// Preserve the exact failed bindings after the same manager's root
    /// fence, before any persistent owner transfer can be attempted.
    pub(super) fn retain_failed_evidence(
        &self,
        manager: &mut AdaptiveSessionManager<B>,
    ) -> Result<(), AdaptiveManagerError> {
        if let Some(error) = manager.first_v2_collection_failure() {
            let mut first =
                self.first_collection_error
                    .try_lock()
                    .map_err(|error| match error {
                        std::sync::TryLockError::WouldBlock => {
                            AdaptiveManagerError::AuxiliaryPending
                        }
                        std::sync::TryLockError::Poisoned(_) => AdaptiveManagerError::Invalid(
                            "V3 own collection error custody poisoned",
                        ),
                    })?;
            if first.is_none() {
                *first = Some(error.clone());
            }
        }
        let mut slot = self
            .failed_evidence
            .try_lock()
            .map_err(|error| match error {
                std::sync::TryLockError::WouldBlock => AdaptiveManagerError::AuxiliaryPending,
                std::sync::TryLockError::Poisoned(_) => {
                    AdaptiveManagerError::Invalid("V3 own failed evidence custody poisoned")
                }
            })?;
        if slot.is_some() {
            return Ok(());
        }
        let original = manager.take_failed_collection_evidence()?;
        if !original.is_empty() {
            *slot = Some(original);
        }
        Ok(())
    }

    pub(super) fn remember_session_failure(&self, failure: &SearchSessionFailure) {
        let mut first = self
            .first_session_failure
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if first.is_none() {
            *first = Some(failure.clone());
        }
    }

    /// Retain the original root frame and its immutable shutdown context for a
    /// later same-manager recovery. A previous unresolved frame is never
    /// replaced by a newer root.
    pub(super) fn retain_root(
        &self,
        frame: &mut Option<super::collection_v3_root::OwnCollectionRootRunV3>,
        context: &SearchSessionContext,
    ) -> Result<(), AdaptiveManagerError> {
        let mut retained = self.recovery_root.lock().unwrap_or_else(|p| p.into_inner());
        let mut stored_context = self
            .cleanup_context
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if retained.is_some() || stored_context.is_some() {
            return Err(AdaptiveManagerError::Invalid(
                "V3 own recovery frame already retained",
            ));
        }
        if frame.is_some() {
            *stored_context = Some(context.clone());
            *retained = frame.take();
        }
        Ok(())
    }

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
    /// Transfer the exact failed bindings to a caller that will preserve
    /// them. This transfer grants no collection or physical-success status.
    pub(crate) fn take_board_failed_evidence_v3(
        &self,
    ) -> Result<
        Option<
            Vec<
                super::super::manager::RoventFailedCollectionEvidence<
                    AdaptivePortQuestion,
                    B::Output,
                >,
            >,
        >,
        SearchSessionFailure,
    > {
        let Some(collection) = self.rovent_collection_v3.as_ref() else {
            return Ok(None);
        };
        collection
            .failed_evidence
            .try_lock()
            .map(|mut slot| slot.take())
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())
    }

    /// Original first root error is diagnostic and remains distinct from a
    /// later successful physical cleanup of this same session.
    pub fn board_collection_first_execution_error_v3(
        &self,
    ) -> Result<Option<AdaptiveManagerError>, SearchSessionFailure> {
        let Some(collection) = self.rovent_collection_v3.as_ref() else {
            return Ok(None);
        };
        collection
            .first_execution_error
            .try_lock()
            .map(|error| error.clone())
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())
    }

    pub fn board_collection_first_session_failure_v3(
        &self,
    ) -> Result<Option<SearchSessionFailure>, SearchSessionFailure> {
        let Some(collection) = self.rovent_collection_v3.as_ref() else {
            return Ok(None);
        };
        collection
            .first_session_failure
            .try_lock()
            .map(|failure| failure.clone())
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())
    }

    pub fn board_collection_first_observation_error_v3(
        &self,
    ) -> Result<Option<AdaptiveCollectionError>, SearchSessionFailure> {
        let Some(collection) = self.rovent_collection_v3.as_ref() else {
            return Ok(None);
        };
        collection
            .first_collection_error
            .try_lock()
            .map(|error| error.clone())
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())
    }

    /// Reobserve only the session's retained V3 root and manager. This is one
    /// nonblocking cleanup step; a false result remains physically unknown.
    pub(crate) fn recover_rovent_collection_v3(&self) -> Result<bool, SearchSessionFailure> {
        let Some(collection) = self.rovent_collection_v3.as_ref() else {
            return Ok(true);
        };
        let mut session = self
            .state
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        let mut retained = collection
            .recovery_root
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        let mut saved_context = collection
            .cleanup_context
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        let Some(frame) = retained.as_mut() else {
            return Ok(!collection.outstanding.load(Ordering::Acquire) && saved_context.is_none());
        };
        let context = saved_context
            .as_ref()
            .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
        let SessionState::Manager(manager) = &mut *session else {
            return Err(SearchSessionFailure::physical_completion_unknown());
        };
        let result = catch_unwind(AssertUnwindSafe(
            || -> Result<bool, SearchSessionFailure> {
                match frame.revoke_tick(collection) {
                    Ok(()) => {}
                    Err(AdaptiveManagerError::AuxiliaryPending) => return Ok(false),
                    Err(error) => {
                        return Err(SearchSessionFailure::debug(
                            "RoventOwnCollectionRevokeV3",
                            &error,
                        )
                        .unknown_if_unclosed());
                    }
                }
                if manager.phase() == ManagerPhase::Running {
                    manager
                        .begin_root_quiesce(context.shutdown_deadline)
                        .map_err(|error| {
                            SearchSessionFailure::debug("RoventOwnCollectionDrainV3", &error)
                                .unknown_if_unclosed()
                        })?;
                }
                if manager.phase() != ManagerPhase::Finished {
                    manager.tick().map_err(|error| {
                        SearchSessionFailure::debug("RoventOwnCollectionDrainV3", &error)
                            .unknown_if_unclosed()
                    })?;
                }
                if manager.phase() != ManagerPhase::Finished {
                    return Ok(false);
                }
                collection
                    .retain_failed_evidence(manager)
                    .map_err(|error| {
                        SearchSessionFailure::debug("RoventOwnCollectionFailedEvidenceV3", &error)
                            .unknown_if_unclosed()
                    })?;
                match frame.settle_tick(collection, manager) {
                    Ok(true) => Ok(true),
                    Ok(false) | Err(AdaptiveManagerError::AuxiliaryPending) => Ok(false),
                    Err(error) => Err(SearchSessionFailure::debug(
                        "RoventOwnCollectionSettlementV3",
                        &error,
                    )
                    .unknown_if_unclosed()),
                }
            },
        ));
        match result {
            Ok(Ok(true)) => {
                retained.take();
                saved_context.take();
                Ok(true)
            }
            Ok(Err(failure)) => {
                collection.remember_session_failure(&failure);
                Err(failure)
            }
            Ok(result) => result,
            Err(payload) => {
                collection
                    .panics
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(payload);
                let failure = SearchSessionFailure::physical_completion_unknown();
                collection.remember_session_failure(&failure);
                Err(failure)
            }
        }
    }

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
            first_execution_error: Mutex::new(None),
            first_session_failure: Mutex::new(None),
            first_collection_error: Mutex::new(None),
            panics: Mutex::new(Vec::new()),
            handoff: Mutex::new(None),
            custody: Mutex::new(None),
            no_capture: Mutex::new(None),
            deferred_cleanup: Mutex::new(None),
            prior_cleanup_failures: Mutex::new(Vec::new()),
            recovery_root: Mutex::new(None),
            cleanup_context: Mutex::new(None),
            failed_evidence: Mutex::new(None),
        });
        Ok(())
    }
}
