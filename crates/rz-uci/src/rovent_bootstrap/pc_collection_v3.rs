//! Original V3 P/C collection owners. No V2 admission or receipt is recreated.

use super::RoventStartupControl;
use super::pc_session::RoventPcSessionManager;
use crate::pals_adaptive::{
    AdaptiveCollectionError, AdaptiveCollectionObserverBoxV3, AdaptiveManagerError,
    RoleOnlyCollectionV3InstallAttempt, RoleOnlyCollectionV3Rejection, RoleOnlyShutdownReceipt,
    RoventCollectionRootAdmissionV3, RoventNativeCollectionOriginV3,
    RoventNoCaptureRootCapabilityV3, RoventRoleOnlyNoCaptureFenceV3,
    RoventTeacherScopePhysicalAckV3,
};
use crate::pals_native::ticketed::VirtualReviewTicketOutput;
use crate::rovent_external_launch::RoventTeacherScopeAggregateAckV3;
use rz_search::pals::engine::pc_port::PcPortQuestion;
use std::{
    any::Any,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
};

/// Internal custody remains with the original P/C session. A reported root
/// closure alone does not release a pending teacher or an unresolved marker.
pub(crate) struct RoventPcCollectionStateV3 {
    pub(crate) origin: RoventNativeCollectionOriginV3,
    pub(crate) hooks: Arc<dyn RoventPcCollectionHooksV3>,
    pub(crate) outstanding: AtomicBool,
    pub(crate) handoff: Mutex<Option<Arc<Mutex<RoventPcCollectionHandoffV3>>>>,
    pub(crate) custody: Mutex<Option<Arc<Mutex<RoventPcCollectionRootCustodyV3>>>>,
    pub(crate) no_capture:
        Mutex<Option<(RoventNoCaptureRootCapabilityV3, AdaptiveCollectionError)>>,
    pub(crate) deferred_cleanup: Mutex<Option<Arc<Mutex<RoventPcDeferredCleanupV3>>>>,
    pub(crate) prior_cleanup_failures: Mutex<Vec<Arc<Mutex<RoventPcDeferredCleanupV3>>>>,
    pub(crate) cleanup_context: Mutex<Option<crate::search_driver::SearchSessionContext>>,
    pub(crate) recovery_root: Mutex<Option<super::pc_collection_v3_root::PcCollectionRootRunV3>>,
    pub(crate) admission_failure: Mutex<Option<RoventPcCollectionAdmissionFailureV3>>,
    panics: Mutex<Vec<Box<dyn Any + Send>>>,
}

/// Diagnostic snapshot only; this does not grant release or completion authority.
pub(crate) enum RoventPcCleanupObservationV3 {
    NoFailureObserved,
    FirstFailure {
        kind: &'static str,
        detail: Option<String>,
        recovered: bool,
    },
    Unknown,
}

impl RoventPcCollectionStateV3 {
    pub(crate) fn new(
        origin: RoventNativeCollectionOriginV3,
        hooks: Arc<dyn RoventPcCollectionHooksV3>,
    ) -> Self {
        Self {
            origin,
            hooks,
            outstanding: AtomicBool::new(false),
            handoff: Mutex::new(None),
            custody: Mutex::new(None),
            no_capture: Mutex::new(None),
            deferred_cleanup: Mutex::new(None),
            prior_cleanup_failures: Mutex::new(Vec::new()),
            cleanup_context: Mutex::new(None),
            recovery_root: Mutex::new(None),
            admission_failure: Mutex::new(None),
            panics: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn retain_panic(&self, payload: Box<dyn Any + Send>) {
        let mut panics = self.panics.lock().unwrap_or_else(|p| p.into_inner());
        if panics.try_reserve(1).is_ok() {
            panics.push(payload);
        } else {
            // Never drop an unclassified original owner on allocation failure.
            std::mem::forget(payload);
        }
    }

    pub(crate) fn cleanup_observation(&self) -> RoventPcCleanupObservationV3 {
        // Match the cleanup writer's lock order and never stall the product.
        let Ok(active) = self.deferred_cleanup.try_lock() else {
            return RoventPcCleanupObservationV3::Unknown;
        };
        let Ok(history) = self.prior_cleanup_failures.try_lock() else {
            return RoventPcCleanupObservationV3::Unknown;
        };
        let (first, recovered) = match history.first() {
            Some(first) => (first, true),
            None => match active.as_ref() {
                Some(first) => (first, false),
                None => return RoventPcCleanupObservationV3::NoFailureObserved,
            },
        };
        let Ok(first) = first.try_lock() else {
            return RoventPcCleanupObservationV3::Unknown;
        };
        let (kind, detail) = match first.failure() {
            RoventPcDeferredCleanupFailureV3::Busy => ("busy", None),
            RoventPcDeferredCleanupFailureV3::Error(error) => (
                "error",
                Some(
                    crate::search_driver::SearchSessionFailure::debug("RoventPcV3Cleanup", error)
                        .detail,
                ),
            ),
            RoventPcDeferredCleanupFailureV3::Panic(payload) => {
                let message = payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| payload.downcast_ref::<&'static str>().copied());
                (
                    "panic",
                    message.map(|message| {
                        crate::search_driver::SearchSessionFailure::debug(
                            "RoventPcV3CleanupPanic",
                            &message,
                        )
                        .detail
                    }),
                )
            }
        };
        RoventPcCleanupObservationV3::FirstFailure {
            kind,
            detail,
            recovered,
        }
    }
}

/// This handoff is constructed only after the same manager's root fence and
/// settlement. Teacher completion and corpus publication remain separate.
pub struct RoventPcCollectionHandoffV3 {
    pub(crate) admission: RoventCollectionRootAdmissionV3,
    pub(crate) observer: AdaptiveCollectionObserverBoxV3<PcPortQuestion, VirtualReviewTicketOutput>,
    pub(crate) settlement: Result<(), AdaptiveCollectionError>,
    pub(crate) native_fence: RoleOnlyShutdownReceipt,
    teacher_physical_ack: Option<RoventTeacherScopePhysicalAckV3>,
    teacher_aggregate_ack: Option<RoventTeacherScopeAggregateAckV3>,
}

impl RoventPcCollectionHandoffV3 {
    pub(crate) fn new(
        admission: RoventCollectionRootAdmissionV3,
        observer: AdaptiveCollectionObserverBoxV3<PcPortQuestion, VirtualReviewTicketOutput>,
        settlement: Result<(), AdaptiveCollectionError>,
        native_fence: RoleOnlyShutdownReceipt,
    ) -> Self {
        Self {
            admission,
            observer,
            settlement,
            native_fence,
            teacher_physical_ack: None,
            teacher_aggregate_ack: None,
        }
    }

    pub fn submit_teacher_physical_ack(
        &mut self,
        ack: RoventTeacherScopePhysicalAckV3,
    ) -> Result<(), RoventTeacherScopePhysicalAckV3> {
        if self.teacher_physical_ack.is_some() || !ack.matches_admission(&self.admission) {
            return Err(ack);
        }
        self.teacher_physical_ack = Some(ack);
        Ok(())
    }

    pub fn submit_teacher_aggregate_ack(
        &mut self,
        ack: RoventTeacherScopeAggregateAckV3,
    ) -> Result<(), RoventTeacherScopeAggregateAckV3> {
        if self.teacher_aggregate_ack.is_some() || !ack.matches_admission(&self.admission) {
            return Err(ack);
        }
        self.teacher_aggregate_ack = Some(ack);
        Ok(())
    }

    pub fn teacher_aggregate_ack(&self) -> Option<&RoventTeacherScopeAggregateAckV3> {
        self.teacher_aggregate_ack.as_ref()
    }

    pub fn parts_mut(
        &mut self,
    ) -> (
        &RoventCollectionRootAdmissionV3,
        &mut AdaptiveCollectionObserverBoxV3<PcPortQuestion, VirtualReviewTicketOutput>,
        &Result<(), AdaptiveCollectionError>,
        &RoleOnlyShutdownReceipt,
    ) {
        (
            &self.admission,
            &mut self.observer,
            &self.settlement,
            &self.native_fence,
        )
    }

    pub fn into_parts(
        self,
    ) -> (
        RoventCollectionRootAdmissionV3,
        AdaptiveCollectionObserverBoxV3<PcPortQuestion, VirtualReviewTicketOutput>,
        Result<(), AdaptiveCollectionError>,
        RoleOnlyShutdownReceipt,
        Option<RoventTeacherScopePhysicalAckV3>,
        Option<RoventTeacherScopeAggregateAckV3>,
    ) {
        (
            self.admission,
            self.observer,
            self.settlement,
            self.native_fence,
            self.teacher_physical_ack,
            self.teacher_aggregate_ack,
        )
    }
}

/// The original joined orchestration worker and the same root handoff.
/// Teacher physical and aggregate completion have separate evidence.
pub struct RoventPcCollectionRootCustodyV3 {
    handoff: Arc<Mutex<RoventPcCollectionHandoffV3>>,
    original_worker: Box<dyn Any + Send>,
}

/// An unresolved no-capture marker cleanup, retained with its exact root and
/// native fence. This is not evidence of marker removal or teacher retirement.
pub struct RoventPcDeferredCleanupV3 {
    pub(crate) capability: RoventNoCaptureRootCapabilityV3,
    pub(crate) reason: AdaptiveCollectionError,
    pub(crate) fence: RoventRoleOnlyNoCaptureFenceV3,
    pub(crate) failure: RoventPcDeferredCleanupFailureV3,
    pub(crate) retention_failure: Option<AdaptiveManagerError>,
}

pub enum RoventPcDeferredCleanupFailureV3 {
    Busy,
    Error(AdaptiveManagerError),
    Panic(Box<dyn Any + Send>),
}

impl RoventPcDeferredCleanupV3 {
    pub fn capability(&self) -> &RoventNoCaptureRootCapabilityV3 {
        &self.capability
    }
    pub fn reason(&self) -> &AdaptiveCollectionError {
        &self.reason
    }
    pub fn fence(&self) -> &RoventRoleOnlyNoCaptureFenceV3 {
        &self.fence
    }
    pub fn failure(&self) -> &RoventPcDeferredCleanupFailureV3 {
        &self.failure
    }
    pub fn retention_failure(&self) -> Option<&AdaptiveManagerError> {
        self.retention_failure.as_ref()
    }
}
impl RoventPcCollectionRootCustodyV3 {
    fn from_joined_worker(
        handoff: Arc<Mutex<RoventPcCollectionHandoffV3>>,
        original_worker: Box<dyn Any + Send>,
    ) -> Self {
        Self {
            handoff,
            original_worker,
        }
    }
    pub fn matches_handoff(&self, handoff: &Arc<Mutex<RoventPcCollectionHandoffV3>>) -> bool {
        Arc::ptr_eq(&self.handoff, handoff)
    }
    pub fn handoff(&self) -> &Arc<Mutex<RoventPcCollectionHandoffV3>> {
        &self.handoff
    }
    pub fn original_worker(&self) -> &(dyn Any + Send) {
        self.original_worker.as_ref()
    }
    pub fn into_parts(self) -> (Arc<Mutex<RoventPcCollectionHandoffV3>>, Box<dyn Any + Send>) {
        (self.handoff, self.original_worker)
    }
}

pub type RootHandoffV3 = Arc<Mutex<RoventPcCollectionHandoffV3>>;
/// Receiver records the exact same-root handoff actually delivered to worker.
pub struct RoventPcRootHandoffReceiverV3 {
    receiver: Option<mpsc::Receiver<RootHandoffV3>>,
    admission: RoventCollectionRootAdmissionV3,
    received: Arc<Mutex<Option<RootHandoffV3>>>,
    receive_state: Arc<PcHandoffReceiveStateV3>,
}
struct PcHandoffReceiveStateV3 {
    received_any: AtomicBool,
    receiver_dropped: AtomicBool,
}
impl Drop for RoventPcRootHandoffReceiverV3 {
    fn drop(&mut self) {
        // The marker is published only after the original receiver and any
        // queued handoffs have finished dropping. A panic leaves it unknown.
        drop(self.receiver.take());
        self.receive_state
            .receiver_dropped
            .store(true, Ordering::Release);
    }
}
impl RoventPcRootHandoffReceiverV3 {
    pub fn recv_timeout(
        &self,
        timeout: std::time::Duration,
    ) -> Result<Option<RootHandoffV3>, AdaptiveManagerError> {
        let handoff = match self
            .receiver
            .as_ref()
            .expect("live V3 receiver")
            .recv_timeout(timeout)
        {
            Ok(value) => value,
            Err(mpsc::RecvTimeoutError::Timeout) => return Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(AdaptiveManagerError::Invalid(
                    "V3 root handoff channel closed",
                ));
            }
        };
        // Record even an invalid handoff before inspecting it: it was received.
        self.receive_state
            .received_any
            .store(true, Ordering::Release);
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
pub type RoventPcRootWorkerTaskV3 =
    Box<dyn FnOnce(RoventPcRootHandoffReceiverV3) -> Box<dyn Any + Send> + Send + 'static>;

/// Owns the exact orchestration thread and admission. A worker result is not
/// evidence of teacher physical completion or corpus publication.
#[must_use]
pub struct RoventPcCollectionWorkerV3 {
    key: RoventPcCollectionWorkerKeyV3,
    admission: RoventCollectionRootAdmissionV3,
    sender: mpsc::SyncSender<RootHandoffV3>,
    received: Arc<Mutex<Option<RootHandoffV3>>>,
    receive_state: Arc<PcHandoffReceiveStateV3>,
    thread: Option<std::thread::JoinHandle<Box<dyn Any + Send>>>,
}

/// The owner retains this token to recover an unresolved worker from the
/// process-local quarantine. It does not imply completion or authorization.
#[derive(Clone)]
pub struct RoventPcCollectionWorkerKeyV3(Arc<()>);
impl RoventPcCollectionWorkerKeyV3 {
    /// Same original worker identity only; this is not a completion proof.
    pub fn same_original(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
struct QuarantinedPcWorkerV3 {
    key: RoventPcCollectionWorkerKeyV3,
    admission: RoventCollectionRootAdmissionV3,
    sender: mpsc::SyncSender<RootHandoffV3>,
    received: Arc<Mutex<Option<RootHandoffV3>>>,
    receive_state: Arc<PcHandoffReceiveStateV3>,
    thread: std::thread::JoinHandle<Box<dyn Any + Send>>,
}
enum PcWorkerSlotV3 {
    Reserved(RoventPcCollectionWorkerKeyV3),
    Quarantined(QuarantinedPcWorkerV3),
}
impl PcWorkerSlotV3 {
    fn matches(&self, key: &RoventPcCollectionWorkerKeyV3) -> bool {
        let owner = match self {
            Self::Reserved(owner) => owner,
            Self::Quarantined(owner) => &owner.key,
        };
        Arc::ptr_eq(&owner.0, &key.0)
    }
}
fn quarantined_pc_workers_v3() -> &'static Mutex<Vec<PcWorkerSlotV3>> {
    static WORKERS: OnceLock<Mutex<Vec<PcWorkerSlotV3>>> = OnceLock::new();
    WORKERS.get_or_init(|| Mutex::new(Vec::new()))
}
fn release_pc_worker_slot_v3(key: &RoventPcCollectionWorkerKeyV3) {
    let mut slots = quarantined_pc_workers_v3()
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    if let Some(index) = slots.iter().position(|slot| slot.matches(key)) {
        slots.swap_remove(index);
    }
}

pub struct RoventPcCollectionWorkerSpawnFailureV3 {
    pub admission: RoventCollectionRootAdmissionV3,
    pub error: std::io::Error,
    pub task: RoventPcRootWorkerTaskV3,
}

/// Failure details retained separately from a task that may hold the
/// recovery owner. This value is not, on its own, proof of a spawn attempt:
/// the existing public failure can also be assembled by a caller. Only the
/// same producer's original result slot can establish its origin.
pub struct RoventPcCollectionWorkerSpawnFailureDetailsV3 {
    admission: RoventCollectionRootAdmissionV3,
    error: std::io::Error,
}

impl RoventPcCollectionWorkerSpawnFailureDetailsV3 {
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }

    pub fn error(&self) -> &std::io::Error {
        &self.error
    }
}

impl RoventPcCollectionWorkerSpawnFailureV3 {
    /// Separate the retained diagnostics from the original task without
    /// invoking or dropping it. The caller must explicitly dispose of or
    /// retain the task outside any owner lock and account for a failing drop.
    /// This decomposition does not prove that task disposal completed.
    pub fn into_details_and_task(
        self,
    ) -> (
        RoventPcCollectionWorkerSpawnFailureDetailsV3,
        RoventPcRootWorkerTaskV3,
    ) {
        let Self {
            admission,
            error,
            task,
        } = self;
        (
            RoventPcCollectionWorkerSpawnFailureDetailsV3 { admission, error },
            task,
        )
    }
}

pub enum RoventPcCollectionWorkerJoinFailureV3 {
    WrongRoot,
    AlreadyJoined,
    OwnerLockPoisoned,
    WorkerSlotMissing,
    Panicked(Box<dyn Any + Send>),
    CompletedWithoutHandoff(Box<dyn Any + Send>),
}

/// Only the original orchestration thread joined successfully and its exact
/// receiver was dropped without receiving any handoff. This is not teacher,
/// native, GPU, route, or collection completion evidence.
pub struct RoventPcCollectionNoHandoffCustodyV3 {
    key: RoventPcCollectionWorkerKeyV3,
    admission: RoventCollectionRootAdmissionV3,
    original_worker: Box<dyn Any + Send>,
}

impl RoventPcCollectionNoHandoffCustodyV3 {
    pub fn matches_worker(&self, key: &RoventPcCollectionWorkerKeyV3) -> bool {
        self.key.same_original(key)
    }

    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }

    pub fn into_original_worker(self) -> Box<dyn Any + Send> {
        self.original_worker
    }
}

pub enum RoventPcCollectionNoHandoffJoinFailureV3 {
    AlreadyJoined,
    HandoffReceived,
    OwnerLockPoisoned,
    WorkerSlotMissing,
    Panicked(Box<dyn Any + Send>),
}

impl RoventPcCollectionWorkerV3 {
    /// Identity only; this does not attest worker or teacher completion.
    pub fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.admission.same_original_admission(admission)
    }

    /// Store this original spawn result in the caller's already-owned slot
    /// before returning. A full slot rejects the attempt without spawning and
    /// returns the original admission and task to their caller. A stored
    /// worker or failure proves only the spawn result, not physical completion.
    /// If `spawn` unwinds, callers must retain the unresolved claim as unknown.
    pub fn spawn_into(
        admission: RoventCollectionRootAdmissionV3,
        task: RoventPcRootWorkerTaskV3,
        destination: &mut Option<Result<Self, RoventPcCollectionWorkerSpawnFailureV3>>,
    ) -> Result<(), (RoventCollectionRootAdmissionV3, RoventPcRootWorkerTaskV3)> {
        if destination.is_some() {
            return Err((admission, task));
        }
        *destination = Some(Self::spawn(admission, task));
        Ok(())
    }

    pub fn spawn(
        admission: RoventCollectionRootAdmissionV3,
        task: RoventPcRootWorkerTaskV3,
    ) -> Result<Self, RoventPcCollectionWorkerSpawnFailureV3> {
        let key = RoventPcCollectionWorkerKeyV3(Arc::new(()));
        {
            let mut slots = quarantined_pc_workers_v3()
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            if let Err(error) = slots.try_reserve(1) {
                return Err(RoventPcCollectionWorkerSpawnFailureV3 {
                    admission,
                    error: std::io::Error::other(error),
                    task,
                });
            }
            // Reserve before spawning, so quarantine on Drop does not allocate.
            slots.push(PcWorkerSlotV3::Reserved(key.clone()));
        }
        let (sender, receiver) = mpsc::sync_channel(1);
        let received = Arc::new(Mutex::new(None));
        let receive_state = Arc::new(PcHandoffReceiveStateV3 {
            received_any: AtomicBool::new(false),
            receiver_dropped: AtomicBool::new(false),
        });
        let worker_receiver = RoventPcRootHandoffReceiverV3 {
            receiver: Some(receiver),
            admission: admission.clone(),
            received: Arc::clone(&received),
            receive_state: Arc::clone(&receive_state),
        };
        // A failed thread creation must return the original task and its owners.
        let slot = Arc::new(Mutex::new(Some(task)));
        let worker_slot = Arc::clone(&slot);
        let thread = std::thread::Builder::new()
            .name("rovent-pc-collection-v3".to_owned())
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
                receive_state,
                thread: Some(thread),
            }),
            Err(error) => {
                release_pc_worker_slot_v3(&key);
                let task = slot
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .take()
                    .expect("failed spawn retains V3 worker task");
                Err(RoventPcCollectionWorkerSpawnFailureV3 {
                    admission,
                    error,
                    task,
                })
            }
        }
    }

    pub fn sender(&self) -> mpsc::SyncSender<RootHandoffV3> {
        self.sender.clone()
    }

    pub fn recovery_key(&self) -> RoventPcCollectionWorkerKeyV3 {
        self.key.clone()
    }

    /// Recover only the original process-local worker matching this token.
    pub fn recover(key: &RoventPcCollectionWorkerKeyV3) -> Option<Self> {
        let mut workers = quarantined_pc_workers_v3()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let index = workers.iter().position(|slot| slot.matches(key))?;
        if !matches!(&workers[index], PcWorkerSlotV3::Quarantined(_)) {
            return None;
        }
        let PcWorkerSlotV3::Quarantined(owner) =
            std::mem::replace(&mut workers[index], PcWorkerSlotV3::Reserved(key.clone()))
        else {
            unreachable!()
        };
        Some(Self {
            key: owner.key,
            admission: owner.admission,
            sender: owner.sender,
            received: owner.received,
            receive_state: owner.receive_state,
            thread: Some(owner.thread),
        })
    }

    /// Join the original worker only after its original receiver is gone and
    /// no handoff, including an invalid one, was ever received. Busy returns
    /// None without consuming the handle; a panic returns its original payload.
    pub fn try_join_without_handoff(
        &mut self,
    ) -> Result<
        Option<RoventPcCollectionNoHandoffCustodyV3>,
        RoventPcCollectionNoHandoffJoinFailureV3,
    > {
        let thread = self
            .thread
            .as_ref()
            .ok_or(RoventPcCollectionNoHandoffJoinFailureV3::AlreadyJoined)?;
        if !thread.is_finished() || !self.receive_state.receiver_dropped.load(Ordering::Acquire) {
            return Ok(None);
        }
        if self.receive_state.received_any.load(Ordering::Acquire) {
            return Err(RoventPcCollectionNoHandoffJoinFailureV3::HandoffReceived);
        }
        let mut slots = match quarantined_pc_workers_v3().try_lock() {
            Ok(slots) => slots,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(RoventPcCollectionNoHandoffJoinFailureV3::OwnerLockPoisoned);
            }
        };
        let Some(slot) = slots.iter().position(|slot| matches!(slot, PcWorkerSlotV3::Reserved(key) if Arc::ptr_eq(&key.0, &self.key.0))) else {
            return Err(RoventPcCollectionNoHandoffJoinFailureV3::WorkerSlotMissing);
        };
        let worker = self.thread.take().expect("checked V3 worker handle");
        slots.swap_remove(slot);
        drop(slots);
        match worker.join() {
            Ok(original_worker) => Ok(Some(RoventPcCollectionNoHandoffCustodyV3 {
                key: self.key.clone(),
                admission: self.admission.clone(),
                original_worker,
            })),
            Err(payload) => Err(RoventPcCollectionNoHandoffJoinFailureV3::Panicked(payload)),
        }
    }

    pub fn try_join(
        &mut self,
        handoff: &RootHandoffV3,
    ) -> Result<Option<RoventPcCollectionRootCustodyV3>, RoventPcCollectionWorkerJoinFailureV3>
    {
        let original = match handoff.try_lock() {
            Ok(original) => original,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(RoventPcCollectionWorkerJoinFailureV3::OwnerLockPoisoned);
            }
        };
        if !original.admission.same_original_admission(&self.admission) {
            return Err(RoventPcCollectionWorkerJoinFailureV3::WrongRoot);
        }
        drop(original);
        let thread = self
            .thread
            .as_ref()
            .ok_or(RoventPcCollectionWorkerJoinFailureV3::AlreadyJoined)?;
        if !thread.is_finished() {
            return Ok(None);
        }
        // Do not consume the JoinHandle until both owner records can be read
        // and the pre-reserved slot can be released without waiting.
        let received = match self.received.try_lock() {
            Ok(received) => received,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(RoventPcCollectionWorkerJoinFailureV3::OwnerLockPoisoned);
            }
        };
        let matches_handoff = received
            .as_ref()
            .is_some_and(|received| Arc::ptr_eq(received, handoff));
        if received.is_some() && !matches_handoff {
            // Preserve the original handle and its actual received handoff.
            return Err(RoventPcCollectionWorkerJoinFailureV3::WrongRoot);
        }
        drop(received);
        let mut slots = match quarantined_pc_workers_v3().try_lock() {
            Ok(slots) => slots,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(RoventPcCollectionWorkerJoinFailureV3::OwnerLockPoisoned);
            }
        };
        let Some(slot) = slots.iter().position(|slot| matches!(slot, PcWorkerSlotV3::Reserved(key) if Arc::ptr_eq(&key.0, &self.key.0))) else {
            return Err(RoventPcCollectionWorkerJoinFailureV3::WorkerSlotMissing);
        };
        let worker = self.thread.take().expect("checked V3 worker handle");
        slots.swap_remove(slot);
        drop(slots);
        // The original handle remains in this frame. Never hold the shared
        // registry while an OS join or TLS cleanup completes.
        let result = worker.join();
        match result {
            Ok(original_worker) => {
                if !matches_handoff {
                    return Err(
                        RoventPcCollectionWorkerJoinFailureV3::CompletedWithoutHandoff(
                            original_worker,
                        ),
                    );
                }
                Ok(Some(RoventPcCollectionRootCustodyV3::from_joined_worker(
                    Arc::clone(handoff),
                    original_worker,
                )))
            }
            Err(payload) => Err(RoventPcCollectionWorkerJoinFailureV3::Panicked(payload)),
        }
    }
}

impl Drop for RoventPcCollectionWorkerV3 {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            let mut slots = quarantined_pc_workers_v3()
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let owner = QuarantinedPcWorkerV3 {
                key: self.key.clone(),
                admission: self.admission.clone(),
                sender: self.sender.clone(),
                received: Arc::clone(&self.received),
                receive_state: Arc::clone(&self.receive_state),
                thread,
            };
            if let Some(slot) = slots.iter_mut().find(|slot| slot.matches(&self.key)) {
                if matches!(slot, PcWorkerSlotV3::Reserved(_)) {
                    *slot = PcWorkerSlotV3::Quarantined(owner);
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

pub enum RoventPcCollectionAdmissionOutcomeV3 {
    Installed,
    /// This exact admission did not start an observer or teacher.
    Pending {
        admission: RoventCollectionRootAdmissionV3,
    },
    /// Installation is unknown; the original attempt must be retained.
    PendingUncertain {
        admission: RoventCollectionRootAdmissionV3,
    },
    DeferredNoCapture {
        capability: RoventNoCaptureRootCapabilityV3,
        reason: AdaptiveCollectionError,
    },
}

pub struct RoventPcCollectionAdmissionFailureV3 {
    pub error: AdaptiveManagerError,
    pub admission: RoventCollectionRootAdmissionV3,
    pub owners: Box<dyn Any + Send>,
}

/// A borrowed original manager; this port cannot manufacture an admission.
pub struct RoventPcCollectionAdmissionPortV3<'a> {
    pub(crate) manager: &'a mut RoventPcSessionManager,
}
impl RoventPcCollectionAdmissionPortV3<'_> {
    pub fn no_capture_capability(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<RoventNoCaptureRootCapabilityV3, AdaptiveManagerError> {
        self.manager.no_capture_root_v3(admission)
    }
    pub fn install(
        &mut self,
        admission: RoventCollectionRootAdmissionV3,
        observer: AdaptiveCollectionObserverBoxV3<PcPortQuestion, VirtualReviewTicketOutput>,
    ) -> Result<(), RoleOnlyCollectionV3Rejection<PcPortQuestion, VirtualReviewTicketOutput>> {
        self.manager
            .install_collection_observer_v3(admission, observer)
    }

    /// Return the original admission and observer on an explicit Busy; no
    /// observer installation or new ownership is implied by that outcome.
    pub fn try_install(
        &mut self,
        admission: RoventCollectionRootAdmissionV3,
        observer: AdaptiveCollectionObserverBoxV3<PcPortQuestion, VirtualReviewTicketOutput>,
    ) -> Result<
        RoleOnlyCollectionV3InstallAttempt<PcPortQuestion, VirtualReviewTicketOutput>,
        RoleOnlyCollectionV3Rejection<PcPortQuestion, VirtualReviewTicketOutput>,
    > {
        self.manager
            .try_install_collection_observer_v3(admission, observer)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventPcCustodyReturnV3 {
    Accepted,
    Busy,
}

/// Callbacks are nonblocking, root-scoped and retain their original owners.
pub trait RoventPcCollectionHooksV3: Send + Sync {
    /// Trusted internal boundary for the installed original hook. `Some`
    /// must retain the actual evidence for every admitted root, worker,
    /// teacher and observer; the native fence alone is insufficient.
    /// `None` leaves all custody in this hook for a later nonblocking retry.
    /// An error or panic leaves settlement unresolved and is never approval.
    fn try_finish_launch(
        &self,
        _native: &super::launch::RoventPcCollectionNativeFenceV3,
    ) -> Result<Option<Box<dyn Any + Send>>, AdaptiveManagerError> {
        Ok(None)
    }

    fn register_startup_control(
        &self,
        control: &RoventStartupControl,
    ) -> Result<(), AdaptiveManagerError>;
    fn admit(
        &self,
        port: &mut RoventPcCollectionAdmissionPortV3<'_>,
        admission: RoventCollectionRootAdmissionV3,
    ) -> Result<RoventPcCollectionAdmissionOutcomeV3, RoventPcCollectionAdmissionFailureV3>;
    fn pump_root(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<(), AdaptiveManagerError>;
    fn revoke_root(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<(), AdaptiveManagerError>;
    /// A timing/state observation only; it does not grant V2 ownership.
    fn revoke_root_observed(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
        _observation: super::pc_session::RoventPcRevokeObservationV2,
    ) -> Result<(), AdaptiveManagerError> {
        self.revoke_root(admission)
    }
    fn finish_deferred(
        &self,
        capability: &RoventNoCaptureRootCapabilityV3,
        fence: &RoventRoleOnlyNoCaptureFenceV3,
    ) -> Result<crate::pals_adaptive::AdaptiveCollectionBoundaryV2, AdaptiveManagerError>;
    fn retain_deferred_cleanup(
        &self,
        cleanup: Arc<Mutex<RoventPcDeferredCleanupV3>>,
    ) -> Result<RoventPcCustodyReturnV3, AdaptiveManagerError>;
    fn handoff_sender_root(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<mpsc::SyncSender<Arc<Mutex<RoventPcCollectionHandoffV3>>>, AdaptiveManagerError>;
    fn try_take_root_custody(
        &self,
        handoff: &Arc<Mutex<RoventPcCollectionHandoffV3>>,
    ) -> Result<Option<RoventPcCollectionRootCustodyV3>, AdaptiveManagerError>;
    fn return_root_custody(
        &self,
        custody: Arc<Mutex<RoventPcCollectionRootCustodyV3>>,
    ) -> Result<RoventPcCustodyReturnV3, AdaptiveManagerError>;
}
