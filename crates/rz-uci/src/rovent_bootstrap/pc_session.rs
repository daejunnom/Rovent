//! Native P/C product owner. No researcher, PVS engine, CPU credit ledger or
//! checker metadata exists in this state machine.

use super::configuration::{ExecutionConfiguration, StartupConfigurationDocument};
use super::launch::storage::RoventStorageOwnedRuntimeRegistration;
use super::nnuec_source::LoadedRoventNnuecSource;
use super::process_boundary::VerifiedOwnedRoventProcessBoundaryV2;
use super::product::SharedRoventStorageRegistration;
use super::storage_lifecycle::{SessionStorageLifecycle, StorageSemanticOwner};
use super::{
    PreparedRoventNative, PreparedRoventNativeV2, RoventNativeBackend, RoventNativeService,
    RoventNativeStartupSnapshot, RoventPinnedNativeSources,
};
use crate::EngineIdentity;
use crate::engine::{OwnerRegistry, ProcessClock};
use crate::pals_adaptive::{
    AdaptiveCollectionError, AdaptiveCollectionObserverBoxV2, AdaptiveManagerError, ManagerPhase,
    PcPortDriver, RoleOnlyCollectionV2Rejection, RoleOnlyExecutionManager,
    RoleOnlyManagerRegistration, RoleOnlyManagerSources, RoleOnlyShutdownReceipt,
    RoventCollectionRootAdmission, RoventCollectionRootAdmissionV2, RoventNativeCollectionOrigin,
    RoventNativeCollectionOriginV2,
};
use crate::pals_native::ticketed::{TicketNativeEvent, VirtualReviewTicketOutput};
use crate::rovent_root_admission::source_seal::{
    InitialAdmissionOutcome, InitialAdmissionOutcomeV2, PreparedSourceV2, RoventRootAdmissionV2,
    SourceSeal, StoppedInitial, StoppedInitialV2,
};
use crate::rovent_root_admission::{RoventRootAdmission, RoventRootAdmissionPolicy};
use crate::search_driver::{
    DriverPhysicalCompletion, DriverWork, SearchKind, SearchSessionContext, SearchSessionDriver,
    SearchSessionFailure, SearchSessionReport,
};
use rz_contracts::pals::SearchAuthority;
use rz_contracts::{ContractError, Digest};
use rz_eval::error::{BackendError, FailureKind};
use rz_position::{BoardMove, Position};
use rz_runtime::contracts::ContractClock;
use rz_search::cpu_value::nnuec::v2::review::PreparationActor;
use rz_search::pals::engine::adaptive_port::{AdaptivePortModel, AdaptiveRulesOwnerIssuer};
use rz_search::pals::engine::pc_port::{PcFrontierStartup, PcPortQuestion, PcSemanticPort};
use rz_search::pals::engine::{PcSemanticOwner, PostRepairRecheckPolicy};
use rz_search::pals::rovent_frontier::{
    FrontierError, FrontierGoalStatus, FrontierLimits, FrontierRegistration, FrontierSnapshot,
    FrontierStop,
};
use sha2::{Digest as _, Sha256};
use std::{
    any::Any,
    num::NonZeroU16,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex, TryLockError,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

type Backend = RoventNativeBackend<PcPortQuestion>;
type Model = AdaptivePortModel;
enum PcSources {
    V1(Arc<RoventPinnedNativeSources>),
    V2(Arc<RoventPinnedNativeSources<rz_experiments::RoventInputLockV2>>),
}
enum PcPreparedRoot {
    V1(SourceSeal),
    V2(PreparedSourceV2),
}
enum PcRootAdmission {
    V1(Arc<RoventRootAdmission>),
    V2(Arc<RoventRootAdmissionV2>),
}
impl PcRootAdmission {
    fn budget_digest(&self) -> Result<Digest, SearchSessionFailure> {
        match self {
            Self::V1(value) => Ok(value.root_budget_digest()),
            Self::V2(value) => value
                .descriptor()
                .root_budget_digest()
                .map_err(|e| SearchSessionFailure::debug("RoventPcV2Budget", &e)),
        }
    }
}
impl PcSources {
    fn v1(&self) -> Option<&Arc<RoventPinnedNativeSources>> {
        match self {
            Self::V1(source) => Some(source),
            Self::V2(_) => None,
        }
    }
    fn forget_original(&self) {
        match self {
            Self::V1(source) => std::mem::forget(Arc::clone(source)),
            Self::V2(source) => std::mem::forget(Arc::clone(source)),
        }
    }
}
pub type RoventPcSessionManager =
    RoleOnlyExecutionManager<PcPortQuestion, Backend, PcPortDriver<Model>>;
type Manager = RoventPcSessionManager;
pub type RoventPcAdmissionObserver = Arc<
    dyn Fn(
            &mut RoventPcSessionManager,
            RoventCollectionRootAdmission,
        ) -> Result<(), AdaptiveManagerError>
        + Send
        + Sync,
>;
pub type RoventPcClosureObserver =
    Arc<dyn Fn(&mut RoventPcSessionManager) -> Result<(), AdaptiveManagerError> + Send + Sync>;

/// The original collection objects after native fence and settlement. Only
/// the session can construct this value; receiving it is not evidence that
/// the separate teacher process has retired or its outputs were consumed.
pub struct RoventPcCollectionHandoffV2 {
    admission: RoventCollectionRootAdmissionV2,
    observer: AdaptiveCollectionObserverBoxV2<PcPortQuestion, VirtualReviewTicketOutput>,
    settlement: Result<(), AdaptiveCollectionError>,
    native_fence: RoleOnlyShutdownReceipt,
    teacher_physical_ack: Option<crate::pals_adaptive::RoventTeacherScopePhysicalAckV2>,
    teacher_aggregate_ack: Option<crate::rovent_external_launch::RoventTeacherScopeAggregateAckV2>,
}

impl RoventPcCollectionHandoffV2 {
    /// Accept exactly one physical teacher-scope acknowledgement from this
    /// same root. A mismatch or duplicate returns the original capability.
    /// This does not attest semantic success or dataset publication.
    pub fn submit_teacher_physical_ack(
        &mut self,
        ack: crate::pals_adaptive::RoventTeacherScopePhysicalAckV2,
    ) -> Result<(), crate::pals_adaptive::RoventTeacherScopePhysicalAckV2> {
        if self.teacher_physical_ack.is_some() || !ack.matches_admission(&self.admission) {
            return Err(ack);
        }
        self.teacher_physical_ack = Some(ack);
        Ok(())
    }

    /// Retain one aggregate proof for this exact root. It does not on its own
    /// attest semantic success, publication, or authority to resume a root.
    pub fn submit_teacher_aggregate_ack(
        &mut self,
        ack: crate::rovent_external_launch::RoventTeacherScopeAggregateAckV2,
    ) -> Result<(), crate::rovent_external_launch::RoventTeacherScopeAggregateAckV2> {
        if self.teacher_aggregate_ack.is_some() || !ack.matches_admission(&self.admission) {
            return Err(ack);
        }
        self.teacher_aggregate_ack = Some(ack);
        Ok(())
    }

    pub fn teacher_aggregate_ack(
        &self,
    ) -> Option<&crate::rovent_external_launch::RoventTeacherScopeAggregateAckV2> {
        self.teacher_aggregate_ack.as_ref()
    }

    /// Inspect or operate on the original observer while the session retains
    /// another Arc to this handoff. Borrowing cannot move out its owners.
    pub fn parts_mut(
        &mut self,
    ) -> (
        &RoventCollectionRootAdmissionV2,
        &mut AdaptiveCollectionObserverBoxV2<PcPortQuestion, VirtualReviewTicketOutput>,
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

    /// Transfer the original objects to their collection owner without
    /// replacing the admission or synthesizing a native fence.
    pub fn into_parts(
        self,
    ) -> (
        RoventCollectionRootAdmissionV2,
        AdaptiveCollectionObserverBoxV2<PcPortQuestion, VirtualReviewTicketOutput>,
        Result<(), AdaptiveCollectionError>,
        RoleOnlyShutdownReceipt,
        Option<crate::pals_adaptive::RoventTeacherScopePhysicalAckV2>,
        Option<crate::rovent_external_launch::RoventTeacherScopeAggregateAckV2>,
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

/// The original joined per-root collection worker and the same handoff.
/// Custody does not assert dataset publication or semantic teacher success.
pub struct RoventPcCollectionRootCustodyV2 {
    handoff: Arc<Mutex<RoventPcCollectionHandoffV2>>,
    original_worker: Box<dyn Any + Send>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventPcCustodyReturnV2 {
    Accepted,
    Busy,
}

/// Read-only timing and state observed at a root-scoped revoke call. These
/// fields confer no authority to cancel, complete, or publish work.
#[derive(Clone, Copy, Debug)]
pub struct RoventPcRevokeObservationV2 {
    pub origin: RoventPcRevokeOriginV2,
    pub elapsed_from_start: Option<Duration>,
    pub context_acceptance: RoventPcContextAcceptanceV2,
    pub soft_expired: bool,
    pub manager_phase: ManagerPhase,
    pub goal: Option<FrontierGoalStatus>,
}
#[derive(Clone, Copy, Debug)]
pub enum RoventPcRevokeOriginV2 { LoopCondition, RunEnded }
#[derive(Clone, Copy, Debug)]
pub enum RoventPcContextAcceptanceV2 { Accepted, Rejected, Unavailable }

/// Failed marker cleanup after the same root's native fence. This owns the
/// original capability and diagnostic; it never claims collection completion.
pub struct RoventPcDeferredCleanupV2 {
    capability: crate::pals_adaptive::RoventNoCaptureRootCapabilityV2,
    reason: AdaptiveCollectionError,
    fence: RoleOnlyShutdownReceipt,
    cleanup_failure: RoventPcDeferredCleanupFailureV2,
    retention_failure: Option<AdaptiveManagerError>,
}
pub enum RoventPcDeferredCleanupFailureV2 {
    Busy,
    Error(AdaptiveManagerError),
    Panic(Box<dyn Any + Send>),
}
impl RoventPcDeferredCleanupV2 {
    pub fn capability(&self) -> &crate::pals_adaptive::RoventNoCaptureRootCapabilityV2 { &self.capability }
    pub fn reason(&self) -> &AdaptiveCollectionError { &self.reason }
    pub fn fence(&self) -> &RoleOnlyShutdownReceipt { &self.fence }
    pub fn failure(&self) -> &RoventPcDeferredCleanupFailureV2 { &self.cleanup_failure }
    pub fn retention_failure(&self) -> Option<&AdaptiveManagerError> { self.retention_failure.as_ref() }
}

impl RoventPcCollectionRootCustodyV2 {
    /// The adapter constructs this only after joining the original worker.
    pub fn from_joined_worker(
        handoff: Arc<Mutex<RoventPcCollectionHandoffV2>>,
        original_worker: Box<dyn Any + Send>,
    ) -> Self {
        Self {
            handoff,
            original_worker,
        }
    }

    pub fn matches_handoff(&self, handoff: &Arc<Mutex<RoventPcCollectionHandoffV2>>) -> bool {
        Arc::ptr_eq(&self.handoff, handoff)
    }

    pub fn handoff(&self) -> &Arc<Mutex<RoventPcCollectionHandoffV2>> {
        &self.handoff
    }

    pub fn original_worker(&self) -> &(dyn Any + Send) {
        self.original_worker.as_ref()
    }

    pub fn into_parts(self) -> (Arc<Mutex<RoventPcCollectionHandoffV2>>, Box<dyn Any + Send>) {
        (self.handoff, self.original_worker)
    }
}

/// An original V2 collection owner. The hook must retain its teacher and
/// publication custody; these callbacks do not prove physical completion.
/// `admit`, `pump`, and `revoke` run while the root's session owner is held:
/// implementations must be bounded and nonblocking and must not synchronously
/// acquire a root/teacher lock or wait for teacher I/O. They may instead pass
/// work to an independently owned worker with bounded admission.
pub trait RoventPcCollectionHooksV2: Send + Sync {
    /// Bind the original startup control once, before native preparation.
    /// This only registers the identity; it must not spawn or wait for work.
    fn register_startup_control(
        &self,
        control: &super::RoventStartupControl,
    ) -> Result<(), AdaptiveManagerError>;
    fn admit(
        &self,
        port: &mut RoventPcCollectionAdmissionPortV2<'_>,
        admission: RoventCollectionRootAdmissionV2,
    ) -> Result<RoventPcCollectionAdmissionOutcomeV2, RoventPcCollectionAdmissionFailureV2>;
    /// Remove the exact no-capture marker only after the native root fence.
    fn finish_deferred(
        &self,
        _capability: &crate::pals_adaptive::RoventNoCaptureRootCapabilityV2,
        _fence: &RoleOnlyShutdownReceipt,
    ) -> Result<crate::pals_adaptive::AdaptiveCollectionBoundaryV2, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid("P/C no-capture cleanup unsupported"))
    }
    /// Keep the exact unresolved cleanup and its original diagnostic. Success
    /// means custody only, not marker removal or collection publication.
    fn retain_deferred_cleanup(
        &self,
        _cleanup: Arc<Mutex<RoventPcDeferredCleanupV2>>,
    ) -> Result<RoventPcCustodyReturnV2, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid("P/C deferred cleanup custody unsupported"))
    }
    fn pump(&self) -> Result<(), AdaptiveManagerError>;
    fn revoke(&self);
    /// Scoped hooks must act only on this original admission. Legacy global
    /// hooks are retained for compatibility but are not safe for reuse.
    fn pump_root(&self, _root: &RoventCollectionRootAdmissionV2) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid("P/C root-scoped pump unsupported"))
    }
    fn revoke_root(&self, _root: &RoventCollectionRootAdmissionV2) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid("P/C root-scoped revoke unsupported"))
    }
    /// This callback must be bounded and nonblocking like `revoke_root`.
    fn revoke_root_observed(
        &self,
        root: &RoventCollectionRootAdmissionV2,
        _observation: RoventPcRevokeObservationV2,
    ) -> Result<(), AdaptiveManagerError> {
        self.revoke_root(root)
    }
    /// Return a bounded channel reserved for this original root. The session
    /// sends directly with `try_send` and retains another Arc whether delivery
    /// fails, the receiver is dropped, or teacher completion is unresolved.
    fn handoff_sender(
        &self,
    ) -> std::sync::mpsc::SyncSender<Arc<Mutex<RoventPcCollectionHandoffV2>>>;
    fn handoff_sender_root(
        &self,
        _root: &RoventCollectionRootAdmissionV2,
    ) -> Result<std::sync::mpsc::SyncSender<Arc<Mutex<RoventPcCollectionHandoffV2>>>, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid("P/C root-scoped handoff unsupported"))
    }
    /// Nonblocking; `None` preserves the still-running original worker.
    fn try_take_root_custody(
        &self,
        _handoff: &Arc<Mutex<RoventPcCollectionHandoffV2>>,
    ) -> Result<Option<RoventPcCollectionRootCustodyV2>, AdaptiveManagerError> {
        Ok(None)
    }
    /// Retain the exact joined custody before allowing the root to finish.
    /// Success transfers no semantic or publication authority.
    fn return_root_custody(
        &self,
        _custody: Arc<Mutex<RoventPcCollectionRootCustodyV2>>,
    ) -> Result<RoventPcCustodyReturnV2, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid("P/C V2 custody receiver unavailable"))
    }
}

/// Pending never starts a teacher or observer. Deferred requires the original
/// native no-capture marker to have been registered before returning it.
pub enum RoventPcCollectionAdmissionOutcomeV2 {
    Installed,
    Pending { admission: RoventCollectionRootAdmissionV2 },
    DeferredNoCapture {
        capability: crate::pals_adaptive::RoventNoCaptureRootCapabilityV2,
        reason: AdaptiveCollectionError,
    },
}

/// The rejected admission and every acquired observer/teacher owner remain
/// with this value until the caller explicitly recovers or quarantines it.
pub struct RoventPcCollectionAdmissionFailureV2 {
    pub error: AdaptiveManagerError,
    pub admission: RoventCollectionRootAdmissionV2,
    pub owners: Box<dyn Any + Send>,
}

/// Borrowed access to the actual manager for one root admission. This port
/// cannot manufacture a root or issue an acceptance witness.
pub struct RoventPcCollectionAdmissionPortV2<'a> {
    manager: &'a mut RoventPcSessionManager,
}

impl RoventPcCollectionAdmissionPortV2<'_> {
    pub fn no_capture_capability(
        &self,
        admission: &RoventCollectionRootAdmissionV2,
    ) -> Result<crate::pals_adaptive::RoventNoCaptureRootCapabilityV2, AdaptiveManagerError> {
        self.manager.no_capture_root_v2(admission)
    }
    pub fn install(
        &mut self,
        admission: RoventCollectionRootAdmissionV2,
        observer: AdaptiveCollectionObserverBoxV2<PcPortQuestion, VirtualReviewTicketOutput>,
    ) -> Result<(), RoleOnlyCollectionV2Rejection<PcPortQuestion, VirtualReviewTicketOutput>> {
        self.manager
            .install_collection_observer_v2(admission, observer)
    }
}

/// Failure keeps the original observer and root authority. The returned live
/// objects are not completion evidence and may require separate quarantine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RoventPcV2FrontierAdmissionStatus {
    Rejected,
    Panicked,
}

pub(crate) enum RoventPcV2ObserverAttachRejection {
    BeforeAdmission {
        error: AdaptiveManagerError,
        status: RoventPcV2FrontierAdmissionStatus,
        root: Arc<RoventRootAdmissionV2>,
        observer: AdaptiveCollectionObserverBoxV2<PcPortQuestion, VirtualReviewTicketOutput>,
        // The original panic payload may itself own live resources. It must
        // remain with the rejected root and observer until safely handled.
        panic_payload: Option<Box<dyn Any + Send>>,
    },
    Manager(RoleOnlyCollectionV2Rejection<PcPortQuestion, VirtualReviewTicketOutput>),
}

/// Attach an observation-only V2 consumer to the existing manager and the
/// actual frontier. Teacher process ownership and product launch remain
/// separate and are not authorized by this operation.
pub(crate) fn attach_pc_v2_collection_observer(
    manager: &mut RoventPcSessionManager,
    origin: &RoventNativeCollectionOriginV2,
    root: Arc<RoventRootAdmissionV2>,
    root_budget: Digest,
    selective_policy: Digest,
    observer: AdaptiveCollectionObserverBoxV2<PcPortQuestion, VirtualReviewTicketOutput>,
) -> Result<(), RoventPcV2ObserverAttachRejection> {
    let admission = catch_unwind(AssertUnwindSafe(|| {
        match manager.driver().port().rovent_frontier() {
            Some(frontier) => RoventCollectionRootAdmissionV2::from_attached_frontier(
                origin,
                frontier,
                Arc::clone(&root),
                root_budget,
                selective_policy,
            )
            .map_err(AdaptiveManagerError::from),
            None => Err(AdaptiveManagerError::Invalid(
                "P/C V2 collection has no actual attached frontier",
            )),
        }
    }));
    let admission = match admission {
        Ok(Ok(admission)) => admission,
        Ok(Err(error)) => {
            return Err(RoventPcV2ObserverAttachRejection::BeforeAdmission {
                error,
                status: RoventPcV2FrontierAdmissionStatus::Rejected,
                root,
                observer,
                panic_payload: None,
            });
        }
        Err(payload) => {
            return Err(RoventPcV2ObserverAttachRejection::BeforeAdmission {
                error: AdaptiveManagerError::Invalid(
                    "P/C V2 collection frontier admission panicked",
                ),
                status: RoventPcV2FrontierAdmissionStatus::Panicked,
                root,
                observer,
                panic_payload: Some(payload),
            });
        }
    };
    manager
        .install_collection_observer_v2(admission, observer)
        .map_err(RoventPcV2ObserverAttachRejection::Manager)
}
#[derive(Default)]
pub struct RoventPcCollectionHooks {
    pub root_admission: Option<RoventPcAdmissionObserver>,
    pub root_closure: Option<RoventPcClosureObserver>,
    pub cooperative: Option<Arc<dyn Fn() -> Result<(), AdaptiveManagerError> + Send + Sync>>,
}

pub const ROVENT_PC_ROOT_DOMAIN: &str = "rz-rovent-pc-native-root/1";
pub const ROVENT_PC_INITIAL_DOMAIN: &str = "rz-rovent-pc-native-initial-stop/1";
pub struct RoventPcRootReceipt {
    pub domain: &'static str,
    pub authority: SearchAuthority,
    pub input_lock: Digest,
    /// Original startup lock and each go's actual workload are distinct seals.
    pub root_admission: Arc<RoventRootAdmission>,
    pub frontier: FrontierSnapshot,
    pub physical: RoleOnlyShutdownReceipt,
    pub repair: Option<rz_search::pals::engine::pc_port::PcRepairReport>,
}

/// Actual /2 root outcome, retaining its original typed admission and fence.
pub struct RoventPcRootReceiptV2 {
    pub domain: &'static str,
    pub authority: SearchAuthority,
    pub input_lock: Digest,
    pub root_admission: Arc<RoventRootAdmissionV2>,
    pub frontier: FrontierSnapshot,
    pub physical: RoleOnlyShutdownReceipt,
    pub repair: Option<rz_search::pals::engine::pc_port::PcRepairReport>,
}
impl std::fmt::Debug for RoventPcRootReceiptV2 {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("RoventPcRootReceiptV2")
            .field("domain", &self.domain)
            .field("authority", &self.authority)
            .field("input_lock", &self.input_lock)
            .field("root_admission", &self.root_admission)
            .field("frontier", &self.frontier)
            .field("physical", &self.physical)
            .field("repair_present", &self.repair.is_some())
            .finish()
    }
}
impl std::fmt::Debug for RoventPcRootReceipt {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("RoventPcRootReceipt")
            .field("domain", &self.domain)
            .field("authority", &self.authority)
            .field("input_lock", &self.input_lock)
            .field("root_admission", &self.root_admission)
            .field("frontier", &self.frontier)
            .field("physical", &self.physical)
            .field("repair_present", &self.repair.is_some())
            .finish()
    }
}
fn pc_root_work(
    frontier: &FrontierSnapshot,
    physical: &RoleOnlyShutdownReceipt,
) -> Result<DriverWork, SearchSessionFailure> {
    if !frontier.work.native_usage_known
        || frontier.work.native_nodes != 0
        || frontier.work.native_qsearch_nodes != 0
        || frontier.work.reserved_native_nodes != 0
        || frontier.work.outstanding_native_executions != 0
    {
        return Err(failure(
            "RoventPcResearcherPresent",
            "R-free work contains native R accounting",
        ));
    }
    let total = frontier.work.total_node_visits().ok_or_else(|| {
        failure(
            "RoventPcWorkOverflow",
            "original node observations overflow",
        )
    })?;
    let counts = &physical.counters;
    Ok(DriverWork::RoventNative {
        root_entries: frontier.work.root_entries,
        total_node_visits: Some(total),
        checked_transitions: frontier.work.checked_transitions,
        native_nodes: Some(0),
        native_qsearch_nodes: Some(0),
        submitted_role_tickets: counts.submitted_role_tickets,
        consumed_role_outputs: counts.consumed_role_outputs,
        completed_nn_inputs: (counts.nn_input_count_unknown == 0)
            .then_some(counts.completed_nn_inputs),
        consumed_nn_inputs: (counts.consumed_nn_input_count_unknown == 0)
            .then_some(counts.consumed_nn_inputs),
        completed_depth: frontier.horizon.completed_depth.map(|depth| depth.0),
        maximum_examined_ply: frontier.horizon.maximum_examined_ply.0,
    })
}

/// An original Start stopped before a root, view or native ticket was created.
/// The private basis is issued only with the prepared native owner or after a
/// real root's clean physical transfer. This is not a new physical fence.
#[derive(Clone, Debug)]
pub struct RoventPcInitialReceipt {
    authority: SearchAuthority,
    input_lock: Digest,
    source: Arc<StoppedInitial>,
    idle_custody: IdleCustody,
}

#[derive(Clone, Debug)]
pub struct RoventPcInitialReceiptV2 {
    pub authority: SearchAuthority,
    pub input_lock: Digest,
    pub source: Arc<StoppedInitialV2>,
    previous: Option<Arc<RoventPcRootReceiptV2>>,
}
impl RoventPcInitialReceiptV2 {
    pub fn previous_clean_root(&self) -> Option<&Arc<RoventPcRootReceiptV2>> {
        self.previous.as_ref()
    }
}
impl RoventPcInitialReceipt {
    pub fn domain(&self) -> &'static str {
        ROVENT_PC_INITIAL_DOMAIN
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn input_lock_digest(&self) -> Digest {
        self.input_lock
    }
    pub fn source(&self) -> &Arc<StoppedInitial> {
        &self.source
    }
    pub fn previous_clean_root(&self) -> Option<&Arc<RoventPcRootReceipt>> {
        match &self.idle_custody {
            IdleCustody::StartupReady => None,
            IdleCustody::RetiredRoot(receipt) => Some(receipt),
            IdleCustody::RetiredRootV2(_) => None,
        }
    }
    pub fn used_original_startup_ready(&self) -> bool {
        matches!(self.idle_custody, IdleCustody::StartupReady)
    }
    fn report(&self) -> SearchSessionReport {
        initial_stop_report(self.source.fallback())
    }
}
fn initial_stop_report(fallback: Option<BoardMove>) -> SearchSessionReport {
    SearchSessionReport {
        best_move: fallback,
        work: DriverWork::RoventNative {
            root_entries: 0,
            total_node_visits: Some(0),
            checked_transitions: 0,
            native_nodes: Some(0),
            native_qsearch_nodes: Some(0),
            submitted_role_tickets: 0,
            consumed_role_outputs: 0,
            completed_nn_inputs: Some(0),
            consumed_nn_inputs: Some(0),
            completed_depth: None,
            maximum_examined_ply: 0,
        },
        mate_strategy: None,
    }
}

#[derive(Clone, Debug)]
enum IdleCustody {
    StartupReady,
    RetiredRoot(Arc<RoventPcRootReceipt>),
    RetiredRootV2(Arc<RoventPcRootReceiptV2>),
}
struct Idle {
    native: NativeOwner,
    owner: Option<Box<PcSemanticOwner<Model>>>,
    next_request_sequence: u64,
    game: Option<u64>,
    custody: IdleCustody,
}
struct NativeOwner {
    service: RoventNativeService<PcPortQuestion>,
    finished: Option<bool>,
    joined: bool,
    unexpected: Vec<TicketNativeEvent<VirtualReviewTicketOutput>>,
    // A recovered native source retains the original typed root/cache closure
    // even if subsequent semantic handoff or receipt publication is rejected.
    // Unknown State drop quarantines this slot with its original process.
    _root_closure: Option<RoleOnlyShutdownReceipt>,
}
impl NativeOwner {
    fn new(service: RoventNativeService<PcPortQuestion>) -> Self {
        Self {
            service,
            finished: None,
            joined: false,
            unexpected: Vec::new(),
            _root_closure: None,
        }
    }
    fn retain_root_closure(mut self, closure: RoleOnlyShutdownReceipt) -> Self {
        self._root_closure = Some(closure);
        self
    }
    fn finish(&mut self, until: Instant) -> Result<(), SearchSessionFailure> {
        self.service.begin_shutdown(until);
        loop {
            let signal = self.service.signal();
            let observed = signal.version();
            while let Some(event) = self.service.poll() {
                match event {
                    TicketNativeEvent::Finished {
                        physical_shutdown_confirmed,
                    } => {
                        self.finished =
                            Some(self.finished.unwrap_or(true) && physical_shutdown_confirmed);
                    }
                    TicketNativeEvent::Ready { .. } | TicketNativeEvent::RootQuiesced(_) => {}
                    other => self.unexpected.push(other),
                }
            }
            self.joined = self
                .service
                .try_join()
                .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
            if self.joined && self.finished == Some(true) {
                return if self.unexpected.is_empty() {
                    Ok(())
                } else {
                    Err(failure(
                        "RoventPcNativeProtocol",
                        "unexpected native events retained until physical shutdown",
                    ))
                };
            }
            if Instant::now() >= until || self.finished == Some(false) {
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
            signal.wait_changed(
                observed,
                until
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(10)),
            );
        }
    }
}
// One mutex-protected session slot owns the current native/semantic custody.
// RejectedPort retains that exact port until actual cleanup, not in a queue.
#[allow(clippy::large_enum_variant)]
enum State {
    Idle(Box<Idle>),
    Manager(Box<Manager>),
    RejectedPort {
        native: NativeOwner,
        port: PcPortDriver<Model>,
    },
    RejectedOwner {
        native: NativeOwner,
        owner: Box<PcSemanticOwner<Model>>,
    },
    ArchiveRetained {
        native: NativeOwner,
    },
    Closed,
    Moving,
}

/// Startup rejection returns the original prepared physical owner intact.
pub struct RoventPcSessionRejection {
    pub failure: SearchSessionFailure,
    pub prepared: PreparedRoventNative<PcPortQuestion>,
}
pub struct RoventPcSessionV2Rejection {
    pub failure: SearchSessionFailure,
    pub prepared: PreparedRoventNativeV2<PcPortQuestion>,
    pub process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
    pub loaded: Option<LoadedRoventNnuecSource<rz_experiments::RoventInputLockV2>>,
    pub collection_hooks: Option<Arc<dyn RoventPcCollectionHooksV2>>,
}
/// No preparation thread is started before this constructor has succeeded.
/// Rejection therefore returns both original startup owners without a join.
pub struct RoventPcSharedNnuecSessionRejection {
    pub failure: SearchSessionFailure,
    pub prepared: PreparedRoventNative<PcPortQuestion>,
    pub loaded: LoadedRoventNnuecSource,
}
struct ConstructionRejection {
    failure: SearchSessionFailure,
    prepared: PreparedRoventNative<PcPortQuestion>,
    loaded: Option<LoadedRoventNnuecSource>,
}
/// Exactly one optional RuntimeHelper; no R engine or native node owner.
enum SharedNnuecLoaded {
    V1(LoadedRoventNnuecSource),
    V2(LoadedRoventNnuecSource<rz_experiments::RoventInputLockV2>),
}
impl SharedNnuecLoaded {
    fn model(&self) -> &Arc<rz_search::cpu_value::nnuec::v2::Model> {
        match self {
            Self::V1(source) => source.model(),
            Self::V2(source) => source.model(),
        }
    }
}
struct SharedNnuecOwner {
    loaded: SharedNnuecLoaded,
    actor: Mutex<SharedNnuecActor>,
}
/// Readback only. An actual thread ID is neither an affinity claim nor a
/// preparation, model-consumption or physical-closure capability.
#[derive(Clone, Debug, Default)]
pub struct RoventPcSharedNnuecReadback {
    pub spawned_runtime_helpers: u8,
    pub worker_thread_id: Option<std::thread::ThreadId>,
    pub worker_joined: bool,
    /// No per-thread OS affinity readback is performed by this owner.
    pub worker_thread_affinity: Option<Vec<u32>>,
}
#[derive(Default)]
struct SharedNnuecActor {
    actor: Option<PreparationActor>,
    signal: rz_runtime::CompletionSignal,
    readback: RoventPcSharedNnuecReadback,
}
impl SharedNnuecActor {
    fn finish(&mut self, until: Instant) -> Result<(), SearchSessionFailure> {
        let Some(actor) = self.actor.as_mut() else {
            return if self.readback.spawned_runtime_helpers == 0 || self.readback.worker_joined {
                Ok(())
            } else {
                Err(SearchSessionFailure::physical_completion_unknown())
            };
        };
        actor.request_shutdown();
        loop {
            let observed = self.signal.version();
            match actor.try_join() {
                Ok(Some(())) => {
                    self.readback.worker_joined = true;
                    self.actor.take();
                    return Ok(());
                }
                Ok(None) if Instant::now() < until => {
                    // The actor notifies for real commands. Thread exit has
                    // no separate notification, so bounded polling still
                    // checks the original join without a busy-spin loop.
                    self.signal.wait_changed(
                        observed,
                        until
                            .saturating_duration_since(Instant::now())
                            .min(Duration::from_millis(10)),
                    );
                }
                _ => return Err(SearchSessionFailure::physical_completion_unknown()),
            }
        }
    }
}
pub struct RoventPcSessionDriver {
    sources: PcSources,
    original_v2_owner: Option<rz_experiments::RoventOriginalInputLock>,
    process_v2: Option<Arc<VerifiedOwnedRoventProcessBoundaryV2>>,
    snapshot: RoventNativeStartupSnapshot,
    configuration: StartupConfigurationDocument,
    clock: ProcessClock,
    issuer: AdaptiveRulesOwnerIssuer,
    identity: EngineIdentity,
    implementation: Digest,
    root_policy: RoventRootAdmissionPolicy,
    collection_origin: Option<RoventNativeCollectionOrigin>,
    collection_hooks: RoventPcCollectionHooks,
    collection_origin_v2: Option<RoventNativeCollectionOriginV2>,
    collection_hooks_v2: Option<Arc<dyn RoventPcCollectionHooksV2>>,
    collection_v2_outstanding: AtomicBool,
    retained_collection_handoff_v2: Mutex<Option<Arc<Mutex<RoventPcCollectionHandoffV2>>>>,
    retained_collection_custody_v2: Mutex<Option<Arc<Mutex<RoventPcCollectionRootCustodyV2>>>>,
    retained_no_capture_v2: Mutex<Option<(crate::pals_adaptive::RoventNoCaptureRootCapabilityV2, AdaptiveCollectionError)>>,
    retained_deferred_cleanup_v2: Mutex<Option<Arc<Mutex<RoventPcDeferredCleanupV2>>>>,
    retained_collection_failure_v2: Mutex<Option<RoventPcCollectionAdmissionFailureV2>>,
    retained_collection_panic_v2: Mutex<(Option<Box<dyn Any + Send>>, Option<Box<dyn Any + Send>>)>,
    selective_policy: Digest,
    frontier_limits: FrontierLimits,
    closed: AtomicBool,
    reset_requested: AtomicBool,
    state: Mutex<State>,
    latest: Mutex<Option<Arc<RoventPcRootReceipt>>>,
    latest_initial: Mutex<Option<Arc<RoventPcInitialReceipt>>>,
    latest_v2: Mutex<Option<Arc<RoventPcRootReceiptV2>>>,
    latest_initial_v2: Mutex<Option<Arc<RoventPcInitialReceiptV2>>>,
    /// A report already taken from the port survives any later failed transfer.
    /// The exclusive root owner permits at most one such in-progress report.
    retained_repair: Mutex<Option<rz_search::pals::engine::pc_port::PcRepairReport>>,
    shared_nnuec: Option<SharedNnuecOwner>,
    storage: Mutex<Option<Box<SessionStorageLifecycle>>>,
}
impl RoventPcSessionDriver {
    /// Source/configuration-derived identity only. This pure calculation opens
    /// no assets and constructs no session, Ready, clock or root authority.
    /// Native construction uses this same producer after its actual admission.
    pub fn implementation_digest_from_metadata(
        configuration: &StartupConfigurationDocument,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Digest, BackendError> {
        if !matches!(
            configuration.fields().execution,
            ExecutionConfiguration::RolesOnly { .. }
        ) {
            return Err(super::fail(
                FailureKind::UnsupportedModel,
                "P/C session metadata requires the R-free assembly",
            ));
        }
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pc-native-session/1\0");
        hash.update(super::compiled_bootstrap_source_digest());
        // The existing digest producer validates the original closed document.
        hash.update(configuration.configuration_digest()?);
        hash.update(include_bytes!("pc_session.rs"));
        hash.update([match root_policy {
            RoventRootAdmissionPolicy::ExactRegistered => 0,
            RoventRootAdmissionPolicy::DynamicGo => 1,
        }]);
        Ok(Digest(hash.finalize().into()))
    }

    /// Consume the original V2 Ready/source/process owners as one unit.
    /// This path supports only roles-only, archive-off execution.
    #[allow(clippy::result_large_err, clippy::too_many_arguments)]
    pub fn new_v2(
        prepared: PreparedRoventNativeV2<PcPortQuestion>,
        process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventPcSessionV2Rejection> {
        Self::construct_v2(
            prepared,
            None,
            process,
            clock,
            owners,
            identity,
            root_policy,
            None,
        )
    }

    #[allow(clippy::result_large_err, clippy::too_many_arguments)]
    pub fn new_v2_with_shared_nnuec(
        prepared: PreparedRoventNativeV2<PcPortQuestion>,
        loaded: LoadedRoventNnuecSource<rz_experiments::RoventInputLockV2>,
        process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventPcSessionV2Rejection> {
        Self::construct_v2(
            prepared,
            Some(loaded),
            process,
            clock,
            owners,
            identity,
            root_policy,
            None,
        )
    }

    /// Prepare the same V2 P/C session with its original collection hook.
    /// This is an integration seam, not proof of teacher retirement or corpus
    /// publication. Startup rejection returns the hook alongside the worker.
    #[allow(clippy::result_large_err, clippy::too_many_arguments)]
    pub fn new_v2_collection(
        prepared: PreparedRoventNativeV2<PcPortQuestion>,
        loaded: Option<LoadedRoventNnuecSource<rz_experiments::RoventInputLockV2>>,
        process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        hooks: Arc<dyn RoventPcCollectionHooksV2>,
    ) -> Result<Self, RoventPcSessionV2Rejection> {
        Self::construct_v2(
            prepared,
            loaded,
            process,
            clock,
            owners,
            identity,
            root_policy,
            Some(hooks),
        )
    }

    #[allow(clippy::result_large_err, clippy::too_many_arguments)]
    fn construct_v2(
        prepared: PreparedRoventNativeV2<PcPortQuestion>,
        loaded: Option<LoadedRoventNnuecSource<rz_experiments::RoventInputLockV2>>,
        process: Arc<VerifiedOwnedRoventProcessBoundaryV2>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        collection_hooks_v2: Option<Arc<dyn RoventPcCollectionHooksV2>>,
    ) -> Result<Self, RoventPcSessionV2Rejection> {
        let build = (|| -> Result<_, SearchSessionFailure> {
            let sources = prepared.sources();
            if !Arc::ptr_eq(sources.lock_owner(), process.input_lock())
                || !process.matches_startup_control(prepared.startup_control())
                || sources.private_root_path() != process.private_cwd()
                || !sources
                    .same_original_root(process.original_root())
                    .map_err(|e| SearchSessionFailure::debug("RoventPcV2Root", &e))?
            {
                return Err(failure(
                    "RoventPcV2Custody",
                    "original lock, root, or startup control differs",
                ));
            }
            let configuration = sources.configuration().clone();
            configuration
                .require_supported_native_capabilities()
                .map_err(|e| SearchSessionFailure::debug("RoventPcCapability", &e))?;
            match (configuration.fields().execution, &loaded) {
                (
                    ExecutionConfiguration::RolesOnly {
                        shared_nnuec_readout: None,
                        ..
                    },
                    None,
                ) if sources.lock().manifest.engine.nnuec.is_none() => {}
                (
                    ExecutionConfiguration::RolesOnly {
                        shared_nnuec_readout: Some(readout),
                        ..
                    },
                    Some(loaded),
                ) => {
                    if !declares_shared_nnuec_helper(&sources.lock().manifest.resources.pools) {
                        return Err(failure(
                            "RoventPcNnuecResources",
                            "shared readout requires one declared RuntimeHelper thread",
                        ));
                    }
                    let actual = loaded.model().registration().map_err(|error| {
                        SearchSessionFailure::debug("RoventPcNnuecRegistration", &error)
                    })?;
                    if !loaded.matches_sources(sources)
                        || loaded.input_lock_digest() != prepared.snapshot().input_lock
                        || loaded.registration() != &actual
                        || readout.context_namespace_policy != actual.context_namespace_policy()
                        || readout.local_namespace_policy != actual.local_namespace_policy()
                    {
                        return Err(failure(
                            "RoventPcNnuecStartup",
                            "original NNUEC source/registration/namespace mismatch",
                        ));
                    }
                }
                _ => {
                    return Err(failure(
                        "RoventPcV2Startup",
                        "shared readout and original loaded NNUEC owner must be selected together",
                    ));
                }
            }
            let collection_origin_v2 = match (
                sources.lock().manifest.purpose,
                sources.collection_preload_sources().is_some(),
                collection_hooks_v2.is_some(),
            ) {
                (rz_experiments::RoventPurpose::Collection, true, true) => {
                    Some(prepared.collection_identity().map_err(|error| {
                        SearchSessionFailure::debug("RoventPcV2CollectionOrigin", &error)
                    })?)
                }
                (rz_experiments::RoventPurpose::Collection, _, _) | (_, true, _) | (_, _, true) => {
                    return Err(failure(
                        "RoventPcV2CollectionUnsupported",
                        "V2 collection requires its original preload and explicit hook owner",
                    ));
                }
                _ => None,
            };
            if prepared.snapshot().mode.has_native_r()
                || prepared.snapshot().process_epoch != clock.epoch().0
                || identity.name.is_empty()
                || identity.author.is_empty()
            {
                return Err(failure(
                    "RoventPcV2Startup",
                    "unsupported V2 session profile",
                ));
            }
            let native_clock = prepared.service.ready_clock().ok_or_else(|| {
                failure("RoventPcV2Startup", "actual native Ready clock is absent")
            })?;
            let original_clock = clock.runtime_clock();
            if prepared.service.is_quarantined()
                || prepared.service.native_acceptance_verifier().is_none()
                || native_clock.domain() != original_clock.domain()
                || native_clock.origin() != original_clock.origin()
                || prepared.snapshot().input_lock != sources.lock_digest()
            {
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
            let original_owner =
                rz_experiments::RoventOriginalInputLock::from_v2(Arc::clone(sources.lock_owner()))
                    .map_err(|e| SearchSessionFailure::debug("RoventPcV2Lock", &e.error))?;
            let namespace = prepared
                .snapshot()
                .model_namespace()
                .map_err(|e| SearchSessionFailure::debug("RoventPcNamespace", &e))?;
            let model = AdaptivePortModel::new_with_namespace(
                prepared.snapshot().role_model_identity(),
                Some(
                    prepared
                        .snapshot()
                        .model_value_identity()
                        .map_err(|e| SearchSessionFailure::debug("RoventPcNamespace", &e))?,
                ),
                namespace,
            )
            .map_err(|e| SearchSessionFailure::debug("RoventPcModelMirror", &e))?;
            let owner = PcSemanticOwner::new_selected_rovent(
                configuration
                    .fields()
                    .semantic
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?,
                model,
                configuration
                    .fields()
                    .review
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?,
                configuration
                    .fields()
                    .index
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?,
            )
            .map_err(|e| SearchSessionFailure::debug("RoventPcSemanticOwner", &e))?;
            let implementation =
                Self::implementation_digest_from_metadata(&configuration, root_policy)
                    .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?;
            let selective_policy = Digest(
                configuration
                    .search_configuration_digest()
                    .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?,
            );
            let frontier_limits = configuration
                .fields()
                .frontier
                .actual()
                .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?;
            Ok((
                configuration,
                collection_origin_v2,
                original_owner,
                owner,
                implementation,
                selective_policy,
                frontier_limits,
            ))
        })();
        let (
            configuration,
            collection_origin_v2,
            original_owner,
            owner,
            implementation,
            selective_policy,
            frontier_limits,
        ) = match build {
            Ok(value) => value,
            Err(failure) => {
                return Err(RoventPcSessionV2Rejection {
                    failure,
                    prepared,
                    process,
                    loaded,
                    collection_hooks: collection_hooks_v2,
                });
            }
        };
        let (native, snapshot, sources) = prepared.into_parts();
        let issuer: AdaptiveRulesOwnerIssuer = Arc::new(move || owners.allocate());
        Ok(Self {
            sources: PcSources::V2(sources),
            original_v2_owner: Some(original_owner),
            process_v2: Some(process),
            snapshot,
            configuration,
            clock,
            issuer,
            identity,
            implementation,
            root_policy,
            collection_origin: None,
            collection_hooks: RoventPcCollectionHooks::default(),
            collection_origin_v2,
            collection_hooks_v2,
            collection_v2_outstanding: AtomicBool::new(false),
            retained_collection_handoff_v2: Mutex::new(None),
            retained_collection_custody_v2: Mutex::new(None),
            retained_no_capture_v2: Mutex::new(None),
            retained_deferred_cleanup_v2: Mutex::new(None),
            retained_collection_failure_v2: Mutex::new(None),
            retained_collection_panic_v2: Mutex::new((None, None)),
            selective_policy,
            frontier_limits,
            closed: AtomicBool::new(false),
            reset_requested: AtomicBool::new(false),
            state: Mutex::new(State::Idle(Box::new(Idle {
                native: NativeOwner::new(native),
                owner: Some(Box::new(owner)),
                next_request_sequence: 1,
                game: None,
                custody: IdleCustody::StartupReady,
            }))),
            latest: Mutex::new(None),
            latest_initial: Mutex::new(None),
            latest_v2: Mutex::new(None),
            latest_initial_v2: Mutex::new(None),
            retained_repair: Mutex::new(None),
            shared_nnuec: loaded.map(|loaded| SharedNnuecOwner {
                loaded: SharedNnuecLoaded::V2(loaded),
                actor: Mutex::new(SharedNnuecActor::default()),
            }),
            storage: Mutex::new(None),
        })
    }

    #[allow(clippy::result_large_err)]
    pub fn new(
        prepared: PreparedRoventNative<PcPortQuestion>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventPcSessionRejection> {
        Self::new_with_collection_hooks(
            prepared,
            clock,
            owners,
            identity,
            root_policy,
            RoventPcCollectionHooks::default(),
        )
    }
    #[allow(clippy::result_large_err)]
    pub fn new_with_collection_hooks(
        prepared: PreparedRoventNative<PcPortQuestion>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        collection_hooks: RoventPcCollectionHooks,
    ) -> Result<Self, RoventPcSessionRejection> {
        Self::construct(
            prepared,
            None,
            clock,
            owners,
            identity,
            root_policy,
            collection_hooks,
        )
        .map_err(|rejected| RoventPcSessionRejection {
            failure: rejected.failure,
            prepared: rejected.prepared,
        })
    }
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn from_prepared_with_shared_nnuec(
        prepared: PreparedRoventNative<PcPortQuestion>,
        loaded: LoadedRoventNnuecSource,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventPcSharedNnuecSessionRejection> {
        Self::from_prepared_with_shared_nnuec_and_collection_hooks(
            prepared,
            loaded,
            clock,
            owners,
            identity,
            root_policy,
            RoventPcCollectionHooks::default(),
        )
    }
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn from_prepared_with_shared_nnuec_and_collection_hooks(
        prepared: PreparedRoventNative<PcPortQuestion>,
        loaded: LoadedRoventNnuecSource,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        collection_hooks: RoventPcCollectionHooks,
    ) -> Result<Self, RoventPcSharedNnuecSessionRejection> {
        Self::construct(
            prepared,
            Some(loaded),
            clock,
            owners,
            identity,
            root_policy,
            collection_hooks,
        )
        .map_err(|rejected| RoventPcSharedNnuecSessionRejection {
            failure: rejected.failure,
            prepared: rejected.prepared,
            loaded: rejected.loaded.expect("original shared source retained"),
        })
    }
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn construct(
        prepared: PreparedRoventNative<PcPortQuestion>,
        loaded: Option<LoadedRoventNnuecSource>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        collection_hooks: RoventPcCollectionHooks,
    ) -> Result<Self, ConstructionRejection> {
        let build = (|| -> Result<_, SearchSessionFailure> {
            let configuration = prepared.sources().configuration().clone();
            let collection_origin = match prepared.sources().lock().manifest.purpose {
                rz_experiments::RoventPurpose::Collection
                    if collection_hooks.root_admission.is_some() =>
                {
                    Some(prepared.collection_identity().map_err(|error| {
                        SearchSessionFailure::debug("RoventPcCollectionOrigin", &error)
                    })?)
                }
                rz_experiments::RoventPurpose::Collection => {
                    return Err(failure(
                        "RoventPcCollectionHooks",
                        "Collection requires the actual root observer",
                    ));
                }
                _ if collection_hooks.root_admission.is_some()
                    || collection_hooks.root_closure.is_some()
                    || collection_hooks.cooperative.is_some() =>
                {
                    return Err(failure(
                        "RoventPcCollectionHooks",
                        "collection hooks require the explicit Collection source selection",
                    ));
                }
                _ => None,
            };
            configuration
                .require_supported_native_capabilities()
                .map_err(|e| SearchSessionFailure::debug("RoventPcCapability", &e))?;
            match (configuration.fields().execution, &loaded) {
                (
                    ExecutionConfiguration::RolesOnly {
                        shared_nnuec_readout: None,
                        ..
                    },
                    None,
                ) => {}
                (
                    ExecutionConfiguration::RolesOnly {
                        shared_nnuec_readout: Some(readout),
                        ..
                    },
                    Some(loaded),
                ) => {
                    if !declares_shared_nnuec_helper(
                        &prepared.sources().lock().manifest.resources.pools,
                    ) {
                        return Err(failure(
                            "RoventPcNnuecResources",
                            "shared readout requires one declared RuntimeHelper thread",
                        ));
                    }
                    let actual = loaded.model().registration().map_err(|error| {
                        SearchSessionFailure::debug("RoventPcNnuecRegistration", &error)
                    })?;
                    if !loaded.matches_sources(prepared.sources())
                        || loaded.input_lock_digest() != prepared.snapshot().input_lock
                        || loaded.registration() != &actual
                        || readout.context_namespace_policy != actual.context_namespace_policy()
                        || readout.local_namespace_policy != actual.local_namespace_policy()
                    {
                        return Err(failure(
                            "RoventPcNnuecStartup",
                            "original NNUEC source/registration/namespace mismatch",
                        ));
                    }
                }
                _ => {
                    return Err(failure(
                        "RoventPcNnuecStartup",
                        "shared readout declaration and original loaded owner must be selected together",
                    ));
                }
            }
            if !matches!(
                configuration.fields().execution,
                ExecutionConfiguration::RolesOnly { .. }
            ) || prepared.snapshot().mode.has_native_r()
                || prepared.snapshot().process_epoch != clock.epoch().0
                || identity.name.is_empty()
                || identity.author.is_empty()
            {
                return Err(failure("RoventPcStartup", "invalid actual R-free startup"));
            }
            let native_clock = prepared
                .service
                .ready_clock()
                .ok_or_else(|| failure("RoventPcStartup", "actual native Ready clock is absent"))?;
            let original_clock = clock.runtime_clock();
            if prepared.service.is_quarantined()
                || native_clock.domain() != original_clock.domain()
                || native_clock.origin() != original_clock.origin()
            {
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
            let namespace = prepared
                .snapshot()
                .model_namespace()
                .map_err(|e| SearchSessionFailure::debug("RoventPcNamespace", &e))?;
            let model = AdaptivePortModel::new_with_namespace(
                prepared.snapshot().role_model_identity(),
                Some(
                    prepared
                        .snapshot()
                        .model_value_identity()
                        .map_err(|e| SearchSessionFailure::debug("RoventPcNamespace", &e))?,
                ),
                namespace,
            )
            .map_err(|e| SearchSessionFailure::debug("RoventPcModelMirror", &e))?;
            let owner = PcSemanticOwner::new_selected_rovent(
                configuration
                    .fields()
                    .semantic
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?,
                model,
                configuration
                    .fields()
                    .review
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?,
                configuration
                    .fields()
                    .index
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?,
            )
            .map_err(|e| SearchSessionFailure::debug("RoventPcSemanticOwner", &e))?;
            let implementation =
                Self::implementation_digest_from_metadata(&configuration, root_policy)
                    .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?;
            let selective_policy = Digest(
                configuration
                    .search_configuration_digest()
                    .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?,
            );
            let frontier_limits = configuration
                .fields()
                .frontier
                .actual()
                .map_err(|e| SearchSessionFailure::debug("RoventPcConfiguration", &e))?;
            Ok((
                configuration,
                collection_origin,
                owner,
                implementation,
                selective_policy,
                frontier_limits,
            ))
        })();
        let (
            configuration,
            collection_origin,
            owner,
            implementation,
            selective_policy,
            frontier_limits,
        ) = match build {
            Ok(value) => value,
            Err(failure) => {
                return Err(ConstructionRejection {
                    failure,
                    prepared,
                    loaded,
                });
            }
        };
        let (native, snapshot, sources) = prepared.into_parts();
        let issuer: AdaptiveRulesOwnerIssuer = Arc::new(move || owners.allocate());
        Ok(Self {
            sources: PcSources::V1(sources),
            original_v2_owner: None,
            process_v2: None,
            snapshot,
            configuration,
            clock,
            issuer,
            identity,
            implementation,
            root_policy,
            collection_origin,
            collection_hooks,
            collection_origin_v2: None,
            collection_hooks_v2: None,
            collection_v2_outstanding: AtomicBool::new(false),
            retained_collection_handoff_v2: Mutex::new(None),
            retained_collection_custody_v2: Mutex::new(None),
            retained_no_capture_v2: Mutex::new(None),
            retained_deferred_cleanup_v2: Mutex::new(None),
            retained_collection_failure_v2: Mutex::new(None),
            retained_collection_panic_v2: Mutex::new((None, None)),
            selective_policy,
            frontier_limits,
            closed: AtomicBool::new(false),
            reset_requested: AtomicBool::new(false),
            state: Mutex::new(State::Idle(Box::new(Idle {
                native: NativeOwner::new(native),
                owner: Some(Box::new(owner)),
                next_request_sequence: 1,
                game: None,
                custody: IdleCustody::StartupReady,
            }))),
            latest: Mutex::new(None),
            latest_initial: Mutex::new(None),
            latest_v2: Mutex::new(None),
            latest_initial_v2: Mutex::new(None),
            retained_repair: Mutex::new(None),
            shared_nnuec: loaded.map(|loaded| SharedNnuecOwner {
                loaded: SharedNnuecLoaded::V1(loaded),
                actor: Mutex::new(SharedNnuecActor::default()),
            }),
            storage: Mutex::new(None),
        })
    }
    #[allow(clippy::result_large_err)]
    pub(crate) fn install_storage_registration(
        &self,
        original: RoventStorageOwnedRuntimeRegistration,
    ) -> Result<
        Arc<SharedRoventStorageRegistration>,
        (RoventStorageOwnedRuntimeRegistration, SearchSessionFailure),
    > {
        let Some(sources) = self.sources.v1() else {
            return Err((
                original,
                failure("RoventPcV2Storage", "V2 storage is not connected"),
            ));
        };
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(_) => {
                return Err((
                    original,
                    physical_unknown_at(
                        "RoventPcStorageInstall",
                        "original session owner is busy/poisoned",
                    ),
                ));
            }
        };
        let mut slot = match self.storage.try_lock() {
            Ok(slot) => slot,
            Err(_) => {
                return Err((
                    original,
                    physical_unknown_at("RoventPcStorageInstall", "storage slot is busy/poisoned"),
                ));
            }
        };
        let valid_idle = matches!(&*state, State::Idle(idle)
            if matches!(idle.custody, IdleCustody::StartupReady)
                && idle.game.is_none() && idle.owner.as_ref().is_some_and(|owner| owner.stores().generation() == 0 && owner.stores().root().is_none())
                && idle.native.finished.is_none() && !idle.native.joined
                && !idle.native.service.is_quarantined() && idle.native.service.occupancy() == 0);
        if slot.is_some() || self.closed.load(Ordering::Acquire) || !valid_idle {
            return Err((
                original,
                physical_unknown_at(
                    "RoventPcStorageInstall",
                    "storage requires the unused original native StartupReady owner",
                ),
            ));
        }
        if let Err(error) =
            original.validate_session_attachment(sources.lock(), &self.clock, self.implementation)
        {
            return Err((
                original,
                SearchSessionFailure::debug("RoventPcStorageNamespace", &error),
            ));
        }
        if let Err(error) = original
            .profile()
            .validate_for_manifest(&sources.lock().manifest)
        {
            return Err((
                original,
                SearchSessionFailure::debug("RoventPcStorageResources", &error),
            ));
        }
        let shared = Arc::new(SharedRoventStorageRegistration::new(original));
        *slot = Some(SessionStorageLifecycle::new(Arc::clone(&shared)));
        Ok(shared)
    }
    pub fn shared_nnuec_readback(
        &self,
    ) -> Result<Option<RoventPcSharedNnuecReadback>, ContractError> {
        self.shared_nnuec
            .as_ref()
            .map(|owner| {
                owner
                    .actor
                    .try_lock()
                    .map(|actor| actor.readback.clone())
                    .map_err(|_| poisoned())
            })
            .transpose()
    }
    pub fn latest_root(&self) -> Result<Option<Arc<RoventPcRootReceipt>>, ContractError> {
        self.latest
            .lock()
            .map(|value| value.clone())
            .map_err(|_| poisoned())
    }
    pub(crate) fn storage_observation(
        &self,
    ) -> Result<Option<super::storage_lifecycle::StorageLifecycleObservation>, &'static str> {
        let state = self
            .state
            .try_lock()
            .map_err(|_| "storage_semantic_owner_busy_or_poisoned")?;
        let storage = self
            .storage
            .try_lock()
            .map_err(|_| "storage_lifecycle_busy_or_poisoned")?;
        let Some(storage) = storage.as_ref() else {
            return Ok(None);
        };
        let mut observation = storage.observation();
        if let State::Idle(idle) = &*state {
            if let Some(owner) = &idle.owner {
                let archive = owner
                    .stores()
                    .archive_owner_snapshot()
                    .map_err(|_| "storage_hot_owner_readback_failed")?;
                observation.observe_hot_owner(
                    owner.records().len(),
                    owner.stores().hot_stats(),
                    archive,
                );
            } else {
                observation.complete = false;
            }
        } else if !matches!(&*state, State::Closed) {
            observation.complete = false;
        }
        Ok(Some(observation))
    }
    pub fn latest_initial(&self) -> Result<Option<Arc<RoventPcInitialReceipt>>, ContractError> {
        self.latest_initial
            .lock()
            .map(|value| value.clone())
            .map_err(|_| poisoned())
    }
    pub fn latest_root_v2(&self) -> Result<Option<Arc<RoventPcRootReceiptV2>>, ContractError> {
        self.latest_v2
            .lock()
            .map(|value| value.clone())
            .map_err(|_| poisoned())
    }
    /// Diagnostic sampling must not delay the original UCI response.
    pub(crate) fn try_observe_root_v2(
        &self,
    ) -> Result<Option<Arc<RoventPcRootReceiptV2>>, &'static str> {
        self.latest_v2
            .try_lock()
            .map(|value| value.clone())
            .map_err(|_| "pc_v2_root_busy_or_poisoned")
    }
    pub fn latest_initial_v2(
        &self,
    ) -> Result<Option<Arc<RoventPcInitialReceiptV2>>, ContractError> {
        self.latest_initial_v2
            .lock()
            .map(|value| value.clone())
            .map_err(|_| poisoned())
    }
    pub(crate) fn try_observe_initial_v2(
        &self,
    ) -> Result<Option<Arc<RoventPcInitialReceiptV2>>, &'static str> {
        self.latest_initial_v2
            .try_lock()
            .map(|value| value.clone())
            .map_err(|_| "pc_v2_initial_busy_or_poisoned")
    }
    /// A failed transfer cannot silently discard the already recovered report.
    /// Only a closed root owner exposes this move; live work keeps its custody.
    pub fn take_failed_repair_report(
        &self,
    ) -> Result<Option<rz_search::pals::engine::pc_port::PcRepairReport>, ContractError> {
        if !self.closed.load(Ordering::Acquire) {
            return Err(ContractError::new(
                rz_contracts::ErrorCode::InvalidInput,
                rz_contracts::Stage::Output,
                "repair recovery requires a closed product root owner",
            ));
        }
        self.retained_repair
            .lock()
            .map(|mut report| report.take())
            .map_err(|_| poisoned())
    }
    fn run_root(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        if self.closed.load(Ordering::Acquire) {
            return Err(physical_unknown_at(
                "RoventPcRootAdmission",
                "session is closed or quarantined",
            ));
        }
        let cap = NonZeroU16::new(self.configuration.fields().frontier.maximum_goal_plies)
            .expect("admitted nonzero goal cap");
        context
            .goal
            .validate(cap)
            .map_err(|e| SearchSessionFailure::debug("RoventPcGoal", &e))?;
        if context.authority.implementation != self.implementation
            || context.authority.epoch != self.clock.epoch()
        {
            return Err(failure(
                "RoventPcAuthority",
                "foreign original process authority",
            ));
        }
        let legal = position.legal_moves();
        if context.root_moves.as_ref().is_some_and(|moves| {
            moves.is_empty() || moves.iter().any(|movement| !legal.contains(movement))
        }) {
            return Err(failure(
                "RoventPcRootRestriction",
                "invalid original root restriction",
            ));
        }
        let source = match &self.sources {
            PcSources::V1(sources) => PcPreparedRoot::V1(
                SourceSeal::prepare(
                    sources.lock(),
                    self.root_policy,
                    position,
                    context,
                    &self.clock,
                )
                .map_err(|error| SearchSessionFailure::debug("RoventPcOriginalRoot", &error))?,
            ),
            PcSources::V2(_) => PcPreparedRoot::V2(
                PreparedSourceV2::prepare(
                    self.original_v2_owner
                        .as_ref()
                        .expect("V2 source retains original owner"),
                    self.root_policy,
                    position,
                    context,
                    &self.clock,
                )
                .map_err(|error| SearchSessionFailure::debug("RoventPcOriginalRoot", &error))?,
            ),
        };
        let mut state = loop {
            match self.state.try_lock() {
                Ok(state) => break state,
                Err(TryLockError::Poisoned(_)) => {
                    return Err(physical_unknown_at(
                        "RoventPcRootState",
                        "session owner mutex is poisoned",
                    ));
                }
                Err(TryLockError::WouldBlock) => {
                    if Instant::now() >= context.shutdown_deadline {
                        return Err(physical_unknown_at(
                            "RoventPcRootState",
                            "original cleanup boundary expired while acquiring session custody",
                        ));
                    }
                    std::thread::yield_now();
                }
            }
        };
        if self.closed.load(Ordering::Acquire) {
            return Err(physical_unknown_at(
                "RoventPcRootAdmission",
                "session closed before source gate",
            ));
        }
        // Hold the same actual idle owner through the final source gate. A
        // stopped Start consumes neither reset intent nor root registration.
        let idle = match &*state {
            State::Idle(idle)
                if idle.native.finished.is_none()
                    && !idle.native.joined
                    && idle.owner.is_some()
                    && !idle.native.service.is_quarantined() =>
            {
                idle
            }
            State::Idle(idle) => {
                return Err(physical_unknown_at(
                    "RoventPcNativeIdle",
                    format!(
                        "finished={:?}, joined={}, quarantined={}",
                        idle.native.finished,
                        idle.native.joined,
                        idle.native.service.is_quarantined()
                    ),
                ));
            }
            _ => {
                return Err(physical_unknown_at(
                    "RoventPcRootState",
                    "original idle custody is absent",
                ));
            }
        };
        let root_admission = match source {
            PcPreparedRoot::V1(source) => match source
                .into_initial()
                .map_err(|error| SearchSessionFailure::debug("RoventPcOriginalRoot", &error))?
            {
                InitialAdmissionOutcome::Admitted(admitted) => {
                    PcRootAdmission::V1(Arc::new(admitted))
                }
                InitialAdmissionOutcome::StoppedInitial(stopped) => {
                    let receipt = Arc::new(RoventPcInitialReceipt {
                        authority: context.authority,
                        input_lock: self.snapshot.input_lock,
                        source: Arc::new(stopped),
                        idle_custody: idle.custody.clone(),
                    });
                    let report = receipt.report();
                    match self.latest_initial.lock() {
                        Ok(mut latest) => *latest = Some(receipt),
                        Err(_) => {
                            self.closed.store(true, Ordering::Release);
                            return Err(physical_unknown_at(
                                "RoventPcStoppedInitial",
                                "initial receipt owner is poisoned",
                            ));
                        }
                    }
                    return Ok(report);
                }
            },
            PcPreparedRoot::V2(source) => match source
                .into_initial()
                .map_err(|error| SearchSessionFailure::debug("RoventPcOriginalRoot", &error))?
            {
                InitialAdmissionOutcomeV2::Admitted(admitted) => {
                    PcRootAdmission::V2(Arc::new(admitted))
                }
                InitialAdmissionOutcomeV2::StoppedInitial(stopped) => {
                    let previous = match &idle.custody {
                        IdleCustody::RetiredRootV2(value) => Some(Arc::clone(value)),
                        _ => None,
                    };
                    let fallback = stopped.fallback();
                    let receipt = Arc::new(RoventPcInitialReceiptV2 {
                        authority: context.authority,
                        input_lock: self.snapshot.input_lock,
                        source: Arc::new(stopped),
                        previous,
                    });
                    match self.latest_initial_v2.lock() {
                        Ok(mut latest) => *latest = Some(receipt),
                        Err(_) => {
                            self.closed.store(true, Ordering::Release);
                            return Err(physical_unknown_at(
                                "RoventPcStoppedInitialV2",
                                "initial receipt owner is poisoned",
                            ));
                        }
                    }
                    return Ok(initial_stop_report(fallback));
                }
            },
        };
        let root_budget_digest = root_admission.budget_digest()?;
        let retained = std::mem::replace(&mut *state, State::Moving);
        let mut idle = match retained {
            State::Idle(idle) => idle,
            other => {
                *state = other;
                return Err(physical_unknown_at(
                    "RoventPcRootTransfer",
                    "original idle custody changed before transfer",
                ));
            }
        };
        if idle.native.finished.is_some() || idle.native.joined {
            *state = State::Idle(idle);
            self.closed.store(true, Ordering::Release);
            return Err(physical_unknown_at(
                "RoventPcNativeIdle",
                "native owner already finished or joined before root transfer",
            ));
        }
        let reset = self.reset_requested.swap(false, Ordering::AcqRel)
            || idle.game != Some(context.authority.game.0);
        let mut storage = match self.storage.try_lock() {
            Ok(storage) => storage,
            Err(_) => {
                *state = State::Idle(idle);
                self.closed.store(true, Ordering::Release);
                return Err(physical_unknown_at(
                    "RoventPcStorage",
                    "original storage slot is busy/poisoned",
                ));
            }
        };
        if reset && !storage.as_ref().is_some_and(|storage| storage.configured()) {
            if let Err(error) = idle
                .owner
                .as_mut()
                .expect("original semantic owner")
                .try_adaptive_new_game(
                    context.authority,
                    context.control.deadline,
                    &context.control.cancellation,
                )
            {
                *state = State::Idle(idle);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::debug("RoventPcNewGame", &error));
            }
            idle.game = Some(context.authority.game.0);
        }
        if let Some(storage) = storage.as_mut() {
            let original_admission = match &root_admission {
                PcRootAdmission::V1(value) => Arc::clone(value),
                PcRootAdmission::V2(_) => {
                    *state = State::Idle(idle);
                    return Err(failure("RoventPcV2Storage", "V2 storage is not connected"));
                }
            };
            let owner = idle.owner.take().expect("original semantic owner");
            let signal = idle.native.service.signal();
            match storage.prepare(
                StorageSemanticOwner::Pc(owner),
                original_admission,
                reset,
                position,
                &signal,
            ) {
                Ok(StorageSemanticOwner::Pc(owner)) => {
                    idle.owner = Some(owner);
                    idle.game = Some(context.authority.game.0);
                }
                Ok(owner) => {
                    storage.retain_wrong_return(owner);
                    *state = State::Idle(idle);
                    self.closed.store(true, Ordering::Release);
                    return Err(physical_unknown_at(
                        "RoventPcStorageReturn",
                        "returned semantic kind differs",
                    ));
                }
                Err(mut rejected) => {
                    if let Some(owner) = rejected.owner.take() {
                        match owner {
                            StorageSemanticOwner::Pc(owner) => idle.owner = Some(owner),
                            owner => storage.retain_wrong_return(owner),
                        }
                    }
                    *state = State::Idle(idle);
                    self.closed.store(true, Ordering::Release);
                    return Err(rejected.failure);
                }
            }
        }
        drop(storage);
        let registration = match idle
            .owner
            .as_mut()
            .expect("original semantic owner")
            .register_adaptive_root(position, context.authority)
        {
            Ok(value) => value,
            Err(error) => {
                *state = State::Idle(idle);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::debug(
                    "RoventPcRootRegistration",
                    &error,
                ));
            }
        };
        let original_deadline = match self.clock.deadline(context.control.deadline) {
            Ok(value) => value,
            Err(error) => {
                *state = State::Idle(idle);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::debug("RoventPcClock", &error));
            }
        };
        // This root retains the same optional actor slot through every manager
        // transfer. A failed/unknown root leaves the actor and loaded model in
        // session custody, including when the manager is quarantined.
        let mut shared_actor = match self
            .shared_nnuec
            .as_ref()
            .map(|owner| owner.actor.try_lock())
            .transpose()
        {
            Ok(actor) => actor,
            Err(_) => {
                *state = State::Idle(idle);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        if let (Some(owner), Some(actor)) = (&self.shared_nnuec, shared_actor.as_mut()) {
            if actor.actor.is_none() {
                if actor.readback.spawned_runtime_helpers != 0 {
                    *state = State::Idle(idle);
                    self.closed.store(true, Ordering::Release);
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
                let signal = idle.native.service.signal();
                let created = match PreparationActor::spawn_notified(
                    Arc::clone(owner.loaded.model()),
                    signal.clone(),
                ) {
                    Ok(created) => created,
                    Err(error) => {
                        *state = State::Idle(idle);
                        self.closed.store(true, Ordering::Release);
                        return Err(SearchSessionFailure::debug("RoventPcNnuecActor", &error));
                    }
                };
                actor.readback.spawned_runtime_helpers = 1;
                actor.readback.worker_thread_id = created.worker_thread_id();
                actor.signal = signal;
                actor.actor = Some(created);
            }
        }
        let Idle {
            native,
            owner,
            next_request_sequence,
            game: _,
            custody: _,
        } = *idle;
        let owner = owner.expect("original semantic owner returned from archive");
        let mut port = match PcSemanticPort::new_registered_retaining(
            *owner,
            position.clone(),
            context.authority,
            original_deadline,
            context.control.deadline,
            self.snapshot.mode,
            Arc::clone(&self.issuer),
            registration,
            context.root_moves.as_deref(),
        ) {
            Ok(port) => port,
            Err((owner, error)) => {
                *state = State::RejectedOwner {
                    native,
                    owner: Box::new(owner),
                };
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::debug("RoventPcPort", &error));
            }
        };
        let model_namespace = match port.model_namespace() {
            Ok(namespace) => Arc::new(namespace.clone()),
            Err(error) => {
                *state = State::RejectedPort {
                    native,
                    port: PcPortDriver::new(port),
                };
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::debug("RoventPcNamespace", &error));
            }
        };
        let frontier = FrontierRegistration {
            authority: context.authority,
            mode: self.snapshot.mode,
            original_deadline,
            root_budget: root_budget_digest,
            goal: context.goal,
            nodes: context.requested_nodes,
            selective_policy: self.selective_policy,
            root_rules: port.root_rules(),
            root_moves: Arc::clone(port.root_move_domain()),
            root_state: port.root_state_id(),
            model_namespace,
            native_ledger: None,
        };
        let root_budget = frontier.root_budget;
        let configured = configure_root_port(
            &mut port,
            frontier,
            self.frontier_limits.clone(),
            PcFrontierStartup {
                root_budget,
                clock: self.clock.b_clock(),
            },
            self.configuration
                .post_repair_policy_for(self.snapshot.mode),
        );
        if let Err(error) = configured {
            *state = State::RejectedPort {
                native,
                port: PcPortDriver::new(port),
            };
            self.closed.store(true, Ordering::Release);
            return Err(error);
        }
        let collection_admission = match &self.collection_origin {
            Some(origin) => {
                let original_admission = match &root_admission {
                    PcRootAdmission::V1(value) => Arc::clone(value),
                    PcRootAdmission::V2(_) => {
                        *state = State::RejectedPort {
                            native,
                            port: PcPortDriver::new(port),
                        };
                        self.closed.store(true, Ordering::Release);
                        return Err(failure(
                            "RoventPcV2Collection",
                            "V2 Collection is not connected",
                        ));
                    }
                };
                let admission = port
                    .rovent_frontier()
                    .ok_or_else(|| failure("RoventPcCollectionRoot", "actual Frontier is absent"))
                    .and_then(|frontier| {
                        RoventCollectionRootAdmission::from_attached_frontier(
                            origin,
                            frontier,
                            self.snapshot.input_lock,
                            root_budget,
                            self.selective_policy,
                            original_admission,
                        )
                        .map_err(|error| {
                            SearchSessionFailure::debug("RoventPcCollectionRoot", &error)
                        })
                    });
                match admission {
                    Ok(admission) => Some(admission),
                    Err(error) => {
                        *state = State::RejectedPort {
                            native,
                            port: PcPortDriver::new(port),
                        };
                        self.closed.store(true, Ordering::Release);
                        return Err(error);
                    }
                }
            }
            None => None,
        };
        let sources = RoleOnlyManagerSources {
            native: native.service,
            driver: PcPortDriver::new(port),
        };
        let registration = RoleOnlyManagerRegistration {
            authority: context.authority,
            mode: self.snapshot.mode,
            root_budget,
            hard_deadline: context.control.deadline,
            token_budget: self.configuration.fields().review.token_budget,
            first_request_sequence: next_request_sequence,
            role_tickets_max: self.configuration.fields().execution.role_tickets_max(),
            role_call_limit: self.configuration.fields().semantic.max_role_calls,
            event_order: self.configuration.fields().execution.event_order().actual(),
        };
        let mut manager =
            match Manager::new_retaining(registration, self.clock.runtime_clock(), sources) {
                Ok(value) => Box::new(value),
                Err(rejected) => {
                    *state = State::RejectedPort {
                        native: NativeOwner::new(rejected.sources.native),
                        port: rejected.sources.driver,
                    };
                    self.closed.store(true, Ordering::Release);
                    return Err(SearchSessionFailure::debug(
                        "RoventPcManagerAdmission",
                        &rejected.error,
                    ));
                }
            };
        let mut v2_hook_started = false;
        let mut v2_hook_revoked = false;
        let mut v2_admission_failure = None;
        let mut v2_current_admission = None;
        let mut no_capture_v2 = None;
        let run = catch_unwind(AssertUnwindSafe(
            || -> Result<(), crate::pals_adaptive::AdaptiveManagerError> {
                manager.register_cleanup_deadline(context.shutdown_deadline)?;
                if manager.registered_cleanup_deadline() != Some(context.shutdown_deadline) {
                    return Err(crate::pals_adaptive::AdaptiveManagerError::Invalid(
                        "P/C original cleanup boundary",
                    ));
                }
                if let Some(clock) = &context.effective_clock {
                    manager.register_effective_clock(clock.clone())?;
                }
                if let Some(owner) = &self.shared_nnuec {
                    manager.enable_shared_nnuec_preparation(Arc::clone(owner.loaded.model()))?;
                }
                if let (Some(observer), Some(admission)) =
                    (&self.collection_hooks.root_admission, collection_admission)
                {
                    observer(&mut manager, admission)?;
                }
                if let Some(origin) = &self.collection_origin_v2 {
                    let original_root = match &root_admission {
                        PcRootAdmission::V2(root) => Arc::clone(root),
                        PcRootAdmission::V1(_) => {
                            return Err(AdaptiveManagerError::Invalid("P/C V2 root owner absent"));
                        }
                    };
                    let frontier = manager.driver().port().rovent_frontier().ok_or(
                        AdaptiveManagerError::Invalid("P/C V2 actual frontier absent"),
                    )?;
                    let admission = RoventCollectionRootAdmissionV2::from_attached_frontier(
                        origin,
                        frontier,
                        original_root,
                        root_budget,
                        self.selective_policy,
                    )?;
                    let expected = admission.clone();
                    v2_current_admission = Some(expected.clone());
                    let hook =
                        self.collection_hooks_v2
                            .as_ref()
                            .ok_or(AdaptiveManagerError::Invalid(
                                "P/C V2 collection hook owner absent",
                            ))?;
                    loop {
                        v2_hook_started = true;
                        self.collection_v2_outstanding.store(true, Ordering::Release);
                        let mut port = RoventPcCollectionAdmissionPortV2 { manager: &mut manager };
                        match hook.admit(&mut port, admission.clone()) {
                            Ok(RoventPcCollectionAdmissionOutcomeV2::Installed) => {
                                let actual = manager.collection_root_admission_v2().ok_or(
                                    AdaptiveManagerError::Invalid("P/C V2 observer was not installed"),
                                )?;
                                if !actual.same_original_admission(&expected) {
                                    return Err(AdaptiveManagerError::Invalid("P/C installed V2 admission owner differs"));
                                }
                                break;
                            }
                            Ok(RoventPcCollectionAdmissionOutcomeV2::DeferredNoCapture { capability, reason }) => {
                                if !capability.same_original_admission(&expected)
                                    || manager.collection_root_admission_v2().is_some() {
                                    v2_admission_failure = Some(RoventPcCollectionAdmissionFailureV2 {
                                        error: AdaptiveManagerError::Invalid("P/C no-capture owner mismatch"),
                                        admission: expected.clone(), owners: Box::new((capability, reason)),
                                    });
                                    return Err(AdaptiveManagerError::Invalid("P/C no-capture owner mismatch"));
                                }
                                no_capture_v2 = Some((capability, reason));
                                v2_hook_started = false;
                                break;
                            }
                            Ok(RoventPcCollectionAdmissionOutcomeV2::Pending { admission }) => {
                                if !admission.same_original_admission(&expected)
                                    || manager.collection_root_admission_v2().is_some() {
                                    v2_admission_failure = Some(RoventPcCollectionAdmissionFailureV2 {
                                        error: AdaptiveManagerError::Invalid("P/C pending admission owner mismatch"),
                                        admission, owners: Box::new(()),
                                    });
                                    return Err(AdaptiveManagerError::Invalid("P/C pending admission owner mismatch"));
                                }
                                v2_hook_started = false;
                                if !context.accepts().unwrap_or(false) || context.soft_expired(Instant::now()) {
                                    return Err(AdaptiveManagerError::Deadline);
                                }
                                std::thread::sleep(Duration::from_millis(1));
                            }
                            Err(rejected) => {
                                let error = if rejected.admission.same_original_admission(&expected) {
                                    rejected.error.clone()
                                } else {
                                    AdaptiveManagerError::Invalid("P/C rejected V2 admission owner differs")
                                };
                                v2_admission_failure = Some(rejected);
                                return Err(error);
                            }
                        }
                    }
                }
                let scanned = manager.driver_mut().port_mut().prepare_rovent_root_rules();
                if let Err(error) = scanned {
                    let goal = manager
                        .driver()
                        .port()
                        .rovent_frontier_receipt()
                        .map(|receipt| receipt.snapshot.goal);
                    if !matches!(
                        (error, goal),
                        (
                            FrontierError::NodeBudget,
                            Some(FrontierGoalStatus::Stopped(FrontierStop::WorkBudget))
                        ) | (
                            FrontierError::Deadline,
                            Some(FrontierGoalStatus::Stopped(FrontierStop::Deadline))
                        )
                    ) {
                        return Err(error.into());
                    }
                }
                let mut published = None;
                loop {
                    let signal = manager.activity_signal();
                    let observed = signal.version();
                    if manager.phase() == ManagerPhase::Running {
                        if !context.accepts().unwrap_or(false)
                            || context.soft_expired(Instant::now())
                        {
                            manager.begin_root_quiesce(context.shutdown_deadline)?;
                        } else {
                            let actual = manager.driver().port().rovent_frontier_receipt().ok_or(
                                crate::pals_adaptive::AdaptiveManagerError::Invalid(
                                    "P/C actual frontier absent",
                                ),
                            )?;
                            match actual.snapshot.goal {
                                FrontierGoalStatus::DepthSatisfied
                                | FrontierGoalStatus::MateSatisfied => {
                                    manager.begin_goal_quiesce(context.shutdown_deadline)?
                                }
                                FrontierGoalStatus::SelectiveFrontier
                                | FrontierGoalStatus::GoalUnresolved
                                | FrontierGoalStatus::Stopped(
                                    FrontierStop::WorkBudget
                                    | FrontierStop::RoleBudget
                                    | FrontierStop::Deadline
                                    | FrontierStop::Canceled,
                                ) => manager.begin_root_quiesce(context.shutdown_deadline)?,
                                FrontierGoalStatus::Stopped(_) => {
                                    return Err(
                                        crate::pals_adaptive::AdaptiveManagerError::PhysicalUnknown,
                                    );
                                }
                                FrontierGoalStatus::Pending => {}
                            }
                        }
                    }
                    if v2_hook_started && !v2_hook_revoked {
                        let revoke_now = Instant::now();
                        let revoke_acceptance = context.accepts();
                        let revoke_soft_expired = context.soft_expired(revoke_now);
                        if !revoke_acceptance.as_ref().copied().unwrap_or(false)
                            || revoke_soft_expired || manager.phase() != ManagerPhase::Running {
                        if let Some(hook) = &self.collection_hooks_v2 {
                            let observation = RoventPcRevokeObservationV2 {
                                origin: RoventPcRevokeOriginV2::LoopCondition,
                                elapsed_from_start: context.original_time_budget
                                    .and_then(|budget| revoke_now.checked_duration_since(budget.start)),
                                context_acceptance: match revoke_acceptance {
                                    Ok(true) => RoventPcContextAcceptanceV2::Accepted,
                                    Ok(false) => RoventPcContextAcceptanceV2::Rejected,
                                    Err(_) => RoventPcContextAcceptanceV2::Unavailable,
                                },
                                soft_expired: revoke_soft_expired,
                                manager_phase: manager.phase(),
                                goal: manager.driver().port().rovent_frontier_receipt()
                                    .map(|receipt| receipt.snapshot.goal),
                            };
                            hook.revoke_root_observed(v2_current_admission.as_ref().ok_or(
                                AdaptiveManagerError::Invalid("P/C revoke original root unavailable"),
                            )?, observation)?;
                        }
                        v2_hook_revoked = true;
                        }
                    }
                    if v2_hook_started {
                        if let Some(hook) = &self.collection_hooks_v2 {
                            hook.pump_root(v2_current_admission.as_ref().ok_or(
                                AdaptiveManagerError::Invalid("P/C pump original root unavailable"),
                            )?)?;
                        }
                    }
                    if let Some(pump) = &self.collection_hooks.cooperative {
                        pump()?;
                    }
                    if let Some(actor) =
                        shared_actor.as_ref().and_then(|owner| owner.actor.as_ref())
                    {
                        manager.pump_shared_nnuec_preparation(actor)?;
                    }
                    let changed = manager.tick()?;
                    if let Some(actor) =
                        shared_actor.as_ref().and_then(|owner| owner.actor.as_ref())
                    {
                        manager.pump_shared_nnuec_preparation(actor)?;
                    }
                    if context.accepts().unwrap_or(false) {
                        if let Some(movement) = manager.driver().result().best_move {
                            if !legal.contains(&movement)
                                || context
                                    .root_moves
                                    .as_ref()
                                    .is_some_and(|roots| !roots.contains(&movement))
                            {
                                return Err(crate::pals_adaptive::AdaptiveManagerError::Invalid(
                                    "P/C progress escaped original root restriction",
                                ));
                            }
                            if published != Some(movement) {
                                progress(movement);
                                published = Some(movement);
                            }
                        }
                    }
                    if manager.phase() == ManagerPhase::Finished {
                        return Ok(());
                    }
                    if manager.phase() == ManagerPhase::Quarantined
                        || Instant::now() >= context.shutdown_deadline
                    {
                        return Err(crate::pals_adaptive::AdaptiveManagerError::PhysicalUnknown);
                    }
                    if !changed {
                        signal.wait_changed(
                            observed,
                            context
                                .shutdown_deadline
                                .saturating_duration_since(Instant::now())
                                .min(Duration::from_millis(10)),
                        );
                    }
                }
            },
        ));
        if let Some(no_capture) = no_capture_v2.take() {
            *self.retained_no_capture_v2.lock().unwrap_or_else(|p| p.into_inner()) = Some(no_capture);
        }
        if let Some(rejected) = v2_admission_failure {
            *self
                .retained_collection_failure_v2
                .lock()
                .unwrap_or_else(|p| p.into_inner()) = Some(rejected);
        }
        let (run_result, mut original_panic) = match run {
            Ok(result) => (Some(result), None),
            Err(payload) => (None, Some(payload)),
        };
        if v2_hook_started && !v2_hook_revoked {
            if let Some(hook) = &self.collection_hooks_v2 {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    let root = v2_current_admission.as_ref().ok_or(
                        AdaptiveManagerError::Invalid("P/C revoke original root unavailable"),
                    )?;
                    let now = Instant::now();
                    let observation = RoventPcRevokeObservationV2 {
                        origin: RoventPcRevokeOriginV2::RunEnded,
                        elapsed_from_start: context.original_time_budget
                            .and_then(|budget| now.checked_duration_since(budget.start)),
                        context_acceptance: match context.accepts() {
                            Ok(true) => RoventPcContextAcceptanceV2::Accepted,
                            Ok(false) => RoventPcContextAcceptanceV2::Rejected,
                            Err(_) => RoventPcContextAcceptanceV2::Unavailable,
                        },
                        soft_expired: context.soft_expired(now),
                        manager_phase: manager.phase(),
                        goal: manager.driver().port().rovent_frontier_receipt()
                            .map(|receipt| receipt.snapshot.goal),
                    };
                    hook.revoke_root_observed(root, observation)
                }));
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        if let Some(payload) = original_panic.take() {
                            self.retained_collection_panic_v2.lock().unwrap_or_else(|p| p.into_inner()).0 = Some(payload);
                        }
                        *state = State::Manager(manager);
                        self.closed.store(true, Ordering::Release);
                        return Err(physical_unknown_at("RoventPcV2CollectionRevoke", error));
                    }
                    Err(payload) => {
                        let mut retained = self.retained_collection_panic_v2.lock().unwrap_or_else(|p| p.into_inner());
                        retained.0 = original_panic.take();
                        retained.1 = Some(payload);
                        *state = State::Manager(manager);
                        self.closed.store(true, Ordering::Release);
                        return Err(physical_unknown_at(
                            "RoventPcV2CollectionRevokePanic",
                            "original hook and manager retained",
                        ));
                    }
                }
            }
        }
        if let Some(payload) = original_panic {
            if v2_hook_started {
                self.retained_collection_panic_v2
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .0 = Some(payload);
            } else {
                // Keep the legacy no-collection path unchanged.
                drop(payload);
            }
            *state = State::Manager(manager);
            self.closed.store(true, Ordering::Release);
            return Err(physical_unknown_at(
                "RoventPcRootExecution",
                "root closure panicked; original manager retained",
            ));
        }
        if !matches!(&run_result, Some(Ok(()))) {
            *state = State::Manager(manager);
            self.closed.store(true, Ordering::Release);
            return Err(match run_result {
                Some(Err(error)) => {
                    let mut failure = SearchSessionFailure::debug("RoventPcRoot", &error);
                    failure.physical_completion = DriverPhysicalCompletion::Unknown;
                    failure
                }
                _ => unreachable!("original panic retained above"),
            });
        }
        let physical = match manager.shutdown_receipt() {
            Ok(receipt) if receipt.permits_semantic_closure() => receipt,
            Ok(receipt) => {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(physical_unknown_at(
                    "RoventPcShutdownReceipt",
                    format!(
                        "closure rejected: joined={}, shutdown_confirmed={}, pending={}, retained={}, quarantined={}, root_fence={:?}, shared_closure={}",
                        receipt.native_process_joined,
                        receipt.native_physical_shutdown_confirmed,
                        receipt.pending_tickets,
                        receipt.retained_outputs,
                        receipt.quarantined,
                        receipt.root_fence(),
                        receipt.shared_nnuec_closure().is_some()
                    ),
                ));
            }
            Err(error) => {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(physical_unknown_at("RoventPcShutdownReceipt", error));
            }
        };
        let no_capture = self.retained_no_capture_v2.lock().unwrap_or_else(|p| p.into_inner())
            .as_ref().map(|(capability, _)| capability.clone());
        if let Some(capability) = no_capture {
            if !capability.matches_role_only_shutdown_receipt(&physical) {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(physical_unknown_at("RoventPcNoCaptureFence", "same-root native fence unavailable"));
            }
            let Some(hook) = &self.collection_hooks_v2 else {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(physical_unknown_at("RoventPcNoCaptureOwner", "original collection hook unavailable"));
            };
            let signal = manager.activity_signal();
            let cleanup_failure = loop {
                let observed = signal.version();
                match catch_unwind(AssertUnwindSafe(|| hook.finish_deferred(&capability, &physical))) {
                    Ok(Ok(crate::pals_adaptive::AdaptiveCollectionBoundaryV2::Ready)) => break None,
                    Ok(Ok(crate::pals_adaptive::AdaptiveCollectionBoundaryV2::Busy)) if Instant::now() < context.shutdown_deadline => {
                        signal.wait_changed(observed, context.shutdown_deadline.saturating_duration_since(Instant::now()).min(Duration::from_millis(10)));
                    }
                    Ok(Ok(crate::pals_adaptive::AdaptiveCollectionBoundaryV2::Busy)) => {
                        break Some(RoventPcDeferredCleanupFailureV2::Busy);
                    }
                    Ok(Err(error)) => {
                        break Some(RoventPcDeferredCleanupFailureV2::Error(error));
                    }
                    Err(payload) => {
                        break Some(RoventPcDeferredCleanupFailureV2::Panic(payload));
                    }
                }
            };
            if let Some(cleanup_failure) = cleanup_failure {
                let (capability, reason) = self.retained_no_capture_v2.lock().unwrap_or_else(|p| p.into_inner()).take()
                    .expect("original no-capture owner remains with this root");
                let cleanup = Arc::new(Mutex::new(RoventPcDeferredCleanupV2 {
                    capability, reason, fence: physical.clone(), cleanup_failure, retention_failure: None,
                }));
                *self.retained_deferred_cleanup_v2.lock().unwrap_or_else(|p| p.into_inner()) = Some(Arc::clone(&cleanup));
                let transferred = loop {
                    let observed = signal.version();
                    match catch_unwind(AssertUnwindSafe(|| hook.retain_deferred_cleanup(Arc::clone(&cleanup)))) {
                        Ok(Ok(RoventPcCustodyReturnV2::Accepted)) => break true,
                        Ok(Ok(RoventPcCustodyReturnV2::Busy)) if Instant::now() < context.shutdown_deadline => {
                            signal.wait_changed(observed, context.shutdown_deadline.saturating_duration_since(Instant::now()).min(Duration::from_millis(10)));
                        }
                        Ok(Ok(RoventPcCustodyReturnV2::Busy)) => break false,
                        Ok(Err(error)) => {
                            cleanup.lock().unwrap_or_else(|p| p.into_inner()).retention_failure = Some(error);
                            break false;
                        }
                        Err(payload) => {
                            self.retained_collection_panic_v2.lock().unwrap_or_else(|p| p.into_inner()).0 = Some(payload);
                            break false;
                        }
                    }
                };
                if !transferred {
                    *state = State::Manager(manager);
                    self.closed.store(true, Ordering::Release);
                    return Err(failure("RoventPcNoCaptureCustody", "native fence confirmed; original marker cleanup and failure retained"));
                }
                self.retained_deferred_cleanup_v2.lock().unwrap_or_else(|p| p.into_inner()).take();
            } else {
                self.retained_no_capture_v2.lock().unwrap_or_else(|p| p.into_inner()).take();
            }
            self.collection_v2_outstanding.store(false, Ordering::Release);
        }
        if v2_hook_started {
            // Acquire the channel before removing any owner from the manager.
            // A hook panic therefore leaves the actual observer in place.
            let sender = match catch_unwind(AssertUnwindSafe(|| {
                self.collection_hooks_v2
                    .as_ref()
                    .expect("started hook")
                    .handoff_sender_root(v2_current_admission.as_ref().ok_or(
                        AdaptiveManagerError::Invalid("P/C handoff original root unavailable"),
                    )?)
            })) {
                Ok(Ok(sender)) => sender,
                Ok(Err(error)) => {
                    *state = State::Manager(manager);
                    self.closed.store(true, Ordering::Release);
                    return Err(physical_unknown_at("RoventPcV2HandoffSender", error));
                }
                Err(payload) => {
                    self.retained_collection_panic_v2
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .0 = Some(payload);
                    *state = State::Manager(manager);
                    self.closed.store(true, Ordering::Release);
                    return Err(physical_unknown_at(
                        "RoventPcV2HandoffSenderPanic",
                        "original observer and manager retained",
                    ));
                }
            };
            let (admission, observer, settlement) = match manager.take_collection_observer_v2() {
                Ok(Some(original)) => original,
                _ => {
                    *state = State::Manager(manager);
                    self.closed.store(true, Ordering::Release);
                    return Err(physical_unknown_at(
                        "RoventPcV2HandoffUnavailable",
                        "original observer and manager retained",
                    ));
                }
            };
            let handoff = Arc::new(Mutex::new(RoventPcCollectionHandoffV2 {
                admission,
                observer,
                settlement,
                native_fence: physical.clone(),
                teacher_physical_ack: None,
                teacher_aggregate_ack: None,
            }));
            *self
                .retained_collection_handoff_v2
                .lock()
                .unwrap_or_else(|p| p.into_inner()) = Some(Arc::clone(&handoff));
            let same_root = handoff.lock().unwrap_or_else(|p| p.into_inner())
                .admission.same_original_admission(v2_current_admission.as_ref().expect("started root"));
            if !same_root {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(physical_unknown_at("RoventPcV2HandoffRoot", "original handoff and manager retained"));
            }
            if sender.try_send(Arc::clone(&handoff)).is_err() {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(physical_unknown_at(
                    "RoventPcV2HandoffUndelivered",
                    "original handoff and manager retained",
                ));
            }
            // Reclaim only the same joined root worker, without blocking the
            // collection worker on this manager. A scope ACK by itself does
            // not prove worker join, terminal recovery, or publication.
            let signal = manager.activity_signal();
            let hook = self.collection_hooks_v2.as_ref().expect("started hook");
            let mut custody_state = String::from("worker join not yet observed");
            let mut joined = None;
            loop {
                let observed = signal.version();
                let result =
                    catch_unwind(AssertUnwindSafe(|| hook.try_take_root_custody(&handoff)));
                match result {
                    Ok(Ok(Some(custody))) => {
                        let matches = custody.matches_handoff(&handoff);
                        let custody = Arc::new(Mutex::new(custody));
                        *self
                            .retained_collection_custody_v2
                            .lock()
                            .unwrap_or_else(|p| p.into_inner()) = Some(Arc::clone(&custody));
                        if matches {
                            joined = Some(custody);
                        } else {
                            custody_state = "foreign joined worker custody retained; original root unresolved".into();
                        }
                        break;
                    }
                    Ok(Ok(None)) if Instant::now() < context.shutdown_deadline => {
                        signal.wait_changed(
                            observed,
                            context
                                .shutdown_deadline
                                .saturating_duration_since(Instant::now())
                                .min(Duration::from_millis(10)),
                        );
                    }
                    Ok(Ok(None)) => break,
                    Ok(Err(error)) => {
                        custody_state = format!(
                            "original worker custody request failed; owner retained by hook: {error:?}"
                        );
                        break;
                    }
                    Err(payload) => {
                        self.retained_collection_panic_v2
                            .lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .0 = Some(payload);
                        custody_state =
                            "worker custody callback panicked; original owners retained".into();
                        break;
                    }
                }
            }
            let mut returned = false;
            if let Some(custody) = joined {
                loop {
                    let observed = signal.version();
                    let aggregate_ready = match handoff.try_lock() {
                        Ok(original) => original.teacher_aggregate_ack().is_some(),
                        Err(std::sync::TryLockError::WouldBlock) => {
                            if Instant::now() >= context.shutdown_deadline { break; }
                            signal.wait_changed(observed, context.shutdown_deadline
                                .saturating_duration_since(Instant::now()).min(Duration::from_millis(10)));
                            continue;
                        }
                        Err(std::sync::TryLockError::Poisoned(_)) => {
                            custody_state = "original handoff lock poisoned".into();
                            break;
                        }
                    };
                    if !aggregate_ready {
                        custody_state = "joined worker lacks original aggregate physical proof".into();
                        break;
                    }
                    match catch_unwind(AssertUnwindSafe(|| hook.return_root_custody(Arc::clone(&custody)))) {
                        Ok(Ok(RoventPcCustodyReturnV2::Accepted)) => { returned = true; break; }
                        Ok(Ok(RoventPcCustodyReturnV2::Busy)) if Instant::now() < context.shutdown_deadline => {
                            signal.wait_changed(observed, context.shutdown_deadline
                                .saturating_duration_since(Instant::now()).min(Duration::from_millis(10)));
                        }
                        Ok(Ok(RoventPcCustodyReturnV2::Busy)) => {
                            custody_state = "custody receiver remained busy until the original deadline".into();
                            break;
                        }
                        Ok(Err(error)) => { custody_state = format!("custody receiver failed: {error:?}"); break; }
                        Err(payload) => {
                            self.retained_collection_panic_v2.lock().unwrap_or_else(|p| p.into_inner()).0 = Some(payload);
                            custody_state = "custody receiver panicked; original owner retained".into();
                            break;
                        }
                    }
                }
            }
            if !returned {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(physical_unknown_at("RoventPcV2CollectionCustodyPending", custody_state));
            }
            // The collection owner retains the exact joined worker and handoff.
            // It separately records publication and settlement outcomes.
            self.retained_collection_custody_v2.lock().unwrap_or_else(|p| p.into_inner()).take();
            self.retained_collection_handoff_v2.lock().unwrap_or_else(|p| p.into_inner()).take();
            self.collection_v2_outstanding.store(false, Ordering::Release);
        }
        if let Some(observer) = &self.collection_hooks.root_closure {
            if let Err(error) = observer(&mut manager) {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::debug(
                    "RoventPcCollectionClosure",
                    &error,
                ));
            }
        }
        // Finished proves the original physical closure, not successful
        // semantic consumption. Normal stop/expiry and validated canceled
        // native ACKs already have separate manager control paths; never
        // reinterpret a recorded failure by matching its backend text.
        let semantic_failure = manager.failures().next().map(|error| {
            let mut failure = SearchSessionFailure::debug("RoventPcManagerFailure", error);
            failure.detail.push_str(&format!(
                "; recorded_failures={}, actual_frontier_goal={:?}",
                manager.failures().count(),
                manager
                    .driver()
                    .port()
                    .rovent_frontier_receipt()
                    .map(|receipt| receipt.snapshot.goal),
            ));
            failure
        });
        if let Some(failure) = semantic_failure {
            // The same manager still owns every original failure, port,
            // receipt and native source through finish's actual join path.
            // Its verified closure above permits Confirmed failure reporting.
            *state = State::Manager(manager);
            self.closed.store(true, Ordering::Release);
            return Err(failure);
        }
        let result = manager.driver().result();
        let next = match physical.request_sequence_high_water.checked_add(1) {
            Some(next) => next,
            None => {
                *state = State::Manager(manager);
                self.closed.store(true, Ordering::Release);
                return Err(failure(
                    "RoventPcSequence",
                    "actual role request sequence exhausted",
                ));
            }
        };
        // Verify the manager and the original driver/actor receipt while the
        // same Frontier still owns its original budget and Rules identity.
        // Retiring it first would destroy the basis for shared cache closure.
        let sources = match manager.into_sources() {
            Ok(sources) => sources,
            Err((retained, error)) => {
                *state = State::Manager(Box::new(retained));
                self.closed.store(true, Ordering::Release);
                return Err(physical_unknown_at("RoventPcIntoSources", error));
            }
        };
        let native = NativeOwner::new(sources.native).retain_root_closure(physical.clone());
        let mut port = match sources.driver.into_port(&physical) {
            Ok(port) => port,
            Err((port, error)) => {
                let detail = format!(
                    "error={error:?}, authority_matches={}, mode_matches={}, actual_root_budget={:?}, receipt_root_budget={:?}, shared_required={}, shared_closure_matches={}",
                    port.port().authority() == physical.authority,
                    port.port().mode() == physical.mode,
                    port.port().original_root_budget(),
                    physical.root_budget,
                    port.port().requires_shared_nnuec_readout(),
                    physical
                        .shared_nnuec_closure()
                        .is_some_and(|closure| port.port().shared_nnuec_closure_matches(closure)),
                );
                *state = State::RejectedPort { native, port };
                self.closed.store(true, Ordering::Release);
                return Err(physical_unknown_at("RoventPcIntoPort", detail));
            }
        };
        let repair = match port.take_repair_report() {
            Ok(report) => report,
            Err(error) => {
                *state = State::RejectedPort {
                    native,
                    port: PcPortDriver::new(port),
                };
                self.closed.store(true, Ordering::Release);
                return Err(SearchSessionFailure::debug("RoventPcRepairReport", &error));
            }
        };
        *self
            .retained_repair
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = repair;
        let frontier = match port.retire_rovent_frontier() {
            Ok(Some(receipt)) => receipt.snapshot,
            Ok(None) => {
                *state = State::RejectedPort {
                    native,
                    port: PcPortDriver::new(port),
                };
                self.closed.store(true, Ordering::Release);
                return Err(physical_unknown_at(
                    "RoventPcRetireFrontier",
                    "original Frontier is absent",
                ));
            }
            Err(error) => {
                *state = State::RejectedPort {
                    native,
                    port: PcPortDriver::new(port),
                };
                self.closed.store(true, Ordering::Release);
                return Err(physical_unknown_at("RoventPcRetireFrontier", error));
            }
        };
        let repair = self
            .retained_repair
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        let custody = match root_admission {
            PcRootAdmission::V1(root_admission) => {
                let receipt = Arc::new(RoventPcRootReceipt {
                    domain: ROVENT_PC_ROOT_DOMAIN,
                    authority: context.authority,
                    input_lock: self.snapshot.input_lock,
                    root_admission,
                    frontier,
                    physical: physical.clone(),
                    repair,
                });
                match self.latest.lock() {
                    Ok(mut latest) => *latest = Some(Arc::clone(&receipt)),
                    Err(_) => {
                        std::mem::forget(receipt);
                        *state = State::RejectedPort {
                            native,
                            port: PcPortDriver::new(port),
                        };
                        self.closed.store(true, Ordering::Release);
                        return Err(physical_unknown_at(
                            "RoventPcPublishReceipt",
                            "root receipt owner is poisoned",
                        ));
                    }
                }
                IdleCustody::RetiredRoot(receipt)
            }
            PcRootAdmission::V2(root_admission) => {
                let receipt = Arc::new(RoventPcRootReceiptV2 {
                    domain: "rz-rovent-pc-native-root/2",
                    authority: context.authority,
                    input_lock: self.snapshot.input_lock,
                    root_admission,
                    frontier,
                    physical: physical.clone(),
                    repair,
                });
                match self.latest_v2.lock() {
                    Ok(mut latest) => *latest = Some(Arc::clone(&receipt)),
                    Err(_) => {
                        std::mem::forget(receipt);
                        *state = State::RejectedPort {
                            native,
                            port: PcPortDriver::new(port),
                        };
                        self.closed.store(true, Ordering::Release);
                        return Err(physical_unknown_at(
                            "RoventPcPublishReceiptV2",
                            "root receipt owner is poisoned",
                        ));
                    }
                }
                IdleCustody::RetiredRootV2(receipt)
            }
        };
        // Publish the physical receipt before any fallible work extraction or
        // strategy clone, as in the original V1 lifecycle.
        let (frontier, physical) = match &custody {
            IdleCustody::RetiredRoot(receipt) => (&receipt.frontier, &receipt.physical),
            IdleCustody::RetiredRootV2(receipt) => (&receipt.frontier, &receipt.physical),
            IdleCustody::StartupReady => unreachable!("root receipt was just published"),
        };
        let work = match pc_root_work(frontier, physical) {
            Ok(work) => work,
            Err(error) => {
                *state = State::RejectedPort {
                    native,
                    port: PcPortDriver::new(port),
                };
                self.closed.store(true, Ordering::Release);
                return Err(error);
            }
        };
        let mate_strategy = frontier
            .mate_strategy
            .clone()
            .filter(|strategy| result.best_move == Some(strategy.best_move()));
        let owner = port.into_owner();
        *state = State::Idle(Box::new(Idle {
            native,
            owner: Some(Box::new(owner)),
            next_request_sequence: next,
            game: Some(context.authority.game.0),
            custody,
        }));
        if result.best_move.is_some_and(|movement| {
            !legal.contains(&movement)
                || context
                    .root_moves
                    .as_ref()
                    .is_some_and(|moves| !moves.contains(&movement))
        }) {
            self.closed.store(true, Ordering::Release);
            return Err(failure(
                "RoventPcRootRestriction",
                "actual result escaped original root restriction",
            ));
        }
        Ok(SearchSessionReport {
            best_move: result.best_move,
            work,
            mate_strategy,
        })
    }
    fn finish(&self, until: Instant) -> Result<(), SearchSessionFailure> {
        self.closed.store(true, Ordering::Release);
        let mut state = self
            .state
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        let mut shared_actor = self
            .shared_nnuec
            .as_ref()
            .map(|owner| owner.actor.try_lock())
            .transpose()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        let mut storage = self
            .storage
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        let mut retained = std::mem::replace(&mut *state, State::Moving);
        loop {
            match retained {
                State::Closed => {
                    *state = State::Closed;
                    return Ok(());
                }
                State::Idle(mut idle) => {
                    if let Some(actor) = shared_actor.as_mut() {
                        if let Err(error) = actor.finish(until) {
                            *state = State::Idle(idle);
                            return Err(error);
                        }
                    }
                    let result = idle.native.finish(until);
                    if storage.is_some() && result.is_err() {
                        *state = State::Idle(idle);
                        return result.map_err(storage_unknown_failure);
                    }
                    if idle.native.joined && idle.native.finished == Some(true) {
                        if let Some(storage) = storage.as_mut() {
                            let signal = idle.native.service.signal();
                            match storage.finish(
                                idle.owner.take().map(StorageSemanticOwner::Pc),
                                until,
                                &signal,
                            ) {
                                Ok(()) => {}
                                Err(mut rejected) => {
                                    if let Some(owner) = rejected.owner.take() {
                                        match owner {
                                            StorageSemanticOwner::Pc(owner) => {
                                                idle.owner = Some(owner)
                                            }
                                            owner => storage.retain_wrong_return(owner),
                                        }
                                    }
                                    *state = State::Idle(idle);
                                    return Err(rejected.failure);
                                }
                            }
                        }
                        *state = State::Closed;
                    } else {
                        *state = State::Idle(idle);
                    }
                    return result;
                }
                State::RejectedOwner { mut native, owner } => {
                    if let Some(actor) = shared_actor.as_mut() {
                        if let Err(error) = actor.finish(until) {
                            *state = State::RejectedOwner { native, owner };
                            return Err(error);
                        }
                    }
                    let result = native.finish(until);
                    if storage.is_some() && result.is_err() {
                        *state = State::RejectedOwner { native, owner };
                        return result.map_err(storage_unknown_failure);
                    }
                    if native.joined && native.finished == Some(true) {
                        if let Some(storage) = storage.as_mut() {
                            let signal = native.service.signal();
                            if let Err(mut rejected) = storage.finish(
                                Some(StorageSemanticOwner::Pc(owner)),
                                until,
                                &signal,
                            ) {
                                *state = match rejected.owner.take() {
                                    Some(StorageSemanticOwner::Pc(owner)) => {
                                        State::RejectedOwner { native, owner }
                                    }
                                    other => {
                                        if let Some(owner) = other {
                                            storage.retain_wrong_return(owner);
                                        }
                                        State::ArchiveRetained { native }
                                    }
                                };
                                return Err(rejected.failure);
                            }
                        }
                        *state = State::Closed;
                    } else {
                        *state = State::RejectedOwner { native, owner };
                    }
                    return result;
                }
                State::RejectedPort { mut native, port } => {
                    if let Some(actor) = shared_actor.as_mut() {
                        if let Err(error) = actor.finish(until) {
                            *state = State::RejectedPort { native, port };
                            return Err(error);
                        }
                    }
                    let result = native.finish(until);
                    if storage.is_some() {
                        if result.is_err() || !native.joined || native.finished != Some(true) {
                            *state = State::RejectedPort { native, port };
                            return result
                                .map_err(storage_unknown_failure)
                                .and(Err(SearchSessionFailure::physical_completion_unknown()));
                        }
                        let Some(receipt) = native._root_closure.as_ref() else {
                            *state = State::RejectedPort { native, port };
                            return Err(physical_unknown_at(
                                "RoventPcArchivePort",
                                "original semantic closure receipt absent",
                            ));
                        };
                        match port.into_port(receipt) {
                            Ok(port) => {
                                retained = State::RejectedOwner {
                                    native,
                                    owner: Box::new(port.into_owner()),
                                };
                                continue;
                            }
                            Err((port, error)) => {
                                *state = State::RejectedPort { native, port };
                                return Err(storage
                                    .as_mut()
                                    .expect("selected storage")
                                    .retain_port_error(error));
                            }
                        }
                    }
                    if native.joined && native.finished == Some(true) {
                        *state = State::Closed;
                    } else {
                        *state = State::RejectedPort { native, port };
                    }
                    return result;
                }
                State::ArchiveRetained { mut native } => {
                    if let Some(actor) = shared_actor.as_mut() {
                        if let Err(error) = actor.finish(until) {
                            *state = State::ArchiveRetained { native };
                            return Err(error);
                        }
                    }
                    if let Err(error) = native.finish(until) {
                        *state = State::ArchiveRetained { native };
                        return Err(storage_unknown_failure(error));
                    }
                    let outcome = match storage.as_mut() {
                        Some(storage) => {
                            match storage.finish(None, until, &native.service.signal()) {
                                Ok(()) => Ok(()),
                                Err(mut rejected) => {
                                    if let Some(owner) = rejected.owner.take() {
                                        storage.retain_wrong_return(owner);
                                    }
                                    Err(rejected.failure)
                                }
                            }
                        }
                        None => Err(SearchSessionFailure::physical_completion_unknown()),
                    };
                    *state = State::ArchiveRetained { native };
                    return outcome.and(Err(SearchSessionFailure::physical_completion_unknown()));
                }
                State::Manager(mut manager) => {
                    let actor = shared_actor.as_ref().and_then(|owner| owner.actor.as_ref());
                    if manager.phase() != ManagerPhase::Finished
                        && Self::drain_manager(&mut manager, actor, until).is_err()
                    {
                        *state = State::Manager(manager);
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                    let receipt = match manager.shutdown_receipt() {
                        Ok(value) if value.permits_semantic_closure() => value,
                        _ => {
                            *state = State::Manager(manager);
                            return Err(SearchSessionFailure::physical_completion_unknown());
                        }
                    };
                    if self.collection_v2_outstanding.load(Ordering::Acquire) {
                        *state = State::Manager(manager);
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                    let sources = match manager.into_sources() {
                        Ok(value) => value,
                        Err((manager, _)) => {
                            *state = State::Manager(Box::new(manager));
                            return Err(SearchSessionFailure::physical_completion_unknown());
                        }
                    };
                    let mut native =
                        NativeOwner::new(sources.native).retain_root_closure(receipt.clone());
                    native.finished = receipt.native_physical_shutdown_confirmed.then_some(true);
                    native.joined = receipt.native_process_joined;
                    // Process shutdown does not reuse the semantic owner. Keep its
                    // actual port and every view lease until the native owner joins.
                    retained = State::RejectedPort {
                        native,
                        port: sources.driver,
                    };
                }
                State::Moving => {
                    *state = State::Moving;
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
            }
        }
    }
    fn drain_manager(
        manager: &mut Manager,
        actor: Option<&PreparationActor>,
        until: Instant,
    ) -> Result<(), AdaptiveManagerError> {
        manager.begin_shutdown(until)?;
        if let Some(actor) = actor {
            loop {
                let signal = manager.activity_signal();
                let observed = signal.version();
                manager.pump_shared_nnuec_preparation(actor)?;
                let changed = manager.tick()?;
                manager.pump_shared_nnuec_preparation(actor)?;
                match manager.phase() {
                    ManagerPhase::Finished => return Ok(()),
                    ManagerPhase::Quarantined => return Err(AdaptiveManagerError::PhysicalUnknown),
                    _ if Instant::now() >= until => {
                        return Err(AdaptiveManagerError::PhysicalUnknown);
                    }
                    _ => {}
                }
                if !changed {
                    signal.wait_changed(
                        observed,
                        until
                            .saturating_duration_since(Instant::now())
                            .min(Duration::from_millis(10)),
                    );
                }
            }
        } else {
            manager
                .run_root_until(until, &AtomicBool::new(false))
                .and_then(|receipt| {
                    receipt
                        .permits_semantic_closure()
                        .then_some(())
                        .ok_or(AdaptiveManagerError::PhysicalUnknown)
                })
        }
    }
}
impl SearchSessionDriver for RoventPcSessionDriver {
    fn supports_ponder_clock(&self) -> bool {
        true
    }
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        crate::bridge::NodeBudgetSemantics::RoventPositionVisits
    }
    fn validate_go(
        &self,
        position: &Position,
        go: &crate::parser::GoLimits,
    ) -> Result<(), ContractError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(ContractError::new(
                rz_contracts::ErrorCode::BackendFailure,
                rz_contracts::Stage::Admission,
                "Rovent P/C physical owner is closed or quarantined",
            ));
        }
        let max = self.search_goal_capability().ok_or_else(poisoned)?;
        match &self.sources {
            PcSources::V1(sources) => crate::rovent_root_admission::preadmission::validate_go(
                position,
                go,
                sources.lock(),
                self.root_policy,
                max,
            ),
            PcSources::V2(sources) => crate::rovent_root_admission::preadmission::validate_go_v2(
                position,
                go,
                sources.lock(),
                self.root_policy,
                max,
            ),
        }
        .map_err(|error| {
            ContractError::new(
                rz_contracts::ErrorCode::InvalidInput,
                rz_contracts::Stage::Admission,
                error.detail,
            )
        })?;
        Ok(())
    }
    fn kind(&self) -> SearchKind {
        SearchKind::Rovent
    }
    fn identity(&self) -> EngineIdentity {
        self.identity.clone()
    }
    fn implementation(&self) -> Digest {
        self.implementation
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        NonZeroU16::new(self.configuration.fields().frontier.maximum_goal_plies)
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(ContractError::new(
                rz_contracts::ErrorCode::BackendFailure,
                rz_contracts::Stage::Output,
                "Rovent P/C physical owner is closed or quarantined",
            ));
        }
        self.reset_requested.store(true, Ordering::Release);
        Ok(())
    }
    fn finish_session(
        &self,
        until: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        self.finish(until)?;
        Ok(Some(DriverPhysicalCompletion::Confirmed))
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        self.run_root(position, context, progress)
    }
}
impl Drop for RoventPcSessionDriver {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|poison| poison.into_inner());
        if !matches!(state, State::Closed) {
            let retained = std::mem::replace(state, State::Moving);
            std::mem::forget(retained);
            // Preserve the original descriptors with the quarantined physical
            // owner instead of releasing startup provenance on an unknown drop.
            self.sources.forget_original();
            if let Some(process) = self.process_v2.take() {
                std::mem::forget(process);
            }
            if let Some(hooks) = self.collection_hooks_v2.take() {
                std::mem::forget(hooks);
            }
            if let Some(handoff) = self
                .retained_collection_handoff_v2
                .get_mut()
                .unwrap_or_else(|p| p.into_inner())
                .take()
            {
                std::mem::forget(handoff);
            }
            if let Some(custody) = self
                .retained_collection_custody_v2
                .get_mut()
                .unwrap_or_else(|p| p.into_inner())
                .take()
            {
                std::mem::forget(custody);
            }
            if let Some(capability) = self.retained_no_capture_v2.get_mut().unwrap_or_else(|p| p.into_inner()).take() {
                std::mem::forget(capability);
            }
            if let Some(cleanup) = self.retained_deferred_cleanup_v2.get_mut().unwrap_or_else(|p| p.into_inner()).take() {
                std::mem::forget(cleanup);
            }
            if let Some(failure) = self
                .retained_collection_failure_v2
                .get_mut()
                .unwrap_or_else(|p| p.into_inner())
                .take()
            {
                std::mem::forget(failure);
            }
            let panics = self
                .retained_collection_panic_v2
                .get_mut()
                .unwrap_or_else(|p| p.into_inner());
            if let Some(payload) = panics.0.take() {
                std::mem::forget(payload);
            }
            if let Some(payload) = panics.1.take() {
                std::mem::forget(payload);
            }
            if let Some(owner) = self.shared_nnuec.take() {
                // Do not run PreparationActor's joining Drop on an unknown
                // manager/root. Its original cache/terminal/model owners stay
                // quarantined with the physical source instead.
                std::mem::forget(owner);
            }
            let storage = self
                .storage
                .get_mut()
                .unwrap_or_else(|poison| poison.into_inner());
            if let Some(storage) = storage.take() {
                std::mem::forget(storage);
            }
        }
    }
}
/// The same production ordering is used by every P/C session configuration.
/// No role input/manager exists yet. Recheck binds the already attached
/// original per-go budget, never an invented startup budget or silent default.
fn configure_root_port(
    port: &mut PcSemanticPort<Model>,
    frontier: FrontierRegistration,
    limits: FrontierLimits,
    actual: PcFrontierStartup,
    recheck: PostRepairRecheckPolicy,
) -> Result<(), SearchSessionFailure> {
    port.attach_rovent_frontier(frontier, limits, actual)
        .map_err(|error| SearchSessionFailure::debug("RoventPcFrontier", &error))?;
    port.enable_post_repair_recheck(recheck)
        .map_err(|error| SearchSessionFailure::debug("RoventPcRecheckPolicy", &error))?;
    if port.post_repair_recheck() != recheck {
        return Err(failure(
            "RoventPcRecheckPolicy",
            "actual port policy differs from its registered configuration",
        ));
    }
    Ok(())
}
/// Keep the original physical-unknown code and ownership classification while
/// preserving the actual failed boundary for post-run diagnosis.
fn physical_unknown_at(stage: &'static str, detail: impl std::fmt::Debug) -> SearchSessionFailure {
    let mut failure = SearchSessionFailure::physical_completion_unknown();
    failure.detail = format!("{stage}: {detail:?}");
    failure
}
fn storage_unknown_failure(mut error: SearchSessionFailure) -> SearchSessionFailure {
    error.physical_completion = DriverPhysicalCompletion::Unknown;
    error
}
fn failure(code: &'static str, detail: &str) -> SearchSessionFailure {
    SearchSessionFailure {
        physical_completion: DriverPhysicalCompletion::Confirmed,
        code,
        detail: detail.into(),
    }
}
/// Declaration gate only. Existing resource admission remains the owner of
/// quota, topology and observed OS affinity; this grants none of those facts.
fn declares_shared_nnuec_helper(pools: &[rz_experiments::RoventThreadPool]) -> bool {
    pools.iter().any(|pool| {
        // This actor owns a distinct physical thread; it cannot claim to
        // share the manager/caller's single physical pool.
        pool.actors.len() == 1
            && pool
                .actors
                .contains(&rz_experiments::RoventActor::RuntimeHelper)
            && pool.max_threads >= 1
    })
}
fn poisoned() -> ContractError {
    ContractError::new(
        rz_contracts::ErrorCode::BackendFailure,
        rz_contracts::Stage::Output,
        "Rovent P/C observation mutex poisoned",
    )
}

#[cfg(test)]
mod shared_nnuec_resource_tests {
    use super::declares_shared_nnuec_helper;
    use rz_experiments::{RoventActor, RoventThreadPool};
    use std::collections::BTreeSet;

    #[test]
    fn shared_readout_rejects_absent_or_zero_thread_helper_declarations() {
        assert!(!declares_shared_nnuec_helper(&[]));
        assert!(!declares_shared_nnuec_helper(&[RoventThreadPool {
            actors: BTreeSet::from([RoventActor::PcCaller]),
            max_threads: 1,
            cpu_charge_millicpus: 1_000,
        }]));
        assert!(!declares_shared_nnuec_helper(&[RoventThreadPool {
            actors: BTreeSet::from([RoventActor::RuntimeHelper]),
            max_threads: 0,
            cpu_charge_millicpus: 0,
        }]));
    }

    #[test]
    fn shared_readout_rejects_a_helper_disguised_as_the_manager_pool() {
        assert!(!declares_shared_nnuec_helper(&[RoventThreadPool {
            actors: BTreeSet::from([RoventActor::Manager, RoventActor::RuntimeHelper]),
            max_threads: 1,
            cpu_charge_millicpus: 1_000,
        }]));
    }

    #[test]
    fn shared_readout_accepts_one_declared_helper_in_original_pool() {
        assert!(declares_shared_nnuec_helper(&[RoventThreadPool {
            actors: BTreeSet::from([RoventActor::RuntimeHelper]),
            max_threads: 1,
            cpu_charge_millicpus: 1_000,
        }]));
    }
}

#[cfg(test)]
mod root_policy_tests {
    use super::*;
    use rz_contracts::rovent::{ExecutionMode, SearchGoal};
    use rz_contracts::{
        ClockDomain, Deadline, GameGeneration, MonotonicTick, OwnerId, PrecisionProfile,
        ProcessEpoch, RootGeneration,
    };
    use rz_search::contract_time::InstantClock;
    use rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace;
    use rz_search::pals::engine::{
        LegalOrderRoleMock, PalsSemanticConfig, PcSemanticProfile, RoleModel,
    };
    use rz_search::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
    use std::sync::atomic::AtomicU64;

    #[test]
    fn iterative_pc_policy_binds_original_frontier_before_first_role_input() {
        let value = LegalOrderRoleMock.value_identity().unwrap().clone();
        let namespace = AdaptiveModelNamespace::new(
            value.clone(),
            Digest([5; 32]),
            Digest([6; 32]),
            PrecisionProfile::Fp32,
            1,
        )
        .unwrap();
        let model = AdaptivePortModel::new_with_namespace(
            value.model.clone(),
            Some(value),
            namespace.clone(),
        )
        .unwrap();
        let mut owner = PcSemanticOwner::new_selected_rovent(
            PcSemanticProfile {
                semantic: PalsSemanticConfig {
                    beam_width: 1,
                    line_plies: 6,
                    max_nodes: 257,
                    max_records: 128,
                    max_role_calls: 64,
                },
                maximum_root_prefix_plies: 16,
            },
            model,
            ReviewSelectionPolicy::default(),
            ReviewIndexLimits::default(),
        )
        .unwrap();
        let position = Position::startpos();
        let authority = SearchAuthority {
            epoch: ProcessEpoch(23),
            game: GameGeneration(0),
            root: RootGeneration(7),
            implementation: Digest([7; 32]),
        };
        let clock = InstantClock::new(ClockDomain(authority.epoch), Instant::now());
        let deadline = Deadline {
            clock: clock.domain(),
            at: MonotonicTick(20_000_000_000),
        };
        let registration = owner.register_adaptive_root(&position, authority).unwrap();
        let next = Arc::new(AtomicU64::new(1));
        let issuer: AdaptiveRulesOwnerIssuer =
            Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::Relaxed))));
        let mut port = PcSemanticPort::new_registered_retaining(
            owner,
            position,
            authority,
            deadline,
            clock.instant_at(deadline.at).unwrap(),
            ExecutionMode::CpuPc,
            issuer,
            registration,
            None,
        )
        .map_err(|(_, error)| error)
        .unwrap();
        let policy = PostRepairRecheckPolicy::IterativeFrozenModelWdlV2;
        assert_eq!(port.original_root_budget(), None);
        assert!(port.enable_post_repair_recheck(policy).is_err());
        assert_eq!(
            port.post_repair_recheck(),
            PostRepairRecheckPolicy::Disabled
        );
        let budget = Digest([11; 32]);
        let frontier = FrontierRegistration {
            authority,
            mode: ExecutionMode::CpuPc,
            original_deadline: deadline,
            root_budget: budget,
            goal: SearchGoal {
                depth_plies: NonZeroU16::new(1),
                mate_moves: None,
            },
            nodes: Some(2_000),
            selective_policy: Digest([12; 32]),
            root_rules: port.root_rules(),
            root_moves: Arc::clone(port.root_move_domain()),
            root_state: port.root_state_id(),
            model_namespace: Arc::new(namespace),
            native_ledger: None,
        };
        configure_root_port(
            &mut port,
            frontier,
            FrontierLimits {
                maximum_goal_plies: NonZeroU16::new(16).unwrap(),
                maximum_frames: 64,
                maximum_paths: 128,
                maximum_mate_nodes: 256,
                maximum_native_receipts: 1,
            },
            PcFrontierStartup {
                root_budget: budget,
                clock,
            },
            policy,
        )
        .unwrap();
        assert_eq!(port.original_root_budget(), Some(budget));
        assert_eq!(port.post_repair_recheck(), policy);
        assert!(port.enable_post_repair_recheck(policy).is_err());
    }
}
