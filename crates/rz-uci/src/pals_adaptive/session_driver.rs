//! Process-owned adaptive UCI session. A root fence returns the same native
//! session, CPU engine/TT and Search Store; only process finish releases them.
//! Failed ownership transfers are retained in the session, never mapped to an
//! empty successful search or replaced by a synchronous/mock backend.

mod checker_metadata;
pub use checker_metadata::*;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod collection_v2;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use collection_v2::*;
mod completion_observation;
mod idle_physical_close;
#[cfg(feature = "pals-adaptive")]
mod initial_stop;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod profiled_retirement;
mod root_admission;
mod rovent;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod rovent_external;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod storage;
use completion_observation::CompletionJournal;
pub use completion_observation::{
    AdaptiveCompletionDiagnostic, AdaptiveCompletionFailure, AdaptiveCompletionObservation,
    AdaptiveCompletionReasonCounts, AdaptiveCompletionReturn,
};
#[cfg(feature = "pals-adaptive")]
use initial_stop::RoventPcrIdleCustody;
#[cfg(feature = "pals-adaptive")]
pub use initial_stop::{
    ROVENT_PCR_INITIAL_DOMAIN, ROVENT_PCR_INITIAL_DOMAIN_V2, RoventPcrInitialReceipt,
    RoventPcrInitialReceiptV2,
};
use root_admission::{RoventPcrRootAdmission, RoventPcrRootSource};
#[cfg(feature = "pals-adaptive")]
use root_admission::{RoventPcrRootPolicy, RoventPcrRootPolicyV2};
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use rovent::ROVENT_PROFILED_EXTERNAL_ROOT_DOMAIN;
pub use rovent::{ROVENT_NATIVE_ROOT_DOMAIN, RoventNativeRootReceipt, RoventPcrSessionProfile};
#[cfg(feature = "pals-adaptive")]
pub use rovent::{ROVENT_NATIVE_ROOT_V2_DOMAIN, RoventNativeRootReceiptV2};
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use rovent_external::{
    RoventExternalSessionAdmissionRejection, RoventExternalSessionAdmissionRejectionV2,
    RoventExternalSessionSource, RoventExternalSessionSources, RoventExternalSessionSourcesV2,
    RoventExternalSourcePreparationRejection,
};

use super::AdaptiveSearchDriver as _;
use super::{
    AdaptiveCollectionDispatchObserver, AdaptiveExecutionLimits, AdaptiveExecutionManager,
    AdaptiveExternalTeacherObserver, AdaptiveManagerAdmissionRejection, AdaptiveManagerError,
    AdaptiveManagerRegistration, AdaptiveManagerSources, AdaptivePortDriver,
    AdaptiveSequenceContinuation, AdaptiveShutdownReceipt, ExternalManagerWorker, ManagerPhase,
    PersistentCpuOwner,
};
use crate::EngineIdentity;
use crate::engine::ProcessClock;
use crate::pals_native::ticketed::{TicketNativeBackend, TicketNativeEvent, TicketedNativeService};
use crate::search_driver::work::{AttemptObservation, ProcessWorkJournal};
use crate::search_driver::{
    DriverWork, PalsCheckerRegistration, PalsResolverIdentity, PalsSearchPolicyIdentity,
    ProcessSearchWorkReceipt, SearchKind, SearchSessionContext, SearchSessionDriver,
    SearchSessionFailure, SearchSessionReport,
};
use rz_contracts::pals::SearchAuthority;
use rz_contracts::{ContractError, Digest, ErrorCode, ProcessEpoch, Stage};
use rz_position::{BoardMove, Position};
use rz_runtime::adaptive::{ControllerConfig, CoordinatorPolicy};
use rz_runtime::contracts::ContractClock as _;
use rz_search::cpu_checker::{CheckerIdentity, CheckerShutdown, ExternalUciIdentity};
use rz_search::cpu_worker::CpuWorker;
use rz_search::external_cpu::{CpuEndpointSelection, ExternalSlicePolicy};
use rz_search::pals::coordinator::ProposerRankingOutput;
use rz_search::pals::engine::adaptive_port::archive_worker::{
    ArchiveOperation, ArchiveOwner, ArchiveQuarantine, ArchiveTransfer, ArchiveWork,
    ArchiveWorkReceipt, ArchiveWorker,
};
use rz_search::pals::engine::adaptive_port::{
    AdaptiveEnginePort, AdaptivePortModel, AdaptivePortQuestion, AdaptiveRulesOwnerIssuer,
};
use rz_search::pals::engine::{PalsCompletion, PalsCounters, PalsEngine, PalsLimits};
use rz_search::pals::review_context::ReviewView;
use sha2::{Digest as _, Sha256};
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

pub type AdaptiveSessionManager<B> =
    AdaptiveExecutionManager<AdaptivePortQuestion, B, AdaptivePortDriver<AdaptivePortModel>>;
type SessionManager<B> = AdaptiveSessionManager<B>;
type AdmissionRejection<B> = AdaptiveManagerAdmissionRejection<
    AdaptivePortQuestion,
    B,
    AdaptivePortDriver<AdaptivePortModel>,
>;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
type ProfiledAdmissionRejection<B> = super::ProfiledExternalManagerAdmissionRejection<
    AdaptivePortQuestion,
    B,
    AdaptivePortDriver<AdaptivePortModel>,
>;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
type ProfiledAdmissionRejectionV2<B> = super::ProfiledExternalManagerAdmissionRejection<
    AdaptivePortQuestion,
    B,
    AdaptivePortDriver<AdaptivePortModel>,
    rz_experiments::RoventInputLockV2,
>;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
type SessionExternalRootSource<'a> = Option<SessionExternalJoined<'a>>;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
enum SessionExternalJoined<'a> {
    V1(&'a crate::rovent_external_launch::RoventAdmittedExternalRootSource),
    V2(&'a crate::rovent_external_launch::RoventAdmittedExternalRootSourceV2),
}
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
enum OwnedSessionExternalJoined {
    V1(crate::rovent_external_launch::RoventAdmittedExternalRootSource),
    V2(crate::rovent_external_launch::RoventAdmittedExternalRootSourceV2),
}
#[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
type SessionExternalRootSource<'a> =
    Option<&'a rz_search::external_cpu::ProfiledExternalRootSource>;

/// Runs on the actual UCI search owner after a real root fence and before the
/// manager is dismantled. Collection can seal actual turns/action outcomes.
pub type AdaptiveRootClosureObserver<B> =
    Arc<dyn Fn(&mut SessionManager<B>) -> Result<(), AdaptiveManagerError> + Send + Sync>;

/// Collection setup executes on the real exclusive Search owner, before root
/// Rules preparation or native ranking admission.
pub type AdaptiveRootAdmissionObserver<B> =
    Arc<dyn Fn(&mut SessionManager<B>) -> Result<(), AdaptiveManagerError> + Send + Sync>;
/// The completion is taken from the actual semantic result after the root fence,
/// rather than inferred by a collection caller from elapsed time or counters.
pub type AdaptiveCompletedRootObserver<B> = Arc<
    dyn Fn(&mut SessionManager<B>, PalsCompletion) -> Result<(), AdaptiveManagerError>
        + Send
        + Sync,
>;
/// Receives the original root-local utility action after actual physical closure.
pub type AdaptiveUtilityRouteCompletionObserver = Arc<
    dyn Fn(&Arc<super::CompletedUtilityRouteAction>) -> Result<(), AdaptiveManagerError>
        + Send
        + Sync,
>;
/// Receives the original cohort trial after same-owner physical retirement.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub type AdaptiveUtilityTrialRetirementObserver = Arc<
    dyn Fn(&Arc<super::UtilityTrialPhysicalRetirement>) -> Result<(), AdaptiveManagerError>
        + Send
        + Sync,
>;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub type AdaptiveUtilityInterventionTrialRetirementObserver = Arc<
    dyn Fn(
            &Arc<super::UtilityInterventionTrialPhysicalRetirement>,
        ) -> Result<(), AdaptiveManagerError>
        + Send
        + Sync,
>;

// Declare after taking the session state lock: failure closes admission before
// that lock is released, including unwinding and an already waiting next go.
struct InitialGameFailureGuard<'a> {
    closed: &'a AtomicBool,
    armed: bool,
}
impl Drop for InitialGameFailureGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.closed.store(true, Ordering::Release);
        }
    }
}

pub struct AdaptiveSessionCollectionHooks<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    pub root_admission: Option<AdaptiveRootAdmissionObserver<B>>,
    pub root_closure: Option<AdaptiveCompletedRootObserver<B>>,
    /// Historical N05 output from the same original ticket/input lineage.
    /// Only the Session's actual same-actor engine/cache ACK boundary issues
    /// this cap; an earlier root_closure callback cannot mint whole cost.
    /// The Session retains the original Arc on callback error or unwinding.
    pub utility_route_completed: Option<AdaptiveUtilityRouteCompletionObserver>,
    /// Cohort physical retirement is issued only from the original trial
    /// admission and the same actually retired researcher owner. It cannot be
    /// supplied by relabeling a historical root-local Completed action.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub utility_trial_retired: Option<AdaptiveUtilityTrialRetirementObserver>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub utility_intervention_trial_retired:
        Option<AdaptiveUtilityInterventionTrialRetirementObserver>,
    pub external_teacher: Option<AdaptiveExternalTeacherObserver>,
    /// Nonblocking outer-session pump. It does not run inside a native/CPU
    /// worker and cannot change the manager's original execution authority.
    pub cooperative: Option<Arc<dyn Fn() -> Result<(), AdaptiveManagerError> + Send + Sync>>,
}
impl<B> Default for AdaptiveSessionCollectionHooks<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    fn default() -> Self {
        Self {
            root_admission: None,
            root_closure: None,
            utility_route_completed: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            utility_trial_retired: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            utility_intervention_trial_retired: None,
            external_teacher: None,
            cooperative: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdaptiveSessionOptions {
    pub implementation: Digest,
    pub identity: EngineIdentity,
    pub max_rounds: u64,
    pub max_cpu_nodes: u64,
    pub cpu_depth: u16,
    pub token_budget: u16,
    pub policy: CoordinatorPolicy,
    pub endpoints: CpuEndpointSelection,
    pub controller: ControllerConfig,
    pub device_digest: Digest,
    pub cpu_device_digest: Digest,
    pub external_policy: ExternalSlicePolicy,
}
impl AdaptiveSessionOptions {
    fn validate(&self) -> Result<(), SearchSessionFailure> {
        if self.implementation == Digest([0; 32])
            || self.identity.name.is_empty()
            || self.identity.author.is_empty()
            || self.max_rounds == 0
            || self.max_cpu_nodes == 0
            || self.cpu_depth == 0
            || self.cpu_depth > 64
            || ![128, 256, 512].contains(&self.token_budget)
            || self.controller.policy != self.policy
        {
            return Err(failure(
                "AdaptiveSessionDeclaration",
                "invalid explicit startup settings",
            ));
        }
        self.endpoints
            .validate()
            .map_err(|e| SearchSessionFailure::debug("AdaptiveSessionDeclaration", &e))?;
        Ok(())
    }
}

/// Supplied by the real startup owner. There is no factory/model loader here.
/// `cpu` contains the already-created engines; no duplicate TT is allocated.
pub struct AdaptiveSessionSources<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
{
    pub native: TicketedNativeService<ReviewView, AdaptivePortQuestion, B>,
    pub engine: PalsEngine<AdaptivePortModel>,
    pub cpu: Vec<PersistentCpuOwner>,
    pub external: Option<ExternalManagerWorker>,
    pub checker_metadata: AdaptiveCheckerMetadata,
}
pub struct AdaptiveSessionAdmissionRejection<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
{
    pub failure: SearchSessionFailure,
    pub sources: AdaptiveSessionSources<B>,
}

struct IdleOwners<B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>> {
    native: TicketedNativeService<ReviewView, AdaptivePortQuestion, B>,
    engine: Option<Box<PalsEngine<AdaptivePortModel>>>,
    // A legacy Engine-only transfer must not reinterpret or discard an Idle
    // owner. Retained/pending failure states keep this original value intact.
    unexpected_archive_owner: Option<Box<ArchiveOwner<AdaptivePortModel>>>,
    cpu: Vec<PersistentCpuOwner>,
    external: Option<ExternalManagerWorker>,
    researcher_source: SessionResearcherSource,
    continuation: Option<AdaptiveSequenceContinuation>,
    native_finished: Option<bool>,
    native_joined: bool,
    external_finished: Option<CheckerShutdown>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    profiled_retirement:
        Option<rz_search::pals::engine::adaptive_port::ProfiledExternalRootRetirement>,
    #[cfg(feature = "pals-adaptive")]
    rovent_idle_custody: RoventPcrIdleCustody,
}
struct RestoredSemanticOwner {
    port: AdaptiveEnginePort<AdaptivePortModel>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    retirement: Option<rz_search::pals::engine::adaptive_port::ProfiledExternalRootRetirement>,
}

/// The old public source entry always installs Legacy. Only the separate
/// pinned external-R constructor can install the move-only profiled source.
enum SessionResearcherSource {
    Legacy,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    ProfiledExternal(Box<RoventExternalSessionSource>),
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    ProfiledExternalV2(Box<RoventExternalSessionSource<rz_experiments::RoventInputLockV2>>),
}
#[derive(Clone, Copy)]
enum SessionResearcherClass {
    Legacy,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    ProfiledExternal,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    ProfiledExternalV2,
}
impl SessionResearcherSource {
    fn is_profiled_external(&self) -> bool {
        match self {
            Self::Legacy => false,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Self::ProfiledExternal(_) => true,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Self::ProfiledExternalV2(_) => true,
        }
    }
}

/// Whole-root evidence keeps the actual source kind. External reported work
/// never becomes a successful zero-charge native CPU shutdown receipt.
#[derive(Clone)]
#[allow(
    clippy::large_enum_variant,
    reason = "one bounded original root receipt is retained inline without copying physical owners"
)]
enum SessionRootFence {
    Native(AdaptiveShutdownReceipt),
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    Profiled(super::ProfiledExternalShutdownReceipt),
}
#[allow(
    clippy::large_enum_variant,
    reason = "one transient unique owner moves into the manager without cloning its custody or allocating a second worker"
)]
enum SessionManagerSource {
    Legacy,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    Profiled(Box<super::ProfiledExternalManagerSource>),
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    ProfiledV2(Box<super::ProfiledExternalManagerSource<rz_experiments::RoventInputLockV2>>),
}

// The actual move-only external source can be much larger than the legacy
// manager selector. Keep its typed owner intact on the heap, and keep the
// version-specific Result temporaries out of the common UCI root frame.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[inline(never)]
fn move_profiled_manager_source(
    source: Box<RoventExternalSessionSource>,
    root: Arc<crate::rovent_root_admission::RoventRootAdmission>,
) -> Result<SessionManagerSource, (SessionResearcherSource, SearchSessionFailure)> {
    match source.into_manager(root) {
        Ok(owner) => Ok(SessionManagerSource::Profiled(Box::new(owner))),
        Err((source, error)) => Err((
            SessionResearcherSource::ProfiledExternal(Box::new(source)),
            error,
        )),
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[inline(never)]
fn move_profiled_manager_source_v2(
    source: Box<RoventExternalSessionSource<rz_experiments::RoventInputLockV2>>,
    root: Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV2>,
) -> Result<SessionManagerSource, (SessionResearcherSource, SearchSessionFailure)> {
    match source.into_manager(root) {
        Ok(owner) => Ok(SessionManagerSource::ProfiledV2(Box::new(owner))),
        Err((source, error)) => Err((
            SessionResearcherSource::ProfiledExternalV2(Box::new(source)),
            error,
        )),
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[inline(never)]
fn return_profiled_manager_source(
    source: Box<super::ProfiledExternalManagerSource>,
    metadata: AdaptiveCheckerMetadata,
) -> SessionResearcherSource {
    SessionResearcherSource::ProfiledExternal(Box::new(RoventExternalSessionSource::from_manager(
        *source, metadata,
    )))
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[inline(never)]
fn return_profiled_manager_source_v2(
    source: Box<super::ProfiledExternalManagerSource<rz_experiments::RoventInputLockV2>>,
    metadata: AdaptiveCheckerMetadata,
) -> SessionResearcherSource {
    SessionResearcherSource::ProfiledExternalV2(Box::new(
        RoventExternalSessionSource::from_manager(*source, metadata),
    ))
}

enum SessionManagerRejection<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    Legacy(Box<AdmissionRejection<B>>),
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    Profiled(Box<ProfiledAdmissionRejection<B>>),
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    ProfiledV2(Box<ProfiledAdmissionRejectionV2<B>>),
}
impl SessionRootFence {
    fn permits_root_reuse(&self) -> bool {
        match self {
            Self::Native(receipt) => receipt.permits_root_reuse(),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Self::Profiled(receipt) => receipt.permits_root_reuse(),
        }
    }
    fn permits_archive_close(&self) -> bool {
        match self {
            Self::Native(receipt) => receipt.permits_archive_close(),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Self::Profiled(receipt) => receipt.permits_archive_close(),
        }
    }
    #[cfg(feature = "pals-adaptive")]
    fn idle_custody(&self) -> RoventPcrIdleCustody {
        match self {
            Self::Native(receipt) => RoventPcrIdleCustody::RetiredRoot(receipt.clone()),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Self::Profiled(receipt) => RoventPcrIdleCustody::RetiredProfiledRoot(receipt.clone()),
        }
    }
}

// Retained process owners stay on the heap while failure/root transitions move
// the state. Empty variants must not reserve every possible engine payload on
// the caller's stack, and a failed transition keeps the same exclusive owner.
enum SessionState<B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>>
where
    B::Output: ProposerRankingOutput,
{
    Idle(Box<IdleOwners<B>>),
    Manager(Box<SessionManager<B>>),
    AdmissionRejected(Box<AdmissionRejection<B>>),
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    ProfiledAdmissionRejected(Box<ProfiledAdmissionRejection<B>>),
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    ProfiledAdmissionRejectedV2(Box<ProfiledAdmissionRejectionV2<B>>),
    PortRejected {
        owners: Box<IdleOwners<B>>,
        driver: AdaptivePortDriver<AdaptivePortModel>,
    },
    SpawnRejected {
        owners: Box<IdleOwners<B>>,
        driver: AdaptivePortDriver<AdaptivePortModel>,
        workers: Vec<CpuWorker>,
    },
    ArchivePending {
        owners: Box<IdleOwners<B>>,
        worker: ArchiveWorker<AdaptivePortModel>,
        transfer: ArchiveTransfer,
        authority: SearchAuthority,
    },
    ArchiveQuarantined {
        owners: Box<IdleOwners<B>>,
        worker: ArchiveWorker<AdaptivePortModel>,
        quarantine: Box<ArchiveQuarantine<AdaptivePortModel>>,
    },
    ArchiveJoinedPending {
        owners: Box<IdleOwners<B>>,
        worker: ArchiveWorker<AdaptivePortModel>,
    },
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    ProfiledRetirementPending {
        owners: Box<IdleOwners<B>>,
        worker: ArchiveWorker<AdaptivePortModel>,
        transfer: ArchiveTransfer,
        authority: SearchAuthority,
    },
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    ProfiledRetirementRetained {
        owners: Box<IdleOwners<B>>,
        worker: ArchiveWorker<AdaptivePortModel>,
        payload: Box<
            rz_search::pals::engine::adaptive_port::archive_worker::PersistProfiledRetirementWork,
        >,
    },
    Closed(AdaptiveSessionProcessReceipt),
    // Exists only under the exclusive mutex while moving owners; never a
    // reusable state. Poison/panic is a failed ownership boundary.
    Moving,
}

/// Producer-only process receipt. Root quiescence cannot construct this type.
#[derive(Clone, Debug)]
pub struct AdaptiveSessionProcessReceipt {
    epoch: ProcessEpoch,
    last_authority: Option<SearchAuthority>,
    native_owner_joined: bool,
    native_physical_shutdown_confirmed: bool,
    external_shutdown: Option<CheckerShutdown>,
    store_archive_closed: bool,
    latest_root: Option<AdaptiveShutdownReceipt>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    latest_profiled_root: Option<super::ProfiledExternalShutdownReceipt>,
}
impl AdaptiveSessionProcessReceipt {
    pub fn epoch(&self) -> ProcessEpoch {
        self.epoch
    }
    pub fn last_authority(&self) -> Option<SearchAuthority> {
        self.last_authority
    }
    pub fn native_owner_joined(&self) -> bool {
        self.native_owner_joined
    }
    pub fn native_physical_shutdown_confirmed(&self) -> bool {
        self.native_physical_shutdown_confirmed
    }
    pub fn external_shutdown(&self) -> Option<CheckerShutdown> {
        self.external_shutdown
    }
    pub fn store_archive_closed(&self) -> bool {
        self.store_archive_closed
    }
    pub fn latest_root(&self) -> Option<&AdaptiveShutdownReceipt> {
        self.latest_root.as_ref()
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn latest_profiled_root(&self) -> Option<&super::ProfiledExternalShutdownReceipt> {
        self.latest_profiled_root.as_ref()
    }
    pub fn permits_archive_close(&self) -> bool {
        self.native_owner_joined
            && self.native_physical_shutdown_confirmed
            && self
                .external_shutdown
                .is_none_or(|r| r.cleanup_complete && !r.quarantined && !r.ownership_lost)
            && self
                .latest_root
                .as_ref()
                .is_none_or(AdaptiveShutdownReceipt::permits_root_reuse)
            && {
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                {
                    self.latest_profiled_root.as_ref().is_none_or(
                        super::ProfiledExternalShutdownReceipt::permits_semantic_closure,
                    )
                }
                #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
                {
                    true
                }
            }
    }
    pub fn permits_process_close(&self) -> bool {
        self.permits_archive_close() && self.store_archive_closed
    }
}

#[derive(Clone, Debug)]
pub struct AdaptiveUciOutputObservation {
    pub authority: SearchAuthority,
    pub bestmove: String,
    pub from_report: bool,
}
#[derive(Clone, Debug)]
pub struct AdaptiveSessionRetention {
    pub stage: &'static str,
    pub retained_cpu_engines: usize,
    pub retained_cpu_workers: usize,
    pub native_pending: usize,
    pub archive_owner_pending: bool,
}
/// Read-only projection of one retained owner. No decoder or constructor can
/// turn these observations back into a worker, source, fence or live authority.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[derive(Clone, Debug)]
pub(crate) struct AdaptiveRetainedSessionObservation {
    pub stage: &'static str,
    pub authority: Option<SearchAuthority>,
    pub next_authority: Option<SearchAuthority>,
    pub manager_phase: Option<&'static str>,
    pub manager_first_retained_failure: Option<SearchSessionFailure>,
    pub manager_retained_failure_count: Option<usize>,
    pub archive_operation: Option<&'static str>,
    pub archive_quarantine_reason: Option<SearchSessionFailure>,
    pub native_physical: Option<AdaptiveShutdownReceipt>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub profiled_physical: Option<super::ProfiledExternalShutdownReceipt>,
    pub physical_read_failure: Option<SearchSessionFailure>,
}
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl AdaptiveRetainedSessionObservation {
    fn new(stage: &'static str) -> Self {
        Self {
            stage,
            authority: None,
            next_authority: None,
            manager_phase: None,
            manager_first_retained_failure: None,
            manager_retained_failure_count: None,
            archive_operation: None,
            archive_quarantine_reason: None,
            native_physical: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            profiled_physical: None,
            physical_read_failure: None,
        }
    }
}
struct PendingOutput {
    authority: SearchAuthority,
    report: Option<BoardMove>,
    legal: Vec<BoardMove>,
}

pub struct AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    options: AdaptiveSessionOptions,
    limits: AdaptiveExecutionLimits,
    explicit_native_limits: bool,
    researcher_class: SessionResearcherClass,
    clock: ProcessClock,
    issuer: AdaptiveRulesOwnerIssuer,
    registration: PalsCheckerRegistration,
    checker_metadata: AdaptiveCheckerMetadata,
    external_metadata_bound: AtomicBool,
    state: Mutex<SessionState<B>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    storage: Mutex<Option<storage::EngineStorageSlot>>,
    work: ProcessWorkJournal,
    completion: Mutex<CompletionJournal>,
    reset_requests: AtomicU64,
    closed: AtomicBool,
    dispatch_observer: Option<AdaptiveCollectionDispatchObserver>,
    closure_observer: Option<AdaptiveRootClosureObserver<B>>,
    collection_hooks: AdaptiveSessionCollectionHooks<B>,
    #[cfg(feature = "pals-adaptive-diagnostics")]
    continuation_diagnostic: std::sync::OnceLock<super::ContinuationDiagnostic>,
    latest_root: Mutex<Option<AdaptiveShutdownReceipt>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    latest_profiled_root: Mutex<Option<super::ProfiledExternalShutdownReceipt>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    profiled_retirement_sink: Mutex<
        Option<
            rz_search::pals::engine::adaptive_port::profiled_retirement_archive::RegisteredProfiledRetirementSink,
        >,
    >,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    profiled_retirement_commits: Mutex<VecDeque<serde_json::Value>>,
    rovent: Option<RoventPcrSessionProfile>,
    #[cfg(feature = "pals-adaptive")]
    rovent_root_policy: Option<RoventPcrRootPolicy>,
    #[cfg(feature = "pals-adaptive")]
    rovent_root_policy_v2: Option<RoventPcrRootPolicyV2>,
    rovent_collection_origin: Option<super::RoventNativeCollectionOrigin>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    rovent_collection_origin_v2: Option<super::RoventNativeCollectionOriginV2>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    rovent_collection_hooks_v2: Option<Arc<dyn RoventBoardCollectionHooksV2<B>>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    rovent_collection_failure_v2: Mutex<Option<RoventBoardCollectionAdmissionFailureV2>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    rovent_collection_panic_v2: Mutex<Vec<Box<dyn std::any::Any + Send>>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    rovent_collection_handoff_v2: Mutex<Option<Arc<Mutex<RoventBoardCollectionHandoffV2<B::Output>>>>>,
    latest_rovent_root: Mutex<Option<Arc<RoventNativeRootReceipt>>>,
    #[cfg(feature = "pals-adaptive")]
    latest_rovent_root_v2: Mutex<Option<Arc<RoventNativeRootReceiptV2>>>,
    #[cfg(feature = "pals-adaptive")]
    latest_rovent_initial: Mutex<Option<Arc<RoventPcrInitialReceipt>>>,
    #[cfg(feature = "pals-adaptive")]
    latest_rovent_initial_v2: Mutex<Option<Arc<RoventPcrInitialReceiptV2>>>,
    archive_receipts: Mutex<VecDeque<ArchiveWorkReceipt>>,
    pending_output: Mutex<Option<PendingOutput>>,
    accepted_outputs: Mutex<VecDeque<AdaptiveUciOutputObservation>>,
}

impl<B> AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    fn root_shutdown_receipt(
        &self,
        manager: &SessionManager<B>,
    ) -> Result<SessionRootFence, AdaptiveManagerError> {
        match self.researcher_class {
            SessionResearcherClass::Legacy => {
                manager.shutdown_receipt().map(SessionRootFence::Native)
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionResearcherClass::ProfiledExternal
            | SessionResearcherClass::ProfiledExternalV2 => manager
                .profiled_shutdown_receipt()
                .map(SessionRootFence::Profiled),
        }
    }

    fn run_session_root_observed(
        &self,
        manager: &mut SessionManager<B>,
        until: Instant,
        canceled: &AtomicBool,
        observer: &mut dyn FnMut(&mut SessionManager<B>) -> Result<(), AdaptiveManagerError>,
    ) -> Result<SessionRootFence, AdaptiveManagerError> {
        match self.researcher_class {
            SessionResearcherClass::Legacy => manager
                .run_root_until_observed(until, canceled, observer)
                .map(SessionRootFence::Native),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionResearcherClass::ProfiledExternal
            | SessionResearcherClass::ProfiledExternalV2 => manager
                .run_profiled_root_until_observed(until, canceled, observer)
                .map(SessionRootFence::Profiled),
        }
    }

    fn run_session_process_close(
        &self,
        manager: &mut SessionManager<B>,
        until: Instant,
    ) -> Result<SessionRootFence, AdaptiveManagerError> {
        match self.researcher_class {
            SessionResearcherClass::Legacy => manager
                .run_until(until, &AtomicBool::new(false))
                .map(SessionRootFence::Native),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionResearcherClass::ProfiledExternal
            | SessionResearcherClass::ProfiledExternalV2 => manager
                .run_profiled_until(until, &AtomicBool::new(false))
                .map(SessionRootFence::Profiled),
        }
    }

    fn researcher_limit_observation(&self, manager: &SessionManager<B>) -> Option<bool> {
        match self.researcher_class {
            SessionResearcherClass::Legacy => Some(manager.cpu_node_limit_reached()),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionResearcherClass::ProfiledExternal
            | SessionResearcherClass::ProfiledExternalV2 => {
                manager.profiled_reported_node_limit_reached()
            }
        }
    }

    fn attach_rovent_profile(
        &self,
        profile: &RoventPcrSessionProfile,
        manager: &mut SessionManager<B>,
        context: &SearchSessionContext,
        admission: &RoventPcrRootAdmission,
        _external_source: SessionExternalRootSource<'_>,
    ) -> Result<(), AdaptiveManagerError> {
        match self.researcher_class {
            SessionResearcherClass::Legacy => profile.attach(
                manager,
                context,
                &self.clock,
                self.rovent_collection_origin.as_ref(),
                admission,
            ),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionResearcherClass::ProfiledExternal => profile.attach_profiled_external(
                manager,
                context,
                &self.clock,
                self.rovent_collection_origin.as_ref(),
                admission,
                match _external_source {
                    Some(SessionExternalJoined::V1(source)) => source,
                    _ => {
                        return Err(AdaptiveManagerError::Invalid(
                            "actual /1 external root source is absent",
                        ));
                    }
                },
            ),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionResearcherClass::ProfiledExternalV2 => {
                if self.rovent_collection_origin.is_some() {
                    return Err(AdaptiveManagerError::Invalid(
                        "/2 external Collection origin is not admitted",
                    ));
                }
                let Some(SessionExternalJoined::V2(source)) = _external_source else {
                    return Err(AdaptiveManagerError::Invalid(
                        "actual /2 external root source is absent",
                    ));
                };
                profile.attach_profiled_external_v2(
                    manager,
                    context,
                    &self.clock,
                    admission,
                    source,
                )
            }
        }
    }

    fn admit_session_manager(
        &self,
        registration: AdaptiveManagerRegistration,
        sources: AdaptiveManagerSources<
            AdaptivePortQuestion,
            B,
            AdaptivePortDriver<AdaptivePortModel>,
        >,
        researcher: SessionManagerSource,
        continuation: Option<AdaptiveSequenceContinuation>,
    ) -> Result<Box<SessionManager<B>>, SessionManagerRejection<B>> {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        match researcher {
            SessionManagerSource::Profiled(external) => {
                return SessionManager::new_profiled_external_retaining(
                    registration,
                    self.clock.runtime_clock(),
                    self.limits,
                    sources,
                    *external,
                    continuation,
                )
                .map(|facade| Box::new(facade.into_session_manager()))
                .map_err(|rejected| SessionManagerRejection::Profiled(Box::new(rejected)));
            }
            SessionManagerSource::ProfiledV2(external) => {
                return SessionManager::new_profiled_external_v2_retaining(
                    registration,
                    self.clock.runtime_clock(),
                    self.limits,
                    sources,
                    *external,
                    continuation,
                )
                .map(|facade| Box::new(facade.into_session_manager()))
                .map_err(|rejected| SessionManagerRejection::ProfiledV2(Box::new(rejected)));
            }
            SessionManagerSource::Legacy => {}
        }
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        let SessionManagerSource::Legacy = researcher;
        let admitted = match (continuation, self.explicit_native_limits) {
            (Some(continuation), true) => SessionManager::new_continued_retaining_with_limits(
                registration,
                self.clock.runtime_clock(),
                self.limits,
                sources,
                continuation,
            ),
            (None, true) => SessionManager::new_retaining_with_limits(
                registration,
                self.clock.runtime_clock(),
                self.limits,
                sources,
            ),
            (Some(continuation), false) => SessionManager::new_continued_retaining(
                registration,
                self.clock.runtime_clock(),
                sources,
                continuation,
            ),
            (None, false) => {
                SessionManager::new_retaining(registration, self.clock.runtime_clock(), sources)
            }
        };
        admitted
            .map(Box::new)
            .map_err(|rejected| SessionManagerRejection::Legacy(Box::new(rejected)))
    }

    #[allow(clippy::type_complexity)]
    fn return_manager_owners(
        &self,
        manager: Box<SessionManager<B>>,
        state: &mut SessionState<B>,
        process_closed: bool,
    ) -> Result<
        (
            Box<IdleOwners<B>>,
            AdaptivePortDriver<AdaptivePortModel>,
            SessionRootFence,
        ),
        SearchSessionFailure,
    > {
        match self.researcher_class {
            SessionResearcherClass::Legacy => {
                let parts = if process_closed {
                    manager.into_process_parts_boxed()
                } else {
                    manager.into_persistent_parts_boxed()
                };
                let parts = match parts {
                    Ok(parts) => parts,
                    Err(rejected) => {
                        *state = SessionState::Manager(rejected.manager);
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                };
                let (native, cpu, external, driver, continuation, receipt) =
                    parts.into_components();
                let native_finished =
                    process_closed.then_some(receipt.native_physical_shutdown_confirmed);
                let native_joined = receipt.native_owner_joined;
                let fence = SessionRootFence::Native(receipt);
                Ok((
                    Box::new(IdleOwners {
                        native,
                        cpu,
                        external,
                        researcher_source: SessionResearcherSource::Legacy,
                        engine: None,
                        unexpected_archive_owner: None,
                        continuation: Some(continuation),
                        native_finished,
                        native_joined,
                        external_finished: None,
                        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                        profiled_retirement: None,
                        #[cfg(feature = "pals-adaptive")]
                        rovent_idle_custody: fence.idle_custody(),
                    }),
                    driver,
                    fence,
                ))
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionResearcherClass::ProfiledExternal => {
                let parts = if process_closed {
                    manager.into_profiled_process_parts()
                } else {
                    manager.into_profiled_persistent_parts()
                };
                let parts = match parts {
                    Ok(parts) => parts,
                    Err(rejected) => {
                        *state = SessionState::Manager(Box::new(rejected.manager));
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                };
                let (native, external, driver, continuation, receipt) = parts.into_components();
                let external = RoventExternalSessionSource::from_manager(
                    external,
                    self.checker_metadata.clone(),
                );
                let external_finished = external.shutdown_receipt();
                let native_finished =
                    process_closed.then_some(receipt.native_physical_shutdown_confirmed());
                let native_joined = receipt.native_process_joined();
                let fence = SessionRootFence::Profiled(receipt);
                Ok((
                    Box::new(IdleOwners {
                        native,
                        cpu: Vec::new(),
                        external: None,
                        researcher_source: SessionResearcherSource::ProfiledExternal(Box::new(
                            external,
                        )),
                        engine: None,
                        unexpected_archive_owner: None,
                        continuation: Some(continuation),
                        native_finished,
                        native_joined,
                        external_finished,
                        profiled_retirement: None,
                        rovent_idle_custody: fence.idle_custody(),
                    }),
                    driver,
                    fence,
                ))
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionResearcherClass::ProfiledExternalV2 => {
                let parts = if process_closed {
                    manager.into_profiled_v2_process_parts()
                } else {
                    manager.into_profiled_v2_persistent_parts()
                };
                let parts = match parts {
                    Ok(parts) => parts,
                    Err(rejected) => {
                        *state = SessionState::Manager(Box::new(rejected.manager));
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                };
                let (native, external, driver, continuation, receipt) = parts.into_components();
                let external = RoventExternalSessionSource::from_manager(
                    external,
                    self.checker_metadata.clone(),
                );
                let external_finished = external.shutdown_receipt();
                let native_finished =
                    process_closed.then_some(receipt.native_physical_shutdown_confirmed());
                let native_joined = receipt.native_process_joined();
                let fence = SessionRootFence::Profiled(receipt);
                Ok((
                    Box::new(IdleOwners {
                        native,
                        cpu: Vec::new(),
                        external: None,
                        researcher_source: SessionResearcherSource::ProfiledExternalV2(Box::new(
                            external,
                        )),
                        engine: None,
                        unexpected_archive_owner: None,
                        continuation: Some(continuation),
                        native_finished,
                        native_joined,
                        external_finished,
                        profiled_retirement: None,
                        rovent_idle_custody: fence.idle_custody(),
                    }),
                    driver,
                    fence,
                ))
            }
        }
    }

    // A failed fence/adoption returns the same driver and its Port/Store/archive
    // custody intact; recovery must not depend on a new failure allocation.
    #[allow(clippy::result_large_err)]
    fn restore_semantic_owner(
        &self,
        driver: AdaptivePortDriver<AdaptivePortModel>,
        fence: &SessionRootFence,
    ) -> Result<RestoredSemanticOwner, (AdaptivePortDriver<AdaptivePortModel>, AdaptiveManagerError)>
    {
        match fence {
            SessionRootFence::Native(receipt) => {
                driver.into_port(receipt).map(|port| RestoredSemanticOwner {
                    port,
                    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                    retirement: None,
                })
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionRootFence::Profiled(receipt) => {
                driver
                    .into_profiled_port(receipt)
                    .map(|(port, retirement)| RestoredSemanticOwner {
                        port,
                        retirement: Some(retirement),
                    })
            }
        }
    }

    fn record_root_fence(&self, fence: &SessionRootFence) -> Result<(), SearchSessionFailure> {
        match fence {
            SessionRootFence::Native(receipt) => {
                *self
                    .latest_root
                    .lock()
                    .map_err(|_| failure("AdaptiveRootReceipt", "receipt mutex poisoned"))? =
                    Some(receipt.clone());
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionRootFence::Profiled(receipt) => {
                *self
                    .latest_profiled_root
                    .lock()
                    .map_err(|_| failure("ProfiledRootReceipt", "receipt mutex poisoned"))? =
                    Some(receipt.clone());
            }
        }
        Ok(())
    }

    #[allow(
        clippy::result_large_err,
        reason = "admission rejection returns the original native, CPU and Store owners without a failure-path allocation"
    )]
    pub fn new(
        options: AdaptiveSessionOptions,
        sources: AdaptiveSessionSources<B>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        dispatch_observer: Option<AdaptiveCollectionDispatchObserver>,
        closure_observer: Option<AdaptiveRootClosureObserver<B>>,
    ) -> Result<Self, AdaptiveSessionAdmissionRejection<B>> {
        Self::admit_with_resource_limits(
            options,
            AdaptiveExecutionLimits::default(),
            sources,
            clock,
            issuer,
            registration,
            dispatch_observer,
            closure_observer,
            AdaptiveSessionCollectionHooks::default(),
            false,
        )
    }
    /// Admit declarations only when the already-created engine ceilings agree.
    /// Root startup configures engines before capturing their immutable profile;
    /// this constructor never modifies a retained TT/token namespace in place.
    #[allow(
        clippy::too_many_arguments,
        clippy::result_large_err,
        reason = "independent admission authorities stay explicit and rejection returns their original owners without allocating"
    )]
    pub fn new_with_resource_limits(
        options: AdaptiveSessionOptions,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveSessionSources<B>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        dispatch_observer: Option<AdaptiveCollectionDispatchObserver>,
        closure_observer: Option<AdaptiveRootClosureObserver<B>>,
    ) -> Result<Self, AdaptiveSessionAdmissionRejection<B>> {
        Self::admit_with_resource_limits(
            options,
            limits,
            sources,
            clock,
            issuer,
            registration,
            dispatch_observer,
            closure_observer,
            AdaptiveSessionCollectionHooks::default(),
            true,
        )
    }
    #[allow(
        clippy::too_many_arguments,
        clippy::result_large_err,
        reason = "independent admission authorities stay explicit and rejection returns their original owners without allocating"
    )]
    pub fn new_with_resource_limits_and_collection_hooks(
        options: AdaptiveSessionOptions,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveSessionSources<B>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        dispatch_observer: Option<AdaptiveCollectionDispatchObserver>,
        closure_observer: Option<AdaptiveRootClosureObserver<B>>,
        collection_hooks: AdaptiveSessionCollectionHooks<B>,
    ) -> Result<Self, AdaptiveSessionAdmissionRejection<B>> {
        Self::admit_with_resource_limits(
            options,
            limits,
            sources,
            clock,
            issuer,
            registration,
            dispatch_observer,
            closure_observer,
            collection_hooks,
            true,
        )
    }
    #[allow(
        clippy::too_many_arguments,
        clippy::result_large_err,
        reason = "independent admission authorities stay explicit and rejection returns their original owners without allocating"
    )]
    fn admit_with_resource_limits(
        options: AdaptiveSessionOptions,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveSessionSources<B>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        dispatch_observer: Option<AdaptiveCollectionDispatchObserver>,
        closure_observer: Option<AdaptiveRootClosureObserver<B>>,
        collection_hooks: AdaptiveSessionCollectionHooks<B>,
        explicit_native_limits: bool,
    ) -> Result<Self, AdaptiveSessionAdmissionRejection<B>> {
        Self::admit_with_researcher_source(
            options,
            limits,
            sources,
            clock,
            issuer,
            registration,
            dispatch_observer,
            closure_observer,
            collection_hooks,
            explicit_native_limits,
            SessionResearcherSource::Legacy,
        )
        .map_err(|(rejected, source)| {
            debug_assert!(!source.is_profiled_external());
            rejected
        })
    }

    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn admit_with_researcher_source(
        options: AdaptiveSessionOptions,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveSessionSources<B>,
        clock: ProcessClock,
        issuer: AdaptiveRulesOwnerIssuer,
        registration: PalsCheckerRegistration,
        dispatch_observer: Option<AdaptiveCollectionDispatchObserver>,
        closure_observer: Option<AdaptiveRootClosureObserver<B>>,
        collection_hooks: AdaptiveSessionCollectionHooks<B>,
        explicit_native_limits: bool,
        researcher_source: SessionResearcherSource,
    ) -> Result<
        Self,
        (
            AdaptiveSessionAdmissionRejection<B>,
            SessionResearcherSource,
        ),
    > {
        let profiled_external = researcher_source.is_profiled_external();
        let researcher_class = match &researcher_source {
            SessionResearcherSource::Legacy => SessionResearcherClass::Legacy,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionResearcherSource::ProfiledExternal(_) => {
                SessionResearcherClass::ProfiledExternal
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionResearcherSource::ProfiledExternalV2(_) => {
                SessionResearcherClass::ProfiledExternalV2
            }
        };
        let checked = (|| {
            options.validate()?;
            limits
                .validate()
                .map_err(|e| SearchSessionFailure::debug("AdaptiveResourceLimits", &e))?;
            if explicit_native_limits
                && (sources.native.pending_capacity() != limits.role_tickets_max
                    || sources.native.command_capacity() != limits.worker_commands_max)
            {
                return Err(failure(
                    "AdaptiveNativeCapacity",
                    "actual native channel differs from the declared resource limits",
                ));
            }
            let worker_shape = if profiled_external {
                sources.cpu.is_empty() && sources.external.is_none()
            } else {
                sources.cpu.len() == usize::from(options.controller.max_workers)
                    && [1, 2, 4].contains(&sources.cpu.len())
            };
            if !worker_shape
                || sources.engine.checker_identity() != &registration.identity
                || sources.engine.model_identity() != registration.role_model
                || sources.engine.review_policy().is_none()
                || sources.checker_metadata.identity() != &registration.identity
                || sources.checker_metadata.conditions() != registration.conditions
                || sources.checker_metadata.capabilities() != registration.capabilities
                || !owned_declarations_match(
                    sources.checker_metadata.owned_descriptor(),
                    registration.owned_descriptor.as_ref(),
                )
                || sources.checker_metadata.resume_policy() != registration.cpu_resume_policy
            {
                return Err(failure(
                    "AdaptiveSessionSources",
                    "actual owner/startup metadata mismatch",
                ));
            }
            let resolver = sources.engine.resolver_policy();
            if registration.resolver_version != resolver.version()
                || <[u8; 32]>::from(Sha256::digest(registration.resolver_semantics.as_bytes()))
                    != resolver.semantics_sha256()
            {
                return Err(failure(
                    "AdaptiveResolverDeclaration",
                    "actual resolver differs from registered namespace",
                ));
            }
            let expected_value =
                if resolver == rz_search::pals::engine::ResolverPolicy::ModelWdlRestricted {
                    sources.engine.model_value_identity()
                } else {
                    None
                };
            if expected_value != registration.model_value.as_ref() {
                return Err(failure(
                    "AdaptiveModelValueDeclaration",
                    "actual resolver model value differs from registered namespace",
                ));
            }
            let recheck = sources.engine.post_repair_recheck_policy();
            if sources.engine.search_identity() != recheck.search_identity()
                || sources.engine.refinement_conditions() != recheck.conditions()
            {
                return Err(failure(
                    "AdaptiveSearchLaneUnsupported",
                    "effective lane differs from the registered receipt consumer",
                ));
            }
            for (at, owner) in sources.cpu.iter().enumerate() {
                if owner.engine.paused_stack_max_bytes() != limits.paused_bytes_per_worker_max {
                    return Err(failure(
                        "AdaptiveCpuNamespace",
                        "actual paused owner ceiling differs from the resource declaration",
                    ));
                }
                if matches!(registration.identity, CheckerIdentity::Owned(_)) {
                    if at == 0 {
                        sources
                            .checker_metadata
                            .validate_owned_engine(&owner.engine)
                            .map_err(|e| SearchSessionFailure::debug("AdaptiveCpuNamespace", &e))?;
                    } else {
                        sources
                            .checker_metadata
                            .validate_owned_namespace(&owner.engine)
                            .map_err(|e| SearchSessionFailure::debug("AdaptiveCpuNamespace", &e))?;
                    }
                }
                if owner.profile.max_depth < options.cpu_depth {
                    return Err(failure(
                        "AdaptiveCpuNamespace",
                        "declared depth exceeds actual CPU owner",
                    ));
                }
            }
            if sources.native.ready_clock().is_some_and(|ready| {
                ready.origin() != clock.runtime_clock().origin() || ready.domain() != clock.domain()
            }) {
                return Err(failure(
                    "AdaptiveSessionClock",
                    "actual native process origin differs",
                ));
            }
            if let Some(external) = &sources.external {
                if !external
                    .worker
                    .configuration_matches(&external.config, external.owner)
                {
                    return Err(failure(
                        "AdaptiveExternalNamespace",
                        "actual external owner configuration differs",
                    ));
                }
            }
            if matches!(registration.identity, CheckerIdentity::ExternalUci(_))
                && sources.external.is_none()
                && !profiled_external
            {
                return Err(failure(
                    "AdaptiveExternalNamespace",
                    "external checker has no actual process owner",
                ));
            }
            Ok(())
        })();
        if let Err(failure) = checked {
            return Err((
                AdaptiveSessionAdmissionRejection { failure, sources },
                researcher_source,
            ));
        }
        let resolver = PalsResolverIdentity::from_semantics(
            registration.resolver_version,
            registration.resolver_semantics,
        );
        let recheck = sources.engine.post_repair_recheck_policy();
        let work = match (
            recheck.registration_version(),
            recheck.registration_policy(),
            recheck.conditions(),
        ) {
            (Some(version), Some(policy), Some(conditions)) => {
                let identity = PalsSearchPolicyIdentity {
                    version: version.into(),
                    policy: policy.into(),
                    search_identity: recheck.search_identity().into(),
                    conditions_sha256: Sha256::digest(conditions.as_bytes()).into(),
                };
                match ProcessWorkJournal::new_pals_with_search_policy(resolver, identity) {
                    Ok(work) => work,
                    Err(error) => {
                        return Err((
                            AdaptiveSessionAdmissionRejection {
                                failure: SearchSessionFailure::debug(
                                    "AdaptiveSearchPolicy",
                                    &error,
                                ),
                                sources,
                            },
                            researcher_source,
                        ));
                    }
                }
            }
            _ => ProcessWorkJournal::new_pals(resolver),
        };
        let AdaptiveSessionSources {
            native,
            engine,
            cpu,
            external,
            checker_metadata,
        } = sources;
        Ok(Self {
            completion: Mutex::new(CompletionJournal::new(
                clock.epoch(),
                options.implementation,
            )),
            options,
            limits,
            explicit_native_limits,
            researcher_class,
            clock,
            issuer,
            registration,
            checker_metadata,
            external_metadata_bound: AtomicBool::new(false),
            state: Mutex::new(SessionState::Idle(Box::new(IdleOwners {
                native,
                engine: Some(Box::new(engine)),
                unexpected_archive_owner: None,
                cpu,
                external,
                researcher_source,
                continuation: None,
                native_finished: None,
                native_joined: false,
                external_finished: None,
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                profiled_retirement: None,
                #[cfg(feature = "pals-adaptive")]
                rovent_idle_custody: RoventPcrIdleCustody::StartupReady,
            }))),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            storage: Mutex::new(None),
            work,
            reset_requests: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            dispatch_observer,
            closure_observer,
            collection_hooks,
            #[cfg(feature = "pals-adaptive-diagnostics")]
            continuation_diagnostic: std::sync::OnceLock::new(),
            latest_root: Mutex::new(None),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            latest_profiled_root: Mutex::new(None),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            profiled_retirement_sink: Mutex::new(None),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            profiled_retirement_commits: Mutex::new(VecDeque::with_capacity(64)),
            rovent: None,
            #[cfg(feature = "pals-adaptive")]
            rovent_root_policy: None,
            #[cfg(feature = "pals-adaptive")]
            rovent_root_policy_v2: None,
            rovent_collection_origin: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            rovent_collection_origin_v2: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            rovent_collection_hooks_v2: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            rovent_collection_failure_v2: Mutex::new(None),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            rovent_collection_panic_v2: Mutex::new(Vec::new()),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            rovent_collection_handoff_v2: Mutex::new(None),
            latest_rovent_root: Mutex::new(None),
            #[cfg(feature = "pals-adaptive")]
            latest_rovent_root_v2: Mutex::new(None),
            #[cfg(feature = "pals-adaptive")]
            latest_rovent_initial: Mutex::new(None),
            #[cfg(feature = "pals-adaptive")]
            latest_rovent_initial_v2: Mutex::new(None),
            archive_receipts: Mutex::new(VecDeque::with_capacity(64)),
            pending_output: Mutex::new(None),
            accepted_outputs: Mutex::new(VecDeque::with_capacity(64)),
        })
    }

    pub fn process_epoch(&self) -> ProcessEpoch {
        self.clock.epoch()
    }
    pub fn execution_limits(&self) -> AdaptiveExecutionLimits {
        self.limits
    }
    /// Opt-in metadata only, once while the original retained process is idle.
    #[cfg(feature = "pals-adaptive-diagnostics")]
    pub fn install_continuation_diagnostic(
        &self,
        diagnostic: super::ContinuationDiagnostic,
    ) -> Result<(), super::ContinuationDiagnosticError> {
        use super::ContinuationDiagnosticError as Error;
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(Error::Busy),
            Err(TryLockError::Poisoned(_)) => return Err(Error::Poisoned),
        };
        let latest = match self.latest_root.try_lock() {
            Ok(latest) => latest,
            Err(TryLockError::WouldBlock) => return Err(Error::Busy),
            Err(TryLockError::Poisoned(_)) => return Err(Error::Poisoned),
        };
        if self.closed.load(Ordering::Acquire)
            || !matches!(&*state, SessionState::Idle(_))
            || latest.is_some()
        {
            return Err(Error::AdmissionAlreadyStarted);
        }
        self.continuation_diagnostic
            .set(diagnostic)
            .map_err(|_| Error::AlreadyInstalled)
    }
    /// Last returned run's actual final semantic reason and driver return,
    /// in a separate bounded diagnostic. Compare its authority before use;
    /// an active root has not returned yet. This grants no output/close authority.
    pub fn completion_diagnostic(&self) -> Result<AdaptiveCompletionDiagnostic, ContractError> {
        self.completion
            .lock()
            .map(|journal| journal.snapshot())
            .map_err(|_| poisoned("adaptive completion diagnostic unavailable"))
    }
    /// A nonblocking metadata read of the original idle native owner. This
    /// does not observe the bootstrap caller as the native thread, start any
    /// work, or turn a failed/absent OS reading into an applied placement.
    pub fn native_ready_cpu_thread_observation(
        &self,
    ) -> Result<
        Option<
            Result<rz_runtime::adaptive::CpuThreadReadback, rz_runtime::adaptive::CpuResourceError>,
        >,
        SearchSessionFailure,
    > {
        let state = self.state.try_lock().map_err(|_| {
            failure(
                "AdaptiveResourceOwnerBusy",
                "original owner busy or poisoned",
            )
        })?;
        match &*state {
            SessionState::Idle(owners) if !self.closed.load(Ordering::Acquire) => {
                Ok(owners.native.ready_cpu_thread_observation())
            }
            _ => Err(failure(
                "AdaptiveResourceOwnerUnavailable",
                "native startup owner is not idle and live",
            )),
        }
    }
    /// Only the successful actual startup owner's immutable process identity
    /// crosses this boundary. The read-only resource probe must bracket PID,
    /// process group and start ticks itself; readyok proves no applied option.
    pub fn external_startup_process_identity(
        &self,
    ) -> Result<Option<rz_search::cpu_checker::ExternalProcessIdentity>, SearchSessionFailure> {
        let state = self.state.try_lock().map_err(|_| {
            failure(
                "AdaptiveResourceOwnerBusy",
                "original owner busy or poisoned",
            )
        })?;
        let SessionState::Idle(owners) = &*state else {
            return Err(failure(
                "AdaptiveResourceOwnerUnavailable",
                "external startup owner is not idle",
            ));
        };
        if self.closed.load(Ordering::Acquire) {
            return Err(failure(
                "AdaptiveResourceOwnerUnavailable",
                "admission closed",
            ));
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let SessionResearcherSource::ProfiledExternal(source) = &owners.researcher_source {
            return source
                .running_owner()
                .map(|owner| {
                    owner
                        .verified_startup()
                        .map(|startup| startup.map(|startup| startup.process_identity()))
                        .map_err(|error| {
                            SearchSessionFailure::debug("RoventExternalResourceSource", &error)
                        })
                })
                .unwrap_or(Ok(None));
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let SessionResearcherSource::ProfiledExternalV2(source) = &owners.researcher_source {
            return source
                .running_owner()
                .map(|owner| {
                    owner
                        .verified_startup()
                        .map(|startup| startup.map(|startup| startup.process_identity()))
                        .map_err(|error| {
                            SearchSessionFailure::debug("RoventExternalResourceSource", &error)
                        })
                })
                .unwrap_or(Ok(None));
        }
        let Some(external) = &owners.external else {
            return Ok(None);
        };
        let startup = external.worker.startup_report().ok_or_else(|| {
            failure(
                "AdaptiveResourceStartupUnknown",
                "actual startup not observed",
            )
        })?;
        startup.result.map_err(|error| {
            SearchSessionFailure::debug("AdaptiveResourceStartupFailed", &error)
        })?;
        let process = startup
            .attempt
            .and_then(|attempt| attempt.external)
            .map(|attempt| attempt.process)
            .ok_or_else(|| {
                failure(
                    "AdaptiveResourceIdentityUnknown",
                    "original process evidence missing",
                )
            })?;
        if process.quit_sent
            || process.exit_observed
            || process.cleanup_complete
            || process.quarantined
            || process.ownership_lost
        {
            return Err(failure(
                "AdaptiveResourceOwnerUnavailable",
                "historical startup process is no longer reusable",
            ));
        }
        let identity = process.process_identity.filter(|identity| {
            identity.pid > 0
                && identity.process_group > 0
                && identity.proc_start_ticks > 0
                && identity.pid != std::process::id()
                && identity.pid <= i32::MAX as u32
                && identity.process_group <= i32::MAX as u32
        });
        identity.map(Some).ok_or_else(|| {
            failure(
                "AdaptiveResourceIdentityUnknown",
                "native spawn identity unavailable",
            )
        })
    }
    // The actual native bootstrap alone owns the private collection context.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn collection_context_ready(&self) -> Result<(), ContractError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(poisoned("adaptive process admission closed"));
        }
        let state = self
            .state
            .try_lock()
            .map_err(|_| poisoned("actual adaptive owner is busy or poisoned"))?;
        if !matches!(&*state, SessionState::Idle(_)) {
            return Err(poisoned("actual adaptive owner is retained or unclosed"));
        }
        let pending = self
            .pending_output
            .lock()
            .map_err(|_| poisoned("adaptive collection output authority"))?;
        if pending.is_some() {
            return Err(poisoned("previous adaptive output authority is unretired"));
        }
        Ok(())
    }
    /// Revocation is owned by the UCI collection context wrapper. It grants no
    /// UCI publication observation and cannot acknowledge an emitted bestmove.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn retire_collection_output(
        &self,
        authority: SearchAuthority,
    ) -> Result<(), ContractError> {
        // Match execute's state -> pending lock order. A running Search owner
        // may still publish its final pending report; retire only after it has
        // returned, and retain the outer active context when this guard rejects.
        let _state = self
            .state
            .try_lock()
            .map_err(|_| poisoned("actual adaptive owner has not returned"))?;
        let mut pending = self
            .pending_output
            .lock()
            .map_err(|_| poisoned("adaptive collection output authority"))?;
        if pending
            .as_ref()
            .is_some_and(|registered| registered.authority != authority)
        {
            return Err(poisoned("stale adaptive collection retirement"));
        }
        *pending = None;
        Ok(())
    }
    pub fn retained_owner_status(&self) -> Result<AdaptiveSessionRetention, ContractError> {
        // Failure diagnostics must not wait behind the very search whose
        // original drain deadline has expired. Busy is unobserved retention,
        // never evidence that the retained owners or physical leases are gone.
        let state = self
            .state
            .try_lock()
            .map_err(|_| poisoned("adaptive owner retention unavailable or busy"))?;
        let snapshot = |stage, owners: &IdleOwners<B>, workers, archive| AdaptiveSessionRetention {
            stage,
            retained_cpu_engines: owners.cpu.len(),
            retained_cpu_workers: workers,
            native_pending: owners.native.occupancy(),
            archive_owner_pending: archive || owners.unexpected_archive_owner.is_some(),
        };
        Ok(match &*state {
            SessionState::Idle(owners) => snapshot("idle", owners, 0, false),
            SessionState::Manager(manager) => AdaptiveSessionRetention {
                stage: "manager_retained",
                retained_cpu_engines: 0,
                retained_cpu_workers: usize::from(self.options.controller.max_workers),
                native_pending: manager.native_occupancy(),
                archive_owner_pending: manager.driver().port().is_none(),
            },
            SessionState::AdmissionRejected(rejected) => AdaptiveSessionRetention {
                stage: "admission_rejected",
                retained_cpu_engines: 0,
                retained_cpu_workers: rejected.sources.own_workers.len(),
                native_pending: rejected.sources.native.occupancy(),
                archive_owner_pending: rejected.sources.driver.port().is_none(),
            },
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionState::ProfiledAdmissionRejected(rejected) => AdaptiveSessionRetention {
                stage: "profiled_admission_rejected",
                retained_cpu_engines: 0,
                retained_cpu_workers: 0,
                native_pending: rejected.sources.native.occupancy(),
                archive_owner_pending: rejected.sources.driver.port().is_none(),
            },
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionState::ProfiledAdmissionRejectedV2(rejected) => AdaptiveSessionRetention {
                stage: "profiled_admission_rejected_v2",
                retained_cpu_engines: 0,
                retained_cpu_workers: 0,
                native_pending: rejected.sources.native.occupancy(),
                archive_owner_pending: rejected.sources.driver.port().is_none(),
            },
            SessionState::PortRejected { owners, driver } => {
                snapshot("port_rejected", owners, 0, driver.port().is_none())
            }
            SessionState::SpawnRejected {
                owners,
                driver,
                workers,
            } => snapshot(
                "cpu_spawn_rejected",
                owners,
                workers.len(),
                driver.port().is_none(),
            ),
            SessionState::ArchivePending {
                owners,
                worker,
                transfer,
                authority,
            } => {
                let _scope = (transfer.nonce(), authority);
                snapshot("archive_pending", owners, 0, worker.is_busy())
            }
            SessionState::ArchiveQuarantined {
                owners,
                worker,
                quarantine,
            } => {
                let _reason = &quarantine.reason;
                snapshot("archive_quarantined", owners, 0, worker.is_busy())
            }
            SessionState::ArchiveJoinedPending { owners, worker } => {
                snapshot("archive_join_pending", owners, 0, worker.is_busy())
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionState::ProfiledRetirementPending { owners, worker, .. } => {
                snapshot("profiled_retirement_pending", owners, 0, worker.is_busy())
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionState::ProfiledRetirementRetained {
                owners,
                worker,
                payload,
            } => {
                let _retained = payload.retirement();
                snapshot(
                    "profiled_retirement_return_rejected",
                    owners,
                    0,
                    worker.is_busy(),
                )
            }
            SessionState::Closed(_) => AdaptiveSessionRetention {
                stage: "process_closed",
                retained_cpu_engines: 0,
                retained_cpu_workers: 0,
                native_pending: 0,
                archive_owner_pending: false,
            },
            SessionState::Moving => AdaptiveSessionRetention {
                stage: "owner_transfer_unknown",
                retained_cpu_engines: 0,
                retained_cpu_workers: 0,
                native_pending: 0,
                archive_owner_pending: true,
            },
        })
    }
    /// Only the explicitly enabled product diagnostic calls this after run
    /// returns. Never wait behind a live owner, reopen an asset, poll a worker,
    /// consume an event, acknowledge a fence or alter its failure/custody.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn retained_session_observation(
        &self,
    ) -> Result<AdaptiveRetainedSessionObservation, ContractError> {
        let state = self
            .state
            .try_lock()
            .map_err(|_| poisoned("retained diagnostic owner unavailable or busy"))?;
        let stage = match &*state {
            SessionState::Idle(_) => "idle",
            SessionState::Manager(_) => "manager_retained",
            SessionState::AdmissionRejected(_) => "admission_rejected",
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionState::ProfiledAdmissionRejected(_) => "profiled_admission_rejected",
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionState::ProfiledAdmissionRejectedV2(_) => "profiled_admission_rejected_v2",
            SessionState::PortRejected { .. } => "port_rejected",
            SessionState::SpawnRejected { .. } => "cpu_spawn_rejected",
            SessionState::ArchivePending { .. } => "archive_pending",
            SessionState::ArchiveQuarantined { .. } => "archive_quarantined",
            SessionState::ArchiveJoinedPending { .. } => "archive_join_pending",
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionState::ProfiledRetirementPending { .. } => "profiled_retirement_pending",
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionState::ProfiledRetirementRetained { .. } => {
                "profiled_retirement_return_rejected"
            }
            SessionState::Closed(_) => "process_closed",
            SessionState::Moving => "owner_transfer_unknown",
        };
        let mut observed = AdaptiveRetainedSessionObservation::new(stage);
        match &*state {
            SessionState::Manager(manager) => {
                observed.authority = manager.driver().port().map(|port| port.authority());
                observed.manager_phase = Some(match manager.phase() {
                    ManagerPhase::Running => "running",
                    ManagerPhase::Draining => "draining",
                    ManagerPhase::Quarantined => "quarantined",
                    ManagerPhase::Finished => "finished",
                });
                observed.manager_retained_failure_count = Some(manager.failures().count());
                observed.manager_first_retained_failure = manager
                    .failures()
                    .next()
                    .map(|cause| retained_cause("RetainedManagerEarliestFailure", cause));
                match self.root_shutdown_receipt(manager) {
                    Ok(SessionRootFence::Native(receipt)) => {
                        observed.authority = Some(receipt.authority);
                        observed.native_physical = Some(receipt);
                    }
                    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                    Ok(SessionRootFence::Profiled(receipt)) => {
                        observed.authority = Some(receipt.authority());
                        observed.profiled_physical = Some(receipt);
                    }
                    Err(cause) => {
                        observed.physical_read_failure =
                            Some(retained_cause("RetainedPhysicalReceiptRead", &cause))
                    }
                }
            }
            SessionState::ArchiveQuarantined { quarantine, .. } => {
                use rz_search::pals::engine::adaptive_port::archive_worker::ArchiveOperationKind;
                observed.authority = Some(quarantine.receipt.authority);
                observed.next_authority = quarantine.receipt.next_authority;
                observed.archive_operation = Some(match quarantine.receipt.operation {
                    ArchiveOperationKind::PrepareRoot => "prepare_root",
                    ArchiveOperationKind::LoadRoot => "load_root",
                    ArchiveOperationKind::RestoreRootIfArchived => "restore_root_if_archived",
                    ArchiveOperationKind::FinishIdleGame => "finish_idle_game",
                    ArchiveOperationKind::ReclaimHot => "reclaim_hot",
                    ArchiveOperationKind::PrepareForeign => "prepare_foreign",
                    ArchiveOperationKind::NewGame => "new_game",
                    ArchiveOperationKind::PrepareProfiledForeign => "prepare_profiled_foreign",
                    ArchiveOperationKind::PersistProfiledRetirement => {
                        "persist_profiled_retirement"
                    }
                });
                observed.archive_quarantine_reason =
                    Some(retained_archive_cause(&quarantine.reason));
            }
            SessionState::ArchivePending { authority, .. } => observed.authority = Some(*authority),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionState::ProfiledRetirementPending { authority, .. } => {
                observed.authority = Some(*authority)
            }
            SessionState::AdmissionRejected(rejected) => {
                observed.manager_first_retained_failure =
                    Some(retained_cause("RetainedManagerAdmission", &rejected.error))
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionState::ProfiledAdmissionRejected(rejected) => {
                observed.manager_first_retained_failure = Some(retained_cause(
                    "RetainedProfiledManagerAdmission",
                    &rejected.error,
                ))
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionState::ProfiledAdmissionRejectedV2(rejected) => {
                observed.manager_first_retained_failure = Some(retained_cause(
                    "RetainedProfiledManagerAdmissionV2",
                    &rejected.error,
                ))
            }
            _ => {}
        }
        Ok(observed)
    }
    pub fn latest_root_receipt(&self) -> Result<Option<AdaptiveShutdownReceipt>, ContractError> {
        self.latest_root
            .lock()
            .map(|r| r.clone())
            .map_err(|_| poisoned("adaptive root receipt"))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn latest_profiled_root_receipt(
        &self,
    ) -> Result<Option<super::ProfiledExternalShutdownReceipt>, ContractError> {
        self.latest_profiled_root
            .lock()
            .map(|receipt| receipt.clone())
            .map_err(|_| poisoned("profiled root receipt owner poisoned"))
    }
    pub fn uci_output_observations(
        &self,
    ) -> Result<Vec<AdaptiveUciOutputObservation>, ContractError> {
        self.accepted_outputs
            .lock()
            .map(|r| r.iter().cloned().collect())
            .map_err(|_| poisoned("adaptive output observations"))
    }

    fn archive_operation(
        &self,
        state: &mut SessionState<B>,
        mut owners: Box<IdleOwners<B>>,
        next: SearchAuthority,
        operation: ArchiveOperation,
        until: Instant,
        context: &SearchSessionContext,
    ) -> Result<Box<IdleOwners<B>>, SearchSessionFailure> {
        let engine = owners.engine.take().expect("exclusive idle Search owner");
        let authority = match engine.adaptive_archive_authority(next) {
            Ok(authority) => authority,
            Err(error) => {
                owners.engine = Some(engine);
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::debug(
                    "AdaptiveArchiveAuthority",
                    &error,
                ));
            }
        };
        let signal = owners.native.signal();
        let mut worker = match ArchiveWorker::start_with_notifier(move || signal.notify()) {
            Ok(worker) => worker,
            Err(error) => {
                owners.engine = Some(engine);
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::debug("AdaptiveArchiveOwner", &error));
            }
        };
        let transfer = match worker.submit(
            ArchiveOwner::Engine(*engine),
            authority,
            ArchiveWork {
                operation,
                deadline: until,
                max_io_bytes: 16 * 1024 * 1024,
            },
        ) {
            Ok(transfer) => transfer,
            Err(rejected) => {
                let error =
                    SearchSessionFailure::debug("AdaptiveArchiveAdmission", &rejected.error);
                match rejected.owner {
                    ArchiveOwner::Engine(engine) => owners.engine = Some(Box::new(engine)),
                    owner @ (ArchiveOwner::Port(_) | ArchiveOwner::Idle(_)) => {
                        owners.unexpected_archive_owner = Some(Box::new(owner));
                        worker.request_shutdown();
                        *state = SessionState::ArchiveJoinedPending { owners, worker };
                        return Err(error.unknown_if_unclosed());
                    }
                }
                owners = join_archive_lifecycle(state, owners, worker, context.shutdown_deadline)?;
                *state = SessionState::Idle(owners);
                return Err(error);
            }
        };
        loop {
            match worker.try_receive(&transfer) {
                Ok(Some(completed)) => match completed.install(authority, transfer.nonce()) {
                    Ok((ArchiveOwner::Engine(engine), receipt)) => {
                        owners.engine = Some(Box::new(engine));
                        match self.archive_receipts.lock() {
                            Ok(mut receipts) => {
                                if receipts.len() == 64 {
                                    receipts.pop_front();
                                }
                                receipts.push_back(receipt);
                            }
                            Err(_) => {
                                worker.request_shutdown();
                                *state = SessionState::ArchiveJoinedPending { owners, worker };
                                return Err(SearchSessionFailure::physical_completion_unknown());
                            }
                        }
                        return join_archive_lifecycle(
                            state,
                            owners,
                            worker,
                            context.shutdown_deadline,
                        );
                    }
                    Ok((owner, _)) => {
                        // Submission was Engine-only; retain any invalid owner.
                        match legacy_archive_engine(owner) {
                            Ok(engine) => owners.engine = Some(engine),
                            Err(owner) => {
                                owners.unexpected_archive_owner = Some(owner);
                                worker.request_shutdown();
                                *state = SessionState::ArchiveJoinedPending { owners, worker };
                                return Err(failure(
                                    "AdaptiveArchiveReturn",
                                    "returned Idle owner cannot replace the legacy Engine",
                                )
                                .unknown_if_unclosed());
                            }
                        }
                        owners = join_archive_lifecycle(
                            state,
                            owners,
                            worker,
                            context.shutdown_deadline,
                        )?;
                        *state = SessionState::Idle(owners);
                        return Err(failure(
                            "AdaptiveArchiveReturn",
                            "returned owner kind differs",
                        ));
                    }
                    Err(quarantine) => {
                        *state = SessionState::ArchiveQuarantined {
                            owners,
                            worker,
                            quarantine: Box::new(quarantine),
                        };
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                },
                Ok(None) if Instant::now() < until && context.accepts().unwrap_or(false) => {
                    std::thread::sleep(Duration::from_millis(1))
                }
                _ => {
                    transfer.cancel();
                    worker.request_shutdown();
                    *state = SessionState::ArchivePending {
                        owners,
                        worker,
                        transfer,
                        authority,
                    };
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
            }
        }
    }

    fn execute(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
        observation: &mut AttemptObservation,
        semantic_completion: &mut Option<PalsCompletion>,
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        if self.rovent.is_none()
            && (context.control.max_simulations == 0
                || context.control.max_simulations > self.options.max_cpu_nodes)
        {
            return Err(failure(
                "AdaptiveCpuCredit",
                "go CPU node credit exceeds registered startup maximum",
            ));
        }
        if self.closed.load(Ordering::Acquire) {
            // closed may mean retained failed-lifecycle quarantine, not an
            // actual physical finish. A later go cannot turn it into success.
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        if context.authority.epoch != self.clock.epoch()
            || context.authority.implementation != self.options.implementation
        {
            return Err(failure(
                "AdaptiveSessionAuthority",
                "closed or mismatched UCI process authority",
            ));
        }
        let rovent_root_legal = self.rovent_root_legal(position, context)?;
        let root_source = self.prepare_rovent_go(position, context)?;
        let mut state = loop {
            match self.state.try_lock() {
                Ok(state) => break state,
                Err(TryLockError::WouldBlock)
                    if Instant::now() < context.shutdown_deadline
                        && (root_source.is_some() || context.accepts().unwrap_or(false)) =>
                {
                    std::thread::sleep(Duration::from_millis(1))
                }
                Err(TryLockError::WouldBlock) => {
                    return Err(failure(
                        "AdaptiveSessionBusy",
                        "previous search/lifecycle owner has not returned",
                    ));
                }
                Err(TryLockError::Poisoned(_)) => {
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
            }
        };
        if self.closed.load(Ordering::Acquire) {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        // Resolve the final source gate only while borrowing the actual idle
        // owner. Initially stopped product roots leave reset/game/Store/credit
        // unchanged and return no live capability or root-dependent work.
        let root_admission =
            match root_source {
                None => None,
                Some(RoventPcrRootSource::Legacy(budget)) => Some(RoventPcrRootAdmission {
                    budget,
                    #[cfg(feature = "pals-adaptive")]
                    original: None,
                    #[cfg(feature = "pals-adaptive")]
                    original_v2: None,
                }),
                #[cfg(feature = "pals-adaptive")]
                Some(RoventPcrRootSource::Original(source)) => {
                    use crate::rovent_root_admission::source_seal::InitialAdmissionOutcome;
                    match source.into_initial().map_err(|error| {
                        SearchSessionFailure::debug("RoventRootAdmission", &error)
                    })? {
                        InitialAdmissionOutcome::Admitted(original) => {
                            let original = Arc::new(original);
                            Some(RoventPcrRootAdmission {
                                budget: original.root_budget_digest(),
                                original: Some(original),
                                original_v2: None,
                            })
                        }
                        InitialAdmissionOutcome::StoppedInitial(stopped) => {
                            let SessionState::Idle(owners) = &*state else {
                                return Err(SearchSessionFailure::physical_completion_unknown());
                            };
                            return self.finish_rovent_initial(owners, context, stopped);
                        }
                    }
                }
                #[cfg(feature = "pals-adaptive")]
                Some(RoventPcrRootSource::OriginalV2(source)) => {
                    use crate::rovent_root_admission::source_seal::InitialAdmissionOutcomeV2;
                    let SessionState::Idle(owners) = &*state else {
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    };
                    let original_owner = self
                        .rovent_root_policy_v2
                        .as_ref()
                        .is_some_and(|policy| policy.owner.same_original_owner(source.owner()));
                    self.validate_initial_idle(owners, original_owner)?;
                    match source.into_initial().map_err(|error| {
                        SearchSessionFailure::debug("RoventRootAdmissionV2", &error)
                    })? {
                        InitialAdmissionOutcomeV2::Admitted(original) => {
                            let budget =
                                original
                                    .descriptor()
                                    .root_budget_digest()
                                    .map_err(|error| {
                                        SearchSessionFailure::debug("RoventRootBudgetV2", &error)
                                    })?;
                            Some(RoventPcrRootAdmission {
                                budget,
                                original: None,
                                original_v2: Some(Arc::new(original)),
                            })
                        }
                        InitialAdmissionOutcomeV2::StoppedInitial(stopped) => {
                            let SessionState::Idle(owners) = &*state else {
                                return Err(SearchSessionFailure::physical_completion_unknown());
                            };
                            return self.finish_rovent_initial_v2(owners, context, stopped);
                        }
                    }
                }
            };
        let mut initial_game = InitialGameFailureGuard {
            closed: &self.closed,
            armed: false,
        };
        let mut owners = match std::mem::replace(&mut *state, SessionState::Moving) {
            SessionState::Idle(owners) => owners,
            retained => {
                *state = retained;
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if owners.profiled_retirement.is_some() {
            *state = SessionState::Idle(owners);
            return Err(failure(
                "RoventExternalEvidencePending",
                "the original retired external packet/Store evidence still requires handoff",
            ));
        }
        if !context.accepts().unwrap_or(false) {
            *state = SessionState::Idle(owners);
            return Err(failure(
                "AdaptiveSessionCanceled",
                "root revoked before actual admission",
            ));
        }
        let root_legal = rovent_root_legal.unwrap_or_else(|| position.legal_moves());
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let SessionResearcherSource::ProfiledExternal(source) = &mut owners.researcher_source {
            let original = root_admission
                .as_ref()
                .and_then(|admission| admission.original.clone());
            let Some(original) = original else {
                *state = SessionState::Idle(owners);
                return Err(failure(
                    "RoventExternalRoot",
                    "external R requires the original per-go capability",
                ));
            };
            if let Err(error) = source.prepare_actual_root(original, &self.clock) {
                self.closed.store(true, Ordering::Release);
                *state = SessionState::Idle(owners);
                return Err(error.unknown_if_unclosed());
            }
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let SessionResearcherSource::ProfiledExternalV2(source) = &mut owners.researcher_source {
            let Some(original) = root_admission
                .as_ref()
                .and_then(|admission| admission.original_v2.clone())
            else {
                *state = SessionState::Idle(owners);
                return Err(failure(
                    "RoventExternalRoot",
                    "external R requires the original /2 per-go capability",
                ));
            };
            if let Err(error) = source.prepare_actual_root(original, &self.clock) {
                self.closed.store(true, Ordering::Release);
                *state = SessionState::Idle(owners);
                return Err(error.unknown_if_unclosed());
            }
        }
        match self.pending_output.lock() {
            Ok(mut output) => {
                *output = Some(PendingOutput {
                    authority: context.authority,
                    report: None,
                    legal: root_legal.clone(),
                })
            }
            Err(_) => {
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        }
        if let Err(error) = self.bind_actual_external(&owners) {
            *state = SessionState::Idle(owners);
            return Err(error);
        }
        let engine = owners.engine.as_ref().expect("idle Search owner");
        let game = match engine.adaptive_archive_authority(context.authority) {
            Ok(authority) => authority.game,
            Err(error) => {
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::debug("AdaptiveRootNamespace", &error));
            }
        };
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let storage_prepared = match self.prepare_engine_storage(
            &mut owners,
            root_admission.as_ref(),
            game != context.authority.game,
            position,
            context,
            &mut initial_game.armed,
        ) {
            Ok(selected) => selected,
            Err(error) => {
                self.closed.store(true, Ordering::Release);
                *state = SessionState::Idle(owners);
                return Err(error.unknown_if_unclosed());
            }
        };
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        let storage_prepared = false;
        if !storage_prepared && game != context.authority.game {
            if let Err(error) = self.reset_actual_cpu(&mut owners, context, &mut initial_game.armed)
            {
                // A failed reset may have changed an actual external game or
                // cleared a CPU owner. Retain it for process cleanup instead
                // of retrying against the old semantic Store authority.
                self.closed.store(true, Ordering::Release);
                *state = SessionState::Idle(owners);
                return Err(error);
            }
            owners = match self.archive_operation(
                &mut state,
                owners,
                context.authority,
                ArchiveOperation::NewGame {
                    next_authority: context.authority,
                },
                context.control.deadline,
                context,
            ) {
                Ok(owners) => owners,
                Err(error) => {
                    // A reset arm may have been issued but not consumed by the
                    // failed lifecycle. Preserve it with its exact owner and
                    // close further roots; process finish is the only recovery.
                    self.closed.store(true, Ordering::Release);
                    return Err(error);
                }
            };
        }
        if !storage_prepared
            && owners
                .engine
                .as_ref()
                .expect("Search owner")
                .archive_enabled()
        {
            owners = self.archive_operation(
                &mut state,
                owners,
                context.authority,
                ArchiveOperation::PrepareRoot {
                    position: position.clone(),
                },
                context.control.deadline,
                context,
            )?;
        }
        let deadline = match self.clock.deadline(context.control.deadline) {
            Ok(deadline) => deadline,
            Err(error) => {
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::debug("AdaptiveRootDeadline", &error));
            }
        };
        let mut engine = owners.engine.take().expect("Search owner");
        let root = match engine.register_adaptive_root(position, context.authority) {
            Ok(root) => root,
            Err(error) => {
                owners.engine = Some(engine);
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::debug(
                    "AdaptiveRootRegistration",
                    &error,
                ));
            }
        };
        let limits = PalsLimits {
            deadline: context.control.deadline,
            max_rounds: self.options.max_rounds,
            max_cpu_nodes: if self.rovent.is_some() {
                self.options.max_cpu_nodes
            } else {
                context.control.max_simulations
            },
            cpu_depth: self.options.cpu_depth,
        };
        let port = match AdaptiveEnginePort::new_registered_retaining(
            *engine,
            position.clone(),
            context.authority,
            deadline,
            limits,
            Arc::clone(&self.issuer),
            root,
        ) {
            Ok(port) => port,
            Err((mut engine, error)) => {
                let abandoned = engine.abandon_unconsumed_adaptive_root(context.authority);
                let failure = SearchSessionFailure::debug("AdaptiveRootPort", &(error, abandoned));
                owners.engine = Some(Box::new(engine));
                *state = SessionState::Idle(owners);
                return Err(failure);
            }
        };
        let mut driver = match AdaptivePortDriver::new_with_coordinator_policy_retaining(
            port,
            context.control.deadline,
            Arc::new(owners.native.signal()),
            self.options.external_policy,
            self.options.policy,
        ) {
            Ok(driver) => driver,
            Err((port, error)) => {
                owners.engine = Some(Box::new(port.into_engine()));
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::debug("AdaptiveSearchOwner", &error));
            }
        };
        if let Some(observer) = &self.dispatch_observer {
            if let Err(error) = driver.set_collection_dispatch_observer(Arc::clone(observer)) {
                *state = SessionState::PortRejected { owners, driver };
                return Err(SearchSessionFailure::debug(
                    "AdaptiveCollectionObserver",
                    &error,
                ));
            }
        }
        if let Some(observer) = &self.collection_hooks.external_teacher {
            if let Err(error) = driver.set_external_teacher_observer(Arc::clone(observer)) {
                *state = SessionState::PortRejected { owners, driver };
                return Err(SearchSessionFailure::debug(
                    "AdaptiveTeacherObserver",
                    &error,
                ));
            }
        }
        let mut workers = Vec::with_capacity(owners.cpu.len());
        while let Some(owner) = owners.cpu.pop() {
            match CpuWorker::spawn_notified_retaining_with_capacity(
                owner.profile.owner,
                owner.engine,
                owners.native.signal(),
                self.limits.worker_commands_max,
            ) {
                Ok(worker) if worker.profile() == owner.profile => workers.push(worker),
                Ok(worker) => {
                    workers.push(worker);
                    return self.retain_spawn_failure(&mut state, owners, driver, workers, None);
                }
                Err((engine, error)) => {
                    owners.cpu.push(PersistentCpuOwner {
                        profile: owner.profile,
                        engine,
                    });
                    return self.retain_spawn_failure(
                        &mut state,
                        owners,
                        driver,
                        workers,
                        Some(error),
                    );
                }
            }
        }
        workers.reverse();
        let first_request_sequence = owners
            .continuation
            .as_ref()
            .map_or(1, |c| c.first_request_sequence());
        let registration = AdaptiveManagerRegistration {
            authority: context.authority,
            hard_deadline: context.control.deadline,
            token_budget: self.options.token_budget,
            policy: self.options.policy,
            first_request_sequence,
            native_cpu_node_limit: match self.researcher_class {
                SessionResearcherClass::Legacy => {
                    if self.rovent.is_some() {
                        self.options.max_cpu_nodes
                    } else {
                        context.control.max_simulations
                    }
                }
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                SessionResearcherClass::ProfiledExternal
                | SessionResearcherClass::ProfiledExternalV2 => 0,
            },
            endpoints: self.options.endpoints,
            controller: self.options.controller,
            device_digest: self.options.device_digest,
            cpu_device_digest: self.options.cpu_device_digest,
        };
        let researcher = match std::mem::replace(
            &mut owners.researcher_source,
            SessionResearcherSource::Legacy,
        ) {
            SessionResearcherSource::Legacy => SessionManagerSource::Legacy,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionResearcherSource::ProfiledExternal(source) => {
                let Some(original) = root_admission
                    .as_ref()
                    .and_then(|admission| admission.original.clone())
                else {
                    owners.researcher_source = SessionResearcherSource::ProfiledExternal(source);
                    *state = SessionState::PortRejected { owners, driver };
                    return Err(failure(
                        "RoventExternalManagerSource",
                        "original root capability is missing",
                    ));
                };
                match move_profiled_manager_source(source, original) {
                    Ok(external) => external,
                    Err((source, error)) => {
                        owners.researcher_source = source;
                        *state = SessionState::PortRejected { owners, driver };
                        return Err(error.unknown_if_unclosed());
                    }
                }
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            SessionResearcherSource::ProfiledExternalV2(source) => {
                let Some(original) = root_admission
                    .as_ref()
                    .and_then(|admission| admission.original_v2.clone())
                else {
                    owners.researcher_source = SessionResearcherSource::ProfiledExternalV2(source);
                    *state = SessionState::PortRejected { owners, driver };
                    return Err(failure(
                        "RoventExternalManagerSource",
                        "original /2 root capability is missing",
                    ));
                };
                match move_profiled_manager_source_v2(source, original) {
                    Ok(external) => external,
                    Err((source, error)) => {
                        owners.researcher_source = source;
                        *state = SessionState::PortRejected { owners, driver };
                        return Err(error.unknown_if_unclosed());
                    }
                }
            }
        };
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let external_joined = {
            let captured = match &researcher {
                SessionManagerSource::Legacy => Ok(None),
                SessionManagerSource::Profiled(external) => {
                    external.owner().frontier_root_source().and_then(|source| {
                        source
                            .map(|source| Some(OwnedSessionExternalJoined::V1(source)))
                            .ok_or(rz_search::cpu_checker::CheckerError::Invalid(
                                "actual external source is pending",
                            ))
                    })
                }
                SessionManagerSource::ProfiledV2(external) => external
                    .owner()
                    .frontier_root_source_v2()
                    .and_then(|source| {
                        source
                            .map(|source| Some(OwnedSessionExternalJoined::V2(source)))
                            .ok_or(rz_search::cpu_checker::CheckerError::Invalid(
                                "actual /2 external source is pending",
                            ))
                    }),
            };
            match captured {
                Ok(source) => source,
                Err(error) => {
                    match researcher {
                        SessionManagerSource::Profiled(external) => {
                            owners.researcher_source = return_profiled_manager_source(
                                external,
                                self.checker_metadata.clone(),
                            )
                        }
                        SessionManagerSource::ProfiledV2(external) => {
                            owners.researcher_source = return_profiled_manager_source_v2(
                                external,
                                self.checker_metadata.clone(),
                            )
                        }
                        SessionManagerSource::Legacy => {}
                    }
                    *state = SessionState::PortRejected { owners, driver };
                    return Err(
                        SearchSessionFailure::debug("RoventExternalJoinedSource", &error)
                            .unknown_if_unclosed(),
                    );
                }
            }
        };
        let external_source = {
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            {
                external_joined.as_ref().map(|joined| match joined {
                    OwnedSessionExternalJoined::V1(source) => SessionExternalJoined::V1(source),
                    OwnedSessionExternalJoined::V2(source) => SessionExternalJoined::V2(source),
                })
            }
            #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
            {
                None
            }
        };
        let sources = AdaptiveManagerSources {
            native: owners.native,
            own_workers: workers,
            external: owners.external,
            driver,
        };
        let admitted =
            self.admit_session_manager(registration, sources, researcher, owners.continuation);
        let mut manager = match admitted {
            Ok(manager) => manager,
            Err(SessionManagerRejection::Legacy(rejected)) => {
                let error =
                    SearchSessionFailure::debug("AdaptiveManagerAdmission", &rejected.error);
                *state = SessionState::AdmissionRejected(rejected);
                return Err(error);
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Err(SessionManagerRejection::Profiled(rejected)) => {
                let error = SearchSessionFailure::debug(
                    "ProfiledExternalManagerAdmission",
                    &rejected.error,
                );
                *state = SessionState::ProfiledAdmissionRejected(rejected);
                return Err(error);
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Err(SessionManagerRejection::ProfiledV2(rejected)) => {
                let error = SearchSessionFailure::debug(
                    "ProfiledExternalManagerAdmissionV2",
                    &rejected.error,
                );
                *state = SessionState::ProfiledAdmissionRejectedV2(rejected);
                return Err(error);
            }
        };
        *observation = AttemptObservation::PalsStarted;
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let mut board_collection_v2_started = false;
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let mut board_collection_v2_revoked = false;
        let admitted = catch_unwind(AssertUnwindSafe(|| {
            if let Some(profile) = &self.rovent {
                match self.researcher_class {
                    SessionResearcherClass::Legacy => {
                        manager.register_cleanup_deadline(context.shutdown_deadline)?
                    }
                    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                    SessionResearcherClass::ProfiledExternal
                    | SessionResearcherClass::ProfiledExternalV2 => {}
                }
                if manager.registered_cleanup_deadline() != Some(context.shutdown_deadline) {
                    return Err(AdaptiveManagerError::Invalid(
                        "Rovent original cleanup boundary differs",
                    ));
                }
                if let Some(clock) = &context.effective_clock {
                    manager.register_effective_clock(clock.clone())?;
                }
                self.attach_rovent_profile(
                    profile,
                    &mut manager,
                    context,
                    root_admission
                        .as_ref()
                        .ok_or(AdaptiveManagerError::Invalid(
                            "Rovent root admission absent",
                        ))?,
                    external_source,
                )?;
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                if self.rovent_collection_origin_v2.is_some() {
                    // Mark before invoking an external hook: it may acquire a
                    // teacher owner even if admission fails or unwinds.
                    board_collection_v2_started = true;
                    self.admit_board_collection_v2(
                        &mut manager,
                        root_admission.as_ref().ok_or(AdaptiveManagerError::Invalid(
                            "V2 board root admission absent",
                        ))?,
                        profile,
                    )?;
                }
            }
            #[cfg(feature = "pals-adaptive-diagnostics")]
            if let Some(diagnostic) = self.continuation_diagnostic.get() {
                manager.install_continuation_diagnostic(diagnostic.clone());
                manager
                    .driver_mut()
                    .install_role_consumption_diagnostic(diagnostic);
            }
            if let Some(observer) = &self.collection_hooks.root_admission {
                observer(&mut manager)?;
            }
            if self.rovent.is_some() {
                if !context.accepts()? {
                    manager.begin_root_quiesce(context.shutdown_deadline)?;
                    drain_root(&mut manager, context.shutdown_deadline)?;
                    return self.root_shutdown_receipt(&manager);
                }
                if let Err(error) = manager
                    .driver_mut()
                    .prepare_rovent_root_rules(self.clock.domain(), self.clock.now()?)
                {
                    if !self.rovent_controlled_scan_stop(&manager, &error)? {
                        return Err(error);
                    }
                }
            } else {
                manager.driver_mut().prepare_root_rules()?;
            }
            if manager.driver().root_is_terminal()? || self.rovent_should_quiesce(&manager)? {
                manager.begin_root_quiesce(context.shutdown_deadline)?;
                drain_root(&mut manager, context.shutdown_deadline)?;
                self.root_shutdown_receipt(&manager)
            } else {
                if context.soft_expired(Instant::now()) || !context.accepts().unwrap_or(false) {
                    manager.begin_root_quiesce(context.shutdown_deadline)?;
                    drain_root(&mut manager, context.shutdown_deadline)?;
                    return self.root_shutdown_receipt(&manager);
                }
                let input = manager.driver_mut().root_input()?;
                manager.prepare_root_ranking(input)?;
                let mut last_published = None;
                let mut last_semantic_snapshot = None;
                self.run_session_root_observed(
                    &mut manager,
                    context.shutdown_deadline,
                    &context.control.cancellation,
                    &mut |manager| {
                        if manager.phase() == ManagerPhase::Running
                            && (context.soft_expired(Instant::now())
                                || !context.accepts().unwrap_or(false)
                                || self.rovent_should_quiesce(manager)?)
                        {
                            manager.begin_root_quiesce(context.shutdown_deadline)?;
                        }
                        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                        if board_collection_v2_started {
                            if manager.phase() != ManagerPhase::Running {
                                if !board_collection_v2_revoked {
                                    if let Some(hooks) = &self.rovent_collection_hooks_v2 {
                                        hooks.revoke();
                                    }
                                    board_collection_v2_revoked = true;
                                }
                            } else if let Some(hooks) = &self.rovent_collection_hooks_v2 {
                                hooks.pump()?;
                            }
                        }
                        if let Some(pump) = &self.collection_hooks.cooperative {
                            pump()?;
                        }
                        let current = manager.driver().port().map(|port| port.counters());
                        if context.accepts().unwrap_or(false)
                            && current.is_some()
                            && current != last_semantic_snapshot
                        {
                            // Quiet notification ticks inspect only controls.
                            // A Search counter mutation triggers the bounded
                            // resolver; it is not a completed CPU frame claim.
                            last_semantic_snapshot = current;
                            // Only a coherent Search result can publish a move;
                            // callback ticks and partial CPU frames grant none.
                            if let Ok(result) = manager.driver().result(PalsCompletion::RoundLimit)
                            {
                                if let Some(movement) = result.best_move {
                                    if last_published != Some(movement)
                                        && root_legal.contains(&movement)
                                    {
                                        progress(movement);
                                        last_published = Some(movement);
                                    }
                                }
                            }
                        }
                        Ok(())
                    },
                )
            }
        }));
        let admitted = match admitted {
            Ok(result) => result,
            Err(payload) => {
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                if board_collection_v2_started {
                    self.rovent_collection_panic_v2.lock().unwrap_or_else(|p| p.into_inner()).push(payload);
                    if let Some(hooks) = &self.rovent_collection_hooks_v2 {
                        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| hooks.revoke())) {
                            // Both panic payloads can own resources. Retain a
                            // second one independently if revocation panics.
                            self.rovent_collection_panic_v2.lock().unwrap_or_else(|p| p.into_inner()).push(payload);
                        }
                    }
                }
                #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
                drop(payload);
                *state = SessionState::Manager(manager);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if board_collection_v2_started && !board_collection_v2_revoked {
            if let Some(hooks) = &self.rovent_collection_hooks_v2 {
                if let Err(payload) = catch_unwind(AssertUnwindSafe(|| hooks.revoke())) {
                    self.rovent_collection_panic_v2.lock().unwrap_or_else(|p| p.into_inner()).push(payload);
                    *state = SessionState::Manager(manager);
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
            }
        }
        if let Err(error) = admitted {
            let _cleanup = catch_unwind(AssertUnwindSafe(|| {
                let _ = manager.begin_root_quiesce(context.shutdown_deadline);
                let _ = drain_root(&mut manager, context.shutdown_deadline);
            }));
            let _ = self.observe_rovent_root(&manager, context, root_admission.as_ref());
            // Even cleanup panic retains the whole actual manager before any
            // caller-visible failure; a local owner must not unwind and drop.
            #[cfg(feature = "pals-adaptive-diagnostics")]
            if let Some(diagnostic) = self.continuation_diagnostic.get() {
                manager
                    .driver()
                    .copy_role_consumption_diagnostic(diagnostic, false);
            }
            *state = SessionState::Manager(manager);
            return Err(
                SearchSessionFailure::debug("AdaptiveRootExecution", &error).unknown_if_unclosed()
            );
        }
        let receipt = match self.root_shutdown_receipt(&manager) {
            Ok(receipt) => receipt,
            Err(_) => {
                *state = SessionState::Manager(manager);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        if !receipt.permits_root_reuse() || manager.phase() != ManagerPhase::Finished {
            // Preserve the first original failure before retaining the complete
            // manager. Its cause does not confirm a fence or relax quarantine.
            let failure = failed_root_closure(manager.failures().next());
            let _ = self.observe_rovent_root(&manager, context, root_admission.as_ref());
            *state = SessionState::Manager(manager);
            return Err(failure);
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if board_collection_v2_started {
            // Get the nonblocking destination while the fenced manager still
            // owns the original observer. A panic retains both owners.
            let sender = match catch_unwind(AssertUnwindSafe(|| {
                self.rovent_collection_hooks_v2.as_ref().expect("registered hook").handoff_sender()
            })) {
                Ok(sender) => sender,
                Err(payload) => {
                    self.rovent_collection_panic_v2.lock()
                        .unwrap_or_else(|p| p.into_inner()).push(payload);
                    *state = SessionState::Manager(manager);
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
            };
            let (admission, observer, settlement) = match manager.take_collection_observer_v2() {
                Ok(Some(original)) => original,
                _ => {
                    *state = SessionState::Manager(manager);
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
            };
            let handoff = Arc::new(Mutex::new(RoventBoardCollectionHandoffV2 {
                admission, observer, settlement, native_fence: receipt.clone(),
            }));
            *self.rovent_collection_handoff_v2.lock().unwrap_or_else(|p| p.into_inner()) =
                Some(Arc::clone(&handoff));
            let _delivery = sender.try_send(handoff);
            // Delivery or the native fence is not teacher retirement. The
            // same handoff remains retained even if the channel is full or
            // disconnected; neither condition permits a successful result.
            *state = SessionState::Manager(manager);
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        #[cfg(feature = "pals-adaptive-diagnostics")]
        if let Some(diagnostic) = self.continuation_diagnostic.get() {
            manager
                .driver()
                .copy_role_consumption_diagnostic(diagnostic, true);
        }
        let publication_capacity = match manager.driver().rovent_publication_capacity_closed() {
            Ok(closed) => closed,
            Err(error) => {
                *state = SessionState::Manager(manager);
                return Err(SearchSessionFailure::debug(
                    "RoventCapacityObservation",
                    &error,
                ));
            }
        };
        let completion = if context.cancellation.is_canceled()
            || context.control.cancellation.load(Ordering::Acquire)
        {
            PalsCompletion::Canceled
        } else if Instant::now() >= context.control.deadline {
            PalsCompletion::Deadline
        } else if manager.driver().root_is_terminal().unwrap_or(false) {
            PalsCompletion::Terminal
        } else if publication_capacity {
            PalsCompletion::Capacity
        } else if matches!(self.researcher_limit_observation(&manager), Some(true)) {
            PalsCompletion::CpuNodeLimit
        } else if manager.role_call_limit_reached() {
            PalsCompletion::RoleCallLimit
        } else {
            PalsCompletion::RoundLimit
        };
        let semantic = match catch_unwind(AssertUnwindSafe(|| manager.driver().result(completion)))
        {
            Ok(result) => result,
            Err(_) => {
                *state = SessionState::Manager(manager);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        if let Ok(result) = &semantic {
            *semantic_completion = Some(result.completion);
            *observation = AttemptObservation::Pals {
                counters: result.counters,
                root_coverage_observed: result.counters.root_scope_observation_complete,
            };
        }
        let collection =
            match catch_unwind(AssertUnwindSafe(|| -> Result<(), AdaptiveManagerError> {
                self.closure_observer
                    .as_ref()
                    .map_or(Ok(()), |observer| observer(&mut manager))?;
                if let Some(observer) = &self.collection_hooks.root_closure {
                    let completion = semantic
                        .as_ref()
                        .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))?
                        .completion;
                    observer(&mut manager, completion)?;
                }
                Ok(())
            })) {
                Ok(result) => result,
                Err(_) => {
                    *state = SessionState::Manager(manager);
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
            };
        let retained_failure = manager.failures().next().cloned();
        let rovent_receipt =
            match self.observe_rovent_root(&manager, context, root_admission.as_ref()) {
                Ok(receipt) => receipt,
                Err(error) => {
                    *state = SessionState::Manager(manager);
                    return Err(error.unknown_if_unclosed());
                }
            };
        if self.rovent.is_some() {
            match manager.driver_mut().retire_rovent_frontier() {
                Ok(Some(_)) => {}
                Ok(None) => {
                    *state = SessionState::Manager(manager);
                    return Err(failure(
                        "RoventFrontierRetirement",
                        "actual Rovent frontier is missing",
                    )
                    .unknown_if_unclosed());
                }
                Err(error) => {
                    *state = SessionState::Manager(manager);
                    return Err(
                        SearchSessionFailure::debug("RoventFrontierRetirement", &error)
                            .unknown_if_unclosed(),
                    );
                }
            }
        }
        let (mut owners, driver, receipt) =
            self.return_manager_owners(manager, &mut state, false)?;
        match self.restore_semantic_owner(driver, &receipt) {
            Ok(restored) => {
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                {
                    owners.profiled_retirement = restored
                        .retirement
                        .filter(|retirement| retirement.retained_questions() != 0);
                }
                owners.engine = Some(Box::new(restored.port.into_engine()));
            }
            Err((driver, _)) => {
                *state = SessionState::PortRejected { owners, driver };
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        }
        if let Err(error) = self.record_root_fence(&receipt) {
            *state = SessionState::Idle(owners);
            return Err(error.unknown_if_unclosed());
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let owners =
            self.persist_profiled_retirement(&mut state, owners, context.shutdown_deadline)?;
        *state = SessionState::Idle(owners);
        collection.map_err(|e| SearchSessionFailure::debug("AdaptiveCollectionClosure", &e))?;
        if let Some(error) = retained_failure {
            return Err(SearchSessionFailure::debug("AdaptiveRootFailure", &error));
        }
        let result = semantic.map_err(|e| SearchSessionFailure::debug("AdaptiveRootResult", &e))?;
        if result
            .best_move
            .is_some_and(|movement| !root_legal.contains(&movement))
        {
            return Err(failure(
                "RoventRootRestriction",
                "actual result escaped its original legal root domain",
            ));
        }
        if result.resolver_version != self.registration.resolver_version {
            return Err(failure(
                "AdaptiveResolverChanged",
                "actual result resolver differs from startup",
            ));
        }
        *self
            .pending_output
            .lock()
            .map_err(|_| failure("AdaptiveOutputOwner", "output mutex poisoned"))? =
            Some(PendingOutput {
                authority: context.authority,
                report: result.best_move,
                legal: root_legal,
            });
        if context.accepts().unwrap_or(false) {
            if let Some(movement) = result.best_move {
                progress(movement);
            }
        }
        initial_game.armed = false;
        let mate_strategy = rovent_receipt
            .as_ref()
            .and_then(|receipt| receipt.frontier())
            .and_then(|frontier| frontier.mate_strategy.clone())
            .filter(|strategy| result.best_move == Some(strategy.best_move()));
        Ok(SearchSessionReport {
            mate_strategy,
            best_move: result.best_move,
            work: match rovent_receipt {
                Some(receipt) => receipt.work()?,
                None => pals_work(result.counters),
            },
        })
    }

    fn retain_spawn_failure(
        &self,
        state: &mut SessionState<B>,
        owners: Box<IdleOwners<B>>,
        driver: AdaptivePortDriver<AdaptivePortModel>,
        workers: Vec<CpuWorker>,
        error: Option<rz_search::cpu_worker::CpuWorkerError>,
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        // Every started thread and every not-started CPU engine remains owned.
        // The driver retains these owners in a dedicated failed boundary.
        *state = SessionState::SpawnRejected {
            owners,
            driver,
            workers,
        };
        Err(error
            .as_ref()
            .map_or_else(SearchSessionFailure::physical_completion_unknown, |e| {
                SearchSessionFailure::debug("AdaptiveCpuSpawn", e).unknown_if_unclosed()
            }))
    }

    fn bind_actual_external(&self, owners: &IdleOwners<B>) -> Result<(), SearchSessionFailure> {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let SessionResearcherSource::ProfiledExternalV2(source) = &owners.researcher_source {
            let owner = source
                .running_owner()
                .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
            return self
                .checker_metadata
                .bind_profiled_external_owner(
                    owner,
                    self.clock.epoch(),
                    self.options.implementation,
                )
                .map_err(|error| {
                    SearchSessionFailure::debug("RoventProfiledMetadataOwnerV2", &error)
                });
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let SessionResearcherSource::ProfiledExternal(source) = &owners.researcher_source {
            let owner = source
                .running_owner()
                .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
            return self
                .checker_metadata
                .bind_profiled_external_owner(
                    owner,
                    self.clock.epoch(),
                    self.options.implementation,
                )
                .map_err(|error| {
                    SearchSessionFailure::debug("RoventProfiledMetadataOwner", &error)
                });
        }
        if !matches!(self.registration.identity, CheckerIdentity::ExternalUci(_))
            || self.external_metadata_bound.load(Ordering::Acquire)
        {
            return Ok(());
        }
        let external = owners.external.as_ref().ok_or_else(|| {
            failure(
                "AdaptiveExternalOwnerBinding",
                "actual external owner missing",
            )
        })?;
        self.checker_metadata
            .bind_external_owner(
                &external.worker,
                &external.config,
                external.owner,
                self.clock.epoch(),
                self.options.implementation,
            )
            .map_err(|e| SearchSessionFailure::debug("AdaptiveExternalOwnerBinding", &e))?;
        self.external_metadata_bound.store(true, Ordering::Release);
        Ok(())
    }

    fn reset_actual_cpu(
        &self,
        owners: &mut IdleOwners<B>,
        context: &SearchSessionContext,
        initial_game_bound: &mut bool,
    ) -> Result<(), SearchSessionFailure> {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let SessionResearcherSource::ProfiledExternalV2(source) = &owners.researcher_source {
            let owner = source
                .running_owner()
                .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
            let proof = owner.worker().current_reset_proof().ok_or_else(|| {
                failure(
                    "RoventProfiledResetV2",
                    "actual new-game reset proof is missing",
                )
            })?;
            self.checker_metadata
                .arm_profiled_external_reset(owner, &proof, context.control.deadline)
                .map_err(|error| SearchSessionFailure::debug("RoventProfiledResetV2", &error))?;
            *initial_game_bound = proof.is_initial();
            return Ok(());
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let SessionResearcherSource::ProfiledExternal(source) = &owners.researcher_source {
            let owner = source
                .running_owner()
                .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
            let proof = owner.worker().current_reset_proof().ok_or_else(|| {
                failure(
                    "RoventProfiledReset",
                    "actual new-game reset proof is missing",
                )
            })?;
            self.checker_metadata
                .arm_profiled_external_reset(owner, &proof, context.control.deadline)
                .map_err(|error| SearchSessionFailure::debug("RoventProfiledReset", &error))?;
            *initial_game_bound = proof.is_initial();
            return Ok(());
        }
        if let Some(external) = &owners.external {
            if !external
                .worker
                .clock_matches(self.clock.domain(), self.clock.runtime_clock().origin())
            {
                return Err(failure(
                    "AdaptiveExternalClock",
                    "actual external owner clock differs before game binding",
                ));
            }
            if !context.accepts().unwrap_or(false) {
                return Err(failure(
                    "AdaptiveExternalNewGame",
                    "root revoked before reset",
                ));
            }
            let initial = external
                .worker
                .bind_initial_game(
                    context.authority,
                    context.control.deadline,
                    &context.control.cancellation,
                )
                .map_err(|e| SearchSessionFailure::debug("AdaptiveExternalInitialGame", &e))?;
            if let Some(proof) = initial {
                *initial_game_bound = true;
                if matches!(self.registration.identity, CheckerIdentity::ExternalUci(_)) {
                    self.checker_metadata
                        .arm_external_initial_game(&external.worker, &external.config, proof)
                        .map_err(|e| {
                            SearchSessionFailure::debug("AdaptiveExternalInitialGameProof", &e)
                        })?;
                } else {
                    external
                        .worker
                        .consume_initial_game_binding(proof)
                        .map_err(|e| {
                            SearchSessionFailure::debug("AdaptiveExternalInitialGameProof", &e)
                        })?;
                }
            } else {
                external
                    .worker
                    .request_new_game(context.authority, context.control.deadline)
                    .map_err(|e| SearchSessionFailure::debug("AdaptiveExternalNewGame", &e))?;
                loop {
                    match external.worker.new_game_fence() {
                        Some(Ok(proof)) => {
                            if matches!(self.registration.identity, CheckerIdentity::ExternalUci(_))
                            {
                                self.checker_metadata
                                    .arm_external_new_game(&external.config, &proof)
                                    .map_err(|e| {
                                        SearchSessionFailure::debug(
                                            "AdaptiveExternalResetProof",
                                            &e,
                                        )
                                    })?;
                            }
                            break;
                        }
                        Some(Err(error)) => {
                            return Err(SearchSessionFailure::debug(
                                "AdaptiveExternalReset",
                                &error,
                            ));
                        }
                        None if context.accepts().unwrap_or(false) => {
                            std::thread::sleep(Duration::from_millis(1))
                        }
                        None => return Err(SearchSessionFailure::physical_completion_unknown()),
                    }
                }
            }
        }
        for (index, owner) in owners.cpu.iter_mut().enumerate() {
            if index == 0 && matches!(self.registration.identity, CheckerIdentity::Owned(_)) {
                self.checker_metadata
                    .reset_owned_and_arm(
                        &mut owner.engine,
                        context.control.deadline,
                        &context.control.cancellation,
                    )
                    .map_err(|e| SearchSessionFailure::debug("AdaptiveOwnCpuReset", &e))?;
            } else {
                owner.engine.clear();
            }
        }
        Ok(())
    }

    /// Explicit process finish after UCI has retired its output authority. The
    /// same native session receives Finish and is joined once; root receipts do
    /// not stand in for this actual physical release.
    pub fn finish_process(
        &self,
        until: Instant,
    ) -> Result<AdaptiveSessionProcessReceipt, SearchSessionFailure> {
        self.closed.store(true, Ordering::Release);
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let Some(hooks) = &self.rovent_collection_hooks_v2 {
            // A native process fence cannot attest the separately owned V2
            // teacher. Until a same-owner retirement handoff exists, do not
            // detach or discard the manager's original collection observer.
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| hooks.revoke())) {
                self.rovent_collection_panic_v2.lock()
                    .unwrap_or_else(|poison| poison.into_inner()).push(payload);
            }
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        let mut state = self
            .state
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        if let SessionState::Closed(receipt) = &*state {
            return Ok(receipt.clone());
        }
        let retained = std::mem::replace(&mut *state, SessionState::Moving);
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let retained = self.finish_retained_profiled_io(&mut state, retained, until)?;
        let retained = match retained {
            SessionState::Manager(mut manager) => {
                let root_reusable = manager.phase() == ManagerPhase::Finished
                    && self
                        .root_shutdown_receipt(&manager)
                        .is_ok_and(|r| r.permits_root_reuse());
                if !root_reusable {
                    let finish = catch_unwind(AssertUnwindSafe(|| {
                        manager.begin_shutdown(until)?;
                        self.run_session_process_close(&mut manager, until)
                    }));
                    if !matches!(finish, Ok(Ok(ref receipt)) if receipt.permits_archive_close()) {
                        *state = SessionState::Manager(manager);
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                }
                let (mut owners, driver, receipt) =
                    self.return_manager_owners(manager, &mut state, !root_reusable)?;
                if !root_reusable {
                    if let Some(external) = &owners.external {
                        match external.worker.shutdown_snapshot() {
                            Some(Ok(receipt)) => owners.external_finished = Some(receipt),
                            _ => {
                                *state = SessionState::PortRejected { owners, driver };
                                return Err(SearchSessionFailure::physical_completion_unknown());
                            }
                        }
                    }
                }
                match self.restore_semantic_owner(driver, &receipt) {
                    Ok(restored) => {
                        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                        {
                            owners.profiled_retirement = restored
                                .retirement
                                .filter(|retirement| retirement.retained_questions() != 0);
                        }
                        owners.engine = Some(Box::new(restored.port.into_engine()));
                    }
                    Err((driver, _)) => {
                        *state = SessionState::PortRejected { owners, driver };
                        return Err(SearchSessionFailure::physical_completion_unknown());
                    }
                }
                if let Err(error) = self.record_root_fence(&receipt) {
                    *state = SessionState::Idle(owners);
                    return Err(error.unknown_if_unclosed());
                }
                SessionState::Idle(owners)
            }
            other => other,
        };
        let mut owners = match retained {
            SessionState::Idle(owners) => owners,
            retained => {
                *state = retained;
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        if let Err(error) = self.finish_idle_physical_owners(&mut owners, until) {
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            let error = self.abort_engine_storage(error, &owners, until);
            *state = SessionState::Idle(owners);
            return Err(error);
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let mut owners = self.persist_profiled_retirement(&mut state, owners, until)?;
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let storage_finished = match self.finish_engine_storage(&mut owners, until) {
            Ok(result) => result,
            Err(error) => {
                *state = SessionState::Idle(owners);
                return Err(error.unknown_if_unclosed());
            }
        };
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        let storage_finished: Option<Option<SearchAuthority>> = None;
        let (last_authority, closed) = match storage_finished {
            Some(last_authority) => (last_authority, true),
            None => {
                let engine = owners.engine.as_mut().expect("quiescent Search owner");
                let last_authority = engine.last_adaptive_authority();
                let closed =
                    match catch_unwind(AssertUnwindSafe(|| engine.finish_followup_owners(until))) {
                        Ok(Ok(snapshot)) => snapshot.archive.as_ref().is_ok_and(|archive| {
                            archive.as_ref().is_none_or(|a| a.admission_closed)
                        }),
                        _ => false,
                    };
                (last_authority, closed)
            }
        };
        if !closed {
            *state = SessionState::Idle(owners);
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        let latest_root = match self.latest_root.lock() {
            Ok(receipt) => receipt.clone(),
            Err(_) => {
                *state = SessionState::Idle(owners);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let latest_profiled_root = match self.latest_profiled_root.lock() {
            Ok(receipt) => receipt.clone(),
            Err(_) => {
                *state = SessionState::Idle(owners);
                return Err(failure(
                    "ProfiledProcessReceipt",
                    "root receipt mutex poisoned",
                ));
            }
        };
        let receipt = AdaptiveSessionProcessReceipt {
            epoch: self.clock.epoch(),
            last_authority,
            native_owner_joined: owners.native_joined,
            native_physical_shutdown_confirmed: owners.native_finished == Some(true),
            external_shutdown: owners.external_finished,
            store_archive_closed: closed,
            latest_root,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            latest_profiled_root,
        };
        if !receipt.permits_process_close() {
            *state = SessionState::Idle(owners);
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        *state = SessionState::Closed(receipt.clone());
        Ok(receipt)
    }
}

impl<B> SearchSessionDriver for AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    fn supports_ponder_clock(&self) -> bool {
        #[cfg(feature = "pals-adaptive")]
        {
            self.rovent_root_policy.is_some() || self.rovent_root_policy_v2.is_some()
        }
        #[cfg(not(feature = "pals-adaptive"))]
        {
            false
        }
    }
    fn kind(&self) -> SearchKind {
        if self.rovent.is_some() {
            SearchKind::Rovent
        } else {
            SearchKind::Pals
        }
    }
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        if self.rovent.is_some() {
            crate::bridge::NodeBudgetSemantics::RoventPositionVisits
        } else {
            crate::bridge::NodeBudgetSemantics::LegacySimulations
        }
    }
    fn validate_go(
        &self,
        position: &Position,
        go: &crate::parser::GoLimits,
    ) -> Result<(), ContractError> {
        #[cfg(feature = "pals-adaptive")]
        if let Some(registered) = &self.rovent_root_policy {
            let profile = self
                .rovent
                .as_ref()
                .ok_or_else(|| poisoned("Rovent startup profile absent"))?;
            crate::rovent_root_admission::preadmission::validate_go(
                position,
                go,
                &registered.lock,
                registered.policy,
                profile.maximum_goal_plies,
            )
            .map_err(|error| {
                ContractError::new(ErrorCode::InvalidInput, Stage::Admission, error.detail)
            })?;
        }
        #[cfg(feature = "pals-adaptive")]
        if let Some(registered) = &self.rovent_root_policy_v2 {
            let profile = self
                .rovent
                .as_ref()
                .ok_or_else(|| poisoned("Rovent startup profile absent"))?;
            let lock = registered
                .owner
                .v2()
                .ok_or_else(|| poisoned("Rovent /2 startup lock absent"))?;
            crate::rovent_root_admission::preadmission::validate_go_v2(
                position,
                go,
                lock,
                registered.policy,
                profile.maximum_goal_plies,
            )
            .map_err(|error| {
                ContractError::new(ErrorCode::InvalidInput, Stage::Admission, error.detail)
            })?;
        }
        #[cfg(not(feature = "pals-adaptive"))]
        let _ = (position, go);
        Ok(())
    }
    fn search_goal_capability(&self) -> Option<std::num::NonZeroU16> {
        self.rovent
            .as_ref()
            .map(|profile| profile.maximum_goal_plies)
    }
    fn finish_session(
        &self,
        until: Instant,
    ) -> Result<Option<crate::search_driver::DriverPhysicalCompletion>, SearchSessionFailure> {
        self.finish_rovent_session(until)
    }
    fn implementation(&self) -> Digest {
        self.options.implementation
    }
    fn identity(&self) -> EngineIdentity {
        self.options.identity.clone()
    }
    fn work_receipt(&self) -> Result<Option<ProcessSearchWorkReceipt>, ContractError> {
        if self.rovent.is_some() {
            Ok(None)
        } else {
            self.work.snapshot().map(Some)
        }
    }
    fn pals_checker_registration(&self) -> Option<PalsCheckerRegistration> {
        self.rovent.is_none().then(|| self.registration.clone())
    }
    fn checker_started(&self) -> Option<bool> {
        match self.state.try_lock() {
            Ok(state) => {
                match &*state {
                    SessionState::Idle(owners) => Some(owners.external.as_ref().is_none_or(|e| {
                        e.worker.startup_report().is_some_and(|r| r.result.is_ok())
                    })),
                    // This getter never promotes manager admission to an
                    // actual external startup observation.
                    SessionState::Manager(_) => None,
                    _ => None,
                }
            }
            Err(_) => None,
        }
    }
    fn checker_startup_uci(&self) -> Result<Option<ExternalUciIdentity>, SearchSessionFailure> {
        let state = self
            .state
            .try_lock()
            .map_err(|_| failure("AdaptiveCheckerBusy", "actual owner is busy"))?;
        if let SessionState::Idle(owners) = &*state {
            return owners
                .external
                .as_ref()
                .and_then(|e| e.worker.startup_report())
                .map_or(Ok(None), |r| {
                    r.result
                        .map(Some)
                        .map_err(|e| SearchSessionFailure::debug("AdaptiveCheckerStartup", &e))
                });
        }
        Ok(None)
    }
    fn start_checker(
        &self,
        deadline: Instant,
        cancel: Arc<AtomicBool>,
    ) -> Result<Option<bool>, SearchSessionFailure> {
        loop {
            let started = self.checker_started();
            if started == Some(true) {
                let state = self
                    .state
                    .try_lock()
                    .map_err(|_| failure("AdaptiveCheckerBusy", "actual startup owner busy"))?;
                if let SessionState::Idle(owners) = &*state {
                    self.bind_actual_external(owners)?;
                }
                return Ok(started);
            }
            // Preserve an actual startup failure instead of spending the rest
            // of the deadline waiting for an impossible ready response.
            self.checker_startup_uci()?;
            if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
                return Err(failure(
                    "AdaptiveCheckerStartup",
                    "actual startup not ready before original deadline",
                ));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    fn finish_checker(
        &self,
        deadline: Instant,
    ) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        self.finish_process(deadline)
            .map(|receipt| receipt.external_shutdown())
    }
    fn checker_shutdown(&self) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        let state = self
            .state
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        Ok(match &*state {
            SessionState::Closed(receipt) => receipt.external_shutdown,
            _ => None,
        })
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(poisoned("adaptive process already closed"));
        }
        self.reset_requests
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                old.checked_add(1)
            })
            .map(|_| ())
            .map_err(|_| poisoned("adaptive reset request sequence exhausted"))
    }
    fn accept_uci_output(
        &self,
        authority: SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        let mut pending = self
            .pending_output
            .lock()
            .map_err(|_| poisoned("adaptive output authority"))?;
        let registered = pending
            .as_ref()
            .ok_or_else(|| poisoned("no actual adaptive root output"))?;
        validate_output(registered, authority, bestmove, from_report)?;
        let mut observations = self
            .accepted_outputs
            .lock()
            .map_err(|_| poisoned("adaptive output receipt"))?;
        if observations.len() == 64 {
            observations.pop_front();
        }
        observations.push_back(AdaptiveUciOutputObservation {
            authority,
            bestmove: bestmove.to_owned(),
            from_report,
        });
        *pending = None;
        Ok(())
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        let legacy_journal = self.rovent.is_none();
        if legacy_journal {
            if let Err(error) = self.work.begin() {
                let failure = SearchSessionFailure::debug("AdaptiveWorkJournal", &error);
                if let Ok(mut journal) = self.completion.lock() {
                    journal.record(context.authority, None, Err(&failure));
                }
                return Err(failure);
            }
        }
        let mut observation = AttemptObservation::NoWork;
        let mut semantic_completion = None;
        let result = self.execute(
            position,
            context,
            progress,
            &mut observation,
            &mut semantic_completion,
        );
        let recorded = if legacy_journal {
            self.work.finish(context, &result, observation)
        } else {
            Ok(())
        };
        let returned = match (result, recorded) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(SearchSessionFailure::debug("AdaptiveWorkJournal", &error)),
            (Ok(result), Ok(())) => Ok(result),
        };
        // Diagnostics cannot replace the original return or make an Err into
        // bounded success. Poison is exposed by the getter, never as a count 0.
        if let Ok(mut journal) = self.completion.lock() {
            journal.record(
                context.authority,
                semantic_completion,
                returned.as_ref().map(|_| ()),
            );
        }
        returned
    }
}

impl<B> Drop for AdaptiveSessionDriver<B>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|poison| poison.into_inner());
        let retained = std::mem::replace(state, SessionState::Moving);
        let physically_closed = matches!(retained, SessionState::Closed(_));
        if physically_closed {
            drop(retained);
        } else {
            // At most one complete process owner per driver is retained. This
            // is failure quarantine, not successful shutdown or a cache. In
            // particular, dropping an Arc must not detach the native service
            // or block while dropping live CPU workers and free their Store.
            let _quarantined_owner: &'static mut SessionState<B> = Box::leak(Box::new(retained));
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if !physically_closed {
            // The manager quarantine alone does not own the external hook,
            // rejected admission or callback panic payload. Keep those same
            // owners alongside it when teacher completion is unknown.
            let collection_owners = (
                self.rovent_collection_origin_v2.take(),
                self.rovent_collection_hooks_v2.take(),
                self.rovent_collection_failure_v2.get_mut()
                    .unwrap_or_else(|poison| poison.into_inner()).take(),
                self.rovent_collection_handoff_v2.get_mut()
                    .unwrap_or_else(|poison| poison.into_inner()).take(),
                std::mem::take(self.rovent_collection_panic_v2.get_mut()
                    .unwrap_or_else(|poison| poison.into_inner())),
            );
            std::mem::forget(collection_owners);
            let storage = self
                .storage
                .get_mut()
                .unwrap_or_else(|poison| poison.into_inner());
            if let Some(original) = storage.take() {
                std::mem::forget(original);
            }
        }
    }
}

impl SearchSessionFailure {
    fn unknown_if_unclosed(mut self) -> Self {
        self.physical_completion = crate::search_driver::DriverPhysicalCompletion::Unknown;
        self
    }
}
// Debug formatting uses the existing bounded 512-byte writer, not an
// unbounded format! allocation. The original truncation suffix is retained;
// the JSON observation applies its own 512-byte field limit separately.
// A cause projection never certifies completion.
#[cfg(any(test, all(feature = "pals-adaptive", feature = "onnx-cpu")))]
fn retained_cause(code: &'static str, cause: &dyn std::fmt::Debug) -> SearchSessionFailure {
    SearchSessionFailure::debug(code, cause).unknown_if_unclosed()
}
#[cfg(any(test, all(feature = "pals-adaptive", feature = "onnx-cpu")))]
fn retained_archive_cause(
    cause: &rz_search::pals::engine::adaptive_port::archive_worker::ArchiveQuarantineReason,
) -> SearchSessionFailure {
    use rz_search::pals::engine::adaptive_port::archive_worker::ArchiveQuarantineReason;
    let code = match cause {
        ArchiveQuarantineReason::Operation(_) => "ArchiveQuarantineOperation",
        ArchiveQuarantineReason::Panic => "ArchiveQuarantinePanic",
        ArchiveQuarantineReason::Canceled => "ArchiveQuarantineCanceled",
        ArchiveQuarantineReason::Deadline => "ArchiveQuarantineDeadline",
        ArchiveQuarantineReason::InvalidReturn(_) => "ArchiveQuarantineInvalidReturn",
    };
    retained_cause(code, cause)
}
fn failed_root_closure(first: Option<&AdaptiveManagerError>) -> SearchSessionFailure {
    first.map_or_else(SearchSessionFailure::physical_completion_unknown, |cause| {
        SearchSessionFailure::debug("AdaptiveRootClosure", cause).unknown_if_unclosed()
    })
}
fn drain_root<B>(
    manager: &mut SessionManager<B>,
    until: Instant,
) -> Result<(), AdaptiveManagerError>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    while manager.phase() != ManagerPhase::Finished {
        manager.tick()?;
        if Instant::now() >= until {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}
fn join_archive_lifecycle<B>(
    state: &mut SessionState<B>,
    owners: Box<IdleOwners<B>>,
    mut worker: ArchiveWorker<AdaptivePortModel>,
    until: Instant,
) -> Result<Box<IdleOwners<B>>, SearchSessionFailure>
where
    B: TicketNativeBackend<ReviewView, AdaptivePortQuestion>,
    B::Output: ProposerRankingOutput,
{
    worker.request_shutdown();
    if owners.unexpected_archive_owner.is_some() {
        // Joining the I/O worker cannot turn a retained owner-kind failure
        // into an idle legacy Engine or a successful process-close receipt.
        *state = SessionState::ArchiveJoinedPending { owners, worker };
        return Err(SearchSessionFailure::physical_completion_unknown());
    }
    loop {
        match worker.try_join() {
            Ok(Some(receipt))
                if receipt.worker_joined && receipt.owners_returned && receipt.admission_closed =>
            {
                return Ok(owners);
            }
            Ok(None) if Instant::now() < until => std::thread::sleep(Duration::from_millis(1)),
            _ => {
                *state = SessionState::ArchiveJoinedPending { owners, worker };
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        }
    }
}

fn legacy_archive_engine(
    owner: ArchiveOwner<AdaptivePortModel>,
) -> Result<Box<PalsEngine<AdaptivePortModel>>, Box<ArchiveOwner<AdaptivePortModel>>> {
    match owner {
        ArchiveOwner::Engine(engine) => Ok(Box::new(engine)),
        ArchiveOwner::Port(port) => Ok(Box::new(port.into_engine())),
        owner @ ArchiveOwner::Idle(_) => Err(Box::new(owner)),
    }
}
fn owned_declarations_match(
    actual: Option<rz_search::cpu_checker::OwnedCheckerDescriptor>,
    declared: Option<&rz_search::cpu_checker::OwnedCheckerDescriptor>,
) -> bool {
    match (actual, declared) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.config.profile == b.config.profile
                && a.config.tt_entries == b.config.tt_entries
                && a.config.max_depth == b.config.max_depth
                && a.config.quiescence_ply == b.config.quiescence_ply
                && a.search_identity == b.search_identity
                && a.search_conditions == b.search_conditions
                && a.capabilities == b.capabilities
        }
        _ => false,
    }
}
fn pals_work(counters: PalsCounters) -> DriverWork {
    DriverWork::Pals {
        rounds: counters.rounds,
        cpu_nodes: counters.cpu_nodes,
        completed_tasks: counters.completed_cpu_tasks,
        consumed_role_outputs: counters.consumed_role_outputs,
        retained_situations: counters.retained_situations,
    }
}
fn failure(code: &'static str, detail: &'static str) -> SearchSessionFailure {
    SearchSessionFailure {
        physical_completion: crate::search_driver::DriverPhysicalCompletion::Confirmed,
        code,
        detail: detail.into(),
    }
}
fn poisoned(detail: &'static str) -> ContractError {
    ContractError::new(ErrorCode::BackendFailure, Stage::Output, detail)
}
fn validate_output(
    pending: &PendingOutput,
    authority: SearchAuthority,
    bestmove: &str,
    from_report: bool,
) -> Result<(), ContractError> {
    if pending.authority != authority {
        return Err(poisoned("stale adaptive UCI output authority"));
    }
    let movement = if bestmove == "0000" {
        None
    } else {
        Some(
            bestmove
                .parse::<BoardMove>()
                .map_err(|_| poisoned("malformed adaptive bestmove"))?,
        )
    };
    if movement.is_some_and(|m| !pending.legal.contains(&m))
        || (movement.is_none() && !pending.legal.is_empty())
        || (from_report && movement != pending.report)
    {
        return Err(poisoned(
            "adaptive output does not match actual legal root/report",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "session_driver/lifecycle_tests.rs"]
mod lifecycle_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::{GameGeneration, RootGeneration};
    fn authority(root: u64) -> SearchAuthority {
        SearchAuthority {
            epoch: ProcessEpoch(3),
            game: GameGeneration(2),
            root: RootGeneration(root),
            implementation: Digest([4; 32]),
        }
    }
    #[test]
    fn unexpected_idle_archive_return_retains_original_pc_owner_and_store() {
        use rz_contracts::PrecisionProfile;
        use rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace;
        use rz_search::pals::engine::{PalsSemanticConfig, PcSemanticOwner};
        use rz_search::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
        use rz_search::pals::value::{MODEL_WDL_VALUE_SEMANTICS, ModelValueIdentity};

        let value = ModelValueIdentity {
            semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
            model: "unexpected-idle-owner-fixture".into(),
            encoding: "metadata-only-test".into(),
            precision: "fp32".into(),
            model_epoch: [9; 32],
        };
        let namespace = AdaptiveModelNamespace::new(
            value.clone(),
            Digest([9; 32]),
            Digest([10; 32]),
            PrecisionProfile::Fp32,
            1,
        )
        .unwrap();
        let model =
            AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value), namespace)
                .unwrap();
        let owner = Box::new(
            PcSemanticOwner::new_selected(
                PalsSemanticConfig {
                    max_nodes: 257,
                    max_records: 4,
                    ..Default::default()
                },
                model,
                ReviewSelectionPolicy::default(),
                ReviewIndexLimits::default(),
            )
            .unwrap(),
        );
        let original_owner = std::ptr::from_ref(owner.as_ref());
        let original_store = owner.stores().read_owner();
        let idle = owner
            .into_idle_archive()
            .unwrap_or_else(|_| panic!("fresh original owner must be logically idle"));
        let retained = match legacy_archive_engine(ArchiveOwner::Idle(idle)) {
            Ok(_) => panic!("Idle cannot become a legacy Engine or Port"),
            Err(retained) => retained,
        };
        let ArchiveOwner::Idle(idle) = *retained else {
            panic!("the unexpected original owner kind must remain Idle")
        };
        assert!(original_store.matches(idle.stores()));
        let restored = idle
            .into_pc()
            .unwrap_or_else(|_| panic!("the original PC owner kind must be preserved"));
        assert_eq!(std::ptr::from_ref(restored.as_ref()), original_owner);
        assert!(original_store.matches(restored.stores()));
    }
    #[test]
    fn uci_output_checks_exact_root_and_separates_report_from_legal_fallback() {
        let legal = Position::startpos().legal_moves();
        let pending = PendingOutput {
            authority: authority(7),
            report: Some("e2e4".parse().unwrap()),
            legal,
        };
        assert!(validate_output(&pending, authority(7), "e2e4", true).is_ok());
        assert!(validate_output(&pending, authority(8), "e2e4", true).is_err());
        assert!(validate_output(&pending, authority(7), "d2d4", true).is_err());
        assert!(validate_output(&pending, authority(7), "d2d4", false).is_ok());
        assert!(validate_output(&pending, authority(7), "e2e5", false).is_err());
        assert!(validate_output(&pending, authority(7), "0000", false).is_err());
    }
    #[test]
    fn retained_manager_cause_is_bounded_and_cannot_confirm_physical_completion() {
        let cause = AdaptiveManagerError::Semantic("원인".repeat(1024));
        let observed = retained_cause("RetainedManagerEarliestFailure", &cause);
        assert_eq!(observed.code, "RetainedManagerEarliestFailure");
        assert!(observed.detail.starts_with("Semantic("));
        assert!(observed.detail.len() <= 512 + " [cause truncated]".len());
        assert!(observed.detail.ends_with(" [cause truncated]"));
        assert_eq!(
            observed.physical_completion,
            crate::search_driver::DriverPhysicalCompletion::Unknown
        );
        assert!(matches!(cause, AdaptiveManagerError::Semantic(ref value) if value.len() > 512));
    }
    #[test]
    fn archive_cause_keeps_operation_failure_distinct_from_deadline_and_cancel() {
        use rz_search::pals::engine::adaptive_port::archive_worker::ArchiveQuarantineReason;
        let operation =
            ArchiveQuarantineReason::Operation(rz_search::pals::engine::PalsError::Capacity);
        let observed = retained_archive_cause(&operation);
        assert_eq!(observed.code, "ArchiveQuarantineOperation");
        assert!(observed.detail.contains("Capacity"));
        assert_eq!(
            retained_archive_cause(&ArchiveQuarantineReason::Deadline).code,
            "ArchiveQuarantineDeadline"
        );
        assert_eq!(
            retained_archive_cause(&ArchiveQuarantineReason::Canceled).code,
            "ArchiveQuarantineCanceled"
        );
        assert_eq!(
            observed.physical_completion,
            crate::search_driver::DriverPhysicalCompletion::Unknown
        );
        assert!(matches!(operation, ArchiveQuarantineReason::Operation(_)));
    }
    #[test]
    fn process_receipt_requires_join_and_physical_finish_independently() {
        let mut receipt = AdaptiveSessionProcessReceipt {
            epoch: ProcessEpoch(3),
            last_authority: None,
            native_owner_joined: false,
            native_physical_shutdown_confirmed: false,
            external_shutdown: None,
            store_archive_closed: true,
            latest_root: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            latest_profiled_root: None,
        };
        assert!(!receipt.permits_process_close());
        receipt.native_owner_joined = true;
        assert!(!receipt.permits_process_close());
        receipt.native_physical_shutdown_confirmed = true;
        assert!(receipt.permits_process_close());
        receipt.store_archive_closed = false;
        assert!(!receipt.permits_process_close());
    }
}
