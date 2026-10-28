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
    AdaptiveManagerError, ManagerPhase, PcPortDriver, RoleOnlyExecutionManager,
    RoleOnlyManagerRegistration, RoleOnlyManagerSources, RoleOnlyShutdownReceipt,
    RoventCollectionRootAdmission, RoventNativeCollectionOrigin,
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
struct SharedNnuecOwner {
    loaded: LoadedRoventNnuecSource,
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
        let build =
            (|| -> Result<_, SearchSessionFailure> {
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
                if !matches!(
                    configuration.fields().execution,
                    ExecutionConfiguration::RolesOnly {
                        shared_nnuec_readout: None,
                        ..
                    }
                ) || sources.lock().manifest.engine.nnuec.is_some()
                    || sources.collection_preload_sources().is_some()
                    || prepared.snapshot().mode.has_native_r()
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
                let original_owner = rz_experiments::RoventOriginalInputLock::from_v2(Arc::clone(
                    sources.lock_owner(),
                ))
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
                let owner =
                    PcSemanticOwner::new_selected_rovent(
                        configuration.fields().semantic.actual().map_err(|e| {
                            SearchSessionFailure::debug("RoventPcConfiguration", &e)
                        })?,
                        model,
                        configuration.fields().review.actual().map_err(|e| {
                            SearchSessionFailure::debug("RoventPcConfiguration", &e)
                        })?,
                        configuration.fields().index.actual().map_err(|e| {
                            SearchSessionFailure::debug("RoventPcConfiguration", &e)
                        })?,
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
                    original_owner,
                    owner,
                    implementation,
                    selective_policy,
                    frontier_limits,
                ))
            })();
        let (
            configuration,
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
            shared_nnuec: None,
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
                loaded,
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
    pub fn latest_initial_v2(
        &self,
    ) -> Result<Option<Arc<RoventPcInitialReceiptV2>>, ContractError> {
        self.latest_initial_v2
            .lock()
            .map(|value| value.clone())
            .map_err(|_| poisoned())
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
        if !matches!(run, Ok(Ok(()))) {
            *state = State::Manager(manager);
            self.closed.store(true, Ordering::Release);
            return Err(match run {
                Ok(Err(error)) => {
                    let mut failure = SearchSessionFailure::debug("RoventPcRoot", &error);
                    failure.physical_completion = DriverPhysicalCompletion::Unknown;
                    failure
                }
                _ => physical_unknown_at(
                    "RoventPcRootExecution",
                    "root closure panicked; original manager retained",
                ),
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
