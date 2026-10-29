//! Actual native P/C/R NNUEC session, with deferred first C/TT/worker ownership.
//!
//! The loaded model and original source descriptors survive every root. A
//! preparing semantic owner is not an R engine: only the first admitted C can
//! create the session's preparation actor and materialize its real researcher.
//! Unknown transfers retain all original owners rather than replacing them.

#[cfg(feature = "pals-adaptive-diagnostics")]
pub(crate) mod diagnostics;
mod initial_cleanup;
#[cfg(test)]
pub(in crate::rovent_bootstrap) mod pre_materialization_tests;
mod retirement;
mod root_run;

use super::configuration::{
    ExecutionConfiguration, ResearchValueConfiguration, StartupConfigurationDocument,
};
use super::launch::storage::RoventStorageOwnedRuntimeRegistration;
use super::nnuec_preparing::{
    InitialNnuecFailure, InitialNnuecPhase, InitialNnuecPreparing, InitialNnuecReady,
    InitialNnuecStep,
};
use super::nnuec_source::LoadedRoventNnuecSource;
use super::product::SharedRoventStorageRegistration;
use super::storage_lifecycle::{SessionStorageLifecycle, StorageSemanticOwner};
use super::{
    PreparedRoventNative, PreparedRoventNativeV2, RoventNativeBackend, RoventNativeService,
    RoventNativeStartupSnapshot, RoventPinnedNativeSources,
};
use crate::EngineIdentity;
use crate::engine::{OwnerRegistry, ProcessClock};
use crate::pals_adaptive::AdaptiveSearchDriver as _;
use crate::pals_adaptive::{
    AdaptiveExecutionLimits, AdaptiveExecutionManager, AdaptiveManagerCounters,
    AdaptiveManagerError, AdaptiveManagerRegistration, AdaptiveManagerSources,
    AdaptiveNnuecRetiredOwner, AdaptivePortDriver, AdaptiveRegisteredLedgerSources,
    AdaptiveSessionCollectionHooks, AdaptiveShutdownReceipt, ManagerPhase,
    AdaptiveCollectionObserverBoxV2,
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
    DriverPhysicalCompletion, DriverWork, PalsCheckerRegistration, SearchKind,
    SearchSessionContext, SearchSessionDriver, SearchSessionFailure, SearchSessionReport,
};
use initial_cleanup::{
    NnuecInitialCleaned, NnuecInitialCleanup, NnuecInitialCleanupObservations,
    NnuecInitialCleanupStep,
};
use retirement::{NnuecRootRetirement, NnuecRootRetirementPins, NnuecRootRetirementStep};
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::CpuWorkerId;
use rz_contracts::{ContractError, Digest};
use rz_eval::error::{BackendError, FailureKind};
use rz_position::{BoardMove, Position};
use rz_runtime::adaptive::{ControllerConfig, CoordinatorPolicy};
use rz_runtime::contracts::ContractClock;
use rz_search::cpu_value::nnuec::v2::retirement::NnuecCacheReleaseAck;
use rz_search::cpu_value::nnuec::v2::review::{PreparationActor, PreparationError};
use rz_search::cpu_worker::{
    CpuCreditLedger, CpuWorker, CpuWorkerError, DeferredNnuecCpuPlan, DeferredNnuecRSource,
    NnuecAuthorityRevocation,
};
use rz_search::external_cpu::{CpuEndpointSelection, ExternalSlicePolicy};
use rz_search::pals::engine::adaptive_port::{
    AdaptiveEnginePort, AdaptivePortModel, AdaptivePortQuestion, AdaptiveRulesOwnerIssuer,
};
use rz_search::pals::engine::preparing_root::{
    InitialResearcherReleaseReceipt, ResearcherPreparingError, ResearcherPreparingOwner,
    ResearcherRootStartup, RetiredPreparingRoot,
};
use rz_search::pals::engine::{
    PalsCompletion, PalsConfig, PalsLimits, ResearcherSemanticProfile, ResolverPolicy,
};
use rz_search::pals::rovent_engine::RootMoveDomain;
use rz_search::pals::rovent_frontier::{
    FrontierError, FrontierGoalStatus, FrontierLimits, FrontierSnapshot, FrontierStop,
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

type Backend = RoventNativeBackend<AdaptivePortQuestion>;
type Manager =
    AdaptiveExecutionManager<AdaptivePortQuestion, Backend, AdaptivePortDriver<AdaptivePortModel>>;
type SemanticOwner = ResearcherPreparingOwner<AdaptivePortModel>;
pub type RoventNnuecCollectionHooks = AdaptiveSessionCollectionHooks<Backend>;

/// Original NNUEC collection objects handed off after the native root fence.
/// The handoff does not attest retirement of a separate teacher process.
pub struct RoventNnuecCollectionHandoffV2 {
    admission: RoventCollectionRootAdmissionV2,
    observer: AdaptiveCollectionObserverBoxV2<AdaptivePortQuestion, VirtualReviewTicketOutput>,
    settlement: Result<(), crate::pals_adaptive::AdaptiveCollectionError>,
    native_fence: AdaptiveShutdownReceipt,
    collection_failures: Box<[crate::pals_adaptive::AdaptiveCollectionError]>,
    teacher_physical_ack: Option<crate::pals_adaptive::RoventTeacherScopePhysicalAckV2>,
    teacher_aggregate_ack: Option<crate::rovent_external_launch::RoventTeacherScopeAggregateAckV2>,
}

impl RoventNnuecCollectionHandoffV2 {
    /// Collection diagnostics do not imply a failure of the native search.
    pub fn collection_failures(&self) -> &[crate::pals_adaptive::AdaptiveCollectionError] {
        &self.collection_failures
    }
    /// This attests physical and worker cleanup only. It is not teacher or
    /// corpus success; duplicate/foreign evidence returns its original owner.
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

    pub fn teacher_aggregate_ack(&self) -> Option<&crate::rovent_external_launch::RoventTeacherScopeAggregateAckV2> {
        self.teacher_aggregate_ack.as_ref()
    }

    /// Keep the original acknowledgement; duplicate or different-root
    /// evidence is rejected intact. This does not certify worker join.
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

    pub fn teacher_physical_ack(&self) -> Option<&crate::pals_adaptive::RoventTeacherScopePhysicalAckV2> {
        self.teacher_physical_ack.as_ref()
    }

    pub fn parts_mut(&mut self) -> (
        &RoventCollectionRootAdmissionV2,
        &mut AdaptiveCollectionObserverBoxV2<AdaptivePortQuestion, VirtualReviewTicketOutput>,
        &Result<(), crate::pals_adaptive::AdaptiveCollectionError>,
        &AdaptiveShutdownReceipt,
    ) {
        (&self.admission, &mut self.observer, &self.settlement, &self.native_fence)
    }

    pub fn into_parts(self) -> (
        RoventCollectionRootAdmissionV2,
        AdaptiveCollectionObserverBoxV2<AdaptivePortQuestion, VirtualReviewTicketOutput>,
        Result<(), crate::pals_adaptive::AdaptiveCollectionError>,
        AdaptiveShutdownReceipt,
        Box<[crate::pals_adaptive::AdaptiveCollectionError]>,
        Option<crate::pals_adaptive::RoventTeacherScopePhysicalAckV2>,
        Option<crate::rovent_external_launch::RoventTeacherScopeAggregateAckV2>,
    ) {
        (self.admission, self.observer, self.settlement, self.native_fence, self.collection_failures, self.teacher_physical_ack, self.teacher_aggregate_ack)
    }
}

/// /2 collection is an independent, live owner. The hook keeps its teacher
/// and publication custody. `revoke` is nonblocking, idempotent and acts only
/// on this owner's actual root and teacher. Callbacks prove no physical fence.
/// This seam does not authorize collection until the /2 corpus and teacher
/// lifecycle consumers can retain and drain the same owners.
pub trait RoventNnuecCollectionHooksV2: Send + Sync {
    /// Register the actual startup control before work can be spawned.
    fn register_startup_control(&self, control: &super::RoventStartupControl) -> Result<(), AdaptiveManagerError>;
    fn admit(
        &self,
        port: &mut RoventNnuecCollectionAdmissionPortV2<'_>,
        admission: RoventCollectionRootAdmissionV2,
    ) -> Result<(), RoventNnuecCollectionAdmissionFailureV2>;
    fn pump(&self) -> Result<(), AdaptiveManagerError>;
    fn revoke(&self);
    /// A bounded nonblocking handoff; delivery is not teacher retirement.
    fn handoff_sender(&self) -> std::sync::mpsc::SyncSender<Arc<Mutex<RoventNnuecCollectionHandoffV2>>>;

}

/// Failure custody returned by the /2 hook. This must contain the original
/// admission and any observer/teacher owners acquired during the attempt;
/// the session retains the whole value when admission fails.
pub struct RoventNnuecCollectionAdmissionFailureV2 {
    pub error: AdaptiveManagerError,
    pub admission: RoventCollectionRootAdmissionV2,
    pub owners: Box<dyn Any + Send>,
}

/// Narrow admission access to the actual session manager. This borrowed port
/// cannot manufacture admission or worker authority, and cannot escape a root.
pub struct RoventNnuecCollectionAdmissionPortV2<'a> {
    manager: &'a mut Manager,
}

pub struct RoventNnuecCollectionObserverRejectionV2 {
    pub error: AdaptiveManagerError,
    pub admission: RoventCollectionRootAdmissionV2,
    pub observer: AdaptiveCollectionObserverBoxV2<AdaptivePortQuestion, VirtualReviewTicketOutput>,
    pub observer_status: crate::pals_adaptive::AdaptiveCollectionV2ObserverStatus,
    pub panic_payload: Option<Box<dyn Any + Send>>,
}

impl RoventNnuecCollectionAdmissionPortV2<'_> {
    pub fn install(
        &mut self,
        admission: RoventCollectionRootAdmissionV2,
        observer: AdaptiveCollectionObserverBoxV2<AdaptivePortQuestion, VirtualReviewTicketOutput>,
    ) -> Result<(), RoventNnuecCollectionObserverRejectionV2> {
        self.manager.install_collection_observer_v2(admission, observer)
            .map_err(|failure| RoventNnuecCollectionObserverRejectionV2 {
                error: failure.error,
                admission: failure.admission,
                observer: failure.observer,
                observer_status: failure.observer_status,
                panic_payload: failure.panic_payload,
            })
    }
}
type RootReturn =
    Result<(Box<Idle>, SearchSessionReport, NnuecRootReceipt), (State, SearchSessionFailure)>;

#[derive(Clone, Debug)]
enum NnuecRootAdmission {
    V1(Arc<RoventRootAdmission>),
    V2(Arc<RoventRootAdmissionV2>),
}
impl NnuecRootAdmission {
    fn v1(&self) -> Option<&Arc<RoventRootAdmission>> {
        match self {
            Self::V1(a) => Some(a),
            _ => None,
        }
    }
    fn v2(&self) -> Option<&Arc<RoventRootAdmissionV2>> {
        match self {
            Self::V2(a) => Some(a),
            _ => None,
        }
    }
    fn digest(&self) -> Digest {
        match self {
            Self::V1(a) => a.digest(),
            Self::V2(a) => a
                .descriptor()
                .digest()
                .expect("original sealed descriptor was validated"),
        }
    }
    fn root_budget_digest(&self) -> Digest {
        match self {
            Self::V1(a) => a.root_budget_digest(),
            Self::V2(a) => a
                .descriptor()
                .root_budget_digest()
                .expect("original sealed root budget was validated"),
        }
    }
    fn original_context(&self) -> &SearchSessionContext {
        match self {
            Self::V1(a) => a.original_context(),
            Self::V2(a) => a.original_context(),
        }
    }
    fn allowed_root_moves(&self) -> &Arc<[BoardMove]> {
        match self {
            Self::V1(a) => a.allowed_root_moves(),
            Self::V2(a) => a.allowed_root_moves(),
        }
    }
    fn authority(&self) -> SearchAuthority {
        match self {
            Self::V1(a) => a.authority(),
            Self::V2(a) => a.authority(),
        }
    }
    fn hard_deadline(&self) -> rz_contracts::Deadline {
        match self {
            Self::V1(a) => a.hard_deadline(),
            Self::V2(a) => a.hard_deadline(),
        }
    }
    fn runtime_clock(&self) -> &rz_runtime::contracts::ContractSystemClock {
        match self {
            Self::V1(a) => a.runtime_clock(),
            Self::V2(a) => a.runtime_clock(),
        }
    }
}

#[derive(Clone, Debug)]
enum NnuecRootReceipt {
    V1(Arc<RoventNnuecRootReceipt>),
    V2(Arc<RoventNnuecRootReceiptV2>),
}
impl NnuecRootReceipt {
    fn v1(&self) -> Option<&Arc<RoventNnuecRootReceipt>> {
        match self {
            Self::V1(r) => Some(r),
            _ => None,
        }
    }
    fn frontier(&self) -> Option<&FrontierSnapshot> {
        match self {
            Self::V1(r) => r.frontier.as_ref(),
            Self::V2(r) => r.frontier.as_ref(),
        }
    }
    fn work(&self) -> Result<DriverWork, SearchSessionFailure> {
        match self {
            Self::V1(r) => r.work(),
            Self::V2(r) => r.work(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RoventNnuecRootReceiptV2 {
    pub domain: &'static str,
    pub authority: SearchAuthority,
    pub input_lock: Digest,
    pub root_admission: Arc<RoventRootAdmissionV2>,
    pub frontier: Option<FrontierSnapshot>,
    pub physical: Option<AdaptiveShutdownReceipt>,
    pub initial_retirement: Option<Arc<RoventNnuecInitialRetirementReceipt>>,
    pub counters: AdaptiveManagerCounters,
    pub stopped_before_research_execution: bool,
    pub deadline_before_frontier: bool,
    pre_materialization_retirement: Option<Arc<PreMaterializationRetirement>>,
}
impl RoventNnuecRootReceiptV2 {
    pub(crate) fn pre_materialization_retirement_matches(&self) -> bool {
        let Some(original) = &self.pre_materialization_retirement else {
            return false;
        };
        let Some(frontier) = &self.frontier else {
            return false;
        };
        original
            .admission
            .v2()
            .is_some_and(|a| Arc::ptr_eq(a, &self.root_admission))
            && self.domain == "rz-rovent-nnuec-native-root/2"
            && self.input_lock == self.root_admission.startup_lock_digest()
            && original.root_budget
                == self
                    .root_admission
                    .descriptor()
                    .root_budget_digest()
                    .expect("validated original root")
            && original.cleanup_deadline == self.root_admission.original_context().shutdown_deadline
            && original.frontier_authority == self.authority
            && original.frontier_authority == self.root_admission.authority()
            && frontier.authority == original.frontier_authority
            && original.frontier_deadline == self.root_admission.hard_deadline()
            && frontier.original_deadline == original.frontier_deadline
            && frontier.goal == original.frontier_goal
            && matches!(
                frontier.goal,
                FrontierGoalStatus::DepthSatisfied
                    | FrontierGoalStatus::MateSatisfied
                    | FrontierGoalStatus::Stopped(
                        FrontierStop::WorkBudget | FrontierStop::Deadline | FrontierStop::Canceled
                    )
            )
            && frontier.work == original.frontier_work
            && self.stopped_before_research_execution
            && !self.deadline_before_frontier
            && self.initial_retirement.is_none()
            && self.physical.is_none()
            && self.counters == AdaptiveManagerCounters::default()
            && original.ledger.authority == self.authority
            && original.ledger.clock_domain == self.root_admission.runtime_clock().domain()
            && original.ledger.clock_origin == self.root_admission.runtime_clock().origin()
            && original.ledger.spent_native_nodes == 0
            && original.ledger.reserved_native_nodes == 0
            && original.ledger.active_jobs == 0
            && original.ledger.outstanding_executions == 0
            && original.ledger.quarantined_jobs == 0
            && original.ledger.highest_job_sequence == 0
            && original.ledger.highest_execution_sequence == 0
            && frontier.work.native_usage_known
            && frontier.work.native_nodes == 0
            && frontier.work.native_qsearch_nodes == 0
            && frontier.work.reserved_native_nodes == 0
            && frontier.work.outstanding_native_executions == 0
    }

    fn work(&self) -> Result<DriverWork, SearchSessionFailure> {
        nnuec_receipt_work(
            self.frontier.as_ref(),
            self.counters,
            self.deadline_before_frontier,
            self.stopped_before_research_execution,
            self.physical.is_some(),
        )
    }
}

pub const ROVENT_NNUEC_ROOT_DOMAIN: &str = "rz-rovent-nnuec-native-root/1";
pub const ROVENT_NNUEC_INITIAL_DOMAIN: &str = "rz-rovent-nnuec-native-initial-stop/1";

/// Original pre-materialization retirement observation. Only the private
/// preparing_retirement_receipt producer creates it, after consuming Search's
/// RetiredPreparingRoot and validating its original ledger/clock/controls.
/// This is not a manager fence, native shutdown or a new execution authority.
#[derive(Clone, Debug)]
struct PreMaterializationRetirement {
    admission: NnuecRootAdmission,
    root_budget: Digest,
    cleanup_deadline: Instant,
    frontier_authority: SearchAuthority,
    frontier_deadline: rz_contracts::Deadline,
    frontier_goal: FrontierGoalStatus,
    frontier_work: rz_search::pals::rovent_frontier::FrontierNodeWork,
    ledger: rz_search::cpu_worker::CpuLedgerRegistrationObservation,
}

/// Producer-only original cleanup witnesses, distinct from a manager root
/// fence. They grant no search result, native execution, or new Ready owner.
pub struct RoventNnuecInitialRetirementReceipt {
    original_phase: InitialNnuecPhase,
    original_failure: Option<InitialNnuecFailure>,
    release: Option<InitialResearcherReleaseReceipt>,
    cache_release: NnuecCacheReleaseAck,
    observations: NnuecInitialCleanupObservations,
}
impl std::fmt::Debug for RoventNnuecInitialRetirementReceipt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoventNnuecInitialRetirementReceipt")
            .field("original_phase", &self.original_phase)
            .field("original_failure", &self.original_failure)
            .field("has_original_release", &self.release.is_some())
            .field("cache_release", &self.cache_release)
            .finish_non_exhaustive()
    }
}
impl RoventNnuecInitialRetirementReceipt {
    pub fn original_phase(&self) -> InitialNnuecPhase {
        self.original_phase
    }
    pub fn original_failure(&self) -> Option<&InitialNnuecFailure> {
        self.original_failure.as_ref()
    }
    pub fn release(&self) -> Option<&InitialResearcherReleaseReceipt> {
        self.release.as_ref()
    }
    pub fn cache_release(&self) -> &NnuecCacheReleaseAck {
        &self.cache_release
    }
    pub fn observations(&self) -> &NnuecInitialCleanupObservations {
        &self.observations
    }
}

/// Original semantic work and an optional actual manager researcher fence.
/// Initial C/TT cleanup is recorded separately from a manager root fence.
#[derive(Clone, Debug)]
pub struct RoventNnuecRootReceipt {
    pub domain: &'static str,
    pub authority: SearchAuthority,
    pub input_lock: Digest,
    pub root_admission: Arc<RoventRootAdmission>,
    pub frontier: Option<FrontierSnapshot>,
    pub physical: Option<AdaptiveShutdownReceipt>,
    pub initial_retirement: Option<Arc<RoventNnuecInitialRetirementReceipt>>,
    pub counters: AdaptiveManagerCounters,
    pub stopped_before_research_execution: bool,
    pub deadline_before_frontier: bool,
    // A caller cannot recreate this private original-retirement witness by
    // constructing a public root descriptor or by reporting zero counters.
    pre_materialization_retirement: Option<Arc<PreMaterializationRetirement>>,
}
impl RoventNnuecRootReceipt {
    pub(crate) fn pre_materialization_retirement_matches(&self) -> bool {
        let Some(original) = &self.pre_materialization_retirement else {
            return false;
        };
        let Some(frontier) = &self.frontier else {
            return false;
        };
        original
            .admission
            .v1()
            .is_some_and(|a| Arc::ptr_eq(a, &self.root_admission))
            && self.domain == ROVENT_NNUEC_ROOT_DOMAIN
            && self.input_lock == self.root_admission.startup_lock_digest()
            && original.root_budget == self.root_admission.root_budget_digest()
            && original.cleanup_deadline == self.root_admission.original_context().shutdown_deadline
            && original.frontier_authority == self.authority
            && original.frontier_authority == self.root_admission.authority()
            && frontier.authority == original.frontier_authority
            && original.frontier_deadline == self.root_admission.hard_deadline()
            && frontier.original_deadline == original.frontier_deadline
            && frontier.goal == original.frontier_goal
            && matches!(
                frontier.goal,
                FrontierGoalStatus::DepthSatisfied
                    | FrontierGoalStatus::MateSatisfied
                    | FrontierGoalStatus::Stopped(
                        FrontierStop::WorkBudget | FrontierStop::Deadline | FrontierStop::Canceled
                    )
            )
            && frontier.work == original.frontier_work
            && self.stopped_before_research_execution
            && !self.deadline_before_frontier
            && self.initial_retirement.is_none()
            && self.physical.is_none()
            && self.counters == AdaptiveManagerCounters::default()
            && original.ledger.authority == self.authority
            && original.ledger.clock_domain == self.root_admission.runtime_clock().domain()
            && original.ledger.clock_origin == self.root_admission.runtime_clock().origin()
            && original.ledger.spent_native_nodes == 0
            && original.ledger.reserved_native_nodes == 0
            && original.ledger.active_jobs == 0
            && original.ledger.outstanding_executions == 0
            && original.ledger.quarantined_jobs == 0
            && original.ledger.highest_job_sequence == 0
            && original.ledger.highest_execution_sequence == 0
            && frontier.work.native_usage_known
            && frontier.work.native_nodes == 0
            && frontier.work.native_qsearch_nodes == 0
            && frontier.work.reserved_native_nodes == 0
            && frontier.work.outstanding_native_executions == 0
    }

    pub fn work(&self) -> Result<DriverWork, SearchSessionFailure> {
        nnuec_receipt_work(
            self.frontier.as_ref(),
            self.counters,
            self.deadline_before_frontier,
            self.stopped_before_research_execution,
            self.physical.is_some(),
        )
    }
}

fn nnuec_receipt_work(
    frontier: Option<&FrontierSnapshot>,
    counters: AdaptiveManagerCounters,
    deadline_before_frontier: bool,
    stopped_before_research_execution: bool,
    physical_present: bool,
) -> Result<DriverWork, SearchSessionFailure> {
    let Some(frontier) = frontier else {
        if deadline_before_frontier && stopped_before_research_execution && !physical_present {
            return Ok(zero_work());
        }
        return Err(unknown(
            "RoventNnuecFrontier",
            "original frontier observation absent",
        ));
    };
    let total = if frontier.work.native_usage_known {
        Some(
            frontier
                .work
                .total_node_visits()
                .ok_or_else(|| unknown("RoventNnuecWork", "original work overflow"))?,
        )
    } else {
        None
    };
    Ok(DriverWork::RoventNative {
        root_entries: frontier.work.root_entries,
        total_node_visits: total,
        checked_transitions: frontier.work.checked_transitions,
        native_nodes: frontier
            .work
            .native_usage_known
            .then_some(frontier.work.native_nodes),
        native_qsearch_nodes: frontier
            .work
            .native_usage_known
            .then_some(frontier.work.native_qsearch_nodes),
        submitted_role_tickets: counters.submitted_role_tickets,
        consumed_role_outputs: counters.consumed_role_outputs,
        completed_nn_inputs: (counters.nn_input_count_unknown == 0)
            .then_some(counters.completed_nn_inputs),
        consumed_nn_inputs: (counters.consumed_nn_input_count_unknown == 0)
            .then_some(counters.consumed_nn_inputs),
        completed_depth: frontier.horizon.completed_depth.map(|depth| depth.0),
        maximum_examined_ply: frontier.horizon.maximum_examined_ply.0,
    })
}

#[derive(Clone, Debug)]
pub struct RoventNnuecInitialReceipt {
    authority: SearchAuthority,
    input_lock: Digest,
    source: Arc<StoppedInitial>,
    previous_root: Option<Arc<RoventNnuecRootReceipt>>,
}
impl RoventNnuecInitialReceipt {
    pub fn domain(&self) -> &'static str {
        ROVENT_NNUEC_INITIAL_DOMAIN
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
    pub fn previous_clean_root(&self) -> Option<&Arc<RoventNnuecRootReceipt>> {
        self.previous_root.as_ref()
    }
    pub fn used_original_startup_ready(&self) -> bool {
        self.previous_root.is_none()
    }
}

struct NativeOwner {
    service: RoventNativeService<AdaptivePortQuestion>,
    finished: Option<bool>,
    joined: bool,
    unexpected: Vec<TicketNativeEvent<VirtualReviewTicketOutput>>,
}
impl NativeOwner {
    fn new(service: RoventNativeService<AdaptivePortQuestion>) -> Self {
        Self {
            service,
            finished: None,
            joined: false,
            unexpected: Vec::new(),
        }
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
                            Some(self.finished.unwrap_or(true) && physical_shutdown_confirmed)
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
                    Err(unknown(
                        "RoventNnuecNativeProtocol",
                        "unexpected native terminal retained",
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
struct Idle {
    native: NativeOwner,
    owner: Option<Box<SemanticOwner>>,
    actor: Option<PreparationActor>,
    next_request_sequence: u64,
    game: Option<u64>,
    previous_root: Option<NnuecRootReceipt>,
}
enum State {
    Idle(Box<Idle>),
    /// A move-only owner bundle. Its type has never implied a clean fence.
    Retained {
        _owners: Box<dyn Any + Send>,
    },
    Closed,
    Moving,
}

pub struct RoventNnuecSessionRejection {
    pub failure: SearchSessionFailure,
    pub prepared: PreparedRoventNative<AdaptivePortQuestion>,
    pub loaded: LoadedRoventNnuecSource,
}

enum NnuecSources {
    V1(Arc<RoventPinnedNativeSources>),
    V2(Arc<RoventPinnedNativeSources<rz_experiments::RoventInputLockV2>>),
}
impl NnuecSources {
    fn v1(&self) -> Option<&Arc<RoventPinnedNativeSources>> {
        match self {
            Self::V1(v) => Some(v),
            _ => None,
        }
    }
    fn forget_original(&self) {
        match self {
            Self::V1(v) => std::mem::forget(Arc::clone(v)),
            Self::V2(v) => std::mem::forget(Arc::clone(v)),
        }
    }
}
enum NnuecLoaded {
    V1(LoadedRoventNnuecSource),
    V2(LoadedRoventNnuecSource<rz_experiments::RoventInputLockV2>),
}
impl NnuecLoaded {
    fn model(&self) -> &Arc<rz_search::cpu_value::nnuec::v2::Model> {
        match self {
            Self::V1(v) => v.model(),
            Self::V2(v) => v.model(),
        }
    }
}
pub struct RoventNnuecSessionV2Rejection {
    pub failure: SearchSessionFailure,
    pub prepared: PreparedRoventNativeV2<AdaptivePortQuestion>,
    pub loaded: LoadedRoventNnuecSource<rz_experiments::RoventInputLockV2>,
    pub process: Arc<super::process_boundary::VerifiedOwnedRoventProcessBoundaryV2>,
    pub collection_hooks: Option<Arc<dyn RoventNnuecCollectionHooksV2>>,
}
#[derive(Clone, Debug)]
pub struct RoventNnuecInitialReceiptV2 {
    pub authority: SearchAuthority,
    pub input_lock: Digest,
    pub source: Arc<StoppedInitialV2>,
    previous_root: Option<NnuecRootReceipt>,
}

impl RoventNnuecInitialReceiptV2 {
    pub fn previous_clean_root(&self) -> Option<&Arc<RoventNnuecRootReceiptV2>> {
        match &self.previous_root {
            Some(NnuecRootReceipt::V2(root)) => Some(root),
            _ => None,
        }
    }
    pub fn used_original_startup_ready(&self) -> bool {
        self.previous_root.is_none()
    }
}

/// Dedicated NNUEC session: no bootstrap researcher or legacy V5 receipt.
pub struct RoventNnuecSessionDriver {
    sources: NnuecSources,
    loaded: NnuecLoaded,
    original_v2_owner: Option<rz_experiments::RoventOriginalInputLock>,
    process_v2: Option<Arc<super::process_boundary::VerifiedOwnedRoventProcessBoundaryV2>>,
    snapshot: RoventNativeStartupSnapshot,
    configuration: StartupConfigurationDocument,
    clock: ProcessClock,
    issuer: AdaptiveRulesOwnerIssuer,
    identity: EngineIdentity,
    implementation: Digest,
    root_policy: RoventRootAdmissionPolicy,
    collection_origin: Option<RoventNativeCollectionOrigin>,
    collection_hooks: RoventNnuecCollectionHooks,
    collection_origin_v2: Option<RoventNativeCollectionOriginV2>,
    collection_hooks_v2: Option<Arc<dyn RoventNnuecCollectionHooksV2>>,
    /// The channel can lose its receiver after a successful send. Preserve
    /// the same original handoff until outer custody is actually proven.
    v2_handoff_custody: Mutex<Option<Arc<Mutex<RoventNnuecCollectionHandoffV2>>>>,
    v2_revoke_panic: Mutex<Option<Box<dyn Any + Send>>>,
    selective_policy: Digest,
    frontier_limits: FrontierLimits,
    execution_limits: AdaptiveExecutionLimits,
    controller: ControllerConfig,
    policy: CoordinatorPolicy,
    max_rounds: u64,
    max_native_nodes: u64,
    cpu_depth: u16,
    closed: AtomicBool,
    reset_requested: AtomicBool,
    state: Mutex<State>,
    latest: Mutex<Option<Arc<RoventNnuecRootReceipt>>>,
    latest_v2: Mutex<Option<Arc<RoventNnuecRootReceiptV2>>>,
    latest_initial: Mutex<Option<Arc<RoventNnuecInitialReceipt>>>,
    latest_initial_v2: Mutex<Option<Arc<RoventNnuecInitialReceiptV2>>>,
    latest_checker: Mutex<Option<PalsCheckerRegistration>>,
    storage: Mutex<Option<Box<SessionStorageLifecycle>>>,
    #[cfg(feature = "pals-adaptive-diagnostics")]
    nnuec_diagnostics: diagnostics::NnuecDiagnosticJournal,
}
impl RoventNnuecSessionDriver {
    /// A secondary revoke panic must not replace the original root error or
    /// release resources hidden in its payload while physical state is unknown.
    fn retain_v2_revoke_panic(&self, payload: Box<dyn Any + Send>) {
        match self.v2_revoke_panic.try_lock() {
            Ok(mut retained) if retained.is_none() => *retained = Some(payload),
            _ => std::mem::forget(payload),
        }
    }

    /// Source/configuration-derived identity only. The declared own-NNUEC
    /// assembly is checked without loading weights or constructing any native
    /// session, Ready, clock, worker or root authority.
    pub fn implementation_digest_from_metadata(
        document: &StartupConfigurationDocument,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Digest, BackendError> {
        if !matches!(
            document.fields().execution,
            ExecutionConfiguration::OwnResearcher {
                value: ResearchValueConfiguration::NnuecV2 { .. },
                ..
            }
        ) {
            return Err(super::fail(
                FailureKind::UnsupportedModel,
                "NNUEC session metadata requires the own-NNUEC assembly",
            ));
        }
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-nnuec-native-session/1\0");
        hash.update(super::compiled_bootstrap_source_digest());
        hash.update(include_bytes!("nnuec_session.rs"));
        hash.update(include_bytes!("nnuec_session/root_run.rs"));
        hash.update(include_bytes!("nnuec_session/retirement.rs"));
        hash.update(include_bytes!("nnuec_session/initial_cleanup.rs"));
        hash.update(include_bytes!("nnuec_session/diagnostics.rs"));
        hash.update(include_bytes!("nnuec_session/diagnostics/cpu_slices.rs"));
        // The existing digest producer validates the original closed document.
        hash.update(document.configuration_digest()?);
        hash.update([match root_policy {
            RoventRootAdmissionPolicy::ExactRegistered => 0,
            RoventRootAdmissionPolicy::DynamicGo => 1,
        }]);
        Ok(Digest(hash.finalize().into()))
    }

    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn new(
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        loaded: LoadedRoventNnuecSource,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventNnuecSessionRejection> {
        Self::new_with_collection_hooks(
            prepared,
            loaded,
            clock,
            owners,
            identity,
            root_policy,
            None,
        )
    }
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn new_with_collection_hooks(
        prepared: PreparedRoventNative<AdaptivePortQuestion>,
        loaded: LoadedRoventNnuecSource,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        collection_hooks: Option<RoventNnuecCollectionHooks>,
    ) -> Result<Self, RoventNnuecSessionRejection> {
        let built = (|| -> Result<_, SearchSessionFailure> {
            let document = prepared.sources().configuration();
            document
                .require_supported_native_capabilities()
                .map_err(|e| SearchSessionFailure::debug("RoventNnuecCapability", &e))?;
            let fields = document.fields();
            let ExecutionConfiguration::OwnResearcher {
                event_order,
                queues,
                max_rounds,
                max_native_nodes,
                cpu_depth,
                cpu_nodes_per_task,
                cpu_plan,
                value,
                controller,
                ..
            } = fields.execution
            else {
                return Err(failure(
                    "RoventNnuecStartup",
                    "actual own-R configuration required",
                ));
            };
            if !matches!(value, ResearchValueConfiguration::NnuecV2 { .. })
                || !prepared.snapshot().mode.has_native_r()
                || prepared.snapshot().process_epoch != clock.epoch().0
                || prepared.snapshot().input_lock != loaded.input_lock_digest()
                || !loaded.matches_sources(prepared.sources())
                || identity.name.is_empty()
                || identity.author.is_empty()
            {
                return Err(failure(
                    "RoventNnuecStartup",
                    "loaded NNUEC/source/process identity mismatch",
                ));
            }
            let collection_origin = match (
                prepared.sources().lock().manifest.purpose,
                &collection_hooks,
            ) {
                (rz_experiments::RoventPurpose::Collection, Some(hooks))
                    if hooks.root_admission.is_some() =>
                {
                    Some(prepared.collection_identity().map_err(|error| {
                        SearchSessionFailure::debug("RoventNnuecCollectionOrigin", &error)
                    })?)
                }
                (rz_experiments::RoventPurpose::Collection, _) | (_, Some(_)) => {
                    return Err(failure(
                        "RoventNnuecCollectionHooks",
                        "explicit Collection and actual root observer must be selected together",
                    ));
                }
                (_, None) => None,
            };
            let ready = prepared
                .service
                .ready_clock()
                .ok_or_else(|| failure("RoventNnuecStartup", "actual native Ready absent"))?;
            let original = clock.runtime_clock();
            if prepared.service.is_quarantined()
                || ready.domain() != original.domain()
                || ready.origin() != original.origin()
            {
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
            let control = controller
                .actual(*event_order)
                .map_err(|e| SearchSessionFailure::debug("RoventNnuecController", &e))?;
            // One initial Bound materializes exactly one physical researcher.
            // Wider declarations are rejected rather than silently resized.
            if control.max_workers != 1 || control.initial_workers != 1 {
                return Err(failure(
                    "RoventNnuecWorkers",
                    "this same-source assembly requires one declared own worker",
                ));
            }
            let limits = queues
                .actual()
                .map_err(|e| SearchSessionFailure::debug("RoventNnuecQueues", &e))?;
            if cpu_plan.paused_bytes_limit != limits.paused_bytes_per_worker_max {
                return Err(failure(
                    "RoventNnuecQueues",
                    "actual paused-owner budget differs from queues",
                ));
            }
            let root_tie_policy = document
                .own_r_root_tie_policy()
                .map_err(|e| SearchSessionFailure::debug("RoventNnuecTiePolicy", &e))?
                .ok_or_else(|| failure("RoventNnuecTiePolicy", "own-R policy is required"))?;
            let plan = DeferredNnuecCpuPlan::new_with_root_tie_policy(
                cpu_plan
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecPlan", &e))?,
                cpu_plan.ordering(),
                cpu_plan.paused_bytes_limit,
                cpu_plan.preserve_paused_side_scope,
                limits.worker_commands_max,
                cpu_plan.affinity(),
                root_tie_policy,
            )
            .map_err(|e| SearchSessionFailure::debug("RoventNnuecPlan", &e))?;
            let source = DeferredNnuecRSource::new(Arc::clone(loaded.model()), plan);
            let model = AdaptivePortModel::new_with_namespace(
                prepared.snapshot().role_model_identity(),
                Some(
                    prepared
                        .snapshot()
                        .model_value_identity()
                        .map_err(|e| SearchSessionFailure::debug("RoventNnuecNamespace", &e))?,
                ),
                prepared
                    .snapshot()
                    .model_namespace()
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecNamespace", &e))?,
            )
            .map_err(|e| SearchSessionFailure::debug("RoventNnuecNamespace", &e))?;
            let owner = ResearcherPreparingOwner::new_selected(
                ResearcherSemanticProfile {
                    config: PalsConfig {
                        beam_width: fields.semantic.beam_width,
                        line_plies: fields.semantic.line_plies,
                        max_nodes: fields.semantic.max_nodes,
                        max_records: fields.semantic.max_records,
                        max_role_calls: fields.semantic.max_role_calls,
                        cpu_nodes_per_task: *cpu_nodes_per_task,
                    },
                    maximum_root_prefix_plies: fields.semantic.maximum_root_prefix_plies,
                },
                model,
                ResolverPolicy::ModelWdlRestricted,
                document.post_repair_policy_for(prepared.snapshot().mode),
                fields
                    .review
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecReview", &e))?,
                fields
                    .index
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecIndex", &e))?,
                source,
            )
            .map_err(|e| SearchSessionFailure::debug("RoventNnuecSemantic", &e))?;
            let implementation =
                Self::implementation_digest_from_metadata(document, root_policy)
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecConfiguration", &e))?;
            Ok((
                owner,
                collection_origin,
                implementation,
                Digest(
                    document
                        .search_configuration_digest()
                        .map_err(|e| SearchSessionFailure::debug("RoventNnuecConfiguration", &e))?,
                ),
                fields
                    .frontier
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecFrontier", &e))?,
                limits,
                control,
                event_order.actual(),
                *max_rounds,
                *max_native_nodes,
                *cpu_depth,
            ))
        })();
        let (
            owner,
            collection_origin,
            implementation,
            selective_policy,
            frontier_limits,
            execution_limits,
            controller,
            policy,
            max_rounds,
            max_native_nodes,
            cpu_depth,
        ) = match built {
            Ok(value) => value,
            Err(failure) => {
                return Err(RoventNnuecSessionRejection {
                    failure,
                    prepared,
                    loaded,
                });
            }
        };
        let configuration = prepared.sources().configuration().clone();
        let (native, snapshot, sources) = prepared.into_parts();
        let issuer: AdaptiveRulesOwnerIssuer = Arc::new(move || owners.allocate());
        Ok(Self {
            sources: NnuecSources::V1(sources),
            loaded: NnuecLoaded::V1(loaded),
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
            collection_hooks: collection_hooks.unwrap_or_default(),
            collection_origin_v2: None,
            collection_hooks_v2: None,
            v2_handoff_custody: Mutex::new(None),
            v2_revoke_panic: Mutex::new(None),
            selective_policy,
            frontier_limits,
            execution_limits,
            controller,
            policy,
            max_rounds,
            max_native_nodes,
            cpu_depth,
            closed: AtomicBool::new(false),
            reset_requested: AtomicBool::new(false),
            state: Mutex::new(State::Idle(Box::new(Idle {
                native: NativeOwner::new(native),
                owner: Some(Box::new(owner)),
                actor: None,
                next_request_sequence: 1,
                game: None,
                previous_root: None,
            }))),
            latest: Mutex::new(None),
            latest_v2: Mutex::new(None),
            latest_initial: Mutex::new(None),
            latest_initial_v2: Mutex::new(None),
            latest_checker: Mutex::new(None),
            storage: Mutex::new(None),
            #[cfg(feature = "pals-adaptive-diagnostics")]
            nnuec_diagnostics: diagnostics::NnuecDiagnosticJournal::default(),
        })
    }
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn new_v2(
        prepared: PreparedRoventNativeV2<AdaptivePortQuestion>,
        loaded: LoadedRoventNnuecSource<rz_experiments::RoventInputLockV2>,
        process: Arc<super::process_boundary::VerifiedOwnedRoventProcessBoundaryV2>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
    ) -> Result<Self, RoventNnuecSessionV2Rejection> {
        Self::new_v2_inner(
            prepared,
            loaded,
            process,
            clock,
            owners,
            identity,
            root_policy,
            None,
        )
    }

    /// Internal /2 seam only. Product launch continues to use `new_v2` and
    /// rejects collection until its teacher and corpus owners are connected.
    #[allow(clippy::too_many_arguments, clippy::result_large_err, dead_code)]
    pub fn new_v2_with_collection_hooks(
        prepared: PreparedRoventNativeV2<AdaptivePortQuestion>,
        loaded: LoadedRoventNnuecSource<rz_experiments::RoventInputLockV2>,
        process: Arc<super::process_boundary::VerifiedOwnedRoventProcessBoundaryV2>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        hooks: Arc<dyn RoventNnuecCollectionHooksV2>,
    ) -> Result<Self, RoventNnuecSessionV2Rejection> {
        Self::new_v2_inner(
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

    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn new_v2_inner(
        prepared: PreparedRoventNativeV2<AdaptivePortQuestion>,
        loaded: LoadedRoventNnuecSource<rz_experiments::RoventInputLockV2>,
        process: Arc<super::process_boundary::VerifiedOwnedRoventProcessBoundaryV2>,
        clock: ProcessClock,
        owners: Arc<OwnerRegistry>,
        identity: EngineIdentity,
        root_policy: RoventRootAdmissionPolicy,
        collection_hooks_v2: Option<Arc<dyn RoventNnuecCollectionHooksV2>>,
    ) -> Result<Self, RoventNnuecSessionV2Rejection> {
        let built = (|| -> Result<_, SearchSessionFailure> {
            let sources = prepared.sources();
            if !Arc::ptr_eq(sources.lock_owner(), process.input_lock())
                || !process.matches_startup_control(prepared.startup_control())
                || sources.private_root_path() != process.private_cwd()
                || !sources
                    .same_original_root(process.original_root())
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecV2Root", &e))?
            {
                return Err(failure(
                    "RoventNnuecV2Custody",
                    "original V2 source, process and control differ",
                ));
            }
            let collection_origin_v2 = match (
                sources.lock().manifest.purpose,
                sources.collection_preload_sources().is_some(),
                collection_hooks_v2.is_some(),
            ) {
                (rz_experiments::RoventPurpose::Collection, true, true) => {
                    Some(prepared.collection_identity().map_err(|error| {
                        SearchSessionFailure::debug("RoventNnuecV2CollectionOrigin", &error)
                    })?)
                }
                (rz_experiments::RoventPurpose::Collection, _, _) | (_, true, _) | (_, _, true) => {
                    return Err(failure(
                        "RoventNnuecV2CollectionUnsupported",
                        "V2 collection requires original preload and an explicit V2 hook owner",
                    ));
                }
                _ => None,
            };
            let document = sources.configuration();
            document
                .require_supported_native_capabilities()
                .map_err(|e| SearchSessionFailure::debug("RoventNnuecCapability", &e))?;
            let fields = document.fields();
            let ExecutionConfiguration::OwnResearcher {
                event_order,
                queues,
                max_rounds,
                max_native_nodes,
                cpu_depth,
                cpu_nodes_per_task,
                cpu_plan,
                value,
                controller,
                ..
            } = fields.execution
            else {
                return Err(failure(
                    "RoventNnuecStartup",
                    "actual own-R configuration required",
                ));
            };
            if !matches!(value, ResearchValueConfiguration::NnuecV2 { .. })
                || !prepared.snapshot().mode.has_native_r()
                || prepared.snapshot().process_epoch != clock.epoch().0
                || prepared.snapshot().input_lock != loaded.input_lock_digest()
                || !loaded.matches_sources(prepared.sources())
                || identity.name.is_empty()
                || identity.author.is_empty()
            {
                return Err(failure(
                    "RoventNnuecStartup",
                    "loaded NNUEC/source/process identity mismatch",
                ));
            }
            let collection_origin = None;
            let original_owner =
                rz_experiments::RoventOriginalInputLock::from_v2(Arc::clone(sources.lock_owner()))
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecV2Lock", &e.error))?;
            let ready = prepared
                .service
                .ready_clock()
                .ok_or_else(|| failure("RoventNnuecStartup", "actual native Ready absent"))?;
            let original = clock.runtime_clock();
            if prepared.service.is_quarantined()
                || prepared.service.native_acceptance_verifier().is_none()
                || ready.domain() != original.domain()
                || ready.origin() != original.origin()
            {
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
            let control = controller
                .actual(*event_order)
                .map_err(|e| SearchSessionFailure::debug("RoventNnuecController", &e))?;
            // One initial Bound materializes exactly one physical researcher.
            // Wider declarations are rejected rather than silently resized.
            if control.max_workers != 1 || control.initial_workers != 1 {
                return Err(failure(
                    "RoventNnuecWorkers",
                    "this same-source assembly requires one declared own worker",
                ));
            }
            let limits = queues
                .actual()
                .map_err(|e| SearchSessionFailure::debug("RoventNnuecQueues", &e))?;
            if cpu_plan.paused_bytes_limit != limits.paused_bytes_per_worker_max {
                return Err(failure(
                    "RoventNnuecQueues",
                    "actual paused-owner budget differs from queues",
                ));
            }
            let root_tie_policy = document
                .own_r_root_tie_policy()
                .map_err(|e| SearchSessionFailure::debug("RoventNnuecTiePolicy", &e))?
                .ok_or_else(|| failure("RoventNnuecTiePolicy", "own-R policy is required"))?;
            let plan = DeferredNnuecCpuPlan::new_with_root_tie_policy(
                cpu_plan
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecPlan", &e))?,
                cpu_plan.ordering(),
                cpu_plan.paused_bytes_limit,
                cpu_plan.preserve_paused_side_scope,
                limits.worker_commands_max,
                cpu_plan.affinity(),
                root_tie_policy,
            )
            .map_err(|e| SearchSessionFailure::debug("RoventNnuecPlan", &e))?;
            let source = DeferredNnuecRSource::new(Arc::clone(loaded.model()), plan);
            let model = AdaptivePortModel::new_with_namespace(
                prepared.snapshot().role_model_identity(),
                Some(
                    prepared
                        .snapshot()
                        .model_value_identity()
                        .map_err(|e| SearchSessionFailure::debug("RoventNnuecNamespace", &e))?,
                ),
                prepared
                    .snapshot()
                    .model_namespace()
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecNamespace", &e))?,
            )
            .map_err(|e| SearchSessionFailure::debug("RoventNnuecNamespace", &e))?;
            let owner = ResearcherPreparingOwner::new_selected(
                ResearcherSemanticProfile {
                    config: PalsConfig {
                        beam_width: fields.semantic.beam_width,
                        line_plies: fields.semantic.line_plies,
                        max_nodes: fields.semantic.max_nodes,
                        max_records: fields.semantic.max_records,
                        max_role_calls: fields.semantic.max_role_calls,
                        cpu_nodes_per_task: *cpu_nodes_per_task,
                    },
                    maximum_root_prefix_plies: fields.semantic.maximum_root_prefix_plies,
                },
                model,
                ResolverPolicy::ModelWdlRestricted,
                document.post_repair_policy_for(prepared.snapshot().mode),
                fields
                    .review
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecReview", &e))?,
                fields
                    .index
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecIndex", &e))?,
                source,
            )
            .map_err(|e| SearchSessionFailure::debug("RoventNnuecSemantic", &e))?;
            let implementation =
                Self::implementation_digest_from_metadata(document, root_policy)
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecConfiguration", &e))?;
            Ok((
                owner,
                original_owner,
                collection_origin,
                collection_origin_v2,
                implementation,
                Digest(
                    document
                        .search_configuration_digest()
                        .map_err(|e| SearchSessionFailure::debug("RoventNnuecConfiguration", &e))?,
                ),
                fields
                    .frontier
                    .actual()
                    .map_err(|e| SearchSessionFailure::debug("RoventNnuecFrontier", &e))?,
                limits,
                control,
                event_order.actual(),
                *max_rounds,
                *max_native_nodes,
                *cpu_depth,
            ))
        })();
        let (
            owner,
            original_owner,
            collection_origin,
            collection_origin_v2,
            implementation,
            selective_policy,
            frontier_limits,
            execution_limits,
            controller,
            policy,
            max_rounds,
            max_native_nodes,
            cpu_depth,
        ) = match built {
            Ok(value) => value,
            Err(failure) => {
                return Err(RoventNnuecSessionV2Rejection {
                    failure,
                    prepared,
                    loaded,
                    process,
                    collection_hooks: collection_hooks_v2,
                });
            }
        };
        let configuration = prepared.sources().configuration().clone();
        let (native, snapshot, sources) = prepared.into_parts();
        let issuer: AdaptiveRulesOwnerIssuer = Arc::new(move || owners.allocate());
        Ok(Self {
            sources: NnuecSources::V2(sources),
            loaded: NnuecLoaded::V2(loaded),
            original_v2_owner: Some(original_owner),
            process_v2: Some(process),
            snapshot,
            configuration,
            clock,
            issuer,
            identity,
            implementation,
            root_policy,
            collection_origin,
            collection_hooks: RoventNnuecCollectionHooks::default(),
            collection_origin_v2,
            collection_hooks_v2,
            v2_handoff_custody: Mutex::new(None),
            v2_revoke_panic: Mutex::new(None),
            selective_policy,
            frontier_limits,
            execution_limits,
            controller,
            policy,
            max_rounds,
            max_native_nodes,
            cpu_depth,
            closed: AtomicBool::new(false),
            reset_requested: AtomicBool::new(false),
            state: Mutex::new(State::Idle(Box::new(Idle {
                native: NativeOwner::new(native),
                owner: Some(Box::new(owner)),
                actor: None,
                next_request_sequence: 1,
                game: None,
                previous_root: None,
            }))),
            latest: Mutex::new(None),
            latest_v2: Mutex::new(None),
            latest_initial: Mutex::new(None),
            latest_initial_v2: Mutex::new(None),
            latest_checker: Mutex::new(None),
            storage: Mutex::new(None),
            #[cfg(feature = "pals-adaptive-diagnostics")]
            nnuec_diagnostics: diagnostics::NnuecDiagnosticJournal::default(),
        })
    }
    #[cfg(feature = "pals-adaptive-diagnostics")]
    pub(crate) fn enable_nnuec_diagnostics(&self) {
        self.nnuec_diagnostics.enable();
    }
    #[cfg(feature = "pals-adaptive-diagnostics")]
    pub(crate) fn nnuec_owner_observation(
        &self,
    ) -> Result<Option<diagnostics::NnuecOwnerObservation>, &'static str> {
        self.nnuec_diagnostics.snapshot()
    }
    #[cfg(feature = "pals-adaptive-diagnostics")]
    pub(crate) fn nnuec_cpu_slice_observation(
        &self,
    ) -> Result<Option<diagnostics::cpu_slices::CpuSliceOwnerObservation>, &'static str> {
        self.nnuec_diagnostics.cpu_slice_snapshot()
    }
    #[allow(clippy::result_large_err)]
    pub(crate) fn install_storage_registration(
        &self,
        original: RoventStorageOwnedRuntimeRegistration,
    ) -> Result<
        Arc<SharedRoventStorageRegistration>,
        (RoventStorageOwnedRuntimeRegistration, SearchSessionFailure),
    > {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(_) => {
                return Err((
                    original,
                    SearchSessionFailure::physical_completion_unknown(),
                ));
            }
        };
        let mut slot = match self.storage.try_lock() {
            Ok(slot) => slot,
            Err(_) => {
                return Err((
                    original,
                    SearchSessionFailure::physical_completion_unknown(),
                ));
            }
        };
        let valid_idle = matches!(&*state, State::Idle(idle)
            if idle.previous_root.is_none() && idle.actor.is_none() && idle.game.is_none()
                && idle.owner.as_ref().is_some_and(|owner| owner.stores().generation() == 0 && owner.stores().root().is_none())
                && idle.native.finished.is_none() && !idle.native.joined
                && !idle.native.service.is_quarantined() && idle.native.service.occupancy() == 0);
        if slot.is_some() || self.closed.load(Ordering::Acquire) || !valid_idle {
            return Err((
                original,
                unknown(
                    "RoventNnuecStorageInstall",
                    "storage requires the unused original native StartupReady owner",
                ),
            ));
        }
        let Some(v1_sources) = self.sources.v1() else {
            return Err((
                original,
                failure(
                    "RoventNnuecV2StorageUnsupported",
                    "V2 storage owner is not connected",
                ),
            ));
        };
        if let Err(error) = original.validate_session_attachment(
            v1_sources.lock(),
            &self.clock,
            self.implementation,
        ) {
            return Err((
                original,
                SearchSessionFailure::debug("RoventNnuecStorageNamespace", &error),
            ));
        }
        if let Err(error) = original
            .profile()
            .validate_for_manifest(&v1_sources.lock().manifest)
        {
            return Err((
                original,
                SearchSessionFailure::debug("RoventNnuecStorageResources", &error),
            ));
        }
        let shared = Arc::new(SharedRoventStorageRegistration::new(original));
        *slot = Some(SessionStorageLifecycle::new(Arc::clone(&shared)));
        Ok(shared)
    }

    /// Original read-only Frontier/work observation. A later owner transfer
    /// or same-actor retirement may still fail; this is not run-success or a
    /// substitute for the actual returned report and physical completion.
    pub fn latest_root(&self) -> Result<Option<Arc<RoventNnuecRootReceipt>>, ContractError> {
        self.latest
            .lock()
            .map(|r| r.clone())
            .map_err(|_| poisoned())
    }
    pub fn latest_root_v2(&self) -> Result<Option<Arc<RoventNnuecRootReceiptV2>>, ContractError> {
        self.latest_v2
            .lock()
            .map(|r| r.clone())
            .map_err(|_| poisoned())
    }
    /// Diagnostic sampling must not delay the original UCI response.
    pub(crate) fn try_observe_root_v2(
        &self,
    ) -> Result<Option<Arc<RoventNnuecRootReceiptV2>>, &'static str> {
        self.latest_v2
            .try_lock()
            .map(|r| r.clone())
            .map_err(|_| "nnuec_v2_root_busy_or_poisoned")
    }
    pub fn latest_initial_v2(
        &self,
    ) -> Result<Option<Arc<RoventNnuecInitialReceiptV2>>, ContractError> {
        self.latest_initial_v2
            .lock()
            .map(|r| r.clone())
            .map_err(|_| poisoned())
    }
    pub(crate) fn try_observe_initial_v2(
        &self,
    ) -> Result<Option<Arc<RoventNnuecInitialReceiptV2>>, &'static str> {
        self.latest_initial_v2
            .try_lock()
            .map(|r| r.clone())
            .map_err(|_| "nnuec_v2_initial_busy_or_poisoned")
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
        if matches!(&*state, State::Retained { .. } | State::Moving) {
            observation.complete = false;
        }
        Ok(Some(observation))
    }
    pub fn latest_initial(&self) -> Result<Option<Arc<RoventNnuecInitialReceipt>>, ContractError> {
        self.latest_initial
            .lock()
            .map(|r| r.clone())
            .map_err(|_| poisoned())
    }

    /// Preserve the original observation before any later work projection or
    /// owner transfer can fail. This is neither a successful root return nor
    /// the actor/engine retirement ACK. On poison the caller still owns this
    /// same Arc and must retain it with the original physical owners.
    fn observe_root_receipt(
        &self,
        receipt: &Arc<RoventNnuecRootReceipt>,
    ) -> Result<(), SearchSessionFailure> {
        let mut latest = self.latest.lock().map_err(|_| {
            unknown(
                "RoventNnuecRootObservation",
                "original root receipt observation owner is poisoned",
            )
        })?;
        *latest = Some(Arc::clone(receipt));
        Ok(())
    }

    fn make_executed_receipt(
        &self,
        context: &SearchSessionContext,
        admission: &NnuecRootAdmission,
        frontier: &FrontierSnapshot,
        physical: &AdaptiveShutdownReceipt,
        counters: AdaptiveManagerCounters,
    ) -> NnuecRootReceipt {
        match admission {
            NnuecRootAdmission::V1(root) => {
                NnuecRootReceipt::V1(Arc::new(RoventNnuecRootReceipt {
                    domain: ROVENT_NNUEC_ROOT_DOMAIN,
                    authority: context.authority,
                    input_lock: self.snapshot.input_lock,
                    root_admission: Arc::clone(root),
                    frontier: Some(frontier.clone()),
                    physical: Some(physical.clone()),
                    initial_retirement: None,
                    counters,
                    stopped_before_research_execution: false,
                    deadline_before_frontier: false,
                    pre_materialization_retirement: None,
                }))
            }
            NnuecRootAdmission::V2(root) => {
                NnuecRootReceipt::V2(Arc::new(RoventNnuecRootReceiptV2 {
                    domain: "rz-rovent-nnuec-native-root/2",
                    authority: context.authority,
                    input_lock: self.snapshot.input_lock,
                    root_admission: Arc::clone(root),
                    frontier: Some(frontier.clone()),
                    physical: Some(physical.clone()),
                    initial_retirement: None,
                    counters,
                    stopped_before_research_execution: false,
                    deadline_before_frontier: false,
                    pre_materialization_retirement: None,
                }))
            }
        }
    }
    fn observe_any_root_receipt(
        &self,
        receipt: &NnuecRootReceipt,
    ) -> Result<(), SearchSessionFailure> {
        match receipt {
            NnuecRootReceipt::V1(receipt) => self.observe_root_receipt(receipt),
            NnuecRootReceipt::V2(receipt) => {
                let mut latest = self.latest_v2.lock().map_err(|_| {
                    unknown(
                        "RoventNnuecRootObservation",
                        "original V2 root observation is poisoned",
                    )
                })?;
                *latest = Some(Arc::clone(receipt));
                Ok(())
            }
        }
    }

    fn retain<T: Any + Send>(
        &self,
        owners: T,
        code: &'static str,
        detail: &str,
    ) -> (State, SearchSessionFailure) {
        self.closed.store(true, Ordering::Release);
        (
            State::Retained {
                _owners: Box::new(owners),
            },
            unknown(code, detail),
        )
    }

    // Keep the original native/cleanup owners, context/admission, sequence and
    // game identity separate while applying their existing cleanup gate.
    #[allow(clippy::too_many_arguments)]
    fn cleanup_initial(
        &self,
        native: NativeOwner,
        cleanup: super::nnuec_preparing::InitialNnuecCleanup,
        context: &SearchSessionContext,
        admission: NnuecRootAdmission,
        next_request_sequence: u64,
        game: Option<u64>,
        code: &'static str,
    ) -> RootReturn {
        let signal = native.service.signal();
        // Capture the original caller gate before cleanup revokes that same
        // root's cancellation owner. Cleanup itself cannot invent a stop.
        let original_control = context.accepts();
        let original_stop =
            matches!(original_control, Ok(false)) || context.soft_expired(Instant::now());
        let control_error = original_control.err();
        let mut cleanup = NnuecInitialCleanup::new(cleanup);
        loop {
            if Instant::now() >= context.shutdown_deadline {
                return Err(self.retain(
                    (native, cleanup, control_error),
                    code,
                    "original preparation cleanup deadline expired; exact owners retained",
                ));
            }
            let observed = signal.version();
            match cleanup.step() {
                NnuecInitialCleanupStep::Pending(pending) => cleanup = pending,
                NnuecInitialCleanupStep::Ready(cleaned) => {
                    if !original_stop
                        || control_error.is_some()
                        || !matches!(
                            cleaned.original_failure.as_ref(),
                            None | Some(InitialNnuecFailure::Preparation(
                                PreparationError::Cancelled | PreparationError::Expired
                            )) | Some(InitialNnuecFailure::Worker(
                                CpuWorkerError::NnuecAuthorityRevoked(
                                    NnuecAuthorityRevocation::Cancelled
                                        | NnuecAuthorityRevocation::Expired
                                )
                            ))
                        )
                    {
                        return Err(self.retain(
                            (native, cleaned, control_error),
                            code,
                            "actual initial preparation failure is not a normal control stop",
                        ));
                    }
                    let NnuecInitialCleaned {
                        retired,
                        actor,
                        original_phase,
                        original_failure,
                        release,
                        cache_release,
                        observations,
                    } = cleaned;
                    let proof = Arc::new(RoventNnuecInitialRetirementReceipt {
                        original_phase,
                        original_failure,
                        release,
                        cache_release,
                        observations,
                    });
                    return self.finish_preparing_retirement(
                        native,
                        retired,
                        Some(actor),
                        next_request_sequence,
                        game,
                        context,
                        admission,
                        Some(proof),
                    );
                }
                NnuecInitialCleanupStep::Retained(retained) => {
                    return Err(self.retain(
                        (native, retained, control_error),
                        code,
                        "actual physical cleanup observed; original semantic owners remain quarantined",
                    ));
                }
            }
            signal.wait_changed(
                observed,
                context
                    .shutdown_deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(1)),
            );
        }
    }

    /// Only an actual preparing-root retirement issues this move-only owner.
    /// This path never constructs a manager, native ticket, R grant or worker.
    #[allow(clippy::too_many_arguments)]
    fn finish_preparing_retirement(
        &self,
        native: NativeOwner,
        retired: RetiredPreparingRoot<AdaptivePortModel>,
        actor: Option<PreparationActor>,
        next_request_sequence: u64,
        game: Option<u64>,
        context: &SearchSessionContext,
        admission: NnuecRootAdmission,
        initial_retirement: Option<Arc<RoventNnuecInitialRetirementReceipt>>,
    ) -> RootReturn {
        let (retired, receipt) = match preparing_retirement_receipt_any(
            retired,
            context,
            admission.clone(),
            self.snapshot.input_lock,
            &self.clock,
            self.max_native_nodes,
            initial_retirement,
        ) {
            Ok(observed) => observed,
            Err(rejection) => {
                return Err(self.retain(
                    (native, rejection, actor),
                    "RoventNnuecInitialRetirement",
                    "actual initial retirement differs from the original root/ledger controls",
                ));
            }
        };
        if let Err(error) = self.observe_any_root_receipt(&receipt) {
            return Err(self.retain(
                (
                    native,
                    retired.owner,
                    actor,
                    retired.ledger,
                    retired.root_rules,
                    retired.root_moves,
                    receipt,
                ),
                "RoventNnuecRootObservation",
                &error.detail,
            ));
        }
        let work = match receipt.work() {
            Ok(work) => work,
            Err(error) => {
                return Err(self.retain(
                    (
                        native,
                        retired.owner,
                        actor,
                        retired.ledger,
                        retired.root_rules,
                        retired.root_moves,
                        receipt,
                    ),
                    "RoventNnuecInitialWork",
                    &error.detail,
                ));
            }
        };
        let mate_strategy = receipt.frontier().and_then(|f| f.mate_strategy.clone());
        let best_move = mate_strategy
            .as_ref()
            .map(|s| s.best_move())
            .or_else(|| admission.allowed_root_moves().first().copied());
        let report = SearchSessionReport {
            best_move,
            work,
            mate_strategy,
        };
        Ok((
            Box::new(Idle {
                native,
                owner: Some(Box::new(retired.owner)),
                actor,
                next_request_sequence,
                game,
                previous_root: Some(receipt.clone()),
            }),
            report,
            receipt,
        ))
    }

    /// The initial handoff already owns a real Ready worker. A control stop
    /// here needs no expired live-driver constructor and no manager admission.
    #[allow(clippy::too_many_arguments)]
    fn cleanup_adopted_initial(
        &self,
        native: NativeOwner,
        mut port: AdaptiveEnginePort<AdaptivePortModel>,
        mut worker: CpuWorker,
        actor: PreparationActor,
        pins: NnuecRootRetirementPins,
        ledger: CpuCreditLedger,
        root_budget: Digest,
        cleanup_deadline: Instant,
        root_moves: Arc<RootMoveDomain>,
        next_request_sequence: u64,
        game: Option<u64>,
        context: &SearchSessionContext,
        admission: NnuecRootAdmission,
    ) -> RootReturn {
        let signal = native.service.signal();
        let root_rules = port.root_rules();
        context.control.cancellation.store(true, Ordering::Release);
        match port.with_rovent_frontier(|frontier, _| {
            frontier.cancel();
            Ok(())
        }) {
            Ok(Some(())) => {}
            other => {
                return Err(self.retain(
                    (native, port, worker, actor, pins, ledger, root_moves, other),
                    "RoventNnuecAdoptedCancel",
                    "original adopted frontier cancellation was not observed",
                ));
            }
        }
        worker.request_shutdown();
        let shutdown = loop {
            if Instant::now() >= context.shutdown_deadline {
                return Err(self.retain(
                    (native, port, worker, actor, pins, ledger, root_moves),
                    "RoventNnuecAdoptedJoin",
                    "original cleanup deadline expired before actual Ready worker join",
                ));
            }
            let observed = signal.version();
            match worker.try_join() {
                Ok(Some(shutdown)) => break shutdown,
                Ok(None) => {}
                Err(error) => {
                    let detail = format!("{error:?}");
                    return Err(self.retain(
                        (native, port, worker, actor, pins, ledger, root_moves, error),
                        "RoventNnuecAdoptedJoin",
                        &detail,
                    ));
                }
            }
            signal.wait_changed(
                observed,
                context
                    .shutdown_deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(1)),
            );
        };
        if !initial_cleanup::permits_initial_reuse(&shutdown) {
            return Err(self.retain(
                (
                    native, port, worker, actor, pins, ledger, root_moves, shutdown,
                ),
                "RoventNnuecAdoptedStartup",
                "actual initial Ready/affinity/join readbacks do not permit reuse",
            ));
        }
        let recovered = match worker.take_nnuec_retirement() {
            Ok(engine) => engine,
            Err(error) => {
                let detail = format!("{error:?}");
                return Err(self.retain(
                    (
                        native, port, worker, actor, pins, ledger, root_moves, shutdown, error,
                    ),
                    "RoventNnuecAdoptedEngine",
                    &detail,
                ));
            }
        };
        let frontier = match port.retire_rovent_frontier() {
            Ok(Some(frontier)) => frontier,
            other => {
                return Err(self.retain(
                    (
                        native, port, worker, actor, pins, ledger, root_moves, shutdown, recovered,
                        other,
                    ),
                    "RoventNnuecAdoptedFrontier",
                    "original adopted frontier retirement was not observed",
                ));
            }
        };
        let mut retirement = NnuecRootRetirement::new(
            actor,
            AdaptiveNnuecRetiredOwner {
                engine: recovered,
                last_installation: None,
            },
            pins,
            port.into_engine(),
        );
        let retired = loop {
            if Instant::now() >= context.shutdown_deadline {
                return Err(self.retain(
                    (
                        native, retirement, worker, ledger, root_moves, shutdown, frontier,
                    ),
                    "RoventNnuecAdoptedRelease",
                    "original cleanup deadline expired before actual same-actor release",
                ));
            }
            let observed = signal.version();
            match retirement.step() {
                NnuecRootRetirementStep::Pending(pending) => retirement = pending,
                NnuecRootRetirementStep::Ready(retired) => break retired,
                NnuecRootRetirementStep::RetainedFailure(failed) => {
                    let detail = format!(
                        "actual initial engine/cache release failed in {:?}: {}; original owners retained",
                        failed.phase(),
                        failed.error(),
                    );
                    return Err(self.retain(
                        (
                            native, failed, worker, ledger, root_moves, shutdown, frontier,
                        ),
                        "RoventNnuecAdoptedRelease",
                        &detail,
                    ));
                }
            }
            signal.wait_changed(
                observed,
                context
                    .shutdown_deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(1)),
            );
        };
        let observations = NnuecInitialCleanupObservations {
            root_cancel_requested: true,
            ready_shutdown_requested: true,
            ready_shutdown: Some(shutdown),
            engine_release_matches_original: Some(
                retired
                    .engine_release
                    .matches_source(retired.owner.source()),
            ),
            cache_clears_released_engine: Some(
                retired
                    .engine_release
                    .matches_cache_release(&retired.cache_release),
            ),
            ..Default::default()
        };
        let proof = Arc::new(RoventNnuecInitialRetirementReceipt {
            original_phase: InitialNnuecPhase::AdoptPort,
            original_failure: None,
            release: Some(InitialResearcherReleaseReceipt::Worker(
                retired.engine_release,
            )),
            cache_release: retired.cache_release,
            observations,
        });
        self.finish_preparing_retirement(
            native,
            RetiredPreparingRoot {
                owner: retired.owner,
                ledger,
                root_budget,
                frontier: Some(frontier),
                deadline_before_frontier: false,
                root_rules,
                root_moves,
                cleanup_deadline,
            },
            Some(retired.actor),
            next_request_sequence,
            game,
            context,
            admission,
            Some(proof),
        )
    }

    fn run_root(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        if self.closed.load(Ordering::Acquire) {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        context
            .goal
            .validate(self.frontier_limits.maximum_goal_plies)
            .map_err(|e| SearchSessionFailure::debug("RoventNnuecGoal", &e))?;
        if context.authority.implementation != self.implementation
            || context.authority.epoch != self.clock.epoch()
        {
            return Err(failure(
                "RoventNnuecAuthority",
                "foreign original authority",
            ));
        }
        enum InitialSource {
            V1(SourceSeal),
            V2(PreparedSourceV2),
        }
        let source = match &self.sources {
            NnuecSources::V1(source) => InitialSource::V1(
                SourceSeal::prepare(
                    source.lock(),
                    self.root_policy,
                    position,
                    context,
                    &self.clock,
                )
                .map_err(|e| SearchSessionFailure::debug("RoventNnuecOriginalRoot", &e))?,
            ),
            NnuecSources::V2(_) => InitialSource::V2(
                PreparedSourceV2::prepare(
                    self.original_v2_owner
                        .as_ref()
                        .expect("original V2 owner retained"),
                    self.root_policy,
                    position,
                    context,
                    &self.clock,
                )
                .map_err(|e| SearchSessionFailure::debug("RoventNnuecOriginalRoot", &e))?,
            ),
        };
        let mut state = loop {
            match self.state.try_lock() {
                Ok(s) => break s,
                Err(TryLockError::Poisoned(_)) => {
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
                Err(TryLockError::WouldBlock) if Instant::now() < context.shutdown_deadline => {
                    std::thread::yield_now()
                }
                Err(_) => return Err(SearchSessionFailure::physical_completion_unknown()),
            }
        };
        let idle = match &*state {
            State::Idle(idle)
                if !self.closed.load(Ordering::Acquire)
                    && idle.native.finished.is_none()
                    && !idle.native.joined
                    && !idle.native.service.is_quarantined()
                    && idle.owner.is_some()
                    && idle.native.service.occupancy() == 0 =>
            {
                idle
            }
            _ => return Err(SearchSessionFailure::physical_completion_unknown()),
        };
        let admission = match source {
            InitialSource::V1(source) => match source
                .into_initial()
                .map_err(|e| SearchSessionFailure::debug("RoventNnuecOriginalRoot", &e))?
            {
                InitialAdmissionOutcome::Admitted(value) => NnuecRootAdmission::V1(Arc::new(value)),
                InitialAdmissionOutcome::StoppedInitial(source) => {
                    let receipt = Arc::new(RoventNnuecInitialReceipt {
                        authority: context.authority,
                        input_lock: self.snapshot.input_lock,
                        source: Arc::new(source),
                        previous_root: idle.previous_root.as_ref().and_then(|r| r.v1().cloned()),
                    });
                    let report = SearchSessionReport {
                        best_move: receipt.source.fallback(),
                        work: zero_work(),
                        mate_strategy: None,
                    };
                    *self.latest_initial.lock().map_err(|_| {
                        self.closed.store(true, Ordering::Release);
                        SearchSessionFailure::physical_completion_unknown()
                    })? = Some(receipt);
                    return Ok(report);
                }
            },
            InitialSource::V2(source) => match source
                .into_initial()
                .map_err(|e| SearchSessionFailure::debug("RoventNnuecOriginalRootV2", &e))?
            {
                InitialAdmissionOutcomeV2::Admitted(value) => {
                    NnuecRootAdmission::V2(Arc::new(value))
                }
                InitialAdmissionOutcomeV2::StoppedInitial(source) => {
                    let receipt = Arc::new(RoventNnuecInitialReceiptV2 {
                        authority: context.authority,
                        input_lock: self.snapshot.input_lock,
                        source: Arc::new(source),
                        previous_root: idle.previous_root.clone(),
                    });
                    let report = SearchSessionReport {
                        best_move: receipt.source.fallback(),
                        work: zero_work(),
                        mate_strategy: None,
                    };
                    *self.latest_initial_v2.lock().map_err(|_| {
                        self.closed.store(true, Ordering::Release);
                        SearchSessionFailure::physical_completion_unknown()
                    })? = Some(receipt);
                    return Ok(report);
                }
            },
        };
        let idle = match std::mem::replace(&mut *state, State::Moving) {
            State::Idle(idle) => idle,
            other => {
                *state = other;
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
        };
        match self.run_admitted(idle, position, context, admission, progress) {
            Ok((idle, report, _receipt)) => {
                *state = State::Idle(idle);
                // Each actual receipt was already observed at its original
                // creation boundary, before fallible physical-owner transfers.
                Ok(report)
            }
            Err((retained, error)) => {
                *state = retained;
                Err(error)
            }
        }
    }

    fn finish(&self, until: Instant) -> Result<(), SearchSessionFailure> {
        self.closed.store(true, Ordering::Release);
        let (v2_teacher_unproven, revoke_panicked) = if let Some(hooks) = &self.collection_hooks_v2
        {
            // Request the separate teacher shutdown but continue draining any
            // independently recoverable native, actor and storage owners.
            let panicked = match catch_unwind(AssertUnwindSafe(|| hooks.revoke())) {
                Ok(()) => false,
                Err(payload) => {
                    self.retain_v2_revoke_panic(payload);
                    true
                }
            };
            (true, panicked)
        } else {
            (false, false)
        };
        let revoke_panicked = revoke_panicked
            || (v2_teacher_unproven
                && self
                    .v2_revoke_panic
                    .try_lock()
                    .map(|retained| retained.is_some())
                    .unwrap_or(true));
        let teacher_unknown = || {
            if revoke_panicked {
                unknown(
                    "RoventNnuecV2RevokePanic",
                    "V2 teacher revoke panicked; original payload and owners retained",
                )
            } else {
                SearchSessionFailure::physical_completion_unknown()
            }
        };
        let mut state = self
            .state
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        if matches!(*state, State::Closed) {
            return if v2_teacher_unproven {
                Err(teacher_unknown())
            } else {
                Ok(())
            };
        }
        let idle = match &mut *state {
            State::Idle(idle) => idle,
            State::Retained { _owners } => {
                let drained = self.drain_retained_storage(_owners.as_mut(), until);
                return match drained {
                    Err(error) => Err(error),
                    Ok(()) if v2_teacher_unproven => Err(teacher_unknown()),
                    Ok(()) => Ok(()),
                };
            }
            _ => return Err(SearchSessionFailure::physical_completion_unknown()),
        };
        if let Some(actor) = &mut idle.actor {
            actor.request_shutdown();
            loop {
                match actor.try_join() {
                    Ok(Some(())) => break,
                    Ok(None) if Instant::now() < until => std::thread::yield_now(),
                    _ => return Err(SearchSessionFailure::physical_completion_unknown()),
                }
            }
            idle.actor.take();
        }
        idle.native.finish(until)?;
        if idle.native.joined && idle.native.finished == Some(true) {
            let mut storage = self
                .storage
                .try_lock()
                .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
            if let Some(storage) = storage.as_mut() {
                let signal = idle.native.service.signal();
                if let Err(mut rejected) = storage.finish(
                    idle.owner.take().map(StorageSemanticOwner::Pcr),
                    until,
                    &signal,
                ) {
                    if let Some(owner) = rejected.owner.take() {
                        match owner {
                            StorageSemanticOwner::Pcr(owner) => idle.owner = Some(owner),
                            owner => storage.retain_wrong_return(owner),
                        }
                    }
                    return Err(rejected.failure);
                }
            }
            if v2_teacher_unproven {
                // Native/actor shutdown is known, but the hook and its actual
                // teacher owner must remain quarantined rather than Closed.
                return Err(teacher_unknown());
            }
            *state = State::Closed;
            Ok(())
        } else {
            Err(SearchSessionFailure::physical_completion_unknown())
        }
    }
}

impl SearchSessionDriver for RoventNnuecSessionDriver {
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
        match &self.sources {
            NnuecSources::V1(source) => crate::rovent_root_admission::preadmission::validate_go(
                position,
                go,
                source.lock(),
                self.root_policy,
                self.frontier_limits.maximum_goal_plies,
            ),
            NnuecSources::V2(source) => crate::rovent_root_admission::preadmission::validate_go_v2(
                position,
                go,
                source.lock(),
                self.root_policy,
                self.frontier_limits.maximum_goal_plies,
            ),
        }
        .map_err(|e| {
            ContractError::new(
                rz_contracts::ErrorCode::InvalidInput,
                rz_contracts::Stage::Admission,
                e.detail,
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
    fn pals_checker_registration(&self) -> Option<PalsCheckerRegistration> {
        self.latest_checker
            .lock()
            .ok()
            .and_then(|registration| registration.clone())
    }
    fn checker_started(&self) -> Option<bool> {
        self.latest_checker
            .lock()
            .ok()
            .map(|registration| registration.is_some())
    }
    fn implementation(&self) -> Digest {
        self.implementation
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        Some(self.frontier_limits.maximum_goal_plies)
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(poisoned());
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
impl Drop for RoventNnuecSessionDriver {
    fn drop(&mut self) {
        // Outer delivery alone does not prove that the receiver retained the
        // original observer/data. Unclaimed evidence is quarantined on drop.
        if let Some(handoff) = self.v2_handoff_custody.get_mut()
            .unwrap_or_else(|poison| poison.into_inner()).take() {
            std::mem::forget(handoff);
        }
        let state = self.state.get_mut().unwrap_or_else(|p| p.into_inner());
        if !matches!(state, State::Closed) {
            std::mem::forget(std::mem::replace(state, State::Moving));
            // The Frontier was already retired when this read-only receipt
            // was observed. Quarantining the physical owners must not drop
            // their sole original frontier/admission observation on the way
            // out. A normally joined Closed state releases it as before.
            if let Some(receipt) = self
                .latest
                .get_mut()
                .unwrap_or_else(|p| p.into_inner())
                .take()
            {
                std::mem::forget(receipt);
            }
            if let Some(receipt) = self
                .latest_v2
                .get_mut()
                .unwrap_or_else(|p| p.into_inner())
                .take()
            {
                std::mem::forget(receipt);
            }
            if let Some(receipt) = self
                .latest_initial_v2
                .get_mut()
                .unwrap_or_else(|p| p.into_inner())
                .take()
            {
                std::mem::forget(receipt);
            }
            self.sources.forget_original();
            if let Some(process) = &self.process_v2 {
                std::mem::forget(Arc::clone(process));
            }
            // A /2 hook owns its teacher/collector and may have started a
            // physical worker before a callback failed or panicked. The
            // unresolved owner must outlive this session just like the native
            // and source owners; a successful callback is not a fence.
            if let Some(hooks) = self.collection_hooks_v2.take() {
                std::mem::forget(hooks);
            }
            if let Some(payload) = self
                .v2_revoke_panic
                .get_mut()
                .unwrap_or_else(|poison| poison.into_inner())
                .take()
            {
                std::mem::forget(payload);
            }
            let storage = self
                .storage
                .get_mut()
                .unwrap_or_else(|poison| poison.into_inner());
            if let Some(storage) = storage.take() {
                std::mem::forget(storage);
            }
            std::mem::forget(Arc::clone(self.loaded.model()));
        }
    }
}
fn should_quiesce(manager: &Manager) -> Result<bool, AdaptiveManagerError> {
    let receipt =
        manager
            .driver()
            .rovent_frontier_receipt()?
            .ok_or(AdaptiveManagerError::Invalid(
                "actual NNUEC frontier absent",
            ))?;
    match receipt.snapshot.goal {
        FrontierGoalStatus::Pending => Ok(false),
        FrontierGoalStatus::DepthSatisfied
        | FrontierGoalStatus::MateSatisfied
        | FrontierGoalStatus::SelectiveFrontier
        | FrontierGoalStatus::GoalUnresolved
        | FrontierGoalStatus::Stopped(
            FrontierStop::Canceled
            | FrontierStop::Deadline
            | FrontierStop::WorkBudget
            | FrontierStop::RoleBudget,
        ) => Ok(true),
        FrontierGoalStatus::Stopped(FrontierStop::Capacity) => {
            if manager.driver().rovent_publication_capacity_closed()? {
                Ok(true)
            } else {
                Err(AdaptiveManagerError::PhysicalUnknown)
            }
        }
        _ => Err(AdaptiveManagerError::PhysicalUnknown),
    }
}
type PreparingRetirementReceiptRejection = (
    RetiredPreparingRoot<AdaptivePortModel>,
    Option<Arc<RoventNnuecInitialRetirementReceipt>>,
    Result<rz_search::cpu_worker::CpuLedgerRegistrationObservation, CpuWorkerError>,
);

/// Consume Search's actual retired owner before issuing its observation. A
/// rejection returns every original owner/readback to the session's retention.
/// Kept private: tests use this same producer, never a witness constructor.
#[allow(clippy::result_large_err)]
fn preparing_retirement_receipt(
    mut retired: RetiredPreparingRoot<AdaptivePortModel>,
    context: &SearchSessionContext,
    admission: Arc<RoventRootAdmission>,
    input_lock: Digest,
    clock: &ProcessClock,
    native_node_limit: u64,
    initial_retirement: Option<Arc<RoventNnuecInitialRetirementReceipt>>,
) -> Result<
    (
        RetiredPreparingRoot<AdaptivePortModel>,
        Arc<RoventNnuecRootReceipt>,
    ),
    PreparingRetirementReceiptRejection,
> {
    let original_ledger = retired.ledger.registration_observation();
    if retired.root_budget != admission.root_budget_digest()
        || retired.cleanup_deadline != context.shutdown_deadline
        || !matches!(original_ledger, Ok(observed)
            if observed.authority == context.authority
                && observed.clock_domain == clock.domain()
                && observed.clock_origin == clock.runtime_clock().origin()
                && observed.native_node_limit == native_node_limit
                && observed.spent_native_nodes == 0
                && observed.reserved_native_nodes == 0
                && observed.active_jobs == 0
                && observed.outstanding_executions == 0
                && observed.quarantined_jobs == 0)
    {
        return Err((retired, initial_retirement, original_ledger));
    }
    // An initial C/TT cleanup has its own release/cache ACK and must not
    // be reclassified as a path that never materialized anything.
    let pre_materialization_retirement = if initial_retirement.is_none() {
        retired.frontier.as_ref().map(|frontier| {
            Arc::new(PreMaterializationRetirement {
                admission: NnuecRootAdmission::V1(Arc::clone(&admission)),
                root_budget: retired.root_budget,
                cleanup_deadline: retired.cleanup_deadline,
                frontier_authority: frontier.snapshot.authority,
                frontier_deadline: frontier.snapshot.original_deadline,
                frontier_goal: frontier.snapshot.goal,
                frontier_work: frontier.snapshot.work,
                ledger: original_ledger.expect("original retirement validated its ledger"),
            })
        })
    } else {
        None
    };
    let receipt = Arc::new(RoventNnuecRootReceipt {
        domain: ROVENT_NNUEC_ROOT_DOMAIN,
        authority: context.authority,
        input_lock,
        root_admission: admission.clone(),
        frontier: retired.frontier.take().map(|r| r.snapshot),
        physical: None,
        initial_retirement,
        counters: AdaptiveManagerCounters::default(),
        stopped_before_research_execution: true,
        deadline_before_frontier: retired.deadline_before_frontier,
        pre_materialization_retirement,
    });
    Ok((retired, receipt))
}

#[allow(clippy::result_large_err)]
fn preparing_retirement_receipt_any(
    retired: RetiredPreparingRoot<AdaptivePortModel>,
    context: &SearchSessionContext,
    admission: NnuecRootAdmission,
    input_lock: Digest,
    clock: &ProcessClock,
    native_node_limit: u64,
    initial_retirement: Option<Arc<RoventNnuecInitialRetirementReceipt>>,
) -> Result<
    (RetiredPreparingRoot<AdaptivePortModel>, NnuecRootReceipt),
    PreparingRetirementReceiptRejection,
> {
    match admission {
        NnuecRootAdmission::V1(root) => preparing_retirement_receipt(
            retired,
            context,
            root,
            input_lock,
            clock,
            native_node_limit,
            initial_retirement,
        )
        .map(|(retired, receipt)| (retired, NnuecRootReceipt::V1(receipt))),
        NnuecRootAdmission::V2(root) => {
            let mut retired = retired;
            let observed = retired.ledger.registration_observation();
            if retired.root_budget
                != root
                    .descriptor()
                    .root_budget_digest()
                    .expect("validated original root")
                || retired.cleanup_deadline != context.shutdown_deadline
                || !matches!(observed, Ok(o) if o.authority == context.authority && o.clock_domain == clock.domain() && o.clock_origin == clock.runtime_clock().origin() && o.native_node_limit == native_node_limit && o.spent_native_nodes == 0 && o.reserved_native_nodes == 0 && o.active_jobs == 0 && o.outstanding_executions == 0 && o.quarantined_jobs == 0)
            {
                return Err((retired, initial_retirement, observed));
            }
            let witness = if initial_retirement.is_none() {
                retired.frontier.as_ref().map(|frontier| {
                    Arc::new(PreMaterializationRetirement {
                        admission: NnuecRootAdmission::V2(Arc::clone(&root)),
                        root_budget: retired.root_budget,
                        cleanup_deadline: retired.cleanup_deadline,
                        frontier_authority: frontier.snapshot.authority,
                        frontier_deadline: frontier.snapshot.original_deadline,
                        frontier_goal: frontier.snapshot.goal,
                        frontier_work: frontier.snapshot.work,
                        ledger: observed.expect("original retirement ledger validated"),
                    })
                })
            } else {
                None
            };
            let receipt = RoventNnuecRootReceiptV2 {
                domain: "rz-rovent-nnuec-native-root/2",
                authority: context.authority,
                input_lock,
                root_admission: root,
                frontier: retired.frontier.take().map(|r| r.snapshot),
                physical: None,
                initial_retirement,
                counters: AdaptiveManagerCounters::default(),
                stopped_before_research_execution: true,
                deadline_before_frontier: retired.deadline_before_frontier,
                pre_materialization_retirement: witness,
            };
            Ok((retired, NnuecRootReceipt::V2(Arc::new(receipt))))
        }
    }
}

fn zero_work() -> DriverWork {
    DriverWork::RoventNative {
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
    }
}
fn failure(code: &'static str, detail: &str) -> SearchSessionFailure {
    SearchSessionFailure {
        physical_completion: DriverPhysicalCompletion::Confirmed,
        code,
        detail: detail.into(),
    }
}
fn unknown(code: &'static str, detail: &str) -> SearchSessionFailure {
    SearchSessionFailure {
        physical_completion: DriverPhysicalCompletion::Unknown,
        code,
        detail: detail.into(),
    }
}
fn poisoned() -> ContractError {
    ContractError::new(
        rz_contracts::ErrorCode::BackendFailure,
        rz_contracts::Stage::Output,
        "Rovent NNUEC owner closed or observation poisoned",
    )
}
