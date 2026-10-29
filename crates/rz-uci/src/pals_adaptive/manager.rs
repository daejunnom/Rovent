use super::cpu_slice_observation::{
    AdaptiveCpuSliceAccountingObservation, AdaptiveCpuSliceDiagnostic,
    AdaptiveCpuSliceDiagnosticError, AdaptiveCpuSliceDiagnosticUnavailable,
    AdaptiveCpuSliceScopeObservation,
};
use super::native_accounting::AdaptiveNativeAccountingSnapshot;
use super::validator::{AcceptedValidator, CompletedValidatorAction};
#[cfg(feature = "pals-adaptive-diagnostics")]
use super::{
    ContinuationDiagnostic, ContinuationDiagnosticUnavailable, ContinuationPendingObservation,
    ContinuationPendingSource, ContinuationResolutionObservation, ContinuationResolutionSource,
    ContinuationTransitionKind,
};
use crate::pals_native::ticketed::{
    NativeRootFence, PreparationId, TicketExecutionEvidence, TicketNativeBackend,
    TicketNativeEvent, TicketNativeObservationDisposition, TicketNativeObservationFailure,
    TicketNativeObservationFailures, TicketNativeObservationStage, TicketOwnerRunWitness,
    TicketPreparation, TicketServiceError, TicketedNativeService,
};
use rz_contracts::pals::{Move16, Role, RoleOutput, RolePayload, SearchAuthority};
use rz_contracts::pals_adaptive::{
    CpuJobEnvelope, CpuJobId, CpuWorkerId, ExternalCpuJobEnvelope, ExternalSliceGrant,
    ReviewViewSeal, RoleTicketEnvelope, SliceGrant,
};
use rz_contracts::rovent::{
    ExecutionMode, QuestionEnvelope, ReciprocalQuestionId, SearchRecipient,
};
use rz_contracts::{
    ClockDomain, ContractError, Deadline, Digest, ExecutionId, MonotonicTick, RequestId,
};
use rz_position::BoardMove;
use rz_position::contracts::{ContractPosition, ContractState};
use rz_runtime::adaptive::{
    AdaptiveController, BacklogEstimate, CacheClass, CensorReason, ControllerConfig,
    ControllerInput, CoordinatorPolicy, CostClass, CostDisposition, CostEstimator, CostObservation,
    CostStage, ExternalFeedbackWaitPrediction, FeedbackWaitPrediction, InvocationClass,
    ObservedCost, ReadyJob, ReadySelector, ReadyServiceCost, RunningCost, WorkerProgress,
};
use rz_runtime::contracts::{ContractClock as _, ContractSystemClock};
use rz_search::contract_time::InstantClock;
use rz_search::cpu::{
    CpuCompletion, CpuControlMailbox, CpuEngine, CpuReciprocalReply, CpuReciprocalRequest,
    CpuReciprocalScope,
};
use rz_search::cpu_worker::{
    CPU_JOB_CAPACITY, CpuAdmissionRejection, CpuCreditLedger, CpuJobTermination, CpuLedgerIdentity,
    CpuSliceReport, CpuWorker, CpuWorkerError, CpuWorkerEvent, CpuWorkerEventObservation,
    CpuWorkerProfile, CpuWorkerProgress, CpuWorkerProgressObservation, CpuWorkerSliceTiming,
    ReleasedCpuReservation, SettledCpuSlice,
};
use rz_search::external_cpu::worker::{
    ExternalCpuWorker, ExternalJob, ExternalRootFence, ExternalTeacherDispatch,
    ExternalWorkerCompletion, ExternalWorkerProgress, ExternalWorkerProgressObservation,
    ExternalWorkerReport, ExternalWorkerScope, external_evaluation_namespace,
    external_question_digest, external_root_restriction_digest, external_search_conditions,
};
use rz_search::external_cpu::{
    CpuEndpoint, CpuEndpointSelection, CpuExecutionUse, ExternalCpuConfig, ExternalSlicePolicy,
};
use rz_search::pals::coordinator::{
    CheckedExternalCpuBinding, ClosedFrame, CoordinatorError, CoordinatorHistoryUsage,
    CoordinatorWork, FrameBudget, FrameCompletionScope, FrameId, FrameInput, FramePhase,
    FrameTransition, FreshPairSeal, NativeAcceptancePermit, NativeAcceptanceReceipt,
    PhysicalRoleCompletion, ProposerRankingOutput, ProposerRankingPublished, RefinementMachine,
    RefinementStage, RoleCompletion, RoleFailure, RoleSubmission, RoleTicketCumulativeObservation,
    RoleTicketRegistry, StepEvidence, StepExecution, ValidatorActionHistoryUsage,
};
use rz_search::pals::engine::adaptive_port::validator_port::{
    AdaptiveValidatorAction, AdaptiveValidatorActionPlan, ValidatorActionId, ValidatorActionOutcome,
};
use rz_search::pals::engine::adaptive_port::{
    AdaptiveDeferredRepair, AdaptiveGoalLocalAdmission, AdaptiveQueuedRecheck,
    AdaptiveReviewLeaseAdmission, AdaptiveRulesPairProof, CheckedRootCandidate,
    DeferredRootCandidate, GoalLocalBlockReason, GoalLocalInput, GoalLocalStep,
};
use rz_search::pals::review_context::ReviewView;
use rz_search::pals::rovent_frontier::{
    FrontierError, FrontierFrameKind, NativeCloseReason, NativeReservation,
};
use sha2::{Digest as _, Sha256};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

const MAX_EVENT_DRAIN: usize = 256;
const MAX_ROLE_PENDING: usize = 16;

mod admitted;
mod cleanup;
mod collection;
mod cpu_semantic;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod utility_cohort;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod utility_intervention;
mod utility_route;
pub use cpu_semantic::{
    AdaptiveCpuCaptureCustody, AdaptiveCpuCapturePlan, AdaptiveCpuSemanticObserver,
    AdaptiveCpuSemanticObserverBox, AdaptiveCpuSemanticRetirement,
    AdaptiveCpuSemanticRetirementParts,
};
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use utility_cohort::{
    UtilityCohortReadyDisposition, UtilityCohortReadyObserver, UtilityCohortReadyOffer,
    UtilityCohortReadyRejection, UtilityCohortReadySnapshot, UtilityCohortTrialCustody,
};
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use utility_intervention::{
    UtilityInterventionSourceDisposition, UtilityInterventionSourceObserver,
};
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use utility_route::UtilityCohortRootClosureProof;
pub use utility_route::{
    UtilityRouteClosureProof, UtilityRouteFinalizationRejection, UtilityRoutePendingClosure,
};
mod effective_clock;
mod researcher_resources;
pub use researcher_resources::AdaptiveResearcherSourceKind;
use researcher_resources::{NativeResearcherResources, ResearcherResources};
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod profiled_external_source;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use profiled_external_source::{
    ExternalManagerLock, ExternalManagerRoot, ProfiledExternalManagerSource,
    ProfiledExternalSourceRejection,
};
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod profiled_external_receipt;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use profiled_external_receipt::{
    ProfiledExternalPersistentParts, ProfiledExternalShutdownReceipt,
};
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod profiled_external_dispatch;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod profiled_external_manager;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub use profiled_external_manager::{
    ProfiledExternalExecutionManager, ProfiledExternalManagerAdmissionRejection,
    ProfiledExternalPersistentRejection,
};
mod frontier;
mod goal_local;
mod goals;
mod nnuec;
mod reciprocal;
mod registered_ledger;
mod role_only;
use cleanup::{CleanupBoundary, automatic_cleanup_deadline};
pub use collection::*;
pub use frontier::{RoventNativeAccountingObservation, RoventReviewLeaseObservation};
#[cfg(feature = "pals-adaptive-diagnostics")]
pub use nnuec::NnuecPreparationAckObservation;
pub use nnuec::{
    AdaptiveNnuecRetiredOwner, AdaptiveNnuecRootParts, NnuecPreparationRecord, NnuecWorkError,
};
pub use reciprocal::{
    PeerFollowupDisposition, PeerFollowupObservation, RoventPeerFollowupPolicy,
    RoventReciprocalPolicy,
};
pub use registered_ledger::{
    AdaptiveRegisteredLedgerAdmissionRejection, AdaptiveRegisteredLedgerSources,
};
pub use role_only::*;

/// Search-issued immutable child input. The engine port keeps the opaque CPU
/// request seal and Store pins; the manager owns submission and physical custody.
pub struct AdaptiveReciprocalQuestion<Q> {
    pub question: QuestionEnvelope,
    pub input: FrameInput<ReviewView, Q>,
}
pub struct AdaptivePeerQuestion<Q> {
    pub origin: rz_search::pals::coordinator::AcceptedRoleOrigin,
    pub question: QuestionEnvelope,
    pub input: FrameInput<ReviewView, Q>,
}

/// Concrete queue/owner ceilings. These values are applied during physical
/// owner admission, rather than copied into receipts without enforcement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveExecutionLimits {
    pub waiting_jobs_max: usize,
    pub worker_commands_max: usize,
    pub role_tickets_max: usize,
    pub paused_bytes_per_worker_max: usize,
}
impl Default for AdaptiveExecutionLimits {
    fn default() -> Self {
        Self {
            waiting_jobs_max: CPU_JOB_CAPACITY,
            worker_commands_max: rz_search::cpu_worker::CPU_WORKER_COMMAND_CAPACITY,
            role_tickets_max: MAX_ROLE_PENDING,
            paused_bytes_per_worker_max: rz_search::cpu::CPU_PAUSED_STACK_MAX_BYTES,
        }
    }
}
impl AdaptiveExecutionLimits {
    pub fn validate(self) -> Result<(), AdaptiveManagerError> {
        if !(1..=CPU_JOB_CAPACITY).contains(&self.waiting_jobs_max)
            || !(1..=rz_search::cpu_worker::CPU_WORKER_COMMAND_CAPACITY)
                .contains(&self.worker_commands_max)
            || !(1..=MAX_ROLE_PENDING).contains(&self.role_tickets_max)
            || !(1..=rz_search::cpu::CPU_PAUSED_STACK_MAX_BYTES)
                .contains(&self.paused_bytes_per_worker_max)
        {
            return Err(AdaptiveManagerError::Invalid(
                "actual queue/paused resource limits",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum AdaptiveManagerError {
    NativeObservation(TicketNativeObservationFailure),
    Collection(AdaptiveCollectionError),
    Coordinator(CoordinatorError),
    Contract(ContractError),
    Native(TicketServiceError),
    Cpu(CpuWorkerError),
    Frontier(FrontierError),
    GoalLocalBlocked(GoalLocalBlockReason),
    NnuecPreparation(NnuecWorkError),
    NnuecCapture(Arc<rz_search::pals::engine::adaptive_port::AdaptiveNnuecCaptureError>),
    ProfiledExternal(rz_search::cpu_checker::CheckerError),
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    ProfiledQueryUnsupported(rz_search::pals::engine::adaptive_port::ProfiledExternalUnsupported),
    AdmissionClosed,
    ReviewLeaseBackpressure {
        current: usize,
        capacity: usize,
        required_new: usize,
        replacement_reserve: usize,
    },
    Invalid(&'static str),
    Semantic(String),
    Capacity(&'static str),
    PhysicalUnknown,
    Deadline,
    AuxiliaryPending,
    RoleCallLimit,
}
impl From<CoordinatorError> for AdaptiveManagerError {
    fn from(error: CoordinatorError) -> Self {
        Self::Coordinator(error)
    }
}
impl From<ContractError> for AdaptiveManagerError {
    fn from(error: ContractError) -> Self {
        Self::Contract(error)
    }
}
impl From<TicketServiceError> for AdaptiveManagerError {
    fn from(error: TicketServiceError) -> Self {
        Self::Native(error)
    }
}
impl From<CpuWorkerError> for AdaptiveManagerError {
    fn from(error: CpuWorkerError) -> Self {
        Self::Cpu(error)
    }
}
impl From<FrontierError> for AdaptiveManagerError {
    fn from(error: FrontierError) -> Self {
        Self::Frontier(error)
    }
}
impl std::fmt::Display for AdaptiveManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PALS adaptive manager: {self:?}")
    }
}
impl std::error::Error for AdaptiveManagerError {}

/// Bounded descriptors, never an instruction to do I/O on the manager thread.
pub struct OwnCpuQuestion {
    pub position: ContractPosition,
    pub root_moves: Option<Vec<BoardMove>>,
    pub target_depth: u16,
    pub total_native_node_credit: u64,
}
pub struct ForeignCpuQuestion {
    pub position: ContractPosition,
    pub root_moves: Option<Vec<BoardMove>>,
    pub target_depth: u16,
    pub policy: ExternalSlicePolicy,
}
pub enum CpuQuestion {
    Own(OwnCpuQuestion),
    External(ForeignCpuQuestion),
    /// Validated own question with no further native work credit. No physical
    /// job or completed evidence is manufactured at this boundary.
    OwnBudgetExhausted {
        semantic_limit: u64,
        semantic_spent: u64,
    },
}

/// Commit result is produced by the Rules/store owner from the ORIGINAL view.
/// No completion boolean supplied by a worker can construct Rules evidence.
pub struct CheckedStep<Q> {
    pub evidence: Digest,
    pub transition: FrameTransition<ReviewView, Q>,
    pub fresh_pair: Option<FreshPairSeal>,
    pub rules_pair: Option<AdaptiveRulesPairProof>,
    pub queued_recheck: Option<AdaptiveQueuedRecheck>,
}
pub struct CheckedRulesStep<Q> {
    pub endpoint: Arc<ContractState>,
    pub checked: CheckedStep<Q>,
}
pub struct CandidateFrame<Q> {
    pub first_move: Move16,
    pub input: FrameInput<ReviewView, Q>,
    pub budget: FrameBudget,
    pub deferred: Option<DeferredRootCandidate<Q>>,
    pub initialized: Option<CheckedRootCandidate<Q>>,
}

/// Search-owned semantic bridge. Implementations reuse PalsEngine's exact
/// graph/store/resolver and selected index. These methods may replay bounded
/// Rules lines and commit hot facts; they MUST NOT run a model, CPU search,
/// archive I/O, an external pipe drain, or wait for a worker. Archive effects
/// remain with the independent archive owner and finish after worker fences.
/// A failed target bind returns every original owner to the caller.
pub type UtilityInterventionTargetBinding<Q> = Result<
    rz_search::pals::engine::adaptive_port::utility_route::UtilityInterventionTarget<Q>,
    (
        AdaptiveManagerError,
        rz_search::pals::engine::adaptive_port::utility_route::UtilityRoutePreparation,
        rz_search::pals::coordinator::UtilityInterventionReadySource<Q>,
        Arc<rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture>,
    ),
>;

pub trait AdaptiveSearchDriver<Q> {
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn utility_cohort_common_work(
        &self,
    ) -> Result<
        Option<(
            Option<u64>,
            rz_search::pals::rovent_frontier::FrontierNodeWork,
        )>,
        AdaptiveManagerError,
    > {
        Ok(None)
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn freeze_utility_witness_bodies(
        &self,
        _: &rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteSemanticOutcome,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::utility_route::UtilityConditionalWitnessBodies,
        AdaptiveManagerError,
    > {
        Err(AdaptiveManagerError::Invalid(
            "utility original witness bodies unsupported",
        ))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn utility_cohort_offer_semantics(
        &self,
        _: &CoordinatorWork<ReviewView, Q>,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::UtilityCohortOfferSemanticSource<Q>,
        AdaptiveManagerError,
    > {
        Err(AdaptiveManagerError::Invalid(
            "utility cohort offer semantics unsupported",
        ))
    }
    fn prepare_utility_route(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: u16,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::utility_route::UtilityRoutePreparation,
        AdaptiveManagerError,
    > {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual utility source",
        ))
    }
    #[allow(clippy::result_large_err)]
    fn bind_utility_intervention_target(
        &self,
        preparation: rz_search::pals::engine::adaptive_port::utility_route::UtilityRoutePreparation,
        source: rz_search::pals::coordinator::UtilityInterventionReadySource<Q>,
        input: Arc<rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture>,
    ) -> UtilityInterventionTargetBinding<Q> {
        Err((
            AdaptiveManagerError::Invalid("intervention target factory unsupported"),
            preparation,
            source,
            input,
        ))
    }
    #[allow(clippy::result_large_err)]
    fn prepare_utility_intervention(
        &mut self,
        target: rz_search::pals::engine::adaptive_port::utility_route::UtilityInterventionTarget<Q>,
        _: rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteRecipient,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::utility_route::UtilityInterventionCandidate<Q>,
        (
            AdaptiveManagerError,
            rz_search::pals::engine::adaptive_port::utility_route::UtilityInterventionTarget<Q>,
        ),
    > {
        Err((
            AdaptiveManagerError::Invalid("intervention query factory unsupported"),
            target,
        ))
    }
    #[allow(clippy::result_large_err)]
    fn bind_utility_intervention_route(
        &self,
        admission:rz_search::pals::engine::adaptive_port::utility_route::UtilityInterventionAdmission<Q>,
        _: &CoordinatorWork<ReviewView, Q>,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        (
            AdaptiveManagerError,
            rz_search::pals::engine::adaptive_port::utility_route::UtilityInterventionAdmission<Q>,
        ),
    > {
        Err((
            AdaptiveManagerError::Invalid("intervention action factory unsupported"),
            admission,
        ))
    }
    fn prepare_utility_intervention_arm_request(
        &self,
        _: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &rz_search::cpu_value::nnuec::v2::Model,
        _: Arc<AtomicBool>,
    ) -> Result<rz_search::cpu_value::nnuec::v2::review::Request, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "intervention arm Request unsupported",
        ))
    }
    #[allow(clippy::result_large_err)]
    fn attach_utility_intervention_arm_request(
        &self,
        _: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        _: &CoordinatorWork<ReviewView, Q>,
        request: rz_search::cpu_value::nnuec::v2::review::Request,
    ) -> Result<
        rz_search::cpu_value::nnuec::v2::review::Request,
        (
            AdaptiveManagerError,
            rz_search::cpu_value::nnuec::v2::review::Request,
        ),
    > {
        Err((
            AdaptiveManagerError::Invalid("intervention original arm Request unsupported"),
            request,
        ))
    }
    #[allow(clippy::result_large_err)]
    fn bind_utility_intervention_arm_capture(
        &self,
        _: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        _: &CoordinatorWork<ReviewView, Q>,
        input: Arc<rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture>,
    ) -> Result<
        (),
        (
            AdaptiveManagerError,
            Arc<rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture>,
        ),
    > {
        Err((
            AdaptiveManagerError::Invalid("intervention arm capture unsupported"),
            input,
        ))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn utility_intervention_arm_metadata(
        &self,
        _: &rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        _: &CoordinatorWork<ReviewView, Q>,
    ) -> Result<serde_json::Value, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "intervention arm metadata unsupported",
        ))
    }
    fn prepare_utility_route_input_request(
        &self,
        _: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRoutePreparation,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &rz_search::cpu_value::nnuec::v2::Model,
        _: Arc<AtomicBool>,
    ) -> Result<rz_search::cpu_value::nnuec::v2::review::Request, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual utility Request factory",
        ))
    }
    #[allow(clippy::result_large_err)]
    fn bind_utility_route_input(
        &self,
        preparation: rz_search::pals::engine::adaptive_port::utility_route::UtilityRoutePreparation,
        _: &CoordinatorWork<ReviewView, Q>,
        input: Arc<rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture>,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        (
            AdaptiveManagerError,
            rz_search::pals::engine::adaptive_port::utility_route::UtilityRoutePreparation,
            Arc<rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture>,
        ),
    > {
        Err((
            AdaptiveManagerError::Invalid("driver has no utility input binding"),
            preparation,
            input,
        ))
    }
    fn validate_utility_route_work(
        &self,
        _: &rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        _: &CoordinatorWork<ReviewView, Q>,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual utility dependency",
        ))
    }
    fn begin_utility_route_commit(
        &self,
        _: &rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        _: &CoordinatorWork<ReviewView, Q>,
        _: StepExecution,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteCommitGuard,
        AdaptiveManagerError,
    > {
        Err(AdaptiveManagerError::Invalid(
            "driver has no utility precommit guard",
        ))
    }
    fn register_utility_route_descendant(
        &self,
        _: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        _: &CoordinatorWork<ReviewView, Q>,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no original utility descendant source",
        ))
    }
    fn register_utility_route_peer(
        &self,
        _: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &rz_search::pals::coordinator::ReciprocalDependencyProof<ReviewView, Q>,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual sealed utility peer",
        ))
    }
    fn commit_utility_route_step(
        &self,
        _: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        _: &rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteCommitGuard,
        _: &CoordinatorWork<ReviewView, Q>,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual utility CPU consumption",
        ))
    }
    fn commit_utility_route_await(
        &self,
        _: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual utility Await source",
        ))
    }
    fn commit_utility_route_role(
        &self,
        _: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        _: &rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteCommitGuard,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &rz_search::pals::coordinator::AcceptedRoleStep,
        _: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual utility accepted head consumption",
        ))
    }
    fn record_utility_route_r_pc_consumption(
        &self,
        _: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        _: &rz_search::pals::engine::adaptive_port::CpuSemanticWitness,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no utility selected R observation",
        ))
    }
    #[allow(clippy::result_large_err)]
    fn finish_utility_route_semantics(
        &self,
        action: rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteSemanticOutcome,
        (
            AdaptiveManagerError,
            rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        ),
    > {
        Err((
            AdaptiveManagerError::Invalid("driver has no utility semantic closure"),
            action,
        ))
    }
    fn reserve_cpu_nnuec_capture(
        &self,
        _: &rz_search::pals::engine::adaptive_port::AdaptiveNnuecReviewSeal,
        _: Arc<rz_search::cpu_value::nnuec::v2::Model>,
        _: Arc<rz_search::cpu_value::nnuec::v2::review::BoundNnuecReviewContext>,
        _: u64,
        _: rz_search::cpu_value::nnuec::v2::capture::NnuecCaptureLimits,
    ) -> Result<
        rz_search::cpu_value::nnuec::v2::capture::NnuecCaptureReservation,
        AdaptiveManagerError,
    > {
        Err(AdaptiveManagerError::Invalid(
            "driver has no original R capture factory",
        ))
    }
    fn cpu_semantic_publication_recorded(
        &self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuSliceReport,
    ) -> bool {
        false
    }
    fn cpu_semantic_witnesses(
        &self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuSliceReport,
        _: &SettledCpuSlice,
        _: &rz_search::cpu_value::nnuec::v2::capture::NnuecCaptureLineage,
    ) -> Result<Vec<rz_search::pals::engine::adaptive_port::CpuSemanticWitness>, AdaptiveManagerError>
    {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual R semantic witness factory",
        ))
    }
    fn selected_cpu_pc_consumption(
        &self,
        _: &rz_search::pals::engine::adaptive_port::CpuSemanticWitness,
        _: &FrameInput<ReviewView, Q>,
        _: &rz_search::pals::coordinator::AcceptedRoleStep,
        _: &RoleOutput,
    ) -> Result<
        Option<rz_search::pals::engine::adaptive_port::CpuSemanticWitness>,
        AdaptiveManagerError,
    > {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual selected R observation consumer",
        ))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_profiled_external(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &ExternalCpuConfig,
        _: &rz_search::external_cpu::ProfiledExternalRootSource,
        _: ReciprocalQuestionId,
        _: ExecutionId,
        _: rz_search::external_cpu::ExternalGoConstraint,
        _: Duration,
        _: rz_runtime::CompletionSignal,
        _: ClockDomain,
        _: MonotonicTick,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::AdaptiveProfiledExternalPreparation,
        AdaptiveManagerError,
    > {
        Err(AdaptiveManagerError::ProfiledExternal(
            rz_search::cpu_checker::CheckerError::Unsupported(
                "driver has no profiled external producer",
            ),
        ))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn profiled_external_admitted(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &rz_search::external_cpu::ProfiledExternalRootSource,
        _: &rz_search::external_cpu::ProfiledExternalDispatch,
        _: ClockDomain,
        _: MonotonicTick,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::AdaptiveProfiledExternalAdmission,
        AdaptiveManagerError,
    > {
        Err(AdaptiveManagerError::ProfiledExternal(
            rz_search::cpu_checker::CheckerError::Unsupported(
                "driver has no actual profiled external binding",
            ),
        ))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn commit_profiled_external(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &rz_search::external_cpu::ProfiledExternalRootSource,
        _: rz_search::pals::rovent_frontier::FrontierExternalReservation,
        _: &Arc<rz_search::external_cpu::ProfiledExternalCompletion>,
    ) -> Result<CheckedStep<Q>, AdaptiveManagerError> {
        Err(AdaptiveManagerError::ProfiledExternal(
            rz_search::cpu_checker::CheckerError::Unsupported(
                "driver has no actual profiled external consumer",
            ),
        ))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn observe_profiled_external(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &rz_search::external_cpu::ProfiledExternalRootSource,
        _: rz_search::pals::rovent_frontier::FrontierExternalReservation,
        _: &Arc<rz_search::external_cpu::ProfiledExternalCompletion>,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::ProfiledExternal(
            rz_search::cpu_checker::CheckerError::Unsupported(
                "driver has no profiled external raw observer",
            ),
        ))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn consume_profiled_reciprocal_cpu(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &rz_search::external_cpu::ProfiledExternalRootSource,
        _: rz_search::pals::rovent_frontier::FrontierExternalReservation,
        _: &Arc<rz_search::external_cpu::ProfiledExternalCompletion>,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::CheckedProfiledExternalResponse,
        AdaptiveManagerError,
    > {
        Err(AdaptiveManagerError::ProfiledExternal(
            rz_search::cpu_checker::CheckerError::Unsupported(
                "driver has no actual profiled reciprocal consumer",
            ),
        ))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn reject_profiled_external(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::ProfiledExternal(
            rz_search::cpu_checker::CheckerError::Unsupported(
                "driver has no unsubmitted profiled question",
            ),
        ))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn validate_profiled_external_registration(
        &self,
        _: &rz_search::external_cpu::ProfiledExternalRootSource,
        _: &Arc<crate::rovent_root_admission::RoventRootAdmission>,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::ProfiledExternal(
            rz_search::cpu_checker::CheckerError::Unsupported(
                "driver has no original profiled external source registration",
            ),
        ))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn validate_profiled_external_registration_v2(
        &self,
        _: &rz_search::external_cpu::ProfiledExternalRootSource,
        _: &Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV2>,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::ProfiledExternal(
            rz_search::cpu_checker::CheckerError::Unsupported(
                "driver has no original V2 profiled external source registration",
            ),
        ))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn profiled_external_work(
        &self,
    ) -> Option<rz_search::pals::rovent_frontier::FrontierExternalWork> {
        None
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn profiled_retained_host_bytes(&self) -> Option<u64> {
        None
    }
    fn root_rules(&self) -> Arc<ContractState>;
    fn validate_cpu_nnuec_model(
        &self,
        _: &rz_search::cpu_value::nnuec::v2::Model,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual NNUEC researcher",
        ))
    }
    fn prepare_cpu_nnuec_review(
        &self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &rz_search::cpu_value::nnuec::v2::Model,
        _: Arc<AtomicBool>,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::AdaptiveNnuecReviewRequest,
        AdaptiveManagerError,
    > {
        Err(AdaptiveManagerError::Invalid(
            "driver has no original NNUEC question",
        ))
    }
    fn install_cpu_nnuec_review(
        &self,
        _: &rz_search::pals::engine::adaptive_port::AdaptiveNnuecReviewSeal,
        _: Arc<rz_search::cpu_value::nnuec::v2::Model>,
        _: Arc<rz_search::cpu_value::nnuec::v2::review::BoundNnuecReviewContext>,
    ) -> Result<rz_search::cpu_worker::PreparedValueInstallation, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no original NNUEC installation",
        ))
    }
    fn accept_cpu_nnuec_installation(
        &mut self,
        _: &rz_search::pals::engine::adaptive_port::AdaptiveNnuecReviewSeal,
        _: Arc<rz_search::cpu_value::nnuec::v2::Model>,
        _: Arc<rz_search::cpu_value::nnuec::v2::review::BoundNnuecReviewContext>,
        _: &rz_search::cpu_worker::ValueInstallReceipt,
        _: &CpuWorker,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual NNUEC install acknowledgement",
        ))
    }
    fn release_cpu_nnuec_installation(
        &mut self,
        _: &rz_search::pals::engine::adaptive_port::AdaptiveNnuecReviewSeal,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual NNUEC installation cleanup",
        ))
    }
    fn next_rovent_goal_input(
        &mut self,
        _: ClockDomain,
        _: MonotonicTick,
    ) -> Result<AdaptiveGoalLocalAdmission<Q>, AdaptiveManagerError> {
        Ok(AdaptiveGoalLocalAdmission::NoQuestion)
    }
    fn validate_rovent_goal_input(
        &self,
        _: &GoalLocalInput<Q>,
        _: &RoleTicketEnvelope,
        _: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual GoalLocal source",
        ))
    }
    fn consume_rovent_goal_input(
        &mut self,
        _: &GoalLocalInput<Q>,
        _: &rz_search::pals::coordinator::AcceptedRoleStep,
        _: &RoleOutput,
        _: Option<TicketExecutionEvidence>,
        _: ClockDomain,
        _: MonotonicTick,
    ) -> Result<GoalLocalStep, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual GoalLocal consumer",
        ))
    }
    fn discard_rovent_goal_input(
        &mut self,
        _: &GoalLocalInput<Q>,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual GoalLocal cleanup",
        ))
    }
    fn rovent_goal_has_pending(&self) -> Result<bool, AdaptiveManagerError> {
        Ok(false)
    }
    fn close_rovent_goal_admission(&mut self) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    fn release_rovent_goal_state(&mut self) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    fn rovent_goal_consumption_recorded(
        &self,
        _: &RoleTicketEnvelope,
        _: &ReviewView,
    ) -> Result<bool, AdaptiveManagerError> {
        Ok(false)
    }
    fn validate_collection_admission(
        &self,
        _: &RoventCollectionRootAdmission,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual Rovent collection frontier",
        ))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn validate_collection_admission_v2(
        &self,
        _: &RoventCollectionRootAdmissionV2,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual V2 collection frontier",
        ))
    }
    /// A driver must not enable V2 observation until all acceptance and
    /// consumption paths and the external teacher lifecycle are wired.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn collection_v2_observation_complete(&self) -> bool {
        false
    }
    fn rovent_root_budget(&self) -> Result<Option<Digest>, AdaptiveManagerError> {
        Ok(None)
    }
    /// Read the same current controller; no caller-supplied stop/receipt can
    /// establish a capacity boundary for another root or a legacy driver.
    fn rovent_public_record_capacity_closed(&self) -> Result<bool, AdaptiveManagerError> {
        Ok(false)
    }
    /// The exact public review-bundle producer stopped this same controller.
    /// Other index/view/sequence/owner failures do not establish this boundary.
    fn rovent_public_review_index_capacity_closed(&self) -> Result<bool, AdaptiveManagerError> {
        Ok(false)
    }
    fn rovent_publication_capacity_closed(&self) -> Result<bool, AdaptiveManagerError> {
        Ok(self.rovent_public_record_capacity_closed()?
            || self.rovent_public_review_index_capacity_closed()?)
    }
    fn close_rovent_role_call_budget(
        &mut self,
        _: u64,
        _: u64,
    ) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    fn rovent_frame_admitted(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: FrontierFrameKind,
        _: ClockDomain,
        _: MonotonicTick,
    ) -> Result<bool, AdaptiveManagerError> {
        Ok(false)
    }
    fn rovent_lane_selected(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CoordinatorWork<ReviewView, Q>,
        _: ClockDomain,
        _: MonotonicTick,
    ) -> Result<bool, AdaptiveManagerError> {
        Ok(false)
    }
    fn rovent_fresh_accepted(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &rz_search::pals::coordinator::AcceptedRoleStep,
        _: &RoleOutput,
        _: Option<TicketExecutionEvidence>,
        _: ClockDomain,
        _: MonotonicTick,
    ) -> Result<bool, AdaptiveManagerError> {
        Ok(false)
    }
    fn rovent_rules_endpoint(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: ClockDomain,
        _: MonotonicTick,
    ) -> Result<bool, AdaptiveManagerError> {
        Ok(false)
    }
    fn prepare_rovent_goal_boundary(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
    ) -> Result<
        Option<rz_search::pals::engine::adaptive_port::AdaptiveGoalBoundary<Q>>,
        AdaptiveManagerError,
    > {
        Ok(None)
    }
    fn rovent_root_round_sealed(
        &mut self,
        _: Digest,
        _: ClockDomain,
        _: MonotonicTick,
    ) -> Result<bool, AdaptiveManagerError> {
        Ok(false)
    }
    fn validate_registered_native_ledger(
        &self,
        _: &CpuLedgerIdentity,
        _: &rz_search::cpu_worker::CpuLedgerRegistrationObservation,
        _: Digest,
        _: Instant,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no original registered native ledger",
        ))
    }
    fn review_lease_admission(
        &self,
    ) -> Result<Option<AdaptiveReviewLeaseAdmission>, AdaptiveManagerError> {
        Ok(None)
    }
    /// Optional common-node controller. Legacy PALS has no such owner.
    fn frontier_native_capacity(
        &mut self,
        _: u64,
        _: ClockDomain,
        _: MonotonicTick,
    ) -> Result<Option<u64>, AdaptiveManagerError> {
        Ok(None)
    }
    fn frontier_native_reserve(
        &mut self,
        _: &CpuJobEnvelope,
        _: SliceGrant,
        _: ClockDomain,
        _: MonotonicTick,
    ) -> Result<Option<NativeReservation>, AdaptiveManagerError> {
        Ok(None)
    }
    fn frontier_native_settled(
        &mut self,
        _: &NativeReservation,
        _: &CpuSliceReport,
        _: &SettledCpuSlice,
    ) -> Result<NativeCloseReason, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid("native frontier absent"))
    }
    fn frontier_native_released(
        &mut self,
        _: &NativeReservation,
        _: &ReleasedCpuReservation,
    ) -> Result<NativeCloseReason, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid("native frontier absent"))
    }
    fn frontier_native_unknown(
        &mut self,
        _: &NativeReservation,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid("native frontier absent"))
    }
    fn reciprocal_parent(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuJobEnvelope,
        _: ReciprocalQuestionId,
        _: ExecutionMode,
        _: Digest,
    ) -> Result<QuestionEnvelope, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no reciprocal CPU parent consumer",
        ))
    }
    fn register_cpu_reciprocal_scope(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuJobEnvelope,
        _: &QuestionEnvelope,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no existing reciprocal R scope consumer",
        ))
    }
    fn prepare_cpu_reciprocal(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuReciprocalRequest,
        _: ReciprocalQuestionId,
        _: ExecutionMode,
        _: Digest,
    ) -> Result<AdaptiveReciprocalQuestion<Q>, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no reciprocal native input consumer",
        ))
    }
    fn cpu_reciprocal_wait(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no reciprocal wait receipt consumer",
        ))
    }
    fn discard_cpu_reciprocal(
        &mut self,
        _: ReciprocalQuestionId,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no reciprocal pin retirement consumer",
        ))
    }
    // The original work, accepted head/output, question identity, mode, budget
    // and recipient remain independent checked provenance at this boundary.
    #[allow(clippy::too_many_arguments)]
    fn prepare_accepted_role_reciprocal(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &rz_search::pals::coordinator::AcceptedRoleStep,
        _: &RoleOutput,
        _: ReciprocalQuestionId,
        _: ExecutionMode,
        _: Digest,
        _: SearchRecipient,
    ) -> Result<Option<AdaptivePeerQuestion<Q>>, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no accepted-role peer question producer",
        ))
    }
    fn accept_accepted_role_source(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &rz_search::pals::coordinator::AcceptedRoleStep,
        _: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no staged accepted role source consumer",
        ))
    }
    #[cfg(feature = "pals-adaptive-diagnostics")]
    fn diagnostic_accepted_source_revision(
        &self,
        _: &rz_search::pals::coordinator::AcceptedRoleStep,
    ) -> Option<u64> {
        None
    }
    fn consume_accepted_role_reciprocal(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &rz_search::pals::coordinator::AcceptedRoleStep,
        _: &RoleOutput,
        _: Option<TicketExecutionEvidence>,
    ) -> Result<Digest, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no accepted peer opinion consumer",
        ))
    }
    fn consume_reciprocal_cpu(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuSliceReport,
    ) -> Result<Digest, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual reciprocal CPU answer consumer",
        ))
    }
    fn consume_cpu_reciprocal(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &rz_search::pals::coordinator::AcceptedRoleStep,
        _: &RoleOutput,
        _: Option<TicketExecutionEvidence>,
        _: &[(CpuReciprocalRequest, ReciprocalQuestionId)],
    ) -> Result<Vec<CpuReciprocalReply>, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no accepted reciprocal reply consumer",
        ))
    }
    fn discard_accepted_role_reciprocal(
        &mut self,
        _: ReciprocalQuestionId,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no accepted peer pin retirement consumer",
        ))
    }
    /// Immutable root registration, separate from per-frame ceilings, CPU
    /// credit, semantic consumption and physical NN/cache accounting.
    fn root_role_call_limit(&self) -> Result<Option<u64>, AdaptiveManagerError> {
        Ok(None)
    }
    fn collection_before_dispatch(
        &mut self,
        _: &RoleTicketEnvelope,
        _: &FrameInput<ReviewView, Q>,
    ) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    /// Poll an independent archive owner; never perform its I/O here.
    fn poll_auxiliary(&mut self) -> Result<bool, AdaptiveManagerError> {
        Ok(false)
    }
    fn semantic_owner_available(&self) -> bool {
        true
    }
    fn prepare_semantic_commit(&mut self) -> Result<bool, AdaptiveManagerError> {
        Ok(self.semantic_owner_available())
    }
    fn begin_auxiliary_shutdown(&mut self, _: Instant) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    /// Close the private archive command loop only after all physical search
    /// owners have joined and their raw receipts were settled by the driver.
    fn finish_auxiliary_shutdown(&mut self) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    fn auxiliary_shutdown_confirmed(&self) -> bool {
        true
    }
    /// Exact endpoint Rules proof, never a fabricated worker completion.
    fn short_circuit(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
    ) -> Result<Option<CheckedRulesStep<Q>>, AdaptiveManagerError> {
        Ok(None)
    }
    fn ranking_published(
        &mut self,
        ranking: &ProposerRankingPublished,
    ) -> Result<Vec<CandidateFrame<Q>>, AdaptiveManagerError>;
    fn root_candidate_admitted(
        &mut self,
        _: FrameId,
        _: Move16,
        _: &FrameInput<ReviewView, Q>,
    ) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    fn next_root_round_input(
        &mut self,
    ) -> Result<Option<FrameInput<ReviewView, Q>>, AdaptiveManagerError> {
        Ok(None)
    }
    fn next_fixed_root_candidate(
        &mut self,
    ) -> Result<Option<CandidateFrame<Q>>, AdaptiveManagerError> {
        Ok(None)
    }
    /// Search-issued deferred recheck retains the exact parent/source seal.
    /// It is not interchangeable with a caller-created Repair frame.
    fn next_fixed_iterative_repair(
        &mut self,
    ) -> Result<Option<AdaptiveDeferredRepair<Q>>, AdaptiveManagerError> {
        Ok(None)
    }
    /// Validation only: no record/index/consumption change before native ack.
    fn validate_role(
        &self,
        work: &CoordinatorWork<ReviewView, Q>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError>;
    fn validate_validator(
        &self,
        _: &FrameInput<ReviewView, Q>,
        _: &RoleTicketEnvelope,
        _: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no Collection V question consumer",
        ))
    }
    fn validator_accepted(
        &mut self,
        _: &FrameInput<ReviewView, Q>,
        _: AcceptedValidator,
    ) -> Result<AdaptiveValidatorActionPlan<Q>, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual V action consumer",
        ))
    }
    fn register_supported_validator_action(
        &mut self,
        _: AdaptiveValidatorAction,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no registered action collection consumer",
        ))
    }
    fn validator_input(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &[CpuWorkerProfile],
        _: CpuEndpointSelection,
        _: FrameBudget,
    ) -> Result<Option<FrameInput<ReviewView, Q>>, AdaptiveManagerError> {
        Ok(None)
    }
    fn validator_action_admitted(
        &mut self,
        _: ValidatorActionId,
        _: &rz_search::pals::coordinator::ValidatorActionAdmission,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual V action binding",
        ))
    }
    fn commit_role(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        physical: Option<TicketExecutionEvidence>,
    ) -> Result<CheckedStep<Q>, AdaptiveManagerError>;
    /// Use the manager's already resolved endpoint. The driver must not choose
    /// the Collection teacher again after native R admission/preparation.
    fn cpu_question(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        profiles: &[CpuWorkerProfile],
        external: Option<&ExternalCpuConfig>,
        endpoint: CpuEndpoint,
        physical_budget_exhausted: bool,
    ) -> Result<CpuQuestion, AdaptiveManagerError>;
    fn commit_cpu(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        report: &CpuSliceReport,
    ) -> Result<CheckedStep<Q>, AdaptiveManagerError>;
    /// Account a retained/late slice in the Store namespace without moving
    /// the refinement stage or granting current search consumption.
    fn cpu_slice_observed(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    /// Original known terminal with no CPU score. This is a cleanup consumer,
    /// never an ordinary commit/after_cpu or a successful reciprocal response.
    fn cpu_no_value_terminated(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuSliceReport,
    ) -> Result<Digest, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "CPU no-value consumer is unavailable",
        ))
    }
    fn cpu_partial_continuation_eligible(
        &self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuSliceReport,
    ) -> bool {
        false
    }
    /// The original CPU question outlives its first feedback transition. Its
    /// later actual receipts may update only that question's evidence/custody.
    fn cpu_detached_slice_observed(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        report: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        self.cpu_slice_observed(work, report)
    }
    fn cpu_token_discarded(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuJobTermination,
    ) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    fn cpu_admitted(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuJobEnvelope,
        _: &SliceGrant,
    ) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    fn cpu_rejected(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuJobEnvelope,
        _: &CpuAdmissionRejection,
    ) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    fn external_admitted(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &ExternalCpuJobEnvelope,
        _: &ExternalSliceGrant,
        _: ClockDomain,
        _: MonotonicTick,
    ) -> Result<CheckedExternalCpuBinding, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no checked external namespace bridge",
        ))
    }
    fn external_rejected(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &ExternalCpuJobEnvelope,
        _: &ExternalSliceGrant,
    ) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    fn commit_external(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        report: &ExternalWorkerReport,
    ) -> Result<CheckedStep<Q>, AdaptiveManagerError>;
    fn external_report_observed(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &ExternalWorkerReport,
    ) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    /// Optional collection evidence comes from the actual external owner,
    /// never from a caller-created Job/Report success flag.
    fn external_teacher_submitted(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &ExternalJob,
        _: &ExternalTeacherDispatch,
    ) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    fn external_teacher_completed(
        &mut self,
        _: &ExternalTeacherDispatch,
        _: &ExternalWorkerCompletion,
        _: AdaptiveTeacherTargetEligibility,
    ) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    fn publish_closed(&mut self, closed: &ClosedFrame) -> Result<(), AdaptiveManagerError>;
    fn role_consumed(
        &mut self,
        ticket: &RoleTicketEnvelope,
        view: &Arc<ReviewView>,
    ) -> Result<(), AdaptiveManagerError>;
    fn cpu_progress(&mut self, _: &CpuWorkerProgress) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    /// The original live question accompanies owner-issued progress. Legacy
    /// drivers retain their old callback; actual drivers can check its context.
    fn cpu_progress_for(
        &mut self,
        _: &CoordinatorWork<ReviewView, Q>,
        progress: &CpuWorkerProgress,
    ) -> Result<(), AdaptiveManagerError> {
        self.cpu_progress(progress)
    }
    fn cpu_progress_request_eligible(
        &self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CpuJobEnvelope,
        _: &SliceGrant,
    ) -> bool {
        false
    }
    /// Read actual single-writer consumption after a later semantic failure.
    /// The default carries no evidence; the concrete Search port verifies the
    /// original authority/ticket/view without consuming or updating it again.
    fn role_consumption_recorded(
        &self,
        _: &RoleTicketEnvelope,
        _: &ReviewView,
    ) -> Result<bool, AdaptiveManagerError> {
        Ok(false)
    }
    /// Legacy hook retained for callers. A scalar alone is not authority for
    /// the actual manager's calibrated S-policy prediction.
    fn yield_feedback_gain(&self, _: &CoordinatorWork<ReviewView, Q>) -> Option<Duration> {
        None
    }
    /// A scheduling forecast, not realized saving, witness, or V utility.
    /// The concrete driver must validate its original coherent publication
    /// and a downstream consumer before querying the observed cost model.
    fn predict_yield_feedback(
        &self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &CostEstimator,
        _: CostClass,
        _: Duration,
        _: Duration,
    ) -> Option<FeedbackWaitPrediction> {
        None
    }
    fn external_feedback_eligible(
        &self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &ExternalWorkerProgress,
    ) -> bool {
        false
    }
    fn validate_rules_pair(
        &self,
        _: &CoordinatorWork<ReviewView, Q>,
        _: &AdaptiveRulesPairProof,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no exact Rules-pair proof consumer",
        ))
    }
}
/// The actual manager attests whether the submitted question was still live
/// when its private owner terminal was received. This is separate from owner
/// physical completion, Store commit and search consumption.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveTeacherTargetEligibility {
    job: ExternalCpuJobEnvelope,
    grant: ExternalSliceGrant,
    eligible: bool,
}
impl AdaptiveTeacherTargetEligibility {
    pub fn matches_submission(&self, submitted: &ExternalTeacherDispatch) -> bool {
        self.job == submitted.job() && self.grant == submitted.grant()
    }
    pub fn target_eligible(&self) -> bool {
        self.eligible
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveManagerRegistration {
    pub authority: SearchAuthority,
    pub hard_deadline: Instant,
    pub token_budget: u16,
    pub policy: CoordinatorPolicy,
    pub first_request_sequence: u64,
    pub native_cpu_node_limit: u64,
    pub endpoints: CpuEndpointSelection,
    pub controller: ControllerConfig,
    /// Full identity of the loaded device/runtime, not a guessed GPU ordinal.
    pub device_digest: Digest,
    pub cpu_device_digest: Digest,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AdaptiveManagerCounters {
    pub native_preparations: u64,
    pub submitted_role_tickets: u64,
    pub physical_role_completions: u64,
    pub completed_nn_inputs: u64,
    pub nn_input_count_unknown: u64,
    pub consumed_role_outputs: u64,
    /// Physical inputs acknowledged by the actual native owner as accepted.
    pub accepted_nn_inputs: u64,
    pub accepted_nn_input_count_unknown: u64,
    /// Physical inputs belonging to a Search-consumed original output. Native
    /// acceptance can precede a failed Store commit, so these stay separate.
    pub consumed_nn_inputs: u64,
    pub consumed_nn_input_count_unknown: u64,
    /// Known zero-NN consumption; no cache-hit claim is inferred from zero.
    pub zero_nn_outputs_consumed: u64,
    pub unused_role_results: u64,
    pub canceled_preparations: u64,
    pub cpu_slice_receipts: u64,
    pub cpu_progress_publications: u64,
    pub cpu_yields: u64,
    pub cpu_yield_requests: u64,
    pub cpu_yield_acknowledgements: u64,
    pub cpu_natural_completions: u64,
    /// Actual V ticket admission, independent of physical NN/cache counts.
    pub validator_ticket_admissions: u64,
    pub foreign_reports: u64,
    pub foreign_reported_nodes: u64,
    pub foreign_node_count_unknown: u64,
    /// Foreign stop/drain controls, never exact native stack resumptions.
    pub foreign_yield_requests: u64,
    pub foreign_yield_acknowledgements: u64,
    pub rejected_events: u64,
    pub notification_waits: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagerPhase {
    Running,
    Draining,
    Quarantined,
    Finished,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveCloseMode {
    ProcessShutdown,
    RootQuiesce,
}

/// Diagnostic data only. It grants no ticket, admission, completion or control
/// authority. Native queues and physical execution can change independently.
#[cfg(feature = "search-work-receipts")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptivePendingQuestionSnapshot {
    pub authority: SearchAuthority,
    pub phase: ManagerPhase,
    pub sampled_at: MonotonicTick,
    pub original_deadline: Deadline,
    pub questions: Vec<AdaptivePendingQuestionRow>,
}

/// One actual retained manager binding, not a counter delta or native phase.
/// A ticket-less row is a preparation reservation, not a submitted question.
#[cfg(feature = "search-work-receipts")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptivePendingQuestionRow {
    pub preparation: PreparationId,
    pub frame: Option<FrameId>,
    pub stage: RefinementStage,
    pub question: Digest,
    pub view: ReviewViewSeal,
    pub retained_view_digest: Digest,
    pub retained_view_seal_digest: Digest,
    pub selected_record_revisions: Vec<u64>,
    pub ticket: Option<RoleTicketEnvelope>,
    /// Last phase processed by the manager, not the native owner's live state.
    pub registry_observed_phase: Option<rz_search::pals::coordinator::TicketPhase>,
    pub dispatch_pending: bool,
    pub completion_received: bool,
    pub consumption_recorded: bool,
}

/// Observation loss never enters AdaptiveManagerError or changes Search work.
#[cfg(feature = "search-work-receipts")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptivePendingSnapshotUnavailable {
    Capacity,
    ClockUnavailable,
    ViewSealMismatch,
    RegistryBindingMismatch,
}

/// One immutable root closure observation, issued from actual settled owners.
/// The raw lifecycle counters retain their producer's scope: ticket dispatch
/// or zero-NN consumption never becomes an NN forward/cache-hit assertion.
/// Consumers aggregate each authority once and retain unavailable observations.
pub struct AdaptiveRootObservationSnapshot {
    authority: SearchAuthority,
    close_mode: AdaptiveCloseMode,
    manager: AdaptiveManagerCounters,
    native_accounting: AdaptiveNativeAccountingSnapshot,
    role_tickets: RoleTicketCumulativeObservation,
    history_peak: CoordinatorHistoryUsage,
    validator_history_peak: ValidatorActionHistoryUsage,
    cpu_credit_limit: u64,
    cpu_spent: u64,
    cpu_reserved: u64,
    validator_loaded: bool,
    receipt: AdaptiveShutdownReceipt,
}
impl AdaptiveRootObservationSnapshot {
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn close_mode(&self) -> AdaptiveCloseMode {
        self.close_mode
    }
    pub fn manager_counters(&self) -> AdaptiveManagerCounters {
        self.manager
    }
    pub fn native_accounting(&self) -> AdaptiveNativeAccountingSnapshot {
        self.native_accounting
    }
    pub fn role_ticket_accounting(&self) -> RoleTicketCumulativeObservation {
        self.role_tickets
    }
    pub fn history_peak(&self) -> CoordinatorHistoryUsage {
        self.history_peak
    }
    pub fn validator_history_peak(&self) -> ValidatorActionHistoryUsage {
        self.validator_history_peak
    }
    pub fn cpu_credit_limit(&self) -> u64 {
        self.cpu_credit_limit
    }
    pub fn cpu_spent(&self) -> u64 {
        self.cpu_spent
    }
    pub fn cpu_outstanding_reserved(&self) -> u64 {
        self.cpu_reserved
    }
    /// Registered manager profile only. Actual loaded/called native roles are
    /// observed separately by the retained session owner.
    pub fn validator_loaded(&self) -> bool {
        self.validator_loaded
    }
    pub fn completed_nn_inputs(&self) -> Option<u64> {
        (self.manager.nn_input_count_unknown == 0).then_some(self.manager.completed_nn_inputs)
    }
    pub fn accepted_nn_inputs(&self) -> Option<u64> {
        (self.manager.accepted_nn_input_count_unknown == 0)
            .then_some(self.manager.accepted_nn_inputs)
    }
    pub fn cache_evaluations_accepted(&self) -> Option<u64> {
        self.native_accounting.cache_evaluations_accepted()
    }
    /// Requires original actual command observations for every mailbox
    /// dispatch. This does not use the registered V profile or ticket count as
    /// a physical-call readback.
    pub fn validator_physically_called(&self) -> Option<bool> {
        self.native_accounting
            .validator_called_for_dispatches(self.role_tickets.service_dispatched)
    }
    pub fn shutdown_receipt(&self) -> &AdaptiveShutdownReceipt {
        &self.receipt
    }
}

/// Issued by this actual manager only after one gated action and its root
/// physical fence. Public getters are observations, not a caller constructor.
pub struct ExclusiveActionClosureProof {
    authority: SearchAuthority,
    source: RoleTicketEnvelope,
    action: ValidatorActionId,
    started: Instant,
    finished: Instant,
    baseline: AdaptiveManagerCounters,
    final_counters: AdaptiveManagerCounters,
    question: ExclusiveQuestionSeal,
}
#[derive(Clone, Copy)]
struct ExclusiveQuestionSeal {
    parent_question: Digest,
    parent_view: ReviewViewSeal,
    parent_input: Digest,
    parent_public_context: Digest,
    frozen_context: Digest,
    support: [bool; 7],
}
impl ExclusiveActionClosureProof {
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn source(&self) -> &RoleTicketEnvelope {
        &self.source
    }
    pub fn action(&self) -> ValidatorActionId {
        self.action
    }
    pub fn started(&self) -> Instant {
        self.started
    }
    pub fn finished(&self) -> Instant {
        self.finished
    }
    pub fn wall_cost(&self) -> Duration {
        self.finished.saturating_duration_since(self.started)
    }
    pub fn baseline_counters(&self) -> AdaptiveManagerCounters {
        self.baseline
    }
    pub fn final_counters(&self) -> AdaptiveManagerCounters {
        self.final_counters
    }
    pub fn cost_scope(&self) -> &'static str {
        "exclusive_v_decision_and_single_action_with_required_drain"
    }
    pub fn parent_question(&self) -> Digest {
        self.question.parent_question
    }
    pub fn parent_view(&self) -> ReviewViewSeal {
        self.question.parent_view
    }
    pub fn parent_input(&self) -> Digest {
        self.question.parent_input
    }
    pub fn parent_public_context(&self) -> Digest {
        self.question.parent_public_context
    }
    /// Content seal issued from the original Search-owned frozen question,
    /// independent of this isolated root's process-local execution identifiers.
    pub fn frozen_context_digest(&self) -> Digest {
        self.question.frozen_context
    }
    pub fn action_support_mask(&self) -> [bool; 7] {
        self.question.support
    }
}
struct ExclusiveActionScope {
    parent: FrameId,
    started: Instant,
    baseline: AdaptiveManagerCounters,
    source: Option<RoleTicketEnvelope>,
    action: Option<ValidatorActionId>,
    frame: Option<FrameId>,
    completed: bool,
    question: Option<ExclusiveQuestionSeal>,
}
impl ExclusiveActionScope {
    fn allows(&self, frame: FrameId) -> bool {
        if self.action.is_some() {
            self.frame == Some(frame) && !self.completed
        } else {
            self.parent == frame
        }
    }
}

/// Archive owner may close ONLY after this receipt's predicate is true. A CPU
/// joined thread alone never refunds unknown usage or proves the GPU fence.
#[derive(Clone, Debug)]
pub struct AdaptiveShutdownReceipt {
    pub authority: SearchAuthority,
    pub native_owner_joined: bool,
    /// A root fence keeps the actual model/session owner alive; it is never
    /// recorded as an OS thread join or process-level session shutdown.
    pub native_owner_quiesced: bool,
    pub native_physical_shutdown_confirmed: bool,
    pub own_cpu_joined: usize,
    pub expected_own_cpu_owners: usize,
    pub own_cpu_panicked: usize,
    pub external_shutdown_confirmed: Option<bool>,
    pub external_owner_quiesced: Option<bool>,
    pub auxiliary_shutdown_confirmed: bool,
    pub native_cpu_spent: u64,
    pub native_cpu_reserved: u64,
    pub pending_tickets: usize,
    pub quarantined: bool,
    pub request_sequence_high_water: u64,
    pub counters: AdaptiveManagerCounters,
    pub native_cpu_thread_observation: Option<
        Result<rz_runtime::adaptive::CpuThreadReadback, rz_runtime::adaptive::CpuResourceError>,
    >,
    pub own_cpu_thread_observations: Vec<
        Option<
            Result<rz_runtime::adaptive::CpuThreadReadback, rz_runtime::adaptive::CpuResourceError>,
        >,
    >,
    root_fence: Option<NativeRootFence>,
    external_root_fence: Option<ExternalRootFence>,
    external_owner: Option<CpuWorkerId>,
}
impl AdaptiveShutdownReceipt {
    fn external_closed(&self) -> bool {
        match self.external_owner {
            None => {
                self.external_shutdown_confirmed.is_none()
                    && self.external_owner_quiesced.is_none()
                    && self.external_root_fence.is_none()
            }
            Some(owner) => {
                self.external_shutdown_confirmed == Some(true)
                    || (self.external_owner_quiesced == Some(true)
                        && self.external_root_fence.is_some_and(|fence| {
                            fence.authority() == self.authority
                                && fence.owner() == owner
                                && fence.permits_reuse()
                        }))
            }
        }
    }
    pub fn permits_semantic_closure(&self) -> bool {
        ((self.native_owner_joined && self.native_physical_shutdown_confirmed)
            || (self.native_owner_quiesced
                && self.root_fence.is_some_and(|fence| {
                    fence.authority() == self.authority && fence.permits_reuse()
                })))
            && self.own_cpu_panicked == 0
            && self.own_cpu_joined == self.expected_own_cpu_owners
            && self.native_cpu_reserved == 0
            && self.pending_tickets == 0
            && !self.quarantined
            && self.auxiliary_shutdown_confirmed
            && self.external_closed()
    }
    pub fn permits_archive_close(&self) -> bool {
        self.native_owner_joined
            && self.native_physical_shutdown_confirmed
            && self.own_cpu_panicked == 0
            && self.own_cpu_joined == self.expected_own_cpu_owners
            && self.native_cpu_reserved == 0
            && self.pending_tickets == 0
            && !self.quarantined
            && self.auxiliary_shutdown_confirmed
            && (self.external_owner.is_none() || self.external_shutdown_confirmed == Some(true))
    }
    pub fn permits_root_reuse(&self) -> bool {
        self.native_owner_quiesced && !self.native_owner_joined && self.permits_semantic_closure()
    }
}

struct RoleBinding<Q, O> {
    preparation: PreparationId,
    prepared_observed: bool,
    cancellation_requested: bool,
    work: Option<CoordinatorWork<ReviewView, Q>>,
    input: FrameInput<ReviewView, Q>,
    ticket: Option<RoleTicketEnvelope>,
    // Registry admission spends the reservation before subsequent fallible
    // frame binding. Keep that charge even if the original preparation must
    // drain without ever becoming an executable native ticket.
    admission_charged: bool,
    dispatch_pending: bool,
    collection_registered: bool,
    rovent_collection_registered: bool,
    physical: Option<TicketExecutionEvidence>,
    prepared_at: Instant,
    preparation_elapsed: Option<Duration>,
    queue_admitted_at: Option<Instant>,
    queue_wait: Option<Duration>,
    dispatched_at: Option<Instant>,
    service_completed: bool,
    class: Option<CostClass>,
    delivered: Option<Arc<O>>,
    ack_accepted: Option<bool>,
    ack_dispatched: bool,
    revoked_acceptance: Option<RoleFailure>,
    native_acceptance_recorded: bool,
    consumption_recorded: bool,
    // A failed owner readback cannot establish whether search consumed the
    // accepted result. Retain its binding rather than retry or count it unused.
    consumption_unknown: bool,
    purpose: RolePurpose,
    reciprocal_acceptance: Option<rz_search::pals::coordinator::AcceptedRoleStep>,
    goal_local: Option<GoalLocalInput<Q>>,
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum RolePurpose {
    RootRanking,
    Refinement,
    Validator,
    Reciprocal,
    RoventRefinement,
    RoventFresh,
    RoventGoalLocal,
}
struct NativeAck {
    preparation: PreparationId,
    ticket: RoleTicketEnvelope,
    accepted: bool,
    permit: Option<NativeAcceptancePermit>,
}
struct OwnBinding<Q> {
    work: CoordinatorWork<ReviewView, Q>,
    envelope: CpuJobEnvelope,
    controls: Arc<CpuControlMailbox>,
    grant: SliceGrant,
    started_at: Instant,
    class: CostClass,
    resume_pending: bool,
    feedback_forwarded: bool,
    progress_sequence: u64,
    progress_since_slice: bool,
    progress_published_at: Option<Instant>,
    owner_run_started_at: Option<Instant>,
    resume_cost_observed_execution: Option<ExecutionId>,
    last_yield_ack: u64,
    requested_yield: Option<(u64, Instant)>,
    pending_slice: Option<CpuSliceReport>,
    // A prior paused stack may terminate before an already-enqueued Resume
    // returns its definite no-work report. Retain both original obligations.
    queued_resume_termination: Option<CpuJobTermination>,
    validator_reviewed_execution: Option<ExecutionId>,
    reciprocal: Option<reciprocal::OwnReciprocalCustody>,
    // Linear custody of the same grant in the root's common-node controller.
    // Its accepted native-ledger receipt is kept until the semantic owner
    // returns; a cold Store never turns a reservation into zero work.
    native_frontier: Option<NativeReservation>,
    nnuec: Option<Arc<nnuec::NnuecJobCustody>>,
    semantic_settlement: Option<SettledCpuSlice>,
}
struct OwnSlot<Q> {
    worker: Option<CpuWorker>,
    binding: Option<OwnBinding<Q>>,
    joined: bool,
    panicked: bool,
    settled_execution: Option<ExecutionId>,
    recovered_engine: Option<CpuEngine>,
    ready_readback: Option<
        Result<rz_runtime::adaptive::CpuThreadReadback, rz_runtime::adaptive::CpuResourceError>,
    >,
    last_value_installation: Option<rz_search::cpu_worker::ValueInstallReceipt>,
    nnuec_retirement: Option<rz_search::cpu_worker::RecoveredNnuecEngine>,
}
struct ForeignBinding<Q> {
    work: CoordinatorWork<ReviewView, Q>,
    job: ExternalJob,
    teacher_dispatch: ExternalTeacherDispatch,
    started_at: Instant,
    class: CostClass,
    progress_sequence: u64,
    observed_progress: Option<ExternalWorkerProgressObservation>,
    requested_yield: Option<(u64, Instant)>,
}
/// An advice is bound to the exact physical execution it evaluated. This
/// private S-policy selection is not CPU evidence or a utility target.
#[derive(Clone, Copy)]
enum YieldSelection {
    Own {
        index: usize,
        execution: ExecutionId,
        prediction: FeedbackWaitPrediction,
    },
    External {
        scope: ExternalWorkerScope,
        prediction: ExternalFeedbackWaitPrediction,
    },
}
impl YieldSelection {
    fn gain(self) -> Duration {
        match self {
            Self::Own { prediction, .. } => prediction.gain_before_return_cost(),
            Self::External { prediction, .. } => prediction.gain_before_return_cost(),
        }
    }
    fn return_cost(self) -> rz_runtime::adaptive::CostEstimate {
        match self {
            Self::Own { prediction, .. } => prediction.return_cost(),
            Self::External { prediction, .. } => prediction.return_cost(),
        }
    }
}
/// Failed processing retains the actual completion and cost/physical evidence;
/// keeping an Arc is not a second model output or a new acceptance authority.
pub struct RetainedNativeCompletion<O> {
    pub completion: RoleCompletion<Arc<O>>,
    pub evidence: Option<TicketExecutionEvidence>,
    pub owner_run: TicketOwnerRunWitness,
    pub owner_service_elapsed: Option<Duration>,
}
// The bounded deferred queue retains original owners and terminal obligations.
// Inline variants avoid allocating error-recovery boxes per deferred receipt.
#[allow(clippy::large_enum_variant)]
enum DeferredCommit<Q> {
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    ProfiledExternal {
        binding: profiled_external_dispatch::ProfiledExternalBinding<Q>,
        packet: Arc<rz_search::external_cpu::ProfiledExternalCompletion>,
    },
    Role {
        ticket: RoleTicketEnvelope,
        accepted: bool,
    },
    Own {
        binding: OwnBinding<Q>,
        report: CpuSliceReport,
    },
    OwnNoValue {
        binding: OwnBinding<Q>,
        report: CpuSliceReport,
        // A publication retry retains the actual committed Unknown closure;
        // it must not attempt to retire the already-removed frame twice.
        closed: Option<ClosedFrame>,
    },
    External {
        binding: ForeignBinding<Q>,
        report: ExternalWorkerReport,
    },
    Closed(ClosedFrame),
    OwnRaw {
        work: CoordinatorWork<ReviewView, Q>,
        report: CpuSliceReport,
    },
    DetachedOwn {
        work: CoordinatorWork<ReviewView, Q>,
        report: CpuSliceReport,
    },
    TokenDiscard {
        work: CoordinatorWork<ReviewView, Q>,
        termination: CpuJobTermination,
    },
    ReciprocalWait {
        work: CoordinatorWork<ReviewView, Q>,
        report: CpuSliceReport,
    },
    ReciprocalDiscard {
        question: ReciprocalQuestionId,
    },
    PeerDiscard {
        question: ReciprocalQuestionId,
    },
}
pub struct ExternalManagerWorker {
    pub worker: ExternalCpuWorker,
    pub config: ExternalCpuConfig,
    pub owner: CpuWorkerId,
}

/// Ownership returned only after a real root fence. The same native backend
/// and CPU engine/TT values are moved, never reconstructed from metadata.
pub struct PersistentCpuOwner {
    pub profile: CpuWorkerProfile,
    pub engine: CpuEngine,
}
pub struct AdaptiveSequenceContinuation {
    authority: SearchAuthority,
    next_request: u64,
    next_job: u64,
    next_execution: u64,
    next_control: u64,
}
impl AdaptiveSequenceContinuation {
    pub fn first_request_sequence(&self) -> u64 {
        self.next_request
    }
    pub fn previous_authority(&self) -> SearchAuthority {
        self.authority
    }
}
pub struct AdaptivePersistentParts<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    native: TicketedNativeService<ReviewView, Q, B>,
    cpu: Vec<PersistentCpuOwner>,
    external: Option<ExternalManagerWorker>,
    driver: D,
    continuation: AdaptiveSequenceContinuation,
    receipt: AdaptiveShutdownReceipt,
}
type PersistentComponents<Q, B, D> = (
    TicketedNativeService<ReviewView, Q, B>,
    Vec<PersistentCpuOwner>,
    Option<ExternalManagerWorker>,
    D,
    AdaptiveSequenceContinuation,
    AdaptiveShutdownReceipt,
);
impl<Q, B, D> AdaptivePersistentParts<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    pub fn receipt(&self) -> &AdaptiveShutdownReceipt {
        &self.receipt
    }
    pub fn into_components(self) -> PersistentComponents<Q, B, D> {
        (
            self.native,
            self.cpu,
            self.external,
            self.driver,
            self.continuation,
            self.receipt,
        )
    }
}
/// Failure returns the complete still-owned manager, including quarantined
/// session/buffers, rather than dropping ownership on a failed reuse claim.
pub struct AdaptivePersistentRejection<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    pub error: AdaptiveManagerError,
    pub manager: AdaptiveExecutionManager<Q, B, D>,
}
/// A session already owns the original manager in a Box.  A rejected transfer
/// returns that very allocation instead of materializing the whole manager on
/// the caller's stack or changing its physical ownership.
pub(crate) struct AdaptiveBoxedPersistentRejection<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    pub error: AdaptiveManagerError,
    pub manager: Box<AdaptiveExecutionManager<Q, B, D>>,
}
/// Admission owns the actual process/session and engines. Every failed check
/// returns this complete source bundle rather than dropping physical owners.
pub struct AdaptiveManagerSources<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    pub native: TicketedNativeService<ReviewView, Q, B>,
    pub own_workers: Vec<CpuWorker>,
    pub external: Option<ExternalManagerWorker>,
    pub driver: D,
}
pub struct AdaptiveManagerAdmissionRejection<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    pub error: AdaptiveManagerError,
    pub sources: AdaptiveManagerSources<Q, B, D>,
    pub continuation: Option<AdaptiveSequenceContinuation>,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveCollectionV2ObserverStatus {
    AdmissionNotCalled,
    AdmissionFailed,
    AdmissionPanicked,
    AdmissionSucceededRecheckFailed,
    AdmissionSucceededRecheckPanicked,
    ManagerValidationPanicked,
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub(crate) struct AdaptiveCollectionV2Rejection<Q, O> {
    pub error: AdaptiveManagerError,
    pub admission: RoventCollectionRootAdmissionV2,
    pub observer: AdaptiveCollectionObserverBoxV2<Q, O>,
    /// Failed or panicked callbacks may have changed this original observer;
    /// the status does not grant permission to reuse it.
    pub observer_status: AdaptiveCollectionV2ObserverStatus,
    /// A panic payload can own resources. Keep it with the rejected observer.
    pub panic_payload: Option<Box<dyn std::any::Any + Send>>,
}

/// The actual single writer owns native service handles and private CPU workers.
/// tick is bounded and nonblocking. It does not execute inference/search or join
/// a live worker. Two pending role questions retain independent submitted views.
pub struct AdaptiveExecutionManager<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    registration: AdaptiveManagerRegistration,
    limits: AdaptiveExecutionLimits,
    clock: ContractSystemClock,
    cpu_clock: InstantClock,
    deadline: Deadline,
    native: TicketedNativeService<ReviewView, Q, B>,
    registry: RoleTicketRegistry<ReviewView, Arc<B::Output>>,
    machine: RefinementMachine<ReviewView, Q>,
    driver: D,
    role_call_limit: Option<u64>,
    collection_observer: Option<AdaptiveCollectionObserverBox<Q, B::Output>>,
    collection_admission: Option<RoventCollectionRootAdmission>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    collection_observer_v2: Option<AdaptiveCollectionObserverBoxV2<Q, B::Output>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    collection_v2_disabled: bool,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    collection_v2_disabled_first_ticket: Option<RoleTicketEnvelope>,
    /// A Busy capture may already hold an original staged reservation. Keep
    /// this ticket ahead of any later collection dispatch until it resolves.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    collection_v2_busy_ticket: Option<RoleTicketEnvelope>,
    /// The first collection-only failure is retained even if the bounded
    /// diagnostic history rolls over. It is not a native completion result.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    collection_v2_first_failure: Option<AdaptiveCollectionError>,
    /// A failed bypass must stop admission; retrying or treating it as a
    /// completed capture could drop the original input owner.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    collection_v2_disable_failed: bool,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    collection_admission_v2: Option<RoventCollectionRootAdmissionV2>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    collection_v2_settlement: Option<Result<(), AdaptiveCollectionError>>,
    cpu_semantic: Option<cpu_semantic::CpuSemanticState<Q>>,
    utility_route: Option<utility_route::UtilityRouteState<Q>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    utility_cohort: Option<utility_cohort::UtilityCohortState<Q>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    utility_intervention: Option<utility_intervention::UtilityInterventionState<Q>>,
    utility_route_sequence: u64,
    // Failed unique prepared-job cleanup retains the same job and grant. No
    // unverified rejection can refund credit or discard its capture custody.
    retained_cpu_jobs: Vec<(rz_search::cpu_worker::CpuJob, SliceGrant)>,
    roles: Vec<RoleBinding<Q, B::Output>>,
    acknowledgements: VecDeque<NativeAck>,
    own: Vec<OwnSlot<Q>>,
    external: Option<ExternalManagerWorker>,
    foreign: Option<ForeignBinding<Q>>,
    researcher: ResearcherResources,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    profiled_binding: Option<profiled_external_dispatch::ProfiledExternalBinding<Q>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    profiled_orphan: Option<profiled_external_source::ProfiledExternalManagerCompletion>,
    cpu_slice_diagnostic: OnceLock<AdaptiveCpuSliceDiagnostic>,
    #[cfg(feature = "pals-adaptive-diagnostics")]
    continuation_diagnostic: Option<ContinuationDiagnostic>,
    ready: ReadySelector<FrameId, Move16, Digest>,
    costs: CostEstimator,
    advised_yield: Option<YieldSelection>,
    /// Actual completed role namespaces retained only for this root. A lookup
    /// for the next Critic is a prospective S scope, not a submitted role or
    /// a cache observation. The bounded index never owns model/view buffers.
    observed_role_classes: Vec<(Role, CostClass)>,
    observation_sequence: u64,
    next_job: u64,
    next_execution: u64,
    next_control: u64,
    phase: ManagerPhase,
    native_finished: Option<bool>,
    native_joined: bool,
    native_root_fence: Option<NativeRootFence>,
    close_mode: AdaptiveCloseMode,
    native_root_quiesce_requested: bool,
    external_finished: Option<bool>,
    external_root_fence: Option<ExternalRootFence>,
    external_startup_observed: bool,
    counters: AdaptiveManagerCounters,
    native_accounting: AdaptiveNativeAccountingSnapshot,
    shutdown_until: Option<Instant>,
    cleanup_boundary: Option<CleanupBoundary>,
    effective_clock: Option<crate::rovent_ponder::RoventPonderClockHandle>,
    native_ready: bool,
    native_verifier_registered: bool,
    retained_native_permits: Vec<NativeAcceptancePermit>,
    retained_native_receipts: Vec<NativeAcceptanceReceipt>,
    failures: VecDeque<AdaptiveManagerError>,
    quarantine_rejections: Vec<RoleCompletion<Arc<B::Output>>>,
    retained_native_completions: Vec<RetainedNativeCompletion<B::Output>>,
    native_observation_failures: Vec<TicketNativeObservationFailure>,
    quarantined_cpu_receipts: Vec<CpuWorkerEvent>,
    quarantined_foreign_receipts: Vec<ExternalWorkerReport>,
    ready_ids: Vec<FrameId>,
    deferred: VecDeque<DeferredCommit<Q>>,
    native_frontier_events: VecDeque<frontier::NativeFrontierEvent>,
    review_leases: RoventReviewLeaseObservation,
    validator_waiting: Vec<FrameId>,
    root_started: bool,
    cpu_node_limit_reached: bool,
    root_ranking_proof: Option<ProposerRankingPublished>,
    exclusive_requested: bool,
    registered_action: Option<AdaptiveValidatorAction>,
    exclusive_scope: Option<ExclusiveActionScope>,
    finished_at: Option<Instant>,
    reciprocal_policy: Option<RoventReciprocalPolicy>,
    reciprocal_routing: Option<reciprocal::RoventQuestionRouting>,
    next_reciprocal_question: u64,
    peer_policy: Option<RoventPeerFollowupPolicy>,
    accepted_roles: Vec<reciprocal::AcceptedRoleCustody<Q, B::Output>>,
    peer_sources_admitted: u16,
    peer_observations: Vec<PeerFollowupObservation>,
    // A local duty is never represented by an invented refinement frame.
    // Keep the exact private source until native retirement or known cleanup.
    goal_prepared: Option<GoalLocalInput<Q>>,
    goal_continuation_pending: bool,
    nnuec: Option<nnuec::NnuecRootPreparation>,
    nnuec_pending: Option<nnuec::NnuecWorkPreparation<Q>>,
    nnuec_jobs: Vec<Arc<nnuec::NnuecJobCustody>>,
    nnuec_observations: Vec<NnuecPreparationRecord>,
    nnuec_observations_omitted: u64,
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    pub fn new(
        registration: AdaptiveManagerRegistration,
        clock: ContractSystemClock,
        native: TicketedNativeService<ReviewView, Q, B>,
        own_workers: Vec<CpuWorker>,
        external: Option<ExternalManagerWorker>,
        driver: D,
    ) -> Result<Self, AdaptiveManagerError> {
        Self::new_retaining(
            registration,
            clock,
            AdaptiveManagerSources {
                native,
                own_workers,
                external,
                driver,
            },
        )
        .map_err(|rejected| rejected.error)
    }
    // Rejected admission returns every original process/session/worker by
    // value; recovery must not require allocating a new error box.
    #[allow(clippy::result_large_err)]
    pub fn new_retaining(
        registration: AdaptiveManagerRegistration,
        clock: ContractSystemClock,
        sources: AdaptiveManagerSources<Q, B, D>,
    ) -> Result<Self, AdaptiveManagerAdmissionRejection<Q, B, D>> {
        Self::admit_retaining_with_limits(
            registration,
            clock,
            AdaptiveExecutionLimits::default(),
            sources,
            false,
        )
    }
    // Preserve the same allocation-free source-owner rejection as new_retaining.
    #[allow(clippy::result_large_err)]
    pub fn new_retaining_with_limits(
        registration: AdaptiveManagerRegistration,
        clock: ContractSystemClock,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveManagerSources<Q, B, D>,
    ) -> Result<Self, AdaptiveManagerAdmissionRejection<Q, B, D>> {
        Self::admit_retaining_with_limits(registration, clock, limits, sources, true)
    }
    // A failed resource/authority check retains the complete source bundle.
    #[allow(clippy::result_large_err)]
    fn admit_retaining_with_limits(
        registration: AdaptiveManagerRegistration,
        clock: ContractSystemClock,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveManagerSources<Q, B, D>,
        check_native_limits: bool,
    ) -> Result<Self, AdaptiveManagerAdmissionRejection<Q, B, D>> {
        Self::admit_sources_with_ledger(
            registration,
            clock,
            limits,
            sources,
            check_native_limits,
            None,
        )
        .map_err(|rejected| {
            debug_assert!(rejected.original.is_none());
            AdaptiveManagerAdmissionRejection {
                error: rejected.error,
                sources: rejected.sources,
                continuation: None,
            }
        })
    }
    #[allow(clippy::result_large_err)]
    fn admit_sources_with_ledger(
        registration: AdaptiveManagerRegistration,
        clock: ContractSystemClock,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveManagerSources<Q, B, D>,
        check_native_limits: bool,
        original: Option<registered_ledger::RegisteredLedger>,
    ) -> Result<Self, registered_ledger::SourceAdmissionRejection<Q, B, D>> {
        // Preserve the move-only ledger outside every fallible source check.
        // In particular, no temporary replacement ledger is constructed.
        let original_registration = match original
            .as_ref()
            .map(|original| {
                registered_ledger::validate_registered_ledger::<Q, D>(
                    &registration,
                    &clock,
                    &sources.driver,
                    original,
                )
            })
            .transpose()
        {
            Ok(registered) => registered,
            Err(error) => {
                return Err(registered_ledger::SourceAdmissionRejection {
                    error,
                    sources,
                    original,
                });
            }
        };
        let native = &sources.native;
        let own_workers = &sources.own_workers;
        let external = &sources.external;
        let prepared = (|| -> Result<_, AdaptiveManagerError> {
            limits.validate()?;
            let role_call_limit = sources.driver.root_role_call_limit()?;
            if role_call_limit == Some(0) {
                return Err(AdaptiveManagerError::Invalid("root role call limit"));
            }
            if check_native_limits
                && (native.pending_capacity() != limits.role_tickets_max
                    || native.command_capacity() != limits.worker_commands_max)
            {
                return Err(AdaptiveManagerError::Invalid(
                    "actual native owner resource limits differ",
                ));
            }
            registration
                .endpoints
                .validate()
                .map_err(|_| AdaptiveManagerError::Invalid("CPU endpoint/resource policy"))?;
            if registration.authority.epoch != clock.domain().0
                || registration.controller.policy != registration.policy
                || ![1, 2, 4].contains(&own_workers.len())
                || usize::from(registration.controller.max_workers) != own_workers.len()
                || registration.controller.initial_workers == 0
                || usize::from(registration.controller.initial_workers) > own_workers.len()
                || registration.native_cpu_node_limit == 0
                || ![128, 256, 512].contains(&registration.token_budget)
            {
                return Err(AdaptiveManagerError::Invalid(
                    "manager source/resource registration",
                ));
            }
            if registration.endpoints.execution_use == CpuExecutionUse::Deployment
                && external.is_some()
                && registration.endpoints.cpu_r != CpuEndpoint::ExternalUci
            {
                return Err(AdaptiveManagerError::Invalid(
                    "unused CPU_T owner in V-free deployment",
                ));
            }
            if registration.endpoints.cpu_t == CpuEndpoint::ExternalUci && external.is_none() {
                return Err(AdaptiveManagerError::Invalid(
                    "declared CPU_T has no real owner",
                ));
            }
            if let Some(external) = external {
                if !external
                    .worker
                    .configuration_matches(&external.config, external.owner)
                    || !external
                        .worker
                        .clock_matches(clock.domain(), clock.origin())
                {
                    return Err(AdaptiveManagerError::Invalid(
                        "external actual configuration/owner/clock mismatch",
                    ));
                }
                external
                    .worker
                    .replace_root(registration.authority)
                    .map_err(|error| {
                        AdaptiveManagerError::Semantic(format!(
                            "external root registration: {error:?}"
                        ))
                    })?;
            }
            for (at, worker) in own_workers.iter().enumerate() {
                if original_registration.is_some() {
                    if !matches!(worker.ready_readback(), Some(Ok(_)))
                        || (worker.thread_affinity_requested()
                            && !matches!(worker.thread_affinity_ready(), Some(Ok(_))))
                    {
                        return Err(AdaptiveManagerError::Invalid(
                            "original ledger requires an actually ready CPU owner",
                        ));
                    }
                    // This actual owner API excludes an in-progress value
                    // replacement; a declared profile is not Ready evidence.
                    worker.acknowledged_profile()?;
                }
                if worker.command_capacity() != limits.worker_commands_max
                    || worker.paused_bytes_limit() != limits.paused_bytes_per_worker_max
                {
                    return Err(AdaptiveManagerError::Invalid(
                        "actual CPU owner resource limits differ",
                    ));
                }
                if own_workers[..at]
                    .iter()
                    .any(|old| old.profile().owner == worker.profile().owner)
                    || external
                        .as_ref()
                        .is_some_and(|foreign| foreign.owner == worker.profile().owner)
                {
                    return Err(AdaptiveManagerError::Invalid(
                        "duplicate physical CPU owner",
                    ));
                }
            }
            let cpu_clock = InstantClock::new(clock.domain(), clock.origin());
            let deadline = cpu_clock
                .deadline_at(registration.hard_deadline)
                .map_err(|_| AdaptiveManagerError::Deadline)?;
            native.replace_root(registration.authority)?;
            let native_ready = match native.ready_clock() {
                Some(ready) => {
                    if ready.domain() != clock.domain() || ready.origin() != clock.origin() {
                        return Err(AdaptiveManagerError::Invalid(
                            "persistent native clock mismatch",
                        ));
                    }
                    true
                }
                None => false,
            };
            let mut registry = RoleTicketRegistry::new_with_capacity(
                registration.authority,
                registration.token_budget,
                registration.policy,
                registration.first_request_sequence,
                limits.role_tickets_max,
            )?;
            let native_verifier_registered = match native.native_acceptance_verifier() {
                Some(verifier) => {
                    registry.register_native_acceptance_verifier(verifier)?;
                    true
                }
                None => false,
            };
            let machine = RefinementMachine::new(
                registration.authority,
                deadline,
                registration.token_budget,
            )?;
            let ready = ReadySelector::new(limits.waiting_jobs_max)
                .map_err(|_| AdaptiveManagerError::Capacity("ready selector"))?;
            let costs = CostEstimator::new(4096)
                .map_err(|_| AdaptiveManagerError::Capacity("cost estimator"))?;
            let controller = AdaptiveController::new(registration.controller)
                .map_err(|_| AdaptiveManagerError::Invalid("adaptive controller profile"))?;
            Ok((
                cpu_clock,
                deadline,
                native_ready,
                registry,
                native_verifier_registered,
                machine,
                ready,
                costs,
                controller,
                role_call_limit,
            ))
        })();
        let (
            cpu_clock,
            deadline,
            native_ready,
            registry,
            native_verifier_registered,
            machine,
            ready,
            costs,
            controller,
            role_call_limit,
        ) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                return Err(registered_ledger::SourceAdmissionRejection {
                    error,
                    sources,
                    original,
                });
            }
        };
        let (ledger, next_job, next_execution, cleanup_boundary) = if let Some(original) = original
        {
            let (observed, cleanup) = original_registration
                .expect("same retained ledger validated before source admission");
            (
                original.ledger,
                observed.highest_job_sequence + 1,
                observed.highest_execution_sequence + 1,
                Some(cleanup),
            )
        } else {
            let ledger = match CpuCreditLedger::new(
                registration.authority,
                registration.native_cpu_node_limit,
                cpu_clock,
            ) {
                Ok(ledger) => ledger,
                Err(error) => {
                    return Err(registered_ledger::SourceAdmissionRejection {
                        error: error.into(),
                        sources,
                        original: None,
                    });
                }
            };
            (ledger, 1, 1, None)
        };
        Ok(Self::assemble(
            registration,
            clock,
            limits,
            sources,
            admitted::ManagerAssembly {
                cpu_clock,
                deadline,
                native_ready,
                registry,
                native_verifier_registered,
                machine,
                ready,
                costs,
                researcher: ResearcherResources::Native(NativeResearcherResources {
                    ledger,
                    controller,
                }),
                role_call_limit,
                next_job,
                next_execution,
                next_control: 1,
                cleanup_boundary,
            },
        ))
    }
    pub fn new_continued(
        registration: AdaptiveManagerRegistration,
        clock: ContractSystemClock,
        native: TicketedNativeService<ReviewView, Q, B>,
        own_workers: Vec<CpuWorker>,
        external: Option<ExternalManagerWorker>,
        driver: D,
        continuation: AdaptiveSequenceContinuation,
    ) -> Result<Self, AdaptiveManagerError> {
        Self::new_continued_retaining(
            registration,
            clock,
            AdaptiveManagerSources {
                native,
                own_workers,
                external,
                driver,
            },
            continuation,
        )
        .map_err(|rejected| rejected.error)
    }
    // Keep both physical owners and the original sequence continuation on
    // rejection without allocating a second failure container.
    #[allow(clippy::result_large_err)]
    pub fn new_continued_retaining(
        registration: AdaptiveManagerRegistration,
        clock: ContractSystemClock,
        sources: AdaptiveManagerSources<Q, B, D>,
        continuation: AdaptiveSequenceContinuation,
    ) -> Result<Self, AdaptiveManagerAdmissionRejection<Q, B, D>> {
        Self::admit_continued_retaining_with_limits(
            registration,
            clock,
            AdaptiveExecutionLimits::default(),
            sources,
            continuation,
            false,
        )
    }
    // Same by-value ownership recovery, including the original continuation.
    #[allow(clippy::result_large_err)]
    pub fn new_continued_retaining_with_limits(
        registration: AdaptiveManagerRegistration,
        clock: ContractSystemClock,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveManagerSources<Q, B, D>,
        continuation: AdaptiveSequenceContinuation,
    ) -> Result<Self, AdaptiveManagerAdmissionRejection<Q, B, D>> {
        Self::admit_continued_retaining_with_limits(
            registration,
            clock,
            limits,
            sources,
            continuation,
            true,
        )
    }
    // Invalid continuation cannot drop or reconstruct its still-owned sources.
    #[allow(clippy::result_large_err)]
    fn admit_continued_retaining_with_limits(
        registration: AdaptiveManagerRegistration,
        clock: ContractSystemClock,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveManagerSources<Q, B, D>,
        continuation: AdaptiveSequenceContinuation,
        check_native_limits: bool,
    ) -> Result<Self, AdaptiveManagerAdmissionRejection<Q, B, D>> {
        if registration.authority.epoch != continuation.authority.epoch
            || registration.authority.implementation != continuation.authority.implementation
            || registration.authority.root.0 <= continuation.authority.root.0
            || registration.authority.game.0 < continuation.authority.game.0
            || registration.first_request_sequence != continuation.next_request
        {
            return Err(AdaptiveManagerAdmissionRejection {
                error: AdaptiveManagerError::Invalid(
                    "persistent process authority/sequence continuation",
                ),
                sources,
                continuation: Some(continuation),
            });
        }
        let mut manager = match Self::admit_retaining_with_limits(
            registration,
            clock,
            limits,
            sources,
            check_native_limits,
        ) {
            Ok(manager) => manager,
            Err(mut rejected) => {
                rejected.continuation = Some(continuation);
                return Err(rejected);
            }
        };
        manager.next_job = continuation.next_job;
        manager.next_execution = continuation.next_execution;
        manager.next_control = continuation.next_control;
        Ok(manager)
    }
    // Failed physical-fence/reuse checks return the intact manager, retaining
    // any quarantined buffers; no error-box allocation is needed to keep it.
    #[allow(clippy::result_large_err)]
    pub fn into_persistent_parts(
        self,
    ) -> Result<AdaptivePersistentParts<Q, B, D>, AdaptivePersistentRejection<Q, B, D>> {
        self.into_settled_parts(AdaptiveCloseMode::RootQuiesce)
    }
    /// Explicit process quit/join teardown. Returned closed native/external
    /// owners cannot be reused: their receipt never permits root admission.
    // Failure retains the same still-owned manager as root-fence recovery.
    #[allow(clippy::result_large_err)]
    pub fn into_process_parts(
        self,
    ) -> Result<AdaptivePersistentParts<Q, B, D>, AdaptivePersistentRejection<Q, B, D>> {
        self.into_settled_parts(AdaptiveCloseMode::ProcessShutdown)
    }
    /// Reuse a manager already held on the heap without moving the complete
    /// physical owner onto the session's stack, including on rejection.
    pub(crate) fn into_persistent_parts_boxed(
        self: Box<Self>,
    ) -> Result<AdaptivePersistentParts<Q, B, D>, AdaptiveBoxedPersistentRejection<Q, B, D>> {
        self.into_settled_parts_boxed(AdaptiveCloseMode::RootQuiesce)
    }
    pub(crate) fn into_process_parts_boxed(
        self: Box<Self>,
    ) -> Result<AdaptivePersistentParts<Q, B, D>, AdaptiveBoxedPersistentRejection<Q, B, D>> {
        self.into_settled_parts_boxed(AdaptiveCloseMode::ProcessShutdown)
    }
    fn into_settled_parts_boxed(
        mut self: Box<Self>,
        mode: AdaptiveCloseMode,
    ) -> Result<AdaptivePersistentParts<Q, B, D>, AdaptiveBoxedPersistentRejection<Q, B, D>> {
        let (receipt, cpu, continuation) = match self.prepare_settled_transfer(mode) {
            Ok(parts) => parts,
            Err(error) => {
                return Err(AdaptiveBoxedPersistentRejection {
                    error,
                    manager: self,
                });
            }
        };
        // All fallible checks and CPU transfers are finished. Move the actual
        // fields separately from the same allocation; do not move `*self`.
        Ok(AdaptivePersistentParts {
            native: self.native,
            cpu,
            external: self.external,
            driver: self.driver,
            continuation,
            receipt,
        })
    }
    // Existing by-value API remains compatible and uses the same checks.
    #[allow(clippy::result_large_err)]
    fn into_settled_parts(
        mut self,
        mode: AdaptiveCloseMode,
    ) -> Result<AdaptivePersistentParts<Q, B, D>, AdaptivePersistentRejection<Q, B, D>> {
        let (receipt, cpu, continuation) = match self.prepare_settled_transfer(mode) {
            Ok(parts) => parts,
            Err(error) => {
                return Err(AdaptivePersistentRejection {
                    error,
                    manager: self,
                });
            }
        };
        Ok(AdaptivePersistentParts {
            native: self.native,
            cpu,
            external: self.external,
            driver: self.driver,
            continuation,
            receipt,
        })
    }
    /// Every failure happens while the original manager still owns all of its
    /// resources. Successfully recovered CPU engines remain in their slots if
    /// a later worker transfer fails.
    fn prepare_settled_transfer(
        &mut self,
        mode: AdaptiveCloseMode,
    ) -> Result<
        (
            AdaptiveShutdownReceipt,
            Vec<PersistentCpuOwner>,
            AdaptiveSequenceContinuation,
        ),
        AdaptiveManagerError,
    > {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if self.collection_observer_v2.is_some() || self.collection_admission_v2.is_some() {
            return Err(AdaptiveManagerError::Invalid(
                "V2 collection observer must be returned",
            ));
        }
        if self.nnuec.is_some() || self.own.iter().any(|slot| slot.nnuec_retirement.is_some()) {
            return Err(AdaptiveManagerError::Invalid(
                "NNUEC requires actual actor retirement transfer",
            ));
        }
        let receipt = match self.shutdown_receipt() {
            Ok(receipt)
                if self.phase == ManagerPhase::Finished
                    && self.close_mode == mode
                    && match mode {
                        AdaptiveCloseMode::RootQuiesce => receipt.permits_root_reuse(),
                        AdaptiveCloseMode::ProcessShutdown => receipt.permits_archive_close(),
                    } =>
            {
                receipt
            }
            Ok(_) => {
                return Err(AdaptiveManagerError::Invalid(
                    "owner transfer needs actual finished root/process fence",
                ));
            }
            Err(error) => return Err(error),
        };
        for index in 0..self.own.len() {
            if self.own[index].recovered_engine.is_some() {
                continue;
            }
            let Some(worker) = self.own[index].worker.as_mut() else {
                return Err(AdaptiveManagerError::Invalid(
                    "persistent CPU owner missing",
                ));
            };
            match worker.take_engine() {
                Ok(engine) => self.own[index].recovered_engine = Some(engine),
                Err(error) => return Err(error.into()),
            }
        }
        let cpu = self
            .own
            .iter_mut()
            .map(|slot| PersistentCpuOwner {
                profile: slot
                    .worker
                    .as_ref()
                    .expect("all owner handles checked")
                    .profile(),
                engine: slot
                    .recovered_engine
                    .take()
                    .expect("all actual engine transfers checked"),
            })
            .collect();
        let continuation = AdaptiveSequenceContinuation {
            authority: self.registration.authority,
            next_request: self.registry.next_request_sequence(),
            next_job: self.next_job,
            next_execution: self.next_execution,
            next_control: self.next_control,
        };
        Ok((receipt, cpu, continuation))
    }
    pub fn phase(&self) -> ManagerPhase {
        self.phase
    }
    pub fn counters(&self) -> AdaptiveManagerCounters {
        self.counters
    }
    /// Borrow the original admitted source; this read grants no new root,
    /// authority, native execution or collection capability.
    pub fn collection_root_admission(&self) -> Option<&RoventCollectionRootAdmission> {
        self.collection_admission.as_ref()
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn collection_root_admission_v2(&self) -> Option<&RoventCollectionRootAdmissionV2> {
        self.collection_admission_v2.as_ref()
    }

    /// Issue the original no-capture scope only before any root role/observer
    /// starts. The caller must still register it with the actual native input
    /// observer and retain the cause; this does not prove physical completion.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn no_capture_root_v2(
        &self,
        admission: &RoventCollectionRootAdmissionV2,
    ) -> Result<collection::RoventNoCaptureRootCapabilityV2, AdaptiveManagerError> {
        if self.root_started || !self.roles.is_empty() || self.phase != ManagerPhase::Running
            || self.collection_observer_v2.is_some() || self.collection_admission_v2.is_some()
            || self.collection_observer.is_some() || self.collection_admission.is_some()
            || admission.root_admission().authority() != self.registration.authority
            || admission.root_admission().hard_deadline() != self.deadline
            || !Arc::ptr_eq(admission.rules(), &self.driver.root_rules())
        {
            return Err(AdaptiveManagerError::Invalid("no-capture root scope or admission order"));
        }
        let verifier = self.native.native_acceptance_verifier().ok_or(
            AdaptiveManagerError::Invalid("no-capture native verifier unavailable"),
        )?;
        collection::RoventNoCaptureRootCapabilityV2::new(admission.clone(), verifier)
            .map_err(|error| match error {
                collection::RoventNoCaptureIssueErrorV2::Busy(_original) => AdaptiveManagerError::AuxiliaryPending,
                collection::RoventNoCaptureIssueErrorV2::Rejected(error) => error.into(),
            })
    }

    /// Install an observation-only V2 owner at the actual attached root.
    /// On error return the original admission and observer to the caller.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn install_collection_observer_v2(
        &mut self,
        admission: RoventCollectionRootAdmissionV2,
        mut observer: AdaptiveCollectionObserverBoxV2<Q, B::Output>,
    ) -> Result<(), AdaptiveCollectionV2Rejection<Q, B::Output>> {
        let mut observer_called = false;
        let mut observer_succeeded = false;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<(), AdaptiveManagerError> {
                if self.root_started
                    || !self.roles.is_empty()
                    || self.phase != ManagerPhase::Running
                    || self.collection_observer.is_some()
                    || self.collection_admission.is_some()
                    || self.collection_observer_v2.is_some()
                    || self.collection_admission_v2.is_some()
                {
                    return Err(AdaptiveManagerError::Invalid(
                        "V2 collection observer installation order",
                    ));
                }
                if !self.driver.collection_v2_observation_complete() {
                    return Err(AdaptiveManagerError::Invalid(
                        "V2 adaptive collection observation is not fully connected",
                    ));
                }
                let root = admission.root_admission();
                if root.authority() != self.registration.authority
                    || root.hard_deadline() != self.deadline
                    || !Arc::ptr_eq(admission.rules(), &self.driver.root_rules())
                {
                    return Err(AdaptiveManagerError::Invalid("V2 collection root scope"));
                }
                let verifier = self.native.native_acceptance_verifier().ok_or(
                    AdaptiveManagerError::Invalid("V2 original native verifier absent"),
                )?;
                admission.validate_native_owner(&verifier)?;
                self.driver.validate_collection_admission_v2(&admission)?;
                observer_called = true;
                observer.admitted(&admission)?;
                observer_succeeded = true;
                // The callback can race expiration or revoke. Do not install
                // an admission based only on the earlier scope observation.
                let verifier = self.native.native_acceptance_verifier().ok_or(
                    AdaptiveManagerError::Invalid("V2 original native verifier absent"),
                )?;
                admission.validate_native_owner(&verifier)?;
                self.driver.validate_collection_admission_v2(&admission)?;
                self.check_running()?;
                Ok(())
            },
        ));
        let rejected = match result {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some((
                error,
                if observer_succeeded {
                    AdaptiveCollectionV2ObserverStatus::AdmissionSucceededRecheckFailed
                } else if observer_called {
                    AdaptiveCollectionV2ObserverStatus::AdmissionFailed
                } else {
                    AdaptiveCollectionV2ObserverStatus::AdmissionNotCalled
                },
                None,
            )),
            Err(payload) => Some((
                AdaptiveManagerError::Invalid("adaptive V2 collection admission panicked"),
                if observer_succeeded {
                    AdaptiveCollectionV2ObserverStatus::AdmissionSucceededRecheckPanicked
                } else if observer_called {
                    AdaptiveCollectionV2ObserverStatus::AdmissionPanicked
                } else {
                    AdaptiveCollectionV2ObserverStatus::ManagerValidationPanicked
                },
                Some(payload),
            )),
        };
        if let Some((error, observer_status, panic_payload)) = rejected {
            return Err(AdaptiveCollectionV2Rejection {
                error,
                admission,
                observer,
                observer_status,
                panic_payload,
            });
        }
        self.collection_admission_v2 = Some(admission);
        self.collection_observer_v2 = Some(observer);
        Ok(())
    }

    /// Return the original objects after the actual owner fence and an
    /// attempted observer settlement. The original settlement result is
    /// returned separately; teacher process retirement is separate.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn take_collection_observer_v2(
        &mut self,
    ) -> Result<
        Option<(
            RoventCollectionRootAdmissionV2,
            AdaptiveCollectionObserverBoxV2<Q, B::Output>,
            Result<(), AdaptiveCollectionError>,
        )>,
        AdaptiveManagerError,
    > {
        if self.collection_observer_v2.is_none() && self.collection_admission_v2.is_none() {
            return Ok(None);
        }
        if self.collection_observer_v2.is_none() || self.collection_admission_v2.is_none() {
            return Err(AdaptiveManagerError::Invalid(
                "V2 collection owner mismatch",
            ));
        }
        let receipt = self.shutdown_receipt()?;
        if self.phase != ManagerPhase::Finished
            || self.collection_v2_settlement.is_none()
            || !(receipt.permits_root_reuse() || receipt.permits_archive_close())
        {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        Ok(Some((
            self.collection_admission_v2
                .take()
                .expect("checked admission"),
            self.collection_observer_v2
                .take()
                .expect("checked observer"),
            self.collection_v2_settlement
                .take()
                .expect("checked settlement"),
        )))
    }
    /// 실제 수집을 요청한 caller만 root 시작 전에 설치한다. 기존 dispatch
    /// observer와 native owner 입력 observer는 별도 원 소유 경계로 유지한다.
    pub fn set_collection_observer(
        &mut self,
        mut observer: AdaptiveCollectionObserverBox<Q, B::Output>,
    ) -> Result<(), AdaptiveManagerError> {
        if self.root_started
            || !self.roles.is_empty()
            || self.phase != ManagerPhase::Running
            || self.collection_observer.is_some()
            || self.has_collection_v2()
        {
            return Err(AdaptiveManagerError::Invalid(
                "collection observer installation order",
            ));
        }
        let admission = self
            .collection_admission
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "collection requires original Rovent root admission",
            ))?;
        observer.admitted(admission)?;
        self.collection_observer = Some(observer);
        Ok(())
    }
    pub(crate) fn admit_collection_root(
        &mut self,
        admission: RoventCollectionRootAdmission,
    ) -> Result<(), AdaptiveManagerError> {
        self.driver.validate_collection_admission(&admission)?;
        if self.root_started
            || !self.roles.is_empty()
            || self.collection_admission.is_some()
            || self.has_collection_v2()
            || self.phase != ManagerPhase::Running
            || admission.authority() != self.registration.authority
            || !Arc::ptr_eq(admission.rules(), &self.driver.root_rules())
        {
            return Err(AdaptiveManagerError::Invalid(
                "original Rovent collection root scope",
            ));
        }
        self.collection_admission = Some(admission);
        Ok(())
    }
    fn has_collection_v2(&self) -> bool {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        {
            self.collection_admission_v2.is_some() || self.collection_observer_v2.is_some()
        }
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        {
            false
        }
    }
    fn has_collection_failure(&self) -> bool {
        !self.collection_failure_is_independent()
            && self
                .failures
                .iter()
                .any(|error| matches!(error, AdaptiveManagerError::Collection(_)))
    }
    fn collection_failure_is_independent(&self) -> bool {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        {
            self.collection_observer_v2.is_some()
                && self.collection_admission_v2.is_some()
                && self.collection_observer.is_none()
                && !self.collection_v2_disable_failed
        }
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        {
            false
        }
    }
    fn has_execution_failure(&self) -> bool {
        self.failures.iter().any(|error| {
            !matches!(error, AdaptiveManagerError::Collection(_))
                || !self.collection_failure_is_independent()
        })
    }
    /// V2 Collection is optional to the native search. Its original failure
    /// remains in the receipt; an unavailable disabled marker still fails
    /// closed before the next native dispatch. Legacy Collection stays strict.
    pub(crate) fn record_collection_caller_failure(
        &mut self,
        error: AdaptiveCollectionError,
        until: Instant,
    ) -> Result<(), AdaptiveManagerError> {
        self.record_failure(AdaptiveManagerError::Collection(error));
        if self.collection_failure_is_independent() {
            Ok(())
        } else {
            self.begin_root_quiesce(until)
        }
    }
    /// 다음 root에서 재사용할 observer 반환은 실제 owner fence 뒤만 허용한다.
    pub fn take_collection_observer(
        &mut self,
    ) -> Result<Option<AdaptiveCollectionObserverBox<Q, B::Output>>, AdaptiveManagerError> {
        if self.phase != ManagerPhase::Finished {
            return Err(AdaptiveManagerError::Invalid(
                "collection observer requires settled root",
            ));
        }
        Ok(self.collection_observer.take())
    }
    /// Explicit diagnostic opt-in before root/CPU admission. This can be used
    /// from the read-only diagnostic root hook: it installs observation only,
    /// never a callback, worker control, credit or semantic acceptance permit.
    /// Snapshot is available solely through the caller's explicit handle.
    pub fn install_cpu_slice_diagnostic(
        &self,
        diagnostic: AdaptiveCpuSliceDiagnostic,
    ) -> Result<(), AdaptiveCpuSliceDiagnosticError> {
        if self.phase != ManagerPhase::Running
            || self.root_started
            || !self.roles.is_empty()
            || self.own.iter().any(|slot| slot.binding.is_some())
            || self.counters.cpu_slice_receipts != 0
        {
            return Err(AdaptiveCpuSliceDiagnosticError::AdmissionAlreadyStarted);
        }
        if self.cpu_slice_diagnostic.get().is_some() {
            return Err(AdaptiveCpuSliceDiagnosticError::AlreadyInstalled);
        }
        diagnostic.bind(self.registration.authority)?;
        self.cpu_slice_diagnostic
            .set(diagnostic)
            .map_err(|rejected| {
                rejected
                    .mark_unavailable(AdaptiveCpuSliceDiagnosticUnavailable::InstallationRejected);
                AdaptiveCpuSliceDiagnosticError::AlreadyInstalled
            })
    }
    /// Actual service-held preparations, pending and physically completed but
    /// unacknowledged outputs. Logical counter differences cannot replace it.
    pub fn native_occupancy(&self) -> usize {
        self.native.occupancy()
    }
    pub fn root_observation_snapshot(
        &self,
    ) -> Result<AdaptiveRootObservationSnapshot, AdaptiveManagerError> {
        let receipt = self.shutdown_receipt()?;
        if self.phase != ManagerPhase::Finished || !receipt.permits_semantic_closure() {
            return Err(AdaptiveManagerError::Invalid(
                "root observations require actual settled owner closure",
            ));
        }
        let (cpu_spent, cpu_reserved) = self.native_ledger()?.global_usage()?;
        Ok(AdaptiveRootObservationSnapshot {
            authority: self.registration.authority,
            close_mode: self.close_mode,
            manager: self.counters,
            native_accounting: self.native_accounting,
            role_tickets: self.registry.cumulative_observation(),
            history_peak: self.machine.history_peak(),
            validator_history_peak: self.machine.validator_action_history_peak(),
            cpu_credit_limit: self.registration.native_cpu_node_limit,
            cpu_spent,
            cpu_reserved,
            validator_loaded: self.registration.endpoints.load_v,
            receipt,
        })
    }
    #[cfg(feature = "pals-adaptive-diagnostics")]
    pub(super) fn install_continuation_diagnostic(&mut self, diagnostic: ContinuationDiagnostic) {
        if self.phase != ManagerPhase::Running
            || self.root_started
            || !self.roles.is_empty()
            || self.own.iter().any(|slot| slot.binding.is_some())
            || self.continuation_diagnostic.is_some()
        {
            diagnostic.unavailable(ContinuationDiagnosticUnavailable::ScopeMismatch);
            return;
        }
        diagnostic.bind(self.registration.authority);
        self.continuation_diagnostic = Some(diagnostic);
    }
    #[cfg(feature = "pals-adaptive-diagnostics")]
    fn observe_pending_diagnostic(
        &self,
        source: ContinuationPendingSource,
        trigger: RequestId,
        queue_admitted_at: Option<Instant>,
    ) {
        if let Some(diagnostic) = &self.continuation_diagnostic {
            if diagnostic.pending_enabled() {
                diagnostic.pending(
                    self.registration.authority,
                    ContinuationPendingObservation {
                        source,
                        trigger,
                        queue_admitted_at,
                        snapshot: self.pending_question_snapshot(),
                    },
                );
            }
        }
    }
    pub fn driver(&self) -> &D {
        &self.driver
    }
    pub fn driver_mut(&mut self) -> &mut D {
        &mut self.driver
    }
    /// Register the caller's original finite cleanup deadline before this
    /// root admits work. This policy can shorten closure, never extend it.
    pub fn register_cleanup_deadline(
        &mut self,
        original_cleanup_deadline: Instant,
    ) -> Result<(), AdaptiveManagerError> {
        if self.cleanup_boundary.is_some()
            || self.phase != ManagerPhase::Running
            || self.root_started
            || self.shutdown_until.is_some()
            || self.counters.native_preparations != 0
            || !self.roles.is_empty()
            || self.own.iter().any(|slot| slot.binding.is_some())
            || self.foreign.is_some()
            || self.profiled_busy()
        {
            return Err(AdaptiveManagerError::Invalid(
                "original cleanup deadline admission",
            ));
        }
        self.cleanup_boundary = Some(CleanupBoundary::admit(
            self.registration.hard_deadline,
            original_cleanup_deadline,
        )?);
        Ok(())
    }
    pub fn registered_cleanup_deadline(&self) -> Option<Instant> {
        self.cleanup_boundary.map(CleanupBoundary::original)
    }
    fn cleanup_until(&self) -> Result<Instant, AdaptiveManagerError> {
        automatic_cleanup_deadline(self.cleanup_boundary)
    }
    pub fn enable_exclusive_action_capture(&mut self) -> Result<(), AdaptiveManagerError> {
        if self.root_started
            || self.phase != ManagerPhase::Running
            || self.exclusive_requested
            || self.registration.endpoints.execution_use != CpuExecutionUse::Collection
            || !self.registration.endpoints.load_v
        {
            return Err(AdaptiveManagerError::Invalid(
                "exclusive action admission must precede root P",
            ));
        }
        self.exclusive_requested = true;
        self.close_mode = AdaptiveCloseMode::RootQuiesce;
        Ok(())
    }
    /// Collection-only deterministic action enumeration. The exact action
    /// remains subject to the actual Search-issued support plan after V ack.
    pub fn enable_exclusive_registered_action(
        &mut self,
        action: AdaptiveValidatorAction,
    ) -> Result<(), AdaptiveManagerError> {
        if self.root_started
            || self.phase != ManagerPhase::Running
            || self.exclusive_requested
            || self.registration.endpoints.execution_use != CpuExecutionUse::Collection
            || !self.registration.endpoints.load_v
        {
            return Err(AdaptiveManagerError::Invalid(
                "registered action is Collection-only before root P",
            ));
        }
        self.driver.register_supported_validator_action(action)?;
        self.enable_exclusive_action_capture()?;
        self.registered_action = Some(action);
        Ok(())
    }
    pub fn exclusive_action_closure_proof(
        &self,
    ) -> Result<ExclusiveActionClosureProof, AdaptiveManagerError> {
        let receipt = self.shutdown_receipt()?;
        let scope = self
            .exclusive_scope
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid("no exclusive action scope"))?;
        if self.phase != ManagerPhase::Finished
            || !receipt.permits_root_reuse()
            || !scope.completed
            || self.failures().next().is_some()
        {
            return Err(AdaptiveManagerError::Invalid(
                "exclusive action lacks actual root fence",
            ));
        }
        Ok(ExclusiveActionClosureProof {
            authority: self.registration.authority,
            source: scope
                .source
                .ok_or(AdaptiveManagerError::Invalid("exclusive V source missing"))?,
            action: scope
                .action
                .ok_or(AdaptiveManagerError::Invalid("exclusive action missing"))?,
            started: scope.started,
            finished: self.finished_at.ok_or(AdaptiveManagerError::Invalid(
                "exclusive actual closure time missing",
            ))?,
            baseline: scope.baseline,
            final_counters: self.counters,
            question: scope.question.ok_or(AdaptiveManagerError::Invalid(
                "exclusive frozen parent/mask missing",
            ))?,
        })
    }
    pub fn complete_exclusive_action(
        &self,
        semantic: ValidatorActionOutcome,
    ) -> Result<CompletedValidatorAction, AdaptiveManagerError> {
        let proof = self.exclusive_action_closure_proof()?;
        CompletedValidatorAction::after_exclusive_manager_shutdown(
            semantic,
            &self.shutdown_receipt()?,
            &proof,
        )
    }
    pub fn failures(&self) -> impl Iterator<Item = &AdaptiveManagerError> {
        self.failures.iter()
    }
    /// The original independent collection failure survives bounded-history
    /// rollover and must be included in the caller's partial receipt.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn first_v2_collection_failure(&self) -> Option<&AdaptiveCollectionError> {
        self.collection_v2_first_failure.as_ref()
    }
    pub fn retained_native_completions(&self) -> &[RetainedNativeCompletion<B::Output>] {
        &self.retained_native_completions
    }

    /// Original non-physical capture failures. These are not NN failures.
    pub fn native_observation_failures(&self) -> &[TicketNativeObservationFailure] {
        &self.native_observation_failures
    }

    fn observe_native_failures(
        &mut self,
        ticket: &RoleTicketEnvelope,
        failures: &TicketNativeObservationFailures,
        retired: bool,
    ) -> Option<AdaptiveManagerError> {
        let mut gate = None;
        for failure in failures.iter() {
            let error = if failure.cause == RoleFailure::PhysicalUnknown {
                Some(AdaptiveManagerError::PhysicalUnknown)
            } else if failure.ticket != *ticket
                || match failure.stage {
                    TicketNativeObservationStage::Submitted
                    | TicketNativeObservationStage::Completed => retired,
                    TicketNativeObservationStage::Retired => !retired,
                    TicketNativeObservationStage::Retrieval => false,
                }
                || failure.disposition == TicketNativeObservationDisposition::RequiredAfterPhysical
            {
                Some(AdaptiveManagerError::NativeObservation(failure.clone()))
            } else {
                None
            };
            if self.native_observation_failures.len()
                < self.limits.role_tickets_max.saturating_mul(2)
            {
                self.native_observation_failures.push(failure.clone());
            } else if gate.is_none() {
                gate = Some(AdaptiveManagerError::NativeObservation(failure.clone()));
            }
            if gate.is_none() || matches!(&error, Some(AdaptiveManagerError::PhysicalUnknown)) {
                gate = error.or(gate);
            }
        }
        gate
    }
    pub fn native_cpu_usage(&self) -> Result<(u64, u64), AdaptiveManagerError> {
        Ok(self.native_ledger()?.global_usage()?)
    }
    /// Actual single-writer identity for opt-in root-controller attachment.
    /// A digest or a freshly constructed ledger cannot replace this owner.
    pub fn cpu_ledger_identity(&self) -> CpuLedgerIdentity {
        self.try_cpu_ledger_identity()
            .expect("native manager facade owns its original ledger")
    }
    pub fn try_cpu_ledger_identity(&self) -> Result<CpuLedgerIdentity, AdaptiveManagerError> {
        Ok(self.native_ledger()?.identity())
    }
    pub fn cpu_node_limit_reached(&self) -> bool {
        self.cpu_node_limit_reached
    }
    /// Root-wide logical calls spent and reserved for preparation. Registry
    /// admission charges a call once; semantic acceptance and NN input counts
    /// are separate observations. A binding without a ticket reserves one
    /// place before the asynchronous native preparation can finish. Conversion
    /// to a registry charge precedes fallible frame binding, so an error does
    /// not count the same original call as both spent and reserved.
    pub fn role_call_budget_usage(&self) -> (u64, u64) {
        (
            self.registry.cumulative_observation().admitted,
            self.roles
                .iter()
                .filter(|binding| !binding.admission_charged)
                .count() as u64,
        )
    }
    pub fn role_call_limit_reached(&self) -> bool {
        let (spent, _) = self.role_call_budget_usage();
        self.role_call_limit.is_some_and(|limit| spent >= limit)
    }
    fn role_call_admission_available(&self) -> bool {
        let (spent, reserved) = self.role_call_budget_usage();
        self.role_call_limit
            .is_none_or(|limit| spent.saturating_add(reserved) < limit)
    }
    pub fn has_pending_work(&self) -> bool {
        self.ordinary_pending_work()
            || self.goal_prepared.is_some()
            || self.goal_continuation_pending
            || self.utility_scope_open()
    }
    fn ordinary_pending_work(&self) -> bool {
        !self.roles.is_empty()
            || !self.accepted_roles.is_empty()
            || self.machine.frames().next().is_some()
            || self.own.iter().any(|slot| slot.binding.is_some())
            || self.foreign.is_some()
            || self.profiled_busy()
            || !self.deferred.is_empty()
            || !self.native_frontier_events.is_empty()
            || !self.driver.semantic_owner_available()
    }
    /// The root P call is issued exactly once. Candidate frames are admitted
    /// later from the private published-ranking proof, never from copied logits.
    pub fn prepare_root_ranking(
        &mut self,
        input: FrameInput<ReviewView, Q>,
    ) -> Result<PreparationId, AdaptiveManagerError> {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let ResearcherResources::ProfiledExternal(source) = &self.researcher {
            if let Some(root) = source.root_admission_v1() {
                self.driver
                    .validate_profiled_external_registration(source.current_source(), root)?;
            } else if let Some(root) = source.root_admission_v2() {
                self.driver
                    .validate_profiled_external_registration_v2(source.current_source(), root)?;
            } else {
                return Err(AdaptiveManagerError::Invalid(
                    "profiled external original root absent",
                ));
            }
        }
        if input.stage != RefinementStage::ProposerRanking
            || !self.roles.is_empty()
            || self.machine.frames().next().is_some()
        {
            return Err(AdaptiveManagerError::Invalid(
                "root ranking admission order",
            ));
        }
        let preparation = self.prepare_role(None, input, RolePurpose::RootRanking)?;
        self.root_started = true;
        Ok(preparation)
    }
    /// V is a separate Collection role question, never a root P ranking or a
    /// candidate refinement frame. Deployment has no V owner/call admission.
    pub fn prepare_validator(
        &mut self,
        input: FrameInput<ReviewView, Q>,
    ) -> Result<PreparationId, AdaptiveManagerError> {
        if self.registration.endpoints.execution_use != CpuExecutionUse::Collection
            || !self.registration.endpoints.load_v
            || input.stage != RefinementStage::ValidatorTaskRanking
        {
            return Err(AdaptiveManagerError::Invalid(
                "V requires explicit Collection TaskPolicy",
            ));
        }
        self.prepare_role(None, input, RolePurpose::Validator)
    }
    /// Opt-in bounded observation of this manager's retained questions only.
    /// The immutable borrow excludes all manager/registry transitions through
    /// the complete copy; sampled_at is a point inside that borrow. No service
    /// lock, event drain, current-view publication, backend, I/O or callback is
    /// entered. Physical execution may proceed concurrently and is not sampled.
    /// Every row is returned or the whole observation is unavailable. Callers
    /// must not turn diagnostic failure into admission, cancellation or retry.
    #[cfg(feature = "search-work-receipts")]
    pub fn pending_question_snapshot(
        &self,
    ) -> Result<AdaptivePendingQuestionSnapshot, AdaptivePendingSnapshotUnavailable> {
        use AdaptivePendingSnapshotUnavailable as Unavailable;
        if self.roles.len() > self.limits.role_tickets_max
            || self.roles.len() > MAX_ROLE_PENDING
            || self.registry.occupancy() > MAX_ROLE_PENDING
        {
            return Err(Unavailable::Capacity);
        }
        let sampled_at = self
            .clock
            .try_now()
            .map_err(|_| Unavailable::ClockUnavailable)?;
        let mut questions: Vec<AdaptivePendingQuestionRow> = Vec::new();
        questions
            .try_reserve_exact(self.roles.len())
            .map_err(|_| Unavailable::Capacity)?;
        for binding in &self.roles {
            if questions.iter().any(|row| {
                row.preparation == binding.preparation
                    || row.ticket.is_some_and(|ticket| {
                        binding.ticket.is_some_and(|other| ticket.id == other.id)
                    })
            }) {
                return Err(Unavailable::RegistryBindingMismatch);
            }
            validate_view(
                &binding.input.view,
                &binding.input.view_seal,
                self.registration.authority,
                self.registration.token_budget,
            )
            .map_err(|_| Unavailable::ViewSealMismatch)?;
            let registry_observed_phase = if let Some(ticket) = binding.ticket {
                if !binding.admission_charged
                    || self.registry.ticket(ticket.id) != Some(&ticket)
                    || ticket.authority != self.registration.authority
                    || ticket.question != binding.input.question_digest
                    || ticket.view != binding.input.view_seal
                    || Some(ticket.role) != binding.input.stage.role()
                    || ticket.original_deadline != self.deadline
                {
                    return Err(Unavailable::RegistryBindingMismatch);
                }
                Some(
                    self.registry
                        .phase(ticket.id)
                        .ok_or(Unavailable::RegistryBindingMismatch)?,
                )
            } else {
                // Failed frame binding can retain an admitted registry charge
                // before RoleBinding::ticket is assigned. Do not invent zero
                // pending tickets or silently present an incomplete ID join.
                if binding.admission_charged {
                    return Err(Unavailable::RegistryBindingMismatch);
                }
                None
            };
            let mut selected_record_revisions = Vec::new();
            selected_record_revisions
                .try_reserve_exact(binding.input.view.records().len())
                .map_err(|_| Unavailable::Capacity)?;
            selected_record_revisions.extend(
                binding
                    .input
                    .view
                    .records()
                    .iter()
                    .map(|record| record.revision),
            );
            questions.push(AdaptivePendingQuestionRow {
                preparation: binding.preparation,
                frame: binding.work.as_ref().map(|work| work.frame),
                stage: binding.input.stage,
                question: binding.input.question_digest,
                view: binding.input.view_seal,
                retained_view_digest: Digest(binding.input.view.digest()),
                retained_view_seal_digest: Digest(binding.input.view.seal_digest()),
                selected_record_revisions,
                ticket: binding.ticket,
                registry_observed_phase,
                dispatch_pending: binding.dispatch_pending,
                completion_received: binding.service_completed,
                consumption_recorded: binding.consumption_recorded,
            });
        }
        if questions.iter().filter(|row| row.ticket.is_some()).count() != self.registry.occupancy()
        {
            return Err(Unavailable::RegistryBindingMismatch);
        }
        Ok(AdaptivePendingQuestionSnapshot {
            authority: self.registration.authority,
            phase: self.phase,
            sampled_at,
            original_deadline: self.deadline,
            questions,
        })
    }

    fn check_running(&self) -> Result<MonotonicTick, AdaptiveManagerError> {
        if self.phase != ManagerPhase::Running {
            return Err(AdaptiveManagerError::Invalid("admission is closed"));
        }
        let now = self
            .clock
            .try_now()
            .map_err(|_| AdaptiveManagerError::Deadline)?;
        self.deadline
            .accepts(self.clock.domain(), now)
            .map_err(|_| AdaptiveManagerError::Deadline)?;
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if self.cohort_action_expired(Instant::now()) {
            return Err(AdaptiveManagerError::AdmissionClosed);
        }
        Ok(now)
    }
    fn role_pending_limit(&self) -> usize {
        let declared = if self.registration.policy == CoordinatorPolicy::Synchronous {
            1
        } else {
            self.limits.role_tickets_max
        };
        self.driver
            .review_lease_admission()
            .ok()
            .flatten()
            .filter(|state| state.frontier_owner_present)
            .map_or(declared, |state| {
                declared.min(
                    state
                        .capacity
                        .saturating_sub(1)
                        .saturating_sub(state.frontier_replacement_reserve),
                )
            })
    }
    /// Jobs waiting before their first physical CPU slice share this ceiling.
    /// A started paused owner waiting for V feedback is not a new queued Job;
    /// charging it again can deadlock a one-seat queue behind an idle worker.
    fn waiting_jobs_occupancy(&self) -> usize {
        self.ready
            .len()
            .saturating_add(usize::from(self.nnuec_pending.is_some()))
            .saturating_add(
                self.validator_waiting
                    .iter()
                    .filter(|frame| {
                        !self.own.iter().any(|slot| {
                            slot.binding
                                .as_ref()
                                .is_some_and(|binding| binding.work.frame == **frame)
                        }) && !self
                            .foreign
                            .as_ref()
                            .is_some_and(|binding| binding.work.frame == **frame)
                    })
                    .count(),
            )
    }
    fn validator_waiting_limit(&self) -> usize {
        self.limits
            .waiting_jobs_max
            .saturating_add(self.own.len())
            .saturating_add(usize::from(self.external.is_some()))
            .min(CPU_JOB_CAPACITY)
    }
    fn prepare_role(
        &mut self,
        work: Option<CoordinatorWork<ReviewView, Q>>,
        input: FrameInput<ReviewView, Q>,
        purpose: RolePurpose,
    ) -> Result<PreparationId, AdaptiveManagerError> {
        if !self.can_admit_work() {
            return Err(AdaptiveManagerError::AdmissionClosed);
        }
        if self.utility_scope_open()
            && work
                .as_ref()
                .is_none_or(|work| !self.utility_allows_frame(work.frame))
        {
            return Err(AdaptiveManagerError::Invalid(
                "unrelated role in utility route scope",
            ));
        }
        if let Some(scope) = &self.exclusive_scope {
            let allowed = match purpose {
                RolePurpose::RootRanking | RolePurpose::RoventGoalLocal => false,
                RolePurpose::Validator => scope.action.is_none(),
                RolePurpose::Refinement
                | RolePurpose::Reciprocal
                | RolePurpose::RoventRefinement
                | RolePurpose::RoventFresh => {
                    work.as_ref().is_some_and(|work| scope.allows(work.frame))
                }
            };
            if !allowed {
                return Err(AdaptiveManagerError::Invalid(
                    "unrelated role in exclusive action scope",
                ));
            }
        }
        let now = self.check_running()?;
        if !self.role_call_admission_available() {
            return Err(AdaptiveManagerError::RoleCallLimit);
        }
        let role = input.stage.role().ok_or(AdaptiveManagerError::Invalid(
            "CPU stage sent to native owner",
        ))?;
        if role == Role::Validator && !self.registration.endpoints.load_v {
            return Err(AdaptiveManagerError::Invalid("V call in V-free deployment"));
        }
        if self.roles.len() >= self.role_pending_limit() {
            return Err(AdaptiveManagerError::Capacity("pending role questions"));
        }
        validate_view(
            &input.view,
            &input.view_seal,
            self.registration.authority,
            self.registration.token_budget,
        )?;
        input
            .view_seal
            .validate(self.registration.authority, self.registration.token_budget)?;
        self.deadline
            .accepts(self.clock.domain(), now)
            .map_err(|_| AdaptiveManagerError::Deadline)?;
        let question = TicketPreparation {
            question: input.question_digest,
            seal: input.view_seal,
            role,
            deadline: self.deadline,
            view: Arc::clone(&input.view),
            query: input.question.as_ref().clone(),
        };
        let preparation = self
            .native
            .prepare(question)
            .map_err(|(error, _)| AdaptiveManagerError::Native(error))?;
        self.roles.push(RoleBinding {
            preparation,
            prepared_observed: false,
            cancellation_requested: false,
            work,
            input,
            ticket: None,
            admission_charged: false,
            dispatch_pending: false,
            collection_registered: false,
            rovent_collection_registered: false,
            physical: None,
            prepared_at: Instant::now(),
            preparation_elapsed: None,
            queue_admitted_at: None,
            queue_wait: None,
            dispatched_at: None,
            service_completed: false,
            class: None,
            delivered: None,
            ack_accepted: None,
            ack_dispatched: false,
            revoked_acceptance: None,
            native_acceptance_recorded: false,
            consumption_recorded: false,
            consumption_unknown: false,
            purpose,
            reciprocal_acceptance: None,
            goal_local: None,
        });
        self.counters.native_preparations += 1;
        Ok(preparation)
    }
    /// One bounded owner turn. Receiving messages precedes dispatch so native
    /// completions and control acks cannot starve behind the ordinary work queue.
    pub fn tick(&mut self) -> Result<bool, AdaptiveManagerError> {
        if self.phase == ManagerPhase::Finished {
            return Ok(false);
        }
        if self.phase == ManagerPhase::Running && self.effective_hard_expired() {
            self.begin_selected_shutdown(self.cleanup_until()?)?;
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if self.phase == ManagerPhase::Running && self.cohort_action_expired(Instant::now()) {
            self.begin_root_quiesce(self.cleanup_until()?)?;
        }
        let mut changed = false;
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        match self.poll_profiled_events() {
            Ok(progress) => changed |= progress,
            Err(error) => self.record_failure(error),
        }
        match self.driver.poll_auxiliary() {
            Ok(progress) => changed |= progress,
            Err(error) => self.record_failure(error),
        }
        for _ in 0..MAX_EVENT_DRAIN {
            let Some(event) = self.native.poll() else {
                break;
            };
            changed = true;
            if let Err(error) = self.native_event(event) {
                self.physical_event_error(error)?;
            }
        }
        for slot in 0..self.own.len() {
            for _ in 0..16 {
                let event = match self.own[slot].worker.as_ref() {
                    Some(worker) => worker.try_event_observed(),
                    None => break,
                };
                match event {
                    Ok(Some(event)) => {
                        changed = true;
                        if let Err(error) = self.own_event_observed(slot, event) {
                            self.physical_event_error(error)?;
                        }
                    }
                    Ok(None) => break,
                    Err(error) => {
                        if !self.own[slot].joined && self.phase == ManagerPhase::Running {
                            self.record_failure(AdaptiveManagerError::Cpu(error));
                        }
                        break;
                    }
                }
            }
            if let Some(progress) = self.own[slot]
                .worker
                .as_ref()
                .and_then(CpuWorker::latest_progress_observed)
            {
                changed = true;
                if let Err(error) = self.own_progress_observed(slot, progress) {
                    self.record_failure(error);
                }
            }
        }
        if !self.external_startup_observed {
            if let Some(report) = self
                .external
                .as_ref()
                .and_then(|external| external.worker.startup_report())
            {
                self.external_startup_observed = true;
                changed = true;
                if let Err(error) = report.result {
                    self.record_failure(AdaptiveManagerError::Semantic(format!(
                        "external startup: {error:?}"
                    )));
                }
            }
        }
        if let Some(completion) = self
            .external
            .as_ref()
            .and_then(|external| external.worker.take_report_attested())
        {
            changed = true;
            if let Err(error) = self.foreign_event(completion) {
                self.physical_event_error(error)?;
            }
        }
        if let Some((sequence, progress)) = self
            .external
            .as_ref()
            .and_then(|external| external.worker.take_progress_observed())
        {
            changed = true;
            if let Err(error) = self.foreign_progress_observed(sequence, progress) {
                self.record_failure(error);
            }
        }
        if let Err(error) = self.flush_acks() {
            self.record_failure(error);
        }
        // Reliable physical owner events above are always processed before an
        // optional collector callback can fail. Its original capture remains
        // separate custody and cannot undo an accepted native-ledger receipt.
        self.poll_cpu_semantic();
        if self.has_execution_failure() && self.shutdown_until.is_none() {
            self.begin_selected_shutdown(self.cleanup_until()?)?;
        }
        if self.phase == ManagerPhase::Running {
            if self.check_running().is_err() {
                self.begin_selected_shutdown(self.cleanup_until()?)?;
            } else {
                for action in [
                    Self::flush_deferred,
                    Self::dispatch_roles,
                    Self::deliver_roles,
                    Self::schedule,
                    Self::schedule_root_round,
                ] {
                    match action(self) {
                        Ok(progress) => changed |= progress,
                        Err(error) => {
                            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                            if matches!(error, AdaptiveManagerError::AdmissionClosed)
                                && self.cohort_action_expired(Instant::now())
                            {
                                self.begin_root_quiesce(self.cleanup_until()?)?;
                                changed = true;
                                break;
                            }
                            self.frontier_boundary_or_failure(error)?;
                            break;
                        }
                    }
                    if self.has_collection_failure() {
                        break;
                    }
                }
                let exclusive_completed = self
                    .exclusive_scope
                    .as_ref()
                    .is_some_and(|scope| scope.completed);
                if !self.has_execution_failure()
                    && (exclusive_completed
                        // Every admitted role retains its original physical
                        // completion/ack/semantic commit obligations. Budget
                        // exhaustion only closes new work; it cannot discard
                        // an already reserved output to manufacture success.
                        || (self.role_call_limit_reached() && self.roles.is_empty() && self.accepted_roles.is_empty() && self.driver.semantic_owner_available()))
                {
                    let until = self.cleanup_until()?;
                    if exclusive_completed {
                        self.begin_root_quiesce(until)?;
                    } else {
                        let limit = self.role_call_limit.ok_or(AdaptiveManagerError::Invalid(
                            "settled role-call limit absent",
                        ))?;
                        let (spent, _) = self.role_call_budget_usage();
                        if let Err(error) = self.driver.close_rovent_role_call_budget(spent, limit)
                        {
                            self.record_failure(error);
                        }
                        self.begin_selected_shutdown(until)?;
                    }
                }
                if self.has_execution_failure() {
                    self.begin_selected_shutdown(self.cleanup_until()?)?;
                }
            }
        } else {
            match self.deliver_roles() {
                Ok(progress) => changed |= progress,
                Err(error) => self.record_failure(error),
            }
            match self.flush_deferred() {
                Ok(progress) => changed |= progress,
                Err(error) => self.record_failure(error),
            }
            match self.poll_shutdown() {
                Ok(progress) => changed |= progress,
                Err(error) => self.record_failure(error),
            }
        }
        Ok(changed)
    }
    /// These event handlers return a future-admission error only after
    /// retaining the original owner, receipt/report and deferred commit.
    /// Their actual terminal must still drain when a cohort window expires.
    fn physical_event_error(
        &mut self,
        error: AdaptiveManagerError,
    ) -> Result<(), AdaptiveManagerError> {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if matches!(error, AdaptiveManagerError::AdmissionClosed)
            && self.cohort_action_expired(Instant::now())
        {
            self.begin_root_quiesce(self.cleanup_until()?)?;
            return Ok(());
        }
        self.frontier_boundary_or_failure(error)
    }
    fn frontier_boundary_or_failure(
        &mut self,
        error: AdaptiveManagerError,
    ) -> Result<(), AdaptiveManagerError> {
        let controlled_capacity = self.phase == ManagerPhase::Running
            && match &error {
                AdaptiveManagerError::Frontier(FrontierError::PublicRecordCapacity) => {
                    self.driver.rovent_public_record_capacity_closed()?
                }
                AdaptiveManagerError::Frontier(FrontierError::PublicReviewIndexCapacity) => {
                    self.driver.rovent_public_review_index_capacity_closed()?
                }
                _ => false,
            };
        if controlled_capacity {
            // The rejected output/report remains in its original binding or
            // deferred queue. Quiescence drains it without semantic retry or
            // fabricated utility; the original frontier retains Capacity.
            self.fail_utility_route();
            self.begin_root_quiesce(self.cleanup_until()?)?;
        } else {
            self.record_failure(error);
        }
        Ok(())
    }
    fn schedule_root_round(&mut self) -> Result<bool, AdaptiveManagerError> {
        if !self.root_started
            || !self.can_admit_work()
            || self.ordinary_pending_work()
            || self.has_execution_failure()
            || self.exclusive_scope.is_some()
            || self.utility_scope_open()
            || !self.role_call_admission_available()
        {
            return Ok(false);
        }
        if let Some(candidate) = self.driver.next_fixed_root_candidate()? {
            let ranking = self
                .root_ranking_proof
                .as_ref()
                .ok_or(AdaptiveManagerError::Invalid(
                    "lazy candidate has no original published ranking",
                ))?;
            let input = candidate.input.clone();
            let now = self.check_running()?;
            let frame = match (candidate.deferred, candidate.initialized) {
                (Some(deferred), None) => self.machine.admit_deferred_candidate(
                    ranking,
                    deferred,
                    self.clock.domain(),
                    now,
                )?,
                (None, Some(initialized)) => self.machine.admit_checked_candidate(
                    ranking,
                    initialized,
                    self.clock.domain(),
                    now,
                )?,
                (None, None) => self.machine.admit_candidate(
                    ranking,
                    candidate.first_move,
                    candidate.input,
                    candidate.budget,
                    self.clock.domain(),
                    now,
                )?,
                (Some(_), Some(_)) => {
                    return Err(AdaptiveManagerError::Invalid(
                        "candidate has conflicting admission proofs",
                    ));
                }
            };
            self.driver
                .root_candidate_admitted(frame, candidate.first_move, &input)?;
            self.rovent_candidate_ready(frame)?;
            return Ok(true);
        }
        if let Some(recheck) = self.driver.next_fixed_iterative_repair()? {
            let now = self.check_running()?;
            self.machine
                .enqueue_checked_recheck(recheck, self.clock.domain(), now)?;
            return Ok(true);
        }
        self.root_ranking_proof = None;
        let Some(input) = self.driver.next_root_round_input()? else {
            self.rovent_seal_root_round()?;
            return self.drive_goal_local();
        };
        self.prepare_root_ranking(input)?;
        Ok(true)
    }
    fn publish_closed_frame(&mut self, closed: &ClosedFrame) -> Result<(), AdaptiveManagerError> {
        self.driver.publish_closed(closed)?;
        if let Some(scope) = &mut self.exclusive_scope {
            if scope.frame == Some(closed.id) {
                scope.completed = true;
            }
        }
        Ok(())
    }
    fn record_failure(&mut self, error: AdaptiveManagerError) {
        if matches!(
            error,
            AdaptiveManagerError::AuxiliaryPending | AdaptiveManagerError::AdmissionClosed
        ) {
            return;
        }
        if let AdaptiveManagerError::ReviewLeaseBackpressure {
            current,
            capacity,
            required_new,
            replacement_reserve,
        } = &error
        {
            self.review_leases.record_blocked(
                *current,
                *capacity,
                *required_new,
                *replacement_reserve,
            );
            if self.review_lease_return_can_progress() {
                // The existing source packet remains in its original binding
                // or deferred queue. Wait for a real owner event; never drop
                // its view to make room or count this as semantic success.
                return;
            }
            self.review_leases.terminal_capacity = true;
        }
        if self.failures.len() == 64 {
            self.failures.pop_front();
        }
        let unsafe_completion = matches!(error, AdaptiveManagerError::PhysicalUnknown)
            || (matches!(error, AdaptiveManagerError::NnuecPreparation(_))
                && (!self.nnuec_cleanup_settled() || !self.utility_cleanup_settled()))
            || matches!(
                error,
                AdaptiveManagerError::GoalLocalBlocked(GoalLocalBlockReason::Frontier(
                    FrontierError::NativeUnknown
                ))
            )
            || matches!(
                error,
                AdaptiveManagerError::Frontier(FrontierError::NativeUnknown)
            )
            || matches!(
                error,
                AdaptiveManagerError::Cpu(
                    CpuWorkerError::UnknownUsage | CpuWorkerError::Quarantined
                )
            );
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let independent_collection = matches!(&error, AdaptiveManagerError::Collection(_))
            && self.collection_failure_is_independent();
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if independent_collection {
            if let AdaptiveManagerError::Collection(original) = &error {
                self.collection_v2_first_failure
                    .get_or_insert_with(|| original.clone());
            }
        }
        // A utility experiment depends on its collection evidence even when
        // ordinary native/search work does not. Never turn the failed target
        // into a valid utility sample merely to keep the search alive.
        self.fail_utility_route();
        self.failures.push_back(error);
        if unsafe_completion {
            self.phase = ManagerPhase::Quarantined;
        }
    }
    fn native_event(
        &mut self,
        event: TicketNativeEvent<B::Output>,
    ) -> Result<(), AdaptiveManagerError> {
        let mut collection_failures = Vec::new();
        if let Some(observer) = &mut self.collection_observer {
            if let Err(error) = observer.native_event(&event) {
                collection_failures.push(error);
            }
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let Some(observer) = &mut self.collection_observer_v2 {
            let verifier = self.native.native_acceptance_verifier();
            let result = self
                .collection_admission_v2
                .as_ref()
                .zip(verifier.as_ref())
                .ok_or_else(|| {
                    AdaptiveCollectionError::new(
                        AdaptiveCollectionStage::NativeEvent,
                        "NativeOwnerUnavailable",
                        "original admission or native verifier unavailable",
                    )
                })
                .and_then(|(admission, verifier)| {
                    collection::RoventCollectionNativeEventWitnessV2::new(
                        admission, verifier, &event,
                    )
                })
                .and_then(|witness| observer.native_event(witness));
            if let Err(error) = result {
                collection_failures.push(error);
            }
        }
        // 수집 오류가 non-Clone permit/receipt나 물리 완료 packet을 버리게
        // 해서는 안 된다. 원 runtime 처리를 먼저 마치고 오류를 따로 남긴다.
        let runtime_result = (|| -> Result<(), AdaptiveManagerError> {
            match event {
                TicketNativeEvent::Ready { clock } => {
                    if clock.domain() != self.clock.domain()
                        || clock.origin() != self.clock.origin()
                    {
                        return Err(AdaptiveManagerError::Invalid(
                            "native owner clock origin mismatch",
                        ));
                    }
                    self.native_ready = true;
                    if self.phase == ManagerPhase::Running && !self.native_verifier_registered {
                        // Startup may finish after a terminal root or cancellation
                        // has already revoked admission. Its real Ready remains a
                        // clock/owner observation; it cannot install new acceptance
                        // authority into the closed registry. The next root uses
                        // the same native owner's persistent verifier readback.
                        let verifier = self.native.native_acceptance_verifier().ok_or(
                            AdaptiveManagerError::Invalid(
                                "native Ready has no owner acceptance verifier",
                            ),
                        )?;
                        self.registry
                            .register_native_acceptance_verifier(verifier)?;
                        self.native_verifier_registered = true;
                    }
                }
                TicketNativeEvent::StartupFailed(failure) => {
                    return Err(AdaptiveManagerError::Semantic(format!(
                        "native startup: {failure:?}"
                    )));
                }
                TicketNativeEvent::Prepared(prepared) => {
                    let index = self
                        .roles
                        .iter()
                        .position(|binding| binding.preparation == prepared.preparation)
                        .ok_or(AdaptiveManagerError::Invalid("unknown native preparation"))?;
                    let binding = &self.roles[index];
                    if binding.ticket.is_some()
                        || binding.prepared_observed
                        || binding.admission_charged
                        || prepared.question != binding.input.question_digest
                        || prepared.view != binding.input.view_seal
                        || Some(prepared.role) != binding.input.stage.role()
                        || prepared.original_deadline != self.deadline
                    {
                        return Err(AdaptiveManagerError::Invalid(
                            "prepared immutable question echo mismatch",
                        ));
                    }
                    self.roles[index].prepared_observed = true;
                    if self.phase != ManagerPhase::Running || self.check_running().is_err() {
                        // Close may have canceled this same preparation before
                        // its older Prepared event reached the manager. Retain
                        // that real request; a retired control is not recanceled.
                        if !self.roles[index].cancellation_requested {
                            self.native.cancel(prepared.preparation)?;
                            self.roles[index].cancellation_requested = true;
                        }
                        self.counters.canceled_preparations += 1;
                        return Ok(());
                    }
                    let now = self.check_running()?;
                    let binding = &self.roles[index];
                    let ticket = self.registry.admit(
                        RoleSubmission {
                            question: prepared.question,
                            view: prepared.view,
                            representation: prepared.representation,
                            role: prepared.role,
                            original_deadline: prepared.original_deadline,
                        },
                        Arc::clone(&binding.input.view),
                        self.clock.domain(),
                        now,
                    )?;
                    let work = binding.work.clone();
                    let preparation_elapsed = binding.prepared_at.elapsed();
                    self.roles[index].admission_charged = true;
                    if let Some(work) = &work {
                        if matches!(
                            work.input.stage,
                            RefinementStage::FreshRepairedEndpoint
                                | RefinementStage::FreshCounterEndpoint
                        ) {
                            // Endpoint input differs, while the public scope is
                            // frozen. The actual Root Rules perspective is fixed.
                            self.machine.seal_fresh_pair(
                                work.frame,
                                FreshPairSeal::from_ticket(
                                    &ticket,
                                    work.input.public_context,
                                    self.driver.root_rules().snapshot().side_to_move(),
                                ),
                            )?;
                        }
                        self.machine
                            .bind_role(work.frame, &ticket, self.clock.domain(), now)?;
                    }
                    let class = role_cost_class(&ticket, self.registration.device_digest);
                    // Cache/Fresh ownership is known only at the actual completion.
                    // Retain this duration rather than training an assumed Miss.
                    self.roles[index].ticket = Some(ticket);
                    self.roles[index].class = Some(class);
                    self.roles[index].preparation_elapsed = Some(preparation_elapsed);
                    self.roles[index].dispatch_pending = true;
                    if let Some(frame) = self.roles[index].work.as_ref().map(|work| work.frame) {
                        self.roles[index].work = Some(
                            self.machine
                                .bound_work(frame, StepExecution::Role(ticket.id))?,
                        );
                    }
                    self.counters.submitted_role_tickets += 1;
                    self.counters.validator_ticket_admissions +=
                        u64::from(self.roles[index].purpose == RolePurpose::Validator);
                }
                TicketNativeEvent::Started(ticket) => {
                    self.registry.physical_started(&ticket)?;
                    let binding = self.role_mut(ticket.id)?;
                    if binding.ticket != Some(ticket) {
                        return Err(AdaptiveManagerError::Invalid("started ticket echo"));
                    }
                }
                TicketNativeEvent::OwnerRunStarted { ticket, started_at } => {
                    if started_at < self.clock.origin() || started_at > Instant::now() {
                        return Err(AdaptiveManagerError::Invalid(
                            "owner service clock timestamp",
                        ));
                    }
                    let binding = self.role_mut(ticket.id)?;
                    if binding.ticket != Some(ticket) || binding.dispatched_at.is_some() {
                        return Err(AdaptiveManagerError::Invalid("owner service start ticket"));
                    }
                    let admitted_at =
                        binding
                            .queue_admitted_at
                            .ok_or(AdaptiveManagerError::Invalid(
                                "owner start lacks queue admission clock",
                            ))?;
                    binding.queue_wait =
                        Some(started_at.checked_duration_since(admitted_at).ok_or(
                            AdaptiveManagerError::Invalid("owner start predates queue admission"),
                        )?);
                    binding.dispatched_at = Some(started_at);
                }
                TicketNativeEvent::Completed {
                    completion,
                    evidence,
                    owner_run,
                    owner_service_elapsed,
                    observation_failures,
                } => {
                    let retained = RetainedNativeCompletion {
                        completion: RoleCompletion {
                            ticket: completion.ticket,
                            physical: completion.physical,
                            result: completion.result.clone(),
                        },
                        evidence,
                        owner_run,
                        owner_service_elapsed,
                    };
                    if let Some(error) = self.observe_native_failures(
                        &completion.ticket,
                        &observation_failures,
                        false,
                    ) {
                        self.retained_native_completions.push(retained);
                        return Err(error);
                    }
                    #[cfg(feature = "pals-adaptive-diagnostics")]
                    let diagnostic_completion_ticket = completion.ticket.id;
                    let outcome = (|| -> Result<(), AdaptiveManagerError> {
                        let ticket = completion.ticket;
                        let physical = completion.physical;
                        let success = completion.result.is_ok();
                        let failure = completion.result.as_ref().err().cloned();
                        let binding = self
                            .roles
                            .iter_mut()
                            .find(|binding| binding.ticket == Some(ticket))
                            .ok_or(AdaptiveManagerError::Invalid(
                                "completion has no original native binding",
                            ))?;
                        if binding.service_completed {
                            // Reject the repeated source event before replacing
                            // the original evidence or charging raw NN inputs.
                            return Err(AdaptiveManagerError::Invalid(
                                "duplicate original native completion",
                            ));
                        }
                        if owner_run.ticket() != &ticket {
                            return Err(AdaptiveManagerError::Invalid(
                                "completion owner-run witness differs from original ticket",
                            ));
                        }
                        binding.physical = evidence;
                        binding.service_completed = true;
                        let mut class = binding.class;
                        if let (Some(class), Some(evidence)) = (&mut class, evidence) {
                            if let Some(fresh) = evidence.physical_fresh {
                                class.invocation = if fresh {
                                    InvocationClass::Fresh
                                } else {
                                    InvocationClass::Warm
                                };
                            }
                            class.cache = observed_public_cache_class(physical, evidence);
                        }
                        binding.class = class;
                        let preparation_elapsed = binding.preparation_elapsed.take();
                        let queue_wait = binding.queue_wait.take();
                        // Receive-time differences can collapse to zero when Started
                        // and Completed share one manager turn. Use owner measurement.
                        // A public cache hit may still run a private role head.
                        // Zero/unknown NN inputs alone never decide cache class.
                        let elapsed = if evidence.is_some_and(|actual| {
                            actual.measurement_complete
                                && actual.completed_nn_inputs.is_some()
                                && actual.physical_fresh.is_some()
                        }) {
                            owner_service_elapsed
                        } else {
                            None
                        };
                        let purpose = binding.purpose;
                        self.native_accounting
                            .observe_completion(&ticket, &owner_run, physical, success, evidence)?;
                        let disposition = if physical == PhysicalRoleCompletion::Unknown {
                            CostDisposition::Censored(CensorReason::PhysicalUnknown)
                        } else if success {
                            CostDisposition::Completed
                        } else {
                            CostDisposition::Censored(CensorReason::Failed)
                        };
                        if let Some(class) = class {
                            if disposition == CostDisposition::Completed && elapsed.is_some() {
                                let mut query = class;
                                query.invocation = InvocationClass::Fresh;
                                query.cache = CacheClass::Unknown;
                                if !self.observed_role_classes.contains(&(ticket.role, query))
                                    && self.observed_role_classes.len() < 64
                                {
                                    self.observed_role_classes.push((ticket.role, query));
                                }
                            }
                            for (stage, elapsed) in [
                                (CostStage::Preparation, preparation_elapsed),
                                (CostStage::QueueWait, queue_wait),
                            ] {
                                self.observe_observation(
                                    class,
                                    stage,
                                    CostObservation {
                                        elapsed: elapsed
                                            .map_or(ObservedCost::Unknown, ObservedCost::Known),
                                        disposition: if elapsed.is_some() {
                                            CostDisposition::Completed
                                        } else {
                                            disposition
                                        },
                                    },
                                )?;
                            }
                            self.observe_observation(
                                class,
                                CostStage::GpuService,
                                CostObservation {
                                    elapsed: elapsed
                                        .map_or(ObservedCost::Unknown, ObservedCost::Known),
                                    disposition,
                                },
                            )?;
                            // No backend phase witness separates the physical fence
                            // wait from service. Do not relabel service as this cost.
                            self.observe_observation(
                                class,
                                CostStage::PhysicalCompletionWait,
                                CostObservation {
                                    elapsed: ObservedCost::Unknown,
                                    disposition,
                                },
                            )?;
                        }
                        if let Some(evidence) = evidence {
                            if evidence.measurement_complete {
                                if let Some(count) = evidence.completed_nn_inputs {
                                    self.counters.completed_nn_inputs = self
                                        .counters
                                        .completed_nn_inputs
                                        .checked_add(count)
                                        .ok_or(AdaptiveManagerError::Capacity(
                                            "completed NN input count",
                                        ))?;
                                } else {
                                    self.counters.nn_input_count_unknown += 1;
                                }
                            } else {
                                self.counters.nn_input_count_unknown += 1;
                            }
                        } else if owner_run.backend_run_invoked()
                            || physical != PhysicalRoleCompletion::Confirmed
                            || success
                        {
                            self.counters.nn_input_count_unknown += 1;
                        }
                        self.counters.physical_role_completions +=
                            u64::from(physical == PhysicalRoleCompletion::Confirmed);
                        let completed = match purpose {
                            RolePurpose::RootRanking => {
                                self.registry.complete_proposer_ranking(completion)
                            }
                            RolePurpose::Validator => {
                                self.registry.complete_validator_task(completion)
                            }
                            RolePurpose::Refinement => self.registry.complete(completion),
                            RolePurpose::Reciprocal => {
                                self.registry.complete_reciprocal_role(completion)
                            }
                            RolePurpose::RoventRefinement
                            | RolePurpose::RoventFresh
                            | RolePurpose::RoventGoalLocal => {
                                self.registry.complete_reciprocal_role(completion)
                            }
                        };
                        if let Err(rejected) = completed {
                            self.counters.rejected_events += 1;
                            if rejected.completion.physical == PhysicalRoleCompletion::Unknown {
                                // Worker protocol bounds this storage. On corruption,
                                // retaining is mandatory; do not invent a fence.
                                self.quarantine_rejections.push(rejected.completion);
                                return Err(AdaptiveManagerError::PhysicalUnknown);
                            }
                            return Err(AdaptiveManagerError::Coordinator(rejected.error));
                        }
                        if physical == PhysicalRoleCompletion::Unknown {
                            return Err(AdaptiveManagerError::PhysicalUnknown);
                        }
                        if self.phase == ManagerPhase::Running {
                            if let Some(failure) = failure {
                                return Err(AdaptiveManagerError::Semantic(format!(
                                    "native role execution: {failure:?}"
                                )));
                            }
                        }
                        Ok(())
                    })();
                    if let Err(error) = outcome {
                        // Preserve unknown binding, observation overflow and
                        // registry rejection just as carefully as backend errors.
                        self.retained_native_completions.push(retained);
                        return Err(error);
                    }
                    #[cfg(feature = "pals-adaptive-diagnostics")]
                    self.observe_pending_diagnostic(
                        ContinuationPendingSource::CompletionProcessed,
                        diagnostic_completion_ticket,
                        None,
                    );
                }
                TicketNativeEvent::PreparationFailed {
                    preparation,
                    failure,
                } => {
                    if let Some(index) = self
                        .roles
                        .iter()
                        .position(|binding| binding.preparation == preparation)
                    {
                        if self.roles[index].ticket.is_none()
                            && self.roles[index].goal_local.is_none()
                        {
                            self.roles.remove(index);
                        } else {
                            return Err(AdaptiveManagerError::Invalid(
                                "prepared ticket failed after issue",
                            ));
                        }
                    }
                    if self.phase == ManagerPhase::Running {
                        return Err(AdaptiveManagerError::Semantic(format!(
                            "native preparation: {failure:?}"
                        )));
                    }
                }
                TicketNativeEvent::Rejected { preparation, error } => {
                    if self.phase != ManagerPhase::Running {
                        if let Some(index) = self.roles.iter().position(|binding| {
                            binding.preparation == preparation
                                && binding.ticket.is_none()
                                && binding.goal_local.is_none()
                        }) {
                            self.roles.remove(index);
                        }
                    } else {
                        return Err(AdaptiveManagerError::Native(error));
                    }
                }
                TicketNativeEvent::Retired {
                    ticket,
                    accepted,
                    result,
                    native_receipt,
                    rejected_permit,
                    observation_failures,
                } => {
                    if let Some(error) =
                        self.observe_native_failures(&ticket, &observation_failures, true)
                    {
                        if let Some(receipt) = native_receipt {
                            self.retained_native_receipts.push(receipt);
                        }
                        if let Some(permit) = rejected_permit {
                            self.retained_native_permits.push(permit);
                        }
                        return Err(error);
                    }
                    let index = self
                        .roles
                        .iter()
                        .position(|binding| binding.ticket == Some(ticket));
                    let Some(index) = index else {
                        // A malformed/late echo cannot discard linear owner
                        // evidence. Keep it quarantined until a real fence.
                        if let Some(permit) = rejected_permit {
                            self.retained_native_permits.push(permit);
                        }
                        if let Some(receipt) = native_receipt {
                            self.retained_native_receipts.push(receipt);
                        }
                        return Err(AdaptiveManagerError::Invalid("unknown native retirement"));
                    };
                    let revoked_ack = rejected_permit.as_ref().is_some_and(|permit| {
                        self.is_revoked_acceptance_ack(
                            index,
                            &ticket,
                            accepted,
                            &result,
                            permit,
                            native_receipt.is_some(),
                        )
                    });
                    if let Some(permit) = rejected_permit {
                        self.retained_native_permits.push(permit);
                    }
                    if revoked_ack {
                        // The owner rejected this exact queued Accept before
                        // successful backend acceptance. Preserve its original
                        // output/permit and typed reason until the real fence;
                        // do not create an accepted step or semantic retirement.
                        self.roles[index].revoked_acceptance =
                            Some(result.expect_err("validated original revoked Accept"));
                        return Ok(());
                    }
                    if let Err(failure) = result {
                        if let Some(receipt) = native_receipt {
                            self.retained_native_receipts.push(receipt);
                        }
                        // A failed accept reports accepted=false. Preserve the
                        // original output until the owner's discard/fence rather
                        // than treating the echo as an accepted seed.
                        return Err(if failure == RoleFailure::PhysicalUnknown {
                            AdaptiveManagerError::PhysicalUnknown
                        } else {
                            AdaptiveManagerError::Semantic(format!(
                                "native retirement: {failure:?}"
                            ))
                        });
                    }
                    if self.roles[index].ack_accepted != Some(accepted) {
                        if let Some(receipt) = native_receipt {
                            self.retained_native_receipts.push(receipt);
                        }
                        return Err(AdaptiveManagerError::Invalid(
                            "native acceptance ack mismatch",
                        ));
                    }
                    if accepted && !self.roles[index].native_acceptance_recorded {
                        let mut native_accounting = self.native_accounting;
                        let accounting_result = (|| {
                            native_accounting.observe_owner_acceptance(
                                ticket.role,
                                self.roles[index].physical,
                            )?;
                            let count = self.roles[index]
                                .physical
                                .filter(|physical| physical.measurement_complete)
                                .and_then(|physical| physical.completed_nn_inputs);
                            let (inputs, unknown) = match count {
                                Some(count) => (
                                    self.counters.accepted_nn_inputs.checked_add(count),
                                    Some(self.counters.accepted_nn_input_count_unknown),
                                ),
                                None => (
                                    Some(self.counters.accepted_nn_inputs),
                                    self.counters.accepted_nn_input_count_unknown.checked_add(1),
                                ),
                            };
                            Ok::<_, AdaptiveManagerError>((
                                inputs.ok_or(AdaptiveManagerError::Capacity(
                                    "native accepted NN count",
                                ))?,
                                unknown.ok_or(AdaptiveManagerError::Capacity(
                                    "native accepted unknown count",
                                ))?,
                            ))
                        })();
                        let (inputs, unknown) = match accounting_result {
                            Ok(counts) => counts,
                            Err(error) => {
                                if let Some(receipt) = native_receipt {
                                    self.retained_native_receipts.push(receipt);
                                }
                                return Err(error);
                            }
                        };
                        self.counters.accepted_nn_inputs = inputs;
                        self.counters.accepted_nn_input_count_unknown = unknown;
                        self.roles[index].native_acceptance_recorded = true;
                        self.native_accounting = native_accounting;
                    }
                    let published_head = matches!(
                        self.roles[index].purpose,
                        RolePurpose::RootRanking | RolePurpose::Validator
                    );
                    if accepted
                        && matches!(
                            self.roles[index].purpose,
                            RolePurpose::Reciprocal
                                | RolePurpose::RoventRefinement
                                | RolePurpose::RoventFresh
                                | RolePurpose::RoventGoalLocal
                        )
                    {
                        let receipt = native_receipt.ok_or(AdaptiveManagerError::Invalid(
                            "reciprocal acceptance has no actual native-owner receipt",
                        ))?;
                        if let (ManagerPhase::Running, Ok(now)) = (self.phase, self.check_running())
                        {
                            match self.registry.accept_reciprocal_native(
                                receipt,
                                self.clock.domain(),
                                now,
                            ) {
                                Ok(accepted_step) => {
                                    self.roles[index].reciprocal_acceptance = Some(accepted_step);
                                    if let Some(observer) = &mut self.collection_observer {
                                        let binding = &self.roles[index];
                                        if let (Some(proof), Some(output)) =
                                            (&binding.reciprocal_acceptance, &binding.delivered)
                                        {
                                            if let Err(error) = observer.accepted(
                                                proof,
                                                &binding.input,
                                                output.as_ref(),
                                            ) {
                                                collection_failures.push(error);
                                            }
                                        } else {
                                            collection_failures.push(AdaptiveCollectionError::new(AdaptiveCollectionStage::Accepted,"MissingOriginalOutput","accepted native proof has no original retained output"));
                                        }
                                    }
                                    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                                    let collection_verifier =
                                        self.native.native_acceptance_verifier();
                                    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                                    if let Some(observer) = &mut self.collection_observer_v2 {
                                        let binding = &self.roles[index];
                                        if let (Some(proof), Some(output), Some(admission)) = (
                                            &binding.reciprocal_acceptance,
                                            &binding.delivered,
                                            &self.collection_admission_v2,
                                        ) {
                                            let raw = output.proposer_ranking_output();
                                            let result = collection_verifier
                                                .as_ref()
                                                .ok_or_else(|| {
                                                    AdaptiveCollectionError::new(
                                                        AdaptiveCollectionStage::Accepted,
                                                        "NativeVerifierUnavailable",
                                                        "V2 original native verifier unavailable",
                                                    )
                                                })
                                                .and_then(|verifier| {
                                                    RoventCollectionAcceptedWitnessV2::new(
                                                        admission, verifier, proof, raw,
                                                    )
                                                })
                                                .and_then(|witness| {
                                                    observer.accepted(
                                                        witness,
                                                        &binding.input,
                                                        output.as_ref(),
                                                    )
                                                });
                                            if let Err(error) = result {
                                                collection_failures.push(error);
                                            }
                                        } else {
                                            collection_failures.push(AdaptiveCollectionError::new(AdaptiveCollectionStage::Accepted,
                                                "MissingOriginalOutput", "V2 accepted proof has no original output or admission"));
                                        }
                                    }
                                }
                                Err(rejected) => {
                                    self.retained_native_receipts.push(rejected.receipt);
                                    return Err(rejected.error.into());
                                }
                            }
                        } else {
                            self.retained_native_receipts.push(receipt);
                        }
                    } else if accepted && published_head {
                        let receipt = native_receipt.ok_or(AdaptiveManagerError::Invalid(
                            "P/V acceptance has no actual native-owner receipt",
                        ))?;
                        if let (ManagerPhase::Running, Ok(now)) = (self.phase, self.check_running())
                        {
                            if let Err(rejected) = self.registry.confirm_native_acceptance(
                                receipt,
                                self.clock.domain(),
                                now,
                            ) {
                                self.retained_native_receipts.push(rejected.receipt);
                                return Err(rejected.error.into());
                            }
                        } else {
                            // Logical revocation never turns a late owner success
                            // into new search-consumption permission.
                            self.retained_native_receipts.push(receipt);
                        }
                    } else if let Some(receipt) = native_receipt {
                        self.retained_native_receipts.push(receipt);
                        return Err(AdaptiveManagerError::Invalid(
                            "unexpected native publication receipt",
                        ));
                    }
                    // Retirement may need an archive owner turn before its hot
                    // commit. Keep the original ticket/binding until that barrier
                    // succeeds, including when receipt delivery raced shutdown.
                    self.defer(DeferredCommit::Role { ticket, accepted })?;
                }
                TicketNativeEvent::AcceptanceRejected {
                    preparation,
                    ticket,
                    error,
                    permit,
                } => {
                    // This linear authority must survive failure until the actual
                    // root/process fence; reissuing a permit would be a duplicate.
                    let original = self.roles.iter().any(|binding| {
                        binding.preparation == preparation && binding.ticket == Some(ticket)
                    });
                    self.retained_native_permits.push(permit);
                    if !original {
                        return Err(AdaptiveManagerError::Invalid(
                            "unknown native acceptance rejection",
                        ));
                    }
                    return Err(error.into());
                }
                TicketNativeEvent::Finished {
                    physical_shutdown_confirmed,
                } => {
                    if self.native_finished.is_some() {
                        return Err(AdaptiveManagerError::Invalid("duplicate native shutdown"));
                    }
                    self.native_finished = Some(physical_shutdown_confirmed);
                    if !physical_shutdown_confirmed {
                        return Err(AdaptiveManagerError::PhysicalUnknown);
                    }
                }
                TicketNativeEvent::RootQuiesced(fence) => {
                    if !self.native_root_quiesce_requested
                        || fence.authority() != self.registration.authority
                        || self.native_root_fence.is_some()
                    {
                        return Err(AdaptiveManagerError::Invalid(
                            "unexpected native root fence",
                        ));
                    }
                    self.native_root_fence = Some(fence);
                    if !fence.permits_reuse() {
                        return Err(AdaptiveManagerError::PhysicalUnknown);
                    }
                }
            }
            Ok(())
        })();
        for error in collection_failures {
            self.record_failure(error.into());
        }
        runtime_result
    }
    fn role_mut(
        &mut self,
        request: RequestId,
    ) -> Result<&mut RoleBinding<Q, B::Output>, AdaptiveManagerError> {
        self.roles
            .iter_mut()
            .find(|binding| binding.ticket.is_some_and(|ticket| ticket.id == request))
            .ok_or(AdaptiveManagerError::Invalid("unknown role request"))
    }
    fn is_revoked_acceptance_ack(
        &mut self,
        index: usize,
        ticket: &RoleTicketEnvelope,
        accepted: bool,
        result: &Result<(), RoleFailure>,
        permit: &NativeAcceptancePermit,
        has_native_receipt: bool,
    ) -> bool {
        let binding = &self.roles[index];
        let actual_expiry = self.clock.try_now().ok().is_some_and(|now| {
            self.clock.domain() == ticket.original_deadline.clock
                && now.0 >= ticket.original_deadline.at.0
        });
        let cancellation_failure = match result {
            Err(RoleFailure::Canceled) => true,
            Err(RoleFailure::Expired) => actual_expiry,
            _ => false,
        };
        if self.phase != ManagerPhase::Draining
            || self.shutdown_until.is_none()
            || ticket.authority != self.registration.authority
            || ticket.original_deadline != self.deadline
            || binding.ticket != Some(*ticket)
            || accepted
            || has_native_receipt
            || !binding.cancellation_requested
            || !binding.ack_dispatched
            || binding.ack_accepted != Some(true)
            || binding.revoked_acceptance.is_some()
            || binding.native_acceptance_recorded
            || binding.consumption_recorded
            || binding.reciprocal_acceptance.is_some()
            || !binding.service_completed
            || !cancellation_failure
            || binding.delivered.as_ref().and_then(|output| {
                rz_search::pals::coordinator::native_acceptance_output_digest(
                    output.proposer_ranking_output(),
                )
                .ok()
            }) != Some(permit.output_digest())
        {
            return false;
        }
        let purpose = binding.purpose;
        // Delivered + successful payload + original issued seal in the
        // registry also proves Confirmed physical completion: Unknown enters
        // Quarantined and cannot issue or satisfy either permit validator.
        match purpose {
            RolePurpose::RootRanking | RolePurpose::Validator => self
                .registry
                .validate_revoked_published_head_native_permit(ticket, permit)
                .is_ok(),
            RolePurpose::Reciprocal
            | RolePurpose::RoventRefinement
            | RolePurpose::RoventFresh
            | RolePurpose::RoventGoalLocal => self
                .registry
                .validate_revoked_reciprocal_native_permit(ticket, permit)
                .is_ok(),
            RolePurpose::Refinement => false,
        }
    }
    fn retire_role(
        &mut self,
        ticket: RoleTicketEnvelope,
        accepted: bool,
    ) -> Result<(), AdaptiveManagerError> {
        let acceptance_started = Instant::now();
        let index = self
            .roles
            .iter()
            .position(|binding| binding.ticket == Some(ticket))
            .ok_or(AdaptiveManagerError::Invalid("unknown semantic retirement"))?;
        if self.roles[index].consumption_unknown {
            return Err(AdaptiveManagerError::Invalid(
                "semantic consumption remains unknown",
            ));
        }
        if accepted
            && self.roles[index].purpose == RolePurpose::RoventRefinement
            && self.phase == ManagerPhase::Running
            && self.check_running().is_ok()
        {
            return self.retain_accepted_role_for_peer(index);
        }
        if accepted && self.phase == ManagerPhase::Running && self.check_running().is_ok() {
            // Fallible hot commits must not erase the original immutable
            // question or owner evidence. A deferred retry/drain still owns
            // the same binding until every required semantic step finishes.
            let binding = &self.roles[index];
            let class = binding.class;
            let purpose = binding.purpose;
            let input = binding.input.clone();
            let work = binding.work.clone();
            let physical = binding.physical;
            let output =
                binding
                    .delivered
                    .as_ref()
                    .cloned()
                    .ok_or(AdaptiveManagerError::Invalid(
                        "accepted native ack has no output",
                    ))?;
            if let Some(work) = &work {
                self.utility_begin_commit(work, StepExecution::Role(ticket.id))?;
            }
            let utility_work = work.clone();
            let attempted = (|| -> Result<(), AdaptiveManagerError> {
                if purpose == RolePurpose::RoventGoalLocal {
                    self.consume_goal_local_role(
                        index,
                        &ticket,
                        output.proposer_ranking_output(),
                        physical,
                    )?;
                } else if purpose == RolePurpose::Reciprocal {
                    let work = work.ok_or(AdaptiveManagerError::Invalid(
                        "reciprocal role lost its exact child frame",
                    ))?;
                    self.consume_reciprocal_role(
                        index,
                        &work,
                        &ticket,
                        output.proposer_ranking_output(),
                        physical,
                    )?;
                } else if let Some(work) = work {
                    if purpose == RolePurpose::RoventFresh {
                        let accepted = self.roles[index].reciprocal_acceptance.as_ref().ok_or(
                            AdaptiveManagerError::Invalid(
                                "Fresh endpoint lost original retired native proof",
                            ),
                        )?;
                        self.driver.rovent_fresh_accepted(
                            &work,
                            accepted,
                            output.proposer_ranking_output(),
                            physical,
                            self.clock.domain(),
                            self.check_running()?,
                        )?;
                    }
                    let committed = self.driver.commit_role(
                        &work,
                        &ticket,
                        output.proposer_ranking_output(),
                        physical,
                    )?;
                    self.driver.role_consumed(&ticket, &input.view)?;
                    if purpose != RolePurpose::RoventFresh {
                        self.registry.accept(
                            &ticket,
                            self.clock.domain(),
                            self.check_running()?,
                        )?;
                    }
                    self.mark_role_consumed(index, false)?;
                    if self.has_collection_failure() {
                        return Ok(());
                    }
                    self.utility_role_at(index, &work, output.proposer_ranking_output())?;
                    self.resolve(work, StepExecution::Role(ticket.id), committed)?;
                } else if purpose == RolePurpose::RootRanking {
                    let ranking = self.registry.accept_proposer_ranking(
                        &ticket,
                        self.driver.root_rules(),
                        self.clock.domain(),
                        self.check_running()?,
                    )?;
                    self.root_ranking_proof = Some(ranking.clone());
                    let candidates = self.driver.ranking_published(&ranking)?;
                    if candidates.len() > 64 {
                        return Err(AdaptiveManagerError::Capacity("root candidate fanout"));
                    }
                    self.driver.role_consumed(&ticket, &input.view)?;
                    self.mark_role_consumed(index, false)?;
                    let mut ranking_observation_failed = false;
                    // Collection observes the actual root ranking even when
                    // no optional reciprocal/peer routing was configured.
                    if self.reciprocal_routing.is_some()
                        || self.peer_policy.is_some()
                        || self.collection_observer.is_some()
                        || self.has_collection_v2()
                    {
                        if let Some(observer) = &mut self.collection_observer {
                            let proof = match self.collection_admission.as_ref() {
                                Some(admission) => {
                                    RoventRootRankingAcceptance::from_current_ranking(
                                        &ranking, admission,
                                    )
                                }
                                None => Err(AdaptiveCollectionError::new(
                                    AdaptiveCollectionStage::Accepted,
                                    "MissingRoventRootAdmission",
                                    "root ranking has no original Rovent collection admission",
                                )),
                            };
                            let result = proof.and_then(|proof| {
                                observer.root_ranking(&proof, &input, output.as_ref())
                            });
                            if let Err(error) = result {
                                ranking_observation_failed = true;
                                self.record_failure(error.into());
                            }
                        }
                        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                        if let Some(observer) = &mut self.collection_observer_v2 {
                            let result = self
                                .collection_admission_v2
                                .as_ref()
                                .ok_or_else(|| {
                                    AdaptiveCollectionError::new(
                                        AdaptiveCollectionStage::Accepted,
                                        "MissingV2Admission",
                                        "V2 root ranking has no original admission",
                                    )
                                })
                                .and_then(|admission| {
                                    admission.validate_output(
                                        &ticket,
                                        output.proposer_ranking_output(),
                                    )?;
                                    RoventRootRankingAcceptanceV2::from_current_ranking(
                                        &ranking, admission,
                                    )
                                })
                                .and_then(|proof| {
                                    observer.adaptive_root_ranking(&proof, &input, output.as_ref())
                                });
                            if let Err(error) = result {
                                ranking_observation_failed = true;
                                self.record_failure(error.into());
                            }
                        }
                    }
                    // Ranking and search consumption have already happened.
                    // An optional V2 collection failure cannot undo them;
                    // strict/legacy collection still stops candidate fanout.
                    if ranking_observation_failed && !self.collection_failure_is_independent() {
                        return Ok(());
                    }
                    for candidate in candidates {
                        let input = candidate.input.clone();
                        let now = self.check_running()?;
                        let frame = match (candidate.deferred, candidate.initialized) {
                            (Some(deferred), None) => self.machine.admit_deferred_candidate(
                                &ranking,
                                deferred,
                                self.clock.domain(),
                                now,
                            )?,
                            (None, Some(initialized)) => self.machine.admit_checked_candidate(
                                &ranking,
                                initialized,
                                self.clock.domain(),
                                now,
                            )?,
                            (None, None) => self.machine.admit_candidate(
                                &ranking,
                                candidate.first_move,
                                candidate.input,
                                candidate.budget,
                                self.clock.domain(),
                                now,
                            )?,
                            (Some(_), Some(_)) => {
                                return Err(AdaptiveManagerError::Invalid(
                                    "candidate has conflicting admission proofs",
                                ));
                            }
                        };
                        self.driver
                            .root_candidate_admitted(frame, candidate.first_move, &input)?;
                        self.rovent_candidate_ready(frame)?;
                    }
                } else if purpose == RolePurpose::Validator {
                    let published = self.registry.accept_validator_task(
                        &ticket,
                        self.clock.domain(),
                        self.check_running()?,
                    )?;
                    let proof = AcceptedValidator::from_published(published);
                    let plan = self.driver.validator_accepted(&input, proof)?;
                    let id = plan.id();
                    let parent = plan.parent();
                    if let Some(scope) = &self.exclusive_scope {
                        if scope.parent != parent
                            || scope.action.is_some()
                            || plan.source() != &ticket
                        {
                            return Err(AdaptiveManagerError::Invalid(
                                "exclusive action source/parent changed",
                            ));
                        }
                        if self.registered_action.is_some_and(|action| {
                            plan.action() != action
                                || plan.selection_provenance().name()
                                    != "registered-supported-enumeration-v1"
                        }) {
                            return Err(AdaptiveManagerError::Invalid(
                                "registered action differs from actual issued supported plan",
                            ));
                        }
                    }
                    let question = ExclusiveQuestionSeal {
                        parent_question: plan.parent_question(),
                        parent_view: plan.parent_view(),
                        parent_input: plan.parent_input(),
                        parent_public_context: plan.parent_public_context(),
                        frozen_context: plan.frozen_context_digest(),
                        support: plan.action_support_mask(),
                    };
                    let admission = self.machine.admit_validator_action(
                        plan,
                        self.clock.domain(),
                        self.check_running()?,
                    )?;
                    self.driver.validator_action_admitted(id, &admission)?;
                    if let Some(scope) = &mut self.exclusive_scope {
                        scope.source = Some(ticket);
                        scope.action = Some(id);
                        scope.question = Some(question);
                        scope.frame = match admission {
                            rz_search::pals::coordinator::ValidatorActionAdmission::Frame(
                                frame,
                            )
                            | rz_search::pals::coordinator::ValidatorActionAdmission::Resumed(
                                frame,
                            ) => Some(frame),
                            rz_search::pals::coordinator::ValidatorActionAdmission::Defer => None,
                        };
                        scope.completed = scope.frame.is_none();
                        // Keep the original parent held. Only the one issued
                        // action may dispatch; parent work is revoked at closure.
                    } else {
                        self.validator_waiting.retain(|frame| *frame != parent);
                    }
                    self.driver.role_consumed(&ticket, &input.view)?;
                    self.mark_role_consumed(index, false)?;
                } else {
                    return Err(AdaptiveManagerError::Invalid(
                        "role purpose lacks refinement work",
                    ));
                }
                if let Some(class) = class {
                    self.observe(
                        class,
                        CostStage::Acceptance,
                        acceptance_started.elapsed(),
                        CostDisposition::Completed,
                    )?;
                }
                Ok(())
            })();
            if let Err(error) = attempted {
                // Store consumption precedes next-view materialization. A
                // later failure must retain both the semantic error and the
                // actual source consumption, without retrying its commit.
                let consumed = if purpose == RolePurpose::RoventGoalLocal {
                    self.driver
                        .rovent_goal_consumption_recorded(&ticket, &input.view)
                } else {
                    self.driver.role_consumption_recorded(&ticket, &input.view)
                };
                match consumed {
                    Ok(true) if !self.roles[index].consumption_recorded => {
                        if let Err(observation) = self.mark_role_consumed(index, true) {
                            self.record_failure(observation);
                        }
                    }
                    Ok(_) => {}
                    Err(readback) => {
                        if !self.roles[index].consumption_recorded {
                            self.roles[index].consumption_unknown = true;
                        }
                        self.record_failure(readback);
                    }
                }
                if self.roles[index].consumption_recorded {
                    if let Some(work) = &utility_work {
                        if let Err(observation) =
                            self.utility_role_at(index, work, output.proposer_ranking_output())
                        {
                            self.record_failure(observation);
                        }
                    }
                }
                return Err(error);
            }
            self.roles.remove(index);
        } else {
            // A prior successful acceptance may have retired the registry
            // before a later Store/transition error. Its original binding is
            // still retained here; closure never invents another consumption.
            if self.registry.ticket(ticket.id).is_some() {
                self.registry.discard_delivered(&ticket)?;
            }
            let consumed = self.roles[index].consumption_recorded;
            if let Some(original) = &self.roles[index].goal_local {
                self.driver.discard_rovent_goal_input(original)?;
            }
            self.roles.remove(index);
            if !consumed {
                self.counters.unused_role_results += 1;
            }
        }
        Ok(())
    }
    fn mark_role_consumed(
        &mut self,
        index: usize,
        after_error_readback: bool,
    ) -> Result<(), AdaptiveManagerError> {
        if self.roles[index].consumption_recorded {
            return Ok(());
        }
        // Driver confirmed consumption before this method. Preserve that fact
        // even if the separate accounting or observer path fails.
        self.roles[index].consumption_recorded = true;
        let ticket = self.roles[index]
            .ticket
            .ok_or(AdaptiveManagerError::Invalid(
                "original consumed role ticket missing",
            ))?;
        let physical = self.roles[index].physical;
        self.record_role_consumption(ticket, physical)?;
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let (Some(accepted), Some(output)) = (
            self.roles[index].reciprocal_acceptance.as_ref(),
            self.roles[index].delivered.as_ref(),
        ) {
            let verifier = self.native.native_acceptance_verifier();
            let observed = collection::notify_adaptive_consumed_v2(
                &mut self.collection_observer_v2,
                self.collection_admission_v2.as_ref(),
                verifier.as_ref(),
                accepted,
                output.proposer_ranking_output(),
                after_error_readback,
            );
            if let Err(error) = observed {
                self.record_failure(error.into());
            }
        } else if self.collection_observer_v2.is_some()
            && self.roles[index].reciprocal_acceptance.is_some()
        {
            self.record_failure(
                AdaptiveCollectionError::new(
                    AdaptiveCollectionStage::Accepted,
                    "MissingOriginalOutput",
                    "V2 consumed proof has no retained output",
                )
                .into(),
            );
        }
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        let _ = after_error_readback;
        let failures = if let (Some(state), Some(accepted), Some(output)) = (
            self.cpu_semantic.as_mut(),
            self.roles[index].reciprocal_acceptance.as_ref(),
            self.roles[index].delivered.as_ref(),
        ) {
            cpu_semantic::promote_selected(
                state,
                &self.driver,
                &self.roles[index].input,
                accepted,
                output.proposer_ranking_output(),
            )
        } else {
            Vec::new()
        };
        for failure in failures {
            self.record_cpu_semantic_failure(failure);
        }
        self.poll_cpu_semantic();
        Ok(())
    }
    fn record_role_consumption(
        &mut self,
        ticket: RoleTicketEnvelope,
        physical: Option<TicketExecutionEvidence>,
    ) -> Result<(), AdaptiveManagerError> {
        let mut counters = self.counters;
        let mut native_accounting = self.native_accounting;
        native_accounting.observe_search_consumption(ticket.role, physical)?;
        counters.consumed_role_outputs = counters
            .consumed_role_outputs
            .checked_add(1)
            .ok_or(AdaptiveManagerError::Capacity("role consumption count"))?;
        match physical
            .filter(|physical| physical.measurement_complete)
            .and_then(|physical| physical.completed_nn_inputs)
        {
            Some(count) => {
                counters.consumed_nn_inputs = counters
                    .consumed_nn_inputs
                    .checked_add(count)
                    .ok_or(AdaptiveManagerError::Capacity("consumed NN input count"))?;
                counters.zero_nn_outputs_consumed += u64::from(count == 0);
            }
            None => counters.consumed_nn_input_count_unknown += 1,
        }
        self.counters = counters;
        self.native_accounting = native_accounting;
        Ok(())
    }
    fn defer(&mut self, event: DeferredCommit<Q>) -> Result<(), AdaptiveManagerError> {
        // Physical owners bound this to sixteen roles, four CPU tasks, one
        // external task and at most sixty-four candidate terminal reports.
        if self.deferred.len() >= 128 {
            return Err(AdaptiveManagerError::Capacity(
                "archive handoff commit buffer",
            ));
        }
        self.deferred.push_back(event);
        Ok(())
    }
    fn flush_deferred(&mut self) -> Result<bool, AdaptiveManagerError> {
        // The root common ledger receives only the real native ledger's
        // opaque success receipt, before any later semantic CPU publication.
        let mut changed = self.flush_frontier_native()?;
        for index in 0..self.own.len() {
            if self.own[index]
                .binding
                .as_ref()
                .is_some_and(|binding| binding.queued_resume_termination.is_some())
            {
                // These two original receipts are one retirement obligation.
                // Generic raw publication cannot consume just one side while
                // an archive transfer/frontier settlement is still pending.
                if self.own[index]
                    .binding
                    .as_ref()
                    .is_some_and(|binding| binding.pending_slice.is_some())
                {
                    changed |= self.retire_queued_resume_after_owner_termination(index)?;
                }
                continue;
            }
            if !self.driver.semantic_owner_available() {
                break;
            }
            if let Some(binding) = &mut self.own[index].binding {
                if let Some(report) = binding.pending_slice.take() {
                    let observed = if binding.feedback_forwarded {
                        self.driver
                            .cpu_detached_slice_observed(&binding.work, &report)
                    } else {
                        self.driver.cpu_slice_observed(&binding.work, &report)
                    };
                    if let Err(error) = observed {
                        binding.pending_slice = Some(report);
                        return Err(error);
                    }
                    changed = true;
                }
            }
        }
        while self.driver.semantic_owner_available() {
            if self.phase == ManagerPhase::Running && !self.driver.prepare_semantic_commit()? {
                break;
            }
            let Some(mut event) = self.deferred.pop_front() else {
                break;
            };
            let changed_before_event = changed;
            let consumed_before_event = self.counters.consumed_role_outputs;
            changed = true;
            let result = match &mut event {
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                DeferredCommit::ProfiledExternal { binding, packet } => {
                    self.commit_profiled_packet(binding, packet)
                }
                DeferredCommit::Role { ticket, accepted } => self.retire_role(*ticket, *accepted),
                DeferredCommit::Own { binding, report } => {
                    if self.phase == ManagerPhase::Running {
                        self.commit_own(binding, report)
                    } else {
                        self.driver
                            .cpu_slice_observed(&binding.work, report)
                            .and_then(|_| self.release_nnuec_report(&binding.work, report))
                    }
                }
                DeferredCommit::OwnNoValue {
                    binding,
                    report,
                    closed,
                } => self.commit_own_no_value(binding, report, closed),
                DeferredCommit::External { binding, report } => {
                    if self.phase == ManagerPhase::Running {
                        self.commit_foreign(binding, report)
                    } else {
                        self.driver.external_report_observed(&binding.work, report)
                    }
                }
                DeferredCommit::Closed(closed) => self.publish_closed_frame(closed),
                DeferredCommit::OwnRaw { work, report } => self
                    .driver
                    .cpu_slice_observed(work, report)
                    .and_then(|_| self.release_nnuec_report(work, report)),
                DeferredCommit::DetachedOwn { work, report } => self
                    .driver
                    .cpu_detached_slice_observed(work, report)
                    .and_then(|_| self.release_nnuec_report(work, report)),
                DeferredCommit::TokenDiscard { work, termination } => self
                    .driver
                    .cpu_token_discarded(work, termination)
                    .and_then(|_| self.release_nnuec_terminal(work, termination.job)),
                DeferredCommit::ReciprocalWait { work, report } => {
                    self.driver.cpu_reciprocal_wait(work, report)
                }
                DeferredCommit::ReciprocalDiscard { question } => {
                    self.driver.discard_cpu_reciprocal(*question)
                }
                DeferredCommit::PeerDiscard { question } => {
                    self.driver.discard_accepted_role_reciprocal(*question)
                }
            };
            if let Err(error) = result {
                // A failed hot commit cannot erase the original worker receipt.
                // Shutdown may still settle it without restoring consumption.
                self.deferred.push_front(event);
                if self.semantic_replay_can_wait(&error)? {
                    // Native terminals are drained/settled before this queue.
                    // Keep the exact accepted head and make no synthetic tick
                    // progress while waiting for its actual owner receipt.
                    return Ok(changed_before_event
                        || self.counters.consumed_role_outputs != consumed_before_event);
                }
                return Err(error);
            }
            // A collection failure closes normal admission, but once shutdown
            // has started every already-admitted terminal still has to drain.
            // The failure remains in `self.failures`, so using it as an
            // unconditional stop condition here would repeatedly strand the
            // remaining deferred receipts during draining.
            if self.phase == ManagerPhase::Running && self.has_collection_failure() {
                break;
            }
        }
        Ok(changed)
    }
    fn dispatch_roles(&mut self) -> Result<bool, AdaptiveManagerError> {
        if !self.native_ready || !self.can_admit_work() {
            return Ok(false);
        }
        while let Some(dispatch) = self.registry.take_dispatch() {
            self.role_mut(dispatch.ticket.id)?.dispatch_pending = true;
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if let Some(busy) = self.collection_v2_busy_ticket {
            if !self
                .roles
                .iter()
                .any(|role| role.dispatch_pending && role.ticket == Some(busy))
            {
                return Err(AdaptiveManagerError::Invalid(
                    "original busy collection ticket lost",
                ));
            }
        }
        let mut changed = false;
        for index in 0..self.roles.len() {
            if !self.roles[index].dispatch_pending {
                continue;
            }
            let Some(ticket) = self.roles[index].ticket else {
                continue;
            };
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            if self
                .collection_v2_busy_ticket
                .is_some_and(|busy| busy != ticket)
            {
                continue;
            }
            if self.registration.endpoints.execution_use == CpuExecutionUse::Collection
                && !self.roles[index].collection_registered
            {
                self.driver
                    .collection_before_dispatch(&ticket, &self.roles[index].input)?;
                self.roles[index].collection_registered = true;
            }
            if !self.roles[index].rovent_collection_registered {
                let result = match &mut self.collection_observer {
                    Some(observer) => observer.dispatch(&ticket, &self.roles[index].input),
                    None => Ok(()),
                };
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                let mut collection_errors = Vec::new();
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                let mut boundary_busy = false;
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                let result = result.and_then(|()| {
                    match &mut self.collection_observer_v2 {
                        Some(observer) => {
                            let verifier = self.native.native_acceptance_verifier();
                            let (admission, verifier) = self.collection_admission_v2.as_ref()
                                .zip(verifier.as_ref())
                                .ok_or_else(|| AdaptiveCollectionError::new(
                                    AdaptiveCollectionStage::Dispatch,
                                    "MissingV2Admission",
                                    "V2 observer lost its original admission or native verifier",
                                ))?;
                            let witness = match collection::RoventCollectionDispatchWitnessV2::try_new(
                                admission, verifier, &ticket,
                            ) {
                                Ok(witness) => witness,
                                Err(collection::RoventDispatchWitnessFailureV2::Ticket(error))
                                    if error.kind == collection::RoventTicketValidationKindV2::Busy => {
                                        boundary_busy = true;
                                        return Ok(());
                                    }
                                Err(collection::RoventDispatchWitnessFailureV2::Ticket(error)) => return Err(error.error),
                                Err(collection::RoventDispatchWitnessFailureV2::NativeIdentity(error)) => return Err(error),
                            };
                            if self.collection_v2_disabled {
                                let first = self.collection_v2_disabled_first_ticket.ok_or_else(||
                                    AdaptiveCollectionError::new(AdaptiveCollectionStage::Dispatch,
                                        "MissingDisabledBoundary", "disabled root lacks original first ticket"))?;
                                if ticket.id.epoch != first.id.epoch || ticket.authority != first.authority {
                                    return Err(AdaptiveCollectionError::new(AdaptiveCollectionStage::Dispatch,
                                        "DisabledBoundaryMismatch", "ticket differs from disabled root"));
                                }
                                if ticket.id.sequence < first.id.sequence {
                                    match observer.allow_disabled_prior_ticket(witness)? {
                                        collection::AdaptiveCollectionBoundaryV2::Ready => {}
                                        collection::AdaptiveCollectionBoundaryV2::Busy => boundary_busy = true,
                                    }
                                }
                                return Ok(());
                            }
                            if let Some(original) = self.collection_v2_first_failure.as_ref() {
                                match observer.disable_after_failure(witness, original) {
                                    Ok(collection::AdaptiveCollectionBoundaryV2::Ready) => {
                                        self.collection_v2_disabled = true;
                                        self.collection_v2_disabled_first_ticket = Some(ticket);
                                    }
                                    Ok(collection::AdaptiveCollectionBoundaryV2::Busy) => boundary_busy = true,
                                    Err(error) => {
                                        collection_errors.push(error);
                                        return Err(original.clone());
                                    }
                                }
                                return Ok(());
                            }
                            match observer.dispatch(witness, &self.roles[index].input) {
                                Ok(collection::AdaptiveCollectionDispatchV2::Captured) => {}
                                Ok(collection::AdaptiveCollectionDispatchV2::Busy) => boundary_busy = true,
                                Ok(collection::AdaptiveCollectionDispatchV2::DisabledBackpressure) => {
                                    self.collection_v2_disabled = true;
                                    self.collection_v2_disabled_first_ticket = Some(ticket);
                                }
                                Err(original) => {
                                    self.collection_v2_first_failure = Some(original.clone());
                                    collection_errors.push(original.clone());
                                    let witness = match collection::RoventCollectionDispatchWitnessV2::try_new(
                                        admission, verifier, &ticket,
                                    ) {
                                        Ok(witness) => witness,
                                        Err(collection::RoventDispatchWitnessFailureV2::Ticket(error))
                                            if error.kind == collection::RoventTicketValidationKindV2::Busy => {
                                                boundary_busy = true;
                                                return Ok(());
                                            }
                                        Err(collection::RoventDispatchWitnessFailureV2::Ticket(error)) => return Err(error.error),
                                        Err(collection::RoventDispatchWitnessFailureV2::NativeIdentity(error)) => return Err(error),
                                    };
                                    match observer.disable_after_failure(witness, &original) {
                                        Ok(collection::AdaptiveCollectionBoundaryV2::Ready) => {
                                            self.collection_v2_disabled = true;
                                            self.collection_v2_disabled_first_ticket = Some(ticket);
                                        }
                                        Ok(collection::AdaptiveCollectionBoundaryV2::Busy) => boundary_busy = true,
                                        Err(error) => {
                                            collection_errors.push(error);
                                            return Err(original);
                                        }
                                    }
                                }
                            }
                            Ok(())
                        }
                        None => Ok(()),
                    }
                });
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                {
                    if result.is_err() && self.collection_observer_v2.is_some() {
                        self.collection_v2_disable_failed = true;
                    }
                    for error in collection_errors {
                        self.record_failure(error.into());
                    }
                }
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                if boundary_busy {
                    // Retain the same unissued ticket and retry only inside
                    // its original admission/deadline; a staged reservation
                    // must resolve before another collection can disable.
                    self.collection_v2_busy_ticket = Some(ticket);
                    break;
                }
                if let Err(error) = result {
                    let cancel = self.native.cancel(self.roles[index].preparation);
                    if cancel.is_ok() {
                        self.roles[index].cancellation_requested = true;
                    }
                    let abandoned = self
                        .collection_observer
                        .as_mut()
                        .map(|observer| observer.abandoned(&ticket));
                    if let Some(Err(other)) = abandoned {
                        self.record_failure(other.into());
                    }
                    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                    if let Some(observer) = &mut self.collection_observer_v2 {
                        if let Err(other) = observer.abandoned(&ticket) {
                            self.record_failure(other.into());
                        }
                    }
                    if let Err(native) = cancel {
                        self.record_failure(error.into());
                        return Err(native.into());
                    }
                    return Err(error.into());
                }
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                if self.collection_v2_busy_ticket == Some(ticket) {
                    self.collection_v2_busy_ticket = None;
                }
                self.roles[index].rovent_collection_registered = true;
            }
            if self.roles[index].queue_admitted_at.is_some() {
                return Err(AdaptiveManagerError::Invalid("duplicate queue admission"));
            }
            // Preserve the original prepared ticket if a collector callback
            // crossed its future-admission window before actual enqueue.
            if !self.can_admit_work() {
                break;
            }
            match self
                .native
                .execute_with_admission_clock(self.roles[index].preparation, ticket)
            {
                Ok(admitted_at) => {
                    self.roles[index].queue_admitted_at = Some(admitted_at);
                    self.roles[index].dispatch_pending = false;
                    #[cfg(feature = "pals-adaptive-diagnostics")]
                    self.observe_pending_diagnostic(
                        ContinuationPendingSource::QueueAdmitted,
                        ticket.id,
                        Some(admitted_at),
                    );
                    changed = true;
                }
                Err(TicketServiceError::Capacity) => break,
                Err(error) => {
                    let abandoned = self
                        .collection_observer
                        .as_mut()
                        .map(|observer| observer.abandoned(&ticket));
                    if let Some(Err(other)) = abandoned {
                        self.record_failure(other.into());
                    }
                    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                    if let Some(observer) = &mut self.collection_observer_v2 {
                        if let Err(other) = observer.abandoned(&ticket) {
                            self.record_failure(other.into());
                        }
                    }
                    return Err(error.into());
                }
            }
        }
        Ok(changed)
    }
    fn deliver_roles(&mut self) -> Result<bool, AdaptiveManagerError> {
        if self.phase == ManagerPhase::Running && !self.driver.semantic_owner_available() {
            return Ok(false);
        }
        let mut changed = false;
        for _ in 0..self.limits.role_tickets_max {
            let now = self
                .clock
                .try_now()
                .map_err(|_| AdaptiveManagerError::Deadline)?;
            let Some(ready) = self.registry.take_ready(self.clock.domain(), now) else {
                break;
            };
            changed = true;
            let ticket = ready.ticket;
            let index = self
                .roles
                .iter()
                .position(|binding| binding.ticket == Some(ticket))
                .ok_or(AdaptiveManagerError::Invalid(
                    "delivered output has no preparation owner",
                ))?;
            let accepted = match ready.result {
                Ok(output) if ready.logically_eligible && self.phase == ManagerPhase::Running => {
                    validate_role_output(output.proposer_ranking_output(), &ticket, &ready.view)?;
                    if let Some(work) = &self.roles[index].work {
                        self.driver.validate_role(
                            work,
                            &ticket,
                            output.proposer_ranking_output(),
                        )?;
                    } else if self.roles[index].purpose == RolePurpose::RoventGoalLocal {
                        let original = self.roles[index].goal_local.as_ref().ok_or(
                            AdaptiveManagerError::Invalid("GoalLocal original input missing"),
                        )?;
                        self.driver.validate_rovent_goal_input(
                            original,
                            &ticket,
                            output.proposer_ranking_output(),
                        )?;
                    } else if self.roles[index].purpose == RolePurpose::RootRanking {
                        validate_root_ranking_view(&ready.view, self.driver.root_rules().as_ref())?;
                    } else {
                        self.driver.validate_validator(
                            &self.roles[index].input,
                            &ticket,
                            output.proposer_ranking_output(),
                        )?;
                    }
                    self.roles[index].delivered = Some(output);
                    true
                }
                Ok(output) => {
                    self.roles[index].delivered = Some(output);
                    false
                }
                Err(_) => false,
            };
            self.roles[index].ack_accepted = Some(accepted);
            if self.acknowledgements.len() >= self.limits.role_tickets_max {
                return Err(AdaptiveManagerError::Capacity("native ack queue"));
            }
            let permit = if accepted
                && matches!(
                    self.roles[index].purpose,
                    RolePurpose::Reciprocal
                        | RolePurpose::RoventRefinement
                        | RolePurpose::RoventFresh
                        | RolePurpose::RoventGoalLocal
                ) {
                Some(self.registry.issue_reciprocal_native_acceptance(
                    &ticket,
                    self.clock.domain(),
                    now,
                )?)
            } else if accepted
                && matches!(
                    self.roles[index].purpose,
                    RolePurpose::RootRanking | RolePurpose::Validator
                )
            {
                Some(
                    self.registry
                        .issue_native_acceptance(&ticket, self.clock.domain(), now)?,
                )
            } else {
                None
            };
            self.acknowledgements.push_back(NativeAck {
                preparation: self.roles[index].preparation,
                ticket,
                accepted,
                permit,
            });
        }
        self.flush_acks()?;
        Ok(changed)
    }
    fn flush_acks(&mut self) -> Result<(), AdaptiveManagerError> {
        // The original native close now owns disposal. Preserve unsent ACKs
        // and linear permits until its fence; do not enqueue into a retired root.
        if self.phase != ManagerPhase::Running {
            return Ok(());
        }
        while let Some(mut ack) = self.acknowledgements.pop_front() {
            let result = if let Some(permit) = ack.permit.take() {
                match self
                    .native
                    .accept_with_permit(ack.preparation, ack.ticket, permit)
                {
                    Ok(()) => Ok(()),
                    Err((error, permit)) => {
                        ack.permit = Some(permit);
                        Err(error)
                    }
                }
            } else if ack.accepted {
                self.native.accept(ack.preparation, ack.ticket)
            } else {
                self.native.discard(ack.preparation, ack.ticket)
            };
            match result {
                Ok(()) => {
                    let binding = self
                        .roles
                        .iter_mut()
                        .find(|binding| {
                            binding.preparation == ack.preparation
                                && binding.ticket == Some(ack.ticket)
                        })
                        .ok_or(AdaptiveManagerError::Invalid(
                            "original dispatched native ACK binding",
                        ))?;
                    binding.ack_dispatched = true;
                }
                Err(TicketServiceError::Capacity) => {
                    self.acknowledgements.push_front(ack);
                    break;
                }
                Err(error) => {
                    self.acknowledgements.push_front(ack);
                    return Err(error.into());
                }
            }
        }
        Ok(())
    }
    /// Independent progress request on the original live physical question.
    /// It uses the atomic control mailbox, does not Yield/cancel/reserve nodes,
    /// and is acknowledged only at the worker's coherent iteration boundary.
    /// Fixed/synchronous policies do not automatically issue this request.
    pub fn request_cpu_progress(&mut self, job: CpuJobId) -> Result<u64, AdaptiveManagerError> {
        self.check_running()?;
        let index = self
            .own
            .iter()
            .position(|slot| {
                slot.binding.as_ref().is_some_and(|binding| {
                    binding.envelope.id == job
                        && binding.grant.job == job
                        && slot
                            .worker
                            .as_ref()
                            .is_some_and(|worker| binding.grant.owner == worker.profile().owner)
                        && !binding.resume_pending
                        && binding.pending_slice.is_none()
                        && self.driver.cpu_progress_request_eligible(
                            &binding.work,
                            &binding.envelope,
                            &binding.grant,
                        )
                })
            })
            .ok_or(AdaptiveManagerError::Invalid(
                "PublishProgress original CPU question",
            ))?;
        let sequence = self.issue_control()?;
        self.own[index]
            .binding
            .as_ref()
            .expect("checked original CPU owner")
            .controls
            .request_publish_progress(sequence)
            .map_err(|_| AdaptiveManagerError::Invalid("PublishProgress control admission"))?;
        Ok(sequence)
    }

    fn own_progress_observed(
        &mut self,
        slot: usize,
        observation: CpuWorkerProgressObservation,
    ) -> Result<(), AdaptiveManagerError> {
        let progress = observation.progress();
        let eligible = self.phase == ManagerPhase::Running
            && self.own[slot].settled_execution != Some(progress.execution)
            && self.own[slot].binding.as_ref().is_some_and(|binding| {
                progress.job == binding.envelope.id
                    && progress.execution == binding.grant.execution
                    && progress.owner == binding.grant.owner
                    && progress.sequence > binding.progress_sequence
            });
        self.own_progress(slot, progress.clone())?;
        if !eligible {
            return Ok(());
        }
        let binding = self.own[slot]
            .binding
            .as_mut()
            .expect("validated original publication");
        // The restoration callback repeats an old coherent iteration. Keep
        // its original progress semantics, but do not make it new forecast
        // eligibility or a newly completed interval sample.
        binding.progress_published_at = observation
            .completed_progress_interval()
            .map(|_| observation.published_at());
        binding.owner_run_started_at = Some(observation.run_started_at());
        let class = binding.class;
        let resume = observation
            .resume_admission_and_restore()
            .filter(|_| binding.resume_cost_observed_execution != Some(progress.execution));
        if resume.is_some() {
            binding.resume_cost_observed_execution = Some(progress.execution);
        }
        if let Some(interval) = observation.completed_progress_interval() {
            self.observe(
                class,
                CostStage::CpuProgressInterval,
                interval,
                CostDisposition::Completed,
            )?;
        }
        if let Some(resume) = resume {
            self.observe(
                class,
                CostStage::CpuResumeAdmissionAndRestore,
                resume,
                CostDisposition::Completed,
            )?;
        }
        Ok(())
    }
    fn own_progress(
        &mut self,
        slot: usize,
        progress: CpuWorkerProgress,
    ) -> Result<(), AdaptiveManagerError> {
        if self.phase != ManagerPhase::Running {
            return Ok(());
        }
        // The owner's coalesced progress may trail its reliable final receipt.
        // Such a publication is never recharged or used after the final result.
        if self.own[slot].settled_execution == Some(progress.execution) {
            return Ok(());
        }
        let binding = self.own[slot]
            .binding
            .as_mut()
            .ok_or(AdaptiveManagerError::Invalid(
                "CPU progress has no live question",
            ))?;
        if progress.job != binding.envelope.id
            || progress.execution != binding.grant.execution
            || progress.owner != binding.grant.owner
            || progress.sequence <= binding.progress_sequence
        {
            return Err(AdaptiveManagerError::Invalid(
                "stale or wrong-owner CPU progress",
            ));
        }
        if !binding.feedback_forwarded {
            self.machine.publish_progress(
                binding.work.frame,
                StepExecution::Cpu {
                    job: progress.job,
                    execution: progress.execution,
                },
                progress.sequence,
            )?;
        }
        binding.progress_sequence = progress.sequence;
        binding.progress_since_slice = true;
        // Progress is not credit settlement and cannot publish incomplete alpha.
        if self.driver.semantic_owner_available() {
            self.driver.cpu_progress_for(&binding.work, &progress)?;
        }
        self.counters.cpu_progress_publications += 1;
        Ok(())
    }
    fn own_event(
        &mut self,
        slot: usize,
        event: CpuWorkerEvent,
    ) -> Result<(), AdaptiveManagerError> {
        self.own_event_with_timing(slot, event, None)
    }
    fn own_event_observed(
        &mut self,
        slot: usize,
        event: CpuWorkerEventObservation,
    ) -> Result<(), AdaptiveManagerError> {
        let (original, timing) = event.into_parts();
        self.own_event_with_timing(slot, original, timing)
    }
    fn own_event_with_timing(
        &mut self,
        slot: usize,
        event: CpuWorkerEvent,
        timing: Option<CpuWorkerSliceTiming>,
    ) -> Result<(), AdaptiveManagerError> {
        match event {
            CpuWorkerEvent::Slice(report) => {
                self.counters.cpu_slice_receipts += 1;
                let settled = match self.native_ledger_mut()?.settle_attested(&report) {
                    Ok(settled) => settled,
                    Err(error) => {
                        self.quarantine_frontier_slot(slot);
                        if let Some(diagnostic) = self.cpu_slice_diagnostic.get() {
                            diagnostic.mark_unavailable(
                                AdaptiveCpuSliceDiagnosticUnavailable::OriginalReceiptRejected,
                            );
                        }
                        self.quarantined_cpu_receipts
                            .push(CpuWorkerEvent::Slice(report));
                        self.phase = ManagerPhase::Quarantined;
                        return Err(error.into());
                    }
                };
                self.own[slot].settled_execution = Some(report.grant.execution);
                let Some(mut binding) = self.own[slot].binding.take() else {
                    if let Some(diagnostic) = self.cpu_slice_diagnostic.get() {
                        diagnostic.mark_unavailable(
                            AdaptiveCpuSliceDiagnosticUnavailable::OriginalReceiptRejected,
                        );
                    }
                    self.quarantined_cpu_receipts
                        .push(CpuWorkerEvent::Slice(report));
                    self.phase = ManagerPhase::Quarantined;
                    return Err(AdaptiveManagerError::Invalid(
                        "CPU slice has no submitted question",
                    ));
                };
                if report.grant != binding.grant {
                    if let Some(reservation) = binding.native_frontier.take() {
                        self.retain_frontier_unknown(reservation);
                    }
                    if let Some(diagnostic) = self.cpu_slice_diagnostic.get() {
                        diagnostic.mark_unavailable(
                            AdaptiveCpuSliceDiagnosticUnavailable::OriginalReceiptRejected,
                        );
                    }
                    self.own[slot].binding = Some(binding);
                    self.quarantined_cpu_receipts
                        .push(CpuWorkerEvent::Slice(report));
                    self.phase = ManagerPhase::Quarantined;
                    return Err(AdaptiveManagerError::Invalid(
                        "CPU slice grant echo mismatch",
                    ));
                }
                // Validate the original submitted question before preserving a
                // worker cause or taking the exact soft no-value path.
                if report.grant.job != binding.envelope.id
                    || binding.work.frame.authority != binding.envelope.id.authority
                    || binding.work.input.question_digest != binding.envelope.question
                    || binding.work.original_deadline != binding.envelope.hard_deadline
                {
                    self.own[slot].binding = Some(binding);
                    self.quarantined_cpu_receipts
                        .push(CpuWorkerEvent::Slice(report));
                    self.phase = ManagerPhase::Quarantined;
                    return Err(AdaptiveManagerError::Invalid(
                        "CPU receipt original question binding",
                    ));
                }
                let rejected_after_owner_termination = binding.queued_resume_termination.is_some()
                    && report.admission_rejected
                    && report.actual_work == Some(rz_search::cpu::CpuWork::default())
                    && !report.terminal
                    && !report.continuation_retained;
                if let Err(error) = &report.result {
                    if !rejected_after_owner_termination
                        && error.value_authority_stop().is_none()
                        && !self.failures.iter().any(|failure| {
                            matches!(failure, AdaptiveManagerError::Cpu(CpuWorkerError::Cpu(_)))
                        })
                    {
                        // Preserve the first hard cause before any later
                        // generic recovery failure. The raw typed error stays
                        // in the original binding/deferred receipt as well.
                        self.record_failure(AdaptiveManagerError::Cpu(CpuWorkerError::Cpu(
                            error.clone(),
                        )));
                    }
                }
                // The same accepted native-ledger receipt remains available
                // for collection after actual Store consumption. It is never
                // recreated from reported counters or a global snapshot.
                binding.semantic_settlement = Some(settled.clone());
                if let Some(reservation) = binding.native_frontier.take() {
                    self.retain_frontier_settlement(reservation, report.clone(), settled);
                    if let Err(error) = self.flush_frontier_native() {
                        // Keep original execution/receipt custody even when
                        // common accounting rejects it. No zero-work fallback.
                        binding.pending_slice = Some(report);
                        self.own[slot].binding = Some(binding);
                        self.phase = ManagerPhase::Quarantined;
                        return Err(error);
                    }
                }
                // Copy only after the original ledger and binding/grant
                // checks. This is prior to timing/semantic/yield processing;
                // those paths may still fail with their original errors.
                if let Some(diagnostic) = self.cpu_slice_diagnostic.get() {
                    let accounting = self
                        .native_ledger()?
                        .global_usage()
                        .ok()
                        .zip(self.native_ledger()?.usage(report.grant.job).ok())
                        .map(
                            |(
                                (global_spent, global_reserved),
                                (job_spent, job_reserved, job_quarantined),
                            )| {
                                AdaptiveCpuSliceAccountingObservation {
                                    global_spent,
                                    global_reserved,
                                    job_spent,
                                    job_reserved,
                                    job_quarantined,
                                }
                            },
                        );
                    diagnostic.record_settled(
                        AdaptiveCpuSliceScopeObservation {
                            authority: self.registration.authority,
                            frame: binding.work.frame,
                            stage: binding.work.input.stage,
                            question: binding.work.input.question_digest,
                            view: binding.work.input.view_seal,
                            public_context: binding.work.input.public_context,
                            job: binding.envelope,
                            requested_yield_sequence: binding
                                .requested_yield
                                .map(|(sequence, _)| sequence),
                            prior_yield_ack: binding.last_yield_ack,
                            binding_progress_sequence: binding.progress_sequence,
                            progress_since_slice: binding.progress_since_slice,
                            feedback_already_forwarded: binding.feedback_forwarded,
                        },
                        &report,
                        accounting,
                        timing,
                    );
                }
                if binding.queued_resume_termination.is_some() {
                    // The real terminal already discarded the OLD paused
                    // token; this newer queued grant has only now settled.
                    // Preserve both original reports until their joint close.
                    binding.pending_slice = Some(report);
                    self.own[slot].binding = Some(binding);
                    return self
                        .retire_queued_resume_after_owner_termination(slot)
                        .map(|_| ());
                }
                // Original worker boundaries only. Queue delivery, paused
                // residence and manager processing are outside these scopes.
                let timing_result = (|| {
                    if !report.admission_rejected
                        && report.result.is_ok()
                        && report.actual_work.is_some()
                    {
                        if let Some(timing) = timing {
                            self.observe(
                                binding.class,
                                CostStage::FeedbackReturn,
                                timing.return_processing(),
                                CostDisposition::Completed,
                            )?;
                            if let Some(resume) =
                                timing.resume_admission_and_restore().filter(|_| {
                                    binding.resume_cost_observed_execution
                                        != Some(report.grant.execution)
                                })
                            {
                                self.observe(
                                    binding.class,
                                    CostStage::CpuResumeAdmissionAndRestore,
                                    resume,
                                    CostDisposition::Completed,
                                )?;
                                binding.resume_cost_observed_execution =
                                    Some(report.grant.execution);
                            }
                        }
                    }
                    Ok::<(), AdaptiveManagerError>(())
                })();
                if let Err(error) = timing_result {
                    if report.continuation_retained {
                        binding.pending_slice = Some(report);
                        self.own[slot].binding = Some(binding);
                    } else {
                        self.defer(DeferredCommit::OwnRaw {
                            work: binding.work,
                            report,
                        })?;
                    }
                    return Err(error);
                }
                if report.terminal
                    && report.result.as_ref().is_ok_and(|actual| {
                        matches!(
                            actual.report.completion,
                            CpuCompletion::DepthLimit
                                | CpuCompletion::NodeLimit
                                | CpuCompletion::Terminal(_)
                        )
                    })
                {
                    self.counters.cpu_natural_completions += 1;
                }
                let disposition = if report.result.is_ok() && report.terminal {
                    CostDisposition::Completed
                } else {
                    CostDisposition::Censored(CensorReason::Interrupted)
                };
                let observed = self.observe_observation(
                    binding.class,
                    CostStage::CpuSearch,
                    CostObservation {
                        elapsed: report
                            .result
                            .as_ref()
                            .map_or(ObservedCost::Unknown, |actual| {
                                ObservedCost::Known(actual.report.elapsed)
                            }),
                        disposition,
                    },
                );
                if let Err(error) = observed {
                    if report.continuation_retained {
                        binding.pending_slice = Some(report);
                        self.own[slot].binding = Some(binding);
                    } else {
                        self.defer(DeferredCommit::OwnRaw {
                            work: binding.work,
                            report,
                        })?;
                    }
                    return Err(error);
                }
                if report.admission_rejected {
                    // A rejected command remains the original admission error;
                    // it is not relabelled as an entered no-value termination.
                    // Credit has been settled by a definite no-work report. Do
                    // not manufacture a CPU result or move an existing stack.
                    if report.continuation_retained {
                        binding.resume_pending = false;
                        binding.pending_slice = Some(report);
                        self.own[slot].binding = Some(binding);
                    } else {
                        self.defer(DeferredCommit::OwnRaw {
                            work: binding.work,
                            report,
                        })?;
                    }
                    return Err(AdaptiveManagerError::Invalid(
                        "private CPU admission was rejected",
                    ));
                }
                if report
                    .result
                    .as_ref()
                    .err()
                    .and_then(|error| error.value_authority_stop())
                    .is_some()
                {
                    // Keep the original worker terminal and its typed cause.
                    // The Search consumer validates every terminal/known-work
                    // condition; no JobTerminated packet or score is minted.
                    self.defer(DeferredCommit::OwnNoValue {
                        binding,
                        report,
                        closed: None,
                    })?;
                    self.flush_deferred()?;
                    return Ok(());
                }
                if report
                    .result
                    .as_ref()
                    .is_ok_and(|actual| actual.report.completion == CpuCompletion::AwaitReciprocal)
                {
                    return self.retain_reciprocal_wait(slot, binding, report);
                }
                if report.continuation_retained && !report.terminal {
                    if binding.feedback_forwarded {
                        // First feedback already advanced the logical frame.
                        // Keep physical custody on the original question; its
                        // future reports must never advance that new frame.
                        let ack = report.controls.and_then(|control| {
                            (control.yield_ack > binding.last_yield_ack)
                                .then_some(control.yield_ack)
                        });
                        let mut timing_error = None;
                        if let Some(ack) = ack {
                            binding.last_yield_ack = ack;
                            self.counters.cpu_yield_acknowledgements += 1;
                            if let Some((_, requested_at)) = binding
                                .requested_yield
                                .filter(|(request, _)| ack >= *request)
                            {
                                timing_error = self
                                    .observe_yield_ack_latency(binding.class, requested_at, timing)
                                    .err();
                            }
                        }
                        self.counters.cpu_yields += 1;
                        binding.pending_slice = Some(report);
                        binding.resume_pending = self.phase == ManagerPhase::Running;
                        binding.progress_since_slice = false;
                        binding.progress_published_at = None;
                        binding.requested_yield = None;
                        self.own[slot].binding = Some(binding);
                        self.flush_deferred()?;
                        return timing_error.map_or(Ok(()), Err);
                    }
                    if self.registration.policy == CoordinatorPolicy::Adaptive
                        && self.phase == ManagerPhase::Running
                        && self.driver.semantic_owner_available()
                        && self
                            .driver
                            .cpu_partial_continuation_eligible(&binding.work, &report)
                    {
                        // Resolve while the exact original execution is still
                        // Waiting. commit_cpu consumes the real receipt once;
                        // do not also call cpu_slice_observed for this report.
                        if let Err(error) = self.commit_own(&binding, &report) {
                            binding.pending_slice = Some(report);
                            self.own[slot].binding = Some(binding);
                            return Err(error);
                        }
                        binding.feedback_forwarded = true;
                        let mut yield_timing_error = None;
                        if let Some(ack) = report.controls.and_then(|control| {
                            (control.yield_ack > binding.last_yield_ack)
                                .then_some(control.yield_ack)
                        }) {
                            binding.last_yield_ack = ack;
                            self.counters.cpu_yield_acknowledgements += 1;
                            if let Some((request, requested_at)) = binding
                                .requested_yield
                                .filter(|(request, _)| ack >= *request)
                            {
                                let _ = request;
                                if let Err(error) = self.observe_yield_ack_latency(
                                    binding.class,
                                    requested_at,
                                    timing,
                                ) {
                                    yield_timing_error = Some(error);
                                }
                            }
                        }
                        binding.resume_pending = true;
                        binding.progress_since_slice = false;
                        binding.progress_published_at = None;
                        binding.requested_yield = None;
                        self.counters.cpu_yields += 1;
                        self.own[slot].binding = Some(binding);
                        return match yield_timing_error {
                            Some(error) => Err(error),
                            None => Ok(()),
                        };
                    }
                    if self.driver.semantic_owner_available() {
                        if let Err(error) = self.driver.cpu_slice_observed(&binding.work, &report) {
                            binding.pending_slice = Some(report);
                            self.own[slot].binding = Some(binding);
                            return Err(error);
                        }
                    } else {
                        binding.pending_slice = Some(report.clone());
                    }
                    if self.phase != ManagerPhase::Running {
                        // Revoked authority cannot accept a value/progress or
                        // resume. Retain the same owner until its actual token
                        // discard receipt, which is settled separately.
                        binding.resume_pending = false;
                        self.own[slot].binding = Some(binding);
                        return Ok(());
                    }
                    let Some(work) = report.actual_work else {
                        self.own[slot].binding = Some(binding);
                        return Err(AdaptiveManagerError::PhysicalUnknown);
                    };
                    let ack = report.controls.and_then(|controls| {
                        (controls.yield_ack > binding.last_yield_ack).then_some(controls.yield_ack)
                    });
                    if let Some(ack) = ack {
                        binding.last_yield_ack = ack;
                        self.counters.cpu_yield_acknowledgements += 1;
                    }
                    if let Some((requested, requested_at)) = binding.requested_yield {
                        if ack.is_some_and(|ack| ack >= requested) {
                            let observed =
                                self.observe_yield_ack_latency(binding.class, requested_at, timing);
                            if let Err(error) = observed {
                                binding.pending_slice = Some(report);
                                self.own[slot].binding = Some(binding);
                                return Err(error);
                            }
                            binding.requested_yield = None;
                        }
                    }
                    let advancing = match self.machine.observe_yield(
                        binding.work.frame,
                        StepExecution::Cpu {
                            job: binding.envelope.id,
                            execution: report.grant.execution,
                        },
                        ack,
                        work.nodes,
                        binding.progress_since_slice,
                    ) {
                        Ok(advancing) => advancing,
                        Err(error) => {
                            binding.pending_slice = Some(report);
                            self.own[slot].binding = Some(binding);
                            return Err(error.into());
                        }
                    };
                    binding.resume_pending = advancing;
                    binding.progress_since_slice = false;
                    self.counters.cpu_yields += 1;
                    if !advancing {
                        let controls = Arc::clone(&binding.controls);
                        self.own[slot].binding = Some(binding);
                        let sequence = self.issue_control()?;
                        controls.request_cancel(sequence).map_err(|_| {
                            AdaptiveManagerError::Invalid("empty Yield loop cancel")
                        })?;
                        return Ok(());
                    }
                    self.own[slot].binding = Some(binding);
                } else if binding.feedback_forwarded {
                    self.defer(DeferredCommit::DetachedOwn {
                        work: binding.work,
                        report,
                    })?;
                    self.flush_deferred()?;
                } else if self.phase == ManagerPhase::Running {
                    self.defer(DeferredCommit::Own { binding, report })?;
                    self.flush_deferred()?;
                } else {
                    self.defer(DeferredCommit::OwnRaw {
                        work: binding.work,
                        report,
                    })?;
                    self.flush_deferred()?;
                }
            }
            CpuWorkerEvent::JobTerminated(termination) => {
                if let Err(error) = self
                    .native_ledger_mut()?
                    .accept_job_termination(&termination)
                {
                    self.quarantine_frontier_slot(slot);
                    self.quarantined_cpu_receipts
                        .push(CpuWorkerEvent::JobTerminated(termination));
                    self.phase = ManagerPhase::Quarantined;
                    return Err(error.into());
                }
                let Some(mut binding) = self.own[slot].binding.take() else {
                    self.quarantined_cpu_receipts
                        .push(CpuWorkerEvent::JobTerminated(termination));
                    self.phase = ManagerPhase::Quarantined;
                    return Err(AdaptiveManagerError::Invalid(
                        "CPU termination has no owner question",
                    ));
                };
                if termination.job != binding.envelope.id
                    || termination.owner != binding.grant.owner
                {
                    self.own[slot].binding = Some(binding);
                    self.quarantined_cpu_receipts
                        .push(CpuWorkerEvent::JobTerminated(termination));
                    self.phase = ManagerPhase::Quarantined;
                    return Err(AdaptiveManagerError::Invalid(
                        "CPU token termination owner mismatch",
                    ));
                }
                if termination.last_execution != binding.grant.execution {
                    let usage = self.native_ledger()?.usage(binding.envelope.id)?;
                    if binding.queued_resume_termination.is_some()
                        || termination.last_execution.epoch != binding.grant.execution.epoch
                        || termination.last_execution.sequence >= binding.grant.execution.sequence
                        || usage.1 != binding.grant.node_credit
                        || usage.2
                    {
                        self.own[slot].binding = Some(binding);
                        self.quarantined_cpu_receipts
                            .push(CpuWorkerEvent::JobTerminated(termination));
                        self.phase = ManagerPhase::Quarantined;
                        return Err(AdaptiveManagerError::Invalid(
                            "CPU termination does not precede its outstanding Resume",
                        ));
                    }
                    // The ledger accepted the old stack's actual terminal but
                    // intentionally retained the newer reservation. Its owner
                    // must still return that queued command's original report.
                    binding.resume_pending = false;
                    binding.queued_resume_termination = Some(termination);
                    self.own[slot].binding = Some(binding);
                    return Ok(());
                }
                if let Some(report) = binding.pending_slice {
                    self.defer(DeferredCommit::OwnRaw {
                        work: binding.work.clone(),
                        report,
                    })?;
                }
                if let Some(custody) = binding.reciprocal {
                    if !custody.observed {
                        self.defer(DeferredCommit::ReciprocalWait {
                            work: binding.work.clone(),
                            report: custody.report,
                        })?;
                    }
                    if let Some(question) = custody.question {
                        self.defer(DeferredCommit::ReciprocalDiscard {
                            question: question.id,
                        })?;
                    }
                }
                self.defer(DeferredCommit::TokenDiscard {
                    work: binding.work.clone(),
                    termination: termination.clone(),
                })?;
                if binding.feedback_forwarded {
                    // Revocation belongs to the old paused CPU custody. It
                    // cannot close the already-advanced P/C logical frame.
                    return Ok(());
                }
                let closed = self.machine.retire_suspended_cpu(
                    binding.work.frame,
                    &termination,
                    receipt_digest(b"cpu-token-discard", termination.last_execution),
                    "CPU owner terminated",
                )?;
                self.defer(DeferredCommit::Closed(closed))?;
            }
        }
        Ok(())
    }
    fn retire_queued_resume_after_owner_termination(
        &mut self,
        slot: usize,
    ) -> Result<bool, AdaptiveManagerError> {
        if !self.driver.semantic_owner_available() {
            return Ok(false);
        }
        let binding = self.own[slot]
            .binding
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "terminated queued Resume binding",
            ))?;
        let termination =
            binding
                .queued_resume_termination
                .as_ref()
                .ok_or(AdaptiveManagerError::Invalid(
                    "original paused owner terminal",
                ))?;
        let report = binding
            .pending_slice
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "original queued Resume report",
            ))?;
        let settled = binding
            .semantic_settlement
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "queued Resume ledger settlement",
            ))?;
        if binding.native_frontier.is_some() || self.pending_frontier_for(binding.envelope.id) {
            return Ok(false);
        }
        if !report.admission_rejected
            || report.actual_work != Some(rz_search::cpu::CpuWork::default())
            || report.result.is_ok()
            || report.terminal
            || report.continuation_retained
            || settled.grant() != binding.grant
            || settled.ledger_identity() != &self.native_ledger()?.identity()
            || settled.terminated_owner_receipt() != Some(termination)
            || binding.reciprocal.is_some()
        {
            return Err(AdaptiveManagerError::Invalid(
                "terminated owner requires its settled no-work Resume rejection",
            ));
        }
        // Reserve every deferred seat before changing Search or releasing the
        // binding. Capacity failure leaves the two original reports in custody.
        let slots = 2 + usize::from(!binding.feedback_forwarded);
        if self
            .deferred
            .len()
            .checked_add(slots)
            .is_none_or(|sum| sum > 128)
        {
            return Ok(false);
        }
        self.deferred
            .try_reserve(slots)
            .map_err(|_| AdaptiveManagerError::Capacity("terminated Resume handoff"))?;
        let closed = if binding.feedback_forwarded {
            None
        } else {
            Some(self.machine.retire_rejected_cpu_resume(
                binding.work.frame,
                termination,
                settled,
                receipt_digest(b"cpu-queued-resume-discard", report.grant.execution),
                "CPU owner terminated before queued Resume",
            )?)
        };
        let mut binding = self.own[slot]
            .binding
            .take()
            .expect("validated original binding");
        let report = binding
            .pending_slice
            .take()
            .expect("retained original report");
        let termination = binding
            .queued_resume_termination
            .take()
            .expect("retained original terminal");
        self.deferred.push_back(DeferredCommit::OwnRaw {
            work: binding.work.clone(),
            report,
        });
        self.deferred.push_back(DeferredCommit::TokenDiscard {
            work: binding.work,
            termination,
        });
        if let Some(closed) = closed {
            self.deferred.push_back(DeferredCommit::Closed(closed));
        }
        Ok(true)
    }
    fn observe_yield_ack_latency(
        &mut self,
        class: CostClass,
        requested_at: Instant,
        timing: Option<CpuWorkerSliceTiming>,
    ) -> Result<(), AdaptiveManagerError> {
        self.observe_observation(
            class,
            CostStage::YieldLatency,
            CostObservation {
                elapsed: timing
                    .and_then(|observed| {
                        observed
                            .report_ready_at()
                            .checked_duration_since(requested_at)
                    })
                    .map_or(ObservedCost::Unknown, ObservedCost::Known),
                disposition: CostDisposition::Completed,
            },
        )
    }

    fn foreign_progress_observed(
        &mut self,
        sequence: u64,
        observation: ExternalWorkerProgressObservation,
    ) -> Result<(), AdaptiveManagerError> {
        let Some(binding) = self.foreign.as_mut() else {
            // A coalesced publication can remain after its terminal packet.
            // No current execution is authorized to consume it.
            return Ok(());
        };
        let progress = observation.progress();
        if binding.job.scope() != progress.scope {
            return Err(AdaptiveManagerError::Invalid("old-go external progress"));
        }
        if sequence <= binding.progress_sequence {
            return Ok(());
        }
        binding.progress_sequence = sequence;
        if self.phase != ManagerPhase::Running {
            return Ok(());
        }
        self.machine.publish_progress(
            binding.work.frame,
            StepExecution::ExternalCpu {
                job: progress.scope.job,
                execution: progress.scope.execution,
            },
            sequence,
        )?;
        // Store the producer's whole raw tuple and original timestamps. It is
        // not a native completed iteration or strict CPU-credit observation.
        let class = binding.class;
        let interval = observation.completed_coherent_tuple_interval();
        binding.observed_progress = Some(observation);
        if let Some(elapsed) = interval {
            self.observe(
                class,
                CostStage::ExternalCoherentTupleInterval,
                elapsed,
                CostDisposition::Completed,
            )?;
        }
        Ok(())
    }
    fn foreign_event(
        &mut self,
        completion: ExternalWorkerCompletion,
    ) -> Result<(), AdaptiveManagerError> {
        let report = completion.report();
        let Some(binding) = self.foreign.take() else {
            self.quarantined_foreign_receipts
                .push(completion.into_report());
            self.phase = ManagerPhase::Quarantined;
            return Err(AdaptiveManagerError::Invalid(
                "external report has no issued go",
            ));
        };
        let original_scope_matches = report.job == binding.job.envelope
            && report.grant == binding.job.grant
            && report.scope == binding.job.scope()
            && !report.exact_resume_supported;
        // Observe the owner terminal packet before a Store/archive commit can
        // defer or reject it. The collector retains failed/stale packets as
        // raw evidence and independently decides target eligibility.
        let eligibility = AdaptiveTeacherTargetEligibility {
            job: binding.job.envelope,
            grant: binding.job.grant,
            eligible: original_scope_matches
                && self.phase == ManagerPhase::Running
                && !report.controls.is_cancelled()
                && self.check_running().is_ok()
                && self.machine.frame(binding.work.frame).is_some_and(|frame| {
                    frame.phase
                        == FramePhase::Waiting(StepExecution::ExternalCpu {
                            job: binding.job.envelope.id,
                            execution: binding.job.grant.execution,
                        })
                }),
        };
        let teacher_observed = self.driver.external_teacher_completed(
            &binding.teacher_dispatch,
            &completion,
            eligibility,
        );
        if !original_scope_matches {
            self.foreign = Some(binding);
            self.quarantined_foreign_receipts
                .push(completion.into_report());
            self.phase = ManagerPhase::Quarantined;
            if let Err(error) = teacher_observed {
                self.record_failure(error);
            }
            return Err(AdaptiveManagerError::Invalid(
                "external old-go or unsupported exact resume",
            ));
        }
        let timing = completion.timing();
        let report = completion.into_report();
        self.counters.foreign_reports += 1;
        let reported = report
            .result
            .as_ref()
            .ok()
            .and_then(|result| result.report.work.nodes)
            .or_else(|| {
                report
                    .attempt
                    .as_ref()
                    .and_then(|attempt| attempt.work.nodes)
            });
        let mut observation_error = None;
        if let Some(nodes) = reported {
            self.counters.foreign_reported_nodes = self
                .counters
                .foreign_reported_nodes
                .checked_add(nodes)
                .unwrap_or_else(|| {
                    observation_error =
                        Some(AdaptiveManagerError::Capacity("external reported-node sum"));
                    self.counters.foreign_reported_nodes
                });
        } else {
            self.counters.foreign_node_count_unknown += 1;
        }
        let completed = report
            .result
            .as_ref()
            .is_ok_and(|result| result.readiness_barrier_completed);
        let timing_observed = (|| {
            if let Some(timing) = timing {
                self.observe(
                    binding.class,
                    CostStage::QueueWait,
                    timing.queue_wait(),
                    CostDisposition::Completed,
                )?;
                if completed && timing.go_issued_at().is_some() {
                    self.observe(
                        binding.class,
                        CostStage::FeedbackReturn,
                        timing.return_processing(),
                        CostDisposition::Completed,
                    )?;
                    if let Some(elapsed) = timing.stop_drain() {
                        self.observe(
                            binding.class,
                            CostStage::ExternalStopDrain,
                            elapsed,
                            CostDisposition::Completed,
                        )?;
                    }
                    if let Some((sequence, requested_at)) = binding.requested_yield {
                        if report.controls.yield_ack >= sequence {
                            self.counters.foreign_yield_acknowledgements += 1;
                            self.observe(
                                binding.class,
                                CostStage::YieldLatency,
                                timing
                                    .report_ready_at()
                                    .checked_duration_since(requested_at)
                                    .ok_or(AdaptiveManagerError::Invalid(
                                        "external Yield owner clock",
                                    ))?,
                                CostDisposition::Completed,
                            )?;
                        }
                    }
                }
            }
            Ok::<_, AdaptiveManagerError>(())
        })();
        let observed = self.observe(
            binding.class,
            CostStage::CpuSearch,
            binding.started_at.elapsed(),
            if completed {
                CostDisposition::Completed
            } else {
                CostDisposition::Censored(CensorReason::Failed)
            },
        );
        // No foreign node report is ever sent through the native credit ledger.
        self.defer(DeferredCommit::External { binding, report })?;
        teacher_observed?;
        timing_observed?;
        if let Some(error) = observation_error {
            return Err(error);
        }
        observed?;
        self.flush_deferred()?;
        Ok(())
    }
    fn validate_own_nnuec_custody(
        &self,
        binding: &OwnBinding<Q>,
    ) -> Result<(), AdaptiveManagerError> {
        if let Some(original) = &binding.nnuec {
            if original.seal.frame() != binding.work.frame
                || original.seal.question_digest() != binding.envelope.question
                || !Arc::ptr_eq(original.seal.view(), &binding.work.input.view)
                || original.receipt.binding_digest() != binding.envelope.evaluation_namespace
                || !original.receipt.matches_bound(&original.bound)
                || !original.receipt.installed_profile().is_some_and(|profile| {
                    profile.owner == binding.grant.owner
                        && profile.search_conditions == binding.envelope.search_conditions
                })
            {
                return Err(AdaptiveManagerError::Invalid(
                    "CPU result lost original installed NNUEC custody",
                ));
            }
        }
        Ok(())
    }
    fn commit_own_no_value(
        &mut self,
        binding: &OwnBinding<Q>,
        report: &CpuSliceReport,
        closed: &mut Option<ClosedFrame>,
    ) -> Result<(), AdaptiveManagerError> {
        self.validate_own_nnuec_custody(binding)?;
        let receipt_seal = self.driver.cpu_no_value_terminated(&binding.work, report)?;
        self.release_nnuec_report(&binding.work, report)?;
        if binding.feedback_forwarded {
            // Only old CPU custody ended. Its already-advanced logical frame
            // and any new P/C/Fresh responsibility remain untouched.
            return Ok(());
        }
        if closed.is_none() {
            *closed = Some(self.machine.retire_cpu_no_value(
                &binding.work,
                report,
                receipt_seal,
            )?);
        }
        let closed = closed
            .as_ref()
            .expect("actual no-value retirement retained");
        self.release_accepted_role_waiters(closed.id)?;
        // The existing publisher records an Unknown closure only. In particular
        // there is no utility_cpu_committed/after_cpu/goal completion here.
        self.publish_closed_frame(closed)
    }
    fn commit_own(
        &mut self,
        binding: &OwnBinding<Q>,
        report: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        let utility_execution = StepExecution::Cpu {
            job: binding.envelope.id,
            execution: report.grant.execution,
        };
        self.utility_begin_commit(&binding.work, utility_execution)?;
        self.validate_own_nnuec_custody(binding)?;
        if self
            .accepted_roles
            .iter()
            .any(|held| held.child == Some(binding.work.frame))
        {
            let consumed = self.driver.consume_reciprocal_cpu(&binding.work, report);
            if consumed.is_ok()
                || self
                    .driver
                    .cpu_semantic_publication_recorded(&binding.work, report)
            {
                self.utility_cpu_committed(&binding.work, utility_execution)?;
                self.observe_cpu_semantic_use(binding, report);
            }
            let digest = consumed?;
            self.release_nnuec_report(&binding.work, report)?;
            self.machine.resolve_reciprocal_cpu_response(
                StepEvidence {
                    frame: binding.work.frame,
                    stage: binding.work.input.stage,
                    execution: StepExecution::Cpu {
                        job: binding.envelope.id,
                        execution: report.grant.execution,
                    },
                    submitted_question: binding.work.input.question_digest,
                    submitted_view: binding.work.input.view_seal,
                    completed_evidence: digest,
                },
                report,
                self.clock.domain(),
                self.check_running()?,
            )?;
            self.release_accepted_role_waiters(binding.work.frame)?;
            return Ok(());
        }
        let committed = self.driver.commit_cpu(&binding.work, report);
        if committed.is_ok()
            || self
                .driver
                .cpu_semantic_publication_recorded(&binding.work, report)
        {
            self.utility_cpu_committed(&binding.work, utility_execution)?;
            self.observe_cpu_semantic_use(binding, report);
        }
        let checked = committed?;
        self.release_nnuec_report(&binding.work, report)?;
        self.resolve(
            binding.work.clone(),
            StepExecution::Cpu {
                job: binding.envelope.id,
                execution: report.grant.execution,
            },
            checked,
        )
    }
    fn commit_foreign(
        &mut self,
        binding: &ForeignBinding<Q>,
        report: &ExternalWorkerReport,
    ) -> Result<(), AdaptiveManagerError> {
        let checked = self.driver.commit_external(&binding.work, report)?;
        self.resolve(
            binding.work.clone(),
            StepExecution::ExternalCpu {
                job: report.job.id,
                execution: report.grant.execution,
            },
            checked,
        )
    }
    fn resolve(
        &mut self,
        work: CoordinatorWork<ReviewView, Q>,
        execution: StepExecution,
        checked: CheckedStep<Q>,
    ) -> Result<(), AdaptiveManagerError> {
        let rules_pair_scope = matches!(
            &checked.transition,
            FrameTransition::Complete {
                scope: FrameCompletionScope::CompletedRulesPair
            }
        );
        match (
            &checked.rules_pair,
            rules_pair_scope,
            checked.fresh_pair.is_some(),
        ) {
            (Some(proof), true, false) => self.driver.validate_rules_pair(&work, proof)?,
            (None, false, _) => {}
            _ => {
                return Err(AdaptiveManagerError::Invalid(
                    "Rules pair scope/provenance mismatch",
                ));
            }
        }
        if let Some(pair) = checked.fresh_pair {
            self.machine.seal_fresh_pair(work.frame, pair)?;
        }
        let now = self.check_running()?;
        if let Some(queued) = checked.queued_recheck {
            self.machine
                .register_checked_recheck_queue(queued, self.clock.domain(), now)?;
        }
        let evidence = StepEvidence {
            frame: work.frame,
            stage: work.input.stage,
            execution,
            submitted_question: work.input.question_digest,
            submitted_view: work.input.view_seal,
            completed_evidence: checked.evidence,
        };
        #[cfg(feature = "pals-adaptive-diagnostics")]
        let diagnostic_transition = match &checked.transition {
            FrameTransition::Continue(next) => ContinuationTransitionKind::Continue {
                next_stage: next.stage,
            },
            FrameTransition::Complete { scope } => {
                ContinuationTransitionKind::Complete { scope: *scope }
            }
            FrameTransition::Unknown { reason } => ContinuationTransitionKind::Unknown { reason },
        };
        #[cfg(feature = "pals-adaptive-diagnostics")]
        let checked_rules_pair = checked.rules_pair.is_some();
        let resolved = match checked.rules_pair {
            Some(proof) => {
                self.machine
                    .resolve_checked_rules_pair(evidence, &proof, self.clock.domain(), now)
            }
            None => self
                .machine
                .resolve(evidence, checked.transition, self.clock.domain(), now),
        };
        #[cfg(feature = "pals-adaptive-diagnostics")]
        if let Some(diagnostic) = &self.continuation_diagnostic {
            diagnostic.resolution(ContinuationResolutionObservation {
                authority: self.registration.authority,
                evidence,
                transition: diagnostic_transition,
                checked_at: now,
                clock: self.clock.domain(),
                source: ContinuationResolutionSource::MachineTransition {
                    checked_rules_pair,
                    accepted: resolved.is_ok(),
                },
                closed_frame: resolved
                    .as_ref()
                    .ok()
                    .and_then(|closed| closed.as_ref().map(|frame| frame.id)),
                closed_scope: resolved
                    .as_ref()
                    .ok()
                    .and_then(|closed| closed.as_ref().and_then(|frame| frame.completion_scope)),
                closed_unknown: resolved
                    .as_ref()
                    .ok()
                    .and_then(|closed| closed.as_ref().and_then(|frame| frame.unknown)),
            });
        }
        let closed = resolved?;
        if let Some(closed) = closed {
            let published = self.publish_closed_frame(&closed);
            #[cfg(feature = "pals-adaptive-diagnostics")]
            if let Some(diagnostic) = &self.continuation_diagnostic {
                diagnostic.resolution(ContinuationResolutionObservation {
                    authority: self.registration.authority,
                    evidence,
                    transition: diagnostic_transition,
                    checked_at: now,
                    clock: self.clock.domain(),
                    source: ContinuationResolutionSource::ClosedFramePublication {
                        published: published.is_ok(),
                    },
                    closed_frame: Some(closed.id),
                    closed_scope: closed.completion_scope,
                    closed_unknown: closed.unknown,
                });
            }
            published?;
        } else {
            self.rovent_lane_after_resolve(&work)?;
            self.utility_register_descendant(work.frame)?;
        }
        Ok(())
    }
    fn schedule(&mut self) -> Result<bool, AdaptiveManagerError> {
        if !self.role_call_admission_available() && self.peer_policy.is_none() {
            return Ok(false);
        }
        if !self.driver.prepare_semantic_commit()? {
            return Ok(false);
        }
        let peers_changed = self.drive_accepted_role_parents()?;
        if self.maybe_close_utility_route()? {
            return Ok(true);
        }
        if !self.can_admit_work() {
            return Ok(peers_changed);
        }
        let reciprocal_changed = self.admit_waiting_reciprocal_questions()?;
        let now = self.check_running()?;
        let now_duration = Duration::from_nanos(now.0);
        // Small ID list only; no record payload, tensor or statistics snapshot
        // is cloned/sorted on the manager tick.
        let new_ready: Vec<_> = self
            .machine
            .frames()
            .filter(|frame| {
                frame.phase == FramePhase::Ready
                    && self.utility_allows_frame(frame.id)
                    && self.machine.waiting_on(frame.id).is_none()
                    && self
                        .exclusive_scope
                        .as_ref()
                        .is_none_or(|scope| scope.allows(frame.id))
                    && !self.ready_ids.contains(&frame.id)
                    && !self.profiled_frame_pending(frame.id)
                    && self
                        .nnuec_pending
                        .as_ref()
                        .is_none_or(|pending| pending.work.frame != frame.id)
                    && !self.validator_waiting.contains(&frame.id)
                    && !self.roles.iter().any(|role| {
                        role.work
                            .as_ref()
                            .is_some_and(|work| work.frame == frame.id)
                    })
                    && !self.own.iter().any(|slot| {
                        slot.binding.as_ref().is_some_and(|binding| {
                            binding.work.frame == frame.id && !binding.feedback_forwarded
                        })
                    })
                    && !self
                        .foreign
                        .as_ref()
                        .is_some_and(|binding| binding.work.frame == frame.id)
            })
            .take(
                self.limits
                    .waiting_jobs_max
                    .saturating_sub(self.waiting_jobs_occupancy()),
            )
            .map(|frame| {
                (
                    frame.id,
                    frame.original_first_move,
                    frame.input.view_seal.root_state.semantic,
                    frame.original_deadline,
                    frame.input.stage,
                )
            })
            .collect();
        for (id, candidate, locality, deadline, stage) in new_ready {
            self.ready
                .admit(ReadyJob {
                    job: id,
                    candidate,
                    locality,
                    hard_deadline: Duration::from_nanos(deadline.at.0),
                    enqueued_at: now_duration,
                    dependencies_ready: true,
                    dependency_priority: u8::from(matches!(
                        stage,
                        RefinementStage::FreshRepairedEndpoint
                            | RefinementStage::FreshCounterEndpoint
                    )),
                    service_cost: ReadyServiceCost::Unknown,
                })
                .map_err(|_| AdaptiveManagerError::Capacity("ready frame descriptor"))?;
            self.ready_ids.push(id);
        }
        let advice = if self.researcher_source_kind() == AdaptiveResearcherSourceKind::Native {
            Some(self.advice(now)?)
        } else {
            None
        };
        let native_worker_limit = advice.map_or(0, |advice| usize::from(advice.workers));
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let cohort_changed =
            self.maybe_admit_utility_cohort()? | self.maybe_admit_utility_intervention()?;
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        let cohort_changed = false;
        let mut changed = peers_changed || reciprocal_changed || cohort_changed;
        if advice.is_some_and(|advice| advice.yield_selected_worker) {
            if let Some(selection) = self.advised_yield.take() {
                // Advice and delivery refer to the same original execution;
                // do not rescan and transfer one owner's forecast to another.
                match selection {
                    YieldSelection::Own {
                        index, execution, ..
                    } => {
                        if self.own[index].binding.as_ref().is_some_and(|binding| {
                            binding.grant.execution == execution
                                && !binding.feedback_forwarded
                                && !binding.resume_pending
                                && binding.reciprocal.is_none()
                                && binding.requested_yield.is_none()
                        }) {
                            let sequence = self.issue_control()?;
                            let binding = self.own[index]
                                .binding
                                .as_mut()
                                .expect("selected CPU binding");
                            let requested_at = Instant::now();
                            if scheduling_yield(&binding.controls, sequence)? {
                                binding.requested_yield = Some((sequence, requested_at));
                                self.counters.cpu_yield_requests += 1;
                                changed = true;
                            }
                        }
                    }
                    YieldSelection::External { scope, .. } => {
                        if self.foreign.as_ref().is_some_and(|binding| {
                            binding.job.scope() == scope && binding.requested_yield.is_none()
                        }) {
                            let sequence = self.issue_control()?;
                            let binding = self.foreign.as_mut().expect("selected external binding");
                            let requested_at = Instant::now();
                            if scheduling_yield(&binding.job.control, sequence)? {
                                binding.requested_yield = Some((sequence, requested_at));
                                self.counters.foreign_yield_requests += 1;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
        for index in 0..self.own.len() {
            if !self.role_call_admission_available() {
                let retained_peer_r = self.own[index].binding.as_ref().is_some_and(|binding| {
                    self.accepted_roles
                        .iter()
                        .any(|held| held.child == Some(binding.work.frame))
                });
                if self.peer_policy.is_none() {
                    break;
                }
                if !retained_peer_r {
                    continue;
                }
            }
            if self.own[index].binding.as_ref().is_some_and(|binding| {
                binding.feedback_forwarded
                    && binding.resume_pending
                    && binding.pending_slice.is_none()
            }) {
                let frame = self.own[index]
                    .binding
                    .as_ref()
                    .expect("detached custody")
                    .work
                    .frame;
                if self.machine.frame(frame).is_none() {
                    // The logical question is closed. Ask the same physical
                    // owner to discard its token; retain custody until ack.
                    let sequence = self.issue_control()?;
                    let binding = self.own[index].binding.as_mut().expect("detached custody");
                    binding
                        .controls
                        .request_cancel(sequence)
                        .map_err(|_| AdaptiveManagerError::Invalid("detached CPU cancel"))?;
                    binding.resume_pending = false;
                    changed = true;
                } else if self.utility_scope_open() {
                    // The completed action step's old stack is sunk custody,
                    // not another utility route. Cancel the same owner/token;
                    // its terminal and final drain remain inside this cost.
                    let sequence = self.issue_control()?;
                    let binding = self.own[index]
                        .binding
                        .as_mut()
                        .expect("original utility CPU owner");
                    binding.controls.request_cancel(sequence).map_err(|_| {
                        AdaptiveManagerError::Invalid("utility detached CPU cancel")
                    })?;
                    binding.resume_pending = false;
                    changed = true;
                } else if let Some(slice) = advice.and_then(|advice| advice.slice) {
                    let result = self.resume_cpu(index, slice.node_credit, slice.soft_lease);
                    changed |= self.native_admission_result(result)?;
                }
                continue;
            }
            if self.own[index].binding.as_ref().is_some_and(|binding| {
                !binding.feedback_forwarded
                    && binding.resume_pending
                    && binding.pending_slice.is_none()
                    && binding
                        .reciprocal
                        .as_ref()
                        .is_none_or(|held| held.reply.is_some())
                    && !self.validator_waiting.contains(&binding.work.frame)
                    && self
                        .exclusive_scope
                        .as_ref()
                        .is_none_or(|scope| scope.allows(binding.work.frame))
            }) {
                if self.own[index]
                    .binding
                    .as_ref()
                    .is_some_and(|binding| binding.reciprocal.is_none())
                    && self.registration.endpoints.load_v
                    && self.registration.endpoints.execution_use == CpuExecutionUse::Collection
                {
                    let binding = self.own[index]
                        .binding
                        .as_ref()
                        .expect("checked paused binding");
                    if binding.validator_reviewed_execution != Some(binding.grant.execution) {
                        let work = binding.work.clone();
                        let execution = binding.grant.execution;
                        let budget = self
                            .machine
                            .frame(work.frame)
                            .ok_or(AdaptiveManagerError::Invalid("paused V parent disappeared"))?
                            .budget;
                        let profiles: Vec<_> = self
                            .own
                            .iter()
                            .filter_map(|slot| slot.worker.as_ref().map(CpuWorker::profile))
                            .collect();
                        if let Some(input) = self.driver.validator_input(
                            &work,
                            &profiles,
                            self.registration.endpoints,
                            budget,
                        )? {
                            if self.roles.len() >= self.role_pending_limit()
                                || self.validator_waiting.len() >= self.validator_waiting_limit()
                            {
                                // Keep this same paused owner/token until its
                                // role seat is available. This is feedback for
                                // a started Job, not another CPU Job admission.
                                continue;
                            }
                            self.prepare_validator(input)?;
                            self.validator_waiting.push(work.frame);
                            self.own[index]
                                .binding
                                .as_mut()
                                .expect("same paused owner")
                                .validator_reviewed_execution = Some(execution);
                            changed = true;
                            continue;
                        }
                        self.own[index]
                            .binding
                            .as_mut()
                            .expect("same paused owner")
                            .validator_reviewed_execution = Some(execution);
                    }
                }
                let Some(slice) = advice.and_then(|advice| advice.slice) else {
                    continue;
                };
                let result = self.resume_cpu(index, slice.node_credit, slice.soft_lease);
                changed |= self.native_admission_result(result)?;
            }
        }
        if self.phase != ManagerPhase::Running {
            return Ok(changed);
        }
        if self.registration.policy == CoordinatorPolicy::Synchronous
            && (!self.roles.is_empty()
                || self.own.iter().any(|slot| {
                    slot.binding
                        .as_ref()
                        .is_some_and(|binding| binding.reciprocal.is_none())
                })
                || self.foreign.is_some()
                || self.profiled_busy()
                || !self.deferred.is_empty())
        {
            return Ok(changed);
        }
        for _ in 0..64 {
            if (self.roles.len() >= self.role_pending_limit()
                || !self.role_call_admission_available())
                && self.peer_policy.is_none()
            {
                break;
            }
            let own_available = self
                .own
                .iter()
                .take(native_worker_limit)
                .any(|slot| slot.binding.is_none() && !slot.joined);
            // Do not keep selecting an ineligible CPU job merely to alter its
            // deficit. The role queue may still progress while CPU is occupied.
            let available: Vec<_> = self
                .ready_ids
                .iter()
                .filter_map(|id| {
                    self.machine.frame(*id).map(|frame| {
                        (
                            *id,
                            (if frame.input.stage.role().is_some() {
                                self.role_call_admission_available()
                                    && self.roles.len() < self.role_pending_limit()
                            } else {
                                (own_available && self.nnuec_work_eligible(frame.id))
                                    || (self.external.is_some() && self.foreign.is_none())
                                    || self.profiled_dispatch_available()
                            }) && self.utility_allows_frame(frame.id)
                                && self
                                    .exclusive_scope
                                    .as_ref()
                                    .is_none_or(|scope| scope.allows(frame.id)),
                        )
                    })
                })
                .collect();
            for (id, eligible) in available {
                self.ready
                    .set_dependencies_ready(&id, eligible)
                    .map_err(|_| AdaptiveManagerError::Invalid("ready dependency update"))?;
            }
            let selected = if let Some(pending) = self
                .nnuec_pending
                .as_ref()
                .filter(|pending| pending.job_ready() && pending.slot < native_worker_limit)
            {
                Some(pending.selection.clone())
            } else {
                self.ready
                    .select(now_duration, None)
                    .map_err(|_| AdaptiveManagerError::Invalid("ready fairness selector"))?
            };
            let Some(selected) = selected else {
                break;
            };
            self.ready_ids.retain(|id| *id != selected.job.job);
            let work = self.machine.ready_work(
                selected.job.job,
                self.clock.domain(),
                self.check_running()?,
            )?;
            let utility_started = match self.maybe_start_utility_route(&work) {
                Ok(started) => started,
                Err(AdaptiveManagerError::AuxiliaryPending) => {
                    self.ready.admit(selected.job).map_err(|_| {
                        AdaptiveManagerError::Capacity("original utility waiting descriptor")
                    })?;
                    self.ready_ids.push(work.frame);
                    return Ok(changed);
                }
                Err(error) => return Err(error),
            };
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            let preparing_intervention = self.utility_intervention.is_some();
            #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
            let preparing_intervention = false;
            if utility_started && (work.input.stage.role().is_some() || preparing_intervention) {
                self.ready.admit(selected.job).map_err(|_| {
                    AdaptiveManagerError::Capacity("original utility ready descriptor")
                })?;
                self.ready_ids.push(work.frame);
                return Ok(true);
            }
            let reciprocal_child = self.is_reciprocal_child(work.frame);
            if let Some(mut terminal) = if reciprocal_child {
                None
            } else {
                self.driver.short_circuit(&work)?
            } {
                self.driver.rovent_rules_endpoint(
                    &work,
                    self.clock.domain(),
                    self.check_running()?,
                )?;
                if terminal.checked.fresh_pair.is_some() {
                    return Err(AdaptiveManagerError::Invalid(
                        "Rules proof cannot claim neural Fresh pair",
                    ));
                }
                if terminal.checked.rules_pair.is_some()
                    && terminal.checked.queued_recheck.is_some()
                {
                    return Err(AdaptiveManagerError::Invalid(
                        "Rules pair cannot also register a Repair recheck queue",
                    ));
                }
                if let Some(proof) = terminal.checked.queued_recheck.take() {
                    self.machine.register_checked_recheck_queue(
                        proof,
                        self.clock.domain(),
                        self.check_running()?,
                    )?;
                }
                let closed = if let Some(proof) = terminal.checked.rules_pair {
                    self.driver.validate_rules_pair(&work, &proof)?;
                    if !matches!(
                        terminal.checked.transition,
                        FrameTransition::Complete {
                            scope: FrameCompletionScope::CompletedRulesPair
                        }
                    ) {
                        return Err(AdaptiveManagerError::Invalid(
                            "ready Rules-pair transition differs",
                        ));
                    }
                    Some(self.machine.resolve_ready_rules_pair(
                        work.frame,
                        &proof,
                        terminal.checked.evidence,
                        self.clock.domain(),
                        self.check_running()?,
                    )?)
                } else {
                    self.machine.resolve_rules_terminal(
                        work.frame,
                        &terminal.endpoint,
                        terminal.checked.evidence,
                        terminal.checked.transition,
                        self.clock.domain(),
                        self.check_running()?,
                    )?
                };
                if let Some(closed) = closed {
                    self.publish_closed_frame(&closed)?;
                }
                changed = true;
                continue;
            }
            if !reciprocal_child {
                if let Some(boundary) = self.driver.prepare_rovent_goal_boundary(&work)? {
                    self.machine.advance_original_goal_boundary(
                        work.frame,
                        boundary,
                        self.clock.domain(),
                        self.check_running()?,
                    )?;
                    changed = true;
                    continue;
                }
            }
            if self.registration.endpoints.load_v && !reciprocal_child {
                if self.exclusive_requested && self.exclusive_scope.is_none() {
                    if !self.roles.is_empty()
                        || self.own.iter().any(|slot| slot.binding.is_some())
                        || self.foreign.is_some()
                        || !self.deferred.is_empty()
                        || self.registry.occupancy() != 0
                        || self.native_credit_outstanding()?
                    {
                        return Err(AdaptiveManagerError::Invalid(
                            "exclusive scope requires idle physical owners",
                        ));
                    }
                    self.exclusive_scope = Some(ExclusiveActionScope {
                        parent: work.frame,
                        started: Instant::now(),
                        baseline: self.counters,
                        source: None,
                        action: None,
                        frame: None,
                        completed: false,
                        question: None,
                    });
                }
                // An exclusive action's child never launches a second V
                // decision. Its original V-selected action owns this scope.
                if self
                    .exclusive_scope
                    .as_ref()
                    .is_some_and(|scope| scope.action.is_some())
                {
                    // Proceed directly to the selected physical continuation.
                } else {
                    let profiles: Vec<_> = self
                        .own
                        .iter()
                        .filter_map(|slot| slot.worker.as_ref().map(CpuWorker::profile))
                        .collect();
                    let budget = self
                        .machine
                        .frame(work.frame)
                        .ok_or(AdaptiveManagerError::Invalid("V parent disappeared"))?
                        .budget;
                    if let Some(input) = self.driver.validator_input(
                        &work,
                        &profiles,
                        self.registration.endpoints,
                        budget,
                    )? {
                        let parent = work.frame;
                        if self.waiting_jobs_occupancy() >= self.limits.waiting_jobs_max
                            || self.validator_waiting.len() >= self.validator_waiting_limit()
                        {
                            return Err(AdaptiveManagerError::Capacity("V pending parents"));
                        }
                        self.prepare_validator(input)?;
                        self.validator_waiting.push(parent);
                        changed = true;
                        continue;
                    }
                    if self.exclusive_requested {
                        return Err(AdaptiveManagerError::Invalid(
                            "exclusive scope has no actual V support question",
                        ));
                    }
                }
            }
            if work.input.stage.role().is_some() {
                let input = work.input.clone();
                let purpose = if reciprocal_child {
                    RolePurpose::Reciprocal
                } else if goals::is_fresh_endpoint(work.input.stage)
                    && self.driver.rovent_root_budget()?.is_some()
                {
                    RolePurpose::RoventFresh
                } else if self.peer_policy.is_some()
                    && matches!(
                        work.input.stage,
                        RefinementStage::ProposalContinuation
                            | RefinementStage::Repair
                            | RefinementStage::CriticDivergence
                            | RefinementStage::CriticReply
                            | RefinementStage::CounterContinuation
                    )
                {
                    RolePurpose::RoventRefinement
                } else {
                    RolePurpose::Refinement
                };
                match self.prepare_role(Some(work), input, purpose) {
                    Ok(_) => {
                        changed = true;
                    }
                    Err(AdaptiveManagerError::Native(TicketServiceError::Capacity)) => break,
                    Err(error) => return Err(error),
                }
            } else {
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                if self.researcher_source_kind() == AdaptiveResearcherSourceKind::ProfiledExternal {
                    changed |= self.begin_profiled_cpu(work)?;
                    if self.registration.policy == CoordinatorPolicy::Synchronous {
                        break;
                    }
                    continue;
                }
                let (spent, reserved) = self.native_ledger()?.global_usage()?;
                let total = spent
                    .checked_add(reserved)
                    .ok_or(AdaptiveManagerError::Invalid(
                        "native CPU spent/reserved overflow",
                    ))?;
                if total > self.registration.native_cpu_node_limit {
                    return Err(AdaptiveManagerError::Invalid(
                        "native CPU root credit overrun",
                    ));
                }
                let endpoint = dispatch_cpu_endpoint(
                    self.registration.endpoints,
                    self.researcher_source_kind(),
                    self.nnuec.is_some(),
                );
                // Outstanding reservations are not completed work. Leave the
                // still-ready frame for the next tick after its real receipt.
                if endpoint == CpuEndpoint::Own
                    && total == self.registration.native_cpu_node_limit
                    && reserved != 0
                {
                    continue;
                }
                let physical_budget_exhausted = endpoint == CpuEndpoint::Own
                    && spent == self.registration.native_cpu_node_limit
                    && reserved == 0;
                if self.nnuec.is_some()
                    && endpoint == CpuEndpoint::Own
                    && !physical_budget_exhausted
                    && self.nnuec_pending.is_none()
                {
                    let Some(slot) =
                        self.own.iter().take(native_worker_limit).position(|slot| {
                            slot.binding.is_none() && !slot.joined && !slot.panicked
                        })
                    else {
                        break;
                    };
                    // Common quota is checked before C preparation or TT work.
                    // No native reservation or new Job exists at this point.
                    let capacity = self.driver.frontier_native_capacity(
                        1,
                        self.clock.domain(),
                        self.check_running()?,
                    );
                    match capacity {
                        Ok(Some(0)) => {
                            changed |= self.native_admission_result(Err(
                                AdaptiveManagerError::Frontier(FrontierError::NodeBudget),
                            ))?;
                            return Ok(changed);
                        }
                        Ok(_) => {}
                        Err(error) if self.semantic_replay_can_wait(&error)? => {
                            self.ready.admit(selected.job).map_err(|_| {
                                AdaptiveManagerError::Capacity("original NNUEC ready descriptor")
                            })?;
                            self.ready_ids.push(work.frame);
                            return Ok(changed);
                        }
                        Err(error) => {
                            changed |= self.native_admission_result(Err(error))?;
                            return Ok(changed);
                        }
                    }
                    self.start_nnuec_work(&work, slot, selected)?;
                    return Ok(true);
                }
                let profiles: Vec<_> = if self.nnuec.is_some() && !physical_budget_exhausted {
                    let pending =
                        self.nnuec_pending
                            .as_ref()
                            .ok_or(AdaptiveManagerError::Invalid(
                                "NNUEC exact work preparation missing",
                            ))?;
                    if pending.work.frame != work.frame || !pending.job_ready() {
                        return Err(AdaptiveManagerError::Invalid(
                            "NNUEC work is not acknowledged by original owner",
                        ));
                    }
                    vec![
                        self.own[pending.slot]
                            .worker
                            .as_ref()
                            .ok_or(AdaptiveManagerError::Invalid(
                                "NNUEC original worker missing",
                            ))?
                            .acknowledged_profile()?,
                    ]
                } else {
                    self.own
                        .iter()
                        .filter_map(|slot| slot.worker.as_ref().map(CpuWorker::profile))
                        .collect()
                };
                let question = match self.driver.cpu_question(
                    &work,
                    &profiles,
                    self.external.as_ref().map(|external| &external.config),
                    endpoint,
                    physical_budget_exhausted,
                ) {
                    Ok(question) => question,
                    Err(AdaptiveManagerError::Semantic(detail)) => {
                        let usage = self
                            .native_ledger()
                            .and_then(|ledger| ledger.global_usage().map_err(Into::into));
                        return Err(AdaptiveManagerError::Semantic(format!(
                            "CPU question dispatch manager_spent_reserved={usage:?}; {detail}"
                        )));
                    }
                    Err(other) => return Err(other),
                };
                match question {
                    CpuQuestion::OwnBudgetExhausted {
                        semantic_limit,
                        semantic_spent,
                    } => {
                        if !physical_budget_exhausted
                            || semantic_limit != self.registration.native_cpu_node_limit
                            || semantic_spent > spent
                            || self.has_execution_failure()
                            || !self.quarantined_cpu_receipts.is_empty()
                        {
                            return Err(AdaptiveManagerError::Invalid(
                                "semantic CPU budget boundary lacks settled physical credit",
                            ));
                        }
                        self.cpu_node_limit_reached = true;
                        self.begin_root_quiesce(self.cleanup_until()?)?;
                        return Ok(true);
                    }
                    CpuQuestion::Own(question) => {
                        let slot = if self.nnuec.is_some() {
                            self.nnuec_pending.as_ref().map(|pending| pending.slot)
                        } else {
                            self.own
                                .iter()
                                .take(native_worker_limit)
                                .position(|slot| slot.binding.is_none() && !slot.joined)
                        };
                        let Some(slot) = slot else {
                            break;
                        };
                        let Some(slice) = advice.and_then(|advice| advice.slice) else {
                            break;
                        };
                        let result = self.begin_cpu(
                            slot,
                            work,
                            question,
                            slice.node_credit,
                            slice.soft_lease,
                        );
                        changed |= self.native_admission_result(result)?;
                        if self.phase != ManagerPhase::Running {
                            return Ok(changed);
                        }
                    }
                    CpuQuestion::External(question) => {
                        if self.foreign.is_some() {
                            break;
                        }
                        self.begin_foreign(work, question)?;
                        changed = true;
                    }
                }
            }
            if self.registration.policy == CoordinatorPolicy::Synchronous {
                break;
            }
        }
        Ok(changed)
    }
    /// The exact worker used for gain/cost advice is also the only worker that
    /// can receive its Yield. Do not evaluate one owner then control another.
    fn selected_yield_worker(&self) -> Option<(usize, FeedbackWaitPrediction)> {
        // Adaptive's completed-iteration handoff drives the existing P/C
        // continuation without V. Fixed/synchronous order stays unchanged.
        if self.phase != ManagerPhase::Running
            || self.registration.policy != CoordinatorPolicy::Adaptive
            || !self.driver.semantic_owner_available()
            || self.roles.len() >= self.role_pending_limit()
            || self.validator_waiting.len() >= self.validator_waiting_limit()
        {
            return None;
        }
        let now = Instant::now();
        self.own.iter().enumerate().find_map(|(index, slot)| {
            let binding = slot.binding.as_ref()?;
            if !binding.progress_since_slice
                || binding.feedback_forwarded
                || binding.requested_yield.is_some()
                || binding.resume_pending
                || binding.pending_slice.is_some()
                || self.validator_waiting.contains(&binding.work.frame)
                || self
                    .exclusive_scope
                    .as_ref()
                    .is_some_and(|scope| !scope.allows(binding.work.frame))
            {
                return None;
            }
            let since_publication = now.checked_duration_since(binding.progress_published_at?)?;
            let owner_elapsed = now.checked_duration_since(binding.owner_run_started_at?)?;
            let lease_remaining = binding.grant.soft_lease.checked_sub(owner_elapsed)?;
            let prediction = self.driver.predict_yield_feedback(
                &binding.work,
                &self.costs,
                binding.class,
                since_publication,
                lease_remaining,
            )?;
            Some((index, prediction))
        })
    }
    fn selected_external_yield(&self) -> Option<ExternalFeedbackWaitPrediction> {
        if self.phase != ManagerPhase::Running
            || self.registration.policy != CoordinatorPolicy::Adaptive
            || !self.driver.semantic_owner_available()
            || self.roles.len() >= self.role_pending_limit()
            || self.validator_waiting.len() >= self.validator_waiting_limit()
        {
            return None;
        }
        let binding = self.foreign.as_ref()?;
        let observation = binding.observed_progress.as_ref()?;
        if binding.requested_yield.is_some()
            || !observation.is_new_coherent_tuple()
            || observation.stop_sent_at().is_some()
            || observation.go_issued_at().is_none()
            || observation.progress().scope != binding.job.scope()
            || self.validator_waiting.contains(&binding.work.frame)
            || self
                .exclusive_scope
                .as_ref()
                .is_some_and(|scope| !scope.allows(binding.work.frame))
            || !self
                .driver
                .external_feedback_eligible(&binding.work, observation.progress())
            || !self.machine.frame(binding.work.frame).is_some_and(|frame| {
                frame.phase
                    == FramePhase::Waiting(StepExecution::ExternalCpu {
                        job: binding.job.envelope.id,
                        execution: binding.job.grant.execution,
                    })
            })
        {
            return None;
        }
        let now = Instant::now();
        self.costs.external_feedback_wait_prediction(
            binding.class,
            now.checked_duration_since(observation.published_at())?,
            observation
                .quantum_deadline()?
                .checked_duration_since(now)?,
        )
    }
    fn advice(
        &mut self,
        now: MonotonicTick,
    ) -> Result<rz_runtime::adaptive::ControllerAdvice, AdaptiveManagerError> {
        let selected = self
            .selected_yield_worker()
            .map(|(index, prediction)| YieldSelection::Own {
                index,
                execution: self.own[index]
                    .binding
                    .as_ref()
                    .expect("selected CPU owner")
                    .grant
                    .execution,
                prediction,
            })
            .or_else(|| {
                self.selected_external_yield()
                    .map(|prediction| YieldSelection::External {
                        scope: self
                            .foreign
                            .as_ref()
                            .expect("selected external owner")
                            .job
                            .scope(),
                        prediction,
                    })
            });
        self.advised_yield = selected;
        let predicted_critic_bucket = selected.map(|selection| match selection {
            YieldSelection::Own { index, .. } => {
                self.own[index]
                    .binding
                    .as_ref()
                    .expect("selected owner")
                    .work
                    .input
                    .view_seal
                    .length_bucket
            }
            YieldSelection::External { .. } => {
                self.foreign
                    .as_ref()
                    .expect("selected foreign owner")
                    .work
                    .input
                    .view_seal
                    .length_bucket
            }
        });
        let role_class = if let Some(bucket) = predicted_critic_bucket {
            self.observed_role_classes.iter().find_map(|(role, class)| {
                (*role == Role::Critic && class.input_bucket == bucket).then_some(*class)
            })
        } else {
            self.roles.iter().find_map(|binding| binding.class)
        };
        // Prediction across observed completed source classes does not label
        // the pending role as an actual Hit/Miss or prove physical completion.
        let role_service = role_class.and_then(|class| {
            self.costs
                .service_prediction_envelope(class)
                .map(|envelope| envelope.estimate())
        });
        let running = self
            .roles
            .iter()
            .filter(|binding| !binding.service_completed)
            .find_map(|binding| binding.dispatched_at.zip(binding.class))
            .map(|(started, class)| RunningCost {
                class,
                elapsed: ObservedCost::Known(started.elapsed()),
            });
        let queued: Vec<_> = self
            .roles
            .iter()
            .filter(|binding| !binding.service_completed && binding.dispatched_at.is_none())
            .filter_map(|binding| binding.class)
            .collect();
        let backlog = if queued.len() + usize::from(running.is_some())
            != self
                .roles
                .iter()
                .filter(|binding| !binding.service_completed && binding.ack_accepted.is_none())
                .count()
        {
            BacklogEstimate::Unknown {
                known_lower_bound: Duration::ZERO,
            }
        } else {
            self.costs
                .predicted_gpu_backlog(running, &queued)
                .map_err(|_| AdaptiveManagerError::Invalid("GPU backlog bound"))?
        };
        let yield_overhead = selected
            .map(|selection| {
                // Include any calibrated request-to-ack cost conservatively; the
                // owner return-processing phase alone excludes safe-boundary wait.
                let class = match selection {
                    YieldSelection::Own { index, .. } => {
                        self.own[index]
                            .binding
                            .as_ref()
                            .expect("selected owner")
                            .class
                    }
                    YieldSelection::External { .. } => {
                        self.foreign
                            .as_ref()
                            .expect("selected external owner")
                            .class
                    }
                };
                self.costs
                    .estimate(class, CostStage::YieldLatency)
                    .filter(|latency| latency.p95 > selection.return_cost().p95)
                    .unwrap_or_else(|| selection.return_cost())
            })
            .or_else(|| {
                self.own
                    .iter()
                    .find_map(|slot| slot.binding.as_ref())
                    .and_then(|binding| self.costs.estimate(binding.class, CostStage::YieldLatency))
            });
        let input = ControllerInput {
            observation_sequence: self.observation_sequence,
            now: Duration::from_nanos(now.0),
            deadline_remaining: Duration::from_nanos(self.deadline.at.0.saturating_sub(now.0)),
            role_service,
            yield_overhead,
            backlog,
            waiting_cpu_jobs: self
                .ready
                .len()
                .saturating_add(usize::from(self.nnuec_pending.is_some())),
            // Typed S forecast is distinct from realized benefit/witness.
            selected_progress: selected.map(|selection| WorkerProgress {
                has_new_consistent_progress: true,
                expected_feedback_gain: Some(selection.gain()),
            }),
        };
        self.researcher
            .native_mut()?
            .controller
            .advise(input)
            .map_err(|_| AdaptiveManagerError::Invalid("adaptive advice"))
    }
    fn begin_cpu(
        &mut self,
        index: usize,
        work: CoordinatorWork<ReviewView, Q>,
        question: OwnCpuQuestion,
        requested_credit: u64,
        requested_lease: Duration,
    ) -> Result<(), AdaptiveManagerError> {
        if !self.can_admit_work() {
            return Err(AdaptiveManagerError::AdmissionClosed);
        }
        let profile = self.own[index]
            .worker
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid("CPU owner joined"))?
            .profile();
        let nnuec = self.claim_nnuec_job(&work, index)?;
        let state = question
            .position
            .export()
            .map_err(|_| AdaptiveManagerError::Invalid("CPU Rules export"))?;
        let job = CpuJobId {
            authority: self.registration.authority,
            sequence: self.issue_job()?,
        };
        let envelope = CpuJobEnvelope {
            id: job,
            state_identity: state.snapshot().identity(),
            question: work.input.question_digest,
            root_restriction: rz_search::cpu_worker::cpu_root_restriction_digest(
                question.root_moves.as_deref(),
            ),
            evaluation_namespace: profile.evaluation_namespace,
            search_conditions: profile.search_conditions,
            total_native_node_credit: question.total_native_node_credit,
            hard_deadline: self.deadline,
        };
        let mut prepared = self.own[index]
            .worker
            .as_ref()
            .expect("checked owner")
            .prepare_job(
                envelope,
                question.position,
                question.root_moves,
                question.target_depth,
                &self.cpu_clock,
            )?;
        if let Some(policy) = self
            .reciprocal_policy
            .filter(|_| !self.is_reciprocal_child(work.frame))
        {
            let parent = if let Some(original) = self.machine.reciprocal_question(work.frame) {
                self.driver
                    .register_cpu_reciprocal_scope(&work, &envelope, &original)?;
                original
            } else {
                let id = self.issue_reciprocal_question()?;
                let parent = self.driver.reciprocal_parent(
                    &work,
                    &envelope,
                    id,
                    policy.mode,
                    policy.root_budget,
                )?;
                self.machine.register_reciprocal_parent(
                    work.frame,
                    parent,
                    self.clock.domain(),
                    self.check_running()?,
                )?;
                parent
            };
            prepared.enable_reciprocal_questions(CpuReciprocalScope {
                job: envelope,
                owner: profile.owner,
                parent,
                recipient: policy.recipient,
                max_questions: policy.max_questions_per_job,
                node_interval: policy.node_interval,
            })?;
        }
        self.native_ledger_mut()?.register(envelope)?;
        let grant =
            match self.next_grant(envelope, profile.owner, requested_credit, requested_lease) {
                Ok(grant) => grant,
                Err(error) => {
                    self.native_ledger_mut()?.retire_unstarted(envelope.id)?;
                    return Err(error);
                }
            };
        let controls = Arc::clone(&prepared.mailbox);
        let class = cpu_cost_class(
            profile.evaluation_namespace,
            work.input.view_seal.length_bucket,
            self.registration.cpu_device_digest,
        );
        let admitted_at = match self.check_running() {
            Ok(now) => now,
            Err(error) => {
                self.native_ledger_mut()?.retire_unstarted(envelope.id)?;
                return Err(error);
            }
        };
        if let Err(error) = self.native_ledger_mut()?.reserve(grant) {
            self.native_ledger_mut()?.retire_unstarted(envelope.id)?;
            return Err(error.into());
        }
        let native_frontier = match self.driver.frontier_native_reserve(
            &envelope,
            grant,
            self.clock.domain(),
            admitted_at,
        ) {
            Ok(reservation) => reservation,
            Err(error) => {
                let rejected = self.own[index]
                    .worker
                    .as_ref()
                    .expect("checked owner")
                    .reject_prepared(prepared, grant)?;
                self.release_frontier_before_enqueue(None, rejected.proof())?;
                self.native_ledger_mut()?.retire_unstarted(envelope.id)?;
                return Err(error);
            }
        };
        if let Err(error) = self.driver.cpu_admitted(&work, &envelope, &grant) {
            let rejected = match self.own[index]
                .worker
                .as_ref()
                .expect("checked owner")
                .reject_prepared(prepared, grant)
            {
                Ok(rejected) => rejected,
                Err(rejection_error) => {
                    if let Some(reservation) = native_frontier {
                        self.retain_frontier_unknown(reservation);
                    }
                    return Err(rejection_error.into());
                }
            };
            self.release_frontier_before_enqueue(native_frontier, rejected.proof())?;
            self.native_ledger_mut()?.retire_unstarted(envelope.id)?;
            return Err(error);
        }
        // PonderHit or root control may close future admission during bounded
        // preparation. Only this actual unqueued owner proof may release the
        // original grant; an effective clock never refunds it by itself.
        let admission_error = if !self.can_admit_work() {
            Some(AdaptiveManagerError::AdmissionClosed)
        } else if let Some(capture) = nnuec.as_ref().and_then(|custody| custody.capture.as_ref()) {
            prepared
                .enable_nnuec_capture(capture.clone())
                .err()
                .map(|(error, _same_capture)| AdaptiveManagerError::Cpu(error))
        } else {
            None
        };
        let admission_error = admission_error
            .or_else(|| (!self.can_admit_work()).then_some(AdaptiveManagerError::AdmissionClosed));
        if let Some(error) = admission_error {
            match self.own[index]
                .worker
                .as_ref()
                .expect("checked owner")
                .reject_prepared_retaining(prepared, grant)
            {
                Ok(rejected) => {
                    self.release_frontier_before_enqueue(native_frontier, rejected.proof())?;
                    self.native_ledger_mut()?.retire_unstarted(envelope.id)?;
                    self.driver
                        .cpu_rejected(&work, &envelope, rejected.proof())?;
                    return Err(error);
                }
                Err((cleanup_error, original_job)) => {
                    self.retained_cpu_jobs.push((original_job, grant));
                    if let Some(reservation) = native_frontier {
                        self.retain_frontier_unknown(reservation);
                    }
                    self.native_ledger_mut()?.quarantine(envelope.id)?;
                    self.phase = ManagerPhase::Quarantined;
                    return Err(cleanup_error.into());
                }
            }
        }
        match self.own[index]
            .worker
            .as_ref()
            .expect("checked owner")
            .begin_retaining(prepared, grant)
        {
            Ok(()) => {}
            Err(rejected) => {
                self.release_frontier_before_enqueue(native_frontier, rejected.proof())?;
                self.native_ledger_mut()?.retire_unstarted(envelope.id)?;
                self.driver
                    .cpu_rejected(&work, &envelope, rejected.proof())?;
                return Err(AdaptiveManagerError::Cpu(rejected.reason().clone()));
            }
        }
        if let Some(state) = &mut self.cpu_semantic {
            state.enqueued(envelope);
        }
        // The manager is the only writer. No worker response is read between
        // successful admission and binding the exact execution to this frame.
        let frame = work.frame;
        self.own[index].binding = Some(OwnBinding {
            work,
            envelope,
            controls,
            grant,
            started_at: Instant::now(),
            class,
            resume_pending: false,
            feedback_forwarded: false,
            progress_sequence: 0,
            progress_since_slice: false,
            progress_published_at: None,
            owner_run_started_at: None,
            resume_cost_observed_execution: None,
            last_yield_ack: 0,
            requested_yield: None,
            pending_slice: None,
            queued_resume_termination: None,
            validator_reviewed_execution: None,
            reciprocal: None,
            native_frontier,
            nnuec,
            semantic_settlement: None,
        });
        self.machine
            .bind_cpu(frame, &envelope, &grant, self.clock.domain(), admitted_at)?;
        self.own[index]
            .binding
            .as_mut()
            .expect("actual admitted CPU binding")
            .work = self.machine.bound_work(
            frame,
            StepExecution::Cpu {
                job: envelope.id,
                execution: grant.execution,
            },
        )?;
        #[cfg(feature = "pals-adaptive-diagnostics")]
        if let (Some(diagnostic), Some(binding)) = (
            self.cpu_slice_diagnostic.get(),
            self.own[index].binding.as_ref(),
        ) {
            if let Some(custody) = &binding.nnuec {
                // The worker acknowledged this actual custody before Begin.
                // This row attests enqueue + frame bind only; invocation,
                // numeric readout and paused resume need their own receipts.
                diagnostic.record_nnuec_chain(
                    super::cpu_slice_observation::NnuecChainEvent::NnuecJobBound(
                        super::cpu_slice_observation::NnuecJobBindingObservation {
                            frame: binding.work.frame,
                            question: binding.work.input.question_digest,
                            view: binding.work.input.view_seal,
                            full_view: custody.seal.full_view_digest(),
                            job: binding.envelope,
                            first_grant: binding.grant,
                            installation_sequence: custody.receipt.installation_id().sequence(),
                            installation_succeeded: custody.receipt.is_installed(),
                            binding: custody.receipt.binding_digest(),
                            installed_owner: custody
                                .receipt
                                .installed_profile()
                                .map(|profile| profile.owner),
                            actual_stack_owner_id: custody
                                .receipt
                                .result()
                                .ok()
                                .map(|readback| readback.actual_stack_owner_id()),
                        },
                    ),
                );
            }
        }
        Ok(())
    }
    fn resume_cpu(
        &mut self,
        index: usize,
        credit: u64,
        lease: Duration,
    ) -> Result<(), AdaptiveManagerError> {
        if !self.can_admit_work() {
            return Err(AdaptiveManagerError::AdmissionClosed);
        }
        if self.own[index].worker.is_none() {
            return Err(AdaptiveManagerError::Invalid("paused CPU worker joined"));
        }
        let mut binding = self.own[index]
            .binding
            .take()
            .ok_or(AdaptiveManagerError::Invalid("missing paused CPU owner"))?;
        if binding.native_frontier.is_some() || self.pending_frontier_for(binding.envelope.id) {
            self.own[index].binding = Some(binding);
            return Err(AdaptiveManagerError::Frontier(FrontierError::NodesReserved));
        }
        let reply = match binding.reciprocal.as_ref() {
            Some(held) => match held.reply.clone() {
                Some(reply) => Some(reply),
                None => {
                    self.own[index].binding = Some(binding);
                    return Err(AdaptiveManagerError::Invalid(
                        "reciprocal checkpoint lacks accepted reply",
                    ));
                }
            },
            None => None,
        };
        let grant = match self.next_grant(binding.envelope, binding.grant.owner, credit, lease) {
            Ok(grant) => grant,
            Err(error) => {
                self.own[index].binding = Some(binding);
                return Err(error);
            }
        };
        let admitted_at = match self.check_running() {
            Ok(now) => now,
            Err(error) => {
                self.own[index].binding = Some(binding);
                return Err(error);
            }
        };
        if let Err(error) = self.native_ledger_mut()?.reserve(grant) {
            self.own[index].binding = Some(binding);
            return Err(error.into());
        }
        let native_frontier = match self.driver.frontier_native_reserve(
            &binding.envelope,
            grant,
            self.clock.domain(),
            admitted_at,
        ) {
            Ok(reservation) => reservation,
            Err(error) => {
                let rejected = self.own[index]
                    .worker
                    .as_ref()
                    .expect("checked paused owner")
                    .reject_unsubmitted_resume(grant);
                self.own[index].binding = Some(binding);
                self.release_frontier_before_enqueue(None, rejected?.proof())?;
                return Err(error);
            }
        };
        if let Err(error) = self
            .driver
            .cpu_admitted(&binding.work, &binding.envelope, &grant)
        {
            let rejected = self.own[index]
                .worker
                .as_ref()
                .expect("checked paused owner")
                .reject_unsubmitted_resume(grant);
            self.own[index].binding = Some(binding);
            let rejected = match rejected {
                Ok(rejected) => rejected,
                Err(rejection_error) => {
                    if let Some(reservation) = native_frontier {
                        self.retain_frontier_unknown(reservation);
                    }
                    return Err(rejection_error.into());
                }
            };
            self.release_frontier_before_enqueue(native_frontier, rejected.proof())?;
            return Err(error);
        }
        let final_admission_error = self
            .check_running()
            .err()
            .or_else(|| (!self.can_admit_work()).then_some(AdaptiveManagerError::AdmissionClosed));
        if let Some(admission_error) = final_admission_error {
            let rejected = self.own[index]
                .worker
                .as_ref()
                .expect("same paused CPU worker")
                .reject_unsubmitted_resume(grant);
            self.own[index].binding = Some(binding);
            let rejected = match rejected {
                Ok(rejected) => rejected,
                Err(error) => {
                    if let Some(reservation) = native_frontier {
                        self.retain_frontier_unknown(reservation);
                    }
                    return Err(error.into());
                }
            };
            self.release_frontier_before_enqueue(native_frontier, rejected.proof())?;
            let binding = self.own[index]
                .binding
                .as_ref()
                .expect("same paused custody");
            self.driver
                .cpu_rejected(&binding.work, &binding.envelope, rejected.proof())?;
            return Err(admission_error);
        }
        let submitted = match reply {
            Some(reply) => self.own[index]
                .worker
                .as_ref()
                .expect("checked paused CPU worker")
                .resume_with_reply_attested(grant, reply),
            None => self.own[index]
                .worker
                .as_ref()
                .expect("checked paused CPU worker")
                .resume_attested(grant),
        };
        match submitted {
            Ok(()) => {}
            Err(rejected) => {
                let released =
                    self.release_frontier_before_enqueue(native_frontier, rejected.proof());
                let cleanup =
                    self.driver
                        .cpu_rejected(&binding.work, &binding.envelope, rejected.proof());
                self.own[index].binding = Some(binding);
                released?;
                cleanup?;
                return Err(AdaptiveManagerError::Cpu(rejected.reason().clone()));
            }
        }
        let frame = binding.work.frame;
        let envelope = binding.envelope;
        let reciprocal_question = binding
            .reciprocal
            .as_ref()
            .and_then(|held| held.question)
            .map(|question| question.id);
        binding.grant = grant;
        binding.native_frontier = native_frontier;
        binding.reciprocal = None;
        binding.resume_pending = false;
        binding.started_at = Instant::now();
        binding.progress_published_at = None;
        binding.owner_run_started_at = None;
        self.own[index].settled_execution = None;
        self.own[index].binding = Some(binding);
        if self.own[index]
            .binding
            .as_ref()
            .expect("resumed owner")
            .feedback_forwarded
        {
            // The owner continues its ORIGINAL question independently of the
            // newer logical P/C frame. Never bind this grant to that frame.
            return Ok(());
        }
        match self.machine.frame(frame).map(|frame| frame.phase) {
            Some(FramePhase::Suspended) => self.machine.make_resumed_ready(frame)?,
            // The actual accepted ResumeTask already made this same owner
            // ready. bind_cpu still checks the original job and new grant.
            Some(FramePhase::Ready) => {}
            _ => {
                return Err(AdaptiveManagerError::Invalid(
                    "same-owner resume frame phase",
                ));
            }
        }
        self.machine
            .bind_cpu(frame, &envelope, &grant, self.clock.domain(), admitted_at)?;
        self.own[index]
            .binding
            .as_mut()
            .expect("actual resumed CPU binding")
            .work = self.machine.bound_work(
            frame,
            StepExecution::Cpu {
                job: envelope.id,
                execution: grant.execution,
            },
        )?;
        if let Some(question) = reciprocal_question {
            self.defer(DeferredCommit::ReciprocalDiscard { question })?;
        }
        Ok(())
    }
    fn next_grant(
        &mut self,
        envelope: CpuJobEnvelope,
        owner: CpuWorkerId,
        credit: u64,
        lease: Duration,
    ) -> Result<SliceGrant, AdaptiveManagerError> {
        let (spent, reserved, quarantined) = self.native_ledger()?.usage(envelope.id)?;
        let (global_spent, global_reserved) = self.native_ledger()?.global_usage()?;
        if quarantined || reserved != 0 {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        let mut credit = credit
            .min(envelope.total_native_node_credit.saturating_sub(spent))
            .min(
                self.registration
                    .native_cpu_node_limit
                    .saturating_sub(global_spent)
                    .saturating_sub(global_reserved),
            );
        let now = self.check_running()?;
        if let Some(capacity) =
            self.driver
                .frontier_native_capacity(credit, self.clock.domain(), now)?
        {
            credit = credit.min(capacity);
            if credit == 0 {
                return Err(AdaptiveManagerError::Frontier(FrontierError::NodesReserved));
            }
        }
        let lease = lease.min(Duration::from_nanos(
            self.deadline.at.0.saturating_sub(now.0),
        ));
        if credit == 0 || lease.is_zero() {
            return Err(AdaptiveManagerError::Capacity(
                "native CPU credit/deadline exhausted",
            ));
        }
        Ok(SliceGrant {
            job: envelope.id,
            execution: ExecutionId::new(self.registration.authority.epoch, self.issue_execution()?),
            owner,
            node_credit: credit,
            soft_lease: lease,
            control_sequence: self.issue_control()?,
        })
    }
    fn begin_foreign(
        &mut self,
        work: CoordinatorWork<ReviewView, Q>,
        question: ForeignCpuQuestion,
    ) -> Result<(), AdaptiveManagerError> {
        if !self.can_admit_work() {
            return Err(AdaptiveManagerError::AdmissionClosed);
        }
        if self.registration.endpoints.cpu_r != CpuEndpoint::ExternalUci
            && self.registration.endpoints.cpu_t != CpuEndpoint::ExternalUci
        {
            return Err(AdaptiveManagerError::Invalid(
                "external CPU is not declared for this execution",
            ));
        }
        let state = question
            .position
            .export()
            .map_err(|_| AdaptiveManagerError::Invalid("external Rules export"))?;
        let external = self
            .external
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid("missing external owner"))?;
        let roots = external_root_restriction_digest(question.root_moves.as_deref());
        let evaluation = external_evaluation_namespace(&external.config.identity)
            .map_err(|_| AdaptiveManagerError::Invalid("external evaluation source pin"))?;
        let conditions =
            external_search_conditions(&external.config, question.target_depth, question.policy);
        let question_digest =
            external_question_digest(state.snapshot().identity(), roots, evaluation, conditions);
        // Physical UCI and semantic view/question digests are different
        // namespaces. Their checked producer mapping is required below.
        let owner = external.owner;
        let signal = external.worker.signal();
        let id = CpuJobId {
            authority: self.registration.authority,
            sequence: self.issue_job()?,
        };
        let execution =
            ExecutionId::new(self.registration.authority.epoch, self.issue_execution()?);
        let envelope = ExternalCpuJobEnvelope {
            id,
            state_identity: state.snapshot().identity(),
            question: question_digest,
            root_restriction: roots,
            evaluation_namespace: evaluation,
            search_conditions: conditions,
            requested_reported_node_hint: question.policy.requested_reported_nodes,
            hard_deadline: self.deadline,
        };
        let grant = ExternalSliceGrant {
            job: id,
            execution,
            owner,
            requested_reported_node_hint: question.policy.requested_reported_nodes,
            soft_quantum: question.policy.quantum,
            control_sequence: self.issue_control()?,
        };
        grant.validate(&envelope, self.clock.domain(), self.check_running()?)?;
        let scope = ExternalWorkerScope {
            job: id,
            execution,
            owner,
        };
        let job = ExternalJob {
            envelope,
            grant,
            execution,
            owner,
            position: question.position.into_position(),
            root_moves: question.root_moves,
            max_depth: question.target_depth,
            policy: question.policy,
            control: rz_runtime::adaptive::ControlMailbox::new(scope, signal),
        };
        let admitted_at = self.check_running()?;
        let proof = self.driver.external_admitted(
            &work,
            &envelope,
            &grant,
            self.clock.domain(),
            admitted_at,
        )?;
        let worker = &mut self
            .external
            .as_mut()
            .expect("checked external owner")
            .worker;
        let submitted = if self.registration.policy == CoordinatorPolicy::Adaptive {
            worker.submit_observed(job.clone())
        } else {
            worker.submit_attested(job.clone())
        };
        let teacher_dispatch = match submitted {
            Ok(dispatch) => dispatch,
            Err(error) => {
                self.driver.external_rejected(&work, &envelope, &grant)?;
                return Err(AdaptiveManagerError::Semantic(format!(
                    "external admission: {error:?}"
                )));
            }
        };
        let frame = work.frame;
        let class = cpu_cost_class(
            evaluation,
            work.input.view_seal.length_bucket,
            self.registration.cpu_device_digest,
        );
        self.foreign = Some(ForeignBinding {
            work,
            job,
            teacher_dispatch,
            started_at: Instant::now(),
            class,
            progress_sequence: 0,
            observed_progress: None,
            requested_yield: None,
        });
        self.machine.bind_checked_external_cpu(
            frame,
            &envelope,
            &grant,
            &proof,
            self.clock.domain(),
            admitted_at,
        )?;
        self.foreign
            .as_mut()
            .expect("actual admitted external binding")
            .work = self.machine.bound_work(
            frame,
            StepExecution::ExternalCpu {
                job: envelope.id,
                execution: grant.execution,
            },
        )?;
        let binding = self.foreign.as_ref().expect("retained external owner");
        self.driver.external_teacher_submitted(
            &binding.work,
            &binding.job,
            &binding.teacher_dispatch,
        )?;
        Ok(())
    }
    fn issue_job(&mut self) -> Result<u64, AdaptiveManagerError> {
        issue(&mut self.next_job)
    }
    fn issue_execution(&mut self) -> Result<u64, AdaptiveManagerError> {
        issue(&mut self.next_execution)
    }
    fn issue_control(&mut self) -> Result<u64, AdaptiveManagerError> {
        issue(&mut self.next_control)
    }
    fn observe(
        &mut self,
        class: CostClass,
        stage: CostStage,
        elapsed: Duration,
        disposition: CostDisposition,
    ) -> Result<(), AdaptiveManagerError> {
        self.observe_observation(
            class,
            stage,
            CostObservation {
                elapsed: ObservedCost::Known(elapsed),
                disposition,
            },
        )
    }
    fn observe_observation(
        &mut self,
        class: CostClass,
        stage: CostStage,
        observation: CostObservation,
    ) -> Result<(), AdaptiveManagerError> {
        self.costs
            .observe(class, stage, observation)
            .map_err(|_| AdaptiveManagerError::Capacity("bounded cost observations"))?;
        self.observation_sequence = self
            .observation_sequence
            .checked_add(1)
            .ok_or(AdaptiveManagerError::Capacity("cost observation sequence"))?;
        Ok(())
    }
    /// No manager join waits. Cancellation travels outside ordinary queues,
    /// then each owner retains its state until its real fence/terminal receipt.
    pub fn begin_shutdown(&mut self, until: Instant) -> Result<(), AdaptiveManagerError> {
        self.begin_close(until, AdaptiveCloseMode::ProcessShutdown)
    }
    fn begin_selected_shutdown(&mut self, until: Instant) -> Result<(), AdaptiveManagerError> {
        self.begin_close(until, self.close_mode)
    }
    pub fn begin_root_quiesce(&mut self, until: Instant) -> Result<(), AdaptiveManagerError> {
        self.begin_close(until, AdaptiveCloseMode::RootQuiesce)
    }
    fn begin_close(
        &mut self,
        until: Instant,
        mode: AdaptiveCloseMode,
    ) -> Result<(), AdaptiveManagerError> {
        let until = self
            .cleanup_boundary
            .map_or(until, |boundary| boundary.cap(until));
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let until = self
            .cohort_cleanup_ceiling()
            .map_or(until, |cap| cap.min(until));
        if self.phase == ManagerPhase::Finished {
            return Ok(());
        }
        let escalation = self.shutdown_until.is_some()
            && self.close_mode == AdaptiveCloseMode::RootQuiesce
            && mode == AdaptiveCloseMode::ProcessShutdown;
        if self.shutdown_until.is_some() && self.close_mode != mode && !escalation {
            return Err(AdaptiveManagerError::Invalid(
                "root/process closure mode changed",
            ));
        }
        self.close_mode = mode;
        // A failed root fence may escalate to an explicit process quit/join.
        // Its old failure and quarantine remain sticky. This new cleanup
        // window never extends any original search/admission deadline.
        let until = if escalation {
            until
        } else {
            self.shutdown_until.map_or(until, |old| old.min(until))
        };
        self.shutdown_until = Some(until);
        if self.phase != ManagerPhase::Quarantined {
            self.phase = ManagerPhase::Draining;
        }
        self.registry.revoke_all();
        self.machine.revoke();
        self.cancel_utility_preparation();
        // The same semantic owner closes its non-submitted local duties now.
        // If an archive transfer owns it, manager admission is already closed
        // and the exact owner close is retried before post-fence retirement.
        if self.driver.semantic_owner_available() {
            if let Err(error) = self.driver.close_rovent_goal_admission() {
                self.record_failure(error);
            }
        }
        if let Some(preparation) = &self.nnuec {
            preparation.canceled.store(true, Ordering::Release);
        }
        // First revoke question authority, then cancel each same-owner task.
        for index in 0..self.own.len() {
            if self.own[index]
                .binding
                .as_ref()
                .is_some_and(|binding| close_cancel_needed(&binding.controls))
            {
                let sequence = self.issue_control()?;
                if let Some(binding) = &self.own[index].binding {
                    let _ = binding.controls.request_cancel(sequence);
                }
            }
            if let Some(worker) = &self.own[index].worker {
                worker.request_shutdown();
            }
        }
        if self
            .foreign
            .as_ref()
            .is_some_and(|binding| close_cancel_needed(&binding.job.control))
        {
            let sequence = self.issue_control()?;
            if let Some(binding) = &self.foreign {
                let _ = binding.job.control.request_cancel(sequence);
            }
        }
        if let Some(external) = &self.external {
            match mode {
                AdaptiveCloseMode::ProcessShutdown => external.worker.request_shutdown(until),
                AdaptiveCloseMode::RootQuiesce => external
                    .worker
                    .request_root_quiesce(self.registration.authority, until)
                    .map_err(|error| {
                        AdaptiveManagerError::Semantic(format!("external root quiesce: {error:?}"))
                    })?,
            }
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        self.begin_profiled_close(until, mode)?;
        // Prepared/running logical cancellation never releases a GPU buffer.
        for role in &mut self.roles {
            if !role.cancellation_requested {
                role.cancellation_requested = self.native.cancel(role.preparation).is_ok();
            }
        }
        match mode {
            AdaptiveCloseMode::ProcessShutdown => self.native.begin_shutdown(until),
            AdaptiveCloseMode::RootQuiesce => {
                self.native.begin_root_quiesce(until)?;
                self.native_root_quiesce_requested = true;
            }
        }
        self.driver.begin_auxiliary_shutdown(until)?;
        Ok(())
    }
    fn poll_shutdown(&mut self) -> Result<bool, AdaptiveManagerError> {
        let mut changed = false;
        if self.shutdown_until.is_none() {
            let until = self.cleanup_until()?;
            self.begin_selected_shutdown(until)?;
            changed = true;
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        {
            changed |= self.poll_profiled_close()?;
        }
        for index in 0..self.own.len() {
            if self.own[index].joined {
                continue;
            }
            let joined = self.own[index]
                .worker
                .as_mut()
                .ok_or(AdaptiveManagerError::Invalid("missing CPU owner"))?
                .try_join()?;
            let Some(receipt) = joined else {
                continue;
            };
            let nnuec_readback_error = if let Some(pending) = &mut self.nnuec_pending {
                if pending.slot == index {
                    pending.observe_shutdown(&receipt).err()
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(actual) = receipt
                .value_installation_receipt
                .as_ref()
                .filter(|actual| actual.is_installed())
            {
                self.own[index].last_value_installation = Some(actual.clone());
            }
            changed = true;
            // Receipts retained at owner shutdown MUST settle before dropping
            // the worker and before archive close; progress never settles them.
            for event in receipt.events {
                if let Err(error) = self.own_event(index, event) {
                    self.record_failure(error);
                }
            }
            self.own[index].joined = receipt.thread_joined;
            self.own[index].panicked = receipt.thread_panicked;
            self.own[index].ready_readback = receipt.ready_readback;
            if let Some(error) = nnuec_readback_error {
                self.record_failure(AdaptiveManagerError::NnuecPreparation(error));
            }
            if let Some(error) = receipt.value_installation_error {
                // The independently retained terminal is not contingent on
                // there still being a matching pending installation slot.
                self.record_failure(AdaptiveManagerError::Cpu(error));
            }
            if !receipt.engine_recoverable {
                // A joined, non-panicked thread is not an engine ownership
                // fence. Keep actual usage settlement above, but never issue
                // a clean archive/root reuse receipt for this owner.
                self.record_failure(AdaptiveManagerError::Cpu(CpuWorkerError::Quarantined));
            }
            if receipt.thread_panicked {
                self.quarantine_frontier_slot(index);
                if let Some(job) = self.own[index]
                    .binding
                    .as_ref()
                    .map(|binding| binding.envelope.id)
                {
                    self.native_ledger_mut()?.quarantine(job)?;
                }
                self.phase = ManagerPhase::Quarantined;
            }
        }
        if self.close_mode == AdaptiveCloseMode::RootQuiesce && self.external_root_fence.is_none() {
            if let Some(external) = &self.external {
                if let Some(result) = external.worker.root_fence() {
                    match result {
                        Ok(fence)
                            if fence.authority() == self.registration.authority
                                && fence.owner() == external.owner
                                && fence.permits_reuse() =>
                        {
                            self.external_root_fence = Some(fence);
                            changed = true;
                        }
                        _ => {
                            self.external_finished = Some(false);
                            self.phase = ManagerPhase::Quarantined;
                            return Err(AdaptiveManagerError::PhysicalUnknown);
                        }
                    }
                }
            }
        }
        if self.close_mode == AdaptiveCloseMode::ProcessShutdown && self.external_finished.is_none()
        {
            if let Some(external) = &mut self.external {
                if let Some(result) = external.worker.try_join() {
                    changed = true;
                    self.external_finished = Some(result.as_ref().is_ok_and(|receipt| {
                        receipt.cleanup_complete
                            && receipt.exit_observed
                            && receipt.stdout_drained
                            && receipt.stderr_drained
                            && !receipt.quarantined
                            && !receipt.ownership_lost
                    }));
                    if self.external_finished == Some(false) {
                        self.phase = ManagerPhase::Quarantined;
                    }
                }
            }
        }
        if !self.native_joined && self.native_finished.is_some() {
            self.native_joined = self.native.try_join()?;
            changed |= self.native_joined;
        }
        let native_closed = (self.native_joined && self.native_finished == Some(true))
            || (self.close_mode == AdaptiveCloseMode::RootQuiesce
                && self.native_root_fence.is_some_and(|fence| {
                    fence.authority() == self.registration.authority && fence.permits_reuse()
                }));
        if native_closed
            && self.own.iter().all(|slot| slot.joined)
            && self.external_owner_closed()
            && self.deferred.is_empty()
            && self.nnuec_cleanup_settled()
            && self.utility_cleanup_settled()
            && self.native_frontier_events.is_empty()
            && self.own.iter().all(|slot| {
                slot.binding
                    .as_ref()
                    .is_none_or(|binding| binding.pending_slice.is_none())
            })
        {
            // A physical fence cannot resolve an ambiguous semantic commit.
            // Keep the original bindings and driver custody intact; in
            // particular do not discard a GoalLocal input or count its result
            // as unused when owner readback failed.
            if self.roles.iter().any(|binding| binding.consumption_unknown)
                || self
                    .accepted_roles
                    .iter()
                    .any(|held| held.binding.consumption_unknown)
            {
                self.phase = ManagerPhase::Quarantined;
                return Ok(changed);
            }
            self.release_nnuec_custody()?;
            self.driver.close_rovent_goal_admission()?;
            self.discard_goal_local_custody()?;
            if self.driver.rovent_goal_has_pending()? {
                return Err(AdaptiveManagerError::Invalid(
                    "GoalLocal owner did not retire after native fence",
                ));
            }
            self.driver.release_rovent_goal_state()?;
            self.driver.finish_auxiliary_shutdown()?;
        }
        if native_closed
            && self.own.iter().all(|slot| slot.joined)
            && self.external_owner_closed()
            && self.driver.auxiliary_shutdown_confirmed()
            && self.deferred.is_empty()
            && self.nnuec_cleanup_settled()
            && self.utility_cleanup_settled()
            && self.native_frontier_events.is_empty()
            && self.own.iter().all(|slot| {
                slot.binding
                    .as_ref()
                    .is_none_or(|binding| binding.pending_slice.is_none())
            })
            && self.phase != ManagerPhase::Quarantined
        {
            // Finished is an owner fence, not a late output acceptance.
            // Canceled questions are retired with zero search consumption.
            if let Some(observer) = &mut self.collection_observer {
                let tickets = self
                    .roles
                    .iter()
                    .filter_map(|r| r.ticket)
                    .collect::<Vec<_>>();
                if let Err(error) = observer.settled(&tickets) {
                    self.record_failure(error.into());
                }
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            if let Some(observer) = &mut self.collection_observer_v2 {
                let tickets = self
                    .roles
                    .iter()
                    .filter_map(|r| r.ticket)
                    // A peer source is moved out of `roles` after native
                    // acceptance. If shutdown arrives before search consumes
                    // it, its actual ticket still owns an observer input and
                    // must be retired at this same physical fence.
                    .chain(
                        self.accepted_roles
                            .iter()
                            .filter_map(|held| held.binding.ticket),
                    )
                    .collect::<Vec<_>>();
                let result = observer.settled(&tickets);
                self.collection_v2_settlement = Some(result.clone());
                if let Err(error) = result {
                    self.record_failure(error.into());
                }
            }
            let retired = self.registry.retire_revoked_after_worker_join()?;
            self.counters.unused_role_results = self
                .counters
                .unused_role_results
                .checked_add(retired.len() as u64)
                .ok_or(AdaptiveManagerError::Capacity("unused result sum"))?;
            self.roles.clear();
            let unused_accepted = self
                .accepted_roles
                .iter()
                .filter(|held| !held.binding.consumption_recorded)
                .count() as u64;
            self.counters.unused_role_results = self
                .counters
                .unused_role_results
                .checked_add(unused_accepted)
                .ok_or(AdaptiveManagerError::Capacity("unused accepted role sum"))?;
            self.accepted_roles.clear();
            self.acknowledgements.clear();
            self.retained_native_permits.clear();
            self.retained_native_receipts.clear();
            self.ready_ids.clear();
            self.validator_waiting.clear();
            self.root_ranking_proof = None;
            if self.native_credit_outstanding()?
                || self.own.iter().any(|slot| slot.binding.is_some())
                || self.foreign.is_some()
                || !self.retained_cpu_jobs.is_empty()
            {
                self.phase = ManagerPhase::Quarantined;
            } else {
                // Retain the original intervention source if final credit or
                // CPU custody checks quarantine this otherwise drained root.
                self.machine.retired_after_worker_join();
                self.phase = ManagerPhase::Finished;
                self.finished_at.get_or_insert_with(Instant::now);
            }
        }
        Ok(changed)
    }
    pub fn shutdown_receipt(&self) -> Result<AdaptiveShutdownReceipt, AdaptiveManagerError> {
        let (spent, reserved) = self.native_ledger()?.global_usage()?;
        Ok(AdaptiveShutdownReceipt {
            authority: self.registration.authority,
            native_owner_joined: self.native_joined,
            native_owner_quiesced: self
                .native_root_fence
                .is_some_and(|fence| fence.permits_reuse()),
            native_physical_shutdown_confirmed: self.native_finished == Some(true),
            own_cpu_joined: self.own.iter().filter(|slot| slot.joined).count(),
            expected_own_cpu_owners: self.own.len(),
            own_cpu_panicked: self.own.iter().filter(|slot| slot.panicked).count(),
            external_shutdown_confirmed: self.external.as_ref().and(self.external_finished),
            external_owner_quiesced: self.external.as_ref().map(|_| {
                self.external_root_fence
                    .is_some_and(|fence| fence.permits_reuse())
            }),
            auxiliary_shutdown_confirmed: self.driver.auxiliary_shutdown_confirmed()
                && self.nnuec_cleanup_settled()
                && self.utility_cleanup_settled()
                && self.nnuec_pending.is_none()
                && self.nnuec_jobs.is_empty()
                && self.goal_prepared.is_none()
                && !self.goal_continuation_pending,
            native_cpu_spent: spent,
            native_cpu_reserved: reserved,
            pending_tickets: self.registry.occupancy(),
            quarantined: self.phase == ManagerPhase::Quarantined
                || self.native.is_quarantined()
                || !self.quarantine_rejections.is_empty()
                || !self.quarantined_cpu_receipts.is_empty()
                || !self.quarantined_foreign_receipts.is_empty()
                || !self.native_frontier_events.is_empty(),
            // Common accounting remains separate from the original
            // native ledger totals but can veto a clean owner closure.
            request_sequence_high_water: self.registry.next_request_sequence().saturating_sub(1),
            counters: self.counters,
            native_cpu_thread_observation: self.native.ready_cpu_thread_observation(),
            own_cpu_thread_observations: self
                .own
                .iter()
                .map(|slot| {
                    slot.worker
                        .as_ref()
                        .and_then(|worker| worker.ready_readback().cloned())
                        .or_else(|| slot.ready_readback.clone())
                })
                .collect(),
            root_fence: self.native_root_fence,
            external_root_fence: self.external_root_fence,
            external_owner: self.external.as_ref().map(|external| external.owner),
        })
    }
    fn external_owner_closed(&self) -> bool {
        self.profiled_owner_closed()
            && self
                .external
                .as_ref()
                .is_none_or(|external| match self.close_mode {
                    AdaptiveCloseMode::ProcessShutdown => self.external_finished == Some(true),
                    AdaptiveCloseMode::RootQuiesce => {
                        self.external_root_fence.is_some_and(|fence| {
                            fence.authority() == self.registration.authority
                                && fence.owner() == external.owner
                                && fence.permits_reuse()
                        })
                    }
                })
    }
    /// Caller chooses and renews a finite window. There is no user-visible
    /// hidden infinite retry or manager spin loop. Notification shares the CPU
    /// and native service signal when workers were constructed spawn_notified.
    pub fn run_root_until(
        &mut self,
        until: Instant,
        canceled: &AtomicBool,
    ) -> Result<AdaptiveShutdownReceipt, AdaptiveManagerError> {
        self.run_root_until_observed(until, canceled, &mut |_| Ok(()))
    }
    /// Observe coherent Search-owned progress and UCI control after each tick.
    /// The observer may close admission at its UCI soft boundary; it cannot
    /// change original ticket/job deadlines or manufacture completed evidence.
    /// Fallible observation leaves this manager and all actual owners intact.
    pub fn run_root_until_observed(
        &mut self,
        until: Instant,
        canceled: &AtomicBool,
        observer: &mut dyn FnMut(&mut Self) -> Result<(), AdaptiveManagerError>,
    ) -> Result<AdaptiveShutdownReceipt, AdaptiveManagerError> {
        if self.shutdown_until.is_some() || self.phase != ManagerPhase::Running {
            return Err(AdaptiveManagerError::Invalid("root quiesce run admission"));
        }
        self.close_mode = AdaptiveCloseMode::RootQuiesce;
        self.run_until_observed(until, canceled, observer)
    }
    pub fn run_until(
        &mut self,
        until: Instant,
        canceled: &AtomicBool,
    ) -> Result<AdaptiveShutdownReceipt, AdaptiveManagerError> {
        self.run_until_observed(until, canceled, &mut |_| Ok(()))
    }
    fn run_until_observed(
        &mut self,
        until: Instant,
        canceled: &AtomicBool,
        observer: &mut dyn FnMut(&mut Self) -> Result<(), AdaptiveManagerError>,
    ) -> Result<AdaptiveShutdownReceipt, AdaptiveManagerError> {
        self.run_owner_until_observed(until, canceled, observer)?;
        self.shutdown_receipt()
    }
    fn run_owner_until_observed(
        &mut self,
        until: Instant,
        canceled: &AtomicBool,
        observer: &mut dyn FnMut(&mut Self) -> Result<(), AdaptiveManagerError>,
    ) -> Result<(), AdaptiveManagerError> {
        let until = self
            .cleanup_boundary
            .map_or(until, |boundary| boundary.cap(until));
        loop {
            let signal = self.native.signal();
            let observed = signal.version();
            if canceled.load(Ordering::Acquire) && self.phase == ManagerPhase::Running {
                self.begin_selected_shutdown(self.cleanup_until()?)?;
            }
            // Apply the caller's soft/admission boundary before a tick may
            // dispatch another immutable question. This control observation
            // does not alter its original hard deadline.
            observer(self)?;
            let changed = self.tick()?;
            // A quiet physical call still crosses the original UCI soft
            // admission boundary. Observation itself creates no completion
            // or progress evidence; the caller compares coherent results.
            observer(self)?;
            if self.phase == ManagerPhase::Running && self.root_started && !self.has_pending_work()
            {
                self.begin_selected_shutdown(self.cleanup_until()?)?;
            }
            if self.phase == ManagerPhase::Finished {
                return Ok(());
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            let owner_until = self
                .cohort_cleanup_ceiling()
                .map_or(until, |cap| cap.min(until));
            #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
            let owner_until = until;
            if Instant::now() >= owner_until {
                self.begin_selected_shutdown(owner_until)?;
                self.tick()?;
                return Ok(());
            }
            if !changed {
                // The closing poll bounds the scheduling gap between final
                // owner notify and JoinHandle::is_finished publication.
                let wait = owner_until
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(10));
                self.counters.notification_waits += 1;
                signal.wait_changed(observed, wait);
            }
        }
    }
}

// The manager is the sole control writer for its Job. Once cancellation is
// published, retain that exact sequence for the owner acknowledgement and
// terminal receipt; shutdown must not replace an in-flight cancellation.
fn close_cancel_needed<S>(controls: &rz_runtime::adaptive::ControlMailbox<S>) -> bool {
    let snapshot = controls.snapshot();
    !snapshot.terminal && !snapshot.is_cancelled()
}

/// A natural terminal/cancel racing with S-policy advice remains an owner
/// terminal to drain. It is not a failed search or an acknowledged Yield.
fn scheduling_yield<S>(
    controls: &rz_runtime::adaptive::ControlMailbox<S>,
    sequence: u64,
) -> Result<bool, AdaptiveManagerError> {
    match controls.request_yield(sequence) {
        Ok(()) => Ok(true),
        Err(rz_runtime::adaptive::ControlError::Closed) => Ok(false),
        Err(_) => Err(AdaptiveManagerError::Invalid("adaptive Yield control")),
    }
}

fn issue(next: &mut u64) -> Result<u64, AdaptiveManagerError> {
    let issued = *next;
    *next = next
        .checked_add(1)
        .ok_or(AdaptiveManagerError::Capacity("manager sequence exhausted"))?;
    Ok(issued)
}
fn validate_view(
    view: &ReviewView,
    seal: &ReviewViewSeal,
    authority: SearchAuthority,
    tokens: u16,
) -> Result<(), AdaptiveManagerError> {
    seal.validate(authority, tokens)?;
    if seal.view_sequence != view.sequence()
        || seal.view_digest != Digest(view.seal_digest())
        || seal.context_policy != Digest(view.policy_digest())
        || seal.actual_tokens != view.token_count()
        || usize::from(seal.selected_records) != view.records().len()
        || usize::from(seal.anchors) != view.anchors().len()
        || seal.length_bucket != view.length_bucket()
        || view.generation() != authority.root.0
    {
        return Err(AdaptiveManagerError::Invalid(
            "actual immutable review seal",
        ));
    }
    Ok(())
}
fn validate_root_ranking_view(
    view: &ReviewView,
    root: &ContractState,
) -> Result<(), AdaptiveManagerError> {
    let anchor = view
        .anchors()
        .first()
        .ok_or(AdaptiveManagerError::Invalid("root anchor absent"))?;
    if root.terminal_wdl().is_some()
        || !anchor.snapshot.same_state(root.rules().snapshot())
        || !view.query().prefix.is_empty()
        || !view.query().proposal.is_empty()
        || view.query().counterexample.is_some()
        || !view.query().divergences.is_empty()
    {
        return Err(AdaptiveManagerError::Invalid(
            "root ranking exact Rules/history",
        ));
    }
    Ok(())
}
fn validate_role_output(
    output: &RoleOutput,
    ticket: &RoleTicketEnvelope,
    view: &ReviewView,
) -> Result<(), AdaptiveManagerError> {
    if output.id != ticket.id
        || output.authority != ticket.authority
        || output.key != ticket.representation
        || output.private_latent.len() != 16 * 384
        || output.private_latent.iter().any(|value| !value.is_finite())
    {
        return Err(AdaptiveManagerError::Invalid(
            "native output identity/latent",
        ));
    }
    let valid = match (&output.payload, ticket.role) {
        (
            RolePayload::Proposal {
                candidate_logits,
                wdl,
            },
            Role::Proposer,
        ) => {
            candidate_logits.len() == view.query().legal.len()
                && candidate_logits.iter().all(|value| value.is_finite())
                && wdl.probabilities().iter().all(|value| value.is_finite())
        }
        (
            RolePayload::Counterexample {
                candidate_logits,
                divergence_logits,
                wdl,
            },
            Role::Critic,
        ) => {
            candidate_logits.len() == view.query().legal.len()
                && divergence_logits.len() == view.query().divergences.len()
                && candidate_logits
                    .iter()
                    .chain(divergence_logits)
                    .all(|value| value.is_finite())
                && wdl.probabilities().iter().all(|value| value.is_finite())
        }
        (RolePayload::TaskRanking { task_logits }, Role::Validator) => {
            task_logits.len() == 7 && task_logits.iter().all(|value| value.is_finite())
        }
        _ => false,
    };
    if !valid {
        return Err(AdaptiveManagerError::Invalid(
            "native legal-order/head/finite output",
        ));
    }
    Ok(())
}
fn role_cost_class(ticket: &RoleTicketEnvelope, device: Digest) -> CostClass {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-role-cost-class/1\0");
    hash.update([ticket.role as u8]);
    hash.update(ticket.representation.encoding.0);
    CostClass {
        model: ticket.representation.model.0,
        role_or_checker: hash.finalize().into(),
        input_bucket: ticket.view.length_bucket,
        invocation: InvocationClass::Fresh,
        cache: CacheClass::Unknown,
        device: device.0,
    }
}
fn observed_public_cache_class(
    physical: PhysicalRoleCompletion,
    evidence: TicketExecutionEvidence,
) -> CacheClass {
    if physical != PhysicalRoleCompletion::Confirmed
        || !evidence.measurement_complete
        || evidence.physical_fresh.is_none()
    {
        return CacheClass::Unknown;
    }
    let Some(accounting) = evidence.accounting else {
        return CacheClass::Unknown;
    };
    match (
        accounting.public_cache_hits(),
        accounting.public_cache_misses(),
    ) {
        (hits, 0) if hits > 0 => CacheClass::Hit,
        (0, misses) if misses > 0 => CacheClass::Miss,
        (hits, misses) if hits > 0 && misses > 0 => CacheClass::Mixed,
        _ => CacheClass::Unknown,
    }
}
/// Native NNUEC is the actual R owner in both product uses. Collection's
/// independent teacher does not replace that worker. Legacy PALS still uses
/// its existing Collection teacher/deployment researcher selection.
fn dispatch_cpu_endpoint(
    endpoints: CpuEndpointSelection,
    source: AdaptiveResearcherSourceKind,
    has_nnuec: bool,
) -> CpuEndpoint {
    if (has_nnuec && source == AdaptiveResearcherSourceKind::Native)
        || endpoints.execution_use == CpuExecutionUse::Deployment
    {
        endpoints.cpu_r
    } else {
        endpoints.cpu_t
    }
}

#[cfg(test)]
mod endpoint_selection_tests {
    use super::*;

    #[test]
    fn native_nnuec_keeps_r_while_legacy_collection_keeps_teacher_selection() {
        let mut collection = CpuEndpointSelection::deployment();
        collection.execution_use = CpuExecutionUse::Collection;
        assert_eq!(
            dispatch_cpu_endpoint(collection, AdaptiveResearcherSourceKind::Native, true),
            CpuEndpoint::Own,
        );
        collection.cpu_t = CpuEndpoint::ExternalUci;
        assert_eq!(
            dispatch_cpu_endpoint(collection, AdaptiveResearcherSourceKind::Native, false),
            CpuEndpoint::ExternalUci,
        );
        assert_eq!(
            dispatch_cpu_endpoint(
                CpuEndpointSelection::deployment(),
                AdaptiveResearcherSourceKind::Native,
                false,
            ),
            CpuEndpoint::Own,
        );
    }
}

#[cfg(test)]
mod observed_cache_tests {
    use super::*;
    use crate::pals_native::ticketed::TicketNativeAccountingEvidence;

    fn actual(hits: u64, misses: u64) -> TicketExecutionEvidence {
        TicketExecutionEvidence {
            execution: None,
            completed_nn_inputs: Some(1),
            measurement_complete: true,
            physical_fresh: Some(true),
            accounting: Some(TicketNativeAccountingEvidence {
                public_cache_hits: hits,
                public_cache_misses: misses,
                public_nn_runs_attempted: 0,
                public_nn_runs_completed: 0,
                public_nn_runs_failed_known: 0,
                role_nn_runs_attempted: 1,
                role_nn_runs_completed: 1,
                role_nn_runs_failed_known: 0,
                validator_nn_runs_attempted: 0,
                validator_nn_runs_completed: 0,
                validator_nn_runs_failed_known: 0,
            }),
        }
    }
    #[test]
    fn public_hit_with_private_nn_work_is_still_hit() {
        let confirmed = PhysicalRoleCompletion::Confirmed;
        assert_eq!(
            observed_public_cache_class(confirmed, actual(1, 0)),
            CacheClass::Hit
        );
        assert_eq!(
            observed_public_cache_class(confirmed, actual(0, 1)),
            CacheClass::Miss
        );
        assert_eq!(
            observed_public_cache_class(confirmed, actual(1, 1)),
            CacheClass::Mixed
        );
        assert_eq!(
            observed_public_cache_class(confirmed, actual(0, 0)),
            CacheClass::Unknown
        );
        let mut no_accounting = actual(1, 0);
        no_accounting.accounting = None;
        assert_eq!(
            observed_public_cache_class(confirmed, no_accounting),
            CacheClass::Unknown
        );
        let mut incomplete = actual(0, 1);
        incomplete.measurement_complete = false;
        assert_eq!(
            observed_public_cache_class(confirmed, incomplete),
            CacheClass::Unknown
        );
        assert_eq!(
            observed_public_cache_class(PhysicalRoleCompletion::Unknown, actual(0, 1)),
            CacheClass::Unknown
        );
    }
}
fn cpu_cost_class(namespace: Digest, bucket: u16, device: Digest) -> CostClass {
    CostClass {
        model: namespace.0,
        role_or_checker: namespace.0,
        input_bucket: bucket,
        invocation: InvocationClass::Fresh,
        cache: CacheClass::Miss,
        device: device.0,
    }
}
fn receipt_digest(domain: &[u8], execution: ExecutionId) -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-manager-event/1\0");
    hash.update(domain);
    hash.update(execution.epoch.0.to_le_bytes());
    hash.update(execution.sequence.to_le_bytes());
    Digest(hash.finalize().into())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
