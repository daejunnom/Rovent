//! Thin actual Search owner. Physical work remains on the manager's workers.
//! Archive maintenance transfers the complete port rather than cloning Store.
use super::*;
use crate::pals_native::ticketed::TicketExecutionEvidence;
use rz_contracts::pals::{RoleOutput, SearchAuthority};
use rz_contracts::pals_adaptive::{
    CpuJobEnvelope, ExternalCpuJobEnvelope, ExternalSliceGrant, ReviewViewSeal, RoleTicketEnvelope,
    SliceGrant,
};
use rz_contracts::{ClockDomain, Digest, MonotonicTick};
use rz_position::contracts::ContractState;
use rz_runtime::CompletionSignal;
use rz_search::cpu_worker::{
    CpuAdmissionRejection, CpuJobTermination, CpuSliceReport, CpuWorkerProfile, CpuWorkerProgress,
    ReleasedCpuReservation, SettledCpuSlice,
};
use rz_search::external_cpu::worker::{
    ExternalJob, ExternalTeacherDispatch, ExternalWorkerCompletion, ExternalWorkerReport,
};
use rz_search::external_cpu::{
    CpuEndpoint, CpuEndpointSelection, CpuExecutionUse, ExternalCpuConfig, ExternalSlicePolicy,
};
use rz_search::pals::coordinator::{
    CheckedExternalCpuBinding, ClosedFrame, CoordinatorWork, FrameBudget, FrameId, FrameInput,
    ProposerRankingPublished, StepExecution, ValidatorActionAdmission,
};
use rz_search::pals::engine::adaptive_port::archive_worker::{
    ArchiveMaintenanceStamp, ArchiveOperation, ArchiveOwner, ArchiveQuarantine, ArchiveTransfer,
    ArchiveWork, ArchiveWorkReceipt, ArchiveWorker,
};
use rz_search::pals::engine::adaptive_port::validator_port::{
    AdaptiveValidatorAction, AdaptiveValidatorActionPlan, AdaptiveValidatorQuestion,
    ValidatorActionId, ValidatorActionOutcome,
};
use rz_search::pals::engine::adaptive_port::{
    AdaptiveDeferredRepair, AdaptiveEnginePort, AdaptiveFrontierReceipt, AdaptiveFrontierStartup,
    AdaptivePortMaterializedStep, AdaptivePortQuestion, AdaptivePortRoleEvidence,
    AdaptiveReviewLeaseAdmission, AdaptiveRootFanoutPolicy, AdaptiveRulesPairProof,
};
use rz_search::pals::engine::{PalsCompletion, PalsError, PalsResult, RoleModel};
use rz_search::pals::review_context::ReviewView;
use rz_search::pals::rovent_frontier::{
    FrontierFrameKind, FrontierLimits, FrontierRegistration, NativeCloseReason, NativeReservation,
};
use rz_search::pals::store::{ForeignAppendPermit, ObservationId, StoreError, StorePins};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};
mod profiled_external;
use profiled_external::ProfiledDriverState;

const MAX_RETAINED_QUESTIONS: usize = 64;
const MAX_ARCHIVE_RECEIPTS: usize = 64;

mod role_only;
pub use role_only::PcPortDriver;

/// A coherent iteration publication is scoped to its original question and
/// physical owner. It is not CPU credit settlement or a new semantic witness.
struct PublishedCpuFeedback {
    frame: FrameId,
    view: ReviewViewSeal,
    public_context: Digest,
    job: CpuJobEnvelope,
    grant: SliceGrant,
    sequence: u64,
    progress: Option<CpuWorkerProgress>,
}

impl PublishedCpuFeedback {
    fn matches_work<Q>(&self, work: &CoordinatorWork<ReviewView, Q>) -> bool {
        self.frame == work.frame
            && self.job.question == work.input.question_digest
            && self.view == work.input.view_seal
            && self.public_context == work.input.public_context
            && self.job.hard_deadline == work.original_deadline
    }
    fn publish(&mut self, progress: &CpuWorkerProgress) -> Result<(), AdaptiveManagerError> {
        if progress.job != self.job.id
            || progress.execution != self.grant.execution
            || progress.owner != self.grant.owner
            || progress.sequence <= self.sequence
        {
            return Err(AdaptiveManagerError::Invalid(
                "original CPU progress authority",
            ));
        }
        self.sequence = progress.sequence;
        self.progress = Some(progress.clone());
        Ok(())
    }
}

/// The observer registers the original immutable input before native Execute.
/// The optional V question is the Search-issued support plan, not caller masks.
pub type AdaptiveCollectionDispatchObserver = Arc<
    dyn Fn(
            &RoleTicketEnvelope,
            &FrameInput<ReviewView, AdaptivePortQuestion>,
            Option<&AdaptiveValidatorQuestion>,
        ) -> Result<(), AdaptiveManagerError>
        + Send
        + Sync,
>;

/// Actual external owner evidence. The submitted view is immutable and remains
/// identified as the CPU question; it is never relabelled as a native output.
/// Collection explicitly joins an earlier sealed P/C input by exact endpoint.
// Keep the original bounded job and completion inline at the observer seam;
// boxing would add allocation around the sealed dispatch/receipt owners.
#[allow(clippy::large_enum_variant)]
pub enum AdaptiveExternalTeacherEvent {
    Submitted {
        input: FrameInput<ReviewView, AdaptivePortQuestion>,
        original: ExternalJob,
        dispatch: ExternalTeacherDispatch,
    },
    Completed {
        dispatch: ExternalTeacherDispatch,
        completion: ExternalWorkerCompletion,
        eligibility: AdaptiveTeacherTargetEligibility,
    },
}
pub type AdaptiveExternalTeacherObserver =
    Arc<dyn Fn(AdaptiveExternalTeacherEvent) -> Result<(), AdaptiveManagerError> + Send + Sync>;

pub struct AdaptivePortDriver<M: RoleModel + 'static> {
    // Keep actual and quarantined engine owners on the heap so moving this
    // lifecycle owner does not reserve several complete engines on the stack.
    port: Option<Box<AdaptiveEnginePort<M>>>,
    rules: Arc<ContractState>,
    authority: SearchAuthority,
    hard_deadline: Instant,
    archive: Option<ArchiveWorker<M>>,
    transfer: Option<ArchiveTransfer>,
    foreign_transfer_question: Option<Digest>,
    foreign_permit: Option<(Digest, ForeignAppendPermit)>,
    profiled: ProfiledDriverState,
    archive_receipts: VecDeque<ArchiveWorkReceipt>,
    quarantine: Option<Box<ArchiveQuarantine<M>>>,
    unexpected_archive_owner: Option<Box<ArchiveOwner<M>>>,
    closing: bool,
    archive_close_requested: bool,
    archive_joined: bool,
    external_policy: ExternalSlicePolicy,
    // A raw receipt is committed once. Building its next continuation may
    // require another owner turn and must not duplicate TaskTable settlement.
    cpu_committed: HashMap<Digest, (ObservationId, bool)>,
    cpu_feedback: HashMap<Digest, PublishedCpuFeedback>,
    foreign_committed: HashMap<Digest, (ObservationId, bool)>,
    validator_questions: HashMap<Digest, AdaptiveValidatorQuestion>,
    validator_seen: HashSet<Digest>,
    action_frames: HashMap<FrameId, ValidatorActionId>,
    action_outcomes: VecDeque<ValidatorActionOutcome>,
    collection_observer: Option<AdaptiveCollectionDispatchObserver>,
    external_teacher_observer: Option<AdaptiveExternalTeacherObserver>,
    maintenance_stamp: Option<ArchiveMaintenanceStamp>,
    registered_action: Option<AdaptiveValidatorAction>,
}
impl<M: RoleModel + 'static> AdaptivePortDriver<M> {
    /// `port` must already have its exact root and model/checker metadata.
    /// Creating the archive owner starts only a bounded private command loop.
    pub fn new(
        port: AdaptiveEnginePort<M>,
        hard_deadline: Instant,
        signal: Arc<CompletionSignal>,
        external_policy: ExternalSlicePolicy,
    ) -> Result<Self, AdaptiveManagerError> {
        Self::new_with_coordinator_policy(
            port,
            hard_deadline,
            signal,
            external_policy,
            rz_runtime::adaptive::CoordinatorPolicy::Fixed,
        )
    }
    pub fn new_with_coordinator_policy(
        port: AdaptiveEnginePort<M>,
        hard_deadline: Instant,
        signal: Arc<CompletionSignal>,
        external_policy: ExternalSlicePolicy,
        policy: rz_runtime::adaptive::CoordinatorPolicy,
    ) -> Result<Self, AdaptiveManagerError> {
        Self::new_with_coordinator_policy_retaining(
            port,
            hard_deadline,
            signal,
            external_policy,
            policy,
        )
        .map_err(|(_, error)| error)
    }
    // Failure returns the original port and its session/Store owners intact;
    // an error box would add allocation to this ownership-preserving rollback.
    #[allow(clippy::result_large_err)]
    pub fn new_with_coordinator_policy_retaining(
        mut port: AdaptiveEnginePort<M>,
        hard_deadline: Instant,
        signal: Arc<CompletionSignal>,
        external_policy: ExternalSlicePolicy,
        policy: rz_runtime::adaptive::CoordinatorPolicy,
    ) -> Result<Self, (AdaptiveEnginePort<M>, AdaptiveManagerError)> {
        if Instant::now() >= hard_deadline {
            return Err((port, AdaptiveManagerError::Deadline));
        }
        if let Err(error) = port.configure_root_fanout(match policy {
            rz_runtime::adaptive::CoordinatorPolicy::Adaptive => {
                AdaptiveRootFanoutPolicy::SimultaneousAdaptive
            }
            _ => AdaptiveRootFanoutPolicy::SequentialFixed,
        }) {
            return Err((port, semantic(error)));
        }
        let rules = port.root_rules();
        let authority = port.authority();
        let archive = if port.engine().archive_enabled() {
            match ArchiveWorker::start_with_notifier(move || signal.notify()) {
                Ok(worker) => Some(worker),
                Err(error) => {
                    return Err((
                        port,
                        AdaptiveManagerError::Semantic(format!("archive owner spawn: {error}")),
                    ));
                }
            }
        } else {
            None
        };
        let archive_joined = archive.is_none();
        Ok(Self {
            port: Some(Box::new(port)),
            rules,
            authority,
            hard_deadline,
            archive,
            transfer: None,
            foreign_transfer_question: None,
            foreign_permit: None,
            profiled: ProfiledDriverState::default(),
            archive_receipts: VecDeque::with_capacity(MAX_ARCHIVE_RECEIPTS),
            quarantine: None,
            unexpected_archive_owner: None,
            closing: false,
            archive_close_requested: false,
            archive_joined,
            external_policy,
            cpu_committed: HashMap::new(),
            cpu_feedback: HashMap::new(),
            foreign_committed: HashMap::new(),
            validator_questions: HashMap::new(),
            validator_seen: HashSet::new(),
            action_frames: HashMap::new(),
            action_outcomes: VecDeque::new(),
            collection_observer: None,
            external_teacher_observer: None,
            maintenance_stamp: None,
            registered_action: None,
        })
    }
    #[cfg(feature = "pals-adaptive-diagnostics")]
    pub(super) fn install_role_consumption_diagnostic(
        &mut self,
        diagnostic: &ContinuationDiagnostic,
    ) {
        if !diagnostic.options().role_consumption {
            return;
        }
        let authority = self.authority;
        let installed = self
            .hot()
            .is_ok_and(|port| port.enable_role_consumption_observation());
        diagnostic.role_install(authority, installed);
    }
    #[cfg(feature = "pals-adaptive-diagnostics")]
    pub(super) fn copy_role_consumption_diagnostic(
        &self,
        diagnostic: &ContinuationDiagnostic,
        original_root_fence: bool,
    ) {
        diagnostic.role_readback(
            self.authority,
            self.port()
                .and_then(|port| port.role_consumption_observation()),
            original_root_fence,
        );
        diagnostic.profiled_feedback_readback(
            self.authority,
            self.port()
                .map(|port| port.profiled_feedback_consumption_observation()),
        );
    }
    pub fn port(&self) -> Option<&AdaptiveEnginePort<M>> {
        self.port.as_deref()
    }
    pub fn attach_rovent_frontier(
        &mut self,
        registration: FrontierRegistration,
        limits: FrontierLimits,
        actual: AdaptiveFrontierStartup<'_>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .attach_rovent_frontier(registration, limits, actual, clock, now)?;
        Ok(())
    }
    /// Evidence recovery does not retire the controller or release its pins.
    pub fn rovent_frontier_receipt(
        &self,
    ) -> Result<Option<AdaptiveFrontierReceipt>, AdaptiveManagerError> {
        Ok(self
            .port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .rovent_frontier_receipt())
    }
    pub fn retire_rovent_frontier(
        &mut self,
    ) -> Result<Option<AdaptiveFrontierReceipt>, AdaptiveManagerError> {
        Ok(self.hot()?.retire_rovent_frontier()?)
    }
    pub fn set_collection_dispatch_observer(
        &mut self,
        observer: AdaptiveCollectionDispatchObserver,
    ) -> Result<(), AdaptiveManagerError> {
        if self.collection_observer.is_some() || self.closing {
            return Err(AdaptiveManagerError::Invalid(
                "collection observer installation",
            ));
        }
        self.collection_observer = Some(observer);
        Ok(())
    }
    pub fn root_input(
        &mut self,
    ) -> Result<FrameInput<ReviewView, AdaptivePortQuestion>, AdaptiveManagerError> {
        self.hot()?.root_input().map_err(semantic)
    }
    pub fn set_external_teacher_observer(
        &mut self,
        observer: AdaptiveExternalTeacherObserver,
    ) -> Result<(), AdaptiveManagerError> {
        if self.external_teacher_observer.is_some() || self.closing {
            return Err(AdaptiveManagerError::Invalid(
                "teacher observer installation",
            ));
        }
        self.external_teacher_observer = Some(observer);
        Ok(())
    }
    pub fn prepare_root_rules(&mut self) -> Result<(), AdaptiveManagerError> {
        self.hot()?.prepare_root_rules().map_err(semantic)
    }
    pub fn prepare_rovent_root_rules(
        &mut self,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?.prepare_rovent_root_rules(clock, now)?;
        Ok(())
    }
    pub fn root_state_id(&self) -> Result<rz_search::pals::store::StateId, AdaptiveManagerError> {
        Ok(self
            .port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .root_state_id())
    }
    pub fn model_namespace(
        &self,
    ) -> Result<&rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace, AdaptiveManagerError>
    {
        self.port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .model_namespace()
            .map_err(semantic)
    }
    pub fn root_is_terminal(&self) -> Result<bool, AdaptiveManagerError> {
        Ok(self
            .port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .root_is_terminal())
    }
    pub fn result(&self, completion: PalsCompletion) -> Result<PalsResult, AdaptiveManagerError> {
        self.port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .result(completion)
            .map_err(semantic)
    }
    pub fn archive_receipts(&self) -> impl Iterator<Item = &ArchiveWorkReceipt> {
        self.archive_receipts.iter()
    }
    pub fn take_action_outcome(&mut self) -> Option<ValidatorActionOutcome> {
        self.action_outcomes.pop_front()
    }
    /// Inspect original semantic results before a collector's validation can
    /// consume them. A copied observation grants no action-completion authority.
    pub fn action_outcomes(&self) -> impl Iterator<Item = &ValidatorActionOutcome> {
        self.action_outcomes.iter()
    }
    /// Retained semantic data never grants a physical completion on its own.
    pub fn complete_action(
        &self,
        outcome: ValidatorActionOutcome,
        receipt: &AdaptiveShutdownReceipt,
    ) -> Result<CompletedValidatorAction, AdaptiveManagerError> {
        CompletedValidatorAction::after_manager_shutdown(outcome, receipt)
    }
    // A failed closure returns this complete lifecycle owner with every
    // session/worker/quarantine intact, without an extra error-box allocation.
    #[allow(clippy::result_large_err)]
    pub fn into_port(
        mut self,
        receipt: &AdaptiveShutdownReceipt,
    ) -> Result<AdaptiveEnginePort<M>, (Self, AdaptiveManagerError)> {
        if !receipt.permits_semantic_closure()
            || self.profiled.source.is_some()
            || receipt.authority != self.authority
            || !self.archive_joined
            || self.transfer.is_some()
            || self.quarantine.is_some()
            || self.unexpected_archive_owner.is_some()
        {
            return Err((
                self,
                AdaptiveManagerError::Invalid("Search owner requires actual shutdown fence"),
            ));
        }
        match self.port.take() {
            Some(port) => Ok(*port),
            None => Err((self, AdaptiveManagerError::PhysicalUnknown)),
        }
    }
    fn hot(&mut self) -> Result<&mut AdaptiveEnginePort<M>, AdaptiveManagerError> {
        if self.quarantine.is_some() {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        self.port
            .as_deref_mut()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)
    }
    /// Called with an owned immutable view's actual incoming pins. Existing
    /// active pins remain sealed by ArchiveWorker; no all-Store clone occurs.
    pub fn request_reclaim(
        &mut self,
        incoming_pins: StorePins,
    ) -> Result<(), AdaptiveManagerError> {
        self.transfer(ArchiveOperation::ReclaimHot { incoming_pins }, None)
    }
    fn transfer(
        &mut self,
        operation: ArchiveOperation,
        question: Option<Digest>,
    ) -> Result<(), AdaptiveManagerError> {
        if self.archive_close_requested
            || self.transfer.is_some()
            || self.foreign_permit.is_some()
            || self.profiled.permit.is_some()
            || (self.closing
                && !matches!(
                    &operation,
                    ArchiveOperation::PrepareForeign { .. }
                        | ArchiveOperation::PrepareProfiledForeign { .. }
                ))
        {
            return Err(AdaptiveManagerError::AuxiliaryPending);
        }
        let port = self
            .port
            .take()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?;
        let Some(worker) = &mut self.archive else {
            self.port = Some(port);
            return Err(AdaptiveManagerError::Invalid(
                "archive operation without declared archive owner",
            ));
        };
        let work = ArchiveWork {
            deadline: match &operation {
                ArchiveOperation::PrepareForeign { issued, .. } => {
                    self.hard_deadline.min(issued.retain_until())
                }
                ArchiveOperation::PrepareProfiledForeign { issued, .. } => {
                    self.hard_deadline.min(issued.retain_until())
                }
                _ => self.hard_deadline,
            },
            max_io_bytes: port.archive_remaining_io().min(16 * 1024 * 1024),
            operation,
        };
        let profiled = matches!(
            &work.operation,
            ArchiveOperation::PrepareProfiledForeign { .. }
        );
        match worker.submit(ArchiveOwner::Port(*port), self.authority, work) {
            Ok(transfer) => {
                self.transfer = Some(transfer);
                if profiled {
                    self.profiled.transfer_question = question;
                } else {
                    self.foreign_transfer_question = question;
                }
                Ok(())
            }
            Err(rejected) => {
                match rejected.owner {
                    ArchiveOwner::Port(port) => self.port = Some(Box::new(port)),
                    owner @ (ArchiveOwner::Engine(_) | ArchiveOwner::Idle(_)) => {
                        self.unexpected_archive_owner = Some(Box::new(owner));
                    }
                }
                Err(semantic(rejected.error))
            }
        }
    }
    fn foreign_raw(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        receipt: &ExternalWorkerReport,
    ) -> Result<(ObservationId, bool), AdaptiveManagerError> {
        let key = work.input.question_digest;
        if let Some(committed) = self.foreign_committed.get(&key) {
            return Ok(*committed);
        }
        if self.foreign_committed.len() >= MAX_RETAINED_QUESTIONS {
            return Err(AdaptiveManagerError::Capacity(
                "foreign committed questions",
            ));
        }
        let permit = match self.foreign_permit.take() {
            Some((question, permit)) if question == key => Some(permit),
            Some(other) => {
                self.foreign_permit = Some(other);
                return Err(AdaptiveManagerError::AuxiliaryPending);
            }
            None => None,
        };
        let result = match permit {
            Some(permit) => self.hot()?.commit_external_permitted(work, receipt, permit),
            None => self.hot()?.commit_external_report(work, receipt),
        };
        match result {
            Ok(committed) => {
                self.foreign_committed.insert(key, committed);
                Ok(committed)
            }
            Err(PalsError::Store(rz_search::pals::store::StoreError::ArchiveBudgetRequired)) => {
                let publication = self
                    .hot()?
                    .prepare_external_publication(work, receipt)
                    .map_err(semantic)?;
                self.transfer(
                    ArchiveOperation::PrepareForeign {
                        observation: publication.observation,
                        issued: publication.issued,
                    },
                    Some(key),
                )?;
                Err(AdaptiveManagerError::AuxiliaryPending)
            }
            Err(error) => Err(semantic(error)),
        }
    }
    fn action_before(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        execution: StepExecution,
    ) -> Result<
        Option<rz_search::pals::engine::adaptive_port::validator_port::ActionCommitGuard>,
        AdaptiveManagerError,
    > {
        let Some(action) = self.action_frames.get(&work.frame).copied() else {
            return Ok(None);
        };
        self.hot()?
            .begin_validator_action_commit(action, work, execution)
            .map(Some)
            .map_err(semantic)
    }
    fn action_after(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        guard: Option<rz_search::pals::engine::adaptive_port::validator_port::ActionCommitGuard>,
    ) -> Result<(), AdaptiveManagerError> {
        if let Some(guard) = guard {
            self.hot()?
                .finish_validator_action_commit(guard, work)
                .map_err(semantic)?;
        }
        Ok(())
    }
}
fn semantic(error: PalsError) -> AdaptiveManagerError {
    match error {
        PalsError::Rovent(error) => AdaptiveManagerError::Frontier(error),
        PalsError::Store(StoreError::ReviewLeaseBackpressure {
            current,
            capacity,
            required_new,
            replacement_reserve,
        }) => AdaptiveManagerError::ReviewLeaseBackpressure {
            current,
            capacity,
            required_new,
            replacement_reserve,
        },
        other => AdaptiveManagerError::Semantic(other.to_string()),
    }
}
fn nnuec_preparation(
    error: rz_search::pals::engine::adaptive_port::AdaptiveNnuecPreparationError,
) -> AdaptiveManagerError {
    use rz_search::pals::engine::adaptive_port::AdaptiveNnuecPreparationError as Error;
    match error {
        Error::Semantic(error) => semantic(error),
        Error::Preparation(error) => {
            AdaptiveManagerError::NnuecPreparation(NnuecWorkError::Preparation(error))
        }
        Error::Worker(error) => AdaptiveManagerError::Cpu(error),
        Error::StateMismatch => AdaptiveManagerError::NnuecPreparation(NnuecWorkError::Invalid(
            "NNUEC original question/model/owner mismatch",
        )),
        Error::Neural(error) => {
            AdaptiveManagerError::Semantic(format!("NNUEC preparation: {error:?}"))
        }
    }
}
fn checked(step: AdaptivePortMaterializedStep) -> CheckedStep<AdaptivePortQuestion> {
    CheckedStep {
        evidence: step.evidence,
        transition: step.transition,
        fresh_pair: step.fresh_pair,
        rules_pair: step.rules_pair,
        queued_recheck: step.queued_recheck,
    }
}
impl<M: RoleModel + 'static> AdaptiveSearchDriver<AdaptivePortQuestion> for AdaptivePortDriver<M> {
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
        Ok(self
            .port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .rovent_frontier()
            .map(|frontier| {
                let snapshot = frontier.snapshot();
                (snapshot.requested_nodes, snapshot.work)
            }))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn freeze_utility_witness_bodies(
        &self,
        outcome: &rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteSemanticOutcome,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::utility_route::UtilityConditionalWitnessBodies,
        AdaptiveManagerError,
    > {
        self.port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .freeze_utility_witness_bodies(outcome)
            .map_err(semantic)
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn utility_cohort_offer_semantics(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::UtilityCohortOfferSemanticSource,
        AdaptiveManagerError,
    > {
        self.port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .utility_cohort_offer_semantics(work)
            .map_err(semantic)
    }
    fn prepare_utility_route(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        target: u16,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::utility_route::UtilityRoutePreparation,
        AdaptiveManagerError,
    > {
        self.port
            .as_deref_mut()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .prepare_utility_route(work, target)
            .map_err(semantic)
    }
    fn prepare_utility_route_input_request(
        &self,
        prep: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRoutePreparation,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        model: &rz_search::cpu_value::nnuec::v2::Model,
        canceled: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<rz_search::cpu_value::nnuec::v2::review::Request, AdaptiveManagerError> {
        self.port
            .as_deref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .prepare_utility_route_input_request(prep, work, model, canceled)
            .map_err(|error| AdaptiveManagerError::Semantic(format!("{error:?}")))
    }
    fn bind_utility_intervention_target(
        &self,
        prep: rz_search::pals::engine::adaptive_port::utility_route::UtilityRoutePreparation,
        source: rz_search::pals::coordinator::UtilityInterventionReadySource<AdaptivePortQuestion>,
        input: Arc<rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture>,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::utility_route::UtilityInterventionTarget,
        (
            AdaptiveManagerError,
            rz_search::pals::engine::adaptive_port::utility_route::UtilityRoutePreparation,
            rz_search::pals::coordinator::UtilityInterventionReadySource<AdaptivePortQuestion>,
            Arc<rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture>,
        ),
    > {
        let Some(port) = self.port.as_deref() else {
            return Err((AdaptiveManagerError::AuxiliaryPending, prep, source, input));
        };
        port.bind_utility_intervention_target(prep, source, input)
            .map_err(|r| (semantic(r.error), r.preparation, r.source, r.input))
    }
    fn prepare_utility_intervention(
        &mut self,
        target: rz_search::pals::engine::adaptive_port::utility_route::UtilityInterventionTarget,
        recipient: rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteRecipient,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::utility_route::UtilityInterventionCandidate,
        (
            AdaptiveManagerError,
            rz_search::pals::engine::adaptive_port::utility_route::UtilityInterventionTarget,
        ),
    > {
        let Some(port) = self.port.as_deref_mut() else {
            return Err((AdaptiveManagerError::AuxiliaryPending, target));
        };
        port.prepare_utility_intervention(target, recipient)
            .map_err(|(e, t)| (semantic(e), t))
    }
    fn bind_utility_intervention_route(
        &self,
        admission:rz_search::pals::engine::adaptive_port::utility_route::UtilityInterventionAdmission,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        (
            AdaptiveManagerError,
            rz_search::pals::engine::adaptive_port::utility_route::UtilityInterventionAdmission,
        ),
    > {
        let Some(port) = self.port.as_deref() else {
            return Err((AdaptiveManagerError::AuxiliaryPending, admission));
        };
        port.bind_utility_intervention_route(admission, work)
            .map_err(|r| (semantic(r.error), r.admission))
    }
    fn prepare_utility_intervention_arm_request(
        &self,
        action: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        model: &rz_search::cpu_value::nnuec::v2::Model,
        cancel: Arc<AtomicBool>,
    ) -> Result<rz_search::cpu_value::nnuec::v2::review::Request, AdaptiveManagerError> {
        self.port
            .as_deref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .prepare_utility_intervention_arm_request(action, work, model, cancel)
            .map_err(|e| AdaptiveManagerError::Semantic(format!("{e:?}")))
    }
    fn attach_utility_intervention_arm_request(
        &self,
        action: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        request: rz_search::cpu_value::nnuec::v2::review::Request,
    ) -> Result<
        rz_search::cpu_value::nnuec::v2::review::Request,
        (
            AdaptiveManagerError,
            rz_search::cpu_value::nnuec::v2::review::Request,
        ),
    > {
        let Some(port) = self.port.as_deref() else {
            return Err((AdaptiveManagerError::AuxiliaryPending, request));
        };
        port.attach_utility_intervention_arm_request(action, work, request)
            .map_err(|(e, r)| (AdaptiveManagerError::Semantic(format!("{e:?}")), r))
    }
    fn bind_utility_intervention_arm_capture(
        &self,
        action: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        input: Arc<rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture>,
    ) -> Result<
        (),
        (
            AdaptiveManagerError,
            Arc<rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture>,
        ),
    > {
        let Some(port) = self.port.as_deref() else {
            return Err((AdaptiveManagerError::AuxiliaryPending, input));
        };
        port.bind_utility_intervention_arm_capture(action, work, input)
            .map_err(|(e, i)| (semantic(e), i))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn utility_intervention_arm_metadata(
        &self,
        action: &rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<serde_json::Value, AdaptiveManagerError> {
        super::utility_intervention_historical::arm_query(action, work)
    }
    fn bind_utility_route_input(
        &self,
        prep: rz_search::pals::engine::adaptive_port::utility_route::UtilityRoutePreparation,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        input: Arc<rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture>,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        (
            AdaptiveManagerError,
            rz_search::pals::engine::adaptive_port::utility_route::UtilityRoutePreparation,
            Arc<rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture>,
        ),
    > {
        let Some(port) = self.port.as_deref() else {
            return Err((AdaptiveManagerError::AuxiliaryPending, prep, input));
        };
        port.bind_utility_route_input(prep, work, input)
            .map_err(|rejected| {
                (
                    semantic(rejected.error),
                    rejected.preparation,
                    rejected.input,
                )
            })
    }
    fn validate_utility_route_work(
        &self,
        action: &rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .as_deref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .validate_utility_route_work(action, work)
            .map_err(semantic)
    }
    fn begin_utility_route_commit(
        &self,
        action: &rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        execution: StepExecution,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteCommitGuard,
        AdaptiveManagerError,
    > {
        self.port
            .as_deref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .begin_utility_route_commit(action, work, execution)
            .map_err(semantic)
    }
    fn register_utility_route_descendant(
        &self,
        action: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .as_deref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .register_utility_route_descendant(action, work)
            .map_err(semantic)
    }
    fn register_utility_route_peer(
        &self,
        action: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        proof: &rz_search::pals::coordinator::ReciprocalDependencyProof<
            ReviewView,
            AdaptivePortQuestion,
        >,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .as_deref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .register_utility_route_peer(action, work, proof)
            .map_err(semantic)
    }
    fn commit_utility_route_step(
        &self,
        action: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        guard: &rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteCommitGuard,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .as_deref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .commit_utility_route_step(action, guard, work)
            .map_err(semantic)
    }
    fn commit_utility_route_await(
        &self,
        action: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .as_deref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .commit_utility_route_await(action, work, report)
            .map_err(semantic)
    }
    fn commit_utility_route_role(
        &self,
        action: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        guard: &rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteCommitGuard,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .as_deref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .commit_utility_route_role(action, guard, work, accepted, output)
            .map_err(semantic)
    }
    fn record_utility_route_r_pc_consumption(
        &self,
        action: &mut rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteAction,
        witness: &rz_search::pals::engine::adaptive_port::CpuSemanticWitness,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .as_deref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .record_utility_route_r_pc_consumption(action, witness)
            .map_err(semantic)
    }
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
        let Some(port) = self.port.as_deref() else {
            return Err((AdaptiveManagerError::AuxiliaryPending, action));
        };
        port.finish_utility_route_semantics(action)
            .map_err(|(error, action)| (semantic(error), action))
    }
    fn reserve_cpu_nnuec_capture(
        &self,
        seal: &rz_search::pals::engine::adaptive_port::AdaptiveNnuecReviewSeal,
        model: Arc<rz_search::cpu_value::nnuec::v2::Model>,
        bound: Arc<rz_search::cpu_value::nnuec::v2::review::BoundNnuecReviewContext>,
        sequence: u64,
        limits: rz_search::cpu_value::nnuec::v2::capture::NnuecCaptureLimits,
    ) -> Result<
        rz_search::cpu_value::nnuec::v2::capture::NnuecCaptureReservation,
        AdaptiveManagerError,
    > {
        self.port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .reserve_cpu_nnuec_capture(seal, model, bound, sequence, limits)
            .map_err(|error| AdaptiveManagerError::NnuecCapture(Arc::new(error)))
    }
    fn cpu_semantic_publication_recorded(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
    ) -> bool {
        self.port
            .as_deref()
            .is_some_and(|port| port.cpu_semantic_publication_recorded(work, report))
    }
    fn cpu_semantic_witnesses(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
        settled: &SettledCpuSlice,
        capture: &rz_search::cpu_value::nnuec::v2::capture::NnuecCaptureLineage,
    ) -> Result<Vec<rz_search::pals::engine::adaptive_port::CpuSemanticWitness>, AdaptiveManagerError>
    {
        self.port
            .as_deref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .cpu_semantic_witnesses(work, report, settled, capture)
            .map_err(semantic)
    }
    fn selected_cpu_pc_consumption(
        &self,
        source: &rz_search::pals::engine::adaptive_port::CpuSemanticWitness,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
        accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
        output: &RoleOutput,
    ) -> Result<
        Option<rz_search::pals::engine::adaptive_port::CpuSemanticWitness>,
        AdaptiveManagerError,
    > {
        self.port
            .as_deref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .selected_cpu_pc_consumption(source, input, accepted, output)
            .map_err(semantic)
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn validate_profiled_external_registration(
        &self,
        source: &rz_search::external_cpu::ProfiledExternalRootSource,
        root: &Arc<crate::rovent_root_admission::RoventRootAdmission>,
    ) -> Result<(), AdaptiveManagerError> {
        self.profile_validate(source, root)
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn validate_profiled_external_registration_v2(
        &self,
        source: &rz_search::external_cpu::ProfiledExternalRootSource,
        root: &Arc<crate::rovent_root_admission::source_seal::RoventRootAdmissionV2>,
    ) -> Result<(), AdaptiveManagerError> {
        self.profile_validate_v2(source, root)
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn profiled_external_work(
        &self,
    ) -> Option<rz_search::pals::rovent_frontier::FrontierExternalWork> {
        self.profile_work()
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn profiled_retained_host_bytes(&self) -> Option<u64> {
        self.profile_retained_host_bytes()
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn prepare_profiled_external(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        config: &ExternalCpuConfig,
        source: &rz_search::external_cpu::ProfiledExternalRootSource,
        id: rz_contracts::rovent::ReciprocalQuestionId,
        execution: rz_contracts::ExecutionId,
        constraint: rz_search::external_cpu::ExternalGoConstraint,
        quantum: Duration,
        signal: CompletionSignal,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::AdaptiveProfiledExternalPreparation,
        AdaptiveManagerError,
    > {
        self.profile_prepare(
            work, config, source, id, execution, constraint, quantum, signal, clock, now,
        )
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn profiled_external_admitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &rz_search::external_cpu::ProfiledExternalRootSource,
        dispatch: &rz_search::external_cpu::ProfiledExternalDispatch,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::AdaptiveProfiledExternalAdmission,
        AdaptiveManagerError,
    > {
        self.profile_admitted(work, source, dispatch, clock, now)
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn commit_profiled_external(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &rz_search::external_cpu::ProfiledExternalRootSource,
        reservation: rz_search::pals::rovent_frontier::FrontierExternalReservation,
        packet: &Arc<rz_search::external_cpu::ProfiledExternalCompletion>,
    ) -> Result<CheckedStep<AdaptivePortQuestion>, AdaptiveManagerError> {
        self.profile_commit(work, source, reservation, packet)
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn observe_profiled_external(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &rz_search::external_cpu::ProfiledExternalRootSource,
        reservation: rz_search::pals::rovent_frontier::FrontierExternalReservation,
        packet: &Arc<rz_search::external_cpu::ProfiledExternalCompletion>,
    ) -> Result<(), AdaptiveManagerError> {
        self.profile_observe(work, source, reservation, packet)
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn consume_profiled_reciprocal_cpu(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &rz_search::external_cpu::ProfiledExternalRootSource,
        reservation: rz_search::pals::rovent_frontier::FrontierExternalReservation,
        packet: &Arc<rz_search::external_cpu::ProfiledExternalCompletion>,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::CheckedProfiledExternalResponse,
        AdaptiveManagerError,
    > {
        self.profile_consume_reciprocal(work, source, reservation, packet)
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn reject_profiled_external(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), AdaptiveManagerError> {
        self.profile_reject(work)
    }
    fn review_lease_admission(
        &self,
    ) -> Result<Option<AdaptiveReviewLeaseAdmission>, AdaptiveManagerError> {
        let observed = self
            .port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .review_lease_admission()
            .map_err(semantic)?;
        Ok(observed.frontier_owner_present.then_some(observed))
    }
    fn frontier_native_capacity(
        &mut self,
        requested: u64,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Option<u64>, AdaptiveManagerError> {
        self.hot()?
            .with_rovent_frontier(|frontier, _| frontier.native_capacity(requested, clock, now))
            .map_err(Into::into)
    }
    fn frontier_native_reserve(
        &mut self,
        job: &CpuJobEnvelope,
        grant: SliceGrant,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Option<NativeReservation>, AdaptiveManagerError> {
        self.hot()?
            .with_rovent_frontier(|frontier, _| frontier.reserve_native(job, grant, clock, now))
            .map_err(Into::into)
    }
    fn frontier_native_settled(
        &mut self,
        reservation: &NativeReservation,
        report: &CpuSliceReport,
        accepted: &SettledCpuSlice,
    ) -> Result<NativeCloseReason, AdaptiveManagerError> {
        self.hot()?
            .with_rovent_frontier(|frontier, _| {
                frontier.settle_native(reservation, report, accepted)
            })?
            .ok_or(AdaptiveManagerError::Invalid(
                "native frontier owner absent",
            ))
    }
    fn frontier_native_released(
        &mut self,
        reservation: &NativeReservation,
        accepted: &ReleasedCpuReservation,
    ) -> Result<NativeCloseReason, AdaptiveManagerError> {
        self.hot()?
            .with_rovent_frontier(|frontier, _| frontier.release_unqueued(reservation, accepted))?
            .ok_or(AdaptiveManagerError::Invalid(
                "native frontier owner absent",
            ))
    }
    fn frontier_native_unknown(
        &mut self,
        reservation: &NativeReservation,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .with_rovent_frontier(|frontier, _| frontier.native_usage_unknown(reservation))?
            .ok_or(AdaptiveManagerError::Invalid(
                "native frontier owner absent",
            ))
    }
    fn register_cpu_reciprocal_scope(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &CpuJobEnvelope,
        question: &rz_contracts::rovent::QuestionEnvelope,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .register_cpu_reciprocal_scope(work, job, question)
            .map_err(semantic)
    }
    fn prepare_accepted_role_reciprocal(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
        output: &RoleOutput,
        id: rz_contracts::rovent::ReciprocalQuestionId,
        mode: rz_contracts::rovent::ExecutionMode,
        budget: Digest,
        recipient: rz_contracts::rovent::SearchRecipient,
    ) -> Result<Option<AdaptivePeerQuestion<AdaptivePortQuestion>>, AdaptiveManagerError> {
        Ok(self
            .hot()?
            .prepare_accepted_role_reciprocal(work, accepted, output, id, mode, budget, recipient)
            .map_err(semantic)?
            .map(|prepared| AdaptivePeerQuestion {
                origin: prepared.origin,
                question: prepared.question,
                input: prepared.input,
            }))
    }
    fn accept_accepted_role_source(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .accept_accepted_role_source(work, accepted, output)
            .map_err(semantic)
    }
    #[cfg(feature = "pals-adaptive-diagnostics")]
    fn diagnostic_accepted_source_revision(
        &self,
        accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
    ) -> Option<u64> {
        // Observation only: do not invoke hot(), mutate archive ownership or
        // turn absent/failed publication evidence into a manager error.
        if self.quarantine.is_some() {
            return None;
        }
        self.port
            .as_deref()?
            .diagnostic_accepted_source_revision(accepted)
    }
    fn consume_accepted_role_reciprocal(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
        output: &RoleOutput,
        physical: Option<TicketExecutionEvidence>,
    ) -> Result<Digest, AdaptiveManagerError> {
        let physical = physical.map(|evidence| AdaptivePortRoleEvidence {
            execution: evidence.execution,
            completed_nn_inputs: evidence.completed_nn_inputs,
            measurement_complete: evidence.measurement_complete,
            physical_fresh: evidence.physical_fresh,
        });
        self.hot()?
            .consume_accepted_role_reciprocal(work, accepted, output, physical)
            .map_err(semantic)
    }
    fn consume_reciprocal_cpu(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
    ) -> Result<Digest, AdaptiveManagerError> {
        self.hot()?
            .consume_reciprocal_cpu(work, report)
            .map_err(semantic)
    }
    fn discard_accepted_role_reciprocal(
        &mut self,
        id: rz_contracts::rovent::ReciprocalQuestionId,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?.discard_accepted_role_reciprocal(id);
        Ok(())
    }
    fn reciprocal_parent(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &CpuJobEnvelope,
        id: rz_contracts::rovent::ReciprocalQuestionId,
        mode: rz_contracts::rovent::ExecutionMode,
        root_budget: Digest,
    ) -> Result<rz_contracts::rovent::QuestionEnvelope, AdaptiveManagerError> {
        self.hot()?
            .cpu_reciprocal_parent(work, job, id, mode, root_budget)
            .map_err(semantic)
    }
    fn prepare_cpu_reciprocal(
        &mut self,
        parent: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        request: &rz_search::cpu::CpuReciprocalRequest,
        id: rz_contracts::rovent::ReciprocalQuestionId,
        mode: rz_contracts::rovent::ExecutionMode,
        root_budget: Digest,
    ) -> Result<AdaptiveReciprocalQuestion<AdaptivePortQuestion>, AdaptiveManagerError> {
        let prepared = self
            .hot()?
            .prepare_cpu_reciprocal(parent, request, id, mode, root_budget)
            .map_err(semantic)?;
        Ok(AdaptiveReciprocalQuestion {
            question: prepared.question,
            input: prepared.input,
        })
    }
    fn cpu_reciprocal_wait(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .cpu_reciprocal_wait(work, report)
            .map_err(semantic)
    }
    fn discard_cpu_reciprocal(
        &mut self,
        id: rz_contracts::rovent::ReciprocalQuestionId,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?.discard_cpu_reciprocal(id);
        Ok(())
    }
    fn consume_cpu_reciprocal(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
        output: &RoleOutput,
        physical: Option<TicketExecutionEvidence>,
        requests: &[(
            rz_search::cpu::CpuReciprocalRequest,
            rz_contracts::rovent::ReciprocalQuestionId,
        )],
    ) -> Result<Vec<rz_search::cpu::CpuReciprocalReply>, AdaptiveManagerError> {
        let physical = physical.map(|evidence| AdaptivePortRoleEvidence {
            execution: evidence.execution,
            completed_nn_inputs: evidence.completed_nn_inputs,
            measurement_complete: evidence.measurement_complete,
            physical_fresh: evidence.physical_fresh,
        });
        let answer = self
            .hot()?
            .consume_cpu_reciprocal(work, accepted, output, physical)
            .map_err(semantic)?;
        requests
            .iter()
            .map(|(request, id)| {
                self.hot()?
                    .cpu_reciprocal_reply(&answer, request, *id)
                    .map_err(semantic)
            })
            .collect()
    }
    fn root_rules(&self) -> Arc<ContractState> {
        Arc::clone(&self.rules)
    }
    fn validate_cpu_nnuec_model(
        &self,
        model: &rz_search::cpu_value::nnuec::v2::Model,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .validate_cpu_nnuec_model(model)
            .map_err(nnuec_preparation)
    }
    fn prepare_cpu_nnuec_review(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        model: &rz_search::cpu_value::nnuec::v2::Model,
        canceled: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::AdaptiveNnuecReviewRequest,
        AdaptiveManagerError,
    > {
        self.port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .prepare_cpu_nnuec_review(work, model, canceled)
            .map_err(nnuec_preparation)
    }
    fn install_cpu_nnuec_review(
        &self,
        seal: &rz_search::pals::engine::adaptive_port::AdaptiveNnuecReviewSeal,
        model: Arc<rz_search::cpu_value::nnuec::v2::Model>,
        bound: Arc<rz_search::cpu_value::nnuec::v2::review::BoundNnuecReviewContext>,
    ) -> Result<rz_search::cpu_worker::PreparedValueInstallation, AdaptiveManagerError> {
        self.port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .install_cpu_nnuec_review(seal, model, bound)
            .map_err(nnuec_preparation)
    }
    fn accept_cpu_nnuec_installation(
        &mut self,
        seal: &rz_search::pals::engine::adaptive_port::AdaptiveNnuecReviewSeal,
        model: Arc<rz_search::cpu_value::nnuec::v2::Model>,
        bound: Arc<rz_search::cpu_value::nnuec::v2::review::BoundNnuecReviewContext>,
        receipt: &rz_search::cpu_worker::ValueInstallReceipt,
        worker: &rz_search::cpu_worker::CpuWorker,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .accept_cpu_nnuec_installation(seal, model, bound, receipt, worker)
            .map_err(nnuec_preparation)
    }
    fn release_cpu_nnuec_installation(
        &mut self,
        seal: &rz_search::pals::engine::adaptive_port::AdaptiveNnuecReviewSeal,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .release_cpu_nnuec_installation(seal)
            .map_err(nnuec_preparation)
    }
    fn next_rovent_goal_input(
        &mut self,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<
        rz_search::pals::engine::adaptive_port::AdaptiveGoalLocalAdmission,
        AdaptiveManagerError,
    > {
        self.hot()?
            .next_rovent_goal_input(clock, now)
            .map_err(semantic)
    }
    fn validate_rovent_goal_input(
        &self,
        input: &rz_search::pals::engine::adaptive_port::GoalLocalInput,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .validate_rovent_goal_input(input, ticket, output)
            .map_err(semantic)
    }
    fn consume_rovent_goal_input(
        &mut self,
        input: &rz_search::pals::engine::adaptive_port::GoalLocalInput,
        accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
        output: &RoleOutput,
        physical: Option<TicketExecutionEvidence>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<rz_search::pals::engine::adaptive_port::GoalLocalStep, AdaptiveManagerError> {
        let evidence = physical.ok_or(AdaptiveManagerError::Invalid(
            "GoalLocal native execution evidence absent",
        ))?;
        let physical = AdaptivePortRoleEvidence {
            execution: evidence.execution,
            completed_nn_inputs: evidence.completed_nn_inputs,
            measurement_complete: evidence.measurement_complete,
            physical_fresh: evidence.physical_fresh,
        };
        self.hot()?
            .consume_rovent_goal_input(input, accepted, output, physical, clock, now)
            .map_err(semantic)
    }
    fn discard_rovent_goal_input(
        &mut self,
        input: &rz_search::pals::engine::adaptive_port::GoalLocalInput,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .discard_rovent_goal_input(input)
            .map_err(semantic)
    }
    fn rovent_goal_has_pending(&self) -> Result<bool, AdaptiveManagerError> {
        Ok(self
            .port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .rovent_goal_has_pending())
    }
    fn close_rovent_goal_admission(&mut self) -> Result<(), AdaptiveManagerError> {
        self.hot()?.close_rovent_goal_admission().map_err(semantic)
    }
    fn prepare_rovent_goal_boundary(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<
        Option<rz_search::pals::engine::adaptive_port::AdaptiveGoalBoundary<AdaptivePortQuestion>>,
        AdaptiveManagerError,
    > {
        self.hot()?
            .prepare_rovent_goal_boundary(work)
            .map_err(semantic)
    }
    fn release_rovent_goal_state(&mut self) -> Result<(), AdaptiveManagerError> {
        self.hot()?.release_rovent_goal_state().map_err(semantic)
    }
    fn rovent_goal_consumption_recorded(
        &self,
        ticket: &RoleTicketEnvelope,
        view: &ReviewView,
    ) -> Result<bool, AdaptiveManagerError> {
        Ok(self
            .port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .rovent_goal_consumption_recorded(ticket, view))
    }
    fn validate_collection_admission(
        &self,
        admission: &RoventCollectionRootAdmission,
    ) -> Result<(), AdaptiveManagerError> {
        let frontier = self
            .port
            .as_ref()
            .and_then(|port| port.rovent_frontier())
            .ok_or(AdaptiveManagerError::Invalid(
                "collection requires the actual attached Rovent frontier",
            ))?;
        let original = frontier.registration();
        if !admission.validates_registration(
            self.authority,
            original.mode,
            original.root_budget,
            original.original_deadline,
        ) || admission.selective_policy() != original.selective_policy
            || !Arc::ptr_eq(admission.rules(), &original.root_rules)
            || !Arc::ptr_eq(admission.root_moves(), &original.root_moves)
        {
            return Err(AdaptiveManagerError::Invalid(
                "collection cap differs from the actual PCR registration",
            ));
        }
        Ok(())
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn validate_collection_admission_v2(
        &self,
        admission: &crate::pals_adaptive::RoventCollectionRootAdmissionV2,
    ) -> Result<(), AdaptiveManagerError> {
        let frontier = self
            .port
            .as_ref()
            .and_then(|port| port.rovent_frontier())
            .ok_or(AdaptiveManagerError::Invalid(
                "V2 collection requires the actual attached Rovent frontier",
            ))?;
        admission.validate_frontier(frontier)?;
        let original = frontier.registration();
        let root = admission.root_admission();
        let namespace = admission
            .native_origin()
            .startup_snapshot()
            .model_namespace()
            .map_err(|_| AdaptiveManagerError::Invalid("V2 original Ready namespace"))?;
        let active = root
            .original_context()
            .accepts()
            .map_err(|_| AdaptiveManagerError::Invalid("V2 collection root unavailable"))?;
        if !active
            || root.authority() != self.authority
            || root.authority() != original.authority
            || root.hard_deadline() != original.original_deadline
            || root.goal() != original.goal
            || root.requested_nodes() != original.nodes
            || admission.root_budget() != original.root_budget
            || admission.selective_policy() != original.selective_policy
            || admission
                .native_origin()
                .sources()
                .lock()
                .manifest
                .engine
                .mode
                != original.mode
            || original.model_namespace.as_ref() != &namespace
            || !Arc::ptr_eq(admission.rules(), &original.root_rules)
            || !Arc::ptr_eq(admission.root_moves(), &original.root_moves)
        {
            return Err(AdaptiveManagerError::Invalid(
                "V2 collection cap differs from the actual PCR registration",
            ));
        }
        Ok(())
    }
    fn rovent_root_budget(&self) -> Result<Option<Digest>, AdaptiveManagerError> {
        Ok(self
            .port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .rovent_frontier()
            .map(|frontier| frontier.registration().root_budget))
    }
    fn rovent_public_record_capacity_closed(&self) -> Result<bool, AdaptiveManagerError> {
        let port = self
            .port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?;
        let Some(frontier) = port.rovent_frontier() else {
            return Ok(false);
        };
        if frontier.registration().authority != self.authority {
            return Err(AdaptiveManagerError::Invalid(
                "capacity controller differs from the actual driver root",
            ));
        }
        Ok(frontier.public_record_capacity_refused())
    }
    fn close_rovent_role_call_budget(
        &mut self,
        admitted: u64,
        limit: u64,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .close_rovent_role_call_budget(admitted, limit)
            .map_err(semantic)
    }
    fn rovent_public_review_index_capacity_closed(&self) -> Result<bool, AdaptiveManagerError> {
        let port = self
            .port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?;
        let Some(frontier) = port.rovent_frontier() else {
            return Ok(false);
        };
        if frontier.registration().authority != self.authority {
            return Err(AdaptiveManagerError::Invalid(
                "review capacity controller differs from the actual driver root",
            ));
        }
        Ok(frontier.public_review_index_capacity_refused())
    }
    fn rovent_lane_selected(
        &mut self,
        original: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        selected: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<bool, AdaptiveManagerError> {
        self.hot()?
            .rovent_lane_selected(original, selected, clock, now)
            .map_err(semantic)
    }
    fn rovent_frame_admitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        kind: FrontierFrameKind,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<bool, AdaptiveManagerError> {
        self.hot()?
            .rovent_frame_admitted(work, kind, clock, now)
            .map_err(AdaptiveManagerError::Frontier)
    }
    fn rovent_fresh_accepted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
        output: &RoleOutput,
        physical: Option<TicketExecutionEvidence>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<bool, AdaptiveManagerError> {
        let evidence = physical.ok_or(AdaptiveManagerError::Invalid(
            "Fresh endpoint lacks original native execution evidence",
        ))?;
        let physical = AdaptivePortRoleEvidence {
            execution: evidence.execution,
            completed_nn_inputs: evidence.completed_nn_inputs,
            measurement_complete: evidence.measurement_complete,
            physical_fresh: evidence.physical_fresh,
        };
        self.hot()?
            .rovent_fresh_accepted(work, accepted, output, physical, clock, now)
            .map_err(semantic)
    }
    fn rovent_rules_endpoint(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<bool, AdaptiveManagerError> {
        self.hot()?
            .rovent_rules_endpoint(work, clock, now)
            .map_err(AdaptiveManagerError::Frontier)
    }
    fn rovent_root_round_sealed(
        &mut self,
        root_budget: Digest,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<bool, AdaptiveManagerError> {
        self.hot()?
            .rovent_root_round_sealed(root_budget, clock, now)
            .map_err(AdaptiveManagerError::Frontier)
    }
    fn validate_registered_native_ledger(
        &self,
        identity: &rz_search::cpu_worker::CpuLedgerIdentity,
        observed: &rz_search::cpu_worker::CpuLedgerRegistrationObservation,
        root_budget: Digest,
        hard_deadline: Instant,
    ) -> Result<(), AdaptiveManagerError> {
        let port = self
            .port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?;
        let frontier = port.rovent_frontier().ok_or(AdaptiveManagerError::Invalid(
            "original ledger requires the actual attached frontier",
        ))?;
        let registration = frontier.registration();
        let work = frontier.snapshot().work;
        let clock = rz_search::contract_time::InstantClock::new(
            observed.clock_domain,
            observed.clock_origin,
        );
        if registration.authority != self.authority
            || observed.authority != self.authority
            || registration.original_deadline.clock != observed.clock_domain
            || clock
                .instant_at(registration.original_deadline.at)
                .map_err(|_| AdaptiveManagerError::Deadline)?
                != hard_deadline
            || self.hard_deadline != hard_deadline
            || registration.root_budget != root_budget
            || registration.native_ledger.as_ref() != Some(identity)
            || !Arc::ptr_eq(&registration.root_rules, &self.rules)
            || !work.native_usage_known
            || work.native_nodes != observed.spent_native_nodes
            || work.reserved_native_nodes != observed.reserved_native_nodes
            || work.outstanding_native_executions != observed.outstanding_executions
        {
            return Err(AdaptiveManagerError::Invalid(
                "original native ledger differs from its actual root frontier",
            ));
        }
        Ok(())
    }
    fn root_role_call_limit(&self) -> Result<Option<u64>, AdaptiveManagerError> {
        Ok(Some(
            self.port
                .as_ref()
                .ok_or(AdaptiveManagerError::AuxiliaryPending)?
                .role_call_limit(),
        ))
    }
    fn collection_before_dispatch(
        &mut self,
        ticket: &RoleTicketEnvelope,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), AdaptiveManagerError> {
        if let Some(observer) = &self.collection_observer {
            let validator = self.validator_questions.get(&input.question_digest);
            if ticket.role == rz_contracts::pals::Role::Validator && validator.is_none() {
                return Err(AdaptiveManagerError::Invalid(
                    "V capture lacks actual prepared support question",
                ));
            }
            observer(ticket, input, validator)?;
        }
        Ok(())
    }
    fn poll_auxiliary(&mut self) -> Result<bool, AdaptiveManagerError> {
        let mut changed = false;
        if let (Some(worker), Some(transfer)) = (&mut self.archive, &self.transfer) {
            if let Some(completed) = worker.try_receive(transfer).map_err(|error| {
                AdaptiveManagerError::Semantic(format!("archive receive: {error:?}"))
            })? {
                let nonce = transfer.nonce();
                let foreign = self.foreign_transfer_question.take();
                let profiled = self.profiled.transfer_question.take();
                self.transfer = None;
                let installed = if let Some(question) = profiled {
                    completed
                        .install_with_profiled_foreign(self.authority, nonce)
                        .map(|(owner, receipt, permit)| {
                            self.profiled.permit = Some((question, permit));
                            (owner, receipt)
                        })
                } else if let Some(question) = foreign {
                    completed.install_with_foreign(self.authority, nonce).map(
                        |(owner, receipt, permit)| {
                            self.foreign_permit = Some((question, permit));
                            (owner, receipt)
                        },
                    )
                } else {
                    completed.install(self.authority, nonce)
                };
                match installed {
                    Ok((ArchiveOwner::Port(port), receipt)) => {
                        self.port = Some(Box::new(port));
                        if self.archive_receipts.len() == MAX_ARCHIVE_RECEIPTS {
                            self.archive_receipts.pop_front();
                        }
                        self.archive_receipts.push_back(receipt);
                    }
                    Ok((owner @ (ArchiveOwner::Engine(_) | ArchiveOwner::Idle(_)), _)) => {
                        self.unexpected_archive_owner = Some(Box::new(owner));
                        return Err(AdaptiveManagerError::Invalid(
                            "archive returned a different semantic owner kind",
                        ));
                    }
                    Err(quarantine) => {
                        self.quarantine = Some(Box::new(quarantine));
                        return Err(AdaptiveManagerError::PhysicalUnknown);
                    }
                }
                changed = true;
            }
        }
        if self.archive_close_requested && self.transfer.is_none() && !self.archive_joined {
            if let Some(worker) = &mut self.archive {
                worker.request_shutdown();
                if let Some(receipt) = worker.try_join().map_err(|error| {
                    AdaptiveManagerError::Semantic(format!("archive join: {error:?}"))
                })? {
                    if !receipt.admission_closed
                        || !receipt.worker_joined
                        || !receipt.owners_returned
                    {
                        return Err(AdaptiveManagerError::PhysicalUnknown);
                    }
                    self.archive_joined = true;
                    changed = true;
                }
            }
        }
        Ok(changed)
    }
    fn semantic_owner_available(&self) -> bool {
        self.port.is_some() && self.quarantine.is_none() && self.unexpected_archive_owner.is_none()
    }
    fn prepare_semantic_commit(&mut self) -> Result<bool, AdaptiveManagerError> {
        if !self.semantic_owner_available() {
            return Ok(false);
        }
        if self.closing || self.archive.is_none() {
            return Ok(true);
        }
        let proposal = self
            .hot()?
            .archive_maintenance_proposal()
            .map_err(semantic)?;
        let Some(proposal) = proposal else {
            return Ok(true);
        };
        if self.maintenance_stamp == Some(proposal.stamp) {
            // A fully pinned/no-change owner does not retry the same I/O each
            // manager tick. A real Store commit/revision changes the stamp.
            return Ok(true);
        }
        let stamp = proposal.stamp;
        self.transfer(
            ArchiveOperation::ReclaimHot {
                incoming_pins: proposal.incoming_pins,
            },
            None,
        )?;
        self.maintenance_stamp = Some(stamp);
        Ok(false)
    }
    fn begin_auxiliary_shutdown(&mut self, _: Instant) -> Result<(), AdaptiveManagerError> {
        self.closing = true;
        Ok(())
    }
    fn finish_auxiliary_shutdown(&mut self) -> Result<(), AdaptiveManagerError> {
        if self.transfer.is_some()
            || self.foreign_permit.is_some()
            || self.profiled.permit.is_some()
            || self.profiled.transfer_question.is_some()
        {
            return Err(AdaptiveManagerError::AuxiliaryPending);
        }
        self.archive_close_requested = true;
        if let Some(worker) = &mut self.archive {
            worker.request_shutdown();
        }
        Ok(())
    }
    fn auxiliary_shutdown_confirmed(&self) -> bool {
        self.archive_joined
            && self.transfer.is_none()
            && self.port.is_some()
            && self.quarantine.is_none()
            && self.unexpected_archive_owner.is_none()
            && self.port.as_ref().is_none_or(|port| {
                port.rovent_frontier_receipt().is_none_or(|receipt| {
                    receipt.snapshot.work.native_usage_known
                        && receipt.snapshot.work.reserved_native_nodes == 0
                        && receipt.snapshot.work.outstanding_native_executions == 0
                })
            })
    }
    fn short_circuit(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<Option<CheckedRulesStep<AdaptivePortQuestion>>, AdaptiveManagerError> {
        let Some(step) = self.hot()?.short_circuit(work).map_err(semantic)? else {
            return Ok(None);
        };
        let materialized = self
            .hot()?
            .materialize_step(step.checked)
            .map_err(semantic)?;
        Ok(Some(CheckedRulesStep {
            endpoint: step.endpoint,
            checked: checked(materialized),
        }))
    }
    fn ranking_published(
        &mut self,
        ranking: &ProposerRankingPublished,
    ) -> Result<Vec<CandidateFrame<AdaptivePortQuestion>>, AdaptiveManagerError> {
        self.hot()?
            .ranking_published(ranking)
            .map(|candidates| {
                candidates
                    .into_iter()
                    .map(|candidate| CandidateFrame {
                        first_move: candidate.first_move,
                        input: candidate.input,
                        budget: candidate.budget,
                        deferred: candidate.deferred,
                        initialized: candidate.initialized,
                    })
                    .collect()
            })
            .map_err(semantic)
    }
    fn root_candidate_admitted(
        &mut self,
        frame: FrameId,
        first: rz_contracts::pals::Move16,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .root_candidate_admitted(frame, first, input)
            .map_err(semantic)
    }
    fn next_root_round_input(
        &mut self,
    ) -> Result<Option<FrameInput<ReviewView, AdaptivePortQuestion>>, AdaptiveManagerError> {
        self.hot()?.next_root_round_input().map_err(semantic)
    }
    fn next_fixed_root_candidate(
        &mut self,
    ) -> Result<Option<CandidateFrame<AdaptivePortQuestion>>, AdaptiveManagerError> {
        self.hot()?
            .next_fixed_root_candidate()
            .map(|candidate| {
                candidate.map(|candidate| CandidateFrame {
                    first_move: candidate.first_move,
                    input: candidate.input,
                    budget: candidate.budget,
                    deferred: candidate.deferred,
                    initialized: candidate.initialized,
                })
            })
            .map_err(semantic)
    }
    fn next_fixed_iterative_repair(
        &mut self,
    ) -> Result<Option<AdaptiveDeferredRepair<AdaptivePortQuestion>>, AdaptiveManagerError> {
        self.hot()?.next_fixed_iterative_repair().map_err(semantic)
    }
    fn validate_role(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        self.port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .validate_role(work, ticket, output)
            .map_err(semantic)
    }
    fn commit_role(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        physical: Option<TicketExecutionEvidence>,
    ) -> Result<CheckedStep<AdaptivePortQuestion>, AdaptiveManagerError> {
        let guard = self.action_before(work, StepExecution::Role(ticket.id))?;
        let physical = physical.map(|evidence| AdaptivePortRoleEvidence {
            execution: evidence.execution,
            completed_nn_inputs: evidence.completed_nn_inputs,
            measurement_complete: evidence.measurement_complete,
            physical_fresh: evidence.physical_fresh,
        });
        let step = self
            .hot()?
            .commit_role(work, ticket, output, physical)
            .map_err(semantic)?;
        // Selection of the next immutable view follows consumption of the
        // exact view submitted for this accepted native ticket.
        self.hot()?
            .role_consumed(ticket, &work.input.view)
            .map_err(semantic)?;
        self.action_after(work, guard)?;
        let materialized = self.hot()?.materialize_step(step).map_err(semantic)?;
        Ok(checked(materialized))
    }
    fn role_consumed(
        &mut self,
        ticket: &RoleTicketEnvelope,
        view: &Arc<ReviewView>,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?.role_consumed(ticket, view).map_err(semantic)
    }
    fn cpu_question(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        profiles: &[CpuWorkerProfile],
        external: Option<&ExternalCpuConfig>,
        endpoint: CpuEndpoint,
        physical_budget_exhausted: bool,
    ) -> Result<CpuQuestion, AdaptiveManagerError> {
        match endpoint {
            CpuEndpoint::Own => {
                let port = self.hot()?;
                let question = port
                    .prepare_cpu_question(work, profiles, physical_budget_exhausted)
                    .map_err(|error| {
                        AdaptiveManagerError::Semantic(
                            port.cpu_question_failure_detail(work, &error),
                        )
                    })?;
                let Some(question) = question else {
                    let (semantic_limit, semantic_spent) = port.semantic_cpu_budget();
                    return Ok(CpuQuestion::OwnBudgetExhausted {
                        semantic_limit,
                        semantic_spent,
                    });
                };
                Ok(CpuQuestion::Own(OwnCpuQuestion {
                    position: question.position,
                    root_moves: question.root_moves,
                    target_depth: question.target_depth,
                    total_native_node_credit: question.total_native_node_credit,
                }))
            }
            CpuEndpoint::ExternalUci => {
                let config = external.ok_or(AdaptiveManagerError::Invalid(
                    "foreign endpoint lacks actual config",
                ))?;
                let policy = self.external_policy;
                let question = self
                    .hot()?
                    .external_question(work, config, policy)
                    .map_err(semantic)?;
                Ok(CpuQuestion::External(ForeignCpuQuestion {
                    position: question.position,
                    root_moves: question.root_moves,
                    target_depth: question.target_depth,
                    policy: question.policy,
                }))
            }
            CpuEndpoint::Disabled => {
                Err(AdaptiveManagerError::Invalid("no checker owner selected"))
            }
        }
    }
    fn cpu_admitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &CpuJobEnvelope,
        grant: &SliceGrant,
    ) -> Result<(), AdaptiveManagerError> {
        let key = work.input.question_digest;
        if let Some(original) = self.cpu_feedback.get(&key) {
            if !original.matches_work(work)
                || original.job != *job
                || original.grant.owner != grant.owner
            {
                return Err(AdaptiveManagerError::Invalid("CPU progress resume context"));
            }
        } else if self.cpu_feedback.len() >= MAX_RETAINED_QUESTIONS {
            return Err(AdaptiveManagerError::Capacity("CPU progress questions"));
        }
        self.hot()?
            .cpu_admitted(work, job, grant)
            .map_err(semantic)?;
        match self.cpu_feedback.get_mut(&key) {
            Some(original) => {
                original.grant = *grant;
                original.progress = None;
            }
            None => {
                self.cpu_feedback.insert(
                    key,
                    PublishedCpuFeedback {
                        frame: work.frame,
                        view: work.input.view_seal,
                        public_context: work.input.public_context,
                        job: *job,
                        grant: *grant,
                        sequence: 0,
                        progress: None,
                    },
                );
            }
        }
        Ok(())
    }
    fn cpu_progress_for(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        progress: &CpuWorkerProgress,
    ) -> Result<(), AdaptiveManagerError> {
        let original = self
            .cpu_feedback
            .get_mut(&work.input.question_digest)
            .ok_or(AdaptiveManagerError::Invalid(
                "CPU progress lacks original admission",
            ))?;
        if self.closing || work.frame.authority != self.authority || !original.matches_work(work) {
            return Err(AdaptiveManagerError::Invalid(
                "CPU progress question context",
            ));
        }
        original.publish(progress)
    }
    fn predict_yield_feedback(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        costs: &rz_runtime::adaptive::CostEstimator,
        class: rz_runtime::adaptive::CostClass,
        since_publication: Duration,
        lease_remaining: Duration,
    ) -> Option<rz_runtime::adaptive::FeedbackWaitPrediction> {
        if self.closing || work.frame.authority != self.authority {
            return None;
        }
        let original = self.cpu_feedback.get(&work.input.question_digest)?;
        if !original.matches_work(work) {
            return None;
        }
        let progress = original.progress.as_ref()?;
        if !self
            .port
            .as_ref()?
            .published_cpu_continuation_eligible(work, progress)
        {
            return None;
        }
        // This only forecasts waiting until eligible feedback can be consumed.
        // The model uses original completed owner timing, never CPU agreement,
        // depth as utility, a fabricated witness, or paused residence time.
        costs.feedback_wait_prediction(class, since_publication, lease_remaining)
    }
    fn cpu_progress_request_eligible(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &CpuJobEnvelope,
        grant: &SliceGrant,
    ) -> bool {
        !self.closing
            && work.frame.authority == self.authority
            && self
                .port
                .as_ref()
                .is_some_and(|port| port.cpu_progress_request_eligible(work, job, grant))
    }
    fn cpu_partial_continuation_eligible(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
    ) -> bool {
        !self.closing
            && work.frame.authority == self.authority
            && self
                .port
                .as_ref()
                .is_some_and(|port| port.yielded_cpu_continuation_eligible(work, report))
    }
    fn external_feedback_eligible(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        progress: &rz_search::external_cpu::worker::ExternalWorkerProgress,
    ) -> bool {
        !self.closing
            && work.frame.authority == self.authority
            && self
                .port
                .as_ref()
                .is_some_and(|port| port.published_external_continuation_eligible(work, progress))
    }
    fn cpu_detached_slice_observed(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        // The original CPU question remains the consumer. No after_cpu or
        // next-frame transition is issued for later receipts from this owner.
        self.cpu_slice_observed(work, report)?;
        if report.terminal {
            self.cpu_feedback.remove(&work.input.question_digest);
        }
        Ok(())
    }
    fn cpu_slice_observed(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        if let Some(original) = self.cpu_feedback.get_mut(&work.input.question_digest) {
            original.progress = None;
        }
        if self.closing {
            self.hot()?
                .observe_cpu_report(work, report)
                .map_err(semantic)
        } else {
            self.hot()?
                .commit_cpu_report(work, report)
                .map(|_| ())
                .map_err(semantic)
        }
    }
    fn cpu_no_value_terminated(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
    ) -> Result<Digest, AdaptiveManagerError> {
        // Unlike cpu_slice_observed, this remains observation-only while
        // Running, and cannot enter commit_cpu_report/after_cpu.
        let original = self
            .hot()?
            .observe_cpu_no_value_termination(work, report)
            .map_err(semantic)?;
        self.cpu_feedback.remove(&work.input.question_digest);
        Ok(original)
    }
    fn cpu_rejected(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &CpuJobEnvelope,
        proof: &CpuAdmissionRejection,
    ) -> Result<(), AdaptiveManagerError> {
        if !proof.is_verified_unqueued() {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        self.hot()?
            .reject_cpu_admission(work, job, proof)
            .map_err(semantic)?;
        self.cpu_feedback.remove(&work.input.question_digest);
        Ok(())
    }
    fn role_consumption_recorded(
        &self,
        ticket: &RoleTicketEnvelope,
        view: &ReviewView,
    ) -> Result<bool, AdaptiveManagerError> {
        self.port
            .as_ref()
            .ok_or(AdaptiveManagerError::AuxiliaryPending)?
            .role_consumption_recorded(ticket, view)
            .map_err(semantic)
    }
    fn cpu_token_discarded(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        termination: &CpuJobTermination,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .retire_cpu_token(work, termination)
            .map_err(semantic)?;
        self.cpu_feedback.remove(&work.input.question_digest);
        Ok(())
    }
    fn commit_cpu(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &CpuSliceReport,
    ) -> Result<CheckedStep<AdaptivePortQuestion>, AdaptiveManagerError> {
        let key = work.input.question_digest;
        if let Some(original) = self.cpu_feedback.get_mut(&key) {
            original.progress = None;
        }
        let guard = self.action_before(
            work,
            StepExecution::Cpu {
                job: report.grant.job,
                execution: report.grant.execution,
            },
        )?;
        let actual = if let Some(committed) = self.cpu_committed.get(&key) {
            *committed
        } else {
            if self.cpu_committed.len() >= MAX_RETAINED_QUESTIONS {
                return Err(AdaptiveManagerError::Capacity("CPU committed questions"));
            }
            let committed = self
                .hot()?
                .commit_cpu_report(work, report)
                .map_err(semantic)?;
            self.cpu_committed.insert(key, committed);
            committed
        };
        let step = self
            .hot()?
            .after_cpu(work, actual.0, actual.1)
            .map_err(semantic)?;
        self.action_after(work, guard)?;
        self.cpu_committed.remove(&key);
        let materialized = self.hot()?.materialize_step(step).map_err(semantic)?;
        self.cpu_feedback.remove(&key);
        Ok(checked(materialized))
    }
    fn external_admitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &ExternalCpuJobEnvelope,
        grant: &ExternalSliceGrant,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<CheckedExternalCpuBinding, AdaptiveManagerError> {
        self.hot()?
            .external_admitted(work, job, grant, clock, now)
            .map_err(semantic)
    }
    fn external_rejected(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        job: &ExternalCpuJobEnvelope,
        grant: &ExternalSliceGrant,
    ) -> Result<(), AdaptiveManagerError> {
        self.hot()?
            .external_rejected(work, job, grant)
            .map_err(semantic)
    }
    fn external_teacher_submitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        original: &ExternalJob,
        dispatch: &ExternalTeacherDispatch,
    ) -> Result<(), AdaptiveManagerError> {
        if !dispatch.matches_job(original)
            || original.envelope.id.authority != self.authority
            || work.input.view_seal.view_digest.0 != work.input.view.seal_digest()
            || work.input.view_seal.view_sequence != work.input.view.sequence()
        {
            return Err(AdaptiveManagerError::Invalid(
                "actual teacher submission scope",
            ));
        }
        if let Some(observer) = &self.external_teacher_observer {
            observer(AdaptiveExternalTeacherEvent::Submitted {
                input: work.input.clone(),
                original: original.clone(),
                dispatch: dispatch.clone(),
            })?;
        }
        Ok(())
    }
    fn external_teacher_completed(
        &mut self,
        dispatch: &ExternalTeacherDispatch,
        completion: &ExternalWorkerCompletion,
        eligibility: AdaptiveTeacherTargetEligibility,
    ) -> Result<(), AdaptiveManagerError> {
        if let Some(observer) = &self.external_teacher_observer {
            observer(AdaptiveExternalTeacherEvent::Completed {
                dispatch: dispatch.clone(),
                completion: completion.clone(),
                eligibility,
            })?;
        }
        Ok(())
    }
    fn external_report_observed(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &ExternalWorkerReport,
    ) -> Result<(), AdaptiveManagerError> {
        let key = work.input.question_digest;
        let permit = match self.foreign_permit.take() {
            Some((question, permit)) if question == key => Some(permit),
            Some(other) => {
                self.foreign_permit = Some(other);
                return Err(AdaptiveManagerError::AuxiliaryPending);
            }
            None => None,
        };
        let observed = match permit {
            Some(permit) => self.hot()?.observe_external_permitted(work, report, permit),
            None => self.hot()?.observe_external_report(work, report),
        };
        match observed {
            Ok(()) => Ok(()),
            Err(PalsError::Store(rz_search::pals::store::StoreError::ArchiveBudgetRequired)) => {
                let publication = self
                    .hot()?
                    .prepare_external_publication(work, report)
                    .map_err(semantic)?;
                self.transfer(
                    ArchiveOperation::PrepareForeign {
                        observation: publication.observation,
                        issued: publication.issued,
                    },
                    Some(key),
                )?;
                Err(AdaptiveManagerError::AuxiliaryPending)
            }
            Err(error) => Err(semantic(error)),
        }
    }
    fn commit_external(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        report: &ExternalWorkerReport,
    ) -> Result<CheckedStep<AdaptivePortQuestion>, AdaptiveManagerError> {
        let guard = self.action_before(
            work,
            StepExecution::ExternalCpu {
                job: report.job.id,
                execution: report.grant.execution,
            },
        )?;
        let actual = self.foreign_raw(work, report)?;
        let step = self
            .hot()?
            .after_cpu(work, actual.0, actual.1)
            .map_err(semantic)?;
        self.action_after(work, guard)?;
        self.foreign_committed.remove(&work.input.question_digest);
        let materialized = self.hot()?.materialize_step(step).map_err(semantic)?;
        Ok(checked(materialized))
    }
    fn publish_closed(&mut self, closed: &ClosedFrame) -> Result<(), AdaptiveManagerError> {
        self.hot()?.publish_closed(closed).map_err(semantic)?;
        if let Some(action) = self.action_frames.remove(&closed.id) {
            if closed.unknown.is_none() {
                let outcome = self
                    .hot()?
                    .finish_validator_action_semantics(action, Some(closed))
                    .map_err(semantic)?;
                if self.action_outcomes.len() >= MAX_RETAINED_QUESTIONS {
                    return Err(AdaptiveManagerError::Capacity("V action outcomes"));
                }
                self.action_outcomes.push_back(outcome);
            }
        }
        Ok(())
    }
    fn validate_rules_pair(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        proof: &AdaptiveRulesPairProof,
    ) -> Result<(), AdaptiveManagerError> {
        proof.validate(work).map_err(semantic)
    }
    fn validator_input(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        profiles: &[CpuWorkerProfile],
        selection: CpuEndpointSelection,
        budget: FrameBudget,
    ) -> Result<Option<FrameInput<ReviewView, AdaptivePortQuestion>>, AdaptiveManagerError> {
        if !selection.load_v || selection.execution_use != CpuExecutionUse::Collection {
            return Ok(None);
        }
        let review_key = self.hot()?.validator_review_key(work).map_err(semantic)?;
        if self.validator_seen.contains(&review_key) {
            return Ok(None);
        }
        if self.validator_seen.len() >= MAX_RETAINED_QUESTIONS {
            return Err(AdaptiveManagerError::Capacity("V root contexts"));
        }
        let question = self
            .hot()?
            .prepare_validator_question(work, budget, profiles, selection)
            .map_err(semantic)?;
        let input = question.input().clone();
        self.validator_seen.insert(review_key);
        self.validator_questions
            .insert(input.question_digest, question);
        Ok(Some(input))
    }
    fn validate_validator(
        &self,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        let question = self.validator_questions.get(&input.question_digest).ok_or(
            AdaptiveManagerError::Invalid("V result has no original support question"),
        )?;
        if question.input().view_seal != input.view_seal
            || ticket.view != input.view_seal
            || ticket.question != input.question_digest
            || ticket.authority != self.authority
            || output.id != ticket.id
            || output.authority != ticket.authority
            || output.key != ticket.representation
        {
            return Err(AdaptiveManagerError::Invalid(
                "V original support question or actual output",
            ));
        }
        if !matches!(
            output.payload,
            rz_contracts::pals::RolePayload::TaskRanking { .. }
        ) {
            return Err(AdaptiveManagerError::Invalid("V actual task head"));
        }
        Ok(())
    }
    fn validator_accepted(
        &mut self,
        input: &FrameInput<ReviewView, AdaptivePortQuestion>,
        proof: AcceptedValidator,
    ) -> Result<AdaptiveValidatorActionPlan, AdaptiveManagerError> {
        let question = self
            .validator_questions
            .remove(&input.question_digest)
            .ok_or(AdaptiveManagerError::Invalid(
                "accepted V lacks original support question",
            ))?;
        match self.registered_action {
            Some(action) => proof.plan_registered_action(self.hot()?, &question, action),
            None => proof.select(self.hot()?, &question),
        }
    }
    fn register_supported_validator_action(
        &mut self,
        action: AdaptiveValidatorAction,
    ) -> Result<(), AdaptiveManagerError> {
        if self.closing
            || self.registered_action.is_some()
            || !self.validator_seen.is_empty()
            || !self.action_frames.is_empty()
        {
            return Err(AdaptiveManagerError::Invalid(
                "registered action must precede V submission",
            ));
        }
        self.registered_action = Some(action);
        Ok(())
    }
    fn validator_action_admitted(
        &mut self,
        id: ValidatorActionId,
        admission: &ValidatorActionAdmission,
    ) -> Result<(), AdaptiveManagerError> {
        match admission {
            ValidatorActionAdmission::Frame(frame) | ValidatorActionAdmission::Resumed(frame) => {
                if self.action_frames.contains_key(frame) {
                    return Err(AdaptiveManagerError::Invalid(
                        "two action owners for one frame",
                    ));
                }
                self.hot()?
                    .bind_validator_action(id, *frame)
                    .map_err(semantic)?;
                self.action_frames.insert(*frame, id);
            }
            ValidatorActionAdmission::Defer => {
                let outcome = self
                    .hot()?
                    .finish_validator_action_semantics(id, None)
                    .map_err(semantic)?;
                if self.action_outcomes.len() >= MAX_RETAINED_QUESTIONS {
                    return Err(AdaptiveManagerError::Capacity("V action outcomes"));
                }
                self.action_outcomes.push_back(outcome);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod published_progress_tests {
    use super::*;
    use rz_contracts::pals_adaptive::{CpuJobId, CpuWorkerId};
    use rz_contracts::{
        Deadline, ExecutionId, GameGeneration, OwnerId, ProcessEpoch, RootGeneration,
        StateIdentity, StateRevision,
    };
    use rz_search::cpu::CpuIterationProgress;

    fn original() -> PublishedCpuFeedback {
        let authority = SearchAuthority {
            epoch: ProcessEpoch(3),
            game: GameGeneration(2),
            root: RootGeneration(4),
            implementation: Digest([1; 32]),
        };
        let state_identity = StateIdentity {
            owner: OwnerId(1),
            revision: StateRevision(5),
            semantic: Digest([2; 32]),
        };
        let job = CpuJobEnvelope {
            id: CpuJobId {
                authority,
                sequence: 1,
            },
            state_identity,
            question: Digest([3; 32]),
            root_restriction: Digest([4; 32]),
            evaluation_namespace: Digest([5; 32]),
            search_conditions: Digest([6; 32]),
            total_native_node_credit: 8192,
            hard_deadline: Deadline {
                clock: ClockDomain(authority.epoch),
                at: MonotonicTick(20_000_000),
            },
        };
        PublishedCpuFeedback {
            frame: FrameId {
                authority,
                sequence: 1,
            },
            view: ReviewViewSeal {
                authority,
                view_sequence: 1,
                context_policy: Digest([7; 32]),
                view_digest: Digest([8; 32]),
                root_state: state_identity,
                actual_tokens: 0,
                selected_records: 0,
                anchors: 1,
                length_bucket: 8,
            },
            public_context: Digest([9; 32]),
            job,
            grant: SliceGrant {
                job: job.id,
                execution: ExecutionId::new(authority.epoch, 1),
                owner: CpuWorkerId(0),
                node_credit: 4096,
                soft_lease: Duration::from_millis(8),
                control_sequence: 1,
            },
            sequence: 0,
            progress: None,
        }
    }
    #[test]
    fn published_progress_rejects_other_question_owner_and_old_slice() {
        let mut original = original();
        let progress = CpuWorkerProgress {
            job: original.job.id,
            execution: original.grant.execution,
            owner: original.grant.owner,
            sequence: 1,
            iteration: CpuIterationProgress {
                best_move: None,
                score: 100,
                completed_depth: 1,
                nodes: 4,
            },
        };
        for wrong in [
            CpuWorkerProgress {
                job: CpuJobId {
                    sequence: 2,
                    ..progress.job
                },
                ..progress.clone()
            },
            CpuWorkerProgress {
                owner: CpuWorkerId(1),
                ..progress.clone()
            },
            CpuWorkerProgress {
                execution: ExecutionId::new(progress.execution.epoch, 2),
                ..progress.clone()
            },
        ] {
            assert!(original.publish(&wrong).is_err());
            assert!(original.progress.is_none());
            assert_eq!(original.sequence, 0);
        }
        original.publish(&progress).unwrap();
        assert!(original.publish(&progress).is_err());
        assert_eq!(original.progress.as_ref().unwrap().iteration.nodes, 4);
        original.grant.execution.sequence = 2;
        original.progress = None;
        assert!(
            original
                .publish(&CpuWorkerProgress {
                    sequence: 2,
                    ..progress.clone()
                })
                .is_err()
        );
        original
            .publish(&CpuWorkerProgress {
                execution: original.grant.execution,
                sequence: 2,
                ..progress
            })
            .unwrap();
        // Publication did not change the immutable Job or settle any credit.
        assert_eq!(original.job.total_native_node_credit, 8192);
    }
}
