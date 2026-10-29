//! Selected V3 review inputs on the existing bounded ticket/physical worker.
//!
//! The immutable Search view, rather than the raw record store, is the only
//! input. Backend identities come from the loaded owner. Logical expiry never
//! releases a running lease, and only a manager ack can register a Warm seed.
use super::native_bridge::NativeTicketQuestion;
use super::service::*;
use crate::pals_native::{self, virtual_encoding::pals_virtual_rules_semantic_digest};
use rz_contracts::pals::{RepresentationKey, Role, RoleOutput, RolePayload};
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_contracts::{Digest, ExecutionId, PrecisionProfile, RequestId, Wdl};
use rz_eval::error::BackendError;
use rz_eval::pals_model::virtual_review::{
    self as model, PalsVirtualReviewInput, ReviewAnchorInput, ReviewExecutionIdentity,
};
use rz_eval::pals_model::{PalsModelConfig, PalsRawOutput, PalsRole};
use rz_eval::pals_onnx::virtual_review::{
    PalsVirtualReviewOnnxBackend, VirtualReviewCompletion, VirtualReviewInvocation,
    VirtualReviewNativeCommand, VirtualReviewNativeResult,
};
use rz_eval::pals_onnx::{PalsAppliedSessionSettings, VirtualReviewCudaPlacementWitness};
use rz_eval::worker::{PhysicalLease, PhysicalPoll, SingleWorker};
use rz_position::{BoardMove, Position};
use rz_runtime::Clock;
use rz_runtime::contracts::{ContractClock, ContractSystemClock};
use rz_search::pals::coordinator::{PhysicalRoleCompletion, RoleFailure};
use rz_search::pals::review_context::ReviewView;
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::Poll;
use std::time::{Duration, Instant};

type CommandResult = Result<VirtualReviewNativeResult, BackendError>;
type NativeWorker = SingleWorker<VirtualReviewNativeCommand, CommandResult>;
type NativeLease = PhysicalLease<VirtualReviewNativeCommand, CommandResult>;

pub struct VirtualReviewPreparedTicket {
    pub input: PalsVirtualReviewInput,
    pub identity: ReviewExecutionIdentity,
    pub representation: RepresentationKey,
}
pub struct VirtualReviewTicketOutput {
    pub output: RoleOutput,
    pub evidence: TicketExecutionEvidence,
    // Opaque backend-issued authority. Callers cannot construct a seed by
    // hashing an arbitrary latent or accepting another ticket's completion.
    completion: VirtualReviewCompletion,
}
impl VirtualReviewTicketOutput {
    /// Raw, physically completed numbers; the opaque acceptance token stays private.
    pub fn raw_output(&self) -> &PalsRawOutput {
        self.completion.raw_output()
    }
    pub fn physical_invocation(&self) -> VirtualReviewPhysicalInvocationEvidence {
        VirtualReviewPhysicalInvocationEvidence {
            invocation_key: Digest(self.completion.invocation_key),
            accepted_seed_digest: self.completion.accepted_seed_digest.map(Digest),
            physical_fresh: !self.completion.warm_started,
        }
    }
}
impl rz_search::pals::coordinator::ProposerRankingOutput for VirtualReviewTicketOutput {
    fn proposer_ranking_output(&self) -> &RoleOutput {
        &self.output
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualReviewPhysicalInvocationEvidence {
    /// Actual backend-issued invocation domain; Warm differs from source Fresh.
    pub invocation_key: Digest,
    pub accepted_seed_digest: Option<Digest>,
    pub physical_fresh: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VirtualReviewCaptureFailureStage {
    Submitted,
    Completed,
    Retired,
}

/// Classification is explicit, never inferred from an error message. Legacy
/// observers default to Required so invalid input and unknown failures cannot
/// be silently reclassified as optional instrumentation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VirtualReviewCaptureFailure {
    Required(RoleFailure),
    Observation(RoleFailure),
}
impl VirtualReviewCaptureFailure {
    pub fn into_cause(self) -> RoleFailure {
        match self { Self::Required(cause) | Self::Observation(cause) => cause }
    }
}
impl From<RoleFailure> for VirtualReviewCaptureFailure {
    fn from(value: RoleFailure) -> Self { Self::Required(value) }
}

/// Collection is opt-in. Callbacks run on the native ticket owner and may only
/// seal/enqueue bounded records; no training or synchronous archive I/O belongs
/// here. Prepared/submitted input is never proof of NN admission or consumption.
pub trait VirtualReviewCollectionObserver: Send + 'static {
    fn prepared(
        &mut self,
        _: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        _: &VirtualReviewPreparedTicket,
    ) -> Result<(), RoleFailure> {
        Ok(())
    }
    fn submitted(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        prepared: &VirtualReviewPreparedTicket,
    ) -> Result<(), RoleFailure>;
    fn submitted_classified(
        &mut self,
        context: &TicketOwnerDispatchContext<'_, ReviewView, NativeTicketQuestion, VirtualReviewPreparedTicket>,
    ) -> Result<(), VirtualReviewCaptureFailure> {
        self.submitted(context.ticket(), context.question(), context.prepared()).map_err(VirtualReviewCaptureFailure::Required)
    }
    /// A capture-only failure may be isolated only when the observer has
    /// retained the original failure and revoked publication for this
    /// collection owner. Returning false keeps the historical fail-closed path.
    /// This does not acknowledge or alter any physical NN lifetime.
    fn preserve_observation_failure(
        &mut self,
        _: VirtualReviewCaptureFailureStage,
        _: &RoleTicketEnvelope,
        _: &RoleFailure,
    ) -> bool {
        false
    }
    fn completed(
        &mut self,
        ticket: &RoleTicketEnvelope,
        physical: PhysicalRoleCompletion,
        output: Option<&RoleOutput>,
        evidence: Option<TicketExecutionEvidence>,
        invocation: Option<VirtualReviewPhysicalInvocationEvidence>,
    ) -> Result<(), RoleFailure>;
    fn completed_classified(
        &mut self,
        ticket: &RoleTicketEnvelope,
        physical: PhysicalRoleCompletion,
        output: Option<&RoleOutput>,
        evidence: Option<TicketExecutionEvidence>,
        invocation: Option<VirtualReviewPhysicalInvocationEvidence>,
    ) -> Result<(), VirtualReviewCaptureFailure> {
        self.completed(ticket, physical, output, evidence, invocation).map_err(VirtualReviewCaptureFailure::Required)
    }
    fn retired(&mut self, ticket: &RoleTicketEnvelope, accepted: bool) -> Result<(), RoleFailure>;
    fn retired_classified(&mut self, ticket: &RoleTicketEnvelope, accepted: bool) -> Result<(), VirtualReviewCaptureFailure> {
        self.retired(ticket, accepted).map_err(VirtualReviewCaptureFailure::Required)
    }
    /// Post-Run, backend-issued V3 placement proof. Consumers may seal/enqueue
    /// this exact scope; a legacy P/C witness never certifies V or seeded Warm.
    fn runtime_audited(
        &mut self,
        _: &VirtualReviewCudaPlacementWitness,
    ) -> Result<(), RoleFailure> {
        Ok(())
    }
}

fn isolate_capture_failure(
    observer: &mut Option<Box<dyn VirtualReviewCollectionObserver>>,
    stage: VirtualReviewCaptureFailureStage,
    ticket: &RoleTicketEnvelope,
    failure: VirtualReviewCaptureFailure,
) -> Result<(), RoleFailure> {
    let failure = match failure {
        VirtualReviewCaptureFailure::Required(cause) => return Err(cause),
        VirtualReviewCaptureFailure::Observation(cause) => cause,
    };
    if failure == RoleFailure::PhysicalUnknown
        || !observer.as_mut().is_some_and(|o| o.preserve_observation_failure(stage, ticket, &failure))
    {
        return Err(failure);
    }
    *observer = None;
    Ok(())
}

#[derive(Default)]
struct AcceptedWarmProfileWindow {
    combinations: u8,
    sealed: bool,
}
impl AcceptedWarmProfileWindow {
    fn observe(&mut self, role: Role, physical_fresh: bool) -> bool {
        self.combinations |= match (role, physical_fresh) {
            (Role::Proposer, true) => 1,
            (Role::Proposer, false) => 2,
            (Role::Critic, true) => 4,
            (Role::Critic, false) => 8,
            (Role::Validator, _) => 0,
        };
        self.combinations == 15 && !self.sealed
    }
}

pub struct NativeVirtualReviewTicketBackend {
    worker: NativeWorker,
    applied_session_settings: Option<Arc<[PalsAppliedSessionSettings]>>,
    clock: ContractSystemClock,
    config: PalsModelConfig,
    model_epoch: [u8; 32],
    backend_digest: [u8; 32],
    validator_present: bool,
    approximate_warm_capable: bool,
    virtual_cuda_placement_required: bool,
    drain_limit: Duration,
    game: u64,
    execution_sequence: u64,
    request_high_water: u64,
    outstanding: Option<NativeLease>,
    physical_unknown: bool,
    game_reset_failed: bool,
    completed_nn_seen: bool,
    runtime_audited: bool,
    accepted_warm_profile_window: Option<AcceptedWarmProfileWindow>,
    evidence: Option<(RequestId, TicketExecutionEvidence)>,
    completed_tickets: BTreeMap<RequestId, RoleTicketEnvelope>,
    observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
    #[cfg(feature = "native-validation-diagnostics")]
    validation_trace: Option<
        rz_eval::pals_onnx::virtual_review::native_validation_diagnostics::GpuLifecycleTrace,
    >,
}
pub type VirtualReviewRoleTicketService =
    TicketedNativeService<ReviewView, NativeTicketQuestion, NativeVirtualReviewTicketBackend>;

impl NativeVirtualReviewTicketBackend {
    /// Construct only in TicketedNativeService::spawn's owner factory. Model
    /// load/verification and physical worker creation never run on the manager.
    pub fn new(
        backend: PalsVirtualReviewOnnxBackend,
        clock: ContractSystemClock,
        drain_limit: Duration,
    ) -> Result<Self, RoleFailure> {
        Self::new_with_collection_observer(backend, clock, drain_limit, None)
    }
    /// Install the optional observer at original worker construction, before
    /// Ready or any request. Failure retains the same startup ownership rules.
    pub fn new_with_collection_observer(
        backend: PalsVirtualReviewOnnxBackend,
        clock: ContractSystemClock,
        drain_limit: Duration,
        observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
    ) -> Result<Self, RoleFailure> {
        if drain_limit.is_zero()
            || backend.rules_input_semantic_digest() != pals_virtual_rules_semantic_digest()
        {
            return Err(RoleFailure::InvalidOutput);
        }
        let existing = backend.snapshot_stats().map_err(backend_failure)?;
        if existing.admitted_role_requests != 0 || existing.completed_nn_inputs != 0 {
            return Err(RoleFailure::Unsupported(
                "ticketed V3 requires an unused physical owner",
            ));
        }
        // The loader already performs strict initial dependency/mapping checks.
        // verify_runtime is a post-Run final audit and must not be called before
        // the first real invocation or replaced by an unmeasured warm-up.
        let config = backend.model_config().clone();
        let model_epoch = backend.model_epoch();
        let backend_digest = backend.backend_digest();
        let validator_present = backend.validator_present();
        let approximate_warm_capable = backend.approximate_warm_capable();
        let virtual_cuda_placement_required = backend.virtual_cuda_placement_required();
        let game = backend.game_generation();
        let applied_session_settings = backend.applied_session_settings();
        #[cfg(feature = "native-validation-diagnostics")]
        let validation_trace = backend.gpu_lifecycle_validation_trace();
        Ok(Self {
            worker: backend.into_worker().map_err(backend_failure)?,
            applied_session_settings: Some(applied_session_settings),
            clock,
            config,
            model_epoch,
            backend_digest,
            validator_present,
            approximate_warm_capable,
            virtual_cuda_placement_required,
            drain_limit,
            game,
            execution_sequence: 0,
            request_high_water: 0,
            outstanding: None,
            physical_unknown: false,
            game_reset_failed: false,
            completed_nn_seen: false,
            runtime_audited: false,
            accepted_warm_profile_window: None,
            evidence: None,
            completed_tickets: BTreeMap::new(),
            #[cfg(feature = "native-validation-diagnostics")]
            validation_trace,
            observer,
        })
    }
    pub fn set_collection_observer(
        &mut self,
        observer: Box<dyn VirtualReviewCollectionObserver>,
    ) -> Result<(), RoleFailure> {
        if self.execution_sequence != 0 || self.outstanding.is_some() || self.observer.is_some() {
            return Err(RoleFailure::InvalidOutput);
        }
        self.observer = Some(observer);
        Ok(())
    }
    /// 원본 session 생성 성공의 설정 기록만 전달하며 첫 NN/OS 관측을 만들지 않는다.
    pub fn applied_session_settings(&self) -> Option<Arc<[PalsAppliedSessionSettings]>> {
        self.applied_session_settings.as_ref().map(Arc::clone)
    }
    pub fn validator_present(&self) -> bool {
        self.validator_present
    }
    pub fn approximate_warm_capable(&self) -> bool {
        self.approximate_warm_capable
    }
    pub fn model_epoch(&self) -> [u8; 32] {
        self.model_epoch
    }
    pub fn backend_digest(&self) -> [u8; 32] {
        self.backend_digest
    }

    /// 진단 capture만 선택한다. 실제 P/C Fresh/Warm의 원 accepted ack가
    /// 모두 모이면 idle worker에서 profile을 닫되 최종 runtime 감사는 남긴다.
    pub fn enable_accepted_warm_profile_window(&mut self) -> Result<(), RoleFailure> {
        if !self.virtual_cuda_placement_required
            || !self.approximate_warm_capable
            || self.validator_present
            || self.execution_sequence != 0
            || self.completed_nn_seen
            || self.outstanding.is_some()
            || self.physical_unknown
            || self.game_reset_failed
            || self.runtime_audited
            || self.accepted_warm_profile_window.is_some()
        {
            return Err(RoleFailure::Unsupported(
                "accepted Warm profile window requires an unused V-free CUDA Warm owner",
            ));
        }
        self.accepted_warm_profile_window = Some(AcceptedWarmProfileWindow::default());
        Ok(())
    }

    fn seal_warm_profile_after_accept(
        &mut self,
        role: Role,
        physical_fresh: bool,
        until: Instant,
    ) -> Result<(), RoleFailure> {
        let Some(window) = &mut self.accepted_warm_profile_window else {
            return Ok(());
        };
        if !window.observe(role, physical_fresh) {
            return Ok(());
        }
        if self.physical_unknown || self.outstanding.is_some() {
            return Err(RoleFailure::PhysicalUnknown);
        }
        if Instant::now() >= until {
            return Err(RoleFailure::Expired);
        }
        match self.execute_command(
            VirtualReviewNativeCommand::VerifyVirtualCudaPlacement,
            until,
        )? {
            VirtualReviewNativeResult::VirtualCudaPlacementVerified(_) => {
                self.accepted_warm_profile_window
                    .as_mut()
                    .expect("opt-in profile window retained")
                    .sealed = true;
                // runtime_audited stays false. finish must still verify final
                // dependency/mapping state, cached placement and physical join.
                Ok(())
            }
            _ => Err(RoleFailure::InvalidOutput),
        }
    }

    /// The physical deadline is solely a drain bound. The original ticket
    /// deadline remains unchanged for logical admission/consumption.
    fn physical_until(&self, hard: Instant) -> Result<Instant, RoleFailure> {
        hard.checked_add(self.drain_limit)
            .ok_or(RoleFailure::InvalidOutput)
    }
    fn execute_command(
        &mut self,
        command: VirtualReviewNativeCommand,
        until: Instant,
    ) -> Result<VirtualReviewNativeResult, RoleFailure> {
        if self.physical_unknown || self.outstanding.is_some() {
            return Err(RoleFailure::PhysicalUnknown);
        }
        self.outstanding = Some(self.worker.submit(command).map_err(backend_failure)?);
        loop {
            match self
                .outstanding
                .as_mut()
                .expect("owned physical lease")
                .poll()
            {
                PhysicalPoll::Ready(result) => {
                    self.outstanding = None;
                    return result.map_err(backend_failure);
                }
                PhysicalPoll::Pending if Instant::now() < until => {
                    // Only the dedicated owner waits; the manager remains
                    // nonblocking and can still issue atomic cancellation.
                    std::thread::sleep(Duration::from_micros(100));
                }
                PhysicalPoll::Pending | PhysicalPoll::Quarantined | PhysicalPoll::Consumed => {
                    self.physical_unknown = true;
                    return Err(RoleFailure::PhysicalUnknown);
                }
            }
        }
    }
    fn reset_game(&mut self, game: u64, until: Instant) -> Result<(), RoleFailure> {
        if self.game_reset_failed || game < self.game {
            return Err(RoleFailure::InvalidOutput);
        }
        if game != self.game {
            match self.execute_command(VirtualReviewNativeCommand::ResetTo(game), until) {
                Ok(VirtualReviewNativeResult::ResetTo { game_generation })
                    if game_generation == game =>
                {
                    self.game = game;
                    // ResetTo physically clears the backend's completion bank.
                    // Older immutable service outputs still retire logically.
                    self.completed_tickets.clear();
                }
                Ok(_) => {
                    self.game_reset_failed = true;
                    return Err(RoleFailure::InvalidOutput);
                }
                Err(failure) => {
                    self.game_reset_failed = true;
                    return Err(failure);
                }
            }
        }
        Ok(())
    }
}

impl TicketNativeBackend<ReviewView, NativeTicketQuestion> for NativeVirtualReviewTicketBackend {
    type Prepared = VirtualReviewPreparedTicket;
    type Output = VirtualReviewTicketOutput;
    fn clock(&self) -> ContractSystemClock {
        self.clock.clone()
    }
    fn prepare(
        &mut self,
        question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        canceled: &AtomicBool,
    ) -> Result<(Self::Prepared, RepresentationKey), RoleFailure> {
        if self.physical_unknown {
            return Err(RoleFailure::PhysicalUnknown);
        }
        if self.game_reset_failed {
            return Err(RoleFailure::InvalidOutput);
        }
        if canceled.load(Ordering::Acquire) {
            return Err(RoleFailure::Canceled);
        }
        if question.seal.authority.epoch != self.clock.domain().0
            || question.query.legacy_full_records.is_some()
        {
            return Err(RoleFailure::InvalidOutput);
        }
        if question.role == Role::Validator && !self.validator_present {
            return Err(RoleFailure::Unsupported(
                "deployment does not load a V graph",
            ));
        }
        question
            .deadline
            .accepts(self.clock.domain(), self.clock.now())
            .map_err(|_| RoleFailure::Expired)?;
        let (input, identity) = encode_view(question, self.model_epoch, self.backend_digest)?;
        let endpoint_history =
            pals_native::pals_history_digest(&question.query.position).map_err(role_failure)?;
        let representation = representation(
            &input,
            &identity,
            endpoint_history,
            question.query.context.public_revision,
        )?;
        let prepared = VirtualReviewPreparedTicket {
            input,
            identity,
            representation,
        };
        if let Some(observer) = &mut self.observer {
            observer.prepared(question, &prepared)?;
        }
        Ok((prepared, representation))
    }
    fn prepared_host_bytes(&self, prepared: &Self::Prepared) -> u64 {
        // Reserve both the immutable service input and the physical command's
        // owned input copy. Native ORT backing/workspace is backend-observed.
        (std::mem::size_of::<VirtualReviewPreparedTicket>() as u64)
            .saturating_add(prepared.input.owned_bytes() as u64 * 2)
            .saturating_add(prepared.identity.anchor_history_seals.capacity() as u64 * 32 * 2)
    }
    fn question_host_bytes(
        &self,
        question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
    ) -> u64 {
        question_bytes(question)
    }
    fn run(
        &mut self,
        _: &RoleTicketEnvelope,
        _: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        _: &Self::Prepared,
        _: &AtomicBool,
    ) -> TicketPhysicalResult<Self::Output> {
        TicketPhysicalResult {
            physical: PhysicalRoleCompletion::Confirmed,
            output: Err(RoleFailure::Unsupported("original ticket service owner required")),
        }
    }
    fn run_owned(
        &mut self,
        context: TicketOwnerDispatchContext<'_, ReviewView, NativeTicketQuestion, Self::Prepared>,
        canceled: &AtomicBool,
    ) -> TicketPhysicalResult<Self::Output> {
        let ticket = context.ticket();
        let question = context.question();
        let prepared = context.prepared();
        self.evidence = None;
        let mut completed_invocation = None;
        let mut observer_submitted = false;
        let result = (|| {
            if ticket.representation != prepared.representation
                || ticket.role != prepared.representation.role
                || ticket.id.sequence <= self.request_high_water
            {
                return Err(RoleFailure::InvalidOutput);
            }
            dispatch_allowed(ticket, &self.clock, question.view.token_budget(), canceled)?;
            self.request_high_water = ticket.id.sequence;
            let now = self
                .clock
                .try_now()
                .map_err(|_| RoleFailure::InvalidOutput)?;
            let clock_hard = Instant::now()
                .checked_add(self.clock.elapsed(now, ticket.original_deadline.at))
                .ok_or(RoleFailure::InvalidOutput)?;
            // Either supplied clock can tighten a physical drain. Neither may
            // silently extend the original registered logical deadline.
            let until = self.physical_until(question.query.deadline.min(clock_hard))?;
            self.reset_game(ticket.authority.game.0, until)?;
            dispatch_allowed(ticket, &self.clock, question.view.token_budget(), canceled)?;
            if let Some(observer) = &mut self.observer {
                match observer.submitted_classified(&context) {
                    Ok(()) => observer_submitted = true,
                    Err(failure) => {
                        isolate_capture_failure(&mut self.observer, VirtualReviewCaptureFailureStage::Submitted, ticket, failure)?;
                    }
                }
            }
            // A reset command or opt-in collection enqueue may have yielded
            // while stop/expiry arrived. Never dispatch using the drain bound
            // as authority to extend the original logical request lifetime.
            dispatch_allowed(ticket, &self.clock, question.view.token_budget(), canceled)?;
            self.execution_sequence = self
                .execution_sequence
                .checked_add(1)
                .ok_or(RoleFailure::InvalidOutput)?;
            let execution = ExecutionId::new(self.clock.domain().0, self.execution_sequence);
            let invocation = VirtualReviewInvocation {
                input: prepared.input.clone(),
                identity: prepared.identity.clone(),
                request_history: prepared.representation.history.0,
            };
            let command = VirtualReviewNativeCommand::EvaluateTicket {
                ticket: *ticket,
                invocation,
            };
            #[cfg(feature = "native-validation-diagnostics")]
            let command = if self.validation_trace.is_some() {
                match command {
                    VirtualReviewNativeCommand::EvaluateTicket { ticket, invocation } => {
                        VirtualReviewNativeCommand::EvaluateTicketForValidation {
                            ticket,
                            execution,
                            invocation,
                        }
                    }
                    _ => unreachable!("original ticket command"),
                }
            } else {
                command
            };
            let result = self.execute_command(command, until)?;
            let VirtualReviewNativeResult::TicketEvaluationWithEvidence {
                outcome,
                actual_completed_nn_inputs_delta,
                complete,
                owner_accounting,
            } = result
            else {
                return Err(RoleFailure::InvalidOutput);
            };
            let completion = outcome.map_err(backend_failure);
            let evidence = TicketExecutionEvidence {
                execution: Some(execution),
                completed_nn_inputs: actual_completed_nn_inputs_delta,
                measurement_complete: complete,
                physical_fresh: completion.as_ref().ok().map(|value| !value.warm_started),
                accounting: owner_accounting.map(|accounting| TicketNativeAccountingEvidence {
                    public_cache_hits: accounting.public_cache_hits(),
                    public_cache_misses: accounting.public_cache_misses(),
                    public_nn_runs_attempted: accounting.public_nn_runs_attempted(),
                    public_nn_runs_completed: accounting.public_nn_runs_completed(),
                    public_nn_runs_failed_known: accounting.public_nn_runs_failed_known(),
                    role_nn_runs_attempted: accounting.role_nn_runs_attempted(),
                    role_nn_runs_completed: accounting.role_nn_runs_completed(),
                    role_nn_runs_failed_known: accounting.role_nn_runs_failed_known(),
                    validator_nn_runs_attempted: accounting.validator_nn_runs_attempted(),
                    validator_nn_runs_completed: accounting.validator_nn_runs_completed(),
                    validator_nn_runs_failed_known: accounting.validator_nn_runs_failed_known(),
                }),
            };
            self.evidence = Some((ticket.id, evidence));
            self.completed_nn_seen |=
                actual_completed_nn_inputs_delta.is_some_and(|count| count > 0);
            let completion = completion?;
            completed_invocation = Some(VirtualReviewPhysicalInvocationEvidence {
                invocation_key: Digest(completion.invocation_key),
                accepted_seed_digest: completion.accepted_seed_digest.map(Digest),
                physical_fresh: !completion.warm_started,
            });
            if self.completed_tickets.len() >= 16
                || self.completed_tickets.insert(ticket.id, *ticket).is_some()
            {
                return Err(RoleFailure::InvalidOutput);
            }
            let output = decode_output(ticket, &prepared.input, &self.config, &completion.raw)?;
            Ok(VirtualReviewTicketOutput {
                output,
                evidence,
                completion,
            })
        })();
        let physical = if self.physical_unknown {
            PhysicalRoleCompletion::Unknown
        } else {
            PhysicalRoleCompletion::Confirmed
        };
        #[cfg(feature = "native-validation-diagnostics")]
        if let Some(trace) = &self.validation_trace {
            if let Some((id, evidence)) = self.evidence {
                if id == ticket.id {
                    if let Some(execution) = evidence.execution {
                        trace.backend_completed(
                            *ticket,
                            execution,
                            completed_invocation.map(|value| value.invocation_key),
                            completed_invocation.and_then(|value| value.accepted_seed_digest),
                            completed_invocation.is_some_and(|value| value.physical_fresh),
                            evidence.completed_nn_inputs,
                            evidence.measurement_complete
                                && physical == PhysicalRoleCompletion::Confirmed
                                && result.is_ok(),
                        );
                    }
                }
            }
        }
        let mut result = result;
        if observer_submitted && let Some(observer) = &mut self.observer {
            let evidence = self
                .evidence
                .as_ref()
                .and_then(|(id, value)| (*id == ticket.id).then_some(*value));
            if let Err(failure) = observer.completed_classified(
                ticket,
                physical,
                result.as_ref().ok().map(|value| &value.output),
                evidence,
                completed_invocation,
            ) {
                if let Err(failure) = isolate_capture_failure(
                    &mut self.observer, VirtualReviewCaptureFailureStage::Completed, ticket, failure,
                ) && physical == PhysicalRoleCompletion::Confirmed && result.is_ok() {
                    result = Err(failure);
                }
            }
        }
        TicketPhysicalResult {
            physical,
            output: result,
        }
    }
    fn output_host_bytes(&self, output: &Self::Output) -> u64 {
        let logits = match &output.output.payload {
            RolePayload::Proposal {
                candidate_logits, ..
            } => candidate_logits.capacity(),
            RolePayload::Counterexample {
                candidate_logits,
                divergence_logits,
                ..
            } => candidate_logits
                .capacity()
                .saturating_add(divergence_logits.capacity()),
            RolePayload::TaskRanking { task_logits } => task_logits.capacity(),
        };
        let raw = &output.completion.raw;
        (std::mem::size_of::<VirtualReviewTicketOutput>()
            + (logits
                + output.output.private_latent.capacity()
                + raw.candidate_logits.capacity()
                + raw.private_latent.capacity()
                + raw.divergence_logits.as_ref().map_or(0, Vec::capacity))
                * 4) as u64
    }
    fn acceptance_output_digest(&self, output: &Self::Output) -> Option<Digest> {
        // 원래 native P/V head만 식별한다. Search가 별도로 판정한 Rules 종료
        // 가치나 이후의 view를 이 출력에 덮어쓰지 않으며, C는 기존 경로를 따른다.
        rz_search::pals::coordinator::native_acceptance_output_digest(&output.output).ok()
    }
    fn execution_evidence(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Option<TicketExecutionEvidence> {
        if self
            .evidence
            .as_ref()
            .is_some_and(|(id, _)| *id == ticket.id)
        {
            self.evidence.take().map(|(_, evidence)| evidence)
        } else {
            None
        }
    }
    fn accept(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        prepared: &Self::Prepared,
        output: &Self::Output,
    ) -> Result<(), RoleFailure> {
        if output.output.id != ticket.id
            || output.output.authority != ticket.authority
            || output.output.key != prepared.representation
            || ticket.representation != prepared.representation
        {
            return Err(RoleFailure::InvalidOutput);
        }
        let _ = question;
        let until = self.physical_until(Instant::now())?;
        match self.execute_command(
            VirtualReviewNativeCommand::AcceptTicket {
                ticket: *ticket,
                invocation: VirtualReviewInvocation {
                    input: prepared.input.clone(),
                    identity: prepared.identity.clone(),
                    request_history: prepared.representation.history.0,
                },
                completion: output.completion.clone(),
            },
            until,
        )? {
            VirtualReviewNativeResult::TicketAccepted => {
                self.completed_tickets.remove(&ticket.id);
                if let Some(observer) = &mut self.observer {
                    if let Err(failure) = observer.retired_classified(ticket, true) {
                        isolate_capture_failure(&mut self.observer, VirtualReviewCaptureFailureStage::Retired, ticket, failure)?;
                    }
                }
                self.seal_warm_profile_after_accept(
                    ticket.role,
                    !output.completion.warm_started,
                    until,
                )?;
                Ok(())
            }
            _ => Err(RoleFailure::InvalidOutput),
        }
    }
    fn discard(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), RoleFailure> {
        if self.physical_unknown || self.outstanding.is_some() {
            return Err(RoleFailure::PhysicalUnknown);
        }
        let Some(completed) = self.completed_tickets.get(&ticket.id) else {
            // Cancellation may happen in the service before run, or in run
            // before native admission. There is no opaque backend completion
            // to discard, but collection's submitted/prepared slot must retire.
            if let Some(observer) = &mut self.observer {
                if let Err(failure) = observer.retired_classified(ticket, false) {
                    isolate_capture_failure(&mut self.observer, VirtualReviewCaptureFailureStage::Retired, ticket, failure)?;
                }
            }
            return Ok(());
        };
        if completed != ticket {
            return Err(RoleFailure::InvalidOutput);
        }
        let until = Instant::now()
            .checked_add(self.drain_limit)
            .ok_or(RoleFailure::InvalidOutput)?;
        let result =
            match self.execute_command(VirtualReviewNativeCommand::DiscardTicket(*ticket), until) {
                Ok(VirtualReviewNativeResult::TicketDiscarded) => Ok(()),
                Ok(_) => Err(RoleFailure::InvalidOutput),
                Err(failure) => Err(failure),
            };
        if !self.physical_unknown {
            self.completed_tickets.remove(&ticket.id);
            if let Some(observer) = &mut self.observer {
                if let Err(failure) = observer.retired_classified(ticket, false) {
                    let observation = isolate_capture_failure(
                        &mut self.observer, VirtualReviewCaptureFailureStage::Retired, ticket, failure,
                    );
                    // A failed native discard remains the primary physical
                    // result. Capture observers that support isolation retain
                    // their separate cause even when both paths fail.
                    if result.is_ok() { observation?; }
                }
            }
        }
        result
    }
    fn quiesce(&mut self, until: Instant) -> bool {
        // The physical owner revokes only root-scoped completion/seed authority.
        // This is neither ResetTo (game advance) nor shutdown (session release).
        if self.physical_unknown
            || self.game_reset_failed
            || self.outstanding.is_some()
            || Instant::now() >= until
        {
            return false;
        }
        match self.execute_command(VirtualReviewNativeCommand::QuiesceRoot, until) {
            Ok(VirtualReviewNativeResult::RootQuiesced { game_generation })
                if game_generation == self.game =>
            {
                self.completed_tickets.clear();
                self.evidence = None;
                // Execution/request high-water and post-Run audit counters stay
                // with this owner for the eventual model/session shutdown.
                true
            }
            _ => false,
        }
    }
    fn finish(&mut self, until: Instant) -> bool {
        if self.physical_unknown || self.outstanding.is_some() {
            return false;
        }
        let mut audit_passed = !self.game_reset_failed;
        if audit_passed && self.completed_nn_seen && !self.runtime_audited {
            audit_passed = Instant::now() < until
                && matches!(
                    self.execute_command(VirtualReviewNativeCommand::VerifyRuntime, until),
                    Ok(VirtualReviewNativeResult::RuntimeVerified)
                );
            if audit_passed && self.virtual_cuda_placement_required {
                // The actual loaded policy declares this V3 domain. VerifyRuntime
                // checks dependencies/mappings first; this proof then seals all
                // registered V3 roles and actual Fresh/seeded-Warm executions.
                audit_passed = Instant::now() < until
                    && match self.execute_command(
                        VirtualReviewNativeCommand::VerifyVirtualCudaPlacement,
                        until,
                    ) {
                        Ok(VirtualReviewNativeResult::VirtualCudaPlacementVerified(witness)) => {
                            let accepted = self
                                .observer
                                .as_mut()
                                .is_none_or(|observer| observer.runtime_audited(&witness).is_ok());
                            #[cfg(feature = "native-validation-diagnostics")]
                            if accepted {
                                if let Some(trace) = &self.validation_trace {
                                    trace.actual_placement_verified();
                                }
                            }
                            accepted
                        }
                        _ => false,
                    };
            }
            self.runtime_audited = audit_passed;
            if self.physical_unknown || self.outstanding.is_some() {
                // Unknown completion still owns native buffers/workspace. A
                // semantic audit rejection cannot turn quarantine into a fence.
                return false;
            }
        }
        // A known audit/collection failure still closes physical admission and
        // attempts a bounded native drain/join. Its failed acceptance remains
        // false even when native destruction completes successfully.
        loop {
            match self.worker.try_shutdown() {
                Poll::Ready(result) => return audit_passed && result.is_ok(),
                Poll::Pending if Instant::now() < until => {
                    std::thread::sleep(Duration::from_micros(100))
                }
                Poll::Pending => return false,
            }
        }
    }
}

fn encode_view(
    question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
    model_epoch: [u8; 32],
    backend: [u8; 32],
) -> Result<(PalsVirtualReviewInput, ReviewExecutionIdentity), RoleFailure> {
    let view = &question.view;
    let context = &question.query.context;
    if question.seal.view_digest.0 != view.seal_digest()
        || question.seal.context_policy.0 != view.policy_digest()
        || question.seal.view_sequence != view.sequence()
        || question.seal.actual_tokens != view.token_count()
        || usize::from(question.seal.selected_records) != view.records().len()
        || usize::from(question.seal.anchors) != view.anchors().len()
        || question.seal.length_bucket != view.length_bucket()
        || question.seal.authority.game.0 != context.game_generation
    {
        return Err(RoleFailure::InvalidOutput);
    }
    let first = view.anchors().first().ok_or(RoleFailure::InvalidOutput)?;
    if question.seal.root_state.semantic
        != rz_position::contracts::rules_snapshot_semantic_digest(&first.snapshot)
    {
        return Err(RoleFailure::InvalidOutput);
    }
    let (input, identity) = pals_native::virtual_encoding::prepare_selected_virtual_review_input(
        &question.query.position,
        context,
        view,
        model_epoch,
        backend,
    )
    .map_err(role_failure)?;
    let role = match question.role {
        Role::Proposer => PalsRole::Proposer,
        Role::Critic => PalsRole::Critic,
        Role::Validator => PalsRole::Validator,
    };
    if input.role != role {
        return Err(RoleFailure::InvalidOutput);
    }
    Ok((input, identity))
}

/// Reuse the existing exact Rules board/metadata encoder; collection callers
/// receive no authority to replace history or turn teacher CP into a feature.
pub fn exact_review_anchor(position: &Position) -> Result<ReviewAnchorInput, RoleFailure> {
    pals_native::virtual_encoding::exact_review_anchor_input(position).map_err(role_failure)
}
/// Pure original key/mask calculation with unchanged bytes. It creates only a
/// RepresentationKey, never a ticket, worker, permit, physical or training cap.
pub fn virtual_review_representation_for_input(
    input: &PalsVirtualReviewInput,
    identity: &ReviewExecutionIdentity,
    endpoint_history: [u8; 32],
    revision: u64,
) -> Result<RepresentationKey, RoleFailure> {
    representation(input, identity, endpoint_history, revision)
}

fn representation(
    input: &PalsVirtualReviewInput,
    identity: &ReviewExecutionIdentity,
    endpoint_history: [u8; 32],
    revision: u64,
) -> Result<RepresentationKey, RoleFailure> {
    let mut candidates = Sha256::new();
    for movement in &input.candidates {
        candidates.update(
            movement
                .packed()
                .map_err(|_| RoleFailure::InvalidOutput)?
                .to_le_bytes(),
        );
    }
    let prepared = input.prepare().map_err(|_| RoleFailure::InvalidOutput)?;
    let mut masks = Sha256::new();
    masks.update(b"rz-pals-virtual-review-physical-masks/1\0");
    for value in [
        prepared.anchors,
        prepared.records_count,
        prepared.candidates_count,
        prepared.divergences_count,
        prepared.length_bucket,
    ] {
        masks.update((value as u64).to_le_bytes());
    }
    for mask in [
        &prepared.anchor_mask,
        &prepared.record_mask,
        &prepared.record_line_mask,
        &prepared.query_line_mask,
        &prepared.candidate_mask,
        &prepared.divergence_mask,
    ] {
        masks.update((mask.len() as u64).to_le_bytes());
        for value in mask {
            masks.update([u8::from(*value)]);
        }
    }
    Ok(RepresentationKey {
        input: Digest(
            input
                .fresh_key(identity)
                .map_err(|_| RoleFailure::InvalidOutput)?
                .0,
        ),
        history: Digest(endpoint_history),
        candidates: Digest(candidates.finalize().into()),
        mask_positions: Digest(masks.finalize().into()),
        model: Digest(identity.model_epoch),
        encoding: Digest(Sha256::digest(model::ENCODING.as_bytes()).into()),
        precision: PrecisionProfile::Fp32,
        frozen_epoch: 1,
        record_revision: revision,
        role: match input.role {
            PalsRole::Proposer => Role::Proposer,
            PalsRole::Critic => Role::Critic,
            PalsRole::Validator => Role::Validator,
        },
    })
}
fn decode_output(
    ticket: &RoleTicketEnvelope,
    input: &PalsVirtualReviewInput,
    config: &PalsModelConfig,
    raw: &PalsRawOutput,
) -> Result<RoleOutput, RoleFailure> {
    if raw.candidate_logits.len() != input.candidates.len()
        || raw.private_latent.len() != config.latent_elements()
        || raw
            .candidate_logits
            .iter()
            .chain(&raw.private_latent)
            .any(|v| !v.is_finite())
    {
        return Err(RoleFailure::InvalidOutput);
    }
    let wdl_values = stable_wdl(raw.wdl_logits)?;
    let wdl = Wdl::try_new(wdl_values[0], wdl_values[1], wdl_values[2], 1e-4)
        .map_err(|_| RoleFailure::InvalidOutput)?;
    let payload = match input.role {
        PalsRole::Proposer if raw.divergence_logits.is_none() && raw.task_logits.is_none() => {
            RolePayload::Proposal {
                candidate_logits: raw.candidate_logits.clone(),
                wdl,
            }
        }
        PalsRole::Critic if raw.task_logits.is_none() => {
            let logits = raw
                .divergence_logits
                .as_ref()
                .ok_or(RoleFailure::InvalidOutput)?;
            if logits.len() != input.divergences.len() || logits.iter().any(|v| !v.is_finite()) {
                return Err(RoleFailure::InvalidOutput);
            }
            RolePayload::Counterexample {
                candidate_logits: raw.candidate_logits.clone(),
                divergence_logits: logits.clone(),
                wdl,
            }
        }
        PalsRole::Validator if raw.divergence_logits.is_none() => {
            let logits = raw.task_logits.ok_or(RoleFailure::InvalidOutput)?;
            if logits.iter().any(|v| !v.is_finite()) {
                return Err(RoleFailure::InvalidOutput);
            }
            RolePayload::TaskRanking {
                task_logits: logits.to_vec(),
            }
        }
        _ => return Err(RoleFailure::InvalidOutput),
    };
    Ok(RoleOutput {
        id: ticket.id,
        authority: ticket.authority,
        key: ticket.representation,
        payload,
        private_latent: raw.private_latent.clone(),
    })
}
fn stable_wdl(logits: [f32; 3]) -> Result<[f32; 3], RoleFailure> {
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(RoleFailure::InvalidOutput);
    }
    let maximum = logits.into_iter().fold(f32::NEG_INFINITY, f32::max);
    let values = logits.map(|value| ((value - maximum) as f64).exp());
    let sum: f64 = values.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(RoleFailure::InvalidOutput);
    }
    Ok(values.map(|value| (value / sum) as f32))
}
#[cfg(test)]
fn ordered_line_digest(domain: &[u8], line: &[BoardMove]) -> Result<[u8; 32], RoleFailure> {
    rz_search::pals::engine::role_line_sha256(domain, line).map_err(|_| RoleFailure::InvalidOutput)
}
#[cfg(test)]
fn divergence_digest(indices: &[usize]) -> Result<[u8; 32], RoleFailure> {
    rz_search::pals::engine::role_divergence_sha256(indices).map_err(|_| RoleFailure::InvalidOutput)
}
fn question_bytes(question: &TicketPreparation<ReviewView, NativeTicketQuestion>) -> u64 {
    let local = std::mem::size_of::<NativeTicketQuestion>().checked_add(
        question
            .query
            .context
            .prefix
            .capacity()
            .saturating_mul(std::mem::size_of::<BoardMove>()),
    );
    question
        .view
        .owned_host_bytes()
        .and_then(|bytes| bytes.checked_add(u64::try_from(local?).ok()?))
        .and_then(|bytes| {
            bytes.checked_add(
                u64::try_from(
                    question
                        .query
                        .position
                        .snapshot()
                        .retained_history_bytes()?,
                )
                .ok()?,
            )
        })
        .unwrap_or(u64::MAX)
}
fn dispatch_allowed(
    ticket: &RoleTicketEnvelope,
    clock: &ContractSystemClock,
    token_budget: u16,
    canceled: &AtomicBool,
) -> Result<(), RoleFailure> {
    if canceled.load(Ordering::Acquire) {
        return Err(RoleFailure::Canceled);
    }
    let now = clock.try_now().map_err(|_| RoleFailure::InvalidOutput)?;
    ticket
        .validate(clock.domain(), now, token_budget)
        .map_err(|error| {
            if error.code == rz_contracts::ErrorCode::Expired {
                RoleFailure::Expired
            } else {
                RoleFailure::InvalidOutput
            }
        })
}
fn role_failure(error: rz_search::pals::engine::RoleError) -> RoleFailure {
    match error {
        rz_search::pals::engine::RoleError::Canceled => RoleFailure::Canceled,
        rz_search::pals::engine::RoleError::Deadline => RoleFailure::Expired,
        rz_search::pals::engine::RoleError::InvalidOutput => RoleFailure::InvalidOutput,
        rz_search::pals::engine::RoleError::PhysicalCompletionUnknown => {
            RoleFailure::PhysicalUnknown
        }
        other => RoleFailure::Backend(other.to_string()),
    }
}
fn backend_failure(error: BackendError) -> RoleFailure {
    RoleFailure::Backend(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::pals::SearchAuthority;
    use rz_contracts::pals_adaptive::ReviewViewSeal;
    use rz_contracts::{
        ClockDomain, Deadline, GameGeneration, MonotonicTick, OwnerId, ProcessEpoch,
        RootGeneration, StateIdentity, StateRevision,
    };
    use rz_position::Color;
    use rz_search::cpu::CpuScoreScope;
    use rz_search::pals::engine::{RecordKind, RoleQueryPurpose};
    use rz_search::pals::engine::{RoleLogicalContext, RoleRecord};
    use rz_search::pals::review_context::{
        ReviewAnchorMode, ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy,
    };
    use rz_search::pals::store::{PalsStores, StoreLimits};
    use std::sync::Arc;

    struct FailureObserver {
        preserve: bool,
        seen: Arc<std::sync::Mutex<Vec<(VirtualReviewCaptureFailureStage, RoleFailure)>>>,
    }
    impl VirtualReviewCollectionObserver for FailureObserver {
        fn submitted(&mut self, _: &RoleTicketEnvelope, _: &TicketPreparation<ReviewView, NativeTicketQuestion>, _: &VirtualReviewPreparedTicket) -> Result<(), RoleFailure> { unreachable!() }
        fn completed(&mut self, _: &RoleTicketEnvelope, _: PhysicalRoleCompletion, _: Option<&RoleOutput>, _: Option<TicketExecutionEvidence>, _: Option<VirtualReviewPhysicalInvocationEvidence>) -> Result<(), RoleFailure> { unreachable!() }
        fn retired(&mut self, _: &RoleTicketEnvelope, _: bool) -> Result<(), RoleFailure> { unreachable!() }
        fn preserve_observation_failure(&mut self, stage: VirtualReviewCaptureFailureStage, _: &RoleTicketEnvelope, failure: &RoleFailure) -> bool {
            if self.preserve { self.seen.lock().unwrap().push((stage, failure.clone())); }
            self.preserve
        }
    }

    #[test]
    fn capture_failure_is_independent_only_when_preserved_and_never_masks_physical_unknown() {
        let (_store, question) = question(Position::startpos(), &[], &[]);
        let (input, identity) = encode_view(&question, [1; 32], [2; 32]).unwrap();
        let ticket = RoleTicketEnvelope {
            id: RequestId::new(question.seal.authority.epoch, 1), authority: question.seal.authority,
            question: question.question, view: question.seal,
            representation: representation(&input, &identity, identity.rules_history, 0).unwrap(),
            role: question.role, original_deadline: question.deadline,
        };
        for stage in [VirtualReviewCaptureFailureStage::Submitted, VirtualReviewCaptureFailureStage::Completed, VirtualReviewCaptureFailureStage::Retired] {
            let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
            let cause = RoleFailure::Backend("capture queue unavailable".into());
            let mut observer: Option<Box<dyn VirtualReviewCollectionObserver>> = Some(Box::new(FailureObserver { preserve: true, seen: Arc::clone(&seen) }));
            assert_eq!(isolate_capture_failure(&mut observer, stage, &ticket, VirtualReviewCaptureFailure::Observation(cause.clone())), Ok(()));
            assert!(observer.is_none());
            assert_eq!(*seen.lock().unwrap(), vec![(stage, cause)]);
            let mut unpreserved: Option<Box<dyn VirtualReviewCollectionObserver>> = Some(Box::new(FailureObserver { preserve: false, seen: Arc::clone(&seen) }));
            assert_eq!(isolate_capture_failure(&mut unpreserved, stage, &ticket, VirtualReviewCaptureFailure::Observation(RoleFailure::InvalidOutput)), Err(RoleFailure::InvalidOutput));
            assert!(unpreserved.is_some());
            assert_eq!(isolate_capture_failure(&mut unpreserved, stage, &ticket, VirtualReviewCaptureFailure::Observation(RoleFailure::PhysicalUnknown)), Err(RoleFailure::PhysicalUnknown));
            assert!(unpreserved.is_some());
            let mut optional: Option<Box<dyn VirtualReviewCollectionObserver>> = Some(Box::new(FailureObserver { preserve: true, seen: Arc::clone(&seen) }));
            assert_eq!(isolate_capture_failure(&mut optional, stage, &ticket, VirtualReviewCaptureFailure::Required(RoleFailure::InvalidOutput)), Err(RoleFailure::InvalidOutput));
            assert!(optional.is_some());
        }
    }

    struct MockShutdownReceipt(Arc<AtomicBool>);
    impl Drop for MockShutdownReceipt {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }
    // No model/session is loaded: these workers return mock audit outcomes and
    // test only the bridge's known-failure drain/join contract.
    fn mock_finish_backend(
        worker: NativeWorker,
        required: bool,
    ) -> NativeVirtualReviewTicketBackend {
        NativeVirtualReviewTicketBackend {
            worker,
            applied_session_settings: None,
            clock: ContractSystemClock::new(ProcessEpoch(7)),
            config: PalsModelConfig::default(),
            model_epoch: [1; 32],
            backend_digest: [2; 32],
            validator_present: false,
            approximate_warm_capable: false,
            virtual_cuda_placement_required: required,
            drain_limit: Duration::from_secs(1),
            game: 1,
            execution_sequence: 0,
            request_high_water: 0,
            outstanding: None,
            physical_unknown: false,
            game_reset_failed: false,
            completed_nn_seen: true,
            runtime_audited: false,
            accepted_warm_profile_window: None,
            evidence: None,
            completed_tickets: BTreeMap::new(),
            #[cfg(feature = "native-validation-diagnostics")]
            validation_trace: None,
            observer: None,
        }
    }
    fn mock_audit_failure() -> BackendError {
        BackendError::new(
            rz_eval::error::FailureKind::BackendUnavailable,
            rz_eval::error::FailureStage::Backend,
            "mock known audit rejection",
        )
    }

    // 합성 worker 응답으로 제어 순서만 검사한다. 실제 CUDA kernel 증거가 아니다.
    fn mock_profile_window_witness() -> VirtualReviewCudaPlacementWitness {
        VirtualReviewCudaPlacementWitness {
            schema: "authored-control-only-mock".into(),
            inventory_sha256: [1; 32],
            manifest_sha256: [2; 32],
            runtime_sha256: [3; 32],
            runtime_bundle_sha256: [4; 32],
            layout: "authored-control-only-mock".into(),
            artifact_kind: "approx_warm".into(),
            optimization: rz_eval::pals_onnx::PalsGraphOptimization::Disable,
            initialization: vec![],
            kernels: BTreeMap::new(),
            validator_private_kernels: 0,
            validator_task_head_kernels: 0,
            warm_seed_identity_kernels: 0,
            category_provenance: "mock-only-no-model-or-kernel-loaded".into(),
        }
    }

    #[test]
    fn accepted_warm_profile_window_requires_all_four_combinations() {
        let mut window = AcceptedWarmProfileWindow::default();
        for _ in 0..6 {
            assert!(!window.observe(Role::Proposer, true));
            assert!(!window.observe(Role::Proposer, false));
        }
        assert!(!window.observe(Role::Validator, false));
        assert!(!window.observe(Role::Critic, true));
        assert!(window.observe(Role::Critic, false));
        window.sealed = true;
        assert!(!window.observe(Role::Critic, false));
        assert!(!window.observe(Role::Proposer, true));
    }

    #[test]
    fn accepted_warm_profile_window_seals_once_and_keeps_final_runtime_audit() {
        let dropped = Arc::new(AtomicBool::new(false));
        let receipt = MockShutdownReceipt(Arc::clone(&dropped));
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&order);
        let worker = SingleWorker::spawn(move |command: &VirtualReviewNativeCommand| {
            let _ = &receipt;
            match command {
                VirtualReviewNativeCommand::VerifyVirtualCudaPlacement => {
                    observed.lock().unwrap().push("placement");
                    Ok(VirtualReviewNativeResult::VirtualCudaPlacementVerified(
                        Box::new(mock_profile_window_witness()),
                    ))
                }
                VirtualReviewNativeCommand::VerifyRuntime => {
                    observed.lock().unwrap().push("runtime");
                    Ok(VirtualReviewNativeResult::RuntimeVerified)
                }
                _ => panic!("profile window must not dispatch an inference"),
            }
        })
        .unwrap();
        let mut backend = mock_finish_backend(worker, true);
        backend.approximate_warm_capable = true;
        backend.completed_nn_seen = false;
        backend.enable_accepted_warm_profile_window().unwrap();
        let until = Instant::now() + Duration::from_secs(2);
        for role in [Role::Proposer, Role::Critic] {
            for physical_fresh in [true, false] {
                backend
                    .seal_warm_profile_after_accept(role, physical_fresh, until)
                    .unwrap();
            }
        }
        assert!(
            backend
                .accepted_warm_profile_window
                .as_ref()
                .unwrap()
                .sealed
        );
        assert!(!backend.runtime_audited);
        assert_eq!(*order.lock().unwrap(), ["placement"]);
        backend
            .seal_warm_profile_after_accept(Role::Critic, false, until)
            .unwrap();
        assert_eq!(*order.lock().unwrap(), ["placement"]);
        backend.completed_nn_seen = true;
        assert!(backend.finish(until));
        assert_eq!(
            *order.lock().unwrap(),
            ["placement", "runtime", "placement"]
        );
        assert!(backend.runtime_audited);
        assert!(dropped.load(Ordering::Acquire));
    }

    #[test]
    fn accepted_warm_profile_window_failure_preserves_failed_audit_and_joins() {
        let dropped = Arc::new(AtomicBool::new(false));
        let receipt = MockShutdownReceipt(Arc::clone(&dropped));
        let worker = SingleWorker::spawn(move |command: &VirtualReviewNativeCommand| {
            let _ = &receipt;
            match command {
                VirtualReviewNativeCommand::VerifyVirtualCudaPlacement => Err(mock_audit_failure()),
                VirtualReviewNativeCommand::VerifyRuntime => {
                    Ok(VirtualReviewNativeResult::RuntimeVerified)
                }
                _ => panic!("failed profile window must not dispatch an inference"),
            }
        })
        .unwrap();
        let mut backend = mock_finish_backend(worker, true);
        backend.approximate_warm_capable = true;
        backend.completed_nn_seen = false;
        backend.enable_accepted_warm_profile_window().unwrap();
        let until = Instant::now() + Duration::from_secs(2);
        for (role, physical_fresh) in [
            (Role::Proposer, true),
            (Role::Proposer, false),
            (Role::Critic, true),
        ] {
            backend
                .seal_warm_profile_after_accept(role, physical_fresh, until)
                .unwrap();
        }
        let failure = backend
            .seal_warm_profile_after_accept(Role::Critic, false, until)
            .unwrap_err();
        assert_eq!(
            failure,
            RoleFailure::Backend(mock_audit_failure().to_string())
        );
        assert!(
            !backend
                .accepted_warm_profile_window
                .as_ref()
                .unwrap()
                .sealed
        );
        assert!(!backend.runtime_audited);
        assert!(!backend.physical_unknown);
        backend.completed_nn_seen = true;
        assert!(!backend.finish(until));
        assert!(dropped.load(Ordering::Acquire));
    }

    #[test]
    fn accepted_warm_profile_window_rejects_late_opt_in_and_expired_seal() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let worker = SingleWorker::spawn(move |_: &VirtualReviewNativeCommand| {
            observed.fetch_add(1, Ordering::AcqRel);
            panic!("expired profile window must not dispatch a command")
        })
        .unwrap();
        let mut backend = mock_finish_backend(worker, true);
        backend.approximate_warm_capable = true;
        assert!(matches!(
            backend.enable_accepted_warm_profile_window(),
            Err(RoleFailure::Unsupported(
                "accepted Warm profile window requires an unused V-free CUDA Warm owner"
            ))
        ));
        backend.completed_nn_seen = false;
        backend.enable_accepted_warm_profile_window().unwrap();
        let until = Instant::now();
        for (role, physical_fresh) in [
            (Role::Proposer, true),
            (Role::Proposer, false),
            (Role::Critic, true),
        ] {
            backend
                .seal_warm_profile_after_accept(role, physical_fresh, until)
                .unwrap();
        }
        assert_eq!(
            backend.seal_warm_profile_after_accept(Role::Critic, false, until),
            Err(RoleFailure::Expired)
        );
        assert!(
            !backend
                .accepted_warm_profile_window
                .as_ref()
                .unwrap()
                .sealed
        );
        assert!(!backend.runtime_audited);
        assert_eq!(calls.load(Ordering::Acquire), 0);
        assert!(backend.finish(Instant::now() + Duration::from_secs(2)));
    }

    #[test]
    fn new_game_advances_only_after_matching_physical_reset_ack() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = std::sync::Arc::clone(&calls);
        let worker = SingleWorker::spawn(move |command: &VirtualReviewNativeCommand| {
            observed.fetch_add(1, Ordering::AcqRel);
            match command {
                VirtualReviewNativeCommand::ResetTo(2) => {
                    Ok(VirtualReviewNativeResult::ResetTo { game_generation: 2 })
                }
                _ => panic!("unexpected fixture command"),
            }
        })
        .unwrap();
        let mut backend = mock_finish_backend(worker, false);
        assert!(backend.applied_session_settings().is_none());
        backend.request_high_water = 42;
        backend.execution_sequence = 7;
        backend
            .reset_game(2, Instant::now() + Duration::from_secs(2))
            .unwrap();
        assert_eq!(backend.game, 2);
        assert_eq!(backend.request_high_water, 42);
        assert_eq!(backend.execution_sequence, 7);
        // Repeated publication of this same game is metadata, not another
        // ResetTo or a worker/session teardown.
        backend
            .reset_game(2, Instant::now() + Duration::from_secs(2))
            .unwrap();
        assert_eq!(calls.load(Ordering::Acquire), 1);
        assert_eq!(backend.model_epoch(), [1; 32]);
        assert_eq!(backend.backend_digest(), [2; 32]);
        backend.completed_nn_seen = false;
        assert!(backend.finish(Instant::now() + Duration::from_secs(2)));
    }

    #[test]
    fn mismatched_physical_reset_ack_does_not_publish_new_game() {
        let worker = SingleWorker::spawn(move |command: &VirtualReviewNativeCommand| {
            assert!(matches!(command, VirtualReviewNativeCommand::ResetTo(2)));
            Ok(VirtualReviewNativeResult::ResetTo { game_generation: 3 })
        })
        .unwrap();
        let mut backend = mock_finish_backend(worker, false);
        backend.request_high_water = 42;
        backend.execution_sequence = 7;
        assert!(matches!(
            backend.reset_game(2, Instant::now() + Duration::from_secs(2)),
            Err(RoleFailure::InvalidOutput)
        ));
        assert_eq!(backend.game, 1);
        assert_eq!(backend.request_high_water, 42);
        assert_eq!(backend.execution_sequence, 7);
        assert!(backend.game_reset_failed);
        assert!(matches!(
            backend.reset_game(2, Instant::now() + Duration::from_secs(2)),
            Err(RoleFailure::InvalidOutput)
        ));
        backend.completed_nn_seen = false;
        assert!(!backend.finish(Instant::now() + Duration::from_secs(2)));
        // The semantic reset failure does not manufacture unknown physical
        // completion, and still joins the already idle native worker.
        assert!(!backend.physical_unknown);
        assert!(matches!(backend.worker.try_shutdown(), Poll::Ready(Ok(()))));
    }
    #[test]
    fn known_runtime_audit_failure_still_joins_without_success_claim() {
        let dropped = Arc::new(AtomicBool::new(false));
        let receipt = MockShutdownReceipt(Arc::clone(&dropped));
        let worker = SingleWorker::spawn(move |command: &VirtualReviewNativeCommand| {
            let _ = &receipt;
            assert!(matches!(command, VirtualReviewNativeCommand::VerifyRuntime));
            Err(mock_audit_failure())
        })
        .unwrap();
        let mut backend = mock_finish_backend(worker, false);
        assert!(!backend.finish(Instant::now() + Duration::from_secs(2)));
        assert!(dropped.load(Ordering::Acquire));
        assert!(!backend.runtime_audited);
        assert!(!backend.physical_unknown);
    }
    #[test]
    fn root_quiesce_reuses_owner_without_game_reset_or_inference() {
        let dropped = Arc::new(AtomicBool::new(false));
        let receipt = MockShutdownReceipt(Arc::clone(&dropped));
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&order);
        let worker = SingleWorker::spawn(move |command: &VirtualReviewNativeCommand| {
            let _ = &receipt;
            match command {
                VirtualReviewNativeCommand::QuiesceRoot => {
                    observed.lock().unwrap().push("root_quiesce");
                    Ok(VirtualReviewNativeResult::RootQuiesced { game_generation: 1 })
                }
                VirtualReviewNativeCommand::VerifyRuntime => {
                    observed.lock().unwrap().push("runtime");
                    Ok(VirtualReviewNativeResult::RuntimeVerified)
                }
                _ => panic!("root fence must not reset a game or dispatch inference"),
            }
        })
        .unwrap();
        let mut backend = mock_finish_backend(worker, false);
        backend.request_high_water = 42;
        backend.execution_sequence = 7;
        assert!(backend.quiesce(Instant::now() + Duration::from_secs(2)));
        assert_eq!(backend.game, 1);
        assert_eq!(backend.request_high_water, 42);
        assert_eq!(backend.execution_sequence, 7);
        assert!(backend.completed_nn_seen);
        assert!(!backend.runtime_audited);
        assert!(!dropped.load(Ordering::Acquire));
        assert!(backend.finish(Instant::now() + Duration::from_secs(2)));
        assert_eq!(*order.lock().unwrap(), ["root_quiesce", "runtime"]);
        assert!(dropped.load(Ordering::Acquire));
    }

    #[test]
    fn root_quiesce_rejects_mismatched_generation_without_relabeling_game() {
        let worker = SingleWorker::spawn(move |command: &VirtualReviewNativeCommand| {
            assert!(matches!(command, VirtualReviewNativeCommand::QuiesceRoot));
            Ok(VirtualReviewNativeResult::RootQuiesced { game_generation: 2 })
        })
        .unwrap();
        let mut backend = mock_finish_backend(worker, false);
        backend.completed_nn_seen = false;
        assert!(!backend.quiesce(Instant::now() + Duration::from_secs(2)));
        assert_eq!(backend.game, 1);
        assert!(backend.finish(Instant::now() + Duration::from_secs(2)));
    }
    #[test]
    fn registered_virtual_audit_follows_runtime_and_failure_still_joins() {
        let dropped = Arc::new(AtomicBool::new(false));
        let receipt = MockShutdownReceipt(Arc::clone(&dropped));
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&order);
        let worker = SingleWorker::spawn(move |command: &VirtualReviewNativeCommand| {
            let _ = &receipt;
            match command {
                VirtualReviewNativeCommand::VerifyRuntime => {
                    observed.lock().unwrap().push("runtime");
                    Ok(VirtualReviewNativeResult::RuntimeVerified)
                }
                VirtualReviewNativeCommand::VerifyVirtualCudaPlacement => {
                    observed.lock().unwrap().push("virtual_v3");
                    Err(mock_audit_failure())
                }
                _ => panic!("finish must not dispatch an inference"),
            }
        })
        .unwrap();
        let mut backend = mock_finish_backend(worker, true);
        assert!(!backend.finish(Instant::now() + Duration::from_secs(2)));
        assert_eq!(*order.lock().unwrap(), ["runtime", "virtual_v3"]);
        assert!(dropped.load(Ordering::Acquire));
        assert!(!backend.runtime_audited);
        assert!(!backend.physical_unknown);
    }

    fn question(
        root: Position,
        prefix_text: &[&str],
        records: &[(&str, Option<CpuScoreScope>)],
    ) -> (
        PalsStores,
        TicketPreparation<ReviewView, NativeTicketQuestion>,
    ) {
        question_with_counter(root, prefix_text, records, None)
    }
    fn question_with_counter(
        root: Position,
        prefix_text: &[&str],
        records: &[(&str, Option<CpuScoreScope>)],
        counter: Option<&[BoardMove]>,
    ) -> (
        PalsStores,
        TicketPreparation<ReviewView, NativeTicketQuestion>,
    ) {
        question_with_options(root, prefix_text, records, counter, false)
    }
    fn question_with_options(
        root: Position,
        prefix_text: &[&str],
        records: &[(&str, Option<CpuScoreScope>)],
        counter: Option<&[BoardMove]>,
        reverse_legal: bool,
    ) -> (
        PalsStores,
        TicketPreparation<ReviewView, NativeTicketQuestion>,
    ) {
        let mut stores = PalsStores::new(StoreLimits::default());
        let situation = stores.focus_actual_moves(root.snapshot()).unwrap();
        let situation_value = stores.situations.get(situation).unwrap();
        let state = situation_value.state;
        let focus = situation_value.focus;
        let mut index = ReviewContextIndex::default();
        for (i, (text, scope)) in records.iter().enumerate() {
            let record = RoleRecord {
                revision: i as u64 + 1,
                parent_revision: None,
                supersedes_revision: None,
                origin_state: state,
                kind: RecordKind::Proposal,
                line: text
                    .split_whitespace()
                    .map(|text| BoardMove::from_uci(text).unwrap())
                    .collect(),
                value: None,
                completed_depth: 0,
                score_scope: *scope,
                cpu_observation: None,
                perspective: root.side_to_move(),
                critical: true,
            };
            index.publish(&stores, &record, None).unwrap();
        }
        let prefix: Vec<_> = prefix_text
            .iter()
            .map(|text| BoardMove::from_uci(text).unwrap())
            .collect();
        let mut endpoint = root.clone();
        for movement in &prefix {
            endpoint.make_move(*movement).unwrap();
        }
        let mut legal = endpoint.legal_moves();
        if reverse_legal {
            legal.reverse();
        }
        let view = Arc::new(
            index
                .select(
                    &stores,
                    state,
                    ReviewQuerySequences {
                        purpose: RoleQueryPurpose::ProposePolicy,
                        prefix: &prefix,
                        proposal: &[],
                        counterexample: counter,
                        legal: &legal,
                        divergences: &[],
                    },
                    ReviewSelectionPolicy {
                        mode: ReviewAnchorMode::AnchoredCompact,
                        max_anchors: 4,
                        token_budget: 128,
                        ..ReviewSelectionPolicy::default()
                    },
                )
                .unwrap(),
        );
        let authority = SearchAuthority {
            epoch: ProcessEpoch(7),
            game: GameGeneration(1),
            root: RootGeneration(stores.generation()),
            implementation: Digest([9; 32]),
        };
        let seal = ReviewViewSeal {
            authority,
            view_sequence: view.sequence(),
            context_policy: Digest(view.policy_digest()),
            view_digest: Digest(view.seal_digest()),
            root_state: StateIdentity {
                owner: OwnerId(1),
                revision: StateRevision(root.snapshot().revision()),
                semantic: rz_position::contracts::rules_snapshot_semantic_digest(&root.snapshot()),
            },
            actual_tokens: view.token_count(),
            selected_records: view.records().len() as u16,
            anchors: view.anchors().len() as u8,
            length_bucket: view.length_bucket(),
        };
        let context = RoleLogicalContext {
            game_generation: 1,
            search_generation: stores.generation(),
            situation,
            state,
            focus,
            purpose: RoleQueryPurpose::ProposePolicy,
            prefix: prefix.clone(),
            focus_sha256: ordered_line_digest(b"focus", &[]).unwrap(),
            prefix_sha256: ordered_line_digest(b"prefix", &prefix).unwrap(),
            proposal_sha256: ordered_line_digest(b"proposal", &[]).unwrap(),
            refutation_sha256: counter
                .map(|line| ordered_line_digest(b"refutation", line).unwrap()),
            divergence_sha256: divergence_digest(&[]).unwrap(),
            public_revision: records.len() as u64,
            situation_revision: 0,
        };
        (
            stores,
            TicketPreparation {
                question: Digest([8; 32]),
                seal,
                role: Role::Proposer,
                deadline: Deadline {
                    clock: ClockDomain(authority.epoch),
                    at: MonotonicTick(10_000_000_000),
                },
                view,
                query: NativeTicketQuestion {
                    position: endpoint,
                    context,
                    deadline: Instant::now() + Duration::from_secs(10),
                    legacy_full_records: None,
                },
            },
        )
    }
    #[test]
    fn complete_legal_permutation_preserves_the_submitted_policy_order() {
        let (_store, ordered) = question(Position::startpos(), &[], &[]);
        let (_store, reversed) = question_with_options(Position::startpos(), &[], &[], None, true);
        let (ordered, _) = encode_view(&ordered, [1; 32], [2; 32]).unwrap();
        let (reversed, _) = encode_view(&reversed, [1; 32], [2; 32]).unwrap();
        assert_eq!(
            ordered.candidates.into_iter().rev().collect::<Vec<_>>(),
            reversed.candidates
        );
    }
    #[test]
    fn dispatch_recheck_distinguishes_stop_expiry_and_identity_failure() {
        let (_store, question) = question(Position::startpos(), &[], &[]);
        let (input, identity) = encode_view(&question, [1; 32], [2; 32]).unwrap();
        let clock = ContractSystemClock::new(question.seal.authority.epoch);
        let mut ticket = RoleTicketEnvelope {
            id: RequestId::new(question.seal.authority.epoch, 1),
            authority: question.seal.authority,
            question: question.question,
            view: question.seal,
            representation: representation(&input, &identity, identity.rules_history, 0).unwrap(),
            role: question.role,
            original_deadline: Deadline {
                clock: clock.domain(),
                at: MonotonicTick(u64::MAX),
            },
        };
        let canceled = AtomicBool::new(false);
        assert!(dispatch_allowed(&ticket, &clock, 128, &canceled).is_ok());
        canceled.store(true, Ordering::Release);
        assert_eq!(
            dispatch_allowed(&ticket, &clock, 128, &canceled),
            Err(RoleFailure::Canceled)
        );
        canceled.store(false, Ordering::Release);
        ticket.original_deadline.at = MonotonicTick(0);
        assert_eq!(
            dispatch_allowed(&ticket, &clock, 128, &canceled),
            Err(RoleFailure::Expired)
        );
        ticket.original_deadline.at = MonotonicTick(u64::MAX);
        ticket.role = Role::Critic;
        assert_eq!(
            dispatch_allowed(&ticket, &clock, 128, &canceled),
            Err(RoleFailure::InvalidOutput)
        );
    }
    #[test]
    fn empty_present_counter_is_not_an_absent_counter() {
        let (_store, absent) = question(Position::startpos(), &[], &[]);
        let (_store, present) = question_with_counter(Position::startpos(), &[], &[], Some(&[]));
        let (absent_input, absent_identity) = encode_view(&absent, [1; 32], [2; 32]).unwrap();
        let (present_input, present_identity) = encode_view(&present, [1; 32], [2; 32]).unwrap();
        assert!(absent_input.query.counter.is_empty());
        assert!(present_input.query.counter.is_empty());
        assert!(!absent_input.query.counter_present);
        assert!(present_input.query.counter_present);
        assert_ne!(
            absent_input.query.features(),
            present_input.query.features()
        );
        assert_ne!(
            absent_input.fresh_key(&absent_identity).unwrap(),
            present_input.fresh_key(&present_identity).unwrap()
        );
    }
    #[test]
    fn actual_prefix_endpoint_is_distinct_from_root_and_keeps_both_exact_histories() {
        let (_store, question) = question(Position::startpos(), &["e2e4"], &[]);
        let (input, identity) = encode_view(&question, [1; 32], [2; 32]).unwrap();
        assert_eq!(input.query.perspective, -1);
        assert_eq!(input.query.prefix.len(), 1);
        assert_eq!(input.token_budget, 128);
        assert_eq!(identity.rules_history, identity.anchor_history_seals[0]);
        let history = pals_native::pals_history_digest(&question.query.position).unwrap();
        assert_ne!(identity.rules_history, history);
        let key = representation(&input, &identity, history, 0).unwrap();
        assert_eq!(key.history.0, history);
        assert_eq!(key.input.0, input.fresh_key(&identity).unwrap().0);
        assert_eq!(
            input.candidates,
            question
                .view
                .query()
                .legal
                .iter()
                .copied()
                .map(pals_native::candidate)
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn actual_game_history_is_sealed_but_never_appended_as_review_tokens() {
        let mut root = Position::startpos();
        for text in ["e2e4", "e7e5", "g1f3", "b8c6"] {
            root.make_uci(text).unwrap();
        }
        let (_store, question) = question(root, &[], &[]);
        let (input, identity) = encode_view(&question, [1; 32], [2; 32]).unwrap();
        assert_eq!(input.validate().unwrap().token_count, 0);
        assert_eq!(
            identity.rules_history,
            pals_native::pals_history_digest(&question.query.position).unwrap()
        );
        assert_eq!(identity.anchor_history_seals.len(), 1);
        assert!(input.query.prefix.is_empty());
    }
    #[test]
    fn mismatched_endpoint_or_query_seal_is_rejected_before_native() {
        let (_store, mut question) = question(Position::startpos(), &["e2e4"], &[]);
        question.query.position = Position::startpos();
        assert!(matches!(
            encode_view(&question, [1; 32], [2; 32]),
            Err(RoleFailure::InvalidOutput)
        ));
        let (_store, mut question) = self::question(Position::startpos(), &[], &[]);
        question.query.context.proposal_sha256[0] ^= 1;
        assert!(matches!(
            encode_view(&question, [1; 32], [2; 32]),
            Err(RoleFailure::InvalidOutput)
        ));
    }
    #[test]
    fn terminal_feature_is_rules_replayed_not_inherited_from_cpu_scope() {
        let root = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
        let (_store, question) =
            question(root, &[], &[("f7g7", Some(CpuScoreScope::FrontierOnly))]);
        let (input, _) = encode_view(&question, [1; 32], [2; 32]).unwrap();
        assert!(input.records[0].rules_terminal);
        assert!(!input.records[0].completed_iteration);
        assert_eq!(input.records[0].features().len(), 8);
        assert_eq!(input.records[0].features()[6], 1.0);
    }
    #[test]
    fn wdl_uses_exact_input_side_perspective_and_rejects_nonfinite_logits() {
        assert_eq!(stable_wdl([0.0; 3]).unwrap(), [1.0 / 3.0; 3]);
        assert!(stable_wdl([f32::NAN, 0.0, 0.0]).is_err());
        let values = stable_wdl([100.0, 99.0, 98.0]).unwrap();
        assert!(values[0] > values[1] && values[1] > values[2]);
    }
    #[test]
    fn identical_board_metadata_with_different_exact_history_has_different_execution_keys() {
        let mut first = Position::startpos();
        let mut second = Position::startpos();
        for text in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            first.make_uci(text).unwrap();
        }
        for text in ["b1c3", "b8c6", "c3b1", "c6b8"] {
            second.make_uci(text).unwrap();
        }
        assert_eq!(first.to_fen(), second.to_fen());
        let (_store, first) = question(first, &[], &[]);
        let (_store, second) = question(second, &[], &[]);
        let (first_input, first_identity) = encode_view(&first, [1; 32], [2; 32]).unwrap();
        let (second_input, second_identity) = encode_view(&second, [1; 32], [2; 32]).unwrap();
        assert_eq!(first_input, second_input);
        assert_ne!(first_identity.rules_history, second_identity.rules_history);
        assert_ne!(
            first_input.fresh_key(&first_identity).unwrap(),
            second_input.fresh_key(&second_identity).unwrap()
        );
        assert_ne!(
            first_input.warm_context_key(&first_identity).unwrap(),
            second_input.warm_context_key(&second_identity).unwrap()
        );
    }
    #[test]
    fn old_idea_anchor_keeps_its_own_board_and_ordered_history() {
        let mut old = Position::startpos();
        for text in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            old.make_uci(text).unwrap();
        }
        let root = Position::from_fen(&old.to_fen()).unwrap();
        let mut stores = PalsStores::new(StoreLimits::default());
        let old_situation = stores.focus_actual_moves(old.snapshot()).unwrap();
        let origin = stores.situations.get(old_situation).unwrap().state;
        let current = stores.states.insert(root.snapshot()).unwrap();
        let mut index = ReviewContextIndex::default();
        let bundle = index
            .publish(
                &stores,
                &RoleRecord {
                    revision: 1,
                    parent_revision: None,
                    supersedes_revision: None,
                    origin_state: origin,
                    kind: RecordKind::Proposal,
                    line: vec![BoardMove::from_uci("e2e4").unwrap()],
                    value: None,
                    completed_depth: 0,
                    score_scope: None,
                    cpu_observation: None,
                    perspective: Color::White,
                    critical: true,
                },
                None,
            )
            .unwrap();
        let legal = root.legal_moves();
        let view = Arc::new(
            index
                .select(
                    &stores,
                    current,
                    ReviewQuerySequences {
                        purpose: RoleQueryPurpose::ProposePolicy,
                        prefix: &[],
                        proposal: &[],
                        counterexample: None,
                        legal: &legal,
                        divergences: &[],
                    },
                    ReviewSelectionPolicy {
                        mode: ReviewAnchorMode::AnchoredCompact,
                        max_anchors: 4,
                        required_bundle: Some(bundle),
                        token_budget: 128,
                        ..ReviewSelectionPolicy::default()
                    },
                )
                .unwrap(),
        );
        let (_other_store, mut question) = question(root, &[], &[]);
        question.query.context.public_revision = 1;
        question.query.context.search_generation = stores.generation();
        question.query.context.state = current;
        question.seal.view_sequence = view.sequence();
        question.seal.view_digest = Digest(view.seal_digest());
        question.seal.context_policy = Digest(view.policy_digest());
        question.seal.actual_tokens = view.token_count();
        question.seal.selected_records = view.records().len() as u16;
        question.seal.anchors = view.anchors().len() as u8;
        question.seal.length_bucket = view.length_bucket();
        question.view = view;
        let (input, identity) = encode_view(&question, [1; 32], [2; 32]).unwrap();
        assert_eq!(input.anchors.len(), 2);
        assert_eq!(input.records[0].anchor, 1);
        assert_eq!(identity.anchor_history_seals.len(), 2);
        assert_eq!(
            identity.anchor_history_seals[0],
            pals_native::pals_history_digest(&question.query.position).unwrap()
        );
        assert_eq!(
            identity.anchor_history_seals[1],
            pals_native::pals_history_digest(&old).unwrap()
        );
        assert_ne!(
            identity.anchor_history_seals[0],
            identity.anchor_history_seals[1]
        );
        let mut reordered = identity.clone();
        reordered.anchor_history_seals.swap(0, 1);
        assert!(input.fresh_key(&reordered).is_err());
    }
}
