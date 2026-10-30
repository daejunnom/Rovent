//! One bounded, exact-work NNUEC preparation. The Session keeps the single
//! preparation actor; this custody only borrows its nonblocking endpoints.

use super::*;
use rz_contracts::Digest;
use rz_search::cpu_value::nnuec::v2::Model;
use rz_search::cpu_value::nnuec::v2::review::{
    BoundNnuecReviewContext, Completion, NnuecPreparationTerminalAck, PreparationActor,
    PreparationError, PreparationObservation, Provenance, Request,
};
use rz_search::cpu_worker::{
    CpuWorker, CpuWorkerError, CpuWorkerProfile, CpuWorkerShutdown, InstallId,
    PreparedValueInstallation, ValueInstallReceipt,
};
use rz_search::pals::coordinator::CoordinatorWork;
use rz_search::pals::engine::adaptive_port::AdaptiveNnuecReviewSeal;
use rz_search::pals::review_context::ReviewView;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug)]
pub enum NnuecWorkError {
    Preparation(PreparationError),
    Worker(CpuWorkerError),
    Invalid(&'static str),
}

/// Same-question preparation telemetry. Actor wall service is distinct from
/// process/thread CPU time and from every P/C ticket or native node charge.
#[derive(Clone, Debug)]
pub struct NnuecPreparationRecord {
    pub frame: FrameId,
    pub question: Digest,
    pub view: ReviewViewSeal,
    pub owner: CpuWorkerId,
    pub submitted: bool,
    pub actor_terminal: Option<PreparationObservation>,
    pub cpu_thread_time: Option<Duration>,
    pub cancellation_requested: bool,
    pub manager_error: Option<NnuecWorkError>,
    /// Read-only scalar copy of this preparation's actual acknowledgement.
    /// No Bound/view/worker authority is cloned into diagnostic history.
    #[cfg(feature = "pals-adaptive-diagnostics")]
    pub original_ack: NnuecPreparationAckObservation,
}

#[cfg(feature = "pals-adaptive-diagnostics")]
#[derive(Clone, Debug)]
pub struct NnuecPreparationAckObservation {
    pub stage: String,
    pub actor_terminal_ack_observed: bool,
    pub driver_attested: bool,
    pub installation_requested_sequence: Option<u64>,
    pub installation_receipt_sequence: Option<u64>,
    pub installation_succeeded: Option<bool>,
    pub binding: Option<Digest>,
    pub installed_owner: Option<CpuWorkerId>,
    pub actual_stack_owner_id: Option<u64>,
}

/// The same joined worker's engine, to be moved to the Session's retained actor.
/// Initial Ready has no replacement receipt; it is deliberately not fabricated.
pub struct AdaptiveNnuecRetiredOwner {
    pub engine: rz_search::cpu_worker::RecoveredNnuecEngine,
    pub last_installation: Option<ValueInstallReceipt>,
}

pub struct AdaptiveNnuecRootParts<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    pub native: TicketedNativeService<ReviewView, Q, B>,
    pub cpu: Vec<AdaptiveNnuecRetiredOwner>,
    pub external: Option<ExternalManagerWorker>,
    pub driver: D,
    pub continuation: AdaptiveSequenceContinuation,
    pub receipt: AdaptiveShutdownReceipt,
    pub preparation_observations: Vec<NnuecPreparationRecord>,
    pub preparation_observations_omitted: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Unsubmitted,
    Submitted,
    Bound,
    Payload,
    AwaitingInstallation,
    Installed,
    Failed,
}

/// A single root declaration, without a worker, TT, encoder or actor clone.
pub(super) struct NnuecRootPreparation {
    pub model: Arc<Model>,
    pub canceled: Arc<AtomicBool>,
}

/// The frame stays Ready and retains its original fair-queue descriptor while
/// this custody blocks only its selected idle owner. It never moves a token.
pub(super) struct NnuecWorkPreparation<Q> {
    pub work: CoordinatorWork<ReviewView, Q>,
    pub selection:
        rz_runtime::adaptive::ReadySelection<FrameId, rz_contracts::pals::Move16, Digest>,
    pub slot: usize,
    owner: CpuWorkerId,
    pub seal: AdaptiveNnuecReviewSeal,
    expected: Provenance,
    canceled: Arc<AtomicBool>,
    phase: Phase,
    actor_outstanding: bool,
    installation_outstanding: bool,
    request: Option<Request>,
    bound: Option<Arc<BoundNnuecReviewContext>>,
    payload: Option<PreparedValueInstallation>,
    installation: Option<(InstallId, Digest, CpuWorkerProfile)>,
    receipt: Option<ValueInstallReceipt>,
    // A mismatched completion is evidence of a protocol error. Keep it rather
    // than silently treating the expected actor request as physically settled.
    unmatched_completion: Option<Completion>,
    failure: Option<NnuecWorkError>,
    attested: bool,
    submitted: bool,
    observation: Option<PreparationObservation>,
    terminal_ack: Option<NnuecPreparationTerminalAck>,
}

pub(super) struct NnuecJobCustody {
    pub seal: AdaptiveNnuecReviewSeal,
    pub bound: Arc<BoundNnuecReviewContext>,
    pub receipt: ValueInstallReceipt,
    pub capture: Option<rz_search::cpu_value::nnuec::v2::capture::NnuecCaptureReservation>,
    // The original actor terminal stays with its exact request/job custody.
    _terminal_ack: NnuecPreparationTerminalAck,
}

impl<Q> NnuecWorkPreparation<Q> {
    pub fn new(
        work: CoordinatorWork<ReviewView, Q>,
        selection: rz_runtime::adaptive::ReadySelection<
            FrameId,
            rz_contracts::pals::Move16,
            Digest,
        >,
        slot: usize,
        owner: CpuWorkerId,
        original: (Request, AdaptiveNnuecReviewSeal),
        canceled: Arc<AtomicBool>,
    ) -> Result<Self, NnuecWorkError> {
        let (request, seal) = original;
        let expected = request.provenance().clone();
        if selection.job.job != work.frame
            || selection.job.candidate != work.original_first_move
            || selection.job.hard_deadline != Duration::from_nanos(work.original_deadline.at.0)
            || seal.frame() != work.frame
            || seal.authority() != work.frame.authority
            || seal.question_digest() != work.input.question_digest
            || seal.full_view_digest().0 != work.input.view.digest()
            || !Arc::ptr_eq(seal.view(), &work.input.view)
            || seal.model() != &expected.model
            || expected.sequence != work.input.view.sequence()
            || expected.view_seal != work.input.view.seal_digest()
        {
            return Err(NnuecWorkError::Invalid(
                "NNUEC work differs from original admission",
            ));
        }
        Ok(Self {
            work,
            selection,
            slot,
            owner,
            seal,
            expected,
            canceled,
            phase: Phase::Unsubmitted,
            actor_outstanding: false,
            installation_outstanding: false,
            request: Some(request),
            bound: None,
            payload: None,
            installation: None,
            receipt: None,
            unmatched_completion: None,
            failure: None,
            attested: false,
            submitted: false,
            observation: None,
            terminal_ack: None,
        })
    }

    /// Encoding and C forward occur only on the Session-owned actor. A full
    /// actor returns the same original Request and selected-view pin.
    pub fn pump_actor(
        &mut self,
        actor: &PreparationActor,
        may_submit: bool,
    ) -> Result<bool, NnuecWorkError> {
        if self.phase == Phase::Unsubmitted {
            if !may_submit || self.canceled.load(Ordering::Acquire) {
                return Ok(false);
            }
            let request = self.request.take().ok_or(NnuecWorkError::Invalid(
                "NNUEC original request custody missing",
            ))?;
            match actor.try_submit(request) {
                Ok(()) => {
                    self.actor_outstanding = true;
                    self.submitted = true;
                    self.phase = Phase::Submitted;
                    return Ok(true);
                }
                Err((PreparationError::Busy, request)) => {
                    self.request = Some(request);
                    return Ok(false);
                }
                Err((error, request)) => {
                    self.request = Some(request);
                    let error = NnuecWorkError::Preparation(error);
                    self.failure = Some(error.clone());
                    self.phase = Phase::Failed;
                    return Err(error);
                }
            }
        }
        if self.phase != Phase::Submitted {
            return Ok(false);
        }
        let Some(completion) = actor.try_complete().map_err(NnuecWorkError::Preparation)? else {
            return Ok(false);
        };
        if completion.provenance != self.expected {
            self.unmatched_completion = Some(completion);
            let error = NnuecWorkError::Invalid("NNUEC actor returned another original question");
            self.failure = Some(error.clone());
            self.phase = Phase::Failed;
            return Err(error);
        }
        self.actor_outstanding = false;
        self.observation = Some(completion.observation);
        self.terminal_ack = Some(completion.terminal_ack);
        match completion.result {
            Ok(bound) => {
                if bound.provenance() != &self.expected {
                    let error = NnuecWorkError::Invalid("NNUEC bound provenance differs");
                    self.bound = Some(Arc::new(bound));
                    self.failure = Some(error.clone());
                    self.phase = Phase::Failed;
                    return Err(error);
                }
                self.bound = Some(Arc::new(bound));
                self.phase = Phase::Bound;
                Ok(true)
            }
            Err(error) => {
                let error = NnuecWorkError::Preparation(error);
                self.failure = Some(error.clone());
                self.phase = Phase::Failed;
                Err(error)
            }
        }
    }

    /// The driver must validate this bound against the original semantic
    /// Store and registered model before constructing an installation payload.
    pub fn prepared_bound(&self) -> Option<&Arc<BoundNnuecReviewContext>> {
        (self.phase == Phase::Bound)
            .then_some(self.bound.as_ref())
            .flatten()
    }

    pub fn stage_installation(
        &mut self,
        payload: PreparedValueInstallation,
    ) -> Result<(), (NnuecWorkError, PreparedValueInstallation)> {
        if self.phase != Phase::Bound
            || payload.provenance() != &self.expected
            || payload.value_identity() != &self.expected.model
            || self.canceled.load(Ordering::Acquire)
        {
            return Err((
                NnuecWorkError::Invalid("NNUEC installation differs from original bound"),
                payload,
            ));
        }
        self.payload = Some(payload);
        self.phase = Phase::Payload;
        Ok(())
    }

    pub fn try_install(&mut self, worker: &CpuWorker) -> Result<bool, NnuecWorkError> {
        if self.phase != Phase::Payload || self.canceled.load(Ordering::Acquire) {
            return Ok(false);
        }
        let previous = worker
            .acknowledged_profile()
            .map_err(NnuecWorkError::Worker)?;
        let payload = self.payload.take().ok_or(NnuecWorkError::Invalid(
            "NNUEC original installation payload missing",
        ))?;
        let binding = payload.binding_digest();
        match worker.try_install_prepared_value(payload) {
            Ok(id) => {
                self.installation = Some((id, binding, previous));
                self.installation_outstanding = true;
                self.phase = Phase::AwaitingInstallation;
                Ok(true)
            }
            Err(rejected) => {
                let (error, payload) = rejected.into_parts();
                self.payload = Some(payload);
                if matches!(error, CpuWorkerError::Capacity) {
                    Ok(false)
                } else {
                    let error = NnuecWorkError::Worker(error);
                    self.failure = Some(error.clone());
                    self.phase = Phase::Failed;
                    Err(error)
                }
            }
        }
    }

    pub fn poll_installation(&mut self, worker: &mut CpuWorker) -> Result<bool, NnuecWorkError> {
        if self.phase != Phase::AwaitingInstallation {
            return Ok(false);
        }
        let Some(receipt) = worker
            .try_install_readback()
            .map_err(NnuecWorkError::Worker)?
        else {
            return Ok(false);
        };
        let matches = self.matches_installation(&receipt);
        let live = matches
            && !self.canceled.load(Ordering::Acquire)
            && receipt.is_installed()
            && self
                .bound
                .as_ref()
                .is_some_and(|bound| receipt.matches_bound(bound))
            && worker.validates_installation_receipt(&receipt);
        self.receipt = Some(receipt);
        if !matches {
            self.phase = Phase::Failed;
            return Err(NnuecWorkError::Invalid(
                "NNUEC installation receipt differs",
            ));
        }
        self.installation_outstanding = false;
        if self.canceled.load(Ordering::Acquire) {
            // The matching owner terminal is sufficient for cleanup. A prior
            // logical cancellation prevents all new Job admission, regardless
            // of whether the replacement itself finished successfully.
            self.phase = Phase::Failed;
            return Ok(true);
        }
        if let Err(error) = self
            .receipt
            .as_ref()
            .expect("actual receipt retained")
            .result()
        {
            let error = NnuecWorkError::Worker(error.clone());
            self.failure = Some(error.clone());
            self.phase = Phase::Failed;
            return Err(error);
        }
        if !live {
            let error = NnuecWorkError::Invalid("NNUEC installation has no live admission");
            self.failure = Some(error.clone());
            self.phase = Phase::Failed;
            return Err(error);
        }
        self.phase = Phase::Installed;
        Ok(true)
    }

    /// A shutdown readback can settle cleanup custody only. It never restores
    /// the live worker authority needed to issue another CPU Job.
    pub fn observe_shutdown(&mut self, report: &CpuWorkerShutdown) -> Result<(), NnuecWorkError> {
        if self.phase != Phase::AwaitingInstallation {
            return Ok(());
        }
        if let Some(receipt) = &report.value_installation_receipt {
            self.receipt = Some(receipt.clone());
        }
        if let Some(error) = &report.value_installation_error {
            return Err(NnuecWorkError::Worker(error.clone()));
        }
        if let Some(receipt) = &report.value_installation_receipt {
            if !self.matches_installation(receipt) {
                return Err(NnuecWorkError::Invalid(
                    "joined owner returned another installation",
                ));
            }
            self.installation_outstanding = false;
            self.phase = Phase::Failed;
        }
        Ok(())
    }

    pub fn installed(&self) -> Option<(&Arc<BoundNnuecReviewContext>, &ValueInstallReceipt)> {
        if self.phase != Phase::Installed || self.canceled.load(Ordering::Acquire) {
            return None;
        }
        Some((self.bound.as_ref()?, self.receipt.as_ref()?))
    }

    pub fn job_ready(&self) -> bool {
        self.attested && self.installed().is_some() && self.terminal_ack.is_some()
    }

    pub fn cleanup_settled(&self) -> bool {
        !self.actor_outstanding
            && !self.installation_outstanding
            && self.unmatched_completion.is_none()
    }

    pub fn failure(&self) -> Option<&NnuecWorkError> {
        self.failure.as_ref()
    }
    pub fn observation(&self) -> NnuecPreparationRecord {
        NnuecPreparationRecord {
            frame: self.work.frame,
            question: self.work.input.question_digest,
            view: self.work.input.view_seal,
            owner: self.owner,
            submitted: self.submitted,
            actor_terminal: self.observation,
            // No thread/process CPU clock witness is available here.
            cpu_thread_time: None,
            cancellation_requested: self.canceled.load(Ordering::Acquire),
            manager_error: self.failure.clone(),
            #[cfg(feature = "pals-adaptive-diagnostics")]
            original_ack: NnuecPreparationAckObservation {
                stage: format!("{:?}", self.work.input.stage)
                    .chars()
                    .take(32)
                    .collect(),
                actor_terminal_ack_observed: self.terminal_ack.is_some(),
                driver_attested: self.attested,
                installation_requested_sequence: self.installation.map(|(id, _, _)| id.sequence()),
                installation_receipt_sequence: self
                    .receipt
                    .as_ref()
                    .map(|r| r.installation_id().sequence()),
                installation_succeeded: self
                    .receipt
                    .as_ref()
                    .map(ValueInstallReceipt::is_installed),
                binding: self
                    .receipt
                    .as_ref()
                    .map(ValueInstallReceipt::binding_digest),
                installed_owner: self
                    .receipt
                    .as_ref()
                    .and_then(ValueInstallReceipt::installed_profile)
                    .map(|p| p.owner),
                actual_stack_owner_id: self
                    .receipt
                    .as_ref()
                    .and_then(|r| r.result().ok())
                    .map(|r| r.actual_stack_owner_id()),
            },
        }
    }

    // The manager has one pending preparation. A not-ready transfer returns
    // that exact Request/Bound/ACK owner without allocating on failure.
    #[allow(clippy::result_large_err)]
    pub fn into_job_custody(self) -> Result<NnuecJobCustody, Self> {
        if !self.job_ready() {
            return Err(self);
        }
        Ok(NnuecJobCustody {
            seal: self.seal,
            bound: self.bound.expect("installed original Bound"),
            receipt: self.receipt.expect("installed actual owner receipt"),
            capture: None,
            _terminal_ack: self.terminal_ack.expect("original actor terminal"),
        })
    }

    fn matches_installation(&self, receipt: &ValueInstallReceipt) -> bool {
        self.installation.is_some_and(|(id, binding, previous)| {
            receipt.installation_id() == id
                && receipt.binding_digest() == binding
                && receipt.previous_profile() == previous
        })
    }
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    /// Startup declaration only. The existing Ready worker remains the same
    /// owner; no model preparation, TT allocation or worker creation occurs.
    pub fn enable_nnuec_preparation(
        &mut self,
        model: Arc<Model>,
    ) -> Result<(), AdaptiveManagerError> {
        self.check_running()?;
        // This preparation installs the engine's native R evaluator. Collection
        // may independently use a foreign teacher, or leave CPU_T disabled when
        // that teacher is owned by the outer collector.
        let endpoint = self.registration.endpoints.cpu_r;
        if self.root_started
            || self.nnuec.is_some()
            || self.nnuec_pending.is_some()
            || endpoint != CpuEndpoint::Own
            || self.driver.rovent_root_budget()?.is_none()
        {
            return Err(AdaptiveManagerError::Invalid(
                "NNUEC preparation idle root registration",
            ));
        }
        self.driver.validate_cpu_nnuec_model(&model)?;
        self.nnuec = Some(NnuecRootPreparation {
            model,
            canceled: Arc::new(AtomicBool::new(false)),
        });
        Ok(())
    }

    /// Called by the Session's existing before/after-tick observer, after its
    /// original UCI cancel/replacement/deadline checks. The actor itself stays
    /// with that Session until all CPU owners have drained and joined.
    pub fn pump_nnuec_preparation(
        &mut self,
        actor: &PreparationActor,
    ) -> Result<bool, AdaptiveManagerError> {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if self.phase == ManagerPhase::Running && self.cohort_action_expired(Instant::now()) {
            self.begin_root_quiesce(self.cleanup_until()?)?;
        }
        let utility_changed = match self.pump_utility_preparation(actor) {
            Ok(changed) => changed,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Err(AdaptiveManagerError::AdmissionClosed)
                if self.cohort_action_expired(Instant::now()) =>
            {
                self.begin_root_quiesce(self.cleanup_until()?)?;
                return Ok(true);
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            Err(error @ AdaptiveManagerError::InterventionNeutralMismatch { .. }) => {
                return self.quiesce_neutral_mismatch(error);
            }
            Err(error) => return Err(error),
        };
        // The utility P/C preparation is the same actor's sole request. It
        // cannot compete with or drain a separate original R completion.
        let may_submit = self.phase == ManagerPhase::Running && self.can_admit_work();
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if self.cohort_cache_request_pending() && may_submit {
            if let Some(request) = self
                .nnuec_pending
                .as_mut()
                .and_then(|pending| pending.request.take())
            {
                let request = match self.attach_cohort_cache_prelude(request) {
                    Ok(request) => request,
                    Err((AdaptiveManagerError::AuxiliaryPending, request)) => {
                        self.nnuec_pending
                            .as_mut()
                            .expect("original R preparation")
                            .request = Some(request);
                        return Ok(utility_changed);
                    }
                    Err((error, request)) => {
                        self.nnuec_pending
                            .as_mut()
                            .expect("original R preparation")
                            .request = Some(request);
                        return Err(error);
                    }
                };
                self.nnuec_pending
                    .as_mut()
                    .expect("original R preparation")
                    .request = Some(request);
            }
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let may_submit = may_submit && !self.cohort_cache_request_pending();
        let Some(pending) = &mut self.nnuec_pending else {
            return Ok(utility_changed);
        };
        let closing = self.phase != ManagerPhase::Running;
        let mut changed = utility_changed
            | match pending.pump_actor(actor, may_submit) {
                Ok(changed) => changed,
                Err(NnuecWorkError::Preparation(
                    PreparationError::Cancelled | PreparationError::Expired,
                )) if closing => true,
                Err(error) => return Err(AdaptiveManagerError::NnuecPreparation(error)),
            };
        let utility_bound = pending.prepared_bound().cloned();
        if let Some(bound) = utility_bound.filter(|_| !closing) {
            // Release the mutable pending borrow before the utility owner
            // validates the exact same Bound/input capture.
            let _ = pending;
            match self.utility_bind_bound(bound) {
                Ok(bound) => changed |= bound,
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                Err(AdaptiveManagerError::AdmissionClosed)
                    if self.cohort_action_expired(Instant::now()) =>
                {
                    self.begin_root_quiesce(self.cleanup_until()?)?;
                    return Ok(true);
                }
                Err(error) => return Err(error),
            }
        }
        let pending = self.nnuec_pending.as_mut().expect("original R preparation");
        if closing {
            if let Some(worker) = self.own[pending.slot].worker.as_mut() {
                changed |= pending
                    .poll_installation(worker)
                    .map_err(AdaptiveManagerError::NnuecPreparation)?;
            }
            if pending.cleanup_settled()
                && self.driver.semantic_owner_available()
                && self.driver.prepare_semantic_commit()?
            {
                self.driver.release_cpu_nnuec_installation(&pending.seal)?;
                let observation = pending.observation();
                let work = pending.work.clone();
                let input = pending
                    .bound
                    .as_ref()
                    .and_then(|bound| bound.utility_input_capture().cloned());
                let terminal = pending.terminal_ack.take();
                let actor_observation = pending.observation;
                let submitted = pending.submitted;
                self.retain_utility_preparation_cleanup(
                    &work,
                    input,
                    terminal,
                    actor_observation,
                    submitted,
                )?;
                self.nnuec_pending = None;
                self.retain_nnuec_observation(observation);
                changed = true;
            }
            return Ok(changed);
        }
        if !self.driver.semantic_owner_available() || !self.driver.prepare_semantic_commit()? {
            return Ok(changed);
        }
        if let Some(bound) = pending.prepared_bound().filter(|_| may_submit).cloned() {
            let model = Arc::clone(
                &self
                    .nnuec
                    .as_ref()
                    .ok_or(AdaptiveManagerError::Invalid(
                        "NNUEC original root model missing",
                    ))?
                    .model,
            );
            let payload = self
                .driver
                .install_cpu_nnuec_review(&pending.seal, model, bound)?;
            if let Err((error, payload)) = pending.stage_installation(payload) {
                pending.payload = Some(payload);
                pending.phase = Phase::Failed;
                pending.failure = Some(error.clone());
                return Err(AdaptiveManagerError::NnuecPreparation(error));
            }
            changed = true;
        }
        let slot = &mut self.own[pending.slot];
        if slot.binding.is_some() || slot.joined || slot.panicked {
            return Err(AdaptiveManagerError::Invalid(
                "NNUEC selected owner acquired another Job",
            ));
        }
        let worker = slot.worker.as_mut().ok_or(AdaptiveManagerError::Invalid(
            "NNUEC selected owner missing",
        ))?;
        if may_submit {
            changed |= pending
                .try_install(worker)
                .map_err(AdaptiveManagerError::NnuecPreparation)?;
        }
        changed |= pending
            .poll_installation(worker)
            .map_err(AdaptiveManagerError::NnuecPreparation)?;
        if let Some((bound, receipt)) = pending.installed().filter(|_| !pending.attested) {
            let model = Arc::clone(
                &self
                    .nnuec
                    .as_ref()
                    .ok_or(AdaptiveManagerError::Invalid(
                        "NNUEC original root model missing",
                    ))?
                    .model,
            );
            self.driver.accept_cpu_nnuec_installation(
                &pending.seal,
                model,
                Arc::clone(bound),
                receipt,
                worker,
            )?;
            slot.last_value_installation = Some(receipt.clone());
            pending.attested = true;
            changed = true;
        }
        Ok(changed)
    }

    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(super) fn quiesce_neutral_mismatch(
        &mut self,
        error: AdaptiveManagerError,
    ) -> Result<bool, AdaptiveManagerError> {
        if !matches!(error, AdaptiveManagerError::InterventionNeutralMismatch { .. }) {
            return Err(error);
        }
        // A non-comparable target is an actual failed trial, not a physical
        // completion failure. Preserve its cause and rejected owner, then
        // drain the original actors before attempting any physical proof.
        self.record_failure(error);
        self.begin_root_quiesce(self.cleanup_until()?)?;
        Ok(true)
    }

    pub(super) fn start_nnuec_work(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        slot: usize,
        selection: rz_runtime::adaptive::ReadySelection<
            FrameId,
            rz_contracts::pals::Move16,
            Digest,
        >,
    ) -> Result<(), AdaptiveManagerError> {
        if self.nnuec_pending.is_some()
            || self.own[slot].binding.is_some()
            || self.own[slot].joined
            || self.own[slot].panicked
        {
            return Err(AdaptiveManagerError::Invalid(
                "NNUEC requires original idle owner",
            ));
        }
        let original = self
            .nnuec
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid("NNUEC root not registered"))?;
        let request = self.driver.prepare_cpu_nnuec_review(
            work,
            &original.model,
            Arc::clone(&original.canceled),
        )?;
        let canceled = Arc::clone(&original.canceled);
        let (request, seal) = request.into_parts();
        let request = self.utility_attach_r_request(work, request)?;
        self.nnuec_pending = Some(
            NnuecWorkPreparation::new(
                work.clone(),
                selection,
                slot,
                self.own[slot]
                    .worker
                    .as_ref()
                    .ok_or(AdaptiveManagerError::Invalid(
                        "NNUEC original owner missing",
                    ))?
                    .profile()
                    .owner,
                (request, seal),
                canceled,
            )
            .map_err(AdaptiveManagerError::NnuecPreparation)?,
        );
        Ok(())
    }

    pub(super) fn nnuec_work_eligible(&self, frame: FrameId) -> bool {
        self.nnuec_pending.as_ref().is_none_or(|pending| {
            pending.work.frame == frame
                && pending.job_ready()
                && self.own[pending.slot].binding.is_none()
                && !self.own[pending.slot].joined
        })
    }

    pub(super) fn nnuec_cleanup_settled(&self) -> bool {
        self.nnuec_pending
            .as_ref()
            .is_none_or(NnuecWorkPreparation::cleanup_settled)
    }

    /// Readback for the Session's final physical engine retirement join. These
    /// receipts are observations of the original owner; cloning grants no new
    /// installation or evaluator authority.
    pub fn nnuec_installation_receipts(&self) -> Vec<(CpuWorkerId, ValueInstallReceipt)> {
        self.own
            .iter()
            .filter_map(|slot| {
                let receipt = slot.last_value_installation.as_ref()?;
                Some((receipt.installed_profile()?.owner, receipt.clone()))
            })
            .collect()
    }
    pub fn nnuec_preparation_failure(&self) -> Option<&NnuecWorkError> {
        self.nnuec_pending
            .as_ref()
            .and_then(NnuecWorkPreparation::failure)
    }
    /// Bounded scalar/error records only. No view, bound, model, or large
    /// statistics snapshot is cloned by the manager event loop.
    pub fn nnuec_preparation_observations(
        &self,
    ) -> impl Iterator<Item = NnuecPreparationRecord> + '_ {
        self.nnuec_observations.iter().cloned().chain(
            self.nnuec_pending
                .as_ref()
                .map(NnuecWorkPreparation::observation),
        )
    }
    /// An explicit telemetry gap, never an additional search or preparation
    /// budget. Admission is independent of this bounded detail buffer.
    pub fn nnuec_preparation_observations_omitted(&self) -> u64 {
        self.nnuec_observations_omitted
    }
    fn retain_nnuec_observation(&mut self, observation: NnuecPreparationRecord) {
        if self.nnuec_observations.len() == CPU_JOB_CAPACITY {
            self.nnuec_observations.remove(0);
            self.nnuec_observations_omitted = self.nnuec_observations_omitted.saturating_add(1);
        }
        self.nnuec_observations.push(observation);
    }

    /// This transfer does not mark the evaluator/cache released. The Session
    /// must obtain its same actor's actual engine and cache acknowledgements
    /// before reusing the semantic owner in a subsequent root.
    #[allow(clippy::result_large_err)]
    pub fn into_nnuec_root_parts(
        mut self,
    ) -> Result<AdaptiveNnuecRootParts<Q, B, D>, AdaptivePersistentRejection<Q, B, D>> {
        let receipt = match self.shutdown_receipt() {
            Ok(receipt)
                if self.nnuec.is_some()
                    && self.cpu_semantic.is_none()
                    && self.utility_route.is_none()
                    && {
                        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                        {
                            self.utility_cohort
                                .as_ref()
                                .is_none_or(|state| !state.observer_retained())
                        }
                        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
                        {
                            true
                        }
                    }
                    && self.retained_cpu_jobs.is_empty()
                    && self.phase == ManagerPhase::Finished
                    && self.close_mode == AdaptiveCloseMode::RootQuiesce
                    && receipt.permits_root_reuse() =>
            {
                receipt
            }
            Ok(_) => {
                return Err(AdaptivePersistentRejection {
                    error: AdaptiveManagerError::Invalid(
                        "NNUEC transfer requires original finished root fence",
                    ),
                    manager: self,
                });
            }
            Err(error) => {
                return Err(AdaptivePersistentRejection {
                    error,
                    manager: self,
                });
            }
        };
        for index in 0..self.own.len() {
            if self.own[index].nnuec_retirement.is_some() {
                continue;
            }
            if self.own[index].recovered_engine.is_some() {
                return Err(AdaptivePersistentRejection {
                    error: AdaptiveManagerError::Invalid(
                        "NNUEC original engine already transferred through legacy path",
                    ),
                    manager: self,
                });
            }
            let Some(worker) = self.own[index].worker.as_mut() else {
                return Err(AdaptivePersistentRejection {
                    error: AdaptiveManagerError::Invalid("NNUEC joined original worker missing"),
                    manager: self,
                });
            };
            match worker.take_nnuec_retirement() {
                Ok(engine) => self.own[index].nnuec_retirement = Some(engine),
                Err(error) => {
                    return Err(AdaptivePersistentRejection {
                        error: error.into(),
                        manager: self,
                    });
                }
            }
        }
        let cpu = self
            .own
            .iter_mut()
            .map(|slot| AdaptiveNnuecRetiredOwner {
                engine: slot
                    .nnuec_retirement
                    .take()
                    .expect("all original retirement owners validated"),
                last_installation: slot.last_value_installation.take(),
            })
            .collect();
        let continuation = AdaptiveSequenceContinuation {
            authority: self.registration.authority,
            next_request: self.registry.next_request_sequence(),
            next_job: self.next_job,
            next_execution: self.next_execution,
            next_control: self.next_control,
        };
        Ok(AdaptiveNnuecRootParts {
            native: self.native,
            cpu,
            external: self.external,
            driver: self.driver,
            continuation,
            receipt,
            preparation_observations: self.nnuec_observations,
            preparation_observations_omitted: self.nnuec_observations_omitted,
        })
    }

    /// Transfer only an acknowledged original preparation into this exact Job.
    /// A second root-level reference protects all error/raw-report paths until
    /// the semantic installation owner confirms terminal/no-grant/no-token.
    pub(super) fn claim_nnuec_job(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        slot: usize,
    ) -> Result<Option<Arc<NnuecJobCustody>>, AdaptiveManagerError> {
        if self.nnuec.is_none() {
            return Ok(None);
        }
        if self.nnuec_jobs.len() >= CPU_JOB_CAPACITY {
            return Err(AdaptiveManagerError::Capacity("NNUEC retained Job custody"));
        }
        let pending = self
            .nnuec_pending
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "NNUEC original installed work missing",
            ))?;
        if pending.slot != slot
            || !pending.job_ready()
            || pending.work.frame != work.frame
            || pending.work.original_deadline != work.original_deadline
            || pending.work.input.stage != work.input.stage
            || pending.work.input.question_digest != work.input.question_digest
            || pending.work.input.view_seal != work.input.view_seal
            || !Arc::ptr_eq(&pending.work.input.view, &work.input.view)
            || !Arc::ptr_eq(&pending.work.input.question, &work.input.question)
            || !self.own[slot].worker.as_ref().is_some_and(|worker| {
                pending
                    .installed()
                    .is_some_and(|(_, receipt)| worker.validates_installation_receipt(receipt))
            })
        {
            return Err(AdaptiveManagerError::Invalid(
                "NNUEC Job differs from acknowledged work",
            ));
        }
        let pending = self
            .nnuec_pending
            .take()
            .expect("validated original preparation");
        let capture = match self.reserve_nnuec_job_capture(
            work,
            &pending.seal,
            pending.bound.as_ref().expect("installed original Bound"),
        ) {
            Ok(capture) => capture,
            Err(error) => {
                self.nnuec_pending = Some(pending);
                return Err(error);
            }
        };
        let observation = pending.observation();
        let custody = match pending.into_job_custody() {
            Ok(mut custody) => {
                custody.capture = capture;
                Arc::new(custody)
            }
            Err(pending) => {
                self.nnuec_pending = Some(pending);
                return Err(AdaptiveManagerError::Invalid(
                    "NNUEC preparation no longer live",
                ));
            }
        };
        self.retain_nnuec_observation(observation);
        self.nnuec_jobs.push(Arc::clone(&custody));
        Ok(Some(custody))
    }

    /// Extra semantic pins are released only after actual owner joins and
    /// known receipt settlement. This is separate from engine/cache release.
    pub(super) fn release_nnuec_custody(&mut self) -> Result<(), AdaptiveManagerError> {
        if !self.nnuec_cleanup_settled() {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        for custody in &self.nnuec_jobs {
            self.driver.release_cpu_nnuec_installation(&custody.seal)?;
        }
        self.nnuec_jobs.clear();
        if let Some(pending) = &self.nnuec_pending {
            self.driver.release_cpu_nnuec_installation(&pending.seal)?;
            let observation = pending.observation();
            self.retain_nnuec_observation(observation);
        }
        self.nnuec_pending = None;
        Ok(())
    }

    pub(super) fn release_nnuec_report(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        report: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        if report.terminal && !report.continuation_retained {
            self.release_nnuec_terminal(work, report.grant.job)?;
        }
        Ok(())
    }

    pub(super) fn release_nnuec_terminal(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        job: CpuJobId,
    ) -> Result<(), AdaptiveManagerError> {
        let Some(index) = self.nnuec_jobs.iter().position(|custody| {
            custody.seal.frame() == work.frame
                && custody.seal.question_digest() == work.input.question_digest
                && Arc::ptr_eq(custody.seal.view(), &work.input.view)
        }) else {
            return Ok(());
        };
        let (_, reserved, unknown) = self.native_ledger()?.usage(job)?;
        if reserved != 0 || unknown {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        self.driver
            .release_cpu_nnuec_installation(&self.nnuec_jobs[index].seal)?;
        self.nnuec_jobs.remove(index);
        Ok(())
    }
}
