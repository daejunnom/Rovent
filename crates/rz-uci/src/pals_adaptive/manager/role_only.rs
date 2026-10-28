//! Actual P/C-only ticket manager. There is no CPU worker, native-node ledger,
//! adaptive CPU controller, R frame machine, evaluator or stack in this owner.
//! Encoding and inference stay on the existing single native service worker.
use super::collection::{AdaptiveCollectionObserver, RoventCollectionRootAdmission};
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
use super::collection::{
    AdaptiveCollectionObserverBoxV2, RoleOnlySearchConsumptionV2, RoventCollectionRootAdmissionV2,
};
use super::*;
use rz_search::cpu_value::nnuec::v2::Model as SharedNnuecModel;
use rz_search::cpu_value::nnuec::v2::retirement::{NnuecCacheInvalidation, NnuecCacheReleaseAck};
use rz_search::cpu_value::nnuec::v2::review::{
    Completion as SharedNnuecCompletion, PreparationActor, PreparationError,
    Request as SharedNnuecRequest,
};
use rz_search::pals::coordinator::AcceptedRoleStep;
use rz_search::pals::engine::NoResearcher;
use rz_search::pals::engine::pc_port::{
    PcNnuecReadoutObservation, PcNnuecReadoutRequest, PcNnuecReadoutSeal, PcNnuecReadoutTerminal,
    PcSharedReadoutRootClosure,
};

pub trait RoleOnlySearchDriver<Q> {
    fn researcher(&self) -> &NoResearcher;
    fn validate_registration(
        &self,
        authority: SearchAuthority,
        mode: ExecutionMode,
        hard_deadline: Instant,
        root_budget: Digest,
    ) -> Result<(), AdaptiveManagerError>;
    fn next_role(&mut self) -> Result<Option<FrameInput<ReviewView, Q>>, AdaptiveManagerError>;
    fn register_shared_nnuec_readout(
        &mut self,
        _: Arc<SharedNnuecModel>,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no shared P/C NNUEC readout",
        ))
    }
    fn requires_shared_nnuec_readout(&self) -> bool {
        false
    }
    fn prepare_shared_nnuec_readout(
        &mut self,
        _: &FrameInput<ReviewView, Q>,
        _: &Arc<SharedNnuecModel>,
        _: Arc<AtomicBool>,
    ) -> Result<PcNnuecReadoutRequest, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no same-input P/C NNUEC preparation",
        ))
    }
    fn consume_shared_nnuec_readout(
        &mut self,
        _: &FrameInput<ReviewView, Q>,
        _: &PcNnuecReadoutSeal,
        _: &Arc<SharedNnuecModel>,
        _: &SharedNnuecCompletion,
    ) -> Result<PcNnuecReadoutObservation, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no same-input P/C NNUEC consumption",
        ))
    }
    fn shared_nnuec_readout_consumed(
        &self,
        _: &FrameInput<ReviewView, Q>,
    ) -> Result<bool, AdaptiveManagerError> {
        Ok(false)
    }
    #[allow(clippy::type_complexity)]
    fn close_shared_nnuec_readouts(
        &self,
        _: &PreparationActor,
        acknowledgement: NnuecCacheReleaseAck,
        terminals: Vec<PcNnuecReadoutTerminal>,
    ) -> Result<
        PcSharedReadoutRootClosure,
        (
            AdaptiveManagerError,
            NnuecCacheReleaseAck,
            Vec<PcNnuecReadoutTerminal>,
        ),
    > {
        Err((
            AdaptiveManagerError::Invalid("driver has no shared P/C NNUEC closure"),
            acknowledgement,
            terminals,
        ))
    }
    fn shared_nnuec_closure_matches(&self, _: &PcSharedReadoutRootClosure) -> bool {
        false
    }
    fn role_submitted(
        &mut self,
        input: &FrameInput<ReviewView, Q>,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), AdaptiveManagerError>;
    fn validate_role(
        &self,
        input: &FrameInput<ReviewView, Q>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError>;
    fn consume_accepted(
        &mut self,
        input: &FrameInput<ReviewView, Q>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: Option<TicketExecutionEvidence>,
    ) -> Result<(), AdaptiveManagerError>;
    fn role_consumption_recorded(
        &self,
        _: &RoleTicketEnvelope,
        _: &FrameInput<ReviewView, Q>,
    ) -> Result<bool, AdaptiveManagerError> {
        Ok(false)
    }
    fn role_discarded(
        &mut self,
        input: &FrameInput<ReviewView, Q>,
        ticket: Option<&RoleTicketEnvelope>,
    ) -> Result<(), AdaptiveManagerError>;
    fn is_complete(&self) -> bool;
    /// Optional typed observation from the original settled registry. Legacy
    /// drivers retain their existing close behavior.
    fn close_role_call_budget(&mut self, _: u64, _: u64) -> Result<(), AdaptiveManagerError> {
        Ok(())
    }
    fn quiesce_satisfied_goal(&mut self) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual satisfied root goal",
        ))
    }
    fn revoke(&mut self) -> Result<(), AdaptiveManagerError>;
    fn review_lease_admission(&self) -> Result<AdaptiveReviewLeaseAdmission, AdaptiveManagerError>;
    fn validate_collection_admission(
        &self,
        _: &RoventCollectionRootAdmission,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual native collection root admission",
        ))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    fn validate_collection_admission_v2(
        &self,
        _: &RoventCollectionRootAdmissionV2,
    ) -> Result<(), AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "driver has no actual V2 native collection root admission",
        ))
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct RoleOnlyCollectionV2Rejection<Q, O> {
    pub error: AdaptiveManagerError,
    pub admission: RoventCollectionRootAdmissionV2,
    pub observer: AdaptiveCollectionObserverBoxV2<Q, O>,
}

/// Pure role configuration. R counts, R credit and R control are absent by type.
#[derive(Clone, Copy, Debug)]
pub struct RoleOnlyManagerRegistration {
    pub authority: SearchAuthority,
    pub mode: ExecutionMode,
    pub root_budget: Digest,
    pub hard_deadline: Instant,
    pub token_budget: u16,
    pub first_request_sequence: u64,
    pub role_tickets_max: usize,
    pub role_call_limit: u64,
    pub event_order: CoordinatorPolicy,
}

struct RoleOnlyBinding<Q, O> {
    preparation: PreparationId,
    input: FrameInput<ReviewView, Q>,
    ticket: Option<RoleTicketEnvelope>,
    prepared_observed: bool,
    dispatch_pending: bool,
    dispatched: bool,
    owner_run_started: bool,
    service_completed: bool,
    physical: Option<TicketExecutionEvidence>,
    delivered: Option<Arc<O>>,
    ack_accepted: Option<bool>,
    ack_dispatched: bool,
    cancellation_requested: bool,
    revoked_acceptance: Option<RoleFailure>,
    retired: bool,
    accepted: Option<AcceptedRoleStep>,
    collection_dispatched: bool,
    collection_abandoned: bool,
    consumption_recorded: bool,
}

pub struct RoleOnlyManagerSources<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    pub native: TicketedNativeService<ReviewView, Q, B>,
    pub driver: D,
}
pub struct RoleOnlyManagerRejection<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    pub error: AdaptiveManagerError,
    pub sources: RoleOnlyManagerSources<Q, B, D>,
}

/// Separate receipt domain; zero R work is a structural absence, not a copy of
/// an internal PALS ledger or a four-session legacy acceptance condition.
#[derive(Clone, Debug)]
pub struct RoleOnlyShutdownReceipt {
    pub authority: SearchAuthority,
    pub mode: ExecutionMode,
    pub root_budget: Digest,
    pub native_process_joined: bool,
    pub native_physical_shutdown_confirmed: bool,
    pub pending_tickets: usize,
    pub retained_outputs: usize,
    pub quarantined: bool,
    pub request_sequence_high_water: u64,
    pub counters: AdaptiveManagerCounters,
    /// Actual owner ACK rejections after this root revoked an issued permit.
    /// These are unused results, never accepted NN inputs or search work.
    pub canceled_acceptance_acks: u64,
    pub expired_acceptance_acks: u64,
    pub review_leases: RoventReviewLeaseObservation,
    native_root_fence: Option<NativeRootFence>,
    close_mode: AdaptiveCloseMode,
    shared_nnuec_required: bool,
    shared_nnuec_closure: Option<Arc<PcSharedReadoutRootClosure>>,
}
impl RoleOnlyShutdownReceipt {
    pub fn permits_semantic_closure(&self) -> bool {
        !self.mode.has_native_r()
            && !self.quarantined
            && self.pending_tickets == 0
            && self.retained_outputs == 0
            && (!self.shared_nnuec_required || self.shared_nnuec_closure.is_some())
            && match self.close_mode {
                AdaptiveCloseMode::ProcessShutdown => {
                    self.native_process_joined && self.native_physical_shutdown_confirmed
                }
                AdaptiveCloseMode::RootQuiesce => self.native_root_fence.is_some_and(|fence| {
                    fence.authority() == self.authority && fence.permits_reuse()
                }),
            }
    }
    pub fn root_fence(&self) -> Option<NativeRootFence> {
        self.native_root_fence
    }
    /// Observation of the original actor/root closure, not a replacement for
    /// the native physical fence or a new evaluator/NN completion authority.
    pub fn shared_nnuec_closure(&self) -> Option<&Arc<PcSharedReadoutRootClosure>> {
        self.shared_nnuec_closure.as_ref()
    }
}

struct PendingSharedReadout<Q> {
    input: FrameInput<ReviewView, Q>,
    request: Option<SharedNnuecRequest>,
    seal: Option<PcNnuecReadoutSeal>,
    completion: Option<SharedNnuecCompletion>,
    submitted: bool,
}

/// This is immutable-question/terminal custody only. The Session owns the one
/// existing preparation actor; no R frame, CpuJob, evaluator or worker exists
/// here. Numeric readouts never change P/C WDL, mate proofs or node counters.
struct SharedReadoutPreparation<Q> {
    model: Arc<SharedNnuecModel>,
    canceled: Arc<AtomicBool>,
    pending: Option<PendingSharedReadout<Q>>,
    terminals: Vec<PcNnuecReadoutTerminal>,
    invalidation: Option<NnuecCacheInvalidation>,
    cache_release: Option<NnuecCacheReleaseAck>,
    closure: Option<Arc<PcSharedReadoutRootClosure>>,
    unknown: bool,
}

pub struct RoleOnlyExecutionManager<Q, B, D>
where
    B: TicketNativeBackend<ReviewView, Q>,
{
    registration: RoleOnlyManagerRegistration,
    clock: ContractSystemClock,
    deadline: Deadline,
    native: TicketedNativeService<ReviewView, Q, B>,
    registry: RoleTicketRegistry<ReviewView, Arc<B::Output>>,
    driver: D,
    bindings: Vec<RoleOnlyBinding<Q, B::Output>>,
    acks: VecDeque<NativeAck>,
    pending_input: Option<FrameInput<ReviewView, Q>>,
    pending_preparation: Option<TicketPreparation<ReviewView, Q>>,
    shared_nnuec: Option<SharedReadoutPreparation<Q>>,
    phase: ManagerPhase,
    native_ready: bool,
    verifier_registered: bool,
    native_finished: Option<bool>,
    native_joined: bool,
    native_root_fence: Option<NativeRootFence>,
    root_quiesce_requested: bool,
    close_mode: AdaptiveCloseMode,
    shutdown_until: Option<Instant>,
    cleanup_boundary: Option<CleanupBoundary>,
    effective_clock: Option<crate::rovent_ponder::RoventPonderClockHandle>,
    counters: AdaptiveManagerCounters,
    canceled_acceptance_acks: u64,
    expired_acceptance_acks: u64,
    accounting: AdaptiveNativeAccountingSnapshot,
    review_leases: RoventReviewLeaseObservation,
    failures: VecDeque<AdaptiveManagerError>,
    retained_completions: Vec<RetainedNativeCompletion<B::Output>>,
    retained_receipts: Vec<NativeAcceptanceReceipt>,
    retained_permits: Vec<NativeAcceptancePermit>,
    collection: Option<Box<dyn AdaptiveCollectionObserver<Q, B::Output>>>,
    collection_admission: Option<RoventCollectionRootAdmission>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    collection_v2: Option<AdaptiveCollectionObserverBoxV2<Q, B::Output>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    collection_admission_v2: Option<RoventCollectionRootAdmissionV2>,
    collection_settled: bool,
}

impl<Q, B, D> RoleOnlyExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: RoleOnlySearchDriver<Q>,
{
    #[allow(clippy::result_large_err)]
    pub fn new_retaining(
        registration: RoleOnlyManagerRegistration,
        clock: ContractSystemClock,
        sources: RoleOnlyManagerSources<Q, B, D>,
    ) -> Result<Self, RoleOnlyManagerRejection<Q, B, D>> {
        let admission = (|| {
            if registration.mode.has_native_r()
                || registration.authority.epoch != clock.domain().0
                || registration.role_call_limit == 0
                || registration.first_request_sequence == 0
                || !(1..=MAX_ROLE_PENDING).contains(&registration.role_tickets_max)
                || registration.hard_deadline <= Instant::now()
            {
                return Err(AdaptiveManagerError::Invalid(
                    "P/C-only manager registration",
                ));
            }
            let _ = sources.driver.researcher();
            sources.driver.validate_registration(
                registration.authority,
                registration.mode,
                registration.hard_deadline,
                registration.root_budget,
            )?;
            let deadline = InstantClock::new(clock.domain(), clock.origin())
                .deadline_at(registration.hard_deadline)
                .map_err(|_| AdaptiveManagerError::Deadline)?;
            sources.native.replace_root(registration.authority)?;
            let native_ready = match sources.native.ready_clock() {
                Some(actual)
                    if actual.domain() == clock.domain() && actual.origin() == clock.origin() =>
                {
                    true
                }
                Some(_) => return Err(AdaptiveManagerError::Invalid("P/C native clock origin")),
                None => false,
            };
            let mut registry = RoleTicketRegistry::new_with_capacity(
                registration.authority,
                registration.token_budget,
                registration.event_order,
                registration.first_request_sequence,
                registration.role_tickets_max,
            )?;
            let verifier_registered =
                if let Some(verifier) = sources.native.native_acceptance_verifier() {
                    registry.register_native_acceptance_verifier(verifier)?;
                    true
                } else {
                    false
                };
            Ok((deadline, native_ready, registry, verifier_registered))
        })();
        let (deadline, native_ready, registry, verifier_registered) = match admission {
            Ok(admission) => admission,
            Err(error) => return Err(RoleOnlyManagerRejection { error, sources }),
        };
        Ok(Self {
            registration,
            clock,
            deadline,
            native: sources.native,
            registry,
            driver: sources.driver,
            bindings: Vec::with_capacity(registration.role_tickets_max),
            acks: VecDeque::with_capacity(registration.role_tickets_max),
            pending_input: None,
            pending_preparation: None,
            shared_nnuec: None,
            phase: ManagerPhase::Running,
            native_ready,
            verifier_registered,
            native_finished: None,
            native_joined: false,
            native_root_fence: None,
            root_quiesce_requested: false,
            close_mode: AdaptiveCloseMode::RootQuiesce,
            shutdown_until: None,
            cleanup_boundary: None,
            effective_clock: None,
            counters: AdaptiveManagerCounters::default(),
            canceled_acceptance_acks: 0,
            expired_acceptance_acks: 0,
            accounting: AdaptiveNativeAccountingSnapshot::default(),
            review_leases: RoventReviewLeaseObservation::new(registration.role_tickets_max),
            failures: VecDeque::with_capacity(64),
            retained_completions: Vec::with_capacity(16),
            retained_receipts: Vec::with_capacity(16),
            retained_permits: Vec::with_capacity(16),
            collection: None,
            collection_admission: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            collection_v2: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            collection_admission_v2: None,
            collection_settled: false,
        })
    }
    pub fn driver(&self) -> &D {
        &self.driver
    }
    pub fn driver_mut(&mut self) -> &mut D {
        &mut self.driver
    }
    /// Register the original model before the first question. Explicit None
    /// baselines do not call this endpoint; declared shared readout cannot
    /// silently fall back to the old unprepared P/C path.
    pub fn enable_shared_nnuec_preparation(
        &mut self,
        model: Arc<SharedNnuecModel>,
    ) -> Result<(), AdaptiveManagerError> {
        if self.phase != ManagerPhase::Running
            || self.shutdown_until.is_some()
            || self.shared_nnuec.is_some()
            || self.pending_input.is_some()
            || self.pending_preparation.is_some()
            || !self.bindings.is_empty()
            || self.counters.native_preparations != 0
        {
            return Err(AdaptiveManagerError::Invalid(
                "P/C shared NNUEC registration must precede the first question",
            ));
        }
        self.driver
            .register_shared_nnuec_readout(Arc::clone(&model))?;
        if !self.driver.requires_shared_nnuec_readout() {
            return Err(AdaptiveManagerError::Invalid(
                "P/C driver did not retain shared NNUEC registration",
            ));
        }
        self.shared_nnuec = Some(SharedReadoutPreparation {
            model,
            canceled: Arc::new(AtomicBool::new(false)),
            pending: None,
            terminals: Vec::new(),
            invalidation: None,
            cache_release: None,
            closure: None,
            unknown: false,
        });
        Ok(())
    }

    /// The Session lends its existing actor before/after tick, including while
    /// draining. Completion and cache release remain nonblocking, same-owner
    /// operations and do not create a CPU search worker or native node credit.
    pub fn pump_shared_nnuec_preparation(
        &mut self,
        actor: &PreparationActor,
    ) -> Result<bool, AdaptiveManagerError> {
        if self.phase == ManagerPhase::Finished {
            return Ok(false);
        }
        if self.phase == ManagerPhase::Running
            && (self.effective_hard_expired() || self.now().is_err())
        {
            self.begin_root_quiesce(self.cleanup_until()?)?;
        }
        let Some(mut preparation) = self.shared_nnuec.take() else {
            return if self.driver.requires_shared_nnuec_readout() {
                Err(AdaptiveManagerError::Invalid(
                    "required P/C shared NNUEC preparation is absent",
                ))
            } else {
                Ok(false)
            };
        };
        let result = self.pump_shared_readout(actor, &mut preparation);
        self.shared_nnuec = Some(preparation);
        if let Err(error) = &result {
            self.record_failure(error.clone());
        }
        result
    }

    fn pump_shared_readout(
        &mut self,
        actor: &PreparationActor,
        preparation: &mut SharedReadoutPreparation<Q>,
    ) -> Result<bool, AdaptiveManagerError> {
        if preparation.unknown || preparation.closure.is_some() {
            return Ok(false);
        }
        let closing = self.phase != ManagerPhase::Running;
        if closing {
            preparation.canceled.store(true, Ordering::Release);
        }
        let may_submit = !closing && self.can_admit_work();
        let mut changed = false;
        if preparation.pending.is_none()
            && may_submit
            && let Some(input) = self.pending_input.as_ref()
            && !self.driver.shared_nnuec_readout_consumed(input)?
        {
            let request = self.driver.prepare_shared_nnuec_readout(
                input,
                &preparation.model,
                Arc::clone(&preparation.canceled),
            )?;
            let (request, seal) = request.into_parts();
            preparation.pending = Some(PendingSharedReadout {
                input: input.clone(),
                request: Some(request),
                seal: Some(seal),
                completion: None,
                submitted: false,
            });
            changed = true;
        }
        if let Some(pending) = preparation.pending.as_mut()
            && !pending.submitted
            && closing
        {
            if pending.request.is_none() || pending.seal.is_none() {
                preparation.unknown = true;
                return Err(AdaptiveManagerError::Invalid(
                    "unsubmitted P/C NNUEC original request/seal custody is absent",
                ));
            }
            let request = pending
                .request
                .take()
                .expect("original unsubmitted request");
            let seal = pending.seal.take().expect("original unsubmitted seal");
            match seal.retire_unsubmitted(request) {
                Ok(terminal) => preparation.terminals.push(terminal),
                Err((error, seal, request)) => {
                    pending.seal = Some(seal);
                    pending.request = Some(request);
                    preparation.unknown = true;
                    return Err(AdaptiveManagerError::Semantic(error.to_string()));
                }
            }
            preparation.pending = None;
            changed = true;
        }
        if let Some(pending) = preparation.pending.as_mut() {
            if !pending.submitted && may_submit {
                let request = pending.request.take().ok_or(AdaptiveManagerError::Invalid(
                    "P/C NNUEC original request custody is absent",
                ))?;
                match actor.try_submit(request) {
                    Ok(()) => {
                        pending.submitted = true;
                        changed = true;
                    }
                    Err((PreparationError::Busy, original)) => pending.request = Some(original),
                    Err((error, original)) => {
                        pending.request = Some(original);
                        return Err(AdaptiveManagerError::NnuecPreparation(
                            NnuecWorkError::Preparation(error),
                        ));
                    }
                }
            }
            if pending.submitted && pending.completion.is_none() {
                match actor.try_complete() {
                    Ok(Some(completion)) => {
                        pending.completion = Some(completion);
                        changed = true;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        preparation.unknown = true;
                        return Err(AdaptiveManagerError::NnuecPreparation(
                            NnuecWorkError::Preparation(error),
                        ));
                    }
                }
            }
            // A suspended execution clock does not consume or discard a
            // completed question. Keep its exact owners until admission or
            // original root closure permits the next action.
            if (may_submit || closing) && pending.completion.is_some() {
                if pending.seal.is_none() {
                    preparation.unknown = true;
                    return Err(AdaptiveManagerError::Invalid(
                        "P/C NNUEC original completion seal is absent",
                    ));
                }
                let completion = pending.completion.take().expect("original completion");
                let seal = pending.seal.take().expect("original completion seal");
                let consumed = if closing {
                    Ok(())
                } else {
                    self.driver
                        .consume_shared_nnuec_readout(
                            &pending.input,
                            &seal,
                            &preparation.model,
                            &completion,
                        )
                        .map(|_| ())
                };
                match seal.retire(completion) {
                    Ok(terminal) => preparation.terminals.push(terminal),
                    Err((error, seal, completion)) => {
                        pending.seal = Some(seal);
                        pending.completion = Some(completion);
                        preparation.unknown = true;
                        return Err(AdaptiveManagerError::Semantic(error.to_string()));
                    }
                }
                preparation.pending = None;
                changed = true;
                consumed?;
            }
        }
        if closing && preparation.pending.is_none() {
            if preparation.invalidation.is_none() {
                preparation.invalidation =
                    Some(actor.invalidate_context_cache().map_err(|error| {
                        AdaptiveManagerError::NnuecPreparation(NnuecWorkError::Preparation(error))
                    })?);
                changed = true;
            }
            if preparation.cache_release.is_none() {
                preparation.cache_release = actor
                    .try_context_cache_release(
                        preparation
                            .invalidation
                            .as_ref()
                            .expect("original invalidation"),
                    )
                    .map_err(|error| {
                        AdaptiveManagerError::NnuecPreparation(NnuecWorkError::Preparation(error))
                    })?;
            }
            if let Some(acknowledgement) = preparation.cache_release.take() {
                let terminals = std::mem::take(&mut preparation.terminals);
                match self
                    .driver
                    .close_shared_nnuec_readouts(actor, acknowledgement, terminals)
                {
                    Ok(closure) => {
                        let matches = self.driver.shared_nnuec_closure_matches(&closure);
                        preparation.closure = Some(Arc::new(closure));
                        if !matches {
                            preparation.unknown = true;
                            return Err(AdaptiveManagerError::PhysicalUnknown);
                        }
                        changed = true;
                    }
                    Err((error, acknowledgement, terminals)) => {
                        preparation.cache_release = Some(acknowledgement);
                        preparation.terminals = terminals;
                        preparation.unknown = true;
                        return Err(error);
                    }
                }
            }
        }
        Ok(changed)
    }

    fn shared_readout_cleanup_settled(&self) -> bool {
        self.shared_nnuec.as_ref().map_or(
            !self.driver.requires_shared_nnuec_readout(),
            |preparation| {
                !preparation.unknown
                    && preparation.pending.is_none()
                    && preparation.closure.is_some()
            },
        )
    }
    pub fn register_cleanup_deadline(
        &mut self,
        original_cleanup_deadline: Instant,
    ) -> Result<(), AdaptiveManagerError> {
        if self.cleanup_boundary.is_some()
            || self.phase != ManagerPhase::Running
            || self.shutdown_until.is_some()
            || self.counters.native_preparations != 0
            || !self.bindings.is_empty()
            || self.pending_input.is_some()
            || self.pending_preparation.is_some()
        {
            return Err(AdaptiveManagerError::Invalid(
                "P/C original cleanup deadline admission",
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
    pub fn register_effective_clock(
        &mut self,
        handle: crate::rovent_ponder::RoventPonderClockHandle,
    ) -> Result<(), AdaptiveManagerError> {
        if self.phase != ManagerPhase::Running
            || self.counters.native_preparations != 0
            || self.pending_input.is_some()
            || self.pending_preparation.is_some()
            || self.effective_clock.is_some()
        {
            return Err(AdaptiveManagerError::Invalid(
                "P/C effective clock startup order",
            ));
        }
        effective_clock::admit(
            &handle,
            self.registration.authority,
            &self.clock,
            self.deadline,
            self.cleanup_boundary,
        )?;
        self.effective_clock = Some(handle);
        Ok(())
    }
    pub fn can_admit_work(&self) -> bool {
        let now = Instant::now();
        self.effective_clock
            .as_ref()
            .is_none_or(|handle| handle.snapshot(now).can_admit(now))
    }
    fn effective_hard_expired(&self) -> bool {
        let now = Instant::now();
        self.effective_clock
            .as_ref()
            .is_some_and(|handle| handle.snapshot(now).hard_expired(now))
    }
    /// Observe its version before polling this manager and the caller's
    /// controls, then wait on that same version to avoid a lost wakeup.
    pub fn activity_signal(&self) -> rz_runtime::CompletionSignal {
        self.native.signal()
    }
    pub fn phase(&self) -> ManagerPhase {
        self.phase
    }
    pub fn failures(&self) -> impl Iterator<Item = &AdaptiveManagerError> {
        self.failures.iter()
    }
    pub fn pending_tickets(&self) -> usize {
        self.registry.occupancy()
    }
    pub fn counters(&self) -> AdaptiveManagerCounters {
        self.counters
    }
    pub fn native_accounting(&self) -> AdaptiveNativeAccountingSnapshot {
        self.accounting
    }
    pub fn install_collection_observer(
        &mut self,
        admission: RoventCollectionRootAdmission,
        mut observer: Box<dyn AdaptiveCollectionObserver<Q, B::Output>>,
    ) -> Result<(), AdaptiveManagerError> {
        if self.collection.is_some()
            || self.collection_admission.is_some()
            || self.has_collection_v2()
            || self.phase != ManagerPhase::Running
            || self.registry.cumulative_observation().admitted != 0
            || !self.bindings.is_empty()
            || self.pending_input.is_some()
        {
            return Err(AdaptiveManagerError::Invalid(
                "P/C collection root observer admission",
            ));
        }
        if !admission.validates_registration(
            self.registration.authority,
            self.registration.mode,
            self.registration.root_budget,
            self.deadline,
        ) {
            return Err(AdaptiveManagerError::Invalid(
                "P/C collection differs from original root registration",
            ));
        }
        self.driver.validate_collection_admission(&admission)?;
        observer.admitted(&admission)?;
        self.collection_admission = Some(admission);
        self.collection = Some(observer);
        Ok(())
    }

    /// Event-flow fixture only; it never creates a root admission or exists
    /// in a product build. Native tickets and acceptance stay real.
    #[cfg(test)]
    pub(crate) fn install_collection_observer_fixture(
        &mut self,
        observer: Box<dyn AdaptiveCollectionObserver<Q, B::Output>>,
    ) -> Result<(), AdaptiveManagerError> {
        if self.collection.is_some()
            || self.collection_admission.is_some()
            || self.has_collection_v2()
            || self.phase != ManagerPhase::Running
            || !self.bindings.is_empty()
            || self.pending_input.is_some()
        {
            return Err(AdaptiveManagerError::Invalid(
                "P/C collection test fixture state",
            ));
        }
        self.collection = Some(observer);
        Ok(())
    }

    fn has_collection_v2(&self) -> bool {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        {
            self.collection_v2.is_some() || self.collection_admission_v2.is_some()
        }
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        {
            false
        }
    }

    /// Internal V2 admission preserves its original objects on every failure.
    /// Installing an observer is not authority to start a teacher or a product.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn install_collection_observer_v2(
        &mut self,
        admission: RoventCollectionRootAdmissionV2,
        mut observer: AdaptiveCollectionObserverBoxV2<Q, B::Output>,
    ) -> Result<(), RoleOnlyCollectionV2Rejection<Q, B::Output>> {
        let result =
            (|| {
                if self.collection.is_some()
                    || self.collection_admission.is_some()
                    || self.has_collection_v2()
                    || self.phase != ManagerPhase::Running
                    || self.registry.cumulative_observation().admitted != 0
                    || !self.bindings.is_empty()
                    || self.pending_input.is_some()
                {
                    return Err(AdaptiveManagerError::Invalid(
                        "P/C V2 collection observer admission",
                    ));
                }
                let root = admission.root_admission();
                if root.authority() != self.registration.authority
                    || root.hard_deadline() != self.deadline
                    || admission.root_budget() != self.registration.root_budget
                    || admission
                        .native_origin()
                        .sources()
                        .lock()
                        .manifest
                        .engine
                        .mode
                        != self.registration.mode
                {
                    return Err(AdaptiveManagerError::Invalid(
                        "P/C V2 collection root registration",
                    ));
                }
                let verifier = self.native.native_acceptance_verifier().ok_or(
                    AdaptiveManagerError::Invalid("P/C V2 original native verifier absent"),
                )?;
                admission.validate_native_owner(&verifier)?;
                self.driver.validate_collection_admission_v2(&admission)?;
                observer.admitted(&admission)?;
                Ok(())
            })();
        if let Err(error) = result {
            return Err(RoleOnlyCollectionV2Rejection {
                error,
                admission,
                observer,
            });
        }
        self.collection_admission_v2 = Some(admission);
        self.collection_v2 = Some(observer);
        Ok(())
    }

    /// Return the original observer and admission only after the native root
    /// fence and collection settlement. This says nothing about a teacher.
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(crate) fn take_collection_observer_v2(
        &mut self,
    ) -> Result<
        Option<(
            RoventCollectionRootAdmissionV2,
            AdaptiveCollectionObserverBoxV2<Q, B::Output>,
        )>,
        AdaptiveManagerError,
    > {
        if self.collection_v2.is_none() && self.collection_admission_v2.is_none() {
            return Ok(None);
        }
        if self.collection_v2.is_none() || self.collection_admission_v2.is_none() {
            return Err(AdaptiveManagerError::Invalid(
                "P/C V2 collection owner mismatch",
            ));
        }
        let receipt = self.shutdown_receipt()?;
        if self.phase != ManagerPhase::Finished
            || !self.collection_settled
            || !receipt.permits_semantic_closure()
        {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        Ok(Some((
            self.collection_admission_v2
                .take()
                .expect("checked admission"),
            self.collection_v2.take().expect("checked observer"),
        )))
    }

    fn now(&self) -> Result<MonotonicTick, AdaptiveManagerError> {
        let now = self
            .clock
            .try_now()
            .map_err(|_| AdaptiveManagerError::Deadline)?;
        self.deadline
            .accepts(self.clock.domain(), now)
            .map_err(|_| AdaptiveManagerError::Deadline)?;
        Ok(now)
    }
    fn record_failure(&mut self, error: AdaptiveManagerError) {
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
            if self
                .bindings
                .iter()
                .any(|binding| !binding.service_completed)
                || !self.acks.is_empty()
            {
                return;
            }
            self.review_leases.terminal_capacity = true;
        }
        if matches!(error, AdaptiveManagerError::PhysicalUnknown) {
            self.phase = ManagerPhase::Quarantined;
        }
        if self.failures.len() == 64 {
            self.failures.pop_front();
        }
        self.failures.push_back(error);
    }
    fn pending_limit(&self) -> Result<usize, AdaptiveManagerError> {
        let observed = self.driver.review_lease_admission()?;
        let declared = if self.registration.event_order == CoordinatorPolicy::Synchronous {
            1
        } else {
            self.registration.role_tickets_max
        };
        Ok(if observed.frontier_owner_present {
            declared.min(
                observed
                    .capacity
                    .saturating_sub(1)
                    .saturating_sub(observed.frontier_replacement_reserve),
            )
        } else {
            declared
        })
    }
    fn index(&self, ticket: RoleTicketEnvelope) -> Result<usize, AdaptiveManagerError> {
        self.bindings
            .iter()
            .position(|binding| binding.ticket == Some(ticket))
            .ok_or(AdaptiveManagerError::Invalid("P/C original ticket binding"))
    }
    fn schedule(&mut self) -> Result<bool, AdaptiveManagerError> {
        if !self.can_admit_work() {
            return Ok(false);
        }
        if self.driver.requires_shared_nnuec_readout() && self.shared_nnuec.is_none() {
            return Err(AdaptiveManagerError::Invalid(
                "required P/C NNUEC readout was not enabled",
            ));
        }
        let mut changed = false;
        while self.phase == ManagerPhase::Running && self.bindings.len() < self.pending_limit()? {
            let reserved = self
                .bindings
                .iter()
                .filter(|binding| binding.ticket.is_none())
                .count() as u64;
            if self
                .registry
                .cumulative_observation()
                .admitted
                .saturating_add(reserved)
                >= self.registration.role_call_limit
            {
                break;
            }
            if self.pending_input.is_none() {
                self.pending_input = self.driver.next_role()?;
            }
            let Some(input) = self.pending_input.as_ref() else {
                break;
            };
            let role = input.stage.role().ok_or(AdaptiveManagerError::Invalid(
                "CPU stage in P/C-only manager",
            ))?;
            if !matches!(role, Role::Proposer | Role::Critic) {
                return Err(AdaptiveManagerError::Invalid(
                    "non P/C role in P/C-only manager",
                ));
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
            let now = self.now()?;
            self.deadline
                .accepts(self.clock.domain(), now)
                .map_err(|_| AdaptiveManagerError::Deadline)?;
            if self.shared_nnuec.is_some() && !self.driver.shared_nnuec_readout_consumed(input)? {
                // The same original FrameInput stays pinned here. Only the
                // Session's nonblocking actor pump can make this gate ready.
                break;
            }
            if self.pending_preparation.is_none() {
                self.pending_preparation = Some(TicketPreparation {
                    question: input.question_digest,
                    seal: input.view_seal,
                    role,
                    deadline: self.deadline,
                    view: Arc::clone(&input.view),
                    query: (*input.question).clone(),
                });
            }
            let source = self
                .pending_preparation
                .take()
                .expect("original unqueued preparation");
            let preparation = match self.native.prepare(source) {
                Ok(preparation) => preparation,
                Err((TicketServiceError::Capacity, original)) => {
                    self.pending_preparation = Some(original);
                    break;
                }
                Err((error, original)) => {
                    self.pending_preparation = Some(original);
                    return Err(error.into());
                }
            };
            self.bindings.push(RoleOnlyBinding {
                preparation,
                input: self.pending_input.take().expect("same pending input"),
                ticket: None,
                prepared_observed: false,
                dispatch_pending: false,
                dispatched: false,
                owner_run_started: false,
                service_completed: false,
                physical: None,
                delivered: None,
                ack_accepted: None,
                ack_dispatched: false,
                cancellation_requested: false,
                revoked_acceptance: None,
                retired: false,
                accepted: None,
                collection_dispatched: false,
                collection_abandoned: false,
                consumption_recorded: false,
            });
            self.counters.native_preparations += 1;
            changed = true;
        }
        Ok(changed)
    }
    fn dispatch(&mut self) -> Result<bool, AdaptiveManagerError> {
        if !self.native_ready
            || !self.can_admit_work()
            || self.phase != ManagerPhase::Running
            || !self.failures.is_empty()
            || self.now().is_err()
        {
            return Ok(false);
        }
        while let Some(dispatch) = self.registry.take_dispatch() {
            let index = self.index(dispatch.ticket)?;
            self.bindings[index].dispatch_pending = true;
        }
        let mut changed = false;
        for index in 0..self.bindings.len() {
            if !self.bindings[index].dispatch_pending {
                continue;
            }
            let binding = &self.bindings[index];
            let ticket = binding.ticket.expect("prepared actual ticket");
            if !binding.collection_dispatched {
                if let Some(observer) = &mut self.collection {
                    observer
                        .dispatch(&ticket, &binding.input)
                        .map_err(AdaptiveManagerError::Collection)?;
                }
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                if let Some(observer) = &mut self.collection_v2 {
                    self.collection_admission_v2
                        .as_ref()
                        .ok_or(AdaptiveManagerError::Invalid(
                            "P/C V2 collection admission absent",
                        ))?
                        .validate_ticket(&ticket)?;
                    observer.dispatch(&ticket, &binding.input)?;
                }
                self.bindings[index].collection_dispatched = true;
            }
            let binding = &self.bindings[index];
            match self
                .native
                .execute_with_admission_clock(binding.preparation, ticket)
            {
                Ok(_) => {
                    self.bindings[index].dispatch_pending = false;
                    self.bindings[index].dispatched = true;
                    changed = true;
                }
                Err(TicketServiceError::Capacity) => break,
                Err(error) => return Err(error.into()),
            }
        }
        Ok(changed)
    }
    fn deliver(&mut self) -> Result<bool, AdaptiveManagerError> {
        let mut changed = false;
        for _ in 0..self.registration.role_tickets_max {
            let now = self
                .clock
                .try_now()
                .map_err(|_| AdaptiveManagerError::Deadline)?;
            let Some(ready) = self.registry.take_ready(self.clock.domain(), now) else {
                break;
            };
            let ticket = ready.ticket;
            let index = self.index(ticket)?;
            let accepted = ready.logically_eligible
                && self.phase == ManagerPhase::Running
                && ready.result.is_ok();
            if let Ok(output) = ready.result {
                self.bindings[index].delivered = Some(output);
            }
            let permit = if accepted {
                let binding = &self.bindings[index];
                let output = binding
                    .delivered
                    .as_ref()
                    .expect("eligible output")
                    .proposer_ranking_output();
                validate_role_output(output, &ticket, &ready.view)?;
                self.driver.validate_role(&binding.input, &ticket, output)?;
                Some(self.registry.issue_reciprocal_native_acceptance(
                    &ticket,
                    self.clock.domain(),
                    now,
                )?)
            } else {
                None
            };
            self.bindings[index].ack_accepted = Some(accepted);
            self.acks.push_back(NativeAck {
                preparation: self.bindings[index].preparation,
                ticket,
                accepted,
                permit,
            });
            changed = true;
        }
        self.flush_acks()?;
        Ok(changed)
    }
    fn flush_acks(&mut self) -> Result<(), AdaptiveManagerError> {
        // Root/process close already asks the actual native owner to dispose
        // of every held ticket. Keep unsent ACKs and their original permits
        // here until that fence instead of enqueueing new commands that could
        // race the owner's completed root cleanup.
        if self.phase != ManagerPhase::Running {
            return Ok(());
        }
        while let Some(mut ack) = self.acks.pop_front() {
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
            } else {
                self.native.discard(ack.preparation, ack.ticket)
            };
            if let Err(error) = result {
                self.acks.push_front(ack);
                return if error == TicketServiceError::Capacity {
                    Ok(())
                } else {
                    Err(error.into())
                };
            }
            let index = self.index(ack.ticket)?;
            self.bindings[index].ack_dispatched = true;
        }
        Ok(())
    }
    fn consume_retired(&mut self) -> Result<bool, AdaptiveManagerError> {
        let mut changed = false;
        let mut index = 0;
        while index < self.bindings.len() {
            if !self.bindings[index].retired || self.bindings[index].revoked_acceptance.is_some() {
                index += 1;
                continue;
            }
            let will_consume = self.phase == ManagerPhase::Running
                && self.failures.is_empty()
                && self.now().is_ok()
                && !self.bindings[index].consumption_recorded
                && self.bindings[index].accepted.is_some();
            if !will_consume
                && !self.bindings[index].consumption_recorded
                && !self.bindings[index].collection_abandoned
            {
                if let (Some(observer), Some(ticket)) =
                    (&mut self.collection, self.bindings[index].ticket.as_ref())
                {
                    observer
                        .abandoned(ticket)
                        .map_err(AdaptiveManagerError::Collection)?;
                }
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                if let (Some(observer), Some(ticket)) = (
                    &mut self.collection_v2,
                    self.bindings[index].ticket.as_ref(),
                ) {
                    observer.abandoned(ticket)?;
                }
                self.bindings[index].collection_abandoned = true;
            }
            let binding = &self.bindings[index];
            if will_consume {
                if let Some(accepted) = &binding.accepted {
                    let output = binding
                        .delivered
                        .as_ref()
                        .ok_or(AdaptiveManagerError::Invalid("accepted P/C output custody"))?;
                    let returned = self.driver.consume_accepted(
                        &binding.input,
                        accepted,
                        output.proposer_ranking_output(),
                        binding.physical,
                    );
                    let consumed = if returned.is_ok() {
                        true
                    } else {
                        match self
                            .driver
                            .role_consumption_recorded(accepted.ticket(), &binding.input)
                        {
                            Ok(consumed) => consumed,
                            Err(readback) => {
                                self.record_failure(readback);
                                false
                            }
                        }
                    };
                    if consumed {
                        self.mark_consumed(index)?;
                        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                        if let Some(observer) = &mut self.collection_v2 {
                            let accepted = self.bindings[index].accepted.as_ref().ok_or(
                                AdaptiveManagerError::Invalid("P/C V2 consumed proof absent"),
                            )?;
                            let observation = if returned.is_ok() {
                                RoleOnlySearchConsumptionV2::DriverReturnedSuccess
                            } else {
                                RoleOnlySearchConsumptionV2::DriverReadbackAfterError
                            };
                            if let Err(error) = observer.consumed(accepted, observation) {
                                // Consumption already happened. Keep both that
                                // fact and the observer error without replay.
                                self.record_failure(AdaptiveManagerError::Collection(error));
                            }
                        }
                    }
                    returned?;
                } else {
                    self.driver
                        .role_discarded(&binding.input, binding.ticket.as_ref())?;
                    self.counters.unused_role_results += 1;
                }
            } else if !binding.consumption_recorded {
                self.driver
                    .role_discarded(&binding.input, binding.ticket.as_ref())?;
                self.counters.unused_role_results += 1;
            }
            self.bindings.remove(index);
            changed = true;
        }
        Ok(changed)
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
        let binding = &self.bindings[index];
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
            || accepted
            || has_native_receipt
            || !binding.cancellation_requested
            || !binding.ack_dispatched
            || binding.ack_accepted != Some(true)
            || binding.retired
            || binding.revoked_acceptance.is_some()
            || binding.accepted.is_some()
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
        self.registry
            .validate_revoked_reciprocal_native_permit(ticket, permit)
            .is_ok()
    }

    fn mark_consumed(&mut self, index: usize) -> Result<(), AdaptiveManagerError> {
        let binding = &self.bindings[index];
        if binding.consumption_recorded {
            return Ok(());
        }
        let ticket = binding
            .ticket
            .ok_or(AdaptiveManagerError::Invalid("P/C consumed ticket missing"))?;
        let mut counters = self.counters;
        let mut accounting = self.accounting;
        counters.consumed_role_outputs =
            counters
                .consumed_role_outputs
                .checked_add(1)
                .ok_or(AdaptiveManagerError::Capacity(
                    "P/C consumed output counter",
                ))?;
        accounting.observe_search_consumption(ticket.role, binding.physical)?;
        match binding
            .physical
            .filter(|e| e.measurement_complete)
            .and_then(|e| e.completed_nn_inputs)
        {
            Some(count) => {
                counters.consumed_nn_inputs = counters
                    .consumed_nn_inputs
                    .checked_add(count)
                    .ok_or(AdaptiveManagerError::Capacity("P/C consumed input counter"))?;
                counters.zero_nn_outputs_consumed = counters
                    .zero_nn_outputs_consumed
                    .checked_add(u64::from(count == 0))
                    .ok_or(AdaptiveManagerError::Capacity("P/C zero input counter"))?;
            }
            None => {
                counters.consumed_nn_input_count_unknown = counters
                    .consumed_nn_input_count_unknown
                    .checked_add(1)
                    .ok_or(AdaptiveManagerError::Capacity("P/C unknown input counter"))?
            }
        }
        self.counters = counters;
        self.accounting = accounting;
        self.bindings[index].consumption_recorded = true;
        Ok(())
    }
    fn event(&mut self, event: TicketNativeEvent<B::Output>) -> Result<(), AdaptiveManagerError> {
        match event {
            TicketNativeEvent::Ready { clock } => {
                if clock.domain() != self.clock.domain() || clock.origin() != self.clock.origin() {
                    return Err(AdaptiveManagerError::Invalid("P/C native Ready origin"));
                }
                self.native_ready = true;
                if self.phase == ManagerPhase::Running && !self.verifier_registered {
                    let verifier = self
                        .native
                        .native_acceptance_verifier()
                        .ok_or(AdaptiveManagerError::Invalid("P/C native verifier absent"))?;
                    self.registry
                        .register_native_acceptance_verifier(verifier)?;
                    self.verifier_registered = true;
                }
            }
            TicketNativeEvent::StartupFailed(error) => {
                return Err(AdaptiveManagerError::Semantic(format!(
                    "P/C native startup: {error:?}"
                )));
            }
            TicketNativeEvent::Prepared(prepared) => {
                let index = self
                    .bindings
                    .iter()
                    .position(|binding| binding.preparation == prepared.preparation)
                    .ok_or(AdaptiveManagerError::Invalid(
                        "P/C native preparation owner",
                    ))?;
                let input = &self.bindings[index].input;
                if self.bindings[index].ticket.is_some()
                    || self.bindings[index].prepared_observed
                    || input.question_digest != prepared.question
                    || input.view_seal != prepared.view
                    || input.stage.role() != Some(prepared.role)
                    || prepared.original_deadline != self.deadline
                {
                    return Err(AdaptiveManagerError::Invalid(
                        "P/C native prepared immutable echo",
                    ));
                }
                self.bindings[index].prepared_observed = true;
                if self.phase != ManagerPhase::Running || self.now().is_err() {
                    // Close may already have canceled this exact original
                    // preparation. The worker can then dispose of it and
                    // publish PreparationFailed before this older Prepared
                    // echo is consumed. Do not issue a redundant cancel after
                    // its control has retired; an unproven/missing original
                    // preparation still takes the strict native error path.
                    if !self.bindings[index].cancellation_requested {
                        self.native.cancel(prepared.preparation)?;
                        self.bindings[index].cancellation_requested = true;
                    }
                    self.counters.canceled_preparations += 1;
                    return Ok(());
                }
                let now = self.now()?;
                let input = &self.bindings[index].input;
                let ticket = self.registry.admit(
                    RoleSubmission {
                        question: prepared.question,
                        view: prepared.view,
                        representation: prepared.representation,
                        role: prepared.role,
                        original_deadline: prepared.original_deadline,
                    },
                    Arc::clone(&input.view),
                    self.clock.domain(),
                    now,
                )?;
                self.bindings[index].ticket = Some(ticket);
                self.driver
                    .role_submitted(&self.bindings[index].input, &ticket)?;
                self.counters.submitted_role_tickets += 1;
            }
            TicketNativeEvent::Started(ticket) => {
                let _ = self.index(ticket)?;
                self.registry.physical_started(&ticket)?;
            }
            TicketNativeEvent::OwnerRunStarted { ticket, started_at } => {
                let index = self.index(ticket)?;
                if !self.bindings[index].dispatched
                    || self.bindings[index].owner_run_started
                    || started_at < self.clock.origin()
                    || started_at > Instant::now()
                {
                    return Err(AdaptiveManagerError::Invalid("P/C owner run identity/time"));
                }
                self.bindings[index].owner_run_started = true;
            }
            TicketNativeEvent::Completed {
                completion,
                evidence,
                owner_run,
                owner_service_elapsed,
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
                let outcome = (|| {
                    let ticket = completion.ticket;
                    let index = self.index(ticket)?;
                    if self.bindings[index].service_completed
                        || retained.owner_run.ticket() != &ticket
                    {
                        return Err(AdaptiveManagerError::Invalid(
                            "P/C duplicate/foreign native owner completion",
                        ));
                    }
                    self.bindings[index].service_completed = true;
                    self.bindings[index].physical = evidence;
                    self.accounting.observe_completion(
                        &ticket,
                        &retained.owner_run,
                        completion.physical,
                        completion.result.is_ok(),
                        evidence,
                    )?;
                    match evidence
                        .filter(|e| e.measurement_complete)
                        .and_then(|e| e.completed_nn_inputs)
                    {
                        Some(count) => self.counters.completed_nn_inputs += count,
                        None => self.counters.nn_input_count_unknown += 1,
                    }
                    self.counters.physical_role_completions +=
                        u64::from(completion.physical == PhysicalRoleCompletion::Confirmed);
                    let physical = completion.physical;
                    let failure = completion.result.as_ref().err().cloned();
                    match self.registry.complete_reciprocal_role(completion) {
                        Ok(()) => {}
                        Err(rejected) => return Err(rejected.error.into()),
                    }
                    if physical == PhysicalRoleCompletion::Unknown {
                        return Err(AdaptiveManagerError::PhysicalUnknown);
                    }
                    if self.phase == ManagerPhase::Running {
                        if let Some(failure) = failure {
                            return Err(AdaptiveManagerError::Semantic(format!(
                                "P/C native execution: {failure:?}"
                            )));
                        }
                    }
                    Ok(())
                })();
                if let Err(error) = outcome {
                    self.retained_completions.push(retained);
                    return Err(error);
                }
            }
            TicketNativeEvent::PreparationFailed {
                preparation,
                failure,
            } => {
                if self.phase == ManagerPhase::Running {
                    return Err(AdaptiveManagerError::Semantic(format!(
                        "P/C native preparation: {failure:?}"
                    )));
                }
                if let Some(index) = self
                    .bindings
                    .iter()
                    .position(|b| b.preparation == preparation && b.ticket.is_none())
                {
                    self.driver
                        .role_discarded(&self.bindings[index].input, None)?;
                    self.bindings.remove(index);
                }
            }
            TicketNativeEvent::Rejected { preparation, error } => {
                if self.phase == ManagerPhase::Running {
                    return Err(error.into());
                }
                if let Some(index) = self
                    .bindings
                    .iter()
                    .position(|b| b.preparation == preparation && b.ticket.is_none())
                {
                    self.driver
                        .role_discarded(&self.bindings[index].input, None)?;
                    self.bindings.remove(index);
                }
            }
            TicketNativeEvent::Retired {
                ticket,
                accepted,
                result,
                native_receipt,
                rejected_permit,
            } => {
                let index = match self.index(ticket) {
                    Ok(index) => index,
                    Err(error) => {
                        if let Some(permit) = rejected_permit {
                            self.retained_permits.push(permit);
                        }
                        if let Some(receipt) = native_receipt {
                            self.retained_receipts.push(receipt);
                        }
                        return Err(error);
                    }
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
                let had_rejected_permit = rejected_permit.is_some();
                if let Some(permit) = rejected_permit {
                    self.retained_permits.push(permit);
                }
                if revoked_ack {
                    let failure = result.expect_err("validated canceled/expired native ACK");
                    let counter = if failure == RoleFailure::Canceled {
                        &mut self.canceled_acceptance_acks
                    } else {
                        &mut self.expired_acceptance_acks
                    };
                    *counter = counter
                        .checked_add(1)
                        .ok_or(AdaptiveManagerError::Capacity(
                            "P/C revoked native acceptance acknowledgement counter",
                        ))?;
                    // backend.accept did not succeed and the native owner
                    // still holds this original output. Neither a cancel nor
                    // this error ACK releases its view or permit. Only the
                    // actual root/process fence below may retire this binding.
                    self.bindings[index].revoked_acceptance = Some(failure);
                    self.bindings[index].retired = true;
                    return Ok(());
                }
                if result.is_err()
                    || had_rejected_permit
                    || !self.bindings[index].ack_dispatched
                    || self.bindings[index].ack_accepted != Some(accepted)
                    || self.bindings[index].retired
                {
                    if let Some(receipt) = native_receipt {
                        self.retained_receipts.push(receipt);
                    }
                    return Err(if result == Err(RoleFailure::PhysicalUnknown) {
                        AdaptiveManagerError::PhysicalUnknown
                    } else {
                        AdaptiveManagerError::Invalid("P/C native retirement/ack differs")
                    });
                }
                if accepted {
                    let receipt = native_receipt.ok_or(AdaptiveManagerError::Invalid(
                        "P/C actual acceptance receipt absent",
                    ))?;
                    self.accounting
                        .observe_owner_acceptance(ticket.role, self.bindings[index].physical)?;
                    match self.bindings[index]
                        .physical
                        .filter(|e| e.measurement_complete)
                        .and_then(|e| e.completed_nn_inputs)
                    {
                        Some(count) => self.counters.accepted_nn_inputs += count,
                        None => self.counters.accepted_nn_input_count_unknown += 1,
                    }
                    if self.phase == ManagerPhase::Running && self.now().is_ok() {
                        match self.registry.accept_reciprocal_native(
                            receipt,
                            self.clock.domain(),
                            self.now()?,
                        ) {
                            Ok(accepted) => {
                                self.bindings[index].accepted = Some(accepted);
                                // The native retirement already occurred. Keep
                                // that fact even when an observer rejects the
                                // accepted record; it is not search consumption.
                                self.bindings[index].retired = true;
                                if let Some(observer) = &mut self.collection {
                                    let binding = &self.bindings[index];
                                    observer
                                        .accepted(
                                            binding
                                                .accepted
                                                .as_ref()
                                                .expect("actual accepted proof"),
                                            &binding.input,
                                            binding
                                                .delivered
                                                .as_ref()
                                                .ok_or(AdaptiveManagerError::Invalid(
                                                    "accepted capture output custody",
                                                ))?
                                                .as_ref(),
                                        )
                                        .map_err(AdaptiveManagerError::Collection)?;
                                }
                                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                                if let Some(observer) = &mut self.collection_v2 {
                                    let binding = &self.bindings[index];
                                    let accepted =
                                        binding.accepted.as_ref().expect("actual accepted proof");
                                    let output = binding.delivered.as_ref().ok_or(
                                        AdaptiveManagerError::Invalid(
                                            "accepted V2 capture output custody",
                                        ),
                                    )?;
                                    let role_output = output.proposer_ranking_output();
                                    self.collection_admission_v2
                                        .as_ref()
                                        .ok_or(AdaptiveManagerError::Invalid(
                                            "P/C V2 collection admission absent",
                                        ))?
                                        .validate_output(accepted.ticket(), role_output)?;
                                    if !accepted.validates_output(role_output) {
                                        return Err(AdaptiveManagerError::Invalid(
                                            "P/C V2 accepted output differs",
                                        ));
                                    }
                                    observer.accepted(accepted, &binding.input, output.as_ref())?;
                                }
                            }
                            Err(rejected) => {
                                self.retained_receipts.push(rejected.receipt);
                                return Err(rejected.error.into());
                            }
                        }
                    } else {
                        self.retained_receipts.push(receipt);
                    }
                } else {
                    if let Some(receipt) = native_receipt {
                        self.retained_receipts.push(receipt);
                        return Err(AdaptiveManagerError::Invalid(
                            "discard returned acceptance receipt",
                        ));
                    }
                    self.registry.discard_delivered(&ticket)?;
                }
                self.bindings[index].retired = true;
            }
            TicketNativeEvent::AcceptanceRejected { permit, error, .. } => {
                self.retained_permits.push(permit);
                return Err(error.into());
            }
            TicketNativeEvent::Finished {
                physical_shutdown_confirmed,
            } => {
                if self.native_finished.is_some() {
                    return Err(AdaptiveManagerError::Invalid(
                        "duplicate P/C process shutdown",
                    ));
                }
                self.native_finished = Some(physical_shutdown_confirmed);
                if !physical_shutdown_confirmed {
                    return Err(AdaptiveManagerError::PhysicalUnknown);
                }
            }
            TicketNativeEvent::RootQuiesced(fence) => {
                if !self.root_quiesce_requested
                    || fence.authority() != self.registration.authority
                    || !fence.permits_reuse()
                    || self.native_root_fence.is_some()
                {
                    return Err(AdaptiveManagerError::PhysicalUnknown);
                }
                self.native_root_fence = Some(fence);
            }
        }
        Ok(())
    }
    pub fn begin_root_quiesce(&mut self, until: Instant) -> Result<(), AdaptiveManagerError> {
        self.begin_close(until, AdaptiveCloseMode::RootQuiesce)
    }
    /// A normal goal close is separate from stop/replacement/expiry. The
    /// actual semantic owner must verify its original live satisfied goal;
    /// a copied snapshot or an empty role queue cannot authorize this path.
    pub fn begin_goal_quiesce(&mut self, until: Instant) -> Result<(), AdaptiveManagerError> {
        if self.phase != ManagerPhase::Running
            || self.shutdown_until.is_some()
            || !self.failures.is_empty()
            || self.effective_hard_expired()
        {
            return Err(AdaptiveManagerError::Invalid(
                "satisfied goal cannot replace prior root termination",
            ));
        }
        self.now()?;
        self.driver.quiesce_satisfied_goal()?;
        if !self.driver.is_complete() {
            return Err(AdaptiveManagerError::Invalid(
                "satisfied goal did not close role admission",
            ));
        }
        self.begin_close(until, AdaptiveCloseMode::RootQuiesce)
    }
    pub fn begin_shutdown(&mut self, until: Instant) -> Result<(), AdaptiveManagerError> {
        self.begin_close(until, AdaptiveCloseMode::ProcessShutdown)
    }
    fn begin_close(
        &mut self,
        until: Instant,
        mode: AdaptiveCloseMode,
    ) -> Result<(), AdaptiveManagerError> {
        let until = self
            .cleanup_boundary
            .map_or(until, |boundary| boundary.cap(until));
        if self.phase == ManagerPhase::Finished {
            return Ok(());
        }
        let escalation = self.shutdown_until.is_some()
            && self.close_mode == AdaptiveCloseMode::RootQuiesce
            && mode == AdaptiveCloseMode::ProcessShutdown;
        if self.shutdown_until.is_some() && self.close_mode != mode && !escalation {
            return Err(AdaptiveManagerError::Invalid("P/C closure mode change"));
        }
        let until = if escalation {
            until
        } else {
            self.shutdown_until.map_or(until, |old| old.min(until))
        };
        self.close_mode = mode;
        self.shutdown_until = Some(until);
        if let Some(preparation) = &self.shared_nnuec {
            preparation.canceled.store(true, Ordering::Release);
        }
        if self.phase != ManagerPhase::Quarantined {
            self.phase = ManagerPhase::Draining;
        }
        self.registry.revoke_all();
        if !self.driver.is_complete() {
            self.driver.revoke()?;
        }
        if let Some(input) = self.pending_input.as_ref() {
            self.driver.role_discarded(input, None)?;
        }
        self.pending_input = None;
        // prepare's returned question proves this owner never admitted its
        // physical lease. Keep it through semantic discard, then release it.
        self.pending_preparation = None;
        for binding in &mut self.bindings {
            binding.cancellation_requested |= self.native.cancel(binding.preparation).is_ok();
        }
        match mode {
            AdaptiveCloseMode::RootQuiesce => {
                self.native.begin_root_quiesce(until)?;
                self.root_quiesce_requested = true;
            }
            AdaptiveCloseMode::ProcessShutdown => self.native.begin_shutdown(until),
        }
        Ok(())
    }
    fn cleanup_until(&self) -> Result<Instant, AdaptiveManagerError> {
        automatic_cleanup_deadline(self.cleanup_boundary)
    }
    pub fn tick(&mut self) -> Result<bool, AdaptiveManagerError> {
        if self.phase == ManagerPhase::Finished {
            return Ok(false);
        }
        if self.phase == ManagerPhase::Running && self.effective_hard_expired() {
            self.begin_root_quiesce(self.cleanup_until()?)?;
        }
        let mut changed = false;
        for _ in 0..MAX_EVENT_DRAIN {
            let Some(event) = self.native.poll() else {
                break;
            };
            changed = true;
            let collection_failure = self
                .collection
                .as_mut()
                .and_then(|observer| observer.native_event(&event).err());
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            let collection_failure_v2 = self
                .collection_v2
                .as_mut()
                .and_then(|observer| observer.native_event(&event).err());
            if let Err(error) = self.event(event) {
                self.record_failure(error);
            }
            if let Some(error) = collection_failure {
                self.record_failure(AdaptiveManagerError::Collection(error));
            }
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            if let Some(error) = collection_failure_v2 {
                self.record_failure(AdaptiveManagerError::Collection(error));
            }
        }
        for action in [
            Self::flush_acks_changed,
            Self::deliver,
            Self::consume_retired,
        ] {
            match action(self) {
                Ok(progress) => changed |= progress,
                Err(error) => self.record_failure(error),
            }
        }
        if (self.phase == ManagerPhase::Running
            && (!self.failures.is_empty() || self.now().is_err()))
            || (self.phase == ManagerPhase::Quarantined && self.shutdown_until.is_none())
        {
            self.begin_close(self.cleanup_until()?, self.close_mode)?;
        }
        if self.phase == ManagerPhase::Running {
            match self.schedule() {
                Ok(progress) => changed |= progress,
                Err(error) => self.record_failure(error),
            }
            match self.dispatch() {
                Ok(progress) => changed |= progress,
                Err(error) => self.record_failure(error),
            }
            let spent = self.registry.cumulative_observation().admitted;
            if self.failures.is_empty()
                && self.bindings.is_empty()
                && self.pending_input.is_none()
                && spent >= self.registration.role_call_limit
            {
                if let Err(error) = self
                    .driver
                    .close_role_call_budget(spent, self.registration.role_call_limit)
                {
                    self.record_failure(error);
                }
            }
            if !self.failures.is_empty()
                || (self.bindings.is_empty()
                    && self.pending_input.is_none()
                    && (self.driver.is_complete() || spent >= self.registration.role_call_limit))
            {
                self.begin_close(self.cleanup_until()?, self.close_mode)?;
            }
        }
        if self.phase != ManagerPhase::Running {
            if self.native_finished.is_some() && !self.native_joined {
                self.native_joined = self.native.try_join()?;
                changed |= self.native_joined;
            }
            // A historical root fence can arrive after escalation to process
            // shutdown. Preserve it, but only a real process join closes that
            // newer command; the process cannot be reused through this fence.
            let fenced = match self.close_mode {
                AdaptiveCloseMode::RootQuiesce => self
                    .native_root_fence
                    .is_some_and(|fence| fence.permits_reuse()),
                AdaptiveCloseMode::ProcessShutdown => {
                    self.native_joined && self.native_finished == Some(true)
                }
            };
            if fenced
                && self.shared_readout_cleanup_settled()
                && self.phase != ManagerPhase::Quarantined
            {
                if !self.collection_settled {
                    if let Some(observer) = &mut self.collection {
                        let remaining: Vec<_> = self
                            .bindings
                            .iter()
                            .filter_map(|binding| binding.ticket)
                            .collect();
                        observer
                            .settled(&remaining)
                            .map_err(AdaptiveManagerError::Collection)?;
                    }
                    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                    if let Some(observer) = &mut self.collection_v2 {
                        let remaining: Vec<_> = self
                            .bindings
                            .iter()
                            .filter_map(|binding| binding.ticket)
                            .collect();
                        observer.settled(&remaining)?;
                    }
                    self.collection_settled = true;
                }
                self.registry.retire_revoked_after_worker_join()?;
                // Only the actual owner fence permits releasing unconsumed
                // revoked inputs or linear permits, never logical cancel.
                for binding in &self.bindings {
                    if !binding.consumption_recorded {
                        self.driver
                            .role_discarded(&binding.input, binding.ticket.as_ref())?;
                    }
                }
                self.counters.unused_role_results += self
                    .bindings
                    .iter()
                    .filter(|binding| !binding.consumption_recorded)
                    .count() as u64;
                self.bindings.clear();
                self.acks.clear();
                self.retained_receipts.clear();
                self.retained_permits.clear();
                if !self.retained_completions.is_empty() || self.native.is_quarantined() {
                    self.phase = ManagerPhase::Quarantined;
                } else {
                    self.phase = ManagerPhase::Finished;
                }
            }
        }
        Ok(changed)
    }
    fn flush_acks_changed(&mut self) -> Result<bool, AdaptiveManagerError> {
        let before = self.acks.len();
        self.flush_acks()?;
        Ok(self.acks.len() != before)
    }
    pub fn shutdown_receipt(&self) -> Result<RoleOnlyShutdownReceipt, AdaptiveManagerError> {
        let mut leases = self.review_leases;
        let actual = self.driver.review_lease_admission()?;
        leases.last = Some(actual);
        leases.effective_role_ticket_ceiling = self.pending_limit()?;
        Ok(RoleOnlyShutdownReceipt {
            authority: self.registration.authority,
            mode: self.registration.mode,
            root_budget: self.registration.root_budget,
            native_process_joined: self.native_joined,
            native_physical_shutdown_confirmed: self.native_finished == Some(true),
            pending_tickets: self.registry.occupancy(),
            retained_outputs: self.bindings.len()
                + self.retained_completions.len()
                + usize::from(self.pending_input.is_some())
                + usize::from(self.pending_preparation.is_some())
                + usize::from(!self.shared_readout_cleanup_settled()),
            quarantined: self.phase == ManagerPhase::Quarantined
                || self.native.is_quarantined()
                || self
                    .shared_nnuec
                    .as_ref()
                    .is_some_and(|preparation| preparation.unknown),
            request_sequence_high_water: self.registry.next_request_sequence().saturating_sub(1),
            counters: self.counters,
            canceled_acceptance_acks: self.canceled_acceptance_acks,
            expired_acceptance_acks: self.expired_acceptance_acks,
            review_leases: leases,
            native_root_fence: self.native_root_fence,
            close_mode: self.close_mode,
            shared_nnuec_required: self.driver.requires_shared_nnuec_readout(),
            shared_nnuec_closure: self
                .shared_nnuec
                .as_ref()
                .and_then(|preparation| preparation.closure.clone()),
        })
    }
    pub fn run_root_until(
        &mut self,
        until: Instant,
        canceled: &AtomicBool,
    ) -> Result<RoleOnlyShutdownReceipt, AdaptiveManagerError> {
        let until = self
            .cleanup_boundary
            .map_or(until, |boundary| boundary.cap(until));
        loop {
            let signal = self.native.signal();
            let observed = signal.version();
            if canceled.load(Ordering::Acquire) && self.phase == ManagerPhase::Running {
                self.begin_root_quiesce(self.cleanup_until()?)?;
            }
            let changed = self.tick()?;
            if self.phase == ManagerPhase::Finished {
                return self.shutdown_receipt();
            }
            if Instant::now() >= until {
                self.begin_close(until, self.close_mode)?;
                self.tick()?;
                return self.shutdown_receipt();
            }
            if !changed {
                self.counters.notification_waits += 1;
                signal.wait_changed(
                    observed,
                    until
                        .saturating_duration_since(Instant::now())
                        .min(Duration::from_millis(10)),
                );
            }
        }
    }
    #[allow(clippy::result_large_err)]
    pub fn into_sources(
        self,
    ) -> Result<RoleOnlyManagerSources<Q, B, D>, (Self, AdaptiveManagerError)> {
        let receipt = match self.shutdown_receipt() {
            Ok(receipt) => receipt,
            Err(error) => return Err((self, error)),
        };
        if self.phase != ManagerPhase::Finished || !receipt.permits_semantic_closure() {
            return Err((self, AdaptiveManagerError::PhysicalUnknown));
        }
        if self.has_collection_v2() {
            return Err((
                self,
                AdaptiveManagerError::Invalid("P/C V2 collection observer must be returned"),
            ));
        }
        Ok(RoleOnlyManagerSources {
            native: self.native,
            driver: self.driver,
        })
    }
}
