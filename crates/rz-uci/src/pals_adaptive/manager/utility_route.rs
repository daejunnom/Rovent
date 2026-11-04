//! Collection-only utility actions have their own exclusive cost owner. A
//! declaration is not an action: the actual ready Work and private Search
//! source must be admitted before the single Session actor is borrowed.
use super::*;
use rz_search::cpu_value::nnuec::v2::review::{
    BoundNnuecReviewContext, Completion, NnuecPreparationTerminalAck, PreparationActor,
    PreparationError, PreparationObservation, Provenance, Request,
};
use rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture;
use rz_search::pals::coordinator::AcceptedRoleStep;
use rz_search::pals::engine::adaptive_port::utility_route::{
    UtilityRouteAction, UtilityRouteActionId, UtilityRouteCommitGuard, UtilityRoutePreparation,
    UtilityRouteRecipient, UtilityRouteSemanticOutcome, UtilityRouteUnresolvedAction,
    UtilityRouteUnresolvedReason,
};
use std::collections::HashMap;

/// Only a fully exclusive, physically closed action can mint this proof. Its
/// final boundary is later than the required same-actor evaluator/cache release.
pub struct UtilityRouteClosureProof {
    action: UtilityRouteActionId,
    authority: SearchAuthority,
    root_budget: Digest,
    sequence: u64,
    started: Instant,
    finished: Instant,
    baseline: AdaptiveManagerCounters,
    final_counters: AdaptiveManagerCounters,
    resource_profile: Digest,
    cost_epoch: Digest,
    native_baseline: (u64, u64),
    native_final: (u64, u64),
}
impl UtilityRouteClosureProof {
    pub fn action(&self) -> &UtilityRouteActionId {
        &self.action
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn root_budget(&self) -> Digest {
        self.root_budget
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn started(&self) -> Instant {
        self.started
    }
    pub fn finished(&self) -> Instant {
        self.finished
    }
    pub fn wall_cost(&self) -> Duration {
        self.finished.duration_since(self.started)
    }
    pub fn baseline_counters(&self) -> AdaptiveManagerCounters {
        self.baseline
    }
    pub fn final_counters(&self) -> AdaptiveManagerCounters {
        self.final_counters
    }
    pub fn resource_profile(&self) -> Digest {
        self.resource_profile
    }
    pub fn cost_epoch(&self) -> Digest {
        self.cost_epoch
    }
    pub fn native_credit_baseline(&self) -> (u64, u64) {
        self.native_baseline
    }
    pub fn native_credit_final(&self) -> (u64, u64) {
        self.native_final
    }
}

/// The root fence ends worker work, but not the Session's later actual TT/Bound
/// and actor-cache release. This move-only owner keeps that unfinished interval
/// explicit; a root receipt alone cannot become whole-cost training evidence.
pub struct UtilityRoutePendingClosure<Q> {
    state: UtilityRouteState<Q>,
    receipt: AdaptiveShutdownReceipt,
    root_finished: Instant,
    final_counters: AdaptiveManagerCounters,
    native_final: (u64, u64),
    expected_releases: Vec<ExpectedUtilityEngineRelease>,
    failed_proof: Option<UtilityRouteClosureProof>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    cohort: Option<super::utility_cohort::UtilityCohortTrialCustody<Q>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    cohort_materialized: Option<crate::pals_adaptive::UtilityTrialMaterializedSource>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    cohort_materialization_error: Option<AdaptiveManagerError>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    intervention: Option<crate::pals_adaptive::UtilityInterventionTrialAdmission<Q>>,
}
struct ExpectedUtilityEngineRelease {
    profile: CpuWorkerProfile,
    stack_owner: u64,
    binding: Digest,
    provenance: Provenance,
}
pub struct UtilityRouteFinalizationRejection<Q> {
    pub pending: UtilityRoutePendingClosure<Q>,
    pub error: AdaptiveManagerError,
}
/// A cohort observes its real whole interval separately from historical
/// root-local Completed actions. Only this manager's pending owner can mint
/// it, after the original physical release acknowledgements have matched.
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub struct UtilityCohortRootClosureProof {
    cohort: Digest,
    trial: u64,
    receipt: AdaptiveShutdownReceipt,
    started: Instant,
    root_finished: Instant,
    release_finished: Instant,
    baseline: AdaptiveManagerCounters,
    final_counters: AdaptiveManagerCounters,
    native_baseline: (u64, u64),
    native_final: (u64, u64),
    outcome: Option<UtilityRouteSemanticOutcome>,
    unresolved: Option<UtilityRouteUnresolvedAction>,
    historical_input: Option<Arc<NnuecUtilityInputCapture>>,
    reason: Option<&'static str>,
    materialized: Option<crate::pals_adaptive::UtilityTrialMaterializedSource>,
    materialization_error: Option<AdaptiveManagerError>,
}
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
impl UtilityCohortRootClosureProof {
    pub fn cohort_id(&self) -> Digest {
        self.cohort
    }
    pub fn trial_sequence(&self) -> u64 {
        self.trial
    }
    pub fn action(&self) -> Option<&UtilityRouteActionId> {
        self.outcome
            .as_ref()
            .map(|outcome| outcome.id())
            .or_else(|| self.unresolved.as_ref().map(|unresolved| unresolved.id()))
    }
    pub fn receipt(&self) -> &AdaptiveShutdownReceipt {
        &self.receipt
    }
    pub fn started(&self) -> Instant {
        self.started
    }
    pub fn root_finished(&self) -> Instant {
        self.root_finished
    }
    pub fn release_finished(&self) -> Instant {
        self.release_finished
    }
    pub fn baseline_counters(&self) -> AdaptiveManagerCounters {
        self.baseline
    }
    pub fn final_counters(&self) -> AdaptiveManagerCounters {
        self.final_counters
    }
    pub fn native_credit_baseline(&self) -> (u64, u64) {
        self.native_baseline
    }
    pub fn native_credit_final(&self) -> (u64, u64) {
        self.native_final
    }
    pub fn outcome(&self) -> Option<&UtilityRouteSemanticOutcome> {
        self.outcome.as_ref()
    }
    pub fn unresolved(&self) -> Option<&UtilityRouteUnresolvedAction> {
        self.unresolved.as_ref()
    }
    pub fn historical_input(&self) -> Option<&Arc<NnuecUtilityInputCapture>> {
        self.historical_input.as_ref()
    }
    pub fn reason(&self) -> Option<&'static str> {
        self.reason
    }
    pub fn materialization_error(&self) -> Option<&AdaptiveManagerError> {
        self.materialization_error.as_ref()
    }
    pub(crate) fn take_materialized(
        &mut self,
    ) -> Option<crate::pals_adaptive::UtilityTrialMaterializedSource> {
        self.materialized.take()
    }
}
impl<Q> UtilityRoutePendingClosure<Q> {
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn is_intervention(&self) -> bool {
        self.intervention.is_some()
    }
    pub fn is_cohort(&self) -> bool {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        {
            self.cohort.is_some()
        }
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        false
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub fn cohort_cleanup_deadline(&self) -> Option<Instant> {
        if let Some(trial) = &self.intervention {
            return Some(
                trial
                    .window()
                    .map_or(trial.snapshot.declared_cleanup_ceiling(), |w| {
                        w.cleanup_cutoff()
                    }),
            );
        }
        self.cohort.as_ref().map(|owner| {
            owner.trial().window().map_or(
                owner.trial().snapshot().declared_cleanup_ceiling(),
                |window| window.cleanup_cutoff(),
            )
        })
    }
    fn actual_release_matches(
        &self,
        source: &rz_search::cpu_worker::DeferredNnuecRSource,
        releases: &[rz_search::cpu_worker::NnuecEngineReleaseAck],
        cache: &rz_search::cpu_value::nnuec::v2::retirement::NnuecCacheReleaseAck,
    ) -> bool {
        self.receipt.permits_semantic_closure()
            && self.state.actor_settled()
            && !self.expected_releases.is_empty()
            && releases.len() == self.expected_releases.len()
            && self.expected_releases.iter().all(|expected| {
                releases
                    .iter()
                    .filter(|release| {
                        release.matches_source(source)
                            && release.matches_cache_release(cache)
                            && release.profile() == expected.profile
                            && release.actual_stack_owner_id() == expected.stack_owner
                            && release.binding_digest() == expected.binding
                            && release.provenance() == &expected.provenance
                    })
                    .count()
                    == 1
            })
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    #[allow(clippy::result_large_err)]
    pub(crate) fn finish_intervention_with_nnuec_release(
        mut self,
        source: &rz_search::cpu_worker::DeferredNnuecRSource,
        releases: &[rz_search::cpu_worker::NnuecEngineReleaseAck],
        cache: &rz_search::cpu_value::nnuec::v2::retirement::NnuecCacheReleaseAck,
    ) -> Result<
        (
            crate::pals_adaptive::UtilityInterventionTrialAdmission<Q>,
            UtilityCohortRootClosureProof,
        ),
        UtilityRouteFinalizationRejection<Q>,
    > {
        let captures_match = self.state.outcome.as_ref().is_none_or(|o| {
            o.physical_input_captures()
                .all(|i| i.matches_context_cache_release(cache))
        }) && self.state.unresolved.as_ref().is_none_or(|o| {
            o.physical_input_captures()
                .all(|i| i.matches_context_cache_release(cache))
        }) && self
            .state
            .historical_input
            .as_ref()
            .is_none_or(|i| i.matches_context_cache_release(cache));
        if self.intervention.is_none()
            || self.cohort.is_some()
            || !self.actual_release_matches(source, releases, cache)
            || !captures_match
            || self
                .state
                .terminal_ack
                .as_ref()
                .is_some_and(|a| !a.matches_cache_release(cache))
        {
            return Err(UtilityRouteFinalizationRejection {
                pending: self,
                error: AdaptiveManagerError::Invalid(
                    "intervention actual engine/cache/capture release mismatch",
                ),
            });
        }
        let release_finished = Instant::now();
        if release_finished < self.root_finished {
            return Err(UtilityRouteFinalizationRejection {
                pending: self,
                error: AdaptiveManagerError::Invalid(
                    "intervention release precedes actual root fence",
                ),
            });
        }
        let trial = self
            .intervention
            .take()
            .expect("validated original /2 trial");
        let reason = if let Some(reason) = trial.unavailable_reason() {
            Some(reason)
        } else if self.state.started.is_none() {
            Some("missing_actual_gate")
        } else if trial.materialization_error.is_some() {
            Some("historical_materialization_failed")
        } else if self.state.failed {
            Some("semantic_failure")
        } else if self.state.outcome.is_none() && self.state.preparation_interrupted {
            Some("preparation_interrupted")
        } else if self.state.outcome.is_none() {
            Some("missing_semantic_consumption")
        } else {
            None
        };
        let snapshot = &trial.snapshot.actual;
        let proof = UtilityCohortRootClosureProof {
            cohort: trial.cohort_id(),
            trial: trial.sequence(),
            receipt: self.receipt,
            started: self.state.started.unwrap_or(snapshot.captured_at()),
            root_finished: self.root_finished,
            release_finished,
            baseline: if self.state.started.is_some() {
                self.state.baseline
            } else {
                snapshot.counters()
            },
            final_counters: self.final_counters,
            native_baseline: if self.state.started.is_some() {
                self.state.native_baseline
            } else {
                (
                    snapshot.native_credit().spent_native_nodes,
                    snapshot.native_credit().reserved_native_nodes,
                )
            },
            native_final: self.native_final,
            outcome: self.state.outcome.take(),
            unresolved: self.state.unresolved.take(),
            historical_input: self.state.historical_input.take(),
            reason,
            materialized: None,
            materialization_error: None,
        };
        Ok((trial, proof))
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    #[allow(clippy::result_large_err)]
    pub(crate) fn finish_cohort_with_nnuec_release(
        mut self,
        source: &rz_search::cpu_worker::DeferredNnuecRSource,
        releases: &[rz_search::cpu_worker::NnuecEngineReleaseAck],
        cache: &rz_search::cpu_value::nnuec::v2::retirement::NnuecCacheReleaseAck,
    ) -> Result<
        (
            super::utility_cohort::UtilityCohortTrialCustody<Q>,
            UtilityCohortRootClosureProof,
        ),
        UtilityRouteFinalizationRejection<Q>,
    > {
        let input = self
            .state
            .outcome
            .as_ref()
            .map(|outcome| outcome.input())
            .or_else(|| {
                self.state
                    .unresolved
                    .as_ref()
                    .map(|unresolved| unresolved.input())
            })
            .or(self.state.historical_input.as_ref());
        let valid = self.cohort.is_some()
            && self.actual_release_matches(source, releases, cache)
            && input.is_none_or(|input| input.matches_context_cache_release(cache))
            && self
                .state
                .terminal_ack
                .as_ref()
                .is_none_or(|terminal| terminal.matches_cache_release(cache));
        if !valid {
            return Err(UtilityRouteFinalizationRejection {
                pending: self,
                error: AdaptiveManagerError::Invalid("cohort actual engine/cache release mismatch"),
            });
        }
        let release_finished = Instant::now();
        if release_finished < self.root_finished {
            return Err(UtilityRouteFinalizationRejection {
                pending: self,
                error: AdaptiveManagerError::Invalid("cohort release precedes root fence"),
            });
        }
        let trial = self.cohort.take().expect("validated original trial");
        let snapshot = trial.trial().snapshot();
        let reason = if self.state.started.is_none() {
            Some("missing_actual_gate")
        } else if self.cohort_materialization_error.is_some() {
            Some("historical_materialization_failed")
        } else if self.state.failed {
            Some("semantic_failure")
        } else if self.state.outcome.is_none()
            && trial
                .trial()
                .window()
                .is_some_and(|window| self.root_finished >= window.action_cutoff())
        {
            Some("cohort_action_cutoff")
        } else if self.state.outcome.is_none() && self.state.preparation_interrupted {
            Some("preparation_interrupted")
        } else if self.state.outcome.is_none() {
            Some("missing_semantic_consumption")
        } else {
            None
        };
        let baseline = if self.state.started.is_some() {
            self.state.baseline
        } else {
            snapshot.counters()
        };
        let native_baseline = if self.state.started.is_some() {
            self.state.native_baseline
        } else {
            (
                snapshot.native_credit().spent_native_nodes,
                snapshot.native_credit().reserved_native_nodes,
            )
        };
        let proof = UtilityCohortRootClosureProof {
            cohort: trial.trial().cohort_id(),
            trial: trial.trial().sequence(),
            started: self.state.started.unwrap_or(snapshot.captured_at()),
            receipt: self.receipt,
            root_finished: self.root_finished,
            release_finished,
            baseline,
            final_counters: self.final_counters,
            native_baseline,
            native_final: self.native_final,
            outcome: self.state.outcome.take(),
            unresolved: self.state.unresolved.take(),
            historical_input: self.state.historical_input.take(),
            reason,
            materialized: self.cohort_materialized.take(),
            materialization_error: self.cohort_materialization_error.take(),
        };
        Ok((trial, proof))
    }
    pub fn requested_recipient(&self) -> UtilityRouteRecipient {
        self.state.recipient
    }
    pub fn target_record(&self) -> u16 {
        self.state.target
    }
    pub fn preparation(&self) -> Option<&UtilityRoutePreparation> {
        self.state.preparation.as_ref()
    }
    /// No exact retained input-memory size is currently attested. This is not
    /// a zero-byte or released-cap claim.
    pub fn retained_host_bytes(&self) -> Option<u64> {
        None
    }
    pub fn rejected_cost_proof(&self) -> Option<&UtilityRouteClosureProof> {
        self.failed_proof.as_ref()
    }
    pub fn preparation_observation(&self) -> Option<&PreparationObservation> {
        self.state.preparation_observation.as_ref()
    }
    pub fn semantic(&self) -> Option<&UtilityRouteSemanticOutcome> {
        self.state.outcome.as_ref()
    }
    pub fn unresolved(&self) -> Option<&UtilityRouteUnresolvedAction> {
        self.state.unresolved.as_ref()
    }
    pub fn receipt(&self) -> &AdaptiveShutdownReceipt {
        &self.receipt
    }
    pub fn started(&self) -> Option<Instant> {
        self.state.started
    }
    pub fn root_finished(&self) -> Instant {
        self.root_finished
    }
    pub fn final_counters(&self) -> AdaptiveManagerCounters {
        self.final_counters
    }
    /// The same Session owns these unforgeable release ACKs. Worker closure,
    /// cache release, semantic facts and whole cost remain separate evidence.
    #[allow(clippy::result_large_err)]
    pub(crate) fn finish_with_nnuec_release(
        mut self,
        source: &rz_search::cpu_worker::DeferredNnuecRSource,
        releases: &[rz_search::cpu_worker::NnuecEngineReleaseAck],
        cache: &rz_search::cpu_value::nnuec::v2::retirement::NnuecCacheReleaseAck,
    ) -> Result<
        crate::pals_adaptive::CompletedUtilityRouteAction,
        UtilityRouteFinalizationRejection<Q>,
    > {
        let valid = !self.is_cohort()
            && {
                #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
                {
                    !self.is_intervention()
                }
                #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
                {
                    true
                }
            }
            && self.actual_release_matches(source, releases, cache)
            && !self.state.failed
            && self.state.semantics_closed
            && self.receipt.permits_semantic_closure()
            && self.state.outcome.as_ref().is_some_and(|outcome| {
                outcome.input().matches_context_cache_release(cache)
                    && (self.state.recipient == UtilityRouteRecipient::Researcher
                        || self.state.terminal_ack.as_ref().is_some_and(|terminal| {
                            terminal.allows_c_only_cleanup()
                                && outcome.input().matches_preparation_terminal(terminal)
                        }))
            });
        if !valid {
            return Err(UtilityRouteFinalizationRejection {
                pending: self,
                error: AdaptiveManagerError::Invalid(
                    "utility actual engine/cache/semantic closure mismatch",
                ),
            });
        }
        let finished = Instant::now();
        if finished < self.root_finished {
            return Err(UtilityRouteFinalizationRejection {
                pending: self,
                error: AdaptiveManagerError::Invalid("utility final cleanup precedes root fence"),
            });
        }
        let outcome = self
            .state
            .outcome
            .take()
            .expect("validated semantic outcome");
        let proof = UtilityRouteClosureProof {
            action: outcome.id().clone(),
            authority: self.receipt.authority,
            root_budget: outcome.root_budget(),
            sequence: self.state.sequence,
            started: self.state.started.expect("actual gate start"),
            finished,
            baseline: self.state.baseline,
            final_counters: self.final_counters,
            resource_profile: self.state.resource_profile,
            cost_epoch: self.state.cost_epoch,
            native_baseline: self.state.native_baseline,
            native_final: self.native_final,
        };
        match crate::pals_adaptive::CompletedUtilityRouteAction::after_manager_shutdown(
            outcome,
            proof,
            &self.receipt,
        ) {
            Ok(completed) => Ok(completed),
            Err(rejected) => {
                self.state.outcome = Some(rejected.semantic);
                self.failed_proof = Some(rejected.proof);
                self.state.failed = true;
                Err(UtilityRouteFinalizationRejection {
                    pending: self,
                    error: rejected.error,
                })
            }
        }
    }
    /// Original unresolved custody is returned intact; it does not claim a
    /// complete cost, actor release, zero service, or a successful target.
    pub fn into_unresolved(mut self, reason: UtilityRouteUnresolvedReason) -> Self {
        if let Some(action) = self.state.action.take() {
            self.state.unresolved = Some(action.into_unresolved(reason));
        }
        if let Some(outcome) = self.state.outcome.take() {
            self.state.unresolved = Some(outcome.into_unresolved(reason));
        }
        self
    }
}

pub(super) struct UtilityRouteState<Q> {
    pub(super) target: u16,
    pub(super) recipient: UtilityRouteRecipient,
    pub(super) sequence: u64,
    pub(super) started: Option<Instant>,
    pub(super) baseline: AdaptiveManagerCounters,
    pub(super) native_baseline: (u64, u64),
    pub(super) resource_profile: Digest,
    pub(super) cost_epoch: Digest,
    pub(super) original: Option<CoordinatorWork<ReviewView, Q>>,
    pub(super) preparation: Option<UtilityRoutePreparation>,
    pub(super) request: Option<Request>,
    pub(super) expected: Option<Provenance>,
    pub(super) actor_outstanding: bool,
    pub(super) bound: Option<Arc<BoundNnuecReviewContext>>,
    pub(super) terminal_ack: Option<NnuecPreparationTerminalAck>,
    pub(super) preparation_observation: Option<PreparationObservation>,
    pub(super) unmatched_completion: Option<Completion>,
    pub(super) historical_input: Option<Arc<NnuecUtilityInputCapture>>,
    pub(super) request_submitted: bool,
    pub(super) action: Option<UtilityRouteAction>,
    pub(super) outcome: Option<UtilityRouteSemanticOutcome>,
    pub(super) unresolved: Option<UtilityRouteUnresolvedAction>,
    pub(super) guards: HashMap<StepExecution, UtilityRouteCommitGuard>,
    pub(super) committed: Vec<StepExecution>,
    pub(super) failed: bool,
    pub(super) preparation_interrupted: bool,
    pub(super) semantics_closed: bool,
    pub(super) cancel: Arc<AtomicBool>,
}
impl<Q> UtilityRouteState<Q> {
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(super) fn action_for_cohort(&self) -> Option<&UtilityRouteAction> {
        self.action.as_ref()
    }
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(super) fn outcome_for_cohort(&self) -> Option<&UtilityRouteSemanticOutcome> {
        self.outcome.as_ref()
    }
    fn declared(target: u16, recipient: UtilityRouteRecipient, sequence: u64) -> Self {
        Self {
            target,
            recipient,
            sequence,
            started: None,
            baseline: AdaptiveManagerCounters::default(),
            native_baseline: (0, 0),
            resource_profile: Digest([0; 32]),
            cost_epoch: Digest([0; 32]),
            original: None,
            preparation: None,
            request: None,
            expected: None,
            actor_outstanding: false,
            bound: None,
            terminal_ack: None,
            preparation_observation: None,
            unmatched_completion: None,
            historical_input: None,
            request_submitted: false,
            action: None,
            outcome: None,
            unresolved: None,
            guards: HashMap::new(),
            committed: Vec::new(),
            failed: false,
            preparation_interrupted: false,
            semantics_closed: false,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }
    fn is_open(&self) -> bool {
        self.started.is_some()
    }
    fn actor_settled(&self) -> bool {
        !self.actor_outstanding && self.unmatched_completion.is_none()
    }
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    pub(super) fn declare_cohort_utility_route(
        &mut self,
        target: u16,
        recipient: UtilityRouteRecipient,
        sequence: u64,
    ) -> Result<(), AdaptiveManagerError> {
        if self.utility_route.is_some() || !self.root_started {
            return Err(AdaptiveManagerError::Invalid(
                "cohort actual route admission order",
            ));
        }
        self.utility_route = Some(UtilityRouteState::declared(target, recipient, sequence));
        Ok(())
    }
    /// The first actual matching Ready work is selected by the existing fair
    /// queue. No alternative offer, frame, input or support bit is synthesized.
    pub fn request_exclusive_utility_route(
        &mut self,
        target_record: u16,
        recipient: UtilityRouteRecipient,
    ) -> Result<(), AdaptiveManagerError> {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if self.utility_cohort.is_some() || self.utility_intervention.is_some() {
            return Err(AdaptiveManagerError::Invalid(
                "cohort cannot relabel root-local utility route",
            ));
        }
        if self.root_started
            || self.phase != ManagerPhase::Running
            || self.utility_route.is_some()
            || self.exclusive_requested
            || self.registration.endpoints.load_v
            || self.peer_policy.is_none()
            || self.nnuec.is_none()
            || self.registration.endpoints.execution_use != CpuExecutionUse::Collection
            || self.researcher_source_kind() != AdaptiveResearcherSourceKind::Native
            || self.collection_admission.is_none()
            || self.cleanup_boundary.is_none()
            || (recipient == UtilityRouteRecipient::Researcher && self.cpu_semantic.is_none())
        {
            return Err(AdaptiveManagerError::Invalid(
                "utility route requires original idle NNUEC Collection admission",
            ));
        }
        self.driver.validate_collection_admission(
            self.collection_admission
                .as_ref()
                .expect("checked admission"),
        )?;
        self.utility_route_sequence = self
            .utility_route_sequence
            .checked_add(1)
            .ok_or(AdaptiveManagerError::Capacity("utility route sequence"))?;
        self.utility_route = Some(UtilityRouteState::declared(
            target_record,
            recipient,
            self.utility_route_sequence,
        ));
        Ok(())
    }

    pub(super) fn utility_scope_open(&self) -> bool {
        self.utility_route
            .as_ref()
            .is_some_and(UtilityRouteState::is_open)
    }
    pub(super) fn utility_cleanup_settled(&self) -> bool {
        self.utility_route
            .as_ref()
            .is_none_or(UtilityRouteState::actor_settled)
    }
    pub(super) fn cancel_utility_preparation(&self) {
        if let Some(state) = &self.utility_route {
            state.cancel.store(true, Ordering::Release);
        }
    }
    pub(super) fn fail_utility_route(&mut self) {
        if let Some(state) = self.utility_route.as_mut().filter(|state| state.is_open()) {
            state.failed = true;
            if let Some(action) = &mut state.action {
                action.mark_failed();
            }
        }
    }
    pub(super) fn utility_allows_frame(&self, frame: FrameId) -> bool {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if !self.intervention_allows_frame(frame) {
            return false;
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if self.utility_cohort.is_some() {
            let Some(entry) = self.machine.frame(frame) else {
                return false;
            };
            let work = CoordinatorWork {
                frame: entry.id,
                original_first_move: entry.original_first_move,
                original_deadline: entry.original_deadline,
                budget: entry.budget,
                input: entry.input.clone(),
            };
            if !self.cohort_allows_work(&work) {
                return false;
            }
        }
        let Some(state) = self.utility_route.as_ref().filter(|state| state.is_open()) else {
            return true;
        };
        if state.failed || state.semantics_closed {
            return false;
        }
        let Some(action) = &state.action else {
            return false;
        };
        self.machine
            .frames()
            .find(|entry| entry.id == frame)
            .is_some_and(|entry| {
                let work = CoordinatorWork {
                    frame: entry.id,
                    original_first_move: entry.original_first_move,
                    original_deadline: entry.original_deadline,
                    budget: entry.budget,
                    input: entry.input.clone(),
                };
                self.driver
                    .validate_utility_route_work(action, &work)
                    .is_ok()
            })
    }

    /// Gate start precedes even private preparation admission. Already active
    /// foreign work or another actual owner makes this interval unavailable.
    pub(super) fn maybe_start_utility_route(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
    ) -> Result<bool, AdaptiveManagerError> {
        let Some(state) = self.utility_route.as_ref() else {
            return Ok(false);
        };
        if state.is_open() {
            return Ok(false);
        }
        let recipient = match work.input.stage.role() {
            Some(Role::Proposer) => UtilityRouteRecipient::Proposer,
            Some(Role::Critic) => UtilityRouteRecipient::Critic,
            Some(Role::Validator) => return Ok(false),
            None => UtilityRouteRecipient::Researcher,
        };
        if recipient != state.recipient {
            return Ok(false);
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if !self.cohort_allows_work(work) {
            return Err(AdaptiveManagerError::Invalid(
                "cohort selected original Work changed",
            ));
        }
        if !self.roles.is_empty()
            || !self.accepted_roles.is_empty()
            || self.own.iter().any(|slot| slot.binding.is_some())
            || self.foreign.is_some()
            || self.profiled_busy()
            || !self.deferred.is_empty()
            || self.registry.occupancy() != 0
            || self.native.occupancy() != 0
            || self.native_credit_outstanding()?
            || self.nnuec_pending.is_some()
            || !self.driver.semantic_owner_available()
        {
            return Err(AdaptiveManagerError::AuxiliaryPending);
        }
        let started = Instant::now();
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        self.arm_utility_cohort(work, started)?;
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        self.arm_utility_intervention(work, started)?;
        let native_baseline = self.native_ledger()?.global_usage()?;
        let mut profile = Sha256::new();
        profile.update(b"rz-rovent-utility-original-resource-owner/1\0");
        profile.update(self.registration.device_digest.0);
        profile.update(self.registration.cpu_device_digest.0);
        profile.update((self.own.len() as u64).to_le_bytes());
        profile.update((self.limits.role_tickets_max as u64).to_le_bytes());
        profile.update(self.registration.native_cpu_node_limit.to_le_bytes());
        // This binds the original loaded configuration, not a fabricated OS
        // enforcement observation. Different root windows cannot be paired.
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        profile.update(
            self.collection_admission
                .as_ref()
                .expect("checked Collection")
                .native_origin()
                .raw_sha256()
                .as_bytes(),
        );
        let resource_profile = Digest(profile.finalize().into());
        let mut epoch = Sha256::new();
        epoch.update(b"rz-rovent-utility-root-cost-window/1\0");
        epoch.update(resource_profile.0);
        epoch.update(work.original_deadline.at.0.to_le_bytes());
        epoch.update(self.registration.authority.epoch.0.to_le_bytes());
        epoch.update(self.registration.authority.root.0.to_le_bytes());
        epoch.update(
            self.registration
                .hard_deadline
                .saturating_duration_since(started)
                .as_nanos()
                .to_le_bytes(),
        );
        let cost_epoch = Digest(epoch.finalize().into());
        let state = self.utility_route.as_mut().expect("declared utility route");
        state.started = Some(started);
        state.baseline = self.counters;
        state.native_baseline = native_baseline;
        state.resource_profile = resource_profile;
        state.cost_epoch = cost_epoch;
        state.original = Some(work.clone());
        // Preserve the opened interval even if source admission fails.
        let preparation = self.driver.prepare_utility_route(work, state.target)?;
        state.preparation = Some(preparation);
        if recipient != UtilityRouteRecipient::Researcher || {
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            {
                self.utility_intervention.is_some()
            }
            #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
            {
                false
            }
        } {
            let model = &self.nnuec.as_ref().expect("checked NNUEC root").model;
            let request = self.driver.prepare_utility_route_input_request(
                state.preparation.as_mut().expect("source preparation"),
                work,
                model,
                Arc::clone(&state.cancel),
            )?;
            state.expected = Some(request.provenance().clone());
            state.request = Some(request);
        }
        Ok(true)
    }

    pub(super) fn utility_attach_r_request(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        request: Request,
    ) -> Result<Request, AdaptiveManagerError> {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if self.utility_intervention.is_some() {
            if let Some(state) = self
                .utility_route
                .as_mut()
                .filter(|s| s.action.is_some() && s.recipient == UtilityRouteRecipient::Researcher)
            {
                let action = state.action.as_mut().expect("actual intervention action");
                return match self
                    .driver
                    .attach_utility_intervention_arm_request(action, work, request)
                {
                    Ok(request) => {
                        state.expected = Some(request.provenance().clone());
                        Ok(request)
                    }
                    Err((error, request)) => {
                        state.request = Some(request);
                        state.failed = true;
                        Err(error)
                    }
                };
            }
        }
        let Some(state) = self.utility_route.as_mut().filter(|state| {
            state.is_open()
                && state.action.is_none()
                && state.recipient == UtilityRouteRecipient::Researcher
        }) else {
            return Ok(request);
        };
        if state.original.as_ref().is_none_or(|original| {
            original.frame != work.frame
                || original.input.question_digest != work.input.question_digest
        }) {
            state.request = Some(request);
            state.failed = true;
            return Err(AdaptiveManagerError::Invalid(
                "utility R original Request changed",
            ));
        }
        match state
            .preparation
            .as_mut()
            .expect("R source preparation")
            .attach_input_capture(request)
        {
            Ok(request) => {
                state.expected = Some(request.provenance().clone());
                Ok(request)
            }
            Err((error, request)) => {
                state.request = Some(request);
                state.failed = true;
                Err(AdaptiveManagerError::Semantic(format!("{error:?}")))
            }
        }
    }

    pub(super) fn utility_bind_bound(
        &mut self,
        bound: Arc<BoundNnuecReviewContext>,
    ) -> Result<bool, AdaptiveManagerError> {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if self.utility_intervention.is_some() {
            return self.bind_intervention_neutral_bound(bound);
        }
        let Some(state) = self
            .utility_route
            .as_mut()
            .filter(|state| state.is_open() && state.action.is_none())
        else {
            return Ok(false);
        };
        let input = bound
            .utility_input_capture()
            .cloned()
            .ok_or(AdaptiveManagerError::Invalid(
                "utility original preparation has no actual input capture",
            ))?;
        let prep = state
            .preparation
            .take()
            .ok_or(AdaptiveManagerError::Invalid(
                "utility preparation custody missing",
            ))?;
        let work = state.original.as_ref().expect("original utility Work");
        let bound_action = match self.driver.bind_utility_route_input(prep, work, input) {
            Ok(action) => {
                state.action = Some(action);
                state.bound = Some(bound);
                true
            }
            Err((error, prep, input)) => {
                state.preparation = Some(prep);
                state.historical_input = Some(input);
                state.bound = Some(bound);
                return Err(error);
            }
        };
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        self.bind_utility_cohort_action()?;
        Ok(bound_action)
    }
    pub(super) fn retain_utility_preparation_cleanup(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        input: Option<Arc<NnuecUtilityInputCapture>>,
        terminal: Option<NnuecPreparationTerminalAck>,
        observation: Option<PreparationObservation>,
        submitted: bool,
    ) -> Result<(), AdaptiveManagerError> {
        let Some(state) = self.utility_route.as_mut().filter(|state| state.is_open()) else {
            return Ok(());
        };
        if state.action.is_some()
            || state.outcome.is_some()
            || state.recipient != UtilityRouteRecipient::Researcher
        {
            return Ok(());
        }
        if state.original.as_ref().is_none_or(|original| {
            original.frame != work.frame
                || !Arc::ptr_eq(&original.input.question, &work.input.question)
                || !Arc::ptr_eq(&original.input.view, &work.input.view)
        }) {
            return Err(AdaptiveManagerError::Invalid(
                "utility cleanup original Work mismatch",
            ));
        }
        state.historical_input = input.or_else(|| state.historical_input.take());
        state.terminal_ack = terminal.or_else(|| state.terminal_ack.take());
        state.preparation_observation = observation.or(state.preparation_observation);
        state.request_submitted |= submitted;
        Ok(())
    }

    /// Uses the same Session actor before its ordinary R pump. There is only
    /// one outstanding utility preparation and no competing completion drain.
    pub(super) fn pump_utility_preparation(
        &mut self,
        actor: &PreparationActor,
    ) -> Result<bool, AdaptiveManagerError> {
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        {
            let (changed, ready) = self.pump_intervention_cache_prelude(actor)?;
            if !ready {
                return Ok(changed);
            }
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        {
            let (changed, ready) = self.pump_cohort_cache_prelude(actor)?;
            if !ready {
                return Ok(changed);
            }
            if self.cohort_cache_request_pending()
                && self
                    .utility_route
                    .as_ref()
                    .is_some_and(|state| state.request.is_some())
            {
                let request = self
                    .utility_route
                    .as_mut()
                    .expect("cohort route")
                    .request
                    .take()
                    .expect("actual request");
                let request = match self.attach_cohort_cache_prelude(request) {
                    Ok(request) => request,
                    Err((error, request)) => {
                        self.utility_route.as_mut().expect("cohort route").request = Some(request);
                        return Err(error);
                    }
                };
                self.utility_route.as_mut().expect("cohort route").request = Some(request);
            }
        }
        let may_submit = self.phase == ManagerPhase::Running && self.can_admit_work();
        let retry_bound = self
            .utility_route
            .as_ref()
            .filter(|state| {
                state.recipient != UtilityRouteRecipient::Researcher
                    && state.action.is_none()
                    && state.preparation.is_some()
                    && !state.failed
                    && may_submit
            })
            .and_then(|state| state.bound.clone());
        if let Some(bound) = retry_bound {
            return self.utility_bind_bound(bound);
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let neutral_intervention = self.utility_intervention.is_some()
            && self
                .utility_route
                .as_ref()
                .is_some_and(|s| s.action.is_none());
        #[cfg(not(all(feature = "pals-adaptive", feature = "onnx-cpu")))]
        let neutral_intervention = false;
        let Some(state) = self.utility_route.as_mut().filter(|state| {
            state.is_open()
                && (state.recipient != UtilityRouteRecipient::Researcher || neutral_intervention)
        }) else {
            return Ok(false);
        };
        if let Some(request) = state.request.take() {
            if !may_submit {
                state.request = Some(request);
                return Ok(false);
            }
            match actor.try_submit(request) {
                Ok(()) => {
                    state.actor_outstanding = true;
                    state.request_submitted = true;
                    return Ok(true);
                }
                Err((PreparationError::Busy, request)) => {
                    state.request = Some(request);
                    return Ok(false);
                }
                Err((error, request)) => {
                    state.request = Some(request);
                    state.failed = true;
                    return Err(AdaptiveManagerError::NnuecPreparation(
                        nnuec::NnuecWorkError::Preparation(error),
                    ));
                }
            }
        }
        if !state.actor_outstanding {
            return Ok(false);
        }
        let Some(completion) = actor.try_complete().map_err(|error| {
            AdaptiveManagerError::NnuecPreparation(nnuec::NnuecWorkError::Preparation(error))
        })?
        else {
            return Ok(false);
        };
        if state.expected.as_ref() != Some(&completion.provenance) {
            state.unmatched_completion = Some(completion);
            state.failed = true;
            return Err(AdaptiveManagerError::Invalid(
                "utility actor returned another original Request",
            ));
        }
        state.actor_outstanding = false;
        state.terminal_ack = Some(completion.terminal_ack);
        state.preparation_observation = Some(completion.observation);
        match completion.result {
            Ok(bound) if may_submit => {
                self.utility_bind_bound(Arc::new(bound))?;
            }
            Ok(bound) => {
                state.historical_input = bound.utility_input_capture().cloned();
                state.bound = Some(Arc::new(bound));
                state.preparation_interrupted = true;
            }
            Err(PreparationError::Cancelled | PreparationError::Expired)
                if self.phase != ManagerPhase::Running =>
            {
                // The actual terminal and observation were retained above.
                // This is an interrupted preparation with a real cleanup
                // obligation, not a fabricated successful utility action.
                state.preparation_interrupted = true;
            }
            Err(error) => {
                state.failed = true;
                return Err(AdaptiveManagerError::NnuecPreparation(
                    nnuec::NnuecWorkError::Preparation(error),
                ));
            }
        }
        Ok(true)
    }

    pub(super) fn utility_begin_commit(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        execution: StepExecution,
    ) -> Result<(), AdaptiveManagerError> {
        let Some(state) = self.utility_route.as_mut().filter(|state| state.is_open()) else {
            return Ok(());
        };
        if state.committed.contains(&execution) || state.guards.contains_key(&execution) {
            return Ok(());
        }
        let action = state.action.as_ref().ok_or(AdaptiveManagerError::Invalid(
            "utility input not bound before actual commit",
        ))?;
        let guard = self
            .driver
            .begin_utility_route_commit(action, work, execution)?;
        state.guards.insert(execution, guard);
        Ok(())
    }
    pub(super) fn utility_cpu_committed(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        execution: StepExecution,
    ) -> Result<(), AdaptiveManagerError> {
        let Some(state) = self.utility_route.as_mut().filter(|state| state.is_open()) else {
            return Ok(());
        };
        self.driver.commit_utility_route_step(
            state
                .action
                .as_mut()
                .ok_or(AdaptiveManagerError::Invalid("utility action missing"))?,
            state
                .guards
                .get(&execution)
                .ok_or(AdaptiveManagerError::Invalid(
                    "utility original precommit guard missing",
                ))?,
            work,
        )?;
        if !state.committed.contains(&execution) {
            state.committed.push(execution);
        }
        Ok(())
    }
    pub(super) fn utility_await_committed(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        report: &CpuSliceReport,
        settled: Option<&SettledCpuSlice>,
    ) -> Result<(), AdaptiveManagerError> {
        if !self.utility_scope_open() {
            return Ok(());
        }
        let settled = settled.ok_or(AdaptiveManagerError::Invalid(
            "utility Await has no actual settled ledger receipt",
        ))?;
        if settled.ledger_identity() != &self.native_ledger()?.identity()
            || settled.grant() != report.grant
            || Some(settled.actual_work()) != report.actual_work
            || settled.admission_rejected()
            || settled.terminal()
        {
            return Err(AdaptiveManagerError::Invalid(
                "utility Await differs from original settled slice",
            ));
        }
        let Some(state) = self.utility_route.as_mut().filter(|state| state.is_open()) else {
            return Ok(());
        };
        self.driver.commit_utility_route_await(
            state.action.as_mut().ok_or(AdaptiveManagerError::Invalid(
                "utility original action missing at Await",
            ))?,
            work,
            report,
        )
    }
    pub(super) fn utility_register_descendant(
        &mut self,
        frame: FrameId,
    ) -> Result<(), AdaptiveManagerError> {
        let Some(state) = self.utility_route.as_mut().filter(|state| state.is_open()) else {
            return Ok(());
        };
        let entry = self
            .machine
            .frame(frame)
            .ok_or(AdaptiveManagerError::Invalid(
                "utility actual child disappeared",
            ))?;
        let work = CoordinatorWork {
            frame,
            original_first_move: entry.original_first_move,
            original_deadline: entry.original_deadline,
            budget: entry.budget,
            input: entry.input.clone(),
        };
        self.driver.register_utility_route_descendant(
            state.action.as_mut().ok_or(AdaptiveManagerError::Invalid(
                "utility descendant before original input",
            ))?,
            &work,
        )
    }
    pub(super) fn utility_role_committed(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        let execution = StepExecution::Role(accepted.ticket().id);
        let Some(state) = self.utility_route.as_mut().filter(|state| state.is_open()) else {
            return Ok(());
        };
        self.driver.commit_utility_route_role(
            state
                .action
                .as_mut()
                .ok_or(AdaptiveManagerError::Invalid("utility action missing"))?,
            state
                .guards
                .get(&execution)
                .ok_or(AdaptiveManagerError::Invalid(
                    "utility original precommit guard missing",
                ))?,
            work,
            accepted,
            output,
        )?;
        if !state.committed.contains(&execution) {
            state.committed.push(execution);
        }
        if state.recipient == UtilityRouteRecipient::Researcher {
            if let Some(collector) = &self.cpu_semantic {
                for feedback in collector.feedback() {
                    if !work
                        .input
                        .view
                        .records()
                        .iter()
                        .any(|record| record.cpu_observation == Some(feedback.observation()))
                    {
                        continue;
                    }
                    if let Some(witness) = self.driver.selected_cpu_pc_consumption(
                        &feedback,
                        &work.input,
                        accepted,
                        output,
                    )? {
                        self.driver.record_utility_route_r_pc_consumption(
                            state.action.as_mut().expect("actual action"),
                            &witness,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
    pub(super) fn utility_register_peer(
        &mut self,
        parent: FrameId,
        child: FrameId,
    ) -> Result<(), AdaptiveManagerError> {
        if !self.utility_scope_open() {
            return Ok(());
        }
        let now = self.check_running()?;
        let proof =
            self.machine
                .reciprocal_dependency_proof(parent, child, self.clock.domain(), now)?;
        let entry = self
            .machine
            .frame(child)
            .ok_or(AdaptiveManagerError::Invalid(
                "utility canonical joined child disappeared",
            ))?;
        let work = CoordinatorWork {
            frame: child,
            original_first_move: entry.original_first_move,
            original_deadline: entry.original_deadline,
            budget: entry.budget,
            input: entry.input.clone(),
        };
        let state = self.utility_route.as_mut().expect("original utility scope");
        self.driver.register_utility_route_peer(
            state.action.as_mut().ok_or(AdaptiveManagerError::Invalid(
                "utility peer before original input",
            ))?,
            &work,
            &proof,
        )
    }
    pub(super) fn utility_role_at(
        &mut self,
        index: usize,
        work: &CoordinatorWork<ReviewView, Q>,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        if !self.utility_scope_open() {
            return Ok(());
        }
        let proof =
            self.roles[index]
                .reciprocal_acceptance
                .take()
                .ok_or(AdaptiveManagerError::Invalid(
                    "utility source has no actual retired native acceptance",
                ))?;
        let result = self.utility_role_committed(work, &proof, output);
        self.roles[index].reciprocal_acceptance = Some(proof);
        result
    }

    pub(super) fn maybe_close_utility_route(&mut self) -> Result<bool, AdaptiveManagerError> {
        let Some(state) = self.utility_route.as_ref().filter(|state| state.is_open()) else {
            return Ok(false);
        };
        if self.phase != ManagerPhase::Running
            || state.semantics_closed
            || state.failed
            || !state
                .action
                .as_ref()
                .is_some_and(UtilityRouteAction::semantic_consumption_complete)
            || !self.roles.is_empty()
            || !self.accepted_roles.is_empty()
            || !self.deferred.is_empty()
            || self.own.iter().any(|slot| {
                slot.binding.as_ref().is_some_and(|binding| {
                    binding.reciprocal.is_some() || !binding.feedback_forwarded
                })
            })
        {
            return Ok(false);
        }
        self.check_running()?;
        let state = self.utility_route.as_mut().expect("utility scope");
        let action = state.action.take().expect("consumed utility action");
        match self.driver.finish_utility_route_semantics(action) {
            Ok(outcome) => {
                state.outcome = Some(outcome);
                state.semantics_closed = true;
            }
            Err((error, action)) => {
                state.action = Some(action);
                state.failed = true;
                return Err(error);
            }
        }
        let until = self
            .registered_cleanup_deadline()
            .ok_or(AdaptiveManagerError::Invalid(
                "utility original cleanup absent",
            ))?;
        self.begin_root_quiesce(until)?;
        Ok(true)
    }

    /// Explicit transfer is required before NNUEC owner extraction. The caller
    /// must finish same-actor engine/cache cleanup before any whole-cost proof.
    pub fn take_utility_route_retirement(
        &mut self,
    ) -> Result<Option<UtilityRoutePendingClosure<Q>>, AdaptiveManagerError> {
        if self.utility_route.is_none() {
            return Ok(None);
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        self.materialize_utility_cohort_trial()?;
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        self.materialize_utility_intervention_trial()?;
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if self
            .utility_intervention
            .as_ref()
            .is_some_and(|s| s.observer.is_some())
        {
            return Err(AdaptiveManagerError::Invalid(
                "intervention original source observer must be recovered",
            ));
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        if self
            .utility_cohort
            .as_ref()
            .is_some_and(|state| state.observer_retained())
        {
            return Err(AdaptiveManagerError::Invalid(
                "cohort original Ready observer must be recovered",
            ));
        }
        let receipt = self.shutdown_receipt()?;
        if self.phase != ManagerPhase::Finished
            || !receipt.permits_semantic_closure()
            || self
                .utility_route
                .as_ref()
                .is_some_and(|state| !state.actor_settled())
        {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        let native_final = self.native_ledger()?.global_usage()?;
        let root_finished = self.finished_at.ok_or(AdaptiveManagerError::Invalid(
            "utility actual root-fence finish time missing",
        ))?;
        let mut expected_releases = Vec::with_capacity(self.own.len());
        for slot in &mut self.own {
            if slot.nnuec_retirement.is_none() {
                let worker = slot
                    .worker
                    .as_mut()
                    .ok_or(AdaptiveManagerError::PhysicalUnknown)?;
                slot.nnuec_retirement = Some(worker.take_nnuec_retirement()?);
            }
            let owner = slot
                .nnuec_retirement
                .as_ref()
                .expect("same joined retirement owner");
            expected_releases.push(ExpectedUtilityEngineRelease {
                profile: owner.profile(),
                stack_owner: owner.actual_stack_owner_id(),
                binding: owner.binding_digest(),
                provenance: owner.provenance().clone(),
            });
        }
        let mut state = self.utility_route.take().expect("utility source");
        if !state.semantics_closed {
            if let Some(action) = state.action.take() {
                state.unresolved =
                    Some(action.into_unresolved(UtilityRouteUnresolvedReason::MissingConsumption));
            }
        }
        // Extra manager Bound custody ends here. The actor cache/evaluator
        // remain owned by the Session and require their own actual ACKs.
        state.bound = None;
        state.original = None;
        state.request = None;
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let (cohort, cohort_materialized, cohort_materialization_error) = self
            .utility_cohort
            .take()
            .map_or((None, None, None), |state| state.into_trial());
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let intervention = self
            .utility_intervention
            .take()
            .and_then(|mut state| state.trial.take());
        Ok(Some(UtilityRoutePendingClosure {
            state,
            receipt,
            root_finished,
            final_counters: self.counters,
            native_final,
            expected_releases,
            failed_proof: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            cohort,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            cohort_materialized,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            cohort_materialization_error,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            intervention,
        }))
    }
}
