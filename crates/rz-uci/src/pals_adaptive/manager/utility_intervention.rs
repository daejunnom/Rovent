//! Opt-in /2 admission. The normal scheduler never advertises invented Ready
//! alternatives; an original neutral source is prepared before a new arm exists.
use super::*;
use crate::pals_adaptive::utility_intervention::*;
use rz_search::cpu_value::nnuec::v2::retirement::{NnuecCacheInvalidation, NnuecCacheReleaseAck};
use rz_search::cpu_value::nnuec::v2::review::{BoundNnuecReviewContext, PreparationActor};
use rz_search::pals::engine::adaptive_port::utility_route::{
    UtilityInterventionAdmission, UtilityInterventionCandidate, UtilityInterventionTarget,
    UtilityRouteRecipient,
};

pub trait UtilityInterventionSourceObserver<Q>: Send {
    fn source(
        &mut self,
        snapshot: UtilityInterventionSourceSnapshot<Q>,
    ) -> UtilityInterventionSourceDisposition<Q>;
}
pub enum UtilityInterventionSourceDisposition<Q> {
    ContinuePrelude,
    Trial(Box<UtilityInterventionTrialAdmission<Q>>),
    Rejected(Box<UtilityInterventionTrialAdmissionRejection<Q>>),
}
pub(super) struct UtilityInterventionState<Q> {
    pub(super) profile: Digest,
    pub(super) prelude_cutoff: Instant,
    pub(super) maximum_cleanup: Instant,
    pub(super) observer: Option<Box<dyn UtilityInterventionSourceObserver<Q>>>,
    pub(super) trial: Option<UtilityInterventionTrialAdmission<Q>>,
    pub(super) rejection: Option<UtilityInterventionTrialAdmissionRejection<Q>>,
    pub(super) window: Option<crate::pals_adaptive::UtilityTrialWindow>,
    pub(super) rejected_target: Option<UtilityInterventionTarget<Q>>,
    pub(super) rejected_source:
        Option<rz_search::pals::coordinator::UtilityInterventionReadySource<Q>>,
    pub(super) rejected_candidate: Option<UtilityInterventionCandidate<Q>>,
    pub(super) rejected_admission: Option<UtilityInterventionAdmission<Q>>,
    cache_invalidation: Option<NnuecCacheInvalidation>,
    cache_ack: Option<NnuecCacheReleaseAck>,
    cache_attached: bool,
}
impl<Q> UtilityInterventionState<Q> {
    /// Observe the original prelude without declaring a trial or a route.
    pub(super) fn awaiting_source(
        profile: Digest,
        prelude_cutoff: Instant,
        maximum_cleanup: Instant,
        observer: Box<dyn UtilityInterventionSourceObserver<Q>>,
    ) -> Self {
        Self {
            profile,
            prelude_cutoff,
            maximum_cleanup,
            observer: Some(observer),
            trial: None,
            rejection: None,
            window: None,
            rejected_target: None,
            rejected_source: None,
            rejected_candidate: None,
            rejected_admission: None,
            cache_invalidation: None,
            cache_ack: None,
            cache_attached: false,
        }
    }
}
#[cfg(test)]
struct ContinueOriginalPrelude;
#[cfg(test)]
impl<Q> UtilityInterventionSourceObserver<Q> for ContinueOriginalPrelude {
    fn source(
        &mut self,
        _: UtilityInterventionSourceSnapshot<Q>,
    ) -> UtilityInterventionSourceDisposition<Q> {
        UtilityInterventionSourceDisposition::ContinuePrelude
    }
}
impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    /// Keep the component probe inside the manager's private ownership boundary.
    /// It neither calls public Collection admission nor mints an actual trial.
    #[cfg(test)]
    pub(in crate::pals_adaptive) fn pump_nnuec_with_unselected_intervention_for_test(
        &mut self,
        actor: &PreparationActor,
        installed_bound_checked: &mut bool,
    ) -> Result<bool, AdaptiveManagerError> {
        assert!(self.utility_intervention.is_none());
        assert!(self.utility_route.is_none());
        self.utility_intervention = Some(UtilityInterventionState::awaiting_source(
            Digest([126; 32]),
            self.registration.hard_deadline,
            self.registered_cleanup_deadline()
                .expect("original registered cleanup"),
            Box::new(ContinueOriginalPrelude),
        ));
        // The same actor's real prepared_bound passes through utility_bind_bound
        // during this pump before its original R installation can continue.
        let pumped = self.pump_nnuec_preparation(actor);
        if !*installed_bound_checked
            && let Some(bound) = self
                .nnuec_jobs
                .first()
                .map(|custody| Arc::clone(&custody.bound))
        {
            assert!(bound.utility_input_capture().is_none());
            assert!(!self.utility_bind_bound(Arc::clone(&bound)).unwrap());
            // A declared route is no longer pure prelude. This ordinary R
            // Bound has no neutral capture and must still fail closed.
            self.declare_cohort_utility_route(0, UtilityRouteRecipient::Researcher, 1)
                .unwrap();
            let rejected = self.utility_bind_bound(bound);
            let route = self.utility_route.take().unwrap();
            assert!(route.action.is_none() && route.preparation.is_none());
            assert!(matches!(
                rejected,
                Err(AdaptiveManagerError::Invalid(
                    "intervention neutral Request has no capture"
                ))
            ));
            *installed_bound_checked = true;
        }
        let original = self.utility_intervention.take().unwrap();
        assert!(original.trial.is_none() && original.rejection.is_none());
        assert!(original.observer.is_some());
        assert!(self.utility_route.is_none());
        pumped
    }

    #[allow(clippy::result_large_err)]
    pub fn set_utility_intervention_source_observer(
        &mut self,
        profile: &rz_experiments::rovent_utility_cohort::RoventUtilityInterventionCohortV2,
        observer: Box<dyn UtilityInterventionSourceObserver<Q>>,
    ) -> Result<
        (),
        (
            AdaptiveManagerError,
            Box<dyn UtilityInterventionSourceObserver<Q>>,
        ),
    > {
        let checked = (|| {
            self.check_running()?;
            profile
                .validate()
                .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
            if self.root_started
                || self.utility_intervention.is_some()
                || self.utility_cohort.is_some()
                || self.utility_route.is_some()
                || self.exclusive_requested
                || self.registration.endpoints.load_v
                || self.peer_policy.is_none()
                || self.nnuec.is_none()
                || self.cpu_semantic.is_none()
                || self.registration.endpoints.execution_use != CpuExecutionUse::Collection
                || self.researcher_source_kind() != AdaptiveResearcherSourceKind::Native
                || self.cleanup_boundary.is_none()
            {
                return Err(AdaptiveManagerError::Invalid(
                    "intervention original Collection admission order",
                ));
            }
            let admission =
                self.collection_admission
                    .as_ref()
                    .ok_or(AdaptiveManagerError::Invalid(
                        "intervention Collection source missing",
                    ))?;
            self.driver.validate_collection_admission(admission)?;
            let actual = admission.root_admission();
            if actual.runtime_clock().origin() != self.clock.origin()
                || actual.runtime_clock().domain() != self.clock.domain()
            {
                return Err(AdaptiveManagerError::Invalid(
                    "intervention original clock differs",
                ));
            }
            let start = self
                .clock
                .origin()
                .checked_add(Duration::from_nanos(
                    actual.allocation_clock().input.start.at_ns,
                ))
                .ok_or(AdaptiveManagerError::Deadline)?;
            let prelude = start
                .checked_add(Duration::from_nanos(profile.input.prelude_ns))
                .ok_or(AdaptiveManagerError::Deadline)?;
            let action = prelude
                .checked_add(Duration::from_nanos(profile.input.action_ns))
                .ok_or(AdaptiveManagerError::Deadline)?;
            let cleanup = action
                .checked_add(Duration::from_nanos(profile.input.cleanup_ns))
                .ok_or(AdaptiveManagerError::Deadline)?;
            if action > self.registration.hard_deadline
                || cleanup
                    > self
                        .registered_cleanup_deadline()
                        .expect("checked original cleanup")
            {
                return Err(AdaptiveManagerError::Invalid(
                    "intervention windows exceed original root",
                ));
            }
            Ok((
                profile
                    .digest()
                    .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?,
                prelude,
                cleanup,
            ))
        })();
        match checked {
            Ok((profile, prelude_cutoff, maximum_cleanup)) => {
                self.utility_intervention = Some(UtilityInterventionState::awaiting_source(
                    profile,
                    prelude_cutoff,
                    maximum_cleanup,
                    observer,
                ));
                Ok(())
            }
            Err(e) => Err((e, observer)),
        }
    }
    pub(super) fn intervention_action_expired(&self, now: Instant) -> bool {
        self.utility_intervention
            .as_ref()
            .is_some_and(|s| now >= s.window.map_or(s.prelude_cutoff, |w| w.action_cutoff()))
    }
    pub(super) fn intervention_cleanup_ceiling(&self) -> Option<Instant> {
        self.utility_intervention
            .as_ref()
            .map(|s| s.window.map_or(s.maximum_cleanup, |w| w.cleanup_cutoff()))
    }
    pub(super) fn intervention_allows_frame(&self, frame: FrameId) -> bool {
        let Some(s) = &self.utility_intervention else {
            return true;
        };
        let Some(trial) = &s.trial else {
            return true;
        };
        if !self.utility_scope_open() {
            return trial.source_frame() == frame;
        }
        let Some(route) = &self.utility_route else {
            return false;
        };
        let Some(action) = &route.action else {
            return false;
        };
        // P/C and R both require the exact originally registered arm capture.
        // R's normal initial preparation must be allowed to create that capture.
        action.id().recipient() == UtilityRouteRecipient::Researcher
            || action.intervention_arm_capture().is_some()
    }
    pub(super) fn maybe_admit_utility_intervention(
        &mut self,
    ) -> Result<bool, AdaptiveManagerError> {
        let Some(s) = &self.utility_intervention else {
            return Ok(false);
        };
        if s.trial.is_some() || s.rejection.is_some() || s.observer.is_none() {
            return Ok(false);
        }
        let bounds = (s.profile, s.prelude_cutoff, s.maximum_cleanup);
        let actual = match self.capture_utility_source_snapshot(bounds.0, bounds.1, bounds.2) {
            Ok(v) => v,
            Err(AdaptiveManagerError::AuxiliaryPending) => return Ok(false),
            Err(e) => return Err(e),
        };
        let result = self
            .utility_intervention
            .as_mut()
            .expect("intervention")
            .observer
            .as_mut()
            .expect("observer")
            .source(UtilityInterventionSourceSnapshot { actual });
        match result {
            UtilityInterventionSourceDisposition::ContinuePrelude => Ok(false),
            UtilityInterventionSourceDisposition::Rejected(r) => {
                let r = *r;
                let error = r.error.clone();
                self.utility_intervention
                    .as_mut()
                    .expect("intervention")
                    .rejection = Some(r);
                Err(error)
            }
            UtilityInterventionSourceDisposition::Trial(trial) => {
                let trial = *trial;
                let valid = trial.profile_digest() == bounds.0
                    && trial.snapshot.authority() == self.registration.authority
                    && trial.snapshot.root_budget()
                        == self.driver.rovent_root_budget()?.ok_or(
                            AdaptiveManagerError::Invalid("intervention root budget absent"),
                        )?
                    && trial.snapshot.clock().domain() == self.clock.domain()
                    && trial.snapshot.clock().origin() == self.clock.origin();
                let source = trial
                    .snapshot
                    .actual
                    .offers()
                    .iter()
                    .find(|o| o.frame() == trial.source_frame())
                    .ok_or(AdaptiveManagerError::Invalid(
                        "intervention original source offer missing",
                    ))?;
                let recipient = source.recipient();
                let target = trial.profile().target_record;
                self.utility_intervention
                    .as_mut()
                    .expect("intervention")
                    .trial = Some(trial);
                if !valid {
                    return Err(AdaptiveManagerError::Invalid(
                        "intervention original trial registration differs",
                    ));
                }
                self.utility_route_sequence = self.utility_route_sequence.checked_add(1).ok_or(
                    AdaptiveManagerError::Capacity("intervention action sequence"),
                )?;
                self.declare_cohort_utility_route(target, recipient, self.utility_route_sequence)?;
                Ok(true)
            }
        }
    }
    pub(super) fn arm_utility_intervention(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        started: Instant,
    ) -> Result<(), AdaptiveManagerError> {
        let Some(s) = &mut self.utility_intervention else {
            return Ok(());
        };
        let trial = s.trial.as_mut().ok_or(AdaptiveManagerError::Invalid(
            "intervention original trial missing",
        ))?;
        if !trial.matches_source(work) {
            return Err(AdaptiveManagerError::Invalid(
                "intervention source Work allocation changed",
            ));
        }
        s.window = Some(trial.arm_window(started)?);
        Ok(())
    }
    /// Intercept an actual neutral/arm completion only after its source was
    /// selected. An unselected observer leaves the ordinary R installation live.
    pub(super) fn bind_intervention_neutral_bound(
        &mut self,
        bound: Arc<BoundNnuecReviewContext>,
    ) -> Result<bool, AdaptiveManagerError> {
        let Some(intervention) = &self.utility_intervention else {
            return Ok(false);
        };
        // The actor also completes ordinary R preparations while the original
        // observer waits for an eligible source. An observer alone cannot turn
        // those Bounds into neutral intervention inputs. Once a trial, route,
        // or rejection exists, retain the strict route/capture checks below.
        if intervention.trial.is_none()
            && self.utility_route.is_none()
            && intervention.rejection.is_none()
        {
            return Ok(false);
        }
        let state = self
            .utility_route
            .as_mut()
            .ok_or(AdaptiveManagerError::Invalid("intervention route missing"))?;
        if state.action.is_some() {
            return self.bind_intervention_arm_bound(bound);
        }
        let input = bound
            .utility_input_capture()
            .cloned()
            .ok_or(AdaptiveManagerError::Invalid(
                "intervention neutral Request has no capture",
            ))?;
        let prep = state
            .preparation
            .take()
            .ok_or(AdaptiveManagerError::Invalid(
                "intervention original preparation custody missing",
            ))?;
        let work = state
            .original
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "intervention source Work missing",
            ))?;
        let source = self.machine.capture_intervention_source(
            work.frame,
            self.clock.domain(),
            self.clock
                .try_now()
                .map_err(|_| AdaptiveManagerError::Deadline)?,
        )?;
        let target = match self
            .driver
            .bind_utility_intervention_target(prep, source, input)
        {
            Ok(t) => t,
            Err((error, prep, source, input)) => {
                state.preparation = Some(prep);
                state.historical_input = Some(input);
                state.bound = Some(bound);
                self.utility_intervention
                    .as_mut()
                    .expect("intervention")
                    .rejected_source = Some(source);
                return Err(error);
            }
        };
        state.bound = Some(bound);
        let intervention = self.utility_intervention.as_mut().expect("intervention");
        let trial = intervention.trial.as_mut().expect("trial");
        match trial.bind_target(&target) {
            Ok(true) => {}
            Ok(false) => {
                intervention.rejected_target = Some(target);
                let until = self
                    .registered_cleanup_deadline()
                    .ok_or(AdaptiveManagerError::Invalid("intervention cleanup absent"))?;
                self.begin_root_quiesce(until)?;
                return Ok(true);
            }
            Err(error) => {
                if matches!(error, AdaptiveManagerError::InterventionNeutralMismatch { .. }) {
                    // Preserve the exact rejected input as historical unbound
                    // evidence, never as the shared target or a valid label.
                    state.historical_input = Some(Arc::clone(target.input_capture()));
                }
                intervention.rejected_target = Some(target);
                state.failed = true;
                return Err(error);
            }
        }
        let desired = trial.recipient();
        let candidate = match self.driver.prepare_utility_intervention(target, desired) {
            Ok(c) => c,
            Err((error, target)) => {
                intervention.rejected_target = Some(target);
                state.failed = true;
                return Err(error);
            }
        };
        let now = self
            .clock
            .try_now()
            .map_err(|_| AdaptiveManagerError::Deadline)?;
        let (frame, admission) =
            match self
                .machine
                .admit_intervention(candidate, self.clock.domain(), now)
            {
                Ok(v) => v,
                Err((error, candidate)) => {
                    intervention.rejected_candidate = Some(candidate);
                    state.failed = true;
                    return Err(AdaptiveManagerError::Coordinator(error));
                }
            };
        let work = self.machine.ready_work(frame, self.clock.domain(), now)?;
        let action = match self
            .driver
            .bind_utility_intervention_route(admission, &work)
        {
            Ok(a) => a,
            Err((error, admission)) => {
                intervention.rejected_admission = Some(admission);
                state.failed = true;
                return Err(error);
            }
        };
        let metadata = self
            .driver
            .utility_intervention_arm_metadata(&action, &work)?;
        trial.bind_action(&action, metadata, state.cost_epoch)?;
        state.original = Some(work.clone());
        state.recipient = desired;
        state.action = Some(action);
        // Remove the original normal-ready descriptor. The coordinator has
        // consumed that exact frame; its retained source is not executable.
        self.ready_ids.retain(|id| *id != trial.source_frame());
        // The selected descriptor was re-admitted during neutral preparation.
        // If it has already been removed by the actual selector, no new one is minted.
        let _ = self.ready.remove(&trial.source_frame());
        if desired != UtilityRouteRecipient::Researcher {
            let model = &self.nnuec.as_ref().expect("NNUEC").model;
            let request = self.driver.prepare_utility_intervention_arm_request(
                state.action.as_mut().expect("action"),
                &work,
                model,
                Arc::clone(&state.cancel),
            )?;
            state.expected = Some(request.provenance().clone());
            state.request = Some(request);
        }
        Ok(true)
    }
    pub(super) fn bind_intervention_arm_bound(
        &mut self,
        bound: Arc<BoundNnuecReviewContext>,
    ) -> Result<bool, AdaptiveManagerError> {
        let Some(intervention) = &mut self.utility_intervention else {
            return Ok(false);
        };
        let state = self
            .utility_route
            .as_mut()
            .ok_or(AdaptiveManagerError::Invalid("intervention route missing"))?;
        let Some(action) = state.action.as_mut() else {
            return Ok(false);
        };
        if action.intervention_arm_capture().is_some() {
            return Ok(false);
        }
        let input = bound
            .utility_input_capture()
            .cloned()
            .ok_or(AdaptiveManagerError::Invalid(
                "intervention actual arm capture missing",
            ))?;
        let work = state.original.as_ref().expect("original arm Work");
        match self
            .driver
            .bind_utility_intervention_arm_capture(action, work, input)
        {
            Ok(()) => {}
            Err((error, input)) => {
                state.historical_input = Some(input);
                state.bound = Some(bound);
                state.failed = true;
                return Err(error);
            }
        }
        let metadata = self
            .driver
            .utility_intervention_arm_metadata(action, work)?;
        intervention
            .trial
            .as_mut()
            .expect("trial")
            .replace_arm_metadata(metadata)?;
        state.bound = Some(bound);
        Ok(true)
    }
    pub(super) fn pump_intervention_cache_prelude(
        &mut self,
        actor: &PreparationActor,
    ) -> Result<(bool, bool), AdaptiveManagerError> {
        let Some(s) = self
            .utility_intervention
            .as_mut()
            .filter(|s| s.window.is_some())
        else {
            return Ok((false, true));
        };
        if s.cache_attached {
            return Ok((false, true));
        }
        if self.phase != ManagerPhase::Running {
            return Ok((false, false));
        }
        if s.cache_ack.is_none() {
            if s.cache_invalidation.is_none() {
                s.cache_invalidation = Some(actor.invalidate_context_cache().map_err(|e| {
                    AdaptiveManagerError::NnuecPreparation(nnuec::NnuecWorkError::Preparation(e))
                })?);
                return Ok((true, false));
            }
            let ack = actor
                .try_context_cache_release(
                    s.cache_invalidation
                        .as_ref()
                        .expect("original invalidation"),
                )
                .map_err(|e| {
                    AdaptiveManagerError::NnuecPreparation(nnuec::NnuecWorkError::Preparation(e))
                })?;
            let Some(ack) = ack else {
                return Ok((false, false));
            };
            s.cache_ack = Some(ack);
        }
        let Some(request) = self.utility_route.as_mut().and_then(|r| r.request.take()) else {
            return Ok((false, false));
        };
        match request.with_utility_cache_prelude(s.cache_ack.take().expect("actual ACK")) {
            Ok(request) => {
                s.cache_attached = true;
                self.utility_route.as_mut().expect("route").request = Some(request);
                Ok((true, true))
            }
            Err((error, request, ack)) => {
                s.cache_ack = Some(ack);
                self.utility_route.as_mut().expect("route").request = Some(request);
                Err(AdaptiveManagerError::NnuecPreparation(
                    nnuec::NnuecWorkError::Preparation(error),
                ))
            }
        }
    }
    pub fn take_utility_intervention_source_observer(
        &mut self,
    ) -> Result<Option<Box<dyn UtilityInterventionSourceObserver<Q>>>, AdaptiveManagerError> {
        if self.phase != ManagerPhase::Finished
            || !self.shutdown_receipt()?.permits_semantic_closure()
        {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        Ok(self
            .utility_intervention
            .as_mut()
            .and_then(|s| s.observer.take()))
    }
    pub fn materialize_utility_intervention_trial(&mut self) -> Result<bool, AdaptiveManagerError> {
        if self.utility_intervention.is_none() {
            return Ok(false);
        }
        if self.phase != ManagerPhase::Finished
            || !self.shutdown_receipt()?.permits_semantic_closure()
        {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        let Some(route) = &self.utility_route else {
            return Ok(false);
        };
        let Some(outcome) = &route.outcome else {
            return Ok(false);
        };
        let bodies = self.driver.freeze_utility_witness_bodies(outcome);
        let trial = self
            .utility_intervention
            .as_mut()
            .expect("intervention")
            .trial
            .as_mut()
            .ok_or(AdaptiveManagerError::Invalid(
                "intervention original trial missing",
            ))?;
        match bodies {
            Ok(bodies) => {
                let _ = trial.materialize(outcome, &bodies, route.cost_epoch);
            }
            Err(error) => trial.materialization_error = Some(error),
        }
        // A detached-copy failure is retained as Unknown; the same actual
        // actor/engine still proceeds through its required release and drain.
        Ok(true)
    }
}
