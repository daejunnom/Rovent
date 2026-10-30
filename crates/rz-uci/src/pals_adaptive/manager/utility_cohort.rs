//! Actual idle-prelude offers for optional independent utility trials. This
//! file is separate from historical root-local utility costs: a cohort cannot
//! rewrite an earlier action's resource epoch or manufacture an available route.
use super::*;
use crate::pals_adaptive::utility_cohort::{
    UtilityTrialAdmission, UtilityTrialBoundAdmission, UtilityTrialWindow,
};
use crate::pals_adaptive::utility_cohort_historical::UtilityTrialMaterializedSource;
use rz_runtime::adaptive::{CpuResourceError, CpuThreadReadback};
use rz_search::cpu_value::nnuec::v2::retirement::{NnuecCacheInvalidation, NnuecCacheReleaseAck};
use rz_search::cpu_value::nnuec::v2::review::{PreparationActor, Request};
use rz_search::cpu_worker::CpuLedgerRegistrationObservation;
use rz_search::pals::engine::adaptive_port::UtilityCohortOfferSemanticSource;
use rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteRecipient;
use std::sync::Weak;

/// This callback runs once per real idle scheduling turn. It may only join
/// the already loaded cohort owner; it performs no I/O, inference or wait.
pub trait UtilityCohortReadyObserver<Q>: Send {
    fn ready(
        &mut self,
        snapshot: UtilityCohortReadySnapshot<Q>,
    ) -> UtilityCohortReadyDisposition<Q>;
}
pub enum UtilityCohortReadyDisposition<Q> {
    ContinuePrelude,
    Trial(UtilityTrialAdmission<Q>),
    Rejected(UtilityCohortReadyRejection<Q>),
}
pub struct UtilityCohortReadyRejection<Q> {
    pub snapshot: UtilityCohortReadySnapshot<Q>,
    pub error: AdaptiveManagerError,
}
pub(super) struct UtilityCohortState<Q> {
    target: u16,
    profile: Digest,
    prelude_cutoff: Instant,
    maximum_cleanup: Instant,
    observer: Option<Box<dyn UtilityCohortReadyObserver<Q>>>,
    trial: Option<UtilityTrialAdmission<Q>>,
    bound: Option<UtilityTrialBoundAdmission<Q>>,
    window: Option<UtilityTrialWindow>,
    rejection: Option<UtilityCohortReadyRejection<Q>>,
    cache_invalidation: Option<NnuecCacheInvalidation>,
    cache_ack: Option<NnuecCacheReleaseAck>,
    cache_attached: bool,
    materialized: Option<UtilityTrialMaterializedSource>,
    materialization_error: Option<AdaptiveManagerError>,
}
impl<Q> UtilityCohortState<Q> {
    fn declared(
        target: u16,
        profile: Digest,
        prelude_cutoff: Instant,
        maximum_cleanup: Instant,
        observer: Box<dyn UtilityCohortReadyObserver<Q>>,
    ) -> Self {
        Self {
            target,
            profile,
            prelude_cutoff,
            maximum_cleanup,
            observer: Some(observer),
            trial: None,
            bound: None,
            window: None,
            rejection: None,
            cache_invalidation: None,
            cache_ack: None,
            cache_attached: false,
            materialized: None,
            materialization_error: None,
        }
    }
    fn selected(&self) -> Option<&UtilityTrialAdmission<Q>> {
        self.trial
            .as_ref()
            .or_else(|| self.bound.as_ref().map(|b| b.trial()))
    }
    pub(super) fn observer_retained(&self) -> bool {
        self.observer.is_some()
    }
    pub(super) fn into_trial(
        self,
    ) -> (
        Option<UtilityCohortTrialCustody<Q>>,
        Option<UtilityTrialMaterializedSource>,
        Option<AdaptiveManagerError>,
    ) {
        let trial = if let Some(bound) = self.bound {
            Some(UtilityCohortTrialCustody::Bound(bound))
        } else {
            self.trial.map(UtilityCohortTrialCustody::Admitted)
        };
        (trial, self.materialized, self.materialization_error)
    }
}
// A manager selects one trial: binding moves that same admission into Bound,
// and rejection restores it. Retirement transfers this single custody slot.
#[allow(clippy::large_enum_variant)]
pub enum UtilityCohortTrialCustody<Q> {
    Admitted(UtilityTrialAdmission<Q>),
    Bound(UtilityTrialBoundAdmission<Q>),
}
impl<Q> UtilityCohortTrialCustody<Q> {
    pub fn trial(&self) -> &UtilityTrialAdmission<Q> {
        match self {
            Self::Admitted(trial) => trial,
            Self::Bound(bound) => bound.trial(),
        }
    }
}

/// One exact currently Ready source. A numeric FrameId or matching digest is
/// insufficient to substitute a different immutable Q/View allocation.
pub struct UtilityCohortReadyOffer<Q> {
    frame: FrameId,
    recipient: UtilityRouteRecipient,
    stage: RefinementStage,
    first: Move16,
    question: Digest,
    view: ReviewViewSeal,
    full_view: Digest,
    public_context: Digest,
    budget: FrameBudget,
    deadline: Deadline,
    query: Weak<Q>,
    selected: Weak<ReviewView>,
    semantic: UtilityCohortOfferSemanticSource<Q>,
}
impl<Q> UtilityCohortReadyOffer<Q> {
    /// Read-only source eligibility, not an alternative recipient offer.
    pub(in crate::pals_adaptive) fn intervention_target_matches(&self, target: u16) -> bool {
        self.selected.upgrade().is_some_and(|view| {
            view.records().get(usize::from(target)).is_some_and(|r| {
                r.kind == rz_search::pals::engine::RecordKind::Proposal && !r.line.is_empty()
            }) && view
                .records_metadata()
                .get(usize::from(target))
                .is_some_and(|m| {
                    m.anchor == 0
                        && m.original_first_move.is_some_and(|movement| {
                            rz_contracts::Move::try_from(movement)
                                .is_ok_and(|movement| Move16::encode(movement) == self.first)
                        })
                })
        })
    }
    pub fn semantic_source(&self) -> Option<&UtilityCohortOfferSemanticSource<Q>> {
        Some(&self.semantic)
    }
    pub fn frame(&self) -> FrameId {
        self.frame
    }
    pub fn recipient(&self) -> UtilityRouteRecipient {
        self.recipient
    }
    pub fn stage(&self) -> RefinementStage {
        self.stage
    }
    pub fn original_first_move(&self) -> Move16 {
        self.first
    }
    pub fn question(&self) -> Digest {
        self.question
    }
    pub fn view(&self) -> ReviewViewSeal {
        self.view
    }
    pub fn full_view(&self) -> Digest {
        self.full_view
    }
    pub fn public_context(&self) -> Digest {
        self.public_context
    }
    pub fn budget(&self) -> FrameBudget {
        self.budget
    }
    pub fn original_deadline(&self) -> Deadline {
        self.deadline
    }
    pub fn matches_ready_work(&self, work: &CoordinatorWork<ReviewView, Q>) -> bool {
        self.frame == work.frame
            && self.first == work.original_first_move
            && self.recipient.search_recipient()
                == match work.input.stage.role() {
                    Some(Role::Proposer) => SearchRecipient::Proposer,
                    Some(Role::Critic) => SearchRecipient::Critic,
                    Some(Role::Validator) => return false,
                    None => SearchRecipient::Researcher,
                }
            && self.stage == work.input.stage
            && self.question == work.input.question_digest
            && self.view == work.input.view_seal
            && self.full_view == Digest(work.input.view.digest())
            && self.public_context == work.input.public_context
            && self.budget == work.budget
            && self.deadline == work.original_deadline
            && self.query.ptr_eq(&Arc::downgrade(&work.input.question))
            && self.selected.ptr_eq(&Arc::downgrade(&work.input.view))
            && self.semantic.matches_work(work)
    }
}

/// Only capture_utility_cohort_ready_snapshot constructs this snapshot from
/// the real manager's clean prelude. It keeps source identity, never strong
/// ReviewView/model pins, and conveys no new credit or physical execution.
pub struct UtilityCohortReadySnapshot<Q> {
    admission: RoventCollectionRootAdmission,
    authority: SearchAuthority,
    root_budget: Digest,
    clock: ContractSystemClock,
    hard_deadline: Instant,
    cleanup_deadline: Instant,
    captured_at: Instant,
    profile: Digest,
    prelude_cutoff: Instant,
    maximum_cleanup: Instant,
    counters: AdaptiveManagerCounters,
    native_credit: CpuLedgerRegistrationObservation,
    native_ledger: CpuLedgerIdentity,
    common_nodes: Option<u64>,
    common_work: rz_search::pals::rovent_frontier::FrontierNodeWork,
    role_credit: (u64, u64),
    offers: Vec<UtilityCohortReadyOffer<Q>>,
    selected_offers: [Option<usize>; 3],
    native_threads: Option<Result<CpuThreadReadback, CpuResourceError>>,
    cpu_threads: Vec<Option<Result<CpuThreadReadback, CpuResourceError>>>,
}
impl<Q> UtilityCohortReadySnapshot<Q> {
    pub fn cohort_profile_digest(&self) -> Digest {
        self.profile
    }
    pub fn prelude_cutoff(&self) -> Instant {
        self.prelude_cutoff
    }
    pub fn declared_cleanup_ceiling(&self) -> Instant {
        self.maximum_cleanup
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn root_budget(&self) -> Digest {
        self.root_budget
    }
    pub fn clock(&self) -> &ContractSystemClock {
        &self.clock
    }
    pub fn hard_deadline(&self) -> Instant {
        self.hard_deadline
    }
    pub fn cleanup_deadline(&self) -> Instant {
        self.cleanup_deadline
    }
    pub fn captured_at(&self) -> Instant {
        self.captured_at
    }
    pub fn admission(&self) -> &RoventCollectionRootAdmission {
        &self.admission
    }
    pub fn actual_root(&self) -> &Arc<crate::rovent_root_admission::RoventRootAdmission> {
        self.admission.root_admission()
    }
    pub fn offers(&self) -> &[UtilityCohortReadyOffer<Q>] {
        &self.offers
    }
    pub fn selected_offer_index(&self, recipient: UtilityRouteRecipient) -> Option<usize> {
        self.selected_offers[recipient.output_index() - 4]
    }
    pub fn counters(&self) -> AdaptiveManagerCounters {
        self.counters
    }
    pub fn native_credit(&self) -> CpuLedgerRegistrationObservation {
        self.native_credit
    }
    pub fn native_ledger_identity(&self) -> &CpuLedgerIdentity {
        &self.native_ledger
    }
    pub fn requested_common_nodes(&self) -> Option<u64> {
        self.common_nodes
    }
    pub fn common_node_work(&self) -> rz_search::pals::rovent_frontier::FrontierNodeWork {
        self.common_work
    }
    pub fn role_credit(&self) -> (u64, u64) {
        self.role_credit
    }
    pub fn native_thread_observation(
        &self,
    ) -> Option<&Result<CpuThreadReadback, CpuResourceError>> {
        self.native_threads.as_ref()
    }
    pub fn own_thread_observations(
        &self,
    ) -> &[Option<Result<CpuThreadReadback, CpuResourceError>>] {
        &self.cpu_threads
    }
    pub fn matches_ready_work(&self, work: &CoordinatorWork<ReviewView, Q>) -> bool {
        self.offers
            .iter()
            .any(|offer| offer.matches_ready_work(work))
    }
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    pub fn set_utility_cohort_ready_observer(
        &mut self,
        target_record: u16,
        profile: &rz_experiments::rovent_utility_cohort::RoventUtilityComparisonCohort,
        observer: Box<dyn UtilityCohortReadyObserver<Q>>,
    ) -> Result<(), (AdaptiveManagerError, Box<dyn UtilityCohortReadyObserver<Q>>)> {
        let checked = (|| {
            self.check_running()?;
            if self.root_started
                || self.utility_cohort.is_some()
                || self.utility_intervention.is_some()
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
                    "cohort original admission order",
                ));
            }
            let admission =
                self.collection_admission
                    .as_ref()
                    .ok_or(AdaptiveManagerError::Invalid(
                        "cohort original Collection admission absent",
                    ))?;
            self.driver.validate_collection_admission(admission)?;
            profile
                .validate()
                .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
            let actual = admission.root_admission();
            if actual.runtime_clock().origin() != self.clock.origin()
                || actual.runtime_clock().domain() != self.clock.domain()
            {
                return Err(AdaptiveManagerError::Invalid(
                    "cohort original prelude clock differs",
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
                || cleanup > self.registered_cleanup_deadline().expect("checked cleanup")
            {
                return Err(AdaptiveManagerError::Invalid(
                    "cohort declared windows exceed original root",
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
            Ok((digest, prelude, cleanup)) => {
                self.utility_cohort = Some(UtilityCohortState::declared(
                    target_record,
                    digest,
                    prelude,
                    cleanup,
                    observer,
                ));
                Ok(())
            }
            Err(error) => Err((error, observer)),
        }
    }
    pub(super) fn cohort_allows_work(&self, work: &CoordinatorWork<ReviewView, Q>) -> bool {
        self.utility_cohort
            .as_ref()
            .and_then(UtilityCohortState::selected)
            .is_none_or(|trial| trial.matches_ready_work(work) || self.utility_scope_open())
    }
    pub(super) fn cohort_action_expired(&self, now: Instant) -> bool {
        self.intervention_action_expired(now)
            || self.utility_cohort.as_ref().is_some_and(|state| {
                now >= state
                    .window
                    .map_or(state.prelude_cutoff, |window| window.action_cutoff())
            })
    }
    pub(super) fn cohort_cleanup_ceiling(&self) -> Option<Instant> {
        self.intervention_cleanup_ceiling().or_else(|| {
            self.utility_cohort.as_ref().map(|state| {
                state
                    .window
                    .map_or(state.maximum_cleanup, |window| window.cleanup_cutoff())
            })
        })
    }
    /// A trial narrows future work only. All original submitted deadlines,
    /// ledger reservations and role/NN counts remain untouched.
    pub(super) fn maybe_admit_utility_cohort(&mut self) -> Result<bool, AdaptiveManagerError> {
        if self.utility_cohort.as_ref().is_none_or(|state| {
            state.observer.is_none() || state.selected().is_some() || state.rejection.is_some()
        }) {
            return Ok(false);
        }
        let snapshot = match self.capture_utility_cohort_ready_snapshot() {
            Ok(snapshot) => snapshot,
            Err(AdaptiveManagerError::AuxiliaryPending) => return Ok(false),
            Err(error) => return Err(error),
        };
        let decision = self
            .utility_cohort
            .as_mut()
            .expect("cohort")
            .observer
            .as_mut()
            .expect("cohort callback")
            .ready(snapshot);
        match decision {
            UtilityCohortReadyDisposition::ContinuePrelude => Ok(false),
            UtilityCohortReadyDisposition::Rejected(rejection) => {
                let error = rejection.error.clone();
                self.utility_cohort.as_mut().expect("cohort").rejection = Some(rejection);
                Err(error)
            }
            UtilityCohortReadyDisposition::Trial(trial) => {
                let checked = (|| {
                    trial.validate_manager_registration(
                        self.registration.authority,
                        self.driver
                            .rovent_root_budget()?
                            .ok_or(AdaptiveManagerError::Invalid("cohort Frontier absent"))?,
                        &self.clock,
                        self.registration.hard_deadline,
                        self.registered_cleanup_deadline()
                            .ok_or(AdaptiveManagerError::Invalid("cohort cleanup absent"))?,
                    )
                })();
                let target = self.utility_cohort.as_ref().expect("cohort").target;
                let checked = checked.and_then(|()| {
                    if trial.profile_digest()
                        != self.utility_cohort.as_ref().expect("cohort").profile
                    {
                        return Err(AdaptiveManagerError::Invalid(
                            "cohort original profile differs",
                        ));
                    }
                    Ok(())
                });
                let recipient = trial.recipient();
                self.utility_cohort.as_mut().expect("cohort").trial = Some(trial);
                self.utility_route_sequence = self
                    .utility_route_sequence
                    .checked_add(1)
                    .ok_or(AdaptiveManagerError::Capacity("utility route sequence"))?;
                self.declare_cohort_utility_route(target, recipient, self.utility_route_sequence)?;
                checked?;
                Ok(true)
            }
        }
    }
    pub(super) fn arm_utility_cohort(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        started: Instant,
    ) -> Result<(), AdaptiveManagerError> {
        let Some(state) = &mut self.utility_cohort else {
            return Ok(());
        };
        if started >= state.prelude_cutoff {
            return Err(AdaptiveManagerError::AdmissionClosed);
        }
        let trial = state.trial.as_mut().ok_or(AdaptiveManagerError::Invalid(
            "cohort original trial absent",
        ))?;
        if !trial.matches_ready_work(work) {
            return Err(AdaptiveManagerError::Invalid(
                "cohort selected original Work changed",
            ));
        }
        state.window = Some(trial.arm_window(started)?);
        Ok(())
    }
    pub(super) fn bind_utility_cohort_action(&mut self) -> Result<(), AdaptiveManagerError> {
        let Some(state) = &mut self.utility_cohort else {
            return Ok(());
        };
        if state.bound.is_some() {
            return Ok(());
        }
        let action = self
            .utility_route
            .as_ref()
            .and_then(|state| state.action_for_cohort())
            .ok_or(AdaptiveManagerError::Invalid("cohort actual Action absent"))?;
        let trial = state.trial.take().ok_or(AdaptiveManagerError::Invalid(
            "cohort original trial absent",
        ))?;
        match trial.bind_submitted_input(action) {
            Ok(bound) => {
                state.bound = Some(bound);
                Ok(())
            }
            Err(rejected) => {
                state.trial = Some(rejected.trial);
                Err(rejected.error)
            }
        }
    }
    pub fn take_utility_cohort_ready_observer(
        &mut self,
    ) -> Result<Option<Box<dyn UtilityCohortReadyObserver<Q>>>, AdaptiveManagerError> {
        let receipt = self.shutdown_receipt()?;
        if self.phase != ManagerPhase::Finished || !receipt.permits_semantic_closure() {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        Ok(self
            .utility_cohort
            .as_mut()
            .and_then(|state| state.observer.take()))
    }
    pub fn materialize_utility_cohort_trial(&mut self) -> Result<bool, AdaptiveManagerError> {
        if self.utility_cohort.is_none() {
            return Ok(false);
        }
        let receipt = self.shutdown_receipt()?;
        if self.phase != ManagerPhase::Finished || !receipt.permits_semantic_closure() {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        let state = self.utility_cohort.as_mut().expect("cohort");
        if state.materialized.is_some() || state.materialization_error.is_some() {
            return Ok(false);
        }
        let Some(bound) = state.bound.as_ref() else {
            return Ok(false);
        };
        let Some(outcome) = self
            .utility_route
            .as_ref()
            .and_then(|route| route.outcome_for_cohort())
        else {
            return Ok(false);
        };
        let materialized = self
            .driver
            .freeze_utility_witness_bodies(outcome)
            .and_then(|bodies| bound.materialize_semantics_with_witness_bodies(outcome, &bodies));
        match materialized {
            Ok(materialized) => {
                state.materialized = Some(materialized);
                Ok(true)
            }
            Err(error) => {
                state.materialization_error = Some(error);
                Ok(true)
            }
        }
    }
    pub(super) fn pump_cohort_cache_prelude(
        &mut self,
        actor: &PreparationActor,
    ) -> Result<(bool, bool), AdaptiveManagerError> {
        let Some(state) = self
            .utility_cohort
            .as_mut()
            .filter(|state| state.window.is_some())
        else {
            return Ok((false, true));
        };
        if state.cache_attached || state.cache_ack.is_some() {
            return Ok((false, true));
        }
        if self.phase != ManagerPhase::Running {
            return Ok((false, false));
        }
        if state.cache_invalidation.is_none() {
            state.cache_invalidation = Some(actor.invalidate_context_cache().map_err(|e| {
                AdaptiveManagerError::NnuecPreparation(nnuec::NnuecWorkError::Preparation(e))
            })?);
            return Ok((true, false));
        }
        let released = actor
            .try_context_cache_release(state.cache_invalidation.as_ref().expect("cache ticket"))
            .map_err(|e| {
                AdaptiveManagerError::NnuecPreparation(nnuec::NnuecWorkError::Preparation(e))
            })?;
        match released {
            Some(ack) => {
                state.cache_ack = Some(ack);
                Ok((true, true))
            }
            None => Ok((false, false)),
        }
    }
    pub(super) fn cohort_cache_request_pending(&self) -> bool {
        self.utility_cohort
            .as_ref()
            .is_some_and(|state| state.window.is_some() && !state.cache_attached)
    }
    // A pending/rejected prelude returns the original Request without a new
    // failure allocation; a rejected attachment also restores its exact ACK.
    #[allow(clippy::result_large_err)]
    pub(super) fn attach_cohort_cache_prelude(
        &mut self,
        request: Request,
    ) -> Result<Request, (AdaptiveManagerError, Request)> {
        let Some(state) = self
            .utility_cohort
            .as_mut()
            .filter(|state| state.window.is_some())
        else {
            return Ok(request);
        };
        if state.cache_attached {
            return Ok(request);
        }
        let Some(ack) = state.cache_ack.take() else {
            return Err((AdaptiveManagerError::AuxiliaryPending, request));
        };
        match request.with_utility_cache_prelude(ack) {
            Ok(request) => {
                state.cache_attached = true;
                Ok(request)
            }
            Err((error, request, ack)) => {
                state.cache_ack = Some(ack);
                Err((
                    AdaptiveManagerError::NnuecPreparation(nnuec::NnuecWorkError::Preparation(
                        error,
                    )),
                    request,
                ))
            }
        }
    }
    /// Called after original root preparation and real prelude retirement.
    /// Busy means that same owner must progress; it never causes a fake idle
    /// snapshot, a role result discard, a fresh worker or a credit refund.
    pub fn capture_utility_cohort_ready_snapshot(
        &mut self,
    ) -> Result<UtilityCohortReadySnapshot<Q>, AdaptiveManagerError> {
        let state = self
            .utility_cohort
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "cohort original profile absent",
            ))?;
        let bounds = (state.profile, state.prelude_cutoff, state.maximum_cleanup);
        self.capture_utility_source_snapshot(bounds.0, bounds.1, bounds.2)
    }
    // Same original sampler, with a distinct caller-owned declaration. It
    // records actual source offers; alternative intervention arms are absent.
    pub(super) fn capture_utility_source_snapshot(
        &mut self,
        profile: Digest,
        prelude_cutoff: Instant,
        maximum_cleanup: Instant,
    ) -> Result<UtilityCohortReadySnapshot<Q>, AdaptiveManagerError> {
        let now = self.check_running()?;
        if !self.root_started
            || self.registration.endpoints.execution_use != CpuExecutionUse::Collection
            || self.researcher_source_kind() != AdaptiveResearcherSourceKind::Native
            || self.nnuec.is_none()
            || self.registration.endpoints.load_v
            || self.exclusive_requested
            || self.utility_route.is_some()
        {
            return Err(AdaptiveManagerError::Invalid(
                "utility cohort requires original native NNUEC Collection prelude",
            ));
        }
        if !self.can_admit_work()
            || !self.roles.is_empty()
            || !self.accepted_roles.is_empty()
            || !self.acknowledgements.is_empty()
            || self.registry.occupancy() != 0
            || self.native.occupancy() != 0
            || self
                .own
                .iter()
                .any(|slot| slot.binding.is_some() || slot.joined || slot.panicked)
            || self.foreign.is_some()
            || self.profiled_busy()
            || !self.deferred.is_empty()
            || !self.native_frontier_events.is_empty()
            || self.nnuec_pending.is_some()
            || self.native.is_quarantined()
            || !self.nnuec_jobs.is_empty()
            || !self.retained_cpu_jobs.is_empty()
            || self.goal_prepared.is_some()
            || self.goal_continuation_pending
            || !self.failures.is_empty()
            || !self.driver.semantic_owner_available()
            || !self.driver.prepare_semantic_commit()?
        {
            return Err(AdaptiveManagerError::AuxiliaryPending);
        }
        let admission = self
            .collection_admission
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "cohort has no actual Collection source admission",
            ))?;
        self.driver.validate_collection_admission(admission)?;
        let root_budget =
            self.driver
                .rovent_root_budget()?
                .ok_or(AdaptiveManagerError::Invalid(
                    "cohort actual Frontier missing",
                ))?;
        let cleanup = self
            .registered_cleanup_deadline()
            .ok_or(AdaptiveManagerError::Invalid(
                "cohort original cleanup absent",
            ))?;
        let native_credit = self.native_ledger()?.registration_observation()?;
        let (common_nodes, common_work) =
            self.driver
                .utility_cohort_common_work()?
                .ok_or(AdaptiveManagerError::Invalid(
                    "cohort actual common quota absent",
                ))?;
        if !common_work.native_usage_known
            || common_work.reserved_native_nodes != 0
            || common_work.outstanding_native_executions != 0
        {
            return Err(AdaptiveManagerError::AuxiliaryPending);
        }
        if native_credit.reserved_native_nodes != 0
            || native_credit.outstanding_executions != 0
            || native_credit.active_jobs != 0
            || native_credit.quarantined_jobs != 0
        {
            return Err(AdaptiveManagerError::AuxiliaryPending);
        }
        let mut offers = Vec::new();
        for frame in self.machine.frames().filter(|frame| {
            frame.phase == FramePhase::Ready && self.machine.waiting_on(frame.id).is_none()
        }) {
            if offers.len() == 64 {
                return Err(AdaptiveManagerError::Capacity("cohort actual ready offers"));
            }
            let recipient = match frame.input.stage.role() {
                Some(Role::Proposer) => UtilityRouteRecipient::Proposer,
                Some(Role::Critic) => UtilityRouteRecipient::Critic,
                Some(Role::Validator) => continue,
                None => UtilityRouteRecipient::Researcher,
            };
            if recipient == UtilityRouteRecipient::Researcher
                && native_credit.spent_native_nodes >= native_credit.native_node_limit
            {
                continue;
            }
            if recipient != UtilityRouteRecipient::Researcher
                && !self.role_call_admission_available()
            {
                continue;
            }
            // ready_work rechecks the original exact frame, original clock,
            // deadline and private dependency state without changing credit.
            let work = self
                .machine
                .ready_work(frame.id, self.clock.domain(), now)?;
            let semantic = self.driver.utility_cohort_offer_semantics(&work)?;
            if !semantic.matches_work(&work) {
                return Err(AdaptiveManagerError::Invalid(
                    "cohort semantic offer differs from actual Ready Work",
                ));
            }
            offers.push(UtilityCohortReadyOffer {
                frame: work.frame,
                recipient,
                stage: work.input.stage,
                first: work.original_first_move,
                question: work.input.question_digest,
                view: work.input.view_seal,
                full_view: Digest(work.input.view.digest()),
                public_context: work.input.public_context,
                budget: work.budget,
                deadline: work.original_deadline,
                query: Arc::downgrade(&work.input.question),
                selected: Arc::downgrade(&work.input.view),
                semantic,
            });
        }
        if offers.is_empty() {
            return Err(AdaptiveManagerError::AuxiliaryPending);
        }
        for id in &self.ready_ids {
            self.ready
                .set_dependencies_ready(id, offers.iter().any(|offer| offer.frame == *id))
                .map_err(|_| AdaptiveManagerError::Invalid("cohort original ready dependency"))?;
        }
        let mut selected_offers = [None; 3];
        for recipient in [
            UtilityRouteRecipient::Proposer,
            UtilityRouteRecipient::Critic,
            UtilityRouteRecipient::Researcher,
        ] {
            selected_offers[recipient.output_index() - 4] = self
                .ready
                .peek_matching(Duration::from_nanos(now.0), None, |job| {
                    offers
                        .iter()
                        .any(|offer| offer.frame == job.job && offer.recipient == recipient)
                })
                .map_err(|_| AdaptiveManagerError::Invalid("cohort actual fair offer observation"))?
                .and_then(|job| offers.iter().position(|offer| offer.frame == job.job));
        }
        Ok(UtilityCohortReadySnapshot {
            admission: admission.clone(),
            authority: self.registration.authority,
            root_budget,
            clock: self.clock.clone(),
            hard_deadline: self.registration.hard_deadline,
            cleanup_deadline: cleanup,
            captured_at: Instant::now(),
            profile,
            prelude_cutoff,
            maximum_cleanup,
            counters: self.counters,
            native_credit,
            native_ledger: self.native_ledger()?.identity(),
            common_nodes,
            common_work,
            role_credit: self.role_call_budget_usage(),
            offers,
            selected_offers,
            native_threads: self.native.ready_cpu_thread_observation(),
            cpu_threads: self
                .own
                .iter()
                .map(|slot| {
                    slot.worker
                        .as_ref()
                        .and_then(|worker| worker.ready_readback().cloned())
                        .or_else(|| slot.ready_readback.clone())
                })
                .collect(),
        })
    }
}
