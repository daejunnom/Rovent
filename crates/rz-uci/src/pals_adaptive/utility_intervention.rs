//! Explicit frozen-proposal experiments. Normal Ready offers remain original
//! source observations; only Search and the coordinator issue an actual arm.
use super::utility_cohort::{
    UTILITY_COHORT_COST, UTILITY_COHORT_COST_SCOPE, UtilityCohortCostWire, UtilityCohortSourcePins,
    UtilityTrialWindow, counter_value, hex, invalid, value_digest,
};
use super::utility_cohort_historical::{bounded_json, freeze_ready_prelude};
use super::{AdaptiveManagerError, UtilityCohortReadySnapshot, UtilityCohortRootClosureProof};
use crate::rovent_bootstrap::nnuec_source::LoadedRoventNnuecSource;
use rz_contracts::{Digest, pals::SearchAuthority};
use rz_experiments::rovent_utility_cohort::{
    ROVENT_UTILITY_INTERVENTION_TRIAL_DOMAIN, RoventUtilityCohortRecipient,
    RoventUtilityCohortUsage, RoventUtilityInterventionCohortV2,
};
use rz_runtime::contracts::ContractClock as _;
use rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture;
use rz_search::pals::coordinator::CoordinatorWork;
use rz_search::pals::engine::adaptive_port::utility_route::{
    UtilityInterventionTarget, UtilityRouteAction, UtilityRouteActionId, UtilityRouteRecipient,
};
use rz_search::pals::review_context::ReviewView;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The unmodified actual Ready sampler, used only as an intervention's source
/// checkpoint. It makes no claim that its requested alternative arms are Ready.
pub struct UtilityInterventionSourceSnapshot<Q> {
    pub(super) actual: UtilityCohortReadySnapshot<Q>,
}
impl<Q> UtilityInterventionSourceSnapshot<Q> {
    pub fn has_target(&self, target: u16) -> bool {
        self.selected_source_index(target).is_some()
    }
    fn selected_source_index(&self, target: u16) -> Option<usize> {
        [
            UtilityRouteRecipient::Proposer,
            UtilityRouteRecipient::Researcher,
            UtilityRouteRecipient::Critic,
        ]
        .into_iter()
        .filter_map(|r| self.actual.selected_offer_index(r))
        .find(|index| self.actual.offers()[*index].intervention_target_matches(target))
    }
    pub fn admission(&self) -> &crate::pals_adaptive::RoventCollectionRootAdmission {
        self.actual.admission()
    }
    pub fn actual_root(&self) -> &Arc<crate::rovent_root_admission::RoventRootAdmission> {
        self.actual.actual_root()
    }
    pub fn authority(&self) -> SearchAuthority {
        self.actual.authority()
    }
    pub fn root_budget(&self) -> Digest {
        self.actual.root_budget()
    }
    pub fn clock(&self) -> &rz_runtime::contracts::ContractSystemClock {
        self.actual.clock()
    }
    pub fn captured_at(&self) -> Instant {
        self.actual.captured_at()
    }
    pub fn prelude_cutoff(&self) -> Instant {
        self.actual.prelude_cutoff()
    }
    pub fn hard_deadline(&self) -> Instant {
        self.actual.hard_deadline()
    }
    pub fn cleanup_deadline(&self) -> Instant {
        self.actual.cleanup_deadline()
    }
    pub fn declared_cleanup_ceiling(&self) -> Instant {
        self.actual.declared_cleanup_ceiling()
    }
    pub fn counters(&self) -> super::AdaptiveManagerCounters {
        self.actual.counters()
    }
}
struct Shared {
    profile: RoventUtilityInterventionCohortV2,
    profile_digest: Digest,
    cohort: Digest,
    pins: UtilityCohortSourcePins,
    planned: Vec<UtilityRouteRecipient>,
    registration_root: Value,
    state: Mutex<State>,
}
struct State {
    next: usize,
    active: Option<u64>,
    roots: Vec<Arc<crate::rovent_root_admission::RoventRootAdmission>>,
    neutral: Option<(Digest, Value)>,
    supported: Option<Vec<UtilityRouteRecipient>>,
    control: Option<(
        rz_search::pals::coordinator::FrameBudget,
        std::collections::BTreeSet<[u8; 32]>,
    )>,
    durable: Vec<Arc<UtilityInterventionDurableAck>>,
}
/// One move-only sequencing owner. A historical profile, target JSON, raw hash
/// or /1 action cannot reconstruct this current registration.
pub struct UtilityInterventionCohortOwner {
    shared: Arc<Shared>,
}
impl UtilityInterventionCohortOwner {
    pub fn register<Q>(
        profile: RoventUtilityInterventionCohortV2,
        snapshot: &UtilityInterventionSourceSnapshot<Q>,
        loaded: Arc<LoadedRoventNnuecSource>,
    ) -> Result<Self, AdaptiveManagerError> {
        profile
            .validate()
            .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
        let digest = profile
            .digest()
            .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
        let pins = UtilityCohortSourcePins::from_actual(&snapshot.actual, &loaded)?;
        if snapshot.actual.cohort_profile_digest() != digest
            || !pins_match(&pins, &profile)
            || snapshot.captured_at() >= snapshot.prelude_cutoff()
            || snapshot
                .admission()
                .native_origin()
                .original_input_lock()
                .manifest
                .engine
                .mode
                != profile.input.mode.execution_mode()
        {
            return Err(invalid("intervention profile/original source differs"));
        }
        let planned: Vec<_> = profile
            .input
            .requested_recipients
            .iter()
            .copied()
            .map(recipient)
            .collect();
        let cohort = value_digest(
            "rz-rovent-utility-intervention-owner/2",
            &json!({
                "profile":hex(digest),"original_root":hex(snapshot.actual_root().digest()),"source":pins
            }),
        )?;
        Ok(Self {
            shared: Arc::new(Shared {
                profile,
                profile_digest: digest,
                cohort,
                pins,
                planned,
                registration_root: snapshot.admission().raw_receipt().clone(),
                state: Mutex::new(State {
                    next: 0,
                    active: None,
                    roots: Vec::new(),
                    neutral: None,
                    supported: None,
                    control: None,
                    durable: Vec::new(),
                }),
            }),
        })
    }
    pub fn cohort_id(&self) -> Digest {
        self.shared.cohort
    }
    pub fn profile(&self) -> &RoventUtilityInterventionCohortV2 {
        &self.shared.profile
    }
    pub fn profile_digest(&self) -> Digest {
        self.shared.profile_digest
    }
    pub fn source_pins(&self) -> &UtilityCohortSourcePins {
        &self.shared.pins
    }
    pub fn planned_recipients(&self) -> &[UtilityRouteRecipient] {
        &self.shared.planned
    }
    pub fn registration_root(&self) -> &Value {
        &self.shared.registration_root
    }
    pub fn neutral_target(&self) -> Result<Option<Value>, AdaptiveManagerError> {
        Ok(self
            .shared
            .state
            .lock()
            .map_err(|_| invalid("intervention owner poisoned"))?
            .neutral
            .as_ref()
            .map(|(_, v)| v.clone()))
    }
    #[allow(clippy::result_large_err)]
    pub fn admit_trial<Q>(
        &mut self,
        snapshot: UtilityInterventionSourceSnapshot<Q>,
        requested: UtilityRouteRecipient,
        loaded: Arc<LoadedRoventNnuecSource>,
    ) -> Result<UtilityInterventionTrialAdmission<Q>, UtilityInterventionTrialAdmissionRejection<Q>>
    {
        let checked = (|| {
            let pins = UtilityCohortSourcePins::from_actual(&snapshot.actual, &loaded)?;
            let mut state = self
                .shared
                .state
                .lock()
                .map_err(|_| invalid("intervention owner poisoned"))?;
            if state.active.is_some()
                || state.next >= self.shared.planned.len()
                || state.durable.len() != state.next
                || self.shared.planned[state.next] != requested
                || pins != self.shared.pins
                || snapshot.actual.cohort_profile_digest() != self.shared.profile_digest
                || snapshot.captured_at() >= snapshot.prelude_cutoff()
                || state.roots.iter().any(|r| {
                    Arc::ptr_eq(r, snapshot.actual_root()) || r.authority() == snapshot.authority()
                })
            {
                return Err(invalid(
                    "intervention trial is repeated, foreign, busy or out of order",
                ));
            }
            // This selects an ORIGINAL source fairly. It is not an offer of
            // the requested alternative; that is minted only after preparation.
            let selected = snapshot
                .selected_source_index(self.profile().target_record)
                .ok_or_else(|| invalid("intervention has no original fair source"))?;
            let sequence = state.next as u64 + 1;
            let preparation = freeze_ready_prelude(
                &snapshot.actual,
                super::utility_cohort::actual_offer_group(&snapshot.actual)?,
            )?;
            let root = snapshot.admission().raw_receipt().clone();
            bounded_json(&root)?;
            state.active = Some(sequence);
            state.roots.push(Arc::clone(snapshot.actual_root()));
            Ok((sequence, selected, preparation, root))
        })();
        match checked {
            Ok((sequence, selected, preparation, root)) => Ok(UtilityInterventionTrialAdmission {
                shared: Arc::clone(&self.shared),
                snapshot,
                loaded,
                sequence,
                selected,
                preparation,
                root,
                window: None,
                neutral: None,
                arm_query: None,
                action: None,
                cost_epoch: None,
                unavailable: None,
                materialized: None,
                materialization_error: None,
            }),
            Err(error) => Err(UtilityInterventionTrialAdmissionRejection {
                snapshot,
                loaded,
                error,
            }),
        }
    }
}
pub struct UtilityInterventionTrialAdmissionRejection<Q> {
    pub snapshot: UtilityInterventionSourceSnapshot<Q>,
    pub loaded: Arc<LoadedRoventNnuecSource>,
    pub error: AdaptiveManagerError,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UtilityInterventionUnknown {
    InsufficientEligibleActions,
    UnsupportedArm,
}
impl UtilityInterventionUnknown {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InsufficientEligibleActions => "InsufficientEligibleActions",
            Self::UnsupportedArm => "UnsupportedInterventionArm",
        }
    }
}
pub struct UtilityInterventionTrialAdmission<Q> {
    shared: Arc<Shared>,
    pub(super) snapshot: UtilityInterventionSourceSnapshot<Q>,
    loaded: Arc<LoadedRoventNnuecSource>,
    sequence: u64,
    selected: usize,
    preparation: Value,
    root: Value,
    window: Option<UtilityTrialWindow>,
    neutral: Option<(Digest, Value, Arc<NnuecUtilityInputCapture>)>,
    arm_query: Option<Value>,
    action: Option<UtilityRouteActionId>,
    cost_epoch: Option<Digest>,
    unavailable: Option<UtilityInterventionUnknown>,
    pub(super) materialized:
        Option<super::utility_intervention_historical::UtilityInterventionMaterializedSource>,
    pub(super) materialization_error: Option<AdaptiveManagerError>,
}
impl<Q> UtilityInterventionTrialAdmission<Q> {
    pub fn cohort_id(&self) -> Digest {
        self.shared.cohort
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn recipient(&self) -> UtilityRouteRecipient {
        self.shared.planned[self.sequence as usize - 1]
    }
    pub fn profile(&self) -> &RoventUtilityInterventionCohortV2 {
        &self.shared.profile
    }
    pub fn profile_digest(&self) -> Digest {
        self.shared.profile_digest
    }
    pub fn source_pins(&self) -> &UtilityCohortSourcePins {
        &self.shared.pins
    }
    pub fn original_root(&self) -> &Value {
        &self.root
    }
    pub fn preparation(&self) -> &Value {
        &self.preparation
    }
    pub fn window(&self) -> Option<UtilityTrialWindow> {
        self.window
    }
    pub(super) fn source_frame(&self) -> rz_search::pals::coordinator::FrameId {
        self.snapshot.actual.offers()[self.selected].frame()
    }
    pub(super) fn matches_source(&self, work: &CoordinatorWork<ReviewView, Q>) -> bool {
        self.snapshot.actual.offers()[self.selected].matches_ready_work(work)
    }
    pub(super) fn arm_window(
        &mut self,
        started: Instant,
    ) -> Result<UtilityTrialWindow, AdaptiveManagerError> {
        if self.window.is_some() || started >= self.snapshot.prelude_cutoff() {
            return Err(invalid("intervention already started or prelude expired"));
        }
        let window = super::utility_cohort::checked_trial_window(
            started,
            self.profile().input.action_ns,
            self.profile().input.cleanup_ns,
            self.snapshot.hard_deadline(),
            self.snapshot.declared_cleanup_ceiling(),
        )?;
        self.window = Some(window);
        Ok(window)
    }
    pub(super) fn bind_target(
        &mut self,
        target: &UtilityInterventionTarget<Q>,
    ) -> Result<bool, AdaptiveManagerError> {
        if !self.matches_source(target.source_work())
            || self.neutral.is_some()
            || self.profile().target_record != target.target_record()
        {
            return Err(invalid("intervention neutral source allocation changed"));
        }
        let digest = target.neutral_target_digest();
        let wire = super::utility_intervention_historical::neutral_target(target)?;
        let mut state = self
            .shared
            .state
            .lock()
            .map_err(|_| invalid("intervention owner poisoned"))?;
        if let Some((prior, old)) = &state.neutral {
            // Actual source IDs/provenance stay in each trial. The pure Search
            // semantic seal, original bytes and model define common input.
            compare_neutral_targets(*prior, old, digest, &wire)?;
        } else {
            state.neutral = Some((digest, wire.clone()));
        }
        let supported: Vec<_> = target.supported_recipients().collect();
        let control = (
            target.budget(),
            target
                .baseline_witnesses()
                .map(|d| d.0)
                .collect::<std::collections::BTreeSet<_>>(),
        );
        if let Some((budget, baseline)) = &state.control {
            if !same_control(*budget, control.0, baseline, &control.1) {
                return Err(invalid(
                    "intervention original source budget/baseline differs across roots",
                ));
            }
        } else {
            state.control = Some(control);
        }
        if let Some(prior) = &state.supported {
            if *prior != supported {
                return Err(invalid(
                    "intervention actual supported arms differ across independent roots",
                ));
            }
        } else {
            state.supported = Some(supported.clone());
        }
        self.neutral = Some((digest, wire, Arc::clone(target.input_capture())));
        if self
            .shared
            .planned
            .iter()
            .filter(|r| supported.contains(r))
            .count()
            < 2
        {
            self.unavailable = Some(UtilityInterventionUnknown::InsufficientEligibleActions);
            return Ok(false);
        }
        if !supported.contains(&self.recipient()) {
            self.unavailable = Some(UtilityInterventionUnknown::UnsupportedArm);
            return Ok(false);
        }
        Ok(true)
    }
    pub(super) fn bind_action(
        &mut self,
        action: &UtilityRouteAction,
        arm_query: Value,
        cost_epoch: Digest,
    ) -> Result<(), AdaptiveManagerError> {
        let (digest, _, input) = self
            .neutral
            .as_ref()
            .ok_or_else(|| invalid("intervention neutral cap absent"))?;
        if self.action.is_some()
            || action.id().authority() != self.snapshot.authority()
            || action.id().recipient() != self.recipient()
            || !Arc::ptr_eq(action.input(), input)
            || action.intervention_neutral_target_digest() != Some(*digest)
        {
            return Err(invalid(
                "intervention actual action differs from original target",
            ));
        }
        self.arm_query = Some(arm_query);
        self.action = Some(action.id().clone());
        self.cost_epoch = Some(cost_epoch);
        Ok(())
    }
    pub(super) fn materialize(
        &mut self,
        outcome:&rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteSemanticOutcome,
        bodies:&rz_search::pals::engine::adaptive_port::utility_route::UtilityConditionalWitnessBodies,
        cost_epoch: Digest,
    ) -> Result<(), AdaptiveManagerError> {
        if self.materialized.is_some() || self.materialization_error.is_some() {
            return Ok(());
        }
        let value=super::utility_intervention_historical::UtilityInterventionMaterializedSource::from_actual(self,outcome,bodies,cost_epoch);
        match value {
            Ok(v) => self.materialized = Some(v),
            Err(e) => {
                self.materialization_error = Some(e.clone());
                return Err(e);
            }
        }
        Ok(())
    }
    pub(super) fn neutral(&self) -> Option<&(Digest, Value, Arc<NnuecUtilityInputCapture>)> {
        self.neutral.as_ref()
    }
    pub(super) fn action(&self) -> Option<&UtilityRouteActionId> {
        self.action.as_ref()
    }
    pub(super) fn arm_query(&self) -> Option<&Value> {
        self.arm_query.as_ref()
    }
    pub(super) fn unavailable_reason(&self) -> Option<&'static str> {
        self.unavailable.map(UtilityInterventionUnknown::as_str)
    }
    pub(super) fn replace_arm_metadata(
        &mut self,
        value: Value,
    ) -> Result<(), AdaptiveManagerError> {
        let old = self
            .arm_query
            .as_ref()
            .ok_or_else(|| invalid("intervention original arm query missing"))?;
        if old["arm_query_sha256"] != value["arm_query_sha256"]
            || old["frame_sequence"] != value["frame_sequence"]
            || old["neutral_target_sha256"] != value["neutral_target_sha256"]
            || old["view"] != value["view"]
            || value["prepared_input"].is_null()
        {
            return Err(invalid(
                "intervention arm preparation replaced another original question",
            ));
        }
        self.arm_query = Some(value);
        Ok(())
    }
}

pub struct UtilityInterventionTrialRetirementRejection<Q> {
    pub trial: UtilityInterventionTrialAdmission<Q>,
    pub proof: UtilityCohortRootClosureProof,
    pub error: AdaptiveManagerError,
}
/// Issued after original manager, actor/cache, worker, Store pin-zero and
/// bounded materialization. It is unrelated to the legacy /1 retirement type.
pub struct UtilityInterventionTrialPhysicalRetirement {
    shared: Arc<Shared>,
    sequence: u64,
    recipient: UtilityRouteRecipient,
    authority: SearchAuthority,
    source_budget: rz_search::pals::coordinator::FrameBudget,
    root: Arc<crate::rovent_root_admission::RoventRootAdmission>,
    proof: UtilityCohortRootClosureProof,
    neutral: Option<(Digest, Value)>,
    arm_query: Option<Value>,
    materialized:
        Option<super::utility_intervention_historical::UtilityInterventionMaterializedSource>,
    original_root: Value,
    preparation: Value,
    cost: UtilityCohortCostWire,
    physical: Value,
    bytes: Vec<u8>,
    digest: Digest,
    unknown: Option<&'static str>,
}
impl<Q> UtilityInterventionTrialAdmission<Q> {
    #[allow(clippy::result_large_err)]
    pub(crate) fn finish_physical_retirement<M: rz_search::pals::engine::RoleModel>(
        mut self,
        proof: UtilityCohortRootClosureProof,
        original: &rz_search::pals::engine::preparing_root::RetiredResearcherOwner<M>,
    ) -> Result<
        UtilityInterventionTrialPhysicalRetirement,
        UtilityInterventionTrialRetirementRejection<Q>,
    > {
        let checked = (|| {
            let snapshot = &self.snapshot.actual;
            let receipt = proof.receipt();
            let source = snapshot.offers()[self.selected]
                .semantic_source()
                .ok_or_else(|| invalid("intervention actual source missing"))?;
            if proof.cohort_id() != self.cohort_id()
                || proof.trial_sequence() != self.sequence
                || !receipt.permits_semantic_closure()
                || receipt.authority != snapshot.authority()
                || proof.final_counters() != receipt.counters
                || original.owner.last_authority() != Some(snapshot.authority())
                || !source.matches_store(original.owner.stores())
                || original
                    .owner
                    .stores()
                    .review_pin_count()
                    .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?
                    != 0
                || !original
                    .engine_release
                    .matches_source(original.owner.source())
                || !original
                    .engine_release
                    .matches_cache_release(&original.cache_release)
                || !Arc::ptr_eq(original.owner.source().model(), self.loaded.model())
                || proof.native_credit_baseline().1 != 0
                || proof.native_credit_final() != (receipt.native_cpu_spent, 0)
                || proof.native_credit_baseline().0 > proof.native_credit_final().0
                || proof.root_finished() < proof.started()
                || proof.release_finished() < proof.root_finished()
                || proof.baseline_counters() != snapshot.counters()
                || self.window.is_some_and(|w| w.started() != proof.started())
            {
                return Err(invalid(
                    "intervention original physical root/actor/Store/credit not closed",
                ));
            }
            if let Some((digest, _, input)) = &self.neutral {
                if !input.matches_original_store(original.owner.stores())
                    || !input.matches_original_model(self.loaded.model())
                    || !input.matches_context_cache_release(&original.cache_release)
                    || proof.outcome().is_some_and(|o| {
                        Some(o.id()) != self.action.as_ref()
                            || !Arc::ptr_eq(o.input(), input)
                            || o.frozen_context_digest() != *digest
                    })
                {
                    return Err(invalid(
                        "intervention original neutral input/actual action lineage differs",
                    ));
                }
            } else if proof.outcome().is_some() {
                return Err(invalid(
                    "intervention semantic outcome without actual neutral target",
                ));
            }
            if proof.outcome().is_some_and(|o| {
                o.physical_input_captures().any(|i| {
                    !i.matches_original_store(original.owner.stores())
                        || !i.matches_original_model(self.loaded.model())
                        || !i.matches_context_cache_release(&original.cache_release)
                })
            }) || proof.unresolved().is_some_and(|o| {
                o.physical_input_captures().any(|i| {
                    !i.matches_original_store(original.owner.stores())
                        || !i.matches_original_model(self.loaded.model())
                        || !i.matches_context_cache_release(&original.cache_release)
                })
            }) || self.materialized.as_ref().is_some_and(|m| {
                m.created_at() < proof.root_finished() || m.finished_at() > proof.release_finished()
            }) {
                return Err(invalid(
                    "intervention original arm capture or materialization precedes physical source",
                ));
            }
            let finished = Instant::now();
            let origin = snapshot.clock().origin();
            let tick = |time: Instant| {
                time.checked_duration_since(origin)
                    .and_then(|d| u64::try_from(d.as_nanos()).ok())
                    .ok_or_else(|| invalid("intervention clock overflow"))
            };
            let window = self.window;
            let cost = UtilityCohortCostWire {
                domain: UTILITY_COHORT_COST.into(),
                scope: UTILITY_COHORT_COST_SCOPE.into(),
                objective: "exclusive_wall_nanoseconds".into(),
                cost_class_sha256: self.profile().input.cost_class_sha256.clone(),
                cost_epoch_sha256: self.cost_epoch.map(hex),
                resource_plan_sha256: self.source_pins().resource_plan_sha256.clone(),
                resource_epoch: self.profile().input.resource_epoch,
                clock_epoch: snapshot.clock().domain().0.0,
                started_tick: tick(proof.started())?,
                root_fence_tick: tick(proof.root_finished())?,
                actor_release_tick: tick(proof.release_finished())?,
                physical_pin_zero_tick: tick(finished)?,
                whole_wall_ns: u64::try_from(finished.duration_since(proof.started()).as_nanos())
                    .map_err(|_| invalid("intervention cost overflow"))?,
                original_deadline_tick: snapshot.offers()[self.selected].original_deadline().at.0,
                cohort_action_cutoff_tick: tick(
                    window.map_or(snapshot.hard_deadline(), |w| w.action_cutoff()),
                )?,
                cohort_cleanup_cutoff_tick: tick(
                    window.map_or(snapshot.cleanup_deadline(), |w| w.cleanup_cutoff()),
                )?,
                baseline_counters: counter_value(proof.baseline_counters()),
                final_counters: counter_value(proof.final_counters()),
                native_credit_baseline: proof.native_credit_baseline(),
                native_credit_final: proof.native_credit_final(),
                cpu_thread_time_ns: None,
                energy: None,
                complete_scope: proof.outcome().is_some() && self.materialized.is_some(),
            };
            let unknown = if self.profile().input.usage == RoventUtilityCohortUsage::ValidationOnly
            {
                Some("validation_only")
            } else if let Some(r) = proof.reason() {
                Some(r)
            } else if self.materialization_error.is_some() {
                Some("historical_materialization_failed")
            } else if proof.outcome().is_none() {
                Some("missing_original_semantic_outcome")
            } else if self
                .materialized
                .as_ref()
                .is_none_or(|m| !m.has_original_witness_basis())
            {
                Some("missing_original_conditional_witness_basis")
            } else if !snapshot.admission().is_first_product_root() {
                Some("native_product_reused_before_trial")
            } else if !super::utility_cohort::resource_observations_known(snapshot) {
                Some("unknown_original_resource_observation")
            } else if self
                .neutral
                .as_ref()
                .is_none_or(|(_, _, i)| !i.cache_prelude().is_some_and(|p| p.matches_input(i)))
            {
                Some("missing_actual_cache_prelude")
            } else if window.is_none_or(|w| finished > w.cleanup_cutoff()) {
                Some("intervention_cleanup_cutoff")
            } else if proof.final_counters().nn_input_count_unknown
                != proof.baseline_counters().nn_input_count_unknown
                || proof.final_counters().accepted_nn_input_count_unknown
                    != proof.baseline_counters().accepted_nn_input_count_unknown
                || proof.final_counters().consumed_nn_input_count_unknown
                    != proof.baseline_counters().consumed_nn_input_count_unknown
                || proof.final_counters().rejected_events
                    != proof.baseline_counters().rejected_events
            {
                Some("unknown_or_rejected_actual_work")
            } else {
                None
            };
            let physical = super::utility_cohort::physical_closure_value(
                snapshot,
                &proof,
                original,
                self.neutral.as_ref().map(|(_, _, i)| i.as_ref()),
            )?;
            Ok((cost, physical, unknown))
        })();
        let (cost, physical, unknown) = match checked {
            Ok(v) => v,
            Err(error) => {
                return Err(UtilityInterventionTrialRetirementRejection {
                    trial: self,
                    proof,
                    error,
                });
            }
        };
        let mut retired = UtilityInterventionTrialPhysicalRetirement {
            shared: Arc::clone(&self.shared),
            sequence: self.sequence,
            recipient: self.recipient(),
            authority: self.snapshot.authority(),
            source_budget: self.snapshot.actual.offers()[self.selected].budget(),
            root: Arc::clone(self.snapshot.actual_root()),
            original_root: self.root.clone(),
            preparation: self.preparation.clone(),
            proof,
            neutral: self.neutral.take().map(|(d, v, _)| (d, v)),
            arm_query: self.arm_query.take(),
            materialized: self.materialized.take(),
            cost,
            physical,
            bytes: Vec::new(),
            digest: Digest([0; 32]),
            unknown,
        };
        match retired.checkpoint_value().and_then(|v| bounded_json(&v)) {
            Ok(bytes) => {
                retired.digest = framed(ROVENT_UTILITY_INTERVENTION_TRIAL_DOMAIN, &bytes);
                retired.bytes = bytes;
                Ok(retired)
            }
            Err(error) => {
                self.materialized = retired.materialized;
                Err(UtilityInterventionTrialRetirementRejection {
                    trial: self,
                    proof: retired.proof,
                    error,
                })
            }
        }
    }
}
impl UtilityInterventionTrialPhysicalRetirement {
    pub fn cohort_id(&self) -> Digest {
        self.shared.cohort
    }
    pub fn trial_sequence(&self) -> u64 {
        self.sequence
    }
    pub fn recipient(&self) -> UtilityRouteRecipient {
        self.recipient
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn profile(&self) -> &RoventUtilityInterventionCohortV2 {
        &self.shared.profile
    }
    pub fn source_pins(&self) -> &UtilityCohortSourcePins {
        &self.shared.pins
    }
    pub fn neutral_target(&self) -> Option<&Value> {
        self.neutral.as_ref().map(|(_, v)| v)
    }
    pub fn neutral_target_digest(&self) -> Option<Digest> {
        self.neutral.as_ref().map(|(d, _)| *d)
    }
    pub fn materialized(
        &self,
    ) -> Option<&super::utility_intervention_historical::UtilityInterventionMaterializedSource>
    {
        self.materialized.as_ref()
    }
    pub fn physical_closure(&self) -> &Value {
        &self.physical
    }
    pub fn cost(&self) -> &UtilityCohortCostWire {
        &self.cost
    }
    pub fn semantic(
        &self,
    ) -> Option<&rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteSemanticOutcome>
    {
        self.proof.outcome()
    }
    pub fn comparable(&self) -> bool {
        self.unknown.is_none()
    }
    pub fn unknown_reason(&self) -> Option<&'static str> {
        self.unknown
    }
    pub fn original_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn sha256(&self) -> Digest {
        self.digest
    }
    pub fn support_mask(&self) -> [bool; 7] {
        let mut v = [false; 7];
        if let Ok(state) = self.shared.state.lock() {
            if let Some(actual) = &state.supported {
                for r in actual {
                    if self.shared.planned.contains(r) {
                        v[r.output_index()] = true;
                    }
                }
            }
        }
        v
    }
    pub fn matches_root_admission(
        &self,
        root: &Arc<crate::rovent_root_admission::RoventRootAdmission>,
    ) -> bool {
        Arc::ptr_eq(root, &self.root)
    }
    fn checkpoint_value(&self) -> Result<Value, AdaptiveManagerError> {
        let state = self
            .shared
            .state
            .lock()
            .map_err(|_| invalid("intervention owner poisoned"))?;
        let previous = if self.sequence == 1 {
            None
        } else {
            let a = state
                .durable
                .get(self.sequence as usize - 2)
                .ok_or_else(|| invalid("intervention preceding original ACK missing"))?;
            Some(
                json!({"trial_sequence":a.sequence,"trial_sha256":hex(a.trial),"file_sha256":hex(a.file_sha256)}),
            )
        };
        drop(state);
        let unbound=self.proof.historical_input().or_else(||self.proof.unresolved().map(|u|u.input())).map(|i|{
            let original=std::str::from_utf8(i.input_bytes()).map_err(|_|invalid("intervention unresolved input UTF8"))?;
            Ok::<Value,AdaptiveManagerError>(json!({"reference_json":original,"input_sha256":i.input_sha256(),"target_sha256":i.target_sha256(),"submission_target_sha256":i.submission_target_sha256(),"context_sha256":i.context_sha256(),"target_record_ids":i.target_record_ids(),"original_view_sha256":hex(Digest(i.provenance().view_seal)),"original_full_view_sha256":hex(Digest(i.provenance().full_view_digest)),"materialized_inside_complete_scope":false}))
        }).transpose()?;
        Ok(
            json!({"schema":ROVENT_UTILITY_INTERVENTION_TRIAL_DOMAIN,"cohort_id":hex(self.cohort_id()),"trial_sequence":self.sequence,"recipient":self.recipient.as_str(),"profile":self.shared.profile,"profile_sha256":hex(self.shared.profile_digest),"source":self.shared.pins,
            "neutral_target":self.neutral_target(),"neutral_target_sha256":self.neutral_target_digest().map(hex),"arm_query":self.arm_query,
            "materialized_sha256":self.materialized.as_ref().map(|m|hex(m.sha256())),"materialized_json":self.materialized.as_ref().map(|m|std::str::from_utf8(m.original_bytes()).expect("JSON writer UTF8")),
            "historical_unbound_input":unbound,"unresolved_source":self.proof.unresolved().map(|u|json!({"question_sha256":hex(u.id().question()),"frame_sequence":u.id().frame().sequence,"recipient":u.id().recipient().as_str(),"root_budget_sha256":hex(u.root_budget()),"frozen_context_sha256":hex(u.frozen_context_digest()),"reason":u.reason().as_str(),"actual_pc_consumptions":u.pc_consumption().len(),"actual_dependencies":u.dependencies().len(),"actual_r_pc_consumptions":u.r_pc_consumption().len(),"new_witnesses":u.witnesses().iter().map(|w|json!({"observation":w.observation().0,"identity_sha256":hex(w.identity()),"raw_evidence_sha256":hex(w.raw_evidence())})).collect::<Vec<_>>() })),
            "cost":self.cost,"original_root":self.original_root,"preparation":self.preparation,"physical_closure":self.physical,"previous_durable_ack":previous,"resolved":self.proof.outcome().is_some_and(|o|o.semantic_consumption_complete())&&self.materialized.is_some(),"comparable":self.comparable(),"unknown_reason":self.unknown,"support_mask":self.support_mask(),"target_known_mask":[false,false,false,false,false,false,false]}),
        )
    }
}
/// An original synced and bounded-read-back checkpoint FD. JSON is never
/// accepted in place of this live publication acknowledgement.
pub struct UtilityInterventionDurableAck {
    cohort: Digest,
    sequence: u64,
    trial: Digest,
    file_sha256: Digest,
    started: Instant,
    finished: Instant,
    file: std::fs::File,
}
pub struct UtilityInterventionPublicationRejection {
    pub file: std::fs::File,
    pub error: AdaptiveManagerError,
}
impl UtilityInterventionDurableAck {
    #[allow(clippy::result_large_err)]
    pub fn acknowledge_trial_publication(
        file: std::fs::File,
        original: &UtilityInterventionTrialPhysicalRetirement,
    ) -> Result<Self, UtilityInterventionPublicationRejection> {
        Self::acknowledge_trial_publication_inner(file, original, None)
    }
    /// Keep the caller's one publication window through the original FD's
    /// readback and sync. A synchronous OS operation is not preempted here;
    /// its late return is rejected with the same owned file retained.
    #[allow(clippy::result_large_err)]
    pub fn acknowledge_trial_publication_with_control(
        file: std::fs::File,
        original: &UtilityInterventionTrialPhysicalRetirement,
        until: Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<Self, UtilityInterventionPublicationRejection> {
        Self::acknowledge_trial_publication_inner(file, original, Some((until, cancel)))
    }
    #[allow(clippy::result_large_err)]
    fn acknowledge_trial_publication_inner(
        mut file: std::fs::File,
        original: &UtilityInterventionTrialPhysicalRetirement,
        control: Option<(Instant, &std::sync::atomic::AtomicBool)>,
    ) -> Result<Self, UtilityInterventionPublicationRejection> {
        use std::io::{Read, Seek, SeekFrom};
        let started = Instant::now();
        let active = || check_intervention_publication_control(control);
        let checked = (|| {
            active()?;
            let bytes = original.original_bytes();
            if bytes.is_empty()
                || bytes.len() > 32 << 20
                || file
                    .metadata()
                    .map_err(|_| invalid("intervention FD metadata"))?
                    .len()
                    != bytes.len() as u64
            {
                return Err(invalid("intervention FD length"));
            }
            active()?;
            file.seek(SeekFrom::Start(0))
                .map_err(|_| invalid("intervention FD seek"))?;
            let mut chunk = [0u8; 8192];
            let mut offset = 0;
            while offset < bytes.len() {
                active()?;
                let n = (bytes.len() - offset).min(chunk.len());
                file.read_exact(&mut chunk[..n])
                    .map_err(|_| invalid("intervention FD readback"))?;
                if chunk[..n] != bytes[offset..offset + n] {
                    return Err(invalid("intervention original FD bytes differ"));
                }
                offset += n;
            }
            active()?;
            if file
                .read(&mut [0u8; 1])
                .map_err(|_| invalid("intervention FD EOF"))?
                != 0
            {
                return Err(invalid("intervention FD grew"));
            }
            active()?;
            file.sync_all()
                .map_err(|_| invalid("intervention FD sync"))?;
            active()?;
            let digest = Digest(Sha256::digest(bytes).into());
            active()?;
            Ok(digest)
        })();
        match checked {
            Ok(file_sha256) => Ok(Self {
                cohort: original.cohort_id(),
                sequence: original.sequence,
                trial: original.digest,
                file_sha256,
                started,
                finished: Instant::now(),
                file,
            }),
            Err(error) => Err(UtilityInterventionPublicationRejection { file, error }),
        }
    }
    pub fn started(&self) -> Instant {
        self.started
    }
    pub fn finished(&self) -> Instant {
        self.finished
    }
    pub fn cohort_id(&self) -> Digest {
        self.cohort
    }
    pub fn trial_sequence(&self) -> u64 {
        self.sequence
    }
    pub fn trial_sha256(&self) -> Digest {
        self.trial
    }
    pub fn file_sha256(&self) -> Digest {
        self.file_sha256
    }
    pub fn collection_wall(&self) -> Duration {
        self.finished.duration_since(self.started)
    }
    pub fn file(&self) -> &std::fs::File {
        &self.file
    }
}
pub struct UtilityInterventionAckRejection {
    pub ack: Arc<UtilityInterventionDurableAck>,
    pub error: AdaptiveManagerError,
}
fn check_intervention_publication_control(
    control: Option<(Instant, &std::sync::atomic::AtomicBool)>,
) -> Result<(), AdaptiveManagerError> {
    if control.is_some_and(|(until, cancel)| {
        Instant::now() >= until || cancel.load(std::sync::atomic::Ordering::Acquire)
    }) {
        return Err(invalid(
            "intervention original publication canceled/expired; source retained",
        ));
    }
    Ok(())
}
impl UtilityInterventionCohortOwner {
    #[allow(clippy::result_large_err)]
    pub fn accept_durable_ack(
        &mut self,
        original: &UtilityInterventionTrialPhysicalRetirement,
        ack: Arc<UtilityInterventionDurableAck>,
    ) -> Result<(), UtilityInterventionAckRejection> {
        self.accept_durable_ack_inner(original, ack, None)
    }
    /// The deadline is checked after acquiring the original state lock, at
    /// the admission gate. A delayed lock cannot open another trial window.
    #[allow(clippy::result_large_err)]
    pub fn accept_durable_ack_with_control(
        &mut self,
        original: &UtilityInterventionTrialPhysicalRetirement,
        ack: Arc<UtilityInterventionDurableAck>,
        until: Instant,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<(), UtilityInterventionAckRejection> {
        self.accept_durable_ack_inner(original, ack, Some((until, cancel)))
    }
    #[allow(clippy::result_large_err)]
    fn accept_durable_ack_inner(
        &mut self,
        original: &UtilityInterventionTrialPhysicalRetirement,
        ack: Arc<UtilityInterventionDurableAck>,
        control: Option<(Instant, &std::sync::atomic::AtomicBool)>,
    ) -> Result<(), UtilityInterventionAckRejection> {
        let checked = (|| {
            check_intervention_publication_control(control)?;
            let mut s = self
                .shared
                .state
                .lock()
                .map_err(|_| invalid("intervention owner poisoned"))?;
            check_intervention_publication_control(control)?;
            if !Arc::ptr_eq(&self.shared, &original.shared)
                || ack.cohort != self.cohort_id()
                || ack.sequence != original.sequence
                || ack.trial != original.digest
                || s.active != Some(original.sequence)
                || s.next as u64 + 1 != original.sequence
            {
                return Err(invalid(
                    "intervention original ACK is foreign, duplicate or reordered",
                ));
            }
            check_intervention_publication_control(control)?;
            s.durable.push(Arc::clone(&ack));
            s.active = None;
            s.next += 1;
            Ok(())
        })();
        match checked {
            Ok(()) => Ok(()),
            Err(error) => Err(UtilityInterventionAckRejection { ack, error }),
        }
    }
}
pub(super) fn recipient(r: RoventUtilityCohortRecipient) -> UtilityRouteRecipient {
    match r {
        RoventUtilityCohortRecipient::Proposer => UtilityRouteRecipient::Proposer,
        RoventUtilityCohortRecipient::Critic => UtilityRouteRecipient::Critic,
        RoventUtilityCohortRecipient::Researcher => UtilityRouteRecipient::Researcher,
    }
}
fn pins_match(p: &UtilityCohortSourcePins, c: &RoventUtilityInterventionCohortV2) -> bool {
    let i = &c.input;
    p.input_lock_sha256 == i.input_lock_sha256
        && p.source_sha256 == i.source_sha256
        && p.frozen_nnuec_model_sha256 == i.frozen_nnuec_model_sha256
        && p.frozen_nnuec_namespace_sha256 == i.frozen_nnuec_namespace_sha256
        && p.frozen_pc_model_sha256 == i.frozen_pc_model_sha256
        && p.frozen_pc_namespace_sha256 == i.frozen_pc_namespace_sha256
        && p.resource_plan_sha256 == i.resource_plan_sha256
        && p.logical_budget_sha256 == i.logical_budget_sha256
}
pub(super) fn framed(domain: &str, bytes: &[u8]) -> Digest {
    let mut h = Sha256::new();
    h.update(domain.as_bytes());
    h.update([0]);
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(bytes);
    Digest(h.finalize().into())
}

/// A pair is issued only by the original two typed /2 physical retirements.
/// Durations, never absolute ticks from independent clocks, are compared.
pub struct UtilityInterventionPair {
    left: Arc<UtilityInterventionTrialPhysicalRetirement>,
    right: Arc<UtilityInterventionTrialPhysicalRetirement>,
    preferred: Option<UtilityRouteRecipient>,
    unknown: Option<&'static str>,
    bytes: Vec<u8>,
    digest: Digest,
}
pub struct UtilityInterventionPairRejection {
    pub left: Arc<UtilityInterventionTrialPhysicalRetirement>,
    pub right: Arc<UtilityInterventionTrialPhysicalRetirement>,
    pub error: AdaptiveManagerError,
}
impl UtilityInterventionPair {
    #[allow(clippy::result_large_err)]
    pub fn compare(
        left: Arc<UtilityInterventionTrialPhysicalRetirement>,
        right: Arc<UtilityInterventionTrialPhysicalRetirement>,
    ) -> Result<Self, UtilityInterventionPairRejection> {
        let checked = (|| {
            if !Arc::ptr_eq(&left.shared, &right.shared)
                || left.source_pins() != right.source_pins()
            {
                return Err(invalid("intervention pair foreign registered source"));
            }
            if left.sequence == right.sequence
                || left.authority == right.authority
                || left.digest == right.digest
                || left.cost.clock_epoch == right.cost.clock_epoch
            {
                return Err(invalid(
                    "intervention pair does not have independent original roots/clocks",
                ));
            }
            if left.recipient == right.recipient {
                return Err(invalid("intervention pair repeats the same action"));
            }
            if left.source_budget != right.source_budget {
                return Err(invalid("intervention pair original source budgets differ"));
            }
            if let (Some(left_outcome), Some(right_outcome)) = (left.semantic(), right.semantic()) {
                let left_baseline = left_outcome
                    .baseline_witnesses()
                    .map(|d| d.0)
                    .collect::<std::collections::BTreeSet<_>>();
                let right_baseline = right_outcome
                    .baseline_witnesses()
                    .map(|d| d.0)
                    .collect::<std::collections::BTreeSet<_>>();
                if !same_control(
                    left.source_budget,
                    right.source_budget,
                    &left_baseline,
                    &right_baseline,
                ) {
                    return Err(invalid(
                        "intervention pair original semantic witness baselines differ",
                    ));
                }
            }
            if left.neutral_target_digest().is_some()
                && right.neutral_target_digest().is_some()
                && left.neutral_target_digest() != right.neutral_target_digest()
            {
                return Err(invalid("intervention pair neutral targets differ"));
            }
            let set = |t: &UtilityInterventionTrialPhysicalRetirement| {
                t.semantic()
                    .into_iter()
                    .flat_map(|s| s.witnesses())
                    .map(|w| w.identity().0)
                    .collect::<std::collections::BTreeSet<_>>()
            };
            let l = set(&left);
            let r = set(&right);
            let (choice, unknown) = if !left.comparable()
                || !right.comparable()
                || left.neutral_target_digest().is_none()
                || right.neutral_target_digest().is_none()
            {
                (None, Some("unresolved_or_masked_trial"))
            } else {
                pareto(&l, &r, left.cost.whole_wall_ns, right.cost.whole_wall_ns)
            };
            let preferred = choice.map(|left_preferred| {
                if left_preferred {
                    left.recipient
                } else {
                    right.recipient
                }
            });
            let mut support = left.support_mask();
            for (a, b) in support.iter_mut().zip(right.support_mask()) {
                *a |= b;
            }
            let mut known = [false; 7];
            if preferred.is_some() {
                known[left.recipient.output_index()] = true;
                known[right.recipient.output_index()] = true;
            }
            let wire = json!({"schema":rz_experiments::rovent_utility_cohort::ROVENT_UTILITY_INTERVENTION_PAIR_DOMAIN,"cohort_id":hex(left.cohort_id()),
                "neutral_target_sha256":left.neutral_target_digest().or(right.neutral_target_digest()).map(hex),
                "left_trial_sequence":left.sequence,"right_trial_sequence":right.sequence,"left_trial_sha256":hex(left.digest),"right_trial_sha256":hex(right.digest),
                "left_recipient":left.recipient.as_str(),"right_recipient":right.recipient.as_str(),"preferred":preferred.map(|p|p.as_str()),"unknown_reason":unknown,
                "support_mask":support,"target_known_mask":known,"left_new_witnesses":l.iter().map(|d|hex(Digest(*d))).collect::<Vec<_>>(),"right_new_witnesses":r.iter().map(|d|hex(Digest(*d))).collect::<Vec<_>>(),
                "left_whole_wall_ns":left.cost.whole_wall_ns,"right_whole_wall_ns":right.cost.whole_wall_ns,"cost_scope":UTILITY_COHORT_COST_SCOPE,"objective":"exclusive_wall_nanoseconds"});
            Ok((preferred, unknown, bounded_json(&wire)?))
        })();
        match checked {
            Ok((preferred, unknown, bytes)) => {
                let digest = framed(
                    rz_experiments::rovent_utility_cohort::ROVENT_UTILITY_INTERVENTION_PAIR_DOMAIN,
                    &bytes,
                );
                Ok(Self {
                    left,
                    right,
                    preferred,
                    unknown,
                    bytes,
                    digest,
                })
            }
            Err(error) => Err(UtilityInterventionPairRejection { left, right, error }),
        }
    }
    pub fn left(&self) -> &Arc<UtilityInterventionTrialPhysicalRetirement> {
        &self.left
    }
    pub fn right(&self) -> &Arc<UtilityInterventionTrialPhysicalRetirement> {
        &self.right
    }
    pub fn preferred(&self) -> Option<UtilityRouteRecipient> {
        self.preferred
    }
    pub fn unknown_reason(&self) -> Option<&'static str> {
        self.unknown
    }
    pub fn original_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn sha256(&self) -> Digest {
        self.digest
    }
    pub fn target_known_mask(&self) -> [bool; 7] {
        let mut mask = [false; 7];
        if self.preferred.is_some() {
            mask[self.left.recipient.output_index()] = true;
            mask[self.right.recipient.output_index()] = true;
        }
        mask
    }
}
fn pareto(
    left: &std::collections::BTreeSet<[u8; 32]>,
    right: &std::collections::BTreeSet<[u8; 32]>,
    left_cost: u64,
    right_cost: u64,
) -> (Option<bool>, Option<&'static str>) {
    if left.is_empty() && right.is_empty() {
        (None, Some("no_new_valid_witness"))
    } else if left == right && left_cost == right_cost {
        (None, Some("equal_observation"))
    } else if left.is_superset(right) && left_cost <= right_cost {
        (Some(true), None)
    } else if right.is_superset(left) && right_cost <= left_cost {
        (Some(false), None)
    } else {
        (None, Some("pareto_incomparable"))
    }
}
fn same_control(
    left: rz_search::pals::coordinator::FrameBudget,
    right: rz_search::pals::coordinator::FrameBudget,
    left_baseline: &std::collections::BTreeSet<[u8; 32]>,
    right_baseline: &std::collections::BTreeSet<[u8; 32]>,
) -> bool {
    left == right && left_baseline == right_baseline
}

fn compare_neutral_targets(
    expected: Digest,
    previous: &Value,
    actual: Digest,
    current: &Value,
) -> Result<(), AdaptiveManagerError> {
    let reference_matches = previous["reference_json"] == current["reference_json"];
    let model_matches = previous["model_identity"] == current["model_identity"];
    if expected != actual || !reference_matches || !model_matches {
        Err(AdaptiveManagerError::InterventionNeutralMismatch {
            expected, actual, reference_matches, model_matches,
        })
    } else {
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn neutral_mismatch_keeps_both_digests_and_comparison_scope() {
        let first = Digest([1; 32]);
        let other = Digest([2; 32]);
        let original = json!({"reference_json":"original", "model_identity":"model"});
        assert!(compare_neutral_targets(first, &original, first, &original).is_ok());
        let changed = json!({"reference_json":"different", "model_identity":"model"});
        assert!(matches!(compare_neutral_targets(first, &original, other, &changed),
            Err(AdaptiveManagerError::InterventionNeutralMismatch {
                expected, actual, reference_matches: false, model_matches: true,
            }) if expected == first && actual == other));
        let changed = json!({"reference_json":"original", "model_identity":"other"});
        assert!(matches!(compare_neutral_targets(first, &original, first, &changed),
            Err(AdaptiveManagerError::InterventionNeutralMismatch {
                reference_matches: true, model_matches: false, ..
            })));
    }
    #[test]
    fn utility_requires_a_witness_and_does_not_trade_cost_for_more_discovery() {
        let empty = std::collections::BTreeSet::new();
        let one = std::collections::BTreeSet::from([[1; 32]]);
        let two = std::collections::BTreeSet::from([[1; 32], [2; 32]]);
        assert_eq!(
            pareto(&empty, &empty, 1, 100),
            (None, Some("no_new_valid_witness"))
        );
        assert_eq!(
            pareto(&two, &one, 100, 1),
            (None, Some("pareto_incomparable"))
        );
        assert_eq!(pareto(&two, &one, 1, 100), (Some(true), None));
    }
    #[test]
    fn wire_domains_cannot_promote_a_legacy_trial() {
        let raw = b"{}";
        assert_ne!(
            framed("rz-rovent-utility-cohort-trial/1", raw),
            framed("rz-rovent-utility-cohort-trial/2", raw)
        );
        assert_ne!(
            framed("rz-rovent-utility-cohort-pair/2", raw),
            framed("rz-rovent-utility-cohort-trial/2", raw)
        );
    }
    #[test]
    fn equal_neural_input_never_substitutes_source_budget_or_semantic_baseline() {
        let base = rz_search::pals::coordinator::FrameBudget {
            role_submissions_remaining: 4,
            original_cpu_credit: 1024,
            original_store_byte_limit: 4096,
        };
        let witnesses = std::collections::BTreeSet::from([[1; 32], [2; 32]]);
        assert!(same_control(base, base, &witnesses, &witnesses));
        let mut changed = base;
        changed.role_submissions_remaining -= 1;
        assert!(!same_control(base, changed, &witnesses, &witnesses));
        changed = base;
        changed.original_cpu_credit -= 1;
        assert!(!same_control(base, changed, &witnesses, &witnesses));
        changed = base;
        changed.original_store_byte_limit -= 1;
        assert!(!same_control(base, changed, &witnesses, &witnesses));
        assert!(!same_control(
            base,
            base,
            &witnesses,
            &std::collections::BTreeSet::from([[1; 32]])
        ));
    }
}
