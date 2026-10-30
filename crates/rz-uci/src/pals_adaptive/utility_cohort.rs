//! Optional, outcome-before-registration-safe independent utility trials.
//! Historical root-local actions cannot be passed to this owner or promoted.
use super::{AdaptiveManagerCounters, AdaptiveManagerError};
use super::{UtilityCohortReadySnapshot, UtilityCohortRootClosureProof};
use crate::rovent_bootstrap::nnuec_source::LoadedRoventNnuecSource;
use rz_contracts::Digest;
use rz_contracts::pals::SearchAuthority;
use rz_experiments::rovent_utility_cohort::{
    ROVENT_UTILITY_COMPARISON_COST_SCOPE, RoventUtilityCohortRecipient, RoventUtilityCohortUsage,
    RoventUtilityComparisonCohort,
};
use rz_runtime::contracts::{ContractClock as _, ContractSystemClock};
use rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture;
use rz_search::pals::coordinator::CoordinatorWork;
use rz_search::pals::engine::adaptive_port::utility_route::{
    UtilityRouteAction, UtilityRouteActionId, UtilityRouteRecipient,
};
use rz_search::pals::review_context::ReviewView;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const UTILITY_COHORT_TRIAL: &str = "rz-rovent-utility-cohort-trial/1";
pub const UTILITY_COHORT_PAIR: &str = "rz-rovent-utility-cohort-pair/1";
pub const UTILITY_COHORT_MATERIALIZATION: &str = "rz-rovent-utility-cohort-materialization/1";
pub const UTILITY_COHORT_COST: &str = "rz-rovent-utility-cohort-cost/1";
pub const UTILITY_COHORT_COST_SCOPE: &str = ROVENT_UTILITY_COMPARISON_COST_SCOPE;
pub const UTILITY_COHORT_HISTORICAL_MAX_BYTES: usize = 32 << 20;

/// Closed metadata projection of the original source. Deserialization grants
/// no loaded model, current root, cache state, offer or trial capability.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UtilityCohortSourcePins {
    pub input_lock_sha256: String,
    pub source_sha256: String,
    pub frozen_nnuec_model_sha256: String,
    pub frozen_nnuec_namespace_sha256: String,
    pub frozen_pc_model_sha256: String,
    pub frozen_pc_namespace_sha256: String,
    pub resource_plan_sha256: String,
    pub logical_budget_sha256: String,
}
impl UtilityCohortSourcePins {
    pub fn from_actual<Q>(
        snapshot: &UtilityCohortReadySnapshot<Q>,
        source: &LoadedRoventNnuecSource,
    ) -> Result<Self, AdaptiveManagerError> {
        let actual = snapshot.actual_root();
        let admission = snapshot.admission();
        let native = admission.native_origin();
        let original = native.raw_receipt();
        let lock = native.original_input_lock();
        if !native.matches_loaded_nnuec(source)
            || source.input_lock_digest() != actual.startup_lock_digest()
            || snapshot.authority() != actual.authority()
            || snapshot.root_budget() != actual.root_budget_digest()
            || !actual
                .root_snapshot()
                .same_state(admission.rules().rules().snapshot())
        {
            return Err(invalid("cohort source is not the original admitted root"));
        }
        let pc_namespace = utility_pc_namespace_projection(original)?;
        let projected = Self::from_projection(
            lock,
            source.registration(),
            &pc_namespace,
            actual.declared_ceilings(),
        )?;
        if projected.input_lock_sha256 != hex(source.input_lock_digest())
            || projected.source_sha256 != string(original, "source_sha256")?
            || projected.frozen_nnuec_model_sha256
                != source
                    .semantic_weights_identity()
                    .weights_sha256
                    .clone()
                    .ok_or_else(|| invalid("cohort actual NNUEC has no semantic weights"))?
        {
            return Err(invalid("cohort original source projection differs"));
        }
        Ok(projected)
    }

    /// Expected metadata only. The caller must separately verify original
    /// lock/config/export/runtime/NNUEC descriptor bytes. This cannot issue a
    /// Ready snapshot, current root, cache ACK, offer or trial admission.
    pub fn from_registered_metadata(
        lock: &rz_experiments::RoventInputLock,
        registration: &rz_search::cpu_value::nnuec::v2::registration::NnuecV2Registration,
        export_manifest: &rz_experiments::ArtifactRef,
        backend: Digest,
    ) -> Result<Self, AdaptiveManagerError> {
        use rz_eval::pals_model::{PalsModelProfile, virtual_review};
        let model = &lock.manifest.engine.pc.proposer;
        if !lock.manifest.provenance.contains(export_manifest)
            || model.semantics != virtual_review::MODEL_SEMANTICS
            || model.encoding != virtual_review::ENCODING
        {
            return Err(invalid("cohort metadata original export/model differs"));
        }
        let pc_namespace = json!({
            "model_semantics":model.semantics,
            "model_semantics_sha256":hex(Digest(PalsModelProfile::VirtualReviewV1.registered_semantics_sha256())),
            "encoding":model.encoding,
            "encoding_sha256":hex(Digest(Sha256::digest(virtual_review::ENCODING.as_bytes()).into())),
            "encoding_registry_sha256":hex(Digest(virtual_review::registered_encoding_sha256())),
            "model_implementation_sha256":model.implementation_sha256,
            "representation_model_domain":"loaded_checkpoint_sha256",
            "frozen_epoch":1,
            "weights_sha256":model.weights.sha256,
            "export_manifest_sha256":export_manifest.sha256,
            "backend_sha256":hex(backend),
        });
        Self::from_projection(
            lock,
            registration,
            &pc_namespace,
            &Self::registered_ceilings(lock),
        )
    }

    fn registered_ceilings(
        lock: &rz_experiments::RoventInputLock,
    ) -> rz_experiments::RoventRootDeclaredCeilings {
        rz_experiments::RoventRootDeclaredCeilings {
            resource_plan: lock.manifest.resources.clone(),
            registered_max_chess_plies: lock.manifest.engine.search.registered_max_chess_plies,
            cpu_credit_limit: lock.manifest.coordinator.cpu_credit_limit,
            questions_max: lock.manifest.coordinator.questions_max,
            ancestor_depth_max: lock.manifest.coordinator.ancestor_depth_max,
            waiting_jobs_max: lock.manifest.coordinator.waiting_jobs_max,
            role_tickets_max: lock.manifest.coordinator.role_tickets_max,
            wall_time_max_ms: lock.manifest.wall_time_max_ms,
            cleanup_max_ms: lock.manifest.cleanup_max_ms,
            output_bytes_max: lock.manifest.output_bytes_max,
        }
    }

    fn from_projection(
        lock: &rz_experiments::RoventInputLock,
        registration: &rz_search::cpu_value::nnuec::v2::registration::NnuecV2Registration,
        pc_namespace: &Value,
        ceilings: &rz_experiments::RoventRootDeclaredCeilings,
    ) -> Result<Self, AdaptiveManagerError> {
        lock.validate()
            .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
        if ceilings != &Self::registered_ceilings(lock) {
            return Err(invalid(
                "cohort original declared ceilings differ from lock",
            ));
        }
        let pc_namespace = utility_pc_namespace_projection(pc_namespace)?;
        let weights = registration
            .model_identity()
            .weights_sha256
            .clone()
            .ok_or_else(|| invalid("cohort NNUEC registration has no semantic weights"))?;
        Ok(Self {
            input_lock_sha256: lock.canonical_sha256.clone(),
            source_sha256: rz_experiments::rovent_source_aggregate_sha256(
                &lock.manifest.engine.source_commit,
                &lock.manifest.engine.source_components,
            )
            .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?,
            frozen_nnuec_model_sha256: weights,
            frozen_nnuec_namespace_sha256: hex(value_digest(
                "rz-rovent-utility-cohort-nnuec-namespace/1",
                registration,
            )?),
            frozen_pc_model_sha256: string(&pc_namespace, "weights_sha256")?,
            frozen_pc_namespace_sha256: hex(value_digest(
                "rz-rovent-utility-cohort-pc-namespace/1",
                &pc_namespace,
            )?),
            resource_plan_sha256: lock
                .manifest
                .resources
                .canonical_sha256()
                .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?,
            logical_budget_sha256: hex(value_digest(
                "rz-rovent-utility-cohort-logical-budget/1",
                ceilings,
            )?),
        })
    }
    fn matches_profile(&self, p: &RoventUtilityComparisonCohort) -> bool {
        let i = &p.input;
        self.input_lock_sha256 == i.input_lock_sha256
            && self.source_sha256 == i.source_sha256
            && self.frozen_nnuec_model_sha256 == i.frozen_nnuec_model_sha256
            && self.frozen_nnuec_namespace_sha256 == i.frozen_nnuec_namespace_sha256
            && self.frozen_pc_model_sha256 == i.frozen_pc_model_sha256
            && self.frozen_pc_namespace_sha256 == i.frozen_pc_namespace_sha256
            && self.resource_plan_sha256 == i.resource_plan_sha256
            && self.logical_budget_sha256 == i.logical_budget_sha256
    }
}

fn utility_pc_namespace_projection(original: &Value) -> Result<Value, AdaptiveManagerError> {
    let names = [
        "model_semantics",
        "model_semantics_sha256",
        "encoding",
        "encoding_sha256",
        "encoding_registry_sha256",
        "model_implementation_sha256",
        "representation_model_domain",
        "frozen_epoch",
        "weights_sha256",
        "export_manifest_sha256",
        "backend_sha256",
    ];
    let mut selected = serde_json::Map::new();
    for name in names {
        let value = original
            .get(name)
            .ok_or_else(|| invalid("cohort PC namespace field absent"))?;
        if name == "frozen_epoch" {
            if value.as_u64() != Some(1) {
                return Err(invalid("cohort PC namespace frozen epoch differs"));
            }
        } else if name.ends_with("sha256") {
            let text = value
                .as_str()
                .ok_or_else(|| invalid("cohort PC namespace SHA type"))?;
            rz_eval::asset::parse_sha256(text)
                .map_err(|_| invalid("cohort PC namespace SHA invalid"))?;
        } else if value.as_str().is_none_or(|s| s.is_empty() || s.len() > 256) {
            return Err(invalid("cohort PC namespace semantic tag invalid"));
        }
        selected.insert(name.into(), value.clone());
    }
    Ok(Value::Object(selected))
}

/// The first successful source-action bind freezes these original bytes before
/// P/C dispatch or R Begin. A later trial may compare, never overwrite them.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UtilityCohortFrozenDecision {
    pub input_sha256: String,
    pub target_sha256: String,
    pub context_sha256: String,
    pub target_record_ids: Vec<u64>,
    pub input_bytes: u64,
    pub root_rules_sha256: String,
    pub original_first_move: u16,
    pub first_trial_sequence: u64,
}
struct Shared {
    profile: RoventUtilityComparisonCohort,
    profile_digest: Digest,
    cohort: Digest,
    source_pins: UtilityCohortSourcePins,
    offer_group: Digest,
    planned: Vec<UtilityRouteRecipient>,
    state: Mutex<CohortState>,
}
struct CohortState {
    next: usize,
    active: Option<u64>,
    original_roots: Vec<Arc<crate::rovent_root_admission::RoventRootAdmission>>,
    frozen: Option<UtilityCohortFrozenDecision>,
    durable: Vec<Arc<UtilityCohortDurableAck>>,
    poisoned: bool,
}
/// Not Clone: one current trial and its unsettled retirement remain visible.
pub struct UtilityComparisonCohortOwner {
    shared: Arc<Shared>,
}
impl UtilityComparisonCohortOwner {
    pub fn register<Q>(
        profile: RoventUtilityComparisonCohort,
        original: &UtilityCohortReadySnapshot<Q>,
        loaded: Arc<LoadedRoventNnuecSource>,
    ) -> Result<Self, AdaptiveManagerError> {
        profile
            .validate()
            .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
        let profile_digest = profile
            .digest()
            .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
        if original.cohort_profile_digest() != profile_digest
            || original.captured_at() > original.prelude_cutoff()
        {
            return Err(invalid("cohort original manager prelude/profile differs"));
        }
        let pins = UtilityCohortSourcePins::from_actual(original, &loaded)?;
        if !pins.matches_profile(&profile)
            || profile.input.mode.execution_mode()
                != original
                    .admission()
                    .native_origin()
                    .original_input_lock()
                    .manifest
                    .engine
                    .mode
            || original.captured_at() >= original.hard_deadline()
        {
            return Err(invalid("cohort profile/source/mode differs"));
        }
        let offer_group = actual_offer_group(original)?;
        let planned: Vec<_> = profile
            .input
            .requested_recipients
            .iter()
            .copied()
            .map(recipient)
            .filter(|r| original.offers().iter().any(|o| o.recipient() == *r))
            .collect();
        if planned.is_empty() {
            return Err(invalid("cohort has no actual requested ready offer"));
        }
        let mut h = Sha256::new();
        h.update(b"rz-rovent-utility-cohort-owner/1\0");
        h.update(profile_digest.0);
        h.update(original.actual_root().digest().0);
        h.update(offer_group.0);
        for r in &planned {
            h.update([r.output_index() as u8]);
        }
        let cohort = Digest(h.finalize().into());
        Ok(Self {
            shared: Arc::new(Shared {
                profile,
                profile_digest,
                cohort,
                source_pins: pins,
                offer_group,
                planned,
                state: Mutex::new(CohortState {
                    next: 0,
                    active: None,
                    original_roots: Vec::new(),
                    frozen: None,
                    durable: Vec::new(),
                    poisoned: false,
                }),
            }),
        })
    }
    pub fn cohort_id(&self) -> Digest {
        self.shared.cohort
    }
    pub fn profile(&self) -> &RoventUtilityComparisonCohort {
        &self.shared.profile
    }
    pub fn profile_digest(&self) -> Digest {
        self.shared.profile_digest
    }
    pub fn source_pins(&self) -> &UtilityCohortSourcePins {
        &self.shared.source_pins
    }
    pub fn planned_recipients(&self) -> &[UtilityRouteRecipient] {
        &self.shared.planned
    }
    pub fn frozen_decision(
        &self,
    ) -> Result<Option<UtilityCohortFrozenDecision>, AdaptiveManagerError> {
        Ok(self
            .shared
            .state
            .lock()
            .map_err(|_| invalid("cohort owner poisoned"))?
            .frozen
            .clone())
    }
    #[allow(clippy::result_large_err)]
    pub fn admit_trial<Q>(
        &mut self,
        snapshot: UtilityCohortReadySnapshot<Q>,
        requested: UtilityRouteRecipient,
        loaded: Arc<LoadedRoventNnuecSource>,
    ) -> Result<UtilityTrialAdmission<Q>, UtilityTrialAdmissionRejection<Q>> {
        let checked = (|| {
            let pins = UtilityCohortSourcePins::from_actual(&snapshot, &loaded)?;
            let group = actual_offer_group(&snapshot)?;
            let mut state = self
                .shared
                .state
                .lock()
                .map_err(|_| invalid("cohort owner poisoned"))?;
            if state.poisoned
                || state.active.is_some()
                || state.next >= self.shared.planned.len()
                || state.durable.len() != state.next
                || self.shared.planned[state.next] != requested
                || pins != self.shared.source_pins
                || snapshot.cohort_profile_digest() != self.shared.profile_digest
                || snapshot.captured_at() > snapshot.prelude_cutoff()
                || group != self.shared.offer_group
                || snapshot.captured_at() >= snapshot.hard_deadline()
                || state.original_roots.iter().any(|r| {
                    Arc::ptr_eq(r, snapshot.actual_root()) || r.authority() == snapshot.authority()
                })
            {
                return Err(invalid(
                    "cohort trial is busy, repeated, mismatched or out of order",
                ));
            }
            let selected = snapshot
                .selected_offer_index(requested)
                .filter(|index| {
                    snapshot
                        .offers()
                        .get(*index)
                        .is_some_and(|o| o.recipient() == requested)
                })
                .ok_or_else(|| invalid("requested trial has no actual fair ready selection"))?;
            let ready_prelude = super::utility_cohort_historical::freeze_ready_prelude(
                &snapshot,
                self.shared.offer_group,
            )?;
            let original_root = snapshot.admission().raw_receipt().clone();
            let _bounded_root = super::utility_cohort_historical::bounded_json(&original_root)?;
            let sequence = state.next as u64 + 1;
            state.active = Some(sequence);
            state
                .original_roots
                .push(Arc::clone(snapshot.actual_root()));
            Ok((sequence, selected, ready_prelude, original_root))
        })();
        match checked {
            Ok((sequence, selected, ready_prelude, original_root)) => Ok(UtilityTrialAdmission {
                shared: Arc::clone(&self.shared),
                snapshot,
                loaded,
                ready_prelude,
                original_root,
                sequence,
                selected,
                armed: None,
            }),
            Err(error) => Err(UtilityTrialAdmissionRejection {
                snapshot,
                loaded,
                error,
            }),
        }
    }
}
pub struct UtilityTrialAdmissionRejection<Q> {
    pub snapshot: UtilityCohortReadySnapshot<Q>,
    pub loaded: Arc<LoadedRoventNnuecSource>,
    pub error: AdaptiveManagerError,
}
/// Original actual-root scope with a finite registered action cutoff. The
/// cutoff never renews an original Q, Job, clock, credit or hard deadline.
#[derive(Clone, Copy, Debug)]
pub struct UtilityTrialWindow {
    started: Instant,
    action: Instant,
    cleanup: Instant,
}
impl UtilityTrialWindow {
    pub fn started(&self) -> Instant {
        self.started
    }
    pub fn action_cutoff(&self) -> Instant {
        self.action
    }
    pub fn cleanup_cutoff(&self) -> Instant {
        self.cleanup
    }
}
pub struct UtilityTrialAdmission<Q> {
    shared: Arc<Shared>,
    snapshot: UtilityCohortReadySnapshot<Q>,
    loaded: Arc<LoadedRoventNnuecSource>,
    ready_prelude: Value,
    original_root: Value,
    sequence: u64,
    selected: usize,
    armed: Option<UtilityTrialWindow>,
}
impl<Q> UtilityTrialAdmission<Q> {
    pub fn original_root(&self) -> &Value {
        &self.original_root
    }
    pub fn ready_prelude(&self) -> &Value {
        &self.ready_prelude
    }
    pub fn ready_group_sha256(&self) -> Digest {
        self.shared.offer_group
    }
    pub fn loaded_source(&self) -> &Arc<LoadedRoventNnuecSource> {
        &self.loaded
    }
    pub fn profile_digest(&self) -> Digest {
        self.shared.profile_digest
    }
    pub fn source_pins(&self) -> &UtilityCohortSourcePins {
        &self.shared.source_pins
    }
    pub fn profile(&self) -> &RoventUtilityComparisonCohort {
        &self.shared.profile
    }

    pub fn cohort_id(&self) -> Digest {
        self.shared.cohort
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn recipient(&self) -> UtilityRouteRecipient {
        self.snapshot.offers()[self.selected].recipient()
    }
    pub fn snapshot(&self) -> &UtilityCohortReadySnapshot<Q> {
        &self.snapshot
    }
    pub fn window(&self) -> Option<UtilityTrialWindow> {
        self.armed
    }
    pub fn matches_ready_work(&self, w: &CoordinatorWork<ReviewView, Q>) -> bool {
        self.snapshot.offers()[self.selected].matches_ready_work(w)
    }
    pub fn validate_manager_registration(
        &self,
        authority: SearchAuthority,
        root_budget: Digest,
        clock: &ContractSystemClock,
        hard: Instant,
        cleanup: Instant,
    ) -> Result<(), AdaptiveManagerError> {
        if self.snapshot.authority() != authority
            || self.snapshot.root_budget() != root_budget
            || self.snapshot.clock().domain() != clock.domain()
            || self.snapshot.clock().origin() != clock.origin()
            || self.snapshot.hard_deadline() != hard
            || self.snapshot.cleanup_deadline() != cleanup
        {
            return Err(invalid(
                "trial original manager/root/clock/deadline differs",
            ));
        }
        Ok(())
    }
    /// Only the real manager can arm this at its exclusive gate start. A
    /// declaration or historical caller cannot issue another wall interval.
    pub(in crate::pals_adaptive) fn arm_window(
        &mut self,
        started: Instant,
    ) -> Result<UtilityTrialWindow, AdaptiveManagerError> {
        if self.armed.is_some() || started < self.snapshot.captured_at() {
            return Err(invalid(
                "cohort action window is already armed or precedes actual offers",
            ));
        }
        let window = checked_trial_window(
            started,
            self.shared.profile.input.action_ns,
            self.shared.profile.input.cleanup_ns,
            self.snapshot.hard_deadline(),
            self.snapshot.cleanup_deadline(),
        )?;
        self.armed = Some(window);
        Ok(window)
    }
    #[allow(clippy::result_large_err)]
    pub fn bind_submitted_input(
        self,
        action: &UtilityRouteAction,
    ) -> Result<UtilityTrialBoundAdmission<Q>, UtilityTrialBindRejection<Q>> {
        let checked = (|| {
            let window = self
                .armed
                .ok_or_else(|| invalid("cohort gate has no actual armed window"))?;
            let selected = &self.snapshot.offers()[self.selected];
            let input = action.input();
            if action.failed()
                || action.has_committed()
                || action.id().authority() != self.snapshot.authority()
                || action.id().frame() != selected.frame()
                || action.id().question() != selected.question()
                || action.id().recipient() != selected.recipient()
                || action.root_budget() != self.snapshot.root_budget()
                || action.original_deadline() != selected.original_deadline()
                || action.started() < window.started
                || !input.matches_original_model(self.loaded.model())
                || !input
                    .root_snapshot()
                    .same_state(self.snapshot.actual_root().root_snapshot())
            {
                return Err(invalid(
                    "cohort actual submitted action/input differs from original selected offer",
                ));
            }
            if Instant::now() >= window.action {
                return Err(AdaptiveManagerError::AdmissionClosed);
            }
            let frozen = UtilityCohortFrozenDecision {
                input_sha256: input.input_sha256().into(),
                target_sha256: input.target_sha256().into(),
                context_sha256: input.context_sha256().into(),
                target_record_ids: input.target_record_ids().to_vec(),
                input_bytes: u64::try_from(input.input_bytes().len())
                    .map_err(|_| invalid("utility bytes overflow"))?,
                root_rules_sha256: hex(rz_position::contracts::rules_snapshot_semantic_digest(
                    input.root_snapshot(),
                )),
                original_first_move: rz_contracts::pals::Move16::encode(
                    rz_contracts::Move::try_from(input.original_first_move())
                        .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?,
                )
                .bits(),
                first_trial_sequence: self.sequence,
            };
            let mut state = self
                .shared
                .state
                .lock()
                .map_err(|_| invalid("cohort owner poisoned"))?;
            if state.active != Some(self.sequence) || state.poisoned {
                return Err(invalid("cohort original trial no longer active"));
            }
            if let Some(original) = &state.frozen {
                if !same_input(original, &frozen) {
                    return Err(invalid(
                        "independent trial changed canonical utility input/target",
                    ));
                }
            } else {
                state.frozen = Some(frozen);
            }
            // This epoch is issued only by the registered current trial. Old
            // root-local Completed actions have no path into this constructor.
            let mut h = Sha256::new();
            h.update(b"rz-rovent-utility-cohort-cost-epoch/1\0");
            h.update(self.shared.cohort.0);
            h.update(self.shared.profile_digest.0);
            h.update(self.shared.offer_group.0);
            h.update(input.input_sha256().as_bytes());
            h.update(input.target_sha256().as_bytes());
            h.update(UTILITY_COHORT_COST_SCOPE.as_bytes());
            Ok(Digest(h.finalize().into()))
        })();
        match checked {
            Ok(cost_epoch) => Ok(UtilityTrialBoundAdmission {
                trial: self,
                action: action.id().clone(),
                input: Arc::clone(action.input()),
                cost_epoch,
            }),
            Err(error) => Err(UtilityTrialBindRejection { trial: self, error }),
        }
    }
}
pub struct UtilityTrialBindRejection<Q> {
    pub trial: UtilityTrialAdmission<Q>,
    pub error: AdaptiveManagerError,
}
pub struct UtilityTrialBoundAdmission<Q> {
    trial: UtilityTrialAdmission<Q>,
    action: UtilityRouteActionId,
    input: Arc<NnuecUtilityInputCapture>,
    cost_epoch: Digest,
}
impl<Q> UtilityTrialBoundAdmission<Q> {
    pub fn trial(&self) -> &UtilityTrialAdmission<Q> {
        &self.trial
    }
    pub fn action(&self) -> &UtilityRouteActionId {
        &self.action
    }
    pub fn input(&self) -> &Arc<NnuecUtilityInputCapture> {
        &self.input
    }
    pub fn cost_epoch(&self) -> Digest {
        self.cost_epoch
    }
    pub fn cost_scope(&self) -> &'static str {
        UTILITY_COHORT_COST_SCOPE
    }
    pub fn is_validation_only(&self) -> bool {
        self.trial.shared.profile.input.usage == RoventUtilityCohortUsage::ValidationOnly
    }
}

pub(super) fn actual_offer_group<Q>(
    snapshot: &UtilityCohortReadySnapshot<Q>,
) -> Result<Digest, AdaptiveManagerError> {
    let mut offers = Vec::new();
    for offer in snapshot.offers() {
        let source = offer
            .semantic_source()
            .ok_or_else(|| invalid("cohort offer has no original semantic source"))?;
        offers.push((offer.recipient().output_index() as u8, source.digest().0));
    }
    offers.sort_unstable();
    let mut h = Sha256::new();
    h.update(b"rz-rovent-utility-cohort-ready-offer-group/1\0");
    h.update((offers.len() as u64).to_le_bytes());
    for (role, digest) in offers {
        h.update([role]);
        h.update(digest);
    }
    h.update(
        value_digest(
            "rz-rovent-utility-cohort-common-node-work/1",
            &json!({
        "requested_nodes":snapshot.requested_common_nodes(),
        "work":common_node_value(snapshot.common_node_work())}),
        )?
        .0,
    );
    let credit = snapshot.native_credit();
    h.update(credit.native_node_limit.to_le_bytes());
    h.update(credit.spent_native_nodes.to_le_bytes());
    h.update(credit.reserved_native_nodes.to_le_bytes());
    h.update(snapshot.role_credit().0.to_le_bytes());
    h.update(snapshot.role_credit().1.to_le_bytes());
    let mut counters = counter_value(snapshot.counters());
    if let Some(object) = counters.as_object_mut() {
        object.remove("notification_waits");
    }
    h.update(value_digest("rz-rovent-utility-cohort-prelude-counters/1", &counters)?.0);
    h.update(
        value_digest(
            "rz-rovent-utility-cohort-actual-resource-observations/1",
            &resource_observation_value(snapshot),
        )?
        .0,
    );
    Ok(Digest(h.finalize().into()))
}
fn recipient(r: RoventUtilityCohortRecipient) -> UtilityRouteRecipient {
    match r {
        RoventUtilityCohortRecipient::Proposer => UtilityRouteRecipient::Proposer,
        RoventUtilityCohortRecipient::Critic => UtilityRouteRecipient::Critic,
        RoventUtilityCohortRecipient::Researcher => UtilityRouteRecipient::Researcher,
    }
}
fn same_input(a: &UtilityCohortFrozenDecision, b: &UtilityCohortFrozenDecision) -> bool {
    a.input_sha256 == b.input_sha256
        && a.target_sha256 == b.target_sha256
        && a.context_sha256 == b.context_sha256
        && a.target_record_ids == b.target_record_ids
        && a.input_bytes == b.input_bytes
        && a.root_rules_sha256 == b.root_rules_sha256
        && a.original_first_move == b.original_first_move
}
pub(super) fn value_digest<T: Serialize>(
    domain: &str,
    value: &T,
) -> Result<Digest, AdaptiveManagerError> {
    let mut value =
        serde_json::to_value(value).map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
    value.sort_all_objects();
    let bytes =
        serde_json::to_vec(&value).map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
    let mut h = Sha256::new();
    h.update(domain.as_bytes());
    h.update([0]);
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(bytes);
    Ok(Digest(h.finalize().into()))
}
pub(super) fn hex(d: Digest) -> String {
    d.0.iter().map(|b| format!("{b:02x}")).collect()
}
fn string(v: &Value, k: &str) -> Result<String, AdaptiveManagerError> {
    v[k].as_str()
        .filter(|s| s.len() == 64)
        .map(str::to_owned)
        .ok_or_else(|| invalid("missing original source SHA"))
}
pub(super) fn invalid(s: &'static str) -> AdaptiveManagerError {
    AdaptiveManagerError::Invalid(s)
}
pub(super) fn counter_value(c: AdaptiveManagerCounters) -> Value {
    json!({
        "native_preparations":c.native_preparations,"submitted_role_tickets":c.submitted_role_tickets,
        "physical_role_completions":c.physical_role_completions,"completed_nn_inputs":c.completed_nn_inputs,
        "nn_input_count_unknown":c.nn_input_count_unknown,"consumed_role_outputs":c.consumed_role_outputs,
        "accepted_nn_inputs":c.accepted_nn_inputs,"accepted_nn_input_count_unknown":c.accepted_nn_input_count_unknown,
        "consumed_nn_inputs":c.consumed_nn_inputs,"consumed_nn_input_count_unknown":c.consumed_nn_input_count_unknown,
        "zero_nn_outputs_consumed":c.zero_nn_outputs_consumed,"unused_role_results":c.unused_role_results,
        "canceled_preparations":c.canceled_preparations,"cpu_slice_receipts":c.cpu_slice_receipts,
        "cpu_progress_publications":c.cpu_progress_publications,"cpu_yields":c.cpu_yields,
        "cpu_yield_requests":c.cpu_yield_requests,"cpu_yield_acknowledgements":c.cpu_yield_acknowledgements,
        "cpu_natural_completions":c.cpu_natural_completions,"validator_ticket_admissions":c.validator_ticket_admissions,
        "foreign_reports":c.foreign_reports,"foreign_reported_nodes":c.foreign_reported_nodes,
        "foreign_node_count_unknown":c.foreign_node_count_unknown,"foreign_yield_requests":c.foreign_yield_requests,
        "foreign_yield_acknowledgements":c.foreign_yield_acknowledgements,"rejected_events":c.rejected_events,
        "notification_waits":c.notification_waits,
        // Wait-loop scheduling is actual provenance, not a semantic resource allocation.
    })
}

/// Original physical closure and detached source, distinct from durable IO.
/// The same owner records the final pin-zero boundary before packaging this
/// historical checkpoint; no root-local Completed constructor is accepted.
pub struct UtilityTrialPhysicalRetirement {
    shared: Arc<Shared>,
    sequence: u64,
    recipient: UtilityRouteRecipient,
    proof: UtilityCohortRootClosureProof,
    materialized: Option<super::utility_cohort_historical::UtilityTrialMaterializedSource>,
    cost: UtilityCohortCostWire,
    physical_closure: Value,
    ready_prelude: Value,
    original_root: Value,
    checkpoint: Vec<u8>,
    digest: Digest,
    comparable: bool,
    unknown: Option<&'static str>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UtilityCohortCostWire {
    pub domain: String,
    pub scope: String,
    pub objective: String,
    pub cost_class_sha256: String,
    pub cost_epoch_sha256: Option<String>,
    pub resource_plan_sha256: String,
    pub resource_epoch: u64,
    pub clock_epoch: u64,
    pub started_tick: u64,
    pub root_fence_tick: u64,
    pub actor_release_tick: u64,
    pub physical_pin_zero_tick: u64,
    pub whole_wall_ns: u64,
    pub original_deadline_tick: u64,
    pub cohort_action_cutoff_tick: u64,
    pub cohort_cleanup_cutoff_tick: u64,
    pub baseline_counters: Value,
    pub final_counters: Value,
    pub native_credit_baseline: (u64, u64),
    pub native_credit_final: (u64, u64),
    pub cpu_thread_time_ns: Option<u64>,
    pub energy: Option<u64>,
    pub complete_scope: bool,
}
pub struct UtilityTrialRetirementRejection<Q> {
    pub trial: UtilityTrialBoundAdmission<Q>,
    pub proof: UtilityCohortRootClosureProof,
    pub materialized: Option<super::utility_cohort_historical::UtilityTrialMaterializedSource>,
    pub error: AdaptiveManagerError,
}
impl<Q> UtilityTrialBoundAdmission<Q> {
    #[allow(clippy::result_large_err)]
    pub fn finish_physical_retirement<M: rz_search::pals::engine::RoleModel>(
        self,
        proof: UtilityCohortRootClosureProof,
        materialized: Option<super::utility_cohort_historical::UtilityTrialMaterializedSource>,
        original: &rz_search::pals::engine::preparing_root::RetiredResearcherOwner<M>,
    ) -> Result<UtilityTrialPhysicalRetirement, UtilityTrialRetirementRejection<Q>> {
        let checked = (|| {
            let trial = &self.trial;
            let snapshot = &trial.snapshot;
            let window = trial
                .armed
                .ok_or_else(|| invalid("retiring cohort trial has no original window"))?;
            let receipt = proof.receipt();
            let outcome = proof.outcome();
            let known_input = self.input.matches_original_store(original.owner.stores());
            if proof.cohort_id() != trial.cohort_id()
                || proof.trial_sequence() != trial.sequence
                || proof.action() != Some(&self.action)
                || proof.started() != window.started
                || proof.baseline_counters() != snapshot.counters()
                || proof.final_counters() != receipt.counters
                || !receipt.permits_semantic_closure()
                || receipt.authority != snapshot.authority()
                || original.owner.last_authority() != Some(snapshot.authority())
                || original.owner.stores().generation() != snapshot.authority().root.0
                || !known_input
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
                || !self
                    .input
                    .matches_context_cache_release(&original.cache_release)
                || !Arc::ptr_eq(original.owner.source().model(), self.trial.loaded.model())
                || proof.native_credit_baseline().1 != 0
                || proof.native_credit_final() != (receipt.native_cpu_spent, 0)
                || proof.native_credit_baseline().0 > proof.native_credit_final().0
                || proof.root_finished() < window.started
                || proof.release_finished() < proof.root_finished()
            {
                return Err(invalid(
                    "cohort actual root/actor/semantic Store/pin-zero closure differs",
                ));
            }
            if outcome.is_some_and(|o| {
                o.id() != &self.action
                    || !Arc::ptr_eq(o.input(), &self.input)
                    || o.root_budget() != snapshot.root_budget()
            }) {
                return Err(invalid(
                    "cohort physical outcome is not this original bound action",
                ));
            }
            if let Some(m) = &materialized {
                if m.cohort != trial.cohort_id()
                    || m.sequence != trial.sequence
                    || m.action != self.action
                    || m.created_at < proof.root_finished()
                    || m.finished_at > proof.release_finished()
                    || m.wire.input["input_sha256"] != self.input.input_sha256()
                    || m.wire.input["submission_target_sha256"]
                        != self.input.submission_target_sha256()
                {
                    return Err(invalid(
                        "cohort original materialization does not precede release",
                    ));
                }
            }
            let finished = Instant::now();
            if finished < proof.release_finished() {
                return Err(invalid("cohort final cost precedes actual release"));
            }
            let origin = snapshot.clock().origin();
            let tick = |time: Instant| {
                time.checked_duration_since(origin)
                    .and_then(|d| u64::try_from(d.as_nanos()).ok())
                    .ok_or_else(|| invalid("cohort original tick overflow"))
            };
            let wall = finished
                .checked_duration_since(window.started)
                .and_then(|d| u64::try_from(d.as_nanos()).ok())
                .ok_or_else(|| invalid("cohort whole cost overflow"))?;
            let cost = UtilityCohortCostWire {
                domain: UTILITY_COHORT_COST.into(),
                scope: UTILITY_COHORT_COST_SCOPE.into(),
                objective: "exclusive_wall_nanoseconds".into(),
                cost_class_sha256: trial.shared.profile.input.cost_class_sha256.clone(),
                cost_epoch_sha256: Some(hex(self.cost_epoch)),
                resource_plan_sha256: trial.shared.source_pins.resource_plan_sha256.clone(),
                resource_epoch: trial.shared.profile.input.resource_epoch,
                clock_epoch: snapshot.clock().domain().0.0,
                started_tick: tick(window.started)?,
                root_fence_tick: tick(proof.root_finished())?,
                actor_release_tick: tick(proof.release_finished())?,
                physical_pin_zero_tick: tick(finished)?,
                whole_wall_ns: wall,
                original_deadline_tick: snapshot.offers()[trial.selected].original_deadline().at.0,
                cohort_action_cutoff_tick: tick(window.action)?,
                cohort_cleanup_cutoff_tick: tick(window.cleanup)?,
                baseline_counters: counter_value(proof.baseline_counters()),
                final_counters: counter_value(proof.final_counters()),
                native_credit_baseline: proof.native_credit_baseline(),
                native_credit_final: proof.native_credit_final(),
                cpu_thread_time_ns: None,
                energy: None,
                complete_scope: outcome.is_some() && materialized.is_some(),
            };
            let unknown =
                if trial.shared.profile.input.usage == RoventUtilityCohortUsage::ValidationOnly {
                    Some("validation_only")
                } else if proof.reason().is_some() {
                    proof.reason()
                } else if outcome.is_none() {
                    Some("missing_original_semantic_outcome")
                } else if outcome.is_some_and(|o| !o.semantic_consumption_complete()) {
                    Some("missing_original_semantic_consumption")
                } else if materialized.is_none() {
                    Some("missing_bounded_historical_materialization")
                } else if materialized
                    .as_ref()
                    .is_some_and(|m| !m.has_original_witness_bodies())
                {
                    Some("missing_original_conditional_witness_bodies")
                } else if materialized
                    .as_ref()
                    .is_some_and(|m| !m.has_original_witness_basis())
                {
                    Some("missing_original_conditional_witness_basis")
                } else if finished > window.cleanup {
                    Some("cohort_cleanup_cutoff")
                } else if !snapshot.admission().is_first_product_root() {
                    Some("native_product_reused_before_trial")
                } else if !resource_observations_known(snapshot) {
                    Some("unknown_original_resource_observation")
                } else if !self
                    .input
                    .cache_prelude()
                    .is_some_and(|p| p.matches_input(&self.input))
                {
                    Some("missing_actual_cache_prelude")
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
            let physical = physical_closure_value(snapshot, &proof, original, Some(&self.input))?;
            Ok((cost, unknown, physical))
        })();
        let (cost, unknown, physical_closure) = match checked {
            Ok(v) => v,
            Err(error) => {
                return Err(UtilityTrialRetirementRejection {
                    trial: self,
                    proof,
                    materialized,
                    error,
                });
            }
        };
        let mut retired = UtilityTrialPhysicalRetirement {
            shared: Arc::clone(&self.trial.shared),
            sequence: self.trial.sequence,
            recipient: self.trial.recipient(),
            proof,
            materialized,
            cost,
            physical_closure,
            ready_prelude: self.trial.ready_prelude.clone(),
            original_root: self.trial.original_root.clone(),
            checkpoint: Vec::new(),
            digest: Digest([0; 32]),
            comparable: unknown.is_none(),
            unknown,
        };
        // Packaging occurs after the engine-cost boundary. The source bytes
        // were already materialized under the measured interval above.
        let bytes = retired
            .checkpoint_value()
            .and_then(|wire| super::utility_cohort_historical::bounded_json(&wire));
        match bytes {
            Ok(bytes) => {
                retired.digest = bytes_digest(UTILITY_COHORT_TRIAL, &bytes);
                retired.checkpoint = bytes;
                Ok(retired)
            }
            Err(error) => Err(UtilityTrialRetirementRejection {
                trial: self,
                proof: retired.proof,
                materialized: retired.materialized,
                error,
            }),
        }
    }
}
impl UtilityTrialPhysicalRetirement {
    pub fn matches_root_admission(
        &self,
        root: &Arc<crate::rovent_root_admission::RoventRootAdmission>,
    ) -> bool {
        self.shared.state.lock().is_ok_and(|state| {
            self.sequence
                .checked_sub(1)
                .and_then(|index| state.original_roots.get(index as usize))
                .is_some_and(|original| Arc::ptr_eq(original, root))
        })
    }

    pub fn authority(&self) -> SearchAuthority {
        self.proof.receipt().authority
    }
    pub fn support_mask(&self) -> [bool; 7] {
        let mut mask = [false; 7];
        if self.proof.outcome().is_some() {
            mask[self.recipient.output_index()] = true;
        }
        mask
    }

    pub fn cohort_id(&self) -> Digest {
        self.shared.cohort
    }
    pub fn trial_sequence(&self) -> u64 {
        self.sequence
    }
    pub fn recipient(&self) -> UtilityRouteRecipient {
        self.recipient
    }
    pub fn profile(&self) -> &RoventUtilityComparisonCohort {
        &self.shared.profile
    }
    pub fn source_pins(&self) -> &UtilityCohortSourcePins {
        &self.shared.source_pins
    }
    pub fn frozen_decision(
        &self,
    ) -> Result<Option<UtilityCohortFrozenDecision>, AdaptiveManagerError> {
        Ok(self
            .shared
            .state
            .lock()
            .map_err(|_| invalid("cohort owner poisoned"))?
            .frozen
            .clone())
    }
    pub fn materialized(
        &self,
    ) -> Option<&super::utility_cohort_historical::UtilityTrialMaterializedSource> {
        self.materialized.as_ref()
    }
    /// Historical observations only; this JSON cannot mint closure authority.
    pub fn physical_closure(&self) -> &Value {
        &self.physical_closure
    }
    pub fn cost(&self) -> &UtilityCohortCostWire {
        &self.cost
    }
    pub fn comparable(&self) -> bool {
        self.comparable
    }
    pub fn unknown_reason(&self) -> Option<&'static str> {
        self.unknown
    }
    pub fn original_bytes(&self) -> &[u8] {
        &self.checkpoint
    }
    pub fn sha256(&self) -> Digest {
        self.digest
    }
    pub fn semantic(
        &self,
    ) -> Option<&rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteSemanticOutcome>
    {
        self.proof.outcome()
    }
    fn checkpoint_value(&self) -> Result<Value, AdaptiveManagerError> {
        let frozen = self.frozen_decision()?;
        let previous_durable_ack = {
            let state = self
                .shared
                .state
                .lock()
                .map_err(|_| invalid("cohort owner poisoned"))?;
            if self.sequence == 1 {
                None
            } else {
                let prior = self
                    .sequence
                    .checked_sub(2)
                    .and_then(|v| usize::try_from(v).ok())
                    .and_then(|v| state.durable.get(v))
                    .ok_or_else(|| invalid("missing original preceding durable acknowledgement"))?;
                Some(json!({"trial_sequence":prior.trial_sequence(),
                    "trial_sha256":hex(prior.trial_sha256()),"file_sha256":hex(prior.file_sha256())}))
            }
        };
        let unbound_input: Option<Value> = self.proof.historical_input()
            .or_else(||self.proof.unresolved().map(|u|u.input())).map(|input| {
            let text=std::str::from_utf8(input.input_bytes()).map_err(|_|invalid("historical input UTF8"))?;
            Ok::<Value,AdaptiveManagerError>(json!({"reference_json":text,"input_sha256":input.input_sha256(),
                "target_sha256":input.target_sha256(),"submission_target_sha256":input.submission_target_sha256(),
                "context_sha256":input.context_sha256(),"target_record_ids":input.target_record_ids(),
                "original_view_sha256":hex(Digest(input.provenance().view_seal)),
                "original_full_view_sha256":hex(Digest(input.provenance().full_view_digest)),
                "materialized_inside_complete_scope":false}))
        }).transpose()?;

        Ok(
            json!({"schema":UTILITY_COHORT_TRIAL,"cohort_id":hex(self.cohort_id()),
        "trial_sequence":self.sequence,"recipient":self.recipient.as_str(),"profile":self.shared.profile,
        "profile_sha256":hex(self.shared.profile_digest),"source":self.shared.source_pins,
        "frozen_context":frozen,
        "materialized_sha256":self.materialized.as_ref().map(|m|hex(m.sha256())),
        "materialized_json":self.materialized.as_ref().map(|m|std::str::from_utf8(m.original_bytes()).expect("JSON writer emits UTF8")),
        "historical_unbound_input":unbound_input,
        "unresolved_source":self.proof.unresolved().map(|u|json!({"question_sha256":hex(u.id().question()),
            "frame_sequence":u.id().frame().sequence,"recipient":u.id().recipient().as_str(),
            "root_budget_sha256":hex(u.root_budget()),"frozen_context_sha256":hex(u.frozen_context_digest()),
            "reason":u.reason().as_str(),"actual_pc_consumptions":u.pc_consumption().len(),
            "actual_dependencies":u.dependencies().len(),"actual_r_pc_consumptions":u.r_pc_consumption().len(),
            "new_witnesses":u.witnesses().iter().map(|w|json!({"observation":w.observation().0,
                "identity_sha256":hex(w.identity()),"raw_evidence_sha256":hex(w.raw_evidence())})).collect::<Vec<_>>() })),
        "cost":self.cost,"original_root":self.original_root,"ready_prelude":self.ready_prelude,"physical_closure":self.physical_closure,"previous_durable_ack":previous_durable_ack,"resolved":self.proof.outcome().is_some_and(|o|o.semantic_consumption_complete())
            &&self.materialized.is_some(),"comparable":self.comparable,
        "unknown_reason":self.unknown,"support_mask":self.support_mask(),"target_known_mask":[false,false,false,false,false,false,false]}),
        )
    }
}
/// Retains the original verified FD. A raw hash, report or successful return
/// flag cannot stand in for bounded readback and the actual sync operation.
pub struct UtilityCohortDurableAck {
    cohort: Digest,
    sequence: u64,
    trial: Digest,
    file_sha256: Digest,
    started: Instant,
    finished: Instant,
    file: std::fs::File,
}
pub struct UtilityCohortPublicationRejection {
    pub file: std::fs::File,
    pub error: AdaptiveManagerError,
}
impl UtilityCohortDurableAck {
    pub fn started(&self) -> Instant {
        self.started
    }
    pub fn finished(&self) -> Instant {
        self.finished
    }

    #[allow(clippy::result_large_err)]
    pub fn acknowledge_trial_publication(
        mut file: std::fs::File,
        retirement: &UtilityTrialPhysicalRetirement,
    ) -> Result<Self, UtilityCohortPublicationRejection> {
        use std::io::{Read, Seek, SeekFrom};
        let started = Instant::now();
        let checked = (|| {
            let expected = retirement.original_bytes();
            if expected.is_empty()
                || expected.len() > UTILITY_COHORT_HISTORICAL_MAX_BYTES
                || file
                    .metadata()
                    .map_err(|_| invalid("cohort checkpoint metadata unavailable"))?
                    .len()
                    != expected.len() as u64
            {
                return Err(invalid("cohort original checkpoint file length differs"));
            }
            file.seek(SeekFrom::Start(0))
                .map_err(|_| invalid("cohort checkpoint is not seekable"))?;
            // Compare incrementally without a second whole materialization.
            let mut chunk = [0u8; 8192];
            let mut offset = 0;
            while offset < expected.len() {
                let count = (expected.len() - offset).min(chunk.len());
                file.read_exact(&mut chunk[..count])
                    .map_err(|_| invalid("cohort checkpoint bounded read failed"))?;
                if chunk[..count] != expected[offset..offset + count] {
                    return Err(invalid("cohort checkpoint original bytes differ"));
                }
                offset += count;
            }
            let mut extra = [0];
            if file
                .read(&mut extra)
                .map_err(|_| invalid("cohort checkpoint EOF failed"))?
                != 0
            {
                return Err(invalid("cohort checkpoint grew during publication"));
            }
            file.sync_all()
                .map_err(|_| invalid("cohort checkpoint durable sync failed"))?;
            Ok(Digest(Sha256::digest(expected).into()))
        })();
        match checked {
            Ok(file_sha256) => Ok(Self {
                cohort: retirement.cohort_id(),
                sequence: retirement.sequence,
                trial: retirement.digest,
                file_sha256,
                started,
                finished: Instant::now(),
                file,
            }),
            Err(error) => Err(UtilityCohortPublicationRejection { file, error }),
        }
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
pub struct UtilityCohortAckRejection {
    pub ack: Arc<UtilityCohortDurableAck>,
    pub error: AdaptiveManagerError,
}
impl UtilityComparisonCohortOwner {
    #[allow(clippy::result_large_err)]
    pub fn accept_durable_ack(
        &mut self,
        original: &UtilityTrialPhysicalRetirement,
        ack: Arc<UtilityCohortDurableAck>,
    ) -> Result<(), UtilityCohortAckRejection> {
        let checked = (|| {
            let mut state = self
                .shared
                .state
                .lock()
                .map_err(|_| invalid("cohort owner poisoned"))?;
            if !Arc::ptr_eq(&self.shared, &original.shared)
                || ack.cohort != self.shared.cohort
                || ack.sequence != original.sequence
                || ack.trial != original.digest
                || state.active != Some(original.sequence)
                || state.next as u64 + 1 != original.sequence
            {
                return Err(invalid(
                    "cohort durable ACK is foreign, duplicated or out of order",
                ));
            }
            state.durable.push(Arc::clone(&ack));
            state.active = None;
            state.next += 1;
            Ok(())
        })();
        match checked {
            Ok(()) => Ok(()),
            Err(error) => Err(UtilityCohortAckRejection { ack, error }),
        }
    }
}
fn bytes_digest(domain: &str, bytes: &[u8]) -> Digest {
    let mut h = Sha256::new();
    h.update(domain.as_bytes());
    h.update([0]);
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(bytes);
    Digest(h.finalize().into())
}
pub(super) fn resource_observations_known<Q>(snapshot: &UtilityCohortReadySnapshot<Q>) -> bool {
    let known = |r: Option<
        &Result<rz_runtime::adaptive::CpuThreadReadback, rz_runtime::adaptive::CpuResourceError>,
    >| {
        r.is_some_and(|r| {
            r.as_ref().is_ok_and(|r| {
                r.thread_id != 0 && r.affinity.value().is_some_and(|v| !v.is_empty())
            })
        })
    };
    known(snapshot.native_thread_observation())
        && !snapshot.own_thread_observations().is_empty()
        && snapshot
            .own_thread_observations()
            .iter()
            .all(|r| known(r.as_ref()))
}

pub(super) fn resource_observation_value<Q>(snapshot: &UtilityCohortReadySnapshot<Q>) -> Value {
    fn one(
        r: Option<
            &Result<
                rz_runtime::adaptive::CpuThreadReadback,
                rz_runtime::adaptive::CpuResourceError,
            >,
        >,
    ) -> Value {
        match r {
            Some(Ok(r)) => {
                let (method, unknown) = match &r.affinity {
                    rz_runtime::adaptive::CpuReadback::Known { method, .. } => {
                        (Some(format!("{method:?}")), None)
                    }
                    rz_runtime::adaptive::CpuReadback::Unknown(reason) => {
                        (None, Some(format!("{reason:?}")))
                    }
                };
                json!({"actor":format!("{:?}",r.actor),"thread_id_present":r.thread_id!=0,
                    "affinity":r.affinity.value().map(|v|v.iter().map(|c|(c.group,c.index)).collect::<Vec<_>>()),
                    "probe_method":method,"unknown_reason":unknown})
            }
            Some(Err(e)) => json!({"error":format!("{e:?}")}),
            None => Value::Null,
        }
    }
    json!({"native":one(snapshot.native_thread_observation()),
        "own":snapshot.own_thread_observations().iter().map(|r|one(r.as_ref())).collect::<Vec<_>>()})
}

/// An ordered comparison of two original independent trials. Absolute trial
/// authorities remain in their checkpoints; neither an old Completed action
/// nor equal JSON metadata can issue this pair's shared cohort authority.
pub struct UtilityCohortPair {
    left: Arc<UtilityTrialPhysicalRetirement>,
    right: Arc<UtilityTrialPhysicalRetirement>,
    preferred: Option<UtilityRouteRecipient>,
    unknown: Option<&'static str>,
    digest: Digest,
    bytes: Vec<u8>,
}
pub struct UtilityCohortPairRejection {
    pub left: Arc<UtilityTrialPhysicalRetirement>,
    pub right: Arc<UtilityTrialPhysicalRetirement>,
    pub error: AdaptiveManagerError,
}
impl UtilityCohortPair {
    #[allow(clippy::result_large_err)]
    pub fn compare(
        left: Arc<UtilityTrialPhysicalRetirement>,
        right: Arc<UtilityTrialPhysicalRetirement>,
    ) -> Result<Self, UtilityCohortPairRejection> {
        let left_set: BTreeSet<_> = left
            .semantic()
            .into_iter()
            .flat_map(|s| s.witnesses())
            .map(|w| w.identity().0)
            .collect();
        let right_set: BTreeSet<_> = right
            .semantic()
            .into_iter()
            .flat_map(|s| s.witnesses())
            .map(|w| w.identity().0)
            .collect();
        let (preferred, unknown) = if !Arc::ptr_eq(&left.shared, &right.shared) {
            (None, Some("different_registered_cohort"))
        } else if left.sequence == right.sequence
            || left.digest == right.digest
            || left.authority() == right.authority()
        {
            (None, Some("duplicate_independent_trial"))
        } else if left.recipient == right.recipient {
            (None, Some("same_recipient"))
        } else if !left.comparable || !right.comparable {
            (None, Some("unresolved_or_masked_trial"))
        } else if left_set.is_empty() && right_set.is_empty() {
            (None, Some("no_new_valid_witness"))
        } else if left_set == right_set && left.cost.whole_wall_ns == right.cost.whole_wall_ns {
            (None, Some("equal_observation"))
        } else if left_set.is_superset(&right_set)
            && left.cost.whole_wall_ns <= right.cost.whole_wall_ns
        {
            (Some(left.recipient), None)
        } else if right_set.is_superset(&left_set)
            && right.cost.whole_wall_ns <= left.cost.whole_wall_ns
        {
            (Some(right.recipient), None)
        } else {
            (None, Some("pareto_incomparable"))
        };
        let mut known = [false; 7];
        if preferred.is_some() {
            known[left.recipient.output_index()] = true;
            known[right.recipient.output_index()] = true;
        }
        let mut support = left.support_mask();
        for (a, b) in support.iter_mut().zip(right.support_mask()) {
            *a |= b;
        }
        let wire = json!({"schema":UTILITY_COHORT_PAIR,"cohort_id":hex(left.cohort_id()),
            "left_trial_sequence":left.sequence,"right_trial_sequence":right.sequence,
            "left_trial_sha256":hex(left.digest),"right_trial_sha256":hex(right.digest),
            "left_recipient":left.recipient.as_str(),"right_recipient":right.recipient.as_str(),
            "preferred":preferred.map(|p|p.as_str()),"unknown_reason":unknown,
            "support_mask":support,"target_known_mask":known,
            "left_new_witnesses":left_set.iter().map(|v|hex(Digest(*v))).collect::<Vec<_>>(),
            "right_new_witnesses":right_set.iter().map(|v|hex(Digest(*v))).collect::<Vec<_>>(),
            "left_whole_wall_ns":left.cost.whole_wall_ns,"right_whole_wall_ns":right.cost.whole_wall_ns,
            "cost_scope":UTILITY_COHORT_COST_SCOPE,"objective":"exclusive_wall_nanoseconds"});
        match super::utility_cohort_historical::bounded_json(&wire) {
            Ok(bytes) => {
                let digest = bytes_digest(UTILITY_COHORT_PAIR, &bytes);
                Ok(Self {
                    left,
                    right,
                    preferred,
                    unknown,
                    digest,
                    bytes,
                })
            }
            Err(error) => Err(UtilityCohortPairRejection { left, right, error }),
        }
    }
    pub fn left(&self) -> &Arc<UtilityTrialPhysicalRetirement> {
        &self.left
    }
    pub fn right(&self) -> &Arc<UtilityTrialPhysicalRetirement> {
        &self.right
    }
    pub fn preferred(&self) -> Option<UtilityRouteRecipient> {
        self.preferred
    }
    pub fn unknown_reason(&self) -> Option<&'static str> {
        self.unknown
    }
    pub fn sha256(&self) -> Digest {
        self.digest
    }
    pub fn original_bytes(&self) -> &[u8] {
        &self.bytes
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

pub struct UtilityUnboundTrialRetirementRejection<Q> {
    pub trial: UtilityTrialAdmission<Q>,
    pub proof: UtilityCohortRootClosureProof,
    pub error: AdaptiveManagerError,
}
impl<Q> UtilityTrialAdmission<Q> {
    /// Cleanup after a real ready admission, including a prepared input that
    /// never became a live Action. It issues no replacement Action/Bound and
    /// preserves actual source/cost as explicitly unresolved historical facts.
    #[allow(clippy::result_large_err)]
    pub fn retire_unresolved<M: rz_search::pals::engine::RoleModel>(
        self,
        proof: UtilityCohortRootClosureProof,
        original: &rz_search::pals::engine::preparing_root::RetiredResearcherOwner<M>,
    ) -> Result<UtilityTrialPhysicalRetirement, UtilityUnboundTrialRetirementRejection<Q>> {
        let checked = (|| {
            let receipt = proof.receipt();
            let snapshot = &self.snapshot;
            let source = snapshot.offers()[self.selected]
                .semantic_source()
                .ok_or_else(|| invalid("unresolved cohort has no original source owner"))?;
            if proof.cohort_id() != self.cohort_id()
                || proof.trial_sequence() != self.sequence
                || proof.outcome().is_some()
                || proof.action().is_some_and(|action| {
                    action.authority() != snapshot.authority()
                        || action.frame() != snapshot.offers()[self.selected].frame()
                        || action.question() != snapshot.offers()[self.selected].question()
                        || action.recipient() != self.recipient()
                })
                || receipt.authority != snapshot.authority()
                || !receipt.permits_semantic_closure()
                || proof.final_counters() != receipt.counters
                || proof.native_credit_final() != (receipt.native_cpu_spent, 0)
                || proof.baseline_counters() != snapshot.counters()
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
                || proof.root_finished() < proof.started()
                || proof.release_finished() < proof.root_finished()
            {
                return Err(invalid(
                    "unresolved cohort original physical Store/root/source is not closed",
                ));
            }
            if proof
                .historical_input()
                .or_else(|| proof.unresolved().map(|u| u.input()))
                .is_some_and(|input| {
                    !input.matches_original_store(original.owner.stores())
                        || !input.matches_original_model(self.loaded.model())
                        || !input.matches_context_cache_release(&original.cache_release)
                        || !input
                            .root_snapshot()
                            .same_state(snapshot.actual_root().root_snapshot())
                })
            {
                return Err(invalid("unresolved cohort original input/cache differs"));
            }
            let finished = Instant::now();
            let origin = snapshot.clock().origin();
            let tick = |t: Instant| {
                t.checked_duration_since(origin)
                    .and_then(|d| u64::try_from(d.as_nanos()).ok())
                    .ok_or_else(|| invalid("unresolved cohort original clock overflow"))
            };
            let wall = finished
                .checked_duration_since(proof.started())
                .and_then(|d| u64::try_from(d.as_nanos()).ok())
                .ok_or_else(|| invalid("unresolved cohort partial interval overflow"))?;
            let window = self.armed;
            let cost = UtilityCohortCostWire {
                domain: UTILITY_COHORT_COST.into(),
                scope: UTILITY_COHORT_COST_SCOPE.into(),
                objective: "exclusive_wall_nanoseconds".into(),
                cost_class_sha256: self.shared.profile.input.cost_class_sha256.clone(),
                cost_epoch_sha256: None,
                resource_plan_sha256: self.shared.source_pins.resource_plan_sha256.clone(),
                resource_epoch: self.shared.profile.input.resource_epoch,
                clock_epoch: snapshot.clock().domain().0.0,
                started_tick: tick(proof.started())?,
                root_fence_tick: tick(proof.root_finished())?,
                actor_release_tick: tick(proof.release_finished())?,
                physical_pin_zero_tick: tick(finished)?,
                whole_wall_ns: wall,
                original_deadline_tick: snapshot.offers()[self.selected].original_deadline().at.0,
                cohort_action_cutoff_tick: tick(
                    window.map_or(snapshot.hard_deadline(), |w| w.action),
                )?,
                cohort_cleanup_cutoff_tick: tick(
                    window.map_or(snapshot.cleanup_deadline(), |w| w.cleanup),
                )?,
                baseline_counters: counter_value(proof.baseline_counters()),
                final_counters: counter_value(proof.final_counters()),
                native_credit_baseline: proof.native_credit_baseline(),
                native_credit_final: proof.native_credit_final(),
                cpu_thread_time_ns: None,
                energy: None,
                complete_scope: false,
            };
            let input = proof
                .historical_input()
                .or_else(|| proof.unresolved().map(|u| u.input()));
            let physical =
                physical_closure_value(snapshot, &proof, original, input.map(|v| v.as_ref()))?;
            Ok((cost, physical))
        })();
        let (cost, physical_closure) = match checked {
            Ok(c) => c,
            Err(error) => {
                return Err(UtilityUnboundTrialRetirementRejection {
                    trial: self,
                    proof,
                    error,
                });
            }
        };
        let reason = proof.reason().unwrap_or("no_bound_original_action");
        let mut retirement = UtilityTrialPhysicalRetirement {
            shared: Arc::clone(&self.shared),
            sequence: self.sequence,
            recipient: self.recipient(),
            proof,
            materialized: None,
            cost,
            physical_closure,
            ready_prelude: self.ready_prelude.clone(),
            original_root: self.original_root.clone(),
            checkpoint: Vec::new(),
            digest: Digest([0; 32]),
            comparable: false,
            unknown: Some(reason),
        };
        match retirement
            .checkpoint_value()
            .and_then(|v| super::utility_cohort_historical::bounded_json(&v))
        {
            Ok(bytes) => {
                retirement.digest = bytes_digest(UTILITY_COHORT_TRIAL, &bytes);
                retirement.checkpoint = bytes;
                Ok(retirement)
            }
            Err(error) => Err(UtilityUnboundTrialRetirementRejection {
                trial: self,
                proof: retirement.proof,
                error,
            }),
        }
    }
}

/// Projection is issued only after the actual same-owner release and pin-zero
/// checks above. Scalar historical observations do not recreate those caps.
pub(super) fn physical_closure_value<Q, M: rz_search::pals::engine::RoleModel>(
    snapshot: &UtilityCohortReadySnapshot<Q>,
    proof: &UtilityCohortRootClosureProof,
    original: &rz_search::pals::engine::preparing_root::RetiredResearcherOwner<M>,
    input: Option<&NnuecUtilityInputCapture>,
) -> Result<Value, AdaptiveManagerError> {
    let receipt = proof.receipt();
    let release = &original.engine_release;
    let profile = release.profile();
    let provenance = release.provenance();
    let pins = original
        .owner
        .stores()
        .review_pin_count()
        .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
    if pins != 0 {
        return Err(invalid(
            "original review pins reappeared before final retirement",
        ));
    }
    Ok(json!({
        "root_fence_receipt":{
            "authority":{"process_epoch":receipt.authority.epoch.0,"game_generation":receipt.authority.game.0,"root_generation":receipt.authority.root.0,"implementation_sha256":hex(receipt.authority.implementation)},
            "native_owner_joined":receipt.native_owner_joined,
            "native_owner_quiesced":receipt.native_owner_quiesced,
            "native_physical_shutdown_confirmed":receipt.native_physical_shutdown_confirmed,
            "own_cpu_joined":receipt.own_cpu_joined,"expected_own_cpu_owners":receipt.expected_own_cpu_owners,
            "own_cpu_panicked":receipt.own_cpu_panicked,
            "external_shutdown_confirmed":receipt.external_shutdown_confirmed,
            "external_owner_quiesced":receipt.external_owner_quiesced,
            "auxiliary_shutdown_confirmed":receipt.auxiliary_shutdown_confirmed,
            "native_cpu_spent":receipt.native_cpu_spent,"native_cpu_reserved":receipt.native_cpu_reserved,
            "pending_tickets":receipt.pending_tickets,"quarantined":receipt.quarantined,
            "request_sequence_high_water":receipt.request_sequence_high_water,
            "permits_semantic_closure":receipt.permits_semantic_closure()},
        "review_pin_count":pins,
        "engine_release":{
            "owner":release.owner().0,"actual_stack_owner_id":release.actual_stack_owner_id(),
            "profile":{"owner":profile.owner.0,"max_depth":profile.max_depth,
                "evaluation_namespace_sha256":hex(profile.evaluation_namespace),"search_conditions_sha256":hex(profile.search_conditions)},
            "binding_sha256":hex(release.binding_digest()),"model_identity":release.model_identity(),
            "preparation_epoch":release.preparation_epoch(),"release_duration_ns":release.release_time().as_nanos().to_string(),
            "provenance":{"root":provenance.root.0,"generation":provenance.generation,"sequence":provenance.sequence,
                "full_view_sha256":hex(Digest(provenance.full_view_digest)),"view_seal_sha256":hex(Digest(provenance.view_seal)),
                "record_scope_sha256":hex(Digest(provenance.record_scope_digest)),"policy_sha256":hex(Digest(provenance.policy_digest)),
                "model":provenance.model}},
        "context_cache_release":{"epoch":original.cache_release.epoch(),
            "matches_engine_release":release.matches_cache_release(&original.cache_release)},
        "input_cache_prelude":input.and_then(|v|v.cache_prelude().map(|p|json!({"epoch":p.epoch(),"matches_input":p.matches_input(v)}))),
        "native_product":{"original_lifetime_present":snapshot.admission().native_origin().product_lifetime().is_some(),
            "first_product_root":snapshot.admission().is_first_product_root()}
    }))
}

pub(super) fn common_node_value(work: rz_search::pals::rovent_frontier::FrontierNodeWork) -> Value {
    json!({
    "root_entries":work.root_entries,"checked_transitions":work.checked_transitions,
    "native_nodes":work.native_nodes,"native_qsearch_nodes":work.native_qsearch_nodes,
    "reserved_native_nodes":work.reserved_native_nodes,
    "outstanding_native_executions":work.outstanding_native_executions,"native_usage_known":work.native_usage_known})
}

pub(super) fn checked_trial_window(
    started: Instant,
    action_ns: u64,
    cleanup_ns: u64,
    hard: Instant,
    cleanup_hard: Instant,
) -> Result<UtilityTrialWindow, AdaptiveManagerError> {
    if action_ns == 0 || cleanup_ns == 0 {
        return Err(invalid("cohort windows must be finite positive durations"));
    }
    let action = started
        .checked_add(Duration::from_nanos(action_ns))
        .ok_or_else(|| invalid("cohort action cutoff overflow"))?;
    let cleanup = action
        .checked_add(Duration::from_nanos(cleanup_ns))
        .ok_or_else(|| invalid("cohort cleanup cutoff overflow"))?;
    if action > hard || cleanup > cleanup_hard {
        return Err(invalid(
            "registered cohort window does not fit original deadlines",
        ));
    }
    Ok(UtilityTrialWindow {
        started,
        action,
        cleanup,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expected_pc_namespace_excludes_root_authority_and_clock_but_pins_weights_and_backend() {
        let mut original = json!({"model_semantics":"model-v3","model_semantics_sha256":"1".repeat(64),
            "encoding":"encoding-v3","encoding_sha256":"2".repeat(64),
            "encoding_registry_sha256":"3".repeat(64),"model_implementation_sha256":"4".repeat(64),
            "representation_model_domain":"loaded_checkpoint_sha256","frozen_epoch":1,
            "weights_sha256":"5".repeat(64),"export_manifest_sha256":"6".repeat(64),
            "backend_sha256":"7".repeat(64),"process_epoch":1,"actual_start_tick":100,
            "root_authority":{"root":1}});
        let selected = utility_pc_namespace_projection(&original).unwrap();
        let first = value_digest("rz-rovent-utility-cohort-pc-namespace/1", &selected).unwrap();
        original["process_epoch"] = json!(99);
        original["actual_start_tick"] = json!(999);
        original["root_authority"] = json!({"root":99});
        assert_eq!(
            utility_pc_namespace_projection(&original).unwrap(),
            selected
        );
        for name in ["weights_sha256", "backend_sha256", "export_manifest_sha256"] {
            let mut changed = original.clone();
            changed[name] = json!("8".repeat(64));
            assert_ne!(
                value_digest(
                    "rz-rovent-utility-cohort-pc-namespace/1",
                    &utility_pc_namespace_projection(&changed).unwrap()
                )
                .unwrap(),
                first
            );
        }
        original["backend_sha256"] = json!("invalid");
        assert!(utility_pc_namespace_projection(&original).is_err());
        original["backend_sha256"] = json!("7".repeat(64));
        original["frozen_epoch"] = json!(2);
        assert!(utility_pc_namespace_projection(&original).is_err());
    }
    #[test]
    fn registered_window_preserves_exact_original_boundary_without_shrinking() {
        let started = Instant::now();
        let hard = started + Duration::from_nanos(5);
        let cleanup = started + Duration::from_nanos(12);
        let exact =
            checked_trial_window(started, 5, 7, hard, cleanup).expect("exact original ceilings");
        assert_eq!(exact.action_cutoff(), hard);
        assert_eq!(exact.cleanup_cutoff(), cleanup);
        assert!(checked_trial_window(started, 6, 6, hard, cleanup).is_err());
        assert!(checked_trial_window(started, 5, 8, hard, cleanup).is_err());
        assert!(checked_trial_window(started, 0, 12, hard, cleanup).is_err());
    }
    #[test]
    fn canonical_input_allows_independent_trial_order_but_not_changed_target_scope() {
        let a = UtilityCohortFrozenDecision {
            input_sha256: "1".repeat(64),
            target_sha256: "2".repeat(64),
            context_sha256: "3".repeat(64),
            target_record_ids: vec![7, 2, 9],
            input_bytes: 17,
            root_rules_sha256: "4".repeat(64),
            original_first_move: 123,
            first_trial_sequence: 1,
        };
        let mut b = a.clone();
        b.first_trial_sequence = 3;
        assert!(same_input(&a, &b));
        b.target_record_ids.swap(0, 1);
        assert!(!same_input(&a, &b));
        b = a.clone();
        b.original_first_move += 1;
        assert!(!same_input(&a, &b));
        b = a.clone();
        b.input_sha256 = "5".repeat(64);
        assert!(!same_input(&a, &b));
    }
}
