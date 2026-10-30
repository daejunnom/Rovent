//! Detached /2 history. Projection is shared with /1, while original admission
//! and retirement capabilities remain separate. Reading bytes grants no work.
use super::AdaptiveManagerError;
use super::utility_cohort::{hex, invalid};
use super::utility_cohort_historical::{
    bounded_json, project_utility_input, project_utility_semantics,
};
use super::utility_intervention::{UtilityInterventionTrialAdmission, framed};
use rz_contracts::Digest;
use rz_search::pals::coordinator::CoordinatorWork;
use rz_search::pals::engine::adaptive_port::{
    AdaptivePortQuestion,
    utility_route::{
        UtilityConditionalWitnessBodies, UtilityInterventionTarget, UtilityRouteAction,
        UtilityRouteSemanticOutcome, utility_intervention_target_digest_preimage,
    },
};
use rz_search::pals::review_context::ReviewView;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Instant;

pub const UTILITY_INTERVENTION_MATERIALIZATION: &str = "rz-rovent-utility-cohort-materialization/2";
pub(super) fn neutral_target<Q>(
    target: &UtilityInterventionTarget<Q>,
) -> Result<Value, AdaptiveManagerError> {
    let capture = target.input_capture();
    let input = project_utility_input(capture)?;
    let p = utility_intervention_target_digest_preimage(capture)
        .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
    Ok(json!({"domain":"rz-rovent-utility-intervention-target/1",
        "input_sha256":capture.input_sha256(),"target_sha256":capture.target_sha256(),"context_sha256":capture.context_sha256(),
        "target_record":capture.target_record(),"target_record_ids":capture.target_record_ids(),"input_bytes":capture.input_bytes().len(),
        "reference_json":input["reference_json"],"root_snapshot_sha256":hex(p.root_snapshot_digest),
        "source_snapshot_sha256":p.source_snapshot_digests.into_iter().map(hex).collect::<Vec<_>>(),"record_scope_sha256":hex(p.record_scope_digest),
        "model_identity":capture.model_identity(),"original_first_move":target.first_move().bits(),
        "root_rules":input["root_rules"],"records_metadata":input["records_metadata"],"source_rules":input["source_rules"],
        "original_view_sha256":input["provenance"]["view_seal_sha256"],"original_full_view_sha256":input["provenance"]["full_view_sha256"]}))
}
pub(super) fn arm_query(
    action: &UtilityRouteAction,
    work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
) -> Result<Value, AdaptiveManagerError> {
    let source = action
        .intervention_source_work()
        .ok_or_else(|| invalid("intervention original source absent"))?;
    let neutral = action
        .intervention_neutral_target_digest()
        .ok_or_else(|| invalid("intervention neutral digest absent"))?;
    let query = work.input.view.query();
    let question=rz_search::pals::engine::adaptive_port::utility_route::utility_intervention_arm_question_preimage(work).map_err(|e|AdaptiveManagerError::Semantic(e.to_string()))?;
    let moves = super::utility_cohort_historical::moves;
    let rules = super::utility_cohort_historical::rules;
    let prepared = action
        .intervention_arm_capture()
        .map(|input| project_utility_input(input))
        .transpose()?;
    Ok(
        json!({"neutral_target_sha256":hex(neutral),"arm_query_sha256":hex(work.input.question_digest),"recipient":action.id().recipient().as_str(),
        "frame_sequence":work.frame.sequence,"source_frame_sequence":source.frame.sequence,"stage":format!("{:?}",work.input.stage),
        "purpose":format!("{:?}",query.purpose),"original_first_move":work.original_first_move.bits(),
        "view":super::utility_cohort_historical::view(work.input.view_seal),"full_view_sha256":hex(Digest(work.input.view.digest())),
        "public_context_sha256":hex(work.input.public_context),
        "budget":{"role_submissions_remaining":work.budget.role_submissions_remaining,"original_cpu_credit":work.budget.original_cpu_credit,"original_store_byte_limit":work.budget.original_store_byte_limit},
        "deadline_clock":work.original_deadline.clock.0.0,"deadline_tick":work.original_deadline.at.0,
        "query":{"endpoint_rules":rules(&work.input.question.position.snapshot())?,"prefix":moves(&query.prefix)?,"proposal":moves(&query.proposal)?,
            "counterexample":query.counterexample.as_deref().map(moves).transpose()?,"legal":moves(&query.legal)?,"divergences":query.divergences,
            "question_preimage":{"sequence":question.sequence,"stage_code":question.stage_code,"view_digest":hex(question.view_digest),"view_seal_digest":hex(question.view_seal_digest),"focus_prefix_sha256":hex(question.focus_prefix_sha256),"goal_local_sha256":question.goal_local_sha256.map(hex),"legacy_full_records_sha256":question.legacy_full_records_sha256.map(hex)}},
        "prepared_input":prepared}),
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UtilityInterventionMaterializedWire {
    pub schema: String,
    pub cohort_id: String,
    pub trial_sequence: u64,
    pub recipient: String,
    pub profile_sha256: String,
    pub source: super::UtilityCohortSourcePins,
    pub original_root: Value,
    pub preparation: Value,
    pub neutral_target: Value,
    pub arm_query: Value,
    pub action: Value,
    pub semantic: Value,
}
/// Only the original trial/action join can issue this materialized source.
pub struct UtilityInterventionMaterializedSource {
    wire: UtilityInterventionMaterializedWire,
    bytes: Vec<u8>,
    digest: Digest,
    pub(super) created_at: Instant,
    pub(super) finished_at: Instant,
}
impl UtilityInterventionMaterializedSource {
    pub(super) fn from_actual<Q>(
        trial: &UtilityInterventionTrialAdmission<Q>,
        outcome: &UtilityRouteSemanticOutcome,
        bodies: &UtilityConditionalWitnessBodies,
        cost: Digest,
    ) -> Result<Self, AdaptiveManagerError> {
        let created_at = Instant::now();
        let (neutral, neutral_wire, input) = trial
            .neutral()
            .ok_or_else(|| invalid("intervention materialization neutral source missing"))?;
        if trial.action() != Some(outcome.id())
            || !std::sync::Arc::ptr_eq(input, outcome.input())
            || outcome.frozen_context_digest() != *neutral
            || !outcome.semantic_consumption_complete()
            || !bodies.matches_outcome(outcome)
        {
            return Err(invalid(
                "intervention materialization actual action/input/semantic source differs",
            ));
        }
        let arm = trial
            .arm_query()
            .ok_or_else(|| invalid("intervention arm query missing"))?;
        if arm["prepared_input"].is_null() {
            return Err(invalid("intervention original arm preparation unavailable"));
        }
        let projection = project_utility_semantics(
            outcome,
            Some(bodies),
            trial.cohort_id(),
            trial.sequence(),
            trial.recipient(),
            trial.profile_digest(),
            trial.source_pins(),
            trial.original_root(),
            trial.preparation(),
            cost,
        )?;
        let wire = UtilityInterventionMaterializedWire {
            schema: UTILITY_INTERVENTION_MATERIALIZATION.into(),
            cohort_id: hex(trial.cohort_id()),
            trial_sequence: trial.sequence(),
            recipient: trial.recipient().as_str().into(),
            profile_sha256: hex(trial.profile_digest()),
            source: trial.source_pins().clone(),
            original_root: trial.original_root().clone(),
            preparation: trial.preparation().clone(),
            neutral_target: neutral_wire.clone(),
            arm_query: arm.clone(),
            action: projection.action,
            semantic: projection.semantic,
        };
        let bytes = bounded_json(&wire)?;
        let digest = framed(UTILITY_INTERVENTION_MATERIALIZATION, &bytes);
        Ok(Self {
            wire,
            bytes,
            digest,
            created_at,
            finished_at: Instant::now(),
        })
    }
    pub fn wire(&self) -> &UtilityInterventionMaterializedWire {
        &self.wire
    }
    pub fn original_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn sha256(&self) -> Digest {
        self.digest
    }
    pub fn created_at(&self) -> Instant {
        self.created_at
    }
    pub fn finished_at(&self) -> Instant {
        self.finished_at
    }
    pub fn has_original_witness_basis(&self) -> bool {
        let s = &self.wire.semantic;
        s["new_witnesses"].as_array().is_some_and(|w| {
            s["witness_preimages"]
                .as_array()
                .is_some_and(|p| p.len() == w.len())
                && s["witness_basis"].as_array().is_some_and(|b| {
                    b.len() == w.len()
                        && b.iter().all(|row| {
                            row["basis"].is_object() && row["unavailable_reason"].is_null()
                        })
                })
        })
    }
}
