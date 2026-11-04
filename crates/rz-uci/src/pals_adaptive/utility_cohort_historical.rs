//! Bounded historical materialization inside the cohort engine-cost interval.
//! This copies original bytes/facts, never Rules-replays or runs a model. The
//! returned buffer owns no live ReviewView/Bound/native/Store execution lease.
use super::utility_cohort::*;
use rz_contracts::pals::Move16;
use rz_contracts::pals_adaptive::ReviewViewSeal;
use rz_search::pals::coordinator::{AcceptedRoleOrigin, AcceptedRoleOriginKind};
use rz_search::pals::engine::adaptive_port::utility_route::{
    UtilityConditionalWitnessBodies, UtilityRouteSemanticOutcome,
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::io::{self, Write};
use std::time::Instant;

#[derive(Clone, Debug, Serialize)]
pub struct UtilityTrialMaterializedWire {
    pub schema: &'static str,
    pub cohort_id: String,
    pub trial_sequence: u64,
    pub recipient: &'static str,
    pub profile_sha256: String,
    pub source: UtilityCohortSourcePins,
    pub original_root: Value,
    pub prelude: Value,
    pub input: Value,
    pub action: Value,
    pub semantic: Value,
}
/// Only materialize_semantics issues this source. Neither JSON nor a raw hash
/// can be used to reconstruct its original trial/action binding.
pub struct UtilityTrialMaterializedSource {
    pub(super) wire: UtilityTrialMaterializedWire,
    pub(super) bytes: Vec<u8>,
    pub(super) sha256: rz_contracts::Digest,
    pub(super) cohort: rz_contracts::Digest,
    pub(super) sequence: u64,
    pub(super) action: rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteActionId,
    pub(super) created_at: Instant,
    pub(super) finished_at: Instant,
}
impl UtilityTrialMaterializedSource {
    /// A missing proof body stays unknown. A source hash and line alone never
    /// qualify a comparison target; the actual source owner adds those bodies.
    pub fn has_original_witness_bodies(&self) -> bool {
        self.wire
            .semantic
            .get("witness_preimages")
            .and_then(Value::as_array)
            .is_some_and(|b| {
                self.wire.semantic["new_witnesses"]
                    .as_array()
                    .is_some_and(|w| b.len() == w.len())
            })
    }

    /// The separate cohort-only basis carries comparison fields absent from
    /// the legacy raw preimage. Presence is checked only on owner-issued data.
    pub fn has_original_witness_basis(&self) -> bool {
        self.wire
            .semantic
            .get("witness_basis")
            .and_then(Value::as_array)
            .is_some_and(|basis| {
                self.wire.semantic["new_witnesses"]
                    .as_array()
                    .is_some_and(|witnesses| {
                        basis.len() == witnesses.len()
                            && basis.iter().all(|row| {
                                row["basis"].as_object().is_some()
                                    && row["unavailable_reason"].is_null()
                            })
                    })
            })
    }

    pub fn original_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn sha256(&self) -> rz_contracts::Digest {
        self.sha256
    }
    pub fn trial_sequence(&self) -> u64 {
        self.sequence
    }
    pub fn cohort_id(&self) -> rz_contracts::Digest {
        self.cohort
    }
    pub fn wire(&self) -> &UtilityTrialMaterializedWire {
        &self.wire
    }
    pub fn created_at(&self) -> Instant {
        self.created_at
    }
    pub fn finished_at(&self) -> Instant {
        self.finished_at
    }
}
impl<Q> UtilityTrialBoundAdmission<Q> {
    /// The actual manager root fence must have happened before this call;
    /// finalization checks that monotonic ordering against the opaque proof.
    pub fn materialize_semantics(
        &self,
        semantic: &UtilityRouteSemanticOutcome,
    ) -> Result<UtilityTrialMaterializedSource, super::AdaptiveManagerError> {
        self.materialize_original(semantic, None)
    }
    pub fn materialize_semantics_with_witness_bodies(
        &self,
        semantic: &UtilityRouteSemanticOutcome,
        bodies: &UtilityConditionalWitnessBodies,
    ) -> Result<UtilityTrialMaterializedSource, super::AdaptiveManagerError> {
        if !bodies.matches_outcome(semantic) {
            return Err(invalid(
                "conditional witness bodies are not this original outcome",
            ));
        }
        self.materialize_original(semantic, Some(bodies))
    }
    fn materialize_original(
        &self,
        semantic: &UtilityRouteSemanticOutcome,
        bodies: Option<&UtilityConditionalWitnessBodies>,
    ) -> Result<UtilityTrialMaterializedSource, super::AdaptiveManagerError> {
        let created_at = Instant::now();
        let input = semantic.input();
        if semantic.id() != self.action()
            || !std::sync::Arc::ptr_eq(input, self.input())
            || !semantic.semantic_consumption_complete()
            || semantic.root_budget() != self.trial().snapshot().root_budget()
            || semantic.id().authority() != self.trial().snapshot().authority()
        {
            return Err(invalid(
                "cohort materialization source/action/input differs",
            ));
        }
        let trial = self.trial();
        let wire = project_utility_semantics(
            semantic,
            bodies,
            trial.cohort_id(),
            trial.sequence(),
            trial.recipient(),
            trial.profile_digest(),
            trial.source_pins(),
            trial.original_root(),
            trial.ready_prelude(),
            self.cost_epoch(),
        )?;
        let bytes = bounded_json(&wire)?;
        let mut h = Sha256::new();
        h.update(UTILITY_COHORT_MATERIALIZATION.as_bytes());
        h.update([0]);
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(&bytes);
        Ok(UtilityTrialMaterializedSource {
            wire,
            bytes,
            sha256: rz_contracts::Digest(h.finalize().into()),
            cohort: trial.cohort_id(),
            sequence: trial.sequence(),
            action: self.action().clone(),
            created_at,
            finished_at: Instant::now(),
        })
    }
}
/// Byte-preserving input projection shared by the two detached wire domains.
/// Provenance fields are observations; this data never supplies a live Request.
pub(super) fn project_utility_input(
    input: &rz_search::cpu_value::nnuec::v2::utility_capture::NnuecUtilityInputCapture,
) -> Result<Value, super::AdaptiveManagerError> {
    let original = std::str::from_utf8(input.input_bytes())
        .map_err(|_| invalid("utility submitted original input is not UTF8"))?;
    let metadata = input
        .records_metadata()
        .iter()
        .map(|m| {
            let first = m
                .original_first_move
                .map(|v| rz_contracts::Move::try_from(v).map(|v| Move16::encode(v).bits()))
                .transpose()
                .map_err(|e| super::AdaptiveManagerError::Semantic(e.to_string()))?;
            Ok(
                json!({"anchor":m.anchor,"parent":m.parent,"supersedes":m.supersedes,
                "source_state":m.source_state.0,"source_revision":m.source_revision,
                "source_line_offset":m.source_line_offset,"original_first_move":first}),
            )
        })
        .collect::<Result<Vec<_>, super::AdaptiveManagerError>>()?;
    let p = input.provenance();
    Ok(
        json!({"reference_json":original,"bytes":input.input_bytes().len(),
                "input_sha256":input.input_sha256(),"target_sha256":input.target_sha256(),
                "submission_target_sha256":input.submission_target_sha256(),"context_sha256":input.context_sha256(),
                "target_record":input.target_record(),"target_record_ids":input.target_record_ids(),
                "base_identity":input.model_identity(),"root_rules":rules(input.root_snapshot())?,
                "records_metadata":metadata,"source_rules":input.source_snapshots().iter().map(rules).collect::<Result<Vec<_>,super::AdaptiveManagerError>>()?,
                "provenance":{"root_state":p.root.0,"generation":p.generation,"sequence":p.sequence,
                    "view_seal_sha256":hex(rz_contracts::Digest(p.view_seal)),"full_view_sha256":hex(rz_contracts::Digest(p.full_view_digest)),
                    "record_scope_sha256":hex(rz_contracts::Digest(p.record_scope_digest)),"policy_sha256":hex(rz_contracts::Digest(p.policy_digest)),"model":p.model}}),
    )
}

/// Detached projection of the original semantic source. This helper issues no
/// admission, Ready offer, action, physical closure or current publication ACK.
/// Both wire domains reuse the exact witness/ticket/input projection below.
#[allow(clippy::too_many_arguments)]
pub(super) fn project_utility_semantics(
    semantic: &UtilityRouteSemanticOutcome,
    bodies: Option<&UtilityConditionalWitnessBodies>,
    cohort: rz_contracts::Digest,
    sequence: u64,
    recipient: rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteRecipient,
    profile: rz_contracts::Digest,
    source: &UtilityCohortSourcePins,
    original_root: &Value,
    prelude: &Value,
    cost_epoch: rz_contracts::Digest,
) -> Result<UtilityTrialMaterializedWire, super::AdaptiveManagerError> {
    let input = semantic.input();
    let mut baseline: Vec<_> = semantic.baseline_witnesses().map(|d| hex(*d)).collect();
    baseline.sort();
    let witnesses = semantic
            .witnesses()
            .iter()
            .map(|w| {
                Ok(
                    json!({"kind":format!("{:?}",w.kind()),"observation":w.observation().0,
                "identity_sha256":hex(w.identity()),"raw_evidence_sha256":hex(w.raw_evidence()),
                "source_question_sha256":hex(w.source_question()),"source_frame_sequence":w.source_frame().sequence,
                "line":moves(w.line())?}),
                )
            })
            .collect::<Result<Vec<_>, super::AdaptiveManagerError>>()?;
    let witness_preimages = bodies.map(|bodies| bodies.entries().iter().map(|body| {
            let bytes=body.preimage();
            if rz_contracts::Digest(Sha256::digest(bytes).into())!=body.raw_evidence() {
                return Err(invalid("original conditional witness preimage digest differs"));
            }
            Ok(json!({"observation":body.observation().0,"kind":format!("{:?}",body.kind()),
                "identity_sha256":hex(body.identity()),"source_question_sha256":hex(body.source_question()),
                "source_frame_sequence":body.source_frame().sequence,"raw_evidence_sha256":hex(body.raw_evidence()),
                "preimage_bytes":bytes.len(),"preimage_hex":bytes.iter().map(|b|format!("{b:02x}")).collect::<String>()}))
        }).collect::<Result<Vec<_>,super::AdaptiveManagerError>>()).transpose()?;
    let witness_basis = bodies.map(|bodies| bodies.entries().iter().map(|body| {
            let basis = body.basis().map(|basis| {
                let bytes = basis.original_bytes();
                let text = std::str::from_utf8(bytes).map_err(|_|invalid("original conditional basis is not UTF8"))?;
                Ok::<Value,super::AdaptiveManagerError>(json!({"domain":basis.domain(),"sha256":hex(basis.sha256()),
                    "bytes":bytes.len(),"original_json":text}))
            }).transpose()?;
            Ok(json!({"observation":body.observation().0,"kind":format!("{:?}",body.kind()),
                "identity_sha256":hex(body.identity()),"source_question_sha256":hex(body.source_question()),
                "source_frame_sequence":body.source_frame().sequence,"raw_evidence_sha256":hex(body.raw_evidence()),
                "basis":basis,"unavailable_reason":body.basis_unavailable_reason().map(|r|r.as_str())}))
        }).collect::<Result<Vec<_>,super::AdaptiveManagerError>>()).transpose()?;
    let pc: Vec<_> = semantic
        .pc_consumption()
        .iter()
        .map(|p| {
            json!({
            "origin":origin(p.origin()),"question_sha256":hex(p.question()),
            "view":view(p.view()),"full_view_sha256":hex(p.full_view())})
        })
        .collect();
    let dependencies: Vec<_> = semantic
        .dependencies()
        .iter()
        .map(|d| {
            json!({
            "frame_sequence":d.frame().sequence,"question_sha256":hex(d.question()),
            "view":view(d.view()),"full_view_sha256":hex(d.full_view()),
            "parent_frame_sequence":d.coordinator_parent().map(|f|f.sequence),
            "alias":d.peer().map(|(p,_)|peer(p)),"canonical":d.canonical_peer().map(peer),
            "origin":d.peer().and_then(|(_,o)|o).map(|o|origin(&o)),
            "role_cause":d.continuation().and_then(|c|c.role_source()).map(|(r,f,q,v)|json!({
                "request_epoch":r.epoch.0,"request_sequence":r.sequence,"frame_sequence":f.sequence,
                "question_sha256":hex(q),"view_sha256":hex(v)})),
            "cpu_cause":d.continuation().and_then(|c|c.cpu_source()).map(|(j,e,f,q,o)|json!({
                "job_sequence":j.sequence,"native_epoch":e.epoch.0,"native_sequence":e.sequence,
                "frame_sequence":f.sequence,"question_sha256":hex(q),"observation":o.0}))})
        })
        .collect();
    let r:Vec<_>=semantic.r_pc_consumption().iter().map(|w| {
            let job=w.job(); let lineage=w.capture_lineage(); let work=w.attested_slice_work();
            json!({"stage":format!("{:?}",w.stage()),"authority":authority(w.authority()),
                "frame_sequence":w.frame().sequence,"question_sha256":hex(w.question()),
                "submitted_view":view(w.submitted_view()),"submitted_full_view_sha256":hex(w.submitted_full_view_digest()),
                "job":{"authority":authority(job.id.authority),"sequence":job.id.sequence,
                    "state_owner":job.state_identity.owner.0,"state_revision":job.state_identity.revision.0,
                    "state_sha256":hex(job.state_identity.semantic),"question_sha256":hex(job.question),
                    "root_restriction_sha256":hex(job.root_restriction),"evaluation_namespace_sha256":hex(job.evaluation_namespace),
                    "search_conditions_sha256":hex(job.search_conditions),"node_credit":job.total_native_node_credit,
                    "deadline_clock":job.hard_deadline.clock.0.0,"deadline_tick":job.hard_deadline.at.0},
                "owner":w.owner().0,"store_execution":w.store_execution().0,"observation":w.observation().0,
                "native_execution_epoch":w.native_execution().epoch.0,"native_execution_sequence":w.native_execution().sequence,
                "capture_sequence":lineage.capture_sequence(),"binding_sha256":hex(lineage.binding_digest()),
                "capture_full_view_sha256":hex(lineage.full_view_digest()),"capture_view_seal_sha256":hex(lineage.view_seal_digest()),
                "slice_work":{"nodes":work.nodes,"quiescence_nodes":work.quiescence_nodes,"tt_hits":work.tt_hits},
                "selected_pc_ticket":w.selected_pc_ticket().map(ticket),"selected_output_sha256":w.selected_pc_output_digest().map(hex),
                "selected_record":w.selected_record().map(|(r,s)|json!({"revision":r,"origin_state":s.0})),
                "selected_full_view_sha256":w.selected_full_view_digest().map(hex)})
        }).collect();
    let wire = UtilityTrialMaterializedWire {
        schema: UTILITY_COHORT_MATERIALIZATION,
        cohort_id: hex(cohort),
        trial_sequence: sequence,
        recipient: recipient.as_str(),
        profile_sha256: hex(profile),
        source: source.clone(),
        original_root: original_root.clone(),
        prelude: prelude.clone(),
        input: project_utility_input(input)?,
        action: json!({"authority":authority(semantic.id().authority()),"frame_sequence":semantic.id().frame().sequence,
                "question_sha256":hex(semantic.id().question()),"root_budget_sha256":hex(semantic.root_budget()),
                "original_first_move":Move16::encode(rz_contracts::Move::try_from(semantic.original_first_move()).map_err(|e|super::AdaptiveManagerError::Semantic(e.to_string()))?).bits(),
                "deadline_clock":semantic.original_deadline().clock.0.0,"deadline_tick":semantic.original_deadline().at.0,
                "frozen_context_sha256":hex(semantic.frozen_context_digest()),"cost_epoch_sha256":hex(cost_epoch),
                "cost_scope":UTILITY_COHORT_COST_SCOPE}),
        semantic: json!({"baseline_witnesses":baseline,"new_witnesses":witnesses,
                "witness_preimages":witness_preimages,"witness_basis":witness_basis,"pc_consumption":pc,
                "dependencies":dependencies,"r_selected_pc_consumption":r,
                "native_executions":semantic.native_executions().map(|e|json!({"epoch":e.epoch.0,"sequence":e.sequence})).collect::<Vec<_>>() }),
    };
    Ok(wire)
}

pub(super) fn freeze_ready_prelude<Q>(
    snapshot: &super::UtilityCohortReadySnapshot<Q>,
    ready_group: rz_contracts::Digest,
) -> Result<Value, super::AdaptiveManagerError> {
    let offers:Vec<_>=snapshot.offers().iter().map(|o|json!({
            "frame_sequence":o.frame().sequence,"recipient":o.recipient().as_str(),"stage":format!("{:?}",o.stage()),
            "original_first_move":o.original_first_move().bits(),"question_sha256":hex(o.question()),
            "view":view(o.view()),"full_view_sha256":hex(o.full_view()),"public_context_sha256":hex(o.public_context()),
            "semantic_sha256":o.semantic_source().map(|s|hex(s.digest())),
            "budget":{"role_submissions_remaining":o.budget().role_submissions_remaining,
                "original_cpu_credit":o.budget().original_cpu_credit,"original_store_byte_limit":o.budget().original_store_byte_limit},
            "deadline_clock":o.original_deadline().clock.0.0,"deadline_tick":o.original_deadline().at.0})).collect();
    let credit = snapshot.native_credit();
    let captured_tick = snapshot
        .captured_at()
        .checked_duration_since(snapshot.clock().origin())
        .and_then(|d| u64::try_from(d.as_nanos()).ok())
        .ok_or_else(|| invalid("cohort original prelude tick overflow"))?;
    let prelude = json!({"captured_tick":captured_tick,
                "native_credit":{"limit":credit.native_node_limit,"spent":credit.spent_native_nodes,
                    "reserved":credit.reserved_native_nodes,"highest_job_sequence":credit.highest_job_sequence,
                    "highest_execution_sequence":credit.highest_execution_sequence},
                "role_credit":{"spent":snapshot.role_credit().0,"reserved":snapshot.role_credit().1},
                "counters":counter_value(snapshot.counters()),"offers":offers,
                "requested_common_nodes":snapshot.requested_common_nodes(),
                "common_node_work":common_node_value(snapshot.common_node_work()),
                "ready_group_sha256":hex(ready_group),
                "actual_resource_observations":resource_observation_value(snapshot),
                "native_thread_readback":format!("{:?}",snapshot.native_thread_observation()),
                "own_thread_readbacks":format!("{:?}",snapshot.own_thread_observations())});
    // Admission freezes detached metadata while the actual snapshot is live.
    // Existing execution leases never enter this bounded historical object.
    let _bounded = bounded_json(&prelude)?;
    Ok(prelude)
}

pub(super) fn bounded_json<T: Serialize>(v: &T) -> Result<Vec<u8>, super::AdaptiveManagerError> {
    let mut writer = BoundedBytes(Vec::new());
    serde_json::to_writer(&mut writer, v).map_err(|_| {
        invalid("cohort bounded historical serialization failed or exceeded capacity")
    })?;
    Ok(writer.0)
}
struct BoundedBytes(Vec<u8>);
impl Write for BoundedBytes {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self
            .0
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > UTILITY_COHORT_HISTORICAL_MAX_BYTES)
        {
            return Err(io::Error::other("cohort historical byte limit"));
        }
        let needed = self.0.len() + bytes.len();
        if needed > self.0.capacity() {
            let target = needed
                .max(self.0.capacity().saturating_mul(2).max(1024))
                .min(UTILITY_COHORT_HISTORICAL_MAX_BYTES);
            self.0
                .try_reserve_exact(target - self.0.len())
                .map_err(|_| io::Error::other("cohort historical allocation"))?;
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn authority(a: rz_contracts::pals::SearchAuthority) -> Value {
    json!({"process_epoch":a.epoch.0,"game_generation":a.game.0,
    "root_generation":a.root.0,"implementation_sha256":hex(a.implementation)})
}
pub(super) fn rules(
    p: &rz_position::PositionSnapshot,
) -> Result<Value, super::AdaptiveManagerError> {
    let replay = p
        .uci_replay(16_384)
        .map_err(|e| super::AdaptiveManagerError::Semantic(e.to_string()))?;
    Ok(
        json!({"rules_version":p.rules_version(),"variant":p.variant(),
    "semantic_sha256":hex(rz_position::contracts::rules_snapshot_semantic_digest(p)),"fen":p.to_fen(),
    "history_origin":format!("{:?}",p.history_origin()),"history_completeness":format!("{:?}",p.history_completeness()),
    "known_history":p.known_history().map(|f|f.to_fen()).collect::<Vec<_>>(),
    "observer_revision":p.revision(),
    "uci_replay":{"origin":match replay.origin {rz_position::HistoryOrigin::StartPosition=>"start_position",rz_position::HistoryOrigin::Fen=>"fen"},
        "completeness":match replay.completeness {rz_position::HistoryCompleteness::Complete=>"complete",rz_position::HistoryCompleteness::UnknownPrefix=>"unknown_prefix"},
        "start_fen":replay.start_fen,"moves":moves(&replay.moves)?}}),
    )
}
pub(super) fn moves(v: &[rz_position::BoardMove]) -> Result<Vec<u16>, super::AdaptiveManagerError> {
    v.iter()
        .map(|m| {
            rz_contracts::Move::try_from(*m)
                .map(|v| Move16::encode(v).bits())
                .map_err(|e| super::AdaptiveManagerError::Semantic(e.to_string()))
        })
        .collect()
}
fn ticket(t: &rz_contracts::pals_adaptive::RoleTicketEnvelope) -> Value {
    json!({
    "authority":authority(t.authority),"request_epoch":t.id.epoch.0,"request_sequence":t.id.sequence,
    "role":format!("{:?}",t.role),"question_sha256":hex(t.question),"view":view(t.view),
    "representation":{"input":hex(t.representation.input),"history":hex(t.representation.history),
            "candidates":hex(t.representation.candidates),"mask_positions":hex(t.representation.mask_positions),
            "model":hex(t.representation.model),"encoding":hex(t.representation.encoding),
            "precision":format!("{:?}",t.representation.precision),"frozen_epoch":t.representation.frozen_epoch,
            "record_revision":t.representation.record_revision,"role":match t.role {
                rz_contracts::pals::Role::Proposer=>"proposer",rz_contracts::pals::Role::Critic=>"critic",
                rz_contracts::pals::Role::Validator=>"validator"}},"deadline_clock":t.original_deadline.clock.0.0,"deadline_tick":t.original_deadline.at.0})
}
pub(super) fn view(v: ReviewViewSeal) -> Value {
    json!({"authority":authority(v.authority),"sequence":v.view_sequence,
    "context_policy_sha256":hex(v.context_policy),"view_digest":hex(v.view_digest),
    "root_state":{"owner":v.root_state.owner.0,"revision":v.root_state.revision.0,"semantic_sha256":hex(v.root_state.semantic)},
    "actual_tokens":v.actual_tokens,"selected_records":v.selected_records,"anchors":v.anchors,"length_bucket":v.length_bucket})
}
fn origin(o: &AcceptedRoleOrigin) -> Value {
    let (kind, parent) = match o.kind() {
        AcceptedRoleOriginKind::LocalAccepted => ("local_accepted", None),
        AcceptedRoleOriginKind::PeerQuestion(p) => ("peer_question", Some(peer(p))),
    };
    json!({"frame_sequence":o.frame().sequence,"ticket":ticket(o.ticket()),"kind":kind,"peer":parent,
    "output_sha256":hex(o.output_digest()),"seal_sha256":hex(o.seal_digest())})
}
fn peer(p: rz_contracts::rovent::QuestionEnvelope) -> Value {
    json!({
    "authority":authority(p.id.authority),"id":p.id.sequence,"parent":p.parent.map(|v|v.sequence),
    "sender":format!("{:?}",p.sender),"recipient":format!("{:?}",p.recipient),"purpose":format!("{:?}",p.purpose),
    "anchor":{"owner":p.anchor.owner.0,"revision":p.anchor.revision.0,"semantic_sha256":hex(p.anchor.semantic)},
    "anchor_root_ply":p.anchor_root_ply.0,"original_first_move":p.original_first_move.map(|m|m.bits()),
    "prefix_sha256":hex(p.prefix_digest),"line_sha256":p.line_digest.map(hex),"line_offset":p.line_offset.0,
    "question_sha256":hex(p.question_digest),"view":view(p.view),"evidence_revision":p.evidence_revision,
    "root_budget_sha256":hex(p.root_budget),"deadline_clock":p.original_deadline.clock.0.0,"deadline_tick":p.original_deadline.at.0})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_projection_keeps_legacy_materialization_roster_and_exact_bytes() {
        let source = UtilityCohortSourcePins {
            input_lock_sha256: "1".repeat(64),
            source_sha256: "2".repeat(64),
            frozen_nnuec_model_sha256: "3".repeat(64),
            frozen_nnuec_namespace_sha256: "4".repeat(64),
            frozen_pc_model_sha256: "5".repeat(64),
            frozen_pc_namespace_sha256: "6".repeat(64),
            resource_plan_sha256: "7".repeat(64),
            logical_budget_sha256: "8".repeat(64),
        };
        let wire = UtilityTrialMaterializedWire {
            schema: UTILITY_COHORT_MATERIALIZATION,
            cohort_id: "9".repeat(64),
            trial_sequence: 1,
            recipient: "route_p",
            profile_sha256: "a".repeat(64),
            source: source.clone(),
            original_root: json!({"source":"original"}),
            prelude: json!({"source":"ready"}),
            input: json!({"reference_json":"{\"precise\":1}","bytes":13}),
            action: json!({"source":"actual"}),
            semantic: json!({"source":"original witnesses"}),
        };
        // serde struct field order is the original /1 order. Replacing this
        // with a sorted Value, /2 tag or new field would change these bytes.
        let expected = format!(
            "{{\"schema\":\"{}\",\"cohort_id\":\"{}\",\"trial_sequence\":1,\"recipient\":\"route_p\",\"profile_sha256\":\"{}\",\"source\":{},\"original_root\":{{\"source\":\"original\"}},\"prelude\":{{\"source\":\"ready\"}},\"input\":{{\"bytes\":13,\"reference_json\":\"{{\\\"precise\\\":1}}\"}},\"action\":{{\"source\":\"actual\"}},\"semantic\":{{\"source\":\"original witnesses\"}}}}",
            UTILITY_COHORT_MATERIALIZATION,
            "9".repeat(64),
            "a".repeat(64),
            serde_json::to_string(&source).unwrap()
        );
        let bytes = bounded_json(&wire).unwrap();
        assert_eq!(bytes, expected.as_bytes());
        let mut legacy = Sha256::new();
        legacy.update(UTILITY_COHORT_MATERIALIZATION.as_bytes());
        legacy.update([0]);
        legacy.update((bytes.len() as u64).to_le_bytes());
        legacy.update(&bytes);
        let old = rz_contracts::Digest(legacy.finalize().into());
        assert_ne!(
            old,
            super::super::utility_intervention::framed(
                "rz-rovent-utility-cohort-materialization/2",
                &bytes
            )
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes)
                .unwrap()
                .as_object()
                .unwrap()
                .len(),
            11
        );
    }
    #[test]
    fn historical_writer_enforces_total_backing_limit_and_keeps_the_last_valid_bytes() {
        let mut writer = BoundedBytes(Vec::new());
        let chunk = vec![0x55; 1 << 20];
        for _ in 0..32 {
            writer.write_all(&chunk).expect("bounded chunk");
        }
        assert_eq!(writer.0.len(), UTILITY_COHORT_HISTORICAL_MAX_BYTES);
        assert!(writer.0.capacity() <= UTILITY_COHORT_HISTORICAL_MAX_BYTES);
        assert!(writer.write_all(&[0xaa]).is_err());
        assert_eq!(writer.0.len(), UTILITY_COHORT_HISTORICAL_MAX_BYTES);
        assert_eq!(writer.0.last(), Some(&0x55));
    }
}
