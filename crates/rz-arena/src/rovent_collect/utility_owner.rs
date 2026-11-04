//! Original closed utility action publication. Deserializing this historical
//! wire cannot issue a live source, native acceptance, cost or learned target.
//! Root-local /1 stays separate from the opt-in comparison cohort domains.
use super::*;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::ReviewViewSeal;
use rz_search::pals::coordinator::{AcceptedRoleOrigin, AcceptedRoleOriginKind};
use rz_search::pals::engine::adaptive_port::CpuSemanticWitness;
use rz_search::pals::engine::adaptive_port::utility_route::UtilityRouteDependencyWitness;
use rz_uci::pals_adaptive::{
    AdaptiveManagerCounters, CompletedUtilityRouteAction, UtilityRouteComparison,
};
use rz_uci::rovent_bootstrap::nnuec_source::LoadedRoventNnuecSource;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const UTILITY_ROOT_LOCAL_ACTION: &str = "rz-rovent-utility-root-local-action/1";
pub const UTILITY_ROOT_LOCAL_PAIR: &str = "rz-rovent-utility-root-local-pair/1";
pub const UTILITY_PUBLICATION_MAX_BYTES: usize = 32 << 20;

/// Detached wire only. Full historical source/input/causal validation belongs
/// to the separately versioned utility corpus reader; this DTO grants no cap.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UtilityRootLocalActionWireV1 {
    pub schema: String,
    pub collection_sequence: u64,
    pub source: Value,
    pub input: Value,
    pub action: Value,
    pub semantic: Value,
    pub cost: Value,
    pub support_mask: [bool; 7],
    pub target_known_mask: [bool; 7],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UtilityRootLocalPairWireV1 {
    pub schema: String,
    pub collection_sequence: u64,
    pub left_action_sha256: String,
    pub right_action_sha256: String,
    pub original_comparison_sha256: String,
    pub support_mask: [bool; 7],
    pub target_known_mask: [bool; 7],
    pub preferred: Option<String>,
    pub unknown_reason: String,
}

pub struct PublishedUtilityRootLocalAction {
    wire: UtilityRootLocalActionWireV1,
    bytes: Vec<u8>,
    sha256: String,
    original: Arc<CompletedUtilityRouteAction>,
    admission: RoventCollectionRootAdmission,
    source: Arc<LoadedRoventNnuecSource>,
}
impl PublishedUtilityRootLocalAction {
    pub fn wire(&self) -> &UtilityRootLocalActionWireV1 {
        &self.wire
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn original(&self) -> &Arc<CompletedUtilityRouteAction> {
        &self.original
    }
    pub fn admission(&self) -> &RoventCollectionRootAdmission {
        &self.admission
    }
    pub fn nnuec_source(&self) -> &Arc<LoadedRoventNnuecSource> {
        &self.source
    }
}
pub struct PublishedUtilityRootLocalPair {
    wire: UtilityRootLocalPairWireV1,
    bytes: Vec<u8>,
    sha256: String,
}
impl PublishedUtilityRootLocalPair {
    pub fn wire(&self) -> &UtilityRootLocalPairWireV1 {
        &self.wire
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

/// Run only after the Session final release/cache ACK callback. An error leaves
/// the borrowed original cap, source and admission in the caller's custody.
pub fn publish_utility_root_local_action(
    actual: &Arc<CompletedUtilityRouteAction>,
    admission: &RoventCollectionRootAdmission,
    source: &Arc<LoadedRoventNnuecSource>,
    collection_sequence: u64,
) -> Result<PublishedUtilityRootLocalAction, ArenaError> {
    let semantic = actual.semantic();
    let input = semantic.input();
    let p = input.provenance();
    check(
        collection_sequence != 0
            && admission.validates_registration(
                actual.id().authority(),
                semantic.mode(),
                semantic.root_budget(),
                semantic.original_deadline(),
            )
            && source.input_lock_digest() == admission.root_admission().startup_lock_digest()
            && input.matches_original_model(source.model())
            && input.model_identity() == source.semantic_weights_identity()
            && p.model == *source.semantic_weights_identity()
            && p.generation == actual.id().authority().root.0
            && input
                .root_snapshot()
                .same_state(admission.rules().rules().snapshot()),
        "utility actual cap differs from original root/loaded source",
    )?;
    let reference_json = std::str::from_utf8(input.input_bytes())
        .map_err(|_| invalid("utility original Reference is not UTF8"))?
        .to_owned();
    // Preserve the original optional move; encoding failure is an error rather
    // than a JSON Result that could look like successful historical metadata.
    let records: Vec<_> = input
        .records_metadata()
        .iter()
        .map(|r| {
            let first = r
                .original_first_move
                .map(|m| Move16::pack(m).map(|m| m.bits()))
                .transpose()
                .map_err(|e| invalid(e.to_string()))?;
            Ok(
                json!({"anchor":r.anchor,"parent":r.parent,"supersedes":r.supersedes,
            "source_state":r.source_state.0,"source_revision":r.source_revision,
            "source_line_offset":r.source_line_offset,"original_first_move":first}),
            )
        })
        .collect::<Result<_, ArenaError>>()?;
    let mut baseline: Vec<_> = semantic.baseline_witnesses().map(|v| hex(*v)).collect();
    baseline.sort();
    let witnesses = semantic.witnesses().iter().map(|w| Ok(json!({
        "kind":match w.kind() { rz_search::pals::engine::adaptive_port::validator_port::ActionWitnessKind::Counter => "counter",
            rz_search::pals::engine::adaptive_port::validator_port::ActionWitnessKind::Repair => "repair" },
        "observation":w.observation().0,"identity_sha256":hex(w.identity()),
        "raw_evidence_sha256":hex(w.raw_evidence()),"line":packed(w.line())?
    }))).collect::<Result<Vec<_>,ArenaError>>()?;
    let pc: Vec<_> = semantic
        .pc_consumption()
        .iter()
        .map(|p| {
            json!({
        "origin":origin_json(p.origin()),"question_sha256":hex(p.question()),
        "view":view_json(p.view()),"full_view_sha256":hex(p.full_view())})
        })
        .collect();
    let r_pc: Vec<_> = semantic
        .r_pc_consumption()
        .iter()
        .map(r_pc_json)
        .collect::<Result<_, _>>()?;
    let dependencies: Vec<_> = semantic
        .dependencies()
        .iter()
        .map(dependency_json)
        .collect();
    let budget = semantic.original_budget();
    let cost = actual.cost();
    let wall_ns = u64::try_from(cost.wall_cost().as_nanos())
        .map_err(|_| invalid("utility whole wall cost exceeds u64 nanoseconds"))?;
    let wire = UtilityRootLocalActionWireV1 {
        schema: UTILITY_ROOT_LOCAL_ACTION.into(),
        collection_sequence,
        source: json!({"root_admission_sha256":admission.raw_sha256(),
            "root_admission":admission.raw_receipt(),"native_origin_sha256":admission.native_origin().raw_sha256(),
            "input_lock_sha256":hex(source.input_lock_digest()),"nnuec_registration":source.registration(),
            "weights_artifact":source.weights_artifact(),"provenance_artifact":source.provenance_artifact()}),
        input: json!({"reference_json":reference_json,"bytes":input.input_bytes().len(),
            "input_sha256":input.input_sha256(),"target_sha256":input.target_sha256(),
            "submission_target_sha256":input.submission_target_sha256(),"context_sha256":input.context_sha256(),
            "target_record":input.target_record(),"target_record_ids":input.target_record_ids(),
            "base_identity":input.model_identity(),"root_rules":rules_json(input.root_snapshot()),
            "records_metadata":records,"source_rules":input.source_snapshots().iter().map(rules_json).collect::<Vec<_>>(),
            "provenance":{"root_state":p.root.0,"generation":p.generation,"sequence":p.sequence,
                "full_view_sha256":hex(Digest(p.full_view_digest)),"view_seal_sha256":hex(Digest(p.view_seal)),
                "record_scope_sha256":hex(Digest(p.record_scope_digest)),"policy_sha256":hex(Digest(p.policy_digest)),
                "model":p.model}}),
        action: json!({"authority":authority_json(actual.id().authority()),"frame_sequence":actual.id().frame().sequence,
            "question_sha256":hex(actual.id().question()),"recipient":actual.id().recipient().as_str(),
            "output_index":actual.id().recipient().output_index(),"manager_sequence":cost.sequence(),
            "original_first_move":Move16::pack(semantic.original_first_move()).map_err(|e|invalid(e.to_string()))?.bits(),
            "root_budget_sha256":hex(semantic.root_budget()),"original_deadline":{"clock":semantic.original_deadline().clock.0.0,
                "tick":semantic.original_deadline().at.0},"mode":format!("{:?}",semantic.mode()),
            "budget":{"role_submissions_remaining":budget.role_submissions_remaining,
                "original_cpu_credit":budget.original_cpu_credit,"original_store_byte_limit":budget.original_store_byte_limit},
            "frozen_context_sha256":hex(semantic.frozen_context_digest()),"comparison_group_sha256":hex(actual.comparison_group()),
            "original_action_sha256":hex(actual.digest())}),
        semantic: json!({"baseline_witnesses":baseline,"new_witnesses":witnesses,"pc_consumption":pc,
            "r_selected_pc_consumption":r_pc,"dependencies":dependencies,
            "native_executions":semantic.native_executions().map(|e|json!({"epoch":e.epoch.0,"sequence":e.sequence})).collect::<Vec<_>>() }),
        cost: json!({"scope":cost.cost_scope(),"objective":cost.objective(),
            "resource_profile_sha256":hex(cost.resource_profile()),"cost_epoch_sha256":hex(cost.cost_epoch()),
            "whole_wall_ns":wall_ns,"baseline_counters":counters_json(cost.baseline_counters()),
            "final_counters":counters_json(cost.final_counters()),
            "native_credit_baseline":{"spent":cost.native_credit_baseline().0,"reserved":cost.native_credit_baseline().1},
            "native_credit_final":{"spent":cost.native_credit_final().0,"reserved":cost.native_credit_final().1},
            "cpu_thread_time_ns":Value::Null,"energy":Value::Null,"retained_host_bytes":Value::Null}),
        support_mask: actual.support_mask(),
        target_known_mask: [false; 7],
    };
    let (bytes, sha256) = sealed_bytes(UTILITY_ROOT_LOCAL_ACTION, &wire)?;
    Ok(PublishedUtilityRootLocalAction {
        wire,
        bytes,
        sha256,
        original: Arc::clone(actual),
        admission: admission.clone(),
        source: Arc::clone(source),
    })
}

/// Root-local source /1 has one action per root and no comparison-cohort cap.
/// Preserve the actual comparison observation, while keeping every target mask
/// false. A new cohort domain is required for usable known pairs.
pub fn publish_utility_root_local_pair(
    pair: &UtilityRouteComparison,
    left: &PublishedUtilityRootLocalAction,
    right: &PublishedUtilityRootLocalAction,
    collection_sequence: u64,
) -> Result<PublishedUtilityRootLocalPair, ArenaError> {
    check(
        collection_sequence != 0
            && Arc::ptr_eq(pair.left(), &left.original)
            && Arc::ptr_eq(pair.right(), &right.original),
        "utility pair original caps differ",
    )?;
    let wire = UtilityRootLocalPairWireV1 {
        schema: UTILITY_ROOT_LOCAL_PAIR.into(),
        collection_sequence,
        left_action_sha256: left.sha256.clone(),
        right_action_sha256: right.sha256.clone(),
        original_comparison_sha256: hex(pair.digest()),
        support_mask: pair.support_mask(),
        target_known_mask: [false; 7],
        preferred: None,
        unknown_reason: pair
            .unknown_reason()
            .map(|r| r.as_str())
            .unwrap_or("missing_comparison_cohort")
            .into(),
    };
    let (bytes, sha256) = sealed_bytes(UTILITY_ROOT_LOCAL_PAIR, &wire)?;
    Ok(PublishedUtilityRootLocalPair {
        wire,
        bytes,
        sha256,
    })
}
fn sealed_bytes<T: Serialize>(domain: &str, wire: &T) -> Result<(Vec<u8>, String), ArenaError> {
    let bytes = serde_json::to_vec(wire).map_err(|e| invalid(e.to_string()))?;
    check(
        !bytes.is_empty() && bytes.len() <= UTILITY_PUBLICATION_MAX_BYTES,
        "utility publication byte limit",
    )?;
    let mut h = Sha256::new();
    h.update(domain.as_bytes());
    h.update([0]);
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(&bytes);
    Ok((bytes, format!("{:x}", h.finalize())))
}
fn authority_json(a: SearchAuthority) -> Value {
    json!({"process_epoch":a.epoch.0,"game_generation":a.game.0,
    "root_generation":a.root.0,"implementation_sha256":hex(a.implementation)})
}
fn view_json(v: ReviewViewSeal) -> Value {
    json!({"authority":authority_json(v.authority),"sequence":v.view_sequence,
    "context_policy_sha256":hex(v.context_policy),"view_digest":hex(v.view_digest),
    "root_state":{"owner":v.root_state.owner.0,"revision":v.root_state.revision.0,"semantic_sha256":hex(v.root_state.semantic)},
    "actual_tokens":v.actual_tokens,"selected_records":v.selected_records,"anchors":v.anchors,"length_bucket":v.length_bucket})
}
fn origin_json(o: &AcceptedRoleOrigin) -> Value {
    let (kind, peer) = match o.kind() {
        AcceptedRoleOriginKind::LocalAccepted => ("local_accepted", Value::Null),
        AcceptedRoleOriginKind::PeerQuestion(p) => ("peer_question", peer_json(p)),
    };
    json!({"frame_sequence":o.frame().sequence,"ticket":ticket_json(o.ticket()),"output_sha256":hex(o.output_digest()),
        "kind":kind,"peer":peer,"seal_sha256":hex(o.seal_digest())})
}
fn peer_json(p: rz_contracts::rovent::QuestionEnvelope) -> Value {
    json!({"authority":authority_json(p.id.authority),
    "id":p.id.sequence,"parent":p.parent.map(|v|v.sequence),"sender":format!("{:?}",p.sender),
    "recipient":format!("{:?}",p.recipient),"purpose":format!("{:?}",p.purpose),
    "anchor":{"owner":p.anchor.owner.0,"revision":p.anchor.revision.0,"semantic_sha256":hex(p.anchor.semantic)},
    "anchor_root_ply":p.anchor_root_ply.0,"original_first_move":p.original_first_move.map(|m|m.bits()),
    "prefix_sha256":hex(p.prefix_digest),"line_sha256":p.line_digest.map(hex),"line_offset":p.line_offset.0,
    "question_sha256":hex(p.question_digest),"view":view_json(p.view),"evidence_revision":p.evidence_revision,
    "root_budget_sha256":hex(p.root_budget),"deadline":{"clock":p.original_deadline.clock.0.0,"tick":p.original_deadline.at.0}})
}
fn rules_json(p: &rz_position::PositionSnapshot) -> Value {
    json!({"rules_version":p.rules_version(),"variant":p.variant(),
    "semantic_sha256":hex(rz_position::contracts::rules_snapshot_semantic_digest(p)),"fen":p.to_fen(),
    "history_origin":format!("{:?}",p.history_origin()),"history_completeness":format!("{:?}",p.history_completeness()),
    "known_history":p.known_history().map(|v|v.to_fen()).collect::<Vec<_>>()})
}
fn dependency_json(d: &UtilityRouteDependencyWitness) -> Value {
    json!({"frame_sequence":d.frame().sequence,
    "question_sha256":hex(d.question()),"view":view_json(d.view()),"full_view_sha256":hex(d.full_view()),
    "parent_frame_sequence":d.coordinator_parent().map(|f|f.sequence),
    "alias":d.peer().map(|(p,_)|peer_json(p)),"canonical":d.canonical_peer().map(peer_json),
    "source_origin":d.peer().and_then(|(_,o)|o).map(|o|origin_json(&o)),
    "role_cause":d.continuation().and_then(|c|c.role_source()).map(|(r,f,q,v)|json!({"request_epoch":r.epoch.0,
        "request_sequence":r.sequence,"frame_sequence":f.sequence,"question_sha256":hex(q),"view_sha256":hex(v)})),
    "cpu_cause":d.continuation().and_then(|c|c.cpu_source()).map(|(j,e,f,q,o)|json!({"job_sequence":j.sequence,
        "native_execution_epoch":e.epoch.0,"native_execution_sequence":e.sequence,"frame_sequence":f.sequence,
        "question_sha256":hex(q),"observation":o.0}))})
}
fn r_pc_json(w: &CpuSemanticWitness) -> Result<Value, ArenaError> {
    let job = w.job();
    let l = w.capture_lineage();
    let native = w.native_execution();
    let work = w.attested_slice_work();
    Ok(
        json!({"stage":format!("{:?}",w.stage()),"authority":authority_json(w.authority()),
        "frame_sequence":w.frame().sequence,"question_sha256":hex(w.question()),"submitted_view":view_json(w.submitted_view()),
        "submitted_full_view_sha256":hex(w.submitted_full_view_digest()),"job":{"authority":authority_json(job.id.authority),"job_sequence":job.id.sequence,
            "state_owner":job.state_identity.owner.0,"state_revision":job.state_identity.revision.0,
            "state_semantic_sha256":hex(job.state_identity.semantic),"question_sha256":hex(job.question),
            "root_restriction_sha256":hex(job.root_restriction),"evaluation_namespace_sha256":hex(job.evaluation_namespace),
            "search_conditions_sha256":hex(job.search_conditions),"total_native_node_credit":job.total_native_node_credit,
            "hard_deadline_clock_epoch":job.hard_deadline.clock.0.0,"hard_deadline_ns":job.hard_deadline.at.0},
        "owner":w.owner().0,"task_key":serde_json::to_value(w.task_key()).map_err(|e|invalid(e.to_string()))?,
        "store_execution":w.store_execution().0,"native_execution":{"epoch":native.epoch.0,"sequence":native.sequence},
        "observation":w.observation().0,"capture_sequence":l.capture_sequence(),"binding_sha256":hex(l.binding_digest()),
        "capture_full_view_sha256":hex(l.full_view_digest()),"capture_view_seal_sha256":hex(l.view_seal_digest()),
        "attested_slice_work":{"nodes":work.nodes,"quiescence_nodes":work.quiescence_nodes,"tt_hits":work.tt_hits},
        "selected_pc_ticket":w.selected_pc_ticket().map(ticket_json),"selected_output_sha256":w.selected_pc_output_digest().map(hex),
        "selected_record":w.selected_record().map(|(r,s)|json!({"revision":r,"origin_state":s.0})),
        "selected_full_view_sha256":w.selected_full_view_digest().map(hex)}),
    )
}
fn counters_json(c: AdaptiveManagerCounters) -> Value {
    json!({"native_preparations":c.native_preparations,
    "submitted_role_tickets":c.submitted_role_tickets,"physical_role_completions":c.physical_role_completions,
    "completed_nn_inputs":c.completed_nn_inputs,"nn_input_count_unknown":c.nn_input_count_unknown,
    "consumed_role_outputs":c.consumed_role_outputs,"accepted_nn_inputs":c.accepted_nn_inputs,
    "accepted_nn_input_count_unknown":c.accepted_nn_input_count_unknown,"consumed_nn_inputs":c.consumed_nn_inputs,
    "consumed_nn_input_count_unknown":c.consumed_nn_input_count_unknown,"zero_nn_outputs_consumed":c.zero_nn_outputs_consumed,
    "unused_role_results":c.unused_role_results,"canceled_preparations":c.canceled_preparations,"cpu_slice_receipts":c.cpu_slice_receipts,
    "cpu_progress_publications":c.cpu_progress_publications,"cpu_yields":c.cpu_yields,"cpu_yield_requests":c.cpu_yield_requests,
    "cpu_yield_acknowledgements":c.cpu_yield_acknowledgements,"cpu_natural_completions":c.cpu_natural_completions,
    "validator_ticket_admissions":c.validator_ticket_admissions,"foreign_reports":c.foreign_reports,
    "foreign_reported_nodes":c.foreign_reported_nodes,"foreign_node_count_unknown":c.foreign_node_count_unknown,
    "foreign_yield_requests":c.foreign_yield_requests,"foreign_yield_acknowledgements":c.foreign_yield_acknowledgements,
    "rejected_events":c.rejected_events,"notification_waits":c.notification_waits})
}
