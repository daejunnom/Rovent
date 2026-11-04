//! 실제 opaque P/C witness의 보존과 역사적 무결성 검사다.
//! JSON·해시 재계산은 과거 native/Store/physical 권한을 발급하지 않는다.
//! 미래 counter는 원 초기 AttackRanking capability와 실제 두 retired row가
//! 함께 보존된 별도 variant에서만 연결하며 과거 입력에는 삽입하지 않는다.
use super::*;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::pals_adaptive::ReviewViewSeal;
use rz_contracts::rovent::{QuestionEnvelope, ReciprocalQuestionId};
use rz_experiments::rovent_corpus_v2::{
    ROVENT_DIVERGENCE_TARGET_DOMAIN, RoventCorpusManifestV2, RoventDivergenceTargetV1,
    RoventTrainingRecordV2, parse_rovent_visible_input_json,
};
use rz_search::pals::coordinator::{AcceptedRoleOrigin, AcceptedRoleOriginKind, FreshPairSeal};
use rz_search::pals::engine::pc_port::{
    PcAcceptedFreshEndpoint, PcAttackRankedSource, PcDivergenceResolution, PcDivergenceWitness,
    PcRepairAcceptedLine,
};
use serde::Deserialize;

const WITNESS_DOMAIN: &str = "rz-rovent-pc-divergence-witness/1";
const ACTUAL_SOURCE: &str = "rovent_actual_pc_divergence";
const TARGET_SOURCE: &str = "rovent_resolved_pc_divergence";
const ATTACK_DOMAIN: &str = "rz-rovent-pc-attack-ranked-source/1";
const WITH_ATTACK_DOMAIN: &str = "rz-rovent-pc-divergence-source-with-attack/1";

/// 원 future input/native만 보존한다. Head/target/teacher나 재귀 row는 없다.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VisibleRecordEvidence {
    input_json: String,
    input_sha256: String,
    native: RoventNativeLifecycle,
}

struct ParsedVisibleRecordEvidence {
    wire: VisibleRecordEvidence,
    input: rz_experiments::rovent_submitted_input::RoventSubmittedInputV1,
}

/// Borrowed input/native fields only; no training-row/target/origin capability.
struct SubmittedSourceFields<'a> {
    input: &'a rz_experiments::rovent_submitted_input::RoventSubmittedInputV1,
    input_sha256: &'a str,
    native: &'a RoventNativeLifecycle,
}
impl<'a> SubmittedSourceFields<'a> {
    fn from_record(record: &'a RoventTrainingRecordV2) -> Self {
        Self {
            input: &record.input,
            input_sha256: &record.input_sha256,
            native: &record.native,
        }
    }
    fn from_visible(evidence: &'a ParsedVisibleRecordEvidence) -> Self {
        Self {
            input: &evidence.input,
            input_sha256: &evidence.wire.input_sha256,
            native: &evidence.wire.native,
        }
    }
}

fn authority_json(a: SearchAuthority) -> Value {
    json!({"process_epoch":a.epoch.0,"game_generation":a.game.0,
        "root_generation":a.root.0,"implementation_sha256":hex(a.implementation)})
}
fn state_json(s: rz_contracts::StateIdentity) -> Value {
    json!({"owner":s.owner.0,"revision":s.revision.0,"semantic":hex(s.semantic)})
}
fn view_json(v: ReviewViewSeal) -> Value {
    json!({"authority":authority_json(v.authority),"view_sequence":v.view_sequence,
        "context_policy":hex(v.context_policy),"view_digest":hex(v.view_digest),
        "root_state":state_json(v.root_state),"actual_tokens":v.actual_tokens,
        "selected_records":v.selected_records,"anchors":v.anchors,"length_bucket":v.length_bucket})
}
fn question_id_json(id: ReciprocalQuestionId) -> Value {
    json!({"authority":authority_json(id.authority),"sequence":id.sequence})
}
fn peer_json(p: QuestionEnvelope) -> Value {
    json!({"id":question_id_json(p.id),"parent":p.parent.map(question_id_json),
        "sender":format!("{:?}",p.sender),"recipient":format!("{:?}",p.recipient),
        "purpose":format!("{:?}",p.purpose),"anchor":state_json(p.anchor),
        "anchor_root_ply":p.anchor_root_ply.0,
        "original_first_move":p.original_first_move.map(|m|m.bits()),
        "prefix_digest":hex(p.prefix_digest),"line_digest":p.line_digest.map(hex),
        "line_offset":p.line_offset.0,"question_digest":hex(p.question_digest),
        "view":view_json(p.view),"evidence_revision":p.evidence_revision,
        "root_budget":hex(p.root_budget),
        "original_deadline":{"clock":p.original_deadline.clock.0.0,"tick":p.original_deadline.at.0}})
}
fn origin_json(o: &AcceptedRoleOrigin) -> Value {
    let frame = o.frame();
    let mut frame_raw = authority_json(frame.authority);
    frame_raw["sequence"] = json!(frame.sequence);
    let (kind, peer) = match o.kind() {
        AcceptedRoleOriginKind::LocalAccepted => ("local_accepted", Value::Null),
        AcceptedRoleOriginKind::PeerQuestion(p) => ("peer_question", peer_json(p)),
    };
    json!({"frame":frame_raw,"ticket":ticket_json(o.ticket()),"output_digest":hex(o.output_digest()),
        "kind":kind,"peer":peer,"seal_sha256":hex(o.seal_digest())})
}
fn pair_json(p: &FreshPairSeal) -> Value {
    json!({"public_context":hex(p.public_context),"authority":authority_json(p.authority),
        "root_state":state_json(p.root_state),"context_policy":hex(p.context_policy),
        "selected_records":p.selected_records,"anchors":p.anchors,"model":hex(p.model),
        "encoding":hex(p.encoding),"precision":format!("{:?}",p.precision),
        "frozen_epoch":p.frozen_epoch,"record_revision":p.record_revision,
        "role":role_name(p.role),"perspective":color_name(p.perspective == rz_contracts::Color::White)})
}
fn color_name(white: bool) -> &'static str {
    if white { "white" } else { "black" }
}
fn line_json(line: &PcRepairAcceptedLine) -> Result<Value, ArenaError> {
    let kind = match line.kind() {
        rz_search::pals::engine::RecordKind::Repair => "repair",
        rz_search::pals::engine::RecordKind::Counterexample => "counterexample",
        _ => return Err(invalid("actual Repair witness has unsupported line kind")),
    };
    Ok(
        json!({"native_line_sha256":hex(line.digest()),"accepted":origin_json(line.accepted()),
        "line_id":line.line().0 as u64,"observation_id":line.observation().0 as u64,
        "revision":line.revision(),"kind":kind,"moves":packed(line.moves())?}),
    )
}
fn endpoint_json(e: &PcAcceptedFreshEndpoint) -> Result<Value, ArenaError> {
    let physical = e.physical();
    let execution = physical
        .execution
        .ok_or_else(|| invalid("Fresh endpoint execution absent"))?;
    let completed = physical
        .completed_nn_inputs
        .ok_or_else(|| invalid("Fresh endpoint count absent"))?;
    let model = e.model();
    Ok(
        json!({"native_endpoint_sha256":hex(e.digest()),"accepted":origin_json(e.accepted()),
        "line":packed(e.line())?,"pair":pair_json(e.pair()),
        "model":{"semantics":model.semantics,"model":model.model,"encoding":model.encoding,
            "precision":model.precision,"model_epoch_sha256":hex(Digest(model.model_epoch))},
        "perspective":color_name(e.perspective()==rz_position::Color::White),
        "raw_wdl_bits":e.raw_wdl().map(f32::to_bits),"root_wdl_bits":e.root_wdl().map(f32::to_bits),
        "physical":{"execution_epoch":execution.epoch.0,"execution_sequence":execution.sequence,
            "completed_nn_inputs":completed,"measurement_complete":physical.measurement_complete,
            "physical_fresh":physical.physical_fresh}}),
    )
}
fn attack_json(attack: &PcAttackRankedSource) -> Result<Value, ArenaError> {
    Ok(
        json!({"schema":ATTACK_DOMAIN,"native_attack_sha256":hex(attack.digest()),
        "accepted":origin_json(attack.accepted()),"root_budget_sha256":hex(attack.root_budget()),
        "source_sequence":attack.source_sequence(),
        "divergence_index":attack.divergence_index(),"divergence_ply":attack.divergence_ply(),
        "first_move":Move16::pack(attack.original_first_move()).map_err(|e|invalid(e.to_string()))?.bits(),
        "proposal":packed(attack.proposal())?,
        "perspective":color_name(attack.perspective()==rz_position::Color::White)}),
    )
}
fn witness_json(
    w: &PcDivergenceWitness,
    budget: &str,
    visible: Option<&RoventTrainingRecordV2>,
) -> Result<Value, ArenaError> {
    let source = w.source();
    let visible_source = json!({"native_source_sha256":hex(source.visible_source_digest()),
        "accepted":origin_json(source.accepted()),"root_budget_sha256":budget,
        "counter_local_index":source.counter_local_index(),"counter_revision":source.counter_revision(),
        "divergence_index":source.divergence_index(),"divergence_ply":source.divergence_ply(),
        "first_move":Move16::pack(source.first_move()).map_err(|e|invalid(e.to_string()))?.bits(),
        "proposal":packed(source.proposal())?,"counter":packed(source.counter())?,
        "perspective":color_name(source.perspective()==rz_position::Color::White)});
    let source_raw = match (source.attack_ranked_source(), visible) {
        (None, None) => visible_source,
        (Some(attack), Some(visible)) => json!({"schema":WITH_ATTACK_DOMAIN,
            "native_source_sha256":hex(source.digest()),"visible_source":visible_source,
            "visible_source_sequence":source.source_sequence(),
            "attack_ranked_source":attack_json(attack)?,
            "visible_record_evidence":{"input_json":serde_json::to_string(&visible.input).map_err(|e|invalid(e.to_string()))?,"input_sha256":visible.input_sha256,
                "native":visible.native}}),
        _ => {
            return Err(invalid(
                "initial attack source and original visible retired row must be supplied together",
            ));
        }
    };
    let resolution = match w.resolution() {
        PcDivergenceResolution::Supported => "supported",
        PcDivergenceResolution::Refuted => "refuted",
        PcDivergenceResolution::Unknown => "unknown",
        PcDivergenceResolution::Disputed => "disputed",
    };
    let mut raw = json!({"schema":WITNESS_DOMAIN,"native_witness_sha256":hex(w.digest()),
        "native_hash_kind":"unresolved","event_sequence":w.event_sequence(),"reason":w.reason(),
        "resolution":resolution,"target_known":w.target_known(),
        "source":source_raw,
        "repair":w.repair().map(line_json).transpose()?,"recheck":w.recheck().map(line_json).transpose()?,
        "repaired_endpoint":w.repaired_endpoint().map(endpoint_json).transpose()?,
        "counter_endpoint":w.counter_endpoint().map(endpoint_json).transpose()?});
    // Unknown can be an equal completed pair or a partial unresolved cycle.
    // Select only by the original native digest, never by a caller label/reason.
    if witness_digest(&raw, true).ok().as_deref() == Some(hex(w.digest()).as_str()) {
        raw["native_hash_kind"] = json!("frozen_pair");
    } else {
        check(
            witness_digest(&raw, false)? == hex(w.digest()),
            "unrecognized original witness framing",
        )?;
    }
    Ok(raw)
}

/// Only the original opaque witness and retired native row reach this writer.
/// All validation precedes mutation; the producer seals the completed row later.
pub(super) fn attach_actual_pc_divergence(
    row: &mut submitted_v3::RoventRetiredRecordV2,
    witness: &PcDivergenceWitness,
    observed_sequence: u64,
) -> Result<(), ArenaError> {
    check(
        witness.source().attack_ranked_source().is_none(),
        "InitialAttackEvidenceRequired: use both original retired rows for an initial attack witness",
    )?;
    check(
        witness.source().accepted().ticket() == &row.ticket,
        "MissingOriginalAttackSource: actual source ticket is not this submitted input",
    )?;
    attach_actual_pc_divergence_inner(row, witness, None, observed_sequence)
}

/// The optional native attack capability is joined only with its original row
/// and the independent future visible-counter row. Neither row is deserialized
/// into execution custody, and the initial immutable input is never changed.
pub(super) fn attach_actual_pc_divergence_from_visible(
    row: &mut submitted_v3::RoventRetiredRecordV2,
    witness: &PcDivergenceWitness,
    visible: &submitted_v3::RoventRetiredRecordV2,
    observed_sequence: u64,
) -> Result<(), ArenaError> {
    let attack = witness
        .source()
        .attack_ranked_source()
        .ok_or_else(|| invalid("initial attack native capability is absent"))?;
    check(
        attack.accepted().ticket() == &row.ticket
            && witness.source().accepted().ticket() == &visible.ticket,
        "initial and future visible sources must match their original retired tickets",
    )?;
    visible
        .admission
        .validate_ticket(&visible.ticket)
        .map_err(|e| invalid(e.to_string()))?;
    check(
        row.manifest_sha256 == visible.manifest_sha256
            && row.admission.raw_receipt() == visible.admission.raw_receipt()
            && visible.record.native.raw_retirement["root_admission"]
                == *visible.admission.raw_receipt()
            && row.record.lineage.actor_id == visible.record.lineage.actor_id,
        "initial/future visible original manifest, admission and actor differ",
    )?;
    attach_actual_pc_divergence_inner(row, witness, Some(&visible.record), observed_sequence)
}

fn attach_actual_pc_divergence_inner(
    row: &mut submitted_v3::RoventRetiredRecordV2,
    witness: &PcDivergenceWitness,
    visible: Option<&RoventTrainingRecordV2>,
    observed_sequence: u64,
) -> Result<(), ArenaError> {
    row.admission
        .validate_ticket(&row.ticket)
        .map_err(|e| invalid(e.to_string()))?;
    check(
        row.record.native.raw_retirement["root_admission"] == *row.admission.raw_receipt(),
        "divergence original live root admission differs",
    )?;
    check(
        !row.record.head_sources.contains_key("divergence")
            && row.record.targets.divergence.is_none(),
        "divergence source already attached",
    )?;
    let budget = text(&row.record.native.raw_retirement["root_admission"]["root_budget_sha256"])?;
    let raw = json!({"schema":HEAD_DOMAIN,"kind":"pc_divergence","input_sha256":row.record.input_sha256,
        "anchor":row.record.input.rules_anchor,"observed_sequence":observed_sequence,
        "witness":witness_json(witness,budget,visible)?});
    let sha = generation_digest(HEAD_DOMAIN, &raw)?;
    let head = RoventHeadSource {
        actor_id: row.record.lineage.actor_id.clone(),
        source: ACTUAL_SOURCE.into(),
        binary_sha256: None,
        profile_sha256: None,
        raw_receipt: raw,
        raw_receipt_sha256: sha.clone(),
        observed_sequence,
        anchor: row.record.input.rules_anchor.clone(),
    };
    verify_raw(&row.record, &head)?;
    let source = witness.source();
    let slot = source
        .attack_ranked_source()
        .map_or(source.divergence_index(), |attack| {
            attack.divergence_index()
        });
    let mapping = row
        .record
        .input
        .attack_mapping
        .get(usize::from(slot))
        .ok_or_else(|| invalid("actual divergence slot missing"))?;
    let target = if (visible.is_some() || mapping.selected_counter_local_index.is_some())
        && witness.target_known()
    {
        Some(RoventDivergenceTargetV1 {
            schema: ROVENT_DIVERGENCE_TARGET_DOMAIN.into(),
            observed_sequence,
            question_id: row.record.input.question_id.clone(),
            view_sha256: row.record.input.view_sha256.clone(),
            raw_evidence_sha256: sha,
            source: TARGET_SOURCE.into(),
            slot,
            attack_ply: u16::try_from(source.divergence_ply())
                .map_err(|_| invalid("attack ply overflow"))?,
            counter_question_id: hex(source.accepted().ticket().question),
            counter_native_view_sha256: hex(source.accepted().ticket().view.view_digest),
            counter_local_index: source.counter_local_index(),
            counter_revision: source.counter_revision(),
            repair_question_id: hex(witness
                .repair()
                .ok_or_else(|| invalid("known Repair absent"))?
                .accepted()
                .ticket()
                .question),
            status: match witness.resolution() {
                PcDivergenceResolution::Supported => {
                    GenerationDivergenceStatusV3::SupportedAfterRepair
                }
                PcDivergenceResolution::Refuted => GenerationDivergenceStatusV3::RefutedByRepair,
                _ => return Err(invalid("unknown witness cannot issue a known target")),
            },
        })
    } else {
        None
    };
    if let Some(target) = &target {
        verify_target(&row.record, &head, target)?;
    }
    // A future native projection is bounded by the same final row cap, not an
    // unbounded recursive record. Refuse before changing the original row.
    let mut candidate = row.record.clone();
    candidate.divergence_index = target.as_ref().map(|t| t.slot);
    candidate.targets.divergence = target;
    candidate.head_sources.insert("divergence".into(), head);
    if serde_json::to_vec(&candidate)
        .map_err(|e| invalid(e.to_string()))?
        .len()
        > GENERATION_MAX_ROW_BYTES
    {
        return Err(ArenaError::Budget(
            "PcDivergenceEvidenceCapacity: original row retained; final row exceeds 1MiB".into(),
        ));
    }
    row.record = candidate;
    Ok(())
}

/// Recomputes immutable data joins only. It cannot reconstruct an AcceptedRoleStep,
/// physical fence, Store pin, current question or current execution permission.
pub fn verify_historical_pc_divergence(
    record: &RoventTrainingRecordV2,
    manifest: &RoventCorpusManifestV2,
) -> Result<(), ArenaError> {
    record.validate(manifest)?;
    verify_historical_pc_divergence_fields(record, manifest, None)
}

/// Caller validates the exact record/lock domain before entering this shared
/// calculation. Future native evidence remains in its original domain.
pub(super) fn verify_historical_pc_divergence_lock_v2(
    record: &RoventTrainingRecordV2,
    manifest: &RoventCorpusManifestV2,
    lock: &rz_experiments::RoventInputLockV2,
) -> Result<(), ArenaError> {
    verify_historical_pc_divergence_fields(record, manifest, Some(lock))
}

fn verify_historical_pc_divergence_fields(
    record: &RoventTrainingRecordV2,
    manifest: &RoventCorpusManifestV2,
    lock: Option<&rz_experiments::RoventInputLockV2>,
) -> Result<(), ArenaError> {
    if let Some(head) = record.head_sources.get("divergence") {
        verify_raw(record, head)?;
        if let Some(visible) =
            visible_record_evidence(record, &head.raw_receipt["witness"]["source"])?
        {
            if let Some(lock) = lock {
                super::historical_native_lock_v2::verify_historical_lock_v2_input(
                    &visible.input,
                    &visible.wire.input_sha256,
                    &visible.wire.native,
                    manifest,
                    lock,
                )?;
            } else {
                verify_visible_history(&visible, manifest)?;
            }
        }
        if let Some(target) = &record.targets.divergence {
            verify_target(record, head, target)?;
        }
    } else {
        check(
            record.targets.divergence.is_none(),
            "known divergence lacks original raw owner evidence",
        )?;
    }
    Ok(())
}

fn fields(value: &Value, names: &[&str]) -> Result<(), ArenaError> {
    check(
        value
            .as_object()
            .is_some_and(|o| o.len() == names.len() && names.iter().all(|n| o.contains_key(*n))),
        "divergence exact wire field roster",
    )
}
fn number(value: &Value) -> Result<u64, ArenaError> {
    value
        .as_u64()
        .ok_or_else(|| invalid("divergence unsigned integer"))
}
fn text(value: &Value) -> Result<&str, ArenaError> {
    value
        .as_str()
        .ok_or_else(|| invalid("divergence text field"))
}
fn sha(value: &Value) -> Result<[u8; 32], ArenaError> {
    let value = text(value)?;
    check(
        generation_sha(value),
        "divergence digest must be canonical lower-case SHA256",
    )?;
    let mut bytes = [0; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[i * 2..i * 2 + 2], 16)
            .map_err(|_| invalid("invalid hex digest"))?;
    }
    Ok(bytes)
}
fn digest_text(hash: Sha256) -> String {
    hex(Digest(hash.finalize().into()))
}
fn hash_u64(hash: &mut Sha256, value: &Value) -> Result<(), ArenaError> {
    hash.update(number(value)?.to_le_bytes());
    Ok(())
}
fn hash_sha(hash: &mut Sha256, value: &Value) -> Result<(), ArenaError> {
    hash.update(sha(value)?);
    Ok(())
}
fn moves(value: &Value) -> Result<Vec<u16>, ArenaError> {
    let raw = value
        .as_array()
        .ok_or_else(|| invalid("divergence move array"))?;
    check(
        !raw.is_empty() && raw.len() <= 256,
        "bounded nonempty divergence line",
    )?;
    raw.iter()
        .map(|v| {
            let bits = u16::try_from(number(v)?).map_err(|_| invalid("Move16 overflow"))?;
            rz_contracts::pals::Move16::try_from_bits(bits)
                .and_then(|m| m.decode())
                .map_err(|e| invalid(e.to_string()))?;
            Ok(bits)
        })
        .collect()
}
fn hash_line(hash: &mut Sha256, value: &Value) -> Result<(), ArenaError> {
    let line = moves(value)?;
    hash.update((line.len() as u64).to_le_bytes());
    for movement in line {
        hash.update(movement.to_le_bytes());
    }
    Ok(())
}
fn color(value: &Value) -> Result<u8, ArenaError> {
    match text(value)? {
        "white" => Ok(0),
        "black" => Ok(1),
        _ => Err(invalid("divergence perspective")),
    }
}
fn authority(value: &Value) -> Result<(), ArenaError> {
    fields(
        value,
        &[
            "process_epoch",
            "game_generation",
            "root_generation",
            "implementation_sha256",
        ],
    )?;
    for name in ["process_epoch", "game_generation", "root_generation"] {
        number(&value[name])?;
    }
    check(
        number(&value["process_epoch"])? > 0,
        "zero process authority",
    )?;
    sha(&value["implementation_sha256"])?;
    Ok(())
}
fn same_authority(left: &Value, right: &Value) -> Result<(), ArenaError> {
    for key in [
        "process_epoch",
        "game_generation",
        "root_generation",
        "implementation_sha256",
    ] {
        check(left[key] == right[key], "divergence authority changed")?;
    }
    Ok(())
}
fn state(value: &Value) -> Result<(), ArenaError> {
    fields(value, &["owner", "revision", "semantic"])?;
    number(&value["owner"])?;
    number(&value["revision"])?;
    sha(&value["semantic"])?;
    Ok(())
}
fn ticket(value: &Value) -> Result<(), ArenaError> {
    fields(
        value,
        &[
            "process_epoch",
            "game_generation",
            "root_generation",
            "request_sequence",
            "implementation_sha256",
            "question_digest",
            "view_sequence",
            "view_digest",
            "context_policy",
            "actual_tokens",
            "selected_records",
            "anchors",
            "length_bucket",
            "root_state",
            "representation",
            "deadline_clock",
            "deadline_tick",
        ],
    )?;
    for key in [
        "process_epoch",
        "game_generation",
        "root_generation",
        "request_sequence",
        "view_sequence",
        "actual_tokens",
        "selected_records",
        "anchors",
        "length_bucket",
        "deadline_clock",
        "deadline_tick",
    ] {
        number(&value[key])?;
    }
    for key in [
        "implementation_sha256",
        "question_digest",
        "view_digest",
        "context_policy",
    ] {
        sha(&value[key])?;
    }
    state(&value["root_state"])?;
    let rep = &value["representation"];
    fields(
        rep,
        &[
            "input",
            "history",
            "candidates",
            "mask_positions",
            "model",
            "encoding",
            "precision",
            "frozen_epoch",
            "record_revision",
            "role",
        ],
    )?;
    for key in [
        "input",
        "history",
        "candidates",
        "mask_positions",
        "model",
        "encoding",
    ] {
        sha(&rep[key])?;
    }
    check(
        rep["precision"] == "Fp32" && matches!(rep["role"].as_str(), Some("proposer" | "critic")),
        "actual P/C witness requires the original FP32 native profile",
    )?;
    number(&rep["frozen_epoch"])?;
    number(&rep["record_revision"])?;
    check(
        number(&value["process_epoch"])? > 0
            && number(&value["request_sequence"])? > 0
            && number(&value["view_sequence"])? > 0
            && value["deadline_clock"] == value["process_epoch"]
            && number(&value["deadline_tick"])? > 0
            && number(&value["actual_tokens"])? <= 512
            && number(&value["selected_records"])? <= 64
            && (1..=4).contains(&number(&value["anchors"])?)
            && [8, 16, 32, 64, 128, 256].contains(&number(&value["length_bucket"])?),
        "bounded native ticket/view",
    )
}
fn peer(value: &Value, t: &Value, budget: &Value) -> Result<(), ArenaError> {
    fields(
        value,
        &[
            "id",
            "parent",
            "sender",
            "recipient",
            "purpose",
            "anchor",
            "anchor_root_ply",
            "original_first_move",
            "prefix_digest",
            "line_digest",
            "line_offset",
            "question_digest",
            "view",
            "evidence_revision",
            "root_budget",
            "original_deadline",
        ],
    )?;
    let id = &value["id"];
    fields(id, &["authority", "sequence"])?;
    authority(&id["authority"])?;
    same_authority(&id["authority"], t)?;
    check(number(&id["sequence"])? > 0, "peer sequence")?;
    if !value["parent"].is_null() {
        let parent = &value["parent"];
        fields(parent, &["authority", "sequence"])?;
        authority(&parent["authority"])?;
        check(
            parent["authority"] == id["authority"]
                && number(&parent["sequence"])? > 0
                && number(&parent["sequence"])? < number(&id["sequence"])?,
            "peer earlier parent/root",
        )?;
    }
    let recipient = if t["representation"]["role"] == "proposer" {
        "Proposer"
    } else {
        "Critic"
    };
    check(
        matches!(
            value["sender"].as_str(),
            Some("Proposer" | "Critic" | "Researcher")
        ) && value["recipient"] == recipient
            && value["sender"] != value["recipient"]
            && matches!(
                value["purpose"].as_str(),
                Some(
                    "Propose" | "Counter" | "Repair" | "TacticalReview" | "Continue" | "FreshValue"
                )
            ),
        "peer recipient/purpose",
    )?;
    state(&value["anchor"])?;
    check(
        number(&value["anchor_root_ply"])? <= u16::MAX as u64
            && number(&value["line_offset"])? <= u16::MAX as u64,
        "peer ply bounds",
    )?;
    if !value["original_first_move"].is_null() {
        let m = u16::try_from(number(&value["original_first_move"])?)
            .map_err(|_| invalid("peer move overflow"))?;
        rz_contracts::pals::Move16::try_from_bits(m)
            .and_then(|v| v.decode())
            .map_err(|e| invalid(e.to_string()))?;
    }
    sha(&value["prefix_digest"])?;
    if value["line_digest"].is_null() {
        check(
            number(&value["line_offset"])? == 0,
            "peer offset without line",
        )?;
    } else {
        sha(&value["line_digest"])?;
    }
    number(&value["evidence_revision"])?;
    sha(&value["root_budget"])?;
    check(
        value["root_budget"] == *budget && value["question_digest"] == t["question_digest"],
        "peer original budget/question",
    )?;
    let view = &value["view"];
    fields(
        view,
        &[
            "authority",
            "view_sequence",
            "context_policy",
            "view_digest",
            "root_state",
            "actual_tokens",
            "selected_records",
            "anchors",
            "length_bucket",
        ],
    )?;
    authority(&view["authority"])?;
    same_authority(&view["authority"], t)?;
    for key in [
        "view_sequence",
        "context_policy",
        "view_digest",
        "root_state",
        "actual_tokens",
        "selected_records",
        "anchors",
        "length_bucket",
    ] {
        check(view[key] == t[key], "peer original submitted view changed")?;
    }
    fields(&value["original_deadline"], &["clock", "tick"])?;
    check(
        value["original_deadline"]["clock"] == t["deadline_clock"]
            && value["original_deadline"]["tick"] == t["deadline_tick"],
        "peer original deadline",
    )
}
fn origin(value: &Value, original: &Value, budget: &Value, role: &str) -> Result<(), ArenaError> {
    fields(
        value,
        &[
            "frame",
            "ticket",
            "output_digest",
            "kind",
            "peer",
            "seal_sha256",
        ],
    )?;
    let t = &value["ticket"];
    ticket(t)?;
    same_authority(t, original)?;
    check(
        t["root_state"] == original["root_state"]
            && t["deadline_clock"] == original["deadline_clock"]
            && t["deadline_tick"] == original["deadline_tick"]
            && t["representation"]["role"] == role,
        "origin exact root/deadline/role",
    )?;
    for key in ["model", "encoding", "precision", "frozen_epoch"] {
        check(
            t["representation"][key] == original["representation"][key],
            "origin frozen model namespace changed",
        )?;
    }
    let f = &value["frame"];
    fields(
        f,
        &[
            "process_epoch",
            "game_generation",
            "root_generation",
            "implementation_sha256",
            "sequence",
        ],
    )?;
    same_authority(f, t)?;
    check(number(&f["sequence"])? > 0, "accepted frame sequence")?;
    sha(&value["output_digest"])?;
    match text(&value["kind"])? {
        "local_accepted" => check(value["peer"].is_null(), "local source must not invent peer")?,
        "peer_question" => peer(&value["peer"], t, budget)?,
        _ => return Err(invalid("unknown accepted origin kind")),
    }
    check(
        origin_digest(value)? == text(&value["seal_sha256"])?,
        "original accepted origin native seal",
    )
}
fn origin_digest(value: &Value) -> Result<String, ArenaError> {
    let t = &value["ticket"];
    let r = &t["representation"];
    let mut h = Sha256::new();
    h.update(b"rz-rovent-accepted-local-origin-seal/1\0");
    hash_u64(&mut h, &value["frame"]["sequence"])?;
    hash_u64(&mut h, &t["request_sequence"])?;
    for key in ["process_epoch", "game_generation", "root_generation"] {
        hash_u64(&mut h, &t[key])?;
    }
    for key in ["implementation_sha256", "question_digest", "view_digest"] {
        hash_sha(&mut h, &t[key])?;
    }
    hash_u64(&mut h, &t["view_sequence"])?;
    hash_sha(&mut h, &value["output_digest"])?;
    for key in [
        "input",
        "history",
        "candidates",
        "mask_positions",
        "model",
        "encoding",
    ] {
        hash_sha(&mut h, &r[key])?;
    }
    hash_u64(&mut h, &r["frozen_epoch"])?;
    hash_u64(&mut h, &r["record_revision"])?;
    h.update([match text(&r["role"])? {
        "proposer" => 0,
        "critic" => 1,
        _ => return Err(invalid("origin role")),
    }]);
    check(r["precision"] == "Fp32", "actual origin precision")?;
    h.update([0]);
    hash_u64(&mut h, &t["deadline_tick"])?;
    match text(&value["kind"])? {
        "local_accepted" => h.update([0]),
        "peer_question" => {
            h.update([1]);
            hash_u64(&mut h, &value["peer"]["id"]["sequence"])?;
            hash_sha(&mut h, &value["peer"]["question_digest"])?;
        }
        _ => return Err(invalid("origin kind")),
    }
    Ok(digest_text(h))
}
fn source_digest(s: &Value) -> Result<String, ArenaError> {
    let mut h = Sha256::new();
    h.update(b"rz-rovent-pc-divergence-source/1\0");
    hash_sha(&mut h, &s["accepted"]["seal_sha256"])?;
    hash_sha(&mut h, &s["root_budget_sha256"])?;
    h.update(
        u16::try_from(number(&s["counter_local_index"])?)
            .map_err(|_| invalid("counter local overflow"))?
            .to_le_bytes(),
    );
    hash_u64(&mut h, &s["counter_revision"])?;
    h.update(
        u16::try_from(number(&s["divergence_index"])?)
            .map_err(|_| invalid("divergence slot overflow"))?
            .to_le_bytes(),
    );
    hash_u64(&mut h, &s["divergence_ply"])?;
    hash_line(&mut h, &s["proposal"])?;
    hash_line(&mut h, &s["counter"])?;
    Ok(digest_text(h))
}
fn accepted_line(
    value: &Value,
    original: &Value,
    budget: &Value,
    kind: &str,
) -> Result<(), ArenaError> {
    fields(
        value,
        &[
            "native_line_sha256",
            "accepted",
            "line_id",
            "observation_id",
            "revision",
            "kind",
            "moves",
        ],
    )?;
    check(value["kind"] == kind, "Repair/C line kind")?;
    origin(
        &value["accepted"],
        original,
        budget,
        if kind == "repair" {
            "proposer"
        } else {
            "critic"
        },
    )?;
    let revision = number(&value["revision"])?;
    check(
        revision > number(&value["accepted"]["ticket"]["representation"]["record_revision"])?,
        "line publication revision must follow submission",
    )?;
    let mut h = Sha256::new();
    h.update(b"rz-rovent-pc-accepted-repair-line/1\0");
    hash_sha(&mut h, &value["accepted"]["seal_sha256"])?;
    for key in ["line_id", "observation_id", "revision"] {
        hash_u64(&mut h, &value[key])?;
    }
    h.update([if kind == "repair" { 2 } else { 1 }]);
    hash_line(&mut h, &value["moves"])?;
    check(
        digest_text(h) == text(&value["native_line_sha256"])?,
        "actual accepted line native seal",
    )
}
fn wdl(value: &Value) -> Result<[f32; 3], ArenaError> {
    let a = value
        .as_array()
        .filter(|v| v.len() == 3)
        .ok_or_else(|| invalid("three WDL f32 words"))?;
    let mut out = [0.0; 3];
    for (i, bits) in a.iter().enumerate() {
        out[i] =
            f32::from_bits(u32::try_from(number(bits)?).map_err(|_| invalid("WDL word overflow"))?);
    }
    check(
        out.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            && (out.iter().sum::<f32>() - 1.0).abs() <= 1e-4,
        "finite normalized FP32 WDL",
    )?;
    Ok(out)
}
fn endpoint(
    value: &Value,
    original: &Value,
    budget: &Value,
    native: &Value,
    perspective: &Value,
) -> Result<(), ArenaError> {
    fields(
        value,
        &[
            "native_endpoint_sha256",
            "accepted",
            "line",
            "pair",
            "model",
            "perspective",
            "raw_wdl_bits",
            "root_wdl_bits",
            "physical",
        ],
    )?;
    origin(&value["accepted"], original, budget, "proposer")?;
    check(
        value["perspective"] == *perspective,
        "endpoint source viewpoint changed",
    )?;
    let t = &value["accepted"]["ticket"];
    let p = &value["pair"];
    fields(
        p,
        &[
            "public_context",
            "authority",
            "root_state",
            "context_policy",
            "selected_records",
            "anchors",
            "model",
            "encoding",
            "precision",
            "frozen_epoch",
            "record_revision",
            "role",
            "perspective",
        ],
    )?;
    sha(&p["public_context"])?;
    authority(&p["authority"])?;
    same_authority(&p["authority"], t)?;
    for key in [
        "root_state",
        "context_policy",
        "selected_records",
        "anchors",
    ] {
        check(p[key] == t[key], "Fresh pair original ticket view")?;
    }
    for key in [
        "model",
        "encoding",
        "precision",
        "frozen_epoch",
        "record_revision",
        "role",
    ] {
        check(
            p[key] == t["representation"][key],
            "Fresh pair original model/role key",
        )?;
    }
    check(
        p["role"] == "proposer" && p["perspective"] == *perspective,
        "Fresh pair role/viewpoint",
    )?;
    let m = &value["model"];
    fields(
        m,
        &[
            "semantics",
            "model",
            "encoding",
            "precision",
            "model_epoch_sha256",
        ],
    )?;
    check(
        m["semantics"] == rz_search::pals::value::MODEL_WDL_VALUE_SEMANTICS
            && m["model"]
                == format!(
                    "pals-native-virtual-v3:{}:{}:{}",
                    text(&native["export_manifest_sha256"])?,
                    text(&native["weights_sha256"])?,
                    text(&native["backend_sha256"])?
                )
            && m["encoding"]
                == format!(
                    "{}:{}",
                    text(&native["encoding"])?,
                    text(&native["encoding_registry_sha256"])?
                )
            && m["precision"] == "fp32_tf32_off"
            && m["model_epoch_sha256"] == native["weights_sha256"]
            && t["representation"]["model"] == native["weights_sha256"]
            && t["representation"]["encoding"] == native["encoding_sha256"]
            && t["representation"]["frozen_epoch"] == native["frozen_epoch"],
        "Fresh original loaded model identity, not a fabricated ticket hex label",
    )?;
    let raw = wdl(&value["raw_wdl_bits"])?;
    let root = wdl(&value["root_wdl_bits"])?;
    let line = moves(&value["line"])?;
    let expected = if line.len() % 2 == 0 {
        raw
    } else {
        [raw[2], raw[1], raw[0]]
    };
    check(
        root.map(f32::to_bits) == expected.map(f32::to_bits),
        "Fresh endpoint raw/root perspective f32 bits",
    )?;
    let physical = &value["physical"];
    fields(
        physical,
        &[
            "execution_epoch",
            "execution_sequence",
            "completed_nn_inputs",
            "measurement_complete",
            "physical_fresh",
        ],
    )?;
    check(
        physical["execution_epoch"] == t["process_epoch"]
            && number(&physical["execution_sequence"])? > 0
            && number(&physical["completed_nn_inputs"])? > 0
            && physical["measurement_complete"] == true
            && physical["physical_fresh"] == true,
        "original physically completed Fresh input required",
    )?;
    let mut h = Sha256::new();
    h.update(b"rz-rovent-pc-accepted-fresh-endpoint/1\0");
    hash_sha(&mut h, &value["accepted"]["seal_sha256"])?;
    hash_sha(&mut h, &p["public_context"])?;
    for key in [
        "execution_epoch",
        "execution_sequence",
        "completed_nn_inputs",
    ] {
        hash_u64(&mut h, &physical[key])?;
    }
    h.update([color(perspective)?]);
    for v in raw {
        h.update(v.to_bits().to_le_bytes());
    }
    hash_line(&mut h, &value["line"])?;
    check(
        digest_text(h) == text(&value["native_endpoint_sha256"])?,
        "actual Fresh native endpoint seal",
    )
}
fn witness_digest(w: &Value, pair: bool) -> Result<String, ArenaError> {
    let mut h = Sha256::new();
    h.update(if pair {
        &b"rz-rovent-pc-divergence-witness/1\0"[..]
    } else {
        &b"rz-rovent-pc-divergence-unresolved/1\0"[..]
    });
    hash_sha(&mut h, &w["source"]["native_source_sha256"])?;
    for (key, digest) in [
        ("repair", "native_line_sha256"),
        ("recheck", "native_line_sha256"),
        ("repaired_endpoint", "native_endpoint_sha256"),
        ("counter_endpoint", "native_endpoint_sha256"),
    ] {
        if !pair {
            h.update([u8::from(!w[key].is_null())]);
        }
        if !w[key].is_null() {
            hash_sha(&mut h, &w[key][digest])?;
        } else if pair {
            return Err(invalid("full frozen pair fact missing"));
        }
    }
    hash_u64(&mut h, &w["event_sequence"])?;
    if pair {
        h.update([match text(&w["resolution"])? {
            "supported" => 0,
            "refuted" => 1,
            "unknown" => 2,
            "disputed" => 3,
            _ => return Err(invalid("divergence resolution")),
        }]);
    } else {
        let reason = text(&w["reason"])?;
        h.update((reason.len() as u64).to_le_bytes());
        h.update(reason.as_bytes());
    }
    Ok(digest_text(h))
}

fn visible_source(source: &Value) -> &Value {
    if source["schema"] == WITH_ATTACK_DOMAIN {
        &source["visible_source"]
    } else {
        source
    }
}

fn initial_attack(source: &Value) -> Option<&Value> {
    if source["schema"] == WITH_ATTACK_DOMAIN {
        Some(&source["attack_ranked_source"])
    } else {
        None
    }
}

fn visible_record_evidence(
    original: &RoventTrainingRecordV2,
    source: &Value,
) -> Result<Option<ParsedVisibleRecordEvidence>, ArenaError> {
    if source["schema"].is_null() {
        return Ok(None);
    }
    fields(
        source,
        &[
            "schema",
            "native_source_sha256",
            "visible_source",
            "visible_source_sequence",
            "attack_ranked_source",
            "visible_record_evidence",
        ],
    )?;
    check(
        source["schema"] == WITH_ATTACK_DOMAIN,
        "unknown explicit divergence source variant",
    )?;
    let raw = &source["visible_record_evidence"];
    fields(raw, &["input_json", "input_sha256", "native"])?;
    let wire: VisibleRecordEvidence =
        serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
    let input = parse_rovent_visible_input_json(&wire.input_json)?;
    let future = ParsedVisibleRecordEvidence { wire, input };
    future.input.validate()?;
    check(
        future.input.sha256()? == future.wire.input_sha256
            && future.wire.native.raw_submission["input_sha256"] == future.wire.input_sha256
            && future.wire.native.raw_completion["input_sha256"] == future.wire.input_sha256
            && future.wire.native.raw_retirement["input_sha256"] == future.wire.input_sha256
            && future.input.role == GenerationRoleV3::Critic
            && future.wire.native.head == "critic"
            && future.wire.native.submitted_sequence == future.input.capture_sequence
            && original.input.capture_sequence < future.input.capture_sequence
            && future.wire.native.raw_submission["ticket"]["question_digest"]
                == future.input.question_id
            && future.input.game_id == original.input.game_id
            && future.input.opening_id == original.input.opening_id
            && future.input.model_epoch == original.input.model_epoch
            && future.input.context_epoch == original.input.context_epoch
            && future.input.actual_history == original.input.actual_history
            && future.input.rules_anchor == original.input.rules_anchor
            && future.wire.native.source_sha256 == original.native.source_sha256
            && future.wire.native.raw_retirement["root_admission"]
                == original.native.raw_retirement["root_admission"],
        "future visible original submitted input seal and exact source/root/history scope",
    )?;
    Ok(Some(future))
}

fn verify_visible_history(
    future: &ParsedVisibleRecordEvidence,
    manifest: &RoventCorpusManifestV2,
) -> Result<(), ArenaError> {
    future
        .wire
        .native
        .validate_submitted_input_v3(&manifest.common_registration_projection())?;
    let input = future.input.common_validation_projection();
    let replay: rules_replay::RoventSubmittedRulesReplay =
        serde_json::from_value(future.wire.native.raw_submission["rules_replay"].clone())
            .map_err(|e| invalid(e.to_string()))?;
    let attacks = future
        .input
        .attack_mapping
        .iter()
        .map(|slot| usize::from(slot.attack_ply))
        .collect::<Vec<_>>();
    rules_replay::verify_historical_rules_fields(&input, &future.wire.native, &replay, &attacks)?;
    submitted_review_historical::verify_historical_submitted_review_fields(
        &future.input,
        &future.wire.input_sha256,
        &future.wire.native,
    )?;
    historical_scope::verify_historical_submitted_native_fields(
        &input,
        &future.wire.native,
        &manifest.common_registration_projection(),
    )?;
    Ok(())
}

fn attack_digest(attack: &Value) -> Result<String, ArenaError> {
    let mut h = Sha256::new();
    h.update(b"rz-rovent-pc-attack-ranked-source/1\0");
    hash_sha(&mut h, &attack["accepted"]["seal_sha256"])?;
    hash_sha(&mut h, &attack["root_budget_sha256"])?;
    h.update(
        u16::try_from(number(&attack["divergence_index"])?)
            .map_err(|_| invalid("initial attack slot overflow"))?
            .to_le_bytes(),
    );
    hash_u64(&mut h, &attack["divergence_ply"])?;
    h.update(
        u16::try_from(number(&attack["first_move"])?)
            .map_err(|_| invalid("initial first move overflow"))?
            .to_le_bytes(),
    );
    hash_line(&mut h, &attack["proposal"])?;
    Ok(digest_text(h))
}

fn combined_source_digest(source: &Value) -> Result<String, ArenaError> {
    let mut h = Sha256::new();
    h.update(b"rz-rovent-pc-divergence-source-with-attack/1\0");
    hash_sha(&mut h, &source["visible_source"]["native_source_sha256"])?;
    hash_sha(
        &mut h,
        &source["attack_ranked_source"]["native_attack_sha256"],
    )?;
    hash_u64(&mut h, &source["visible_source_sequence"])?;
    hash_u64(&mut h, &source["attack_ranked_source"]["source_sequence"])?;
    Ok(digest_text(h))
}

fn ranked_attack(
    record: &RoventTrainingRecordV2,
    attack: &Value,
    visible: &Value,
) -> Result<(), ArenaError> {
    fields(
        attack,
        &[
            "schema",
            "native_attack_sha256",
            "accepted",
            "root_budget_sha256",
            "source_sequence",
            "divergence_index",
            "divergence_ply",
            "first_move",
            "proposal",
            "perspective",
        ],
    )?;
    check(
        attack["schema"] == ATTACK_DOMAIN,
        "initial ranked attack exact domain",
    )?;
    let original = &record.native.raw_submission["ticket"];
    let budget = &record.native.raw_retirement["root_admission"]["root_budget_sha256"];
    origin(&attack["accepted"], original, budget, "critic")?;
    check(
        record.role == GenerationRoleV3::Critic
            && record.input.sha256()? == record.input_sha256
            && attack["accepted"]["ticket"] == *original
            && attack["accepted"]["output_digest"] == record.native.output_digest
            && attack["accepted"]["output_digest"] == record.native.raw_completion["output_digest"]
            && record.native.raw_completion["ticket"] == *original
            && record.native.raw_retirement["ticket"] == *original
            && attack["root_budget_sha256"] == *budget,
        "initial attack original retired C input/output/budget",
    )?;
    let query = &record.input.selected_view.query;
    let proposal = moves(&attack["proposal"])?;
    let slot = usize::try_from(number(&attack["divergence_index"])?)
        .map_err(|_| invalid("initial attack slot"))?;
    let ply = usize::try_from(number(&attack["divergence_ply"])?)
        .map_err(|_| invalid("initial attack ply"))?;
    let mapping = record
        .input
        .attack_mapping
        .get(slot)
        .ok_or_else(|| invalid("initial attack original slot absent"))?;
    check(
        query.purpose == "divergence_policy"
            && query.prefix.is_empty()
            && !query.counter_present
            && query.counter.is_empty()
            && query.proposal == proposal
            && ply > 0
            && ply < proposal.len()
            && usize::from(mapping.slot) == slot
            && usize::from(mapping.attack_ply) == ply
            && mapping.selected_counter_local_index.is_none()
            && mapping.selected_counter_revision.is_none()
            && proposal.first().copied().map(u64::from) == attack["first_move"].as_u64()
            && attack["perspective"] == color_name(record.input.rules_anchor.white_to_move),
        "initial attack must preserve the original no-counter question, slot and Rules perspective",
    )?;
    let bits = record.native.raw_completion["output"]["payload"]["divergence_logits_bits"]
        .as_array()
        .ok_or_else(|| invalid("initial attack original D logits absent"))?;
    check(
        bits.len() == record.input.attack_mapping.len(),
        "initial attack original D coverage",
    )?;
    let logits = bits
        .iter()
        .map(|word| {
            let value = f32::from_bits(
                u32::try_from(number(word)?).map_err(|_| invalid("initial D word overflow"))?,
            );
            check(value.is_finite(), "initial attack finite D logits")?;
            Ok(value)
        })
        .collect::<Result<Vec<_>, ArenaError>>()?;
    check(
        selected_slot(&logits) == Some(slot),
        "initial attack differs from original D argmax",
    )?;
    check(
        attack_digest(attack)? == text(&attack["native_attack_sha256"])?,
        "initial ranked attack native seal",
    )?;
    // The two slot indices intentionally need not match. Both are sealed in
    // their original immutable queries; only the selected attack ply is shared.
    origin(&visible["accepted"], original, budget, "critic")?;
    let counter = moves(&visible["counter"])?;
    check(
        counter.get(..ply) == proposal.get(..ply)
            && counter.get(ply).is_some()
            && counter.get(ply) != proposal.get(ply),
        "initial attack counter must preserve the proposal prefix and change the original attack move",
    )?;
    check(
        number(&original["request_sequence"])?
            < number(&visible["accepted"]["ticket"]["request_sequence"])?
            && attack["divergence_ply"] == visible["divergence_ply"]
            && attack["proposal"] == visible["proposal"]
            && attack["first_move"] == visible["first_move"]
            && attack["perspective"] == visible["perspective"]
            && attack["root_budget_sha256"] == visible["root_budget_sha256"],
        "initial attack and later visible source changed original root/model/deadline/attack or chronology",
    )
}

fn original_source(record: &SubmittedSourceFields<'_>, s: &Value) -> Result<(), ArenaError> {
    fields(
        s,
        &[
            "native_source_sha256",
            "accepted",
            "root_budget_sha256",
            "counter_local_index",
            "counter_revision",
            "divergence_index",
            "divergence_ply",
            "first_move",
            "proposal",
            "counter",
            "perspective",
        ],
    )?;
    let input = &record.input;
    input.validate()?;
    let original = &record.native.raw_submission["ticket"];
    let budget = &record.native.raw_retirement["root_admission"]["root_budget_sha256"];
    sha(budget)?;
    origin(&s["accepted"], original, budget, "critic")?;
    check(
        record.input.role == GenerationRoleV3::Critic
            && input.sha256()? == record.input_sha256
            && record.native.raw_submission["input_sha256"] == record.input_sha256
            && s["accepted"]["ticket"] == *original
            && s["accepted"]["output_digest"] == record.native.output_digest
            && s["accepted"]["output_digest"] == record.native.raw_completion["output_digest"]
            && record.native.raw_completion["ticket"] == *original
            && record.native.raw_retirement["ticket"] == *original
            && s["root_budget_sha256"] == *budget,
        "source is not the exact original retired C ticket/output/root budget",
    )?;
    let slot =
        usize::try_from(number(&s["divergence_index"])?).map_err(|_| invalid("D slot overflow"))?;
    let local = usize::try_from(number(&s["counter_local_index"])?)
        .map_err(|_| invalid("counter index overflow"))?;
    let revision = number(&s["counter_revision"])?;
    let attack = usize::try_from(number(&s["divergence_ply"])?)
        .map_err(|_| invalid("attack ply overflow"))?;
    let proposal = moves(&s["proposal"])?;
    let counter = moves(&s["counter"])?;
    let first =
        u16::try_from(number(&s["first_move"])?).map_err(|_| invalid("first move overflow"))?;
    let query = &input.selected_view.query;
    check(
        local < 64
            && revision > 0
            && attack > 0
            && attack < proposal.len()
            && attack < counter.len()
            && query.purpose == "divergence_policy"
            && query.prefix.is_empty()
            && query.counter_present
            && query.proposal == proposal
            && query.counter == counter
            && proposal.first() == Some(&first)
            && counter.first() == Some(&first)
            && s["perspective"] == color_name(input.rules_anchor.white_to_move),
        "source exact submitted question/lines/attack/perspective",
    )?;
    let mapping = input
        .attack_mapping
        .get(slot)
        .ok_or_else(|| invalid("source original D slot absent"))?;
    check(
        usize::from(mapping.slot) == slot
            && usize::from(mapping.attack_ply) == attack
            && mapping.selected_counter_local_index.is_none_or(|index| {
                usize::from(index) == local && mapping.selected_counter_revision == Some(revision)
            }),
        "source differs from the immutable original attack mapping",
    )?;
    let selected = input
        .selected_view
        .records
        .get(local)
        .ok_or_else(|| invalid("source original selected counter absent"))?;
    let meta = &record.native.raw_submission["submitted_review"]["record_metadata"][local];
    check(
        selected.kind == GenerationRecordKindV3::Counter
            && selected.moves == counter
            && input
                .selected_view
                .anchors
                .get(usize::from(selected.anchor))
                == Some(&input.rules_anchor)
            && meta["record_revision"].as_u64() == Some(revision)
            && meta["source_revision"].as_u64() == Some(revision)
            && meta["source_line_offset"].as_u64() == Some(0)
            && meta["original_first_move"].as_u64() == Some(u64::from(first)),
        "counter original visible revision/source anchor/first move",
    )?;
    let bits = record.native.raw_completion["output"]["payload"]["divergence_logits_bits"]
        .as_array()
        .ok_or_else(|| invalid("source original C D head bits absent"))?;
    check(
        bits.len() == input.attack_mapping.len() && !bits.is_empty(),
        "source D head exact slot coverage",
    )?;
    let logits = bits
        .iter()
        .map(|word| {
            let value = f32::from_bits(
                u32::try_from(number(word)?).map_err(|_| invalid("D head word overflow"))?,
            );
            check(value.is_finite(), "source finite D head")?;
            Ok(value)
        })
        .collect::<Result<Vec<_>, ArenaError>>()?;
    check(
        selected_slot(&logits) == Some(slot),
        "source slot differs from original finite total_cmp head selection",
    )?;
    check(
        source_digest(s)? == text(&s["native_source_sha256"])?,
        "actual source native seal",
    )?;
    Ok(())
}
fn selected_slot(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
        .map(|(i, _)| i)
}
fn frozen_chain(w: &Value, s: &Value) -> Result<(), ArenaError> {
    let repair = &w["repair"];
    let recheck = &w["recheck"];
    let left = &w["repaired_endpoint"];
    let right = &w["counter_endpoint"];
    check(
        [repair, recheck, left, right].iter().all(|v| v.is_object()),
        "full frozen witness needs all original facts",
    )?;
    let seq = |v: &Value| number(&v["accepted"]["ticket"]["request_sequence"]);
    check(
        seq(s)? < seq(repair)?
            && seq(repair)? < seq(recheck)?
            && seq(recheck)? < seq(left)?
            && seq(recheck)? < seq(right)?
            && seq(left)? != seq(right)?
            && left["pair"] == right["pair"]
            && left["model"] == right["model"]
            && left["line"] == repair["moves"]
            && right["line"] == recheck["moves"],
        "Repair/C/Fresh causal chronology and full frozen pair",
    )?;
    let p = moves(&repair["moves"])?;
    let c = moves(&recheck["moves"])?;
    let counter = moves(&s["counter"])?;
    let fixed = usize::try_from(number(&s["divergence_ply"])?)
        .map_err(|_| invalid("attack ply"))?
        .checked_add(1)
        .ok_or_else(|| invalid("attack overflow"))?;
    check(
        p.len() == c.len()
            && p.first() == counter.first()
            && c.first() == counter.first()
            && p.get(..fixed) == counter.get(..fixed),
        "Repair preserves the original first move and attack prefix",
    )?;
    let changed = p
        .iter()
        .zip(&counter)
        .position(|(a, b)| a != b)
        .ok_or_else(|| invalid("actual P Repair did not change its response"))?;
    let c_changed = c
        .iter()
        .zip(&p)
        .position(|(a, b)| a != b)
        .ok_or_else(|| invalid("actual C continuation did not change its response"))?;
    check(
        changed >= fixed && changed % 2 == 0 && c_changed > changed && c_changed % 2 == 1,
        "P response then independent C response parity",
    )?;
    let a = wdl(&left["root_wdl_bits"])?;
    let b = wdl(&right["root_wdl_bits"])?;
    let expected = conditional_resolution(a, b);
    let reason = if expected == "unknown" {
        "equal frozen conditional expectations"
    } else {
        "actual Repair and C continuation; frozen model conditional comparison"
    };
    check(
        w["resolution"] == expected && w["reason"] == reason,
        "conditional comparison must use original FP32 W-L",
    )
}
fn conditional_resolution(left: [f32; 3], right: [f32; 3]) -> &'static str {
    // Both differences are FP32, exactly as compare_model_wdl; FP64 changes ties.
    let a = left[0] - left[2];
    let b = right[0] - right[2];
    if a > b {
        "refuted"
    } else if a < b {
        "supported"
    } else {
        "unknown"
    }
}
fn replay_lines(
    record: &SubmittedSourceFields<'_>,
    w: &Value,
    source: &Value,
) -> Result<(), ArenaError> {
    let replay: rules_replay::RoventSubmittedRulesReplay =
        serde_json::from_value(record.native.raw_submission["rules_replay"].clone())
            .map_err(|e| invalid(e.to_string()))?;
    let position = replay.original_query_position(&record.input.common_validation_projection())?;
    check(
        exact_rules_anchor(&position)? == record.input.rules_anchor,
        "divergence source exact Rules query anchor",
    )?;
    let mut lines = vec![&source["proposal"], &source["counter"]];
    for key in ["repair", "recheck"] {
        if !w[key].is_null() {
            lines.push(&w[key]["moves"]);
        }
    }
    for key in ["repaired_endpoint", "counter_endpoint"] {
        if !w[key].is_null() {
            lines.push(&w[key]["line"]);
        }
    }
    for line in lines {
        let mut endpoint = position.clone();
        for bits in moves(line)? {
            let movement = rz_contracts::pals::Move16::try_from_bits(bits)
                .and_then(|m| m.decode())
                .map_err(|e| invalid(e.to_string()))?;
            endpoint
                .make_move(BoardMove::try_from(movement).map_err(|e| invalid(e.to_string()))?)
                .map_err(|e| invalid(e.to_string()))?;
        }
    }
    Ok(())
}
fn verify_raw(record: &RoventTrainingRecordV2, head: &RoventHeadSource) -> Result<(), ArenaError> {
    let raw = &head.raw_receipt;
    fields(
        raw,
        &[
            "schema",
            "kind",
            "input_sha256",
            "anchor",
            "observed_sequence",
            "witness",
        ],
    )?;
    check(
        head.source == ACTUAL_SOURCE
            && head.actor_id == record.lineage.actor_id
            && head.binary_sha256.is_none()
            && head.profile_sha256.is_none()
            && raw["schema"] == HEAD_DOMAIN
            && raw["kind"] == "pc_divergence"
            && raw["input_sha256"] == record.input_sha256
            && record.input.sha256()? == record.input_sha256
            && raw["anchor"]
                == serde_json::to_value(&record.input.rules_anchor)
                    .map_err(|e| invalid(e.to_string()))?
            && head.anchor == record.input.rules_anchor
            && number(&raw["observed_sequence"])? == head.observed_sequence
            && head.observed_sequence >= record.input.capture_sequence
            && generation_digest(HEAD_DOMAIN, raw)? == head.raw_receipt_sha256,
        "original divergence input/head/anchor/collector seal",
    )?;
    let w = &raw["witness"];
    fields(
        w,
        &[
            "schema",
            "native_witness_sha256",
            "native_hash_kind",
            "event_sequence",
            "reason",
            "resolution",
            "target_known",
            "source",
            "repair",
            "recheck",
            "repaired_endpoint",
            "counter_endpoint",
        ],
    )?;
    let resolution = text(&w["resolution"])?;
    check(
        w["schema"] == WITNESS_DOMAIN
            && ["supported", "refuted", "unknown", "disputed"].contains(&resolution)
            && w["target_known"].as_bool() == Some(matches!(resolution, "supported" | "refuted")),
        "native witness exact resolution/known mask",
    )?;
    number(&w["event_sequence"])?;
    let reason = text(&w["reason"])?;
    check(
        !reason.is_empty() && reason.len() <= 4096 && !reason.chars().any(char::is_control),
        "bounded original witness reason",
    )?;
    let future = visible_record_evidence(record, &w["source"])?;
    if let Some(visible) = &future {
        check(
            head.observed_sequence >= visible.wire.native.retired_sequence,
            "collector observed divergence before original future visible retirement",
        )?;
    }
    let source_record = future
        .as_ref()
        .map(SubmittedSourceFields::from_visible)
        .unwrap_or_else(|| SubmittedSourceFields::from_record(record));
    let source = visible_source(&w["source"]);
    original_source(&source_record, source)?;
    if let Some(attack) = initial_attack(&w["source"]) {
        ranked_attack(record, attack, source)?;
        check(
            number(&attack["source_sequence"])? > 0
                && number(&attack["source_sequence"])?
                    < number(&w["source"]["visible_source_sequence"])?,
            "initial and visible private source sequence chronology",
        )?;
        check(
            combined_source_digest(&w["source"])? == text(&w["source"]["native_source_sha256"])?,
            "initial/visible combined native source seal",
        )?;
    }
    let original = &source_record.native.raw_submission["ticket"];
    let budget = &source["root_budget_sha256"];
    for (key, kind) in [("repair", "repair"), ("recheck", "counterexample")] {
        if !w[key].is_null() {
            accepted_line(&w[key], original, budget, kind)?;
        }
    }
    let native = &source_record.native.raw_retirement["root_admission"]["native_origin"];
    for key in ["repaired_endpoint", "counter_endpoint"] {
        if !w[key].is_null() {
            endpoint(&w[key], original, budget, native, &source["perspective"])?;
        }
    }
    let pair = match text(&w["native_hash_kind"])? {
        "frozen_pair" => {
            check(
                number(&w["event_sequence"])? > 0,
                "completed pair manager event sequence",
            )?;
            frozen_chain(w, source)?;
            true
        }
        "unresolved" => {
            check(
                resolution == "unknown" && w["target_known"] == false,
                "unresolved witness cannot be known",
            )?;
            false
        }
        _ => return Err(invalid("unrecognized native witness hash kind")),
    };
    check(
        witness_digest(w, pair)? == text(&w["native_witness_sha256"])?,
        "original composite witness native seal",
    )?;
    replay_lines(&source_record, w, source)
}
fn verify_target(
    record: &RoventTrainingRecordV2,
    head: &RoventHeadSource,
    target: &RoventDivergenceTargetV1,
) -> Result<(), ArenaError> {
    target.value(&record.input)?;
    let w = &head.raw_receipt["witness"];
    let s = visible_source(&w["source"]);
    let attack = initial_attack(&w["source"]);
    let slot_source = attack.unwrap_or(s);
    let slot = usize::try_from(number(&slot_source["divergence_index"])?)
        .map_err(|_| invalid("original slot overflow"))?;
    let mapping = record
        .input
        .attack_mapping
        .get(slot)
        .ok_or_else(|| invalid("original slot absent"))?;
    check(
        w["target_known"] == true
            && (attack.is_some() || mapping.selected_counter_local_index.is_some())
            && target.raw_evidence_sha256 == head.raw_receipt_sha256
            && target.observed_sequence == head.observed_sequence
            && usize::from(target.slot) == slot
            && u64::from(target.attack_ply) == number(&s["divergence_ply"])?
            && target.counter_question_id == text(&s["accepted"]["ticket"]["question_digest"])?
            && target.counter_native_view_sha256 == text(&s["accepted"]["ticket"]["view_digest"])?
            && u64::from(target.counter_local_index) == number(&s["counter_local_index"])?
            && target.counter_revision == number(&s["counter_revision"])?
            && target.repair_question_id
                == text(&w["repair"]["accepted"]["ticket"]["question_digest"])?
            && match target.status {
                GenerationDivergenceStatusV3::SupportedAfterRepair => {
                    w["resolution"] == "supported"
                }
                GenerationDivergenceStatusV3::RefutedByRepair => w["resolution"] == "refuted",
            },
        "known target differs from its original visible attack and actual owner witness",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranked_attack_uses_its_original_slot_and_explicit_first_move_framing() {
        // e2e4=12|(28<<6), e7e5=52|(36<<6), independent of Store helpers.
        let raw = json!({"accepted":{"seal_sha256":"03".repeat(32)},
            "root_budget_sha256":"07".repeat(32),"divergence_index":2,"divergence_ply":1,
            "first_move":1804,"proposal":[1804,2356]});
        let mut bytes = b"rz-rovent-pc-attack-ranked-source/1\0".to_vec();
        bytes.extend([3u8; 32]);
        bytes.extend([7u8; 32]);
        bytes.extend(2u16.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(1804u16.to_le_bytes());
        bytes.extend(2u64.to_le_bytes());
        bytes.extend(1804u16.to_le_bytes());
        bytes.extend(2356u16.to_le_bytes());
        assert_eq!(
            attack_digest(&raw).unwrap(),
            hex(Digest(Sha256::digest(bytes).into()))
        );
    }
    #[test]
    fn linked_source_seals_both_original_private_sequences_in_order() {
        let mut raw = json!({"visible_source":{"native_source_sha256":"0b".repeat(32)},
            "attack_ranked_source":{"native_attack_sha256":"0d".repeat(32),"source_sequence":17},
            "visible_source_sequence":29});
        let mut bytes = b"rz-rovent-pc-divergence-source-with-attack/1\0".to_vec();
        bytes.extend([11u8; 32]);
        bytes.extend([13u8; 32]);
        bytes.extend(29u64.to_le_bytes());
        bytes.extend(17u64.to_le_bytes());
        let expected = hex(Digest(Sha256::digest(bytes).into()));
        assert_eq!(combined_source_digest(&raw).unwrap(), expected);
        raw["visible_source_sequence"] = json!(17);
        raw["attack_ranked_source"]["source_sequence"] = json!(29);
        assert_ne!(combined_source_digest(&raw).unwrap(), expected);
    }

    #[test]
    fn origin_binary_framing_and_full_ticket_join_are_separate() {
        let d = |byte: u8| format!("{byte:02x}").repeat(32);
        let t = json!({"process_epoch":3,"game_generation":5,"root_generation":7,"request_sequence":11,
            "implementation_sha256":d(1),"question_digest":d(2),"view_sequence":13,"view_digest":d(3),
            "context_policy":d(4),"actual_tokens":17,"selected_records":1,"anchors":1,"length_bucket":8,
            "root_state":{"owner":19,"revision":2,"semantic":d(5)},
            "representation":{"input":d(6),"history":d(7),"candidates":d(8),"mask_positions":d(9),
                "model":d(10),"encoding":d(11),"precision":"Fp32","frozen_epoch":1,"record_revision":23,"role":"critic"},
            "deadline_clock":3,"deadline_tick":999});
        let mut raw = json!({"frame":{"process_epoch":3,"game_generation":5,"root_generation":7,
            "implementation_sha256":d(1),"sequence":29},"ticket":t,"output_digest":d(12),
            "kind":"local_accepted","peer":null,"seal_sha256":""});
        let mut bytes = b"rz-rovent-accepted-local-origin-seal/1\0".to_vec();
        for n in [29u64, 11, 3, 5, 7] {
            bytes.extend(n.to_le_bytes());
        }
        for byte in [1u8, 2, 3] {
            bytes.extend([byte; 32]);
        }
        bytes.extend(13u64.to_le_bytes());
        bytes.extend([12u8; 32]);
        for byte in [6u8, 7, 8, 9, 10, 11] {
            bytes.extend([byte; 32]);
        }
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(23u64.to_le_bytes());
        bytes.extend([1u8, 0]);
        bytes.extend(999u64.to_le_bytes());
        bytes.push(0);
        let expected = hex(Digest(Sha256::digest(bytes).into()));
        assert_eq!(origin_digest(&raw).unwrap(), expected);
        raw["seal_sha256"] = json!(expected);
        let original = raw["ticket"].clone();
        assert!(origin(&raw, &original, &json!(d(13)), "critic").is_ok());
        // Clock is deliberately absent from the native hash; full ticket joins
        // must still reject it rather than treating a matching hash as authority.
        raw["ticket"]["deadline_clock"] = json!(4);
        assert_eq!(
            origin_digest(&raw).unwrap(),
            text(&raw["seal_sha256"]).unwrap()
        );
        assert!(origin(&raw, &original, &json!(d(13)), "critic").is_err());
    }

    #[test]
    fn source_head_uses_total_f32_order_and_earliest_tie() {
        assert_eq!(selected_slot(&[-0.0, 0.0]), Some(1));
        assert_eq!(selected_slot(&[0.0, 0.0]), Some(0));
        assert_eq!(selected_slot(&[1.0, 0.0, 1.0]), Some(0));
        assert_eq!(selected_slot(&[]), None);
    }
    #[test]
    fn conditional_comparison_keeps_fp32_ties_and_direction() {
        assert_eq!(
            conditional_resolution(
                [0.75, 0.25 - 2f32.powi(-26), 2f32.powi(-26)],
                [0.75, 0.25, 0.0]
            ),
            "unknown"
        );
        assert_eq!(
            conditional_resolution([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            "refuted"
        );
        assert_eq!(
            conditional_resolution([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
            "supported"
        );
    }
    #[test]
    fn unresolved_framing_preserves_presence_and_utf8_byte_length() {
        let zero = "00".repeat(32);
        let one = "01".repeat(32);
        let raw = json!({"source":{"native_source_sha256":zero},"repair":{"native_line_sha256":one},
            "recheck":null,"repaired_endpoint":null,"counter_endpoint":null,"event_sequence":7,
            "reason":"미완료","resolution":"unknown"});
        let mut bytes = b"rz-rovent-pc-divergence-unresolved/1\0".to_vec();
        bytes.extend([0u8; 32]);
        bytes.push(1);
        bytes.extend([1u8; 32]);
        bytes.extend([0, 0, 0]);
        bytes.extend(7u64.to_le_bytes());
        bytes.extend(9u64.to_le_bytes());
        bytes.extend("미완료".as_bytes());
        assert_eq!(
            witness_digest(&raw, false).unwrap(),
            hex(Digest(Sha256::digest(bytes).into()))
        );
        assert!(witness_digest(&raw, true).is_err());
    }
    #[test]
    fn unknown_equal_pair_and_unresolved_have_distinct_native_domains() {
        let sha = "02".repeat(32);
        let raw = json!({"source":{"native_source_sha256":sha},"repair":{"native_line_sha256":sha},
            "recheck":{"native_line_sha256":sha},"repaired_endpoint":{"native_endpoint_sha256":sha},
            "counter_endpoint":{"native_endpoint_sha256":sha},"event_sequence":5,"resolution":"unknown",
            "reason":"equal frozen conditional expectations"});
        let mut bytes = b"rz-rovent-pc-divergence-witness/1\0".to_vec();
        bytes.extend([2u8; 160]);
        bytes.extend(5u64.to_le_bytes());
        bytes.push(2);
        assert_eq!(
            witness_digest(&raw, true).unwrap(),
            hex(Digest(Sha256::digest(bytes).into()))
        );
        assert_ne!(
            witness_digest(&raw, true).unwrap(),
            witness_digest(&raw, false).unwrap()
        );
    }
    #[test]
    fn exact_evidence_roster_and_wdl_words_fail_closed() {
        assert!(fields(&json!({"a":1,"b":2}), &["a"]).is_err());
        assert!(wdl(&json!([f32::NAN.to_bits(), 0, 0])).is_err());
        assert!(wdl(&json!([0, 0, 0])).is_err());
        assert!(sha(&json!("AB".repeat(32))).is_err());
    }
}
