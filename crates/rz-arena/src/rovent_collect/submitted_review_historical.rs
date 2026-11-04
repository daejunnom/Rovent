//! 원 raw/3의 selected review를 Rules로 재생하고 기존 digest를 재계산한다.
//! 공개 wire의 숫자 ID는 역사 메타데이터일 뿐이며, 이 계산은 live Store pin,
//! ReviewView, native 실행·수락·물리 완료나 학습 target 권한을 발급하지 않는다.
use super::*;
use rz_experiments::rovent_corpus_v2::RoventTrainingRecordV2;
use rz_experiments::rovent_submitted_input::{
    ROVENT_NATIVE_RAW_V3_DOMAIN, ROVENT_TRAINING_RECORD_V2_DOMAIN,
};
use rz_search::cpu::CpuScoreScope;
use rz_search::pals::engine::{RecordKind, RoleQueryPurpose, RoleRecord};
use rz_search::pals::review_context::{
    ReviewAnchor, ReviewAnchorMode, ReviewQueryInput, ReviewRecordMetadata, ReviewSelectionPolicy,
    historical::{HistoricalReviewDescriptor, calculate_historical_review_digests},
};
use rz_search::pals::store::StateId;

fn exact_fields(value: &Value, fields: &[&str]) -> bool {
    value.as_object().is_some_and(|object| {
        object.len() == fields.len() && fields.iter().all(|field| object.contains_key(*field))
    })
}

fn state(value: u64) -> Result<StateId, ArenaError> {
    usize::try_from(value)
        .map(StateId)
        .map_err(|_| invalid("submitted review historical StateId width"))
}

fn movement(value: u16) -> Result<BoardMove, ArenaError> {
    let shared = rz_contracts::pals::Move16::try_from_bits(value)
        .and_then(|value| value.decode())
        .map_err(|error| invalid(error.to_string()))?;
    BoardMove::try_from(shared).map_err(|error| invalid(error.to_string()))
}

fn moves(values: &[u16]) -> Result<Vec<BoardMove>, ArenaError> {
    values.iter().copied().map(movement).collect()
}

fn purpose(value: &str) -> Result<RoleQueryPurpose, ArenaError> {
    match value {
        "propose_policy" => Ok(RoleQueryPurpose::ProposePolicy),
        "reply_policy" => Ok(RoleQueryPurpose::ReplyPolicy),
        "repair_policy" => Ok(RoleQueryPurpose::RepairPolicy),
        "divergence_policy" => Ok(RoleQueryPurpose::DivergencePolicy),
        "value_fresh" => Ok(RoleQueryPurpose::ValueFresh),
        "task_policy" => Ok(RoleQueryPurpose::TaskPolicy),
        _ => Err(invalid("submitted review historical query purpose")),
    }
}

fn scope(value: Option<&str>) -> Result<Option<CpuScoreScope>, ArenaError> {
    match value {
        None => Ok(None),
        Some("FrontierOnly") => Ok(Some(CpuScoreScope::FrontierOnly)),
        Some("CompletedIteration") => Ok(Some(CpuScoreScope::CompletedIteration)),
        Some("RulesTerminal") => Ok(Some(CpuScoreScope::RulesTerminal)),
        _ => Err(invalid("submitted review original score scope")),
    }
}

fn sha(value: &[u8; 32]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The caller separately validates the original publication/native chain.
/// This validates historical projections, never reconstructs that capability.
pub(super) fn verify_historical_submitted_review(
    record: &RoventTrainingRecordV2,
) -> Result<(), ArenaError> {
    check(
        record.schema == ROVENT_TRAINING_RECORD_V2_DOMAIN,
        "historical submitted record family",
    )?;
    verify_historical_submitted_review_fields(&record.input, &record.input_sha256, &record.native)
}

/// 원 input/native 필드만 검증하는 공통 계산이다. 미래 증거를 가짜 전체
/// record/target/lineage로 감싸거나 새 publication으로 봉인하지 않는다.
pub(super) fn verify_historical_submitted_review_fields(
    input: &rz_experiments::rovent_submitted_input::RoventSubmittedInputV1,
    input_sha256: &str,
    native: &RoventNativeLifecycle,
) -> Result<(), ArenaError> {
    verify_historical_submitted_review_fields_for_domain(
        input,
        input_sha256,
        native,
        ROVENT_NATIVE_RAW_V3_DOMAIN,
    )
}

/// Checks the original submitted input without rewriting its raw schema.
/// Callers must first validate their exact native domain and root authority.
pub(super) fn verify_historical_submitted_review_fields_for_domain(
    input: &rz_experiments::rovent_submitted_input::RoventSubmittedInputV1,
    input_sha256: &str,
    native: &RoventNativeLifecycle,
    raw_domain: &str,
) -> Result<(), ArenaError> {
    input.validate()?;
    let raw = &native.raw_submission;
    check(
        raw["schema"] == raw_domain
            && input_sha256 == input.sha256()?
            && raw["input_sha256"] == input_sha256,
        "submitted review requires original raw3/submitted-input family and seal",
    )?;
    let raw_review = &raw["submitted_review"];
    check(
        exact_fields(
            raw_review,
            &[
                "schema",
                "sequence",
                "generation",
                "root_state",
                "policy",
                "input_sha256",
                "view_seal_sha256",
                "record_scope_sha256",
                "anchor_states",
                "source_anchors",
                "record_metadata",
            ],
        ) && exact_fields(
            &raw_review["policy"],
            &[
                "mode",
                "token_budget",
                "max_records",
                "max_anchors",
                "required_bundle",
            ],
        ),
        "submitted review exact historical evidence/policy fields",
    )?;
    let review: submitted_v3::RoventSubmittedReviewEvidence =
        serde_json::from_value(raw_review.clone()).map_err(|error| invalid(error.to_string()))?;
    let selected = &input.selected_view;
    let ticket = &raw["ticket"];
    let root = &native.raw_retirement["root_admission"];
    let raw_metadata = raw_review["record_metadata"]
        .as_array()
        .ok_or_else(|| invalid("submitted review original record metadata"))?;
    check(
        review.schema == submitted_v3::ROVENT_SUBMITTED_REVIEW_EVIDENCE
            && review.sequence > 0
            && ticket["view_sequence"].as_u64() == Some(review.sequence)
            && review.generation == native.root_generation
            && raw["root_generation"].as_u64() == Some(review.generation)
            && root["store_root_state"].as_u64() == Some(review.root_state)
            && review.view_seal_sha256 == native.view_digest
            && ticket["view_digest"] == review.view_seal_sha256
            && raw["view_digest"] == review.view_seal_sha256
            && !review.anchor_states.is_empty()
            && review.anchor_states.len() == selected.anchors.len()
            && review.anchor_states.len() <= 4
            && review.anchor_states[0] == review.root_state
            && review.record_metadata.len() == selected.records.len()
            && review.source_anchors.len() == selected.records.len()
            && review.record_metadata.len() <= 64
            && raw_metadata.len() == review.record_metadata.len(),
        "submitted review original root/sequence/anchor/record alignment",
    )?;
    for metadata in raw_metadata {
        // Option<T> serde fields alone allow an omitted key. The raw3 format
        // requires explicit null as well, so missing original scope/value must
        // not be reconstructed from NN features or by trial hash matching.
        check(
            exact_fields(
                metadata,
                &[
                    "local_index",
                    "record_revision",
                    "parent_revision",
                    "supersedes_revision",
                    "origin_state",
                    "source_state",
                    "source_revision",
                    "source_line_offset",
                    "original_first_move",
                    "value",
                    "completed_depth",
                    "score_scope",
                    "critical",
                ],
            ),
            "submitted review exact thirteen original metadata fields",
        )?;
    }
    let replay: rules_replay::RoventSubmittedRulesReplay =
        serde_json::from_value(raw["rules_replay"].clone())
            .map_err(|error| invalid(error.to_string()))?;
    check(
        replay.schema == rules_replay::ROVENT_SUBMITTED_RULES_REPLAY
            && replay.selected_anchors.len() == review.anchor_states.len(),
        "submitted review selected Rules replay alignment",
    )?;
    let actual_root = replay.actual_root.restore()?;
    let positions = replay
        .selected_anchors
        .iter()
        .map(rules_replay::RoventRulesPositionReplay::restore)
        .collect::<Result<Vec<_>, _>>()?;
    check(
        replay.actual_root.moves == input.actual_history
            && actual_root.snapshot().same_state(&positions[0].snapshot()),
        "submitted review exact actual Rules root/history",
    )?;
    let anchors = positions
        .iter()
        .zip(&review.anchor_states)
        .zip(&selected.anchors)
        .map(|((position, id), original)| {
            check(
                exact_rules_anchor(position)? == *original,
                "submitted review exact selected Rules anchor",
            )?;
            Ok(ReviewAnchor {
                state: state(*id)?,
                snapshot: position.snapshot(),
            })
        })
        .collect::<Result<Vec<_>, ArenaError>>()?;
    let source_anchors = review
        .source_anchors
        .iter()
        .zip(&review.record_metadata)
        .map(|(replay, original)| {
            Ok(ReviewAnchor {
                state: state(original.source_state)?,
                snapshot: replay.restore()?.snapshot(),
            })
        })
        .collect::<Result<Vec<_>, ArenaError>>()?;
    let public_revision = ticket["representation"]["record_revision"]
        .as_u64()
        .ok_or_else(|| invalid("submitted review original public revision"))?;
    let mut records = Vec::with_capacity(selected.records.len());
    let mut metadata = Vec::with_capacity(selected.records.len());
    for (slot, (row, original)) in selected
        .records
        .iter()
        .zip(&review.record_metadata)
        .enumerate()
    {
        let anchor = anchors
            .get(usize::from(row.anchor))
            .ok_or_else(|| invalid("submitted review original record anchor"))?;
        let original_scope = scope(original.score_scope.as_deref())?;
        check(
            usize::from(original.local_index) == slot
                && original.record_revision > 0
                && original.record_revision <= public_revision
                && original.source_revision == original.record_revision
                && original.origin_state == anchor.state.0 as u64
                && row.record_id
                    == format!("record-{}-{}", review.generation, original.source_revision)
                && row.observed_sequence == input.capture_sequence
                && row.completed_iteration
                    == (original_scope == Some(CpuScoreScope::CompletedIteration)),
            "submitted review original record identity/revision/scope projection",
        )?;
        records.push(RoleRecord {
            revision: original.record_revision,
            parent_revision: original.parent_revision,
            supersedes_revision: original.supersedes_revision,
            origin_state: state(original.origin_state)?,
            kind: match row.kind {
                GenerationRecordKindV3::Proposal => RecordKind::Proposal,
                GenerationRecordKindV3::Counter => RecordKind::Counterexample,
                GenerationRecordKindV3::Repair => RecordKind::Repair,
                GenerationRecordKindV3::Continuation => RecordKind::CpuVerification,
            },
            line: moves(&row.moves)?,
            value: original.value,
            completed_depth: original.completed_depth,
            score_scope: original_scope,
            // Observation ownership is not part of historical view framing.
            // No live Store handle is reconstructed from this wire record.
            cpu_observation: None,
            perspective: if row.perspective == 1 {
                rz_position::Color::White
            } else {
                rz_position::Color::Black
            },
            critical: original.critical,
        });
        metadata.push(ReviewRecordMetadata {
            anchor: row.anchor,
            parent: row.parent,
            supersedes: row.supersedes,
            source_state: state(original.source_state)?,
            source_revision: original.source_revision,
            source_line_offset: original.source_line_offset,
            original_first_move: original.original_first_move.map(movement).transpose()?,
        });
    }
    for mapping in &input.attack_mapping {
        if let (Some(index), Some(revision)) = (
            mapping.selected_counter_local_index,
            mapping.selected_counter_revision,
        ) {
            check(
                review
                    .record_metadata
                    .get(usize::from(index))
                    .is_some_and(|source| source.source_revision == revision),
                "submitted attack points at another original selected counter revision",
            )?;
        }
    }
    let q = &selected.query;
    check(
        q.anchors == [0; 3],
        "historical ReviewQueryInput uses exact root anchor zero",
    )?;
    let query = ReviewQueryInput {
        purpose: purpose(&q.purpose)?,
        prefix: moves(&q.prefix)?,
        proposal: moves(&q.proposal)?,
        counterexample: q.counter_present.then(|| moves(&q.counter)).transpose()?,
        legal: moves(&input.legal_moves)?,
        divergences: input
            .attack_mapping
            .iter()
            .map(|slot| usize::from(slot.attack_ply))
            .collect(),
    };
    let policy = ReviewSelectionPolicy {
        mode: match review.policy.mode.as_str() {
            "current_root_only" => ReviewAnchorMode::CurrentRootOnly,
            "anchored_compact" => ReviewAnchorMode::AnchoredCompact,
            _ => return Err(invalid("submitted review original selection mode")),
        },
        token_budget: review.policy.token_budget,
        max_records: review.policy.max_records,
        max_anchors: review.policy.max_anchors,
        required_bundle: review.policy.required_bundle,
    };
    let tokens = records
        .iter()
        .map(|row| row.line.len())
        .chain([query.prefix.len(), query.proposal.len(), q.counter.len()])
        .try_fold(0usize, usize::checked_add)
        .and_then(|count| u16::try_from(count).ok())
        .ok_or_else(|| invalid("submitted review actual repeated token count overflow"))?;
    check(
        selected.token_budget == policy.token_budget
            && ticket["actual_tokens"].as_u64() == Some(u64::from(tokens))
            && ticket["selected_records"].as_u64() == Some(records.len() as u64)
            && ticket["anchors"].as_u64() == Some(anchors.len() as u64)
            && ticket["length_bucket"].as_u64() == Some(u64::from(selected.length_bucket)),
        "submitted review original token/allocation metadata",
    )?;
    let calculated = calculate_historical_review_digests(HistoricalReviewDescriptor {
        sequence: review.sequence,
        generation: review.generation,
        root_state: state(review.root_state)?,
        policy,
        anchors: &anchors,
        source_anchors: &source_anchors,
        records: &records,
        metadata: &metadata,
        query: &query,
        tokens,
        bucket: selected.length_bucket,
    })
    .map_err(|error| invalid(error.to_string()))?;
    check(
        sha(&calculated.input_digest) == review.input_sha256
            && sha(&calculated.view_seal_digest) == review.view_seal_sha256
            && sha(&calculated.record_scope_digest) == review.record_scope_sha256
            && sha(&calculated.policy_digest) == selected.context_policy_sha256
            && ticket["context_policy"] == selected.context_policy_sha256,
        "submitted review original input/seal/record-scope/policy digest mismatch",
    )
}
