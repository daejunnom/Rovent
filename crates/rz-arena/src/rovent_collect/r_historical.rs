//! 원 R publication의 Rules·수치 입력을 재검증하는 미등록 순수 소비자다.
//! 원 worker·Bound·Store·현재 native 권한을 복원하지 않고 모델을 평가하지 않는다.
//! 파일 출처와 parameter identity는 publication 부모가 원 bytes에서 따로 검사한다.
use super::*;
use rz_experiments::rovent_r_readout::*;
use rz_search::cpu::CpuScoreScope;
use rz_search::cpu_value::nnuec::{NnuecConfig, rules, v2};
use rz_search::pals::engine::{RecordKind, RoleQueryPurpose, RoleRecord};
use rz_search::pals::review_context::{
    ReviewAnchor, ReviewAnchorMode, ReviewQueryInput, ReviewRecordMetadata, ReviewSelectionPolicy,
    historical::{HistoricalReviewDescriptor, calculate_historical_review_digests},
};
use rz_search::pals::store::{ObservationId, StateId};

#[derive(Clone, Debug)]
pub struct RoventHistoricalRReadoutEvidence {
    pub native_admission_sha256: String,
    pub collection_source_sha256: String,
    pub collection_runtime_library_sha256: String,
    pub startup_input_lock_sha256: String,
    pub root_descriptor: rz_experiments::RoventRootDescriptor,
    pub nnuec_semantics: String,
    pub nnuec_parameter_sha256: String,
    pub nnuec_training_state: rz_search::cpu_value::CpuTrainingState,
    pub nnuec_weights_file_sha256: String,
    pub native_input_sha256: String,
    pub semantic_input_sha256: String,
    pub retirement_sha256: String,
    /// 실제 Rules 궤적을 재생했을 때만 채워진다. 모델 예측 WDL은 이 값이 아니다.
    pub rules_wdl: Option<GenerationWdlTargetV3>,
}

fn same_wire<A: serde::Serialize, B: serde::Serialize>(a: &A, b: &B) -> Result<bool, ArenaError> {
    Ok(serde_json::to_value(a).map_err(|e| invalid(e.to_string()))?
        == serde_json::to_value(b).map_err(|e| invalid(e.to_string()))?)
}

fn parse_sha(value: &str) -> Result<[u8; 32], ArenaError> {
    check(generation_sha(value), "R historical SHA syntax")?;
    let mut result = [0; 32];
    for (slot, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let nibble = |v: u8| if v <= b'9' { v - b'0' } else { v - b'a' + 10 };
        result[slot] = nibble(pair[0]) * 16 + nibble(pair[1]);
    }
    Ok(result)
}
fn state(value: u64) -> Result<StateId, ArenaError> {
    usize::try_from(value)
        .map(StateId)
        .map_err(|_| invalid("R historical StateId width"))
}
fn move16(value: u16) -> Result<BoardMove, ArenaError> {
    let movement = rz_contracts::pals::Move16::try_from_bits(value)
        .and_then(|value| value.decode())
        .map_err(|e| invalid(e.to_string()))?;
    BoardMove::try_from(movement).map_err(|e| invalid(e.to_string()))
}
fn moves(values: &[u16]) -> Result<Vec<BoardMove>, ArenaError> {
    values.iter().copied().map(move16).collect()
}
fn color(value: &str) -> Result<rz_position::Color, ArenaError> {
    match value {
        "white" => Ok(rz_position::Color::White),
        "black" => Ok(rz_position::Color::Black),
        _ => Err(invalid("R historical raw perspective")),
    }
}
fn restore(value: &RoventRRulesEvidence) -> Result<Position, ArenaError> {
    check(
        generation_digest(rules_replay::ROVENT_RULES_POSITION_REPLAY, &value.replay)?
            == value.replay_sha256,
        "R historical replay seal",
    )?;
    let replay: rules_replay::RoventRulesPositionReplay =
        serde_json::from_value(value.replay.clone()).map_err(|e| invalid(e.to_string()))?;
    check(replay.anchor == value.anchor, "R historical replay anchor")?;
    replay.restore()
}
fn purpose(value: &str) -> Result<RoleQueryPurpose, ArenaError> {
    match value {
        "ProposePolicy" => Ok(RoleQueryPurpose::ProposePolicy),
        "ReplyPolicy" => Ok(RoleQueryPurpose::ReplyPolicy),
        "RepairPolicy" => Ok(RoleQueryPurpose::RepairPolicy),
        "DivergencePolicy" => Ok(RoleQueryPurpose::DivergencePolicy),
        "ValueFresh" => Ok(RoleQueryPurpose::ValueFresh),
        "TaskPolicy" => Ok(RoleQueryPurpose::TaskPolicy),
        _ => Err(invalid("R historical query purpose")),
    }
}
fn record(value: &RoventRContextRecord) -> Result<RoleRecord, ArenaError> {
    check(
        value.record_id == format!("public-record-{}", value.record_revision),
        "R original record ID projection",
    )?;
    Ok(RoleRecord {
        revision: value.record_revision,
        parent_revision: value.parent_revision,
        supersedes_revision: value.supersedes_revision,
        origin_state: state(value.origin_state)?,
        kind: match value.kind {
            0 => RecordKind::Proposal,
            1 => RecordKind::Counterexample,
            2 => RecordKind::Repair,
            3 => RecordKind::CpuVerification,
            _ => return Err(invalid("R historical record kind")),
        },
        line: moves(&value.moves)?,
        value: value.value,
        completed_depth: value.completed_depth,
        score_scope: match value.score_scope.as_deref() {
            None => None,
            Some("FrontierOnly") => Some(CpuScoreScope::FrontierOnly),
            Some("CompletedIteration") => Some(CpuScoreScope::CompletedIteration),
            Some("RulesTerminal") => Some(CpuScoreScope::RulesTerminal),
            _ => return Err(invalid("R historical score scope")),
        },
        cpu_observation: value
            .cpu_observation
            .map(|v| {
                usize::try_from(v)
                    .map(ObservationId)
                    .map_err(|_| invalid("R historical observation width"))
            })
            .transpose()?,
        perspective: color(&value.record_perspective)?,
        critical: value.critical,
    })
}

/// 원 publication의 무결성과 별도로 전체 Rules 이력과 입력을 재계산한다.
/// 원 worker의 accepted-use 또는 whole-cost는 이 함수에서도 생성하지 않는다.
pub fn verify_historical_r_readout(
    row: &RoventRReadoutRecord,
    manifest: &RoventRReadoutManifest,
    original_lock: &rz_experiments::RoventInputLock,
    config: &NnuecConfig,
) -> Result<RoventHistoricalRReadoutEvidence, ArenaError> {
    row.validate(manifest)?;
    original_lock.validate()?;
    config.validate().map_err(|e| invalid(e.to_string()))?;
    row.root_descriptor.verify_against(original_lock)?;
    let root: historical_scope::RoventRootAdmissionEvidence =
        serde_json::from_value(row.native_admission.clone()).map_err(|e| invalid(e.to_string()))?;
    let origin = &root.native_origin;
    let authority = &row.question.job.authority;
    let source = rz_experiments::rovent_source_aggregate_sha256(
        &origin.source_commit,
        &origin.source_components,
    )?;
    let execution = &original_lock.manifest;
    check(
        root.schema == "rz-rovent-collection-root-admission/1"
            && origin.schema == "rz-rovent-collection-native-origin/1"
            && origin.purpose == "collection"
            && generation_digest(&origin.schema, &row.native_admission["native_origin"])?
                == root.native_origin_sha256
            && root.process_epoch == authority.process_epoch
            && origin.process_epoch == root.process_epoch
            && root.game_generation == authority.game_generation
            && root.root_generation == authority.root_generation
            && root.implementation_sha256 == authority.implementation_sha256
            && origin.input_lock_sha256 == original_lock.canonical_sha256
            && row.startup_input_lock_sha256 == original_lock.canonical_sha256
            && source == origin.source_sha256
            && source == manifest.collection.source_sha256
            && origin.source_commit == manifest.collection.source_commit
            && origin.source_commit == execution.engine.source_commit
            && origin.source_components == execution.engine.source_components
            && origin.runtime_sha256_domain == "loaded_runtime_library_file_sha256"
            && origin.runtime_sha256 == origin.runtime_library_sha256
            && origin.runtime_sha256 == manifest.collection.runtime_sha256
            && origin.model_semantics_sha256 == manifest.collection.model_semantics_sha256
            && root.root_descriptor == row.root_descriptor
            && root.root_descriptor_sha256 == row.root_descriptor_sha256
            && root.root_budget_sha256 == row.root_descriptor.root_budget_sha256
            && root.deadline_clock == row.root_descriptor.input.clock.hard.clock_epoch
            && root.deadline_tick == row.root_descriptor.input.clock.hard.at_ns
            && root.root_moves == row.root_descriptor.input.allowed_root_moves
            && root.root_state.semantic == row.root_descriptor.input.state_and_history_sha256
            && origin.mode == format!("{:?}", execution.engine.mode)
            && origin.teacher_profiles == execution.engine.teachers,
        "R historical root/source/runtime/full-lock join",
    )?;
    let nnuec = execution
        .engine
        .nnuec
        .as_ref()
        .ok_or_else(|| invalid("R original lock has no NNUEC"))?;
    check(
        nnuec.semantics == row.bound.model_semantics
            && nnuec.semantics == v2::SEMANTICS
            && nnuec.weights.sha256 == row.bound.weights_file_sha256,
        "R original NNUEC semantics/raw file pin",
    )?;
    let input: v2::ReferenceInput = rz_experiments::decode_json(&row.input.reference_input_json)?;
    let anchors = row
        .bound
        .context_anchors
        .iter()
        .map(restore)
        .collect::<Result<Vec<_>, _>>()?;
    let source_anchors = row
        .bound
        .context_source_anchors
        .iter()
        .map(restore)
        .collect::<Result<Vec<_>, _>>()?;
    let local = restore(&row.bound.local_anchor)?;
    let current = restore(&row.bound.current_rules)?;
    check(
        !anchors.is_empty()
            && anchors.len() == row.bound.context_anchor_states.len()
            && source_anchors.len() == row.bound.context_records.len()
            && exact_rules_anchor(&anchors[0])?.rules_state_sha256 == root.root_state.semantic,
        "R historical root/selected/source anchor alignment",
    )?;
    let records = row
        .bound
        .context_records
        .iter()
        .map(record)
        .collect::<Result<Vec<_>, _>>()?;
    let metadata = row
        .bound
        .context_records
        .iter()
        .map(|v| {
            Ok(ReviewRecordMetadata {
                anchor: v.anchor,
                parent: v.parent,
                supersedes: v.supersedes,
                source_state: state(v.source_state)?,
                source_revision: v.source_revision,
                source_line_offset: v.source_line_offset,
                original_first_move: v.original_first_move.map(move16).transpose()?,
            })
        })
        .collect::<Result<Vec<_>, ArenaError>>()?;
    let anchors = anchors
        .iter()
        .zip(&row.bound.context_anchor_states)
        .map(|(p, s)| {
            Ok(ReviewAnchor {
                state: state(*s)?,
                snapshot: p.snapshot(),
            })
        })
        .collect::<Result<Vec<_>, ArenaError>>()?;
    let source_anchors = source_anchors
        .iter()
        .zip(&row.bound.context_records)
        .map(|(p, r)| {
            Ok(ReviewAnchor {
                state: state(r.source_state)?,
                snapshot: p.snapshot(),
            })
        })
        .collect::<Result<Vec<_>, ArenaError>>()?;
    let q = &row.bound.context_query;
    let query = ReviewQueryInput {
        purpose: purpose(&q.purpose)?,
        prefix: moves(&q.prefix)?,
        proposal: moves(&q.proposal)?,
        counterexample: q.counterexample.as_deref().map(moves).transpose()?,
        legal: moves(&q.legal)?,
        divergences: q.divergences.iter().copied().map(usize::from).collect(),
    };
    let p = &row.bound.selection_policy;
    let policy = ReviewSelectionPolicy {
        mode: match p.mode.as_str() {
            "current_root_only" => ReviewAnchorMode::CurrentRootOnly,
            "anchored_compact" => ReviewAnchorMode::AnchoredCompact,
            _ => return Err(invalid("R historical selection mode")),
        },
        token_budget: p.token_budget,
        max_records: p.max_records,
        max_anchors: p.max_anchors,
        required_bundle: p.required_bundle,
    };
    let digests = calculate_historical_review_digests(HistoricalReviewDescriptor {
        sequence: row.bound.sequence,
        generation: row.bound.generation,
        root_state: state(row.bound.root_state)?,
        policy,
        anchors: &anchors,
        source_anchors: &source_anchors,
        records: &records,
        metadata: &metadata,
        query: &query,
        tokens: row.bound.actual_tokens,
        bucket: row.bound.length_bucket,
    })
    .map_err(|e| invalid(e.to_string()))?;
    check(
        digests.input_digest == parse_sha(&row.bound.full_view_sha256)?
            && digests.view_seal_digest == parse_sha(&row.bound.view_seal_sha256)?
            && digests.record_scope_digest == parse_sha(&row.bound.record_scope_sha256)?
            && digests.policy_digest == parse_sha(&row.bound.policy_sha256)?,
        "R exact original historical view digests",
    )?;
    check(
        input.schema == v2::INPUT_SCHEMA
            && input.board_features == rules::reference_board_features(&current)
            && input.side_to_move
                == if current.side_to_move() == rz_position::Color::White {
                    "white"
                } else {
                    "black"
                }
            && input.context_lines.len() == records.len()
            && records.len() <= config.max_lines
            && v2::context_sha256(&input.context_lines, &input.context_edges)
                .map_err(|e| invalid(e.to_string()))?
                == row.bound.context_sha256,
        "R exact Rules B/context numeric representation",
    )?;
    let mut edges = Vec::new();
    for (index, ((record, meta), line)) in records
        .iter()
        .zip(&metadata)
        .zip(&input.context_lines)
        .enumerate()
    {
        check(
            record.line.len() <= config.max_plies,
            "R original model context plies",
        )?;
        let position = Position::from_snapshot(
            &anchors[usize::from(meta.anchor)].snapshot,
            rz_position::PositionLimits {
                max_history_positions: 16_385,
                ..Default::default()
            },
        )
        .map_err(|e| invalid(e.to_string()))?;
        let expected = rules::reference_line_from_rules(&position, &record.line)
            .map_err(|e| invalid(e.to_string()))?;
        // The original first move may precede a rebased suffix; transform its
        // token through the original source anchor, not the selected anchor.
        let original_first = if let Some(m) = meta.original_first_move {
            let source = Position::from_snapshot(
                &source_anchors[index].snapshot,
                rz_position::PositionLimits {
                    max_history_positions: 16_385,
                    ..Default::default()
                },
            )
            .map_err(|e| invalid(e.to_string()))?;
            Some(
                rules::reference_line_from_rules(&source, &[m])
                    .map_err(|e| invalid(e.to_string()))?
                    .moves[0],
            )
        } else {
            None
        };
        check(
            line.id == index as u64
                && line.kind == row.bound.context_records[index].kind
                && line.anchor_features == expected.anchor_features
                && line.perspective == expected.perspective
                && same_wire(&line.moves, &expected.moves)?
                && same_wire(&line.original_first_move, &original_first)?,
            "R exact context line/source-first numerical input",
        )?;
        for (parent, kind) in [(meta.parent, 0), (meta.supersedes, 1)] {
            if let Some(parent) = parent {
                edges.push(v2::ReferenceEdge {
                    source: u64::from(parent),
                    target: index as u64,
                    kind,
                });
            }
        }
    }
    check(
        same_wire(&input.context_edges, &edges)?,
        "R exact directed input relations",
    )?;
    let local_trace: rules_replay::RoventRulesPositionReplay =
        serde_json::from_value(row.bound.local_anchor.replay.clone())
            .map_err(|e| invalid(e.to_string()))?;
    let current_trace: rules_replay::RoventRulesPositionReplay =
        serde_json::from_value(row.bound.current_rules.replay.clone())
            .map_err(|e| invalid(e.to_string()))?;
    check(
        local_trace.origin == current_trace.origin
            && local_trace.completeness == current_trace.completeness
            && local_trace.start_fen == current_trace.start_fen
            && current_trace.moves.starts_with(&local_trace.moves),
        "R original local/current history prefix",
    )?;
    let suffix = moves(&current_trace.moves[local_trace.moves.len()..])?;
    check(
        suffix.len() <= config.max_plies,
        "R original model local plies",
    )?;
    let expected =
        rules::reference_line_from_rules(&local, &suffix).map_err(|e| invalid(e.to_string()))?;
    check(
        same_wire(&input.local_line, &expected)?,
        "R exact Rules local input",
    )?;
    let out = &row.numeric;
    check(
        out.board_accumulators_bits
            .iter()
            .all(|v| v.len() == config.board_width)
            && out.context_accumulator_bits.len() == config.context_width
            && out.local_accumulator_bits.len() == config.local_width
            && out.hidden_bits.len() == config.hidden_width,
        "R original numeric output dimension declaration",
    )?;
    if let Some(wdl) = &row.wdl {
        // Shared extraction of the existing P/C Rules replay function is the
        // only requested registered-source prerequisite. No second Rules codec.
        rules_replay::verify_rules_wdl_replay(
            &wdl.raw_receipt,
            &row.lineage.game_id,
            &row.bound.current_rules.anchor,
            &wdl.target,
        )?;
    }
    Ok(RoventHistoricalRReadoutEvidence {
        native_admission_sha256: row.native_admission_sha256.clone(),
        collection_source_sha256: source,
        collection_runtime_library_sha256: origin.runtime_library_sha256.clone(),
        startup_input_lock_sha256: original_lock.canonical_sha256.clone(),
        root_descriptor: row.root_descriptor.clone(),
        nnuec_semantics: row.bound.model_semantics.clone(),
        nnuec_parameter_sha256: row.bound.parameter_sha256.clone(),
        nnuec_training_state: serde_json::from_value(row.bound.training_state.clone())
            .map_err(|e| invalid(e.to_string()))?,
        nnuec_weights_file_sha256: row.bound.weights_file_sha256.clone(),
        native_input_sha256: row.input.native_input_sha256.clone(),
        semantic_input_sha256: row.input.semantic_input_sha256.clone(),
        retirement_sha256: row.retirement_sha256.clone(),
        rules_wdl: row.wdl.as_ref().map(|v| v.target.clone()),
    })
}
