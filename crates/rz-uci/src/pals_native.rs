//! Exact Rules-to-PALS model composition and a single native physical owner.
//! P/C heads cross the typed PALS scheduler; Python is never a product callback.
pub use rz_eval::pals_device_resources::{
    NativeCudaPackingArtifactResourcesInput, NativeCudaPackingGraphInspectionInput,
    NativeCudaRecordPagePayloadDeclaration, NativeCudaRecordPagesInvocationInput,
    NativeCudaRecordPagesLimitsInput, NativeCudaRecordPagesResourceInput,
};
use rz_eval::pals_model::{
    PalsCandidateToken, PalsFullLineInput, PalsModelConfig, PalsModelInput, PalsModelProfile,
    PalsRecordLine, PalsRecordToken, PalsRole,
};
use rz_position::{
    BoardMove, Color, HistoryCompleteness, HistoryOrigin, PieceKind, Position, Square,
};
use rz_search::cpu::CpuScoreScope;
use rz_search::pals::engine::{DivergenceQuery, RecordKind, RoleError, RoleQuery, RoleRecord};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::time::Instant;

pub const PALS_RULES_ENCODING: &str = "rz-pals-rules-fields-v1";
#[path = "pals_native/selected_view.rs"]
mod selected_view;
use selected_view::{selected_records, validate_selected_query};
#[cfg(feature = "search-work-receipts")]
#[path = "pals_native/legacy_descriptor.rs"]
mod legacy_descriptor;
#[cfg(test)]
#[path = "pals_native/selected_view_tests.rs"]
mod selected_view_tests;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
pub mod startup_legacy;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[path = "pals_native/startup_virtual.rs"]
pub mod startup_virtual;
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu", target_os = "linux"))]
#[path = "pals_native/startup_system_v6.rs"]
pub(crate) mod startup_system_v6;
#[path = "pals_native/virtual_encoding.rs"]
pub mod virtual_encoding;
#[cfg(feature = "search-work-receipts")]
pub use legacy_descriptor::{
    LegacyRulesDescriptorSummary, validate_pals_legacy_rules_descriptor_bytes,
};
pub const PALS_RULES_FULL_LINE_ENCODING: &str = "rz-pals-rules-full-line-v2";
pub const PALS_RULES_BASE_SEMANTICS: &str = "board:a1-h8;empty=0;white_PNBRQK=1..6;black_PNBRQK=7..12;history:newest_first_exact_fen;records:latest_kinds_and_author_critical_pinned;candidate_order:Rules;divergence_query:fields_1,3,4,8..14=0";
const MAX_HISTORY: usize = 4096;

/// Index order, normalization and missing-value meaning are model semantics.
/// A source digest is recorded separately: changing a worker or a comment must
/// not silently change the exact-input cache's encoding meaning.
pub const PALS_METADATA_FEATURES: [&str; 16] = [
    "white_to_move:boolean",
    "white_king_castle:boolean",
    "white_queen_castle:boolean",
    "black_king_castle:boolean",
    "black_queen_castle:boolean",
    "ep_file:/7;absent=-1",
    "ep_rank:/7;absent=-1",
    "halfmove_clock:/150",
    "fullmove_number:/512",
    "known_history_frames:/4096",
    "unknown_history_prefix:boolean",
    "known_repetition_count:exact_count_as_f32",
    "in_check:boolean",
    "white_piece_count:/16",
    "black_piece_count:/16",
    "constant:1",
];
pub const PALS_RECORD_FEATURES: [&str; 16] = [
    "kind:proposal=0,counterexample=1,repair=2,cpu_verification=3",
    "has_completed_value:boolean",
    "completed_value:/30000;missing=0",
    "completed_depth:/64",
    "score_scope:absent=-1,frontier_only=0,completed=1,rules_terminal=2",
    "white_score_perspective:boolean",
    "origin_state_id:advisory_f32_not_identity",
    "line_length:/256",
    "first_from:/63;missing=0",
    "first_to:/63;missing=0",
    "first_promotion:/4;none=0,Q=1,R=2,B=3,N=4",
    "has_first_move:boolean",
    "last_from:/63;missing=0",
    "last_to:/63;missing=0",
    "last_promotion:/4;none=0,Q=1,R=2,B=3,N=4",
    "unknown_value:boolean",
];
pub const PALS_QUERY_FEATURES: [&str; 16] = [
    "mode:propose=0,reply=1,repair=2,divergence=3",
    "prefix_length:/256",
    "proposal_length:/256",
    "has_counterexample:boolean",
    "counterexample_length:/256",
    "candidate_count:role=/256,divergence=/128",
    "situation_revision:advisory_f32_not_identity",
    "deadline_remaining:seconds",
    "white_to_move:boolean",
    "prefix_last_from:/63;missing=0",
    "prefix_last_to:/63;missing=0",
    "prefix_last_promotion:/4;none=0,Q=1,R=2,B=3,N=4",
    "proposal_first_from:/63;missing=0",
    "proposal_first_to:/63;missing=0",
    "proposal_first_promotion:/4;none=0,Q=1,R=2,B=3,N=4",
    "constant:1",
];
pub const PALS_DIVERGENCE_FEATURES: [&str; 8] = [
    "ply_index:/256",
    "opponent_to_move_relative_root:boolean",
    "prefix_in_check:boolean",
    "movement_from:/63",
    "movement_to:/63",
    "promotion:/4;none=0,Q=1,R=2,B=3,N=4",
    "prefix_halfmove_clock:/150",
    "prefix_legal_candidate_count:/256",
];
pub const PALS_FULL_LINE_FEATURES: [&str; 9] = [
    "full_line.record_tokens",
    "full_line.record_lengths",
    "full_line.record_mask",
    "full_line.query_tokens",
    "full_line.query_lengths",
    "full_line.query_mask",
    "full_line.record_kind",
    "full_line.parent_local_index",
    "full_line.supersedes_local_index",
];
pub fn pals_rules_profile_for_config(config: &PalsModelConfig) -> &'static str {
    if config.profile.uses_virtual_review() {
        return virtual_encoding::PALS_VIRTUAL_RULES_ENCODING;
    }
    if config.profile.uses_full_line() {
        PALS_RULES_FULL_LINE_ENCODING
    } else {
        PALS_RULES_ENCODING
    }
}
pub fn pals_record_features_for_config(config: &PalsModelConfig) -> [&'static str; 16] {
    let mut features = PALS_RECORD_FEATURES;
    if config.profile.uses_full_line() {
        features[6] = "origin_state_id:disabled_v2_zero";
    }
    features
}
pub fn pals_query_features_for_config(config: &PalsModelConfig) -> [&'static str; 16] {
    let mut features = PALS_QUERY_FEATURES;
    if config.profile.uses_full_line() {
        features[6] = "situation_revision:disabled_v2_zero";
        features[7] = "deadline_remaining:disabled_v2_zero";
    }
    features
}
pub fn pals_rules_semantic_fields_for_config(
    config: &PalsModelConfig,
) -> impl Iterator<Item = &'static str> {
    if config.profile.uses_virtual_review() {
        return virtual_encoding::pals_virtual_rules_semantic_fields()
            .collect::<Vec<_>>()
            .into_iter();
    }
    let mut fields = vec![
        pals_rules_profile_for_config(config),
        PALS_RULES_BASE_SEMANTICS,
    ];
    fields.extend(PALS_METADATA_FEATURES);
    fields.extend(pals_record_features_for_config(config));
    fields.extend(pals_query_features_for_config(config));
    fields.extend(PALS_DIVERGENCE_FEATURES);
    if config.profile.uses_full_line() {
        fields.extend(PALS_FULL_LINE_FEATURES);
    }
    fields.into_iter()
}
pub fn pals_rules_semantic_fields() -> impl Iterator<Item = &'static str> {
    std::iter::once(PALS_RULES_ENCODING)
        .chain(std::iter::once(PALS_RULES_BASE_SEMANTICS))
        .chain(PALS_METADATA_FEATURES)
        .chain(PALS_RECORD_FEATURES)
        .chain(PALS_QUERY_FEATURES)
        .chain(PALS_DIVERGENCE_FEATURES)
}
pub fn pals_rules_encoding_semantic_digest() -> [u8; 32] {
    pals_rules_encoding_semantic_digest_for_config(&PalsModelConfig::baseline())
}
pub fn pals_rules_encoding_semantic_digest_for_profile(profile: PalsModelProfile) -> [u8; 32] {
    pals_rules_encoding_semantic_digest_for_config(&PalsModelConfig::for_profile(profile))
}
pub fn pals_rules_encoding_semantic_digest_for_config(config: &PalsModelConfig) -> [u8; 32] {
    let mut h = Sha256::new();
    for field in pals_rules_semantic_fields_for_config(config) {
        h.update((field.len() as u64).to_le_bytes());
        h.update(field.as_bytes());
    }
    h.finalize().into()
}
/// Fresh model/input namespace, distinct from the Rules-only manifest profile.
/// Preserve the existing schema bytes followed by the exact Rules digest. This
/// pure identity getter grants no loaded-model or native execution authority;
/// PrivateWarm continues to use its separate capability-owned encoding.
pub fn pals_fresh_encoding_semantic_digest() -> [u8; 32] {
    pals_fresh_encoding_semantic_digest_for_config(&PalsModelConfig::baseline())
}
pub fn pals_fresh_encoding_semantic_digest_for_profile(profile: PalsModelProfile) -> [u8; 32] {
    pals_fresh_encoding_semantic_digest_for_config(&PalsModelConfig::for_profile(profile))
}
pub fn pals_fresh_encoding_semantic_digest_for_config(config: &PalsModelConfig) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(config.profile.encoding_schema());
    if config.profile != PalsModelProfile::LegacySummaryV1 {
        h.update(config.profile.model_semantics());
        h.update(config.profile.as_str());
    }
    h.update(pals_rules_encoding_semantic_digest_for_config(config));
    h.finalize().into()
}
pub fn pals_native_source_digest() -> [u8; 32] {
    Sha256::digest(include_bytes!("pals_native.rs")).into()
}
pub fn pals_native_selected_review_source_digest() -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-native-selected-view/1");
    hash.update(pals_native_source_digest());
    hash.update(include_bytes!("pals_native/selected_view.rs"));
    hash.finalize().into()
}
#[cfg(feature = "onnx-cpu")]
pub fn pals_native_private_warm_source_digest() -> [u8; 32] {
    private_warm::source_digest()
}
#[cfg(all(feature = "onnx-cpu", feature = "experimental-io-binding"))]
pub fn pals_native_cuda_warm_adapter_source_digest() -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rz-pals-native-private-warm-adapter/1");
    hash.update(pals_native_source_digest());
    hash.update(private_warm::source_digest());
    hash.update(rz_eval::pals_onnx::cuda_private_warm_implementation_digest());
    hash.finalize().into()
}
/// Shape compatibility alone does not prove that a model was trained on the
/// same feature meanings. This pin verifies the declared Rules transformation;
/// actual learned-input suitability still needs separate data/model evidence.
pub fn verify_pals_rules_profile(
    profile: Option<&str>,
    semantic_digest: Option<[u8; 32]>,
) -> Result<(), RoleError> {
    verify_pals_rules_profile_for_config(profile, semantic_digest, &PalsModelConfig::baseline())
}
pub fn verify_pals_rules_profile_for_config(
    profile: Option<&str>,
    semantic_digest: Option<[u8; 32]>,
    config: &PalsModelConfig,
) -> Result<(), RoleError> {
    match (profile, semantic_digest) {
        (Some(profile), Some(digest)) if profile == pals_rules_profile_for_config(config) && digest == pals_rules_encoding_semantic_digest_for_config(config) => Ok(()),
        (None, _) | (_, None) => Err(RoleError::Backend("PALS product export lacks the required Rules input semantic profile; legacy numeric fixtures do not authorize product execution".into())),
        _ => Err(RoleError::Backend("PALS export Rules input semantic profile differs from this encoder".into())),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeQueryKind {
    Propose,
    Reply,
    Repair,
    Divergence,
}

fn model_error(error: impl std::fmt::Display) -> RoleError {
    RoleError::Backend(error.to_string())
}
#[cfg(any(feature = "onnx-cpu", test))]
fn runtime_admission_error(error: rz_contracts::ContractError) -> RoleError {
    // Preserve the typed role deadline used by PALS bounded search completion.
    // Other contract failures retain their original Backend diagnostic.
    if error.stage == rz_contracts::Stage::Admission
        && error.code == rz_contracts::ErrorCode::Expired
    {
        RoleError::Deadline
    } else {
        model_error(error)
    }
}
fn flag(value: bool) -> f32 {
    f32::from(u8::from(value))
}
fn candidate(movement: BoardMove) -> PalsCandidateToken {
    PalsCandidateToken {
        from: movement.from.index(),
        to: movement.to.index(),
        promotion: match movement.promotion {
            None => 0,
            Some(PieceKind::Queen) => 1,
            Some(PieceKind::Rook) => 2,
            Some(PieceKind::Bishop) => 3,
            Some(PieceKind::Knight) => 4,
            _ => 255,
        },
    }
}
/// Streaming exact known history. Unknown prefixes remain explicitly distinct;
/// compact network features never substitute for repetition/terminal authority.
pub fn pals_history_digest(position: &Position) -> Result<[u8; 32], RoleError> {
    if position.known_history_len() > MAX_HISTORY {
        return Err(RoleError::Backend(
            "PALS exact history budget exceeded".into(),
        ));
    }
    let snapshot = position.snapshot();
    let mut h = Sha256::new();
    h.update(PALS_RULES_ENCODING);
    h.update([match snapshot.history_completeness() {
        HistoryCompleteness::Complete => 0,
        HistoryCompleteness::UnknownPrefix => 1,
    }]);
    h.update([match snapshot.history_origin() {
        HistoryOrigin::StartPosition => 0,
        HistoryOrigin::Fen => 1,
    }]);
    h.update((snapshot.known_history_len() as u64).to_le_bytes());
    for frame in snapshot.known_history() {
        let fen = frame.to_fen();
        h.update((fen.len() as u64).to_le_bytes());
        h.update(fen.as_bytes());
    }
    Ok(h.finalize().into())
}
fn board_and_metadata(position: &Position) -> Result<(Vec<u8>, [f32; 16]), RoleError> {
    let mut board = Vec::with_capacity(64);
    let mut pieces = [0_u32; 2];
    for index in 0..64 {
        let piece = position.piece_at(Square::new(index).map_err(model_error)?);
        board.push(match piece {
            None => 0,
            Some(piece) => {
                let offset = if piece.color == Color::White {
                    pieces[0] += 1;
                    0
                } else {
                    pieces[1] += 1;
                    6
                };
                offset
                    + match piece.kind {
                        PieceKind::Pawn => 1,
                        PieceKind::Knight => 2,
                        PieceKind::Bishop => 3,
                        PieceKind::Rook => 4,
                        PieceKind::Queen => 5,
                        PieceKind::King => 6,
                    }
            }
        });
    }
    let rights = position.castling_rights();
    let ep = position.en_passant_target();
    let mut metadata = [0.0; 16];
    metadata[0] = flag(position.side_to_move() == Color::White);
    for i in 0..4 {
        metadata[i + 1] = flag(rights & (1 << i) != 0);
    }
    metadata[5] = ep.map_or(-1.0, |s| f32::from(s.file()) / 7.0);
    metadata[6] = ep.map_or(-1.0, |s| f32::from(s.rank()) / 7.0);
    metadata[7] = position.halfmove_clock() as f32 / 150.0;
    metadata[8] = position.fullmove_number() as f32 / 512.0;
    metadata[9] = position.known_history_len() as f32 / MAX_HISTORY as f32;
    metadata[10] = flag(position.history_completeness() == HistoryCompleteness::UnknownPrefix);
    metadata[11] = position.known_repetition_count() as f32;
    metadata[12] = flag(position.in_check());
    metadata[13] = pieces[0] as f32 / 16.0;
    metadata[14] = pieces[1] as f32 / 16.0;
    metadata[15] = 1.0;
    Ok((board, metadata))
}
type EncodedPalsRecords = (Vec<PalsRecordToken>, Vec<u64>, Option<Vec<PalsRecordLine>>);
fn records(
    records: &[RoleRecord],
    revision: u64,
    config: &PalsModelConfig,
) -> Result<EncodedPalsRecords, RoleError> {
    if records.len() > 16_384 {
        return Err(RoleError::Backend(
            "PALS public record scan budget exceeded".into(),
        ));
    }
    let mut critical = BTreeSet::new();
    for kind in [
        RecordKind::Proposal,
        RecordKind::Counterexample,
        RecordKind::Repair,
        RecordKind::CpuVerification,
    ] {
        if let Some(index) = records.iter().rposition(|r| r.kind == kind) {
            critical.insert(index);
        }
    }
    for (index, record) in records.iter().enumerate() {
        if record.revision > revision {
            return Err(RoleError::Backend(
                "future public record in PALS input".into(),
            ));
        }
        if record.critical {
            critical.insert(index);
        }
    }
    if critical.len() > config.max_records {
        return Err(RoleError::Backend(
            "PALS critical context exceeds registered record width".into(),
        ));
    }
    let mut admitted = critical.clone();
    // The bounded view retains direct evidence for its active critical records.
    // Historical noncritical relationships stay in raw/archive provenance and
    // may be absent from this model view; no unbounded ancestor walk is used.
    if config.profile.uses_full_line() {
        for &index in &critical {
            for required in [
                records[index].parent_revision,
                records[index].supersedes_revision,
            ]
            .into_iter()
            .flatten()
            {
                let mut matches = records
                    .iter()
                    .enumerate()
                    .filter(|(_, record)| record.revision == required);
                let (target, _) = matches.next().ok_or_else(|| {
                    RoleError::Backend("PALS full-line required relationship is unavailable".into())
                })?;
                if matches.next().is_some() || target == index {
                    return Err(RoleError::Backend(
                        "PALS full-line relationship is ambiguous or self-referential".into(),
                    ));
                }
                admitted.insert(target);
            }
        }
        if admitted.len() > config.max_records {
            return Err(RoleError::Backend(
                "PALS direct critical relationships exceed record bound".into(),
            ));
        }
    }
    for index in (0..records.len()).rev() {
        if admitted.len() == config.max_records {
            break;
        }
        admitted.insert(index);
    }
    let mut result = Vec::with_capacity(admitted.len());
    let mut required = Vec::with_capacity(critical.len());
    let selected: Vec<_> = admitted.into_iter().collect();
    let mut full_line = config
        .profile
        .uses_full_line()
        .then(|| Vec::with_capacity(selected.len()));
    for &index in &selected {
        let record = &records[index];
        if record.line.len() > 256 {
            return Err(RoleError::Backend(
                "PALS record line budget exceeded".into(),
            ));
        }
        let id = (index as u64)
            .checked_add(1)
            .ok_or_else(|| RoleError::Backend("PALS record ID overflow".into()))?;
        let is_critical = critical.contains(&index);
        if is_critical {
            required.push(id);
        }
        let mut features = [0.0; 16];
        features[0] = match record.kind {
            RecordKind::Proposal => 0.0,
            RecordKind::Counterexample => 1.0,
            RecordKind::Repair => 2.0,
            RecordKind::CpuVerification => 3.0,
        };
        features[1] = flag(record.value.is_some());
        features[2] = record.value.unwrap_or(0) as f32 / 30_000.0;
        features[3] = f32::from(record.completed_depth) / 64.0;
        features[4] = match record.score_scope {
            None => -1.0,
            Some(CpuScoreScope::FrontierOnly) => 0.0,
            Some(CpuScoreScope::CompletedIteration) => 1.0,
            Some(CpuScoreScope::RulesTerminal) => 2.0,
        };
        features[5] = flag(record.perspective == Color::White);
        // An anchor identifier is a provenance field, never a calibrated score.
        features[6] = if config.profile.uses_full_line() {
            0.0
        } else {
            record.origin_state.0 as f32
        };
        features[7] = record.line.len() as f32 / 256.0;
        if let Some(movement) = record.line.first() {
            let c = candidate(*movement);
            c.validate().map_err(model_error)?;
            features[8] = f32::from(c.from) / 63.0;
            features[9] = f32::from(c.to) / 63.0;
            features[10] = f32::from(c.promotion) / 4.0;
            features[11] = 1.0;
        }
        if let Some(movement) = record.line.last() {
            let c = candidate(*movement);
            c.validate().map_err(model_error)?;
            features[12] = f32::from(c.from) / 63.0;
            features[13] = f32::from(c.to) / 63.0;
            features[14] = f32::from(c.promotion) / 4.0;
        }
        features[15] = flag(record.value.is_none());
        result.push(PalsRecordToken {
            record_id: id,
            revision: record.revision,
            critical: is_critical,
            features,
        });
        if let Some(lines) = &mut full_line {
            let local = |relationship: Option<u64>| -> Result<Option<usize>, RoleError> {
                let Some(revision) = relationship else {
                    return Ok(None);
                };
                let mut matches = selected
                    .iter()
                    .enumerate()
                    .filter(|(_, index)| records[**index].revision == revision);
                let target = matches.next().map(|(slot, _)| slot);
                if matches.next().is_some() {
                    return Err(RoleError::Backend(
                        "PALS selected relationship is ambiguous".into(),
                    ));
                }
                if target.is_none() && is_critical {
                    return Err(RoleError::Backend(
                        "PALS selected required relationship is missing".into(),
                    ));
                }
                Ok(target)
            };
            lines.push(PalsRecordLine {
                moves: record.line.iter().copied().map(candidate).collect(),
                parent: local(record.parent_revision)?,
                supersedes: local(record.supersedes_revision)?,
                parent_required: is_critical && record.parent_revision.is_some(),
                supersedes_required: is_critical && record.supersedes_revision.is_some(),
            });
        }
    }
    Ok((result, required, full_line))
}
pub fn prepare_role_input(
    query: &RoleQuery<'_>,
    kind: NativeQueryKind,
    model_epoch: [u8; 32],
) -> Result<PalsModelInput, RoleError> {
    prepare_role_input_for_config(query, kind, model_epoch, &PalsModelConfig::baseline())
}
pub fn prepare_role_input_for_config(
    query: &RoleQuery<'_>,
    kind: NativeQueryKind,
    model_epoch: [u8; 32],
    config: &PalsModelConfig,
) -> Result<PalsModelInput, RoleError> {
    prepare_role_input_with_view(query, kind, model_epoch, config, None)
}
pub fn prepare_selected_role_input_for_config(
    query: &RoleQuery<'_>,
    kind: NativeQueryKind,
    model_epoch: [u8; 32],
    config: &PalsModelConfig,
    view: &rz_search::pals::review_context::ReviewView,
) -> Result<PalsModelInput, RoleError> {
    if view.query().purpose == rz_search::pals::engine::RoleQueryPurpose::TaskPolicy {
        return Err(RoleError::Unsupported(
            "TaskPolicy requires the explicit V3 collection adapter",
        ));
    }
    validate_selected_query(view, query)?;
    prepare_role_input_with_view(query, kind, model_epoch, config, Some(view))
}
fn prepare_role_input_with_view(
    query: &RoleQuery<'_>,
    kind: NativeQueryKind,
    model_epoch: [u8; 32],
    config: &PalsModelConfig,
    view: Option<&rz_search::pals::review_context::ReviewView>,
) -> Result<PalsModelInput, RoleError> {
    query.check_control()?;
    if kind == NativeQueryKind::Divergence
        || query.prefix.len() > 256
        || query.proposal.len() > 256
        || query.counterexample.is_some_and(|line| line.len() > 256)
    {
        return Err(RoleError::InvalidOutput);
    }
    if query.legal != query.position.legal_moves() {
        return Err(RoleError::Backend(
            "PALS model candidates differ from exact Rules legal order".into(),
        ));
    }
    let (board, metadata) = board_and_metadata(query.position)?;
    let (records, required_critical_records, full_line_records) = match view {
        Some(view) => selected_records(view, query.revision, config)?,
        None => records(query.records, query.revision, config)?,
    };
    let candidates = query.legal.iter().copied().map(candidate).collect();
    let mut features = [0.0; 16];
    features[0] = match kind {
        NativeQueryKind::Propose => 0.0,
        NativeQueryKind::Reply => 1.0,
        NativeQueryKind::Repair => 2.0,
        NativeQueryKind::Divergence => unreachable!(),
    };
    features[1] = query.prefix.len() as f32 / 256.0;
    features[2] = query.proposal.len() as f32 / 256.0;
    features[3] = flag(query.counterexample.is_some());
    features[4] = query.counterexample.map_or(0.0, |l| l.len() as f32 / 256.0);
    features[5] = query.legal.len() as f32 / 256.0;
    features[6] = if config.profile.uses_full_line() {
        0.0
    } else {
        query.revision as f32
    };
    features[7] = if config.profile.uses_full_line() {
        0.0
    } else {
        query
            .deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f32()
    };
    features[8] = flag(query.position.side_to_move() == Color::White);
    if let Some(movement) = query.prefix.last() {
        let c = candidate(*movement);
        features[9] = f32::from(c.from) / 63.0;
        features[10] = f32::from(c.to) / 63.0;
        features[11] = f32::from(c.promotion) / 4.0;
    }
    if let Some(movement) = query.proposal.first() {
        let c = candidate(*movement);
        features[12] = f32::from(c.from) / 63.0;
        features[13] = f32::from(c.to) / 63.0;
        features[14] = f32::from(c.promotion) / 4.0;
    }
    features[15] = 1.0;
    let input = PalsModelInput {
        role: if kind == NativeQueryKind::Reply {
            PalsRole::Critic
        } else {
            PalsRole::Proposer
        },
        board,
        metadata,
        records,
        required_critical_records,
        candidates,
        divergence_features: vec![],
        query: features,
        situation_revision: query.revision,
        history_digest: pals_history_digest(query.position)?,
        model_epoch,
        full_line: full_line_records.map(|records| PalsFullLineInput {
            records,
            query_prefix: query.prefix.iter().copied().map(candidate).collect(),
            query_proposal: query.proposal.iter().copied().map(candidate).collect(),
            query_counter: query
                .counterexample
                .unwrap_or(&[])
                .iter()
                .copied()
                .map(candidate)
                .collect(),
        }),
    };
    input.validate(config).map_err(model_error)?;
    Ok(input)
}
pub fn prepare_divergence_input(
    query: &DivergenceQuery<'_>,
    model_epoch: [u8; 32],
) -> Result<PalsModelInput, RoleError> {
    prepare_divergence_input_for_config(query, model_epoch, &PalsModelConfig::baseline())
}
pub fn prepare_divergence_input_for_config(
    query: &DivergenceQuery<'_>,
    model_epoch: [u8; 32],
    config: &PalsModelConfig,
) -> Result<PalsModelInput, RoleError> {
    prepare_divergence_input_with_view(query, model_epoch, config, None)
}
pub fn prepare_selected_divergence_input_for_config(
    query: &DivergenceQuery<'_>,
    model_epoch: [u8; 32],
    config: &PalsModelConfig,
    view: &rz_search::pals::review_context::ReviewView,
) -> Result<PalsModelInput, RoleError> {
    let anchor = view.anchors().first().ok_or(RoleError::InvalidOutput)?;
    if !query.root.snapshot().same_state(&anchor.snapshot)
        || !view.query().prefix.is_empty()
        || query.proposal != view.query().proposal
        || query.candidates != view.query().divergences
    {
        return Err(RoleError::InvalidOutput);
    }
    prepare_divergence_input_with_view(query, model_epoch, config, Some(view))
}
fn prepare_divergence_input_with_view(
    query: &DivergenceQuery<'_>,
    model_epoch: [u8; 32],
    config: &PalsModelConfig,
    view: Option<&rz_search::pals::review_context::ReviewView>,
) -> Result<PalsModelInput, RoleError> {
    if query.cancel.load(std::sync::atomic::Ordering::Acquire) {
        return Err(RoleError::Canceled);
    }
    if Instant::now() >= query.deadline {
        return Err(RoleError::Deadline);
    }
    if query.proposal.len() > 256
        || query.candidates.is_empty()
        || query.candidates.len() > 128
        || query
            .candidates
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != query.candidates.len()
    {
        return Err(RoleError::InvalidOutput);
    }
    let (board, metadata) = board_and_metadata(query.root)?;
    let (records, required_critical_records, full_line_records) = match view {
        Some(view) => selected_records(view, query.revision, config)?,
        None => records(query.records, query.revision, config)?,
    };
    let mut prefix = query.root.clone();
    let mut by_ply = Vec::with_capacity(query.proposal.len());
    for (ply, movement) in query.proposal.iter().enumerate() {
        let c = candidate(*movement);
        c.validate().map_err(model_error)?;
        by_ply.push([
            ply as f32 / 256.0,
            flag(prefix.side_to_move() != query.root.side_to_move()),
            flag(prefix.in_check()),
            f32::from(c.from) / 63.0,
            f32::from(c.to) / 63.0,
            f32::from(c.promotion) / 4.0,
            prefix.halfmove_clock() as f32 / 150.0,
            prefix.legal_moves().len() as f32 / 256.0,
        ]);
        prefix.make_move(*movement).map_err(model_error)?;
    }
    let mut divergence_features = Vec::with_capacity(query.candidates.len());
    for &ply in query.candidates {
        let features = by_ply.get(ply).ok_or(RoleError::InvalidOutput)?;
        if features[1] != 1.0 {
            return Err(RoleError::Backend(
                "critic divergence is not an opponent turn".into(),
            ));
        }
        divergence_features.push(*features);
    }
    let mut features = [0.0; 16];
    features[0] = 3.0;
    features[2] = query.proposal.len() as f32 / 256.0;
    features[5] = query.candidates.len() as f32 / 128.0;
    features[6] = if config.profile.uses_full_line() {
        0.0
    } else {
        query.revision as f32
    };
    features[7] = if config.profile.uses_full_line() {
        0.0
    } else {
        query
            .deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f32()
    };
    features[15] = 1.0;
    let input = PalsModelInput {
        role: PalsRole::Critic,
        board,
        metadata,
        records,
        required_critical_records,
        candidates: vec![],
        divergence_features,
        query: features,
        situation_revision: query.revision,
        history_digest: pals_history_digest(query.root)?,
        model_epoch,
        full_line: full_line_records.map(|records| PalsFullLineInput {
            records,
            query_prefix: vec![],
            query_proposal: query.proposal.iter().copied().map(candidate).collect(),
            query_counter: vec![],
        }),
    };
    input.validate(config).map_err(model_error)?;
    Ok(input)
}

#[path = "pals_native/ticketed.rs"]
pub mod ticketed;

#[cfg(feature = "onnx-cpu")]
#[path = "pals_native/private_warm.rs"]
mod private_warm;
#[cfg(feature = "onnx-cpu")]
pub use private_warm::{NativePrivateWarmObservation, NativePrivateWarmRevocation};

#[cfg(feature = "onnx-cpu")]
#[path = "pals_native/recorded_witness.rs"]
mod recorded_witness;
#[cfg(feature = "onnx-cpu")]
pub use recorded_witness::{
    NativeRecordedCpuPristineAck, NativeRecordedInputWitness, RECORDED_INPUT_WITNESS_SCHEMA,
    RECORDED_INPUT_WITNESS_SCOPE, recorded_input_witness_source_digest,
};

#[cfg(feature = "onnx-cpu")]
mod native {
    use super::private_warm::{NativeWarmState, WarmPhysical};
    use super::*;
    use rz_contracts::pals::{
        ExecutionMode, Move16, PALS_CONTRACT_REVISION, RepresentationKey, Role, RoleOutput,
        RolePayload, RoleRequest, SearchAuthority, SituationHandle,
    };
    use rz_contracts::{
        ByteBudget, CancelToken, ContractError, Deadline, Digest, ErrorCode, ExecutionId,
        GameGeneration, MonotonicTick, OwnerId, PrecisionProfile, ProcessEpoch, RequestId,
        RootGeneration, Stage, Wdl,
    };
    use rz_eval::error::BackendError;
    use rz_eval::pals_onnx::{
        HostRecordPageObservationBoundary, HostRecordPageObservationHandle,
        HostRecordPageObservationOutcome, HostRecordPageObservationSnapshot,
        HostRecordPageObservationStatus, HostRecordPagePolicy, HostRecordPageSnapshot,
        PalsAppliedSessionSettings, PalsBackendStats, PalsCudaControlPolicy,
        PalsCudaPlacementWitness, PalsGraphOptimization, PalsNativeCommand,
        PalsNativeMappingWitness, PalsNativeOwnerAccounting, PalsNativeResult, PalsOnnxBackend,
        PalsSessionResidency, PalsStartupInputCompletionBinding, PalsStartupSnapshotStatus,
        PalsStartupStageProbe, PalsStartupStageSnapshot, PalsWarmCapability, PalsWarmInput,
    };
    #[cfg(feature = "experimental-io-binding")]
    use rz_eval::pals_onnx::{PalsCudaWarmCapability, PalsCudaWarmInput};
    use rz_eval::worker::{PhysicalLease, PhysicalPoll, SingleWorker};
    use rz_position::contracts::ContractPosition;
    use rz_runtime::contracts::{ContractClock, ContractSystemClock};
    use rz_runtime::pals::{
        PalsAdapter, PalsRuntime, PalsScope, PalsTerminal, PhysicalRoleOutput, RepresentationScope,
        RuntimeRoleRequest, SharedPalsScope, SharedRepresentationScope,
    };
    use rz_runtime::{Backend, BackendResult, Clock, Limits, Resources};
    use rz_search::pals::engine::{
        MODEL_WDL_VALUE_SEMANTICS, ModelValueIdentity, ModelValueOutput, RecheckFinished,
        RecheckPrepared, RoleAcceptance, RoleEvaluation, RoleLogicalContext, RoleModel,
        RoleQueryPurpose, RoleSearchClosure, RoleValueExecutionEvidence,
    };
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::Poll;
    use std::time::Duration;

    const REQUEST_BYTES: u64 = 1024 * 1024;
    const DEFAULT_DRAIN_LIMIT: Duration = Duration::from_secs(2);
    const DEPLOYMENT_FROZEN_EPOCH: u64 = 1;
    static EPOCHS: AtomicU64 = AtomicU64::new(0x5041_4c53_0000_0000);
    static RULE_OWNERS: AtomicU64 = AtomicU64::new(0x5041_4c52_0000_0000);
    static RECORDED_WITNESS_LEASES: AtomicU64 = AtomicU64::new(1);
    type NativeWorker = SingleWorker<PalsNativeCommand, Result<PalsNativeResult, BackendError>>;
    type NativePhysicalLease =
        PhysicalLease<PalsNativeCommand, Result<PalsNativeResult, BackendError>>;
    type Runtime = PalsRuntime<PalsModelInput, WorkerBackend, ContractSystemClock>;
    fn fault(code: ErrorCode, detail: &'static str) -> ContractError {
        ContractError::new(code, Stage::Backend, detail)
    }
    fn next(counter: &AtomicU64) -> Result<u64, RoleError> {
        counter
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_add(1))
            .map_err(|_| RoleError::Backend("PALS owner ID epoch exhausted".into()))
    }
    fn frontier_value_identity(
        identity: &str,
        model_epoch: [u8; 32],
        encoding: Digest,
    ) -> ModelValueIdentity {
        ModelValueIdentity {
            semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
            model: identity.into(),
            encoding: encoding.0.iter().map(|b| format!("{b:02x}")).collect(),
            precision: "fp32".into(),
            model_epoch,
        }
    }
    fn check_context(
        query: &RoleQuery<'_>,
        context: &RoleLogicalContext,
        purpose: RoleQueryPurpose,
    ) -> Result<(), RoleError> {
        if context.purpose != purpose
            || context.public_revision != query.revision
            || context.prefix != query.prefix
        {
            return Err(RoleError::Backend(
                "Native role logical query context mismatch".into(),
            ));
        }
        Ok(())
    }
    struct WorkerOwner {
        worker: Mutex<NativeWorker>,
        physical_completed: AtomicU64,
        completed: AtomicU64,
        physical_failed: AtomicU64,
        validation_failed: AtomicU64,
        delivered: AtomicU64,
        consumed: AtomicU64,
        canceled: AtomicU64,
        expired: AtomicU64,
        new_game_resets: AtomicU64,
        in_flight: AtomicU64,
        game_generation: AtomicU64,
        pending_game_target: AtomicU64,
        game_lifecycle: Mutex<()>,
        game_reset_failure: Mutex<Option<RoleError>>,
        request_high_water: AtomicU64,
        execution_high_water: AtomicU64,
        quarantined: AtomicBool,
        finishing: AtomicBool,
        shutdown: AtomicBool,
        model_epoch: [u8; 32],
        model_config: PalsModelConfig,
        manifest_digest: [u8; 32],
        encoding_semantic_digest: [u8; 32],
        adapter_source_digest: [u8; 32],
        trained: bool,
        residency: PalsSessionResidency,
        // 실제 SessionBuilder가 성공한 record만 보존. synthetic worker는 None이다.
        applied_session_settings: Option<Arc<[PalsAppliedSessionSettings]>>,
        process_epoch: ProcessEpoch,
        // The original startup FDs and profile directory stay with the actual
        // worker owner, including retained physical leases and quarantine.
        // Legacy path loaders have no registered source capability.
        _registered_sources: Option<Arc<dyn rz_eval::pals_onnx::PalsOwnedSources>>,
        last_failure: Mutex<Option<NativeFailureReceipt>>,
        final_stats: Mutex<Option<NativeBackendStatsReceipt>>,
        stats_lease: Mutex<Option<NativePhysicalLease>>,
        cuda_record_page_snapshot: Mutex<Option<NativeCudaRecordPageSnapshotReceipt>>,
        cuda_record_page_snapshot_error: Mutex<Option<NativeStartupErrorKind>>,
        execution: NativeExecutionReceipt,
        additional_host_bytes: u64,
        host_record_page_observer: Option<HostRecordPageObservationHandle>,
        // Zero preserves the legacy absent/default wire; explicit selection
        // is copied into immutable receipt snapshots, never worker inputs.
        startup_probe_timeout_ms: AtomicU64,
        startup_stage_probe: Option<PalsStartupStageProbe>,
        startup_probe: Mutex<Option<NativeStartupProbeReceipt>>,
        startup_ready: AtomicBool,
        final_mapping_confirmed: AtomicBool,
        final_loading_mapping: Mutex<Option<Box<PalsNativeMappingWitness>>>,
        observer: Mutex<Option<Box<dyn NativeRoleObserver>>>,
        observer_failures: AtomicU64,
        last_observer_failure: Mutex<Option<NativeFailureReceipt>>,
        // One actual submitted role lease, never a high-water reconstruction.
        role_execution: Mutex<Option<ObservedRoleExecution>>,
        // V4-only accounting is selected before admission. It changes no
        // model input and adds no second NN or Stats-control invocation.
        followup_evidence: AtomicBool,
        prepared_perspective: Mutex<Option<(RequestId, rz_contracts::Color)>>,
        private_warm: Option<NativeWarmOwner>,
        warm_admission: Mutex<Option<WarmAdmission>>,
    }
    #[derive(Clone, Copy)]
    enum NativeWarmCapability {
        Cpu(PalsWarmCapability),
        #[cfg(feature = "experimental-io-binding")]
        Cuda(PalsCudaWarmCapability),
    }
    impl NativeWarmCapability {
        fn model_identity(&self) -> rz_eval::pals_private::PrivateModelIdentity {
            match self {
                Self::Cpu(cap) => cap.private_model_identity(),
                #[cfg(feature = "experimental-io-binding")]
                Self::Cuda(cap) => cap.private_model_identity(),
            }
        }
        fn implementation_digest(&self) -> [u8; 32] {
            match self {
                Self::Cpu(cap) => cap.implementation_digest(),
                #[cfg(feature = "experimental-io-binding")]
                Self::Cuda(cap) => cap.implementation_digest(),
            }
        }
        fn owner_bytes(&self) -> u64 {
            match self {
                Self::Cpu(_) => private_warm::OWNER_RESERVATION_BYTES,
                #[cfg(feature = "experimental-io-binding")]
                Self::Cuda(_) => private_warm::CUDA_OWNER_RESERVATION_BYTES,
            }
        }
        fn payload_bytes(&self) -> u64 {
            match self {
                Self::Cpu(_) => private_warm::PAYLOAD_BYTES,
                #[cfg(feature = "experimental-io-binding")]
                Self::Cuda(_) => private_warm::CUDA_PAYLOAD_BYTES,
            }
        }
        fn state(&self, game: u64) -> Result<Arc<NativeWarmState>, RoleError> {
            match self {
                Self::Cpu(cap) => NativeWarmState::new(cap.private_model_identity(), game),
                #[cfg(feature = "experimental-io-binding")]
                Self::Cuda(cap) => NativeWarmState::new_cuda(
                    cap.private_model_identity(),
                    game,
                    cap.model_config(),
                ),
            }
        }
        fn payload(&self, input: PalsModelInput) -> Result<NativeWarmPayload, BackendError> {
            match self {
                Self::Cpu(cap) => PalsWarmInput::fresh(cap, input).map(NativeWarmPayload::Cpu),
                #[cfg(feature = "experimental-io-binding")]
                Self::Cuda(cap) => {
                    PalsCudaWarmInput::fresh(cap, input).map(NativeWarmPayload::Cuda)
                }
            }
        }
    }
    enum NativeWarmPayload {
        Cpu(PalsWarmInput),
        #[cfg(feature = "experimental-io-binding")]
        Cuda(PalsCudaWarmInput),
    }
    impl NativeWarmPayload {
        fn owned_host_bytes(&self) -> u64 {
            match self {
                Self::Cpu(input) => input.owned_host_bytes(),
                #[cfg(feature = "experimental-io-binding")]
                Self::Cuda(input) => input.owned_host_bytes(),
            }
        }
        fn initial_latent_bits(&self) -> &[u32] {
            match self {
                Self::Cpu(input) => input.initial_latent_bits(),
                #[cfg(feature = "experimental-io-binding")]
                Self::Cuda(input) => input.initial_latent_bits(),
            }
        }
        fn bind_lease(
            &mut self,
            cap: &NativeWarmCapability,
            lease: &rz_eval::pals_private::PrivateSeedLease,
        ) -> Result<(), BackendError> {
            match (self, cap) {
                (Self::Cpu(input), NativeWarmCapability::Cpu(cap)) => input.bind_lease(cap, lease),
                #[cfg(feature = "experimental-io-binding")]
                (Self::Cuda(input), NativeWarmCapability::Cuda(cap)) => {
                    input.bind_lease(cap, lease)
                }
                #[cfg(feature = "experimental-io-binding")]
                _ => Err(BackendError::new(
                    rz_eval::error::FailureKind::IdentityMismatch,
                    rz_eval::error::FailureStage::Admission,
                    "native warm capability/payload domains differ",
                )),
            }
        }
        fn command(self) -> PalsNativeCommand {
            match self {
                Self::Cpu(input) => PalsNativeCommand::EvaluatePrivateWarm(input),
                #[cfg(feature = "experimental-io-binding")]
                Self::Cuda(input) => PalsNativeCommand::EvaluateCudaWarm(input),
            }
        }
        fn retain_backup(self, physical: &mut WarmPhysical) {
            match self {
                Self::Cpu(input) => physical.backup_payload = Some(input),
                #[cfg(feature = "experimental-io-binding")]
                Self::Cuda(input) => physical.backup_cuda_payload = Some(input),
            }
        }
    }
    struct NativeWarmOwner {
        capability: NativeWarmCapability,
        state: Arc<NativeWarmState>,
        #[cfg(feature = "experimental-io-binding")]
        observer: Option<rz_eval::pals_onnx::PalsCudaWarmObservationHandle>,
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct NativeCudaWarmObservationLimits {
        pub max_leases: u32,
        /// Explicit owned CUDA K/V payload limit; excludes ORT workspace.
        pub device_bytes_max: u64,
    }
    /// Original load/owner-admission cause. Reading this diagnostic neither
    /// certifies a physical Run nor changes any worker cleanup authority.
    #[derive(Clone, Debug)]
    pub struct NativeCudaWarmLoadError {
        pub stage: &'static str,
        pub cause: RoleError,
        pub backend: Option<Box<BackendError>>,
    }
    #[cfg(feature = "experimental-io-binding")]
    impl NativeCudaWarmLoadError {
        fn role(stage: &'static str, cause: RoleError) -> Self {
            Self {
                stage,
                cause,
                backend: None,
            }
        }
        fn backend(stage: &'static str, cause: BackendError) -> Self {
            Self {
                stage,
                cause: model_error(&cause),
                backend: Some(Box::new(cause)),
            }
        }
    }
    /// A single prepared physical handoff. Dropping it without real Ready
    /// conservatively retains the bank, frozen query and full backup payload.
    struct WarmAdmission {
        state: Arc<NativeWarmState>,
        payload: Option<NativeWarmPayload>,
        physical: Option<WarmPhysical>,
    }
    impl Drop for WarmAdmission {
        fn drop(&mut self) {
            if let Some(physical) = self.physical.take() {
                self.state
                    .retain_unknown(physical, "unconfirmed_admission_drop");
            }
        }
    }
    #[derive(Clone, Copy)]
    pub enum NativePreparedContext<'a> {
        Role {
            query: &'a RoleQuery<'a>,
            kind: NativeQueryKind,
        },
        Divergence {
            query: &'a DivergenceQuery<'a>,
        },
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum NativeRoleRejection {
        Admission,
        Canceled,
        Deadline,
        Stale,
        Backend,
        InvalidOutput,
        PhysicalCompletionUnknown,
        Observer,
        SupersededUnconsumed,
        SearchFinishedUnconsumed,
        SearchFailedUnconsumed,
        GameReset,
    }
    /// Actual request/Runtime execution association supplied by a submitted Lease.
    /// This copyable observation is not a factory/source/binary or whole-cost proof.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct NativeRoleExecutionBinding {
        pub request: RequestId,
        pub execution: ExecutionId,
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum NativeRoleCompletionUnknown {
        Quarantined,
        Consumed,
        DrainExpired,
    }
    /// Raw physical observation, distinct from head validation, delivery and
    /// search consumption. CompletionUnknown is not physical terminal completion.
    /// An unknown observation can precede a separately observed actual late Ready;
    /// that fact never clears quarantine or authorizes normal delivery/recovery.
    #[derive(Clone, Copy)]
    pub enum NativeRoleTerminal<'a> {
        Ready(Result<&'a PalsNativeResult, &'a BackendError>),
        CompletionUnknown(NativeRoleCompletionUnknown),
    }
    /// Immutable actual Warm admission. The seed bits belong to the bound full
    /// worker payload, and the bank snapshot is observed after its real begin.
    /// This callback grants no input/output replacement or seed acceptance.
    #[derive(Clone, Copy)]
    pub struct NativePrivateWarmPrepared<'a> {
        pub invocation: rz_eval::pals_private::PrivateInvocation,
        pub initial_latent_bits: &'a [u32],
        pub seed_provenance: Option<&'a rz_eval::pals_private::PrivateSeedProvenance>,
        pub bank: &'a NativePrivateWarmObservation,
    }
    struct ObservedRoleExecution {
        binding: NativeRoleExecutionBinding,
        unknown_observed: bool,
        ready_observed: bool,
        // The first secondary observer failure survives pending/unknown work.
        observer_error: Option<RoleError>,
        prepared_state: rz_contracts::StateIdentity,
        prepared_perspective: Option<rz_contracts::Color>,
        input_sha256: [u8; 32],
        fresh: Option<bool>,
        completed_nn_inputs: Option<u64>,
        measurement_complete: bool,
        owner_accounting: Option<PalsNativeOwnerAccounting>,
    }
    impl ObservedRoleExecution {
        fn ticket_execution_evidence(&self) -> Option<super::ticketed::TicketExecutionEvidence> {
            if !self.ready_observed || self.unknown_observed {
                return None;
            }
            Some(super::ticketed::TicketExecutionEvidence {
                execution: Some(self.binding.execution),
                completed_nn_inputs: self.completed_nn_inputs,
                measurement_complete: self.measurement_complete,
                physical_fresh: self.fresh,
                accounting: self.owner_accounting.map(native_ticket_accounting),
            })
        }
    }
    fn native_ticket_accounting(
        accounting: PalsNativeOwnerAccounting,
    ) -> super::ticketed::TicketNativeAccountingEvidence {
        // 동일 binding으로 수신한 producer 소유 계수를 그대로 투영한다.
        // 명령 수·출력 유효성·검색 소비로 physical 사용량을 추정하지 않는다.
        super::ticketed::TicketNativeAccountingEvidence {
            public_cache_hits: accounting.public_cache_hits(),
            public_cache_misses: accounting.public_cache_misses(),
            public_nn_runs_attempted: accounting.public_nn_runs_attempted(),
            public_nn_runs_completed: accounting.public_nn_runs_completed(),
            public_nn_runs_failed_known: accounting.public_nn_runs_failed_known(),
            role_nn_runs_attempted: accounting.role_nn_runs_attempted(),
            role_nn_runs_completed: accounting.role_nn_runs_completed(),
            role_nn_runs_failed_known: accounting.role_nn_runs_failed_known(),
            validator_nn_runs_attempted: accounting.validator_nn_runs_attempted(),
            validator_nn_runs_completed: accounting.validator_nn_runs_completed(),
            validator_nn_runs_failed_known: accounting.validator_nn_runs_failed_known(),
        }
    }
    /// Optional bounded collector. All callbacks run on the calling/search
    /// thread and use its clock; startup probe inputs never enter this stream.
    pub trait NativeRoleObserver: Send {
        fn prepared(
            &mut self,
            id: RequestId,
            input: &PalsModelInput,
            context: NativePreparedContext<'_>,
        ) -> Result<(), RoleError>;
        /// Optional actual logical provenance; the legacy observer keeps its
        /// existing prepared callback and serialized input/lineage unchanged.
        fn prepared_with_logical_context(
            &mut self,
            id: RequestId,
            input: &PalsModelInput,
            context: NativePreparedContext<'_>,
            _logical: Option<&RoleLogicalContext>,
        ) -> Result<(), RoleError> {
            self.prepared(id, input, context)
        }
        /// Explicit Warm collector entry point. It never falls back to the
        /// legacy Fresh prepared callback or relabels a seeded invocation.
        fn prepared_private_warm(
            &mut self,
            _id: RequestId,
            _input: &PalsModelInput,
            _context: NativePreparedContext<'_>,
            _logical: Option<&RoleLogicalContext>,
            _prepared: NativePrivateWarmPrepared<'_>,
        ) -> Result<(), RoleError> {
            Err(RoleError::Backend(
                "collector does not declare the private Warm observation namespace".into(),
            ))
        }
        /// Called only after the actual worker submit and Lease ownership exist.
        /// Failure cannot undo physical admission; the runtime retains/drains it.
        fn dispatched(&mut self, _binding: NativeRoleExecutionBinding) -> Result<(), RoleError> {
            Ok(())
        }
        /// Reports the same actual binding as dispatched. Pending has no Ready
        /// event, and unknown is emitted at most once before any actual late Ready.
        fn terminal(
            &mut self,
            _binding: NativeRoleExecutionBinding,
            _event: NativeRoleTerminal<'_>,
        ) -> Result<(), RoleError> {
            Ok(())
        }
        fn accepted_context(
            &mut self,
            _id: RequestId,
            _acceptance: &RoleAcceptance<'_>,
        ) -> Result<(), RoleError> {
            Ok(())
        }
        fn recheck_prepared(&mut self, _event: RecheckPrepared<'_>) -> Result<(), RoleError> {
            Ok(())
        }
        fn recheck_finished(&mut self, _event: RecheckFinished<'_>) -> Result<(), RoleError> {
            Ok(())
        }
        fn physically_completed(
            &mut self,
            _id: RequestId,
            _result: Result<&rz_eval::pals_model::PalsRawOutput, &BackendError>,
        ) -> Result<(), RoleError> {
            Ok(())
        }
        /// Actual selection from this owner accompanies the original physical
        /// return. The immutable marker grants no additional Run/fence or current
        /// resident-bank observation. Legacy observers keep their old callback.
        fn physically_completed_with_execution(
            &mut self,
            id: RequestId,
            result: Result<&rz_eval::pals_model::PalsRawOutput, &BackendError>,
            _execution: &NativeExecutionReceipt,
        ) -> Result<(), RoleError> {
            self.physically_completed(id, result)
        }
        fn delivered(&mut self, _id: RequestId) -> Result<(), RoleError> {
            Ok(())
        }
        fn rejected(
            &mut self,
            _id: RequestId,
            _reason: NativeRoleRejection,
        ) -> Result<(), RoleError> {
            Ok(())
        }
        fn accepted(&mut self, _id: RequestId) -> Result<(), RoleError> {
            Ok(())
        }
    }
    /// Provider and admission declarations are not measured native/VRAM peaks.
    #[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
    pub struct NativeCudaLoadingIdentity {
        pub profile: &'static str,
        pub canonical_sha256: [u8; 32],
    }
    /// Explicit owner selection. These are bounded allocation declarations,
    /// not measured process, ORT workspace or VRAM peaks.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct NativeHostRecordPageLimits {
        pub max_page_entries: usize,
        pub max_page_bytes: u64,
        pub max_transient_bytes: u64,
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct NativeOwnerOptions {
        pub drain_limit: Duration,
        pub host_record_pages: Option<NativeHostRecordPageLimits>,
    }
    /// One caller-owned absolute invocation clock. Synchronous native loading is
    /// checked before/after each stage; this value does not interrupt that call.
    /// Cleanup always retains the original whole deadline, never `now + reserve`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct NativeInvocationBudget {
        started: Instant,
        execution_until: Instant,
        whole_until: Instant,
        cleanup_reserve: Duration,
    }
    impl NativeInvocationBudget {
        pub fn new(
            started: Instant,
            execution_until: Instant,
            whole_until: Instant,
            cleanup_reserve: Duration,
        ) -> Result<Self, RoleError> {
            Self::new_at(
                started,
                execution_until,
                whole_until,
                cleanup_reserve,
                Instant::now(),
            )
        }
        fn new_at(
            started: Instant,
            execution_until: Instant,
            whole_until: Instant,
            cleanup_reserve: Duration,
            now: Instant,
        ) -> Result<Self, RoleError> {
            if started > now
                || started >= execution_until
                || execution_until >= whole_until
                || cleanup_reserve.is_zero()
                || execution_until.checked_add(cleanup_reserve) != Some(whole_until)
                || now.checked_add(cleanup_reserve).is_none()
            {
                return Err(RoleError::Backend(
                    "Native invocation requires past S < E < W, E + positive finite D = W".into(),
                ));
            }
            Ok(Self {
                started,
                execution_until,
                whole_until,
                cleanup_reserve,
            })
        }
        pub const fn started(self) -> Instant {
            self.started
        }
        pub const fn execution_until(self) -> Instant {
            self.execution_until
        }
        pub const fn whole_until(self) -> Instant {
            self.whole_until
        }
        pub const fn cleanup_reserve(self) -> Duration {
            self.cleanup_reserve
        }
        fn check_at(self, now: Instant, cancel: &AtomicBool) -> Result<(), RoleError> {
            if cancel.load(Ordering::Acquire) {
                return Err(RoleError::Canceled);
            }
            if now >= self.execution_until {
                return Err(RoleError::Deadline);
            }
            Ok(())
        }
        fn failure(
            self,
            stage: NativeLoadStage,
            cause: RoleError,
            cancel: &AtomicBool,
            finish: Option<NativeRoleFinishHandle>,
        ) -> NativeLoadFailure {
            self.failure_at(stage, cause, Instant::now(), cancel, finish)
        }
        fn failure_at(
            self,
            stage: NativeLoadStage,
            cause: RoleError,
            now: Instant,
            cancel: &AtomicBool,
            finish: Option<NativeRoleFinishHandle>,
        ) -> NativeLoadFailure {
            NativeLoadFailure {
                stage,
                cause,
                elapsed: now.saturating_duration_since(self.started),
                execution_expired: now >= self.execution_until,
                whole_expired: now >= self.whole_until,
                canceled: cancel.load(Ordering::Acquire),
                finish,
                backend_error: None,
            }
        }
        fn check_stage(
            self,
            stage: NativeLoadStage,
            cancel: &AtomicBool,
        ) -> Result<(), NativeLoadFailure> {
            self.check_at(Instant::now(), cancel)
                .map_err(|cause| self.failure(stage, cause, cancel, None))
        }
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum NativeLoadStage {
        Admission,
        RuntimeLoad,
        BackendLoad,
        OwnerConstruction,
        PostConstruction,
    }
    /// Actual synchronous load failure and current clock/cancel facts. `finish`
    /// is an owned cleanup opportunity, not a physical shutdown/release receipt.
    /// None means cleanup is unobserved, including failures before owner capture;
    /// it never means zero workers, zero native work, or successful release.
    pub struct NativeLoadFailure {
        pub stage: NativeLoadStage,
        pub cause: RoleError,
        pub elapsed: Duration,
        pub execution_expired: bool,
        pub whole_expired: bool,
        pub canceled: bool,
        pub finish: Option<NativeRoleFinishHandle>,
        /// Owned original error from the direct runtime/backend loading call.
        /// RoleError-only owner construction and clock/admission paths leave
        /// this None (unobserved), not a complete C diagnostic journal.
        pub backend_error: Option<Box<BackendError>>,
    }
    impl NativeLoadFailure {
        fn with_backend_error(mut self, error: BackendError) -> Self {
            // Move the bounded original, including its owned local diagnostic;
            // do not reconstruct it from Display or clone native text.
            self.backend_error = Some(Box::new(error));
            self
        }
    }
    impl std::fmt::Debug for NativeLoadFailure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("NativeLoadFailure")
                .field("stage", &self.stage)
                .field("cause", &self.cause)
                .field("elapsed", &self.elapsed)
                .field("execution_expired", &self.execution_expired)
                .field("whole_expired", &self.whole_expired)
                .field("canceled", &self.canceled)
                .field("finish_handle_available", &self.finish.is_some())
                .field("backend_error", &self.backend_error)
                .finish()
        }
    }
    impl std::fmt::Display for NativeLoadFailure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Native invocation load failed at {:?}: {:?}",
                self.stage, self.cause
            )
        }
    }
    impl std::error::Error for NativeLoadFailure {}
    fn validate_cpu_fresh_invocation(
        config: &rz_eval::pals_onnx::PalsOnnxConfig,
        options: NativeOwnerOptions,
        budget: NativeInvocationBudget,
        cancel: &AtomicBool,
    ) -> Result<(), NativeLoadFailure> {
        budget.check_stage(NativeLoadStage::Admission, cancel)?;
        let validation = (|| {
            options.validate(config)?;
            if config.provider != rz_eval::onnx::Provider::Cpu
                || config.device_public_memory
                || options.host_record_pages.is_some()
                || options.drain_limit != budget.cleanup_reserve()
                || !(1..=2).contains(&config.intra_threads)
            {
                return Err(RoleError::Backend(
                    "Native invocation loader requires CPU Fresh, 1..=2 threads, no record pages and exact cleanup reserve".into(),
                ));
            }
            Ok(())
        })();
        validation
            .map_err(|cause| budget.failure(NativeLoadStage::Admission, cause, cancel, None))?;
        budget.check_stage(NativeLoadStage::Admission, cancel)
    }
    fn validate_cpu_invocation_bundle(bundle: Option<[u8; 32]>) -> Result<(), RoleError> {
        // This is only a decision over the actual pin getter in the factory.
        // None does not certify a CPU build, provider execution or source pin.
        if bundle.is_some() {
            return Err(RoleError::Backend(
                "CPU Fresh invocation refuses a CUDA native runtime bundle".into(),
            ));
        }
        Ok(())
    }
    fn accept_invocation_owner(
        mut model: NativeRoleModel,
        budget: NativeInvocationBudget,
        cancel: &AtomicBool,
    ) -> Result<NativeRoleModel, NativeLoadFailure> {
        if let Err(cause) = budget.check_at(Instant::now(), cancel) {
            let finish = Some(model.finish_handle());
            return Err(budget.failure(NativeLoadStage::PostConstruction, cause, cancel, finish));
        }
        // This private binding is issued only by the actual invocation factory.
        // A later caller cannot replace original S/E/W with a new relative window.
        model.cpu_invocation_budget = Some(budget);
        Ok(model)
    }
    /// Owned opt-in input. Execution identity, namespace, Session and placement
    /// are obtained from the loaded backend and this owner, never the caller.
    pub struct NativeCudaRecordPagesRegistration {
        pub checked_graph: rz_eval::pals_onnx::CheckedFixedPackingGraph,
        pub limits: rz_eval::pals_onnx::DevicePagesLimits,
        pub invocation: rz_eval::pals_onnx::DevicePageInvocationDeclaration,
        pub profile_root: std::path::PathBuf,
        pub frozen_epoch: u64,
    }
    impl NativeCudaRecordPagesRegistration {
        pub fn for_deployment(
            checked_graph: rz_eval::pals_onnx::CheckedFixedPackingGraph,
            limits: rz_eval::pals_onnx::DevicePagesLimits,
            invocation: rz_eval::pals_onnx::DevicePageInvocationDeclaration,
            profile_root: std::path::PathBuf,
        ) -> Self {
            Self {
                checked_graph,
                limits,
                invocation,
                profile_root,
                frozen_epoch: DEPLOYMENT_FROZEN_EPOCH,
            }
        }
    }
    /// Limits and backing reservations supplied at admission. These do not
    /// measure native allocator/session/VRAM peaks or prove completed work.
    #[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
    pub struct NativeCudaRecordPagesResourceDeclaration {
        pub max_blocks: usize,
        pub max_bank_whole_payload_bytes: u64,
        pub max_registry_entries: usize,
        pub max_container_host_bytes: u64,
        pub max_invocation_host_bytes: u64,
        pub max_invocation_device_bytes: u64,
        pub public_session_bytes: u64,
        pub packing_session_bytes: u64,
        pub private_session_bytes: u64,
        pub private_output_payload: NativeCudaRecordPagePayloadDeclaration,
        pub original_input_and_transfer_payload: NativeCudaRecordPagePayloadDeclaration,
        pub additional_owner_metadata_payload: NativeCudaRecordPagePayloadDeclaration,
    }
    impl NativeCudaRecordPagesResourceDeclaration {
        fn from_registration(
            registration: &NativeCudaRecordPagesRegistration,
        ) -> Result<Self, RoleError> {
            let limits = registration.limits;
            let invocation = registration.invocation;
            let nonzero = |value: Option<u64>| {
                value
                    .filter(|value| *value > 0)
                    .ok_or(RoleError::InvalidOutput)
            };
            if limits.max_blocks == 0
                || limits.max_registry_entries == 0
                || limits.max_bank_whole_payload_bytes == 0
                || limits.max_container_host_bytes == 0
                || limits.max_invocation_host_bytes == 0
                || limits.max_invocation_device_bytes == 0
            {
                return Err(RoleError::InvalidOutput);
            }
            let private_output_payload = invocation
                .private_output_payload
                .ok_or(RoleError::InvalidOutput)?;
            let original_input_and_transfer_payload = invocation
                .original_input_and_transfer_payload
                .ok_or(RoleError::InvalidOutput)?;
            let additional_owner_metadata_payload = invocation
                .additional_owner_metadata_payload
                .ok_or(RoleError::InvalidOutput)?;
            let public_session_bytes = nonzero(invocation.public_session_bytes)?;
            let packing_session_bytes = nonzero(invocation.packing_session_bytes)?;
            let private_session_bytes = nonzero(invocation.private_session_bytes)?;
            let sessions = public_session_bytes
                .checked_add(packing_session_bytes)
                .and_then(|n| n.checked_add(private_session_bytes))
                .ok_or(RoleError::InvalidOutput)?;
            let mut host = 0u64;
            let mut device = sessions;
            for payload in [
                private_output_payload,
                original_input_and_transfer_payload,
                additional_owner_metadata_payload,
            ] {
                if payload.total().map_err(|_| RoleError::InvalidOutput)? == 0 {
                    return Err(RoleError::InvalidOutput);
                }
                host = host
                    .checked_add(payload.host)
                    .ok_or(RoleError::InvalidOutput)?;
                device = device
                    .checked_add(payload.device)
                    .ok_or(RoleError::InvalidOutput)?;
            }
            let artifact = registration
                .checked_graph
                .registered_artifacts()
                .resources();
            if host > limits.max_invocation_host_bytes
                || device > limits.max_invocation_device_bytes
                || invocation.packing_session_bytes != Some(artifact.declared_packing_session_bytes)
                || additional_owner_metadata_payload.host
                    < registration
                        .checked_graph
                        .known_retained_host_bytes()
                        .map_err(|_| RoleError::InvalidOutput)?
            {
                return Err(RoleError::InvalidOutput);
            }
            Ok(Self {
                max_blocks: limits.max_blocks,
                max_bank_whole_payload_bytes: limits.max_bank_whole_payload_bytes,
                max_registry_entries: limits.max_registry_entries,
                max_container_host_bytes: limits.max_container_host_bytes,
                max_invocation_host_bytes: limits.max_invocation_host_bytes,
                max_invocation_device_bytes: limits.max_invocation_device_bytes,
                public_session_bytes,
                packing_session_bytes,
                private_session_bytes,
                private_output_payload: NativeCudaRecordPagePayloadDeclaration {
                    host: private_output_payload.host,
                    device: private_output_payload.device,
                },
                original_input_and_transfer_payload: NativeCudaRecordPagePayloadDeclaration {
                    host: original_input_and_transfer_payload.host,
                    device: original_input_and_transfer_payload.device,
                },
                additional_owner_metadata_payload: NativeCudaRecordPagePayloadDeclaration {
                    host: additional_owner_metadata_payload.host,
                    device: additional_owner_metadata_payload.device,
                },
            })
        }
    }
    /// Immutable selection from the actual installed owner. Initial live-bank
    /// counters are deliberately absent; later observation is a separate ACK.
    #[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
    pub struct NativeCudaRecordPagesDeclaration {
        pub schema: &'static str,
        pub mode: &'static str,
        pub process_epoch: u64,
        pub installation_game_generation: u64,
        pub frozen_epoch: u64,
        pub model_manifest_sha256: [u8; 32],
        pub model_epoch: [u8; 32],
        pub public_graph_sha256: [u8; 32],
        pub encoding_sha256: [u8; 32],
        pub runtime_sha256: [u8; 32],
        pub runtime_identity_scope: &'static str,
        pub device_id: u32,
        pub packing_graph_sha256: [u8; 32],
        pub packing_graph_bytes: u64,
        pub packing_manifest_sha256: [u8; 32],
        pub packing_manifest_bytes: u64,
        pub known_retained_artifact_host_bytes: u64,
        /// Tracked implementation provenance, independent of frozen model/input
        /// semantics and obtained from the actual backend implementation getter.
        pub implementation_sha256: [u8; 32],
        pub initialized_scope: &'static str,
        pub initialized_provider_count: usize,
        pub initialized_log_sha256: [u8; 32],
        pub initialized_log_digest_domain: &'static str,
        pub model_native_sessions: usize,
        pub packing_native_sessions: usize,
        /// The invocation upper bounds already include public/packing/private
        /// session declarations, bank/join/input/backing owners and metadata.
        /// They replace legacy transient/session aggregation for this selection.
        pub resource_accounting_scope: &'static str,
        pub declared_resources: NativeCudaRecordPagesResourceDeclaration,
    }
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeCudaRecordPageStatsReceipt {
        pub admitted_views: u64,
        pub public_subset_runs_attempted: u64,
        pub public_subset_runs_completed: u64,
        pub packing_runs_attempted: u64,
        pub packing_runs_completed: u64,
        pub private_joined_runs_attempted: u64,
        pub private_joined_runs_completed: u64,
        pub certified_board_slices: u64,
        pub certified_record_slices: u64,
        pub published_blocks: u64,
        pub last_unique_whole_pins: usize,
        pub last_reserved_host_bytes: u64,
        pub last_reserved_device_bytes: u64,
    }
    /// Actual worker observation, not an owned CUDA capability or a new fence.
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeCudaRecordPageSnapshotReceipt {
        pub schema: &'static str,
        pub initialized_scope: &'static str,
        pub initialized_provider_count: usize,
        pub packing_native_sessions: usize,
        pub initialized_log_sha256: [u8; 32],
        pub initialized_log_digest_domain: &'static str,
        pub packing_graph_sha256: [u8; 32],
        pub process_epoch: u64,
        pub game_generation: u64,
        pub model_manifest_sha256: [u8; 32],
        pub model_epoch: [u8; 32],
        pub public_graph_sha256: [u8; 32],
        pub encoding_sha256: [u8; 32],
        pub runtime_sha256: [u8; 32],
        pub runtime_identity_scope: &'static str,
        pub device_id: u32,
        pub live_blocks: usize,
        pub certified_projections: usize,
        pub whole_owner_host_bytes: u64,
        pub whole_owner_device_bytes: u64,
        pub active_invocation: bool,
        pub physical_completion_unknown: bool,
        pub quarantined: bool,
        pub stats: NativeCudaRecordPageStatsReceipt,
    }
    #[cfg(feature = "experimental-io-binding")]
    impl From<rz_eval::pals_onnx::CudaRecordPageSnapshot> for NativeCudaRecordPageSnapshotReceipt {
        fn from(value: rz_eval::pals_onnx::CudaRecordPageSnapshot) -> Self {
            let stats = value.stats;
            Self {
                schema: value.schema,
                initialized_scope: value.initialized_scope,
                initialized_provider_count: value.initialized_provider_count,
                packing_native_sessions: value.packing_native_sessions,
                initialized_log_sha256: value.initialized_log_sha256,
                initialized_log_digest_domain: value.initialized_log_digest_domain,
                packing_graph_sha256: value.packing_graph_sha256,
                process_epoch: value.process_epoch,
                game_generation: value.game_generation,
                model_manifest_sha256: value.model_manifest_sha256,
                model_epoch: value.model_epoch,
                public_graph_sha256: value.public_graph_sha256,
                encoding_sha256: value.encoding_sha256,
                runtime_sha256: value.runtime_sha256,
                runtime_identity_scope: value.runtime_identity_scope,
                device_id: value.device_id,
                live_blocks: value.live_blocks,
                certified_projections: value.certified_projections,
                whole_owner_host_bytes: value.whole_owner_host_bytes,
                whole_owner_device_bytes: value.whole_owner_device_bytes,
                active_invocation: value.active_invocation,
                physical_completion_unknown: value.physical_completion_unknown,
                quarantined: value.quarantined,
                stats: NativeCudaRecordPageStatsReceipt {
                    admitted_views: stats.admitted_views,
                    public_subset_runs_attempted: stats.public_subset_runs_attempted,
                    public_subset_runs_completed: stats.public_subset_runs_completed,
                    packing_runs_attempted: stats.packing_runs_attempted,
                    packing_runs_completed: stats.packing_runs_completed,
                    private_joined_runs_attempted: stats.private_joined_runs_attempted,
                    private_joined_runs_completed: stats.private_joined_runs_completed,
                    certified_board_slices: stats.certified_board_slices,
                    certified_record_slices: stats.certified_record_slices,
                    published_blocks: stats.published_blocks,
                    last_unique_whole_pins: stats.last_unique_whole_pins,
                    last_reserved_host_bytes: stats.last_reserved_host_bytes,
                    last_reserved_device_bytes: stats.last_reserved_device_bytes,
                },
            }
        }
    }
    fn validate_cuda_record_pages_selection(
        config: &rz_eval::pals_onnx::PalsOnnxConfig,
        options: NativeOwnerOptions,
        frozen_epoch: u64,
        warm: bool,
    ) -> Result<(), RoleError> {
        if !cfg!(all(
            feature = "experimental-io-binding",
            feature = "onnx-cuda",
            target_os = "linux"
        )) || !matches!(config.provider, rz_eval::onnx::Provider::Cuda { device_id, .. } if device_id >= 0)
            || config.device_public_memory
            || !config.cache_public_memory
            || options.host_record_pages.is_some()
            || warm
            || frozen_epoch != DEPLOYMENT_FROZEN_EPOCH
        {
            return Err(RoleError::Backend("Registered CUDA pages require the explicit Linux CUDA OwnCPU/Fresh path; incompatible selection or epoch".into()));
        }
        Ok(())
    }
    fn validate_cuda_record_page_session_declarations(
        config: &rz_eval::pals_onnx::PalsOnnxConfig,
        declaration: &NativeCudaRecordPagesResourceDeclaration,
    ) -> Result<(), RoleError> {
        let rz_eval::onnx::Provider::Cuda { arena_bytes, .. } = config.provider else {
            return Err(RoleError::InvalidOutput);
        };
        let arena = u64::try_from(arena_bytes).map_err(|_| RoleError::InvalidOutput)?;
        if declaration.public_session_bytes != arena || declaration.private_session_bytes != arena {
            return Err(RoleError::InvalidOutput);
        }
        Ok(())
    }
    #[cfg(feature = "experimental-io-binding")]
    fn installed_cuda_record_pages_declaration(
        backend: &PalsOnnxBackend,
        epoch: ProcessEpoch,
        frozen_epoch: u64,
        artifact: rz_eval::pals_onnx::PackingArtifactRegistration,
        retained: u64,
        declared_resources: NativeCudaRecordPagesResourceDeclaration,
    ) -> Result<NativeCudaRecordPagesDeclaration, RoleError> {
        let snapshot = backend
            .cuda_record_page_snapshot()
            .map_err(model_error)?
            .ok_or(RoleError::InvalidOutput)?;
        let public = backend
            .residency()
            .graphs
            .iter()
            .find(|graph| graph.role == "public")
            .ok_or(RoleError::InvalidOutput)?;
        let encoding = pals_fresh_encoding_semantic_digest();
        let device_id = match backend.config().provider {
            rz_eval::onnx::Provider::Cuda { device_id, .. } => {
                u32::try_from(device_id).map_err(|_| RoleError::InvalidOutput)?
            }
            _ => return Err(RoleError::InvalidOutput),
        };
        if snapshot.schema != "rz-pals-resident-cuda-record-pages-observation/1"
            || snapshot.process_epoch != epoch.0
            || snapshot.model_epoch != backend.model_epoch()
            || snapshot.model_manifest_sha256 != backend.manifest_digest()
            || snapshot.public_graph_sha256 != public.sha256
            || snapshot.encoding_sha256 != encoding
            || Some(snapshot.runtime_sha256) != backend.runtime_bundle_digest()
            || snapshot.device_id != device_id
            || snapshot.packing_graph_sha256 != artifact.graph.sha256
            || snapshot.initialized_scope != "initialized_provider_count"
            || snapshot.runtime_identity_scope != "closed_cuda_library_bundle"
            || snapshot.initialized_log_digest_domain
                != "rz-pals-fixed-packing-initialized-provider-count/1"
            || snapshot.initialized_provider_count != 275
            || snapshot.packing_native_sessions != 1
            || backend.residency().native_sessions != 2
            || snapshot.active_invocation
            || snapshot.physical_completion_unknown
            || snapshot.quarantined
            || frozen_epoch != DEPLOYMENT_FROZEN_EPOCH
        {
            return Err(RoleError::InvalidOutput);
        }
        Ok(NativeCudaRecordPagesDeclaration {
            schema: "rz-pals-native-cuda-record-pages-selection/1",
            mode: "registered-packing-v1",
            process_epoch: snapshot.process_epoch,
            installation_game_generation: snapshot.game_generation,
            frozen_epoch,
            model_manifest_sha256: snapshot.model_manifest_sha256,
            model_epoch: snapshot.model_epoch,
            public_graph_sha256: snapshot.public_graph_sha256,
            encoding_sha256: snapshot.encoding_sha256,
            runtime_sha256: snapshot.runtime_sha256,
            runtime_identity_scope: snapshot.runtime_identity_scope,
            device_id: snapshot.device_id,
            packing_graph_sha256: snapshot.packing_graph_sha256,
            packing_graph_bytes: artifact.graph.bytes,
            packing_manifest_sha256: artifact.manifest.sha256,
            packing_manifest_bytes: artifact.manifest.bytes,
            known_retained_artifact_host_bytes: retained,
            implementation_sha256: rz_eval::pals_onnx::cuda_record_pages_implementation_digest(),
            initialized_scope: snapshot.initialized_scope,
            initialized_provider_count: snapshot.initialized_provider_count,
            initialized_log_sha256: snapshot.initialized_log_sha256,
            initialized_log_digest_domain: snapshot.initialized_log_digest_domain,
            model_native_sessions: backend.residency().native_sessions,
            packing_native_sessions: snapshot.packing_native_sessions,
            resource_accounting_scope: "combined_resident_owner_invocation_including_sessions_and_backing",
            declared_resources,
        })
    }
    #[cfg(feature = "experimental-io-binding")]
    fn checked_cuda_record_page_observation(
        snapshot: NativeCudaRecordPageSnapshotReceipt,
        selection: &NativeCudaRecordPagesDeclaration,
    ) -> Result<NativeCudaRecordPageSnapshotReceipt, RoleError> {
        if snapshot.schema != "rz-pals-resident-cuda-record-pages-observation/1"
            || snapshot.process_epoch != selection.process_epoch
            || snapshot.game_generation < selection.installation_game_generation
            || snapshot.model_manifest_sha256 != selection.model_manifest_sha256
            || snapshot.model_epoch != selection.model_epoch
            || snapshot.public_graph_sha256 != selection.public_graph_sha256
            || snapshot.encoding_sha256 != selection.encoding_sha256
            || snapshot.runtime_sha256 != selection.runtime_sha256
            || snapshot.runtime_identity_scope != selection.runtime_identity_scope
            || snapshot.device_id != selection.device_id
            || snapshot.packing_graph_sha256 != selection.packing_graph_sha256
            || snapshot.initialized_scope != selection.initialized_scope
            || snapshot.initialized_provider_count != selection.initialized_provider_count
            || snapshot.initialized_log_sha256 != selection.initialized_log_sha256
            || snapshot.initialized_log_digest_domain != selection.initialized_log_digest_domain
            || snapshot.packing_native_sessions != selection.packing_native_sessions
            || snapshot.active_invocation
            || snapshot.physical_completion_unknown
            || snapshot.quarantined
            || snapshot.stats.last_reserved_host_bytes
                > selection.declared_resources.max_invocation_host_bytes
            || snapshot.stats.last_reserved_device_bytes
                > selection.declared_resources.max_invocation_device_bytes
        {
            return Err(RoleError::InvalidOutput);
        }
        Ok(snapshot)
    }
    impl NativeOwnerOptions {
        fn validate(self, config: &rz_eval::pals_onnx::PalsOnnxConfig) -> Result<(), RoleError> {
            if self.drain_limit.is_zero() || Instant::now().checked_add(self.drain_limit).is_none()
            {
                return Err(RoleError::Backend(
                    "PALS drain limit must be positive and finite".into(),
                ));
            }
            if let Some(limits) = self.host_record_pages {
                if config.device_public_memory || !config.cache_public_memory {
                    return Err(RoleError::Backend("PALS host record pages require host caching; device public memory combination is unsupported".into()));
                }
                if !(2..=1024).contains(&limits.max_page_entries)
                    || !(1..=16 * 1024 * 1024).contains(&limits.max_page_bytes)
                    || !(1..=16 * 1024 * 1024).contains(&limits.max_transient_bytes)
                {
                    return Err(RoleError::Backend(
                        "PALS host record page declarations are invalid".into(),
                    ));
                }
            }
            Ok(())
        }
    }
    #[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
    pub struct NativeHostRecordPageDeclaration {
        pub schema: &'static str,
        pub public_graph_sha256: [u8; 32],
        pub projection_semantics: &'static str,
        pub precision: &'static str,
        /// Source provenance of the implementation, separate from the frozen
        /// model, weights and full Rules/input encoding meaning.
        pub implementation_sha256: [u8; 32],
        pub max_page_entries: usize,
        pub max_page_bytes: u64,
        pub max_transient_bytes: u64,
    }
    impl NativeHostRecordPageDeclaration {
        fn from_policy(policy: HostRecordPagePolicy) -> Self {
            Self {
                schema: "rz-pals-host-record-pages/1",
                public_graph_sha256: policy.public_graph_sha256,
                projection_semantics: policy.projection_semantics,
                precision: "fp32",
                implementation_sha256: rz_eval::pals_onnx::host_record_page_implementation_digest(),
                max_page_entries: policy.max_page_entries,
                max_page_bytes: policy.max_page_bytes,
                max_transient_bytes: policy.max_transient_bytes,
            }
        }
        fn reserved_host_bytes(&self) -> Result<u64, RoleError> {
            self.max_page_bytes
                .checked_add(self.max_transient_bytes)
                .ok_or(RoleError::InvalidOutput)
        }
    }
    #[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
    pub struct NativeHostRecordPageBankReceipt {
        pub entries: usize,
        pub reserved_bytes: u64,
        pub pinned_entries: usize,
        pub pinned_bytes: u64,
        pub max_entries: usize,
        pub max_bytes: u64,
        pub entry_backing_bytes: u64,
    }
    #[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
    pub struct NativeHostRecordPageStatsReceipt {
        pub view_hits: u64,
        pub view_misses: u64,
        pub board_hits: u64,
        pub board_misses: u64,
        pub record_hits: u64,
        pub record_misses: u64,
        pub public_calls_attempted: u64,
        pub public_calls_completed: u64,
        pub encoded_record_tokens: u64,
        pub submitted_record_tokens: u64,
        pub contextual_board_encodes_completed: u64,
        pub joins_completed: u64,
        pub joined_bytes: u64,
        pub evicted_pages: u64,
        pub max_reserved_host_transient_bytes: u64,
    }
    #[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
    pub struct NativeHostRecordPageSnapshotReceipt {
        pub declaration: NativeHostRecordPageDeclaration,
        pub bank: NativeHostRecordPageBankReceipt,
        pub stats: NativeHostRecordPageStatsReceipt,
        pub active_pin_count: usize,
        pub retained_join_bytes: u64,
        pub active_subset_bytes: u64,
        pub active_join_backing_bytes: u64,
        pub active_full_input_bytes: u64,
        pub transient_reservation_bytes: u64,
        pub quarantined: bool,
    }
    impl From<HostRecordPageSnapshot> for NativeHostRecordPageSnapshotReceipt {
        fn from(snapshot: HostRecordPageSnapshot) -> Self {
            let bank = snapshot.bank;
            let stats = snapshot.stats;
            Self {
                declaration: NativeHostRecordPageDeclaration::from_policy(snapshot.policy),
                bank: NativeHostRecordPageBankReceipt {
                    entries: bank.entries,
                    reserved_bytes: bank.reserved_bytes,
                    pinned_entries: bank.pinned_entries,
                    pinned_bytes: bank.pinned_bytes,
                    max_entries: bank.max_entries,
                    max_bytes: bank.max_bytes,
                    entry_backing_bytes: bank.entry_backing_bytes,
                },
                stats: NativeHostRecordPageStatsReceipt {
                    view_hits: stats.view_hits,
                    view_misses: stats.view_misses,
                    board_hits: stats.board_hits,
                    board_misses: stats.board_misses,
                    record_hits: stats.record_hits,
                    record_misses: stats.record_misses,
                    public_calls_attempted: stats.public_calls_attempted,
                    public_calls_completed: stats.public_calls_completed,
                    encoded_record_tokens: stats.encoded_record_tokens,
                    submitted_record_tokens: stats.submitted_record_tokens,
                    contextual_board_encodes_completed: stats.contextual_board_encodes_completed,
                    joins_completed: stats.joins_completed,
                    joined_bytes: stats.joined_bytes,
                    evicted_pages: stats.evicted_pages,
                    max_reserved_host_transient_bytes: stats.max_reserved_host_transient_bytes,
                },
                active_pin_count: snapshot.active_pin_count,
                retained_join_bytes: snapshot.retained_join_bytes,
                active_subset_bytes: snapshot.active_subset_bytes,
                active_join_backing_bytes: snapshot.active_join_backing_bytes,
                active_full_input_bytes: snapshot.active_full_input_bytes,
                transient_reservation_bytes: snapshot.transient_reservation_bytes,
                quarantined: snapshot.quarantined,
            }
        }
    }
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeHostRecordPageCommandReceipt {
        pub command_ordinal: u64,
        pub boundary: HostRecordPageObservationBoundary,
        pub outcome: HostRecordPageObservationOutcome,
        pub snapshot: NativeHostRecordPageSnapshotReceipt,
    }
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeHostRecordPageObservationReceipt {
        pub schema: &'static str,
        pub status: HostRecordPageObservationStatus,
        pub attempted_command_ordinal: u64,
        /// Last actually captured command; missing/old remains explicit.
        pub latest: Option<NativeHostRecordPageCommandReceipt>,
    }
    /// Validate actual metadata without upgrading it to a native completion
    /// fence or a process peak measurement. Failed paths may retain an older
    /// snapshot and explicit diagnostic/physical-unknown status.
    pub(crate) fn validate_host_record_page_evidence(
        native: &NativeRoleReceipt,
        require_final: bool,
    ) -> Result<(), RoleError> {
        let Some(declaration) = &native.execution.host_record_pages else {
            return if native.host_record_page_observation.is_none() {
                Ok(())
            } else {
                Err(RoleError::InvalidOutput)
            };
        };
        if native.execution.device_public_memory
            || declaration.schema != "rz-pals-host-record-pages/1"
            || declaration.precision != "fp32"
            || declaration.projection_semantics
                != rz_eval::pals_model::INDEPENDENT_RECORD_PROJECTION_SEMANTICS
            || declaration.implementation_sha256
                != rz_eval::pals_onnx::host_record_page_implementation_digest()
            || !(2..=1024).contains(&declaration.max_page_entries)
            || !(1..=16 * 1024 * 1024).contains(&declaration.max_page_bytes)
            || !(1..=16 * 1024 * 1024).contains(&declaration.max_transient_bytes)
            || native
                .residency
                .graphs
                .iter()
                .filter(|graph| graph.role == "public")
                .count()
                != 1
            || !native.residency.graphs.iter().any(|graph| {
                graph.role == "public" && graph.sha256 == declaration.public_graph_sha256
            })
        {
            return Err(RoleError::InvalidOutput);
        }
        declaration.reserved_host_bytes()?;
        let Some(observation) = &native.host_record_page_observation else {
            return if require_final {
                Err(RoleError::InvalidOutput)
            } else {
                Ok(())
            };
        };
        if observation.schema != "rz-pals-host-record-page-observation/1" {
            return Err(RoleError::InvalidOutput);
        }
        if observation.status == HostRecordPageObservationStatus::Available
            && !observation.latest.as_ref().is_some_and(|latest| {
                latest.command_ordinal == observation.attempted_command_ordinal
            })
        {
            return Err(RoleError::InvalidOutput);
        }
        if let Some(latest) = &observation.latest {
            let snapshot = &latest.snapshot;
            let bank = &snapshot.bank;
            let stats = &snapshot.stats;
            if snapshot.declaration != *declaration
                || latest.command_ordinal > observation.attempted_command_ordinal
                || (latest.command_ordinal == 0)
                    != (latest.boundary == HostRecordPageObservationBoundary::BeforeWorker
                        && latest.outcome == HostRecordPageObservationOutcome::BeforeWorker)
                || bank.max_entries != declaration.max_page_entries
                || bank.entries > bank.max_entries
                || bank.pinned_entries > bank.entries
                || bank.pinned_bytes > bank.reserved_bytes
                || bank.reserved_bytes > bank.max_bytes
                || bank
                    .max_bytes
                    .checked_add(bank.entry_backing_bytes)
                    .is_none_or(|bytes| bytes > declaration.max_page_bytes)
                || stats.public_calls_completed > stats.public_calls_attempted
                || stats.encoded_record_tokens > stats.submitted_record_tokens
                || stats.contextual_board_encodes_completed != stats.public_calls_completed
                || stats.max_reserved_host_transient_bytes > declaration.max_transient_bytes
                || snapshot.transient_reservation_bytes > declaration.max_transient_bytes
                || snapshot.retained_join_bytes > declaration.max_transient_bytes
                || snapshot.active_subset_bytes > declaration.max_transient_bytes
                || snapshot.active_join_backing_bytes > declaration.max_transient_bytes
                || snapshot.active_full_input_bytes > declaration.max_transient_bytes
                || (latest.outcome == HostRecordPageObservationOutcome::PhysicalCompletionUnknown
                    && !snapshot.quarantined)
            {
                return Err(RoleError::InvalidOutput);
            }
        }
        if require_final {
            let latest = observation
                .latest
                .as_ref()
                .ok_or(RoleError::InvalidOutput)?;
            let snapshot = &latest.snapshot;
            if observation.status != HostRecordPageObservationStatus::Available
                || latest.command_ordinal != observation.attempted_command_ordinal
                || latest.command_ordinal == 0
                || latest.boundary != HostRecordPageObservationBoundary::SnapshotStats
                || latest.outcome != HostRecordPageObservationOutcome::ReturnedOk
                || snapshot.quarantined
                || snapshot.bank.pinned_entries != 0
                || snapshot.bank.pinned_bytes != 0
                || snapshot.active_pin_count != 0
                || snapshot.active_subset_bytes != 0
                || snapshot.active_join_backing_bytes != 0
                || snapshot.active_full_input_bytes != 0
                || snapshot.transient_reservation_bytes != 0
                || native.quarantined
                || native.physical_runs_in_flight != 0
                || !native.physical_shutdown_confirmed
                || !native.native_buffers_released
            {
                return Err(RoleError::InvalidOutput);
            }
        }
        Ok(())
    }
    impl From<HostRecordPageObservationSnapshot> for NativeHostRecordPageObservationReceipt {
        fn from(observation: HostRecordPageObservationSnapshot) -> Self {
            Self {
                schema: "rz-pals-host-record-page-observation/1",
                status: observation.status,
                attempted_command_ordinal: observation.attempted_command_ordinal,
                latest: observation
                    .latest
                    .map(|latest| NativeHostRecordPageCommandReceipt {
                        command_ordinal: latest.command_ordinal,
                        boundary: latest.boundary,
                        outcome: latest.outcome,
                        snapshot: latest.snapshot.into(),
                    }),
            }
        }
    }
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeExecutionReceipt {
        pub provider: &'static str,
        pub device_id: Option<i32>,
        /// ORT allocator limit per session, excluding unobserved allocations.
        pub session_arena_bytes: Option<u64>,
        pub runtime_sha256: [u8; 32],
        pub runtime_bundle_sha256: Option<[u8; 32]>,
        /// Identity of the explicitly loaded metadata-only control inventory.
        /// None means strict CUDA or CPU, not an observed placement witness.
        pub cuda_control_inventory_sha256: Option<[u8; 32]>,
        /// Only the actual explicitly selected nondefault runtime profile is
        /// recorded here. Missing preserves historic eager/CPU wire formats.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cuda_loading_profile: Option<NativeCudaLoadingIdentity>,
        /// Candidate-only actual loaded configuration, not plan or exit proof.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cudnn_conv_search_diagnostic: Option<rz_eval::pals_onnx::PalsCuDnnExecutionDiagnostic>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub startup_probe_timeout_ms: Option<u64>,
        pub transient_request_device_bytes: u64,
        pub transient_execution_device_bytes: u64,
        pub pinned_request_bytes: u64,
        pub device_public_memory: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub host_record_pages: Option<NativeHostRecordPageDeclaration>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub private_warm: Option<NativePrivateWarmDeclaration>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cuda_record_pages: Option<NativeCudaRecordPagesDeclaration>,
    }
    impl NativeExecutionReceipt {
        fn request_device_reservation(&self) -> u64 {
            // The selected whole-owner reservation already includes the original
            // input and transfers. Retain the legacy numbers as declarations in
            // the receipt, without charging their device backing a second time.
            if self.cuda_record_pages.is_some() {
                0
            } else {
                self.transient_request_device_bytes
            }
        }
        fn execution_device_reservation(&self) -> u64 {
            self.cuda_record_pages
                .as_ref()
                .map_or(self.transient_execution_device_bytes, |selection| {
                    selection.declared_resources.max_invocation_device_bytes
                })
        }
        fn runtime_device_reservation(&self) -> Result<u64, RoleError> {
            self.request_device_reservation()
                .checked_add(self.execution_device_reservation())
                .ok_or(RoleError::InvalidOutput)
        }
    }
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativePrivateWarmDeclaration {
        pub schema: &'static str,
        pub query_semantics: &'static str,
        pub export_manifest_sha256: [u8; 32],
        pub encoding_semantic_sha256: [u8; 32],
        pub model_epoch: [u8; 32],
        pub frozen_epoch: u64,
        pub implementation_sha256: [u8; 32],
        pub loader_implementation_sha256: [u8; 32],
        pub frozen_contexts_per_role: usize,
        pub accepted_seeds_per_role: usize,
        pub bank_bytes: u64,
        pub transient_bytes: u64,
        pub payload_bytes: u64,
        pub reserved_owner_bytes: u64,
        pub value_always_fresh: bool,
        pub native_cpu_loaded_capability: bool,
    }
    /// Initialization work is never converted into search role consumption.
    #[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum NativeStartupErrorKind {
        Unavailable,
        Unsupported,
        InvalidOutput,
        Canceled,
        Deadline,
        PhysicalCompletionUnknown,
        Backend,
    }
    impl From<&RoleError> for NativeStartupErrorKind {
        fn from(error: &RoleError) -> Self {
            match error {
                RoleError::Unavailable => Self::Unavailable,
                RoleError::Unsupported(_) => Self::Unsupported,
                RoleError::InvalidOutput => Self::InvalidOutput,
                RoleError::Canceled => Self::Canceled,
                RoleError::Deadline => Self::Deadline,
                RoleError::PhysicalCompletionUnknown => Self::PhysicalCompletionUnknown,
                RoleError::Backend(_) => Self::Backend,
            }
        }
    }
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeStartupInputObservation {
        pub startup_request_ordinal: Option<u8>,
        pub backend_owner_id: Option<u64>,
        pub role: PalsRole,
        pub input_key: [u8; 32],
        pub logical_records: usize,
        /// Observed from the immutable job retained by the actual Ready lease.
        pub retained_input_key: Option<[u8; 32]>,
        pub raw_decode_succeeded: Option<bool>,
    }
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeStartupCommandObservation {
        pub command: &'static str,
        pub admitted: bool,
        pub entered_elapsed_ns: Option<u64>,
        pub returned_elapsed_ns: Option<u64>,
        /// None means not observed, including an admitted, still pending job.
        pub physical_completion_confirmed: Option<bool>,
        pub return_error: Option<NativeStartupErrorKind>,
        pub fence_observation: Option<&'static str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub input_observation: Option<NativeStartupInputObservation>,
    }

    fn bind_startup_input_observations(
        snapshot: &mut PalsStartupStageSnapshot,
        commands: &[NativeStartupCommandObservation],
        witness: Option<&PalsCudaPlacementWitness>,
    ) {
        let complete = snapshot.snapshot_status == PalsStartupSnapshotStatus::Available
            && snapshot.capture_started == Some(true)
            && snapshot.capture_closed
            && !snapshot.overflow
            && !snapshot.recording_contended
            && !snapshot.recording_poisoned;
        for input in snapshot.input_observations.iter_mut().flatten() {
            input.completion_binding = None;
            input.completion_binding_error = Some("incomplete_input_diagnostic");
            if !complete {
                continue;
            }
            let mut matching = commands.iter().enumerate().filter(|(_, command)| {
                command.input_observation.as_ref().is_some_and(|identity| {
                    identity.startup_request_ordinal == Some(input.request_ordinal)
                })
            });
            let Some((ordinal, command)) = matching.next() else {
                input.completion_binding_error = Some("missing_retained_command_input");
                continue;
            };
            if matching.next().is_some() {
                input.completion_binding_error = Some("ambiguous_retained_command_input");
                continue;
            }
            let identity = command
                .input_observation
                .as_ref()
                .expect("matched input identity");
            let Some(owner) = input.backend_owner_id.filter(|owner| *owner != 0) else {
                input.completion_binding_error = Some("missing_backend_owner_identity");
                continue;
            };
            if identity.backend_owner_id != Some(owner)
                || identity.role != input.role
                || identity.input_key != input.input_key
                || input.prepared_input_key != input.input_key
                || identity.retained_input_key != Some(input.input_key)
                || identity.logical_records != input.logical_records
            {
                input.completion_binding_error = Some("retained_command_input_or_owner_mismatch");
                continue;
            }
            if !command.admitted
                || command.physical_completion_confirmed != Some(true)
                || command.return_error.is_some()
                || identity.raw_decode_succeeded != Some(true)
                || input.private_run_returned_ok != Some(true)
                || input.role_evaluation_returned_ok != Some(true)
                || !(input.public_run_returned_ok == Some(true)
                    || input.public_cache_hit == Some(true))
            {
                input.completion_binding_error = Some("command_completion_or_decode_not_confirmed");
                continue;
            }
            let Some(witness) = witness else {
                input.completion_binding_error = Some("missing_same_worker_profile_ack");
                continue;
            };
            input.completion_binding = Some(PalsStartupInputCompletionBinding {
                command_ordinal: ordinal + 1,
                backend_owner_id: owner,
                retained_input_key: identity.retained_input_key.expect("matched retained key"),
                physical_completion_confirmed: true,
                raw_decode_succeeded: true,
                inventory_sha256: witness.inventory_sha256,
                public_profile_sha256: witness.public.profile_sha256,
                shared_pc_profile_sha256: witness.shared_pc.profile_sha256,
            });
            input.completion_binding_error = None;
        }
    }
    /// A bounded failure-only timeline. Successful legacy wire stays unchanged.
    /// Loading time is observed at the actual CLI factory, not inferred from
    /// whole-process duration; library errors remain in last_failure separately.
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeStartupFailureDiagnostic {
        pub schema: &'static str,
        pub model_loading_elapsed_ns: Option<u64>,
        pub probe_budget_ns: Option<u64>,
        pub probe_elapsed_ns: Option<u64>,
        pub drain_limit_ns: Option<u64>,
        pub current_stage: &'static str,
        pub primary_error: Option<NativeStartupErrorKind>,
        pub logical_deadline_expired: bool,
        pub commands: Vec<NativeStartupCommandObservation>,
        pub command_observation_overflow: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub backend_stages: Option<PalsStartupStageSnapshot>,
    }
    /// Observed successful cold-start phases under an explicit probe budget.
    /// This is timing evidence, not failure evidence or resource/NN admission.
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeStartupTimingObservation {
        pub schema: &'static str,
        pub model_loading_elapsed_ns: Option<u64>,
        pub probe_budget_ns: Option<u64>,
        pub probe_elapsed_ns: Option<u64>,
        pub drain_limit_ns: Option<u64>,
        pub commands: Vec<NativeStartupCommandObservation>,
        pub command_observation_overflow: bool,
    }
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeStartupProbeReceipt {
        pub completed_proposer_calls: u64,
        pub completed_critic_calls: u64,
        pub runtime_mapping_confirmed: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub runtime_loading_mapping: Option<Box<PalsNativeMappingWitness>>,
        /// Same-worker NN-zero ACK after physically completed public/P/C probes.
        /// Library origin confirmation alone never supplies this evidence.
        pub cuda_placement_witness: Option<Box<PalsCudaPlacementWitness>>,
        pub reset_completed: bool,
        /// Actual cumulative snapshot after probe and reset; final statistics
        /// include this work and must not be reported as search-only inputs.
        pub backend_stats: Option<NativeBackendStatsReceipt>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub failure_diagnostic: Option<NativeStartupFailureDiagnostic>,
        /// Explicit diagnostic startup only. These events do not authorize
        /// readiness, GPU ownership, physical completion or NN consumption.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub backend_stages: Option<PalsStartupStageSnapshot>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub timing_observation: Option<NativeStartupTimingObservation>,
        /// Same-worker metadata ACK after the real startup probes and reset.
        /// Its absence is never filled with the initial installation snapshot.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cuda_record_page_observation: Option<NativeCudaRecordPageSnapshotReceipt>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cuda_record_page_observation_error: Option<NativeStartupErrorKind>,
    }
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeRoleSourceIdentity {
        pub checkpoint_sha256: [u8; 32],
        pub frozen_epoch: u64,
        pub export_manifest_sha256: [u8; 32],
        pub encoding_semantic_sha256: [u8; 32],
        pub adapter_source_sha256: [u8; 32],
        pub model_configuration: PalsModelConfig,
        pub trained: bool,
        pub execution: NativeExecutionReceipt,
    }
    fn transient_device_layout_for_config(c: &PalsModelConfig) -> Result<(u64, u64), RoleError> {
        use rz_eval::pals_model::{
            DIVERGENCE_FEATURES, METADATA_FEATURES, QUERY_FEATURES, RECORD_FEATURES,
        };
        let to_u64 = |n| u64::try_from(n).map_err(|_| RoleError::InvalidOutput);
        let r = to_u64(c.max_records)?;
        let candidates = to_u64(c.max_candidates)?;
        let divergences = to_u64(c.max_divergences)?;
        // Actual bounded ActiveInputs tensor layout: i64 board/candidates,
        // f32 numeric fields, bool masks and the shared P/C scalar condition.
        let full_line_inputs = if c.profile.uses_full_line() {
            // Exact fixed ABI: record/query move tokens, bool masks and two
            // local relationships per selected record. No device peak claim.
            r * 256 * (3 * 8 + 1) + 3 * 256 * (3 * 8 + 1) + r * 2 * 8
        } else {
            0
        };
        let inputs = 64 * 8
            + to_u64(METADATA_FEATURES)? * 4
            + r * to_u64(RECORD_FEATURES)? * 4
            + r
            + candidates * 3 * 8
            + candidates
            + divergences * to_u64(DIVERGENCE_FEATURES)? * 4
            + divergences
            + to_u64(QUERY_FEATURES)? * 4
            + 1
            + full_line_inputs;
        let tokens = to_u64(c.public_memory_tokens(c.max_records).map_err(model_error)?)?;
        // Two public K/V heads and mask plus private latent/policy/WDL/critic
        // outputs. Native graph workspace/weights remain the separately
        // declared session allocator domain; these are tensor reservations.
        let outputs = 2 * to_u64(c.kv_heads)? * tokens * to_u64(c.head_dimension)? * 4
            + tokens
            + to_u64(c.latent_elements())? * 4
            + candidates * 4
            + 3 * 4
            + divergences * 4;
        Ok((inputs, outputs))
    }
    fn cuda_warm_tensor_layout_for_config(
        config: &PalsModelConfig,
        cuda_warm: bool,
    ) -> Result<(u64, u64), RoleError> {
        let (inputs, outputs) = transient_device_layout_for_config(config)?;
        if !cuda_warm {
            return Ok((inputs, outputs));
        }
        let tokens = config
            .public_memory_tokens(config.max_records)
            .map_err(model_error)?;
        let kv_bytes = [2, config.kv_heads, tokens, config.head_dimension, 4]
            .into_iter()
            .try_fold(1_u64, |total, value| {
                total.checked_mul(u64::try_from(value).ok()?)
            })
            .ok_or(RoleError::InvalidOutput)?;
        let seed_bytes = u64::try_from(config.latent_elements())
            .ok()
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(1))
            .ok_or(RoleError::InvalidOutput)?;
        // The Warm ABI always supplies initial_latent and scalar warm_start,
        // even for Fresh. Public replacement allocates its new K/V before the
        // previous owned K/V are dropped. Charge both lifetimes, without
        // claiming ORT workspace/allocator peaks or loosening its payload cap.
        Ok((
            inputs
                .checked_add(seed_bytes)
                .ok_or(RoleError::InvalidOutput)?,
            outputs
                .checked_add(kv_bytes)
                .ok_or(RoleError::InvalidOutput)?,
        ))
    }
    fn cuda_tensor_layout_for_backend(backend: &PalsOnnxBackend) -> Result<(u64, u64), RoleError> {
        #[cfg(feature = "experimental-io-binding")]
        let cuda_warm = backend.private_cuda_warm_capability().is_some();
        #[cfg(not(feature = "experimental-io-binding"))]
        let cuda_warm = false;
        cuda_warm_tensor_layout_for_config(backend.model_config(), cuda_warm)
    }
    #[cfg(any(feature = "experimental-io-binding", test))]
    fn validate_cuda_warm_observation_admission(
        limits: NativeCudaWarmObservationLimits,
        execution: &NativeExecutionReceipt,
        actual_cuda_warm: bool,
    ) -> Result<(), BackendError> {
        use rz_eval::error::{FailureKind as K, FailureStage as S};
        let detail = if limits.max_leases != 1 {
            Some("CUDA Warm observation admission requires exactly one lease")
        } else if limits.device_bytes_max == 0 {
            Some("CUDA Warm observation device payload limit must be positive")
        } else if !actual_cuda_warm {
            Some("CUDA Warm observation admission requires the actual loaded CUDA Warm capability")
        } else if limits.device_bytes_max > execution.execution_device_reservation() {
            Some(
                "CUDA Warm observation payload limit exceeds the charged physical tensor reservation",
            )
        } else {
            None
        };
        detail.map_or(Ok(()), |detail| {
            Err(BackendError::new(K::InvalidInput, S::Admission, detail))
        })
    }
    fn execution_receipt(backend: &PalsOnnxBackend) -> Result<NativeExecutionReceipt, RoleError> {
        let (provider, device_id, session_arena_bytes, request, physical) =
            match backend.config().provider {
                rz_eval::onnx::Provider::Cpu => ("cpu", None, None, 0, 0),
                rz_eval::onnx::Provider::Cuda {
                    device_id,
                    arena_bytes,
                } => {
                    if !cfg!(all(feature = "onnx-cuda", target_os = "linux")) {
                        return Err(RoleError::Backend(
                            "PALS CUDA requires onnx-cuda on Linux; CPU fallback forbidden".into(),
                        ));
                    }
                    let (request, physical) = cuda_tensor_layout_for_backend(backend)?;
                    (
                        "cuda",
                        Some(device_id),
                        Some(u64::try_from(arena_bytes).map_err(|_| RoleError::InvalidOutput)?),
                        request,
                        physical,
                    )
                }
            };
        let cuda_loading_profile = match backend.native_loading_profile() {
            Some(rz_eval::onnx::NativeLoadingProfile::CuDnnShimLazyV1) => {
                if provider != "cuda"
                    || backend.config().device_public_memory
                    || backend.cuda_control_inventory_digest().is_none()
                {
                    return Err(RoleError::Backend("PALS experimental loading policy requires explicit CUDA host/control startup".into()));
                }
                Some(NativeCudaLoadingIdentity {
                    profile: rz_eval::onnx::NativeLoadingProfile::CuDnnShimLazyV1.identifier(),
                    canonical_sha256: backend
                        .native_loading_profile_digest()
                        .ok_or(RoleError::InvalidOutput)?,
                })
            }
            Some(rz_eval::onnx::NativeLoadingProfile::EagerCuda12Cudnn9V1) | None => None,
        };
        let private_warm =
            backend
                .private_warm_capability()
                .map(|cap| NativePrivateWarmDeclaration {
                    schema: cap.schema(),
                    query_semantics: cap.query_semantics(),
                    export_manifest_sha256: cap.manifest_digest(),
                    encoding_semantic_sha256: cap.private_model_identity().encoding,
                    model_epoch: cap.model_epoch(),
                    frozen_epoch: cap.private_model_identity().frozen_epoch,
                    implementation_sha256: private_warm::source_digest(),
                    loader_implementation_sha256: cap.implementation_digest(),
                    frozen_contexts_per_role: private_warm::MAX_FROZEN_CONTEXTS_PER_ROLE,
                    accepted_seeds_per_role: 1,
                    bank_bytes: private_warm::BANK_BYTES,
                    transient_bytes: private_warm::TRANSIENT_BYTES,
                    payload_bytes: private_warm::PAYLOAD_BYTES,
                    reserved_owner_bytes: private_warm::OWNER_RESERVATION_BYTES,
                    value_always_fresh: true,
                    native_cpu_loaded_capability: true,
                });
        #[cfg(feature = "experimental-io-binding")]
        let private_warm = private_warm.or_else(|| {
            backend
                .private_cuda_warm_capability()
                .map(|cap| NativePrivateWarmDeclaration {
                    schema: cap.schema(),
                    query_semantics: cap.query_semantics(),
                    export_manifest_sha256: cap.manifest_digest(),
                    encoding_semantic_sha256: cap.private_model_identity().encoding,
                    model_epoch: cap.model_epoch(),
                    frozen_epoch: cap.private_model_identity().frozen_epoch,
                    implementation_sha256: private_warm::source_digest(),
                    loader_implementation_sha256: cap.implementation_digest(),
                    frozen_contexts_per_role: 0,
                    accepted_seeds_per_role: 1,
                    bank_bytes: private_warm::BANK_BYTES,
                    transient_bytes: private_warm::TRANSIENT_BYTES,
                    payload_bytes: private_warm::CUDA_PAYLOAD_BYTES,
                    reserved_owner_bytes: private_warm::CUDA_OWNER_RESERVATION_BYTES,
                    value_always_fresh: true,
                    native_cpu_loaded_capability: false,
                })
        });
        Ok(NativeExecutionReceipt {
            provider,
            device_id,
            session_arena_bytes,
            runtime_sha256: backend.runtime_binary_digest(),
            runtime_bundle_sha256: backend.runtime_bundle_digest(),
            cuda_control_inventory_sha256: backend.cuda_control_inventory_digest(),
            cuda_loading_profile,
            cudnn_conv_search_diagnostic: backend
                .cudnn_conv_search_diagnostic()
                .map_err(model_error)?,
            startup_probe_timeout_ms: None,
            transient_request_device_bytes: request,
            transient_execution_device_bytes: physical,
            // Rust owns pageable host tensors in this first host-K/V path.
            // Unknown internal ORT staging is not a claimed observed zero.
            pinned_request_bytes: 0,
            device_public_memory: backend.config().device_public_memory,
            host_record_pages: backend
                .host_record_page_snapshot()
                .map(|snapshot| NativeHostRecordPageDeclaration::from_policy(snapshot.policy)),
            private_warm,
            cuda_record_pages: None,
        })
    }
    pub(crate) fn validate_runtime_loading_mapping(
        witness: &PalsNativeMappingWitness,
        execution: &NativeExecutionReceipt,
    ) -> Result<(), BackendError> {
        let identity = execution.cuda_loading_profile.as_ref();
        fn set(names: &[String]) -> std::collections::BTreeSet<&str> {
            names.iter().map(String::as_str).collect()
        }
        let required = set(&witness.required_nvidia_files);
        let mapped = set(&witness.mapped_nvidia_files);
        let absent = set(&witness.deferred_nvidia_not_mapped);
        let ort = set(&witness.mapped_ort_files);
        let declared = std::collections::BTreeSet::from(rz_eval::onnx::NVIDIA_LOAD_ORDER);
        let expected_required: std::collections::BTreeSet<_> =
            rz_eval::onnx::NativeLoadingProfile::CuDnnShimLazyV1
                .eager_indices()
                .iter()
                .map(|&index| rz_eval::onnx::NVIDIA_LOAD_ORDER[index])
                .collect();
        if execution.provider != "cuda"
            || witness.schema != "rovezero.pals-native-mapping-witness.v1"
            || witness.scope != "exclusive_physical_worker_full_runtime_origin"
            || witness.runtime_sha256 != execution.runtime_sha256
            || execution.runtime_bundle_sha256 != Some(witness.runtime_bundle_sha256)
            || !identity.is_some_and(|identity| {
                identity.profile == witness.loading_profile
                    && identity.profile
                        == rz_eval::onnx::NativeLoadingProfile::CuDnnShimLazyV1.identifier()
                    && identity.canonical_sha256 == witness.loading_profile_sha256
                    && identity.canonical_sha256
                        == rz_eval::asset::sha256(
                            rz_eval::onnx::NativeLoadingProfile::CuDnnShimLazyV1
                                .canonical_descriptor()
                                .as_bytes(),
                        )
            })
            || witness.declared_nvidia_files != 16
            || required.len() != witness.required_nvidia_files.len()
            || required != expected_required
            || mapped.len() != witness.mapped_nvidia_files.len()
            || absent.len() != witness.deferred_nvidia_not_mapped.len()
            || !required.is_subset(&mapped)
            || !mapped.is_disjoint(&absent)
            || mapped
                .union(&absent)
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                != declared
            || mapped
                .iter()
                .chain(absent.iter())
                .any(|name| name.is_empty() || name.len() > 128 || !name.is_ascii())
            || ort.len() != witness.mapped_ort_files.len()
            || ort != std::collections::BTreeSet::from(rz_eval::onnx::ORT_LIBRARY_NAMES)
        {
            return Err(BackendError::new(
                rz_eval::error::FailureKind::IdentityMismatch,
                rz_eval::error::FailureStage::Output,
                "PALS runtime loading mapping ACK differs from its actual pinned execution identity",
            ));
        }
        Ok(())
    }
    fn validate_startup_cuda_witness(
        witness: &PalsCudaPlacementWitness,
        execution: &NativeExecutionReceipt,
        manifest: [u8; 32],
        residency: &PalsSessionResidency,
    ) -> Result<(), RoleError> {
        // The backend owns exact opcode/provenance/profile validation. This
        // worker boundary separately binds its typed ACK to this actual owner.
        let matching_graphs = witness.initialization.len() == 2
            && ["public", "shared_pc"].iter().all(|role| {
                let mut placements = witness
                    .initialization
                    .iter()
                    .filter(|placement| placement.role == *role);
                let Some(placement) = placements.next() else {
                    return false;
                };
                placements.next().is_none()
                    && placement.assigned_nodes > 0
                    && placement.cuda_nodes > 0
                    && placement.optimization == PalsGraphOptimization::Disable
                    && placement
                        .cuda_nodes
                        .checked_add(placement.approved_cpu_control_nodes)
                        == Some(placement.assigned_nodes)
                    && residency
                        .graphs
                        .iter()
                        .any(|graph| graph.role == *role && graph.sha256 == placement.graph_sha256)
            });
        if execution.provider != "cuda"
            || execution.cuda_control_inventory_sha256 != Some(witness.inventory_sha256)
            || witness.manifest_sha256 != manifest
            || witness.runtime_sha256 != execution.runtime_sha256
            || execution.runtime_bundle_sha256 != Some(witness.runtime_bundle_sha256)
            || witness.schema != "rovezero.pals-cuda-metadata-control.v2"
            || witness.optimization != PalsGraphOptimization::Disable
            || !matching_graphs
            || witness.public.cuda_kernels == 0
            || witness.public.cuda_transfer_kernels != 0
            || witness.public.neural_kernels == 0
            || witness.shared_pc.cuda_kernels == 0
            || witness.shared_pc.cuda_transfer_kernels > witness.shared_pc.cuda_kernels
            || witness.shared_pc.proposer_private_kernels == 0
            || witness.shared_pc.critic_private_kernels == 0
        {
            return Err(RoleError::Backend(
                "PALS CUDA placement ACK is incomplete or belongs to a different pinned owner"
                    .into(),
            ));
        }
        Ok(())
    }
    /// Only an acknowledged exclusive worker snapshot supplies these counters.
    /// No semantic role completion count is expanded into guessed graph work.
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeBackendStatsReceipt {
        pub admitted_role_requests: u64,
        pub public_cache_hits: u64,
        pub public_cache_misses: u64,
        pub public_nn_runs_attempted: u64,
        pub public_nn_runs_completed: u64,
        pub public_nn_runs_failed_known: u64,
        pub role_nn_runs_attempted: u64,
        pub role_nn_runs_completed: u64,
        pub role_nn_runs_failed_known: u64,
        pub completed_nn_inputs: u64,
        pub validated_public_outputs: u64,
        pub validated_role_outputs: u64,
        pub new_game_resets: u64,
        pub live_public_cache_entries: u64,
    }
    impl From<PalsBackendStats> for NativeBackendStatsReceipt {
        fn from(value: PalsBackendStats) -> Self {
            Self {
                admitted_role_requests: value.admitted_role_requests,
                public_cache_hits: value.public_cache_hits,
                public_cache_misses: value.public_cache_misses,
                public_nn_runs_attempted: value.public_nn_runs_attempted,
                public_nn_runs_completed: value.public_nn_runs_completed,
                public_nn_runs_failed_known: value.public_nn_runs_failed_known,
                role_nn_runs_attempted: value.role_nn_runs_attempted,
                role_nn_runs_completed: value.role_nn_runs_completed,
                role_nn_runs_failed_known: value.role_nn_runs_failed_known,
                completed_nn_inputs: value.completed_nn_inputs,
                validated_public_outputs: value.validated_public_outputs,
                validated_role_outputs: value.validated_role_outputs,
                new_game_resets: value.new_game_resets,
                live_public_cache_entries: value.live_public_cache_entries,
            }
        }
    }
    fn require_recorded_nn_zero(stats: &NativeBackendStatsReceipt) -> Result<(), RoleError> {
        if [
            stats.admitted_role_requests,
            stats.public_cache_hits,
            stats.public_cache_misses,
            stats.public_nn_runs_attempted,
            stats.public_nn_runs_completed,
            stats.public_nn_runs_failed_known,
            stats.role_nn_runs_attempted,
            stats.role_nn_runs_completed,
            stats.role_nn_runs_failed_known,
            stats.completed_nn_inputs,
            stats.validated_public_outputs,
            stats.validated_role_outputs,
            stats.new_game_resets,
            stats.live_public_cache_entries,
        ]
        .into_iter()
        .any(|count| count != 0)
        {
            return Err(RoleError::Backend("Recorded CPU invocation already performed NN/cache/role work before pristine fence".into()));
        }
        Ok(())
    }
    /// Bounded typed diagnostics. Library message text and paths are never
    /// copied into a portable receipt.
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeFailureReceipt {
        pub kind: String,
        pub stage: String,
        pub detail: &'static str,
        pub cause_code: Option<String>,
        pub cause_prefix_sha256: Option<[u8; 32]>,
        pub cause_hashed_bytes: Option<u16>,
        pub cause_truncated: Option<bool>,
    }
    impl WorkerOwner {
        fn invalidate_game_observations(&self) {
            *self.final_stats.lock().unwrap_or_else(|e| e.into_inner()) = None;
            *self
                .cuda_record_page_snapshot
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = None;
            *self
                .cuda_record_page_snapshot_error
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = None;
        }
        fn request_game_target(&self, target: u64) -> Result<(), RoleError> {
            let _guard = self
                .game_lifecycle
                .lock()
                .map_err(|_| RoleError::Unavailable)?;
            if self.finishing.load(Ordering::Acquire) || self.shutdown.load(Ordering::Acquire) {
                return Err(RoleError::Unavailable);
            }
            if target <= self.game_generation.load(Ordering::Acquire) {
                return Err(RoleError::InvalidOutput);
            }
            self.game_generation.store(target, Ordering::Release);
            self.pending_game_target.store(target, Ordering::Release);
            self.invalidate_game_observations();
            Ok(())
        }
        fn remember_game_reset_failure(&self, error: RoleError) -> RoleError {
            let mut first = self
                .game_reset_failure
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            first.get_or_insert(error).clone()
        }
        fn accept_game_target(&self, target: u64) -> Result<(), RoleError> {
            if self.game_generation.load(Ordering::Acquire) != target
                || self.pending_game_target.load(Ordering::Acquire) != target
            {
                return Err(RoleError::Backend(
                    "PALS reset ACK target differs from current logical game".into(),
                ));
            }
            if let Some(warm) = &self.private_warm {
                warm.state.reset_after_known_fence(target)?;
            }
            // Old ACKs can never spend a newer target. Invalidate again before
            // admitting fresh final observations after this actual reset fence.
            self.invalidate_game_observations();
            self.pending_game_target
                .compare_exchange(target, 0, Ordering::AcqRel, Ordering::Acquire)
                .map_err(|_| {
                    RoleError::Backend("PALS pending reset target changed before ACK".into())
                })?;
            self.new_game_resets.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }
        fn execution_receipt(&self) -> NativeExecutionReceipt {
            let mut receipt = self.execution.clone();
            let selected = self.startup_probe_timeout_ms.load(Ordering::Acquire);
            receipt.startup_probe_timeout_ms = (selected != 0).then_some(selected);
            receipt
        }
        fn observe(
            &self,
            stage: &'static str,
            callback: impl FnOnce(&mut dyn NativeRoleObserver) -> Result<(), RoleError>,
        ) -> Result<(), RoleError> {
            let result = match self.observer.lock() {
                Ok(mut slot) => match slot.as_mut() {
                    Some(observer) => {
                        // Observer code must not unwind out of a post-submit
                        // handoff and drop a still unconfirmed physical Lease.
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            callback(observer.as_mut())
                        }))
                        .unwrap_or_else(|_| {
                            Err(RoleError::Backend("PALS observer callback panicked".into()))
                        })
                    }
                    None => Ok(()),
                },
                Err(_) => Err(RoleError::Backend("PALS observer owner poisoned".into())),
            };
            if result.is_err() {
                self.observer_failures.fetch_add(1, Ordering::AcqRel);
                *self
                    .last_observer_failure
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Some(NativeFailureReceipt {
                    kind: "Observer".into(),
                    stage: stage.into(),
                    detail: "bounded native role observer failed",
                    cause_code: None,
                    cause_prefix_sha256: None,
                    cause_hashed_bytes: None,
                    cause_truncated: None,
                });
                // A logging/sink error cannot retroactively make an already
                // completed physical native run become completion-unknown.
                return Err(RoleError::Backend(
                    "PALS native role observer failed; actual physical work is retained".into(),
                ));
            }
            Ok(())
        }
        fn remember_role_observer_error(
            &self,
            binding: NativeRoleExecutionBinding,
            error: RoleError,
        ) {
            let mut slot = self
                .role_execution
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(observed) = slot.as_mut()
                && observed.binding == binding
                && observed.observer_error.is_none()
            {
                observed.observer_error = Some(error);
            }
        }
        fn observe_role_terminal(
            &self,
            binding: NativeRoleExecutionBinding,
            event: NativeRoleTerminal<'_>,
        ) -> Result<(), RoleError> {
            // Pure bounded state is released before invoking observer code. The
            // Ready and unknown flags are separate: a real late Ready is retained
            // without converting the preceding unknown into success.
            {
                let mut slot = self
                    .role_execution
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                let Some(observed) = slot.as_mut().filter(|o| o.binding == binding) else {
                    return Err(RoleError::Backend(
                        "PALS role Lease observation binding mismatch".into(),
                    ));
                };
                let notified = match event {
                    NativeRoleTerminal::Ready(_) => &mut observed.ready_observed,
                    NativeRoleTerminal::CompletionUnknown(_) => {
                        if observed.ready_observed {
                            return Ok(());
                        }
                        &mut observed.unknown_observed
                    }
                };
                if *notified {
                    return Ok(());
                }
                *notified = true;
            }
            self.observe("bound_physical_terminal", |observer| {
                observer.terminal(binding, event)
            })
        }
        fn observe_role_unknown(&self, request: RequestId, reason: NativeRoleCompletionUnknown) {
            let binding = self
                .role_execution
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .filter(|o| o.binding.request == request && !o.ready_observed)
                .map(|o| o.binding);
            if let Some(binding) = binding
                && let Err(error) = self
                    .observe_role_terminal(binding, NativeRoleTerminal::CompletionUnknown(reason))
            {
                self.remember_role_observer_error(binding, error);
            }
        }
        fn remember_failure(&self, error: &BackendError) {
            let receipt = NativeFailureReceipt {
                kind: format!("{:?}", error.kind),
                stage: format!("{:?}", error.stage),
                detail: error.detail,
                cause_code: error.cause.map(|c| format!("{:?}", c.code)),
                cause_prefix_sha256: error.cause.map(|c| c.prefix_sha256),
                cause_hashed_bytes: error.cause.map(|c| c.hashed_bytes),
                cause_truncated: error.cause.map(|c| c.truncated),
            };
            *self.last_failure.lock().unwrap_or_else(|e| e.into_inner()) = Some(receipt);
        }
    }
    #[derive(Clone)]
    pub struct NativeRoleFinishHandle {
        owner: Arc<WorkerOwner>,
    }
    #[derive(Clone, Debug, serde::Serialize)]
    pub struct NativeRoleReceipt {
        /// Completed native role calls, including a physically completed failure.
        pub physically_completed_role_calls: u64,
        /// Successful raw native role callbacks, distinct from actual ONNX
        /// graph inputs. Resets, Stats controls and failed callbacks are excluded.
        pub completed_role_inputs: u64,
        pub failed_physical_role_calls: u64,
        pub invalid_role_outputs: u64,
        pub delivered_role_inputs: u64,
        pub search_consumed_role_inputs: u64,
        pub canceled_requests: u64,
        pub expired_requests: u64,
        pub completed_new_game_resets: u64,
        pub process_epoch: u64,
        pub game_generation: u64,
        /// Actual immutable deployment epoch used by both scope and input keys.
        pub frozen_epoch: u64,
        pub request_high_water: u64,
        pub execution_high_water: u64,
        pub physical_runs_in_flight: u64,
        pub quarantined: bool,
        pub physical_shutdown_confirmed: bool,
        /// Native backend closure/session destruction is confirmed by a join;
        /// this does not attest destruction of unrelated Rust search records.
        pub native_buffers_released: bool,
        pub model_epoch: [u8; 32],
        pub export_manifest_sha256: [u8; 32],
        pub encoding_semantic_sha256: [u8; 32],
        pub adapter_source_sha256: [u8; 32],
        /// Export declaration, not independent proof that optimizer steps ran.
        pub trained: bool,
        pub residency: PalsSessionResidency,
        pub last_failure: Option<NativeFailureReceipt>,
        /// None means unobserved. A pre-shutdown cache count is not a claim that
        /// native cache buffers survive the subsequent confirmed worker join.
        pub backend_stats: Option<NativeBackendStatsReceipt>,
        pub backend_stats_observation: Option<&'static str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub host_record_page_observation: Option<NativeHostRecordPageObservationReceipt>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub private_warm_observation: Option<NativePrivateWarmObservation>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub private_warm_observation_unavailable: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cuda_record_page_observation: Option<NativeCudaRecordPageSnapshotReceipt>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cuda_record_page_observation_error: Option<NativeStartupErrorKind>,
        pub execution: NativeExecutionReceipt,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cuda_record_page_observation_scope: Option<&'static str>,
        pub startup_probe: Option<NativeStartupProbeReceipt>,
        /// None means not observed/applicable, not a failed or successful audit.
        pub final_runtime_mapping_confirmed: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub final_runtime_loading_mapping: Option<Box<PalsNativeMappingWitness>>,
        pub observer_failures: u64,
        pub last_observer_failure: Option<NativeFailureReceipt>,
    }
    pub(crate) fn validate_private_warm_evidence(
        receipt: &NativeRoleReceipt,
        require_complete: bool,
    ) -> Result<(), RoleError> {
        validate_private_warm_evidence_with_terminal_scope(receipt, require_complete, None)
    }
    pub(crate) fn validate_private_warm_evidence_with_terminal_scope(
        receipt: &NativeRoleReceipt,
        require_complete: bool,
        terminal: Option<&crate::pals_attestation::followup::NativeCudaWarmObservationV4>,
    ) -> Result<(), RoleError> {
        let Some(declaration) = &receipt.execution.private_warm else {
            return if receipt.private_warm_observation.is_none()
                && receipt.private_warm_observation_unavailable.is_none()
            {
                Ok(())
            } else {
                Err(RoleError::InvalidOutput)
            };
        };
        let domain = if declaration.schema == rz_eval::pals_onnx::PRIVATE_WARM_SCHEMA {
            receipt.execution.provider == "cpu"
                && !receipt.execution.device_public_memory
                && declaration.query_semantics == rz_eval::pals_onnx::FROZEN_QUERY_SEMANTICS_V1
                && declaration.frozen_contexts_per_role
                    == private_warm::MAX_FROZEN_CONTEXTS_PER_ROLE
                && declaration.payload_bytes == private_warm::PAYLOAD_BYTES
                && declaration.reserved_owner_bytes == private_warm::OWNER_RESERVATION_BYTES
                && declaration.native_cpu_loaded_capability
        } else {
            #[cfg(feature = "experimental-io-binding")]
            {
                declaration.schema == rz_eval::pals_onnx::PRIVATE_CUDA_WARM_SCHEMA
                    && receipt.execution.provider == "cuda"
                    && receipt.execution.device_public_memory
                    && declaration.query_semantics
                        == rz_eval::pals_onnx::PRIVATE_CUDA_QUERY_SEMANTICS
                    && declaration.frozen_contexts_per_role == 0
                    && declaration.payload_bytes == private_warm::CUDA_PAYLOAD_BYTES
                    && declaration.reserved_owner_bytes
                        == private_warm::CUDA_OWNER_RESERVATION_BYTES
                    && !declaration.native_cpu_loaded_capability
            }
            #[cfg(not(feature = "experimental-io-binding"))]
            {
                false
            }
        };
        if !domain
            || declaration.export_manifest_sha256 != receipt.export_manifest_sha256
            || declaration.encoding_semantic_sha256 != receipt.encoding_semantic_sha256
            || declaration.model_epoch != receipt.model_epoch
            || declaration.frozen_epoch != receipt.frozen_epoch
            || declaration.implementation_sha256 != private_warm::source_digest()
            || declaration.loader_implementation_sha256 == [0; 32]
            || declaration.accepted_seeds_per_role != 1
            || declaration.bank_bytes != private_warm::BANK_BYTES
            || declaration.transient_bytes != private_warm::TRANSIENT_BYTES
            || !declaration.value_always_fresh
        {
            return Err(RoleError::InvalidOutput);
        }
        let Some(observation) = &receipt.private_warm_observation else {
            return if !require_complete
                && receipt.private_warm_observation_unavailable == Some(true)
            {
                Ok(())
            } else {
                Err(RoleError::InvalidOutput)
            };
        };
        if receipt.private_warm_observation_unavailable.is_some()
            || observation.entries_per_role.iter().any(|&n| n > 1)
            || observation.entries_per_role[2] != 0
            || observation.reserved_bank_bytes > declaration.bank_bytes
            || observation.provisional_bytes > declaration.transient_bytes
            || observation.retained_full_payload_bytes > declaration.payload_bytes
            || observation
                .frozen_contexts_per_role
                .iter()
                .any(|&n| n > declaration.frozen_contexts_per_role)
            || observation
                .first_query_input_sha256_per_role
                .iter()
                .any(|v| v.len() > declaration.frozen_contexts_per_role)
            || observation.known_seed_completions > receipt.physically_completed_role_calls
            || observation.accepted_seeds > observation.known_seed_completions
            || observation.revoked_seeds > observation.accepted_seeds
        {
            return Err(RoleError::InvalidOutput);
        }
        if require_complete
            && (!receipt.physical_shutdown_confirmed
                || receipt.physical_runs_in_flight != 0
                || receipt.quarantined
                || observation.quarantined
                || (observation.admission_closed
                    && !terminal.is_some_and(|scope| {
                        scope.complete
                            && scope.bank_closed
                            && scope.backend_dropped
                            && scope.worker_joined
                            && scope.buffers_released
                            && scope.admission_closed
                            && scope.leases_active == 0
                            && scope.leases_quarantined == 0
                            && scope.device_bytes_current == 0
                            && scope.device_bytes_retained == 0
                            && !declaration.native_cpu_loaded_capability
                            && observation.entries_per_role == [0; 3]
                    }))
                || observation.active_lease.is_some()
                || observation.pinned_entries != 0
                || observation.pending_acceptance
                || observation
                    .accepted_seeds
                    .saturating_sub(observation.revoked_seeds)
                    > receipt.search_consumed_role_inputs
                || observation.last_revocation_failure.is_some()
                || observation.retained_prepared_input
                || observation.retained_full_payload_bytes != 0)
        {
            return Err(RoleError::InvalidOutput);
        }
        Ok(())
    }
    impl NativeRoleFinishHandle {
        #[cfg(feature = "experimental-io-binding")]
        pub fn cuda_warm_followup_observation(
            &self,
        ) -> Option<crate::pals_attestation::followup::NativeCudaWarmObservationV4> {
            use crate::pals_attestation::followup as wire;
            let warm = self.owner.private_warm.as_ref()?;
            let handle = warm.observer.as_ref()?;
            let bank = warm.state.followup_observation()?;
            let observed = handle.snapshot();
            let latest = observed.latest?;
            let joined = self.owner.shutdown.load(Ordering::Acquire);
            let snapshot = warm.state.snapshot().ok();
            let bank_empty = snapshot.as_ref().is_some_and(|s| {
                s.entries_per_role == [0; 3]
                    && s.active_lease.is_none()
                    && s.pinned_entries == 0
                    && !s.pending_acceptance
                    && !s.retained_prepared_input
                    && !s.quarantined
            });
            let complete = observed.status
                == rz_eval::pals_onnx::PalsCudaWarmObservationStatus::Available
                && observed.attempted_event_sequence == latest.event_sequence
                && latest.accounting_complete
                && bank.complete;
            let cap = handle.capability();
            Some(wire::NativeCudaWarmObservationV4 {
                measurement_contract: "rz-pals-cuda-warm-owner-accounting-v4/1".into(),
                backend_owner_id: latest.backend_owner_id,
                bank_owner_id: bank.owner_id,
                physical_event_sequence: latest.event_sequence,
                seed_event_sequence: bank.event_sequence,
                capability_identity: wire::hex(&cap.capability_token()),
                device_identity: format!("cuda:{}", cap.device_id()),
                capability_available: self
                    .owner
                    .startup_ready
                    .load(Ordering::Acquire)
                    .then_some(true),
                max_leases: latest.limits.max_leases,
                device_bytes_max: latest.limits.device_bytes_max,
                leases_admitted: latest.leases_admitted,
                leases_physically_completed: latest.leases_physically_completed,
                leases_quarantined: latest.leases_quarantined,
                leases_active: latest.leases_active,
                leases_peak: latest.leases_peak,
                startup_leases_admitted: latest.startup_leases_admitted,
                startup_leases_physically_completed: latest.startup_leases_physically_completed,
                startup_leases_quarantined: latest.startup_leases_quarantined,
                device_bytes_current: latest.device_bytes_live,
                device_bytes_peak: latest.device_bytes_peak,
                device_bytes_retained: latest.device_bytes_retained,
                accepted_seed_consumptions: bank.accepted_seed_consumptions,
                seed_context_checked_consumptions: bank.accepted_seed_context_checked,
                rejected_seed_contexts: bank.rejected_seed_contexts,
                fresh_value_evaluations: bank.fresh_value_evaluations,
                value_always_fresh: bank.value_always_fresh,
                admission_closed: latest.admission_closed,
                bank_closed: bank.closed,
                backend_dropped: latest.backend_dropped,
                worker_joined: joined,
                buffers_released: joined && latest.buffers_released && bank.closed && bank_empty,
                complete,
            })
        }
        #[cfg(not(feature = "experimental-io-binding"))]
        pub fn cuda_warm_followup_observation(
            &self,
        ) -> Option<crate::pals_attestation::followup::NativeCudaWarmObservationV4> {
            None
        }
        pub fn receipt(&self) -> NativeRoleReceipt {
            let backend_stats = self
                .owner
                .final_stats
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let backend_stats_observation = backend_stats
                .as_ref()
                .map(|_| "exclusive_worker_before_shutdown");
            let private_warm = self
                .owner
                .private_warm
                .as_ref()
                .map(|warm| warm.state.snapshot());
            let cuda_record_page_observation = self
                .owner
                .cuda_record_page_snapshot
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let cuda_record_page_observation_scope = cuda_record_page_observation
                .as_ref()
                .map(|_| "exclusive_worker_before_shutdown");
            NativeRoleReceipt {
                physically_completed_role_calls: self
                    .owner
                    .physical_completed
                    .load(Ordering::Acquire),
                completed_role_inputs: self.owner.completed.load(Ordering::Acquire),
                failed_physical_role_calls: self.owner.physical_failed.load(Ordering::Acquire),
                invalid_role_outputs: self.owner.validation_failed.load(Ordering::Acquire),
                delivered_role_inputs: self.owner.delivered.load(Ordering::Acquire),
                search_consumed_role_inputs: self.owner.consumed.load(Ordering::Acquire),
                canceled_requests: self.owner.canceled.load(Ordering::Acquire),
                expired_requests: self.owner.expired.load(Ordering::Acquire),
                completed_new_game_resets: self.owner.new_game_resets.load(Ordering::Acquire),
                process_epoch: self.owner.process_epoch.0,
                game_generation: self.owner.game_generation.load(Ordering::Acquire),
                frozen_epoch: DEPLOYMENT_FROZEN_EPOCH,
                request_high_water: self.owner.request_high_water.load(Ordering::Acquire),
                execution_high_water: self.owner.execution_high_water.load(Ordering::Acquire),
                physical_runs_in_flight: self.owner.in_flight.load(Ordering::Acquire),
                quarantined: self.owner.quarantined.load(Ordering::Acquire)
                    || self
                        .owner
                        .private_warm
                        .as_ref()
                        .is_some_and(|w| w.state.is_quarantined()),
                physical_shutdown_confirmed: self.owner.shutdown.load(Ordering::Acquire),
                native_buffers_released: self.owner.shutdown.load(Ordering::Acquire)
                    && !self.owner.quarantined.load(Ordering::Acquire)
                    && self.owner.in_flight.load(Ordering::Acquire) == 0,
                model_epoch: self.owner.model_epoch,
                export_manifest_sha256: self.owner.manifest_digest,
                encoding_semantic_sha256: self.owner.encoding_semantic_digest,
                adapter_source_sha256: self.owner.adapter_source_digest,
                trained: self.owner.trained,
                residency: self.owner.residency.clone(),
                last_failure: self
                    .owner
                    .last_failure
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone(),
                backend_stats,
                backend_stats_observation,
                host_record_page_observation: self
                    .owner
                    .host_record_page_observer
                    .as_ref()
                    .map(|observer| observer.snapshot().into()),
                private_warm_observation: private_warm
                    .as_ref()
                    .and_then(|r| r.as_ref().ok())
                    .cloned(),
                private_warm_observation_unavailable: private_warm
                    .as_ref()
                    .and_then(|r| r.as_ref().err())
                    .map(|_| true),
                cuda_record_page_observation,
                cuda_record_page_observation_error: *self
                    .owner
                    .cuda_record_page_snapshot_error
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()),
                execution: self.owner.execution_receipt(),
                cuda_record_page_observation_scope,
                startup_probe: self
                    .owner
                    .startup_probe
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone(),
                final_runtime_mapping_confirmed: self
                    .owner
                    .final_mapping_confirmed
                    .load(Ordering::Acquire)
                    .then_some(true),
                final_runtime_loading_mapping: self
                    .owner
                    .final_loading_mapping
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone(),
                observer_failures: self.owner.observer_failures.load(Ordering::Acquire),
                last_observer_failure: self
                    .owner
                    .last_observer_failure
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone(),
            }
        }
        fn collect_final_stats(&self, until: Instant) -> Result<(), RoleError> {
            loop {
                if self.owner.quarantined.load(Ordering::Acquire) {
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                if let Some(error) = self
                    .owner
                    .game_reset_failure
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
                {
                    return Err(error);
                }
                if self.owner.pending_game_target.load(Ordering::Acquire) == 0
                    && self
                        .owner
                        .final_stats
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .is_some()
                    && (self.owner.execution.cuda_loading_profile.is_none()
                        || self
                            .owner
                            .final_loading_mapping
                            .lock()
                            .map_err(|_| RoleError::PhysicalCompletionUnknown)?
                            .is_some())
                    && (self.owner.execution.cuda_record_pages.is_none()
                        || self
                            .owner
                            .cuda_record_page_snapshot
                            .lock()
                            .map_err(|_| RoleError::PhysicalCompletionUnknown)?
                            .is_some())
                {
                    return Ok(());
                }
                let mut slot = self
                    .owner
                    .stats_lease
                    .lock()
                    .map_err(|_| RoleError::PhysicalCompletionUnknown)?;
                if slot.is_none() && self.owner.in_flight.load(Ordering::Acquire) == 0 {
                    let mut worker = self
                        .owner
                        .worker
                        .lock()
                        .map_err(|_| RoleError::PhysicalCompletionUnknown)?;
                    // The other owner may have consumed this shared cleanup
                    // window. Expiry never admits a new NN-zero control. The
                    // worker is known idle here; finish may still attempt its
                    // shutdown without converting this into unknown Run work.
                    if Instant::now() >= until {
                        return Err(RoleError::Deadline);
                    }
                    let pending_target = self.owner.pending_game_target.load(Ordering::Acquire);
                    let lease = worker
                        .submit(if pending_target != 0 {
                            PalsNativeCommand::ResetTo(pending_target)
                        } else if self.owner.execution.provider == "cuda"
                            && !self.owner.final_mapping_confirmed.load(Ordering::Acquire)
                        {
                            PalsNativeCommand::VerifyRuntime
                        } else if self.owner.execution.cuda_loading_profile.is_some()
                            && self
                                .owner
                                .final_loading_mapping
                                .lock()
                                .map_err(|_| RoleError::PhysicalCompletionUnknown)?
                                .is_none()
                        {
                            PalsNativeCommand::ObserveRuntimeMappings
                        } else {
                            #[cfg(feature = "experimental-io-binding")]
                            {
                                if self.owner.execution.cuda_record_pages.is_some()
                                    && self
                                        .owner
                                        .cuda_record_page_snapshot
                                        .lock()
                                        .map_err(|_| RoleError::PhysicalCompletionUnknown)?
                                        .is_none()
                                {
                                    PalsNativeCommand::SnapshotCudaRecordPages
                                } else {
                                    PalsNativeCommand::SnapshotStats
                                }
                            }
                            #[cfg(not(feature = "experimental-io-binding"))]
                            {
                                PalsNativeCommand::SnapshotStats
                            }
                        })
                        .map_err(|error| {
                            self.owner.remember_failure(&error);
                            let error = model_error(error);
                            if pending_target != 0 {
                                self.owner.remember_game_reset_failure(error)
                            } else {
                                error
                            }
                        })?;
                    self.owner.in_flight.fetch_add(1, Ordering::AcqRel);
                    *slot = Some(lease);
                }
                if let Some(lease) = slot.as_mut() {
                    let expected_verify = matches!(lease.input(), PalsNativeCommand::VerifyRuntime);
                    let expected_mapping =
                        matches!(lease.input(), PalsNativeCommand::ObserveRuntimeMappings);
                    let expected_stats = matches!(lease.input(), PalsNativeCommand::SnapshotStats);
                    let expected_reset = match lease.input() {
                        PalsNativeCommand::ResetTo(target) => Some(*target),
                        _ => None,
                    };
                    #[cfg(feature = "experimental-io-binding")]
                    let expected_pages =
                        matches!(lease.input(), PalsNativeCommand::SnapshotCudaRecordPages);
                    match lease.poll() {
                        PhysicalPoll::Ready(result) => {
                            self.owner.in_flight.fetch_sub(1, Ordering::AcqRel);
                            *slot = None;
                            match result {
                                Ok(PalsNativeResult::ResetTo { game_generation })
                                    if expected_reset == Some(game_generation) =>
                                {
                                    self.owner.accept_game_target(game_generation).map_err(
                                        |error| self.owner.remember_game_reset_failure(error),
                                    )?;
                                }
                                Ok(PalsNativeResult::RuntimeVerified)
                                    if self.owner.execution.provider == "cuda"
                                        && expected_verify =>
                                {
                                    self.owner
                                        .final_mapping_confirmed
                                        .store(true, Ordering::Release);
                                }
                                Ok(PalsNativeResult::RuntimeMappingsObserved(witness))
                                    if expected_mapping =>
                                {
                                    validate_runtime_loading_mapping(
                                        &witness,
                                        &self.owner.execution,
                                    )
                                    .map_err(|error| {
                                        self.owner.remember_failure(&error);
                                        model_error(error)
                                    })?;
                                    *self
                                        .owner
                                        .final_loading_mapping
                                        .lock()
                                        .map_err(|_| RoleError::PhysicalCompletionUnknown)? =
                                        Some(witness);
                                }
                                Ok(PalsNativeResult::Stats(stats)) if expected_stats => {
                                    stats.validate().map_err(|error| {
                                        self.owner.remember_failure(&error);
                                        model_error(error)
                                    })?;
                                    *self
                                        .owner
                                        .final_stats
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner()) = Some(stats.into());
                                    if self.owner.execution.cuda_record_pages.is_none()
                                        && self.owner.pending_game_target.load(Ordering::Acquire)
                                            == 0
                                    {
                                        return Ok(());
                                    }
                                }
                                #[cfg(feature = "experimental-io-binding")]
                                Ok(PalsNativeResult::CudaRecordPagesObserved(snapshot))
                                    if expected_pages =>
                                {
                                    let selection = self
                                        .owner
                                        .execution
                                        .cuda_record_pages
                                        .as_ref()
                                        .ok_or(RoleError::InvalidOutput)?;
                                    let snapshot = checked_cuda_record_page_observation(
                                        (*snapshot).into(),
                                        selection,
                                    )?;
                                    if snapshot.game_generation
                                        != self.owner.game_generation.load(Ordering::Acquire)
                                    {
                                        return Err(RoleError::InvalidOutput);
                                    }
                                    *self
                                        .owner
                                        .cuda_record_page_snapshot
                                        .lock()
                                        .map_err(|_| RoleError::Unavailable)? = Some(snapshot);
                                    *self
                                        .owner
                                        .cuda_record_page_snapshot_error
                                        .lock()
                                        .map_err(|_| RoleError::Unavailable)? = None;
                                }
                                Ok(_) => {
                                    let error = RoleError::Backend(
                                        "PALS Stats lease received a different control response"
                                            .into(),
                                    );
                                    return Err(if expected_reset.is_some() {
                                        self.owner.remember_game_reset_failure(error)
                                    } else {
                                        error
                                    });
                                }
                                Err(error) => {
                                    self.owner.remember_failure(&error);
                                    let error = model_error(error);
                                    return Err(if expected_reset.is_some() {
                                        self.owner.remember_game_reset_failure(error)
                                    } else {
                                        error
                                    });
                                }
                            }
                        }
                        PhysicalPoll::Quarantined | PhysicalPoll::Consumed => {
                            if let Ok(Some(error)) = lease.quarantine_cause() {
                                self.owner.remember_failure(&error);
                            }
                            self.owner.quarantined.store(true, Ordering::Release);
                            return Err(RoleError::PhysicalCompletionUnknown);
                        }
                        PhysicalPoll::Pending => {}
                    }
                }
                if Instant::now() >= until {
                    if slot.is_some() || self.owner.in_flight.load(Ordering::Acquire) != 0 {
                        // An already admitted control/role remains physically
                        // pending. Keep its owners and original unknown state.
                        self.owner.quarantined.store(true, Ordering::Release);
                        return Err(RoleError::PhysicalCompletionUnknown);
                    }
                    // A completed Verify/Mapping control is a known fence,
                    // but expiry does not permit the next Stats admission.
                    return Err(RoleError::Deadline);
                }
                drop(slot);
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        pub fn finish(&self, until: Instant) -> Result<NativeRoleReceipt, RoleError> {
            {
                let _guard = self
                    .owner
                    .game_lifecycle
                    .lock()
                    .map_err(|_| RoleError::PhysicalCompletionUnknown)?;
                self.owner.finishing.store(true, Ordering::Release);
            }
            if self
                .owner
                .private_warm
                .as_ref()
                .is_some_and(|warm| warm.state.is_quarantined())
            {
                self.owner.quarantined.store(true, Ordering::Release);
            }
            if self.owner.quarantined.load(Ordering::Acquire) {
                if self.owner.execution.cuda_record_pages.is_some()
                    && self
                        .owner
                        .cuda_record_page_snapshot
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .is_none()
                {
                    *self
                        .owner
                        .cuda_record_page_snapshot_error
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) =
                        Some(NativeStartupErrorKind::PhysicalCompletionUnknown);
                }
                return Err(RoleError::PhysicalCompletionUnknown);
            }
            if self.owner.shutdown.load(Ordering::Acquire) {
                if let Some(error) = self
                    .owner
                    .game_reset_failure
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
                {
                    return Err(error);
                }
                let receipt = self.receipt();
                return if receipt.backend_stats.is_some()
                    && self.owner.pending_game_target.load(Ordering::Acquire) == 0
                {
                    Ok(receipt)
                } else {
                    Err(RoleError::Backend(
                        "PALS shutdown completed without an accepted final backend snapshot".into(),
                    ))
                };
            }
            let stats_failure = match self.collect_final_stats(until) {
                Ok(()) => None,
                Err(RoleError::PhysicalCompletionUnknown) => {
                    if self.owner.execution.cuda_record_pages.is_some() {
                        *self
                            .owner
                            .cuda_record_page_snapshot_error
                            .lock()
                            .unwrap_or_else(|e| e.into_inner()) =
                            Some(NativeStartupErrorKind::PhysicalCompletionUnknown);
                    }
                    self.owner.quarantined.store(true, Ordering::Release);
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                // A failed but physically completed Stats control does not
                // authorize metrics. Still join/drop the idle native owner in
                // the same cleanup window and preserve the original failure.
                Err(error) => {
                    if self.owner.execution.cuda_record_pages.is_some()
                        && self
                            .owner
                            .cuda_record_page_snapshot
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .is_none()
                    {
                        *self
                            .owner
                            .cuda_record_page_snapshot_error
                            .lock()
                            .unwrap_or_else(|e| e.into_inner()) =
                            Some(NativeStartupErrorKind::from(&error));
                    }
                    Some(error)
                }
            };
            loop {
                if self.owner.quarantined.load(Ordering::Acquire) {
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                let outcome = self
                    .owner
                    .worker
                    .lock()
                    .map_err(|_| RoleError::Backend("PALS worker shutdown lock poisoned".into()))?
                    .try_shutdown();
                match outcome {
                    Poll::Ready(Ok(())) => {
                        if self.owner.in_flight.load(Ordering::Acquire) != 0 {
                            self.owner.quarantined.store(true, Ordering::Release);
                            return Err(RoleError::PhysicalCompletionUnknown);
                        }
                        self.owner.shutdown.store(true, Ordering::Release);
                        #[cfg(feature = "experimental-io-binding")]
                        if let Some(warm) = self
                            .owner
                            .private_warm
                            .as_ref()
                            .filter(|w| w.observer.is_some())
                        {
                            // Real worker join implies backend Drop has finished.
                            // Close the accepted host bank in its current game only.
                            warm.state.close_after_known_fence()?;
                        }
                        if stats_failure.is_none()
                            && self.owner.pending_game_target.load(Ordering::Acquire) != 0
                        {
                            return Err(RoleError::Backend(
                                "PALS shutdown has an unacknowledged game target".into(),
                            ));
                        }
                        return stats_failure.map_or_else(|| Ok(self.receipt()), Err);
                    }
                    Poll::Ready(Err(error)) => {
                        self.owner.remember_failure(&error);
                        self.owner.quarantined.store(true, Ordering::Release);
                        return Err(RoleError::PhysicalCompletionUnknown);
                    }
                    Poll::Pending if Instant::now() >= until => {
                        self.owner.quarantined.store(true, Ordering::Release);
                        return Err(RoleError::PhysicalCompletionUnknown);
                    }
                    Poll::Pending => std::thread::sleep(Duration::from_millis(1)),
                }
            }
        }
    }
    struct WorkerBackend {
        owner: Arc<WorkerOwner>,
    }
    struct Lease {
        physical: NativePhysicalLease,
        request: Arc<RuntimeRoleRequest<PalsModelInput>>,
        execution: ExecutionId,
        warm: Option<WarmAdmission>,
        // A post-submit observer error cannot return dispatch Err and strand
        // this Lease. Runtime polling must retain it through actual completion.
        observer_failed: bool,
        #[cfg(feature = "experimental-io-binding")]
        warm_expected: Option<CudaWarmExpected>,
    }
    #[cfg(feature = "experimental-io-binding")]
    #[derive(Clone, Copy)]
    struct CudaWarmExpected {
        binding: rz_eval::pals_onnx::PalsCudaWarmExecutionBinding,
        ordinal: Option<u64>,
        input_key: [u8; 32],
        invocation: rz_eval::pals_private::PrivateInvocation,
    }
    impl Backend<PalsAdapter<PalsModelInput, ContractSystemClock>> for WorkerBackend {
        type Lease = Lease;
        fn additional_resources(&self, _: &[Arc<RuntimeRoleRequest<PalsModelInput>>]) -> Resources {
            Resources {
                host_bytes: self.owner.additional_host_bytes,
                device_bytes: self.owner.execution.execution_device_reservation(),
                pinned_bytes: 0,
            }
        }
        fn dispatch(
            &mut self,
            execution: &ExecutionId,
            requests: &[Arc<RuntimeRoleRequest<PalsModelInput>>],
        ) -> Result<Self::Lease, ContractError> {
            if requests.len() != 1
                || self.owner.quarantined.load(Ordering::Acquire)
                || self.owner.finishing.load(Ordering::Acquire)
                || self.owner.shutdown.load(Ordering::Acquire)
                || self
                    .owner
                    .role_execution
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .is_some_and(|o| !o.ready_observed)
            {
                return Err(fault(
                    ErrorCode::BackendUnavailable,
                    "PALS native worker is unavailable or batch is unsupported",
                ));
            }
            let request = Arc::clone(&requests[0]);
            // Warm has two full preflight payloads, reserved before seed
            // admission. One enters the worker and one survives unknown work.
            let mut warm = if self.owner.private_warm.is_some() {
                let mut slot = self.owner.warm_admission.lock().map_err(|_| {
                    fault(
                        ErrorCode::BackendFailure,
                        "PALS warm handoff owner poisoned",
                    )
                })?;
                if !slot.as_ref().is_some_and(|a| {
                    a.physical
                        .as_ref()
                        .is_some_and(|p| p.prepared.id == request.role().id)
                }) {
                    return Err(fault(
                        ErrorCode::Stale,
                        "PALS warm handoff request mismatch",
                    ));
                }
                slot.take()
            } else {
                None
            };
            #[cfg(feature = "experimental-io-binding")]
            let warm_expected = if let (Some(admission), Some(observer)) = (
                warm.as_mut(),
                self.owner
                    .private_warm
                    .as_ref()
                    .and_then(|w| w.observer.as_ref()),
            ) {
                let physical = admission.physical.as_ref().ok_or_else(|| {
                    fault(ErrorCode::BackendFailure, "missing Warm physical owner")
                })?;
                let value = physical.prepared.logical.purpose == RoleQueryPurpose::ValueFresh;
                let binding = rz_eval::pals_onnx::PalsCudaWarmExecutionBinding {
                    backend_owner_id: observer.owner_id(),
                    purpose: if value {
                        rz_eval::pals_onnx::PalsCudaWarmInvocationPurpose::ValueFresh
                    } else {
                        rz_eval::pals_onnx::PalsCudaWarmInvocationPurpose::Policy
                    },
                    request: Some(request.role().id),
                    execution: Some(*execution),
                    bank_lease_id: physical.seed().map(|seed| seed.id()),
                };
                let snapshot = observer.snapshot();
                let ordinal = (snapshot.status
                    == rz_eval::pals_onnx::PalsCudaWarmObservationStatus::Available)
                    .then(|| {
                        snapshot
                            .latest
                            .as_ref()
                            .and_then(|s| s.leases_admitted.checked_add(1))
                    })
                    .flatten();
                let expected = CudaWarmExpected {
                    binding,
                    ordinal,
                    input_key: physical
                        .prepared
                        .input
                        .canonical_input_key(&self.owner.model_config)
                        .map_err(|_| fault(ErrorCode::BackendFailure, "invalid Warm input key"))?,
                    invocation: physical
                        .invocation()
                        .map_err(|_| fault(ErrorCode::BackendFailure, "invalid Warm invocation"))?,
                };
                let Some(NativeWarmPayload::Cuda(payload)) = admission.payload.as_mut() else {
                    return Err(fault(
                        ErrorCode::UnsupportedContract,
                        "CUDA observation requires actual CUDA payload",
                    ));
                };
                payload
                    .bind_execution_observation(observer, binding)
                    .map_err(|_| {
                        // An observation bind failure is incomplete evidence;
                        // it does not prove that a bank seed identity failed.
                        admission.state.observe_join_incomplete();
                        fault(
                            ErrorCode::BackendFailure,
                            "CUDA Warm execution binding failed",
                        )
                    })?;
                Some(expected)
            } else {
                None
            };
            let command = if let Some(admission) = &mut warm {
                admission
                    .payload
                    .take()
                    .ok_or_else(|| fault(ErrorCode::BackendFailure, "PALS warm payload missing"))?
                    .command()
            } else if self.owner.followup_evidence.load(Ordering::Acquire) {
                PalsNativeCommand::EvaluateWithEvidence(request.role().state.as_ref().clone())
            } else {
                // Default Fresh ownership/serialization is unchanged.
                PalsNativeCommand::Evaluate(request.role().state.as_ref().clone())
            };
            let physical = self
                .owner
                .worker
                .lock()
                .map_err(|_| {
                    fault(
                        ErrorCode::BackendFailure,
                        "PALS physical owner lock poisoned",
                    )
                })?
                .submit(command)
                .map_err(|_| {
                    fault(
                        ErrorCode::BackendFailure,
                        "PALS physical worker rejected native admission",
                    )
                })?;
            self.owner.in_flight.fetch_add(1, Ordering::AcqRel);
            self.owner
                .execution_high_water
                .store(execution.sequence, Ordering::Release);
            let mut lease = Lease {
                physical,
                request,
                execution: *execution,
                warm,
                observer_failed: false,
                #[cfg(feature = "experimental-io-binding")]
                warm_expected,
            };
            let binding = NativeRoleExecutionBinding {
                request: lease.request.role().id,
                execution: lease.execution,
            };
            let prepared_perspective =
                self.owner
                    .prepared_perspective
                    .lock()
                    .ok()
                    .and_then(|slot| {
                        slot.filter(|(id, _)| *id == binding.request)
                            .map(|(_, color)| color)
                    });
            let fresh = lease.warm.as_ref().map_or(Some(true), |warm| {
                warm.physical
                    .as_ref()
                    .and_then(|physical| physical.invocation().ok())
                    .map(|invocation| {
                        matches!(
                            invocation,
                            rz_eval::pals_private::PrivateInvocation::Fresh { .. }
                        )
                    })
            });
            *self
                .owner
                .role_execution
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Some(ObservedRoleExecution {
                binding,
                unknown_observed: false,
                ready_observed: false,
                observer_error: None,
                prepared_state: lease.request.role().state_identity,
                prepared_perspective,
                input_sha256: lease.request.role().key.input.0,
                fresh,
                completed_nn_inputs: None,
                measurement_complete: false,
                owner_accounting: None,
            });
            if let Err(error) = self
                .owner
                .observe("role_dispatched", |observer| observer.dispatched(binding))
            {
                lease.observer_failed = true;
                self.owner.remember_role_observer_error(binding, error);
            }
            // Even callback poison/panic/failure leaves the actual physical
            // Lease in runtime ownership. No observer-only quarantine is made.
            Ok(lease)
        }
        fn poll(
            &mut self,
            lease: &mut Lease,
        ) -> Poll<Vec<BackendResult<PalsAdapter<PalsModelInput, ContractSystemClock>>>> {
            let binding = NativeRoleExecutionBinding {
                request: lease.request.role().id,
                execution: lease.execution,
            };
            let result = match lease.physical.poll() {
                PhysicalPoll::Pending => return Poll::Pending,
                unknown @ (PhysicalPoll::Quarantined | PhysicalPoll::Consumed) => {
                    if let Ok(Some(error)) = lease.physical.quarantine_cause() {
                        // Preserve the backend's original cause before observers.
                        self.owner.remember_failure(&error);
                    }
                    self.owner.quarantined.store(true, Ordering::Release);
                    let reason = if matches!(unknown, PhysicalPoll::Quarantined) {
                        NativeRoleCompletionUnknown::Quarantined
                    } else {
                        NativeRoleCompletionUnknown::Consumed
                    };
                    if let Err(error) = self.owner.observe_role_terminal(
                        binding,
                        NativeRoleTerminal::CompletionUnknown(reason),
                    ) {
                        lease.observer_failed = true;
                        self.owner.remember_role_observer_error(binding, error);
                    }
                    if let Some(mut warm) = lease.warm.take() {
                        if let Some(physical) = warm.physical.take() {
                            warm.state
                                .retain_unknown(physical, "physical_worker_unknown");
                        }
                    }
                    return Poll::Pending;
                }
                PhysicalPoll::Ready(result) => result,
            };
            self.owner.in_flight.fetch_sub(1, Ordering::AcqRel);
            self.owner.physical_completed.fetch_add(1, Ordering::AcqRel);
            #[cfg(feature = "experimental-io-binding")]
            if let Some(expected) = lease.warm_expected {
                let joined=self.owner.private_warm.as_ref().and_then(|w|w.observer.as_ref()).and_then(|handle| {
                    let snapshot=handle.snapshot();
                    if snapshot.status!=rz_eval::pals_onnx::PalsCudaWarmObservationStatus::Available {return None;}
                    let latest=snapshot.latest?;
                    if latest.event_sequence!=snapshot.attempted_event_sequence || !latest.accounting_complete {return None;}
                    latest.last_invocation.filter(|actual|actual.binding==expected.binding
                        && Some(actual.ordinal)==expected.ordinal && actual.input_key==expected.input_key
                        && actual.invocation==expected.invocation
                        && actual.state==rz_eval::pals_onnx::PalsCudaWarmInvocationState::CompletedKnown)
                });
                if let Some(warm) = lease.warm.as_ref() {
                    warm.state.observe_ready_join(
                        expected.invocation,
                        expected.binding.purpose
                            == rz_eval::pals_onnx::PalsCudaWarmInvocationPurpose::ValueFresh,
                        joined
                            .as_ref()
                            .is_some_and(|actual| actual.private_run_completed),
                        joined.is_some(),
                    );
                }
                if let Ok(mut slot) = self.owner.role_execution.lock()
                    && let Some(observed) = slot.as_mut().filter(|o| o.binding == binding)
                {
                    observed.completed_nn_inputs =
                        joined.and_then(|actual| actual.actual_completed_nn_inputs_delta);
                    observed.measurement_complete =
                        joined.is_some_and(|actual| actual.nn_evidence_complete);
                }
            }
            if let Err(error) = &result {
                self.owner.remember_failure(error);
            }
            if let Ok(PalsNativeResult::EvaluationWithEvidence {
                outcome,
                actual_completed_nn_inputs_delta,
                complete,
                owner_accounting,
            }) = &result
            {
                if let Err(error) = outcome {
                    self.owner.remember_failure(error);
                }
                if let Ok(mut observed) = self.owner.role_execution.lock()
                    && let Some(observed) = observed.as_mut().filter(|o| o.binding == binding)
                {
                    observed.completed_nn_inputs = *actual_completed_nn_inputs_delta;
                    observed.measurement_complete = *complete;
                    observed.owner_accounting = *owner_accounting;
                }
            }
            if let Err(error) = self
                .owner
                .observe_role_terminal(binding, NativeRoleTerminal::Ready(result.as_ref()))
            {
                lease.observer_failed = true;
                self.owner.remember_role_observer_error(binding, error);
            }
            let legacy_observation = match &result {
                Ok(PalsNativeResult::Evaluation(raw)) => {
                    self.owner.observe("physical_return", |observer| {
                        observer.physically_completed_with_execution(
                            lease.request.role().id,
                            Ok(raw),
                            &self.owner.execution_receipt(),
                        )
                    })
                }
                Ok(PalsNativeResult::EvaluationWithEvidence { outcome, .. }) => {
                    self.owner.observe("physical_return", |observer| {
                        observer.physically_completed_with_execution(
                            lease.request.role().id,
                            outcome.as_ref(),
                            &self.owner.execution_receipt(),
                        )
                    })
                }
                Err(error) => self.owner.observe("physical_return", |observer| {
                    observer.physically_completed_with_execution(
                        lease.request.role().id,
                        Err(error),
                        &self.owner.execution_receipt(),
                    )
                }),
                _ => Ok(()),
            };
            if let Err(error) = legacy_observation {
                lease.observer_failed = true;
                self.owner.remember_role_observer_error(binding, error);
            }
            let raw = match result {
                Ok(PalsNativeResult::Evaluation(raw)) => {
                    self.owner.completed.fetch_add(1, Ordering::AcqRel);
                    Ok(raw)
                }
                Ok(PalsNativeResult::EvaluationWithEvidence { outcome, .. }) => match outcome {
                    Ok(raw) => {
                        self.owner.completed.fetch_add(1, Ordering::AcqRel);
                        Ok(raw)
                    }
                    Err(_) => {
                        self.owner.physical_failed.fetch_add(1, Ordering::AcqRel);
                        Err(fault(
                            ErrorCode::BackendFailure,
                            "PALS native role execution failed after physical completion",
                        ))
                    }
                },
                Ok(_) => {
                    self.owner.validation_failed.fetch_add(1, Ordering::AcqRel);
                    Err(fault(
                        ErrorCode::UnsupportedContract,
                        "PALS role lease received a control response",
                    ))
                }
                Err(_) => {
                    self.owner.physical_failed.fetch_add(1, Ordering::AcqRel);
                    Err(fault(
                        ErrorCode::BackendFailure,
                        "PALS native role execution failed after physical completion",
                    ))
                }
            };
            let raw_was_successful = raw.is_ok();
            let output = raw.and_then(|raw| {
                let request = lease.request.role();
                let decoded = raw
                    .decode(&request.state, &self.owner.model_config)
                    .map_err(|_| {
                        fault(
                            ErrorCode::NumericalFailure,
                            "PALS native role head validation failed",
                        )
                    })?;
                let wdl = Wdl::try_new(decoded.wdl[0], decoded.wdl[1], decoded.wdl[2], 1e-4)?;
                let payload = match request.role {
                    Role::Proposer => RolePayload::Proposal {
                        candidate_logits: raw.candidate_logits,
                        wdl,
                    },
                    Role::Critic => RolePayload::Counterexample {
                        candidate_logits: raw.candidate_logits,
                        divergence_logits: raw.divergence_logits.ok_or_else(|| {
                            fault(
                                ErrorCode::NumericalFailure,
                                "PALS C divergence head missing",
                            )
                        })?,
                        wdl,
                    },
                    Role::Validator => {
                        return Err(fault(
                            ErrorCode::UnsupportedContract,
                            "V is absent from PALS product export",
                        ));
                    }
                };
                Ok(PhysicalRoleOutput {
                    execution: lease.execution,
                    output: RoleOutput {
                        id: request.id,
                        authority: request.authority,
                        key: request.key,
                        payload,
                        private_latent: raw.private_latent,
                    },
                })
            });
            if raw_was_successful && output.is_err() {
                self.owner.validation_failed.fetch_add(1, Ordering::AcqRel);
            }
            let observer_failed = lease.observer_failed
                || self
                    .owner
                    .role_execution
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .is_some_and(|o| o.binding == binding && o.observer_error.is_some());
            let mut output = if observer_failed && output.is_ok() {
                Err(fault(
                    ErrorCode::BackendFailure,
                    "PALS physical result observer failed after confirmed native completion",
                ))
            } else {
                output
            };
            if let Some(mut warm) = lease.warm.take() {
                if let Some(physical) = warm.physical.take() {
                    // Only this actual Ready path may mark the bank complete.
                    let latent = output
                        .as_ref()
                        .ok()
                        .map(|o| o.output.private_latent.as_slice());
                    if let Err(error) = warm.state.completed(physical, latent) {
                        output = Err(fault(
                            match error {
                                RoleError::Canceled => ErrorCode::Canceled,
                                RoleError::Deadline => ErrorCode::Expired,
                                _ => ErrorCode::BackendFailure,
                            },
                            "PALS warm completed output could not be staged",
                        ));
                    }
                }
            }
            Poll::Ready(vec![BackendResult {
                request_id: lease.request.role().id,
                output,
            }])
        }
    }

    pub struct NativeRoleModel {
        identity: String,
        value_identity: ModelValueIdentity,
        epoch: ProcessEpoch,
        model_epoch: [u8; 32],
        model: Digest,
        encoding: Digest,
        clock: ContractSystemClock,
        scope: SharedPalsScope,
        runtime: Runtime,
        owner: Arc<WorkerOwner>,
        sequence: u64,
        game: u64,
        unusable: bool,
        drain_limit: Duration,
        pending_new_game: bool,
        control_lease: Option<NativePhysicalLease>,
        startup_attempted: bool,
        startup_probe_configured: bool,
        delivered_request: Option<RequestId>,
        last_value_execution: Option<RoleValueExecutionEvidence>,
        selected_review: Option<Arc<rz_search::pals::review_context::ReviewView>>,
        ticketed_delivered: Vec<rz_contracts::pals_adaptive::RoleTicketEnvelope>,
        startup_loading_elapsed_ns: Option<u64>,
        startup_clock: Option<Instant>,
        startup_diagnostic: Option<NativeStartupFailureDiagnostic>,
        cpu_invocation_budget: Option<NativeInvocationBudget>,
        cpu_invocation_load_interval: Option<(Instant, Instant)>,
    }
    impl NativeRoleModel {
        pub(super) fn ticketed_clock(&self) -> ContractSystemClock {
            self.clock.clone()
        }
        pub(super) fn ticketed_config(&self) -> PalsModelConfig {
            self.owner.model_config.clone()
        }
        pub(super) fn ticketed_model_epoch(&self) -> [u8; 32] {
            self.model_epoch
        }
        pub(super) fn ticketed_require_fresh(&self) -> Result<(), RoleError> {
            if self.owner.private_warm.is_some()
                || self.sequence != 0
                || self.delivered_request.is_some()
            {
                return Err(RoleError::Backend("UnsupportedCapability: ticketed native requires an unused Fresh owner; synchronous Warm is preserved separately".into()));
            }
            Ok(())
        }
        pub(super) fn ticketed_representation(
            &self,
            input: &PalsModelInput,
        ) -> Result<RepresentationKey, RoleError> {
            let packed = input
                .candidates
                .iter()
                .map(|candidate| candidate.packed().map_err(model_error))
                .collect::<Result<Vec<_>, _>>()?;
            let mut candidates = Sha256::new();
            for value in packed {
                candidates.update(value.to_le_bytes());
            }
            let mut masks = Sha256::new();
            for value in [
                input.records.len(),
                input.candidates.len(),
                input.divergence_features.len(),
            ] {
                masks.update((value as u64).to_le_bytes());
            }
            Ok(RepresentationKey {
                input: Digest(
                    input
                        .canonical_input_key(&self.owner.model_config)
                        .map_err(model_error)?,
                ),
                history: Digest(input.history_digest),
                candidates: Digest(candidates.finalize().into()),
                mask_positions: Digest(masks.finalize().into()),
                model: self.model,
                encoding: self.encoding,
                precision: PrecisionProfile::Fp32,
                frozen_epoch: DEPLOYMENT_FROZEN_EPOCH,
                record_revision: input.situation_revision,
                role: if input.role == PalsRole::Critic {
                    Role::Critic
                } else {
                    Role::Proposer
                },
            })
        }
        // Ticket authority, exact Rules position, prepared view and cancellation
        // remain independent inputs at the physical owner's admission boundary.
        #[allow(clippy::too_many_arguments)]
        pub(super) fn ticketed_run(
            &mut self,
            ticket: &rz_contracts::pals_adaptive::RoleTicketEnvelope,
            input: PalsModelInput,
            position: &Position,
            until: Instant,
            canceled: &AtomicBool,
            prepared_context: NativePreparedContext<'_>,
            logical: &RoleLogicalContext,
        ) -> Result<super::ticketed::NativeTicketOutput, RoleError> {
            if ticket.authority.game.0 > self.game {
                self.owner.request_game_target(ticket.authority.game.0)?;
                self.game = ticket.authority.game.0;
                self.pending_new_game = true;
            }
            let output = self.run_with_ticket(
                input,
                position,
                until,
                canceled,
                prepared_context,
                Some(logical),
                Some(ticket),
            )?;
            if self.delivered_request.take() != Some(ticket.id)
                || output.id != ticket.id
                || output.authority != ticket.authority
                || output.key != ticket.representation
            {
                self.unusable = true;
                return Err(RoleError::InvalidOutput);
            }
            // The previous request is now independent of all recent slots.
            // Later submissions cannot overwrite this output or its evidence.
            let mut slot = self
                .owner
                .role_execution
                .lock()
                .map_err(|_| RoleError::Unavailable)?;
            let observed = slot
                .as_ref()
                .filter(|entry| entry.binding.request == ticket.id);
            // 완료 불명 상태에서는 원본 관측과 lease 근거를 보존한다.
            // slot을 먼저 소비하면 실패 증거/격리에 필요한 소유자를 잃는다.
            if observed.is_some_and(|entry| !entry.ready_observed || entry.unknown_observed) {
                self.owner.quarantined.store(true, Ordering::Release);
                return Err(RoleError::PhysicalCompletionUnknown);
            }
            let evidence = observed.and_then(ObservedRoleExecution::ticket_execution_evidence);
            if observed.is_some() {
                slot.take();
            }
            drop(slot);
            self.ticketed_delivered.push(*ticket);
            Ok(super::ticketed::NativeTicketOutput {
                output,
                execution: evidence.and_then(|entry| entry.execution),
                completed_nn_inputs: evidence.and_then(|entry| entry.completed_nn_inputs),
                measurement_complete: evidence.is_some_and(|entry| entry.measurement_complete),
                physical_fresh: evidence.and_then(|entry| entry.physical_fresh),
                accounting: evidence.and_then(|entry| entry.accounting),
            })
        }
        pub(super) fn ticketed_acknowledge(
            &mut self,
            ticket: &rz_contracts::pals_adaptive::RoleTicketEnvelope,
            accepted: bool,
        ) -> Result<(), RoleError> {
            if accepted
                && (self
                    .scope
                    .get()
                    .is_none_or(|scope| scope.authority != ticket.authority)
                    || ticket
                        .original_deadline
                        .accepts(self.clock.domain(), self.clock.now())
                        .is_err()
                    || self.owner.quarantined.load(Ordering::Acquire))
            {
                return Err(RoleError::Canceled);
            }
            let Some(index) = self
                .ticketed_delivered
                .iter()
                .position(|entry| entry == ticket)
            else {
                return if accepted {
                    Err(RoleError::InvalidOutput)
                } else {
                    Ok(())
                };
            };
            if accepted {
                self.owner
                    .observe("accepted_ticket", |observer| observer.accepted(ticket.id))?;
                self.ticketed_delivered.remove(index);
                self.owner.consumed.fetch_add(1, Ordering::AcqRel);
                Ok(())
            } else {
                self.ticketed_delivered.remove(index);
                self.owner.observe("closed_ticket", |observer| {
                    observer.rejected(ticket.id, NativeRoleRejection::SearchFinishedUnconsumed)
                })
            }
        }
        pub(super) fn ticketed_close_pending(&mut self) {
            for ticket in std::mem::take(&mut self.ticketed_delivered) {
                let _ = self.owner.observe("closed_ticket", |observer| {
                    observer.rejected(ticket.id, NativeRoleRejection::Canceled)
                });
            }
        }
        /// Root fence preserves the loaded session and its physical worker.
        /// Unlike finish(), this never closes scheduler admission or joins the
        /// worker. Every reservation/control lease must already be reclaimed.
        pub(super) fn ticketed_quiesce(&mut self, until: Instant) -> bool {
            if !self.ticketed_delivered.is_empty()
                || self
                    .preflight_new_game(until, &AtomicBool::new(false))
                    .is_err()
                || self
                    .close_unconsumed(NativeRoleRejection::Canceled)
                    .is_err()
                || !self.owner.startup_ready.load(Ordering::Acquire)
                || !self
                    .owner
                    .stats_lease
                    .try_lock()
                    .is_ok_and(|slot| slot.is_none())
            {
                return false;
            }
            let state = self.runtime.state();
            if state.closed
                || state.logical_requests != 0
                || state.reserved_requests != 0
                || state.queued != 0
                || state.executions != 0
                || state.reserved.host_bytes != 0
                || state.reserved.device_bytes != 0
                || state.reserved.pinned_bytes != 0
            {
                return false;
            }
            let Ok(mut execution) = self.owner.role_execution.try_lock() else {
                return false;
            };
            if execution
                .as_ref()
                .is_some_and(|entry| !entry.ready_observed || entry.unknown_observed)
            {
                return false;
            }
            let Ok(mut perspective) = self.owner.prepared_perspective.try_lock() else {
                return false;
            };
            execution.take();
            perspective.take();
            self.last_value_execution = None;
            self.selected_review = None;
            Instant::now() < until
        }
        pub(super) fn ticketed_take_execution_evidence(
            &self,
            ticket: &rz_contracts::pals_adaptive::RoleTicketEnvelope,
        ) -> Option<super::ticketed::TicketExecutionEvidence> {
            let mut slot = self.owner.role_execution.lock().ok()?;
            let observed = slot
                .as_ref()
                .filter(|entry| entry.binding.request == ticket.id)?;
            let evidence = observed.ticket_execution_evidence()?;
            slot.take();
            Some(evidence)
        }
        /// Enable metadata from the same physical execution before the first
        /// startup/search admission. Legacy callers keep the Evaluate wire.
        pub fn enable_followup_execution_evidence(&mut self) -> Result<(), RoleError> {
            if self.sequence != 0
                || self.startup_attempted
                || self.owner.in_flight.load(Ordering::Acquire) != 0
                || self.owner.finishing.load(Ordering::Acquire)
                || self.owner.shutdown.load(Ordering::Acquire)
                || self.owner.followup_evidence.swap(true, Ordering::AcqRel)
            {
                return Err(RoleError::Backend(
                    "followup evidence must be selected once before admission".into(),
                ));
            }
            Ok(())
        }
        fn capture_value_execution(
            &mut self,
            context: &RoleLogicalContext,
            output: &ModelValueOutput,
            expected_state: &rz_position::PositionSnapshot,
        ) {
            self.last_value_execution = None;
            if !self.owner.followup_evidence.load(Ordering::Acquire) {
                return;
            }
            let Ok(observation) = self.owner.role_execution.try_lock() else {
                return;
            };
            let expected_state_semantic =
                rz_position::contracts::rules_snapshot_semantic_digest(expected_state);
            let Some(observation) = observation.as_ref().filter(|observation| {
                Some(observation.binding.request) == self.delivered_request
                    && observation.input_sha256 == output.input_sha256
                    && output.identity == self.value_identity
                    && observation.prepared_state.semantic == expected_state_semantic
                    && observation.prepared_state.revision.0 == expected_state.revision()
                    && output.state == expected_state.position_identity()
                    && output.perspective == expected_state.side_to_move()
            }) else {
                return;
            };
            let Some(prepared_perspective) = observation.prepared_perspective else {
                return;
            };
            let expected_perspective = match output.perspective {
                Color::White => rz_contracts::Color::White,
                Color::Black => rz_contracts::Color::Black,
            };
            if prepared_perspective != expected_perspective
                || observation.binding.request.epoch != self.epoch
                || observation.binding.execution.epoch != self.epoch
            {
                return;
            }
            self.last_value_execution = Some(RoleValueExecutionEvidence {
                request: observation.binding.request,
                execution: observation.binding.execution,
                prepared_state: observation.prepared_state,
                prepared_perspective,
                input_sha256: observation.input_sha256,
                model_epoch: self.model_epoch,
                context: context.into(),
                fresh: observation.fresh,
                physically_completed: observation.ready_observed.then_some(true),
                completed_nn_inputs: observation.completed_nn_inputs,
                complete: observation.ready_observed
                    && !observation.unknown_observed
                    && observation.measurement_complete
                    && observation
                        .completed_nn_inputs
                        .is_some_and(|count| (1..=2).contains(&count))
                    && observation.fresh == Some(true),
            });
        }
        pub fn load_pinned(
            export: &std::path::Path,
            expected_export_sha256: &str,
            pin: &rz_eval::runtime_pin::RuntimeLibraryPin,
            config: rz_eval::pals_onnx::PalsOnnxConfig,
            drain_limit: Duration,
        ) -> Result<Self, RoleError> {
            Self::load_pinned_with_options(
                export,
                expected_export_sha256,
                pin,
                config,
                NativeOwnerOptions {
                    drain_limit,
                    host_record_pages: None,
                },
            )
        }
        pub fn load_pinned_with_options(
            export: &std::path::Path,
            expected_export_sha256: &str,
            pin: &rz_eval::runtime_pin::RuntimeLibraryPin,
            config: rz_eval::pals_onnx::PalsOnnxConfig,
            options: NativeOwnerOptions,
        ) -> Result<Self, RoleError> {
            options.validate(&config)?;
            let runtime = rz_eval::onnx::OrtRuntime::load(pin).map_err(model_error)?;
            let backend = PalsOnnxBackend::load(export, expected_export_sha256, runtime, config)
                .map_err(model_error)?;
            Self::new_with_options(backend, options)
        }
        /// Additive CPU Fresh loader using the caller's original absolute clock.
        /// No synchronous ORT/session call is forcibly timed out. A late return
        /// is rejected with its original failure/elapsed facts and any actually
        /// captured owner handle; the caller must finish using budget.whole_until().
        /// ORT's process-lifetime pinned bootstrap/latch remains separate from
        /// native worker/session join; a finish handle does not unload that latch.
        /// This factory neither admits Query/2 nor proves registered binary,
        /// provider execution, physical closure or measured whole invocation cost.
        pub fn load_pinned_cpu_fresh_for_invocation(
            export: &std::path::Path,
            expected_export_sha256: &str,
            pin: &rz_eval::runtime_pin::RuntimeLibraryPin,
            config: rz_eval::pals_onnx::PalsOnnxConfig,
            options: NativeOwnerOptions,
            budget: NativeInvocationBudget,
            cancel: &AtomicBool,
        ) -> Result<Self, NativeLoadFailure> {
            let load_started = Instant::now();
            validate_cpu_fresh_invocation(&config, options, budget, cancel)?;
            validate_cpu_invocation_bundle(pin.bundle_digest())
                .map_err(|cause| budget.failure(NativeLoadStage::Admission, cause, cancel, None))?;
            budget.check_stage(NativeLoadStage::RuntimeLoad, cancel)?;
            let runtime = rz_eval::onnx::OrtRuntime::load(pin).map_err(|cause| {
                budget
                    .failure(
                        NativeLoadStage::RuntimeLoad,
                        model_error(&cause),
                        cancel,
                        None,
                    )
                    .with_backend_error(cause)
            })?;
            budget.check_stage(NativeLoadStage::RuntimeLoad, cancel)?;
            budget.check_stage(NativeLoadStage::BackendLoad, cancel)?;
            // The existing Fresh loader supplies no warm/resident/host-page
            // registration. Its own manifest/runtime validation is unchanged.
            let backend = PalsOnnxBackend::load(export, expected_export_sha256, runtime, config)
                .map_err(|cause| {
                    budget
                        .failure(
                            NativeLoadStage::BackendLoad,
                            model_error(&cause),
                            cancel,
                            None,
                        )
                        .with_backend_error(cause)
                })?;
            budget.check_stage(NativeLoadStage::BackendLoad, cancel)?;
            if backend.private_warm_capability().is_some() {
                return Err(budget.failure(
                    NativeLoadStage::BackendLoad,
                    RoleError::Backend(
                        "Native invocation loader refuses a private warm backend".into(),
                    ),
                    cancel,
                    None,
                ));
            }
            budget.check_stage(NativeLoadStage::OwnerConstruction, cancel)?;
            let mut finish = None;
            let model = Self::new_with_optional_cuda_record_pages_and_capture(
                backend,
                options,
                None,
                Some(&mut finish),
                None,
            )
            .map_err(|cause| {
                budget.failure(NativeLoadStage::OwnerConstruction, cause, cancel, finish)
            })?;
            // The returned model owns exactly the Arc captured before runtime
            // initialization. Post-check uses that owner, with no cleanup call
            // or new relative deadline hidden inside the failure path.
            let mut model = accept_invocation_owner(model, budget, cancel)?;
            let load_finished = Instant::now();
            if let Err(cause) = budget.check_at(load_finished, cancel) {
                let finish = Some(model.finish_handle());
                return Err(budget.failure(
                    NativeLoadStage::PostConstruction,
                    cause,
                    cancel,
                    finish,
                ));
            }
            // Actual factory entry through successful owner construction and
            // original-clock admission. Generic loaders never fill this fact.
            model.cpu_invocation_load_interval = Some((load_started, load_finished));
            Ok(model)
        }
        /// Explicit CPU-only approximate artifact domain. A legacy Fresh
        /// manifest never silently gains a seed input or frozen query policy.
        pub fn load_pinned_cpu_private_warm_with_options(
            export: &std::path::Path,
            expected_export_sha256: &str,
            pin: &rz_eval::runtime_pin::RuntimeLibraryPin,
            config: rz_eval::pals_onnx::PalsOnnxConfig,
            options: NativeOwnerOptions,
        ) -> Result<Self, RoleError> {
            options.validate(&config)?;
            if config.provider != rz_eval::onnx::Provider::Cpu || config.device_public_memory {
                return Err(RoleError::Backend(
                    "Native private warm supports only explicitly selected CPU/host graphs".into(),
                ));
            }
            let runtime = rz_eval::onnx::OrtRuntime::load(pin).map_err(model_error)?;
            let backend = PalsOnnxBackend::load_cpu_private_warm(
                export,
                expected_export_sha256,
                runtime,
                config,
            )
            .map_err(model_error)?;
            Self::new_with_options(backend, options)
        }
        #[cfg(feature = "experimental-io-binding")]
        pub fn load_with_runtime_and_cuda_private_warm_with_options(
            export: &std::path::Path,
            expected_export_sha256: &str,
            runtime: rz_eval::onnx::OrtRuntime,
            config: rz_eval::pals_onnx::PalsOnnxConfig,
            options: NativeOwnerOptions,
            control: Option<(PalsCudaControlPolicy, &std::path::Path)>,
        ) -> Result<Self, RoleError> {
            Self::load_with_runtime_and_cuda_private_warm_and_observations(
                export,
                expected_export_sha256,
                runtime,
                config,
                options,
                control,
                None,
            )
        }
        #[cfg(feature = "experimental-io-binding")]
        pub fn load_with_runtime_and_cuda_private_warm_and_observations(
            export: &std::path::Path,
            expected_export_sha256: &str,
            runtime: rz_eval::onnx::OrtRuntime,
            config: rz_eval::pals_onnx::PalsOnnxConfig,
            options: NativeOwnerOptions,
            control: Option<(PalsCudaControlPolicy, &std::path::Path)>,
            observation_limits: Option<NativeCudaWarmObservationLimits>,
        ) -> Result<Self, RoleError> {
            Self::load_with_runtime_and_cuda_private_warm_and_observations_diagnostic(
                export,
                expected_export_sha256,
                runtime,
                config,
                options,
                control,
                observation_limits,
            )
            .map_err(|failure| failure.cause)
        }
        /// Opt-in diagnostic sibling. Keeps the backend cause owned and typed
        /// before the historical RoleError projection, including load failures
        /// that occur before a physical worker or finish handle exists.
        #[cfg(feature = "experimental-io-binding")]
        pub fn load_with_runtime_and_cuda_private_warm_and_observations_diagnostic(
            export: &std::path::Path,
            expected_export_sha256: &str,
            runtime: rz_eval::onnx::OrtRuntime,
            config: rz_eval::pals_onnx::PalsOnnxConfig,
            options: NativeOwnerOptions,
            control: Option<(PalsCudaControlPolicy, &std::path::Path)>,
            observation_limits: Option<NativeCudaWarmObservationLimits>,
        ) -> Result<Self, NativeCudaWarmLoadError> {
            options
                .validate(&config)
                .map_err(|cause| NativeCudaWarmLoadError::role("owner_options", cause))?;
            if options.host_record_pages.is_some() {
                return Err(NativeCudaWarmLoadError::role("owner_options", RoleError::Backend("CUDA Warm requires the device public owner; host record pages cannot be selected".into())));
            }
            let backend = match control {
                Some((policy, profile)) => {
                    PalsOnnxBackend::load_cuda_private_warm_with_control_policy(
                        export,
                        expected_export_sha256,
                        runtime,
                        config,
                        policy,
                        profile,
                    )
                }
                None => PalsOnnxBackend::load_cuda_private_warm(
                    export,
                    expected_export_sha256,
                    runtime,
                    config,
                ),
            }
            .map_err(|cause| NativeCudaWarmLoadError::backend("backend_load", cause))?;
            if let Some(limits) = observation_limits {
                let execution = execution_receipt(&backend).map_err(|cause| {
                    NativeCudaWarmLoadError::role("execution_declaration", cause)
                })?;
                validate_cuda_warm_observation_admission(
                    limits,
                    &execution,
                    backend.private_cuda_warm_capability().is_some(),
                )
                .map_err(|cause| {
                    NativeCudaWarmLoadError::backend("observation_admission", cause)
                })?;
            }
            match observation_limits {
                Some(limits) => Self::new_with_cuda_warm_observations(backend, options, limits),
                None => Self::new_with_options(backend, options),
            }
            .map_err(|cause| NativeCudaWarmLoadError::role("owner_construction", cause))
        }
        pub fn load_pinned_with_cuda_control_policy(
            export: &std::path::Path,
            expected_export_sha256: &str,
            pin: &rz_eval::runtime_pin::RuntimeLibraryPin,
            config: rz_eval::pals_onnx::PalsOnnxConfig,
            policy: PalsCudaControlPolicy,
            profile_root: &std::path::Path,
            drain_limit: Duration,
        ) -> Result<Self, RoleError> {
            let runtime = rz_eval::onnx::OrtRuntime::load(pin).map_err(model_error)?;
            Self::load_with_runtime_and_cuda_control_policy(
                export,
                expected_export_sha256,
                runtime,
                config,
                policy,
                profile_root,
                drain_limit,
            )
        }
        /// A bootstrap-pinned runtime may carry an explicitly registered CUDA
        /// loading policy. The default pinned factory above still uses load().
        pub fn load_with_runtime_and_cuda_control_policy(
            export: &std::path::Path,
            expected_export_sha256: &str,
            runtime: rz_eval::onnx::OrtRuntime,
            config: rz_eval::pals_onnx::PalsOnnxConfig,
            policy: PalsCudaControlPolicy,
            profile_root: &std::path::Path,
            drain_limit: Duration,
        ) -> Result<Self, RoleError> {
            Self::load_with_runtime_and_cuda_control_policy_with_options(
                export,
                expected_export_sha256,
                runtime,
                config,
                policy,
                profile_root,
                NativeOwnerOptions {
                    drain_limit,
                    host_record_pages: None,
                },
            )
        }
        pub fn load_with_runtime_and_cuda_control_policy_with_options(
            export: &std::path::Path,
            expected_export_sha256: &str,
            runtime: rz_eval::onnx::OrtRuntime,
            config: rz_eval::pals_onnx::PalsOnnxConfig,
            policy: PalsCudaControlPolicy,
            profile_root: &std::path::Path,
            options: NativeOwnerOptions,
        ) -> Result<Self, RoleError> {
            options.validate(&config)?;
            let backend = PalsOnnxBackend::load_with_cuda_control_policy(
                export,
                expected_export_sha256,
                runtime,
                config,
                policy,
                profile_root,
            )
            .map_err(model_error)?;
            Self::new_with_options(backend, options)
        }
        pub fn new(backend: PalsOnnxBackend) -> Result<Self, RoleError> {
            Self::new_with_drain_limit(backend, DEFAULT_DRAIN_LIMIT)
        }
        /// The product's UCI shutdown window supplies the same finite drain
        /// limit. There is no independent 30-second native tail hidden behind
        /// a shorter protocol fence.
        pub fn new_with_drain_limit(
            backend: PalsOnnxBackend,
            drain_limit: Duration,
        ) -> Result<Self, RoleError> {
            Self::new_with_options(
                backend,
                NativeOwnerOptions {
                    drain_limit,
                    host_record_pages: None,
                },
            )
        }
        /// Install explicit record pages before native worker ownership and
        /// before the first chess P/C admission. Already-selected backend
        /// policies remain explicit selections and are attested as such.
        pub fn new_with_options(
            backend: PalsOnnxBackend,
            options: NativeOwnerOptions,
        ) -> Result<Self, RoleError> {
            Self::new_with_optional_cuda_record_pages(backend, options, None)
        }
        /// Additive V5 Fresh owner construction. The supplied clock is the
        /// process owner's immutable epoch and origin, used from the beginning
        /// by scope, runtime, request/execution IDs and physical worker metadata.
        /// Existing V1-V4 factories keep their internally allocated clocks.
        #[cfg(feature = "pals-adaptive")]
        pub(crate) fn new_registered_fresh_owner(
            backend: PalsOnnxBackend,
            options: NativeOwnerOptions,
            clock: ContractSystemClock,
            sources: Arc<dyn rz_eval::pals_onnx::PalsOwnedSources>,
        ) -> Result<Self, (RoleError, Option<NativeRoleFinishHandle>)> {
            if clock.domain().0.0 == 0
                || clock.origin() > Instant::now()
                || backend.model_config().profile != PalsModelProfile::FullLineInteractionV2
                || backend.private_warm_capability().is_some()
            {
                return Err((RoleError::InvalidOutput, None));
            }
            #[cfg(feature = "experimental-io-binding")]
            if backend.private_cuda_warm_capability().is_some() {
                return Err((RoleError::InvalidOutput, None));
            }
            let mut finish = None;
            Self::new_with_owner_registration(
                backend,
                options,
                None,
                Some(&mut finish),
                None,
                Some(clock),
                Some(sources),
            )
            .map_err(|cause| (cause, finish))
        }
        #[cfg(feature = "pals-adaptive")]
        pub(crate) fn registered_adaptive_namespace(
            &self,
        ) -> Result<rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace, RoleError>
        {
            rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace::new(
                self.value_identity.clone(),
                self.model,
                self.encoding,
                PrecisionProfile::Fp32,
                DEPLOYMENT_FROZEN_EPOCH,
            )
        }
        #[cfg(feature = "experimental-io-binding")]
        pub fn new_with_cuda_warm_observations(
            backend: PalsOnnxBackend,
            options: NativeOwnerOptions,
            limits: NativeCudaWarmObservationLimits,
        ) -> Result<Self, RoleError> {
            Self::new_with_optional_cuda_record_pages_and_capture(
                backend,
                options,
                None,
                None,
                Some(limits),
            )
        }
        /// Explicit resident factory for the OwnCPU product path. External CPU
        /// helpers are unsupported by this selection and must be refused before
        /// model loading by the caller. No caller namespace or epoch is accepted.
        pub fn new_with_registered_cuda_record_pages(
            backend: PalsOnnxBackend,
            options: NativeOwnerOptions,
            registration: NativeCudaRecordPagesRegistration,
        ) -> Result<Self, RoleError> {
            Self::new_with_optional_cuda_record_pages(backend, options, Some(registration))
        }
        /// Reuse the existing pinned runtime/control loader, then install the
        /// owned checked body before this backend enters a worker. The pair keeps
        /// finite protocol drain options separate from resident graph ownership.
        pub fn load_with_runtime_and_registered_cuda_record_pages(
            export: &std::path::Path,
            expected_export_sha256: &str,
            runtime: rz_eval::onnx::OrtRuntime,
            config: rz_eval::pals_onnx::PalsOnnxConfig,
            policy: PalsCudaControlPolicy,
            model_profile_root: &std::path::Path,
            owner: (NativeOwnerOptions, NativeCudaRecordPagesRegistration),
        ) -> Result<Self, RoleError> {
            let (options, registration) = owner;
            options.validate(&config)?;
            validate_cuda_record_pages_selection(
                &config,
                options,
                registration.frozen_epoch,
                false,
            )?;
            let resources =
                NativeCudaRecordPagesResourceDeclaration::from_registration(&registration)?;
            validate_cuda_record_page_session_declarations(&config, &resources)?;
            let backend = PalsOnnxBackend::load_with_cuda_control_policy(
                export,
                expected_export_sha256,
                runtime,
                config,
                policy,
                model_profile_root,
            )
            .map_err(model_error)?;
            Self::new_with_registered_cuda_record_pages(backend, options, registration)
        }
        fn new_with_optional_cuda_record_pages(
            backend: PalsOnnxBackend,
            options: NativeOwnerOptions,
            registration: Option<NativeCudaRecordPagesRegistration>,
        ) -> Result<Self, RoleError> {
            Self::new_with_optional_cuda_record_pages_and_capture(
                backend,
                options,
                registration,
                None,
                None,
            )
        }
        fn new_with_optional_cuda_record_pages_and_capture(
            backend: PalsOnnxBackend,
            options: NativeOwnerOptions,
            registration: Option<NativeCudaRecordPagesRegistration>,
            finish_capture: Option<&mut Option<NativeRoleFinishHandle>>,
            warm_observation_limits: Option<NativeCudaWarmObservationLimits>,
        ) -> Result<Self, RoleError> {
            Self::new_with_owner_registration(
                backend,
                options,
                registration,
                finish_capture,
                warm_observation_limits,
                None,
                None,
            )
        }
        #[allow(clippy::too_many_arguments)]
        fn new_with_owner_registration(
            mut backend: PalsOnnxBackend,
            options: NativeOwnerOptions,
            registration: Option<NativeCudaRecordPagesRegistration>,
            finish_capture: Option<&mut Option<NativeRoleFinishHandle>>,
            warm_observation_limits: Option<NativeCudaWarmObservationLimits>,
            registered_clock: Option<ContractSystemClock>,
            registered_sources: Option<Arc<dyn rz_eval::pals_onnx::PalsOwnedSources>>,
        ) -> Result<Self, RoleError> {
            options.validate(&backend.config())?;
            let drain_limit = options.drain_limit;
            let warm_capability = backend
                .private_warm_capability()
                .copied()
                .map(NativeWarmCapability::Cpu);
            #[cfg(feature = "experimental-io-binding")]
            let warm_capability = warm_capability.or_else(|| {
                backend
                    .private_cuda_warm_capability()
                    .cloned()
                    .map(NativeWarmCapability::Cuda)
            });
            let resident_resources = registration
                .as_ref()
                .map(|registration| {
                    validate_cuda_record_pages_selection(
                        &backend.config(),
                        options,
                        registration.frozen_epoch,
                        warm_capability.is_some(),
                    )?;
                    let resources =
                        NativeCudaRecordPagesResourceDeclaration::from_registration(registration)?;
                    validate_cuda_record_page_session_declarations(&backend.config(), &resources)?;
                    if backend.residency().native_sessions != 2
                        || backend.residency().layout != "shared_pc_if"
                    {
                        return Err(RoleError::InvalidOutput);
                    }
                    Ok(resources)
                })
                .transpose()?;
            #[cfg(feature = "experimental-io-binding")]
            if registration.is_none()
                && backend
                    .cuda_record_page_snapshot()
                    .map_err(model_error)?
                    .is_some()
            {
                return Err(RoleError::Backend(
                    "Resident CUDA backend requires the explicit owned factory".into(),
                ));
            }
            if matches!(warm_capability, Some(NativeWarmCapability::Cpu(_)))
                && (backend.config().provider != rz_eval::onnx::Provider::Cpu
                    || backend.config().device_public_memory)
            {
                return Err(RoleError::Backend(
                    "Native private warm CPU/host capability cannot select CUDA/device memory"
                        .into(),
                ));
            }
            if let Some(limits) = options.host_record_pages {
                let graph = backend
                    .residency()
                    .graphs
                    .iter()
                    .find(|graph| graph.role == "public")
                    .ok_or(RoleError::InvalidOutput)?;
                let mut policy = HostRecordPagePolicy::for_registered_graph(graph.sha256);
                policy.max_page_entries = limits.max_page_entries;
                policy.max_page_bytes = limits.max_page_bytes;
                policy.max_transient_bytes = limits.max_transient_bytes;
                backend
                    .enable_host_record_pages(policy)
                    .map_err(model_error)?;
            }
            let execution = execution_receipt(&backend)?;
            #[cfg(not(feature = "experimental-io-binding"))]
            if warm_observation_limits.is_some() {
                return Err(RoleError::Unavailable);
            }
            #[cfg(feature = "experimental-io-binding")]
            let warm_observer = warm_observation_limits
                .map(|limits| {
                    validate_cuda_warm_observation_admission(
                        limits,
                        &execution,
                        matches!(warm_capability, Some(NativeWarmCapability::Cuda(_))),
                    )
                    .map_err(model_error)?;
                    backend
                        .enable_cuda_private_warm_observations(
                            rz_eval::pals_onnx::PalsCudaWarmRuntimeLimits {
                                max_leases: limits.max_leases,
                                device_bytes_max: limits.device_bytes_max,
                            },
                        )
                        .map_err(model_error)
                })
                .transpose()?;
            let page_reservation = execution
                .host_record_pages
                .as_ref()
                .map_or(Ok(0), NativeHostRecordPageDeclaration::reserved_host_bytes)?;
            let additional_host_bytes = REQUEST_BYTES
                .checked_add(page_reservation)
                .and_then(|b| {
                    b.checked_add(
                        warm_capability
                            .as_ref()
                            .map_or(0, NativeWarmCapability::owner_bytes),
                    )
                })
                .and_then(|b| {
                    b.checked_add(
                        resident_resources
                            .as_ref()
                            .map_or(0, |r| r.max_invocation_host_bytes),
                    )
                })
                .ok_or(RoleError::InvalidOutput)?;
            let runtime_host_bytes = (4 * REQUEST_BYTES)
                .checked_add(page_reservation)
                .and_then(|b| {
                    b.checked_add(
                        warm_capability
                            .as_ref()
                            .map_or(0, NativeWarmCapability::owner_bytes),
                    )
                })
                .and_then(|b| {
                    b.checked_add(
                        resident_resources
                            .as_ref()
                            .map_or(0, |r| r.max_invocation_host_bytes),
                    )
                })
                .ok_or(RoleError::InvalidOutput)?;
            let is_cuda = execution.provider == "cuda";
            // Dormant until prepare_startup establishes the shared clock.
            // Successful default startup discards the optional diagnostics.
            let startup_stage_probe = is_cuda.then(|| backend.enable_startup_stage_probe());
            if !is_cuda {
                backend.verify_runtime().map_err(model_error)?;
            }
            // CUDA load already verifies pinned dependencies. The full origin
            // audit requires the first physical Run, performed by this same
            // worker in prepare_startup before protocol readiness is served.
            let model_config = backend.model_config().clone();
            verify_pals_rules_profile_for_config(
                backend.rules_input_profile(),
                backend.rules_input_semantic_sha256(),
                &model_config,
            )?;
            let model_epoch = backend.model_epoch();
            let model = Digest(backend.manifest_digest());
            let encoding = Digest(warm_capability.as_ref().map_or_else(
                || pals_fresh_encoding_semantic_digest_for_config(&model_config),
                |cap| cap.model_identity().encoding,
            ));
            let adapter_source_digest = if let Some(cap) = &warm_capability {
                let mut hash = Sha256::new();
                hash.update(b"rz-pals-native-private-warm-adapter/1");
                hash.update(pals_native_source_digest());
                hash.update(private_warm::source_digest());
                hash.update(cap.implementation_digest());
                hash.finalize().into()
            } else {
                pals_native_source_digest()
            };
            let trained = backend.is_trained();
            let residency = backend.residency().clone();
            let applied_session_settings = backend.applied_session_settings();
            let mut identity = format!(
                "pals-onnx-pc-fp32-{}",
                model
                    .0
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            );
            if warm_capability.is_some() {
                identity.push_str(
                    if matches!(warm_capability, Some(NativeWarmCapability::Cpu(_))) {
                        "-cpu-approx-warm-query-v1"
                    } else {
                        "-cuda-approx-warm-v2"
                    },
                );
            }
            if is_cuda {
                identity.push_str(&format!(
                    "-cuda-device{}-arena{}",
                    execution.device_id.unwrap_or(-1),
                    execution.session_arena_bytes.unwrap_or(0)
                ));
            }
            // The selected graph/encoding/FP32 namespace is fixed at startup.
            // Each delivered value still carries its actual prepared input key.
            let value_identity = frontier_value_identity(&identity, model_epoch, encoding);
            value_identity.validate()?;
            let epoch = match &registered_clock {
                Some(clock) => clock.domain().0,
                None => ProcessEpoch(next(&EPOCHS)?),
            };
            let cuda_record_pages = if let Some(registration) = registration {
                #[cfg(feature = "experimental-io-binding")]
                {
                    let artifact = registration.checked_graph.registration();
                    let retained = registration
                        .checked_graph
                        .known_retained_host_bytes()
                        .map_err(|_| RoleError::InvalidOutput)?;
                    let frozen_epoch = registration.frozen_epoch;
                    let declared_resources = resident_resources.ok_or(RoleError::InvalidOutput)?;
                    backend
                        .install_registered_cuda_record_pages(
                            rz_eval::pals_onnx::PalsCudaRecordPageRegistration {
                                checked_graph: registration.checked_graph,
                                process_epoch: epoch,
                                frozen_epoch,
                                limits: registration.limits,
                                invocation: registration.invocation,
                                profile_root: registration.profile_root,
                            },
                        )
                        .map_err(model_error)?;
                    Some(installed_cuda_record_pages_declaration(
                        &backend,
                        epoch,
                        frozen_epoch,
                        artifact,
                        retained,
                        declared_resources,
                    )?)
                }
                #[cfg(not(feature = "experimental-io-binding"))]
                {
                    let _ = registration;
                    return Err(RoleError::Backend(
                        "Resident CUDA pages require experimental-io-binding; fallback forbidden"
                            .into(),
                    ));
                }
            } else {
                None
            };
            let execution = NativeExecutionReceipt {
                cuda_record_pages,
                ..execution
            };
            let game = if warm_capability.is_some() { 0 } else { 1 };
            let private_warm = warm_capability
                .map(|capability| {
                    capability.state(game).and_then(|state| {
                        #[cfg(feature = "experimental-io-binding")]
                        if warm_observer.is_some() {
                            state.enable_followup_observations()?;
                        }
                        Ok(NativeWarmOwner {
                            capability,
                            state,
                            #[cfg(feature = "experimental-io-binding")]
                            observer: warm_observer,
                        })
                    })
                })
                .transpose()?;
            let clock = registered_clock.unwrap_or_else(|| ContractSystemClock::new(epoch));
            let authority = SearchAuthority {
                epoch,
                game: GameGeneration(game),
                root: RootGeneration(0),
                implementation: Digest(adapter_source_digest),
            };
            let scope = SharedPalsScope::new(PalsScope {
                authority,
                model,
                encoding,
                precision: PrecisionProfile::Fp32,
                frozen_epoch: DEPLOYMENT_FROZEN_EPOCH,
                mode: ExecutionMode::Deployment,
            });
            let (worker, host_record_page_observer) = if execution.host_record_pages.is_some() {
                let (worker, observer) = backend
                    .controlled_worker_with_host_record_page_observations()
                    .map_err(model_error)?;
                (worker, Some(observer))
            } else {
                (backend.controlled_worker().map_err(model_error)?, None)
            };
            let owner = Arc::new(WorkerOwner {
                worker: Mutex::new(worker),
                physical_completed: AtomicU64::new(0),
                completed: AtomicU64::new(0),
                physical_failed: AtomicU64::new(0),
                validation_failed: AtomicU64::new(0),
                delivered: AtomicU64::new(0),
                consumed: AtomicU64::new(0),
                canceled: AtomicU64::new(0),
                expired: AtomicU64::new(0),
                new_game_resets: AtomicU64::new(0),
                in_flight: AtomicU64::new(0),
                game_generation: AtomicU64::new(game),
                pending_game_target: AtomicU64::new(0),
                game_lifecycle: Mutex::new(()),
                game_reset_failure: Mutex::new(None),
                request_high_water: AtomicU64::new(0),
                execution_high_water: AtomicU64::new(0),
                quarantined: AtomicBool::new(false),
                finishing: AtomicBool::new(false),
                shutdown: AtomicBool::new(false),
                model_epoch,
                model_config,
                manifest_digest: model.0,
                encoding_semantic_digest: encoding.0,
                adapter_source_digest,
                trained,
                residency,
                applied_session_settings: Some(applied_session_settings),
                process_epoch: epoch,
                _registered_sources: registered_sources,
                last_failure: Mutex::new(None),
                final_stats: Mutex::new(None),
                stats_lease: Mutex::new(None),
                cuda_record_page_snapshot: Mutex::new(None),
                cuda_record_page_snapshot_error: Mutex::new(None),
                execution: execution.clone(),
                additional_host_bytes,
                host_record_page_observer,
                startup_probe_timeout_ms: AtomicU64::new(0),
                startup_stage_probe,
                startup_probe: Mutex::new(is_cuda.then_some(NativeStartupProbeReceipt {
                    completed_proposer_calls: 0,
                    completed_critic_calls: 0,
                    runtime_mapping_confirmed: false,
                    runtime_loading_mapping: None,
                    cuda_placement_witness: None,
                    reset_completed: false,
                    backend_stats: None,
                    failure_diagnostic: None,
                    backend_stages: None,
                    timing_observation: None,
                    cuda_record_page_observation: None,
                    cuda_record_page_observation_error: None,
                })),
                startup_ready: AtomicBool::new(!is_cuda),
                final_mapping_confirmed: AtomicBool::new(false),
                final_loading_mapping: Mutex::new(None),
                observer: Mutex::new(None),
                observer_failures: AtomicU64::new(0),
                last_observer_failure: Mutex::new(None),
                role_execution: Mutex::new(None),
                followup_evidence: AtomicBool::new(false),
                prepared_perspective: Mutex::new(None),
                private_warm,
                warm_admission: Mutex::new(None),
            });
            if let Some(slot) = finish_capture {
                // Actual worker/owner exists here, before all fallible adapter,
                // resource and Runtime construction. Default factories do not
                // request a capture and retain their existing error semantics.
                *slot = Some(NativeRoleFinishHandle {
                    owner: Arc::clone(&owner),
                });
            }
            let runtime = PalsRuntime::new(
                PalsAdapter::new(scope.clone(), clock.clone()).map_err(model_error)?,
                WorkerBackend {
                    owner: Arc::clone(&owner),
                },
                Limits {
                    max_requests: 1,
                    max_batch_items: 1,
                    max_executions: 1,
                    max_batch_wait: Duration::ZERO,
                    max_queue_age: Duration::from_secs(180),
                    deadline_reserve: Duration::ZERO,
                    memory: Resources {
                        host_bytes: runtime_host_bytes,
                        device_bytes: execution.runtime_device_reservation()?,
                        pinned_bytes: execution.pinned_request_bytes,
                    },
                },
                64,
            )
            .map_err(model_error)?;
            Ok(Self {
                identity,
                value_identity,
                epoch,
                model_epoch,
                model,
                encoding,
                clock,
                scope,
                runtime,
                owner,
                sequence: 0,
                game,
                unusable: false,
                drain_limit,
                pending_new_game: false,
                control_lease: None,
                startup_attempted: false,
                startup_probe_configured: false,
                delivered_request: None,
                last_value_execution: None,
                selected_review: None,
                ticketed_delivered: Vec::with_capacity(16),
                startup_loading_elapsed_ns: None,
                startup_clock: None,
                startup_diagnostic: None,
                cpu_invocation_budget: None,
                cpu_invocation_load_interval: None,
            })
        }
        pub fn finish_handle(&self) -> NativeRoleFinishHandle {
            NativeRoleFinishHandle {
                owner: Arc::clone(&self.owner),
            }
        }
        pub fn model_epoch(&self) -> [u8; 32] {
            self.model_epoch
        }
        pub fn model_config(&self) -> &PalsModelConfig {
            &self.owner.model_config
        }
        /// 원본 session 생성 성공 설정. OS thread inventory/affinity/첫 NN 증명은 아니다.
        pub fn applied_session_settings(&self) -> Option<Arc<[PalsAppliedSessionSettings]>> {
            self.owner.applied_session_settings.as_ref().map(Arc::clone)
        }
        fn fresh_command(&self, input: PalsModelInput) -> Result<PalsNativeCommand, RoleError> {
            #[cfg(feature = "experimental-io-binding")]
            if let Some(NativeWarmOwner {
                capability: NativeWarmCapability::Cuda(cap),
                ..
            }) = &self.owner.private_warm
            {
                let mut payload = PalsCudaWarmInput::fresh(cap, input).map_err(model_error)?;
                if let Some(observer) = self
                    .owner
                    .private_warm
                    .as_ref()
                    .and_then(|warm| warm.observer.as_ref())
                {
                    payload
                        .bind_execution_observation(
                            observer,
                            rz_eval::pals_onnx::PalsCudaWarmExecutionBinding {
                                backend_owner_id: observer.owner_id(),
                                purpose: rz_eval::pals_onnx::PalsCudaWarmInvocationPurpose::Startup,
                                request: None,
                                execution: None,
                                bank_lease_id: None,
                            },
                        )
                        .map_err(model_error)?;
                }
                return Ok(PalsNativeCommand::EvaluateCudaWarm(payload));
            }
            Ok(PalsNativeCommand::Evaluate(input))
        }
        /// Actual CPU invocation-factory interval in the caller's Instant domain.
        /// None is unobserved, including generic loaders; it is never zero cost.
        pub fn invocation_load_interval(&self) -> Option<(Instant, Instant)> {
            self.cpu_invocation_load_interval
        }
        pub fn source_identity(&self) -> NativeRoleSourceIdentity {
            NativeRoleSourceIdentity {
                checkpoint_sha256: self.model_epoch,
                frozen_epoch: DEPLOYMENT_FROZEN_EPOCH,
                export_manifest_sha256: self.owner.manifest_digest,
                encoding_semantic_sha256: self.owner.encoding_semantic_digest,
                adapter_source_sha256: self.owner.adapter_source_digest,
                model_configuration: self.owner.model_config.clone(),
                trained: self.owner.trained,
                execution: self.owner.execution_receipt(),
            }
        }
        /// Observe actual NN-zero counters immediately before a borrowed captured
        /// input loop. The ACK proves this fence only; it does not promise that a
        /// caller can release the model borrow and retain future pristine state.
        pub fn witness_recorded_cpu_pristine(
            &mut self,
            budget: NativeInvocationBudget,
            cancel: &AtomicBool,
        ) -> Result<NativeRecordedCpuPristineAck, RoleError> {
            let started = Instant::now();
            budget.check_at(started, cancel)?;
            let load = self.cpu_invocation_load_interval.ok_or_else(|| RoleError::Backend(
                "Recorded pristine ACK requires the actual CPU invocation factory load interval".into()
            ))?;
            if self.cpu_invocation_budget != Some(budget)
                || self.owner.execution.provider != "cpu"
                || self.owner.execution.runtime_bundle_sha256.is_some()
                || self.owner.execution.device_public_memory
                || self.owner.execution.host_record_pages.is_some()
                || self.owner.execution.private_warm.is_some()
                || self.owner.execution.cuda_record_pages.is_some()
                || self.owner.private_warm.is_some()
                || self.drain_limit != budget.cleanup_reserve()
                || self.sequence != 0
                || self.owner.physical_completed.load(Ordering::Acquire) != 0
                || self.owner.consumed.load(Ordering::Acquire) != 0
                || self
                    .owner
                    .observer
                    .lock()
                    .map_err(|_| RoleError::Unavailable)?
                    .is_some()
                || load.0 < budget.started()
                || load.1 < load.0
                || load.1 > started
            {
                return Err(RoleError::Backend("Recorded pristine ACK requires original CPU Fresh factory, clock and unused product owner".into()));
            }
            let stats = self.recorded_witness_stats(budget, cancel)?;
            require_recorded_nn_zero(&stats)?;
            let finished = Instant::now();
            budget.check_at(finished, cancel)?;
            let ns = |at: Instant| {
                u64::try_from(at.duration_since(budget.started()).as_nanos())
                    .map_err(|_| RoleError::InvalidOutput)
            };
            let ack = NativeRecordedCpuPristineAck {
                schema: "rz-pals-native-recorded-cpu-pristine-ack/1",
                observation: "actual_exclusive_cpu_worker_snapshot_stats_ready;nn_zero_at_this_fence;future_exclusive_model_borrow_required",
                source_identity: self.source_identity(),
                process_epoch: self.epoch.0,
                backend_stats: stats,
                started_elapsed_ns: ns(started)?,
                completed_elapsed_ns: ns(finished)?,
                physical_completion_confirmed: true,
                invocation_budget: budget,
                invocation_load_interval: load,
                work_started_at: started,
                work_finished_at: finished,
            };
            budget.check_at(Instant::now(), cancel)?;
            Ok(ack)
        }
        /// Execute the exact immutable captured tensor in a separate observation
        /// namespace. Historical query[7] is preserved, not regenerated as the
        /// current remaining deadline. The typed result proves only this actual
        /// CPU inference; the capture owner must separately bind source bytes,
        /// Rules, parent/request registration and independent CPU conditions.
        /// No product RoleQuery, search consumption or startup-probe count is
        /// manufactured by this witness-only path.
        pub fn witness_recorded_cpu_input(
            &mut self,
            input: PalsModelInput,
            expected_input_key: [u8; 32],
            budget: NativeInvocationBudget,
            cancel: &AtomicBool,
        ) -> Result<NativeRecordedInputWitness, RoleError> {
            budget.check_at(Instant::now(), cancel)?;
            if self.cpu_invocation_budget != Some(budget)
                || self.owner.execution.provider != "cpu"
                || self.owner.execution.runtime_bundle_sha256.is_some()
                || self.owner.execution.device_public_memory
                || self.owner.execution.host_record_pages.is_some()
                || self.owner.execution.private_warm.is_some()
                || self.owner.execution.cuda_record_pages.is_some()
                || self.owner.private_warm.is_some()
                || self.drain_limit != budget.cleanup_reserve()
                || self.sequence != 0
                || self.owner.physical_completed.load(Ordering::Acquire) != 0
                || self.owner.consumed.load(Ordering::Acquire) != 0
                || self
                    .owner
                    .observer
                    .lock()
                    .map_err(|_| RoleError::Unavailable)?
                    .is_some()
                || !matches!(input.role, PalsRole::Proposer | PalsRole::Critic)
                || input.model_epoch != self.model_epoch
            {
                return Err(RoleError::Backend(
                    "Recorded input witness requires CPU Fresh P/C owner, exact model and original cleanup reserve".into(),
                ));
            }
            input
                .validate(&self.owner.model_config)
                .map_err(model_error)?;
            if input
                .canonical_input_key(&self.owner.model_config)
                .map_err(model_error)?
                != expected_input_key
            {
                return Err(RoleError::Backend(
                    "Recorded exact input fingerprint differs".into(),
                ));
            }
            let work_started_at = Instant::now();
            budget.check_at(work_started_at, cancel)?;
            let before = self.recorded_witness_stats(budget, cancel)?;
            let lease_sequence = next(&RECORDED_WITNESS_LEASES)?;
            let (returned, actual_input) =
                self.recorded_witness_command(PalsNativeCommand::Evaluate(input), budget, cancel)?;
            let PalsNativeResult::Evaluation(raw_output) = returned else {
                return Err(RoleError::InvalidOutput);
            };
            let input = actual_input.ok_or(RoleError::InvalidOutput)?;
            // Decode the immutable physical job, never a reconstructed RoleQuery.
            raw_output
                .decode(&input, &self.owner.model_config)
                .map_err(model_error)?;
            if input
                .canonical_input_key(&self.owner.model_config)
                .map_err(model_error)?
                != expected_input_key
            {
                return Err(RoleError::InvalidOutput);
            }
            budget.check_at(Instant::now(), cancel)?;
            let after = self.recorded_witness_stats(budget, cancel)?;
            let public = after
                .public_nn_runs_completed
                .checked_sub(before.public_nn_runs_completed)
                .ok_or(RoleError::InvalidOutput)?;
            if public > 1
                || after
                    .public_nn_runs_attempted
                    .checked_sub(before.public_nn_runs_attempted)
                    != Some(public)
                || after
                    .role_nn_runs_attempted
                    .checked_sub(before.role_nn_runs_attempted)
                    != Some(1)
                || after
                    .role_nn_runs_completed
                    .checked_sub(before.role_nn_runs_completed)
                    != Some(1)
                || after
                    .admitted_role_requests
                    .checked_sub(before.admitted_role_requests)
                    != Some(1)
                || after
                    .completed_nn_inputs
                    .checked_sub(before.completed_nn_inputs)
                    != public.checked_add(1)
                || after.public_nn_runs_failed_known != before.public_nn_runs_failed_known
                || after.role_nn_runs_failed_known != before.role_nn_runs_failed_known
                || after
                    .validated_role_outputs
                    .checked_sub(before.validated_role_outputs)
                    != Some(1)
            {
                return Err(RoleError::Backend(
                    "Recorded witness actual graph counters differ".into(),
                ));
            }
            let raw_output_bit_projection = crate::pals_cpu_task::strategic_action::native_replay::native_raw_output_bit_projection(&raw_output)
                .map_err(|error| RoleError::Backend(format!("Recorded raw projection: {error:?}")))?;
            let work_finished_at = Instant::now();
            budget.check_at(work_finished_at, cancel)?;
            let ns = |at: Instant| {
                u64::try_from(at.saturating_duration_since(budget.started()).as_nanos())
                    .map_err(|_| RoleError::InvalidOutput)
            };
            let witness = NativeRecordedInputWitness {
                schema: RECORDED_INPUT_WITNESS_SCHEMA,
                scope: RECORDED_INPUT_WITNESS_SCOPE,
                implementation_sha256: recorded_input_witness_source_digest(),
                source_identity: self.source_identity(),
                process_epoch: self.epoch.0,
                lease_sequence,
                input_key: expected_input_key,
                historical_deadline_feature_bits: input.query[7].to_bits(),
                input,
                raw_output,
                raw_output_bit_projection,
                started_elapsed_ns: ns(work_started_at)?,
                completed_elapsed_ns: ns(work_finished_at)?,
                execution_until_elapsed_ns: ns(budget.execution_until())?,
                whole_until_elapsed_ns: ns(budget.whole_until())?,
                physical_completion_confirmed: true,
                backend_stats_before: before,
                backend_stats_after: after,
                invocation_budget: budget,
                invocation_load_interval: self.cpu_invocation_load_interval,
                invocation_load_elapsed_ns: self
                    .cpu_invocation_load_interval
                    .map(|(started, finished)| -> Result<_, RoleError> {
                        Ok((ns(started)?, ns(finished)?))
                    })
                    .transpose()?,
                work_started_at,
                work_finished_at,
            };
            // Final live gate after allocating the observation. A late or
            // cancelled return never exposes a successful typed witness.
            budget.check_at(Instant::now(), cancel)?;
            Ok(witness)
        }
        fn recorded_witness_stats(
            &mut self,
            budget: NativeInvocationBudget,
            cancel: &AtomicBool,
        ) -> Result<NativeBackendStatsReceipt, RoleError> {
            match self
                .recorded_witness_command(PalsNativeCommand::SnapshotStats, budget, cancel)?
                .0
            {
                PalsNativeResult::Stats(stats) => {
                    stats.validate().map_err(model_error)?;
                    Ok(stats.into())
                }
                _ => Err(RoleError::InvalidOutput),
            }
        }
        fn recorded_witness_command(
            &mut self,
            command: PalsNativeCommand,
            budget: NativeInvocationBudget,
            cancel: &AtomicBool,
        ) -> Result<(PalsNativeResult, Option<PalsModelInput>), RoleError> {
            budget.check_at(Instant::now(), cancel)?;
            if self.owner.quarantined.load(Ordering::Acquire) {
                return Err(RoleError::PhysicalCompletionUnknown);
            }
            if self.unusable
                || !self.owner.startup_ready.load(Ordering::Acquire)
                || self.owner.finishing.load(Ordering::Acquire)
                || self.owner.shutdown.load(Ordering::Acquire)
                || self.control_lease.is_some()
                || self.runtime.state().executions != 0
                || self.owner.in_flight.load(Ordering::Acquire) != 0
                || self.delivered_request.is_some()
                || self.pending_new_game
            {
                return Err(RoleError::Unavailable);
            }
            // The input/session is retained by this actual lease through Ready
            // or quarantine. No now+reserve timeout replaces original W.
            self.control_lease = Some({
                let mut worker = self
                    .owner
                    .worker
                    .lock()
                    .map_err(|_| RoleError::Unavailable)?;
                let _game_guard = self
                    .owner
                    .game_lifecycle
                    .lock()
                    .map_err(|_| RoleError::Unavailable)?;
                budget.check_at(Instant::now(), cancel)?;
                if self.owner.finishing.load(Ordering::Acquire)
                    || self.owner.shutdown.load(Ordering::Acquire)
                {
                    return Err(RoleError::Unavailable);
                }
                let lease = worker.submit(command).map_err(model_error)?;
                self.owner.in_flight.fetch_add(1, Ordering::AcqRel);
                lease
            });
            let mut logical_error = None;
            loop {
                if logical_error.is_none() {
                    logical_error = budget.check_at(Instant::now(), cancel).err();
                }
                let lease = self
                    .control_lease
                    .as_mut()
                    .expect("admitted recorded witness lease");
                match lease.poll() {
                    PhysicalPoll::Ready(result) => {
                        self.owner.in_flight.fetch_sub(1, Ordering::AcqRel);
                        let actual_input = match lease.input() {
                            PalsNativeCommand::Evaluate(input)
                            | PalsNativeCommand::EvaluateWithEvidence(input) => Some(input.clone()),
                            _ => None,
                        };
                        self.control_lease = None;
                        let result = result.map_err(|error| {
                            self.owner.remember_failure(&error);
                            model_error(error)
                        })?;
                        if let Some(error) = logical_error {
                            return Err(error);
                        }
                        budget.check_at(Instant::now(), cancel)?;
                        return Ok((result, actual_input));
                    }
                    PhysicalPoll::Quarantined | PhysicalPoll::Consumed => {
                        if let Ok(Some(error)) = lease.quarantine_cause() {
                            self.owner.remember_failure(&error);
                        }
                        self.owner.quarantined.store(true, Ordering::Release);
                        self.unusable = true;
                        return Err(RoleError::PhysicalCompletionUnknown);
                    }
                    PhysicalPoll::Pending => {}
                }
                if Instant::now() >= budget.whole_until() {
                    // Retain the actual lease and worker closure/session/input.
                    // A logical cancel or E expiry never confirms physical end.
                    self.owner.quarantined.store(true, Ordering::Release);
                    self.unusable = true;
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        pub fn set_observer(
            &mut self,
            observer: Box<dyn NativeRoleObserver>,
        ) -> Result<(), RoleError> {
            if self.owner.private_warm.is_some() {
                return Err(RoleError::Backend("Native private warm collector schema is unsupported; Fresh observer evidence is not warm provenance".into()));
            }
            self.install_observer(observer)
        }
        /// Opt into the actual private Warm namespace. The collector must
        /// implement prepared_private_warm and sees borrowed immutable payloads.
        pub fn set_private_warm_observer(
            &mut self,
            observer: Box<dyn NativeRoleObserver>,
        ) -> Result<(), RoleError> {
            if self.owner.private_warm.is_none() {
                return Err(RoleError::Backend(
                    "private Warm observer requires an actual loaded private Warm capability"
                        .into(),
                ));
            }
            self.install_observer(observer)
        }
        fn install_observer(
            &mut self,
            observer: Box<dyn NativeRoleObserver>,
        ) -> Result<(), RoleError> {
            if self.runtime.state().executions != 0
                || self.owner.in_flight.load(Ordering::Acquire) != 0
                || self.delivered_request.is_some()
            {
                return Err(RoleError::Unavailable);
            }
            *self
                .owner
                .observer
                .lock()
                .map_err(|_| RoleError::Unavailable)? = Some(observer);
            Ok(())
        }
        /// Close only the most recent delivered, unconsumed output's accounting
        /// token. No physical lease, cancellation, drain or buffer lifetime is
        /// changed; the search owner calls this before its collector drains.
        pub fn close_unconsumed(&mut self, reason: NativeRoleRejection) -> Result<(), RoleError> {
            if let Some(id) = self.delivered_request.take() {
                if let Some(warm) = &self.owner.private_warm {
                    warm.state.reject_output(Some(id));
                }
                if let Err(error) = self.owner.observe("search_output_closed", |observer| {
                    observer.rejected(id, reason)
                }) {
                    self.unusable = true;
                    return Err(error);
                }
            }
            if self.runtime.state().executions == 0
                && !self.owner.quarantined.load(Ordering::Acquire)
            {
                self.selected_review = None;
            }
            Ok(())
        }
        /// CUDA readiness is obtained by the existing exclusive worker, not
        /// by an untracked temporary session or fabricated search consumption.
        pub fn observe_startup_loading(&mut self, elapsed: Duration) -> Result<(), RoleError> {
            if self.startup_attempted || self.startup_loading_elapsed_ns.is_some() {
                return Err(RoleError::Unavailable);
            }
            self.startup_loading_elapsed_ns = u64::try_from(elapsed.as_nanos()).ok();
            Ok(())
        }
        /// Explicit cold-start admission budget, independent of the arena's
        /// protocol readiness and of the post-cancel physical drain limit.
        /// Absent selection keeps the existing 15-second behavior and wire.
        pub fn configure_startup_probe_timeout(
            &mut self,
            selected_ms: Option<u64>,
        ) -> Result<Duration, RoleError> {
            if self.startup_attempted || self.startup_probe_configured {
                return Err(RoleError::Unavailable);
            }
            if selected_ms.is_some_and(|value| !(1..=180_000).contains(&value)) {
                return Err(RoleError::InvalidOutput);
            }
            self.startup_probe_configured = true;
            self.owner
                .startup_probe_timeout_ms
                .store(selected_ms.unwrap_or(0), Ordering::Release);
            Ok(Duration::from_millis(selected_ms.unwrap_or(15_000)))
        }
        fn startup_elapsed_ns(&self) -> Option<u64> {
            self.startup_clock
                .and_then(|start| u64::try_from(start.elapsed().as_nanos()).ok())
        }
        pub fn prepare_startup(&mut self, until: Instant) -> Result<(), RoleError> {
            if self.owner.startup_ready.load(Ordering::Acquire) {
                return Ok(());
            }
            if self.startup_attempted || self.unusable {
                return Err(RoleError::Unavailable);
            }
            self.startup_attempted = true;
            let started = Instant::now();
            self.startup_clock = Some(started);
            if let Some(probe) = self.owner.startup_stage_probe.as_ref() {
                if self.owner.startup_probe_timeout_ms.load(Ordering::Acquire) != 0 {
                    let _ = probe.start_with_input_observations(started);
                } else {
                    let _ = probe.start(started);
                }
            }
            self.startup_diagnostic = Some(NativeStartupFailureDiagnostic {
                schema: "rz-pals-startup-failure-diagnostic/1",
                model_loading_elapsed_ns: self.startup_loading_elapsed_ns,
                probe_budget_ns: u64::try_from(until.saturating_duration_since(started).as_nanos())
                    .ok(),
                probe_elapsed_ns: None,
                drain_limit_ns: u64::try_from(self.drain_limit.as_nanos()).ok(),
                current_stage: "prepare_proposer_input",
                primary_error: None,
                logical_deadline_expired: false,
                commands: Vec::with_capacity(7),
                command_observation_overflow: false,
                backend_stages: None,
            });
            let outcome = (|| {
                let position = Position::startpos();
                let legal = position.legal_moves();
                let cancel = AtomicBool::new(false);
                for kind in [NativeQueryKind::Propose, NativeQueryKind::Reply] {
                    if let Some(diagnostic) = self.startup_diagnostic.as_mut() {
                        diagnostic.current_stage = if kind == NativeQueryKind::Reply {
                            "prepare_critic_input"
                        } else {
                            "prepare_proposer_input"
                        };
                    }
                    let input = prepare_role_input_for_config(
                        &RoleQuery {
                            position: &position,
                            legal: &legal,
                            prefix: &[],
                            proposal: &[],
                            counterexample: None,
                            records: &[],
                            revision: 0,
                            deadline: until,
                            cancel: &cancel,
                        },
                        kind,
                        self.model_epoch,
                        &self.owner.model_config,
                    )?;
                    if !matches!(
                        self.startup_command(self.fresh_command(input)?, until)?,
                        PalsNativeResult::Evaluation(_)
                    ) {
                        return Err(RoleError::InvalidOutput);
                    }
                }
                if !matches!(
                    self.startup_command(PalsNativeCommand::VerifyRuntime, until)?,
                    PalsNativeResult::RuntimeVerified
                ) {
                    return Err(RoleError::InvalidOutput);
                }
                self.owner
                    .startup_probe
                    .lock()
                    .map_err(|_| RoleError::Unavailable)?
                    .as_mut()
                    .ok_or(RoleError::Unavailable)?
                    .runtime_mapping_confirmed = true;
                if self.owner.execution.provider == "cuda" {
                    if self.owner.execution.cuda_loading_profile.is_some() {
                        let mapping = match self
                            .startup_command(PalsNativeCommand::ObserveRuntimeMappings, until)?
                        {
                            PalsNativeResult::RuntimeMappingsObserved(witness) => witness,
                            _ => return Err(RoleError::InvalidOutput),
                        };
                        validate_runtime_loading_mapping(&mapping, &self.owner.execution).map_err(
                            |error| {
                                self.owner.remember_failure(&error);
                                model_error(error)
                            },
                        )?;
                        self.owner
                            .startup_probe
                            .lock()
                            .map_err(|_| RoleError::Unavailable)?
                            .as_mut()
                            .ok_or(RoleError::Unavailable)?
                            .runtime_loading_mapping = Some(mapping);
                    }
                    let witness = match self
                        .startup_command(PalsNativeCommand::VerifyCudaPlacement, until)?
                    {
                        PalsNativeResult::CudaPlacementVerified(witness) => witness,
                        _ => return Err(RoleError::InvalidOutput),
                    };
                    validate_startup_cuda_witness(
                        &witness,
                        &self.owner.execution,
                        self.owner.manifest_digest,
                        &self.owner.residency,
                    )?;
                    self.owner
                        .startup_probe
                        .lock()
                        .map_err(|_| RoleError::Unavailable)?
                        .as_mut()
                        .ok_or(RoleError::Unavailable)?
                        .cuda_placement_witness = Some(witness);
                }
                let pending_target = self.owner.pending_game_target.load(Ordering::Acquire);
                if pending_target == 0 {
                    // Preserve the existing physical 0 -> logical 1 startup.
                    if !matches!(
                        self.startup_command(PalsNativeCommand::NewGame, until)?,
                        PalsNativeResult::NewGame
                    ) {
                        return Err(RoleError::InvalidOutput);
                    }
                } else {
                    match self
                        .startup_command(PalsNativeCommand::ResetTo(pending_target), until)
                        .map_err(|error| {
                            if error == RoleError::PhysicalCompletionUnknown {
                                error
                            } else {
                                self.owner.remember_game_reset_failure(error)
                            }
                        })? {
                        PalsNativeResult::ResetTo { game_generation }
                            if game_generation == pending_target =>
                        {
                            self.owner
                                .accept_game_target(game_generation)
                                .map_err(|error| self.owner.remember_game_reset_failure(error))?;
                            self.pending_new_game = false;
                        }
                        _ => {
                            return Err(self
                                .owner
                                .remember_game_reset_failure(RoleError::InvalidOutput));
                        }
                    }
                }
                self.owner
                    .startup_probe
                    .lock()
                    .map_err(|_| RoleError::Unavailable)?
                    .as_mut()
                    .ok_or(RoleError::Unavailable)?
                    .reset_completed = true;
                #[cfg(feature = "experimental-io-binding")]
                if self.owner.execution.cuda_record_pages.is_some() {
                    let observed = (|| {
                        let snapshot = match self
                            .startup_command(PalsNativeCommand::SnapshotCudaRecordPages, until)?
                        {
                            PalsNativeResult::CudaRecordPagesObserved(snapshot) => *snapshot,
                            _ => return Err(RoleError::InvalidOutput),
                        };
                        let selection = self
                            .owner
                            .execution
                            .cuda_record_pages
                            .as_ref()
                            .ok_or(RoleError::InvalidOutput)?;
                        let snapshot =
                            checked_cuda_record_page_observation(snapshot.into(), selection)?;
                        if snapshot.game_generation
                            != self.owner.game_generation.load(Ordering::Acquire)
                            || snapshot.live_blocks != 0
                            || snapshot.certified_projections != 0
                        {
                            return Err(RoleError::InvalidOutput);
                        }
                        Ok(snapshot)
                    })();
                    let mut probe = self
                        .owner
                        .startup_probe
                        .lock()
                        .map_err(|_| RoleError::Unavailable)?;
                    let probe = probe.as_mut().ok_or(RoleError::Unavailable)?;
                    match observed {
                        Ok(snapshot) => probe.cuda_record_page_observation = Some(snapshot),
                        Err(error) => {
                            probe.cuda_record_page_observation_error =
                                Some(NativeStartupErrorKind::from(&error));
                            return Err(error);
                        }
                    }
                }
                let stats = match self.startup_command(PalsNativeCommand::SnapshotStats, until)? {
                    PalsNativeResult::Stats(stats) => stats,
                    _ => return Err(RoleError::InvalidOutput),
                };
                stats.validate().map_err(model_error)?;
                let cache_entries = stats.live_public_cache_entries;
                self.owner
                    .startup_probe
                    .lock()
                    .map_err(|_| RoleError::Unavailable)?
                    .as_mut()
                    .ok_or(RoleError::Unavailable)?
                    .backend_stats = Some(stats.into());
                if cache_entries != 0 {
                    return Err(RoleError::InvalidOutput);
                }
                if Instant::now() >= until {
                    return Err(RoleError::Deadline);
                }
                self.owner.startup_ready.store(true, Ordering::Release);
                Ok(())
            })();
            let mut backend_stages = self
                .owner
                .startup_stage_probe
                .as_ref()
                .map(PalsStartupStageProbe::snapshot_and_stop);
            if let Some(snapshot) = backend_stages.as_mut()
                && snapshot.input_observations.is_some()
            {
                let witness = self.owner.startup_probe.lock().ok().and_then(|probe| {
                    probe
                        .as_ref()
                        .and_then(|probe| probe.cuda_placement_witness.clone())
                });
                if let Some(diagnostic) = self.startup_diagnostic.as_ref() {
                    bind_startup_input_observations(
                        snapshot,
                        &diagnostic.commands,
                        witness.as_deref(),
                    );
                }
            }
            if let Err(error) = &outcome {
                self.unusable = true;
                if self.owner.execution.cuda_record_pages.is_some()
                    && let Ok(mut probe) = self.owner.startup_probe.lock()
                    && let Some(probe) = probe.as_mut()
                    && probe.cuda_record_page_observation.is_none()
                {
                    probe
                        .cuda_record_page_observation_error
                        .get_or_insert(NativeStartupErrorKind::from(error));
                }
                let elapsed = self.startup_elapsed_ns();
                if let Some(mut diagnostic) = self.startup_diagnostic.take() {
                    diagnostic.probe_elapsed_ns = elapsed;
                    diagnostic.primary_error = Some(NativeStartupErrorKind::from(error));
                    diagnostic.logical_deadline_expired = Instant::now() >= until;
                    diagnostic.backend_stages = backend_stages;
                    if let Ok(mut probe) = self.owner.startup_probe.lock()
                        && let Some(probe) = probe.as_mut()
                    {
                        probe.failure_diagnostic = Some(diagnostic);
                    }
                }
            } else {
                if self.owner.startup_probe_timeout_ms.load(Ordering::Acquire) != 0 {
                    let elapsed = self.startup_elapsed_ns();
                    let timing = self.startup_diagnostic.take().map(|diagnostic| {
                        NativeStartupTimingObservation {
                            schema: "rz-pals-startup-timing-observation/1",
                            model_loading_elapsed_ns: diagnostic.model_loading_elapsed_ns,
                            probe_budget_ns: diagnostic.probe_budget_ns,
                            probe_elapsed_ns: elapsed,
                            drain_limit_ns: diagnostic.drain_limit_ns,
                            commands: diagnostic.commands,
                            command_observation_overflow: diagnostic.command_observation_overflow,
                        }
                    });
                    if let Ok(mut probe) = self.owner.startup_probe.lock()
                        && let Some(probe) = probe.as_mut()
                    {
                        probe.backend_stages = backend_stages;
                        probe.timing_observation = timing;
                    }
                }
                self.startup_diagnostic = None;
            }
            self.startup_clock = None;
            outcome
        }
        fn startup_command(
            &mut self,
            command: PalsNativeCommand,
            until: Instant,
        ) -> Result<PalsNativeResult, RoleError> {
            let kind = command.diagnostic_kind();
            let elapsed = self.startup_elapsed_ns();
            let input_observation = self.startup_input_observation(&command);
            // A diagnostic cap is not a command admission or completion rule.
            // Suppress capture on overflow without mutating a prior event or
            // replacing the actual command outcome.
            if self.startup_diagnostic.as_ref().is_some_and(|d| {
                d.commands.len()
                    >= if self.owner.execution.cuda_record_pages.is_some() {
                        8
                    } else {
                        7
                    }
            }) {
                let mut diagnostic = self.startup_diagnostic.take();
                if let Some(diagnostic) = diagnostic.as_mut() {
                    diagnostic.command_observation_overflow = true;
                    diagnostic.current_stage = kind;
                }
                let result = self.startup_command_inner(command, until);
                self.startup_diagnostic = diagnostic;
                return result;
            }
            if let Some(diagnostic) = self.startup_diagnostic.as_mut() {
                diagnostic.current_stage = kind;
                diagnostic.commands.push(NativeStartupCommandObservation {
                    command: kind,
                    admitted: false,
                    entered_elapsed_ns: elapsed,
                    returned_elapsed_ns: None,
                    physical_completion_confirmed: None,
                    return_error: None,
                    fence_observation: None,
                    input_observation,
                });
            }
            let result = self.startup_command_inner(command, until);
            let elapsed = self.startup_elapsed_ns();
            if let Some(observation) = self
                .startup_diagnostic
                .as_mut()
                .and_then(|d| d.commands.last_mut())
            {
                observation.returned_elapsed_ns = elapsed;
                observation.return_error = result.as_ref().err().map(NativeStartupErrorKind::from);
            }
            result
        }
        fn startup_input_observation(
            &self,
            command: &PalsNativeCommand,
        ) -> Option<NativeStartupInputObservation> {
            if self.owner.startup_probe_timeout_ms.load(Ordering::Acquire) == 0 {
                return None;
            }
            let input = match command {
                PalsNativeCommand::Evaluate(input)
                | PalsNativeCommand::EvaluateWithEvidence(input) => input,
                PalsNativeCommand::EvaluatePrivateWarm(input) => input.input(),
                #[cfg(feature = "experimental-io-binding")]
                PalsNativeCommand::EvaluateCudaWarm(input) => input.input(),
                _ => return None,
            };
            #[cfg(feature = "experimental-io-binding")]
            let backend_owner_id = self
                .owner
                .private_warm
                .as_ref()
                .and_then(|warm| warm.observer.as_ref())
                .map(|observer| observer.owner_id());
            #[cfg(not(feature = "experimental-io-binding"))]
            let backend_owner_id = None;
            Some(NativeStartupInputObservation {
                startup_request_ordinal: None,
                backend_owner_id,
                role: input.role,
                input_key: input.canonical_input_key(&self.owner.model_config).ok()?,
                logical_records: input.records.len(),
                retained_input_key: None,
                raw_decode_succeeded: None,
            })
        }
        fn startup_command_inner(
            &mut self,
            command: PalsNativeCommand,
            until: Instant,
        ) -> Result<PalsNativeResult, RoleError> {
            if self.owner.quarantined.load(Ordering::Acquire) {
                return Err(RoleError::PhysicalCompletionUnknown);
            }
            if Instant::now() >= until {
                if let Some(observation) = self
                    .startup_diagnostic
                    .as_mut()
                    .and_then(|d| d.commands.last_mut())
                {
                    observation.fence_observation = Some("logical_deadline_before_admission");
                }
                return Err(RoleError::Deadline);
            }
            if self.runtime.state().executions != 0
                || self.owner.in_flight.load(Ordering::Acquire) != 0
                || self.control_lease.is_some()
            {
                return Err(RoleError::Unavailable);
            }
            let physical_until = until
                .checked_add(self.drain_limit)
                .ok_or(RoleError::InvalidOutput)?;
            self.control_lease = Some(
                self.owner
                    .worker
                    .lock()
                    .map_err(|_| RoleError::Unavailable)?
                    .submit(command)
                    .map_err(model_error)?,
            );
            self.owner.in_flight.fetch_add(1, Ordering::AcqRel);
            if let Some(diagnostic) = self.startup_diagnostic.as_mut() {
                let ordinal = u8::try_from(
                    diagnostic
                        .commands
                        .iter()
                        .filter(|command| command.admitted && command.input_observation.is_some())
                        .count()
                        + 1,
                )
                .ok();
                if let Some(observation) = diagnostic.commands.last_mut() {
                    observation.admitted = true;
                    if let Some(input) = observation.input_observation.as_mut() {
                        input.startup_request_ordinal = ordinal;
                    }
                }
            }
            loop {
                let lease = self.control_lease.as_mut().expect("admitted startup lease");
                match lease.poll() {
                    PhysicalPoll::Ready(result) => {
                        self.owner.in_flight.fetch_sub(1, Ordering::AcqRel);
                        if let Some(observation) = self
                            .startup_diagnostic
                            .as_mut()
                            .and_then(|d| d.commands.last_mut())
                        {
                            observation.physical_completion_confirmed = Some(true);
                        }
                        // The retained immutable job gives the exact input for
                        // output validation without recomputing/duplicating it.
                        let checked = match result {
                            Ok(result) => {
                                let input = match lease.input() {
                                    PalsNativeCommand::Evaluate(input)
                                    | PalsNativeCommand::EvaluateWithEvidence(input) => Some(input),
                                    PalsNativeCommand::EvaluatePrivateWarm(input) => {
                                        Some(input.input())
                                    }
                                    #[cfg(feature = "experimental-io-binding")]
                                    PalsNativeCommand::EvaluateCudaWarm(input) => {
                                        Some(input.input())
                                    }
                                    _ => None,
                                };
                                if let (Some(input), PalsNativeResult::Evaluation(raw)) =
                                    (input, &result)
                                {
                                    let checked = raw
                                        .decode(input, &self.owner.model_config)
                                        .map_err(model_error);
                                    if let Some(observation) = self
                                        .startup_diagnostic
                                        .as_mut()
                                        .and_then(|diagnostic| diagnostic.commands.last_mut())
                                        .and_then(|command| command.input_observation.as_mut())
                                    {
                                        observation.retained_input_key = input
                                            .canonical_input_key(&self.owner.model_config)
                                            .ok();
                                        observation.raw_decode_succeeded = Some(checked.is_ok());
                                    }
                                    if checked.is_ok() {
                                        let mut probe = self
                                            .owner
                                            .startup_probe
                                            .lock()
                                            .map_err(|_| RoleError::Unavailable)?;
                                        let probe = probe.as_mut().ok_or(RoleError::Unavailable)?;
                                        if input.role == PalsRole::Critic {
                                            probe.completed_critic_calls += 1;
                                        } else {
                                            probe.completed_proposer_calls += 1;
                                        }
                                    }
                                    checked.map(|_| result)
                                } else {
                                    Ok(result)
                                }
                            }
                            Err(error) => {
                                self.owner.remember_failure(&error);
                                Err(model_error(error))
                            }
                        };
                        self.control_lease = None;
                        return checked;
                    }
                    PhysicalPoll::Quarantined | PhysicalPoll::Consumed => {
                        if let Ok(Some(error)) = lease.quarantine_cause() {
                            self.owner.remember_failure(&error);
                        }
                        self.owner.quarantined.store(true, Ordering::Release);
                        if let Some(observation) = self
                            .startup_diagnostic
                            .as_mut()
                            .and_then(|d| d.commands.last_mut())
                        {
                            observation.fence_observation = Some("physical_worker_quarantined");
                        }
                        return Err(RoleError::PhysicalCompletionUnknown);
                    }
                    PhysicalPoll::Pending => {}
                }
                if Instant::now() >= physical_until {
                    self.owner.quarantined.store(true, Ordering::Release);
                    if let Some(observation) = self
                        .startup_diagnostic
                        .as_mut()
                        .and_then(|d| d.commands.last_mut())
                    {
                        observation.fence_observation =
                            Some("pending_at_registered_physical_fence");
                    }
                    self.owner.remember_failure(&BackendError::new(
                        rz_eval::error::FailureKind::BackendFailure,
                        rz_eval::error::FailureStage::Backend,
                        "PALS startup physical completion remains unconfirmed at its registered fence",
                    ));
                    // Keep the lease and worker closure's session/input pins.
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        fn drain_deadline(&self, observed: Instant, query_until: Instant) -> Instant {
            let observed_end = observed.checked_add(self.drain_limit).unwrap_or(observed);
            query_until
                .checked_add(self.drain_limit)
                .map_or(observed_end, |end| end.min(observed_end))
        }
        fn ensure_new_game(
            &mut self,
            until: Instant,
            canceled: &AtomicBool,
        ) -> Result<(), RoleError> {
            if !self.pending_new_game {
                return Ok(());
            }
            if canceled.load(Ordering::Acquire) {
                return Err(RoleError::Canceled);
            }
            if Instant::now() >= until {
                return Err(RoleError::Deadline);
            }
            if self.runtime.state().executions != 0
                || self.owner.in_flight.load(Ordering::Acquire) != 0
            {
                self.unusable = true;
                self.owner.quarantined.store(true, Ordering::Release);
                return Err(RoleError::PhysicalCompletionUnknown);
            }
            if self.control_lease.is_some() {
                self.unusable = true;
                self.owner.quarantined.store(true, Ordering::Release);
                return Err(RoleError::PhysicalCompletionUnknown);
            }
            let target = self.game;
            self.control_lease = Some({
                let mut worker = self.owner.worker.lock().map_err(|_| {
                    RoleError::Backend("PALS cache-reset owner lock poisoned".into())
                })?;
                let _game_guard = self
                    .owner
                    .game_lifecycle
                    .lock()
                    .map_err(|_| RoleError::Unavailable)?;
                if canceled.load(Ordering::Acquire) {
                    return Err(RoleError::Canceled);
                }
                if Instant::now() >= until {
                    return Err(RoleError::Deadline);
                }
                if self.owner.finishing.load(Ordering::Acquire)
                    || self.owner.shutdown.load(Ordering::Acquire)
                {
                    return Err(RoleError::Unavailable);
                }
                let lease = match worker.submit(PalsNativeCommand::ResetTo(target)) {
                    Ok(lease) => lease,
                    Err(error) => {
                        self.owner.remember_failure(&error);
                        self.unusable = true;
                        return Err(self.owner.remember_game_reset_failure(model_error(error)));
                    }
                };
                self.owner.in_flight.fetch_add(1, Ordering::AcqRel);
                lease
            });
            let mut terminal = None;
            let mut drain_until = None;
            loop {
                let observed = Instant::now();
                if terminal.is_none() && canceled.load(Ordering::Acquire) {
                    terminal = Some(RoleError::Canceled);
                    self.owner.canceled.fetch_add(1, Ordering::AcqRel);
                    drain_until = Some(self.drain_deadline(observed, until));
                } else if terminal.is_none() && observed >= until {
                    terminal = Some(RoleError::Deadline);
                    self.owner.expired.fetch_add(1, Ordering::AcqRel);
                    drain_until = Some(self.drain_deadline(observed, until));
                }
                let outcome = self
                    .control_lease
                    .as_mut()
                    .expect("admitted reset lease")
                    .poll();
                match outcome {
                    PhysicalPoll::Ready(result) => {
                        self.owner.in_flight.fetch_sub(1, Ordering::AcqRel);
                        self.control_lease = None;
                        match result {
                            Ok(PalsNativeResult::ResetTo { game_generation })
                                if game_generation == target =>
                            {
                                if let Err(error) = self.owner.accept_game_target(game_generation) {
                                    self.unusable = true;
                                    return Err(self.owner.remember_game_reset_failure(error));
                                }
                                self.pending_new_game = false;
                                if terminal.is_none() && canceled.load(Ordering::Acquire) {
                                    self.owner.canceled.fetch_add(1, Ordering::AcqRel);
                                    terminal = Some(RoleError::Canceled);
                                } else if terminal.is_none() && Instant::now() >= until {
                                    self.owner.expired.fetch_add(1, Ordering::AcqRel);
                                    terminal = Some(RoleError::Deadline);
                                }
                                return terminal.map_or(Ok(()), Err);
                            }
                            Ok(_) => {
                                self.unusable = true;
                                return Err(self.owner.remember_game_reset_failure(
                                    RoleError::Backend(
                                        "PALS reset lease received a different response".into(),
                                    ),
                                ));
                            }
                            Err(error) => {
                                self.owner.remember_failure(&error);
                                self.unusable = true;
                                return Err(self
                                    .owner
                                    .remember_game_reset_failure(model_error(error)));
                            }
                        }
                    }
                    PhysicalPoll::Quarantined | PhysicalPoll::Consumed => {
                        if let Ok(Some(error)) = self
                            .control_lease
                            .as_ref()
                            .expect("retained reset lease")
                            .quarantine_cause()
                        {
                            self.owner.remember_failure(&error);
                        }
                        self.unusable = true;
                        self.owner.quarantined.store(true, Ordering::Release);
                        // Retain this control lease as well as the worker's
                        // closure/job pins. No clear acknowledgement is invented.
                        return Err(RoleError::PhysicalCompletionUnknown);
                    }
                    PhysicalPoll::Pending => {}
                }
                if drain_until.is_some_and(|end| Instant::now() >= end) {
                    self.unusable = true;
                    self.owner.quarantined.store(true, Ordering::Release);
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        fn run(
            &mut self,
            input: PalsModelInput,
            position: &Position,
            until: Instant,
            canceled: &std::sync::atomic::AtomicBool,
            prepared_context: NativePreparedContext<'_>,
            logical: Option<&RoleLogicalContext>,
        ) -> Result<RoleOutput, RoleError> {
            self.run_with_ticket(
                input,
                position,
                until,
                canceled,
                prepared_context,
                logical,
                None,
            )
        }
        // The synchronous compatibility path and ticketed path share one owner;
        // keep optional ticket authority distinct from model input and context.
        #[allow(clippy::too_many_arguments)]
        pub(super) fn run_with_ticket(
            &mut self,
            input: PalsModelInput,
            position: &Position,
            until: Instant,
            canceled: &AtomicBool,
            prepared_context: NativePreparedContext<'_>,
            logical: Option<&RoleLogicalContext>,
            ticket: Option<&rz_contracts::pals_adaptive::RoleTicketEnvelope>,
        ) -> Result<RoleOutput, RoleError> {
            if self
                .owner
                .private_warm
                .as_ref()
                .is_some_and(|warm| warm.state.is_quarantined())
            {
                self.owner.quarantined.store(true, Ordering::Release);
            }
            if self.owner.quarantined.load(Ordering::Acquire) {
                return Err(RoleError::PhysicalCompletionUnknown);
            }
            if self.unusable
                || !self.owner.startup_ready.load(Ordering::Acquire)
                || self.owner.finishing.load(Ordering::Acquire)
                || self.owner.shutdown.load(Ordering::Acquire)
            {
                return Err(RoleError::Unavailable);
            }
            if canceled.load(Ordering::Acquire) {
                return Err(RoleError::Canceled);
            }
            if Instant::now() >= until {
                return Err(RoleError::Deadline);
            }
            self.ensure_new_game(until, canceled)?;
            if let Some(ticket) = ticket {
                if self.owner.private_warm.is_some() {
                    return Err(RoleError::Backend(
                        "ticketed Warm capability is not installed; Fresh is an explicit profile"
                            .into(),
                    ));
                }
                if ticket.id.epoch != self.epoch
                    || ticket.authority.epoch != self.epoch
                    || ticket.id.sequence <= self.sequence
                    || ticket.authority.game.0 != self.game
                    || ticket.original_deadline.clock != self.clock.domain()
                    || self.ticketed_delivered.len() >= 16
                    || self.delivered_request.is_some()
                {
                    return Err(RoleError::InvalidOutput);
                }
                self.sequence = ticket.id.sequence;
            } else {
                if self.delivered_request.is_some() {
                    self.close_unconsumed(NativeRoleRejection::SupersededUnconsumed)?;
                }
                self.sequence = self
                    .sequence
                    .checked_add(1)
                    .ok_or_else(|| RoleError::Backend("PALS request sequence exhausted".into()))?;
            }
            self.owner
                .request_high_water
                .store(self.sequence, Ordering::Release);
            let mut scope = self.scope.get().ok_or(RoleError::Unavailable)?;
            if let Some(ticket) = ticket {
                scope.authority = ticket.authority;
            } else {
                scope.authority.game = GameGeneration(self.game);
                scope.authority.root = RootGeneration(self.sequence);
            }
            self.scope.update(scope).map_err(model_error)?;
            let now = self.clock.now();
            let remaining = u64::try_from(
                until.saturating_duration_since(Instant::now()).as_nanos(),
            )
            .map_err(|_| RoleError::Backend("PALS deadline outside finite clock domain".into()))?;
            let deadline = ticket.map_or(
                Deadline {
                    clock: self.clock.domain(),
                    at: MonotonicTick(now.0.checked_add(remaining).ok_or_else(|| {
                        RoleError::Backend("PALS monotonic deadline exhausted".into())
                    })?),
                },
                |ticket| ticket.original_deadline,
            );
            let state_owner = OwnerId(next(&RULE_OWNERS)?);
            let exact = ContractPosition::new(state_owner, position.clone())
                .export()
                .map_err(model_error)?;
            let state_identity = exact.snapshot().identity();
            let cancel = CancelToken::new();
            let id = RequestId::new(self.epoch, self.sequence);
            let mut warm_admission = None;
            let input = if let Some(warm) = &self.owner.private_warm {
                let logical = logical.ok_or_else(|| {
                    RoleError::Backend("Native warm requires checked logical query context".into())
                })?;
                if logical.game_generation != self.game {
                    return Err(RoleError::Backend(
                        "Native warm game generation differs from Search".into(),
                    ));
                }
                let prepared = warm.state.prepare(
                    id,
                    input,
                    position.snapshot(),
                    logical,
                    until,
                    cancel.clone(),
                    canceled,
                )?;
                // All full storage is allocated/budgeted before bank.begin.
                let mut payload = warm
                    .capability
                    .payload(prepared.input.as_ref().clone())
                    .map_err(model_error)?;
                let mut backup = warm
                    .capability
                    .payload(prepared.input.as_ref().clone())
                    .map_err(model_error)?;
                if payload.owned_host_bytes() > warm.capability.payload_bytes()
                    || backup.owned_host_bytes() > warm.capability.payload_bytes()
                {
                    return Err(RoleError::Backend(
                        "Native warm full payload reservation exhausted".into(),
                    ));
                }
                if canceled.load(Ordering::Acquire) {
                    cancel.cancel();
                    return Err(RoleError::Canceled);
                }
                let mut physical = warm.state.begin(prepared)?;
                let bound = if let Some(seed) = physical.seed() {
                    payload
                        .bind_lease(&warm.capability, seed)
                        .and_then(|_| backup.bind_lease(&warm.capability, seed))
                } else {
                    Ok(())
                };
                backup.retain_backup(&mut physical);
                if let Err(error) = bound {
                    warm.state.observe_context_rejection();
                    warm.state
                        .retain_unknown(physical, "prelaunch_payload_bind_failed");
                    self.owner.quarantined.store(true, Ordering::Release);
                    self.unusable = true;
                    return Err(model_error(error));
                }
                let prepared_input = physical.prepared.input.clone();
                warm_admission = Some(WarmAdmission {
                    state: warm.state.clone(),
                    payload: Some(payload),
                    physical: Some(physical),
                });
                prepared_input
            } else {
                Arc::new(input)
            };
            let role = if input.role == PalsRole::Critic {
                Role::Critic
            } else {
                Role::Proposer
            };
            let packed_candidates = input
                .candidates
                .iter()
                .map(|c| c.packed().map_err(model_error))
                .collect::<Result<Vec<_>, _>>()?;
            let key = RepresentationKey {
                input: Digest(if let Some(admission) = &warm_admission {
                    match admission
                        .physical
                        .as_ref()
                        .ok_or(RoleError::Unavailable)?
                        .invocation()?
                    {
                        rz_eval::pals_private::PrivateInvocation::Fresh { input_key } => input_key,
                        rz_eval::pals_private::PrivateInvocation::ApproxWarmV1 {
                            invocation_key,
                            ..
                        }
                        | rz_eval::pals_private::PrivateInvocation::ApproxCudaWarmV2 {
                            invocation_key,
                            ..
                        } => invocation_key,
                    }
                } else {
                    input
                        .canonical_input_key(&self.owner.model_config)
                        .map_err(model_error)?
                }),
                history: Digest(input.history_digest),
                candidates: Digest({
                    let mut hash = Sha256::new();
                    for candidate in &packed_candidates {
                        hash.update(candidate.to_le_bytes());
                    }
                    hash.finalize().into()
                }),
                mask_positions: Digest(
                    Sha256::digest(
                        [
                            input.records.len() as u64,
                            input.candidates.len() as u64,
                            input.divergence_features.len() as u64,
                        ]
                        .into_iter()
                        .flat_map(u64::to_le_bytes)
                        .collect::<Vec<_>>(),
                    )
                    .into(),
                ),
                model: self.model,
                encoding: self.encoding,
                precision: PrecisionProfile::Fp32,
                frozen_epoch: DEPLOYMENT_FROZEN_EPOCH,
                record_revision: input.situation_revision,
                role,
            };
            if ticket.is_some_and(|ticket| ticket.representation != key || ticket.role != role) {
                return Err(RoleError::InvalidOutput);
            }
            let candidates: Vec<_> = packed_candidates
                .into_iter()
                .map(|bits| Move16::try_from_bits(bits).map_err(model_error))
                .collect::<Result<_, _>>()?;
            let public_records: Vec<_> = input
                .records
                .iter()
                .map(|r| {
                    let mut h = Sha256::new();
                    h.update(r.record_id.to_le_bytes());
                    h.update(r.revision.to_le_bytes());
                    h.update([u8::from(r.critical)]);
                    for value in r.features {
                        h.update(value.to_bits().to_le_bytes());
                    }
                    Digest(h.finalize().into())
                })
                .collect();
            let divergence_count = u16::try_from(input.divergence_features.len())
                .map_err(|_| RoleError::InvalidOutput)?;
            let situation = SituationHandle {
                slot: 0,
                generation: self.sequence,
            };
            let request = Arc::new(RoleRequest {
                revision: PALS_CONTRACT_REVISION,
                id,
                authority: scope.authority,
                situation,
                state_identity,
                state: input,
                key,
                role,
                mode: ExecutionMode::Deployment,
                candidates: candidates.into(),
                public_records: public_records.into(),
                max_records: 128,
                recurrent_steps: 2,
                divergence_count,
                deadline,
                cancel: cancel.clone(),
                resources: ByteBudget {
                    host: REQUEST_BYTES,
                    device: self.owner.execution.request_device_reservation(),
                    pinned: self.owner.execution.pinned_request_bytes,
                },
            });
            if self.owner.followup_evidence.load(Ordering::Acquire) {
                *self
                    .owner
                    .prepared_perspective
                    .lock()
                    .map_err(|_| RoleError::Unavailable)? = Some((
                    id,
                    if position.side_to_move() == Color::White {
                        rz_contracts::Color::White
                    } else {
                        rz_contracts::Color::Black
                    },
                ));
            }
            let representation = SharedRepresentationScope::new(RepresentationScope {
                situation,
                state: state_identity,
                key,
            });
            if let Some(admission) = &warm_admission {
                let physical = admission.physical.as_ref().ok_or(RoleError::Unavailable)?;
                let payload = admission.payload.as_ref().ok_or(RoleError::Unavailable)?;
                self.owner.observe("prepared_private_warm", |observer| {
                    let bank = admission.state.snapshot()?;
                    let prepared = NativePrivateWarmPrepared {
                        invocation: physical.invocation()?,
                        initial_latent_bits: payload.initial_latent_bits(),
                        seed_provenance: physical.seed().and_then(|seed| seed.seed_provenance()),
                        bank: &bank,
                    };
                    observer.prepared_private_warm(
                        id,
                        request.state.as_ref(),
                        prepared_context,
                        logical,
                        prepared,
                    )
                })?;
            } else {
                self.owner.observe("prepared", |observer| {
                    observer.prepared_with_logical_context(
                        id,
                        request.state.as_ref(),
                        prepared_context,
                        logical,
                    )
                })?;
            }
            if let Some(admission) = warm_admission {
                let mut slot = self
                    .owner
                    .warm_admission
                    .lock()
                    .map_err(|_| RoleError::Unavailable)?;
                if slot.is_some() {
                    return Err(RoleError::Backend(
                        "Native warm handoff slot occupied".into(),
                    ));
                }
                *slot = Some(admission);
            }
            if let Err(error) = self
                .runtime
                .submit(RuntimeRoleRequest::new(request, representation))
            {
                if self.owner.private_warm.is_some() {
                    if let Ok(mut slot) = self.owner.warm_admission.lock() {
                        slot.take();
                    }
                    self.owner.quarantined.store(true, Ordering::Release);
                    self.unusable = true;
                }
                let _ = self.owner.observe("admission", |observer| {
                    observer.rejected(id, NativeRoleRejection::Admission)
                });
                return Err(runtime_admission_error(error));
            }
            let mut outcome = None;
            let mut drain_until = None;
            loop {
                if self
                    .owner
                    .private_warm
                    .as_ref()
                    .is_some_and(|warm| warm.state.is_quarantined())
                {
                    self.owner.quarantined.store(true, Ordering::Release);
                }
                if canceled.load(Ordering::Acquire) && !cancel.is_canceled() {
                    drain_until = Some(self.drain_deadline(Instant::now(), until));
                    cancel.cancel();
                    self.owner.canceled.fetch_add(1, Ordering::AcqRel);
                    self.runtime.cancel(id).map_err(model_error)?;
                }
                if let Some(terminal) = self.runtime.poll() {
                    outcome = Some(match terminal {
                        PalsTerminal::Completed(output) => Ok(output.output),
                        PalsTerminal::Canceled(_) => Err(RoleError::Canceled),
                        PalsTerminal::Expired(_) => {
                            self.owner.expired.fetch_add(1, Ordering::AcqRel);
                            Err(RoleError::Deadline)
                        }
                        PalsTerminal::Stale(_) => Err(RoleError::Backend(
                            "PALS native result has stale authority".into(),
                        )),
                        PalsTerminal::Failed { error, .. } => Err(model_error(error)),
                    });
                    drain_until.get_or_insert(self.drain_deadline(Instant::now(), until));
                }
                if self.owner.quarantined.load(Ordering::Acquire) {
                    self.unusable = true;
                    cancel.cancel();
                    self.owner
                        .observe_role_unknown(id, NativeRoleCompletionUnknown::Quarantined);
                    let _ = self.runtime.cancel(id);
                    self.runtime.pump();
                    let _ = self.owner.observe("physical_unknown", |observer| {
                        observer.rejected(id, NativeRoleRejection::PhysicalCompletionUnknown)
                    });
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                if outcome.is_some() && self.runtime.state().executions == 0 {
                    let result = outcome.take().expect("terminal outcome");
                    if canceled.load(Ordering::Acquire) {
                        if let Some(warm) = &self.owner.private_warm {
                            warm.state.reject_output(Some(id));
                        }
                        let _ = self.owner.observe("canceled", |observer| {
                            observer.rejected(id, NativeRoleRejection::Canceled)
                        });
                        return Err(RoleError::Canceled);
                    }
                    if Instant::now() >= until {
                        if let Some(warm) = &self.owner.private_warm {
                            warm.state.reject_output(Some(id));
                        }
                        let _ = self.owner.observe("deadline", |observer| {
                            observer.rejected(id, NativeRoleRejection::Deadline)
                        });
                        return Err(RoleError::Deadline);
                    }
                    if result.is_ok() {
                        self.owner
                            .observe("delivered", |observer| observer.delivered(id))?;
                        if canceled.load(Ordering::Acquire) || Instant::now() >= until {
                            if let Some(warm) = &self.owner.private_warm {
                                warm.state.reject_output(Some(id));
                            }
                            let (reason, error) = if canceled.load(Ordering::Acquire) {
                                (NativeRoleRejection::Canceled, RoleError::Canceled)
                            } else {
                                (NativeRoleRejection::Deadline, RoleError::Deadline)
                            };
                            let _ = self.owner.observe("delivery_expired", |observer| {
                                observer.rejected(id, reason)
                            });
                            return Err(error);
                        }
                        self.owner.delivered.fetch_add(1, Ordering::AcqRel);
                        self.delivered_request = Some(id);
                    } else {
                        if let Some(warm) = &self.owner.private_warm {
                            warm.state.reject_output(Some(id));
                        }
                        let reason = match &result {
                            Err(RoleError::Canceled) => NativeRoleRejection::Canceled,
                            Err(RoleError::Deadline) => NativeRoleRejection::Deadline,
                            Err(RoleError::InvalidOutput) => NativeRoleRejection::InvalidOutput,
                            Err(RoleError::PhysicalCompletionUnknown) => {
                                NativeRoleRejection::PhysicalCompletionUnknown
                            }
                            _ => NativeRoleRejection::Backend,
                        };
                        let _ = self
                            .owner
                            .observe("rejected", |observer| observer.rejected(id, reason));
                    }
                    return result;
                }
                if drain_until.is_some_and(|deadline| Instant::now() >= deadline) {
                    self.unusable = true;
                    self.owner.quarantined.store(true, Ordering::Release);
                    self.owner
                        .observe_role_unknown(id, NativeRoleCompletionUnknown::DrainExpired);
                    let now = self.clock.now();
                    let _ = self.runtime.begin_shutdown(Deadline {
                        clock: self.clock.domain(),
                        at: now,
                    });
                    let _ = self.owner.observe("physical_unknown", |observer| {
                        observer.rejected(id, NativeRoleRejection::PhysicalCompletionUnknown)
                    });
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        fn evaluate(
            &mut self,
            query: RoleQuery<'_>,
            kind: NativeQueryKind,
            logical: Option<&RoleLogicalContext>,
        ) -> Result<RoleEvaluation, RoleError> {
            let input = if let Some(view) = &self.selected_review {
                prepare_selected_role_input_for_config(
                    &query,
                    kind,
                    self.model_epoch,
                    &self.owner.model_config,
                    view,
                )?
            } else {
                prepare_role_input_for_config(
                    &query,
                    kind,
                    self.model_epoch,
                    &self.owner.model_config,
                )?
            };
            let output = self.run(
                input,
                query.position,
                query.deadline,
                query.cancel,
                NativePreparedContext::Role {
                    query: &query,
                    kind,
                },
                logical,
            )?;
            match output.payload {
                RolePayload::Proposal {
                    candidate_logits,
                    wdl,
                }
                | RolePayload::Counterexample {
                    candidate_logits,
                    wdl,
                    ..
                } => Ok(RoleEvaluation {
                    logits: candidate_logits,
                    wdl: wdl.probabilities(),
                }),
                _ => Err(RoleError::InvalidOutput),
            }
        }
    }
    impl RoleModel for NativeRoleModel {
        fn supports_selected_review_view(&self) -> bool {
            true
        }
        fn bind_selected_review_view(
            &mut self,
            view: Arc<rz_search::pals::review_context::ReviewView>,
            context: &RoleLogicalContext,
        ) -> Result<(), RoleError> {
            if context.purpose == RoleQueryPurpose::TaskPolicy {
                return Err(RoleError::Unsupported(
                    "V1/V2 native adapter does not implement TaskPolicy",
                ));
            }
            if self.owner.in_flight.load(Ordering::Acquire) != 0
                || self.delivered_request.is_some()
                || view.generation() != context.search_generation
                || view.query().purpose != context.purpose
                || view.query().prefix != context.prefix
            {
                return Err(RoleError::InvalidOutput);
            }
            self.selected_review = Some(view);
            Ok(())
        }
        fn identity(&self) -> &str {
            &self.identity
        }
        fn value_identity(&self) -> Option<&ModelValueIdentity> {
            Some(&self.value_identity)
        }
        fn evaluate_value(&mut self, query: RoleQuery<'_>) -> Result<ModelValueOutput, RoleError> {
            self.last_value_execution = None;
            // Frontier evaluation uses the existing proposer forward/shared WDL
            // head. Its candidate-policy output is unused. This is a contextual
            // model estimate, not a candidate value or foreign-CP calibration.
            let state = query.position.position_identity();
            let perspective = query.position.side_to_move();
            let input = if let Some(view) = &self.selected_review {
                prepare_selected_role_input_for_config(
                    &query,
                    NativeQueryKind::Propose,
                    self.model_epoch,
                    &self.owner.model_config,
                    view,
                )?
            } else {
                prepare_role_input_for_config(
                    &query,
                    NativeQueryKind::Propose,
                    self.model_epoch,
                    &self.owner.model_config,
                )?
            };
            let output = self.run(
                input,
                query.position,
                query.deadline,
                query.cancel,
                NativePreparedContext::Role {
                    query: &query,
                    kind: NativeQueryKind::Propose,
                },
                None,
            )?;
            // This key was formed from the exact request before physical Run.
            // Do not reconstruct it after Run: remaining-time features change.
            let input_sha256 = output.key.input.0;
            let RolePayload::Proposal { wdl, .. } = output.payload else {
                return Err(RoleError::InvalidOutput);
            };
            Ok(ModelValueOutput {
                identity: self.value_identity.clone(),
                input_sha256,
                state,
                perspective,
                wdl: wdl.probabilities(),
            })
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            self.evaluate(query, NativeQueryKind::Propose, None)
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            self.evaluate(query, NativeQueryKind::Reply, None)
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            self.evaluate(query, NativeQueryKind::Repair, None)
        }
        fn recheck_prepared(&mut self, event: RecheckPrepared<'_>) -> Result<(), RoleError> {
            self.owner.observe("recheck_prepared", |observer| {
                observer.recheck_prepared(event)
            })
        }
        fn recheck_finished(&mut self, event: RecheckFinished<'_>) -> Result<(), RoleError> {
            // This is a prepaid logical close callback. Deadline/cancel must not
            // suppress it, and it never submits or infers a physical native Run.
            self.owner.observe("recheck_finished", |observer| {
                observer.recheck_finished(event)
            })
        }
        fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            let input = if let Some(view) = &self.selected_review {
                prepare_selected_divergence_input_for_config(
                    &query,
                    self.model_epoch,
                    &self.owner.model_config,
                    view,
                )?
            } else {
                prepare_divergence_input_for_config(
                    &query,
                    self.model_epoch,
                    &self.owner.model_config,
                )?
            };
            let output = self.run(
                input,
                query.root,
                query.deadline,
                query.cancel,
                NativePreparedContext::Divergence { query: &query },
                None,
            )?;
            match output.payload {
                RolePayload::Counterexample {
                    divergence_logits, ..
                } if divergence_logits.len() == query.candidates.len() => Ok(divergence_logits),
                _ => Err(RoleError::InvalidOutput),
            }
        }
        fn new_game(&mut self) {
            if self.owner.finishing.load(Ordering::Acquire)
                || self.owner.shutdown.load(Ordering::Acquire)
            {
                self.unusable = true;
                return;
            }
            let _ = self.close_unconsumed(NativeRoleRejection::GameReset);
            match self.game.checked_add(1) {
                Some(value) => {
                    if self.owner.request_game_target(value).is_err() {
                        self.unusable = true;
                        return;
                    }
                    self.game = value;
                    self.pending_new_game = true;
                }
                None => self.unusable = true,
            }
            // Clear on the same physical worker before the next NN admission;
            // epoch and execution/request high-water marks never reset.
        }
        fn new_game_with_generation(&mut self, generation: Option<u64>) {
            if self.owner.finishing.load(Ordering::Acquire)
                || self.owner.shutdown.load(Ordering::Acquire)
            {
                self.unusable = true;
                return;
            }
            if self.owner.private_warm.is_none() {
                self.new_game();
                return;
            }
            let _ = self.close_unconsumed(NativeRoleRejection::GameReset);
            match generation {
                Some(game) if game > self.game => {
                    if self.owner.request_game_target(game).is_err() {
                        self.unusable = true;
                        return;
                    }
                    self.game = game;
                    self.pending_new_game = true;
                }
                _ => self.unusable = true,
            }
        }
        fn propose_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<RoleEvaluation, RoleError> {
            check_context(&query, context, RoleQueryPurpose::ProposePolicy)?;
            self.evaluate(query, NativeQueryKind::Propose, Some(context))
        }
        fn reply_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<RoleEvaluation, RoleError> {
            check_context(&query, context, RoleQueryPurpose::ReplyPolicy)?;
            self.evaluate(query, NativeQueryKind::Reply, Some(context))
        }
        fn repair_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<RoleEvaluation, RoleError> {
            check_context(&query, context, RoleQueryPurpose::RepairPolicy)?;
            self.evaluate(query, NativeQueryKind::Repair, Some(context))
        }
        fn divergences_with_context(
            &mut self,
            query: DivergenceQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<Vec<f32>, RoleError> {
            if context.purpose != RoleQueryPurpose::DivergencePolicy
                || context.public_revision != query.revision
            {
                return Err(RoleError::Backend(
                    "Native divergence logical context mismatch".into(),
                ));
            }
            let input = if let Some(view) = &self.selected_review {
                prepare_selected_divergence_input_for_config(
                    &query,
                    self.model_epoch,
                    &self.owner.model_config,
                    view,
                )?
            } else {
                prepare_divergence_input_for_config(
                    &query,
                    self.model_epoch,
                    &self.owner.model_config,
                )?
            };
            let output = self.run(
                input,
                query.root,
                query.deadline,
                query.cancel,
                NativePreparedContext::Divergence { query: &query },
                Some(context),
            )?;
            match output.payload {
                RolePayload::Counterexample {
                    divergence_logits, ..
                } if divergence_logits.len() == query.candidates.len() => Ok(divergence_logits),
                _ => Err(RoleError::InvalidOutput),
            }
        }
        fn evaluate_value_with_context(
            &mut self,
            query: RoleQuery<'_>,
            context: &RoleLogicalContext,
        ) -> Result<ModelValueOutput, RoleError> {
            self.last_value_execution = None;
            check_context(&query, context, RoleQueryPurpose::ValueFresh)?;
            let expected_state = query.position.snapshot();
            if self.owner.private_warm.is_none() {
                // Context is copied into evidence only. Preserve the existing
                // Fresh input, graph and logical-free preparation path.
                let output = self.evaluate_value(query)?;
                self.capture_value_execution(context, &output, &expected_state);
                return Ok(output);
            }
            let state = query.position.position_identity();
            let perspective = query.position.side_to_move();
            let input = if let Some(view) = &self.selected_review {
                prepare_selected_role_input_for_config(
                    &query,
                    NativeQueryKind::Propose,
                    self.model_epoch,
                    &self.owner.model_config,
                    view,
                )?
            } else {
                prepare_role_input_for_config(
                    &query,
                    NativeQueryKind::Propose,
                    self.model_epoch,
                    &self.owner.model_config,
                )?
            };
            let output = self.run(
                input,
                query.position,
                query.deadline,
                query.cancel,
                NativePreparedContext::Role {
                    query: &query,
                    kind: NativeQueryKind::Propose,
                },
                Some(context),
            )?;
            let RolePayload::Proposal { wdl, .. } = output.payload else {
                return Err(RoleError::InvalidOutput);
            };
            let output = ModelValueOutput {
                identity: self.value_identity.clone(),
                input_sha256: output.key.input.0,
                state,
                perspective,
                wdl: wdl.probabilities(),
            };
            self.capture_value_execution(context, &output, &expected_state);
            Ok(output)
        }
        fn last_value_execution_evidence(&self) -> Option<RoleValueExecutionEvidence> {
            self.last_value_execution
        }
        fn preflight_new_game(
            &mut self,
            deadline: Instant,
            cancel: &AtomicBool,
        ) -> Result<(), RoleError> {
            if cancel.load(Ordering::Acquire) {
                return Err(RoleError::Canceled);
            }
            if Instant::now() >= deadline {
                return Err(RoleError::Deadline);
            }
            if self.unusable || self.owner.quarantined.load(Ordering::Acquire) {
                return Err(RoleError::PhysicalCompletionUnknown);
            }
            if self.owner.finishing.load(Ordering::Acquire)
                || self.owner.shutdown.load(Ordering::Acquire)
            {
                return Err(RoleError::Unavailable);
            }
            if self.pending_new_game || self.owner.pending_game_target.load(Ordering::Acquire) != 0
            {
                // An acknowledged ResetTo is required before another game
                // transition. Preflight never changes the requested target.
                return Err(RoleError::Backend(
                    "Native new-game preflight has an unacknowledged game target".into(),
                ));
            }
            // A retained unknown execution/control lease cannot authorize an
            // old Store replacement. This query schedules no Reset or NN.
            if self.runtime.state().executions != 0
                || self.owner.in_flight.load(Ordering::Acquire) != 0
                || self.control_lease.is_some()
                || self
                    .owner
                    .warm_admission
                    .try_lock()
                    .map_err(|_| RoleError::Unavailable)?
                    .is_some()
                || self
                    .owner
                    .role_execution
                    .try_lock()
                    .map_err(|_| RoleError::Unavailable)?
                    .as_ref()
                    .is_some_and(|execution| {
                        !execution.ready_observed || execution.unknown_observed
                    })
            {
                return Err(RoleError::PhysicalCompletionUnknown);
            }
            if let Some(error) = self
                .owner
                .game_reset_failure
                .try_lock()
                .map_err(|_| RoleError::Unavailable)?
                .as_ref()
            {
                return Err(error.clone());
            }
            if let Some(warm) = &self.owner.private_warm {
                let snapshot = warm.state.snapshot()?;
                if snapshot.quarantined
                    || snapshot.pending_acceptance
                    || snapshot.retained_prepared_input
                    || snapshot.active_lease.is_some()
                    || snapshot.pinned_entries != 0
                    || snapshot.game_generation != self.game
                {
                    return Err(RoleError::PhysicalCompletionUnknown);
                }
                if snapshot.admission_closed {
                    return Err(RoleError::Unavailable);
                }
            }
            if cancel.load(Ordering::Acquire) {
                return Err(RoleError::Canceled);
            }
            if Instant::now() >= deadline {
                return Err(RoleError::Deadline);
            }
            Ok(())
        }
        fn accepted_output_checked(
            &mut self,
            acceptance: RoleAcceptance<'_>,
        ) -> Result<(), RoleError> {
            acceptance.check_control()?;
            let Some(warm) = &self.owner.private_warm else {
                let id = self.delivered_request;
                self.accepted_output();
                if let Some(id) = id {
                    self.owner.observe("accepted_context", |observer| {
                        observer.accepted_context(id, &acceptance)
                    })?;
                }
                return Ok(());
            };
            let id = self.delivered_request.ok_or_else(|| {
                RoleError::Backend("Native warm has no delivered request to accept".into())
            })?;
            if let Err(error) = warm.state.accept(id, &acceptance) {
                warm.state.reject_output(Some(id));
                self.delivered_request = None;
                return Err(error);
            }
            self.delivered_request = None;
            self.owner.consumed.fetch_add(1, Ordering::AcqRel);
            self.owner
                .observe("accepted", |observer| observer.accepted(id))?;
            self.owner.observe("accepted_context", |observer| {
                observer.accepted_context(id, &acceptance)
            })?;
            Ok(())
        }
        fn accepted_output(&mut self) {
            if self.owner.private_warm.is_some() {
                let _ = self.close_unconsumed(NativeRoleRejection::SearchFailedUnconsumed);
                // Unchecked legacy acknowledgement never commits a warm seed.
                return;
            }
            // The search owner calls this only after its late/cancel/shape gate;
            // delivering a result or clearing a cache never creates consumption.
            if let Some(id) = self.delivered_request.take() {
                self.owner.consumed.fetch_add(1, Ordering::AcqRel);
                if self
                    .owner
                    .observe("accepted", |observer| observer.accepted(id))
                    .is_err()
                {
                    self.unusable = true;
                }
            }
            if self.runtime.state().executions == 0
                && !self.owner.quarantined.load(Ordering::Acquire)
            {
                self.selected_review = None;
            }
        }
        fn finish_search(&mut self, reason: RoleSearchClosure) {
            let reason = match reason {
                RoleSearchClosure::Completed => NativeRoleRejection::SearchFinishedUnconsumed,
                RoleSearchClosure::Canceled => NativeRoleRejection::Canceled,
                RoleSearchClosure::Deadline => NativeRoleRejection::Deadline,
                RoleSearchClosure::Failed => NativeRoleRejection::SearchFailedUnconsumed,
                RoleSearchClosure::PhysicalCompletionUnknown => {
                    NativeRoleRejection::PhysicalCompletionUnknown
                }
            };
            let _ = self.close_unconsumed(reason);
        }
    }

    impl Drop for NativeRoleModel {
        fn drop(&mut self) {
            if self.owner.quarantined.load(Ordering::Acquire)
                || self.owner.in_flight.load(Ordering::Acquire) != 0
                || self.runtime.state().executions != 0
            {
                if let Some(view) = self.selected_review.take() {
                    // Dropping a model is no proof of a physical fence. Keep
                    // the one bounded synchronous selected-view pin on unknown.
                    std::mem::forget(view);
                }
            }
        }
    }

    #[cfg(test)]
    mod lifecycle_tests {
        use super::*;
        use rz_eval::error::{FailureKind, FailureStage};
        use rz_eval::pals_model::PalsRawOutput;
        use rz_eval::worker::PhysicalRun;

        fn unexpected_cuda_placement() -> PhysicalRun<Result<PalsNativeResult, BackendError>> {
            PhysicalRun::Complete(Err(BackendError::new(
                FailureKind::BackendUnavailable,
                FailureStage::Admission,
                "CPU lifecycle fixture has no CUDA placement witness",
            )))
        }

        // Pure synthetic metadata and the in-process worker seam only. This
        // fixture does not mint a checked body, loaded Session, CUDA placement,
        // device backing, physical GPU fence or measured allocator peak.
        #[cfg(feature = "experimental-io-binding")]
        fn resident_boundary_fixture() -> (
            NativeExecutionReceipt,
            rz_eval::pals_onnx::CudaRecordPageSnapshot,
        ) {
            let selection = NativeCudaRecordPagesDeclaration {
                schema: "rz-pals-native-cuda-record-pages-selection/1",
                mode: "registered-packing-v1",
                process_epoch: 7,
                installation_game_generation: 0,
                frozen_epoch: DEPLOYMENT_FROZEN_EPOCH,
                model_manifest_sha256: [3; 32],
                model_epoch: [4; 32],
                public_graph_sha256: [5; 32],
                encoding_sha256: [6; 32],
                runtime_sha256: [2; 32],
                runtime_identity_scope: "closed_cuda_library_bundle",
                device_id: 0,
                packing_graph_sha256: [8; 32],
                packing_graph_bytes: 256,
                packing_manifest_sha256: [9; 32],
                packing_manifest_bytes: 128,
                known_retained_artifact_host_bytes: 1024,
                implementation_sha256: [10; 32],
                initialized_scope: "initialized_provider_count",
                initialized_provider_count: 275,
                initialized_log_sha256: [11; 32],
                initialized_log_digest_domain: "rz-pals-fixed-packing-initialized-provider-count/1",
                model_native_sessions: 2,
                packing_native_sessions: 1,
                resource_accounting_scope: "combined_resident_owner_invocation_including_sessions_and_backing",
                declared_resources: NativeCudaRecordPagesResourceDeclaration {
                    max_blocks: 2,
                    max_bank_whole_payload_bytes: 8192,
                    max_registry_entries: 4,
                    max_container_host_bytes: 4096,
                    max_invocation_host_bytes: 16384,
                    max_invocation_device_bytes: 65536,
                    public_session_bytes: 4096,
                    packing_session_bytes: 4096,
                    private_session_bytes: 4096,
                    private_output_payload: NativeCudaRecordPagePayloadDeclaration {
                        host: 0,
                        device: 1024,
                    },
                    original_input_and_transfer_payload: NativeCudaRecordPagePayloadDeclaration {
                        host: 1024,
                        device: 1024,
                    },
                    additional_owner_metadata_payload: NativeCudaRecordPagePayloadDeclaration {
                        host: 4096,
                        device: 0,
                    },
                },
            };
            let snapshot = rz_eval::pals_onnx::CudaRecordPageSnapshot {
                schema: "rz-pals-resident-cuda-record-pages-observation/1",
                initialized_scope: selection.initialized_scope,
                initialized_provider_count: selection.initialized_provider_count,
                packing_native_sessions: selection.packing_native_sessions,
                initialized_log_sha256: selection.initialized_log_sha256,
                initialized_log_digest_domain: selection.initialized_log_digest_domain,
                packing_graph_sha256: selection.packing_graph_sha256,
                process_epoch: selection.process_epoch,
                game_generation: 1,
                model_manifest_sha256: selection.model_manifest_sha256,
                model_epoch: selection.model_epoch,
                public_graph_sha256: selection.public_graph_sha256,
                encoding_sha256: selection.encoding_sha256,
                runtime_sha256: selection.runtime_sha256,
                runtime_identity_scope: selection.runtime_identity_scope,
                device_id: selection.device_id,
                live_blocks: 0,
                certified_projections: 0,
                whole_owner_host_bytes: 256,
                whole_owner_device_bytes: 512,
                active_invocation: false,
                physical_completion_unknown: false,
                quarantined: false,
                stats: rz_eval::pals_onnx::CudaRecordPageStats::default(),
            };
            let execution = NativeExecutionReceipt {
                provider: "cuda",
                device_id: Some(0),
                session_arena_bytes: Some(4096),
                runtime_sha256: [1; 32],
                runtime_bundle_sha256: Some([2; 32]),
                cuda_control_inventory_sha256: Some([12; 32]),
                cuda_loading_profile: None,
                cudnn_conv_search_diagnostic: None,
                startup_probe_timeout_ms: None,
                transient_request_device_bytes: 11,
                transient_execution_device_bytes: 22,
                pinned_request_bytes: 19,
                device_public_memory: false,
                host_record_pages: None,
                private_warm: None,
                cuda_record_pages: Some(selection),
            };
            (execution, snapshot)
        }

        // Pure clock/admission checks and controlled SingleWorker owner checks.
        // These fixtures do not load ORT/model files or prove provider, binary,
        // source-registration, native inference or whole-cost execution.
        #[test]
        fn conv_default_optional_wire_preserves_legacy_and_metadata_clones() {
            // Serialization-only fixture; no loaded backend or GPU claim.
            let mut execution = NativeExecutionReceipt {
                provider: "cpu",
                device_id: None,
                session_arena_bytes: None,
                runtime_sha256: [1; 32],
                runtime_bundle_sha256: None,
                cuda_control_inventory_sha256: None,
                cuda_loading_profile: None,
                cudnn_conv_search_diagnostic: None,
                startup_probe_timeout_ms: None,
                transient_request_device_bytes: 0,
                transient_execution_device_bytes: 0,
                pinned_request_bytes: 0,
                device_public_memory: false,
                host_record_pages: None,
                private_warm: None,
                cuda_record_pages: None,
            };
            let baseline = serde_json::to_value(&execution).unwrap();
            assert!(baseline.get("cudnn_conv_search_diagnostic").is_none());
            assert!(baseline.get("cuda_loading_profile").is_none());
            let descriptor = serde_json::json!({"scope":"synthetic_configuration_not_loaded_proof","selected_profile":"default-device-warm-v1"});
            let raw = serde_json::to_vec(&descriptor).unwrap();
            execution.cudnn_conv_search_diagnostic =
                Some(rz_eval::pals_onnx::PalsCuDnnExecutionDiagnostic {
                    descriptor,
                    descriptor_sha256: rz_eval::asset::sha256(&raw),
                    descriptor_encoding: "serde_json_utf8_fixed_descriptor_object",
                    actual_cudnn_plan: "unknown",
                    numerical_note: "unknown",
                    tf32_actual: "unknown",
                    tf32_retry: "unknown",
                    production_default_changed: false,
                    warm_success_claim: false,
                    pilot_success_claim: false,
                });
            let first = serde_json::to_value(execution.clone()).unwrap();
            let second = serde_json::to_value(execution.clone()).unwrap();
            assert_eq!(first, second);
            let observed = &first["cudnn_conv_search_diagnostic"];
            assert_eq!(observed["actual_cudnn_plan"], "unknown");
            assert_eq!(observed["tf32_actual"], "unknown");
            assert_eq!(observed["tf32_retry"], "unknown");
            assert_eq!(observed["warm_success_claim"], false);
            assert_eq!(
                observed["descriptor_sha256"],
                serde_json::to_value(rz_eval::asset::sha256(&raw)).unwrap()
            );
            assert_eq!(baseline["provider"], first["provider"]);
        }
        #[test]
        fn invocation_budget_preserves_original_absolute_clock_and_is_copy() {
            fn requires_copy<T: Copy>() {}
            requires_copy::<NativeInvocationBudget>();
            let started = Instant::now();
            let execution = started + Duration::from_secs(2);
            let reserve = Duration::from_secs(1);
            let whole = execution + reserve;
            let budget =
                NativeInvocationBudget::new_at(started, execution, whole, reserve, started)
                    .unwrap();
            assert_eq!(budget.started(), started);
            assert_eq!(budget.execution_until(), execution);
            assert_eq!(budget.whole_until(), whole);
            assert_eq!(budget.cleanup_reserve(), reserve);
            let cancel = AtomicBool::new(false);
            assert!(
                budget
                    .check_at(execution - Duration::from_nanos(1), &cancel)
                    .is_ok()
            );
            assert!(matches!(
                budget.check_at(execution, &cancel),
                Err(RoleError::Deadline)
            ));
            assert_eq!(budget.whole_until(), whole);
        }
        #[test]
        fn invocation_budget_rejects_future_zero_order_mismatch_and_overflow() {
            let now = Instant::now();
            let one = Duration::from_secs(1);
            let execution = now + one;
            let whole = execution + one;
            for (started, end, all, reserve) in [
                (now + one, now + one + one, now + one + one + one, one),
                (now, now, whole, one),
                (now, execution, execution, one),
                (now, execution, whole, Duration::ZERO),
                (now, execution, whole + one, one),
                (now, execution, whole, Duration::MAX),
            ] {
                assert!(matches!(
                    NativeInvocationBudget::new_at(started, end, all, reserve, now),
                    Err(RoleError::Backend(_))
                ));
            }
            // The public constructor applies the same future-start rejection.
            assert!(
                NativeInvocationBudget::new(
                    now + Duration::from_secs(60),
                    now + Duration::from_secs(61),
                    now + Duration::from_secs(62),
                    one
                )
                .is_err()
            );
        }
        #[test]
        fn invocation_admission_rejects_cancel_and_expiry_before_native_loading() {
            let now = Instant::now();
            let reserve = Duration::from_secs(2);
            let options = NativeOwnerOptions {
                drain_limit: reserve,
                host_record_pages: None,
            };
            let config = rz_eval::pals_onnx::PalsOnnxConfig::cpu();
            let execution = now + Duration::from_secs(60);
            let live =
                NativeInvocationBudget::new(now, execution, execution + reserve, reserve).unwrap();
            let cancel = AtomicBool::new(true);
            let canceled =
                validate_cpu_fresh_invocation(&config, options, live, &cancel).unwrap_err();
            assert_eq!(canceled.stage, NativeLoadStage::Admission);
            assert!(matches!(canceled.cause, RoleError::Canceled));
            assert!(canceled.canceled && !canceled.execution_expired && !canceled.whole_expired);
            assert!(canceled.finish.is_none());

            let old_start = now - Duration::from_secs(3);
            let expired_end = now - Duration::from_millis(1);
            let expired_reserve = Duration::from_secs(60);
            let expired = NativeInvocationBudget::new(
                old_start,
                expired_end,
                expired_end + expired_reserve,
                expired_reserve,
            )
            .unwrap();
            let expired_options = NativeOwnerOptions {
                drain_limit: expired_reserve,
                ..options
            };
            cancel.store(false, Ordering::Release);
            let failure = validate_cpu_fresh_invocation(&config, expired_options, expired, &cancel)
                .unwrap_err();
            assert_eq!(failure.stage, NativeLoadStage::Admission);
            assert!(matches!(failure.cause, RoleError::Deadline));
            assert!(failure.execution_expired && !failure.whole_expired && !failure.canceled);
            assert!(failure.elapsed >= Duration::from_secs(3));
            assert!(failure.finish.is_none()); // unobserved cleanup, not worker=0 proof
        }
        #[test]
        fn invocation_admission_rejects_cpu_config_pages_and_drain_mismatch() {
            let now = Instant::now();
            let reserve = Duration::from_secs(2);
            let execution = now + Duration::from_secs(60);
            let budget =
                NativeInvocationBudget::new(now, execution, execution + reserve, reserve).unwrap();
            let config = rz_eval::pals_onnx::PalsOnnxConfig::cpu();
            let options = NativeOwnerOptions {
                drain_limit: reserve,
                host_record_pages: None,
            };
            let cancel = AtomicBool::new(false);
            assert!(validate_cpu_fresh_invocation(&config, options, budget, &cancel).is_ok());
            let mut cuda = config;
            cuda.provider = rz_eval::onnx::Provider::Cuda {
                device_id: 0,
                arena_bytes: 1024,
            };
            let mut device = config;
            device.device_public_memory = true;
            let mut threads = config;
            threads.intra_threads = 0;
            let pages = NativeOwnerOptions {
                host_record_pages: Some(NativeHostRecordPageLimits {
                    max_page_entries: 2,
                    max_page_bytes: 1024,
                    max_transient_bytes: 1024,
                }),
                ..options
            };
            let other_drain = NativeOwnerOptions {
                drain_limit: Duration::from_secs(1),
                ..options
            };
            for (configuration, owner) in [
                (cuda, options),
                (device, options),
                (threads, options),
                (config, pages),
                (config, other_drain),
            ] {
                let failure = validate_cpu_fresh_invocation(&configuration, owner, budget, &cancel)
                    .unwrap_err();
                assert_eq!(failure.stage, NativeLoadStage::Admission);
                assert!(matches!(failure.cause, RoleError::Backend(_)));
                assert!(failure.finish.is_none());
                assert!(!failure.canceled && !failure.execution_expired && !failure.whole_expired);
            }
        }
        #[test]
        fn invocation_failure_preserves_primary_cause_and_separate_clock_facts() {
            let started = Instant::now();
            let execution = started + Duration::from_secs(1);
            let reserve = Duration::from_secs(1);
            let whole = execution + reserve;
            let budget =
                NativeInvocationBudget::new_at(started, execution, whole, reserve, started)
                    .unwrap();
            let cancel = AtomicBool::new(true);
            let failure = budget.failure_at(
                NativeLoadStage::RuntimeLoad,
                RoleError::Backend("original runtime cause".into()),
                whole,
                &cancel,
                None,
            );
            assert_eq!(failure.stage, NativeLoadStage::RuntimeLoad);
            assert!(
                matches!(&failure.cause, RoleError::Backend(detail) if detail == "original runtime cause")
            );
            assert_eq!(failure.elapsed, Duration::from_secs(2));
            assert!(failure.execution_expired && failure.whole_expired && failure.canceled);
            assert!(failure.finish.is_none());
            assert!(failure.backend_error.is_none());
        }
        #[test]
        fn invocation_direct_backend_error_preserves_typed_original_and_clock_facts() {
            // Injected bounded errors only: no library/runtime/model is loaded.
            // The same builder is used by the two direct native load boundaries.
            let started = Instant::now();
            let execution = started + Duration::from_secs(1);
            let reserve = Duration::from_secs(1);
            let whole = execution + reserve;
            let budget =
                NativeInvocationBudget::new_at(started, execution, whole, reserve, started)
                    .unwrap();
            let cancel = AtomicBool::new(true);
            let text = "x".repeat(rz_eval::error::CAUSE_PREFIX_BYTES + 7);
            for stage in [NativeLoadStage::RuntimeLoad, NativeLoadStage::BackendLoad] {
                let backend = BackendError::new(
                    FailureKind::BackendUnavailable,
                    FailureStage::Backend,
                    "injected original native load cause",
                )
                .with_external_cause(rz_eval::error::CauseCode::RuntimeInitialize, &text)
                .with_diagnostic("injected-bounded-native", &text);
                let expected_cause = backend.cause;
                let expected_projection = backend.to_string();
                // Preserve the original diagnostic allocation by ownership move.
                let diagnostic_address = std::ptr::from_ref(backend.native.as_deref().unwrap());
                let failure = budget
                    .failure_at(stage, model_error(&backend), whole, &cancel, None)
                    .with_backend_error(backend);
                assert_eq!(failure.stage, stage);
                assert!(
                    matches!(&failure.cause, RoleError::Backend(value) if value == &expected_projection)
                );
                assert_eq!(failure.elapsed, Duration::from_secs(2));
                assert!(failure.execution_expired && failure.whole_expired && failure.canceled);
                assert!(failure.finish.is_none());
                let original = failure.backend_error.as_deref().unwrap();
                assert_eq!(original.kind, FailureKind::BackendUnavailable);
                assert_eq!(original.stage, FailureStage::Backend);
                assert_eq!(original.detail, "injected original native load cause");
                assert_eq!(original.cause, expected_cause);
                let cause = original.cause.unwrap();
                assert_eq!(cause.code, rz_eval::error::CauseCode::RuntimeInitialize);
                assert_eq!(
                    usize::from(cause.hashed_bytes),
                    rz_eval::error::CAUSE_PREFIX_BYTES
                );
                assert!(cause.truncated && !cause.formatting_failed);
                let diagnostic = original.native.as_deref().unwrap();
                assert_eq!(std::ptr::from_ref(diagnostic), diagnostic_address);
                assert_eq!(diagnostic.message.len(), rz_eval::error::CAUSE_PREFIX_BYTES);
                assert!(diagnostic.truncated);
            }
        }
        #[test]
        fn invocation_cpu_runtime_admission_refuses_cuda_bundle_before_bootstrap() {
            assert!(validate_cpu_invocation_bundle(None).is_ok());
            assert!(matches!(
                validate_cpu_invocation_bundle(Some([0; 32])),
                Err(RoleError::Backend(_))
            ));
            assert!(matches!(
                validate_cpu_invocation_bundle(Some([255; 32])),
                Err(RoleError::Backend(_))
            ));
        }
        #[test]
        fn invocation_postconstruction_cancel_retains_same_actual_finish_owner() {
            let model = fixture_model(binding_fixture_command);
            let actual_owner = Arc::clone(&model.owner);
            let now = Instant::now();
            let reserve = Duration::from_secs(2);
            let execution = now + Duration::from_secs(60);
            let budget =
                NativeInvocationBudget::new(now, execution, execution + reserve, reserve).unwrap();
            let cancel = AtomicBool::new(true);
            let failure = accept_invocation_owner(model, budget, &cancel)
                .err()
                .unwrap();
            assert_eq!(failure.stage, NativeLoadStage::PostConstruction);
            assert!(matches!(failure.cause, RoleError::Canceled));
            assert!(failure.canceled && !failure.execution_expired && !failure.whole_expired);
            assert!(failure.backend_error.is_none());
            let handle = failure.finish.unwrap();
            assert!(Arc::ptr_eq(&handle.owner, &actual_owner));
            assert!(!handle.receipt().physical_shutdown_confirmed);
            // The one absolute W is used; no fresh now+D cleanup window exists.
            let receipt = handle.finish(budget.whole_until()).unwrap();
            assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
            assert_eq!(receipt.physically_completed_role_calls, 0);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
        }
        #[test]
        fn invocation_postconstruction_expiry_keeps_owner_and_expired_whole_clock() {
            let model = fixture_model(binding_fixture_command);
            let actual_owner = Arc::clone(&model.owner);
            let now = Instant::now();
            let started = now - Duration::from_secs(4);
            let execution = now - Duration::from_secs(2);
            let reserve = Duration::from_secs(1);
            let whole = execution + reserve;
            let budget = NativeInvocationBudget::new(started, execution, whole, reserve).unwrap();
            let cancel = AtomicBool::new(false);
            let failure = accept_invocation_owner(model, budget, &cancel)
                .err()
                .unwrap();
            assert_eq!(failure.stage, NativeLoadStage::PostConstruction);
            assert!(matches!(failure.cause, RoleError::Deadline));
            assert!(failure.execution_expired && failure.whole_expired && !failure.canceled);
            let handle = failure.finish.unwrap();
            assert!(Arc::ptr_eq(&handle.owner, &actual_owner));
            // No cleanup is attempted or successful release claimed after W.
            // The caller still owns this same handle and the original past W.
            assert!(budget.whole_until() < Instant::now());
            let receipt = handle.receipt();
            assert!(!receipt.physical_shutdown_confirmed && !receipt.native_buffers_released);
            assert_eq!(receipt.physically_completed_role_calls, 0);
        }
        #[test]
        fn resident_none_preserves_legacy_receipt_omission_and_copy_options() {
            fn requires_copy<T: Copy>() {}
            requires_copy::<NativeOwnerOptions>();
            let model = fixture_model(|command| match command {
                PalsNativeCommand::SnapshotStats => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::Stats(PalsBackendStats::default())))
                }
                _ => unexpected_cuda_placement(),
            });
            require_probe(&model);
            let before = serde_json::to_value(model.finish_handle().receipt()).unwrap();
            assert!(before["execution"].get("cuda_record_pages").is_none());
            assert!(before.get("cuda_record_page_observation").is_none());
            assert!(before.get("cuda_record_page_observation_error").is_none());
            assert!(before.get("cuda_record_page_observation_scope").is_none());
            assert!(
                before["startup_probe"]
                    .get("cuda_record_page_observation")
                    .is_none()
            );
            assert!(
                before["startup_probe"]
                    .get("cuda_record_page_observation_error")
                    .is_none()
            );
            let ended = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(1))
                .unwrap();
            let after = serde_json::to_string(&ended).unwrap();
            assert!(!after.contains("cuda_record_page"));
            assert!(ended.physical_shutdown_confirmed && ended.native_buffers_released);
        }

        #[test]
        fn resident_selection_refuses_incompatible_modes_before_model_loading() {
            let options = NativeOwnerOptions {
                drain_limit: Duration::from_secs(1),
                host_record_pages: None,
            };
            let mut config = rz_eval::pals_onnx::PalsOnnxConfig::cpu();
            assert!(
                validate_cuda_record_pages_selection(
                    &config,
                    options,
                    DEPLOYMENT_FROZEN_EPOCH,
                    false
                )
                .is_err()
            );
            config.provider = rz_eval::onnx::Provider::Cuda {
                device_id: 0,
                arena_bytes: 4096,
            };
            assert_eq!(
                validate_cuda_record_pages_selection(
                    &config,
                    options,
                    DEPLOYMENT_FROZEN_EPOCH,
                    false
                )
                .is_ok(),
                cfg!(all(
                    feature = "experimental-io-binding",
                    feature = "onnx-cuda",
                    target_os = "linux"
                ))
            );
            for mutation in 0..6 {
                let mut c = config;
                let mut o = options;
                let mut epoch = DEPLOYMENT_FROZEN_EPOCH;
                let mut warm = false;
                match mutation {
                    0 => c.device_public_memory = true,
                    1 => c.cache_public_memory = false,
                    2 => {
                        o.host_record_pages = Some(NativeHostRecordPageLimits {
                            max_page_entries: 2,
                            max_page_bytes: 1024,
                            max_transient_bytes: 1024,
                        })
                    }
                    3 => epoch += 1,
                    4 => warm = true,
                    _ => {
                        c.provider = rz_eval::onnx::Provider::Cuda {
                            device_id: -1,
                            arena_bytes: 4096,
                        }
                    }
                }
                assert!(validate_cuda_record_pages_selection(&c, o, epoch, warm).is_err());
            }
        }

        #[cfg(feature = "experimental-io-binding")]
        #[test]
        fn resident_owner_device_reservation_replaces_legacy_transients_once() {
            let (mut execution, _) = resident_boundary_fixture();
            let selected = execution.cuda_record_pages.take().unwrap();
            assert_eq!(execution.request_device_reservation(), 11);
            assert_eq!(execution.execution_device_reservation(), 22);
            assert_eq!(execution.runtime_device_reservation().unwrap(), 33);
            execution.cuda_record_pages = Some(selected);
            assert_eq!(execution.request_device_reservation(), 0);
            assert_eq!(execution.execution_device_reservation(), 65536);
            assert_eq!(execution.runtime_device_reservation().unwrap(), 65536);
            assert_eq!(execution.pinned_request_bytes, 19);
            // Legacy transient declarations remain visible but do not create
            // an additional charge in the selected combined-owner scope.
            let wire = serde_json::to_value(&execution).unwrap();
            assert_eq!(wire["transient_request_device_bytes"], 11);
            assert_eq!(wire["transient_execution_device_bytes"], 22);
            execution.cuda_record_pages = None;
            execution.transient_execution_device_bytes = u64::MAX;
            assert!(execution.runtime_device_reservation().is_err());
        }

        #[cfg(feature = "experimental-io-binding")]
        #[test]
        fn resident_session_declarations_match_the_actual_model_arena() {
            let (execution, _) = resident_boundary_fixture();
            let mut declaration = execution.cuda_record_pages.unwrap().declared_resources;
            let mut config = rz_eval::pals_onnx::PalsOnnxConfig::cpu();
            assert!(validate_cuda_record_page_session_declarations(&config, &declaration).is_err());
            config.provider = rz_eval::onnx::Provider::Cuda {
                device_id: 0,
                arena_bytes: 4096,
            };
            assert!(validate_cuda_record_page_session_declarations(&config, &declaration).is_ok());
            declaration.private_session_bytes += 1;
            assert!(validate_cuda_record_page_session_declarations(&config, &declaration).is_err());
            declaration.private_session_bytes -= 1;
            declaration.public_session_bytes -= 1;
            assert!(validate_cuda_record_page_session_declarations(&config, &declaration).is_err());
        }

        #[cfg(feature = "experimental-io-binding")]
        #[test]
        fn resident_snapshot_requires_exact_owner_pins_idle_state_and_bounded_reservations() {
            let (execution, actual) = resident_boundary_fixture();
            let mut selection = execution.cuda_record_pages.unwrap();
            let snapshot: NativeCudaRecordPageSnapshotReceipt = actual.into();
            assert!(checked_cuda_record_page_observation(snapshot.clone(), &selection).is_ok());
            for mutation in 0..11 {
                let mut wrong = snapshot.clone();
                match mutation {
                    0 => wrong.process_epoch += 1,
                    1 => wrong.model_epoch = [0; 32],
                    2 => wrong.packing_graph_sha256 = [0; 32],
                    3 => wrong.runtime_sha256 = [0; 32],
                    4 => wrong.initialized_provider_count -= 1,
                    5 => wrong.initialized_log_sha256 = [0; 32],
                    6 => wrong.active_invocation = true,
                    7 => wrong.physical_completion_unknown = true,
                    8 => wrong.quarantined = true,
                    9 => {
                        wrong.stats.last_reserved_host_bytes =
                            selection.declared_resources.max_invocation_host_bytes + 1
                    }
                    _ => {
                        wrong.stats.last_reserved_device_bytes =
                            selection.declared_resources.max_invocation_device_bytes + 1
                    }
                }
                assert!(checked_cuda_record_page_observation(wrong, &selection).is_err());
            }
            selection.installation_game_generation = 2;
            assert!(checked_cuda_record_page_observation(snapshot, &selection).is_err());
        }

        #[cfg(feature = "experimental-io-binding")]
        #[test]
        fn resident_final_snapshot_is_an_actual_control_ack_not_the_initial_marker() {
            let (execution, mut snapshot) = resident_boundary_fixture();
            snapshot.live_blocks = 1;
            snapshot.certified_projections = 2;
            snapshot.stats.packing_runs_completed = 3;
            let pages = Arc::new(AtomicU64::new(0));
            let called = Arc::clone(&pages);
            let model = fixture_model_with_execution(
                move |command| {
                    PhysicalRun::Complete(Ok(match command {
                        PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                        PalsNativeCommand::SnapshotCudaRecordPages => {
                            called.fetch_add(1, Ordering::AcqRel);
                            PalsNativeResult::CudaRecordPagesObserved(Box::new(snapshot.clone()))
                        }
                        PalsNativeCommand::SnapshotStats => {
                            PalsNativeResult::Stats(PalsBackendStats::default())
                        }
                        _ => PalsNativeResult::NewGame,
                    }))
                },
                Some(execution),
            );
            let finish = model.finish_handle();
            assert!(finish.receipt().execution.cuda_record_pages.is_some());
            assert!(finish.receipt().cuda_record_page_observation.is_none());
            let receipt = finish
                .finish(Instant::now() + Duration::from_secs(1))
                .unwrap();
            let observed = receipt.cuda_record_page_observation.as_ref().unwrap();
            assert_eq!(pages.load(Ordering::Acquire), 1);
            assert_eq!(observed.live_blocks, 1);
            assert_eq!(observed.certified_projections, 2);
            assert_eq!(observed.stats.packing_runs_completed, 3);
            assert_eq!(
                receipt.cuda_record_page_observation_scope,
                Some("exclusive_worker_before_shutdown")
            );
            assert!(receipt.cuda_record_page_observation_error.is_none());
            assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
            assert_eq!(receipt.physically_completed_role_calls, 0);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
        }

        #[cfg(feature = "experimental-io-binding")]
        #[test]
        fn resident_wrong_metadata_response_preserves_failure_and_missing_observation() {
            let (execution, _) = resident_boundary_fixture();
            let model = fixture_model_with_execution(
                |command| {
                    PhysicalRun::Complete(Ok(match command {
                        PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                        // Deliberately wrong NN-zero metadata response; never a page ACK.
                        _ => PalsNativeResult::Stats(PalsBackendStats::default()),
                    }))
                },
                Some(execution),
            );
            let finish = model.finish_handle();
            assert!(matches!(
                finish.finish(Instant::now() + Duration::from_secs(1)),
                Err(RoleError::Backend(_))
            ));
            let receipt = finish.receipt();
            assert!(receipt.cuda_record_page_observation.is_none());
            assert!(receipt.cuda_record_page_observation_scope.is_none());
            assert_eq!(
                receipt.cuda_record_page_observation_error,
                Some(NativeStartupErrorKind::Backend)
            );
            assert!(receipt.backend_stats.is_none());
            assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
            assert_eq!(receipt.physically_completed_role_calls, 0);
        }

        #[cfg(feature = "experimental-io-binding")]
        #[test]
        fn resident_expired_finish_never_dispatches_or_fabricates_a_snapshot() {
            let (execution, _) = resident_boundary_fixture();
            let calls = Arc::new(AtomicU64::new(0));
            let called = Arc::clone(&calls);
            let model = fixture_model_with_execution(
                move |command| {
                    called.fetch_add(1, Ordering::AcqRel);
                    match command {
                        PalsNativeCommand::VerifyRuntime => {
                            PhysicalRun::Complete(Ok(PalsNativeResult::RuntimeVerified))
                        }
                        _ => unexpected_cuda_placement(),
                    }
                },
                Some(execution),
            );
            // An already admitted control is physically joined within the
            // fixture's explicit bound. Consuming that ACK after expiry must
            // not admit the subsequent resident observation command.
            complete_admitted_deadline_control(&model, PalsNativeCommand::VerifyRuntime);
            let finish = model.finish_handle();
            assert!(matches!(
                finish.finish(Instant::now()),
                Err(RoleError::Deadline)
            ));
            let receipt = finish.receipt();
            assert_eq!(calls.load(Ordering::Acquire), 1);
            assert!(receipt.cuda_record_page_observation.is_none());
            assert_eq!(
                receipt.cuda_record_page_observation_error,
                Some(NativeStartupErrorKind::Deadline)
            );
            assert!(receipt.cuda_record_page_observation_scope.is_none());
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
        }

        #[cfg(feature = "experimental-io-binding")]
        #[test]
        fn resident_failed_startup_keeps_original_error_and_separate_final_ack() {
            let (execution, snapshot) = resident_boundary_fixture();
            let mut model = fixture_model_with_execution(
                move |command| match command {
                    PalsNativeCommand::Evaluate(_) | PalsNativeCommand::EvaluateWithEvidence(_) => {
                        PhysicalRun::Complete(Err(BackendError::new(
                            FailureKind::BackendFailure,
                            FailureStage::Backend,
                            "synthetic startup primary failure",
                        )))
                    }
                    PalsNativeCommand::VerifyRuntime => {
                        PhysicalRun::Complete(Ok(PalsNativeResult::RuntimeVerified))
                    }
                    PalsNativeCommand::SnapshotCudaRecordPages => PhysicalRun::Complete(Ok(
                        PalsNativeResult::CudaRecordPagesObserved(Box::new(snapshot.clone())),
                    )),
                    PalsNativeCommand::SnapshotStats => PhysicalRun::Complete(Ok(
                        PalsNativeResult::Stats(PalsBackendStats::default()),
                    )),
                    _ => unexpected_cuda_placement(),
                },
                Some(execution),
            );
            require_probe(&model);
            assert!(matches!(
                model.prepare_startup(Instant::now() + Duration::from_secs(1)),
                Err(RoleError::Backend(_))
            ));
            let finish = model.finish_handle();
            let failed = finish.receipt();
            let probe = failed.startup_probe.as_ref().unwrap();
            assert!(probe.cuda_record_page_observation.is_none());
            assert_eq!(
                probe.cuda_record_page_observation_error,
                Some(NativeStartupErrorKind::Backend)
            );
            assert!(failed.cuda_record_page_observation.is_none());
            assert!(!model.owner.startup_ready.load(Ordering::Acquire));
            let ended = finish
                .finish(Instant::now() + Duration::from_secs(1))
                .unwrap();
            assert!(ended.cuda_record_page_observation.is_some());
            assert!(
                ended
                    .startup_probe
                    .as_ref()
                    .unwrap()
                    .cuda_record_page_observation
                    .is_none()
            );
            assert_eq!(
                ended
                    .startup_probe
                    .as_ref()
                    .unwrap()
                    .cuda_record_page_observation_error,
                Some(NativeStartupErrorKind::Backend)
            );
            assert!(ended.last_failure.is_some());
            assert!(!model.owner.startup_ready.load(Ordering::Acquire));
        }

        #[test]
        fn startup_cuda_witness_requires_owner_pins_and_both_private_branches() {
            use rz_eval::pals_onnx::{PalsGraphPlacement, PalsKernelWitness};
            // Synthetic boundary evidence exercises admission only; it is not
            // a CUDA device, physical kernel or native placement attestation.
            let execution = NativeExecutionReceipt {
                provider: "cuda",
                device_id: Some(0),
                session_arena_bytes: Some(2 * 1024 * 1024 * 1024),
                runtime_sha256: [1; 32],
                runtime_bundle_sha256: Some([2; 32]),
                cuda_control_inventory_sha256: Some([3; 32]),
                cuda_loading_profile: None,
                cudnn_conv_search_diagnostic: None,
                startup_probe_timeout_ms: None,
                transient_request_device_bytes: 0,
                transient_execution_device_bytes: 0,
                pinned_request_bytes: 0,
                device_public_memory: false,
                host_record_pages: None,
                private_warm: None,
                cuda_record_pages: None,
            };
            let graphs: Vec<_> = ["public", "shared_pc"]
                .into_iter()
                .map(|role| rz_eval::pals_onnx::PalsGraphIdentity {
                    role: role.into(),
                    sha256: [4; 32],
                    serialized_bytes: 1,
                })
                .collect();
            let residency = PalsSessionResidency {
                graphs,
                native_sessions: 2,
                layout: "synthetic-boundary-test".into(),
                reader_initializer_bank: None,
                role_reader_weights_shared: None,
                native_resident_parameter_bytes: None,
                vram_peak_bytes: None,
            };
            let kernels = PalsKernelWitness {
                profile_sha256: [5; 32],
                cuda_kernels: 2,
                cuda_transfer_kernels: 0,
                approved_cpu_control_kernels: 1,
                neural_kernels: 2,
                proposer_private_kernels: 1,
                critic_private_kernels: 1,
            };
            let mut witness = PalsCudaPlacementWitness {
                schema: "rovezero.pals-cuda-metadata-control.v2".into(),
                inventory_sha256: [3; 32],
                manifest_sha256: [6; 32],
                runtime_sha256: [1; 32],
                runtime_bundle_sha256: [2; 32],
                optimization: PalsGraphOptimization::Disable,
                initialization: ["public", "shared_pc"]
                    .into_iter()
                    .map(|role| PalsGraphPlacement {
                        role: role.into(),
                        graph_sha256: [4; 32],
                        log_sha256: [7; 32],
                        assigned_nodes: 3,
                        cuda_nodes: 2,
                        approved_cpu_control_nodes: 1,
                        optimization: PalsGraphOptimization::Disable,
                        approved_transfers: Vec::new(),
                        recursive_coverage: "synthetic-boundary-test".into(),
                    })
                    .collect(),
                public: kernels.clone(),
                shared_pc: kernels,
                category_provenance: "synthetic-boundary-test".into(),
            };
            assert!(
                validate_startup_cuda_witness(&witness, &execution, [6; 32], &residency).is_ok()
            );
            assert!(
                validate_startup_cuda_witness(&witness, &execution, [8; 32], &residency).is_err()
            );
            witness.optimization = PalsGraphOptimization::Level1;
            assert!(
                validate_startup_cuda_witness(&witness, &execution, [6; 32], &residency).is_err()
            );
            witness.optimization = PalsGraphOptimization::Disable;
            witness.shared_pc.critic_private_kernels = 0;
            assert!(
                validate_startup_cuda_witness(&witness, &execution, [6; 32], &residency).is_err()
            );
            witness.shared_pc.critic_private_kernels = 1;
            witness.initialization[0].graph_sha256 = [8; 32];
            assert!(
                validate_startup_cuda_witness(&witness, &execution, [6; 32], &residency).is_err()
            );
        }

        #[test]
        fn host_page_options_reject_device_combination_before_owner_construction() {
            let options = NativeOwnerOptions {
                drain_limit: Duration::from_secs(1),
                host_record_pages: Some(NativeHostRecordPageLimits {
                    max_page_entries: 257,
                    max_page_bytes: 2 * 1024 * 1024,
                    max_transient_bytes: 2 * 1024 * 1024,
                }),
            };
            let mut config = rz_eval::pals_onnx::PalsOnnxConfig::cpu();
            options.validate(&config).unwrap();
            config.device_public_memory = true;
            assert!(options.validate(&config).is_err());
            config.device_public_memory = false;
            config.cache_public_memory = false;
            assert!(options.validate(&config).is_err());
            config.cache_public_memory = true;
            let mut invalid = options;
            invalid.host_record_pages.as_mut().unwrap().max_page_entries = 1;
            assert!(invalid.validate(&config).is_err());
            invalid = options;
            invalid.drain_limit = Duration::ZERO;
            assert!(invalid.validate(&config).is_err());
        }
        #[test]
        fn host_page_receipt_distinguishes_default_declaration_observation_and_fence() {
            let model = fixture_model(|command| {
                PhysicalRun::Complete(Ok(match command {
                    PalsNativeCommand::SnapshotStats => {
                        PalsNativeResult::Stats(PalsBackendStats::default())
                    }
                    _ => PalsNativeResult::NewGame,
                }))
            });
            let handle = model.finish_handle();
            let default = handle
                .finish(Instant::now() + Duration::from_secs(1))
                .unwrap();
            let bytes = serde_json::to_string(&default).unwrap();
            assert!(!bytes.contains("host_record_page"));
            validate_host_record_page_evidence(&default, true).unwrap();
            // Pure synthetic metadata admission checks; neither this fixture
            // nor a declaration is a native ONNX/GPU execution attestation.
            let mut selected = default;
            selected
                .residency
                .graphs
                .push(rz_eval::pals_onnx::PalsGraphIdentity {
                    role: "public".into(),
                    sha256: [9; 32],
                    serialized_bytes: 1,
                });
            let declaration = NativeHostRecordPageDeclaration::from_policy(
                HostRecordPagePolicy::for_registered_graph([9; 32]),
            );
            assert_eq!(
                declaration.reserved_host_bytes().unwrap(),
                4 * REQUEST_BYTES
            );
            selected.execution.host_record_pages = Some(declaration.clone());
            let snapshot = NativeHostRecordPageSnapshotReceipt {
                declaration,
                bank: NativeHostRecordPageBankReceipt {
                    entries: 1,
                    reserved_bytes: 100,
                    pinned_entries: 0,
                    pinned_bytes: 0,
                    max_entries: 257,
                    max_bytes: REQUEST_BYTES,
                    entry_backing_bytes: 64,
                },
                stats: NativeHostRecordPageStatsReceipt {
                    view_hits: 1,
                    view_misses: 1,
                    board_hits: 1,
                    board_misses: 1,
                    record_hits: 2,
                    record_misses: 1,
                    public_calls_attempted: 1,
                    public_calls_completed: 1,
                    encoded_record_tokens: 1,
                    submitted_record_tokens: 1,
                    contextual_board_encodes_completed: 1,
                    joins_completed: 2,
                    joined_bytes: 200,
                    evicted_pages: 0,
                    max_reserved_host_transient_bytes: 1000,
                },
                active_pin_count: 0,
                retained_join_bytes: 100,
                active_subset_bytes: 0,
                active_join_backing_bytes: 0,
                active_full_input_bytes: 0,
                transient_reservation_bytes: 0,
                quarantined: false,
            };
            selected.host_record_page_observation = Some(NativeHostRecordPageObservationReceipt {
                schema: "rz-pals-host-record-page-observation/1",
                status: HostRecordPageObservationStatus::Available,
                attempted_command_ordinal: 3,
                latest: Some(NativeHostRecordPageCommandReceipt {
                    command_ordinal: 3,
                    boundary: HostRecordPageObservationBoundary::SnapshotStats,
                    outcome: HostRecordPageObservationOutcome::ReturnedOk,
                    snapshot,
                }),
            });
            validate_host_record_page_evidence(&selected, true).unwrap();
            let mut wrong = selected.clone();
            wrong
                .execution
                .host_record_pages
                .as_mut()
                .unwrap()
                .implementation_sha256 = [0; 32];
            assert!(validate_host_record_page_evidence(&wrong, false).is_err());
            wrong = selected.clone();
            wrong
                .host_record_page_observation
                .as_mut()
                .unwrap()
                .latest
                .as_mut()
                .unwrap()
                .snapshot
                .bank
                .reserved_bytes = 3 * REQUEST_BYTES;
            assert!(validate_host_record_page_evidence(&wrong, false).is_err());
            wrong = selected.clone();
            wrong.host_record_page_observation.as_mut().unwrap().status =
                HostRecordPageObservationStatus::Contended;
            validate_host_record_page_evidence(&wrong, false).unwrap();
            assert!(validate_host_record_page_evidence(&wrong, true).is_err());
            wrong = selected.clone();
            let actual = wrong
                .host_record_page_observation
                .as_mut()
                .unwrap()
                .latest
                .as_mut()
                .unwrap();
            actual.outcome = HostRecordPageObservationOutcome::PhysicalCompletionUnknown;
            actual.snapshot.quarantined = true;
            actual.snapshot.active_pin_count = 1;
            actual.snapshot.bank.pinned_entries = 1;
            actual.snapshot.bank.pinned_bytes = 100;
            actual.snapshot.active_full_input_bytes = 200;
            actual.snapshot.transient_reservation_bytes = 1000;
            wrong.quarantined = true;
            wrong.physical_shutdown_confirmed = false;
            wrong.native_buffers_released = false;
            validate_host_record_page_evidence(&wrong, false).unwrap();
            assert!(validate_host_record_page_evidence(&wrong, true).is_err());
            // Missing selected observations remain missing on failure, never
            // converted into an all-zero successful final snapshot.
            selected.host_record_page_observation = None;
            validate_host_record_page_evidence(&selected, false).unwrap();
            assert!(validate_host_record_page_evidence(&selected, true).is_err());
        }
        fn complete_admitted_deadline_control(model: &NativeRoleModel, command: PalsNativeCommand) {
            let lease = model.owner.worker.lock().unwrap().submit(command).unwrap();
            model.owner.in_flight.fetch_add(1, Ordering::AcqRel);
            *model.owner.stats_lease.lock().unwrap() = Some(lease);
            // The in-process fixture's actual join makes this admitted output
            // available without consuming its physical lease. No GPU/process
            // proof or fresh product cleanup window is claimed by this helper.
            let until = Instant::now() + Duration::from_secs(1);
            loop {
                match model.owner.worker.lock().unwrap().try_shutdown() {
                    Poll::Ready(Ok(())) => break,
                    Poll::Ready(Err(error)) => panic!("deadline fixture worker failed: {error:?}"),
                    Poll::Pending => {
                        assert!(
                            Instant::now() < until,
                            "deadline fixture join exceeded its bound"
                        );
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
            }
        }
        #[test]
        fn expired_idle_final_stats_never_admit_a_new_control() {
            let calls = Arc::new(AtomicU64::new(0));
            let actual_calls = Arc::clone(&calls);
            let model = fixture_model(move |command| {
                actual_calls.fetch_add(1, Ordering::AcqRel);
                PhysicalRun::Complete(Ok(match command {
                    PalsNativeCommand::SnapshotStats => {
                        PalsNativeResult::Stats(PalsBackendStats::default())
                    }
                    _ => PalsNativeResult::RuntimeVerified,
                }))
            });
            let handle = model.finish_handle();
            assert!(matches!(
                handle.collect_final_stats(Instant::now()),
                Err(RoleError::Deadline)
            ));
            assert_eq!(calls.load(Ordering::Acquire), 0);
            let receipt = handle.receipt();
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert!(!receipt.quarantined);
            assert!(receipt.backend_stats.is_none());
            assert!(handle.owner.stats_lease.lock().unwrap().is_none());
            // Test-only explicit later cleanup; the product caller receives
            // Deadline and never grants itself a second owner drain window.
            let receipt = handle
                .finish(Instant::now() + Duration::from_secs(1))
                .unwrap();
            assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
            assert_eq!(calls.load(Ordering::Acquire), 1);
        }
        #[test]
        fn already_completed_stats_are_preserved_after_the_shared_deadline() {
            let model = fixture_model(|command| {
                PhysicalRun::Complete(Ok(match command {
                    PalsNativeCommand::SnapshotStats => {
                        PalsNativeResult::Stats(PalsBackendStats::default())
                    }
                    _ => PalsNativeResult::RuntimeVerified,
                }))
            });
            complete_admitted_deadline_control(&model, PalsNativeCommand::SnapshotStats);
            let handle = model.finish_handle();
            let receipt = handle.finish(Instant::now()).unwrap();
            assert!(receipt.backend_stats.is_some());
            assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
            assert!(!receipt.quarantined);
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert_eq!(receipt.physically_completed_role_calls, 0);
            assert!(handle.owner.stats_lease.lock().unwrap().is_none());
        }
        #[test]
        fn completed_verify_expiry_preserves_idle_fence_without_new_stats_admission() {
            let (mut execution, _) = loading_mapping_fixture();
            execution.cuda_loading_profile = None;
            let calls = Arc::new(AtomicU64::new(0));
            let actual_calls = Arc::clone(&calls);
            // Synthetic CUDA metadata only; this is a known in-process control
            // return, not a real CUDA mapping/NN execution attestation.
            let model = fixture_model_with_execution(
                move |command| {
                    actual_calls.fetch_add(1, Ordering::AcqRel);
                    PhysicalRun::Complete(Ok(match command {
                        PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                        _ => PalsNativeResult::Stats(PalsBackendStats::default()),
                    }))
                },
                Some(execution),
            );
            complete_admitted_deadline_control(&model, PalsNativeCommand::VerifyRuntime);
            let handle = model.finish_handle();
            assert!(matches!(
                handle.finish(Instant::now()),
                Err(RoleError::Deadline)
            ));
            let receipt = handle.receipt();
            assert_eq!(calls.load(Ordering::Acquire), 1);
            assert!(receipt.final_runtime_mapping_confirmed == Some(true));
            assert!(receipt.backend_stats.is_none());
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
            assert!(!receipt.quarantined);
            assert!(handle.owner.stats_lease.lock().unwrap().is_none());
        }
        #[test]
        fn ticketed_actual_worker_keeps_fixed_root_and_accepts_older_original_output() {
            // Deterministic raw heads through the existing native runtime and
            // SingleWorker. No ORT library, model asset or GPU is loaded.
            let mut model = recorded_fixture_model(None, false);
            assert!(model.applied_session_settings().is_none());
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            let until = Instant::now() + Duration::from_secs(2);
            let authority = SearchAuthority {
                epoch: model.epoch,
                game: GameGeneration(1),
                root: RootGeneration(7),
                implementation: Digest([9; 32]),
            };
            let state = ContractPosition::new(OwnerId(2), position.clone())
                .export()
                .unwrap()
                .snapshot()
                .identity();
            let mut outputs = Vec::new();
            for sequence in 1..=2 {
                let query = RoleQuery {
                    position: &position,
                    legal: &legal,
                    prefix: &[],
                    proposal: &[],
                    counterexample: None,
                    records: &[],
                    revision: sequence,
                    deadline: until,
                    cancel: &cancel,
                };
                let input = prepare_role_input_for_config(
                    &query,
                    NativeQueryKind::Propose,
                    model.model_epoch,
                    &model.owner.model_config,
                )
                .unwrap();
                let ticket = rz_contracts::pals_adaptive::RoleTicketEnvelope {
                    id: RequestId::new(model.epoch, sequence),
                    authority,
                    question: Digest([sequence as u8; 32]),
                    view: rz_contracts::pals_adaptive::ReviewViewSeal {
                        authority,
                        view_sequence: sequence,
                        context_policy: Digest([8; 32]),
                        view_digest: Digest([sequence as u8; 32]),
                        root_state: state,
                        actual_tokens: 0,
                        selected_records: 0,
                        anchors: 1,
                        length_bucket: 8,
                    },
                    representation: model.ticketed_representation(&input).unwrap(),
                    role: Role::Proposer,
                    original_deadline: Deadline {
                        clock: model.clock.domain(),
                        at: MonotonicTick(model.clock.now().0 + 1_500_000_000),
                    },
                };
                let context = RoleLogicalContext {
                    game_generation: 1,
                    search_generation: 7,
                    situation: rz_search::pals::store::SituationId {
                        slot: 0,
                        generation: 7,
                    },
                    state: rz_search::pals::store::StateId(0),
                    focus: rz_search::pals::store::LineId(0),
                    purpose: RoleQueryPurpose::ProposePolicy,
                    prefix: vec![],
                    focus_sha256: [1; 32],
                    prefix_sha256: [2; 32],
                    proposal_sha256: [3; 32],
                    refutation_sha256: None,
                    divergence_sha256: [4; 32],
                    public_revision: sequence,
                    situation_revision: sequence,
                };
                let output = model
                    .ticketed_run(
                        &ticket,
                        input,
                        &position,
                        until,
                        &cancel,
                        NativePreparedContext::Role {
                            query: &query,
                            kind: NativeQueryKind::Propose,
                        },
                        &context,
                    )
                    .unwrap();
                assert_eq!(output.output.id, ticket.id);
                assert_eq!(output.output.key, ticket.representation);
                // 이 synthetic worker는 owner 계수를 발급하지 않는다. 실제 요청/
                // 출력 하나를 public/private/V 실행 수로 추정하면 안 된다.
                assert!(output.accounting.is_none());
                let mut observation = ObservedRoleExecution {
                    binding: NativeRoleExecutionBinding {
                        request: ticket.id,
                        execution: output.execution.unwrap(),
                    },
                    ready_observed: false,
                    unknown_observed: false,
                    observer_error: Some(RoleError::Unavailable),
                    prepared_state: state,
                    prepared_perspective: Some(rz_contracts::Color::White),
                    input_sha256: ticket.representation.input.0,
                    fresh: Some(true),
                    completed_nn_inputs: Some(1),
                    measurement_complete: true,
                    owner_accounting: None,
                };
                assert!(observation.ticket_execution_evidence().is_none());
                observation.ready_observed = true;
                let known = observation.ticket_execution_evidence().unwrap();
                assert_eq!(known.completed_nn_inputs, Some(1));
                assert!(known.accounting.is_none());
                observation.unknown_observed = true;
                assert!(observation.ticket_execution_evidence().is_none());
                assert_eq!(observation.completed_nn_inputs, Some(1));
                assert_eq!(model.scope.get().unwrap().authority.root, RootGeneration(7));
                assert!(model.delivered_request.is_none());
                assert!(model.owner.role_execution.lock().unwrap().is_none());
                outputs.push((ticket, output));
            }
            assert_eq!(model.owner.physical_completed.load(Ordering::Acquire), 2);
            assert_eq!(model.owner.consumed.load(Ordering::Acquire), 0);
            assert_eq!(model.ticketed_delivered.len(), 2);
            model.ticketed_acknowledge(&outputs[0].0, true).unwrap();
            model.ticketed_acknowledge(&outputs[1].0, true).unwrap();
            assert!(model.ticketed_acknowledge(&outputs[0].0, true).is_err());
            assert_eq!(model.owner.consumed.load(Ordering::Acquire), 2);
            let receipt = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(1))
                .unwrap();
            assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
        }
        fn fixture_model<F>(run: F) -> NativeRoleModel
        where
            F: FnMut(&PalsNativeCommand) -> PhysicalRun<Result<PalsNativeResult, BackendError>>
                + Send
                + 'static,
        {
            fixture_model_with_execution(run, None)
        }
        fn fixture_model_with_execution<F>(
            run: F,
            execution: Option<NativeExecutionReceipt>,
        ) -> NativeRoleModel
        where
            F: FnMut(&PalsNativeCommand) -> PhysicalRun<Result<PalsNativeResult, BackendError>>
                + Send
                + 'static,
        {
            let epoch = ProcessEpoch(next(&EPOCHS).unwrap());
            let model = Digest([3; 32]);
            let encoding = Digest(pals_fresh_encoding_semantic_digest());
            let clock = ContractSystemClock::new(epoch);
            let scope = SharedPalsScope::new(PalsScope {
                authority: SearchAuthority {
                    epoch,
                    game: GameGeneration(1),
                    root: RootGeneration(0),
                    implementation: Digest(pals_native_source_digest()),
                },
                model,
                encoding,
                precision: PrecisionProfile::Fp32,
                frozen_epoch: DEPLOYMENT_FROZEN_EPOCH,
                mode: ExecutionMode::Deployment,
            });
            let owner = Arc::new(WorkerOwner {
                worker: Mutex::new(SingleWorker::spawn_with_outcome(run).unwrap()),
                physical_completed: AtomicU64::new(0),
                completed: AtomicU64::new(0),
                physical_failed: AtomicU64::new(0),
                validation_failed: AtomicU64::new(0),
                delivered: AtomicU64::new(0),
                consumed: AtomicU64::new(0),
                canceled: AtomicU64::new(0),
                expired: AtomicU64::new(0),
                new_game_resets: AtomicU64::new(0),
                in_flight: AtomicU64::new(0),
                game_generation: AtomicU64::new(1),
                pending_game_target: AtomicU64::new(0),
                game_lifecycle: Mutex::new(()),
                game_reset_failure: Mutex::new(None),
                request_high_water: AtomicU64::new(0),
                execution_high_water: AtomicU64::new(0),
                quarantined: AtomicBool::new(false),
                shutdown: AtomicBool::new(false),
                model_epoch: [3; 32],
                model_config: PalsModelConfig::baseline(),
                manifest_digest: [3; 32],
                encoding_semantic_digest: encoding.0,
                adapter_source_digest: pals_native_source_digest(),
                trained: false,
                finishing: AtomicBool::new(false),
                process_epoch: epoch,
                _registered_sources: None,
                last_failure: Mutex::new(None),
                final_stats: Mutex::new(None),
                stats_lease: Mutex::new(None),
                additional_host_bytes: REQUEST_BYTES,
                cuda_record_page_snapshot: Mutex::new(None),
                cuda_record_page_snapshot_error: Mutex::new(None),
                host_record_page_observer: None,
                applied_session_settings: None,
                residency: PalsSessionResidency {
                    graphs: vec![],
                    native_sessions: 0,
                    layout: "native-worker-fixture".into(),
                    reader_initializer_bank: None,
                    role_reader_weights_shared: None,
                    native_resident_parameter_bytes: None,
                    vram_peak_bytes: None,
                },
                execution: execution.unwrap_or(NativeExecutionReceipt {
                    provider: "cpu",
                    device_id: None,
                    session_arena_bytes: None,
                    runtime_sha256: [0; 32],
                    runtime_bundle_sha256: None,
                    cuda_control_inventory_sha256: None,
                    cuda_loading_profile: None,
                    cudnn_conv_search_diagnostic: None,
                    startup_probe_timeout_ms: None,
                    transient_request_device_bytes: 0,
                    transient_execution_device_bytes: 0,
                    pinned_request_bytes: 0,
                    device_public_memory: false,
                    host_record_pages: None,
                    private_warm: None,
                    cuda_record_pages: None,
                }),
                startup_probe: Mutex::new(None),
                startup_probe_timeout_ms: AtomicU64::new(0),
                startup_stage_probe: None,
                startup_ready: AtomicBool::new(true),
                final_mapping_confirmed: AtomicBool::new(false),
                final_loading_mapping: Mutex::new(None),
                observer: Mutex::new(None),
                observer_failures: AtomicU64::new(0),
                last_observer_failure: Mutex::new(None),
                role_execution: Mutex::new(None),
                followup_evidence: AtomicBool::new(false),
                prepared_perspective: Mutex::new(None),
                private_warm: None,
                warm_admission: Mutex::new(None),
            });
            let runtime = PalsRuntime::new(
                PalsAdapter::new(scope.clone(), clock.clone()).unwrap(),
                WorkerBackend {
                    owner: Arc::clone(&owner),
                },
                Limits {
                    max_requests: 1,
                    max_batch_items: 1,
                    max_executions: 1,
                    max_batch_wait: Duration::ZERO,
                    max_queue_age: Duration::from_secs(2),
                    deadline_reserve: Duration::ZERO,
                    memory: Resources {
                        host_bytes: 4 * REQUEST_BYTES,
                        device_bytes: 0,
                        pinned_bytes: 0,
                    },
                },
                64,
            )
            .unwrap();
            NativeRoleModel {
                identity: "pals-native-lifecycle-fixture".into(),
                value_identity: frontier_value_identity(
                    "pals-native-lifecycle-fixture",
                    [3; 32],
                    encoding,
                ),
                epoch,
                model_epoch: [3; 32],
                model,
                encoding,
                clock,
                scope,
                runtime,
                owner,
                sequence: 0,
                game: 1,
                unusable: false,
                drain_limit: Duration::from_secs(1),
                pending_new_game: false,
                control_lease: None,
                startup_attempted: false,
                startup_probe_configured: false,
                delivered_request: None,
                last_value_execution: None,
                selected_review: None,
                ticketed_delivered: Vec::with_capacity(16),
                startup_loading_elapsed_ns: None,
                startup_clock: None,
                startup_diagnostic: None,
                cpu_invocation_budget: None,
                cpu_invocation_load_interval: None,
            }
        }
        fn output(input: &PalsModelInput) -> PalsNativeResult {
            PalsNativeResult::Evaluation(PalsRawOutput {
                candidate_logits: vec![0.0; input.candidates.len()],
                wdl_logits: [0.0; 3],
                divergence_logits: (input.role == PalsRole::Critic)
                    .then(|| vec![0.0; input.divergence_features.len()]),
                task_logits: None,
                private_latent: vec![0.0; PalsModelConfig::baseline().latent_elements()],
            })
        }
        /// These fixtures exercise the new lease guards with a mock physical
        /// worker; they do not attest actual ORT inference or source capture.
        fn recorded_fixture_model(
            canceled_on_evaluate: Option<Arc<AtomicBool>>,
            invalid_raw: bool,
        ) -> NativeRoleModel {
            let mut stats = PalsBackendStats::default();
            fixture_model(move |command| match command {
                PalsNativeCommand::Evaluate(input)
                | PalsNativeCommand::EvaluateWithEvidence(input) => {
                    stats.admitted_role_requests += 1;
                    stats.public_cache_misses += 1;
                    stats.public_nn_runs_attempted += 1;
                    stats.public_nn_runs_completed += 1;
                    stats.role_nn_runs_attempted += 1;
                    stats.role_nn_runs_completed += 1;
                    stats.completed_nn_inputs += 2;
                    stats.validated_public_outputs += 1;
                    stats.validated_role_outputs += 1;
                    if let Some(cancel) = &canceled_on_evaluate {
                        cancel.store(true, Ordering::Release);
                    }
                    let mut result = output(input);
                    if invalid_raw && let PalsNativeResult::Evaluation(raw) = &mut result {
                        raw.private_latent.clear();
                    }
                    PhysicalRun::Complete(Ok(result))
                }
                PalsNativeCommand::SnapshotStats => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::Stats(stats.clone())))
                }
                PalsNativeCommand::VerifyRuntime => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::RuntimeVerified))
                }
                PalsNativeCommand::NewGame => PhysicalRun::Complete(Ok(PalsNativeResult::NewGame)),
                PalsNativeCommand::ResetTo(target) => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::ResetTo {
                        game_generation: *target,
                    }))
                }
                _ => unexpected_cuda_placement(),
            })
        }
        fn recorded_fixture_budget() -> NativeInvocationBudget {
            let started = Instant::now();
            let execution = started + Duration::from_secs(2);
            NativeInvocationBudget::new(
                started,
                execution,
                execution + Duration::from_secs(1),
                Duration::from_secs(1),
            )
            .unwrap()
        }
        fn recorded_fixture_input(cancel: &AtomicBool) -> PalsModelInput {
            let position = Position::startpos();
            let legal = position.legal_moves();
            let mut input = prepare_role_input(
                &query(&position, &legal, cancel),
                NativeQueryKind::Propose,
                [3; 32],
            )
            .unwrap();
            // Historical data deliberately differs from the current live E.
            input.query[7] = 123.25;
            input
        }
        #[test]
        fn recorded_witness_requires_the_original_actual_factory_budget() {
            let cancel = AtomicBool::new(false);
            let budget = recorded_fixture_budget();
            let mut model = recorded_fixture_model(None, false);
            let input = recorded_fixture_input(&cancel);
            let key = input
                .canonical_input_key(&PalsModelConfig::baseline())
                .unwrap();
            assert!(
                model
                    .witness_recorded_cpu_input(input.clone(), key, budget, &cancel)
                    .is_err()
            );
            assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 0);
            let mut model = accept_invocation_owner(model, budget, &cancel).unwrap();
            let replacement = recorded_fixture_budget();
            assert!(
                model
                    .witness_recorded_cpu_input(input, key, replacement, &cancel)
                    .is_err()
            );
            assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 0);
            let receipt = model.finish_handle().finish(budget.whole_until()).unwrap();
            assert_eq!(receipt.backend_stats.unwrap().completed_nn_inputs, 0);
        }
        #[test]
        fn recorded_pristine_gate_rejects_generic_owner_and_cancel_before_control() {
            let mut model = fixture_model(binding_fixture_command);
            let budget = recorded_fixture_budget();
            let cancel = AtomicBool::new(false);
            assert!(matches!(
                model.witness_recorded_cpu_pristine(budget, &cancel),
                Err(RoleError::Backend(_))
            ));
            assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 0);
            assert!(model.control_lease.is_none());
            cancel.store(true, Ordering::Release);
            assert!(matches!(
                model.witness_recorded_cpu_pristine(budget, &cancel),
                Err(RoleError::Canceled)
            ));
            let done = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert_eq!(done.physically_completed_role_calls, 0);
            assert_eq!(done.backend_stats.unwrap().completed_nn_inputs, 0);
        }
        #[test]
        fn recorded_pristine_counters_refuse_any_previous_graph_or_cache_work() {
            let zero: NativeBackendStatsReceipt = PalsBackendStats::default().into();
            assert!(require_recorded_nn_zero(&zero).is_ok());
            for mutation in 0..14 {
                let mut stats = zero.clone();
                match mutation {
                    0 => stats.admitted_role_requests = 1,
                    1 => stats.public_cache_hits = 1,
                    2 => stats.public_cache_misses = 1,
                    3 => stats.public_nn_runs_attempted = 1,
                    4 => stats.public_nn_runs_completed = 1,
                    5 => stats.public_nn_runs_failed_known = 1,
                    6 => stats.role_nn_runs_attempted = 1,
                    7 => stats.role_nn_runs_completed = 1,
                    8 => stats.role_nn_runs_failed_known = 1,
                    9 => stats.completed_nn_inputs = 1,
                    10 => stats.validated_public_outputs = 1,
                    11 => stats.validated_role_outputs = 1,
                    12 => stats.new_game_resets = 1,
                    _ => stats.live_public_cache_entries = 1,
                }
                assert!(require_recorded_nn_zero(&stats).is_err());
            }
        }
        #[test]
        fn recorded_witness_preserves_historical_input_bits_and_actual_graph_work() {
            let cancel = AtomicBool::new(false);
            let budget = recorded_fixture_budget();
            let model = recorded_fixture_model(None, false);
            let mut model = accept_invocation_owner(model, budget, &cancel).unwrap();
            let input = recorded_fixture_input(&cancel);
            let key = input
                .canonical_input_key(&PalsModelConfig::baseline())
                .unwrap();
            let witness = model
                .witness_recorded_cpu_input(input, key, budget, &cancel)
                .unwrap();
            assert_eq!(
                witness.historical_deadline_feature_bits(),
                123.25f32.to_bits()
            );
            assert_eq!(witness.input().query[7].to_bits(), 123.25f32.to_bits());
            assert_eq!(witness.input_key(), key);
            assert_eq!(witness.invocation_budget(), budget);
            assert!(witness.work_started_at() >= budget.started());
            assert!(witness.work_finished_at() < budget.execution_until());
            assert!(witness.physical_completion_confirmed());
            assert_eq!(witness.backend_stats_before().completed_nn_inputs, 0);
            assert_eq!(witness.backend_stats_after().completed_nn_inputs, 2);
            assert!(witness.raw_output_bit_projection().len() > 32 * 1024);
            let receipt = model.finish_handle().finish(budget.whole_until()).unwrap();
            assert!(receipt.physical_shutdown_confirmed);
            assert_eq!(receipt.physically_completed_role_calls, 0);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
            assert!(receipt.startup_probe.is_none());
            assert_eq!(receipt.backend_stats.unwrap().completed_nn_inputs, 2);
        }
        #[test]
        fn recorded_witness_refuses_changed_bits_epoch_and_nonfinite_before_admission() {
            let cancel = AtomicBool::new(false);
            let budget = recorded_fixture_budget();
            let model = recorded_fixture_model(None, false);
            let mut model = accept_invocation_owner(model, budget, &cancel).unwrap();
            let input = recorded_fixture_input(&cancel);
            let key = input
                .canonical_input_key(&PalsModelConfig::baseline())
                .unwrap();
            let mut changed = input.clone();
            changed.query[7] = f32::from_bits(changed.query[7].to_bits() + 1);
            assert!(
                model
                    .witness_recorded_cpu_input(changed, key, budget, &cancel)
                    .is_err()
            );
            let mut changed = input.clone();
            changed.model_epoch = [4; 32];
            assert!(
                model
                    .witness_recorded_cpu_input(changed, key, budget, &cancel)
                    .is_err()
            );
            let mut changed = input;
            changed.query[7] = f32::NAN;
            assert!(
                model
                    .witness_recorded_cpu_input(changed, key, budget, &cancel)
                    .is_err()
            );
            let receipt = model.finish_handle().finish(budget.whole_until()).unwrap();
            assert_eq!(receipt.backend_stats.unwrap().completed_nn_inputs, 0);
        }
        #[test]
        fn recorded_witness_cancel_after_submit_drains_without_issuing_a_capability() {
            let cancel = Arc::new(AtomicBool::new(false));
            let budget = recorded_fixture_budget();
            let model = recorded_fixture_model(Some(Arc::clone(&cancel)), false);
            let mut model = accept_invocation_owner(model, budget, &cancel).unwrap();
            let input = recorded_fixture_input(&cancel);
            let key = input
                .canonical_input_key(&PalsModelConfig::baseline())
                .unwrap();
            assert!(matches!(
                model.witness_recorded_cpu_input(input, key, budget, &cancel),
                Err(RoleError::Canceled)
            ));
            assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 0);
            assert!(model.control_lease.is_none());
            assert!(!model.owner.quarantined.load(Ordering::Acquire));
            let receipt = model.finish_handle().finish(budget.whole_until()).unwrap();
            assert!(receipt.physical_shutdown_confirmed);
            assert_eq!(receipt.backend_stats.unwrap().completed_nn_inputs, 2);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
        }
        #[test]
        fn recorded_witness_invalid_head_preserves_known_physical_completion() {
            let cancel = AtomicBool::new(false);
            let budget = recorded_fixture_budget();
            let model = recorded_fixture_model(None, true);
            let mut model = accept_invocation_owner(model, budget, &cancel).unwrap();
            let input = recorded_fixture_input(&cancel);
            let key = input
                .canonical_input_key(&PalsModelConfig::baseline())
                .unwrap();
            assert!(
                model
                    .witness_recorded_cpu_input(input, key, budget, &cancel)
                    .is_err()
            );
            assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 0);
            assert!(model.control_lease.is_none());
            let receipt = model.finish_handle().finish(budget.whole_until()).unwrap();
            assert!(receipt.physical_shutdown_confirmed);
            assert_eq!(receipt.backend_stats.unwrap().completed_nn_inputs, 2);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
        }
        #[test]
        fn recorded_witness_unknown_completion_retains_the_actual_lease() {
            let cancel = AtomicBool::new(false);
            let budget = recorded_fixture_budget();
            let model = fixture_model(|command| match command {
                PalsNativeCommand::SnapshotStats => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::Stats(PalsBackendStats::default())))
                }
                PalsNativeCommand::Evaluate(_) | PalsNativeCommand::EvaluateWithEvidence(_) => {
                    PhysicalRun::Quarantined(binding_backend_error())
                }
                _ => unexpected_cuda_placement(),
            });
            let mut model = accept_invocation_owner(model, budget, &cancel).unwrap();
            let input = recorded_fixture_input(&cancel);
            let key = input
                .canonical_input_key(&PalsModelConfig::baseline())
                .unwrap();
            assert!(matches!(
                model.witness_recorded_cpu_input(input, key, budget, &cancel),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 1);
            assert!(model.control_lease.is_some());
            assert!(model.owner.quarantined.load(Ordering::Acquire));
            let receipt = model.finish_handle().receipt();
            assert!(!receipt.physical_shutdown_confirmed);
            assert!(!receipt.native_buffers_released);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
        }
        #[test]
        fn recorded_witness_late_ready_is_known_completion_without_capability() {
            let cancel = AtomicBool::new(false);
            let input = recorded_fixture_input(&cancel);
            let key = input
                .canonical_input_key(&PalsModelConfig::baseline())
                .unwrap();
            let mut model = fixture_model(|command| match command {
                PalsNativeCommand::SnapshotStats => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::Stats(PalsBackendStats::default())))
                }
                PalsNativeCommand::Evaluate(input)
                | PalsNativeCommand::EvaluateWithEvidence(input) => {
                    std::thread::sleep(Duration::from_millis(150));
                    PhysicalRun::Complete(Ok(output(input)))
                }
                PalsNativeCommand::VerifyRuntime => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::RuntimeVerified))
                }
                _ => unexpected_cuda_placement(),
            });
            model.drain_limit = Duration::from_millis(500);
            let started = Instant::now();
            let execution = started + Duration::from_millis(100);
            let budget = NativeInvocationBudget::new(
                started,
                execution,
                execution + model.drain_limit,
                model.drain_limit,
            )
            .unwrap();
            let mut model = accept_invocation_owner(model, budget, &cancel).unwrap();
            assert!(matches!(
                model.witness_recorded_cpu_input(input, key, budget, &cancel),
                Err(RoleError::Deadline)
            ));
            assert!(Instant::now() >= budget.execution_until());
            assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 0);
            assert!(model.control_lease.is_none());
            assert!(!model.owner.quarantined.load(Ordering::Acquire));
            let receipt = model.finish_handle().finish(budget.whole_until()).unwrap();
            assert!(receipt.physical_shutdown_confirmed);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
        }
        #[test]
        fn recorded_witness_pending_through_original_whole_fence_keeps_lease() {
            let cancel = AtomicBool::new(false);
            let input = recorded_fixture_input(&cancel);
            let key = input
                .canonical_input_key(&PalsModelConfig::baseline())
                .unwrap();
            let (release, released) = std::sync::mpsc::sync_channel(1);
            let entered = Arc::new(AtomicBool::new(false));
            let worker_entered = Arc::clone(&entered);
            let mut model = fixture_model(move |command| match command {
                PalsNativeCommand::SnapshotStats => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::Stats(PalsBackendStats::default())))
                }
                PalsNativeCommand::Evaluate(input)
                | PalsNativeCommand::EvaluateWithEvidence(input) => {
                    worker_entered.store(true, Ordering::Release);
                    released.recv_timeout(Duration::from_secs(2)).unwrap();
                    PhysicalRun::Complete(Ok(output(input)))
                }
                _ => unexpected_cuda_placement(),
            });
            model.drain_limit = Duration::from_millis(100);
            let started = Instant::now();
            let execution = started + Duration::from_millis(200);
            let budget = NativeInvocationBudget::new(
                started,
                execution,
                execution + model.drain_limit,
                model.drain_limit,
            )
            .unwrap();
            let mut model = accept_invocation_owner(model, budget, &cancel).unwrap();
            assert!(matches!(
                model.witness_recorded_cpu_input(input, key, budget, &cancel),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            assert!(entered.load(Ordering::Acquire));
            assert!(Instant::now() >= budget.whole_until());
            assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 1);
            assert!(model.control_lease.is_some());
            assert!(model.owner.quarantined.load(Ordering::Acquire));
            assert!(model.unusable);
            assert!(!model.finish_handle().receipt().physical_shutdown_confirmed);
            // Release/join only this test's mock after the production W gate.
            // The late Ready never clears quarantine or issues a capability.
            release.send(()).unwrap();
            let test_cleanup_until = Instant::now() + Duration::from_secs(2);
            loop {
                assert!(Instant::now() < test_cleanup_until);
                match model.control_lease.as_mut().unwrap().poll() {
                    PhysicalPoll::Ready(Ok(PalsNativeResult::Evaluation(_))) => break,
                    PhysicalPoll::Pending => std::thread::yield_now(),
                    _ => panic!("released test worker must complete its original job"),
                }
            }
            assert!(model.owner.quarantined.load(Ordering::Acquire));
            assert!(model.unusable);
            loop {
                assert!(Instant::now() < test_cleanup_until);
                match model.owner.worker.lock().unwrap().try_shutdown() {
                    Poll::Ready(Ok(())) => break,
                    Poll::Pending => std::thread::yield_now(),
                    Poll::Ready(Err(error)) => panic!("test-only mock shutdown: {error}"),
                }
            }
            assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 1);
            assert!(!model.finish_handle().receipt().physical_shutdown_confirmed);
        }
        fn query<'a>(
            position: &'a Position,
            legal: &'a [BoardMove],
            cancel: &'a AtomicBool,
        ) -> RoleQuery<'a> {
            RoleQuery {
                position,
                legal,
                prefix: &[],
                proposal: &[],
                counterexample: None,
                records: &[],
                revision: 0,
                deadline: Instant::now() + Duration::from_secs(2),
                cancel,
            }
        }
        #[test]
        fn default_fresh_receipt_has_no_warm_domain_or_observation_fields() {
            let model = fixture_model(|command| match command {
                PalsNativeCommand::Evaluate(input)
                | PalsNativeCommand::EvaluateWithEvidence(input) => {
                    PhysicalRun::Complete(Ok(output(input)))
                }
                _ => unexpected_cuda_placement(),
            });
            let receipt = model.finish_handle().receipt();
            validate_private_warm_evidence(&receipt, false).unwrap();
            // Controlled owner/receipt propagation uses the same authoritative
            // Fresh namespace as production; no second hash algorithm is copied.
            assert_eq!(
                receipt.encoding_semantic_sha256,
                pals_fresh_encoding_semantic_digest()
            );
            assert_eq!(model.encoding.0, receipt.encoding_semantic_sha256);
            let wire = serde_json::to_value(&receipt).unwrap();
            assert!(wire.get("private_warm_observation").is_none());
            assert!(wire.get("private_warm_observation_unavailable").is_none());
            assert!(wire["execution"].get("private_warm").is_none());
            assert_eq!(receipt.game_generation, 1);
        }
        #[test]
        fn native_drop_retains_unconfirmed_seed_handoff_without_fabricating_ready() {
            // Native ownership fixture only. No synthetic loaded Warm capability
            // is constructed and no native graph or model execution is claimed.
            use rz_eval::pals_private::{PrivateModelIdentity, PrivatePrecision};
            use rz_search::pals::store::{LineId, SituationId, StateId};
            let model = fixture_model(|_| unexpected_cuda_placement());
            let core = NativeWarmState::new(
                PrivateModelIdentity {
                    model: [1; 32],
                    encoding: [2; 32],
                    precision: PrivatePrecision::Fp32,
                    model_epoch: [3; 32],
                    frozen_epoch: 1,
                },
                0,
            )
            .unwrap();
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            let query = query(&position, &legal, &cancel);
            let context = RoleLogicalContext {
                game_generation: 0,
                search_generation: 1,
                situation: SituationId {
                    slot: 0,
                    generation: 1,
                },
                state: StateId(0),
                focus: LineId(0),
                purpose: RoleQueryPurpose::ProposePolicy,
                prefix: vec![],
                focus_sha256: [4; 32],
                prefix_sha256: [5; 32],
                proposal_sha256: [6; 32],
                refutation_sha256: None,
                divergence_sha256: [7; 32],
                public_revision: 0,
                situation_revision: 0,
            };
            let input = prepare_role_input(&query, NativeQueryKind::Propose, [3; 32]).unwrap();
            let prepared = core
                .prepare(
                    RequestId::new(ProcessEpoch(1), 1),
                    input,
                    position.snapshot(),
                    &context,
                    query.deadline,
                    CancelToken::new(),
                    &cancel,
                )
                .unwrap();
            let input = Arc::downgrade(&prepared.input);
            let physical = core.begin(prepared).unwrap();
            *model.owner.warm_admission.lock().unwrap() = Some(WarmAdmission {
                state: core.clone(),
                payload: None,
                physical: Some(physical),
            });
            drop(model);
            let observation = core.snapshot().unwrap();
            assert!(observation.quarantined && observation.retained_prepared_input);
            assert_eq!(observation.known_seed_completions, 0);
            assert_eq!(observation.accepted_seeds, 0);
            assert!(input.upgrade().is_some());
            assert!(core.reset_after_known_fence(1).is_err());
        }
        #[test]
        fn frontier_value_preserves_actual_prepared_key_and_single_consumption() {
            let keys = Arc::new(Mutex::new(Vec::new()));
            let observed = Arc::clone(&keys);
            let mut model = fixture_model(move |command| match command {
                PalsNativeCommand::Evaluate(input)
                | PalsNativeCommand::EvaluateWithEvidence(input) => {
                    observed.lock().unwrap().push(
                        input
                            .canonical_input_key(&PalsModelConfig::baseline())
                            .unwrap(),
                    );
                    PhysicalRun::Complete(Ok(output(input)))
                }
                PalsNativeCommand::SnapshotStats => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::Stats(PalsBackendStats::default())))
                }
                _ => panic!("unexpected frontier fixture command"),
            });
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            let identity = model.value_identity().unwrap().clone();
            assert_eq!(identity.model, model.identity());
            assert_eq!(identity.model_epoch, [3; 32]);
            assert_eq!(identity.precision, "fp32");
            let mut delivered = Vec::new();
            for revision in [0, 1] {
                let mut request = query(&position, &legal, &cancel);
                request.revision = revision;
                let value = model.evaluate_value(request).unwrap();
                value.validate(&position, &identity).unwrap();
                assert_eq!(value.perspective, Color::White);
                delivered.push(value.input_sha256);
                assert_eq!(
                    model.owner.consumed.load(Ordering::Acquire),
                    revision,
                    "delivery is not search consumption",
                );
                model.accepted_output();
                model.accepted_output();
                assert_eq!(model.owner.consumed.load(Ordering::Acquire), revision + 1);
                assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 0);
            }
            assert_eq!(*keys.lock().unwrap(), delivered);
            assert_ne!(
                delivered[0], delivered[1],
                "revision is actual input identity"
            );
            assert_eq!(model.owner.completed.load(Ordering::Acquire), 2);
            model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
        }

        #[test]
        fn followup_value_evidence_joins_prepared_rules_perspective_and_actual_nn_delta() {
            // Mock physical worker evidence only; no ORT/CUDA execution is
            // claimed. The real submit/Ready path supplies the captured binding.
            use rz_search::pals::store::{LineId, SituationId, StateId};
            for case in 0..15 {
                let delta = match case {
                    1 => None,
                    2 => Some(0),
                    3 => Some(3),
                    14 => Some(1),
                    _ => Some(2),
                };
                let mut model = fixture_model(move |command| match command {
                    PalsNativeCommand::EvaluateWithEvidence(input) => {
                        let PalsNativeResult::Evaluation(raw) = output(input) else {
                            unreachable!();
                        };
                        PhysicalRun::Complete(Ok(PalsNativeResult::EvaluationWithEvidence {
                            outcome: Ok(raw),
                            actual_completed_nn_inputs_delta: delta,
                            complete: true,
                            owner_accounting: None,
                        }))
                    }
                    PalsNativeCommand::SnapshotStats => PhysicalRun::Complete(Ok(
                        PalsNativeResult::Stats(PalsBackendStats::default()),
                    )),
                    _ => panic!("unexpected followup value fixture command"),
                });
                model.enable_followup_execution_evidence().unwrap();
                let position = Position::startpos();
                let snapshot = position.snapshot();
                let legal = position.legal_moves();
                let cancel = AtomicBool::new(false);
                let context = RoleLogicalContext {
                    game_generation: model.game,
                    search_generation: 1,
                    situation: SituationId {
                        slot: 0,
                        generation: 1,
                    },
                    state: StateId(0),
                    focus: LineId(0),
                    purpose: RoleQueryPurpose::ValueFresh,
                    prefix: vec![],
                    focus_sha256: [4; 32],
                    prefix_sha256: [5; 32],
                    proposal_sha256: [6; 32],
                    refutation_sha256: None,
                    divergence_sha256: [7; 32],
                    public_revision: 0,
                    situation_revision: 0,
                };
                let mut value = model
                    .evaluate_value_with_context(query(&position, &legal, &cancel), &context)
                    .unwrap();
                {
                    let mut slot = model.owner.role_execution.lock().unwrap();
                    let observed = slot.as_mut().unwrap();
                    match case {
                        4 => observed.prepared_state.semantic.0[0] ^= 1,
                        5 => observed.prepared_state.revision.0 += 1,
                        6 => observed.prepared_perspective = Some(rz_contracts::Color::Black),
                        7 => observed.input_sha256[0] ^= 1,
                        8 => observed.binding.execution.epoch.0 += 1,
                        9 => observed.fresh = Some(false),
                        10 => observed.unknown_observed = true,
                        11 => observed.measurement_complete = false,
                        12 => {
                            let mut other = position.clone();
                            other
                                .make_move(BoardMove::from_uci("e2e4").unwrap())
                                .unwrap();
                            let other_state = other.position_identity();
                            assert_ne!(other_state, value.state, "case {case}");
                            value.state = other_state;
                        }
                        13 => value.perspective = Color::Black,
                        _ => {}
                    }
                }
                model.capture_value_execution(&context, &value, &snapshot);
                let evidence = model.last_value_execution_evidence();
                if matches!(case, 4..=8 | 12..=13) {
                    assert!(evidence.is_none(), "case {case}");
                } else {
                    let evidence = evidence.unwrap();
                    assert_eq!(evidence.complete, matches!(case, 0 | 14), "case {case}");
                    assert_eq!(evidence.completed_nn_inputs, delta);
                    assert_eq!(evidence.physically_completed, Some(true));
                }
                model.accepted_output();
                model
                    .finish_handle()
                    .finish(Instant::now() + Duration::from_secs(2))
                    .unwrap();
            }
        }

        #[test]
        fn canceled_or_expired_frontier_value_never_dispatches_or_consumes() {
            let mut model = fixture_model(|command| match command {
                PalsNativeCommand::SnapshotStats => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::Stats(PalsBackendStats::default())))
                }
                _ => panic!("canceled frontier dispatched physical work"),
            });
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(true);
            assert!(matches!(
                model.evaluate_value(query(&position, &legal, &cancel)),
                Err(RoleError::Canceled),
            ));
            cancel.store(false, Ordering::Release);
            let mut request = query(&position, &legal, &cancel);
            request.deadline = Instant::now();
            assert!(matches!(
                model.evaluate_value(request),
                Err(RoleError::Deadline)
            ));
            assert_eq!(model.owner.completed.load(Ordering::Acquire), 0);
            assert_eq!(model.owner.consumed.load(Ordering::Acquire), 0);
            assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 0);
            model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
        }

        #[test]
        fn transient_cuda_reservations_follow_bounded_tensor_layout() {
            let (request, execution) =
                transient_device_layout_for_config(&PalsModelConfig::baseline()).unwrap();
            assert!(request > 0 && execution > request);
            assert!(request + execution < REQUEST_BYTES);
            // A transient tensor reservation never becomes a session/peak claim.
            assert!(execution < 2 * 1024 * 1024 * 1024);
        }
        #[test]
        fn cuda_warm_load_admission_charges_seed_and_old_new_public_kv() {
            let config = PalsModelConfig::full_line_interaction_v2();
            let ordinary = transient_device_layout_for_config(&config).unwrap();
            assert_eq!(ordinary, (860033, 224974));
            assert_eq!(
                cuda_warm_tensor_layout_for_config(&config, false).unwrap(),
                ordinary
            );
            let warm = cuda_warm_tensor_layout_for_config(&config, true).unwrap();
            assert_eq!(warm, (884610, 423630));
            // CPU arithmetic fixture, never a loaded capability/GPU witness.
            let (mut execution, _) = loading_mapping_fixture();
            execution.transient_request_device_bytes = warm.0;
            execution.transient_execution_device_bytes = warm.1;
            let limits = NativeCudaWarmObservationLimits {
                max_leases: 1,
                device_bytes_max: 397312,
            };
            assert!(validate_cuda_warm_observation_admission(limits, &execution, true).is_ok());
            assert_eq!(execution.execution_device_reservation(), warm.1);
            assert_eq!(
                execution.runtime_device_reservation().unwrap(),
                warm.0 + warm.1
            );
            for (limits, capability) in [
                (
                    NativeCudaWarmObservationLimits {
                        max_leases: 2,
                        ..limits
                    },
                    true,
                ),
                (
                    NativeCudaWarmObservationLimits {
                        device_bytes_max: 0,
                        ..limits
                    },
                    true,
                ),
                (
                    NativeCudaWarmObservationLimits {
                        device_bytes_max: warm.1 + 1,
                        ..limits
                    },
                    true,
                ),
                (limits, false),
            ] {
                let cause =
                    validate_cuda_warm_observation_admission(limits, &execution, capability)
                        .unwrap_err();
                assert_eq!(cause.stage, rz_eval::error::FailureStage::Admission);
                assert_eq!(cause.kind, rz_eval::error::FailureKind::InvalidInput);
            }
            execution.transient_execution_device_bytes = ordinary.1;
            let cause =
                validate_cuda_warm_observation_admission(limits, &execution, true).unwrap_err();
            assert_eq!(
                cause.detail,
                "CUDA Warm observation payload limit exceeds the charged physical tensor reservation"
            );
        }
        fn loading_mapping_fixture() -> (NativeExecutionReceipt, PalsNativeMappingWitness) {
            let profile = rz_eval::onnx::NativeLoadingProfile::CuDnnShimLazyV1;
            let digest = rz_eval::asset::sha256(profile.canonical_descriptor().as_bytes());
            let execution = NativeExecutionReceipt {
                provider: "cuda",
                device_id: Some(0),
                session_arena_bytes: Some(2 * 1024 * 1024 * 1024),
                runtime_sha256: [1; 32],
                runtime_bundle_sha256: Some([2; 32]),
                cuda_control_inventory_sha256: Some([3; 32]),
                cudnn_conv_search_diagnostic: None,
                cuda_loading_profile: Some(NativeCudaLoadingIdentity {
                    profile: profile.identifier(),
                    canonical_sha256: digest,
                }),
                startup_probe_timeout_ms: None,
                transient_request_device_bytes: 0,
                transient_execution_device_bytes: 0,
                pinned_request_bytes: 0,
                device_public_memory: false,
                host_record_pages: None,
                private_warm: None,
                cuda_record_pages: None,
            };
            let roots: Vec<_> = profile
                .eager_indices()
                .iter()
                .map(|&index| rz_eval::onnx::NVIDIA_LOAD_ORDER[index].to_owned())
                .collect();
            let absent: Vec<_> = rz_eval::onnx::NVIDIA_LOAD_ORDER
                .iter()
                .filter(|name| !roots.iter().any(|root| root.as_str() == **name))
                .map(|name| (*name).to_owned())
                .collect();
            let witness = PalsNativeMappingWitness {
                schema: "rovezero.pals-native-mapping-witness.v1",
                runtime_sha256: execution.runtime_sha256,
                runtime_bundle_sha256: execution.runtime_bundle_sha256.unwrap(),
                loading_profile: profile.identifier(),
                loading_profile_sha256: digest,
                scope: "exclusive_physical_worker_full_runtime_origin",
                declared_nvidia_files: 16,
                required_nvidia_files: roots.clone(),
                mapped_nvidia_files: roots,
                deferred_nvidia_not_mapped: absent,
                mapped_ort_files: rz_eval::onnx::ORT_LIBRARY_NAMES
                    .iter()
                    .map(|name| (*name).to_owned())
                    .collect(),
            };
            (execution, witness)
        }
        fn require_probe(model: &NativeRoleModel) {
            model.owner.startup_ready.store(false, Ordering::Release);
            *model.owner.startup_probe.lock().unwrap() = Some(NativeStartupProbeReceipt {
                completed_proposer_calls: 0,
                completed_critic_calls: 0,
                runtime_mapping_confirmed: false,
                runtime_loading_mapping: None,
                cuda_placement_witness: None,
                reset_completed: false,
                backend_stats: None,
                failure_diagnostic: None,
                backend_stages: None,
                timing_observation: None,
                cuda_record_page_observation: None,
                cuda_record_page_observation_error: None,
            });
        }
        #[test]
        fn startup_input_completion_requires_retained_owner_key_ready_decode_and_profile() {
            use rz_eval::pals_onnx::{PalsKernelWitness, PalsStartupInputObservation};
            // CPU metadata fixtures only: no session, provider, Tensor or NN is
            // created, and their synthetic profile pins are not GPU evidence.
            let observed = PalsStartupInputObservation {
                request_ordinal: 1,
                role: PalsRole::Proposer,
                backend_owner_id: Some(7),
                input_key: [9; 32],
                prepared_input_key: [9; 32],
                public_memory_key: [8; 32],
                logical_records: 0,
                prepared_records: 1,
                record_mask: Some(vec![false]),
                board: None,
                records: None,
                record_mask_tensor: None,
                memory_key: None,
                memory_value: None,
                memory_mask: None,
                public_cache_hit: Some(false),
                public_run_returned_ok: Some(true),
                private_run_returned_ok: Some(true),
                role_evaluation_returned_ok: Some(true),
                completion_binding: None,
                completion_binding_error: None,
            };
            let snapshot = PalsStartupStageSnapshot {
                schema: "rovezero.pals-startup-backend-stages.v1",
                clock_scope: "caller_supplied_monotonic_startup_origin",
                snapshot_status: PalsStartupSnapshotStatus::Available,
                capture_started: Some(true),
                capture_closed: true,
                snapshot_elapsed_ns: Some(1),
                captured_requests: 1,
                max_requests: 2,
                max_events: 64,
                overflow: false,
                recording_contended: false,
                recording_poisoned: false,
                events: vec![],
                input_observations: Some(vec![observed]),
            };
            let command = NativeStartupCommandObservation {
                command: "evaluate_cuda_warm",
                admitted: true,
                entered_elapsed_ns: Some(1),
                returned_elapsed_ns: Some(2),
                physical_completion_confirmed: Some(true),
                return_error: None,
                fence_observation: None,
                input_observation: Some(NativeStartupInputObservation {
                    startup_request_ordinal: Some(1),
                    backend_owner_id: Some(7),
                    role: PalsRole::Proposer,
                    input_key: [9; 32],
                    logical_records: 0,
                    retained_input_key: Some([9; 32]),
                    raw_decode_succeeded: Some(true),
                }),
            };
            let kernel = PalsKernelWitness {
                profile_sha256: [5; 32],
                cuda_kernels: 1,
                cuda_transfer_kernels: 0,
                approved_cpu_control_kernels: 0,
                neural_kernels: 1,
                proposer_private_kernels: 1,
                critic_private_kernels: 1,
            };
            let witness = PalsCudaPlacementWitness {
                schema: "synthetic-metadata-only-test".into(),
                inventory_sha256: [3; 32],
                manifest_sha256: [4; 32],
                runtime_sha256: [1; 32],
                runtime_bundle_sha256: [2; 32],
                optimization: PalsGraphOptimization::Disable,
                initialization: vec![],
                public: kernel.clone(),
                shared_pc: kernel,
                category_provenance: "synthetic-metadata-only-test".into(),
            };
            let mut completed = snapshot.clone();
            bind_startup_input_observations(
                &mut completed,
                std::slice::from_ref(&command),
                Some(&witness),
            );
            let input = &completed.input_observations.as_ref().unwrap()[0];
            let bound = input.completion_binding.as_ref().unwrap();
            assert_eq!(bound.retained_input_key, [9; 32]);
            assert_eq!(bound.backend_owner_id, 7);
            assert_eq!(bound.public_profile_sha256, witness.public.profile_sha256);
            assert!(input.completion_binding_error.is_none());
            assert!(input.memory_key.is_none()); // Never impute a shape from completion.

            for mutation in 0..20 {
                let mut snapshot = snapshot.clone();
                let mut command = command.clone();
                let input = &mut snapshot.input_observations.as_mut().unwrap()[0];
                let identity = command.input_observation.as_mut().unwrap();
                match mutation {
                    0 => command.admitted = false,
                    1 => command.physical_completion_confirmed = None,
                    2 => command.return_error = Some(NativeStartupErrorKind::Backend),
                    3 => identity.backend_owner_id = Some(8),
                    4 => identity.startup_request_ordinal = Some(2),
                    5 => identity.input_key = [10; 32],
                    6 => identity.retained_input_key = Some([10; 32]),
                    7 => identity.raw_decode_succeeded = Some(false),
                    8 => input.private_run_returned_ok = Some(false),
                    9 => snapshot.recording_contended = true,
                    10 => {} // Missing profile ACK below.
                    11 => input.prepared_input_key = [10; 32],
                    12 => input.backend_owner_id = None,
                    13 => identity.raw_decode_succeeded = None,
                    14 => identity.logical_records = 1,
                    15 => identity.role = PalsRole::Critic,
                    16 => identity.retained_input_key = None,
                    17 => input.role_evaluation_returned_ok = None,
                    18 => {
                        input.public_cache_hit = Some(false);
                        input.public_run_returned_ok = None;
                    }
                    19 => snapshot.snapshot_status = PalsStartupSnapshotStatus::Contended,
                    _ => unreachable!(),
                }
                let profile = (mutation != 10).then_some(&witness);
                bind_startup_input_observations(&mut snapshot, &[command], profile);
                let input = &snapshot.input_observations.unwrap()[0];
                assert!(input.completion_binding.is_none(), "mutation {mutation}");
                assert!(
                    input.completion_binding_error.is_some(),
                    "mutation {mutation}"
                );
                assert_eq!(input.record_mask, Some(vec![false]));
            }
            let mut ambiguous = snapshot.clone();
            bind_startup_input_observations(
                &mut ambiguous,
                &[command.clone(), command],
                Some(&witness),
            );
            assert!(
                ambiguous.input_observations.unwrap()[0]
                    .completion_binding
                    .is_none()
            );
        }
        #[test]
        fn startup_probe_explicit_budget_preserves_default_wire_and_bounds() {
            for selected in [None, Some(1), Some(15_000), Some(120_000), Some(180_000)] {
                let mut model = fixture_model(|command| match command {
                    PalsNativeCommand::SnapshotStats => PhysicalRun::Complete(Ok(
                        PalsNativeResult::Stats(PalsBackendStats::default()),
                    )),
                    _ => unexpected_cuda_placement(),
                });
                let budget = model.configure_startup_probe_timeout(selected).unwrap();
                assert_eq!(budget, Duration::from_millis(selected.unwrap_or(15_000)));
                let execution = model.finish_handle().receipt().execution;
                assert_eq!(execution.startup_probe_timeout_ms, selected);
                let encoded = serde_json::to_value(execution).unwrap();
                assert_eq!(
                    encoded.get("startup_probe_timeout_ms").is_some(),
                    selected.is_some()
                );
                assert!(model.configure_startup_probe_timeout(selected).is_err());
                model
                    .finish_handle()
                    .finish(Instant::now() + Duration::from_secs(1))
                    .unwrap();
            }
            for selected in [0, 180_001, u64::MAX] {
                let mut model = fixture_model(|command| match command {
                    PalsNativeCommand::SnapshotStats => PhysicalRun::Complete(Ok(
                        PalsNativeResult::Stats(PalsBackendStats::default()),
                    )),
                    _ => unexpected_cuda_placement(),
                });
                assert!(matches!(
                    model.configure_startup_probe_timeout(Some(selected)),
                    Err(RoleError::InvalidOutput)
                ));
                assert_eq!(
                    model
                        .finish_handle()
                        .receipt()
                        .execution
                        .startup_probe_timeout_ms,
                    None
                );
                model
                    .finish_handle()
                    .finish(Instant::now() + Duration::from_secs(1))
                    .unwrap();
            }
        }
        #[test]
        fn failed_startup_publishes_partial_mapping_evidence_without_masking_primary() {
            use crate::pals_attestation::{PalsReceiptWriter, STARTUP_FILE, TERMINATION_FILE};
            let (execution, witness) = loading_mapping_fixture();
            let mut model = fixture_model_with_execution(
                move |command| match command {
                    PalsNativeCommand::Evaluate(_) | PalsNativeCommand::EvaluateWithEvidence(_) => {
                        PhysicalRun::Complete(Err(BackendError::new(
                            FailureKind::BackendFailure,
                            FailureStage::Backend,
                            "bounded known startup fixture failure",
                        )))
                    }
                    PalsNativeCommand::VerifyRuntime => {
                        PhysicalRun::Complete(Ok(PalsNativeResult::RuntimeVerified))
                    }
                    PalsNativeCommand::ObserveRuntimeMappings => PhysicalRun::Complete(Ok(
                        PalsNativeResult::RuntimeMappingsObserved(Box::new(witness.clone())),
                    )),
                    PalsNativeCommand::SnapshotStats => PhysicalRun::Complete(Ok(
                        PalsNativeResult::Stats(PalsBackendStats::default()),
                    )),
                    _ => unexpected_cuda_placement(),
                },
                Some(execution),
            );
            require_probe(&model);
            let budget = model
                .configure_startup_probe_timeout(Some(120_000))
                .unwrap();
            model
                .observe_startup_loading(Duration::from_millis(3))
                .unwrap();
            let primary = model.prepare_startup(Instant::now() + budget).unwrap_err();
            assert!(matches!(primary, RoleError::Backend(_)));
            let finish = model.finish_handle();
            let partial = finish.receipt();
            let diagnostic = partial
                .startup_probe
                .as_ref()
                .unwrap()
                .failure_diagnostic
                .as_ref()
                .unwrap();
            assert_eq!(diagnostic.model_loading_elapsed_ns, Some(3_000_000));
            assert_eq!(
                diagnostic.primary_error,
                Some(NativeStartupErrorKind::Backend)
            );
            assert_eq!(diagnostic.current_stage, "proposer_evaluation");
            assert_eq!(diagnostic.commands.len(), 1);
            assert!(diagnostic.commands[0].admitted);
            assert_eq!(
                diagnostic.commands[0].physical_completion_confirmed,
                Some(true)
            );
            assert!(diagnostic.commands[0].returned_elapsed_ns.is_some());
            assert!(
                partial
                    .startup_probe
                    .as_ref()
                    .unwrap()
                    .runtime_loading_mapping
                    .is_none()
            );
            assert_eq!(partial.physical_runs_in_flight, 0);
            assert_eq!(partial.completed_role_inputs, 0);
            assert_eq!(partial.search_consumed_role_inputs, 0);
            let root = std::env::temp_dir().join(format!(
                "rz-pals-failed-startup-{}-{}",
                std::process::id(),
                EPOCHS.fetch_add(1, Ordering::AcqRel)
            ));
            std::fs::create_dir(&root).unwrap();
            // This mock worker receipt has a static fixture identity, independent
            // of the debug test executable's size. Production open still observes
            // the actual executable under its original 128 MiB bound.
            let mut writer =
                PalsReceiptWriter::open_static_identity_fixture(&root, [1; 32]).unwrap();
            assert_eq!(
                writer.startup(partial.clone(), None).unwrap_err().stage,
                "selected PALS CUDA loading profile requires an actual startup mapping ACK"
            );
            writer.failed_startup(partial, &primary, None).unwrap();
            let ended = finish
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert!(ended.physical_shutdown_confirmed && ended.native_buffers_released);
            assert_eq!(
                writer
                    .termination(ended.clone(), true, None)
                    .unwrap_err()
                    .stage,
                "failed PALS startup cannot publish a successful service termination"
            );
            writer.termination(ended, false, None).unwrap();
            drop(writer);
            let slot = root.join(crate::process_receipts::process_run_id());
            for file in [STARTUP_FILE, TERMINATION_FILE] {
                let value: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(slot.join(file)).unwrap()).unwrap();
                assert_eq!(value["endpoint_id"], "static-pals-receipt-fixture");
                assert_eq!(value["startup_failure"], "backend");
                assert_eq!(value["service_exit_success"], false);
                assert_eq!(
                    value["native"]["execution"]["startup_probe_timeout_ms"],
                    120_000
                );
                assert_eq!(
                    value["native"]["startup_probe"]["failure_diagnostic"]["primary_error"],
                    "backend"
                );
                std::fs::remove_file(slot.join(file)).unwrap();
            }
            std::fs::remove_dir(slot).unwrap();
            std::fs::remove_dir(root).unwrap();
        }
        #[test]
        fn pending_startup_fence_keeps_lease_and_unknown_completion_in_partial_evidence() {
            let mut model = fixture_model(|command| match command {
                PalsNativeCommand::Evaluate(input)
                | PalsNativeCommand::EvaluateWithEvidence(input) => {
                    std::thread::sleep(Duration::from_millis(200));
                    PhysicalRun::Complete(Ok(output(input)))
                }
                _ => unexpected_cuda_placement(),
            });
            require_probe(&model);
            model.configure_startup_probe_timeout(Some(100)).unwrap();
            model.drain_limit = Duration::from_millis(5);
            let result = model.prepare_startup(Instant::now() + Duration::from_millis(100));
            assert!(matches!(result, Err(RoleError::PhysicalCompletionUnknown)));
            let receipt = model.finish_handle().receipt();
            let diagnostic = receipt.startup_probe.unwrap().failure_diagnostic.unwrap();
            assert_eq!(
                diagnostic.primary_error,
                Some(NativeStartupErrorKind::PhysicalCompletionUnknown)
            );
            assert!(diagnostic.logical_deadline_expired);
            assert_eq!(diagnostic.commands[0].physical_completion_confirmed, None);
            let input = diagnostic.commands[0].input_observation.as_ref().unwrap();
            assert_eq!(input.startup_request_ordinal, Some(1));
            assert!(input.retained_input_key.is_none());
            assert!(input.raw_decode_succeeded.is_none());
            assert_eq!(
                diagnostic.commands[0].fence_observation,
                Some("pending_at_registered_physical_fence")
            );
            assert!(diagnostic.commands[0].admitted);
            assert!(
                receipt.quarantined
                    && !receipt.physical_shutdown_confirmed
                    && !receipt.native_buffers_released
            );
            assert_eq!(receipt.physical_runs_in_flight, 1);
            assert!(receipt.last_failure.is_some());
            assert!(model.control_lease.is_some());
            assert!(matches!(
                model
                    .finish_handle()
                    .finish(Instant::now() + Duration::from_secs(1)),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
        }
        #[test]
        fn startup_command_diagnostic_overflow_does_not_change_command_or_prior_events() {
            let mut model = fixture_model(|command| match command {
                PalsNativeCommand::SnapshotStats => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::Stats(PalsBackendStats::default())))
                }
                _ => unexpected_cuda_placement(),
            });
            model.startup_diagnostic = Some(NativeStartupFailureDiagnostic {
                schema: "rz-pals-startup-failure-diagnostic/1",
                model_loading_elapsed_ns: None,
                probe_budget_ns: None,
                probe_elapsed_ns: None,
                drain_limit_ns: None,
                current_stage: "prior_command",
                primary_error: None,
                logical_deadline_expired: false,
                command_observation_overflow: false,
                backend_stages: None,
                commands: vec![
                    NativeStartupCommandObservation {
                        command: "prior_command",
                        admitted: true,
                        entered_elapsed_ns: Some(3),
                        returned_elapsed_ns: Some(7),
                        physical_completion_confirmed: Some(true),
                        return_error: None,
                        fence_observation: None,
                        input_observation: None,
                    };
                    7
                ],
            });
            assert!(matches!(
                model.startup_command(
                    PalsNativeCommand::SnapshotStats,
                    Instant::now() + Duration::from_secs(1)
                ),
                Ok(PalsNativeResult::Stats(_))
            ));
            let diagnostic = model.startup_diagnostic.as_ref().unwrap();
            assert!(diagnostic.command_observation_overflow);
            assert_eq!(diagnostic.commands.len(), 7);
            assert!(
                diagnostic
                    .commands
                    .iter()
                    .all(|event| event.command == "prior_command"
                        && event.returned_elapsed_ns == Some(7))
            );
            let receipt = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(1))
                .unwrap();
            assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
            assert_eq!(receipt.completed_role_inputs, 0);
        }
        #[test]
        fn loading_mapping_ack_requires_exact_pin_profile_partition_and_full_ort() {
            let (execution, witness) = loading_mapping_fixture();
            assert!(validate_runtime_loading_mapping(&witness, &execution).is_ok());
            for mutation in 0..6 {
                let mut changed = witness.clone();
                match mutation {
                    0 => changed.runtime_sha256 = [9; 32],
                    1 => changed.loading_profile_sha256 = [9; 32],
                    2 => {
                        changed.mapped_ort_files.pop().unwrap();
                    }
                    3 => changed
                        .mapped_nvidia_files
                        .push(changed.mapped_nvidia_files[0].clone()),
                    4 => {
                        changed.deferred_nvidia_not_mapped[0] =
                            changed.mapped_nvidia_files[0].clone()
                    }
                    _ => changed.required_nvidia_files[0] = "unpinned-library.so".into(),
                }
                assert!(validate_runtime_loading_mapping(&changed, &execution).is_err());
            }
            let mut cpu = execution.clone();
            cpu.provider = "cpu";
            assert!(validate_runtime_loading_mapping(&witness, &cpu).is_err());
            cpu.cuda_loading_profile = None;
            let encoded = serde_json::to_value(cpu).unwrap();
            assert!(encoded.get("cuda_loading_profile").is_none());
        }
        #[test]
        fn final_loading_mapping_control_ack_is_nn_zero_and_failure_is_preserved() {
            for invalid in [false, true] {
                let (execution, mut witness) = loading_mapping_fixture();
                if invalid {
                    witness.runtime_sha256 = [9; 32];
                }
                let commands = Arc::new(Mutex::new(Vec::new()));
                let observed = Arc::clone(&commands);
                let model = fixture_model_with_execution(
                    move |command| {
                        let (kind, output) = match command {
                            PalsNativeCommand::VerifyRuntime => {
                                ("verify", PalsNativeResult::RuntimeVerified)
                            }
                            PalsNativeCommand::ObserveRuntimeMappings => (
                                "mapping",
                                PalsNativeResult::RuntimeMappingsObserved(Box::new(
                                    witness.clone(),
                                )),
                            ),
                            PalsNativeCommand::SnapshotStats => (
                                "stats",
                                PalsNativeResult::Stats(PalsBackendStats::default()),
                            ),
                            _ => return unexpected_cuda_placement(),
                        };
                        observed.lock().unwrap().push(kind);
                        PhysicalRun::Complete(Ok(output))
                    },
                    Some(execution),
                );
                // A synthetic worker tests receipt and lease accounting only.
                // It neither initializes CUDA nor advertises native readiness.
                let finish = model.finish_handle();
                let result = finish.finish(Instant::now() + Duration::from_secs(2));
                assert_eq!(result.is_ok(), !invalid);
                let receipt = finish.receipt();
                assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
                assert!(!receipt.quarantined);
                assert_eq!(receipt.completed_role_inputs, 0);
                assert_eq!(receipt.physically_completed_role_calls, 0);
                assert_eq!(receipt.delivered_role_inputs, 0);
                assert_eq!(receipt.search_consumed_role_inputs, 0);
                assert_eq!(receipt.physical_runs_in_flight, 0);
                if invalid {
                    assert!(receipt.last_failure.is_some());
                    assert!(receipt.final_runtime_loading_mapping.is_none());
                    assert!(receipt.backend_stats.is_none());
                    assert_eq!(*commands.lock().unwrap(), ["verify", "mapping"]);
                } else {
                    assert!(receipt.last_failure.is_none());
                    assert!(receipt.final_runtime_loading_mapping.is_some());
                    assert_eq!(receipt.backend_stats.unwrap().completed_nn_inputs, 0);
                    assert_eq!(*commands.lock().unwrap(), ["verify", "mapping", "stats"]);
                }
            }
        }
        #[test]
        fn startup_worker_probe_does_not_create_search_consumption() {
            for selected in [None, Some(2_000)] {
                let mut stats = PalsBackendStats::default();
                let mut cached = false;
                let mut model = fixture_model(move |command| {
                    PhysicalRun::Complete(Ok(match command {
                        PalsNativeCommand::Evaluate(input)
                        | PalsNativeCommand::EvaluateWithEvidence(input) => {
                            stats.admitted_role_requests += 1;
                            if cached {
                                stats.public_cache_hits += 1;
                            } else {
                                stats.public_cache_misses += 1;
                                stats.public_nn_runs_attempted += 1;
                                stats.public_nn_runs_completed += 1;
                                stats.completed_nn_inputs += 1;
                                stats.validated_public_outputs += 1;
                                cached = true;
                            }
                            stats.role_nn_runs_attempted += 1;
                            stats.role_nn_runs_completed += 1;
                            stats.completed_nn_inputs += 1;
                            stats.validated_role_outputs += 1;
                            output(input)
                        }
                        PalsNativeCommand::NewGame => {
                            cached = false;
                            stats.new_game_resets += 1;
                            PalsNativeResult::NewGame
                        }
                        PalsNativeCommand::ResetTo(target) => {
                            cached = false;
                            stats.new_game_resets += 1;
                            PalsNativeResult::ResetTo {
                                game_generation: *target,
                            }
                        }
                        PalsNativeCommand::EvaluatePrivateWarm(_)
                        | PalsNativeCommand::VerifyCudaPlacement
                        | PalsNativeCommand::ObserveRuntimeMappings => {
                            return unexpected_cuda_placement();
                        }
                        #[cfg(feature = "experimental-io-binding")]
                        PalsNativeCommand::EvaluateCudaWarm(_) => {
                            return unexpected_cuda_placement();
                        }
                        #[cfg(feature = "experimental-io-binding")]
                        PalsNativeCommand::SnapshotCudaRecordPages => {
                            return unexpected_cuda_placement();
                        }
                        PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                        PalsNativeCommand::SnapshotStats => PalsNativeResult::Stats(stats.clone()),
                    }))
                });
                // Only the control-flow is simulated by this CPU worker fixture;
                // it is not provider/device/NN capability evidence.
                model.owner.startup_ready.store(false, Ordering::Release);
                *model.owner.startup_probe.lock().unwrap() = Some(NativeStartupProbeReceipt {
                    completed_proposer_calls: 0,
                    completed_critic_calls: 0,
                    runtime_mapping_confirmed: false,
                    runtime_loading_mapping: None,
                    cuda_placement_witness: None,
                    reset_completed: false,
                    backend_stats: None,
                    failure_diagnostic: None,
                    backend_stages: None,
                    timing_observation: None,
                    cuda_record_page_observation: None,
                    cuda_record_page_observation_error: None,
                });
                let budget = model.configure_startup_probe_timeout(selected).unwrap();
                model
                    .observe_startup_loading(Duration::from_millis(3))
                    .unwrap();
                model.prepare_startup(Instant::now() + budget).unwrap();
                let receipt = model.finish_handle().receipt();
                let startup = receipt.startup_probe.unwrap();
                let encoded = serde_json::to_value(&startup).unwrap();
                assert!(encoded.get("failure_diagnostic").is_none());
                assert!(encoded.get("backend_stages").is_none());
                assert_eq!(
                    encoded.get("timing_observation").is_some(),
                    selected.is_some()
                );
                if let Some(timing) = startup.timing_observation.as_ref() {
                    assert_eq!(timing.model_loading_elapsed_ns, Some(3_000_000));
                    assert!(timing.probe_elapsed_ns.is_some());
                    assert_eq!(timing.commands.len(), 5);
                    assert!(
                        timing
                            .commands
                            .iter()
                            .all(|event| event.physical_completion_confirmed == Some(true))
                    );
                    for (ordinal, command) in timing.commands.iter().take(2).enumerate() {
                        let input = command.input_observation.as_ref().unwrap();
                        assert_eq!(input.startup_request_ordinal, Some((ordinal + 1) as u8));
                        assert_eq!(input.retained_input_key, Some(input.input_key));
                        assert_eq!(input.raw_decode_succeeded, Some(true));
                    }
                }
                assert_eq!(startup.completed_proposer_calls, 1);
                assert_eq!(startup.completed_critic_calls, 1);
                assert!(startup.runtime_mapping_confirmed && startup.reset_completed);
                assert_eq!(startup.backend_stats.unwrap().completed_nn_inputs, 3);
                assert_eq!(receipt.completed_role_inputs, 0);
                assert_eq!(receipt.delivered_role_inputs, 0);
                assert_eq!(receipt.search_consumed_role_inputs, 0);
                assert_eq!(receipt.completed_new_game_resets, 0);
                assert_eq!(receipt.request_high_water, 0);
                assert_eq!(receipt.physical_runs_in_flight, 0);
                model
                    .finish_handle()
                    .finish(Instant::now() + Duration::from_secs(2))
                    .unwrap();
            }
        }
        // Controlled worker/observer fixtures only. No ORT/model/provider/source,
        // registered binary, whole-cost, CUDA or training authority is produced.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum BindingTerminalFact {
            Evaluation,
            BackendFailure,
            Control,
            Unknown(NativeRoleCompletionUnknown),
        }
        #[derive(Default)]
        struct BindingFacts {
            prepared: Vec<RequestId>,
            dispatched: Vec<NativeRoleExecutionBinding>,
            terminal: Vec<(NativeRoleExecutionBinding, BindingTerminalFact)>,
            delivered: Vec<RequestId>,
            accepted: Vec<RequestId>,
            inflight_at_dispatch: Vec<u64>,
            backend_details: Vec<&'static str>,
        }
        struct BindingObserver {
            facts: Arc<Mutex<BindingFacts>>,
            owner: std::sync::Weak<WorkerOwner>,
            fail_dispatch: bool,
            panic_dispatch: bool,
            fail_terminal: bool,
            overwrite_high_water: bool,
            release: Option<std::sync::mpsc::Sender<()>>,
        }
        fn binding_observer(
            facts: &Arc<Mutex<BindingFacts>>,
            owner: &Arc<WorkerOwner>,
        ) -> BindingObserver {
            BindingObserver {
                facts: Arc::clone(facts),
                owner: Arc::downgrade(owner),
                fail_dispatch: false,
                panic_dispatch: false,
                fail_terminal: false,
                overwrite_high_water: false,
                release: None,
            }
        }
        impl NativeRoleObserver for BindingObserver {
            fn prepared(
                &mut self,
                id: RequestId,
                _: &PalsModelInput,
                _: NativePreparedContext<'_>,
            ) -> Result<(), RoleError> {
                self.facts.lock().unwrap().prepared.push(id);
                Ok(())
            }
            fn dispatched(&mut self, binding: NativeRoleExecutionBinding) -> Result<(), RoleError> {
                let owner = self.owner.upgrade().unwrap();
                {
                    let mut facts = self.facts.lock().unwrap();
                    facts.dispatched.push(binding);
                    facts
                        .inflight_at_dispatch
                        .push(owner.in_flight.load(Ordering::Acquire));
                }
                if self.overwrite_high_water {
                    // Deliberately unrelated metadata, proving callbacks use
                    // the actual Lease rather than reading this mutable counter.
                    owner.execution_high_water.store(999, Ordering::Release);
                }
                if let Some(release) = self.release.take() {
                    release.send(()).unwrap();
                }
                if self.panic_dispatch {
                    panic!("controlled dispatch observer panic");
                }
                if self.fail_dispatch {
                    return Err(RoleError::Backend(
                        "controlled dispatch observer failure".into(),
                    ));
                }
                Ok(())
            }
            fn terminal(
                &mut self,
                binding: NativeRoleExecutionBinding,
                event: NativeRoleTerminal<'_>,
            ) -> Result<(), RoleError> {
                let mut facts = self.facts.lock().unwrap();
                let fact = match event {
                    NativeRoleTerminal::Ready(Ok(PalsNativeResult::Evaluation(_))) => {
                        BindingTerminalFact::Evaluation
                    }
                    NativeRoleTerminal::Ready(Ok(_)) => BindingTerminalFact::Control,
                    NativeRoleTerminal::Ready(Err(error)) => {
                        facts.backend_details.push(error.detail);
                        BindingTerminalFact::BackendFailure
                    }
                    NativeRoleTerminal::CompletionUnknown(reason) => {
                        BindingTerminalFact::Unknown(reason)
                    }
                };
                facts.terminal.push((binding, fact));
                drop(facts);
                if self.fail_terminal {
                    return Err(RoleError::Backend(
                        "controlled terminal observer failure".into(),
                    ));
                }
                Ok(())
            }
            fn delivered(&mut self, id: RequestId) -> Result<(), RoleError> {
                self.facts.lock().unwrap().delivered.push(id);
                Ok(())
            }
            fn accepted(&mut self, id: RequestId) -> Result<(), RoleError> {
                self.facts.lock().unwrap().accepted.push(id);
                Ok(())
            }
        }
        fn binding_fixture_command(
            command: &PalsNativeCommand,
        ) -> PhysicalRun<Result<PalsNativeResult, BackendError>> {
            match command {
                PalsNativeCommand::Evaluate(input)
                | PalsNativeCommand::EvaluateWithEvidence(input) => {
                    PhysicalRun::Complete(Ok(output(input)))
                }
                PalsNativeCommand::NewGame => PhysicalRun::Complete(Ok(PalsNativeResult::NewGame)),
                PalsNativeCommand::ResetTo(target) => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::ResetTo {
                        game_generation: *target,
                    }))
                }
                PalsNativeCommand::SnapshotStats => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::Stats(PalsBackendStats::default())))
                }
                _ => unexpected_cuda_placement(),
            }
        }
        fn binding_backend_error() -> BackendError {
            BackendError::new(
                FailureKind::BackendFailure,
                FailureStage::Backend,
                "binding fixture original backend cause",
            )
        }
        #[test]
        fn role_binding_uses_actual_dispatch_and_ready_ids_across_new_game() {
            let mut model = fixture_model(binding_fixture_command);
            let facts = Arc::new(Mutex::new(BindingFacts::default()));
            let mut observer = binding_observer(&facts, &model.owner);
            observer.overwrite_high_water = true;
            model.set_observer(Box::new(observer)).unwrap();
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            for reset in [false, true] {
                if reset {
                    model.new_game();
                }
                model.propose(query(&position, &legal, &cancel)).unwrap();
                model.accepted_output();
            }
            {
                let facts = facts.lock().unwrap();
                assert_eq!(facts.prepared.len(), 2);
                assert_eq!(facts.dispatched.len(), 2);
                assert_eq!(facts.terminal.len(), 2);
                assert_eq!(facts.inflight_at_dispatch, [1, 1]);
                for index in 0..2 {
                    let binding = facts.dispatched[index];
                    assert_eq!(binding.request, facts.prepared[index]);
                    assert_eq!(binding.request.epoch, model.epoch);
                    assert_ne!(binding.execution.sequence, 999);
                    assert_eq!(
                        facts.terminal[index],
                        (binding, BindingTerminalFact::Evaluation)
                    );
                    assert_eq!(facts.delivered[index], binding.request);
                    assert_eq!(facts.accepted[index], binding.request);
                }
                assert_ne!(facts.dispatched[0].execution, facts.dispatched[1].execution);
            }
            let receipt = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert_eq!(receipt.execution_high_water, 999);
            assert_eq!(receipt.physically_completed_role_calls, 2);
            assert_eq!(receipt.search_consumed_role_inputs, 2);
            assert_eq!(receipt.observer_failures, 0);
            assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
        }
        #[test]
        fn post_submit_observer_error_and_panic_keep_lease_until_actual_ready() {
            for panic_dispatch in [false, true] {
                let (release, waiting) = std::sync::mpsc::channel();
                let mut model = fixture_model(move |command| match command {
                    PalsNativeCommand::Evaluate(input)
                    | PalsNativeCommand::EvaluateWithEvidence(input) => {
                        waiting.recv_timeout(Duration::from_secs(3)).unwrap();
                        PhysicalRun::Complete(Ok(output(input)))
                    }
                    _ => binding_fixture_command(command),
                });
                let facts = Arc::new(Mutex::new(BindingFacts::default()));
                let mut observer = binding_observer(&facts, &model.owner);
                observer.fail_dispatch = !panic_dispatch;
                observer.panic_dispatch = panic_dispatch;
                observer.release = Some(release);
                model.set_observer(Box::new(observer)).unwrap();
                let position = Position::startpos();
                let legal = position.legal_moves();
                let cancel = AtomicBool::new(false);
                assert!(matches!(
                    model.propose(query(&position, &legal, &cancel)),
                    Err(RoleError::Backend(_))
                ));
                model.accepted_output();
                {
                    let facts = facts.lock().unwrap();
                    assert_eq!(facts.inflight_at_dispatch, [1]);
                    assert_eq!(
                        facts.terminal,
                        [(facts.dispatched[0], BindingTerminalFact::Evaluation)]
                    );
                    assert!(facts.delivered.is_empty() && facts.accepted.is_empty());
                }
                let observed = model.owner.role_execution.lock().unwrap();
                assert!(observed.as_ref().unwrap().ready_observed);
                assert!(observed.as_ref().unwrap().observer_error.is_some());
                drop(observed);
                let handle = model.finish_handle();
                let before = handle.receipt();
                assert_eq!(before.physically_completed_role_calls, 1);
                assert_eq!(before.physical_runs_in_flight, 0);
                assert_eq!(before.observer_failures, 1);
                assert_eq!(
                    before.last_observer_failure.as_ref().unwrap().stage,
                    "role_dispatched"
                );
                assert!(!before.quarantined);
                let done = handle
                    .finish(Instant::now() + Duration::from_secs(2))
                    .unwrap();
                assert!(done.physical_shutdown_confirmed && done.native_buffers_released);
                assert_eq!(done.search_consumed_role_inputs, 0);
            }
        }
        #[test]
        fn backend_failure_stays_primary_when_dispatch_and_terminal_observers_fail() {
            let mut model = fixture_model(|command| match command {
                PalsNativeCommand::Evaluate(_) | PalsNativeCommand::EvaluateWithEvidence(_) => {
                    PhysicalRun::Complete(Err(binding_backend_error()))
                }
                _ => binding_fixture_command(command),
            });
            let facts = Arc::new(Mutex::new(BindingFacts::default()));
            let mut observer = binding_observer(&facts, &model.owner);
            observer.fail_dispatch = true;
            observer.fail_terminal = true;
            model.set_observer(Box::new(observer)).unwrap();
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            assert!(matches!(
                model.propose(query(&position, &legal, &cancel)),
                Err(RoleError::Backend(_))
            ));
            {
                let facts = facts.lock().unwrap();
                assert_eq!(
                    facts.terminal,
                    [(facts.dispatched[0], BindingTerminalFact::BackendFailure)]
                );
                assert_eq!(
                    facts.backend_details,
                    ["binding fixture original backend cause"]
                );
                assert!(facts.delivered.is_empty() && facts.accepted.is_empty());
            }
            let receipt = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert_eq!(
                receipt.last_failure.as_ref().unwrap().detail,
                "binding fixture original backend cause"
            );
            assert_eq!(receipt.failed_physical_role_calls, 1);
            assert_eq!(receipt.physically_completed_role_calls, 1);
            assert_eq!(receipt.observer_failures, 2);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
            assert!(!receipt.quarantined);
            assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
        }
        #[test]
        fn unknown_binding_is_once_only_and_never_a_fake_ready_after_observer_error() {
            let mut model = fixture_model(|_| PhysicalRun::Quarantined(binding_backend_error()));
            let facts = Arc::new(Mutex::new(BindingFacts::default()));
            let mut observer = binding_observer(&facts, &model.owner);
            observer.fail_terminal = true;
            model.set_observer(Box::new(observer)).unwrap();
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            assert!(matches!(
                model.propose(query(&position, &legal, &cancel)),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            for _ in 0..3 {
                model.runtime.pump();
            }
            {
                let facts = facts.lock().unwrap();
                assert_eq!(
                    facts.terminal,
                    [(
                        facts.dispatched[0],
                        BindingTerminalFact::Unknown(NativeRoleCompletionUnknown::Quarantined)
                    )]
                );
                assert!(facts.delivered.is_empty() && facts.accepted.is_empty());
            }
            let handle = model.finish_handle();
            assert!(matches!(
                handle.finish(Instant::now() + Duration::from_secs(1)),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            let receipt = handle.receipt();
            assert_eq!(receipt.physically_completed_role_calls, 0);
            assert_eq!(receipt.physical_runs_in_flight, 1);
            assert_eq!(receipt.observer_failures, 1);
            assert_eq!(
                receipt.last_failure.as_ref().unwrap().detail,
                "binding fixture original backend cause"
            );
            assert!(
                receipt.quarantined
                    && !receipt.physical_shutdown_confirmed
                    && !receipt.native_buffers_released
            );
        }
        #[test]
        fn actual_late_ready_after_drain_unknown_never_reopens_consumption() {
            let (release, waiting) = std::sync::mpsc::channel();
            let mut model = fixture_model(move |command| match command {
                PalsNativeCommand::Evaluate(input)
                | PalsNativeCommand::EvaluateWithEvidence(input) => {
                    waiting.recv_timeout(Duration::from_secs(3)).unwrap();
                    PhysicalRun::Complete(Ok(output(input)))
                }
                _ => binding_fixture_command(command),
            });
            model.drain_limit = Duration::from_millis(10);
            let facts = Arc::new(Mutex::new(BindingFacts::default()));
            model
                .set_observer(Box::new(binding_observer(&facts, &model.owner)))
                .unwrap();
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            let mut request = query(&position, &legal, &cancel);
            // Waiting is controlled independently of the logical deadline. No
            // fake Ready or shortened node/model computation stands in for it.
            request.deadline = Instant::now() + Duration::from_millis(500);
            assert!(matches!(
                model.propose(request),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            {
                let facts = facts.lock().unwrap();
                assert_eq!(facts.dispatched.len(), 1);
                assert_eq!(
                    facts.terminal,
                    [(
                        facts.dispatched[0],
                        BindingTerminalFact::Unknown(NativeRoleCompletionUnknown::DrainExpired)
                    )]
                );
            }
            assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 1);
            release.send(()).unwrap();
            let until = Instant::now() + Duration::from_secs(2);
            // Existing PalsRuntime allows pumping a timed-out retained owner;
            // only its actual backend Ready decrements the physical counter.
            while model.owner.in_flight.load(Ordering::Acquire) != 0 && Instant::now() < until {
                model.runtime.pump();
                std::thread::sleep(Duration::from_millis(1));
            }
            {
                let facts = facts.lock().unwrap();
                let binding = facts.dispatched[0];
                assert_eq!(
                    facts.terminal,
                    [
                        (
                            binding,
                            BindingTerminalFact::Unknown(NativeRoleCompletionUnknown::DrainExpired)
                        ),
                        (binding, BindingTerminalFact::Evaluation),
                    ]
                );
                assert!(facts.delivered.is_empty() && facts.accepted.is_empty());
            }
            model.accepted_output();
            assert!(matches!(
                model.propose(query(&position, &legal, &cancel)),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            let handle = model.finish_handle();
            assert!(matches!(
                handle.finish(until),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            let receipt = handle.receipt();
            assert_eq!(receipt.physically_completed_role_calls, 1);
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
            assert!(receipt.quarantined && !receipt.native_buffers_released);
        }
        #[test]
        fn poisoned_observer_keeps_actual_ready_and_original_backend_cause() {
            let target = Arc::new(Mutex::new(std::sync::Weak::<WorkerOwner>::new()));
            let actual_target = Arc::clone(&target);
            let mut model = fixture_model(move |command| match command {
                PalsNativeCommand::Evaluate(_) | PalsNativeCommand::EvaluateWithEvidence(_) => {
                    let owner = actual_target.lock().unwrap().upgrade().unwrap();
                    // Poison after physical submit; never skip the real Ready.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _guard = owner.observer.lock().unwrap();
                        panic!("controlled observer mutex poison");
                    }));
                    PhysicalRun::Complete(Err(binding_backend_error()))
                }
                _ => binding_fixture_command(command),
            });
            *target.lock().unwrap() = Arc::downgrade(&model.owner);
            let facts = Arc::new(Mutex::new(BindingFacts::default()));
            model
                .set_observer(Box::new(binding_observer(&facts, &model.owner)))
                .unwrap();
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            assert!(matches!(
                model.propose(query(&position, &legal, &cancel)),
                Err(RoleError::Backend(_))
            ));
            let receipt = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert_eq!(
                receipt.last_failure.as_ref().unwrap().detail,
                "binding fixture original backend cause"
            );
            assert_eq!(receipt.failed_physical_role_calls, 1);
            assert_eq!(receipt.physically_completed_role_calls, 1);
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert!(receipt.observer_failures >= 2);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
            assert!(!receipt.quarantined);
            assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
        }
        #[test]
        fn bound_ready_control_is_known_physical_work_but_not_a_role_output() {
            let mut model = fixture_model(|command| match command {
                PalsNativeCommand::Evaluate(_) | PalsNativeCommand::EvaluateWithEvidence(_) => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::NewGame))
                }
                _ => binding_fixture_command(command),
            });
            let facts = Arc::new(Mutex::new(BindingFacts::default()));
            model
                .set_observer(Box::new(binding_observer(&facts, &model.owner)))
                .unwrap();
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            assert!(model.propose(query(&position, &legal, &cancel)).is_err());
            {
                let facts = facts.lock().unwrap();
                assert_eq!(
                    facts.terminal,
                    [(facts.dispatched[0], BindingTerminalFact::Control)]
                );
                assert!(facts.delivered.is_empty() && facts.accepted.is_empty());
            }
            let receipt = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert_eq!(receipt.physically_completed_role_calls, 1);
            assert_eq!(receipt.completed_role_inputs, 0);
            assert_eq!(receipt.invalid_role_outputs, 1);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
            assert_eq!(receipt.observer_failures, 0);
        }
        struct InputObserver {
            input_keys: Arc<Mutex<Vec<[u8; 32]>>>,
            stages: Arc<Mutex<Vec<(u64, &'static str)>>>,
            fail_physical: bool,
        }
        impl NativeRoleObserver for InputObserver {
            fn prepared(
                &mut self,
                id: RequestId,
                input: &PalsModelInput,
                context: NativePreparedContext<'_>,
            ) -> Result<(), RoleError> {
                let NativePreparedContext::Role { query, .. } = context else {
                    return Err(RoleError::InvalidOutput);
                };
                assert_eq!(input.history_digest, pals_history_digest(query.position)?);
                assert_eq!(input.situation_revision, query.revision);
                self.input_keys.lock().unwrap().push(
                    input
                        .canonical_input_key(&PalsModelConfig::baseline())
                        .map_err(model_error)?,
                );
                self.stages.lock().unwrap().push((id.sequence, "prepared"));
                Ok(())
            }
            fn physically_completed(
                &mut self,
                id: RequestId,
                result: Result<&PalsRawOutput, &BackendError>,
            ) -> Result<(), RoleError> {
                assert!(result.is_ok());
                self.stages.lock().unwrap().push((id.sequence, "physical"));
                if self.fail_physical {
                    Err(RoleError::PhysicalCompletionUnknown)
                } else {
                    Ok(())
                }
            }
            fn delivered(&mut self, id: RequestId) -> Result<(), RoleError> {
                self.stages.lock().unwrap().push((id.sequence, "delivered"));
                Ok(())
            }
            fn accepted(&mut self, id: RequestId) -> Result<(), RoleError> {
                self.stages.lock().unwrap().push((id.sequence, "accepted"));
                Ok(())
            }
        }
        #[test]
        fn observer_sees_the_exact_once_prepared_input_and_authorized_consumption() {
            let actual = Arc::new(Mutex::new(Vec::new()));
            let executed = Arc::clone(&actual);
            let mut model = fixture_model(move |command| {
                PhysicalRun::Complete(Ok(match command {
                    PalsNativeCommand::Evaluate(input)
                    | PalsNativeCommand::EvaluateWithEvidence(input) => {
                        executed.lock().unwrap().push(
                            input
                                .canonical_input_key(&PalsModelConfig::baseline())
                                .unwrap(),
                        );
                        output(input)
                    }
                    PalsNativeCommand::NewGame => PalsNativeResult::NewGame,
                    PalsNativeCommand::ResetTo(target) => PalsNativeResult::ResetTo {
                        game_generation: *target,
                    },
                    PalsNativeCommand::EvaluatePrivateWarm(_)
                    | PalsNativeCommand::VerifyCudaPlacement
                    | PalsNativeCommand::ObserveRuntimeMappings => {
                        return unexpected_cuda_placement();
                    }
                    #[cfg(feature = "experimental-io-binding")]
                    PalsNativeCommand::EvaluateCudaWarm(_) => return unexpected_cuda_placement(),
                    #[cfg(feature = "experimental-io-binding")]
                    PalsNativeCommand::SnapshotCudaRecordPages => {
                        return unexpected_cuda_placement();
                    }
                    PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                    PalsNativeCommand::SnapshotStats => {
                        PalsNativeResult::Stats(PalsBackendStats::default())
                    }
                }))
            });
            let observed = Arc::new(Mutex::new(Vec::new()));
            let stages = Arc::new(Mutex::new(Vec::new()));
            model
                .set_observer(Box::new(InputObserver {
                    input_keys: Arc::clone(&observed),
                    stages: Arc::clone(&stages),
                    fail_physical: false,
                }))
                .unwrap();
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            model.propose(query(&position, &legal, &cancel)).unwrap();
            model.accepted_output();
            model.accepted_output(); // Duplicate semantic acceptance cannot consume twice.
            assert_eq!(*actual.lock().unwrap(), *observed.lock().unwrap());
            assert_eq!(
                *stages.lock().unwrap(),
                [
                    (1, "prepared"),
                    (1, "physical"),
                    (1, "delivered"),
                    (1, "accepted")
                ]
            );
            assert_eq!(
                model.finish_handle().receipt().search_consumed_role_inputs,
                1
            );
            model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
        }
        #[test]
        fn observer_error_cannot_reclassify_confirmed_physical_work_as_unknown() {
            let mut model = fixture_model(|command| {
                PhysicalRun::Complete(Ok(match command {
                    PalsNativeCommand::Evaluate(input)
                    | PalsNativeCommand::EvaluateWithEvidence(input) => output(input),
                    PalsNativeCommand::NewGame => PalsNativeResult::NewGame,
                    PalsNativeCommand::ResetTo(target) => PalsNativeResult::ResetTo {
                        game_generation: *target,
                    },
                    PalsNativeCommand::EvaluatePrivateWarm(_)
                    | PalsNativeCommand::VerifyCudaPlacement
                    | PalsNativeCommand::ObserveRuntimeMappings => {
                        return unexpected_cuda_placement();
                    }
                    #[cfg(feature = "experimental-io-binding")]
                    PalsNativeCommand::EvaluateCudaWarm(_) => return unexpected_cuda_placement(),
                    #[cfg(feature = "experimental-io-binding")]
                    PalsNativeCommand::SnapshotCudaRecordPages => {
                        return unexpected_cuda_placement();
                    }
                    PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                    PalsNativeCommand::SnapshotStats => {
                        PalsNativeResult::Stats(PalsBackendStats::default())
                    }
                }))
            });
            model
                .set_observer(Box::new(InputObserver {
                    input_keys: Arc::new(Mutex::new(Vec::new())),
                    stages: Arc::new(Mutex::new(Vec::new())),
                    fail_physical: true,
                }))
                .unwrap();
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            assert!(matches!(
                model.propose(query(&position, &legal, &cancel)),
                Err(RoleError::Backend(_))
            ));
            let receipt = model.finish_handle().receipt();
            assert_eq!(receipt.completed_role_inputs, 1);
            assert_eq!(receipt.observer_failures, 1);
            assert_eq!(receipt.delivered_role_inputs, 0);
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert!(!receipt.quarantined);
            assert!(receipt.last_observer_failure.is_some());
            let done = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert!(done.physical_shutdown_confirmed && done.native_buffers_released);
        }
        #[test]
        fn search_closure_rejects_unconsumed_delivery_before_the_next_root() {
            struct ClosureObserver(Arc<Mutex<Vec<(u64, NativeRoleRejection)>>>);
            impl NativeRoleObserver for ClosureObserver {
                fn prepared(
                    &mut self,
                    _: RequestId,
                    _: &PalsModelInput,
                    _: NativePreparedContext<'_>,
                ) -> Result<(), RoleError> {
                    Ok(())
                }
                fn rejected(
                    &mut self,
                    id: RequestId,
                    reason: NativeRoleRejection,
                ) -> Result<(), RoleError> {
                    self.0.lock().unwrap().push((id.sequence, reason));
                    Ok(())
                }
            }
            let mut model = fixture_model(|command| {
                PhysicalRun::Complete(Ok(match command {
                    PalsNativeCommand::Evaluate(input)
                    | PalsNativeCommand::EvaluateWithEvidence(input) => output(input),
                    PalsNativeCommand::NewGame => PalsNativeResult::NewGame,
                    PalsNativeCommand::ResetTo(target) => PalsNativeResult::ResetTo {
                        game_generation: *target,
                    },
                    PalsNativeCommand::EvaluatePrivateWarm(_)
                    | PalsNativeCommand::VerifyCudaPlacement
                    | PalsNativeCommand::ObserveRuntimeMappings => {
                        return unexpected_cuda_placement();
                    }
                    #[cfg(feature = "experimental-io-binding")]
                    PalsNativeCommand::EvaluateCudaWarm(_) => return unexpected_cuda_placement(),
                    #[cfg(feature = "experimental-io-binding")]
                    PalsNativeCommand::SnapshotCudaRecordPages => {
                        return unexpected_cuda_placement();
                    }
                    PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                    PalsNativeCommand::SnapshotStats => {
                        PalsNativeResult::Stats(PalsBackendStats::default())
                    }
                }))
            });
            let rejected = Arc::new(Mutex::new(Vec::new()));
            model
                .set_observer(Box::new(ClosureObserver(Arc::clone(&rejected))))
                .unwrap();
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            model.propose(query(&position, &legal, &cancel)).unwrap();
            model.finish_search(RoleSearchClosure::Deadline);
            model.finish_search(RoleSearchClosure::Deadline);
            let receipt = model.finish_handle().receipt();
            assert_eq!(
                *rejected.lock().unwrap(),
                [(1, NativeRoleRejection::Deadline)]
            );
            assert!(model.delivered_request.is_none());
            assert_eq!(receipt.physically_completed_role_calls, 1);
            assert_eq!(receipt.delivered_role_inputs, 1);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
            assert_eq!(receipt.canceled_requests, 0);
            assert_eq!(receipt.expired_requests, 0);
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert!(!receipt.quarantined && !receipt.physical_shutdown_confirmed);
            model.new_game();
            model.propose(query(&position, &legal, &cancel)).unwrap();
            model.accepted_output();
            model.finish_search(RoleSearchClosure::Completed);
            assert_eq!(rejected.lock().unwrap().len(), 1);
            let done = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert_eq!(done.search_consumed_role_inputs, 1);
            assert!(done.physical_shutdown_confirmed && done.native_buffers_released);
        }
        #[test]
        fn reset_ack_and_consumption_are_not_extra_nn_executions() {
            let resets = Arc::new(AtomicU64::new(0));
            let observed = Arc::clone(&resets);
            let mut model = fixture_model(move |command| {
                PhysicalRun::Complete(Ok(match command {
                    PalsNativeCommand::Evaluate(input)
                    | PalsNativeCommand::EvaluateWithEvidence(input) => output(input),
                    PalsNativeCommand::NewGame => {
                        observed.fetch_add(1, Ordering::SeqCst);
                        PalsNativeResult::NewGame
                    }
                    PalsNativeCommand::ResetTo(target) => {
                        observed.fetch_add(1, Ordering::SeqCst);
                        PalsNativeResult::ResetTo {
                            game_generation: *target,
                        }
                    }
                    PalsNativeCommand::SnapshotStats => {
                        PalsNativeResult::Stats(PalsBackendStats::default())
                    }
                    PalsNativeCommand::EvaluatePrivateWarm(_)
                    | PalsNativeCommand::VerifyCudaPlacement
                    | PalsNativeCommand::ObserveRuntimeMappings => {
                        return unexpected_cuda_placement();
                    }
                    #[cfg(feature = "experimental-io-binding")]
                    PalsNativeCommand::EvaluateCudaWarm(_) => return unexpected_cuda_placement(),
                    #[cfg(feature = "experimental-io-binding")]
                    PalsNativeCommand::SnapshotCudaRecordPages => {
                        return unexpected_cuda_placement();
                    }
                    PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                }))
            });
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            model.propose(query(&position, &legal, &cancel)).unwrap();
            model.accepted_output();
            model.new_game();
            assert_eq!(resets.load(Ordering::SeqCst), 0);
            model.propose(query(&position, &legal, &cancel)).unwrap();
            let receipt = model.finish_handle().receipt();
            assert_eq!(resets.load(Ordering::SeqCst), 1);
            assert_eq!(receipt.completed_new_game_resets, 1);
            assert_eq!(receipt.physically_completed_role_calls, 2);
            assert_eq!(receipt.completed_role_inputs, 2);
            assert_eq!(receipt.delivered_role_inputs, 2);
            assert_eq!(receipt.search_consumed_role_inputs, 1);
            assert_eq!(receipt.game_generation, 2);
            assert_eq!(receipt.request_high_water, 2);
            assert_eq!(receipt.execution_high_water, 2);
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert!(receipt.backend_stats.is_none());
            let done = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert!(done.physical_shutdown_confirmed && done.native_buffers_released);
            assert!(done.backend_stats.is_some());
            assert_eq!(
                done.backend_stats_observation,
                Some("exclusive_worker_before_shutdown")
            );
            assert_eq!(done.completed_role_inputs, receipt.completed_role_inputs);
            assert_eq!(
                done.physically_completed_role_calls,
                receipt.physically_completed_role_calls
            );
        }
        #[test]
        fn repeated_logical_resets_target_one_physical_reset_before_nn() {
            let targets = Arc::new(Mutex::new(Vec::new()));
            let actual = Arc::clone(&targets);
            let mut model = fixture_model(move |command| {
                if let PalsNativeCommand::ResetTo(target) = command {
                    actual.lock().unwrap().push(*target);
                }
                binding_fixture_command(command)
            });
            model.new_game();
            model.new_game();
            assert!(targets.lock().unwrap().is_empty());
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            model.propose(query(&position, &legal, &cancel)).unwrap();
            model.accepted_output();
            assert_eq!(*targets.lock().unwrap(), [3]);
            assert_eq!(model.owner.pending_game_target.load(Ordering::Acquire), 0);
            let done = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert_eq!(done.game_generation, 3);
            assert_eq!(done.completed_new_game_resets, 1);
            assert_eq!(done.physically_completed_role_calls, 1);
            assert_eq!(done.search_consumed_role_inputs, 1);
        }
        #[test]
        fn repeated_logical_resets_are_acknowledged_before_quit_stats() {
            let commands = Arc::new(Mutex::new(Vec::new()));
            let actual = Arc::clone(&commands);
            let mut model = fixture_model(move |command| {
                actual.lock().unwrap().push(match command {
                    PalsNativeCommand::ResetTo(target) => format!("reset:{target}"),
                    PalsNativeCommand::SnapshotStats => "stats".into(),
                    _ => "other".into(),
                });
                binding_fixture_command(command)
            });
            // A stale previous observation must not short-circuit the reset.
            *model.owner.final_stats.lock().unwrap() = Some(PalsBackendStats::default().into());
            model.new_game();
            model.new_game();
            assert!(model.owner.final_stats.lock().unwrap().is_none());
            let done = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert_eq!(*commands.lock().unwrap(), ["reset:3", "stats"]);
            assert_eq!(done.game_generation, 3);
            assert_eq!(done.completed_new_game_resets, 1);
            assert_eq!(done.physically_completed_role_calls, 0);
            assert_eq!(done.backend_stats.unwrap().completed_nn_inputs, 0);
            assert!(done.physical_shutdown_confirmed && done.native_buffers_released);
            // A finished owner cannot publish a target after the final fence.
            assert_eq!(
                model.owner.request_game_target(4),
                Err(RoleError::Unavailable)
            );
            assert_eq!(model.owner.pending_game_target.load(Ordering::Acquire), 0);
        }
        #[test]
        fn reset_wrong_ack_preserves_first_known_failure_without_retry() {
            for backend_failure in [false, true] {
                let attempts = Arc::new(AtomicU64::new(0));
                let actual = Arc::clone(&attempts);
                let mut model = fixture_model(move |command| match command {
                    PalsNativeCommand::ResetTo(target) => {
                        actual.fetch_add(1, Ordering::AcqRel);
                        PhysicalRun::Complete(if backend_failure {
                            Err(binding_backend_error())
                        } else {
                            Ok(PalsNativeResult::ResetTo {
                                game_generation: *target - 1,
                            })
                        })
                    }
                    _ => binding_fixture_command(command),
                });
                model.new_game();
                let cancel = AtomicBool::new(false);
                let primary = model
                    .ensure_new_game(Instant::now() + Duration::from_secs(2), &cancel)
                    .unwrap_err();
                assert!(matches!(primary, RoleError::Backend(_)));
                assert!(model.unusable);
                assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 0);
                assert!(model.control_lease.is_none());
                assert_eq!(model.owner.pending_game_target.load(Ordering::Acquire), 2);
                let handle = model.finish_handle();
                assert_eq!(
                    handle
                        .finish(Instant::now() + Duration::from_secs(2))
                        .unwrap_err(),
                    primary
                );
                assert_eq!(
                    handle
                        .finish(Instant::now() + Duration::from_secs(2))
                        .unwrap_err(),
                    primary
                );
                assert_eq!(attempts.load(Ordering::Acquire), 1);
                let receipt = handle.receipt();
                assert!(!receipt.quarantined && receipt.physical_shutdown_confirmed);
                assert!(receipt.backend_stats.is_none());
                assert_eq!(receipt.completed_new_game_resets, 0);
            }
        }
        #[cfg(feature = "experimental-io-binding")]
        #[test]
        fn quit_resets_resident_generation_before_fresh_final_snapshot() {
            let (execution, mut snapshot) = resident_boundary_fixture();
            let stale: NativeCudaRecordPageSnapshotReceipt = snapshot.clone().into();
            let mut model = fixture_model_with_execution(
                move |command| {
                    PhysicalRun::Complete(Ok(match command {
                        PalsNativeCommand::ResetTo(target) => {
                            assert!(*target > snapshot.game_generation);
                            snapshot.game_generation = *target;
                            snapshot.live_blocks = 0;
                            snapshot.certified_projections = 0;
                            PalsNativeResult::ResetTo {
                                game_generation: *target,
                            }
                        }
                        PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                        PalsNativeCommand::SnapshotCudaRecordPages => {
                            PalsNativeResult::CudaRecordPagesObserved(Box::new(snapshot.clone()))
                        }
                        PalsNativeCommand::SnapshotStats => {
                            PalsNativeResult::Stats(PalsBackendStats::default())
                        }
                        _ => return unexpected_cuda_placement(),
                    }))
                },
                Some(execution),
            );
            *model.owner.cuda_record_page_snapshot.lock().unwrap() = Some(stale);
            *model.owner.final_stats.lock().unwrap() = Some(PalsBackendStats::default().into());
            model.new_game();
            model.new_game();
            assert!(
                model
                    .owner
                    .cuda_record_page_snapshot
                    .lock()
                    .unwrap()
                    .is_none()
            );
            let done = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert_eq!(done.game_generation, 3);
            assert_eq!(
                done.cuda_record_page_observation.unwrap().game_generation,
                3
            );
            assert_eq!(done.completed_new_game_resets, 1);
            assert_eq!(done.physically_completed_role_calls, 0);
        }
        #[test]
        fn reset_target_jump_and_nonmonotonic_ack_do_not_spend_newer_target() {
            let model = fixture_model(binding_fixture_command);
            model.owner.request_game_target(11).unwrap();
            assert_eq!(
                model.owner.request_game_target(11),
                Err(RoleError::InvalidOutput)
            );
            assert_eq!(
                model.owner.request_game_target(10),
                Err(RoleError::InvalidOutput)
            );
            assert!(model.owner.accept_game_target(2).is_err());
            assert_eq!(model.owner.pending_game_target.load(Ordering::Acquire), 11);
            let done = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert_eq!(done.game_generation, 11);
            assert_eq!(done.completed_new_game_resets, 1);
        }
        #[test]
        fn waiting_target_publication_rechecks_the_actual_finish_gate() {
            let model = fixture_model(binding_fixture_command);
            let owner = Arc::clone(&model.owner);
            let waiting = Arc::new(AtomicBool::new(false));
            let entered = Arc::clone(&waiting);
            let guard = model.owner.game_lifecycle.lock().unwrap();
            let publish = std::thread::spawn(move || {
                entered.store(true, Ordering::Release);
                owner.request_game_target(2)
            });
            let wait_until = Instant::now() + Duration::from_secs(2);
            while !waiting.load(Ordering::Acquire) {
                assert!(Instant::now() < wait_until);
                std::thread::yield_now();
            }
            // The producer began before finish, but cannot publish after its
            // synchronized flag. No stale pre-lock bool grants authority.
            model.owner.finishing.store(true, Ordering::Release);
            drop(guard);
            assert_eq!(publish.join().unwrap(), Err(RoleError::Unavailable));
            assert_eq!(model.owner.game_generation.load(Ordering::Acquire), 1);
            assert_eq!(model.owner.pending_game_target.load(Ordering::Acquire), 0);
            model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
        }
        #[test]
        fn canceled_reset_ready_keeps_actual_ack_without_nn_admission() {
            let cancel = Arc::new(AtomicBool::new(false));
            let signal = Arc::clone(&cancel);
            let mut model = fixture_model(move |command| {
                if matches!(command, PalsNativeCommand::ResetTo(_)) {
                    signal.store(true, Ordering::Release);
                }
                binding_fixture_command(command)
            });
            model.new_game();
            assert_eq!(
                model.ensure_new_game(Instant::now() + Duration::from_secs(2), &cancel),
                Err(RoleError::Canceled)
            );
            assert_eq!(model.owner.pending_game_target.load(Ordering::Acquire), 0);
            assert!(!model.pending_new_game);
            let done = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert_eq!(done.completed_new_game_resets, 1);
            assert_eq!(done.physically_completed_role_calls, 0);
        }
        #[test]
        fn late_reset_ready_applies_target_but_preserves_logical_deadline() {
            let (entered, actual_entry) = std::sync::mpsc::sync_channel(1);
            let (release, actual_release) = std::sync::mpsc::sync_channel(1);
            let mut model = fixture_model(move |command| {
                if matches!(command, PalsNativeCommand::ResetTo(_)) {
                    entered.send(()).unwrap();
                    actual_release.recv_timeout(Duration::from_secs(5)).unwrap();
                }
                binding_fixture_command(command)
            });
            model.drain_limit = Duration::from_secs(2);
            model.new_game();
            let cancel = AtomicBool::new(false);
            // Establish E only after the fixture and target exist. ACK release
            // requires actual admission/entry first, then observed E expiry;
            // a guessed worker sleep duration is never the late-Ready witness.
            let until = Instant::now() + Duration::from_secs(1);
            let release_after_expiry = std::thread::spawn(move || {
                actual_entry.recv_timeout(Duration::from_secs(2)).unwrap();
                while Instant::now() < until {
                    std::thread::sleep(Duration::from_millis(1));
                }
                assert!(Instant::now() >= until);
                release.send(()).unwrap();
            });
            assert_eq!(
                model.ensure_new_game(until, &cancel),
                Err(RoleError::Deadline)
            );
            release_after_expiry.join().unwrap();
            assert_eq!(model.owner.pending_game_target.load(Ordering::Acquire), 0);
            assert!(model.control_lease.is_none());
            assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 0);
            let done = model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert_eq!(done.completed_new_game_resets, 1);
            assert_eq!(done.physically_completed_role_calls, 0);
        }
        #[test]
        fn quit_reset_pending_at_whole_fence_retains_lease_and_target() {
            let (release, released) = std::sync::mpsc::sync_channel(1);
            let entered = Arc::new(AtomicBool::new(false));
            let actual = Arc::clone(&entered);
            let mut model = fixture_model(move |command| {
                if matches!(command, PalsNativeCommand::ResetTo(_)) {
                    actual.store(true, Ordering::Release);
                    released.recv_timeout(Duration::from_secs(2)).unwrap();
                }
                binding_fixture_command(command)
            });
            model.new_game();
            let handle = model.finish_handle();
            let whole = Instant::now() + Duration::from_millis(100);
            assert!(matches!(
                handle.finish(whole),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            assert!(entered.load(Ordering::Acquire));
            assert!(Instant::now() >= whole);
            assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 1);
            assert!(model.owner.stats_lease.lock().unwrap().is_some());
            assert_eq!(model.owner.pending_game_target.load(Ordering::Acquire), 2);
            let receipt = handle.receipt();
            assert!(receipt.quarantined && !receipt.physical_shutdown_confirmed);
            assert!(receipt.backend_stats.is_none());
            assert_eq!(receipt.completed_new_game_resets, 0);
            // Test-only mock release never promotes the original unknown fence.
            release.send(()).unwrap();
            let cleanup = Instant::now() + Duration::from_secs(2);
            loop {
                assert!(Instant::now() < cleanup);
                match model
                    .owner
                    .stats_lease
                    .lock()
                    .unwrap()
                    .as_mut()
                    .unwrap()
                    .poll()
                {
                    PhysicalPoll::Ready(Ok(PalsNativeResult::ResetTo { game_generation: 2 })) => {
                        break;
                    }
                    PhysicalPoll::Pending => std::thread::yield_now(),
                    _ => panic!("released mock must finish its exact original reset"),
                }
            }
            loop {
                assert!(Instant::now() < cleanup);
                match model.owner.worker.lock().unwrap().try_shutdown() {
                    Poll::Ready(Ok(())) => break,
                    Poll::Pending => std::thread::yield_now(),
                    _ => panic!("test-only mock reset shutdown failed"),
                }
            }
            assert_eq!(model.owner.in_flight.load(Ordering::Acquire), 1);
            assert_eq!(model.owner.pending_game_target.load(Ordering::Acquire), 2);
            assert!(!handle.receipt().physical_shutdown_confirmed);
        }
        #[test]
        fn canceled_logical_result_waits_for_physical_completion() {
            let cancel = Arc::new(AtomicBool::new(false));
            let signal = Arc::clone(&cancel);
            let mut model = fixture_model(move |command| match command {
                PalsNativeCommand::Evaluate(input)
                | PalsNativeCommand::EvaluateWithEvidence(input) => {
                    signal.store(true, Ordering::Release);
                    std::thread::sleep(Duration::from_millis(5));
                    PhysicalRun::Complete(Ok(output(input)))
                }
                PalsNativeCommand::NewGame => PhysicalRun::Complete(Ok(PalsNativeResult::NewGame)),
                PalsNativeCommand::ResetTo(target) => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::ResetTo {
                        game_generation: *target,
                    }))
                }
                PalsNativeCommand::SnapshotStats => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::Stats(PalsBackendStats::default())))
                }
                PalsNativeCommand::EvaluatePrivateWarm(_)
                | PalsNativeCommand::VerifyCudaPlacement
                | PalsNativeCommand::ObserveRuntimeMappings => unexpected_cuda_placement(),
                #[cfg(feature = "experimental-io-binding")]
                PalsNativeCommand::EvaluateCudaWarm(_) => unexpected_cuda_placement(),
                #[cfg(feature = "experimental-io-binding")]
                PalsNativeCommand::SnapshotCudaRecordPages => unexpected_cuda_placement(),
                PalsNativeCommand::VerifyRuntime => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::RuntimeVerified))
                }
            });
            let position = Position::startpos();
            let legal = position.legal_moves();
            assert!(matches!(
                model.propose(query(&position, &legal, &cancel)),
                Err(RoleError::Canceled)
            ));
            let receipt = model.finish_handle().receipt();
            assert_eq!(receipt.completed_role_inputs, 1);
            assert_eq!(receipt.delivered_role_inputs, 0);
            assert_eq!(receipt.search_consumed_role_inputs, 0);
            assert_eq!(receipt.canceled_requests, 1);
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert!(!receipt.quarantined);
            model
                .finish_handle()
                .finish(Instant::now() + Duration::from_secs(2))
                .unwrap();
        }
        #[test]
        fn physical_failure_and_invalid_head_do_not_count_as_consumed_inputs() {
            for invalid_head in [false, true] {
                let mut model = fixture_model(move |command| match command {
                    PalsNativeCommand::NewGame => {
                        PhysicalRun::Complete(Ok(PalsNativeResult::NewGame))
                    }
                    PalsNativeCommand::ResetTo(target) => {
                        PhysicalRun::Complete(Ok(PalsNativeResult::ResetTo {
                            game_generation: *target,
                        }))
                    }
                    PalsNativeCommand::SnapshotStats => PhysicalRun::Complete(Ok(
                        PalsNativeResult::Stats(PalsBackendStats::default()),
                    )),
                    PalsNativeCommand::EvaluatePrivateWarm(_)
                    | PalsNativeCommand::VerifyCudaPlacement
                    | PalsNativeCommand::ObserveRuntimeMappings => unexpected_cuda_placement(),
                    #[cfg(feature = "experimental-io-binding")]
                    PalsNativeCommand::EvaluateCudaWarm(_) => unexpected_cuda_placement(),
                    #[cfg(feature = "experimental-io-binding")]
                    PalsNativeCommand::SnapshotCudaRecordPages => unexpected_cuda_placement(),
                    PalsNativeCommand::VerifyRuntime => {
                        PhysicalRun::Complete(Ok(PalsNativeResult::RuntimeVerified))
                    }
                    PalsNativeCommand::Evaluate(input)
                    | PalsNativeCommand::EvaluateWithEvidence(input) => {
                        if invalid_head {
                            let PalsNativeResult::Evaluation(mut raw) = output(input) else {
                                unreachable!()
                            };
                            raw.wdl_logits[0] = f32::NAN;
                            PhysicalRun::Complete(Ok(PalsNativeResult::Evaluation(raw)))
                        } else {
                            PhysicalRun::Complete(Err(BackendError::new(
                                FailureKind::BackendFailure,
                                FailureStage::Backend,
                                "native lifecycle fixture failure",
                            )))
                        }
                    }
                });
                let position = Position::startpos();
                let legal = position.legal_moves();
                let cancel = AtomicBool::new(false);
                assert!(model.propose(query(&position, &legal, &cancel)).is_err());
                let receipt = model.finish_handle().receipt();
                assert_eq!(receipt.physically_completed_role_calls, 1);
                assert_eq!(receipt.completed_role_inputs, u64::from(invalid_head));
                assert_eq!(receipt.failed_physical_role_calls, u64::from(!invalid_head));
                assert_eq!(receipt.invalid_role_outputs, u64::from(invalid_head));
                assert_eq!(receipt.delivered_role_inputs, 0);
                assert_eq!(receipt.search_consumed_role_inputs, 0);
                assert_eq!(receipt.physical_runs_in_flight, 0);
                model
                    .finish_handle()
                    .finish(Instant::now() + Duration::from_secs(2))
                    .unwrap();
            }
        }
        #[test]
        fn unknown_completion_is_fatal_retains_pins_and_never_reopens_admission() {
            let mut model = fixture_model(|_| {
                PhysicalRun::Quarantined(BackendError::new(
                    FailureKind::BackendFailure,
                    FailureStage::Backend,
                    "unknown lifecycle fixture completion",
                ))
            });
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            assert!(matches!(
                model.propose(query(&position, &legal, &cancel)),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            model.new_game();
            assert!(matches!(
                model.propose(query(&position, &legal, &cancel)),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            let handle = model.finish_handle();
            assert!(matches!(
                handle.finish(Instant::now() + Duration::from_secs(1)),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            let receipt = handle.receipt();
            assert!(
                receipt.quarantined
                    && !receipt.physical_shutdown_confirmed
                    && !receipt.native_buffers_released
            );
            assert_eq!(receipt.physical_runs_in_flight, 1);
            assert_eq!(receipt.completed_role_inputs, 0);
            assert_eq!(receipt.request_high_water, 1);
            assert!(receipt.last_failure.is_some());
            assert!(receipt.backend_stats.is_none());
        }
        #[test]
        fn unknown_stats_completion_does_not_fabricate_nn_counters() {
            let mut model = fixture_model(|command| match command {
                PalsNativeCommand::Evaluate(input)
                | PalsNativeCommand::EvaluateWithEvidence(input) => {
                    PhysicalRun::Complete(Ok(output(input)))
                }
                PalsNativeCommand::NewGame => PhysicalRun::Complete(Ok(PalsNativeResult::NewGame)),
                PalsNativeCommand::ResetTo(target) => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::ResetTo {
                        game_generation: *target,
                    }))
                }
                PalsNativeCommand::SnapshotStats => PhysicalRun::Quarantined(BackendError::new(
                    FailureKind::BackendFailure,
                    FailureStage::Backend,
                    "unknown Stats fixture completion",
                )),
                PalsNativeCommand::EvaluatePrivateWarm(_)
                | PalsNativeCommand::VerifyCudaPlacement
                | PalsNativeCommand::ObserveRuntimeMappings => unexpected_cuda_placement(),
                #[cfg(feature = "experimental-io-binding")]
                PalsNativeCommand::EvaluateCudaWarm(_) => unexpected_cuda_placement(),
                #[cfg(feature = "experimental-io-binding")]
                PalsNativeCommand::SnapshotCudaRecordPages => unexpected_cuda_placement(),
                PalsNativeCommand::VerifyRuntime => {
                    PhysicalRun::Complete(Ok(PalsNativeResult::RuntimeVerified))
                }
            });
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            model.propose(query(&position, &legal, &cancel)).unwrap();
            model.accepted_output();
            let handle = model.finish_handle();
            assert!(matches!(
                handle.finish(Instant::now() + Duration::from_secs(2)),
                Err(RoleError::PhysicalCompletionUnknown)
            ));
            let receipt = handle.receipt();
            assert_eq!(receipt.completed_role_inputs, 1);
            assert_eq!(receipt.search_consumed_role_inputs, 1);
            assert!(receipt.backend_stats.is_none());
            assert!(receipt.backend_stats_observation.is_none());
            assert!(receipt.quarantined && !receipt.native_buffers_released);
            assert_eq!(receipt.physical_runs_in_flight, 1);
        }
        #[test]
        fn invalid_stats_ack_preserves_failure_but_cleans_known_idle_owner() {
            let mut model = fixture_model(|command| {
                PhysicalRun::Complete(Ok(match command {
                    PalsNativeCommand::Evaluate(input)
                    | PalsNativeCommand::EvaluateWithEvidence(input) => output(input),
                    PalsNativeCommand::NewGame => PalsNativeResult::NewGame,
                    PalsNativeCommand::ResetTo(target) => PalsNativeResult::ResetTo {
                        game_generation: *target,
                    },
                    PalsNativeCommand::EvaluatePrivateWarm(_)
                    | PalsNativeCommand::VerifyCudaPlacement
                    | PalsNativeCommand::ObserveRuntimeMappings => {
                        return unexpected_cuda_placement();
                    }
                    #[cfg(feature = "experimental-io-binding")]
                    PalsNativeCommand::EvaluateCudaWarm(_) => return unexpected_cuda_placement(),
                    #[cfg(feature = "experimental-io-binding")]
                    PalsNativeCommand::SnapshotCudaRecordPages => {
                        return unexpected_cuda_placement();
                    }
                    PalsNativeCommand::VerifyRuntime => PalsNativeResult::RuntimeVerified,
                    PalsNativeCommand::SnapshotStats => PalsNativeResult::Stats(PalsBackendStats {
                        completed_nn_inputs: 1,
                        ..PalsBackendStats::default()
                    }),
                }))
            });
            let position = Position::startpos();
            let legal = position.legal_moves();
            let cancel = AtomicBool::new(false);
            model.propose(query(&position, &legal, &cancel)).unwrap();
            let handle = model.finish_handle();
            assert!(matches!(
                handle.finish(Instant::now() + Duration::from_secs(2)),
                Err(RoleError::Backend(_))
            ));
            let receipt = handle.receipt();
            assert!(receipt.physical_shutdown_confirmed && receipt.native_buffers_released);
            assert!(!receipt.quarantined);
            assert!(receipt.backend_stats.is_none() && receipt.last_failure.is_some());
            assert_eq!(receipt.physical_runs_in_flight, 0);
            assert_eq!(receipt.completed_role_inputs, 1);
            assert!(
                handle
                    .finish(Instant::now() + Duration::from_secs(1))
                    .is_err()
            );
        }
    }
}
#[cfg(feature = "onnx-cpu")]
pub(crate) use native::validate_host_record_page_evidence;
#[cfg(feature = "onnx-cpu")]
pub(crate) use native::validate_private_warm_evidence;
#[cfg(feature = "onnx-cpu")]
pub(crate) use native::validate_private_warm_evidence_with_terminal_scope;
#[cfg(feature = "onnx-cpu")]
pub(crate) use native::validate_runtime_loading_mapping;
#[cfg(feature = "onnx-cpu")]
pub use native::{
    NativeBackendStatsReceipt, NativeCudaLoadingIdentity, NativeCudaRecordPageSnapshotReceipt,
    NativeCudaRecordPageStatsReceipt, NativeCudaRecordPagesDeclaration,
    NativeCudaRecordPagesRegistration, NativeCudaRecordPagesResourceDeclaration,
    NativeCudaWarmLoadError, NativeCudaWarmObservationLimits, NativeExecutionReceipt,
    NativeFailureReceipt, NativeHostRecordPageBankReceipt, NativeHostRecordPageCommandReceipt,
    NativeHostRecordPageDeclaration, NativeHostRecordPageLimits,
    NativeHostRecordPageObservationReceipt, NativeHostRecordPageSnapshotReceipt,
    NativeHostRecordPageStatsReceipt, NativeInvocationBudget, NativeLoadFailure, NativeLoadStage,
    NativeOwnerOptions, NativePreparedContext, NativePrivateWarmDeclaration,
    NativePrivateWarmPrepared, NativeRoleCompletionUnknown, NativeRoleExecutionBinding,
    NativeRoleFinishHandle, NativeRoleModel, NativeRoleObserver, NativeRoleReceipt,
    NativeRoleRejection, NativeRoleSourceIdentity, NativeRoleTerminal,
    NativeStartupCommandObservation, NativeStartupErrorKind, NativeStartupFailureDiagnostic,
    NativeStartupInputObservation, NativeStartupProbeReceipt, NativeStartupTimingObservation,
};

#[cfg(test)]
mod tests {
    use super::*;
    use rz_search::pals::store::StateId;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;
    #[test]
    fn only_typed_runtime_admission_expired_is_a_role_deadline() {
        use rz_contracts::{ContractError, ErrorCode, Stage};

        let codes = [
            ErrorCode::InvalidInput,
            ErrorCode::UnsupportedContract,
            ErrorCode::BackendUnavailable,
            ErrorCode::ResourceExhausted,
            ErrorCode::NumericalFailure,
            ErrorCode::IdentityMismatch,
            ErrorCode::BackendFailure,
            ErrorCode::Canceled,
            ErrorCode::Expired,
            ErrorCode::Stale,
        ];
        let stages = [
            Stage::Contract,
            Stage::Admission,
            Stage::Backend,
            Stage::Output,
            Stage::Backup,
        ];
        for detail in ["deadline has expired", "PALS runtime scheduling fault"] {
            for stage in stages {
                for code in codes {
                    let error = ContractError::new(code, stage, detail);
                    if stage == Stage::Admission && code == ErrorCode::Expired {
                        assert!(matches!(
                            runtime_admission_error(error),
                            RoleError::Deadline
                        ));
                    } else {
                        assert!(matches!(
                            runtime_admission_error(error),
                            RoleError::Backend(message) if message == error.to_string()
                        ));
                    }
                }
            }
        }
        assert!(matches!(
            model_error("Admission/Expired: deadline has expired"),
            RoleError::Backend(_)
        ));
    }
    #[test]
    fn product_input_meaning_requires_the_exact_rules_profile() {
        let expected = pals_rules_encoding_semantic_digest();
        assert!(verify_pals_rules_profile(Some(PALS_RULES_ENCODING), Some(expected)).is_ok());
        assert!(verify_pals_rules_profile(None, None).is_err());
        assert!(verify_pals_rules_profile(Some(PALS_RULES_ENCODING), None).is_err());
        assert!(verify_pals_rules_profile(Some("different_fields"), Some(expected)).is_err());
        assert!(verify_pals_rules_profile(Some(PALS_RULES_ENCODING), Some([0; 32])).is_err());
        assert_ne!(pals_native_source_digest(), [0; 32]);
    }
    fn full_record(revision: u64, line: Vec<BoardMove>) -> RoleRecord {
        RoleRecord {
            revision,
            parent_revision: None,
            supersedes_revision: None,
            origin_state: StateId(65_000),
            kind: RecordKind::Proposal,
            line,
            value: None,
            completed_depth: 0,
            score_scope: None,
            cpu_observation: None,
            perspective: Color::White,
            critical: false,
        }
    }
    #[test]
    fn full_line_middle_moves_are_observed_and_identity_slots_are_disabled() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let moves = |middle: &str, back: &str| {
            ["e2e4", middle, "g1f3", back, "f1e2"]
                .into_iter()
                .map(|value| BoardMove::from_uci(value).unwrap())
                .collect::<Vec<_>>()
        };
        let first = vec![full_record(7, moves("g8f6", "f6g8"))];
        let second = vec![full_record(7, moves("b8c6", "c6b8"))];
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &first[0].line,
            counterexample: Some(&first[0].line),
            records: &first,
            revision: 8,
            deadline: Instant::now() + Duration::from_secs(1),
            cancel: &cancel,
        };
        let config = PalsModelConfig::full_line_interaction_v2();
        let a = prepare_role_input_for_config(&query, NativeQueryKind::Repair, [1; 32], &config)
            .unwrap();
        let b = prepare_role_input_for_config(
            &RoleQuery {
                records: &second,
                ..query
            },
            NativeQueryKind::Repair,
            [1; 32],
            &config,
        )
        .unwrap();
        // Summaries have equal length/endpoints; the full sequence makes these
        // different real inputs rather than a hash surrogate for the evidence.
        assert_eq!(a.records, b.records);
        assert_eq!(a.query, b.query);
        assert_eq!(a.records[0].features[6], 0.0);
        assert_eq!(a.query[6..8], [0.0, 0.0]);
        assert_eq!(
            a.full_line.as_ref().unwrap().query_proposal,
            first[0]
                .line
                .iter()
                .copied()
                .map(candidate)
                .collect::<Vec<_>>()
        );
        assert_ne!(
            a.canonical_input_key(&config).unwrap(),
            b.canonical_input_key(&config).unwrap()
        );
        assert_eq!(pals_rules_semantic_fields().count(), 58);
        assert_eq!(pals_rules_semantic_fields_for_config(&config).count(), 67);
        assert!(
            verify_pals_rules_profile_for_config(
                Some(PALS_RULES_ENCODING),
                Some(pals_rules_encoding_semantic_digest()),
                &config
            )
            .is_err()
        );
    }
    #[test]
    fn full_line_relationships_are_local_and_only_direct_critical_relations_are_required() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let mut history: Vec<_> = (1..=130)
            .map(|revision| {
                let mut record = full_record(revision, vec![]);
                record.parent_revision = (revision > 1).then_some(revision - 1);
                record
            })
            .collect();
        let config = PalsModelConfig::full_line_interaction_v2();
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &history,
            revision: 130,
            deadline: Instant::now() + Duration::from_secs(1),
            cancel: &cancel,
        };
        let input =
            prepare_role_input_for_config(&query, NativeQueryKind::Propose, [1; 32], &config)
                .unwrap();
        let lines = input.full_line.unwrap().records;
        assert_eq!(lines.len(), 128);
        assert_eq!(history[2].parent_revision, Some(2)); // raw relation retained
        assert_eq!(lines[0].parent, None); // historical relation outside model view
        assert!(!lines[0].parent_required);
        assert_eq!(lines[127].parent, Some(126));
        assert!(lines[127].parent_required);
        history[129].parent_revision = Some(999);
        let missing = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &history,
            revision: 130,
            deadline: Instant::now() + Duration::from_secs(1),
            cancel: &cancel,
        };
        assert!(
            prepare_role_input_for_config(&missing, NativeQueryKind::Propose, [1; 32], &config)
                .is_err()
        );
    }
    #[test]
    fn actual_rules_board_rights_history_and_legal_order() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &[],
            revision: 0,
            deadline: Instant::now() + Duration::from_secs(1),
            cancel: &cancel,
        };
        let input = prepare_role_input(&query, NativeQueryKind::Propose, [1; 32]).unwrap();
        assert_eq!(input.board[0], 4);
        assert_eq!(input.board[60], 12);
        assert_eq!(input.metadata[1..5], [1.0; 4]);
        assert_eq!(input.candidates.len(), legal.len());
        let fen = Position::from_fen(&position.to_fen()).unwrap();
        assert_ne!(
            pals_history_digest(&position).unwrap(),
            pals_history_digest(&fen).unwrap()
        );
    }
    #[test]
    fn unknown_records_are_critical_and_overflow_is_explicit() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let cancel = AtomicBool::new(false);
        let unknown: Vec<_> = (0..129)
            .map(|i| RoleRecord {
                revision: i,
                parent_revision: None,
                supersedes_revision: None,
                origin_state: StateId(0),
                kind: RecordKind::Counterexample,
                line: vec![],
                value: None,
                completed_depth: 0,
                score_scope: None,
                cpu_observation: None,
                perspective: Color::White,
                critical: true,
            })
            .collect();
        let query = RoleQuery {
            position: &position,
            legal: &legal,
            prefix: &[],
            proposal: &[],
            counterexample: None,
            records: &unknown,
            revision: 129,
            deadline: Instant::now() + Duration::from_secs(1),
            cancel: &cancel,
        };
        assert!(prepare_role_input(&query, NativeQueryKind::Propose, [1; 32]).is_err());
        let query = RoleQuery {
            records: &unknown[..128],
            ..query
        };
        let input = prepare_role_input(&query, NativeQueryKind::Propose, [1; 32]).unwrap();
        assert_eq!(input.required_critical_records.len(), 128);
        assert_eq!(input.records[0].features[1], 0.0);
        assert_eq!(input.records[0].features[15], 1.0);
        let old_unknown: Vec<_> = unknown
            .iter()
            .cloned()
            .map(|mut record| {
                record.critical = false;
                record
            })
            .collect();
        let query = RoleQuery {
            records: &old_unknown,
            ..query
        };
        let input = prepare_role_input(&query, NativeQueryKind::Propose, [1; 32]).unwrap();
        // Historical unknowns are not all current critical commitments. Their
        // status remains explicit, but only the latest kind is mandatorily pinned.
        assert_eq!(input.records.len(), 128);
        assert_eq!(input.required_critical_records, [129]);
        assert!(input.records.iter().all(|r| r.features[15] == 1.0));
    }
    #[test]
    fn divergence_is_actual_replayed_opponent_prefix() {
        let root = Position::startpos();
        let proposal: Vec<_> = ["e2e4", "e7e5", "g1f3", "b8c6"]
            .into_iter()
            .map(|m| BoardMove::from_uci(m).unwrap())
            .collect();
        let cancel = AtomicBool::new(false);
        let query = DivergenceQuery {
            root: &root,
            proposal: &proposal,
            candidates: &[1, 3],
            records: &[],
            revision: 0,
            deadline: Instant::now() + Duration::from_secs(1),
            cancel: &cancel,
        };
        let input = prepare_divergence_input(&query, [1; 32]).unwrap();
        assert_eq!(input.divergence_features.len(), 2);
        assert!(input.candidates.is_empty());
        assert_eq!(input.divergence_features[0][1], 1.0);
        let query = DivergenceQuery {
            candidates: &[0],
            ..query
        };
        assert!(prepare_divergence_input(&query, [1; 32]).is_err());
    }
    #[test]
    fn all_promotion_tokens_roundtrip_to_common_move16() {
        for (piece, bits) in [
            (PieceKind::Queen, 1),
            (PieceKind::Rook, 2),
            (PieceKind::Bishop, 3),
            (PieceKind::Knight, 4),
        ] {
            let movement = BoardMove::new(
                Square::new(48).unwrap(),
                Square::new(56).unwrap(),
                Some(piece),
            )
            .unwrap();
            let token = candidate(movement);
            assert_eq!(token.promotion, bits);
            assert_eq!(
                PalsCandidateToken::from_packed(token.packed().unwrap()).unwrap(),
                token
            );
        }
    }
}
