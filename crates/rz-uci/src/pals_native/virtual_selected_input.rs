//! Pure V3 projection of the immutable Search-selected review view.
//!
//! Rules snapshots retain exact history and anchor-local relationships. This
//! module performs no model/backend execution and grants no ticket authority.
use crate::pals_native;
use rz_eval::pals_model::PalsRole;
use rz_eval::pals_model::virtual_review::{
    self as model, PalsVirtualReviewInput, ReviewAnchorInput, ReviewExecutionIdentity,
    ReviewQueryInput, ReviewQueryPurpose, ReviewRecordInput, ReviewRecordKind,
};
use rz_position::{BoardMove, Color, PlayStatus, Position, PositionLimits};
use rz_search::cpu::CpuScoreScope;
use rz_search::pals::engine::{
    RecordKind, RoleError, RoleLogicalContext, RoleQueryPurpose, role_divergence_sha256,
    role_line_sha256,
};
use rz_search::pals::review_context::ReviewView;

/// Prepare exact selected-review V3 inputs without loading a neural backend.
/// The caller retains ticket/seal authority; this validates only the immutable
/// view, Rules endpoint and logical query context before projecting tensors.
pub fn prepare_selected_virtual_review_input(
    position: &Position,
    context: &RoleLogicalContext,
    view: &ReviewView,
    model_epoch: [u8; 32],
    backend: [u8; 32],
) -> Result<(PalsVirtualReviewInput, ReviewExecutionIdentity), RoleError> {
    let q = view.query();
    if view.generation() != context.search_generation
        || q.purpose != context.purpose
        || q.prefix != context.prefix
        || context.prefix_sha256 != ordered_line_digest(b"prefix", &q.prefix)?
        || context.proposal_sha256 != ordered_line_digest(b"proposal", &q.proposal)?
        || context.refutation_sha256
            != q.counterexample
                .as_deref()
                .map(|line| ordered_line_digest(b"refutation", line))
                .transpose()?
        || context.divergence_sha256 != divergence_digest(&q.divergences)?
        || view.records().len() != view.records_metadata().len()
    {
        return Err(RoleError::InvalidOutput);
    }
    let first = view.anchors().first().ok_or(RoleError::InvalidOutput)?;
    if first.state != view.root_state() {
        return Err(RoleError::InvalidOutput);
    }
    // The Search anchor snapshots carry complete known Rules history. FEN-only
    // reconstruction would lose repetition evidence and is deliberately absent.
    let positions = view
        .anchors()
        .iter()
        .map(|anchor| {
            Position::from_snapshot(&anchor.snapshot, PositionLimits::default())
                .map_err(|_| RoleError::InvalidOutput)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let anchors = positions
        .iter()
        .map(exact_review_anchor_input)
        .collect::<Result<Vec<_>, RoleError>>()?;
    let anchor_history_seals = positions
        .iter()
        .map(pals_native::pals_history_digest)
        .collect::<Result<Vec<_>, _>>()?;
    // The model's anchor zero is the exact root; the request position is the
    // Rules endpoint of its explicit virtual prefix, not necessarily that root.
    let root = positions.first().ok_or(RoleError::InvalidOutput)?;
    let rules_history = anchor_history_seals[0];
    let mut endpoint = root.clone();
    for movement in &q.prefix {
        endpoint
            .make_move(*movement)
            .map_err(|_| RoleError::InvalidOutput)?;
    }
    if !endpoint.snapshot().same_state(&position.snapshot()) {
        return Err(RoleError::InvalidOutput);
    }
    let records = view
        .records()
        .iter()
        .zip(view.records_metadata())
        .map(|(record, meta)| {
            if record.revision > context.public_revision {
                return Err(RoleError::InvalidOutput);
            }
            let anchor = positions
                .get(meta.anchor as usize)
                .ok_or(RoleError::InvalidOutput)?;
            let mut endpoint = anchor.clone();
            for movement in &record.line {
                endpoint
                    .make_move(*movement)
                    .map_err(|_| RoleError::InvalidOutput)?;
            }
            let terminal = matches!(
                endpoint
                    .classify_position()
                    .map_err(|_| RoleError::InvalidOutput)?
                    .play_status,
                PlayStatus::Terminal { .. }
            );
            Ok(ReviewRecordInput {
                kind: match record.kind {
                    RecordKind::Proposal => ReviewRecordKind::Proposal,
                    RecordKind::Counterexample => ReviewRecordKind::Counterexample,
                    RecordKind::Repair => ReviewRecordKind::Repair,
                    RecordKind::CpuVerification => ReviewRecordKind::CpuVerification,
                },
                perspective: perspective(record.perspective),
                completed_iteration: record.score_scope == Some(CpuScoreScope::CompletedIteration),
                rules_terminal: terminal,
                line: record
                    .line
                    .iter()
                    .copied()
                    .map(pals_native::candidate)
                    .collect(),
                anchor: meta.anchor,
                parent: meta.parent,
                supersedes: meta.supersedes,
            })
        })
        .collect::<Result<Vec<_>, RoleError>>()?;
    let role = match q.purpose {
        RoleQueryPurpose::ProposePolicy
        | RoleQueryPurpose::RepairPolicy
        | RoleQueryPurpose::ValueFresh => PalsRole::Proposer,
        RoleQueryPurpose::ReplyPolicy | RoleQueryPurpose::DivergencePolicy => PalsRole::Critic,
        RoleQueryPurpose::TaskPolicy => PalsRole::Validator,
    };
    let legal = position.legal_moves();
    // Contract policy order belongs to the immutable request, not the Rules
    // generator. Verify complete membership/uniqueness without reordering it.
    let legal_matches = (q.legal.len() == legal.len()
        && q.legal.iter().enumerate().all(|(index, movement)| {
            legal.contains(movement) && !q.legal[..index].contains(movement)
        }))
        || (q.purpose == RoleQueryPurpose::TaskPolicy && q.legal.is_empty());
    if q.purpose != RoleQueryPurpose::DivergencePolicy
        && (!legal_matches || !q.divergences.is_empty())
    {
        return Err(RoleError::InvalidOutput);
    }
    if q.purpose == RoleQueryPurpose::DivergencePolicy
        && (role != PalsRole::Critic || !q.legal.is_empty())
    {
        return Err(RoleError::InvalidOutput);
    }
    for line in [&q.prefix, &q.proposal]
        .into_iter()
        .chain(q.counterexample.iter())
    {
        let mut replay = root.clone();
        for movement in line {
            replay
                .make_move(*movement)
                .map_err(|_| RoleError::InvalidOutput)?;
        }
    }
    let input = PalsVirtualReviewInput {
        schema: model::INPUT_SCHEMA.into(),
        role,
        anchors,
        records,
        query: ReviewQueryInput {
            purpose: match q.purpose {
                RoleQueryPurpose::ProposePolicy => ReviewQueryPurpose::ProposePolicy,
                RoleQueryPurpose::ReplyPolicy => ReviewQueryPurpose::ReplyPolicy,
                RoleQueryPurpose::RepairPolicy => ReviewQueryPurpose::RepairPolicy,
                RoleQueryPurpose::DivergencePolicy => ReviewQueryPurpose::DivergencePolicy,
                RoleQueryPurpose::ValueFresh => ReviewQueryPurpose::ValueFresh,
                RoleQueryPurpose::TaskPolicy => ReviewQueryPurpose::TaskPolicy,
            },
            perspective: perspective(position.side_to_move()),
            counter_present: q.counterexample.is_some(),
            prefix: q
                .prefix
                .iter()
                .copied()
                .map(pals_native::candidate)
                .collect(),
            proposal: q
                .proposal
                .iter()
                .copied()
                .map(pals_native::candidate)
                .collect(),
            counter: q
                .counterexample
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .copied()
                .map(pals_native::candidate)
                .collect(),
            anchors: [0; 3],
        },
        candidates: q
            .legal
            .iter()
            .copied()
            .map(pals_native::candidate)
            .collect(),
        divergences: encode_divergences(root, &q.proposal, &q.divergences)?,
        token_budget: view.token_budget(),
    };
    let admission = input.validate().map_err(|_| RoleError::InvalidOutput)?;
    if admission.token_count != view.token_count() as usize
        || admission.length_bucket != view.length_bucket() as usize
    {
        return Err(RoleError::InvalidOutput);
    }
    Ok((
        input,
        ReviewExecutionIdentity {
            model_epoch,
            backend,
            rules_history,
            anchor_history_seals,
            fp32: true,
        },
    ))
}

fn perspective(color: Color) -> i8 {
    if color == Color::White { 1 } else { -1 }
}

/// Pure original Rules board/metadata calculation. It issues no ReviewView,
/// ticket, native worker, physical-completion or training authority.
pub fn exact_review_anchor_input(position: &Position) -> Result<ReviewAnchorInput, RoleError> {
    let (board, metadata) = pals_native::board_and_metadata(position)?;
    Ok(ReviewAnchorInput { board, metadata })
}

fn ordered_line_digest(domain: &[u8], line: &[BoardMove]) -> Result<[u8; 32], RoleError> {
    role_line_sha256(domain, line).map_err(|_| RoleError::InvalidOutput)
}
fn divergence_digest(indices: &[usize]) -> Result<[u8; 32], RoleError> {
    role_divergence_sha256(indices).map_err(|_| RoleError::InvalidOutput)
}
fn encode_divergences(
    root: &Position,
    proposal: &[BoardMove],
    indices: &[usize],
) -> Result<Vec<[f32; 8]>, RoleError> {
    let mut seen = std::collections::BTreeSet::new();
    if indices.iter().any(|index| !seen.insert(*index)) {
        return Err(RoleError::InvalidOutput);
    }
    let mut position = root.clone();
    let mut by_ply = Vec::with_capacity(proposal.len());
    for (ply, movement) in proposal.iter().copied().enumerate() {
        let token = pals_native::candidate(movement);
        by_ply.push([
            ply as f32 / 256.0,
            u8::from(position.side_to_move() != root.side_to_move()) as f32,
            u8::from(position.in_check()) as f32,
            token.from as f32 / 63.0,
            token.to as f32 / 63.0,
            token.promotion as f32 / 4.0,
            position.halfmove_clock() as f32 / 150.0,
            position.legal_moves().len() as f32 / 256.0,
        ]);
        position
            .make_move(movement)
            .map_err(|_| RoleError::InvalidOutput)?;
    }
    indices
        .iter()
        .map(|index| {
            let feature = *by_ply.get(*index).ok_or(RoleError::InvalidOutput)?;
            if feature[1] != 1.0 {
                return Err(RoleError::InvalidOutput);
            }
            Ok(feature)
        })
        .collect()
}

/// Reuse the original bounded Rules/proposal/index calculation for historical
/// comparison. This function grants no live or historical execution authority.
#[cfg(feature = "onnx-cpu")]
pub fn encode_review_divergences(
    root: &Position,
    proposal: &[BoardMove],
    indices: &[usize],
) -> Result<Vec<[f32; 8]>, RoleError> {
    if proposal.len() > 256 || indices.len() > 256 {
        return Err(RoleError::InvalidOutput);
    }
    encode_divergences(root, proposal, indices)
}
