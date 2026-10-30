//! 과거 selected-view 자료의 순수 Rules·메타데이터 검증과 digest 재계산.
//!
//! 이 모듈은 `ReviewView`, Store pin, source owner, native 실행·소비 권한이나
//! 학습 target을 발급하지 않는다. 같은 숫자 ID나 digest가 현재 Store에 속한다는
//! 주장도 하지 않는다. 호출자는 원 publication의 무결성과 출처를 별도로 검증한다.
//!
//! 원 record 순서의 source anchor도 요구하며, 실제 Rules 이력의 차분으로 원
//! prefix와 선택된 suffix를 연결한다. FEN 일치나 숫자 state ID만으로 원 prefix를
//! 추정하지 않는다. 이 재생 검증도 원 관측자의 진위나 live 소비 권한은 증명하지 않는다.

use std::collections::{BTreeMap, BTreeSet};

use rz_position::{MAX_UCI_REPLAY_PLIES, Position, PositionLimits};

use super::{
    MAX_LINE, ReviewAnchor, ReviewAnchorMode, ReviewError, ReviewQueryInput, ReviewRecordMetadata,
    ReviewSelectionPolicy, RoleRecord, StateId, checked_rebase_offset, checked_replay,
    input_digest, record_scope_digest, snapshot_digest, view_seal_digest,
};

/// 원 자료의 읽기 전용 projection이다. 필드의 숫자 ID는 원 자료의 역사 메타데이터며
/// 현재 Store·worker·모델의 authority가 아니다. 원 선택 순서를 그대로 전달한다.
#[derive(Clone, Copy, Debug)]
pub struct HistoricalReviewDescriptor<'a> {
    pub sequence: u64,
    pub generation: u64,
    pub root_state: StateId,
    /// 의미 digest는 기존 정책대로 required_bundle을 제외한다. 해당 제출 권한의
    /// 원 bundle 소유권·포함 여부는 이 계산으로 복원하지 않으며 원 publication이 보존한다.
    pub policy: ReviewSelectionPolicy,
    pub anchors: &'a [ReviewAnchor],
    /// 원 publication에서 복원한 source snapshot으로, record와 같은 순서다.
    /// 선택 후 anchor 0으로 rebase된 record도 원 source를 별도로 유지한다.
    pub source_anchors: &'a [ReviewAnchor],
    pub records: &'a [RoleRecord],
    pub metadata: &'a [ReviewRecordMetadata],
    pub query: &'a ReviewQueryInput,
    pub tokens: u16,
    pub bucket: u16,
}

/// 단순한 계산 결과다. 어느 필드도 현재 실행·수락·물리 완료·target 권한이 아니다.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistoricalReviewDigests {
    pub input_digest: [u8; 32],
    pub view_seal_digest: [u8; 32],
    pub record_scope_digest: [u8; 32],
    pub policy_digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HistoricalReviewError {
    InvalidDescriptor(&'static str),
    DigestMismatch(&'static str),
    Review(ReviewError),
}

impl From<ReviewError> for HistoricalReviewError {
    fn from(value: ReviewError) -> Self {
        Self::Review(value)
    }
}

impl std::fmt::Display for HistoricalReviewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PALS historical review metadata: {self:?}")
    }
}

impl std::error::Error for HistoricalReviewError {}

fn require(condition: bool, message: &'static str) -> Result<(), HistoricalReviewError> {
    if condition {
        Ok(())
    } else {
        Err(HistoricalReviewError::InvalidDescriptor(message))
    }
}

/// 기존 selected-view의 hash framing을 그대로 사용한다. 별도 history/hash codec,
/// Store, index 또는 모델을 만들지 않으며 정상 자료를 정렬·수정하지 않는다.
pub fn calculate_historical_review_digests(
    descriptor: HistoricalReviewDescriptor<'_>,
) -> Result<HistoricalReviewDigests, HistoricalReviewError> {
    validate_descriptor(descriptor)?;
    let input = input_digest(
        descriptor.anchors,
        descriptor.records,
        descriptor.metadata,
        descriptor.query,
        descriptor.policy,
        descriptor.tokens,
        descriptor.bucket,
    )?;
    Ok(HistoricalReviewDigests {
        input_digest: input,
        view_seal_digest: view_seal_digest(
            input,
            descriptor.sequence,
            descriptor.generation,
            descriptor.root_state,
            descriptor.metadata,
        ),
        record_scope_digest: record_scope_digest(
            descriptor.anchors,
            descriptor.records,
            descriptor.metadata,
            descriptor.policy,
        )?,
        policy_digest: descriptor.policy.digest(),
    })
}

/// 기대한 원 digest들과의 일치만 검사한다. 해시 일치가 원 source의 진위 또는
/// live capability의 복원을 뜻하지 않으며 원 publication 검사를 대신하지 않는다.
pub fn verify_historical_review_digests(
    descriptor: HistoricalReviewDescriptor<'_>,
    expected: &HistoricalReviewDigests,
) -> Result<(), HistoricalReviewError> {
    let actual = calculate_historical_review_digests(descriptor)?;
    for (name, actual, expected) in [
        ("input", actual.input_digest, expected.input_digest),
        (
            "view seal",
            actual.view_seal_digest,
            expected.view_seal_digest,
        ),
        (
            "record scope",
            actual.record_scope_digest,
            expected.record_scope_digest,
        ),
        ("policy", actual.policy_digest, expected.policy_digest),
    ] {
        if actual != expected {
            return Err(HistoricalReviewError::DigestMismatch(name));
        }
    }
    Ok(())
}

fn validate_descriptor(
    descriptor: HistoricalReviewDescriptor<'_>,
) -> Result<(), HistoricalReviewError> {
    descriptor.policy.validate()?;
    require(descriptor.sequence != 0, "zero unissued view sequence")?;
    require(
        !descriptor.anchors.is_empty()
            && descriptor.anchors.len() <= usize::from(descriptor.policy.max_anchors)
            && descriptor.anchors[0].state == descriptor.root_state,
        "anchor count or exact root ordering",
    )?;
    require(
        descriptor.records.len() == descriptor.metadata.len()
            && descriptor.records.len() == descriptor.source_anchors.len()
            && descriptor.records.len() <= usize::from(descriptor.policy.max_records),
        "record count or metadata pairing",
    )?;
    let mut anchor_states = BTreeSet::new();
    let mut anchor_digests = BTreeSet::new();
    let mut state_histories = BTreeMap::new();
    for anchor in descriptor.anchors {
        let digest = snapshot_digest(&anchor.snapshot)?;
        require(
            anchor_states.insert(anchor.state) && anchor_digests.insert(digest),
            "duplicate selected anchor",
        )?;
        state_histories.insert(anchor.state, digest);
        // Replay validation is delegated to Rules; a FEN-only substitute would
        // lose the exact historical trace included by snapshot_digest.
        checked_replay(&anchor.snapshot, &[])?;
    }
    for anchor in descriptor.source_anchors {
        let digest = snapshot_digest(&anchor.snapshot)?;
        require(
            state_histories
                .insert(anchor.state, digest)
                .is_none_or(|previous| previous == digest),
            "historical state ID has contradictory exact histories",
        )?;
    }
    let query = descriptor.query;
    let query_lines = [
        query.prefix.as_slice(),
        query.proposal.as_slice(),
        query.counterexample.as_deref().unwrap_or(&[]),
    ];
    require(
        query_lines.iter().all(|line| line.len() <= MAX_LINE)
            && query.legal.len() <= 256
            && query.divergences.len() <= 256,
        "query line, candidate or divergence capacity",
    )?;
    let root = &descriptor.anchors[0].snapshot;
    for line in query_lines {
        checked_replay(root, line)?;
    }
    let query_position = checked_replay(root, &query.prefix)?;
    let legal = Position::from_snapshot(&query_position, PositionLimits::default())
        .map_err(|_| ReviewError::InvalidLine)?
        .legal_moves();
    let mut candidates = BTreeSet::new();
    for movement in &query.legal {
        require(
            legal.contains(movement)
                && candidates.insert(
                    super::Move16::pack(*movement)
                        .map_err(ReviewError::from)?
                        .bits(),
                ),
            "illegal or duplicate ordered query candidate",
        )?;
    }
    let mut divergences = BTreeSet::new();
    for ply in &query.divergences {
        require(
            *ply < query.proposal.len() && divergences.insert(*ply),
            "out-of-line or duplicate divergence position",
        )?;
    }
    let mut revisions = BTreeSet::new();
    let mut used_anchors = BTreeSet::from([0]);
    let mut tokens = query_lines.iter().map(|line| line.len()).sum::<usize>();
    let mut longest = query_lines.iter().map(|line| line.len()).max().unwrap_or(0);
    for (local, (record, metadata)) in descriptor
        .records
        .iter()
        .zip(descriptor.metadata)
        .enumerate()
    {
        let anchor = descriptor
            .anchors
            .get(usize::from(metadata.anchor))
            .ok_or(HistoricalReviewError::InvalidDescriptor("record anchor"))?;
        let source = &descriptor.source_anchors[local];
        require(
            record.origin_state == anchor.state
                && source.state == metadata.source_state
                && record.revision == metadata.source_revision
                && revisions.insert(record.revision)
                && usize::from(metadata.source_line_offset)
                    .checked_add(record.line.len())
                    .is_some_and(|length| length <= MAX_LINE),
            "record anchor, revision or original line length",
        )?;
        validate_original_source(source, anchor, record, metadata)?;
        require(
            descriptor.policy.mode != ReviewAnchorMode::CurrentRootOnly || metadata.anchor == 0,
            "legacy view cannot contain an old anchor",
        )?;
        if metadata.source_line_offset == 0 {
            require(
                record.line.first().copied() == metadata.original_first_move,
                "original first move differs from unrebased line",
            )?;
        } else {
            require(
                metadata.anchor == 0 && metadata.original_first_move.is_some(),
                "rebased suffix lacks original first move or uses old anchor",
            )?;
        }
        for (relation, revision) in [
            (metadata.parent, record.parent_revision),
            (metadata.supersedes, record.supersedes_revision),
        ] {
            match (relation, revision) {
                (None, None) => {}
                (Some(relation), Some(revision)) if usize::from(relation) < local => {
                    let parent = &descriptor.records[usize::from(relation)];
                    let parent_metadata = &descriptor.metadata[usize::from(relation)];
                    require(
                        parent.revision == revision
                            && revision < record.revision
                            && parent_metadata.anchor == metadata.anchor
                            && parent_metadata.source_state == metadata.source_state
                            && parent_metadata.original_first_move == metadata.original_first_move,
                        "relation leaves original coherent bundle",
                    )?;
                }
                _ => {
                    return Err(HistoricalReviewError::InvalidDescriptor(
                        "missing or forward local relation",
                    ));
                }
            }
        }
        checked_replay(&anchor.snapshot, &record.line)?;
        used_anchors.insert(usize::from(metadata.anchor));
        tokens += record.line.len();
        longest = longest.max(record.line.len());
    }
    require(
        used_anchors.len() == descriptor.anchors.len(),
        "unused selected anchor",
    )?;
    require(
        tokens == usize::from(descriptor.tokens)
            && tokens <= usize::from(descriptor.policy.token_budget),
        "actual repeated query and record token total",
    )?;
    let bucket = [8, 16, 32, 64, 128, 256]
        .into_iter()
        .find(|bucket| longest <= *bucket)
        .ok_or(HistoricalReviewError::InvalidDescriptor("line bucket"))?;
    require(
        usize::from(descriptor.bucket) == bucket,
        "non-minimal or unsupported physical length bucket",
    )
}

fn validate_original_source(
    source: &ReviewAnchor,
    selected: &ReviewAnchor,
    record: &RoleRecord,
    metadata: &ReviewRecordMetadata,
) -> Result<(), HistoricalReviewError> {
    let original_line = if source.snapshot.same_state(&selected.snapshot) {
        require(
            metadata.source_line_offset == 0,
            "same exact source anchor cannot strip a prefix",
        )?;
        record.line.clone()
    } else {
        let source_trace = source
            .snapshot
            .uci_replay(MAX_UCI_REPLAY_PLIES)
            .map_err(|_| ReviewError::AnchorMismatch)?;
        let selected_trace = selected
            .snapshot
            .uci_replay(MAX_UCI_REPLAY_PLIES)
            .map_err(|_| ReviewError::AnchorMismatch)?;
        require(
            source_trace.origin == selected_trace.origin
                && source_trace.completeness == selected_trace.completeness
                && source_trace.start_fen == selected_trace.start_fen
                && selected_trace.moves.starts_with(&source_trace.moves),
            "source and selected anchor have unrelated exact histories",
        )?;
        let prefix = &selected_trace.moves[source_trace.moves.len()..];
        require(
            prefix.len() == usize::from(metadata.source_line_offset)
                && prefix.len() + record.line.len() <= MAX_LINE,
            "source line offset differs from actual Rules history delta",
        )?;
        let mut original = Vec::with_capacity(prefix.len() + record.line.len());
        original.extend_from_slice(prefix);
        original.extend_from_slice(&record.line);
        original
    };
    require(
        original_line.first().copied() == metadata.original_first_move
            && checked_rebase_offset(&source.snapshot, &selected.snapshot, &original_line)?
                == Some(usize::from(metadata.source_line_offset)),
        "original first move or exact rebased source endpoint",
    )
}

#[cfg(test)]
mod tests;
