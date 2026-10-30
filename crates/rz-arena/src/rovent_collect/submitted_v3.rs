//! 새 raw/3 제출 adapter다. 원 query/tensor와 공격 ply를 보존하며,
//! legacy record-index 입력이나 미래 counter를 제조하지 않는다.
use super::*;
use rz_experiments::rovent_corpus_v2::RoventTrainingRecordV2;
use rz_experiments::rovent_r_readout::RoventRSelectionPolicy;
use rz_experiments::rovent_submitted_input::{
    ROVENT_SUBMITTED_INPUT_DOMAIN, RoventAttackMapping, RoventSubmittedInputV1,
};
use serde::{Deserialize, Serialize};

pub const ROVENT_SUBMITTED_REVIEW_EVIDENCE: &str = "rz-rovent-submitted-review-evidence/1";

/// 공개 wire를 deserialize하여 원 producer custody를 만들 수 없다.
pub struct PublishedRoventRecordV2 {
    pub(super) record: RoventTrainingRecordV2,
    pub(super) manifest_sha256: String,
    pub(super) admission: RoventCollectionRootAdmission,
}
impl PublishedRoventRecordV2 {
    pub fn record(&self) -> &RoventTrainingRecordV2 {
        &self.record
    }
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
    pub fn admission(&self) -> &RoventCollectionRootAdmission {
        &self.admission
    }
}
pub struct RoventRetiredRecordV2 {
    pub(super) record: RoventTrainingRecordV2,
    pub(super) manifest_sha256: String,
    pub(super) ticket: RoleTicketEnvelope,
    pub(super) admission: RoventCollectionRootAdmission,
}
impl RoventRetiredRecordV2 {
    pub fn input(&self) -> &RoventSubmittedInputV1 {
        &self.record.input
    }
    pub fn ticket(&self) -> &RoleTicketEnvelope {
        &self.ticket
    }
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
    pub fn admission(&self) -> &RoventCollectionRootAdmission {
        &self.admission
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventSubmittedRecordMetadata {
    pub local_index: u16,
    pub record_revision: u64,
    pub parent_revision: Option<u64>,
    pub supersedes_revision: Option<u64>,
    pub origin_state: u64,
    pub source_state: u64,
    pub source_revision: u64,
    pub source_line_offset: u16,
    pub original_first_move: Option<u16>,
    pub value: Option<i32>,
    pub completed_depth: u16,
    pub score_scope: Option<String>,
    pub critical: bool,
}

/// selected tensor를 중복하지 않는 원 ReviewView의 historical 보충 근거다.
/// 공개 DTO를 읽어 live view/pin/native/수락 권한을 복원할 수 없다.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventSubmittedReviewEvidence {
    pub schema: String,
    pub sequence: u64,
    pub generation: u64,
    pub root_state: u64,
    pub policy: RoventRSelectionPolicy,
    pub input_sha256: String,
    pub view_seal_sha256: String,
    pub record_scope_sha256: String,
    pub anchor_states: Vec<u64>,
    pub source_anchors: Vec<RoventRulesPositionReplay>,
    pub record_metadata: Vec<RoventSubmittedRecordMetadata>,
}

/// native owner가 실제 prepare한 입력/원 view를 빌려 호출하는 순수 capture다.
/// 이 결과만으로 submitted/physical/retirement authority를 발급하지 않는다.
pub fn capture_rovent_submitted_input(
    ticket: &RoleTicketEnvelope,
    view: &ReviewView,
    input: &PalsVirtualReviewInput,
    context: &GenerationCaptureContextV3,
) -> Result<(RoventSubmittedInputV1, RoventSubmittedReviewEvidence), ArenaError> {
    let common = crate::pals_collect_v3::capture_submitted_review_common_fields(
        ticket, view, input, context,
    )?;
    let query = view.query();
    check(
        query.divergences.len() == input.divergences.len(),
        "actual attack head slot count",
    )?;
    // 수열이 같은 record가 여럿이면 source를 임의 선택하지 않는다. 입력의
    // 공격 slot 자체는 유지하며 미래 owner witness가 별도 target을 제공한다.
    let visible = query.counterexample.as_ref().and_then(|counter| {
        let mut matching = view
            .records()
            .iter()
            .zip(view.records_metadata())
            .enumerate()
            .filter(|(_, (record, metadata))| {
                record.kind == rz_search::pals::engine::RecordKind::Counterexample
                    && record.line == *counter
                    && metadata.anchor == input.query.anchors[2]
            });
        let one = matching.next()?;
        if matching.next().is_some() {
            None
        } else {
            Some((one.0, one.1.1.source_revision))
        }
    });
    let attack_mapping = query
        .divergences
        .iter()
        .enumerate()
        .map(|(slot, ply)| {
            Ok(RoventAttackMapping {
                slot: u16::try_from(slot).map_err(|_| invalid("attack slot overflow"))?,
                attack_ply: u16::try_from(*ply).map_err(|_| invalid("attack ply overflow"))?,
                selected_counter_local_index: visible
                    .map(|(index, _)| {
                        u16::try_from(index)
                            .map_err(|_| invalid("visible counter local index overflow"))
                    })
                    .transpose()?,
                selected_counter_revision: visible.map(|(_, revision)| revision),
            })
        })
        .collect::<Result<Vec<_>, ArenaError>>()?;
    let captured = RoventSubmittedInputV1 {
        schema: ROVENT_SUBMITTED_INPUT_DOMAIN.into(),
        game_id: common.game_id,
        opening_id: common.opening_id,
        line_genealogy_id: common.line_genealogy_id,
        transposition_sha256: common.transposition_sha256,
        role: common.role,
        model_epoch: common.model_epoch,
        context_epoch: common.context_epoch,
        capture_sequence: common.capture_sequence,
        question_id: common.question_id,
        worker_id: common.worker_id,
        job_id: common.job_id,
        slice_id: common.slice_id,
        rules_anchor: common.rules_anchor,
        actual_history: common.actual_history,
        selected_view: common.selected_view,
        view_sha256: common.view_sha256,
        legal_moves: common.legal_moves,
        action_support_mask: common.action_support_mask,
        encoded_model_input: common.encoded_model_input,
        attack_mapping,
    };
    captured.validate()?;
    let metadata = view
        .records()
        .iter()
        .zip(view.records_metadata())
        .enumerate()
        .map(|(index, (record, source))| {
            Ok(RoventSubmittedRecordMetadata {
                local_index: u16::try_from(index)
                    .map_err(|_| invalid("selected metadata index overflow"))?,
                record_revision: record.revision,
                parent_revision: record.parent_revision,
                supersedes_revision: record.supersedes_revision,
                origin_state: record.origin_state.0 as u64,
                source_state: source.source_state.0 as u64,
                source_revision: source.source_revision,
                source_line_offset: source.source_line_offset,
                original_first_move: source
                    .original_first_move
                    .map(|m| packed(&[m]).map(|v| v[0]))
                    .transpose()?,
                value: record.value,
                completed_depth: record.completed_depth,
                score_scope: record.score_scope.map(|scope| {
                    match scope {
                        rz_search::cpu::CpuScoreScope::FrontierOnly => "FrontierOnly",
                        rz_search::cpu::CpuScoreScope::CompletedIteration => "CompletedIteration",
                        rz_search::cpu::CpuScoreScope::RulesTerminal => "RulesTerminal",
                    }
                    .into()
                }),
                critical: record.critical,
            })
        })
        .collect::<Result<Vec<_>, ArenaError>>()?;
    let policy = view.selection_policy();
    let source_anchors = view
        .source_snapshots()
        .iter()
        .map(|snapshot| {
            let position = Position::from_snapshot(
                snapshot,
                rz_position::PositionLimits {
                    max_history_positions: 16_385,
                    ..Default::default()
                },
            )
            .map_err(|error| invalid(error.to_string()))?;
            RoventRulesPositionReplay::capture(&position)
        })
        .collect::<Result<Vec<_>, ArenaError>>()?;
    check(
        source_anchors.len() == metadata.len(),
        "original selected source snapshot coverage",
    )?;
    let review = RoventSubmittedReviewEvidence {
        schema: ROVENT_SUBMITTED_REVIEW_EVIDENCE.into(),
        sequence: view.sequence(),
        generation: view.generation(),
        root_state: view.root_state().0 as u64,
        policy: RoventRSelectionPolicy {
            mode: match policy.mode {
                rz_search::pals::review_context::ReviewAnchorMode::CurrentRootOnly => {
                    "current_root_only"
                }
                rz_search::pals::review_context::ReviewAnchorMode::AnchoredCompact => {
                    "anchored_compact"
                }
            }
            .into(),
            token_budget: policy.token_budget,
            max_records: policy.max_records,
            max_anchors: policy.max_anchors,
            required_bundle: policy.required_bundle,
        },
        input_sha256: view.digest().iter().map(|v| format!("{v:02x}")).collect(),
        view_seal_sha256: view
            .seal_digest()
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect(),
        record_scope_sha256: view
            .record_scope_digest()
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect(),
        anchor_states: view
            .anchors()
            .iter()
            .map(|anchor| anchor.state.0 as u64)
            .collect(),
        source_anchors,
        record_metadata: metadata,
    };
    Ok((captured, review))
}
