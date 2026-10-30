//! Rovent 원 제출 입력의 명시 형식이다. legacy GenerationInputV3의
//! divergence record-index 의미를 공격 ply로 재해석하지 않는다.
use crate::ManifestError;
use crate::pals_data_v3::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const ROVENT_SUBMITTED_INPUT_DOMAIN: &str = "rz-rovent-submitted-input/1";
pub const ROVENT_NATIVE_RAW_V3_DOMAIN: &str = "rz-rovent-raw-record/3";
pub const ROVENT_TRAINING_RECORD_V2_DOMAIN: &str = "rz-rovent-training-record/2";
pub const ROVENT_CORPUS_V2_DOMAIN: &str = "rz-rovent-corpus-manifest/2";
pub const ROVENT_INDEX_V2_DOMAIN: &str = "rz-rovent-corpus-index/2";
pub const ROVENT_PUBLICATION_V2_DOMAIN: &str = "rz-rovent-corpus-publication/2";

/// 원 D head slot의 공격 위치와 제출 당시 식별된 counter를 구분한다.
/// counter를 나중에 발견해도 이 불변 입력의 None을 덮어쓰지 않는다.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventAttackMapping {
    pub slot: u16,
    /// 원 query.proposal의 0-based root-relative ply다.
    pub attack_ply: u16,
    pub selected_counter_local_index: Option<u16>,
    pub selected_counter_revision: Option<u64>,
}

/// 원 수치 payload와 query를 그대로 보존하는 독립 입력 DTO다.
/// 이 공개 형식의 검증/해시는 native 실행·수락·학습 권한이 아니다.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventSubmittedInputV1 {
    pub schema: String,
    pub game_id: String,
    pub opening_id: String,
    pub line_genealogy_id: String,
    pub transposition_sha256: String,
    pub role: GenerationRoleV3,
    pub model_epoch: u64,
    pub context_epoch: u64,
    pub capture_sequence: u64,
    pub question_id: String,
    pub worker_id: String,
    pub job_id: String,
    pub slice_id: String,
    pub rules_anchor: GenerationRulesAnchorV3,
    pub actual_history: Vec<u16>,
    pub selected_view: GenerationSelectedViewV3,
    pub view_sha256: String,
    pub legal_moves: Vec<u16>,
    pub action_support_mask: [bool; 7],
    pub encoded_model_input: serde_json::Value,
    pub attack_mapping: Vec<RoventAttackMapping>,
}

impl RoventSubmittedInputV1 {
    /// 공통 Rules/정책/WDL 검증 계산에만 쓰는 임시 projection이다. 이 값은
    /// legacy V3 입력이나 기록으로 발행하지 않으며 divergence 검증을 대신하지
    /// 않는다. 실제 query/tensor를 수정하거나 빈 D tensor를 만들지 않는다.
    pub fn common_validation_projection(&self) -> GenerationInputV3 {
        GenerationInputV3 {
            game_id: self.game_id.clone(),
            opening_id: self.opening_id.clone(),
            line_genealogy_id: self.line_genealogy_id.clone(),
            transposition_sha256: self.transposition_sha256.clone(),
            role: self.role,
            model_epoch: self.model_epoch,
            context_epoch: self.context_epoch,
            capture_sequence: self.capture_sequence,
            question_id: self.question_id.clone(),
            worker_id: self.worker_id.clone(),
            job_id: self.job_id.clone(),
            slice_id: self.slice_id.clone(),
            rules_anchor: self.rules_anchor.clone(),
            actual_history: self.actual_history.clone(),
            selected_view: self.selected_view.clone(),
            view_sha256: self.view_sha256.clone(),
            legal_moves: self.legal_moves.clone(),
            action_support_mask: self.action_support_mask,
            encoded_model_input: self.encoded_model_input.clone(),
            // 공통 계산은 이 legacy 전용 필드를 읽지 않는다. 새 D 후보/순서는
            // 원 tensor와 attack_mapping으로 전부 보존하며 아래에서 검증한다.
            divergence_record_ids: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        ensure(
            self.schema == ROVENT_SUBMITTED_INPUT_DOMAIN,
            "explicit Rovent submitted-input domain",
        )?;
        let common = self.common_validation_projection();
        common.validate_common_fields()?;
        ensure(
            self.role != GenerationRoleV3::Validator && self.action_support_mask == [false; 7],
            "four Rovent profiles are V-free; legacy V inputs are separate",
        )?;
        let rows = self.encoded_model_input["divergences"]
            .as_array()
            .ok_or_else(|| ManifestError::Integrity("original divergence rows required".into()))?;
        ensure(
            rows.len() == self.attack_mapping.len() && rows.len() <= 128,
            "one original attack mapping per native D slot",
        )?;
        let mut seen = BTreeSet::new();
        for (slot, mapping) in self.attack_mapping.iter().enumerate() {
            let ply = usize::from(mapping.attack_ply);
            ensure(
                usize::from(mapping.slot) == slot
                    && seen.insert(mapping.attack_ply)
                    && ply > 0
                    && ply < self.selected_view.query.proposal.len()
                    && ply < 256,
                "original attack slot order/unique proposal ply",
            )?;
            let feature = rows[slot]
                .as_array()
                .and_then(|row| row.first())
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    ManifestError::Integrity("native attack row feature required".into())
                })?;
            ensure(
                (feature as f32).to_bits() == ((mapping.attack_ply as f32) / 256.0).to_bits(),
                "native attack feature does not join the original ply",
            )?;
            match (
                mapping.selected_counter_local_index,
                mapping.selected_counter_revision,
            ) {
                (None, None) => {}
                (Some(index), Some(revision)) => {
                    let record = self
                        .selected_view
                        .records
                        .get(usize::from(index))
                        .ok_or_else(|| {
                            ManifestError::Integrity("submitted counter local index".into())
                        })?;
                    ensure(
                        revision > 0
                            && record.kind == GenerationRecordKindV3::Counter
                            && self.selected_view.query.counter_present
                            && record.moves == self.selected_view.query.counter,
                        "submitted counter must already exist in the exact selected view",
                    )?;
                    // 원 revision은 native raw/3의 Rules replay.metadata와 별도
                    // 대조한다. record_id 문자열을 source authority로 취급하지 않는다.
                }
                _ => {
                    return Err(ManifestError::Integrity(
                        "partial submitted counter mapping".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, ManifestError> {
        self.validate()?;
        let mut raw =
            serde_json::to_value(self).map_err(|e| ManifestError::Integrity(e.to_string()))?;
        project_generation_input_float_bits(&mut raw)?;
        generation_digest(ROVENT_SUBMITTED_INPUT_DOMAIN, &raw)
    }

    pub fn policy_probabilities(
        &self,
        target: &GenerationPolicyTargetV3,
    ) -> Result<Vec<f64>, ManifestError> {
        self.validate()?;
        target.probabilities_for_source(
            &self.common_validation_projection(),
            "rovent_completed_teacher_multipv",
        )
    }
    pub fn wdl_probabilities(
        &self,
        target: &GenerationWdlTargetV3,
    ) -> Result<[f64; 3], ManifestError> {
        self.validate()?;
        target.probabilities(&self.common_validation_projection())
    }
}
