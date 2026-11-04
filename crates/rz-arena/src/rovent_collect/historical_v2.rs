//! 새 closed family의 역사 데이터 검증이다. 원 raw/3를 /2로 재태그하거나
//! 새 native/Store/worker 권한을 발급하지 않고 기존 Rules·입력 계산을 공유한다.
use super::*;
use rz_experiments::rovent_corpus_v2::{RoventCorpusManifestV2, RoventTrainingRecordV2};

pub fn verify_historical_submitted_record_evidence(
    record: &RoventTrainingRecordV2,
    manifest: &RoventCorpusManifestV2,
) -> Result<RoventHistoricalRecordEvidence, ArenaError> {
    record.validate(manifest)?;
    let common = common_scope_projection(record);
    let replay: RoventSubmittedRulesReplay =
        serde_json::from_value(record.native.raw_submission["rules_replay"].clone())
            .map_err(|_| invalid("raw3 original Rules replay missing/malformed"))?;
    let attacks = record
        .input
        .attack_mapping
        .iter()
        .map(|slot| usize::from(slot.attack_ply))
        .collect::<Vec<_>>();
    rules_replay::verify_historical_rules_fields(&common.input, &record.native, &replay, &attacks)?;
    if let Some(target) = &record.targets.wdl {
        let raw = &record
            .head_sources
            .get("wdl")
            .ok_or_else(|| invalid("raw3 WDL original head missing"))?
            .raw_receipt;
        rules_replay::verify_rules_wdl_replay(
            raw,
            &record.input.game_id,
            &record.input.rules_anchor,
            target,
        )?;
    }
    super::submitted_review_historical::verify_historical_submitted_review(record)?;
    let scope = historical_scope::verify_historical_submitted_native_fields(
        &common.input,
        &record.native,
        &manifest.common_registration_projection(),
    )?;
    divergence_owner::verify_historical_pc_divergence(record, manifest)?;
    Ok(scope)
}

/// 이 값은 공유 계산의 임시 필드 projection뿐이다. schema와 원 raw/3를 그대로
/// 보존하며 legacy validate/publish/reader에 넘기거나 별도 artifact로 쓰지 않는다.
pub(super) fn common_scope_projection(record: &RoventTrainingRecordV2) -> RoventTrainingRecord {
    RoventTrainingRecord {
        schema: record.schema.clone(),
        role: record.role,
        split: record.split,
        input: record.input.common_validation_projection(),
        input_sha256: record.input_sha256.clone(),
        origin_sha256: record.origin_sha256.clone(),
        record_sha256: record.record_sha256.clone(),
        lineage: record.lineage.clone(),
        native: record.native.clone(),
        head_sources: record.head_sources.clone(),
        targets: GenerationTargetsV3 {
            policy: record.targets.policy.clone(),
            wdl: record.targets.wdl.clone(),
            divergence: None,
            task: None,
        },
        divergence_index: None,
        nnuec_input: None,
        nnuec_input_sha256: None,
        nnuec_model_sha256: None,
        nnuec_semantics: None,
        nnuec_wdl: None,
    }
}
