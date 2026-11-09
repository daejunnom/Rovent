//! Historical /3 CPU and system-CUDA record verification. These checks join
//! original metadata and replay Rules; they do not create live authority,
//! prove binary custody, or turn an unknown physical completion into a fence.
use super::{check, hex, invalid};
use crate::ArenaError;
use rz_experiments::rovent_corpus::RoventNativeLifecycle;
use rz_experiments::rovent_corpus_lock_v3::{
    ROVENT_LOCK_V3_NATIVE_RAW_DOMAIN,
    cpu_native::ROVENT_LOCK_V3_CPU_NATIVE_RAW_DOMAIN,
    record::{RoventCorpusManifestLockV3, RoventTrainingRecordLockV3},
    validate_native_lifecycle_lock_v3,
};
use rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2;
use rz_experiments::rovent_submitted_input::RoventSubmittedInputV1;
use rz_experiments::{NativeRuntimeSourceV6, RoventInputLockV3, RoventRootDescriptorV3};
use rz_position::BoardMove;

/// Original identities validated as metadata; the caller still verifies the
/// original file pins, publication, and any live completion evidence.
pub struct VerifiedHistoricalNativeRootMetadataV3 {
    pub input_lock_sha256: String,
    pub root_sha256: String,
}

/// Verify a complete /3 record without converting it into a /2 record.
pub fn verify_historical_lock_v3_record_evidence(
    record: &RoventTrainingRecordLockV3,
    manifest: &RoventCorpusManifestLockV3,
    lock: &RoventInputLockV3,
) -> Result<VerifiedHistoricalNativeRootMetadataV3, ArenaError> {
    record.validate(manifest, lock)?;
    let row = &record.0;
    let scope = verify_historical_lock_v3_input(
        &row.input,
        &row.input_sha256,
        &row.native,
        &manifest.registration,
        lock,
    )?;
    if let Some(target) = &row.targets.wdl {
        let raw = &row
            .head_sources
            .get("wdl")
            .ok_or_else(|| invalid("lock-v3 WDL original head missing"))?
            .raw_receipt;
        super::rules_replay::verify_rules_wdl_replay(
            raw,
            &row.input.game_id,
            &row.input.rules_anchor,
            target,
        )?;
    }
    super::divergence_owner::verify_historical_pc_divergence_lock_v3(
        row,
        &manifest.registration,
        lock,
    )?;
    Ok(scope)
}

/// Validate either an original row or a separately stored future visible
/// observation, retaining its original runtime-specific native domain.
pub(super) fn verify_historical_lock_v3_input(
    input: &RoventSubmittedInputV1,
    input_sha256: &str,
    native: &RoventNativeLifecycle,
    manifest: &RoventCorpusManifestV2,
    lock: &RoventInputLockV3,
) -> Result<VerifiedHistoricalNativeRootMetadataV3, ArenaError> {
    validate_native_lifecycle_lock_v3(native, manifest, lock)?;
    let raw = &native.raw_retirement["root_admission"];
    let root_sha = native.raw_retirement["root_admission_sha256"]
        .as_str()
        .ok_or_else(|| invalid("lock-v3 root digest missing"))?;
    // The lifecycle validator above parses the whole original CPU/CUDA root
    // with deny_unknown_fields, validates its digest and joins it to the lock.
    let descriptor: RoventRootDescriptorV3 = serde_json::from_value(raw["root_descriptor"].clone())
        .map_err(|error| invalid(format!("lock-v3 root descriptor: {error}")))?;
    let revision = raw["root_state"]["revision"]
        .as_u64()
        .ok_or_else(|| invalid("lock-v3 root revision missing"))?;
    let replay: super::rules_replay::RoventSubmittedRulesReplay =
        serde_json::from_value(native.raw_submission["rules_replay"].clone())
            .map_err(|error| invalid(format!("lock-v3 Rules replay: {error}")))?;
    let common = input.common_validation_projection();
    let indices = input
        .attack_mapping
        .iter()
        .map(|slot| usize::from(slot.attack_ply))
        .collect::<Vec<_>>();
    super::rules_replay::verify_historical_rules_fields(&common, native, &replay, &indices)?;
    let raw_domain = match &lock.manifest.system_runtime_lock.runtime {
        NativeRuntimeSourceV6::Cpu => ROVENT_LOCK_V3_CPU_NATIVE_RAW_DOMAIN,
        NativeRuntimeSourceV6::SystemCuda(_) => ROVENT_LOCK_V3_NATIVE_RAW_DOMAIN,
    };
    super::submitted_review_historical::verify_historical_submitted_review_fields_for_domain(
        input,
        input_sha256,
        native,
        raw_domain,
    )?;
    let actual = replay.actual_root.restore()?;
    let root_moves = descriptor
        .input
        .allowed_root_moves
        .iter()
        .map(|movement| BoardMove::from_uci(movement).map_err(|error| invalid(error.to_string())))
        .collect::<Result<Vec<_>, _>>()?;
    let legal = actual.legal_moves();
    check(
        root_moves.iter().enumerate().all(|(slot, movement)| {
            legal.contains(movement) && !root_moves[..slot].contains(movement)
        }) && hex(rz_position::contracts::rules_snapshot_semantic_digest(
            &actual.snapshot(),
        )) == descriptor.input.state_and_history_sha256
            && actual.revision() == revision
            && actual
                .ordered_legal_moves()
                .moves()
                .iter()
                .copied()
                .filter(|movement| root_moves.contains(movement))
                .collect::<Vec<_>>()
                == root_moves,
        "lock-v3 historical Rules root and ordered restriction mismatch",
    )?;
    let observed = &raw["native_origin"]["startup_observation"];
    let weights_sha = observed["weights_sha256"]
        .as_str()
        .ok_or_else(|| invalid("lock-v3 original weights identity missing"))?;
    let backend_sha = observed["backend_sha256"]
        .as_str()
        .ok_or_else(|| invalid("lock-v3 original backend identity missing"))?;
    super::historical_scope::verify_original_native_key_for_sources(
        &common,
        native,
        &replay,
        weights_sha,
        backend_sha,
    )?;
    Ok(VerifiedHistoricalNativeRootMetadataV3 {
        input_lock_sha256: lock.canonical_sha256.clone(),
        root_sha256: root_sha.to_owned(),
    })
}
