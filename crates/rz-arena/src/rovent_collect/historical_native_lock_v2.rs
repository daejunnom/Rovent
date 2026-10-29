//! Strict metadata consumer for the separate /2 input-lock origin/root domain.
//! This is not a corpus, Rules replay, execution, liveness or completion gate.
use super::{check, generation_digest, generation_sha, hex, invalid};
use crate::ArenaError;
use rz_contracts::Digest;
use rz_experiments::rovent_corpus::RoventNativeLifecycle;
use rz_experiments::rovent_corpus_lock_v2::{
    ROVENT_LOCK_V2_NATIVE_RAW_DOMAIN,
    record::{RoventCorpusManifestLockV2, RoventTrainingRecordLockV2},
};
use rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2;
use rz_experiments::rovent_submitted_input::RoventSubmittedInputV1;
use rz_experiments::{
    RoventExternalProfilePin, RoventInputLockV2, RoventRootAllocationClockV2,
    RoventRootDescriptorV2,
};
use rz_position::BoardMove;
use rz_uci::pals_adaptive::{
    ROVENT_COLLECTION_NATIVE_ORIGIN_LOCK_V2, ROVENT_COLLECTION_ROOT_ADMISSION_LOCK_V2,
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    purpose: String,
    mode: String,
    source_commit: String,
    source_components: BTreeMap<String, String>,
    source_sha256_domain: String,
    source_sha256: String,
    model_semantics: String,
    model_semantics_sha256: String,
    encoding: String,
    encoding_sha256: String,
    encoding_registry_sha256: String,
    model_implementation_sha256: String,
    teacher_profiles: Vec<RoventExternalProfilePin>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StartupObservation {
    domain: String,
    process_epoch: u64,
    mode: String,
    topology: String,
    weights_sha256: String,
    export_manifest_sha256: String,
    backend_sha256: String,
    runtime_library_sha256: String,
    runtime_bundle_codec_sha256: Option<String>,
    native_configuration_sha256: String,
    startup_configuration_sha256: String,
    search_configuration_sha256: String,
    validator_loaded: bool,
    physical_workers_created: usize,
    neural_service_executed: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Namespace {
    value_semantics: String,
    value_model: String,
    value_encoding: String,
    value_precision: String,
    representation_model_domain: String,
    representation_model_sha256: String,
    representation_encoding_sha256: String,
    representation_precision: String,
    frozen_epoch: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeOrigin {
    schema: String,
    input_lock_domain: String,
    input_lock_sha256: String,
    declaration: Declaration,
    startup_observation: StartupObservation,
    registered_namespace: Namespace,
    process_boundary_observation: String,
    neural_completion: String,
    physical_completion: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RootState {
    owner: u64,
    revision: u64,
    semantic: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Root {
    schema: String,
    native_origin: NativeOrigin,
    native_origin_sha256: String,
    process_epoch: u64,
    game_generation: u64,
    root_generation: u64,
    implementation_sha256: String,
    deadline_clock: u64,
    deadline_tick: u64,
    root_budget_sha256: String,
    selective_policy_sha256: String,
    root_state: RootState,
    store_root_state: usize,
    root_moves: Vec<String>,
    depth_plies: Option<u16>,
    mate_moves: Option<u16>,
    requested_nodes: Option<u64>,
    root_descriptor: RoventRootDescriptorV2,
    root_descriptor_sha256: String,
    allocation_clock: RoventRootAllocationClockV2,
    allocation_clock_sha256: String,
}

/// Confirmed declarations and internal metadata joins only. Constructing this
/// does not prove original file bytes, Rules history, a worker's actual run,
/// acceptance, teacher results, physical completion or corpus eligibility.
pub struct VerifiedHistoricalNativeRootMetadataV2 {
    pub input_lock_sha256: String,
    pub native_origin_sha256: String,
    pub root_sha256: String,
    pub root_descriptor_sha256: String,
    pub allocation_clock_sha256: String,
}

/// Read only the exact /2 format. The caller must separately hold and verify
/// the original input-file pins and any actual completion/Rules evidence.
pub fn verify_historical_native_root_metadata_v2(
    input_lock_json: &str,
    root_json: &str,
    expected_root_sha256: &str,
) -> Result<VerifiedHistoricalNativeRootMetadataV2, ArenaError> {
    let lock = RoventInputLockV2::from_json(input_lock_json)?;
    let raw: Value = rz_experiments::decode_json(root_json)?;
    let root: Root = serde_json::from_value(raw.clone())
        .map_err(|error| invalid(format!("V2 root metadata: {error}")))?;
    let native_raw = raw
        .get("native_origin")
        .ok_or_else(|| invalid("V2 native origin missing"))?;
    let root_sha = generation_digest(ROVENT_COLLECTION_ROOT_ADMISSION_LOCK_V2, &raw)?;
    let native_sha = generation_digest(ROVENT_COLLECTION_NATIVE_ORIGIN_LOCK_V2, native_raw)?;
    let origin = &root.native_origin;
    let declared = &origin.declaration;
    let observed = &origin.startup_observation;
    let namespace = &origin.registered_namespace;
    let desc = &root.root_descriptor.input;
    let model = &lock.manifest.engine.pc.proposer;
    let expected_mode = match lock.manifest.engine.mode {
        rz_contracts::rovent::ExecutionMode::HybridPcr => "hybrid_pcr",
        rz_contracts::rovent::ExecutionMode::HybridPc => "hybrid_pc",
        rz_contracts::rovent::ExecutionMode::CpuPcr => "cpu_pcr",
        rz_contracts::rovent::ExecutionMode::CpuPc => "cpu_pc",
    };
    // Reuse the bounded /2 parsers as well as their joins. Deserializing a
    // nested descriptor directly would skip their individual size checks.
    let descriptor_json =
        serde_json::to_string(&root.root_descriptor).map_err(|error| invalid(error.to_string()))?;
    let allocation_json = serde_json::to_string(&root.allocation_clock)
        .map_err(|error| invalid(error.to_string()))?;
    let _declarations = super::historical_declarations_v2::verify_historical_declarations_v2(
        input_lock_json,
        &descriptor_json,
        &allocation_json,
    )?;
    let descriptor_sha = hex(root.root_descriptor.digest()?);
    let allocation_sha = hex(root.allocation_clock.digest()?);
    let source_sha = rz_experiments::rovent_source_aggregate_sha256(
        &lock.manifest.engine.source_commit,
        &lock.manifest.engine.source_components,
    )?;
    let encoding_sha = hex(Digest({
        use sha2::Digest as _;
        sha2::Sha256::digest(rz_eval::pals_model::virtual_review::ENCODING.as_bytes()).into()
    }));
    let semantics_sha = hex(Digest(
        rz_eval::pals_model::PalsModelProfile::VirtualReviewV1.registered_semantics_sha256(),
    ));
    let registry_sha = hex(Digest(
        rz_eval::pals_model::virtual_review::registered_encoding_sha256(),
    ));
    check(
        generation_sha(expected_root_sha256)
            && root_sha == expected_root_sha256
            && root.schema == ROVENT_COLLECTION_ROOT_ADMISSION_LOCK_V2
            && origin.schema == ROVENT_COLLECTION_NATIVE_ORIGIN_LOCK_V2
            && root.native_origin_sha256 == native_sha
            && origin.input_lock_domain == rz_experiments::ROVENT_LOCK_V2_DOMAIN
            && origin.input_lock_sha256 == lock.canonical_sha256
            && declared.purpose == "collection"
            && lock.manifest.purpose == rz_experiments::RoventPurpose::Collection
            && declared.mode == expected_mode
            && observed.mode == expected_mode
            && declared.source_commit == lock.manifest.engine.source_commit
            && declared.source_components == lock.manifest.engine.source_components
            && declared.source_sha256_domain == rz_experiments::ROVENT_SOURCE_AGGREGATE_DOMAIN
            && declared.source_sha256 == source_sha
            && declared.model_semantics == model.semantics
            && declared.model_semantics_sha256 == semantics_sha
            && declared.encoding == model.encoding
            && declared.encoding_sha256 == encoding_sha
            && declared.encoding_registry_sha256 == registry_sha
            && declared.model_implementation_sha256 == model.implementation_sha256
            && declared.teacher_profiles == lock.manifest.engine.teachers
            && observed.domain == rz_uci::rovent_bootstrap::ROVENT_NATIVE_STARTUP_DOMAIN
            && observed.process_epoch == root.process_epoch
            && observed.process_epoch != 0
            && observed.weights_sha256 == model.weights.sha256
            && observed.startup_configuration_sha256 == lock.manifest.startup_configuration_sha256
            && observed.search_configuration_sha256
                == lock.manifest.startup_search_configuration_sha256
            && !observed.validator_loaded
            && observed.physical_workers_created == 1
            && !observed.neural_service_executed
            && origin.process_boundary_observation == "unknown"
            && origin.neural_completion == "unknown"
            && origin.physical_completion == "unknown"
            && namespace.representation_model_domain == "loaded_checkpoint_sha256"
            && namespace.representation_model_sha256 == observed.weights_sha256
            && namespace.representation_encoding_sha256 == encoding_sha
            && namespace.representation_precision == "fp32"
            && namespace.value_precision == "fp32_tf32_off"
            // Match the actual Ready snapshot's model_value_identity and
            // model_namespace, rather than accepting arbitrary labels as
            // evidence of the registered model.  These are declarations;
            // they still do not prove a live worker or physical completion.
            && namespace.frozen_epoch == 1
            && namespace.value_semantics == rz_search::pals::value::MODEL_WDL_VALUE_SEMANTICS
            && namespace.value_model == format!(
                "pals-native-virtual-v3:{}:{}:{}",
                observed.export_manifest_sha256, observed.weights_sha256, observed.backend_sha256,
            )
            && namespace.value_encoding == format!(
                "{}:{}",
                rz_eval::pals_model::virtual_review::ENCODING, registry_sha,
            )
            && root.root_descriptor_sha256 == descriptor_sha
            && root.allocation_clock_sha256 == allocation_sha
            && desc.authority.epoch == root.process_epoch
            && desc.authority.game == root.game_generation
            && desc.authority.root == root.root_generation
            && desc.authority.implementation_sha256 == root.implementation_sha256
            && desc.clock.hard.clock_epoch == root.deadline_clock
            && desc.clock.hard.at_ns == root.deadline_tick
            && desc.state_and_history_sha256 == root.root_state.semantic
            && desc.allowed_root_moves == root.root_moves
            && desc.original_go.depth_plies.map(|value| value.get()) == root.depth_plies
            && desc.original_go.mate_moves.map(|value| value.get()) == root.mate_moves
            && desc.original_go.requested_nodes == root.requested_nodes
            && root.root_descriptor.root_budget_sha256 == root.root_budget_sha256
            && generation_sha(&root.selective_policy_sha256)
            && [
                &observed.export_manifest_sha256,
                &observed.backend_sha256,
                &observed.runtime_library_sha256,
                &observed.native_configuration_sha256,
            ]
            .into_iter()
            .all(|value| generation_sha(value))
            && observed
                .runtime_bundle_codec_sha256
                .as_ref()
                .is_none_or(|value| generation_sha(value))
            && matches!(
                observed.topology.as_str(),
                "roles_only" | "own_researcher" | "external_researcher"
            ),
        "V2 historical origin/root declaration or digest mismatch",
    )?;
    // These process-local identifiers are metadata only; Rules replay and
    // actual Store identity cannot be reconstructed from them here.
    let _ = (
        root.root_state.owner,
        root.root_state.revision,
        root.store_root_state,
    );
    Ok(VerifiedHistoricalNativeRootMetadataV2 {
        input_lock_sha256: lock.canonical_sha256,
        native_origin_sha256: native_sha,
        root_sha256: root_sha,
        root_descriptor_sha256: descriptor_sha,
        allocation_clock_sha256: allocation_sha,
    })
}

/// Verify a /2-lock record's persisted metadata and Rules replay. This does
/// not read publication files, prove the original file custody or grant live
/// worker, teacher, root or physical-completion authority.
pub fn verify_historical_lock_v2_record_evidence(
    record: &RoventTrainingRecordLockV2,
    manifest: &RoventCorpusManifestLockV2,
    lock: &RoventInputLockV2,
) -> Result<VerifiedHistoricalNativeRootMetadataV2, ArenaError> {
    record.validate(manifest, lock)?;
    let row = &record.0;
    let scope = verify_historical_lock_v2_input(
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
            .ok_or_else(|| invalid("lock-v2 WDL original head missing"))?
            .raw_receipt;
        super::rules_replay::verify_rules_wdl_replay(
            raw,
            &row.input.game_id,
            &row.input.rules_anchor,
            target,
        )?;
    }
    super::divergence_owner::verify_historical_pc_divergence_lock_v2(
        row,
        &manifest.registration,
        lock,
    )?;
    Ok(scope)
}

/// Common /2-lock submitted input verification for a record or a separately
/// stored future visible observation. Original domains and raw values are
/// never retagged, and completion metadata is not a live physical fence.
pub(super) fn verify_historical_lock_v2_input(
    input: &RoventSubmittedInputV1,
    input_sha256: &str,
    native: &RoventNativeLifecycle,
    manifest: &RoventCorpusManifestV2,
    lock: &RoventInputLockV2,
) -> Result<VerifiedHistoricalNativeRootMetadataV2, ArenaError> {
    rz_experiments::rovent_corpus_lock_v2::validate_native_lifecycle_lock_v2(
        native, manifest, lock,
    )?;
    let raw = &native.raw_retirement["root_admission"];
    let root: Root = serde_json::from_value(raw.clone())
        .map_err(|error| invalid(format!("V2 root metadata: {error}")))?;
    let root_sha = native.raw_retirement["root_admission_sha256"]
        .as_str()
        .ok_or_else(|| invalid("lock-v2 root digest missing"))?;
    let lock_json = serde_json::to_string(lock).map_err(|error| invalid(error.to_string()))?;
    let root_json = serde_json::to_string(raw).map_err(|error| invalid(error.to_string()))?;
    let scope = verify_historical_native_root_metadata_v2(&lock_json, &root_json, root_sha)?;
    let replay: super::rules_replay::RoventSubmittedRulesReplay =
        serde_json::from_value(native.raw_submission["rules_replay"].clone())
            .map_err(|error| invalid(format!("lock-v2 Rules replay: {error}")))?;
    let common = input.common_validation_projection();
    let indices = input
        .attack_mapping
        .iter()
        .map(|slot| usize::from(slot.attack_ply))
        .collect::<Vec<_>>();
    super::rules_replay::verify_historical_rules_fields(&common, native, &replay, &indices)?;
    super::submitted_review_historical::verify_historical_submitted_review_fields_for_domain(
        input,
        input_sha256,
        native,
        ROVENT_LOCK_V2_NATIVE_RAW_DOMAIN,
    )?;
    let actual = replay.actual_root.restore()?;
    let root_moves = root
        .root_moves
        .iter()
        .map(|movement| BoardMove::from_uci(movement).map_err(|error| invalid(error.to_string())))
        .collect::<Result<Vec<_>, _>>()?;
    let legal = actual.legal_moves();
    check(
        root_moves.iter().enumerate().all(|(slot, movement)| {
            legal.contains(movement) && !root_moves[..slot].contains(movement)
        }) && hex(rz_position::contracts::rules_snapshot_semantic_digest(
            &actual.snapshot(),
        )) == root.root_descriptor.input.state_and_history_sha256
            && actual.revision() == root.root_state.revision
            && actual
                .ordered_legal_moves()
                .moves()
                .iter()
                .copied()
                .filter(|movement| root_moves.contains(movement))
                .collect::<Vec<_>>()
                == root_moves,
        "lock-v2 historical Rules root and ordered restriction mismatch",
    )?;
    for pin in manifest.teachers.values() {
        check(
            root.native_origin
                .declaration
                .teacher_profiles
                .iter()
                .any(|profile| {
                    profile.canonical_sha256 == pin.canonical_sha256
                        && profile.capability_sha256 == pin.capability_sha256
                        && profile.profile.sha256 == pin.profile_sha256
                        && profile.binary.sha256 == pin.binary_sha256
                        && profile.binary.bytes == pin.binary_bytes
                        && profile.source_commit == pin.source_commit
                }),
            "lock-v2 historical teacher absent from original declaration",
        )?;
    }
    super::historical_scope::verify_original_native_key_for_sources(
        &common,
        native,
        &replay,
        &root.native_origin.startup_observation.weights_sha256,
        &root.native_origin.startup_observation.backend_sha256,
    )?;
    Ok(scope)
}
