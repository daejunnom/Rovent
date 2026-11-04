//! Metadata validation for the separate /2 input-lock native/root domain.
//! No decoded value grants a live root, worker, teacher, or physical fence.
use crate::pals_data_v3::{generation_digest, generation_sha};
use crate::rovent_corpus::RoventNativeLifecycle;
use crate::rovent_corpus_v2::RoventCorpusManifestV2;
use crate::{
    ManifestError, RoventExternalProfilePin, RoventInputLockV2, RoventRootAllocationClockV2,
    RoventRootDescriptorV2,
};
use rz_contracts::{Digest, rovent::ExecutionMode};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

pub const ROVENT_LOCK_V2_NATIVE_RAW_DOMAIN: &str = "rz-rovent-native-lock-v2-raw/1";
pub const ROVENT_LOCK_V2_ORIGIN_DOMAIN: &str = "rz-rovent-collection-native-origin-lock-v2/1";
pub const ROVENT_LOCK_V2_ROOT_DOMAIN: &str = "rz-rovent-collection-root-admission-lock-v2/1";

fn require(ok: bool, message: &'static str) -> Result<(), ManifestError> {
    if ok {
        Ok(())
    } else {
        Err(ManifestError::Integrity(message.into()))
    }
}
fn hex(value: Digest) -> String {
    value.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

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

/// Verifies the exact source/lock/root metadata join. The supplied `Value`
/// must have been decoded with the crate's bounded duplicate-key decoder;
/// live identity and historical Rules replay remain separate responsibilities.
pub fn validate_lock_v2_root_metadata(
    raw: &Value,
    expected_sha256: &Value,
    lock: &RoventInputLockV2,
) -> Result<(), ManifestError> {
    lock.validate()?;
    let serialized =
        serde_json::to_string(raw).map_err(|e| ManifestError::Integrity(e.to_string()))?;
    let root: Root = crate::decode_json(&serialized)?;
    let origin = &root.native_origin;
    let declared = &origin.declaration;
    let observed = &origin.startup_observation;
    let namespace = &origin.registered_namespace;
    let descriptor = &root.root_descriptor;
    let input = &descriptor.input;
    let model = &lock.manifest.engine.pc.proposer;
    let mode = match lock.manifest.engine.mode {
        ExecutionMode::HybridPcr => "hybrid_pcr",
        ExecutionMode::HybridPc => "hybrid_pc",
        ExecutionMode::CpuPcr => "cpu_pcr",
        ExecutionMode::CpuPc => "cpu_pc",
    };
    let root_digest = generation_digest(ROVENT_LOCK_V2_ROOT_DOMAIN, raw)?;
    let native_digest = generation_digest(ROVENT_LOCK_V2_ORIGIN_DOMAIN, &raw["native_origin"])?;
    let source_digest = crate::rovent_source_aggregate_sha256(
        &lock.manifest.engine.source_commit,
        &lock.manifest.engine.source_components,
    )?;
    // Apply the per-object bounds as well as typed joins, not merely a bound
    // on the containing record or publication.
    let descriptor_json =
        serde_json::to_string(descriptor).map_err(|e| ManifestError::Integrity(e.to_string()))?;
    let allocation_json = serde_json::to_string(&root.allocation_clock)
        .map_err(|e| ManifestError::Integrity(e.to_string()))?;
    let bounded_descriptor = RoventRootDescriptorV2::from_json(&descriptor_json)?;
    let bounded_allocation = RoventRootAllocationClockV2::from_json(&allocation_json)?;
    bounded_descriptor.verify_against(lock)?;
    bounded_allocation.verify_against(&bounded_descriptor, lock)?;
    require(
        expected_sha256.as_str() == Some(root_digest.as_str())
            && root.schema == ROVENT_LOCK_V2_ROOT_DOMAIN
            && origin.schema == ROVENT_LOCK_V2_ORIGIN_DOMAIN
            && root.native_origin_sha256 == native_digest
            && origin.input_lock_domain == crate::ROVENT_LOCK_V2_DOMAIN
            && origin.input_lock_sha256 == lock.canonical_sha256
            && lock.manifest.purpose == crate::RoventPurpose::Collection
            && declared.purpose == "collection"
            && declared.mode == mode
            && observed.mode == mode
            && declared.source_commit == lock.manifest.engine.source_commit
            && declared.source_components == lock.manifest.engine.source_components
            && declared.source_sha256_domain == crate::ROVENT_SOURCE_AGGREGATE_DOMAIN
            && declared.source_sha256 == source_digest
            && declared.model_semantics == model.semantics
            && declared.encoding == model.encoding
            && declared.model_implementation_sha256 == model.implementation_sha256
            && declared.teacher_profiles == lock.manifest.engine.teachers
            && observed.domain == "rz-rovent-native-startup/1"
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
            && matches!(
                observed.topology.as_str(),
                "roles_only" | "own_researcher" | "external_researcher"
            )
            && namespace.representation_model_domain == "loaded_checkpoint_sha256"
            && namespace.representation_model_sha256 == observed.weights_sha256
            && namespace.representation_encoding_sha256 == declared.encoding_sha256
            && namespace.representation_precision == "fp32"
            && namespace.frozen_epoch > 0
            && !namespace.value_semantics.is_empty()
            && !namespace.value_model.is_empty()
            && !namespace.value_encoding.is_empty()
            && namespace.value_precision == "fp32_tf32_off"
            && root.root_descriptor_sha256 == hex(descriptor.digest()?)
            && root.allocation_clock_sha256 == hex(root.allocation_clock.digest()?)
            && input.authority.epoch == root.process_epoch
            && input.authority.game == root.game_generation
            && input.authority.root == root.root_generation
            && input.authority.implementation_sha256 == root.implementation_sha256
            && input.clock.hard.clock_epoch == root.deadline_clock
            && input.clock.hard.at_ns == root.deadline_tick
            && input.state_and_history_sha256 == root.root_state.semantic
            && input.allowed_root_moves == root.root_moves
            && input.original_go.depth_plies.map(|v| v.get()) == root.depth_plies
            && input.original_go.mate_moves.map(|v| v.get()) == root.mate_moves
            && input.original_go.requested_nodes == root.requested_nodes
            && descriptor.root_budget_sha256 == root.root_budget_sha256
            && !root.root_moves.is_empty()
            && root.root_moves.len() <= 218
            && generation_sha(&root.selective_policy_sha256)
            && [
                &declared.model_semantics_sha256,
                &declared.encoding_sha256,
                &declared.encoding_registry_sha256,
                &observed.export_manifest_sha256,
                &observed.backend_sha256,
                &observed.runtime_library_sha256,
                &observed.native_configuration_sha256,
            ]
            .into_iter()
            .all(|s| generation_sha(s))
            && observed
                .runtime_bundle_codec_sha256
                .as_ref()
                .is_none_or(|s| generation_sha(s)),
        "lock-v2 collection origin/root metadata mismatch",
    )?;
    let _ = (
        root.root_state.owner,
        root.root_state.revision,
        root.store_root_state,
    );
    Ok(())
}

/// Verify native submit/physical completion/retirement metadata and the
/// distinct lock/2 root domain. Actual historical Rules, file and teacher
/// evidence require the separate consumer and do not become live authority.
pub fn validate_native_lifecycle_lock_v2(
    native: &RoventNativeLifecycle,
    manifest: &RoventCorpusManifestV2,
    lock: &RoventInputLockV2,
) -> Result<(), ManifestError> {
    manifest.validate()?;
    native.validate_submitted_input_for_domain(
        &manifest.common_registration_projection(),
        ROVENT_LOCK_V2_NATIVE_RAW_DOMAIN,
        |root, sha| validate_lock_v2_root_metadata(root, sha, lock),
        |root| root["native_origin"]["declaration"]["purpose"] == "collection",
    )?;
    let raw = &native.raw_retirement["root_admission"];
    let origin = &raw["native_origin"];
    let ticket = &native.raw_submission["ticket"];
    let source = &origin["declaration"];
    let observed = &origin["startup_observation"];
    let namespace = &origin["registered_namespace"];
    require(
        raw["process_epoch"].as_u64() == Some(native.process_epoch)
            && raw["game_generation"].as_u64() == Some(native.game_generation)
            && raw["root_generation"].as_u64() == Some(native.root_generation)
            && ticket["implementation_sha256"] == raw["implementation_sha256"]
            && ticket["deadline_clock"] == raw["deadline_clock"]
            && ticket["deadline_tick"] == raw["deadline_tick"]
            && ticket["root_state"] == raw["root_state"]
            && ticket["representation"]["frozen_epoch"] == namespace["frozen_epoch"]
            && source["source_sha256"] == manifest.source_sha256
            && source["source_commit"] == manifest.source_commit
            && source["model_semantics_sha256"] == manifest.model_semantics_sha256
            && observed["runtime_library_sha256"] == manifest.runtime_sha256
            && observed["weights_sha256"] == native.model_sha256
            && source["encoding_sha256"] == native.encoding_sha256,
        "lock-v2 native ticket/source/namespace does not match root",
    )
}
