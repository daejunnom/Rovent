//! Metadata join for the original /3 CPU native owner and root.
//! Parsing validates declarations only; it does not recreate live authority,
//! prove compiled-source provenance, or convert a prior receipt.
use crate::pals_data_v3::{generation_digest, generation_sha};
use crate::rovent_corpus::RoventNativeLifecycle;
use crate::rovent_corpus_v2::RoventCorpusManifestV2;
use crate::{
    ManifestError, NativeRuntimeSourceV6, RoventBackend, RoventDevice, RoventExternalProfilePin,
    RoventInputLockV3, RoventModelPin, RoventRootAllocationClockV3, RoventRootDescriptorV3,
};
use rz_contracts::{Digest, rovent::ExecutionMode};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

pub const ROVENT_LOCK_V3_CPU_NATIVE_RAW_DOMAIN: &str = "rz-rovent-native-cpu-lock-v3-raw/1";
pub const ROVENT_LOCK_V3_CPU_ORIGIN_DOMAIN: &str =
    "rz-rovent-collection-native-origin-cpu-lock-v3/1";
pub const ROVENT_LOCK_V3_CPU_ROOT_DOMAIN: &str =
    "rz-rovent-collection-root-admission-cpu-lock-v3/1";

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
    source_commit: Option<String>,
    source_components: BTreeMap<String, String>,
    model_semantics: String,
    encoding: String,
    validator_model: Option<RoventModelPin>,
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
    provider: String,
    provider_ready: bool,
    cpu_session_roles: Vec<String>,
    native_configuration_sha256: String,
    startup_configuration_sha256: String,
    search_configuration_sha256: String,
    validator_loaded: bool,
    physical_workers_created: usize,
    neural_service_executed: bool,
}

fn has_collection_cpu_sessions(
    observed: &StartupObservation,
    validator_pinned: bool,
    expected_configuration_sha256: &str,
) -> bool {
    observed.provider == "cpu"
        && observed.provider_ready
        && observed.validator_loaded
        && validator_pinned
        && observed.cpu_session_roles == ["public", "shared_pc", "validator"]
        && observed.native_configuration_sha256 == expected_configuration_sha256
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
    root_descriptor: RoventRootDescriptorV3,
    root_descriptor_sha256: String,
    allocation_clock: RoventRootAllocationClockV3,
    allocation_clock_sha256: String,
}

pub fn validate_lock_v3_cpu_root_metadata(
    raw: &Value,
    expected_sha256: &Value,
    lock: &RoventInputLockV3,
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
    require(
        matches!(
            lock.manifest.system_runtime_lock.runtime,
            NativeRuntimeSourceV6::Cpu
        ) && lock.manifest.engine.pc.backend == RoventBackend::OrtCpu
            && lock.manifest.engine.pc.device == RoventDevice::Cpu
            && lock.manifest.engine.pc.runtime_artifacts.len() == 1,
        "explicit CPU runtime, backend and device are required",
    )?;
    let mode = match lock.manifest.engine.mode {
        ExecutionMode::HybridPcr => "hybrid_pcr",
        ExecutionMode::HybridPc => "hybrid_pc",
        ExecutionMode::CpuPcr => "cpu_pcr",
        ExecutionMode::CpuPc => "cpu_pc",
    };
    let root_digest = generation_digest(ROVENT_LOCK_V3_CPU_ROOT_DOMAIN, raw)?;
    let native_digest = generation_digest(ROVENT_LOCK_V3_CPU_ORIGIN_DOMAIN, &raw["native_origin"])?;
    let descriptor_json =
        serde_json::to_string(descriptor).map_err(|e| ManifestError::Integrity(e.to_string()))?;
    let allocation_json = serde_json::to_string(&root.allocation_clock)
        .map_err(|e| ManifestError::Integrity(e.to_string()))?;
    let bounded_descriptor = RoventRootDescriptorV3::from_json(&descriptor_json)?;
    let bounded_allocation = RoventRootAllocationClockV3::from_json(&allocation_json)?;
    bounded_descriptor.verify_against(lock)?;
    bounded_allocation.verify_against(&bounded_descriptor, lock)?;
    require(
        expected_sha256.as_str() == Some(root_digest.as_str())
            && root.schema == ROVENT_LOCK_V3_CPU_ROOT_DOMAIN
            && origin.schema == ROVENT_LOCK_V3_CPU_ORIGIN_DOMAIN
            && root.native_origin_sha256 == native_digest
            && origin.input_lock_domain == crate::ROVENT_LOCK_V3_DOMAIN
            && origin.input_lock_sha256 == lock.canonical_sha256
            && lock.manifest.purpose == crate::RoventPurpose::Collection
            && declared.purpose == "collection"
            && declared.mode == mode
            && observed.mode == mode
            && declared.source_commit == lock.manifest.engine.source_commit
            && declared.source_components == lock.manifest.engine.source_components
            && declared.model_semantics == model.semantics
            && declared.encoding == model.encoding
            && declared.validator_model == lock.manifest.engine.pc.validator
            && declared.teacher_profiles == lock.manifest.engine.teachers
            && observed.domain == "rz-rovent-native-startup/1"
            && observed.process_epoch == root.process_epoch
            && observed.process_epoch != 0
            && observed.weights_sha256 == model.weights.sha256
            && lock.manifest.engine.pc.runtime_artifacts[0].sha256
                == observed.runtime_library_sha256
            && has_collection_cpu_sessions(
                observed,
                lock.manifest.engine.pc.validator.is_some(),
                &lock.manifest.engine.pc.backend_configuration_sha256,
            )
            && observed.startup_configuration_sha256 == lock.manifest.startup_configuration_sha256
            && observed.search_configuration_sha256
                == lock.manifest.startup_search_configuration_sha256
            && observed.physical_workers_created == 1
            && !observed.neural_service_executed
            && origin.process_boundary_observation == "unknown"
            && origin.neural_completion == "unknown"
            && origin.physical_completion == "unknown"
            && matches!(
                (
                    &lock.manifest.engine.researcher,
                    lock.manifest.engine.mode.has_native_r(),
                    observed.topology.as_str(),
                ),
                (None, false, "roles_only")
                    | (
                        Some(crate::RoventResearcher::Own { .. }),
                        true,
                        "own_researcher"
                    )
                    | (
                        Some(crate::RoventResearcher::External { .. }),
                        true,
                        "external_researcher"
                    )
            )
            && namespace.representation_model_domain == "loaded_checkpoint_sha256"
            && namespace.representation_model_sha256 == observed.weights_sha256
            && generation_sha(&namespace.representation_encoding_sha256)
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
                &observed.export_manifest_sha256,
                &observed.backend_sha256,
                &observed.runtime_library_sha256,
                &observed.native_configuration_sha256,
            ]
            .into_iter()
            .all(|s| generation_sha(s)),
        "lock-v3 CPU collection origin/root metadata mismatch",
    )?;
    let _ = (
        root.root_state.owner,
        root.root_state.revision,
        root.store_root_state,
    );
    Ok(())
}

pub fn validate_native_lifecycle_lock_v3_cpu(
    native: &RoventNativeLifecycle,
    manifest: &RoventCorpusManifestV2,
    lock: &RoventInputLockV3,
) -> Result<(), ManifestError> {
    manifest.validate()?;
    native.validate_submitted_input_for_domain(
        &manifest.common_registration_projection(),
        ROVENT_LOCK_V3_CPU_NATIVE_RAW_DOMAIN,
        |root, sha| validate_lock_v3_cpu_root_metadata(root, sha, lock),
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
            && source["source_commit"] == manifest.source_commit
            && native.source_sha256 == manifest.source_sha256
            && observed["runtime_library_sha256"] == manifest.runtime_sha256
            && observed["weights_sha256"] == native.model_sha256
            && namespace["representation_encoding_sha256"] == native.encoding_sha256,
        "lock-v3 CPU native ticket/source/namespace does not match root",
    )
}

#[cfg(test)]
#[path = "cpu_native_root_tests.rs"]
mod root_tests;

#[cfg(test)]
mod tests {
    use super::{StartupObservation, has_collection_cpu_sessions};
    use serde_json::json;

    #[test]
    fn cpu_collection_rejects_missing_validator_and_forged_provider() {
        let raw = json!({
            "domain": "rz-rovent-native-startup/1",
            "process_epoch": 1,
            "mode": "hybrid_pcr",
            "topology": "own_researcher",
            "weights_sha256": "a",
            "export_manifest_sha256": "b",
            "backend_sha256": "c",
            "runtime_library_sha256": "d",
            "provider": "cpu",
            "provider_ready": true,
            "cpu_session_roles": ["public", "shared_pc", "validator"],
            "native_configuration_sha256": "e",
            "startup_configuration_sha256": "f",
            "search_configuration_sha256": "g",
            "validator_loaded": true,
            "physical_workers_created": 1,
            "neural_service_executed": false
        });
        let parse = |value| serde_json::from_value::<StartupObservation>(value).unwrap();
        assert!(has_collection_cpu_sessions(&parse(raw.clone()), true, "e"));
        assert!(!has_collection_cpu_sessions(
            &parse(raw.clone()),
            false,
            "e"
        ));
        assert!(!has_collection_cpu_sessions(&parse(raw.clone()), true, "f"));
        let mut invalid = raw.clone();
        invalid["validator_loaded"] = json!(false);
        assert!(!has_collection_cpu_sessions(&parse(invalid), true, "e"));
        let mut invalid = raw.clone();
        invalid["cpu_session_roles"] = json!(["public", "shared_pc"]);
        assert!(!has_collection_cpu_sessions(&parse(invalid), true, "e"));
        let mut invalid = raw;
        invalid["provider"] = json!("cuda");
        assert!(!has_collection_cpu_sessions(&parse(invalid), true, "e"));
    }
}
