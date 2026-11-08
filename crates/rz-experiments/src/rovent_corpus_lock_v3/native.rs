//! Metadata join for the original /3 system-CUDA native owner and root.
//! Parsing validates declarations only; it does not recreate live authority,
//! prove compiled-source provenance, or convert a prior receipt.
use crate::pals_data_v3::{generation_digest, generation_sha};
use crate::rovent_corpus::RoventNativeLifecycle;
use crate::rovent_corpus_v2::RoventCorpusManifestV2;
use crate::{
    ManifestError, NativeRuntimeSourceV6, RoventExternalProfilePin, RoventInputLockV3,
    RoventModelPin, RoventRootAllocationClockV3, RoventRootDescriptorV3,
    SystemLibraryObservationV6, SystemRuntimeSpecV6,
};
use rz_contracts::{Digest, rovent::ExecutionMode};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

pub const ROVENT_LOCK_V3_NATIVE_RAW_DOMAIN: &str = "rz-rovent-native-lock-v3-raw/1";
pub const ROVENT_LOCK_V3_ORIGIN_DOMAIN: &str = "rz-rovent-collection-native-origin-lock-v3/1";
pub const ROVENT_LOCK_V3_ROOT_DOMAIN: &str = "rz-rovent-collection-root-admission-lock-v3/1";

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
fn explicit_optional<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
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
    system_runtime_identity_sha256: String,
    system_libraries: Vec<SystemLibraryObservationV6>,
    #[serde(deserialize_with = "explicit_optional")]
    unregistered_native_mappings: Option<Vec<String>>,
    provider_ready: bool,
    native_configuration_sha256: String,
    startup_configuration_sha256: String,
    search_configuration_sha256: String,
    validator_loaded: bool,
    physical_workers_created: usize,
    neural_service_executed: bool,
}

/// Checks the system provider observation without deriving readiness from a
/// startup domain or from the mere presence of registered libraries.
fn validate_system_startup(
    observed: &StartupObservation,
    system: &SystemRuntimeSpecV6,
) -> Result<(), ManifestError> {
    let core = system
        .libraries
        .iter()
        .find(|item| item.role == "ort-core")
        .ok_or_else(|| ManifestError::Integrity("system ORT core identity missing".into()))?;
    system.validate_ready_observed(&observed.system_libraries)?;
    require(
        matches!(
            observed.domain.as_str(),
            "rz-rovent-native-startup/1" | "rz-rovent-native-system-startup/1"
        ) && observed.runtime_library_sha256 == core.sha256
            && observed.system_runtime_identity_sha256 == system.system_identity_sha256
            && observed
                .unregistered_native_mappings
                .as_ref()
                .is_some_and(Vec::is_empty)
            && observed.provider_ready,
        "lock-v3 collection system CUDA startup observation mismatch",
    )
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

pub fn validate_lock_v3_root_metadata(
    raw: &Value,
    expected_sha256: &Value,
    lock: &RoventInputLockV3,
) -> Result<(), ManifestError> {
    if matches!(
        lock.manifest.system_runtime_lock.runtime,
        NativeRuntimeSourceV6::Cpu
    ) {
        return super::cpu_native::validate_lock_v3_cpu_root_metadata(raw, expected_sha256, lock);
    }
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
    let system = match &lock.manifest.system_runtime_lock.runtime {
        NativeRuntimeSourceV6::SystemCuda(spec) => spec,
        NativeRuntimeSourceV6::Cpu => {
            return Err(ManifestError::Integrity(
                "system CUDA /3 collection requires explicit system runtime".into(),
            ));
        }
    };
    let mode = match lock.manifest.engine.mode {
        ExecutionMode::HybridPcr => "hybrid_pcr",
        ExecutionMode::HybridPc => "hybrid_pc",
        ExecutionMode::CpuPcr => "cpu_pcr",
        ExecutionMode::CpuPc => "cpu_pc",
    };
    let root_digest = generation_digest(ROVENT_LOCK_V3_ROOT_DOMAIN, raw)?;
    let native_digest = generation_digest(ROVENT_LOCK_V3_ORIGIN_DOMAIN, &raw["native_origin"])?;
    let descriptor_json =
        serde_json::to_string(descriptor).map_err(|e| ManifestError::Integrity(e.to_string()))?;
    let allocation_json = serde_json::to_string(&root.allocation_clock)
        .map_err(|e| ManifestError::Integrity(e.to_string()))?;
    let bounded_descriptor = RoventRootDescriptorV3::from_json(&descriptor_json)?;
    let bounded_allocation = RoventRootAllocationClockV3::from_json(&allocation_json)?;
    bounded_descriptor.verify_against(lock)?;
    bounded_allocation.verify_against(&bounded_descriptor, lock)?;
    validate_system_startup(observed, system)?;
    require(
        expected_sha256.as_str() == Some(root_digest.as_str())
            && root.schema == ROVENT_LOCK_V3_ROOT_DOMAIN
            && origin.schema == ROVENT_LOCK_V3_ORIGIN_DOMAIN
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
            && observed.process_epoch == root.process_epoch
            && observed.process_epoch != 0
            && observed.weights_sha256 == model.weights.sha256
            && observed.startup_configuration_sha256 == lock.manifest.startup_configuration_sha256
            && observed.search_configuration_sha256
                == lock.manifest.startup_search_configuration_sha256
            && observed.validator_loaded == lock.manifest.engine.pc.validator.is_some()
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
        "lock-v3 collection origin/root metadata mismatch",
    )?;
    let _ = (
        root.root_state.owner,
        root.root_state.revision,
        root.store_root_state,
    );
    Ok(())
}

pub fn validate_native_lifecycle_lock_v3(
    native: &RoventNativeLifecycle,
    manifest: &RoventCorpusManifestV2,
    lock: &RoventInputLockV3,
) -> Result<(), ManifestError> {
    if matches!(
        lock.manifest.system_runtime_lock.runtime,
        NativeRuntimeSourceV6::Cpu
    ) {
        return super::cpu_native::validate_native_lifecycle_lock_v3_cpu(native, manifest, lock);
    }
    manifest.validate()?;
    native.validate_submitted_input_for_domain(
        &manifest.common_registration_projection(),
        ROVENT_LOCK_V3_NATIVE_RAW_DOMAIN,
        |root, sha| validate_lock_v3_root_metadata(root, sha, lock),
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
        "lock-v3 native ticket/source/namespace does not match root",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SystemLibrarySpecV6, SystemPlatformV6};

    fn ready_system() -> (SystemRuntimeSpecV6, StartupObservation) {
        let libraries: Vec<_> = ["ort-core", "ort-provider-cuda"]
            .into_iter()
            .enumerate()
            .map(|(index, role)| SystemLibrarySpecV6 {
                role: role.into(),
                requested_path: format!("/usr/lib/{role}.so"),
                canonical_path: format!("/usr/lib/{role}.so.1"),
                version: None,
                sha256: format!("{:064x}", index + 1),
                bytes: (index + 1) as u64,
                device: 5,
                inode: (index + 1) as u64,
                required_mapping: true,
            })
            .collect();
        let observations = libraries
            .iter()
            .map(|library| SystemLibraryObservationV6 {
                role: library.role.clone(),
                canonical_path: library.canonical_path.clone(),
                sha256: library.sha256.clone(),
                bytes: library.bytes,
                device: library.device,
                inode: library.inode,
                mapped: Some(true),
            })
            .collect();
        let system = SystemRuntimeSpecV6 {
            platform: SystemPlatformV6::Linux,
            system_identity_sha256: format!("{:064x}", 42),
            cuda_device_id: 0,
            cuda_arena_bytes: 1024,
            startup_ms: 1000,
            drain_ms: 200,
            capture_max_libraries: 2,
            capture_max_file_bytes: 2,
            capture_max_total_bytes: 3,
            capture_timeout_ms: 100,
            libraries,
        };
        let observed = StartupObservation {
            domain: "rz-rovent-native-system-startup/1".into(),
            process_epoch: 1,
            mode: "cpu_pc".into(),
            topology: "roles_only".into(),
            weights_sha256: "1".repeat(64),
            export_manifest_sha256: "2".repeat(64),
            backend_sha256: "3".repeat(64),
            runtime_library_sha256: system.libraries[0].sha256.clone(),
            system_runtime_identity_sha256: system.system_identity_sha256.clone(),
            system_libraries: observations,
            unregistered_native_mappings: Some(vec![]),
            provider_ready: true,
            native_configuration_sha256: "4".repeat(64),
            startup_configuration_sha256: "5".repeat(64),
            search_configuration_sha256: "6".repeat(64),
            validator_loaded: false,
            physical_workers_created: 1,
            neural_service_executed: false,
        };
        (system, observed)
    }

    #[test]
    fn system_startup_accepts_actual_and_legacy_domains_without_relaxing_readiness() {
        let (system, mut observed) = ready_system();
        validate_system_startup(&observed, &system).unwrap();
        observed.domain = "rz-rovent-native-startup/1".into();
        validate_system_startup(&observed, &system).unwrap();
        observed.domain = "rz-rovent-native-unknown/1".into();
        assert!(validate_system_startup(&observed, &system).is_err());
        observed.domain = "rz-rovent-native-system-startup/1".into();
        observed.provider_ready = false;
        assert!(validate_system_startup(&observed, &system).is_err());
        observed.provider_ready = true;
        observed.system_libraries[1].mapped = None;
        assert!(validate_system_startup(&observed, &system).is_err());
        observed.system_libraries[1].mapped = Some(true);
        observed.system_runtime_identity_sha256 = "0".repeat(64);
        assert!(validate_system_startup(&observed, &system).is_err());
        observed.system_runtime_identity_sha256 = system.system_identity_sha256.clone();
        observed.unregistered_native_mappings = None;
        assert!(validate_system_startup(&observed, &system).is_err());
    }
}
