//! Synthetic declaration for CPU-only proxy preparation tests.
//!
//! The referenced model, engine and system libraries do not exist: this
//! fixture only exercises lock parsing and admission, never native startup.

use std::collections::{BTreeMap, BTreeSet};

use rz_experiments::{
    ArtifactRef, NativeRuntimeSourceV6, PALS_SYSTEM_STARTUP_CONFIG_V6_DOMAIN,
    PALS_V5_ENCODING_SCHEMA, PALS_V5_LINE_BUCKETS, PALS_V5_MODEL_SCHEMA, PALS_V5_MODEL_SEMANTICS,
    PalsArtifactDomainV5, PalsComponentV3, PalsCoordinatorV5, PalsCpuEndpointsV5,
    PalsCpuSelectionV5, PalsEngineModeV5, PalsExecutionWindowV5, PalsModelBackendV3,
    PalsModelSpecificationV5, PalsPrecisionV3, PalsResourcePlanV5, PalsReviewContextV5, PalsRoleV3,
    PalsStartupAssetPurposeV5, PalsStartupAssetV5, PalsSystemCpuPlacementV6, PalsSystemEndpointV6,
    PalsSystemNativeOptionsV6, PalsSystemSourcePolicyV6, PalsSystemStartupConfigV6,
    PalsSystemStartupLockV6, SystemLibrarySpecV6, SystemPlatformV6, SystemRuntimeLockV6,
    SystemRuntimeSpecV6,
};
use sha2::{Digest, Sha256};

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn artifact(path: &str) -> ArtifactRef {
    let payload = format!("synthetic test artifact: {path}");
    ArtifactRef {
        path: path.into(),
        sha256: digest(payload.as_bytes()),
        bytes: payload.len() as u64,
        source: "https://example.org/synthetic-test-only".into(),
        license: "MIT".into(),
    }
}

fn component(name: &str) -> PalsComponentV3 {
    PalsComponentV3 {
        semantic_id: name.into(),
        implementation_sha256: digest(format!("synthetic component: {name}").as_bytes()),
        options: BTreeMap::new(),
    }
}

pub(super) fn fixture_lock() -> PalsSystemStartupLockV6 {
    let runtime = SystemRuntimeSpecV6 {
        platform: SystemPlatformV6::Linux,
        libraries: ["ort-core", "ort-provider-cuda"]
            .into_iter()
            .enumerate()
            .map(|(index, role)| {
                let payload = format!("synthetic system library: {role}");
                SystemLibrarySpecV6 {
                    role: role.into(),
                    requested_path: format!("/test/lib{index}.so"),
                    canonical_path: format!("/test/lib{index}.so"),
                    version: None,
                    sha256: digest(payload.as_bytes()),
                    bytes: payload.len() as u64,
                    device: 1,
                    inode: index as u64 + 1,
                    required_mapping: true,
                }
            })
            .collect(),
        system_identity_sha256: digest(b"synthetic system identity"),
        cuda_device_id: 0,
        cuda_arena_bytes: 1024,
        startup_ms: 30_000,
        drain_ms: 1_000,
        capture_max_libraries: 2,
        capture_max_file_bytes: 200,
        capture_max_total_bytes: 400,
        capture_timeout_ms: 100,
    };
    let options = PalsSystemNativeOptionsV6 {
        cache_public_memory: false,
        device_public_memory: false,
        cudnn_conv_search: "legacy_implicit_exhaustive".into(),
        approximate_warm: false,
        cpu_placement: PalsSystemCpuPlacementV6::StrictCudaOnly,
        startup_ms: runtime.startup_ms,
        drain_ms: runtime.drain_ms,
    };
    let mut runtime_component = component("system-runtime/test");
    runtime_component.options = options.canonical_runtime_options(&runtime);
    let resources = PalsResourcePlanV5 {
        gpu_count: 1,
        gpu_model: Some("declared-test-gpu".into()),
        ..Default::default()
    };
    let model = PalsModelSpecificationV5 {
        profile: "virtual_review_v1".into(),
        model_schema: PALS_V5_MODEL_SCHEMA.into(),
        model_semantics: PALS_V5_MODEL_SEMANTICS.into(),
        encoding_schema: PALS_V5_ENCODING_SCHEMA.into(),
        semantic_sha256: digest(b"synthetic model semantics"),
        encoding_sha256: digest(b"synthetic encoding"),
        implementation_sha256: digest(b"synthetic model implementation"),
        weights: artifact("test/weights"),
        artifact_domain: PalsArtifactDomainV5::Untrained,
        provenance: None,
        checkpoint_domain: None,
        epoch: 1,
        backend: PalsModelBackendV3::OrtCuda,
        precision: PalsPrecisionV3::Fp32,
        tf32: false,
        exported_roles: BTreeSet::from([PalsRoleV3::Proposer, PalsRoleV3::Critic]),
        width: 384,
        latent_count: 16,
        temporal_width: 64,
        candidate_head_width: 128,
        record_feature_axes: 8,
        query_feature_axes: 8,
        max_records: 64,
        max_anchors: 4,
        line_buckets: PALS_V5_LINE_BUCKETS.to_vec(),
        one_dynamic_line_graph: true,
        ids_as_numeric_features: false,
        fresh_warm_separate_domains: true,
    };
    let endpoint = PalsSystemEndpointV6 {
        id: "test-engine".into(),
        binary: artifact("test/engine"),
        source_commit: None,
        mode: PalsEngineModeV5::Deployment,
        model,
        cpu: PalsCpuEndpointsV5 {
            cpu_t: PalsCpuSelectionV5::Disabled,
            cpu_r: PalsCpuSelectionV5::Own,
            own: component("own/test"),
            external: None,
        },
        context: PalsReviewContextV5::default(),
        coordinator: PalsCoordinatorV5::default(),
        resources,
        search: component("search/test"),
        runtime: runtime_component,
        v_generate: false,
        v_load: false,
        v_call: false,
    };
    use PalsStartupAssetPurposeV5 as Purpose;
    let assets = [
        (Purpose::Weights, "weights"),
        (Purpose::ExportManifest, "export"),
        (Purpose::RulesDescriptor, "rules"),
        (Purpose::PublicGraph, "public"),
        (Purpose::SharedPcGraph, "shared"),
        (Purpose::InitializationEvidence, "initialization"),
        (Purpose::InitializationCheckpointMetadata, "checkpoint"),
    ]
    .into_iter()
    .map(|(purpose, name)| PalsStartupAssetV5 {
        artifact: artifact(&format!("test/{name}")),
        snapshot_path: format!("inputs/test/{name}"),
        purpose,
    })
    .collect();
    let config = PalsSystemStartupConfigV6 {
        domain: PALS_SYSTEM_STARTUP_CONFIG_V6_DOMAIN.into(),
        run_id: "test-system-startup".into(),
        endpoint,
        window: PalsExecutionWindowV5 {
            window_id: "test-window".into(),
            wall_time_max_ms: 60_000,
            cleanup_max_ms: 2_000,
            output_bytes_max: 1024,
            agent_renewable: true,
        },
        source_components: BTreeMap::new(),
        source_policy: PalsSystemSourcePolicyV6::Observe,
        assets,
        system_runtime_lock: SystemRuntimeLockV6::new(NativeRuntimeSourceV6::SystemCuda(runtime))
            .expect("synthetic runtime declaration must be valid"),
        native_options: options,
        process_generation: 1,
        runtime_directory: "baseline-runtime".into(),
    };
    PalsSystemStartupLockV6::new(config).expect("synthetic startup declaration must be valid")
}
