//! Authored metadata fixtures only: no file reads, framework, model or provider.
//! These references exercise the scope boundary; they certify no real G0 asset
//! or process. Numeric completion and game/startup authority remain separate.
use super::*;
use crate::{
    PALS_CUDNN_IMPLICIT_EXHAUSTIVE_V5, PALS_STARTUP_CONFIG_V5_DOMAIN, PALS_V5_ENCODING_SCHEMA,
    PALS_V5_LINE_BUCKETS, PALS_V5_MODEL_SCHEMA, PALS_V5_MODEL_SEMANTICS, PalsArtifactDomainV5,
    PalsComponentV3, PalsCoordinatorV5, PalsCpuEndpointsV5, PalsCpuSelectionV5, PalsEngineV5,
    PalsModelSpecificationV5, PalsNativeStartupOptionsV5, PalsNumericalScopeV5, PalsResourcePlanV5,
    PalsReviewContextV5, PalsRoleV3, PalsStartupAssetV5, compiled_source_components_v5,
};

fn fixture_artifact(path: &str) -> ArtifactRef {
    ArtifactRef {
        path: format!("fixtures/{path}"),
        sha256: "a".repeat(64),
        bytes: 1,
        source: "https://example.org/authored-numerical-scope-fixture".into(),
        license: "MIT".into(),
    }
}

fn fixture_asset(purpose: Purpose, name: &str) -> PalsStartupAssetV5 {
    PalsStartupAssetV5 {
        artifact: fixture_artifact(name),
        snapshot_path: format!("inputs/model/{name}"),
        purpose,
    }
}

fn scope() -> PalsNumericalScopeV5 {
    let registered_encoding = rz_eval::pals_model::virtual_review::registered_encoding_sha256()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    PalsNumericalScopeV5 {
        mode: PalsEngineModeV5::Deployment,
        model: PalsModelSpecificationV5 {
            profile: "virtual_review_v1".into(),
            model_schema: PALS_V5_MODEL_SCHEMA.into(),
            model_semantics: PALS_V5_MODEL_SEMANTICS.into(),
            encoding_schema: PALS_V5_ENCODING_SCHEMA.into(),
            semantic_sha256: "c".repeat(64),
            encoding_sha256: registered_encoding,
            implementation_sha256: "e".repeat(64),
            weights: fixture_artifact("weights.bin"),
            artifact_domain: PalsArtifactDomainV5::Untrained,
            provenance: None,
            checkpoint_domain: None,
            epoch: 1,
            backend: PalsModelBackendV3::OrtCpu,
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
        },
        source_commit: "f".repeat(40),
        source_components: compiled_source_components_v5().unwrap(),
        assets: [
            (Purpose::Weights, "weights.bin"),
            (Purpose::ExportManifest, "export.json"),
            (Purpose::PublicGraph, "public_memory.onnx"),
            (Purpose::SharedPcGraph, "shared_pc_if.onnx"),
            (Purpose::RuntimeLibrary, "runtime.so"),
            (Purpose::RulesDescriptor, "rules.json"),
            (Purpose::InitializationCheckpointMetadata, "checkpoint.json"),
            (Purpose::InitializationEvidence, "initialization.json"),
        ]
        .into_iter()
        .map(|(purpose, name)| fixture_asset(purpose, name))
        .collect(),
        training_executed: false,
    }
}

fn component(id: &str) -> PalsComponentV3 {
    PalsComponentV3 {
        semantic_id: id.into(),
        implementation_sha256: "b".repeat(64),
        options: BTreeMap::new(),
    }
}

fn startup(value: &PalsNumericalScopeV5) -> PalsAdaptiveStartupConfigV5 {
    PalsAdaptiveStartupConfigV5 {
        schema_version: 5,
        domain: PALS_STARTUP_CONFIG_V5_DOMAIN.into(),
        semantic_lock_sha256: "1".repeat(64),
        endpoint: PalsEngineV5 {
            id: "authored-scope-fixture".into(),
            binary: fixture_artifact("engine"),
            source_commit: value.source_commit.clone(),
            mode: value.mode,
            model: value.model.clone(),
            cpu: PalsCpuEndpointsV5 {
                cpu_t: PalsCpuSelectionV5::Disabled,
                cpu_r: PalsCpuSelectionV5::Own,
                own: component("own/1"),
                external: None,
            },
            context: PalsReviewContextV5::default(),
            coordinator: PalsCoordinatorV5::default(),
            resources: PalsResourcePlanV5::default(),
            search: component("search/1"),
            runtime: component("runtime/1"),
            v_generate: false,
            v_load: false,
            v_call: false,
        },
        runtime_directory: "baseline-runtime".into(),
        native_options: PalsNativeStartupOptionsV5 {
            native_loading_profile: None,
            cuda_device_id: None,
            cuda_arena_bytes: None,
            cache_public_memory: true,
            device_public_memory: false,
            cudnn_conv_search: PALS_CUDNN_IMPLICIT_EXHAUSTIVE_V5.into(),
            approximate_warm: false,
            startup_ms: 10_000,
            drain_ms: 5_000,
        },
        assets: value.assets.clone(),
        source_components: value.source_components.clone(),
        training_executed: value.training_executed,
    }
}

#[test]
fn scope_copies_only_exact_numerical_inputs_without_completing_startup() {
    let original = scope();
    let config = startup(&original);
    assert!(config.endpoint.validate().is_ok());
    let copied = PalsNumericalScopeV5::from_config(&config);
    assert_eq!(copied.mode, config.endpoint.mode);
    assert_eq!(copied.model, config.endpoint.model);
    assert_eq!(copied.source_commit, config.endpoint.source_commit);
    assert_eq!(copied.source_components, config.source_components);
    assert_eq!(copied.assets, config.assets);
    assert_eq!(copied.training_executed, config.training_executed);
    assert!(copied.validate_pre_acceptance().is_ok());
    assert!(copied.asset(Purpose::NumericalAcceptance).is_err());
    assert!(copied.asset(Purpose::NumericalRawEvidence).is_err());

    // CPU_T and whole-engine mode authority are not granted by this scope.
    // Deliberately invalid Deployment CPU_T metadata is irrelevant to the
    // cloned tensor-math inputs; full startup keeps its separate strict gate.
    let mut invalid_whole_engine = config;
    invalid_whole_engine.endpoint.cpu.cpu_t = PalsCpuSelectionV5::Own;
    invalid_whole_engine.endpoint.v_load = true;
    assert!(invalid_whole_engine.endpoint.validate().is_err());
    assert!(
        PalsNumericalScopeV5::from_config(&invalid_whole_engine)
            .validate_pre_acceptance()
            .is_ok()
    );
}

#[test]
fn pre_acceptance_rejects_each_missing_generated_or_runtime_reference() {
    let original = scope();
    for purpose in [
        Purpose::Weights,
        Purpose::ExportManifest,
        Purpose::PublicGraph,
        Purpose::SharedPcGraph,
        Purpose::RuntimeLibrary,
        Purpose::RulesDescriptor,
        Purpose::InitializationCheckpointMetadata,
        Purpose::InitializationEvidence,
    ] {
        let mut missing = original.clone();
        missing.assets.retain(|asset| asset.purpose != purpose);
        assert!(missing.validate_pre_acceptance().is_err(), "{purpose:?}");
        assert!(missing.asset(purpose).is_err(), "{purpose:?}");
    }
}

#[test]
fn pre_acceptance_preserves_exact_weights_and_unique_owned_assets() {
    let original = scope();
    let mut foreign = original.clone();
    foreign.model.weights.sha256 = "b".repeat(64);
    assert!(foreign.validate_pre_acceptance().is_err());
    let mut ambiguous = original.clone();
    ambiguous
        .assets
        .push(fixture_asset(Purpose::ExportManifest, "second-export.json"));
    assert!(ambiguous.asset(Purpose::ExportManifest).is_err());
    assert!(ambiguous.validate_pre_acceptance().is_err());
    let mut duplicate_snapshot = original.clone();
    duplicate_snapshot.assets[1].snapshot_path = duplicate_snapshot.assets[0].snapshot_path.clone();
    assert!(duplicate_snapshot.validate_pre_acceptance().is_err());
    let mut duplicate_source_path = original;
    duplicate_source_path.assets[1].artifact.path =
        duplicate_source_path.assets[0].artifact.path.clone();
    assert!(duplicate_source_path.validate_pre_acceptance().is_err());
}

#[test]
fn pre_acceptance_rejects_missing_or_foreign_compiled_numerical_sources() {
    let original = scope();
    for key in NATIVE_COMPONENTS {
        let mut missing = original.clone();
        missing.source_components.remove(key);
        assert!(missing.validate_pre_acceptance().is_err(), "{key}");
        let mut foreign = original.clone();
        let actual = &original.source_components[key];
        let alternative = if actual == &"0".repeat(64) { "1" } else { "0" };
        foreign
            .source_components
            .insert(key.into(), alternative.repeat(64));
        assert!(foreign.validate_pre_acceptance().is_err(), "{key}");
    }
}

#[test]
fn pre_acceptance_rejects_non_lowercase_commit_and_training_transport() {
    let original = scope();
    for invalid in [
        "F".repeat(40),
        "f".repeat(39),
        "g".repeat(40),
        String::new(),
    ] {
        let mut changed = original.clone();
        changed.source_commit = invalid;
        assert!(changed.validate_pre_acceptance().is_err());
    }
    let mut training = original;
    training.training_executed = true;
    assert!(training.validate_pre_acceptance().is_err());
}

#[test]
fn pre_acceptance_keeps_model_encoding_backend_precision_and_role_boundaries() {
    let original = scope();
    let mut foreign_encoding = original.clone();
    foreign_encoding.model.encoding_sha256 = "0".repeat(64);
    assert!(numerical_scope_bindings_v5(&foreign_encoding).is_err());
    let mut unsupported_backend = original.clone();
    unsupported_backend.model.backend = PalsModelBackendV3::RustCpu;
    assert!(numerical_scope_bindings_v5(&unsupported_backend).is_err());
    let mut tf32 = original.clone();
    tf32.model.tf32 = true;
    assert!(tf32.validate_pre_acceptance().is_err());
    let mut collection = original;
    collection.mode = PalsEngineModeV5::Collection;
    collection
        .model
        .exported_roles
        .insert(PalsRoleV3::Validator);
    assert!(collection.validate_pre_acceptance().is_err());
    collection.assets.push(fixture_asset(
        Purpose::ValidatorGraph,
        "role_validator.onnx",
    ));
    assert!(collection.validate_pre_acceptance().is_ok());
    collection.mode = PalsEngineModeV5::Deployment;
    collection
        .model
        .exported_roles
        .remove(&PalsRoleV3::Validator);
    assert!(collection.validate_pre_acceptance().is_err());
}

#[test]
fn scope_codec_keeps_actual_weights_required_and_rejects_extra_authority_fields() {
    let original = serde_json::to_value(scope()).unwrap();
    let mut missing_weights = original.clone();
    missing_weights["model"]
        .as_object_mut()
        .unwrap()
        .remove("weights");
    assert!(serde_json::from_value::<PalsNumericalScopeV5>(missing_weights).is_err());
    let mut fake_authority = original;
    fake_authority
        .as_object_mut()
        .unwrap()
        .insert("startup_accepted".into(), serde_json::Value::Bool(true));
    assert!(serde_json::from_value::<PalsNumericalScopeV5>(fake_authority).is_err());
}

#[test]
fn pre_acceptance_does_not_turn_untrained_metadata_into_generation_provenance() {
    let original = scope();
    let mut relabeled = original.clone();
    relabeled.assets.push(fixture_asset(
        Purpose::GenerationSpecification,
        "generation.json",
    ));
    assert!(relabeled.validate_pre_acceptance().is_err());
    let mut actual_generation = original;
    actual_generation.model.artifact_domain = PalsArtifactDomainV5::RealGeneration;
    // A domain flag cannot stand in for actual provenance/checkpoint metadata.
    assert!(actual_generation.validate_pre_acceptance().is_err());
}

#[test]
fn pre_acceptance_rejects_unsafe_private_names_and_invalid_artifact_identity() {
    let original = scope();
    for invalid in [
        "inputs/../foreign.json",
        "inputs/.env",
        "/inputs/export.json",
    ] {
        let mut unsafe_name = original.clone();
        unsafe_name.assets[1].snapshot_path = invalid.into();
        assert!(unsafe_name.validate_pre_acceptance().is_err());
    }
    let mut invalid = original;
    invalid.assets[1].artifact.sha256 = "A".repeat(64);
    assert!(invalid.validate_pre_acceptance().is_err());
}

#[test]
fn cpu_numerical_scope_cannot_claim_cuda_assets_or_control_inventory() {
    let original = scope();
    for purpose in [
        Purpose::CudaLibrary,
        Purpose::CudaManifest,
        Purpose::CudaControlInventory,
    ] {
        let mut cuda_claim = original.clone();
        cuda_claim
            .assets
            .push(fixture_asset(purpose, "cuda-claim.json"));
        assert!(cuda_claim.validate_pre_acceptance().is_err(), "{purpose:?}");
    }
}

#[test]
fn pre_acceptance_checks_asset_total_overflow_without_a_storage_threshold() {
    let original = scope();
    let mut oversized = original.clone();
    oversized
        .assets
        .iter_mut()
        .find(|asset| asset.purpose == Purpose::ExportManifest)
        .unwrap()
        .artifact
        .bytes = 18 * 1024 * 1024 * 1024;
    assert!(oversized.validate_pre_acceptance().is_ok());
    let mut overflowing = original;
    overflowing
        .assets
        .iter_mut()
        .find(|asset| asset.purpose == Purpose::ExportManifest)
        .unwrap()
        .artifact
        .bytes = u64::MAX;
    assert!(overflowing.validate_pre_acceptance().is_err());
}

#[test]
fn a_status_placeholder_cannot_complete_numerical_scope_admission() {
    let mut value = scope();
    let bytes = br#"{"status":"passed","cuda":"passed"}"#;
    let mut acceptance = fixture_asset(Purpose::NumericalAcceptance, "acceptance.json");
    acceptance.artifact.bytes = bytes.len() as u64;
    acceptance.artifact.sha256 = format!("{:x}", Sha256::digest(bytes));
    value.assets.push(acceptance);
    value
        .assets
        .push(fixture_asset(Purpose::NumericalRawEvidence, "raw.json"));
    assert!(value.validate_pre_acceptance().is_ok());
    let scoped = validate_pals_numerical_scope_acceptance_v5(&value, bytes, |_, _| {
        panic!("a status placeholder must fail before any artifact read")
    })
    .unwrap_err();
    let completed = validate_pals_numerical_acceptance_v5(&startup(&value), bytes, |_, _| {
        panic!("the completed config adapter must preserve the same early refusal")
    })
    .unwrap_err();
    assert_eq!(scoped.to_string(), completed.to_string());
}

#[test]
fn config_and_scope_reject_the_same_hash_matched_invalid_export_before_reads() {
    let mut value = scope();
    let bytes = b"{}";
    let export = value
        .assets
        .iter_mut()
        .find(|asset| asset.purpose == Purpose::ExportManifest)
        .unwrap();
    export.artifact.bytes = bytes.len() as u64;
    export.artifact.sha256 = format!("{:x}", Sha256::digest(bytes));
    assert!(value.validate_pre_acceptance().is_ok());

    // Both entry points reach the same producer metadata core after an exact
    // byte/hash join. Invalid metadata cannot yield a success summary, numeric
    // acceptance or an initializer/trainer/arena issuer; no child runs here.
    let scoped = crate::validate_pals_scope_artifact_domain_v5(&value, bytes, |_, _| {
        panic!("invalid export must fail before reading producer metadata")
    })
    .unwrap_err();
    let completed = crate::validate_pals_artifact_domain_v5(&startup(&value), bytes, |_, _| {
        panic!("the completed config adapter must preserve the same early refusal")
    })
    .unwrap_err();
    assert_eq!(scoped.to_string(), completed.to_string());
}

#[test]
fn numerical_raw_nonfinite_guards_remain_independent_of_scope_metadata() {
    assert!(scope().validate_pre_acceptance().is_ok());
    assert!(compare_raw(&[0.], &[0.]).is_ok());
    assert!(compare_probabilities(&[0.], &[0.]).is_ok());
    let finite = PalsNumericalTensorV5 {
        name: "finite-boundary".into(),
        dtype: PalsNumericalDtypeV5::F32Le,
        shape: vec![1, 1],
        bytes: 0f32.to_le_bytes().to_vec(),
    };
    assert!(pals_numerical_tensor_digest_v5(&[finite], PalsRole::Proposer).is_ok());
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(compare_raw(&[0.], &[invalid]).is_err());
        assert!(compare_probabilities(&[0.], &[invalid]).is_err());
        let tensor = PalsNumericalTensorV5 {
            name: "finite-boundary".into(),
            dtype: PalsNumericalDtypeV5::F32Le,
            shape: vec![1, 1],
            bytes: invalid.to_le_bytes().to_vec(),
        };
        assert!(pals_numerical_tensor_digest_v5(&[tensor], PalsRole::Proposer).is_err());
    }
}

fn native_run() -> PalsNumericalRawRunV5 {
    use rz_eval::pals_model::virtual_review::{
        ReviewAnchorInput, ReviewQueryInput, ReviewQueryPurpose,
    };
    let value = scope();
    let input = PalsVirtualReviewInput {
        schema: "rovezero.pals-virtual-review-input.v1".into(),
        role: PalsRole::Proposer,
        anchors: vec![ReviewAnchorInput {
            board: vec![0; 64],
            metadata: [0.; 16],
        }],
        records: Vec::new(),
        query: ReviewQueryInput {
            purpose: ReviewQueryPurpose::ProposePolicy,
            perspective: 1,
            prefix: Vec::new(),
            proposal: Vec::new(),
            counter: Vec::new(),
            counter_present: false,
            anchors: [0; 3],
        },
        candidates: vec![PalsCandidateToken {
            from: 1,
            to: 18,
            promotion: 0,
        }],
        divergences: Vec::new(),
        token_budget: 512,
    };
    let bindings = numerical_scope_bindings_v5(&value).unwrap();
    PalsNumericalRawRunV5 {
        schema_version: 5,
        domain: PALS_NUMERICAL_RAW_V5_DOMAIN.into(),
        run_id: "authored-no-physical-execution".into(),
        producer: PalsNumericalProducerV5::RustOrtNative,
        producer_source_component: bindings.source_components["model_native"].clone(),
        bindings,
        invocation: PalsNumericalInvocationV5::Fresh,
        batch: 1,
        observed_provider: Some("CPUExecutionProvider".into()),
        public_cache_hits: 0,
        completed_nn_inputs: 2,
        physical_completion_known: true,
        cases: vec![PalsNumericalRawCaseV5 {
            id: "authored-physical-guard-fixture".into(),
            candidate_order: input.candidates.clone(),
            input: PalsNumericalInputV5::VirtualReview(input),
            input_tensors: Vec::new(),
            input_tensor_sha256: "a".repeat(64),
            raw_output: PalsRawOutput {
                candidate_logits: vec![0.],
                wdl_logits: [0.; 3],
                divergence_logits: None,
                task_logits: None,
                private_latent: vec![0.; 16 * 384],
            },
            physical_completion_id: Some("authored-completion-id-not-native-evidence".into()),
        }],
    }
}

#[test]
fn numerical_raw_physical_unknown_and_foreign_source_remain_fail_closed() {
    // This tests only the unchanged run-envelope guard, not case math or a
    // provider. The authored run never passes complete numerical admission.
    let original = native_run();
    let expected = original.bindings.clone();
    assert!(validate_run(&original, &expected, PalsNumericalProducerV5::RustOrtNative).is_ok());
    let mut unknown = original.clone();
    unknown.physical_completion_known = false;
    assert!(validate_run(&unknown, &expected, PalsNumericalProducerV5::RustOrtNative).is_err());
    let mut missing_id = original.clone();
    missing_id.cases[0].physical_completion_id = None;
    assert!(
        validate_run(
            &missing_id,
            &expected,
            PalsNumericalProducerV5::RustOrtNative
        )
        .is_err()
    );
    let mut foreign = original.clone();
    foreign.producer_source_component = "0".repeat(64);
    assert!(validate_run(&foreign, &expected, PalsNumericalProducerV5::RustOrtNative).is_err());
    let mut wrong_inputs = original.clone();
    wrong_inputs.completed_nn_inputs = 1;
    assert!(
        validate_run(
            &wrong_inputs,
            &expected,
            PalsNumericalProducerV5::RustOrtNative
        )
        .is_err()
    );
    let mut reference = original;
    reference.producer = PalsNumericalProducerV5::IndependentPytorchFp32;
    reference.producer_source_component = expected.source_components["model_reference"].clone();
    assert!(
        validate_run(
            &reference,
            &expected,
            PalsNumericalProducerV5::IndependentPytorchFp32
        )
        .is_err()
    );
}
