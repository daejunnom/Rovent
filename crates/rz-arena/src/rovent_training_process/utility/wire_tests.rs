//! Actual Rust utility DTO serialization -> independent stdlib-only F decoders.
//! The file pins are metadata operands, not existing artifact/physical/native
//! evidence. No model, optimizer, cohort owner or execution lease is created.

use super::cost::*;
use super::execution::*;
use super::registration::*;
use super::registration_cohort::*;
use crate::rovent_training_process::recovery::metadata_wire_tests::decode_python_fixture;
use crate::rovent_training_process::recovery::{
    RoventTrainerRuntimeEndpoint, rovent_trainer_dependency_set_sha256,
};
use crate::rovent_training_process::{RoventTrainingFilePin, RoventTrainingResourcePolicy};
use rz_experiments::pals_data_v3::generation_digest;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::path::Path;

const FIXTURE_SCHEMA: &str = "rz-rovent-utility-metadata-cross-language-fixture/1";
const RESULT_SCHEMA: &str = "rz-rovent-utility-metadata-cross-language-result/1";

fn h(label: &str) -> String {
    format!("{:x}", Sha256::digest(label.as_bytes()))
}
fn pin(path: &str, label: &str) -> RoventTrainingFilePin {
    RoventTrainingFilePin {
        path: path.into(),
        sha256: h(label),
        bytes: 128,
    }
}
fn parameter(name: &str, shape: Vec<u32>) -> RoventUtilityParameter {
    RoventUtilityParameter {
        checkpoint_path: format!("parameters.{name}"),
        shape,
        dtype: "float32".into(),
        planned_requires_grad: true,
    }
}
fn fixture() -> Value {
    let recipe = RoventUtilityTrainingRecipe::fixed("utility-wire-owner").unwrap();
    let config = RoventUtilityBaseConfig {
        board_width: 2,
        line_width: 3,
        context_width: 4,
        local_width: 5,
        hidden_width: 6,
        causal_window: 3,
        max_lines: 64,
        max_plies: 256,
    };
    let spec = RoventUtilityParameterSpec {
        hidden_width: 2,
        input_width: config.utility_input_width().unwrap(),
        wire_to_parameter: RoventUtilityParameterNames {
            hidden_weights: "utility.hidden_weights".into(),
            hidden_bias: "utility.hidden_bias".into(),
            output_weights: "utility.output_weights".into(),
            output_bias: "utility.output_bias".into(),
        },
        parameters: RoventUtilityParameters {
            hidden_weights: parameter("hidden_weights", vec![2, 17]),
            hidden_bias: parameter("hidden_bias", vec![2]),
            output_weights: parameter("output_weights", vec![7, 2]),
            output_bias: parameter("output_bias", vec![7]),
        },
        optimizer_owner: recipe.optimizer_owner.clone(),
        frozen_components: ROVENT_UTILITY_FROZEN_COMPONENTS
            .iter()
            .map(|name| (*name).into())
            .collect(),
    };
    let spec_sha256 = spec
        .canonical_sha256(&config, &recipe.optimizer_owner)
        .unwrap();
    let source = RoventUtilityCohortSourceRegistration {
        collection_source_sha256: h("collection-source"),
        trainer_source_sha256: h("trainer-source"),
        cost_epoch_sha256: h("original-cohort-cost-epoch"),
        resource_profile_sha256: h("resource-profile"),
        cost_scope: ROVENT_UTILITY_COHORT_COST_SCOPE.into(),
        objective: ROVENT_UTILITY_COST_OBJECTIVE.into(),
        frozen_base: RoventUtilityFrozenBase {
            schema: ROVENT_UTILITY_BASE_SCHEMA.into(),
            semantics: ROVENT_UTILITY_BASE_SEMANTICS.into(),
            weights_sha256: h("frozen-base-semantic-parameter-identity"),
            config,
        },
        utility: RoventUtilityInitialParameters {
            schema: ROVENT_UTILITY_SCHEMA.into(),
            semantics: ROVENT_UTILITY_SEMANTICS.into(),
            initial_weights_sha256: h("initial-utility-semantic-parameter-identity"),
            base_weights_sha256: h("frozen-base-semantic-parameter-identity"),
            parameter_spec_sha256: spec_sha256,
        },
    };
    let preparation = prepare_utility_cohort_training(&recipe, &source, &spec).unwrap();
    let preparation_sha = validate_utility_cohort_preparation(&preparation).unwrap();
    let window = RoventUtilityTrainWindow {
        schema: ROVENT_UTILITY_WINDOW_DOMAIN.into(),
        window_id: "utility-wire-window".into(),
        run_id: "utility-wire-run".into(),
        preparation_sha256: preparation_sha.clone(),
        dataset_sha256: h("dataset-canonical-semantic-seal"),
        device: "cpu".into(),
        precision: "fp32".into(),
        tf32: false,
        deterministic: true,
        cpu_threads: 2,
        interop_threads: 1,
        training_seed: 31,
        microbatch: 1,
        accumulation: 8,
        max_optimizer_updates: 4,
        checkpoint_every_updates: 2,
        max_wall_time_ns: 60_000_000_000,
        cleanup_time_ns: 30_000_000_000,
        max_read_bytes: 1 << 20,
        max_output_bytes: 1 << 20,
        max_groups: 16,
        max_contexts: 128,
    };
    let endpoints = vec![
        RoventTrainerRuntimeEndpoint {
            name: "torch.__file__".into(),
            artifact: pin(
                "/metadata-fixture/deps/torch/__init__.py",
                "torch-python-endpoint",
            ),
        },
        RoventTrainerRuntimeEndpoint {
            name: "torch._C.__file__".into(),
            artifact: pin(
                "/metadata-fixture/deps/torch/_C.so",
                "torch-native-endpoint",
            ),
        },
    ];
    let registration = RoventUtilityExecutionRegistration {
        schema: ROVENT_UTILITY_EXECUTION_REGISTRATION.into(),
        execution_id: "utility-wire-execution".into(),
        action: RoventUtilityExecutionAction::Train,
        export_request: RoventUtilityRequiredNullable(None),
        preparation: pin(
            "/metadata-fixture/inputs/preparation.json",
            "preparation-original-raw-file",
        ),
        window_spec: pin(
            "/metadata-fixture/inputs/window.json",
            "window-original-raw-file",
        ),
        trainer_source_sha256: source.trainer_source_sha256.clone(),
        trainer_python_executable: pin("/metadata-fixture/python3", "trainer-python-raw-image"),
        trainer_python_version: "3.12.3".into(),
        dependency_directories: vec!["/metadata-fixture/deps".into()],
        trainer_dependency_set_sha256: rovent_trainer_dependency_set_sha256(&endpoints).unwrap(),
        runtime_endpoints: endpoints,
        current_dataset: RoventUtilityDatasetReference {
            root: "/metadata-fixture/cohort".into(),
            publication: pin(
                "/metadata-fixture/cohort/publication.json",
                "dataset-original-raw-publication",
            ),
            durable_ack: pin(
                "/metadata-fixture/cohort/acknowledgements/original-ack.json",
                "dataset-original-durable-ack-raw-file",
            ),
        },
        ancestor_datasets: vec![],
        initialization_source: RoventUtilityInitializationSource::Initialize {
            utility: pin(
                "/metadata-fixture/inputs/utility.json",
                "initial-utility-original-raw-file",
            ),
        },
        frozen_nnuec_base: pin(
            "/metadata-fixture/inputs/frozen-base.json",
            "frozen-base-original-raw-file",
        ),
        frozen_pc_sources: ["proposer", "critic"]
            .into_iter()
            .map(|role| RoventUtilityFrozenPcSource {
                role: role.into(),
                artifact: pin(
                    &format!("/metadata-fixture/inputs/{role}.onnx"),
                    &format!("frozen-{role}-original-raw-file"),
                ),
                model_semantics_sha256: h(&format!("frozen-{role}-model-semantics")),
                model_identity_sha256: h(&format!("frozen-{role}-model-identity")),
                encoding_sha256: h(&format!("frozen-{role}-encoding")),
            })
            .collect(),
        resources_policy: RoventTrainingResourcePolicy {
            cpu_quota_millicpus: 2000,
            memory_high_bytes: 6 << 30,
            memory_max_bytes: 12 << 30,
            pids_max: 64,
            affinity: vec![0, 2],
            device: "cpu".into(),
        },
        require_total_quota: true,
        output_root: "/metadata-fixture/output".into(),
        control_root: "/metadata-fixture/control".into(),
        checkpoint_directory: "/metadata-fixture/output/checkpoints".into(),
        receipt_path: "/metadata-fixture/output/receipt.json".into(),
        max_rpc_messages: 512,
        max_rpc_bytes: 1 << 20,
        max_read_bytes: 16 << 20,
        max_child_read_bytes: 1 << 20,
        max_output_bytes: 2 << 20,
        max_artifact_files: 256,
    };
    registration
        .join_cohort_window(&window, &preparation)
        .unwrap();
    let cost = RoventUtilityCheckpointCost::from_recorded_parts(
        RoventUtilityRecordedCost {
            forward_calls: 1,
            backward_calls: 2,
            samples: 3,
            wall_time_ns: (1_u64 << 53) + 1,
            partial_accumulation_ns: 5,
            replay_ns: 6,
            masked_groups: 7,
            output_bytes: 8,
        },
        RoventUtilityRecordedCost {
            forward_calls: 11,
            backward_calls: 12,
            samples: 13,
            wall_time_ns: (1_u64 << 53) + 3,
            partial_accumulation_ns: 15,
            replay_ns: 16,
            masked_groups: 17,
            output_bytes: 18,
        },
        RoventUtilityRecordedCost {
            forward_calls: 21,
            backward_calls: 22,
            samples: 23,
            wall_time_ns: 24,
            partial_accumulation_ns: 25,
            replay_ns: 26,
            masked_groups: 27,
            output_bytes: 28,
        },
        RoventUtilityClosureTail::Unknown {
            reason: "original_window_unclosed".into(),
            evidence: None,
        },
        vec![RoventUtilityInheritedUnknownTail {
            checkpoint_publication_sha256: h("ancestor-cost-publication"),
            execution_id: "ancestor-execution".into(),
            window_id: "ancestor-window".into(),
            tail: RoventUtilityClosureTail::Unknown {
                reason: "original_tail_unmeasured".into(),
                evidence: Some(RoventUtilityCostEvidence {
                    path: "receipts/ancestor-tail.json".into(),
                    sha256: h("ancestor-original-tail-raw-file"),
                    bytes: 128,
                }),
            },
        }],
    )
    .unwrap();
    json!({"schema":FIXTURE_SCHEMA,"registration":registration,"preparation":preparation,"window":window,
        "checkpoint_cost":cost,"expected_recipe_sha256":recipe.canonical_sha256().unwrap(),
        "expected_source_registration_sha256":source.canonical_sha256(&recipe,&spec).unwrap(),
        "expected_parameter_spec_sha256":spec.canonical_sha256(&source.frozen_base.config,&recipe.optimizer_owner).unwrap(),
        "expected_preparation_sha256":preparation_sha,"expected_window_sha256":window.canonical_sha256().unwrap(),
        "expected_execution_registration_sha256":registration.canonical_sha256().unwrap(),
        "expected_checkpoint_cost_sha256":generation_digest(ROVENT_UTILITY_CHECKPOINT_COST,&cost).unwrap()})
}

fn decode(
    value: &Value,
) -> crate::rovent_training_process::recovery::metadata_wire_tests::DecoderOutput {
    let helper = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../experiments/model-research/pals/tests/rovent_utility_metadata_wire_decode.py");
    decode_python_fixture(value, &helper)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Accepted {
    schema: String,
    status: String,
    recipe_sha256: String,
    source_registration_sha256: String,
    parameter_spec_sha256: String,
    preparation_sha256: String,
    window_sha256: String,
    execution_registration_sha256: String,
    checkpoint_cost_sha256: String,
    performed_optimizer_updates: u64,
    past_native_authority_reissued: bool,
    execution_authorized: bool,
    neural_imports: bool,
}
fn assert_accepted(fixture: &Value) {
    let output = decode(fixture);
    assert!(
        output.status.success(),
        "F strict utility decoder rejected actual Rust DTOs: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Accepted =
        serde_json::from_slice(&output.stdout).expect("strict complete utility metadata result");
    assert_eq!(result.schema, RESULT_SCHEMA);
    assert_eq!(result.status, "accepted_metadata_only");
    for (name, actual) in [
        ("recipe", result.recipe_sha256),
        ("source_registration", result.source_registration_sha256),
        ("parameter_spec", result.parameter_spec_sha256),
        ("preparation", result.preparation_sha256),
        ("window", result.window_sha256),
        (
            "execution_registration",
            result.execution_registration_sha256,
        ),
        ("checkpoint_cost", result.checkpoint_cost_sha256),
    ] {
        assert_eq!(fixture[format!("expected_{name}_sha256")], actual);
    }
    assert_eq!(result.performed_optimizer_updates, 0);
    assert!(!result.past_native_authority_reissued);
    assert!(!result.execution_authorized);
    assert!(!result.neural_imports);
}
fn has_floating_number(value: &Value) -> bool {
    match value {
        Value::Number(number) => !number.is_u64() && !number.is_i64(),
        Value::Array(values) => values.iter().any(has_floating_number),
        Value::Object(values) => values.values().any(has_floating_number),
        _ => false,
    }
}

#[test]
fn actual_utility_cohort_dtos_have_float_free_independent_canonical_wire() {
    let fixture = fixture();
    let registration = &fixture["registration"];
    let preparation = &fixture["preparation"];
    assert!(!has_floating_number(&fixture));
    assert_eq!(fixture["window"].as_object().unwrap().len(), 22);
    assert_eq!(registration.as_object().unwrap().len(), 29);
    assert_eq!(registration.get("export_request"), Some(&Value::Null));
    assert!(registration["current_dataset"].get("durable_ack").is_some());
    assert_eq!(
        fixture["checkpoint_cost"]["closure_tail"].get("evidence"),
        Some(&Value::Null)
    );
    assert_eq!(
        fixture["checkpoint_cost"]["ancestor_cost"]["wall_time_ns"].as_u64(),
        Some((1_u64 << 53) + 1)
    );
    assert_ne!(
        registration["frozen_nnuec_base"]["sha256"],
        preparation["registration"]["frozen_base"]["weights_sha256"]
    );
    assert_ne!(
        registration["preparation"]["sha256"],
        preparation["preparation_sha256"]
    );
    assert_ne!(
        registration["current_dataset"]["publication"]["sha256"],
        fixture["window"]["dataset_sha256"]
    );
    assert_ne!(
        registration["window_spec"]["sha256"],
        fixture["expected_window_sha256"]
    );
    assert_ne!(
        preparation["registration"]["collection_source_sha256"],
        preparation["registration"]["trainer_source_sha256"]
    );
    assert_accepted(&fixture);
}

#[test]
fn independent_utility_decoder_rejects_omitted_required_nulls() {
    let original = fixture();
    let mut missing_export = original.clone();
    missing_export["registration"]
        .as_object_mut()
        .unwrap()
        .remove("export_request");
    assert!(
        !decode(&missing_export).status.success(),
        "absent export_request silently became null"
    );
    let mut missing_unknown_evidence = original.clone();
    missing_unknown_evidence["checkpoint_cost"]["closure_tail"]
        .as_object_mut()
        .unwrap()
        .remove("evidence");
    assert!(
        !decode(&missing_unknown_evidence).status.success(),
        "absent unknown evidence silently became null"
    );
    let mut missing_durable_ack = original;
    missing_durable_ack["registration"]["current_dataset"]
        .as_object_mut()
        .unwrap()
        .remove("durable_ack");
    assert!(
        !decode(&missing_durable_ack).status.success(),
        "missing original dataset durable acknowledgement was ignored"
    );
}

#[test]
fn independent_utility_decoder_rejects_v1_v2_preparation_scope_mixing() {
    let original = fixture();
    let mut retagged = original.clone();
    retagged["preparation"]["schema"] = json!(ROVENT_UTILITY_PREPARATION_DOMAIN);
    assert!(
        !decode(&retagged).status.success(),
        "cohort preparation silently retagged to V1"
    );
    let mut root_local_cost = original;
    root_local_cost["preparation"]["registration"]["cost_scope"] = json!(ROVENT_UTILITY_COST_SCOPE);
    assert!(
        !decode(&root_local_cost).status.success(),
        "V1 route cost entered cohort V2 preparation"
    );
}

#[test]
fn independent_utility_decoder_rejects_unknown_fields_booleans_and_numeric_coefficients() {
    let original = fixture();
    let mut unknown = original.clone();
    unknown["registration"]["source_authority_verified"] = json!(true);
    assert!(
        !decode(&unknown).status.success(),
        "unknown execution success field was ignored"
    );
    let mut boolean = original.clone();
    boolean["window"]["max_optimizer_updates"] = json!(true);
    assert!(
        !decode(&boolean).status.success(),
        "boolean became an optimizer count"
    );
    let mut coefficient = original;
    coefficient["preparation"]["recipe"]["optimizer"]["learning_rate"] = json!(0.0001);
    assert!(
        !decode(&coefficient).status.success(),
        "a float replaced the exact decimal string recipe"
    );
}

#[test]
fn independent_utility_decoder_rejects_raw_canonical_alias_and_foreign_trainer_join() {
    let original = fixture();
    let mut raw_alias = original.clone();
    raw_alias["window"]["preparation_sha256"] =
        raw_alias["registration"]["preparation"]["sha256"].clone();
    assert!(
        !decode(&raw_alias).status.success(),
        "raw preparation file SHA replaced its canonical metadata seal"
    );
    let mut trainer = original;
    trainer["registration"]["trainer_source_sha256"] = json!(h("different-trainer-source"));
    assert!(
        !decode(&trainer).status.success(),
        "individually valid envelopes silently used different trainer sources"
    );
}
