//! Actual Rust DTO serialization -> independent Python strict metadata decoder.
//! These are schema-only operands: no advertised artifact exists and no
//! tensor, historical native/physical authority or execution lease is issued.
//! Run with an explicit absolute RZ_ROVENT_METADATA_PYTHON interpreter. Missing
//! configuration is a visible test failure, never an ignored/PATH fallback.

use super::*;
use rz_experiments::{ArtifactRef, RoventNnuecPin};
use rz_search::cpu_value::nnuec::v2 as nnuec_v2;
use sha2::{Digest as _, Sha256};
use std::{
    io::{Read, Write},
    process::{Child, Command, ExitStatus, Stdio},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    thread,
    time::{Duration, Instant},
};

const FIXTURE_SCHEMA: &str = "rz-rovent-metadata-cross-language-fixture/1";
const RESULT_SCHEMA: &str = "rz-rovent-metadata-cross-language-result/1";
const MAX_PIPE_BYTES: usize = 64 << 10;
const CHILD_WALL: Duration = Duration::from_secs(30);

fn h(label: &str) -> String {
    format!("{:x}", Sha256::digest(label.as_bytes()))
}
fn pin(path: &str, label: &str) -> RoventTrainingFilePin {
    RoventTrainingFilePin {
        path: PathBuf::from(path),
        sha256: h(label),
        bytes: 128,
    }
}
fn relative_pin(path: &str, label: &str) -> RoventPublicationFile {
    RoventPublicationFile {
        path: path.into(),
        sha256: h(label),
        bytes: 128,
    }
}
fn roles(proposer: u64, critic: u64, validator: u64) -> BTreeMap<String, u64> {
    [
        ("proposer".into(), proposer),
        ("critic".into(), critic),
        ("validator".into(), validator),
    ]
    .into()
}

fn recipe(auxiliary_enabled: bool) -> Value {
    let actors = ["stockfish", "reckless"]
        .into_iter()
        .enumerate()
        .map(|(slot, family)| {
            json!({"actor_id":family,"family":family,
                "source_revision":if slot==0 {"1".repeat(40)}else{"2".repeat(40)},
                "source_sha256":h(&format!("{family}-source")),
                "binary_sha256":h(&format!("{family}-binary")),
                "profile_sha256":h(&format!("{family}-profile")),
                "options_sha256":h(&format!("{family}-options")),
                "assets_sha256":h(&format!("{family}-assets")),
                "rights_sha256":h(&format!("{family}-rights"))})
        })
        .collect::<Vec<_>>();
    let stage = |id: &str, kind: &str, parent: Option<&str>, teachers: &[&str]| {
        json!({"stage_id":id,"kind":kind,"parent_stage_id":parent,
            "teacher_actor_ids":teachers,"round_id":format!("{id}-round"),
            "role_updates":roles(2,2,0)})
    };
    json!({"schema":"rz-rovent-generation-series/1","series_id":"wire-series",
        "student_id":"one-student","learner_id":"one-learner",
        "source_sha256":h("collection-source"),"model_semantics_sha256":h("model-semantics"),
        "actors":actors,"parameter_ownership":{"pc_shared":"pc","p_private":"pc","c_private":"pc",
            "v_private":"v","nnuec":if auxiliary_enabled {"nnuec-private"} else {"pc"},
            "membership_sha256":h("membership")},
        "stages":[stage("common","common_student",None,&["stockfish","reckless"]),
            stage("stockfish-ft","teacher_finetune",Some("common"),&["stockfish"]),
            stage("reckless-ft","teacher_finetune",Some("common"),&["reckless"])],
        "opening_splits":{"train":"train","valid":"validation","held":"holdout"},
        "global_holdout_openings":["held"]})
}

/// Constructors use the real Rust types. No serde Value -> DTO conversion can
/// conceal a renamed/added field or the enum's actual wire representation.
fn fixture_with_training_epoch(training_model_epoch: u64) -> Value {
    fixture_with_auxiliary(
        training_model_epoch,
        RoventRAuxiliaryPresenceState::Absent,
        1e-8,
    )
}

fn original_nnuec_pin() -> RoventNnuecPin {
    let artifact = |path: &str, label: &str| ArtifactRef {
        path: path.into(),
        sha256: h(label),
        bytes: 128,
        source: "https://github.com/daejunnom/RoveZero".into(),
        license: "MIT".into(),
    };
    RoventNnuecPin {
        schema: nnuec_v2::SCHEMA.into(),
        semantics: nnuec_v2::SEMANTICS.into(),
        encoding: nnuec_v2::INPUT_SCHEMA.into(),
        implementation_sha256: h("nnuec-implementation"),
        configuration_sha256: h("nnuec-configuration"),
        weights: artifact("nnuec/weights.bin", "nnuec-original-raw-weights"),
        provenance: artifact("nnuec/provenance.json", "nnuec-original-provenance"),
        diagnostic: true,
        context_namespace_policy_sha256: h("nnuec-context-namespace-policy"),
        local_namespace_policy_sha256: h("nnuec-local-namespace-policy"),
        search_conditions_sha256: h("nnuec-search-conditions"),
    }
}

/// Every optional field is built in its real type. No post-serialization
/// insertion can hide an absent/null alias or synthesize a default R object.
fn fixture_with_auxiliary(
    training_model_epoch: u64,
    auxiliary_presence: RoventRAuxiliaryPresenceState,
    auxiliary_weight: f64,
) -> Value {
    let recipe = recipe(auxiliary_presence == RoventRAuxiliaryPresenceState::Enabled);
    let recipe_sha256 = generation_digest("rz-rovent-generation-series/1", &recipe).unwrap();
    let runtime_endpoints = vec![
        RoventTrainerRuntimeEndpoint {
            name: "torch.__file__".into(),
            artifact: pin(
                "/metadata-fixture/site/torch/__init__.py",
                "torch-python-endpoint",
            ),
        },
        RoventTrainerRuntimeEndpoint {
            name: "torch._C.__file__".into(),
            artifact: pin(
                "/metadata-fixture/site/torch/_C.so",
                "torch-native-endpoint",
            ),
        },
    ];
    let dependency_sha = rovent_trainer_dependency_set_sha256(&runtime_endpoints).unwrap();
    let window = RoventTrainWindowV2 {
        schema: ROVENT_TRAIN_WINDOW_V2.into(),
        window_id: "wire-window".into(),
        recipe_sha256: recipe_sha256.clone(),
        stage_id: "common".into(),
        collection_source_sha256: h("collection-source"),
        collection_runtime_library_sha256: h("collection-ort-library"),
        collection_model_semantics_sha256: h("model-semantics"),
        collection_model_epoch: (1_u64 << 53) + 1,
        collection_context_epoch: (1_u64 << 53) + 3,
        trainer_source_sha256: h("trainer-source"),
        trainer_python_executable_sha256: h("trainer-python-executable"),
        trainer_dependency_set_sha256: dependency_sha.clone(),
        trainer_dependency_scope: ROVENT_TRAINER_DEPENDENCY_SCOPE.into(),
        training_model_semantics_sha256: h("model-semantics"),
        training_model_epoch,
        nnuec_collection_model_sha256: h("nnuec-collection-model"),
        nnuec_semantics: nnuec_v2::SEMANTICS.into(),
        phase: "pc".into(),
        frozen_pc_sha256: None,
        device: "cpu".into(),
        precision: "fp32".into(),
        tf32: false,
        deterministic: true,
        cpu_threads: 2,
        interop_threads: 1,
        microbatch: 1,
        accumulation: 8,
        optimizer: json!({"name":"adamw","lr":1e-4,"betas":[0.9,0.999],"epsilon":1e-8,
            "weight_decay":0.01,"clip_norm":1.0,"schedule":"constant"}),
        sampling_seed: 29,
        training_seed: 31,
        loss_weights: HEADS[..4]
            .iter()
            .map(|head| ((*head).to_owned(), 1.0))
            .collect(),
        nnuec_loss: RoventNnuecLoss {
            loss: "wdl_cross_entropy".into(),
            target: "same_exact_rules_anchor".into(),
            normalization: "valid_rows".into(),
            weight: 0.5,
        },
        nnuec_r_auxiliary: match auxiliary_presence {
            RoventRAuxiliaryPresenceState::Absent => RoventRAuxiliaryPresence::Absent,
            RoventRAuxiliaryPresenceState::Null => RoventRAuxiliaryPresence::Null,
            RoventRAuxiliaryPresenceState::Enabled => {
                RoventRAuxiliaryPresence::Enabled(RoventRAuxiliaryRecipe {
                    schema: ROVENT_R_AUXILIARY_RECIPE.into(),
                    manifest_file_sha256: h("r-manifest-raw-file"),
                    index_file_sha256: h("r-index-raw-file"),
                    sampling_seed: 37,
                    rows_per_pc_update: 64,
                    weight: auxiliary_weight,
                    normalization: ROVENT_R_AUXILIARY_NORMALIZATION.into(),
                    optimizer_owner: "nnuec-private".into(),
                    supervision_source: ROVENT_R_AUXILIARY_SUPERVISION.into(),
                })
            }
        },
        max_updates: 4,
        max_groups: 4,
        max_samples: 32,
        max_wall_ns: 1_000_000_000,
        max_output_bytes: 1 << 20,
        checkpoint_interval: 4,
    };
    window.validate().unwrap();
    let profile_sha = window.profile_sha256().unwrap();
    let dataset = RoventHistoricalDatasetRegistration {
        root: "/metadata-fixture/dataset".into(),
        publication: pin(
            "/metadata-fixture/dataset/publication.json",
            "dataset-publication",
        ),
        collection_source_commit: "a".repeat(40),
        collection_source_sha256: window.collection_source_sha256.clone(),
        collection_runtime_library_sha256: window.collection_runtime_library_sha256.clone(),
        model_semantics_sha256: window.collection_model_semantics_sha256.clone(),
        model_epoch: window.collection_model_epoch,
        context_epoch: window.collection_context_epoch,
        recipe_sha256: recipe_sha256.clone(),
        stage_id: window.stage_id.clone(),
        role_bindings: ROLES
            .iter()
            .map(|role| {
                (
                    (*role).into(),
                    RoventHistoricalRoleBinding {
                        model_semantics_sha256: h("model-semantics"),
                        model_identity_sha256: h(&format!("collection-{role}-ticket-model")),
                        weights_sha256: h(&format!("collection-{role}-raw-weights")),
                        encoding_sha256: h(&format!("collection-{role}-encoding")),
                    },
                )
            })
            .collect(),
    };
    let registration = RoventRecoveredTrainingRegistration {
        schema: ROVENT_RECOVERY_REGISTRATION.into(),
        execution_id: "wire-new-execution".into(),
        action: RoventRecoveredTrainingAction::Train,
        export_request: None,
        recipe: pin("/metadata-fixture/recipe.json", "recipe-raw-bytes"),
        membership: pin("/metadata-fixture/membership.json", "membership"),
        window_spec: pin("/metadata-fixture/window.json", "window-raw-bytes"),
        trainer_source_sha256: window.trainer_source_sha256.clone(),
        trainer_python_executable: pin("/metadata-fixture/python3", "trainer-python-executable"),
        trainer_python_version: "3.12.3".into(),
        dependency_directories: vec!["/metadata-fixture/site".into()],
        trainer_dependency_set_sha256: dependency_sha,
        runtime_endpoints,
        current_dataset: dataset,
        ancestor_datasets: vec![],
        r_auxiliary_dataset: match auxiliary_presence {
            RoventRAuxiliaryPresenceState::Absent => RoventRAuxiliaryPresence::Absent,
            RoventRAuxiliaryPresenceState::Null => RoventRAuxiliaryPresence::Null,
            RoventRAuxiliaryPresenceState::Enabled => {
                RoventRAuxiliaryPresence::Enabled(RoventHistoricalRAuxiliaryDatasetRegistration {
                    root: "/metadata-fixture/r-dataset".into(),
                    publication: pin(
                        "/metadata-fixture/r-dataset/r-publication.json",
                        "r-dataset-publication",
                    ),
                    collection_source_commit: "a".repeat(40),
                    collection_source_sha256: window.collection_source_sha256.clone(),
                    collection_runtime_library_sha256: window
                        .collection_runtime_library_sha256
                        .clone(),
                    collection_model_semantics_sha256: window
                        .collection_model_semantics_sha256
                        .clone(),
                    collection_model_epoch: window.collection_model_epoch,
                    collection_context_epoch: window.collection_context_epoch,
                    collection_recipe_sha256: window.recipe_sha256.clone(),
                    collection_stage_id: window.stage_id.clone(),
                    nnuec: original_nnuec_pin(),
                    nnuec_parameter_sha256: window.nnuec_collection_model_sha256.clone(),
                })
            }
        },
        ancestor_r_auxiliary_datasets: RoventRAuxiliaryPresence::Absent,
        initialization_source: RoventTrainingInputSource::Resume {
            checkpoint: RoventHistoricalCheckpointRegistration {
                root: "/metadata-fixture/checkpoint".into(),
                publication: pin(
                    "/metadata-fixture/checkpoint/publication.json",
                    "checkpoint-publication",
                ),
                reclamation: pin("/metadata-fixture/old-reclamation.json", "old-reclamation"),
            },
        },
        rules_profile: pin("/metadata-fixture/rules.json", "rules-profile"),
        expected_initial_completed: roles(1, 0, 0),
        expected_ancestor_completed: roles(0, 0, 0),
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
        max_rpc_messages: 1024,
        max_rpc_bytes: 1 << 20,
        max_read_bytes: 16 << 20,
        max_child_read_bytes: 8 << 20,
        max_output_bytes: 1 << 20,
        max_artifact_files: 64,
    };
    registration.join_window(&window).unwrap();
    let publication = RoventCheckpointPublication {
        schema: ROVENT_CHECKPOINT_PUBLICATION.into(),
        execution_id: "wire-old-execution".into(),
        window_id: window.window_id.clone(),
        collection_source_sha256: window.collection_source_sha256.clone(),
        collection_runtime_library_sha256: window.collection_runtime_library_sha256.clone(),
        trainer_source_sha256: window.trainer_source_sha256.clone(),
        trainer_python_executable_sha256: window.trainer_python_executable_sha256.clone(),
        trainer_dependency_set_sha256: window.trainer_dependency_set_sha256.clone(),
        training_model_semantics_sha256: window.training_model_semantics_sha256.clone(),
        initialization_source: RoventCheckpointInitializationSource::Initialize {
            pc_checkpoint_sha256: h("initial-pc-checkpoint"),
            nnuec_checkpoint_sha256: h("initial-nnuec-checkpoint"),
        },
        collection_model_epoch: window.collection_model_epoch,
        collection_context_epoch: window.collection_context_epoch,
        training_model_epoch: window.training_model_epoch,
        recipe_sha256,
        stage_id: window.stage_id.clone(),
        training_profile_sha256: profile_sha.clone(),
        membership_sha256: registration.membership.sha256.clone(),
        current_dataset_publication_sha256: registration.current_dataset.publication.sha256.clone(),
        ancestor_dataset_publication_sha256: vec![],
        tensor: relative_pin("tensors/checkpoint.pt", "tensor"),
        checkpoint_retirement: relative_pin("receipts/checkpoint.json", "checkpoint-retirement"),
        update_receipts: vec![relative_pin("receipts/update-1.json", "update-1")],
        terminal_receipt: relative_pin("receipts/terminal.json", "terminal"),
        original_recipe: relative_pin("inputs/recipe.json", "recipe-raw-bytes"),
        original_membership: relative_pin("inputs/membership.json", "membership"),
        original_window_spec: relative_pin("inputs/window.json", "window-raw-bytes"),
        parent_publication: None,
        r_auxiliary: match auxiliary_presence {
            RoventRAuxiliaryPresenceState::Absent => RoventRAuxiliaryPresence::Absent,
            RoventRAuxiliaryPresenceState::Null => RoventRAuxiliaryPresence::Null,
            RoventRAuxiliaryPresenceState::Enabled => {
                let source = registration.r_auxiliary_dataset.enabled().unwrap();
                let auxiliary = window.nnuec_r_auxiliary.enabled().unwrap();
                RoventRAuxiliaryPresence::Enabled(RoventCheckpointRAuxiliary {
                    schema: ROVENT_CHECKPOINT_R_AUXILIARY.into(),
                    dataset_publication_sha256: source.publication.sha256.clone(),
                    manifest_file_sha256: auxiliary.manifest_file_sha256.clone(),
                    index_file_sha256: auxiliary.index_file_sha256.clone(),
                    // Shape-only pinned metadata, not a claim that an old
                    // ledger, native lease or physical completion exists.
                    parent_ledger_projection: relative_pin(
                        "receipts/r-ledger-projection-shape.json",
                        "r-ledger-projection-shape-only",
                    ),
                })
            }
        },
        boundary: RoventCheckpointBoundary {
            completed_updates: roles(1, 0, 0),
            ancestor_updates: roles(0, 0, 0),
            model_state_sha256: h("model-state"),
            optimizer_state_sha256: h("optimizer-state"),
            scheduler_state_sha256: h("scheduler-state"),
            rng_state_sha256: h("rng-state"),
            sampler_state_sha256: h("sampler-state"),
            retained_input_set_sha256: h("input-set"),
            retained_input_count: 8,
            retained_origin_set_sha256: h("origin-set"),
            retained_origin_count: 8,
            snapshot_cost: COST_FIELDS
                .iter()
                .map(|field| ((*field).into(), if *field == "samples" { 8 } else { 0 }))
                .collect(),
        },
        recorded_tail_cost: RoventRecordedTailCost::Unknown {
            reason: "schema_fixture_has_no_observed_tail".into(),
            evidence: None,
        },
        recorded_closure: RoventRecordedClosure {
            process_scope: "schema_fixture_not_old_run_closure".into(),
            physical_scope: "unknown_no_physical_completion_capability".into(),
            evidence: relative_pin("receipts/closure-shape.json", "closure-shape-only"),
        },
    };
    publication.validate().unwrap();
    assert_ne!(
        window.collection_runtime_library_sha256,
        window.trainer_python_executable_sha256
    );
    assert_ne!(
        window.collection_source_sha256,
        window.trainer_source_sha256
    );
    assert_ne!(window.collection_model_epoch, window.training_model_epoch);
    json!({"schema":FIXTURE_SCHEMA,
        "registration":serde_json::to_value(registration).unwrap(),
        "window":serde_json::to_value(window).unwrap(),
        "checkpoint_publication":serde_json::to_value(publication).unwrap(),
        "recipe":recipe,"expected_profile_sha256":profile_sha})
}

fn fixture() -> Value {
    fixture_with_training_epoch((1_u64 << 53) + 5)
}

struct OwnedDecoder(Child);
impl Drop for OwnedDecoder {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}
pub(crate) struct DecoderOutput {
    pub(crate) status: ExitStatus,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
}
fn bounded_read(
    mut pipe: impl Read,
    overrun: Arc<AtomicBool>,
    combined_bytes: Arc<AtomicUsize>,
) -> Vec<u8> {
    let mut output = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let count = pipe.read(&mut buffer).expect("metadata decoder pipe read");
        if count == 0 {
            return output;
        }
        if combined_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |bytes| {
                bytes
                    .checked_add(count)
                    .filter(|size| *size <= MAX_PIPE_BYTES)
            })
            .is_err()
        {
            overrun.store(true, Ordering::Release);
            return output;
        }
        output.extend_from_slice(&buffer[..count]);
    }
}
fn decode_in_python(value: &Value) -> DecoderOutput {
    let helper = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../experiments/model-research/pals/tests/rovent_metadata_wire_decode.py");
    decode_python_fixture(value, &helper)
}
pub(crate) fn decode_python_fixture(value: &Value, helper: &Path) -> DecoderOutput {
    assert!(
        helper.is_absolute(),
        "metadata helper must have an explicit absolute path"
    );
    let python = std::env::var_os("RZ_ROVENT_METADATA_PYTHON")
        .map(PathBuf::from)
        .expect("set RZ_ROVENT_METADATA_PYTHON to the exact absolute stdlib-only decoder interpreter; this cross-language test is not ignored");
    assert!(
        python.is_absolute(),
        "decoder interpreter must be absolute; no PATH fallback"
    );
    let input = serde_json::to_vec(value).unwrap();
    assert!(
        input.len() <= MAX_PIPE_BYTES,
        "bounded metadata stdin fixture"
    );
    let started = Instant::now();
    let mut child = OwnedDecoder(
        Command::new(python)
            .args(["-I", "-S", "-B"])
            .arg(helper)
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start the explicitly configured metadata-only Python decoder"),
    );
    let mut stdin = child.0.stdin.take().unwrap();
    let writer = thread::spawn(move || stdin.write_all(&input));
    let overrun = Arc::new(AtomicBool::new(false));
    let combined_bytes = Arc::new(AtomicUsize::new(0));
    let stdout = child.0.stdout.take().unwrap();
    let stderr = child.0.stderr.take().unwrap();
    let out_flag = Arc::clone(&overrun);
    let err_flag = Arc::clone(&overrun);
    let out_bytes = Arc::clone(&combined_bytes);
    let stdout = thread::spawn(move || bounded_read(stdout, out_flag, out_bytes));
    let stderr = thread::spawn(move || bounded_read(stderr, err_flag, combined_bytes));
    let mut bounded_failure = None;
    let status = loop {
        if let Some(status) = child.0.try_wait().expect("observe owned decoder status") {
            break status;
        }
        if started.elapsed() >= CHILD_WALL || overrun.load(Ordering::Acquire) {
            bounded_failure = Some("metadata decoder exceeded its 30s/64KiB bound");
            child.0.kill().expect("stop owned metadata decoder");
            break child.0.wait().expect("reap owned metadata decoder");
        }
        thread::sleep(Duration::from_millis(10));
    };
    let write_result = writer.join().expect("metadata stdin writer joined");
    let stdout = stdout.join().expect("metadata stdout reader joined");
    let stderr = stderr.join().expect("metadata stderr reader joined");
    assert!(
        bounded_failure.is_none(),
        "{}",
        bounded_failure.unwrap_or("")
    );
    assert!(
        !overrun.load(Ordering::Acquire),
        "metadata decoder output exceeded 64KiB"
    );
    if status.success() {
        write_result.expect("accepted decoder consumed the complete Rust wire fixture");
    }
    DecoderOutput {
        status,
        stdout,
        stderr,
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AcceptedMetadata {
    schema: String,
    status: String,
    profile_sha256: String,
    past_native_authority_reissued: bool,
    execution_authorized: bool,
    neural_imports: bool,
}

fn assert_python_accepts_fixture(fixture: Value) {
    let output = decode_in_python(&fixture);
    assert!(
        output.status.success(),
        "metadata decoder rejected actual Rust serialization: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: AcceptedMetadata = serde_json::from_slice(&output.stdout)
        .expect("strict complete metadata-only Python acceptance record");
    assert_eq!(result.schema, RESULT_SCHEMA);
    assert_eq!(result.status, "accepted");
    assert_eq!(
        result.profile_sha256,
        fixture["expected_profile_sha256"].as_str().unwrap()
    );
    assert!(!result.past_native_authority_reissued);
    assert!(!result.execution_authorized);
    assert!(!result.neural_imports);
}

#[test]
fn actual_typed_rust_metadata_is_accepted_by_independent_python_decoder() {
    assert_python_accepts_fixture(fixture());
}

#[test]
fn ipc_check_registration_round_trips_through_python_without_training_authority() {
    let mut value = fixture();
    let mut registration: RoventRecoveredTrainingRegistration =
        serde_json::from_value(value["registration"].clone()).unwrap();
    registration.action = RoventRecoveredTrainingAction::IpcCheck;
    registration.validate().unwrap();
    value["registration"] = serde_json::to_value(registration).unwrap();
    assert_eq!(value["registration"]["action"], "ipc_check");
    assert_python_accepts_fixture(value);
}

#[test]
fn typed_zero_training_epoch_preserves_positive_collection_epochs() {
    // Zero is an original training/output epoch, not an absent value or a
    // replacement for the positive epoch of the sealed collection input.
    // Both typed window/publication constructors receive the same exact u64.
    let fixture = fixture_with_training_epoch(0);
    assert_eq!(fixture["window"]["training_model_epoch"], 0);
    assert_eq!(fixture["checkpoint_publication"]["training_model_epoch"], 0);
    assert!(
        fixture["window"]["collection_model_epoch"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        fixture["window"]["collection_context_epoch"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_python_accepts_fixture(fixture);
}

#[test]
fn typed_r_auxiliary_states_preserve_presence_and_independent_profile_digests() {
    let mut profiles = BTreeSet::new();
    for (offset, presence) in [
        RoventRAuxiliaryPresenceState::Absent,
        RoventRAuxiliaryPresenceState::Null,
        RoventRAuxiliaryPresenceState::Enabled,
    ]
    .into_iter()
    .enumerate()
    {
        let epoch = (1_u64 << 53) + 5 + (offset as u64 * 2);
        let fixture = fixture_with_auxiliary(epoch, presence, 1e-8);
        let window = &fixture["window"];
        let registration = &fixture["registration"];
        let publication = &fixture["checkpoint_publication"];
        assert_eq!(window["training_model_epoch"].as_u64(), Some(epoch));
        assert_eq!(publication["training_model_epoch"].as_u64(), Some(epoch));
        assert_eq!(
            window["collection_context_epoch"].as_u64(),
            Some((1_u64 << 53) + 3)
        );
        assert!(registration.get("ancestor_r_auxiliary_datasets").is_none());
        match presence {
            RoventRAuxiliaryPresenceState::Absent => {
                assert!(window.get("nnuec_r_auxiliary").is_none());
                assert!(registration.get("r_auxiliary_dataset").is_none());
                assert!(publication.get("r_auxiliary").is_none());
                assert_eq!(fixture["recipe"]["parameter_ownership"]["nnuec"], "pc");
            }
            RoventRAuxiliaryPresenceState::Null => {
                assert_eq!(window.get("nnuec_r_auxiliary"), Some(&Value::Null));
                assert_eq!(registration.get("r_auxiliary_dataset"), Some(&Value::Null));
                assert_eq!(publication.get("r_auxiliary"), Some(&Value::Null));
                assert_eq!(fixture["recipe"]["parameter_ownership"]["nnuec"], "pc");
            }
            RoventRAuxiliaryPresenceState::Enabled => {
                let auxiliary = &window["nnuec_r_auxiliary"];
                let dataset = &registration["r_auxiliary_dataset"];
                let sidecar = &publication["r_auxiliary"];
                assert_eq!(
                    fixture["recipe"]["parameter_ownership"]["nnuec"],
                    "nnuec-private"
                );
                assert_eq!(auxiliary["optimizer_owner"], "nnuec-private");
                assert_eq!(auxiliary["rows_per_pc_update"], 64);
                assert_eq!(
                    auxiliary["weight"].as_f64().unwrap().to_bits(),
                    1e-8_f64.to_bits()
                );
                assert_eq!(dataset["nnuec"]["schema"], nnuec_v2::SCHEMA);
                assert_eq!(dataset["nnuec"]["semantics"], nnuec_v2::SEMANTICS);
                assert_eq!(dataset["nnuec"]["encoding"], nnuec_v2::INPUT_SCHEMA);
                assert_ne!(
                    dataset["nnuec_parameter_sha256"],
                    dataset["nnuec"]["weights"]["sha256"]
                );
                assert_eq!(
                    sidecar["dataset_publication_sha256"],
                    dataset["publication"]["sha256"]
                );
                for name in ["manifest_file_sha256", "index_file_sha256"] {
                    assert_eq!(sidecar[name], auxiliary[name]);
                }
            }
        }
        assert!(
            profiles.insert(
                fixture["expected_profile_sha256"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            )
        );
        assert_python_accepts_fixture(fixture);
    }
    // Training epoch is excluded from the fixed profile. Thus the distinction
    // between Absent and Null here comes from actual optional-field presence.
    assert_eq!(profiles.len(), 3);
}

#[test]
fn typed_enabled_r_weight_preserves_adjacent_f64_profile_coefficients() {
    let first_weight = 1e-8_f64;
    let second_weight = f64::from_bits(first_weight.to_bits() + 1);
    let first = fixture_with_auxiliary(
        (1_u64 << 53) + 13,
        RoventRAuxiliaryPresenceState::Enabled,
        first_weight,
    );
    let second = fixture_with_auxiliary(
        (1_u64 << 53) + 13,
        RoventRAuxiliaryPresenceState::Enabled,
        second_weight,
    );
    assert_ne!(first_weight.to_bits(), second_weight.to_bits());
    assert_ne!(
        first["expected_profile_sha256"],
        second["expected_profile_sha256"]
    );
    for (fixture, weight) in [(first, first_weight), (second, second_weight)] {
        assert_eq!(
            fixture["window"]["nnuec_r_auxiliary"]["weight"]
                .as_f64()
                .unwrap()
                .to_bits(),
            weight.to_bits()
        );
        assert_python_accepts_fixture(fixture);
    }
}

#[test]
fn independent_python_decoder_rejects_r_presence_and_original_source_substitution() {
    let original = fixture_with_auxiliary(
        (1_u64 << 53) + 17,
        RoventRAuxiliaryPresenceState::Enabled,
        0.125,
    );
    let mut missing_owner = original.clone();
    missing_owner["registration"]["r_auxiliary_dataset"] = Value::Null;
    assert!(
        !decode_in_python(&missing_owner).status.success(),
        "enabled R window silently acquired a null current dataset"
    );
    let mut other_manifest = original.clone();
    other_manifest["checkpoint_publication"]["r_auxiliary"]["manifest_file_sha256"] =
        json!(h("different-r-manifest-raw-file"));
    assert!(
        !decode_in_python(&other_manifest).status.success(),
        "checkpoint sidecar silently substituted the original raw R manifest"
    );
    let mut other_dataset = original;
    other_dataset["checkpoint_publication"]["r_auxiliary"]["dataset_publication_sha256"] =
        json!(h("different-r-dataset-publication"));
    assert!(
        !decode_in_python(&other_dataset).status.success(),
        "checkpoint sidecar silently substituted the registered original R dataset"
    );
}

#[test]
fn independent_python_decoder_rejects_unknown_fields_and_boolean_epochs() {
    let original = fixture();
    let mut unknown = original.clone();
    unknown["registration"]["unexpected_wire_field"] = json!("must-not-be-ignored");
    assert!(
        !decode_in_python(&unknown).status.success(),
        "Python decoder silently accepted an unknown Rust metadata field"
    );
    let mut boolean = original;
    boolean["window"]["collection_model_epoch"] = json!(true);
    assert!(
        !decode_in_python(&boolean).status.success(),
        "Python decoder confused a boolean with an integer epoch"
    );
}
