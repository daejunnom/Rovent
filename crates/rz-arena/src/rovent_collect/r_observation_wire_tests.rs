//! 실제 Rust DTO serde와 Python 순수 역사 검증의 형식 대조다.
//! 합성 snapshot은 원 파일·Rules·native/물리 완료·학습 권한을 발급하지 않는다.
//! 실제 Collection owner/cap 인수는 이 검사와 별도로 수행한다.
#![cfg(feature = "pals-collection-onnx")]

use super::{fixture as base_fixture, h};
use crate::rovent_collect::{
    ROVENT_R_PUBLICATION_DOMAIN, ROVENT_R_SEMANTIC_OBSERVATIONS, ROVENT_R_SEMANTIC_TASK,
    ROVENT_R_UNSUBMITTED_OBSERVATIONS, RoventPublicationFile, RoventRCorpusPublication,
    RoventRPublicationAdmission, RoventRPublicationRetirement, RoventRSemanticObservation,
    RoventRSemanticObservations, RoventRSemanticRetirement, RoventRSemanticSelectedPc,
    RoventRSemanticStage, RoventRSemanticWork, RoventRUnsubmittedKind,
    RoventRUnsubmittedObservation, RoventRUnsubmittedObservations,
};
use rz_experiments::pals_data_v3::generation_digest;
use rz_experiments::rovent_r_readout::{
    R_READOUT_RETIREMENT_DOMAIN, RoventRCaptureEnding, RoventRCaptureLimits,
    RoventRCaptureRetirement,
};
use rz_search::cpu_value::{CpuTrainingState, CpuValueIdentity, nnuec::v2};
use rz_search::pals::store::{StateId, TaskKey, TaskQuestion};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::path::Path;

const FIXTURE_SCHEMA: &str = "rz-rovent-r-observation-wire-fixture/1";
const RESULT_SCHEMA: &str = "rz-rovent-r-observation-wire-result/1";

fn pin(path: String, label: &str) -> RoventPublicationFile {
    // These advertised one-byte payload files do not exist. Only the metadata
    // and the actual publication JSON bytes below participate in this test.
    RoventPublicationFile {
        path,
        sha256: h(label),
        bytes: 1,
    }
}

fn fixture(question: TaskQuestion) -> Value {
    let base = base_fixture();
    let original = &base["records"][0];
    // The old retirement was already constructed and serialized by the parent
    // typed fixture. Every new supplement DTO below is a struct literal.
    let terminal: RoventRCaptureRetirement =
        serde_json::from_value(original["retirement"].clone()).unwrap();
    let terminal_sha = generation_digest(R_READOUT_RETIREMENT_DOMAIN, &terminal).unwrap();
    assert_eq!(original["retirement_sha256"], terminal_sha);
    let mut not_started_job = terminal.job.clone();
    not_started_job.job_sequence = 2;
    not_started_job.question_sha256 = h("wire-not-started-question");
    let not_started = RoventRCaptureRetirement {
        job: not_started_job.clone(),
        worker_owner: 0,
        frame_sequence: 2,
        capture_sequence: 2,
        ending: RoventRCaptureEnding::NotStarted,
        limits: RoventRCaptureLimits {
            maximum_readouts: 64,
            every_nth_score: 2,
            maximum_input_bytes: 1048576,
        },
        attempted: 0,
        selected: 0,
        retained_readouts: 0,
        overflow: 0,
        retained_input_bytes: 0,
        selected_input_preparation_ns: 0,
        error: None,
    };
    let not_started_sha = generation_digest(R_READOUT_RETIREMENT_DOMAIN, &not_started).unwrap();
    let authority = terminal.job.authority.clone();
    let admission_sha = original["native_admission_sha256"].as_str().unwrap();
    let lock_sha = original["startup_input_lock_sha256"].as_str().unwrap();
    let identity = CpuValueIdentity {
        semantics: v2::SEMANTICS.into(),
        weights_sha256: Some(h("wire-semantic-NNUEC-parameters")),
        training: CpuTrainingState::Untrained,
    };
    let registration = json!({"model_identity":identity});
    let mut readouts: Vec<String> = base["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|record| record["record_sha256"].as_str().unwrap().to_owned())
        .collect();
    readouts.sort();
    let raw_weights = h("wire-raw-NNUEC-file");
    let raw_provenance = h("wire-NNUEC-provenance");
    let publication = RoventRCorpusPublication {
        schema: ROVENT_R_PUBLICATION_DOMAIN.into(),
        manifest: pin("r-corpus-manifest.json".into(), "wire-advertised-manifest"),
        index: pin("r-corpus-index.json".into(), "wire-advertised-index"),
        chunks: vec![pin("r-chunks/000000.jsonl".into(), "wire-advertised-chunk")],
        retirements: vec![
            RoventRPublicationRetirement {
                retirement_sha256: terminal_sha.clone(),
                file: pin(
                    format!("r-retirements/{terminal_sha}.json"),
                    "wire-advertised-terminal",
                ),
                readout_record_sha256: readouts.clone(),
            },
            RoventRPublicationRetirement {
                retirement_sha256: not_started_sha.clone(),
                file: pin(
                    format!("r-retirements/{not_started_sha}.json"),
                    "wire-advertised-not-started",
                ),
                readout_record_sha256: vec![],
            },
        ],
        canonical_manifest_sha256: original["manifest_sha256"].as_str().unwrap().into(),
        canonical_index_sha256: h("wire-canonical-index"),
        source_commit: base["manifest"]["collection"]["source_commit"]
            .as_str()
            .unwrap()
            .into(),
        source_sha256: base["manifest"]["collection"]["source_sha256"]
            .as_str()
            .unwrap()
            .into(),
        runtime_sha256: base["manifest"]["collection"]["runtime_sha256"]
            .as_str()
            .unwrap()
            .into(),
        record_count: 2,
        total_payload_bytes: 8,
        admissions: vec![RoventRPublicationAdmission {
            native_admission_sha256: admission_sha.into(),
            input_lock: pin(
                format!("input-locks/{lock_sha}.json"),
                "wire-advertised-lock",
            ),
            canonical_input_lock_sha256: lock_sha.into(),
            runtime_bundle_manifest: None,
            nnuec_weights: RoventPublicationFile {
                path: format!("nnuec-weights/{raw_weights}.json"),
                sha256: raw_weights.clone(),
                bytes: 1,
            },
            nnuec_provenance: RoventPublicationFile {
                path: format!("nnuec-provenance/{raw_provenance}.json"),
                sha256: raw_provenance,
                bytes: 1,
            },
            nnuec_parameter_sha256: h("wire-semantic-NNUEC-parameters"),
            nnuec_registration_sha256: generation_digest(
                "rz-rovent-r-nnuec-registration/1",
                &registration,
            )
            .unwrap(),
            nnuec_registration: registration,
        }],
    };
    publication.validate().unwrap();
    let publication_sha = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&publication).unwrap())
    );
    let task = TaskKey {
        // Store IDs are deliberately different from the Rules-owner number.
        state: StateId(91),
        line: None,
        question,
        root_moves: vec![],
        model: 0,
        epoch: 0,
        value_identity: Some(identity),
        checker_identity: None,
        cpu_condition: Some("synthetic-original-NNUEC-condition".into()),
        profile: 7,
        condition: 9,
        input_revision: 0,
        requested_depth: 1,
        node_budget: terminal.job.total_native_node_credit,
    };
    let task_value = serde_json::to_value(task).unwrap();
    let raw = RoventRSemanticObservation {
        retirement_sha256: terminal_sha.clone(),
        stage: RoventRSemanticStage::RawPublished,
        native_execution_epoch: 1,
        native_execution_sequence: 5,
        store_execution: 17,
        observation: 19,
        task_key_sha256: generation_digest(ROVENT_R_SEMANTIC_TASK, &task_value).unwrap(),
        task_key: task_value,
        job_sequence: terminal.job.job_sequence,
        worker_owner: terminal.worker_owner,
        frame_sequence: terminal.frame_sequence,
        capture_sequence: terminal.capture_sequence,
        question_sha256: terminal.job.question_sha256.clone(),
        binding_sha256: terminal.job.evaluation_namespace_sha256.clone(),
        submitted_view_sha256: original["bound"]["view_seal_sha256"]
            .as_str()
            .unwrap()
            .into(),
        submitted_full_view_sha256: original["bound"]["full_view_sha256"]
            .as_str()
            .unwrap()
            .into(),
        native_slice_work: RoventRSemanticWork {
            nodes: 3,
            quiescence_nodes: 1,
            tt_hits: 2,
        },
        selected_pc: None,
    };
    let mut feedback = raw.clone();
    feedback.stage = RoventRSemanticStage::SearchFeedbackConsumed;
    let mut selected = raw.clone();
    selected.stage = RoventRSemanticStage::SelectedPcViewConsumed;
    selected.selected_pc = Some(RoventRSemanticSelectedPc {
        request_epoch: 1,
        request_sequence: 23,
        question_sha256: h("wire-selected-PC-question"),
        view_sha256: h("wire-selected-PC-view"),
        full_view_sha256: h("wire-selected-PC-full-view"),
        output_sha256: h("wire-selected-PC-output"),
        record_revision: 4,
        record_origin_state: 91,
    });
    let mut semantic = RoventRSemanticObservations {
        schema: ROVENT_R_SEMANTIC_OBSERVATIONS.into(),
        r_publication_file_sha256: publication_sha.clone(),
        native_admission_sha256: admission_sha.into(),
        authority: authority.clone(),
        retirements: vec![
            RoventRSemanticRetirement {
                retirement_sha256: terminal_sha,
                retirement: terminal,
                readout_record_sha256: readouts,
            },
            RoventRSemanticRetirement {
                retirement_sha256: not_started_sha.clone(),
                retirement: not_started.clone(),
                readout_record_sha256: vec![],
            },
        ],
        observations: vec![raw, feedback, selected],
        unrecorded_witness_attempts: 0,
        manager_failures: vec![],
        per_readout_consumption: false,
        whole_action_cost: None,
        wdl_target: None,
        utility_target: None,
        semantic_sha256: String::new(),
    };
    semantic.semantic_sha256 =
        generation_digest(ROVENT_R_SEMANTIC_OBSERVATIONS, &semantic).unwrap();
    semantic
        .validate_historical(&publication, &publication_sha)
        .unwrap();
    let mut unsubmitted = RoventRUnsubmittedObservations {
        schema: ROVENT_R_UNSUBMITTED_OBSERVATIONS.into(),
        native_admission: original["native_admission"].clone(),
        native_admission_sha256: admission_sha.into(),
        input_lock_sha256: lock_sha.into(),
        nnuec_weights_raw_sha256: raw_weights,
        nnuec_parameter_sha256: h("wire-semantic-NNUEC-parameters"),
        authority,
        r_publication_file_sha256: Some(publication_sha.clone()),
        observations: vec![
            RoventRUnsubmittedObservation {
                kind: RoventRUnsubmittedKind::Unsubmitted,
                frame_sequence: 3,
                question_sha256: h("wire-unsubmitted-question"),
                full_view_sha256: h("wire-unsubmitted-full-view"),
                view_seal_sha256: h("wire-unsubmitted-view-seal"),
                binding_sha256: h("wire-unsubmitted-binding"),
                capture_sequence: 3,
                original_deadline_clock_epoch: 1,
                original_deadline_ns: 1000,
                job: None,
                worker_owner: None,
                retirement_sha256: None,
                retirement: None,
                activation_observed: false,
            },
            RoventRUnsubmittedObservation {
                kind: RoventRUnsubmittedKind::NotStarted,
                frame_sequence: not_started.frame_sequence,
                question_sha256: not_started_job.question_sha256.clone(),
                full_view_sha256: h("wire-not-started-full-view"),
                view_seal_sha256: h("wire-not-started-view-seal"),
                binding_sha256: not_started_job.evaluation_namespace_sha256.clone(),
                capture_sequence: not_started.capture_sequence,
                original_deadline_clock_epoch: 1,
                original_deadline_ns: 1000,
                job: Some(not_started_job),
                worker_owner: Some(0),
                retirement_sha256: Some(not_started_sha),
                retirement: Some(not_started),
                activation_observed: false,
            },
        ],
        per_readout_consumption: false,
        native_node_usage: None,
        whole_action_cost: None,
        wdl_target: None,
        utility_target: None,
        observations_sha256: String::new(),
    };
    unsubmitted.observations_sha256 =
        generation_digest(ROVENT_R_UNSUBMITTED_OBSERVATIONS, &unsubmitted).unwrap();
    unsubmitted
        .validate_historical(admission_sha, Some(&publication), Some(&publication_sha))
        .unwrap();
    json!({"schema":FIXTURE_SCHEMA,"classification":"synthetic_wire_only_no_capture_authority",
        "base":base,"publication":publication,"publication_file_sha256":publication_sha,
        "semantic":semantic,"unsubmitted":unsubmitted})
}

fn decode(
    value: &Value,
) -> crate::rovent_training_process::recovery::metadata_wire_tests::DecoderOutput {
    let helper = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../experiments/model-research/pals/tests/rovent_r_observation_wire_decode.py");
    crate::rovent_training_process::recovery::metadata_wire_tests::decode_python_fixture(
        value, &helper,
    )
}

fn reseal_semantic(value: &mut Value) {
    value["semantic"]["semantic_sha256"] = json!("");
    value["semantic"]["semantic_sha256"] =
        json!(generation_digest(ROVENT_R_SEMANTIC_OBSERVATIONS, &value["semantic"]).unwrap());
}
fn reseal_unsubmitted(value: &mut Value) {
    value["unsubmitted"]["observations_sha256"] = json!("");
    value["unsubmitted"]["observations_sha256"] =
        json!(generation_digest(ROVENT_R_UNSUBMITTED_OBSERVATIONS, &value["unsubmitted"]).unwrap());
}
fn assert_rejected(value: &Value, boundary: &str) {
    let output = decode(value);
    assert!(
        !output.status.success(),
        "malformed wire was accepted at {boundary}"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(boundary),
        "wrong rejection boundary for {boundary}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn typed_r_observation_dtos_preserve_task_enum_stages_and_unknown_targets_in_python() {
    for question in [
        TaskQuestion::AnalyzePosition,
        TaskQuestion::FindAlternative { divergence_ply: 3 },
    ] {
        let value = fixture(question);
        let output = decode(&value);
        assert!(
            output.status.success(),
            "typed observation wire rejected: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["schema"], RESULT_SCHEMA);
        assert_eq!(result["status"], "accepted_wire_only");
        assert_eq!(result["records"], 2);
        assert_eq!(result["retirements"], 2);
        assert_eq!(result["semantic_execution_count"], 1);
        assert_eq!(
            result["semantic_stages"],
            json!({"raw_published":1,"search_feedback_consumed":1,"selected_pc_view_consumed":1})
        );
        assert_eq!(result["unsubmitted_count"], 1);
        assert_eq!(result["not_started_count"], 1);
        assert_eq!(
            result["semantic_sha256"],
            value["semantic"]["semantic_sha256"]
        );
        assert_eq!(
            result["observations_sha256"],
            value["unsubmitted"]["observations_sha256"]
        );
        for name in [
            "full_typed_source_validation",
            "source_files_verified",
            "rules_replayed",
            "original_native_authority",
            "physical_completion_verified",
            "training_authorized",
            "execution_authorized",
            "neural_imports",
        ] {
            assert_eq!(result[name], false, "wire-only decoder claimed {name}");
        }
    }
}

#[test]
fn independent_r_observation_decoder_rejects_wrong_stage_enum_and_absence_claims() {
    let original = fixture(TaskQuestion::FindAlternative { divergence_ply: 3 });
    let mut reordered = original.clone();
    reordered["semantic"]["observations"]
        .as_array_mut()
        .unwrap()
        .swap(0, 1);
    reseal_semantic(&mut reordered);
    assert_rejected(&reordered, "feedback requires the earlier raw fact");

    let mut retyped = original.clone();
    retyped["semantic"]["observations"][0]["task_key"]["question"]["FindAlternative"]["divergence_ply"] =
        json!(false);
    let task_sha = generation_digest(
        ROVENT_R_SEMANTIC_TASK,
        &retyped["semantic"]["observations"][0]["task_key"],
    )
    .unwrap();
    retyped["semantic"]["observations"][0]["task_key_sha256"] = json!(task_sha);
    reseal_semantic(&mut retyped);
    assert_rejected(&retyped, "unsigned integer");

    let mut invented = original.clone();
    invented["unsubmitted"]["observations"][0]["worker_owner"] = json!(0);
    reseal_unsubmitted(&mut invented);
    assert_rejected(&invented, "absence cannot invent a Job/worker/terminal");

    let mut activated = original.clone();
    activated["unsubmitted"]["observations"][1]["activation_observed"] = json!(true);
    reseal_unsubmitted(&mut activated);
    assert_rejected(&activated, "activation-free observation required");

    let mut extra = original;
    extra["semantic"]["observations"][0]["task_key"]["unexpected"] = json!(null);
    reseal_semantic(&mut extra);
    assert_rejected(&extra, "R semantic original TaskKey");
}
