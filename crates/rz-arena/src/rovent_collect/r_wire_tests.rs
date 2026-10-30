//! 실제 Rust DTO의 serde 결과와 독립 Python decoder를 잇는 형식 검사다.
//! 아래 값은 합성 wire 피연산자이며 native capture, Rules 재생, 파일 출처,
//! 교사·학습 권한 또는 실제 신경망 실행을 주장하지 않는다.
use rz_experiments::pals_data_v3::{GenerationRulesAnchorV3, GenerationSplitV3, generation_digest};
use rz_experiments::rovent_corpus::RoventCorpusManifest;
use rz_experiments::rovent_r_readout::*;
use rz_experiments::*;
use rz_search::cpu_value::nnuec::{NnuecConfig, ReferenceLine, v2};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroU16,
    path::Path,
};

#[path = "r_observation_wire_tests.rs"]
mod observation_wire;

const FIXTURE_SCHEMA: &str = "rz-rovent-r-wire-fixture/1";
const RESULT_SCHEMA: &str = "rz-rovent-r-wire-result/1";
fn h(label: &str) -> String {
    format!("{:x}", Sha256::digest(label.as_bytes()))
}
fn hex(digest: rz_contracts::Digest) -> String {
    digest.0.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn artifact(path: &str, label: &str) -> Value {
    json!({"path":path,"sha256":h(label),"bytes":1,
        "source":"synthetic-wire-only","license":"synthetic-wire-only"})
}
fn rules_evidence() -> RoventRRulesEvidence {
    let anchor = GenerationRulesAnchorV3 {
        rules_state_sha256: h("wire-current-state"),
        rules_history_sha256: h("wire-current-history"),
        board_fen: "8/8/8/8/8/8/4K3/7k w - - 0 1".into(),
        white_to_move: true,
    };
    let replay = json!({"schema":"rz-rovent-rules-position-replay/1","origin":"fen",
        "completeness":"unknown_prefix","start_fen":anchor.board_fen,"moves":[],
        "observer_revision":1,"anchor":anchor});
    RoventRRulesEvidence {
        anchor,
        replay_sha256: generation_digest("rz-rovent-rules-position-replay/1", &replay).unwrap(),
        replay,
    }
}

/// 모든 새 R DTO는 실제 struct literal이다. Value를 DTO로 역직렬화하여
/// Rust 필드의 추가·이름 변경을 숨기는 경로는 이 fixture에 없다.
fn fixture() -> Value {
    let source_commit = "1".repeat(40);
    let components = BTreeMap::from([("synthetic-wire-source".into(), h("wire-component"))]);
    let source = rovent_source_aggregate_sha256(&source_commit, &components).unwrap();
    let manifest = RoventRReadoutManifest {
        schema: R_READOUT_MANIFEST_DOMAIN.into(),
        collection: RoventCorpusManifest {
            schema: "rz-rovent-corpus-manifest/1".into(),
            corpus_id: "wire-r-only".into(),
            source_commit: source_commit.clone(),
            source_sha256: source.clone(),
            runtime_sha256: h("wire-loaded-runtime-file"),
            model_semantics_sha256: h("wire-PC-semantics"),
            model_epoch: 1,
            context_epoch: 2,
            recipe_sha256: h("wire-recipe-declaration"),
            stage_id: "wire-stage".into(),
            round: 0,
            round_id: "wire-round".into(),
            student: "wire-student".into(),
            learner: "wire-learner".into(),
            one_learner: true,
            inherited_input_sha256: vec![],
            inherited_origin_sha256: vec![],
            teachers: BTreeMap::new(),
            opening_splits: BTreeMap::from([("wire-opening".into(), GenerationSplitV3::Train)]),
            ancestor_index_sha256: vec![],
        },
        pc_ancestor_index_sha256: vec![],
        r_ancestor_index_sha256: vec![],
    };
    let evidence = rules_evidence();
    let nnuec = json!({"schema":"rz-nnuec-fp32-v2","semantics":v2::SEMANTICS,
        "encoding":"wire-origin-first","implementation_sha256":h("wire-NNUEC-implementation"),
        "configuration_sha256":h("wire-NNUEC-configuration"),
        "weights":artifact("wire-nnuec.json","wire-raw-NNUEC-file"),
        "provenance":artifact("wire-nnuec-provenance.json","wire-NNUEC-provenance"),
        "diagnostic":false,"context_namespace_policy_sha256":h("wire-C-namespace"),
        "local_namespace_policy_sha256":h("wire-L-namespace"),
        "search_conditions_sha256":h("wire-search-conditions")});
    let resources = RoventResourcePlan {
        cpu_quota_millicpus: 2000,
        pools: vec![RoventThreadPool {
            actors: BTreeSet::from([RoventActor::Manager]),
            max_threads: 1,
            cpu_charge_millicpus: 1000,
        }],
        memory_high_bytes: 6 << 30,
        memory_max_bytes: 12 << 30,
        gpu_devices: 0,
        physical_nn_concurrency: 1,
        topology_policy: "SharedControl".into(),
    };
    let search = json!({"node_unit":ROVENT_POSITION_VISIT_UNIT,"requested_nodes":4096,
        "depth_plies":null,"mate_moves":null,"searchmoves":null,"registered_max_chess_plies":256,
        "root_state_and_history_sha256":evidence.anchor.rules_state_sha256,
        "rules_profile_sha256":h("wire-Rules-profile"),
        "search_configuration_sha256":h("wire-search-config")});
    let coordinator = json!({"policy_sha256":h("wire-coordinator"),"cpu_credit_limit":4096,
        "questions_max":64,"ancestor_depth_max":8,"waiting_jobs_max":64,"role_tickets_max":16});
    // 이 선언은 실행 가능한 full input lock이 아니다. decoder의 persisted
    // join만 검사하며 live owner와 파일을 만드는 constructor는 호출하지 않는다.
    let execution = json!({"domain":"rz-rovent-execution/1","contract_revision":"0.1",
        "run_id":"wire-not-executable","purpose":"collection","engine":{
            "source_commit":source_commit,"source_components":components,"mode":"hybrid_pcr",
            "search":search,"teachers":[],"nnuec":nnuec},"resources":resources,
        "coordinator":coordinator,"wall_time_max_ms":1000,"cleanup_max_ms":500,
        "output_bytes_max":1048576,"agent_renewable":true,"provenance":[]});
    let startup_sha = generation_digest("rz-rovent-execution-canonical/1", &execution).unwrap();
    let lock = json!({"domain":"rz-rovent-input-lock/1","manifest":execution,"canonical_sha256":startup_sha});
    let deadline = |at_ns| RoventRootDeadline {
        clock_epoch: 1,
        at_ns,
    };
    let descriptor = RoventRootDescriptor::seal(RoventRootDescriptorInput {
        policy: RoventRootAdmissionPolicy::ExactRegistered,
        startup_lock_sha256: startup_sha.clone(),
        state_and_history_sha256: evidence.anchor.rules_state_sha256.clone(),
        rules_profile_sha256: h("wire-Rules-profile"),
        authority: RoventRootAuthority {
            epoch: 1,
            game: 1,
            root: 1,
            implementation_sha256: h("wire-UCI-implementation"),
        },
        node_unit: ROVENT_POSITION_VISIT_UNIT.into(),
        search_configuration_sha256: h("wire-search-config"),
        coordinator_policy_sha256: h("wire-coordinator"),
        original_go: RoventResolvedGo {
            coverage: RoventOriginalGoCoverage::SemanticContextOnly,
            requested_nodes: Some(4096),
            depth_plies: None,
            mate_moves: None,
            searchmoves: None,
            transport_simulation_cap: 4096,
            original_parsed_limits: None,
        },
        allowed_root_moves: vec!["e2e3".into()],
        clock: RoventRootClocks {
            hard: deadline(1000),
            soft: deadline(500),
            admission: deadline(800),
            cleanup: deadline(2000),
        },
        declared_ceilings: RoventRootDeclaredCeilings {
            resource_plan: resources,
            registered_max_chess_plies: NonZeroU16::new(256).unwrap(),
            cpu_credit_limit: 4096,
            questions_max: 64,
            ancestor_depth_max: 8,
            waiting_jobs_max: 64,
            role_tickets_max: 16,
            wall_time_max_ms: 1000,
            cleanup_max_ms: 500,
            output_bytes_max: 1048576,
        },
    })
    .unwrap();
    let descriptor_sha = hex(descriptor.digest().unwrap());
    let origin = json!({"schema":"rz-rovent-collection-native-origin/1","purpose":"collection",
        "process_epoch":1,"mode":"HybridPcr","input_lock_sha256":startup_sha,
        "source_commit":manifest.collection.source_commit,"source_components":components,"source_sha256":source,
        "weights_sha256":h("wire-PC-checkpoint-domain"),"export_manifest_sha256":h("wire-PC-export"),
        "backend_sha256":h("wire-backend"),"runtime_library_sha256":manifest.collection.runtime_sha256,
        "runtime_bundle_sha256":null,"native_configuration_sha256":h("wire-native-config"),
        "model_semantics":"wire-PC-semantics","model_semantics_sha256":manifest.collection.model_semantics_sha256,
        "encoding":"wire-PC-encoding","encoding_sha256":h("wire-PC-encoding"),
        "encoding_registry_sha256":h("wire-encoding-registry"),"model_implementation_sha256":h("wire-PC-implementation"),
        "frozen_epoch":1,"representation_model_domain":"loaded_checkpoint_sha256",
        "runtime_sha256_domain":"loaded_runtime_library_file_sha256","runtime_sha256":manifest.collection.runtime_sha256,
        "teacher_profiles":[]});
    let native_admission = json!({"schema":"rz-rovent-collection-root-admission/1","native_origin":origin,
        "native_origin_sha256":generation_digest("rz-rovent-collection-native-origin/1",&origin).unwrap(),
        "process_epoch":1,"game_generation":1,"root_generation":1,"implementation_sha256":h("wire-UCI-implementation"),
        "deadline_clock":1,"deadline_tick":1000,"root_budget_sha256":descriptor.root_budget_sha256,
        "selective_policy_sha256":h("wire-selective-policy"),"root_state":{
            "owner":1,"revision":1,"semantic":evidence.anchor.rules_state_sha256},"store_root_state":1,
        "root_moves":["e2e3"],"depth_plies":null,"mate_moves":null,"requested_nodes":4096,
        "root_descriptor":descriptor,"root_descriptor_sha256":descriptor_sha});
    let reference_input = v2::ReferenceInput {
        schema: "rz-nnuec-fp32-input/2".into(),
        board_features: [vec![768, 774, 783], vec![769, 774, 783]],
        side_to_move: "white".into(),
        context_lines: vec![],
        context_edges: vec![],
        local_line: ReferenceLine {
            anchor_features: vec![768, 774, 783],
            perspective: "white".into(),
            moves: vec![],
        },
    };
    let original_json = serde_json::to_string(&reference_input).unwrap();
    let mut prefix = Sha256::new();
    prefix.update(R_CAPTURE_INPUT_DOMAIN.as_bytes());
    prefix.update([0]);
    prefix.update(original_json.as_bytes());
    let native_input_sha = format!("{:x}", prefix.finalize());
    let bound = RoventRBoundEvidence {
        model_semantics: v2::SEMANTICS.into(),
        parameter_sha256: h("wire-semantic-NNUEC-parameters"),
        training_state: json!({"kind":"untrained"}),
        weights_file_sha256: h("wire-raw-NNUEC-file"),
        collection_model_epoch: 1,
        collection_context_epoch: 2,
        root_state: 1,
        generation: 1,
        sequence: 1,
        full_view_sha256: h("wire-full-view"),
        view_seal_sha256: h("wire-view-seal"),
        record_scope_sha256: h("wire-record-scope"),
        policy_sha256: h("wire-review-policy"),
        context_sha256: v2::context_sha256(
            &reference_input.context_lines,
            &reference_input.context_edges,
        )
        .unwrap(),
        context_query: RoventRContextQuery {
            purpose: "ProposePolicy".into(),
            prefix: vec![],
            proposal: vec![],
            counterexample: None,
            legal: vec![],
            divergences: vec![],
        },
        selection_policy: RoventRSelectionPolicy {
            mode: "current_root_only".into(),
            token_budget: 512,
            max_records: 64,
            max_anchors: 1,
            required_bundle: None,
        },
        actual_tokens: 0,
        token_budget: 512,
        length_bucket: 8,
        context_records: vec![],
        context_anchors: vec![evidence.clone()],
        context_source_anchors: vec![],
        context_anchor_states: vec![1],
        local_anchor: evidence.clone(),
        local_encoded_perspective: "white".into(),
        current_rules: evidence,
    };
    let input =
        RoventRInputEvidence::from_original_json(original_json, &bound, &native_input_sha).unwrap();
    let job = RoventRJob {
        authority: RoventRAuthority {
            process_epoch: 1,
            game_generation: 1,
            root_generation: 1,
            implementation_sha256: h("wire-UCI-implementation"),
        },
        job_sequence: 1,
        state_owner: 1,
        state_revision: 1,
        state_semantic_sha256: h("wire-current-state"),
        question_sha256: h("wire-question"),
        root_restriction_sha256: h("wire-root-restriction"),
        evaluation_namespace_sha256: h("wire-evaluation-namespace"),
        search_conditions_sha256: h("wire-search-conditions"),
        total_native_node_credit: 4096,
        hard_deadline_clock_epoch: 1,
        hard_deadline_ns: 1000,
    };
    let retirement = RoventRCaptureRetirement {
        job: job.clone(),
        worker_owner: 0,
        frame_sequence: 1,
        capture_sequence: 1,
        ending: RoventRCaptureEnding::Unknown,
        limits: RoventRCaptureLimits {
            maximum_readouts: 64,
            every_nth_score: 2,
            maximum_input_bytes: 1048576,
        },
        attempted: 3,
        selected: 2,
        retained_readouts: 2,
        overflow: 0,
        retained_input_bytes: input.reference_input_bytes * 2,
        selected_input_preparation_ns: 12,
        error: Some("synthetic_wire_only_no_capture_authority".into()),
    };
    let mut edge_bits = [0; 11];
    edge_bits[..4].copy_from_slice(&[
        1e-8_f32.to_bits(),
        1.0_f32.to_bits(),
        (-0.0_f32).to_bits(),
        0.1_f32.to_bits(),
    ]);
    let numeric = RoventRNumericBits {
        schema: "rz-nnuec-fp32-output/2".into(),
        parameter_sha256: bound.parameter_sha256.clone(),
        board_accumulators_bits: [vec![0], vec![0]],
        context_accumulator_bits: vec![0],
        local_accumulator_bits: vec![0],
        hidden_bits: vec![0],
        raw_outputs_bits: edge_bits,
        value_bits: 0,
        wdl_logits_bits: [0; 3],
        wdl_bits: [0.25_f32.to_bits(), 0.5_f32.to_bits(), 0.25_f32.to_bits()],
    };
    let mut rows = Vec::new();
    for readout_sequence in [0, 2] {
        let mut row = RoventRReadoutRecord {
            schema: R_READOUT_RECORD_DOMAIN.into(),
            manifest_sha256: manifest.sha256().unwrap(),
            split: GenerationSplitV3::Train,
            lineage: RoventRLineage {
                game_id: "wire-game".into(),
                opening_id: "wire-opening".into(),
                line_genealogy_id: "wire-line".into(),
                transposition_sha256: h("wire-transposition"),
                actor_id: "wire-student-white".into(),
                teacher_actor_id: None,
                ancestor_origin_sha256: vec![],
                pc_genealogy: None,
            },
            native_admission: native_admission.clone(),
            native_admission_sha256: generation_digest(
                "rz-rovent-collection-root-admission/1",
                &native_admission,
            )
            .unwrap(),
            root_descriptor: descriptor.clone(),
            root_descriptor_sha256: descriptor_sha.clone(),
            startup_input_lock_sha256: startup_sha.clone(),
            question: RoventRQuestion {
                job: job.clone(),
                worker_owner: 0,
                frame_sequence: 1,
                capture_sequence: 1,
                readout_sequence,
                observed_tick_ns: 1100 + readout_sequence,
            },
            bound: bound.clone(),
            input: input.clone(),
            numeric: numeric.clone(),
            retirement: retirement.clone(),
            retirement_sha256: String::new(),
            semantic_consumption: None,
            whole_action_cost: None,
            wdl: None,
            utility: RoventRUtilityState {
                support_mask: [false; 7],
                target_known_mask: [false; 7],
                unknown_reason: "no_actual_action_closure".into(),
            },
            origin_sha256: String::new(),
            record_sha256: String::new(),
        };
        row.seal(&manifest).unwrap();
        rows.push(row);
    }
    assert_eq!(
        rows[0].input.semantic_input_sha256,
        rows[1].input.semantic_input_sha256
    );
    assert_ne!(rows[0].record_sha256, rows[1].record_sha256);
    let origins = rows
        .iter()
        .map(RoventRReadoutOrigin::from_record)
        .collect::<Vec<_>>();
    let config = NnuecConfig {
        board_width: 1,
        line_width: 1,
        context_width: 1,
        local_width: 1,
        hidden_width: 1,
        causal_window: 3,
        max_lines: 4,
        max_plies: 8,
    };
    let edge = json!({"small":1e-8_f64,"one":1.0_f64,"negative_zero":-0.0_f64,"float32":f64::from(0.1_f32),"integer":1});
    json!({"schema":FIXTURE_SCHEMA,"classification":"synthetic_wire_only_no_capture_authority",
        "manifest":manifest,"records":rows,"origins":origins,"input_locks":BTreeMap::from([(startup_sha,lock)]),
        "nnuec_config":config,"edge_payload":edge,
        "edge_sha256":r_readout_f32_digest(R_READOUT_FLOAT_BITS_DOMAIN,&edge).unwrap()})
}

#[test]
fn typed_rust_r_readouts_keep_original_utf8_and_f32_words_in_python() {
    let fixture = fixture();
    let helper = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../experiments/model-research/pals/tests/rovent_r_wire_decode.py");
    let output =
        crate::rovent_training_process::recovery::metadata_wire_tests::decode_python_fixture(
            &fixture, &helper,
        );
    assert!(
        output.status.success(),
        "independent R decoder rejected actual Rust DTOs: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["schema"], RESULT_SCHEMA);
    assert_eq!(result["status"], "accepted_wire_only");
    assert_eq!(result["records"], 2);
    assert_eq!(result["readout_sequences"], json!([0, 2]));
    assert_eq!(result["edge_sha256"], fixture["edge_sha256"]);
    assert_eq!(result["original_capture_authority"], false);
    assert_eq!(result["rules_replayed"], false);
    assert_eq!(result["execution_authorized"], false);
    assert_eq!(result["neural_imports"], false);
}

#[test]
fn independent_r_decoder_rejects_extra_fields_and_retyped_occurrence() {
    let helper = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../experiments/model-research/pals/tests/rovent_r_wire_decode.py");
    let decode = |value: &Value| {
        crate::rovent_training_process::recovery::metadata_wire_tests::decode_python_fixture(
            value, &helper,
        )
    };
    let original = fixture();
    let mut extra = original.clone();
    extra["records"][0]["bound"]["unexpected_wire_field"] = json!(true);
    assert!(!decode(&extra).status.success());
    let mut boolean = original;
    boolean["records"][0]["question"]["readout_sequence"] = json!(false);
    assert!(!decode(&boolean).status.success());
}
