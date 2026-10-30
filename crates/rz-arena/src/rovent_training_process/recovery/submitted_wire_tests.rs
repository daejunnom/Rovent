//! Actual typed submitted-input/manifest2/index2 -> isolated Python codec.
//! Synthetic metadata only: no native witness, Rules replay, tensor/model,
//! published corpus or training/physical authority is constructed here.
use super::metadata_wire_tests::decode_python_fixture;
use super::*;
use rz_experiments::pals_data_v3::*;
use rz_experiments::rovent_corpus_v2::{RoventCorpusIndexV2, RoventCorpusManifestV2};
use rz_experiments::rovent_submitted_input::*;
use sha2::{Digest as _, Sha256};

const FIXTURE: &str = "rz-rovent-submitted-wire-fixture/1";
const RESULT: &str = "rz-rovent-submitted-wire-result/1";
fn h(label: &str) -> String {
    format!("{:x}", Sha256::digest(label.as_bytes()))
}
fn moves(line: &[u16]) -> Value {
    json!(
        line.iter()
            .map(|movement| [movement & 63, (movement >> 6) & 63, movement >> 12])
            .collect::<Vec<_>>()
    )
}

fn input(visible_counter: bool) -> RoventSubmittedInputV1 {
    let anchor = GenerationRulesAnchorV3 {
        rules_state_sha256: h("synthetic-state"),
        rules_history_sha256: h("synthetic-history"),
        board_fen: "8/8/8/8/8/8/4K3/7k w - - 0 1".into(),
        white_to_move: true,
    };
    let counter = if visible_counter { vec![64, 1] } else { vec![] };
    let records = if visible_counter {
        vec![GenerationSelectedRecordV3 {
            record_id: "visible-counter".into(),
            kind: GenerationRecordKindV3::Counter,
            anchor: 0,
            moves: counter.clone(),
            parent: None,
            supersedes: None,
            observed_sequence: 7,
            perspective: 1,
            completed_iteration: true,
            rules_terminal: false,
        }]
    } else {
        vec![]
    };
    let view = GenerationSelectedViewV3 {
        context_policy_sha256: h("synthetic-context-policy"),
        anchors: vec![anchor.clone()],
        records,
        query: GenerationSelectedQueryV3 {
            purpose: "divergence_policy".into(),
            anchors: [0; 3],
            prefix: vec![],
            proposal: (0..129)
                .map(|slot| if slot % 2 == 0 { 64 } else { 1 })
                .collect(),
            counter,
            counter_present: visible_counter,
            white_to_move: true,
        },
        token_budget: 512,
        length_bucket: 256,
    };
    let edge = [
        -0.0_f32,
        f32::from_bits(1),
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::EPSILON,
        -1.0,
        1.0,
    ];
    let mut metadata = vec![0.0_f64; 16];
    for (slot, value) in edge.iter().enumerate() {
        metadata[slot] = f64::from(*value);
    }
    let divergences = (1..=128)
        .map(|ply| {
            let mut row = vec![f64::from(ply) / 256.0];
            row.extend(edge.iter().map(|value| f64::from(*value)));
            row
        })
        .collect::<Vec<_>>();
    let encoded_records=view.records.iter().map(|record|json!({"kind":"counterexample","perspective":record.perspective,
        "completed_iteration":record.completed_iteration,"rules_terminal":record.rules_terminal,"line":moves(&record.moves),
        "anchor":record.anchor,"parent":record.parent,"supersedes":record.supersedes})).collect::<Vec<_>>();
    let query = &view.query;
    let payload = json!({"schema":"rovezero.pals-virtual-review-input.v1","role":"critic",
        "anchors":[{"board":vec![0_u8;64],"metadata":metadata}],"records":encoded_records,
        "query":{"purpose":query.purpose,"perspective":1,"anchors":query.anchors,"prefix":moves(&query.prefix),
            "proposal":moves(&query.proposal),"counter":moves(&query.counter),"counter_present":query.counter_present},
        "candidates":moves(&[64,128]),"divergences":divergences,"token_budget":view.token_budget});
    let input = RoventSubmittedInputV1 {
        schema: ROVENT_SUBMITTED_INPUT_DOMAIN.into(),
        game_id: "synthetic-game".into(),
        opening_id: "synthetic-opening".into(),
        line_genealogy_id: "synthetic-line".into(),
        transposition_sha256: h("synthetic-transposition"),
        role: GenerationRoleV3::Critic,
        model_epoch: (1_u64 << 53) + 1,
        context_epoch: (1_u64 << 53) + 3,
        capture_sequence: 9,
        question_id: h("original-question"),
        worker_id: "wire-worker".into(),
        job_id: "wire-job".into(),
        slice_id: "wire-slice".into(),
        rules_anchor: anchor,
        actual_history: vec![64],
        view_sha256: view.sha256().unwrap(),
        selected_view: view,
        legal_moves: vec![64, 128],
        action_support_mask: [false; 7],
        encoded_model_input: payload,
        attack_mapping: (0..128)
            .map(|slot| RoventAttackMapping {
                slot,
                attack_ply: slot + 1,
                selected_counter_local_index: visible_counter.then_some(0),
                selected_counter_revision: visible_counter.then_some((1_u64 << 53) + 5),
            })
            .collect(),
    };
    input.validate().unwrap();
    input
}

fn fixture(visible_counter: bool) -> Value {
    let input = input(visible_counter);
    let manifest = RoventCorpusManifestV2 {
        schema: ROVENT_CORPUS_V2_DOMAIN.into(),
        corpus_id: "synthetic-wire-only".into(),
        source_commit: "a".repeat(40),
        source_sha256: h("source"),
        runtime_sha256: h("runtime"),
        model_semantics_sha256: h("semantics"),
        model_epoch: input.model_epoch,
        context_epoch: input.context_epoch,
        recipe_sha256: h("recipe"),
        stage_id: "synthetic-stage".into(),
        round: 0,
        round_id: "synthetic-round".into(),
        student: "synthetic-student".into(),
        learner: "synthetic-learner".into(),
        one_learner: true,
        inherited_input_sha256: vec![],
        inherited_origin_sha256: vec![],
        teachers: BTreeMap::new(),
        opening_splits: BTreeMap::from([(input.opening_id.clone(), GenerationSplitV3::Train)]),
        ancestor_index_sha256: vec![],
    };
    manifest.validate().unwrap();
    // Empty index is a typed domain/hash operand, never a published corpus.
    let index = RoventCorpusIndexV2 {
        schema: ROVENT_INDEX_V2_DOMAIN.into(),
        manifest_sha256: manifest.sha256().unwrap(),
        record_count: 0,
        chunks: vec![],
        origins: vec![],
    };
    index.validate(&manifest).unwrap();
    json!({"schema":FIXTURE,"classification":"synthetic_wire_only_no_capture_authority","execution_authorized":false,
        "input":serde_json::to_value(&input).unwrap(),"manifest":serde_json::to_value(&manifest).unwrap(),
        "index":serde_json::to_value(&index).unwrap(),"expected_input_sha256":input.sha256().unwrap(),
        "expected_manifest_sha256":manifest.sha256().unwrap(),"expected_index_sha256":index.sha256().unwrap(),
        "source_domains":{"record_domain":RoventPcRecordFamily::SubmittedInputV1.record_domain(),"native_raw_domain":ROVENT_NATIVE_RAW_V3_DOMAIN}})
}
fn decode(fixture: &Value) -> metadata_wire_tests::DecoderOutput {
    let helper = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../experiments/model-research/pals/tests/rovent_submitted_wire_decode.py");
    decode_python_fixture(fixture, &helper)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Accepted {
    schema: String,
    status: String,
    input_sha256: String,
    manifest_sha256: String,
    index_sha256: String,
    attack_slots: usize,
    visible_counter: bool,
    execution_authorized: bool,
    original_capture_authority: bool,
    rules_replayed: bool,
    native_witness_validated: bool,
    neural_imports: bool,
}
fn accepted(fixture: &Value) {
    let output = decode(fixture);
    assert!(
        output.status.success(),
        "strict submitted decoder: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Accepted =
        rz_experiments::decode_json(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
    assert_eq!(result.schema, RESULT);
    assert_eq!(result.status, "accepted_wire_only");
    assert!(
        !result.execution_authorized
            && !result.original_capture_authority
            && !result.rules_replayed
            && !result.native_witness_validated
            && !result.neural_imports
    );
    assert_eq!(
        result.input_sha256,
        fixture["expected_input_sha256"].as_str().unwrap()
    );
    assert_eq!(
        result.manifest_sha256,
        fixture["expected_manifest_sha256"].as_str().unwrap()
    );
    assert_eq!(
        result.index_sha256,
        fixture["expected_index_sha256"].as_str().unwrap()
    );
    assert_eq!(result.attack_slots, 128);
    assert_eq!(
        result.visible_counter,
        fixture["input"]["attack_mapping"][0]["selected_counter_local_index"].is_number()
    );
}

#[test]
fn typed_submitted_domains_f32_bits_and_128_attack_slots_agree_with_python() {
    for visible in [false, true] {
        accepted(&fixture(visible));
    }
}
#[test]
fn attack_ply_is_not_a_selected_record_index_and_missing_counter_stays_absent() {
    let original = fixture(false);
    let mut wrong_ply = original.clone();
    wrong_ply["input"]["attack_mapping"][0]["attack_ply"] = json!(0);
    assert!(
        serde_json::from_value::<RoventSubmittedInputV1>(wrong_ply["input"].clone())
            .unwrap()
            .validate()
            .is_err()
    );
    assert!(!decode(&wrong_ply).status.success());
    let mut future = original;
    future["input"]["attack_mapping"][0]["selected_counter_local_index"] = json!(0);
    future["input"]["attack_mapping"][0]["selected_counter_revision"] = json!(7);
    assert!(!decode(&future).status.success());
    let mut partial = fixture(true);
    partial["input"]["attack_mapping"][0]["selected_counter_revision"] = Value::Null;
    assert!(!decode(&partial).status.success());
}
#[test]
fn legacy_family_future_records_and_tampered_f32_payload_are_rejected() {
    let original = fixture(true);
    let mut legacy = original.clone();
    legacy["input"]["schema"] = json!(GENERATION_INPUT_DOMAIN);
    assert!(!decode(&legacy).status.success());
    let mut legacy_index = original.clone();
    legacy_index["index"]["schema"] = json!(rz_experiments::rovent_corpus::INDEX_DOMAIN);
    assert!(!decode(&legacy_index).status.success());
    let mut future = original.clone();
    future["input"]["selected_view"]["records"][0]["observed_sequence"] = json!(10);
    // Keep the view's own seal valid so this case reaches the cross-snapshot
    // observation boundary rather than failing on a stale view digest.
    let future_view: GenerationSelectedViewV3 =
        serde_json::from_value(future["input"]["selected_view"].clone()).unwrap();
    future["input"]["view_sha256"] = json!(future_view.sha256().unwrap());
    assert!(!decode(&future).status.success());
    let mut bits = original;
    bits["input"]["encoded_model_input"]["anchors"][0]["metadata"][0] = json!(0.0);
    assert!(!decode(&bits).status.success());
}
