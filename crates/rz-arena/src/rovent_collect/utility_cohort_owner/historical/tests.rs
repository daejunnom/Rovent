//! Independently authored original-wire operands for the detached verifier.
//! These are not native/Store/training capabilities or actual engine results.
use super::*;
use serde_json::json;

fn h(byte: u8) -> String {
    hex(rz_contracts::Digest([byte; 32]))
}
fn moves(text: &str) -> Vec<rz_position::BoardMove> {
    text.split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}
fn bits(line: &[rz_position::BoardMove]) -> Vec<u16> {
    line.iter()
        .map(|v| Move16::encode(rz_contracts::Move::try_from(*v).unwrap()).bits())
        .collect()
}
fn json<T: serde::Serialize>(value: T) -> String {
    serde_json::to_string(&value).unwrap()
}
fn object(fields: &[(&str, String)]) -> String {
    format!(
        "{{{}}}",
        fields
            .iter()
            .map(|(key, value)| format!("{}:{value}", json(key)))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn model() -> v2::Model {
    use rz_search::cpu_value::nnuec as n;
    let mut parameters = serde_json::Map::new();
    for (name, size) in [
        ("board_weights", 800),
        ("board_bias", 1),
        ("anchor_weights", 800),
        ("anchor_bias", 1),
        ("from_weights", 64),
        ("to_weights", 64),
        ("promotion_weights", 5),
        ("ply_weights", 16),
        ("token_bias", 1),
        ("causal_weights", 3),
        ("causal_bias", 1),
        ("context_projection", 1),
        ("context_bias", 1),
        ("local_projection", 1),
        ("local_bias", 1),
        ("hidden_weights", 4),
        ("hidden_bias", 1),
        ("output_weights", 11),
        ("output_bias", 11),
    ] {
        parameters.insert(name.into(), json!(vec![0.0_f32; size]));
    }
    let value = json!({"schema":v2::SCHEMA,"semantics":v2::SEMANTICS,
        "base":{"schema":n::NNUEC_SCHEMA,"semantics":n::NNUEC_SEMANTICS,
            "config":{"board_width":1,"line_width":1,"context_width":1,"local_width":1,"hidden_width":1,"causal_window":3,"max_lines":64,"max_plies":16},
            "training":{"kind":"untrained"},"parameters":parameters,"output_scale":1.0},
        "parameters":{"kind_embeddings":[0.0,0.0,0.0,0.0],"pair_parent":[0.0,0.0],"pair_child":[0.0,0.0],"pair_bias":[0.0,0.0],"pair_projection":[0.0]}});
    // Parameter/schema identity only: no NN encode, forward or optimizer.
    v2::Model::new(serde_json::from_value(value).unwrap()).unwrap()
}
fn root(position: &Position) -> RoventRootAdmissionEvidence {
    let deadline = |at| json!({"clock_epoch":1,"at_ns":at});
    let plan = json!({"cpu_quota_millicpus":2000,"pools":[{"actors":["manager","pc_caller"],"max_threads":2,"cpu_charge_millicpus":2000}],
        "memory_high_bytes":6_u64<<30,"memory_max_bytes":12_u64<<30,"gpu_devices":0,"physical_nn_concurrency":1,"topology_policy":"SharedControl"});
    let allowed = position
        .ordered_legal_moves()
        .moves()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let semantic = exact_rules_anchor(position).unwrap().rules_state_sha256;
    let descriptor=rz_experiments::RoventRootDescriptor::seal(serde_json::from_value(json!({
        "policy":"dynamic_go","startup_lock_sha256":h(20),"state_and_history_sha256":semantic,"rules_profile_sha256":h(21),
        "authority":{"epoch":1,"game":1,"root":1,"implementation_sha256":h(1)},"node_unit":"rovent-position-visits/1",
        "search_configuration_sha256":h(22),"coordinator_policy_sha256":h(23),
        "original_go":{"coverage":"semantic_context_only","requested_nodes":null,"depth_plies":null,"mate_moves":null,
            "searchmoves":null,"transport_simulation_cap":4096,"original_parsed_limits":null},"allowed_root_moves":allowed,
        "clock":{"hard":deadline(10000),"soft":deadline(9000),"admission":deadline(9000),"cleanup":deadline(11000)},
        "declared_ceilings":{"resource_plan":plan,"registered_max_chess_plies":256,"cpu_credit_limit":4096,"questions_max":64,
            "ancestor_depth_max":8,"waiting_jobs_max":64,"role_tickets_max":16,"wall_time_max_ms":1000,"cleanup_max_ms":500,"output_bytes_max":1048576}})).unwrap()).unwrap();
    let origin = json!({"schema":"rz-rovent-collection-native-origin/1","purpose":"collection","process_epoch":1,"mode":"CpuPc",
        "input_lock_sha256":h(20),"source_commit":"0123456789012345678901234567890123456789","source_components":{"fixture":h(2)},
        "source_sha256":h(3),"weights_sha256":h(5),"export_manifest_sha256":h(4),"backend_sha256":h(7),"runtime_library_sha256":h(8),
        "runtime_bundle_sha256":null,"native_configuration_sha256":h(9),"model_semantics":"synthetic parser source",
        "model_semantics_sha256":h(10),"encoding":"synthetic parser encoding","encoding_sha256":h(6),"encoding_registry_sha256":h(11),
        "model_implementation_sha256":h(12),"frozen_epoch":1,"representation_model_domain":"synthetic parser model",
        "runtime_sha256_domain":"loaded_runtime_library_file_sha256","runtime_sha256":h(8),"teacher_profiles":[]});
    serde_json::from_value(json!({"schema":"rz-rovent-collection-root-admission/1","native_origin":origin,
        "native_origin_sha256":generation_digest("rz-rovent-collection-native-origin/1",&origin).unwrap(),
        "process_epoch":1,"game_generation":1,"root_generation":1,"implementation_sha256":h(1),"deadline_clock":1,"deadline_tick":10000,
        "root_budget_sha256":descriptor.root_budget_sha256,"selective_policy_sha256":h(24),
        "root_state":{"owner":1,"revision":position.revision(),"semantic":semantic},"store_root_state":1,"root_moves":allowed,
        "depth_plies":null,"mate_moves":null,"requested_nodes":null,"root_descriptor_sha256":hex(descriptor.digest().unwrap()),"root_descriptor":descriptor})).unwrap()
}
fn snapshot(position: &Position) -> String {
    let snapshot = position.snapshot();
    let replay = snapshot.uci_replay(16384).unwrap();
    object(&[
        (
            "origin",
            json(
                if replay.origin == rz_position::HistoryOrigin::StartPosition {
                    0
                } else {
                    1
                },
            ),
        ),
        ("start_fen", json(replay.start_fen)),
        ("endpoint_fen", json(snapshot.to_fen())),
        ("moves", json(bits(&replay.moves))),
        ("revision", json(snapshot.revision())),
    ])
}
fn endpoint(root: &Position, line: &[rz_position::BoardMove], value: i32) -> String {
    let mut position = root.clone();
    for (ply, movement) in line.iter().enumerate() {
        position.make_move(*movement).unwrap_or_else(|error| {
            panic!(
                "original Rules fixture ply {} ({movement}) rejected: {error:?}",
                ply + 1
            )
        });
    }
    assert!(matches!(
        position.classify_position().unwrap().play_status,
        rz_position::PlayStatus::Terminal { .. }
    ));
    object(&[
        ("kind", json("Rules")),
        ("line", json(bits(line))),
        ("snapshot", snapshot(&position)),
        ("root_value", json(value)),
    ])
}
fn stable(text: &str) -> u64 {
    text.bytes().fold(0xcbf29ce484222325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}
fn summary(
    root: &Position,
    id: usize,
    line: &[rz_position::BoardMove],
    repair: bool,
    value: f32,
    supersedes: Option<usize>,
) -> String {
    let scope = object(&[
        ("kind", json("Model")),
        ("model", json(0)),
        ("encoding", json(stable("pals-restricted-summary-v1"))),
        ("input", json(0)),
    ]);
    let score = object(&[
        ("kind", json("Estimate")),
        ("bits", json(value.to_bits())),
        ("side", json(0)),
    ]);
    let stored = object(&[("id", json(id)), ("moves", json(bits(line)))]);
    object(&[
        ("id", json(id)),
        ("state_id", json(1)),
        ("state", snapshot(root)),
        ("line", stored),
        ("source", json(0)),
        ("epoch", json(0)),
        ("value", "null".into()),
        ("checker", "null".into()),
        ("work", "null".into()),
        ("external", "null".into()),
        ("model", "null".into()),
        ("input", "null".into()),
        ("condition", "null".into()),
        ("cpu_pv", "null".into()),
        ("scope", scope),
        ("score", score),
        ("budget", json(0)),
        ("kind", json(if repair { "Repair" } else { "Refutation" })),
        ("supersedes", json(supersedes)),
        ("execution", "null".into()),
    ])
}
fn recipe(
    root: &Position,
    before: String,
    after: String,
    superseded: Option<String>,
    previous: Option<String>,
) -> String {
    object(&[
        ("root", snapshot(root)),
        ("before", before),
        ("after", after),
        ("superseded", superseded.unwrap_or("null".into())),
        ("previous", previous.unwrap_or("null".into())),
    ])
}
fn witness_identity(root: &Position, line: &[rz_position::BoardMove], repair: bool) -> String {
    let mut h = Sha256::new();
    h.update(b"rz-pals-action-unique-witness/1\0");
    h.update(rz_position::contracts::rules_snapshot_semantic_digest(&root.snapshot()).0);
    h.update([u8::from(repair)]);
    for m in bits(line) {
        h.update(m.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn preimage(
    root: &Position,
    observation_line: &[rz_position::BoardMove],
    actual: &[rz_position::BoardMove],
    repair: bool,
    value: f32,
) -> Vec<u8> {
    use rz_search::pals::store::ObservationKind as K;
    let mut result = b"rz-rovent-utility-actual-conditional-witness/1\0".to_vec();
    result.extend_from_slice(&[13; 32]);
    result.extend_from_slice(&7_u64.to_le_bytes());
    result.extend_from_slice(&unhex(&witness_identity(root, actual, repair)).unwrap());
    result.push(if repair {
        K::Repair as u8
    } else {
        K::Refutation as u8
    });
    result.extend_from_slice(
        &rz_position::contracts::rules_snapshot_semantic_digest(&root.snapshot()).0,
    );
    result.push(1);
    result.extend_from_slice(&(observation_line.len() as u64).to_le_bytes());
    for m in bits(observation_line) {
        result.extend_from_slice(&m.to_le_bytes());
    }
    result.push(0);
    result.extend_from_slice(&0_u64.to_le_bytes());
    result.extend_from_slice(&0_u64.to_le_bytes());
    result.extend_from_slice(&[0, 0, 1, 0]);
    result.extend_from_slice(&value.to_bits().to_le_bytes());
    result
}
fn basis(
    root: &Position,
    original: &RoventRootAdmissionEvidence,
    actual: &[rz_position::BoardMove],
    raw: &[u8],
    repair: bool,
    observation: String,
    recipe: String,
) -> String {
    let compare = object(&[("kind", json("Finite")), ("recipe", recipe)]);
    let auth = object(&[
        ("epoch", json(1)),
        ("game", json(1)),
        ("root", json(1)),
        ("implementation", json([1_u8; 32])),
    ]);
    object(&[
        (
            "schema",
            json("rz-rovent-utility-cohort-conditional-basis/1"),
        ),
        (
            "domain",
            json("rz-rovent-utility-cohort-conditional-basis/1"),
        ),
        ("root", snapshot(root)),
        ("authority", auth),
        ("root_state", json(1)),
        ("kind", json(u8::from(repair))),
        ("line", json(bits(actual))),
        ("source_question", json([13_u8; 32])),
        ("source_frame_sequence", json(7)),
        (
            "identity",
            json(hash32(&witness_identity(root, actual, repair)).unwrap().0),
        ),
        ("raw_evidence", json(Sha256::digest(raw).to_vec())),
        (
            "root_budget",
            json(hash32(&original.root_budget_sha256).unwrap().0),
        ),
        ("original_store_byte_limit", json(16_u64 << 20)),
        ("observation", observation),
        ("comparison", compare),
    ])
}

fn known_rules_case() -> (
    Position,
    RoventRootAdmissionEvidence,
    v2::Model,
    Value,
    Vec<rz_position::BoardMove>,
) {
    let root = Position::from_fen("8/7k/5K2/8/8/8/8/Q7 w - - 0 1").unwrap();
    let original = super::tests::root(&root);
    let model = model();
    // Qb1 checks h7 while leaving both Kg8 and Kh8 legal. The counter
    // reaches a real stalemate; the different continuation repairs it into
    // a real mate. No queen line passes through its own king on f6.
    let proposal = moves("a1b1 h7g8 b1b6 g8h7 b6b7 h7h8 b7g7");
    let counter = moves("a1b1 h7h8 b1g6");
    let repair = moves("a1b1 h7h8 b1e1 h8h7 f6f7 h7h8 e1h1");
    let old = summary(&root, 11, &proposal, false, 0.0, None);
    let prior = recipe(
        &root,
        endpoint(&root, &proposal, 30000),
        endpoint(&root, &counter, 0),
        None,
        None,
    );
    let repaired = summary(&root, 12, &repair, true, 30000.0, Some(11));
    let new = recipe(
        &root,
        endpoint(&root, &counter, 0),
        endpoint(&root, &repair, 30000),
        Some(old),
        Some(prior),
    );
    let raw = preimage(&root, &repair, &repair, true, 30000.0);
    let body = basis(&root, &original, &repair, &raw, true, repaired, new);
    let tuple = json!({"observation":12,"kind":"Repair","identity_sha256":witness_identity(&root,&repair,true),"source_question_sha256":h(13),
        "source_frame_sequence":7,"raw_evidence_sha256":sha(&raw)});
    let mut pre = tuple.clone();
    pre["preimage_bytes"] = json!(raw.len());
    pre["preimage_hex"] = json!(raw.iter().map(|v| format!("{v:02x}")).collect::<String>());
    let mut b = tuple.clone();
    b["basis"] = json!({"domain":"rz-rovent-utility-cohort-conditional-basis/1","sha256":sha(body.as_bytes()),"bytes":body.len(),"original_json":body});
    b["unavailable_reason"] = Value::Null;
    let mut witness = tuple;
    witness["line"] = json!(bits(&repair));
    (
        root,
        original,
        model,
        json!({"witness_preimages":[pre],"witness_basis":[b],"new_witnesses":[witness],"pc_consumption":[],"r_selected_pc_consumption":[]}),
        repair,
    )
}

#[test]
fn rules_counter_ancestry_and_repair_basis_are_checked_before_known_candidate() {
    let (root, original, model, semantic, line) = known_rules_case();
    let witness = &semantic["new_witnesses"][0];
    let witnesses = BTreeMap::from([(
        string(witness, "identity_sha256").unwrap().to_owned(),
        (witness, line.clone()),
    )]);
    assert!(
        verify_witness_basis(&semantic, &witnesses, &root, &original, &model, 16 << 20).unwrap()
    );
    for axis in [
        "basis_sha",
        "authority",
        "anchor",
        "root_budget",
        "store_limit",
        "source_question",
        "raw_preimage",
    ] {
        let mut altered = semantic.clone();
        let row = &mut altered["witness_basis"][0];
        match axis {
            "basis_sha" => row["basis"]["sha256"] = json!(h(99)),
            "raw_preimage" => altered["witness_preimages"][0]["preimage_hex"] = json!("00"),
            _ => {
                let text = row["basis"]["original_json"].as_str().unwrap();
                let changed = match axis {
                    "authority" => text.replacen("\"epoch\":1", "\"epoch\":2", 1),
                    "anchor" => text.replacen("\"root_state\":1", "\"root_state\":2", 1),
                    "root_budget" => text.replacen(
                        &json(hash32(&original.root_budget_sha256).unwrap().0),
                        &json([97_u8; 32]),
                        1,
                    ),
                    "store_limit" => text.replacen(
                        "\"original_store_byte_limit\":16777216",
                        "\"original_store_byte_limit\":16777215",
                        1,
                    ),
                    "source_question" => text.replacen(&json([13_u8; 32]), &json([96_u8; 32]), 1),
                    _ => unreachable!(),
                };
                assert_ne!(changed, text, "mutation did not change {axis}");
                row["basis"]["bytes"] = json!(changed.len());
                row["basis"]["sha256"] = json!(sha(changed.as_bytes()));
                row["basis"]["original_json"] = json!(changed);
            }
        }
        let modified_witness = &altered["new_witnesses"][0];
        let expected = BTreeMap::from([(
            string(modified_witness, "identity_sha256")
                .unwrap()
                .to_owned(),
            (modified_witness, line.clone()),
        )]);
        assert!(
            verify_witness_basis(&altered, &expected, &root, &original, &model, 16 << 20).is_err(),
            "accepted changed {axis}"
        );
    }
}

#[test]
fn a_missing_actual_basis_is_masked_while_an_unrelated_body_is_rejected() {
    let (root, original, model, mut semantic, line) = known_rules_case();
    semantic["witness_basis"][0]["basis"] = Value::Null;
    semantic["witness_basis"][0]["unavailable_reason"] = json!("missing_original_comparison");
    let witness = &semantic["new_witnesses"][0];
    let expected = BTreeMap::from([(
        string(witness, "identity_sha256").unwrap().to_owned(),
        (witness, line.clone()),
    )]);
    assert!(
        !verify_witness_basis(&semantic, &expected, &root, &original, &model, 16 << 20).unwrap()
    );
    semantic["witness_basis"][0]["identity_sha256"] = json!(h(98));
    let witness = &semantic["new_witnesses"][0];
    let expected = BTreeMap::from([(
        string(witness, "identity_sha256").unwrap().to_owned(),
        (witness, line),
    )]);
    assert!(
        verify_witness_basis(&semantic, &expected, &root, &original, &model, 16 << 20).is_err()
    );
}

fn original_origin(root: &RoventRootAdmissionEvidence) -> Value {
    let view = json!({"authority":authority(root),"sequence":1,"context_policy_sha256":h(18),"view_digest":h(14),
        "root_state":{"owner":1,"revision":root.root_state.revision,"semantic_sha256":root.root_state.semantic},
        "actual_tokens":3,"selected_records":2,"anchors":1,"length_bucket":8});
    let representation = json!({"input":h(25),"history":h(26),"candidates":h(27),"mask_positions":h(28),"model":h(5),"encoding":h(6),
        "precision":"Fp32","frozen_epoch":1,"record_revision":1,"role":"proposer"});
    let ticket = json!({"authority":authority(root),"request_epoch":1,"request_sequence":1,"role":"Proposer","question_sha256":h(13),
        "view":view,"representation":representation,"deadline_clock":1,"deadline_tick":root.deadline_tick});
    let mut seal = Sha256::new();
    seal.update(b"rz-rovent-accepted-local-origin-seal/1\0");
    for value in [7_u64, 1, 1, 1, 1] {
        seal.update(value.to_le_bytes());
    }
    seal.update([1; 32]);
    seal.update([13; 32]);
    seal.update([14; 32]);
    seal.update(1_u64.to_le_bytes());
    seal.update([17; 32]);
    for byte in [25_u8, 26, 27, 28, 5, 6] {
        seal.update([byte; 32]);
    }
    seal.update(1_u64.to_le_bytes());
    seal.update(1_u64.to_le_bytes());
    seal.update([0, 0]);
    seal.update(root.deadline_tick.to_le_bytes());
    seal.update([0]);
    json!({"frame_sequence":7,"ticket":ticket,"kind":"local_accepted","peer":null,"output_sha256":h(17),"seal_sha256":format!("{:x}",seal.finalize())})
}
fn profile() -> RoventUtilityComparisonCohort {
    serde_json::from_value(json!({"domain":"rz-rovent-utility-comparison-cohort/1","input":{
        "input_lock_sha256":h(20),"source_sha256":h(3),"frozen_nnuec_model_sha256":h(4),"frozen_nnuec_namespace_sha256":h(5),
        "frozen_pc_model_sha256":h(6),"frozen_pc_namespace_sha256":h(7),"mode":"cpu_pc","requested_recipients":["proposer","critic"],
        "prelude_ns":10,"action_ns":200,"cleanup_ns":100,"logical_budget_sha256":h(8),"ready_policy_sha256":h(9),"cache_policy_sha256":h(10),
        "cost_class_sha256":h(11),"resource_plan_sha256":h(12),"resource_epoch":1,"usage":"collection_candidate","cost_scope":ROVENT_UTILITY_COHORT_COST_SCOPE}})).unwrap()
}
fn original_cost() -> Value {
    let baseline = Value::Object(
        COUNTERS
            .iter()
            .map(|key| ((*key).into(), json!(0)))
            .collect(),
    );
    let mut final_counters = baseline.clone();
    for key in [
        "native_preparations",
        "submitted_role_tickets",
        "physical_role_completions",
        "completed_nn_inputs",
        "consumed_role_outputs",
        "accepted_nn_inputs",
        "consumed_nn_inputs",
    ] {
        final_counters[key] = json!(1);
    }
    json!({"domain":"rz-rovent-utility-cohort-cost/1","scope":ROVENT_UTILITY_COHORT_COST_SCOPE,"objective":"exclusive_wall_nanoseconds",
        "cost_class_sha256":h(11),"cost_epoch_sha256":h(13),"resource_plan_sha256":h(12),"resource_epoch":1,"clock_epoch":1,
        "started_tick":20,"root_fence_tick":30,"actor_release_tick":40,"physical_pin_zero_tick":50,"whole_wall_ns":30,
        "original_deadline_tick":10000,"cohort_action_cutoff_tick":220,"cohort_cleanup_cutoff_tick":320,
        "baseline_counters":baseline,"final_counters":final_counters,"native_credit_baseline":[0,0],"native_credit_final":[0,0],
        "cpu_thread_time_ns":null,"energy":null,"complete_scope":true})
}
fn physical(root: &RoventRootAdmissionEvidence, model: &v2::Model) -> Value {
    json!({"root_fence_receipt":{"authority":authority(root),"native_owner_joined":true,"native_owner_quiesced":false,"native_physical_shutdown_confirmed":true,
        "own_cpu_joined":1,"expected_own_cpu_owners":1,"own_cpu_panicked":0,"external_shutdown_confirmed":null,"external_owner_quiesced":null,
        "auxiliary_shutdown_confirmed":true,"native_cpu_spent":0,"native_cpu_reserved":0,"pending_tickets":0,"quarantined":false,
        "request_sequence_high_water":1,"permits_semantic_closure":true},"review_pin_count":0,
        "engine_release":{"owner":0,"actual_stack_owner_id":1,"profile":{"owner":0,"max_depth":16,"evaluation_namespace_sha256":h(29),"search_conditions_sha256":h(30)},
            "binding_sha256":h(29),"model_identity":model.identity(),"preparation_epoch":1,"release_duration_ns":"10",
            "provenance":{"root":1,"generation":1,"sequence":1,"full_view_sha256":h(31),"view_seal_sha256":h(32),"record_scope_sha256":h(33),"policy_sha256":h(34),"model":model.identity()}},
        "context_cache_release":{"epoch":1,"matches_engine_release":true},"input_cache_prelude":{"epoch":1,"matches_input":true},
        "native_product":{"original_lifetime_present":true,"first_product_root":true}})
}

#[test]
fn original_rules_basis_consumed_source_and_own_clock_whole_cost_join() {
    let (position, root, model, mut semantic, line) = known_rules_case();
    let origin = original_origin(&root);
    verify_origin(&origin, &root, 1).unwrap();
    semantic["baseline_witnesses"] = json!([]);
    semantic["dependencies"] = json!([]);
    semantic["native_executions"] = json!([]);
    semantic["pc_consumption"] = json!([{"origin":origin,"question_sha256":h(13),"view":origin["ticket"]["view"],"full_view_sha256":h(19)}]);
    let body = json!({"recipient":"route_p","action":{"frame_sequence":7,"question_sha256":h(13)},
        "prelude":{"offers":[{"recipient":"route_p","budget":{"original_store_byte_limit":16_u64<<20}}]},"semantic":semantic});
    let trial = json!({"physical_closure":physical(&root,&model)});
    let cost = verify_cost(&original_cost(), &profile()).unwrap();
    let closed = verify_physical(&trial, &cost, &model, Some(&root)).unwrap();
    let observed = verify_semantic(&body, &root, &position, &closed, &model).unwrap();
    assert!(observed.bodies_complete);
    assert_eq!(
        observed.witnesses,
        BTreeSet::from([witness_identity(&position, &line, true)])
    );
    for axis in [
        "physical_authority",
        "clock",
        "fence",
        "consumed",
        "output",
        "wall_cost",
    ] {
        let mut changed_trial = trial.clone();
        let mut changed_cost = original_cost();
        let mut changed_body = body.clone();
        match axis {
            "physical_authority" => {
                changed_trial["physical_closure"]["root_fence_receipt"]["authority"]["process_epoch"] =
                    json!(2)
            }
            "clock" => changed_cost["clock_epoch"] = json!(2),
            "fence" => {
                changed_trial["physical_closure"]["root_fence_receipt"]["native_physical_shutdown_confirmed"] =
                    json!(false)
            }
            "consumed" => changed_body["semantic"]["pc_consumption"] = json!([]),
            "output" => {
                changed_body["semantic"]["pc_consumption"][0]["origin"]["output_sha256"] =
                    json!(h(99))
            }
            "wall_cost" => changed_cost["whole_wall_ns"] = json!(31),
            _ => unreachable!(),
        }
        let result = (|| {
            let cost = verify_cost(&changed_cost, &profile())?;
            let physical = verify_physical(&changed_trial, &cost, &model, Some(&root))?;
            verify_semantic(&changed_body, &root, &position, &physical, &model)
        })();
        assert!(result.is_err(), "accepted changed {axis}");
    }
}

#[test]
fn duplicate_json_keys_and_crossed_target_masks_are_never_accepted() {
    assert!(decode::<Value>(br#"{"declared":1,"declared":2}"#).is_err());
    assert!(mask(&json!([false, false, false, false, true, false])).is_err());
    assert!(mask(&json!([false, false, false, false, 1, false, false])).is_err());
    assert!(fields(&json!({"known":true,"unknown":false}), &["known"]).is_err());
}

#[test]
fn submitted_input_keeps_the_original_selected_view_and_root_provenance() {
    let root = root(&Position::startpos());
    let selected = json!({
        "view": {
            "sequence": 7,
            "view_digest": h(14),
            "context_policy_sha256": h(18)
        },
        "full_view_sha256": h(19)
    });
    let input = json!({"provenance": {
        "root_state": root.store_root_state,
        "generation": root.root_generation,
        "sequence": 7,
        "view_seal_sha256": h(14),
        "full_view_sha256": h(19),
        "policy_sha256": h(18)
    }});
    verify_submitted_input_source(&input, &selected, &root).unwrap();
    for key in [
        "root_state",
        "generation",
        "sequence",
        "view_seal_sha256",
        "full_view_sha256",
        "policy_sha256",
    ] {
        let mut changed = input.clone();
        changed["provenance"][key] = if changed["provenance"][key].is_number() {
            json!(99)
        } else {
            json!(h(99))
        };
        assert!(
            verify_submitted_input_source(&changed, &selected, &root).is_err(),
            "accepted replaced original provenance {key}"
        );
    }
}

#[test]
fn submission_seal_keeps_original_revision_state_and_u16_rebase_framing() {
    // Historical framing operands only; this creates no live ReviewView or
    // native acceptance. Spell the producer bytes independently below.
    let mut metadata = vec![
        RecordMetadata {
            anchor: 0,
            parent: None,
            supersedes: None,
            source_state: 11,
            source_revision: 41,
            source_line_offset: 0,
            original_first_move: None,
        },
        RecordMetadata {
            anchor: 0,
            parent: Some(0),
            supersedes: None,
            source_state: 11,
            source_revision: 43,
            source_line_offset: 2,
            original_first_move: None,
        },
    ];
    let mut raw = b"rz-pals-review-view-seal/1\0".to_vec();
    raw.extend_from_slice(&[19_u8; 32]);
    for number in [7_u64, 1, 1] {
        raw.extend_from_slice(&number.to_le_bytes());
    }
    for (revision, state, offset) in [(41_u64, 11_u64, 0_u16), (43, 11, 2)] {
        raw.extend_from_slice(&revision.to_le_bytes());
        raw.extend_from_slice(&state.to_le_bytes());
        raw.extend_from_slice(&offset.to_le_bytes());
    }
    let provenance = json!({"full_view_sha256": h(19), "view_seal_sha256": sha(&raw),
        "sequence": 7, "generation": 1, "root_state": 1});
    verify_input_submission_seal(&provenance, &metadata).unwrap();
    metadata[1].source_revision = 44;
    assert!(verify_input_submission_seal(&provenance, &metadata).is_err());
    metadata[1].source_revision = 43;
    metadata[1].source_state = 12;
    assert!(verify_input_submission_seal(&provenance, &metadata).is_err());
    metadata[1].source_state = 11;
    metadata[1].source_line_offset = 3;
    assert!(verify_input_submission_seal(&provenance, &metadata).is_err());
    assert!(
        serde_json::from_value::<RecordMetadata>(json!({"anchor": 0, "parent": null,
        "supersedes": null, "source_state": 11, "source_revision": 43,
        "source_line_offset": 65536, "original_first_move": null}))
        .is_err()
    );
}
