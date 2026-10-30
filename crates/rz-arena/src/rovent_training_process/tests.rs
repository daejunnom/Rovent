//! Pure authority-ledger cases. Authored row seals below are private test
//! fixtures; they are not native corpus, model, optimizer or process evidence.
use super::*;

fn h(c: char) -> String {
    c.to_string().repeat(64)
}
fn counts() -> BTreeMap<String, u64> {
    ROLES.map(|r| (r.to_string(), 0)).into()
}
fn ledger(masked: bool) -> TrainerLedger {
    let window = json!({"window_id":"window-1","stage_id":"common","recipe_sha256":h('a'),"source_sha256":h('b'),"runtime_sha256":h('c'),
        "model_semantics_sha256":h('d'),"frozen_pc_sha256":null,"phase":"pc","max_updates":2,"max_groups":4,"max_samples":32,
        "loss_weights":{"policy":1,"wdl":1,"divergence":1,"task":1},"nnuec_loss":{"weight":1}});
    let recipe = json!({"stages":[{"stage_id":"common","parent_stage_id":null,"teacher_actor_ids":["sf","reckless"],"role_updates":{"proposer":2,"critic":2,"validator":1}}]});
    let row = NativeRowAuthority {
        family: RoventPcRecordFamily::LegacyRecordV1,
        input_descriptor: json!({"schema":"rz-rovent-training-input-source-descriptor/1",
            "source_domain":rz_experiments::rovent_corpus::RECORD_DOMAIN,"input_domain":rz_experiments::pals_data_v3::GENERATION_INPUT_DOMAIN,
            "native_raw_domain":rz_experiments::rovent_corpus::NATIVE_RAW_DOMAIN,"input_sha256":h('2')}),
        origin: RoventCorpusOrigin {
            record_sha256: h('1'),
            input_sha256: h('2'),
            origin_sha256: h('3'),
            split: GenerationSplitV3::Train,
            game_id: "game-1".into(),
            opening_id: "opening-1".into(),
            line_genealogy_id: "line-1".into(),
            transposition_sha256: h('4'),
            actor_id: "actor-1".into(),
            teacher_actor_id: Some("sf".into()),
            ancestor_origin_sha256: vec![],
        },
        role: "proposer".into(),
        retirement_sha256: h('5'),
        head_raw_sha256: BTreeMap::from([("policy".into(), h('6'))]),
        valid_heads: BTreeMap::from(
            HEADS.map(|name| (name.into(), u64::from(name == "policy" && !masked))),
        ),
    };
    TrainerLedger::new(
        window,
        recipe,
        h('7'),
        h('8'),
        BTreeMap::from([(h('3'), row)]),
        counts(),
        counts(),
    )
    .unwrap()
}

#[test]
fn submitted_family_admission_keeps_original_sha_and_all_128_attack_descriptors() {
    // Private metadata fixture only: no PublishedRecord/corpus/native/process
    // capability or numerical input is constructed by this ledger test.
    let mut l = ledger(false);
    let attacks = (0..128)
        .map(|slot| {
            json!({"slot":slot,"attack_ply":slot,
        "selected_counter_local_index":null,"selected_counter_revision":null})
        })
        .collect::<Vec<_>>();
    let row = l.rows.get_mut(&h('3')).unwrap();
    row.family = RoventPcRecordFamily::SubmittedInputV1;
    row.input_descriptor = json!({"schema":"rz-rovent-training-input-source-descriptor/1",
        "source_domain":rz_experiments::rovent_submitted_input::ROVENT_TRAINING_RECORD_V2_DOMAIN,
        "input_domain":rz_experiments::rovent_submitted_input::ROVENT_SUBMITTED_INPUT_DOMAIN,
        "native_raw_domain":rz_experiments::rovent_submitted_input::ROVENT_NATIVE_RAW_V3_DOMAIN,
        "input_sha256":h('2'),"attack_mapping":attacks});
    row.validate_source_descriptor().unwrap();
    let begin = begin(&l);
    l.begin(&begin).unwrap();
    let mut request = consume();
    request["schema"] = json!("rz-rovent-trainer-consume/2");
    let original = request.clone();
    l.consume_with_schema(&request, "rz-rovent-trainer-consume/2")
        .unwrap();
    assert_eq!(request, original);
    assert_eq!(request.as_object().unwrap().len(), 8);
    assert_eq!(
        l.rows[&h('3')].input_descriptor["attack_mapping"]
            .as_array()
            .unwrap()
            .len(),
        128
    );
    l.rows.get_mut(&h('3')).unwrap().input_descriptor["input_domain"] =
        json!(rz_experiments::pals_data_v3::GENERATION_INPUT_DOMAIN);
    assert!(
        l.consume_with_schema(&request, "rz-rovent-trainer-consume/2")
            .is_err()
    );
}
fn begin(l: &TrainerLedger) -> Value {
    json!({"schema":"rz-rovent-trainer-begin/1","recipe_sha256":l.window["recipe_sha256"],"stage_id":"common",
    "training_profile_sha256":l.profile_sha256,"corpus_sha256":l.corpus_sha256,"source_sha256":l.window["source_sha256"],
    "runtime_sha256":l.window["runtime_sha256"],"completed_updates":counts(),"ancestor_updates":counts(),"frozen_pc_sha256":null})
}
fn consume() -> Value {
    json!({"schema":"rz-rovent-trainer-consume/1","input_sha256":h('2'),"origin_sha256":h('3'),"record_sha256":h('1'),
    "native_retirement_sha256":h('5'),"head_raw_sha256":{"policy":h('6')},"role":"proposer","split":"train"})
}
fn update(l: &TrainerLedger) -> Value {
    json!({"schema":"rz-rovent-trainer-update/1","event":{"schema":"rz-rovent-original-learner-update/1",
    "window_id":"window-1","stage_id":"common","role":"proposer","completed_role_updates":l.completed["proposer"]+1,
    "profile_sha256":l.profile_sha256,"corpus_sha256":l.corpus_sha256,"phase":"pc","sample_origins":vec![h('3');8],"sample_inputs":vec![h('2');8],
    "head_counts":{"policy":8,"wdl":0,"divergence":0,"task":0,"nnuec_wdl":0},"actual_model_state_sha256":h('9')},
    "model_state_sha256":h('9'),"optimizer_state_sha256":h('a'),"scheduler_state_sha256":h('b'),"rng_state_sha256":h('c'),
    "sampler_state_sha256":h('d'),"cost":COST_FIELDS.map(|name|(name,8_u64)).into_iter().collect::<BTreeMap<_,_>>(),"pending_microbatches":0})
}

#[test]
fn original_begin_is_single_use_and_binds_fixed_profile_and_counts() {
    let mut l = ledger(false);
    let mut wrong = begin(&l);
    wrong["training_profile_sha256"] = json!(h('f'));
    assert!(l.begin(&wrong).is_err());
    assert!(!l.begun);
    l.begin(&begin(&l)).unwrap();
    assert!(l.begin(&begin(&l)).is_err());
}
#[test]
fn raw_or_foreign_sample_cannot_create_native_retirement_authority() {
    let mut l = ledger(false);
    l.begin(&begin(&l)).unwrap();
    for field in ["input_sha256", "record_sha256", "native_retirement_sha256"] {
        let mut wrong = consume();
        wrong[field] = json!(h('f'));
        assert!(l.consume(&wrong).is_err());
        assert!(l.pending.is_empty());
    }
    let mut wrong = consume();
    wrong["split"] = json!("holdout");
    assert!(l.consume(&wrong).is_err());
    l.rows.clear();
    assert!(l.consume(&consume()).is_err());
}
#[test]
fn only_eight_original_same_role_samples_can_commit_one_update() {
    let mut l = ledger(false);
    l.begin(&begin(&l)).unwrap();
    for _ in 0..7 {
        l.consume(&consume()).unwrap();
    }
    assert!(l.update(&update(&l)).is_err());
    assert_eq!(l.completed["proposer"], 0);
    l.consume(&consume()).unwrap();
    let valid = update(&l);
    let mut wrong = valid.clone();
    wrong["event"]["head_counts"]["policy"] = json!(7);
    assert!(l.update(&wrong).is_err());
    assert_eq!(l.pending.len(), 8);
    l.update(&valid).unwrap();
    assert_eq!(l.completed["proposer"], 1);
    assert!(l.pending.is_empty());
    assert!(l.update(&valid).is_err());
    assert_eq!(l.updates.len(), 1);
}
#[test]
fn all_masked_groups_cost_samples_but_never_create_optimizer_updates() {
    let mut l = ledger(true);
    l.begin(&begin(&l)).unwrap();
    for _ in 0..8 {
        l.consume(&consume()).unwrap();
    }
    assert!(l.pending.is_empty());
    assert_eq!(l.masked_groups, 1);
    assert_eq!(l.samples_admitted, 8);
    assert!(l.update(&update(&l)).is_err());
    assert_eq!(l.completed["proposer"], 0);
}
#[test]
fn pending_partial_group_and_same_recipe_window_budget_are_not_refunded() {
    let mut l = ledger(false);
    l.window["max_samples"] = json!(3);
    l.begin(&begin(&l)).unwrap();
    for _ in 0..3 {
        l.consume(&consume()).unwrap();
    }
    assert!(l.consume(&consume()).is_err());
    assert_eq!(l.samples_admitted, 3);
    assert_eq!(l.pending.len(), 3);
    let mut l = ledger(false);
    l.window["phase"] = json!("v");
    l.profile_sha256 = training_profile_digest(&l.window).unwrap();
    l.begin(&begin(&l)).unwrap();
    for _ in 0..8 {
        l.consume(&consume()).unwrap();
    }
    assert!(l.update(&update(&l)).is_err());
    assert_eq!(l.completed["proposer"], 0);
}
