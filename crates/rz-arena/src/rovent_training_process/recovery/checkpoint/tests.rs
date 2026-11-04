//! Authored metadata fixtures only: no files, model, subprocess, or old run
//! authority are opened or created by these cases.
use super::*;

fn h(c: char) -> String {
    c.to_string().repeat(64)
}
fn counts(n: u64) -> BTreeMap<String, u64> {
    ROLES.map(|r| (r.into(), n)).into()
}
fn cost(n: u64) -> BTreeMap<String, u64> {
    COST_FIELDS.map(|r| (r.into(), n)).into()
}
fn pin(path: &str) -> RoventPublicationFile {
    RoventPublicationFile {
        path: path.into(),
        sha256: h('a'),
        bytes: 1,
    }
}
fn boundary() -> RoventCheckpointBoundary {
    let empty = BTreeSet::new();
    RoventCheckpointBoundary {
        completed_updates: counts(0),
        ancestor_updates: counts(0),
        model_state_sha256: h('1'),
        optimizer_state_sha256: h('2'),
        scheduler_state_sha256: h('3'),
        rng_state_sha256: h('4'),
        sampler_state_sha256: h('5'),
        retained_input_set_sha256: set_seal("input", &empty).unwrap(),
        retained_input_count: 0,
        retained_origin_set_sha256: set_seal("origin", &empty).unwrap(),
        retained_origin_count: 0,
        snapshot_cost: cost(20),
    }
}
fn publication() -> RoventCheckpointPublication {
    RoventCheckpointPublication {
        schema: ROVENT_CHECKPOINT_PUBLICATION.into(),
        execution_id: "execution-1".into(),
        window_id: "window-1".into(),
        collection_source_sha256: h('1'),
        collection_runtime_library_sha256: h('2'),
        trainer_source_sha256: h('3'),
        trainer_python_executable_sha256: h('4'),
        trainer_dependency_set_sha256: h('5'),
        training_model_semantics_sha256: h('d'),
        initialization_source: RoventCheckpointInitializationSource::Initialize {
            pc_checkpoint_sha256: h('6'),
            nnuec_checkpoint_sha256: h('7'),
        },
        collection_model_epoch: 3,
        collection_context_epoch: 4,
        training_model_epoch: 9,
        recipe_sha256: h('8'),
        stage_id: "common".into(),
        training_profile_sha256: h('9'),
        membership_sha256: h('b'),
        current_dataset_publication_sha256: h('c'),
        ancestor_dataset_publication_sha256: vec![],
        tensor: pin("tensor.pt"),
        checkpoint_retirement: pin("checkpoint.json"),
        update_receipts: vec![],
        terminal_receipt: pin("terminal.json"),
        original_recipe: pin("recipe.json"),
        original_membership: pin("membership.json"),
        original_window_spec: pin("window.json"),
        parent_publication: None,
        boundary: boundary(),
        recorded_tail_cost: RoventRecordedTailCost::Unknown {
            reason: "original end cost was not observed".into(),
            evidence: None,
        },
        recorded_closure: RoventRecordedClosure {
            process_scope: "historical report".into(),
            physical_scope: "historical report".into(),
            evidence: pin("closure.json"),
        },
        r_auxiliary: RoventRAuxiliaryPresence::Absent,
    }
}
fn documents() -> (RoventCheckpointPublication, Value, Value, Value) {
    let mut p = publication();
    let recipe = json!({"schema":"rz-rovent-generation-series/1","model_semantics_sha256":h('d'),
        "stages":[{"stage_id":"common","parent_stage_id":null,"role_updates":counts(2)}]});
    p.recipe_sha256 = digest_value("rz-rovent-generation-series/1", &recipe).unwrap();
    let membership = json!({"schema":"rz-rovent-trainable-parameter-membership/2","groups":[]});
    p.membership_sha256 =
        digest_value("rz-rovent-trainable-parameter-membership/2", &membership).unwrap();
    let window = json!({"schema":"rz-rovent-train-window/1","recipe_sha256":p.recipe_sha256,
        "window_id":p.window_id,"stage_id":p.stage_id,"source_sha256":h('e'),"runtime_sha256":h('f'),
        "model_semantics_sha256":h('d'),"phase":"pc","frozen_pc_sha256":null,
        "max_updates":2,"max_groups":4,"max_samples":32,
        "loss_weights":{"policy":1,"wdl":1,"divergence":1,"task":1},"nnuec_loss":{"weight":1}});
    p.training_profile_sha256 = training_profile_digest(&window).unwrap();
    (p, recipe, membership, window)
}
fn ledger() -> TrainerLedger {
    let (p, recipe, _, window) = documents();
    let row = NativeRowAuthority {
        family: RoventPcRecordFamily::LegacyRecordV1,
        input_descriptor: json!({
            "schema":"rz-rovent-training-input-source-descriptor/1",
            "source_domain":rz_experiments::rovent_corpus::RECORD_DOMAIN,
            "input_domain":rz_experiments::pals_data_v3::GENERATION_INPUT_DOMAIN,
            "native_raw_domain":rz_experiments::rovent_corpus::NATIVE_RAW_DOMAIN,
            "input_sha256":h('2')
        }),
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
            teacher_actor_id: None,
            ancestor_origin_sha256: vec![],
        },
        role: "proposer".into(),
        retirement_sha256: h('5'),
        head_raw_sha256: BTreeMap::from([("policy".into(), h('6'))]),
        valid_heads: HEADS
            .map(|name| (name.into(), u64::from(name == "policy")))
            .into(),
    };
    let mut result = TrainerLedger::new(
        window,
        recipe,
        p.membership_sha256,
        h('8'),
        BTreeMap::from([(h('3'), row)]),
        counts(0),
        counts(0),
    )
    .unwrap();
    result.begun = true;
    result
}

#[test]
fn checkpoint_row_join_preserves_original_input_family() {
    // This deliberately authored private collision fixture is neither a
    // parsed corpus nor authority. A matching opaque seal cannot erase the
    // original source domain when joining two retained row owners.
    let source = ledger();
    let legacy = source.rows.values().next().unwrap();
    let mut submitted = legacy.clone();
    submitted.family = RoventPcRecordFamily::SubmittedInputV1;
    submitted.input_descriptor = json!({
        "schema":"rz-rovent-training-input-source-descriptor/1",
        "source_domain":rz_experiments::rovent_submitted_input::ROVENT_TRAINING_RECORD_V2_DOMAIN,
        "input_domain":rz_experiments::rovent_submitted_input::ROVENT_SUBMITTED_INPUT_DOMAIN,
        "native_raw_domain":rz_experiments::rovent_submitted_input::ROVENT_NATIVE_RAW_V3_DOMAIN,
        "input_sha256":legacy.origin.input_sha256,
        "attack_mapping":[]
    });
    assert!(!same_row(legacy, &submitted));
    assert!(same_row(legacy, &legacy.clone()));
}

fn update_receipt(l: &TrainerLedger, index: usize, n: u64) -> Value {
    let update = json!({"schema":"rz-rovent-trainer-update/1","event":{"schema":"rz-rovent-original-learner-update/1",
        "window_id":"window-1","stage_id":"common","role":"proposer","completed_role_updates":l.completed["proposer"]+1,
        "profile_sha256":l.profile_sha256,"corpus_sha256":l.corpus_sha256,"phase":"pc",
        "sample_origins":vec![h('3');8],"sample_inputs":vec![h('2');8],
        "head_counts":{"policy":8,"wdl":0,"divergence":0,"task":0,"nnuec_wdl":0},"actual_model_state_sha256":h('1')},
        "model_state_sha256":h('1'),"optimizer_state_sha256":h('2'),"scheduler_state_sha256":h('3'),
        "rng_state_sha256":h('4'),"sampler_state_sha256":h('5'),"cost":cost(n),"pending_microbatches":0});
    json!({"schema":ROVENT_ORIGINAL_UPDATE,"execution_id":"execution-1","window_id":"window-1",
        "sequence":index+1,"original_native_lineage_verified":true,"update":update})
}

fn auxiliary_recipe() -> RoventRAuxiliaryRecipe {
    RoventRAuxiliaryRecipe {
        schema: ROVENT_R_AUXILIARY_RECIPE.into(),
        manifest_file_sha256: h('a'),
        index_file_sha256: h('b'),
        sampling_seed: 29,
        rows_per_pc_update: 2,
        weight: 0.5,
        normalization: ROVENT_R_AUXILIARY_NORMALIZATION.into(),
        optimizer_owner: "nnuec-private".into(),
        supervision_source: ROVENT_R_AUXILIARY_SUPERVISION.into(),
    }
}
fn auxiliary_sidecar() -> RoventCheckpointRAuxiliary {
    RoventCheckpointRAuxiliary {
        schema: ROVENT_CHECKPOINT_R_AUXILIARY.into(),
        dataset_publication_sha256: h('c'),
        manifest_file_sha256: h('a'),
        index_file_sha256: h('b'),
        parent_ledger_projection: pin("r-auxiliary-ledger.json"),
    }
}

#[test]
fn checkpoint_r_auxiliary_preserves_absence_null_and_exact_sidecar_pins() {
    let p = publication();
    let absent = serde_json::to_value(&p).unwrap();
    assert!(!absent.as_object().unwrap().contains_key("r_auxiliary"));
    let decoded: RoventCheckpointPublication = serde_json::from_value(absent.clone()).unwrap();
    assert!(decoded.r_auxiliary.is_absent());
    assert_eq!(serde_json::to_value(decoded).unwrap(), absent);
    let mut null = absent.clone();
    null["r_auxiliary"] = Value::Null;
    let decoded: RoventCheckpointPublication = serde_json::from_value(null.clone()).unwrap();
    assert!(decoded.r_auxiliary.is_null());
    assert_eq!(serde_json::to_value(decoded).unwrap(), null);
    let mut enabled = absent;
    enabled["r_auxiliary"] = json!(auxiliary_sidecar());
    let decoded: RoventCheckpointPublication = serde_json::from_value(enabled.clone()).unwrap();
    decoded.validate().unwrap();
    assert_eq!(serde_json::to_value(decoded).unwrap(), enabled);
    enabled["r_auxiliary"]["native_authority"] = json!(true);
    assert!(serde_json::from_value::<RoventCheckpointPublication>(enabled).is_err());
}

#[test]
fn r_sidecar_artifact_aliases_and_unpinned_owner_identity_are_rejected() {
    let mut p = publication();
    let mut sidecar = auxiliary_sidecar();
    sidecar.parent_ledger_projection = p.checkpoint_retirement.clone();
    p.r_auxiliary = RoventRAuxiliaryPresence::Enabled(sidecar);
    assert!(p.validate().is_err());
    let mut sidecar = auxiliary_sidecar();
    sidecar.manifest_file_sha256 = "canonical-is-not-an-original-file-pin".into();
    p.r_auxiliary = RoventRAuxiliaryPresence::Enabled(sidecar);
    assert!(p.validate().is_err());
}

#[test]
fn r_lineage_union_keeps_old_r_seals_across_a_disabled_stage() {
    // A's original R seals and C's new R seals stay in the R namespace. The
    // disabled B stage contributes no R set, and its P/C sets are never read.
    let original_a = BTreeSet::from([h('a'), h('b')]);
    let original_c = BTreeSet::from([h('b'), h('c')]);
    let mut inherited = BTreeSet::new();
    union_r_seals(&mut inherited, &original_a).unwrap();
    union_r_seals(&mut inherited, &BTreeSet::new()).unwrap();
    union_r_seals(&mut inherited, &original_c).unwrap();
    assert_eq!(inherited, BTreeSet::from([h('a'), h('b'), h('c')]));
    assert!(!inherited.contains(&h('f'))); // unrelated P/C input seal
    let mut invalid = BTreeSet::new();
    assert!(union_r_seals(&mut invalid, &BTreeSet::from(["not-a-sha".into()])).is_err());
    assert!(invalid.is_empty());
}

#[test]
fn strict_publication_rejects_duplicate_unknown_fields_and_ambiguous_initialization() {
    let p = publication();
    p.validate().unwrap();
    let raw = serde_json::to_string(&p).unwrap();
    let duplicate = format!(
        "{{\"schema\":\"{}\",{}",
        ROVENT_CHECKPOINT_PUBLICATION,
        raw.strip_prefix('{').unwrap()
    );
    assert!(decode::<RoventCheckpointPublication>(duplicate.as_bytes()).is_err());
    let mut value = serde_json::to_value(&p).unwrap();
    value["tensor_validated"] = json!(true);
    assert!(serde_json::from_value::<RoventCheckpointPublication>(value).is_err());
    let old = json!({"kind":"initialize","training_initial_checkpoint_sha256":h('a')});
    assert!(serde_json::from_value::<RoventCheckpointInitializationSource>(old).is_err());
}
#[test]
fn publication_requires_unique_relative_artifacts_and_nonempty_unknown_reason() {
    let mut p = publication();
    p.recorded_closure.evidence = p.terminal_receipt.clone();
    assert!(p.validate().is_err());
    p = publication();
    p.tensor.path = "../tensor.pt".into();
    assert!(p.validate().is_err());
    p = publication();
    p.recorded_tail_cost = RoventRecordedTailCost::Unknown {
        reason: String::new(),
        evidence: None,
    };
    assert!(p.validate().is_err());
    p = publication();
    p.ancestor_dataset_publication_sha256 = vec![h('c'), h('a')];
    assert!(p.validate().is_err());
}
#[test]
fn original_v1_identity_bytes_are_not_relabelled_as_separate_trainer_runtime() {
    let (p, recipe, membership, window) = documents();
    assert_ne!(
        window["runtime_sha256"],
        p.collection_runtime_library_sha256
    );
    assert_ne!(window["source_sha256"], p.trainer_source_sha256);
    assert_eq!(
        validate_documents(&p, &recipe, &membership, &window).unwrap(),
        p.training_profile_sha256
    );
    let mut changed = window.clone();
    changed["runtime_sha256"] = json!(h('a'));
    assert!(validate_documents(&p, &recipe, &membership, &changed).is_err());
    changed = window;
    changed["schema"] = json!("rz-rovent-train-window/3");
    assert!(validate_documents(&p, &recipe, &membership, &changed).is_err());
}
fn v2_documents() -> (RoventCheckpointPublication, Value, Value, Value) {
    let (mut p, recipe, membership, _) = documents();
    let window = json!({"schema":ROVENT_TRAIN_WINDOW_V2,"window_id":p.window_id,"stage_id":p.stage_id,"recipe_sha256":p.recipe_sha256,
        "collection_source_sha256":p.collection_source_sha256,"collection_runtime_library_sha256":p.collection_runtime_library_sha256,
        "collection_model_semantics_sha256":p.training_model_semantics_sha256,
        "collection_model_epoch":p.collection_model_epoch,"collection_context_epoch":p.collection_context_epoch,
        "trainer_source_sha256":p.trainer_source_sha256,"trainer_python_executable_sha256":p.trainer_python_executable_sha256,
        "trainer_dependency_set_sha256":p.trainer_dependency_set_sha256,"trainer_dependency_scope":ROVENT_TRAINER_DEPENDENCY_SCOPE,
        "training_model_semantics_sha256":p.training_model_semantics_sha256,"training_model_epoch":9,
        "nnuec_collection_model_sha256":h('a'),"nnuec_semantics":"nnuec-v1","phase":"pc","frozen_pc_sha256":null,
        "device":"cpu","precision":"fp32","tf32":false,"deterministic":true,"cpu_threads":2,"interop_threads":1,
        "microbatch":1,"accumulation":8,
        "optimizer":{"name":"adamw","lr":1e-4,"betas":[0.9,0.999],"epsilon":1e-8,"weight_decay":0.01,"clip_norm":1.0,"schedule":"constant"},
        "sampling_seed":29,"training_seed":31,"loss_weights":{"policy":1.0,"wdl":1.0,"divergence":1.0,"task":1.0},
        "nnuec_loss":{"loss":"wdl_cross_entropy","target":"same_exact_rules_anchor","normalization":"valid_rows","weight":1.0},
        "max_updates":2,"max_groups":4,"max_samples":32,"max_wall_ns":1000000000,"max_output_bytes":1024,"checkpoint_interval":1});
    let typed: RoventTrainWindowV2 = serde_json::from_value(window.clone()).unwrap();
    p.training_profile_sha256 = typed.profile_sha256().unwrap();
    (p, recipe, membership, window)
}
#[test]
fn explicit_v2_joins_collection_trainer_and_separate_epochs() {
    let (p, recipe, membership, window) = v2_documents();
    let typed: RoventTrainWindowV2 = serde_json::from_value(window.clone()).unwrap();
    assert_ne!(typed.training_model_epoch, p.collection_model_epoch);
    assert_eq!(
        validate_documents(&p, &recipe, &membership, &window).unwrap(),
        p.training_profile_sha256
    );
    for key in [
        "trainer_python_executable_sha256",
        "trainer_dependency_set_sha256",
        "collection_runtime_library_sha256",
    ] {
        let mut wrong = window.clone();
        wrong[key] = json!(h('f'));
        assert!(validate_documents(&p, &recipe, &membership, &wrong).is_err());
    }
    let mut wrong = window;
    wrong["collection_context_epoch"] = json!(p.collection_context_epoch + 1);
    assert!(validate_documents(&p, &recipe, &membership, &wrong).is_err());
}
#[test]
fn enabled_checkpoint_needs_an_exact_external_r_dataset_owner() {
    let (mut p, recipe, _, mut window) = v2_documents();
    let budget = RoventRecoveryReadBudget::new(
        1024,
        Instant::now() + std::time::Duration::from_secs(1),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    p.r_auxiliary = RoventRAuxiliaryPresence::Enabled(auxiliary_sidecar());
    window["nnuec_r_auxiliary"] = json!(auxiliary_recipe());
    assert!(
        initial_r_auxiliary_ledger(&p, &window, None, &recipe, None, &BTreeMap::new(), &budget)
            .is_err()
    );
    assert_eq!(budget.observation()["charged_bytes"], 0);
    // A serialized sidecar and an invented native-success flag cannot issue
    // the missing opaque owner, and no file/NN/process work was attempted.
    assert!(p.r_auxiliary.enabled().is_some());
}

#[test]
fn explicit_null_publication_keeps_original_disabled_retirement_bytes() {
    let (mut p, recipe, _, mut window) = v2_documents();
    p.r_auxiliary = RoventRAuxiliaryPresence::Null;
    window["nnuec_r_auxiliary"] = Value::Null;
    let budget = RoventRecoveryReadBudget::new(
        1024,
        Instant::now() + std::time::Duration::from_secs(1),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert!(
        initial_r_auxiliary_ledger(&p, &window, None, &recipe, None, &BTreeMap::new(), &budget)
            .unwrap()
            .is_none()
    );
    let mut state = p.boundary.state_value();
    state["schema"] = json!("rz-rovent-trainer-checkpoint/2");
    state["checkpoint"] =
        json!({"path":"/old/tensor.pt","sha256":p.tensor.sha256,"bytes":p.tensor.bytes});
    let receipt = json!({"schema":"rz-rovent-recovered-checkpoint-retirement/1",
        "execution_id":p.execution_id,"window_id":p.window_id,"source_kind":ROVENT_RECOVERY_SOURCE_KIND,
        "past_native_authority_reissued":false,"checkpoint":state});
    validate_retirement(&p, &receipt, None, ROVENT_TRAIN_WINDOW_V2).unwrap();
    assert!(receipt["checkpoint"].get("r_auxiliary").is_none());
    let mut changed = receipt;
    changed["checkpoint"]["r_auxiliary"] = Value::Null;
    assert!(validate_retirement(&p, &changed, None, ROVENT_TRAIN_WINDOW_V2).is_err());
}
#[test]
fn initialize_cannot_skip_a_declared_recipe_parent() {
    let (p, mut recipe, _, _) = documents();
    initial_counts(&p, &recipe, None).unwrap();
    recipe["stages"][0]["parent_stage_id"] = json!("earlier");
    assert!(initial_counts(&p, &recipe, None).is_err());
    recipe["stages"][0]
        .as_object_mut()
        .unwrap()
        .remove("parent_stage_id");
    assert!(initial_counts(&p, &recipe, None).is_err());
}
#[test]
fn snapshot_is_a_cost_barrier_before_later_updates() {
    let p = publication();
    let mut l = ledger();
    let first = update_receipt(&l, 0, 10);
    validate_update(&p, &mut l, &first, 0).unwrap();
    let mut b = p.boundary.clone();
    b.completed_updates = l.completed.clone();
    let captured = boundary_snapshot(&mut l, &b).unwrap().unwrap();
    assert_eq!(l.last_cost, Some(json!(cost(20))));
    let bad = update_receipt(&l, 1, 15);
    assert!(validate_update(&p, &mut l, &bad, 1).is_err());
    assert_eq!(captured.2.unwrap()["event"]["completed_role_updates"], 1);
    assert_eq!(l.completed["proposer"], 1);
}
#[test]
fn historical_update_checks_original_row_and_sequence_before_committing() {
    let p = publication();
    let mut l = ledger();
    let mut raw = update_receipt(&l, 0, 10);
    raw["sequence"] = json!(2);
    assert!(validate_update(&p, &mut l, &raw, 0).is_err());
    assert!(l.pending.is_empty());
    raw = update_receipt(&l, 0, 10);
    raw["update"]["event"]["sample_origins"][0] = json!(h('f'));
    assert!(validate_update(&p, &mut l, &raw, 0).is_err());
    assert!(l.updates.is_empty());
}
#[test]
fn recovered_receipts_require_explicit_v2_domains_and_no_reissued_native_flag() {
    let p = publication();
    let mut l = ledger();
    l.window["schema"] = json!(ROVENT_TRAIN_WINDOW_V2);
    let old = update_receipt(&l, 0, 10);
    assert!(validate_update(&p, &mut l, &old, 0).is_err());
    let mut new = old;
    new["schema"] = json!("rz-rovent-recovered-learner-completion/1");
    new.as_object_mut()
        .unwrap()
        .remove("original_native_lineage_verified");
    new["source_kind"] = json!(ROVENT_RECOVERY_SOURCE_KIND);
    new["past_native_authority_reissued"] = json!(false);
    new["update"]["schema"] = json!("rz-rovent-trainer-update/2");
    new["update"]["event"]["schema"] = json!("rz-rovent-recovered-learner-update/1");
    validate_update(&p, &mut l, &new, 0).unwrap();
    assert_eq!(l.completed["proposer"], 1);
    let mut state = p.boundary.state_value();
    state["schema"] = json!("rz-rovent-trainer-checkpoint/2");
    state["checkpoint"] =
        json!({"path":"/old/checkpoint.pt","sha256":p.tensor.sha256,"bytes":p.tensor.bytes});
    let raw = json!({"schema":"rz-rovent-recovered-checkpoint-retirement/1","execution_id":p.execution_id,
        "window_id":p.window_id,"source_kind":ROVENT_RECOVERY_SOURCE_KIND,"past_native_authority_reissued":false,"checkpoint":state});
    validate_retirement(&p, &raw, None, ROVENT_TRAIN_WINDOW_V2).unwrap();
    assert!(validate_retirement(&p, &raw, None, "rz-rovent-train-window/1").is_err());
    let mut wrong = raw;
    wrong["past_native_authority_reissued"] = json!(true);
    assert!(validate_retirement(&p, &wrong, None, ROVENT_TRAIN_WINDOW_V2).is_err());
}
#[test]
fn retained_set_seals_bind_kind_full_membership_and_count() {
    let set = BTreeSet::from([h('a'), h('b')]);
    assert_ne!(
        set_seal("input", &set).unwrap(),
        set_seal("origin", &set).unwrap()
    );
    let mut b = boundary();
    b.retained_input_count = 2;
    b.retained_origin_count = 2;
    b.retained_input_set_sha256 = set_seal("input", &set).unwrap();
    b.retained_origin_set_sha256 = set_seal("origin", &set).unwrap();
    validate_boundary_sets(&b, &set, &set).unwrap();
    assert!(validate_boundary_sets(&b, &BTreeSet::from([h('a')]), &set).is_err());
    b.retained_origin_count = 1;
    assert!(validate_boundary_sets(&b, &set, &set).is_err());
}
#[test]
fn retirement_binds_five_states_counts_and_complete_accumulation() {
    let p = publication();
    let mut state = p.boundary.state_value();
    state["schema"] = json!("rz-rovent-trainer-checkpoint/1");
    state["checkpoint"] =
        json!({"path":"/old/host/checkpoint.pt","sha256":p.tensor.sha256,"bytes":p.tensor.bytes});
    let raw = json!({"schema":ROVENT_ORIGINAL_CHECKPOINT,"execution_id":p.execution_id,"window_id":p.window_id,"checkpoint":state});
    validate_retirement(&p, &raw, None, "rz-rovent-train-window/1").unwrap();
    for key in STATE_HASHES {
        let mut wrong = raw.clone();
        wrong["checkpoint"][key] = json!(h('f'));
        assert!(validate_retirement(&p, &wrong, None, "rz-rovent-train-window/1").is_err());
    }
    let mut wrong = raw.clone();
    wrong["checkpoint"]["pending_microbatches"] = json!(1);
    assert!(validate_retirement(&p, &wrong, None, "rz-rovent-train-window/1").is_err());
    wrong = raw;
    wrong["checkpoint"]["completed_updates"]["critic"] = json!(1);
    assert!(validate_retirement(&p, &wrong, None, "rz-rovent-train-window/1").is_err());
}
#[test]
fn known_tail_is_exact_eight_counter_difference_without_refunds() {
    let p = publication();
    let delta = cost(10);
    let value = json!({"schema":ROVENT_RECORDED_TAIL_COST,"execution_id":p.execution_id,"window_id":p.window_id,
        "snapshot_cost":p.boundary.snapshot_cost,"end_cost":cost(30),"delta":delta});
    validate_tail_value(&p, &delta, &value, Some(&json!(cost(25)))).unwrap();
    let mut wrong = value.clone();
    wrong["end_cost"]["replay_ns"] = json!(19);
    assert!(validate_tail_value(&p, &delta, &wrong, None).is_err());
    wrong = value.clone();
    wrong["delta"]["masked_groups"] = json!(0);
    assert!(validate_tail_value(&p, &delta, &wrong, None).is_err());
    assert!(validate_tail_value(&p, &delta, &value, Some(&json!(cost(31)))).is_err());
    let mut missing = delta.clone();
    missing.remove("output_bytes");
    assert!(cost_map(&missing).is_err());
}
fn closure(p: &RoventCheckpointPublication) -> RoventOriginalTrainingClosure {
    RoventOriginalTrainingClosure {
        schema: ROVENT_ORIGINAL_TRAINING_CLOSURE.into(),
        execution_id: p.execution_id.clone(),
        window_id: p.window_id.clone(),
        terminal_receipt_sha256: p.terminal_receipt.sha256.clone(),
        terminal_receipt_bytes: p.terminal_receipt.bytes,
        process_receipt: Some(crate::process::ProcessReceipt {
            supervisor_version: 1,
            pid: 123,
            elapsed_ns: 100,
            cleanup_elapsed_ns: Some(1),
            cleanup_timing_unavailable_reason: None,
            stop: crate::process::ProcessStop::Exited,
            exit_code: Some(0),
            exit_signal: None,
            group_cleanup: crate::process::CleanupStatus::Gone,
            stdout_bytes: 0,
            stderr_bytes: 0,
            observed_output_bytes: 0,
            descendant_cleanup_required: false,
            child_limit_enforcement: "fixture".into(),
            watched_artifact_bytes: Some(1),
            artifact_limit_enforcement: None,
            errors: vec![],
        }),
        child_identity: RoventHistoricalChildIdentity {
            boot_id: "12345678-1234-1234-1234-123456789abc".into(),
            pid: 123,
            start_ticks: 456,
            cgroup_path: "/rovent/run-1".into(),
            cgroup_device: 1,
            cgroup_inode: 2,
        },
        child_reap: RoventHistoricalChildReap::OwnedChildReaped,
        physical_completion: RoventHistoricalPhysicalCompletion::Confirmed,
    }
}
#[test]
fn closure_is_historical_classification_and_unknown_is_never_closed() {
    let p = publication();
    let terminal = json!({"physical_completion":"confirmed"});
    let c = closure(&p);
    assert_eq!(
        validate_closure(&p, &c, &terminal).unwrap(),
        RoventHistoricalClosureClassification::RecordedClosed
    );
    let mut changed = c.clone();
    changed.child_reap = RoventHistoricalChildReap::Unresolved;
    assert_eq!(
        validate_closure(&p, &changed, &terminal).unwrap(),
        RoventHistoricalClosureClassification::Unresolved
    );
    changed = c.clone();
    changed.physical_completion = RoventHistoricalPhysicalCompletion::Unknown;
    assert_eq!(
        validate_closure(&p, &changed, &terminal).unwrap(),
        RoventHistoricalClosureClassification::Unresolved
    );
    changed = c.clone();
    changed.child_identity.pid += 1;
    assert!(validate_closure(&p, &changed, &terminal).is_err());
    changed = c;
    changed.terminal_receipt_bytes += 1;
    assert!(validate_closure(&p, &changed, &terminal).is_err());
}
#[test]
fn boundary_only_report_cannot_fabricate_terminal_or_process_success() {
    let mut p = publication();
    p.recorded_tail_cost = RoventRecordedTailCost::Unknown {
        reason: "original_window_unclosed_after_boundary".into(),
        evidence: None,
    };
    let mut c = closure(&p);
    c.schema = ROVENT_CHECKPOINT_BOUNDARY_CLOSURE.into();
    c.process_receipt = None;
    c.child_reap = RoventHistoricalChildReap::Unresolved;
    c.physical_completion = RoventHistoricalPhysicalCompletion::Unknown;
    let report = json!({"schema":ROVENT_CHECKPOINT_BOUNDARY_REPORT,"execution_id":p.execution_id,"window_id":p.window_id,
        "completed_updates":p.boundary.completed_updates,"physical_completion":"unknown","arena_eligible":false,
        "status":"checkpoint_boundary_only","checkpoint_retirement_sha256":p.checkpoint_retirement.sha256});
    validate_terminal(&p, &report, &ledger(), &c).unwrap();
    assert_eq!(
        validate_closure(&p, &c, &report).unwrap(),
        RoventHistoricalClosureClassification::Unresolved
    );
    let mut wrong = report.clone();
    wrong["status"] = json!("completed");
    assert!(validate_terminal(&p, &wrong, &ledger(), &c).is_err());
    wrong = report.clone();
    wrong["checkpoint_retirement_sha256"] = json!(h('f'));
    assert!(validate_terminal(&p, &wrong, &ledger(), &c).is_err());
    c.process_receipt = closure(&p).process_receipt;
    assert!(validate_closure(&p, &c, &report).is_err());
    c.process_receipt = None;
    let mut value = serde_json::to_value(&c).unwrap();
    value.as_object_mut().unwrap().remove("process_receipt");
    assert!(parse_recorded_closure(value).is_err());
    p.recorded_tail_cost = RoventRecordedTailCost::Known {
        delta: cost(0),
        evidence: pin("tail.json"),
    };
    assert!(validate_terminal(&p, &report, &ledger(), &c).is_err());
}
