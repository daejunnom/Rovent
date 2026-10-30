use super::*;
use rz_contracts::rovent::ROVENT_SEARCH_REVISION;
use rz_experiments::{
    ROVENT_EXECUTION_DOMAIN, ROVENT_POSITION_VISIT_UNIT, RoventExecutionManifest,
};
use serde_json::{Value, json};

fn artifact(name: &str) -> Value {
    json!({"path":format!("fixture/{name}"), "sha256":"a".repeat(64),
        "bytes":32, "source":"https://example.org/source-only-fixture", "license":"MIT"})
}
pub(super) fn wire(mode: ExecutionMode) -> Value {
    let native_r = mode.has_native_r();
    let gpu = mode.pc_compute() == rz_contracts::rovent::PcCompute::Accelerator;
    let model = json!({"semantics":virtual_review::MODEL_SEMANTICS,
        "encoding":virtual_review::ENCODING, "policy_head":ROVENT_NATIVE_POLICY_HEAD,
        "value_head":ROVENT_NATIVE_VALUE_HEAD, "implementation_sha256":"b".repeat(64),
        "weights":artifact("weights"), "provenance":artifact("model-provenance"), "diagnostic":false});
    let mode = match mode {
        ExecutionMode::HybridPcr => "hybrid_pcr",
        ExecutionMode::HybridPc => "hybrid_pc",
        ExecutionMode::CpuPcr => "cpu_pcr",
        ExecutionMode::CpuPc => "cpu_pc",
    };
    let mut pools =
        vec![json!({"actors":["manager","pc_caller"],"max_threads":1,"cpu_charge_millicpus":1000})];
    if native_r {
        pools.push(json!({"actors":["own_r"],"max_threads":1,"cpu_charge_millicpus":1000}));
    }
    json!({"domain":ROVENT_EXECUTION_DOMAIN,"contract_revision":ROVENT_SEARCH_REVISION,
        "run_id":"rovent-native-pure-admission","purpose":"validation",
        "engine":{"source_commit":"a".repeat(40),"source_components":{"fixture":"c".repeat(64)},
            "binary":artifact("rovent"), "mode":mode,
            "pc":{"proposer":model,"critic":model,"validator":null,
                "backend":if gpu {"ort_cuda"} else {"ort_cpu"},
                "device":if gpu {json!({"device":"cuda","ordinal":0,"registered_device":"fixture-declaration-only"})}
                    else {json!({"device":"cpu"})},
                "backend_configuration_sha256":"d".repeat(64),"runtime_artifacts":[artifact("runtime")],
                "fp32":true,"tf32":false,"batch_width":1,"ort_intra_threads":1,"ort_inter_threads":1,"ort_spinning":false},
            "researcher":if native_r {json!({"provider":"own","implementation_sha256":"e".repeat(64),
                "search_configuration_sha256":"f".repeat(64)})} else {Value::Null},
            "teachers":[],"nnuec":null,
            "search":{"node_unit":ROVENT_POSITION_VISIT_UNIT,"requested_nodes":null,"depth_plies":3,"mate_moves":null,
                "registered_max_chess_plies":64,"searchmoves":null,"search_configuration_sha256":"1".repeat(64),
                "root_state_and_history_sha256":"2".repeat(64),"rules_profile_sha256":"3".repeat(64)}},
        "resources":{"cpu_quota_millicpus":2000,"pools":pools,"memory_high_bytes":6_u64<<30,
            "memory_max_bytes":12_u64<<30,"gpu_devices":u32::from(gpu),"physical_nn_concurrency":1,"topology_policy":"shared-control/1"},
        "coordinator":{"policy_sha256":"4".repeat(64),"questions_max":64,"ancestor_depth_max":16,
            "waiting_jobs_max":16,"role_tickets_max":16,"cpu_credit_limit":if native_r {4096} else {0},
            "original_root_budget_sha256":"5".repeat(64),"original_deadline_seal_sha256":"6".repeat(64)},
        "wall_time_max_ms":60000,"cleanup_max_ms":30000,"output_bytes_max":1048576,"agent_renewable":true,
        "provenance":[artifact("source-provenance")]})
}
fn lock(wire: Value) -> RoventInputLock {
    RoventExecutionManifest::from_json(&wire.to_string())
        .unwrap()
        .lock()
        .unwrap()
}

#[test]
fn four_mode_topology_admission_is_declared_without_constructing_researchers() {
    for mode in [
        ExecutionMode::HybridPcr,
        ExecutionMode::HybridPc,
        ExecutionMode::CpuPcr,
        ExecutionMode::CpuPc,
    ] {
        let declaration = lock(wire(mode));
        assert_eq!(
            admitted_topology(&declaration).unwrap(),
            if mode.has_native_r() {
                RoventStartupTopology::OwnResearcher
            } else {
                RoventStartupTopology::RolesOnly
            }
        );
        assert_eq!(
            declaration.manifest.engine.researcher.is_some(),
            mode.has_native_r()
        );
    }
}

#[test]
fn native_admission_keeps_v_and_diagnostic_assets_out_even_in_validation() {
    let mut with_v = wire(ExecutionMode::CpuPc);
    with_v["engine"]["pc"]["validator"] = with_v["engine"]["pc"]["proposer"].clone();
    assert!(admitted_topology(&lock(with_v)).is_err());
    let mut diagnostic = wire(ExecutionMode::CpuPc);
    diagnostic["engine"]["pc"]["proposer"]["diagnostic"] = json!(true);
    diagnostic["engine"]["pc"]["critic"]["diagnostic"] = json!(true);
    assert!(admitted_topology(&lock(diagnostic)).is_err());
    let mut collection = wire(ExecutionMode::CpuPcr);
    collection["purpose"] = json!("collection");
    assert!(admitted_topology(&lock(collection)).is_err());
}

#[test]
fn different_pc_weights_or_heads_are_not_coerced_into_one_shared_native_graph() {
    let mut different_weights = wire(ExecutionMode::CpuPc);
    different_weights["engine"]["pc"]["critic"]["weights"]["sha256"] = json!("f".repeat(64));
    assert!(admitted_topology(&lock(different_weights)).is_err());
    let mut different_head = wire(ExecutionMode::CpuPc);
    different_head["engine"]["pc"]["proposer"]["policy_head"] = json!("future-head/1");
    different_head["engine"]["pc"]["critic"]["policy_head"] = json!("future-head/1");
    assert!(admitted_topology(&lock(different_head)).is_err());
}

#[test]
fn cloned_startup_control_retains_original_cancel_and_deadline() {
    let original = RoventStartupControl {
        until: Instant::now() + Duration::from_secs(1),
        cleanup_until: Instant::now() + Duration::from_secs(2),
        cancellation: Arc::new(AtomicBool::new(false)),
    };
    let clone = original.clone();
    assert_eq!(clone.until, original.until);
    assert_eq!(clone.cleanup_until, original.cleanup_until);
    assert!(Arc::ptr_eq(&clone.cancellation, &original.cancellation));
    original.cancellation.store(true, Ordering::Release);
    assert_eq!(
        clone.check().unwrap_err().detail,
        "Rovent startup was canceled"
    );
}

#[test]
fn native_snapshot_uses_original_model_namespace_without_claiming_first_inference() {
    let snapshot = RoventNativeStartupSnapshot {
        domain: ROVENT_NATIVE_STARTUP_DOMAIN,
        input_lock: Digest([1; 32]),
        process_epoch: 1,
        mode: ExecutionMode::CpuPc,
        topology: RoventStartupTopology::RolesOnly,
        weights: Digest([2; 32]),
        export_manifest: Digest([3; 32]),
        backend: Digest([4; 32]),
        runtime_library: Digest([5; 32]),
        runtime_bundle: None,
        native_configuration: Digest([6; 32]),
        startup_configuration: Digest([7; 32]),
        search_configuration: Digest([8; 32]),
        validator_loaded: false,
        physical_workers_created: 1,
        session_residency: PalsSessionResidency {
            graphs: Vec::new(),
            native_sessions: 0,
            layout: "fixture-no-sessions".into(),
            reader_initializer_bank: None,
            role_reader_weights_shared: None,
            native_resident_parameter_bytes: None,
            vram_peak_bytes: None,
        },
        neural_service_executed: false,
    };
    assert!(
        snapshot
            .role_model_identity()
            .starts_with("pals-native-virtual-v3:")
    );
    assert_eq!(
        snapshot.model_value_identity().unwrap().model_epoch,
        [2; 32]
    );
    assert!(snapshot.model_namespace().is_ok());
    assert!(!snapshot.neural_service_executed && !snapshot.validator_loaded);
}

#[test]
fn v2_native_sources_use_the_original_startup_pin_without_requiring_duplicate_provenance() {
    use super::pinned::NativeInputLock;
    use rz_eval::pals_onnx::virtual_review::VirtualReviewOwnedSources;
    use rz_experiments::{
        ArtifactRef, ROVENT_EXECUTION_V2_DOMAIN, RoventExecutionManifestV2, RoventInputLockV2,
    };

    // This is a real /2 lock, not a fabricated /1 projection. Its typed
    // startup document is deliberately absent from the common provenance.
    let mut declaration = wire(ExecutionMode::CpuPc);
    declaration["domain"] = json!(ROVENT_EXECUTION_V2_DOMAIN);
    declaration["startup_document"] = artifact("v2-startup");
    declaration["startup_configuration_sha256"] = json!("7".repeat(64));
    declaration["startup_search_configuration_sha256"] = json!("1".repeat(64));
    declaration["own_r_root_tie"] = json!({"kind":"not_applicable", "reason":"pc_only"});
    let lock = RoventExecutionManifestV2::from_json(&declaration.to_string())
        .unwrap()
        .lock()
        .unwrap();
    let startup: ArtifactRef = serde_json::from_value(artifact("v2-startup")).unwrap();
    assert!(!lock.manifest.provenance.contains(&startup));
    assert_eq!(
        lock.topology(None).unwrap(),
        RoventStartupTopology::RolesOnly
    );
    assert!(lock.admit_startup_artifact(&startup).is_ok());
    let mut wrong = startup.clone();
    wrong.sha256 = "f".repeat(64);
    assert!(lock.admit_startup_artifact(&wrong).is_err());

    fn original_sources<T: VirtualReviewOwnedSources>() {}
    original_sources::<RoventPinnedNativeSources<RoventInputLockV2>>();
}
