use super::*;
use crate::rovent_root_allocation::tests::{artifact, digest_hex, fixture, sha_value};
use serde_json::{Value, json};

fn cpu_collection_root_fixture() -> (crate::RoventInputLockV3, Value, Value) {
    use crate::pals_data_v3::generation_digest;
    let (old, old_descriptor, mut allocation_input) = fixture();
    let mut wire = serde_json::to_value(&old.manifest).unwrap();
    wire["domain"] = json!(crate::ROVENT_EXECUTION_V3_DOMAIN);
    wire["purpose"] = json!("collection");
    wire["engine"]["pc"]["validator"] = wire["engine"]["pc"]["proposer"].clone();
    wire["engine"]["pc"]["backend"] = json!("ort_cpu");
    wire["engine"]["pc"]["runtime_artifacts"] = json!([artifact("cpu-ort")]);
    wire["engine"]["pc"]["ort_intra_threads"] = json!(1);
    wire["engine"]["pc"]["ort_inter_threads"] = json!(1);
    wire["engine"]["pc"]["ort_spinning"] = json!(false);
    let coordinator = wire["coordinator"].as_object_mut().unwrap();
    coordinator.remove("original_root_budget_sha256");
    coordinator.remove("original_deadline_seal_sha256");
    wire["startup_document"] = artifact("startup");
    wire["startup_configuration_sha256"] = json!(sha_value('4'));
    wire["startup_search_configuration_sha256"] =
        json!(old.manifest.engine.search.search_configuration_sha256);
    wire["own_r_root_tie"] = json!({"kind":"not_applicable", "reason":"pc_only"});
    wire["system_runtime_lock"] = serde_json::to_value(
        crate::SystemRuntimeLockV6::new(crate::NativeRuntimeSourceV6::Cpu).unwrap(),
    )
    .unwrap();
    let lock = crate::RoventExecutionManifestV3::from_json(&wire.to_string())
        .unwrap()
        .lock()
        .unwrap();
    let mut descriptor_input = old_descriptor.input;
    descriptor_input.startup_lock_sha256 = lock.canonical_sha256.clone();
    let descriptor = crate::RoventRootDescriptorV3::seal(descriptor_input, &lock).unwrap();
    allocation_input.startup_lock_sha256 = lock.canonical_sha256.clone();
    allocation_input.root_descriptor_sha256 = digest_hex(descriptor.digest().unwrap());
    let allocation =
        crate::RoventRootAllocationClockV3::seal(allocation_input, &descriptor, &lock).unwrap();
    let pc = &lock.manifest.engine.pc;
    let origin = json!({
        "schema": ROVENT_LOCK_V3_CPU_ORIGIN_DOMAIN,
        "input_lock_domain": crate::ROVENT_LOCK_V3_DOMAIN,
        "input_lock_sha256": lock.canonical_sha256,
        "declaration": {
            "purpose": "collection", "mode": "cpu_pc",
            "source_commit": lock.manifest.engine.source_commit,
            "source_components": lock.manifest.engine.source_components,
            "model_semantics": pc.proposer.semantics, "encoding": pc.proposer.encoding,
            "validator_model": pc.validator, "teacher_profiles": lock.manifest.engine.teachers
        },
        "startup_observation": {
            "domain":"rz-rovent-native-startup/1", "process_epoch":23,
            "mode":"cpu_pc", "topology":"roles_only", "weights_sha256":pc.proposer.weights.sha256,
            "export_manifest_sha256":sha_value('b'), "backend_sha256":sha_value('c'),
            "runtime_library_sha256":pc.runtime_artifacts[0].sha256,
            "provider":"cpu", "provider_ready":true,
            "cpu_session_roles":["public", "shared_pc", "validator"],
            "native_configuration_sha256":pc.backend_configuration_sha256,
            "startup_configuration_sha256":lock.manifest.startup_configuration_sha256,
            "search_configuration_sha256":lock.manifest.startup_search_configuration_sha256,
            "validator_loaded":true, "physical_workers_created":1, "neural_service_executed":false
        },
        "registered_namespace": {
            "value_semantics":"test", "value_model":"test", "value_encoding":"test",
            "value_precision":"fp32_tf32_off", "representation_model_domain":"loaded_checkpoint_sha256",
            "representation_model_sha256":pc.proposer.weights.sha256,
            "representation_encoding_sha256":sha_value('d'), "representation_precision":"fp32", "frozen_epoch":1
        },
        "process_boundary_observation":"unknown", "neural_completion":"unknown", "physical_completion":"unknown"
    });
    let origin_sha = generation_digest(ROVENT_LOCK_V3_CPU_ORIGIN_DOMAIN, &origin).unwrap();
    let input = &descriptor.input;
    let root = json!({
        "schema": ROVENT_LOCK_V3_CPU_ROOT_DOMAIN, "native_origin":origin,
        "native_origin_sha256":origin_sha, "process_epoch":input.authority.epoch,
        "game_generation":input.authority.game, "root_generation":input.authority.root,
        "implementation_sha256":input.authority.implementation_sha256,
        "deadline_clock":input.clock.hard.clock_epoch, "deadline_tick":input.clock.hard.at_ns,
        "root_budget_sha256":descriptor.root_budget_sha256, "selective_policy_sha256":sha_value('5'),
        "root_state":{"owner":1,"revision":1,"semantic":input.state_and_history_sha256},
        "store_root_state":1,"root_moves":input.allowed_root_moves,
        "depth_plies":input.original_go.depth_plies.map(|v| v.get()),
        "mate_moves":input.original_go.mate_moves.map(|v| v.get()),
        "requested_nodes":input.original_go.requested_nodes,
        "root_descriptor_sha256":digest_hex(descriptor.digest().unwrap()),"root_descriptor":descriptor,
        "allocation_clock_sha256":digest_hex(allocation.digest().unwrap()),"allocation_clock":allocation
    });
    let sha = json!(generation_digest(ROVENT_LOCK_V3_CPU_ROOT_DOMAIN, &root).unwrap());
    (lock, root, sha)
}

#[test]
fn cpu_collection_root_accepts_original_and_rejects_resealed_forgery() {
    use crate::pals_data_v3::generation_digest;
    let (lock, root, sha) = cpu_collection_root_fixture();
    assert!(validate_lock_v3_cpu_root_metadata(&root, &sha, &lock).is_ok());
    for (field, value) in [
        ("native_configuration_sha256", json!(sha_value('9'))),
        ("validator_loaded", json!(false)),
        ("provider", json!("cuda")),
        ("topology", json!("own_researcher")),
    ] {
        let mut forged = root.clone();
        forged["native_origin"]["startup_observation"][field] = value;
        forged["native_origin_sha256"] = json!(
            generation_digest(ROVENT_LOCK_V3_CPU_ORIGIN_DOMAIN, &forged["native_origin"]).unwrap()
        );
        let forged_sha = json!(generation_digest(ROVENT_LOCK_V3_CPU_ROOT_DOMAIN, &forged).unwrap());
        assert!(
            validate_lock_v3_cpu_root_metadata(&forged, &forged_sha, &lock).is_err(),
            "{field}"
        );
    }
}
