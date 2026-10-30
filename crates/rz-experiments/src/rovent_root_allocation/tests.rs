//! Metadata only: these fixtures issue no live clock, authority, or engine.
use super::*;
use crate::{
    ROVENT_EXECUTION_DOMAIN, ROVENT_POSITION_VISIT_UNIT, RoventExecutionManifest, RoventInputLock,
    RoventOriginalGoCoverage, RoventParsedGo, RoventResolvedGo, RoventRootAdmissionPolicy,
    RoventRootAuthority, RoventRootClocks, RoventRootDeadline, RoventRootDeclaredCeilings,
    RoventRootDescriptor, RoventRootDescriptorInput,
};
use rz_contracts::rovent::ROVENT_SEARCH_REVISION;
use serde_json::{Value, json};
use std::num::NonZeroU16;

fn sha_value(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn digest_hex(digest: rz_contracts::Digest) -> String {
    digest.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn artifact(name: &str) -> Value {
    json!({"path":format!("fixture/{name}"),"sha256":sha_value('a'),"bytes":32,
        "source":"https://example.org/metadata-only-allocation-fixture","license":"fixture-only"})
}

fn fixture() -> (
    RoventInputLock,
    RoventRootDescriptor,
    RoventRootAllocationInput,
) {
    let model = json!({"semantics":"fixture-pc","encoding":"fixture-encoding",
        "policy_head":"legal-candidates/1","value_head":"side-to-move-wdl/1",
        "implementation_sha256":sha_value('b'),"weights":artifact("weights"),
        "provenance":artifact("model-provenance"),"diagnostic":false});
    let wire = json!({
        "domain":ROVENT_EXECUTION_DOMAIN,"contract_revision":ROVENT_SEARCH_REVISION,
        "run_id":"allocation-source-fixture","purpose":"validation",
        "engine":{"source_commit":"a".repeat(40),"source_components":{"fixture":sha_value('c')},
            "binary":artifact("binary"),"mode":"cpu_pc",
            "pc":{"proposer":model,"critic":model,"validator":null,
                "backend":"rust_cpu","device":{"device":"cpu"},
                "backend_configuration_sha256":sha_value('d'),"runtime_artifacts":[],
                "fp32":true,"tf32":false,"batch_width":1,
                "ort_intra_threads":null,"ort_inter_threads":null,"ort_spinning":null},
            "researcher":null,"teachers":[],"nnuec":null,
            "search":{"node_unit":ROVENT_POSITION_VISIT_UNIT,"requested_nodes":7,
                "depth_plies":2,"mate_moves":null,"registered_max_chess_plies":64,
                "searchmoves":["d2d4","e2e4"],"search_configuration_sha256":sha_value('1'),
                "root_state_and_history_sha256":sha_value('2'),"rules_profile_sha256":sha_value('3')}},
        "resources":{"cpu_quota_millicpus":2000,
            "pools":[{"actors":["manager","pc_caller"],"max_threads":1,"cpu_charge_millicpus":1000}],
            "memory_high_bytes":6_u64<<30,"memory_max_bytes":12_u64<<30,
            "gpu_devices":0,"physical_nn_concurrency":1,"topology_policy":"shared-control/1"},
        "coordinator":{"policy_sha256":sha_value('5'),"questions_max":64,
            "ancestor_depth_max":16,"waiting_jobs_max":16,"role_tickets_max":16,
            "cpu_credit_limit":0,"original_root_budget_sha256":sha_value('6'),
            "original_deadline_seal_sha256":sha_value('7')},
        "wall_time_max_ms":60000,"cleanup_max_ms":1000,"output_bytes_max":1048576,
        "agent_renewable":true,"provenance":[artifact("source-provenance")]});
    let lock = RoventExecutionManifest::from_json(&wire.to_string())
        .unwrap()
        .lock()
        .unwrap();
    let manifest = &lock.manifest;
    let coordinator = &manifest.coordinator;
    let search = &manifest.engine.search;
    let searchmoves = Some(vec!["e2e4".into(), "d2d4".into()]);
    let descriptor = RoventRootDescriptor::seal(RoventRootDescriptorInput {
        policy: RoventRootAdmissionPolicy::ExactRegistered,
        startup_lock_sha256: lock.canonical_sha256.clone(),
        state_and_history_sha256: search.root_state_and_history_sha256.clone(),
        rules_profile_sha256: search.rules_profile_sha256.clone(),
        authority: RoventRootAuthority {
            epoch: 23,
            game: 0,
            root: 1,
            implementation_sha256: sha_value('4'),
        },
        node_unit: search.node_unit.clone(),
        search_configuration_sha256: search.search_configuration_sha256.clone(),
        coordinator_policy_sha256: coordinator.policy_sha256.clone(),
        original_go: RoventResolvedGo {
            coverage: RoventOriginalGoCoverage::FullOriginalGo,
            requested_nodes: Some(7),
            depth_plies: NonZeroU16::new(2),
            mate_moves: None,
            searchmoves: searchmoves.clone(),
            transport_simulation_cap: 4096,
            original_parsed_limits: Some(RoventParsedGo {
                searchmoves,
                depth_plies: NonZeroU16::new(2),
                mate_moves: None,
                movetime_ms: Some(1000),
                white_time_ms: None,
                black_time_ms: None,
                white_increment_ms: None,
                black_increment_ms: None,
                moves_to_go: None,
                nodes: Some(7),
                infinite: false,
                ponder: false,
            }),
        },
        allowed_root_moves: vec!["e2e4".into(), "d2d4".into()],
        clock: RoventRootClocks {
            hard: tick(1_000_000_000),
            soft: tick(800_000_000),
            admission: tick(900_000_000),
            cleanup: tick(1_500_000_000),
        },
        declared_ceilings: RoventRootDeclaredCeilings {
            resource_plan: manifest.resources.clone(),
            registered_max_chess_plies: search.registered_max_chess_plies,
            cpu_credit_limit: coordinator.cpu_credit_limit,
            questions_max: coordinator.questions_max,
            ancestor_depth_max: coordinator.ancestor_depth_max,
            waiting_jobs_max: coordinator.waiting_jobs_max,
            role_tickets_max: coordinator.role_tickets_max,
            wall_time_max_ms: manifest.wall_time_max_ms,
            cleanup_max_ms: manifest.cleanup_max_ms,
            output_bytes_max: manifest.output_bytes_max,
        },
    })
    .unwrap();
    let input = RoventRootAllocationInput {
        root_descriptor_sha256: digest_hex(descriptor.digest().unwrap()),
        startup_lock_sha256: lock.canonical_sha256.clone(),
        authority: descriptor.input.authority.clone(),
        start: tick(100_000_000),
        output: tick(1_100_000_000),
        allocation_ns: 1_000_000_000,
        wall_time_max_ms: manifest.wall_time_max_ms,
        cleanup_max_ms: manifest.cleanup_max_ms,
    };
    (lock, descriptor, input)
}

fn tick(at_ns: u64) -> RoventRootDeadline {
    RoventRootDeadline {
        clock_epoch: 23,
        at_ns,
    }
}

#[test]
fn v2_allocation_rejects_old_wire_and_joins_exact_v2_descriptor_and_lock() {
    let (old_lock, old_descriptor, mut input) = fixture();
    let old = RoventRootAllocationClock::seal(input.clone(), &old_descriptor, &old_lock).unwrap();
    let old_bytes = old.canonical_bytes().unwrap();
    let mut wire = serde_json::to_value(&old_lock.manifest).unwrap();
    let object = wire.as_object_mut().unwrap();
    object.insert("domain".into(), json!(crate::ROVENT_EXECUTION_V2_DOMAIN));
    object.insert("startup_document".into(), artifact("startup"));
    object.insert("startup_configuration_sha256".into(), json!(sha_value('4')));
    object.insert(
        "startup_search_configuration_sha256".into(),
        json!(old_lock.manifest.engine.search.search_configuration_sha256),
    );
    object.insert(
        "own_r_root_tie".into(),
        json!({"kind":"not_applicable", "reason":"pc_only"}),
    );
    let lock = crate::RoventExecutionManifestV2::from_json(&wire.to_string())
        .unwrap()
        .lock()
        .unwrap();
    let mut descriptor_input = old_descriptor.input.clone();
    descriptor_input.startup_lock_sha256 = lock.canonical_sha256.clone();
    let descriptor = RoventRootDescriptorV2::seal(descriptor_input, &lock).unwrap();
    input.startup_lock_sha256 = lock.canonical_sha256.clone();
    input.root_descriptor_sha256 = digest_hex(descriptor.digest().unwrap());
    let allocation = RoventRootAllocationClockV2::seal(input, &descriptor, &lock).unwrap();
    let encoded = allocation.canonical_bytes().unwrap();
    let text = std::str::from_utf8(&encoded).unwrap();
    assert!(RoventRootAllocationClock::from_json(text).is_err());
    assert!(
        RoventRootAllocationClockV2::from_json(std::str::from_utf8(&old_bytes).unwrap()).is_err()
    );
    assert_eq!(
        RoventRootAllocationClockV2::from_json(text).unwrap(),
        allocation
    );
    allocation.verify_against(&descriptor, &lock).unwrap();
    let mut bad = allocation.clone();
    bad.input.root_descriptor_sha256 = sha_value('9');
    assert!(bad.verify_against(&descriptor, &lock).is_err());
    let mut bad = allocation.clone();
    bad.input.authority.root += 1;
    assert!(bad.verify_against(&descriptor, &lock).is_err());
    let mut bad = allocation.clone();
    bad.input.cleanup_max_ms += 1;
    assert!(bad.verify_against(&descriptor, &lock).is_err());
    let mut bad = allocation.clone();
    bad.input.start.clock_epoch += 1;
    assert!(bad.verify_against(&descriptor, &lock).is_err());
    let mut other = lock.clone();
    other.manifest.run_id.push_str("-other");
    other = other.manifest.lock().unwrap();
    assert!(allocation.verify_against(&descriptor, &other).is_err());
    assert_eq!(old.canonical_bytes().unwrap(), old_bytes);
    old.verify_against(&old_descriptor, &old_lock).unwrap();
}

fn wrapper(input: RoventRootAllocationInput) -> RoventRootAllocationClock {
    RoventRootAllocationClock {
        domain: ROVENT_ROOT_ALLOCATION_CLOCK_DOMAIN.into(),
        input,
    }
}

#[test]
fn sealed_roundtrip_is_stable_and_does_not_rewrite_existing_descriptor() {
    let (lock, descriptor, input) = fixture();
    let old_bytes = descriptor.canonical_bytes().unwrap();
    let old_digest = descriptor.digest().unwrap();
    let clock = RoventRootAllocationClock::seal(input.clone(), &descriptor, &lock).unwrap();
    clock.validate().unwrap();
    clock.verify_against(&descriptor, &lock).unwrap();
    let canonical = clock.canonical_bytes().unwrap();
    let decoded =
        RoventRootAllocationClock::from_json(std::str::from_utf8(&canonical).unwrap()).unwrap();
    assert_eq!(decoded.input, input);
    assert_eq!(decoded.domain, ROVENT_ROOT_ALLOCATION_CLOCK_DOMAIN);
    assert_eq!(decoded.canonical_bytes().unwrap(), canonical);
    assert_eq!(decoded.digest().unwrap(), clock.digest().unwrap());
    assert_ne!(clock.digest().unwrap(), old_digest);
    assert_eq!(descriptor.canonical_bytes().unwrap(), old_bytes);
    assert_eq!(descriptor.digest().unwrap(), old_digest);
}

#[test]
fn wire_key_order_does_not_change_canonical_clock_identity() {
    let (lock, descriptor, input) = fixture();
    let clock = RoventRootAllocationClock::seal(input.clone(), &descriptor, &lock).unwrap();
    let reversed = format!(
        "{{\"input\":{},\"domain\":{}}}",
        serde_json::to_string(&input).unwrap(),
        serde_json::to_string(ROVENT_ROOT_ALLOCATION_CLOCK_DOMAIN).unwrap()
    );
    let decoded = RoventRootAllocationClock::from_json(&reversed).unwrap();
    assert_eq!(
        decoded.canonical_bytes().unwrap(),
        clock.canonical_bytes().unwrap()
    );
    assert_eq!(decoded.digest().unwrap(), clock.digest().unwrap());
    let mut changed = input;
    changed.output.at_ns += 1;
    changed.allocation_ns += 1;
    let changed = RoventRootAllocationClock::seal(changed, &descriptor, &lock).unwrap();
    assert_ne!(changed.digest().unwrap(), clock.digest().unwrap());
}

#[test]
fn strict_decoder_rejects_duplicates_and_unknown_fields_at_each_depth() {
    let (_, _, input) = fixture();
    let text = serde_json::to_string(&wrapper(input)).unwrap();
    for malformed in [
        text.replacen(
            "\"domain\":",
            &format!("\"domain\":\"{ROVENT_ROOT_ALLOCATION_CLOCK_DOMAIN}\",\"domain\":"),
            1,
        ),
        text.replacen(
            "\"allocation_ns\":",
            "\"allocation_ns\":1,\"allocation_ns\":",
            1,
        ),
        text.replacen("\"epoch\":23", "\"epoch\":23,\"epoch\":23", 1),
        text.replacen(
            "\"clock_epoch\":23",
            "\"clock_epoch\":23,\"clock_epoch\":23",
            1,
        ),
        text.replacen('{', "{\"unknown\":0,", 1),
        text.replacen("\"allocation_ns\":", "\"unknown\":0,\"allocation_ns\":", 1),
        text.replacen("\"epoch\":23", "\"unknown\":0,\"epoch\":23", 1),
        text.replacen("\"clock_epoch\":23", "\"unknown\":0,\"clock_epoch\":23", 1),
    ] {
        assert!(RoventRootAllocationClock::from_json(&malformed).is_err());
    }
}

#[test]
fn parser_byte_budget_and_hash_syntax_are_not_relaxed() {
    let (_, _, input) = fixture();
    let text = serde_json::to_string(&wrapper(input.clone())).unwrap();
    let padded = format!("{text}{}", " ".repeat(MAX_ROOT_ALLOCATION_BYTES + 1));
    assert!(RoventRootAllocationClock::from_json(&padded).is_err());
    for bad in [
        String::new(),
        "a".repeat(63),
        "A".repeat(64),
        "g".repeat(64),
    ] {
        for field in 0..3 {
            let mut invalid = input.clone();
            match field {
                0 => invalid.root_descriptor_sha256 = bad.clone(),
                1 => invalid.startup_lock_sha256 = bad.clone(),
                _ => invalid.authority.implementation_sha256 = bad.clone(),
            }
            assert!(wrapper(invalid).validate().is_err());
        }
    }
    let mut wrong_domain = wrapper(input);
    wrong_domain.domain = "rz-rovent-root-admission/1".into();
    assert!(wrong_domain.validate().is_err());
}

#[test]
fn standalone_clock_requires_exact_epoch_interval_and_finite_bounds() {
    let (_, _, input) = fixture();
    for case in 0..11 {
        let mut invalid = input.clone();
        match case {
            0 => invalid.start.clock_epoch += 1,
            1 => invalid.output.clock_epoch += 1,
            2 => invalid.start.at_ns = invalid.output.at_ns + 1,
            3 => invalid.allocation_ns += 1,
            4 => invalid.wall_time_max_ms = 0,
            5 => invalid.wall_time_max_ms = 86_400_001,
            6 => invalid.cleanup_max_ms = 0,
            7 => invalid.cleanup_max_ms = 30_001,
            8 => invalid.wall_time_max_ms = 1,
            9 => invalid.wall_time_max_ms = u64::MAX,
            _ => invalid.cleanup_max_ms = u64::MAX,
        }
        assert!(wrapper(invalid).validate().is_err(), "case {case}");
    }
    let mut overflow = input;
    overflow.start.at_ns = u64::MAX - 1;
    overflow.output.at_ns = 0;
    overflow.allocation_ns = 2;
    assert!(wrapper(overflow).validate().is_err());
}

#[test]
fn valid_standalone_metadata_cannot_substitute_descriptor_lock_or_authority() {
    let (lock, descriptor, input) = fixture();
    for case in 0..5 {
        let mut changed = input.clone();
        match case {
            0 => changed.root_descriptor_sha256 = sha_value('8'),
            1 => changed.startup_lock_sha256 = sha_value('9'),
            2 => changed.authority.game += 1,
            3 => changed.authority.root += 1,
            _ => changed.authority.implementation_sha256 = sha_value('e'),
        }
        let changed = wrapper(changed);
        changed.validate().unwrap();
        assert!(
            changed.verify_against(&descriptor, &lock).is_err(),
            "case {case}"
        );
        assert!(RoventRootAllocationClock::seal(changed.input, &descriptor, &lock).is_err());
    }
    let mut forged_descriptor = descriptor;
    forged_descriptor.input.authority.root += 1;
    assert!(
        wrapper(input)
            .verify_against(&forged_descriptor, &lock)
            .is_err()
    );
}

#[test]
fn original_wall_and_cleanup_ceilings_must_join_exactly() {
    let (lock, descriptor, input) = fixture();
    for case in 0..4 {
        let mut changed = input.clone();
        match case {
            0 => changed.wall_time_max_ms += 1,
            1 => changed.wall_time_max_ms -= 1,
            2 => changed.cleanup_max_ms += 1,
            _ => changed.cleanup_max_ms -= 1,
        }
        let changed = wrapper(changed);
        changed.validate().unwrap();
        assert!(
            changed.verify_against(&descriptor, &lock).is_err(),
            "case {case}"
        );
    }
}

#[test]
fn allocation_encloses_original_search_but_does_not_extend_cleanup() {
    let (lock, descriptor, input) = fixture();
    for (start, output) in [
        (850_000_000, 1_100_000_000),
        (100_000_000, 999_999_999),
        (100_000_000, 1_500_000_001),
    ] {
        let mut changed = input.clone();
        changed.start.at_ns = start;
        changed.output.at_ns = output;
        changed.allocation_ns = output - start;
        let changed = wrapper(changed);
        changed.validate().unwrap();
        assert!(changed.verify_against(&descriptor, &lock).is_err());
    }
    let mut exact_boundaries = input;
    exact_boundaries.start = descriptor.input.clock.soft;
    exact_boundaries.output = descriptor.input.clock.hard;
    exact_boundaries.allocation_ns = exact_boundaries.output.at_ns - exact_boundaries.start.at_ns;
    RoventRootAllocationClock::seal(exact_boundaries, &descriptor, &lock).unwrap();
}

#[test]
fn zero_allocation_and_maximum_tick_are_metadata_boundaries_not_completion_evidence() {
    let (lock, descriptor, mut input) = fixture();
    let mut descriptor_input = descriptor.input.clone();
    descriptor_input.clock.soft = tick(u64::MAX);
    descriptor_input.clock.admission = tick(u64::MAX);
    descriptor_input.clock.hard = tick(u64::MAX);
    descriptor_input.clock.cleanup = tick(u64::MAX);
    let descriptor = RoventRootDescriptor::seal(descriptor_input).unwrap();
    input.root_descriptor_sha256 = digest_hex(descriptor.digest().unwrap());
    input.start = tick(u64::MAX);
    input.output = tick(u64::MAX);
    input.allocation_ns = 0;
    let clock = RoventRootAllocationClock::seal(input, &descriptor, &lock).unwrap();
    clock.verify_against(&descriptor, &lock).unwrap();
    let mut wire = serde_json::to_value(&clock).unwrap();
    wire["input"]["actual_finished_at_ns"] = json!(u64::MAX);
    assert!(RoventRootAllocationClock::from_json(&wire.to_string()).is_err());
}
