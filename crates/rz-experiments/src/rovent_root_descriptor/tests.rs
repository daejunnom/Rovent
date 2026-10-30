//! Pure metadata fixtures. No live Rules/clock/process/engine is constructed.
use super::*;
use crate::{ROVENT_EXECUTION_DOMAIN, RoventExecutionManifest};
use rz_contracts::rovent::ROVENT_SEARCH_REVISION;

fn sha_value(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn artifact(name: &str) -> Value {
    json!({"path":format!("fixture/{name}"),"sha256":sha_value('a'),"bytes":32,
        "source":"https://example.org/source-only-root-descriptor-fixture","license":"fixture-only"})
}

fn lock() -> RoventInputLock {
    let model = json!({"semantics":"fixture-pc","encoding":"fixture-encoding",
        "policy_head":"legal-candidates/1","value_head":"side-to-move-wdl/1",
        "implementation_sha256":sha_value('b'),"weights":artifact("weights"),
        "provenance":artifact("model-provenance"),"diagnostic":false});
    let wire = json!({
        "domain":ROVENT_EXECUTION_DOMAIN,"contract_revision":ROVENT_SEARCH_REVISION,
        "run_id":"root-descriptor-source-fixture","purpose":"validation",
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
    RoventExecutionManifest::from_json(&wire.to_string())
        .unwrap()
        .lock()
        .unwrap()
}

fn lock_v2(base: &RoventInputLock) -> RoventInputLockV2 {
    let mut wire = serde_json::to_value(&base.manifest).unwrap();
    let object = wire.as_object_mut().unwrap();
    object.insert("domain".into(), json!(crate::ROVENT_EXECUTION_V2_DOMAIN));
    object.insert("startup_document".into(), artifact("startup"));
    object.insert("startup_configuration_sha256".into(), json!(sha_value('4')));
    object.insert(
        "startup_search_configuration_sha256".into(),
        json!(base.manifest.engine.search.search_configuration_sha256),
    );
    object.insert(
        "own_r_root_tie".into(),
        json!({"kind":"not_applicable", "reason":"pc_only"}),
    );
    crate::RoventExecutionManifestV2::from_json(&wire.to_string())
        .unwrap()
        .lock()
        .unwrap()
}

#[test]
fn v2_descriptor_has_distinct_wire_and_joins_only_exact_v2_lock() {
    let old = lock();
    let old_descriptor = RoventRootDescriptor::seal(input(&old)).unwrap();
    let old_bytes = old_descriptor.canonical_bytes().unwrap();
    let old_digest = old_descriptor.digest().unwrap();
    let new_lock = lock_v2(&old);
    let mut new_input = input(&old);
    new_input.startup_lock_sha256 = new_lock.canonical_sha256.clone();
    let new_descriptor = RoventRootDescriptorV2::seal(new_input, &new_lock).unwrap();
    let wire = new_descriptor.canonical_bytes().unwrap();
    let text = std::str::from_utf8(&wire).unwrap();
    assert!(RoventRootDescriptor::from_json(text).is_err());
    assert!(RoventRootDescriptorV2::from_json(std::str::from_utf8(&old_bytes).unwrap()).is_err());
    assert_eq!(
        RoventRootDescriptorV2::from_json(text).unwrap(),
        new_descriptor
    );
    assert!(new_descriptor.verify_against(&new_lock).is_ok());
    let mut other = new_lock.clone();
    other.manifest.run_id.push_str("-other");
    other = other.manifest.lock().unwrap();
    assert!(new_descriptor.verify_against(&other).is_err());
    let mut bad = new_descriptor.clone();
    bad.startup_lock_domain = crate::ROVENT_LOCK_DOMAIN.into();
    assert!(bad.validate().is_err());
    let mut bad_input = new_descriptor.input.clone();
    bad_input.search_configuration_sha256 = sha_value('9');
    assert!(RoventRootDescriptorV2::seal(bad_input, &new_lock).is_err());
    let mut bad_input = new_descriptor.input.clone();
    bad_input.declared_ceilings.output_bytes_max += 1;
    assert!(RoventRootDescriptorV2::seal(bad_input, &new_lock).is_err());
    let mut bad_input = new_descriptor.input.clone();
    bad_input.clock.hard.clock_epoch += 1;
    assert!(RoventRootDescriptorV2::seal(bad_input, &new_lock).is_err());
    assert_eq!(old_descriptor.canonical_bytes().unwrap(), old_bytes);
    assert_eq!(old_descriptor.digest().unwrap(), old_digest);
    old_descriptor.verify_against(&old).unwrap();
}

fn input(lock: &RoventInputLock) -> RoventRootDescriptorInput {
    let manifest = &lock.manifest;
    let coordinator = &manifest.coordinator;
    let search = &manifest.engine.search;
    let searchmoves = Some(vec!["e2e4".into(), "d2d4".into()]);
    let deadline = |at_ns| RoventRootDeadline {
        clock_epoch: 23,
        at_ns,
    };
    RoventRootDescriptorInput {
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
                movetime_ms: None,
                white_time_ms: Some(120000),
                black_time_ms: Some(120000),
                white_increment_ms: Some(1000),
                black_increment_ms: Some(1000),
                moves_to_go: None,
                nodes: Some(7),
                infinite: false,
                ponder: false,
            }),
        },
        allowed_root_moves: vec!["e2e4".into(), "d2d4".into()],
        clock: RoventRootClocks {
            hard: deadline(1_000_000_000),
            soft: deadline(800_000_000),
            admission: deadline(900_000_000),
            cleanup: deadline(1_500_000_000),
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
    }
}

fn raw(input: &mut RoventRootDescriptorInput) -> &mut RoventParsedGo {
    input.original_go.original_parsed_limits.as_mut().unwrap()
}

#[test]
fn sealed_roundtrip_recomputes_every_identity_and_preserves_raw_go() {
    let lock = lock();
    let input = input(&lock);
    let descriptor = RoventRootDescriptor::seal(input.clone()).unwrap();
    descriptor.validate().unwrap();
    descriptor.verify_against(&lock).unwrap();
    let canonical = descriptor.canonical_bytes().unwrap();
    let decoded =
        RoventRootDescriptor::from_json(std::str::from_utf8(&canonical).unwrap()).unwrap();
    assert_eq!(decoded, descriptor);
    assert_eq!(decoded.input, input);
    assert_eq!(decoded.canonical_bytes().unwrap(), canonical);
    assert_eq!(
        descriptor.root_budget_digest().unwrap(),
        parse_digest(&descriptor.root_budget_sha256).unwrap()
    );
    assert_eq!(
        descriptor.deadline_seal_digest().unwrap(),
        parse_digest(&descriptor.deadline_seal_sha256).unwrap()
    );
    assert_ne!(
        descriptor.digest().unwrap(),
        descriptor.root_budget_digest().unwrap()
    );
    assert_ne!(
        descriptor.root_budget_digest().unwrap(),
        descriptor.deadline_seal_digest().unwrap()
    );
}

#[test]
fn original_order_and_repetitions_are_preserved_while_sets_join() {
    let lock = lock();
    let first = RoventRootDescriptor::seal(input(&lock)).unwrap();
    let mut repeated = input(&lock);
    raw(&mut repeated).searchmoves = Some(vec!["d2d4".into(), "e2e4".into(), "e2e4".into()]);
    let repeated = RoventRootDescriptor::seal(repeated).unwrap();
    repeated.verify_against(&lock).unwrap();
    assert_eq!(
        first.input.allowed_root_moves,
        repeated.input.allowed_root_moves
    );
    assert_eq!(
        repeated
            .input
            .original_go
            .original_parsed_limits
            .as_ref()
            .unwrap()
            .searchmoves
            .as_ref()
            .unwrap()
            .len(),
        3
    );
    assert_ne!(
        first.root_budget_digest().unwrap(),
        repeated.root_budget_digest().unwrap()
    );

    let mut many = input(&lock);
    many.policy = RoventRootAdmissionPolicy::DynamicGo;
    many.allowed_root_moves = vec!["e2e4".into()];
    many.original_go.searchmoves = Some(many.allowed_root_moves.clone());
    raw(&mut many).searchmoves = Some(vec!["e2e4".into(); 219]);
    let many = RoventRootDescriptor::seal(many).unwrap();
    many.verify_against(&lock).unwrap();
    assert_eq!(many.input.allowed_root_moves.len(), 1);
    assert_eq!(
        many.input
            .original_go
            .original_parsed_limits
            .unwrap()
            .searchmoves
            .unwrap()
            .len(),
        219
    );
}

#[test]
fn raw_and_distinct_move_budgets_are_independent_and_bounded() {
    let lock = lock();
    let mut too_many_raw = input(&lock);
    raw(&mut too_many_raw).searchmoves = Some(vec!["e2e4".into(); 5000]);
    assert!(RoventRootDescriptor::seal(too_many_raw).is_err());
    for token in ["0000", "e2e2", "e2e9", "a7a8k", "a7a8Q", "é2e4"] {
        let mut invalid = input(&lock);
        raw(&mut invalid).searchmoves = Some(vec![token.into()]);
        assert!(RoventRootDescriptor::seal(invalid).is_err());
    }
    let mut duplicate = input(&lock);
    duplicate.allowed_root_moves.push("e2e4".into());
    duplicate.original_go.searchmoves = Some(duplicate.allowed_root_moves.clone());
    assert!(RoventRootDescriptor::seal(duplicate).is_err());
    let mut empty = input(&lock);
    empty.allowed_root_moves.clear();
    empty.original_go.searchmoves = Some(vec![]);
    raw(&mut empty).searchmoves = Some(vec![]);
    assert!(RoventRootDescriptor::seal(empty).is_err());
    let mut oversized = input(&lock);
    let moves: Vec<_> = (0..64)
        .flat_map(|from| {
            (0..64)
                .filter(move |to| *to != from)
                .map(move |to| (from, to))
        })
        .take(219)
        .map(|(from, to)| {
            format!(
                "{}{}{}{}",
                char::from(b'a' + (from % 8) as u8),
                1 + from / 8,
                char::from(b'a' + (to % 8) as u8),
                1 + to / 8
            )
        })
        .collect();
    oversized.allowed_root_moves = moves.clone();
    oversized.original_go.searchmoves = Some(moves.clone());
    raw(&mut oversized).searchmoves = Some(moves);
    assert!(RoventRootDescriptor::seal(oversized).is_err());
}

#[test]
fn full_raw_go_requires_exact_resolved_nodes_goal_and_move_set() {
    let lock = lock();
    for change in 0..6 {
        let mut invalid = input(&lock);
        match change {
            0 => raw(&mut invalid).nodes = Some(0),
            1 => raw(&mut invalid).depth_plies = NonZeroU16::new(3),
            2 => raw(&mut invalid).mate_moves = NonZeroU16::new(1),
            3 => raw(&mut invalid).searchmoves = None,
            4 => raw(&mut invalid).searchmoves = Some(vec!["a2a4".into()]),
            _ => invalid.allowed_root_moves.reverse(),
        }
        assert!(
            RoventRootDescriptor::seal(invalid).is_err(),
            "change {change}"
        );
    }
}

#[test]
fn coverage_never_invents_missing_original_limits() {
    let lock = lock();
    let mut missing = input(&lock);
    missing.original_go.original_parsed_limits = None;
    assert!(RoventRootDescriptor::seal(missing.clone()).is_err());
    missing.original_go.coverage = RoventOriginalGoCoverage::SemanticContextOnly;
    let semantic = RoventRootDescriptor::seal(missing).unwrap();
    semantic.verify_against(&lock).unwrap();
    assert!(semantic.input.original_go.original_parsed_limits.is_none());
    let mut understated = input(&lock);
    understated.original_go.coverage = RoventOriginalGoCoverage::SemanticContextOnly;
    assert!(RoventRootDescriptor::seal(understated).is_err());
}

#[test]
fn dynamic_nodes_zero_is_not_unspecified_or_a_legacy_manifest_rewrite() {
    let lock = lock();
    let mut zero = input(&lock);
    zero.policy = RoventRootAdmissionPolicy::DynamicGo;
    zero.original_go.requested_nodes = Some(0);
    raw(&mut zero).nodes = Some(0);
    let zero_descriptor = RoventRootDescriptor::seal(zero.clone()).unwrap();
    zero_descriptor.verify_against(&lock).unwrap();
    let mut unspecified = zero.clone();
    unspecified.original_go.requested_nodes = None;
    raw(&mut unspecified).nodes = None;
    let unspecified = RoventRootDescriptor::seal(unspecified).unwrap();
    assert_ne!(
        zero_descriptor.root_budget_digest().unwrap(),
        unspecified.root_budget_digest().unwrap()
    );
    zero.policy = RoventRootAdmissionPolicy::ExactRegistered;
    assert!(
        RoventRootDescriptor::seal(zero)
            .unwrap()
            .verify_against(&lock)
            .is_err()
    );
    let mut legacy = lock.manifest;
    legacy.engine.search.requested_nodes = Some(0);
    assert!(legacy.lock().is_err());
}

#[test]
fn clock_changes_only_deadline_and_whole_seals_with_nanosecond_precision() {
    let lock = lock();
    let initial = RoventRootDescriptor::seal(input(&lock)).unwrap();
    let mut later = initial.input.clone();
    later.clock.hard.at_ns += 1;
    let later = RoventRootDescriptor::seal(later).unwrap();
    assert_eq!(
        initial.root_budget_digest().unwrap(),
        later.root_budget_digest().unwrap()
    );
    assert_ne!(
        initial.deadline_seal_digest().unwrap(),
        later.deadline_seal_digest().unwrap()
    );
    assert_ne!(initial.digest().unwrap(), later.digest().unwrap());
    assert!(
        !String::from_utf8(initial.canonical_bytes().unwrap())
            .unwrap()
            .contains("Instant")
    );
}

#[test]
fn clock_epoch_order_and_exact_cleanup_ceiling_are_checked() {
    let lock = lock();
    for change in 0..6 {
        let mut invalid = input(&lock);
        match change {
            0 => invalid.clock.soft.clock_epoch += 1,
            1 => invalid.clock.soft.at_ns = invalid.clock.admission.at_ns + 1,
            2 => invalid.clock.admission.at_ns = invalid.clock.hard.at_ns + 1,
            3 => invalid.clock.cleanup.at_ns = invalid.clock.hard.at_ns - 1,
            4 => invalid.clock.cleanup.at_ns = invalid.clock.hard.at_ns + 1_000_000_001,
            _ => invalid.authority.epoch = 0,
        }
        assert!(
            RoventRootDescriptor::seal(invalid).is_err(),
            "change {change}"
        );
    }
    let mut exact = input(&lock);
    exact.clock.cleanup.at_ns = exact.clock.hard.at_ns + 1_000_000_000;
    RoventRootDescriptor::seal(exact).unwrap();
}

#[test]
fn goal_capability_and_nonzero_wire_types_are_not_clamped() {
    let lock = lock();
    let mut deep = input(&lock);
    deep.original_go.depth_plies = NonZeroU16::new(65);
    raw(&mut deep).depth_plies = NonZeroU16::new(65);
    assert!(RoventRootDescriptor::seal(deep).is_err());
    let mut overflow = input(&lock);
    overflow.declared_ceilings.registered_max_chess_plies = NonZeroU16::new(u16::MAX).unwrap();
    overflow.original_go.mate_moves = NonZeroU16::new(32768);
    raw(&mut overflow).mate_moves = NonZeroU16::new(32768);
    assert!(RoventRootDescriptor::seal(overflow).is_err());
    let descriptor = RoventRootDescriptor::seal(input(&lock)).unwrap();
    let mut wire = serde_json::to_value(descriptor).unwrap();
    wire["input"]["original_go"]["depth_plies"] = json!(0);
    assert!(RoventRootDescriptor::from_json(&wire.to_string()).is_err());
}

#[test]
fn exact_registered_subset_is_set_equal_but_resolved_allowed_order_is_exact() {
    let lock = lock();
    let descriptor = RoventRootDescriptor::seal(input(&lock)).unwrap();
    assert_ne!(
        descriptor.input.original_go.searchmoves,
        lock.manifest.engine.search.searchmoves
    );
    descriptor.verify_against(&lock).unwrap();
    let mut unrestricted = descriptor.input.clone();
    unrestricted.original_go.searchmoves = None;
    raw(&mut unrestricted).searchmoves = None;
    assert!(
        RoventRootDescriptor::seal(unrestricted.clone())
            .unwrap()
            .verify_against(&lock)
            .is_err()
    );
    unrestricted.policy = RoventRootAdmissionPolicy::DynamicGo;
    RoventRootDescriptor::seal(unrestricted)
        .unwrap()
        .verify_against(&lock)
        .unwrap();
}

#[test]
fn dynamic_roots_inherit_lock_conditions_without_claiming_registered_workload() {
    let lock = lock();
    let mut other = input(&lock);
    other.state_and_history_sha256 = sha_value('8');
    other.original_go.requested_nodes = Some(19);
    other.original_go.depth_plies = NonZeroU16::new(3);
    raw(&mut other).nodes = Some(19);
    raw(&mut other).depth_plies = NonZeroU16::new(3);
    let exact = RoventRootDescriptor::seal(other.clone()).unwrap();
    assert!(exact.verify_against(&lock).is_err());
    other.policy = RoventRootAdmissionPolicy::DynamicGo;
    let dynamic = RoventRootDescriptor::seal(other).unwrap();
    dynamic.verify_against(&lock).unwrap();
    assert_eq!(dynamic.input.policy, RoventRootAdmissionPolicy::DynamicGo);
    assert_eq!(dynamic.input.startup_lock_sha256, lock.canonical_sha256);
}

#[test]
fn every_declared_ceiling_and_configuration_join_is_exact() {
    let lock = lock();
    for change in 0..17 {
        let mut altered = input(&lock);
        match change {
            0 => altered.startup_lock_sha256 = sha_value('0'),
            1 => altered.rules_profile_sha256 = sha_value('0'),
            2 => altered.search_configuration_sha256 = sha_value('0'),
            3 => altered.coordinator_policy_sha256 = sha_value('0'),
            4 => altered.declared_ceilings.resource_plan.cpu_quota_millicpus += 1,
            5 => {
                altered.declared_ceilings.registered_max_chess_plies = NonZeroU16::new(63).unwrap()
            }
            6 => altered.declared_ceilings.cpu_credit_limit += 1,
            7 => altered.declared_ceilings.questions_max += 1,
            8 => altered.declared_ceilings.ancestor_depth_max += 1,
            9 => altered.declared_ceilings.waiting_jobs_max += 1,
            10 => altered.declared_ceilings.role_tickets_max += 1,
            11 => altered.declared_ceilings.wall_time_max_ms += 1,
            12 => altered.declared_ceilings.cleanup_max_ms += 1,
            13 => altered.declared_ceilings.output_bytes_max += 1,
            14 => altered.declared_ceilings.resource_plan.memory_high_bytes -= 1,
            15 => altered.declared_ceilings.resource_plan.topology_policy = "other/1".into(),
            _ => altered.node_unit = "other-nodes/1".into(),
        }
        match RoventRootDescriptor::seal(altered) {
            Ok(descriptor) => assert!(descriptor.verify_against(&lock).is_err(), "change {change}"),
            Err(_) if change == 16 => {}
            Err(error) => panic!("unexpected fixture rejection {change}: {error}"),
        }
    }
    let descriptor = RoventRootDescriptor::seal(input(&lock)).unwrap();
    let mut tampered = lock;
    tampered.manifest.coordinator.questions_max += 1;
    assert!(descriptor.verify_against(&tampered).is_err());
}

#[test]
fn full_original_time_and_output_fields_stay_in_budget_identity() {
    let lock = lock();
    let baseline = RoventRootDescriptor::seal(input(&lock)).unwrap();
    for change in 0..8 {
        let mut different = baseline.input.clone();
        let go = raw(&mut different);
        match change {
            0 => go.white_time_ms = Some(123456),
            1 => go.black_time_ms = Some(234567),
            2 => go.white_increment_ms = Some(321),
            3 => go.black_increment_ms = Some(654),
            4 => go.moves_to_go = Some(17),
            5 => go.ponder = true,
            _ => {
                go.white_time_ms = None;
                go.black_time_ms = None;
                go.white_increment_ms = None;
                go.black_increment_ms = None;
                if change == 6 {
                    go.movetime_ms = Some(500);
                } else {
                    go.infinite = true;
                }
            }
        }
        let changed = RoventRootDescriptor::seal(different).unwrap();
        changed.verify_against(&lock).unwrap();
        assert_ne!(
            baseline.root_budget_digest().unwrap(),
            changed.root_budget_digest().unwrap(),
            "change {change}"
        );
        assert_eq!(baseline.input.clock, changed.input.clock);
    }
}

#[test]
fn strict_decoder_rejects_duplicate_unknown_nonfinite_and_oversized_input() {
    let lock = lock();
    let descriptor = RoventRootDescriptor::seal(input(&lock)).unwrap();
    let text = String::from_utf8(descriptor.canonical_bytes().unwrap()).unwrap();
    let duplicate = text.replacen("\"epoch\":23", "\"epoch\":23,\"epoch\":23", 1);
    assert_ne!(duplicate, text);
    assert!(RoventRootDescriptor::from_json(&duplicate).is_err());
    for parent in ["", "input", "go", "raw", "resource"] {
        let mut value = serde_json::to_value(&descriptor).unwrap();
        let object = match parent {
            "" => value.as_object_mut().unwrap(),
            "input" => value["input"].as_object_mut().unwrap(),
            "go" => value["input"]["original_go"].as_object_mut().unwrap(),
            "raw" => value["input"]["original_go"]["original_parsed_limits"]
                .as_object_mut()
                .unwrap(),
            _ => value["input"]["declared_ceilings"]["resource_plan"]
                .as_object_mut()
                .unwrap(),
        };
        object.insert("unexpected".into(), json!(true));
        assert!(
            RoventRootDescriptor::from_json(&value.to_string()).is_err(),
            "{parent}"
        );
    }
    let nonfinite = text.replacen("\"at_ns\":1000000000", "\"at_ns\":NaN", 1);
    assert_ne!(nonfinite, text);
    assert!(RoventRootDescriptor::from_json(&nonfinite).is_err());
    let overflow = text.replacen(
        "\"requested_nodes\":7",
        "\"requested_nodes\":18446744073709551616",
        1,
    );
    assert_ne!(overflow, text);
    assert!(RoventRootDescriptor::from_json(&overflow).is_err());
    assert!(RoventRootDescriptor::from_json(&" ".repeat(MAX_DESCRIPTOR_BYTES + 1)).is_err());
}

#[test]
fn changed_public_fields_or_seals_require_resealing_and_keep_domains_separate() {
    let lock = lock();
    let descriptor = RoventRootDescriptor::seal(input(&lock)).unwrap();
    for change in 0..4 {
        let mut altered = descriptor.clone();
        match change {
            0 => altered.domain = ROVENT_ROOT_BUDGET_DOMAIN.into(),
            1 => altered.root_budget_sha256 = sha_value('0'),
            2 => altered.deadline_seal_sha256 = sha_value('0'),
            _ => altered.input.authority.root += 1,
        }
        assert!(altered.validate().is_err());
        assert!(altered.digest().is_err());
    }
    let payload = canonical(&json!({"same":true})).unwrap();
    assert_ne!(
        domain_hash(ROVENT_ROOT_BUDGET_DOMAIN, &payload),
        domain_hash(ROVENT_ROOT_DEADLINE_DOMAIN, &payload)
    );
}

#[test]
fn nested_key_order_is_canonical_but_array_order_is_preserved() {
    let first = canonical(&json!({"z":{"b":2,"a":1},"a":[{"z":0,"a":1}]})).unwrap();
    let second = canonical(&json!({"a":[{"a":1,"z":0}],"z":{"a":1,"b":2}})).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        String::from_utf8(first).unwrap(),
        r#"{"a":[{"a":1,"z":0}],"z":{"a":1,"b":2}}"#
    );
    assert_ne!(
        canonical(&json!(["e2e4", "d2d4"])).unwrap(),
        canonical(&json!(["d2d4", "e2e4"])).unwrap()
    );
}

#[test]
fn syntax_validation_never_claims_rules_legality_or_live_authority() {
    let lock = lock();
    let mut syntactic = input(&lock);
    syntactic.policy = RoventRootAdmissionPolicy::DynamicGo;
    syntactic.allowed_root_moves = vec![
        "e2e5".into(),
        "a7a8q".into(),
        "a7a8r".into(),
        "a7a8b".into(),
        "a7a8n".into(),
    ];
    syntactic.original_go.searchmoves = Some(syntactic.allowed_root_moves.clone());
    raw(&mut syntactic).searchmoves = Some(syntactic.allowed_root_moves.clone());
    // Tokens are syntactically valid only. No Position, registry, clock or
    // neural provider exists in this fixture and no such proof is returned.
    RoventRootDescriptor::seal(syntactic)
        .unwrap()
        .verify_against(&lock)
        .unwrap();
}
