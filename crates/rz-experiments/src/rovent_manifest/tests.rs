use super::*;

#[test]
fn source_aggregate_has_independent_fixed_vector_and_commit_scope() {
    let components = BTreeMap::from([
        ("model_native".into(), "b".repeat(64)),
        ("control".into(), "c".repeat(64)),
    ]);
    let value = rovent_source_aggregate_sha256(&"a".repeat(40), &components).unwrap();
    // Independently framed with Python hashlib + struct.pack('<I', ...).
    assert_eq!(
        value,
        "7907bf84fed557a761623040c773f826bf3424a628ebfe289eb1d7ec6663de13"
    );
    assert_ne!(
        value,
        rovent_source_aggregate_sha256(&"b".repeat(40), &components).unwrap()
    );
    let mut changed = components.clone();
    changed.insert("control".into(), "d".repeat(64));
    assert_ne!(
        value,
        rovent_source_aggregate_sha256(&"a".repeat(40), &changed).unwrap()
    );
}

#[test]
fn source_aggregate_rejects_unbounded_or_placeholder_identity() {
    let valid = BTreeMap::from([("control".into(), "c".repeat(64))]);
    assert!(rovent_source_aggregate_sha256(&"A".repeat(40), &valid).is_err());
    assert!(rovent_source_aggregate_sha256(&"a".repeat(40), &BTreeMap::new()).is_err());
    let too_many = (0..65)
        .map(|i| (format!("component-{i}"), "c".repeat(64)))
        .collect();
    assert!(rovent_source_aggregate_sha256(&"a".repeat(40), &too_many).is_err());
    for (name, hash) in [
        ("unknown".into(), "c".repeat(64)),
        ("x".repeat(4097), "c".repeat(64)),
        ("control".into(), "0".repeat(64)),
    ] {
        assert!(
            rovent_source_aggregate_sha256(&"a".repeat(40), &BTreeMap::from([(name, hash)]))
                .is_err()
        );
    }
}

fn sha_value(c: char) -> String {
    c.to_string().repeat(64)
}
fn artifact(name: &str) -> ArtifactRef {
    ArtifactRef {
        path: format!("fixture/{name}"),
        sha256: sha_value('a'),
        bytes: 32,
        source: "https://example.org/rovent-fixture".into(),
        license: "MIT".into(),
    }
}
fn model() -> RoventModelPin {
    RoventModelPin {
        semantics: "fixture-model/1".into(),
        encoding: "fixture-input/1".into(),
        policy_head: "legal-candidates/1".into(),
        value_head: "side-to-move-wdl/1".into(),
        implementation_sha256: sha_value('b'),
        weights: artifact("weights.bin"),
        provenance: artifact("provenance.json"),
        diagnostic: false,
    }
}
fn manifest() -> RoventExecutionManifest {
    RoventExecutionManifest {
        domain: ROVENT_EXECUTION_DOMAIN.into(),
        contract_revision: ROVENT_SEARCH_REVISION.into(),
        run_id: "rovent-fixture-1".into(),
        purpose: RoventPurpose::Validation,
        engine: RoventEngineDeclaration {
            source_commit: "a".repeat(40),
            source_components: BTreeMap::from([("control".into(), sha_value('c'))]),
            binary: artifact("rovent"),
            mode: ExecutionMode::CpuPc,
            pc: RoventPcConfiguration {
                proposer: model(),
                critic: model(),
                validator: None,
                backend: RoventBackend::RustCpu,
                device: RoventDevice::Cpu,
                backend_configuration_sha256: sha_value('d'),
                runtime_artifacts: vec![],
                fp32: true,
                tf32: false,
                batch_width: 1,
                ort_intra_threads: None,
                ort_inter_threads: None,
                ort_spinning: None,
            },
            researcher: None,
            teachers: vec![],
            nnuec: None,
            search: RoventSearchDeclaration {
                node_unit: ROVENT_POSITION_VISIT_UNIT.into(),
                requested_nodes: None,
                depth_plies: None,
                mate_moves: None,
                registered_max_chess_plies: NonZeroU16::new(64).unwrap(),
                searchmoves: None,
                search_configuration_sha256: sha_value('e'),
                root_state_and_history_sha256: sha_value('f'),
                rules_profile_sha256: sha_value('1'),
            },
        },
        resources: RoventResourcePlan {
            cpu_quota_millicpus: 2000,
            pools: vec![RoventThreadPool {
                actors: BTreeSet::from([RoventActor::Manager, RoventActor::PcCaller]),
                max_threads: 1,
                cpu_charge_millicpus: 1000,
            }],
            memory_high_bytes: 6 << 30,
            memory_max_bytes: 12 << 30,
            gpu_devices: 0,
            physical_nn_concurrency: 1,
            topology_policy: "shared-control/1".into(),
        },
        coordinator: RoventCoordinatorDeclaration {
            policy_sha256: sha_value('2'),
            questions_max: 1024,
            ancestor_depth_max: 64,
            waiting_jobs_max: 64,
            role_tickets_max: 16,
            cpu_credit_limit: 100_000,
            original_root_budget_sha256: sha_value('3'),
            original_deadline_seal_sha256: sha_value('4'),
        },
        wall_time_max_ms: 60_000,
        cleanup_max_ms: 30_000,
        output_bytes_max: 1 << 20,
        agent_renewable: true,
        provenance: vec![artifact("registration.json")],
    }
}
fn observed<T>(value: T) -> RoventObserved<T> {
    RoventObserved::Observed {
        value,
        method: "fixture-owner/1".into(),
    }
}
fn external(purpose: RoventExternalPurpose) -> RoventExternalProfilePin {
    RoventExternalProfilePin {
        domain: ROVENT_EXTERNAL_PROFILE_DOMAIN.into(),
        profile: artifact("external-profile.json"),
        canonical_sha256: sha_value('8'),
        capability_sha256: sha_value('9'),
        purpose,
        binary: artifact("external-engine"),
        source_commit: "b".repeat(40),
        requested_threads: 1,
        requested_hash_mib: 256,
    }
}
fn receipt(lock: &RoventInputLock) -> RoventExecutionReceipt {
    let m = &lock.manifest;
    RoventExecutionReceipt {
        domain: ROVENT_RECEIPT_DOMAIN.into(),
        run_id: m.run_id.clone(),
        input_lock_sha256: lock.canonical_sha256.clone(),
        functional_accepted: true,
        controlled_comparison_eligible: false,
        source_content_verified: observed(true),
        source_and_binary_evidence: observed(artifact("source.json")),
        model_content_verified: observed(true),
        model_configuration_sha256: observed(m.model_configuration_sha256().unwrap()),
        model_content_evidence: observed(artifact("models.json")),
        pc_device: observed(m.engine.pc.device.clone()),
        backend_configuration_sha256: observed(m.engine.pc.backend_configuration_sha256.clone()),
        calls: observed(RoventExecutedCalls {
            r_calls: 0,
            native_nodes: 0,
            qsearch_nodes: 0,
            see_calls: Some(0),
            external_teacher_queries: 0,
            external_research_queries: 0,
            nnuec_readouts: 0,
            validator_models_loaded: 0,
            validator_models_generated: 0,
            validator_calls: 0,
            completed_nn_inputs: 0,
            search_consumed_evaluations: 0,
        }),
        nodes: observed(RoventNodeObservation {
            unit: ROVENT_POSITION_VISIT_UNIT.into(),
            requested_nodes: m.engine.search.requested_nodes,
            root_entries: 1,
            checked_transitions: 0,
            native_nodes: 0,
            total_node_visits: 1,
        }),
        lifecycle: observed(RoventLifecycleObservation {
            submitted: 0,
            terminal_before_dispatch: 0,
            cache_completed: 0,
            dispatched: 0,
            queued: 0,
            physically_completed: 0,
            in_flight: 0,
            completion_unknown: 0,
            released: 0,
            nn_inputs_completed: 0,
            nn_inputs_accepted: 0,
            cache_evaluations_accepted: 0,
            search_evaluations_consumed: 0,
            cpu_spent: 0,
            cpu_outstanding_reserved: 0,
            root_budget_sha256: m.coordinator.original_root_budget_sha256.clone(),
            original_deadline_seal_sha256: m.coordinator.original_deadline_seal_sha256.clone(),
            authority_anchor_view_and_owner_verified: true,
            archive_closed: true,
            evidence: artifact("lifecycle.json"),
        }),
        goal: observed(RoventGoalObservation {
            completed_chess_plies: None,
            depth_target_satisfied: false,
            rules_terminal_closed_early: false,
            open_required_obligations: 0,
            completed_boundary_evaluations: 0,
            mate: None,
            emitted_score_mate: None,
            bestmove_emitted: false,
            bestmove_output_authorized: false,
            evidence: artifact("goal.json"),
        }),
        root_moves_rules_seal: observed(sha_value('5')),
        external_profiles_verified: observed(vec![]),
        physical_state: observed(RoventPhysicalState::Complete),
        worker_joined_and_processes_closed: observed(true),
        resources: RoventResourceObservation {
            limits_applied: RoventObserved::Unknown,
            process_tree_quota_millicpus: RoventObserved::Unknown,
            memory_high_bytes: RoventObserved::Unknown,
            memory_max_bytes: RoventObserved::Unknown,
            inventory_complete: RoventObserved::Unknown,
            kernel_threads: RoventObserved::Unknown,
            verified_plan_readback_sha256: RoventObserved::Unknown,
            host_peak_bytes: RoventObserved::Unknown,
            vram_peak_bytes: RoventObserved::Unknown,
            evidence: artifact("resources.json"),
        },
        failures: BTreeSet::new(),
        full_gc_requested: false,
        memory_contention: false,
        wall_time_ms: 1000,
        cleanup_ms: 10,
        output_bytes: 1024,
        execution_evidence: artifact("execution.json"),
    }
}

#[test]
fn strict_new_domain_and_lock_do_not_convert_legacy_or_tampered_input() {
    let m = manifest();
    let encoded = serde_json::to_string(&m).unwrap();
    assert!(encoded.contains("\"mode\":\"cpu_pc\""));
    let roundtrip = RoventExecutionManifest::from_json(&encoded).unwrap();
    assert_eq!(roundtrip, m);
    assert!(
        RoventExecutionManifest::from_json(&encoded.replacen(
            "\"domain\":",
            "\"extra\":0,\"domain\":",
            1
        ))
        .is_err()
    );
    assert!(
        RoventExecutionManifest::from_json(&encoded.replacen(
            "\"run_id\":",
            "\"run_id\":\"duplicate\",\"run_id\":",
            1
        ))
        .is_err()
    );
    assert!(
        RoventExecutionManifest::from_json(
            &encoded.replace(ROVENT_EXECUTION_DOMAIN, "rz-pals-execution-v5/1")
        )
        .is_err()
    );
    let mut lock = m.lock().unwrap();
    let lock_json = serde_json::to_string(&lock).unwrap();
    assert_eq!(RoventInputLock::from_json(&lock_json).unwrap(), lock);
    lock.manifest.run_id = "different".into();
    assert!(lock.validate().is_err());
}

fn policy_manifest_v2(
    base: RoventExecutionManifest,
    tie: RoventRootTieDeclarationV2,
) -> RoventExecutionManifestV2 {
    let search_digest = base.engine.search.search_configuration_sha256.clone();
    RoventExecutionManifestV2 {
        domain: ROVENT_EXECUTION_V2_DOMAIN.into(),
        contract_revision: base.contract_revision,
        run_id: base.run_id,
        purpose: base.purpose,
        engine: base.engine,
        resources: base.resources,
        coordinator: base.coordinator,
        wall_time_max_ms: base.wall_time_max_ms,
        cleanup_max_ms: base.cleanup_max_ms,
        output_bytes_max: base.output_bytes_max,
        agent_renewable: base.agent_renewable,
        provenance: base.provenance,
        startup_document: artifact("startup-v4.json"),
        startup_configuration_sha256: sha_value('4'),
        startup_search_configuration_sha256: search_digest,
        own_r_root_tie: tie,
    }
}

#[test]
fn original_versioned_lock_owner_retains_exact_arc_and_borrowed_fields() {
    use crate::RoventOriginalInputLock;
    use std::sync::Arc;

    let original_v1 = Arc::new(manifest().lock().unwrap());
    let original_v2 = Arc::new(
        policy_manifest_v2(
            manifest(),
            RoventRootTieDeclarationV2::NotApplicable {
                reason: RoventOwnRNotApplicableV2::PcOnly,
            },
        )
        .lock()
        .unwrap(),
    );
    let v1_bytes = serde_json::to_vec(original_v1.as_ref()).unwrap();
    let v2_bytes = serde_json::to_vec(original_v2.as_ref()).unwrap();
    let v1 = RoventOriginalInputLock::from_v1(Arc::clone(&original_v1)).unwrap();
    let v2 = RoventOriginalInputLock::from_v2(Arc::clone(&original_v2)).unwrap();

    assert!(Arc::ptr_eq(v1.v1().unwrap(), &original_v1));
    assert!(Arc::ptr_eq(v2.v2().unwrap(), &original_v2));
    assert!(v1.v2().is_none() && v2.v1().is_none());
    assert!(v1.same_original_owner(&v1.clone()));
    assert!(v2.same_original_owner(&v2.clone()));
    let separate = RoventOriginalInputLock::from_v2(Arc::new((*original_v2).clone())).unwrap();
    assert!(!v2.same_original_owner(&separate));
    assert!(!v1.same_original_owner(&v2));
    assert_eq!(v1.common().canonical_sha256, original_v1.canonical_sha256);
    assert_eq!(v2.common().canonical_sha256, original_v2.canonical_sha256);
    assert!(std::ptr::eq(
        v2.common().engine,
        &original_v2.manifest.engine
    ));
    assert!(std::ptr::eq(
        v1.common().resources,
        &original_v1.manifest.resources
    ));
    assert!(v1.validate().is_ok() && v2.validate().is_ok());
    assert_eq!(serde_json::to_vec(original_v1.as_ref()).unwrap(), v1_bytes);
    assert_eq!(serde_json::to_vec(original_v2.as_ref()).unwrap(), v2_bytes);
}

#[test]
fn original_versioned_lock_owner_returns_invalid_original_to_caller() {
    use crate::RoventOriginalInputLock;
    use std::sync::Arc;

    let mut invalid_v1 = manifest().lock().unwrap();
    invalid_v1.canonical_sha256 = sha_value('9');
    let original_v1 = Arc::new(invalid_v1);
    let failure = RoventOriginalInputLock::from_v1(Arc::clone(&original_v1)).unwrap_err();
    assert!(Arc::ptr_eq(&failure.original, &original_v1));

    let mut invalid_v2 = policy_manifest_v2(
        manifest(),
        RoventRootTieDeclarationV2::NotApplicable {
            reason: RoventOwnRNotApplicableV2::PcOnly,
        },
    )
    .lock()
    .unwrap();
    invalid_v2.canonical_sha256 = sha_value('9');
    let original_v2 = Arc::new(invalid_v2);
    let failure = RoventOriginalInputLock::from_v2(Arc::clone(&original_v2)).unwrap_err();
    assert!(Arc::ptr_eq(&failure.original, &original_v2));
}

#[test]
fn policy_v2_domains_lock_and_pc_observations_do_not_downgrade() {
    let old = manifest();
    let old_bytes = old.canonical_bytes().unwrap();
    let m = policy_manifest_v2(
        old.clone(),
        RoventRootTieDeclarationV2::NotApplicable {
            reason: RoventOwnRNotApplicableV2::PcOnly,
        },
    );
    let wire = serde_json::to_string(&m).unwrap();
    assert!(RoventExecutionManifest::from_json(&wire).is_err());
    assert_eq!(RoventExecutionManifestV2::from_json(&wire).unwrap(), m);
    assert_eq!(old.canonical_bytes().unwrap(), old_bytes);
    let mut lock = m.lock().unwrap();
    let lock_wire = serde_json::to_string(&lock).unwrap();
    assert!(RoventInputLock::from_json(&lock_wire).is_err());
    assert_eq!(RoventInputLockV2::from_json(&lock_wire).unwrap(), lock);
    let mut r = RoventExecutionReceiptV2 {
        domain: ROVENT_RECEIPT_V2_DOMAIN.into(),
        run_id: lock.manifest.run_id.clone(),
        input_lock_sha256: lock.canonical_sha256.clone(),
        own_r_activity: RoventOwnRActivityV2::Unknown,
        root_tie_observation: RoventRootTieObservationV2::NotApplicable {
            reason: RoventOwnRNotApplicableV2::PcOnly,
            evidence: artifact("pc-no-r.json"),
        },
        execution_evidence: artifact("execution.json"),
    };
    assert_eq!(
        r.policy_consistency(&lock).unwrap(),
        RoventPolicyRecordConsistencyV2::Unknown
    );
    r.own_r_activity = RoventOwnRActivityV2::NotApplicable {
        evidence: artifact("pc-activity.json"),
    };
    assert_eq!(
        r.policy_consistency(&lock).unwrap(),
        RoventPolicyRecordConsistencyV2::ConsistentUnattested
    );
    let old_lock = old.lock().unwrap();
    assert!(
        RoventExecutionReceipt::from_json(&serde_json::to_string(&r).unwrap(), &old_lock).is_err()
    );
    lock.manifest.startup_configuration_sha256 = sha_value('9');
    assert!(lock.validate().is_err());
}

#[test]
fn policy_v2_distinguishes_own_not_issued_ready_unknown_and_mismatch() {
    let mut base = manifest();
    base.engine.mode = ExecutionMode::CpuPcr;
    base.engine.researcher = Some(RoventResearcher::Own {
        implementation_sha256: sha_value('6'),
        search_configuration_sha256: sha_value('7'),
    });
    base.resources.pools[0].actors.insert(RoventActor::OwnR);
    base.engine.nnuec = Some(RoventNnuecPin {
        schema: "nnuec-fixture/1".into(),
        semantics: "fixture/1".into(),
        encoding: "fixture/1".into(),
        implementation_sha256: sha_value('1'),
        configuration_sha256: sha_value('2'),
        weights: artifact("nnuec.bin"),
        provenance: artifact("nnuec-provenance.json"),
        diagnostic: false,
        context_namespace_policy_sha256: sha_value('3'),
        local_namespace_policy_sha256: sha_value('4'),
        search_conditions_sha256: sha_value('5'),
    });
    let m = policy_manifest_v2(
        base,
        RoventRootTieDeclarationV2::Own {
            policy: RoventRootTiePolicyV2::RulesOrderConfirmed,
            expected_search_identity: "fixture-cpu/2".into(),
            expected_worker_conditions_sha256: sha_value('6'),
        },
    );
    // NNUEC plan conditions and actual CPU worker conditions are distinct domains.
    assert_ne!(
        m.engine.nnuec.as_ref().unwrap().search_conditions_sha256,
        sha_value('6')
    );
    let lock = m.lock().unwrap();
    let mut r = RoventExecutionReceiptV2 {
        domain: ROVENT_RECEIPT_V2_DOMAIN.into(),
        run_id: lock.manifest.run_id.clone(),
        input_lock_sha256: lock.canonical_sha256.clone(),
        own_r_activity: RoventOwnRActivityV2::NotIssued {
            evidence: artifact("not-issued.json"),
        },
        root_tie_observation: RoventRootTieObservationV2::NotIssued {
            reason: RoventOwnRNotIssuedReasonV2::NnuecPreMaterializationStop,
            evidence: artifact("pre-materialization-stop.json"),
        },
        execution_evidence: artifact("execution.json"),
    };
    assert_eq!(
        r.policy_consistency(&lock).unwrap(),
        RoventPolicyRecordConsistencyV2::ConsistentUnattested
    );
    // A pre-materialization stop requires an explicit, structurally valid
    // witness; an empty placeholder cannot be used as evidence.
    if let RoventRootTieObservationV2::NotIssued { evidence, .. } = &mut r.root_tie_observation {
        evidence.bytes = 0;
    }
    assert!(r.policy_consistency(&lock).is_err());
    if let RoventRootTieObservationV2::NotIssued { evidence, .. } = &mut r.root_tie_observation {
        *evidence = artifact("pre-materialization-stop.json");
    }
    let mut board_only_manifest = lock.manifest.clone();
    board_only_manifest.engine.nnuec = None;
    let board_only_lock = board_only_manifest.lock().unwrap();
    let mut board_only_receipt = r.clone();
    board_only_receipt.input_lock_sha256 = board_only_lock.canonical_sha256.clone();
    assert_eq!(
        board_only_receipt
            .policy_consistency(&board_only_lock)
            .unwrap(),
        RoventPolicyRecordConsistencyV2::Mismatch
    );
    r.own_r_activity = RoventOwnRActivityV2::Unknown;
    assert_eq!(
        r.policy_consistency(&lock).unwrap(),
        RoventPolicyRecordConsistencyV2::Unknown
    );
    r.own_r_activity = RoventOwnRActivityV2::Issued {
        jobs: std::num::NonZeroU64::new(1).unwrap(),
        owner_id: "owner-1".into(),
        root_authority_sha256: sha_value('8'),
        job_set_sha256: sha_value('9'),
        evidence: artifact("job.json"),
    };
    assert_eq!(
        r.policy_consistency(&lock).unwrap(),
        RoventPolicyRecordConsistencyV2::Mismatch
    );
    r.root_tie_observation = RoventRootTieObservationV2::Observed {
        policy: RoventRootTiePolicyV2::RulesOrderConfirmed,
        search_identity: "fixture-cpu/2".into(),
        worker_conditions_sha256: sha_value('6'),
        nnuec_plan: RoventNnuecPlanObservationV2::Observed {
            conditions_sha256: sha_value('5'),
            evidence: artifact("nnuec-plan.json"),
        },
        root_state_and_history_sha256: lock
            .manifest
            .engine
            .search
            .root_state_and_history_sha256
            .clone(),
        root_authority_sha256: sha_value('8'),
        owner_id: "owner-1".into(),
        ready_evidence: artifact("ready.json"),
        retirement_evidence: None,
    };
    assert_eq!(
        r.policy_consistency(&lock).unwrap(),
        RoventPolicyRecordConsistencyV2::ConsistentUnattested
    );
    // Ready is an owner observation, so a subsequent R job need not have
    // been issued. The not-issued witness is still required independently.
    let issued_activity = r.own_r_activity.clone();
    r.own_r_activity = RoventOwnRActivityV2::NotIssued {
        evidence: artifact("ready-but-no-job.json"),
    };
    assert_eq!(
        r.policy_consistency(&lock).unwrap(),
        RoventPolicyRecordConsistencyV2::ConsistentUnattested
    );
    r.own_r_activity = issued_activity;
    let (parsed, verdict) =
        RoventExecutionReceiptV2::parse_and_assess(&serde_json::to_string(&r).unwrap(), &lock)
            .unwrap();
    assert_eq!(parsed, r);
    assert_eq!(
        verdict,
        RoventPolicyRecordConsistencyV2::ConsistentUnattested
    );
    if let RoventRootTieObservationV2::Observed { nnuec_plan, .. } = &mut r.root_tie_observation {
        *nnuec_plan = RoventNnuecPlanObservationV2::Unknown;
    }
    assert_eq!(
        r.policy_consistency(&lock).unwrap(),
        RoventPolicyRecordConsistencyV2::Unknown
    );
    if let RoventRootTieObservationV2::Observed { nnuec_plan, .. } = &mut r.root_tie_observation {
        *nnuec_plan = RoventNnuecPlanObservationV2::Observed {
            conditions_sha256: sha_value('7'),
            evidence: artifact("wrong-plan.json"),
        };
    }
    assert_eq!(
        r.policy_consistency(&lock).unwrap(),
        RoventPolicyRecordConsistencyV2::Mismatch
    );
    if let RoventRootTieObservationV2::Observed {
        nnuec_plan,
        root_authority_sha256,
        ..
    } = &mut r.root_tie_observation
    {
        *nnuec_plan = RoventNnuecPlanObservationV2::Observed {
            conditions_sha256: sha_value('5'),
            evidence: artifact("nnuec-plan.json"),
        };
        *root_authority_sha256 = sha_value('9');
    }
    assert_eq!(
        r.policy_consistency(&lock).unwrap(),
        RoventPolicyRecordConsistencyV2::Mismatch
    );
    if let RoventRootTieObservationV2::Observed { policy, .. } = &mut r.root_tie_observation {
        *policy = RoventRootTiePolicyV2::LegacyFirstSearched;
    }
    if let RoventRootTieObservationV2::Observed {
        root_authority_sha256,
        ..
    } = &mut r.root_tie_observation
    {
        *root_authority_sha256 = sha_value('8');
    }
    r.own_r_activity = RoventOwnRActivityV2::Unknown;
    assert_eq!(
        r.policy_consistency(&lock).unwrap(),
        RoventPolicyRecordConsistencyV2::Mismatch
    );
}

#[test]
fn policy_v2_external_researcher_is_not_pc_or_own() {
    let mut base = manifest();
    base.engine.mode = ExecutionMode::CpuPcr;
    base.engine.researcher = Some(RoventResearcher::External {
        profile: Box::new(external(RoventExternalPurpose::ResearchChecker)),
    });
    base.resources.pools[0]
        .actors
        .insert(RoventActor::ExternalR);
    let mut m = policy_manifest_v2(
        base,
        RoventRootTieDeclarationV2::NotApplicable {
            reason: RoventOwnRNotApplicableV2::ExternalResearcher,
        },
    );
    assert!(m.validate().is_ok());
    m.own_r_root_tie = RoventRootTieDeclarationV2::NotApplicable {
        reason: RoventOwnRNotApplicableV2::PcOnly,
    };
    assert!(m.validate().is_err());
}

#[test]
fn all_four_modes_require_the_actual_device_and_researcher_assembly() {
    for mode in [
        ExecutionMode::CpuPc,
        ExecutionMode::CpuPcr,
        ExecutionMode::HybridPc,
        ExecutionMode::HybridPcr,
    ] {
        let mut m = manifest();
        m.engine.mode = mode;
        if mode.has_native_r() {
            m.engine.researcher = Some(RoventResearcher::Own {
                implementation_sha256: sha_value('6'),
                search_configuration_sha256: sha_value('7'),
            });
            m.resources.pools.push(RoventThreadPool {
                actors: BTreeSet::from([RoventActor::OwnR]),
                max_threads: 1,
                cpu_charge_millicpus: 1000,
            });
        }
        if mode.pc_compute() == PcCompute::Accelerator {
            m.engine.pc.backend = RoventBackend::OrtCuda;
            m.engine.pc.device = RoventDevice::Cuda {
                ordinal: 0,
                registered_device: "test-gpu/1".into(),
            };
            m.engine.pc.runtime_artifacts = vec![artifact("ort.so")];
            m.engine.pc.ort_intra_threads = Some(1);
            m.engine.pc.ort_inter_threads = Some(1);
            m.engine.pc.ort_spinning = Some(false);
            m.resources.gpu_devices = 1;
        }
        m.validate().unwrap();
        let mut wrong = m.clone();
        wrong.engine.pc.device = if wrong.engine.pc.device == RoventDevice::Cpu {
            RoventDevice::Cuda {
                ordinal: 0,
                registered_device: "test-gpu/1".into(),
            }
        } else {
            RoventDevice::Cpu
        };
        assert!(wrong.validate().is_err());
        let mut wrong_r = m;
        wrong_r.engine.researcher = if mode.has_native_r() {
            None
        } else {
            Some(RoventResearcher::Own {
                implementation_sha256: sha_value('6'),
                search_configuration_sha256: sha_value('7'),
            })
        };
        assert!(wrong_r.validate().is_err());
    }
}

#[test]
fn root_limits_and_selective_depth_do_not_become_empty_unrestricted_or_local_depth() {
    let mut m = manifest();
    for moves in [
        vec![],
        vec!["e2e2".into()],
        vec!["0000".into()],
        vec!["e2e4".into(), "e2e4".into()],
    ] {
        m.engine.search.searchmoves = Some(moves);
        assert!(m.validate().is_err());
    }
    m.engine.search.searchmoves = Some(vec!["e2e4".into()]);
    m.engine.search.depth_plies = NonZeroU16::new(3);
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    let RoventObserved::Observed { value: goal, .. } = &mut r.goal else {
        unreachable!()
    };
    goal.completed_chess_plies = Some(3);
    goal.depth_target_satisfied = true;
    assert!(r.validate(&lock).is_err());
    let RoventObserved::Observed { value: goal, .. } = &mut r.goal else {
        unreachable!()
    };
    goal.completed_boundary_evaluations = 1;
    goal.open_required_obligations = 1;
    assert!(r.validate(&lock).is_err());
    let RoventObserved::Observed { value: goal, .. } = &mut r.goal else {
        unreachable!()
    };
    goal.open_required_obligations = 0;
    r.validate(&lock).unwrap();
}

#[test]
fn r_free_and_deployment_reject_observed_hidden_calls_or_v_generation() {
    let lock = manifest().lock().unwrap();
    for which in 0..5 {
        let mut r = receipt(&lock);
        let RoventObserved::Observed { value: calls, .. } = &mut r.calls else {
            unreachable!()
        };
        match which {
            0 => calls.r_calls = 1,
            1 => calls.native_nodes = 1,
            2 => calls.qsearch_nodes = 1,
            3 => calls.see_calls = Some(1),
            _ => calls.external_teacher_queries = 1,
        }
        assert!(r.validate(&lock).is_err());
    }
    let mut m = manifest();
    m.purpose = RoventPurpose::Deployment;
    m.engine.pc.validator = Some(model());
    assert!(m.validate().is_err());
    m.engine.pc.validator = None;
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    let RoventObserved::Observed { value: calls, .. } = &mut r.calls else {
        unreachable!()
    };
    calls.validator_models_generated = 1;
    assert!(r.validate(&lock).is_err());
}

#[test]
fn physical_reservations_and_overflow_cannot_become_accepted_shutdown() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    r.validate(&lock).unwrap();
    let RoventObserved::Observed { value: l, .. } = &mut r.lifecycle else {
        unreachable!()
    };
    l.cpu_outstanding_reserved = 1;
    assert!(r.validate(&lock).is_err());
    r.functional_accepted = false;
    r.validate(&lock).unwrap();
    let RoventObserved::Observed { value: l, .. } = &mut r.lifecycle else {
        unreachable!()
    };
    l.cpu_spent = u64::MAX;
    assert!(r.validate(&lock).is_err());
    let mut r = receipt(&lock);
    r.functional_accepted = false;
    r.physical_state = observed(RoventPhysicalState::QuarantinedUnknown);
    assert!(r.validate(&lock).is_err());
    r.failures.insert(RoventFailure::PhysicalCompletionUnknown);
    r.validate(&lock).unwrap();

    let mut r = receipt(&lock);
    r.functional_accepted = false;
    let RoventObserved::Observed { value: l, .. } = &mut r.lifecycle else {
        unreachable!()
    };
    l.submitted = 1;
    l.dispatched = 1;
    l.in_flight = 1;
    l.released = 1;
    assert!(r.validate(&lock).is_err());
    let mut r = receipt(&lock);
    r.functional_accepted = false;
    let RoventObserved::Observed { value: l, .. } = &mut r.lifecycle else {
        unreachable!()
    };
    l.nn_inputs_completed = 1;
    assert!(r.validate(&lock).is_err());
    let mut r = receipt(&lock);
    r.functional_accepted = false;
    let RoventObserved::Observed { value: l, .. } = &mut r.lifecycle else {
        unreachable!()
    };
    l.cache_evaluations_accepted = 1;
    assert!(r.validate(&lock).is_err());
}

#[test]
fn model_and_backend_content_unknown_cannot_inherit_acceptance() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    r.model_content_verified = RoventObserved::Unknown;
    assert!(r.validate(&lock).is_err());
    r.functional_accepted = false;
    r.validate(&lock).unwrap();
    let mut r = receipt(&lock);
    r.model_configuration_sha256 = observed(sha_value('9'));
    assert!(r.validate(&lock).is_err());
    let mut r = receipt(&lock);
    r.pc_device = observed(RoventDevice::Cuda {
        ordinal: 0,
        registered_device: "other-device".into(),
    });
    assert!(r.validate(&lock).is_err());
}

#[test]
fn thread_inventory_and_millicpu_demand_are_distinct_and_bounded() {
    let mut m = manifest();
    m.resources.pools[0].max_threads = 20;
    m.validate().unwrap();
    m.resources.pools[0].cpu_charge_millicpus = 2001;
    assert!(m.validate().is_err());
    m.resources.pools[0].cpu_charge_millicpus = 1000;
    m.resources.pools.push(m.resources.pools[0].clone());
    assert!(m.validate().is_err());
    let mut m = manifest();
    m.engine.pc.backend = RoventBackend::OrtCpu;
    m.engine.pc.runtime_artifacts = vec![artifact("ort.so")];
    m.engine.pc.ort_intra_threads = Some(4);
    m.engine.pc.ort_inter_threads = Some(1);
    m.engine.pc.ort_spinning = Some(false);
    assert!(m.validate().is_err());
    m.resources.pools.push(RoventThreadPool {
        actors: BTreeSet::from([RoventActor::OrtIntra]),
        max_threads: 3,
        cpu_charge_millicpus: 1000,
    });
    m.validate().unwrap();
}

#[test]
fn unknown_resources_gc_and_contention_never_support_controlled_comparison() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    r.controlled_comparison_eligible = true;
    assert!(r.validate(&lock).is_err());
    r.resources.limits_applied = observed(true);
    r.resources.process_tree_quota_millicpus = observed(2000);
    r.resources.memory_high_bytes = observed(6 << 30);
    r.resources.memory_max_bytes = observed(12 << 30);
    r.resources.inventory_complete = observed(true);
    r.resources.kernel_threads = observed(1);
    r.resources.verified_plan_readback_sha256 =
        observed(lock.manifest.resources.canonical_sha256().unwrap());
    r.resources.host_peak_bytes = observed(1024);
    r.validate(&lock).unwrap();
    assert_eq!(r.resources.vram_peak_bytes, RoventObserved::Unknown);
    r.full_gc_requested = true;
    assert!(r.validate(&lock).is_err());
    r.full_gc_requested = false;
    r.memory_contention = true;
    assert!(r.validate(&lock).is_err());
}

#[test]
fn forced_mate_upper_bound_is_distinct_from_exact_uci_score_and_restricted_loss() {
    let mut m = manifest();
    m.engine.search.mate_moves = NonZeroU16::new(2);
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    let RoventObserved::Observed { value: goal, .. } = &mut r.goal else {
        unreachable!()
    };
    goal.mate = Some(RoventMateObservation::ForcedMateWithin {
        root_wins: true,
        upper_plies: 3,
        restricted_loss: false,
        evidence: artifact("proof.json"),
    });
    r.validate(&lock).unwrap();
    let RoventObserved::Observed { value: goal, .. } = &mut r.goal else {
        unreachable!()
    };
    goal.emitted_score_mate = Some(2);
    assert!(r.validate(&lock).is_err());
    let RoventObserved::Observed { value: goal, .. } = &mut r.goal else {
        unreachable!()
    };
    goal.mate = Some(RoventMateObservation::ExactMateDistance {
        root_wins: true,
        plies: 3,
        restricted_loss: false,
        evidence: artifact("proof.json"),
    });
    r.validate(&lock).unwrap();
    let mut m = manifest();
    m.engine.search.searchmoves = Some(vec!["e2e4".into()]);
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    let RoventObserved::Observed { value: goal, .. } = &mut r.goal else {
        unreachable!()
    };
    goal.mate = Some(RoventMateObservation::ForcedMateWithin {
        root_wins: false,
        upper_plies: 2,
        restricted_loss: false,
        evidence: artifact("proof.json"),
    });
    assert!(r.validate(&lock).is_err());
}

#[test]
fn external_profiles_keep_teacher_checker_opponent_purpose_and_verified_pin_separate() {
    let mut m = manifest();
    m.purpose = RoventPurpose::Collection;
    m.engine
        .teachers
        .push(external(RoventExternalPurpose::Teacher));
    assert!(m.validate().is_err()); // P-C cannot secretly add CPU_T.
    m.engine.mode = ExecutionMode::CpuPcr;
    m.engine.researcher = Some(RoventResearcher::External {
        profile: Box::new(external(RoventExternalPurpose::ArenaOpponent)),
    });
    m.resources.pools.push(RoventThreadPool {
        actors: BTreeSet::from([RoventActor::ExternalR, RoventActor::Teacher]),
        max_threads: 2,
        cpu_charge_millicpus: 1000,
    });
    assert!(m.validate().is_err());
    let mut research_profile = external(RoventExternalPurpose::ResearchChecker);
    research_profile.canonical_sha256 = sha_value('7');
    m.engine.researcher = Some(RoventResearcher::External {
        profile: Box::new(research_profile),
    });
    m.validate().unwrap();
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    assert!(r.validate(&lock).is_err());
    r.external_profiles_verified = observed(vec![sha_value('8'), sha_value('7')]);
    r.validate(&lock).unwrap();
    r.external_profiles_verified = RoventObserved::Unknown;
    assert!(r.validate(&lock).is_err());
}

#[test]
fn v2_pc_collection_can_declare_independent_teacher_without_changing_v1() {
    for mode in [ExecutionMode::CpuPc, ExecutionMode::HybridPc] {
        for shared_pool in [false, true] {
            let mut original = manifest();
            original.purpose = RoventPurpose::Collection;
            original.engine.mode = mode;
            original
                .engine
                .teachers
                .push(external(RoventExternalPurpose::Teacher));
            if mode.pc_compute() == PcCompute::Accelerator {
                original.engine.pc.backend = RoventBackend::OrtCuda;
                original.engine.pc.device = RoventDevice::Cuda {
                    ordinal: 0,
                    registered_device: "test-gpu/1".into(),
                };
                original.engine.pc.runtime_artifacts = vec![artifact("ort.so")];
                original.engine.pc.ort_intra_threads = Some(1);
                original.engine.pc.ort_inter_threads = Some(1);
                original.engine.pc.ort_spinning = Some(false);
                original.resources.gpu_devices = 1;
            }
            if shared_pool {
                original.resources.pools[0]
                    .actors
                    .insert(RoventActor::Teacher);
            } else {
                original.resources.pools.push(RoventThreadPool {
                    actors: BTreeSet::from([RoventActor::Teacher]),
                    max_threads: 1,
                    cpu_charge_millicpus: 1000,
                });
            }
            assert!(original.validate().is_err(), "V1 stays P-C teacher-free");
            let v2 = policy_manifest_v2(
                original,
                RoventRootTieDeclarationV2::NotApplicable {
                    reason: RoventOwnRNotApplicableV2::PcOnly,
                },
            );
            v2.validate().unwrap();
            let lock = v2.lock().unwrap();
            let bytes = serde_json::to_string(&lock).unwrap();
            assert_eq!(RoventInputLockV2::from_json(&bytes).unwrap(), lock);
        }
    }
}

#[test]
fn v2_pc_collection_teacher_preserves_pin_purpose_and_real_resource_checks() {
    let mut original = manifest();
    original.purpose = RoventPurpose::Collection;
    original
        .engine
        .teachers
        .push(external(RoventExternalPurpose::Teacher));
    original.resources.pools.push(RoventThreadPool {
        actors: BTreeSet::from([RoventActor::Teacher]),
        max_threads: 1,
        cpu_charge_millicpus: 1000,
    });
    let v2 = policy_manifest_v2(
        original,
        RoventRootTieDeclarationV2::NotApplicable {
            reason: RoventOwnRNotApplicableV2::PcOnly,
        },
    );
    v2.validate().unwrap();
    for purpose in [RoventPurpose::Validation, RoventPurpose::Deployment] {
        let mut wrong = v2.clone();
        wrong.purpose = purpose;
        assert!(wrong.validate().is_err());
    }
    let mut wrong = v2.clone();
    wrong.engine.teachers[0].purpose = RoventExternalPurpose::ArenaOpponent;
    assert!(wrong.validate().is_err());
    let mut wrong = v2.clone();
    wrong.engine.teachers.push(wrong.engine.teachers[0].clone());
    wrong.resources.pools[1].max_threads = 2;
    assert!(wrong.validate().is_err(), "duplicate canonical teacher");
    let mut wrong = v2.clone();
    wrong.engine.teachers[0].profile.sha256 = "invalid".into();
    assert!(wrong.validate().is_err());
    let mut wrong = v2.clone();
    wrong.resources.pools.pop();
    assert!(
        wrong.validate().is_err(),
        "teacher must have a real CPU pool"
    );
    let mut wrong = v2.clone();
    wrong.resources.cpu_quota_millicpus = 1000;
    assert!(wrong.validate().is_err(), "all original pool charges count");
    let mut wrong = v2;
    wrong.engine.researcher = Some(RoventResearcher::Own {
        implementation_sha256: sha_value('6'),
        search_configuration_sha256: sha_value('7'),
    });
    assert!(
        wrong.validate().is_err(),
        "teacher does not authorize P-C R work"
    );
}

#[test]
fn v2_pcr_collection_keeps_the_original_r_and_teacher_validation() {
    for mode in [ExecutionMode::CpuPcr, ExecutionMode::HybridPcr] {
        let mut original = manifest();
        original.purpose = RoventPurpose::Collection;
        original.engine.mode = mode;
        original.engine.researcher = Some(RoventResearcher::Own {
            implementation_sha256: sha_value('6'),
            search_configuration_sha256: sha_value('7'),
        });
        original
            .engine
            .teachers
            .push(external(RoventExternalPurpose::Teacher));
        original.resources.pools.push(RoventThreadPool {
            actors: BTreeSet::from([RoventActor::OwnR, RoventActor::Teacher]),
            max_threads: 2,
            cpu_charge_millicpus: 1000,
        });
        if mode.pc_compute() == PcCompute::Accelerator {
            original.engine.pc.backend = RoventBackend::OrtCuda;
            original.engine.pc.device = RoventDevice::Cuda {
                ordinal: 0,
                registered_device: "test-gpu/1".into(),
            };
            original.engine.pc.runtime_artifacts = vec![artifact("ort.so")];
            original.engine.pc.ort_intra_threads = Some(1);
            original.engine.pc.ort_inter_threads = Some(1);
            original.engine.pc.ort_spinning = Some(false);
            original.resources.gpu_devices = 1;
        }
        original.validate().unwrap();
        let v2 = policy_manifest_v2(
            original,
            RoventRootTieDeclarationV2::Own {
                policy: RoventRootTiePolicyV2::RulesOrderConfirmed,
                expected_search_identity: "own-r-fixture/1".into(),
                expected_worker_conditions_sha256: sha_value('9'),
            },
        );
        v2.validate().unwrap();
        let mut wrong = v2;
        wrong.engine.researcher = None;
        assert!(wrong.validate().is_err());
    }
}

#[test]
fn pcr_mode_does_not_authorize_unregistered_teacher_or_nnuec_work() {
    let mut m = manifest();
    m.engine.mode = ExecutionMode::CpuPcr;
    m.engine.researcher = Some(RoventResearcher::Own {
        implementation_sha256: sha_value('6'),
        search_configuration_sha256: sha_value('7'),
    });
    m.resources.pools.push(RoventThreadPool {
        actors: BTreeSet::from([RoventActor::OwnR]),
        max_threads: 1,
        cpu_charge_millicpus: 1000,
    });
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    let RoventObserved::Observed { value: calls, .. } = &mut r.calls else {
        unreachable!()
    };
    calls.external_teacher_queries = 1;
    assert!(r.validate(&lock).is_err());
    let mut r = receipt(&lock);
    let RoventObserved::Observed { value: calls, .. } = &mut r.calls else {
        unreachable!()
    };
    calls.nnuec_readouts = 1;
    assert!(r.validate(&lock).is_err());
}

#[test]
fn original_position_visit_limit_is_not_a_simulation_or_native_only_budget() {
    let mut m = manifest();
    m.engine.search.requested_nodes = Some(2);
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    let RoventObserved::Observed { value: nodes, .. } = &mut r.nodes else {
        unreachable!()
    };
    nodes.checked_transitions = 1;
    nodes.total_node_visits = 2;
    r.validate(&lock).unwrap();
    let RoventObserved::Observed { value: nodes, .. } = &mut r.nodes else {
        unreachable!()
    };
    nodes.checked_transitions = 2;
    nodes.total_node_visits = 3;
    assert!(r.validate(&lock).is_err());
    let mut r = receipt(&lock);
    let RoventObserved::Observed { value: nodes, .. } = &mut r.nodes else {
        unreachable!()
    };
    nodes.checked_transitions = u64::MAX;
    nodes.total_node_visits = 0;
    assert!(r.validate(&lock).is_err());
    let mut r = receipt(&lock);
    let RoventObserved::Observed { value: nodes, .. } = &mut r.nodes else {
        unreachable!()
    };
    nodes.unit = "simulations".into();
    assert!(r.validate(&lock).is_err());
}

#[test]
fn early_rules_terminal_satisfies_depth_without_fabricating_unreached_plies() {
    let mut m = manifest();
    m.engine.search.depth_plies = NonZeroU16::new(8);
    let lock = m.lock().unwrap();
    let mut r = receipt(&lock);
    let RoventObserved::Observed { value: goal, .. } = &mut r.goal else {
        unreachable!()
    };
    goal.completed_chess_plies = Some(1);
    goal.depth_target_satisfied = true;
    goal.completed_boundary_evaluations = 1;
    assert!(r.validate(&lock).is_err()); // A short NN/PV line cannot close the target.
    let RoventObserved::Observed { value: goal, .. } = &mut r.goal else {
        unreachable!()
    };
    goal.rules_terminal_closed_early = true;
    r.validate(&lock).unwrap();
    let roundtrip =
        RoventExecutionReceipt::from_json(&serde_json::to_string(&r).unwrap(), &lock).unwrap();
    let RoventObserved::Observed { value: goal, .. } = roundtrip.goal else {
        unreachable!()
    };
    assert_eq!(goal.completed_chess_plies, Some(1));
    assert!(goal.depth_target_satisfied && goal.rules_terminal_closed_early);
    let RoventObserved::Observed { value: goal, .. } = &mut r.goal else {
        unreachable!()
    };
    goal.open_required_obligations = 1;
    assert!(r.validate(&lock).is_err()); // A terminal on one branch cannot hide open duties.
    let RoventObserved::Observed { value: goal, .. } = &mut r.goal else {
        unreachable!()
    };
    goal.open_required_obligations = 0;
    goal.completed_chess_plies = Some(8);
    assert!(r.validate(&lock).is_err()); // Early closure must retain the real reached ply.
}

#[test]
fn failed_mode_observations_are_preserved_only_with_explicit_failure() {
    let lock = manifest().lock().unwrap();
    let mut r = receipt(&lock);
    let RoventObserved::Observed { value: calls, .. } = &mut r.calls else {
        unreachable!()
    };
    calls.external_teacher_queries = 1;
    r.functional_accepted = false;
    assert!(r.validate(&lock).is_err());
    r.failures.insert(RoventFailure::ModeViolation);
    r.validate(&lock).unwrap();
    r.functional_accepted = true;
    assert!(r.validate(&lock).is_err());
}
