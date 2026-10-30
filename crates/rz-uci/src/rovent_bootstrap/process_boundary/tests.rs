//! Focused source/process-boundary fixtures. No model or engine is created.
use super::*;
use std::{
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

fn control() -> RoventStartupControl {
    let until = Instant::now() + Duration::from_secs(30);
    RoventStartupControl {
        until,
        cleanup_until: until + Duration::from_secs(1),
        cancellation: Arc::new(AtomicBool::new(false)),
    }
}

#[test]
fn exact_embedded_components_accept_and_unknown_or_changed_bytes_reject() {
    let actual = compiled_process_source_components().unwrap();
    verify_component_declaration(&actual, None, false).unwrap();
    assert_eq!(
        actual.get(PROCESS_SOURCE_COMPONENT),
        Some(&hex(compiled_process_boundary_source_digest()))
    );
    assert_eq!(
        actual.get("rovent_bootstrap"),
        Some(&hex(compiled_bootstrap_source_digest()))
    );
    let mut unknown = actual.clone();
    unknown.insert("unregistered-final-session".into(), "1".repeat(64));
    assert_eq!(
        verify_component_declaration(&unknown, None, false)
            .unwrap_err()
            .kind,
        ProcessBoundaryFailure::SourceComponent
    );
    let mut wrong = actual.clone();
    wrong.insert("rules".into(), "0".repeat(64));
    assert!(verify_component_declaration(&wrong, None, false).is_err());
    let mut missing = actual;
    missing.remove(PROCESS_SOURCE_COMPONENT);
    assert!(verify_component_declaration(&missing, None, false).is_err());
}

#[test]
fn own_researcher_sources_are_required_without_requiring_r_for_pc_only() {
    let mut declared = compiled_process_source_components().unwrap();
    for key in ["cpu_search", "cpu_worker", "cpu_credit"] {
        declared.remove(key);
    }
    verify_component_declaration(&declared, None, false).unwrap();
    let own = RoventResearcher::Own {
        implementation_sha256: "1".repeat(64),
        search_configuration_sha256: "2".repeat(64),
    };
    assert!(verify_component_declaration(&declared, Some(&own), false).is_err());
    // PC-only NNUEC readout registers its implementation, without R execution.
    assert!(verify_component_declaration(&declared, None, true).is_err());
    let actual = compiled_process_source_components().unwrap();
    declared.insert("cpu_search".into(), actual["cpu_search"].clone());
    verify_component_declaration(&declared, None, true).unwrap();
}

#[test]
fn commit_registration_is_not_replaced_by_a_fabricated_valid_sha() {
    let registered = env!("RZ_PALS_COMPILED_SOURCE_COMMIT");
    if registered.len() == 40 {
        verify_commit_registration(registered).unwrap();
    } else {
        assert_eq!(
            verify_commit_registration(registered).unwrap_err().kind,
            ProcessBoundaryFailure::SourceCommitRegistration
        );
    }
    let other = if registered == "0".repeat(40) {
        "1".repeat(40)
    } else {
        "0".repeat(40)
    };
    assert_eq!(
        verify_commit_registration(&other).unwrap_err().kind,
        ProcessBoundaryFailure::SourceCommitRegistration
    );
    assert!(verify_commit_registration("unregistered").is_err());
}

#[test]
fn original_control_is_not_renewed_and_cancel_precedes_expiry() {
    let active = control();
    check_control(&active).unwrap();
    let mut expired = active.clone();
    expired.until = Instant::now() - Duration::from_millis(1);
    assert_eq!(
        check_control(&expired).unwrap_err().kind,
        ProcessBoundaryFailure::Expired
    );
    active.cancellation.store(true, Ordering::Release);
    assert!(Arc::ptr_eq(&active.cancellation, &expired.cancellation));
    assert_eq!(
        check_control(&expired).unwrap_err().kind,
        ProcessBoundaryFailure::Cancelled
    );
}

#[test]
fn owned_transfer_requires_same_startup_clock_and_cancellation_owner() {
    let original = control();
    verify_original_startup_control(&original, &original.clone()).unwrap();
    let mut changed = original.clone();
    changed.cancellation = Arc::new(AtomicBool::new(false));
    assert!(verify_original_startup_control(&original, &changed).is_err());
    changed = original.clone();
    changed.until += Duration::from_nanos(1);
    assert!(verify_original_startup_control(&original, &changed).is_err());
    changed = original.clone();
    changed.cleanup_until += Duration::from_nanos(1);
    assert!(verify_original_startup_control(&original, &changed).is_err());
    original.cancellation.store(true, Ordering::Release);
    assert_eq!(
        verify_original_startup_control(&original, &original.clone())
            .unwrap_err()
            .kind,
        ProcessBoundaryFailure::Cancelled
    );
}

#[cfg(not(target_os = "linux"))]
#[test]
fn unsupported_platform_never_promotes_current_exe_file_to_mapped_proof() {
    assert_eq!(
        require_mapped_executable_profile().unwrap_err().kind,
        ProcessBoundaryFailure::UnsupportedMappedExecutable
    );
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use cap_fs_ext::MetadataExt as _;
    use std::{
        fs,
        io::{Seek, SeekFrom},
        os::fd::AsRawFd,
        path::PathBuf,
        sync::atomic::AtomicU64,
    };
    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            let time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "rz-rovent-process-fixture-{}-{time}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&root).unwrap();
            Self { root }
        }
        fn capability(&self) -> Dir {
            Dir::open_ambient_dir(&self.root, cap_std::ambient_authority()).unwrap()
        }
        fn cwd_file(&self) -> File {
            File::open(&self.root).unwrap()
        }
        fn binary(&self, bytes: &[u8]) -> (ArtifactRef, File) {
            let path = self.root.join("binary");
            fs::write(&path, bytes).unwrap();
            (
                ArtifactRef {
                    path: "inputs/binary".into(),
                    bytes: bytes.len() as u64,
                    sha256: hex(Sha256::digest(bytes).into()),
                    source: "https://example.org/unexecuted-process-boundary-fixture".into(),
                    license: "fixture-only".into(),
                },
                File::open(path).unwrap(),
            )
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            if self
                .root
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("rz-rovent-process-fixture-"))
                && fs::symlink_metadata(&self.root)
                    .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
            {
                // Fixed owned fixture names only; no recursive/shared cleanup.
                for name in ["binary", "original", ".git"] {
                    let _ = fs::remove_file(self.root.join(name));
                }
                let _ = fs::remove_dir(&self.root);
            }
        }
    }

    // Declarations for rejected ownership/pure-identity fixtures only. This
    // lock does not match the actual compiled process and issues no proof.
    fn unexecuted_lock(binary: ArtifactRef) -> Arc<RoventInputLock> {
        let artifact = |name: &str| ArtifactRef {
            path: format!("inputs/{name}"),
            sha256: "b".repeat(64),
            bytes: 1,
            source: "https://example.org/unexecuted-process-boundary-fixture".into(),
            license: "fixture-only".into(),
        };
        let model = serde_json::json!({
            "semantics":"boundary-owner-fixture/1","encoding":"boundary-input/1",
            "policy_head":"legal-candidates/1","value_head":"side-to-move-wdl/1",
            "implementation_sha256":"b".repeat(64),"weights":artifact("weights.bin"),
            "provenance":artifact("model.json"),"diagnostic":false
        });
        let commit = if env!("RZ_PALS_COMPILED_SOURCE_COMMIT") == "a".repeat(40) {
            "b".repeat(40)
        } else {
            "a".repeat(40)
        };
        let value = serde_json::json!({
            "domain":"rz-rovent-execution/1",
            "contract_revision":rz_contracts::rovent::ROVENT_SEARCH_REVISION,
            "run_id":"boundary-original-owner-fixture","purpose":"validation",
            "engine":{
                "source_commit":commit,"source_components":{"control":"c".repeat(64)},
                "binary":binary,"mode":"cpu_pc",
                "pc":{"proposer":model.clone(),"critic":model,"validator":null,
                    "backend":"rust_cpu","device":{"device":"cpu"},
                    "backend_configuration_sha256":"d".repeat(64),"runtime_artifacts":[],
                    "fp32":true,"tf32":false,"batch_width":1,"ort_intra_threads":null,
                    "ort_inter_threads":null,"ort_spinning":null},
                "researcher":null,"teachers":[],"nnuec":null,
                "search":{"node_unit":"rovent-position-visits/1","requested_nodes":null,
                    "depth_plies":null,"mate_moves":null,"registered_max_chess_plies":64,
                    "searchmoves":null,"search_configuration_sha256":"e".repeat(64),
                    "root_state_and_history_sha256":"f".repeat(64),
                    "rules_profile_sha256":"1".repeat(64)}},
            "resources":{"cpu_quota_millicpus":2000,"pools":[
                {"actors":["manager","pc_caller"],"max_threads":1,"cpu_charge_millicpus":1000}],
                "memory_high_bytes":3u64<<30,"memory_max_bytes":4u64<<30,"gpu_devices":0,
                "physical_nn_concurrency":1,"topology_policy":"shared-control/1"},
            "coordinator":{"policy_sha256":"2".repeat(64),"questions_max":64,
                "ancestor_depth_max":64,"waiting_jobs_max":64,"role_tickets_max":16,
                "cpu_credit_limit":0,"original_root_budget_sha256":"3".repeat(64),
                "original_deadline_seal_sha256":"4".repeat(64)},
            "wall_time_max_ms":120000,"cleanup_max_ms":30000,"output_bytes_max":1u64<<20,
            "agent_renewable":true,"provenance":[artifact("registration.json")]
        });
        Arc::new(
            rz_experiments::RoventExecutionManifest::from_json(&value.to_string())
                .unwrap()
                .lock()
                .unwrap(),
        )
    }

    fn unexecuted_inputs(
        fixture: &Fixture,
        control: RoventStartupControl,
    ) -> RoventProcessBoundaryInputs {
        let (artifact, original_binary) = fixture.binary(b"unexecuted owned boundary fixture");
        RoventProcessBoundaryInputs {
            lock: unexecuted_lock(artifact),
            original_binary,
            original_root: fixture.capability(),
            private_cwd: fixture.root.clone(),
            control,
        }
    }

    // Declaration-only /2 fixture: no observation or live proof is fabricated.
    fn unexecuted_inputs_v2(
        fixture: &Fixture,
        control: RoventStartupControl,
    ) -> RoventProcessBoundaryInputsV2 {
        use rz_experiments::{
            ROVENT_EXECUTION_V2_DOMAIN, RoventExecutionManifestV2, RoventOwnRNotApplicableV2,
            RoventRootTieDeclarationV2,
        };
        let v1 = unexecuted_inputs(fixture, control);
        let base = &v1.lock.manifest;
        let manifest = RoventExecutionManifestV2 {
            domain: ROVENT_EXECUTION_V2_DOMAIN.into(),
            contract_revision: base.contract_revision.clone(),
            run_id: base.run_id.clone(),
            purpose: base.purpose,
            engine: base.engine.clone(),
            resources: base.resources.clone(),
            coordinator: base.coordinator.clone(),
            wall_time_max_ms: base.wall_time_max_ms,
            cleanup_max_ms: base.cleanup_max_ms,
            output_bytes_max: base.output_bytes_max,
            agent_renewable: base.agent_renewable,
            provenance: base.provenance.clone(),
            startup_document: base.engine.binary.clone(),
            startup_configuration_sha256: base.coordinator.policy_sha256.clone(),
            startup_search_configuration_sha256: base
                .engine
                .search
                .search_configuration_sha256
                .clone(),
            own_r_root_tie: RoventRootTieDeclarationV2::NotApplicable {
                reason: RoventOwnRNotApplicableV2::PcOnly,
            },
        };
        RoventProcessBoundaryInputsV2 {
            lock: Arc::new(manifest.lock().unwrap()),
            original_binary: v1.original_binary,
            original_root: v1.original_root,
            private_cwd: v1.private_cwd,
            control: v1.control,
        }
    }

    #[test]
    fn owned_v2_rejection_returns_original_lock_descriptors_and_controls() {
        let fixture = Fixture::new();
        for canceled in [false, true] {
            let mut original_control = control();
            if canceled {
                original_control.cancellation.store(true, Ordering::Release);
            } else {
                original_control.until = Instant::now() - Duration::from_millis(1);
            }
            let mut inputs = unexecuted_inputs_v2(&fixture, original_control.clone());
            inputs.original_binary.seek(SeekFrom::Start(7)).unwrap();
            let lock = Arc::clone(&inputs.lock);
            let binary_fd = inputs.original_binary.as_raw_fd();
            let root_fd = inputs.original_root.as_raw_fd();
            let failure = verify_actual_owned_process_boundary_v2(inputs).unwrap_err();
            assert_eq!(
                failure.error.kind,
                if canceled {
                    ProcessBoundaryFailure::Cancelled
                } else {
                    ProcessBoundaryFailure::Expired
                }
            );
            let mut returned = failure.inputs;
            assert!(Arc::ptr_eq(&returned.lock, &lock));
            assert_eq!(returned.original_binary.as_raw_fd(), binary_fd);
            assert_eq!(returned.original_root.as_raw_fd(), root_fd);
            assert_eq!(returned.original_binary.stream_position().unwrap(), 7);
            assert_eq!(returned.private_cwd, fixture.root);
            assert_eq!(returned.control.until, original_control.until);
            assert_eq!(
                returned.control.cleanup_until,
                original_control.cleanup_until
            );
            assert!(Arc::ptr_eq(
                &returned.control.cancellation,
                &original_control.cancellation
            ));
        }
    }

    #[test]
    fn owned_v2_partial_source_failure_keeps_original_inputs() {
        let fixture = Fixture::new();
        let inputs = unexecuted_inputs_v2(&fixture, control());
        let lock = Arc::clone(&inputs.lock);
        let binary_fd = inputs.original_binary.as_raw_fd();
        let root_fd = inputs.original_root.as_raw_fd();
        let failure = verify_actual_owned_process_boundary_v2(inputs).unwrap_err();
        assert_eq!(
            failure.error.kind,
            ProcessBoundaryFailure::SourceCommitRegistration
        );
        assert!(Arc::ptr_eq(&failure.inputs.lock, &lock));
        assert_eq!(failure.inputs.original_binary.as_raw_fd(), binary_fd);
        assert_eq!(failure.inputs.original_root.as_raw_fd(), root_fd);
    }

    #[test]
    fn owned_v2_invalid_digest_returns_original_inputs_before_process_observation() {
        let fixture = Fixture::new();
        let mut inputs = unexecuted_inputs_v2(&fixture, control());
        Arc::make_mut(&mut inputs.lock).canonical_sha256 = "0".repeat(64);
        let lock = Arc::clone(&inputs.lock);
        let binary_fd = inputs.original_binary.as_raw_fd();
        let root_fd = inputs.original_root.as_raw_fd();
        let failure = verify_actual_owned_process_boundary_v2(inputs).unwrap_err();
        assert_eq!(failure.error.kind, ProcessBoundaryFailure::InvalidInputLock);
        assert!(Arc::ptr_eq(&failure.inputs.lock, &lock));
        assert_eq!(failure.inputs.original_binary.as_raw_fd(), binary_fd);
        assert_eq!(failure.inputs.original_root.as_raw_fd(), root_fd);
    }

    #[test]
    fn owned_public_rejection_returns_original_lock_descriptors_cursor_and_controls() {
        let fixture = Fixture::new();
        for canceled in [false, true] {
            let mut original_control = control();
            if canceled {
                original_control.cancellation.store(true, Ordering::Release);
            } else {
                original_control.until = Instant::now() - Duration::from_millis(1);
            }
            let mut inputs = unexecuted_inputs(&fixture, original_control.clone());
            inputs.original_binary.seek(SeekFrom::Start(7)).unwrap();
            let lock = Arc::clone(&inputs.lock);
            let binary_fd = inputs.original_binary.as_raw_fd();
            let root_fd = inputs.original_root.as_raw_fd();
            let failure = verify_actual_owned_process_boundary(inputs).unwrap_err();
            assert_eq!(
                failure.error.kind,
                if canceled {
                    ProcessBoundaryFailure::Cancelled
                } else {
                    ProcessBoundaryFailure::Expired
                }
            );
            let mut returned = failure.inputs;
            assert!(Arc::ptr_eq(&returned.lock, &lock));
            assert_eq!(returned.original_binary.as_raw_fd(), binary_fd);
            assert_eq!(returned.original_root.as_raw_fd(), root_fd);
            assert_eq!(returned.original_binary.stream_position().unwrap(), 7);
            assert_eq!(returned.private_cwd, fixture.root);
            assert_eq!(returned.control.until, original_control.until);
            assert_eq!(
                returned.control.cleanup_until,
                original_control.cleanup_until
            );
            assert!(Arc::ptr_eq(
                &returned.control.cancellation,
                &original_control.cancellation
            ));
            assert!(returned.original_binary.metadata().unwrap().is_file());
            assert!(returned.original_root.dir_metadata().unwrap().is_dir());
        }
    }

    #[test]
    fn owned_actual_verifier_partial_source_failure_keeps_complete_original_inputs() {
        let fixture = Fixture::new();
        let inputs = unexecuted_inputs(&fixture, control());
        let lock = Arc::clone(&inputs.lock);
        let cancellation = Arc::clone(&inputs.control.cancellation);
        let until = inputs.control.until;
        let original_binary = inputs.original_binary.as_raw_fd();
        let original_root = inputs.original_root.as_raw_fd();
        let failure = verify_actual_owned_process_boundary(inputs).unwrap_err();
        assert_eq!(
            failure.error.kind,
            ProcessBoundaryFailure::SourceCommitRegistration
        );
        assert!(Arc::ptr_eq(&failure.inputs.lock, &lock));
        assert!(Arc::ptr_eq(
            &failure.inputs.control.cancellation,
            &cancellation
        ));
        assert_eq!(failure.inputs.control.until, until);
        assert_eq!(failure.inputs.original_binary.as_raw_fd(), original_binary);
        assert_eq!(failure.inputs.original_root.as_raw_fd(), original_root);
        assert!(failure.inputs.original_binary.metadata().unwrap().is_file());
        assert!(
            failure
                .inputs
                .original_root
                .dir_metadata()
                .unwrap()
                .is_dir()
        );
    }

    #[test]
    fn owner_identity_rejects_equal_lock_and_reopened_same_inode_descriptors() {
        let fixture = Fixture::new();
        let original = unexecuted_inputs(&fixture, control());
        let check = |lock: &RoventInputLock,
                     binary: &File,
                     root: &Dir,
                     registered: &Arc<RoventInputLock>| {
            verify_original_owner_objects(&original, lock, binary, root, registered)
        };
        check(
            &original.lock,
            &original.original_binary,
            &original.original_root,
            &original.lock,
        )
        .unwrap();
        let equal_lock = Arc::new(original.lock.as_ref().clone());
        assert_eq!(original.lock.canonical_sha256, equal_lock.canonical_sha256);
        assert_eq!(
            check(
                &equal_lock,
                &original.original_binary,
                &original.original_root,
                &equal_lock
            )
            .unwrap_err()
            .kind,
            ProcessBoundaryFailure::InvalidInputLock
        );
        assert!(
            check(
                &equal_lock,
                &original.original_binary,
                &original.original_root,
                &original.lock
            )
            .is_err()
        );
        let reopened_binary = File::open(fixture.root.join("binary")).unwrap();
        assert_eq!(
            descriptor_identity(&original.original_binary).unwrap(),
            descriptor_identity(&reopened_binary).unwrap()
        );
        assert_eq!(
            check(
                &original.lock,
                &reopened_binary,
                &original.original_root,
                &original.lock
            )
            .unwrap_err()
            .kind,
            ProcessBoundaryFailure::BinaryDescriptor
        );
        let reopened_root = fixture.capability();
        let original_metadata = original.original_root.dir_metadata().unwrap();
        let reopened_metadata = reopened_root.dir_metadata().unwrap();
        assert_eq!(
            (original_metadata.dev(), original_metadata.ino()),
            (reopened_metadata.dev(), reopened_metadata.ino())
        );
        assert_eq!(
            check(
                &original.lock,
                &original.original_binary,
                &reopened_root,
                &original.lock
            )
            .unwrap_err()
            .kind,
            ProcessBoundaryFailure::WorkingDirectory
        );
        // Pure identity comparison never consumes or substitutes any owner.
        check(
            &original.lock,
            &original.original_binary,
            &original.original_root,
            &original.lock,
        )
        .unwrap();
    }

    #[test]
    fn equal_numeric_observation_does_not_replace_the_same_actual_observation_owner() {
        // Private comparison values only: no Verified{Owned}... proof is
        // constructed, no actual process admission/success is manufactured.
        let values = || ProcessBoundaryObservation {
            executable: ExecutableIdentityV1 {
                sha256: "b".repeat(64),
                bytes: 7,
                identity_source: ExecutableIdentitySourceV1::LinuxProcSelfExe,
            },
            input_lock: Digest([1; 32]),
            source_aggregate: Digest([2; 32]),
        };
        let original = Arc::new(values());
        let shared = Arc::clone(&original);
        let independent = Arc::new(values());
        assert!(same_observed_values(&original, &independent));
        verify_original_observation(&original, &shared).unwrap();
        assert_eq!(
            verify_original_observation(&original, &independent)
                .unwrap_err()
                .kind,
            ProcessBoundaryFailure::MappedExecutable
        );
        assert_eq!(Arc::strong_count(&original), 2);
        assert_eq!(Arc::strong_count(&independent), 1);
    }

    #[test]
    fn owned_types_hold_send_sync_static_inputs_without_borrowed_proofs() {
        fn assert_static<T: Send + Sync + 'static>() {}
        assert_static::<RoventProcessBoundaryInputs>();
        assert_static::<VerifiedOwnedRoventProcessBoundary>();
    }

    #[test]
    fn original_binary_hash_preserves_cursor_and_rejects_changed_bytes_and_bounds() {
        let fixture = Fixture::new();
        let bytes = b"unexecuted original binary fixture";
        let (artifact, mut file) = fixture.binary(bytes);
        file.seek(SeekFrom::Start(7)).unwrap();
        verify_binary_descriptor(&artifact, &file, &control()).unwrap();
        assert_eq!(file.stream_position().unwrap(), 7);
        let mut too_big = artifact.clone();
        too_big.bytes = MAX_EXECUTABLE_BYTES + 1;
        assert_eq!(
            verify_binary_descriptor(&too_big, &file, &control())
                .unwrap_err()
                .kind,
            ProcessBoundaryFailure::BinaryDescriptor
        );
        fs::write(fixture.root.join("binary"), vec![b'x'; bytes.len()]).unwrap();
        assert_eq!(
            verify_binary_descriptor(&artifact, &file, &control())
                .unwrap_err()
                .kind,
            ProcessBoundaryFailure::BinaryDescriptor
        );
    }

    #[test]
    fn binary_cancellation_keeps_original_handle_and_cursor() {
        let fixture = Fixture::new();
        let (artifact, mut file) = fixture.binary(b"unexecuted canceled binary fixture");
        file.seek(SeekFrom::Start(5)).unwrap();
        let control = control();
        control.cancellation.store(true, Ordering::Release);
        assert_eq!(
            verify_binary_descriptor(&artifact, &file, &control)
                .unwrap_err()
                .kind,
            ProcessBoundaryFailure::Cancelled
        );
        assert_eq!(file.stream_position().unwrap(), 5);
    }

    #[test]
    fn mapped_binary_comparison_requires_actual_linux_origin_and_exact_bytes() {
        let fixture = Fixture::new();
        let (artifact, _file) = fixture.binary(b"unexecuted identity comparison fixture");
        let mut observed = ExecutableIdentityV1 {
            sha256: artifact.sha256.clone(),
            bytes: artifact.bytes,
            identity_source: ExecutableIdentitySourceV1::LinuxProcSelfExe,
        };
        verify_mapped_executable(&artifact, &observed).unwrap();
        observed.identity_source = ExecutableIdentitySourceV1::CurrentExecutableFile;
        assert_eq!(
            verify_mapped_executable(&artifact, &observed)
                .unwrap_err()
                .kind,
            ProcessBoundaryFailure::MappedExecutable
        );
        observed.identity_source = ExecutableIdentitySourceV1::LinuxProcSelfExe;
        observed.bytes += 1;
        assert!(verify_mapped_executable(&artifact, &observed).is_err());
        observed.bytes = artifact.bytes;
        observed.sha256 = "0".repeat(64);
        assert!(verify_mapped_executable(&artifact, &observed).is_err());
    }

    #[test]
    fn private_directory_requires_exact_path_and_original_cwd_and_root_capabilities() {
        let first = Fixture::new();
        let second = Fixture::new();
        let actual = first.cwd_file();
        let root = first.capability();
        verify_private_directory_at(&first.root, &first.root, &actual, &root).unwrap();
        assert!(verify_private_directory_at(&second.root, &first.root, &actual, &root).is_err());
        assert!(
            verify_private_directory_at(&first.root, &first.root, &second.cwd_file(), &root)
                .is_err()
        );
        assert!(
            verify_private_directory_at(&first.root, &first.root, &actual, &second.capability())
                .is_err()
        );
        assert!(
            verify_private_directory_at(
                Path::new("relative"),
                Path::new("relative"),
                &actual,
                &root
            )
            .is_err()
        );
    }

    #[test]
    fn source_checkout_marker_is_rejected_without_reading_repository_contents() {
        let fixture = Fixture::new();
        fs::write(
            fixture.root.join(".git"),
            b"marker-only; contents are never read",
        )
        .unwrap();
        assert_eq!(
            verify_private_directory_at(
                &fixture.root,
                &fixture.root,
                &fixture.cwd_file(),
                &fixture.capability()
            )
            .unwrap_err()
            .kind,
            ProcessBoundaryFailure::SourceCheckout
        );
    }

    #[test]
    fn replaced_named_root_does_not_rebind_the_original_cwd_descriptor() {
        let fixture = Fixture::new();
        let root = fixture.capability();
        let actual = fixture.cwd_file();
        let original = fixture.root.with_extension("original");
        fs::rename(&fixture.root, &original).unwrap();
        fs::create_dir(&fixture.root).unwrap();
        let result = verify_private_directory_at(&fixture.root, &fixture.root, &actual, &root);
        // Restore the owned names before assertion, including failure cases.
        fs::remove_dir(&fixture.root).unwrap();
        fs::rename(&original, &fixture.root).unwrap();
        assert_eq!(
            result.unwrap_err().kind,
            ProcessBoundaryFailure::WorkingDirectory
        );
    }
}
