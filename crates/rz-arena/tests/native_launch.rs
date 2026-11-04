use rz_arena::encode_fastchess_native_args;
use std::ffi::OsString;

#[test]
fn native_inner_arguments_are_one_complete_quoted_fastchess_value() {
    let tokens = vec![
        OsString::from("--onnx-cpu"),
        OsString::from("--attestation"),
        OsString::from("--onnx-model=/owned path/model"),
        OsString::from("--ort-sha256=abcd"),
        OsString::from("--runtime-cache-root=/owned path/cache"),
    ];
    let encoded = encode_fastchess_native_args(&tokens).unwrap();
    assert_eq!(
        encoded.to_str().unwrap(),
        "args=\"--onnx-cpu\" \"--attestation\" \"--onnx-model=/owned path/model\" \"--ort-sha256=abcd\" \"--runtime-cache-root=/owned path/cache\""
    );
    for bad in ["", "x\ny", "x\ty", "x'y", "x\"y", "x\\y"] {
        assert!(encode_fastchess_native_args(&[bad.into()]).is_err());
    }
    assert!(encode_fastchess_native_args(&[]).is_err());
    assert!(encode_fastchess_native_args(&vec!["x".into(); 33]).is_err());
    assert!(encode_fastchess_native_args(&vec!["x".repeat(4096).into(); 5]).is_err());
}

#[cfg(target_os = "linux")]
#[test]
fn native_snapshot_uses_new_inode_preserves_prior_attempt_and_fails_closed_on_pin_error() {
    use rz_arena::{
        FASTCHESS_SOURCE_COMMIT, FASTCHESS_SOURCE_URL, FASTCHESS_VERSION, native_admission_closed,
        opening_pgn_for_spec, prepare_native_launch,
    };
    use rz_experiments::*;
    use sha2::{Digest, Sha256};
    use std::{fs, os::unix::fs::PermissionsExt};
    // Synthetic bytes exercise pre-spawn ownership only. They are not neural
    // assets, native executable validation or actual CPU NN execution evidence.
    let base = std::env::temp_dir().join(format!("rovezero-native-owner-{}", std::process::id()));
    fs::create_dir(&base).expect("exclusive test output directory");
    let source = base.join("source");
    let output = base.join("output");
    fs::create_dir(&source).unwrap();
    fs::create_dir(&output).unwrap();
    // An unrelated preexisting link is outside this new attempt. Previously a
    // recursive whole-root admission scan rejected it before the owner could
    // be created; this must not block independent work or delete the link.
    std::os::unix::fs::symlink(&source, output.join("unrelated-prior-link")).unwrap();
    // A sparse file models an unrelated retained history above the old 32GiB
    // root gate without allocating or writing that much data.
    let prior_history = output.join("unrelated-prior-history");
    let prior_logical_bytes = 1_u64 << 36;
    fs::File::create(&prior_history)
        .unwrap()
        .set_len(prior_logical_bytes)
        .unwrap();
    let reference = |name: &str, bytes: &[u8], provenance: &str| {
        fs::write(source.join(name), bytes).unwrap();
        ArtifactRef {
            path: name.into(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
            bytes: bytes.len() as u64,
            source: provenance.into(),
            license: "Synthetic ownership fixture only".into(),
        }
    };
    let opening = OpeningSpec {
        id: "owner-opening".into(),
        initial: InitialPosition::Startpos,
        fen: None,
        moves: vec!["e2e4".into(), "e7e5".into()],
        history: HistoryCompleteness::Complete,
        history_origin: "startpos-complete-trace".into(),
    };
    let pgn = opening_pgn_for_spec(&opening, 8).unwrap();
    let opening_artifact = reference(
        "opening.pgn",
        pgn.as_bytes(),
        "https://example.org/rovezero-owner-test",
    );
    let artifacts = NativeArtifactRole::ALL
        .into_iter()
        .enumerate()
        .map(|(index, role)| NativeArtifactBinding {
            role,
            artifact: reference(
                &format!("input-{index}"),
                &[index as u8, 1, 2, 3],
                "https://example.org/rovezero-owner-test",
            ),
        })
        .collect();
    let baseline = CpuNativeLaunchSpecV1 {
        role: NativeEngineRole::Baseline,
        engine_id: "native-baseline".into(),
        source_commit: "a".repeat(40),
        target: "x86_64-unknown-linux-gnu".into(),
        artifacts,
        profile: NativeCpuProfileV1::fixed("b".repeat(64), "c".repeat(64)),
    };
    let mut candidate = baseline.clone();
    candidate.role = NativeEngineRole::Candidate;
    candidate.engine_id = "native-candidate".into();
    let runner = reference("fastchess", b"synthetic runner bytes", FASTCHESS_SOURCE_URL);
    let spec = IntegrationPairSpecV1 {
        schema_version: 1,
        purpose: NativeIntegrationPurpose::CpuNnIntegration,
        strength_eligible: false,
        contract_revision: "0.1".into(),
        run_id: "owner-run".into(),
        pair_id: "owner-pair".into(),
        engines: [baseline, candidate],
        white_order: [NativeEngineRole::Baseline, NativeEngineRole::Candidate],
        opening,
        opening_artifact,
        runner: ToolIdentity {
            version: FASTCHESS_VERSION.into(),
            source_url: FASTCHESS_SOURCE_URL.into(),
            source_commit: FASTCHESS_SOURCE_COMMIT.into(),
            binary: runner,
            dirty: false,
            dirty_patch: None,
            build_mode: "synthetic-not-built".into(),
            compiler: "synthetic-not-built".into(),
            target: "x86_64-unknown-linux-gnu".into(),
            isa: "x86-64".into(),
        },
        clock: NativeMovetimeV1 { movetime_ms: 100 },
        max_plies: 8,
        timeouts: NativeTimeoutsV1 {
            startup_ms: 1000,
            handshake_ms: 1000,
            runtime_ms: 10000,
            drain_ms: 1000,
            shutdown_ms: 1000,
        },
        budget: NativeResourceBudgetV1 {
            max_input_bytes: 4096,
            max_output_bytes: 256 * 1024,
            max_runtime_bytes: 512 * 1024,
            max_artifact_bytes: 1024 * 1024,
            max_child_processes: 3,
            max_runtime_files: 32,
            max_runtime_depth: 4,
            address_space_per_process_bytes: 2 * 1024 * 1024 * 1024,
        },
    }
    .lock()
    .unwrap();
    fs::hard_link(source.join("input-1"), source.join("source-hardlink")).unwrap();
    let owner = prepare_native_launch(&spec, &source, &output, "attempt-one").unwrap();
    assert!(output.join("unrelated-prior-link").is_symlink());
    assert_eq!(
        fs::metadata(&prior_history).unwrap().len(),
        prior_logical_bytes
    );
    for (relative, expected_mode) in [
        ("attempt-one", 0o700),
        ("attempt-one/inputs", 0o500),
        ("attempt-one/baseline-runtime", 0o700),
        ("attempt-one/candidate-runtime", 0o700),
    ] {
        assert_eq!(
            fs::metadata(output.join(relative))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            expected_mode,
            "owned directory permission boundary differs: {relative}"
        );
    }
    assert_eq!(owner.snapshots().len(), 7);
    assert!(
        owner
            .snapshots()
            .iter()
            .all(|p| p.distinct_source_inode && p.closed_writer_read_only)
    );
    let snapshot = owner
        .snapshots()
        .iter()
        .find(|p| p.artifact.path == "input-1")
        .unwrap();
    let private = output
        .join("attempt-one")
        .join(&snapshot.snapshot_relative_path);
    fs::write(
        source.join("source-hardlink"),
        b"changed source through another inode name",
    )
    .unwrap();
    assert_eq!(fs::read(&private).unwrap(), vec![1, 1, 2, 3]);
    assert!(prepare_native_launch(&spec, &source, &output, "parallel-attempt").is_err());
    assert!(!output.join("parallel-attempt").exists());
    drop(owner);
    assert!(prepare_native_launch(&spec, &source, &output, "failed-pin-attempt").is_err());
    assert!(
        !native_admission_closed(),
        "pre-spawn failure releases the lease without an unresolved child"
    );
    assert!(output.join("attempt-one").is_dir());
    assert!(!output.join("failed-pin-attempt").exists());
    // Only this exclusive synthetic test tree is removed. No runtime/NN owner
    // was started, and its readonly input-directory mode is restored first.
    fs::set_permissions(
        output.join("attempt-one/inputs"),
        fs::Permissions::from_mode(0o700),
    )
    .unwrap();
    fs::remove_dir_all(&base).unwrap();
}
