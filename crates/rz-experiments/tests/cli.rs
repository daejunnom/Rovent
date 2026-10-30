use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let parent = std::env::temp_dir().join("rovezero-e01-cli-tests");
        fs::create_dir_all(&parent).expect("create temporary test root");
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after Unix epoch")
            .as_nanos();
        let path = parent.join(format!("{}-{timestamp}-{sequence}", std::process::id()));
        fs::create_dir(&path).expect("create temporary test directory");
        Self(path)
    }

    fn file(&self, name: &str, contents: impl AsRef<[u8]>) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, contents).expect("write test input");
        path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run<I, S>(args: I) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut command = Command::new(env!("CARGO_BIN_EXE_rz-experiments"));
    command.args(args);
    run_command(command)
}

fn run_command(mut command: Command) -> Output {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run CLI");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if child.try_wait().expect("inspect CLI status").is_some() {
            return child.wait_with_output().expect("collect CLI output");
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output().expect("reap timed out CLI");
            panic!("CLI exceeded five-second test deadline: {output:?}");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../experiments/baselines/fixtures/e01-input.json")
}

fn assert_failure(output: &Output) -> String {
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty(), "failures must not print success");
    let diagnostic = String::from_utf8(output.stderr.clone()).expect("UTF-8 diagnostic");
    assert!(!diagnostic.trim().is_empty());
    diagnostic
}

#[test]
fn incomplete_or_unknown_commands_fail() {
    for args in [
        vec![],
        vec!["validate"],
        vec!["lock", "input.json"],
        vec!["verify"],
        vec!["run", "input.json"],
        vec!["validate", "input.json", "extra"],
    ] {
        assert_failure(&run(args));
    }
}

#[test]
fn malformed_json_does_not_create_a_lock() {
    let directory = TestDirectory::new();
    let input = directory.file("bad.json", b"{broken");
    let output = directory.0.join("locked.json");
    assert_failure(&run([OsStr::new("validate"), input.as_os_str()]));
    assert_failure(&run([
        OsStr::new("lock"),
        input.as_os_str(),
        output.as_os_str(),
    ]));
    assert!(!output.exists());
    assert_failure(&run([OsStr::new("verify"), input.as_os_str()]));
}

#[test]
fn json_input_is_bounded_and_strict_utf8() {
    let directory = TestDirectory::new();
    let oversized = directory.file("large.json", vec![b' '; 4 * 1024 * 1024 + 1]);
    let invalid_utf8 = directory.file("non-utf8.json", [b'{', 0xff, b'}']);
    for command in ["validate", "verify"] {
        let error = assert_failure(&run([OsStr::new(command), oversized.as_os_str()]));
        assert!(error.contains("4 MiB"));
        let error = assert_failure(&run([OsStr::new(command), invalid_utf8.as_os_str()]));
        assert!(error.contains("UTF-8"));
    }
    let output = directory.0.join("locked.json");
    let error = assert_failure(&run([
        OsStr::new("lock"),
        oversized.as_os_str(),
        output.as_os_str(),
    ]));
    assert!(error.contains("4 MiB"));
    assert!(!output.exists());
}

#[test]
fn existing_output_is_preserved() {
    let directory = TestDirectory::new();
    let output = directory.file("locked.json", b"existing evidence\n");
    let error = assert_failure(&run([
        OsStr::new("lock"),
        fixture().as_os_str(),
        output.as_os_str(),
    ]));
    assert!(error.contains("cannot create output"));
    assert_eq!(
        fs::read(&output).expect("read existing output"),
        b"existing evidence\n"
    );
}

#[test]
fn input_lock_and_integrity_verification_retain_readiness_false() {
    let directory = TestDirectory::new();
    let locked = directory.0.join("locked.json");
    let input = fixture();
    let output = run([OsStr::new("validate"), input.as_os_str()]);
    assert!(output.status.success(), "{:?}", output);
    assert!(String::from_utf8_lossy(&output.stdout).contains("execution_ready=false"));
    let output = run([OsStr::new("lock"), input.as_os_str(), locked.as_os_str()]);
    assert!(output.status.success(), "{:?}", output);
    assert_eq!(output.stdout, b"input locked; execution_ready=false\n");
    let output = run([OsStr::new("verify"), locked.as_os_str()]);
    assert!(output.status.success(), "{:?}", output);
    assert_eq!(
        output.stdout,
        b"locked input integrity verified; execution_ready=false\n"
    );
    let artifacts = input.parent().expect("fixture directory");
    let output = run([
        OsStr::new("verify"),
        locked.as_os_str(),
        OsStr::new("--artifact-root"),
        artifacts.as_os_str(),
        OsStr::new("--max-artifact-bytes"),
        OsStr::new("1048576"),
    ]);
    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        output.stdout,
        b"locked input integrity and artifacts verified; execution_ready=false\n"
    );
}

fn rovent_fixture() -> String {
    let hash = "a".repeat(64);
    let artifact = serde_json::json!({"path":"fixture/input.bin", "sha256":hash, "bytes":32,
        "source":"https://example.org/rovent-cli-fixture", "license":"MIT"});
    let model = serde_json::json!({"semantics":"fixture-model/1", "encoding":"fixture-input/1",
        "policy_head":"legal-policy/1", "value_head":"wdl/1", "implementation_sha256":hash,
        "weights":artifact, "provenance":artifact, "diagnostic":false});
    serde_json::json!({
        "domain":"rz-rovent-execution/1", "contract_revision":"rovent-search/0.1",
        "run_id":"rovent-cli-fixture", "purpose":"validation",
        "engine": {"source_commit":"b".repeat(40), "source_components":{"control":hash}, "binary":artifact,
            "mode":"cpu_pc", "pc":{"proposer":model, "critic":model, "validator":null,
                "backend":"rust_cpu", "device":{"device":"cpu"}, "backend_configuration_sha256":hash,
                "runtime_artifacts":[], "fp32":true, "tf32":false, "batch_width":1,
                "ort_intra_threads":null, "ort_inter_threads":null, "ort_spinning":null},
            "researcher":null, "teachers":[], "nnuec":null,
            "search":{"node_unit":"rovent-position-visits/1", "requested_nodes":null,
                "depth_plies":null, "mate_moves":null, "registered_max_chess_plies":64,
                "searchmoves":null, "search_configuration_sha256":hash,
                "root_state_and_history_sha256":hash, "rules_profile_sha256":hash}},
        "resources":{"cpu_quota_millicpus":2000, "pools":[{"actors":["manager","pc_caller"],
            "max_threads":1, "cpu_charge_millicpus":1000}], "memory_high_bytes":6_u64<<30,
            "memory_max_bytes":12_u64<<30, "gpu_devices":0, "physical_nn_concurrency":1,
            "topology_policy":"shared-control/1"},
        "coordinator":{"policy_sha256":hash, "questions_max":1024, "ancestor_depth_max":64,
            "waiting_jobs_max":64, "role_tickets_max":16, "cpu_credit_limit":0,
            "original_root_budget_sha256":hash, "original_deadline_seal_sha256":hash},
        "wall_time_max_ms":60000, "cleanup_max_ms":30000, "output_bytes_max":1048576,
        "agent_renewable":true, "provenance":[artifact]
    }).to_string()
}

#[test]
fn rovent_cli_locks_new_domain_preserves_legacy_separation_and_never_authorizes_execution() {
    let directory = TestDirectory::new();
    let input = directory.file("rovent.json", rovent_fixture());
    let locked = directory.0.join("rovent.lock.json");
    let output = run([OsStr::new("rovent-validate"), input.as_os_str()]);
    assert!(output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stdout).contains("execution_ready=false"));
    assert_failure(&run([OsStr::new("validate"), input.as_os_str()]));
    assert_failure(&run([OsStr::new("rovent-validate"), fixture().as_os_str()]));
    let output = run([
        OsStr::new("rovent-lock"),
        input.as_os_str(),
        locked.as_os_str(),
    ]);
    assert!(output.status.success(), "{output:?}");
    let original = fs::read(&locked).unwrap();
    let lock = rz_experiments::RoventInputLock::from_json(std::str::from_utf8(&original).unwrap())
        .unwrap();
    assert_failure(&run([
        OsStr::new("rovent-lock"),
        input.as_os_str(),
        locked.as_os_str(),
    ]));
    assert_eq!(fs::read(&locked).unwrap(), original);
    let hash = "a".repeat(64);
    let artifact = serde_json::json!({"path":"fixture/evidence.json", "sha256":hash, "bytes":32,
        "source":"https://example.org/rovent-cli-fixture", "license":"MIT"});
    let unknown = serde_json::json!({"status":"unknown"});
    let receipt = serde_json::json!({
        "domain":"rz-rovent-execution-receipt/1", "run_id":lock.manifest.run_id,
        "input_lock_sha256":lock.canonical_sha256,
        "functional_accepted":false, "controlled_comparison_eligible":false,
        "source_content_verified":unknown, "source_and_binary_evidence":unknown,
        "model_content_verified":unknown, "model_configuration_sha256":unknown, "model_content_evidence":unknown,
        "pc_device":unknown, "backend_configuration_sha256":unknown, "calls":unknown,
        "nodes":unknown, "lifecycle":unknown, "goal":unknown, "root_moves_rules_seal":unknown,
        "external_profiles_verified":unknown, "physical_state":unknown, "worker_joined_and_processes_closed":unknown,
        "resources":{"limits_applied":unknown, "process_tree_quota_millicpus":unknown,
            "memory_high_bytes":unknown, "memory_max_bytes":unknown, "inventory_complete":unknown,
            "kernel_threads":unknown, "verified_plan_readback_sha256":unknown,
            "host_peak_bytes":unknown, "vram_peak_bytes":unknown, "evidence":artifact},
        "failures":[], "full_gc_requested":false, "memory_contention":false,
        "wall_time_ms":0, "cleanup_ms":0, "output_bytes":0, "execution_evidence":artifact
    });
    let receipt_path = directory.file("unobserved-receipt.json", receipt.to_string());
    let output = run([
        OsStr::new("rovent-verify-receipt"),
        locked.as_os_str(),
        receipt_path.as_os_str(),
    ]);
    assert!(output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stdout).contains("execution_attested=false"));
    let mut wrong = receipt;
    wrong["functional_accepted"] = serde_json::json!(true);
    let wrong_path = directory.file("false-acceptance.json", wrong.to_string());
    assert_failure(&run([
        OsStr::new("rovent-verify-receipt"),
        locked.as_os_str(),
        wrong_path.as_os_str(),
    ]));
}

#[test]
fn rovent_cli_rejects_incomplete_arguments_and_malformed_inputs_before_writing() {
    for args in [
        vec!["rovent-validate"],
        vec!["rovent-lock", "input"],
        vec!["rovent-verify-receipt", "lock"],
        vec!["rovent-validate", "input", "extra"],
    ] {
        assert_failure(&run(args));
    }
    let directory = TestDirectory::new();
    let malformed = directory.file("bad.json", b"{bad");
    let output = directory.0.join("not-created.json");
    assert_failure(&run([
        OsStr::new("rovent-lock"),
        malformed.as_os_str(),
        output.as_os_str(),
    ]));
    assert!(!output.exists());
    let oversized = directory.file("oversized.json", vec![b' '; 4 * 1024 * 1024 + 1]);
    assert!(
        assert_failure(&run([OsStr::new("rovent-validate"), oversized.as_os_str()]))
            .contains("4 MiB")
    );
}

#[test]
fn artifact_verification_requires_an_explicit_positive_budget() {
    for options in [
        vec!["--artifact-root", "artifacts"],
        vec!["--max-artifact-bytes", "100"],
        vec!["--artifact-root", "artifacts", "--max-artifact-bytes", "0"],
        vec!["--artifact-root", "artifacts", "--max-artifact-bytes", "-1"],
        vec![
            "--artifact-root",
            "artifacts",
            "--max-artifact-bytes",
            "18446744073709551616",
        ],
        vec!["--artifact-root", "artifacts", "--artifact-root", "other"],
        vec!["--max-artifact-bytes"],
    ] {
        let mut args = vec!["verify", "input-is-not-read.json"];
        args.extend(options);
        let error = assert_failure(&run(args));
        assert!(!error.contains("cannot open input"));
    }
}

#[test]
fn requested_artifact_verification_rejects_missing_files() {
    let directory = TestDirectory::new();
    let locked = directory.0.join("locked.json");
    let artifacts = directory.0.join("artifacts");
    fs::create_dir(&artifacts).expect("create empty artifact root");
    let output = run([
        OsStr::new("lock"),
        fixture().as_os_str(),
        locked.as_os_str(),
    ]);
    assert!(output.status.success(), "{output:?}");
    // Supplying both options must actually inspect files, rather than only verify
    // the envelope as the default verify command does.
    let error = assert_failure(&run([
        OsStr::new("verify"),
        locked.as_os_str(),
        OsStr::new("--max-artifact-bytes"),
        OsStr::new("1048576"),
        OsStr::new("--artifact-root"),
        artifacts.as_os_str(),
    ]));
    assert!(error.contains("artifact verification"));
}

#[cfg(unix)]
#[test]
fn fifo_input_is_rejected_without_waiting_for_a_writer() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let directory = TestDirectory::new();
    let fifo = directory.0.join("input.fifo");
    let path = CString::new(fifo.as_os_str().as_bytes()).expect("FIFO path without NUL");
    // The C string remains live throughout this call; mkfifo does not retain it.
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    for command in ["validate", "verify"] {
        let error = assert_failure(&run([OsStr::new(command), fifo.as_os_str()]));
        assert!(error.contains("regular file"));
    }
    let output = directory.0.join("locked.json");
    assert_failure(&run([
        OsStr::new("lock"),
        fifo.as_os_str(),
        output.as_os_str(),
    ]));
    assert!(!output.exists());
}

#[cfg(unix)]
#[test]
fn partial_write_failure_preserves_output_and_other_files() {
    use std::os::unix::process::CommandExt;

    let directory = TestDirectory::new();
    let output = directory.0.join("locked.json");
    let sibling = directory.file("preserved.txt", b"existing evidence\n");
    let mut command = Command::new(env!("CARGO_BIN_EXE_rz-experiments"));
    command.args([
        OsStr::new("lock"),
        fixture().as_os_str(),
        output.as_os_str(),
    ]);
    // Restrict only this child. Ignoring SIGXFSZ makes write_all return EFBIG so
    // the CLI's write-error handling runs instead of terminating by signal.
    // These calls use initialized values and only async-signal-safe libc APIs.
    unsafe {
        command.pre_exec(|| {
            if libc::signal(libc::SIGXFSZ, libc::SIG_IGN) == libc::SIG_ERR {
                return Err(std::io::Error::last_os_error());
            }
            let limit = libc::rlimit {
                rlim_cur: 1024,
                rlim_max: 1024,
            };
            if libc::setrlimit(libc::RLIMIT_FSIZE, &limit) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let error = assert_failure(&run_command(command));
    assert!(error.contains("partial output preserved"));
    let partial = fs::read(&output).expect("partial output retained for inspection");
    assert!(!partial.is_empty());
    assert!(partial.len() <= 1024);
    assert_eq!(
        fs::read(sibling).expect("read preserved sibling"),
        b"existing evidence\n"
    );
}
