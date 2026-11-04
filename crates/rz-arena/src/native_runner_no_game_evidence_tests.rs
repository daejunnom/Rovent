//! NoGame evidence custody on callback unwind and partial persistence.
//! The payload and unverified receipt below are test data, not a process,
//! device, game, or physical-fence observation.

use super::*;
use crate::pals_adaptive_launch::{
    NO_GAME_SMOKE_PLAN_DOMAIN, NoGameExecutionStartV1, NoGameSmokeBudgetV1, NoGameSmokeEndpointV1,
    NoGameSmokePlanV1, v6_test_pair_fixture,
};
use rz_experiments::{ArtifactRef, NativeEngineRole, PalsSystemStartupLockV6};
use sha2::{Digest as _, Sha256};
use std::{
    cell::Cell,
    fs,
    ops::{Deref, DerefMut},
    os::unix::fs::{MetadataExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct OwnedTestRoot(PathBuf, Cell<bool>, (u64, u64, u32));

impl OwnedTestRoot {
    fn new() -> Self {
        let base = std::env::temp_dir().canonicalize().unwrap();
        let root = base.join(format!(
            "rz-nogame-evidence-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
        fs::create_dir(root.join("sources")).unwrap();
        fs::create_dir(root.join("outputs")).unwrap();
        let metadata = fs::symlink_metadata(&root).unwrap();
        Self(
            root,
            Cell::new(true),
            (metadata.dev(), metadata.ino(), metadata.uid()),
        )
    }
}

impl Drop for OwnedTestRoot {
    fn drop(&mut self) {
        if !self.1.get() {
            return;
        }
        let Ok(metadata) = fs::symlink_metadata(&self.0) else {
            return;
        };
        if !self.0.is_absolute()
            || !metadata.is_dir()
            || metadata.file_type().is_symlink()
            || (metadata.dev(), metadata.ino(), metadata.uid()) != self.2
        {
            return;
        }
        // Refuse cleanup if any existing ancestor is a link, not an owned
        // directory, or crosses devices. Never follow a replacement ancestor.
        for rel in [
            "outputs",
            "outputs/attempt",
            "outputs/attempt/inputs",
            "outputs/attempt/inputs/test",
        ] {
            match fs::symlink_metadata(self.0.join(rel)) {
                Ok(m)
                    if m.is_dir()
                        && !m.file_type().is_symlink()
                        && m.uid() == self.2.2
                        && m.dev() == self.2.0 => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
                _ => return,
            }
        }
        // Only restore permissions for the known, exclusively created test
        // snapshot, after its original owner and lease have been dropped.
        for rel in ["outputs/attempt/inputs/test", "outputs/attempt/inputs"] {
            let path = self.0.join(rel);
            if fs::symlink_metadata(&path).is_ok_and(|m| m.is_dir() && !m.file_type().is_symlink())
            {
                let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o700));
            }
        }
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The only process output installed by this fixture is synthetic. If a test
/// panics, discard it only after checking that no real child was ever spawned;
/// otherwise preserve the original owner and its directory for inspection.
struct FixtureOwner<'a> {
    owner: PreparedIndependentNoGameOwner,
    root: &'a OwnedTestRoot,
}

impl Deref for FixtureOwner<'_> {
    type Target = PreparedIndependentNoGameOwner;
    fn deref(&self) -> &Self::Target {
        &self.owner
    }
}

impl DerefMut for FixtureOwner<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.owner
    }
}

impl Drop for FixtureOwner<'_> {
    fn drop(&mut self) {
        if self.owner.spawn_custody.child().is_none()
            && self.owner.spawn_custody.last_spawn_pid.is_none()
            && !self.owner.spawn_custody.pending.load(Ordering::Acquire)
        {
            drop(self.owner.process.take());
        } else {
            self.root.1.set(false);
        }
    }
}

fn materialize(root: &Path, artifact: &mut ArtifactRef, bytes: &[u8]) {
    let target = root.join(&artifact.path);
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    fs::write(target, bytes).unwrap();
    artifact.bytes = bytes.len() as u64;
    artifact.sha256 = format!("{:x}", Sha256::digest(bytes));
}

fn prepared_original(root: &OwnedTestRoot) -> PreparedIndependentNoGameOwner {
    let source = root.0.join("sources");
    let pair = v6_test_pair_fixture();
    let mut endpoint = pair.endpoints[0].clone();
    let mut config = endpoint.startup_lock.config.clone();
    config.window.wall_time_max_ms = 60_000;
    materialize(
        &source,
        &mut config.endpoint.binary,
        b"fixture-child-binary",
    );
    for asset in &mut config.assets {
        let payload = format!("fixture-asset:{}", asset.artifact.path);
        materialize(&source, &mut asset.artifact, payload.as_bytes());
        if asset.purpose == rz_experiments::PalsStartupAssetPurposeV5::Weights {
            config.endpoint.model.weights = asset.artifact.clone();
        }
    }
    endpoint.startup_lock = PalsSystemStartupLockV6::new(config).unwrap();
    let lock_bytes = endpoint.startup_lock.canonical_bytes().unwrap();
    materialize(&source, &mut endpoint.startup_lock_artifact, &lock_bytes);
    let mut proxy = pair.runner.binary.clone();
    proxy.path = "test/independent-proxy".into();
    materialize(&source, &mut proxy, b"fixture-proxy-binary");
    let plan = NoGameSmokePlanV1 {
        domain: NO_GAME_SMOKE_PLAN_DOMAIN.into(),
        declared_role: NativeEngineRole::Baseline,
        endpoint: NoGameSmokeEndpointV1::from(&endpoint),
        proxy_binary: proxy,
        proxy_snapshot_path: "inputs/test/independent-proxy".into(),
        budget: NoGameSmokeBudgetV1 {
            max_input_bytes: 1 << 20,
            max_runtime_bytes: 1 << 20,
            max_output_bytes: 1 << 20,
            max_artifact_bytes: 3 << 20,
            max_child_processes: 2,
            max_runtime_files: 16,
            max_runtime_depth: 4,
            address_space_per_process_bytes: 1 << 30,
        },
        paired_source_sha256: None,
    }
    .lock()
    .unwrap();
    let execution = Arc::new(NoGameExecutionStartV1::begin(&plan, [19; 16], 1).unwrap());
    prepare_independent_no_game_original(
        &plan,
        &execution,
        [19; 16],
        1,
        [1, 2],
        &source,
        &root.0.join("outputs"),
        "attempt",
    )
    .unwrap_or_else(|_| panic!("fixture original owner preparation failed"))
}

fn unresolved_test_payload() -> ProcessOutput {
    ProcessOutput {
        receipt: crate::ProcessReceipt {
            supervisor_version: 1,
            pid: 0,
            elapsed_ns: 0,
            cleanup_elapsed_ns: None,
            cleanup_timing_unavailable_reason: Some("synthetic-fixture-unknown".into()),
            stop: crate::ProcessStop::IoFailure,
            exit_code: None,
            exit_signal: None,
            group_cleanup: CleanupStatus::Unverified,
            stdout_bytes: 3,
            stderr_bytes: 3,
            observed_output_bytes: 6,
            descendant_cleanup_required: false,
            child_limit_enforcement: "synthetic-fixture".into(),
            watched_artifact_bytes: None,
            artifact_limit_enforcement: None,
            errors: vec!["synthetic-fixture".into()],
        },
        stdout: b"out".to_vec(),
        stderr: b"err".to_vec(),
        pending_child: None,
    }
}

#[test]
#[ignore = "exclusive native lease; run this test alone"]
fn original_nogame_evidence_survives_callback_panic_and_partial_create_new() {
    let root = OwnedTestRoot::new();
    let mut owner = FixtureOwner {
        owner: prepared_original(&root),
        root: &root,
    };
    // Explicitly synthetic unverified bytes exercise borrowing and storage;
    // no process is spawned and no physical fence is manufactured.
    owner.process = Some(unresolved_test_payload());
    assert!(owner.unresolved());
    let original_address = owner.process.as_ref().unwrap().stdout.as_ptr();
    for grants in [true, false] {
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<Option<()>, ()> = owner.with_raw_evidence(|view| {
                assert!(view.physical_or_process_completion_unknown);
                if grants {
                    let _ = view.with_grants::<()>(|_| panic!("fixture inspection panic"));
                } else {
                    let _ = view.with_control::<()>(|_| panic!("fixture inspection panic"));
                }
                Ok(())
            });
        }));
        assert!(unwind.is_err());
    }
    let preserved: Result<Option<()>, ()> = owner.with_raw_evidence(|view| {
        assert!(view.physical_or_process_completion_unknown);
        assert!(matches!(
            view.with_grants(|g| g.len()),
            IndependentNoGameBorrowResult::Observed(2)
        ));
        assert!(matches!(
            view.with_control(|c| c.progress().retained_grants),
            IndependentNoGameBorrowResult::Observed(0)
        ));
        assert_eq!(view.process.unwrap().stdout.as_ptr(), original_address);
        assert_eq!(view.process.unwrap().stdout, b"out");
        Ok(())
    });
    assert!(matches!(preserved, Ok(Some(()))));

    let attempt = root.0.join("outputs/attempt");
    fs::write(attempt.join("evidence.stdout"), b"existing").unwrap();
    let saved = owner.persist_raw_evidence("evidence").unwrap();
    assert!(
        saved
            .files
            .iter()
            .any(|(name, result)| name == "evidence.stdout" && result.is_err())
    );
    assert!(
        saved
            .files
            .iter()
            .any(|(name, result)| name == "evidence.stderr" && result.is_ok())
    );
    assert!(
        saved
            .files
            .iter()
            .any(|(name, result)| name == "evidence.summary.json" && result.is_ok())
    );
    assert_eq!(
        fs::read(attempt.join("evidence.stdout")).unwrap(),
        b"existing"
    );
    assert_eq!(fs::read(attempt.join("evidence.stderr")).unwrap(), b"err");
    let summary: serde_json::Value =
        serde_json::from_slice(&fs::read(attempt.join("evidence.summary.json")).unwrap()).unwrap();
    assert_eq!(summary["physical_or_process_completion_unknown"], true);
    assert!(owner.unresolved());
    assert_eq!(
        owner.process.as_ref().unwrap().stdout.as_ptr(),
        original_address
    );
    assert_eq!(owner.process.as_ref().unwrap().stderr, b"err");
    assert!(owner.persist_raw_evidence("second").is_err());

    // Discard the explicit test-only payload before dropping the real owner;
    // no child was ever spawned and no device/session fence is asserted.
    assert!(owner.spawn_custody.child().is_none());
    assert!(owner.spawn_custody.last_spawn_pid.is_none());
    assert!(!owner.spawn_custody.pending.load(Ordering::Acquire));
    drop(owner.process.take());
    drop(owner);
}
