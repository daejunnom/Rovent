//! CPU-only V3 transfer checks using an actual child process boundary.
//! These do not start an archive worker, load a model, or prove a GPU fence.

use super::*;
use crate::rovent_bootstrap::process_boundary::{
    RoventProcessBoundaryInputsV3, SourceEvidencePolicy,
    verify_actual_owned_process_boundary_v3_with_policy,
};
use rz_contracts::{Digest, ProcessEpoch};
use rz_experiments::{
    NativeRuntimeSourceV6, ROVENT_EXECUTION_V3_DOMAIN, ROVENT_STORAGE_MEASUREMENT_CONTRACT,
    ROVENT_STORAGE_PROFILE_DOMAIN, RoventArchiveRuntimeLimits, RoventExecutionManifestV3,
    RoventStorageHelpers, RoventStoragePolicy, RoventStorageSourcePins, SystemRuntimeLockV6,
};
use sha2::{Digest as _, Sha256};
use std::io::Read;
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

static NEXT: AtomicU64 = AtomicU64::new(0);
const CHILD_ROOT: &str = "ROVENT_V3_TRANSFER_TEST_ROOT";

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn artifact(path: &str, bytes: &[u8]) -> ArtifactRef {
    ArtifactRef {
        path: path.into(),
        bytes: bytes.len() as u64,
        sha256: sha(bytes),
        source: "https://example.org/v3-storage-cpu-fixture".into(),
        license: "MIT".into(),
    }
}

/// The test process has a separate cwd; no process-wide chdir is used by the
/// parent test harness. Child stdout belongs to this test only.
#[test]
fn v3_transfer_real_process_boundary_child() {
    let base = std::env::temp_dir().canonicalize().unwrap();
    let root = base.join(format!(
        "rovent-v3-transfer-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&root).unwrap();
    for name in ["private", "archive", "checkout", "protected"] {
        std::fs::create_dir(root.join(name)).unwrap();
    }
    let output = Command::new(std::env::current_exe().unwrap())
        .current_dir(root.join("private"))
        .env(CHILD_ROOT, &root)
        .arg("v3_transfer_child_body")
        .arg("--nocapture")
        .output()
        .unwrap();
    // Only the freshly created child tree in the canonical temp directory.
    let info = std::fs::symlink_metadata(&root).unwrap();
    assert!(root.is_absolute() && root.parent() == Some(base.as_path()));
    assert!(
        root.file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("rovent-v3-transfer-")
    );
    assert!(info.is_dir() && !info.file_type().is_symlink());
    std::fs::remove_dir_all(&root).unwrap();
    assert!(
        output.status.success(),
        "V3 transfer child failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn v3_transfer_child_body() {
    let Some(root) = std::env::var_os(CHILD_ROOT).map(PathBuf::from) else {
        return;
    };
    let private_path = root.join("private");
    assert_eq!(std::env::current_dir().unwrap(), private_path);
    let inputs = private_path.join("inputs");
    std::fs::create_dir(&inputs).unwrap();
    let mut pins = Vec::new();
    for (name, body) in [
        ("store.rs", b"test storage source\n".as_slice()),
        ("codec.rs", b"test codec source\n".as_slice()),
        ("launch.rs", b"test launch source\n".as_slice()),
    ] {
        std::fs::write(inputs.join(name), body).unwrap();
        pins.push(artifact(&format!("inputs/{name}"), body));
    }
    let profile = RoventStorageProfile {
        schema: ROVENT_STORAGE_PROFILE_DOMAIN.into(),
        policy: RoventStoragePolicy::ArchiveOnV1,
        archive_limits: RoventArchiveRuntimeLimits {
            game_bytes_max: 1 << 20,
            global_bytes_max: 2 << 20,
            index_entries_max: 16,
            index_bytes_max: 1024,
            load_bytes_max: 1024,
            load_deadline_max_ms: 100,
            record_payload_bytes_max: 1024,
            max_load_pins: 1,
        },
        maintenance_io_bytes_max: NATIVE_MAINTENANCE_IO_BYTES,
        worker_threads: 1,
        command_capacity: 1,
        completion_capacity: 1,
        helpers: RoventStorageHelpers {
            shared_nnuec: 0,
            archive: 1,
        },
        source_pins: RoventStorageSourcePins {
            source: pins[0].clone(),
            codec: pins[1].clone(),
            launch: pins[2].clone(),
        },
        measurement_contract: ROVENT_STORAGE_MEASUREMENT_CONTRACT.into(),
    };
    let profile_bytes = serde_json::to_vec(&profile).unwrap();
    std::fs::write(inputs.join("storage-profile.json"), &profile_bytes).unwrap();
    let profile_ref = artifact("inputs/storage-profile.json", &profile_bytes);
    let exe = std::env::current_exe().unwrap();
    let mut image = std::fs::File::open(&exe).unwrap();
    let image_bytes = image.metadata().unwrap().len();
    let mut image_hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = image.read(&mut buffer).unwrap();
        if read == 0 {
            break;
        }
        image_hash.update(&buffer[..read]);
    }
    let binary = ArtifactRef {
        path: "inputs/current-test-executable".into(),
        bytes: image_bytes,
        sha256: format!("{:x}", image_hash.finalize()),
        source: "https://example.org/v3-storage-cpu-fixture".into(),
        license: "MIT".into(),
    };
    let placeholder = artifact("inputs/fixture-model", b"fixture model");
    let model = serde_json::json!({
        "semantics":"storage-fixture-model/1","encoding":"storage-fixture-input/1",
        "policy_head":"legal-candidates/1","value_head":"side-to-move-wdl/1",
        "implementation_sha256":sha(b"model-implementation"),
        "weights":placeholder,"provenance":artifact("inputs/model-provenance", b"fixture"),
        "diagnostic":false
    });
    let search_digest = sha(b"search-configuration");
    let mut provenance = vec![profile_ref.clone()];
    provenance.extend(pins);
    let wire = serde_json::json!({
        "domain":ROVENT_EXECUTION_V3_DOMAIN,
        "contract_revision":rz_contracts::rovent::ROVENT_SEARCH_REVISION,
        "run_id":"storage-v3-transfer-test","purpose":"validation",
        "engine":{
            "source_commit":null,"source_components":{"storage":sha(b"source-components")},
            "binary":binary,"mode":"cpu_pc",
            "pc":{"proposer":model.clone(),"critic":model,"validator":null,
                "backend":"rust_cpu","device":{"device":"cpu"},
                "backend_configuration_sha256":sha(b"backend"),"runtime_artifacts":[],
                "fp32":true,"tf32":false,"batch_width":1,"ort_intra_threads":null,
                "ort_inter_threads":null,"ort_spinning":null},
            "researcher":null,"teachers":[],"nnuec":null,
            "search":{"node_unit":"rovent-position-visits/1","requested_nodes":null,
                "depth_plies":null,"mate_moves":null,"registered_max_chess_plies":64,
                "searchmoves":null,"search_configuration_sha256":search_digest,
                "root_state_and_history_sha256":sha(b"root"),
                "rules_profile_sha256":sha(b"rules")}
        },
        "resources":{"cpu_quota_millicpus":2000,"pools":[
            {"actors":["manager","pc_caller"],"max_threads":1,"cpu_charge_millicpus":1000},
            {"actors":["runtime_helper"],"max_threads":1,"cpu_charge_millicpus":1000}],
            "memory_high_bytes":3u64<<30,"memory_max_bytes":4u64<<30,"gpu_devices":0,
            "physical_nn_concurrency":1,"topology_policy":"shared-control/1"},
        "coordinator":{"policy_sha256":sha(b"policy"),"questions_max":64,
            "ancestor_depth_max":64,"waiting_jobs_max":64,"role_tickets_max":16,
            "cpu_credit_limit":0},
        "wall_time_max_ms":120000,"cleanup_max_ms":30000,"output_bytes_max":1u64<<20,
        "agent_renewable":true,"provenance":provenance,
        "startup_document":artifact("inputs/startup.json", b"fixture startup"),
        "startup_configuration_sha256":sha(b"startup"),
        "startup_search_configuration_sha256":search_digest,
        "own_r_root_tie":{"kind":"not_applicable","reason":"pc_only"},
        "system_runtime_lock":SystemRuntimeLockV6::new(NativeRuntimeSourceV6::Cpu).unwrap()
    });
    let lock = Arc::new(
        RoventExecutionManifestV3::from_json(&wire.to_string())
            .unwrap()
            .lock()
            .unwrap(),
    );
    let until = Instant::now() + Duration::from_secs(300);
    let control = RoventStartupControl {
        until,
        cleanup_until: until + Duration::from_secs(30),
        cancellation: Arc::new(AtomicBool::new(false)),
    };
    let private = Dir::open_ambient_dir(&private_path, cap_std::ambient_authority()).unwrap();
    let boundary = verify_actual_owned_process_boundary_v3_with_policy(
        RoventProcessBoundaryInputsV3 {
            lock: Arc::clone(&lock),
            original_binary: std::fs::File::open(exe).unwrap(),
            original_root: private.try_clone().unwrap(),
            private_cwd: private_path.clone(),
            control: control.clone(),
        },
        SourceEvidencePolicy::FunctionalWithSourceObservation,
    )
    .unwrap_or_else(|failure| panic!("real V3 boundary failed: {:?}", failure.error));
    let mut original = Some(
        open_rovent_storage_byte_custody_v3(
            Arc::clone(&lock),
            &private,
            RoventStorageAdmissionRequest {
                profile_path: &profile_ref.path,
                profile_sha256: &profile_ref.sha256,
                profile_bytes: profile_ref.bytes,
                archive_root: &root.join("archive"),
                checkout_root: &root.join("checkout"),
                private_snapshot_root: &private_path,
                protected_execution_root: &root.join("protected"),
            },
            control.clone(),
        )
        .unwrap_or_else(|failure| panic!("V3 byte admission failed: {:?}", failure.error)),
    );
    let fd = original
        .as_ref()
        .unwrap()
        .owners()
        .profile_file()
        .unwrap()
        .as_raw_fd();
    let original_identity = original
        .as_ref()
        .unwrap()
        .owners()
        .profile_file()
        .unwrap()
        .metadata()
        .unwrap();
    let transfer = RoventStorageTransferSlotV3::new_empty();
    transfer.adopt_from(&mut original).unwrap();
    assert!(original.is_none());
    assert!(transfer.adopt_from(&mut original).is_err());
    let clock = ProcessClock::new(ProcessEpoch(1));
    let implementation = Digest(Sha256::digest(b"fixture-implementation").into());

    // Failed and panicking preparation leave the exact original owner in the
    // scope-owned slot. The panic is outside the infallible commit interval.
    let invalid_clock = ProcessClock::new(ProcessEpoch(0));
    assert!(
        transfer
            .prepare(&boundary, &invalid_clock, implementation)
            .is_err()
    );
    let held = transfer.custody.lock().unwrap();
    assert!(transfer.prepare(&boundary, &clock, implementation).is_err());
    drop(held);
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _prepared = transfer.prepare(&boundary, &clock, implementation).unwrap();
        panic!("test panic after preparation");
    }));
    assert!(panic_result.is_err());
    {
        let slot = transfer.custody.lock().unwrap();
        let owner = slot.as_ref().unwrap().owners();
        assert!(Arc::ptr_eq(owner.input_lock(), &lock));
        assert_eq!(owner.profile_file().unwrap().as_raw_fd(), fd);
        let current = owner.profile_file().unwrap().metadata().unwrap();
        assert_eq!(
            (current.dev(), current.ino(), current.len()),
            (
                original_identity.dev(),
                original_identity.ino(),
                original_identity.len()
            )
        );
        assert!(owner.retains_root_capability());
    }

    // Two independently prepared values contend for one original owner.
    let first = transfer.prepare(&boundary, &clock, implementation).unwrap();
    let duplicate = transfer.prepare(&boundary, &clock, implementation).unwrap();
    let after_commit_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        first.commit().unwrap();
        panic!("test panic after committed transfer");
    }));
    assert!(after_commit_panic.is_err());
    assert!(duplicate.commit().is_err());
    assert!(transfer.custody.lock().unwrap().is_none());
    let registration = transfer.registration.lock().unwrap();
    let owner = registration.as_ref().unwrap().owners();
    assert!(Arc::ptr_eq(owner.input_lock(), &lock));
    assert_eq!(owner.profile_file().unwrap().as_raw_fd(), fd);
    let current = owner.profile_file().unwrap().metadata().unwrap();
    assert_eq!(
        (current.dev(), current.ino(), current.len()),
        (
            original_identity.dev(),
            original_identity.ino(),
            original_identity.len()
        )
    );
    assert!(owner.retains_root_capability());
    drop(registration);

    // An original cancellation is never reset or renewed. A separately
    // admitted owner shows cancellation after prepare leaves custody intact.
    let mut second = Some(
        open_rovent_storage_byte_custody_v3(
            Arc::clone(&lock),
            &private,
            RoventStorageAdmissionRequest {
                profile_path: &profile_ref.path,
                profile_sha256: &profile_ref.sha256,
                profile_bytes: profile_ref.bytes,
                archive_root: &root.join("archive"),
                checkout_root: &root.join("checkout"),
                private_snapshot_root: &private_path,
                protected_execution_root: &root.join("protected"),
            },
            control.clone(),
        )
        .unwrap_or_else(|failure| panic!("second V3 admission failed: {:?}", failure.error)),
    );
    let second_fd = second
        .as_ref()
        .unwrap()
        .owners()
        .profile_file()
        .unwrap()
        .as_raw_fd();
    let cancel_slot = RoventStorageTransferSlotV3::new_empty();
    cancel_slot.adopt_from(&mut second).unwrap();
    let prepared = cancel_slot
        .prepare(&boundary, &clock, implementation)
        .unwrap();
    control.cancellation.store(true, Ordering::Release);
    assert!(prepared.commit().is_err());
    let cancel_custody = cancel_slot.custody.lock().unwrap();
    let owner = cancel_custody.as_ref().unwrap().owners();
    assert!(Arc::ptr_eq(owner.input_lock(), &lock));
    assert_eq!(owner.profile_file().unwrap().as_raw_fd(), second_fd);
    assert!(owner.retains_root_capability());
    assert!(cancel_slot.registration.lock().unwrap().is_none());
}
