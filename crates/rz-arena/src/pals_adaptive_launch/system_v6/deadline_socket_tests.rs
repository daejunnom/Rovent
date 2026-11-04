//! Linux CPU-only deadline transport checks with an actual supervised child.
//! These do not launch a model, verify a snapshot FD, or establish GPU completion.

use super::*;
use crate::process::{self, CleanupStatus, OwnedArtifactTreeWatch, ProcessLimits, ProcessStop};
use rz_contracts::pals::deadline_grant_v1::{DeadlineGrant, DeadlineRequest, GRANT_BYTES};
use rz_experiments::{NativeRuntimeSourceV6, PalsSystemStartupLockV6, SystemRuntimeLockV6};
use sha2::{Digest as _, Sha256};
use std::{
    ffi::OsString,
    fs::{self, File},
    io::{Read, Write},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

const MODE: &str = ".rz-deadline-test-mode";
const REQUESTS: &str = ".rz-deadline-test-requests";
const FASTCHESS_HELPER: &str =
    "pals_adaptive_launch::system_v6::deadline_control::socket_tests::fastchess_helper";
const PROXY_HELPER: &str =
    "pals_adaptive_launch::system_v6::deadline_control::socket_tests::proxy_helper";
static NEXT: AtomicU64 = AtomicU64::new(0);

/// Executed only in the temporary supervised directory. It is a process-tree
/// fixture, not a Fastchess binary or a chess game.
#[test]
fn fastchess_helper() {
    let Ok(mode) = fs::read_to_string(MODE) else {
        return;
    };
    if mode == "rogue" {
        std::thread::sleep(Duration::from_secs(30));
        return;
    }
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", PROXY_HELPER, "--nocapture"])
        .status()
        .unwrap();
    assert!(status.success());
}

/// The child sends requests on separate actual sockets from the same PID.
/// Authority is still granted only by the original supervisor's callback.
#[test]
fn proxy_helper() {
    let Ok(mode) = fs::read_to_string(MODE) else {
        return;
    };
    let requests = fs::read(REQUESTS).unwrap();
    assert_eq!(requests.len() % REQUEST_BYTES, 0);
    for request in requests.chunks_exact(REQUEST_BYTES) {
        let mut peer = UnixStream::connect("../deadline.sock").unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(20)))
            .unwrap();
        peer.write_all(request).unwrap();
        let mut bytes = [0; GRANT_BYTES];
        let result = peer.read_exact(&mut bytes);
        if mode != "success" {
            // The supervisor intentionally rejects the request or interrupts
            // the partially sent frame. This is not a completed grant.
            assert!(result.is_err());
            return;
        }
        result.unwrap();
        let grant = DeadlineGrant::decode(&bytes).unwrap();
        assert_eq!(grant.scope, DeadlineRequest::decode(request).unwrap().scope);
    }
}

fn long_pair() -> LockedPalsSystemArenaPairV6 {
    let mut pair = super::super::pair::tests::fixture();
    for endpoint in &mut pair.endpoints {
        let mut config = endpoint.startup_lock.config.clone();
        config.window.wall_time_max_ms = 60_000;
        config.window.cleanup_max_ms = 10_000;
        config.native_options.startup_ms = 30_000;
        let NativeRuntimeSourceV6::SystemCuda(mut runtime) =
            config.system_runtime_lock.runtime.clone()
        else {
            panic!("test declaration must use the existing V6 fixture");
        };
        runtime.startup_ms = 30_000;
        config.endpoint.runtime.options = config.native_options.canonical_runtime_options(&runtime);
        config.system_runtime_lock =
            SystemRuntimeLockV6::new(NativeRuntimeSourceV6::SystemCuda(runtime)).unwrap();
        endpoint.startup_lock = PalsSystemStartupLockV6::new(config).unwrap();
        let bytes = endpoint.startup_lock.canonical_bytes().unwrap();
        endpoint.startup_lock_artifact.bytes = bytes.len() as u64;
        endpoint.startup_lock_artifact.sha256 = format!("{:x}", Sha256::digest(&bytes));
        endpoint.handshake_max_ms = 30_000;
    }
    pair.lock().unwrap()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Case {
    Success,
    WrongRole,
    WrongLock,
    ReusedIdentity,
    PartialCleanup,
    RoguePeer,
}

fn run_case(case: Case) {
    let base = std::env::temp_dir().canonicalize().unwrap();
    let root = base.join(format!(
        "rz-v6-deadline-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).unwrap();
    let work = root.join("work");
    fs::create_dir(&work).unwrap();
    let socket_path = root.join("deadline.sock");
    let listener = UnixListener::bind(&socket_path).unwrap();
    let executable = std::env::current_exe().unwrap();
    let pin = File::open(&executable).unwrap();
    let cwd = File::open(&work).unwrap();
    let pair = long_pair();
    let mut execution = PalsV6ExecutionStart::begin(&pair).unwrap();
    let digest = Sha256::digest(pair.sha256().as_bytes());
    let mut owner_id = [0_u8; 16];
    owner_id.copy_from_slice(&digest[..16]);
    assert_ne!(owner_id, [0; 16]);
    let epoch = 1;
    execution
        .register_deadline_identity(owner_id, epoch)
        .unwrap();
    // This is an earlier supervisor-held pre-spawn bound, not a claim that
    // Fastchess exposed the actual proxy spawn timestamp.
    let start_bound = execution.started();
    let window = execution
        .deadline_owner_window(&pair, NativeEngineRole::Baseline, start_bound)
        .unwrap();
    let mut control = DeadlineControl::new(
        &listener,
        &window,
        &pin,
        owner_id,
        epoch,
        execution.cleanup_until(),
    )
    .unwrap();
    let lock_sha = decode_sha256(
        &pair
            .endpoint(NativeEngineRole::Baseline)
            .startup_lock
            .canonical_sha256,
    )
    .unwrap();
    let generation = pair
        .endpoint(NativeEngineRole::Baseline)
        .startup_lock
        .config
        .process_generation;
    let phases = [
        DeadlinePhase::Startup,
        DeadlinePhase::Run,
        DeadlinePhase::Cleanup,
    ];
    let scopes = phases.map(|phase| DeadlineScope {
        owner_id,
        owner_epoch: epoch,
        role: DeadlineRole::Baseline,
        game: 1,
        generation,
        request_id: match phase {
            DeadlinePhase::Startup => 11,
            DeadlinePhase::Run => 12,
            DeadlinePhase::Cleanup => 13,
        },
        lock_sha256: lock_sha,
        phase,
    });
    let mut sent = if case == Case::PartialCleanup {
        vec![scopes[2].clone()]
    } else {
        scopes.to_vec()
    };
    if case == Case::WrongRole {
        sent[0].role = DeadlineRole::Candidate;
    }
    if case == Case::WrongLock {
        sent[0].lock_sha256 = Sha256::digest(b"different original test lock").into();
    }
    let mut wire = Vec::new();
    for scope in &sent {
        wire.extend_from_slice(
            &DeadlineRequest {
                scope: scope.clone(),
            }
            .encode()
            .unwrap(),
        );
    }
    fs::write(work.join(REQUESTS), wire).unwrap();
    fs::write(
        work.join(MODE),
        if case == Case::Success {
            "success"
        } else if case == Case::RoguePeer {
            "rogue"
        } else {
            "rejected"
        },
    )
    .unwrap();
    let rogue = if case == Case::RoguePeer {
        let path = socket_path.clone();
        Some(std::thread::spawn(move || {
            let mut stream = UnixStream::connect(path).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(20)))
                .unwrap();
            let mut byte = [0];
            let _ = stream.read(&mut byte);
        }))
    } else {
        None
    };
    let limits = ProcessLimits {
        wall_ms: 60_000,
        shutdown_grace_ms: 5_000,
        max_output_bytes: 1 << 20,
        max_child_processes: 8,
    };
    let watch = OwnedArtifactTreeWatch::bounded_output(1 << 20, 1 << 20);
    let mut custody = SpawnCustody::default();
    let mut bound_pid = None;
    let mut grants = Vec::new();
    let mut captured_error = None;
    let mut partial_before = None;
    let mut callback = |owner: &SpawnCustody, until: Instant| {
        let observed = if case == Case::PartialCleanup && bound_pid.is_some() {
            None
        } else {
            match control.poll(owner, until) {
                Ok(observation) => observation,
                Err(error) => {
                    captured_error = Some(error.to_string());
                    return Err(error);
                }
            }
        };
        if let Some(observation) = observed {
            grants.push(observation);
        }
        if bound_pid.is_none() {
            if let Some(peer) = control.pending_peer()? {
                let leader = owner.child().unwrap().id();
                assert_ne!(peer.pid, leader);
                assert_eq!(peer.start_ticks, process_identity(peer.pid)?.start_ticks);
                let registered = control.admit_original_endpoint(
                    &pair,
                    &execution,
                    NativeEngineRole::Baseline,
                    1,
                    start_bound,
                    [11, 12, 13],
                )?;
                assert_eq!(registered, scopes);
                bound_pid = Some(peer.pid);
                if case == Case::ReusedIdentity {
                    control.admitted[0].2.start_ticks =
                        control.admitted[0].2.start_ticks.checked_add(1).unwrap();
                }
            }
        }
        if case == Case::PartialCleanup && bound_pid.is_some() {
            let peer = control.peer.as_mut().unwrap();
            match peer.stream.read(&mut peer.request[peer.read..]) {
                Ok(0) => return Err(failure("fixture_peer_eof")),
                Ok(read) => peer.read += read,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(()),
                Err(_) => return Err(failure("fixture_peer_read")),
            }
            if peer.read == REQUEST_BYTES {
                let request = DeadlineRequest::decode(&peer.request).unwrap();
                assert_eq!(request.scope, scopes[2]);
                let frame = window.grant(&request.scope, &peer.request).unwrap();
                match peer.stream.write(&frame[..1]) {
                    Ok(1) => {}
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(()),
                    _ => return Err(failure("fixture_partial_write")),
                }
                peer.response = Some(frame.clone());
                peer.response_until = Some(execution.cleanup_until());
                peer.written = 1;
                let index = control
                    .admitted
                    .iter()
                    .position(|(scope, _, _)| scope == &request.scope)
                    .unwrap();
                let (used_scope, _, _) = control.admitted.remove(index);
                control.used.push(used_scope);
                partial_before = Some(frame.clone());
                // Explicit controlled phase transition after one actual byte;
                // this is not the real process cleanup callback integration.
                let error =
                    match control.poll_cleanup(owner, Instant::now() + Duration::from_secs(5)) {
                        Err(error) => error,
                        Ok(_) => panic!("partial frame was not rejected"),
                    };
                assert!(
                    error
                        .to_string()
                        .contains("response_deadline_changed_after_partial_write")
                );
                let peer = control.peer.as_ref().unwrap();
                assert_eq!(peer.written, 1);
                assert_eq!(peer.response.as_ref().unwrap(), &frame);
                assert_eq!(peer.response_until, Some(execution.cleanup_until()));
                assert!(control.used.contains(&request.scope));
                captured_error = Some(error.to_string());
                return Err(error);
            }
        }
        Ok(())
    };
    let output = process::supervise_tree_observed_with_spawn_and_control(
        &pin,
        &[
            OsString::from("--exact"),
            OsString::from(FASTCHESS_HELPER),
            OsString::from("--nocapture"),
        ],
        &cwd,
        limits,
        None,
        &watch,
        None,
        Some(&mut custody),
        Some(&mut callback),
    )
    .unwrap();
    if let Some(rogue) = rogue {
        rogue.join().unwrap();
    }
    assert_eq!(
        output.receipt.group_cleanup,
        CleanupStatus::Gone,
        "owned process group was not confirmed gone"
    );
    match case {
        Case::Success => {
            assert_eq!(output.receipt.stop, ProcessStop::Exited);
            assert_eq!(output.receipt.exit_code, Some(0));
            assert_eq!(grants.len(), 3);
            for (grant, scope) in grants.iter().zip(&scopes) {
                assert_eq!(grant.peer_pid, bound_pid.unwrap());
                assert_eq!(&grant.scope, scope);
            }
        }
        Case::WrongRole | Case::WrongLock | Case::ReusedIdentity => {
            assert!(
                captured_error
                    .as_deref()
                    .unwrap_or_default()
                    .contains("scope_not_admitted")
            );
            assert!(grants.is_empty());
        }
        Case::PartialCleanup => {
            assert!(partial_before.is_some());
            assert!(
                captured_error
                    .as_deref()
                    .unwrap_or_default()
                    .contains("response_deadline_changed_after_partial_write")
            );
            assert!(grants.is_empty());
        }
        Case::RoguePeer => {
            assert!(
                captured_error
                    .as_deref()
                    .unwrap_or_default()
                    .contains("peer_relation")
            );
            assert!(grants.is_empty());
        }
    }
    let metadata = fs::symlink_metadata(&root).unwrap();
    assert!(
        root.is_absolute()
            && root.parent() == Some(base.as_path())
            && metadata.is_dir()
            && !metadata.file_type().is_symlink()
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn original_child_socket_grants_three_bound_phases() {
    run_case(Case::Success);
}

#[test]
fn original_child_rejects_wrong_role_lock_and_start_identity() {
    run_case(Case::WrongRole);
    run_case(Case::WrongLock);
    // A deliberately altered retained start tick tests the reuse check; it is
    // not a claim that the operating system actually recycled a PID.
    run_case(Case::ReusedIdentity);
}

#[test]
fn earlier_cleanup_rejects_an_actually_partially_sent_frame() {
    run_case(Case::PartialCleanup);
}

#[test]
fn parent_process_cannot_impersonate_fastchess_child() {
    run_case(Case::RoguePeer);
}
