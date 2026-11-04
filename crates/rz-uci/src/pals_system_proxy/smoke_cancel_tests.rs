//! CPU-only direct-parent NoGame cancellation fixture; no native/GPU work.
use super::*;
use crate::pals_system_proxy::{
    proxy_entry_smoke_lock_v1::OriginalSmokeLockAdmission,
    proxy_entry_smoke_prepare_v1::{
        NoGameProxyPrepareOwnerV1, OriginalProxyStdioV1, SmokePreparationStageV1,
    },
    proxy_entry_smoke_v1::ProxyEntryFunctionalSmokeV1,
    smoke_expected_argv_v1::ParsedSmokeExpectedArgvV1,
    smoke_prepare_lock_fixture::fixture_lock,
};
use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
use rz_contracts::pals::{
    deadline_grant_v1::DeadlineRole,
    proxy_entry_smoke_cancel_v1::{
        SMOKE_CANCEL_ACK_BYTES, SmokeCancelAckV1, SmokeCancelDispositionV1, SmokeCancelRequestV1,
    },
    proxy_entry_smoke_grant_v1::{
        PROXY_ENTRY_SMOKE_REQUEST_BYTES, ProxyEntrySmokeChallenge, ProxyEntrySmokeGrant,
        ProxyEntrySmokePhase, ProxyEntrySmokePolicy, ProxyEntrySmokeRequest, ProxyEntrySmokeScope,
    },
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read, Write},
    net::Shutdown,
    os::{
        fd::{AsFd, AsRawFd},
        linux::net::SocketAddrExt,
        unix::{
            fs::MetadataExt,
            net::{SocketAddr, UnixListener, UnixStream},
        },
    },
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn hex32(value: &str) -> [u8; 32] {
    let mut result = [0; 32];
    for (index, byte) in result.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap();
    }
    result
}
fn fd_identity(fd: i32) -> (u64, u64) {
    let meta = fs::metadata(format!("/proc/self/fd/{fd}")).unwrap();
    (meta.dev(), meta.ino())
}
struct Fixture {
    root: PathBuf,
    child: Option<Child>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = fs::remove_file(self.root.join("inputs/startup.lock"));
        let _ = fs::remove_dir(self.root.join("inputs"));
        let _ = fs::remove_dir(&self.root);
    }
}
fn accept(listener: &UnixListener, child: u32, until: Instant) -> UnixStream {
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                assert_eq!(
                    getsockopt(&stream, PeerCredentials).unwrap().pid() as u32,
                    child
                );
                stream
                    .set_read_timeout(Some(Duration::from_secs(15)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(15)))
                    .unwrap();
                return stream;
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < until => {
                thread::sleep(Duration::from_millis(2))
            }
            Err(error) => panic!("accept child: {error}"),
        }
    }
}

#[test]
fn direct_parent_cancel_preserves_scope_and_failure_owner() {
    for scenario in ["accepted", "scope", "partial"] {
        run_parent(scenario);
    }
}
fn run_parent(scenario: &str) {
    let nonce = format!(
        "{}-{}-{scenario}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let root = std::env::temp_dir().join(format!("rz-smoke-cancel-{nonce}"));
    fs::create_dir(&root).unwrap();
    let mut fixture = Fixture { root, child: None };
    let lock = fixture_lock();
    fs::create_dir(fixture.root.join("inputs")).unwrap();
    fs::write(
        fixture.root.join("inputs/startup.lock"),
        lock.canonical_bytes().unwrap(),
    )
    .unwrap();
    let name = format!("rz-smoke-cancel-{nonce}");
    let addr = SocketAddr::from_abstract_name(name.as_bytes()).unwrap();
    let listener = UnixListener::bind_addr(&addr).unwrap();
    listener.set_nonblocking(true).unwrap();
    let hash = Sha256::digest(format!("owner-{nonce}").as_bytes());
    let mut owner_id = [0; 16];
    owner_id.copy_from_slice(&hash[..16]);
    let owner_hex: String = owner_id.iter().map(|byte| format!("{byte:02x}")).collect();
    let plan_sha = sha(format!("plan-{nonce}").as_bytes());
    let proxy_sha = sha(format!("proxy-{nonce}").as_bytes());
    let child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "pals_system_proxy::proxy_entry_smoke_cancel_v1::tests::cancel_child",
            "--nocapture",
        ])
        .current_dir(&fixture.root)
        .env("RZ_SMOKE_CANCEL_CHILD", scenario)
        .env("RZ_SMOKE_CANCEL_ROOT", &fixture.root)
        .env("RZ_SMOKE_CANCEL_SOCKET", &name)
        .env("RZ_SMOKE_CANCEL_OWNER", owner_hex)
        .env("RZ_SMOKE_CANCEL_PLAN", plan_sha)
        .env("RZ_SMOKE_CANCEL_PROXY", proxy_sha)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let child_id = child.id();
    fixture.child = Some(child);
    let until = Instant::now() + Duration::from_secs(20);
    let run = ProxyEntrySmokeScope {
        policy: ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1,
        owner_id,
        owner_epoch: 1,
        role: DeadlineRole::Baseline,
        generation: lock.config.process_generation,
        request_id: 1,
        lock_sha256: hex32(&lock.canonical_sha256),
        phase: ProxyEntrySmokePhase::Run,
    };
    let mut cleanup = run.clone();
    cleanup.phase = ProxyEntrySmokePhase::Cleanup;
    cleanup.request_id = 2;
    let challenge = ProxyEntrySmokeChallenge {
        run: run.clone(),
        cleanup: cleanup.clone(),
    }
    .encode()
    .unwrap();
    let mut grants = Vec::new();
    for expected in [&run, &cleanup] {
        let mut stream = accept(&listener, child_id, until);
        stream.write_all(&challenge).unwrap();
        let mut frame = [0; PROXY_ENTRY_SMOKE_REQUEST_BYTES];
        stream.read_exact(&mut frame).unwrap();
        assert_eq!(
            &ProxyEntrySmokeRequest::decode(&frame).unwrap().scope,
            expected
        );
        stream
            .write_all(
                &ProxyEntrySmokeGrant {
                    scope: expected.clone(),
                    remaining_ns: Duration::from_secs(18).as_nanos() as u64,
                }
                .encode()
                .unwrap(),
            )
            .unwrap();
        grants.push(stream);
    }
    let mut control = accept(&listener, child_id, until);
    let mut requested_scope = run.clone();
    if scenario == "scope" {
        requested_scope.role = DeadlineRole::Candidate;
    }
    let frame = SmokeCancelRequestV1 {
        scope: requested_scope,
        sequence: 1,
    }
    .encode()
    .unwrap();
    if scenario == "partial" {
        control.write_all(&frame[..13]).unwrap();
        control.shutdown(Shutdown::Write).unwrap();
    } else {
        control.write_all(&frame).unwrap();
        if scenario == "accepted" {
            for expected in [
                SmokeCancelDispositionV1::Accepted,
                SmokeCancelDispositionV1::Duplicate,
            ] {
                if expected == SmokeCancelDispositionV1::Duplicate {
                    control.write_all(&frame).unwrap();
                }
                let mut ack = [0; SMOKE_CANCEL_ACK_BYTES];
                control.read_exact(&mut ack).unwrap();
                let ack = SmokeCancelAckV1::decode(&ack).unwrap();
                assert_eq!(ack.scope, run);
                assert_eq!(ack.sequence, 1);
                assert_eq!(ack.disposition, expected);
            }
        }
    }
    let child = fixture.child.as_mut().unwrap();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "child failed: {status}");
            break;
        }
        assert!(Instant::now() < until, "child timed out");
        thread::sleep(Duration::from_millis(2));
    }
    fixture.child = None;
    drop(grants);
}

#[test]
fn cancel_child() {
    let Some(scenario) = std::env::var_os("RZ_SMOKE_CANCEL_CHILD") else {
        return;
    };
    let scenario = scenario.to_str().unwrap();
    let entry_at = Instant::now();
    let root = PathBuf::from(std::env::var_os("RZ_SMOKE_CANCEL_ROOT").unwrap());
    let lock = fixture_lock();
    let cancelled = Arc::new(AtomicBool::new(false));
    let cleanup_cancelled = Arc::new(AtomicBool::new(false));
    let admission = OriginalSmokeLockAdmission::open(
        &root.join("inputs/startup.lock"),
        &lock.canonical_sha256,
        lock.config.native_options.startup_ms,
        &root,
        entry_at,
        &cancelled,
    )
    .unwrap_or_else(|error| panic!("lock admission: {}", error.error));
    let lock_fd = admission.retained_file_fd();
    let lock_id = fd_identity(lock_fd);
    let args = vec![
        "--pals-smoke-policy=proxy-entry-v1".into(),
        format!(
            "--pals-smoke-owner-id={}",
            std::env::var("RZ_SMOKE_CANCEL_OWNER").unwrap()
        ),
        "--pals-smoke-owner-epoch=1".into(),
        "--pals-smoke-role=baseline".into(),
        format!("--pals-smoke-generation={}", lock.config.process_generation),
        format!("--pals-smoke-lock-sha256={}", lock.canonical_sha256),
        format!(
            "--pals-smoke-plan-sha256={}",
            std::env::var("RZ_SMOKE_CANCEL_PLAN").unwrap()
        ),
        format!(
            "--pals-smoke-proxy-sha256={}",
            std::env::var("RZ_SMOKE_CANCEL_PROXY").unwrap()
        ),
    ];
    let parsed = ParsedSmokeExpectedArgvV1::from_arguments(args)
        .unwrap_or_else(|(error, _)| panic!("{error}"));
    let stdio = OriginalProxyStdioV1 {
        incoming: std::io::stdin().as_fd().try_clone_to_owned().unwrap(),
        outgoing: std::io::stdout().as_fd().try_clone_to_owned().unwrap(),
        diagnostics: std::io::stderr().as_fd().try_clone_to_owned().unwrap(),
    };
    let stdio_fds = [
        stdio.incoming.as_raw_fd(),
        stdio.outgoing.as_raw_fd(),
        stdio.diagnostics.as_raw_fd(),
    ];
    let stdio_ids = stdio_fds.map(fd_identity);
    let mut owner = NoGameProxyPrepareOwnerV1::new(
        parsed,
        ProxyEntryFunctionalSmokeV1::from_local_entry(entry_at),
        admission,
        stdio,
        &cancelled,
        &cleanup_cancelled,
    )
    .unwrap_or_else(|(error, _, _, _, _)| panic!("prepare: {error}"));
    let name = std::env::var("RZ_SMOKE_CANCEL_SOCKET").unwrap();
    let address = SocketAddr::from_abstract_name(name.as_bytes()).unwrap();
    let until = Instant::now() + Duration::from_secs(15);
    for desired in [
        SmokePreparationStageV1::AwaitCleanupChannel,
        SmokePreparationStageV1::GrantsPrepared,
    ] {
        let stream = UnixStream::connect_addr(&address).unwrap();
        owner
            .attach_channel(stream, until)
            .unwrap_or_else(|(error, _)| panic!("attach: {error}"));
        loop {
            let stage = owner.poll().unwrap();
            if stage == desired {
                break;
            }
            assert!(Instant::now() < until, "grant exchange timed out");
            thread::sleep(Duration::from_millis(2));
        }
    }
    let prepared = owner
        .into_prepared_grants()
        .unwrap_or_else(|_| panic!("grants not prepared"));
    let stream = UnixStream::connect_addr(&address).unwrap();
    let fd = stream.as_raw_fd();
    let socket_id = fd_identity(fd);
    let mut receiver = NoGameCancelReceiverV1::from_original_bundle(stream, &prepared.bundle)
        .unwrap_or_else(|(error, _)| panic!("cancel attach: {error}"));
    assert!(receiver.matches_window(&cancelled, prepared.bundle.window.cleanup_until));
    if scenario == "accepted" {
        for _ in 0..2 {
            loop {
                if receiver.poll().unwrap() {
                    break;
                }
                assert!(Instant::now() < until, "cancel ack timed out");
                thread::sleep(Duration::from_millis(2));
            }
        }
        assert!(cancelled.load(Ordering::Acquire));
        assert!(receiver.first_failure().is_none());
    } else {
        let error = loop {
            match receiver.poll() {
                Err(error) => break error,
                Ok(_) if Instant::now() < until => thread::sleep(Duration::from_millis(2)),
                Ok(_) => panic!("cancel error timed out"),
            }
        };
        let kind = if scenario == "scope" {
            io::ErrorKind::InvalidData
        } else {
            io::ErrorKind::UnexpectedEof
        };
        assert_eq!(error.kind(), kind);
        assert_eq!(receiver.first_failure().unwrap().kind, kind);
        assert_eq!(receiver.poll().unwrap_err().kind(), kind);
        assert!(!cancelled.load(Ordering::Acquire));
    }
    assert!(!cleanup_cancelled.load(Ordering::Acquire));
    assert_eq!(fd_identity(fd), socket_id);
    assert_eq!(fd_identity(lock_fd), lock_id);
    assert_eq!(stdio_fds.map(fd_identity), stdio_ids);
    let same = receiver.into_stream();
    assert_eq!(same.as_raw_fd(), fd);
    assert_eq!(fd_identity(fd), socket_id);
    drop(prepared);
}
