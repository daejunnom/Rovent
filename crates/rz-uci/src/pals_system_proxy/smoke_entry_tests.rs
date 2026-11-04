//! NoGame entry preparation only: no child engine, model, or GPU execution.
use super::*;
use crate::pals_system_proxy::{
    proxy_entry_smoke_prepare_v1::OriginalProxyStdioV1,
    smoke_prepare_lock_fixture::fixture_lock_with_engine,
};
use nix::{
    fcntl::OFlag,
    sys::socket::{getsockopt, sockopt::PeerCredentials},
    unistd::pipe2,
};
use rz_contracts::pals::{
    deadline_grant_v1::DeadlineRole,
    proxy_entry_smoke_grant_v1::{
        PROXY_ENTRY_SMOKE_REQUEST_BYTES, ProxyEntrySmokeChallenge, ProxyEntrySmokeGrant,
        ProxyEntrySmokePhase, ProxyEntrySmokePolicy, ProxyEntrySmokeRequest, ProxyEntrySmokeScope,
    },
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read, Write},
    os::{
        fd::AsRawFd,
        linux::net::SocketAddrExt,
        unix::{
            fs::{DirBuilderExt, MetadataExt, PermissionsExt},
            net::{SocketAddr, UnixListener, UnixStream},
        },
    },
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Arc, atomic::AtomicBool},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn hex32(value: &str) -> [u8; 32] {
    let mut output = [0; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap();
    }
    output
}
fn id(fd: i32) -> (u64, u64) {
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
        for path in ["inputs/startup.lock", "test/engine"] {
            let _ = fs::remove_file(self.root.join(path));
        }
        for path in ["inputs", "test"] {
            let _ = fs::remove_dir(self.root.join(path));
        }
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
fn listener(name: &str) -> UnixListener {
    let address = SocketAddr::from_abstract_name(name.as_bytes()).unwrap();
    let result = UnixListener::bind_addr(&address).unwrap();
    result.set_nonblocking(true).unwrap();
    result
}
#[test]
fn entry_prepares_direct_parent_grants_and_cancel_or_retains_failure() {
    for scenario in ["success", "missing_cancel"] {
        run_parent(scenario);
    }
}
fn run_parent(scenario: &str) {
    let nonce = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let root = std::env::temp_dir().join(format!("rz-entry-{nonce}"));
    fs::DirBuilder::new().mode(0o700).create(&root).unwrap();
    let mut fixture = Fixture { root, child: None };
    fs::create_dir(fixture.root.join("inputs")).unwrap();
    fs::create_dir(fixture.root.join("test")).unwrap();
    // ELF header is sufficient for the pin; this byte string is never executed.
    let engine = format!("\u{7f}ELF synthetic entry test {nonce}").into_bytes();
    let lock = fixture_lock_with_engine(&engine);
    fs::write(fixture.root.join("test/engine"), &engine).unwrap();
    fs::set_permissions(
        fixture.root.join("test/engine"),
        fs::Permissions::from_mode(0o700),
    )
    .unwrap();
    fs::write(
        fixture.root.join("inputs/startup.lock"),
        lock.canonical_bytes().unwrap(),
    )
    .unwrap();
    let grant_name = format!("rz-entry-grant-{nonce}");
    let cancel_name = format!("rz-entry-cancel-{nonce}");
    let grant_listener = listener(&grant_name);
    let cancel_listener = (scenario == "success").then(|| listener(&cancel_name));
    let owner_hash = Sha256::digest(format!("owner-{nonce}").as_bytes());
    let mut owner_id = [0; 16];
    owner_id.copy_from_slice(&owner_hash[..16]);
    let owner_hex: String = owner_id.iter().map(|byte| format!("{byte:02x}")).collect();
    let child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "pals_system_proxy::proxy_entry_smoke_entry_v1::tests::entry_child",
            "--nocapture",
        ])
        .current_dir(&fixture.root)
        .env("RZ_ENTRY_SCENARIO", scenario)
        .env("RZ_ENTRY_ROOT", &fixture.root)
        .env("RZ_ENTRY_ENGINE_NONCE", &nonce)
        .env("RZ_ENTRY_GRANT", &grant_name)
        .env("RZ_ENTRY_CANCEL", &cancel_name)
        .env("RZ_ENTRY_OWNER", owner_hex)
        .env("RZ_ENTRY_PLAN", sha(format!("plan-{nonce}").as_bytes()))
        .env("RZ_ENTRY_PROXY", sha(format!("proxy-{nonce}").as_bytes()))
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
        let mut stream = accept(&grant_listener, child_id, until);
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
    let cancel = cancel_listener
        .as_ref()
        .map(|socket| accept(socket, child_id, until));
    let child = fixture.child.as_mut().unwrap();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "entry child: {status}");
            break;
        }
        assert!(Instant::now() < until, "entry child timed out");
        thread::sleep(Duration::from_millis(2));
    }
    fixture.child = None;
    drop(cancel);
    drop(grants);
}
#[test]
fn entry_child() {
    let Some(scenario) = std::env::var_os("RZ_ENTRY_SCENARIO") else {
        return;
    };
    let entry_at = Instant::now();
    let root = PathBuf::from(std::env::var_os("RZ_ENTRY_ROOT").unwrap());
    let nonce = std::env::var("RZ_ENTRY_ENGINE_NONCE").unwrap();
    let engine = format!("\u{7f}ELF synthetic entry test {nonce}").into_bytes();
    let lock = fixture_lock_with_engine(&engine);
    let args = vec![
        "--pals-proxy-entry-functional-smoke=true".into(),
        "--pals-system-proxy=true".into(),
        format!(
            "--pals-system-child-binary={}",
            root.join("test/engine").display()
        ),
        format!(
            "--pals-system-startup-lock={}",
            root.join("inputs/startup.lock").display()
        ),
        format!(
            "--pals-system-startup-lock-sha256={}",
            lock.canonical_sha256
        ),
        format!(
            "--pals-system-startup-ms={}",
            lock.config.native_options.startup_ms
        ),
        format!("--pals-system-parent-abstract=rz-entry-child-{nonce}"),
        format!(
            "--pals-arena-grant-socket=abstract:{}",
            std::env::var("RZ_ENTRY_GRANT").unwrap()
        ),
        format!(
            "--pals-arena-cancel-socket=abstract:{}",
            std::env::var("RZ_ENTRY_CANCEL").unwrap()
        ),
        "--pals-smoke-policy=proxy-entry-v1".into(),
        format!(
            "--pals-smoke-owner-id={}",
            std::env::var("RZ_ENTRY_OWNER").unwrap()
        ),
        "--pals-smoke-owner-epoch=1".into(),
        "--pals-smoke-role=baseline".into(),
        format!("--pals-smoke-generation={}", lock.config.process_generation),
        format!("--pals-smoke-lock-sha256={}", lock.canonical_sha256),
        format!(
            "--pals-smoke-plan-sha256={}",
            std::env::var("RZ_ENTRY_PLAN").unwrap()
        ),
        format!(
            "--pals-smoke-proxy-sha256={}",
            std::env::var("RZ_ENTRY_PROXY").unwrap()
        ),
    ];
    let (incoming, _incoming_peer) = pipe2(OFlag::O_CLOEXEC).unwrap();
    let (_outgoing_peer, outgoing) = pipe2(OFlag::O_CLOEXEC).unwrap();
    let (_diagnostics_peer, diagnostics) = pipe2(OFlag::O_CLOEXEC).unwrap();
    let fds = [
        incoming.as_raw_fd(),
        outgoing.as_raw_fd(),
        diagnostics.as_raw_fd(),
    ];
    let identities = fds.map(id);
    let stdio = OriginalProxyStdioV1 {
        incoming,
        outgoing,
        diagnostics,
    };
    let run = Arc::new(AtomicBool::new(false));
    let cleanup = Arc::new(AtomicBool::new(false));
    let result = prepare_entry(args, entry_at, &root, stdio, &run, &cleanup, 4096);
    match (scenario.to_str().unwrap(), result) {
        ("success", Ok(owner)) => {
            assert_eq!(
                owner.bundle.lock_admission.lock().canonical_sha256,
                lock.canonical_sha256
            );
            assert_eq!(fds.map(id), identities);
            drop(owner);
        }
        (
            "missing_cancel",
            Err(NoGameEntryFailureV1::Cancel {
                error: _error,
                owner,
                stream,
            }),
        ) => {
            assert!(stream.is_none());
            assert_eq!(
                owner.bundle.lock_admission.lock().canonical_sha256,
                lock.canonical_sha256
            );
            assert_eq!(fds.map(id), identities);
            drop(owner);
        }
        (_, Ok(_)) => panic!("unexpected entry success"),
        (_, Err(_)) => panic!("unexpected entry failure"),
    }
}
