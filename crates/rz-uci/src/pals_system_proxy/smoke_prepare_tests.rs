//! Direct-parent CPU fixture. It does not start native code or read model assets.
use super::*;
use crate::pals_system_proxy::smoke_prepare_lock_fixture::fixture_lock;
use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
use rz_contracts::pals::{
    deadline_grant_v1::DeadlineRole,
    proxy_entry_smoke_grant_v1::{
        ProxyEntrySmokeChallenge, ProxyEntrySmokePhase, ProxyEntrySmokePolicy, ProxyEntrySmokeScope,
    },
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    net::Shutdown,
    os::{
        fd::{AsFd, AsRawFd},
        unix::{
            fs::MetadataExt,
            net::{UnixListener, UnixStream},
        },
    },
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::atomic::AtomicBool,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

fn fd_identity(fd: i32) -> (u64, u64) {
    let metadata = fs::metadata(format!("/proc/self/fd/{fd}")).expect("held FD");
    (metadata.dev(), metadata.ino())
}

fn digest(input: &[u8]) -> String {
    format!("{:x}", Sha256::digest(input))
}

fn decode32(s: &str) -> [u8; 32] {
    let mut value = [0u8; 32];
    for (idx, out) in value.iter_mut().enumerate() {
        *out = u8::from_str_radix(&s[idx * 2..idx * 2 + 2], 16).unwrap();
    }
    value
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
        // Only remove our own named synthetic files; unexpected contents remain.
        let _ = fs::remove_file(self.root.join("inputs/startup.lock"));
        let _ = fs::remove_dir(self.root.join("inputs"));
        let _ = fs::remove_file(self.root.join("channel.sock"));
        let _ = fs::remove_dir(&self.root);
    }
}

#[test]
fn real_parent_partial_challenge_preserves_original_owners() {
    let nonce = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let root = std::env::temp_dir().join(format!("rz-smoke-prepare-{nonce}"));
    fs::create_dir(&root).unwrap();
    let mut fixture = Fixture { root, child: None };
    let lock = fixture_lock();
    fs::create_dir(fixture.root.join("inputs")).unwrap();
    fs::write(
        fixture.root.join("inputs/startup.lock"),
        lock.canonical_bytes().unwrap(),
    )
    .unwrap();
    let listener = UnixListener::bind(fixture.root.join("channel.sock")).unwrap();
    listener.set_nonblocking(true).unwrap();
    let owner_hash = Sha256::digest(format!("owner-{nonce}").as_bytes());
    let mut owner_id = [0u8; 16];
    owner_id.copy_from_slice(&owner_hash[..16]);
    let owner_hex: String = owner_id.iter().map(|b| format!("{b:02x}")).collect();
    let plan_sha = digest(format!("plan-{nonce}").as_bytes());
    let proxy_sha = digest(format!("proxy-{nonce}").as_bytes());
    let child = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("pals_system_proxy::proxy_entry_smoke_prepare_v1::tests::partial_challenge_child")
        .arg("--nocapture")
        .current_dir(&fixture.root)
        .env("RZ_SMOKE_PREPARE_CHILD", "1")
        .env("RZ_SMOKE_PREPARE_ROOT", &fixture.root)
        .env("RZ_SMOKE_PREPARE_OWNER", &owner_hex)
        .env("RZ_SMOKE_PREPARE_PLAN", &plan_sha)
        .env("RZ_SMOKE_PREPARE_PROXY", &proxy_sha)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    fixture.child = Some(child);
    let until = Instant::now() + Duration::from_secs(20);
    let mut channel = loop {
        match listener.accept() {
            Ok((channel, _)) => break channel,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < until => {
                thread::sleep(Duration::from_millis(2))
            }
            Err(error) => panic!("accept child: {error}"),
        }
    };
    let peer = getsockopt(&channel, PeerCredentials).unwrap();
    assert_eq!(peer.pid() as u32, fixture.child.as_ref().unwrap().id());
    let run = ProxyEntrySmokeScope {
        policy: ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1,
        owner_id,
        owner_epoch: 1,
        role: DeadlineRole::Baseline,
        generation: lock.config.process_generation,
        request_id: 1,
        lock_sha256: decode32(&lock.canonical_sha256),
        phase: ProxyEntrySmokePhase::Run,
    };
    let mut cleanup = run.clone();
    cleanup.phase = ProxyEntrySmokePhase::Cleanup;
    cleanup.request_id = 2;
    let encoded = ProxyEntrySmokeChallenge { run, cleanup }.encode().unwrap();
    channel.write_all(&encoded[..13]).unwrap();
    channel.shutdown(Shutdown::Write).unwrap();
    let child = fixture.child.as_mut().unwrap();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "child test failed: {status}");
            break;
        }
        assert!(Instant::now() < until, "child test timed out");
        thread::sleep(Duration::from_millis(2));
    }
    fixture.child = None;
}

#[test]
fn partial_challenge_child() {
    if std::env::var_os("RZ_SMOKE_PREPARE_CHILD").is_none() {
        return;
    }
    let entry_at = Instant::now();
    let root = PathBuf::from(std::env::var_os("RZ_SMOKE_PREPARE_ROOT").unwrap());
    let lock = fixture_lock();
    let cancelled = Arc::new(AtomicBool::new(false));
    let cleanup_cancelled = Arc::new(AtomicBool::new(false));
    let entry = ProxyEntryFunctionalSmokeV1::from_local_entry(entry_at);
    let admission = match OriginalSmokeLockAdmission::open(
        &root.join("inputs/startup.lock"),
        &lock.canonical_sha256,
        lock.config.native_options.startup_ms,
        &root,
        entry_at,
        &cancelled,
    ) {
        Ok(value) => value,
        Err(error) => panic!("lock admission: {}", error.error),
    };
    let lock_fd = admission.retained_file_fd();
    let lock_identity = fd_identity(lock_fd);
    let arguments = vec![
        "--pals-smoke-policy=proxy-entry-v1".into(),
        format!(
            "--pals-smoke-owner-id={}",
            std::env::var("RZ_SMOKE_PREPARE_OWNER").unwrap()
        ),
        "--pals-smoke-owner-epoch=1".into(),
        "--pals-smoke-role=baseline".into(),
        format!("--pals-smoke-generation={}", lock.config.process_generation),
        format!("--pals-smoke-lock-sha256={}", lock.canonical_sha256),
        format!(
            "--pals-smoke-plan-sha256={}",
            std::env::var("RZ_SMOKE_PREPARE_PLAN").unwrap()
        ),
        format!(
            "--pals-smoke-proxy-sha256={}",
            std::env::var("RZ_SMOKE_PREPARE_PROXY").unwrap()
        ),
    ];
    let parsed =
        ParsedSmokeExpectedArgvV1::from_arguments(arguments).unwrap_or_else(|(e, _)| panic!("{e}"));
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
        entry,
        admission,
        stdio,
        &cancelled,
        &cleanup_cancelled,
    )
    .unwrap_or_else(|(e, _, _, _, _)| panic!("{e}"));
    let channel = UnixStream::connect(root.join("channel.sock")).unwrap();
    let socket_fd = channel.as_raw_fd();
    let socket_id = fd_identity(socket_fd);
    let until = Instant::now() + Duration::from_secs(15);
    owner
        .attach_channel(channel, until)
        .unwrap_or_else(|(e, _)| panic!("{e}"));
    let failure = loop {
        match owner.poll() {
            Ok(_) if Instant::now() < until => thread::sleep(Duration::from_millis(2)),
            Ok(_) => panic!("challenge did not finish"),
            Err(error) => break error,
        }
    };
    assert_eq!(failure.kind(), io::ErrorKind::UnexpectedEof);
    assert_eq!(owner.stage(), SmokePreparationStageV1::Failed);
    assert_eq!(owner.first_failure().unwrap().0, failure.kind());
    assert_eq!(owner.poll().unwrap_err().kind(), failure.kind());
    assert_eq!(
        owner.challenge.as_ref().unwrap().retained_fd_and_read(),
        Some((socket_fd, 13))
    );
    assert_eq!(fd_identity(socket_fd), socket_id);
    assert_eq!(fd_identity(lock_fd), lock_identity);
    assert_eq!(stdio_fds.map(fd_identity), stdio_ids);
    let owner = match owner.into_prepared_grants() {
        Err(owner) => owner,
        Ok(_) => panic!("partial challenge accepted"),
    };
    assert_eq!(
        owner.challenge.as_ref().unwrap().retained_fd_and_read(),
        Some((socket_fd, 13))
    );
    assert_eq!(fd_identity(lock_fd), lock_identity);
    assert_eq!(stdio_fds.map(fd_identity), stdio_ids);
}
