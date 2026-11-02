//! Parent side of the original child-bound system CUDA admission channel.
//!
//! The caller directly spawns and exclusively retains `Child`, accepts its
//! stream from a private listener and retains that listener's filesystem
//! owner. Pathname possession is not authentication: each operation checks
//! the kernel's actual peer PID and UID against the still-owned child. This
//! channel is liveness-only and cannot prove GPU completion or permit a kill.

use nix::{
    errno::Errno,
    fcntl::{FcntlArg, FdFlag, fcntl},
    sys::socket::{MsgFlags, getsockopt, recv, sockopt::PeerCredentials},
    unistd::geteuid,
};
use serde::{Deserialize, Serialize};
use std::{
    io::{self, Read, Write},
    os::{
        fd::{AsFd, AsRawFd},
        unix::net::{UnixListener, UnixStream},
    },
    process::Child,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

const REQUEST_SCHEMA: &str = "rz-rovent-parent-admission/1";
const ACK_SCHEMA: &str = "rz-rovent-parent-admission-ack/1";
const SCOPE: &str = "parent-child-connection-not-gpu-fence";
const FRAME_LIMIT: usize = 4096;

#[derive(Serialize)]
struct Request<'a> {
    schema: &'static str,
    child_pid: u32,
    process_epoch: u64,
    lock_sha256: &'a str,
    cuda_ordinal: i32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Ack {
    schema: String,
    child_pid: u32,
    process_epoch: u64,
    lock_sha256: String,
    cuda_ordinal: i32,
    scope: String,
}

/// The caller retains this exact stream together with its original `Child`.
/// A second connection, child JSON report, or a completed handshake is never
/// substituted for physical GPU completion or a process exit observation.
pub(super) struct AuthenticatedChildChannel {
    stream: UnixStream,
    child_pid: u32,
    child_uid: u32,
    process_epoch: u64,
    lock_sha256: String,
    cuda_ordinal: i32,
}

fn invalid(detail: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, detail)
}

fn remaining(until: Instant) -> io::Result<Duration> {
    until
        .checked_duration_since(Instant::now())
        .filter(|duration| !duration.is_zero())
        .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "child admission deadline expired"))
}

fn valid_lock(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_cloexec(stream: &impl AsFd) -> io::Result<()> {
    // Safe borrowed-FD fcntl only reads the original descriptor's flags.
    let flags = fcntl(stream, FcntlArg::F_GETFD).map_err(io::Error::other)?;
    if !FdFlag::from_bits_truncate(flags).contains(FdFlag::FD_CLOEXEC) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "child admission stream is inheritable",
        ));
    }
    Ok(())
}

fn require_child(
    stream: &UnixStream,
    child: &mut Child,
    expected_pid: u32,
    expected_uid: u32,
) -> io::Result<()> {
    if child.id() != expected_pid || child.try_wait()?.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "original child already exited",
        ));
    }
    let peer = getsockopt(stream, PeerCredentials).map_err(io::Error::other)?;
    if peer.pid() <= 0 || peer.pid() as u32 != expected_pid || peer.uid() != expected_uid {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "original child peer differs",
        ));
    }
    Ok(())
}

fn require_live(stream: &UnixStream) -> io::Result<()> {
    let mut probe = [0u8; 1];
    match recv(
        stream.as_raw_fd(),
        &mut probe,
        MsgFlags::MSG_PEEK | MsgFlags::MSG_DONTWAIT,
    ) {
        Ok(0) => Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "child admission channel closed",
        )),
        Ok(_) => Err(invalid("unexpected data on child liveness channel")),
        Err(Errno::EAGAIN) => Ok(()),
        Err(error) => Err(io::Error::from_raw_os_error(error as i32)),
    }
}

fn next_poll(until: Instant, cancelled: &AtomicBool) -> io::Result<Duration> {
    if cancelled.load(Ordering::Acquire) {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "child admission canceled",
        ));
    }
    Ok(remaining(until)?.min(Duration::from_millis(100)))
}

fn write_frame(
    stream: &mut UnixStream,
    bytes: &[u8],
    until: Instant,
    cancelled: &AtomicBool,
) -> io::Result<()> {
    if bytes.len() >= FRAME_LIMIT {
        return Err(invalid("child admission request exceeds bound"));
    }
    let mut frame = Vec::with_capacity(bytes.len() + 1);
    frame.extend_from_slice(bytes);
    frame.push(b'\n');
    let mut offset = 0;
    while offset < frame.len() {
        stream.set_write_timeout(Some(next_poll(until, cancelled)?))?;
        match stream.write(&frame[offset..]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "child admission request not written",
                ));
            }
            Ok(count) => offset += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                continue;
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn read_ack(stream: &mut UnixStream, until: Instant, cancelled: &AtomicBool) -> io::Result<Ack> {
    let mut frame = Vec::with_capacity(256);
    loop {
        stream.set_read_timeout(Some(next_poll(until, cancelled)?))?;
        let mut byte = [0u8; 1];
        match stream.read(&mut byte) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "child admission response closed",
                ));
            }
            Ok(_) if byte[0] == b'\n' => break,
            Ok(_) if frame.len() < FRAME_LIMIT - 1 => frame.push(byte[0]),
            Ok(_) => return Err(invalid("child admission response exceeds bound")),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                continue;
            }
            Err(error) => return Err(error),
        }
    }
    serde_json::from_slice(&frame).map_err(|_| invalid("invalid child admission acknowledgement"))
}

fn validate_ack(ack: &Ack, pid: u32, epoch: u64, lock: &str, ordinal: i32) -> io::Result<()> {
    if ack.schema != ACK_SCHEMA
        || ack.scope != SCOPE
        || ack.child_pid != pid
        || ack.process_epoch != epoch
        || ack.lock_sha256 != lock
        || ack.cuda_ordinal != ordinal
    {
        return Err(invalid("child admission acknowledgement differs"));
    }
    Ok(())
}

impl AuthenticatedChildChannel {
    /// The stream must be the exclusive accepted stream from the caller's
    /// private listener. `child` must be the actual, directly spawned child,
    /// retained without a competing waiter or another owner reaping its PID.
    /// Same UID and the original peer PID are checked before and after ACK.
    pub(super) fn authenticate(
        mut stream: UnixStream,
        child: &mut Child,
        epoch: u64,
        lock_sha256: &str,
        cuda_ordinal: i32,
        until: Instant,
        cancelled: &AtomicBool,
    ) -> io::Result<Self> {
        let child_pid = child.id();
        if child_pid == 0 || epoch == 0 || !valid_lock(lock_sha256) || cuda_ordinal < 0 {
            return Err(invalid("invalid original child admission identity"));
        }
        let child_uid = geteuid().as_raw();
        next_poll(until, cancelled)?;
        require_cloexec(&stream)?;
        stream.set_nonblocking(false)?;
        require_child(&stream, child, child_pid, child_uid)?;
        let request = Request {
            schema: REQUEST_SCHEMA,
            child_pid,
            process_epoch: epoch,
            lock_sha256,
            cuda_ordinal,
        };
        let bytes =
            serde_json::to_vec(&request).map_err(|_| invalid("cannot encode child admission"))?;
        write_frame(&mut stream, &bytes, until, cancelled)?;
        let ack = read_ack(&mut stream, until, cancelled)?;
        validate_ack(&ack, child_pid, epoch, lock_sha256, cuda_ordinal)?;
        next_poll(until, cancelled)?;
        stream.set_read_timeout(None)?;
        stream.set_write_timeout(None)?;
        require_child(&stream, child, child_pid, child_uid)?;
        require_live(&stream)?;
        next_poll(until, cancelled)?;
        Ok(Self {
            stream,
            child_pid,
            child_uid,
            process_epoch: epoch,
            lock_sha256: lock_sha256.to_owned(),
            cuda_ordinal,
        })
    }

    /// Recheck both the original unreaped child and the original socket.
    /// This records only a live connection, never GPU execution or completion.
    pub(super) fn ensure_live(&self, child: &mut Child) -> io::Result<()> {
        require_child(&self.stream, child, self.child_pid, self.child_uid)?;
        require_live(&self.stream)
    }

    pub(super) fn matches(&self, epoch: u64, lock_sha256: &str, cuda_ordinal: i32) -> bool {
        self.process_epoch == epoch
            && self.lock_sha256 == lock_sha256
            && self.cuda_ordinal == cuda_ordinal
    }
}

/// Accept only the actual directly spawned child from the executor's exclusive
/// private listener. An unrelated peer cannot acquire authority and is
/// discarded; the original deadline and cancellation apply to the entire
/// accept + handshake. The executor owns listener creation, path identity and
/// cleanup; this function never unlinks a socket path or signals a process.
#[allow(clippy::too_many_arguments)]
pub(super) fn accept_and_authenticate_child(
    listener: &UnixListener,
    child: &mut Child,
    epoch: u64,
    lock_sha256: &str,
    cuda_ordinal: i32,
    until: Instant,
    cancelled: &AtomicBool,
) -> io::Result<AuthenticatedChildChannel> {
    if epoch == 0 || !valid_lock(lock_sha256) || cuda_ordinal < 0 {
        return Err(invalid("invalid original child admission identity"));
    }
    require_cloexec(listener)?;
    // The caller guarantees this listener has no clones shared with another
    // owner; set_nonblocking changes its underlying file-status flags.
    listener.set_nonblocking(true)?;
    let pid = child.id();
    let uid = geteuid().as_raw();
    loop {
        let interval = next_poll(until, cancelled)?.min(Duration::from_millis(5));
        if pid == 0 || child.try_wait()?.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "original child exited before admission",
            ));
        }
        match listener.accept() {
            Ok((stream, _)) => match require_child(&stream, child, pid, uid) {
                Ok(()) => {
                    return AuthenticatedChildChannel::authenticate(
                        stream,
                        child,
                        epoch,
                        lock_sha256,
                        cuda_ordinal,
                        until,
                        cancelled,
                    );
                }
                Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {
                    // Avoid a busy loop on repeated unrelated peers; no
                    // message or authority is accepted from them.
                    drop(stream);
                    std::thread::sleep(interval);
                }
                Err(error) => return Err(error),
            },
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => std::thread::sleep(interval),
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The outer test launches only this helper in a separate test process.
    /// When run normally it does nothing and requires no CUDA or model.
    #[test]
    fn direct_child_handshake_helper() {
        let Some(path) = std::env::var_os("RZ_TEST_DIRECT_PARENT_SOCKET") else {
            return;
        };
        let parent =
            crate::rovent_bootstrap::launch::ConnectedParent::capture_original_parent().unwrap();
        let stream = UnixStream::connect(path).unwrap();
        let connected = crate::rovent_bootstrap::launch::ConnectedParent::authenticate(
            stream,
            std::process::id(),
            parent,
            9,
            &"1".repeat(64),
            0,
            Instant::now() + Duration::from_secs(5),
        )
        .unwrap();
        assert!(connected.ensure_live().is_ok());
        let until = Instant::now() + Duration::from_secs(5);
        while connected.ensure_live().is_ok() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(connected.ensure_live().is_err());
    }

    #[test]
    fn direct_child_authenticates_and_unrelated_peer_is_ignored() {
        use std::{
            fs,
            os::unix::fs::DirBuilderExt,
            process::{Command, Stdio},
        };
        struct TestDirectory(std::path::PathBuf);
        impl Drop for TestDirectory {
            fn drop(&mut self) {
                let _ = fs::remove_file(self.0.join("socket"));
                let _ = fs::remove_dir(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::path::PathBuf::from(format!(
            "/tmp/rz-parent-auth-{}-{nonce}",
            std::process::id()
        ));
        fs::DirBuilder::new().mode(0o700).create(&path).unwrap();
        let temp = TestDirectory(path);
        let socket = temp.0.join("socket");
        let listener = UnixListener::bind(&socket).unwrap();
        // This connection has the supervisor's PID, not the child's PID.
        let unrelated = UnixStream::connect(&socket).unwrap();
        let module = module_path!()
            .split_once("::")
            .map(|(_, tail)| tail)
            .unwrap_or(module_path!());
        let test_name = format!("{module}::direct_child_handshake_helper");
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &test_name])
            .env("RZ_TEST_DIRECT_PARENT_SOCKET", &socket)
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let cancelled = AtomicBool::new(false);
        let channel = accept_and_authenticate_child(
            &listener,
            &mut child,
            9,
            &"1".repeat(64),
            0,
            Instant::now() + Duration::from_secs(5),
            &cancelled,
        )
        .unwrap();
        assert!(channel.matches(9, &"1".repeat(64), 0));
        assert!(channel.ensure_live(&mut child).is_ok());
        drop(channel);
        assert!(child.wait().unwrap().success());
        drop(unrelated);
    }

    #[test]
    fn ack_binds_pid_epoch_lock_device_and_scope() {
        let lock = "a".repeat(64);
        let mut ack = Ack {
            schema: ACK_SCHEMA.into(),
            child_pid: 51,
            process_epoch: 3,
            lock_sha256: lock.clone(),
            cuda_ordinal: 1,
            scope: SCOPE.into(),
        };
        assert!(validate_ack(&ack, 51, 3, &lock, 1).is_ok());
        ack.process_epoch = 4;
        assert!(validate_ack(&ack, 51, 3, &lock, 1).is_err());
        ack.process_epoch = 3;
        ack.scope = "claimed-gpu-fence".into();
        assert!(validate_ack(&ack, 51, 3, &lock, 1).is_err());
    }

    #[test]
    fn rejects_extra_unframed_liveness_data_and_closed_peer() {
        let (reader, mut writer) = UnixStream::pair().unwrap();
        assert!(require_cloexec(&reader).is_ok());
        assert!(require_live(&reader).is_ok());
        writer.write_all(b"unexpected").unwrap();
        assert!(require_live(&reader).is_err());
        drop(writer);
        let (reader, writer) = UnixStream::pair().unwrap();
        drop(writer);
        assert!(require_live(&reader).is_err());
    }

    #[test]
    fn lock_and_original_deadline_are_bounded() {
        assert!(valid_lock(&"1".repeat(64)));
        assert!(!valid_lock(&"G".repeat(64)));
        assert!(!valid_lock("abc"));
        assert!(remaining(Instant::now()).is_err());
        let cancelled = AtomicBool::new(true);
        assert_eq!(
            next_poll(Instant::now() + Duration::from_secs(1), &cancelled)
                .unwrap_err()
                .kind(),
            io::ErrorKind::Interrupted
        );
    }
}
