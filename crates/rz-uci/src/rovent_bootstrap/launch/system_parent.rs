//! A live parent/child channel for one original system-CUDA launch.
//!
//! This proves only the connected parent identity and submitted launch values.
//! It does not prove a device UUID, exclusive device use, physical completion,
//! or that an external process will survive termination by the OS.

use nix::{
    errno::Errno,
    sys::socket::{MsgFlags, getsockopt, recv, sockopt::PeerCredentials},
    unistd::{geteuid, getpid, getppid},
};
use serde::{Deserialize, Serialize};
use std::{
    io::{self, Read, Write},
    os::{fd::AsRawFd, unix::net::UnixStream},
    time::Instant,
};

const REQUEST_SCHEMA: &str = "rz-rovent-parent-admission/1";
const ACK_SCHEMA: &str = "rz-rovent-parent-admission-ack/1";
const SCOPE: &str = "parent-child-connection-not-gpu-fence";
const FRAME_LIMIT: usize = 4096;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    child_pid: u32,
    process_epoch: u64,
    lock_sha256: String,
    cuda_ordinal: i32,
}

#[derive(Serialize)]
struct Ack<'a> {
    schema: &'static str,
    child_pid: u32,
    process_epoch: u64,
    lock_sha256: &'a str,
    cuda_ordinal: i32,
    scope: &'static str,
}

/// Original socket and context. Only the parent launch module can construct it.
/// An `Arc<ConnectedParent>` can retain this very stream in the launch scope.
pub(crate) struct ConnectedParent {
    stream: UnixStream,
    parent_pid: i32,
    parent_uid: u32,
    process_epoch: u64,
    lock_sha256: String,
    cuda_ordinal: i32,
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn remaining(until: Instant) -> io::Result<std::time::Duration> {
    until
        .checked_duration_since(Instant::now())
        .filter(|value| !value.is_zero())
        .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "parent handshake deadline expired"))
}

fn peer_identity(stream: &UnixStream) -> io::Result<(i32, u32)> {
    let peer = getsockopt(stream, PeerCredentials).map_err(io::Error::other)?;
    Ok((peer.pid(), peer.uid()))
}

fn validate_peer(stream: &UnixStream, parent_pid: i32, parent_uid: u32) -> io::Result<()> {
    if getppid().as_raw() != parent_pid
        || geteuid().as_raw() != parent_uid
        || peer_identity(stream)? != (parent_pid, parent_uid)
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "original parent peer differs",
        ));
    }
    Ok(())
}

fn read_request(stream: &mut UnixStream, until: Instant) -> io::Result<Request> {
    let mut frame = Vec::with_capacity(256);
    loop {
        stream.set_read_timeout(Some(remaining(until)?))?;
        let mut byte = [0u8; 1];
        match stream.read(&mut byte) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "parent channel closed",
                ));
            }
            Ok(_) if byte[0] == b'\n' => break,
            Ok(_) if frame.len() < FRAME_LIMIT - 1 => frame.push(byte[0]),
            Ok(_) => return Err(invalid("parent admission frame exceeds bound")),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "parent admission read timed out",
                ));
            }
            Err(error) => return Err(error),
        }
    }
    serde_json::from_slice(&frame).map_err(|_| invalid("invalid parent admission frame"))
}

fn validate_request(
    request: &Request,
    child_pid: u32,
    epoch: u64,
    lock: &str,
    ordinal: i32,
) -> io::Result<()> {
    if request.schema != REQUEST_SCHEMA
        || request.child_pid != child_pid
        || request.process_epoch != epoch
        || request.lock_sha256 != lock
        || request.cuda_ordinal != ordinal
    {
        return Err(invalid("parent admission differs from original launch"));
    }
    Ok(())
}

fn channel_is_live(stream: &UnixStream) -> io::Result<()> {
    let mut probe = [0u8; 1];
    match recv(
        stream.as_raw_fd(),
        &mut probe,
        MsgFlags::MSG_PEEK | MsgFlags::MSG_DONTWAIT,
    ) {
        Ok(0) => Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "original parent disconnected",
        )),
        Ok(_) => Err(invalid("unexpected data on parent admission channel")),
        Err(Errno::EAGAIN) => Ok(()),
        Err(error) => Err(io::Error::from_raw_os_error(error as i32)),
    }
}

// After a control acknowledgement the parent may immediately queue its
// first command. This checks only for EOF, unlike the liveness-only stream.
#[cfg(feature = "onnx-cuda")]
fn control_channel_is_open(stream: &UnixStream) -> io::Result<()> {
    let mut probe = [0u8; 1];
    match recv(
        stream.as_raw_fd(),
        &mut probe,
        MsgFlags::MSG_PEEK | MsgFlags::MSG_DONTWAIT,
    ) {
        Ok(0) => Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "original parent control channel disconnected",
        )),
        Ok(_) | Err(Errno::EAGAIN) => Ok(()),
        Err(error) => Err(io::Error::from_raw_os_error(error as i32)),
    }
}

impl ConnectedParent {
    /// Capture before publishing the endpoint; later reparenting is rejected.
    pub(crate) fn capture_original_parent() -> io::Result<i32> {
        let pid = getppid().as_raw();
        if pid <= 0 {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "original parent pid unavailable",
            ));
        }
        Ok(pid)
    }

    /// Authenticate the accepted stream against the actual parent PID/UID and
    /// the original launch values. The original deadline is never renewed.
    pub(crate) fn authenticate(
        stream: UnixStream,
        child_pid: u32,
        original_parent_pid: i32,
        epoch: u64,
        lock_sha256: &str,
        cuda_ordinal: i32,
        until: Instant,
    ) -> io::Result<Self> {
        Self::authenticate_inner(
            stream,
            child_pid,
            original_parent_pid,
            epoch,
            lock_sha256,
            cuda_ordinal,
            until,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn authenticate_inner(
        mut stream: UnixStream,
        child_pid: u32,
        original_parent_pid: i32,
        epoch: u64,
        lock_sha256: &str,
        cuda_ordinal: i32,
        until: Instant,
        control: bool,
    ) -> io::Result<Self> {
        if child_pid != getpid().as_raw() as u32 {
            return Err(invalid("child pid differs from the running process"));
        }
        // The caller captures this PID before publishing the endpoint. Taking
        // getppid only after accept could accept a reparented process.
        let parent_pid = original_parent_pid;
        let parent_uid = geteuid().as_raw();
        validate_peer(&stream, parent_pid, parent_uid)?;
        // The accepted original stream is not shared before admission. Use
        // kernel timeouts consistently even if its listener was nonblocking.
        stream.set_nonblocking(false)?;
        let request = read_request(&mut stream, until)?;
        validate_request(&request, child_pid, epoch, lock_sha256, cuda_ordinal)?;
        validate_peer(&stream, parent_pid, parent_uid)?;
        channel_is_live(&stream)?;
        let ack = Ack {
            schema: ACK_SCHEMA,
            child_pid,
            process_epoch: epoch,
            lock_sha256,
            cuda_ordinal,
            scope: SCOPE,
        };
        let mut bytes = serde_json::to_vec(&ack)
            .map_err(|_| invalid("cannot serialize parent acknowledgement"))?;
        if bytes.len() >= FRAME_LIMIT {
            return Err(invalid("parent acknowledgement exceeds frame bound"));
        }
        bytes.push(b'\n');
        let mut written = 0;
        while written < bytes.len() {
            stream.set_write_timeout(Some(remaining(until)?))?;
            match stream.write(&bytes[written..]) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "parent acknowledgement not written",
                    ));
                }
                Ok(count) => written += count,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
        remaining(until)?;
        stream.set_read_timeout(None)?;
        stream.set_write_timeout(None)?;
        validate_peer(&stream, parent_pid, parent_uid)?;
        if control {
            #[cfg(feature = "onnx-cuda")]
            control_channel_is_open(&stream)?;
            #[cfg(not(feature = "onnx-cuda"))]
            return Err(invalid("separate control requires CUDA support"));
        } else {
            channel_is_live(&stream)?;
        }
        remaining(until)?;
        Ok(Self {
            stream,
            parent_pid,
            parent_uid,
            process_epoch: epoch,
            lock_sha256: lock_sha256.to_owned(),
            cuda_ordinal,
        })
    }

    /// Recheck the same socket immediately before native entry. A later parent
    /// failure is not a GPU fence and must follow normal original-owner drain.
    pub(crate) fn ensure_live(&self) -> io::Result<()> {
        validate_peer(&self.stream, self.parent_pid, self.parent_uid)?;
        channel_is_live(&self.stream)
    }

    pub(crate) fn matches(&self, epoch: u64, lock_sha256: &str, cuda_ordinal: i32) -> bool {
        self.process_epoch == epoch
            && self.lock_sha256 == lock_sha256
            && self.cuda_ordinal == cuda_ordinal
    }

    /// Authenticate a separate, same-parent control connection against this
    /// original admission. The admission socket remains liveness-only; control
    /// bytes must never be multiplexed onto it. Returning this stream grants no
    /// cleanup or physical-completion authority and does not implement a
    /// control protocol. The caller must retain the original parent owner.
    #[cfg(feature = "onnx-cuda")]
    pub(crate) fn authenticate_separate_control(
        &self,
        stream: UnixStream,
        until: Instant,
    ) -> io::Result<UnixStream> {
        self.ensure_live()?;
        let control = Self::authenticate_inner(
            stream,
            getpid().as_raw() as u32,
            self.parent_pid,
            self.process_epoch,
            &self.lock_sha256,
            self.cuda_ordinal,
            until,
            true,
        )?;
        self.ensure_live()?;
        Ok(control.stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Write, os::unix::net::UnixStream, time::Duration};

    #[test]
    fn request_binds_all_original_fields() {
        let request = Request {
            schema: REQUEST_SCHEMA.into(),
            child_pid: 7,
            process_epoch: 8,
            lock_sha256: "lock".into(),
            cuda_ordinal: 0,
        };
        assert!(validate_request(&request, 7, 8, "lock", 0).is_ok());
        assert!(validate_request(&request, 6, 8, "lock", 0).is_err());
        assert!(validate_request(&request, 7, 9, "lock", 0).is_err());
        assert!(validate_request(&request, 7, 8, "other", 0).is_err());
        assert!(validate_request(&request, 7, 8, "lock", 1).is_err());
    }

    #[test]
    fn bounded_frames_and_eof_are_not_success() {
        let (mut server, mut client) = UnixStream::pair().unwrap();
        client.write_all(&vec![b'x'; FRAME_LIMIT]).unwrap();
        assert_eq!(
            read_request(&mut server, Instant::now() + Duration::from_secs(1))
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        let (mut server, client) = UnixStream::pair().unwrap();
        drop(client);
        assert_eq!(
            read_request(&mut server, Instant::now() + Duration::from_secs(1))
                .unwrap_err()
                .kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    fn live_channel_and_closed_channel_are_distinct() {
        let (server, client) = UnixStream::pair().unwrap();
        assert!(channel_is_live(&server).is_ok());
        drop(client);
        assert_eq!(
            channel_is_live(&server).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[cfg(feature = "onnx-cuda")]
    #[test]
    fn control_open_check_preserves_immediately_queued_command() {
        let (mut server, mut client) = UnixStream::pair().unwrap();
        client.write_all(b"status\n").unwrap();
        assert!(control_channel_is_open(&server).is_ok());
        assert_eq!(
            channel_is_live(&server).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        let mut command = [0; 7];
        server.read_exact(&mut command).unwrap();
        assert_eq!(&command, b"status\n");
        drop(client);
        assert_eq!(
            control_channel_is_open(&server).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    fn same_process_peer_is_not_the_original_parent() {
        let (server, _client) = UnixStream::pair().unwrap();
        let result = ConnectedParent::authenticate(
            server,
            getpid().as_raw() as u32,
            getppid().as_raw(),
            8,
            "lock",
            0,
            Instant::now() + Duration::from_secs(1),
        );
        assert!(matches!(result, Err(error) if error.kind() == io::ErrorKind::PermissionDenied));
    }
}
