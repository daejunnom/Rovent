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
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

const REQUEST_SCHEMA: &str = "rz-rovent-parent-admission/1";
const ACK_SCHEMA: &str = "rz-rovent-parent-admission-ack/1";
const SCOPE: &str = "parent-child-connection-not-gpu-fence";
#[cfg(feature = "onnx-cuda")]
const IMAGE_GRANT_SCHEMA: &str = "rz-pals-system-image-grant/1";
#[cfg(feature = "onnx-cuda")]
const IMAGE_ACK_SCHEMA: &str = "rz-pals-system-image-grant-ack/1";
#[cfg(feature = "onnx-cuda")]
const IMAGE_SCOPE: &str = "same-child-image-observed-not-gpu-fence";
#[cfg(feature = "onnx-cuda")]
const CLEANUP_REQUEST_SCHEMA: &str = "rz-pals-child-cleanup-request/1";
#[cfg(feature = "onnx-cuda")]
const CLEANUP_GRANT_SCHEMA: &str = "rz-pals-child-cleanup-grant/1";
#[cfg(feature = "onnx-cuda")]
const CLEANUP_ACK_SCHEMA: &str = "rz-pals-child-cleanup-ack/1";
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

#[cfg(feature = "onnx-cuda")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageGrant {
    schema: String,
    child_pid: u32,
    process_epoch: u64,
    lock_sha256: String,
    cuda_ordinal: i32,
    scope: String,
}

#[cfg(feature = "onnx-cuda")]
#[derive(Serialize)]
struct ImageGrantAck<'a> {
    schema: &'static str,
    child_pid: u32,
    process_epoch: u64,
    lock_sha256: &'a str,
    cuda_ordinal: i32,
    scope: &'static str,
}

#[cfg(feature = "onnx-cuda")]
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CleanupFrame {
    schema: String,
    child_pid: u32,
    process_epoch: u64,
    lock_sha256: String,
    cuda_ordinal: i32,
    request_id: u64,
    remaining_ns: u64,
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
    original_until: Instant,
    original_cancelled: Option<Arc<AtomicBool>>,
}

/// Permission to attempt native entry after the direct parent confirmed the
/// same child image. This does not attest device identity, completion, future
/// parent liveness or concurrent-writer exclusion.
#[cfg(feature = "onnx-cuda")]
pub(crate) struct PalsPostImageAdmissionV6 {
    parent: Arc<ConnectedParent>,
    control: UnixStream,
    cleanup_exchange_started: bool,
    cleanup_sent: Vec<u8>,
    cleanup_received: Vec<u8>,
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

fn checked_wait(until: Instant, cancelled: Option<&AtomicBool>) -> io::Result<std::time::Duration> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "parent handshake cancelled",
        ));
    }
    let left = remaining(until)?;
    Ok(if cancelled.is_some() {
        left.min(std::time::Duration::from_millis(50))
    } else {
        left
    })
}

fn read_request_controlled(
    stream: &mut UnixStream,
    until: Instant,
    cancelled: Option<&AtomicBool>,
) -> io::Result<Request> {
    let mut frame = Vec::with_capacity(256);
    loop {
        stream.set_read_timeout(Some(checked_wait(until, cancelled)?))?;
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
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) && cancelled.is_some() =>
            {
                continue;
            }
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

#[cfg(test)]
fn read_request(stream: &mut UnixStream, until: Instant) -> io::Result<Request> {
    read_request_controlled(stream, until, None)
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

#[cfg(feature = "onnx-cuda")]
fn read_image_grant(
    stream: &mut UnixStream,
    until: Instant,
    cancelled: &AtomicBool,
) -> io::Result<ImageGrant> {
    let mut frame = Vec::with_capacity(256);
    loop {
        stream.set_read_timeout(Some(checked_wait(until, Some(cancelled))?))?;
        let mut byte = [0u8; 1];
        match stream.read(&mut byte) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "image grant channel closed",
                ));
            }
            Ok(_) if byte[0] == b'\n' => break,
            Ok(_) if frame.len() < FRAME_LIMIT - 1 => frame.push(byte[0]),
            Ok(_) => return Err(invalid("image grant frame exceeds bound")),
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
    serde_json::from_slice(&frame).map_err(|_| invalid("invalid image grant frame"))
}

#[cfg(feature = "onnx-cuda")]
fn read_cleanup_frame(
    stream: &mut UnixStream,
    until: Instant,
    cancelled: &AtomicBool,
    received: &mut Vec<u8>,
) -> io::Result<CleanupFrame> {
    let mut frame = Vec::with_capacity(256);
    loop {
        stream.set_read_timeout(Some(checked_wait(until, Some(cancelled))?))?;
        let mut byte = [0u8; 1];
        match stream.read(&mut byte) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "cleanup deadline channel closed",
                ));
            }
            Ok(_) if byte[0] == b'\n' => {
                received.push(byte[0]);
                break;
            }
            Ok(_) if frame.len() < FRAME_LIMIT - 1 => {
                frame.push(byte[0]);
                received.push(byte[0]);
            }
            Ok(_) => {
                received.push(byte[0]);
                return Err(invalid("cleanup deadline frame exceeds bound"));
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::Interrupted
                        | io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                ) =>
            {
                continue;
            }
            Err(error) => return Err(error),
        }
    }
    serde_json::from_slice(&frame).map_err(|_| invalid("invalid cleanup deadline frame"))
}

#[cfg(feature = "onnx-cuda")]
fn conservative_cleanup_deadline(
    requested_at: Instant,
    remaining_ns: u64,
    local_cap: Instant,
    now: Instant,
) -> io::Result<Instant> {
    if remaining_ns == 0 {
        return Err(invalid("cleanup deadline grant has no remaining time"));
    }
    let deadline = requested_at
        .checked_add(std::time::Duration::from_nanos(remaining_ns))
        .ok_or_else(|| invalid("cleanup deadline overflow"))?
        .min(local_cap);
    if now >= deadline {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "cleanup deadline expired",
        ));
    }
    Ok(deadline)
}

#[cfg(feature = "onnx-cuda")]
fn write_cleanup_frame(
    stream: &mut UnixStream,
    bytes: &[u8],
    until: Instant,
    cancelled: &AtomicBool,
    sent: &mut Vec<u8>,
) -> io::Result<()> {
    if bytes.len() >= FRAME_LIMIT {
        return Err(invalid("cleanup deadline frame exceeds bound"));
    }
    let mut frame = bytes.to_vec();
    frame.push(b'\n');
    let mut offset = 0;
    while offset < frame.len() {
        stream.set_write_timeout(Some(checked_wait(until, Some(cancelled))?))?;
        match stream.write(&frame[offset..]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "cleanup deadline frame not written",
                ));
            }
            Ok(count) => {
                sent.extend_from_slice(&frame[offset..offset + count]);
                offset += count;
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::Interrupted
                        | io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                ) =>
            {
                continue;
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(feature = "onnx-cuda")]
fn write_image_ack(
    stream: &mut UnixStream,
    bytes: &[u8],
    until: Instant,
    cancelled: &AtomicBool,
) -> io::Result<()> {
    if bytes.len() >= FRAME_LIMIT {
        return Err(invalid("image grant ack exceeds bound"));
    }
    let mut frame = Vec::with_capacity(bytes.len() + 1);
    frame.extend_from_slice(bytes);
    frame.push(b'\n');
    let mut offset = 0;
    while offset < frame.len() {
        stream.set_write_timeout(Some(checked_wait(until, Some(cancelled))?))?;
        match stream.write(&frame[offset..]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "image grant ack not written",
                ));
            }
            Ok(n) => offset += n,
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

#[cfg(feature = "onnx-cuda")]
fn validate_image_grant(
    grant: &ImageGrant,
    child_pid: u32,
    epoch: u64,
    lock: &str,
    ordinal: i32,
) -> io::Result<()> {
    if grant.schema != IMAGE_GRANT_SCHEMA
        || grant.child_pid != child_pid
        || grant.process_epoch != epoch
        || grant.lock_sha256 != lock
        || grant.cuda_ordinal != ordinal
        || grant.scope != IMAGE_SCOPE
    {
        return Err(invalid(
            "post-image grant differs from original parent scope",
        ));
    }
    Ok(())
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
            None,
        )
    }

    /// As `authenticate`, while preserving the original cancellation token
    /// during all bounded handshake reads and writes.
    pub(crate) fn authenticate_with_cancel(
        stream: UnixStream,
        child_pid: u32,
        original_parent_pid: i32,
        epoch: u64,
        lock_sha256: &str,
        cuda_ordinal: i32,
        until: Instant,
        cancelled: Arc<AtomicBool>,
    ) -> io::Result<Self> {
        let mut parent = Self::authenticate_inner(
            stream,
            child_pid,
            original_parent_pid,
            epoch,
            lock_sha256,
            cuda_ordinal,
            until,
            false,
            Some(cancelled.as_ref()),
        )?;
        parent.original_cancelled = Some(cancelled);
        Ok(parent)
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
        cancelled: Option<&AtomicBool>,
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
        let request = read_request_controlled(&mut stream, until, cancelled)?;
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
            stream.set_write_timeout(Some(checked_wait(until, cancelled)?))?;
            match stream.write(&bytes[written..]) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "parent acknowledgement not written",
                    ));
                }
                Ok(count) => written += count,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                    ) && cancelled.is_some() =>
                {
                    continue;
                }
                Err(error) => return Err(error),
            }
        }
        checked_wait(until, cancelled)?;
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
        checked_wait(until, cancelled)?;
        Ok(Self {
            stream,
            parent_pid,
            parent_uid,
            process_epoch: epoch,
            lock_sha256: lock_sha256.to_owned(),
            cuda_ordinal,
            original_until: until,
            original_cancelled: None,
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
            None,
        )?;
        self.ensure_live()?;
        Ok(control.stream)
    }

    /// The cleanup caller owns a separate cancellation token and its original
    /// cleanup deadline; neither the startup token nor its expired window is
    /// substituted for this handshake. The stream still grants no fence.
    #[cfg(feature = "onnx-cuda")]
    pub(crate) fn authenticate_separate_control_with_cancel(
        &self,
        stream: UnixStream,
        until: Instant,
        cancelled: &AtomicBool,
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
            Some(cancelled),
        )?;
        self.ensure_live()?;
        checked_wait(until, Some(cancelled))?;
        Ok(control.stream)
    }

    /// A distinct control stream is authenticated to the same direct parent
    /// and V6 identity before receiving a bounded post-image grant. The grant
    /// only authorizes a later native-entry attempt; it is not a GPU fence.
    #[cfg(feature = "onnx-cuda")]
    pub(crate) fn await_pals_post_image_grant(
        self: &Arc<Self>,
        stream: UnixStream,
        until: Instant,
        cancelled: &Arc<AtomicBool>,
    ) -> io::Result<PalsPostImageAdmissionV6> {
        let original = self
            .original_cancelled
            .as_ref()
            .ok_or_else(|| invalid("original parent cancellation owner unavailable"))?;
        if !Arc::ptr_eq(original, cancelled) {
            return Err(invalid(
                "post-image cancellation owner differs from original",
            ));
        }
        let until = until.min(self.original_until);
        self.ensure_live()?;
        let connected = Self::authenticate_inner(
            stream,
            getpid().as_raw() as u32,
            self.parent_pid,
            self.process_epoch,
            &self.lock_sha256,
            self.cuda_ordinal,
            until,
            true,
            Some(cancelled.as_ref()),
        )?;
        let mut control = connected.stream;
        let grant = read_image_grant(&mut control, until, cancelled)?;
        validate_image_grant(
            &grant,
            getpid().as_raw() as u32,
            self.process_epoch,
            &self.lock_sha256,
            self.cuda_ordinal,
        )?;
        checked_wait(until, Some(cancelled))?;
        self.ensure_live()?;
        validate_peer(&control, self.parent_pid, self.parent_uid)?;
        let ack = ImageGrantAck {
            schema: IMAGE_ACK_SCHEMA,
            child_pid: grant.child_pid,
            process_epoch: self.process_epoch,
            lock_sha256: &self.lock_sha256,
            cuda_ordinal: self.cuda_ordinal,
            scope: IMAGE_SCOPE,
        };
        let bytes =
            serde_json::to_vec(&ack).map_err(|_| invalid("cannot encode image grant ack"))?;
        write_image_ack(&mut control, &bytes, until, cancelled)?;
        checked_wait(until, Some(cancelled))?;
        self.ensure_live()?;
        validate_peer(&control, self.parent_pid, self.parent_uid)?;
        control.set_read_timeout(None)?;
        control.set_write_timeout(None)?;
        channel_is_live(&control)?;
        Ok(PalsPostImageAdmissionV6 {
            parent: Arc::clone(self),
            control,
            cleanup_exchange_started: false,
            cleanup_sent: Vec::new(),
            cleanup_received: Vec::new(),
        })
    }
}

#[cfg(feature = "onnx-cuda")]
impl PalsPostImageAdmissionV6 {
    /// Transport progress only; an incomplete write can have reached the peer.
    pub(crate) fn cleanup_exchange_progress(&self) -> (&[u8], &[u8]) {
        (&self.cleanup_sent, &self.cleanup_received)
    }
    /// Derive a conservative local deadline from the original direct parent.
    /// This exchange is optional and single-use; a failed exchange leaves the
    /// same stream in this owner, with delivery potentially unknown.
    pub(crate) fn receive_cleanup_deadline_once(
        &mut self,
        io_until: Instant,
        local_cap: Instant,
        cancelled: &Arc<AtomicBool>,
    ) -> io::Result<Instant> {
        if self.cleanup_exchange_started {
            return Err(invalid("cleanup deadline exchange already attempted"));
        }
        self.cleanup_exchange_started = true;
        let original = self
            .parent
            .original_cancelled
            .as_ref()
            .ok_or_else(|| invalid("original parent cancellation owner unavailable"))?;
        if !Arc::ptr_eq(original, cancelled) {
            return Err(invalid(
                "cleanup deadline cancellation owner differs from original",
            ));
        }
        let io_until = io_until.min(self.parent.original_until);
        self.parent.ensure_live()?;
        validate_peer(
            &self.control,
            self.parent.parent_pid,
            self.parent.parent_uid,
        )?;
        let request = CleanupFrame {
            schema: CLEANUP_REQUEST_SCHEMA.into(),
            child_pid: getpid().as_raw() as u32,
            process_epoch: self.parent.process_epoch,
            lock_sha256: self.parent.lock_sha256.clone(),
            cuda_ordinal: self.parent.cuda_ordinal,
            request_id: 1,
            remaining_ns: 0,
        };
        let bytes = serde_json::to_vec(&request)
            .map_err(|_| invalid("cannot encode cleanup deadline request"))?;
        let requested_at = Instant::now();
        write_cleanup_frame(
            &mut self.control,
            &bytes,
            io_until,
            cancelled,
            &mut self.cleanup_sent,
        )?;
        let grant = read_cleanup_frame(
            &mut self.control,
            io_until,
            cancelled,
            &mut self.cleanup_received,
        )?;
        if grant.schema != CLEANUP_GRANT_SCHEMA
            || grant.child_pid != request.child_pid
            || grant.process_epoch != request.process_epoch
            || grant.lock_sha256 != request.lock_sha256
            || grant.cuda_ordinal != request.cuda_ordinal
            || grant.request_id != request.request_id
            || grant.remaining_ns == 0
        {
            return Err(invalid(
                "cleanup deadline grant differs from original request",
            ));
        }
        let deadline = conservative_cleanup_deadline(
            requested_at,
            grant.remaining_ns,
            local_cap,
            Instant::now(),
        )?;
        self.parent.ensure_live()?;
        validate_peer(
            &self.control,
            self.parent.parent_pid,
            self.parent.parent_uid,
        )?;
        let ack = CleanupFrame {
            schema: CLEANUP_ACK_SCHEMA.into(),
            ..grant
        };
        let bytes = serde_json::to_vec(&ack)
            .map_err(|_| invalid("cannot encode cleanup deadline acknowledgement"))?;
        write_cleanup_frame(
            &mut self.control,
            &bytes,
            io_until,
            cancelled,
            &mut self.cleanup_sent,
        )?;
        checked_wait(io_until, Some(cancelled))?;
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "cleanup deadline expired",
            ));
        }
        self.parent.ensure_live()?;
        validate_peer(
            &self.control,
            self.parent.parent_pid,
            self.parent.parent_uid,
        )?;
        self.control.set_read_timeout(None)?;
        self.control.set_write_timeout(None)?;
        control_channel_is_open(&self.control)?;
        Ok(deadline)
    }

    /// Compare the original in-process owner, not only its claimed PID or
    /// launch values. This observation grants no new admission or fence.
    pub(crate) fn belongs_to_parent(&self, parent: &Arc<ConnectedParent>) -> bool {
        Arc::ptr_eq(&self.parent, parent)
    }

    /// Recheck this exact parent and dedicated control stream immediately
    /// before each native entry using the original deadline/cancel token.
    pub(crate) fn ensure_permitted(
        &self,
        parent: &Arc<ConnectedParent>,
        until: Instant,
        cancelled: &Arc<AtomicBool>,
    ) -> io::Result<()> {
        let original = self
            .parent
            .original_cancelled
            .as_ref()
            .ok_or_else(|| invalid("original parent cancellation owner unavailable"))?;
        if !Arc::ptr_eq(original, cancelled) {
            return Err(invalid(
                "post-image cancellation owner differs from original",
            ));
        }
        let until = until.min(self.parent.original_until);
        checked_wait(until, Some(cancelled))?;
        if !Arc::ptr_eq(&self.parent, parent) {
            return Err(invalid("post-image admission belongs to another parent"));
        }
        self.parent.ensure_live()?;
        validate_peer(
            &self.control,
            self.parent.parent_pid,
            self.parent.parent_uid,
        )?;
        channel_is_live(&self.control)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Write, os::unix::net::UnixStream, time::Duration};

    #[cfg(feature = "onnx-cuda")]
    #[test]
    fn cleanup_deadline_is_anchored_before_request_and_respects_local_cap() {
        let start = Instant::now();
        let local_cap = start + Duration::from_secs(8);
        let now = start + Duration::from_secs(3);
        assert_eq!(
            conservative_cleanup_deadline(start, 10_000_000_000, local_cap, now).unwrap(),
            local_cap
        );
        assert_eq!(
            conservative_cleanup_deadline(start, 5_000_000_000, local_cap, now).unwrap(),
            start + Duration::from_secs(5)
        );
        assert_eq!(
            conservative_cleanup_deadline(start, 3_000_000_000, local_cap, now)
                .unwrap_err()
                .kind(),
            io::ErrorKind::TimedOut
        );
        assert!(conservative_cleanup_deadline(start, 0, local_cap, now).is_err());
    }

    #[cfg(feature = "onnx-cuda")]
    #[test]
    fn cleanup_deadline_partial_frame_is_retained_on_eof() {
        let (mut reader, mut writer) = UnixStream::pair().unwrap();
        let partial = b"{\"schema\":\"incomplete";
        writer.write_all(partial).unwrap();
        drop(writer);
        let mut received = Vec::new();
        let error = match read_cleanup_frame(
            &mut reader,
            Instant::now() + Duration::from_secs(1),
            &AtomicBool::new(false),
            &mut received,
        ) {
            Ok(_) => panic!("incomplete frame was accepted"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        assert_eq!(received, partial);
    }

    #[cfg(feature = "onnx-cuda")]
    #[test]
    fn post_image_grant_is_bound_to_all_original_identity_fields() {
        let grant = ImageGrant {
            schema: IMAGE_GRANT_SCHEMA.into(),
            child_pid: 7,
            process_epoch: 8,
            lock_sha256: "lock".into(),
            cuda_ordinal: 1,
            scope: IMAGE_SCOPE.into(),
        };
        assert!(validate_image_grant(&grant, 7, 8, "lock", 1).is_ok());
        assert!(validate_image_grant(&grant, 6, 8, "lock", 1).is_err());
        assert!(validate_image_grant(&grant, 7, 9, "lock", 1).is_err());
        assert!(validate_image_grant(&grant, 7, 8, "different", 1).is_err());
        assert!(validate_image_grant(&grant, 7, 8, "lock", 0).is_err());
    }

    #[cfg(feature = "onnx-cuda")]
    #[test]
    fn post_image_grant_read_obeys_original_cancellation() {
        let (mut server, _client) = UnixStream::pair().unwrap();
        let cancelled = AtomicBool::new(true);
        let error = match read_image_grant(
            &mut server,
            Instant::now() + Duration::from_secs(1),
            &cancelled,
        ) {
            Ok(_) => panic!("cancelled grant must not be accepted"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
    }

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
