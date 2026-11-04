//! Bounded child runtime-finish claims, separate from physical custody.
//!
//! This does not issue `PalsPhysicalReleaseV6`: the original worker's
//! pre-native recovery or session-release evidence has not yet been bound
//! across the dedicated parent/child control channel. Neither a child JSON
//! claim, an authenticated peer, nor a process exit can replace that proof.

use super::DirectPalsChildV6;
use nix::{
    errno::Errno,
    sys::socket::{MsgFlags, getsockopt, recv, sockopt::PeerCredentials},
    unistd::geteuid,
};
use rz_experiments::SystemRuntimeReceiptV6;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{
    io::{self, Read, Write},
    os::{fd::AsRawFd, unix::net::UnixStream},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

#[allow(dead_code)]
pub(crate) mod session_shutdown_report_v1;

const MAX_FINISH_FRAME: usize = 256 << 10;
const IO_QUANTUM: Duration = Duration::from_millis(50);
const RELEASE_DOMAIN: &str = "rz-pals-session-release-event/1";
const RELEASE_ACK_DOMAIN: &str = "rz-pals-session-release-ack/1";

/// A wire event is a claim until the parent ties it to its exact live child,
/// original process owner and registered executable. It is session-scoped and
/// never asserts that unrelated GPU work or the whole device is idle.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PalsSessionReleaseEventV6 {
    domain: String,
    process_id: u32,
    process_generation: u64,
    startup_lock_sha256: String,
    completed_synchronous_runs: u64,
    source: String,
    scope: String,
}

/// The exact event bytes and any partial response remain available to the
/// caller on every failed send or acknowledgement read.
#[derive(Debug)]
pub(crate) struct PalsReleaseSendFailureV6 {
    pub(crate) reason: String,
    pub(crate) stage: PalsReleaseSendFailureStageV6,
    pub(crate) io_error: Option<io::Error>,
    pub(crate) event_bytes: Vec<u8>,
    pub(crate) event_frame_bytes_written: usize,
    pub(crate) acknowledgement_bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PalsReleaseSendFailureStageV6 {
    Prepare,
    Authenticate,
    SendFrame,
    ReadAcknowledgement,
    ValidateAcknowledgement,
}

impl std::fmt::Display for PalsReleaseSendFailureV6 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PALS release acknowledgement failed: {}", self.reason)
    }
}

impl std::error::Error for PalsReleaseSendFailureV6 {}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PalsSessionReleaseAckV6 {
    domain: String,
    process_id: u32,
    process_generation: u64,
    startup_lock_sha256: String,
    event_sha256: String,
    scope: String,
}

fn release_ack(event: &PalsSessionReleaseEventV6, event_bytes: &[u8]) -> PalsSessionReleaseAckV6 {
    PalsSessionReleaseAckV6 {
        domain: RELEASE_ACK_DOMAIN.into(),
        process_id: event.process_id,
        process_generation: event.process_generation,
        startup_lock_sha256: event.startup_lock_sha256.clone(),
        event_sha256: format!("{:x}", Sha256::digest(event_bytes)),
        scope: "claim_received_not_physical_fence".into(),
    }
}

fn acknowledgement_matches(
    ack: &PalsSessionReleaseAckV6,
    event: &PalsSessionReleaseEventV6,
    event_bytes: &[u8],
) -> bool {
    let expected = release_ack(event, event_bytes);
    ack.domain == expected.domain
        && ack.process_id == expected.process_id
        && ack.process_generation == expected.process_generation
        && ack.startup_lock_sha256 == expected.startup_lock_sha256
        && ack.event_sha256 == expected.event_sha256
        && ack.scope == expected.scope
}

/// Create a wire event only from the existing producer-only process receipt.
/// The caller must take that receipt from this exact process's finish path;
/// epoch alone does not bind a receipt to a particular session owner. No
/// release is inferred from an exit code, file or JSON report.
fn event_from_original_process(
    owner: &crate::pals_adaptive_startup::PinnedPalsSystemStartupV6,
    parent: &crate::rovent_bootstrap::launch::ConnectedParent,
    process: &crate::pals_adaptive::AdaptiveSessionProcessReceipt,
) -> io::Result<PalsSessionReleaseEventV6> {
    use crate::pals_adaptive::NativeSessionReleaseSource;
    use rz_experiments::NativeRuntimeSourceV6;

    let lock = owner.lock();
    let NativeRuntimeSourceV6::SystemCuda(runtime) = &lock.config.system_runtime_lock.runtime
    else {
        return Err(io::Error::other("PALS original release is not system CUDA"));
    };
    if process.epoch().0 != lock.config.process_generation
        || !process.native_owner_joined()
        || !process.native_physical_shutdown_confirmed()
        || !parent.matches(
            lock.config.process_generation,
            &lock.canonical_sha256,
            runtime.cuda_device_id,
        )
    {
        return Err(io::Error::other(
            "PALS original owner release is not confirmed",
        ));
    }
    parent.ensure_live()?;
    let observation = process
        .native_session_release()
        .filter(|item| item.source() == NativeSessionReleaseSource::OriginalReport)
        .ok_or_else(|| io::Error::other("PALS original session release is unavailable"))?;
    Ok(PalsSessionReleaseEventV6 {
        domain: RELEASE_DOMAIN.into(),
        process_id: std::process::id(),
        process_generation: lock.config.process_generation,
        startup_lock_sha256: lock.canonical_sha256.clone(),
        completed_synchronous_runs: observation.release().completed_synchronous_runs,
        source: "original_report".into(),
        scope: "reported_session_only_not_device_fence".into(),
    })
}

/// Authenticate an independent original-parent control stream before writing
/// this one event. It never writes to the parent's liveness-only socket. A
/// matching ACK establishes only that the parent accepted this claim frame;
/// it does not establish GPU completion or a parent-owned physical release.
/// Do not retry a partial exchange on a different stream.
pub(crate) fn send_original_release_event(
    owner: &crate::pals_adaptive_startup::PinnedPalsSystemStartupV6,
    parent: &crate::rovent_bootstrap::launch::ConnectedParent,
    process: &crate::pals_adaptive::AdaptiveSessionProcessReceipt,
    control: UnixStream,
    until: Instant,
    cancelled: &AtomicBool,
) -> Result<PalsSessionReleaseAckV6, PalsReleaseSendFailureV6> {
    let empty_failure = |error: io::Error| PalsReleaseSendFailureV6 {
        reason: error.to_string(),
        stage: PalsReleaseSendFailureStageV6::Prepare,
        io_error: Some(error),
        event_bytes: Vec::new(),
        event_frame_bytes_written: 0,
        acknowledgement_bytes: Vec::new(),
    };
    if cancelled.load(Ordering::Acquire) {
        return Err(empty_failure(io::Error::new(
            io::ErrorKind::Interrupted,
            "PALS release send cancelled",
        )));
    }
    let event = event_from_original_process(owner, parent, process).map_err(empty_failure)?;
    let bytes =
        serde_json::to_vec(&event).map_err(|error| empty_failure(io::Error::other(error)))?;
    if bytes.len() >= MAX_FINISH_FRAME {
        return Err(PalsReleaseSendFailureV6 {
            reason: "PALS release event exceeds bound".into(),
            stage: PalsReleaseSendFailureStageV6::Prepare,
            io_error: None,
            event_bytes: bytes,
            event_frame_bytes_written: 0,
            acknowledgement_bytes: Vec::new(),
        });
    }
    let mut stream = parent
        .authenticate_separate_control_with_cancel(control, until, cancelled)
        .map_err(|error| PalsReleaseSendFailureV6 {
            reason: error.to_string(),
            stage: PalsReleaseSendFailureStageV6::Authenticate,
            io_error: Some(error),
            event_bytes: bytes.clone(),
            event_frame_bytes_written: 0,
            acknowledgement_bytes: Vec::new(),
        })?;
    let mut framed = bytes.clone();
    framed.push(b'\n');
    if let Err((offset, error)) = write_frame(&mut stream, &framed, until, cancelled) {
        return Err(PalsReleaseSendFailureV6 {
            reason: error.to_string(),
            stage: PalsReleaseSendFailureStageV6::SendFrame,
            io_error: Some(error),
            event_bytes: bytes,
            event_frame_bytes_written: offset,
            acknowledgement_bytes: Vec::new(),
        });
    }
    let acknowledgement_bytes = match read_frame_detailed(&mut stream, until, cancelled) {
        Ok(bytes) => bytes,
        Err((partial, error)) => {
            return Err(PalsReleaseSendFailureV6 {
                reason: error.to_string(),
                stage: PalsReleaseSendFailureStageV6::ReadAcknowledgement,
                io_error: Some(error),
                event_bytes: bytes,
                event_frame_bytes_written: framed.len(),
                acknowledgement_bytes: partial,
            });
        }
    };
    let decoded = serde_json::from_slice::<PalsSessionReleaseAckV6>(&acknowledgement_bytes);
    let valid = decoded
        .as_ref()
        .is_ok_and(|ack| acknowledgement_matches(ack, &event, &bytes));
    let timely = !cancelled.load(Ordering::Acquire) && Instant::now() < until;
    match decoded {
        Ok(ack) if timely && valid => Ok(ack),
        decoded => Err(PalsReleaseSendFailureV6 {
            reason: match (timely, decoded) {
                (false, _) => {
                    "PALS release acknowledgement observed after deadline or cancellation".into()
                }
                (true, Ok(_)) => {
                    "PALS release acknowledgement differs from the original claim".into()
                }
                (true, Err(error)) => format!("invalid PALS release acknowledgement: {error}"),
            },
            stage: PalsReleaseSendFailureStageV6::ValidateAcknowledgement,
            io_error: None,
            event_bytes: bytes,
            event_frame_bytes_written: framed.len(),
            acknowledgement_bytes,
        }),
    }
}

/// Even a well-formed `Confirmed` field is a claim from the child, not an
/// independently validated same-worker session release or a GPU-wide fence.
#[derive(Debug)]
pub(super) enum PalsRuntimeFinishClaimV6 {
    Reported {
        bytes: Vec<u8>,
        receipt: SystemRuntimeReceiptV6,
    },
    Invalid {
        bytes: Vec<u8>,
        reason: String,
    },
    Incomplete {
        bytes: Vec<u8>,
        reason: String,
    },
}

#[derive(Debug)]
pub(super) enum PalsSessionReleaseClaimV6 {
    Reported {
        bytes: Vec<u8>,
        event: PalsSessionReleaseEventV6,
        acknowledgement: Result<(), PalsReleaseAckWriteFailureV6>,
    },
    Invalid {
        bytes: Vec<u8>,
        reason: String,
    },
    Incomplete {
        bytes: Vec<u8>,
        reason: String,
    },
}

/// A valid claim is still preserved when acknowledgement delivery fails.
#[derive(Debug)]
pub(super) struct PalsReleaseAckWriteFailureV6 {
    pub(super) bytes: Vec<u8>,
    pub(super) bytes_written: usize,
    pub(super) reason: String,
    pub(super) io_error: Option<io::Error>,
}

fn event_matches(event: &PalsSessionReleaseEventV6, pid: u32, epoch: u64, lock: &str) -> bool {
    event.domain == RELEASE_DOMAIN
        && event.process_id == pid
        && event.process_generation == epoch
        && event.startup_lock_sha256 == lock
        && event.source == "original_report"
        && event.scope == "reported_session_only_not_device_fence"
}

fn require_original_child_peer(
    owner: &mut DirectPalsChildV6,
    stream: &UnixStream,
) -> io::Result<()> {
    let child_pid = owner.child.id();
    if child_pid == 0 || owner.child.try_wait()?.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "original PALS child is not live",
        ));
    }
    let peer = getsockopt(stream, PeerCredentials).map_err(io::Error::other)?;
    if peer.pid() <= 0 || peer.pid() as u32 != child_pid || peer.uid() != geteuid().as_raw() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "PALS finish peer differs from original child",
        ));
    }
    Ok(())
}

/// Send an already framed message without renewing the supplied deadline.
/// Partial writes and timeout-restoration failures retain their exact offset.
fn write_frame(
    stream: &mut UnixStream,
    framed: &[u8],
    until: Instant,
    cancelled: &AtomicBool,
) -> Result<(), (usize, io::Error)> {
    let prior = stream.write_timeout().map_err(|error| (0, error))?;
    let mut offset = 0;
    let result = loop {
        if cancelled.load(Ordering::Acquire) {
            break Err((
                offset,
                io::Error::new(io::ErrorKind::Interrupted, "PALS finish write cancelled"),
            ));
        }
        let Some(remaining) = until
            .checked_duration_since(Instant::now())
            .filter(|time| !time.is_zero())
        else {
            break Err((
                offset,
                io::Error::new(
                    io::ErrorKind::TimedOut,
                    "PALS finish write deadline expired",
                ),
            ));
        };
        if let Err(error) = stream.set_write_timeout(Some(remaining.min(IO_QUANTUM))) {
            break Err((offset, error));
        }
        match stream.write(&framed[offset..]) {
            Ok(0) => {
                break Err((
                    offset,
                    io::Error::new(io::ErrorKind::WriteZero, "PALS finish write returned zero"),
                ));
            }
            Ok(count) => {
                offset += count;
                if offset == framed.len() {
                    if cancelled.load(Ordering::Acquire) {
                        break Err((
                            offset,
                            io::Error::new(
                                io::ErrorKind::Interrupted,
                                "PALS finish write completed after cancellation",
                            ),
                        ));
                    }
                    if Instant::now() >= until {
                        break Err((
                            offset,
                            io::Error::new(
                                io::ErrorKind::TimedOut,
                                "PALS finish write completed after deadline",
                            ),
                        ));
                    }
                    break Ok(());
                }
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
            Err(error) => break Err((offset, error)),
        }
    };
    match stream.set_write_timeout(prior) {
        Ok(()) => result,
        Err(error) => match result {
            Ok(()) => Err((offset, error)),
            Err((written, original)) => Err((
                written,
                io::Error::new(
                    original.kind(),
                    format!("{original}; PALS finish write timeout restore failed: {error}"),
                ),
            )),
        },
    }
}

/// Read exactly one bounded newline-delimited frame. The stream must be the
/// original supervisor-owned, dedicated control connection, not the separate
/// liveness channel or a shared file descriptor. Timeout and cancellation
/// leave the original child and its physical state unresolved. The caller
/// must supply the original registered finish deadline, not a renewed one.
fn read_frame(
    stream: &mut UnixStream,
    until: Instant,
    cancelled: &AtomicBool,
) -> Result<Vec<u8>, (Vec<u8>, String)> {
    read_frame_detailed(stream, until, cancelled)
        .map_err(|(bytes, error)| (bytes, error.to_string()))
}

fn read_frame_detailed(
    stream: &mut UnixStream,
    until: Instant,
    cancelled: &AtomicBool,
) -> Result<Vec<u8>, (Vec<u8>, io::Error)> {
    let prior = stream.read_timeout().map_err(|error| (Vec::new(), error))?;
    let result = read_frame_inner(stream, until, cancelled);
    match stream.set_read_timeout(prior) {
        Ok(()) => result,
        Err(error) => match result {
            Ok(bytes) => Err((bytes, error)),
            Err((bytes, original)) => Err((
                bytes,
                io::Error::new(
                    original.kind(),
                    format!("{original}; timeout restore failed: {error}"),
                ),
            )),
        },
    }
}

fn read_frame_inner(
    stream: &mut UnixStream,
    until: Instant,
    cancelled: &AtomicBool,
) -> Result<Vec<u8>, (Vec<u8>, io::Error)> {
    let mut result = Vec::new();
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err((
                result,
                io::Error::new(
                    io::ErrorKind::Interrupted,
                    "PALS finish observation cancelled",
                ),
            ));
        }
        let Some(remaining) = until
            .checked_duration_since(Instant::now())
            .filter(|time| !time.is_zero())
        else {
            return Err((
                result,
                io::Error::new(
                    io::ErrorKind::TimedOut,
                    "PALS finish observation deadline expired",
                ),
            ));
        };
        if let Err(error) = stream.set_read_timeout(Some(remaining.min(IO_QUANTUM))) {
            return Err((result, error));
        }
        // Peek only as far as the next delimiter, then consume no bytes from
        // a later frame. The owner guarantees a single control-stream reader.
        let mut buffer = [0u8; 4096];
        let limit = buffer.len().min(MAX_FINISH_FRAME + 1 - result.len());
        let count = match recv(stream.as_raw_fd(), &mut buffer[..limit], MsgFlags::MSG_PEEK) {
            Ok(0) => {
                return Err((
                    result,
                    io::Error::new(io::ErrorKind::UnexpectedEof, "PALS finish frame incomplete"),
                ));
            }
            Ok(count) => count,
            Err(Errno::EINTR | Errno::EAGAIN) => continue,
            Err(error) => return Err((result, io::Error::from_raw_os_error(error as i32))),
        };
        let take = buffer[..count]
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(count, |index| index + 1);
        match stream.read(&mut buffer[..take]) {
            Ok(0) => {
                return Err((
                    result,
                    io::Error::new(io::ErrorKind::UnexpectedEof, "PALS finish frame incomplete"),
                ));
            }
            Ok(count) => {
                let complete = buffer[count - 1] == b'\n';
                result.extend_from_slice(&buffer[..count - usize::from(complete)]);
                if result.len() > MAX_FINISH_FRAME {
                    return Err((
                        result,
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "PALS finish frame exceeds bound",
                        ),
                    ));
                }
                if complete {
                    if cancelled.load(Ordering::Acquire) {
                        return Err((
                            result,
                            io::Error::new(
                                io::ErrorKind::Interrupted,
                                "PALS finish frame completed after cancellation",
                            ),
                        ));
                    }
                    if Instant::now() >= until {
                        return Err((
                            result,
                            io::Error::new(
                                io::ErrorKind::TimedOut,
                                "PALS finish frame completed after deadline",
                            ),
                        ));
                    }
                    return Ok(result);
                }
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
            Err(error) => return Err((result, error)),
        }
    }
}

/// Observe one report from the same currently-live direct child and preserve
/// its exact bytes, including malformed reports. This function never releases
/// a child, unlinks a socket, signals a process, or issues a physical proof.
pub(super) fn observe_finish_claim(
    owner: &mut DirectPalsChildV6,
    stream: &mut UnixStream,
    until: Instant,
    cancelled: &AtomicBool,
) -> io::Result<PalsRuntimeFinishClaimV6> {
    require_original_child_peer(owner, stream)?;
    let bytes = match read_frame(stream, until, cancelled) {
        Ok(bytes) => bytes,
        Err((bytes, reason)) => return Ok(PalsRuntimeFinishClaimV6::Incomplete { bytes, reason }),
    };
    if let Err(error) = require_original_child_peer(owner, stream) {
        return Ok(PalsRuntimeFinishClaimV6::Incomplete {
            bytes,
            reason: error.to_string(),
        });
    }
    let lock = &owner.lock.config.system_runtime_lock;
    let decoded = std::str::from_utf8(&bytes)
        .map_err(|error| error.to_string())
        .and_then(|text| {
            SystemRuntimeReceiptV6::from_json(
                text,
                lock,
                owner.child.id(),
                owner.lock.config.process_generation,
            )
            .map_err(|error| error.to_string())
        });
    Ok(match decoded {
        Ok(receipt) => PalsRuntimeFinishClaimV6::Reported { bytes, receipt },
        Err(reason) => PalsRuntimeFinishClaimV6::Invalid { bytes, reason },
    })
}

/// Read a child session-release event on the original exclusive control
/// stream. Kernel peer and registered identity checks are necessary but not
/// by themselves sufficient to issue a physical proof: the supervisor must
/// also bind the executing image and the original producer-only finish path.
pub(super) fn observe_session_release_claim(
    owner: &mut DirectPalsChildV6,
    stream: &mut UnixStream,
    until: Instant,
    cancelled: &AtomicBool,
) -> io::Result<PalsSessionReleaseClaimV6> {
    require_original_child_peer(owner, stream)?;
    let bytes = match read_frame(stream, until, cancelled) {
        Ok(bytes) => bytes,
        Err((bytes, reason)) => return Ok(PalsSessionReleaseClaimV6::Incomplete { bytes, reason }),
    };
    if let Err(error) = require_original_child_peer(owner, stream) {
        return Ok(PalsSessionReleaseClaimV6::Incomplete {
            bytes,
            reason: error.to_string(),
        });
    }
    let event: PalsSessionReleaseEventV6 = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            return Ok(PalsSessionReleaseClaimV6::Invalid {
                bytes,
                reason: error.to_string(),
            });
        }
    };
    if !event_matches(
        &event,
        owner.child.id(),
        owner.lock.config.process_generation,
        &owner.lock.canonical_sha256,
    ) {
        return Ok(PalsSessionReleaseClaimV6::Invalid {
            bytes,
            reason: "PALS session release claim differs from original scope".into(),
        });
    }
    let acknowledgement = match serde_json::to_vec(&release_ack(&event, &bytes)) {
        Ok(mut ack) if ack.len() < MAX_FINISH_FRAME => {
            ack.push(b'\n');
            match write_frame(stream, &ack, until, cancelled) {
                Ok(()) => Ok(()),
                Err((bytes_written, error)) => Err(PalsReleaseAckWriteFailureV6 {
                    bytes: ack,
                    bytes_written,
                    reason: error.to_string(),
                    io_error: Some(error),
                }),
            }
        }
        Ok(ack) => Err(PalsReleaseAckWriteFailureV6 {
            bytes: ack,
            bytes_written: 0,
            reason: "PALS release acknowledgement exceeds bound".into(),
            io_error: None,
        }),
        Err(error) => Err(PalsReleaseAckWriteFailureV6 {
            bytes: Vec::new(),
            bytes_written: 0,
            reason: error.to_string(),
            io_error: Some(io::Error::other(error)),
        }),
    };
    Ok(PalsSessionReleaseClaimV6::Reported {
        bytes,
        event,
        acknowledgement,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn release_event() -> PalsSessionReleaseEventV6 {
        PalsSessionReleaseEventV6 {
            domain: RELEASE_DOMAIN.into(),
            process_id: 23,
            process_generation: 8,
            startup_lock_sha256: format!("{:x}", Sha256::digest(b"registered lock")),
            completed_synchronous_runs: 7,
            source: "original_report".into(),
            scope: "reported_session_only_not_device_fence".into(),
        }
    }

    #[test]
    fn release_ack_binds_exact_event_and_sender_waits_for_parent() {
        let (mut child, mut parent) = UnixStream::pair().unwrap();
        let until = Instant::now() + Duration::from_secs(3);
        let child_event = release_event();
        let child_task = std::thread::spawn(move || {
            let event_bytes = serde_json::to_vec(&child_event).unwrap();
            let mut frame = event_bytes.clone();
            frame.push(b'\n');
            write_frame(&mut child, &frame, until, &AtomicBool::new(false)).unwrap();
            let raw = read_frame(&mut child, until, &AtomicBool::new(false)).unwrap();
            let ack: PalsSessionReleaseAckV6 = serde_json::from_slice(&raw).unwrap();
            assert!(acknowledgement_matches(&ack, &child_event, &event_bytes));
        });
        let bytes = read_frame(&mut parent, until, &AtomicBool::new(false)).unwrap();
        let event: PalsSessionReleaseEventV6 = serde_json::from_slice(&bytes).unwrap();
        let ack = release_ack(&event, &bytes);
        let mut changed = bytes.clone();
        changed.push(b' ');
        assert!(!acknowledgement_matches(&ack, &event, &changed));
        let mut frame = serde_json::to_vec(&ack).unwrap();
        frame.push(b'\n');
        write_frame(&mut parent, &frame, until, &AtomicBool::new(false)).unwrap();
        child_task.join().unwrap();
    }

    #[test]
    fn incomplete_release_ack_keeps_received_bytes() {
        let (mut reader, mut writer) = UnixStream::pair().unwrap();
        writer.write_all(br#"{"domain":"partial"#).unwrap();
        drop(writer);
        let (bytes, error) = read_frame_detailed(
            &mut reader,
            Instant::now() + Duration::from_secs(1),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert_eq!(bytes, br#"{"domain":"partial"#);
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn reads_only_one_frame_and_preserves_remaining_bytes() {
        let (mut reader, mut writer) = UnixStream::pair().unwrap();
        writer.write_all(b"first\nsecond\n").unwrap();
        let cancelled = AtomicBool::new(false);
        let until = Instant::now() + Duration::from_secs(1);
        assert_eq!(
            read_frame(&mut reader, until, &cancelled).unwrap(),
            b"first"
        );
        assert_eq!(
            read_frame(&mut reader, until, &cancelled).unwrap(),
            b"second"
        );
    }

    #[test]
    fn incomplete_frame_retains_bytes_without_claiming_completion() {
        let (mut reader, mut writer) = UnixStream::pair().unwrap();
        writer.write_all(b"partial").unwrap();
        drop(writer);
        let (bytes, _) = read_frame(
            &mut reader,
            Instant::now() + Duration::from_secs(1),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert_eq!(bytes, b"partial");
    }

    #[test]
    fn session_claim_rejects_wrong_scope_and_replayed_identity() {
        let event = PalsSessionReleaseEventV6 {
            domain: RELEASE_DOMAIN.into(),
            process_id: 23,
            process_generation: 8,
            startup_lock_sha256: "a".repeat(64),
            completed_synchronous_runs: 0,
            source: "original_report".into(),
            scope: "reported_session_only_not_device_fence".into(),
        };
        assert!(event_matches(&event, 23, 8, &"a".repeat(64)));
        assert!(!event_matches(&event, 24, 8, &"a".repeat(64)));
        assert!(!event_matches(&event, 23, 9, &"a".repeat(64)));
        assert!(!event_matches(&event, 23, 8, &"b".repeat(64)));
        let mut wrong_scope = event;
        wrong_scope.scope = "device_wide".into();
        assert!(!event_matches(&wrong_scope, 23, 8, &"a".repeat(64)));
    }
}
