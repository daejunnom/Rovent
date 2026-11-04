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
use std::{
    io::{self, Read, Write},
    os::{fd::AsRawFd, unix::net::UnixStream},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

const MAX_FINISH_FRAME: usize = 256 << 10;
const IO_QUANTUM: Duration = Duration::from_millis(50);
const RELEASE_DOMAIN: &str = "rz-pals-session-release-event/1";

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
/// this one event. It never writes to the parent's liveness-only socket. This
/// does not mark a parent-owned child as released; a successful write also
/// does not prove receipt or consumption. Do not retry a partial write on a
/// different stream as though it were a completed event.
pub(crate) fn send_original_release_event(
    owner: &crate::pals_adaptive_startup::PinnedPalsSystemStartupV6,
    parent: &crate::rovent_bootstrap::launch::ConnectedParent,
    process: &crate::pals_adaptive::AdaptiveSessionProcessReceipt,
    control: UnixStream,
    until: Instant,
    cancelled: &AtomicBool,
) -> io::Result<()> {
    if cancelled.load(Ordering::Acquire) {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "PALS release send cancelled",
        ));
    }
    let event = event_from_original_process(owner, parent, process)?;
    let mut stream = parent.authenticate_separate_control(control, until)?;
    let mut bytes = serde_json::to_vec(&event).map_err(io::Error::other)?;
    if bytes.len() >= MAX_FINISH_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "PALS release event exceeds bound",
        ));
    }
    bytes.push(b'\n');
    let mut offset = 0;
    while offset < bytes.len() {
        if cancelled.load(Ordering::Acquire) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                format!("PALS release send cancelled after {offset} bytes"),
            ));
        }
        let remaining = until
            .checked_duration_since(Instant::now())
            .filter(|time| !time.is_zero())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("PALS release send deadline expired after {offset} bytes"),
                )
            })?;
        stream
            .set_write_timeout(Some(remaining.min(IO_QUANTUM)))
            .map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!("PALS release timeout failed after {offset} bytes: {error}"),
                )
            })?;
        match stream.write(&bytes[offset..]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    format!("PALS release event stopped after {offset} bytes"),
                ));
            }
            Ok(count) => offset += count,
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
            Err(error) => {
                return Err(io::Error::new(
                    error.kind(),
                    format!("PALS release send failed after {offset} bytes: {error}"),
                ));
            }
        }
    }
    Ok(())
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
    let prior = stream.read_timeout().map_err(|error| {
        (
            Vec::new(),
            format!("PALS finish timeout unavailable: {error}"),
        )
    })?;
    let result = read_frame_inner(stream, until, cancelled);
    match stream.set_read_timeout(prior) {
        Ok(()) => result,
        Err(error) => match result {
            Ok(bytes) => Err((
                bytes,
                format!("PALS finish timeout restore failed: {error}"),
            )),
            Err((bytes, reason)) => {
                Err((bytes, format!("{reason}; timeout restore failed: {error}")))
            }
        },
    }
}

fn read_frame_inner(
    stream: &mut UnixStream,
    until: Instant,
    cancelled: &AtomicBool,
) -> Result<Vec<u8>, (Vec<u8>, String)> {
    let mut result = Vec::new();
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err((result, "PALS finish observation cancelled".into()));
        }
        let Some(remaining) = until
            .checked_duration_since(Instant::now())
            .filter(|time| !time.is_zero())
        else {
            return Err((result, "PALS finish observation deadline expired".into()));
        };
        if let Err(error) = stream.set_read_timeout(Some(remaining.min(IO_QUANTUM))) {
            return Err((result, format!("PALS finish timeout setup failed: {error}")));
        }
        // Peek only as far as the next delimiter, then consume no bytes from
        // a later frame. The owner guarantees a single control-stream reader.
        let mut buffer = [0u8; 4096];
        let limit = buffer.len().min(MAX_FINISH_FRAME + 1 - result.len());
        let count = match recv(stream.as_raw_fd(), &mut buffer[..limit], MsgFlags::MSG_PEEK) {
            Ok(0) => return Err((result, "PALS finish frame incomplete".into())),
            Ok(count) => count,
            Err(Errno::EINTR | Errno::EAGAIN) => continue,
            Err(error) => return Err((result, error.to_string())),
        };
        let take = buffer[..count]
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(count, |index| index + 1);
        match stream.read(&mut buffer[..take]) {
            Ok(0) => return Err((result, "PALS finish frame incomplete".into())),
            Ok(count) => {
                let complete = buffer[count - 1] == b'\n';
                result.extend_from_slice(&buffer[..count - usize::from(complete)]);
                if result.len() > MAX_FINISH_FRAME {
                    return Err((result, "PALS finish frame exceeds bound".into()));
                }
                if complete {
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
            Err(error) => return Err((result, error.to_string())),
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
    Ok(PalsSessionReleaseClaimV6::Reported { bytes, event })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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
