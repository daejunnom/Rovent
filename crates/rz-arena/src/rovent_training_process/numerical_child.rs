//! OS-only transport observations shared by separately versioned trainers.
//! These helpers parse no registration, request, recipe or training result and
//! never issue dataset, process-completion or optimizer authority.

use super::{ArenaError, check, invalid};
use std::{
    io::{Read, Write},
    os::unix::net::UnixStream,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

pub(super) struct UnixFrameReadErrors {
    pub(super) partial_stop: &'static str,
    pub(super) deadline_cancel: &'static str,
    pub(super) partial_eof: &'static str,
    pub(super) byte_overflow: &'static str,
    pub(super) byte_frame_bound: &'static str,
    pub(super) pipelined: &'static str,
    pub(super) read: &'static str,
}

pub(super) struct UnixFrameReadScope<'a> {
    pub(super) stop: &'a AtomicBool,
    pub(super) cancel: &'a AtomicBool,
    pub(super) whole_until: Instant,
    pub(super) max_total_bytes: u64,
    pub(super) max_frame_bytes: usize,
    pub(super) errors: UnixFrameReadErrors,
}

/// Charges each received byte before interpreting a bounded newline frame.
/// The caller retains its own exact protocol, sequencing and semantic ledger.
pub(super) fn read_unix_frame(
    stream: &mut UnixStream,
    scope: UnixFrameReadScope<'_>,
    total: &mut u64,
) -> Result<Option<Vec<u8>>, ArenaError> {
    let mut raw = Vec::new();
    let mut block = [0; 1024];
    loop {
        if scope.stop.load(Ordering::Acquire) {
            return if raw.is_empty() {
                Ok(None)
            } else {
                Err(invalid(scope.errors.partial_stop))
            };
        }
        check(
            Instant::now() < scope.whole_until && !scope.cancel.load(Ordering::Acquire),
            scope.errors.deadline_cancel,
        )?;
        match stream.read(&mut block) {
            Ok(0) => {
                return if raw.is_empty() {
                    Ok(None)
                } else {
                    Err(invalid(scope.errors.partial_eof))
                };
            }
            Ok(n) => {
                *total = total
                    .checked_add(n as u64)
                    .ok_or_else(|| invalid(scope.errors.byte_overflow))?;
                check(
                    *total <= scope.max_total_bytes && raw.len() + n <= scope.max_frame_bytes,
                    scope.errors.byte_frame_bound,
                )?;
                raw.extend_from_slice(&block[..n]);
                if let Some(at) = raw.iter().position(|b| *b == b'\n') {
                    check(at + 1 == raw.len(), scope.errors.pipelined)?;
                    raw.pop();
                    return Ok(Some(raw));
                }
            }
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::Interrupted
                ) => {}
            Err(_) => return Err(invalid(scope.errors.read)),
        }
    }
}

/// Only frame-length-checked transport bytes, including their final newline.
/// This owner grants no request, execution, learning or physical authority.
pub(super) struct PreparedUnixAck {
    bytes: Vec<u8>,
}

pub(super) fn prepare_unix_ack(
    mut raw: Vec<u8>,
    max_frame_bytes: usize,
    bound_error: &str,
) -> Result<PreparedUnixAck, ArenaError> {
    check(raw.len() < max_frame_bytes, bound_error)?;
    raw.push(b'\n');
    Ok(PreparedUnixAck { bytes: raw })
}

/// Reserve the entire write, including its newline, before issuing any bytes.
/// A failed bound, partial write or lost ACK never refunds the charged total.
/// Legacy protocols explicitly retain their pre-existing inbound-only policy.
pub(super) fn charge_unix_ack(
    total: &mut u64,
    ack: &PreparedUnixAck,
    maximum: u64,
    overflow_error: &str,
    budget_error: &str,
) -> Result<(), ArenaError> {
    *total = total
        .checked_add(ack.bytes.len() as u64)
        .ok_or_else(|| invalid(overflow_error))?;
    check(*total <= maximum, budget_error)
}

pub(super) fn write_unix_ack(
    stream: &mut UnixStream,
    ack: &PreparedUnixAck,
    error: &str,
) -> Result<(), ArenaError> {
    stream.write_all(&ack.bytes).map_err(|_| invalid(error))
}

/// A current SO_PEERCRED observation, without any trainer admission decision.
pub(super) struct UnixPeerCredentials {
    pub(super) pid: u32,
    pub(super) uid: u32,
}

pub(super) fn unix_peer_credentials(
    stream: &UnixStream,
    credential_error: &str,
    pid_error: &str,
) -> Result<UnixPeerCredentials, ArenaError> {
    use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
    let peer = getsockopt(stream, PeerCredentials).map_err(|_| invalid(credential_error))?;
    let pid = u32::try_from(peer.pid()).map_err(|_| invalid(pid_error))?;
    Ok(UnixPeerCredentials {
        pid,
        uid: peer.uid(),
    })
}

/// The caller supplies status bytes read under its original resource policy.
/// Matching PPid is checked before observing the current process-group leader.
/// This predicate cannot attest image, PID lifetime, resources or completion.
pub(super) fn is_direct_parent_group_leader(pid: u32, parent_pid: u32, status: &str) -> bool {
    status
        .lines()
        .find_map(|line| line.strip_prefix("PPid:"))
        .and_then(|value| value.trim().parse::<u32>().ok())
        == Some(parent_pid)
        && nix::unistd::getpgid(Some(nix::unistd::Pid::from_raw(pid as i32)))
            .is_ok_and(|group| group.as_raw() == pid as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ack_frame_bound_includes_the_original_final_newline() {
        let raw = br#"{"accepted":true}"#.to_vec();
        let maximum = raw.len() + 1;
        assert!(prepare_unix_ack(raw.clone(), raw.len(), "bound").is_err());
        let prepared = prepare_unix_ack(raw.clone(), maximum, "bound").expect("bounded ACK");
        assert_eq!(prepared.bytes.len(), maximum);
        assert_eq!(&prepared.bytes[..raw.len()], raw.as_slice());
        assert_eq!(prepared.bytes.last(), Some(&b'\n'));
    }

    #[test]
    fn failed_ack_budget_keeps_the_full_attempt_charged() {
        let prepared = prepare_unix_ack(b"{}".to_vec(), 3, "bound").expect("bounded ACK");
        let mut total = 7;
        assert!(charge_unix_ack(&mut total, &prepared, 9, "overflow", "budget").is_err());
        assert_eq!(total, 10);
    }

    #[test]
    fn lost_ack_write_does_not_refund_prior_inbound_or_outbound_bytes() {
        let (mut sender, receiver) = UnixStream::pair().expect("private test socket pair");
        drop(receiver);
        let prepared = prepare_unix_ack(b"{}".to_vec(), 3, "bound").expect("bounded ACK");
        let mut total = 7;
        charge_unix_ack(&mut total, &prepared, 10, "overflow", "budget")
            .expect("single outbound reservation");
        assert!(write_unix_ack(&mut sender, &prepared, "lost ACK").is_err());
        assert_eq!(total, 10);
    }
}
