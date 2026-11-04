//! Optional direct-owner go rejection observations on a separate datagram FD.
//!
//! This channel never authenticates GPU completion or authorizes a stop. The
//! original parent must bind each packet to its actual child via SCM_CREDENTIALS
//! and to its pending go/epoch/lock, and preserve normal product drain.

use crate::engine::{
    GoAdmissionObserver, GoAdmissionRejection, GoAdmissionSink, PriorSearchObservation,
};
use nix::{
    sys::socket::{
        MsgFlags, SockType as SocketType, getsockopt, send,
        sockopt::{PeerCredentials, SockType},
    },
    unistd::{geteuid, getppid},
};
use rz_contracts::{ErrorCode, Stage};
use std::{
    io,
    os::{fd::AsRawFd, unix::net::UnixDatagram},
    sync::{Arc, OnceLock},
};

const MAX_PACKET_BYTES: usize = 4096;
static EARLY_INHERITED: OnceLock<(i32, UnixDatagram)> = OnceLock::new();

/// Called only at the wrapper's single-threaded process entry, before parsing
/// or starting workers. The caller explicitly declares that this descriptor
/// belongs exclusively to the child and cannot be closed or replaced by any
/// other owner. Retaining the duplicate pins the original socket identity for
/// later checks. No open-file-description or unrelated descriptor is changed.
pub(super) fn seal_early_exclusive(fd: i32) -> io::Result<()> {
    let duplicate = rz_platform::seal_exclusive_inherited_unix_datagram(fd)?;
    EARLY_INHERITED
        .set((fd, duplicate))
        .map_err(|_| io::Error::new(io::ErrorKind::AlreadyExists, "go observer already sealed"))
}

/// Revalidate the original descriptor against the socket pinned at entry.
pub(super) fn early_seal_matches(fd: i32) -> bool {
    EARLY_INHERITED.get().is_some_and(|(original, socket)| {
        *original == fd && rz_platform::mark_inherited_datagram_cloexec(fd, socket).is_ok()
    })
}

struct OriginalParentGoDatagram {
    socket: UnixDatagram,
    parent: Arc<super::ConnectedParent>,
    epoch: u64,
    lock_sha256: String,
}

/// Consume the original inherited datagram capability, only after the V3
/// parent handshake. The caller must keep this observer in the original scope.
pub(super) fn from_original_parent(
    socket: UnixDatagram,
    exclusive_inherited_fd: i32,
    parent: Arc<super::ConnectedParent>,
    epoch: u64,
    lock_sha256: &str,
    cuda_ordinal: i32,
) -> io::Result<GoAdmissionObserver> {
    if !parent.matches(epoch, lock_sha256, cuda_ordinal) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "original V3 context differs",
        ));
    }
    parent.ensure_live()?;
    let kind = getsockopt(&socket, SockType).map_err(io::Error::other)?;
    let peer = getsockopt(&socket, PeerCredentials).map_err(io::Error::other)?;
    if kind != SocketType::Datagram
        || peer.pid() != getppid().as_raw()
        || peer.uid() != geteuid().as_raw()
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "original datagram peer differs",
        ));
    }
    // The explicit child-exclusive inheritance contract permits marking only
    // the original descriptor CLOEXEC after its identity and peer validation.
    // This leaves shared file-status flags and ownership unchanged.
    rz_platform::mark_inherited_datagram_cloexec(exclusive_inherited_fd, &socket)?;
    // Do not change the shared file-description flags on the inherited
    // original; each send below uses MSG_DONTWAIT explicitly.
    Ok(GoAdmissionObserver::from_sink(Arc::new(
        OriginalParentGoDatagram {
            socket,
            parent,
            epoch,
            lock_sha256: lock_sha256.to_owned(),
        },
    )))
}

impl GoAdmissionSink for OriginalParentGoDatagram {
    fn try_publish(&self, event: GoAdmissionRejection) -> bool {
        if event.epoch.0 != self.epoch || self.parent.ensure_live().is_err() {
            return false;
        }
        let packet = serde_json::json!({
            "schema": "rz-rovent-go-admission/1",
            "process_id": std::process::id(),
            "process_epoch": self.epoch,
            "input_lock_sha256": self.lock_sha256,
            "go_sequence": event.go_sequence,
            "game_generation": event.game.0,
            "root_generation": event.root.0,
            "classification": "driver_rejected_before_ticket",
            "prior_search": match event.prior_search {
                PriorSearchObservation::NoOwnerSearchOrDrainObserved => "no_owner_search_or_drain_observed",
                PriorSearchObservation::SearchDrainOrFailurePending => "search_drain_or_failure_pending",
            },
            "stage": stage(event.stage),
            "code": code(event.code),
            "physical_completion": "not_inferred",
        });
        let Ok(bytes) = serde_json::to_vec(&packet) else {
            return false;
        };
        !bytes.is_empty()
            && bytes.len() <= MAX_PACKET_BYTES
            && send(
                self.socket.as_raw_fd(),
                &bytes,
                MsgFlags::MSG_DONTWAIT | MsgFlags::MSG_NOSIGNAL,
            )
            .is_ok_and(|count| count == bytes.len())
    }
}

fn stage(value: Stage) -> &'static str {
    match value {
        Stage::Contract => "contract",
        Stage::Admission => "admission",
        Stage::Backend => "backend",
        Stage::Output => "output",
        Stage::Backup => "backup",
    }
}

fn code(value: ErrorCode) -> &'static str {
    match value {
        ErrorCode::InvalidInput => "invalid_input",
        ErrorCode::UnsupportedContract => "unsupported_contract",
        ErrorCode::BackendUnavailable => "backend_unavailable",
        ErrorCode::ResourceExhausted => "resource_exhausted",
        ErrorCode::NumericalFailure => "numerical_failure",
        ErrorCode::IdentityMismatch => "identity_mismatch",
        ErrorCode::BackendFailure => "backend_failure",
        ErrorCode::Canceled => "canceled",
        ErrorCode::Expired => "expired",
        ErrorCode::Stale => "stale",
    }
}
