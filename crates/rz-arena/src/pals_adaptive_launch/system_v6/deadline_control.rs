//! Nonblocking original-owner deadline grant transport. No GPU completion or
//! process ownership is inferred from a connection, grant, EOF or response.
//!
//! The listener and pinned proxy file are owned by the caller. In particular,
//! this module does not create or remove a socket path. The existing process
//! supervisor currently polls its control callback only while Fastchess is
//! live and before stop; shutdown/release observation needs a separate change.

use super::deadline_grant::OwnerWindow;
use crate::{ArenaError, process::SpawnCustody};
use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
use rz_contracts::pals::deadline_grant_v1::{
    DeadlineGrant, DeadlinePhase, DeadlineRequest, DeadlineScope, REQUEST_BYTES,
};
use std::{
    fs::File,
    io::{self, Read, Write},
    os::unix::{
        fs::MetadataExt,
        net::{UnixListener, UnixStream},
    },
    time::Instant,
};

const MAX_ADMISSIONS: usize = 256;

fn failure(stage: &'static str) -> ArenaError {
    ArenaError::Io(format!("pals_deadline_control:{stage}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProcessIdentity {
    parent: u32,
    group: u32,
    start_ticks: u64,
}

fn process_identity(pid: u32) -> Result<ProcessIdentity, ArenaError> {
    let file = File::open(format!("/proc/{pid}/stat")).map_err(|_| failure("proc_open"))?;
    let mut bytes = Vec::new();
    file.take(4097)
        .read_to_end(&mut bytes)
        .map_err(|_| failure("proc_read"))?;
    if bytes.len() > 4096 {
        return Err(failure("proc_size"));
    }
    let end = bytes
        .iter()
        .rposition(|byte| *byte == b')')
        .ok_or_else(|| failure("proc_format"))?;
    let head = bytes
        .iter()
        .position(|byte| *byte == b'(')
        .ok_or_else(|| failure("proc_format"))?;
    let expected = std::str::from_utf8(&bytes[..head])
        .map_err(|_| failure("proc_pid"))?
        .trim()
        .parse::<u32>()
        .map_err(|_| failure("proc_pid"))?;
    if expected != pid || head >= end {
        return Err(failure("proc_pid"));
    }
    let tail = std::str::from_utf8(&bytes[end + 1..]).map_err(|_| failure("proc_format"))?;
    let fields: Vec<&str> = tail.split_whitespace().collect();
    let number = |index: usize| {
        fields
            .get(index)
            .ok_or_else(|| failure("proc_field"))?
            .parse::<u64>()
            .map_err(|_| failure("proc_field"))
    };
    Ok(ProcessIdentity {
        parent: u32::try_from(number(1)?).map_err(|_| failure("proc_parent"))?,
        group: u32::try_from(number(2)?).map_err(|_| failure("proc_group"))?,
        start_ticks: number(19)?,
    })
}

struct Peer {
    stream: UnixStream,
    pid: u32,
    identity: ProcessIdentity,
    request: [u8; REQUEST_BYTES],
    read: usize,
    response: Option<Vec<u8>>,
    written: usize,
}

/// A completed grant is a transport event, not a physical completion proof.
pub(crate) struct DeadlineGrantObservation {
    pub peer_pid: u32,
    pub peer_start_ticks: u64,
    pub scope: DeadlineScope,
}

pub(crate) struct DeadlineControl<'a> {
    listener: &'a UnixListener,
    window: &'a OwnerWindow,
    proxy_pin: &'a File,
    owner_id: [u8; 16],
    owner_epoch: u64,
    process_cleanup_until: Instant,
    fastchess: Option<(u32, ProcessIdentity)>,
    admitted: Vec<(DeadlineScope, u32)>,
    used: Vec<DeadlineScope>,
    peer: Option<Peer>,
}

impl<'a> DeadlineControl<'a> {
    pub(crate) fn new(
        listener: &'a UnixListener,
        window: &'a OwnerWindow,
        proxy_pin: &'a File,
        owner_id: [u8; 16],
        owner_epoch: u64,
        process_cleanup_until: Instant,
    ) -> Result<Self, ArenaError> {
        if owner_id == [0; 16] || owner_epoch == 0 {
            return Err(failure("owner_identity"));
        }
        listener
            .set_nonblocking(true)
            .map_err(|_| failure("listener_nonblocking"))?;
        Ok(Self {
            listener,
            window,
            proxy_pin,
            owner_id,
            owner_epoch,
            process_cleanup_until,
            fastchess: None,
            admitted: Vec::new(),
            used: Vec::new(),
            peer: None,
        })
    }

    /// Called by the original owner after determining the exact game/role and
    /// lock scope. Admission is one-shot and does not itself accept a peer.
    pub(crate) fn admit(
        &mut self,
        scope: DeadlineScope,
        expected_peer_pid: u32,
    ) -> Result<(), ArenaError> {
        scope.validate().map_err(failure)?;
        if scope.owner_id != self.owner_id
            || scope.owner_epoch != self.owner_epoch
            || self.admitted.len() + self.used.len() >= MAX_ADMISSIONS
            || expected_peer_pid == 0
            || self.admitted.iter().any(|(existing, _)| existing == &scope)
            || self.used.contains(&scope)
        {
            return Err(failure("scope_admission"));
        }
        self.admitted.push((scope, expected_peer_pid));
        Ok(())
    }

    /// One bounded, nonblocking step from `SpawnControl`. Caller supplies the
    /// original supervisor deadline, not a new relative timeout.
    pub(crate) fn poll(
        &mut self,
        custody: &SpawnCustody,
        execution_until: Instant,
    ) -> Result<Option<DeadlineGrantObservation>, ArenaError> {
        if Instant::now() >= execution_until {
            return Err(failure("execution_deadline"));
        }
        let pid = custody
            .child()
            .ok_or_else(|| failure("fastchess_custody"))?
            .id();
        let identity = process_identity(pid)?;
        if identity.group != pid {
            return Err(failure("fastchess_group"));
        }
        match self.fastchess {
            None => self.fastchess = Some((pid, identity)),
            Some((original_pid, original)) if original_pid == pid && original == identity => {}
            Some(_) => return Err(failure("fastchess_identity_changed")),
        }

        if self.peer.is_none() {
            let (stream, _) = match self.listener.accept() {
                Ok(pair) => pair,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(None),
                Err(_) => return Err(failure("accept")),
            };
            stream
                .set_nonblocking(true)
                .map_err(|_| failure("peer_nonblocking"))?;
            let credentials =
                getsockopt(&stream, PeerCredentials).map_err(|_| failure("peer_credentials"))?;
            if credentials.uid() != nix::unistd::geteuid().as_raw() || credentials.pid() <= 0 {
                return Err(failure("peer_uid_or_pid"));
            }
            let peer_pid = u32::try_from(credentials.pid()).map_err(|_| failure("peer_pid"))?;
            let peer_identity = process_identity(peer_pid)?;
            self.verify_peer(pid, peer_pid, peer_identity)?;
            self.peer = Some(Peer {
                stream,
                pid: peer_pid,
                identity: peer_identity,
                request: [0; REQUEST_BYTES],
                read: 0,
                response: None,
                written: 0,
            });
        }

        let peer = self.peer.as_ref().ok_or_else(|| failure("peer_missing"))?;
        self.verify_peer(pid, peer.pid, peer.identity)?;
        let peer = self.peer.as_mut().ok_or_else(|| failure("peer_missing"))?;
        if peer.response.is_none() {
            match peer.stream.read(&mut peer.request[peer.read..]) {
                Ok(0) => return Err(failure("peer_eof")),
                Ok(n) => peer.read += n,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(None),
                Err(_) => return Err(failure("peer_read")),
            }
            if peer.read < REQUEST_BYTES {
                return Ok(None);
            }
            let request = DeadlineRequest::decode(&peer.request).map_err(failure)?;
            let index = self
                .admitted
                .iter()
                .position(|(scope, admitted_pid)| {
                    scope == &request.scope && *admitted_pid == peer.pid
                })
                .ok_or_else(|| failure("scope_not_admitted"))?;
            let (scope, _) = self.admitted.remove(index);
            // Reserve before writing: a partial or failed reply is not retried.
            self.used.push(scope.clone());
            let encoded = self.window.grant(&scope, &peer.request).map_err(failure)?;
            let mut grant = DeadlineGrant::decode(&encoded).map_err(failure)?;
            // Cleanup is bounded by the original supervisor's separately
            // registered cleanup deadline, not by its execution deadline.
            // The latter would falsely deny the cleanup reserve altogether.
            let supervisor_until = if scope.phase == DeadlinePhase::Cleanup {
                self.process_cleanup_until
            } else {
                execution_until
            };
            let supervisor_remaining = supervisor_until
                .checked_duration_since(Instant::now())
                .ok_or_else(|| failure("execution_deadline"))?;
            let supervisor_ns = u64::try_from(supervisor_remaining.as_nanos())
                .map_err(|_| failure("execution_deadline_overflow"))?;
            grant.remaining_ns = grant.remaining_ns.min(supervisor_ns);
            peer.response = Some(grant.encode().map_err(failure)?.to_vec());
        }
        let response = peer
            .response
            .as_ref()
            .ok_or_else(|| failure("response_missing"))?;
        match peer.stream.write(&response[peer.written..]) {
            Ok(0) => return Err(failure("peer_write_zero")),
            Ok(n) => peer.written += n,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(None),
            Err(_) => return Err(failure("peer_write")),
        }
        if peer.written < response.len() {
            return Ok(None);
        }
        let peer = self.peer.take().ok_or_else(|| failure("peer_missing"))?;
        let scope = DeadlineRequest::decode(&peer.request)
            .map_err(failure)?
            .scope;
        Ok(Some(DeadlineGrantObservation {
            peer_pid: peer.pid,
            peer_start_ticks: peer.identity.start_ticks,
            scope,
        }))
    }

    fn verify_peer(
        &self,
        leader: u32,
        pid: u32,
        expected: ProcessIdentity,
    ) -> Result<(), ArenaError> {
        let before = process_identity(pid)?;
        if before != expected || before.parent != leader || before.group != leader {
            return Err(failure("peer_relation"));
        }
        let image = File::open(format!("/proc/{pid}/exe")).map_err(|_| failure("peer_image"))?;
        let actual = image
            .metadata()
            .map_err(|_| failure("peer_image_metadata"))?;
        let pin = self
            .proxy_pin
            .metadata()
            .map_err(|_| failure("proxy_pin_metadata"))?;
        if actual.dev() != pin.dev() || actual.ino() != pin.ino() || !actual.is_file() {
            return Err(failure("peer_image_pin"));
        }
        if process_identity(pid)? != before {
            return Err(failure("peer_identity_changed"));
        }
        Ok(())
    }
}
