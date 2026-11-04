//! Nonblocking original-owner deadline grant transport. No GPU completion or
//! process ownership is inferred from a connection, grant, EOF or response.
//!
//! The listener and pinned proxy file are owned by the caller. In particular,
//! this module does not create or remove a socket path. The existing process
//! supervisor currently polls its control callback only while Fastchess is
//! live and before stop; shutdown/release observation needs a separate change.

use super::{LockedPalsSystemArenaPairV6, PalsV6ExecutionStart, deadline_grant::OwnerWindow};
use crate::{ArenaError, process::SpawnCustody};
use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
use rz_contracts::pals::cancel_control_v1::{
    CANCEL_ACK_BYTES, CANCEL_REQUEST_BYTES, CancelAck, CancelAckDisposition, CancelRequest,
};
use rz_contracts::pals::deadline_grant_v1::{
    DeadlineGrant, DeadlinePhase, DeadlineRequest, DeadlineRole, DeadlineScope, REQUEST_BYTES,
};
use rz_experiments::NativeEngineRole;
use std::{
    fs::File,
    io::{self, Read, Write},
    ops::Deref,
    os::unix::{
        fs::MetadataExt,
        net::{UnixListener, UnixStream},
    },
    sync::Arc,
    time::Instant,
};

#[path = "smoke_deadline_control.rs"]
mod smoke_deadline_control;
pub(crate) use smoke_deadline_control::{
    SmokeCancelProgress, SmokeControlProgress, SmokeDeadlineControl, SmokeGrantObservation,
    SmokeReobserveObservation, SmokeReobserveProgress, validate_smoke_request_ids,
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
    response_until: Option<Instant>,
    written: usize,
    original_endpoint: bool,
}

struct IssuedOriginalRun {
    scope: DeadlineScope,
    pid: u32,
    identity: ProcessIdentity,
    cancel_claimed: bool,
}

/// Evidence for the same peer which received a complete original run grant.
/// The caller must keep the creating control owner alive and recheck it before
/// each I/O operation; this is not cancellation or physical completion.
pub(crate) struct OriginalCancelPeer {
    scope: DeadlineScope,
    pid: u32,
    identity: ProcessIdentity,
}

impl OriginalCancelPeer {
    pub(crate) fn scope(&self) -> &DeadlineScope {
        &self.scope
    }
}

/// Progress is transport observation only. In particular, an ACK means the
/// run-cancel token was accepted, not that the child or GPU has finished.
pub(crate) enum OriginalCancelProgress {
    Pending { written: usize, received: usize },
    Acknowledged,
}

/// The original control owner must retain this same object on all outcomes.
/// A partially sent frame is never reissued on a replacement connection.
#[must_use = "retain the original cancel stream until its owner is settled"]
pub(crate) struct OriginalCancelSender {
    stream: UnixStream,
    peer: OriginalCancelPeer,
    sequence: u64,
    request: [u8; CANCEL_REQUEST_BYTES],
    written: usize,
    ack: [u8; CANCEL_ACK_BYTES],
    received: usize,
    acknowledged: bool,
    failed: Option<String>,
}

impl OriginalCancelSender {
    /// The stream was accepted on an original role-specific channel and must
    /// still match the very same issued run grant and actual connected peer.
    /// A rejected construction returns the same stream to its owner. The
    /// stream may already have been switched to nonblocking mode.
    pub(crate) fn new(
        control: &mut DeadlineControl<'_>,
        custody: &SpawnCustody,
        stream: UnixStream,
        scope: &DeadlineScope,
        sequence: u64,
    ) -> Result<Self, (UnixStream, ArenaError)> {
        // This is a fresh dedicated channel with one run-cancel request;
        // its receiver starts at sequence zero and permits sequence one.
        if sequence != 1 {
            return Err((stream, failure("cancel_initial_sequence")));
        }
        let request = match (CancelRequest {
            scope: scope.clone(),
            sequence,
        })
        .encode()
        {
            Ok(request) => request,
            Err(_) => return Err((stream, failure("cancel_request_invalid"))),
        };
        if stream.set_nonblocking(true).is_err() {
            return Err((stream, failure("cancel_nonblocking")));
        }
        let peer = match control.claim_cancel_peer(custody, &stream, scope) {
            Ok(peer) => peer,
            Err(error) => return Err((stream, error)),
        };
        Ok(Self {
            stream,
            peer,
            sequence,
            request,
            written: 0,
            ack: [0; CANCEL_ACK_BYTES],
            received: 0,
            acknowledged: false,
            failed: None,
        })
    }

    /// One bounded nonblocking turn. The caller supplies its original
    /// pre-signal deadline and never treats an ACK as a process-tree fence.
    pub(crate) fn poll(
        &mut self,
        control: &DeadlineControl<'_>,
        custody: &SpawnCustody,
        until: Instant,
    ) -> Result<OriginalCancelProgress, ArenaError> {
        if self.acknowledged {
            return Ok(OriginalCancelProgress::Acknowledged);
        }
        if let Some(original) = &self.failed {
            return Err(ArenaError::Io(format!("cancel_sender_sticky:{original}")));
        }
        let result = self.poll_inner(control, custody, until);
        if let Err(error) = &result {
            // No new request is sent after a failed or partial transfer. The
            // original stream, offsets, and first cause remain in this owner.
            self.failed = Some(error.to_string());
        }
        result
    }

    fn poll_inner(
        &mut self,
        control: &DeadlineControl<'_>,
        custody: &SpawnCustody,
        until: Instant,
    ) -> Result<OriginalCancelProgress, ArenaError> {
        let until = until.min(control.process_cleanup_until);
        if Instant::now() >= until {
            return Err(failure("cancel_deadline"));
        }
        control.verify_cancel_peer(custody, &self.stream, &self.peer)?;
        if Instant::now() >= until {
            return Err(failure("cancel_deadline"));
        }
        if self.written < self.request.len() {
            match self.stream.write(&self.request[self.written..]) {
                Ok(0) => return Err(failure("cancel_write_zero")),
                Ok(count) => self.written += count,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) => {}
                Err(_) => return Err(failure("cancel_write")),
            }
            if self.written < self.request.len() {
                return Ok(OriginalCancelProgress::Pending {
                    written: self.written,
                    received: self.received,
                });
            }
        }
        if Instant::now() >= until {
            return Err(failure("cancel_deadline"));
        }
        match self.stream.read(&mut self.ack[self.received..]) {
            Ok(0) => return Err(failure("cancel_peer_eof")),
            Ok(count) => self.received += count,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) => {}
            Err(_) => return Err(failure("cancel_read")),
        }
        if self.received < self.ack.len() {
            return Ok(OriginalCancelProgress::Pending {
                written: self.written,
                received: self.received,
            });
        }
        let ack = CancelAck::decode(&self.ack).map_err(failure)?;
        // This sender transmits one request exactly once. A Duplicate response
        // cannot be justified by it and is retained as a protocol failure.
        if ack.scope != *self.peer.scope()
            || ack.sequence != self.sequence
            || ack.disposition != CancelAckDisposition::Accepted
        {
            return Err(failure("cancel_ack_mismatch"));
        }
        control.verify_cancel_peer(custody, &self.stream, &self.peer)?;
        if Instant::now() >= until {
            return Err(failure("cancel_deadline"));
        }
        self.acknowledged = true;
        Ok(OriginalCancelProgress::Acknowledged)
    }
}

/// A completed grant is a transport event, not a physical completion proof.
pub(crate) struct DeadlineGrantObservation {
    pub peer_pid: u32,
    pub peer_start_ticks: u64,
    pub scope: DeadlineScope,
    /// Original upper bound used for this already transmitted frame. A later
    /// earlier stop cannot retroactively update this observation.
    pub granted_until: Instant,
    /// Only the final write was observed in this phase; earlier bytes may
    /// already have been sent in the preceding callback.
    pub completion_observed_during_cleanup: bool,
}

/// A same-user, same-image, direct Fastchess child observation. This does not
/// identify its game/role or authorize an endpoint by itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DeadlinePeerObservation {
    pub pid: u32,
    pub start_ticks: u64,
}

pub(crate) struct DeadlineControl<'a> {
    listener: OriginalResource<'a, UnixListener>,
    window: OriginalResource<'a, OwnerWindow>,
    proxy_pin: OriginalResource<'a, File>,
    owner_id: [u8; 16],
    owner_epoch: u64,
    process_cleanup_until: Instant,
    fastchess: Option<(u32, ProcessIdentity)>,
    admitted: Vec<(DeadlineScope, u32, ProcessIdentity, bool)>,
    used: Vec<DeadlineScope>,
    issued_original_runs: Vec<IssuedOriginalRun>,
    peer: Option<Peer>,
}

/// An owned control retains the same kernel handles in its supervisor bundle;
/// the borrowed form remains available to existing isolated CPU fixtures.
enum OriginalResource<'a, T> {
    Borrowed(&'a T),
    Owned(Arc<T>),
}
impl<T> Deref for OriginalResource<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            Self::Borrowed(value) => value,
            Self::Owned(value) => value.as_ref(),
        }
    }
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
            listener: OriginalResource::Borrowed(listener),
            window: OriginalResource::Borrowed(window),
            proxy_pin: OriginalResource::Borrowed(proxy_pin),
            owner_id,
            owner_epoch,
            process_cleanup_until,
            fastchess: None,
            admitted: Vec::new(),
            used: Vec::new(),
            issued_original_runs: Vec::new(),
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
            || self
                .admitted
                .iter()
                .any(|(existing, _, _, _)| existing == &scope)
            || self.used.contains(&scope)
        {
            return Err(failure("scope_admission"));
        }
        let identity = process_identity(expected_peer_pid)?;
        self.admitted
            .push((scope, expected_peer_pid, identity, false));
        Ok(())
    }

    /// Observe only the already authenticated pending child. The caller must
    /// bind it to the role-specific channel supplied by the original runner;
    /// role and lock values claimed in a request do not establish that binding.
    pub(crate) fn pending_peer(&self) -> Result<Option<DeadlinePeerObservation>, ArenaError> {
        let Some(peer) = self.peer.as_ref() else {
            return Ok(None);
        };
        let (leader, original_leader) =
            self.fastchess.ok_or_else(|| failure("fastchess_custody"))?;
        if process_identity(leader)? != original_leader {
            return Err(failure("fastchess_identity_changed"));
        }
        self.verify_peer(leader, peer.pid, peer.identity)?;
        if process_identity(leader)? != original_leader {
            return Err(failure("fastchess_identity_changed"));
        }
        Ok(Some(DeadlinePeerObservation {
            pid: peer.pid,
            start_ticks: peer.identity.start_ticks,
        }))
    }

    /// Atomically reserve all three phases for the same authenticated child,
    /// deriving generation and lock from the original locked pair. The caller
    /// must have selected this listener via its original per-role launch
    /// channel and supply the supervisor's real child-start observation.
    /// Neither wire claims nor connection time provide either fact.
    pub(crate) fn admit_original_endpoint(
        &mut self,
        pair: &LockedPalsSystemArenaPairV6,
        execution: &PalsV6ExecutionStart,
        role: NativeEngineRole,
        game: u32,
        child_started: Instant,
        request_ids: [u64; 3],
    ) -> Result<[DeadlineScope; 3], ArenaError> {
        if pair.sha256() != execution.pair_sha256()
            || !execution.matches_deadline_identity(self.owner_id, self.owner_epoch)
        {
            return Err(failure("original_pair_mismatch"));
        }
        let startup_until = execution.child_startup_until(role, child_started)?;
        if !self.window.matches_bounds(
            execution.started(),
            startup_until,
            execution.run_until(),
            execution.cleanup_until(),
        ) || self.process_cleanup_until != execution.cleanup_until()
        {
            return Err(failure("original_window_mismatch"));
        }
        let peer = self
            .pending_peer()?
            .ok_or_else(|| failure("peer_missing"))?;
        let endpoint = pair.endpoint(role);
        let lock_sha256 = decode_sha256(&endpoint.startup_lock.canonical_sha256)?;
        let role = match role {
            NativeEngineRole::Baseline => DeadlineRole::Baseline,
            NativeEngineRole::Candidate => DeadlineRole::Candidate,
        };
        let scopes = [
            DeadlinePhase::Startup,
            DeadlinePhase::Run,
            DeadlinePhase::Cleanup,
        ]
        .map(|phase| {
            let index = match phase {
                DeadlinePhase::Startup => 0,
                DeadlinePhase::Run => 1,
                DeadlinePhase::Cleanup => 2,
            };
            DeadlineScope {
                owner_id: self.owner_id,
                owner_epoch: self.owner_epoch,
                role,
                game,
                generation: endpoint.startup_lock.config.process_generation,
                request_id: request_ids[index],
                lock_sha256,
                phase,
            }
        });
        if request_ids.contains(&0)
            || request_ids[0] == request_ids[1]
            || request_ids[0] == request_ids[2]
            || request_ids[1] == request_ids[2]
            || self
                .admitted
                .len()
                .saturating_add(self.used.len())
                .saturating_add(3)
                > MAX_ADMISSIONS
            || scopes.iter().any(|scope| {
                scope.validate().is_err()
                    || self
                        .admitted
                        .iter()
                        .any(|(existing, _, _, _)| existing == scope)
                    || self.used.contains(scope)
            })
        {
            return Err(failure("scope_admission"));
        }
        let identity = self
            .peer
            .as_ref()
            .ok_or_else(|| failure("peer_missing"))?
            .identity;
        self.admitted.extend(
            scopes
                .iter()
                .cloned()
                .map(|scope| (scope, peer.pid, identity, true)),
        );
        Ok(scopes)
    }

    /// One bounded, nonblocking step from `SpawnControl`. Caller supplies the
    /// original supervisor deadline, not a new relative timeout.
    pub(crate) fn poll(
        &mut self,
        custody: &SpawnCustody,
        execution_until: Instant,
    ) -> Result<Option<DeadlineGrantObservation>, ArenaError> {
        self.poll_with_phase(custody, execution_until, false)
    }

    /// Cleanup callback deadlines can be shorter than the original whole
    /// window. Do not apply that bound to requests serviced during normal run.
    pub(crate) fn poll_cleanup(
        &mut self,
        custody: &SpawnCustody,
        cleanup_until: Instant,
    ) -> Result<Option<DeadlineGrantObservation>, ArenaError> {
        self.poll_with_phase(custody, cleanup_until, true)
    }

    fn poll_with_phase(
        &mut self,
        custody: &SpawnCustody,
        supervisor_until: Instant,
        cleanup_callback: bool,
    ) -> Result<Option<DeadlineGrantObservation>, ArenaError> {
        if Instant::now() >= self.process_cleanup_until {
            return Err(failure("cleanup_deadline"));
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
                response_until: None,
                written: 0,
                original_endpoint: false,
            });
            // The original owner must first observe and bind this actual
            // child; never derive its authority from the following wire data.
            return Ok(None);
        }

        let peer = self.peer.as_ref().ok_or_else(|| failure("peer_missing"))?;
        self.verify_peer(pid, peer.pid, peer.identity)?;
        let peer = self.peer.as_mut().ok_or_else(|| failure("peer_missing"))?;
        if cleanup_callback && peer.response.is_some() {
            let request = DeadlineRequest::decode(&peer.request).map_err(failure)?;
            let current_until = peer
                .response_until
                .ok_or_else(|| failure("response_deadline_missing"))?;
            let actual_until = phase_supervisor_limit(
                request.scope.phase,
                self.process_cleanup_until,
                supervisor_until,
                true,
            )
            .min(self.window.phase_deadline(request.scope.phase));
            if pending_frame_needs_rebuild(current_until, actual_until, peer.written)? {
                let encoded = self
                    .window
                    .grant(&request.scope, &peer.request)
                    .map_err(failure)?;
                let mut grant = DeadlineGrant::decode(&encoded).map_err(failure)?;
                let remaining = actual_until
                    .checked_duration_since(Instant::now())
                    .ok_or_else(|| failure("cleanup_deadline"))?;
                let remaining_ns = u64::try_from(remaining.as_nanos())
                    .map_err(|_| failure("cleanup_deadline_overflow"))?;
                grant.remaining_ns = grant.remaining_ns.min(remaining_ns);
                peer.response = Some(grant.encode().map_err(failure)?.to_vec());
                peer.response_until = Some(actual_until);
            }
        }
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
                .position(|(scope, admitted_pid, identity, _)| {
                    scope == &request.scope
                        && *admitted_pid == peer.pid
                        && *identity == peer.identity
                })
                .ok_or_else(|| failure("scope_not_admitted"))?;
            let (scope, _, _, original_endpoint) = self.admitted.remove(index);
            peer.original_endpoint = original_endpoint;
            // Reserve before writing: a partial or failed reply is not retried.
            self.used.push(scope.clone());
            let encoded = self.window.grant(&scope, &peer.request).map_err(failure)?;
            let mut grant = DeadlineGrant::decode(&encoded).map_err(failure)?;
            // Cleanup is bounded by the original supervisor's separately
            // registered cleanup deadline, not by its execution deadline.
            // The latter would falsely deny the cleanup reserve altogether.
            let supervisor_until = phase_supervisor_limit(
                scope.phase,
                self.process_cleanup_until,
                supervisor_until,
                cleanup_callback,
            );
            let supervisor_remaining = supervisor_until
                .checked_duration_since(Instant::now())
                .ok_or_else(|| failure("execution_deadline"))?;
            let supervisor_ns = u64::try_from(supervisor_remaining.as_nanos())
                .map_err(|_| failure("execution_deadline_overflow"))?;
            grant.remaining_ns = grant.remaining_ns.min(supervisor_ns);
            peer.response = Some(grant.encode().map_err(failure)?.to_vec());
            peer.response_until =
                Some(supervisor_until.min(self.window.phase_deadline(scope.phase)));
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
        if peer.original_endpoint && scope.phase == DeadlinePhase::Run {
            self.issued_original_runs.push(IssuedOriginalRun {
                scope: scope.clone(),
                pid: peer.pid,
                identity: peer.identity,
                cancel_claimed: false,
            });
        }
        Ok(Some(DeadlineGrantObservation {
            peer_pid: peer.pid,
            peer_start_ticks: peer.identity.start_ticks,
            scope,
            granted_until: peer
                .response_until
                .ok_or_else(|| failure("response_deadline_missing"))?,
            completion_observed_during_cleanup: cleanup_callback,
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

    /// Claim one cancel channel only after a complete run grant to the same
    /// original endpoint. This verifies the actual connected peer, not role
    /// values asserted by its wire messages. On error the caller retains the
    /// stream and the original owner; no socket or process is closed here.
    pub(crate) fn claim_cancel_peer(
        &mut self,
        custody: &SpawnCustody,
        stream: &UnixStream,
        scope: &DeadlineScope,
    ) -> Result<OriginalCancelPeer, ArenaError> {
        if scope.phase != DeadlinePhase::Run {
            return Err(failure("cancel_scope_phase"));
        }
        let index = self
            .issued_original_runs
            .iter()
            .position(|issued| &issued.scope == scope && !issued.cancel_claimed)
            .ok_or_else(|| failure("cancel_scope_not_issued"))?;
        let issued = &self.issued_original_runs[index];
        let peer = OriginalCancelPeer {
            scope: issued.scope.clone(),
            pid: issued.pid,
            identity: issued.identity,
        };
        self.verify_cancel_peer(custody, stream, &peer)?;
        self.issued_original_runs[index].cancel_claimed = true;
        Ok(peer)
    }

    /// Recheck the exact socket peer, retained grant owner and direct process
    /// relationship. A valid peer is not an ACK or proof of physical release.
    pub(crate) fn verify_cancel_peer(
        &self,
        custody: &SpawnCustody,
        stream: &UnixStream,
        peer: &OriginalCancelPeer,
    ) -> Result<(), ArenaError> {
        if Instant::now() >= self.process_cleanup_until
            || !self.issued_original_runs.iter().any(|issued| {
                issued.scope == peer.scope
                    && issued.pid == peer.pid
                    && issued.identity == peer.identity
            })
        {
            return Err(failure("cancel_scope_not_owned"));
        }
        let (leader, original_leader) =
            self.fastchess.ok_or_else(|| failure("fastchess_custody"))?;
        if custody.child().map(|child| child.id()) != Some(leader)
            || process_identity(leader)? != original_leader
        {
            return Err(failure("fastchess_identity_changed"));
        }
        let credentials =
            getsockopt(stream, PeerCredentials).map_err(|_| failure("cancel_peer_credentials"))?;
        if credentials.uid() != nix::unistd::geteuid().as_raw()
            || credentials.pid() <= 0
            || u32::try_from(credentials.pid()).ok() != Some(peer.pid)
        {
            return Err(failure("cancel_peer_credentials"));
        }
        self.verify_peer(leader, peer.pid, peer.identity)?;
        if process_identity(leader)? != original_leader {
            return Err(failure("fastchess_identity_changed"));
        }
        Ok(())
    }
}

impl DeadlineControl<'static> {
    /// The caller retains these exact socket/window/file objects in the same
    /// original scope on unknown physical completion. No endpoint authority or
    /// cancellation acknowledgement is created by this constructor.
    pub(crate) fn new_owned(
        listener: &Arc<UnixListener>,
        window: &Arc<OwnerWindow>,
        proxy_pin: &Arc<File>,
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
            listener: OriginalResource::Owned(Arc::clone(listener)),
            window: OriginalResource::Owned(Arc::clone(window)),
            proxy_pin: OriginalResource::Owned(Arc::clone(proxy_pin)),
            owner_id,
            owner_epoch,
            process_cleanup_until,
            fastchess: None,
            admitted: Vec::new(),
            used: Vec::new(),
            issued_original_runs: Vec::new(),
            peer: None,
        })
    }
}

fn pending_frame_needs_rebuild(
    old_until: Instant,
    actual_until: Instant,
    written: usize,
) -> Result<bool, ArenaError> {
    if actual_until >= old_until {
        return Ok(false);
    }
    if written != 0 {
        return Err(failure("response_deadline_changed_after_partial_write"));
    }
    Ok(true)
}

fn phase_supervisor_limit(
    phase: DeadlinePhase,
    original_cleanup_until: Instant,
    callback_until: Instant,
    cleanup_callback: bool,
) -> Instant {
    if phase == DeadlinePhase::Cleanup {
        if cleanup_callback {
            original_cleanup_until.min(callback_until)
        } else {
            original_cleanup_until
        }
    } else {
        callback_until
    }
}

fn decode_sha256(text: &str) -> Result<[u8; 32], ArenaError> {
    if text.len() != 64 {
        return Err(failure("lock_digest"));
    }
    let nibble = |byte: u8| match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    };
    let mut result = [0; 32];
    for (index, pair) in text.as_bytes().chunks_exact(2).enumerate() {
        let high = nibble(pair[0]).ok_or_else(|| failure("lock_digest"))?;
        let low = nibble(pair[1]).ok_or_else(|| failure("lock_digest"))?;
        result[index] = (high << 4) | low;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn cleanup_reserve_is_not_cut_by_run_but_obeys_actual_cleanup_callback() {
        let now = Instant::now();
        let run = now + Duration::from_secs(5);
        let actual_cleanup = now + Duration::from_secs(7);
        let original_cleanup = now + Duration::from_secs(10);
        assert_eq!(
            phase_supervisor_limit(DeadlinePhase::Cleanup, original_cleanup, run, false),
            original_cleanup
        );
        assert_eq!(
            phase_supervisor_limit(
                DeadlinePhase::Cleanup,
                original_cleanup,
                actual_cleanup,
                true
            ),
            actual_cleanup
        );
        assert_eq!(
            phase_supervisor_limit(
                DeadlinePhase::Cleanup,
                actual_cleanup,
                original_cleanup,
                true
            ),
            actual_cleanup
        );
        assert_eq!(
            phase_supervisor_limit(DeadlinePhase::Run, original_cleanup, run, false),
            run
        );
    }

    #[test]
    fn shortening_a_partial_frame_fails_without_reissuing_it() {
        let now = Instant::now();
        let old = now + Duration::from_secs(10);
        let shorter = now + Duration::from_secs(5);
        assert!(pending_frame_needs_rebuild(old, shorter, 0).unwrap());
        assert!(pending_frame_needs_rebuild(old, shorter, 1).is_err());
        assert!(!pending_frame_needs_rebuild(old, old, 1).unwrap());
        assert!(!pending_frame_needs_rebuild(shorter, old, 1).unwrap());
    }
}

#[cfg(all(test, target_os = "linux"))]
#[path = "deadline_socket_tests.rs"]
mod socket_tests;
