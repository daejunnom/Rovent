//! Direct-child NoGame smoke deadline control. This does not authenticate a
//! Fastchess game, prove CUDA completion, or change the exact-spawn control.

use super::super::direct_smoke_launch::NoGameBindingPermit;
use super::super::{
    LockedPalsSystemArenaPairV6, NoGameExecutionStartV1, NoGameRoleSocketsV1, PalsV6ExecutionStart,
    PalsV6RoleSockets, ProxyEntrySmokeSnapshotOwnerV1,
    proxy_entry_smoke::ProxyEntryFunctionalSmokeOwnerV1,
};
use super::{
    ProcessIdentity, decode_sha256, failure, pending_frame_needs_rebuild, process_identity,
};
use crate::{ArenaError, NativeLaunchOwner, native_launch::linux, process::SpawnCustody};
use nix::{
    errno::Errno,
    sys::{
        socket::{getsockopt, sockopt::PeerCredentials},
        wait::{Id, WaitPidFlag, WaitStatus, waitid},
    },
    unistd::Pid,
};
use rz_contracts::pals::{
    deadline_grant_v1::DeadlineRole,
    proxy_entry_smoke_grant_v1::{
        PROXY_ENTRY_SMOKE_CHALLENGE_BYTES, PROXY_ENTRY_SMOKE_GRANT_BYTES,
        PROXY_ENTRY_SMOKE_REQUEST_BYTES, ProxyEntrySmokeChallenge, ProxyEntrySmokeGrant,
        ProxyEntrySmokePhase, ProxyEntrySmokePolicy, ProxyEntrySmokeRequest, ProxyEntrySmokeScope,
    },
};
use rz_experiments::NativeEngineRole;
use std::{
    fs::{self, File},
    io::{self, Read, Write},
    os::unix::{
        fs::MetadataExt,
        net::{UnixListener, UnixStream},
    },
    sync::{Arc, Mutex},
    time::Instant,
};

#[path = "smoke_cancel_sender.rs"]
mod smoke_cancel_sender;
pub(crate) use smoke_cancel_sender::SmokeCancelProgress;

struct SmokePeer {
    stream: UnixStream,
    challenge_written: usize,
    request: [u8; PROXY_ENTRY_SMOKE_REQUEST_BYTES],
    read: usize,
    response: Option<[u8; PROXY_ENTRY_SMOKE_GRANT_BYTES]>,
    response_until: Option<Instant>,
    written: usize,
}

/// Validate the same request identity contract before any input is copied.
pub(crate) fn validate_smoke_request_ids(request_ids: [u64; 2]) -> Result<(), ArenaError> {
    if request_ids.contains(&0) || request_ids[0] == request_ids[1] {
        return Err(failure("smoke_request_ids"));
    }
    Ok(())
}

/// Frame delivery only; no process/physical completion is inferred.
pub(crate) struct SmokeGrantObservation {
    pub scope: ProxyEntrySmokeScope,
    pub peer_pid: u32,
    pub peer_start_ticks: u64,
    pub granted_until: Instant,
    pub completion_observed_during_cleanup: bool,
}

/// Byte offsets and first error are diagnostic observations, never a fence.
pub(crate) struct SmokeControlProgress {
    pub challenge_written: Option<usize>,
    pub request_read: Option<usize>,
    pub grant_written: Option<usize>,
    pub retained_grants: usize,
    pub interrupted_grants: usize,
    pub no_game_run_closed: bool,
    pub first_failure: Option<String>,
}

/// Retain this exact owner and any partial stream on failure/unknown closure.
pub(crate) struct SmokeDeadlineControl {
    binding: SmokeBinding,
    listener: Arc<UnixListener>,
    role: NativeEngineRole,
    proxy_pin: Arc<File>,
    scopes: [ProxyEntrySmokeScope; 2],
    challenge: [u8; PROXY_ENTRY_SMOKE_CHALLENGE_BYTES],
    used: [bool; 2],
    child: Option<(u32, ProcessIdentity)>,
    peer: Option<SmokePeer>,
    retained_grants: Vec<SmokePeer>,
    // A stream interrupted by NoGame stop remains owned even when its phase
    // or physical delivery is unknown; it is never resumed as a new Run.
    interrupted_grants: Vec<SmokePeer>,
    no_game_run_closed: bool,
    first_failure: Option<String>,
    cancel_peer: Option<smoke_cancel_sender::SmokeCancelPeer>,
    cancel_failure: Option<Arc<ArenaError>>,
    // Drop after listener and every retained/partial stream.
    _retained_permit: Option<Arc<NoGameBindingPermit>>,
}

enum SmokeBinding {
    Paired {
        smoke: Arc<Mutex<ProxyEntryFunctionalSmokeOwnerV1>>,
        identity: Arc<()>,
        execution: Arc<PalsV6ExecutionStart>,
        sockets: Arc<PalsV6RoleSockets>,
    },
    NoGame {
        owner: Arc<ProxyEntrySmokeSnapshotOwnerV1>,
        execution: Arc<NoGameExecutionStartV1>,
        sockets: Arc<NoGameRoleSocketsV1>,
    },
}

impl SmokeBinding {
    fn run_until(&self) -> Instant {
        match self {
            Self::Paired { execution, .. } => execution.run_until(),
            Self::NoGame { execution, .. } => execution.run_until(),
        }
    }

    fn cleanup_until(&self) -> Instant {
        match self {
            Self::Paired { execution, .. } => execution.cleanup_until(),
            Self::NoGame { execution, .. } => execution.cleanup_until(),
        }
    }

    fn verify_grant_listener(&self, role: NativeEngineRole) -> Result<(), ArenaError> {
        match self {
            Self::Paired { sockets, .. } => sockets.verify_role_grant_listener(role),
            Self::NoGame { sockets, .. } if sockets.role() == role => {
                sockets.verify_grant_listener()
            }
            Self::NoGame { .. } => Err(failure("smoke_role_changed")),
        }
    }
}

impl SmokeDeadlineControl {
    /// Construct from the same independent NoGame owner, without admitting a
    /// paired plan or treating role/channel names as peer or spawn evidence.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_no_game(
        owner: &Arc<ProxyEntrySmokeSnapshotOwnerV1>,
        execution: &Arc<NoGameExecutionStartV1>,
        sockets: &Arc<NoGameRoleSocketsV1>,
        proxy_pin: &Arc<File>,
        permit: &Arc<NoGameBindingPermit>,
        owner_id: [u8; 16],
        owner_epoch: u64,
        request_ids: [u64; 2],
    ) -> Result<Self, ArenaError> {
        validate_smoke_request_ids(request_ids)?;
        if !Arc::ptr_eq(&owner.execution, execution)
            || !execution.matches(&owner.plan, owner_id, owner_epoch)
            || !permit.matches(owner)
        {
            return Err(failure("smoke_original_execution_differs"));
        }
        execution.require_running()?;
        sockets.verify_original(owner, execution)?;
        let plan = owner.plan.input();
        let endpoint_pin = linux::pin(&owner.pins, &plan.proxy_binary)?;
        let original = endpoint_pin
            .file
            .metadata()
            .map_err(|_| failure("smoke_proxy_pin"))?;
        let actual = proxy_pin
            .metadata()
            .map_err(|_| failure("smoke_proxy_pin"))?;
        if !actual.is_file() || (original.dev(), original.ino()) != (actual.dev(), actual.ino()) {
            return Err(failure("smoke_proxy_pin_differs"));
        }
        let lock_sha256 = decode_sha256(&plan.endpoint.startup_lock.canonical_sha256)?;
        let role = owner.role;
        let role_wire = match role {
            NativeEngineRole::Baseline => DeadlineRole::Baseline,
            NativeEngineRole::Candidate => DeadlineRole::Candidate,
        };
        let scopes = [ProxyEntrySmokePhase::Run, ProxyEntrySmokePhase::Cleanup].map(|phase| {
            ProxyEntrySmokeScope {
                policy: ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1,
                owner_id,
                owner_epoch,
                role: role_wire,
                generation: plan.endpoint.startup_lock.config.process_generation,
                request_id: request_ids[match phase {
                    ProxyEntrySmokePhase::Run => 0,
                    ProxyEntrySmokePhase::Cleanup => 1,
                }],
                lock_sha256,
                phase,
            }
        });
        for scope in &scopes {
            scope.validate().map_err(failure)?;
        }
        let challenge = ProxyEntrySmokeChallenge {
            run: scopes[0].clone(),
            cleanup: scopes[1].clone(),
        }
        .encode()
        .map_err(failure)?;
        execution.require_running()?;
        Ok(Self {
            binding: SmokeBinding::NoGame {
                owner: Arc::clone(owner),
                execution: Arc::clone(execution),
                sockets: Arc::clone(sockets),
            },
            listener: Arc::clone(sockets.listener(false)),
            role,
            proxy_pin: Arc::clone(proxy_pin),
            scopes,
            challenge,
            used: [false; 2],
            child: None,
            peer: None,
            retained_grants: Vec::new(),
            interrupted_grants: Vec::new(),
            no_game_run_closed: false,
            first_failure: None,
            cancel_peer: None,
            cancel_failure: None,
            _retained_permit: Some(Arc::clone(permit)),
        })
    }

    /// The request IDs are explicitly registered by this original launcher;
    /// they identify requests and are not observations of a process or game.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        owner: &NativeLaunchOwner<LockedPalsSystemArenaPairV6>,
        smoke: &Arc<Mutex<ProxyEntryFunctionalSmokeOwnerV1>>,
        execution: &Arc<PalsV6ExecutionStart>,
        sockets: &Arc<PalsV6RoleSockets>,
        proxy_pin: &Arc<File>,
        role: NativeEngineRole,
        owner_id: [u8; 16],
        owner_epoch: u64,
        request_ids: [u64; 2],
    ) -> Result<Self, ArenaError> {
        validate_smoke_request_ids(request_ids)?;
        let smoke_guard = smoke.try_lock().map_err(|_| failure("smoke_owner_busy"))?;
        smoke_guard.verify_original_binding(owner, execution, role, owner_id, owner_epoch)?;
        sockets.verify_role_socket_owner(owner, execution, role)?;
        // The smoke owner's verification above already checked this same pin;
        // look it up without repeating the complete snapshot read.
        let sidecar = owner
            .spec()
            .proxy_inputs()
            .ok_or_else(|| failure("smoke_proxy_sidecar"))?;
        let endpoint_pin = linux::pin(&owner.snapshot.pins, &sidecar.proxy_binary)?;
        let original = endpoint_pin
            .file
            .metadata()
            .map_err(|_| failure("smoke_proxy_pin"))?;
        let actual = proxy_pin
            .metadata()
            .map_err(|_| failure("smoke_proxy_pin"))?;
        if !actual.is_file() || (original.dev(), original.ino()) != (actual.dev(), actual.ino()) {
            return Err(failure("smoke_proxy_pin_differs"));
        }
        let endpoint = owner.spec().endpoint(role);
        let lock_sha256 = decode_sha256(&endpoint.startup_lock.canonical_sha256)?;
        let role_wire = match role {
            NativeEngineRole::Baseline => DeadlineRole::Baseline,
            NativeEngineRole::Candidate => DeadlineRole::Candidate,
        };
        let scopes = [ProxyEntrySmokePhase::Run, ProxyEntrySmokePhase::Cleanup].map(|phase| {
            ProxyEntrySmokeScope {
                policy: ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1,
                owner_id,
                owner_epoch,
                role: role_wire,
                generation: endpoint.startup_lock.config.process_generation,
                request_id: request_ids[match phase {
                    ProxyEntrySmokePhase::Run => 0,
                    ProxyEntrySmokePhase::Cleanup => 1,
                }],
                lock_sha256,
                phase,
            }
        });
        for scope in &scopes {
            scope.validate().map_err(failure)?;
        }
        let challenge = ProxyEntrySmokeChallenge {
            run: scopes[0].clone(),
            cleanup: scopes[1].clone(),
        }
        .encode()
        .map_err(failure)?;
        let smoke_identity = Arc::clone(smoke_guard.identity());
        drop(smoke_guard);
        Ok(Self {
            binding: SmokeBinding::Paired {
                smoke: Arc::clone(smoke),
                identity: smoke_identity,
                execution: Arc::clone(execution),
                sockets: Arc::clone(sockets),
            },
            listener: Arc::clone(sockets.listener(role, false)),
            role,
            proxy_pin: Arc::clone(proxy_pin),
            scopes,
            challenge,
            used: [false; 2],
            child: None,
            peer: None,
            retained_grants: Vec::new(),
            interrupted_grants: Vec::new(),
            no_game_run_closed: false,
            first_failure: None,
            cancel_peer: None,
            cancel_failure: None,
            _retained_permit: None,
        })
    }

    /// Scope data is a consistency declaration, never an authenticated launch
    /// or peer; a separate original channel must deliver it to the child.
    pub(crate) fn scopes(&self) -> &[ProxyEntrySmokeScope; 2] {
        &self.scopes
    }

    pub(crate) fn progress(&self) -> SmokeControlProgress {
        SmokeControlProgress {
            challenge_written: self.peer.as_ref().map(|peer| peer.challenge_written),
            request_read: self.peer.as_ref().map(|peer| peer.read),
            grant_written: self.peer.as_ref().map(|peer| peer.written),
            retained_grants: self.retained_grants.len(),
            interrupted_grants: self.interrupted_grants.len(),
            no_game_run_closed: self.no_game_run_closed,
            first_failure: self.first_failure.clone(),
        }
    }

    pub(crate) fn poll_run(
        &mut self,
        custody: &SpawnCustody,
        until: Instant,
    ) -> Result<Option<SmokeGrantObservation>, ArenaError> {
        self.poll(custody, until, false)
    }
    pub(crate) fn poll_cleanup(
        &mut self,
        custody: &SpawnCustody,
        until: Instant,
    ) -> Result<Option<SmokeGrantObservation>, ArenaError> {
        self.poll(custody, until, true)
    }

    fn poll(
        &mut self,
        custody: &SpawnCustody,
        until: Instant,
        cleanup: bool,
    ) -> Result<Option<SmokeGrantObservation>, ArenaError> {
        if cleanup && matches!(&self.binding, SmokeBinding::NoGame { .. }) {
            self.close_no_game_run();
        }
        if let Some(error) = &self.first_failure {
            return Err(ArenaError::Io(format!("smoke_control_sticky:{error}")));
        }
        let result = self.poll_inner(custody, until, cleanup);
        if let Err(error) = &result {
            self.first_failure = Some(error.to_string());
        }
        result
    }

    fn poll_inner(
        &mut self,
        custody: &SpawnCustody,
        callback_until: Instant,
        cleanup: bool,
    ) -> Result<Option<SmokeGrantObservation>, ArenaError> {
        let _smoke_guard = match &self.binding {
            SmokeBinding::Paired {
                smoke, identity, ..
            } => {
                let guard = match smoke.try_lock() {
                    Ok(owner) => owner,
                    Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                    Err(std::sync::TryLockError::Poisoned(_)) => {
                        return Err(failure("smoke_owner_poisoned"));
                    }
                };
                if !Arc::ptr_eq(guard.identity(), identity) {
                    return Err(failure("smoke_owner_replaced"));
                }
                Some(guard)
            }
            SmokeBinding::NoGame {
                owner,
                execution,
                sockets,
                ..
            } => {
                if !Arc::ptr_eq(&owner.execution, execution)
                    || sockets.role() != owner.role
                    || !execution.matches_plan(&owner.plan)
                {
                    return Err(failure("smoke_owner_replaced"));
                }
                None
            }
        };
        if Instant::now() >= self.binding.cleanup_until() {
            return Err(failure("smoke_original_deadline"));
        }
        let pid = custody
            .child()
            .ok_or_else(|| failure("smoke_child_custody"))?
            .id();
        let raw = i32::try_from(pid).map_err(|_| failure("smoke_child_pid"))?;
        match waitid(
            Id::Pid(Pid::from_raw(raw)),
            WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT,
        ) {
            Ok(WaitStatus::StillAlive) => {}
            Err(Errno::EINTR) => return Ok(None),
            _ => return Err(failure("smoke_child_not_live_or_owned")),
        }
        let identity = process_identity(pid)?;
        if identity.parent != std::process::id() || identity.group != pid {
            return Err(failure("smoke_child_relation"));
        }
        if let Some((original_pid, original)) = self.child {
            if original_pid != pid || original != identity {
                return Err(failure("smoke_child_identity_changed"));
            }
        } else {
            self.child = Some((pid, identity));
        }
        let expected = self
            .proxy_pin
            .metadata()
            .map_err(|_| failure("smoke_proxy_pin"))?;
        let image =
            fs::metadata(format!("/proc/{pid}/exe")).map_err(|_| failure("smoke_proxy_image"))?;
        if !expected.is_file()
            || (expected.dev(), expected.ino()) != (image.dev(), image.ino())
            || process_identity(pid)? != identity
        {
            return Err(failure("smoke_proxy_image_differs"));
        }
        self.binding.verify_grant_listener(self.role)?;

        if self.peer.is_none() {
            let (stream, _) = match self.listener.accept() {
                Ok(value) => value,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(None),
                Err(_) => return Err(failure("smoke_accept")),
            };
            // Install before any fallible operation: rejected/partial streams
            // remain attached to the same owner instead of being discarded.
            self.peer = Some(SmokePeer {
                stream,
                challenge_written: 0,
                request: [0; PROXY_ENTRY_SMOKE_REQUEST_BYTES],
                read: 0,
                response: None,
                response_until: None,
                written: 0,
            });
            self.peer
                .as_ref()
                .unwrap()
                .stream
                .set_nonblocking(true)
                .map_err(|_| failure("smoke_peer_nonblocking"))?;
        }
        let run_until = self.binding.run_until();
        let cleanup_until = self.binding.cleanup_until();
        let peer = self
            .peer
            .as_mut()
            .ok_or_else(|| failure("smoke_peer_missing"))?;
        let credentials = getsockopt(&peer.stream, PeerCredentials)
            .map_err(|_| failure("smoke_peer_credentials"))?;
        if credentials.uid() != nix::unistd::geteuid().as_raw()
            || u32::try_from(credentials.pid()).ok() != Some(pid)
            || process_identity(pid)? != identity
        {
            return Err(failure("smoke_peer_differs"));
        }

        if peer.challenge_written < self.challenge.len() {
            let challenge_until = if cleanup {
                cleanup_until.min(callback_until)
            } else {
                run_until.min(callback_until)
            };
            if Instant::now() >= challenge_until {
                return Err(failure("smoke_challenge_deadline"));
            }
            match peer.stream.write(&self.challenge[peer.challenge_written..]) {
                Ok(0) => return Err(failure("smoke_challenge_write_zero")),
                Ok(n) => peer.challenge_written += n,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) =>
                {
                    return Ok(None);
                }
                Err(_) => return Err(failure("smoke_challenge_write")),
            }
            if peer.challenge_written < self.challenge.len() {
                return Ok(None);
            }
        }

        if let Some(old_until) = peer.response_until {
            let scope = ProxyEntrySmokeRequest::decode(&peer.request)
                .map_err(failure)?
                .scope;
            if self.no_game_run_closed && scope.phase == ProxyEntrySmokePhase::Run {
                if self.interrupted_grants.len() >= self.scopes.len() {
                    return Err(failure("smoke_interrupted_grant_capacity"));
                }
                let interrupted = self
                    .peer
                    .take()
                    .ok_or_else(|| failure("smoke_peer_missing"))?;
                self.interrupted_grants.push(interrupted);
                return Ok(None);
            }
            let deadline = phase_deadline(
                scope.phase,
                run_until,
                cleanup_until,
                callback_until,
                cleanup,
            );
            if pending_frame_needs_rebuild(old_until, deadline, peer.written)? {
                let response = make_grant(&scope, deadline)?;
                peer.response = Some(response);
                peer.response_until = Some(deadline);
            }
        }
        if peer.response.is_none() {
            match peer.stream.read(&mut peer.request[peer.read..]) {
                Ok(0) => return Err(failure("smoke_peer_eof")),
                Ok(n) => peer.read += n,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) =>
                {
                    return Ok(None);
                }
                Err(_) => return Err(failure("smoke_peer_read")),
            }
            if peer.read < PROXY_ENTRY_SMOKE_REQUEST_BYTES {
                return Ok(None);
            }
            let scope = ProxyEntrySmokeRequest::decode(&peer.request)
                .map_err(failure)?
                .scope;
            if self.no_game_run_closed && scope.phase == ProxyEntrySmokePhase::Run {
                if self.interrupted_grants.len() >= self.scopes.len() {
                    return Err(failure("smoke_interrupted_grant_capacity"));
                }
                let interrupted = self
                    .peer
                    .take()
                    .ok_or_else(|| failure("smoke_peer_missing"))?;
                self.interrupted_grants.push(interrupted);
                return Ok(None);
            }
            let index = self
                .scopes
                .iter()
                .position(|expected| *expected == scope)
                .ok_or_else(|| failure("smoke_scope_not_admitted"))?;
            if self.used[index] {
                return Err(failure("smoke_scope_reused"));
            }
            self.used[index] = true;
            let deadline = phase_deadline(
                scope.phase,
                run_until,
                cleanup_until,
                callback_until,
                cleanup,
            );
            peer.response = Some(make_grant(&scope, deadline)?);
            peer.response_until = Some(deadline);
        }
        if peer
            .response_until
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(failure("smoke_response_deadline"));
        }
        let response = peer
            .response
            .as_ref()
            .ok_or_else(|| failure("smoke_response_missing"))?;
        match peer.stream.write(&response[peer.written..]) {
            Ok(0) => return Err(failure("smoke_write_zero")),
            Ok(n) => peer.written += n,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                return Ok(None);
            }
            Err(_) => return Err(failure("smoke_write")),
        }
        if peer.written < response.len() {
            return Ok(None);
        }
        let completed = self
            .peer
            .take()
            .ok_or_else(|| failure("smoke_peer_missing"))?;
        let scope = ProxyEntrySmokeRequest::decode(&completed.request)
            .map_err(failure)?
            .scope;
        let observation = SmokeGrantObservation {
            scope,
            peer_pid: pid,
            peer_start_ticks: identity.start_ticks,
            granted_until: completed
                .response_until
                .ok_or_else(|| failure("smoke_deadline_missing"))?,
            completion_observed_during_cleanup: cleanup,
        };
        self.retained_grants.push(completed);
        Ok(Some(observation))
    }

    fn close_no_game_run(&mut self) {
        if self.no_game_run_closed {
            return;
        }
        self.no_game_run_closed = true;
        let known_cleanup = self.peer.as_ref().is_some_and(|peer| {
            peer.read == PROXY_ENTRY_SMOKE_REQUEST_BYTES
                && ProxyEntrySmokeRequest::decode(&peer.request)
                    .is_ok_and(|request| request.scope == self.scopes[1])
        });
        if !known_cleanup {
            if let Some(peer) = self.peer.take() {
                self.interrupted_grants.push(peer);
            }
        }
    }

    /// Offsets are transport observations; a partially written grant has
    /// unknown delivery, even if its requested phase can be decoded.
    pub(crate) fn no_game_interrupted_grants(
        &self,
    ) -> impl Iterator<Item = (usize, usize, usize, Option<ProxyEntrySmokePhase>)> + '_ {
        self.interrupted_grants.iter().map(|peer| {
            let phase = if peer.read == PROXY_ENTRY_SMOKE_REQUEST_BYTES {
                ProxyEntrySmokeRequest::decode(&peer.request)
                    .ok()
                    .map(|request| request.scope.phase)
            } else {
                None
            };
            (peer.challenge_written, peer.read, peer.written, phase)
        })
    }
}

fn phase_deadline(
    phase: ProxyEntrySmokePhase,
    run_until: Instant,
    cleanup_until: Instant,
    callback_until: Instant,
    cleanup: bool,
) -> Instant {
    match phase {
        ProxyEntrySmokePhase::Run => run_until.min(callback_until),
        ProxyEntrySmokePhase::Cleanup if cleanup => cleanup_until.min(callback_until),
        ProxyEntrySmokePhase::Cleanup => cleanup_until,
    }
}

fn make_grant(
    scope: &ProxyEntrySmokeScope,
    deadline: Instant,
) -> Result<[u8; PROXY_ENTRY_SMOKE_GRANT_BYTES], ArenaError> {
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .ok_or_else(|| failure("smoke_deadline"))?;
    let remaining_ns =
        u64::try_from(remaining.as_nanos()).map_err(|_| failure("smoke_deadline_overflow"))?;
    ProxyEntrySmokeGrant {
        scope: scope.clone(),
        remaining_ns,
    }
    .encode()
    .map_err(failure)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn smoke_phase_bounds_use_original_run_and_cleanup() {
        let now = Instant::now();
        let run = now + Duration::from_secs(60);
        let cleanup = now + Duration::from_secs(70);
        let callback = now + Duration::from_secs(30);
        assert_eq!(
            phase_deadline(ProxyEntrySmokePhase::Run, run, cleanup, callback, false),
            callback
        );
        assert_eq!(
            phase_deadline(ProxyEntrySmokePhase::Cleanup, run, cleanup, callback, false),
            cleanup
        );
        assert_eq!(
            phase_deadline(ProxyEntrySmokePhase::Cleanup, run, cleanup, callback, true),
            callback
        );
    }

    #[test]
    fn smoke_grant_keeps_no_game_scope_and_finite_deadline() {
        let scope = ProxyEntrySmokeScope {
            policy: ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1,
            owner_id: [1; 16],
            owner_epoch: 1,
            role: DeadlineRole::Baseline,
            generation: 1,
            request_id: 1,
            lock_sha256: [2; 32],
            phase: ProxyEntrySmokePhase::Run,
        };
        let deadline = Instant::now() + Duration::from_secs(60);
        let grant = ProxyEntrySmokeGrant::decode(&make_grant(&scope, deadline).unwrap()).unwrap();
        assert_eq!(grant.scope, scope);
        assert!(
            grant.remaining_ns > 0
                && grant.remaining_ns <= Duration::from_secs(60).as_nanos() as u64
        );
        assert!(make_grant(&scope, Instant::now() - Duration::from_secs(1)).is_err());
    }
}
