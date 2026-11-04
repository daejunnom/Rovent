//! NoGame functional-smoke grant exchange with the actual direct parent.
//! This retains the exact socket and partial bytes; no Fastchess/game scope,
//! role-listener provenance, later liveness or GPU completion is inferred.

use super::arena_grant_channel::{ProcessIdentity, process_identity};
use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
use rz_contracts::pals::proxy_entry_smoke_grant_v1::{
    PROXY_ENTRY_SMOKE_CHALLENGE_BYTES, PROXY_ENTRY_SMOKE_GRANT_BYTES,
    PROXY_ENTRY_SMOKE_REQUEST_BYTES, ProxyEntrySmokeChallenge, ProxyEntrySmokeGrant,
    ProxyEntrySmokePhase, ProxyEntrySmokePolicy, ProxyEntrySmokeRequest, ProxyEntrySmokeScope,
};
use std::{
    io::{self, Read, Write},
    os::unix::{fs::MetadataExt, net::UnixStream},
    time::{Duration, Instant},
};

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Observation of the live direct arena parent from this exact socket. It is
/// not a peer assertion carried in a wire frame or a role-listener proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OriginalDirectPeer {
    pid: u32,
    uid: u32,
    gid: u32,
    proxy: ProcessIdentity,
    arena: ProcessIdentity,
}

fn observe_direct_parent(stream: &UnixStream) -> io::Result<OriginalDirectPeer> {
    let credentials = getsockopt(stream, PeerCredentials).map_err(io::Error::other)?;
    let pid = u32::try_from(credentials.pid()).map_err(|_| invalid("smoke arena peer pid"))?;
    if pid == 0 {
        return Err(invalid("smoke arena peer pid"));
    }
    let proxy = process_identity(std::process::id())?;
    if proxy.parent != pid {
        return Err(invalid("smoke arena is not direct parent"));
    }
    let arena = process_identity(pid)?;
    if std::fs::metadata(format!("/proc/{pid}"))?.uid() != credentials.uid() {
        return Err(invalid("smoke arena peer uid differs"));
    }
    if process_identity(std::process::id())? != proxy || process_identity(pid)? != arena {
        return Err(invalid("smoke arena parent identity changed"));
    }
    Ok(OriginalDirectPeer {
        pid,
        uid: credentials.uid(),
        gid: credentials.gid(),
        proxy,
        arena,
    })
}

/// Expected identity supplied by the caller from its original smoke owner and
/// sidecar. It is not established by the challenge or an arbitrary CLI value.
pub(super) struct ExpectedSmokeContext {
    pub policy: ProxyEntrySmokePolicy,
    pub owner_id: [u8; 16],
    pub owner_epoch: u64,
    pub role: rz_contracts::pals::deadline_grant_v1::DeadlineRole,
    pub generation: u64,
    pub lock_sha256: [u8; 32],
}

impl ExpectedSmokeContext {
    fn matches(&self, challenge: &ProxyEntrySmokeChallenge) -> bool {
        let scope = &challenge.run;
        scope.policy == self.policy
            && scope.owner_id == self.owner_id
            && scope.owner_epoch == self.owner_epoch
            && scope.role == self.role
            && scope.generation == self.generation
            && scope.lock_sha256 == self.lock_sha256
    }
}

/// A received challenge stays tied to the direct peer observation. It is
/// still not role-listener authorization or proof that the peer stays alive.
pub(super) struct ObservedSmokeChallenge {
    challenge: ProxyEntrySmokeChallenge,
    peer: OriginalDirectPeer,
}

pub(super) enum SmokeChallengePoll {
    Pending,
    Complete,
}

/// Retains the original stream, partial frame and first failure. The caller is
/// responsible for finite task cleanup and separate source/sidecar provenance.
pub(super) struct PendingDirectSmokeChallenge {
    stream: Option<UnixStream>,
    peer: OriginalDirectPeer,
    frame: [u8; PROXY_ENTRY_SMOKE_CHALLENGE_BYTES],
    read: usize,
    challenge: Option<ProxyEntrySmokeChallenge>,
    until: Instant,
    failure: Option<(io::ErrorKind, String, Option<i32>)>,
}

impl PendingDirectSmokeChallenge {
    pub(super) fn begin(
        stream: UnixStream,
        until: Instant,
    ) -> Result<Self, (io::Error, UnixStream)> {
        if Instant::now() >= until {
            return Err((
                io::Error::new(io::ErrorKind::TimedOut, "smoke challenge expired"),
                stream,
            ));
        }
        let peer = match observe_direct_parent(&stream) {
            Ok(peer) => peer,
            Err(error) => return Err((error, stream)),
        };
        if let Err(error) = stream.set_nonblocking(true) {
            return Err((error, stream));
        }
        Ok(Self {
            stream: Some(stream),
            peer,
            frame: [0; PROXY_ENTRY_SMOKE_CHALLENGE_BYTES],
            read: 0,
            challenge: None,
            until,
            failure: None,
        })
    }
    fn fail(&mut self, error: io::Error) -> io::Error {
        if self.failure.is_none() {
            self.failure = Some((error.kind(), error.to_string(), error.raw_os_error()));
        }
        error
    }
    fn reject(mut self, error: io::Error) -> (io::Error, Self) {
        let error = self.fail(error);
        (error, self)
    }
    pub(super) fn first_failure(&self) -> Option<(io::ErrorKind, &str, Option<i32>)> {
        self.failure
            .as_ref()
            .map(|(kind, message, raw)| (*kind, message.as_str(), *raw))
    }
    pub(super) fn poll(&mut self) -> io::Result<SmokeChallengePoll> {
        if let Some((kind, message, _)) = &self.failure {
            return Err(io::Error::new(*kind, message.clone()));
        }
        if self.challenge.is_some() {
            return Ok(SmokeChallengePoll::Complete);
        }
        if Instant::now() >= self.until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "smoke challenge expired",
            )));
        }
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| invalid("original smoke challenge stream missing"))?;
        match observe_direct_parent(stream) {
            Ok(peer) if peer == self.peer => {}
            Ok(_) => return Err(self.fail(invalid("smoke challenge peer changed"))),
            Err(error) => return Err(self.fail(error)),
        }
        if Instant::now() >= self.until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "smoke challenge expired",
            )));
        }
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| invalid("original smoke challenge stream missing"))?;
        match stream.read(&mut self.frame[self.read..]) {
            Ok(0) => {
                return Err(self.fail(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "smoke challenge truncated",
                )));
            }
            Ok(n) => self.read += n,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                return Ok(SmokeChallengePoll::Pending);
            }
            Err(error) => return Err(self.fail(error)),
        }
        if self.read < PROXY_ENTRY_SMOKE_CHALLENGE_BYTES {
            return Ok(SmokeChallengePoll::Pending);
        }
        if Instant::now() >= self.until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "smoke challenge expired",
            )));
        }
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| invalid("original smoke challenge stream missing"))?;
        match observe_direct_parent(stream) {
            Ok(peer) if peer == self.peer => {}
            Ok(_) => return Err(self.fail(invalid("smoke challenge peer changed"))),
            Err(error) => return Err(self.fail(error)),
        }
        if Instant::now() >= self.until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "smoke challenge expired",
            )));
        }
        let challenge = match ProxyEntrySmokeChallenge::decode(&self.frame) {
            Ok(challenge) => challenge,
            Err(reason) => return Err(self.fail(invalid(reason))),
        };
        self.challenge = Some(challenge);
        Ok(SmokeChallengePoll::Complete)
    }
    /// Transfer the same FD into one grant exchange only after independent
    /// expected-context comparison. Run must be first; Cleanup must match the
    /// entire recorded Run challenge and exact original peer observation.
    pub(super) fn into_grant(
        mut self,
        expected: &ExpectedSmokeContext,
        phase: ProxyEntrySmokePhase,
        prior: Option<&ObservedSmokeChallenge>,
    ) -> Result<(PendingDirectSmokeGrant, ObservedSmokeChallenge), (io::Error, Self)> {
        if let Some((kind, message, _)) = &self.failure {
            return Err((io::Error::new(*kind, message.clone()), self));
        }
        if Instant::now() >= self.until {
            return Err(self.reject(io::Error::new(
                io::ErrorKind::TimedOut,
                "smoke challenge expired",
            )));
        }
        let Some(challenge) = self.challenge.as_ref() else {
            return Err((invalid("smoke challenge incomplete"), self));
        };
        if !expected.matches(challenge) {
            return Err(self.reject(invalid(
                "smoke challenge differs from expected original context",
            )));
        }
        let valid_order = match (phase, prior) {
            (ProxyEntrySmokePhase::Run, None) => true,
            (ProxyEntrySmokePhase::Cleanup, Some(prior)) => {
                prior.peer == self.peer && &prior.challenge == challenge
            }
            _ => false,
        };
        if !valid_order {
            return Err(self.reject(invalid("smoke challenge sequence or owner differs")));
        }
        let observed = ObservedSmokeChallenge {
            challenge: challenge.clone(),
            peer: self.peer,
        };
        let scope = match phase {
            ProxyEntrySmokePhase::Run => challenge.run.clone(),
            ProxyEntrySmokePhase::Cleanup => challenge.cleanup.clone(),
        };
        let Some(stream) = self.stream.take() else {
            return Err((invalid("original smoke challenge stream missing"), self));
        };
        match PendingDirectSmokeGrant::begin(scope, stream, self.until) {
            Ok(pending) if pending.peer == self.peer => Ok((pending, observed)),
            Ok(pending) => {
                self.stream = Some(pending.stream);
                Err(self.reject(invalid("smoke grant original peer changed")))
            }
            Err((error, stream)) => {
                self.stream = Some(stream);
                Err(self.reject(error))
            }
        }
    }
}

pub(super) enum SmokeGrantPoll {
    Pending,
    Complete,
}

pub(super) struct PendingDirectSmokeGrant {
    stream: UnixStream,
    peer: OriginalDirectPeer,
    scope: ProxyEntrySmokeScope,
    requested_at: Instant,
    request: [u8; PROXY_ENTRY_SMOKE_REQUEST_BYTES],
    written: usize,
    reply: [u8; PROXY_ENTRY_SMOKE_GRANT_BYTES],
    read: usize,
    until: Instant,
    result: Option<Instant>,
    failure: Option<(io::ErrorKind, String, Option<i32>)>,
}

pub(super) struct OriginalDirectSmokeGrant {
    stream: UnixStream,
    peer: OriginalDirectPeer,
    scope: ProxyEntrySmokeScope,
    requested_at: Instant,
    deadline: Instant,
}

impl PendingDirectSmokeGrant {
    /// The caller owns role-channel provenance and a finite exchange bound.
    /// Setup refusal returns the exact supplied socket to that same caller.
    pub(super) fn begin(
        scope: ProxyEntrySmokeScope,
        stream: UnixStream,
        until: Instant,
    ) -> Result<Self, (io::Error, UnixStream)> {
        if Instant::now() >= until {
            return Err((
                io::Error::new(io::ErrorKind::TimedOut, "smoke grant exchange expired"),
                stream,
            ));
        }
        let peer = match observe_direct_parent(&stream) {
            Ok(peer) => peer,
            Err(error) => return Err((error, stream)),
        };
        if let Err(error) = stream.set_nonblocking(true) {
            return Err((error, stream));
        }
        let request = match (ProxyEntrySmokeRequest {
            scope: scope.clone(),
        })
        .encode()
        {
            Ok(request) => request,
            Err(reason) => return Err((invalid(reason), stream)),
        };
        let requested_at = Instant::now();
        Ok(Self {
            stream,
            peer,
            scope,
            requested_at,
            request,
            written: 0,
            reply: [0; PROXY_ENTRY_SMOKE_GRANT_BYTES],
            read: 0,
            until,
            result: None,
            failure: None,
        })
    }
    fn fail(&mut self, error: io::Error) -> io::Error {
        if self.failure.is_none() {
            self.failure = Some((error.kind(), error.to_string(), error.raw_os_error()));
        }
        error
    }
    pub(super) fn first_failure(&self) -> Option<(io::ErrorKind, &str, Option<i32>)> {
        self.failure
            .as_ref()
            .map(|(kind, message, raw)| (*kind, message.as_str(), *raw))
    }
    fn verify_peer(&self) -> io::Result<()> {
        if observe_direct_parent(&self.stream)? != self.peer {
            return Err(invalid("smoke original arena changed"));
        }
        Ok(())
    }
    pub(super) fn poll(&mut self) -> io::Result<SmokeGrantPoll> {
        if let Some((kind, message, _)) = &self.failure {
            return Err(io::Error::new(*kind, message.clone()));
        }
        if self.result.is_some() {
            return Ok(SmokeGrantPoll::Complete);
        }
        if Instant::now() >= self.until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "smoke grant exchange expired",
            )));
        }
        if let Err(error) = self.verify_peer() {
            return Err(self.fail(error));
        }
        if Instant::now() >= self.until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "smoke grant exchange expired",
            )));
        }
        if self.written < PROXY_ENTRY_SMOKE_REQUEST_BYTES {
            match self.stream.write(&self.request[self.written..]) {
                Ok(0) => {
                    return Err(self.fail(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "smoke grant request truncated",
                    )));
                }
                Ok(n) => self.written += n,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) =>
                {
                    return Ok(SmokeGrantPoll::Pending);
                }
                Err(error) => return Err(self.fail(error)),
            }
            if self.written < PROXY_ENTRY_SMOKE_REQUEST_BYTES {
                return Ok(SmokeGrantPoll::Pending);
            }
        }
        if Instant::now() >= self.until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "smoke grant exchange expired",
            )));
        }
        match self.stream.read(&mut self.reply[self.read..]) {
            Ok(0) => {
                return Err(self.fail(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "smoke grant reply truncated",
                )));
            }
            Ok(n) => self.read += n,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                return Ok(SmokeGrantPoll::Pending);
            }
            Err(error) => return Err(self.fail(error)),
        }
        if self.read < PROXY_ENTRY_SMOKE_GRANT_BYTES {
            return Ok(SmokeGrantPoll::Pending);
        }
        if let Err(error) = self.verify_peer() {
            return Err(self.fail(error));
        }
        if Instant::now() >= self.until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "smoke grant exchange expired",
            )));
        }
        let grant = match ProxyEntrySmokeGrant::decode(&self.reply) {
            Ok(grant) => grant,
            Err(reason) => return Err(self.fail(invalid(reason))),
        };
        if grant.scope != self.scope {
            return Err(self.fail(invalid("smoke grant original scope differs")));
        }
        let deadline = match self
            .requested_at
            .checked_add(Duration::from_nanos(grant.remaining_ns))
        {
            Some(deadline) => deadline,
            None => return Err(self.fail(invalid("smoke grant deadline overflow"))),
        };
        if Instant::now() >= deadline {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "smoke original grant expired",
            )));
        }
        self.result = Some(deadline);
        Ok(SmokeGrantPoll::Complete)
    }
    pub(super) fn into_completed(mut self) -> Result<OriginalDirectSmokeGrant, Self> {
        if self.failure.is_some() {
            return Err(self);
        }
        match self.result.take() {
            Some(deadline) => Ok(OriginalDirectSmokeGrant {
                stream: self.stream,
                peer: self.peer,
                scope: self.scope,
                requested_at: self.requested_at,
                deadline,
            }),
            None => Err(self),
        }
    }
}

impl OriginalDirectSmokeGrant {
    pub(super) fn scope(&self) -> &ProxyEntrySmokeScope {
        &self.scope
    }
    pub(super) fn deadline(&self) -> Instant {
        self.deadline
    }
    pub(super) fn requested_at(&self) -> Instant {
        self.requested_at
    }
    pub(super) fn same_original_peer(&self, other: &Self) -> bool {
        self.peer == other.peer
    }
    /// Remote EOF is not interpreted as liveness or physical completion.
    pub(super) fn verify_original_peer(&self) -> io::Result<()> {
        if observe_direct_parent(&self.stream)? != self.peer {
            return Err(invalid("smoke original arena changed"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::pals::{
        deadline_grant_v1::DeadlineRole,
        proxy_entry_smoke_grant_v1::{ProxyEntrySmokePhase, ProxyEntrySmokePolicy},
    };
    use std::os::fd::AsRawFd;
    #[test]
    fn challenge_requires_independently_expected_owner_role_and_lock() {
        let run = ProxyEntrySmokeScope {
            policy: ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1,
            owner_id: [1; 16],
            owner_epoch: 2,
            role: DeadlineRole::Baseline,
            generation: 3,
            request_id: 4,
            lock_sha256: [5; 32],
            phase: ProxyEntrySmokePhase::Run,
        };
        let mut cleanup = run.clone();
        cleanup.phase = ProxyEntrySmokePhase::Cleanup;
        cleanup.request_id = 6;
        let challenge = ProxyEntrySmokeChallenge {
            run: run.clone(),
            cleanup,
        };
        let mut expected = ExpectedSmokeContext {
            policy: run.policy,
            owner_id: run.owner_id,
            owner_epoch: run.owner_epoch,
            role: run.role,
            generation: run.generation,
            lock_sha256: run.lock_sha256,
        };
        assert!(expected.matches(&challenge));
        expected.role = DeadlineRole::Candidate;
        assert!(!expected.matches(&challenge));
        expected.role = run.role;
        expected.owner_epoch += 1;
        assert!(!expected.matches(&challenge));
        expected.owner_epoch = run.owner_epoch;
        expected.lock_sha256[0] ^= 1;
        assert!(!expected.matches(&challenge));
    }

    #[test]
    fn same_process_peer_is_not_direct_arena_parent_and_fd_returns() {
        let (stream, _other) = UnixStream::pair().unwrap();
        let fd = stream.as_raw_fd();
        let scope = ProxyEntrySmokeScope {
            policy: ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1,
            owner_id: [1; 16],
            owner_epoch: 2,
            role: DeadlineRole::Baseline,
            generation: 3,
            request_id: 4,
            lock_sha256: [5; 32],
            phase: ProxyEntrySmokePhase::Run,
        };
        let (_, stream) = match PendingDirectSmokeGrant::begin(
            scope,
            stream,
            Instant::now() + Duration::from_secs(30),
        ) {
            Ok(_) => panic!("same-process peer cannot be the direct parent"),
            Err(error) => error,
        };
        assert_eq!(stream.as_raw_fd(), fd);
    }
}
