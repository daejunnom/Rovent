//! Original grant socket custody for the proxy's direct arena/Fastchess tree.
//! This binds response bytes to the held stream and observed live ancestry; it
//! does not establish role listener provenance, later peer liveness, or GPU
//! completion. The executable guard remains closed.

use super::{
    cancel_control_v1::CancelReceiverV1,
    cli_v6::OriginalProxyWindowV6,
    deadline_grant::{PendingDeadlineGrant, ReceivedDeadlineGrant, assemble_window_from_refs},
    runner_v6::PalsProxySupervisorV6,
};
use crate::pals_adaptive_startup::{PalsSystemSourceObservationV6, PinnedPalsSystemStartupV6};
use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
use rz_contracts::pals::deadline_grant_v1::{DeadlineScope, GRANT_BYTES, REQUEST_BYTES};
use std::{
    fs::File,
    io::{self, Read, Write},
    os::unix::{fs::MetadataExt, net::UnixStream},
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ProcessIdentity {
    pub(super) parent: u32,
    pub(super) group: u32,
    pub(super) start_ticks: u64,
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn parse_process_stat(pid: u32, bytes: &[u8]) -> io::Result<ProcessIdentity> {
    let left = bytes
        .iter()
        .position(|byte| *byte == b'(')
        .ok_or_else(|| invalid("arena peer proc format"))?;
    let right = bytes
        .iter()
        .rposition(|byte| *byte == b')')
        .ok_or_else(|| invalid("arena peer proc format"))?;
    if left >= right
        || std::str::from_utf8(&bytes[..left])
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            != Some(pid)
    {
        return Err(invalid("arena peer proc identity"));
    }
    let tail =
        std::str::from_utf8(&bytes[right + 1..]).map_err(|_| invalid("arena peer proc format"))?;
    let fields: Vec<&str> = tail.split_whitespace().collect();
    if matches!(
        fields.first().copied(),
        None | Some("Z") | Some("X") | Some("x")
    ) {
        return Err(invalid("arena peer process no longer live"));
    }
    let number = |index: usize| {
        fields
            .get(index)
            .and_then(|value| value.parse::<u64>().ok())
            .ok_or_else(|| invalid("arena peer proc field"))
    };
    Ok(ProcessIdentity {
        parent: u32::try_from(number(1)?).map_err(|_| invalid("arena peer parent"))?,
        group: u32::try_from(number(2)?).map_err(|_| invalid("arena peer group"))?,
        start_ticks: number(19)?,
    })
}

pub(super) fn process_identity(pid: u32) -> io::Result<ProcessIdentity> {
    let mut bytes = Vec::new();
    File::open(format!("/proc/{pid}/stat"))?
        .take(4097)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 4096 {
        return Err(invalid("arena peer proc size"));
    }
    parse_process_stat(pid, &bytes)
}

/// Observation made from this original socket and the live direct ancestry.
/// No string in a grant request is used as evidence of peer PID or role.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct OriginalArenaPeerObservation {
    pid: u32,
    uid: u32,
    gid: u32,
    proxy: ProcessIdentity,
    fastchess_pid: u32,
    fastchess: ProcessIdentity,
    arena: ProcessIdentity,
}

fn observe_peer(stream: &UnixStream) -> io::Result<OriginalArenaPeerObservation> {
    let peer = getsockopt(stream, PeerCredentials).map_err(io::Error::other)?;
    let pid = u32::try_from(peer.pid()).map_err(|_| invalid("arena peer pid"))?;
    if pid == 0 {
        return Err(invalid("arena peer pid"));
    }
    let proxy = process_identity(std::process::id())?;
    let fastchess_pid = proxy.parent;
    if fastchess_pid == 0 {
        return Err(invalid("arena fastchess parent unknown"));
    }
    let fastchess = process_identity(fastchess_pid)?;
    if fastchess.parent != pid || proxy.group != fastchess_pid || fastchess.group != fastchess_pid {
        return Err(invalid("arena peer is not original direct grandparent"));
    }
    let arena = process_identity(pid)?;
    if std::fs::metadata(format!("/proc/{pid}"))?.uid() != peer.uid() {
        return Err(invalid("arena peer uid differs"));
    }
    // PID reuse, reparenting and a live-to-zombie transition are rejected.
    if process_identity(std::process::id())? != proxy
        || process_identity(fastchess_pid)? != fastchess
        || process_identity(pid)? != arena
    {
        return Err(invalid("arena peer ancestry changed"));
    }
    Ok(OriginalArenaPeerObservation {
        pid,
        uid: peer.uid(),
        gid: peer.gid(),
        proxy,
        fastchess_pid,
        fastchess,
        arena,
    })
}

pub(super) enum OriginalGrantPoll {
    Pending,
    Complete,
}

/// One nonblocking original socket, request and partial response are retained
/// together. Caller owns the finite deadline and role-channel provenance.
pub(super) struct PendingOriginalArenaGrant {
    stream: UnixStream,
    peer: OriginalArenaPeerObservation,
    pending: Option<PendingDeadlineGrant>,
    request: [u8; REQUEST_BYTES],
    written: usize,
    reply: [u8; GRANT_BYTES],
    read: usize,
    until: Instant,
    complete: Option<ReceivedDeadlineGrant>,
    failure: Option<(io::ErrorKind, String, Option<i32>)>,
}

pub(super) struct RetainedOriginalArenaGrant {
    pub grant: ReceivedDeadlineGrant,
    pub stream: UnixStream,
    pub peer: OriginalArenaPeerObservation,
}

impl PendingOriginalArenaGrant {
    /// On setup failure returns the same socket to the caller. The supplied
    /// deadline bounds this exchange; it does not replace an arena deadline.
    pub(super) fn begin(
        scope: DeadlineScope,
        stream: UnixStream,
        until: Instant,
    ) -> Result<Self, (io::Error, UnixStream)> {
        if Instant::now() >= until {
            return Err((
                io::Error::new(io::ErrorKind::TimedOut, "arena grant exchange expired"),
                stream,
            ));
        }
        let peer = match observe_peer(&stream) {
            Ok(peer) => peer,
            Err(error) => return Err((error, stream)),
        };
        if let Err(error) = stream.set_nonblocking(true) {
            return Err((error, stream));
        }
        let (pending, request) = match PendingDeadlineGrant::begin(scope) {
            Ok(value) => value,
            Err(error) => return Err((error, stream)),
        };
        Ok(Self {
            stream,
            peer,
            pending: Some(pending),
            request,
            written: 0,
            reply: [0; GRANT_BYTES],
            read: 0,
            until,
            complete: None,
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
    fn verify_owner(&self) -> io::Result<()> {
        if observe_peer(&self.stream)? != self.peer {
            return Err(invalid("original arena grant peer changed"));
        }
        Ok(())
    }
    pub(super) fn poll(&mut self) -> io::Result<OriginalGrantPoll> {
        if let Some((kind, message, _)) = &self.failure {
            return Err(io::Error::new(*kind, message.clone()));
        }
        if self.complete.is_some() {
            return Ok(OriginalGrantPoll::Complete);
        }
        if Instant::now() >= self.until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "arena grant exchange expired",
            )));
        }
        if let Err(error) = self.verify_owner() {
            return Err(self.fail(error));
        }
        if Instant::now() >= self.until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "arena grant exchange expired",
            )));
        }
        if self.written < REQUEST_BYTES {
            match self.stream.write(&self.request[self.written..]) {
                Ok(0) => {
                    return Err(self.fail(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "arena grant request truncated",
                    )));
                }
                Ok(n) => self.written += n,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) =>
                {
                    return Ok(OriginalGrantPoll::Pending);
                }
                Err(error) => return Err(self.fail(error)),
            }
            if self.written < REQUEST_BYTES {
                return Ok(OriginalGrantPoll::Pending);
            }
        }
        if Instant::now() >= self.until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "arena grant exchange expired",
            )));
        }
        match self.stream.read(&mut self.reply[self.read..]) {
            Ok(0) => {
                return Err(self.fail(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "arena grant reply truncated",
                )));
            }
            Ok(n) => self.read += n,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                return Ok(OriginalGrantPoll::Pending);
            }
            Err(error) => return Err(self.fail(error)),
        }
        if self.read < GRANT_BYTES {
            return Ok(OriginalGrantPoll::Pending);
        }
        if Instant::now() >= self.until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "arena grant exchange expired",
            )));
        }
        if let Err(error) = self.verify_owner() {
            return Err(self.fail(error));
        }
        if Instant::now() >= self.until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "arena grant exchange expired",
            )));
        }
        let pending = match self.pending.take() {
            Some(pending) => pending,
            None => return Err(self.fail(invalid("original pending grant unavailable"))),
        };
        let grant = match pending.accept(&self.reply) {
            Ok(grant) => grant,
            Err(reason) => return Err(self.fail(invalid(reason))),
        };
        self.complete = Some(grant);
        Ok(OriginalGrantPoll::Complete)
    }
    /// Only the exact held socket and its completed response move together.
    pub(super) fn into_completed(mut self) -> Result<RetainedOriginalArenaGrant, Self> {
        if self.failure.is_some() {
            return Err(self);
        }
        match self.complete.take() {
            Some(grant) => Ok(RetainedOriginalArenaGrant {
                grant,
                stream: self.stream,
                peer: self.peer,
            }),
            None => Err(self),
        }
    }
}

/// Each received grant and its original socket stay owned together. This
/// input is returned intact on any assembly or peer-observation failure.
pub(super) struct OriginalArenaGrantInputs {
    pub startup: RetainedOriginalArenaGrant,
    pub run: RetainedOriginalArenaGrant,
    pub cleanup: RetainedOriginalArenaGrant,
}

/// A one-shot projection of three original grants; all source streams and
/// original observations remain with this bundle for the entire scope.
pub(super) struct OriginalArenaGrantBundle {
    pub inputs: OriginalArenaGrantInputs,
    pub window: OriginalProxyWindowV6,
    pub observations: [PalsSystemSourceObservationV6; 3],
}

impl OriginalArenaGrantInputs {
    fn verify_same_original_peer(&self) -> io::Result<()> {
        let peer = self.startup.peer;
        if self.run.peer != peer || self.cleanup.peer != peer {
            return Err(invalid("grant peer observations differ"));
        }
        for grant in [&self.startup, &self.run, &self.cleanup] {
            if observe_peer(&grant.stream)? != peer {
                return Err(invalid("original grant peer or ancestry changed"));
            }
        }
        Ok(())
    }

    /// The old bytes-only assembly remains available. This stricter path
    /// consumes this owner only on success, never reconstructing grant bytes
    /// or replacing original streams. Socket retention is not proof the remote
    /// endpoint is open, nor proof of role-listener provenance.
    pub(super) fn assemble(
        self,
        source: &PinnedPalsSystemStartupV6,
        cancelled: &Arc<AtomicBool>,
        cleanup_cancelled: &Arc<AtomicBool>,
    ) -> Result<OriginalArenaGrantBundle, (io::Error, Self)> {
        let result = (|| {
            self.verify_same_original_peer()?;
            let projection = assemble_window_from_refs(
                source,
                &self.startup.grant,
                &self.run.grant,
                &self.cleanup.grant,
                Arc::clone(cancelled),
                Arc::clone(cleanup_cancelled),
            )?;
            self.verify_same_original_peer()?;
            if Instant::now() >= projection.0.startup_until {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "original startup deadline expired",
                ));
            }
            Ok(projection)
        })();
        match result {
            Ok((window, observations)) => Ok(OriginalArenaGrantBundle {
                inputs: self,
                window,
                observations,
            }),
            Err(error) => Err((error, self)),
        }
    }
}

fn matches_original_run_window(
    grant: &ReceivedDeadlineGrant,
    window: &OriginalProxyWindowV6,
) -> bool {
    grant.original_scope().phase == rz_contracts::pals::deadline_grant_v1::DeadlinePhase::Run
        && window.run_until <= grant.pinning_deadline()
        && window.bound_reference <= grant.requested_at()
}

impl RetainedOriginalArenaGrant {
    /// Reobserve the actual original grant stream and direct peer ancestry.
    /// This does not assert the remote stream is still open.
    pub(super) fn verify_original_peer(&self) -> io::Result<()> {
        if observe_peer(&self.stream)? != self.peer {
            return Err(invalid("original arena grant peer changed"));
        }
        Ok(())
    }

    /// Compare both held endpoints to the original recorded peer and process
    /// tree. This does not prove the grant channel is still open or that the
    /// cancellation listener belongs to a particular role.
    pub(super) fn verify_cancel_peer(&self, cancel_stream: &UnixStream) -> io::Result<()> {
        if observe_peer(&self.stream)? != self.peer
            || observe_peer(cancel_stream)? != self.peer
            || observe_peer(&self.stream)? != self.peer
        {
            return Err(invalid(
                "cancel socket differs from original arena grant peer",
            ));
        }
        Ok(())
    }

    /// Transfer the same cancel stream into the original supervisor only after
    /// comparing it with this retained run grant. The caller must supply the
    /// original assembled window and prove the role listener's origin; neither
    /// the scope values nor same peer identity alone establish that origin.
    /// Every refusal returns the exact stream, preserving both channel owners.
    pub(super) fn attach_cancel_to_supervisor(
        &self,
        cancel_stream: UnixStream,
        window: &OriginalProxyWindowV6,
        supervisor: &mut PalsProxySupervisorV6,
    ) -> Result<(), (io::Error, UnixStream)> {
        if !matches_original_run_window(&self.grant, window)
            || Instant::now() >= window.cleanup_until
        {
            return Err((
                invalid("cancel window differs from original run grant"),
                cancel_stream,
            ));
        }
        if let Err(error) = self.verify_cancel_peer(&cancel_stream) {
            return Err((error, cancel_stream));
        }
        if Instant::now() >= window.cleanup_until {
            return Err((
                io::Error::new(io::ErrorKind::TimedOut, "original cleanup deadline expired"),
                cancel_stream,
            ));
        }
        let receiver = CancelReceiverV1::from_original_window(
            cancel_stream,
            self.grant.original_scope().clone(),
            window,
        )?;
        match supervisor.attach_caller_authenticated_cancel(receiver) {
            Ok(()) => Ok(()),
            Err(receiver) => Err((
                invalid("proxy supervisor differs from cancel window"),
                receiver.into_stream(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::pals::deadline_grant_v1::{DeadlinePhase, DeadlineRole};
    use std::os::fd::AsRawFd;
    use std::time::Duration;
    #[test]
    fn mismatched_recorded_peer_is_rejected_without_losing_original_streams() {
        use rz_contracts::pals::deadline_grant_v1::DeadlineGrant;
        let identity = ProcessIdentity {
            parent: 1,
            group: 2,
            start_ticks: 3,
        };
        let peer = OriginalArenaPeerObservation {
            pid: 4,
            uid: 5,
            gid: 6,
            proxy: identity,
            fastchess_pid: 7,
            fastchess: identity,
            arena: identity,
        };
        let grant = |phase, request_id, observed_peer| {
            let scope = DeadlineScope {
                owner_id: [1; 16],
                owner_epoch: 2,
                role: DeadlineRole::Baseline,
                game: 0,
                generation: 3,
                request_id,
                lock_sha256: [5; 32],
                phase,
            };
            let (pending, _) = PendingDeadlineGrant::begin(scope.clone()).unwrap();
            let bytes = DeadlineGrant {
                scope,
                remaining_ns: 60_000_000_000,
            }
            .encode()
            .unwrap();
            let grant = pending.accept(&bytes).unwrap();
            let (stream, _other) = UnixStream::pair().unwrap();
            RetainedOriginalArenaGrant {
                grant,
                stream,
                peer: observed_peer,
            }
        };
        let mut different = peer;
        different.arena.start_ticks += 1;
        let inputs = OriginalArenaGrantInputs {
            startup: grant(DeadlinePhase::Startup, 1, peer),
            run: grant(DeadlinePhase::Run, 2, peer),
            cleanup: grant(DeadlinePhase::Cleanup, 3, different),
        };
        let fds = [
            inputs.startup.stream.as_raw_fd(),
            inputs.run.stream.as_raw_fd(),
            inputs.cleanup.stream.as_raw_fd(),
        ];
        assert!(inputs.verify_same_original_peer().is_err());
        assert_eq!(
            fds,
            [
                inputs.startup.stream.as_raw_fd(),
                inputs.run.stream.as_raw_fd(),
                inputs.cleanup.stream.as_raw_fd()
            ]
        );
    }

    #[test]
    fn assembled_window_never_extends_the_original_run_grant() {
        use rz_contracts::pals::deadline_grant_v1::DeadlineGrant;
        use std::sync::{Arc, atomic::AtomicBool};
        let scope = DeadlineScope {
            owner_id: [1; 16],
            owner_epoch: 2,
            role: DeadlineRole::Baseline,
            game: 0,
            generation: 3,
            request_id: 4,
            lock_sha256: [5; 32],
            phase: DeadlinePhase::Run,
        };
        let before = Instant::now();
        let (pending, _) = PendingDeadlineGrant::begin(scope.clone()).unwrap();
        let response = DeadlineGrant {
            scope,
            remaining_ns: 60_000_000_000,
        }
        .encode()
        .unwrap();
        let grant = pending.accept(&response).unwrap();
        let mut window = OriginalProxyWindowV6 {
            bound_reference: before,
            startup_until: before,
            run_until: grant.pinning_deadline(),
            cleanup_until: grant.pinning_deadline(),
            cancelled: Arc::new(AtomicBool::new(false)),
            cleanup_cancelled: Arc::new(AtomicBool::new(false)),
        };
        assert!(matches_original_run_window(&grant, &window));
        window.run_until = grant.pinning_deadline() + Duration::from_nanos(1);
        assert!(!matches_original_run_window(&grant, &window));
    }

    #[test]
    fn process_identity_uses_actual_pid_parent_group_and_start_ticks() {
        let mut fields = vec!["0"; 20];
        fields[0] = "S";
        fields[1] = "17";
        fields[2] = "19";
        fields[19] = "23";
        let stat = format!("42 (name with ) inside) {}", fields.join(" "));
        assert_eq!(
            parse_process_stat(42, stat.as_bytes()).unwrap(),
            ProcessIdentity {
                parent: 17,
                group: 19,
                start_ticks: 23
            }
        );
        assert!(parse_process_stat(43, stat.as_bytes()).is_err());
        fields[0] = "Z";
        let dead = format!("42 (name) {}", fields.join(" "));
        assert!(parse_process_stat(42, dead.as_bytes()).is_err());
    }

    #[test]
    fn same_process_socket_is_not_an_authenticated_arena_grandparent() {
        let (socket, _peer) = UnixStream::pair().unwrap();
        let fd = socket.as_raw_fd();
        let scope = DeadlineScope {
            owner_id: [1; 16],
            owner_epoch: 2,
            role: DeadlineRole::Baseline,
            game: 0,
            generation: 3,
            request_id: 4,
            lock_sha256: [5; 32],
            phase: DeadlinePhase::Run,
        };
        let (_, returned) = match PendingOriginalArenaGrant::begin(
            scope,
            socket,
            Instant::now() + Duration::from_secs(30),
        ) {
            Ok(_) => panic!("same-process peer cannot be the arena grandparent"),
            Err(error) => error,
        };
        assert_eq!(returned.as_raw_fd(), fd);
    }
}
