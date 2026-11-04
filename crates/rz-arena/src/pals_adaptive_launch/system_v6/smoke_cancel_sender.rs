//! NoGame cancellation on the original role listener and retained run owner.
//! An ACK records request acceptance, not process or GPU completion.

use super::*;
use rz_contracts::pals::proxy_entry_smoke_cancel_v1::{
    SMOKE_CANCEL_ACK_BYTES, SMOKE_CANCEL_REQUEST_BYTES, SmokeCancelAckV1, SmokeCancelDispositionV1,
    SmokeCancelRequestV1,
};

pub(crate) enum SmokeCancelProgress {
    /// No complete original Run grant exists; no cancel bytes were sent.
    NotGranted,
    Pending {
        written: usize,
        received: usize,
    },
    Acknowledged,
    /// The exact original error remains on the same control.
    Failed,
}

pub(super) struct SmokeCancelPeer {
    stream: UnixStream,
    request: [u8; SMOKE_CANCEL_REQUEST_BYTES],
    written: usize,
    ack: [u8; SMOKE_CANCEL_ACK_BYTES],
    received: usize,
    acknowledged: bool,
}

impl SmokeDeadlineControl {
    /// Poll only the original NoGame run. A missing connection is pending;
    /// once accepted, even a failed or partial stream stays in this owner.
    pub(crate) fn poll_no_game_cancel(
        &mut self,
        custody: &SpawnCustody,
        callback_until: Instant,
    ) -> SmokeCancelProgress {
        if matches!(&self.binding, SmokeBinding::NoGame { .. }) {
            self.close_no_game_run();
        }
        if self
            .cancel_peer
            .as_ref()
            .is_some_and(|peer| peer.acknowledged)
        {
            return SmokeCancelProgress::Acknowledged;
        }
        if self.cancel_failure.is_some() {
            return SmokeCancelProgress::Failed;
        }
        match self.poll_no_game_cancel_inner(custody, callback_until) {
            Ok(progress) => progress,
            Err(error) => {
                self.cancel_failure = Some(Arc::new(error));
                SmokeCancelProgress::Failed
            }
        }
    }

    fn poll_no_game_cancel_inner(
        &mut self,
        custody: &SpawnCustody,
        callback_until: Instant,
    ) -> Result<SmokeCancelProgress, ArenaError> {
        let until = callback_until.min(self.binding.cleanup_until());
        if Instant::now() >= until {
            return Err(failure("smoke_cancel_deadline"));
        }
        let sockets = match &self.binding {
            SmokeBinding::NoGame {
                owner,
                execution,
                sockets,
                ..
            } if Arc::ptr_eq(&owner.execution, execution)
                && execution.matches_plan(&owner.plan)
                && sockets.role() == owner.role =>
            {
                Arc::clone(sockets)
            }
            _ => return Err(failure("smoke_cancel_owner")),
        };
        sockets.verify_cancel_listener()?;
        let Some((pid, identity)) = self.child else {
            return Ok(SmokeCancelProgress::NotGranted);
        };
        let Some(original_grant) = self.retained_grants.iter().find(|peer| {
            peer.response.is_some()
                && peer.written == PROXY_ENTRY_SMOKE_GRANT_BYTES
                && ProxyEntrySmokeRequest::decode(&peer.request)
                    .is_ok_and(|request| request.scope == self.scopes[0])
        }) else {
            return Ok(SmokeCancelProgress::NotGranted);
        };
        verify_child_and_peer(
            custody,
            &original_grant.stream,
            pid,
            identity,
            &self.proxy_pin,
            until,
        )?;
        if self.cancel_peer.is_none() {
            let request = SmokeCancelRequestV1 {
                scope: self.scopes[0].clone(),
                sequence: 1,
            }
            .encode()
            .map_err(failure)?;
            let (stream, _) = match sockets.listener(true).accept() {
                Ok(value) => value,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    return Ok(SmokeCancelProgress::Pending {
                        written: 0,
                        received: 0,
                    });
                }
                Err(_) => return Err(failure("smoke_cancel_accept")),
            };
            self.cancel_peer = Some(SmokeCancelPeer {
                stream,
                request,
                written: 0,
                ack: [0; SMOKE_CANCEL_ACK_BYTES],
                received: 0,
                acknowledged: false,
            });
            self.cancel_peer
                .as_ref()
                .unwrap()
                .stream
                .set_nonblocking(true)
                .map_err(|_| failure("smoke_cancel_nonblocking"))?;
        }
        let peer = self
            .cancel_peer
            .as_mut()
            .ok_or_else(|| failure("smoke_cancel_peer_missing"))?;
        if peer.acknowledged {
            return Ok(SmokeCancelProgress::Acknowledged);
        }
        verify_child_and_peer(custody, &peer.stream, pid, identity, &self.proxy_pin, until)?;
        if Instant::now() >= until {
            return Err(failure("smoke_cancel_deadline"));
        }
        if peer.written < peer.request.len() {
            match peer.stream.write(&peer.request[peer.written..]) {
                Ok(0) => return Err(failure("smoke_cancel_write_zero")),
                Ok(count) => peer.written += count,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) => {}
                Err(_) => return Err(failure("smoke_cancel_write")),
            }
            if peer.written < peer.request.len() {
                return Ok(SmokeCancelProgress::Pending {
                    written: peer.written,
                    received: peer.received,
                });
            }
        }
        verify_child_and_peer(custody, &peer.stream, pid, identity, &self.proxy_pin, until)?;
        if Instant::now() >= until {
            return Err(failure("smoke_cancel_deadline"));
        }
        match peer.stream.read(&mut peer.ack[peer.received..]) {
            Ok(0) => return Err(failure("smoke_cancel_eof")),
            Ok(count) => peer.received += count,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) => {}
            Err(_) => return Err(failure("smoke_cancel_read")),
        }
        if peer.received < peer.ack.len() {
            return Ok(SmokeCancelProgress::Pending {
                written: peer.written,
                received: peer.received,
            });
        }
        let ack = SmokeCancelAckV1::decode(&peer.ack).map_err(failure)?;
        if ack.scope != self.scopes[0]
            || ack.sequence != 1
            || ack.disposition != SmokeCancelDispositionV1::Accepted
        {
            return Err(failure("smoke_cancel_ack_mismatch"));
        }
        verify_child_and_peer(custody, &peer.stream, pid, identity, &self.proxy_pin, until)?;
        if Instant::now() >= until {
            return Err(failure("smoke_cancel_deadline"));
        }
        peer.acknowledged = true;
        Ok(SmokeCancelProgress::Acknowledged)
    }

    pub(crate) fn no_game_cancel_first_failure(&self) -> Option<&ArenaError> {
        self.cancel_failure.as_deref()
    }

    /// Share the same typed error with retained evidence without cloning its
    /// contents or transferring its original stream and control owner.
    pub(crate) fn no_game_cancel_failure_owner(&self) -> Option<Arc<ArenaError>> {
        self.cancel_failure.as_ref().map(Arc::clone)
    }

    /// Offsets remain observable after a failure without replacing the stream.
    pub(crate) fn no_game_cancel_offsets(&self) -> Option<(usize, usize, bool)> {
        self.cancel_peer
            .as_ref()
            .map(|peer| (peer.written, peer.received, peer.acknowledged))
    }
}

pub(super) fn verify_child_and_peer(
    custody: &SpawnCustody,
    stream: &UnixStream,
    pid: u32,
    original: ProcessIdentity,
    proxy_pin: &File,
    until: Instant,
) -> Result<(), ArenaError> {
    if custody.child().map(|child| child.id()) != Some(pid) {
        return Err(failure("smoke_cancel_custody"));
    }
    let raw = i32::try_from(pid).map_err(|_| failure("smoke_cancel_pid"))?;
    loop {
        match waitid(
            Id::Pid(Pid::from_raw(raw)),
            WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT,
        ) {
            Ok(WaitStatus::StillAlive) => break,
            Err(Errno::EINTR) if Instant::now() < until => continue,
            Err(Errno::EINTR) => return Err(failure("smoke_cancel_deadline")),
            _ => return Err(failure("smoke_cancel_child_not_live")),
        }
    }
    let current = process_identity(pid)?;
    let expected = proxy_pin
        .metadata()
        .map_err(|_| failure("smoke_cancel_pin"))?;
    let image =
        fs::metadata(format!("/proc/{pid}/exe")).map_err(|_| failure("smoke_cancel_image"))?;
    let peer = getsockopt(stream, PeerCredentials).map_err(|_| failure("smoke_cancel_peer"))?;
    if current != original
        || current.parent != std::process::id()
        || current.group != pid
        || !expected.is_file()
        || (expected.dev(), expected.ino()) != (image.dev(), image.ino())
        || peer.uid() != nix::unistd::geteuid().as_raw()
        || u32::try_from(peer.pid()).ok() != Some(pid)
        || process_identity(pid)? != original
    {
        return Err(failure("smoke_cancel_peer_differs"));
    }
    Ok(())
}
