//! NoGame cancellation from the same direct parent as the original Run grant.
//! Scope bytes never authenticate a peer or establish listener provenance.
//! An ACK is only acceptance, not child, session, or GPU physical completion.

use super::{
    proxy_entry_no_game_smoke_v1::ProxyEntryNoGameAdmittedBundleV1,
    proxy_entry_smoke_channel_v1::DirectSmokeCancelPeer,
};
use rz_contracts::pals::{
    proxy_entry_smoke_cancel_v1::{
        SMOKE_CANCEL_ACK_BYTES, SMOKE_CANCEL_REQUEST_BYTES, SmokeCancelAckV1,
        SmokeCancelDispositionV1, SmokeCancelRequestV1,
    },
    proxy_entry_smoke_grant_v1::{ProxyEntrySmokePhase, ProxyEntrySmokeScope},
};
use std::{
    io::{self, Read, Write},
    os::unix::net::UnixStream,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[cfg(test)]
#[path = "smoke_cancel_tests.rs"]
mod tests;

#[derive(Clone, Debug)]
pub(super) struct NoGameCancelFailure {
    pub kind: io::ErrorKind,
    pub message: String,
    pub raw_os_error: Option<i32>,
}

/// Holds the original cancel stream, bounded partial frame and failure. The
/// original grant and the owner-selected listener remain held by their owners.
pub(super) struct NoGameCancelReceiverV1 {
    stream: UnixStream,
    peer: DirectSmokeCancelPeer,
    scope: ProxyEntrySmokeScope,
    cancelled: Arc<AtomicBool>,
    cleanup_until: Instant,
    request: [u8; SMOKE_CANCEL_REQUEST_BYTES],
    read: usize,
    ack: Option<([u8; SMOKE_CANCEL_ACK_BYTES], usize)>,
    sequence: u64,
    failure: Option<NoGameCancelFailure>,
}

impl NoGameCancelReceiverV1 {
    /// A declined fresh receiver returns its exact original socket.
    pub(super) fn into_stream(self) -> UnixStream {
        self.stream
    }
    /// Every failure returns the exact input stream. Role-listener ownership is
    /// a separate caller responsibility; both peers are checked from real FDs.
    pub(super) fn from_original_bundle(
        stream: UnixStream,
        bundle: &ProxyEntryNoGameAdmittedBundleV1,
    ) -> Result<Self, (io::Error, UnixStream)> {
        let run = &bundle.inputs.run;
        if run.scope().phase != ProxyEntrySmokePhase::Run
            || Instant::now() >= bundle.window.cleanup_until
        {
            return Err((
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "NoGame run scope or cleanup deadline invalid",
                ),
                stream,
            ));
        }
        let peer = match run.cancel_peer(&stream) {
            Ok(peer) => peer,
            Err(error) => return Err((error, stream)),
        };
        if let Err(error) = stream.set_nonblocking(true) {
            return Err((error, stream));
        }
        Ok(Self {
            stream,
            peer,
            scope: run.scope().clone(),
            cancelled: Arc::clone(&bundle.window.cancelled),
            cleanup_until: bundle.window.cleanup_until,
            request: [0; SMOKE_CANCEL_REQUEST_BYTES],
            read: 0,
            ack: None,
            sequence: 0,
            failure: None,
        })
    }

    pub(super) fn matches_window(&self, cancelled: &Arc<AtomicBool>, until: Instant) -> bool {
        Arc::ptr_eq(&self.cancelled, cancelled) && self.cleanup_until == until
    }

    pub(super) fn first_failure(&self) -> Option<&NoGameCancelFailure> {
        self.failure.as_ref()
    }

    fn fail(&mut self, error: io::Error) -> io::Error {
        if self.failure.is_none() {
            self.failure = Some(NoGameCancelFailure {
                kind: error.kind(),
                message: error.to_string(),
                raw_os_error: error.raw_os_error(),
            });
        }
        error
    }

    fn check_control(&mut self) -> io::Result<()> {
        if Instant::now() >= self.cleanup_until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "NoGame cancel cleanup deadline expired",
            )));
        }
        if let Err(error) = self.peer.verify(&self.stream) {
            return Err(self.fail(error));
        }
        if Instant::now() >= self.cleanup_until {
            return Err(self.fail(io::Error::new(
                io::ErrorKind::TimedOut,
                "NoGame cancel cleanup deadline expired",
            )));
        }
        Ok(())
    }

    /// One bounded nonblocking step. First failure and partial offsets stay in
    /// this same owner; setting run cancellation never changes cleanup token.
    pub(super) fn poll(&mut self) -> io::Result<bool> {
        if let Some(error) = &self.failure {
            return Err(io::Error::new(error.kind, error.message.clone()));
        }
        self.check_control()?;
        if self.ack.is_none() {
            match self.stream.read(&mut self.request[self.read..]) {
                Ok(0) => {
                    return Err(self.fail(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "NoGame cancel truncated",
                    )));
                }
                Ok(count) => self.read += count,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) =>
                {
                    return Ok(false);
                }
                Err(error) => return Err(self.fail(error)),
            }
            if self.read < SMOKE_CANCEL_REQUEST_BYTES {
                return Ok(false);
            }
            let request = match SmokeCancelRequestV1::decode(&self.request) {
                Ok(request) => request,
                Err(reason) => {
                    return Err(self.fail(io::Error::new(io::ErrorKind::InvalidData, reason)));
                }
            };
            if request.scope != self.scope {
                return Err(self.fail(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "NoGame cancel scope differs",
                )));
            }
            let disposition = if request.sequence == self.sequence && self.sequence != 0 {
                SmokeCancelDispositionV1::Duplicate
            } else if self.sequence.checked_add(1) == Some(request.sequence) {
                self.check_control()?;
                self.sequence = request.sequence;
                self.cancelled.store(true, Ordering::Release);
                SmokeCancelDispositionV1::Accepted
            } else {
                return Err(self.fail(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "NoGame cancel sequence invalid",
                )));
            };
            let ack = match (SmokeCancelAckV1 {
                scope: request.scope,
                sequence: request.sequence,
                disposition,
            })
            .encode()
            {
                Ok(ack) => ack,
                Err(reason) => {
                    return Err(self.fail(io::Error::new(io::ErrorKind::InvalidData, reason)));
                }
            };
            self.read = 0;
            self.ack = Some((ack, 0));
        }
        self.check_control()?;
        let (ack, written) = self.ack.as_mut().expect("NoGame ack retained");
        match self.stream.write(&ack[*written..]) {
            Ok(0) => {
                return Err(self.fail(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "NoGame cancel ack not written",
                )));
            }
            Ok(count) => *written += count,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                return Ok(false);
            }
            Err(error) => return Err(self.fail(error)),
        }
        if *written < SMOKE_CANCEL_ACK_BYTES {
            return Ok(false);
        }
        self.ack = None;
        Ok(true)
    }
}
