//! Proxy receive boundary for a dedicated original-owner control stream.
//! The caller must authenticate and retain the original peer before transfer;
//! wire fields alone cannot authenticate a process. The acknowledgement is
//! acceptance of run cancellation only, never a cleanup or GPU finish claim.

use super::cli_v6::OriginalProxyWindowV6;
use rz_contracts::pals::{
    cancel_control_v1::{
        CANCEL_ACK_BYTES, CANCEL_REQUEST_BYTES, CancelAck, CancelAckDisposition, CancelRequest,
    },
    deadline_grant_v1::{DeadlinePhase, DeadlineScope},
};
use std::{
    io::{self, Read, Write},
    os::unix::net::UnixStream,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CancelPoll {
    Pending,
    AcknowledgementWritten,
}

/// Owns one exclusive, already authenticated, dedicated channel. It does not
/// clone or reopen the channel. Any malformed frame, disconnect or partial I/O
/// failure poisons it; callers retain this object and its exact stream on error.
pub(super) struct CancelReceiverV1 {
    stream: UnixStream,
    scope: DeadlineScope,
    run_cancelled: Arc<AtomicBool>,
    request: [u8; CANCEL_REQUEST_BYTES],
    received: usize,
    ack: Option<([u8; CANCEL_ACK_BYTES], usize)>,
    last_sequence: u64,
    failed: bool,
}

impl CancelReceiverV1 {
    /// The scope is the owner's authenticated run grant, not an untrusted
    /// request. The same window's run token is used; its cleanup token is not.
    /// On setup failure the original stream is returned to its caller.
    pub(super) fn from_original_window(
        stream: UnixStream,
        scope: DeadlineScope,
        window: &OriginalProxyWindowV6,
    ) -> Result<Self, (io::Error, UnixStream)> {
        if scope.validate().is_err() || scope.phase != DeadlinePhase::Run {
            return Err((
                io::Error::new(io::ErrorKind::InvalidInput, "invalid original run scope"),
                stream,
            ));
        }
        if let Err(error) = stream.set_nonblocking(true) {
            return Err((error, stream));
        }
        Ok(Self {
            stream,
            scope,
            run_cancelled: Arc::clone(&window.cancelled),
            request: [0; CANCEL_REQUEST_BYTES],
            received: 0,
            ack: None,
            last_sequence: 0,
            failed: false,
        })
    }

    fn fail(&mut self, error: io::Error) -> io::Error {
        self.failed = true;
        error
    }

    /// Perform bounded nonblocking work. A request is accepted only after its
    /// complete frame, scope and next sequence are verified. The token is set
    /// before ACK transmission; an ACK write failure does not undo cancellation.
    pub(super) fn poll(&mut self) -> io::Result<CancelPoll> {
        if self.failed {
            return Err(io::Error::other("original cancel channel is failed"));
        }
        if self.ack.is_none() {
            match self.stream.read(&mut self.request[self.received..]) {
                Ok(0) => {
                    return Err(self.fail(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "cancel stream ended before complete request",
                    )));
                }
                Ok(n) => self.received += n,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) =>
                {
                    return Ok(CancelPoll::Pending);
                }
                Err(error) => return Err(self.fail(error)),
            }
            if self.received < CANCEL_REQUEST_BYTES {
                return Ok(CancelPoll::Pending);
            }
            let decoded = match CancelRequest::decode(&self.request) {
                Ok(request) => request,
                Err(reason) => {
                    return Err(self.fail(io::Error::new(io::ErrorKind::InvalidData, reason)));
                }
            };
            if decoded.scope != self.scope {
                return Err(self.fail(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "cancel original scope mismatch",
                )));
            }
            let disposition = if decoded.sequence == self.last_sequence && self.last_sequence != 0 {
                CancelAckDisposition::Duplicate
            } else if self.last_sequence.checked_add(1) == Some(decoded.sequence) {
                self.last_sequence = decoded.sequence;
                self.run_cancelled.store(true, Ordering::Release);
                CancelAckDisposition::Accepted
            } else {
                return Err(self.fail(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "cancel sequence is stale or out of order",
                )));
            };
            let ack = match (CancelAck {
                scope: decoded.scope,
                sequence: decoded.sequence,
                disposition,
            })
            .encode()
            {
                Ok(ack) => ack,
                Err(reason) => {
                    return Err(self.fail(io::Error::new(io::ErrorKind::InvalidData, reason)));
                }
            };
            self.received = 0;
            self.ack = Some((ack, 0));
        }
        let (ack, written) = self.ack.as_mut().expect("ack retained during send");
        match self.stream.write(&ack[*written..]) {
            Ok(0) => {
                return Err(self.fail(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "cancel acknowledgement not written",
                )));
            }
            Ok(n) => *written += n,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                return Ok(CancelPoll::Pending);
            }
            Err(error) => return Err(self.fail(error)),
        }
        if *written < CANCEL_ACK_BYTES {
            return Ok(CancelPoll::Pending);
        }
        self.ack = None;
        Ok(CancelPoll::AcknowledgementWritten)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::pals::deadline_grant_v1::DeadlineRole;
    use std::time::{Duration, Instant};
    fn scope() -> DeadlineScope {
        DeadlineScope {
            owner_id: [1; 16],
            owner_epoch: 2,
            role: DeadlineRole::Baseline,
            game: 3,
            generation: 4,
            request_id: 5,
            lock_sha256: [6; 32],
            phase: DeadlinePhase::Run,
        }
    }
    fn window() -> OriginalProxyWindowV6 {
        let now = Instant::now();
        OriginalProxyWindowV6 {
            bound_reference: now,
            startup_until: now + Duration::from_secs(1),
            run_until: now + Duration::from_secs(2),
            cleanup_until: now + Duration::from_secs(3),
            cancelled: Arc::new(AtomicBool::new(false)),
            cleanup_cancelled: Arc::new(AtomicBool::new(false)),
        }
    }
    #[test]
    fn fragmented_request_duplicate_and_cleanup_token_are_distinct() {
        let (stream, mut owner) = UnixStream::pair().unwrap();
        let window = window();
        let mut receiver = CancelReceiverV1::from_original_window(stream, scope(), &window)
            .ok()
            .unwrap();
        let request = CancelRequest {
            scope: scope(),
            sequence: 1,
        }
        .encode()
        .unwrap();
        owner.write_all(&request[..17]).unwrap();
        assert_eq!(receiver.poll().unwrap(), CancelPoll::Pending);
        assert!(!window.cancelled.load(Ordering::Acquire));
        owner.write_all(&request[17..]).unwrap();
        assert_eq!(receiver.poll().unwrap(), CancelPoll::AcknowledgementWritten);
        let mut bytes = [0; CANCEL_ACK_BYTES];
        owner.read_exact(&mut bytes).unwrap();
        assert_eq!(
            CancelAck::decode(&bytes).unwrap().disposition,
            CancelAckDisposition::Accepted
        );
        assert!(window.cancelled.load(Ordering::Acquire));
        assert!(!window.cleanup_cancelled.load(Ordering::Acquire));
        owner.write_all(&request).unwrap();
        assert_eq!(receiver.poll().unwrap(), CancelPoll::AcknowledgementWritten);
        owner.read_exact(&mut bytes).unwrap();
        assert_eq!(
            CancelAck::decode(&bytes).unwrap().disposition,
            CancelAckDisposition::Duplicate
        );
    }
    #[test]
    fn wrong_scope_and_partial_disconnect_never_set_token() {
        let (stream, mut owner) = UnixStream::pair().unwrap();
        let window = window();
        let mut receiver = CancelReceiverV1::from_original_window(stream, scope(), &window)
            .ok()
            .unwrap();
        let mut other = scope();
        other.owner_epoch += 1;
        owner
            .write_all(
                &CancelRequest {
                    scope: other,
                    sequence: 1,
                }
                .encode()
                .unwrap(),
            )
            .unwrap();
        assert!(receiver.poll().is_err());
        assert!(receiver.poll().is_err());
        assert!(!window.cancelled.load(Ordering::Acquire));
        let (stream, mut owner) = UnixStream::pair().unwrap();
        let mut receiver = CancelReceiverV1::from_original_window(stream, scope(), &window)
            .ok()
            .unwrap();
        owner.write_all(&[1, 2]).unwrap();
        drop(owner);
        assert_eq!(receiver.poll().unwrap(), CancelPoll::Pending);
        assert!(receiver.poll().is_err());
        assert!(!window.cancelled.load(Ordering::Acquire));
    }
}
