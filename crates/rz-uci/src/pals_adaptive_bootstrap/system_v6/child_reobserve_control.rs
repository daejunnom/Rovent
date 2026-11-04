//! Nonblocking same-parent, same-child observation transport.
//!
//! The caller supplies an already authenticated, exclusive original stream.
//! Peer checks here are additional checks, not proof of a prior handshake,
//! listener provenance, native admission, or GPU completion. The original
//! parent, stream, partial frames and first failure remain with this owner.

use crate::rovent_bootstrap::launch::ConnectedParent;
use nix::{
    errno::Errno,
    poll::{PollFd, PollFlags, poll},
    sys::socket::{getsockopt, sockopt::PeerCredentials},
    unistd::{geteuid, getppid},
};
use rz_contracts::pals::child_reobserve_v1::{
    CHILD_REOBSERVE_REQUEST_BYTES, CHILD_REOBSERVE_RESPONSE_BYTES, ChildReobserveRequestV1,
    ChildReobserveResponseV1,
};
use std::{
    io::{self, Read, Write},
    os::{fd::AsFd, unix::net::UnixStream},
    sync::Arc,
    time::Instant,
};

#[derive(Debug)]
pub(super) struct ChildReobserveFailureV1 {
    pub kind: io::ErrorKind,
    pub message: String,
    pub raw_os_error: Option<i32>,
}

/// Retain this owner and its original stream even on transport failure.
#[must_use = "retain the original reobservation channel owner"]
pub(super) struct ChildReobserveControlV1 {
    parent: Arc<ConnectedParent>,
    stream: UnixStream,
    parent_pid: i32,
    parent_uid: u32,
    generation: u64,
    lock_sha256: String,
    ordinal: i32,
    last_sequence: u64,
    input: [u8; CHILD_REOBSERVE_REQUEST_BYTES],
    input_read: usize,
    pending: Option<ChildReobserveRequestV1>,
    output: Option<([u8; CHILD_REOBSERVE_RESPONSE_BYTES], usize)>,
    first_failure: Option<ChildReobserveFailureV1>,
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn window(io_until: Instant) -> io::Result<()> {
    if Instant::now() >= io_until {
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "reobservation I/O window expired",
        ))
    } else {
        Ok(())
    }
}

fn wait_stream_ready(stream: &UnixStream, writing: bool, io_until: Instant) -> io::Result<bool> {
    window(io_until)?;
    let wait_ms = io_until
        .saturating_duration_since(Instant::now())
        .as_millis()
        .min(u16::MAX as u128) as u16;
    if wait_ms == 0 {
        return Ok(false);
    }
    let flags = if writing {
        PollFlags::POLLOUT
    } else {
        PollFlags::POLLIN
    };
    let mut descriptors = [PollFd::new(stream.as_fd(), flags)];
    match poll(&mut descriptors, wait_ms) {
        Ok(0) | Err(Errno::EINTR) => Ok(false),
        Ok(_) => Ok(true),
        Err(error) => Err(io::Error::other(error)),
    }
}

fn peer_matches(stream: &UnixStream, parent_pid: i32, parent_uid: u32) -> io::Result<()> {
    let peer = getsockopt(stream, PeerCredentials).map_err(io::Error::other)?;
    if parent_pid <= 0
        || getppid().as_raw() != parent_pid
        || geteuid().as_raw() != parent_uid
        || peer.pid() != parent_pid
        || peer.uid() != parent_uid
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "original reobservation parent differs",
        ));
    }
    Ok(())
}

fn read_step(stream: &mut UnixStream, input: &mut [u8], offset: &mut usize) -> io::Result<()> {
    match stream.read(&mut input[*offset..]) {
        Ok(0) => Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "reobservation request truncated",
        )),
        Ok(count) => {
            *offset += count;
            Ok(())
        }
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
            ) =>
        {
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn write_step(stream: &mut UnixStream, output: &[u8], offset: &mut usize) -> io::Result<()> {
    match stream.write(&output[*offset..]) {
        Ok(0) => Err(io::Error::new(
            io::ErrorKind::WriteZero,
            "reobservation response truncated",
        )),
        Ok(count) => {
            *offset += count;
            Ok(())
        }
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
            ) =>
        {
            Ok(())
        }
        Err(error) => Err(error),
    }
}

impl ChildReobserveControlV1 {
    /// Failure returns the exact input stream. Changing nonblocking mode affects
    /// every duplicate FD, hence the caller must own the exclusive description.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        parent: Arc<ConnectedParent>,
        stream: UnixStream,
        parent_pid: i32,
        parent_uid: u32,
        generation: u64,
        lock_sha256: String,
        ordinal: i32,
    ) -> Result<Self, (io::Error, UnixStream)> {
        let checked = (|| -> io::Result<()> {
            if generation == 0
                || ordinal < 0
                || lock_sha256.len() != 64
                || !lock_sha256.as_bytes().iter().all(u8::is_ascii_hexdigit)
                || !parent.matches(generation, &lock_sha256, ordinal)
            {
                return Err(invalid("reobservation original parent scope differs"));
            }
            parent.ensure_live()?;
            peer_matches(&stream, parent_pid, parent_uid)?;
            stream.set_nonblocking(true)?;
            parent.ensure_live()?;
            peer_matches(&stream, parent_pid, parent_uid)
        })();
        if let Err(error) = checked {
            return Err((error, stream));
        }
        Ok(Self {
            parent,
            stream,
            parent_pid,
            parent_uid,
            generation,
            lock_sha256,
            ordinal,
            last_sequence: 0,
            input: [0; CHILD_REOBSERVE_REQUEST_BYTES],
            input_read: 0,
            pending: None,
            output: None,
            first_failure: None,
        })
    }

    pub(super) fn first_failure(&self) -> Option<&ChildReobserveFailureV1> {
        self.first_failure.as_ref()
    }

    pub(super) fn has_pending_response(&self) -> bool {
        self.output.is_some()
    }

    /// Wait for the retained stream to become readable or writable within
    /// this observation window. Readiness is not a completed message, peer
    /// acknowledgement, native completion, or renewed execution authority.
    /// Timeout and interruption leave all offsets and the original owner intact.
    pub(super) fn wait_ready(&mut self, io_until: Instant) -> io::Result<bool> {
        self.check_peer()?;
        match wait_stream_ready(&self.stream, self.output.is_some(), io_until) {
            Ok(false) => Ok(false),
            Ok(true) => {
                self.check_peer()?;
                window(io_until)?;
                Ok(true)
            }
            Err(error) if error.kind() == io::ErrorKind::TimedOut => Err(error),
            Err(error) => Err(self.fail(error)),
        }
    }

    /// Offsets describe this exact stream and are not delivery acknowledgements.
    pub(super) fn progress(&self) -> (usize, usize) {
        (
            self.input_read,
            self.output.as_ref().map_or(0, |(_, written)| *written),
        )
    }

    fn fail(&mut self, error: io::Error) -> io::Error {
        if self.first_failure.is_none() {
            self.first_failure = Some(ChildReobserveFailureV1 {
                kind: error.kind(),
                message: error.to_string(),
                raw_os_error: error.raw_os_error(),
            });
        }
        error
    }

    fn check_peer(&mut self) -> io::Result<()> {
        if let Some(first) = &self.first_failure {
            return Err(io::Error::new(first.kind, first.message.clone()));
        }
        let result = (|| -> io::Result<()> {
            if !self
                .parent
                .matches(self.generation, &self.lock_sha256, self.ordinal)
            {
                return Err(invalid("reobservation original parent scope changed"));
            }
            self.parent.ensure_live()?;
            peer_matches(&self.stream, self.parent_pid, self.parent_uid)?;
            self.parent.ensure_live()
        })();
        result.map_err(|error| self.fail(error))
    }

    /// One nonblocking read. A finite I/O window limits this poll only;
    /// expiration preserves the exact frame/owner for a later observation
    /// window and does not renew native or original cleanup authority.
    pub(super) fn poll_request(
        &mut self,
        io_until: Instant,
    ) -> io::Result<Option<&ChildReobserveRequestV1>> {
        self.check_peer()?;
        window(io_until)?;
        if self.output.is_some() {
            return Ok(None);
        }
        if self.pending.is_none() && self.input_read < self.input.len() {
            if let Err(error) = read_step(&mut self.stream, &mut self.input, &mut self.input_read) {
                return Err(self.fail(error));
            }
            self.check_peer()?;
            window(io_until)?;
        }
        if self.pending.is_none() && self.input_read == self.input.len() {
            let request = match ChildReobserveRequestV1::decode(&self.input) {
                Ok(request) => request,
                Err(reason) => return Err(self.fail(invalid(reason))),
            };
            if request.generation != self.generation
                || !request.lock_sha256.eq_ignore_ascii_case(&self.lock_sha256)
                || self.last_sequence.checked_add(1) != Some(request.sequence)
            {
                return Err(self.fail(invalid("reobservation request scope or sequence differs")));
            }
            self.pending = Some(request);
        }
        Ok(self.pending.as_ref())
    }

    /// The caller chooses an observation from the same retained child owner.
    /// Invalid input returns the exact response and poisons the transport.
    pub(super) fn queue_response(
        &mut self,
        response: ChildReobserveResponseV1,
    ) -> Result<(), (io::Error, ChildReobserveResponseV1)> {
        if let Err(error) = self.check_peer() {
            return Err((error, response));
        }
        let matches = self
            .pending
            .as_ref()
            .is_some_and(|request| response.matches_request(request));
        if self.output.is_some() || !matches {
            return Err((
                self.fail(invalid("reobservation response scope or state differs")),
                response,
            ));
        }
        let encoded = match response.encode() {
            Ok(encoded) => encoded,
            Err(reason) => return Err((self.fail(invalid(reason)), response)),
        };
        self.output = Some((encoded, 0));
        Ok(())
    }

    /// One nonblocking write. Completion means bytes were written, not that
    /// the original parent accepted them or that physical work completed.
    pub(super) fn poll_write(&mut self, io_until: Instant) -> io::Result<bool> {
        self.check_peer()?;
        window(io_until)?;
        let Some((output, offset)) = self.output.as_mut() else {
            return Err(self.fail(invalid("reobservation response not queued")));
        };
        if *offset < output.len() {
            if let Err(error) = write_step(&mut self.stream, output, offset) {
                return Err(self.fail(error));
            }
            self.check_peer()?;
            window(io_until)?;
        }
        if self
            .output
            .as_ref()
            .is_some_and(|(output, offset)| *offset == output.len())
        {
            let Some(request) = self.pending.take() else {
                return Err(self.fail(invalid("reobservation request owner missing")));
            };
            self.last_sequence = request.sequence;
            self.output = None;
            self.input.fill(0);
            self.input_read = 0;
            return Ok(true);
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{net::Shutdown, time::Duration};

    #[test]
    fn finite_wait_can_resume_on_same_stream_and_eof_is_not_a_message() {
        let (mut writer, mut reader) = UnixStream::pair().unwrap();
        reader.set_nonblocking(true).unwrap();
        assert!(
            !wait_stream_ready(&reader, false, Instant::now() + Duration::from_millis(20)).unwrap()
        );
        writer.write_all(b"prefix").unwrap();
        writer.shutdown(Shutdown::Write).unwrap();
        assert!(
            wait_stream_ready(&reader, false, Instant::now() + Duration::from_secs(1)).unwrap()
        );
        let mut frame = [0u8; CHILD_REOBSERVE_REQUEST_BYTES];
        let mut offset = 0;
        read_step(&mut reader, &mut frame, &mut offset).unwrap();
        assert_eq!(&frame[..offset], b"prefix");
        assert!(
            wait_stream_ready(&reader, false, Instant::now() + Duration::from_secs(1)).unwrap()
        );
        assert_eq!(
            read_step(&mut reader, &mut frame, &mut offset)
                .unwrap_err()
                .kind(),
            io::ErrorKind::UnexpectedEof
        );
        assert_eq!(&frame[..offset], b"prefix");
    }

    #[test]
    fn partial_read_and_eof_keep_the_original_prefix() {
        let (mut writer, mut reader) = UnixStream::pair().unwrap();
        reader.set_nonblocking(true).unwrap();
        let mut frame = [0u8; CHILD_REOBSERVE_REQUEST_BYTES];
        let mut read = 0;
        writer.write_all(b"prefix").unwrap();
        read_step(&mut reader, &mut frame, &mut read).unwrap();
        assert_eq!(read, b"prefix".len());
        assert_eq!(&frame[..read], b"prefix");
        writer.shutdown(Shutdown::Write).unwrap();
        assert_eq!(
            read_step(&mut reader, &mut frame, &mut read)
                .unwrap_err()
                .kind(),
            io::ErrorKind::UnexpectedEof
        );
        assert_eq!(&frame[..read], b"prefix");
    }

    #[test]
    fn write_progress_uses_the_original_stream() {
        let (mut writer, mut reader) = UnixStream::pair().unwrap();
        writer.set_nonblocking(true).unwrap();
        let frame = [9u8; CHILD_REOBSERVE_RESPONSE_BYTES];
        let mut written = 0;
        write_step(&mut writer, &frame, &mut written).unwrap();
        assert!(written > 0);
        let mut observed = vec![0; written];
        reader.read_exact(&mut observed).unwrap();
        assert_eq!(&frame[..written], observed);
    }
}
