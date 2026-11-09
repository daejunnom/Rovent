//! Shared, bounded transport for the native owner and its outer supervisor.
//! Frames and peer credentials are observations, not device admission,
//! physical completion, or authority to release an original owner.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::time::Instant;

pub const SCHEMA: &str = "rz-rovent-outer-supervisor-ipc/1";
pub const MAX_FRAME_BYTES: usize = 4096;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChildFrame {
    Enter {
        schema: String,
        supervisor_lock_sha256: String,
        supervisor_lock_device: u64,
        supervisor_lock_inode: u64,
        input_lock_sha256: String,
        input_lock_device: u64,
        input_lock_inode: u64,
        socket_dir_device: u64,
        socket_dir_inode: u64,
        process_epoch: u64,
        sequence: u64,
        device_ordinal: i32,
    },
    DeviceIdentity {
        schema: String,
        process_epoch: u64,
        sequence: u64,
        device_uuid_hex: String,
    },
    AbortBeforeNative {
        schema: String,
        process_epoch: u64,
        sequence: u64,
        stage: String,
    },
    Unknown {
        schema: String,
        process_epoch: u64,
        sequence: u64,
        stage: String,
    },
    Fence {
        schema: String,
        process_epoch: u64,
        sequence: u64,
        evidence_sha256: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum SupervisorFrame {
    ProbeGrant {
        schema: String,
        sequence: u64,
        granted: bool,
        reason: String,
    },
    EnterAck {
        schema: String,
        sequence: u64,
        accepted: bool,
        reason: String,
    },
    ObservationAck {
        schema: String,
        sequence: u64,
        released: bool,
        reason: String,
    },
}

/// Kernel peer observation only. Callers must independently bind it to the
/// original registered process, executable, start time, and admission scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnixPeerCredentials {
    pub pid: u32,
    pub uid: u32,
}

pub fn unix_peer_credentials(stream: &UnixStream) -> io::Result<UnixPeerCredentials> {
    use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
    let peer = getsockopt(stream, PeerCredentials).map_err(io::Error::other)?;
    let pid = u32::try_from(peer.pid())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid peer pid"))?;
    Ok(UnixPeerCredentials {
        pid,
        uid: peer.uid(),
    })
}

/// The semantic interpretation of a reply is deliberately left to the caller.
/// A peer credential or a matching sequence alone does not establish authority.
mod sealed {
    pub trait Sealed {}
}

pub trait ScopedFrame: sealed::Sealed + Serialize + DeserializeOwned {
    fn schema(&self) -> &str;
    fn sequence(&self) -> u64;
    fn process_epoch(&self) -> Option<u64>;
    fn kind(&self) -> FrameKind;
}

impl sealed::Sealed for ChildFrame {}
impl sealed::Sealed for SupervisorFrame {}

/// A request's expected response kind is chosen by its caller; this layer
/// does not invent mappings between the currently unconnected frame types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    Enter,
    DeviceIdentity,
    AbortBeforeNative,
    Unknown,
    Fence,
    ProbeGrant,
    EnterAck,
    ObservationAck,
}

impl ScopedFrame for ChildFrame {
    fn schema(&self) -> &str {
        match self {
            Self::Enter { schema, .. }
            | Self::DeviceIdentity { schema, .. }
            | Self::AbortBeforeNative { schema, .. }
            | Self::Unknown { schema, .. }
            | Self::Fence { schema, .. } => schema,
        }
    }
    fn sequence(&self) -> u64 {
        match self {
            Self::Enter { sequence, .. }
            | Self::DeviceIdentity { sequence, .. }
            | Self::AbortBeforeNative { sequence, .. }
            | Self::Unknown { sequence, .. }
            | Self::Fence { sequence, .. } => *sequence,
        }
    }
    fn process_epoch(&self) -> Option<u64> {
        Some(match self {
            Self::Enter { process_epoch, .. }
            | Self::DeviceIdentity { process_epoch, .. }
            | Self::AbortBeforeNative { process_epoch, .. }
            | Self::Unknown { process_epoch, .. }
            | Self::Fence { process_epoch, .. } => *process_epoch,
        })
    }
    fn kind(&self) -> FrameKind {
        match self {
            Self::Enter { .. } => FrameKind::Enter,
            Self::DeviceIdentity { .. } => FrameKind::DeviceIdentity,
            Self::AbortBeforeNative { .. } => FrameKind::AbortBeforeNative,
            Self::Unknown { .. } => FrameKind::Unknown,
            Self::Fence { .. } => FrameKind::Fence,
        }
    }
}

impl ScopedFrame for SupervisorFrame {
    fn schema(&self) -> &str {
        match self {
            Self::ProbeGrant { schema, .. }
            | Self::EnterAck { schema, .. }
            | Self::ObservationAck { schema, .. } => schema,
        }
    }
    fn sequence(&self) -> u64 {
        match self {
            Self::ProbeGrant { sequence, .. }
            | Self::EnterAck { sequence, .. }
            | Self::ObservationAck { sequence, .. } => *sequence,
        }
    }
    // Supervisor frames have no epoch field; their stream must independently
    // be bound to the original peer and scope by a future consumer.
    fn process_epoch(&self) -> Option<u64> {
        None
    }
    fn kind(&self) -> FrameKind {
        match self {
            Self::ProbeGrant { .. } => FrameKind::ProbeGrant,
            Self::EnterAck { .. } => FrameKind::EnterAck,
            Self::ObservationAck { .. } => FrameKind::ObservationAck,
        }
    }
}

#[derive(Debug)]
pub enum WireError {
    Io(io::Error),
    InvalidFrame,
    FrameTooLarge,
    InputBudgetExhausted,
    OutputBudgetExhausted,
    InvalidSchema,
    WrongEpoch,
    DuplicateSequence(u64),
    OutOfOrderSequence(u64),
    UnexpectedAck,
    WrongAck {
        expected: u64,
        actual: u64,
    },
    WrongAckKind {
        expected: FrameKind,
        actual: FrameKind,
    },
    AckPending(u64),
    WritePending,
    Pipeline,
    DeadlineExpired,
    WrongPhase,
    ClosedAfterError,
}

#[derive(Debug)]
pub enum ReadStatus<T> {
    Pending,
    AwaitingLocalAck {
        sequence: u64,
        peer_write_closed: bool,
    },
    Received(T),
}

#[derive(Clone, Copy)]
struct Flight {
    sequence: u64,
    deadline: Instant,
    expected_ack: Option<FrameKind>,
}

#[derive(Clone, Copy)]
enum Outgoing {
    Request,
    Ack,
}

/// Original stream, finite frame budgets, and a single in-flight exchange.
/// Partial I/O keeps its cursor and original hard deadline. A timeout, EOF,
/// malformed frame or mismatched ACK never establishes completion or makes
/// the owner reusable. The caller must retain the original stream and scope.
///
/// Additional bytes observed while awaiting a local ACK are a pipeline
/// violation. A Unix stream cannot prove when an unobserved byte was sent or
/// that the peer actually received an ACK; a future consumer needs its own
/// authenticated peer confirmation for those claims. This is not a GPU fence.
pub struct BoundedWire {
    stream: UnixStream,
    input: Vec<u8>,
    output: Vec<u8>,
    output_offset: usize,
    max_frame: usize,
    remaining_incoming: u32,
    remaining_outgoing: u32,
    expected_epoch: u64,
    incoming_deadline: Option<Instant>,
    inbound: Option<Flight>,
    outbound: Option<Flight>,
    outgoing: Option<Outgoing>,
    last_inbound: Option<u64>,
    last_outbound: Option<u64>,
    failed: bool,
}

struct FrameBuffer {
    bytes: Vec<u8>,
    maximum: usize,
}

impl Write for FrameBuffer {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if data.len() > self.maximum.saturating_sub(self.bytes.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IPC frame exceeds declared bound",
            ));
        }
        self.bytes.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn serialize_frame<T: Serialize>(frame: &T, max_frame: usize) -> io::Result<Vec<u8>> {
    if !(1..=MAX_FRAME_BYTES).contains(&max_frame) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid IPC frame bound",
        ));
    }
    let mut output = FrameBuffer {
        bytes: Vec::new(),
        maximum: max_frame - 1,
    };
    serde_json::to_writer(&mut output, frame).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, "invalid or oversized IPC frame")
    })?;
    output.bytes.push(b'\n');
    Ok(output.bytes)
}

/// Validate actual frame bytes, including the newline, without transmitting or
/// spending a live channel's message budget. This grants no IPC authority.
pub fn validate_frame_fits<T: Serialize>(frame: &T, max_frame: usize) -> io::Result<()> {
    serialize_frame(frame, max_frame).map(|_| ())
}

impl BoundedWire {
    /// A rejected setup returns the very same stream to its original owner.
    pub fn try_new_owned(
        stream: UnixStream,
        max_frame: usize,
        max_messages: u32,
        expected_epoch: u64,
    ) -> Result<Self, (io::Error, UnixStream)> {
        if !(1..=MAX_FRAME_BYTES).contains(&max_frame) || max_messages == 0 || expected_epoch == 0 {
            return Err((
                io::Error::new(io::ErrorKind::InvalidInput, "invalid IPC bounds"),
                stream,
            ));
        }
        if let Err(error) = stream.set_nonblocking(true) {
            return Err((error, stream));
        }
        Ok(Self {
            stream,
            input: Vec::new(),
            output: Vec::new(),
            output_offset: 0,
            max_frame,
            remaining_incoming: max_messages,
            remaining_outgoing: max_messages,
            expected_epoch,
            incoming_deadline: None,
            inbound: None,
            outbound: None,
            outgoing: None,
            last_inbound: None,
            last_outbound: None,
            failed: false,
        })
    }

    fn poll_frame<T: DeserializeOwned>(
        &mut self,
        deadline: Instant,
        allow_after_ack: bool,
    ) -> Result<Option<T>, WireError> {
        if self.remaining_incoming == 0 {
            return Err(WireError::InputBudgetExhausted);
        }
        let mut chunk = [0u8; 1024];
        loop {
            Self::before(deadline)?;
            // An ACK may have shared a read with a legitimate next request.
            // Consume buffered bytes before reading the socket again.
            if let Some(newline) = self.input.iter().position(|byte| *byte == b'\n') {
                if newline + 1 > self.max_frame {
                    return Err(WireError::FrameTooLarge);
                }
                if !allow_after_ack && newline + 1 != self.input.len() {
                    return Err(WireError::Pipeline);
                }
                let frame = serde_json::from_slice(&self.input[..newline])
                    .map_err(|_| WireError::InvalidFrame)?;
                // Keep the original bytes if the acceptance deadline expired.
                Self::before(deadline)?;
                return Ok(Some(frame));
            }
            if self.input.len() > self.max_frame {
                return Err(WireError::FrameTooLarge);
            }
            match self.stream.read(&mut chunk) {
                Ok(0) => {
                    return Err(WireError::Io(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "owner channel closed",
                    )));
                }
                Ok(count) => {
                    self.input.extend_from_slice(&chunk[..count]);
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(None),
                // Return control so the caller can recheck its original
                // deadline. Already-read bytes remain in self.input.
                Err(error) if error.kind() == io::ErrorKind::Interrupted => return Ok(None),
                Err(error) => return Err(WireError::Io(error)),
            }
        }
    }

    fn accept_frame(&mut self) {
        let end = self
            .input
            .iter()
            .position(|byte| *byte == b'\n')
            .expect("only a validated complete frame can be accepted");
        self.input.drain(..=end);
        self.remaining_incoming -= 1;
    }

    fn queue_frame<T: Serialize>(&mut self, frame: &T) -> Result<(), WireError> {
        if self.output_offset != self.output.len() {
            return Err(WireError::WritePending);
        }
        if self.remaining_outgoing == 0 {
            return Err(WireError::OutputBudgetExhausted);
        }
        let bytes = serialize_frame(frame, self.max_frame).map_err(|_| WireError::FrameTooLarge)?;
        self.remaining_outgoing -= 1;
        self.output = bytes;
        self.output_offset = 0;
        Ok(())
    }

    fn active(&self) -> Result<(), WireError> {
        if self.failed {
            Err(WireError::ClosedAfterError)
        } else {
            Ok(())
        }
    }

    fn validate<F: ScopedFrame>(&self, frame: &F) -> Result<(), WireError> {
        if frame.schema() != SCHEMA {
            return Err(WireError::InvalidSchema);
        }
        if frame
            .process_epoch()
            .is_some_and(|epoch| epoch != self.expected_epoch)
        {
            return Err(WireError::WrongEpoch);
        }
        Ok(())
    }

    fn before(deadline: Instant) -> Result<(), WireError> {
        if Instant::now() >= deadline {
            Err(WireError::DeadlineExpired)
        } else {
            Ok(())
        }
    }

    /// Start one request. The deadline is retained until its exact ACK arrives.
    pub fn queue_request<F: ScopedFrame>(
        &mut self,
        frame: &F,
        expected_ack: FrameKind,
        deadline: Instant,
    ) -> Result<(), WireError> {
        self.active()?;
        if let Some(flight) = self.outbound {
            return Err(WireError::AckPending(flight.sequence));
        }
        if self.inbound.is_some() || self.incoming_deadline.is_some() || !self.input.is_empty() {
            return Err(WireError::WrongPhase);
        }
        Self::before(deadline)?;
        self.validate(frame)?;
        if let Some(last) = self.last_outbound {
            if frame.sequence() == last {
                return Err(WireError::DuplicateSequence(last));
            }
            if frame.sequence() < last {
                return Err(WireError::OutOfOrderSequence(frame.sequence()));
            }
        }
        self.queue_frame(frame)?;
        self.outbound = Some(Flight {
            sequence: frame.sequence(),
            deadline,
            expected_ack: Some(expected_ack),
        });
        self.outgoing = Some(Outgoing::Request);
        Ok(())
    }

    /// Poll one incoming request. The earliest supplied deadline is retained
    /// across incomplete frames and retries; it can never be extended.
    pub fn poll_request<F: ScopedFrame>(
        &mut self,
        deadline: Instant,
    ) -> Result<ReadStatus<F>, WireError> {
        self.active()?;
        if self.outbound.is_some() {
            return Err(WireError::WrongPhase);
        }
        if let Some(flight) = self.inbound {
            Self::before(flight.deadline)?;
            return match self.detect_pipeline() {
                Ok(peer_write_closed) => Ok(ReadStatus::AwaitingLocalAck {
                    sequence: flight.sequence,
                    peer_write_closed,
                }),
                Err(error) => {
                    self.failed = true;
                    Err(error)
                }
            };
        }
        let deadline = self
            .incoming_deadline
            .map_or(deadline, |old| old.min(deadline));
        self.incoming_deadline = Some(deadline);
        Self::before(deadline)?;
        let frame = match self.poll_frame::<F>(deadline, false) {
            Ok(Some(frame)) => frame,
            Ok(None) => return Ok(ReadStatus::Pending),
            Err(error) => {
                self.failed = true;
                return Err(error);
            }
        };
        if let Err(error) = self.validate(&frame) {
            self.failed = true;
            return Err(error);
        }
        if let Some(last) = self.last_inbound {
            if frame.sequence() == last {
                self.failed = true;
                return Err(WireError::DuplicateSequence(last));
            }
            if frame.sequence() < last {
                self.failed = true;
                return Err(WireError::OutOfOrderSequence(frame.sequence()));
            }
        }
        Self::before(deadline)?;
        self.accept_frame();
        self.incoming_deadline = None;
        self.inbound = Some(Flight {
            sequence: frame.sequence(),
            deadline,
            expected_ack: None,
        });
        Ok(ReadStatus::Received(frame))
    }

    // A byte actually observed before a local ACK is queued is distinguishable
    // from no data. A race after this observation is not a peer-send timestamp.
    fn detect_pipeline(&mut self) -> Result<bool, WireError> {
        if !self.input.is_empty() {
            return Err(WireError::Pipeline);
        }
        let mut byte = [0u8; 1];
        match self.stream.read(&mut byte) {
            // A peer may stop writing while still waiting to read our ACK.
            // Report the half-close without treating it as a device fence.
            Ok(0) => Ok(true),
            Ok(_) => {
                // Keep the byte in the original owner for diagnosis; unread
                // trailing bytes stay on the same original stream.
                self.input.extend_from_slice(&byte);
                Err(WireError::Pipeline)
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                ) =>
            {
                Ok(false)
            }
            Err(error) => Err(WireError::Io(error)),
        }
    }

    /// Queue the response to the currently observed request; the flight stays
    /// active until the complete response has actually been written locally.
    pub fn queue_ack<F: ScopedFrame>(&mut self, frame: &F) -> Result<(), WireError> {
        self.active()?;
        let flight = self.inbound.ok_or(WireError::UnexpectedAck)?;
        Self::before(flight.deadline)?;
        self.validate(frame)?;
        if frame.sequence() != flight.sequence {
            return Err(WireError::WrongAck {
                expected: flight.sequence,
                actual: frame.sequence(),
            });
        }
        if let Err(error) = self.detect_pipeline() {
            self.failed = true;
            return Err(error);
        }
        self.queue_frame(frame)?;
        self.outgoing = Some(Outgoing::Ack);
        Ok(())
    }

    /// Receive the exact ACK for the original in-flight request. The caller
    /// still validates its meaning and actual peer identity independently.
    pub fn poll_ack<F: ScopedFrame>(&mut self) -> Result<ReadStatus<F>, WireError> {
        self.active()?;
        let flight = self.outbound.ok_or(WireError::UnexpectedAck)?;
        Self::before(flight.deadline)?;
        if self.output_offset != self.output.len() {
            return Err(WireError::WritePending);
        }
        let frame = match self.poll_frame::<F>(flight.deadline, true) {
            Ok(Some(frame)) => frame,
            Ok(None) => return Ok(ReadStatus::Pending),
            Err(error) => {
                self.failed = true;
                return Err(error);
            }
        };
        if let Err(error) = self.validate(&frame) {
            self.failed = true;
            return Err(error);
        }
        if frame.sequence() != flight.sequence {
            self.failed = true;
            return Err(WireError::WrongAck {
                expected: flight.sequence,
                actual: frame.sequence(),
            });
        }
        let expected = flight.expected_ack.ok_or(WireError::WrongPhase)?;
        if frame.kind() != expected {
            self.failed = true;
            return Err(WireError::WrongAckKind {
                expected,
                actual: frame.kind(),
            });
        }
        Self::before(flight.deadline)?;
        self.accept_frame();
        self.last_outbound = Some(flight.sequence);
        self.outbound = None;
        Ok(ReadStatus::Received(frame))
    }

    /// `true` only means the queued bytes were written locally within the
    /// deadline. A request remains outstanding until its matching ACK is
    /// received and validated; this is never a device completion fence.
    pub fn poll_write(&mut self) -> Result<bool, WireError> {
        self.active()?;
        let flight = match self.outgoing {
            Some(Outgoing::Request) => self.outbound,
            Some(Outgoing::Ack) => self.inbound,
            None => return Err(WireError::WrongPhase),
        }
        .ok_or(WireError::WrongPhase)?;
        Self::before(flight.deadline)?;
        while self.output_offset < self.output.len() {
            Self::before(flight.deadline)?;
            // Before writing each ACK fragment, classify additional input
            // actually observed while the original ACK is still unsent.
            // Arrival racing this check cannot establish peer send timing.
            if matches!(self.outgoing, Some(Outgoing::Ack)) {
                if let Err(error) = self.detect_pipeline() {
                    self.failed = true;
                    return Err(error);
                }
            }
            match self.stream.write(&self.output[self.output_offset..]) {
                Ok(0) => {
                    self.failed = true;
                    return Err(WireError::Io(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "owner channel closed",
                    )));
                }
                Ok(count) => self.output_offset += count,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(false),
                // Preserve the write cursor while yielding deadline control.
                Err(error) if error.kind() == io::ErrorKind::Interrupted => return Ok(false),
                Err(error) => {
                    self.failed = true;
                    return Err(WireError::Io(error));
                }
            }
        }
        // The bytes may already have been sent. A late local completion is
        // not an ACK success or permission to reuse the original flight.
        Self::before(flight.deadline)?;
        if matches!(self.outgoing.take(), Some(Outgoing::Ack)) {
            self.last_inbound = Some(flight.sequence);
            self.inbound = None;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_wire(stream: UnixStream, max_frame: usize, max_messages: u32) -> BoundedWire {
        match BoundedWire::try_new_owned(stream, max_frame, max_messages, 1) {
            Ok(wire) => wire,
            Err((error, _original_stream)) => panic!("test wire setup failed: {error}"),
        }
    }

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(2)
    }
    fn request(sequence: u64) -> ChildFrame {
        ChildFrame::Unknown {
            schema: SCHEMA.into(),
            process_epoch: 1,
            sequence,
            stage: "driver".into(),
        }
    }
    fn ack(sequence: u64) -> SupervisorFrame {
        SupervisorFrame::ObservationAck {
            schema: SCHEMA.into(),
            sequence,
            released: false,
            reason: "unknown".into(),
        }
    }
    fn send_raw<T: Serialize>(stream: &mut UnixStream, frame: &T) {
        let mut bytes = serde_json::to_vec(frame).unwrap();
        bytes.push(b'\n');
        stream.write_all(&bytes).unwrap();
    }
    fn drain_line(stream: &mut UnixStream) {
        let mut byte = [0u8; 1];
        loop {
            stream.read_exact(&mut byte).unwrap();
            if byte[0] == b'\n' {
                break;
            }
        }
    }

    #[test]
    fn exchange_requires_matching_ack_and_preserves_owner_on_eof() {
        let (left, right) = UnixStream::pair().unwrap();
        let mut child = make_wire(left, 512, 2);
        let mut supervisor = make_wire(right, 512, 2);
        child
            .queue_request(&request(2), FrameKind::ObservationAck, deadline())
            .unwrap();
        assert!(child.poll_write().unwrap());
        assert!(matches!(
            child.queue_request(&request(3), FrameKind::ObservationAck, deadline()),
            Err(WireError::AckPending(2))
        ));
        assert!(matches!(
            child.poll_ack::<SupervisorFrame>().unwrap(),
            ReadStatus::Pending
        ));
        assert!(matches!(
            supervisor.poll_request::<ChildFrame>(deadline()).unwrap(),
            ReadStatus::Received(ChildFrame::Unknown { sequence: 2, .. })
        ));
        assert!(matches!(
            supervisor.poll_request::<ChildFrame>(deadline()).unwrap(),
            ReadStatus::AwaitingLocalAck {
                sequence: 2,
                peer_write_closed: false
            }
        ));
        supervisor.queue_ack(&ack(2)).unwrap();
        assert!(supervisor.poll_write().unwrap());
        assert!(matches!(
            child.poll_ack::<SupervisorFrame>().unwrap(),
            ReadStatus::Received(SupervisorFrame::ObservationAck {
                released: false,
                ..
            })
        ));
        assert!(matches!(
            child.poll_ack::<SupervisorFrame>(),
            Err(WireError::UnexpectedAck)
        ));
        assert!(matches!(
            child.queue_request(&request(2), FrameKind::ObservationAck, deadline()),
            Err(WireError::DuplicateSequence(2))
        ));
        assert!(matches!(
            child.queue_request(&request(1), FrameKind::ObservationAck, deadline()),
            Err(WireError::OutOfOrderSequence(1))
        ));
        drop(child);
        assert!(
            matches!(supervisor.poll_request::<ChildFrame>(deadline()), Err(WireError::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof)
        );
        assert!(matches!(
            supervisor.poll_request::<ChildFrame>(deadline()),
            Err(WireError::ClosedAfterError)
        ));
    }

    #[test]
    fn duplicate_fields_and_oversized_frames_are_rejected() {
        let (server, mut client) = UnixStream::pair().unwrap();
        let mut wire = make_wire(server, 256, 2);
        client.write_all(b"{\"op\":\"unknown\",\"schema\":\"a\",\"schema\":\"b\",\"process_epoch\":1,\"sequence\":1,\"stage\":\"driver\"}\n").unwrap();
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()),
            Err(WireError::InvalidFrame)
        ));
        let (server, mut client) = UnixStream::pair().unwrap();
        let mut wire = make_wire(server, 8, 1);
        client.write_all(b"123456789").unwrap();
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()),
            Err(WireError::FrameTooLarge)
        ));
    }

    #[test]
    fn rejected_large_output_does_not_consume_the_send_budget() {
        let (left, right) = UnixStream::pair().unwrap();
        let mut child = make_wire(left, 256, 1);
        let mut supervisor = make_wire(right, 256, 1);
        let frame = |stage: String| ChildFrame::Unknown {
            schema: SCHEMA.into(),
            process_epoch: 1,
            sequence: 1,
            stage,
        };
        let actual = frame("driver".into());
        let exact_bytes = serde_json::to_vec(&actual).unwrap().len() + 1;
        assert!(validate_frame_fits(&actual, exact_bytes).is_ok());
        assert_eq!(
            validate_frame_fits(&actual, exact_bytes - 1)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        assert!(matches!(
            child.queue_request(
                &frame("x".repeat(4096)),
                FrameKind::ObservationAck,
                deadline()
            ),
            Err(WireError::FrameTooLarge)
        ));
        child
            .queue_request(&actual, FrameKind::ObservationAck, deadline())
            .unwrap();
        assert!(child.poll_write().unwrap());
        assert!(matches!(
            supervisor.poll_request::<ChildFrame>(deadline()).unwrap(),
            ReadStatus::Received(ChildFrame::Unknown { .. })
        ));
    }

    #[test]
    fn rejected_setup_returns_the_original_stream() {
        let (stream, mut peer) = UnixStream::pair().unwrap();
        let (error, mut original) = match BoundedWire::try_new_owned(stream, 512, 1, 0) {
            Err(rejection) => rejection,
            Ok(_) => panic!("invalid epoch must be rejected"),
        };
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        peer.write_all(b"x").unwrap();
        let mut received = [0u8; 1];
        original.read_exact(&mut received).unwrap();
        assert_eq!(&received, b"x");
    }

    #[test]
    fn partial_input_preserves_bytes_and_original_deadline() {
        let (server, mut client) = UnixStream::pair().unwrap();
        let mut wire = make_wire(server, 512, 2);
        let mut bytes = serde_json::to_vec(&request(4)).unwrap();
        bytes.push(b'\n');
        client.write_all(&bytes[..8]).unwrap();
        let original = deadline();
        assert!(matches!(
            wire.poll_request::<ChildFrame>(original).unwrap(),
            ReadStatus::Pending
        ));
        client.write_all(&bytes[8..]).unwrap();
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()).unwrap(),
            ReadStatus::Received(ChildFrame::Unknown { sequence: 4, .. })
        ));

        let (server, mut client) = UnixStream::pair().unwrap();
        let mut wire = make_wire(server, 512, 2);
        client.write_all(&bytes[..8]).unwrap();
        let short = deadline();
        assert!(matches!(
            wire.poll_request::<ChildFrame>(short).unwrap(),
            ReadStatus::Pending
        ));
        // Advance the observed monotonic state without relying on scheduler timing.
        wire.incoming_deadline = Some(Instant::now() - Duration::from_millis(1));
        client.write_all(&bytes[8..]).unwrap();
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()),
            Err(WireError::DeadlineExpired)
        ));
    }

    #[test]
    fn observed_pipeline_is_distinct_from_waiting_and_oversize() {
        let (server, mut client) = UnixStream::pair().unwrap();
        let mut wire = make_wire(server, 512, 3);
        send_raw(&mut client, &request(1));
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()).unwrap(),
            ReadStatus::Received(_)
        ));
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()).unwrap(),
            ReadStatus::AwaitingLocalAck {
                sequence: 1,
                peer_write_closed: false
            }
        ));
        send_raw(&mut client, &request(2));
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()),
            Err(WireError::Pipeline)
        ));
        assert!(matches!(
            wire.queue_ack(&ack(1)),
            Err(WireError::ClosedAfterError)
        ));

        let (server, mut client) = UnixStream::pair().unwrap();
        let mut bytes = serde_json::to_vec(&request(1)).unwrap();
        bytes.push(b'\n');
        let frame_len = bytes.len();
        bytes.extend_from_slice(&serde_json::to_vec(&request(2)).unwrap());
        bytes.push(b'\n');
        let mut wire = make_wire(server, frame_len, 3);
        client.write_all(&bytes).unwrap();
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()),
            Err(WireError::Pipeline)
        ));

        let (server, mut client) = UnixStream::pair().unwrap();
        let mut wire = make_wire(server, 512, 3);
        send_raw(&mut client, &request(1));
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()).unwrap(),
            ReadStatus::Received(_)
        ));
        wire.queue_ack(&ack(1)).unwrap();
        // A second request arriving after queue_ack but before any ACK bytes
        // are sent must not be accepted as a post-ACK request.
        send_raw(&mut client, &request(2));
        assert!(matches!(wire.poll_write(), Err(WireError::Pipeline)));
        assert!(!wire.input.is_empty());
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()),
            Err(WireError::ClosedAfterError)
        ));
    }

    #[test]
    fn request_after_matching_ack_is_not_rejected_as_pipeline() {
        let (left, right) = UnixStream::pair().unwrap();
        let mut child = make_wire(left, 512, 3);
        let mut supervisor = make_wire(right, 512, 3);
        child
            .queue_request(&request(1), FrameKind::ObservationAck, deadline())
            .unwrap();
        assert!(child.poll_write().unwrap());
        assert!(matches!(
            supervisor.poll_request::<ChildFrame>(deadline()).unwrap(),
            ReadStatus::Received(_)
        ));
        supervisor.queue_ack(&ack(1)).unwrap();
        assert!(supervisor.poll_write().unwrap());
        assert!(matches!(
            child.poll_ack::<SupervisorFrame>().unwrap(),
            ReadStatus::Received(_)
        ));
        child
            .queue_request(&request(2), FrameKind::ObservationAck, deadline())
            .unwrap();
        assert!(child.poll_write().unwrap());
        assert!(matches!(
            supervisor.poll_request::<ChildFrame>(deadline()).unwrap(),
            ReadStatus::Received(ChildFrame::Unknown { sequence: 2, .. })
        ));
    }

    #[test]
    fn ack_and_legitimate_following_request_can_share_a_read() {
        let (server, mut peer) = UnixStream::pair().unwrap();
        let mut wire = make_wire(server, 512, 3);
        wire.queue_request(&request(1), FrameKind::ObservationAck, deadline())
            .unwrap();
        assert!(wire.poll_write().unwrap());
        drain_line(&mut peer);
        let mut bytes = serde_json::to_vec(&ack(1)).unwrap();
        bytes.push(b'\n');
        bytes.extend_from_slice(&serde_json::to_vec(&request(2)).unwrap());
        bytes.push(b'\n');
        // Deterministically exercise one coalesced read without depending on
        // kernel packet scheduling; separate tests use actual socket writes.
        wire.input.extend_from_slice(&bytes);
        assert!(matches!(
            wire.poll_ack::<SupervisorFrame>().unwrap(),
            ReadStatus::Received(_)
        ));
        assert!(matches!(
            wire.queue_request(&request(3), FrameKind::ObservationAck, deadline()),
            Err(WireError::WrongPhase)
        ));
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()).unwrap(),
            ReadStatus::Received(ChildFrame::Unknown { sequence: 2, .. })
        ));
    }

    #[test]
    fn duplicate_wrong_scope_and_wrong_ack_are_not_completion() {
        let (server, mut client) = UnixStream::pair().unwrap();
        let mut wire = make_wire(server, 512, 3);
        send_raw(&mut client, &request(4));
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()).unwrap(),
            ReadStatus::Received(_)
        ));
        assert!(matches!(
            wire.queue_ack(&ack(5)),
            Err(WireError::WrongAck {
                expected: 4,
                actual: 5
            })
        ));
        wire.queue_ack(&ack(4)).unwrap();
        assert!(wire.poll_write().unwrap());
        drain_line(&mut client);
        send_raw(&mut client, &request(4));
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()),
            Err(WireError::DuplicateSequence(4))
        ));

        let (server, mut client) = UnixStream::pair().unwrap();
        let mut wire = make_wire(server, 512, 3);
        let wrong_epoch = ChildFrame::Unknown {
            schema: SCHEMA.into(),
            process_epoch: 9,
            sequence: 1,
            stage: "driver".into(),
        };
        send_raw(&mut client, &wrong_epoch);
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()),
            Err(WireError::WrongEpoch)
        ));

        let (server, mut client) = UnixStream::pair().unwrap();
        let mut wire = make_wire(server, 512, 3);
        wire.queue_request(&request(4), FrameKind::ObservationAck, deadline())
            .unwrap();
        assert!(wire.poll_write().unwrap());
        drain_line(&mut client);
        send_raw(&mut client, &ack(5));
        assert!(matches!(
            wire.poll_ack::<SupervisorFrame>(),
            Err(WireError::WrongAck {
                expected: 4,
                actual: 5
            })
        ));
        assert!(matches!(
            wire.queue_request(&request(5), FrameKind::ObservationAck, deadline()),
            Err(WireError::ClosedAfterError)
        ));

        let (server, mut client) = UnixStream::pair().unwrap();
        let mut wire = make_wire(server, 512, 3);
        wire.queue_request(&request(4), FrameKind::ObservationAck, deadline())
            .unwrap();
        assert!(wire.poll_write().unwrap());
        drain_line(&mut client);
        send_raw(
            &mut client,
            &SupervisorFrame::EnterAck {
                schema: SCHEMA.into(),
                sequence: 4,
                accepted: false,
                reason: "unknown".into(),
            },
        );
        assert!(matches!(
            wire.poll_ack::<SupervisorFrame>(),
            Err(WireError::WrongAckKind {
                expected: FrameKind::ObservationAck,
                actual: FrameKind::EnterAck,
            })
        ));
        assert!(matches!(
            wire.queue_request(&request(5), FrameKind::ObservationAck, deadline()),
            Err(WireError::ClosedAfterError)
        ));
    }

    #[test]
    fn write_half_close_is_not_mistaken_for_pipeline_or_physical_completion() {
        let (server, mut client) = UnixStream::pair().unwrap();
        let mut wire = make_wire(server, 512, 3);
        send_raw(&mut client, &request(1));
        client.shutdown(std::net::Shutdown::Write).unwrap();
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()).unwrap(),
            ReadStatus::Received(_)
        ));
        assert!(matches!(
            wire.poll_request::<ChildFrame>(deadline()).unwrap(),
            ReadStatus::AwaitingLocalAck {
                sequence: 1,
                peer_write_closed: true
            }
        ));
        wire.queue_ack(&ack(1)).unwrap();
        assert!(wire.poll_write().unwrap());
        drain_line(&mut client);
        assert!(
            matches!(wire.poll_request::<ChildFrame>(deadline()), Err(WireError::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof)
        );
    }

    #[test]
    fn late_ack_does_not_clear_the_original_flight() {
        let (server, mut client) = UnixStream::pair().unwrap();
        let mut wire = make_wire(server, 512, 3);
        let short = deadline();
        wire.queue_request(&request(4), FrameKind::ObservationAck, short)
            .unwrap();
        assert!(wire.poll_write().unwrap());
        drain_line(&mut client);
        wire.outbound.as_mut().unwrap().deadline = Instant::now() - Duration::from_millis(1);
        send_raw(&mut client, &ack(4));
        assert!(matches!(
            wire.poll_ack::<SupervisorFrame>(),
            Err(WireError::DeadlineExpired)
        ));
        assert!(matches!(
            wire.queue_request(&request(5), FrameKind::ObservationAck, deadline()),
            Err(WireError::AckPending(4))
        ));
    }
}
