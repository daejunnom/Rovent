//! Shared, bounded transport for the native owner and its outer supervisor.
//! Frames and peer credentials are observations, not device admission,
//! physical completion, or authority to release an original owner.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;

pub const SCHEMA: &str = "rz-rovent-outer-supervisor-ipc/1";
pub const MAX_FRAME_BYTES: usize = 4096;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChildFrame {
    Enter { schema: String, supervisor_lock_sha256: String, supervisor_lock_device: u64, supervisor_lock_inode: u64, input_lock_sha256: String, input_lock_device: u64, input_lock_inode: u64, socket_dir_device: u64, socket_dir_inode: u64, process_epoch: u64, sequence: u64, device_ordinal: i32 },
    DeviceIdentity { schema: String, process_epoch: u64, sequence: u64, device_uuid_hex: String },
    AbortBeforeNative { schema: String, process_epoch: u64, sequence: u64, stage: String },
    Unknown { schema: String, process_epoch: u64, sequence: u64, stage: String },
    Fence { schema: String, process_epoch: u64, sequence: u64, evidence_sha256: String },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum SupervisorFrame {
    ProbeGrant { schema: String, sequence: u64, granted: bool, reason: String },
    EnterAck { schema: String, sequence: u64, accepted: bool, reason: String },
    ObservationAck { schema: String, sequence: u64, released: bool, reason: String },
}

/// Kernel peer observation only. Callers must independently bind it to the
/// original registered process, executable, start time, and admission scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnixPeerCredentials { pub pid: u32, pub uid: u32 }

pub fn unix_peer_credentials(stream: &UnixStream) -> io::Result<UnixPeerCredentials> {
    use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
    let peer = getsockopt(stream, PeerCredentials).map_err(io::Error::other)?;
    let pid = u32::try_from(peer.pid()).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid peer pid"))?;
    Ok(UnixPeerCredentials { pid, uid: peer.uid() })
}

/// Original stream with separate incoming and outgoing finite frame budgets.
/// Partial I/O is retained; EOF, malformed input and exhausted budgets are
/// errors, never physical completion. One frame must be acknowledged before
/// another is sent; pipelining is rejected.
pub struct BoundedWire {
    stream: UnixStream,
    input: Vec<u8>,
    output: Vec<u8>,
    output_offset: usize,
    max_frame: usize,
    remaining_incoming: u32,
    remaining_outgoing: u32,
}

struct FrameBuffer { bytes: Vec<u8>, maximum: usize }

impl Write for FrameBuffer {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if data.len() > self.maximum.saturating_sub(self.bytes.len()) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "IPC frame exceeds declared bound"));
        }
        self.bytes.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

fn serialize_frame<T: Serialize>(frame: &T, max_frame: usize) -> io::Result<Vec<u8>> {
    if !(1..=MAX_FRAME_BYTES).contains(&max_frame) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid IPC frame bound"));
    }
    let mut output = FrameBuffer { bytes: Vec::new(), maximum: max_frame - 1 };
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
    pub fn new(stream: UnixStream, max_frame: usize, max_messages: u32) -> io::Result<Self> {
        Self::try_new_owned(stream, max_frame, max_messages).map_err(|(error, _stream)| error)
    }

    /// A rejected setup returns the very same stream to its original owner.
    pub fn try_new_owned(stream: UnixStream, max_frame: usize, max_messages: u32) -> Result<Self, (io::Error, UnixStream)> {
        if !(1..=MAX_FRAME_BYTES).contains(&max_frame) || max_messages == 0 {
            return Err((io::Error::new(io::ErrorKind::InvalidInput, "invalid IPC bounds"), stream));
        }
        if let Err(error) = stream.set_nonblocking(true) {
            return Err((error, stream));
        }
        Ok(Self { stream, input: Vec::new(), output: Vec::new(), output_offset: 0,
            max_frame, remaining_incoming: max_messages, remaining_outgoing: max_messages })
    }

    pub fn poll_read(&mut self) -> io::Result<Option<ChildFrame>> { self.poll_frame() }
    pub fn poll_read_supervisor(&mut self) -> io::Result<Option<SupervisorFrame>> { self.poll_frame() }

    fn poll_frame<T: DeserializeOwned>(&mut self) -> io::Result<Option<T>> {
        if self.remaining_incoming == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "IPC input message budget exhausted"));
        }
        let mut chunk = [0u8; 1024];
        loop {
            match self.stream.read(&mut chunk) {
                Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "owner channel closed")),
                Ok(count) => {
                    if self.input.len().saturating_add(count) > self.max_frame {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "IPC frame exceeds declared bound"));
                    }
                    self.input.extend_from_slice(&chunk[..count]);
                    if let Some(newline) = self.input.iter().position(|byte| *byte == b'\n') {
                        if newline + 1 != self.input.len() {
                            return Err(io::Error::new(io::ErrorKind::InvalidData, "pipelined IPC frames unsupported"));
                        }
                        let frame = serde_json::from_slice(&self.input[..newline])
                            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid IPC frame"))?;
                        self.input.clear();
                        self.remaining_incoming -= 1;
                        return Ok(Some(frame));
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(None),
                // Return control so the caller can recheck its original
                // deadline. Already-read bytes remain in self.input.
                Err(error) if error.kind() == io::ErrorKind::Interrupted => return Ok(None),
                Err(error) => return Err(error),
            }
        }
    }

    pub fn queue<T: Serialize>(&mut self, frame: &T) -> io::Result<()> {
        if self.output_offset != self.output.len() {
            return Err(io::Error::new(io::ErrorKind::WouldBlock, "previous IPC response pending"));
        }
        if self.remaining_outgoing == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "IPC output message budget exhausted"));
        }
        let bytes = serialize_frame(frame, self.max_frame)?;
        self.remaining_outgoing -= 1;
        self.output = bytes;
        self.output_offset = 0;
        Ok(())
    }

    pub fn poll_write(&mut self) -> io::Result<bool> {
        while self.output_offset < self.output.len() {
            match self.stream.write(&self.output[self.output_offset..]) {
                Ok(0) => return Err(io::Error::new(io::ErrorKind::WriteZero, "owner channel closed")),
                Ok(count) => self.output_offset += count,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(false),
                // Preserve the write cursor while yielding deadline control.
                Err(error) if error.kind() == io::ErrorKind::Interrupted => return Ok(false),
                Err(error) => return Err(error),
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_directions_and_eof_preserve_distinct_states() {
        let (left, right) = UnixStream::pair().unwrap();
        let mut child = BoundedWire::new(left, 512, 2).unwrap();
        let mut supervisor = BoundedWire::new(right, 512, 2).unwrap();
        child.queue(&ChildFrame::Unknown { schema: SCHEMA.into(), process_epoch: 1, sequence: 2, stage: "driver".into() }).unwrap();
        assert!(child.poll_write().unwrap());
        assert!(matches!(supervisor.poll_read().unwrap(), Some(ChildFrame::Unknown { sequence: 2, .. })));
        supervisor.queue(&SupervisorFrame::ObservationAck { schema: SCHEMA.into(), sequence: 2, released: false, reason: "unknown".into() }).unwrap();
        assert!(supervisor.poll_write().unwrap());
        assert!(matches!(child.poll_read_supervisor().unwrap(), Some(SupervisorFrame::ObservationAck { released: false, .. })));
        drop(child);
        assert_eq!(supervisor.poll_read().unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn duplicate_fields_and_oversized_frames_are_rejected() {
        let (server, mut client) = UnixStream::pair().unwrap();
        let mut wire = BoundedWire::new(server, 256, 2).unwrap();
        client.write_all(b"{\"op\":\"unknown\",\"schema\":\"a\",\"schema\":\"b\",\"process_epoch\":1,\"sequence\":1,\"stage\":\"driver\"}\n").unwrap();
        assert_eq!(wire.poll_read().unwrap_err().kind(), io::ErrorKind::InvalidData);
        let (server, mut client) = UnixStream::pair().unwrap();
        let mut wire = BoundedWire::new(server, 8, 1).unwrap();
        client.write_all(b"123456789").unwrap();
        assert_eq!(wire.poll_read().unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn rejected_large_output_does_not_consume_the_send_budget() {
        let (left, right) = UnixStream::pair().unwrap();
        let mut child = BoundedWire::new(left, 256, 1).unwrap();
        let mut supervisor = BoundedWire::new(right, 256, 1).unwrap();
        let frame = |stage: String| ChildFrame::Unknown {
            schema: SCHEMA.into(), process_epoch: 1, sequence: 1, stage,
        };
        let actual = frame("driver".into());
        let exact_bytes = serde_json::to_vec(&actual).unwrap().len() + 1;
        assert!(validate_frame_fits(&actual, exact_bytes).is_ok());
        assert_eq!(validate_frame_fits(&actual, exact_bytes - 1).unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert_eq!(child.queue(&frame("x".repeat(4096))).unwrap_err().kind(), io::ErrorKind::InvalidData);
        child.queue(&actual).unwrap();
        assert!(child.poll_write().unwrap());
        assert!(matches!(supervisor.poll_read().unwrap(), Some(ChildFrame::Unknown { .. })));
    }
}
