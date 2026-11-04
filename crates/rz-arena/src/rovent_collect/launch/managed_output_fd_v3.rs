//! Authenticated original managed-output descriptor and its live supervisor.
//! This capability says nothing about model execution or physical completion.
use cap_std::fs::Dir;
use nix::unistd::{geteuid, getppid};
use rz_platform::{ManagedOutputInheritedFds, ManagedOutputPeerCredentials};
use serde::Deserialize;
use std::io;
use std::thread;
use std::time::{Duration, Instant};

const DOMAIN: &str = "rz-managed-output-lease/1";
const MAX_PACKET: usize = 4096;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LeaseReply {
    schema: String,
    sequence: u64,
    owner_pid: u32,
    device: u64,
    inode: u64,
}

/// This owner always retains both original inherited descriptors and a
/// capability duplicate of the same output directory; no path is reopened.
pub(super) struct ObservedManagedOutputLeaseV3 {
    original: ManagedOutputInheritedFds,
    _directory: Dir,
    original_identity: (u64, u64),
    original_peer: ManagedOutputPeerCredentials,
    next_sequence: u64,
    tainted: bool,
}

impl ObservedManagedOutputLeaseV3 {
    pub(super) fn bind(original: ManagedOutputInheritedFds, deadline: Instant) -> io::Result<Self> {
        let peer = original.peer_credentials()?;
        if peer.pid <= 0 || peer.pid != getppid().as_raw() || peer.uid != geteuid().as_raw() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "managed output peer is not the original direct parent",
            ));
        }
        let original_identity = original.directory_identity()?;
        let directory = Dir::from_std_file(original.try_clone_directory()?);
        let mut owner = Self {
            original,
            _directory: directory,
            original_identity,
            original_peer: peer,
            next_sequence: 1,
            tainted: false,
        };
        owner.verify_live(deadline)?;
        Ok(owner)
    }

    /// Each invocation uses a fresh request/response on the same inherited
    /// socket. Once a request can have been delivered but fails, no later
    /// invocation may mistake its delayed ACK for a fresh liveness result.
    pub(super) fn verify_live(&mut self, deadline: Instant) -> io::Result<()> {
        if self.tainted {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "managed output control owner is uncertain",
            ));
        }
        let result = self.verify_live_inner(deadline);
        if result.is_err() {
            self.tainted = true;
        }
        result
    }

    fn verify_live_inner(&mut self, deadline: Instant) -> io::Result<()> {
        self.check_original_parent_and_directory()?;
        let sequence = self.next_sequence;
        let next_sequence = sequence.checked_add(1).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "managed output sequence exhausted",
            )
        })?;
        let request =
            serde_json::to_vec(&serde_json::json!({"schema": DOMAIN, "sequence": sequence}))
                .map_err(io::Error::other)?;
        loop {
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "managed output request deadline elapsed",
                ));
            }
            if self.original.try_send_packet(&request)? {
                break;
            }
            self.check_original_parent_and_directory()?;
            thread::sleep(
                deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(1)),
            );
        }
        let packet = loop {
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "managed output response deadline elapsed",
                ));
            }
            if let Some(packet) = self.original.try_receive_packet(MAX_PACKET)? {
                break packet;
            }
            self.check_original_parent_and_directory()?;
            thread::sleep(
                deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(1)),
            );
        };
        if packet.credentials.pid != self.original_peer.pid
            || packet.credentials.uid != self.original_peer.uid
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "managed output reply sender differs from original parent",
            ));
        }
        let reply: LeaseReply = serde_json::from_slice(&packet.bytes).map_err(io::Error::other)?;
        if reply.schema != DOMAIN
            || reply.sequence != sequence
            || i32::try_from(reply.owner_pid).ok() != Some(self.original_peer.pid)
            || (reply.device, reply.inode) != self.original_identity
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "managed output live lease or directory identity differs",
            ));
        }
        self.check_original_parent_and_directory()?;
        self.next_sequence = next_sequence;
        Ok(())
    }

    fn check_original_parent_and_directory(&self) -> io::Result<()> {
        if getppid().as_raw() != self.original_peer.pid
            || geteuid().as_raw() != self.original_peer.uid
            || self.original.peer_credentials()? != self.original_peer
            || self.original.directory_identity()? != self.original_identity
        {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "managed output original parent or directory changed",
            ));
        }
        Ok(())
    }
}
