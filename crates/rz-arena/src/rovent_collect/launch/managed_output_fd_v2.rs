//! Version /2 authenticated observation of an output lease and the exact
//! inherited input-directory and plan-file descriptors. This is not yet a
//! product write permit: the prepared model/lock owner and lease declaration
//! must be bound separately by the original collector supervisor.
use nix::unistd::{geteuid, getppid};
use rz_platform::{
    ManagedCollectionInheritedFds, ManagedInputInheritedFds, ManagedOutputInheritedFds,
    ManagedOutputPeerCredentials, managed_output_session_nonce,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const DOMAIN: &str = "rz-managed-output-lease/2";
const MAX_PACKET: usize = 4096;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObjectIdentity {
    device: u64,
    inode: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanIdentity {
    device: u64,
    inode: u64,
    size: u64,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LeaseReply {
    schema: String,
    sequence: u64,
    session_id: String,
    owner_pid: u32,
    output: ObjectIdentity,
    input_directory: ObjectIdentity,
    plan: PlanIdentity,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct PlanMetadata {
    device: u64,
    inode: u64,
    size: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}
fn plan_metadata(file: &std::fs::File) -> io::Result<PlanMetadata> {
    let meta = file.metadata()?;
    if !meta.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "original plan is not a regular file",
        ));
    }
    Ok(PlanMetadata {
        device: meta.dev(),
        inode: meta.ino(),
        size: meta.len(),
        modified: (meta.mtime(), meta.mtime_nsec()),
        changed: (meta.ctime(), meta.ctime_nsec()),
    })
}

/// Authenticates one proposed combination only. Consumers must still prove
/// that their exact prepared input owner, original lock and declared output
/// are the ones admitted by this same supervisor before enabling any write.
pub(super) struct ObservedManagedPlanLeaseV2 {
    output: ManagedOutputInheritedFds,
    input: ManagedInputInheritedFds,
    output_identity: (u64, u64),
    input_identity: (u64, u64),
    plan_original: PlanMetadata,
    expected_plan_sha256: String,
    original_peer: ManagedOutputPeerCredentials,
    session_id: String,
    next_sequence: u64,
    tainted: bool,
    original_binding: OriginalInputBindingV2,
}

/// Retains the original owners and first error even when setup did not finish.
/// A post-request failure is tainted and is not a successful handoff.
pub(super) enum ManagedPlanLeaseRejectionV2 {
    BeforeParts {
        error: io::Error,
        fds: ManagedCollectionInheritedFds,
    },
    BeforeRequest {
        error: io::Error,
        output: ManagedOutputInheritedFds,
        input: ManagedInputInheritedFds,
    },
    AfterRequest {
        error: io::Error,
        owner: ObservedManagedPlanLeaseV2,
    },
}

/// Opaque same-owner identity; possession does not grant output authority.
#[derive(Clone)]
pub(super) struct OriginalInputBindingV2(Arc<()>);
impl OriginalInputBindingV2 {
    pub(super) fn same_original(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl ObservedManagedPlanLeaseV2 {
    pub(super) fn bind(
        fds: ManagedCollectionInheritedFds,
        expected_plan_bytes: u64,
        expected_plan_sha256: String,
        maximum_plan_bytes: u64,
        deadline: Instant,
    ) -> Result<Self, ManagedPlanLeaseRejectionV2> {
        if expected_plan_bytes == 0
            || maximum_plan_bytes == 0
            || expected_plan_bytes > maximum_plan_bytes
            || expected_plan_sha256.len() != 64
            || !expected_plan_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(ManagedPlanLeaseRejectionV2::BeforeParts {
                error: io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "invalid declared plan bound or SHA-256",
                ),
                fds,
            });
        }
        let (output, input) = fds.into_parts();
        let preparation = (|| -> io::Result<_> {
            let peer = output.peer_credentials()?;
            if peer.pid <= 0 || peer.pid != getppid().as_raw() || peer.uid != geteuid().as_raw() {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "managed output peer is not the original direct parent",
                ));
            }
            let plan_original = plan_metadata(&input.try_clone_plan()?)?;
            if plan_original.size != expected_plan_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "original plan length differs from declaration",
                ));
            }
            let output_identity = output.directory_identity()?;
            let input_identity = input.directory_identity()?;
            let nonce = loop {
                if Instant::now() >= deadline {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "managed output /2 session setup deadline elapsed",
                    ));
                }
                match managed_output_session_nonce() {
                    Ok(value) => break value,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(error),
                }
            };
            Ok((peer, plan_original, output_identity, input_identity, nonce))
        })();
        let (peer, plan_original, output_identity, input_identity, nonce) = match preparation {
            Ok(value) => value,
            Err(error) => {
                return Err(ManagedPlanLeaseRejectionV2::BeforeRequest {
                    error,
                    output,
                    input,
                });
            }
        };
        let session_id = nonce.iter().map(|byte| format!("{byte:02x}")).collect();
        let mut owner = Self {
            output,
            input,
            output_identity,
            input_identity,
            plan_original,
            expected_plan_sha256,
            original_peer: peer,
            session_id,
            next_sequence: 1,
            tainted: false,
            original_binding: OriginalInputBindingV2(Arc::new(())),
        };
        if let Err(error) = owner.verify_live(deadline) {
            return Err(ManagedPlanLeaseRejectionV2::AfterRequest { error, owner });
        }
        Ok(owner)
    }

    /// The returned descriptors duplicate the original inherited file
    /// descriptions. They grant no output authority and do not make the plan
    /// immutable; a consumer must keep the observation owner alive.
    pub(super) fn clone_original_input(&self) -> io::Result<(std::fs::File, std::fs::File)> {
        Ok((
            self.input.try_clone_directory()?,
            self.input.try_clone_plan()?,
        ))
    }

    pub(super) fn original_binding(&self) -> OriginalInputBindingV2 {
        self.original_binding.clone()
    }

    pub(super) fn matches_plan(&self, bytes: u64, sha256: &str) -> bool {
        self.plan_original.size == bytes && self.expected_plan_sha256 == sha256
    }

    /// This is only a duplicate of the original output descriptor. Callers
    /// still need the actual native input-lock boundary before any mutation.
    pub(super) fn clone_original_output(&mut self, deadline: Instant) -> io::Result<std::fs::File> {
        self.verify_live(deadline)?;
        self.output.try_clone_directory()
    }

    pub(super) fn verify_live(&mut self, deadline: Instant) -> io::Result<()> {
        if self.tainted {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "managed output /2 control owner is uncertain",
            ));
        }
        let result = self.verify_live_inner(deadline);
        if result.is_err() {
            self.tainted = true;
        }
        result
    }

    fn verify_live_inner(&mut self, deadline: Instant) -> io::Result<()> {
        self.check_original_parent_and_fds()?;
        self.verify_plan(deadline)?;
        let sequence = self.next_sequence;
        let next_sequence = sequence.checked_add(1).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "managed output /2 sequence exhausted",
            )
        })?;
        let request = serde_json::to_vec(&serde_json::json!({"schema": DOMAIN, "sequence": sequence, "session_id": self.session_id.as_str()})).map_err(io::Error::other)?;
        loop {
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "managed output /2 request deadline elapsed",
                ));
            }
            if self.output.try_send_packet(&request)? {
                break;
            }
            self.check_original_parent_and_fds()?;
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
                    "managed output /2 response deadline elapsed",
                ));
            }
            if let Some(packet) = self.output.try_receive_packet(MAX_PACKET)? {
                break packet;
            }
            self.check_original_parent_and_fds()?;
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
                "managed output /2 reply sender differs",
            ));
        }
        let reply: LeaseReply = serde_json::from_slice(&packet.bytes).map_err(io::Error::other)?;
        if reply.schema != DOMAIN
            || reply.sequence != sequence
            || reply.session_id != self.session_id
            || i32::try_from(reply.owner_pid).ok() != Some(self.original_peer.pid)
            || (reply.output.device, reply.output.inode) != self.output_identity
            || (reply.input_directory.device, reply.input_directory.inode) != self.input_identity
            || (reply.plan.device, reply.plan.inode, reply.plan.size)
                != (
                    self.plan_original.device,
                    self.plan_original.inode,
                    self.plan_original.size,
                )
            || reply.plan.sha256 != self.expected_plan_sha256
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "managed output /2 live descriptor binding differs",
            ));
        }
        self.check_original_parent_and_fds()?;
        self.verify_plan(deadline)?;
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "managed output /2 reply verification exceeded deadline",
            ));
        }
        self.next_sequence = next_sequence;
        Ok(())
    }

    fn check_original_parent_and_fds(&self) -> io::Result<()> {
        if getppid().as_raw() != self.original_peer.pid
            || geteuid().as_raw() != self.original_peer.uid
            || self.output.peer_credentials()? != self.original_peer
            || self.output.directory_identity()? != self.output_identity
            || self.input.directory_identity()? != self.input_identity
            || self.input.plan_identity()?
                != (
                    self.plan_original.device,
                    self.plan_original.inode,
                    self.plan_original.size,
                )
        {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "managed output /2 original parent or descriptor changed",
            ));
        }
        Ok(())
    }

    fn verify_plan(&self, deadline: Instant) -> io::Result<()> {
        let plan = self.input.try_clone_plan()?;
        if plan_metadata(&plan)? != self.plan_original {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "original plan metadata changed",
            ));
        }
        let mut hash = Sha256::new();
        let mut offset = 0_u64;
        let mut buffer = [0_u8; 65536];
        while offset < self.plan_original.size {
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "original plan verification deadline elapsed",
                ));
            }
            let remaining = (self.plan_original.size - offset).min(buffer.len() as u64) as usize;
            match plan.read_at(&mut buffer[..remaining], offset) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "original plan was truncated",
                    ));
                }
                Ok(count) => {
                    hash.update(&buffer[..count]);
                    offset += count as u64;
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
        let digest = hash
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        if plan_metadata(&plan)? != self.plan_original || digest != self.expected_plan_sha256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "original plan bytes or metadata changed",
            ));
        }
        Ok(())
    }
}
