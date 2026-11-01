//! Separate supervisor registration for a Rovent V3 native execution.
//!
//! A declaration, canonical digest, or pinned lock is not an authorization,
//! process identity, device reservation, provider observation, or GPU fence.
//! The actual owners must independently verify executable bytes, peer and
//! device observations, and retain the original runtime and IPC capabilities.

use crate::{ArtifactRef, ManifestError, NativeRuntimeSourceV6, RoventInputLockV3, decode_json, digest};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

pub const ROVENT_SUPERVISOR_DECLARATION_V1_DOMAIN: &str = "rz-rovent-supervisor-declaration/1";
pub const ROVENT_SUPERVISOR_LOCK_V1_DOMAIN: &str = "rz-rovent-supervisor-lock/1";
pub const ROVENT_SUPERVISOR_CANONICAL_V1_DOMAIN: &str = "rz-rovent-supervisor-canonical/1";
/// This is a bound on one independently registered metadata file, not a
/// project storage quota or a limit on model and execution assets.
pub const ROVENT_SUPERVISOR_LOCK_V1_MAX_BYTES: u64 = 64 * 1024;

fn invalid(reason: &str) -> ManifestError {
    ManifestError::Integrity(format!("Rovent supervisor /1: {reason}"))
}

fn require(condition: bool, reason: &str) -> Result<(), ManifestError> {
    if condition { Ok(()) } else { Err(invalid(reason)) }
}

fn identifier(value: &str, max: usize) -> bool {
    !value.is_empty()
        && value.len() <= max
        && value != "."
        && value != ".."
        && value.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
}

/// Linux supervisor relative directory name. Hidden directories are allowed;
/// neither a name nor validation grants access to the original root FD. This
/// check does not establish Windows path compatibility.
fn reservation_directory(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 1024
        && !value.starts_with('/')
        && !value.contains(['\\', ':'])
        && !value.chars().any(char::is_control)
        && value.split('/').all(|part| {
            !part.is_empty() && part != "." && part != ".." && part.trim() == part
        })
}

fn sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        && value.bytes().any(|b| b != b'0')
}

/// Only the supported admission policy can be declared; it does not itself
/// reserve any resource. Runtime admission must hold the original capability.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventSupervisorAdmissionV1 {
    ProvisionalReservationBeforeNative,
}

/// This is a requirement on the owner, never a claim that a fence exists.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventSupervisorReleaseV1 {
    SameOriginalOwnerPhysicalFence,
}

/// Conservative reservation domain when aliases or a physical parent device
/// are not established. This excludes nonparticipating programs and is not
/// evidence of GPU-wide exclusivity.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventSupervisorReservationScopeV1 {
    ParticipatingRoventCudaGlobalV1,
}

/// Declarations bind requested inputs. Actual source, peer, UUID/context,
/// reservation and completion observations belong to separate owner receipts.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventSupervisorDeclarationV1 {
    pub domain: String,
    pub authorization_id: String,
    pub owner_id: String,
    pub input_lock_sha256: String,
    /// Supervisor process-generation label, not a V3 model or root epoch.
    /// The runtime owner must verify the live generation independently.
    pub process_epoch: u64,
    /// The native engine's semantic role; this is not a chess color.
    pub engine_role: String,
    pub supervisor_binary: ArtifactRef,
    pub engine_binary: ArtifactRef,
    /// Relative socket name resolved from the original pinned directory FD.
    pub socket_basename: String,
    /// Directory relative to the independently held RoveZero managed root.
    /// Its name is a declaration, not a capability or proof of common inode.
    pub reservation_root: String,
    pub reservation_namespace: String,
    pub reservation_scope: RoventSupervisorReservationScopeV1,
    pub device_ordinal: u32,
    pub driver_library_role: String,
    pub startup_ms: u64,
    pub cleanup_ms: u64,
    /// Protocol bounds are independent of storage and execution budgets.
    pub max_frame_bytes: u32,
    pub max_messages: u32,
    pub admission: RoventSupervisorAdmissionV1,
    pub release: RoventSupervisorReleaseV1,
}

impl RoventSupervisorDeclarationV1 {
    pub fn validate(&self) -> Result<(), ManifestError> {
        require(self.domain == ROVENT_SUPERVISOR_DECLARATION_V1_DOMAIN, "declaration domain mismatch")?;
        require(identifier(&self.authorization_id, 128) && identifier(&self.owner_id, 128)
            && identifier(&self.engine_role, 64) && identifier(&self.driver_library_role, 128)
            && identifier(&self.socket_basename, 80)
            && identifier(&self.reservation_namespace, 128)
            && reservation_directory(&self.reservation_root),
            "invalid bounded identity, socket basename, or relative reservation root")?;
        require(sha256(&self.input_lock_sha256) && self.process_epoch != 0, "invalid V3 lock digest or process epoch")?;
        self.supervisor_binary.validate()?;
        self.engine_binary.validate()?;
        require(self.startup_ms != 0 && self.cleanup_ms != 0
            && i32::try_from(self.device_ordinal).is_ok()
            && (1..=4096).contains(&self.max_frame_bytes)
            && self.max_messages != 0, "invalid finite control limits")
    }

    pub fn validate_for_v3(&self, v3: &RoventInputLockV3) -> Result<(), ManifestError> {
        self.validate()?;
        v3.validate()?;
        require(self.input_lock_sha256 == v3.canonical_sha256
            && self.engine_binary == v3.manifest.engine.binary
            && self.startup_ms <= v3.manifest.wall_time_max_ms
            && self.cleanup_ms <= v3.manifest.cleanup_max_ms,
            "V3 digest, engine declaration, or control budget mismatch")?;
        match &v3.manifest.system_runtime_lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(spec) => require(
                u32::try_from(spec.cuda_device_id).ok() == Some(self.device_ordinal)
                    && spec.libraries.iter().any(|library| library.role == self.driver_library_role),
                "CUDA ordinal or registered driver role differs from the V3 runtime declaration",
            ),
            NativeRuntimeSourceV6::Cpu => Err(invalid("supervisor /1 requires an explicit system CUDA declaration")),
        }
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        let value = serde_json::to_value((ROVENT_SUPERVISOR_CANONICAL_V1_DOMAIN, self))
            .map_err(|error| invalid(&format!("canonical value: {error}")))?;
        let bytes = serde_json::to_vec(&value).map_err(|error| invalid(&format!("canonical bytes: {error}")))?;
        require(bytes.len() as u64 <= ROVENT_SUPERVISOR_LOCK_V1_MAX_BYTES, "canonical declaration exceeds its metadata bound")?;
        Ok(bytes)
    }

    pub fn lock(self) -> Result<RoventSupervisorLockV1, ManifestError> {
        let canonical_sha256 = digest(&self.canonical_bytes()?);
        Ok(RoventSupervisorLockV1 { domain: ROVENT_SUPERVISOR_LOCK_V1_DOMAIN.into(), declaration: self, canonical_sha256 })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventSupervisorLockV1 {
    pub domain: String,
    pub declaration: RoventSupervisorDeclarationV1,
    pub canonical_sha256: String,
}

impl RoventSupervisorLockV1 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        require(!input.is_empty() && input.len() as u64 <= ROVENT_SUPERVISOR_LOCK_V1_MAX_BYTES,
            "registered lock is empty or exceeds its metadata bound")?;
        let lock: Self = decode_json(input)?;
        lock.validate()?;
        Ok(lock)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require(self.domain == ROVENT_SUPERVISOR_LOCK_V1_DOMAIN
            && sha256(&self.canonical_sha256)
            && digest(&self.declaration.canonical_bytes()?) == self.canonical_sha256,
            "lock domain or canonical digest mismatch")
    }

    pub fn validate_for_v3(&self, v3: &RoventInputLockV3) -> Result<(), ManifestError> {
        self.validate()?;
        self.declaration.validate_for_v3(v3)
    }
}

/// Original descriptor for an independently registered raw lock file. This
/// pins the source used during validation, not the authority of the writer.
/// As with `ArtifactRef::open_verified_at`, the owner must prevent concurrent
/// mutation of the inode throughout its use; an open FD alone cannot do so.
#[derive(Debug)]
pub struct RoventSupervisorPinnedLockV1 {
    file: File,
    raw_artifact: ArtifactRef,
    lock: RoventSupervisorLockV1,
}

impl RoventSupervisorPinnedLockV1 {
    pub fn pin_registered_at(root: &cap_std::fs::Dir, artifact: &ArtifactRef, v3: &RoventInputLockV3) -> Result<Self, ManifestError> {
        require(artifact.bytes <= ROVENT_SUPERVISOR_LOCK_V1_MAX_BYTES, "registered lock exceeds its metadata bound")?;
        let mut file = artifact.open_verified_at(root, ROVENT_SUPERVISOR_LOCK_V1_MAX_BYTES)?;
        let mut bytes = Vec::new();
        file.by_ref().take(ROVENT_SUPERVISOR_LOCK_V1_MAX_BYTES + 1).read_to_end(&mut bytes)
            .map_err(|error| invalid(&format!("read registered lock: {error}")))?;
        require(bytes.len() as u64 == artifact.bytes && digest(&bytes) == artifact.sha256,
            "registered raw lock changed while reading")?;
        let raw = std::str::from_utf8(&bytes).map_err(|_| invalid("registered lock is not UTF-8"))?;
        let lock = RoventSupervisorLockV1::from_json(raw)?;
        lock.validate_for_v3(v3)?;
        file.seek(SeekFrom::Start(0)).map_err(|error| invalid(&format!("rewind registered lock: {error}")))?;
        Ok(Self { file, raw_artifact: artifact.clone(), lock })
    }

    pub fn file(&self) -> &File { &self.file }
    pub fn raw_artifact(&self) -> &ArtifactRef { &self.raw_artifact }
    pub fn lock(&self) -> &RoventSupervisorLockV1 { &self.lock }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(path: &str, byte: char) -> ArtifactRef {
        ArtifactRef {
            path: path.into(),
            sha256: byte.to_string().repeat(64),
            bytes: 32,
            source: "https://example.org/rovent-fixture".into(),
            license: "MIT".into(),
        }
    }

    fn declaration() -> RoventSupervisorDeclarationV1 {
        RoventSupervisorDeclarationV1 {
            domain: ROVENT_SUPERVISOR_DECLARATION_V1_DOMAIN.into(),
            authorization_id: "approval-1".into(),
            owner_id: "owner-1".into(),
            input_lock_sha256: "a".repeat(64),
            process_epoch: 23,
            engine_role: "rovent-pc".into(),
            supervisor_binary: artifact("bin/supervisor", 'b'),
            engine_binary: artifact("bin/rovent-native", 'c'),
            socket_basename: "owner-1.sock".into(),
            reservation_root: "run/gpu-reservations".into(),
            reservation_namespace: "rovent-cuda".into(),
            reservation_scope: RoventSupervisorReservationScopeV1::ParticipatingRoventCudaGlobalV1,
            device_ordinal: 0,
            driver_library_role: "cuda-driver".into(),
            startup_ms: 10_000,
            cleanup_ms: 5_000,
            max_frame_bytes: 4_096,
            max_messages: 16,
            admission: RoventSupervisorAdmissionV1::ProvisionalReservationBeforeNative,
            release: RoventSupervisorReleaseV1::SameOriginalOwnerPhysicalFence,
        }
    }

    #[test]
    fn lock_round_trip_is_bound_to_each_declared_identity() {
        let lock = declaration().lock().unwrap();
        let wire = serde_json::to_string(&lock).unwrap();
        assert_eq!(RoventSupervisorLockV1::from_json(&wire).unwrap(), lock);

        let mut variants = Vec::new();
        let mut changed = lock.clone(); changed.declaration.authorization_id.push('2'); variants.push(changed);
        let mut changed = lock.clone(); changed.declaration.owner_id.push('2'); variants.push(changed);
        let mut changed = lock.clone(); changed.declaration.input_lock_sha256 = "d".repeat(64); variants.push(changed);
        let mut changed = lock.clone(); changed.declaration.process_epoch += 1; variants.push(changed);
        let mut changed = lock.clone(); changed.declaration.engine_role.push('2'); variants.push(changed);
        let mut changed = lock.clone(); changed.declaration.engine_binary.sha256 = "d".repeat(64); variants.push(changed);
        let mut changed = lock.clone(); changed.declaration.engine_binary.bytes += 1; variants.push(changed);
        let mut changed = lock.clone(); changed.declaration.supervisor_binary.sha256 = "d".repeat(64); variants.push(changed);
        let mut changed = lock.clone(); changed.declaration.socket_basename.push('2'); variants.push(changed);
        let mut changed = lock.clone(); changed.declaration.reservation_root.push('2'); variants.push(changed);
        let mut changed = lock.clone(); changed.declaration.reservation_namespace.push('2'); variants.push(changed);
        let mut changed = lock.clone(); changed.declaration.device_ordinal += 1; variants.push(changed);
        let mut changed = lock.clone(); changed.declaration.driver_library_role.push('2'); variants.push(changed);
        let mut changed = lock.clone(); changed.declaration.max_frame_bytes -= 1; variants.push(changed);
        for changed in variants {
            assert!(changed.validate().is_err(), "tampered declaration must not verify against original digest");
        }
        let mut changed = lock.clone();
        changed.canonical_sha256 = "e".repeat(64);
        assert!(changed.validate().is_err());
    }

    #[test]
    fn declaration_rejects_unsafe_names_invalid_hashes_and_unbounded_frames() {
        for bad in ["", ".", "..", "../socket", "a/b", "a\\b", "name\n"] {
            let mut d = declaration();
            d.socket_basename = bad.into();
            assert!(d.validate().is_err(), "invalid socket basename: {bad:?}");
        }
        for bad in ["", "..", "role/name", "role name"] {
            let mut d = declaration();
            d.engine_role = bad.into();
            assert!(d.validate().is_err(), "invalid role: {bad:?}");
        }
        let mut d = declaration(); d.engine_role = "r".repeat(65); assert!(d.validate().is_err());
        let mut d = declaration(); d.authorization_id = "a".repeat(129); assert!(d.validate().is_err());
        let mut d = declaration(); d.socket_basename = "s".repeat(81); assert!(d.validate().is_err());
        let mut d = declaration(); d.input_lock_sha256 = "0".repeat(64); assert!(d.validate().is_err());
        let mut d = declaration(); d.engine_binary.path = "../engine".into(); assert!(d.validate().is_err());
        let mut d = declaration(); d.reservation_root = "../other".into(); assert!(d.validate().is_err());
        let mut d = declaration(); d.reservation_root = "run/.locks".into(); assert!(d.validate().is_ok());
        let mut d = declaration(); d.reservation_root = "run//other".into(); assert!(d.validate().is_err());
        let mut d = declaration(); d.reservation_root = "run/../other".into(); assert!(d.validate().is_err());
        let mut d = declaration(); d.reservation_namespace = "../other".into(); assert!(d.validate().is_err());
        let mut d = declaration(); d.process_epoch = 0; assert!(d.validate().is_err());
        let mut d = declaration(); d.device_ordinal = u32::MAX; assert!(d.validate().is_err());
        let mut d = declaration(); d.max_frame_bytes = 0; assert!(d.validate().is_err());
        let mut d = declaration(); d.max_frame_bytes = 4_097; assert!(d.validate().is_err());
        let mut d = declaration(); d.max_messages = 0; assert!(d.validate().is_err());
        let mut d = declaration(); d.max_messages = u32::MAX; assert!(d.validate().is_ok());
    }

    #[test]
    fn strict_bounded_parser_rejects_duplicate_unknown_and_oversized_json() {
        let lock = declaration().lock().unwrap();
        let wire = serde_json::to_string(&lock).unwrap();
        let duplicate = wire.replacen("\"authorization_id\":\"approval-1\"", "\"authorization_id\":\"approval-1\",\"authorization_id\":\"approval-1\"", 1);
        assert_ne!(duplicate, wire);
        assert!(RoventSupervisorLockV1::from_json(&duplicate).is_err());
        let unknown = wire.replacen("\"domain\":", "\"unexpected\":true,\"domain\":", 1);
        assert!(RoventSupervisorLockV1::from_json(&unknown).is_err());
        let oversized = format!("{}{}", wire, " ".repeat(ROVENT_SUPERVISOR_LOCK_V1_MAX_BYTES as usize));
        assert!(RoventSupervisorLockV1::from_json(&oversized).is_err());
    }

    #[test]
    fn canonical_identity_is_independent_of_object_field_order() {
        let original = declaration();
        let value = serde_json::to_value(&original).unwrap();
        let fields = value.as_object().unwrap();
        let reversed = fields.iter().rev().map(|(key, value)| {
            format!("{}:{}", serde_json::to_string(key).unwrap(), serde_json::to_string(value).unwrap())
        }).collect::<Vec<_>>().join(",");
        let decoded: RoventSupervisorDeclarationV1 = decode_json(&format!("{{{reversed}}}")).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(decoded.canonical_bytes().unwrap(), original.canonical_bytes().unwrap());
        assert_eq!(decoded.lock().unwrap().canonical_sha256, original.lock().unwrap().canonical_sha256);
    }
}
