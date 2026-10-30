//! Separate allocation-clock metadata for an unchanged root descriptor /1.
//!
//! The producer supplies the original allocation's start and output boundary
//! in its original monotonic domain. This codec never infers start from a
//! deadline, observes an actual clock, grants a live root capability, or turns
//! the output allocation boundary into an observed bestmove completion time.

use crate::{
    ManifestError, RoventInputLock, RoventInputLockV2, RoventRootAuthority, RoventRootDeadline,
    RoventRootDescriptor, RoventRootDescriptorInput, RoventRootDescriptorV2,
};
use rz_contracts::Digest;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;

pub const ROVENT_ROOT_ALLOCATION_CLOCK_DOMAIN: &str = "rz-rovent-root-allocation-clock/1";
pub const ROVENT_ROOT_ALLOCATION_CLOCK_V2_DOMAIN: &str = "rz-rovent-root-allocation-clock/2";
pub const MAX_ROOT_ALLOCATION_BYTES: usize = 4096;

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventRootAllocationInput {
    pub root_descriptor_sha256: String,
    pub startup_lock_sha256: String,
    pub authority: RoventRootAuthority,
    /// Actual original allocation start; never reconstructed from hard/soft.
    pub start: RoventRootDeadline,
    /// Original allocation's output boundary, not an observed completion tick.
    pub output: RoventRootDeadline,
    pub allocation_ns: u64,
    pub wall_time_max_ms: u64,
    pub cleanup_max_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventRootAllocationClock {
    pub domain: String,
    pub input: RoventRootAllocationInput,
}

/// Separate /2 wire, joined only to the exact typed /2 declarations.
/// No live clock, process observation, or root authority is produced.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventRootAllocationClockV2 {
    pub domain: String,
    pub startup_lock_domain: String,
    pub input: RoventRootAllocationInput,
}

impl RoventRootAllocationClockV2 {
    pub fn seal(
        input: RoventRootAllocationInput,
        descriptor: &RoventRootDescriptorV2,
        lock: &RoventInputLockV2,
    ) -> Result<Self, ManifestError> {
        let value = Self {
            domain: ROVENT_ROOT_ALLOCATION_CLOCK_V2_DOMAIN.into(),
            startup_lock_domain: crate::ROVENT_LOCK_V2_DOMAIN.into(),
            input,
        };
        value.verify_against(descriptor, lock)?;
        Ok(value)
    }

    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        require(
            input.len() <= MAX_ROOT_ALLOCATION_BYTES,
            "allocation-clock JSON exceeds 4 KiB",
        )?;
        let value: Self = crate::decode_json(input)?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.startup_lock_domain == crate::ROVENT_LOCK_V2_DOMAIN,
            "allocation-clock /2 lock domain differs",
        )?;
        validate_body(
            &self.input,
            &self.domain,
            ROVENT_ROOT_ALLOCATION_CLOCK_V2_DOMAIN,
        )?;
        require(
            canonical(self)?.len() <= MAX_ROOT_ALLOCATION_BYTES,
            "allocation-clock canonical JSON exceeds 4 KiB",
        )
    }

    pub fn verify_against(
        &self,
        descriptor: &RoventRootDescriptorV2,
        lock: &RoventInputLockV2,
    ) -> Result<(), ManifestError> {
        descriptor.verify_against(lock)?;
        self.validate()?;
        verify_join(
            &self.input,
            &descriptor.input,
            descriptor.digest()?,
            &lock.canonical_sha256,
            lock.manifest.wall_time_max_ms,
            lock.manifest.cleanup_max_ms,
        )
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        canonical(self)
    }

    pub fn digest(&self) -> Result<Digest, ManifestError> {
        let mut hash = Sha256::new();
        hash.update(ROVENT_ROOT_ALLOCATION_CLOCK_V2_DOMAIN.as_bytes());
        hash.update([0]);
        hash.update(self.canonical_bytes()?);
        Ok(Digest(hash.finalize().into()))
    }
}

impl RoventRootAllocationClock {
    /// Seals only metadata joined to the supplied exact original declarations.
    /// No raw fields of the existing descriptor are added, changed or rehashed
    /// into a replacement /1 descriptor.
    pub fn seal(
        input: RoventRootAllocationInput,
        descriptor: &RoventRootDescriptor,
        lock: &RoventInputLock,
    ) -> Result<Self, ManifestError> {
        let allocation = Self {
            domain: ROVENT_ROOT_ALLOCATION_CLOCK_DOMAIN.into(),
            input,
        };
        allocation.verify_against(descriptor, lock)?;
        Ok(allocation)
    }

    /// Checks the closed metadata body only. Consumers must separately call
    /// verify_against with the actual original descriptor and startup lock.
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        require(
            input.len() <= MAX_ROOT_ALLOCATION_BYTES,
            "allocation-clock JSON exceeds 4 KiB",
        )?;
        let allocation: Self = crate::decode_json(input)?;
        allocation.validate()?;
        Ok(allocation)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        validate_body(
            &self.input,
            &self.domain,
            ROVENT_ROOT_ALLOCATION_CLOCK_DOMAIN,
        )?;
        require(
            canonical(self)?.len() <= MAX_ROOT_ALLOCATION_BYTES,
            "allocation-clock canonical JSON exceeds 4 KiB",
        )
    }

    /// Matching persisted declarations are metadata evidence only. They do not
    /// issue authority, reconstruct a ProcessClock or attest runtime execution.
    pub fn verify_against(
        &self,
        descriptor: &RoventRootDescriptor,
        lock: &RoventInputLock,
    ) -> Result<(), ManifestError> {
        descriptor.verify_against(lock)?;
        self.validate()?;
        verify_join(
            &self.input,
            &descriptor.input,
            descriptor.digest()?,
            &lock.canonical_sha256,
            lock.manifest.wall_time_max_ms,
            lock.manifest.cleanup_max_ms,
        )
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        canonical(self)
    }

    pub fn digest(&self) -> Result<Digest, ManifestError> {
        let mut hash = Sha256::new();
        hash.update(ROVENT_ROOT_ALLOCATION_CLOCK_DOMAIN.as_bytes());
        hash.update([0]);
        hash.update(self.canonical_bytes()?);
        Ok(Digest(hash.finalize().into()))
    }
}

fn validate_body(
    input: &RoventRootAllocationInput,
    domain: &str,
    expected_domain: &str,
) -> Result<(), ManifestError> {
    require(
        domain == expected_domain
            && sha(&input.root_descriptor_sha256)
            && sha(&input.startup_lock_sha256)
            && sha(&input.authority.implementation_sha256)
            && input.authority.epoch != 0,
        "allocation-clock domain, source or authority declaration differs",
    )?;
    require(
        input.start.clock_epoch == input.authority.epoch
            && input.output.clock_epoch == input.authority.epoch
            && input.start.at_ns <= input.output.at_ns,
        "allocation start/output domains or order differ",
    )?;
    require(
        (1..=86_400_000).contains(&input.wall_time_max_ms)
            && (1..=30_000).contains(&input.cleanup_max_ms),
        "allocation-clock declared wall/cleanup ceilings are outside bounds",
    )?;
    let wall_ns = checked_ns(input.wall_time_max_ms, "wall ceiling overflows nanoseconds")?;
    checked_ns(
        input.cleanup_max_ms,
        "cleanup ceiling overflows nanoseconds",
    )?;
    require(
        input.output.at_ns.checked_sub(input.start.at_ns) == Some(input.allocation_ns)
            && input.allocation_ns <= wall_ns,
        "allocation differs from original output minus start or exceeds wall ceiling",
    )?;
    Ok(())
}

fn verify_join(
    input: &RoventRootAllocationInput,
    original: &RoventRootDescriptorInput,
    descriptor_digest: Digest,
    lock_digest: &str,
    wall_time_max_ms: u64,
    cleanup_max_ms: u64,
) -> Result<(), ManifestError> {
    require(
        input.root_descriptor_sha256 == hex(descriptor_digest)
            && input.startup_lock_sha256 == lock_digest
            && input.startup_lock_sha256 == original.startup_lock_sha256
            && input.authority == original.authority,
        "allocation-clock does not join its original descriptor, startup and authority",
    )?;
    require(
        input.wall_time_max_ms == wall_time_max_ms
            && input.wall_time_max_ms == original.declared_ceilings.wall_time_max_ms
            && input.cleanup_max_ms == cleanup_max_ms
            && input.cleanup_max_ms == original.declared_ceilings.cleanup_max_ms,
        "allocation-clock changes an original wall/cleanup ceiling",
    )?;
    let clock = &original.clock;
    require(
        [
            input.start,
            clock.soft,
            clock.admission,
            clock.hard,
            input.output,
            clock.cleanup,
        ]
        .into_iter()
        .all(|tick| tick.clock_epoch == input.authority.epoch)
            && input.start.at_ns <= clock.soft.at_ns
            && clock.soft.at_ns <= clock.admission.at_ns
            && clock.admission.at_ns <= clock.hard.at_ns
            && clock.hard.at_ns <= input.output.at_ns
            && input.output.at_ns <= clock.cleanup.at_ns,
        "allocation-clock original start/soft/admission/hard/output/cleanup order differs",
    )?;
    let cleanup_ns = checked_ns(
        input.cleanup_max_ms,
        "cleanup ceiling overflows nanoseconds",
    )?;
    require(
        clock
            .cleanup
            .at_ns
            .checked_sub(clock.hard.at_ns)
            .is_some_and(|duration| duration <= cleanup_ns),
        "allocation-clock cleanup exceeds its original declared ceiling",
    )
}

fn checked_ns(milliseconds: u64, message: &str) -> Result<u64, ManifestError> {
    milliseconds
        .checked_mul(1_000_000)
        .ok_or_else(|| integrity(message))
}

fn sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hex(digest: Digest) -> String {
    digest.0.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sorted(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| (key, sorted(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(sorted).collect()),
        other => other,
    }
}

fn canonical(value: &impl Serialize) -> Result<Vec<u8>, ManifestError> {
    let value = serde_json::to_value(value).map_err(|error| integrity(&error.to_string()))?;
    serde_json::to_vec(&sorted(value)).map_err(|error| integrity(&error.to_string()))
}

fn require(valid: bool, message: &str) -> Result<(), ManifestError> {
    if valid {
        Ok(())
    } else {
        Err(integrity(message))
    }
}

fn integrity(message: &str) -> ManifestError {
    ManifestError::Integrity(format!("Rovent root allocation clock: {message}"))
}

#[cfg(test)]
#[path = "rovent_root_allocation/tests.rs"]
mod tests;
