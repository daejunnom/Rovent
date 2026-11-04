//! Metadata-only verification of an input-lock /2 and its root declarations.
//!
//! The returned value cannot issue a live root, authenticate a native worker,
//! establish a Rules replay or physical completion, or authorize a teacher
//! target or corpus publication. The supplied JSON is historical metadata;
//! callers must retain and independently verify any actual source files and
//! execution evidence before using it in a later, separately defined reader.
//! No /1 schema or digest is reconstructed here.

use crate::ArenaError;
use rz_contracts::Digest;
use rz_experiments::{RoventInputLockV2, RoventRootAllocationClockV2, RoventRootDescriptorV2};

/// A successful declaration join, not an execution or publication authority.
/// It is deliberately not serializable and has no public constructor.
#[allow(dead_code)]
pub(crate) struct VerifiedHistoricalDeclarationsV2 {
    lock: RoventInputLockV2,
    descriptor: RoventRootDescriptorV2,
    allocation: RoventRootAllocationClockV2,
    descriptor_digest: Digest,
    allocation_digest: Digest,
}

/// Revalidates the actual declared /2 formats and their joins. This does not
/// open files or assert that the supplied bytes were originally consumed.
#[allow(dead_code)]
pub(crate) fn verify_historical_declarations_v2(
    lock_json: &str,
    descriptor_json: &str,
    allocation_json: &str,
) -> Result<VerifiedHistoricalDeclarationsV2, ArenaError> {
    let invalid = |stage: &str, error: &dyn std::fmt::Display| {
        ArenaError::Invalid(format!("V2 historical declaration {stage}: {error}"))
    };
    // Each parser enforces its own byte bound, domain, duplicate/unknown-field
    // policy and canonical checks. Never fall back to a legacy parser.
    let lock =
        RoventInputLockV2::from_json(lock_json).map_err(|error| invalid("input lock", &error))?;
    let descriptor = RoventRootDescriptorV2::from_json(descriptor_json)
        .map_err(|error| invalid("root descriptor", &error))?;
    descriptor
        .verify_against(&lock)
        .map_err(|error| invalid("root/lock join", &error))?;
    let allocation = RoventRootAllocationClockV2::from_json(allocation_json)
        .map_err(|error| invalid("allocation clock", &error))?;
    allocation
        .verify_against(&descriptor, &lock)
        .map_err(|error| invalid("allocation/root/lock join", &error))?;
    let descriptor_digest = descriptor
        .digest()
        .map_err(|error| invalid("root digest", &error))?;
    let allocation_digest = allocation
        .digest()
        .map_err(|error| invalid("allocation digest", &error))?;
    Ok(VerifiedHistoricalDeclarationsV2 {
        lock,
        descriptor,
        allocation,
        descriptor_digest,
        allocation_digest,
    })
}
