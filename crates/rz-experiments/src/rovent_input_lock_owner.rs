//! An original, versioned Rovent lock owner and a read-only shared field view.
//!
//! This is a transport primitive only: validating a declaration grants no
//! process, storage, root, worker, model or physical-completion authority.
//! Neither version is converted into the other. Existing consumers keep their
//! typed APIs until they explicitly support this owner.

use std::sync::Arc;

use crate::{
    ArtifactRef, ManifestError, RoventCoordinatorDeclaration, RoventEngineDeclaration,
    RoventInputLock, RoventInputLockV2, RoventPurpose, RoventResourcePlan,
};

#[derive(Clone, Debug)]
enum OriginalLock {
    V1(Arc<RoventInputLock>),
    V2(Arc<RoventInputLockV2>),
}

/// Retains the actual parsed lock, including its version and canonical digest.
/// Constructing or cloning this wrapper is not an execution observation.
#[derive(Clone, Debug)]
pub struct RoventOriginalInputLock {
    inner: OriginalLock,
}

/// A failed constructor returns the original owner, preserving caller custody.
#[derive(Debug)]
pub struct RoventOriginalInputLockError<T> {
    pub error: ManifestError,
    pub original: Arc<T>,
}

/// Borrowed common declaration fields. They are references into the original
/// lock, not a synthesized manifest or canonical representation. `/2`-specific
/// policy and startup fields remain available only through `v2()`.
#[derive(Clone, Copy, Debug)]
pub struct RoventInputLockCommon<'a> {
    pub lock_domain: &'a str,
    pub canonical_sha256: &'a str,
    pub contract_revision: &'a str,
    pub run_id: &'a str,
    pub purpose: RoventPurpose,
    pub engine: &'a RoventEngineDeclaration,
    pub resources: &'a RoventResourcePlan,
    pub coordinator: &'a RoventCoordinatorDeclaration,
    pub wall_time_max_ms: u64,
    pub cleanup_max_ms: u64,
    pub output_bytes_max: u64,
    pub agent_renewable: bool,
    pub provenance: &'a [ArtifactRef],
}

impl RoventOriginalInputLock {
    pub fn from_v1(
        original: Arc<RoventInputLock>,
    ) -> Result<Self, RoventOriginalInputLockError<RoventInputLock>> {
        if let Err(error) = original.validate() {
            return Err(RoventOriginalInputLockError { error, original });
        }
        Ok(Self {
            inner: OriginalLock::V1(original),
        })
    }

    pub fn from_v2(
        original: Arc<RoventInputLockV2>,
    ) -> Result<Self, RoventOriginalInputLockError<RoventInputLockV2>> {
        if let Err(error) = original.validate() {
            return Err(RoventOriginalInputLockError { error, original });
        }
        Ok(Self {
            inner: OriginalLock::V2(original),
        })
    }

    /// Recheck the exact typed original; this does not observe external files.
    pub fn validate(&self) -> Result<(), ManifestError> {
        match &self.inner {
            OriginalLock::V1(lock) => lock.validate(),
            OriginalLock::V2(lock) => lock.validate(),
        }
    }

    pub fn v1(&self) -> Option<&Arc<RoventInputLock>> {
        match &self.inner {
            OriginalLock::V1(lock) => Some(lock),
            OriginalLock::V2(_) => None,
        }
    }

    pub fn v2(&self) -> Option<&Arc<RoventInputLockV2>> {
        match &self.inner {
            OriginalLock::V2(lock) => Some(lock),
            OriginalLock::V1(_) => None,
        }
    }

    /// Same actual Arc allocation, not merely equal bytes, hashes or fields.
    pub fn same_original_owner(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (OriginalLock::V1(a), OriginalLock::V1(b)) => Arc::ptr_eq(a, b),
            (OriginalLock::V2(a), OriginalLock::V2(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    pub fn common(&self) -> RoventInputLockCommon<'_> {
        macro_rules! fields {
            ($lock:expr) => {{
                let lock = $lock;
                let manifest = &lock.manifest;
                RoventInputLockCommon {
                    lock_domain: &lock.domain,
                    canonical_sha256: &lock.canonical_sha256,
                    contract_revision: &manifest.contract_revision,
                    run_id: &manifest.run_id,
                    purpose: manifest.purpose,
                    engine: &manifest.engine,
                    resources: &manifest.resources,
                    coordinator: &manifest.coordinator,
                    wall_time_max_ms: manifest.wall_time_max_ms,
                    cleanup_max_ms: manifest.cleanup_max_ms,
                    output_bytes_max: manifest.output_bytes_max,
                    agent_renewable: manifest.agent_renewable,
                    provenance: &manifest.provenance,
                }
            }};
        }
        match &self.inner {
            OriginalLock::V1(lock) => fields!(lock),
            OriginalLock::V2(lock) => fields!(lock),
        }
    }
}
