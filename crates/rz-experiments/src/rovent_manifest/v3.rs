//! Explicit system-runtime Rovent admission. No V1 or V2 persisted input is
//! converted, relabelled, or treated as evidence of this domain.
use super::*;
use crate::{NativeRuntimeSourceV6, SystemRuntimeLockV6};

pub const ROVENT_EXECUTION_V3_DOMAIN: &str = "rz-rovent-execution/3";
pub const ROVENT_LOCK_V3_DOMAIN: &str = "rz-rovent-input-lock/3";
pub const ROVENT_CANONICAL_V3_DOMAIN: &str = "rz-rovent-execution-canonical/3";
pub const ROVENT_STARTUP_DOCUMENT_V5_DOMAIN: &str = "rz-rovent-startup-document/5";
pub const ROVENT_STARTUP_CONFIGURATION_V5_DOMAIN: &str = "rz-rovent-startup-configuration/5";
pub const ROVENT_STARTUP_DOCUMENT_V5_MAX_BYTES: usize = 256 << 10;

/// The model, search and resources remain declared explicitly. System-native
/// libraries are registered by a separate V6 lock and are not input artifacts.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventExecutionManifestV3 {
    pub domain: String,
    pub contract_revision: String,
    pub run_id: String,
    pub purpose: RoventPurpose,
    pub engine: RoventEngineDeclaration,
    pub resources: RoventResourcePlan,
    pub coordinator: RoventCoordinatorDeclaration,
    pub wall_time_max_ms: u64,
    pub cleanup_max_ms: u64,
    pub output_bytes_max: u64,
    pub agent_renewable: bool,
    pub provenance: Vec<ArtifactRef>,
    pub startup_document: ArtifactRef,
    pub startup_configuration_sha256: String,
    pub startup_search_configuration_sha256: String,
    pub own_r_root_tie: RoventRootTieDeclarationV2,
    pub system_runtime_lock: SystemRuntimeLockV6,
}

impl RoventExecutionManifestV3 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let manifest: Self = decode_json(input)?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.domain == ROVENT_EXECUTION_V3_DOMAIN,
            "Rovent /3 execution domain mismatch",
        )?;
        self.system_runtime_lock.verify()?;
        let system_cuda = match (&self.system_runtime_lock.runtime, &self.engine.pc.device) {
            (NativeRuntimeSourceV6::Cpu, RoventDevice::Cpu)
                if self.engine.pc.backend != RoventBackend::OrtCuda =>
            {
                false
            }
            (NativeRuntimeSourceV6::SystemCuda(spec), RoventDevice::Cuda { ordinal, .. })
                if self.engine.pc.backend == RoventBackend::OrtCuda
                    && u32::try_from(spec.cuda_device_id).ok() == Some(*ordinal) =>
            {
                true
            }
            _ => {
                return Err(ManifestError::Integrity(
                    "explicit runtime source conflicts with actual backend or device".into(),
                ));
            }
        };
        // This private projection reuses the shared declaration checks only.
        // It cannot be serialized, published or admitted as a /2 lock.
        let common = RoventExecutionManifestV2 {
            domain: ROVENT_EXECUTION_V2_DOMAIN.into(),
            contract_revision: self.contract_revision.clone(),
            run_id: self.run_id.clone(),
            purpose: self.purpose,
            engine: self.engine.clone(),
            resources: self.resources.clone(),
            coordinator: self.coordinator.clone(),
            wall_time_max_ms: self.wall_time_max_ms,
            cleanup_max_ms: self.cleanup_max_ms,
            output_bytes_max: self.output_bytes_max,
            agent_renewable: self.agent_renewable,
            provenance: self.provenance.clone(),
            startup_document: self.startup_document.clone(),
            startup_configuration_sha256: self.startup_configuration_sha256.clone(),
            startup_search_configuration_sha256: self.startup_search_configuration_sha256.clone(),
            own_r_root_tie: self.own_r_root_tie.clone(),
        };
        common.validate_with_system_cuda(system_cuda)?;
        if let NativeRuntimeSourceV6::SystemCuda(spec) = &self.system_runtime_lock.runtime {
            require(
                spec.startup_ms <= self.wall_time_max_ms && spec.drain_ms <= self.cleanup_max_ms,
                "system CUDA startup and drain exceed the registered run window",
            )?;
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        canonical(&(ROVENT_CANONICAL_V3_DOMAIN, self))
    }

    pub fn lock(self) -> Result<RoventInputLockV3, ManifestError> {
        let canonical_sha256 = digest(&self.canonical_bytes()?);
        Ok(RoventInputLockV3 {
            domain: ROVENT_LOCK_V3_DOMAIN.into(),
            manifest: self,
            canonical_sha256,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventInputLockV3 {
    pub domain: String,
    pub manifest: RoventExecutionManifestV3,
    pub canonical_sha256: String,
}

impl RoventInputLockV3 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let lock: Self = decode_json(input)?;
        lock.validate()?;
        Ok(lock)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.domain == ROVENT_LOCK_V3_DOMAIN
                && sha(&self.canonical_sha256)
                && digest(&self.manifest.canonical_bytes()?) == self.canonical_sha256,
            "Rovent /3 lock domain/canonical digest mismatch",
        )
    }
}

/// Versioned startup envelope. The engine owner must additionally deserialize
/// and validate configuration as its own typed /5 configuration and compare
/// the actual configuration/search digests against the manifest.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventStartupDocumentV5 {
    pub domain: String,
    pub system_runtime_lock_sha256: String,
    pub configuration: serde_json::Value,
}

impl RoventStartupDocumentV5 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        require(
            !input.is_empty() && input.len() <= ROVENT_STARTUP_DOCUMENT_V5_MAX_BYTES,
            "startup document /5 exceeds the byte budget",
        )?;
        let document: Self = decode_json(input)?;
        document.validate_structure()?;
        Ok(document)
    }

    fn validate_structure(&self) -> Result<(), ManifestError> {
        require(
            self.domain == ROVENT_STARTUP_DOCUMENT_V5_DOMAIN
                && sha(&self.system_runtime_lock_sha256)
                && self
                    .configuration
                    .as_object()
                    .and_then(|object| object.get("domain"))
                    .and_then(serde_json::Value::as_str)
                    == Some(ROVENT_STARTUP_CONFIGURATION_V5_DOMAIN),
            "startup document /5 domain, runtime identity or configuration domain differs",
        )
    }

    /// Verify the original bytes and envelope identity, not configuration
    /// semantics. Original formatting remains part of the ArtifactRef pin.
    pub fn validate_against(
        &self,
        lock: &RoventInputLockV3,
        original: &[u8],
    ) -> Result<(), ManifestError> {
        lock.validate()?;
        self.validate_structure()?;
        let text = std::str::from_utf8(original)
            .map_err(|_| ManifestError::Integrity("startup /5 is not UTF-8".into()))?;
        require(
            Self::from_json(text)? == *self
                && original.len() as u64 == lock.manifest.startup_document.bytes
                && digest(original) == lock.manifest.startup_document.sha256
                && self.system_runtime_lock_sha256
                    == lock.manifest.system_runtime_lock.canonical_sha256,
            "startup document /5 bytes or system lock identity differ",
        )
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate_structure()?;
        canonical(self)
    }
}
