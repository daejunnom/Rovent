//! Explicit system CUDA startup declaration for PALS.
//!
//! This is a separate domain from the historical V5 bundle startup. Parsing or
//! locking this declaration does not open a library, verify model bytes, start
//! a provider, issue a deadline, or prove GPU completion. Those remain the
//! responsibility of the native owner and its supervising process.
use crate::{
    ArtifactRef, ManifestError, NativeRuntimeSourceV6, PalsArtifactDomainV5, PalsComponentV3,
    PalsCoordinatorV5, PalsCpuEndpointsV5, PalsEngineModeV5, PalsExecutionWindowV5,
    PalsModelBackendV3, PalsModelSpecificationV5, PalsPrecisionV3, PalsResourcePlanV5,
    PalsReviewContextV5, PalsStartupAssetPurposeV5, PalsStartupAssetV5, PalsViewSelectionV5,
    SystemRuntimeLockV6, decode_json, pals_private_snapshot_path_v5,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const PALS_SYSTEM_STARTUP_CONFIG_V6_DOMAIN: &str = "rz-pals-system-startup-config-v6/1";
pub const PALS_SYSTEM_STARTUP_LOCK_V6_DOMAIN: &str = "rz-pals-system-startup-lock-v6/1";

fn invalid(message: &str) -> ManifestError {
    ManifestError::Integrity(message.into())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_text(value: &str, maximum: usize) -> bool {
    !value.is_empty() && value.len() <= maximum && !value.chars().any(char::is_control)
}

fn canonical<T: Serialize>(value: &T) -> Result<Vec<u8>, ManifestError> {
    let mut value = serde_json::to_value(value).map_err(|e| invalid(&e.to_string()))?;
    value.sort_all_objects();
    serde_json::to_vec(&value).map_err(|e| invalid(&e.to_string()))
}

/// Strict CUDA is the only declared policy until a separately authenticated
/// metadata-only placement policy is implemented and observed end-to-end.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PalsSystemCpuPlacementV6 {
    StrictCudaOnly,
}

/// Strict requires a declared source commit; Observe retains absence as
/// unknown. Neither declaration alone verifies the actual running binary.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PalsSystemSourcePolicyV6 {
    Strict,
    Observe,
}

fn explicit_optional<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

/// V6's direct endpoint preserves V5's chess and model fields while allowing
/// an explicitly unknown source commit in the observation-only path.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsSystemEndpointV6 {
    pub id: String,
    pub binary: ArtifactRef,
    #[serde(deserialize_with = "explicit_optional")]
    pub source_commit: Option<String>,
    pub mode: PalsEngineModeV5,
    pub model: PalsModelSpecificationV5,
    pub cpu: PalsCpuEndpointsV5,
    pub context: PalsReviewContextV5,
    pub coordinator: PalsCoordinatorV5,
    pub resources: PalsResourcePlanV5,
    pub search: PalsComponentV3,
    pub runtime: PalsComponentV3,
    pub v_generate: bool,
    pub v_load: bool,
    pub v_call: bool,
}

impl PalsSystemEndpointV6 {
    pub fn validate(&self, source_policy: PalsSystemSourcePolicyV6) -> Result<(), ManifestError> {
        self.binary.validate()?;
        if !is_text(&self.id, 256)
            || self.source_commit.as_ref().is_some_and(|commit| {
                commit.len() != 40
                    || !commit
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            })
            || (source_policy == PalsSystemSourcePolicyV6::Strict && self.source_commit.is_none())
        {
            return Err(invalid(
                "invalid or missing required endpoint source identity",
            ));
        }
        self.model.validate(self.mode)?;
        self.cpu.validate(self.mode)?;
        self.context.validate()?;
        if !match self.context.selection {
            PalsViewSelectionV5::VirtualReview => self.model.profile == "virtual_review_v1",
            PalsViewSelectionV5::V2Full | PalsViewSelectionV5::V2Selected => {
                self.model.profile == "full_line_interaction_v2" && self.context.anchors_max == 1
            }
        } {
            return Err(invalid(
                "view selection differs from declared model semantics",
            ));
        }
        self.coordinator.validate()?;
        self.resources.validate(&self.coordinator.slices)?;
        crate::pals_manifest::component(&self.search)?;
        crate::pals_manifest::component(&self.runtime)?;
        let roles_match = match self.mode {
            PalsEngineModeV5::Collection => self.v_generate && self.v_load && self.v_call,
            PalsEngineModeV5::Deployment => !self.v_generate && !self.v_load && !self.v_call,
        };
        if !roles_match
            || (self.model.backend == PalsModelBackendV3::OrtCuda && self.resources.gpu_count != 1)
        {
            return Err(invalid("mode roles or GPU resources differ from endpoint"));
        }
        Ok(())
    }
}

/// Native session selections, excluding paths and identity owned by the system
/// runtime lock. The declared window does not extend an issued root deadline.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsSystemNativeOptionsV6 {
    pub cache_public_memory: bool,
    pub device_public_memory: bool,
    pub cudnn_conv_search: String,
    pub approximate_warm: bool,
    pub cpu_placement: PalsSystemCpuPlacementV6,
    pub startup_ms: u64,
    pub drain_ms: u64,
}

impl PalsSystemNativeOptionsV6 {
    /// The endpoint records the actual execution configuration, never the
    /// historical eager-bundle loading profile.
    pub fn canonical_runtime_options(
        &self,
        runtime: &crate::SystemRuntimeSpecV6,
    ) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("runtime_source".into(), "system_cuda_v6".into()),
            ("cuda_device_id".into(), runtime.cuda_device_id.to_string()),
            (
                "cuda_arena_bytes".into(),
                runtime.cuda_arena_bytes.to_string(),
            ),
            (
                "cache_public_memory".into(),
                self.cache_public_memory.to_string(),
            ),
            (
                "device_public_memory".into(),
                self.device_public_memory.to_string(),
            ),
            ("cudnn_conv_search".into(), self.cudnn_conv_search.clone()),
            ("approximate_warm".into(), self.approximate_warm.to_string()),
            ("cpu_placement".into(), "strict_cuda_only".into()),
            ("startup_ms".into(), self.startup_ms.to_string()),
            ("drain_ms".into(), self.drain_ms.to_string()),
        ])
    }
}

/// A new startup declaration, independently identified without requiring or
/// converting an old PALS input/startup lock or CUDA bundle manifest.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsSystemStartupConfigV6 {
    pub domain: String,
    pub run_id: String,
    pub endpoint: PalsSystemEndpointV6,
    pub window: PalsExecutionWindowV5,
    pub source_components: BTreeMap<String, String>,
    pub source_policy: PalsSystemSourcePolicyV6,
    pub assets: Vec<PalsStartupAssetV5>,
    pub system_runtime_lock: SystemRuntimeLockV6,
    pub native_options: PalsSystemNativeOptionsV6,
    /// A declared child identity only; the owner must bind the real PID.
    pub process_generation: u64,
    pub runtime_directory: String,
}

impl PalsSystemStartupConfigV6 {
    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let value: Self = decode_json(input)?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.domain != PALS_SYSTEM_STARTUP_CONFIG_V6_DOMAIN
            || !is_text(&self.run_id, 256)
            || self.process_generation == 0
            || !matches!(
                self.runtime_directory.as_str(),
                "baseline-runtime" | "candidate-runtime"
            )
        {
            return Err(invalid("invalid PALS system startup identity"));
        }
        self.endpoint.validate(self.source_policy)?;
        if self.endpoint.model.backend != PalsModelBackendV3::OrtCuda
            || self.endpoint.model.precision != PalsPrecisionV3::Fp32
            || self.endpoint.model.tf32
        {
            return Err(invalid(
                "system CUDA startup requires declared FP32 CUDA and TF32 off",
            ));
        }
        if !is_text(&self.window.window_id, 256)
            || !(1..=86_400_000).contains(&self.window.wall_time_max_ms)
            || !(1..=30_000).contains(&self.window.cleanup_max_ms)
            || !(1..=64u64 << 30).contains(&self.window.output_bytes_max)
            || !self.window.agent_renewable
        {
            return Err(invalid("invalid finite PALS system execution window"));
        }
        if (self.source_policy == PalsSystemSourcePolicyV6::Strict
            && self.source_components.is_empty())
            || self.source_components.len() > 64
            || self
                .source_components
                .iter()
                .any(|(name, digest)| !is_text(name, 128) || !is_sha256(digest))
        {
            return Err(invalid("invalid PALS source component declarations"));
        }
        self.system_runtime_lock.verify()?;
        let NativeRuntimeSourceV6::SystemCuda(runtime) = &self.system_runtime_lock.runtime else {
            return Err(invalid(
                "PALS system CUDA startup requires an explicit system CUDA lock",
            ));
        };
        let options = &self.native_options;
        if options.startup_ms == 0
            || options.startup_ms > self.window.wall_time_max_ms
            || options.drain_ms == 0
            || options.drain_ms > self.window.cleanup_max_ms
            || options.startup_ms != runtime.startup_ms
            || options.drain_ms != runtime.drain_ms
            || options.device_public_memory
            || (options.approximate_warm && self.endpoint.mode == PalsEngineModeV5::Collection)
            || !matches!(
                options.cudnn_conv_search.as_str(),
                "legacy_implicit_exhaustive" | "experimental_default"
            )
            || self.endpoint.runtime.options != options.canonical_runtime_options(runtime)
        {
            return Err(invalid(
                "PALS system native options disagree with endpoint, lock or window",
            ));
        }
        self.validate_assets(runtime)
    }

    fn validate_assets(&self, runtime: &crate::SystemRuntimeSpecV6) -> Result<(), ManifestError> {
        use PalsStartupAssetPurposeV5 as P;
        if self.assets.is_empty() || self.assets.len() > 96 {
            return Err(invalid("invalid PALS system startup asset count"));
        }
        let mut names = BTreeSet::new();
        let mut sources = BTreeSet::new();
        for asset in &self.assets {
            asset.artifact.validate()?;
            if !pals_private_snapshot_path_v5(&asset.snapshot_path)
                || !names.insert(&asset.snapshot_path)
                || !sources.insert(&asset.artifact.path)
                || matches!(
                    asset.purpose,
                    P::RuntimeLibrary | P::CudaLibrary | P::CudaManifest
                )
                || runtime.libraries.iter().any(|library| {
                    library.sha256 == asset.artifact.sha256 && library.bytes == asset.artifact.bytes
                })
            {
                return Err(invalid(
                    "unsafe, duplicate or system-library snapshot asset",
                ));
            }
        }
        let count = |purpose| {
            self.assets
                .iter()
                .filter(|asset| asset.purpose == purpose)
                .count()
        };
        for purpose in [P::Weights, P::ExportManifest, P::RulesDescriptor] {
            if count(purpose) != 1 {
                return Err(invalid("missing or ambiguous model startup asset"));
            }
        }
        if count(P::NumericalAcceptance) > 1 || count(P::CudaControlInventory) != 0 {
            return Err(invalid(
                "ambiguous evidence or unsupported CPU placement inventory",
            ));
        }
        match self.endpoint.model.artifact_domain {
            // A diagnostic asset can be inspected, but this declaration cannot
            // promote it to a model or arena candidate.
            PalsArtifactDomainV5::Diagnostic => {}
            PalsArtifactDomainV5::Untrained
                if count(P::InitializationEvidence) == 1
                    && count(P::InitializationCheckpointMetadata) == 1
                    && count(P::GenerationSpecification) == 0 => {}
            PalsArtifactDomainV5::RealGeneration | PalsArtifactDomainV5::FrozenPcCollection
                if count(P::GenerationSpecification) == 1
                    && count(P::InitializationEvidence) == 0
                    && count(P::InitializationCheckpointMetadata) == 0 => {}
            _ => {
                return Err(invalid(
                    "model initialization or generation provenance differs",
                ));
            }
        }
        let shared = count(P::PublicGraph) == 1
            && count(P::SharedPcGraph) == 1
            && count(P::ProposerGraph) == 0
            && count(P::CriticGraph) == 0;
        let separate = count(P::PublicGraph) == 1
            && count(P::SharedPcGraph) == 0
            && count(P::ProposerGraph) == 1
            && count(P::CriticGraph) == 1;
        if !shared && (self.endpoint.model.profile == "virtual_review_v1" || !separate)
            || count(P::ValidatorGraph)
                != usize::from(self.endpoint.mode == PalsEngineModeV5::Collection)
        {
            return Err(invalid("model graph roles differ from declared mode"));
        }
        if !self
            .assets
            .iter()
            .any(|a| a.purpose == P::Weights && a.artifact == self.endpoint.model.weights)
        {
            return Err(invalid("weights differ from declared endpoint"));
        }
        match &self.endpoint.model.provenance {
            Some(provenance)
                if count(P::Provenance) == 1
                    && self
                        .assets
                        .iter()
                        .any(|a| a.purpose == P::Provenance && &a.artifact == provenance) => {}
            None if count(P::Provenance) == 0 => {}
            _ => return Err(invalid("model provenance differs from endpoint")),
        }
        match &self.endpoint.cpu.external {
            Some(external)
                if count(P::ExternalBinary) == 1
                    && count(P::ExternalProfile) == 1
                    && self.assets.iter().any(|a| {
                        a.purpose == P::ExternalBinary && a.artifact == external.binary
                    })
                    && self.assets.iter().any(|a| {
                        a.purpose == P::ExternalProfile && a.artifact == external.profile
                    }) => {}
            None if count(P::ExternalBinary) == 0 && count(P::ExternalProfile) == 0 => {}
            _ => return Err(invalid("external CPU assets differ from endpoint")),
        }
        Ok(())
    }

    pub fn asset(
        &self,
        purpose: PalsStartupAssetPurposeV5,
    ) -> Result<&PalsStartupAssetV5, ManifestError> {
        let mut found = self.assets.iter().filter(|asset| asset.purpose == purpose);
        let first = found
            .next()
            .ok_or_else(|| invalid("startup asset purpose missing"))?;
        if found.next().is_some() {
            return Err(invalid("startup asset purpose ambiguous"));
        }
        Ok(first)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        canonical(self)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PalsSystemStartupLockV6 {
    pub domain: String,
    pub config: PalsSystemStartupConfigV6,
    pub canonical_sha256: String,
}

impl PalsSystemStartupLockV6 {
    pub fn new(config: PalsSystemStartupConfigV6) -> Result<Self, ManifestError> {
        config.validate()?;
        let bytes = canonical(&(PALS_SYSTEM_STARTUP_LOCK_V6_DOMAIN, &config))?;
        Ok(Self {
            domain: PALS_SYSTEM_STARTUP_LOCK_V6_DOMAIN.into(),
            config,
            canonical_sha256: format!("{:x}", Sha256::digest(bytes)),
        })
    }

    pub fn verify(&self) -> Result<(), ManifestError> {
        if self.domain != PALS_SYSTEM_STARTUP_LOCK_V6_DOMAIN
            || self.canonical_sha256 != Self::new(self.config.clone())?.canonical_sha256
        {
            return Err(invalid("PALS system startup lock identity differs"));
        }
        Ok(())
    }

    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        let value: Self = decode_json(input)?;
        value.verify()?;
        Ok(value)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.verify()?;
        canonical(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        PALS_V5_DIAGNOSTIC_CHECKPOINT_DOMAIN, PALS_V5_ENCODING_SCHEMA, PALS_V5_LINE_BUCKETS,
        PALS_V5_MODEL_SCHEMA, PALS_V5_MODEL_SEMANTICS, PalsCpuSelectionV5, PalsRoleV3,
        SystemLibrarySpecV6, SystemPlatformV6, SystemRuntimeSpecV6,
    };

    // All identities below describe in-memory declarations only. No files are
    // opened and no provider, execution or model acceptance is asserted.
    fn artifact(path: &str) -> ArtifactRef {
        ArtifactRef {
            path: path.into(),
            sha256: "a".repeat(64),
            bytes: 10,
            source: "https://example.org/test-only".into(),
            license: "MIT".into(),
        }
    }

    fn component(name: &str) -> PalsComponentV3 {
        PalsComponentV3 {
            semantic_id: name.into(),
            implementation_sha256: "b".repeat(64),
            options: BTreeMap::new(),
        }
    }

    fn fixture() -> PalsSystemStartupConfigV6 {
        let runtime = SystemRuntimeSpecV6 {
            platform: SystemPlatformV6::Linux,
            libraries: ["ort-core", "ort-provider-cuda"]
                .into_iter()
                .enumerate()
                .map(|(i, role)| SystemLibrarySpecV6 {
                    role: role.into(),
                    requested_path: format!("/test/lib{i}.so"),
                    canonical_path: format!("/test/lib{i}.so"),
                    version: None,
                    sha256: if i == 0 {
                        "1".repeat(64)
                    } else {
                        "2".repeat(64)
                    },
                    bytes: 100,
                    device: 1,
                    inode: i as u64 + 1,
                    required_mapping: true,
                })
                .collect(),
            system_identity_sha256: "3".repeat(64),
            cuda_device_id: 0,
            cuda_arena_bytes: 1024,
            startup_ms: 1000,
            drain_ms: 200,
            capture_max_libraries: 2,
            capture_max_file_bytes: 200,
            capture_max_total_bytes: 400,
            capture_timeout_ms: 100,
        };
        let options = PalsSystemNativeOptionsV6 {
            cache_public_memory: false,
            device_public_memory: false,
            cudnn_conv_search: "legacy_implicit_exhaustive".into(),
            approximate_warm: false,
            cpu_placement: PalsSystemCpuPlacementV6::StrictCudaOnly,
            startup_ms: runtime.startup_ms,
            drain_ms: runtime.drain_ms,
        };
        let mut runtime_component = component("system-runtime/test");
        runtime_component.options = options.canonical_runtime_options(&runtime);
        let mut resources = PalsResourcePlanV5::default();
        resources.gpu_count = 1;
        resources.gpu_model = Some("declared-test-gpu".into());
        let model = PalsModelSpecificationV5 {
            profile: "virtual_review_v1".into(),
            model_schema: PALS_V5_MODEL_SCHEMA.into(),
            model_semantics: PALS_V5_MODEL_SEMANTICS.into(),
            encoding_schema: PALS_V5_ENCODING_SCHEMA.into(),
            semantic_sha256: "c".repeat(64),
            encoding_sha256: "d".repeat(64),
            implementation_sha256: "e".repeat(64),
            weights: artifact("test/weights"),
            artifact_domain: PalsArtifactDomainV5::Untrained,
            provenance: None,
            checkpoint_domain: None,
            epoch: 1,
            backend: PalsModelBackendV3::OrtCuda,
            precision: PalsPrecisionV3::Fp32,
            tf32: false,
            exported_roles: BTreeSet::from([PalsRoleV3::Proposer, PalsRoleV3::Critic]),
            width: 384,
            latent_count: 16,
            temporal_width: 64,
            candidate_head_width: 128,
            record_feature_axes: 8,
            query_feature_axes: 8,
            max_records: 64,
            max_anchors: 4,
            line_buckets: PALS_V5_LINE_BUCKETS.to_vec(),
            one_dynamic_line_graph: true,
            ids_as_numeric_features: false,
            fresh_warm_separate_domains: true,
        };
        let endpoint = PalsSystemEndpointV6 {
            id: "test-engine".into(),
            binary: artifact("test/engine"),
            source_commit: None,
            mode: PalsEngineModeV5::Deployment,
            model,
            cpu: PalsCpuEndpointsV5 {
                cpu_t: PalsCpuSelectionV5::Disabled,
                cpu_r: PalsCpuSelectionV5::Own,
                own: component("own/test"),
                external: None,
            },
            context: PalsReviewContextV5::default(),
            coordinator: PalsCoordinatorV5::default(),
            resources,
            search: component("search/test"),
            runtime: runtime_component,
            v_generate: false,
            v_load: false,
            v_call: false,
        };
        use PalsStartupAssetPurposeV5 as P;
        let assets = [
            (P::Weights, "weights"),
            (P::ExportManifest, "export"),
            (P::RulesDescriptor, "rules"),
            (P::PublicGraph, "public"),
            (P::SharedPcGraph, "shared"),
            (P::InitializationEvidence, "initialization"),
            (P::InitializationCheckpointMetadata, "checkpoint"),
        ]
        .into_iter()
        .map(|(purpose, name)| PalsStartupAssetV5 {
            artifact: artifact(&format!("test/{name}")),
            snapshot_path: format!("inputs/test/{name}"),
            purpose,
        })
        .collect();
        PalsSystemStartupConfigV6 {
            domain: PALS_SYSTEM_STARTUP_CONFIG_V6_DOMAIN.into(),
            run_id: "test-system-startup".into(),
            endpoint,
            window: PalsExecutionWindowV5 {
                window_id: "test-window".into(),
                wall_time_max_ms: 5000,
                cleanup_max_ms: 1000,
                output_bytes_max: 1024,
                agent_renewable: true,
            },
            source_components: BTreeMap::new(),
            source_policy: PalsSystemSourcePolicyV6::Observe,
            assets,
            system_runtime_lock: SystemRuntimeLockV6::new(NativeRuntimeSourceV6::SystemCuda(
                runtime,
            ))
            .unwrap(),
            native_options: options,
            process_generation: 1,
            runtime_directory: "baseline-runtime".into(),
        }
    }

    #[test]
    fn observation_retains_null_source_and_strict_refuses_it() {
        let config = fixture();
        let lock = PalsSystemStartupLockV6::new(config).unwrap();
        let encoded = lock.canonical_bytes().unwrap();
        let decoded =
            PalsSystemStartupLockV6::from_json(std::str::from_utf8(&encoded).unwrap()).unwrap();
        assert_eq!(decoded.config.endpoint.source_commit, None);
        let mut strict = decoded.config;
        strict.source_policy = PalsSystemSourcePolicyV6::Strict;
        assert!(strict.validate().is_err());
        strict.endpoint.source_commit = Some("f".repeat(40));
        strict
            .source_components
            .insert("test".into(), "f".repeat(64));
        assert!(strict.validate().is_ok());
    }

    #[test]
    fn system_startup_rejects_cuda_bundle_snapshot_assets() {
        let mut config = fixture();
        config.validate().unwrap();
        config.assets.push(PalsStartupAssetV5 {
            artifact: artifact("test/cuda-bundle"),
            snapshot_path: "inputs/test/cuda-bundle".into(),
            purpose: PalsStartupAssetPurposeV5::CudaManifest,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn changed_system_lock_cannot_be_used_as_a_valid_declaration() {
        let mut config = fixture();
        config.validate().unwrap();
        if let NativeRuntimeSourceV6::SystemCuda(runtime) = &mut config.system_runtime_lock.runtime
        {
            runtime.cuda_device_id += 1;
        } else {
            panic!("fixture requires the system CUDA declaration");
        }
        assert!(config.validate().is_err());
    }

    #[test]
    fn diagnostic_can_be_declared_without_claiming_model_acceptance() {
        let mut config = fixture();
        config.endpoint.model.artifact_domain = PalsArtifactDomainV5::Diagnostic;
        let provenance = artifact("test/diagnostic-provenance");
        config.endpoint.model.provenance = Some(provenance.clone());
        config.endpoint.model.checkpoint_domain = Some(PALS_V5_DIAGNOSTIC_CHECKPOINT_DOMAIN.into());
        config.assets.push(PalsStartupAssetV5 {
            artifact: provenance,
            snapshot_path: "inputs/test/diagnostic-provenance".into(),
            purpose: PalsStartupAssetPurposeV5::Provenance,
        });
        assert!(
            config
                .assets
                .iter()
                .all(|a| a.purpose != PalsStartupAssetPurposeV5::NumericalAcceptance)
        );
        assert!(PalsSystemStartupLockV6::new(config).is_ok());
        // This result is only declaration validity; it is not an arena,
        // numerical, provider or physical-completion receipt.
    }
}
