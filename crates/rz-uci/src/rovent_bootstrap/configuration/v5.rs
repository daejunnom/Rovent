//! Typed /5 startup configuration inside the original system-runtime envelope.
//! Neither this declaration nor a validated lock starts a native owner.

use super::v2::PostRepairPolicyV2;
use super::v3::{ExternalConfigurationV3, joined_external_question_limit};
use super::v4::{OwnRRootTieV4, ROVENT_OWN_NNUEC_CONDITIONS_V4_DOMAIN};
use super::{
    ConfigurationFields, ExecutionConfiguration, FrontierConfiguration, IndexConfiguration,
    NativeConfiguration, ResearchValueConfiguration, ReviewConfiguration, SemanticConfiguration,
    canonical, framed_digest, invalid,
};
use rz_eval::error::BackendError;
use rz_experiments::{
    NativeRuntimeSourceV6, ROVENT_STARTUP_CONFIGURATION_V5_DOMAIN, RoventInputLockV3,
    RoventRootTieDeclarationV2, RoventRootTiePolicyV2, RoventStartupDocumentV5,
};
use rz_search::cpu::CpuRootTiePolicy;
use rz_search::external_cpu::ExternalGoConstraint;
use rz_search::pals::engine::PostRepairRecheckPolicy;
use serde::{Deserialize, Serialize};

pub const ROVENT_SEARCH_CONFIGURATION_V5_DOMAIN: &str = "rz-rovent-search-configuration/5";

/// The typed native/search payload. The envelope and original bytes are held
/// independently to bind admission to the original input capability.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventStartupConfigurationV5 {
    pub domain: String,
    pub semantic: SemanticConfiguration,
    pub review: ReviewConfiguration,
    pub index: IndexConfiguration,
    pub frontier: FrontierConfiguration,
    pub native: NativeConfiguration,
    pub execution: ExecutionConfiguration,
    pub post_repair: PostRepairPolicyV2,
    pub external: ExternalConfigurationV3,
    pub own_r_root_tie: OwnRRootTieV4,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct TypedEnvelope {
    domain: String,
    system_runtime_lock_sha256: String,
    configuration: RoventStartupConfigurationV5,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoventTypedStartupV5 {
    envelope: TypedEnvelope,
    original_bytes: Vec<u8>,
}

impl RoventTypedStartupV5 {
    pub fn from_json(bytes: &[u8]) -> Result<Self, BackendError> {
        let raw = std::str::from_utf8(bytes).map_err(|_| invalid("startup document /5 UTF-8"))?;
        // Decode the original bytes twice: the common consumer verifies the
        // envelope domain/size; typed decoding rejects duplicate or extra
        // configuration fields without first collapsing them into a Value.
        RoventStartupDocumentV5::from_json(raw)
            .map_err(|_| invalid("startup document /5 envelope"))?;
        let envelope: TypedEnvelope = serde_json::from_slice(bytes)
            .map_err(|_| invalid("strict typed startup configuration /5 JSON"))?;
        let value = Self {
            envelope,
            original_bytes: bytes.to_vec(),
        };
        value.validate()?;
        Ok(value)
    }

    fn configuration(&self) -> &RoventStartupConfigurationV5 {
        &self.envelope.configuration
    }

    pub fn fields(&self) -> ConfigurationFields<'_> {
        let c = self.configuration();
        ConfigurationFields {
            semantic: &c.semantic,
            review: &c.review,
            index: &c.index,
            frontier: &c.frontier,
            native: &c.native,
            execution: &c.execution,
        }
    }

    pub fn validate(&self) -> Result<(), BackendError> {
        let c = self.configuration();
        if c.domain != ROVENT_STARTUP_CONFIGURATION_V5_DOMAIN {
            return Err(invalid("startup configuration /5 domain"));
        }
        self.fields().validate()?;
        joined_external_question_limit(&c.execution, c.external)?;
        match (&c.execution, c.own_r_root_tie) {
            (ExecutionConfiguration::OwnResearcher { .. }, OwnRRootTieV4::Own { .. })
            | (ExecutionConfiguration::RolesOnly { .. }, OwnRRootTieV4::NotApplicable {})
            | (
                ExecutionConfiguration::ExternalResearcher { .. },
                OwnRRootTieV4::NotApplicable {},
            ) => Ok(()),
            _ => Err(invalid(
                "own R root tie policy differs from execution assembly",
            )),
        }
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, BackendError> {
        self.validate()?;
        canonical(&self.envelope)
    }

    pub fn configuration_digest(&self) -> Result<[u8; 32], BackendError> {
        self.validate()?;
        framed_digest(
            ROVENT_STARTUP_CONFIGURATION_V5_DOMAIN,
            &canonical(self.configuration())?,
        )
    }

    pub fn search_configuration_digest(&self) -> Result<[u8; 32], BackendError> {
        self.validate()?;
        let c = self.configuration();
        framed_digest(
            ROVENT_SEARCH_CONFIGURATION_V5_DOMAIN,
            &canonical(&(
                &c.semantic,
                &c.review,
                &c.index,
                &c.frontier,
                &c.execution,
                c.post_repair,
                c.external,
                c.own_r_root_tie,
            ))?,
        )
    }

    pub fn nnuec_search_conditions_digest(&self) -> Result<Option<[u8; 32]>, BackendError> {
        self.validate()?;
        let base = self.fields().nnuec_conditions_digest()?;
        if matches!(
            &self.configuration().execution,
            ExecutionConfiguration::OwnResearcher {
                value: ResearchValueConfiguration::NnuecV2 { .. },
                ..
            }
        ) {
            let base = base.ok_or_else(|| invalid("own R NNUEC conditions missing"))?;
            return framed_digest(
                ROVENT_OWN_NNUEC_CONDITIONS_V4_DOMAIN,
                &canonical(&(base, self.configuration().own_r_root_tie))?,
            )
            .map(Some);
        }
        Ok(base)
    }

    pub fn require_supported_native_capabilities(&self) -> Result<(), BackendError> {
        self.validate()?;
        self.fields().require_supported_native_capabilities()
    }

    pub fn own_r_root_tie_policy(&self) -> Result<Option<CpuRootTiePolicy>, BackendError> {
        self.validate()?;
        Ok(match self.configuration().own_r_root_tie {
            OwnRRootTieV4::NotApplicable {} => None,
            OwnRRootTieV4::Own { policy } => Some(policy.actual()),
        })
    }

    pub fn post_repair_policy(&self) -> PostRepairRecheckPolicy {
        self.configuration().post_repair.actual()
    }

    pub fn external_question_limit(&self) -> Result<Option<ExternalGoConstraint>, BackendError> {
        self.validate()?;
        let c = self.configuration();
        joined_external_question_limit(&c.execution, c.external)
    }

    pub fn admit_against_v3(&self, lock: &RoventInputLockV3) -> Result<(), BackendError> {
        let raw = std::str::from_utf8(&self.original_bytes)
            .map_err(|_| invalid("startup document /5 UTF-8"))?;
        RoventStartupDocumentV5::from_json(raw)
            .and_then(|document| document.validate_against(lock, &self.original_bytes))
            .map_err(|_| invalid("startup /5 bytes or runtime lock mismatch"))?;
        self.validate()?;
        let manifest = &lock.manifest;
        if manifest.startup_configuration_sha256 != super::hex(self.configuration_digest()?)
            || manifest.startup_search_configuration_sha256
                != super::hex(self.search_configuration_digest()?)
        {
            return Err(invalid("locked startup /5 digest differs"));
        }
        // System-native assets are OS-owned and must not be interpreted as a
        // historical CUDA bundle. No load or provider-readiness claim is made.
        if let NativeRuntimeSourceV6::SystemCuda(spec) = &manifest.system_runtime_lock.runtime {
            if self.fields().native.loading_profile.is_some()
                || self.fields().native.cuda_arena_bytes != Some(spec.cuda_arena_bytes)
                || self.fields().native.drain_ms != spec.drain_ms
            {
                return Err(invalid(
                    "system CUDA native configuration differs from runtime lock",
                ));
            }
        }
        let system = match &manifest.system_runtime_lock.runtime {
            NativeRuntimeSourceV6::SystemCuda(spec) => Some(spec),
            NativeRuntimeSourceV6::Cpu => None,
        };
        self.fields().admit_shared_with_system(
            &manifest.engine,
            manifest.coordinator.view(),
            manifest.cleanup_max_ms,
            system,
            || self.configuration_digest(),
            || self.search_configuration_digest(),
            || self.nnuec_search_conditions_digest(),
        )?;
        match (&manifest.own_r_root_tie, self.own_r_root_tie_policy()?) {
            (RoventRootTieDeclarationV2::NotApplicable { .. }, None) => {}
            (RoventRootTieDeclarationV2::Own { policy, .. }, Some(actual))
                if matches!(
                    (policy, actual),
                    (
                        RoventRootTiePolicyV2::LegacyFirstSearched,
                        CpuRootTiePolicy::LegacyFirstSearched
                    ) | (
                        RoventRootTiePolicyV2::RulesOrderConfirmed,
                        CpuRootTiePolicy::RulesOrderConfirmed
                    )
                ) => {}
            _ => return Err(invalid("locked own R policy differs from startup /5")),
        }
        if let ExecutionConfiguration::ExternalResearcher { slice, .. } =
            &self.configuration().execution
        {
            let whole = slice
                .quantum_ms
                .checked_add(slice.stop_drain_reserve_ms)
                .ok_or_else(|| invalid("external quantum plus stop reserve overflows"))?;
            if whole > manifest.wall_time_max_ms {
                return Err(invalid(
                    "external quantum plus stop reserve exceeds locked wall",
                ));
            }
            if let Some(ExternalGoConstraint::MoveTime { milliseconds }) =
                self.external_question_limit()?
            {
                let total = milliseconds
                    .checked_add(slice.stop_drain_reserve_ms)
                    .ok_or_else(|| invalid("external movetime plus stop reserve overflows"))?;
                if total > manifest.wall_time_max_ms {
                    return Err(invalid(
                        "external movetime plus stop reserve exceeds locked wall",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rovent_bootstrap::configuration::{CuDnnSearch, NativeConfiguration};
    use rz_eval::onnx::Provider;
    use rz_experiments::{RoventDevice, SystemLibrarySpecV6, SystemPlatformV6};

    fn spec(arena: u64) -> SystemRuntimeSpecV6 {
        let libraries = ["ort-core", "ort-provider-cuda"]
            .into_iter()
            .enumerate()
            .map(|(index, role)| SystemLibrarySpecV6 {
                role: role.into(),
                requested_path: format!("/fixture/{role}.so"),
                canonical_path: format!("/fixture/{role}.so"),
                version: None,
                sha256: format!("{:064x}", index + 1),
                bytes: 1,
                device: 1,
                inode: (index + 1) as u64,
                required_mapping: true,
            })
            .collect();
        SystemRuntimeSpecV6 {
            platform: SystemPlatformV6::Linux,
            libraries,
            system_identity_sha256: "a".repeat(64),
            cuda_device_id: 0,
            cuda_arena_bytes: arena,
            startup_ms: 100,
            drain_ms: 10,
            capture_max_libraries: 2,
            capture_max_file_bytes: 1,
            capture_max_total_bytes: 2,
            capture_timeout_ms: 50,
        }
    }

    fn native(arena: u64) -> NativeConfiguration {
        NativeConfiguration {
            cache_public_memory: true,
            device_public_memory: false,
            cudnn_conv_search: CuDnnSearch::LegacyImplicitExhaustive,
            loading_profile: None,
            cuda_arena_bytes: Some(arena),
            drain_ms: 10,
            prepared_host_bytes_per_ticket: 1,
            output_host_bytes_per_ticket: 1,
            command_capacity: 1,
        }
    }

    fn cuda_device(ordinal: u32) -> RoventDevice {
        RoventDevice::Cuda {
            ordinal,
            registered_device: "test-device".into(),
        }
    }

    #[test]
    fn typed_v5_document_is_strict_and_does_not_accept_legacy_domain() {
        let payload = serde_json::json!({
            "domain":"rz-rovent-startup-document/5",
            "system_runtime_lock_sha256":"a".repeat(64),
            "configuration":{
                "domain":"rz-rovent-startup-configuration/5",
                "semantic":{"beam_width":4,"line_plies":4,"max_nodes":4096,"max_records":64,"max_role_calls":4096,"maximum_root_prefix_plies":256},
                "review":{"anchor_mode":"anchored_compact","token_budget":512,"max_records":64,"max_anchors":4},
                "index":{"active_bundles":256,"records_per_bundle":64},
                "frontier":{"maximum_goal_plies":256,"maximum_frames":64,"maximum_paths":4096,"maximum_mate_nodes":4096,"maximum_native_receipts":64},
                "native":{"cache_public_memory":true,"device_public_memory":false,"cudnn_conv_search":"legacy_implicit_exhaustive","loading_profile":null,"cuda_arena_bytes":1024,"drain_ms":10,"prepared_host_bytes_per_ticket":1024,"output_host_bytes_per_ticket":1024,"command_capacity":1},
                "execution":{"assembly":"roles_only","event_order":"fixed","role_tickets_max":1,"shared_nnuec_readout":null},
                "post_repair":"disabled",
                "external":{"kind":"absent"},
                "own_r_root_tie":{"kind":"not_applicable"}
            }
        });
        let original = serde_json::to_vec_pretty(&payload).unwrap();
        let parsed = RoventTypedStartupV5::from_json(&original).unwrap();
        assert_eq!(parsed.fields().native.cuda_arena_bytes, Some(1024));
        let mut wrong = payload.clone();
        wrong["configuration"]["domain"] = serde_json::json!("rz-rovent-startup-configuration/4");
        assert!(RoventTypedStartupV5::from_json(&serde_json::to_vec(&wrong).unwrap()).is_err());
        let mut extra = payload;
        extra["configuration"]["unrecognised"] = serde_json::json!(true);
        assert!(RoventTypedStartupV5::from_json(&serde_json::to_vec(&extra).unwrap()).is_err());
    }
    #[test]
    #[cfg(target_pointer_width = "64")]
    fn system_profile_uses_registered_arena_without_changing_legacy_limit() {
        let arena = 8u64 << 30;
        let config = native(arena);
        let device = cuda_device(0);
        assert!(config.onnx(&device, true).is_err());
        let selected = config.onnx_system(&device, &spec(arena)).unwrap();
        assert!(
            matches!(selected.provider, Provider::Cuda { device_id: 0, arena_bytes } if arena_bytes as u64 == arena)
        );
    }

    #[test]
    fn system_profile_rejects_mismatch_without_attempting_a_gpu_run() {
        let arena = 1024;
        let config = native(arena);
        let mut registered = spec(arena);
        assert!(config.onnx_system(&cuda_device(1), &registered).is_err());
        assert!(config.onnx_system(&RoventDevice::Cpu, &registered).is_err());
        registered.cuda_arena_bytes += 1;
        assert!(config.onnx_system(&cuda_device(0), &registered).is_err());
    }
}
