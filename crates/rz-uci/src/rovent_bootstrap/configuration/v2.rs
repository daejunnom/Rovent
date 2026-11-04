//! Explicit startup configuration /2. This is not a migration of /1.
//!
//! The legacy DTO, decoder and canonical meanings remain with the parent /1
//! module. Version dispatch must select this decoder using the original /2
//! bytes; no domain substitution, fallback parse or implicit default exists.
//! Post-repair is a search choice, not neural inference or execution evidence.

use super::{
    ConfigurationFields, ExecutionConfiguration, FrontierConfiguration, IndexConfiguration,
    NativeConfiguration, ROVENT_STARTUP_CONFIGURATION_MAX_BYTES, ReviewConfiguration,
    SemanticConfiguration, canonical, framed_digest, invalid,
};
use rz_eval::error::BackendError;
use rz_experiments::RoventInputLock;
use rz_search::pals::engine::PostRepairRecheckPolicy;
use serde::{Deserialize, Serialize};

pub const ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN: &str = "rz-rovent-startup-configuration/2";
pub const ROVENT_SEARCH_CONFIGURATION_V2_DOMAIN: &str = "rz-rovent-search-configuration/2";

/// Required on every /2 document. No Default implementation grants an omitted
/// policy, and no other existing PALS policy is accepted by this wire schema.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case", try_from = "String")]
pub enum PostRepairPolicyV2 {
    Disabled,
    IterativeFrozenModelWdlV2,
}
impl TryFrom<String> for PostRepairPolicyV2 {
    type Error = &'static str;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "disabled" => Ok(Self::Disabled),
            "iterative_frozen_model_wdl_v2" => Ok(Self::IterativeFrozenModelWdlV2),
            _ => Err("unsupported explicit post-repair policy /2"),
        }
    }
}
impl PostRepairPolicyV2 {
    /// Both PC's enable/readback and PCR's constructor/readback consume this
    /// same actual policy. This conversion does not configure a live owner.
    pub fn actual(self) -> PostRepairRecheckPolicy {
        match self {
            Self::Disabled => PostRepairRecheckPolicy::Disabled,
            Self::IterativeFrozenModelWdlV2 => PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventStartupConfigurationV2 {
    pub domain: String,
    pub semantic: SemanticConfiguration,
    pub review: ReviewConfiguration,
    pub index: IndexConfiguration,
    pub frontier: FrontierConfiguration,
    pub native: NativeConfiguration,
    pub execution: ExecutionConfiguration,
    pub post_repair: PostRepairPolicyV2,
}

impl RoventStartupConfigurationV2 {
    pub fn from_json(bytes: &[u8]) -> Result<Self, BackendError> {
        if bytes.is_empty() || bytes.len() > ROVENT_STARTUP_CONFIGURATION_MAX_BYTES {
            return Err(invalid("startup configuration /2 byte budget"));
        }
        let value: Self = serde_json::from_slice(bytes)
            .map_err(|_| invalid("strict startup configuration /2 JSON"))?;
        value.validate()?;
        Ok(value)
    }

    pub fn fields(&self) -> ConfigurationFields<'_> {
        ConfigurationFields {
            semantic: &self.semantic,
            review: &self.review,
            index: &self.index,
            frontier: &self.frontier,
            native: &self.native,
            execution: &self.execution,
        }
    }
    /// Common constructor constraints are shared without constructing a /1
    /// DTO. The required /2 domain and policy remain version-specific values.
    pub fn validate(&self) -> Result<(), BackendError> {
        if self.domain != ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN {
            return Err(invalid("startup configuration /2 domain"));
        }
        self.fields().validate()
    }
    pub fn admit_against(&self, lock: &RoventInputLock) -> Result<(), BackendError> {
        self.validate()?;
        self.fields().admit_against(
            lock,
            || self.configuration_digest(),
            || self.search_configuration_digest(),
            || self.nnuec_search_conditions_digest(),
        )
    }
    pub fn nnuec_search_conditions_digest(&self) -> Result<Option<[u8; 32]>, BackendError> {
        self.validate()?;
        self.fields().nnuec_conditions_digest()
    }
    pub fn require_supported_native_capabilities(&self) -> Result<(), BackendError> {
        self.validate()?;
        self.fields().require_supported_native_capabilities()
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, BackendError> {
        self.validate()?;
        canonical(self)
    }
    pub fn configuration_digest(&self) -> Result<[u8; 32], BackendError> {
        framed_digest(
            ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN,
            &self.canonical_bytes()?,
        )
    }
    /// The unchanged common search subset plus the required explicit policy.
    /// Native storage/provider controls stay outside this search-only subset.
    pub fn search_configuration_digest(&self) -> Result<[u8; 32], BackendError> {
        self.validate()?;
        framed_digest(
            ROVENT_SEARCH_CONFIGURATION_V2_DOMAIN,
            &canonical(&(
                &self.semantic,
                &self.review,
                &self.index,
                &self.frontier,
                &self.execution,
                self.post_repair,
            ))?,
        )
    }
    pub fn post_repair_policy(&self) -> PostRepairRecheckPolicy {
        self.post_repair.actual()
    }
    /// Call only with an actual owner getter after its enable/constructor step.
    /// This comparison is a join, not proof of a performed repair or recheck.
    pub fn verify_post_repair_readback(
        &self,
        actual: PostRepairRecheckPolicy,
    ) -> Result<(), BackendError> {
        self.validate()?;
        if actual != self.post_repair_policy() {
            return Err(invalid("actual post-repair policy differs from startup /2"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{AnchorMode, CuDnnSearch, EventOrder, RoventStartupConfiguration};
    use super::*;
    use serde_json::json;

    fn pc(policy: PostRepairPolicyV2) -> RoventStartupConfigurationV2 {
        RoventStartupConfigurationV2 {
            domain: ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN.into(),
            semantic: SemanticConfiguration {
                beam_width: 4,
                line_plies: 4,
                max_nodes: 4096,
                max_records: 64,
                max_role_calls: 4096,
                maximum_root_prefix_plies: 256,
            },
            review: ReviewConfiguration {
                anchor_mode: AnchorMode::AnchoredCompact,
                token_budget: 512,
                max_records: 64,
                max_anchors: 4,
            },
            index: IndexConfiguration {
                active_bundles: 256,
                records_per_bundle: 64,
            },
            frontier: FrontierConfiguration {
                maximum_goal_plies: 256,
                maximum_frames: 64,
                maximum_paths: 4096,
                maximum_mate_nodes: 4096,
                maximum_native_receipts: 64,
            },
            native: NativeConfiguration {
                cache_public_memory: true,
                device_public_memory: false,
                cudnn_conv_search: CuDnnSearch::LegacyImplicitExhaustive,
                loading_profile: None,
                cuda_arena_bytes: None,
                drain_ms: 30_000,
                prepared_host_bytes_per_ticket: 1 << 20,
                output_host_bytes_per_ticket: 1 << 20,
                command_capacity: 8,
            },
            execution: ExecutionConfiguration::RolesOnly {
                event_order: EventOrder::Fixed,
                role_tickets_max: 16,
                shared_nnuec_readout: None,
            },
            post_repair: policy,
        }
    }

    #[test]
    fn strict_v2_requires_explicit_supported_policy_and_rejects_other_domains() {
        for policy in [
            PostRepairPolicyV2::Disabled,
            PostRepairPolicyV2::IterativeFrozenModelWdlV2,
        ] {
            let config = pc(policy);
            let bytes = config.canonical_bytes().unwrap();
            assert_eq!(
                RoventStartupConfigurationV2::from_json(&bytes).unwrap(),
                config
            );
            // The original /1 decoder is still its own closed schema.
            assert!(RoventStartupConfiguration::from_json(&bytes).is_err());
        }
        let mut missing = serde_json::to_value(pc(PostRepairPolicyV2::Disabled)).unwrap();
        missing.as_object_mut().unwrap().remove("post_repair");
        assert!(
            RoventStartupConfigurationV2::from_json(&serde_json::to_vec(&missing).unwrap())
                .is_err()
        );
        for unsupported in [
            json!(null),
            json!("same_repaired_line_once_v1"),
            json!({"policy":"disabled"}),
            json!({"disabled":null}),
            json!({"iterative_frozen_model_wdl_v2":null}),
        ] {
            let mut value = serde_json::to_value(pc(PostRepairPolicyV2::Disabled)).unwrap();
            value["post_repair"] = unsupported;
            assert!(
                RoventStartupConfigurationV2::from_json(&serde_json::to_vec(&value).unwrap())
                    .is_err()
            );
        }
        let mut wrong_domain = pc(PostRepairPolicyV2::Disabled);
        wrong_domain.domain = super::super::ROVENT_STARTUP_CONFIGURATION_DOMAIN.into();
        assert!(
            RoventStartupConfigurationV2::from_json(&serde_json::to_vec(&wrong_domain).unwrap())
                .is_err()
        );
        let mut unknown = serde_json::to_value(pc(PostRepairPolicyV2::Disabled)).unwrap();
        unknown["post_repair_default"] = json!("disabled");
        assert!(
            RoventStartupConfigurationV2::from_json(&serde_json::to_vec(&unknown).unwrap())
                .is_err()
        );
    }

    #[test]
    fn v2_policy_changes_full_and_search_hashes_and_requires_exact_actual_readback() {
        let disabled = pc(PostRepairPolicyV2::Disabled);
        let mut selected = disabled.clone();
        selected.post_repair = PostRepairPolicyV2::IterativeFrozenModelWdlV2;
        assert_ne!(
            disabled.configuration_digest().unwrap(),
            selected.configuration_digest().unwrap()
        );
        assert_ne!(
            disabled.search_configuration_digest().unwrap(),
            selected.search_configuration_digest().unwrap()
        );
        assert_eq!(
            selected.post_repair_policy(),
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2
        );
        selected
            .verify_post_repair_readback(PostRepairRecheckPolicy::IterativeFrozenModelWdlV2)
            .unwrap();
        assert!(
            selected
                .verify_post_repair_readback(PostRepairRecheckPolicy::Disabled)
                .is_err()
        );
        assert!(
            disabled
                .verify_post_repair_readback(PostRepairRecheckPolicy::IterativeFrozenModelWdlV2)
                .is_err()
        );
        let search = selected.search_configuration_digest().unwrap();
        let full = selected.configuration_digest().unwrap();
        selected.native.output_host_bytes_per_ticket += 1;
        assert_eq!(selected.search_configuration_digest().unwrap(), search);
        assert_ne!(selected.configuration_digest().unwrap(), full);
    }
}
