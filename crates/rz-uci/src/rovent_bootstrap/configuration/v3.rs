//! Explicit external-researcher question selection in startup configuration /3.
//!
//! This schema never derives Combined from legacy fields or chooses a smaller
//! limit from profile capabilities. Its actual constraint remains unchanged;
//! legacy execution values are independent ceilings for active dimensions.
//! A selected movetime is independent of the worker's resumable slice quantum.
//! Profile purpose, supported query and loaded process capabilities are checked
//! separately by the retained source consumer before any real question starts.

use super::v2::PostRepairPolicyV2;
use super::{
    ConfigurationFields, ExecutionConfiguration, FrontierConfiguration, IndexConfiguration,
    NativeConfiguration, ROVENT_STARTUP_CONFIGURATION_MAX_BYTES, ReviewConfiguration,
    SemanticConfiguration, canonical, framed_digest, invalid,
};
use rz_eval::error::BackendError;
use rz_experiments::RoventInputLock;
use rz_search::external_cpu::ExternalGoConstraint;
use rz_search::pals::engine::PostRepairRecheckPolicy;
use serde::{Deserialize, Serialize};

pub const ROVENT_STARTUP_CONFIGURATION_V3_DOMAIN: &str = "rz-rovent-startup-configuration/3";
pub const ROVENT_SEARCH_CONFIGURATION_V3_DOMAIN: &str = "rz-rovent-search-configuration/3";

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExternalQuestionLimitV3 {
    Depth {
        depth: u16,
    },
    Nodes {
        reported_nodes: u64,
    },
    #[serde(rename = "movetime")]
    MoveTime {
        milliseconds: u64,
    },
    Combined {
        depth: u16,
        reported_nodes: u64,
    },
}
impl ExternalQuestionLimitV3 {
    /// Exact finite query. Capability checks may reject it, never rewrite it.
    pub fn actual(self) -> Result<ExternalGoConstraint, BackendError> {
        let actual = match self {
            Self::Depth { depth } => ExternalGoConstraint::Depth { depth },
            Self::Nodes { reported_nodes } => ExternalGoConstraint::Nodes { reported_nodes },
            Self::MoveTime { milliseconds } => ExternalGoConstraint::MoveTime { milliseconds },
            Self::Combined {
                depth,
                reported_nodes,
            } => ExternalGoConstraint::Combined {
                depth,
                reported_nodes,
            },
        };
        if actual
            .requested_depth()
            .is_some_and(|depth| !(1..=64).contains(&depth))
            || actual.requested_nodes() == Some(0)
            || matches!(
                actual,
                ExternalGoConstraint::MoveTime {
                    milliseconds: 0 | 180_001..
                }
            )
        {
            return Err(invalid("explicit external question exceeds finite bounds"));
        }
        Ok(actual)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExternalConfigurationV3 {
    Absent {},
    Researcher {
        question_limit: ExternalQuestionLimitV3,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventStartupConfigurationV3 {
    pub domain: String,
    pub semantic: SemanticConfiguration,
    pub review: ReviewConfiguration,
    pub index: IndexConfiguration,
    pub frontier: FrontierConfiguration,
    pub native: NativeConfiguration,
    pub execution: ExecutionConfiguration,
    pub post_repair: PostRepairPolicyV2,
    pub external: ExternalConfigurationV3,
}
impl RoventStartupConfigurationV3 {
    pub fn from_json(bytes: &[u8]) -> Result<Self, BackendError> {
        if bytes.is_empty() || bytes.len() > ROVENT_STARTUP_CONFIGURATION_MAX_BYTES {
            return Err(invalid("startup configuration /3 byte budget"));
        }
        let value: Self = serde_json::from_slice(bytes)
            .map_err(|_| invalid("strict startup configuration /3 JSON"))?;
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
    pub fn validate(&self) -> Result<(), BackendError> {
        if self.domain != ROVENT_STARTUP_CONFIGURATION_V3_DOMAIN {
            return Err(invalid("startup configuration /3 domain"));
        }
        self.fields().validate()?;
        self.joined_external_question_limit().map(|_| ())
    }
    fn joined_external_question_limit(&self) -> Result<Option<ExternalGoConstraint>, BackendError> {
        joined_external_question_limit(&self.execution, self.external)
    }
    pub fn external_question_limit(&self) -> Result<Option<ExternalGoConstraint>, BackendError> {
        self.validate()?;
        self.joined_external_question_limit()
    }
    pub fn admit_against(&self, lock: &RoventInputLock) -> Result<(), BackendError> {
        self.validate()?;
        self.fields().admit_against(
            lock,
            || self.configuration_digest(),
            || self.search_configuration_digest(),
            || self.nnuec_search_conditions_digest(),
        )?;
        if let ExecutionConfiguration::ExternalResearcher { slice, .. } = &self.execution {
            let whole_question_ms = slice
                .quantum_ms
                .checked_add(slice.stop_drain_reserve_ms)
                .ok_or_else(|| invalid("external quantum plus stop reserve overflows"))?;
            if whole_question_ms > lock.manifest.wall_time_max_ms {
                return Err(invalid(
                    "original external quantum plus stop reserve exceeds locked wall",
                ));
            }
            if let Some(ExternalGoConstraint::MoveTime { milliseconds }) =
                self.joined_external_question_limit()?
            {
                let original_question_ms = milliseconds
                    .checked_add(slice.stop_drain_reserve_ms)
                    .ok_or_else(|| {
                        invalid("explicit external movetime plus stop reserve overflows")
                    })?;
                if original_question_ms > lock.manifest.wall_time_max_ms {
                    return Err(invalid(
                        "explicit external movetime plus stop reserve exceeds locked wall",
                    ));
                }
            }
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, BackendError> {
        self.validate()?;
        canonical(self)
    }
    pub fn configuration_digest(&self) -> Result<[u8; 32], BackendError> {
        framed_digest(
            ROVENT_STARTUP_CONFIGURATION_V3_DOMAIN,
            &self.canonical_bytes()?,
        )
    }
    pub fn search_configuration_digest(&self) -> Result<[u8; 32], BackendError> {
        self.validate()?;
        framed_digest(
            ROVENT_SEARCH_CONFIGURATION_V3_DOMAIN,
            &canonical(&(
                &self.semantic,
                &self.review,
                &self.index,
                &self.frontier,
                &self.execution,
                self.post_repair,
                self.external,
            ))?,
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
    pub fn post_repair_policy(&self) -> PostRepairRecheckPolicy {
        self.post_repair.actual()
    }
    pub fn verify_post_repair_readback(
        &self,
        actual: PostRepairRecheckPolicy,
    ) -> Result<(), BackendError> {
        self.validate()?;
        if actual != self.post_repair_policy() {
            return Err(invalid("actual post-repair policy differs from startup /3"));
        }
        Ok(())
    }
}

// The /3 and /4 declarations share exactly the same question/ceiling
// validation. This does not transform either version's persisted document.
pub(super) fn joined_external_question_limit(
    execution: &ExecutionConfiguration,
    external: ExternalConfigurationV3,
) -> Result<Option<ExternalGoConstraint>, BackendError> {
    match (execution, external) {
        (
            ExecutionConfiguration::RolesOnly { .. } | ExecutionConfiguration::OwnResearcher { .. },
            ExternalConfigurationV3::Absent {},
        ) => Ok(None),
        (
            ExecutionConfiguration::ExternalResearcher {
                max_reported_nodes,
                cpu_depth,
                cpu_nodes_per_task,
                slice,
                ..
            },
            ExternalConfigurationV3::Researcher { question_limit },
        ) => {
            let actual = question_limit.actual()?;
            if actual
                .requested_depth()
                .is_some_and(|depth| depth > *cpu_depth)
                || actual.requested_nodes().is_some_and(|nodes| {
                    nodes > *cpu_nodes_per_task
                        || nodes > slice.requested_reported_nodes
                        || nodes > *max_reported_nodes
                })
            {
                return Err(invalid(
                    "explicit external question exceeds original execution ceilings",
                ));
            }
            Ok(Some(actual))
        }
        _ => Err(invalid(
            "explicit external policy differs from actual execution assembly",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rovent_bootstrap::configuration::{
        ROVENT_STARTUP_CONFIGURATION_DOMAIN, RoventStartupConfiguration,
        StartupConfigurationDocument,
    };
    use crate::rovent_bootstrap::external_session::external_researcher_implementation_digest_from_metadata;
    use crate::rovent_root_admission::RoventRootAdmissionPolicy;
    use rz_contracts::rovent::ExecutionMode;
    use rz_eval::error::FailureKind;
    use rz_experiments::RoventExecutionManifest;
    use serde_json::{Value, json};

    fn pc_wire() -> Value {
        json!({
            "domain": ROVENT_STARTUP_CONFIGURATION_V3_DOMAIN,
            "semantic": {
                "beam_width": 4, "line_plies": 4, "max_nodes": 4096,
                "max_records": 64, "max_role_calls": 4096,
                "maximum_root_prefix_plies": 256
            },
            "review": {
                "anchor_mode": "anchored_compact", "token_budget": 512,
                "max_records": 64, "max_anchors": 4
            },
            "index": {"active_bundles": 256, "records_per_bundle": 64},
            "frontier": {
                "maximum_goal_plies": 256, "maximum_frames": 64,
                "maximum_paths": 4096, "maximum_mate_nodes": 4096,
                "maximum_native_receipts": 64
            },
            "native": {
                "cache_public_memory": true, "device_public_memory": false,
                "cudnn_conv_search": "legacy_implicit_exhaustive",
                "loading_profile": null, "cuda_arena_bytes": null,
                "drain_ms": 30000, "prepared_host_bytes_per_ticket": 1048576,
                "output_host_bytes_per_ticket": 1048576, "command_capacity": 8
            },
            "execution": {
                "assembly": "roles_only", "event_order": "fixed",
                "role_tickets_max": 16, "shared_nnuec_readout": null
            },
            "post_repair": "disabled",
            "external": {"kind": "absent"}
        })
    }

    fn external_wire(question_limit: Value) -> Value {
        let mut wire = pc_wire();
        wire["execution"] = json!({
            "assembly": "external_researcher", "event_order": "fixed",
            "queues": {
                "waiting_jobs_max": 16, "worker_commands_max": 8,
                "role_tickets_max": 16, "paused_bytes_per_worker_max": 1048576
            },
            "max_rounds": 64, "max_reported_nodes": 64,
            "cpu_depth": 8, "cpu_nodes_per_task": 128,
            "profile": {
                "path": "fixtures/external-profile.json",
                "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                "bytes": 32, "source": "https://example.invalid/profile",
                "license": "fixture"
            },
            "slice": {
                "quantum_ms": 1000, "stop_drain_reserve_ms": 500,
                "requested_reported_nodes": 256, "multipv": 1
            },
            "peers": {"pc_peer": true, "researcher": true, "max_sources_per_root": 16}
        });
        wire["external"] = json!({"kind": "researcher", "question_limit": question_limit});
        wire
    }

    fn decode(wire: &Value) -> Result<RoventStartupConfigurationV3, BackendError> {
        RoventStartupConfigurationV3::from_json(&serde_json::to_vec(wire).unwrap())
    }

    fn external_metadata_lock(document: &StartupConfigurationDocument) -> RoventInputLock {
        let mut wire = crate::rovent_bootstrap::tests::wire(ExecutionMode::CpuPcr);
        let fields = document.fields();
        let ExecutionConfiguration::ExternalResearcher { profile, .. } = fields.execution else {
            panic!("external metadata fixture requires its original external profile")
        };
        let mut binary = wire["engine"]["binary"].clone();
        binary["path"] = json!("fixture/external-engine");
        wire["engine"]["researcher"] = json!({"provider":"external", "profile":{
            "domain":"rz-rovent-external-engine-profile/1", "profile":profile,
            "canonical_sha256":"8".repeat(64), "capability_sha256":"9".repeat(64),
            "purpose":"research_checker", "binary":binary, "source_commit":"b".repeat(40),
            "requested_threads":1, "requested_hash_mib":256
        }});
        wire["resources"]["pools"][1]["actors"] = json!(["external_r"]);
        wire["coordinator"]["policy_sha256"] = json!(crate::rovent_bootstrap::hex(
            document.configuration_digest().unwrap()
        ));
        wire["engine"]["search"]["search_configuration_sha256"] = json!(
            crate::rovent_bootstrap::hex(document.search_configuration_digest().unwrap())
        );
        wire["engine"]["search"]["registered_max_chess_plies"] =
            json!(fields.frontier.maximum_goal_plies);
        RoventExecutionManifest::from_json(&wire.to_string())
            .unwrap()
            .lock()
            .unwrap()
    }

    #[test]
    fn external_metadata_digest_joins_original_lock_and_policy_without_session() {
        let document = StartupConfigurationDocument::V3(
            decode(&external_wire(json!({"kind":"nodes", "reported_nodes":32}))).unwrap(),
        );
        let lock = external_metadata_lock(&document);
        lock.validate().unwrap();
        document.admit_against(&lock).unwrap();
        let exact = external_researcher_implementation_digest_from_metadata(
            &document,
            &lock,
            RoventRootAdmissionPolicy::ExactRegistered,
        )
        .unwrap();
        let dynamic = external_researcher_implementation_digest_from_metadata(
            &document,
            &lock,
            RoventRootAdmissionPolicy::DynamicGo,
        )
        .unwrap();
        assert_ne!(exact, dynamic);

        let mut changed_manifest = lock.manifest.clone();
        changed_manifest.run_id.push_str("-next");
        let changed_lock = changed_manifest.lock().unwrap();
        document.admit_against(&changed_lock).unwrap();
        assert_ne!(lock.canonical_sha256, changed_lock.canonical_sha256);
        assert_ne!(
            dynamic,
            external_researcher_implementation_digest_from_metadata(
                &document,
                &changed_lock,
                RoventRootAdmissionPolicy::DynamicGo,
            )
            .unwrap()
        );
    }

    #[test]
    fn external_metadata_digest_rejects_mismatched_configuration_and_lock() {
        let config = decode(&external_wire(json!({"kind":"nodes", "reported_nodes":32}))).unwrap();
        let document = StartupConfigurationDocument::V3(config.clone());
        let lock = external_metadata_lock(&document);

        let mut changed_semantic = config.clone();
        changed_semantic.semantic.max_nodes += 1;
        changed_semantic.validate().unwrap();
        let mut changed_question = config;
        changed_question.external = ExternalConfigurationV3::Researcher {
            question_limit: ExternalQuestionLimitV3::Nodes { reported_nodes: 31 },
        };
        changed_question.validate().unwrap();
        for changed in [changed_semantic, changed_question] {
            assert!(
                external_researcher_implementation_digest_from_metadata(
                    &StartupConfigurationDocument::V3(changed),
                    &lock,
                    RoventRootAdmissionPolicy::DynamicGo,
                )
                .is_err()
            );
        }

        let mut invalid_lock = lock;
        invalid_lock.canonical_sha256 = "A".repeat(64);
        assert!(invalid_lock.validate().is_err());
        assert!(
            external_researcher_implementation_digest_from_metadata(
                &document,
                &invalid_lock,
                RoventRootAdmissionPolicy::DynamicGo,
            )
            .is_err()
        );
        let roles_only = StartupConfigurationDocument::V3(decode(&pc_wire()).unwrap());
        assert_eq!(
            external_researcher_implementation_digest_from_metadata(
                &roles_only,
                &invalid_lock,
                RoventRootAdmissionPolicy::DynamicGo,
            )
            .unwrap_err()
            .kind,
            FailureKind::UnsupportedModel
        );
    }

    #[test]
    fn external_metadata_digest_rejects_unjoined_backend_and_profile() {
        let document = StartupConfigurationDocument::V3(
            decode(&external_wire(json!({"kind":"nodes", "reported_nodes":32}))).unwrap(),
        );
        let lock = external_metadata_lock(&document);
        let mut different_profile = serde_json::to_value(&lock.manifest).unwrap();
        different_profile["engine"]["researcher"]["profile"]["profile"]["sha256"] =
            json!("c".repeat(64));
        let different_profile = RoventExecutionManifest::from_json(&different_profile.to_string())
            .unwrap()
            .lock()
            .unwrap();
        different_profile.validate().unwrap();
        assert!(
            external_researcher_implementation_digest_from_metadata(
                &document,
                &different_profile,
                RoventRootAdmissionPolicy::DynamicGo,
            )
            .is_err()
        );

        let mut different_backend = serde_json::to_value(&lock.manifest).unwrap();
        different_backend["engine"]["mode"] = json!("hybrid_pcr");
        different_backend["engine"]["pc"]["backend"] = json!("ort_cuda");
        different_backend["engine"]["pc"]["device"] = json!({
            "device":"cuda", "ordinal":0, "registered_device":"fixture-declaration-only"
        });
        different_backend["resources"]["gpu_devices"] = json!(1);
        let different_backend = RoventExecutionManifest::from_json(&different_backend.to_string())
            .unwrap()
            .lock()
            .unwrap();
        different_backend.validate().unwrap();
        assert!(
            external_researcher_implementation_digest_from_metadata(
                &document,
                &different_backend,
                RoventRootAdmissionPolicy::DynamicGo,
            )
            .is_err()
        );
    }

    #[test]
    fn explicit_variants_keep_requested_numbers_below_independent_ceilings() {
        for (wire, expected) in [
            (
                json!({"kind": "depth", "depth": 4}),
                ExternalGoConstraint::Depth { depth: 4 },
            ),
            (
                json!({"kind": "nodes", "reported_nodes": 32}),
                ExternalGoConstraint::Nodes { reported_nodes: 32 },
            ),
            (
                json!({"kind": "movetime", "milliseconds": 3000}),
                ExternalGoConstraint::MoveTime { milliseconds: 3000 },
            ),
            (
                json!({"kind": "combined", "depth": 4, "reported_nodes": 32}),
                ExternalGoConstraint::Combined {
                    depth: 4,
                    reported_nodes: 32,
                },
            ),
        ] {
            let config = decode(&external_wire(wire)).unwrap();
            assert_eq!(config.external_question_limit().unwrap(), Some(expected));
            let bytes = config.canonical_bytes().unwrap();
            let document = StartupConfigurationDocument::from_json(&bytes).unwrap();
            assert!(matches!(document, StartupConfigurationDocument::V3(_)));
            assert_eq!(document.canonical_bytes().unwrap(), bytes);
            assert_eq!(document.external_question_limit().unwrap(), Some(expected));
        }
    }

    #[test]
    fn active_query_dimensions_must_fit_each_original_ceiling() {
        let original = external_wire(json!({"kind": "combined", "depth": 4, "reported_nodes": 32}));
        for (field, value) in [
            ("cpu_depth", 3),
            ("cpu_nodes_per_task", 31),
            ("max_reported_nodes", 31),
        ] {
            let mut wire = original.clone();
            wire["execution"][field] = json!(value);
            assert!(decode(&wire).is_err());
        }
        let mut wire = original;
        wire["execution"]["slice"]["requested_reported_nodes"] = json!(31);
        assert!(decode(&wire).is_err());
        for question in [
            json!({"kind": "depth", "depth": 0}),
            json!({"kind": "depth", "depth": 65}),
            json!({"kind": "nodes", "reported_nodes": 0}),
            json!({"kind": "movetime", "milliseconds": 0}),
            json!({"kind": "movetime", "milliseconds": 180001}),
        ] {
            assert!(decode(&external_wire(question)).is_err());
        }
    }

    #[test]
    fn strict_required_policies_and_execution_join_have_no_defaults() {
        for field in ["post_repair", "external"] {
            let mut wire = pc_wire();
            wire.as_object_mut().unwrap().remove(field);
            assert!(decode(&wire).is_err());
            wire[field] = Value::Null;
            assert!(decode(&wire).is_err());
        }
        for external in [
            json!({"kind": "absent", "question_limit": {"kind": "depth", "depth": 4}}),
            json!({"kind": "researcher", "question_limit": {"kind": "depth", "depth": 4}}),
            json!({"kind": "unknown"}),
        ] {
            let mut wire = pc_wire();
            wire["external"] = external;
            assert!(decode(&wire).is_err());
        }
        for question in [
            json!({"kind": "move_time", "milliseconds": 3000}),
            json!({"kind": "depth", "depth": 4, "reported_nodes": 32}),
            json!({"kind": "combined", "depth": 4}),
            json!({"kind": "nodes", "reported_nodes": 32, "extra": true}),
        ] {
            assert!(decode(&external_wire(question)).is_err());
        }
        let mut wire = external_wire(json!({"kind": "depth", "depth": 4}));
        wire["external"] = json!({"kind": "absent"});
        assert!(decode(&wire).is_err());
        assert_eq!(
            decode(&pc_wire())
                .unwrap()
                .external_question_limit()
                .unwrap(),
            None
        );
    }

    #[test]
    fn post_repair_and_query_are_joined_in_v3_search_digest() {
        let base = decode(&external_wire(json!({"kind": "depth", "depth": 4}))).unwrap();
        let mut changed = base.clone();
        changed.external = ExternalConfigurationV3::Researcher {
            question_limit: ExternalQuestionLimitV3::Depth { depth: 3 },
        };
        assert_ne!(
            base.configuration_digest().unwrap(),
            changed.configuration_digest().unwrap()
        );
        assert_ne!(
            base.search_configuration_digest().unwrap(),
            changed.search_configuration_digest().unwrap()
        );
        changed = base.clone();
        changed.post_repair = PostRepairPolicyV2::IterativeFrozenModelWdlV2;
        assert_ne!(
            base.search_configuration_digest().unwrap(),
            changed.search_configuration_digest().unwrap()
        );
        assert!(
            changed
                .verify_post_repair_readback(PostRepairRecheckPolicy::Disabled)
                .is_err()
        );
        assert!(
            changed
                .verify_post_repair_readback(PostRepairRecheckPolicy::IterativeFrozenModelWdlV2)
                .is_ok()
        );
        changed = base.clone();
        changed.native.prepared_host_bytes_per_ticket += 1;
        assert_ne!(
            base.configuration_digest().unwrap(),
            changed.configuration_digest().unwrap()
        );
        assert_eq!(
            base.search_configuration_digest().unwrap(),
            changed.search_configuration_digest().unwrap()
        );
    }

    #[test]
    fn old_external_documents_stay_decodable_without_implicit_query() {
        // Explicit independent legacy test wires; no production conversion exists.
        for version in [1, 2] {
            let mut wire = external_wire(json!({"kind": "depth", "depth": 4}));
            wire.as_object_mut().unwrap().remove("external");
            if version == 1 {
                wire["domain"] = json!(ROVENT_STARTUP_CONFIGURATION_DOMAIN);
                wire.as_object_mut().unwrap().remove("post_repair");
            } else {
                wire["domain"] = json!(super::super::v2::ROVENT_STARTUP_CONFIGURATION_V2_DOMAIN);
            }
            let bytes = serde_json::to_vec(&wire).unwrap();
            let document = StartupConfigurationDocument::from_json(&bytes).unwrap();
            let error = document.external_question_limit().unwrap_err();
            assert_eq!(error.kind, FailureKind::UnsupportedModel);
            let canonical = document.canonical_bytes().unwrap();
            if version == 1 {
                let direct = RoventStartupConfiguration::from_json(&bytes).unwrap();
                assert_eq!(canonical, direct.canonical_bytes().unwrap());
                assert_eq!(
                    document.search_configuration_digest().unwrap(),
                    direct.search_configuration_digest().unwrap()
                );
            } else {
                let direct =
                    super::super::v2::RoventStartupConfigurationV2::from_json(&bytes).unwrap();
                assert_eq!(canonical, direct.canonical_bytes().unwrap());
                assert_eq!(
                    document.search_configuration_digest().unwrap(),
                    direct.search_configuration_digest().unwrap()
                );
            }
            assert!(decode(&wire).is_err());
        }
    }
}
