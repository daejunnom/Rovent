//! Explicit own-R root tie policy for startup configuration /4.
//!
//! This is a new wire domain; no earlier declaration is upgraded. A /4
//! declaration is not itself an execution lock or evidence of an owner.

use super::v2::PostRepairPolicyV2;
use super::v3::{ExternalConfigurationV3, joined_external_question_limit};
use super::{
    ConfigurationFields, ExecutionConfiguration, FrontierConfiguration, IndexConfiguration,
    NativeConfiguration, ROVENT_STARTUP_CONFIGURATION_MAX_BYTES, ResearchValueConfiguration,
    ReviewConfiguration, SemanticConfiguration, canonical, framed_digest, invalid,
};
use rz_eval::error::BackendError;
use rz_search::cpu::CpuRootTiePolicy;
use rz_search::external_cpu::ExternalGoConstraint;
use rz_search::pals::engine::PostRepairRecheckPolicy;
use serde::{Deserialize, Serialize};

pub const ROVENT_STARTUP_CONFIGURATION_V4_DOMAIN: &str = "rz-rovent-startup-configuration/4";
pub const ROVENT_SEARCH_CONFIGURATION_V4_DOMAIN: &str = "rz-rovent-search-configuration/4";
pub const ROVENT_OWN_NNUEC_CONDITIONS_V4_DOMAIN: &str = "rz-rovent-own-nnuec-search-conditions/4";

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RootTieSelectionV4 {
    LegacyFirstSearched,
    RulesOrderConfirmed,
}

impl RootTieSelectionV4 {
    pub fn actual(self) -> CpuRootTiePolicy {
        match self {
            Self::LegacyFirstSearched => CpuRootTiePolicy::LegacyFirstSearched,
            Self::RulesOrderConfirmed => CpuRootTiePolicy::RulesOrderConfirmed,
        }
    }
}

/// PC and external R never acquire an own-R policy or create a CPU R owner.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OwnRRootTieV4 {
    NotApplicable {},
    Own { policy: RootTieSelectionV4 },
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventStartupConfigurationV4 {
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

impl RoventStartupConfigurationV4 {
    pub fn from_json(bytes: &[u8]) -> Result<Self, BackendError> {
        if bytes.is_empty() || bytes.len() > ROVENT_STARTUP_CONFIGURATION_MAX_BYTES {
            return Err(invalid("startup configuration /4 byte budget"));
        }
        let value: Self = serde_json::from_slice(bytes)
            .map_err(|_| invalid("strict startup configuration /4 JSON"))?;
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
        if self.domain != ROVENT_STARTUP_CONFIGURATION_V4_DOMAIN {
            return Err(invalid("startup configuration /4 domain"));
        }
        self.fields().validate()?;
        joined_external_question_limit(&self.execution, self.external)?;
        match (&self.execution, self.own_r_root_tie) {
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

    pub fn own_r_root_tie_policy(&self) -> Result<Option<CpuRootTiePolicy>, BackendError> {
        self.validate()?;
        Ok(match self.own_r_root_tie {
            OwnRRootTieV4::NotApplicable {} => None,
            OwnRRootTieV4::Own { policy } => Some(policy.actual()),
        })
    }

    pub fn external_question_limit(&self) -> Result<Option<ExternalGoConstraint>, BackendError> {
        self.validate()?;
        joined_external_question_limit(&self.execution, self.external)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, BackendError> {
        self.validate()?;
        canonical(self)
    }

    pub fn configuration_digest(&self) -> Result<[u8; 32], BackendError> {
        framed_digest(
            ROVENT_STARTUP_CONFIGURATION_V4_DOMAIN,
            &self.canonical_bytes()?,
        )
    }

    pub fn search_configuration_digest(&self) -> Result<[u8; 32], BackendError> {
        self.validate()?;
        framed_digest(
            ROVENT_SEARCH_CONFIGURATION_V4_DOMAIN,
            &canonical(&(
                &self.semantic,
                &self.review,
                &self.index,
                &self.frontier,
                &self.execution,
                self.post_repair,
                self.external,
                self.own_r_root_tie,
            ))?,
        )
    }

    pub fn nnuec_search_conditions_digest(&self) -> Result<Option<[u8; 32]>, BackendError> {
        self.validate()?;
        let base = self.fields().nnuec_conditions_digest()?;
        if matches!(
            &self.execution,
            ExecutionConfiguration::OwnResearcher {
                value: ResearchValueConfiguration::NnuecV2 { .. },
                ..
            }
        ) {
            let base = base.ok_or_else(|| invalid("own R NNUEC conditions missing"))?;
            return framed_digest(
                ROVENT_OWN_NNUEC_CONDITIONS_V4_DOMAIN,
                &canonical(&(base, self.own_r_root_tie))?,
            )
            .map(Some);
        }
        // P/C shared readout is independent of own R's policy.
        Ok(base)
    }

    pub fn require_supported_native_capabilities(&self) -> Result<(), BackendError> {
        self.validate()?;
        self.fields().require_supported_native_capabilities()
    }

    pub fn post_repair_policy(&self) -> PostRepairRecheckPolicy {
        self.post_repair.actual()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rovent_bootstrap::configuration::{
        StartupConfigurationDocument, v3::RoventStartupConfigurationV3,
    };
    use rz_contracts::rovent::ExecutionMode;
    use rz_experiments::RoventExecutionManifest;
    use serde_json::{Value, json};

    fn pc_wire() -> Value {
        json!({
            "domain": ROVENT_STARTUP_CONFIGURATION_V4_DOMAIN,
            "semantic": {"beam_width":4,"line_plies":4,"max_nodes":4096,"max_records":64,"max_role_calls":4096,"maximum_root_prefix_plies":256},
            "review": {"anchor_mode":"anchored_compact","token_budget":512,"max_records":64,"max_anchors":4},
            "index": {"active_bundles":256,"records_per_bundle":64},
            "frontier": {"maximum_goal_plies":256,"maximum_frames":64,"maximum_paths":4096,"maximum_mate_nodes":4096,"maximum_native_receipts":64},
            "native": {"cache_public_memory":true,"device_public_memory":false,"cudnn_conv_search":"legacy_implicit_exhaustive","loading_profile":null,"cuda_arena_bytes":null,"drain_ms":30000,"prepared_host_bytes_per_ticket":1048576,"output_host_bytes_per_ticket":1048576,"command_capacity":8},
            "execution": {"assembly":"roles_only","event_order":"fixed","role_tickets_max":16,"shared_nnuec_readout":null},
            "post_repair":"disabled", "external":{"kind":"absent"},
            "own_r_root_tie":{"kind":"not_applicable"}
        })
    }

    fn own_wire() -> Value {
        let mut wire = pc_wire();
        wire["execution"] = json!({
            "assembly":"own_researcher", "event_order":"fixed",
            "queues":{"waiting_jobs_max":8,"worker_commands_max":8,"role_tickets_max":16,"paused_bytes_per_worker_max":1048576},
            "max_rounds":16,"max_native_nodes":4096,"cpu_depth":2,"cpu_nodes_per_task":1024,
            "cpu_plan":{"profile":"independent","tt_entries":4096,"max_depth":8,"quiescence_ply":4,"ordering":"legal_see_v1","paused_bytes_limit":1048576,"preserve_paused_side_scope":true,"affinity":null},
            "value":{"value":"nnuec_v2","context_namespace_policy":"fixture-context","local_namespace_policy":"fixture-local"},
            "controller":{"fixed_node_credit":64,"fixed_soft_lease_us":1000,"initial_workers":1,"max_workers":1,"consecutive_observations":3,"worker_cooldown_ms":2000},
            "reciprocal":{"recipient":"critic","max_questions_per_job":1,"node_interval":1,"first_question_sequence":1},
            "peers":{"pc_peer":true,"researcher":true,"max_sources_per_root":1}
        });
        wire["own_r_root_tie"] = json!({"kind":"own","policy":"legacy_first_searched"});
        wire
    }

    fn external_wire() -> Value {
        let mut wire = pc_wire();
        wire["execution"] = json!({
            "assembly":"external_researcher", "event_order":"fixed",
            "queues":{"waiting_jobs_max":16,"worker_commands_max":8,"role_tickets_max":16,"paused_bytes_per_worker_max":1048576},
            "max_rounds":64,"max_reported_nodes":64,"cpu_depth":8,"cpu_nodes_per_task":128,
            "profile":{"path":"fixtures/external-profile.json","sha256":"a".repeat(64),"bytes":32,"source":"https://example.invalid/profile","license":"fixture"},
            "slice":{"quantum_ms":1000,"stop_drain_reserve_ms":500,"requested_reported_nodes":256,"multipv":1},
            "peers":{"pc_peer":true,"researcher":true,"max_sources_per_root":16}
        });
        wire["external"] =
            json!({"kind":"researcher","question_limit":{"kind":"nodes","reported_nodes":32}});
        wire
    }

    fn decode(wire: &Value) -> Result<StartupConfigurationDocument, BackendError> {
        StartupConfigurationDocument::from_json(&serde_json::to_vec(wire).unwrap())
    }

    #[test]
    fn explicit_policy_matches_assembly_and_rejects_legacy_lock() {
        let pc = decode(&pc_wire()).unwrap();
        assert!(matches!(pc, StartupConfigurationDocument::V4(_)));
        assert_eq!(pc.own_r_root_tie_policy().unwrap(), None);
        let external = decode(&external_wire()).unwrap();
        assert_eq!(external.own_r_root_tie_policy().unwrap(), None);
        assert_eq!(
            external.external_question_limit().unwrap(),
            Some(ExternalGoConstraint::Nodes { reported_nodes: 32 })
        );
        let lock = RoventExecutionManifest::from_json(
            &crate::rovent_bootstrap::tests::wire(ExecutionMode::CpuPc).to_string(),
        )
        .unwrap()
        .lock()
        .unwrap();
        assert!(pc.admit_against(&lock).is_err());
        for mut wire in [pc_wire(), external_wire()] {
            wire["own_r_root_tie"] = json!({"kind":"own","policy":"rules_order_confirmed"});
            assert!(decode(&wire).is_err());
        }
        let mut own = own_wire();
        assert_eq!(
            decode(&own).unwrap().own_r_root_tie_policy().unwrap(),
            Some(CpuRootTiePolicy::LegacyFirstSearched)
        );
        own["own_r_root_tie"] = json!({"kind":"not_applicable"});
        assert!(decode(&own).is_err());
    }

    #[test]
    fn v4_policy_is_required_strict_and_sealed_in_own_nnuec_conditions() {
        let legacy = decode(&own_wire()).unwrap();
        let mut next = own_wire();
        next["own_r_root_tie"] = json!({"kind":"own","policy":"rules_order_confirmed"});
        let selected = decode(&next).unwrap();
        assert_eq!(
            selected.own_r_root_tie_policy().unwrap(),
            Some(CpuRootTiePolicy::RulesOrderConfirmed)
        );
        assert_ne!(
            legacy.configuration_digest().unwrap(),
            selected.configuration_digest().unwrap()
        );
        assert_ne!(
            legacy.search_configuration_digest().unwrap(),
            selected.search_configuration_digest().unwrap()
        );
        assert_ne!(
            legacy.nnuec_conditions_digest().unwrap(),
            selected.nnuec_conditions_digest().unwrap()
        );
        for invalid in [
            json!(null),
            json!({"kind":"own"}),
            json!({"kind":"own","policy":"unknown"}),
            json!({"kind":"not_applicable","extra":true}),
        ] {
            let mut wire = pc_wire();
            wire["own_r_root_tie"] = invalid;
            assert!(decode(&wire).is_err());
        }
        let mut missing = pc_wire();
        missing.as_object_mut().unwrap().remove("own_r_root_tie");
        assert!(decode(&missing).is_err());
    }

    #[test]
    fn pc_shared_readout_remains_r_free_and_v3_wire_is_unchanged() {
        let mut pc = pc_wire();
        pc["execution"]["shared_nnuec_readout"] = json!({"context_namespace_policy":"fixture-context","local_namespace_policy":"fixture-local"});
        let v4 = decode(&pc).unwrap();
        assert_eq!(v4.own_r_root_tie_policy().unwrap(), None);
        assert!(v4.nnuec_conditions_digest().unwrap().is_some());
        let mut v3 = pc.clone();
        v3["domain"] = json!(super::super::v3::ROVENT_STARTUP_CONFIGURATION_V3_DOMAIN);
        v3.as_object_mut().unwrap().remove("own_r_root_tie");
        let bytes = serde_json::to_vec(&v3).unwrap();
        let original = RoventStartupConfigurationV3::from_json(&bytes).unwrap();
        let dispatched = StartupConfigurationDocument::from_json(&bytes).unwrap();
        assert_eq!(
            dispatched.canonical_bytes().unwrap(),
            original.canonical_bytes().unwrap()
        );
        assert_eq!(
            dispatched.configuration_digest().unwrap(),
            original.configuration_digest().unwrap()
        );
        assert_eq!(
            dispatched.search_configuration_digest().unwrap(),
            original.search_configuration_digest().unwrap()
        );
        assert_eq!(
            v4.nnuec_conditions_digest().unwrap(),
            original.nnuec_search_conditions_digest().unwrap()
        );
        assert!(RoventStartupConfigurationV4::from_json(&bytes).is_err());
    }
}
