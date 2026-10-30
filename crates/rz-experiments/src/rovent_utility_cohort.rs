//! Explicit CPU utility-comparison cohort metadata, separate from recipe /1.
//!
//! This closed declaration pins the requested experiment before its prelude.
//! It neither creates an execution lease nor records an available recipient,
//! ready/input/baseline freeze, submitted utility input or physical completion.
//! Only the original UCI owner can issue those opaque current capabilities.
//! In particular, a collection-candidate declaration is not a valid target.

use crate::{ManifestError, decode_json};
use rz_contracts::{Digest, rovent::ExecutionMode};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

pub const ROVENT_UTILITY_COMPARISON_COHORT_DOMAIN: &str = "rz-rovent-utility-comparison-cohort/1";
pub const ROVENT_UTILITY_INTERVENTION_COHORT_DOMAIN: &str = "rz-rovent-utility-comparison-cohort/2";
pub const ROVENT_UTILITY_INTERVENTION_TARGET_DOMAIN: &str =
    "rz-rovent-utility-intervention-target/1";
pub const ROVENT_UTILITY_INTERVENTION_TRIAL_DOMAIN: &str = "rz-rovent-utility-cohort-trial/2";
pub const ROVENT_UTILITY_INTERVENTION_PAIR_DOMAIN: &str = "rz-rovent-utility-cohort-pair/2";
pub const ROVENT_UTILITY_COMPARISON_COST_SCOPE: &str = "cohort_route_with_bounded_materialization_and_physical_pin_zero_excluding_durable_dataset_io_v1";
pub const MAX_ROVENT_UTILITY_COHORT_BYTES: usize = 64 << 10;
pub const MAX_ROVENT_UTILITY_COHORT_WINDOW_NS: u64 = 86_400_000_000_000;

/// This first opt-in profile is CPU only. It does not enable a GPU path or
/// alter the meaning of the existing four-mode execution declaration.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventUtilityCohortMode {
    CpuPcr,
    CpuPc,
}
impl RoventUtilityCohortMode {
    pub const fn execution_mode(self) -> ExecutionMode {
        match self {
            Self::CpuPcr => ExecutionMode::CpuPcr,
            Self::CpuPc => ExecutionMode::CpuPc,
        }
    }
}

/// Requested search recipients, not neural artifact roles or an observed
/// available subset. Their declared order is part of the canonical metadata.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub enum RoventUtilityCohortRecipient {
    Proposer,
    Critic,
    Researcher,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventUtilityCohortUsage {
    ValidationOnly,
    CollectionCandidate,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventUtilityComparisonCohortInput {
    /// Exact original startup lock canonical SHA, not a replacement lock.
    pub input_lock_sha256: String,
    /// Original declared aggregate source identity, separately checked by UCI.
    pub source_sha256: String,
    /// Original model identity and namespace pins retain their own domains;
    /// this codec never substitutes a raw weights-file SHA for a model SHA.
    pub frozen_nnuec_model_sha256: String,
    pub frozen_nnuec_namespace_sha256: String,
    pub frozen_pc_model_sha256: String,
    pub frozen_pc_namespace_sha256: String,
    pub mode: RoventUtilityCohortMode,
    pub requested_recipients: Vec<RoventUtilityCohortRecipient>,
    /// Relative registration ceilings. These are not actual clock readings,
    /// observed stage costs or renewed deadlines issued by deserialization.
    pub prelude_ns: u64,
    pub action_ns: u64,
    pub cleanup_ns: u64,
    pub logical_budget_sha256: String,
    /// Policies only: an actual ready event/cache observation is not encoded.
    pub ready_policy_sha256: String,
    pub cache_policy_sha256: String,
    pub cost_class_sha256: String,
    pub resource_plan_sha256: String,
    pub resource_epoch: u64,
    pub usage: RoventUtilityCohortUsage,
    pub cost_scope: String,
}
impl RoventUtilityComparisonCohortInput {
    /// A checked finite declared window, not permission to start that window.
    pub fn total_window_ns(&self) -> Result<u64, ManifestError> {
        require(
            self.prelude_ns > 0 && self.action_ns > 0 && self.cleanup_ns > 0,
            "prelude, action and cleanup durations must each be positive",
        )?;
        self.prelude_ns
            .checked_add(self.action_ns)
            .and_then(|duration| duration.checked_add(self.cleanup_ns))
            .filter(|duration| *duration <= MAX_ROVENT_UTILITY_COHORT_WINDOW_NS)
            .ok_or_else(|| integrity("declared duration sum overflows or exceeds 24 hours"))
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            [
                &self.input_lock_sha256,
                &self.source_sha256,
                &self.frozen_nnuec_model_sha256,
                &self.frozen_nnuec_namespace_sha256,
                &self.frozen_pc_model_sha256,
                &self.frozen_pc_namespace_sha256,
                &self.logical_budget_sha256,
                &self.ready_policy_sha256,
                &self.cache_policy_sha256,
                &self.cost_class_sha256,
                &self.resource_plan_sha256,
            ]
            .into_iter()
            .all(|value| sha(value))
                && self.resource_epoch > 0,
            "original source/model/namespace/policy/resource pins must be explicit",
        )?;
        require(
            self.cost_scope == ROVENT_UTILITY_COMPARISON_COST_SCOPE,
            "unsupported cohort cost scope",
        )?;
        require(
            (1..=3).contains(&self.requested_recipients.len())
                && self
                    .requested_recipients
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.requested_recipients.len(),
            "requested recipients must be a nonempty unique ordered list of at most three",
        )?;
        require(
            self.mode != RoventUtilityCohortMode::CpuPc
                || !self
                    .requested_recipients
                    .contains(&RoventUtilityCohortRecipient::Researcher),
            "CPU P-C mode cannot request a native R recipient",
        )?;
        self.total_window_ns()?;
        Ok(())
    }
}

/// Passive source-pinned metadata. Public construction, decoding or sealing
/// cannot observe live readiness, grant a trial, select an available subset or
/// turn raw utility observations into a known training label.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventUtilityComparisonCohort {
    pub domain: String,
    pub input: RoventUtilityComparisonCohortInput,
}
impl RoventUtilityComparisonCohort {
    pub fn seal(input: RoventUtilityComparisonCohortInput) -> Result<Self, ManifestError> {
        let cohort = Self {
            domain: ROVENT_UTILITY_COMPARISON_COHORT_DOMAIN.into(),
            input,
        };
        cohort.validate()?;
        Ok(cohort)
    }

    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        require(
            input.len() <= MAX_ROVENT_UTILITY_COHORT_BYTES,
            "cohort JSON exceeds 64 KiB",
        )?;
        // The existing bounded decoder rejects duplicate keys recursively.
        // Derived structs also accept positional sequences, so require both
        // object boundaries before the closed field/type checks below.
        let raw: serde_json::Value = decode_json(input)?;
        require(
            raw.is_object() && raw.get("input").is_some_and(serde_json::Value::is_object),
            "cohort root and input must both be JSON objects",
        )?;
        let cohort: Self =
            serde_json::from_value(raw).map_err(|error| ManifestError::Parse(error.to_string()))?;
        cohort.validate()?;
        Ok(cohort)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.domain == ROVENT_UTILITY_COMPARISON_COHORT_DOMAIN,
            "unsupported utility cohort domain",
        )?;
        self.input.validate()?;
        canonical(self)?;
        Ok(())
    }

    pub fn total_window_ns(&self) -> Result<u64, ManifestError> {
        self.validate()?;
        self.input.total_window_ns()
    }

    /// Compact UTF-8 JSON with recursively sorted object keys. Requested
    /// recipient order and exact u64 values are retained; no floats are used.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        canonical(self)
    }

    /// Raw SHA-256 of canonical bytes, distinct from the framed cohort digest.
    pub fn canonical_sha256(&self) -> Result<String, ManifestError> {
        Ok(crate::digest(&self.canonical_bytes()?))
    }

    /// Domain UTF-8, NUL, then the complete canonical metadata bytes.
    pub fn digest(&self) -> Result<Digest, ManifestError> {
        let mut hash = Sha256::new();
        hash.update(ROVENT_UTILITY_COMPARISON_COHORT_DOMAIN.as_bytes());
        hash.update([0]);
        hash.update(self.canonical_bytes()?);
        Ok(Digest(hash.finalize().into()))
    }
}

/// An explicit experimental intervention, never an assertion that the normal
/// scheduler already offers these alternatives at the same instant.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventUtilityInterventionPolicyV2 {
    /// Extend an actual proposal with P, request C divergence where supported,
    /// or validate that actual proposal with installed own R. Repair requires
    /// separate original counter evidence and is not synthesized by this policy.
    FrozenProposalReviewV1,
}

/// Passive /2 registration. The original neutral target and NNUEC input are
/// captured from each fresh root after the same bounded prelude. A role-specific
/// query/view has its own seal and must not be relabeled the common input.
/// Decoding these declarations cannot issue a target, frame or trial capability.
#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventUtilityInterventionCohortV2 {
    pub domain: String,
    pub input: RoventUtilityComparisonCohortInput,
    pub target_record: u16,
    pub policy: RoventUtilityInterventionPolicyV2,
}
impl RoventUtilityInterventionCohortV2 {
    pub fn seal(
        input: RoventUtilityComparisonCohortInput,
        target_record: u16,
    ) -> Result<Self, ManifestError> {
        let value = Self {
            domain: ROVENT_UTILITY_INTERVENTION_COHORT_DOMAIN.into(),
            input,
            target_record,
            policy: RoventUtilityInterventionPolicyV2::FrozenProposalReviewV1,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn from_json(input: &str) -> Result<Self, ManifestError> {
        require(
            input.len() <= MAX_ROVENT_UTILITY_COHORT_BYTES,
            "intervention JSON exceeds 64 KiB",
        )?;
        let raw: serde_json::Value = decode_json(input)?;
        require(
            raw.is_object() && raw.get("input").is_some_and(serde_json::Value::is_object),
            "intervention root and input must be objects",
        )?;
        let value: Self =
            serde_json::from_value(raw).map_err(|error| ManifestError::Parse(error.to_string()))?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        require(
            self.domain == ROVENT_UTILITY_INTERVENTION_COHORT_DOMAIN,
            "unsupported intervention cohort domain",
        )?;
        // The declared source, window and whole-route cost boundaries retain
        // their exact meanings. Execution policy and evidence have a new domain.
        self.input.validate()?;
        require(
            self.input.requested_recipients.len() >= 2,
            "an intervention comparison requires at least two distinct declared arms",
        )?;
        require(
            self.target_record < 64,
            "intervention target exceeds the selected-view bound",
        )?;
        self.canonical_unchecked()?;
        Ok(())
    }

    pub fn total_window_ns(&self) -> Result<u64, ManifestError> {
        self.validate()?;
        self.input.total_window_ns()
    }

    fn canonical_unchecked(&self) -> Result<Vec<u8>, ManifestError> {
        let mut value =
            serde_json::to_value(self).map_err(|error| ManifestError::Parse(error.to_string()))?;
        value.sort_all_objects();
        let bytes =
            serde_json::to_vec(&value).map_err(|error| ManifestError::Parse(error.to_string()))?;
        require(
            bytes.len() <= MAX_ROVENT_UTILITY_COHORT_BYTES,
            "canonical intervention JSON exceeds 64 KiB",
        )?;
        Ok(bytes)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ManifestError> {
        self.validate()?;
        self.canonical_unchecked()
    }

    pub fn canonical_sha256(&self) -> Result<String, ManifestError> {
        Ok(crate::digest(&self.canonical_bytes()?))
    }

    pub fn digest(&self) -> Result<Digest, ManifestError> {
        let mut hash = Sha256::new();
        hash.update(ROVENT_UTILITY_INTERVENTION_COHORT_DOMAIN.as_bytes());
        hash.update([0]);
        hash.update(self.canonical_bytes()?);
        Ok(Digest(hash.finalize().into()))
    }
}

fn canonical(cohort: &RoventUtilityComparisonCohort) -> Result<Vec<u8>, ManifestError> {
    // validate() bounds all strings and the sole collection before this
    // conversion. This helper never calls validate(), avoiding recursion.
    let mut value =
        serde_json::to_value(cohort).map_err(|error| ManifestError::Parse(error.to_string()))?;
    value.sort_all_objects();
    let bytes =
        serde_json::to_vec(&value).map_err(|error| ManifestError::Parse(error.to_string()))?;
    require(
        bytes.len() <= MAX_ROVENT_UTILITY_COHORT_BYTES,
        "canonical cohort JSON exceeds 64 KiB",
    )?;
    Ok(bytes)
}

fn sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
fn require(valid: bool, message: &str) -> Result<(), ManifestError> {
    if valid {
        Ok(())
    } else {
        Err(integrity(message))
    }
}
fn integrity(message: &str) -> ManifestError {
    ManifestError::Integrity(format!("Rovent utility cohort: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn cohort() -> RoventUtilityComparisonCohort {
        RoventUtilityComparisonCohort::seal(RoventUtilityComparisonCohortInput {
            input_lock_sha256: "1".repeat(64),
            source_sha256: "2".repeat(64),
            frozen_nnuec_model_sha256: "3".repeat(64),
            frozen_nnuec_namespace_sha256: "4".repeat(64),
            frozen_pc_model_sha256: "5".repeat(64),
            frozen_pc_namespace_sha256: "6".repeat(64),
            mode: RoventUtilityCohortMode::CpuPcr,
            requested_recipients: vec![RoventUtilityCohortRecipient::Proposer],
            prelude_ns: 1,
            action_ns: 2,
            cleanup_ns: 3,
            logical_budget_sha256: "7".repeat(64),
            ready_policy_sha256: "8".repeat(64),
            cache_policy_sha256: "9".repeat(64),
            cost_class_sha256: "a".repeat(64),
            resource_plan_sha256: "b".repeat(64),
            resource_epoch: 1,
            usage: RoventUtilityCohortUsage::ValidationOnly,
            cost_scope: ROVENT_UTILITY_COMPARISON_COST_SCOPE.into(),
        })
        .unwrap()
    }

    #[test]
    fn intervention_is_explicit_and_never_decodes_as_a_ready_cohort() {
        let mut input = cohort().input;
        input
            .requested_recipients
            .push(RoventUtilityCohortRecipient::Critic);
        let value = RoventUtilityInterventionCohortV2::seal(input, 0).unwrap();
        let raw = String::from_utf8(value.canonical_bytes().unwrap()).unwrap();
        assert_eq!(
            RoventUtilityInterventionCohortV2::from_json(&raw).unwrap(),
            value
        );
        assert!(RoventUtilityComparisonCohort::from_json(&raw).is_err());
        assert!(
            RoventUtilityInterventionCohortV2::from_json(
                &serde_json::to_string(&cohort()).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn intervention_requires_comparable_declared_arms_and_bounded_target() {
        let mut input = cohort().input;
        assert!(RoventUtilityInterventionCohortV2::seal(input.clone(), 0).is_err());
        input
            .requested_recipients
            .push(RoventUtilityCohortRecipient::Critic);
        let value = RoventUtilityInterventionCohortV2::seal(input.clone(), 63).unwrap();
        assert!(RoventUtilityInterventionCohortV2::seal(input, 64).is_err());
        let mut raw = serde_json::to_value(value).unwrap();
        raw["policy"] = "pretend_existing_ready".into();
        assert!(RoventUtilityInterventionCohortV2::from_json(&raw.to_string()).is_err());
    }

    #[test]
    fn intervention_digest_binds_target_and_requested_order() {
        let mut input = cohort().input;
        input
            .requested_recipients
            .push(RoventUtilityCohortRecipient::Critic);
        let value = RoventUtilityInterventionCohortV2::seal(input, 0).unwrap();
        let mut changed = value.clone();
        changed.target_record = 1;
        assert_ne!(changed.digest().unwrap(), value.digest().unwrap());
        changed = value.clone();
        changed.input.requested_recipients.reverse();
        assert_ne!(changed.digest().unwrap(), value.digest().unwrap());
    }

    #[test]
    fn positional_root_array_is_refused_even_with_valid_field_values() {
        let wire = serde_json::to_value(cohort()).unwrap();
        let positional = json!([wire["domain"], wire["input"]]);
        let raw = serde_json::to_string(&positional).unwrap();
        assert!(RoventUtilityComparisonCohort::from_json(&raw).is_err());
    }

    #[test]
    fn positional_input_array_is_refused_even_with_valid_field_values() {
        let mut wire = serde_json::to_value(cohort()).unwrap();
        let fields = [
            "input_lock_sha256",
            "source_sha256",
            "frozen_nnuec_model_sha256",
            "frozen_nnuec_namespace_sha256",
            "frozen_pc_model_sha256",
            "frozen_pc_namespace_sha256",
            "mode",
            "requested_recipients",
            "prelude_ns",
            "action_ns",
            "cleanup_ns",
            "logical_budget_sha256",
            "ready_policy_sha256",
            "cache_policy_sha256",
            "cost_class_sha256",
            "resource_plan_sha256",
            "resource_epoch",
            "usage",
            "cost_scope",
        ];
        let positional_input = Value::Array(
            fields
                .into_iter()
                .map(|field| wire["input"][field].clone())
                .collect(),
        );
        wire["input"] = positional_input;
        let raw = serde_json::to_string(&wire).unwrap();
        assert!(RoventUtilityComparisonCohort::from_json(&raw).is_err());
    }
}
