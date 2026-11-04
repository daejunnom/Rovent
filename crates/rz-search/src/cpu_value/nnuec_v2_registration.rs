//! Actual compiled V2 capabilities for strict startup pin comparison.
//!
//! This is a read-only descriptor, not a model/input identity replacement or
//! training/arena acceptance. It never loads an artifact, runs a forward or
//! verifies the caller's provenance file. The caller must separately verify
//! its original retained artifact bytes and declared provenance.

use super::{INPUT_SCHEMA, Model, SCHEMA, SEMANTICS};
use crate::cpu_value::CpuValueIdentity;
use crate::cpu_value::nnuec::NnuecError;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;

pub const CONTEXT_NAMESPACE_POLICY: &str = "exact-original-review-view:anchor-rules-history:kind4-parent-supersedes-origin-first:immutable-model:cancel-deadline-epoch:typed-owner-v2";
pub const LOCAL_NAMESPACE_POLICY: &str = "exact-begin-rules-history:ordered-checked-move16:append-undo:fixed-context-authority:same-owner-resume-v2";
pub const REGISTRATION_SCHEMA: &str = "rz-nnuec-v2-registration/1";
pub const CONFIGURATION_DOMAIN: &str = "rz-nnuec-v2-runtime-configuration/1";
const IMPLEMENTATION_DOMAIN: &[u8] = b"rz-nnuec-v2-implementation/1\0";
const CONTEXT_POLICY_DOMAIN: &str = "rz-rovent-nnuec-context-policy/1";
const LOCAL_POLICY_DOMAIN: &str = "rz-rovent-nnuec-local-policy/1";

/// Only the actual loaded model can construct this descriptor. Serializable
/// metadata does not grant a live Bound, worker, task or semantic acceptance.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct NnuecV2Registration {
    domain: &'static str,
    schema: &'static str,
    semantics: &'static str,
    encoding: &'static str,
    implementation_sha256: String,
    configuration_sha256: String,
    context_namespace_policy: &'static str,
    local_namespace_policy: &'static str,
    context_namespace_policy_sha256: String,
    local_namespace_policy_sha256: String,
    model_identity: CpuValueIdentity,
}
impl NnuecV2Registration {
    pub fn schema(&self) -> &str {
        self.schema
    }
    pub fn semantics(&self) -> &str {
        self.semantics
    }
    pub fn encoding(&self) -> &str {
        self.encoding
    }
    pub fn implementation_sha256(&self) -> &str {
        &self.implementation_sha256
    }
    pub fn configuration_sha256(&self) -> &str {
        &self.configuration_sha256
    }
    pub fn context_namespace_policy(&self) -> &str {
        self.context_namespace_policy
    }
    pub fn local_namespace_policy(&self) -> &str {
        self.local_namespace_policy
    }
    pub fn context_namespace_policy_sha256(&self) -> &str {
        &self.context_namespace_policy_sha256
    }
    pub fn local_namespace_policy_sha256(&self) -> &str {
        &self.local_namespace_policy_sha256
    }
    pub fn model_identity(&self) -> &CpuValueIdentity {
        &self.model_identity
    }
}

impl Model {
    /// Returns compiled capabilities and the same immutable model identity.
    /// The weights hash remains the established V2 semantic weights hash;
    /// it is deliberately distinct from the caller's raw file-byte hash.
    pub fn registration(&self) -> Result<NnuecV2Registration, NnuecError> {
        self.identity()
            .validate()
            .map_err(|_| NnuecError::StateMismatch)?;
        let base = self.base_checkpoint();
        #[derive(Serialize)]
        struct Configuration<'a> {
            domain: &'static str,
            schema: &'static str,
            semantics: &'static str,
            encoding: &'static str,
            dimensions: &'a super::super::NnuecConfig,
            output_scale_fp32_bits: u32,
            precision: &'static str,
            evaluation_order: &'static str,
            embedding_layout: &'static str,
            dense_layout: &'static str,
            board_accumulation: &'static str,
            context_accumulation: &'static str,
            parameter_absolute_bound_fp32_bits: u32,
            output_scale_maximum_fp32_bits: u32,
            v2_operand_absolute_power_of_two_bound: u8,
            v2_sum_absolute_power_of_two_bound: u8,
            checkpoint_maximum_bytes: usize,
            base_parameter_allocation_maximum_bytes: usize,
            parameter_capacity_multiple: usize,
            parameter_capacity_minimum_allowance: usize,
            model_tag_capacity_maximum_bytes: usize,
            board_feature_count: usize,
            maximum_embedding_width: usize,
            maximum_hidden_width: usize,
            supported_causal_windows: [usize; 2],
            maximum_context_lines: usize,
            maximum_edges_per_context_line: usize,
            maximum_line_plies: usize,
            output_rows: usize,
            value_authority: &'static str,
        }
        let declaration = Configuration {
            domain: CONFIGURATION_DOMAIN,
            schema: SCHEMA,
            semantics: SEMANTICS,
            encoding: INPUT_SCHEMA,
            dimensions: &base.config,
            output_scale_fp32_bits: base.output_scale.to_bits(),
            precision: "fp32",
            evaluation_order: "scalar-multiply-then-add-round-nearest-even-no-fma",
            embedding_layout: "feature-row-major",
            dense_layout: "output-row-major",
            board_accumulation: "signed192-fp32-operands-unit-2^-149-round-once",
            context_accumulation: "checked-signed256-fp32-contributions-unit-2^-149-round-once",
            parameter_absolute_bound_fp32_bits: super::super::PARAMETER_MAGNITUDE.to_bits(),
            output_scale_maximum_fp32_bits: 10_000.0_f32.to_bits(),
            v2_operand_absolute_power_of_two_bound: 70,
            v2_sum_absolute_power_of_two_bound: 79,
            checkpoint_maximum_bytes: super::super::NNUEC_MAX_BYTES,
            base_parameter_allocation_maximum_bytes: super::super::NNUEC_MAX_BYTES,
            parameter_capacity_multiple: 2,
            parameter_capacity_minimum_allowance: 4,
            model_tag_capacity_maximum_bytes: 256,
            board_feature_count: 800,
            maximum_embedding_width: 256,
            maximum_hidden_width: 64,
            supported_causal_windows: [3, 5],
            maximum_context_lines: 128,
            maximum_edges_per_context_line: 2,
            maximum_line_plies: 256,
            output_rows: super::super::NNUEC_OUTPUTS,
            value_authority: "nonterminal-frontier-clipped-20000-no-rules-proof",
        };
        Ok(NnuecV2Registration {
            domain: REGISTRATION_SCHEMA,
            schema: SCHEMA,
            semantics: SEMANTICS,
            encoding: INPUT_SCHEMA,
            implementation_sha256: implementation_sha256(),
            configuration_sha256: framed_hash(CONFIGURATION_DOMAIN, &canonical(&declaration)?),
            context_namespace_policy: CONTEXT_NAMESPACE_POLICY,
            local_namespace_policy: LOCAL_NAMESPACE_POLICY,
            context_namespace_policy_sha256: framed_hash(
                CONTEXT_POLICY_DOMAIN,
                &canonical(&CONTEXT_NAMESPACE_POLICY)?,
            ),
            local_namespace_policy_sha256: framed_hash(
                LOCAL_POLICY_DOMAIN,
                &canonical(&LOCAL_NAMESPACE_POLICY)?,
            ),
            model_identity: self.identity().clone(),
        })
    }
}

fn implementation_sha256() -> String {
    let mut hash = Sha256::new();
    hash.update(IMPLEMENTATION_DOMAIN);
    for (name, source) in [
        ("cpu_value.rs", include_bytes!("../cpu_value.rs").as_slice()),
        ("nnuec.rs", include_bytes!("nnuec.rs").as_slice()),
        (
            "nnuec_rules.rs",
            include_bytes!("nnuec_rules.rs").as_slice(),
        ),
        ("nnuec_v2.rs", include_bytes!("nnuec_v2.rs").as_slice()),
        (
            "nnuec_v2_cpu.rs",
            include_bytes!("nnuec_v2_cpu.rs").as_slice(),
        ),
        (
            "nnuec_v2_review.rs",
            include_bytes!("nnuec_v2_review.rs").as_slice(),
        ),
        (
            "nnuec_v2_shared_readout.rs",
            include_bytes!("nnuec_v2_shared_readout.rs").as_slice(),
        ),
        (
            "nnuec_v2_review_retirement.rs",
            include_bytes!("nnuec_v2_review_retirement.rs").as_slice(),
        ),
        (
            "nnuec_v2_preparation_terminal.rs",
            include_bytes!("nnuec_v2_preparation_terminal.rs").as_slice(),
        ),
        (
            "nnuec_v2_capture.rs",
            include_bytes!("nnuec_v2_capture.rs").as_slice(),
        ),
        (
            "nnuec_v2_capture_lineage.rs",
            include_bytes!("nnuec_v2_capture_lineage.rs").as_slice(),
        ),
        (
            "nnuec_v2_utility_capture.rs",
            include_bytes!("nnuec_v2_utility_capture.rs").as_slice(),
        ),
        (
            "nnuec_v2_retirement.rs",
            include_bytes!("nnuec_v2_retirement.rs").as_slice(),
        ),
        (
            "cpu_worker/prepared_value.rs",
            include_bytes!("../cpu_worker/prepared_value.rs").as_slice(),
        ),
        (
            "cpu_worker/deferred_value.rs",
            include_bytes!("../cpu_worker/deferred_value.rs").as_slice(),
        ),
        (
            "cpu_worker/retired_value.rs",
            include_bytes!("../cpu_worker/retired_value.rs").as_slice(),
        ),
        (
            "cpu_worker/unstarted_value.rs",
            include_bytes!("../cpu_worker/unstarted_value.rs").as_slice(),
        ),
        (
            "nnuec_v2_registration.rs",
            include_bytes!("nnuec_v2_registration.rs").as_slice(),
        ),
    ] {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((source.len() as u64).to_le_bytes());
        hash.update(source);
    }
    format!("{:x}", hash.finalize())
}

fn framed_hash(domain: &str, bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(domain.as_bytes());
    hash.update([0]);
    hash.update(bytes);
    format!("{:x}", hash.finalize())
}
fn canonical(value: &impl Serialize) -> Result<Vec<u8>, NnuecError> {
    fn ordered(value: Value) -> Value {
        match value {
            Value::Object(values) => Value::Object(
                values
                    .into_iter()
                    .map(|(key, value)| (key, ordered(value)))
                    .collect::<BTreeMap<_, _>>()
                    .into_iter()
                    .collect(),
            ),
            Value::Array(values) => Value::Array(values.into_iter().map(ordered).collect()),
            other => other,
        }
    }
    serde_json::to_value(value)
        .and_then(|v| serde_json::to_vec(&ordered(v)))
        .map_err(|e| NnuecError::Json(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_reads_actual_identity_and_keeps_weights_only_change_separate() {
        let original = Model::new(super::super::tests::checkpoint()).unwrap();
        let mut changed = super::super::tests::checkpoint();
        changed.base.parameters.board_bias[0] = 0.5;
        let changed = Model::new(changed).unwrap();
        let a = original.registration().unwrap();
        let b = changed.registration().unwrap();
        assert_eq!(a.schema(), SCHEMA);
        assert_eq!(a.semantics(), SEMANTICS);
        assert_eq!(a.encoding(), INPUT_SCHEMA);
        assert_eq!(a.model_identity(), original.identity());
        assert_ne!(a.model_identity(), b.model_identity());
        assert_eq!(a.configuration_sha256(), b.configuration_sha256());
        assert_eq!(a.implementation_sha256(), b.implementation_sha256());
        assert_eq!(
            a.context_namespace_policy_sha256(),
            b.context_namespace_policy_sha256()
        );
        assert_eq!(
            a.local_namespace_policy_sha256(),
            b.local_namespace_policy_sha256()
        );
    }

    #[test]
    fn actual_config_and_scale_changes_have_distinct_registration_configuration() {
        let original = Model::new(super::super::tests::checkpoint())
            .unwrap()
            .registration()
            .unwrap();
        let mut dimensions = super::super::tests::checkpoint();
        dimensions.base.config.max_plies = 8;
        dimensions.base.parameters.ply_weights.truncate(8);
        let dimensions = Model::new(dimensions).unwrap().registration().unwrap();
        let mut scale = super::super::tests::checkpoint();
        scale.base.output_scale = 2.0;
        let scale = Model::new(scale).unwrap().registration().unwrap();
        assert_ne!(
            original.configuration_sha256(),
            dimensions.configuration_sha256()
        );
        assert_ne!(
            original.configuration_sha256(),
            scale.configuration_sha256()
        );
    }

    #[test]
    fn policy_hashes_use_existing_startup_json_string_framing() {
        let value = Model::new(super::super::tests::checkpoint())
            .unwrap()
            .registration()
            .unwrap();
        let bytes = serde_json::to_vec(CONTEXT_NAMESPACE_POLICY).unwrap();
        assert_eq!(
            value.context_namespace_policy_sha256(),
            framed_hash(CONTEXT_POLICY_DOMAIN, &bytes)
        );
        let bytes = serde_json::to_vec(LOCAL_NAMESPACE_POLICY).unwrap();
        assert_eq!(
            value.local_namespace_policy_sha256(),
            framed_hash(LOCAL_POLICY_DOMAIN, &bytes)
        );
        assert_ne!(
            value.context_namespace_policy_sha256(),
            value.local_namespace_policy_sha256()
        );
    }
}
