//! Explicit intervention source-family execution declarations.
//! These metadata declarations preserve original pins but grant no execution authority.
#![cfg(target_os = "linux")]

use super::execution::{
    MAX_BINARY_BYTES, MAX_METADATA_BYTES, ROVENT_UTILITY_MAX_READ_BYTES,
    RoventUtilityDatasetReference, RoventUtilityExecutionAction, RoventUtilityFrozenPcSource,
    RoventUtilityInitializationSource, RoventUtilityRequiredNullable, canonical_python_version,
    deserialize_endpoints, deserialize_object, execution_identifier, normalized_absolute,
    pin_shape,
};
use super::execution_v2::RoventUtilityAncestorDatasetRegistration;
use super::intervention_window::RoventUtilityInterventionTrainWindow;
use super::registration_cohort::RoventUtilityCohortSourceRegistration;
use crate::ArenaError;
use crate::rovent_training_process::recovery::{
    RoventTrainerRuntimeEndpoint, rovent_trainer_dependency_set_sha256,
};
use crate::rovent_training_process::{
    MAX_TRAINER_OUTPUT_BYTES, RoventPcRecordFamily, RoventTrainingFilePin,
    RoventTrainingResourcePolicy,
};
use rz_experiments::pals_data_v3::{generation_digest, generation_sha};
use serde::de::{MapAccess, Visitor, value::MapAccessDeserializer};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub const ROVENT_UTILITY_INTERVENTION_EXECUTION_REGISTRATION: &str =
    "rz-rovent-utility-training-registration/3";
fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Integrity(format!("Rovent intervention execution: {}", reason.into()))
}
fn check(condition: bool, reason: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}

/// Each arm owns its original declaration. The legacy codec is neither widened
/// nor recreated from a new-family Value. Only this new decoder knows the
/// explicit utility_intervention source kind.
#[derive(Clone, Debug, Eq, PartialEq)]
// At most 256 original declarations, never a worker message. Keeping the
// bounded original PC/R enum inline preserves its existing ownership layout.
#[allow(clippy::large_enum_variant)]
pub enum RoventUtilityInterventionAncestorRegistration {
    Legacy(RoventUtilityAncestorDatasetRegistration),
    Intervention {
        registration: RoventUtilityDatasetReference,
        preparation: RoventTrainingFilePin,
    },
}
impl Serialize for RoventUtilityInterventionAncestorRegistration {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Legacy(value) => value.serialize(serializer),
            Self::Intervention {
                registration,
                preparation,
            } => {
                #[derive(Serialize)]
                struct Fields<'a> {
                    kind: &'static str,
                    registration: &'a RoventUtilityDatasetReference,
                    preparation: &'a RoventTrainingFilePin,
                }
                Fields {
                    kind: "utility_intervention",
                    registration,
                    preparation,
                }
                .serialize(serializer)
            }
        }
    }
}
impl<'de> Deserialize<'de> for RoventUtilityInterventionAncestorRegistration {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(tag = "kind", deny_unknown_fields)]
        enum New {
            #[serde(rename = "utility_intervention")]
            Intervention {
                registration: RoventUtilityDatasetReference,
                #[serde(deserialize_with = "deserialize_object")]
                preparation: RoventTrainingFilePin,
            },
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        // This temporary is consumed directly into the same bounded layout.
        #[allow(clippy::large_enum_variant)]
        enum Either {
            New(New),
            Legacy(RoventUtilityAncestorDatasetRegistration),
        }
        // The serde content buffer retains map entries, including duplicates;
        // every selected original decoder still rejects unknown/duplicate keys.
        Ok(match Either::deserialize(deserializer)? {
            Either::New(New::Intervention {
                registration,
                preparation,
            }) => Self::Intervention {
                registration,
                preparation,
            },
            Either::Legacy(value) => Self::Legacy(value),
        })
    }
}
impl RoventUtilityInterventionAncestorRegistration {
    pub fn legacy(&self) -> Option<&RoventUtilityAncestorDatasetRegistration> {
        match self {
            Self::Legacy(value) => Some(value),
            _ => None,
        }
    }
    pub fn intervention(&self) -> Option<(&RoventUtilityDatasetReference, &RoventTrainingFilePin)> {
        match self {
            Self::Intervention {
                registration,
                preparation,
            } => Some((registration, preparation)),
            _ => None,
        }
    }
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Legacy(value) => value.kind(),
            _ => "utility_intervention",
        }
    }
    pub fn publication(&self) -> &RoventTrainingFilePin {
        match self {
            Self::Legacy(value) => value.publication(),
            Self::Intervention { registration, .. } => &registration.publication,
        }
    }
    pub fn root(&self) -> &Path {
        match self {
            Self::Legacy(value) => value.root(),
            Self::Intervention { registration, .. } => &registration.root,
        }
    }
    pub fn pc_family(&self) -> Option<RoventPcRecordFamily> {
        self.legacy()
            .and_then(RoventUtilityAncestorDatasetRegistration::pc_family)
    }
    pub fn validate(&self) -> Result<(), ArenaError> {
        match self {
            Self::Legacy(value) => value.validate(),
            Self::Intervention {
                registration,
                preparation,
            } => {
                registration.validate()?;
                pin_shape(preparation, MAX_METADATA_BYTES)
            }
        }
    }
}
utility_object! {
/// Explicit 30-field intervention roster with separately typed source custody. No
/// decoder creates an older request, changes an ancestor's source namespace, or
/// replaces an unverified ancestor inventory with an empty one.
pub struct RoventUtilityInterventionExecutionRegistration {
    pub schema: String,
    pub data_family: String,
    pub execution_id: String,
    pub action: RoventUtilityExecutionAction,
    pub export_request: RoventUtilityRequiredNullable<RoventTrainingFilePin>,
    #[serde(deserialize_with = "deserialize_object")]
    pub preparation: RoventTrainingFilePin,
    #[serde(deserialize_with = "deserialize_object")]
    pub window_spec: RoventTrainingFilePin,
    pub trainer_source_sha256: String,
    #[serde(deserialize_with = "deserialize_object")]
    pub trainer_python_executable: RoventTrainingFilePin,
    pub trainer_python_version: String,
    pub dependency_directories: Vec<PathBuf>,
    pub trainer_dependency_set_sha256: String,
    #[serde(deserialize_with = "deserialize_endpoints")]
    pub runtime_endpoints: Vec<RoventTrainerRuntimeEndpoint>,
    pub current_dataset: RoventUtilityDatasetReference,
    pub ancestor_datasets: Vec<RoventUtilityInterventionAncestorRegistration>,
    pub initialization_source: RoventUtilityInitializationSource,
    #[serde(deserialize_with = "deserialize_object")]
    pub frozen_nnuec_base: RoventTrainingFilePin,
    pub frozen_pc_sources: Vec<RoventUtilityFrozenPcSource>,
    #[serde(deserialize_with = "deserialize_object")]
    pub resources_policy: RoventTrainingResourcePolicy,
    pub require_total_quota: bool,
    pub output_root: PathBuf,
    pub control_root: PathBuf,
    pub checkpoint_directory: PathBuf,
    pub receipt_path: PathBuf,
    pub max_rpc_messages: u64,
    pub max_rpc_bytes: u64,
    pub max_read_bytes: u64,
    pub max_child_read_bytes: u64,
    pub max_output_bytes: u64,
    pub max_artifact_files: usize,
}
}
impl RoventUtilityInterventionExecutionRegistration {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_UTILITY_INTERVENTION_EXECUTION_REGISTRATION
                && self.data_family == "intervention_v2"
                && execution_identifier(&self.execution_id)
                && generation_sha(&self.trainer_source_sha256)
                && canonical_python_version(&self.trainer_python_version)
                && !self.dependency_directories.is_empty()
                && self.dependency_directories.len() <= 8
                && self
                    .dependency_directories
                    .iter()
                    .all(|path| normalized_absolute(path))
                && self
                    .dependency_directories
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.dependency_directories.len()
                && self.trainer_dependency_set_sha256
                    == rovent_trainer_dependency_set_sha256(&self.runtime_endpoints)?,
            "explicit intervention_v2 source/runtime/dependency declarations required",
        )?;
        for endpoint in &self.runtime_endpoints {
            pin_shape(&endpoint.artifact, MAX_BINARY_BYTES)?;
        }
        for pin in [&self.preparation, &self.window_spec] {
            pin_shape(pin, MAX_METADATA_BYTES)?;
        }
        for pin in [&self.trainer_python_executable, &self.frozen_nnuec_base] {
            pin_shape(pin, MAX_BINARY_BYTES)?;
        }
        self.resources_policy.validate()?;
        self.current_dataset.validate()?;
        check(
            self.ancestor_datasets.len() <= 256,
            "bounded original mixed ancestor publication inventory",
        )?;
        let mut publications = BTreeSet::from([self.current_dataset.publication.sha256.as_str()]);
        for ancestor in &self.ancestor_datasets {
            ancestor.validate()?;
            check(
                publications.insert(ancestor.publication().sha256.as_str()),
                "duplicate current/ancestor original publication across families",
            )?;
        }
        self.initialization_source.validate()?;
        match self.action {
            RoventUtilityExecutionAction::Train => check(
                self.export_request.is_none(),
                "Train must carry explicit null export_request",
            )?,
            RoventUtilityExecutionAction::Export => {
                check(
                    matches!(
                        self.initialization_source,
                        RoventUtilityInitializationSource::Resume { .. }
                    ),
                    "utility Export requires an explicit Resume checkpoint",
                )?;
                pin_shape(
                    self.export_request
                        .as_ref()
                        .ok_or_else(|| invalid("Export requires an explicit export_request"))?,
                    MAX_METADATA_BYTES,
                )?;
            }
        }
        check(
            self.frozen_pc_sources.len() == 2,
            "exact frozen proposer and critic declarations required",
        )?;
        let mut roles = BTreeSet::new();
        for source in &self.frozen_pc_sources {
            pin_shape(&source.artifact, MAX_BINARY_BYTES)?;
            check(
                matches!(source.role.as_str(), "proposer" | "critic")
                    && roles.insert(source.role.as_str())
                    && [
                        &source.model_semantics_sha256,
                        &source.model_identity_sha256,
                        &source.encoding_sha256,
                    ]
                    .into_iter()
                    .all(|sha| generation_sha(sha)),
                "frozen P/C role/model/encoding declarations required",
            )?;
        }
        check(
            [
                &self.output_root,
                &self.control_root,
                &self.checkpoint_directory,
            ]
            .into_iter()
            .all(|path| normalized_absolute(path))
                && self.checkpoint_directory.starts_with(&self.output_root)
                && self.checkpoint_directory != self.output_root
                && normalized_absolute(&self.receipt_path)
                && self.receipt_path.parent() == Some(self.output_root.as_path())
                && self
                    .receipt_path
                    .extension()
                    .is_some_and(|extension| extension == "json")
                && self.receipt_path != self.checkpoint_directory
                && !self.output_root.starts_with(&self.control_root)
                && !self.control_root.starts_with(&self.output_root),
            "distinct owned output/control roots, child checkpoint directory and direct JSON receipt required",
        )?;
        check(
            (1..=2_000_000).contains(&self.max_rpc_messages)
                && (1..=MAX_TRAINER_OUTPUT_BYTES).contains(&self.max_rpc_bytes)
                && (1..=ROVENT_UTILITY_MAX_READ_BYTES).contains(&self.max_read_bytes)
                && self.max_child_read_bytes > 0
                && self.max_child_read_bytes <= self.max_read_bytes
                && (1..=MAX_TRAINER_OUTPUT_BYTES).contains(&self.max_output_bytes)
                && (1..=32_768).contains(&self.max_artifact_files),
            "finite shared utility RPC/read/output/file allowances required",
        )
    }
    pub fn canonical_sha256(&self) -> Result<String, ArenaError> {
        self.validate()?;
        generation_digest(ROVENT_UTILITY_INTERVENTION_EXECUTION_REGISTRATION, self)
            .map_err(ArenaError::Manifest)
    }
    /// Actual-runtime metadata entry accepts only exact9 intervention preparation. Original files,
    /// full mixed ancestry and Rules/leakage ownership remain separate checks.
    pub fn join_cohort_window(
        &self,
        window: &RoventUtilityInterventionTrainWindow,
        preparation: &Value,
    ) -> Result<(), ArenaError> {
        self.validate()?;
        window.join_cohort_preparation(preparation)?;
        let source: RoventUtilityCohortSourceRegistration =
            serde_json::from_value(preparation["registration"].clone())
                .map_err(|error| invalid(error.to_string()))?;
        check(
            source.trainer_source_sha256 == self.trainer_source_sha256
                && window.device == self.resources_policy.device
                && window.max_read_bytes == self.max_child_read_bytes
                && window.max_output_bytes <= self.max_output_bytes,
            "cohort window changed original trainer/resource/read/output allowances",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pin(path: &str) -> Value {
        json!({"path":path,"sha256":"1".repeat(64),"bytes":128})
    }
    fn registration() -> Value {
        let endpoints = vec![
            RoventTrainerRuntimeEndpoint {
                name: "torch.__file__".into(),
                artifact: serde_json::from_value(pin("/deps/torch/__init__.py")).unwrap(),
            },
            RoventTrainerRuntimeEndpoint {
                name: "torch._C.__file__".into(),
                artifact: serde_json::from_value(pin("/deps/torch/native.so")).unwrap(),
            },
        ];
        json!({"schema":ROVENT_UTILITY_INTERVENTION_EXECUTION_REGISTRATION,
            "data_family":"intervention_v2","execution_id":"intervention-test","action":"train","export_request":null,
            "preparation":pin("/input/preparation.json"),"window_spec":pin("/input/window.json"),
            "trainer_source_sha256":"2".repeat(64),"trainer_python_executable":pin("/runtime/python3"),"trainer_python_version":"3.12.2",
            "dependency_directories":["/deps"],"trainer_dependency_set_sha256":rovent_trainer_dependency_set_sha256(&endpoints).unwrap(),
            "runtime_endpoints":endpoints,"current_dataset":{"root":"/input/utility-intervention-cohort",
                "publication":pin("/input/utility-intervention-cohort/utility-cohort-publication.json"),
                "durable_ack":pin("/input/utility-intervention-cohort/utility-cohort-durable-ack.json")},
            "ancestor_datasets":[],"initialization_source":{"mode":"initialize","utility":pin("/input/utility.json")},
            "frozen_nnuec_base":pin("/input/base.json"),"frozen_pc_sources":[
                {"role":"proposer","artifact":pin("/input/p.onnx"),"model_semantics_sha256":"3".repeat(64),"model_identity_sha256":"4".repeat(64),"encoding_sha256":"5".repeat(64)},
                {"role":"critic","artifact":pin("/input/c.onnx"),"model_semantics_sha256":"3".repeat(64),"model_identity_sha256":"4".repeat(64),"encoding_sha256":"5".repeat(64)}],
            "resources_policy":{"cpu_quota_millicpus":2000,"memory_high_bytes":6_u64<<30,"memory_max_bytes":12_u64<<30,"pids_max":64,"affinity":[0,2],"device":"cpu"},
            "require_total_quota":true,"output_root":"/run/output","control_root":"/run/control","checkpoint_directory":"/run/output/checkpoints","receipt_path":"/run/output/receipt.json",
            "max_rpc_messages":100,"max_rpc_bytes":1<<20,"max_read_bytes":4<<20,"max_child_read_bytes":1<<20,"max_output_bytes":2<<20,"max_artifact_files":64})
    }

    #[test]
    fn exact_30_field_registration_never_decodes_as_old_registration() {
        let value = registration();
        assert_eq!(value.as_object().unwrap().len(), 30);
        let decoded: RoventUtilityInterventionExecutionRegistration =
            serde_json::from_value(value.clone()).unwrap();
        decoded.validate().unwrap();
        assert!(
            serde_json::from_value::<super::super::RoventUtilityExecutionRegistrationV2>(
                value.clone()
            )
            .is_err()
        );
        let mut old = value.clone();
        old.as_object_mut().unwrap().remove("data_family");
        old["schema"] = json!(super::super::ROVENT_UTILITY_EXECUTION_REGISTRATION_V2);
        assert!(
            serde_json::from_value::<RoventUtilityInterventionExecutionRegistration>(old).is_err()
        );
        let mut changed = decoded;
        changed.data_family = "cohort_v1".into();
        assert!(changed.validate().is_err());
    }

    #[test]
    fn intervention_ancestor_keeps_raw_pins_and_is_not_a_legacy_kind() {
        let value = json!({"kind":"utility_intervention","registration":{
            "root":"/old/utility-intervention-cohort",
            "publication":pin("/old/utility-intervention-cohort/utility-cohort-publication.json"),
            "durable_ack":pin("/old/utility-intervention-cohort/utility-cohort-durable-ack.json")},
            "preparation":pin("/old/original-preparation.json")});
        let decoded: RoventUtilityInterventionAncestorRegistration =
            serde_json::from_value(value.clone()).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded.kind(), "utility_intervention");
        assert!(decoded.legacy().is_none());
        assert!(
            serde_json::from_value::<RoventUtilityAncestorDatasetRegistration>(value.clone())
                .is_err()
        );
        assert_eq!(serde_json::to_value(decoded).unwrap(), value);
    }
}
