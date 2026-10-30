//! Explicit mixed-ancestry declarations for the utility execution /2 lane.
//!
//! The original /1 registration and its canonical bytes are separate. An
//! ancestor keeps its original PC, R or utility registration and raw publication
//! pin; none is projected into an empty /1 inventory or relabelled as utility
//! rows. These typed declarations do not admit files, replay Rules, attest full
//! ancestry coverage or leakage checks, or issue native/training authority.
#![cfg(target_os = "linux")]

use super::execution::{
    MAX_BINARY_BYTES, MAX_METADATA_BYTES, ROVENT_UTILITY_MAX_READ_BYTES,
    RoventUtilityDatasetReference, RoventUtilityExecutionAction, RoventUtilityFrozenPcSource,
    RoventUtilityInitializationSource, RoventUtilityRequiredNullable, canonical_python_version,
    deserialize_endpoints, deserialize_object, execution_identifier, normalized_absolute,
    pin_shape,
};
use super::registration_cohort::RoventUtilityCohortSourceRegistration;
use super::window_v2::RoventUtilityTrainWindowV2;
use crate::ArenaError;
use crate::rovent_training_process::recovery::{
    RoventHistoricalDatasetRegistration, RoventHistoricalRAuxiliaryDatasetRegistration,
    RoventHistoricalRoleBinding, RoventTrainerRuntimeEndpoint,
    rovent_trainer_dependency_set_sha256,
};
use crate::rovent_training_process::{
    MAX_TRAINER_OUTPUT_BYTES, RoventPcRecordFamily, RoventTrainingFilePin,
    RoventTrainingResourcePolicy,
};
use rz_experiments::pals_data_v3::{generation_digest, generation_identifier, generation_sha};
use serde::de::{MapAccess, Visitor, value::MapAccessDeserializer};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const ROVENT_UTILITY_EXECUTION_REGISTRATION_V2: &str =
    "rz-rovent-utility-training-registration/2";

fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Integrity(format!(
        "Rovent utility execution /2 metadata: {}",
        reason.into()
    ))
}
fn check(condition: bool, reason: &str) -> Result<(), ArenaError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}

// Preserve the existing registration DTOs while enforcing dictionary-only
// nested objects at this new boundary. Do not deserialize through Value: that
// would lose duplicate-key evidence before the strict typed decoders see it.
fn deserialize_role_bindings<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<String, RoventHistoricalRoleBinding>, D::Error> {
    struct Binding(RoventHistoricalRoleBinding);
    impl<'de> Deserialize<'de> for Binding {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            deserialize_object(deserializer).map(Self)
        }
    }
    struct Bindings;
    impl<'de> Visitor<'de> for Bindings {
        type Value = BTreeMap<String, RoventHistoricalRoleBinding>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("original, uniquely named PC role binding objects")
        }
        fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
            let mut result = BTreeMap::new();
            while let Some((role, Binding(binding))) = map.next_entry::<String, Binding>()? {
                if result.insert(role, binding).is_some() {
                    return Err(serde::de::Error::custom("duplicate historical PC role"));
                }
                if result.len() > 3 {
                    return Err(serde::de::Error::custom("unbounded historical PC roles"));
                }
            }
            Ok(result)
        }
    }
    deserializer.deserialize_map(Bindings)
}

fn deserialize_pc_registration<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<RoventHistoricalDatasetRegistration, D::Error> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Fields {
        root: PathBuf,
        #[serde(deserialize_with = "deserialize_object")]
        publication: RoventTrainingFilePin,
        collection_source_commit: String,
        collection_source_sha256: String,
        collection_runtime_library_sha256: String,
        model_semantics_sha256: String,
        model_epoch: u64,
        context_epoch: u64,
        recipe_sha256: String,
        stage_id: String,
        #[serde(deserialize_with = "deserialize_role_bindings")]
        role_bindings: BTreeMap<String, RoventHistoricalRoleBinding>,
    }
    let fields: Fields = deserialize_object(deserializer)?;
    Ok(RoventHistoricalDatasetRegistration {
        root: fields.root,
        publication: fields.publication,
        collection_source_commit: fields.collection_source_commit,
        collection_source_sha256: fields.collection_source_sha256,
        collection_runtime_library_sha256: fields.collection_runtime_library_sha256,
        model_semantics_sha256: fields.model_semantics_sha256,
        model_epoch: fields.model_epoch,
        context_epoch: fields.context_epoch,
        recipe_sha256: fields.recipe_sha256,
        stage_id: fields.stage_id,
        role_bindings: fields.role_bindings,
    })
}

fn deserialize_nnuec_pin<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<rz_experiments::RoventNnuecPin, D::Error> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Fields {
        schema: String,
        semantics: String,
        encoding: String,
        implementation_sha256: String,
        configuration_sha256: String,
        #[serde(deserialize_with = "deserialize_object")]
        weights: rz_experiments::ArtifactRef,
        #[serde(deserialize_with = "deserialize_object")]
        provenance: rz_experiments::ArtifactRef,
        diagnostic: bool,
        context_namespace_policy_sha256: String,
        local_namespace_policy_sha256: String,
        search_conditions_sha256: String,
    }
    let fields: Fields = deserialize_object(deserializer)?;
    Ok(rz_experiments::RoventNnuecPin {
        schema: fields.schema,
        semantics: fields.semantics,
        encoding: fields.encoding,
        implementation_sha256: fields.implementation_sha256,
        configuration_sha256: fields.configuration_sha256,
        weights: fields.weights,
        provenance: fields.provenance,
        diagnostic: fields.diagnostic,
        context_namespace_policy_sha256: fields.context_namespace_policy_sha256,
        local_namespace_policy_sha256: fields.local_namespace_policy_sha256,
        search_conditions_sha256: fields.search_conditions_sha256,
    })
}

fn deserialize_r_registration<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<RoventHistoricalRAuxiliaryDatasetRegistration, D::Error> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Fields {
        root: PathBuf,
        #[serde(deserialize_with = "deserialize_object")]
        publication: RoventTrainingFilePin,
        collection_source_commit: String,
        collection_source_sha256: String,
        collection_runtime_library_sha256: String,
        collection_model_semantics_sha256: String,
        collection_model_epoch: u64,
        collection_context_epoch: u64,
        collection_recipe_sha256: String,
        collection_stage_id: String,
        #[serde(deserialize_with = "deserialize_nnuec_pin")]
        nnuec: rz_experiments::RoventNnuecPin,
        nnuec_parameter_sha256: String,
    }
    let fields: Fields = deserialize_object(deserializer)?;
    Ok(RoventHistoricalRAuxiliaryDatasetRegistration {
        root: fields.root,
        publication: fields.publication,
        collection_source_commit: fields.collection_source_commit,
        collection_source_sha256: fields.collection_source_sha256,
        collection_runtime_library_sha256: fields.collection_runtime_library_sha256,
        collection_model_semantics_sha256: fields.collection_model_semantics_sha256,
        collection_model_epoch: fields.collection_model_epoch,
        collection_context_epoch: fields.collection_context_epoch,
        collection_recipe_sha256: fields.collection_recipe_sha256,
        collection_stage_id: fields.collection_stage_id,
        nnuec: fields.nnuec,
        nnuec_parameter_sha256: fields.nnuec_parameter_sha256,
    })
}

/// Original historical declarations, not interchangeable corpus projections.
/// R needs its original generation recipe and each utility ancestor its own
/// original cohort preparation; neither is reconstructed from the new window.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
// One registration owns at most 256 ancestor declarations. Keep each complete
// typed declaration inline instead of adding a separate allocation per R
// ancestor; this is metadata, never a worker message or live execution owner.
#[allow(clippy::large_enum_variant)]
pub enum RoventUtilityAncestorDatasetRegistration {
    PcLegacy {
        registration: RoventHistoricalDatasetRegistration,
    },
    PcSubmittedInputV1 {
        registration: RoventHistoricalDatasetRegistration,
    },
    RAuxiliary {
        registration: RoventHistoricalRAuxiliaryDatasetRegistration,
        recipe: RoventTrainingFilePin,
    },
    UtilityCohort {
        registration: RoventUtilityDatasetReference,
        preparation: RoventTrainingFilePin,
    },
}
impl<'de> Deserialize<'de> for RoventUtilityAncestorDatasetRegistration {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
        // Match the public, bounded declaration layout during strict decoding;
        // the temporary is consumed directly into that same historical type.
        #[allow(clippy::large_enum_variant)]
        enum Fields {
            PcLegacy {
                #[serde(deserialize_with = "deserialize_pc_registration")]
                registration: RoventHistoricalDatasetRegistration,
            },
            PcSubmittedInputV1 {
                #[serde(deserialize_with = "deserialize_pc_registration")]
                registration: RoventHistoricalDatasetRegistration,
            },
            RAuxiliary {
                #[serde(deserialize_with = "deserialize_r_registration")]
                registration: RoventHistoricalRAuxiliaryDatasetRegistration,
                #[serde(deserialize_with = "deserialize_object")]
                recipe: RoventTrainingFilePin,
            },
            UtilityCohort {
                registration: RoventUtilityDatasetReference,
                #[serde(deserialize_with = "deserialize_object")]
                preparation: RoventTrainingFilePin,
            },
        }
        Ok(match deserialize_object(deserializer)? {
            Fields::PcLegacy { registration } => Self::PcLegacy { registration },
            Fields::PcSubmittedInputV1 { registration } => {
                Self::PcSubmittedInputV1 { registration }
            }
            Fields::RAuxiliary {
                registration,
                recipe,
            } => Self::RAuxiliary {
                registration,
                recipe,
            },
            Fields::UtilityCohort {
                registration,
                preparation,
            } => Self::UtilityCohort {
                registration,
                preparation,
            },
        })
    }
}
impl RoventUtilityAncestorDatasetRegistration {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::PcLegacy { .. } => "pc_legacy",
            Self::PcSubmittedInputV1 { .. } => "pc_submitted_input_v1",
            Self::RAuxiliary { .. } => "r_auxiliary",
            Self::UtilityCohort { .. } => "utility_cohort",
        }
    }
    pub fn publication(&self) -> &RoventTrainingFilePin {
        match self {
            Self::PcLegacy { registration } | Self::PcSubmittedInputV1 { registration } => {
                &registration.publication
            }
            Self::RAuxiliary { registration, .. } => &registration.publication,
            Self::UtilityCohort { registration, .. } => &registration.publication,
        }
    }
    pub fn root(&self) -> &Path {
        match self {
            Self::PcLegacy { registration } | Self::PcSubmittedInputV1 { registration } => {
                &registration.root
            }
            Self::RAuxiliary { registration, .. } => &registration.root,
            Self::UtilityCohort { registration, .. } => &registration.root,
        }
    }
    /// A dispatch declaration only; the selected opaque reader must still
    /// compare the original publication, manifest, index and record family.
    pub fn pc_family(&self) -> Option<RoventPcRecordFamily> {
        match self {
            Self::PcLegacy { .. } => Some(RoventPcRecordFamily::LegacyRecordV1),
            Self::PcSubmittedInputV1 { .. } => Some(RoventPcRecordFamily::SubmittedInputV1),
            _ => None,
        }
    }
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            normalized_absolute(self.root()),
            "original ancestor root must be absolute and normalized",
        )?;
        pin_shape(self.publication(), MAX_METADATA_BYTES)?;
        match self {
            Self::PcLegacy { registration } | Self::PcSubmittedInputV1 { registration } => {
                validate_pc_registration(registration)
            }
            Self::RAuxiliary {
                registration,
                recipe,
            } => {
                registration.validate()?;
                // Raw recipe bytes and canonical collection recipe identity
                // are different domains. The actual reader joins both.
                pin_shape(recipe, MAX_METADATA_BYTES)
            }
            Self::UtilityCohort {
                registration,
                preparation,
            } => {
                registration.validate()?;
                pin_shape(preparation, MAX_METADATA_BYTES)
            }
        }
    }
}

fn validate_pc_registration(
    registration: &RoventHistoricalDatasetRegistration,
) -> Result<(), ArenaError> {
    check(
        registration.publication.path == registration.root.join("publication.json")
            && registration.collection_source_commit.len() == 40
            && registration
                .collection_source_commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
            && [
                &registration.collection_source_sha256,
                &registration.collection_runtime_library_sha256,
                &registration.model_semantics_sha256,
                &registration.recipe_sha256,
            ]
            .into_iter()
            .all(|sha| generation_sha(sha))
            && registration.model_epoch > 0
            && registration.context_epoch > 0
            && generation_identifier(&registration.stage_id)
            && (1..=3).contains(&registration.role_bindings.len()),
        "original PC collection source/model/recipe/publication declarations required",
    )?;
    for (role, binding) in &registration.role_bindings {
        check(
            matches!(role.as_str(), "proposer" | "critic" | "validator")
                && [
                    &binding.model_semantics_sha256,
                    &binding.model_identity_sha256,
                    &binding.weights_sha256,
                    &binding.encoding_sha256,
                ]
                .into_iter()
                .all(|sha| generation_sha(sha)),
            "original PC per-role model/raw weights/encoding declarations required",
        )?;
    }
    Ok(())
}

utility_object! {
/// The original 29-field roster, with explicitly typed mixed ancestors. No
/// decoder creates a V1 request, changes an ancestor's source namespace, or
/// replaces an unverified ancestor inventory with an empty one.
pub struct RoventUtilityExecutionRegistrationV2 {
    pub schema: String,
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
    pub ancestor_datasets: Vec<RoventUtilityAncestorDatasetRegistration>,
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
impl RoventUtilityExecutionRegistrationV2 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_UTILITY_EXECUTION_REGISTRATION_V2
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
            "explicit /2 utility source/runtime/dependency declarations required",
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
        generation_digest(ROVENT_UTILITY_EXECUTION_REGISTRATION_V2, self)
            .map_err(ArenaError::Manifest)
    }
    /// Actual-runtime metadata entry accepts cohort PREP2 only. Original files,
    /// full mixed ancestry and Rules/leakage ownership remain separate checks.
    pub fn join_cohort_window(
        &self,
        window: &RoventUtilityTrainWindowV2,
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
    //! Pure metadata fixtures. These do not open an archive or run a trainer.
    use super::*;
    use serde_json::json;

    fn pin(path: &str, pair: &str) -> RoventTrainingFilePin {
        RoventTrainingFilePin {
            path: path.into(),
            sha256: pair.repeat(32),
            bytes: 128,
        }
    }
    fn pc(root: &str, pair: &str) -> RoventHistoricalDatasetRegistration {
        RoventHistoricalDatasetRegistration {
            root: root.into(),
            publication: pin(&format!("{root}/publication.json"), pair),
            collection_source_commit: "a".repeat(40),
            collection_source_sha256: "21".repeat(32),
            collection_runtime_library_sha256: "22".repeat(32),
            model_semantics_sha256: "23".repeat(32),
            model_epoch: 1,
            context_epoch: 2,
            recipe_sha256: "24".repeat(32),
            stage_id: "pc-stage".into(),
            role_bindings: BTreeMap::from([(
                "proposer".into(),
                RoventHistoricalRoleBinding {
                    model_semantics_sha256: "25".repeat(32),
                    model_identity_sha256: "26".repeat(32),
                    weights_sha256: "27".repeat(32),
                    encoding_sha256: "28".repeat(32),
                },
            )]),
        }
    }
    fn request() -> RoventUtilityExecutionRegistrationV2 {
        let runtime_endpoints = vec![
            RoventTrainerRuntimeEndpoint {
                name: "torch.__file__".into(),
                artifact: pin("/deps/torch/__init__.py", "31"),
            },
            RoventTrainerRuntimeEndpoint {
                name: "torch._C.__file__".into(),
                artifact: pin("/deps/torch/native.so", "32"),
            },
        ];
        RoventUtilityExecutionRegistrationV2 {
            schema: ROVENT_UTILITY_EXECUTION_REGISTRATION_V2.into(),
            execution_id: "utility-v2".into(),
            action: RoventUtilityExecutionAction::Train,
            export_request: RoventUtilityRequiredNullable(None),
            preparation: pin("/inputs/preparation.json", "41"),
            window_spec: pin("/inputs/window.json", "42"),
            trainer_source_sha256: "43".repeat(32),
            trainer_python_executable: pin("/runtime/python3", "44"),
            trainer_python_version: "3.12.2".into(),
            dependency_directories: vec!["/deps".into()],
            trainer_dependency_set_sha256: rovent_trainer_dependency_set_sha256(&runtime_endpoints)
                .unwrap(),
            runtime_endpoints,
            current_dataset: RoventUtilityDatasetReference {
                root: "/intake/current".into(),
                publication: pin("/intake/current/utility-cohort-publication.json", "45"),
                durable_ack: pin("/intake/current/utility-cohort-durable-ack.json", "46"),
            },
            ancestor_datasets: vec![
                RoventUtilityAncestorDatasetRegistration::PcLegacy {
                    registration: pc("/intake/pc-old", "51"),
                },
                RoventUtilityAncestorDatasetRegistration::PcSubmittedInputV1 {
                    registration: pc("/intake/pc-submitted", "52"),
                },
                RoventUtilityAncestorDatasetRegistration::UtilityCohort {
                    registration: RoventUtilityDatasetReference {
                        root: "/intake/utility-old".into(),
                        publication: pin(
                            "/intake/utility-old/utility-cohort-publication.json",
                            "53",
                        ),
                        durable_ack: pin(
                            "/intake/utility-old/utility-cohort-durable-ack.json",
                            "54",
                        ),
                    },
                    preparation: pin("/intake/utility-old/original-preparation.json", "55"),
                },
            ],
            initialization_source: RoventUtilityInitializationSource::Initialize {
                utility: pin("/inputs/utility.json", "61"),
            },
            frozen_nnuec_base: pin("/inputs/base.json", "62"),
            frozen_pc_sources: ["proposer", "critic"]
                .into_iter()
                .map(|role| RoventUtilityFrozenPcSource {
                    role: role.into(),
                    artifact: pin(&format!("/inputs/{role}.onnx"), "63"),
                    model_semantics_sha256: "64".repeat(32),
                    model_identity_sha256: "65".repeat(32),
                    encoding_sha256: "66".repeat(32),
                })
                .collect(),
            resources_policy: RoventTrainingResourcePolicy {
                cpu_quota_millicpus: 2000,
                memory_high_bytes: 6 << 30,
                memory_max_bytes: 12 << 30,
                pids_max: 64,
                affinity: vec![0, 2],
                device: "cpu".into(),
            },
            require_total_quota: true,
            output_root: "/run/output".into(),
            control_root: "/run/control".into(),
            checkpoint_directory: "/run/output/checkpoints".into(),
            receipt_path: "/run/output/receipt.json".into(),
            max_rpc_messages: 100,
            max_rpc_bytes: 1 << 20,
            max_read_bytes: 4 << 20,
            max_child_read_bytes: 1 << 20,
            max_output_bytes: 2 << 20,
            max_artifact_files: 64,
        }
    }

    #[test]
    fn typed_family_roundtrip_keeps_original_pc_domains_and_full_inventory() {
        let original = request();
        original.validate().unwrap();
        let raw = serde_json::to_vec(&original).unwrap();
        let decoded: RoventUtilityExecutionRegistrationV2 = serde_json::from_slice(&raw).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(decoded.ancestor_datasets.len(), 3);
        assert_eq!(decoded.ancestor_datasets[0].kind(), "pc_legacy");
        assert_eq!(
            decoded.ancestor_datasets[0].pc_family(),
            Some(RoventPcRecordFamily::LegacyRecordV1)
        );
        assert_eq!(
            decoded.ancestor_datasets[1].pc_family(),
            Some(RoventPcRecordFamily::SubmittedInputV1)
        );
        assert_eq!(
            decoded.ancestor_datasets[0].publication().sha256,
            "51".repeat(32)
        );
        assert_eq!(
            decoded.canonical_sha256().unwrap(),
            original.canonical_sha256().unwrap()
        );
        assert!(
            serde_json::from_slice::<super::super::execution::RoventUtilityExecutionRegistration>(
                &raw
            )
            .is_err()
        );
    }

    #[test]
    fn original_v1_schema_and_untagged_ancestors_are_not_normalized_into_v2() {
        let mut raw = serde_json::to_value(request()).unwrap();
        raw["schema"] = json!(super::super::execution::ROVENT_UTILITY_EXECUTION_REGISTRATION);
        assert!(
            serde_json::from_value::<RoventUtilityExecutionRegistrationV2>(raw.clone())
                .unwrap()
                .validate()
                .is_err()
        );
        raw["schema"] = json!(ROVENT_UTILITY_EXECUTION_REGISTRATION_V2);
        raw["ancestor_datasets"][0] = raw["ancestor_datasets"][0]["registration"].clone();
        assert!(serde_json::from_value::<RoventUtilityExecutionRegistrationV2>(raw).is_err());
    }

    #[test]
    fn explicit_null_is_required_and_nested_sequences_unknowns_and_duplicates_reject() {
        let value = serde_json::to_value(request()).unwrap();
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove("export_request");
        assert!(serde_json::from_value::<RoventUtilityExecutionRegistrationV2>(missing).is_err());
        let mut sequence = value.clone();
        let original_pin = &sequence["ancestor_datasets"][0]["registration"]["publication"];
        sequence["ancestor_datasets"][0]["registration"]["publication"] = json!([
            original_pin["path"],
            original_pin["sha256"],
            original_pin["bytes"]
        ]);
        assert!(serde_json::from_value::<RoventUtilityExecutionRegistrationV2>(sequence).is_err());
        let mut unknown = value;
        unknown["ancestor_datasets"][0]["preparation"] = json!(pin("/extra/prep.json", "71"));
        assert!(serde_json::from_value::<RoventUtilityExecutionRegistrationV2>(unknown).is_err());
        let pc_json = serde_json::to_string(&request().ancestor_datasets[0]).unwrap();
        let duplicate_tag = pc_json.replacen("{", "{\"kind\":\"pc_legacy\",", 1);
        assert!(
            serde_json::from_str::<RoventUtilityAncestorDatasetRegistration>(&duplicate_tag)
                .is_err()
        );
        let binding = serde_json::to_string(&request().ancestor_datasets[0].clone()).unwrap();
        let needle = "\"role_bindings\":{\"proposer\":";
        let duplicate = binding.replacen(
            needle,
            &format!(
                "\"role_bindings\":{{\"proposer\":{},\"proposer\":",
                serde_json::to_string(&pc("/p", "72").role_bindings["proposer"]).unwrap()
            ),
            1,
        );
        assert!(
            serde_json::from_str::<RoventUtilityAncestorDatasetRegistration>(&duplicate).is_err()
        );
    }

    #[test]
    fn cross_family_duplicate_raw_publication_and_wrong_original_path_reject() {
        let mut request = request();
        let duplicate = match &request.ancestor_datasets[0] {
            RoventUtilityAncestorDatasetRegistration::PcLegacy { registration } => {
                registration.clone()
            }
            _ => unreachable!(),
        };
        request.ancestor_datasets.push(
            RoventUtilityAncestorDatasetRegistration::PcSubmittedInputV1 {
                registration: duplicate,
            },
        );
        assert!(request.validate().is_err());
        request.ancestor_datasets.pop();
        if let RoventUtilityAncestorDatasetRegistration::PcLegacy { registration } =
            &mut request.ancestor_datasets[0]
        {
            registration.publication.path = registration.root.join("different.json");
        }
        assert!(request.validate().is_err());
    }

    #[test]
    fn r_and_utility_variants_require_their_own_original_auxiliary_artifact_pins() {
        let utility = serde_json::to_value(&request().ancestor_datasets[2]).unwrap();
        let mut missing = utility;
        missing.as_object_mut().unwrap().remove("preparation");
        assert!(
            serde_json::from_value::<RoventUtilityAncestorDatasetRegistration>(missing).is_err()
        );
        let r = json!({"kind":"r_auxiliary","registration":{
            "root":"/intake/r","publication":pin("/intake/r/r-publication.json","81"),
            "collection_source_commit":"a".repeat(40),"collection_source_sha256":"82".repeat(32),
            "collection_runtime_library_sha256":"83".repeat(32),
            "collection_model_semantics_sha256":"84".repeat(32),"collection_model_epoch":1,
            "collection_context_epoch":2,"collection_recipe_sha256":"85".repeat(32),"collection_stage_id":"r-stage",
            "nnuec":{"schema":"declared-schema","semantics":"declared-semantics","encoding":"declared-encoding",
                "implementation_sha256":"86".repeat(32),"configuration_sha256":"87".repeat(32),
                "weights":{"path":"original/weights.json","sha256":"88".repeat(32),"bytes":128,"source":"https://example.invalid/source","license":"MIT"},
                "provenance":{"path":"original/provenance.json","sha256":"89".repeat(32),"bytes":128,"source":"https://example.invalid/source","license":"MIT"},
                "diagnostic":false,"context_namespace_policy_sha256":"90".repeat(32),
                "local_namespace_policy_sha256":"91".repeat(32),"search_conditions_sha256":"92".repeat(32)},
            "nnuec_parameter_sha256":"93".repeat(32)},
            "recipe":pin("/intake/r/original-recipe.json","94")});
        let parsed: RoventUtilityAncestorDatasetRegistration =
            serde_json::from_value(r.clone()).unwrap();
        assert_eq!(parsed.kind(), "r_auxiliary");
        assert_eq!(serde_json::to_value(parsed).unwrap(), r);
        let mut no_recipe = r.clone();
        no_recipe.as_object_mut().unwrap().remove("recipe");
        assert!(
            serde_json::from_value::<RoventUtilityAncestorDatasetRegistration>(no_recipe).is_err()
        );
        let mut bad_nested = r;
        bad_nested["registration"]["nnuec"]["weights"] = json!([]);
        assert!(
            serde_json::from_value::<RoventUtilityAncestorDatasetRegistration>(bad_nested).is_err()
        );
    }
}
