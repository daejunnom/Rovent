//! Utility-only execution declarations. No decoder or metadata join in this
//! module issues an execution lease, historical/native authority, valid target,
//! observed frozen state, optimizer update, or learned export. In particular,
//! a root-local raw/1 file cannot become a known utility cohort through JSON.
//!
//! The actual parent must separately validate the E/CPU cohort publication,
//! original held files, source/runtime identity, process/resources and tensor
//! state. Collection, semantic parameter, canonical metadata and raw file pins
//! remain distinct namespaces here.
#![cfg(target_os = "linux")]

use super::registration::{
    ROVENT_UTILITY_MAX_PREPARATION_BYTES, ROVENT_UTILITY_PREPARATION_DOMAIN,
    RoventUtilityParameterSpec, RoventUtilitySourceRegistration, RoventUtilityTrainingRecipe,
};
use super::registration_cohort::{
    ROVENT_UTILITY_COHORT_PREPARATION_DOMAIN, RoventUtilityCohortSourceRegistration,
    validate_utility_cohort_preparation,
};
use crate::ArenaError;
use crate::pals_collect_v3::training_process::GenerationTrainingFilePinV3;
use crate::rovent_training_process::recovery::{
    RoventTrainerRuntimeEndpoint, rovent_trainer_dependency_set_sha256,
};
use crate::rovent_training_process::{
    MAX_TRAINER_OUTPUT_BYTES, RoventTrainingFilePin, RoventTrainingResourcePolicy,
};
use rz_experiments::pals_data_v3::{generation_digest, generation_sha};
use serde::de::{MapAccess, Visitor, value::MapAccessDeserializer};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::path::{Component, Path, PathBuf};

pub const ROVENT_UTILITY_EXECUTION_REGISTRATION: &str = "rz-rovent-utility-training-registration/1";
pub const ROVENT_UTILITY_WINDOW_DOMAIN: &str = "rz-rovent-utility-train-window/1";
pub const ROVENT_UTILITY_MAX_UPDATES: u64 = 131_072;
pub const ROVENT_UTILITY_MAX_WALL_NS: u64 = 86_400_000_000_000;
pub const ROVENT_UTILITY_MAX_CLEANUP_NS: u64 = 30_000_000_000;
pub const ROVENT_UTILITY_MAX_READ_BYTES: u64 = MAX_TRAINER_OUTPUT_BYTES * 260;
pub(super) const MAX_METADATA_BYTES: u64 = 16 << 20;
pub(super) const MAX_BINARY_BYTES: u64 = 2 << 30;

fn invalid(reason: impl Into<String>) -> ArenaError {
    ArenaError::Integrity(format!(
        "Rovent utility execution metadata: {}",
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
pub(super) fn execution_identifier(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 128
        && bytes[0].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(*byte, b'_' | b'-'))
}
fn canonical_device(value: &str) -> bool {
    value == "cpu"
        || value.strip_prefix("cuda:").is_some_and(|index| {
            !index.is_empty()
                && index.len() <= 2
                && index.bytes().all(|byte| byte.is_ascii_digit())
                && index
                    .parse::<u32>()
                    .is_ok_and(|number| number < 64 && number.to_string() == index)
        })
}
pub(super) fn canonical_python_version(value: &str) -> bool {
    let parts = value.split('.').collect::<Vec<_>>();
    parts.len() == 3
        && parts[0] == "3"
        && parts.iter().all(|part| {
            !part.is_empty()
                && part.bytes().all(|byte| byte.is_ascii_digit())
                && part
                    .parse::<u32>()
                    .is_ok_and(|number| number.to_string() == *part)
        })
}
pub(super) fn normalized_absolute(path: &Path) -> bool {
    if !path.is_absolute() || path.components().count() > 64 {
        return false;
    }
    if !path.components().all(|part| match part {
        Component::RootDir => true,
        Component::Normal(name) => name
            .to_str()
            .is_some_and(|name| !name.chars().any(char::is_control)),
        _ => false,
    }) {
        return false;
    }
    // Unlike components-only tests, compare the original spelling too: a/./b,
    // repeated separators and a trailing slash are not silently normalized.
    path.components().collect::<PathBuf>().as_os_str() == path.as_os_str()
}
pub(super) fn pin_shape(pin: &RoventTrainingFilePin, maximum: u64) -> Result<(), ArenaError> {
    GenerationTrainingFilePinV3 {
        path: pin.path.clone(),
        sha256: pin.sha256.clone(),
        bytes: pin.bytes,
    }
    .validate(maximum)?;
    check(
        normalized_absolute(&pin.path),
        "original absolute normalized file pin required",
    )
}

/// Existing pure DTOs have struct deserializers which also accept sequences.
/// Adapt them at this new wire boundary without changing their original codecs.
pub(super) fn deserialize_object<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Object<T>(PhantomData<T>);
    impl<'de, T: Deserialize<'de>> Visitor<'de> for Object<T> {
        type Value = T;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("an explicit utility metadata object")
        }
        fn visit_map<M: MapAccess<'de>>(self, map: M) -> Result<T, M::Error> {
            T::deserialize(MapAccessDeserializer::new(map))
        }
    }
    deserializer.deserialize_map(Object(PhantomData))
}

/// Required field whose explicit wire value may be null. deserialize_any is
/// intentional: serde's MissingFieldDeserializer has a special option route
/// that must not turn an omitted key into an accepted explicit null.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct RoventUtilityRequiredNullable<T>(pub Option<T>);
impl<T> RoventUtilityRequiredNullable<T> {
    pub fn as_ref(&self) -> Option<&T> {
        self.0.as_ref()
    }
    pub fn is_none(&self) -> bool {
        self.0.is_none()
    }
    pub fn is_some(&self) -> bool {
        self.0.is_some()
    }
}
impl<'de, T: Deserialize<'de>> Deserialize<'de> for RoventUtilityRequiredNullable<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Nullable<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> Visitor<'de> for Nullable<T> {
            type Value = RoventUtilityRequiredNullable<T>;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an explicit null or utility metadata object")
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(RoventUtilityRequiredNullable(None))
            }
            fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(RoventUtilityRequiredNullable(None))
            }
            fn visit_map<M: MapAccess<'de>>(self, map: M) -> Result<Self::Value, M::Error> {
                T::deserialize(MapAccessDeserializer::new(map))
                    .map(|value| RoventUtilityRequiredNullable(Some(value)))
            }
        }
        deserializer.deserialize_any(Nullable(PhantomData))
    }
}

struct ObjectOnly<T>(T);
impl<'de, T: Deserialize<'de>> Deserialize<'de> for ObjectOnly<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserialize_object(deserializer).map(Self)
    }
}
pub(super) fn deserialize_endpoints<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<RoventTrainerRuntimeEndpoint>, D::Error> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Endpoint {
        name: String,
        #[serde(deserialize_with = "deserialize_object")]
        artifact: RoventTrainingFilePin,
    }
    Vec::<ObjectOnly<Endpoint>>::deserialize(deserializer).map(|endpoints| {
        endpoints
            .into_iter()
            .map(|ObjectOnly(endpoint)| RoventTrainerRuntimeEndpoint {
                name: endpoint.name,
                artifact: endpoint.artifact,
            })
            .collect()
    })
}

utility_object! {
/// Exactly the 22 WINDOW_FIELDS in rovent_utility_training_window.py. Zero
/// optimizer updates remains a valid finite export/inspection window; this
/// metadata never claims that any update has happened.
pub struct RoventUtilityTrainWindow {
    pub schema: String,
    pub window_id: String,
    pub run_id: String,
    pub preparation_sha256: String,
    pub dataset_sha256: String,
    pub device: String,
    pub precision: String,
    pub tf32: bool,
    pub deterministic: bool,
    pub cpu_threads: u32,
    pub interop_threads: u32,
    pub training_seed: u32,
    pub microbatch: u32,
    pub accumulation: u32,
    pub max_optimizer_updates: u64,
    pub checkpoint_every_updates: u64,
    pub max_wall_time_ns: u64,
    pub cleanup_time_ns: u64,
    pub max_read_bytes: u64,
    pub max_output_bytes: u64,
    pub max_groups: u32,
    pub max_contexts: u32,
}
}
impl RoventUtilityTrainWindow {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_UTILITY_WINDOW_DOMAIN
                && execution_identifier(&self.window_id)
                && execution_identifier(&self.run_id)
                && generation_sha(&self.preparation_sha256)
                && generation_sha(&self.dataset_sha256),
            "explicit utility window/run/preparation/dataset metadata required",
        )?;
        check(
            canonical_device(&self.device)
                && self.precision == "fp32"
                && !self.tf32
                && self.deterministic
                && (1..=256).contains(&self.cpu_threads)
                && (1..=256).contains(&self.interop_threads)
                && self.microbatch == 1
                && self.accumulation == 8,
            "deterministic FP32 utility window, microbatch 1 and accumulation 8 required",
        )?;
        check(
            self.max_optimizer_updates <= ROVENT_UTILITY_MAX_UPDATES
                && (1..=ROVENT_UTILITY_MAX_UPDATES).contains(&self.checkpoint_every_updates)
                && (1..=ROVENT_UTILITY_MAX_WALL_NS).contains(&self.max_wall_time_ns)
                && (1..=ROVENT_UTILITY_MAX_CLEANUP_NS).contains(&self.cleanup_time_ns)
                && (1..=ROVENT_UTILITY_MAX_READ_BYTES).contains(&self.max_read_bytes)
                && (1..=MAX_TRAINER_OUTPUT_BYTES).contains(&self.max_output_bytes)
                && self.max_groups > 0
                && self.max_contexts > 0,
            "finite original utility update/group/context/time/read/output limits required",
        )
    }
    pub fn canonical_sha256(&self) -> Result<String, ArenaError> {
        self.validate()?;
        generation_digest(ROVENT_UTILITY_WINDOW_DOMAIN, self).map_err(ArenaError::Manifest)
    }
    pub fn join_preparation(&self, preparation: &Value) -> Result<(), ArenaError> {
        self.validate()?;
        check(
            self.preparation_sha256 == validate_utility_preparation(preparation)?,
            "window changed its exact utility preparation metadata seal",
        )
    }
    /// Version-aware metadata reader only. Actual cohort execution uses the
    /// separately named cohort-only join below, never this general dispatch.
    pub fn join_training_preparation(
        &self,
        preparation: &Value,
    ) -> Result<RoventUtilityPreparationMetadata, ArenaError> {
        self.validate()?;
        let metadata = validate_utility_training_preparation(preparation)?;
        check(
            self.preparation_sha256 == metadata.canonical_sha256,
            "window changed the original preparation family seal",
        )?;
        Ok(metadata)
    }
    /// V2 metadata join only. Original V1 declarations do not enter this lane,
    /// even if numerical parameters, recipe or route labels otherwise agree.
    pub fn join_cohort_preparation(&self, preparation: &Value) -> Result<(), ArenaError> {
        self.validate()?;
        check(
            self.preparation_sha256 == validate_utility_cohort_preparation(preparation)?,
            "cohort window changed its original V2 preparation metadata seal",
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventUtilityPreparationFamily {
    RootLocalV1,
    CohortV2,
}
impl RoventUtilityPreparationFamily {
    pub fn domain(self) -> &'static str {
        match self {
            Self::RootLocalV1 => ROVENT_UTILITY_PREPARATION_DOMAIN,
            Self::CohortV2 => ROVENT_UTILITY_COHORT_PREPARATION_DOMAIN,
        }
    }
}
/// Pure parsing result with an explicit original family. It deliberately is
/// not named Verified and carries no cohort, source or execution authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoventUtilityPreparationMetadata {
    pub family: RoventUtilityPreparationFamily,
    pub canonical_sha256: String,
}
/// F's explicit schema1/schema2 dispatch. Each original parser recomputes its
/// own complete exact result; no default schema, fill, retag or resign exists.
pub fn validate_utility_training_preparation(
    preparation: &Value,
) -> Result<RoventUtilityPreparationMetadata, ArenaError> {
    let schema = preparation
        .as_object()
        .and_then(|object| object.get("schema"))
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("explicit preparation object/schema required"))?;
    let family = match schema {
        ROVENT_UTILITY_PREPARATION_DOMAIN => RoventUtilityPreparationFamily::RootLocalV1,
        ROVENT_UTILITY_COHORT_PREPARATION_DOMAIN => RoventUtilityPreparationFamily::CohortV2,
        _ => {
            return Err(invalid(
                "registered explicit utility preparation V1/V2 domain required",
            ));
        }
    };
    let canonical_sha256 = match family {
        RoventUtilityPreparationFamily::RootLocalV1 => validate_utility_preparation(preparation)?,
        RoventUtilityPreparationFamily::CohortV2 => {
            validate_utility_cohort_preparation(preparation)?
        }
    };
    Ok(RoventUtilityPreparationMetadata {
        family,
        canonical_sha256,
    })
}

/// Recompute F's complete metadata-only preparation. The exact false flags are
/// never changed into successful source/tensor/optimizer observations here.
pub fn validate_utility_preparation(preparation: &Value) -> Result<String, ArenaError> {
    let object = preparation
        .as_object()
        .ok_or_else(|| invalid("preparation object required"))?;
    let field = |name: &str| {
        object
            .get(name)
            .cloned()
            .ok_or_else(|| invalid(format!("missing preparation {name}")))
    };
    let recipe: RoventUtilityTrainingRecipe =
        serde_json::from_value(field("recipe")?).map_err(|error| invalid(error.to_string()))?;
    let registration: RoventUtilitySourceRegistration =
        serde_json::from_value(field("registration")?)
            .map_err(|error| invalid(error.to_string()))?;
    let spec: RoventUtilityParameterSpec = serde_json::from_value(field("parameter_spec")?)
        .map_err(|error| invalid(error.to_string()))?;
    registration.validate(&recipe, &spec)?;
    let mut expected = json!({
        "schema":ROVENT_UTILITY_PREPARATION_DOMAIN,"status":"prepared_metadata_only",
        "recipe":recipe,"recipe_sha256":recipe.canonical_sha256()?,
        "registration":registration,"registration_sha256":registration.canonical_sha256(&recipe,&spec)?,
        "parameter_spec":spec,"parameter_spec_sha256":spec.canonical_sha256(&registration.frozen_base.config,&recipe.optimizer_owner)?,
        "optimizer_owner":recipe.optimizer_owner,"prepared_artifacts":["checkpoint_metadata","export_metadata"],
        "tensor_membership_verified":false,"source_authority_verified":false,
        "optimizer_state_present":false,"checkpoint_tensor_present":false,"export_tensor_present":false,
        "performed_optimizer_updates":0,"execution_authorized":false
    });
    let mut envelope = json!([ROVENT_UTILITY_PREPARATION_DOMAIN, expected]);
    envelope.sort_all_objects();
    check(
        serde_json::to_vec(&envelope)
            .map_err(|error| invalid(error.to_string()))?
            .len()
            <= ROVENT_UTILITY_MAX_PREPARATION_BYTES,
        "preparation digest envelope exceeds 64KiB",
    )?;
    let sha256 = generation_digest(ROVENT_UTILITY_PREPARATION_DOMAIN, &expected)
        .map_err(ArenaError::Manifest)?;
    expected["preparation_sha256"] = json!(sha256);
    check(
        *preparation == expected,
        "preparation differs from the complete original metadata-only result",
    )?;
    Ok(sha256)
}

utility_object! {
/// A caller-declared original publication pin, not a loaded historical cohort
/// owner. The separate durable_ack is an original artifact declaration, not a
/// JSON commit flag. Dataset semantic SHA and raw file SHA are not equated.
pub struct RoventUtilityDatasetReference {
    pub root: PathBuf,
    #[serde(deserialize_with = "deserialize_object")]
    pub publication: RoventTrainingFilePin,
    #[serde(deserialize_with = "deserialize_object")]
    pub durable_ack: RoventTrainingFilePin,
}
}
impl RoventUtilityDatasetReference {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            normalized_absolute(&self.root)
                && self.publication.path.starts_with(&self.root)
                && self.publication.path != self.root,
            "original utility publication escapes normalized registered root",
        )?;
        pin_shape(&self.publication, MAX_METADATA_BYTES)?;
        pin_shape(&self.durable_ack, MAX_METADATA_BYTES)?;
        check(
            self.durable_ack.path.starts_with(&self.root)
                && self.durable_ack.path != self.root
                && self.durable_ack.path != self.publication.path,
            "separate original durable dataset ACK must stay inside its registered root",
        )
    }
}

utility_object! {
/// Original utility checkpoint declaration. Dataset durable ACK semantics are
/// not injected into this separate two-field checkpoint-reference codec.
pub struct RoventUtilityCheckpointReference {
    pub root: PathBuf,
    #[serde(deserialize_with = "deserialize_object")]
    pub publication: RoventTrainingFilePin,
}
}
impl RoventUtilityCheckpointReference {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            normalized_absolute(&self.root)
                && self.publication.path.starts_with(&self.root)
                && self.publication.path != self.root,
            "original utility checkpoint publication escapes its registered root",
        )?;
        pin_shape(&self.publication, MAX_METADATA_BYTES)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoventUtilityExecutionAction {
    Train,
    Export,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum RoventUtilityInitializationSource {
    Initialize {
        utility: RoventTrainingFilePin,
    },
    Resume {
        checkpoint: RoventUtilityCheckpointReference,
    },
    Fork {
        checkpoint: RoventUtilityCheckpointReference,
    },
}
impl<'de> Deserialize<'de> for RoventUtilityInitializationSource {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
        enum Fields {
            Initialize {
                #[serde(deserialize_with = "deserialize_object")]
                utility: RoventTrainingFilePin,
            },
            Resume {
                checkpoint: RoventUtilityCheckpointReference,
            },
            Fork {
                checkpoint: RoventUtilityCheckpointReference,
            },
        }
        Ok(match deserialize_object(deserializer)? {
            Fields::Initialize { utility } => Self::Initialize { utility },
            Fields::Resume { checkpoint } => Self::Resume { checkpoint },
            Fields::Fork { checkpoint } => Self::Fork { checkpoint },
        })
    }
}
impl RoventUtilityInitializationSource {
    pub(super) fn validate(&self) -> Result<(), ArenaError> {
        match self {
            Self::Initialize { utility } => pin_shape(utility, MAX_BINARY_BYTES),
            Self::Resume { checkpoint } | Self::Fork { checkpoint } => checkpoint.validate(),
        }
    }
}

utility_object! {
/// Frozen P/C sources are separately registered rather than inserted into the
/// original eight-field F source codec. These are declarations, not observations
/// of frozen tensors or a deployment model capability.
pub struct RoventUtilityFrozenPcSource {
    pub role: String,
    #[serde(deserialize_with = "deserialize_object")]
    pub artifact: RoventTrainingFilePin,
    pub model_semantics_sha256: String,
    pub model_identity_sha256: String,
    pub encoding_sha256: String,
}
}

utility_object! {
/// The complete new 29-field utility process request. All keys are required,
/// including an explicit null export_request on Train. Source pin declarations
/// must be joined to original files and the new cohort owner before execution.
pub struct RoventUtilityExecutionRegistration {
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
    pub ancestor_datasets: Vec<RoventUtilityDatasetReference>,
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
impl RoventUtilityExecutionRegistration {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_UTILITY_EXECUTION_REGISTRATION
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
            "utility execution source/runtime/dependency declarations required",
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
            "bounded utility ancestor publication inventory",
        )?;
        let mut publications = BTreeSet::from([self.current_dataset.publication.sha256.as_str()]);
        for ancestor in &self.ancestor_datasets {
            ancestor.validate()?;
            check(
                publications.insert(ancestor.publication.sha256.as_str()),
                "duplicate original current/ancestor utility publication",
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
        generation_digest(ROVENT_UTILITY_EXECUTION_REGISTRATION, self).map_err(ArenaError::Manifest)
    }
    /// Pure identity/budget join only. The frozen base FilePin SHA is a raw
    /// file pin and must not equal the preparation's semantic parameter SHA by
    /// assumption. The dataset canonical seal likewise is not its raw file pin.
    pub fn join_window(
        &self,
        window: &RoventUtilityTrainWindow,
        preparation: &Value,
    ) -> Result<(), ArenaError> {
        self.validate()?;
        window.join_preparation(preparation)?;
        let source: RoventUtilitySourceRegistration =
            serde_json::from_value(preparation["registration"].clone())
                .map_err(|error| invalid(error.to_string()))?;
        check(
            source.trainer_source_sha256 == self.trainer_source_sha256
                && window.device == self.resources_policy.device
                && window.max_read_bytes == self.max_child_read_bytes
                && window.max_output_bytes <= self.max_output_bytes,
            "window changed declared utility trainer/resource/read/output allowances",
        )
    }
    /// Metadata checks for the actual parent's cohort-only lane. Passing this
    /// does not validate original trial/cost/physical pin-zero evidence: that
    /// right must still originate in the opaque E/CPU cohort owner.
    pub fn join_cohort_window(
        &self,
        window: &RoventUtilityTrainWindow,
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
    //! Pure metadata source fixtures only; no files, processes or tensors.
    use super::*;

    fn pin(path: &str, byte: &str) -> RoventTrainingFilePin {
        RoventTrainingFilePin {
            path: path.into(),
            sha256: byte.repeat(32),
            bytes: 128,
        }
    }
    fn dataset() -> RoventUtilityDatasetReference {
        RoventUtilityDatasetReference {
            root: "/intake/current".into(),
            publication: pin("/intake/current/publication.json", "11"),
            durable_ack: pin("/intake/current/durable-ack.json", "14"),
        }
    }
    fn checkpoint() -> RoventUtilityCheckpointReference {
        RoventUtilityCheckpointReference {
            root: "/intake/checkpoint".into(),
            publication: pin("/intake/checkpoint/publication.json", "15"),
        }
    }
    fn window() -> RoventUtilityTrainWindow {
        RoventUtilityTrainWindow {
            schema: ROVENT_UTILITY_WINDOW_DOMAIN.into(),
            window_id: "utility-window".into(),
            run_id: "utility-run".into(),
            preparation_sha256: "12".repeat(32),
            dataset_sha256: "13".repeat(32),
            device: "cpu".into(),
            precision: "fp32".into(),
            tf32: false,
            deterministic: true,
            cpu_threads: 2,
            interop_threads: 1,
            training_seed: 31,
            microbatch: 1,
            accumulation: 8,
            max_optimizer_updates: 4,
            checkpoint_every_updates: 2,
            max_wall_time_ns: 60_000_000_000,
            cleanup_time_ns: 30_000_000_000,
            max_read_bytes: 1 << 20,
            max_output_bytes: 1 << 20,
            max_groups: 16,
            max_contexts: 128,
        }
    }
    fn registration() -> RoventUtilityExecutionRegistration {
        let endpoints = vec![
            RoventTrainerRuntimeEndpoint {
                name: "torch.__file__".into(),
                artifact: pin("/deps/torch/__init__.py", "21"),
            },
            RoventTrainerRuntimeEndpoint {
                name: "torch._C.__file__".into(),
                artifact: pin("/deps/torch/native.so", "22"),
            },
        ];
        RoventUtilityExecutionRegistration {
            schema: ROVENT_UTILITY_EXECUTION_REGISTRATION.into(),
            execution_id: "utility-execution".into(),
            action: RoventUtilityExecutionAction::Train,
            export_request: RoventUtilityRequiredNullable(None),
            preparation: pin("/inputs/preparation.json", "30"),
            window_spec: pin("/inputs/window.json", "31"),
            trainer_source_sha256: "32".repeat(32),
            trainer_python_executable: pin("/runtime/python3", "33"),
            trainer_python_version: "3.12.2".into(),
            dependency_directories: vec!["/deps".into()],
            trainer_dependency_set_sha256: rovent_trainer_dependency_set_sha256(&endpoints)
                .unwrap(),
            runtime_endpoints: endpoints,
            current_dataset: dataset(),
            ancestor_datasets: vec![],
            initialization_source: RoventUtilityInitializationSource::Initialize {
                utility: pin("/inputs/utility.json", "34"),
            },
            frozen_nnuec_base: pin("/inputs/frozen-base.json", "35"),
            frozen_pc_sources: ["proposer", "critic"]
                .into_iter()
                .map(|role| RoventUtilityFrozenPcSource {
                    role: role.into(),
                    artifact: pin(&format!("/inputs/{role}.onnx"), "36"),
                    model_semantics_sha256: "37".repeat(32),
                    model_identity_sha256: "38".repeat(32),
                    encoding_sha256: "39".repeat(32),
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

    fn preparation() -> Value {
        let recipe = RoventUtilityTrainingRecipe::fixed("utility-owner").unwrap();
        let config = json!({"board_width":2,"line_width":3,"context_width":4,"local_width":5,
            "hidden_width":6,"causal_window":3,"max_lines":64,"max_plies":256});
        let spec: RoventUtilityParameterSpec = serde_json::from_value(json!({
            "hidden_width":2,"input_width":17,"optimizer_owner":"utility-owner",
            "frozen_components":["nnuec_base","p","c"],
            "wire_to_parameter":{"hidden_weights":"hidden_weights","hidden_bias":"hidden_bias",
                "output_weights":"output_weights","output_bias":"output_bias"},
            "parameters":{
                "hidden_weights":{"checkpoint_path":"parameters.hidden_weights","shape":[2,17],"dtype":"float32","planned_requires_grad":true},
                "hidden_bias":{"checkpoint_path":"parameters.hidden_bias","shape":[2],"dtype":"float32","planned_requires_grad":true},
                "output_weights":{"checkpoint_path":"parameters.output_weights","shape":[7,2],"dtype":"float32","planned_requires_grad":true},
                "output_bias":{"checkpoint_path":"parameters.output_bias","shape":[7],"dtype":"float32","planned_requires_grad":true}
            }
        })).unwrap();
        let base_config = serde_json::from_value(config.clone()).unwrap();
        let source: RoventUtilitySourceRegistration = serde_json::from_value(json!({
            "collection_source_sha256":"10".repeat(32),"trainer_source_sha256":"32".repeat(32),
            "cost_epoch_sha256":"50".repeat(32),"resource_profile_sha256":"40".repeat(32),
            "cost_scope":"exclusive_submitted_global_target_route_and_descendants_with_required_drain",
            "objective":"exclusive_wall_nanoseconds",
            "frozen_base":{"schema":"rz-nnuec-fp32-v2",
                "semantics":"board800-exactb192-causal-line-kind4-directedpair-originfirst-exactc256-fp32-v2",
                "weights_sha256":"20".repeat(32),"config":config},
            "utility":{"schema":"rz-nnuec-utility-fp32-v1","semantics":"nnuec-v2-targetbundle-risk-route-shadow-fp32-v1",
                "initial_weights_sha256":"30".repeat(32),"base_weights_sha256":"20".repeat(32),
                "parameter_spec_sha256":spec.canonical_sha256(&base_config,&recipe.optimizer_owner).unwrap()}
        })).unwrap();
        let mut result = json!({
            "schema":ROVENT_UTILITY_PREPARATION_DOMAIN,"status":"prepared_metadata_only",
            "recipe":recipe,"recipe_sha256":recipe.canonical_sha256().unwrap(),
            "registration":source,"registration_sha256":source.canonical_sha256(&recipe,&spec).unwrap(),
            "parameter_spec":spec,"parameter_spec_sha256":spec.canonical_sha256(&base_config,&recipe.optimizer_owner).unwrap(),
            "optimizer_owner":recipe.optimizer_owner,"prepared_artifacts":["checkpoint_metadata","export_metadata"],
            "tensor_membership_verified":false,"source_authority_verified":false,"optimizer_state_present":false,
            "checkpoint_tensor_present":false,"export_tensor_present":false,"performed_optimizer_updates":0,"execution_authorized":false
        });
        result["preparation_sha256"] =
            json!(generation_digest(ROVENT_UTILITY_PREPARATION_DOMAIN, &result).unwrap());
        result
    }

    fn cohort_preparation() -> Value {
        // A distinct pure metadata fixture, not a conversion of actual V1
        // records or a claim that a cohort trial has been executed.
        let fixture = preparation();
        let recipe = serde_json::from_value(fixture["recipe"].clone()).unwrap();
        let spec = serde_json::from_value(fixture["parameter_spec"].clone()).unwrap();
        let mut source = fixture["registration"].clone();
        source["cost_scope"] =
            json!(super::super::registration_cohort::ROVENT_UTILITY_COHORT_COST_SCOPE);
        let source = serde_json::from_value(source).unwrap();
        super::super::registration_cohort::prepare_utility_cohort_training(&recipe, &source, &spec)
            .unwrap()
    }

    #[test]
    fn explicit_preparation_dispatch_preserves_original_families_and_cohort_only_admission() {
        let v1 = preparation();
        let v2 = cohort_preparation();
        let old = validate_utility_training_preparation(&v1).unwrap();
        let cohort = validate_utility_training_preparation(&v2).unwrap();
        assert_eq!(old.family, RoventUtilityPreparationFamily::RootLocalV1);
        assert_eq!(cohort.family, RoventUtilityPreparationFamily::CohortV2);
        assert_ne!(old.canonical_sha256, cohort.canonical_sha256);
        assert_eq!(v1["recipe_sha256"], v2["recipe_sha256"]);
        assert_eq!(v1["parameter_spec_sha256"], v2["parameter_spec_sha256"]);
        assert_ne!(v1["registration_sha256"], v2["registration_sha256"]);
        let execution = registration();
        let mut window = window();
        window.preparation_sha256 = old.canonical_sha256;
        execution.join_window(&window, &v1).unwrap();
        assert!(execution.join_cohort_window(&window, &v1).is_err());
        window.preparation_sha256 = cohort.canonical_sha256;
        assert_eq!(
            window.join_training_preparation(&v2).unwrap().family,
            RoventUtilityPreparationFamily::CohortV2
        );
        execution.join_cohort_window(&window, &v2).unwrap();
        let mut different_credit = window.clone();
        different_credit.max_read_bytes -= 1;
        assert!(
            execution
                .join_cohort_window(&different_credit, &v2)
                .is_err()
        );
        assert!(execution.join_window(&window, &v2).is_err());
        let mut changed = v1.clone();
        changed["schema"] = json!(ROVENT_UTILITY_COHORT_PREPARATION_DOMAIN);
        assert!(validate_utility_training_preparation(&changed).is_err());
        changed = v2;
        changed["schema"] = json!("rz-rovent-utility-training-preparation/3");
        assert!(validate_utility_training_preparation(&changed).is_err());
        assert_eq!(
            validate_utility_preparation(&v1).unwrap(),
            v1["preparation_sha256"].as_str().unwrap()
        );
    }

    #[test]
    fn preparation_recomputes_only_the_original_metadata_without_promoting_flags() {
        let original = preparation();
        assert_eq!(
            validate_utility_preparation(&original).unwrap(),
            original["preparation_sha256"].as_str().unwrap()
        );
        for name in [
            "tensor_membership_verified",
            "source_authority_verified",
            "optimizer_state_present",
            "checkpoint_tensor_present",
            "export_tensor_present",
            "execution_authorized",
        ] {
            let mut changed = original.clone();
            changed[name] = json!(true);
            assert!(validate_utility_preparation(&changed).is_err());
        }
        let mut changed = original.clone();
        changed["performed_optimizer_updates"] = json!(1);
        assert!(validate_utility_preparation(&changed).is_err());
        changed = original.clone();
        changed["extra"] = json!(false);
        assert!(validate_utility_preparation(&changed).is_err());
        changed = original;
        changed["preparation_sha256"] = json!("ff".repeat(32));
        assert!(validate_utility_preparation(&changed).is_err());
    }

    #[test]
    fn window_joins_canonical_metadata_without_aliasing_raw_file_or_parameter_pins() {
        let preparation = preparation();
        let execution = registration();
        let mut window = window();
        window.preparation_sha256 = validate_utility_preparation(&preparation).unwrap();
        execution.join_window(&window, &preparation).unwrap();
        assert_ne!(execution.preparation.sha256, window.preparation_sha256);
        assert_ne!(
            execution.current_dataset.publication.sha256,
            window.dataset_sha256
        );
        assert_ne!(
            execution.frozen_nnuec_base.sha256,
            preparation["registration"]["frozen_base"]["weights_sha256"]
                .as_str()
                .unwrap()
        );
        let mut changed = execution.clone();
        changed.trainer_source_sha256 = "99".repeat(32);
        assert!(changed.join_window(&window, &preparation).is_err());
        let mut changed = window.clone();
        changed.max_read_bytes = execution.max_child_read_bytes + 1;
        assert!(execution.join_window(&changed, &preparation).is_err());
        changed.max_read_bytes = execution.max_child_read_bytes - 1;
        assert!(execution.join_window(&changed, &preparation).is_err());
        let mut changed = window;
        changed.max_output_bytes = execution.max_output_bytes + 1;
        assert!(execution.join_window(&changed, &preparation).is_err());
    }
    #[test]
    fn exact_window_roster_preserves_zero_updates_and_finite_f_values() {
        let mut value = window();
        value.validate().unwrap();
        assert_eq!(
            serde_json::to_value(&value)
                .unwrap()
                .as_object()
                .unwrap()
                .len(),
            22
        );
        value.max_optimizer_updates = 0;
        value.validate().unwrap();
        value.device = "cuda:63".into();
        value.cpu_threads = 256;
        value.interop_threads = 256;
        value.validate().unwrap();
        for device in ["cuda:00", "cuda:64", "cuda:+1", "CUDA:0"] {
            let mut wrong = value.clone();
            wrong.device = device.into();
            assert!(wrong.validate().is_err());
        }
        value.run_id = "학습".into();
        assert!(value.validate().is_err());
        let mut wrong = window();
        wrong.max_groups = 0;
        assert!(wrong.validate().is_err());
        wrong = window();
        wrong.cleanup_time_ns = ROVENT_UTILITY_MAX_CLEANUP_NS + 1;
        assert!(wrong.validate().is_err());
        let mut raw = serde_json::to_value(window()).unwrap();
        raw["tf32"] = json!(0);
        assert!(serde_json::from_value::<RoventUtilityTrainWindow>(raw).is_err());
    }
    #[test]
    fn execution_requires_null_key_and_original_object_wire_forms() {
        let value = registration();
        value.validate().unwrap();
        let mut raw = serde_json::to_value(&value).unwrap();
        assert_eq!(raw.as_object().unwrap().len(), 29);
        assert!(raw["export_request"].is_null());
        assert_eq!(
            serde_json::from_value::<RoventUtilityExecutionRegistration>(raw.clone()).unwrap(),
            value
        );
        raw.as_object_mut().unwrap().remove("export_request");
        assert!(serde_json::from_value::<RoventUtilityExecutionRegistration>(raw).is_err());
        let mut raw = serde_json::to_value(&value).unwrap();
        raw["preparation"] = json!(["/inputs/preparation.json", "30".repeat(32), 128]);
        assert!(serde_json::from_value::<RoventUtilityExecutionRegistration>(raw).is_err());
        let mut raw = serde_json::to_value(&value).unwrap();
        raw["runtime_endpoints"][0]["artifact"] =
            json!(["/deps/torch/__init__.py", "21".repeat(32), 128]);
        assert!(serde_json::from_value::<RoventUtilityExecutionRegistration>(raw).is_err());
        let mut raw = serde_json::to_value(&value).unwrap();
        raw["resources_policy"] = json!([2000, 6u64 << 30, 12u64 << 30, 64, [0, 2], "cpu"]);
        assert!(serde_json::from_value::<RoventUtilityExecutionRegistration>(raw).is_err());
        let mut raw = serde_json::to_value(&value).unwrap();
        raw["initialization_source"]["utility"] =
            json!(["/inputs/utility.json", "34".repeat(32), 128]);
        assert!(serde_json::from_value::<RoventUtilityExecutionRegistration>(raw).is_err());
        let raw = serde_json::to_string(&value).unwrap();
        let duplicate = raw.replacen(
            "\"export_request\":null",
            "\"export_request\":null,\"export_request\":null",
            1,
        );
        assert!(serde_json::from_str::<RoventUtilityExecutionRegistration>(&duplicate).is_err());
    }
    #[test]
    fn export_requires_request_and_resume_without_invented_stage_fields() {
        let mut value = registration();
        value.action = RoventUtilityExecutionAction::Export;
        value.export_request =
            RoventUtilityRequiredNullable(Some(pin("/inputs/export.json", "41")));
        assert!(value.validate().is_err());
        value.initialization_source = RoventUtilityInitializationSource::Fork {
            checkpoint: checkpoint(),
        };
        assert!(value.validate().is_err());
        value.initialization_source = RoventUtilityInitializationSource::Resume {
            checkpoint: checkpoint(),
        };
        value.validate().unwrap();
        value.export_request = RoventUtilityRequiredNullable(None);
        assert!(value.validate().is_err());
        let mut raw = serde_json::to_value(&value).unwrap();
        raw["initialization_source"]["parent_stage_id"] = json!("invented");
        assert!(serde_json::from_value::<RoventUtilityExecutionRegistration>(raw).is_err());
    }
    #[test]
    fn dataset_ack_is_required_and_not_aliased_to_checkpoint_or_commit_metadata() {
        let value = dataset();
        value.validate().unwrap();
        let mut raw = serde_json::to_value(&value).unwrap();
        assert_eq!(raw.as_object().unwrap().len(), 3);
        raw.as_object_mut().unwrap().remove("durable_ack");
        assert!(serde_json::from_value::<RoventUtilityDatasetReference>(raw).is_err());
        let mut raw = serde_json::to_value(&value).unwrap();
        raw["durable_ack"] = json!(true);
        assert!(serde_json::from_value::<RoventUtilityDatasetReference>(raw).is_err());
        let mut raw = serde_json::to_value(&value).unwrap();
        raw["commit"] = json!(true);
        assert!(serde_json::from_value::<RoventUtilityDatasetReference>(raw).is_err());
        let mut changed = value.clone();
        changed.durable_ack.path = "/other/durable-ack.json".into();
        assert!(changed.validate().is_err());
        changed = value;
        changed.durable_ack.path = changed.publication.path.clone();
        assert!(changed.validate().is_err());
        let checkpoint = checkpoint();
        checkpoint.validate().unwrap();
        let mut raw = serde_json::to_value(checkpoint).unwrap();
        assert_eq!(raw.as_object().unwrap().len(), 2);
        raw["durable_ack"] = json!(pin("/intake/checkpoint/ack.json", "16"));
        assert!(serde_json::from_value::<RoventUtilityCheckpointReference>(raw).is_err());
    }
    #[test]
    fn normalized_paths_raw_pins_and_shared_limits_are_not_coerced() {
        let mut value = registration();
        value.preparation.path = "/inputs/./preparation.json".into();
        assert!(value.validate().is_err());
        value = registration();
        value.current_dataset.publication.path = "/other/publication.json".into();
        assert!(value.validate().is_err());
        value = registration();
        value.frozen_pc_sources[1].role = "proposer".into();
        assert!(value.validate().is_err());
        value = registration();
        value.control_root = "/run/output/control".into();
        assert!(value.validate().is_err());
        value = registration();
        value.max_child_read_bytes = value.max_read_bytes + 1;
        assert!(value.validate().is_err());
        value = registration();
        value.max_artifact_files = 32_769;
        assert!(value.validate().is_err());
        value = registration();
        value.ancestor_datasets.push(dataset());
        assert!(value.validate().is_err());
        for path in ["/inputs//a", "/inputs/a/", "/inputs/../a"] {
            assert!(!normalized_absolute(Path::new(path)));
        }
    }
}
