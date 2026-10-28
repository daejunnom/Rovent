//! New explicit window/registration codec. Collection inference identity and
//! trainer executable/dependency identity are independent. Parsing these values
//! is not an execution lease and never upgrades historical evidence to native
//! or physical authority. Existing live /1 formats are not converted here.
use super::*;

pub const ROVENT_TRAIN_WINDOW_V2: &str = "rz-rovent-train-window/2";
pub const ROVENT_TRAIN_PROFILE_V2: &str = "rz-rovent-fixed-training-profile/2";
pub const ROVENT_TRAINER_DEPENDENCY_SET: &str = "rz-rovent-trainer-dependency-set/1";
pub const ROVENT_TRAINER_DEPENDENCY_SCOPE: &str = "declared_pinned_endpoints_only";
pub const ROVENT_TRAIN_PROFILE_COEFFICIENT_CODEC: &str = "coefficient-f64-bits-v1";

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventTrainerRuntimeEndpoint {
    pub name: String,
    pub artifact: RoventTrainingFilePin,
}

/// Location-independent inventory of explicitly declared endpoints. The file
/// path and current inode are checked separately by the actual process owner.
/// This digest makes no claim about transitive Torch/CUDA native mappings.
pub fn rovent_trainer_dependency_set_sha256(
    endpoints: &[RoventTrainerRuntimeEndpoint],
) -> Result<String, ArenaError> {
    check(
        !endpoints.is_empty() && endpoints.len() <= 64,
        "bounded trainer endpoint inventory",
    )?;
    let mut names = BTreeSet::new();
    let mut inventory = Vec::with_capacity(endpoints.len());
    for endpoint in endpoints {
        check(
            generation_identifier(&endpoint.name) && names.insert(endpoint.name.as_str()),
            "duplicate/invalid logical endpoint name",
        )?;
        pin_shape(&endpoint.artifact, 2 << 30)?;
        inventory.push(json!({"name":endpoint.name,"sha256":endpoint.artifact.sha256,"bytes":endpoint.artifact.bytes}));
    }
    check(
        names.contains("torch.__file__") && names.contains("torch._C.__file__"),
        "explicit Torch Python/native endpoints required",
    )?;
    inventory.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    digest_value(ROVENT_TRAINER_DEPENDENCY_SET, &inventory)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventTrainWindowV2 {
    pub schema: String,
    pub window_id: String,
    pub recipe_sha256: String,
    pub stage_id: String,
    pub collection_source_sha256: String,
    pub collection_runtime_library_sha256: String,
    pub collection_model_semantics_sha256: String,
    pub collection_model_epoch: u64,
    pub collection_context_epoch: u64,
    pub trainer_source_sha256: String,
    pub trainer_python_executable_sha256: String,
    pub trainer_dependency_set_sha256: String,
    pub trainer_dependency_scope: String,
    pub training_model_semantics_sha256: String,
    pub training_model_epoch: u64,
    pub nnuec_collection_model_sha256: String,
    pub nnuec_semantics: String,
    pub phase: String,
    pub frozen_pc_sha256: Option<String>,
    pub device: String,
    pub precision: String,
    pub tf32: bool,
    pub deterministic: bool,
    pub cpu_threads: u32,
    pub interop_threads: u32,
    pub microbatch: u32,
    pub accumulation: u32,
    pub optimizer: Value,
    pub sampling_seed: u32,
    pub training_seed: u32,
    pub loss_weights: BTreeMap<String, f64>,
    pub nnuec_loss: RoventNnuecLoss,
    #[serde(default, skip_serializing_if = "RoventRAuxiliaryPresence::is_absent")]
    pub nnuec_r_auxiliary: RoventRAuxiliaryPresence<RoventRAuxiliaryRecipe>,
    pub max_updates: u64,
    pub max_groups: u64,
    pub max_samples: u64,
    pub max_wall_ns: u64,
    pub max_output_bytes: u64,
    pub checkpoint_interval: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventNnuecLoss {
    pub loss: String,
    pub target: String,
    pub normalization: String,
    pub weight: f64,
}
impl RoventTrainWindowV2 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        if let Some(auxiliary) = self.nnuec_r_auxiliary.enabled() {
            auxiliary.validate()?;
        }
        check(
            self.schema == ROVENT_TRAIN_WINDOW_V2
                && [&self.window_id, &self.stage_id, &self.nnuec_semantics]
                    .into_iter()
                    .all(|v| generation_identifier(v))
                && [
                    &self.recipe_sha256,
                    &self.collection_source_sha256,
                    &self.collection_runtime_library_sha256,
                    &self.collection_model_semantics_sha256,
                    &self.trainer_source_sha256,
                    &self.trainer_python_executable_sha256,
                    &self.trainer_dependency_set_sha256,
                    &self.training_model_semantics_sha256,
                    &self.nnuec_collection_model_sha256,
                ]
                .into_iter()
                .all(|v| generation_sha(v))
                && self.trainer_dependency_scope == ROVENT_TRAINER_DEPENDENCY_SCOPE,
            "strict explicit collection/trainer/model identities required",
        )?;
        // The first codec supports the same registered architecture only.
        // No equality is imposed on collection ORT, Python, or initial weights.
        check(
            self.collection_model_semantics_sha256 == self.training_model_semantics_sha256,
            "different collection/training architecture needs a separately admitted compatibility policy",
        )?;
        check(
            (self.phase == "pc" && self.frozen_pc_sha256.is_none())
                || (self.phase == "v"
                    && self.frozen_pc_sha256.as_deref().is_some_and(generation_sha)),
            "explicit P/C or frozen V phase required",
        )?;
        check(
            canonical_device(&self.device)
                && self.precision == "fp32"
                && !self.tf32
                && self.deterministic
                && (1..=64).contains(&self.cpu_threads)
                && (1..=64).contains(&self.interop_threads)
                && self.microbatch == 1
                && self.accumulation == 8,
            "explicit deterministic FP32 device and registered batch/thread recipe required",
        )?;
        check(
            self.optimizer
                == json!({"name":"adamw","lr":1e-4,"betas":[0.9,0.999],"epsilon":1e-8,
            "weight_decay":0.01,"clip_norm":1.0,"schedule":"constant"}),
            "registered AdamW optimizer recipe changed",
        )?;
        check(
            self.loss_weights.len() == 4
                && HEADS[..4]
                    .iter()
                    .all(|name| self.loss_weights.contains_key(*name))
                && self
                    .loss_weights
                    .values()
                    .all(|weight| weight.is_finite() && *weight >= 0.0)
                && self.nnuec_loss.loss == "wdl_cross_entropy"
                && self.nnuec_loss.target == "same_exact_rules_anchor"
                && self.nnuec_loss.normalization == "valid_rows"
                && self.nnuec_loss.weight.is_finite()
                && self.nnuec_loss.weight > 0.0,
            "explicit finite head/NNUEC loss recipe required",
        )?;
        check(
            (1..=65_536).contains(&self.max_updates)
                && self.max_updates <= self.max_groups
                && self.max_groups <= 65_536
                && (8..=524_288).contains(&self.max_samples)
                && (1..=86_400_000_000_000).contains(&self.max_wall_ns)
                && (1..=2_u64 << 30).contains(&self.max_output_bytes)
                && (1..=250).contains(&self.checkpoint_interval),
            "finite window work/time/output/checkpoint limits required",
        )
    }
    pub fn profile_sha256(&self) -> Result<String, ArenaError> {
        self.validate()?;
        let mut value = serde_json::to_value(self).map_err(|e| invalid(e.to_string()))?;
        let object = value
            .as_object_mut()
            .ok_or_else(|| invalid("window object"))?;
        for name in [
            "window_id",
            "max_updates",
            "max_groups",
            "max_samples",
            "max_wall_ns",
            "max_output_bytes",
            "checkpoint_interval",
            "phase",
            "collection_model_epoch",
            "training_model_epoch",
            "frozen_pc_sha256",
            "nnuec_collection_model_sha256",
        ] {
            object.remove(name);
        }
        // Python and serde_json use different exponent spellings. Only the
        // explicitly declared coefficient subtrees are normalized; arbitrary
        // u64 counts/epochs must never be rounded through f64.
        normalize_coefficients(&mut value["optimizer"])?;
        normalize_coefficients(&mut value["loss_weights"])?;
        normalize_coefficients(&mut value["nnuec_loss"]["weight"])?;
        if self.nnuec_r_auxiliary.enabled().is_some() {
            normalize_coefficients(&mut value["nnuec_r_auxiliary"]["weight"])?;
        }
        digest_value(
            ROVENT_TRAIN_PROFILE_V2,
            &json!({"algorithm":ROVENT_TRAIN_PROFILE_COEFFICIENT_CODEC,"window":value}),
        )
    }
}
fn normalize_coefficients(value: &mut Value) -> Result<(), ArenaError> {
    match value {
        Value::Number(number) => {
            let coefficient = number
                .as_f64()
                .filter(|n| n.is_finite())
                .ok_or_else(|| invalid("finite profile coefficient required"))?;
            *value = Value::String(format!("{:016x}", coefficient.to_bits()));
        }
        Value::Array(values) => {
            for value in values {
                normalize_coefficients(value)?;
            }
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                normalize_coefficients(value)?;
            }
        }
        _ => (),
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoventTrainingInputSource {
    Initialize {
        pc_checkpoint: RoventTrainingFilePin,
        nnuec_checkpoint: RoventTrainingFilePin,
    },
    Resume {
        checkpoint: RoventHistoricalCheckpointRegistration,
    },
    Fork {
        checkpoint: RoventHistoricalCheckpointRegistration,
        parent_stage_id: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoventHistoricalCheckpointRegistration {
    pub root: PathBuf,
    pub publication: RoventTrainingFilePin,
    /// An exact original closure artifact is reread and the old local process
    /// scope is observed by a new OS owner. Merely pinning this file is not it.
    pub reclamation: RoventTrainingFilePin,
}
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RoventRecoveredTrainingAction {
    Train,
    Export,
    /// A current child and its IPC are checked without admitting any update.
    IpcCheck,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRecoveredTrainingRegistration {
    pub schema: String,
    pub execution_id: String,
    pub action: RoventRecoveredTrainingAction,
    pub export_request: Option<RoventTrainingFilePin>,
    pub recipe: RoventTrainingFilePin,
    pub membership: RoventTrainingFilePin,
    pub window_spec: RoventTrainingFilePin,
    pub trainer_source_sha256: String,
    pub trainer_python_executable: RoventTrainingFilePin,
    pub trainer_python_version: String,
    /// Explicit import locations are held/read back separately from endpoint
    /// content pins; no .pth processing or global sys.path fallback is allowed.
    pub dependency_directories: Vec<PathBuf>,
    pub trainer_dependency_set_sha256: String,
    pub runtime_endpoints: Vec<RoventTrainerRuntimeEndpoint>,
    pub current_dataset: RoventHistoricalDatasetRegistration,
    pub ancestor_datasets: Vec<RoventHistoricalDatasetRegistration>,
    #[serde(default, skip_serializing_if = "RoventRAuxiliaryPresence::is_absent")]
    pub r_auxiliary_dataset:
        RoventRAuxiliaryPresence<RoventHistoricalRAuxiliaryDatasetRegistration>,
    #[serde(default, skip_serializing_if = "RoventRAuxiliaryPresence::is_absent")]
    pub ancestor_r_auxiliary_datasets:
        RoventRAuxiliaryPresence<Vec<RoventHistoricalRAuxiliaryDatasetRegistration>>,
    pub initialization_source: RoventTrainingInputSource,
    pub rules_profile: RoventTrainingFilePin,
    pub expected_initial_completed: BTreeMap<String, u64>,
    pub expected_ancestor_completed: BTreeMap<String, u64>,
    pub resources_policy: RoventTrainingResourcePolicy,
    pub require_total_quota: bool,
    pub output_root: PathBuf,
    pub control_root: PathBuf,
    pub checkpoint_directory: PathBuf,
    pub receipt_path: PathBuf,
    pub max_rpc_messages: u64,
    pub max_rpc_bytes: u64,
    pub max_read_bytes: u64,
    /// Reserved from the same parent budget before starting the child. Missing
    /// final child accounting never refunds this conservative upper bound.
    pub max_child_read_bytes: u64,
    pub max_output_bytes: u64,
    /// New trainer profile only; the generic process watch still caps 4096.
    pub max_artifact_files: usize,
}
impl RoventRecoveredTrainingRegistration {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == ROVENT_RECOVERY_REGISTRATION
                && generation_identifier(&self.execution_id)
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
                    == rovent_trainer_dependency_set_sha256(&self.runtime_endpoints)?
                && self.ancestor_datasets.len() <= 256,
            "recovery registration identity/source/endpoint inventory",
        )?;
        self.resources_policy.validate()?;
        if let Some(dataset) = self.r_auxiliary_dataset.enabled() {
            dataset.validate()?;
        }
        if let Some(ancestors) = self.ancestor_r_auxiliary_datasets.enabled() {
            check(
                ancestors.len() <= 256
                    && (self.r_auxiliary_dataset.enabled().is_some()
                        || !matches!(
                            self.initialization_source,
                            RoventTrainingInputSource::Initialize { .. }
                        )),
                "R ancestry requires a bounded current R source or an explicit historical checkpoint",
            )?;
            check(
                ancestors
                    .windows(2)
                    .all(|pair| pair[0].publication.sha256 < pair[1].publication.sha256),
                "R ancestor publications must have original sorted unique pins",
            )?;
            let mut publications = BTreeSet::new();
            if let Some(current) = self.r_auxiliary_dataset.enabled() {
                publications.insert(current.publication.sha256.clone());
            }
            for ancestor in ancestors {
                ancestor.validate()?;
                check(
                    publications.insert(ancestor.publication.sha256.clone()),
                    "duplicate current/ancestor R publication",
                )?;
            }
        }
        for pin in [
            &self.recipe,
            &self.membership,
            &self.window_spec,
            &self.rules_profile,
        ] {
            pin_shape(pin, MAX_METADATA_BYTES)?;
        }
        pin_shape(&self.trainer_python_executable, 2 << 30)?;
        for counts in [
            &self.expected_initial_completed,
            &self.expected_ancestor_completed,
        ] {
            check(
                counts.len() == 3
                    && ROLES.iter().all(|role| counts.contains_key(*role))
                    && counts.values().all(|value| *value <= 65_536),
                "exact bounded role counts",
            )?;
        }
        match &self.initialization_source {
            RoventTrainingInputSource::Initialize {
                pc_checkpoint,
                nnuec_checkpoint,
            } => {
                pin_shape(pc_checkpoint, 2 << 30)?;
                pin_shape(nnuec_checkpoint, 2 << 30)?;
                check(
                    self.action != RoventRecoveredTrainingAction::Export
                        && self.expected_initial_completed.values().all(|v| *v == 0),
                    "initialize cannot export or claim trained initial updates",
                )?;
            }
            RoventTrainingInputSource::Resume { checkpoint } => checkpoint.validate()?,
            RoventTrainingInputSource::Fork {
                checkpoint,
                parent_stage_id,
            } => {
                checkpoint.validate()?;
                check(
                    generation_identifier(parent_stage_id)
                        && self.action != RoventRecoveredTrainingAction::Export
                        && self.expected_initial_completed.values().all(|v| *v == 0),
                    "fork needs explicit parent and fresh current-stage count",
                )?;
            }
        }
        match self.action {
            RoventRecoveredTrainingAction::Train | RoventRecoveredTrainingAction::IpcCheck => {
                check(
                    self.export_request.is_none(),
                    "train/IPC check cannot carry an export request",
                )?
            }
            RoventRecoveredTrainingAction::Export => {
                pin_shape(
                    self.export_request
                        .as_ref()
                        .ok_or_else(|| invalid("explicit export request"))?,
                    MAX_METADATA_BYTES,
                )?;
            }
        }
        let mut paths = BTreeSet::new();
        for root in [
            &self.output_root,
            &self.control_root,
            &self.checkpoint_directory,
        ] {
            check(
                normalized_absolute(root) && paths.insert(root),
                "distinct normalized absolute owner directories required",
            )?;
        }
        check(
            self.checkpoint_directory.starts_with(&self.output_root)
                && self.checkpoint_directory != self.output_root
                && normalized_absolute(&self.receipt_path)
                && self.receipt_path.starts_with(&self.output_root)
                && self.receipt_path != self.output_root
                && !self.output_root.starts_with(&self.control_root)
                && !self.control_root.starts_with(&self.output_root),
            "output/control/receipt ownership paths overlap",
        )?;
        check(
            (1..=2_000_000).contains(&self.max_rpc_messages)
                && (1..=MAX_TRAINER_OUTPUT_BYTES).contains(&self.max_rpc_bytes)
                && (1..=MAX_TRAINER_OUTPUT_BYTES * 260).contains(&self.max_read_bytes)
                && self.max_child_read_bytes > 0
                && self.max_child_read_bytes <= self.max_read_bytes
                && (1..=MAX_TRAINER_OUTPUT_BYTES).contains(&self.max_output_bytes)
                && (1..=32768).contains(&self.max_artifact_files),
            "finite shared RPC/read/output credits required",
        )
    }
    /// Identity-only join. Full source/package/runtime/OS and dataset Rules
    /// checks belong to the concrete current parent before any child import.
    pub fn join_window(&self, window: &RoventTrainWindowV2) -> Result<(), ArenaError> {
        self.validate()?;
        window.validate()?;
        check(
            window.nnuec_r_auxiliary.state() == self.r_auxiliary_dataset.state(),
            "window and R dataset presence differ; null/absence may not be aliased",
        )?;
        if let Some(dataset) = self.r_auxiliary_dataset.enabled() {
            check(
                dataset.collection_source_sha256 == window.collection_source_sha256
                    && dataset.collection_runtime_library_sha256
                        == window.collection_runtime_library_sha256
                    && dataset.collection_model_semantics_sha256
                        == window.collection_model_semantics_sha256
                    && dataset.collection_context_epoch == window.collection_context_epoch
                    && dataset.collection_recipe_sha256 == window.recipe_sha256
                    && dataset.collection_stage_id == window.stage_id
                    && dataset.nnuec.semantics == window.nnuec_semantics
                    && dataset.nnuec_parameter_sha256 == window.nnuec_collection_model_sha256,
                "R dataset changed original collection identities",
            )?;
        }
        check(
            window.trainer_source_sha256 == self.trainer_source_sha256
                && window.trainer_python_executable_sha256 == self.trainer_python_executable.sha256
                && window.trainer_dependency_set_sha256 == self.trainer_dependency_set_sha256
                && window.collection_source_sha256 == self.current_dataset.collection_source_sha256
                && window.collection_runtime_library_sha256
                    == self.current_dataset.collection_runtime_library_sha256
                && window.collection_model_semantics_sha256
                    == self.current_dataset.model_semantics_sha256
                && window.collection_model_epoch == self.current_dataset.model_epoch
                && window.collection_context_epoch == self.current_dataset.context_epoch
                && window.recipe_sha256 == self.current_dataset.recipe_sha256
                && window.stage_id == self.current_dataset.stage_id
                && window.device == self.resources_policy.device
                && window.max_output_bytes <= self.max_output_bytes,
            "new window changed registered collection/trainer/model/resource identities",
        )
    }
}
impl RoventHistoricalCheckpointRegistration {
    fn validate(&self) -> Result<(), ArenaError> {
        check(
            normalized_absolute(&self.root)
                && self.publication.path.starts_with(&self.root)
                && self.publication.path != self.root,
            "checkpoint publication escapes registered root",
        )?;
        pin_shape(&self.publication, MAX_METADATA_BYTES)?;
        pin_shape(&self.reclamation, MAX_METADATA_BYTES)
    }
}
fn pin_shape(pin: &RoventTrainingFilePin, maximum: u64) -> Result<(), ArenaError> {
    GenerationTrainingFilePinV3 {
        path: pin.path.clone(),
        sha256: pin.sha256.clone(),
        bytes: pin.bytes,
    }
    .validate(maximum)
}
fn normalized_absolute(path: &Path) -> bool {
    path.is_absolute()
        && path
            .components()
            .all(|part| !matches!(part, Component::CurDir | Component::ParentDir))
}
fn canonical_device(value: &str) -> bool {
    value == "cpu"
        || value.strip_prefix("cuda:").is_some_and(|n| {
            !n.is_empty()
                && n.bytes().all(|b| b.is_ascii_digit())
                && n.parse::<u32>().is_ok_and(|v| v < 16 && v.to_string() == n)
        })
}
fn canonical_python_version(value: &str) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    fn window() -> RoventTrainWindowV2 {
        RoventTrainWindowV2 {
            schema: ROVENT_TRAIN_WINDOW_V2.into(),
            window_id: "window".into(),
            recipe_sha256: "a".repeat(64),
            stage_id: "common".into(),
            collection_source_sha256: "b".repeat(64),
            collection_runtime_library_sha256: "c".repeat(64),
            collection_model_semantics_sha256: "d".repeat(64),
            collection_model_epoch: 7,
            collection_context_epoch: 11,
            trainer_source_sha256: "e".repeat(64),
            trainer_python_executable_sha256: "f".repeat(64),
            trainer_dependency_set_sha256: "0".repeat(64),
            trainer_dependency_scope: ROVENT_TRAINER_DEPENDENCY_SCOPE.into(),
            training_model_semantics_sha256: "d".repeat(64),
            training_model_epoch: 19,
            nnuec_collection_model_sha256: "1".repeat(64),
            nnuec_semantics: "nnuec-fp32".into(),
            phase: "pc".into(),
            frozen_pc_sha256: None,
            device: "cpu".into(),
            precision: "fp32".into(),
            tf32: false,
            deterministic: true,
            cpu_threads: 2,
            interop_threads: 1,
            microbatch: 1,
            accumulation: 8,
            optimizer: json!({"name":"adamw","lr":1e-4,"betas":[0.9,0.999],"epsilon":1e-8,"weight_decay":0.01,"clip_norm":1.0,"schedule":"constant"}),
            sampling_seed: 29,
            training_seed: 31,
            loss_weights: HEADS[..4]
                .iter()
                .map(|name| ((*name).into(), 1.0))
                .collect(),
            nnuec_loss: RoventNnuecLoss {
                loss: "wdl_cross_entropy".into(),
                target: "same_exact_rules_anchor".into(),
                normalization: "valid_rows".into(),
                weight: 0.5,
            },
            nnuec_r_auxiliary: RoventRAuxiliaryPresence::Absent,
            max_updates: 4,
            max_groups: 4,
            max_samples: 32,
            max_wall_ns: 1_000_000,
            max_output_bytes: 1 << 20,
            checkpoint_interval: 4,
        }
    }
    #[test]
    fn collection_ort_and_trainer_python_are_distinct_valid_domains() {
        let spec = window();
        assert_ne!(
            spec.collection_runtime_library_sha256,
            spec.trainer_python_executable_sha256
        );
        spec.validate().unwrap();
        let mut changed = spec.clone();
        changed.trainer_python_executable_sha256 = "2".repeat(64);
        assert_ne!(
            spec.profile_sha256().unwrap(),
            changed.profile_sha256().unwrap()
        );
        changed = spec.clone();
        changed.collection_runtime_library_sha256 = "3".repeat(64);
        assert_ne!(
            spec.profile_sha256().unwrap(),
            changed.profile_sha256().unwrap()
        );
    }
    #[test]
    fn coefficient_codec_avoids_json_exponent_and_integer_spelling() {
        let mut scientific = json!({"epsilon":1e-8,"unity":1.0});
        let mut decimal =
            serde_json::from_str::<Value>(r#"{"epsilon":0.00000001,"unity":1}"#).unwrap();
        normalize_coefficients(&mut scientific).unwrap();
        normalize_coefficients(&mut decimal).unwrap();
        assert_eq!(scientific, decimal);
        assert_eq!(scientific["epsilon"], "3e45798ee2308c3a");
        assert_eq!(scientific["unity"], "3ff0000000000000");
    }
    #[test]
    fn renewal_preserves_profile_but_actual_coefficient_changes_do_not() {
        let spec = window();
        let mut renewed = spec.clone();
        renewed.window_id = "next-window".into();
        renewed.max_wall_ns *= 2;
        renewed.max_updates = 8;
        renewed.max_groups = 8;
        renewed.max_samples = 64;
        renewed.training_model_epoch += 1;
        assert_eq!(
            spec.profile_sha256().unwrap(),
            renewed.profile_sha256().unwrap()
        );
        renewed.nnuec_loss.weight = 0.25;
        assert_ne!(
            spec.profile_sha256().unwrap(),
            renewed.profile_sha256().unwrap()
        );
    }
    #[test]
    fn explicit_window_rejects_legacy_or_unregistered_semantics() {
        let mut spec = window();
        spec.schema = "rz-rovent-train-window/1".into();
        assert!(spec.validate().is_err());
        spec = window();
        spec.training_model_semantics_sha256 = "3".repeat(64);
        assert!(spec.validate().is_err());
        spec = window();
        spec.nnuec_loss.weight = 0.0;
        assert!(spec.validate().is_err());
        spec = window();
        spec.device = "cuda:00".into();
        assert!(spec.validate().is_err());
    }
    #[test]
    fn dependency_inventory_is_order_independent_and_content_bound() {
        let endpoint = |name: &str, value: &str| RoventTrainerRuntimeEndpoint {
            name: name.into(),
            artifact: RoventTrainingFilePin {
                path: PathBuf::from(format!("/registered/{}", name)),
                sha256: value.repeat(64),
                bytes: 64,
            },
        };
        let a = endpoint("torch.__file__", "a");
        let b = endpoint("torch._C.__file__", "b");
        let first = rovent_trainer_dependency_set_sha256(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(
            first,
            rovent_trainer_dependency_set_sha256(&[b.clone(), a.clone()]).unwrap()
        );
        let mut changed = b.clone();
        changed.artifact.sha256 = "c".repeat(64);
        assert_ne!(
            first,
            rovent_trainer_dependency_set_sha256(&[a.clone(), changed]).unwrap()
        );
        assert!(rovent_trainer_dependency_set_sha256(&[a.clone(), a]).is_err());
    }
}
