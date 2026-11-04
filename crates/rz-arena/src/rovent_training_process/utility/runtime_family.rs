//! Closed learner-family dispatch over original declarations and retained owners.
//! Common field views are not serialized and cannot issue a dataset, peer,
//! checkpoint, execution lease or old preparation envelope.
use super::dataset::{UtilityContextAuthority, VerifiedUtilityDataset};
use super::execution::{
    RoventUtilityCheckpointReference, RoventUtilityDatasetReference, RoventUtilityExecutionAction,
    RoventUtilityFrozenPcSource, RoventUtilityInitializationSource, RoventUtilityRequiredNullable,
};
use super::execution_v2::RoventUtilityExecutionRegistrationV2;
use super::intervention_dataset::{
    RoventUtilityInterventionDatasetRejected, UtilityInterventionContextAuthority,
    VerifiedUtilityInterventionDataset,
};
use super::intervention_execution::RoventUtilityInterventionExecutionRegistration;
use super::intervention_window::RoventUtilityInterventionTrainWindow;
use super::ledger::RestoredUtilityState;
use super::numerical_parts::ValidatedUtilityNumericalParts;
use super::window_v2::RoventUtilityTrainWindowV2;
use super::{
    ancestry::{RoventUtilityInterventionAncestorsRejected, VerifiedUtilityAncestors},
    checkpoint, intervention_checkpoint,
};
use crate::rovent_training_process::recovery::{
    RoventRecoveryReadBudget, RoventTrainerRuntimeEndpoint,
};
use crate::{
    ArenaError,
    rovent_training_process::{RoventTrainingFilePin, RoventTrainingResourcePolicy},
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::{collections::BTreeSet, ops::Deref, path::PathBuf, sync::Arc};
fn invalid(reason: &str) -> ArenaError {
    ArenaError::Integrity(format!("utility family: {reason}"))
}

/// The original metadata family, never inferred from a missing field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum UtilityRuntimeFamily {
    CohortV1,
    InterventionV2,
}
impl UtilityRuntimeFamily {
    pub(super) fn numerical(
        self,
        value: &Value,
    ) -> Result<ValidatedUtilityNumericalParts, ArenaError> {
        match self {
            Self::CohortV1 => ValidatedUtilityNumericalParts::from_cohort(value),
            Self::InterventionV2 => ValidatedUtilityNumericalParts::from_intervention(value),
        }
    }
    /// Protocol wrappers have separate domains. Shared sampler, tensor math,
    /// source closure and cost codecs deliberately keep their original domains.
    pub(super) fn domain(self, legacy: &'static str) -> &'static str {
        if self == Self::CohortV1 {
            return legacy;
        }
        match legacy {
            "rz-rovent-utility-derived-training-state/1" => {
                "rz-rovent-utility-derived-training-state/2"
            }
            "rz-rovent-utility-initial-ledger-projection/1" => {
                "rz-rovent-utility-initial-ledger-projection/2"
            }
            "rz-rovent-utility-learner-completion/1" => "rz-rovent-utility-learner-completion/2",
            "rz-rovent-utility-learner-update/1" => "rz-rovent-utility-learner-update/2",
            "rz-rovent-utility-parent-checkpoint-projection/1" => {
                "rz-rovent-utility-parent-checkpoint-projection/2"
            }
            "rz-rovent-utility-parent-ended-projection/1" => {
                "rz-rovent-utility-parent-ended-projection/2"
            }
            "rz-rovent-utility-parent-ledger-observation/1" => {
                "rz-rovent-utility-parent-ledger-observation/2"
            }
            "rz-rovent-utility-trainer-begin/1" => "rz-rovent-utility-trainer-begin/2",
            "rz-rovent-utility-trainer-checkpoint/1" => "rz-rovent-utility-trainer-checkpoint/2",
            "rz-rovent-utility-trainer-consume/1" => "rz-rovent-utility-trainer-consume/2",
            "rz-rovent-utility-trainer-end/1" => "rz-rovent-utility-trainer-end/2",
            "rz-rovent-utility-trainer-export/1" => "rz-rovent-utility-trainer-export/2",
            "rz-rovent-utility-trainer-resume/1" => "rz-rovent-utility-trainer-resume/2",
            "rz-rovent-utility-trainer-update/1" => "rz-rovent-utility-trainer-update/2",
            "rz-rovent-utility-ledger-event/1" => "rz-rovent-utility-ledger-event/2",
            "rz-rovent-utility-parent-end/1" => "rz-rovent-utility-parent-end/2",
            "rz-rovent-utility-parent-outcome/1" => "rz-rovent-utility-parent-outcome/2",
            "rz-rovent-utility-trainer-ack/1" => "rz-rovent-utility-trainer-ack/2",
            "rz-rovent-utility-trainer-inspection/1" => "rz-rovent-utility-trainer-inspection/2",
            "rz-rovent-utility-trainer-result/1" => "rz-rovent-utility-trainer-result/2",
            "rz-rovent-utility-trainer-rpc/1" => "rz-rovent-utility-trainer-rpc/2",
            "rz-rovent-utility-trainer-startup/1" => "rz-rovent-utility-trainer-startup/2",
            "rz-rovent-utility-export-request/1" => "rz-rovent-utility-export-request/2",
            _ => legacy,
        }
    }
}
// Validation is part of each original decoder arm. An untagged content buffer
// retains duplicate entries; no lossy Value map is used to choose a schema.
macro_rules! validated_arm {
    ($name:ident,$ty:ty) => {
        struct $name($ty);
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let v = <$ty>::deserialize(d)?;
                v.validate().map_err(serde::de::Error::custom)?;
                Ok(Self(v))
            }
        }
    };
}
validated_arm!(CohortRegistration, RoventUtilityExecutionRegistrationV2);
validated_arm!(
    InterventionRegistration,
    RoventUtilityInterventionExecutionRegistration
);
validated_arm!(CohortWindow, RoventUtilityTrainWindowV2);
validated_arm!(InterventionWindow, RoventUtilityInterventionTrainWindow);

#[derive(Clone)]
enum RegistrationSource {
    Cohort(RoventUtilityExecutionRegistrationV2),
    Intervention(RoventUtilityInterventionExecutionRegistration),
}
#[derive(Clone)]
pub(super) struct RuntimeUtilityRegistrationFields {
    pub(super) execution_id: String,
    pub(super) action: RoventUtilityExecutionAction,
    pub(super) export_request: RoventUtilityRequiredNullable<RoventTrainingFilePin>,
    pub(super) preparation: RoventTrainingFilePin,
    pub(super) window_spec: RoventTrainingFilePin,
    pub(super) trainer_source_sha256: String,
    pub(super) trainer_python_executable: RoventTrainingFilePin,
    pub(super) trainer_python_version: String,
    pub(super) dependency_directories: Vec<PathBuf>,
    pub(super) runtime_endpoints: Vec<RoventTrainerRuntimeEndpoint>,
    pub(super) current_dataset: RoventUtilityDatasetReference,
    pub(super) initialization_source: RoventUtilityInitializationSource,
    pub(super) frozen_nnuec_base: RoventTrainingFilePin,
    pub(super) frozen_pc_sources: Vec<RoventUtilityFrozenPcSource>,
    pub(super) resources_policy: RoventTrainingResourcePolicy,
    pub(super) require_total_quota: bool,
    pub(super) output_root: PathBuf,
    pub(super) control_root: PathBuf,
    pub(super) checkpoint_directory: PathBuf,
    pub(super) receipt_path: PathBuf,
    pub(super) max_rpc_messages: u64,
    pub(super) max_rpc_bytes: u64,
    pub(super) max_read_bytes: u64,
    pub(super) max_child_read_bytes: u64,
    pub(super) max_output_bytes: u64,
    pub(super) max_artifact_files: usize,
}
#[derive(Clone)]
pub(super) struct RuntimeUtilityRegistration {
    source: RegistrationSource,
    fields: RuntimeUtilityRegistrationFields,
}
impl From<RoventUtilityExecutionRegistrationV2> for RuntimeUtilityRegistration {
    fn from(v: RoventUtilityExecutionRegistrationV2) -> Self {
        let fields = RuntimeUtilityRegistrationFields {
            execution_id: v.execution_id.clone(),
            action: v.action,
            export_request: v.export_request.clone(),
            preparation: v.preparation.clone(),
            window_spec: v.window_spec.clone(),
            trainer_source_sha256: v.trainer_source_sha256.clone(),
            trainer_python_executable: v.trainer_python_executable.clone(),
            trainer_python_version: v.trainer_python_version.clone(),
            dependency_directories: v.dependency_directories.clone(),
            runtime_endpoints: v.runtime_endpoints.clone(),
            current_dataset: v.current_dataset.clone(),
            initialization_source: v.initialization_source.clone(),
            frozen_nnuec_base: v.frozen_nnuec_base.clone(),
            frozen_pc_sources: v.frozen_pc_sources.clone(),
            resources_policy: v.resources_policy.clone(),
            require_total_quota: v.require_total_quota,
            output_root: v.output_root.clone(),
            control_root: v.control_root.clone(),
            checkpoint_directory: v.checkpoint_directory.clone(),
            receipt_path: v.receipt_path.clone(),
            max_rpc_messages: v.max_rpc_messages,
            max_rpc_bytes: v.max_rpc_bytes,
            max_read_bytes: v.max_read_bytes,
            max_child_read_bytes: v.max_child_read_bytes,
            max_output_bytes: v.max_output_bytes,
            max_artifact_files: v.max_artifact_files,
        };
        Self {
            source: RegistrationSource::Cohort(v),
            fields,
        }
    }
}
impl From<RoventUtilityInterventionExecutionRegistration> for RuntimeUtilityRegistration {
    fn from(v: RoventUtilityInterventionExecutionRegistration) -> Self {
        let fields = RuntimeUtilityRegistrationFields {
            execution_id: v.execution_id.clone(),
            action: v.action,
            export_request: v.export_request.clone(),
            preparation: v.preparation.clone(),
            window_spec: v.window_spec.clone(),
            trainer_source_sha256: v.trainer_source_sha256.clone(),
            trainer_python_executable: v.trainer_python_executable.clone(),
            trainer_python_version: v.trainer_python_version.clone(),
            dependency_directories: v.dependency_directories.clone(),
            runtime_endpoints: v.runtime_endpoints.clone(),
            current_dataset: v.current_dataset.clone(),
            initialization_source: v.initialization_source.clone(),
            frozen_nnuec_base: v.frozen_nnuec_base.clone(),
            frozen_pc_sources: v.frozen_pc_sources.clone(),
            resources_policy: v.resources_policy.clone(),
            require_total_quota: v.require_total_quota,
            output_root: v.output_root.clone(),
            control_root: v.control_root.clone(),
            checkpoint_directory: v.checkpoint_directory.clone(),
            receipt_path: v.receipt_path.clone(),
            max_rpc_messages: v.max_rpc_messages,
            max_rpc_bytes: v.max_rpc_bytes,
            max_read_bytes: v.max_read_bytes,
            max_child_read_bytes: v.max_child_read_bytes,
            max_output_bytes: v.max_output_bytes,
            max_artifact_files: v.max_artifact_files,
        };
        Self {
            source: RegistrationSource::Intervention(v),
            fields,
        }
    }
}
impl Deref for RuntimeUtilityRegistration {
    type Target = RuntimeUtilityRegistrationFields;
    fn deref(&self) -> &Self::Target {
        &self.fields
    }
}
impl Serialize for RuntimeUtilityRegistration {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match &self.source {
            RegistrationSource::Cohort(v) => v.serialize(s),
            RegistrationSource::Intervention(v) => v.serialize(s),
        }
    }
}
impl<'de> Deserialize<'de> for RuntimeUtilityRegistration {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Arms {
            Cohort(CohortRegistration),
            Intervention(InterventionRegistration),
        }
        Ok(match Arms::deserialize(d)? {
            Arms::Cohort(v) => v.0.into(),
            Arms::Intervention(v) => v.0.into(),
        })
    }
}
impl RuntimeUtilityRegistration {
    pub(super) fn family(&self) -> UtilityRuntimeFamily {
        match &self.source {
            RegistrationSource::Cohort(_) => UtilityRuntimeFamily::CohortV1,
            RegistrationSource::Intervention(_) => UtilityRuntimeFamily::InterventionV2,
        }
    }
    pub(super) fn as_cohort(&self) -> Result<&RoventUtilityExecutionRegistrationV2, ArenaError> {
        match &self.source {
            RegistrationSource::Cohort(v) => Ok(v),
            _ => Err(invalid("cohort registration required")),
        }
    }
    pub(super) fn as_intervention(
        &self,
    ) -> Result<&RoventUtilityInterventionExecutionRegistration, ArenaError> {
        match &self.source {
            RegistrationSource::Intervention(v) => Ok(v),
            _ => Err(invalid("intervention registration required")),
        }
    }
    pub(super) fn validate(&self) -> Result<(), ArenaError> {
        match &self.source {
            RegistrationSource::Cohort(v) => v.validate(),
            RegistrationSource::Intervention(v) => v.validate(),
        }
    }
}

#[derive(Clone)]
enum WindowSource {
    Cohort(RoventUtilityTrainWindowV2),
    Intervention(RoventUtilityInterventionTrainWindow),
}
#[derive(Clone)]
pub(super) struct RuntimeUtilityWindowFields {
    pub(super) window_id: String,
    pub(super) run_id: String,
    pub(super) preparation_sha256: String,
    pub(super) dataset_sha256: String,
    pub(super) device: String,
    pub(super) cpu_threads: u32,
    pub(super) interop_threads: u32,
    pub(super) training_seed: u32,
    pub(super) max_optimizer_updates: u64,
    pub(super) max_wall_time_ns: u64,
    pub(super) cleanup_time_ns: u64,
    pub(super) max_output_bytes: u64,
    pub(super) max_groups: u32,
    pub(super) max_contexts: u32,
    pub(super) checkpoint_reserve_ns: u64,
}
#[derive(Clone)]
pub(super) struct RuntimeUtilityWindow {
    source: WindowSource,
    fields: RuntimeUtilityWindowFields,
}
impl From<RoventUtilityTrainWindowV2> for RuntimeUtilityWindow {
    fn from(v: RoventUtilityTrainWindowV2) -> Self {
        let fields = RuntimeUtilityWindowFields {
            window_id: v.window_id.clone(),
            run_id: v.run_id.clone(),
            preparation_sha256: v.preparation_sha256.clone(),
            dataset_sha256: v.dataset_sha256.clone(),
            device: v.device.clone(),
            cpu_threads: v.cpu_threads,
            interop_threads: v.interop_threads,
            training_seed: v.training_seed,
            max_optimizer_updates: v.max_optimizer_updates,
            max_wall_time_ns: v.max_wall_time_ns,
            cleanup_time_ns: v.cleanup_time_ns,
            max_output_bytes: v.max_output_bytes,
            max_groups: v.max_groups,
            max_contexts: v.max_contexts,
            checkpoint_reserve_ns: v.checkpoint_reserve_ns,
        };
        Self {
            source: WindowSource::Cohort(v),
            fields,
        }
    }
}
impl From<RoventUtilityInterventionTrainWindow> for RuntimeUtilityWindow {
    fn from(v: RoventUtilityInterventionTrainWindow) -> Self {
        let fields = RuntimeUtilityWindowFields {
            window_id: v.window_id.clone(),
            run_id: v.run_id.clone(),
            preparation_sha256: v.preparation_sha256.clone(),
            dataset_sha256: v.dataset_sha256.clone(),
            device: v.device.clone(),
            cpu_threads: v.cpu_threads,
            interop_threads: v.interop_threads,
            training_seed: v.training_seed,
            max_optimizer_updates: v.max_optimizer_updates,
            max_wall_time_ns: v.max_wall_time_ns,
            cleanup_time_ns: v.cleanup_time_ns,
            max_output_bytes: v.max_output_bytes,
            max_groups: v.max_groups,
            max_contexts: v.max_contexts,
            checkpoint_reserve_ns: v.checkpoint_reserve_ns,
        };
        Self {
            source: WindowSource::Intervention(v),
            fields,
        }
    }
}
impl Deref for RuntimeUtilityWindow {
    type Target = RuntimeUtilityWindowFields;
    fn deref(&self) -> &Self::Target {
        &self.fields
    }
}
impl Serialize for RuntimeUtilityWindow {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match &self.source {
            WindowSource::Cohort(v) => v.serialize(s),
            WindowSource::Intervention(v) => v.serialize(s),
        }
    }
}
impl<'de> Deserialize<'de> for RuntimeUtilityWindow {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Arms {
            Cohort(CohortWindow),
            Intervention(InterventionWindow),
        }
        Ok(match Arms::deserialize(d)? {
            Arms::Cohort(v) => v.0.into(),
            Arms::Intervention(v) => v.0.into(),
        })
    }
}
impl RuntimeUtilityWindow {
    pub(super) fn family(&self) -> UtilityRuntimeFamily {
        match &self.source {
            WindowSource::Cohort(_) => UtilityRuntimeFamily::CohortV1,
            WindowSource::Intervention(_) => UtilityRuntimeFamily::InterventionV2,
        }
    }
    pub(super) fn as_cohort(&self) -> Result<&RoventUtilityTrainWindowV2, ArenaError> {
        match &self.source {
            WindowSource::Cohort(v) => Ok(v),
            _ => Err(invalid("cohort window required")),
        }
    }
    pub(super) fn as_intervention(
        &self,
    ) -> Result<&RoventUtilityInterventionTrainWindow, ArenaError> {
        match &self.source {
            WindowSource::Intervention(v) => Ok(v),
            _ => Err(invalid("intervention window required")),
        }
    }
    pub(super) fn validate(&self) -> Result<(), ArenaError> {
        match &self.source {
            WindowSource::Cohort(v) => v.validate(),
            WindowSource::Intervention(v) => v.validate(),
        }
    }
    pub(super) fn canonical_sha256(&self) -> Result<String, ArenaError> {
        match &self.source {
            WindowSource::Cohort(v) => v.canonical_sha256(),
            WindowSource::Intervention(v) => v.canonical_sha256(),
        }
    }
}
impl RuntimeUtilityWindow {
    pub(super) fn join_cohort_preparation(&self, p: &Value) -> Result<(), ArenaError> {
        match &self.source {
            WindowSource::Cohort(v) => v.join_cohort_preparation(p),
            WindowSource::Intervention(v) => v.join_cohort_preparation(p),
        }
    }
}
impl RuntimeUtilityRegistration {
    pub(super) fn join_cohort_window(
        &self,
        w: &RuntimeUtilityWindow,
        p: &Value,
    ) -> Result<(), ArenaError> {
        match &self.source {
            RegistrationSource::Cohort(v) => v.join_cohort_window(w.as_cohort()?, p),
            RegistrationSource::Intervention(v) => v.join_cohort_window(w.as_intervention()?, p),
        }
    }
    pub(super) fn open_ancestors(
        &self,
        b: &Arc<RoventRecoveryReadBudget>,
    ) -> Result<VerifiedUtilityAncestors, RuntimeUtilitySourceFailure> {
        match &self.source {
            RegistrationSource::Cohort(v) => {
                VerifiedUtilityAncestors::open(&v.ancestor_datasets, b)
                    .map_err(RuntimeUtilitySourceFailure::Metadata)
            }
            RegistrationSource::Intervention(v) => {
                VerifiedUtilityAncestors::open_intervention(&v.ancestor_datasets, b)
                    .map_err(RuntimeUtilitySourceFailure::Ancestors)
            }
        }
    }
    pub(super) fn open_checkpoint(
        &self,
        r: &RoventUtilityCheckpointReference,
        refs: &[RoventUtilityDatasetReference],
        b: &Arc<RoventRecoveryReadBudget>,
    ) -> Result<RuntimeUtilityCheckpointSource, RuntimeUtilitySourceFailure> {
        match &self.source {
            RegistrationSource::Cohort(v) => {
                checkpoint::VerifiedUtilityCheckpointSource::open(r, refs, &v.ancestor_datasets, b)
                    .map(Arc::new)
                    .map(RuntimeUtilityCheckpointSource::Cohort)
                    .map_err(RuntimeUtilitySourceFailure::Metadata)
            }
            RegistrationSource::Intervention(v) => {
                intervention_checkpoint::VerifiedUtilityCheckpointSource::open(
                    r,
                    refs,
                    &v.ancestor_datasets,
                    b,
                )
                .map(Arc::new)
                .map(RuntimeUtilityCheckpointSource::Intervention)
                .map_err(RuntimeUtilitySourceFailure::Checkpoint)
            }
        }
    }
}

/// Only original retained verified owners enter this enum. JSON cannot create it.
#[derive(Clone)]
pub(super) enum RuntimeUtilityDataset {
    Cohort(Arc<VerifiedUtilityDataset>),
    Intervention(Arc<VerifiedUtilityInterventionDataset>),
}
impl From<Arc<VerifiedUtilityDataset>> for RuntimeUtilityDataset {
    fn from(v: Arc<VerifiedUtilityDataset>) -> Self {
        Self::Cohort(v)
    }
}
impl From<Arc<VerifiedUtilityInterventionDataset>> for RuntimeUtilityDataset {
    fn from(v: Arc<VerifiedUtilityInterventionDataset>) -> Self {
        Self::Intervention(v)
    }
}
pub(super) enum RuntimeUtilityDatasetFailure {
    Cohort(ArenaError),
    Intervention(Box<RoventUtilityInterventionDatasetRejected>),
}
impl RuntimeUtilityDatasetFailure {
    pub(super) fn error(&self) -> &ArenaError {
        match self {
            Self::Cohort(v) => v,
            Self::Intervention(v) => &v.error,
        }
    }
}
impl RuntimeUtilityDataset {
    pub(super) fn open(
        r: &RuntimeUtilityRegistration,
        p: &Value,
        b: &Arc<RoventRecoveryReadBudget>,
    ) -> Result<Self, RuntimeUtilityDatasetFailure> {
        match &r.source {
            RegistrationSource::Cohort(v) => VerifiedUtilityDataset::open(&v.current_dataset, p, b)
                .map(Arc::new)
                .map(Self::Cohort)
                .map_err(RuntimeUtilityDatasetFailure::Cohort),
            RegistrationSource::Intervention(v) => {
                VerifiedUtilityInterventionDataset::open_retaining(&v.current_dataset, p, b)
                    .map(Arc::new)
                    .map(Self::Intervention)
                    .map_err(RuntimeUtilityDatasetFailure::Intervention)
            }
        }
    }
    pub(super) fn family(&self) -> UtilityRuntimeFamily {
        match self {
            Self::Cohort(_) => UtilityRuntimeFamily::CohortV1,
            Self::Intervention(_) => UtilityRuntimeFamily::InterventionV2,
        }
    }
    pub(super) fn as_cohort(&self) -> Result<&Arc<VerifiedUtilityDataset>, ArenaError> {
        match self {
            Self::Cohort(v) => Ok(v),
            _ => Err(invalid("cohort dataset required")),
        }
    }
    pub(super) fn as_intervention(
        &self,
    ) -> Result<&Arc<VerifiedUtilityInterventionDataset>, ArenaError> {
        match self {
            Self::Intervention(v) => Ok(v),
            _ => Err(invalid("intervention dataset required")),
        }
    }
    pub(super) fn validate_execution_sources(
        &self,
        r: &RuntimeUtilityRegistration,
    ) -> Result<(), ArenaError> {
        match self {
            Self::Cohort(v) => v.validate_execution_sources_v2(r.as_cohort()?),
            Self::Intervention(v) => v.validate_execution_sources_v3(r.as_intervention()?),
        }
    }
    pub(super) fn admit_ancestry(
        &self,
        a: &VerifiedUtilityAncestors,
        required: &BTreeSet<String>,
    ) -> Result<(), ArenaError> {
        match self {
            Self::Cohort(v) => v.admit_ancestry(a, required),
            Self::Intervention(v) => v.admit_ancestry(a, required),
        }
    }
    pub(super) fn validate_context_projection(
        &self,
        p: &Value,
    ) -> Result<RuntimeUtilityContext<'_>, ArenaError> {
        match self {
            Self::Cohort(v) => v
                .validate_context_projection(p)
                .map(RuntimeUtilityContext::Cohort),
            Self::Intervention(v) => v
                .validate_context_projection(p)
                .map(RuntimeUtilityContext::Intervention),
        }
    }
    pub(super) fn revalidate(&self, b: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        match self {
            Self::Cohort(v) => v.revalidate(b),
            Self::Intervention(v) => v.revalidate(b),
        }
    }
    pub(super) fn observation(&self) -> Value {
        match self {
            Self::Cohort(v) => v.observation(),
            Self::Intervention(v) => v.observation(),
        }
    }
    pub(super) fn dataset_sha256(&self) -> &str {
        match self {
            Self::Cohort(v) => v.dataset_sha256(),
            Self::Intervention(v) => v.dataset_sha256(),
        }
    }
    pub(super) fn pool_sha256(&self) -> &str {
        match self {
            Self::Cohort(v) => v.pool_sha256(),
            Self::Intervention(v) => v.pool_sha256(),
        }
    }
    pub(super) fn context_ids(&self) -> &[String] {
        match self {
            Self::Cohort(v) => v.context_ids(),
            Self::Intervention(v) => v.context_ids(),
        }
    }
}
pub(super) enum RuntimeUtilityContext<'a> {
    Cohort(&'a UtilityContextAuthority),
    Intervention(&'a UtilityInterventionContextAuthority),
}
impl RuntimeUtilityContext<'_> {
    pub(super) fn sha256(&self) -> &str {
        match self {
            Self::Cohort(v) => v.sha256(),
            Self::Intervention(v) => v.sha256(),
        }
    }
    pub(super) fn source_projection(&self) -> &Value {
        match self {
            Self::Cohort(v) => v.source_projection(),
            Self::Intervention(v) => v.source_projection(),
        }
    }
    pub(super) fn unique_known_pairs(&self) -> u64 {
        match self {
            Self::Cohort(v) => v.unique_known_pairs(),
            Self::Intervention(v) => v.unique_known_pairs(),
        }
    }
    pub(super) fn pair_sha256s(&self) -> &[String] {
        match self {
            Self::Cohort(v) => v.pair_sha256s(),
            Self::Intervention(v) => v.pair_sha256s(),
        }
    }
}
#[derive(Clone)]
pub(super) enum RuntimeUtilityCheckpointSource {
    Cohort(Arc<checkpoint::VerifiedUtilityCheckpointSource>),
    Intervention(Arc<intervention_checkpoint::VerifiedUtilityCheckpointSource>),
}
impl RuntimeUtilityCheckpointSource {
    pub(super) fn as_cohort(
        &self,
    ) -> Result<&Arc<checkpoint::VerifiedUtilityCheckpointSource>, ArenaError> {
        match self {
            Self::Cohort(v) => Ok(v),
            _ => Err(invalid("cohort checkpoint source required")),
        }
    }
    pub(super) fn as_intervention(
        &self,
    ) -> Result<&Arc<intervention_checkpoint::VerifiedUtilityCheckpointSource>, ArenaError> {
        match self {
            Self::Intervention(v) => Ok(v),
            _ => Err(invalid("intervention checkpoint source required")),
        }
    }
    pub(super) fn window(&self) -> RuntimeUtilityWindow {
        match self {
            Self::Cohort(v) => v.window().clone().into(),
            Self::Intervention(v) => v.window().clone().into(),
        }
    }
    pub(super) fn dataset(&self) -> RuntimeUtilityDataset {
        match self {
            Self::Cohort(v) => Arc::clone(v.dataset()).into(),
            Self::Intervention(v) => Arc::clone(v.dataset()).into(),
        }
    }
    pub(super) fn datasets(&self) -> Vec<RuntimeUtilityDataset> {
        match self {
            Self::Cohort(v) => v.datasets().into_iter().map(Into::into).collect(),
            Self::Intervention(v) => v.datasets().into_iter().map(Into::into).collect(),
        }
    }
    pub(super) fn admit_old_run_reclamation(
        &self,
        b: Arc<RoventRecoveryReadBudget>,
    ) -> Result<RuntimeUtilityOldRunReclamation, ArenaError> {
        match self {
            Self::Cohort(v) => v
                .admit_old_run_reclamation(b)
                .map(RuntimeUtilityOldRunReclamation::Cohort),
            Self::Intervention(v) => v
                .admit_old_run_reclamation(b)
                .map(RuntimeUtilityOldRunReclamation::Intervention),
        }
    }
    pub(super) fn preparation(&self) -> &Value {
        match self {
            Self::Cohort(v) => v.preparation(),
            Self::Intervention(v) => v.preparation(),
        }
    }
    pub(super) fn restored(&self) -> Result<RestoredUtilityState, ArenaError> {
        match self {
            Self::Cohort(v) => v.restored(),
            Self::Intervention(v) => v.restored(),
        }
    }
    pub(super) fn required_publication_sha256s(&self) -> BTreeSet<String> {
        match self {
            Self::Cohort(v) => v.required_publication_sha256s(),
            Self::Intervention(v) => v.required_publication_sha256s(),
        }
    }
    pub(super) fn revalidate(&self, b: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        match self {
            Self::Cohort(v) => v.revalidate(b),
            Self::Intervention(v) => v.revalidate(b),
        }
    }
}
pub(super) enum RuntimeUtilityOldRunReclamation {
    Cohort(checkpoint::VerifiedUtilityOldRunReclamation),
    Intervention(intervention_checkpoint::VerifiedUtilityOldRunReclamation),
}
impl RuntimeUtilityOldRunReclamation {
    pub(super) fn revalidate(&self) -> Result<(), ArenaError> {
        match self {
            Self::Cohort(v) => v.revalidate(),
            Self::Intervention(v) => v.revalidate(),
        }
    }
}
pub(super) enum RuntimePublishedUtilityCheckpointArchive {
    Cohort(checkpoint::PublishedUtilityCheckpointArchive),
    Intervention(intervention_checkpoint::PublishedUtilityCheckpointArchive),
}
impl RuntimePublishedUtilityCheckpointArchive {
    pub(super) fn as_cohort(
        &self,
    ) -> Result<&checkpoint::PublishedUtilityCheckpointArchive, ArenaError> {
        match self {
            Self::Cohort(v) => Ok(v),
            _ => Err(invalid("cohort published archive required")),
        }
    }
    pub(super) fn as_intervention(
        &self,
    ) -> Result<&intervention_checkpoint::PublishedUtilityCheckpointArchive, ArenaError> {
        match self {
            Self::Intervention(v) => Ok(v),
            _ => Err(invalid("intervention published archive required")),
        }
    }
    pub(super) fn reserved_output_bytes(&self) -> u64 {
        match self {
            Self::Cohort(v) => v.reserved_output_bytes(),
            Self::Intervention(v) => v.reserved_output_bytes(),
        }
    }
    pub(super) fn publication_pin(&self) -> &RoventTrainingFilePin {
        match self {
            Self::Cohort(v) => v.publication_pin(),
            Self::Intervention(v) => v.publication_pin(),
        }
    }
    pub(super) fn revalidate(&self, b: &RoventRecoveryReadBudget) -> Result<(), ArenaError> {
        match self {
            Self::Cohort(v) => v.revalidate(b),
            Self::Intervention(v) => v.revalidate(b),
        }
    }
}
pub(super) enum RuntimeUtilityCheckpointArchiveFailure {
    Cohort(Box<checkpoint::UtilityCheckpointArchiveFailure>),
    Intervention(Box<intervention_checkpoint::UtilityCheckpointArchiveFailure>),
}
impl RuntimeUtilityCheckpointArchiveFailure {
    pub(super) fn observation(&self) -> Value {
        match self {
            Self::Cohort(v) => v.observation(),
            Self::Intervention(v) => v.observation(),
        }
    }
}
pub(super) fn publish_current(
    source: &super::runtime::ActualUtilityCheckpointBoundary<'_>,
) -> Result<RuntimePublishedUtilityCheckpointArchive, RuntimeUtilityCheckpointArchiveFailure> {
    match source.window().family() {
        UtilityRuntimeFamily::CohortV1 => checkpoint::publish_current_utility_checkpoint(source)
            .map(RuntimePublishedUtilityCheckpointArchive::Cohort)
            .map_err(RuntimeUtilityCheckpointArchiveFailure::Cohort),
        UtilityRuntimeFamily::InterventionV2 => {
            intervention_checkpoint::publish_current_utility_checkpoint(source)
                .map(RuntimePublishedUtilityCheckpointArchive::Intervention)
                .map_err(RuntimeUtilityCheckpointArchiveFailure::Intervention)
        }
    }
}
pub(super) fn publish_closed(
    source: &super::runtime::ActualUtilityClosedCheckpoint<'_>,
) -> Result<RuntimePublishedUtilityCheckpointArchive, RuntimeUtilityCheckpointArchiveFailure> {
    match source.window().family() {
        UtilityRuntimeFamily::CohortV1 => checkpoint::publish_closed_utility_checkpoint(source)
            .map(RuntimePublishedUtilityCheckpointArchive::Cohort)
            .map_err(RuntimeUtilityCheckpointArchiveFailure::Cohort),
        UtilityRuntimeFamily::InterventionV2 => {
            intervention_checkpoint::publish_closed_utility_checkpoint(source)
                .map(RuntimePublishedUtilityCheckpointArchive::Intervention)
                .map_err(RuntimeUtilityCheckpointArchiveFailure::Intervention)
        }
    }
}

/// Errors keep original partial FD owners. Displaying an error never releases
/// its dataset/ancestor/checkpoint custody or refunds charged reads.
pub(super) enum RuntimeUtilitySourceFailure {
    Metadata(ArenaError),
    Ancestors(Box<RoventUtilityInterventionAncestorsRejected>),
    Checkpoint(intervention_checkpoint::UtilityCheckpointReadFailure),
}
impl std::fmt::Display for RuntimeUtilitySourceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Metadata(e) => e.fmt(f),
            Self::Ancestors(e) => std::fmt::Display::fmt(e, f),
            Self::Checkpoint(e) => std::fmt::Display::fmt(e, f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cohort_window() -> RoventUtilityTrainWindowV2 {
        RoventUtilityTrainWindowV2 {
            schema: super::super::window_v2::ROVENT_UTILITY_WINDOW_DOMAIN_V2.into(),
            window_id: "family-window".into(),
            run_id: "family-run".into(),
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
            checkpoint_reserve_ns: 10_000_000_000,
        }
    }

    #[test]
    fn original_window_arm_preserves_bytes_and_hash_without_retagging() {
        let original = cohort_window();
        let raw = serde_json::to_string(&original).unwrap();
        let runtime: RuntimeUtilityWindow = serde_json::from_str(&raw).unwrap();
        assert_eq!(runtime.family(), UtilityRuntimeFamily::CohortV1);
        assert!(runtime.as_intervention().is_err());
        assert_eq!(runtime.as_cohort().unwrap(), &original);
        assert_eq!(serde_json::to_string(&runtime).unwrap(), raw);
        assert_eq!(
            runtime.canonical_sha256().unwrap(),
            original.canonical_sha256().unwrap()
        );
    }

    #[test]
    fn intervention_window_keeps_its_exact_new_domain_and_original_codec() {
        // Only metadata is created here. No dataset, tensor or current peer is
        // reconstructed by choosing this explicit schema.
        let mut value = serde_json::to_value(cohort_window()).unwrap();
        value["schema"] = serde_json::json!(
            super::super::intervention_window::ROVENT_UTILITY_INTERVENTION_WINDOW_DOMAIN
        );
        let original: RoventUtilityInterventionTrainWindow = serde_json::from_value(value).unwrap();
        original.validate().unwrap();
        let raw = serde_json::to_string(&original).unwrap();
        let runtime: RuntimeUtilityWindow = serde_json::from_str(&raw).unwrap();
        assert_eq!(runtime.family(), UtilityRuntimeFamily::InterventionV2);
        assert!(runtime.as_cohort().is_err());
        assert_eq!(runtime.as_intervention().unwrap(), &original);
        assert_eq!(serde_json::to_string(&runtime).unwrap(), raw);
        assert_eq!(
            runtime.canonical_sha256().unwrap(),
            original.canonical_sha256().unwrap()
        );
        assert_ne!(
            runtime.canonical_sha256().unwrap(),
            cohort_window().canonical_sha256().unwrap()
        );
    }

    #[test]
    fn family_decoder_rejects_duplicate_missing_unknown_or_unregistered_windows() {
        let raw = serde_json::to_string(&cohort_window()).unwrap();
        let duplicate = raw.replacen('{', "{\"checkpoint_reserve_ns\":1,", 1);
        assert!(serde_json::from_str::<RuntimeUtilityWindow>(&duplicate).is_err());
        let mut value = serde_json::to_value(cohort_window()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("checkpoint_reserve_ns");
        assert!(serde_json::from_value::<RuntimeUtilityWindow>(value).is_err());
        let mut value = serde_json::to_value(cohort_window()).unwrap();
        value["family_override"] = serde_json::json!("intervention_v2");
        assert!(serde_json::from_value::<RuntimeUtilityWindow>(value).is_err());
        let mut value = serde_json::to_value(cohort_window()).unwrap();
        value["schema"] = serde_json::json!("rz-rovent-utility-train-window/4");
        assert!(serde_json::from_value::<RuntimeUtilityWindow>(value).is_err());
    }

    #[test]
    fn protocol_families_separate_current_ack_but_share_sampler_math_domain() {
        let original = UtilityRuntimeFamily::CohortV1;
        let intervention = UtilityRuntimeFamily::InterventionV2;
        assert_eq!(
            original.domain("rz-rovent-utility-trainer-ack/1"),
            "rz-rovent-utility-trainer-ack/1"
        );
        assert_eq!(
            intervention.domain("rz-rovent-utility-trainer-ack/1"),
            "rz-rovent-utility-trainer-ack/2"
        );
        assert_eq!(
            intervention.domain("rz-rovent-utility-derived-training-state/1"),
            "rz-rovent-utility-derived-training-state/2"
        );
        assert_eq!(
            intervention.domain("rz-rovent-utility-sampler/1"),
            "rz-rovent-utility-sampler/1"
        );
        assert_eq!(
            intervention.domain("rz-rovent-utility-trainer-source-bundle/1"),
            "rz-rovent-utility-trainer-source-bundle/1"
        );
    }
}
