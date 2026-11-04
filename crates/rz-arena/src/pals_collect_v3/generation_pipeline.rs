//! Explicit, sequential first-generation continuation over original owners.
//!
//! This is orchestration of existing source-owned producers, not a second
//! trainer, scheduler or checkpoint decoder. JSON registers work; only the
//! original game, trainer, exporter and native-intake results authorize its
//! next stage. All generated metadata, checkpoints and failures are retained.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use super::*;
use crate::pals_adaptive_launch::{
    G1EvaluationPreparationFailureV5, PalsEvaluationOpeningRegistrationV5, PalsEvaluationOpeningV5,
    PreparedG1EvaluationLaunchesV5, PreparedG1EvaluationStartupV5,
    RegisteredPalsEvaluationTemplatesV5,
};
#[cfg(target_os = "linux")]
use crate::pals_adaptive_launch::{
    finalize_original_g1_evaluation_launches_v5, prepare_original_g1_evaluation_startup_v5,
    register_pals_evaluation_templates_v5,
};
#[cfg(target_os = "linux")]
use crate::process::ProcessLimits;
#[cfg(target_os = "linux")]
use rz_experiments::pals_data_v3::generation_digest;
use rz_experiments::pals_data_v3::generation_sha;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Component, Path, PathBuf};
#[cfg(target_os = "linux")]
use std::sync::atomic::Ordering;
use std::sync::{Arc, atomic::AtomicBool};
#[cfg(target_os = "linux")]
use std::time::{Duration, Instant};

pub const GENERATION_PIPELINE_DOMAIN_V3: &str = "rz-pals-original-generation-pipeline/1";
const TRAIN_REGISTRATION: &str = "rz-pals-generation-training-window-registration/1";
const EXPORT_REGISTRATION: &str = "rz-pals-generation-export-registration/1";
const SPEC_DOMAIN: &str = "rz-pals-train-generation/1";
const CHECKPOINT_DOMAIN: &str = "rz-pals-actual-generation-checkpoint/1";
const MAX_METADATA: u64 = 16 * 1024 * 1024;
const MAX_STAGING: u64 = 18 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationPipelineRuntimeV3 {
    pub python_runtime: GenerationTrainingPythonRuntimeV3,
    pub dependency_directories: Vec<PathBuf>,
    pub runtime_module_pins: Vec<GenerationTrainingFilePinV3>,
    pub training_runtime: Value,
    pub source_components: BTreeMap<String, String>,
    pub export_source_components: BTreeMap<String, String>,
    pub window_ms: u64,
    pub cleanup_ms: u64,
    pub max_read_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationPipelineEvaluationRegistrationV3 {
    pub registration_id: String,
    pub g0_deployment: rz_experiments::PalsEngineV5,
    pub openings: Vec<PalsEvaluationOpeningV5>,
}

/// Static registration for one complete generation. Each original process
/// gets a fresh finite window before preparation; updating that window does
/// not renew an already-running child, refund work or recollect source games.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationPipelineDeclarationV3 {
    pub schema: String,
    pub generation_id: String,
    pub managed_root: PathBuf,
    pub private_cwd: PathBuf,
    pub metadata_directory: PathBuf,
    pub checkpoint_directory: PathBuf,
    pub index_directory: PathBuf,
    pub source_declaration: GenerationTrainingFilePinV3,
    pub initial_checkpoint: GenerationTrainingFilePinV3,
    pub rules_profile: GenerationTrainingFilePinV3,
    pub frozen_action_template: GenerationTrainingFilePinV3,
    /// Original independent reference/native case plan, never G0 evidence
    /// rewritten to the weights produced by the P/C optimizer.
    pub frozen_numerical_registration: GenerationTrainingFilePinV3,
    /// The 32 future InternalWeights launch intentions are registered while
    /// their candidate assets are still the G0 placeholders.
    pub evaluation_templates: GenerationTrainingFilePinV3,
    pub evaluation_numerical_registration: GenerationTrainingFilePinV3,
    pub evaluation_output_root: PathBuf,
    pub frozen_export_directory: PathBuf,
    pub g1_export_directory: PathBuf,
    pub frozen_model_epoch: u64,
    pub runtime: GenerationPipelineRuntimeV3,
    pub setup_ms: u64,
    pub export_ms: u64,
    pub numerical_ms: u64,
    pub action_preparation_ms: u64,
    pub max_staging_bytes: u64,
    /// Declared phase boundaries, checked before any source producer starts.
    pub initial_updates: u64,
    pub pc_terminal_updates: u64,
    pub generation_terminal_updates: u64,
    /// Pure intent registration precedes all 128 source games. Its opaque owner
    /// is retained unchanged for the later native 64-game evaluator.
    pub evaluation: GenerationPipelineEvaluationRegistrationV3,
}
impl GenerationPipelineDeclarationV3 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            self.schema == GENERATION_PIPELINE_DOMAIN_V3
                && plain_identifier(&self.generation_id)
                && self.initial_updates == 0
                && self.pc_terminal_updates == 4000
                && self.generation_terminal_updates == 5000
                && self.frozen_model_epoch > 0
                && (1..=300_000).contains(&self.setup_ms)
                && (1..=3_600_000).contains(&self.export_ms)
                && (1..=3_600_000).contains(&self.numerical_ms)
                && (1..=3_600_000).contains(&self.action_preparation_ms)
                && self.max_staging_bytes > MAX_METADATA
                && self.max_staging_bytes <= MAX_STAGING
                && (1..=3_600_000).contains(&self.runtime.window_ms)
                && (1..=15_000).contains(&self.runtime.cleanup_ms)
                && self.runtime.max_read_bytes > 0
                // Verification I/O is charged per original game owner and is
                // separate from unique-inode disk occupancy. The trainer's
                // bounded ceiling includes both original corpus readbacks.
                && self.runtime.max_read_bytes <= GENERATION_TRAINING_MAX_READ_BYTES_V3
                && (1..=8).contains(&self.runtime.dependency_directories.len())
                && (1..=64).contains(&self.runtime.runtime_module_pins.len()),
            "one declared real generation, fixed update boundaries and finite registered windows",
        )?;
        let owned = [
            &self.private_cwd,
            &self.metadata_directory,
            &self.checkpoint_directory,
            &self.index_directory,
            &self.frozen_export_directory,
            &self.g1_export_directory,
            &self.evaluation_output_root,
        ];
        check(
            absolute_safe(&self.managed_root)
                && owned
                    .iter()
                    .all(|path| absolute_safe(path) && path.starts_with(&self.managed_root))
                && self.metadata_directory != self.managed_root
                && self.checkpoint_directory != self.managed_root
                && self.index_directory != self.managed_root
                && self.evaluation_output_root == self.managed_root
                && self.frozen_export_directory != self.g1_export_directory
                && self
                    .runtime
                    .dependency_directories
                    .iter()
                    .all(|path| absolute_safe(path)),
            "generation exact nonsecret managed roots",
        )?;
        for pin in [
            &self.source_declaration,
            &self.initial_checkpoint,
            &self.rules_profile,
            &self.frozen_action_template,
            &self.frozen_numerical_registration,
            &self.evaluation_templates,
            &self.evaluation_numerical_registration,
        ] {
            validate_pin(
                pin,
                if pin.path.extension().is_some_and(|x| x == "pt") {
                    1024 * 1024 * 1024
                } else {
                    MAX_METADATA
                },
            )?;
            check(
                pin.path.starts_with(&self.managed_root),
                "generation original inputs charged to staging",
            )?;
        }
        validate_pin(&self.runtime.python_runtime.executable, 256 * 1024 * 1024)?;
        check(
            self.runtime.python_runtime.version.starts_with("3.")
                && self.runtime.python_runtime.version.len() <= 16,
            "generation exact Python release registration",
        )?;
        for pin in &self.runtime.runtime_module_pins {
            validate_pin(pin, 1024 * 1024 * 1024)?;
            check(
                self.runtime
                    .dependency_directories
                    .iter()
                    .any(|root| pin.path.starts_with(root)),
                "generation actual runtime module within trusted dependency root",
            )?;
        }
        let train_names = BTreeSet::from([
            "trainer",
            "checkpoint",
            "loss_recipe",
            "model_reference",
            "model_encoding",
            "data_v3",
            "collector",
        ]);
        let export_names = train_names
            .iter()
            .copied()
            .chain(std::iter::once("model_export"))
            .collect::<BTreeSet<_>>();
        check(
            self.runtime
                .source_components
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                == train_names
                && self
                    .runtime
                    .export_source_components
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
                    == export_names
                && self
                    .runtime
                    .source_components
                    .values()
                    .chain(self.runtime.export_source_components.values())
                    .all(|sha| generation_sha(sha)),
            "generation exact compiled trainer/export source map",
        )?;
        Ok(())
    }
}
fn plain_identifier(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 80
        && name
            .bytes()
            .enumerate()
            .all(|(i, x)| x.is_ascii_alphanumeric() || i > 0 && [b'_', b'-'].contains(&x))
}
fn absolute_safe(path: &Path) -> bool {
    path.is_absolute()
        && path.components().count() <= 64
        && path.components().all(|part| match part {
            Component::RootDir => true,
            Component::Normal(name) => name.to_str().is_some_and(|name| {
                let lower = name.to_ascii_lowercase();
                !name.chars().any(char::is_control)
                    && !lower.starts_with(".env")
                    && ![".pem", ".key", ".p12"]
                        .iter()
                        .any(|suffix| lower.ends_with(suffix))
                    && ![
                        "credential",
                        "service_account",
                        "service-account",
                        "serviceaccount",
                        "ssh-key",
                        "sshkey",
                        "api_key",
                        "api-key",
                        "apikey",
                        "secret",
                        "access_token",
                        "refresh_token",
                    ]
                    .iter()
                    .any(|word| lower.contains(word))
            }),
            _ => false,
        })
}
fn validate_pin(pin: &GenerationTrainingFilePinV3, maximum: u64) -> Result<(), ArenaError> {
    check(
        absolute_safe(&pin.path)
            && generation_sha(&pin.sha256)
            && pin.bytes > 0
            && pin.bytes <= maximum,
        "generation original bounded file pin",
    )
}

enum PipelineStageFailureV3 {
    Source(SelfplayCliFailureV3),
    Training(Box<GenerationTrainingFailureV3>),
    Export(Box<GenerationExportFailureV3>),
    Numerical(Box<FrozenNumericalFailureV3>),
    G1Numerical(Box<G1NumericalFailureV3>),
    EvaluationPreparation(Box<G1EvaluationPreparationFailureV5>),
    ActionPreparation(Box<FrozenActionPreparationFailureV3>),
    Actions(Box<ActionCliFailureV3>),
}
struct PipelineOwnerV3 {
    evaluation_openings: Option<Arc<PalsEvaluationOpeningRegistrationV5>>,
    evaluation_templates: Option<Arc<RegisteredPalsEvaluationTemplatesV5>>,
    series: Option<Arc<GenerationVerifiedSeriesArchiveV3>>,
    trainer: Option<Arc<VerifiedGenerationTrainingWindowV3>>,
    frozen: Option<Arc<VerifiedGenerationExportV3>>,
    numerical: Option<Arc<VerifiedFrozenNumericalV3>>,
    prepared_actions: Option<PreparedFrozenActionCollectionV3>,
    actions: Option<Arc<VerifiedActionOutcomeArchiveV3>>,
    g1: Option<Arc<VerifiedGenerationExportV3>>,
    g1_prepared: Option<Arc<PreparedG1EvaluationStartupV5>>,
    g1_numerical: Option<Arc<VerifiedG1NumericalV3>>,
    g1_launches: Option<Arc<PreparedG1EvaluationLaunchesV5>>,
    evaluation_output_root: Option<PathBuf>,
    evaluation_attempt_prefix: Option<String>,
    stage_failure: Option<PipelineStageFailureV3>,
    metadata: Vec<(GenerationTrainingFilePinV3, File)>,
    journal: Vec<Value>,
    failure_directory: Option<PathBuf>,
    output_budget: Option<(PathBuf, u64)>,
}
impl PipelineOwnerV3 {
    fn new() -> Self {
        Self {
            evaluation_openings: None,
            evaluation_templates: None,
            series: None,
            trainer: None,
            frozen: None,
            numerical: None,
            prepared_actions: None,
            actions: None,
            g1: None,
            g1_prepared: None,
            g1_numerical: None,
            g1_launches: None,
            evaluation_output_root: None,
            evaluation_attempt_prefix: None,
            stage_failure: None,
            metadata: vec![],
            journal: vec![],
            failure_directory: None,
            output_budget: None,
        }
    }
}
/// First failing stage keeps every original process/checkpoint/archive owner.
/// There is no automatic retry, reconstructed checkpoint or successful flag.
pub struct GenerationPipelineFailureV3 {
    pub stage: String,
    pub error: ArenaError,
    pub preservation_error: Option<ArenaError>,
    owner: Box<PipelineOwnerV3>,
}
impl GenerationPipelineFailureV3 {
    pub fn evaluation_openings(&self) -> Option<&Arc<PalsEvaluationOpeningRegistrationV5>> {
        self.owner.evaluation_openings.as_ref()
    }
    pub fn series(&self) -> Option<&Arc<GenerationVerifiedSeriesArchiveV3>> {
        self.owner.series.as_ref()
    }
    pub fn trainer(&self) -> Option<&Arc<VerifiedGenerationTrainingWindowV3>> {
        self.owner.trainer.as_ref()
    }
    pub fn actions(&self) -> Option<&Arc<VerifiedActionOutcomeArchiveV3>> {
        self.owner.actions.as_ref()
    }
    pub fn frozen_numerical(&self) -> Option<&Arc<VerifiedFrozenNumericalV3>> {
        self.owner.numerical.as_ref()
    }
    pub fn source_failure(&self) -> Option<&SelfplayCliFailureV3> {
        match self.owner.stage_failure.as_ref() {
            Some(PipelineStageFailureV3::Source(failure)) => Some(failure),
            _ => None,
        }
    }
    pub fn training_failure(&self) -> Option<&GenerationTrainingFailureV3> {
        match self.owner.stage_failure.as_ref() {
            Some(PipelineStageFailureV3::Training(failure)) => Some(failure),
            _ => None,
        }
    }
    pub fn export_failure(&self) -> Option<&GenerationExportFailureV3> {
        match self.owner.stage_failure.as_ref() {
            Some(PipelineStageFailureV3::Export(failure)) => Some(failure),
            _ => None,
        }
    }
    pub fn numerical_failure(&self) -> Option<&FrozenNumericalFailureV3> {
        match self.owner.stage_failure.as_ref() {
            Some(PipelineStageFailureV3::Numerical(failure)) => Some(failure),
            _ => None,
        }
    }
    pub fn g1_numerical_failure(&self) -> Option<&G1NumericalFailureV3> {
        match self.owner.stage_failure.as_ref() {
            Some(PipelineStageFailureV3::G1Numerical(failure)) => Some(failure),
            _ => None,
        }
    }
    pub fn evaluation_preparation_failure(&self) -> Option<&G1EvaluationPreparationFailureV5> {
        match self.owner.stage_failure.as_ref() {
            Some(PipelineStageFailureV3::EvaluationPreparation(failure)) => Some(failure),
            _ => None,
        }
    }
    pub fn action_failure(&self) -> Option<&ActionCliFailureV3> {
        match self.owner.stage_failure.as_ref() {
            Some(PipelineStageFailureV3::Actions(failure)) => Some(failure),
            _ => None,
        }
    }
    pub fn action_preparation_failure(&self) -> Option<&FrozenActionPreparationFailureV3> {
        match self.owner.stage_failure.as_ref() {
            Some(PipelineStageFailureV3::ActionPreparation(failure)) => Some(failure),
            _ => None,
        }
    }
}
impl std::fmt::Debug for GenerationPipelineFailureV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationPipelineFailureV3")
            .field("stage", &self.stage)
            .field("error", &self.error)
            .field("original_owners_retained", &true)
            .field("preservation_error", &self.preservation_error)
            .finish()
    }
}
impl std::fmt::Display for GenerationPipelineFailureV3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {}; original owners retained; automatic retry disabled",
            self.stage, self.error
        )
    }
}
impl std::error::Error for GenerationPipelineFailureV3 {}

pub struct VerifiedGenerationPipelineV3 {
    status: Value,
    owner: Box<PipelineOwnerV3>,
}
impl VerifiedGenerationPipelineV3 {
    pub fn evaluation_templates(&self) -> &Arc<RegisteredPalsEvaluationTemplatesV5> {
        self.owner
            .evaluation_templates
            .as_ref()
            .expect("original precollection evaluation templates")
    }
    pub fn evaluation_openings(&self) -> &Arc<PalsEvaluationOpeningRegistrationV5> {
        self.owner
            .evaluation_openings
            .as_ref()
            .expect("original preregistered final openings")
    }
    pub fn status(&self) -> &Value {
        &self.status
    }
    pub fn series(&self) -> &Arc<GenerationVerifiedSeriesArchiveV3> {
        self.owner
            .series
            .as_ref()
            .expect("successful original series")
    }
    pub fn trainer(&self) -> &Arc<VerifiedGenerationTrainingWindowV3> {
        self.owner
            .trainer
            .as_ref()
            .expect("successful final original trainer")
    }
    pub fn actions(&self) -> &Arc<VerifiedActionOutcomeArchiveV3> {
        self.owner
            .actions
            .as_ref()
            .expect("successful original action archive")
    }
    pub fn frozen_export(&self) -> &Arc<VerifiedGenerationExportV3> {
        self.owner
            .frozen
            .as_ref()
            .expect("successful frozen original export")
    }
    pub fn frozen_numerical(&self) -> &Arc<VerifiedFrozenNumericalV3> {
        self.owner
            .numerical
            .as_ref()
            .expect("original frozen independent numerical producer")
    }
    pub fn g1_export(&self) -> &Arc<VerifiedGenerationExportV3> {
        self.owner
            .g1
            .as_ref()
            .expect("successful G1 original export")
    }
    pub fn g1_numerical(&self) -> &Arc<VerifiedG1NumericalV3> {
        self.owner
            .g1_numerical
            .as_ref()
            .expect("original G1 independent deployment numerical owner")
    }
    pub fn evaluation_launches(&self) -> &Arc<PreparedG1EvaluationLaunchesV5> {
        self.owner
            .g1_launches
            .as_ref()
            .expect("original future launch templates bound to actual G1 and numerical owner")
    }
    pub fn evaluation_output_root(&self) -> &Path {
        self.owner
            .evaluation_output_root
            .as_deref()
            .expect("original registered evaluation output root")
    }
    pub fn evaluation_attempt_prefix(&self) -> &str {
        self.owner
            .evaluation_attempt_prefix
            .as_deref()
            .expect("original registered evaluation attempt namespace")
    }
}

/// The only fresh CLI entry keeps original game/trainer/numerical owners alive
/// through all 32 native pairs. There is no post-CLI JSON restore of the G1 or
/// the preregistered cohort, and no score-based retry or checkpoint selection.
pub fn run_generation_and_evaluation_owned_v3(
    path: &Path,
    expected_sha256: &str,
    cancel: Arc<AtomicBool>,
) -> Result<crate::pals_adaptive_launch::PalsEvaluationSeriesV5, Box<GenerationPipelineFailureV3>> {
    run_generation_and_evaluation_owned_with_storage_policy_v3(
        path,
        expected_sha256,
        cancel,
        crate::pals_adaptive_launch::PalsOwnedStoragePolicy::StrictV5,
    )
}

/// Select an explicit storage admission policy for the evaluation series.
/// The existing V3 entry keeps the strict V5 policy for compatibility.
pub fn run_generation_and_evaluation_owned_with_storage_policy_v3(
    path: &Path,
    expected_sha256: &str,
    cancel: Arc<AtomicBool>,
    storage_policy: crate::pals_adaptive_launch::PalsOwnedStoragePolicy,
) -> Result<crate::pals_adaptive_launch::PalsEvaluationSeriesV5, Box<GenerationPipelineFailureV3>> {
    // This sequential parent retains the original non-Send control owner.
    // Arc preserves its identity through evaluation custody; this owner is
    // never transferred to the CPU/GPU workers or made Send/Sync by the wrapper.
    #[allow(clippy::arc_with_non_send_sync)]
    let pipeline = Arc::new(run_generation_pipeline_owned_v3(
        path,
        expected_sha256,
        Arc::clone(&cancel),
    )?);
    let launches = pipeline.evaluation_launches().launches().to_vec();
    let source_root = pipeline.evaluation_launches().source_root().to_owned();
    let output_root = pipeline.evaluation_output_root().to_owned();
    let prefix = pipeline.evaluation_attempt_prefix().to_owned();
    Ok(
        crate::pals_adaptive_launch::run_pals_evaluation_series_with_storage_policy_v5(
            pipeline,
            launches,
            &source_root,
            &output_root,
            &prefix,
            &cancel,
            storage_policy,
        ),
    )
}

/// Explicit call only. The caller retains the sole GPU execution ownership;
/// every subprocess/action here is sequential and joined before its successor.
/// No parse/import/build path invokes this function. Original independent
/// numerical producers precede final launch preparation; the final arena
/// consumes its returned owner separately in the same-process CLI wrapper.
pub fn run_generation_pipeline_owned_v3(
    path: &Path,
    expected_sha256: &str,
    cancel: Arc<AtomicBool>,
) -> Result<VerifiedGenerationPipelineV3, Box<GenerationPipelineFailureV3>> {
    #[cfg(target_os = "linux")]
    {
        let openings = match prepare_generation_pipeline_openings_v3(path, expected_sha256, &cancel)
        {
            Ok(openings) => openings,
            Err(error) => {
                return Err(Box::new(GenerationPipelineFailureV3 {
                    stage: "evaluation-preregistration".into(),
                    error,
                    preservation_error: None,
                    owner: Box::new(PipelineOwnerV3::new()),
                }));
            }
        };
        run_generation_pipeline_with_openings_owned_v3(path, expected_sha256, openings, cancel)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (path, expected_sha256, cancel);
        Err(Box::new(GenerationPipelineFailureV3 {
            stage: "registration".into(),
            error: invalid("original generation pipeline requires Linux"),
            preservation_error: None,
            owner: Box::new(PipelineOwnerV3::new()),
        }))
    }
}

/// For an integration owner that preregistered the final opening set earlier.
/// The exact Arc is retained; callers cannot substitute it after seeing games.
pub fn run_generation_pipeline_with_openings_owned_v3(
    path: &Path,
    expected_sha256: &str,
    openings: Arc<PalsEvaluationOpeningRegistrationV5>,
    cancel: Arc<AtomicBool>,
) -> Result<VerifiedGenerationPipelineV3, Box<GenerationPipelineFailureV3>> {
    let mut owner = Box::new(PipelineOwnerV3::new());
    owner.evaluation_openings = Some(openings);
    #[allow(unused_mut)]
    let mut stage = "registration".to_owned();
    #[cfg(not(target_os = "linux"))]
    let result: Result<Value, ArenaError> = {
        let _ = (path, expected_sha256, cancel);
        Err(invalid("original generation pipeline requires Linux"))
    };
    #[cfg(target_os = "linux")]
    let result = run_linux(path, expected_sha256, &cancel, &mut owner, &mut stage);
    finish_pipeline_result(result, stage, owner)
}

/// Explicit same-process continuation only at already verified stage
/// boundaries. A failed original child/native owner must be resolved by its
/// original producer first; this API does not retry, clear or reinterpret it.
/// The original declaration/128-game custody is required, so reopening a JSON
/// receipt or invoking the fresh CLI cannot resume a partial generation.
pub fn resume_completed_generation_pipeline_owned_v3(
    mut previous: Box<GenerationPipelineFailureV3>,
    path: &Path,
    expected_sha256: &str,
    cancel: Arc<AtomicBool>,
) -> Result<VerifiedGenerationPipelineV3, Box<GenerationPipelineFailureV3>> {
    let admitted = previous.owner.stage_failure.is_none()
        && previous
            .owner
            .series
            .as_ref()
            .is_some_and(|series| series.archives().len() == 128)
        && previous.owner.evaluation_openings.is_some()
        && previous
            .owner
            .metadata
            .iter()
            .any(|(pin, _)| pin.path == path && pin.sha256 == expected_sha256)
        && generation_sha(expected_sha256);
    if !admitted {
        previous.preservation_error = Some(invalid(
            "completed-stage resume requires unchanged original declaration, 128-game authority and no unresolved original failing child",
        ));
        return Err(previous);
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cancel;
        previous.preservation_error = Some(invalid(
            "completed-stage original continuation requires Linux",
        ));
        Err(previous)
    }
    #[cfg(target_os = "linux")]
    {
        let mut owner = previous.owner;
        owner.journal.push(serde_json::json!({"stage":"explicit_completed_stage_continuation",
            "previous_failed_stage":previous.stage,"previous_error":previous.error.to_string(),
            "additional_source_games":0,"unresolved_child_retry":false,
            "failure_record_preservation_error":previous.preservation_error.as_ref().map(ToString::to_string)}));
        let mut stage = "continuation-registration".to_owned();
        let result = run_linux(path, expected_sha256, &cancel, &mut owner, &mut stage);
        finish_pipeline_result(result, stage, owner)
    }
}

fn finish_pipeline_result(
    result: Result<Value, ArenaError>,
    stage: String,
    owner: Box<PipelineOwnerV3>,
) -> Result<VerifiedGenerationPipelineV3, Box<GenerationPipelineFailureV3>> {
    match result {
        Ok(status) => Ok(VerifiedGenerationPipelineV3 { status, owner }),
        Err(error) => {
            #[allow(unused_mut)]
            let mut preservation_error = None;
            #[cfg(target_os = "linux")]
            if let Some(directory) = owner.failure_directory.as_ref() {
                let receipt = serde_json::json!({"schema":"rz-pals-generation-pipeline-failure/1",
                        "stage":stage,"error":error.to_string(),"stages":owner.journal,
                        "original_owners_retained":true,"automatic_retry":false,"generation_completed":false,
                        "physical_completion":"inspect_original_stage_owner"});
                let diagnostic_until = Instant::now() + Duration::from_secs(15);
                preservation_error = owner.output_budget.as_ref().map_or_else(
                    || Some(invalid("original pipeline output budget absent")),
                    |(root, maximum)| {
                        publish_metadata_checked(
                            root,
                            *maximum,
                            directory,
                            "pipeline-failure.json",
                            &receipt,
                            diagnostic_until,
                            &AtomicBool::new(false),
                        )
                        .err()
                    },
                );
            }
            Err(Box::new(GenerationPipelineFailureV3 {
                stage,
                error,
                preservation_error,
                owner,
            }))
        }
    }
}

/// Pure source-file/Rules intent registration; no collection, optimizer,
/// export, runtime creation or neural execution occurs in this helper.
#[cfg(target_os = "linux")]
pub fn prepare_generation_pipeline_openings_v3(
    path: &Path,
    expected: &str,
    cancel: &AtomicBool,
) -> Result<Arc<PalsEvaluationOpeningRegistrationV5>, ArenaError> {
    check(
        generation_sha(expected) && absolute_safe(path),
        "pipeline original preregistration declaration",
    )?;
    let file = open_file(path)?;
    let size = file
        .metadata()
        .map_err(|_| invalid("pipeline preregistration file metadata"))?
        .len();
    let (_, raw) = read_declared(
        &GenerationTrainingFilePinV3 {
            path: path.to_owned(),
            sha256: expected.into(),
            bytes: size,
        },
        MAX_METADATA,
        Instant::now() + Duration::from_secs(300),
        cancel,
    )?;
    let declaration: GenerationPipelineDeclarationV3 = rz_experiments::decode_json(
        std::str::from_utf8(&raw).map_err(|_| invalid("pipeline preregistration UTF-8"))?,
    )
    .map_err(|error| invalid(error.to_string()))?;
    declaration.validate()?;
    Ok(Arc::new(PalsEvaluationOpeningRegistrationV5::register(
        declaration.evaluation.registration_id,
        declaration.source_declaration.sha256,
        declaration.evaluation.g0_deployment,
        declaration.evaluation.openings,
    )?))
}

#[cfg(target_os = "linux")]
fn run_linux(
    path: &Path,
    expected: &str,
    cancel: &Arc<AtomicBool>,
    owner: &mut PipelineOwnerV3,
    stage: &mut String,
) -> Result<Value, ArenaError> {
    check(
        generation_sha(expected) && absolute_safe(path),
        "pipeline original declaration pin",
    )?;
    let started = Instant::now();
    let preflight_until = started + Duration::from_secs(300);
    let (declaration_file, raw) = read_original_input(
        owner,
        &GenerationTrainingFilePinV3 {
            path: path.to_owned(),
            sha256: expected.into(),
            bytes: open_file(path)?
                .metadata()
                .map_err(|_| invalid("pipeline declaration metadata"))?
                .len(),
        },
        MAX_METADATA,
        preflight_until,
        cancel,
    )?;
    let declaration: GenerationPipelineDeclarationV3 = rz_experiments::decode_json(
        std::str::from_utf8(&raw).map_err(|_| invalid("pipeline declaration UTF-8"))?,
    )
    .map_err(|error| invalid(error.to_string()))?;
    declaration.validate()?;
    let openings = Arc::clone(
        owner
            .evaluation_openings
            .as_ref()
            .ok_or_else(|| invalid("original precollection evaluation owner absent"))?,
    );
    check(
        openings.registration_id() == declaration.evaluation.registration_id
            && openings.generation_manifest_sha256() == declaration.source_declaration.sha256
            && openings.g0_deployment() == &declaration.evaluation.g0_deployment
            && openings.openings() == declaration.evaluation.openings
            && openings.g0_deployment().model.weights.sha256
                == declaration.initial_checkpoint.sha256,
        "pipeline final openings/G0/source declaration must be fixed before source collection",
    )?;
    let setup_until = (started + Duration::from_millis(declaration.setup_ms)).min(preflight_until);
    check_live(setup_until, cancel)?;
    check(
        std::env::current_dir().map_err(|_| invalid("pipeline cwd"))? == declaration.private_cwd,
        "pipeline actual cwd must match registered frozen action private cwd",
    )?;
    let managed = super::selfplay_cli::open_directory(&declaration.managed_root)?;
    super::selfplay_cli::private_root(&managed)?;
    let observed = super::selfplay_cli::scan_staging(
        &managed,
        declaration.max_staging_bytes,
        setup_until,
        cancel,
    )?;
    for directory in [
        &declaration.metadata_directory,
        &declaration.checkpoint_directory,
        &declaration.index_directory,
        &declaration.evaluation_output_root,
    ] {
        let directory = super::selfplay_cli::open_directory(directory)?;
        super::selfplay_cli::private_root(&directory)?;
    }
    owner.failure_directory = Some(declaration.metadata_directory.clone());
    owner.output_budget = Some((
        declaration.managed_root.clone(),
        declaration.max_staging_bytes,
    ));
    owner.evaluation_output_root = Some(declaration.evaluation_output_root.clone());
    owner.evaluation_attempt_prefix = Some(format!("{}-final", declaration.generation_id));
    let compiled = rz_experiments::compiled_source_components_v5()
        .map_err(|error| invalid(error.to_string()))?;
    check(
        declaration
            .runtime
            .source_components
            .iter()
            .chain(&declaration.runtime.export_source_components)
            .all(|(component, hash)| compiled.get(component) == Some(hash)),
        "pipeline exact compiled source registration",
    )?;
    owner.metadata.push((
        GenerationTrainingFilePinV3 {
            path: path.to_owned(),
            sha256: expected.into(),
            bytes: raw.len() as u64,
        },
        declaration_file,
    ));
    drop(raw);
    owner.journal.push(serde_json::json!({"stage":"registered","boundaries":[0,4000,5000],"staging":observed,
        "training_window_ms":declaration.runtime.window_ms,"max_output_bytes":declaration.max_staging_bytes,
        "evaluation_opening_registration_sha256":openings.sha256(),"evaluation_generation_pin":"original_source_declaration_raw_sha256"}));

    *stage = "evaluation-template-preregistration".into();
    if owner.evaluation_templates.is_none() {
        owner.evaluation_templates = Some(Arc::new(
            match register_pals_evaluation_templates_v5(
                declaration.evaluation_templates.clone(),
                Arc::clone(&openings),
                setup_until,
                cancel,
            ) {
                Ok(templates) => templates,
                Err(failure) => {
                    let error = invalid(failure.error.to_string());
                    owner.stage_failure =
                        Some(PipelineStageFailureV3::EvaluationPreparation(failure));
                    return Err(error);
                }
            },
        ));
    }
    let templates = Arc::clone(
        owner
            .evaluation_templates
            .as_ref()
            .expect("preregistered original templates"),
    );
    check(
        Arc::ptr_eq(templates.registration(), &openings)
            && templates.managed_root() == declaration.managed_root
            && declaration.evaluation_output_root == templates.managed_root()
            && templates.private_cwd() == declaration.private_cwd
            && templates
                .source_root()
                .starts_with(&declaration.managed_root),
        "evaluation must retain same preregistered opening owner and complete source/trainer/final managed accounting root",
    )?;

    // Only validate preregistered intent here. Actual numerical authority is
    // still produced after each original export by its two retained captures.
    // Reject a wrong static template/budget before collecting128 source games.
    *stage = "numerical-launch-preregistration".into();
    for (pin, template) in [
        (
            &declaration.frozen_numerical_registration,
            &declaration.frozen_action_template,
        ),
        (
            &declaration.evaluation_numerical_registration,
            &declaration.evaluation_templates,
        ),
    ] {
        let (file, bytes) = read_original_input(owner, pin, MAX_METADATA, setup_until, cancel)?;
        let registration: GenerationNumericalLaunchRegistrationV3 = rz_experiments::decode_json(
            std::str::from_utf8(&bytes).map_err(|_| invalid("numerical preregistration UTF-8"))?,
        )
        .map_err(|error| invalid(error.to_string()))?;
        check_numerical_preregistration(&declaration, &registration, template)?;
        owner.metadata.push((pin.clone(), file));
    }

    *stage = "source128".into();
    check_live(setup_until, cancel)?;
    let (source_file, source_raw) = read_original_input(
        owner,
        &declaration.source_declaration,
        MAX_METADATA,
        setup_until,
        cancel,
    )?;
    let source_registration: SelfplayCliDeclarationV3 = rz_experiments::decode_json(
        std::str::from_utf8(&source_raw)
            .map_err(|_| invalid("original source registration UTF-8"))?,
    )
    .map_err(|error| invalid(error.to_string()))?;
    source_registration.validate()?;
    check(
        source_registration.run_root == declaration.managed_root
            && source_registration.max_staging_bytes == declaration.max_staging_bytes,
        "source accounting must cover the same original complete managed staging root and cap",
    )?;
    check_collection_evaluation_disjoint(&source_registration.plan, openings.openings())?;
    drop(source_raw);
    owner
        .metadata
        .push((declaration.source_declaration.clone(), source_file));
    if owner.series.is_none() {
        let source = match run_selfplay_collect_owned_cli_v3(
            &declaration.source_declaration.path,
            &declaration.source_declaration.sha256,
            cancel,
        ) {
            Ok(source) => source,
            Err(failure) => {
                let error = invalid(failure.to_string());
                owner.stage_failure = Some(PipelineStageFailureV3::Source(failure));
                return Err(error);
            }
        };
        owner
            .journal
            .push(serde_json::json!({"stage":"source128","status":source.status()}));
        owner.series = Some(Arc::new(source.into_archive()));
    }
    let series = Arc::clone(owner.series.as_ref().expect("original series owner"));
    check(
        series.archives().len() == 128,
        "pipeline one exact original 128-game source",
    )?;
    let metadata_until = Instant::now() + Duration::from_millis(declaration.setup_ms);
    scan(&declaration, metadata_until, cancel)?;
    let pc_manifest_pin = publish_or_retain_completed_metadata(
        &declaration,
        owner,
        series
            .path()
            .parent()
            .ok_or_else(|| invalid("original series root"))?,
        "pipeline-pc-manifest.json",
        series.dataset_manifest(),
        metadata_until,
        cancel,
    )?;
    owner.metadata.push((
        pc_manifest_pin.0.clone(),
        pc_manifest_pin
            .1
            .try_clone()
            .map_err(|_| invalid("original PC manifest descriptor clone"))?,
    ));
    let pc_dataset = serde_json::json!({"manifest":pc_manifest_pin.0,"manifest_sha256":series.dataset_manifest_sha256(),
        "corpus_root":series.path().parent()});
    let spec = make_spec(&declaration, &series)?;
    let spec_pin = publish_or_retain_completed_metadata(
        &declaration,
        owner,
        &declaration.metadata_directory,
        "generation-spec.json",
        &spec,
        metadata_until,
        cancel,
    )?;
    owner.metadata.push((
        spec_pin.0.clone(),
        spec_pin
            .1
            .try_clone()
            .map_err(|_| invalid("original spec descriptor clone"))?,
    ));
    let spec_pin = spec_pin.0;
    let source_model = spec["model_sha256"]
        .as_str()
        .ok_or_else(|| invalid("original source model hash"))?
        .to_owned();
    let source_runtime = spec["runtime_sha256"]
        .as_str()
        .ok_or_else(|| invalid("original source runtime hash"))?
        .to_owned();
    check(
        declaration.initial_checkpoint.sha256 == source_model
            && declaration.frozen_model_epoch
                > spec["model_epoch"]
                    .as_u64()
                    .ok_or_else(|| invalid("original G0 epoch"))?,
        "pipeline exact original G0 and separately registered frozen epoch",
    )?;
    let already = verified_completed_boundary(owner.trainer.as_deref())?;
    for window in 0..16 {
        if already.iter().sum::<u64>() >= (window as u64 + 1) * 250 {
            continue;
        }
        *stage = format!("pc-window-{window:02}");
        train_window(
            &declaration,
            &series,
            None,
            owner,
            &spec_pin,
            &pc_dataset,
            &source_model,
            &source_runtime,
            "pc",
            window,
            window as u64 * 250,
            cancel,
        )?;
    }
    let pc = pc_terminal_owner(owner.trainer.as_ref().expect("actual completed PC trainer"))?;
    check(
        completed_updates(pc.receipt())? == [2000, 2000, 0],
        "pipeline actual exact4000 PC terminal owner",
    )?;
    *stage = "frozen4000-export".into();
    if owner.frozen.is_none() {
        owner.frozen = Some(Arc::new(export_stage(
            &declaration,
            Arc::clone(&pc),
            GenerationExportKindV3::FrozenPcCollection,
            owner,
            cancel,
        )?));
    }
    let frozen = Arc::clone(owner.frozen.as_ref().expect("original frozen export"));
    *stage = "frozen4000-independent-numerical".into();
    let numerical_started = Instant::now();
    let numerical_until = numerical_started + Duration::from_millis(declaration.numerical_ms);
    let numerical_cleanup_until =
        numerical_until + Duration::from_millis(declaration.runtime.cleanup_ms);
    scan(&declaration, numerical_until, cancel)?;
    if owner.numerical.is_none() {
        owner.numerical = Some(Arc::new(
            match run_frozen_numerical_original_v3(
                declaration.frozen_numerical_registration.clone(),
                Arc::clone(&frozen),
                numerical_until,
                numerical_cleanup_until,
                Arc::clone(cancel),
            ) {
                Ok(numerical) => numerical,
                Err(failure) => {
                    let error = invalid(failure.error.to_string());
                    owner.stage_failure = Some(PipelineStageFailureV3::Numerical(failure));
                    return Err(error);
                }
            },
        ));
    }
    let numerical = Arc::clone(
        owner
            .numerical
            .as_ref()
            .expect("original frozen numerical owner"),
    );
    check(
        Arc::ptr_eq(numerical.frozen(), &frozen)
            && numerical.prepared().template_pin() == &declaration.frozen_action_template,
        "frozen numerical must retain the same actual exporter and registered original template",
    )?;
    owner.journal.push(
        serde_json::json!({"stage":"frozen4000-independent-numerical",
        "acceptance_pin":numerical.acceptance_pin(),"registered_window_ms":declaration.numerical_ms,
        "original_export_owner_retained":true,"g0_evidence_reused":false}),
    );
    scan(&declaration, numerical_cleanup_until, cancel)?;
    *stage = "frozen-action-preparation".into();
    let preparation_until =
        Instant::now() + Duration::from_millis(declaration.action_preparation_ms);
    scan(&declaration, preparation_until, cancel)?;
    if owner.prepared_actions.is_none() {
        owner.prepared_actions = Some(
            match prepare_frozen_action_collection_owned_v3(
                Arc::clone(numerical.prepared()),
                Arc::clone(&numerical),
                preparation_until,
                cancel,
            ) {
                Ok(prepared) => prepared,
                Err(failure) => {
                    let error = invalid(failure.error.to_string());
                    owner.stage_failure = Some(PipelineStageFailureV3::ActionPreparation(failure));
                    return Err(error);
                }
            },
        );
    }
    *stage = "actual-action-contexts".into();
    let prepared = owner
        .prepared_actions
        .as_ref()
        .expect("original frozen action metadata");
    if owner.actions.is_none() {
        owner.actions = Some(Arc::new(
            match run_action_outcomes_owned_cli_v3(
                prepared.declaration_path(),
                prepared.declaration_file_sha256(),
                series.shared_archives(),
                Arc::clone(cancel),
            ) {
                Ok(actions) => actions,
                Err(failure) => {
                    let error = invalid(failure.error.to_string());
                    owner.stage_failure = Some(PipelineStageFailureV3::Actions(failure));
                    return Err(error);
                }
            },
        ));
    }
    let actions = Arc::clone(
        owner
            .actions
            .as_ref()
            .expect("original actual action owner"),
    );
    check(
        actions.plan().frozen_pc_epoch == declaration.frozen_model_epoch
            && actions.plan().frozen_pc_weights_sha256 == frozen.frozen_pc_sha256(),
        "pipeline frozen source/export/native action join",
    )?;
    owner
        .journal
        .push(serde_json::json!({"stage":"actual-action-contexts","status":actions.status()}));
    for window in 0..4 {
        if already.iter().sum::<u64>() >= 4000 + (window as u64 + 1) * 250 {
            continue;
        }
        *stage = format!("v-window-{window:02}");
        train_window(
            &declaration,
            &series,
            Some(Arc::clone(&actions)),
            owner,
            &spec_pin,
            &pc_dataset,
            &source_model,
            &source_runtime,
            "v",
            window,
            4000 + window as u64 * 250,
            cancel,
        )?;
    }
    let final_trainer = Arc::clone(owner.trainer.as_ref().expect("original final trainer"));
    check(
        completed_updates(final_trainer.receipt())? == [2000, 2000, 1000],
        "pipeline exact5000 actual terminal owner",
    )?;
    *stage = "g1-actual-export".into();
    if owner.g1.is_none() {
        owner.g1 = Some(Arc::new(export_stage(
            &declaration,
            final_trainer,
            GenerationExportKindV3::G1ActualGeneration,
            owner,
            cancel,
        )?));
    }
    let g1 = Arc::clone(owner.g1.as_ref().expect("original actual G1 export"));
    *stage = "g1-evaluation-startup-preparation".into();
    let g1_preparation_until =
        Instant::now() + Duration::from_millis(declaration.action_preparation_ms);
    scan(&declaration, g1_preparation_until, cancel)?;
    if owner.g1_prepared.is_none() {
        owner.g1_prepared = Some(Arc::new(
            match prepare_original_g1_evaluation_startup_v5(
                Arc::clone(&templates),
                Arc::clone(&g1),
                g1_preparation_until,
                cancel,
            ) {
                Ok(prepared) => prepared,
                Err(failure) => {
                    let error = invalid(failure.error.to_string());
                    owner.stage_failure =
                        Some(PipelineStageFailureV3::EvaluationPreparation(failure));
                    return Err(error);
                }
            },
        ));
    }
    let g1_prepared = Arc::clone(
        owner
            .g1_prepared
            .as_ref()
            .expect("original G1 native startup"),
    );
    *stage = "g1-deployment-independent-numerical".into();
    let g1_numerical_until = Instant::now() + Duration::from_millis(declaration.numerical_ms);
    let g1_numerical_cleanup =
        g1_numerical_until + Duration::from_millis(declaration.runtime.cleanup_ms);
    scan(&declaration, g1_numerical_until, cancel)?;
    if owner.g1_numerical.is_none() {
        owner.g1_numerical = Some(Arc::new(
            match run_g1_numerical_original_v3(
                declaration.evaluation_numerical_registration.clone(),
                Arc::clone(&g1_prepared),
                g1_numerical_until,
                g1_numerical_cleanup,
                Arc::clone(cancel),
            ) {
                Ok(numerical) => numerical,
                Err(failure) => {
                    let error = invalid(failure.error.to_string());
                    owner.stage_failure = Some(PipelineStageFailureV3::G1Numerical(failure));
                    return Err(error);
                }
            },
        ));
    }
    let g1_numerical = Arc::clone(
        owner
            .g1_numerical
            .as_ref()
            .expect("original G1 numerical owner"),
    );
    check(
        Arc::ptr_eq(g1_numerical.prepared(), &g1_prepared),
        "G1 numerical must retain the same original prepared deployment owner",
    )?;
    scan(&declaration, g1_numerical_cleanup, cancel)?;
    *stage = "g1-final32-launch-finalization".into();
    let finalization_until =
        Instant::now() + Duration::from_millis(declaration.action_preparation_ms);
    if owner.g1_launches.is_none() {
        owner.g1_launches = Some(Arc::new(match finalize_original_g1_evaluation_launches_v5(
            Arc::clone(&g1_prepared),
            Arc::clone(&g1_numerical),
            finalization_until,
            cancel,
        ) {
            Ok(launches) => launches,
            Err(failure) => {
                let error = invalid(failure.error.to_string());
                owner.stage_failure = Some(PipelineStageFailureV3::EvaluationPreparation(failure));
                return Err(error);
            }
        }));
    }
    owner
        .journal
        .push(serde_json::json!({"stage":"g1-final32-launch-finalization",
        "g1_independent_numerical_acceptance":g1_numerical.acceptance_pin(),
        "original_openings_preserved":true,"native_pair_execution":"not_run"}));
    let final_until = Instant::now() + Duration::from_millis(declaration.setup_ms);
    let final_staging = scan(&declaration, final_until, cancel)?;
    let status = serde_json::json!({"schema":"rz-pals-original-generation-pipeline-result/1","status":"actual_generation_exported_pending_acceptance",
        "generation_id":declaration.generation_id,"source_games":128,"additional_source_games":0,
        "actual_completed_updates":{"proposer":2000,"critic":2000,"validator":1000},
        "stages":owner.journal,"staging":final_staging,"g1":owner.g1.as_ref().expect("G1 export owner").receipt(),
        "numeric_status":"original_frozen_and_G1_independent_numerical_accepted","arena_accepted":false,
        "assurance_scope":"same_original128games_actual_PC4000_frozen_export_original_numerical_actual_action_intake_V1000_actual_G1_export_original_deployment_numerical; no_strength_or_final_arena_acceptance"});
    owner.metadata.push(publish_metadata_checked(
        &declaration.managed_root,
        declaration.max_staging_bytes,
        &declaration.metadata_directory,
        "pipeline-result.json",
        &status,
        final_until,
        cancel,
    )?);
    Ok(status)
}

/// Input-intent overlap is rejected before outcomes exist. This supplements,
/// rather than substitutes for, the collector's exact state/transposition and
/// lineage checks. Imported FEN history is not treated as known startpos history.
fn check_collection_evaluation_disjoint(
    collection: &SelfplayCollectionPlanV3,
    evaluation: &[PalsEvaluationOpeningV5],
) -> Result<(), ArenaError> {
    let ids = collection
        .openings
        .iter()
        .map(|opening| opening.id.as_str())
        .collect::<BTreeSet<_>>();
    let inputs = collection
        .openings
        .iter()
        .map(collection_position_command)
        .collect::<BTreeSet<_>>();
    check(
        evaluation.iter().all(|opening| {
            !ids.contains(opening.opening_id.as_str())
                && !inputs.contains(&opening.position_command)
        }),
        "collection and final evaluation opening IDs/exact registered input must be disjoint",
    )
}

fn collection_position_command(opening: &GenerationOpeningV3) -> String {
    let mut command = match &opening.initial_fen {
        Some(fen) => format!(
            "position fen {}",
            fen.split_whitespace().collect::<Vec<_>>().join(" ")
        ),
        None => "position startpos".to_owned(),
    };
    if !opening.moves.is_empty() {
        command.push_str(" moves ");
        command.push_str(&opening.moves.join(" "));
    }
    command
}

#[cfg(target_os = "linux")]
fn make_spec(
    declaration: &GenerationPipelineDeclarationV3,
    series: &GenerationVerifiedSeriesArchiveV3,
) -> Result<Value, ArenaError> {
    let first = series
        .archives()
        .first()
        .ok_or_else(|| invalid("original series empty"))?
        .request();
    let producer = &series.dataset_manifest()["producer"];
    let source = producer["source_sha256"]
        .as_str()
        .ok_or_else(|| invalid("original collector source pin"))?;
    let model = producer["model_sha256"]
        .as_str()
        .ok_or_else(|| invalid("original collector model pin"))?;
    let runtime = producer["runtime_sha256"]
        .as_str()
        .ok_or_else(|| invalid("original collector runtime pin"))?;
    check(
        [source, model, runtime].into_iter().all(generation_sha)
            && series.archives().iter().all(|game| {
                game.request().model_epoch == first.model_epoch
                    && game.request().context_epoch == first.context_epoch
            }),
        "generation original source/epoch corpus join",
    )?;
    Ok(
        serde_json::json!({"schema":SPEC_DOMAIN,"generation_id":declaration.generation_id,
        "source_sha256":source,"dataset_sha256":series.dataset_manifest_sha256(),
        "split_sha256":first.plan.split_sha256()?,"model_sha256":model,"runtime_sha256":runtime,
        "profile":"virtual_review_v1","model_epoch":first.model_epoch,"context_epoch":first.context_epoch,
        "device":"cuda:0","precision":"fp32","tf32":false,"cpu_threads":2,"microbatch":1,"accumulation":8,
        "optimizer":{"name":"adamw","lr":1e-4,"betas":[0.9,0.999],"epsilon":1e-8,"weight_decay":0.01,"clip_norm":1.0,"schedule":"constant"},
        "collection_inference":{"mode":"fresh","batch_size":1},"role_updates":{"proposer":2000,"critic":2000,"validator":1000},
        "checkpoint_interval":250,"seeds":{"initialization":23,"sampling":29,"training":31,"arena":1},
        "loss_weights":{"policy":1.0,"wdl":1.0,"divergence":1.0,"task":1.0},"eligibility":"actual_training_pending_acceptance"}),
    )
}

fn completed_updates(receipt: &Value) -> Result<[u64; 3], ArenaError> {
    let completed = &receipt["window"]["ledger"]["completed"];
    let value = ["proposer", "critic", "validator"].map(|role| completed[role].as_u64());
    match value {
        [Some(p), Some(c), Some(v)] => Ok([p, c, v]),
        _ => Err(invalid("original trainer completed ledger absent")),
    }
}

fn verified_completed_boundary(
    trainer: Option<&VerifiedGenerationTrainingWindowV3>,
) -> Result<[u64; 3], ArenaError> {
    let Some(trainer) = trainer else {
        return Ok([0, 0, 0]);
    };
    let [p, c, v] = completed_updates(trainer.receipt())?;
    check(
        trainer.durable_resume_available()
            && p <= 2000
            && c <= 2000
            && v <= 1000
            && p == c
            && (p + c + v).is_multiple_of(250)
            && (v == 0 || p == 2000 && c == 2000),
        "completed-stage continuation requires verified durable registered role boundary",
    )?;
    Ok([p, c, v])
}

fn pc_terminal_owner(
    trainer: &Arc<VerifiedGenerationTrainingWindowV3>,
) -> Result<Arc<VerifiedGenerationTrainingWindowV3>, ArenaError> {
    let mut current = Arc::clone(trainer);
    // One actual generation has only 20 original 250-update windows. This is
    // an owner-chain lookup, never a reconstruction from its receipt bytes.
    for _ in 0..20 {
        if completed_updates(current.receipt())? == [2000, 2000, 0] {
            check(
                current.durable_resume_available(),
                "original 4000 boundary must remain durable",
            )?;
            return Ok(current);
        }
        current = Arc::clone(current.previous().ok_or_else(|| {
            invalid("original PC terminal owner missing from retained trainer chain")
        })?);
    }
    Err(invalid(
        "original trainer chain exceeded one registered generation",
    ))
}

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn train_window(
    declaration: &GenerationPipelineDeclarationV3,
    series: &Arc<GenerationVerifiedSeriesArchiveV3>,
    actions: Option<Arc<VerifiedActionOutcomeArchiveV3>>,
    owner: &mut PipelineOwnerV3,
    spec: &GenerationTrainingFilePinV3,
    pc_dataset: &Value,
    model: &str,
    runtime: &str,
    phase: &str,
    index: u32,
    before: u64,
    cancel: &Arc<AtomicBool>,
) -> Result<(), ArenaError> {
    let started = Instant::now();
    let until = started + Duration::from_millis(declaration.runtime.window_ms);
    let whole = until + Duration::from_millis(declaration.runtime.cleanup_ms);
    scan(declaration, until, cancel)?;
    let previous = owner.trainer.as_ref().map(Arc::clone);
    let checkpoint = match previous.as_ref() {
        Some(previous) => previous
            .checkpoint_files()
            .last()
            .map(|(pin, _)| pin.clone())
            .ok_or_else(|| invalid("original durable checkpoint absent"))?,
        None => {
            check(
                before == 0 && phase == "pc",
                "pipeline initializer only at declared0 boundary",
            )?;
            declaration.initial_checkpoint.clone()
        }
    };
    let v_dataset=actions.as_ref().map(|actions| {
        let manifest=serde_json::to_value(actions.manifest()).map_err(|error|invalid(error.to_string()))?;
        let digest=generation_digest(GENERATION_CHUNK_DOMAIN,&manifest).map_err(|error|invalid(error.to_string()))?;
        let pin=actions.held_files().next().ok_or_else(||invalid("original action manifest FD absent"))?.0;
        Ok::<_,ArenaError>(serde_json::json!({"manifest":{"path":actions.directory().join(&pin.path),"sha256":pin.sha256,"bytes":pin.bytes},
            "manifest_sha256":digest,"corpus_root":actions.directory()}))
    }).transpose()?;
    let window_id = format!("{}-{phase}-{index:02}", declaration.generation_id);
    let registration = serde_json::json!({"schema":TRAIN_REGISTRATION,"window_id":window_id,"spec":spec,"phase":phase,
        "datasets":{"pc":pc_dataset,"v":v_dataset},"checkpoint":{"artifact":checkpoint,
            "domain":if previous.is_some(){CHECKPOINT_DOMAIN}else{"rovezero.pals-model-semantics.v3"},
            "mode":if previous.is_some(){"resume"}else{"initialize"}},
        "source_components":declaration.runtime.source_components,"runtime_sha256":runtime,"model_sha256":model,
        "frozen_pc_sha256":actions.as_ref().map(|actions|&actions.plan().frozen_pc_weights_sha256),
        "model_epoch":actions.as_ref().map(|actions|actions.plan().frozen_pc_epoch),"training_runtime":declaration.runtime.training_runtime,
        "staging_root":declaration.managed_root,"index_root":declaration.index_directory,
        "checkpoint_directory":declaration.checkpoint_directory,"receipt_path":declaration.metadata_directory.join(format!("{window_id}-receipt.json")),
        "max_wall_ns":declaration.runtime.window_ms*1_000_000,"max_updates":250,"max_output_bytes":declaration.max_staging_bytes});
    let (pin, file) = publish_metadata_checked(
        &declaration.managed_root,
        declaration.max_staging_bytes,
        &declaration.metadata_directory,
        &format!("{window_id}-registration.json"),
        &registration,
        until,
        cancel,
    )?;
    let registration_sha = generation_digest(TRAIN_REGISTRATION, &registration)
        .map_err(|error| invalid(error.to_string()))?;
    owner.metadata.push((
        pin.clone(),
        file.try_clone()
            .map_err(|_| invalid("original registration descriptor clone"))?,
    ));
    let launch = GenerationTrainingLaunchV3 {
        interpreter: open_file(&declaration.runtime.python_runtime.executable.path)?,
        python_runtime: declaration.runtime.python_runtime.clone(),
        registration: file,
        registration_pin: pin,
        registration_sha256: registration_sha,
        execution_id: window_id.clone(),
        dependency_directories: declaration.runtime.dependency_directories.clone(),
        runtime_module_pins: declaration.runtime.runtime_module_pins.clone(),
        started,
        execution_deadline: until,
        whole_deadline: whole,
        limits: ProcessLimits {
            wall_ms: declaration.runtime.window_ms,
            shutdown_grace_ms: declaration.runtime.cleanup_ms,
            max_output_bytes: 128 * 1024,
            max_child_processes: 1,
        },
        max_read_bytes: declaration.runtime.max_read_bytes,
    };
    let trained = match run_generation_training_original_v3(
        launch,
        Arc::clone(series),
        actions,
        previous,
        cancel,
    ) {
        Ok(trained) => Arc::new(trained),
        Err(failure) => {
            let error = invalid(failure.error.to_string());
            owner.stage_failure = Some(PipelineStageFailureV3::Training(failure));
            return Err(error);
        }
    };
    // Retain first; even an anomalous successor's original checkpoint remains
    // charged evidence, rather than being dropped by a failed postcondition.
    owner.trainer = Some(Arc::clone(&trained));
    let completed = verified_completed_boundary(Some(&trained))?;
    check(
        completed.iter().sum::<u64>() == before + 250 && trained.durable_resume_available(),
        "pipeline requires exact250 completed durable boundary; no implicit retry of partial work",
    )?;
    owner.journal.push(
        serde_json::json!({"stage":window_id,"completed_updates":completed,
        "receipt":trained.parent_receipt(),"registered_wall_ms":declaration.runtime.window_ms,
        "registered_cleanup_ms":declaration.runtime.cleanup_ms}),
    );
    scan(declaration, whole, cancel)?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn export_stage(
    declaration: &GenerationPipelineDeclarationV3,
    trainer: Arc<VerifiedGenerationTrainingWindowV3>,
    kind: GenerationExportKindV3,
    owner: &mut PipelineOwnerV3,
    cancel: &Arc<AtomicBool>,
) -> Result<VerifiedGenerationExportV3, ArenaError> {
    let started = Instant::now();
    let until = started + Duration::from_millis(declaration.export_ms);
    let whole = until + Duration::from_millis(declaration.runtime.cleanup_ms);
    scan(declaration, until, cancel)?;
    let (label, directory) = match kind {
        GenerationExportKindV3::FrozenPcCollection => {
            ("frozen4000", &declaration.frozen_export_directory)
        }
        GenerationExportKindV3::G1ActualGeneration => ("g1-5000", &declaration.g1_export_directory),
    };
    let export_id = format!("{}-{label}", declaration.generation_id);
    let checkpoint = trainer
        .checkpoint_files()
        .last()
        .map(|(pin, _)| pin.clone())
        .ok_or_else(|| invalid("original exporter terminal checkpoint absent"))?;
    let registration = serde_json::json!({"schema":EXPORT_REGISTRATION,"export_id":export_id,"kind":match kind {
        GenerationExportKindV3::FrozenPcCollection=>"frozen_pc_collection",GenerationExportKindV3::G1ActualGeneration=>"g1_actual_generation"},
        "training_registration":trainer.registration_value()?,"original_training_registration_sha256":trainer.registration_sha256(),
        "trainer_receipt":trainer.receipt_file_pin().ok_or_else(||invalid("original trainer terminal receipt pin absent"))?,
        "original_training_execution_id":trainer.execution_id(),"checkpoint":checkpoint,"rules_profile":declaration.rules_profile,
        "export_source_components":declaration.runtime.export_source_components,"model_epoch":declaration.frozen_model_epoch,
        "index_root":declaration.index_directory,"output_directory":directory,
        "receipt_path":declaration.metadata_directory.join(format!("{export_id}-receipt.json")),
        "max_wall_ns":declaration.export_ms*1_000_000,"max_output_bytes":declaration.max_staging_bytes});
    let (pin, file) = publish_metadata_checked(
        &declaration.managed_root,
        declaration.max_staging_bytes,
        &declaration.metadata_directory,
        &format!("{export_id}-registration.json"),
        &registration,
        until,
        cancel,
    )?;
    let registration_sha = generation_digest(EXPORT_REGISTRATION, &registration)
        .map_err(|error| invalid(error.to_string()))?;
    owner.metadata.push((
        pin.clone(),
        file.try_clone()
            .map_err(|_| invalid("original export registration clone"))?,
    ));
    let launch = GenerationExportLaunchV3 {
        interpreter: open_file(&declaration.runtime.python_runtime.executable.path)?,
        python_runtime: declaration.runtime.python_runtime.clone(),
        registration: file,
        registration_pin: pin,
        registration_sha256: registration_sha,
        execution_id: export_id.clone(),
        dependency_directories: declaration.runtime.dependency_directories.clone(),
        runtime_module_pins: declaration.runtime.runtime_module_pins.clone(),
        started,
        execution_deadline: until,
        whole_deadline: whole,
        limits: ProcessLimits {
            wall_ms: declaration.export_ms,
            shutdown_grace_ms: declaration.runtime.cleanup_ms,
            max_output_bytes: 128 * 1024,
            max_child_processes: 1,
        },
        max_read_bytes: declaration.runtime.max_read_bytes,
    };
    let exported = match run_generation_export_original_v3(launch, trainer, cancel) {
        Ok(exported) => exported,
        Err(failure) => {
            let error = invalid(failure.error.to_string());
            owner.stage_failure = Some(PipelineStageFailureV3::Export(failure));
            return Err(error);
        }
    };
    owner.journal.push(serde_json::json!({"stage":export_id,"receipt":exported.parent_receipt(),
        "registered_wall_ms":declaration.export_ms,"native_numeric_acceptance":false,"arena_accepted":false}));
    Ok(exported)
}

#[cfg(target_os = "linux")]
fn check_live(until: Instant, cancel: &AtomicBool) -> Result<(), ArenaError> {
    check(
        Instant::now() < until && !cancel.load(Ordering::Acquire),
        "pipeline original stage canceled/expired",
    )
}
#[cfg(target_os = "linux")]
fn scan(
    declaration: &GenerationPipelineDeclarationV3,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<SelfplayCliStagingObservationV3, ArenaError> {
    check_live(until, cancel)?;
    let root = super::selfplay_cli::open_directory(&declaration.managed_root)?;
    super::selfplay_cli::private_root(&root)?;
    super::selfplay_cli::scan_staging(&root, declaration.max_staging_bytes, until, cancel)
}
#[cfg(target_os = "linux")]
fn open_file(path: &Path) -> Result<File, ArenaError> {
    check(absolute_safe(path), "pipeline nonsecret original file")?;
    let root = super::selfplay_cli::open_directory(
        path.parent()
            .ok_or_else(|| invalid("pipeline file parent"))?,
    )?;
    super::selfplay_cli::open_regular(
        &root,
        path.file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| invalid("pipeline filename"))?,
    )
}
#[cfg(target_os = "linux")]
fn same_original_inode(original: &std::fs::Metadata, current: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    original.dev() == current.dev()
        && original.ino() == current.ino()
        && original.len() == current.len()
        && original.mtime() == current.mtime()
        && original.mtime_nsec() == current.mtime_nsec()
        && original.ctime() == current.ctime()
        && original.ctime_nsec() == current.ctime_nsec()
}

#[cfg(target_os = "linux")]
fn read_original_input(
    owner: &PipelineOwnerV3,
    pin: &GenerationTrainingFilePinV3,
    maximum: u64,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<(File, Vec<u8>), ArenaError> {
    let (named, raw) = read_declared(pin, maximum, until, cancel)?;
    if let Some((original_pin, held)) = owner
        .metadata
        .iter()
        .find(|(original, _)| original.path == pin.path)
    {
        check(
            original_pin == pin,
            "continuation input must retain the exact original file pin",
        )?;
        let original = held
            .metadata()
            .map_err(|_| invalid("original continuation input descriptor"))?;
        let current = named
            .metadata()
            .map_err(|_| invalid("current continuation input descriptor"))?;
        check(
            same_original_inode(&original, &current),
            "continuation input cannot replace the original held inode even with identical bytes",
        )?;
        return Ok((
            held.try_clone()
                .map_err(|_| invalid("original continuation input FD clone"))?,
            raw,
        ));
    }
    Ok((named, raw))
}

#[cfg(target_os = "linux")]
fn read_declared(
    pin: &GenerationTrainingFilePinV3,
    maximum: u64,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<(File, Vec<u8>), ArenaError> {
    use sha2::{Digest, Sha256};
    use std::os::unix::fs::{FileExt, MetadataExt};
    validate_pin(pin, maximum)?;
    check_live(until, cancel)?;
    let file = open_file(&pin.path)?;
    let before = file
        .metadata()
        .map_err(|_| invalid("pipeline original metadata"))?;
    check(
        before.len() == pin.bytes,
        "pipeline original declared bytes",
    )?;
    let mut bytes = vec![0; pin.bytes as usize];
    let mut offset = 0;
    while offset < bytes.len() {
        check_live(until, cancel)?;
        let take = (bytes.len() - offset).min(64 * 1024);
        let count = file
            .read_at(&mut bytes[offset..offset + take], offset as u64)
            .map_err(|_| invalid("pipeline bounded original read"))?;
        check(count > 0, "pipeline truncated original input")?;
        offset += count;
    }
    let current = open_file(&pin.path)?
        .metadata()
        .map_err(|_| invalid("pipeline named metadata"))?;
    let identity = |metadata: &std::fs::Metadata| {
        (
            metadata.dev(),
            metadata.ino(),
            metadata.len(),
            metadata.mtime(),
            metadata.mtime_nsec(),
            metadata.ctime(),
            metadata.ctime_nsec(),
        )
    };
    check(
        identity(&before) == identity(&current)
            && format!("{:x}", Sha256::digest(&bytes)) == pin.sha256,
        "pipeline original input identity/hash changed",
    )?;
    Ok((file, bytes))
}
#[cfg(target_os = "linux")]
fn publish_metadata(
    directory: &Path,
    name: &str,
    value: &Value,
) -> Result<(GenerationTrainingFilePinV3, File), ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    use sha2::{Digest, Sha256};
    use std::io::Write;
    check(
        !name.is_empty()
            && name.len() <= 128
            && name.ends_with(".json")
            && name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || [b'-', b'_', b'.'].contains(&c)),
        "pipeline exclusive metadata basename",
    )?;
    let mut value = value.clone();
    value.sort_all_objects();
    let mut bytes = serde_json::to_vec(&value).map_err(|error| invalid(error.to_string()))?;
    bytes.push(b'\n');
    check(
        bytes.len() as u64 <= MAX_METADATA,
        "pipeline metadata byte capacity",
    )?;
    let root = super::selfplay_cli::open_directory(directory)?;
    super::selfplay_cli::private_root(&root)?;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    let mut writer = root
        .open_with(name, &options)
        .map_err(|_| invalid("pipeline exclusive source-owned metadata publication"))?;
    writer
        .write_all(&bytes)
        .and_then(|_| writer.sync_all())
        .map_err(|_| invalid("pipeline durable metadata publication"))?;
    drop(writer);
    let pin = GenerationTrainingFilePinV3 {
        path: directory.join(name),
        sha256: format!("{:x}", Sha256::digest(&bytes)),
        bytes: bytes.len() as u64,
    };
    let file = super::selfplay_cli::open_regular(&root, name)?;
    // The next original producer verifies this readonly descriptor/hash before
    // dispatch; publishing metadata itself grants no game/native authority.
    Ok((pin, file))
}

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn publish_or_retain_completed_metadata(
    declaration: &GenerationPipelineDeclarationV3,
    owner: &PipelineOwnerV3,
    directory: &Path,
    name: &str,
    value: &Value,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<(GenerationTrainingFilePinV3, File), ArenaError> {
    use sha2::{Digest as _, Sha256};
    let path = directory.join(name);
    if let Some((pin, held)) = owner.metadata.iter().find(|(pin, _)| pin.path == path) {
        let mut canonical = value.clone();
        canonical.sort_all_objects();
        let mut raw = serde_json::to_vec(&canonical).map_err(|error| invalid(error.to_string()))?;
        raw.push(b'\n');
        check(
            raw.len() as u64 == pin.bytes && format!("{:x}", Sha256::digest(&raw)) == pin.sha256,
            "completed metadata must retain the exact original opaque-owner projection",
        )?;
        let (named, _) = read_declared(pin, MAX_METADATA, until, cancel)?;
        let original = held
            .metadata()
            .map_err(|_| invalid("original completed metadata descriptor"))?;
        let current = named
            .metadata()
            .map_err(|_| invalid("current completed metadata descriptor"))?;
        check(
            same_original_inode(&original, &current),
            "completed metadata must preserve original named inode and contents",
        )?;
        return Ok((
            pin.clone(),
            held.try_clone()
                .map_err(|_| invalid("completed metadata original FD clone"))?,
        ));
    }
    // Existing unowned/partial files are not overwritten or promoted here.
    publish_metadata_checked(
        &declaration.managed_root,
        declaration.max_staging_bytes,
        directory,
        name,
        value,
        until,
        cancel,
    )
}

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn publish_metadata_checked(
    managed_root: &Path,
    maximum: u64,
    directory: &Path,
    name: &str,
    value: &Value,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<(GenerationTrainingFilePinV3, File), ArenaError> {
    check_live(until, cancel)?;
    check(
        directory.starts_with(managed_root) && maximum <= MAX_STAGING,
        "pipeline metadata within original staging budget",
    )?;
    let mut canonical = value.clone();
    canonical.sort_all_objects();
    let required = serde_json::to_vec(&canonical)
        .map_err(|error| invalid(error.to_string()))?
        .len() as u64
        + 1;
    check(required <= MAX_METADATA, "pipeline metadata byte capacity")?;
    let root = super::selfplay_cli::open_directory(managed_root)?;
    let observed = super::selfplay_cli::scan_staging(&root, maximum, until, cancel)?;
    check(
        observed
            .unique_inode_bytes
            .checked_add(required)
            .is_some_and(|total| total <= maximum),
        "pipeline metadata cannot exceed original18GiB total staging bound",
    )?;
    publish_metadata(directory, name, value)
}

fn check_numerical_preregistration(
    declaration: &GenerationPipelineDeclarationV3,
    registration: &GenerationNumericalLaunchRegistrationV3,
    template: &GenerationTrainingFilePinV3,
) -> Result<(), ArenaError> {
    check(
        registration.schema == "rz-pals-generation-numerical-launch/1"
            && plain_identifier(&registration.numerical_id)
            && &registration.template_startup == template
            && registration.python_runtime == declaration.runtime.python_runtime
            && registration.max_wall_ns == declaration.numerical_ms * 1_000_000
            && registration.max_output_bytes == declaration.max_staging_bytes
            && (1..=declaration.runtime.max_read_bytes).contains(&registration.max_read_bytes)
            && registration
                .acceptance_path
                .starts_with(&declaration.private_cwd)
            && [
                &registration.reference_output_root,
                &registration.native_output_root,
                &registration.receipt_path,
                &registration.acceptance_path,
            ]
            .into_iter()
            .all(|path| {
                absolute_safe(path)
                    && path.starts_with(&declaration.managed_root)
                    && path != &declaration.managed_root
            })
            && registration.reference_output_root != registration.native_output_root
            && registration.receipt_path != registration.acceptance_path,
        "numerical preregistration must bind original template/runtime/window and complete same-owner staging cap",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn completed_ledger_refuses_missing_or_dispatched_only_work() {
        assert!(completed_updates(&serde_json::json!({"window":{"ledger":{"dispatched":{"proposer":2000,"critic":2000,"validator":1000}}}})).is_err());
        assert_eq!(completed_updates(&serde_json::json!({"window":{"ledger":{"completed":{"proposer":2000,"critic":2000,"validator":0}}}})).unwrap(),[2000,2000,0]);
    }
    #[test]
    fn explicit_generation_names_do_not_cross_owned_metadata_namespace() {
        for name in [
            "",
            "../../outside",
            "hidden/name",
            "space name",
            "x.json",
            "_prefix",
        ] {
            assert!(!plain_identifier(name));
        }
        assert!(plain_identifier("generation-one-23"));
    }
    #[test]
    fn opening_input_overlap_keeps_imported_history_distinct_and_normalizes_spacing() {
        let mut opening = GenerationOpeningV3 {
            id: "collection-01".into(),
            initial_fen: None,
            moves: vec!["e2e4".into(), "e7e5".into()],
            split: GenerationSplitV3::Train,
        };
        assert_eq!(
            collection_position_command(&opening),
            "position startpos moves e2e4 e7e5"
        );
        opening.initial_fen =
            Some("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR   w KQkq - 0 1".into());
        assert_eq!(
            collection_position_command(&opening),
            "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4 e7e5"
        );
        opening.moves.clear();
        assert_eq!(
            collection_position_command(&opening),
            "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn continuation_input_rejects_identical_bytes_in_replacement_inode_and_preserves_cursor() {
        use sha2::{Digest as _, Sha256};
        use std::io::{Seek, SeekFrom};
        use std::os::unix::fs::PermissionsExt;
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let root = std::env::temp_dir().join(format!(
            "rz-generation-input-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = root.join("input.json");
        std::fs::write(&path, b"abc").unwrap();
        let pin = GenerationTrainingFilePinV3 {
            path: path.clone(),
            sha256: format!("{:x}", Sha256::digest(b"abc")),
            bytes: 3,
        };
        let mut held = open_file(&path).unwrap();
        held.seek(SeekFrom::Start(2)).unwrap();
        let mut owner = PipelineOwnerV3::new();
        owner.metadata.push((pin.clone(), held));
        let (mut retained, bytes) = read_original_input(
            &owner,
            &pin,
            16,
            Instant::now() + Duration::from_secs(5),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(bytes, b"abc");
        assert_eq!(retained.stream_position().unwrap(), 2);
        std::fs::rename(&path, root.join("original.json")).unwrap();
        std::fs::write(&path, b"abc").unwrap();
        assert!(
            read_original_input(
                &owner,
                &pin,
                16,
                Instant::now() + Duration::from_secs(5),
                &AtomicBool::new(false)
            )
            .is_err()
        );
        drop(retained);
        drop(owner);
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(root.join("original.json")).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
}
