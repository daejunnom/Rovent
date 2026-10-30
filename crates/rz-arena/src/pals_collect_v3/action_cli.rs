//! Explicit real V-action collection over retained original game owners.
//! The normal orchestrator passes existing opaque archives across P/C freeze;
//! the standalone command may execute an explicitly declared fresh source run.
//! Neither path upgrades JSON into a native, game, or process capability.
use super::*;
#[cfg(target_os = "linux")]
use crate::process::ProcessLimits;
use rz_experiments::ArtifactRef;
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, atomic::AtomicBool};
#[cfg(target_os = "linux")]
use std::time::{Duration, Instant};

pub const ACTION_CLI_DECLARATION_DOMAIN_V3: &str = "rz-pals-action-outcomes-cli-declaration/1";
#[cfg(target_os = "linux")]
const MAX_DECLARATION_BYTES: u64 = 4 * 1024 * 1024;
#[cfg(target_os = "linux")]
const MAX_PROBE_BYTES: u64 = 32 * 1024 * 1024;
const MAX_ROOT_READ_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionCliSavedRootV3 {
    pub game_ordinal: u16,
    pub decision_index: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionCliFrozenIntakeV3 {
    pub interpreter_root: PathBuf,
    pub interpreter: ArtifactRef,
    pub request_root: PathBuf,
    pub request: ArtifactRef,
    pub artifact_directory: PathBuf,
    pub scratch_directory: PathBuf,
    pub dependency_directories: Vec<PathBuf>,
    pub execution_ms: u64,
    pub cleanup_ms: u64,
}
/// A declaration of intended work. The roots refer to original opaque game
/// archives supplied to the typed entry point, never to proof JSON filenames.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionCliDeclarationV3 {
    pub schema: String,
    pub plan: ActionOutcomeCollectionPlanV3,
    pub openings: SelfplayCollectionPlanV3,
    pub producer: GenerationProducerAuthorityV3,
    pub context_epoch: u64,
    pub managed_root: PathBuf,
    pub private_cwd: PathBuf,
    pub startup_config: PathBuf,
    pub startup_lock: PathBuf,
    pub startup_config_sha256: String,
    pub output_directory: PathBuf,
    pub intake: ActionCliFrozenIntakeV3,
    pub preflight_ms: u64,
    pub context_ms: u64,
    pub process_ms: u64,
    pub shutdown_reserve_ms: u64,
    pub publication_ms: u64,
    pub max_staging_bytes: u64,
    pub max_root_read_bytes: u64,
    /// Independent final read-only manifest/chunk verification credit.
    pub max_publication_read_bytes: u64,
    pub roots: Vec<ActionCliSavedRootV3>,
}
impl ActionCliDeclarationV3 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        self.validate_with_minimum_roots(self.plan.desired_contexts as usize)
    }

    // Diagnostics retain the production plan and its 2048/8192 limits, but
    // need only one actually saved root. This never authorizes completion of
    // the production generation or a training dataset.
    fn validate_with_minimum_roots(&self, minimum_roots: usize) -> Result<(), ArenaError> {
        self.plan.validate()?;
        self.openings.validate()?;
        self.producer.validate(&self.openings.opening_splits())?;
        self.intake.interpreter.validate()?;
        self.intake.request.validate()?;
        check(
            self.schema == ACTION_CLI_DECLARATION_DOMAIN_V3
                && self.context_epoch > 0
                && self.producer.input_seals.is_empty()
                && self.producer.actual_game_ids.is_empty()
                && self.producer.model_sha256 == self.plan.frozen_pc_weights_sha256
                && self.producer.opening_split_sha256 == self.openings.split_sha256()?
                && generation_sha(&self.startup_config_sha256)
                && (1..=300_000).contains(&self.preflight_ms)
                && (1..=3_600_000).contains(&self.context_ms)
                && (1..=300_000).contains(&self.process_ms)
                && (1..=30_000).contains(&self.shutdown_reserve_ms)
                && self.shutdown_reserve_ms < self.process_ms
                && self.process_ms <= self.context_ms
                && (1..=1_800_000).contains(&self.publication_ms)
                && (1..=300_000).contains(&self.intake.execution_ms)
                && (1..=5000).contains(&self.intake.cleanup_ms)
                && self.intake.dependency_directories.len() <= 8
                && self.intake.interpreter.bytes <= 64 * 1024 * 1024
                && self.intake.request.bytes <= 128 * 1024
                && self.max_staging_bytes > 0
                && self.max_staging_bytes <= GENERATION_OUTPUT_CAP
                && self.plan.maximum_output_bytes < self.max_staging_bytes
                && self.max_root_read_bytes > 0
                && self.max_root_read_bytes <= 8192 * MAX_ROOT_READ_BYTES
                && self.max_publication_read_bytes > 0
                && self.max_publication_read_bytes <= 2 * GENERATION_OUTPUT_CAP
                && self.roots.len() >= minimum_roots,
            "actual action CLI registered epochs, bounds, frozen parameters and split",
        )?;
        let paths = [
            &self.managed_root,
            &self.private_cwd,
            &self.startup_config,
            &self.startup_lock,
            &self.output_directory,
            &self.intake.interpreter_root,
            &self.intake.request_root,
            &self.intake.artifact_directory,
            &self.intake.scratch_directory,
        ];
        check(
            paths.into_iter().all(|path| absolute_safe(path))
                && self
                    .intake
                    .dependency_directories
                    .iter()
                    .all(|path| absolute_safe(path))
                && [
                    &self.private_cwd,
                    &self.output_directory,
                    &self.intake.artifact_directory,
                    &self.intake.scratch_directory,
                    &self.intake.request_root,
                ]
                .into_iter()
                .all(|path| path.starts_with(&self.managed_root))
                && self.output_directory != self.managed_root
                && self.startup_config.starts_with(&self.private_cwd)
                && self.startup_lock.starts_with(&self.private_cwd)
                && self.intake.interpreter.path.split('/').all(nonsecret_name)
                && self.intake.request.path.split('/').all(nonsecret_name),
            "actual action CLI explicit safe roots and managed staging",
        )?;
        validate_root_schedule(&self.roots, self.plan.maximum_contexts)?;
        self.total_ms()?;
        Ok(())
    }
    fn total_ms(&self) -> Result<u64, ArenaError> {
        u64::from(self.plan.maximum_contexts)
            .checked_mul(self.context_ms)
            .and_then(|value| value.checked_add(self.preflight_ms))
            .and_then(|value| value.checked_add(self.intake.execution_ms))
            .and_then(|value| value.checked_add(self.intake.cleanup_ms))
            .and_then(|value| value.checked_add(self.shutdown_reserve_ms))
            .and_then(|value| value.checked_add(self.publication_ms))
            .ok_or_else(|| invalid("actual action CLI original total window overflow"))
    }
}
fn validate_root_schedule(roots: &[ActionCliSavedRootV3], maximum: u32) -> Result<(), ArenaError> {
    let mut seen = BTreeSet::new();
    check(
        !roots.is_empty() && roots.len() <= maximum as usize,
        "bounded saved actual roots",
    )?;
    for root in roots {
        check(
            root.game_ordinal < 128
                && root.decision_index < 512
                && seen.insert((root.game_ordinal, root.decision_index)),
            "saved actual root schedule has duplicate or out-of-range reference",
        )?;
    }
    Ok(())
}
fn nonsecret_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    !lower.starts_with(".env")
        && !lower.ends_with(".key")
        && !lower.ends_with(".pem")
        && !lower.ends_with(".p12")
        && !lower.contains("credential")
        && !lower.contains("service-account")
        && !lower.contains("service_account")
        && !lower.contains("ssh-key")
        && !lower.contains("api-key")
}
fn absolute_safe(path: &Path) -> bool {
    path.is_absolute()
        && path.components().count() <= 64
        && path.components().all(|component| match component {
            Component::RootDir => true,
            Component::Normal(name) => name.to_str().is_some_and(nonsecret_name),
            _ => false,
        })
}

type FrozenIntakeOwner =
    Arc<Mutex<Option<Result<VerifiedFrozenPcIntakeV3, Box<FrozenPcIntakeFailureV3>>>>>;

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
struct ActionCliOwnerV3 {
    archives: Vec<Arc<VerifiedGenerationGameArchiveV3>>,
    collector: Option<GenerationActionOutcomeCollectorV3>,
    frozen: FrozenIntakeOwner,
    intake_process: Option<rz_uci::pals_adaptive_bootstrap::PreparedAdaptiveProcessV5>,
    factory: Option<ActualGenerationActionOutcomeFactoryV3>,
}
/// Failure keeps the original source, frozen-intake and partial action owners.
/// Its public JSON error never asserts physical cleanup or completed learning.
pub struct ActionCliFailureV3 {
    pub error: ArenaError,
    pub preservation_error: Option<ArenaError>,
    owner: Option<Box<ActionCliOwnerV3>>,
}

/// Results of bounded cleanup attempts on the retained original owners.
/// These states are observations, not a GPU completion or training receipt.
#[derive(Debug, Clone, Serialize)]
pub struct ActionOriginalCleanupObservationV3 {
    pub action_processes: &'static str,
    pub action_receipts: usize,
    pub intake_process: &'static str,
    pub frozen_helper: &'static str,
    pub errors: Vec<String>,
}

impl ActionCliFailureV3 {
    /// Attempt original cleanup before retaining an unresolved failure owner.
    /// Each owner additionally enforces its own original deadline. Failed or
    /// still-pending child/process custody stays in `self`; an outer cgroup or
    /// process-tree observation is never used as a native completion fence.
    pub fn attempt_original_cleanup(
        &mut self,
        until: std::time::Instant,
    ) -> ActionOriginalCleanupObservationV3 {
        let mut observation = ActionOriginalCleanupObservationV3 {
            action_processes: "not_present", action_receipts: 0,
            intake_process: "not_present", frozen_helper: "not_present", errors: Vec::new(),
        };
        let Some(owner) = self.owner.as_mut() else {
            observation.action_processes = "unknown";
            observation.intake_process = "unknown";
            observation.frozen_helper = "unknown";
            observation.errors.push("original action owner absent".into());
            return observation;
        };
        if let Some(factory) = owner.factory.as_mut() {
            if std::time::Instant::now() >= until {
                observation.action_processes = "unknown";
                observation.errors.push("original cleanup deadline elapsed before action cleanup".into());
            } else {
                match factory.attempt_original_cleanup(until) {
                    Ok(None) => observation.action_processes = "not_present",
                    Ok(Some(receipts)) => {
                        observation.action_processes = if receipts.is_empty() {
                            "no_process_observed"
                        } else {
                            "closed_observed"
                        };
                        observation.action_receipts = receipts.len();
                    }
                    Err(error) => {
                        observation.action_processes = "unknown";
                        observation.errors.push(format!("original action cleanup: {error}"));
                    }
                }
            }
        }
        if let Some(process) = owner.intake_process.as_ref() {
            if std::time::Instant::now() >= until {
                observation.intake_process = "unknown";
                observation.errors.push("original cleanup deadline elapsed before intake cleanup".into());
            } else {
                match process.finish_process(until) {
                    Ok(receipt) if receipt.permits_process_close() => observation.intake_process = "closed_observed",
                    Ok(_) => {
                        observation.intake_process = "unknown";
                        observation.errors.push("original intake did not establish process close".into());
                    }
                    Err(error) => {
                        observation.intake_process = "unknown";
                        observation.errors.push(format!("original intake cleanup: {error}"));
                    }
                }
            }
        }
        match owner.frozen.try_lock() {
            Ok(frozen) => match frozen.as_ref() {
                Some(Ok(verified)) if verified.process().transport_complete() => {
                    observation.frozen_helper = "closed_transport_observed";
                }
                Some(Ok(_)) => observation.frozen_helper = "unknown_retained",
                Some(Err(failure)) => {
                    // The failed helper's process/child handle is retained by
                    // its original failure. This is no new kill/reap right.
                    observation.frozen_helper = "unknown_retained";
                    if failure.process.as_ref().is_some_and(|process| process.process().pending_child.is_some()) {
                        observation.errors.push("original frozen helper child remains in custody".into());
                    }
                }
                None => {}
            },
            Err(_) => {
                observation.frozen_helper = "unknown_retained";
                observation.errors.push("original frozen helper custody busy or poisoned".into());
            }
        }
        observation
    }
    pub fn original_archives(&self) -> &[Arc<VerifiedGenerationGameArchiveV3>] {
        self.owner
            .as_ref()
            .map_or(&[], |owner| owner.archives.as_slice())
    }
    pub fn collector(&self) -> Option<&GenerationActionOutcomeCollectorV3> {
        self.owner
            .as_ref()
            .and_then(|owner| owner.collector.as_ref())
    }
    pub fn failed_factory(&self) -> Option<&ActualGenerationActionOutcomeFactoryV3> {
        self.owner.as_ref().and_then(|owner| owner.factory.as_ref())
    }
    pub fn intake_process(
        &self,
    ) -> Option<&rz_uci::pals_adaptive_bootstrap::PreparedAdaptiveProcessV5> {
        self.owner
            .as_ref()
            .and_then(|owner| owner.intake_process.as_ref())
    }
    /// Resolve an original pending helper under its retained owner. A new JSON
    /// proof, reconstructed child or replacement process is never accepted.
    pub fn with_frozen_failure_mut<R>(
        &mut self,
        inspect: impl FnOnce(Option<&mut FrozenPcIntakeFailureV3>) -> R,
    ) -> Result<R, ArenaError> {
        let owner = self
            .owner
            .as_mut()
            .ok_or_else(|| invalid("original action owner absent"))?;
        let mut original = owner
            .frozen
            .lock()
            .map_err(|_| invalid("original frozen owner poisoned"))?;
        Ok(inspect(
            original
                .as_mut()
                .and_then(|result| result.as_mut().err().map(Box::as_mut)),
        ))
    }
}
impl std::fmt::Debug for ActionCliFailureV3 {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("ActionCliFailureV3")
            .field("error", &self.error)
            .field("preservation_error", &self.preservation_error)
            .field("original_owner_retained", &self.owner.is_some())
            .finish()
    }
}
impl std::fmt::Display for ActionCliFailureV3 {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            out,
            "{}; original_owner_retained={}; preservation_failed={}",
            self.error,
            self.owner.is_some(),
            self.preservation_error.is_some()
        )
    }
}
impl std::error::Error for ActionCliFailureV3 {}

/// Successful private collector publication, retaining its original saved
/// games and verified named-P/C intake. No parser or public constructor exists.
/// Raw status/manifest JSON is not a substitute for this actual owner.
pub struct VerifiedActionOutcomeArchiveV3 {
    status: serde_json::Value,
    directory: PathBuf,
    manifest: GenerationChunkManifestV3,
    plan: ActionOutcomeCollectionPlanV3,
    archives: Vec<Arc<VerifiedGenerationGameArchiveV3>>,
    frozen: FrozenIntakeOwner,
    #[cfg(target_os = "linux")]
    root: cap_std::fs::Dir,
    #[cfg(target_os = "linux")]
    files: Vec<ActionArchiveFileV3>,
}
impl VerifiedActionOutcomeArchiveV3 {
    pub fn status(&self) -> &serde_json::Value {
        &self.status
    }
    pub fn directory(&self) -> &Path {
        &self.directory
    }
    pub fn manifest(&self) -> &GenerationChunkManifestV3 {
        &self.manifest
    }
    /// Semantic digest of the original completed manifest, not its file hash.
    pub fn manifest_sha256(&self) -> Result<String, ArenaError> {
        generation_digest(
            "rz-pals-action-outcomes-completed-manifest/1",
            &self.manifest,
        )
        .map_err(Into::into)
    }
    #[cfg(target_os = "linux")]
    pub fn manifest_file_sha256(&self) -> &str {
        &self.files[0].pin.sha256
    }
    pub fn plan(&self) -> &ActionOutcomeCollectionPlanV3 {
        &self.plan
    }
    pub fn source_game_archives(&self) -> &[Arc<VerifiedGenerationGameArchiveV3>] {
        &self.archives
    }
    /// Auxiliary readback, carrying no ability to mint an intake or train.
    pub fn frozen_intake_receipt(&self) -> Result<serde_json::Value, ArenaError> {
        self.frozen
            .lock()
            .map_err(|_| invalid("original frozen owner poisoned"))?
            .as_ref()
            .ok_or_else(|| invalid("original frozen intake absent"))?
            .as_ref()
            .map(|intake| intake.receipt().clone())
            .map_err(|failure| invalid(failure.error.to_string()))
    }
    /// Original readonly descriptors for the parent's pinned trainer launch.
    /// Opening caller JSON at matching names cannot create this owner.
    #[cfg(target_os = "linux")]
    pub fn held_files(&self) -> impl Iterator<Item = (&ArtifactRef, &std::fs::File)> {
        self.files.iter().map(|held| (&held.pin, &held.file))
    }
    pub fn reverify_owned_artifacts(
        &self,
        until: std::time::Instant,
        cancel: &AtomicBool,
        max_read_bytes: u64,
    ) -> Result<u64, ArenaError> {
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (until, cancel, max_read_bytes);
            Err(invalid("action artifact custody requires Linux"))
        }
        #[cfg(target_os = "linux")]
        {
            let named = super::selfplay_cli::open_directory(&self.directory)?;
            same_directory(&self.root, &named)?;
            let mut credit = max_read_bytes;
            check(
                credit > 0 && credit <= 2 * GENERATION_OUTPUT_CAP,
                "action original publication reverify bound",
            )?;
            for held in &self.files {
                verify_archive_file(&self.root, held, until, cancel, &mut credit)?;
            }
            Ok(max_read_bytes - credit)
        }
    }
}
#[cfg(target_os = "linux")]
struct ActionArchiveFileV3 {
    pin: ArtifactRef,
    file: std::fs::File,
}
#[cfg(target_os = "linux")]
struct ActionCliPublicationV3 {
    status: serde_json::Value,
    directory: PathBuf,
    manifest: GenerationChunkManifestV3,
    plan: ActionOutcomeCollectionPlanV3,
    root: cap_std::fs::Dir,
    files: Vec<ActionArchiveFileV3>,
}

/// Use retained G0 game archives after the explicit P/C-freeze stage. No source
/// selfplay, model training, cloud operation, retry or new runtime is implicit.
pub fn run_action_outcomes_owned_cli_v3(
    declaration_path: &Path,
    declaration_sha256: &str,
    archives: Vec<Arc<VerifiedGenerationGameArchiveV3>>,
    cancel: Arc<AtomicBool>,
) -> Result<VerifiedActionOutcomeArchiveV3, Box<ActionCliFailureV3>> {
    #[allow(unused_mut)] // Non-Linux preserves custody while rejecting execution.
    let mut owner = ActionCliOwnerV3 {
        archives,
        collector: None,
        frozen: Arc::new(Mutex::new(None)),
        intake_process: None,
        factory: None,
    };
    #[cfg(not(target_os = "linux"))]
    let result = {
        let _ = (declaration_path, declaration_sha256, cancel);
        Err((
            invalid("actual action collection requires Linux original native owners"),
            None,
        ))
    };
    #[cfg(target_os = "linux")]
    let result = run_linux(declaration_path, declaration_sha256, &mut owner, cancel);
    match result {
        #[cfg(target_os = "linux")]
        Ok(publication) => Ok(VerifiedActionOutcomeArchiveV3 {
            status: publication.status,
            directory: publication.directory,
            manifest: publication.manifest,
            plan: publication.plan,
            archives: std::mem::take(&mut owner.archives),
            frozen: Arc::clone(&owner.frozen),
            root: publication.root,
            files: publication.files,
        }),
        #[cfg(not(target_os = "linux"))]
        Ok(()) => unreachable!("non-Linux execution has no successful publisher"),
        Err((error, preservation_error)) => Err(Box::new(ActionCliFailureV3 {
            error,
            preservation_error,
            owner: Some(Box::new(owner)),
        })),
    }
}

/// One actual game/root diagnostic. This is intentionally a different type
/// from `VerifiedActionOutcomeArchiveV3`: its partial archive cannot become a
/// generation dataset or authorize an optimizer update.
pub struct ActionDiagnosticOutcomeV3 {
    status: serde_json::Value,
    partial: GenerationPartialArchiveV3,
    _owner: Box<ActionCliOwnerV3>,
}

impl ActionDiagnosticOutcomeV3 {
    pub fn status(&self) -> &serde_json::Value { &self.status }
    pub fn partial_archive(&self) -> &GenerationPartialArchiveV3 { &self.partial }
}

/// Runs only the first declared saved root from the one supplied original
/// game. The actual probe, supported actions and original closure are the
/// same owners used by the production collector. `wall` covers the whole
/// diagnostic, including partial archive preservation; it is not renewed.
pub fn run_action_diagnostic_owned_v3(
    declaration_path: &Path,
    declaration_sha256: &str,
    archive: Arc<VerifiedGenerationGameArchiveV3>,
    selected_decision_index: usize,
    cancel: Arc<AtomicBool>,
    wall: std::time::Duration,
) -> Result<ActionDiagnosticOutcomeV3, Box<ActionCliFailureV3>> {
    #[allow(unused_mut)]
    let mut owner = ActionCliOwnerV3 {
        archives: vec![archive], collector: None,
        frozen: Arc::new(Mutex::new(None)), intake_process: None, factory: None,
    };
    #[cfg(not(target_os = "linux"))]
    let result = {
        let _ = (declaration_path, declaration_sha256, selected_decision_index, cancel, wall);
        Err((invalid("actual action diagnostic requires Linux original native owners"), None))
    };
    #[cfg(target_os = "linux")]
    let result = run_diagnostic_linux(declaration_path, declaration_sha256, selected_decision_index, &mut owner, cancel, wall);
    match result {
        Ok((status, partial)) => Ok(ActionDiagnosticOutcomeV3 { status, partial, _owner: Box::new(owner) }),
        Err((error, preservation_error)) => Err(Box::new(ActionCliFailureV3 {
            error, preservation_error, owner: Some(Box::new(owner)),
        })),
    }
}

#[cfg(target_os = "linux")]
fn run_diagnostic_linux(
    path: &Path, expected: &str, selected_decision_index: usize, owner: &mut ActionCliOwnerV3,
    cancel: Arc<AtomicBool>, wall: Duration,
) -> Result<(serde_json::Value, GenerationPartialArchiveV3), (ArenaError, Option<ArenaError>)> {
    use sha2::Digest as _;
    use std::io::Read;
    let started = Instant::now();
    let until = started.checked_add(wall).unwrap_or(started);
    let mut diagnostic_root = None;
    let execute = (|| -> Result<_, ArenaError> {
        check(!wall.is_zero() && wall <= Duration::from_secs(3600)
            && absolute_safe(path) && generation_sha(expected), "diagnostic declaration or wall bound")?;
        let parent = super::selfplay_cli::open_directory(path.parent().ok_or_else(|| invalid("diagnostic declaration parent"))?)?;
        let name = path.file_name().and_then(|name| name.to_str()).ok_or_else(|| invalid("diagnostic declaration name"))?;
        let file = super::selfplay_cli::open_regular(&parent, name)?;
        let length = file.metadata().map_err(|_| invalid("diagnostic declaration metadata"))?.len();
        check(length > 0 && length <= MAX_DECLARATION_BYTES, "diagnostic declaration size")?;
        let mut bytes = Vec::with_capacity(length as usize);
        Read::take(&file, length + 1).read_to_end(&mut bytes).map_err(|_| invalid("diagnostic declaration read"))?;
        check(bytes.len() as u64 == length && format!("{:x}", sha2::Sha256::digest(&bytes)) == expected,
            "diagnostic original declaration SHA-256")?;
        let declaration: ActionCliDeclarationV3 = serde_json::from_slice(&bytes)
            .map_err(|error| invalid(format!("diagnostic declaration: {error}")))?;
        declaration.validate_with_minimum_roots(1)?;
        check(declaration.roots.len() == 1 && declaration.roots[0].game_ordinal == 0
            && declaration.roots[0].decision_index == selected_decision_index,
            "diagnostic requires exactly one root of the original supplied game")?;
        let preflight_until = (started + Duration::from_millis(declaration.preflight_ms)).min(until);
        check_live(preflight_until, &cancel)?;
        let managed = super::selfplay_cli::open_directory(&declaration.managed_root)?;
        super::selfplay_cli::private_root(&managed)?;
        super::selfplay_cli::scan_staging(&managed, declaration.max_staging_bytes, preflight_until, &cancel)?;
        let archive = &owner.archives[0];
        check(archive.request().game_ordinal == 0
            && serde_json::to_vec(&archive.request().plan.openings).map_err(|e| invalid(e.to_string()))?
                == serde_json::to_vec(&declaration.openings.openings).map_err(|e| invalid(e.to_string()))?
            && archive.cwd_path().starts_with(&declaration.managed_root),
            "diagnostic original game owner or openings differ")?;
        let cwd = std::env::current_dir().map_err(|_| invalid("diagnostic current directory"))?;
        check(cwd == declaration.private_cwd, "diagnostic private cwd differs")?;
        let pinned = Arc::new(rz_uci::pals_adaptive_startup::PinnedPalsStartupV5::open(
            rz_uci::pals_adaptive_startup::PalsStartupOptionsV5 {
                config_path: declaration.startup_config.clone(), lock_path: declaration.startup_lock.clone(),
                config_sha256: declaration.startup_config_sha256.clone(), metadata_preflight: false,
            }, &cwd)?);
        rz_experiments::verify_compiled_source_components_v5(&pinned.config.source_components)?;
        check(pinned.config.source_components.get("arena") == Some(&declaration.producer.source_sha256)
            && pinned.asset(rz_experiments::PalsStartupAssetPurposeV5::RuntimeLibrary)?.declaration.artifact.sha256
                == declaration.producer.runtime_sha256,
            "diagnostic compiled collector/runtime source pins")?;
        check_live(preflight_until, &cancel)?;
        load_verified_frozen(&declaration, Arc::clone(&pinned), owner, Arc::clone(&cancel), until)?;
        let registration = owner.frozen.lock().map_err(|_| invalid("diagnostic frozen owner poisoned"))?
            .as_ref().ok_or_else(|| invalid("diagnostic frozen intake absent"))?
            .as_ref().map_err(|failure| invalid(failure.error.to_string()))?.registration().clone();
        check_live(until, &cancel)?;
        let staging = super::selfplay_cli::scan_staging(&managed, declaration.max_staging_bytes, until, &cancel)?;
        check(declaration.plan.maximum_output_bytes.checked_add(MAX_PROBE_BYTES)
            .is_some_and(|needed| needed <= declaration.max_staging_bytes.saturating_sub(staging.unique_inode_bytes)),
            "diagnostic archive and probe reserve")?;
        owner.collector = Some(GenerationActionOutcomeCollectorV3::new(
            declaration.plan.clone(), declaration.openings.clone(), declaration.producer.clone(),
            declaration.context_epoch, declaration.output_directory.clone(), until)?);
        diagnostic_root = Some(declaration.output_directory.clone());
        let selected = &declaration.roots[0];
        let (root, used) = owner.archives[0].saved_action_root(selected.decision_index, until,
            Some(&cancel), declaration.max_root_read_bytes.min(MAX_ROOT_READ_BYTES))?;
        let source_epoch = root.source_epoch();
        let issuer: GenerationActionProcessEpochIssuerV3 = Arc::new(move || loop {
            let epoch = reserve_actual_session_epochs_v3(1)?;
            if epoch.0 != source_epoch { return Ok(epoch); }
        });
        owner.factory = Some(ActualGenerationActionOutcomeFactoryV3::new(
            Arc::clone(&pinned), registration, declaration.producer.clone(), issuer,
            GenerationActualActionLimitsV3 { process_wall: Duration::from_millis(declaration.process_ms),
                shutdown_reserve: Duration::from_millis(declaration.shutdown_reserve_ms) },
        )?.with_cancellation(Arc::clone(&cancel)));
        // `run_context` may use up to 30 seconds of its registered cleanup
        // after its execution deadline. Keep that reserve inside this
        // diagnostic's original whole window.
        let cleanup_reserve = Duration::from_secs(30);
        let latest_context_end = until.checked_sub(cleanup_reserve)
            .ok_or_else(|| invalid("diagnostic cleanup reserve outside wall"))?;
        check(Instant::now() < latest_context_end, "diagnostic has no context and cleanup window")?;
        let context_until = (Instant::now() + Duration::from_millis(declaration.context_ms)).min(latest_context_end);
        let report = owner.collector.as_mut().expect("diagnostic collector owned")
            .run_context(owner.factory.as_mut().expect("diagnostic factory owned"), &root, context_until)?;
        check_live(until, &cancel)?;
        let probe = owner.factory.as_mut().expect("diagnostic factory owned").take_probe_receipt()
            .ok_or_else(|| invalid("diagnostic actual source probe missing"))?;
        let raw = serde_json::to_vec(&probe_json(&probe, Some(0), Some(selected))?).map_err(|e| invalid(e.to_string()))?;
        check(raw.len() as u64 <= MAX_PROBE_BYTES, "diagnostic source probe bound")?;
        publish_new(&declaration.output_directory, "source-probe-0000.json", &raw)?;
        owner.factory = None;
        let collector = owner.collector.as_mut().expect("diagnostic collector owned");
        collector.preserve_partial("nonpromotion one-context functional diagnostic", until)?;
        check_live(until, &cancel)?;
        let partial = collector.partial_archive().ok_or_else(|| invalid("diagnostic partial archive missing"))?.clone();
        Ok((serde_json::json!({
            "schema":"rz-pals-actual-action-diagnostic/1", "status":"one_context_completed",
            "training_eligible":false, "optimizer_updates":0, "context_quota_satisfied":false,
            "actual_game_owners":1, "attempted_contexts":collector.attempted_contexts(),
            "unique_contexts":collector.unique_contexts(), "action_members":report.action_members,
            "known_pairs":report.known_pairs, "partial_manifest":partial.manifest_path,
            "partial_manifest_sha256":partial.manifest_sha256, "root_read_bytes":used,
            "declaration_sha256":expected,
        }), partial))
    })();
    execute.map_err(|error| {
        let mut preservation_error = diagnostic_root.as_ref().and_then(|root| {
            preserve_failed_factory(owner.factory.as_ref(), root, MAX_PROBE_BYTES, until).err()
        });
        if let Some(collector) = owner.collector.as_mut()
            && let Err(archive_error) = collector.preserve_partial(&error.to_string(), until) {
            preservation_error = Some(match preservation_error {
                Some(raw_error) => invalid(format!("diagnostic raw preservation: {raw_error}; partial archive: {archive_error}")),
                None => archive_error,
            });
        }
        (error, preservation_error)
    })
}

#[cfg(target_os = "linux")]
fn run_linux(
    path: &Path,
    expected: &str,
    owner: &mut ActionCliOwnerV3,
    cancel: Arc<AtomicBool>,
) -> Result<ActionCliPublicationV3, (ArenaError, Option<ArenaError>)> {
    use sha2::Digest as _;
    use std::io::Read;
    let started = Instant::now();
    let mut preservation_until = started;
    let mut diagnostic_root = None;
    let mut diagnostic_remaining = MAX_PROBE_BYTES;
    let execute = (|| -> Result<ActionCliPublicationV3, ArenaError> {
        check(
            absolute_safe(path) && generation_sha(expected),
            "actual action declaration pin",
        )?;
        let parent = path
            .parent()
            .ok_or_else(|| invalid("action declaration parent"))?;
        let directory = super::selfplay_cli::open_directory(parent)?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| invalid("action declaration filename"))?;
        let file = super::selfplay_cli::open_regular(&directory, name)?;
        let length = file
            .metadata()
            .map_err(|_| invalid("action declaration metadata"))?
            .len();
        check(
            length > 0 && length <= MAX_DECLARATION_BYTES,
            "action declaration bounded bytes",
        )?;
        let mut bytes = Vec::with_capacity(length as usize);
        Read::take(&file, length + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| invalid("action declaration original read"))?;
        check(
            bytes.len() as u64 == length
                && format!("{:x}", sha2::Sha256::digest(&bytes)) == expected,
            "action declaration original SHA-256",
        )?;
        let declaration: ActionCliDeclarationV3 = serde_json::from_slice(&bytes)
            .map_err(|e| invalid(format!("action declaration: {e}")))?;
        declaration.validate()?;
        preservation_until = started
            .checked_add(Duration::from_millis(declaration.total_ms()?))
            .ok_or_else(|| invalid("action original whole window overflow"))?;
        let preflight_until = started + Duration::from_millis(declaration.preflight_ms);
        check_live(preflight_until, &cancel)?;
        let managed = super::selfplay_cli::open_directory(&declaration.managed_root)?;
        super::selfplay_cli::private_root(&managed)?;
        let staging = super::selfplay_cli::scan_staging(
            &managed,
            declaration.max_staging_bytes,
            preflight_until,
            &cancel,
        )?;
        check(
            owner.archives.len() == 128,
            "action corpus needs original 128-game owners",
        )?;
        for (ordinal, archive) in owner.archives.iter().enumerate() {
            check_live(preflight_until, &cancel)?;
            check(
                usize::from(archive.request().game_ordinal) == ordinal
                    && serde_json::to_vec(&archive.request().plan.openings)
                        .map_err(|error| invalid(error.to_string()))?
                        == serde_json::to_vec(&declaration.openings.openings)
                            .map_err(|error| invalid(error.to_string()))?
                    && archive.cwd_path().starts_with(&declaration.managed_root),
                "action saved roots changed original game assignment/split/staging owner",
            )?;
        }
        let cwd = std::env::current_dir().map_err(|_| invalid("action actual cwd readback"))?;
        check(
            cwd == declaration.private_cwd,
            "action CLI must start in declared actual private cwd",
        )?;
        let pinned = Arc::new(rz_uci::pals_adaptive_startup::PinnedPalsStartupV5::open(
            rz_uci::pals_adaptive_startup::PalsStartupOptionsV5 {
                config_path: declaration.startup_config.clone(),
                lock_path: declaration.startup_lock.clone(),
                config_sha256: declaration.startup_config_sha256.clone(),
                metadata_preflight: false,
            },
            &cwd,
        )?);
        rz_experiments::verify_compiled_source_components_v5(&pinned.config.source_components)?;
        check(
            pinned.config.source_components.get("arena")
                == Some(&declaration.producer.source_sha256)
                && pinned
                    .asset(rz_experiments::PalsStartupAssetPurposeV5::RuntimeLibrary)?
                    .declaration
                    .artifact
                    .sha256
                    == declaration.producer.runtime_sha256,
            "action actual compiled collector/runtime source pins",
        )?;
        check_live(preflight_until, &cancel)?;
        load_verified_frozen(
            &declaration,
            Arc::clone(&pinned),
            owner,
            Arc::clone(&cancel),
            preservation_until,
        )?;
        let registration = owner
            .frozen
            .lock()
            .map_err(|_| invalid("action frozen owner poisoned"))?
            .as_ref()
            .ok_or_else(|| invalid("actual frozen intake callback absent"))?
            .as_ref()
            .map_err(|failure| invalid(failure.error.to_string()))?
            .registration()
            .clone();
        check_live(preservation_until, &cancel)?;
        let staging_after_intake = super::selfplay_cli::scan_staging(
            &managed,
            declaration.max_staging_bytes,
            preservation_until,
            &cancel,
        )?;
        let extra = declaration
            .max_staging_bytes
            .checked_sub(staging_after_intake.unique_inode_bytes)
            .ok_or_else(|| invalid("action actual staging exhausted"))?;
        check(
            declaration
                .plan
                .maximum_output_bytes
                .checked_add(MAX_PROBE_BYTES)
                .is_some_and(|needed| needed <= extra),
            "action archive/probe reserve exceeds observed staging headroom",
        )?;
        owner.collector = Some(GenerationActionOutcomeCollectorV3::new(
            declaration.plan.clone(),
            declaration.openings.clone(),
            declaration.producer.clone(),
            declaration.context_epoch,
            declaration.output_directory.clone(),
            preservation_until,
        )?);
        diagnostic_root = Some(declaration.output_directory.clone());
        let mut read_credit = declaration.max_root_read_bytes;
        let mut probes_written = 0u64;
        let mut known_pairs = 0usize;
        let mut duplicate_contexts = 0usize;
        for (sequence, selected) in declaration.roots.iter().enumerate() {
            if owner
                .collector
                .as_ref()
                .expect("collector owned")
                .unique_contexts()
                == declaration.plan.desired_contexts as usize
            {
                break;
            }
            check_live(preservation_until, &cancel)?;
            let context_started = Instant::now();
            let context_until = context_started
                .checked_add(Duration::from_millis(declaration.context_ms))
                .ok_or_else(|| invalid("action original context window overflow"))?
                .min(preservation_until);
            let archive = &owner.archives[usize::from(selected.game_ordinal)];
            let offered = read_credit.min(MAX_ROOT_READ_BYTES);
            // On a failed read, do not refund an unobserved partial read.
            read_credit = read_credit
                .checked_sub(offered)
                .ok_or_else(|| invalid("action root read credit"))?;
            let (root, used) = archive.saved_action_root(
                selected.decision_index,
                context_until,
                Some(&cancel),
                offered,
            )?;
            read_credit = read_credit
                .checked_add(offered - used)
                .ok_or_else(|| invalid("action root read credit overflow"))?;
            let source_epoch = root.source_epoch();
            let issuer: GenerationActionProcessEpochIssuerV3 = Arc::new(move || {
                loop {
                    let epoch = reserve_actual_session_epochs_v3(1)?;
                    if epoch.0 != source_epoch {
                        return Ok(epoch);
                    }
                }
            });
            owner.factory = Some(
                ActualGenerationActionOutcomeFactoryV3::new(
                    Arc::clone(&pinned),
                    registration.clone(),
                    declaration.producer.clone(),
                    issuer,
                    GenerationActualActionLimitsV3 {
                        process_wall: Duration::from_millis(declaration.process_ms),
                        shutdown_reserve: Duration::from_millis(declaration.shutdown_reserve_ms),
                    },
                )?
                .with_cancellation(Arc::clone(&cancel)),
            );
            let report = owner
                .collector
                .as_mut()
                .expect("collector owned")
                .run_context(
                    owner.factory.as_mut().expect("factory owned"),
                    &root,
                    context_until,
                )?;
            let probe = owner
                .factory
                .as_mut()
                .expect("factory owned")
                .take_probe_receipt()
                .ok_or_else(|| invalid("actual action source probe receipt absent"))?;
            // This raw source probe is auxiliary, never enumerated utility.
            let raw = probe_json(&probe, Some(sequence), Some(selected))?;
            let output = serde_json::to_vec(&raw).map_err(|e| invalid(e.to_string()))?;
            probes_written = probes_written
                .checked_add(output.len() as u64)
                .filter(|total| *total <= MAX_PROBE_BYTES)
                .ok_or_else(|| invalid("action auxiliary probe output quota"))?;
            diagnostic_remaining = diagnostic_remaining
                .checked_sub(output.len() as u64)
                .ok_or_else(|| invalid("action original diagnostic reserve exhausted"))?;
            publish_new(
                &declaration.output_directory,
                &format!("source-probe-{sequence:04}.json"),
                &output,
            )?;
            known_pairs = known_pairs
                .checked_add(report.known_pairs)
                .ok_or_else(|| invalid("action known pair count overflow"))?;
            duplicate_contexts += usize::from(report.duplicate_context);
            owner.factory = None;
            super::selfplay_cli::scan_staging(
                &managed,
                declaration.max_staging_bytes,
                preservation_until,
                &cancel,
            )?;
        }
        let collector = owner.collector.as_mut().expect("collector owned");
        let attempted = collector.attempted_contexts();
        let unique = collector.unique_contexts();
        let (directory, manifest) = collector.finish(preservation_until)?;
        let (archive_root, files, publication_read_bytes) = pin_publication(
            &directory,
            &manifest,
            preservation_until,
            &cancel,
            declaration.max_publication_read_bytes,
        )?;
        let final_staging = super::selfplay_cli::scan_staging(
            &managed,
            declaration.max_staging_bytes,
            preservation_until,
            &cancel,
        )?;
        let status = serde_json::json!({
            "schema":"rz-pals-action-outcomes-cli-result/1", "status":"completed_actual_action_contexts",
            "attempted_contexts":attempted, "unique_contexts":unique, "known_pairs":known_pairs,
            "duplicate_contexts":duplicate_contexts, "archive_directory":directory, "manifest":manifest,
            "declaration_sha256":expected, "source_game_owners":owner.archives.len(),
            "source_components":pinned.config.source_components,
            "auxiliary_probe_bytes":probes_written, "root_read_bytes":declaration.max_root_read_bytes-read_credit,
            "publication_read_bytes":publication_read_bytes,
            "staging_before":staging, "staging_after":final_staging,
            "learning_executed":false, "training_target_sufficiency":"not_assessed",
            "assurance_scope":"actual_named_frozen_PC_intake_original_saved_roots_isolated_actions_physical_close_chunk_publication; no_optimizer_or_generation_training"
        });
        Ok(ActionCliPublicationV3 {
            status,
            directory,
            manifest,
            plan: declaration.plan,
            root: archive_root,
            files,
        })
    })();
    execute.map_err(|error| {
        let until = preservation_until.min(Instant::now() + Duration::from_secs(30));
        let mut preservation_error = diagnostic_root.as_ref().and_then(|root| {
            preserve_failed_factory(owner.factory.as_ref(), root, diagnostic_remaining, until).err()
        });
        if let Some(collector) = owner.collector.as_mut()
            && let Err(archive_error) = collector.preserve_partial(&error.to_string(), until)
        {
            preservation_error = Some(match preservation_error {
                Some(raw_error) => invalid(format!("original action raw preservation: {raw_error}; partial archive: {archive_error}")),
                None => archive_error,
            });
        }
        (error, preservation_error)
    })
}

#[cfg(target_os = "linux")]
fn preserve_failed_factory(
    factory: Option<&ActualGenerationActionOutcomeFactoryV3>,
    root: &Path,
    mut remaining: u64,
    until: Instant,
) -> Result<(), ArenaError> {
    let Some(factory) = factory else {
        return Ok(());
    };
    let mut write = |name: &str, value: &serde_json::Value| -> Result<(), ArenaError> {
        check(
            Instant::now() < until,
            "action diagnostic preservation original bound",
        )?;
        let bytes = serde_json::to_vec(value).map_err(|error| invalid(error.to_string()))?;
        remaining = remaining
            .checked_sub(bytes.len() as u64)
            .ok_or_else(|| invalid("action diagnostic preservation remaining reserved bytes"))?;
        publish_new(root, name, &bytes)
    };
    if let Some(probe) = factory.probe_receipt() {
        write("failed-source-probe.json", &probe_json(probe, None, None)?)?;
    }
    for process in 0..8 {
        let Ok(capture) = factory.retained_capture(process) else {
            break;
        };
        write(
            &format!("failed-action-process-{process}.json"),
            &raw_capture_json(&capture),
        )?;
    }
    for (index, failure) in factory.start_failures().iter().enumerate() {
        let capture = factory.retained_start_failure_capture(index)?;
        write(
            &format!("failed-action-start-{index}.json"),
            &serde_json::json!({
                "schema":"rz-pals-action-original-start-failure/1", "epoch":failure.epoch.0,
                "stage":failure.stage,"detail":failure.detail,
                "observed_wall_ns":failure.observed_wall_cost.as_nanos().to_string(),
                "physical_close":"unknown", "capture":raw_capture_json(&capture),
                "target_authority":false,
            }),
        )?;
    }
    Ok(())
}
#[cfg(target_os = "linux")]
fn raw_capture_json(capture: &GenerationActualActionRawCaptureV3) -> serde_json::Value {
    serde_json::json!({
        "schema":"rz-pals-action-original-retained-raw-capture/1", "records":capture.records,
        "evidence":capture.evidence,"semantic_outcomes":capture.semantic_outcomes,
        "capture_error":capture.capture_error,"target_authority":false,
    })
}

#[cfg(target_os = "linux")]
fn load_verified_frozen(
    declaration: &ActionCliDeclarationV3,
    pinned: Arc<rz_uci::pals_adaptive_startup::PinnedPalsStartupV5>,
    owner: &mut ActionCliOwnerV3,
    cancel: Arc<AtomicBool>,
    whole_until: Instant,
) -> Result<(), ArenaError> {
    use rz_eval::error::{BackendError, FailureKind, FailureStage};
    use rz_uci::pals_adaptive_bootstrap::{
        AdaptiveCollectionHooksV5, prepare_collection_actual_with_cancellation,
    };
    use rz_uci::search_driver::SearchSessionDriver;
    let intake = &declaration.intake;
    let started = Instant::now();
    let execution = (started + Duration::from_millis(intake.execution_ms)).min(whole_until);
    let whole = (execution + Duration::from_millis(intake.cleanup_ms)).min(whole_until);
    check_live(execution, &cancel)?;
    let interpreter_root = super::selfplay_cli::open_directory(&intake.interpreter_root)?;
    let request_root = super::selfplay_cli::open_directory(&intake.request_root)?;
    let launch = FrozenPcVerifierLaunchV3 {
        interpreter: intake
            .interpreter
            .open_verified_at(&interpreter_root, 64 * 1024 * 1024)?,
        interpreter_sha256: intake.interpreter.sha256.clone(),
        interpreter_bytes: intake.interpreter.bytes,
        request: intake.request.open_verified_at(&request_root, 128 * 1024)?,
        request_sha256: intake.request.sha256.clone(),
        request_bytes: intake.request.bytes,
        registered_source_sha256: declaration.producer.source_sha256.clone(),
        dependency_directories: intake.dependency_directories.clone(),
        artifact_directory: intake.artifact_directory.clone(),
        scratch_directory: intake.scratch_directory.clone(),
        started,
        execution_deadline: execution,
        whole_deadline: whole,
        limits: ProcessLimits {
            wall_ms: intake.execution_ms,
            shutdown_grace_ms: intake.cleanup_ms,
            max_output_bytes: 64 * 1024,
            max_child_processes: 1,
        },
    };
    let result_owner = Arc::clone(&owner.frozen);
    let plan = declaration.plan.clone();
    let teacher_binary = declaration.producer.teacher_binary_sha256.clone();
    let teacher_profile = declaration.producer.teacher_profile_sha256.clone();
    let frozen_cancel = Arc::clone(&cancel);
    let native_intake = Box::new(
        move |backend: &rz_eval::pals_onnx::virtual_review::PalsVirtualReviewOnnxBackend,
              native_until: Instant| {
            let mut launch = launch;
            // Native startup may impose a narrower bound. Tighten the already
            // registered original window; never restart its clock inside the hook.
            launch.execution_deadline = launch.execution_deadline.min(native_until);
            launch.whole_deadline = launch
                .whole_deadline
                .min(native_until + Duration::from_millis(launch.limits.shutdown_grace_ms));
            let outcome = run_frozen_pc_intake_v3(
                launch,
                &plan,
                backend,
                teacher_binary,
                teacher_profile,
                Some(&frozen_cancel),
            );
            let error = outcome
                .as_ref()
                .err()
                .map(|failure| failure.error.to_string());
            let mut result = result_owner.lock().map_err(|_| {
                BackendError::new(
                    FailureKind::BackendFailure,
                    FailureStage::Admission,
                    "frozen intake result owner poisoned",
                )
            })?;
            if result.is_some() {
                return Err(BackendError::new(
                    FailureKind::IdentityMismatch,
                    FailureStage::Admission,
                    "duplicate original frozen intake",
                ));
            }
            *result = Some(outcome);
            match error {
                Some(error) => Err(BackendError::new(
                    FailureKind::InvalidInput,
                    FailureStage::Admission,
                    "actual frozen named parameter verification failed",
                )
                .with_diagnostic("FrozenPcIntake", &error)),
                None => Ok(()),
            }
        },
    );
    let (mut capture, [native, unused_native], [dispatch, unused_dispatch]) =
        GenerationGameCaptureV3::new(
            format!("{}-frozen-intake", declaration.plan.run_id),
            declaration.openings.openings[0].id.clone(),
            declaration.plan.frozen_pc_epoch,
            declaration.context_epoch,
            Arc::new(Mutex::new(GenerationChronologyV3::default())),
        )?;
    drop((unused_native, unused_dispatch));
    capture.register_teacher_producer(&declaration.producer)?;
    let [teacher, unused_teacher] = capture.teacher_observers();
    drop(unused_teacher);
    let admitted_teacher = teacher.clone();
    let hooks = AdaptiveCollectionHooksV5 {
        native_observer: Box::new(native),
        dispatch_observer: dispatch,
        teacher_observer: teacher.observer(),
        teacher_admission: Arc::new(move |admitted| {
            admitted_teacher.register_source(admitted).map_err(|error| {
                rz_uci::pals_adaptive_bootstrap::AdaptiveBootstrapError {
                    stage: "FrozenIntakeTeacherSource",
                    detail: error.to_string(),
                }
            })
        }),
        root_admission: Arc::new(|_| Ok(())),
        root_closure: Arc::new(|_, _| Ok(())),
        native_intake: Some(native_intake),
    };
    owner.intake_process = Some(
        prepare_collection_actual_with_cancellation(
            pinned,
            reserve_actual_session_epochs_v3(1)?,
            hooks,
            execution,
            Arc::clone(&cancel),
        )
        .map_err(|error| invalid(error.to_string()))?,
    );
    let prepared = owner
        .intake_process
        .as_ref()
        .expect("intake original process owner");
    // Rules-terminal root opens the real native readiness fence but requests
    // no forward. It is not collected as an action/root/witness/WDL sample.
    let terminal = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        .map_err(|error| invalid(error.to_string()))?;
    capture.activate_root(&terminal)?;
    let mut context = prepared
        .context(execution, whole)
        .map_err(|error| invalid(error.to_string()))?;
    context.control.cancellation = Arc::clone(&cancel);
    let readiness = prepared.driver().run(&terminal, &context, &mut |_| {});
    let retired = if readiness.is_ok() {
        prepared
            .retire_context(&context)
            .map_err(|error| invalid(error.to_string()))
    } else {
        Ok(())
    };
    let closed = prepared
        .finish_process(whole)
        .map_err(|error| invalid(error.to_string()));
    readiness.map_err(|error| invalid(format!("frozen intake native readiness: {error}")))?;
    retired?;
    let close = closed?;
    check(
        close.permits_process_close(),
        "frozen intake native process physical close unknown",
    )?;
    // Read-only metadata stays auxiliary. Never assert a neural forward here.
    publish_new(
        &declaration.intake.scratch_directory,
        "native-intake-process-close.json",
        &serde_json::to_vec(&serde_json::json!({
            "schema":"rz-pals-frozen-intake-process-close/1", "native_forward_executed":false,
            "process":process_json(&close),
        }))
        .map_err(|error| invalid(error.to_string()))?,
    )?;
    let receipt = owner
        .frozen
        .lock()
        .map_err(|_| invalid("original frozen owner poisoned"))?
        .as_ref()
        .ok_or_else(|| invalid("original frozen intake absent"))?
        .as_ref()
        .map_err(|failure| invalid(failure.error.to_string()))?
        .receipt()
        .clone();
    publish_new(
        &declaration.intake.scratch_directory,
        "native-frozen-pc-intake-receipt.json",
        &serde_json::to_vec(&receipt).map_err(|error| invalid(error.to_string()))?,
    )?;
    owner.intake_process = None;
    Ok(())
}
/// Pin only the files returned by the actual completed writer. Caller JSON
/// never reaches this private successful publisher. Reads use an independent
/// original credit and preserve file positions for subsequent consumers.
#[cfg(target_os = "linux")]
fn pin_publication(
    directory: &Path,
    manifest: &GenerationChunkManifestV3,
    until: Instant,
    cancel: &AtomicBool,
    max_read_bytes: u64,
) -> Result<(cap_std::fs::Dir, Vec<ActionArchiveFileV3>, u64), ArenaError> {
    use sha2::{Digest, Sha256};
    manifest.validate()?;
    check(
        max_read_bytes > 0 && max_read_bytes <= 2 * GENERATION_OUTPUT_CAP,
        "action original publication read bound",
    )?;
    check_live(until, cancel)?;
    let root = super::selfplay_cli::open_directory(directory)?;
    super::selfplay_cli::private_root(&root)?;
    let mut value = serde_json::to_value(manifest).map_err(|error| invalid(error.to_string()))?;
    value.sort_all_objects();
    let mut bytes = serde_json::to_vec(&value).map_err(|error| invalid(error.to_string()))?;
    bytes.push(b'\n');
    check(
        bytes.len() <= 16 * 1024 * 1024,
        "action manifest byte bound",
    )?;
    let manifest_pin = ArtifactRef {
        path: "manifest.json".into(),
        bytes: bytes.len() as u64,
        sha256: format!("{:x}", Sha256::digest(&bytes)),
        source: "rz-pals-actual-action-collector/3".into(),
        license: "unspecified-generated-data".into(),
    };
    // No need to retain canonical serialization after deriving the expected
    // hash. The writer already produced the exact original bytes.
    drop(bytes);
    let mut files = Vec::with_capacity(manifest.chunks.len() + 1);
    let mut credit = max_read_bytes;
    for pin in
        std::iter::once(manifest_pin).chain(manifest.chunks.iter().map(|chunk| ArtifactRef {
            path: chunk.path.clone(),
            bytes: chunk.bytes,
            sha256: chunk.sha256.clone(),
            source: "rz-pals-actual-action-collector/3".into(),
            license: "unspecified-generated-data".into(),
        }))
    {
        pin.validate()?;
        let file = super::selfplay_cli::open_regular(&root, &pin.path)?;
        let held = ActionArchiveFileV3 { pin, file };
        verify_archive_file(&root, &held, until, cancel, &mut credit)?;
        files.push(held);
    }
    let named = super::selfplay_cli::open_directory(directory)?;
    same_directory(&root, &named)?;
    Ok((root, files, max_read_bytes - credit))
}

#[cfg(target_os = "linux")]
fn same_directory(held: &cap_std::fs::Dir, named: &cap_std::fs::Dir) -> Result<(), ArenaError> {
    use std::os::unix::fs::MetadataExt;
    let metadata = |root: &cap_std::fs::Dir| {
        root.try_clone()
            .and_then(|clone| clone.into_std_file().metadata())
            .map_err(|_| invalid("action original directory metadata"))
    };
    let old = metadata(held)?;
    let current = metadata(named)?;
    check(
        old.is_dir()
            && current.is_dir()
            && old.dev() == current.dev()
            && old.ino() == current.ino(),
        "action original directory changed",
    )
}

#[cfg(target_os = "linux")]
fn verify_archive_file(
    root: &cap_std::fs::Dir,
    held: &ActionArchiveFileV3,
    until: Instant,
    cancel: &AtomicBool,
    credit: &mut u64,
) -> Result<(), ArenaError> {
    use sha2::{Digest, Sha256};
    use std::os::unix::fs::{FileExt, MetadataExt};
    check_live(until, cancel)?;
    let named = super::selfplay_cli::open_regular(root, &held.pin.path)?;
    let before = held
        .file
        .metadata()
        .map_err(|_| invalid("action held artifact metadata"))?;
    let current = named
        .metadata()
        .map_err(|_| invalid("action named artifact metadata"))?;
    let identical = |metadata: &std::fs::Metadata| {
        metadata.is_file()
            && metadata.nlink() == 1
            && metadata.dev() == before.dev()
            && metadata.ino() == before.ino()
            && metadata.len() == held.pin.bytes
    };
    check(
        identical(&before) && identical(&current),
        "action original artifact inode/type/size changed",
    )?;
    let mut hash = Sha256::new();
    let mut offset = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    while offset < held.pin.bytes {
        check_live(until, cancel)?;
        let requested = (held.pin.bytes - offset).min(buffer.len() as u64);
        check(
            *credit >= requested,
            "action original artifact read credit exhausted",
        )?;
        // Reserve before I/O. A failed/short read does not become free work.
        *credit -= requested;
        let received = held
            .file
            .read_at(&mut buffer[..requested as usize], offset)
            .map_err(|_| invalid("action original artifact read failed"))?;
        check(received > 0, "action original artifact truncated")?;
        hash.update(&buffer[..received]);
        offset += received as u64;
    }
    let after = held
        .file
        .metadata()
        .map_err(|_| invalid("action artifact post-read metadata"))?;
    let named_after = super::selfplay_cli::open_regular(root, &held.pin.path)?;
    let named_after = named_after
        .metadata()
        .map_err(|_| invalid("action named artifact post-read metadata"))?;
    check(
        identical(&after)
            && identical(&named_after)
            && before.mtime() == after.mtime()
            && before.mtime_nsec() == after.mtime_nsec()
            && before.ctime() == after.ctime()
            && before.ctime_nsec() == after.ctime_nsec()
            && format!("{:x}", hash.finalize()) == held.pin.sha256,
        "action original artifact changed during readback",
    )
}
#[cfg(target_os = "linux")]
fn check_live(until: Instant, cancel: &AtomicBool) -> Result<(), ArenaError> {
    check(
        Instant::now() < until && !cancel.load(Ordering::Acquire),
        "actual action canceled/expired",
    )
}
#[cfg(target_os = "linux")]
fn publish_new(root: &Path, name: &str, bytes: &[u8]) -> Result<(), ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    use std::io::Write;
    let root = super::selfplay_cli::open_directory(root)?;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    let mut file = root
        .open_with(name, &options)
        .map_err(|_| invalid("action original exclusive publication"))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|_| invalid("action original durable publication"))
}
#[cfg(target_os = "linux")]
fn probe_json(
    probe: &GenerationActualActionProbeReceiptV3,
    sequence: Option<usize>,
    selected: Option<&ActionCliSavedRootV3>,
) -> Result<serde_json::Value, ArenaError> {
    let capture = probe.capture();
    Ok(serde_json::json!({
        "schema":"rz-pals-action-source-probe-raw/1", "sequence":sequence,
        "source_root":selected, "frozen_context_sha256":probe.registration().frozen_context_sha256(),
        "wall_cost_ns":probe.wall_cost().as_nanos().to_string(), "root":root_json(probe.root_receipt()),
        "process":process_json(probe.process_receipt()), "completion":format!("{:?}", probe.completion()),
        "records":capture.records, "evidence":capture.evidence, "semantic_outcomes":capture.semantic_outcomes,
        "capture_error":capture.capture_error, "utility_membership":false,
    }))
}
#[cfg(target_os = "linux")]
fn root_json(root: &rz_uci::pals_adaptive::AdaptiveShutdownReceipt) -> serde_json::Value {
    serde_json::json!({
        "authority": { "epoch":root.authority.epoch.0,"game":root.authority.game.0,"root":root.authority.root.0 },
        "native_owner_joined":root.native_owner_joined,"native_owner_quiesced":root.native_owner_quiesced,
        "native_physical_shutdown_confirmed":root.native_physical_shutdown_confirmed,
        "own_cpu_joined":root.own_cpu_joined,"expected_own_cpu_owners":root.expected_own_cpu_owners,
        "own_cpu_panicked":root.own_cpu_panicked,"external_shutdown_confirmed":root.external_shutdown_confirmed,
        "external_owner_quiesced":root.external_owner_quiesced,"auxiliary_shutdown_confirmed":root.auxiliary_shutdown_confirmed,
        "native_cpu_spent":root.native_cpu_spent,"native_cpu_reserved":root.native_cpu_reserved,
        "pending_tickets":root.pending_tickets,"quarantined":root.quarantined,
        "request_sequence_high_water":root.request_sequence_high_water,
        "assurance_scope":"raw_original_root_shutdown_observation_not_reconstructible_authority"
    })
}
#[cfg(target_os = "linux")]
fn process_json(
    process: &rz_uci::pals_adaptive::AdaptiveSessionProcessReceipt,
) -> serde_json::Value {
    serde_json::json!({
        "epoch":process.epoch().0,"native_owner_joined":process.native_owner_joined(),
        "native_physical_shutdown_confirmed":process.native_physical_shutdown_confirmed(),
        "store_archive_closed":process.store_archive_closed(),
        "external_shutdown":format!("{:?}",process.external_shutdown()),
        "latest_root":process.latest_root().map(root_json),
        "assurance_scope":"raw_original_process_close_observation_not_reconstructible_authority"
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saved_root_schedule_rejects_duplicates_and_out_of_range_before_execution() {
        let one = ActionCliSavedRootV3 {
            game_ordinal: 0,
            decision_index: 0,
        };
        assert!(validate_root_schedule(std::slice::from_ref(&one), 8192).is_ok());
        assert!(validate_root_schedule(&[one.clone(), one], 8192).is_err());
        assert!(
            validate_root_schedule(
                &[ActionCliSavedRootV3 {
                    game_ordinal: 128,
                    decision_index: 0
                }],
                8192
            )
            .is_err()
        );
        assert!(
            validate_root_schedule(
                &[ActionCliSavedRootV3 {
                    game_ordinal: 0,
                    decision_index: 512
                }],
                8192
            )
            .is_err()
        );
        assert!(validate_root_schedule(&[], 8192).is_err());
    }
    #[test]
    #[cfg(target_os = "linux")]
    fn original_action_paths_reject_secret_and_traversal_names() {
        assert!(absolute_safe(Path::new(
            "/owned/run/action-declaration.json"
        )));
        for path in [
            "/owned/.env",
            "/owned/key.pem",
            "/owned/service_account.json",
            "/owned/../outside",
        ] {
            assert!(!absolute_safe(Path::new(path)));
        }
    }
    #[test]
    #[cfg(target_os = "linux")]
    fn original_readback_keeps_cursor_and_rejects_same_content_replacement() {
        use sha2::{Digest, Sha256};
        use std::io::{Read, Seek, SeekFrom};
        use std::os::unix::fs::PermissionsExt;
        let root_path = std::env::temp_dir().join(format!(
            "rz-action-original-readback-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root_path).unwrap();
        std::fs::set_permissions(&root_path, std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = root_path.join("manifest.json");
        std::fs::write(&path, b"abc").unwrap();
        let root = super::super::selfplay_cli::open_directory(&root_path).unwrap();
        let mut file = super::super::selfplay_cli::open_regular(&root, "manifest.json").unwrap();
        file.seek(SeekFrom::Start(1)).unwrap();
        let held = ActionArchiveFileV3 {
            pin: ArtifactRef {
                path: "manifest.json".into(),
                bytes: 3,
                sha256: format!("{:x}", Sha256::digest(b"abc")),
                source: "cpu-readback-fixture".into(),
                license: "MIT".into(),
            },
            file,
        };
        let cancel = AtomicBool::new(false);
        let mut credit = 3;
        verify_archive_file(
            &root,
            &held,
            Instant::now() + Duration::from_secs(5),
            &cancel,
            &mut credit,
        )
        .unwrap();
        assert_eq!(credit, 0);
        let mut original = &held.file;
        let mut remaining = String::new();
        original.read_to_string(&mut remaining).unwrap();
        assert_eq!(remaining, "bc");
        let old_path = root_path.join("held-original.json");
        std::fs::rename(&path, &old_path).unwrap();
        std::fs::write(&path, b"abc").unwrap();
        assert!(
            verify_archive_file(
                &root,
                &held,
                Instant::now() + Duration::from_secs(5),
                &cancel,
                &mut 3
            )
            .is_err()
        );
        drop((held, root));
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(old_path).unwrap();
        std::fs::remove_dir(root_path).unwrap();
    }
}
