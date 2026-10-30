//! Explicit, sequential actual-generation admission. The declaration describes
//! prepared private directories; it creates neither mounts nor model assets.
//! JSON is never promoted into a native or original-process capability.
use super::*;
#[cfg(target_os = "linux")]
use crate::process::{OwnedArtifactTreeWatch, ProcessLimits};
use rz_experiments::ArtifactRef;
use serde::{Deserialize, Serialize};
#[cfg(target_os = "linux")]
use sha2::Digest as _;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::AtomicBool;
#[cfg(target_os = "linux")]
use std::sync::atomic::Ordering;
#[cfg(target_os = "linux")]
use std::time::{Duration, Instant};

pub const SELFPLAY_CLI_DECLARATION_DOMAIN_V3: &str = "rz-pals-selfplay-cli-declaration/1";
#[cfg(target_os = "linux")]
pub(super) const MAX_DECLARATION_BYTES: u64 = 8 * 1024 * 1024;
pub(super) const MAX_REQUEST_BYTES: u64 = 2 * 1024 * 1024;
pub(super) const MAX_BINARY_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelfplayCliProcessLimitsV3 {
    pub execution_ms: u64,
    pub cleanup_ms: u64,
    pub max_output_bytes: u64,
    pub max_child_processes: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelfplayCliTreeLimitsV3 {
    pub max_total_bytes: u64,
    pub max_file_bytes: u64,
    pub max_files: usize,
    pub max_depth: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelfplayCliRawLimitsV3 {
    pub max_records: usize,
    pub max_evidence: usize,
    pub max_retained_bytes: u64,
    pub max_read_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelfplayCliGameDeclarationV3 {
    pub game_ordinal: u16,
    /// One pre-existing direct child of the exclusively owned run root.
    pub private_directory: String,
    /// Canonical original GenerationWholeGameRequestV3 bytes in that directory.
    pub request: ArtifactRef,
    pub process: SelfplayCliProcessLimitsV3,
    pub tree: SelfplayCliTreeLimitsV3,
    pub raw: SelfplayCliRawLimitsV3,
}

/// Versioned execution declaration, not an execution or training receipt.
/// All 128 windows are declared before the first child. Each is fixed when its
/// original admission starts, before model/asset hashing or startup work.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelfplayCliDeclarationV3 {
    pub schema: String,
    pub plan: SelfplayCollectionPlanV3,
    pub run_root: PathBuf,
    pub binary_root: PathBuf,
    pub binary: ArtifactRef,
    pub preflight_ms: u64,
    pub publication_ms: u64,
    pub max_staging_bytes: u64,
    pub max_index_bytes: u64,
    pub max_archive_read_bytes: u64,
    pub games: Vec<SelfplayCliGameDeclarationV3>,
}
impl SelfplayCliDeclarationV3 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        self.plan.validate()?;
        self.binary.validate()?;
        check(
            self.schema == SELFPLAY_CLI_DECLARATION_DOMAIN_V3
                && absolute_clean(&self.run_root)
                && absolute_clean(&self.binary_root)
                && self.binary.bytes <= MAX_BINARY_BYTES
                && self.binary.path.split('/').all(nonsecret_name)
                && (1..=300_000).contains(&self.preflight_ms)
                // Publication hashes shared inputs again under an explicit
                // read-only budget; it is not a child's cleanup window.
                && (1..=1_800_000).contains(&self.publication_ms)
                && self.max_staging_bytes > 0
                && self.max_staging_bytes <= GENERATION_OUTPUT_CAP
                && self.max_index_bytes > 0
                && self.max_index_bytes <= 16 * 1024 * 1024
                && self.max_archive_read_bytes > 0
                && self.max_archive_read_bytes <= 128 * 4 * GENERATION_OUTPUT_CAP
                && self.games.len() == 128,
            "selfplay CLI explicit root, bounded windows, staging and 128 declarations",
        )?;
        let mut directories = BTreeSet::new();
        for (ordinal, game) in self.games.iter().enumerate() {
            check(directories.insert(&game.private_directory), "selfplay duplicate game directory")?;
            validate_game_declaration_v3(game, ordinal as u16, self.max_staging_bytes)?;
        }
        self.registered_total_ms()?;
        self.staging_data_ceiling()?;
        Ok(())
    }
    fn staging_data_ceiling(&self) -> Result<u64, ArenaError> {
        let diagnostic_bytes = self
            .games
            .iter()
            .map(|game| game.process.max_output_bytes)
            .max()
            .unwrap_or(0);
        self.max_index_bytes
            .checked_add(diagnostic_bytes)
            .and_then(|reserve| self.max_staging_bytes.checked_sub(reserve))
            .filter(|ceiling| *ceiling > 0)
            .ok_or_else(|| {
                invalid("selfplay staging must reserve final/partial index and failed pipe bytes")
            })
    }
    fn registered_total_ms(&self) -> Result<u64, ArenaError> {
        self.games.iter().try_fold(
            self.preflight_ms
                .checked_add(self.publication_ms)
                .ok_or_else(|| invalid("selfplay CLI total window overflow"))?,
            |sum, game| {
                sum.checked_add(game.process.execution_ms)
                    .and_then(|value| value.checked_add(game.process.cleanup_ms))
                    .ok_or_else(|| invalid("selfplay CLI total window overflow"))
            },
        )
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SelfplayCliStagingObservationV3 {
    /// File lengths per unique (device, inode), rather than allocated blocks.
    pub unique_inode_bytes: u64,
    pub logical_alias_bytes: u64,
    pub unique_regular_files: usize,
    pub readonly_shared_aliases: usize,
    pub observed_max_file_bytes: u64,
    pub observed_directories: usize,
    pub observed_max_depth: usize,
    pub enforcement: &'static str,
}

/// A failed invocation retains its actual archive/process owner for explicit
/// recovery. A partial publication is diagnostic; it cannot grant completion.
pub struct SelfplayCliFailureV3 {
    pub error: ArenaError,
    pub partial_publication_error: Option<ArenaError>,
    owner: Option<Box<GenerationSeriesCollectorV3>>,
    rejected_process_owner: Option<Box<GenerationGameProcessFailureV3>>,
}
impl SelfplayCliFailureV3 {
    pub fn collector(&self) -> Option<&GenerationSeriesCollectorV3> {
        self.owner.as_deref()
    }
    pub fn rejected_process_owner(&self) -> Option<&GenerationGameProcessFailureV3> {
        self.rejected_process_owner.as_deref()
    }
}
impl From<ArenaError> for SelfplayCliFailureV3 {
    fn from(error: ArenaError) -> Self {
        Self {
            error,
            partial_publication_error: None,
            owner: None,
            rejected_process_owner: None,
        }
    }
}
impl From<rz_experiments::ManifestError> for SelfplayCliFailureV3 {
    fn from(error: rz_experiments::ManifestError) -> Self {
        ArenaError::from(error).into()
    }
}
impl std::fmt::Debug for SelfplayCliFailureV3 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SelfplayCliFailureV3")
            .field("error", &self.error)
            .field("partial_publication_error", &self.partial_publication_error)
            .field("original_owner_retained", &self.owner.is_some())
            .field(
                "rejected_process_owner_retained",
                &self.rejected_process_owner.is_some(),
            )
            .finish()
    }
}
impl std::fmt::Display for SelfplayCliFailureV3 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}; original_owner_retained={}; partial_publication_failed={}",
            self.error,
            self.owner.is_some(),
            self.partial_publication_error.is_some()
        )
    }
}
impl std::error::Error for SelfplayCliFailureV3 {}

/// The caller owns cancellation. This function never creates an optimizer or
/// training process, changes a mount, removes raw artifacts, or retries a game.
pub fn run_selfplay_collect_cli_v3(
    declaration_path: &Path,
    declaration_sha256: &str,
    cancel: &AtomicBool,
) -> Result<serde_json::Value, SelfplayCliFailureV3> {
    run_selfplay_collect_owned_cli_v3(declaration_path, declaration_sha256, cancel)
        .map(|result| result.status)
}

/// Same original execution as the status CLI, with its actual archive owner
/// retained for a later source-owned action/trainer stage in the same process.
/// Reading saved JSON cannot construct this result or dispatch later stages.
pub struct SelfplayCliVerifiedResultV3 {
    status: serde_json::Value,
    archive: GenerationVerifiedSeriesArchiveV3,
}
impl SelfplayCliVerifiedResultV3 {
    pub fn status(&self) -> &serde_json::Value {
        &self.status
    }
    pub fn archive(&self) -> &GenerationVerifiedSeriesArchiveV3 {
        &self.archive
    }
    pub fn into_archive(self) -> GenerationVerifiedSeriesArchiveV3 {
        self.archive
    }
}

pub fn run_selfplay_collect_owned_cli_v3(
    declaration_path: &Path,
    declaration_sha256: &str,
    cancel: &AtomicBool,
) -> Result<SelfplayCliVerifiedResultV3, SelfplayCliFailureV3> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (declaration_path, declaration_sha256, cancel);
        Err(invalid("selfplay CLI original process transport requires Linux").into())
    }
    #[cfg(target_os = "linux")]
    {
        run_linux(declaration_path, declaration_sha256, cancel)
    }
}

/// Shared bounds for an original game declaration. This does not execute or
/// confer authority on the declaration.
pub(super) fn validate_game_declaration_v3(
    game: &SelfplayCliGameDeclarationV3, ordinal: u16, max_staging_bytes: u64,
) -> Result<(), ArenaError> {
    game.request.validate()?;
    check(
        game.game_ordinal == ordinal
            && direct_name(&game.private_directory)
            && game.request.bytes <= MAX_REQUEST_BYTES
            && game.request.path.split('/').all(nonsecret_name)
            && (1..=3_600_000).contains(&game.process.execution_ms)
            && (1..=30_000).contains(&game.process.cleanup_ms)
            && game.process.max_output_bytes > 0
            && game.process.max_output_bytes <= 16 * 1024 * 1024
            && (1..=64).contains(&game.process.max_child_processes)
            && game.tree.max_total_bytes > 0
            && game.tree.max_total_bytes <= max_staging_bytes
            && game.tree.max_file_bytes > 0
            && game.tree.max_file_bytes <= game.tree.max_total_bytes
            && (1..=4096).contains(&game.tree.max_files)
            && game.tree.max_depth <= 32
            && game.raw.max_records > 0
            && game.raw.max_records <= GENERATION_MAX_RECORDS
            && game.raw.max_evidence > 0
            && game.raw.max_evidence <= 1_048_576
            && game.raw.max_retained_bytes > 0
            && game.raw.max_retained_bytes <= 512 * 1024 * 1024
            && game.raw.max_read_bytes > 0
            && game.raw.max_read_bytes <= 4 * GENERATION_OUTPUT_CAP,
        "selfplay CLI ordered, fresh, bounded game declaration",
    )?;
    Ok(())
}

pub(super) fn direct_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}
pub(super) fn nonsecret_name(name: &str) -> bool {
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
pub(super) fn absolute_clean(path: &Path) -> bool {
    path.is_absolute()
        && path.components().count() <= 64
        && path
            .components()
            .all(|c| matches!(c, Component::RootDir | Component::Normal(_)))
}

#[cfg(target_os = "linux")]
fn run_linux(
    declaration_path: &Path,
    declaration_sha256: &str,
    cancel: &AtomicBool,
) -> Result<SelfplayCliVerifiedResultV3, SelfplayCliFailureV3> {
    let PreparedSelfplayInputsV3 { declaration, run_root, binary, binary_path,
        mut staging, staging_data_ceiling, publication_until } =
        prepare_selfplay_inputs_v3(declaration_path, declaration_sha256, cancel)?;
    let mut collector = GenerationSeriesCollectorV3::new(
        declaration.plan.clone(),
        declaration.run_root.clone(),
        run_root
            .try_clone()
            .map_err(|_| invalid("selfplay root clone"))?
            .into_std_file(),
        GenerationSeriesLimitsV3 {
            max_output_bytes: declaration.max_staging_bytes,
            max_index_bytes: declaration.max_index_bytes,
            max_read_bytes: declaration.max_archive_read_bytes,
        },
        publication_until,
    )?;
    let mut series = GenerationGameProcessSeriesV3::default();
    let mut rejected_process_owner = None;
    let execution_result = (|| -> Result<(), ArenaError> {
        for game in &declaration.games {
            check_live(publication_until, cancel)?;
            let launch = prepare_game_launch_v3(
                &declaration.plan, &declaration.run_root, declaration.max_staging_bytes,
                staging_data_ceiling, game, &run_root, &binary, &declaration.binary, &binary_path, &staging,
                publication_until, cancel,
            )?;
            match series.run_next(launch, Some(cancel)) {
                Ok(verified) => collector.accept(verified, Some(cancel))?,
                Err(failure) => {
                    let message = failure.error.to_string();
                    // Keep original unresolved child custody in the collector. Its
                    // failure owner quarantines physical cleanup if not verified.
                    if let Err(owner) = collector.halt_with_process_failure(failure) {
                        rejected_process_owner = Some(owner);
                    }
                    return Err(invalid(format!(
                        "selfplay game halted; original artifacts retained: {message}"
                    )));
                }
            }
            staging = scan_staging(&run_root, staging_data_ceiling, publication_until, cancel)?;
        }
        Ok(())
    })();
    if let Err(error) = execution_result {
        collector.halt_with_admission_error(error.to_string());
        let partial_publication_error = preserve_partial_publication(
            &mut collector,
            &run_root,
            declaration.max_staging_bytes,
            publication_until,
        )
        .err();
        return Err(SelfplayCliFailureV3 {
            error,
            partial_publication_error,
            owner: Some(Box::new(collector)),
            rejected_process_owner,
        });
    }
    let archive = match collector.finish(Some(cancel)) {
        Ok(archive) => archive,
        Err(failure) => {
            let error = invalid(format!(
                "selfplay final publication failed; raw artifacts retained: {failure:?}"
            ));
            let mut collector = (*failure).into_collector();
            let partial_publication_error = preserve_partial_publication(
                &mut collector,
                &run_root,
                declaration.max_staging_bytes,
                publication_until,
            )
            .err();
            return Err(SelfplayCliFailureV3 {
                error,
                partial_publication_error,
                owner: Some(Box::new(collector)),
                rejected_process_owner,
            });
        }
    };
    let status = serde_json::json!({
        "schema":"rz-pals-selfplay-cli-status/1","status":"completed",
        "games":series.completed_games(),"training_executed":false,
        "declaration_sha256":declaration_sha256,
        "index_sha256":archive.file_sha256(),
        "dataset_manifest_sha256":archive.dataset_manifest_sha256(),
        "index_file":"generation-series-index.json","staging":staging,
        "staging_observation_stage":"after_last_game_before_final_index",
        "publication_bytes_reserved":declaration.max_staging_bytes-staging_data_ceiling,
        "budget_enforcement":"bounded observation; no kernel disk quota asserted"
    });
    Ok(SelfplayCliVerifiedResultV3 { status, archive })
}

/// Original declaration and descriptor preparation for the 128-game series.
#[cfg(target_os = "linux")]
pub(super) struct PreparedSelfplayInputsV3 {
    pub(super) declaration: SelfplayCliDeclarationV3,
    pub(super) run_root: cap_std::fs::Dir,
    pub(super) binary: std::fs::File,
    pub(super) binary_path: PathBuf,
    pub(super) staging: SelfplayCliStagingObservationV3,
    pub(super) staging_data_ceiling: u64,
    pub(super) publication_until: Instant,
}

#[cfg(target_os = "linux")]
pub(super) fn prepare_selfplay_inputs_v3(
    declaration_path: &Path,
    declaration_sha256: &str,
    cancel: &AtomicBool,
) -> Result<PreparedSelfplayInputsV3, ArenaError> {
    use std::io::Read;
    let registered_started = Instant::now();
    check(
        absolute_clean(declaration_path) && generation_sha(declaration_sha256),
        "selfplay CLI absolute original declaration and SHA-256",
    )?;
    let parent = declaration_path
        .parent()
        .ok_or_else(|| invalid("selfplay declaration parent"))?;
    let declaration_owner = open_directory(parent)?;
    let name = declaration_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| invalid("selfplay declaration filename"))?;
    check(
        direct_name(name) && nonsecret_name(name),
        "selfplay declaration safe non-secret filename",
    )?;
    let mut file = open_regular(&declaration_owner, name)?;
    let length = file
        .metadata()
        .map_err(|_| invalid("selfplay declaration metadata"))?
        .len();
    check(
        length > 0 && length <= MAX_DECLARATION_BYTES,
        "selfplay bounded declaration bytes",
    )?;
    let mut bytes = Vec::with_capacity(length as usize);
    Read::take(&mut file, length + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| invalid("selfplay declaration read"))?;
    check(
        bytes.len() as u64 == length
            && format!("{:x}", sha2::Sha256::digest(&bytes)) == declaration_sha256,
        "selfplay original declaration hash",
    )?;
    let declaration: SelfplayCliDeclarationV3 =
        serde_json::from_slice(&bytes).map_err(|_| invalid("selfplay declaration schema"))?;
    declaration.validate()?;
    let preflight_until = registered_started
        .checked_add(Duration::from_millis(declaration.preflight_ms))
        .ok_or_else(|| invalid("selfplay preflight overflow"))?;
    let publication_until = registered_started
        .checked_add(Duration::from_millis(declaration.registered_total_ms()?))
        .ok_or_else(|| invalid("selfplay publication overflow"))?;
    check_live(preflight_until, cancel)?;
    let run_root = open_directory(&declaration.run_root)?;
    let binary_root = open_directory(&declaration.binary_root)?;
    private_root(&run_root)?;
    let staging_data_ceiling = declaration.staging_data_ceiling()?;
    let staging = scan_staging(&run_root, staging_data_ceiling, preflight_until, cancel)?;
    // A metadata scan observes prepared sharing. The CLI does not mount/copy
    // assets. Unique inodes with writable duplicate aliases are rejected.
    let binary = declaration
        .binary
        .open_verified_at(&binary_root, MAX_BINARY_BYTES)?;
    check_live(preflight_until, cancel)?;
    let binary_path = declaration.binary_root.join(&declaration.binary.path);
    Ok(PreparedSelfplayInputsV3 { declaration, run_root, binary, binary_path,
        staging, staging_data_ceiling, publication_until })
}

/// Shared original descriptor-bound preparation for either the normal series
/// or a separately identified one-game diagnostic. This issues no game result.
#[cfg(target_os = "linux")]
pub(super) fn prepare_game_launch_v3(
    plan: &SelfplayCollectionPlanV3,
    run_root_path: &Path,
    max_staging_bytes: u64,
    staging_data_ceiling: u64,
    game: &SelfplayCliGameDeclarationV3,
    run_root: &cap_std::fs::Dir,
    binary: &std::fs::File,
    binary_pin: &ArtifactRef,
    binary_path: &Path,
    staging: &SelfplayCliStagingObservationV3,
    publication_until: Instant,
    cancel: &AtomicBool,
) -> Result<GenerationGameProcessLaunchV3, ArenaError> {
    use cap_fs_ext::DirExt;
    use rz_uci::pals_adaptive_startup::{PalsStartupOptionsV5, PinnedPalsStartupV5};
    use std::io::Read;
    use std::sync::Arc;
    let started = Instant::now();
    let execution_deadline = started
        .checked_add(Duration::from_millis(game.process.execution_ms))
        .ok_or_else(|| invalid("selfplay game execution overflow"))?;
    let whole_deadline = execution_deadline
        .checked_add(Duration::from_millis(game.process.cleanup_ms))
        .ok_or_else(|| invalid("selfplay game cleanup overflow"))?;
    check(
        whole_deadline <= publication_until,
        "selfplay game must fit original series ceiling",
    )?;
    let cwd = run_root
        .open_dir_nofollow(&game.private_directory)
        .map_err(|_| invalid("selfplay predeclared private cwd missing or linked"))?;
    private_root(&cwd)?;
    let request_file = game.request.open_verified_at(&cwd, MAX_REQUEST_BYTES)?;
    check_live(execution_deadline, cancel)?;
    let mut request_bytes = Vec::with_capacity(game.request.bytes as usize);
    Read::take(request_file, game.request.bytes + 1)
        .read_to_end(&mut request_bytes)
        .map_err(|_| invalid("selfplay original game request read"))?;
    let request: GenerationWholeGameRequestV3 = serde_json::from_slice(&request_bytes)
        .map_err(|_| invalid("selfplay original game request schema"))?;
    request.validate()?;
    for (control, config) in request.controls.iter().zip(&request.startup_configs) {
        check(
            control.config_file.split('/').all(nonsecret_name)
                && control.lock_file.split('/').all(nonsecret_name)
                && config.assets.iter().all(|asset| {
                    asset.snapshot_path.split('/').all(nonsecret_name)
                        && asset.artifact.path.split('/').all(nonsecret_name)
                }),
            "selfplay startup paths are not secret-shaped",
        )?;
    }
    let requested_plan = serde_json::to_vec(&request.plan)
        .map_err(|_| invalid("selfplay requested plan serialization"))?;
    let declared_plan = serde_json::to_vec(plan)
        .map_err(|_| invalid("selfplay declared plan serialization"))?;
    check(
        request_bytes.len() as u64 == game.request.bytes
            && format!("{:x}", sha2::Sha256::digest(&request_bytes)) == game.request.sha256
            && request.original_bytes()? == request_bytes
            && request.game_ordinal == game.game_ordinal
            && requested_plan == declared_plan,
        "selfplay original canonical request hash/assignment/plan",
    )?;
    let cwd_path = run_root_path.join(&game.private_directory);
    let prepare = |index: usize| {
        PinnedPalsStartupV5::open(
            PalsStartupOptionsV5 {
                config_path: cwd_path.join(&request.controls[index].config_file),
                lock_path: cwd_path.join(&request.controls[index].lock_file),
                config_sha256: request.startup_config_sha256[index].clone(),
                metadata_preflight: false,
            },
            &cwd_path,
        )
        .map(Arc::new)
        .map_err(ArenaError::from)
    };
    let pinned = [prepare(0)?, prepare(1)?];
    check_live(execution_deadline, cancel)?;
    let current = scan_staging(
        &cwd,
        max_staging_bytes,
        execution_deadline,
        cancel,
    )?;
    let remaining = staging_data_ceiling
        .checked_sub(staging.unique_inode_bytes)
        .ok_or_else(|| invalid("selfplay staging capacity"))?;
    let watch_ceiling = current
        .logical_alias_bytes
        .checked_add(remaining)
        .ok_or_else(|| invalid("selfplay watch allocation overflow"))?
        .min(game.tree.max_total_bytes);
    check(
        current.logical_alias_bytes < watch_ceiling,
        "selfplay no declared output headroom",
    )?;
    check(
        current.observed_max_file_bytes <= game.tree.max_file_bytes
            && current.unique_regular_files + current.readonly_shared_aliases
                <= game.tree.max_files
            && current.observed_directories <= game.tree.max_files
            && current.observed_max_depth <= game.tree.max_depth,
        "selfplay prepared cwd must fit original child watch bounds",
    )?;
    let launch = GenerationGameProcessLaunchV3 {
        binary: binary
            .try_clone()
            .map_err(|_| invalid("selfplay original binary clone"))?,
        binary_path: binary_path.to_path_buf(),
        binary_pin: binary_pin.clone(),
        cwd: cwd.into_std_file(),
        cwd_path,
        pinned,
        request,
        started,
        execution_deadline,
        whole_deadline,
        limits: ProcessLimits {
            wall_ms: game.process.execution_ms,
            shutdown_grace_ms: game.process.cleanup_ms,
            max_output_bytes: game.process.max_output_bytes,
            max_child_processes: game.process.max_child_processes,
        },
        watch: OwnedArtifactTreeWatch {
            max_total_bytes: watch_ceiling,
            max_file_bytes: game.tree.max_file_bytes.min(watch_ceiling),
            max_files: game.tree.max_files,
            max_depth: game.tree.max_depth,
        },
        raw_limits: GenerationGameRawLimitsV3 {
            max_records: game.raw.max_records,
            max_evidence: game.raw.max_evidence,
            max_retained_bytes: game.raw.max_retained_bytes,
            max_read_bytes: game.raw.max_read_bytes,
        },
    };
    Ok(launch)
}

#[cfg(target_os = "linux")]
fn preserve_partial_publication(
    collector: &mut GenerationSeriesCollectorV3,
    run_root: &cap_std::fs::Dir,
    max_staging_bytes: u64,
    publication_until: Instant,
) -> Result<(), ArenaError> {
    // Cancellation ends admission and execution. A bounded metadata scan and
    // diagnostic write may still preserve the stopped execution's originals;
    // neither operation dispatches or resumes work. Failed/partial child trees
    // are included, rather than assuming only verified archives consume space.
    let preservation_only = AtomicBool::new(false);
    let observed = scan_staging(
        run_root,
        max_staging_bytes,
        publication_until,
        &preservation_only,
    );
    let remaining = match observed {
        Ok(observed) => max_staging_bytes
            .checked_sub(observed.unique_inode_bytes)
            .ok_or_else(|| invalid("selfplay partial staging capacity"))?,
        Err(error) => {
            // Unknown headroom never authorizes a guessed diagnostic allowance.
            collector.reserve_partial_publication_capacity_v3(0)?;
            return Err(error);
        }
    };
    collector.reserve_partial_publication_capacity_v3(remaining)?;
    collector.publish_partial().map(|_| ())
}

#[cfg(target_os = "linux")]
pub(super) fn check_live(until: Instant, cancel: &AtomicBool) -> Result<(), ArenaError> {
    check(
        !cancel.load(Ordering::Acquire) && Instant::now() < until,
        "selfplay cancelled or original deadline expired",
    )
}
#[cfg(target_os = "linux")]
pub(super) fn open_directory(path: &Path) -> Result<cap_std::fs::Dir, ArenaError> {
    use cap_fs_ext::DirExt;
    use std::os::unix::fs::OpenOptionsExt;
    check(
        absolute_clean(path),
        "selfplay absolute no-follow directory",
    )?;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open("/")
        .map_err(|_| invalid("selfplay filesystem root pin"))?;
    let mut directory = cap_std::fs::Dir::from_std_file(file);
    for component in path.components() {
        if let Component::Normal(name) = component {
            directory = directory
                .open_dir_nofollow(name)
                .map_err(|_| invalid("selfplay directory unavailable or linked"))?;
        }
    }
    Ok(directory)
}
#[cfg(target_os = "linux")]
pub(super) fn open_regular(
    directory: &cap_std::fs::Dir,
    name: &str,
) -> Result<std::fs::File, ArenaError> {
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    let mut options = cap_std::fs::OpenOptions::new();
    options
        .read(true)
        .follow(FollowSymlinks::No)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC);
    let file = directory
        .open_with(name, &options)
        .map_err(|_| invalid("selfplay no-follow file open"))?
        .into_std();
    use std::os::unix::fs::MetadataExt;
    let metadata = file
        .metadata()
        .map_err(|_| invalid("selfplay file metadata"))?;
    check(
        metadata.is_file() && metadata.nlink() == 1,
        "selfplay regular single-link file",
    )?;
    Ok(file)
}
#[cfg(target_os = "linux")]
pub(crate) fn private_root(directory: &cap_std::fs::Dir) -> Result<(), ArenaError> {
    use std::os::unix::fs::MetadataExt;
    let metadata = directory
        .try_clone()
        .map_err(|_| invalid("selfplay directory clone"))?
        .into_std_file()
        .metadata()
        .map_err(|_| invalid("selfplay private directory metadata"))?;
    check(
        metadata.is_dir()
            && metadata.mode() & 0o077 == 0
            && metadata.uid() == nix::unistd::Uid::effective().as_raw(),
        "selfplay exclusively owned private root",
    )
}
#[cfg(target_os = "linux")]
pub(crate) fn scan_staging(
    root: &cap_std::fs::Dir,
    cap: u64,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<SelfplayCliStagingObservationV3, ArenaError> {
    Ok(scan_staging_inner(root, cap, None, until, cancel)?.0)
}

/// Only the G0 CUDA parent's already prepared, exact-FD tree may contribute
/// validated alias metadata. It is kept separate from ordinary regular-file
/// observations; a provider file count is not relabelled as a regular count.
#[cfg(target_os = "linux")]
pub(crate) struct DelegatedStagingObservation {
    pub(crate) ordinary: SelfplayCliStagingObservationV3,
    pub(crate) delegated: crate::process::DelegatedTreeSnapshot,
}
#[cfg(target_os = "linux")]
impl DelegatedStagingObservation {
    pub(crate) fn accounted_bytes(&self) -> Result<u64, ArenaError> {
        self.ordinary
            .unique_inode_bytes
            .checked_add(self.delegated.bytes)
            .ok_or_else(|| invalid("selfplay delegated staging byte sum overflow"))
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn scan_staging_delegated(
    root: &cap_std::fs::Dir,
    cap: u64,
    delegated: &dyn crate::process::DelegatedArtifactTreeWatch,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<DelegatedStagingObservation, ArenaError> {
    let (ordinary, delegated) = scan_staging_inner(root, cap, Some(delegated), until, cancel)?;
    Ok(DelegatedStagingObservation {
        ordinary,
        delegated: delegated.ok_or_else(|| invalid("selfplay delegated staging tree absent"))?,
    })
}

#[cfg(target_os = "linux")]
fn scan_staging_inner(
    root: &cap_std::fs::Dir,
    cap: u64,
    delegated: Option<&dyn crate::process::DelegatedArtifactTreeWatch>,
    until: Instant,
    cancel: &AtomicBool,
) -> Result<
    (
        SelfplayCliStagingObservationV3,
        Option<crate::process::DelegatedTreeSnapshot>,
    ),
    ArenaError,
> {
    use cap_fs_ext::DirExt;
    use std::os::unix::fs::MetadataExt;
    let mut todo = vec![(
        root.try_clone()
            .map_err(|_| invalid("selfplay scan root clone"))?,
        0usize,
    )];
    let mut seen = BTreeMap::<(u64, u64), (u64, bool)>::new();
    let mut unique_bytes = 0u64;
    let mut logical_bytes = 0u64;
    let mut aliases = 0usize;
    let mut entries = 0usize;
    let mut directories = 1usize;
    let mut max_depth = 0usize;
    let mut max_file_bytes = 0u64;
    let mut delegated_snapshot: Option<crate::process::DelegatedTreeSnapshot> = None;
    while let Some((directory, depth)) = todo.pop() {
        if let Some(watch) = delegated {
            check_live(until, cancel)?;
            let file = directory
                .try_clone()
                .map_err(|_| invalid("selfplay delegated directory clone"))?
                .into_std_file();
            let matched = watch.matches_directory(&file).map_err(invalid)?;
            check_live(until, cancel)?;
            if matched {
                check(
                    depth > 0 && delegated_snapshot.is_none(),
                    "selfplay delegated tree must match exactly once below staging root",
                )?;
                let observed = watch.observe_directory(&file).map_err(invalid)?;
                check_live(until, cancel)?;
                observed.validate().map_err(invalid)?;
                entries = entries
                    .checked_add(observed.entries)
                    .ok_or_else(|| invalid("selfplay delegated staging entry sum overflow"))?;
                check(entries <= 65_536, "selfplay staging bounded entry count")?;
                check(
                    directories
                        .checked_add(observed.directories)
                        .is_some_and(|count| count <= 4096)
                        && depth
                            .checked_add(observed.depth)
                            .is_some_and(|count| count <= 32),
                    "selfplay delegated staging directory/depth count",
                )?;
                check(
                    unique_bytes
                        .checked_add(observed.bytes)
                        .is_some_and(|bytes| bytes <= cap),
                    "selfplay delegated staging Capacity exceeded",
                )?;
                delegated_snapshot = Some(observed);
                continue;
            }
        }
        for entry in directory
            .entries()
            .map_err(|_| invalid("selfplay staging entries"))?
        {
            check_live(until, cancel)?;
            entries += 1;
            check(entries <= 65_536, "selfplay staging bounded entry count")?;
            let entry = entry.map_err(|_| invalid("selfplay staging entry"))?;
            let name = entry.file_name();
            let before = directory
                .symlink_metadata(&name)
                .map_err(|_| invalid("selfplay staging metadata"))?;
            if before.is_dir() {
                directories += 1;
                max_depth = max_depth.max(depth + 1);
                check(
                    depth < 32
                        && directories
                            .checked_add(
                                delegated_snapshot
                                    .as_ref()
                                    .map_or(0, |observed| observed.directories),
                            )
                            .is_some_and(|count| count <= 4096),
                    "selfplay staging directory/depth count",
                )?;
                todo.push((
                    directory
                        .open_dir_nofollow(&name)
                        .map_err(|_| invalid("selfplay staging directory changed or linked"))?,
                    depth + 1,
                ));
            } else {
                let text = name
                    .to_str()
                    .ok_or_else(|| invalid("selfplay staging filename encoding"))?;
                // A prepared execution root is not a credential store.
                check(nonsecret_name(text), "selfplay secret-shaped file excluded")?;
                let file = open_regular(&directory, text)?;
                let metadata = file
                    .metadata()
                    .map_err(|_| invalid("selfplay staging held metadata"))?;
                max_file_bytes = max_file_bytes.max(metadata.len());
                let key = (metadata.dev(), metadata.ino());
                let readonly = nix::sys::statvfs::fstatvfs(&file)
                    .map_err(|_| invalid("selfplay read-only mount observation unavailable"))?
                    .flags()
                    .contains(nix::sys::statvfs::FsFlags::ST_RDONLY);
                logical_bytes = logical_bytes
                    .checked_add(metadata.len())
                    .ok_or_else(|| invalid("selfplay staging logical sum overflow"))?;
                if let Some((length, first_readonly)) = seen.get(&key) {
                    check(
                        *length == metadata.len() && *first_readonly && readonly,
                        "selfplay shared inode aliases require observed read-only mounts",
                    )?;
                    aliases += 1;
                } else {
                    unique_bytes = unique_bytes
                        .checked_add(metadata.len())
                        .ok_or_else(|| invalid("selfplay staging inode sum overflow"))?;
                    check(
                        unique_bytes
                            .checked_add(
                                delegated_snapshot
                                    .as_ref()
                                    .map_or(0, |observed| observed.bytes),
                            )
                            .is_some_and(|bytes| bytes <= cap),
                        "selfplay staging Capacity exceeded",
                    )?;
                    seen.insert(key, (metadata.len(), readonly));
                }
            }
        }
    }
    check(
        delegated.is_none() || delegated_snapshot.is_some(),
        "selfplay delegated staging tree absent",
    )?;
    Ok((
        SelfplayCliStagingObservationV3 {
            unique_inode_bytes: unique_bytes,
            logical_alias_bytes: logical_bytes,
            unique_regular_files: seen.len(),
            readonly_shared_aliases: aliases,
            observed_max_file_bytes: max_file_bytes,
            observed_directories: directories,
            observed_max_depth: max_depth,
            enforcement: "held regular-file lengths; readonly-mount readback; bounded periodic observation",
        },
        delegated_snapshot,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn declaration_names_do_not_grant_parent_or_hidden_authority() {
        for invalid in ["", "..", ".owner", "a/b", "a\\b", "space name"] {
            assert!(!direct_name(invalid));
        }
        assert!(direct_name("game-000"));
    }
    #[test]
    fn explicit_inputs_exclude_secret_shaped_names_before_opening() {
        for name in [
            ".env",
            ".env.production",
            "ssh-key.key",
            "account.pem",
            "service_account.json",
            "application_default_credentials.json",
        ] {
            assert!(!nonsecret_name(name));
        }
        assert!(nonsecret_name("generation-request.json"));
        assert!(nonsecret_name("runtime.so"));
    }

    #[cfg(target_os = "linux")]
    mod delegated_scan {
        use super::*;
        use crate::process::{DelegatedArtifactTreeWatch, DelegatedTreeSnapshot};
        use std::os::unix::fs::{DirBuilderExt, MetadataExt};
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::Duration;

        struct Fixture {
            path: PathBuf,
            directory: cap_std::fs::Dir,
            parent: PathBuf,
            identity: (u64, u64),
            parent_identity: (u64, u64),
        }
        impl Fixture {
            fn new() -> Self {
                static NEXT: AtomicU64 = AtomicU64::new(0);
                let parent = std::env::temp_dir().canonicalize().unwrap();
                let nonce = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                let path = parent.join(format!(
                    "rz-delegated-staging-test-{}-{}-{nonce}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                std::fs::DirBuilder::new()
                    .mode(0o700)
                    .create(&path)
                    .unwrap();
                let metadata = std::fs::symlink_metadata(&path).unwrap();
                let parent_metadata = std::fs::symlink_metadata(&parent).unwrap();
                Self {
                    directory: open_directory(&path).unwrap(),
                    path,
                    parent,
                    identity: (metadata.dev(), metadata.ino()),
                    parent_identity: (parent_metadata.dev(), parent_metadata.ino()),
                }
            }
            fn subdir(&self, name: &str) -> (u64, u64) {
                let path = self.path.join(name);
                std::fs::DirBuilder::new()
                    .mode(0o700)
                    .create(&path)
                    .unwrap();
                let metadata = std::fs::symlink_metadata(path).unwrap();
                (metadata.dev(), metadata.ino())
            }
        }
        impl Drop for Fixture {
            fn drop(&mut self) {
                let (Ok(metadata), Ok(parent)) = (
                    std::fs::symlink_metadata(&self.path),
                    std::fs::symlink_metadata(&self.parent),
                ) else {
                    return;
                };
                if self.path.is_absolute()
                    && self.path.parent() == Some(self.parent.as_path())
                    && metadata.is_dir()
                    && !metadata.file_type().is_symlink()
                    && (metadata.dev(), metadata.ino()) == self.identity
                    && (parent.dev(), parent.ino()) == self.parent_identity
                {
                    std::fs::remove_dir_all(&self.path).unwrap();
                }
            }
        }
        struct MockWatch {
            directories: Vec<(u64, u64)>,
            snapshot: DelegatedTreeSnapshot,
        }
        impl DelegatedArtifactTreeWatch for MockWatch {
            fn matches_directory(&self, file: &std::fs::File) -> Result<bool, &'static str> {
                let metadata = file.metadata().map_err(|_| "fixture directory metadata")?;
                Ok(self.directories.contains(&(metadata.dev(), metadata.ino())))
            }
            fn observe_directory(
                &self,
                file: &std::fs::File,
            ) -> Result<DelegatedTreeSnapshot, &'static str> {
                if !self.matches_directory(file)? {
                    return Err("fixture exact directory required");
                }
                Ok(self.snapshot)
            }
        }
        fn one_file() -> DelegatedTreeSnapshot {
            DelegatedTreeSnapshot {
                bytes: 1,
                files: 1,
                directories: 0,
                entries: 1,
                max_file_bytes: 1,
                depth: 0,
            }
        }
        fn scan(
            fixture: &Fixture,
            cap: u64,
            watch: &MockWatch,
        ) -> Result<DelegatedStagingObservation, ArenaError> {
            scan_staging_delegated(
                &fixture.directory,
                cap,
                watch,
                Instant::now() + Duration::from_secs(5),
                &AtomicBool::new(false),
            )
        }

        #[test]
        fn exact_delegated_alias_metadata_stays_separate_and_charged() {
            let fixture = Fixture::new();
            let identity = fixture.subdir("prepared");
            let root = fixture.path.join("prepared");
            for index in 0..19 {
                std::fs::write(root.join(format!("data-{index:02}")), b"abc").unwrap();
            }
            for index in 0..8 {
                std::os::unix::fs::symlink("data-00", root.join(format!("alias-{index:02}")))
                    .unwrap();
            }
            std::fs::write(fixture.path.join("ordinary"), b"12345").unwrap();
            // Mock-only owner observation: this is not a CUDA admission.
            let watch = MockWatch {
                directories: vec![identity],
                snapshot: DelegatedTreeSnapshot {
                    bytes: 19 * 3 + 8 * 7,
                    files: 27,
                    directories: 0,
                    entries: 27,
                    max_file_bytes: 7,
                    depth: 0,
                },
            };
            let observed = scan(&fixture, 118, &watch).unwrap();
            assert_eq!(observed.ordinary.unique_inode_bytes, 5);
            assert_eq!(observed.ordinary.unique_regular_files, 1);
            assert_eq!(observed.delegated.files, 27);
            assert_eq!(observed.delegated.bytes, 113);
            assert_eq!(observed.accounted_bytes().unwrap(), 118);
            assert!(scan(&fixture, 117, &watch).is_err());
            assert!(
                scan_staging(
                    &fixture.directory,
                    118,
                    Instant::now() + Duration::from_secs(5),
                    &AtomicBool::new(false)
                )
                .is_err()
            );
            std::os::unix::fs::symlink("ordinary", fixture.path.join("unregistered-alias"))
                .unwrap();
            assert!(scan(&fixture, 1024, &watch).is_err());
        }

        #[test]
        fn delegated_directory_must_be_exactly_one_nonroot_match() {
            let fixture = Fixture::new();
            let first = fixture.subdir("first");
            let second = fixture.subdir("second");
            for directories in [vec![], vec![fixture.identity], vec![first, second]] {
                let watch = MockWatch {
                    directories,
                    snapshot: one_file(),
                };
                assert!(scan(&fixture, 1024, &watch).is_err());
            }
        }

        #[test]
        fn delegated_snapshot_counts_depth_and_overflow_keep_original_limits() {
            let fixture = Fixture::new();
            let identity = fixture.subdir("prepared");
            std::fs::write(fixture.path.join("ordinary"), b"12345").unwrap();
            let cases = [
                DelegatedTreeSnapshot {
                    entries: 65_536,
                    ..one_file()
                },
                DelegatedTreeSnapshot {
                    directories: 4096,
                    entries: 4097,
                    depth: 1,
                    ..one_file()
                },
                DelegatedTreeSnapshot {
                    directories: 32,
                    entries: 33,
                    depth: 32,
                    ..one_file()
                },
                DelegatedTreeSnapshot {
                    bytes: u64::MAX,
                    max_file_bytes: u64::MAX,
                    ..one_file()
                },
                DelegatedTreeSnapshot {
                    entries: 0,
                    ..one_file()
                },
            ];
            for snapshot in cases {
                let watch = MockWatch {
                    directories: vec![identity],
                    snapshot,
                };
                assert!(scan(&fixture, u64::MAX, &watch).is_err());
            }
        }

        #[test]
        fn delegated_scan_preserves_original_deadline_and_cancel() {
            let fixture = Fixture::new();
            let watch = MockWatch {
                directories: vec![fixture.subdir("prepared")],
                snapshot: one_file(),
            };
            for (until, canceled) in [
                (Instant::now(), false),
                (Instant::now() + Duration::from_secs(5), true),
            ] {
                assert!(
                    scan_staging_delegated(
                        &fixture.directory,
                        1024,
                        &watch,
                        until,
                        &AtomicBool::new(canceled)
                    )
                    .is_err()
                );
            }
            struct CancelOnObserve<'a> {
                watch: &'a MockWatch,
                cancel: &'a AtomicBool,
            }
            impl DelegatedArtifactTreeWatch for CancelOnObserve<'_> {
                fn matches_directory(&self, file: &std::fs::File) -> Result<bool, &'static str> {
                    self.watch.matches_directory(file)
                }
                fn observe_directory(
                    &self,
                    file: &std::fs::File,
                ) -> Result<DelegatedTreeSnapshot, &'static str> {
                    let observed = self.watch.observe_directory(file)?;
                    self.cancel.store(true, Ordering::Relaxed);
                    Ok(observed)
                }
            }
            let cancel = AtomicBool::new(false);
            assert!(
                scan_staging_delegated(
                    &fixture.directory,
                    1024,
                    &CancelOnObserve {
                        watch: &watch,
                        cancel: &cancel,
                    },
                    Instant::now() + Duration::from_secs(5),
                    &cancel
                )
                .is_err()
            );
        }
    }
}
