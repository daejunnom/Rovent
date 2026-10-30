//! One actual, non-promotable diagnostic game and its retained pre-move root.
//!
//! This is a typed in-process handoff. It does not create a completed
//! generation corpus, issue training authority, or reconstruct authority from
//! a saved receipt. The normal 128-game collection path is unchanged.
use super::{
    ArenaError, GenerationGameProcessFailureV3, GenerationGameProcessLaunchV3,
    GenerationGameProcessSeriesV3, GenerationSavedActionRootV3,
    VerifiedGenerationGameArchiveV3, SelfplayCollectionPlanV3,
    SelfplayCliGameDeclarationV3,
};
use rz_experiments::ArtifactRef;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, atomic::AtomicBool};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const DIAGNOSTIC_GAME_DECLARATION_V1: &str = "rz-pals-diagnostic-game-declaration/1";

/// Independent one-game diagnostic registration. Its plan remains the
/// original 64-opening schedule, but it registers just one actual child and
/// cannot be used as the completed 128-game collection declaration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticGameDeclarationV1 {
    pub schema: String,
    pub plan: SelfplayCollectionPlanV3,
    pub run_root: PathBuf,
    pub binary_root: PathBuf,
    pub binary: ArtifactRef,
    pub game: SelfplayCliGameDeclarationV3,
    pub decision_index: usize,
    pub preflight_ms: u64,
    pub root_read_ms: u64,
    pub max_staging_bytes: u64,
    pub max_root_read_bytes: u64,
}

impl DiagnosticGameDeclarationV1 {
    fn validate(&self) -> Result<(), ArenaError> {
        use super::{check, GENERATION_OUTPUT_CAP};
        self.plan.validate()?;
        self.binary.validate()?;
        check(self.schema == DIAGNOSTIC_GAME_DECLARATION_V1
            && super::selfplay_cli::absolute_clean(&self.run_root)
            && super::selfplay_cli::absolute_clean(&self.binary_root)
            && self.binary.bytes <= super::selfplay_cli::MAX_BINARY_BYTES
            && self.binary.path.split('/').all(super::selfplay_cli::nonsecret_name)
            && (1..=300_000).contains(&self.preflight_ms)
            && (1..=300_000).contains(&self.root_read_ms)
            && self.decision_index < 512
            && self.max_staging_bytes > 0 && self.max_staging_bytes <= GENERATION_OUTPUT_CAP
            && self.max_root_read_bytes > 0
            && self.max_root_read_bytes <= super::game_process::MAX_OUTPUT_BYTES,
            "bounded independent diagnostic declaration")?;
        super::selfplay_cli::validate_game_declaration_v3(&self.game, 0, self.max_staging_bytes)?;
        self.staging_data_ceiling()?;
        Ok(())
    }

    fn staging_data_ceiling(&self) -> Result<u64, ArenaError> {
        self.max_staging_bytes.checked_sub(self.game.process.max_output_bytes)
            .filter(|ceiling| *ceiling > 0)
            .ok_or_else(|| super::invalid("diagnostic staging must reserve failed pipe output"))
    }
}

/// The actual process/archive stays alive for as long as this diagnostic root
/// is held. The archive is not a complete generation dataset.
pub struct DiagnosticGameRootV1 {
    archive: Arc<VerifiedGenerationGameArchiveV3>,
    root: GenerationSavedActionRootV3,
    root_read_bytes: u64,
}

impl DiagnosticGameRootV1 {
    pub fn archive(&self) -> &Arc<VerifiedGenerationGameArchiveV3> { &self.archive }
    pub fn root(&self) -> &GenerationSavedActionRootV3 { &self.root }
    pub fn root_read_bytes(&self) -> u64 { self.root_read_bytes }
}

/// A failure retains the actual failed process owner or the successfully
/// closed game archive; neither arm claims semantic or physical completion.
pub enum DiagnosticGameFailureV1 {
    Preparation(ArenaError),
    Game(Box<GenerationGameProcessFailureV3>),
    RootSelection { error: ArenaError, archive: Arc<VerifiedGenerationGameArchiveV3> },
}

/// Prepare from the separately pinned one-game diagnostic declaration.
/// The underlying game still uses the actual registered plan and startup.
pub fn run_diagnostic_game_from_declaration_v1(
    declaration_path: &Path,
    declaration_sha256: &str,
    decision_index: usize,
    root_until: Instant,
    cancel: &AtomicBool,
    max_read_bytes: u64,
) -> Result<DiagnosticGameRootV1, DiagnosticGameFailureV1> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (declaration_path, declaration_sha256, decision_index, root_until, cancel, max_read_bytes);
        Err(DiagnosticGameFailureV1::Preparation(super::invalid("diagnostic original process transport requires Linux")))
    }
    #[cfg(target_os = "linux")]
    {
        use sha2::Digest as _;
        use std::io::Read;
        let prepare = || -> Result<GenerationGameProcessLaunchV3, ArenaError> {
            let started = Instant::now();
            super::check(super::selfplay_cli::absolute_clean(declaration_path)
                && super::generation_sha(declaration_sha256), "diagnostic original declaration and SHA")?;
            let parent = declaration_path.parent().ok_or_else(|| super::invalid("diagnostic declaration parent"))?;
            let directory = super::selfplay_cli::open_directory(parent)?;
            let name = declaration_path.file_name().and_then(|name| name.to_str())
                .ok_or_else(|| super::invalid("diagnostic declaration filename"))?;
            super::check(super::selfplay_cli::direct_name(name) && super::selfplay_cli::nonsecret_name(name),
                "diagnostic non-secret declaration filename")?;
            let file = super::selfplay_cli::open_regular(&directory, name)?;
            let length = file.metadata().map_err(|_| super::invalid("diagnostic declaration metadata"))?.len();
            super::check(length > 0 && length <= super::selfplay_cli::MAX_DECLARATION_BYTES,
                "diagnostic bounded declaration bytes")?;
            let mut bytes = Vec::with_capacity(length as usize);
            Read::take(file, length + 1).read_to_end(&mut bytes)
                .map_err(|_| super::invalid("diagnostic original declaration read"))?;
            super::check(bytes.len() as u64 == length
                && format!("{:x}", sha2::Sha256::digest(&bytes)) == declaration_sha256,
                "diagnostic original declaration hash")?;
            let declaration: DiagnosticGameDeclarationV1 = serde_json::from_slice(&bytes)
                .map_err(|_| super::invalid("diagnostic declaration schema"))?;
            declaration.validate()?;
            super::check(decision_index == declaration.decision_index,
                "diagnostic decision must match the registered original schedule")?;
            super::check(max_read_bytes > 0 && max_read_bytes <= declaration.max_root_read_bytes,
                "diagnostic root read credit exceeds registration")?;
            let preflight_until = started.checked_add(Duration::from_millis(declaration.preflight_ms))
                .ok_or_else(|| super::invalid("diagnostic preflight overflow"))?;
            let game_ceiling = preflight_until.checked_add(Duration::from_millis(declaration.game.process.execution_ms))
                .and_then(|deadline| deadline.checked_add(Duration::from_millis(declaration.game.process.cleanup_ms)))
                .ok_or_else(|| super::invalid("diagnostic game ceiling overflow"))?;
            let registered_root_until = game_ceiling.checked_add(Duration::from_millis(declaration.root_read_ms))
                .ok_or_else(|| super::invalid("diagnostic root read ceiling overflow"))?;
            super::check(root_until <= registered_root_until, "diagnostic root read deadline exceeds registration")?;
            super::selfplay_cli::check_live(preflight_until, cancel)?;
            let run_root = super::selfplay_cli::open_directory(&declaration.run_root)?;
            let binary_root = super::selfplay_cli::open_directory(&declaration.binary_root)?;
            super::selfplay_cli::private_root(&run_root)?;
            let staging_data_ceiling = declaration.staging_data_ceiling()?;
            let staging = super::selfplay_cli::scan_staging(&run_root, staging_data_ceiling, preflight_until, cancel)?;
            let binary = declaration.binary.open_verified_at(&binary_root, super::selfplay_cli::MAX_BINARY_BYTES)?;
            super::selfplay_cli::check_live(preflight_until, cancel)?;
            let binary_path = declaration.binary_root.join(&declaration.binary.path);
            super::selfplay_cli::prepare_game_launch_v3(
                &declaration.plan, &declaration.run_root, declaration.max_staging_bytes,
                staging_data_ceiling, &declaration.game, &run_root, &binary, &declaration.binary, &binary_path,
                &staging, game_ceiling, cancel,
            )
        };
        let launch = prepare().map_err(DiagnosticGameFailureV1::Preparation)?;
        if root_until <= launch.whole_deadline || max_read_bytes == 0 {
            return Err(DiagnosticGameFailureV1::Preparation(super::invalid(
                "diagnostic root read window must follow the original game cleanup boundary",
            )));
        }
        run_diagnostic_game_root_v1(launch, decision_index, root_until, Some(cancel), max_read_bytes)
    }
}

/// Execute one original collection game and select an actual pre-move root.
///
/// The launch owns its original process windows, binary and startup pins. A
/// fresh series enforces the registered first-game ordinal. `root_until` is a
/// separate finite read/replay boundary, not a renewal of the game's deadline.
/// Its returned typed owner may be used for diagnostics, never as a completed
/// 128-game source or as a training corpus by itself.
pub fn run_diagnostic_game_root_v1(
    launch: GenerationGameProcessLaunchV3,
    decision_index: usize,
    root_until: Instant,
    cancel: Option<&AtomicBool>,
    max_read_bytes: u64,
) -> Result<DiagnosticGameRootV1, DiagnosticGameFailureV1> {
    let mut series = GenerationGameProcessSeriesV3::default();
    let actual = series.run_next(launch, cancel).map_err(DiagnosticGameFailureV1::Game)?;
    let archive = Arc::new(actual.into_archive());
    let (root, root_read_bytes) = match archive.saved_action_root(
        decision_index, root_until, cancel, max_read_bytes,
    ) {
        Ok(root) => root,
        Err(error) => return Err(DiagnosticGameFailureV1::RootSelection { error, archive }),
    };
    Ok(DiagnosticGameRootV1 { archive, root, root_read_bytes })
}
