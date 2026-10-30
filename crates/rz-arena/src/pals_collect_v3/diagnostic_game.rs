//! One actual, non-promotable diagnostic game and its retained pre-move root.
//!
//! This is a typed in-process handoff. It does not create a completed
//! generation corpus, issue training authority, or reconstruct authority from
//! a saved receipt. The normal 128-game collection path is unchanged.
use super::{
    ArenaError, GenerationGameProcessFailureV3, GenerationGameProcessLaunchV3,
    GenerationGameProcessSeriesV3, GenerationSavedActionRootV3,
    VerifiedGenerationGameArchiveV3,
};
use std::sync::{Arc, atomic::AtomicBool};
use std::path::Path;
use std::time::Instant;

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

/// Prepare from the original pinned selfplay declaration and run only its
/// first registered game. The entire 128-game declaration is still validated;
/// it is not claimed to have been executed or published by this diagnostic.
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
        let prepared = super::selfplay_cli::prepare_selfplay_inputs_v3(
            declaration_path, declaration_sha256, cancel, true,
        ).map_err(DiagnosticGameFailureV1::Preparation)?;
        let game = prepared.declaration.games.first().ok_or_else(||
            DiagnosticGameFailureV1::Preparation(super::invalid("diagnostic first game absent")))?;
        let launch = super::selfplay_cli::prepare_game_launch_v3(
            &prepared.declaration, game, &prepared.run_root, &prepared.binary,
            &prepared.binary_path, &prepared.staging, prepared.publication_until, cancel,
        ).map_err(DiagnosticGameFailureV1::Preparation)?;
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
