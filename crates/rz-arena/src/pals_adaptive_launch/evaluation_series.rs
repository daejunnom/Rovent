//! One preregistered final G0/G1 series over the existing native pair executor.
//! Declarations and statistical projections are not native execution evidence.
//! Original closed scopes, failed scopes, PGN and clock observations stay owned.
use super::*;
use crate::{ClosedNativePairArchive, NativePairOutput};
#[cfg(feature = "pals-collection-onnx")]
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub const PALS_EVALUATION_SERIES_V5_DOMAIN: &str = "rz-pals-final-evaluation-series-v5/1";
pub const PALS_EVALUATION_PAIRS_V5: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PalsEvaluationOpeningV5 {
    pub opening_id: String,
    pub position_command: String,
    pub opening_artifact: ArtifactRef,
}

/// Registration happens before collection. This immutable owner is retained by
/// the generation pipeline; there is no decoder to substitute openings after
/// learning or viewing outcomes. It is a declaration, not an execution permit.
#[derive(Debug)]
pub struct PalsEvaluationOpeningRegistrationV5 {
    registration_id: String,
    generation_manifest_sha256: String,
    g0_deployment: PalsEngineV5,
    openings: Vec<PalsEvaluationOpeningV5>,
    sha256: String,
}
impl PalsEvaluationOpeningRegistrationV5 {
    pub fn register(
        registration_id: String,
        generation_manifest_sha256: String,
        g0_deployment: PalsEngineV5,
        openings: Vec<PalsEvaluationOpeningV5>,
    ) -> Result<Self, ArenaError> {
        require(
            series_identifier(&registration_id),
            "invalid final evaluation registration ID",
        )?;
        raw_sha(&generation_manifest_sha256)?;
        g0_deployment.validate()?;
        require(
            g0_deployment.mode == PalsEngineModeV5::Deployment
                && g0_deployment.model.profile == "virtual_review_v1"
                && g0_deployment.model.artifact_domain == PalsArtifactDomainV5::Untrained
                && g0_deployment.context.selection == PalsViewSelectionV5::VirtualReview
                && g0_deployment.cpu.cpu_t == PalsCpuSelectionV5::Disabled
                && g0_deployment.cpu.cpu_r == PalsCpuSelectionV5::Own
                && !g0_deployment.v_generate
                && !g0_deployment.v_load
                && !g0_deployment.v_call,
            "G0 final evaluation must be untrained compact, own CPU_R and V-free",
        )?;
        require(
            openings.len() == PALS_EVALUATION_PAIRS_V5,
            "register exactly 32 ordered final openings",
        )?;
        let mut ids = BTreeSet::new();
        let mut exact_inputs = BTreeSet::new();
        for o in &openings {
            require(
                series_identifier(&o.opening_id) && ids.insert(&o.opening_id),
                "final opening ID duplicated or invalid",
            )?;
            require(
                o.position_command.len() <= 16 * 1024
                    && o.position_command
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                        == o.position_command
                    && exact_inputs.insert(&o.position_command),
                "final opening duplicated or noncanonical",
            )?;
            o.opening_artifact.validate()?;
            let pair = registration_pair(o, &g0_deployment.id, "registered-g1");
            let specification = opening(&pair)?;
            crate::validate_opening_artifact_for_spec(&specification, 256, &o.opening_artifact)?;
        }
        let sha256 = crate::canonical_sha256(&(
            PALS_EVALUATION_SERIES_V5_DOMAIN,
            &registration_id,
            &generation_manifest_sha256,
            &g0_deployment,
            &openings,
            NativePilotStatistics::FixedPairedHoeffding95,
            120_000u64,
            1_000u64,
            256u32,
            1u64,
        ))?;
        Ok(Self {
            registration_id,
            generation_manifest_sha256,
            g0_deployment,
            openings,
            sha256,
        })
    }
    pub fn registration_id(&self) -> &str {
        &self.registration_id
    }
    pub fn generation_manifest_sha256(&self) -> &str {
        &self.generation_manifest_sha256
    }
    pub fn g0_deployment(&self) -> &PalsEngineV5 {
        &self.g0_deployment
    }
    pub fn openings(&self) -> &[PalsEvaluationOpeningV5] {
        &self.openings
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn statistics_method(&self) -> NativePilotStatistics {
        NativePilotStatistics::FixedPairedHoeffding95
    }
}
fn series_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}
pub(super) fn registration_pair(o: &PalsEvaluationOpeningV5, g0: &str, g1: &str) -> PalsPairPlanV5 {
    PalsPairPlanV5 {
        opening_id: o.opening_id.clone(),
        position_command: o.position_command.clone(),
        white_order: [g0.into(), g1.into()],
        base_ms: 120_000,
        increment_ms: 1_000,
        max_plies: 256,
        seed: 1,
        concurrent_games: 1,
        restart_processes_each_game: true,
        ponder: false,
        score_adjudication: false,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PalsEvaluationStatisticsV5 {
    pub method: NativePilotStatistics,
    pub expected_pairs: usize,
    pub attempted_pairs: usize,
    pub completed_pairs: usize,
    /// Sum of the G1 half-points in its two games, indexed 0..4.
    pub counts: [u64; 5],
    pub score_rate: Option<f64>,
    /// Fixed 32 independent opening pairs; absent for every partial series.
    pub final_interval_95: Option<[f64; 2]>,
}
fn summarize_scores(
    scores: &[Option<u8>],
    expected_pairs: usize,
) -> Result<PalsEvaluationStatisticsV5, ArenaError> {
    require(
        expected_pairs == PALS_EVALUATION_PAIRS_V5 && scores.len() <= expected_pairs,
        "final statistics require the original 32-pair registration",
    )?;
    let mut counts = [0u64; 5];
    for score in scores.iter().flatten() {
        require(*score <= 4, "pair score outside 0..4 half-points")?;
        counts[usize::from(*score)] += 1;
    }
    let completed_pairs = counts.iter().sum::<u64>() as usize;
    let total = counts
        .iter()
        .enumerate()
        .map(|(i, n)| i as u64 * n)
        .sum::<u64>();
    let score_rate = (completed_pairs > 0).then(|| total as f64 / (4.0 * completed_pairs as f64));
    // New pure calculation of a previously registered method, not a claim
    // that historical pilot records already implemented this interval.
    let final_interval_95 = if scores.len() == expected_pairs && completed_pairs == expected_pairs {
        let mean = score_rate.expect("32 scored pairs");
        let radius = (40.0_f64.ln() / (2.0 * expected_pairs as f64)).sqrt();
        Some([(mean - radius).max(0.0), (mean + radius).min(1.0)])
    } else {
        None
    };
    Ok(PalsEvaluationStatisticsV5 {
        method: NativePilotStatistics::FixedPairedHoeffding95,
        expected_pairs,
        attempted_pairs: scores.len(),
        completed_pairs,
        counts,
        score_rate,
        final_interval_95,
    })
}
fn game_half_points(
    game: &PalsGameReceiptV5,
    g0: &str,
    g1: &str,
) -> Result<Option<u8>, ArenaError> {
    require(
        game.white_endpoint != game.black_endpoint
            && [&game.white_endpoint, &game.black_endpoint]
                .iter()
                .all(|id| id.as_str() == g0 || id.as_str() == g1)
            && game.base_ms == 120_000
            && game.increment_ms == 1_000
            && game.plies <= 256
            && matches!(game.rules_clock_pgn_check, PalsObservedV3::Observed { .. }),
        "final game endpoint, Fischer clock, ply or actual audit differs",
    )?;
    game.pgn.validate()?;
    if let PalsObservedV3::Observed { value, .. } = &game.rules_clock_pgn_check {
        value.validate()?;
    }
    let winner = match game.result {
        PalsResultV3::WhiteWin => Some(&game.white_endpoint),
        PalsResultV3::BlackWin => Some(&game.black_endpoint),
        PalsResultV3::Draw | PalsResultV3::Incomplete => None,
    };
    match game.termination {
        PalsTerminationV3::IllegalMove
        | PalsTerminationV3::EngineCrash
        | PalsTerminationV3::TimeForfeit => {
            let loser = if game.failed_endpoint.as_ref() == Some(&game.white_endpoint) {
                &game.white_endpoint
            } else if game.failed_endpoint.as_ref() == Some(&game.black_endpoint) {
                &game.black_endpoint
            } else {
                return Err(invalid("engine failure has no original offender"));
            };
            require(
                winner.is_some_and(|id| id != loser),
                "engine failure must remain offender loss",
            )?;
        }
        PalsTerminationV3::Checkmate => require(
            game.failed_endpoint.is_none() && winner.is_some(),
            "mate outcome differs",
        )?,
        PalsTerminationV3::RulesDraw => require(
            game.failed_endpoint.is_none() && game.result == PalsResultV3::Draw,
            "Rules draw differs",
        )?,
        PalsTerminationV3::PlyLimit => require(
            game.failed_endpoint.is_none()
                && game.plies == 256
                && game.result == PalsResultV3::Incomplete,
            "ply cutoff is unscored incomplete",
        )?,
        PalsTerminationV3::RunDeadline | PalsTerminationV3::InfrastructureFailure => {
            return Err(invalid(
                "infrastructure/deadline game cannot enter completed native pair statistics",
            ));
        }
    }
    Ok(match game.result {
        PalsResultV3::Draw => Some(1),
        PalsResultV3::Incomplete => None,
        _ => Some(if winner.is_some_and(|id| id == g1) {
            2
        } else {
            0
        }),
    })
}
fn pair_half_points(
    games: &[PalsGameReceiptV5],
    g0: &str,
    g1: &str,
) -> Result<Option<u8>, ArenaError> {
    require(
        !g0.is_empty() && !g1.is_empty() && g0 != g1 && games.len() == 2,
        "final pair requires exactly two games and distinct G0/G1 endpoints",
    )?;
    require(
        games[0].game_index == 0
            && games[1].game_index == 1
            && games[0].white_endpoint == games[1].black_endpoint
            && games[0].black_endpoint == games[1].white_endpoint,
        "final pair lacks exact ordinal and color exchange",
    )?;
    let first = game_half_points(&games[0], g0, g1)?;
    let second = game_half_points(&games[1], g0, g1)?;
    Ok(first.zip(second).map(|(a, b)| a + b))
}

/// Retains original native custody rather than a reconstructed JSON success.
pub struct PalsEvaluationPairArchiveV5 {
    opening_id: String,
    native: ClosedNativePairArchive<LockedPalsArenaLaunchV5>,
    normalized: PalsNormalizedPairReceiptV5,
    normalized_artifact: ArtifactRef,
    candidate_half_points: Option<u8>,
}
impl PalsEvaluationPairArchiveV5 {
    pub fn opening_id(&self) -> &str {
        &self.opening_id
    }
    pub fn native(&self) -> &ClosedNativePairArchive<LockedPalsArenaLaunchV5> {
        &self.native
    }
    pub fn normalized(&self) -> &PalsNormalizedPairReceiptV5 {
        &self.normalized
    }
    pub fn normalized_artifact(&self) -> &ArtifactRef {
        &self.normalized_artifact
    }
    pub fn candidate_half_points(&self) -> Option<u8> {
        self.candidate_half_points
    }
}

pub enum PalsEvaluationFailureOwnerV5 {
    Preparation(Box<crate::NativePreparationFailure>),
    Execution {
        original: Box<PalsAdaptivePairFailureV5>,
        diagnostic: Option<PalsNormalizedFailedPairReceiptV5>,
        diagnostic_artifact: Option<ArtifactRef>,
        diagnostic_error: Option<String>,
    },
    Normalization {
        cause: ArenaError,
        original: NativePairOutput<LockedPalsArenaLaunchV5>,
    },
    Closure {
        original: NativePairOutput<LockedPalsArenaLaunchV5>,
    },
    Admission(ArenaError),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PalsEvaluationSeriesStatusV5 {
    Completed,
    Incomplete,
    Cancelled,
    InfrastructureFailure,
}

#[derive(Clone, Debug, Serialize)]
pub struct PalsEvaluationSeriesReportV5 {
    pub domain: String,
    pub registration_sha256: String,
    pub status: PalsEvaluationSeriesStatusV5,
    pub candidate_endpoint: String,
    /// Preparation calls, including attempts rejected before child dispatch.
    pub preparation_attempts: usize,
    /// Original pair entry calls, not a claim that every child was spawned.
    pub native_dispatch_attempts: usize,
    pub pair_opening_ids: Vec<String>,
    pub pair_half_points: Vec<Option<u8>>,
    pub normalized_receipts: Vec<ArtifactRef>,
    pub statistics: PalsEvaluationStatisticsV5,
    pub candidate_wins: u64,
    pub candidate_draws: u64,
    pub candidate_losses: u64,
    pub incomplete_games: u64,
    pub engine_failure_games: u64,
    pub infrastructure_failure: Option<String>,
    pub training_executed_by_evaluation: bool,
    pub notes: Vec<String>,
}

#[cfg(feature = "pals-collection-onnx")]
pub struct PalsEvaluationSeriesV5 {
    pipeline: Arc<crate::pals_collect_v3::VerifiedGenerationPipelineV3>,
    registration: Arc<PalsEvaluationOpeningRegistrationV5>,
    launches: Vec<LockedPalsArenaLaunchV5>,
    pairs: Vec<PalsEvaluationPairArchiveV5>,
    failure: Option<PalsEvaluationFailureOwnerV5>,
    report: PalsEvaluationSeriesReportV5,
}
#[cfg(feature = "pals-collection-onnx")]
impl PalsEvaluationSeriesV5 {
    pub fn pipeline(&self) -> &Arc<crate::pals_collect_v3::VerifiedGenerationPipelineV3> {
        &self.pipeline
    }
    pub fn registration(&self) -> &Arc<PalsEvaluationOpeningRegistrationV5> {
        &self.registration
    }
    pub fn pairs(&self) -> &[PalsEvaluationPairArchiveV5] {
        &self.pairs
    }
    pub fn failure(&self) -> Option<&PalsEvaluationFailureOwnerV5> {
        self.failure.as_ref()
    }
    pub fn report(&self) -> &PalsEvaluationSeriesReportV5 {
        &self.report
    }
    /// Includes the failed pair registration, if any. This is declaration
    /// custody; neither an automatic retry nor fresh execution authority.
    pub fn remaining_registrations(&self) -> &[LockedPalsArenaLaunchV5] {
        &self.launches[self.pairs.len()..]
    }
}

#[cfg(feature = "pals-collection-onnx")]
pub(super) fn validate_series(
    registration: &PalsEvaluationOpeningRegistrationV5,
    g1: &crate::pals_collect_v3::VerifiedGenerationExportV3,
    launches: &[LockedPalsArenaLaunchV5],
    source_root: &Path,
) -> Result<String, ArenaError> {
    use crate::pals_collect_v3::GenerationExportKindV3;
    require(
        g1.kind() == GenerationExportKindV3::G1ActualGeneration,
        "final evaluation requires the original actual G1 export, not frozen P/C or diagnostic assets",
    )?;
    require(
        launches.len() == PALS_EVALUATION_PAIRS_V5 && source_root.is_absolute(),
        "final native cohort count or source root differs",
    )?;
    let first = &launches[0].input;
    let baseline = &first.semantic_lock.manifest;
    let g1_id = baseline
        .engines
        .get(1)
        .ok_or_else(|| invalid("G1 endpoint missing"))?
        .id
        .clone();
    require(
        g1_id != registration.g0_deployment.id,
        "G0/G1 endpoint identity aliases",
    )?;
    let mut run_ids = BTreeSet::new();
    let mut windows = BTreeSet::new();
    for (index, launch) in launches.iter().enumerate() {
        let input = &launch.input;
        input.validate()?;
        let manifest = &input.semantic_lock.manifest;
        let mut window = manifest.window.clone();
        window.window_id = baseline.window.window_id.clone();
        require(
            manifest.purpose == PalsRunPurposeV5::EvaluationPair
                && manifest.comparison == PalsComparisonV5::InternalWeights
                && manifest.engines[0] == registration.g0_deployment
                && manifest.engines == baseline.engines
                && manifest.source_pins == baseline.source_pins
                && manifest.rules_profile == baseline.rules_profile
                && manifest.generation == baseline.generation
                && !manifest.training_authorized
                && run_ids.insert(&manifest.run_id)
                && windows.insert(&manifest.window.window_id)
                && manifest.window.wall_time_max_ms <= 20 * 60 * 1000
                && manifest.window.cleanup_max_ms == 30_000
                && window == baseline.window
                && input.runner == first.runner
                && input.budget == first.budget
                && input.storage == first.storage
                && input.source_files == first.source_files,
            "final cohort changed model/options/source/resources/clock budget or reused run identity",
        )?;
        let opening = &registration.openings[index];
        require(
            input.opening_artifact == opening.opening_artifact
                && manifest.pair.as_ref()
                    == Some(&registration_pair(
                        opening,
                        &registration.g0_deployment.id,
                        &g1_id,
                    )),
            "pair order/opening/Fischer/seed differs from the precollection registration",
        )?;
        let endpoint = &manifest.engines[1];
        require(
            endpoint.mode == PalsEngineModeV5::Deployment
                && endpoint.model.profile == "virtual_review_v1"
                && endpoint.model.artifact_domain == PalsArtifactDomainV5::RealGeneration
                && endpoint.model.epoch == g1.model_epoch()
                && endpoint.cpu.cpu_t == PalsCpuSelectionV5::Disabled
                && endpoint.cpu.cpu_r == PalsCpuSelectionV5::Own
                && endpoint.binary == registration.g0_deployment.binary
                && endpoint.source_commit == registration.g0_deployment.source_commit
                && !endpoint.v_generate
                && !endpoint.v_load
                && !endpoint.v_call,
            "actual G1 final deployment/profile/epoch/own CPU_R/V absence differs",
        )?;
        for (endpoint, original) in input.endpoints.iter().zip(&first.endpoints) {
            let mut config = endpoint.configuration.clone();
            config.semantic_lock_sha256 = original.configuration.semantic_lock_sha256.clone();
            require(
                config == original.configuration
                    && endpoint.environment == original.environment
                    && endpoint.cuda_bundle == original.cuda_bundle
                    && endpoint.version == original.version
                    && endpoint.expected_uci_name == original.expected_uci_name
                    && endpoint.handshake_max_ms == original.handshake_max_ms
                    && endpoint.startup_snapshot_path == original.startup_snapshot_path,
                "series changed native startup/model assets/options/readiness/environment between pairs",
            )?;
        }
        #[cfg(target_os = "linux")]
        validate_g1_original_files(g1, &input.endpoints[1].configuration, source_root)?;
    }
    Ok(g1_id)
}

#[cfg(all(feature = "pals-collection-onnx", target_os = "linux"))]
fn validate_g1_original_files(
    g1: &crate::pals_collect_v3::VerifiedGenerationExportV3,
    config: &PalsAdaptiveStartupConfigV5,
    source_root: &Path,
) -> Result<(), ArenaError> {
    let checkpoint = &g1.semantic_receipt()["checkpoint"];
    require(
        checkpoint["sha256"] == config.endpoint.model.weights.sha256
            && checkpoint["bytes"] == config.endpoint.model.weights.bytes,
        "G1 endpoint weights do not match the original exact exported checkpoint",
    )?;
    require(
        g1.trainer().checkpoint_files().any(|(pin, _file)| {
            pin.path == source_root.join(&config.endpoint.model.weights.path)
                && pin.sha256 == config.endpoint.model.weights.sha256
                && pin.bytes == config.endpoint.model.weights.bytes
        }),
        "G1 weights do not join the original final trainer checkpoint descriptor",
    )?;
    let provenance = config
        .endpoint
        .model
        .provenance
        .as_ref()
        .ok_or_else(|| invalid("G1 final endpoint lacks actual provenance"))?;
    let original_provenance = g1.semantic_receipt_file_pin();
    require(
        original_provenance.path == source_root.join(&provenance.path)
            && original_provenance.sha256 == provenance.sha256
            && original_provenance.bytes == provenance.bytes,
        "G1 provenance does not join the original semantic export receipt descriptor",
    )?;
    let expected = &g1.semantic_receipt()["deployment"];
    let manifest = config.asset(PalsStartupAssetPurposeV5::ExportManifest)?;
    require(
        expected["validator_present"] == false
            && expected["manifest"]["sha256"] == manifest.artifact.sha256
            && expected["manifest"]["bytes"] == manifest.artifact.bytes,
        "G1 endpoint is not the original V-free deployment export",
    )?;
    for purpose in [
        PalsStartupAssetPurposeV5::ExportManifest,
        PalsStartupAssetPurposeV5::PublicGraph,
        PalsStartupAssetPurposeV5::SharedPcGraph,
    ] {
        let artifact = &config.asset(purpose)?.artifact;
        require(
            g1.held_files().any(|(pin, _file)| {
                pin.path == source_root.join(&artifact.path)
                    && pin.sha256 == artifact.sha256
                    && pin.bytes == artifact.bytes
            }),
            "G1 deployment artifact does not join the original held export file",
        )?;
    }
    require(
        !config.assets.iter().any(|asset| {
            matches!(
                asset.purpose,
                PalsStartupAssetPurposeV5::ValidatorGraph
                    | PalsStartupAssetPurposeV5::ProposerGraph
                    | PalsStartupAssetPurposeV5::CriticGraph
            )
        }),
        "compact Deployment must retain V-free shared P/C and cannot load a V or separate legacy graph",
    )?;
    // Native preparation subsequently verifies source bytes/inode snapshots,
    // actual trained provenance, numerical acceptance and V-free loaded sessions.
    Ok(())
}

/// Explicit actual-run entry. It accepts the original complete pipeline owner
/// that retained the opening registration before collection, not a JSON status.
/// Only one pair is prepared/running at a time. No retries or replacements.
#[cfg(feature = "pals-collection-onnx")]
pub fn run_pals_evaluation_series_v5(
    pipeline: Arc<crate::pals_collect_v3::VerifiedGenerationPipelineV3>,
    launches: Vec<LockedPalsArenaLaunchV5>,
    source_root: &Path,
    output_root: &Path,
    attempt_prefix: &str,
    cancel: &AtomicBool,
) -> PalsEvaluationSeriesV5 {
    run_pals_evaluation_series_with_storage_policy_v5(
        pipeline,
        launches,
        source_root,
        output_root,
        attempt_prefix,
        cancel,
        PalsOwnedStoragePolicy::StrictV5,
    )
}

#[cfg(feature = "pals-collection-onnx")]
pub fn run_pals_evaluation_series_with_storage_policy_v5(
    pipeline: Arc<crate::pals_collect_v3::VerifiedGenerationPipelineV3>,
    launches: Vec<LockedPalsArenaLaunchV5>,
    source_root: &Path,
    output_root: &Path,
    attempt_prefix: &str,
    cancel: &AtomicBool,
    storage_policy: PalsOwnedStoragePolicy,
) -> PalsEvaluationSeriesV5 {
    let registration = Arc::clone(pipeline.evaluation_openings());
    let mut series = PalsEvaluationSeriesV5 {
        pipeline, registration: Arc::clone(&registration), launches, pairs: Vec::with_capacity(32), failure: None,
        report: PalsEvaluationSeriesReportV5 {
            domain: PALS_EVALUATION_SERIES_V5_DOMAIN.into(), registration_sha256: registration.sha256.clone(),
            status: PalsEvaluationSeriesStatusV5::Incomplete, candidate_endpoint: String::new(),
            preparation_attempts: 0, native_dispatch_attempts: 0,
            pair_opening_ids: vec![], pair_half_points: vec![], normalized_receipts: vec![],
            statistics: summarize_scores(&[], 32).expect("fixed empty statistics"),
            candidate_wins: 0, candidate_draws: 0, candidate_losses: 0, incomplete_games: 0,
            engine_failure_games: 0, infrastructure_failure: None, training_executed_by_evaluation: false,
            notes: vec!["Only original closed native pairs grant this projection; JSON/CSV can neither admit a pair nor restore original custody".into(),
                "The 95% interval is newly calculated using predeclared fixed paired Hoeffding; partial intervals and score-based selection/stopping are forbidden".into(),
                "Full cohort effect compares G0/G1 weights of the same compact V-free own-CPU_R engine; no Elo or promotion is implied".into()],
        },
    };
    if storage_policy == PalsOwnedStoragePolicy::AdvisoryV1 {
        series.report.domain = "rz-pals-final-evaluation-series-storage-advisory-v1/1".into();
        series.report.notes.push("Explicit storage advisory policy; each native pair records its advisory-v1 policy in its source-verification artifact. Output-root usage is not measured by the pair; global storage observation is independent.".into());
    }
    let admission = (|| -> Result<String, ArenaError> {
        require(
            series_identifier(attempt_prefix),
            "invalid serial attempt prefix",
        )?;
        let original = series.pipeline.evaluation_launches();
        require(
            Arc::ptr_eq(original.numerical(), series.pipeline.g1_numerical())
                && Arc::ptr_eq(original.prepared().g1(), series.pipeline.g1_export())
                && Arc::ptr_eq(original.prepared().original().registration(), &registration)
                && original.source_root() == source_root
                && original.prepared().managed_root() == output_root
                && series.pipeline.evaluation_output_root() == output_root
                && series.pipeline.evaluation_attempt_prefix() == attempt_prefix
                && series.launches.len() == original.launches().len()
                && series
                    .launches
                    .iter()
                    .zip(original.launches())
                    .all(|(supplied, owned)| {
                        supplied.sha256() == owned.sha256() && supplied.input() == owned.input()
                    }),
            "final native consumer requires the original pipeline's G1 numerical/template/launch owners",
        )?;
        let candidate_id = validate_series(
            &registration,
            series.pipeline.g1_export(),
            &series.launches,
            source_root,
        )?;
        original.claim_serial_evaluation()?;
        Ok(candidate_id)
    })();
    match admission {
        Ok(id) => series.report.candidate_endpoint = id,
        Err(error) => {
            stop_series(
                &mut series,
                error.to_string(),
                PalsEvaluationFailureOwnerV5::Admission(error),
            );
            return series;
        }
    }
    for index in 0..PALS_EVALUATION_PAIRS_V5 {
        if cancel.load(Ordering::Acquire) {
            series.report.status = PalsEvaluationSeriesStatusV5::Cancelled;
            break;
        }
        let attempt = format!("{attempt_prefix}-pair-{index:02}");
        series.report.preparation_attempts += 1;
        let prepared = match series.pairs.last() {
            Some(previous) => prepare_pals_pair_launch_after_closed_with_storage_policy(
                &series.launches[index],
                &previous.native,
                source_root,
                output_root,
                &attempt,
                storage_policy,
            ),
            None => prepare_pals_pair_launch_with_storage_policy(
                &series.launches[index],
                source_root,
                output_root,
                &attempt,
                storage_policy,
            ),
        };
        let owner = match prepared {
            Ok(owner) => owner,
            Err(error) => {
                let reason = error.to_string();
                stop_series(
                    &mut series,
                    reason,
                    PalsEvaluationFailureOwnerV5::Preparation(error),
                );
                break;
            }
        };
        series.report.native_dispatch_attempts += 1;
        let output = match run_pals_pair_with_storage_policy(owner, Some(cancel)) {
            Ok(output) => output,
            Err(original) => {
                preserve_unscored_attempt(
                    &mut series.report,
                    &registration.openings[index].opening_id,
                );
                let diagnostic = save_pals_failed_run_receipt_v5(&original);
                let (diagnostic, diagnostic_artifact, diagnostic_error) = match diagnostic {
                    Ok((r, a)) => (Some(r), a, None),
                    Err(e) => (None, None, Some(e.to_string())),
                };
                let reason = original.to_string();
                stop_series(
                    &mut series,
                    reason,
                    PalsEvaluationFailureOwnerV5::Execution {
                        original,
                        diagnostic,
                        diagnostic_artifact,
                        diagnostic_error,
                    },
                );
                break;
            }
        };
        let normalized = build_pals_run_receipt_v5(&output).and_then(|(normalized, artifact)| {
            let score = pair_half_points(
                &normalized.normalized.games,
                &registration.g0_deployment.id,
                &series.report.candidate_endpoint,
            )?;
            require(
                normalized.normalized.failures.is_empty()
                    && (normalized.normalized.accepted || score.is_none()),
                "normalized native pair exceeded a registered execution/cleanup/output condition",
            )?;
            Ok((normalized, artifact, score))
        });
        let (normalized, normalized_artifact, score) = match normalized {
            Ok(value) => value,
            Err(cause) => {
                preserve_unscored_attempt(
                    &mut series.report,
                    &registration.openings[index].opening_id,
                );
                let reason = cause.to_string();
                stop_series(
                    &mut series,
                    reason,
                    PalsEvaluationFailureOwnerV5::Normalization {
                        cause,
                        original: output,
                    },
                );
                break;
            }
        };
        let native = match output.into_closed_archive() {
            Ok(archive) => archive,
            Err(original) => {
                preserve_unscored_attempt(
                    &mut series.report,
                    &registration.openings[index].opening_id,
                );
                stop_series(
                    &mut series,
                    "native closure uncertain; original lease retained".into(),
                    PalsEvaluationFailureOwnerV5::Closure { original },
                );
                break;
            }
        };
        let opening_id = registration.openings[index].opening_id.clone();
        count_original_games(&mut series.report, &normalized.normalized.games);
        series.report.pair_opening_ids.push(opening_id.clone());
        series.report.pair_half_points.push(score);
        series
            .report
            .normalized_receipts
            .push(normalized_artifact.clone());
        series.pairs.push(PalsEvaluationPairArchiveV5 {
            opening_id,
            native,
            normalized,
            normalized_artifact,
            candidate_half_points: score,
        });
        series.report.statistics = summarize_scores(&series.report.pair_half_points, 32)
            .expect("scores validated before archive move");
    }
    if series.failure.is_none() && series.pairs.len() == 32 {
        series.report.status = if series.report.statistics.completed_pairs == 32 {
            PalsEvaluationSeriesStatusV5::Completed
        } else {
            PalsEvaluationSeriesStatusV5::Incomplete
        };
    }
    series
}
#[cfg(feature = "pals-collection-onnx")]
fn stop_series(
    series: &mut PalsEvaluationSeriesV5,
    reason: String,
    failure: PalsEvaluationFailureOwnerV5,
) {
    series.report.status = PalsEvaluationSeriesStatusV5::InfrastructureFailure;
    series.report.infrastructure_failure = Some(reason);
    series.failure = Some(failure);
}
#[cfg(feature = "pals-collection-onnx")]
fn preserve_unscored_attempt(report: &mut PalsEvaluationSeriesReportV5, opening_id: &str) {
    report.pair_opening_ids.push(opening_id.into());
    report.pair_half_points.push(None);
    report.statistics = summarize_scores(&report.pair_half_points, 32)
        .expect("original bounded pair attempt without an invented score");
}
#[cfg(feature = "pals-collection-onnx")]
fn count_original_games(report: &mut PalsEvaluationSeriesReportV5, games: &[PalsGameReceiptV5]) {
    for game in games {
        match game.result {
            PalsResultV3::Incomplete => report.incomplete_games += 1,
            PalsResultV3::Draw => report.candidate_draws += 1,
            PalsResultV3::WhiteWin if game.white_endpoint == report.candidate_endpoint => {
                report.candidate_wins += 1
            }
            PalsResultV3::BlackWin if game.black_endpoint == report.candidate_endpoint => {
                report.candidate_wins += 1
            }
            _ => report.candidate_losses += 1,
        }
        if matches!(
            game.termination,
            PalsTerminationV3::IllegalMove
                | PalsTerminationV3::EngineCrash
                | PalsTerminationV3::TimeForfeit
        ) {
            report.engine_failure_games += 1;
        }
    }
}

#[cfg(test)]
#[path = "evaluation_series_tests.rs"]
mod tests;
