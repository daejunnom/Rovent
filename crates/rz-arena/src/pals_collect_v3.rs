//! Actual adaptive collection. Inputs originate in submitted native tickets;
//! later teacher, Rules and action evidence never changes the frozen input.
//! This owner does not run an engine, optimizer, GPU or cloud API implicitly.
use crate::ArenaError;
use rz_experiments::pals_data_v3::*;
use rz_position::contracts::ContractPosition;
use rz_position::{BoardMove, Color, Position};
use rz_search::pals::store::Move16;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

mod writer;
pub use writer::*;
mod writer_worker;
pub use writer_worker::*;
mod actions;
pub use actions::*;
mod collection_pgn;
pub use collection_pgn::*;
#[cfg(feature = "pals-collection")]
mod capture;
#[cfg(feature = "pals-collection")]
pub(crate) use capture::capture_submitted_review_common_fields;
pub(crate) use capture::generation_transposition_sha256;
#[cfg(feature = "pals-collection")]
pub use capture::*;
#[cfg(feature = "pals-collection")]
mod action_driver;
#[cfg(feature = "pals-collection")]
pub use action_driver::*;
#[cfg(feature = "pals-collection-onnx")]
mod driver;
#[cfg(feature = "pals-collection-onnx")]
pub use driver::*;
#[cfg(feature = "pals-collection-onnx")]
mod teacher;
#[cfg(feature = "pals-collection-onnx")]
pub use teacher::*;
#[cfg(feature = "pals-collection-onnx")]
mod actual_selfplay;
#[cfg(feature = "pals-collection-onnx")]
pub use actual_selfplay::*;
#[cfg(feature = "pals-collection-onnx")]
mod actual_actions;
#[cfg(feature = "pals-collection-onnx")]
pub use actual_actions::*;
#[cfg(feature = "pals-collection-onnx")]
mod frozen_intake;
#[cfg(feature = "pals-collection-onnx")]
pub use frozen_intake::*;
#[cfg(feature = "pals-collection-onnx")]
mod game_process;
#[cfg(feature = "pals-collection-onnx")]
pub use game_process::*;
#[cfg(feature = "pals-collection-onnx")]
mod diagnostic_game;
#[cfg(feature = "pals-collection-onnx")]
pub use diagnostic_game::*;
#[cfg(feature = "pals-collection-onnx")]
mod generation_series;
#[cfg(feature = "pals-collection-onnx")]
pub use generation_series::*;
#[cfg(feature = "pals-collection-onnx")]
pub(crate) mod selfplay_cli;
#[cfg(feature = "pals-collection-onnx")]
pub use selfplay_cli::*;
#[cfg(feature = "pals-collection-onnx")]
mod action_cli;
#[cfg(feature = "pals-collection-onnx")]
pub use action_cli::*;
#[cfg(feature = "pals-collection-onnx")]
pub(crate) mod training_process;
#[cfg(feature = "pals-collection-onnx")]
pub use training_process::*;
#[cfg(feature = "pals-collection-onnx")]
mod training_export_process;
#[cfg(feature = "pals-collection-onnx")]
pub use training_export_process::*;
#[cfg(feature = "pals-collection-onnx")]
mod frozen_startup;
#[cfg(feature = "pals-collection-onnx")]
pub use frozen_startup::*;
#[cfg(feature = "pals-collection-onnx")]
mod numerical_process;
#[cfg(feature = "pals-collection-onnx")]
pub use numerical_process::*;
#[cfg(all(feature = "pals-collection-onnx", target_os = "linux"))]
mod g0_cuda_artifact_tree;
#[cfg(all(test, feature = "pals-collection-onnx", target_os = "linux"))]
mod g0_cuda_artifact_tree_tests;
#[cfg(feature = "pals-collection-onnx")]
mod g0_numerical_process;
#[cfg(feature = "pals-collection-onnx")]
pub use g0_numerical_process::*;
#[cfg(feature = "pals-collection-onnx")]
mod g0_bootstrap_process;
#[cfg(feature = "pals-collection-onnx")]
pub use g0_bootstrap_process::*;
#[cfg(feature = "pals-collection-onnx")]
mod g0_deployment_numerical_process;
#[cfg(feature = "pals-collection-onnx")]
pub use g0_deployment_numerical_process::*;
#[cfg(feature = "pals-collection-onnx")]
mod generation_pipeline;
#[cfg(feature = "pals-collection-onnx")]
pub use generation_pipeline::*;

pub const GENERATION_COLLECTOR_V3: &str = "rz-pals-actual-generation-collector/3";
pub const GENERATION_OUTPUT_CAP: u64 = 18 * 1024 * 1024 * 1024;
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

/// Collector event order, independent of the Store's append handles, request
/// sequences and model revisions. The single collector owner advances this
/// for actual input publication and later label publication.
#[derive(Default, Debug)]
pub struct GenerationChronologyV3 {
    last: u64,
}
impl GenerationChronologyV3 {
    pub fn next_event_sequence(&mut self) -> Result<u64, ArenaError> {
        self.last = self
            .last
            .checked_add(1)
            .ok_or_else(|| invalid("collector sequence exhausted"))?;
        Ok(self.last)
    }
    pub fn last_event_sequence(&self) -> u64 {
        self.last
    }
    pub(super) fn label_after(&mut self, input: &GenerationInputV3) -> Result<u64, ArenaError> {
        check(
            input.capture_sequence > 0 && input.capture_sequence <= self.last,
            "label requires original input published by this collector clock",
        )?;
        self.next_event_sequence()
    }
}
fn invalid(message: impl Into<String>) -> ArenaError {
    ArenaError::Invalid(message.into())
}
fn check(ok: bool, message: &str) -> Result<(), ArenaError> {
    if ok { Ok(()) } else { Err(invalid(message)) }
}
fn pack(moves: &[BoardMove]) -> Result<Vec<u16>, ArenaError> {
    moves
        .iter()
        .map(|m| {
            Move16::pack(*m)
                .map(|m| m.bits())
                .map_err(|e| invalid(e.to_string()))
        })
        .collect()
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationOpeningV3 {
    pub id: String,
    pub initial_fen: Option<String>,
    pub moves: Vec<String>,
    pub split: GenerationSplitV3,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelfplayCollectionPlanV3 {
    pub run_id: String,
    pub openings: Vec<GenerationOpeningV3>,
    pub games: u16,
    pub base_time_ms: u64,
    pub increment_ms: u64,
    pub max_plies: u16,
    pub concurrent_games: u8,
    pub seed: u64,
    pub sampling_seed: u64,
    pub process_restart_each_game: bool,
    pub ponder: bool,
    pub maximum_output_bytes: u64,
}
impl SelfplayCollectionPlanV3 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            generation_identifier(&self.run_id)
                && self.openings.len() == 64
                && self.games == 128
                && self.base_time_ms == 120_000
                && self.increment_ms == 1_000
                && self.max_plies == 512
                && self.concurrent_games == 1
                && self.seed == 29
                && self.sampling_seed == 29
                && self.process_restart_each_game
                && !self.ponder
                && self.maximum_output_bytes > 0
                && self.maximum_output_bytes <= GENERATION_OUTPUT_CAP,
            "first generation collection requires the registered finite 128-game paired profile",
        )?;
        let mut ids = BTreeSet::new();
        let mut counts = BTreeMap::new();
        for opening in &self.openings {
            check(
                generation_identifier(&opening.id)
                    && ids.insert(&opening.id)
                    && opening.moves.len() <= 256
                    && opening
                        .initial_fen
                        .as_ref()
                        .is_none_or(|fen| fen.len() <= 512 && !fen.chars().any(char::is_control)),
                "collection opening identity/size",
            )?;
            let mut position = match &opening.initial_fen {
                Some(fen) => Position::from_fen(fen).map_err(|e| invalid(e.to_string()))?,
                None => Position::startpos(),
            };
            for movement in &opening.moves {
                position
                    .make_move(BoardMove::from_uci(movement).map_err(|e| invalid(e.to_string()))?)
                    .map_err(|e| invalid(e.to_string()))?;
            }
            *counts.entry(opening.split).or_insert(0usize) += 1;
        }
        check(
            counts
                == BTreeMap::from([
                    (GenerationSplitV3::Train, 48),
                    (GenerationSplitV3::Validation, 8),
                    (GenerationSplitV3::Holdout, 8),
                ]),
            "opening assignment must be 48/8/8 before outcomes",
        )
    }
    pub fn opening_splits(&self) -> BTreeMap<String, GenerationSplitV3> {
        self.openings
            .iter()
            .map(|o| (o.id.clone(), o.split))
            .collect()
    }
    pub fn split_sha256(&self) -> Result<String, ArenaError> {
        Ok(generation_digest(
            "rz-pals-opening-split/1",
            &self.opening_splits(),
        )?)
    }
    /// A paired ordinal swaps engines' colors; it never reverses the board.
    pub fn games(&self) -> Result<Vec<GenerationGameAssignmentV3>, ArenaError> {
        self.validate()?;
        Ok(self
            .openings
            .iter()
            .enumerate()
            .flat_map(|(pair, o)| {
                [false, true].map(move |swap| GenerationGameAssignmentV3 {
                    game_id: format!("{}-pair-{}-{}", self.run_id, pair + 1, u8::from(swap) + 1),
                    opening_id: o.id.clone(),
                    pair: pair as u16 + 1,
                    engine_a_white: !swap,
                    split: o.split,
                })
            })
            .collect())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationGameAssignmentV3 {
    pub game_id: String,
    pub opening_id: String,
    pub pair: u16,
    pub engine_a_white: bool,
    pub split: GenerationSplitV3,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionOutcomeCollectionPlanV3 {
    pub run_id: String,
    pub desired_contexts: u32,
    pub maximum_contexts: u32,
    pub frozen_pc_epoch: u64,
    pub frozen_pc_weights_sha256: String,
    pub maximum_output_bytes: u64,
}
impl ActionOutcomeCollectionPlanV3 {
    pub fn validate(&self) -> Result<(), ArenaError> {
        check(
            generation_identifier(&self.run_id)
                && self.desired_contexts == 2048
                && self.maximum_contexts == 8192
                && self.frozen_pc_epoch > 0
                && generation_sha(&self.frozen_pc_weights_sha256)
                && self.maximum_output_bytes > 0
                && self.maximum_output_bytes <= GENERATION_OUTPUT_CAP,
            "V collection requires registered frozen P/C and bounded real action contexts",
        )
    }
}

/// Typed actual Rules owner; replaying a virtual endpoint cannot issue this
/// trajectory. Only moves actually selected by the full PALS engine advance it.
pub struct ActualGameTrajectoryV3 {
    game_id: String,
    position: Position,
    actual_moves: Vec<BoardMove>,
    anchors: Vec<GenerationRulesAnchorV3>,
    played_plies: usize,
    finished: bool,
}
impl ActualGameTrajectoryV3 {
    pub fn begin(game_id: String, initial: Position) -> Result<Self, ArenaError> {
        check(generation_identifier(&game_id), "actual game identity")?;
        let anchor = exact_rules_anchor(&initial)?;
        let actual_moves = initial
            .snapshot()
            .uci_replay(16_384)
            .map_err(|e| invalid(e.to_string()))?
            .moves;
        Ok(Self {
            game_id,
            position: initial,
            actual_moves,
            anchors: vec![anchor],
            played_plies: 0,
            finished: false,
        })
    }
    pub fn position(&self) -> &Position {
        &self.position
    }
    pub fn game_id(&self) -> &str {
        &self.game_id
    }
    pub fn actual_history(&self) -> Result<Vec<u16>, ArenaError> {
        pack(&self.actual_moves)
    }
    pub fn current_anchor(&self) -> Result<GenerationRulesAnchorV3, ArenaError> {
        exact_rules_anchor(&self.position)
    }
    pub fn play_pals_move(&mut self, movement: BoardMove) -> Result<(), ArenaError> {
        check(
            !self.finished && self.played_plies < 512,
            "actual game already terminated/ply cap",
        )?;
        self.position
            .make_move(movement)
            .map_err(|e| invalid(e.to_string()))?;
        self.actual_moves.push(movement);
        self.played_plies += 1;
        self.anchors.push(exact_rules_anchor(&self.position)?);
        Ok(())
    }
    pub fn terminate_rules(
        &mut self,
    ) -> Result<(GenerationTrajectoryAuthorityV3, serde_json::Value), ArenaError> {
        use rz_position::{PlayStatus, TerminalReason};
        check(!self.finished, "duplicate actual game result")?;
        let classification = self
            .position
            .classify_position()
            .map_err(|e| invalid(e.to_string()))?;
        let PlayStatus::Terminal { reason, winner } = classification.play_status else {
            return Err(invalid(
                "unfinished/infrastructure game is not a Rules WDL target",
            ));
        };
        let ending = match reason {
            TerminalReason::Checkmate => GenerationRulesEndingV3::Checkmate,
            TerminalReason::Stalemate => GenerationRulesEndingV3::Stalemate,
            TerminalReason::DeadPosition => GenerationRulesEndingV3::DeadPosition,
            TerminalReason::FivefoldRepetition => GenerationRulesEndingV3::Repetition,
            TerminalReason::SeventyFiveMove => GenerationRulesEndingV3::SeventyFiveMove,
        };
        let outcome = match winner {
            Some(Color::White) => GenerationOutcomeV3::WhiteWin,
            Some(Color::Black) => GenerationOutcomeV3::BlackWin,
            None => GenerationOutcomeV3::Draw,
        };
        let receipt = serde_json::json!({"schema":"rz-pals-rules-outcome-evidence/1","game_id":self.game_id,"ending":ending,"outcome":outcome,"rules_terminal":true});
        let authority = GenerationTrajectoryAuthorityV3 {
            game_id: self.game_id.clone(),
            ending,
            outcome,
            anchors: self.anchors.clone(),
            result_evidence_sha256: generation_digest(
                "rz-pals-rules-outcome-evidence/1",
                &receipt,
            )?,
        };
        authority.validate()?;
        self.finished = true;
        Ok((authority, receipt))
    }
    /// Current-position draw claims use Rules availability, never a runner's
    /// repetition string or teacher score. Intended-move claims need their own
    /// verified resulting trajectory and cannot be guessed here.
    pub fn claim_rules_draw(
        &mut self,
        reason: rz_position::ClaimReason,
    ) -> Result<(GenerationTrajectoryAuthorityV3, serde_json::Value), ArenaError> {
        use rz_position::{Availability, ClaimEvidence};
        check(!self.finished, "duplicate actual game result")?;
        let classification = self
            .position
            .classify_position()
            .map_err(|e| invalid(e.to_string()))?;
        check(
            classification.claim_availability.iter().any(|claim| {
                claim.reason == reason
                    && claim.evidence == ClaimEvidence::CurrentPosition
                    && claim.availability == Availability::Available
            }),
            "Rules draw claim is unavailable/unknown",
        )?;
        let ending = match reason {
            rz_position::ClaimReason::ThreefoldRepetition => GenerationRulesEndingV3::Repetition,
            rz_position::ClaimReason::FiftyMove => GenerationRulesEndingV3::FiftyMove,
        };
        let outcome = GenerationOutcomeV3::Draw;
        let receipt = serde_json::json!({"schema":"rz-pals-rules-outcome-evidence/1","game_id":self.game_id,"ending":ending,"outcome":outcome,"rules_terminal":true});
        let authority = GenerationTrajectoryAuthorityV3 {
            game_id: self.game_id.clone(),
            ending,
            outcome,
            anchors: self.anchors.clone(),
            result_evidence_sha256: generation_digest(
                "rz-pals-rules-outcome-evidence/1",
                &receipt,
            )?,
        };
        authority.validate()?;
        self.finished = true;
        Ok((authority, receipt))
    }
}

/// State identity is supplied by Rules, not synthesized from FEN. History is a
/// separate audit seal and is never appended to the review tensor.
pub fn exact_rules_anchor(position: &Position) -> Result<GenerationRulesAnchorV3, ArenaError> {
    use sha2::{Digest, Sha256};
    let owner = NEXT_OWNER
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
        .map_err(|_| invalid("generation Rules owner exhausted"))?;
    let state = ContractPosition::new(rz_contracts::OwnerId(owner), position.clone()).export()?;
    let mut history = Sha256::new();
    history.update(b"rz-pals-generation-rules-history/1\0");
    history.update(
        format!(
            "{:?};{:?}",
            position.history_completeness(),
            position.history_origin()
        )
        .as_bytes(),
    );
    history.update((position.known_history_len() as u64).to_le_bytes());
    for frame in position.snapshot().known_history() {
        let fen = frame.to_fen();
        history.update((fen.len() as u64).to_le_bytes());
        history.update(fen.as_bytes());
    }
    let anchor = GenerationRulesAnchorV3 {
        rules_state_sha256: hex(&state.snapshot().identity().semantic.0),
        rules_history_sha256: format!("{:x}", history.finalize()),
        board_fen: position.to_fen(),
        white_to_move: position.side_to_move() == Color::White,
    };
    anchor.validate()?;
    Ok(anchor)
}

/// Attaches WDL only to an exact actual trajectory root, not to an evaluated
/// virtual continuation. Missing/unfinished/failure outcomes remain masked.
pub fn attach_actual_wdl_v3(
    record: &mut GenerationRecordV3,
    authority: &GenerationTrajectoryAuthorityV3,
    observed_sequence: u64,
    evidence: &mut BTreeMap<String, serde_json::Value>,
) -> Result<bool, ArenaError> {
    record.validate()?;
    authority.validate()?;
    if record.input.role == GenerationRoleV3::Validator {
        return Ok(false);
    }
    check(
        authority.game_id == record.input.game_id,
        "WDL outcome belongs to another game",
    )?;
    // Only a virtual endpoint later present with its exact Rules history on
    // this actual trajectory receives WDL; unplayed endpoints remain masked.
    if !authority.anchors.contains(&record.input.rules_anchor) {
        return Ok(false);
    }
    let mut target = GenerationWdlTargetV3 {
        observed_sequence,
        question_id: record.input.question_id.clone(),
        view_sha256: record.input.view_sha256.clone(),
        raw_evidence_sha256: "0".repeat(64),
        source: "rules_selfplay_trajectory".into(),
        game_id: authority.game_id.clone(),
        anchor: record.input.rules_anchor.clone(),
        trajectory_membership_sha256: authority.sha256()?,
        rules_result_sha256: authority.result_evidence_sha256.clone(),
        ending: authority.ending,
        outcome: authority.outcome,
    };
    target.probabilities(&record.input)?;
    target.raw_evidence_sha256 =
        seal_head_evidence(&record.input_sha256, "wdl", &target, evidence)?;
    check(
        record.future_targets.wdl.is_none(),
        "duplicate WDL attachment",
    )?;
    record.future_targets.wdl = Some(target);
    record.validate()?;
    Ok(true)
}

pub fn seal_head_evidence<T: Serialize>(
    input_sha: &str,
    head: &str,
    target: &T,
    evidence: &mut BTreeMap<String, serde_json::Value>,
) -> Result<String, ArenaError> {
    let mut content = serde_json::to_value(target).map_err(|e| invalid(e.to_string()))?;
    content
        .as_object_mut()
        .ok_or_else(|| invalid("head receipt object"))?
        .remove("raw_evidence_sha256");
    let receipt = serde_json::json!({"schema":"rz-pals-generation-head-evidence/1","input_sha256":input_sha,"head":head,"target":content});
    let sha = generation_digest("rz-pals-generation-head-evidence/1", &receipt)?;
    if let Some(existing) = evidence.get(&sha) {
        check(existing == &receipt, "evidence digest collision")?;
    } else {
        evidence.insert(sha.clone(), receipt);
    }
    Ok(sha)
}

#[cfg(test)]
mod tests;
