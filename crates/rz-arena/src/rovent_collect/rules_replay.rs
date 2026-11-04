//! 새 Rovent 원자료의 Rules replay다. wire를 복원해 새 Rules 관찰 owner를
//! 만들 뿐이며 원 live native/Store/undo/학습 권한을 발급하지 않는다.
//! 현재 상위 모듈에는 등록하지 않은 source 초안이다.
use super::*;
use rz_position::{
    HistoryCompleteness, HistoryOrigin, PositionLimits, PositionSnapshot, UciReplay,
};
use serde::{Deserialize, Serialize};

pub const ROVENT_RULES_POSITION_REPLAY: &str = "rz-rovent-rules-position-replay/1";
pub const ROVENT_SUBMITTED_RULES_REPLAY: &str = "rz-rovent-submitted-rules-replay/1";
pub const ROVENT_ACTUAL_TRAJECTORY_REPLAY: &str = "rz-rovent-actual-rules-trajectory-replay/1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoventReplayOrigin {
    StartPosition,
    Fen,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoventReplayCompleteness {
    Complete,
    UnknownPrefix,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventRulesPositionReplay {
    pub schema: String,
    pub origin: RoventReplayOrigin,
    pub completeness: RoventReplayCompleteness,
    pub start_fen: String,
    pub moves: Vec<u16>,
    pub observer_revision: u64,
    pub anchor: GenerationRulesAnchorV3,
}
fn decode_move(bits: u16) -> Result<BoardMove, ArenaError> {
    let shared = rz_contracts::pals::Move16::try_from_bits(bits)
        .and_then(|value| value.decode())
        .map_err(|error| invalid(error.to_string()))?;
    BoardMove::try_from(shared).map_err(|error| invalid(error.to_string()))
}
fn apply(position: &mut Position, line: &[u16], cap: usize) -> Result<(), ArenaError> {
    check(line.len() <= cap, "Rules replay line capacity")?;
    for bits in line {
        position
            .make_move(decode_move(*bits)?)
            .map_err(|error| invalid(error.to_string()))?;
    }
    Ok(())
}
impl RoventRulesPositionReplay {
    /// Export detached preparation metadata through the original capture and
    /// checked replay codecs. This creates Rules observers only, never a live
    /// collection/native root, worker, action target or execution authority.
    pub fn metadata_from_position(position: &Position) -> Result<Self, ArenaError> {
        let metadata = Self::capture(position)?;
        metadata.restore()?;
        Ok(metadata)
    }
    /// 원 owner의 실제 snapshot에서만 live producer가 호출한다.
    /// 공개 DTO를 읽는 소비자는 이 함수 대신 restore로 무결성만 재검증한다.
    pub(super) fn capture(position: &Position) -> Result<Self, ArenaError> {
        let snapshot = position.snapshot();
        let replay = snapshot
            .uci_replay(16_384)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(Self {
            schema: ROVENT_RULES_POSITION_REPLAY.into(),
            origin: match replay.origin {
                HistoryOrigin::StartPosition => RoventReplayOrigin::StartPosition,
                HistoryOrigin::Fen => RoventReplayOrigin::Fen,
            },
            completeness: match replay.completeness {
                HistoryCompleteness::Complete => RoventReplayCompleteness::Complete,
                HistoryCompleteness::UnknownPrefix => RoventReplayCompleteness::UnknownPrefix,
            },
            start_fen: replay.start_fen,
            moves: packed(&replay.moves)?,
            observer_revision: snapshot.revision(),
            anchor: exact_rules_anchor(position)?,
        })
    }
    pub fn restore(&self) -> Result<Position, ArenaError> {
        check(
            self.schema == ROVENT_RULES_POSITION_REPLAY && self.moves.len() <= 16_384,
            "unsupported/missing Rules position replay",
        )?;
        self.anchor.validate()?;
        let trace = UciReplay {
            origin: match self.origin {
                RoventReplayOrigin::StartPosition => HistoryOrigin::StartPosition,
                RoventReplayOrigin::Fen => HistoryOrigin::Fen,
            },
            completeness: match self.completeness {
                RoventReplayCompleteness::Complete => HistoryCompleteness::Complete,
                RoventReplayCompleteness::UnknownPrefix => HistoryCompleteness::UnknownPrefix,
            },
            start_fen: self.start_fen.clone(),
            moves: self
                .moves
                .iter()
                .map(|bits| decode_move(*bits))
                .collect::<Result<_, _>>()?,
        };
        let limits = PositionLimits {
            max_history_positions: 16_385,
            ..PositionLimits::default()
        };
        let snapshot = PositionSnapshot::from_uci_replay(&trace, self.observer_revision, limits)
            .map_err(|error| invalid(error.to_string()))?;
        let position = Position::from_snapshot(&snapshot, limits)
            .map_err(|error| invalid(error.to_string()))?;
        check(
            exact_rules_anchor(&position)? == self.anchor,
            "replayed Rules anchor/history differs from original",
        )?;
        Ok(position)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventSubmittedRulesReplay {
    pub schema: String,
    pub actual_root: RoventRulesPositionReplay,
    /// 로컬 anchor 순서는 실제 제출 view/input과 같다. 실제 기보는 NN tensor에
    /// 추가하지 않으며 여기 있는 replay는 historical Rules 증거에만 사용한다.
    pub selected_anchors: Vec<RoventRulesPositionReplay>,
}
impl RoventSubmittedRulesReplay {
    pub(super) fn capture(actual_root: &Position, view: &ReviewView) -> Result<Self, ArenaError> {
        check(
            !view.anchors().is_empty() && view.anchors().len() <= 4,
            "submitted Rules anchor capacity",
        )?;
        let mut selected_anchors = Vec::with_capacity(view.anchors().len());
        for anchor in view.anchors() {
            let position = Position::from_snapshot(
                &anchor.snapshot,
                PositionLimits {
                    max_history_positions: 16_385,
                    ..PositionLimits::default()
                },
            )
            .map_err(|error| invalid(error.to_string()))?;
            selected_anchors.push(RoventRulesPositionReplay::capture(&position)?);
        }
        Ok(Self {
            schema: ROVENT_SUBMITTED_RULES_REPLAY.into(),
            actual_root: RoventRulesPositionReplay::capture(actual_root)?,
            selected_anchors,
        })
    }
    /// 원 제출 자료의 고정 query endpoint만 Rules로 복원한다. 현재 native 또는
    /// 교사 실행 권한은 원 live root cap와 별도로 대조해야 한다.
    pub(super) fn original_query_position(
        &self,
        input: &GenerationInputV3,
    ) -> Result<Position, ArenaError> {
        check(
            self.schema == ROVENT_SUBMITTED_RULES_REPLAY,
            "submitted query replay domain",
        )?;
        let slot = usize::from(input.selected_view.query.anchors[0]);
        let anchor = self
            .selected_anchors
            .get(slot)
            .ok_or_else(|| invalid("submitted query replay anchor"))?;
        let mut position = anchor.restore()?;
        apply(&mut position, &input.selected_view.query.prefix, 256)?;
        check(
            exact_rules_anchor(&position)? == input.rules_anchor,
            "submitted query Rules anchor differs",
        )?;
        Ok(position)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoventRulesCompletion {
    Automatic,
    CurrentClaim,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoventRulesClaim {
    ThreefoldRepetition,
    FiftyMove,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventActualTrajectoryReplay {
    pub schema: String,
    pub initial: RoventRulesPositionReplay,
    pub actual_selected_moves: Vec<u16>,
    pub final_position: RoventRulesPositionReplay,
    pub completion: RoventRulesCompletion,
    pub claim: Option<RoventRulesClaim>,
}

/// 원 raw replay의 실제 합법성·이력·선택 수열을 Rules로 다시 계산한다.
/// 이 결과는 데이터 검증이며 현재 native 권한이나 완료 증거가 아니다.
pub fn verify_historical_record_rules(record: &RoventTrainingRecord) -> Result<(), ArenaError> {
    let input = &record.input;
    input.validate()?;
    // Context/local numeric features alone cannot reproduce their exact Rules
    // histories or prove they were frozen before this original submission.
    // Admit this row only after the original NNUEC owner evidence is joined.
    check(
        record.nnuec_input.is_none(),
        "MissingOriginalNnuecContextReplay: numeric NNUEC input alone cannot prove causal Rules history",
    )?;
    check(
        record.native.raw_submission["schema"] == "rz-rovent-native-owner-event/2",
        "historical Rules replay requires explicit native raw /2",
    )?;
    let replay: RoventSubmittedRulesReplay =
        serde_json::from_value(record.native.raw_submission["rules_replay"].clone())
            .map_err(|_| invalid("missing/malformed submitted Rules replay"))?;
    let indices = input
        .divergence_record_ids
        .iter()
        .map(|id| {
            input
                .selected_view
                .records
                .iter()
                .position(|row| &row.record_id == id)
                .ok_or_else(|| invalid("divergence record absent"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    verify_historical_rules_fields(input, &record.native, &replay, &indices)?;
    for name in ["wdl", "nnuec_wdl"] {
        let target = if name == "wdl" {
            record.targets.wdl.as_ref()
        } else {
            record.nnuec_wdl.as_ref()
        };
        if let Some(target) = target {
            verify_wdl_replay(record, name, target)?;
        }
    }
    Ok(())
}

/// 명시 형식 검증을 통과한 입력에만 사용하는 원 Rules/encoder 계산이다.
/// indices는 호출 형식의 실제 의미대로 전달하며 입력이나 raw domain을 재태그하지 않는다.
pub(super) fn verify_historical_rules_fields(
    input: &GenerationInputV3,
    native: &RoventNativeLifecycle,
    replay: &RoventSubmittedRulesReplay,
    indices: &[usize],
) -> Result<(), ArenaError> {
    check(
        replay.schema == ROVENT_SUBMITTED_RULES_REPLAY
            && replay.selected_anchors.len() == input.selected_view.anchors.len()
            && !replay.selected_anchors.is_empty()
            && replay.selected_anchors.len() <= 4,
        "submitted Rules replay schema/anchor allocation",
    )?;
    let actual = replay.actual_root.restore()?;
    let anchors = replay
        .selected_anchors
        .iter()
        .map(RoventRulesPositionReplay::restore)
        .collect::<Result<Vec<_>, _>>()?;
    check(
        replay.actual_root.moves == input.actual_history
            && actual.snapshot().same_state(&anchors[0].snapshot()),
        "actual played history/root does not match submitted anchor zero",
    )?;
    for (slot, position) in anchors.iter().enumerate() {
        check(
            exact_rules_anchor(position)? == input.selected_view.anchors[slot],
            "selected anchor Rules history mismatch",
        )?;
        // The public pure function is the original live Rules encoder, not a
        // second board/metadata implementation or an imported ReviewView.
        let expected = rz_uci::pals_native::virtual_encoding::exact_review_anchor_input(position)
            .map_err(|error| invalid(error.to_string()))?;
        let encoded = &input.encoded_model_input["anchors"][slot];
        check(
            encoded["board"]
                == serde_json::to_value(expected.board)
                    .map_err(|error| invalid(error.to_string()))?,
            "historical encoded anchor board differs from Rules",
        )?;
        let metadata = encoded["metadata"]
            .as_array()
            .ok_or_else(|| invalid("encoded anchor metadata"))?;
        check(
            metadata.len() == expected.metadata.len()
                && metadata
                    .iter()
                    .zip(expected.metadata)
                    .all(|(value, expected)| {
                        value.as_f64().is_some_and(|value| {
                            value.is_finite() && (value as f32).to_bits() == expected.to_bits()
                        })
                    }),
            "historical encoded anchor metadata differs from Rules",
        )?;
    }
    for source in &input.selected_view.records {
        let mut endpoint = anchors
            .get(source.anchor as usize)
            .ok_or_else(|| invalid("record anchor absent"))?
            .clone();
        apply(&mut endpoint, &source.moves, 256)?;
        check(
            source.rules_terminal
                == matches!(
                    endpoint
                        .classify_position()
                        .map_err(|error| invalid(error.to_string()))?
                        .play_status,
                    rz_position::PlayStatus::Terminal { .. }
                ),
            "record terminal marker differs from Rules replay",
        )?;
    }
    let query = &input.selected_view.query;
    let lines = [&query.prefix, &query.proposal, &query.counter];
    for (slot, line) in lines.into_iter().enumerate() {
        let mut endpoint = anchors
            .get(query.anchors[slot] as usize)
            .ok_or_else(|| invalid("query anchor absent"))?
            .clone();
        apply(&mut endpoint, line, 256)?;
        if slot == 0 {
            check(
                exact_rules_anchor(&endpoint)? == input.rules_anchor,
                "query prefix does not reach exact input Rules anchor",
            )?;
            let legal = packed(&endpoint.legal_moves())?;
            if matches!(query.purpose.as_str(), "divergence_policy" | "task_policy") {
                check(
                    input.legal_moves.is_empty(),
                    "scalar-only query has policy candidates",
                )?;
            } else {
                check(
                    legal.len() == input.legal_moves.len()
                        && input
                            .legal_moves
                            .iter()
                            .all(|movement| legal.contains(movement)),
                    "input candidates are not the complete query Rules legal set",
                )?;
            }
        }
    }
    let mut candidates = Sha256::new();
    for movement in &input.legal_moves {
        candidates.update(movement.to_le_bytes());
    }
    check(
        native.raw_submission["ticket"]["representation"]["candidates"]
            == hex(Digest(candidates.finalize().into())),
        "historical candidate order differs from original native key",
    )?;
    let proposal = query
        .proposal
        .iter()
        .map(|bits| decode_move(*bits))
        .collect::<Result<Vec<_>, _>>()?;
    let divergences = rz_uci::pals_native::virtual_encoding::encode_review_divergences(
        &anchors[0],
        &proposal,
        indices,
    )
    .map_err(|error| invalid(error.to_string()))?;
    let encoded = input.encoded_model_input["divergences"]
        .as_array()
        .ok_or_else(|| invalid("encoded divergences"))?;
    check(
        encoded.len() == divergences.len()
            && encoded.iter().zip(divergences).all(|(actual, expected)| {
                actual.as_array().is_some_and(|actual| {
                    actual.len() == expected.len()
                        && actual.iter().zip(expected).all(|(actual, expected)| {
                            actual.as_f64().is_some_and(|value| {
                                value.is_finite() && (value as f32).to_bits() == expected.to_bits()
                            })
                        })
                })
            }),
        "historical divergence input differs from original Rules encoder",
    )?;
    Ok(())
}

fn verify_wdl_replay(
    record: &RoventTrainingRecord,
    name: &str,
    target: &GenerationWdlTargetV3,
) -> Result<(), ArenaError> {
    let raw = &record
        .head_sources
        .get(name)
        .ok_or_else(|| invalid("WDL original head receipt absent"))?
        .raw_receipt;
    verify_rules_wdl_replay(
        raw,
        &record.input.game_id,
        &record.input.rules_anchor,
        target,
    )
}

/// 원 P/C와 R 자료가 같은 Rules 재생을 사용한다. 자료 검사만 수행하며
/// 현재 root, worker, 관측 시각 또는 target 생성 권한은 발급하지 않는다.
pub(super) fn verify_rules_wdl_replay(
    raw: &Value,
    game_id: &str,
    input_anchor: &GenerationRulesAnchorV3,
    target: &GenerationWdlTargetV3,
) -> Result<(), ArenaError> {
    let trace: RoventActualTrajectoryReplay =
        serde_json::from_value(raw["rules_replay"].clone())
            .map_err(|_| invalid("missing actual WDL Rules move trace"))?;
    check(
        trace.schema == ROVENT_ACTUAL_TRAJECTORY_REPLAY && trace.actual_selected_moves.len() <= 512,
        "WDL trajectory replay schema/capacity",
    )?;
    let mut position = trace.initial.restore()?;
    let mut anchors = vec![exact_rules_anchor(&position)?];
    for bits in &trace.actual_selected_moves {
        position
            .make_move(decode_move(*bits)?)
            .map_err(|error| invalid(error.to_string()))?;
        anchors.push(exact_rules_anchor(&position)?);
    }
    let final_position = trace.final_position.restore()?;
    check(
        position.snapshot().same_state(&final_position.snapshot())
            && position.revision() == final_position.revision(),
        "WDL actual moves do not reach original final snapshot/history",
    )?;
    let original: GenerationTrajectoryAuthorityV3 =
        serde_json::from_value(raw["trajectory"].clone())
            .map_err(|error| invalid(error.to_string()))?;
    check(
        original.anchors == anchors
            && original.game_id == game_id
            && original.anchors.contains(&target.anchor)
            && &target.anchor == input_anchor
            && target.trajectory_membership_sha256 == original.sha256()?,
        "WDL input is not an exact member of the actual replayed trajectory",
    )?;
    let classification = position
        .classify_position()
        .map_err(|error| invalid(error.to_string()))?;
    let (ending, outcome) = match trace.completion {
        RoventRulesCompletion::Automatic => {
            check(trace.claim.is_none(), "automatic WDL contains a claim")?;
            let rz_position::PlayStatus::Terminal { reason, winner } = classification.play_status
            else {
                return Err(invalid("WDL terminal result is not Rules terminal"));
            };
            let ending = match reason {
                rz_position::TerminalReason::Checkmate => GenerationRulesEndingV3::Checkmate,
                rz_position::TerminalReason::Stalemate => GenerationRulesEndingV3::Stalemate,
                rz_position::TerminalReason::DeadPosition => GenerationRulesEndingV3::DeadPosition,
                rz_position::TerminalReason::FivefoldRepetition => {
                    GenerationRulesEndingV3::Repetition
                }
                rz_position::TerminalReason::SeventyFiveMove => {
                    GenerationRulesEndingV3::SeventyFiveMove
                }
            };
            let outcome = match winner {
                Some(rz_position::Color::White) => GenerationOutcomeV3::WhiteWin,
                Some(rz_position::Color::Black) => GenerationOutcomeV3::BlackWin,
                None => GenerationOutcomeV3::Draw,
            };
            (ending, outcome)
        }
        RoventRulesCompletion::CurrentClaim => {
            let reason = match trace
                .claim
                .ok_or_else(|| invalid("WDL current claim reason absent"))?
            {
                RoventRulesClaim::ThreefoldRepetition => {
                    rz_position::ClaimReason::ThreefoldRepetition
                }
                RoventRulesClaim::FiftyMove => rz_position::ClaimReason::FiftyMove,
            };
            check(
                classification.claim_availability.iter().any(|claim| {
                    claim.reason == reason
                        && claim.evidence == rz_position::ClaimEvidence::CurrentPosition
                        && claim.availability == rz_position::Availability::Available
                }),
                "WDL current claim is not available in exact Rules history",
            )?;
            (
                if reason == rz_position::ClaimReason::ThreefoldRepetition {
                    GenerationRulesEndingV3::Repetition
                } else {
                    GenerationRulesEndingV3::FiftyMove
                },
                GenerationOutcomeV3::Draw,
            )
        }
    };
    check(
        original.ending == ending
            && original.outcome == outcome
            && target.ending == ending
            && target.outcome == outcome
            && generation_digest("rz-pals-rules-outcome-evidence/1", &raw["rules_receipt"])?
                == original.result_evidence_sha256
            && original.result_evidence_sha256 == target.rules_result_sha256,
        "replayed Rules outcome differs from original head target/receipt",
    )?;
    check(
        raw["rules_receipt"]["schema"] == "rz-pals-rules-outcome-evidence/1"
            && raw["rules_receipt"]["game_id"] == game_id
            && raw["rules_receipt"]["rules_terminal"] == true
            && raw["rules_receipt"]["ending"]
                == serde_json::to_value(ending).map_err(|error| invalid(error.to_string()))?
            && raw["rules_receipt"]["outcome"]
                == serde_json::to_value(outcome).map_err(|error| invalid(error.to_string()))?,
        "original Rules receipt does not describe the replayed result",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rovent_rules_replay_roundtrip_keeps_exact_actual_history() {
        let mut position = Position::startpos();
        for movement in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            position.make_uci(movement).unwrap();
        }
        let replay = RoventRulesPositionReplay::capture(&position).unwrap();
        let restored = replay.restore().unwrap();
        assert!(position.snapshot().same_state(&restored.snapshot()));
        assert_eq!(position.revision(), restored.revision());
        assert_eq!(replay.moves.len(), 4);
    }
    #[test]
    fn rovent_rules_replay_rejects_same_fen_without_original_history() {
        let mut position = Position::startpos();
        for movement in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            position.make_uci(movement).unwrap();
        }
        let mut replay = RoventRulesPositionReplay::capture(&position).unwrap();
        replay.origin = RoventReplayOrigin::Fen;
        replay.completeness = RoventReplayCompleteness::UnknownPrefix;
        replay.start_fen = position.to_fen();
        replay.moves.clear();
        assert!(replay.restore().is_err());
    }
    #[test]
    fn rovent_rules_replay_refuses_illegal_moves_and_prefix_upgrade() {
        let position = Position::startpos();
        let mut replay = RoventRulesPositionReplay::capture(&position).unwrap();
        replay.moves.push(
            Move16::pack(BoardMove::from_uci("e2e5").unwrap())
                .unwrap()
                .bits(),
        );
        replay.observer_revision += 1;
        assert!(replay.restore().is_err());
        let imported = Position::from_fen(&position.to_fen()).unwrap();
        let mut replay = RoventRulesPositionReplay::capture(&imported).unwrap();
        assert_eq!(replay.completeness, RoventReplayCompleteness::UnknownPrefix);
        assert!(replay.restore().is_ok());
        replay.completeness = RoventReplayCompleteness::Complete;
        assert!(replay.restore().is_err());
    }
}
