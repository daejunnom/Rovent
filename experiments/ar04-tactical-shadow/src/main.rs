#![forbid(unsafe_code)]

//! Read-only chess-state observation. This is not an AR04 network or scorer.
use rz_position::{
    Availability, BoardMove, HistoryCompleteness, PlayStatus, Position, TerminalReason,
};
use serde_json::{Value, json};
use std::{env, process::ExitCode};

const MAX_HISTORY_PLIES: usize = 256;
const MAX_REPLIES: usize = 2;

struct Input {
    origin: String,
    fen: Option<String>,
    moves: Vec<String>,
    candidate: String,
    replies: Vec<String>,
}

fn parse_input() -> Result<Input, String> {
    let mut args = env::args().skip(1);
    let mut origin: Option<String> = None;
    let mut fen = None;
    let mut moves = Vec::new();
    let mut candidate = None;
    let mut replies = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--startpos" => {
                if origin.replace("startpos".into()).is_some() {
                    return Err("specify exactly one --startpos or --fen".into());
                }
            }
            "--fen" => {
                if origin.replace("fen".into()).is_some() {
                    return Err("specify exactly one --startpos or --fen".into());
                }
                fen = Some(args.next().ok_or("--fen requires a quoted FEN")?);
            }
            "--move" => moves.push(args.next().ok_or("--move requires UCI notation")?),
            "--candidate" => {
                if candidate.is_some() {
                    return Err("specify exactly one --candidate".into());
                }
                candidate = Some(args.next().ok_or("--candidate requires UCI notation")?);
            }
            "--reply" => replies.push(args.next().ok_or("--reply requires UCI notation")?),
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    if moves.len() > MAX_HISTORY_PLIES || replies.len() > MAX_REPLIES {
        return Err("history or reply limit exceeded".into());
    }
    Ok(Input {
        origin: origin.ok_or("--startpos or --fen is required")?,
        fen,
        moves,
        candidate: candidate.ok_or("--candidate is required")?,
        replies,
    })
}

fn status(position: &Position) -> Result<Value, String> {
    let classification = position.classify_position().map_err(|e| e.to_string())?;
    let (status, reason) = match classification.play_status {
        PlayStatus::Ongoing => ("ongoing", None),
        PlayStatus::Terminal { reason, .. } => (
            "terminal",
            Some(match reason {
                TerminalReason::Checkmate => "checkmate",
                TerminalReason::Stalemate => "stalemate",
                TerminalReason::DeadPosition => "dead_position",
                TerminalReason::FivefoldRepetition => "fivefold_repetition",
                TerminalReason::SeventyFiveMove => "seventy_five_move",
            }),
        ),
    };
    Ok(json!({
        "play_status": status,
        "status_scope": "rz_position_proven_automatic_outcomes_only",
        "terminal_reason": reason,
        "known_repetition_count": classification.history_evidence.known_repetition_count,
        "repetition_evidence_complete": classification.history_evidence.repetition_is_complete,
        "fivefold_repetition_evidence": match classification.history_evidence.fivefold_repetition {
            Availability::Available => "available",
            Availability::Unavailable => "unavailable",
            Availability::Unknown => "unknown",
        },
        "dead_position_scope": "proven_material_subset_only",
        "claim_policy": "not_exercised",
    }))
}

fn run(input: Input) -> Result<Value, String> {
    let mut pos = match input.fen.as_deref() {
        Some(fen) => Position::from_fen(fen).map_err(|e| e.to_string())?,
        None => Position::startpos(),
    };
    let initial_fen = pos.to_fen();
    for (index, mv) in input.moves.iter().enumerate() {
        if status(&pos)?["play_status"] == "terminal" {
            return Err(format!(
                "history continues after terminal state at ply {index}"
            ));
        }
        pos.make_uci(mv)
            .map_err(|e| format!("history ply {}: {e}", index + 1))?;
    }
    let candidate = BoardMove::from_uci(&input.candidate).map_err(|e| e.to_string())?;
    let replies = input
        .replies
        .iter()
        .map(|s| BoardMove::from_uci(s).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let root_status = status(&pos)?;
    let root_legal = pos.legal_moves();
    let available = root_status["play_status"] != "terminal";
    let candidate_legal = available && root_legal.contains(&candidate);
    let mut observed = Vec::new();
    let mut child_fen = None;
    let mut child_status = None;
    if candidate_legal {
        let child = pos.preview_move(candidate).map_err(|e| e.to_string())?;
        let cs = status(&child)?;
        let child_available = cs["play_status"] != "terminal";
        let child_legal = child.legal_moves();
        for reply in replies {
            let legal = child_available && child_legal.contains(&reply);
            let after_fen = if legal {
                Some(
                    child
                        .preview_move(reply)
                        .map_err(|e| e.to_string())?
                        .to_fen(),
                )
            } else {
                None
            };
            observed.push(json!({"move": reply.to_string(), "board_legal": if child_available { Some(legal) } else { None }, "unavailable_terminal": !child_available, "after_fen": after_fen}));
        }
        child_fen = Some(child.to_fen());
        child_status = Some(cs);
    } else {
        for reply in replies {
            observed.push(json!({"move": reply.to_string(), "board_legal": null, "unavailable_reason": "candidate_not_available", "after_fen": null}));
        }
    }
    Ok(json!({
        "schema": "ar04_tactical_shadow.v1",
        "research_circuit": {"status": "not_available", "reason": "manual_circuit_inputs_and_weights_not_registered"},
        "origin": input.origin,
        "fen_as_supplied": input.fen,
        "initial_fen": initial_fen,
        "history_moves": input.moves,
        "history_completeness": match pos.history_completeness() { HistoryCompleteness::Complete => "complete", HistoryCompleteness::UnknownPrefix => "unknown_prefix" },
        "root_fen": pos.to_fen(),
        "root_status": root_status,
        "root_legal_moves": root_legal.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "candidate": {"move": candidate.to_string(), "board_legal": if available { Some(candidate_legal) } else { None }, "unavailable_terminal": !available, "after_fen": child_fen, "after_status": child_status},
        "replies": observed,
        "scope": "board_legality_and_known_automatic_status_only_no_policy_wdl_tactical_or_forced_mate_claim",
    }))
}

fn main() -> ExitCode {
    match parse_input().and_then(run) {
        Ok(value) => {
            println!("{value}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observes_candidate_and_separate_replies_without_circuit_output() {
        let result = run(Input {
            origin: "startpos".into(),
            fen: None,
            moves: vec!["e2e4".into()],
            candidate: "e7e5".into(),
            replies: vec!["g1f3".into(), "b1c3".into()],
        })
        .unwrap();
        assert_eq!(result["history_completeness"], "complete");
        assert_eq!(result["candidate"]["board_legal"], true);
        assert_eq!(result["replies"][0]["board_legal"], true);
        assert_eq!(result["replies"][1]["board_legal"], true);
        assert_ne!(
            result["replies"][0]["after_fen"],
            result["replies"][1]["after_fen"]
        );
        assert_eq!(result["research_circuit"]["status"], "not_available");
    }

    #[test]
    fn fen_history_remains_unknown_and_illegal_candidate_has_no_child() {
        let result = run(Input {
            origin: "fen".into(),
            fen: Some("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".into()),
            moves: vec![],
            candidate: "e2e5".into(),
            replies: vec!["e7e5".into()],
        })
        .unwrap();
        assert_eq!(result["history_completeness"], "unknown_prefix");
        assert_eq!(result["candidate"]["board_legal"], false);
        assert!(result["candidate"]["after_fen"].is_null());
        assert!(result["replies"][0]["board_legal"].is_null());
    }

    #[test]
    fn terminal_state_is_unavailable_not_an_illegal_move_claim() {
        let result = run(Input {
            origin: "fen".into(),
            fen: Some("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1".into()),
            moves: vec![],
            candidate: "h8h7".into(),
            replies: vec![],
        })
        .unwrap();
        assert_eq!(result["root_status"]["terminal_reason"], "checkmate");
        assert!(result["candidate"]["board_legal"].is_null());
        assert_eq!(result["candidate"]["unavailable_terminal"], true);
    }
}
