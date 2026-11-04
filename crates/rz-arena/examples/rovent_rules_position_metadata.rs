//! Detached Rules replay/anchor preparation through the original Rust owners.
//!
//! `cargo run -p rz-arena --features pals-collection-onnx --example
//! rovent_rules_position_metadata -- --position-command="position startpos"`
//! The feature exposes the existing codec; this example performs no model
//! loading, NN execution, native startup, UCI Start, clock or optimizer update.
#![forbid(unsafe_code)]

#[cfg(not(feature = "pals-collection-onnx"))]
fn main() -> std::process::ExitCode {
    use std::io::Write as _;
    let _ = std::io::stderr().lock().write_all(
        b"{\"schema\":\"rz-rovent-rules-position-preparation-error/1\",\"code\":\"unsupported_build_requires_pals_collection_onnx\",\"neural_execution\":false,\"current_root_authority_issued\":false,\"execution_authorized\":false,\"performed_optimizer_updates\":0}\n",
    );
    std::process::ExitCode::from(2)
}

#[cfg(feature = "pals-collection-onnx")]
fn main() -> std::process::ExitCode {
    metadata::main()
}

#[cfg(feature = "pals-collection-onnx")]
mod metadata {
    use rz_arena::rovent_collect::RoventRulesPositionReplay;
    use rz_position::{BoardMove, Position};
    use rz_uci::parser::{self, Command, ParserLimits, PositionBase};
    use serde::Serialize;
    use std::ffi::OsString;
    use std::io::{self, Write};
    use std::process::ExitCode;

    const OUTPUT_MAX_BYTES: usize = 64 << 10;
    const DETAIL_MAX_CHARS: usize = 512;

    #[derive(Debug)]
    struct Error {
        code: &'static str,
        detail: String,
    }
    fn error(code: &'static str, detail: impl ToString) -> Error {
        Error {
            code,
            detail: detail.to_string().chars().take(DETAIL_MAX_CHARS).collect(),
        }
    }
    fn argument(arguments: impl IntoIterator<Item = OsString>) -> Result<String, Error> {
        let mut arguments = arguments.into_iter();
        let value = arguments.next().ok_or_else(|| {
            error(
                "invalid_arguments",
                "one --position-command=POSITION is required",
            )
        })?;
        if arguments.next().is_some() {
            return Err(error(
                "invalid_arguments",
                "only one --position-command=POSITION is accepted",
            ));
        }
        let command = value
            .to_str()
            .and_then(|value| value.strip_prefix("--position-command="))
            .ok_or_else(|| {
                error(
                    "invalid_arguments",
                    "one UTF-8 --position-command=POSITION is required",
                )
            })?;
        if command.len() > ParserLimits::default().max_line_bytes {
            return Err(error(
                "position_command_syntax_rejected",
                "position command exceeds the existing parser byte limit",
            ));
        }
        Ok(command.to_owned())
    }

    /// Closed preparation envelope; these flags describe only this source
    /// path. The enclosed DTO cannot be converted into live root authority.
    #[derive(Serialize)]
    struct Preparation<'a> {
        schema: &'static str,
        position_command: &'a str,
        position_replay: RoventRulesPositionReplay,
        rules_replay_checked: bool,
        neural_execution: bool,
        current_root_authority_issued: bool,
        execution_authorized: bool,
        performed_optimizer_updates: u64,
    }
    fn prepare(command: &str) -> Result<Preparation<'_>, Error> {
        let Command::Position(specification) = parser::parse(command, ParserLimits::default())
            .map_err(|cause| error("position_command_syntax_rejected", cause))?
        else {
            return Err(error(
                "position_command_required",
                "only a UCI position command is accepted",
            ));
        };
        let mut position = match specification.base {
            PositionBase::StartPos => Position::startpos(),
            PositionBase::Fen(fen) => {
                Position::from_fen(&fen).map_err(|cause| error("position_base_rejected", cause))?
            }
        };
        for (ply, movement) in specification.moves.iter().enumerate() {
            let movement = BoardMove::from_uci(movement).map_err(|cause| {
                error(
                    "position_move_syntax_rejected",
                    format!("zero-based ply {ply}: {cause}"),
                )
            })?;
            position.make_move(movement).map_err(|cause| {
                error(
                    "position_move_rejected",
                    format!("zero-based ply {ply}: {cause}"),
                )
            })?;
        }
        let position_replay = RoventRulesPositionReplay::metadata_from_position(&position)
            .map_err(|cause| error("rules_replay_rejected", cause))?;
        Ok(Preparation {
            schema: "rz-rovent-rules-position-preparation/1",
            position_command: command,
            position_replay,
            rules_replay_checked: true,
            neural_execution: false,
            current_root_authority_issued: false,
            execution_authorized: false,
            performed_optimizer_updates: 0,
        })
    }
    fn execute() -> Result<(), Error> {
        let command = argument(std::env::args_os().skip(1))?;
        let prepared = prepare(&command)?;
        let mut bytes = serde_json::to_vec(&prepared)
            .map_err(|cause| error("metadata_encoding_failed", cause))?;
        if bytes.len() >= OUTPUT_MAX_BYTES {
            return Err(error(
                "metadata_output_budget",
                "metadata exceeds the finite stdout byte budget",
            ));
        }
        bytes.push(b'\n');
        io::stdout()
            .lock()
            .write_all(&bytes)
            .map_err(|cause| error("metadata_write_failed", cause))
    }
    pub(super) fn main() -> ExitCode {
        match execute() {
            Ok(()) => ExitCode::SUCCESS,
            Err(cause) => {
                let mut bytes = serde_json::to_vec(&serde_json::json!({
                    "schema":"rz-rovent-rules-position-preparation-error/1",
                    "code":cause.code,"detail":cause.detail,
                    "neural_execution":false,"current_root_authority_issued":false,
                    "execution_authorized":false,"performed_optimizer_updates":0,
                }))
                .expect("bounded error fields serialize");
                bytes.push(b'\n');
                let _ = io::stderr().lock().write_all(&bytes);
                ExitCode::from(2)
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use rz_arena::rovent_collect::{RoventReplayCompleteness, RoventReplayOrigin};

        #[test]
        fn exact_history_replay_and_imported_fen_keep_distinct_original_anchors() {
            let played = prepare("position startpos moves g1f3 g8f6 f3g1 f6g8").unwrap();
            let replay = &played.position_replay;
            assert_eq!(replay.origin, RoventReplayOrigin::StartPosition);
            assert_eq!(replay.completeness, RoventReplayCompleteness::Complete);
            assert_eq!(replay.moves.len(), 4);
            assert_eq!(replay.restore().unwrap().snapshot().known_history_len(), 5);
            let command = format!("position fen {}", replay.anchor.board_fen);
            let imported = prepare(&command).unwrap();
            assert_eq!(imported.position_replay.origin, RoventReplayOrigin::Fen);
            assert_eq!(
                imported.position_replay.completeness,
                RoventReplayCompleteness::UnknownPrefix
            );
            assert_eq!(
                replay.anchor.board_fen,
                imported.position_replay.anchor.board_fen
            );
            assert_ne!(
                replay.anchor.rules_history_sha256,
                imported.position_replay.anchor.rules_history_sha256
            );
            let value = serde_json::to_value(&played).unwrap();
            assert_eq!(value.as_object().unwrap().len(), 8);
            assert_eq!(value["current_root_authority_issued"], false);
            assert_eq!(value["execution_authorized"], false);
            assert_eq!(value["neural_execution"], false);
            assert_eq!(value["performed_optimizer_updates"], 0);
        }

        #[test]
        fn invalid_or_additional_commands_and_illegal_moves_are_rejected() {
            for command in [
                "go nodes 1",
                "position startpos\ngo nodes 1",
                "position startpos moves e2e5",
                "position fen 8/8/8/8/8/8/8/8 w - - 0 1",
            ] {
                assert!(prepare(command).is_err(), "{command}");
            }
            assert!(
                argument([
                    OsString::from("--position-command=position startpos"),
                    OsString::from("--position-command=position startpos")
                ])
                .is_err()
            );
        }
    }
}
