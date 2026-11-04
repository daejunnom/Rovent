//! Explicit test-only UCI ELF. It is not Stockfish, a model, or a training
//! teacher. It supports only startpos histories and two bounded observations.
use rz_position::{BoardMove, Position};
use std::io::{self, BufRead, Read, Write};

const MAX_LINE: u64 = 8192;
const MAX_COMMANDS: usize = 10_000;

fn run() -> Result<(), String> {
    if std::env::args_os().len() != 1 {
        return Err("fixture accepts no arguments".into());
    }
    let input = io::stdin();
    let mut input = input.lock();
    let output = io::stdout();
    let mut output = output.lock();
    let mut position = Position::startpos();
    let mut go_count = 0usize;
    for _ in 0..MAX_COMMANDS {
        let mut raw = Vec::new();
        let n = (&mut input)
            .take(MAX_LINE + 1)
            .read_until(b'\n', &mut raw)
            .map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(());
        }
        if n as u64 > MAX_LINE || raw.last() != Some(&b'\n') {
            return Err("fixture command exceeds bound".into());
        }
        let line = std::str::from_utf8(&raw).map_err(|e| e.to_string())?.trim();
        if line == "uci" {
            writeln!(output, "id name RZ external checker fixture\nid author RoveZero tests\noption name Threads type spin default 1 min 1 max 2\noption name MultiPV type spin default 1 min 1 max 1\nuciok").map_err(|e| e.to_string())?;
        } else if line == "isready" {
            writeln!(output, "readyok").map_err(|e| e.to_string())?;
        } else if line == "ucinewgame" {
            position = Position::startpos();
            go_count = 0;
        } else if line == "setoption name Threads value 1"
            || line == "setoption name Threads value 2"
            || line == "setoption name MultiPV value 1"
        {
            // Deliberately supported fixture configuration only.
        } else if let Some(rest) = line.strip_prefix("position startpos") {
            position = Position::startpos();
            let rest = rest.trim();
            if !rest.is_empty() {
                let moves = rest
                    .strip_prefix("moves ")
                    .ok_or("fixture position format")?;
                for text in moves.split_ascii_whitespace() {
                    let movement = BoardMove::from_uci(text).map_err(|e| e.to_string())?;
                    position.make_move(movement).map_err(|e| e.to_string())?;
                }
            }
        } else if line.starts_with("go ") {
            if line
                .split_ascii_whitespace()
                .any(|word| word == "searchmoves")
            {
                return Err("fixture restricted roots unsupported".into());
            }
            let legal = position.legal_moves();
            if legal.len() < 2 {
                return Err("fixture requires two legal moves".into());
            }
            go_count += 1;
            if go_count == 1 {
                writeln!(
                    output,
                    "info depth 2 seldepth 2 nodes 1 score cp 21 pv {}\nbestmove {}",
                    legal[0], legal[0]
                )
                .map_err(|e| e.to_string())?;
            } else if go_count == 2 {
                writeln!(
                    output,
                    "info depth 2 multipv 1 nodes 1 score cp 32 lowerbound pv {}\nbestmove {}",
                    legal[0], legal[1]
                )
                .map_err(|e| e.to_string())?;
            } else {
                return Err("fixture supports exactly two go requests".into());
            }
        } else if line == "stop" {
            // Each go returned bestmove before another command is read.
        } else if line == "quit" {
            return Ok(());
        } else {
            return Err("fixture command unsupported".into());
        }
        output.flush().map_err(|e| e.to_string())?;
    }
    Err("fixture command limit reached".into())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("test-only teacher fixture: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
