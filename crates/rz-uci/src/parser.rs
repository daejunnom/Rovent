use std::fmt;
use std::num::NonZeroU16;

/// Line size is checked before allocating fields; command-specific bounds
/// are checked before copying an option or move trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParserLimits {
    pub max_line_bytes: usize,
    pub max_moves: usize,
    /// Root-list cardinality bound, checked before copying `searchmoves` and
    /// again against the complete Rules view by the session.
    pub max_legal_moves: usize,
    pub max_option_name_bytes: usize,
    pub max_option_value_bytes: usize,
}

impl Default for ParserLimits {
    fn default() -> Self {
        Self {
            max_line_bytes: 16 * 1024,
            max_moves: 2048,
            max_legal_moves: 4096,
            max_option_name_bytes: 128,
            max_option_value_bytes: 1024,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PositionBase {
    StartPos,
    /// Exactly six FEN fields; chess semantics are checked by Rules.
    Fen(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionSpec {
    pub base: PositionBase,
    pub moves: Vec<String>,
}

impl Default for PositionSpec {
    fn default() -> Self {
        Self {
            base: PositionBase::StartPos,
            moves: Vec::new(),
        }
    }
}

/// UCI fields, not a shared clock/evaluator contract. The timing adapter selects
/// the real side-to-move clock and accounts for output/drain reserves.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GoLimits {
    /// An explicit nonempty root restriction, in the received order. Rules
    /// checks legality and selects its canonical legal-move order later.
    pub searchmoves: Option<Vec<String>>,
    /// Root-relative selective chess-ply horizon, not NN calls or R depth.
    pub depth_plies: Option<NonZeroU16>,
    /// Forced-mate objective in chess moves; proof is a search responsibility.
    pub mate_moves: Option<NonZeroU16>,
    pub movetime_ms: Option<u64>,
    pub white_time_ms: Option<u64>,
    pub black_time_ms: Option<u64>,
    pub white_increment_ms: Option<u64>,
    pub black_increment_ms: Option<u64>,
    pub moves_to_go: Option<u32>,
    pub nodes: Option<u64>,
    pub infinite: bool,
    pub ponder: bool,
}

impl GoLimits {
    /// No UCI stop criterion was supplied; the configured resource profile
    /// remains a finite execution boundary and must be reported to the GUI.
    pub(crate) fn uses_default_resource_limits(&self) -> bool {
        !self.infinite
            && !self.ponder
            && self.movetime_ms.is_none()
            && self.white_time_ms.is_none()
            && self.black_time_ms.is_none()
            && self.nodes.is_none()
            && self.depth_plies.is_none()
            && self.mate_moves.is_none()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Uci,
    IsReady,
    SetOption {
        name: String,
        value: Option<String>,
    },
    Position(PositionSpec),
    Go(GoLimits),
    Stop,
    PonderHit,
    Quit,
    NewGame,
    /// Unknown commands are diagnosed and ignored by the session.
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    LineTooLong,
    InvalidText,
    Empty,
    Malformed(&'static str),
    Unsupported(String),
    Limit(&'static str),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LineTooLong => f.write_str("UCI line exceeds configured byte limit"),
            Self::InvalidText => f.write_str("UCI input must be one ASCII line"),
            Self::Empty => f.write_str("empty UCI command"),
            Self::Malformed(reason) => write!(f, "malformed UCI input: {reason}"),
            Self::Unsupported(field) => write!(f, "unsupported UCI command or field: {field}"),
            Self::Limit(reason) => write!(f, "UCI input exceeds configured limit: {reason}"),
        }
    }
}

impl std::error::Error for ParseError {}

pub fn parse(line: &str, limits: ParserLimits) -> Result<Command, ParseError> {
    parse_with_zero_node_limit(line, limits, ZeroNodeLimit::Reject)
}

/// Syntax selection for an explicitly registered node domain. This does not
/// grant search capability or turn a zero-node request into completed work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ZeroNodeLimit {
    Reject,
    Allow,
}

pub(crate) fn parse_with_zero_node_limit(
    line: &str,
    limits: ParserLimits,
    zero_nodes: ZeroNodeLimit,
) -> Result<Command, ParseError> {
    if line.len() > limits.max_line_bytes {
        return Err(ParseError::LineTooLong);
    }
    // A caller may supply the physical trailing newline, but never two lines.
    let line = line.strip_suffix('\n').unwrap_or(line);
    let line = line.strip_suffix('\r').unwrap_or(line);
    if !line.is_ascii() || line.bytes().any(|b| (b < b' ' && b != b'\t') || b == 0x7f) {
        return Err(ParseError::InvalidText);
    }
    let fields: Vec<_> = line.split_ascii_whitespace().collect();
    let Some(&head) = fields.first() else {
        return Err(ParseError::Empty);
    };
    let tail = &fields[1..];
    match head {
        "uci" => no_args(tail, Command::Uci),
        "isready" => no_args(tail, Command::IsReady),
        "stop" => no_args(tail, Command::Stop),
        "quit" => no_args(tail, Command::Quit),
        "ucinewgame" => no_args(tail, Command::NewGame),
        "setoption" => parse_option(line, limits),
        "position" => parse_position(tail, limits),
        "go" => parse_go(tail, limits, zero_nodes),
        "ponderhit" => no_args(tail, Command::PonderHit),
        _ => Ok(Command::Unknown(head.to_owned())),
    }
}

fn no_args(fields: &[&str], command: Command) -> Result<Command, ParseError> {
    if fields.is_empty() {
        Ok(command)
    } else {
        Err(ParseError::Malformed("unexpected arguments"))
    }
}

fn parse_option(line: &str, limits: ParserLimits) -> Result<Command, ParseError> {
    let (_, after_command) = split_token(line).ok_or(ParseError::Empty)?;
    let Some((keyword, after_name)) = split_token(after_command) else {
        return Err(ParseError::Malformed("setoption requires name"));
    };
    if keyword != "name" {
        return Err(ParseError::Malformed("setoption requires name"));
    }
    // Only syntax separators are trimmed. Option names, combo choices and string
    // paths are literal payloads; joining whitespace tokens silently changes them.
    let payload = after_name.trim_ascii_start();
    let mut tail = payload;
    let mut name_end = payload.len();
    let mut value = None;
    while let Some((token, remainder)) = split_token(tail) {
        if token == "value" {
            name_end = payload.len() - tail.trim_ascii_start().len();
            value = Some(remainder.trim_ascii());
            break;
        }
        tail = remainder;
    }
    let name = payload[..name_end].trim_ascii();
    if name.is_empty() {
        return Err(ParseError::Malformed("option name is empty"));
    }
    if name.len() > limits.max_option_name_bytes {
        return Err(ParseError::Limit("option name"));
    }
    if value.is_some_and(|text| text.len() > limits.max_option_value_bytes) {
        return Err(ParseError::Limit("option value"));
    }
    Ok(Command::SetOption {
        name: name.to_owned(),
        value: value.map(str::to_owned),
    })
}

fn split_token(line: &str) -> Option<(&str, &str)> {
    let line = line.trim_ascii_start();
    if line.is_empty() {
        return None;
    }
    let end = line
        .find(|character: char| character.is_ascii_whitespace())
        .unwrap_or(line.len());
    Some((&line[..end], &line[end..]))
}

fn parse_position(fields: &[&str], limits: ParserLimits) -> Result<Command, ParseError> {
    let (base, used) = match fields.first().copied() {
        Some("startpos") => (PositionBase::StartPos, 1),
        Some("fen") if fields.len() >= 7 => {
            validate_fen_text(&fields[1..7])?;
            (PositionBase::Fen(fields[1..7].join(" ")), 7)
        }
        Some("fen") => return Err(ParseError::Malformed("FEN requires six fields")),
        _ => return Err(ParseError::Malformed("position requires startpos or fen")),
    };
    let tail = &fields[used..];
    if tail.is_empty() {
        return Ok(Command::Position(PositionSpec {
            base,
            moves: Vec::new(),
        }));
    }
    if tail[0] != "moves" || tail.len() == 1 {
        return Err(ParseError::Malformed("expected a nonempty moves trace"));
    }
    if tail.len() - 1 > limits.max_moves {
        return Err(ParseError::Limit("move trace"));
    }
    if tail[1..].iter().any(|m| !is_move_text(m)) {
        return Err(ParseError::Malformed(
            "move requires from/to and optional q/r/b/n promotion",
        ));
    }
    Ok(Command::Position(PositionSpec {
        base,
        moves: tail[1..].iter().map(|s| (*s).to_owned()).collect(),
    }))
}

/// Text validation only. Legality, castling, promotion, and king safety belong
/// to the injected Rules adapter.
pub(crate) fn is_move_text(s: &str) -> bool {
    let b = s.as_bytes();
    (b.len() == 4 || b.len() == 5)
        && (b'a'..=b'h').contains(&b[0])
        && (b'1'..=b'8').contains(&b[1])
        && (b'a'..=b'h').contains(&b[2])
        && (b'1'..=b'8').contains(&b[3])
        && b[..2] != b[2..4]
        && (b.len() == 4 || matches!(b[4], b'q' | b'r' | b'b' | b'n'))
}

fn validate_fen_text(fields: &[&str]) -> Result<(), ParseError> {
    let ranks: Vec<_> = fields[0].split('/').collect();
    if ranks.len() != 8 {
        return Err(ParseError::Malformed("FEN board requires eight ranks"));
    }
    for rank in ranks {
        let mut squares = 0u32;
        for b in rank.bytes() {
            squares += match b {
                b'1'..=b'8' => u32::from(b - b'0'),
                b'p' | b'r' | b'n' | b'b' | b'q' | b'k' | b'P' | b'R' | b'N' | b'B' | b'Q'
                | b'K' => 1,
                _ => return Err(ParseError::Malformed("invalid FEN board character")),
            };
            if squares > 8 {
                return Err(ParseError::Malformed("FEN rank exceeds eight squares"));
            }
        }
        if squares != 8 {
            return Err(ParseError::Malformed("FEN rank requires eight squares"));
        }
    }
    if !matches!(fields[1], "w" | "b") {
        return Err(ParseError::Malformed("invalid FEN active color"));
    }
    if fields[2] != "-" {
        let mut seen = [false; 4];
        for b in fields[2].bytes() {
            let index = match b {
                b'K' => 0,
                b'Q' => 1,
                b'k' => 2,
                b'q' => 3,
                _ => return Err(ParseError::Malformed("invalid FEN castling rights")),
            };
            if seen[index] {
                return Err(ParseError::Malformed("duplicate FEN castling right"));
            }
            seen[index] = true;
        }
    }
    let ep = fields[3].as_bytes();
    if fields[3] != "-"
        && !(ep.len() == 2 && (b'a'..=b'h').contains(&ep[0]) && matches!(ep[1], b'3' | b'6'))
    {
        return Err(ParseError::Malformed("invalid FEN en passant square"));
    }
    decimal::<u32>(fields[4])?;
    if decimal::<u32>(fields[5])? == 0 {
        return Err(ParseError::Malformed("FEN fullmove number is zero"));
    }
    Ok(())
}

fn decimal<T: std::str::FromStr>(s: &str) -> Result<T, ParseError> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(ParseError::Malformed("expected unsigned decimal integer"));
    }
    s.parse()
        .map_err(|_| ParseError::Malformed("integer overflow"))
}

fn is_go_keyword(field: &str) -> bool {
    matches!(
        field,
        "searchmoves"
            | "depth"
            | "mate"
            | "movetime"
            | "wtime"
            | "btime"
            | "winc"
            | "binc"
            | "movestogo"
            | "nodes"
            | "infinite"
            | "ponder"
    )
}

fn parse_go(
    fields: &[&str],
    parser_limits: ParserLimits,
    zero_nodes: ZeroNodeLimit,
) -> Result<Command, ParseError> {
    let mut limits = GoLimits::default();
    let mut i = 0;
    while i < fields.len() {
        let field = fields[i];
        if field == "searchmoves" {
            if limits.searchmoves.is_some() {
                return Err(ParseError::Malformed("duplicate go field"));
            }
            let start = i + 1;
            let end = fields[start..]
                .iter()
                .position(|field| is_go_keyword(field))
                .map_or(fields.len(), |offset| start + offset);
            let moves = &fields[start..end];
            if moves.is_empty() {
                return Err(ParseError::Malformed(
                    "searchmoves requires a nonempty list",
                ));
            }
            if moves.len() > parser_limits.max_legal_moves {
                return Err(ParseError::Limit("searchmoves"));
            }
            if moves.iter().any(|movement| !is_move_text(movement)) {
                return Err(ParseError::Malformed(
                    "searchmoves requires from/to and optional q/r/b/n promotion",
                ));
            }
            limits.searchmoves = Some(
                moves
                    .iter()
                    .map(|movement| (*movement).to_owned())
                    .collect(),
            );
            i = end;
            continue;
        }
        if field == "ponder" {
            if limits.ponder {
                return Err(ParseError::Malformed("duplicate go field"));
            }
            limits.ponder = true;
            i += 1;
            continue;
        }
        if field == "infinite" {
            if limits.infinite {
                return Err(ParseError::Malformed("duplicate go field"));
            }
            limits.infinite = true;
            i += 1;
            continue;
        }
        if !matches!(
            field,
            "depth"
                | "mate"
                | "movetime"
                | "wtime"
                | "btime"
                | "winc"
                | "binc"
                | "movestogo"
                | "nodes"
        ) {
            return Err(ParseError::Unsupported(field.to_owned()));
        }
        let value = fields
            .get(i + 1)
            .ok_or(ParseError::Malformed("go field lacks value"))?;
        let number: u64 = decimal(value)?;
        match field {
            "depth" => {
                let value =
                    u16::try_from(number).map_err(|_| ParseError::Malformed("depth overflow"))?;
                let value = NonZeroU16::new(value)
                    .ok_or(ParseError::Malformed("depth must be positive"))?;
                set_once(&mut limits.depth_plies, value)?;
            }
            "mate" => {
                let value =
                    u16::try_from(number).map_err(|_| ParseError::Malformed("mate overflow"))?;
                let value =
                    NonZeroU16::new(value).ok_or(ParseError::Malformed("mate must be positive"))?;
                // Both forced wins (2*N-1) and forced losses (2*N) use the
                // shared u16 chess-ply domain. Never silently wrap or clamp.
                value
                    .get()
                    .checked_mul(2)
                    .ok_or(ParseError::Malformed("mate ply horizon overflow"))?;
                set_once(&mut limits.mate_moves, value)?;
            }
            "movetime" => set_once(&mut limits.movetime_ms, number)?,
            "wtime" => set_once(&mut limits.white_time_ms, number)?,
            "btime" => set_once(&mut limits.black_time_ms, number)?,
            "winc" => set_once(&mut limits.white_increment_ms, number)?,
            "binc" => set_once(&mut limits.black_increment_ms, number)?,
            "nodes" if number > 0 || zero_nodes == ZeroNodeLimit::Allow => {
                set_once(&mut limits.nodes, number)?
            }
            "nodes" => return Err(ParseError::Malformed("nodes must be positive")),
            "movestogo" => {
                let value = u32::try_from(number)
                    .map_err(|_| ParseError::Malformed("movestogo overflow"))?;
                if value == 0 {
                    return Err(ParseError::Malformed("movestogo must be positive"));
                }
                set_once(&mut limits.moves_to_go, value)?;
            }
            _ => unreachable!(),
        }
        i += 2;
    }
    let has_clock = limits.white_time_ms.is_some() || limits.black_time_ms.is_some();
    if limits.white_time_ms.is_some() != limits.black_time_ms.is_some() {
        return Err(ParseError::Malformed("clock mode requires wtime and btime"));
    }
    if !has_clock
        && (limits.white_increment_ms.is_some()
            || limits.black_increment_ms.is_some()
            || limits.moves_to_go.is_some())
    {
        return Err(ParseError::Malformed("increments/movestogo require clocks"));
    }
    if limits.movetime_ms.is_some() && has_clock {
        return Err(ParseError::Malformed(
            "movetime and clock modes are distinct",
        ));
    }
    // Infinite owns bestmove output until stop even if a finite work objective
    // has completed. Its time/ponder conflicts retain the existing policy.
    if limits.infinite && (limits.ponder || limits.movetime_ms.is_some() || has_clock) {
        return Err(ParseError::Malformed(
            "infinite cannot be combined with ponder or time limits",
        ));
    }
    Ok(Command::Go(limits))
}

fn set_once<T>(slot: &mut Option<T>, value: T) -> Result<(), ParseError> {
    if slot.is_some() {
        return Err(ParseError::Malformed("duplicate go field"));
    }
    *slot = Some(value);
    Ok(())
}
