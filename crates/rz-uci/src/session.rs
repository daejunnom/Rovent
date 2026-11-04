use crate::bridge::NodeBudgetSemantics;
use crate::parser::{
    Command, GoLimits, ParserLimits, PositionSpec, ZeroNodeLimit, is_move_text,
    parse_with_zero_node_limit,
};
use rz_contracts::rovent::SearchGoal;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::num::NonZeroU16;
use std::sync::Arc;

const MAX_OPTIONS: usize = 64;

/// Rules validates a complete temporary position, including all transitions,
/// before Session commits it. Snapshot owns/pins all state needed by a worker.
pub trait PositionPort {
    type Snapshot: Clone;
    type Error: fmt::Display;

    fn prepare(&self, spec: &PositionSpec)
    -> Result<PreparedPosition<Self::Snapshot>, Self::Error>;
}

#[derive(Clone, Debug)]
pub struct PreparedPosition<S> {
    pub snapshot: S,
    /// Duplicate-free Rules order. Session checks text and bounded cardinality.
    pub legal_moves: Vec<String>,
    /// Exact Rules terminal, not evaluation failure or lack of history.
    pub exact_terminal: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineIdentity {
    pub name: String,
    pub author: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OptionKind {
    Check {
        default: bool,
    },
    Spin {
        default: i64,
        min: i64,
        max: i64,
    },
    String {
        default: String,
        max_bytes: usize,
    },
    Combo {
        default: String,
        choices: Vec<String>,
    },
    Button,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptionSpec {
    pub name: String,
    pub kind: OptionKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OptionValue {
    Check(bool),
    Spin(i64),
    String(String),
    Combo(String),
}

/// Private Arc identity prevents fabricated, cross-session, or recycled tickets.
/// The integration adapter maps this B-local token to shared generation IDs.
#[derive(Clone, Debug)]
pub struct SearchTicket(Arc<TicketIdentity>);

#[derive(Debug)]
struct TicketIdentity;

impl PartialEq for SearchTicket {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for SearchTicket {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelReason {
    Stop,
    Deadline,
    ReplacedPosition,
    ReplacedSearch,
    NewGame,
    Quit,
    EndOfInput,
}

#[derive(Clone, Debug)]
pub enum Effect<S> {
    Start {
        ticket: SearchTicket,
        snapshot: S,
        limits: GoLimits,
        options: BTreeMap<String, OptionValue>,
    },
    Cancel {
        ticket: SearchTicket,
        reason: CancelReason,
    },
    /// Clear the game namespace/history/search/correction state and invalidate
    /// its old raw-cache generation while retaining immutable model weights.
    NewGame,
    /// Apply validated options before starting another worker.
    OptionChanged {
        name: String,
        value: OptionValue,
    },
    Button {
        name: String,
    },
    /// Stop dispatching new work, then use the runtime's finite drain contract.
    Shutdown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: &'static str,
    pub message: String,
}

/// Keep protocol and diagnostics separate when writing stdout and stderr.
#[derive(Debug)]
pub struct SessionResult<S> {
    pub protocol: Vec<String>,
    pub diagnostics: Vec<Diagnostic>,
    pub effects: Vec<Effect<S>>,
    pub accepted: bool,
}

impl<S> Default for SessionResult<S> {
    fn default() -> Self {
        Self {
            protocol: Vec::new(),
            diagnostics: Vec::new(),
            effects: Vec::new(),
            accepted: true,
        }
    }
}

impl<S> SessionResult<S> {
    fn reject(code: &'static str, message: impl Into<String>) -> Self {
        let mut out = Self::default();
        out.diagnose(code, message);
        out.accepted = false;
        out
    }

    fn diagnose(&mut self, code: &'static str, message: impl Into<String>) {
        self.diagnostics.push(Diagnostic {
            code,
            message: message.into(),
        });
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionError(pub String);

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for SessionError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SearchCompletion {
    Completed { bestmove: Option<String> },
    Failed { code: String, message: String },
}

struct ActiveSearch {
    ticket: SearchTicket,
    /// Sealed once against Rules, in Rules order. None is unrestricted.
    allowed_root_moves: Option<Arc<[String]>>,
    /// Only validated legal moves enter this slot.
    bestmove: Option<String>,
    infinite: bool,
    worker_started: bool,
    /// Worker completion is consumed once even when infinite keeps the output
    /// owner alive while waiting for stop. It never restarts computation.
    completion_consumed: bool,
}

pub struct Session<P: PositionPort> {
    port: P,
    identity: EngineIdentity,
    parser_limits: ParserLimits,
    options: Vec<OptionSpec>,
    values: BTreeMap<String, OptionValue>,
    position: PreparedPosition<P::Snapshot>,
    active: Option<ActiveSearch>,
    goal_capability: Option<NonZeroU16>,
    node_budget_semantics: NodeBudgetSemantics,
    closed: bool,
}

impl<P: PositionPort> Session<P> {
    pub fn new(
        port: P,
        identity: EngineIdentity,
        options: Vec<OptionSpec>,
        parser_limits: ParserLimits,
    ) -> Result<Self, SessionError> {
        if !valid_text(&identity.name, 128, false) || !valid_text(&identity.author, 128, false) {
            return Err(SessionError(
                "engine identity requires bounded nonempty ASCII text".into(),
            ));
        }
        if options.len() > MAX_OPTIONS {
            return Err(SessionError("too many registered UCI options".into()));
        }
        let mut names = HashSet::new();
        let mut values = BTreeMap::new();
        for spec in &options {
            validate_option_spec(spec, parser_limits)?;
            if !names.insert(spec.name.to_ascii_lowercase()) {
                return Err(SessionError("duplicate UCI option name".into()));
            }
            if let Some(value) = default_value(&spec.kind) {
                values.insert(spec.name.clone(), value);
            }
        }
        let position = port
            .prepare(&PositionSpec::default())
            .map_err(|err| SessionError(format!("initial position rejected: {err}")))?;
        validate_prepared(&position, parser_limits.max_legal_moves)?;
        Ok(Self {
            port,
            identity,
            parser_limits,
            options,
            values,
            position,
            active: None,
            goal_capability: None,
            node_budget_semantics: NodeBudgetSemantics::LegacySimulations,
            closed: false,
        })
    }

    pub fn snapshot(&self) -> &P::Snapshot {
        &self.position.snapshot
    }
    pub fn legal_moves(&self) -> &[String] {
        &self.position.legal_moves
    }
    pub fn option_values(&self) -> &BTreeMap<String, OptionValue> {
        &self.values
    }
    pub fn is_closed(&self) -> bool {
        self.closed
    }
    pub fn parser_limits(&self) -> ParserLimits {
        self.parser_limits
    }
    /// Only a startup-selected driver that implements the chess goals should
    /// register this capability. Legacy sessions reject explicit goals.
    pub fn set_search_goal_capability(
        &mut self,
        max_plies: Option<NonZeroU16>,
    ) -> Result<(), SessionError> {
        if self.closed || self.active.is_some() {
            return Err(SessionError(
                "search goal capability requires an idle open session".into(),
            ));
        }
        self.goal_capability = max_plies;
        Ok(())
    }
    pub fn search_goal_capability(&self) -> Option<NonZeroU16> {
        self.goal_capability
    }

    /// The selected driver registers its node unit before a search starts.
    /// Changing the unit cannot reinterpret an already admitted request.
    pub fn set_node_budget_semantics(
        &mut self,
        semantics: NodeBudgetSemantics,
    ) -> Result<(), SessionError> {
        if self.closed || self.active.is_some() {
            return Err(SessionError(
                "node budget semantics require an idle open session".into(),
            ));
        }
        self.node_budget_semantics = semantics;
        Ok(())
    }

    pub fn node_budget_semantics(&self) -> NodeBudgetSemantics {
        self.node_budget_semantics
    }
    pub fn active_ticket(&self) -> Option<SearchTicket> {
        self.active.as_ref().map(|a| a.ticket.clone())
    }

    /// Same-ticket concrete-owner transition. Only output ownership changes:
    /// no new worker, root, credit or deadline is created. A result already
    /// consumed while pondering is released once without canceling its work.
    pub(crate) fn release_ponder_hit(
        &mut self,
        ticket: &SearchTicket,
    ) -> SessionResult<P::Snapshot> {
        if self.closed || !self.matches(ticket) {
            return stale();
        }
        let active = self.active.as_mut().expect("matching original ticket");
        if !active.infinite {
            return stale();
        }
        active.infinite = false;
        let completed = active.completion_consumed;
        let mut out = SessionResult::default();
        if completed {
            self.finish(&mut out, None);
        }
        out
    }

    pub fn handle_line(&mut self, line: &str) -> SessionResult<P::Snapshot> {
        if self.closed {
            return SessionResult::reject("SessionClosed", "UCI session is closed");
        }
        let zero_nodes = match self.node_budget_semantics {
            NodeBudgetSemantics::LegacySimulations => ZeroNodeLimit::Reject,
            NodeBudgetSemantics::RoventPositionVisits => ZeroNodeLimit::Allow,
        };
        match parse_with_zero_node_limit(line, self.parser_limits, zero_nodes) {
            Ok(command) => self.handle_command(command),
            Err(err) => SessionResult::reject("InvalidCommand", err.to_string()),
        }
    }

    /// Internal owner path for a parser-validated command whose output/time
    /// ownership has been normalized without discarding its root constraints.
    pub(crate) fn handle_command(&mut self, command: Command) -> SessionResult<P::Snapshot> {
        if self.closed {
            return SessionResult::reject("SessionClosed", "UCI session is closed");
        }
        self.dispatch(command)
    }

    /// Untimed progress wrapper. Timed adapters must use
    /// [`Self::progress_with_guard`] to close validation-to-commit races.
    pub fn progress(
        &mut self,
        ticket: &SearchTicket,
        bestmove: &str,
    ) -> SessionResult<P::Snapshot> {
        self.progress_with_guard(ticket, bestmove, || true)
    }

    /// Validate and own the candidate before checking admission exactly once
    /// immediately before committing it. A rejected guard leaves the active
    /// ticket and previous candidate unchanged. Invalid/stale progress can be
    /// rejected before invoking the guard because it cannot mutate the session.
    pub fn progress_with_guard(
        &mut self,
        ticket: &SearchTicket,
        bestmove: &str,
        admit: impl FnOnce() -> bool,
    ) -> SessionResult<P::Snapshot> {
        if !self.matches_work(ticket) {
            return stale();
        }
        if !self.valid_bestmove(bestmove) {
            return SessionResult::reject(
                "IllegalBestMove",
                "search progress is not a legal root move",
            );
        }
        let bestmove = bestmove.to_owned();
        if !admit() {
            return SessionResult::reject(
                "AcceptanceClosed",
                "search admission closed before progress commit",
            );
        }
        self.active
            .as_mut()
            .expect("matching active ticket")
            .bestmove = Some(bestmove);
        SessionResult::default()
    }

    pub fn complete(
        &mut self,
        ticket: &SearchTicket,
        completion: SearchCompletion,
    ) -> SessionResult<P::Snapshot> {
        self.complete_with_guard(ticket, completion, || true)
    }

    /// Prepare the candidate and original failure diagnostics before checking
    /// admission exactly once at the commit boundary. Guard rejection never
    /// consumes the ticket, finalizes output, or changes the previous candidate.
    /// Failed/illegal completions pass through this same gate and preserve their
    /// diagnostic cause even when admission closes during validation.
    pub fn complete_with_guard(
        &mut self,
        ticket: &SearchTicket,
        completion: SearchCompletion,
        admit: impl FnOnce() -> bool,
    ) -> SessionResult<P::Snapshot> {
        if !self.matches_work(ticket) {
            return stale();
        }
        let mut out = SessionResult::default();
        let bestmove = match completion {
            SearchCompletion::Completed {
                bestmove: Some(bestmove),
            } if self.valid_bestmove(&bestmove) => Some(bestmove),
            SearchCompletion::Completed { bestmove: Some(_) } => {
                out.diagnose(
                    "IllegalBestMove",
                    "search completion is not a legal root move",
                );
                out.accepted = false;
                None
            }
            SearchCompletion::Completed { bestmove: None } => None,
            SearchCompletion::Failed { code, message } => {
                out.diagnose("SearchFailed", format!("{code}: {message}"));
                out.accepted = false;
                None
            }
        };
        if !admit() {
            out.diagnose(
                "AcceptanceClosed",
                "search admission closed before completion commit",
            );
            out.accepted = false;
            return out;
        }
        let active = self.active.as_mut().expect("matching active ticket");
        active.completion_consumed = true;
        if let Some(bestmove) = bestmove {
            active.bestmove = Some(bestmove);
        }
        // Finite resource limits can finish an infinite worker, but UCI still
        // owns output until stop/deadline. A failed worker uses the explicit
        // diagnosed failure policy and terminates with a checked legal move.
        if !out.accepted || !active.infinite {
            self.finish(&mut out, None);
        }
        out
    }

    /// Call at the result deadline after closing backup/result admission.
    /// Repeated expiry or a delayed completion cannot emit a second bestmove.
    pub fn expire(&mut self, ticket: &SearchTicket) -> SessionResult<P::Snapshot> {
        if !self.matches(ticket) {
            return stale();
        }
        let mut out = SessionResult::default();
        self.finish(&mut out, Some(CancelReason::Deadline));
        out
    }

    /// EOF has quit semantics; outstanding physical work still needs bounded
    /// runtime drain. This method never invents a rule terminal or search score.
    pub fn end_of_input(&mut self) -> SessionResult<P::Snapshot> {
        self.close(CancelReason::EndOfInput)
    }

    fn dispatch(&mut self, command: Command) -> SessionResult<P::Snapshot> {
        let mut out = SessionResult::default();
        match command {
            Command::Uci => {
                out.protocol.push(format!("id name {}", self.identity.name));
                out.protocol
                    .push(format!("id author {}", self.identity.author));
                out.protocol.extend(self.options.iter().map(option_line));
                out.protocol.push("uciok".into());
            }
            Command::IsReady => out.protocol.push("readyok".into()),
            Command::SetOption { name, value } => return self.set_option(&name, value.as_deref()),
            Command::Position(spec) => return self.replace_position(&spec, false),
            Command::NewGame => return self.replace_position(&PositionSpec::default(), true),
            Command::Go(mut limits) => {
                if limits.ponder {
                    return SessionResult::reject(
                        "PonderNeedsOwner",
                        "ponder requires the concrete engine owner",
                    );
                }
                let goal = SearchGoal {
                    depth_plies: limits.depth_plies,
                    mate_moves: limits.mate_moves,
                };
                if goal.has_goal() {
                    let Some(max_plies) = self.goal_capability else {
                        return SessionResult::reject(
                            "UnsupportedSearchGoal",
                            "selected driver has no registered depth/mate capability",
                        );
                    };
                    if let Err(error) = goal.validate(max_plies) {
                        return SessionResult::reject("InvalidSearchGoal", error.to_string());
                    }
                }
                // A rejected go never cancels the prior search or consumes its
                // output authority. Validate the whole restriction first.
                let allowed_root_moves = match self.seal_root_moves(limits.searchmoves.as_deref()) {
                    Ok(moves) => moves,
                    Err(error) => {
                        return SessionResult::reject("InvalidSearchMoves", error.to_string());
                    }
                };
                limits.searchmoves = allowed_root_moves.as_ref().map(|moves| moves.to_vec());
                self.cancel(&mut out, CancelReason::ReplacedSearch);
                if self.position.exact_terminal && !limits.infinite {
                    out.protocol.push("bestmove 0000".into());
                } else {
                    let ticket = SearchTicket(Arc::new(TicketIdentity));
                    self.active = Some(ActiveSearch {
                        ticket: ticket.clone(),
                        allowed_root_moves,
                        bestmove: None,
                        infinite: limits.infinite,
                        worker_started: !self.position.exact_terminal,
                        completion_consumed: self.position.exact_terminal,
                    });
                    if !self.position.exact_terminal {
                        out.effects.push(Effect::Start {
                            ticket,
                            snapshot: self.position.snapshot.clone(),
                            limits,
                            options: self.values.clone(),
                        });
                    }
                }
            }
            Command::PonderHit => {
                return SessionResult::reject(
                    "PonderNeedsOwner",
                    "ponderhit requires the concrete engine owner",
                );
            }
            Command::Stop => self.finish(&mut out, Some(CancelReason::Stop)),
            Command::Quit => return self.close(CancelReason::Quit),
            Command::Unknown(name) => {
                out.diagnose(
                    "UnknownCommand",
                    format!("ignored unsupported command: {name}"),
                );
                out.accepted = false;
            }
        }
        out
    }

    fn replace_position(
        &mut self,
        spec: &PositionSpec,
        new_game: bool,
    ) -> SessionResult<P::Snapshot> {
        // All fallible work precedes cancellation and state mutation.
        let position = match self.port.prepare(spec) {
            Ok(position) => position,
            Err(err) => return SessionResult::reject("PositionRejected", err.to_string()),
        };
        if let Err(err) = validate_prepared(&position, self.parser_limits.max_legal_moves) {
            return SessionResult::reject("InvalidPositionView", err.to_string());
        }
        let mut out = SessionResult::default();
        self.cancel(
            &mut out,
            if new_game {
                CancelReason::NewGame
            } else {
                CancelReason::ReplacedPosition
            },
        );
        self.position = position;
        if new_game {
            out.effects.push(Effect::NewGame);
        }
        out
    }

    fn set_option(&mut self, name: &str, value: Option<&str>) -> SessionResult<P::Snapshot> {
        if self.active.is_some() {
            return SessionResult::reject("SearchActive", "setoption requires an idle session");
        }
        let Some(spec) = self
            .options
            .iter()
            .find(|spec| spec.name.eq_ignore_ascii_case(name))
        else {
            return SessionResult::reject("UnknownOption", format!("unsupported option: {name}"));
        };
        let value = match checked_option_value(&spec.kind, value) {
            Ok(value) => value,
            Err(err) => return SessionResult::reject("InvalidOption", err.to_string()),
        };
        let mut out = SessionResult::default();
        if let Some(value) = value {
            self.values.insert(spec.name.clone(), value.clone());
            out.effects.push(Effect::OptionChanged {
                name: spec.name.clone(),
                value,
            });
        } else {
            out.effects.push(Effect::Button {
                name: spec.name.clone(),
            });
        }
        out
    }

    fn matches(&self, ticket: &SearchTicket) -> bool {
        !self.closed
            && self
                .active
                .as_ref()
                .is_some_and(|active| active.ticket == *ticket)
    }

    fn matches_work(&self, ticket: &SearchTicket) -> bool {
        self.matches(ticket)
            && !self
                .active
                .as_ref()
                .expect("matching active ticket")
                .completion_consumed
    }

    fn valid_bestmove(&self, bestmove: &str) -> bool {
        let allowed = self
            .active
            .as_ref()
            .and_then(|active| active.allowed_root_moves.as_deref())
            .unwrap_or(&self.position.legal_moves);
        is_move_text(bestmove) && allowed.iter().any(|m| m == bestmove)
    }

    fn seal_root_moves(
        &self,
        requested: Option<&[String]>,
    ) -> Result<Option<Arc<[String]>>, SessionError> {
        let Some(requested) = requested else {
            return Ok(None);
        };
        if requested.is_empty() || requested.len() > self.parser_limits.max_legal_moves {
            return Err(SessionError(
                "explicit root restriction must be nonempty and within the move bound".into(),
            ));
        }
        if requested
            .iter()
            .any(|movement| !self.position.legal_moves.contains(movement))
        {
            return Err(SessionError(
                "searchmoves contains a move outside the exact Rules root legal list".into(),
            ));
        }
        let requested: HashSet<&str> = requested.iter().map(String::as_str).collect();
        let canonical: Vec<_> = self
            .position
            .legal_moves
            .iter()
            .filter(|movement| requested.contains(movement.as_str()))
            .cloned()
            .collect();
        Ok(Some(canonical.into()))
    }

    fn cancel(&mut self, out: &mut SessionResult<P::Snapshot>, reason: CancelReason) {
        if let Some(active) = self.active.take() {
            if active.worker_started {
                out.effects.push(Effect::Cancel {
                    ticket: active.ticket,
                    reason,
                });
            }
        }
    }

    fn finish(&mut self, out: &mut SessionResult<P::Snapshot>, reason: Option<CancelReason>) {
        let Some(active) = self.active.take() else {
            return;
        };
        if let Some(reason) = reason.filter(|_| active.worker_started) {
            out.effects.push(Effect::Cancel {
                ticket: active.ticket,
                reason,
            });
        }
        if self.position.exact_terminal {
            out.protocol.push("bestmove 0000".into());
            return;
        }
        let bestmove = match active.bestmove {
            Some(bestmove) => bestmove,
            None => {
                out.diagnose(
                    "LegalFallback",
                    "no completed valid root move; using first ordered legal move",
                );
                // Session construction/replacement rejects ongoing empty legal views.
                active
                    .allowed_root_moves
                    .as_ref()
                    .map_or(&self.position.legal_moves[0], |moves| &moves[0])
                    .clone()
            }
        };
        out.protocol.push(format!("bestmove {bestmove}"));
    }

    fn close(&mut self, reason: CancelReason) -> SessionResult<P::Snapshot> {
        let mut out = SessionResult::default();
        if self.closed {
            return out;
        }
        self.cancel(&mut out, reason);
        self.closed = true;
        out.effects.push(Effect::Shutdown);
        out
    }
}

fn stale<S>() -> SessionResult<S> {
    SessionResult::reject(
        "StaleSearch",
        "ignored inactive, duplicated, or replaced search result",
    )
}

fn validate_prepared<S>(
    position: &PreparedPosition<S>,
    max_legal_moves: usize,
) -> Result<(), SessionError> {
    if position.legal_moves.len() > max_legal_moves {
        return Err(SessionError(
            "legal move view exceeds configured finite move bound".into(),
        ));
    }
    if !position.exact_terminal && position.legal_moves.is_empty() {
        return Err(SessionError(
            "ongoing position has no legal moves; Rules terminal classification is required".into(),
        ));
    }
    let mut seen = HashSet::new();
    for m in &position.legal_moves {
        if !is_move_text(m) || !seen.insert(m) {
            return Err(SessionError(
                "legal move view contains invalid or duplicate UCI text".into(),
            ));
        }
    }
    Ok(())
}

fn valid_text(text: &str, max_bytes: usize, allow_empty: bool) -> bool {
    text.len() <= max_bytes
        && (allow_empty || !text.is_empty())
        && text.is_ascii()
        && text.bytes().all(|b| (b' '..=b'~').contains(&b))
        && text.trim() == text
}

fn validate_option_spec(spec: &OptionSpec, limits: ParserLimits) -> Result<(), SessionError> {
    if !valid_text(&spec.name, limits.max_option_name_bytes, false)
        || spec.name.split(' ').any(|token| {
            matches!(
                token,
                "name" | "value" | "type" | "default" | "min" | "max" | "var"
            )
        })
    {
        return Err(SessionError("invalid registered option name".into()));
    }
    let good = match &spec.kind {
        OptionKind::Check { .. } | OptionKind::Button => true,
        OptionKind::Spin { default, min, max } => min <= default && default <= max,
        OptionKind::String { default, max_bytes } => {
            *max_bytes <= limits.max_option_value_bytes && valid_text(default, *max_bytes, true)
        }
        OptionKind::Combo { default, choices } => {
            !choices.is_empty()
                && choices.len() <= MAX_OPTIONS
                && choices.contains(default)
                && choices.iter().all(|choice| {
                    valid_text(choice, limits.max_option_value_bytes, false)
                        && !choice.split(' ').any(|token| token == "var")
                })
                && choices.iter().collect::<HashSet<_>>().len() == choices.len()
        }
    };
    if good {
        Ok(())
    } else {
        Err(SessionError(
            "invalid registered option bounds/default".into(),
        ))
    }
}

fn default_value(kind: &OptionKind) -> Option<OptionValue> {
    match kind {
        OptionKind::Check { default } => Some(OptionValue::Check(*default)),
        OptionKind::Spin { default, .. } => Some(OptionValue::Spin(*default)),
        OptionKind::String { default, .. } => Some(OptionValue::String(default.clone())),
        OptionKind::Combo { default, .. } => Some(OptionValue::Combo(default.clone())),
        OptionKind::Button => None,
    }
}

fn checked_option_value(
    kind: &OptionKind,
    text: Option<&str>,
) -> Result<Option<OptionValue>, SessionError> {
    let invalid = || SessionError("option value is missing or outside declared bounds".into());
    let value = match kind {
        OptionKind::Button if text.is_none() => return Ok(None),
        OptionKind::Button => return Err(invalid()),
        OptionKind::Check { .. } => match text {
            Some("true") => OptionValue::Check(true),
            Some("false") => OptionValue::Check(false),
            _ => return Err(invalid()),
        },
        OptionKind::Spin { min, max, .. } => {
            let text = text.ok_or_else(invalid)?;
            let value = text.parse::<i64>().map_err(|_| invalid())?;
            if value < *min || value > *max {
                return Err(invalid());
            }
            OptionValue::Spin(value)
        }
        OptionKind::String { max_bytes, .. } => {
            let text = text.ok_or_else(invalid)?;
            let text = if text == "<empty>" { "" } else { text };
            if !valid_text(text, *max_bytes, true) {
                return Err(invalid());
            }
            OptionValue::String(text.to_owned())
        }
        OptionKind::Combo { choices, .. } => {
            let text = text.ok_or_else(invalid)?;
            if !choices.iter().any(|choice| choice == text) {
                return Err(invalid());
            }
            OptionValue::Combo(text.to_owned())
        }
    };
    Ok(Some(value))
}

fn option_line(spec: &OptionSpec) -> String {
    let tail = match &spec.kind {
        OptionKind::Check { default } => format!("check default {default}"),
        OptionKind::Spin { default, min, max } => {
            format!("spin default {default} min {min} max {max}")
        }
        OptionKind::String { default, .. } => format!(
            "string default {}",
            if default.is_empty() {
                "<empty>"
            } else {
                default
            }
        ),
        OptionKind::Combo { default, choices } => {
            let mut tail = format!("combo default {default}");
            for choice in choices {
                tail.push_str(&format!(" var {choice}"));
            }
            tail
        }
        OptionKind::Button => "button".into(),
    };
    format!("option name {} type {tail}", spec.name)
}
