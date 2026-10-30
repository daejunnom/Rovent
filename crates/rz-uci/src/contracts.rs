//! 공통 0.1 game/root·registry·clock·cancel 권한과 UCI 출력 소유권의 연결.
//! 실제 Rules/registry는 caller가 공급한다. 이 owner는 체스 상태나 digest를 만들지
//! 않으며 Runtime의 raw-cache namespace 초기화와 물리 buffer drain을 대체하지 않는다.

use std::{collections::VecDeque, sync::Arc, time::Instant};

use rz_contracts::{
    AcceptanceScope, Color, ContractError, ErrorCode, GameGeneration, ProcessEpoch, RootGeneration,
    Stage,
};
use rz_search::contract_time::{
    ContractCancellation, ContractClock, ContractDeadlines, InstantClock,
};
use rz_search::driver::SearchControl;
use rz_search::time::TimeBudgetError;

use crate::bridge::{
    BudgetKind, BuildSearchError, BuildSearchSettings, NodeBudgetSemantics, PreparedSearchBinding,
    SearchBinding, SideToMove,
};
use crate::parser::{ZeroNodeLimit, parse_with_zero_node_limit};
use crate::{
    CancelReason, Command, Diagnostic, Effect, Event, GoLimits, PositionPort, SearchCompletion,
    SearchTicket, Session, SessionResult, handle_event,
};

/// UCI publication authority is independent of the selected search algorithm.
/// Evaluation scopes retain the unchanged 0.1 model contract; native CPU/PALS
/// drivers use a search identity and never fabricate a neural model descriptor.
#[derive(Clone, Copy, Debug)]
pub enum SessionScope {
    Evaluation(AcceptanceScope),
    Search(rz_contracts::pals::SearchAuthority),
}
impl PartialEq for SessionScope {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Evaluation(a), Self::Evaluation(b)) => {
                a.game == b.game
                    && a.root == b.root
                    && a.model == b.model
                    && a.encoding == b.encoding
                    && a.backend == b.backend
            }
            (Self::Search(a), Self::Search(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for SessionScope {}
impl SessionScope {
    pub fn game(self) -> GameGeneration {
        match self {
            Self::Evaluation(s) => s.game,
            Self::Search(s) => s.game,
        }
    }
    pub fn root(self) -> RootGeneration {
        match self {
            Self::Evaluation(s) => s.root,
            Self::Search(s) => s.root,
        }
    }
    fn set_generations(&mut self, game: GameGeneration, root: RootGeneration) {
        match self {
            Self::Evaluation(s) => {
                s.game = game;
                s.root = root;
            }
            Self::Search(s) => {
                s.game = game;
                s.root = root;
            }
        }
    }
    pub fn evaluation(self) -> Option<AcceptanceScope> {
        match self {
            Self::Evaluation(s) => Some(s),
            Self::Search(_) => None,
        }
    }
}

/// Typed contract causes remain alongside protocol/legacy backend diagnostics.
/// An event-loop handler forwards `session` only after recording `errors`.
#[derive(Debug)]
pub struct ContractSessionOutcome<S> {
    pub session: SessionResult<S>,
    pub errors: Vec<ContractError>,
}

impl<S> From<SessionResult<S>> for ContractSessionOutcome<S> {
    fn from(session: SessionResult<S>) -> Self {
        Self {
            session,
            errors: Vec::new(),
        }
    }
}

struct CurrentSearch {
    // Same actual UCI Start used by the active driver; absent for legacy paths.
    original_start: Option<Arc<crate::engine::DriverStartSource>>,
    binding: SearchBinding,
    cancellation: ContractCancellation,
    deadlines: ContractDeadlines,
    scope: SessionScope,
    infinite: bool,
    work_completed: bool,
}

struct RetiredCancellation {
    ticket: SearchTicket,
    reason: CancelReason,
}

/// One Session owner's acceptance authority. Epoch, model/encoding/backend
/// handles, and initial generations are explicit inputs from their real owners.
/// Reusing this owner with an independently active Session is rejected.
pub struct ContractSessionOwner {
    scope: SessionScope,
    epoch: ProcessEpoch,
    clock: InstantClock,
    settings: BuildSearchSettings,
    last_dispatch: Instant,
    ticket: Option<SearchTicket>,
    current: Option<CurrentSearch>,
    // Opaque identities only: no position, search work or reusable authority.
    // A service supplies its existing worker bound; old standalone constructors
    // retain their original typed-Stale behavior with a zero-length history.
    retired_cancellations: VecDeque<RetiredCancellation>,
    max_retired_cancellations: usize,
}

impl ContractSessionOwner {
    pub fn new(
        scope: AcceptanceScope,
        epoch: ProcessEpoch,
        clock: InstantClock,
        settings: BuildSearchSettings,
    ) -> Result<Self, ContractError> {
        Self::new_scope(SessionScope::Evaluation(scope), epoch, clock, settings)
    }
    pub fn new_scope(
        scope: SessionScope,
        epoch: ProcessEpoch,
        clock: InstantClock,
        settings: BuildSearchSettings,
    ) -> Result<Self, ContractError> {
        Self::new_scope_with_cancel_history(scope, epoch, clock, settings, 0)
    }

    /// Explicit service-only diagnostic history, bounded by its worker limit.
    /// Remembering a canceled identity never reopens result/work admission.
    /// The zero-history mode preserves the standalone constructor behavior.
    pub fn new_scope_with_cancel_history(
        scope: SessionScope,
        epoch: ProcessEpoch,
        clock: InstantClock,
        settings: BuildSearchSettings,
        max_retired_cancellations: usize,
    ) -> Result<Self, ContractError> {
        if clock.domain().0 != epoch {
            return Err(error(
                ErrorCode::IdentityMismatch,
                "process epoch and UCI clock domain differ",
            ));
        }
        if let SessionScope::Search(authority) = scope
            && authority.epoch != epoch
        {
            return Err(error(
                ErrorCode::IdentityMismatch,
                "search identity and UCI process epoch differ",
            ));
        }
        clock.now()?;
        let mut retired_cancellations = VecDeque::new();
        retired_cancellations
            .try_reserve_exact(max_retired_cancellations)
            .map_err(|_| {
                error(
                    ErrorCode::ResourceExhausted,
                    "bounded canceled-ticket diagnostic history allocation failed",
                )
            })?;
        Ok(Self {
            scope,
            epoch,
            clock,
            settings,
            last_dispatch: Instant::now(),
            ticket: None,
            current: None,
            retired_cancellations,
            max_retired_cancellations,
        })
    }

    pub fn scope(&self) -> AcceptanceScope {
        self.scope
            .evaluation()
            .expect("evaluation-only registry accessor")
    }
    pub fn lifecycle_scope(&self) -> SessionScope {
        self.scope
    }
    pub fn process_epoch(&self) -> ProcessEpoch {
        self.epoch
    }
    pub fn clock(&self) -> &InstantClock {
        &self.clock
    }
    pub fn active_ticket(&self) -> Option<&SearchTicket> {
        self.ticket.as_ref()
    }
    pub fn active_scope(&self) -> Option<AcceptanceScope> {
        self.current.as_ref().and_then(|c| c.scope.evaluation())
    }
    pub fn active_lifecycle_scope(&self) -> Option<SessionScope> {
        self.current.as_ref().map(|c| c.scope)
    }
    pub fn active_binding(&self) -> Option<&SearchBinding> {
        self.current.as_ref().map(|c| {
            mirror_cancel(c);
            &c.binding
        })
    }
    /// Bind only the original active ticket's concrete Start owner. This is
    /// optional for legacy drivers and never constructs search authority.
    pub(crate) fn bind_original_start(
        &mut self,
        ticket: &SearchTicket,
        source: Arc<crate::engine::DriverStartSource>,
    ) -> Result<(), ContractError> {
        let current = self.current.as_mut().ok_or_else(|| error(ErrorCode::Stale, "original Start has no active owner"))?;
        if current.binding.ticket() != ticket || source.ticket() != ticket || current.original_start.is_some() {
            return Err(error(ErrorCode::IdentityMismatch, "original Start differs from active owner"));
        }
        current.original_start = Some(source);
        Ok(())
    }
    pub fn control(&self) -> Option<&SearchControl> {
        self.active_binding().map(SearchBinding::control)
    }
    pub fn cancellation(&self) -> Option<&ContractCancellation> {
        self.current.as_ref().map(|c| &c.cancellation)
    }
    pub fn deadlines(&self) -> Option<&ContractDeadlines> {
        self.current.as_ref().map(|c| &c.deadlines)
    }

    /// Rovent's effective clock is held by the actual Start owner. The common
    /// binding and every submitted request keep their original deadlines; this
    /// method releases only the same ticket's infinite output hold.
    pub(crate) fn release_ponder_hit<P: PositionPort>(
        &mut self,
        session: &mut Session<P>,
        ticket: &SearchTicket,
    ) -> ContractSessionOutcome<P::Snapshot> {
        if let Err(err) = self.ensure_session(session) {
            return rejected(err);
        }
        let Some(current) = self.matching_current(ticket) else {
            return stale(None);
        };
        if current.binding.node_budget_semantics() != NodeBudgetSemantics::RoventPositionVisits
            || !current.infinite
        {
            return rejected(error(
                ErrorCode::UnsupportedContract,
                "normal ponder hit requires the actual Rovent Start",
            ));
        }
        let out = session.release_ponder_hit(ticket);
        if out.accepted {
            current.infinite = false;
        }
        self.synchronize(session);
        out.into()
    }

    /// Registry replacement is explicit and idle-only. Caller must provide the
    /// live handles/backend while preserving this owner's game/root counters.
    pub fn set_registry<P: PositionPort>(
        &mut self,
        session: &Session<P>,
        scope: AcceptanceScope,
    ) -> Result<(), ContractError> {
        self.ensure_session(session)?;
        if session.active_ticket().is_some() {
            return Err(error(
                ErrorCode::InvalidInput,
                "registry update requires an idle UCI session",
            ));
        }
        if scope.game != self.scope.game() || scope.root != self.scope.root() {
            return Err(error(
                ErrorCode::IdentityMismatch,
                "registry update cannot replace game/root generations",
            ));
        }
        if self.scope.evaluation().is_none() {
            return Err(error(
                ErrorCode::UnsupportedContract,
                "search driver has no model registry",
            ));
        }
        self.scope = SessionScope::Evaluation(scope);
        Ok(())
    }

    /// Capture go's original Instant before parsing, Rules preparation, or side
    /// validation. Accepted root commands advance checked generations exactly
    /// once; malformed and rejected Rules transitions preserve prior authority.
    pub fn handle_line<P, F>(
        &mut self,
        session: &mut Session<P>,
        line: &str,
        side: F,
    ) -> ContractSessionOutcome<P::Snapshot>
    where
        P: PositionPort,
        F: Fn(&P::Snapshot) -> Result<Color, ContractError>,
    {
        let start = Instant::now();
        if let Err(err) = self.ensure_session(session) {
            return rejected(err);
        }
        let zero_nodes = match session.node_budget_semantics() {
            NodeBudgetSemantics::LegacySimulations => ZeroNodeLimit::Reject,
            NodeBudgetSemantics::RoventPositionVisits => ZeroNodeLimit::Allow,
        };
        let command = match parse_with_zero_node_limit(line, session.parser_limits(), zero_nodes) {
            Ok(command) => command,
            Err(_) => return session.handle_line(line).into(),
        };
        self.handle_command_at(session, command, side, start)
    }

    /// Concrete UCI owner only: `limits` came from the bounded parser. Ponder
    /// normalizes clock/output ownership while retaining the exact root goals.
    pub(crate) fn handle_normalized_go<P, F>(
        &mut self,
        session: &mut Session<P>,
        limits: GoLimits,
        side: F,
    ) -> ContractSessionOutcome<P::Snapshot>
    where
        P: PositionPort,
        F: Fn(&P::Snapshot) -> Result<Color, ContractError>,
    {
        let start = Instant::now();
        if let Err(err) = self.ensure_session(session) {
            return rejected(err);
        }
        self.handle_command_at(session, Command::Go(limits), side, start)
    }

    fn handle_command_at<P, F>(
        &mut self,
        session: &mut Session<P>,
        command: Command,
        side: F,
        start: Instant,
    ) -> ContractSessionOutcome<P::Snapshot>
    where
        P: PositionPort,
        F: Fn(&P::Snapshot) -> Result<Color, ContractError>,
    {
        let root_change = matches!(
            command,
            Command::Go(_) | Command::Position(_) | Command::NewGame
        );
        let new_game = matches!(command, Command::NewGame);
        let stops_search = matches!(command, Command::Stop | Command::Quit);
        let next = if root_change {
            match self.next_generations(new_game) {
                Ok(next) => Some(next),
                Err(err) => return rejected(err),
            }
        } else {
            None
        };
        // Rovent's UCI nodes are position visits, not the transport's simulation
        // bound. Validate and construct its original time/control once, before
        // Session can cancel the prior search or mint a replacement ticket.
        // The legacy path keeps its existing post-Start admission behavior.
        let mut prepared = None;
        if let Command::Go(limits) = &command {
            if session.node_budget_semantics() == NodeBudgetSemantics::RoventPositionVisits {
                let mut build_detail = None;
                let built = side(session.snapshot()).and_then(|color| {
                    let color = match color {
                        Color::White => SideToMove::White,
                        Color::Black => SideToMove::Black,
                    };
                    let binding = PreparedSearchBinding::prepare(
                        limits,
                        color,
                        start,
                        self.settings,
                        session.search_goal_capability(),
                        session.node_budget_semantics(),
                    )
                    .map_err(|err| {
                        build_detail = Some(err.to_string());
                        map_build_error(err)
                    })?;
                    let deadlines = ContractDeadlines::from_budget(&self.clock, binding.budget())?;
                    Ok((binding, deadlines))
                });
                match built {
                    Ok(binding) => prepared = Some(binding),
                    Err(err) => {
                        let mut out = rejected(err);
                        if let Some(message) = build_detail {
                            out.session.diagnostics.push(Diagnostic {
                                code: "SearchConfigurationFailed",
                                message,
                            });
                        }
                        return out;
                    }
                }
            }
        }
        let previous_ticket = self.current.as_ref().map(|c| c.binding.ticket().clone());
        let mut out = ContractSessionOutcome::from(session.handle_command(command));
        if !out.session.accepted {
            return out;
        }
        // Session has already accepted the command and produced concrete
        // effects. Close the exact original Start's optional Collection gate
        // before its common cancel token, never for a rejected command or a
        // natural completion lacking a matching Cancel effect.
        if out.session.effects.iter().any(|effect| matches!(effect,
            Effect::Cancel { ticket, .. } if previous_ticket.as_ref() == Some(ticket))) {
            if let Some(start) = self.current.as_ref().and_then(|current| current.original_start.as_ref()) {
                start.abort_collection();
            }
        }
        if let Some((game, root)) = next {
            self.close_current();
            self.scope.set_generations(game, root);
        }
        if stops_search {
            self.close_current();
        }
        let effects = std::mem::take(&mut out.session.effects);
        for effect in effects {
            if let Effect::Start {
                ticket,
                snapshot,
                limits,
                options,
            } = effect
            {
                let mut build_detail = None;
                let built = if let Some((binding, deadlines)) = prepared.take() {
                    let binding = binding.bind(ticket.clone());
                    let cancellation = ContractCancellation::from_control(binding.control());
                    Ok(CurrentSearch {
                        original_start: None,
                        binding,
                        deadlines,
                        cancellation,
                        scope: self.scope,
                        infinite: limits.infinite,
                        work_completed: false,
                    })
                } else {
                    side(&snapshot).and_then(|color| {
                        let color = match color {
                            Color::White => SideToMove::White,
                            Color::Black => SideToMove::Black,
                        };
                        let binding = match session.search_goal_capability() {
                            Some(max_plies) => SearchBinding::new_with_goal_capability(
                                ticket.clone(),
                                &limits,
                                color,
                                start,
                                self.settings,
                                max_plies,
                            ),
                            None => SearchBinding::new(
                                ticket.clone(),
                                &limits,
                                color,
                                start,
                                self.settings,
                            ),
                        }
                        .map_err(|err| {
                            build_detail = Some(err.to_string());
                            map_build_error(err)
                        })?;
                        let deadlines =
                            ContractDeadlines::from_budget(&self.clock, binding.budget())?;
                        let cancellation = ContractCancellation::from_control(binding.control());
                        Ok(CurrentSearch {
                            original_start: None,
                            binding,
                            deadlines,
                            cancellation,
                            scope: self.scope,
                            infinite: limits.infinite,
                            work_completed: false,
                        })
                    })
                };
                match built {
                    Ok(current) => {
                        if limits.uses_default_resource_limits() {
                            let wall_ms = self.settings.untimed_limit.as_millis();
                            let notice = match current.binding.node_budget_semantics() {
                                NodeBudgetSemantics::LegacySimulations => format!(
                                    "info string go uses configured resource wall at most {wall_ms} ms and simulation limit {}",
                                    self.settings.max_simulations
                                ),
                                NodeBudgetSemantics::RoventPositionVisits => format!(
                                    "info string go uses configured resource wall at most {wall_ms} ms; other search limits follow the registered engine profile"
                                ),
                            };
                            out.session.protocol.push(notice);
                        }
                        self.current = Some(current);
                        out.session.effects.push(Effect::Start {
                            ticket,
                            snapshot,
                            limits,
                            options,
                        });
                    }
                    Err(err) => {
                        let failed = session.complete(
                            &ticket,
                            SearchCompletion::Failed {
                                code: format!("{:?}/{:?}", err.stage, err.code),
                                message: err.detail.into(),
                            },
                        );
                        merge(&mut out.session, failed);
                        attach_error(&mut out, err);
                        if let Some(message) = build_detail {
                            out.session.diagnostics.push(Diagnostic {
                                code: "SearchConfigurationFailed",
                                message,
                            });
                        }
                    }
                }
            } else {
                out.session.effects.push(effect);
            }
        }
        self.remember_actual_cancellations(previous_ticket.as_ref(), &out.session.effects);
        self.synchronize(session);
        out
    }

    /// Owner dequeue entry point. Search callbacks are never admitted through
    /// the untimed default event reducer; final Session guards re-read this clock.
    pub fn handle_event<P: PositionPort>(
        &mut self,
        session: &mut Session<P>,
        event: Event,
    ) -> ContractSessionOutcome<P::Snapshot> {
        self.handle_event_at(session, event, Instant::now())
    }

    /// Trusted owner dispatch anchor, useful for deterministic boundary tests.
    /// It cannot admit using an old producer tick: each final guard also checks
    /// actual Instant::now and uses the later time in the owned clock domain.
    pub fn handle_event_at<P: PositionPort>(
        &mut self,
        session: &mut Session<P>,
        event: Event,
        now: Instant,
    ) -> ContractSessionOutcome<P::Snapshot> {
        if let Err(err) = self.ensure_session(session) {
            return rejected(err);
        }
        let now = now.max(Instant::now()).max(self.last_dispatch);
        self.last_dispatch = now;
        let previous_ticket = self.current.as_ref().map(|c| c.binding.ticket().clone());
        let out = match event {
            Event::Progress { ticket, bestmove } => self.progress(session, &ticket, &bestmove, now),
            Event::Complete { ticket, completion } => {
                self.complete(session, &ticket, completion, now)
            }
            Event::Deadline(ticket) => self.expire(session, &ticket, now),
            Event::Line(_) => {
                return rejected(error(
                    ErrorCode::InvalidInput,
                    "line events require handle_line with the checked Rules side adapter",
                ));
            }
            Event::EndOfInput => {
                if let Some(start) = self.current.as_ref().and_then(|current| current.original_start.as_ref()) {
                    start.abort_collection();
                }
                self.close_current();
                handle_event(session, Event::EndOfInput).into()
            }
            event => handle_event(session, event).into(),
        };
        self.remember_actual_cancellations(previous_ticket.as_ref(), &out.session.effects);
        self.synchronize(session);
        // The common cancellation clone is authoritative even when an invalid
        // candidate returned before its final guard. Mirror it into legacy work.
        if let Some(current) = &self.current {
            mirror_cancel(current);
        }
        out
    }

    fn progress<P: PositionPort>(
        &mut self,
        session: &mut Session<P>,
        ticket: &SearchTicket,
        bestmove: &str,
        now: Instant,
    ) -> ContractSessionOutcome<P::Snapshot> {
        let scope = self.scope;
        let clock = self.clock;
        let Some(current) = self.matching_current(ticket) else {
            return self.late_result(ticket, "Progress", None);
        };
        if current.work_completed {
            return stale(None);
        }
        mirror_cancel(current);
        let mut failure = None;
        let prepared = session.progress_with_guard(ticket, bestmove, || {
            failure = acceptance(current, scope, clock, now).err();
            failure.is_none()
        });
        match failure {
            Some(err) => reject_prepared(session, current, prepared, err),
            None => prepared.into(),
        }
    }

    fn complete<P: PositionPort>(
        &mut self,
        session: &mut Session<P>,
        ticket: &SearchTicket,
        completion: SearchCompletion,
        now: Instant,
    ) -> ContractSessionOutcome<P::Snapshot> {
        let scope = self.scope;
        let clock = self.clock;
        let Some(current) = self.matching_current(ticket) else {
            return self.late_result(ticket, "Complete", Some(completion));
        };
        if current.work_completed {
            return stale(Some(completion));
        }
        mirror_cancel(current);
        let mut failure = None;
        let mut called = false;
        let prepared = session.complete_with_guard(ticket, completion, || {
            called = true;
            failure = acceptance(current, scope, clock, now).err();
            failure.is_none()
        });
        if called && failure.is_none() {
            current.work_completed = true;
        }
        match failure {
            Some(err) => reject_prepared(session, current, prepared, err),
            None => prepared.into(),
        }
    }

    fn expire<P: PositionPort>(
        &mut self,
        session: &mut Session<P>,
        ticket: &SearchTicket,
        now: Instant,
    ) -> ContractSessionOutcome<P::Snapshot> {
        let scope = self.scope;
        let clock = self.clock;
        let Some(current) = self.matching_current(ticket) else {
            return stale(None);
        };
        // A timer owns output finalization, not worker result admission. A
        // canceled worker cannot strand an otherwise current finite UCI owner.
        if !same_scope(current.scope, scope) {
            return rejected(error(
                ErrorCode::Stale,
                "search game/root or registry scope no longer current",
            ));
        }
        let tick = match clock.tick_at(now.max(Instant::now())) {
            Ok(tick) => tick,
            Err(err) => return rejected(err),
        };
        match current.deadlines.hard.accepts(clock.domain(), tick) {
            Ok(()) => rejected(error(
                ErrorCode::InvalidInput,
                "deadline event arrived before hard boundary",
            )),
            Err(err) if err.code == ErrorCode::Expired && current.work_completed => {
                // Only a reached timer may close passive infinite authority.
                // Completed output waiting never becomes a new resource failure.
                if let Some(start) = &current.original_start { start.abort_collection(); }
                current.cancellation.cancel();
                SessionResult::default().into()
            }
            Err(err) if err.code == ErrorCode::Expired => expire_current(session, current, err),
            Err(err) => rejected(err),
        }
    }

    fn ensure_session<P: PositionPort>(&self, session: &Session<P>) -> Result<(), ContractError> {
        if session.active_ticket().as_ref() != self.ticket.as_ref() {
            return Err(error(
                ErrorCode::Stale,
                "Session active ticket belongs to a different lifecycle owner",
            ));
        }
        Ok(())
    }
    fn next_generations(
        &self,
        new_game: bool,
    ) -> Result<(GameGeneration, RootGeneration), ContractError> {
        let root = self
            .scope
            .root()
            .0
            .checked_add(1)
            .ok_or_else(|| error(ErrorCode::ResourceExhausted, "root generation overflow"))?;
        let game =
            if new_game {
                self.scope.game().0.checked_add(1).ok_or_else(|| {
                    error(ErrorCode::ResourceExhausted, "game generation overflow")
                })?
            } else {
                self.scope.game().0
            };
        Ok((GameGeneration(game), RootGeneration(root)))
    }
    fn close_current(&mut self) {
        if let Some(current) = self.current.take() {
            current.cancellation.cancel();
        }
        self.ticket = None;
    }
    fn remember_actual_cancellations<S>(
        &mut self,
        previous_ticket: Option<&SearchTicket>,
        effects: &[Effect<S>],
    ) {
        if self.max_retired_cancellations == 0 {
            return;
        }
        let Some(previous_ticket) = previous_ticket else {
            return;
        };
        // Only a real Session-generated Cancel effect for this owner's prior
        // common binding qualifies. Rejected commands and foreign events cannot
        // mint a diagnostic exemption; natural completion has no Cancel effect.
        for effect in effects {
            let Effect::Cancel { ticket, reason } = effect else {
                continue;
            };
            if ticket != previous_ticket
                || self
                    .retired_cancellations
                    .iter()
                    .any(|c| c.ticket == *ticket)
            {
                continue;
            }
            if self.retired_cancellations.len() == self.max_retired_cancellations {
                self.retired_cancellations.pop_front();
            }
            self.retired_cancellations.push_back(RetiredCancellation {
                ticket: ticket.clone(),
                reason: *reason,
            });
        }
    }
    fn late_result<S>(
        &self,
        ticket: &SearchTicket,
        event_kind: &'static str,
        completion: Option<SearchCompletion>,
    ) -> ContractSessionOutcome<S> {
        // A real failed publication must retain its typed Stale and SearchFailed
        // causes even when its ticket was canceled normally. Physical-unknown
        // publications are fenced by the engine owner before this reducer.
        if !matches!(completion.as_ref(), Some(SearchCompletion::Failed { .. }))
            && let Some(cancellation) = self
                .retired_cancellations
                .iter()
                .find(|c| c.ticket == *ticket)
        {
            let mut out = SessionResult {
                accepted: false,
                ..SessionResult::default()
            };
            out.diagnostics.push(Diagnostic {
                code: "CanceledSearchResultIgnored",
                message: format!(
                    "ignored late {event_kind} after owned {:?} cancellation; no result or work admission",
                    cancellation.reason
                ),
            });
            return out.into();
        }
        // Eviction only reduces the classification window. It never admits an
        // unknown/foreign ticket or removes a failure's original cause.
        stale(completion)
    }
    fn synchronize<P: PositionPort>(&mut self, session: &Session<P>) {
        self.ticket = session.active_ticket();
        if self.ticket.is_none() {
            self.close_current();
        }
    }
    fn matching_current(&mut self, ticket: &SearchTicket) -> Option<&mut CurrentSearch> {
        self.current
            .as_mut()
            .filter(|current| current.binding.ticket() == ticket)
    }
}

impl Drop for ContractSessionOwner {
    fn drop(&mut self) {
        // An owner dropped with an active Start is an abort, not a natural
        // synchronization close; preserve its optional Collection failure.
        if let Some(start) = self.current.as_ref().and_then(|current| current.original_start.as_ref()) {
            start.abort_collection();
        }
        self.close_current();
    }
}

fn acceptance(
    current: &CurrentSearch,
    scope: SessionScope,
    clock: InstantClock,
    now: Instant,
) -> Result<(), ContractError> {
    if !same_scope(current.scope, scope) {
        return Err(error(
            ErrorCode::Stale,
            "search game/root or registry scope no longer current",
        ));
    }
    if current.cancellation.is_canceled() {
        current.cancellation.cancel();
        return Err(error(
            ErrorCode::Canceled,
            "common logical search cancellation closed admission",
        ));
    }
    current
        .deadlines
        .hard
        .accepts(clock.domain(), clock.tick_at(now.max(Instant::now()))?)
}

fn mirror_cancel(current: &CurrentSearch) {
    if current.cancellation.is_canceled() {
        current.cancellation.cancel();
    }
}
fn same_scope(a: SessionScope, b: SessionScope) -> bool {
    a == b
}
fn error(code: ErrorCode, detail: &'static str) -> ContractError {
    ContractError::new(code, Stage::Admission, detail)
}
fn map_build_error(err: BuildSearchError) -> ContractError {
    match err {
        BuildSearchError::UnsupportedSearchGoal => error(
            ErrorCode::UnsupportedContract,
            "selected driver has no registered depth/mate capability",
        ),
        BuildSearchError::SimulationLimitExceeded { .. } => error(
            ErrorCode::ResourceExhausted,
            "go nodes exceeds configured search work bound",
        ),
        BuildSearchError::UntimedDeadlineOverflow => {
            error(ErrorCode::ResourceExhausted, "untimed deadline overflow")
        }
        BuildSearchError::TimeBudget(
            TimeBudgetError::DurationOverflow(detail) | TimeBudgetError::DeadlineOverflow(detail),
        ) => error(ErrorCode::ResourceExhausted, detail),
        BuildSearchError::TimeBudget(_) => error(
            ErrorCode::InvalidInput,
            "UCI time budget could not be constructed",
        ),
        _ => error(
            ErrorCode::InvalidInput,
            "UCI search settings or limits are invalid",
        ),
    }
}
fn attach_error<S>(out: &mut ContractSessionOutcome<S>, err: ContractError) {
    out.errors.push(err);
    out.session.diagnostics.push(Diagnostic {
        code: "ContractFailure",
        message: err.to_string(),
    });
    out.session.accepted = false;
}
fn rejected<S>(err: ContractError) -> ContractSessionOutcome<S> {
    let mut out = SessionResult::default().into();
    attach_error(&mut out, err);
    out
}
fn stale<S>(completion: Option<SearchCompletion>) -> ContractSessionOutcome<S> {
    let mut out = rejected(error(
        ErrorCode::Stale,
        "inactive or foreign common search ticket",
    ));
    if let Some(SearchCompletion::Failed { code, message }) = completion {
        out.session.diagnostics.push(Diagnostic {
            code: "SearchFailed",
            message: format!("{code}: {message}"),
        });
    }
    out
}
fn merge<S>(out: &mut SessionResult<S>, other: SessionResult<S>) {
    out.protocol.extend(other.protocol);
    out.effects.extend(other.effects);
    out.diagnostics.extend(other.diagnostics);
    out.accepted &= other.accepted;
}
fn reject_prepared<P: PositionPort>(
    session: &mut Session<P>,
    current: &mut CurrentSearch,
    prepared: SessionResult<P::Snapshot>,
    err: ContractError,
) -> ContractSessionOutcome<P::Snapshot> {
    let mut out = if err.code == ErrorCode::Expired {
        expire_current(session, current, err)
    } else {
        rejected(err)
    };
    out.session.diagnostics.extend(
        prepared
            .diagnostics
            .into_iter()
            .filter(|d| d.code != "AcceptanceClosed"),
    );
    out
}
fn expire_current<P: PositionPort>(
    session: &mut Session<P>,
    current: &mut CurrentSearch,
    err: ContractError,
) -> ContractSessionOutcome<P::Snapshot> {
    if let Some(start) = &current.original_start { start.abort_collection(); }
    current.cancellation.cancel();
    let mut out = if current.infinite && current.binding.budget_kind() == BudgetKind::ResourceWall {
        session.complete(
            current.binding.ticket(),
            SearchCompletion::Completed { bestmove: None },
        )
    } else {
        session.expire(current.binding.ticket())
    };
    current.work_completed = true;
    let code = if current.binding.budget_kind() == BudgetKind::ResourceWall {
        "ResourceWallLimit"
    } else {
        "SearchExpired"
    };
    out.diagnostics.push(Diagnostic {
        code,
        message: "common hard result boundary closed search work".into(),
    });
    let mut outcome = out.into();
    attach_error(&mut outcome, err);
    outcome
}
