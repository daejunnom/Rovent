//! B 담당의 임시 UCI ↔ 탐색 연결부. 공통 계약의 generation/clock schema가 아니다.
//!
//! `Effect::Start`의 ticket/limits와 Rules가 확인한 실제 차례를 연결한다. output과
//! worker의 소유권은 Session과 호출자에게 남는다. 이 모듈은 thread나 GPU 작업을
//! 시작하지 않으며 물리 작업의 유한 drain을 보장하는 runtime을 대체하지 않는다.
//! 호출자는 stop/root 교체의 Cancel effect를 처리할 때 해당 binding의 `cancel()`을
//! 먼저 호출한 뒤 SessionResult의 protocol을 출력해야 한다.
//!
//! 기존 생성자는 nodes를 완료된 simulation의 상한으로 해석한다. root prior 초기화는
//! simulation이 아니다. Rovent는 명시한 별도 의미에서 원래 nodes를 position visit
//! 예산으로 남기고 transport simulation 상한과 결합하지 않는다. 기존 의미에서는
//! `max_simulations`를 넘는 nodes 요청을 값을 조용히 줄이지 않고 거부한다.
//! 시간 없는 nodes/infinite 검색도 호출자가 명시한 유한 resource wall 한도를 둔다.
//! 이것은 UCI movetime/clock 의미가 아니며, 한도 도달은 진단되는 자원 실패이다.

use std::fmt;
use std::num::NonZeroU16;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rz_contracts::rovent::SearchGoal;
use rz_search::driver::{SearchControl, StopReason};
use rz_search::time::{TimeBudget, TimeBudgetConfig, TimeBudgetError, TimeControl};

use crate::{
    Diagnostic, GoLimits, PositionPort, SearchCompletion, SearchTicket, Session, SessionResult,
};

/// FEN 문자열 추측 대신 checked snapshot의 실제 차례를 전달한다.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SideToMove {
    White,
    Black,
}

/// Default가 없다. 호출자가 자원 상한과 시간 여유를 명시해 실행 설정을 잠근다.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuildSearchSettings {
    pub time_config: TimeBudgetConfig,
    pub max_simulations: u64,
    pub untimed_limit: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BudgetKind {
    UciTime,
    ResourceWall,
}

/// The selected engine, rather than the transport simulation loop, owns Rovent's
/// original position-visit budget. This choice does not issue search authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeBudgetSemantics {
    LegacySimulations,
    RoventPositionVisits,
}

/// Validated time and transport settings before `Session` issues a Start ticket.
///
/// Preparing does not mutate a session, grant result admission, or become the
/// cancellation owner of an existing search. `bind` consumes this preparation
/// without resampling time or renewing its original deadlines. The caller still
/// owns original Go limits and must bind only the corresponding Start ticket.
#[derive(Debug)]
pub struct PreparedSearchBinding {
    control: SearchControl,
    budget: TimeBudget,
    kind: BudgetKind,
    goal: SearchGoal,
    infinite: bool,
    node_budget_semantics: NodeBudgetSemantics,
}

#[derive(Debug)]
pub struct SearchBinding {
    ticket: SearchTicket,
    control: SearchControl,
    budget: TimeBudget,
    kind: BudgetKind,
    goal: SearchGoal,
    infinite: bool,
    node_budget_semantics: NodeBudgetSemantics,
    work_completed: AtomicBool,
}

impl SearchBinding {
    pub fn new(
        ticket: SearchTicket,
        limits: &GoLimits,
        side: SideToMove,
        start: Instant,
        settings: BuildSearchSettings,
    ) -> Result<Self, BuildSearchError> {
        Self::prepare(
            limits,
            side,
            start,
            settings,
            None,
            NodeBudgetSemantics::LegacySimulations,
        )
        .map(|prepared| prepared.bind(ticket))
    }

    /// Explicit admission for a driver that implements the registered goals.
    /// Legacy callers of `new` cannot silently ignore depth/mate objectives.
    pub fn new_with_goal_capability(
        ticket: SearchTicket,
        limits: &GoLimits,
        side: SideToMove,
        start: Instant,
        settings: BuildSearchSettings,
        registered_max_plies: NonZeroU16,
    ) -> Result<Self, BuildSearchError> {
        Self::prepare(
            limits,
            side,
            start,
            settings,
            Some(registered_max_plies),
            NodeBudgetSemantics::LegacySimulations,
        )
        .map(|prepared| prepared.bind(ticket))
    }

    /// Pure preparation for engines whose node units differ from simulations.
    pub fn prepare(
        limits: &GoLimits,
        side: SideToMove,
        start: Instant,
        settings: BuildSearchSettings,
        goal_capability: Option<NonZeroU16>,
        node_budget_semantics: NodeBudgetSemantics,
    ) -> Result<PreparedSearchBinding, BuildSearchError> {
        PreparedSearchBinding::prepare(
            limits,
            side,
            start,
            settings,
            goal_capability,
            node_budget_semantics,
        )
    }

    pub fn ticket(&self) -> &SearchTicket {
        &self.ticket
    }

    pub fn control(&self) -> &SearchControl {
        &self.control
    }

    pub fn budget(&self) -> &TimeBudget {
        &self.budget
    }

    pub fn budget_kind(&self) -> BudgetKind {
        self.kind
    }

    pub fn goal(&self) -> SearchGoal {
        self.goal
    }

    pub fn node_budget_semantics(&self) -> NodeBudgetSemantics {
        self.node_budget_semantics
    }

    /// Close tree/evaluator result admission before emitting stop/expiry output.
    /// Physical work and buffers still need the runtime's bounded drain.
    pub fn cancel(&self) {
        self.control.cancel();
    }

    /// `now` is the caller's monotonic dispatch tick. The actual current clock is
    /// also checked, so a delayed producer's old enqueue tick cannot admit output.
    /// Session independently checks ticket ownership and bounded root move text.
    pub fn progress<P: PositionPort>(
        &self,
        session: &mut Session<P>,
        bestmove: &str,
        now: Instant,
    ) -> SessionResult<P::Snapshot> {
        self.progress_with_clock(session, bestmove, now, Instant::now)
    }

    fn progress_with_clock<P: PositionPort, C: FnMut() -> Instant>(
        &self,
        session: &mut Session<P>,
        bestmove: &str,
        now: Instant,
        mut read_now: C,
    ) -> SessionResult<P::Snapshot> {
        if !self.is_active(session) {
            return session.progress(&self.ticket, bestmove);
        }
        if self.work_completed.load(Ordering::Acquire) {
            return session.progress(&self.ticket, bestmove);
        }
        let now = now.max(read_now());
        match self.control.stop_reason(now) {
            Some(StopReason::Deadline) => self.expired(session, None),
            Some(StopReason::Canceled) => {
                rejected("CanceledSearch", "ignored canceled search progress")
            }
            _ => {
                let mut final_reason = None;
                let out = session.progress_with_guard(&self.ticket, bestmove, || {
                    // Session prepares/validates the candidate before invoking this
                    // guard. The fresh check is the logical acceptance point.
                    final_reason = self.control.stop_reason(now.max(read_now()));
                    final_reason.is_none()
                });
                self.after_guard(session, out, final_reason, "search progress")
            }
        }
    }

    pub fn complete<P: PositionPort>(
        &self,
        session: &mut Session<P>,
        completion: SearchCompletion,
        now: Instant,
    ) -> SessionResult<P::Snapshot> {
        self.complete_with_clock(session, completion, now, Instant::now)
    }

    fn complete_with_clock<P: PositionPort, C: FnMut() -> Instant>(
        &self,
        session: &mut Session<P>,
        completion: SearchCompletion,
        now: Instant,
        mut read_now: C,
    ) -> SessionResult<P::Snapshot> {
        if !self.is_active(session) || self.work_completed.load(Ordering::Acquire) {
            // Keep the stale worker's source error without changing unrelated output
            // ownership. A stale failure is not the active search's failure.
            let mut out = rejected(
                "StaleSearch",
                "ignored inactive, duplicated, or replaced search result",
            );
            preserve_failure(&mut out, Some(completion));
            return out;
        }
        let now = now.max(read_now());
        match self.control.stop_reason(now) {
            Some(StopReason::Deadline) => self.expired(session, Some(completion)),
            Some(StopReason::Canceled) => {
                let mut out = rejected("CanceledSearch", "ignored canceled search completion");
                preserve_failure(&mut out, Some(completion));
                out
            }
            // Completion does not cancel the control before normal result acceptance.
            // Session owns infinite-output waiting and duplicated completion rejection.
            _ => {
                let mut final_reason = None;
                let mut guard_called = false;
                let out = session.complete_with_guard(&self.ticket, completion, || {
                    guard_called = true;
                    final_reason = self.control.stop_reason(now.max(read_now()));
                    final_reason.is_none()
                });
                if guard_called && final_reason.is_none() {
                    // Failed/illegal completion can be consumed with its diagnostics.
                    // A stale or guard-rejected completion must not close live work.
                    self.work_completed.store(true, Ordering::Release);
                }
                self.after_guard(session, out, final_reason, "search completion")
            }
        }
    }

    /// A timer must use the same hard tick as callbacks. Early expiry is rejected.
    /// Worker cancellation blocks results but cannot suppress this owner's hard
    /// output deadline. A replaced binding still cannot finish another search.
    pub fn expire<P: PositionPort>(
        &self,
        session: &mut Session<P>,
        now: Instant,
    ) -> SessionResult<P::Snapshot> {
        if !self.is_active(session) {
            return session.expire(&self.ticket);
        }
        let now = now.max(Instant::now());
        if now < self.control.deadline {
            return rejected(
                "EarlyDeadline",
                "deadline event arrived before the hard boundary",
            );
        }
        if self.work_completed.load(Ordering::Acquire) {
            // A completed infinite worker has no running resource work. Closing the
            // gate at its old timer does not turn passive output waiting into failure.
            self.cancel();
            return SessionResult::default();
        }
        self.expired(session, None)
    }

    fn is_active<P: PositionPort>(&self, session: &Session<P>) -> bool {
        session.active_ticket().as_ref() == Some(&self.ticket)
    }

    fn after_guard<P: PositionPort>(
        &self,
        session: &mut Session<P>,
        prepared: SessionResult<P::Snapshot>,
        final_reason: Option<StopReason>,
        operation: &str,
    ) -> SessionResult<P::Snapshot> {
        let mut out = match final_reason {
            Some(StopReason::Deadline) => self.expired(session, None),
            Some(StopReason::Canceled) => rejected(
                "CanceledSearch",
                &format!("ignored canceled {operation} at final acceptance"),
            ),
            _ => return prepared,
        };
        // A false Session guard returned no mutation/output/effects, but prepared
        // source errors and illegal-candidate diagnostics still belong to this event.
        out.diagnostics.extend(
            prepared
                .diagnostics
                .into_iter()
                .filter(|diagnostic| diagnostic.code != "AcceptanceClosed"),
        );
        out
    }

    fn expired<P: PositionPort>(
        &self,
        session: &mut Session<P>,
        completion: Option<SearchCompletion>,
    ) -> SessionResult<P::Snapshot> {
        // The control clone held by Tree/worker shares this atomic gate.
        self.cancel();
        // UCI infinite retains its output owner until explicit stop even when the
        // worker's resource wall closes. It holds only the checked candidate; no
        // computation restarts and no late callback can update that candidate.
        let mut out = if self.infinite && self.kind == BudgetKind::ResourceWall {
            session.complete(&self.ticket, SearchCompletion::Completed { bestmove: None })
        } else {
            session.expire(&self.ticket)
        };
        self.work_completed.store(true, Ordering::Release);
        let (code, message) = match self.kind {
            BudgetKind::UciTime => ("SearchExpired", "search result acceptance deadline reached"),
            BudgetKind::ResourceWall => (
                "ResourceWallLimit",
                "explicit untimed search wall resource limit reached",
            ),
        };
        out.diagnostics.push(Diagnostic {
            code,
            message: message.into(),
        });
        out.accepted = false;
        // Session retains the previous checked move/fallback. A late success
        // cannot replace it, and a simultaneous backend failure must not disappear.
        preserve_failure(&mut out, completion);
        out
    }
}

impl PreparedSearchBinding {
    pub fn prepare(
        limits: &GoLimits,
        side: SideToMove,
        start: Instant,
        settings: BuildSearchSettings,
        goal_capability: Option<NonZeroU16>,
        node_budget_semantics: NodeBudgetSemantics,
    ) -> Result<Self, BuildSearchError> {
        validate_limits(limits, node_budget_semantics)?;
        let goal = SearchGoal {
            depth_plies: limits.depth_plies,
            mate_moves: limits.mate_moves,
        };
        if goal.has_goal() {
            let max_plies = goal_capability.ok_or(BuildSearchError::UnsupportedSearchGoal)?;
            goal.validate(max_plies)
                .map_err(|error| BuildSearchError::InvalidGo(error.detail))?;
        }
        if settings.max_simulations == 0 {
            return Err(BuildSearchError::ZeroSimulationLimit);
        }
        if settings.untimed_limit.is_zero() {
            return Err(BuildSearchError::ZeroUntimedLimit);
        }
        // Validate even when unused by the selected mode: settings must be a finite,
        // reusable configuration rather than a latent failure for the next request.
        start
            .checked_add(settings.untimed_limit)
            .ok_or(BuildSearchError::UntimedDeadlineOverflow)?;
        let max_simulations = match node_budget_semantics {
            NodeBudgetSemantics::LegacySimulations => {
                let requested = limits.nodes.unwrap_or(settings.max_simulations);
                if requested > settings.max_simulations {
                    return Err(BuildSearchError::SimulationLimitExceeded {
                        requested,
                        configured: settings.max_simulations,
                    });
                }
                requested
            }
            NodeBudgetSemantics::RoventPositionVisits => settings.max_simulations,
        };

        let (time_control, kind) = if let Some(movetime) = limits.movetime_ms {
            (
                TimeControl::MoveTime(Duration::from_millis(movetime)),
                BudgetKind::UciTime,
            )
        } else if let (Some(white), Some(black)) = (limits.white_time_ms, limits.black_time_ms) {
            // validate_limits established that both side clocks exist.
            let (remaining, increment) = match side {
                SideToMove::White => (white, limits.white_increment_ms.unwrap_or(0)),
                SideToMove::Black => (black, limits.black_increment_ms.unwrap_or(0)),
            };
            (
                TimeControl::Clock {
                    remaining: Duration::from_millis(remaining),
                    increment: Duration::from_millis(increment),
                    moves_to_go: limits.moves_to_go,
                },
                BudgetKind::UciTime,
            )
        } else {
            (
                TimeControl::MoveTime(settings.untimed_limit),
                BudgetKind::ResourceWall,
            )
        };
        let budget = TimeBudget::new(start, time_control, settings.time_config)
            .map_err(BuildSearchError::TimeBudget)?;
        // A Rovent process registers the complete per-go wall allocation,
        // including output reserve. Legacy simulation bindings keep their
        // existing clock policy; only the explicit node-domain opts in.
        if node_budget_semantics == NodeBudgetSemantics::RoventPositionVisits
            && budget.allocation > settings.untimed_limit
        {
            return Err(BuildSearchError::InvalidGo(
                "actual go allocation exceeds the registered Rovent wall ceiling",
            ));
        }
        let control = SearchControl::from_budget(&budget, max_simulations);
        Ok(Self {
            control,
            budget,
            kind,
            goal,
            infinite: limits.infinite,
            node_budget_semantics,
        })
    }

    /// Attach only the Session-issued output ticket; no new clock is sampled.
    pub fn bind(self, ticket: SearchTicket) -> SearchBinding {
        SearchBinding {
            ticket,
            control: self.control,
            budget: self.budget,
            kind: self.kind,
            goal: self.goal,
            infinite: self.infinite,
            node_budget_semantics: self.node_budget_semantics,
            work_completed: AtomicBool::new(false),
        }
    }

    pub fn control(&self) -> &SearchControl {
        &self.control
    }

    pub fn budget(&self) -> &TimeBudget {
        &self.budget
    }

    pub fn budget_kind(&self) -> BudgetKind {
        self.kind
    }

    pub fn goal(&self) -> SearchGoal {
        self.goal
    }

    pub fn node_budget_semantics(&self) -> NodeBudgetSemantics {
        self.node_budget_semantics
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BuildSearchError {
    InvalidGo(&'static str),
    UnsupportedSearchGoal,
    ZeroSimulationLimit,
    ZeroUntimedLimit,
    UntimedDeadlineOverflow,
    SimulationLimitExceeded { requested: u64, configured: u64 },
    TimeBudget(TimeBudgetError),
}

impl fmt::Display for BuildSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGo(reason) => write!(f, "invalid go limits: {reason}"),
            Self::UnsupportedSearchGoal => {
                f.write_str("selected search binding has no registered depth/mate capability")
            }
            Self::ZeroSimulationLimit => {
                f.write_str("configured simulation limit must be positive")
            }
            Self::ZeroUntimedLimit => {
                f.write_str("untimed search requires a positive wall resource limit")
            }
            Self::UntimedDeadlineOverflow => f.write_str("untimed search wall deadline overflow"),
            Self::SimulationLimitExceeded {
                requested,
                configured,
            } => write!(
                f,
                "requested {requested} simulations exceeds configured limit {configured}"
            ),
            Self::TimeBudget(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl std::error::Error for BuildSearchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TimeBudget(error) => Some(error),
            _ => None,
        }
    }
}

fn validate_limits(
    limits: &GoLimits,
    node_budget_semantics: NodeBudgetSemantics,
) -> Result<(), BuildSearchError> {
    if limits.searchmoves.as_ref().is_some_and(|moves| {
        moves.is_empty()
            || moves
                .iter()
                .any(|movement| !crate::parser::is_move_text(movement))
    }) {
        return Err(BuildSearchError::InvalidGo(
            "searchmoves requires a nonempty list of move text",
        ));
    }
    if node_budget_semantics == NodeBudgetSemantics::LegacySimulations && limits.nodes == Some(0) {
        return Err(BuildSearchError::InvalidGo("nodes must be positive"));
    }
    if limits.moves_to_go == Some(0) {
        return Err(BuildSearchError::InvalidGo("movestogo must be positive"));
    }
    let has_clock = limits.white_time_ms.is_some() || limits.black_time_ms.is_some();
    if limits.white_time_ms.is_some() != limits.black_time_ms.is_some() {
        return Err(BuildSearchError::InvalidGo(
            "clock mode requires both side clocks",
        ));
    }
    if !has_clock
        && (limits.white_increment_ms.is_some()
            || limits.black_increment_ms.is_some()
            || limits.moves_to_go.is_some())
    {
        return Err(BuildSearchError::InvalidGo(
            "increments/movestogo require clocks",
        ));
    }
    if limits.movetime_ms.is_some() && has_clock {
        return Err(BuildSearchError::InvalidGo(
            "movetime and clocks are distinct modes",
        ));
    }
    if limits.infinite && (limits.ponder || limits.movetime_ms.is_some() || has_clock) {
        return Err(BuildSearchError::InvalidGo(
            "infinite cannot combine with ponder or time limits",
        ));
    }
    Ok(())
}

fn rejected<S>(code: &'static str, message: &str) -> SessionResult<S> {
    SessionResult {
        diagnostics: vec![Diagnostic {
            code,
            message: message.into(),
        }],
        accepted: false,
        ..SessionResult::default()
    }
}

fn preserve_failure<S>(out: &mut SessionResult<S>, completion: Option<SearchCompletion>) {
    if let Some(SearchCompletion::Failed { code, message }) = completion {
        out.diagnostics.push(Diagnostic {
            code: "SearchFailed",
            message: format!("{code}: {message}"),
        });
        out.accepted = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Effect, EngineIdentity, ParserLimits, PositionSpec, PreparedPosition};

    struct FixtureRules;
    impl PositionPort for FixtureRules {
        type Snapshot = ();
        type Error = &'static str;
        fn prepare(&self, _: &PositionSpec) -> Result<PreparedPosition<()>, Self::Error> {
            Ok(PreparedPosition {
                snapshot: (),
                legal_moves: vec!["a2a3".into(), "b2b3".into()],
                exact_terminal: false,
            })
        }
    }

    fn session() -> Session<FixtureRules> {
        Session::new(
            FixtureRules,
            EngineIdentity {
                name: "bridge-fixture".into(),
                author: "fixture".into(),
            },
            vec![],
            ParserLimits::default(),
        )
        .unwrap()
    }

    fn start(session: &mut Session<FixtureRules>, line: &str) -> (SearchTicket, GoLimits) {
        let out = session.handle_line(line);
        assert!(out.accepted, "{:?}", out.diagnostics);
        match out
            .effects
            .into_iter()
            .find(|effect| matches!(effect, Effect::Start { .. }))
        {
            Some(Effect::Start { ticket, limits, .. }) => (ticket, limits),
            other => panic!("expected Start effect, got {other:?}"),
        }
    }

    fn settings() -> BuildSearchSettings {
        BuildSearchSettings {
            time_config: TimeBudgetConfig::default(),
            max_simulations: 100,
            untimed_limit: Duration::from_secs(60),
        }
    }

    fn binding(session: &mut Session<FixtureRules>, line: &str, now: Instant) -> SearchBinding {
        let (ticket, limits) = start(session, line);
        SearchBinding::new(ticket, &limits, SideToMove::White, now, settings()).unwrap()
    }

    #[test]
    fn own_side_clock_and_increment_are_selected_from_checked_side() {
        let now = Instant::now();
        let (ticket, limits) = start(
            &mut session(),
            "go wtime 30000 btime 60000 winc 1000 binc 2000",
        );
        let white = SearchBinding::new(ticket.clone(), &limits, SideToMove::White, now, settings())
            .unwrap();
        let black =
            SearchBinding::new(ticket, &limits, SideToMove::Black, now, settings()).unwrap();
        // White: 30000/30 + 800 = 1800; black: 60000/30 + 1600 = 3600.
        assert_eq!(white.budget().allocation, Duration::from_millis(5400));
        assert_eq!(black.budget().allocation, Duration::from_millis(10800));
        assert_eq!(white.control().deadline, white.budget().hard_deadline);
        assert_eq!(black.control().soft_deadline, black.budget().soft_deadline);
        assert_eq!(
            white.control().admission_deadline,
            white.budget().admission_deadline
        );
        assert_eq!(white.budget_kind(), BudgetKind::UciTime);
    }

    #[test]
    fn movetime_is_fixed_and_requested_simulation_limit_is_exact() {
        let now = Instant::now();
        let binding = binding(&mut session(), "go movetime 1000 nodes 7", now);
        assert_eq!(binding.budget().allocation, Duration::from_secs(1));
        assert_eq!(binding.control().max_simulations, 7);
        assert_eq!(binding.control().deadline, now + Duration::from_millis(990));
    }

    #[test]
    fn preparation_neither_issues_nor_replaces_a_session_ticket() {
        let now = Instant::now();
        let mut session = session();
        let limits = GoLimits {
            nodes: Some(1),
            ..GoLimits::default()
        };
        let prepared = PreparedSearchBinding::prepare(
            &limits,
            SideToMove::White,
            now,
            settings(),
            None,
            NodeBudgetSemantics::RoventPositionVisits,
        )
        .unwrap();
        assert!(session.active_ticket().is_none());
        let (ticket, _) = start(&mut session, "go nodes 1");
        let active = session.active_ticket();
        let invalid = PreparedSearchBinding::prepare(
            &GoLimits::default(),
            SideToMove::White,
            now,
            settings(),
            None,
            NodeBudgetSemantics::RoventPositionVisits,
        );
        assert!(matches!(invalid, Err(BuildSearchError::InvalidGo(_))));
        assert_eq!(session.active_ticket(), active);
        let binding = prepared.bind(ticket.clone());
        assert_eq!(binding.ticket(), &ticket);
        assert_eq!(session.active_ticket().as_ref(), Some(&ticket));
        assert!(!binding.control().cancellation.load(Ordering::Acquire));
    }

    #[test]
    fn binding_prepared_settings_preserves_original_deadlines_and_control_gate() {
        let now = Instant::now();
        let limits = GoLimits {
            movetime_ms: Some(1000),
            nodes: Some(7),
            ..GoLimits::default()
        };
        let prepared = SearchBinding::prepare(
            &limits,
            SideToMove::White,
            now,
            settings(),
            None,
            NodeBudgetSemantics::LegacySimulations,
        )
        .unwrap();
        let original_control = prepared.control().clone();
        let original_allocation = prepared.budget().allocation;
        let (ticket, _) = start(&mut session(), "go movetime 1000 nodes 7");
        let binding = prepared.bind(ticket);
        assert_eq!(binding.control().deadline, original_control.deadline);
        assert_eq!(
            binding.control().soft_deadline,
            original_control.soft_deadline
        );
        assert_eq!(
            binding.control().admission_deadline,
            original_control.admission_deadline
        );
        assert_eq!(binding.budget().allocation, original_allocation);
        assert_eq!(binding.control().max_simulations, 7);
        assert_eq!(
            binding.node_budget_semantics(),
            NodeBudgetSemantics::LegacySimulations
        );
        assert!(std::sync::Arc::ptr_eq(
            &binding.control().cancellation,
            &original_control.cancellation
        ));
        binding.cancel();
        assert!(original_control.cancellation.load(Ordering::Acquire));
    }

    #[test]
    fn rovent_position_visits_never_become_transport_simulations_or_mutate_go() {
        let now = Instant::now();
        for nodes in [Some(0), Some(1), Some(101), Some(u64::MAX), None] {
            let limits = GoLimits {
                nodes,
                movetime_ms: Some(1000),
                ..GoLimits::default()
            };
            let prepared = PreparedSearchBinding::prepare(
                &limits,
                SideToMove::White,
                now,
                settings(),
                None,
                NodeBudgetSemantics::RoventPositionVisits,
            )
            .unwrap();
            assert_eq!(limits.nodes, nodes);
            assert_eq!(prepared.control().max_simulations, 100);
            assert_eq!(prepared.budget_kind(), BudgetKind::UciTime);
            assert_eq!(
                prepared.control().deadline,
                now + Duration::from_millis(990)
            );
            assert_eq!(
                prepared.node_budget_semantics(),
                NodeBudgetSemantics::RoventPositionVisits
            );
        }
        let prepared = PreparedSearchBinding::prepare(
            &GoLimits {
                nodes: Some(0),
                ..GoLimits::default()
            },
            SideToMove::White,
            now,
            settings(),
            None,
            NodeBudgetSemantics::RoventPositionVisits,
        )
        .unwrap();
        assert_eq!(prepared.budget_kind(), BudgetKind::ResourceWall);
        assert_eq!(prepared.budget().allocation, settings().untimed_limit);
    }

    #[test]
    fn prepared_legacy_nodes_retain_zero_rejection_and_exact_simulation_guard() {
        let now = Instant::now();
        for nodes in [0, 1, 100, 101, u64::MAX] {
            let result = PreparedSearchBinding::prepare(
                &GoLimits {
                    nodes: Some(nodes),
                    ..GoLimits::default()
                },
                SideToMove::White,
                now,
                settings(),
                None,
                NodeBudgetSemantics::LegacySimulations,
            );
            match nodes {
                0 => assert!(matches!(result, Err(BuildSearchError::InvalidGo(_)))),
                1 | 100 => assert_eq!(result.unwrap().control().max_simulations, nodes),
                _ => assert!(matches!(
                    result,
                    Err(BuildSearchError::SimulationLimitExceeded {
                        requested,
                        configured: 100
                    }) if requested == nodes
                )),
            }
        }
    }

    #[test]
    fn rovent_preparation_still_requires_registered_goals_and_finite_settings() {
        let now = Instant::now();
        let limits = GoLimits {
            nodes: Some(0),
            depth_plies: NonZeroU16::new(6),
            mate_moves: NonZeroU16::new(3),
            ..GoLimits::default()
        };
        assert!(matches!(
            PreparedSearchBinding::prepare(
                &limits,
                SideToMove::White,
                now,
                settings(),
                None,
                NodeBudgetSemantics::RoventPositionVisits,
            ),
            Err(BuildSearchError::UnsupportedSearchGoal)
        ));
        let prepared = PreparedSearchBinding::prepare(
            &limits,
            SideToMove::White,
            now,
            settings(),
            NonZeroU16::new(8),
            NodeBudgetSemantics::RoventPositionVisits,
        )
        .unwrap();
        assert_eq!(prepared.goal().depth_plies, limits.depth_plies);
        assert_eq!(prepared.goal().mate_moves, limits.mate_moves);
        assert!(matches!(
            PreparedSearchBinding::prepare(
                &limits,
                SideToMove::White,
                now,
                settings(),
                NonZeroU16::new(4),
                NodeBudgetSemantics::RoventPositionVisits,
            ),
            Err(BuildSearchError::InvalidGo(_))
        ));
        for (configured, expected) in [
            (
                BuildSearchSettings {
                    max_simulations: 0,
                    ..settings()
                },
                BuildSearchError::ZeroSimulationLimit,
            ),
            (
                BuildSearchSettings {
                    untimed_limit: Duration::ZERO,
                    ..settings()
                },
                BuildSearchError::ZeroUntimedLimit,
            ),
            (
                BuildSearchSettings {
                    untimed_limit: Duration::MAX,
                    ..settings()
                },
                BuildSearchError::UntimedDeadlineOverflow,
            ),
        ] {
            assert_eq!(
                PreparedSearchBinding::prepare(
                    &limits,
                    SideToMove::White,
                    now,
                    configured,
                    NonZeroU16::new(8),
                    NodeBudgetSemantics::RoventPositionVisits,
                )
                .unwrap_err(),
                expected
            );
        }
    }

    #[test]
    fn legacy_binding_rejects_goals_and_registered_binding_preserves_their_units() {
        let now = Instant::now();
        let mut session = session();
        session
            .set_search_goal_capability(NonZeroU16::new(8))
            .unwrap();
        let (ticket, limits) = start(&mut session, "go depth 6 mate 3 nodes 7 searchmoves b2b3");
        assert!(matches!(
            SearchBinding::new(ticket.clone(), &limits, SideToMove::White, now, settings()),
            Err(BuildSearchError::UnsupportedSearchGoal)
        ));
        let binding = SearchBinding::new_with_goal_capability(
            ticket,
            &limits,
            SideToMove::White,
            now,
            settings(),
            NonZeroU16::new(8).unwrap(),
        )
        .unwrap();
        assert_eq!(
            binding.goal(),
            SearchGoal {
                depth_plies: NonZeroU16::new(6),
                mate_moves: NonZeroU16::new(3),
            }
        );
        assert_eq!(binding.control().max_simulations, 7);
        assert_eq!(binding.budget_kind(), BudgetKind::ResourceWall);
        let outside = binding.progress(&mut session, "a2a3", now);
        assert!(!outside.accepted);
        assert!(
            outside
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "IllegalBestMove")
        );
        assert_eq!(session.handle_line("stop").protocol, ["bestmove b2b3"]);
    }

    #[test]
    fn registered_depth_and_mate_only_goals_use_a_resource_wall_without_becoming_nodes() {
        let now = Instant::now();
        for line in [
            "go depth 4",
            "go mate 2",
            "go infinite depth 4 mate 2 nodes 3",
        ] {
            let mut session = session();
            session
                .set_search_goal_capability(NonZeroU16::new(8))
                .unwrap();
            let (ticket, limits) = start(&mut session, line);
            let binding = SearchBinding::new_with_goal_capability(
                ticket,
                &limits,
                SideToMove::White,
                now,
                settings(),
                NonZeroU16::new(8).unwrap(),
            )
            .unwrap();
            assert_eq!(binding.goal().depth_plies, limits.depth_plies);
            assert_eq!(binding.goal().mate_moves, limits.mate_moves);
            assert_eq!(binding.budget_kind(), BudgetKind::ResourceWall);
            assert_eq!(
                binding.control().max_simulations,
                limits.nodes.unwrap_or(100)
            );
        }
    }

    #[test]
    fn direct_goals_above_the_registered_horizon_are_not_clamped() {
        let now = Instant::now();
        let (ticket, _) = start(&mut session(), "go nodes 1");
        for (limits, reason) in [
            (
                GoLimits {
                    depth_plies: NonZeroU16::new(9),
                    ..GoLimits::default()
                },
                "depth exceeds the registered chess-ply capability",
            ),
            (
                GoLimits {
                    mate_moves: NonZeroU16::new(5),
                    ..GoLimits::default()
                },
                "mate horizon exceeds the registered chess-ply capability",
            ),
            (
                GoLimits {
                    mate_moves: NonZeroU16::new(u16::MAX),
                    ..GoLimits::default()
                },
                "mate move-to-ply conversion overflow",
            ),
        ] {
            assert!(matches!(
                SearchBinding::new_with_goal_capability(
                    ticket.clone(), &limits, SideToMove::White, now, settings(),
                    NonZeroU16::new(8).unwrap(),
                ),
                Err(BuildSearchError::InvalidGo(actual)) if actual == reason
            ));
        }
    }

    #[test]
    fn untimed_searches_use_explicit_resource_wall_and_infinite_waits_for_stop() {
        let now = Instant::now();
        let mut session = session();
        let binding = binding(&mut session, "go infinite", now);
        assert_eq!(binding.budget_kind(), BudgetKind::ResourceWall);
        assert_eq!(binding.budget().allocation, settings().untimed_limit);
        let out = binding.complete(
            &mut session,
            SearchCompletion::Completed {
                bestmove: Some("b2b3".into()),
            },
            now,
        );
        assert!(out.protocol.is_empty());
        assert_eq!(binding.control().stop_reason(now), None);
        binding.cancel();
        let out = session.handle_line("stop");
        assert_eq!(out.protocol, ["bestmove b2b3"]);
    }

    #[test]
    fn default_go_and_restricted_default_go_use_registered_resource_wall() {
        let now = Instant::now();
        for command in ["go", "go searchmoves b2b3"] {
            let mut session = session();
            let binding = binding(&mut session, command, now);
            assert_eq!(binding.budget_kind(), BudgetKind::ResourceWall);
            assert_eq!(binding.budget().allocation, settings().untimed_limit);
            assert_eq!(
                binding.control().max_simulations,
                settings().max_simulations
            );
            assert!(binding.progress(&mut session, "b2b3", now).accepted);
            let stopped = session.handle_line("stop");
            assert_eq!(stopped.protocol, ["bestmove b2b3"]);
            assert!(
                binding
                    .complete(
                        &mut session,
                        SearchCompletion::Completed {
                            bestmove: Some("b2b3".into()),
                        },
                        now,
                    )
                    .protocol
                    .is_empty()
            );
        }
    }

    #[test]
    fn strict_expiry_closes_control_before_emitting_last_valid_move() {
        let now = Instant::now();
        let mut session = session();
        let binding = binding(&mut session, "go movetime 1000", now);
        assert!(
            binding
                .progress(
                    &mut session,
                    "b2b3",
                    binding.control().deadline - Duration::from_nanos(1)
                )
                .accepted
        );
        let deadline = binding.control().deadline;
        let out = binding.complete(
            &mut session,
            SearchCompletion::Completed {
                bestmove: Some("a2a3".into()),
            },
            deadline,
        );
        assert!(!out.accepted);
        assert_eq!(out.protocol, ["bestmove b2b3"]);
        assert_eq!(
            binding.control().stop_reason(now),
            Some(StopReason::Canceled)
        );
        assert!(out.diagnostics.iter().any(|d| d.code == "SearchExpired"));
        assert!(binding.expire(&mut session, deadline).protocol.is_empty());
    }

    #[test]
    fn expiry_preserves_failed_source_and_checked_fallback() {
        let now = Instant::now();
        let mut session = session();
        let binding = binding(&mut session, "go nodes 3", now);
        let out = binding.complete(
            &mut session,
            SearchCompletion::Failed {
                code: "GpuFail".into(),
                message: "original provider detail".into(),
            },
            binding.control().deadline,
        );
        assert_eq!(out.protocol, ["bestmove a2a3"]);
        assert!(
            out.diagnostics
                .iter()
                .any(|d| d.code == "ResourceWallLimit")
        );
        assert!(
            out.diagnostics
                .iter()
                .any(|d| d.code == "SearchFailed"
                    && d.message == "GpuFail: original provider detail")
        );
        assert!(out.diagnostics.iter().any(|d| d.code == "LegalFallback"));
    }

    #[test]
    fn canceled_callbacks_cannot_output_but_owned_timer_finishes_and_old_ticket_stays_stale() {
        let now = Instant::now();
        let mut session = session();
        let old = binding(&mut session, "go movetime 1000", now);
        old.cancel();
        assert!(!old.progress(&mut session, "b2b3", now).accepted);
        assert!(
            old.complete(
                &mut session,
                SearchCompletion::Completed {
                    bestmove: Some("b2b3".into())
                },
                now
            )
            .protocol
            .is_empty()
        );
        let canceled_timer = old.expire(&mut session, old.control().deadline);
        assert_eq!(canceled_timer.protocol, ["bestmove a2a3"]);
        assert!(
            canceled_timer
                .diagnostics
                .iter()
                .any(|d| d.code == "SearchExpired")
        );
        let current = binding(&mut session, "go movetime 1000", now);
        let out = old.expire(&mut session, old.control().deadline);
        assert!(out.protocol.is_empty());
        assert_eq!(session.active_ticket().as_ref(), Some(current.ticket()));
        assert_eq!(current.control().stop_reason(now), None);
    }

    #[test]
    fn canceled_finite_worker_hard_timer_emits_last_valid_candidate_once() {
        let now = Instant::now();
        let mut session = session();
        let binding = binding(&mut session, "go movetime 1000", now);
        assert!(binding.progress(&mut session, "b2b3", now).accepted);
        binding.cancel();
        let rejected = binding.complete(
            &mut session,
            SearchCompletion::Completed {
                bestmove: Some("a2a3".into()),
            },
            now,
        );
        assert!(!rejected.accepted);
        assert!(rejected.protocol.is_empty());
        let expired = binding.expire(&mut session, binding.control().deadline);
        assert_eq!(expired.protocol, ["bestmove b2b3"]);
        assert!(session.active_ticket().is_none());
        assert!(
            binding
                .expire(&mut session, binding.control().deadline)
                .protocol
                .is_empty()
        );
    }

    #[test]
    fn early_timer_cannot_cancel_search_and_zero_movetime_uses_fallback() {
        let now = Instant::now();
        let mut session = session();
        let timed = binding(&mut session, "go movetime 1000", now);
        assert!(!timed.expire(&mut session, now).accepted);
        assert_eq!(timed.control().stop_reason(now), None);
        let tiny = binding(&mut session, "go movetime 0", now);
        let out = tiny.complete(
            &mut session,
            SearchCompletion::Completed {
                bestmove: Some("b2b3".into()),
            },
            now,
        );
        assert_eq!(out.protocol, ["bestmove a2a3"]);
    }

    #[test]
    fn direct_limits_are_revalidated_and_oversized_nodes_are_not_clamped() {
        let now = Instant::now();
        let (ticket, _) = start(&mut session(), "go nodes 1");
        for limits in [
            GoLimits::default(),
            GoLimits {
                nodes: Some(0),
                ..GoLimits::default()
            },
            GoLimits {
                white_time_ms: Some(1000),
                ..GoLimits::default()
            },
            GoLimits {
                nodes: Some(1),
                white_increment_ms: Some(1),
                ..GoLimits::default()
            },
            GoLimits {
                movetime_ms: Some(10),
                infinite: true,
                ..GoLimits::default()
            },
            GoLimits {
                movetime_ms: Some(10),
                white_time_ms: Some(20),
                black_time_ms: Some(20),
                ..GoLimits::default()
            },
            GoLimits {
                white_time_ms: Some(20),
                black_time_ms: Some(20),
                moves_to_go: Some(0),
                ..GoLimits::default()
            },
        ] {
            assert!(matches!(
                SearchBinding::new(ticket.clone(), &limits, SideToMove::White, now, settings()),
                Err(BuildSearchError::InvalidGo(_))
            ));
        }
        assert!(matches!(
            SearchBinding::new(
                ticket,
                &GoLimits {
                    nodes: Some(101),
                    ..GoLimits::default()
                },
                SideToMove::White,
                now,
                settings()
            ),
            Err(BuildSearchError::SimulationLimitExceeded {
                requested: 101,
                configured: 100
            })
        ));
    }

    #[test]
    fn rovent_wall_ceiling_covers_actual_allocation_without_changing_legacy_clocks() {
        let start = Instant::now();
        let configured = BuildSearchSettings {
            untimed_limit: Duration::from_secs(1),
            ..settings()
        };
        let limits = GoLimits {
            movetime_ms: Some(1001),
            ..GoLimits::default()
        };
        assert!(matches!(
            PreparedSearchBinding::prepare(
                &limits,
                SideToMove::White,
                start,
                configured,
                None,
                NodeBudgetSemantics::RoventPositionVisits
            ),
            Err(BuildSearchError::InvalidGo(
                "actual go allocation exceeds the registered Rovent wall ceiling"
            ))
        ));
        let legacy = PreparedSearchBinding::prepare(
            &limits,
            SideToMove::White,
            start,
            configured,
            None,
            NodeBudgetSemantics::LegacySimulations,
        )
        .unwrap();
        assert_eq!(legacy.budget().allocation, Duration::from_millis(1001));
        let exact = PreparedSearchBinding::prepare(
            &GoLimits {
                movetime_ms: Some(1000),
                ..limits
            },
            SideToMove::White,
            start,
            configured,
            None,
            NodeBudgetSemantics::RoventPositionVisits,
        )
        .unwrap();
        assert_eq!(exact.budget().allocation, configured.untimed_limit);
    }

    #[test]
    fn invalid_finite_settings_fail_before_binding_creation() {
        let now = Instant::now();
        let (ticket, limits) = start(&mut session(), "go nodes 1");
        for (settings, expected) in [
            (
                BuildSearchSettings {
                    max_simulations: 0,
                    ..settings()
                },
                BuildSearchError::ZeroSimulationLimit,
            ),
            (
                BuildSearchSettings {
                    untimed_limit: Duration::ZERO,
                    ..settings()
                },
                BuildSearchError::ZeroUntimedLimit,
            ),
            (
                BuildSearchSettings {
                    untimed_limit: Duration::MAX,
                    ..settings()
                },
                BuildSearchError::UntimedDeadlineOverflow,
            ),
        ] {
            assert_eq!(
                SearchBinding::new(ticket.clone(), &limits, SideToMove::White, now, settings)
                    .unwrap_err(),
                expected
            );
        }
    }

    #[test]
    fn infinite_resource_expiry_stops_work_and_keeps_output_until_stop() {
        let now = Instant::now();
        let mut session = session();
        let binding = binding(&mut session, "go infinite", now);
        assert!(binding.progress(&mut session, "b2b3", now).accepted);
        let out = binding.expire(&mut session, binding.control().deadline);
        assert!(out.protocol.is_empty());
        assert!(!out.accepted);
        assert!(
            out.diagnostics
                .iter()
                .any(|d| d.code == "ResourceWallLimit")
        );
        assert_eq!(session.active_ticket().as_ref(), Some(binding.ticket()));
        assert_eq!(
            binding.control().stop_reason(now),
            Some(StopReason::Canceled)
        );
        let late = binding.complete(
            &mut session,
            SearchCompletion::Completed {
                bestmove: Some("a2a3".into()),
            },
            binding.control().deadline,
        );
        assert!(late.protocol.is_empty());
        assert!(!late.accepted);
        let stopped = session.handle_line("stop");
        assert_eq!(stopped.protocol, ["bestmove b2b3"]);
        assert!(session.handle_line("stop").protocol.is_empty());
    }

    #[test]
    fn producer_old_tick_cannot_replace_move_after_real_deadline() {
        let actual_now = Instant::now();
        let old_tick = actual_now - Duration::from_secs(2);
        let mut session = session();
        let binding = binding(&mut session, "go movetime 1000", old_tick);
        // Previously validated owner progress is retained; queued late output is not.
        assert!(session.progress(binding.ticket(), "b2b3").accepted);
        assert!(old_tick < binding.control().deadline);
        assert!(actual_now >= binding.control().deadline);
        let out = binding.complete(
            &mut session,
            SearchCompletion::Completed {
                bestmove: Some("a2a3".into()),
            },
            old_tick,
        );
        assert!(!out.accepted);
        assert_eq!(out.protocol, ["bestmove b2b3"]);
        assert!(out.diagnostics.iter().any(|d| d.code == "SearchExpired"));
    }

    #[test]
    fn completed_infinite_worker_waits_passively_without_resource_failure() {
        let now = Instant::now();
        let mut session = session();
        let binding = binding(&mut session, "go infinite", now);
        let completed = binding.complete(
            &mut session,
            SearchCompletion::Completed {
                bestmove: Some("b2b3".into()),
            },
            now,
        );
        assert!(completed.accepted);
        assert!(completed.protocol.is_empty());
        let early = binding.expire(&mut session, now);
        assert!(!early.accepted);
        assert!(early.protocol.is_empty());
        assert!(!binding.control().cancellation.load(Ordering::Acquire));
        assert!(early.diagnostics.iter().any(|d| d.code == "EarlyDeadline"));
        let timer = binding.expire(&mut session, binding.control().deadline);
        assert!(timer.accepted);
        assert!(timer.protocol.is_empty());
        assert!(timer.diagnostics.is_empty());
        let duplicate = binding.complete(
            &mut session,
            SearchCompletion::Failed {
                code: "DuplicateProviderError".into(),
                message: "late duplicate source".into(),
            },
            binding.control().deadline,
        );
        assert!(!duplicate.accepted);
        assert!(duplicate.protocol.is_empty());
        assert!(
            duplicate
                .diagnostics
                .iter()
                .any(|d| d.code == "StaleSearch")
        );
        assert!(
            duplicate
                .diagnostics
                .iter()
                .any(|d| d.code == "SearchFailed"
                    && d.message == "DuplicateProviderError: late duplicate source")
        );
        assert_eq!(session.handle_line("stop").protocol, ["bestmove b2b3"]);
    }

    #[test]
    fn final_acceptance_deadline_retains_prior_move_and_prepared_failure() {
        // Clock injection is private: production always samples Instant::now.
        // Independent ticks model validation starting before the hard boundary
        // and finishing exactly at it without sleeps or scheduler assumptions.
        for case in 0..3 {
            let now = Instant::now();
            let mut session = session();
            let binding = binding(&mut session, "go movetime 1000", now);
            assert!(session.progress(binding.ticket(), "b2b3").accepted);
            let deadline = binding.control().deadline;
            let mut ticks = [now, deadline].into_iter();
            let read_now = || ticks.next().expect("bounded fake clock sample");
            let out = match case {
                0 => binding.progress_with_clock(&mut session, "a2a3", now, read_now),
                1 => binding.complete_with_clock(
                    &mut session,
                    SearchCompletion::Completed {
                        bestmove: Some("a2a3".into()),
                    },
                    now,
                    read_now,
                ),
                _ => binding.complete_with_clock(
                    &mut session,
                    SearchCompletion::Failed {
                        code: "ProviderFail".into(),
                        message: "prepared source survives expiry".into(),
                    },
                    now,
                    read_now,
                ),
            };
            assert!(!out.accepted);
            assert_eq!(out.protocol, ["bestmove b2b3"]);
            assert!(out.diagnostics.iter().any(|d| d.code == "SearchExpired"));
            assert!(!out.diagnostics.iter().any(|d| d.code == "AcceptanceClosed"));
            if case == 2 {
                assert!(out.diagnostics.iter().any(|d| d.code == "SearchFailed"
                    && d.message == "ProviderFail: prepared source survives expiry"));
            }
            assert_eq!(
                binding.control().stop_reason(now),
                Some(StopReason::Canceled)
            );
        }
    }

    #[test]
    fn final_acceptance_cancellation_preserves_source_without_consuming_owner() {
        let now = Instant::now();
        let mut session = session();
        let binding = binding(&mut session, "go movetime 1000", now);
        assert!(session.progress(binding.ticket(), "b2b3").accepted);
        let mut samples = 0;
        let out = binding.complete_with_clock(
            &mut session,
            SearchCompletion::Failed {
                code: "ProviderFail".into(),
                message: "prepared source survives cancel".into(),
            },
            now,
            || {
                samples += 1;
                if samples == 2 {
                    binding.cancel();
                }
                now
            },
        );
        assert!(!out.accepted);
        assert!(out.protocol.is_empty());
        assert!(out.effects.is_empty());
        assert_eq!(session.active_ticket().as_ref(), Some(binding.ticket()));
        assert!(!binding.work_completed.load(Ordering::Acquire));
        assert!(out.diagnostics.iter().any(|d| d.code == "CanceledSearch"));
        assert!(out.diagnostics.iter().any(|d| d.code == "SearchFailed"
            && d.message == "ProviderFail: prepared source survives cancel"));
        assert!(!out.diagnostics.iter().any(|d| d.code == "AcceptanceClosed"));
        assert_eq!(session.handle_line("stop").protocol, ["bestmove b2b3"]);
    }
}
