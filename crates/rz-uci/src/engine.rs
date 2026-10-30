//! Concrete Rules → UCI → search → common evaluation assembly.
//! Model loading and physical execution remain the injected factory's responsibility.

pub mod diagnostics;
mod driver_output;
use driver_output::DriverOutputEnvelope;
mod driver_start;
pub(crate) use driver_start::DriverStartSource;
#[cfg(test)]
pub(crate) use driver_start::fixture as driver_start_fixture;

use crate::{bridge::BuildSearchSettings, *};
use rz_contracts as contract;
use rz_position::{
    BoardMove, Position, PositionLimits,
    contracts::{ContractPosition, ContractState, RulesState},
};
use rz_search::{
    TreeLimits,
    driver::{CheckedPosition, SearchControl},
};
use std::{
    collections::VecDeque,
    fmt,
    io::Write,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, SyncSender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

fn failure(code: contract::ErrorCode, detail: &'static str) -> contract::ContractError {
    contract::ContractError::new(code, contract::Stage::Admission, detail)
}

/// All position forks obtain a checked fresh owner. Clone shares immutable state.
#[derive(Debug, Default)]
pub struct OwnerRegistry(AtomicU64);
impl OwnerRegistry {
    pub fn allocate(&self) -> Result<contract::OwnerId, contract::ContractError> {
        self.0
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map(|previous| contract::OwnerId(previous + 1))
            .map_err(|_| {
                failure(
                    contract::ErrorCode::ResourceExhausted,
                    "owner sequence exhausted",
                )
            })
    }
}

#[derive(Clone, Debug)]
pub struct RulesSearchPosition {
    position: Arc<ContractPosition>,
    state: ContractState,
    owners: Arc<OwnerRegistry>,
}
impl RulesSearchPosition {
    fn new(
        position: ContractPosition,
        owners: Arc<OwnerRegistry>,
    ) -> Result<Self, contract::ContractError> {
        let state = position.export()?;
        Ok(Self {
            position: Arc::new(position),
            state,
            owners,
        })
    }
    pub fn state(&self) -> &ContractState {
        &self.state
    }
    pub fn rules_position(&self) -> &Position {
        self.position.position()
    }
}
impl CheckedPosition for RulesSearchPosition {
    type Move = contract::Move;
    type Error = contract::ContractError;
    fn classify(&self) -> Result<Option<f64>, Self::Error> {
        Ok(self.state.terminal_wdl().map(|wdl| f64::from(wdl.value())))
    }
    fn legal_moves(&self) -> Result<Vec<Self::Move>, Self::Error> {
        Ok(self.state.legal_moves().moves().to_vec())
    }
    fn play(&self, movement: &Self::Move) -> Result<Self, Self::Error> {
        let fork = self
            .position
            .fork_from_view(self.owners.allocate()?, &self.state, *movement)?;
        Self::new(fork, Arc::clone(&self.owners))
    }
}

impl rz_search::contracts::ContractPosition for RulesSearchPosition {
    type State = RulesState;
    fn snapshot(&self) -> &contract::PositionSnapshot<RulesState> {
        self.state.snapshot()
    }
    fn legal(&self) -> &contract::LegalMoveView {
        self.state.legal_moves()
    }
    fn play(&self, movement: &contract::Move) -> Result<Self, contract::ContractError> {
        CheckedPosition::play(self, movement)
    }
    fn validate_authority(&self) -> Result<(), contract::ContractError> {
        if !self
            .position
            .position()
            .matches_snapshot(self.state.rules().snapshot())
            || self.state.snapshot().identity() != self.state.legal_moves().state()
        {
            return Err(failure(
                contract::ErrorCode::IdentityMismatch,
                "Rules state no longer owns its frozen legal view",
            ));
        }
        Ok(())
    }
    fn retained_bytes(&self) -> Option<usize> {
        self.state
            .rules()
            .snapshot()
            .retained_history_bytes()?
            .checked_add(self.state.legal_moves().moves().len().checked_mul(256)?)?
            .checked_add(4096)
    }
}

#[derive(Clone, Debug)]
pub struct RulesUciPort {
    owners: Arc<OwnerRegistry>,
    limits: PositionLimits,
}
impl RulesUciPort {
    pub fn new(owners: Arc<OwnerRegistry>, limits: PositionLimits) -> Self {
        Self { owners, limits }
    }
}
impl PositionPort for RulesUciPort {
    type Snapshot = RulesSearchPosition;
    type Error = contract::ContractError;
    fn prepare(
        &self,
        spec: &PositionSpec,
    ) -> Result<PreparedPosition<Self::Snapshot>, Self::Error> {
        // The entire trace lives in a temporary owner. No partial state enters Session.
        let position = match &spec.base {
            PositionBase::StartPos => Position::startpos_with_limits(self.limits)?,
            PositionBase::Fen(fen) => Position::from_fen_with_limits(fen, self.limits)?,
        };
        let mut position = ContractPosition::new(self.owners.allocate()?, position);
        for text in &spec.moves {
            let movement = contract::Move::try_from(BoardMove::from_uci(text)?)?;
            let state = position.export()?;
            position.make_from_view(&state, movement)?;
        }
        let snapshot = RulesSearchPosition::new(position, Arc::clone(&self.owners))?;
        let exact_terminal = snapshot.state.terminal_wdl().is_some();
        // Automatic terminals may still have geometric legal moves; UCI emits 0000.
        let legal_moves = if exact_terminal {
            Vec::new()
        } else {
            snapshot
                .state
                .legal_moves()
                .moves()
                .iter()
                .map(|movement| move_text(*movement))
                .collect::<Result<_, _>>()?
        };
        Ok(PreparedPosition {
            snapshot,
            legal_moves,
            exact_terminal,
        })
    }
}

pub fn move_text(movement: contract::Move) -> Result<String, contract::ContractError> {
    Ok(BoardMove::try_from(movement)?.to_string())
}

/// A clone preserves exactly one process origin and domain for every consumer.
#[derive(Clone)]
pub struct ProcessClock {
    epoch: contract::ProcessEpoch,
    origin: Instant,
    ids: Arc<rz_search::contracts::IdAllocator>,
    executions: Arc<AtomicU64>,
}
impl fmt::Debug for ProcessClock {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessClock")
            .field("epoch", &self.epoch)
            .field("origin", &self.origin)
            .finish_non_exhaustive()
    }
}
impl ProcessClock {
    pub fn new(epoch: contract::ProcessEpoch) -> Self {
        Self {
            epoch,
            origin: Instant::now(),
            ids: Arc::new(rz_search::contracts::IdAllocator::new(epoch)),
            executions: Arc::new(AtomicU64::new(0)),
        }
    }
    pub fn domain(&self) -> contract::ClockDomain {
        contract::ClockDomain(self.epoch)
    }
    pub fn epoch(&self) -> contract::ProcessEpoch {
        self.epoch
    }
    pub(crate) fn reserve_executions(&self, count: u64) -> Result<u64, contract::ContractError> {
        if count == 0 {
            return Err(failure(
                contract::ErrorCode::InvalidInput,
                "physical ID range must be finite and positive",
            ));
        }
        self.executions
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(count)
            })
            .map_err(|_| {
                failure(
                    contract::ErrorCode::ResourceExhausted,
                    "physical execution ID range exhausted",
                )
            })
    }
    pub fn b_clock(&self) -> rz_search::contract_time::InstantClock {
        rz_search::contract_time::InstantClock::new(self.domain(), self.origin)
    }
    /// Native preparation and coordinator deadlines retain the process origin
    /// established before any model loading or UCI go command.
    pub fn runtime_clock(&self) -> rz_runtime::contracts::ContractSystemClock {
        rz_runtime::contracts::ContractSystemClock::with_origin(self.epoch, self.origin)
    }
    pub fn now(&self) -> Result<contract::MonotonicTick, contract::ContractError> {
        self.tick_at(Instant::now())
    }
    pub fn tick_at(
        &self,
        instant: Instant,
    ) -> Result<contract::MonotonicTick, contract::ContractError> {
        // Windows QPC duration conversion may round a tiny negative delta to
        // zero. Compare the actual Instants before converting that delta.
        if instant < self.origin {
            return Err(failure(
                contract::ErrorCode::InvalidInput,
                "tick predates process clock origin",
            ));
        }
        let elapsed = instant.checked_duration_since(self.origin).ok_or_else(|| {
            failure(
                contract::ErrorCode::InvalidInput,
                "tick predates process clock origin",
            )
        })?;
        Ok(contract::MonotonicTick(
            u64::try_from(elapsed.as_nanos()).map_err(|_| {
                failure(
                    contract::ErrorCode::ResourceExhausted,
                    "process clock overflow",
                )
            })?,
        ))
    }
    pub fn deadline(
        &self,
        instant: Instant,
    ) -> Result<contract::Deadline, contract::ContractError> {
        Ok(contract::Deadline {
            clock: self.domain(),
            at: self.tick_at(instant)?,
        })
    }
}
impl rz_search::contract_time::ContractClock for ProcessClock {
    fn domain(&self) -> contract::ClockDomain {
        self.domain()
    }
    fn now(&self) -> Result<contract::MonotonicTick, contract::ContractError> {
        self.now()
    }
}

#[derive(Clone, Debug)]
pub struct SearchAuthority {
    #[cfg(feature = "experimental-notify")]
    signal: Option<rz_runtime::CompletionSignal>,
    current: Arc<Mutex<contract::AcceptanceScope>>,
    cancel: contract::CancelToken,
}
impl SearchAuthority {
    #[cfg(test)]
    pub(crate) fn isolated(scope: contract::AcceptanceScope) -> Self {
        Self {
            #[cfg(feature = "experimental-notify")]
            signal: None,
            current: Arc::new(Mutex::new(scope)),
            cancel: contract::CancelToken::new(),
        }
    }
    pub fn current_scope(&self) -> Result<contract::AcceptanceScope, contract::ContractError> {
        self.current.lock().map(|scope| *scope).map_err(|_| {
            failure(
                contract::ErrorCode::BackendFailure,
                "acceptance authority poisoned",
            )
        })
    }
    pub fn cancel_token(&self) -> contract::CancelToken {
        self.cancel.clone()
    }
    pub fn cancel(&self) {
        self.cancel.cancel();
        #[cfg(feature = "experimental-notify")]
        if let Some(signal) = &self.signal {
            signal.notify();
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvaluatorProfile {
    pub model: Arc<contract::ModelDescriptor>,
    pub precision: contract::PrecisionProfile,
    pub backend: contract::Digest,
    pub compute: contract::ComputeBudget,
    pub bytes: contract::ByteBudget,
}

/// Shutdown closes runtime admission and drains physical work within its finite
/// bound. It must preserve the UCI owner's root output permission: a natural
/// completion can still be queued. UCI closes shared cancellation on its own
/// stop/root/deadline/quit or completed-output acceptance boundary.
pub trait ManagedEvaluator: contract::Evaluator<RulesState> + Send {
    /// Timer source for work which has not reached a physical worker yet.
    #[cfg(feature = "experimental-notify")]
    fn wake_after(&self) -> Option<Duration> {
        None
    }
    fn shutdown(&mut self, deadline: Instant) -> Result<(), contract::ContractError>;
}
pub trait EvaluatorFactory: Send + Sync + 'static {
    #[cfg(feature = "experimental-batch")]
    fn parallelism(&self) -> usize {
        1
    }

    #[cfg(feature = "experimental-notify")]
    fn completion_signal(&self) -> Option<rz_runtime::CompletionSignal> {
        None
    }

    fn reset_game(&self) -> Result<(), contract::ContractError> {
        Ok(())
    }
    /// Optional metadata journal; no payloads or engine authority are retained.
    fn source_trace(
        &self,
    ) -> Option<rz_telemetry::source::SourceJournal<contract::CompletionContext>> {
        None
    }
    fn profile(&self) -> EvaluatorProfile;
    fn create(
        &self,
        clock: ProcessClock,
        authority: SearchAuthority,
    ) -> Result<Box<dyn ManagedEvaluator>, contract::ContractError>;
    /// Derived from the actual encoder/input, never substituted with the rules key.
    fn input_key(
        &self,
        state: &RulesState,
        legal: &[contract::Move],
    ) -> Result<contract::EvalInputKey, contract::ContractError>;
    /// Passive evidence after the existing final backup guard committed. The
    /// observer grants no publication authority and cannot perform another backup.
    fn observe_search_acceptance(
        &self,
        _evaluation: &rz_search::contracts::AcceptedEvaluation,
        _traversed_edges: usize,
    ) -> Result<(), contract::ContractError> {
        Ok(())
    }
}

/// The injected evaluator and Rules owner registry share this process's clock
/// origin and ID allocators across all roots, games and physical worker drains.
pub struct EngineProcess {
    factory: Option<Arc<dyn EvaluatorFactory>>,
    driver: Option<Arc<dyn crate::search_driver::SearchSessionDriver>>,
    owners: Arc<OwnerRegistry>,
    clock: ProcessClock,
    identity: EngineIdentity,
}
impl EngineProcess {
    /// CPU/PALS evidence is distinct from the legacy evaluator's native
    /// receipt. None means this startup does not expose search work evidence.
    pub fn search_work_receipt(
        &self,
    ) -> Result<Option<crate::search_driver::ProcessSearchWorkReceipt>, contract::ContractError>
    {
        match &self.driver {
            Some(driver) => driver.work_receipt(),
            None => Ok(None),
        }
    }
    pub fn new(
        factory: Arc<dyn EvaluatorFactory>,
        owners: Arc<OwnerRegistry>,
        clock: ProcessClock,
    ) -> Self {
        Self {
            factory: Some(factory),
            driver: None,
            owners,
            clock,
            identity: EngineIdentity {
                name: "RoveZero CPU mock integration".into(),
                author: "RoveZero contributors".into(),
            },
        }
    }
    /// Search-only startup. No neural descriptor or evaluator is constructed.
    pub fn with_search_driver(
        driver: Arc<dyn crate::search_driver::SearchSessionDriver>,
        owners: Arc<OwnerRegistry>,
        clock: ProcessClock,
    ) -> Self {
        let identity = driver.identity();
        Self {
            factory: None,
            driver: Some(driver),
            owners,
            clock,
            identity,
        }
    }
    /// Session validates the declared identity before starting protocol service.
    pub fn with_identity(mut self, identity: EngineIdentity) -> Self {
        self.identity = identity;
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EngineSettings {
    pub parser: ParserLimits,
    pub position: PositionLimits,
    pub search: BuildSearchSettings,
    pub tree: TreeLimits,
    pub max_workers: usize,
    pub shutdown_limit: Duration,
    pub final_move_policy: rz_search::tree::FinalMovePolicy,
}
impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            parser: ParserLimits {
                max_legal_moves: 512,
                ..ParserLimits::default()
            },
            position: PositionLimits::default(),
            search: BuildSearchSettings {
                time_config: rz_search::time::TimeBudgetConfig::default(),
                max_simulations: 128,
                untimed_limit: Duration::from_secs(30),
            },
            tree: TreeLimits {
                max_legal_moves: 512,
                max_nodes: 20_000,
                max_edges: 100_000,
                max_depth: 128,
                probability_tolerance: 1e-9,
            },
            max_workers: 2,
            shutdown_limit: Duration::from_secs(2),
            final_move_policy: rz_search::tree::FinalMovePolicy::Visits,
        }
    }
}
/// Fixed-size receipts preserve rejected failures without retaining foreign
/// policy/input buffers. The current common failure has context/error/recovery;
/// it has no separate, unbounded external backend cause field.
#[derive(Clone, Debug)]
pub enum WorkerDiagnosticReceipt {
    Boundary(contract::ContractError),
    DriverWork {
        kind: crate::search_driver::SearchKind,
        work: crate::search_driver::DriverWork,
    },
    RejectedFailure {
        rejection: contract::ContractError,
        failure: Box<contract::EvalFailure>,
    },
    RejectedCompletion {
        rejection: contract::ContractError,
        kind: RejectedCompletionKind,
        context: Box<contract::CompletionContext>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RejectedCompletionKind {
    Completed,
    Canceled,
    Expired,
    Stale,
}

impl PartialEq for WorkerDiagnosticReceipt {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Boundary(left), Self::Boundary(right)) => left == right,
            (Self::DriverWork { kind: lk, work: lw }, Self::DriverWork { kind: rk, work: rw }) => {
                lk == rk && lw == rw
            }
            (
                Self::RejectedFailure {
                    rejection: left,
                    failure: left_failure,
                },
                Self::RejectedFailure {
                    rejection: right,
                    failure: right_failure,
                },
            ) => {
                left == right
                    && left_failure.context == right_failure.context
                    && left_failure.error == right_failure.error
                    && left_failure.recovery == right_failure.recovery
            }
            (
                Self::RejectedCompletion {
                    rejection: left,
                    kind: left_kind,
                    context: left_context,
                },
                Self::RejectedCompletion {
                    rejection: right,
                    kind: right_kind,
                    context: right_context,
                },
            ) => left == right && left_kind == right_kind && left_context == right_context,
            _ => false,
        }
    }
}
impl Eq for WorkerDiagnosticReceipt {}
impl fmt::Display for WorkerDiagnosticReceipt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Boundary(error) => write!(formatter, "{error}"),
            Self::DriverWork { kind, work } => write!(
                formatter,
                "search_kind={kind:?}; work={work:?}; neural_receipt=not_applicable"
            ),
            Self::RejectedFailure { rejection, failure } => write!(
                formatter,
                "{rejection}; rejected failure: {}; recovery={:?}; context={:?}",
                failure.error, failure.recovery, failure.context,
            ),
            Self::RejectedCompletion {
                rejection,
                kind,
                context,
            } => write!(
                formatter,
                "{rejection}; rejected outcome={kind:?}; context={context:?}",
            ),
        }
    }
}

#[derive(Debug)]
struct DiagnosticRetentionFailure {
    error: contract::ContractError,
    receipt: WorkerDiagnosticReceipt,
}
impl fmt::Display for DiagnosticRetentionFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}; original diagnostic: {}",
            self.error, self.receipt
        )
    }
}

/// Original typed failures outlive a closed owner event channel. The protocol
/// completion is a presentation; it cannot replace these source values.
#[derive(Clone, Debug)]
pub enum WorkerFailureSource {
    Factory(contract::ContractError),
    SearchConstructor(contract::ContractError),
    Authority(contract::ContractError),
    Search(Box<rz_search::contracts::ContractSearchFailure>),
    Driver(crate::search_driver::SearchSessionFailure),
    RejectedDriverReport {
        error: contract::ContractError,
        report: Arc<crate::search_driver::SearchSessionReport>,
        diagnostic_failure: Option<(contract::ContractError, Box<WorkerDiagnosticReceipt>)>,
    },
    DiagnosticRetention {
        error: contract::ContractError,
        receipt: Box<WorkerDiagnosticReceipt>,
    },
}
impl WorkerFailureSource {
    fn physical_completion_unknown(&self) -> bool {
        matches!(self, Self::Driver(error) if error.physical_completion == crate::search_driver::DriverPhysicalCompletion::Unknown)
    }
}

#[derive(Clone, Debug)]
pub struct UndeliveredWorkerFailure {
    pub ticket: SearchTicket,
    pub completion: SearchCompletion,
    pub source: WorkerFailureSource,
}

enum WorkerCompletion {
    Completed {
        bestmove: Option<String>,
    },
    Failed {
        completion: SearchCompletion,
        source: WorkerFailureSource,
    },
}
impl WorkerCompletion {
    fn failed(source: WorkerFailureSource) -> Self {
        let completion = match &source {
            WorkerFailureSource::Factory(error)
            | WorkerFailureSource::SearchConstructor(error)
            | WorkerFailureSource::Authority(error) => failed_completion(*error),
            WorkerFailureSource::Search(error) => SearchCompletion::Failed {
                code: "SearchFailed".into(),
                message: format!("{error:?}"),
            },
            WorkerFailureSource::Driver(error) => SearchCompletion::Failed {
                code: error.code.into(),
                message: error.detail.clone(),
            },
            WorkerFailureSource::RejectedDriverReport {
                error,
                diagnostic_failure,
                ..
            } => SearchCompletion::Failed {
                code: "DriverFinalReport".into(),
                message: match diagnostic_failure {
                    Some((diagnostic_error, _)) => {
                        format!("{error}; original work retention failed: {diagnostic_error}")
                    }
                    None => error.to_string(),
                },
            },
            WorkerFailureSource::DiagnosticRetention { error, receipt } => {
                SearchCompletion::Failed {
                    code: "DiagnosticRetentionFailure".into(),
                    message: format!("{error}; original diagnostic: {receipt}"),
                }
            }
        };
        Self::Failed { completion, source }
    }
}

#[derive(Debug)]
pub enum EngineError {
    Contract(contract::ContractError),
    SearchDriverClose(crate::search_driver::SearchSessionFailure),
    Session(SessionError),
    Transport(Box<ServeError<EngineError>>),
    WorkerPanic,
    DrainTimeout,
    UndeliveredDiagnostics(Vec<(WorkerDiagnosticReceipt, u64)>),
    UndeliveredWorkerFailure(Box<UndeliveredWorkerFailure>),
    Cleanup {
        primary: Box<EngineError>,
        cleanup: Vec<EngineError>,
    },
}
impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for EngineError {}
impl From<contract::ContractError> for EngineError {
    fn from(error: contract::ContractError) -> Self {
        Self::Contract(error)
    }
}
struct ActiveBinding {
    ticket: SearchTicket,
    control: SearchControl,
    authority: Option<SearchAuthority>,
    driver_output: Option<Arc<DriverOutputEnvelope>>,
    driver_output_until: Option<Instant>,
    driver_start: Option<Arc<DriverStartSource>>,
    ponder_clock: Option<crate::rovent_ponder::RoventPonderClockOwner>,
}
type FailedPublication = Arc<Mutex<Option<Box<UndeliveredWorkerFailure>>>>;
struct Worker {
    handle: JoinHandle<Result<(), EngineError>>,
    failure: FailedPublication,
}
impl Worker {
    fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }
}
struct Owner {
    session_owner: crate::contracts::ContractSessionOwner,
    active: Option<ActiveBinding>,
    scope: Arc<Mutex<crate::contracts::SessionScope>>,
    evaluation_scope: Option<Arc<Mutex<contract::AcceptanceScope>>>,
    factory: Option<Arc<dyn EvaluatorFactory>>,
    driver: Option<Arc<dyn crate::search_driver::SearchSessionDriver>>,
    clock: ProcessClock,
    ids: Arc<rz_search::contracts::IdAllocator>,
    settings: EngineSettings,
    workers: Vec<Worker>,
    pending_failures: VecDeque<(SearchTicket, SearchCompletion)>,
    pending_bestmove: Option<PendingBestMove>,
    ponder: Option<(crate::contracts::SessionScope, Option<u64>)>,
    held_ponder_bestmove: Option<HeldPonderBestMove>,
    admitted_ponder_report: Option<AdmittedPonderReport>,
    pending_original_go: Option<(SearchTicket, Arc<crate::parser::GoLimits>)>,
    pending_close: Option<Event>,
    fence_failure: Option<EngineError>,
    diagnostics: Arc<Mutex<Vec<(WorkerDiagnosticReceipt, u64)>>>,
}
struct PendingBestMove {
    scope: crate::contracts::SessionScope,
    line: String,
    until: Instant,
    from_driver_report: bool,
    driver_output: Option<DriverOutputPublication>,
}
struct HeldPonderBestMove {
    scope: crate::contracts::SessionScope,
    line: String,
    from_driver_report: bool,
    driver_output: Option<DriverOutputPublication>,
}
struct AdmittedPonderReport {
    ticket: SearchTicket,
    scope: crate::contracts::SessionScope,
    bestmove: String,
    driver_output: Option<DriverOutputPublication>,
}
#[derive(Clone)]
struct DriverOutputPublication {
    envelope: Arc<DriverOutputEnvelope>,
    report: Option<Arc<crate::search_driver::SearchSessionReport>>,
    until: Instant,
}
impl PendingBestMove {
    fn matches_scope(&self, scope: crate::contracts::SessionScope) -> bool {
        self.scope == scope
    }
}
impl Owner {
    #[cfg(test)]
    fn admitted_ponder_output_for_tests(
        &self,
    ) -> Option<(
        &Arc<DriverOutputEnvelope>,
        &Arc<crate::search_driver::SearchSessionReport>,
    )> {
        let publication = self
            .admitted_ponder_report
            .as_ref()
            .and_then(|report| report.driver_output.as_ref())
            .or_else(|| {
                self.held_ponder_bestmove
                    .as_ref()
                    .and_then(|held| held.driver_output.as_ref())
            })?;
        Some((&publication.envelope, publication.report.as_ref()?))
    }
    fn driver_publication_for(
        &self,
        ticket: Option<&SearchTicket>,
    ) -> Result<Option<DriverOutputPublication>, contract::ContractError> {
        let Some(active) = &self.active else {
            return Ok(None);
        };
        if ticket.is_some_and(|ticket| ticket != &active.ticket) {
            return Ok(None);
        }
        let Some(envelope) = &active.driver_output else {
            return Ok(None);
        };
        let scope = self.session_owner.lifecycle_scope();
        if envelope.scope() != scope {
            return Ok(None);
        }
        let report = envelope.report_for(&active.ticket, scope)?;
        let until = active.driver_output_until.ok_or_else(|| {
            failure(
                contract::ErrorCode::IdentityMismatch,
                "actual driver Start has no original output cleanup boundary",
            )
        })?;
        Ok(Some(DriverOutputPublication {
            envelope: Arc::clone(envelope),
            report,
            until,
        }))
    }
    fn command_line(
        &mut self,
        session: &mut Session<RulesUciPort>,
        line: &str,
    ) -> (
        SessionResult<RulesSearchPosition>,
        bool,
        Option<DriverOutputPublication>,
    ) {
        let dispatched_at = Instant::now();
        let parsed = crate::parser::parse_with_zero_node_limit(
            line,
            session.parser_limits(),
            match session.node_budget_semantics() {
                crate::bridge::NodeBudgetSemantics::LegacySimulations => {
                    crate::parser::ZeroNodeLimit::Reject
                }
                crate::bridge::NodeBudgetSemantics::RoventPositionVisits => {
                    crate::parser::ZeroNodeLimit::Allow
                }
            },
        );
        // Capture once, before ponder normalization and Rules canonicalization.
        let original_go = match &parsed {
            Ok(Command::Go(limits)) => Some(Arc::new(limits.clone())),
            _ => None,
        };
        if let (Some(driver), Some(original_go)) = (&self.driver, &original_go) {
            if let Err(error) = driver.validate_go(session.snapshot().rules_position(), original_go)
            {
                let mut rejected = SessionResult {
                    accepted: false,
                    ..SessionResult::default()
                };
                rejected.diagnostics.push(Diagnostic {
                    code: "DriverGoRejected",
                    message: error.to_string(),
                });
                return (rejected, false, None);
            }
            if original_go.ponder && driver.supports_ponder_clock() {
                let mut normal = original_go.as_ref().clone();
                normal.ponder = false;
                normal.infinite = false;
                let side = match session.snapshot().rules_position().side_to_move() {
                    rz_position::Color::White => crate::bridge::SideToMove::White,
                    rz_position::Color::Black => crate::bridge::SideToMove::Black,
                };
                // The allocation's duration is fixed by the received go. A
                // future hit cannot make an oversized clock legal; reject it
                // before the old search is canceled or a new ticket exists.
                if let Err(error) = crate::bridge::PreparedSearchBinding::prepare(
                    &normal,
                    side,
                    dispatched_at,
                    self.settings.search,
                    driver.search_goal_capability(),
                    crate::bridge::NodeBudgetSemantics::RoventPositionVisits,
                ) {
                    let mut rejected = SessionResult {
                        accepted: false,
                        ..SessionResult::default()
                    };
                    rejected.diagnostics.push(Diagnostic {
                        code: "PonderClockRejected",
                        message: error.to_string(),
                    });
                    return (rejected, false, None);
                }
            }
        }
        let enabled = matches!(
            session.option_values().get("Ponder"),
            Some(OptionValue::Check(true))
        );
        let mut out = SessionResult::default();
        let mut from_driver_report = false;
        let mut released_output = None;
        match &parsed {
            Ok(Command::Go(limits)) if limits.ponder => {
                if !enabled
                    || session.node_budget_semantics()
                        == crate::bridge::NodeBudgetSemantics::LegacySimulations
                        && limits.nodes.is_some_and(|n| n > 4096)
                {
                    out.accepted = false;
                    out.diagnostics.push(Diagnostic {
                        code: "PonderRejected",
                        message:
                            "Ponder must be enabled and the node limit must fit the engine envelope"
                                .into(),
                    });
                    return (out, false, None);
                }
                // The opponent's elapsed time is never debited from the supplied own clock.
                // A finite resource wall and simulation ceiling still bound speculative work.
                let mut normalized = limits.clone();
                normalized.ponder = false;
                normalized.infinite = true;
                normalized.movetime_ms = None;
                normalized.white_time_ms = None;
                normalized.black_time_ms = None;
                normalized.white_increment_ms = None;
                normalized.black_increment_ms = None;
                normalized.moves_to_go = None;
                append_contract_result(
                    &mut out,
                    self.session_owner
                        .handle_normalized_go(session, normalized, |s| {
                            Ok(s.state.snapshot().side_to_move())
                        }),
                );
                if out.accepted {
                    self.held_ponder_bestmove = None;
                    self.admitted_ponder_report = None;
                    self.ponder = Some((self.session_owner.lifecycle_scope(), limits.nodes));
                }
            }
            Ok(Command::PonderHit)
                if self
                    .active
                    .as_ref()
                    .is_some_and(|active| active.ponder_clock.is_some()) =>
            {
                return self.hit_rovent_ponder(session, dispatched_at);
            }
            Ok(Command::PonderHit) => {
                if !self
                    .ponder
                    .as_ref()
                    .is_some_and(|(scope, _)| *scope == self.session_owner.lifecycle_scope())
                {
                    out.accepted = false;
                    out.diagnostics.push(Diagnostic {
                        code: "StalePonderHit",
                        message: "no current predicted-root search".into(),
                    });
                    return (out, false, None);
                }
                self.ponder = None;
                let ticket = session.active_ticket();
                // This baseline finishes the precomputed candidate on hit. It does not
                // restart, spend the opponent's time, or reopen an expired request.
                append_contract_result(
                    &mut out,
                    self.session_owner
                        .handle_line(session, "stop", |s| Ok(s.state.snapshot().side_to_move())),
                );
                (from_driver_report, released_output) =
                    self.release_ponder_output(ticket.as_ref(), &mut out);
            }
            Ok(Command::SetOption { name, .. })
                if name.eq_ignore_ascii_case("Ponder")
                    && (session.active_ticket().is_some() || self.ponder.is_some()) =>
            {
                out.accepted = false;
                out.diagnostics.push(Diagnostic {
                    code: "PonderRejected",
                    message: "change Ponder while idle".into(),
                });
            }
            _ => {
                let ticket = session.active_ticket();
                append_contract_result(
                    &mut out,
                    self.session_owner
                        .handle_line(session, line, |s| Ok(s.state.snapshot().side_to_move())),
                );
                if out.accepted && matches!(parsed, Ok(Command::Stop)) {
                    (from_driver_report, released_output) =
                        self.release_ponder_output(ticket.as_ref(), &mut out);
                }
                if out.accepted
                    && matches!(
                        parsed,
                        Ok(Command::Go(_)
                            | Command::Position(_)
                            | Command::NewGame
                            | Command::Stop
                            | Command::Quit)
                    )
                {
                    self.ponder = None;
                    self.held_ponder_bestmove = None;
                    self.admitted_ponder_report = None;
                }
            }
        }
        if out.accepted {
            if let Some(original_go) = original_go {
                self.pending_original_go = out.effects.iter().find_map(|effect| match effect {
                    Effect::Start { ticket, .. } => Some((ticket.clone(), original_go.clone())),
                    _ => None,
                });
            }
        }
        (out, from_driver_report, released_output)
    }

    fn hit_rovent_ponder(
        &mut self,
        session: &mut Session<RulesUciPort>,
        hit_at: Instant,
    ) -> (
        SessionResult<RulesSearchPosition>,
        bool,
        Option<DriverOutputPublication>,
    ) {
        let mut out = SessionResult::default();
        let current_scope = self.session_owner.lifecycle_scope();
        let hit = (|| {
            if !self
                .ponder
                .as_ref()
                .is_some_and(|(scope, _)| *scope == current_scope)
            {
                return Err(failure(
                    contract::ErrorCode::Stale,
                    "no current actual ponder Start",
                ));
            }
            let active = self.active.as_mut().ok_or_else(|| {
                failure(contract::ErrorCode::Stale, "actual ponder Start is absent")
            })?;
            let original = active.driver_start.as_ref().ok_or_else(|| {
                failure(
                    contract::ErrorCode::IdentityMismatch,
                    "actual ponder source is absent",
                )
            })?;
            let mut normal = original.original_go().as_ref().clone();
            normal.ponder = false;
            normal.infinite = false;
            let side = match session.snapshot().rules_position().side_to_move() {
                rz_position::Color::White => crate::bridge::SideToMove::White,
                rz_position::Color::Black => crate::bridge::SideToMove::Black,
            };
            let prepared = crate::bridge::PreparedSearchBinding::prepare(
                &normal,
                side,
                hit_at,
                self.settings.search,
                self.driver
                    .as_ref()
                    .and_then(|driver| driver.search_goal_capability()),
                crate::bridge::NodeBudgetSemantics::RoventPositionVisits,
            )
            .map_err(|_| {
                failure(
                    contract::ErrorCode::InvalidInput,
                    "normal ponder-hit time allocation is invalid",
                )
            })?;
            let disposition = active
                .ponder_clock
                .as_mut()
                .ok_or_else(|| {
                    failure(
                        contract::ErrorCode::IdentityMismatch,
                        "actual ponder clock owner is absent",
                    )
                })?
                .hit(
                    &active.ticket,
                    current_scope,
                    hit_at,
                    &prepared,
                    Instant::now(),
                )?;
            Ok((active.ticket.clone(), disposition))
        })();
        let (ticket, disposition) = match hit {
            Ok(value) => value,
            Err(error) => {
                out.accepted = false;
                out.diagnostics.push(Diagnostic {
                    code: "PonderHitRejected",
                    message: error.to_string(),
                });
                return (out, false, None);
            }
        };
        self.ponder = None;
        if session.active_ticket().as_ref() == Some(&ticket) {
            append_contract_result(
                &mut out,
                self.session_owner.release_ponder_hit(session, &ticket),
            );
            // A hit cannot reopen the expired original resource wall. Only
            // this terminal case follows the same ticket's normal stop path.
            if disposition == crate::rovent_ponder::HitDisposition::NoLiveWork
                && session.active_ticket().as_ref() == Some(&ticket)
            {
                append_contract_result(
                    &mut out,
                    self.session_owner
                        .handle_line(session, "stop", |s| Ok(s.state.snapshot().side_to_move())),
                );
            }
        } else if self.held_ponder_bestmove.is_none() {
            out.accepted = false;
            out.diagnostics.push(Diagnostic {
                code: "StalePonderHit",
                message: "the original Start has neither live work nor a held result".into(),
            });
        }
        let (reported, publication) = self.release_ponder_output(Some(&ticket), &mut out);
        (out, reported, publication)
    }
    fn release_ponder_output(
        &mut self,
        ticket: Option<&SearchTicket>,
        out: &mut SessionResult<RulesSearchPosition>,
    ) -> (bool, Option<DriverOutputPublication>) {
        // Successful infinite work stays in Session until stop/hit. Only its
        // exact admitted ticket, scope and move can acknowledge that report;
        // a coincident fallback after an earlier stop is not report consumption.
        let passive = self.admitted_ponder_report.take().filter(|report| {
            ticket == Some(&report.ticket)
                && report.scope == self.session_owner.lifecycle_scope()
                && out.protocol.iter().any(|line| {
                    line.strip_prefix("bestmove ")
                        .and_then(|text| text.split_ascii_whitespace().next())
                        == Some(report.bestmove.as_str())
                })
        });
        let from_passive_report = passive.is_some();
        let passive_output = passive.and_then(|report| report.driver_output);
        let Some(held) = self.held_ponder_bestmove.take() else {
            return (from_passive_report, passive_output);
        };
        if held.scope != self.session_owner.lifecycle_scope() {
            out.diagnostics.push(Diagnostic {
                code: "StalePonderOutput",
                message: "held predicted-root move has lost its publication authority".into(),
            });
            return (from_passive_report, passive_output);
        }
        out.protocol.push(held.line);
        (
            held.from_driver_report || from_passive_report,
            held.driver_output.or(passive_output),
        )
    }
    fn annotate_ponder(
        &self,
        session: &Session<RulesUciPort>,
        out: &mut SessionResult<RulesSearchPosition>,
    ) {
        if !matches!(
            session.option_values().get("Ponder"),
            Some(OptionValue::Check(true))
        ) {
            return;
        }
        for line in &mut out.protocol {
            let Some(text) = line
                .strip_prefix("bestmove ")
                .filter(|s| *s != "0000" && !s.contains(' '))
            else {
                continue;
            };
            let prediction = (|| -> Result<Option<String>, contract::ContractError> {
                let movement = contract::Move::try_from(BoardMove::from_uci(text)?)?;
                let child = CheckedPosition::play(session.snapshot(), &movement)?;
                if child.state.terminal_wdl().is_some() {
                    return Ok(None);
                }
                child
                    .state
                    .legal_moves()
                    .moves()
                    .first()
                    .copied()
                    .map(move_text)
                    .transpose()
            })();
            match prediction {
                Ok(Some(prediction)) => {
                    line.push_str(" ponder ");
                    line.push_str(&prediction);
                }
                Ok(None) => {}
                Err(error) => out.diagnostics.push(Diagnostic {
                    code: "PonderPredictionFailed",
                    message: format!("{error:?}"),
                }),
            }
        }
    }
    fn hold_ponder_output(
        &mut self,
        session: &Session<RulesUciPort>,
        out: &mut SessionResult<RulesSearchPosition>,
        from_driver_report: bool,
        driver_output: Option<&DriverOutputPublication>,
    ) {
        if session.is_closed() {
            self.ponder = None;
            self.held_ponder_bestmove = None;
            self.admitted_ponder_report = None;
            return;
        }
        if self.ponder.is_none() {
            return;
        }
        for line in std::mem::take(&mut out.protocol) {
            if !line.starts_with("bestmove ") {
                out.protocol.push(line);
                continue;
            }
            if self.held_ponder_bestmove.is_some() {
                out.diagnostics.push(Diagnostic {
                    code: "DuplicatePonderOutput",
                    message: "retained first completed predicted-root move".into(),
                });
            } else {
                self.held_ponder_bestmove = Some(HeldPonderBestMove {
                    scope: self.session_owner.lifecycle_scope(),
                    line,
                    from_driver_report,
                    driver_output: driver_output.cloned(),
                });
            }
        }
    }
    fn deadline_ticket(
        &self,
        delivered: Option<&SearchTicket>,
        now: Instant,
    ) -> Option<SearchTicket> {
        let active = self.active.as_ref()?;
        // Stop/root replacement can leave this physical binding alive while its
        // workers drain. Only the current, uncanceled logical owner may produce
        // a new Deadline; the common consumer guards remain authoritative.
        (self.session_owner.active_ticket() == Some(&active.ticket)
            && !active.control.cancellation.load(Ordering::Acquire)
            && delivered != Some(&active.ticket)
            && now >= active.control.deadline)
            .then(|| active.ticket.clone())
    }
    fn pending_failure(
        &mut self,
        sender: &SyncSender<Event>,
        ticket: SearchTicket,
        completion: SearchCompletion,
    ) {
        self.pending_failures.push_back((ticket, completion));
        // The owner keeps the receipt even if the bounded queue is full.
        let _ = sender.try_send(Event::RejectedInput {
            code: "OwnerWake",
            message: String::new(),
        });
    }
    fn handle(
        &mut self,
        session: &mut Session<RulesUciPort>,
        mut event: Event,
    ) -> SessionResult<RulesSearchPosition> {
        let mut out = SessionResult::default();
        let from_driver_report = matches!(
            &event,
            Event::Complete {
                completion: SearchCompletion::Completed { .. },
                ..
            }
        );
        // A joined search thread cannot certify its separate native worker.
        // Unknown physical completion closes output authority before reducing
        // stop/deadline/complete, including an already frozen legal fallback.
        let unknown = self.workers.iter().position(|worker| {
            worker
                .failure
                .lock()
                .ok()
                .and_then(|receipt| {
                    receipt
                        .as_ref()
                        .map(|receipt| receipt.source.physical_completion_unknown())
                })
                .unwrap_or(false)
        });
        if let Some(index) = unknown {
            self.cancel();
            let (source, poisoned) = take_failed_publication(&self.workers[index].failure);
            let primary = source.map(EngineError::UndeliveredWorkerFailure);
            let error = finish_errors(primary, poisoned.into_iter().collect()).unwrap_err();
            self.close_physical_fence(session, &mut out, error);
            return out;
        }
        let requested_close = matches!(&event, Event::EndOfInput)
            || matches!(&event, Event::Line(line) if matches!(crate::parse(line, self.settings.parser), Ok(Command::Quit)));
        if self.pending_bestmove.is_some() && requested_close {
            // stop/deadline has already committed this move logically. A
            // pipelined quit/EOF must not erase it while its physical fence
            // drains. Keep one closing event; never admit later input work.
            if self.pending_close.is_none() {
                self.pending_close = Some(event);
            }
            event = Event::RejectedInput {
                code: "OwnerWake",
                message: String::new(),
            };
        } else if self.pending_close.is_some()
            && matches!(&event, Event::Line(_) | Event::EndOfInput)
        {
            event = Event::RejectedInput {
                code: "OwnerWake",
                message: String::new(),
            };
        }
        let completion_receipt = match &event {
            Event::Complete {
                ticket,
                completion: completion @ SearchCompletion::Failed { .. },
            } => Some((ticket.clone(), completion.clone())),
            _ => None,
        };
        let passive_report = match &event {
            Event::Complete {
                ticket,
                completion:
                    SearchCompletion::Completed {
                        bestmove: Some(bestmove),
                    },
            } => Some((ticket.clone(), bestmove.clone())),
            _ => None,
        };
        match self.diagnostics.lock() {
            Ok(mut receipts) => {
                for (error, count) in std::mem::take(&mut *receipts) {
                    out.diagnostics.push(Diagnostic {
                        code: "SearchDiagnostic",
                        message: format!("{error}; occurrences={count}"),
                    });
                }
            }
            Err(_) => {
                self.cancel();
                out.diagnostics.push(Diagnostic {
                    code: "DiagnosticFailure",
                    message: "worker diagnostic receipt owner poisoned".into(),
                });
                append_result(&mut out, session.end_of_input());
            }
        }
        while let Some((ticket, completion)) = self.pending_failures.pop_front() {
            let mut failed = SessionResult::default();
            append_contract_result(
                &mut failed,
                self.session_owner
                    .handle_event(session, Event::Complete { ticket, completion }),
            );
            self.annotate_ponder(session, &mut failed);
            self.hold_ponder_output(session, &mut failed, false, None);
            self.defer_bestmove(session, &mut failed, false, None);
            append_result(&mut out, failed);
        }
        let completed_output = match passive_report.as_ref() {
            Some((ticket, _)) => match self.driver_publication_for(Some(ticket)) {
                Ok(publication) => publication,
                Err(error) => {
                    self.close_physical_fence(session, &mut out, error.into());
                    return out;
                }
            },
            None => None,
        };
        let (mut next, held_driver_report, held_output) = match event {
            Event::RejectedInput {
                code: "OwnerWake", ..
            } => (SessionResult::default(), false, None),
            Event::Line(line) => self.command_line(session, &line),
            event => {
                let next = self.session_owner.handle_event(session, event);
                let mut out = SessionResult::default();
                append_contract_result(&mut out, next);
                (out, false, None)
            }
        };
        // Enqueueing is not consumption. A typed failed publication stays with
        // its bounded worker slot until this owner has handled the exact ticket.
        if let Some((ticket, completion)) = completion_receipt {
            let diagnosed = match &completion {
                SearchCompletion::Failed { code, message } => {
                    next.diagnostics.iter().any(|diagnostic| {
                        diagnostic.code == "SearchFailed"
                            && diagnostic.message == format!("{code}: {message}")
                    })
                }
                SearchCompletion::Completed { .. } => false,
            };
            for worker in &self.workers {
                match worker.failure.lock() {
                    Ok(mut receipt) => {
                        if diagnosed
                            && receipt.as_ref().is_some_and(|failure| {
                                failure.ticket == ticket && failure.completion == completion
                            })
                        {
                            receipt.take();
                        }
                    }
                    Err(_) => {
                        self.cancel();
                        out.diagnostics.push(Diagnostic {
                            code: "DiagnosticFailure",
                            message: "worker failed publication owner poisoned".into(),
                        });
                        append_result(&mut out, session.end_of_input());
                    }
                }
            }
        }
        // B closes common/tree cancellation before changing its generations.
        match self.scope.lock() {
            Ok(mut scope) => *scope = self.session_owner.lifecycle_scope(),
            Err(_) => {
                self.cancel();
                out.diagnostics.push(Diagnostic {
                    code: "AuthorityFailure",
                    message: "acceptance authority poisoned".into(),
                });
                append_result(&mut out, session.end_of_input());
            }
        }
        if let Some(current) = &self.evaluation_scope {
            match current.lock() {
                Ok(mut scope) => *scope = self.session_owner.scope(),
                Err(_) => {
                    self.cancel();
                    out.diagnostics.push(Diagnostic {
                        code: "AuthorityFailure",
                        message: "evaluation authority poisoned".into(),
                    });
                    append_result(&mut out, session.end_of_input());
                }
            }
        }
        // A Completed event can be rejected by the common hard-deadline guard
        // and replaced with a frozen fallback. Only the actual admitted
        // completion outcome may acknowledge consumption of a CPU report.
        let admitted_driver_report = (from_driver_report && next.accepted) || held_driver_report;
        let driver_output = held_output.or(completed_output);
        if from_driver_report
            && next.accepted
            && let Some((ticket, bestmove)) = passive_report
            && session.active_ticket().as_ref() == Some(&ticket)
            && self
                .ponder
                .as_ref()
                .is_some_and(|(scope, _)| *scope == self.session_owner.lifecycle_scope())
        {
            self.admitted_ponder_report = Some(AdmittedPonderReport {
                ticket,
                scope: self.session_owner.lifecycle_scope(),
                bestmove,
                driver_output: driver_output.clone(),
            });
        }
        self.annotate_ponder(session, &mut next);
        self.hold_ponder_output(
            session,
            &mut next,
            admitted_driver_report,
            driver_output.as_ref(),
        );
        self.defer_bestmove(session, &mut next, admitted_driver_report, driver_output);
        append_result(&mut out, next);
        self.finish_physical_output(session, &mut out);
        if !session.is_closed()
            && self.pending_bestmove.is_none()
            && self.workers.is_empty()
            && let Some(close) = self.pending_close.take()
        {
            // The joined worker may have retained its final work receipt
            // after the first diagnostic drain. Process closing through
            // the same reducer once, preserving that evidence and errors.
            append_result(&mut out, self.handle(session, close));
        }
        out
    }

    /// Session closes logical admission and freezes its last valid move at the
    /// stop/deadline boundary. Publishing that move waits for every owned worker
    /// to drain and join successfully. Ready/input handling stays cooperative.
    fn defer_bestmove(
        &mut self,
        session: &mut Session<RulesUciPort>,
        out: &mut SessionResult<RulesSearchPosition>,
        from_driver_report: bool,
        driver_output: Option<DriverOutputPublication>,
    ) {
        let scope = self.session_owner.lifecycle_scope();
        if session.is_closed()
            || self
                .pending_bestmove
                .as_ref()
                .is_some_and(|p| !p.matches_scope(scope))
        {
            self.pending_bestmove = None;
        }
        if session.is_closed() {
            return;
        }
        let driver_output = match driver_output {
            Some(publication) => Some(publication),
            None => match self.driver_publication_for(None) {
                Ok(publication) => publication,
                Err(error) => {
                    self.close_physical_fence(session, out, error.into());
                    return;
                }
            },
        };
        // A Rules-only bestmove has no actual Start or driver report. Every
        // actual driver Start uses this publisher even if it was already reaped.
        if self.workers.is_empty() && driver_output.is_none() {
            return;
        }
        for line in std::mem::take(&mut out.protocol) {
            if !line.starts_with("bestmove ") {
                out.protocol.push(line);
                continue;
            }
            if self.pending_bestmove.is_some() {
                self.close_physical_fence(
                    session,
                    out,
                    failure(
                        contract::ErrorCode::IdentityMismatch,
                        "duplicate deferred bestmove",
                    )
                    .into(),
                );
                return;
            }
            let until = driver_output
                .as_ref()
                .map(|output| output.until)
                .or_else(|| Instant::now().checked_add(self.settings.shutdown_limit));
            let Some(until) = until else {
                self.close_physical_fence(
                    session,
                    out,
                    failure(
                        contract::ErrorCode::ResourceExhausted,
                        "output drain deadline overflow",
                    )
                    .into(),
                );
                return;
            };
            self.pending_bestmove = Some(PendingBestMove {
                scope,
                line,
                until,
                from_driver_report,
                driver_output: driver_output.clone(),
            });
        }
    }

    fn finish_physical_output(
        &mut self,
        session: &mut Session<RulesUciPort>,
        out: &mut SessionResult<RulesSearchPosition>,
    ) {
        if session.is_closed() {
            self.pending_bestmove = None;
            return;
        }
        if self.pending_bestmove.is_none() {
            return;
        }
        if let Err(error) = self.reap() {
            self.close_physical_fence(session, out, error);
            return;
        }
        if self.workers.is_empty() {
            let pending = self.pending_bestmove.take().unwrap();
            if pending.scope != self.session_owner.lifecycle_scope() {
                self.close_physical_fence(
                    session,
                    out,
                    failure(
                        contract::ErrorCode::Stale,
                        "actual Start output lost its original lifecycle scope",
                    )
                    .into(),
                );
                return;
            }
            let bestmove = pending
                .line
                .strip_prefix("bestmove ")
                .and_then(|text| text.split_ascii_whitespace().next())
                .unwrap_or("");
            let strategy = if let Some(publication) = &pending.driver_output {
                match (&publication.report, pending.from_driver_report) {
                    (Some(report), _) => publication.envelope.mate_for(
                        report,
                        pending.scope,
                        bestmove,
                        pending.from_driver_report,
                    ),
                    (None, false) => Ok(None),
                    (None, true) => Err(failure(
                        contract::ErrorCode::IdentityMismatch,
                        "reported driver output has no actual Start final report",
                    )),
                }
            } else {
                Ok(None)
            };
            let strategy = match strategy {
                Ok(strategy) => strategy,
                Err(error) => {
                    self.close_physical_fence(session, out, error.into());
                    return;
                }
            };
            if let (Some(driver), crate::contracts::SessionScope::Search(authority)) =
                (&self.driver, pending.scope)
            {
                if let Err(error) =
                    driver.accept_uci_output(authority, bestmove, pending.from_driver_report)
                {
                    self.close_physical_fence(session, out, error.into());
                    return;
                }
            }
            if let Some(strategy) = strategy {
                if let Some(mate) = strategy.conclusion().exact_uci_mate() {
                    out.protocol
                        .push(format!("info score mate {mate} pv {bestmove}"));
                }
            }
            out.protocol.push(pending.line);
        } else if Instant::now() >= self.pending_bestmove.as_ref().unwrap().until {
            self.close_physical_fence(session, out, EngineError::DrainTimeout);
        }
    }

    fn close_physical_fence(
        &mut self,
        session: &mut Session<RulesUciPort>,
        out: &mut SessionResult<RulesSearchPosition>,
        error: EngineError,
    ) {
        self.pending_bestmove = None;
        self.pending_close = None;
        self.ponder = None;
        self.held_ponder_bestmove = None;
        self.admitted_ponder_report = None;
        out.protocol.retain(|line| !line.starts_with("bestmove "));
        out.diagnostics.push(Diagnostic {
            code: "PhysicalFenceFailure",
            message: error.to_string(),
        });
        self.fence_failure = Some(error);
        append_result(out, session.end_of_input());
    }
    fn cancel(&self) {
        if let Some(active) = &self.active {
            // The original Start's optional Collection gate closes before
            // signaling its physical search cancellation; neither waits.
            if let Some(start) = &active.driver_start { start.abort_collection(); }
            active.control.cancel();
            if let Some(clock) = &active.ponder_clock {
                clock.close();
            }
            if let Some(authority) = &active.authority {
                authority.cancel();
            }
        }
    }
    fn reap(&mut self) -> Result<(), EngineError> {
        self.reap_workers(false)
    }
    fn reap_workers(&mut self, closing: bool) -> Result<(), EngineError> {
        let mut index = 0;
        while index < self.workers.len() {
            if self.workers[index].is_finished() {
                if !closing
                    && self.workers[index]
                        .failure
                        .lock()
                        .map_err(|_| {
                            failure(
                                contract::ErrorCode::BackendFailure,
                                "worker failed publication owner poisoned",
                            )
                        })?
                        .is_some()
                {
                    // Keep the unacknowledged slot in max_workers. Starting more
                    // workers cannot grow an orphaned failed-publication history.
                    index += 1;
                    continue;
                }
                let worker = self.workers.swap_remove(index);
                let (unconsumed, poisoned) = take_failed_publication(&worker.failure);
                let joined = worker
                    .handle
                    .join()
                    .map_err(|_| EngineError::WorkerPanic)
                    .and_then(|result| result);
                let joined = match (unconsumed.as_deref(), joined) {
                    (Some(receipt), Err(error)) => {
                        discard_duplicate_delivery_failure(error, receipt).map_or(Ok(()), Err)
                    }
                    (_, result) => result,
                };
                let mut errors = Vec::new();
                if let Some(receipt) = unconsumed {
                    errors.push(EngineError::UndeliveredWorkerFailure(receipt));
                }
                if let Some(error) = poisoned {
                    errors.push(error);
                }
                if let Err(error) = joined {
                    errors.push(error);
                }
                finish_errors(None, errors)?;
            } else {
                index += 1;
            }
        }
        Ok(())
    }
    fn collect_unconsumed_failures(&self) -> Vec<EngineError> {
        let mut errors = Vec::new();
        for worker in &self.workers {
            let (receipt, poisoned) = take_failed_publication(&worker.failure);
            if let Some(receipt) = receipt {
                errors.push(EngineError::UndeliveredWorkerFailure(receipt));
            }
            if let Some(error) = poisoned {
                errors.push(error);
            }
        }
        errors
    }
    fn dispatch(
        &mut self,
        effect: Effect<RulesSearchPosition>,
        sender: &SyncSender<Event>,
    ) -> Result<(), EngineError> {
        match effect {
            Effect::Cancel { ticket, .. } => {
                if self
                    .active
                    .as_ref()
                    .is_some_and(|active| active.ticket == ticket)
                {
                    self.cancel();
                }
            }
            Effect::NewGame => {
                self.cancel();
                if let Some(factory) = &self.factory {
                    factory.reset_game()?;
                }
                if let Some(driver) = &self.driver {
                    driver.reset_game()?;
                }
            }
            Effect::Shutdown => self.cancel(),
            Effect::Start {
                ticket,
                snapshot,
                limits,
                ..
            } => {
                let original_go = self
                    .pending_original_go
                    .take()
                    .and_then(|(original_ticket, go)| (original_ticket == ticket).then_some(go));
                self.reap()?;
                if self.workers.len() >= self.settings.max_workers {
                    self.pending_failure(
                        sender,
                        ticket,
                        SearchCompletion::Failed {
                            code: "WorkerLimit".into(),
                            message: "previous physical worker has not drained".into(),
                        },
                    );
                    return Ok(());
                }
                let mut control = self.session_owner.control().cloned().ok_or_else(|| {
                    failure(
                        contract::ErrorCode::IdentityMismatch,
                        "UCI start has no checked B control",
                    )
                })?;
                if let Some((_, Some(nodes))) = self.ponder {
                    if self.driver.as_ref().is_none_or(|driver| {
                        driver.node_budget_semantics()
                            == crate::bridge::NodeBudgetSemantics::LegacySimulations
                    }) {
                        control.max_simulations = control.max_simulations.min(nodes);
                    }
                }
                let token = self
                    .session_owner
                    .cancellation()
                    .ok_or_else(|| {
                        failure(
                            contract::ErrorCode::IdentityMismatch,
                            "UCI start has no common cancellation",
                        )
                    })?
                    .token();
                let deadlines = *self.session_owner.deadlines().ok_or_else(|| {
                    failure(
                        contract::ErrorCode::IdentityMismatch,
                        "UCI start has no common deadlines",
                    )
                })?;
                if let Some(driver) = &self.driver {
                    let crate::contracts::SessionScope::Search(search_authority) =
                        self.session_owner.active_lifecycle_scope().ok_or_else(|| {
                            failure(
                                contract::ErrorCode::IdentityMismatch,
                                "UCI driver start has no search identity",
                            )
                        })?
                    else {
                        return Err(failure(
                            contract::ErrorCode::IdentityMismatch,
                            "search-only driver received a neural scope",
                        )
                        .into());
                    };
                    let shutdown_deadline = control
                        .deadline
                        .checked_add(self.settings.shutdown_limit)
                        .ok_or_else(|| {
                            failure(
                                contract::ErrorCode::ResourceExhausted,
                                "search driver shutdown deadline overflow",
                            )
                        })?;
                    let mut context = crate::search_driver::SearchSessionContext {
                        authority: search_authority,
                        original_go,
                        original_start: None,
                        effective_clock: None,
                        original_time_budget: self
                            .session_owner
                            .active_binding()
                            .map(|binding| *binding.budget()),
                        root_moves: limits
                            .searchmoves
                            .as_ref()
                            .map(|moves| {
                                moves
                                    .iter()
                                    .map(|movement| {
                                        movement.parse::<BoardMove>().map_err(|_| {
                                            failure(
                                                contract::ErrorCode::IdentityMismatch,
                                                "sealed root move could not be decoded",
                                            )
                                        })
                                    })
                                    .collect::<Result<Vec<_>, _>>()
                                    .map(Arc::from)
                            })
                            .transpose()?,
                        goal: contract::rovent::SearchGoal {
                            depth_plies: limits.depth_plies,
                            mate_moves: limits.mate_moves,
                        },
                        requested_nodes: limits.nodes,
                        control: control.clone(),
                        cancellation: token,
                        shutdown_deadline,
                        current: Arc::clone(&self.scope),
                    };
                    let ponder_clock = if driver.supports_ponder_clock()
                        && context.original_go.as_ref().is_some_and(|go| go.ponder)
                    {
                        let original_budget = context.original_time_budget.ok_or_else(|| {
                            failure(
                                contract::ErrorCode::IdentityMismatch,
                                "actual ponder Start has no original TimeBudget",
                            )
                        })?;
                        let (owner, handle) = crate::rovent_ponder::RoventPonderClockOwner::new(
                            ticket.clone(),
                            search_authority,
                            original_budget,
                            shutdown_deadline,
                            &self.clock,
                        )?;
                        context.effective_clock = Some(handle);
                        Some(owner)
                    } else {
                        None
                    };
                    let original_start = DriverStartSource::from_start(
                        &ticket,
                        snapshot.rules_position(),
                        self.session_owner.active_binding().ok_or_else(|| {
                            failure(
                                contract::ErrorCode::IdentityMismatch,
                                "actual driver Start has no original binding",
                            )
                        })?,
                        &context,
                        &self.clock,
                    )?;
                    context.original_start = Some(Arc::clone(&original_start));
                    if let Err(error) = self.session_owner.bind_original_start(&ticket, Arc::clone(&original_start)) {
                        // This optional Collection association is not part of
                        // driver admission; retain the failure for its later
                        // binding and leave the existing search path intact.
                        original_start.mark_collection_owner_binding_failed(error);
                    }
                    let driver_output = Arc::new(DriverOutputEnvelope::new(
                        original_start.ticket().clone(),
                        search_authority,
                        snapshot.rules_position(),
                        context.root_moves.as_deref(),
                    )?);
                    self.active = Some(ActiveBinding {
                        ticket: ticket.clone(),
                        control,
                        authority: None,
                        driver_output: Some(Arc::clone(&driver_output)),
                        driver_output_until: Some(shutdown_deadline),
                        driver_start: Some(original_start),
                        ponder_clock,
                    });
                    let driver = Arc::clone(driver);
                    let events = sender.clone();
                    let diagnostics = Arc::clone(&self.diagnostics);
                    let shutdown_limit = self.settings.shutdown_limit;
                    let publication = Arc::new(Mutex::new(None));
                    let worker_publication = Arc::clone(&publication);
                    let handle = thread::spawn(move || {
                        let mut progress = |movement: BoardMove| {
                            if context.accepts().unwrap_or(false) {
                                let _ = events.try_send(Event::Progress {
                                    ticket: ticket.clone(),
                                    bestmove: movement.to_string(),
                                });
                            }
                        };
                        let completion =
                            match driver.run(snapshot.rules_position(), &context, &mut progress) {
                                Ok(report) => {
                                    // Charge/preserve the actual returned work before deciding
                                    // whether its final move/proof is publishable.
                                    let work_retention = retain_diagnostic(
                                        &diagnostics,
                                        WorkerDiagnosticReceipt::DriverWork {
                                            kind: driver.kind(),
                                            work: report.work.clone(),
                                        },
                                    );
                                    let report = match driver_output.record_retaining(report) {
                                        Ok(report) => report,
                                        Err((report, error)) => {
                                            return finish_worker(
                                                &events,
                                                ticket,
                                                WorkerCompletion::failed(
                                                    WorkerFailureSource::RejectedDriverReport {
                                                        error,
                                                        report,
                                                        diagnostic_failure: work_retention
                                                            .err()
                                                            .map(|failure| {
                                                                (
                                                                    failure.error,
                                                                    Box::new(failure.receipt),
                                                                )
                                                            }),
                                                    },
                                                ),
                                                &worker_publication,
                                                None,
                                                shutdown_limit,
                                            );
                                        }
                                    };
                                    match work_retention {
                                        Ok(()) => {
                                            let _ = events.try_send(Event::RejectedInput {
                                                code: "OwnerWake",
                                                message: String::new(),
                                            });
                                            WorkerCompletion::Completed {
                                                bestmove: report
                                                    .best_move
                                                    .map(|movement| movement.to_string()),
                                            }
                                        }
                                        Err(error) => WorkerCompletion::failed(
                                            WorkerFailureSource::DiagnosticRetention {
                                                error: error.error,
                                                receipt: Box::new(error.receipt),
                                            },
                                        ),
                                    }
                                }
                                Err(error) => {
                                    WorkerCompletion::failed(WorkerFailureSource::Driver(error))
                                }
                            };
                        finish_worker(
                            &events,
                            ticket,
                            completion,
                            &worker_publication,
                            None,
                            shutdown_limit,
                        )
                    });
                    self.workers.push(Worker {
                        handle,
                        failure: publication,
                    });
                    return Ok(());
                }
                let scope = self.session_owner.active_scope().ok_or_else(|| {
                    failure(
                        contract::ErrorCode::IdentityMismatch,
                        "UCI start has no current scope",
                    )
                })?;
                let factory = self.factory.as_ref().ok_or_else(|| {
                    failure(
                        contract::ErrorCode::IdentityMismatch,
                        "PUCT start has no evaluator factory",
                    )
                })?;
                let authority = SearchAuthority {
                    #[cfg(feature = "experimental-notify")]
                    signal: factory.completion_signal(),
                    current: Arc::clone(self.evaluation_scope.as_ref().ok_or_else(|| {
                        failure(
                            contract::ErrorCode::IdentityMismatch,
                            "PUCT start has no evaluation scope",
                        )
                    })?),
                    cancel: token.clone(),
                };
                self.active = Some(ActiveBinding {
                    ticket: ticket.clone(),
                    control: control.clone(),
                    authority: Some(authority.clone()),
                    driver_output: None,
                    driver_output_until: None,
                    driver_start: None,
                    ponder_clock: None,
                });
                let profile = factory.profile();
                let config = rz_search::contracts::ContractSearchConfig {
                    scope,
                    model: Arc::clone(&profile.model),
                    precision: profile.precision,
                    compute: profile.compute,
                    bytes: profile.bytes,
                    policy_tolerance: 1e-5,
                    wdl_tolerance: 1e-5,
                    cancellation: token,
                    deadlines,
                    ids: Arc::clone(&self.ids),
                    max_simulations: control.max_simulations,
                    tree_limits: self.settings.tree,
                };
                let factory = Arc::clone(factory);
                let restricted_root_moves = limits
                    .searchmoves
                    .as_ref()
                    .map(|moves| {
                        moves
                            .iter()
                            .map(|movement| {
                                let movement = movement.parse::<BoardMove>().map_err(|_| {
                                    failure(
                                        contract::ErrorCode::IdentityMismatch,
                                        "sealed PUCT root move could not be decoded",
                                    )
                                })?;
                                contract::Move::try_from(movement)
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()?;
                let clock = self.clock.clone();
                let shutdown_limit = self.settings.shutdown_limit;
                let self_final_move_policy = self.settings.final_move_policy;
                let events = sender.clone();
                let diagnostics = Arc::clone(&self.diagnostics);
                let publication = Arc::new(Mutex::new(None));
                let worker_publication = Arc::clone(&publication);
                let handle = thread::spawn(move || {
                    let mut runtime = match factory.create(clock.clone(), authority.clone()) {
                        Ok(runtime) => runtime,
                        Err(error) => {
                            return finish_worker(
                                &events,
                                ticket,
                                WorkerCompletion::failed(WorkerFailureSource::Factory(error)),
                                &worker_publication,
                                None,
                                shutdown_limit,
                            );
                        }
                    };
                    let mut search =
                        match rz_search::contracts::ContractSearch::new(snapshot, config) {
                            Ok(search) => search,
                            Err(error) => {
                                return finish_worker(
                                    &events,
                                    ticket,
                                    WorkerCompletion::failed(
                                        WorkerFailureSource::SearchConstructor(error),
                                    ),
                                    &worker_publication,
                                    Some(runtime.as_mut()),
                                    shutdown_limit,
                                );
                            }
                        };
                    if let Some(root_moves) = &restricted_root_moves {
                        if let Err(error) = search.set_root_moves(root_moves) {
                            return finish_worker(
                                &events,
                                ticket,
                                WorkerCompletion::failed(WorkerFailureSource::SearchConstructor(
                                    error,
                                )),
                                &worker_publication,
                                Some(runtime.as_mut()),
                                shutdown_limit,
                            );
                        }
                    }
                    #[cfg(feature = "experimental-batch")]
                    if let Err(error) = search.set_parallelism(factory.parallelism()) {
                        return finish_worker(
                            &events,
                            ticket,
                            WorkerCompletion::failed(WorkerFailureSource::SearchConstructor(error)),
                            &worker_publication,
                            Some(runtime.as_mut()),
                            shutdown_limit,
                        );
                    }
                    search.set_source_trace(factory.source_trace());
                    if let Err(error) = search.set_final_move_policy(self_final_move_policy) {
                        return finish_worker(
                            &events,
                            ticket,
                            WorkerCompletion::failed(WorkerFailureSource::SearchConstructor(error)),
                            &worker_publication,
                            Some(runtime.as_mut()),
                            shutdown_limit,
                        );
                    }
                    let mut previous_move = None;
                    let mut authority_error = None;
                    let mut diagnostic_failure = None;
                    while !search.is_finished() {
                        #[cfg(feature = "experimental-notify")]
                        let wake_version = authority.signal.as_ref().map(|s| s.version());
                        if control.cancellation.load(Ordering::Acquire) {
                            authority.cancel();
                        }
                        if let Err(error) = authority.current_scope() {
                            authority.cancel();
                            authority_error = Some(error);
                        }
                        let mut port = RuntimePort(runtime.as_mut());
                        let event = search.pump(
                            &mut port,
                            &clock,
                            || {
                                authority
                                    .current_scope()
                                    .unwrap_or(contract::AcceptanceScope {
                                        root: contract::RootGeneration(scope.root.0 ^ 1),
                                        ..scope
                                    })
                            },
                            |position, encoding| {
                                if encoding != profile.model.encoding() {
                                    return Err(failure(
                                        contract::ErrorCode::IdentityMismatch,
                                        "search changed encoder descriptor",
                                    ));
                                }
                                factory.input_key(
                                    position.state.rules(),
                                    position.state.legal_moves().moves(),
                                )
                            },
                        );
                        if let rz_search::contracts::ContractPumpEvent::Accepted {
                            evaluation: Some(evaluation),
                            traversed_edges,
                            ..
                        } = &event
                            && let Err(error) = factory
                                .observe_search_acceptance(evaluation.as_ref(), *traversed_edges)
                        {
                            // The already committed tree remains untouched. Preserve
                            // an observer failure and close future logical admission.
                            authority.cancel();
                            authority_error.get_or_insert(error);
                        }
                        if let Err(error) = consume_pump_diagnostic(&event, &diagnostics, &events) {
                            authority.cancel();
                            diagnostic_failure = Some(error);
                        }
                        let best = if cfg!(feature = "experimental-best-move") {
                            search.best_move()
                        } else {
                            search.outcome().best_move
                        };
                        if best != previous_move {
                            previous_move = best;
                            if let Some(bestmove) =
                                best.and_then(|movement| move_text(movement).ok())
                            {
                                let _ = events.try_send(Event::Progress {
                                    ticket: ticket.clone(),
                                    bestmove,
                                });
                            }
                        }
                        if matches!(event, rz_search::contracts::ContractPumpEvent::Waiting) {
                            #[cfg(feature = "experimental-notify")]
                            if let (Some(signal), Some(observed)) =
                                (&authority.signal, wake_version)
                            {
                                // Owner blocking only: backend poll remains cooperative.
                                // Hard deadline bounds waits even without a completion.
                                if !authority.cancel_token().is_canceled() {
                                    signal.wait_changed(
                                        observed,
                                        control
                                            .deadline
                                            .saturating_duration_since(Instant::now())
                                            .min(
                                                runtime
                                                    .wake_after()
                                                    .unwrap_or(Duration::from_secs(1)),
                                            ),
                                    );
                                }
                            } else {
                                // Scripted/mock clocks have no asynchronous completion source.
                                thread::sleep(Duration::from_millis(1));
                            }
                            #[cfg(not(feature = "experimental-notify"))]
                            thread::sleep(Duration::from_millis(1));
                        }
                    }
                    let outcome = search.outcome();
                    let completion = match (diagnostic_failure, authority_error, outcome.status) {
                        (Some(error), _, _) => {
                            WorkerCompletion::failed(WorkerFailureSource::DiagnosticRetention {
                                error: error.error,
                                receipt: Box::new(error.receipt),
                            })
                        }
                        (_, Some(error), _) => {
                            WorkerCompletion::failed(WorkerFailureSource::Authority(error))
                        }
                        (_, _, rz_search::contracts::ContractSearchStatus::Failed(error)) => {
                            WorkerCompletion::failed(WorkerFailureSource::Search(Box::new(error)))
                        }
                        _ => WorkerCompletion::Completed {
                            bestmove: outcome
                                .best_move
                                .and_then(|movement| move_text(movement).ok()),
                        },
                    };
                    finish_worker(
                        &events,
                        ticket,
                        completion,
                        &worker_publication,
                        Some(runtime.as_mut()),
                        shutdown_limit,
                    )
                });
                self.workers.push(Worker {
                    handle,
                    failure: publication,
                });
            }
            Effect::OptionChanged { name, .. } if name.eq_ignore_ascii_case("Ponder") => {}
            Effect::OptionChanged { .. } | Effect::Button { .. } => {
                return Err(failure(
                    contract::ErrorCode::UnsupportedContract,
                    "bootstrap did not register backend options",
                )
                .into());
            }
        }
        Ok(())
    }
}
fn finish_worker(
    sender: &SyncSender<Event>,
    ticket: SearchTicket,
    completion: WorkerCompletion,
    publication: &Mutex<Option<Box<UndeliveredWorkerFailure>>>,
    runtime: Option<&mut dyn ManagedEvaluator>,
    shutdown_limit: Duration,
) -> Result<(), EngineError> {
    let (completion, source) = match completion {
        WorkerCompletion::Completed { bestmove } => {
            (SearchCompletion::Completed { bestmove }, None)
        }
        WorkerCompletion::Failed { completion, source } => (completion, Some(source)),
    };
    let failed = source.is_some();
    let physical_unknown = source
        .as_ref()
        .is_some_and(WorkerFailureSource::physical_completion_unknown);
    let mut errors = Vec::new();
    if let Some(source) = source {
        let receipt = Box::new(UndeliveredWorkerFailure {
            ticket: ticket.clone(),
            completion: completion.clone(),
            source,
        });
        match publication.lock() {
            Ok(mut slot) if slot.is_none() => *slot = Some(receipt),
            Ok(_) => {
                errors.push(EngineError::UndeliveredWorkerFailure(receipt));
                errors.push(
                    failure(
                        contract::ErrorCode::ResourceExhausted,
                        "worker single failed-publication slot already occupied",
                    )
                    .into(),
                );
            }
            Err(_) => {
                errors.push(EngineError::UndeliveredWorkerFailure(receipt));
                errors.push(
                    failure(
                        contract::ErrorCode::BackendFailure,
                        "worker failed publication owner poisoned",
                    )
                    .into(),
                );
            }
        }
    }
    // Keep the original owner-reclaimable until actual owner consumption/reap,
    // including the gap between returning this closure and handle.is_finished.
    let cleanup = runtime.and_then(|runtime| shutdown(runtime, shutdown_limit).err());
    if physical_unknown {
        // Keep the original typed source in its owner-reclaimable slot. No
        // normal completion/fallback acknowledgment may escape this fence.
        let _ = sender.try_send(Event::RejectedInput {
            code: "OwnerWake",
            message: String::new(),
        });
        errors.push(
            failure(
                contract::ErrorCode::BackendFailure,
                "search driver physical completion remains unconfirmed",
            )
            .into(),
        );
        if let Some(error) = cleanup {
            errors.push(EngineError::Contract(error));
        }
        return finish_errors(None, errors);
    }
    // Natural completion follows physical drain. The owner also fences frozen
    // stop/deadline output until this closure has successfully returned and all
    // owned workers have joined; a cleanup error never acknowledges completion.
    let delivery_failed =
        errors.is_empty() && sender.send(Event::Complete { ticket, completion }).is_err() && failed;
    if delivery_failed {
        // Failed sends are reclaimed by the join result. Successful sends keep
        // the slot until the owner acknowledges consumption or closing reaps it.
        let (receipt, poisoned) = copy_failed_publication(publication);
        if let Some(receipt) = receipt {
            errors.push(EngineError::UndeliveredWorkerFailure(receipt));
        }
        if let Some(error) = poisoned {
            errors.push(error);
        }
    }
    // Do not return early on failed delivery: physical shutdown always runs when
    // a runtime was created, and its failure remains separate from the source.
    if let Some(error) = cleanup {
        errors.push(EngineError::Contract(error));
    }
    finish_errors(None, errors)
}

fn take_failed_publication(
    publication: &Mutex<Option<Box<UndeliveredWorkerFailure>>>,
) -> (Option<Box<UndeliveredWorkerFailure>>, Option<EngineError>) {
    match publication.lock() {
        Ok(mut slot) => (slot.take(), None),
        Err(poisoned) => (
            poisoned.into_inner().take(),
            Some(
                failure(
                    contract::ErrorCode::BackendFailure,
                    "worker failed publication owner poisoned",
                )
                .into(),
            ),
        ),
    }
}

fn copy_failed_publication(
    publication: &Mutex<Option<Box<UndeliveredWorkerFailure>>>,
) -> (Option<Box<UndeliveredWorkerFailure>>, Option<EngineError>) {
    match publication.lock() {
        Ok(slot) => (slot.clone(), None),
        Err(poisoned) => (
            poisoned.into_inner().clone(),
            Some(
                failure(
                    contract::ErrorCode::BackendFailure,
                    "worker failed publication owner poisoned",
                )
                .into(),
            ),
        ),
    }
}

/// Failed send returns one bounded typed copy while its original stays visible
/// to the owner. Reaping acknowledges the original once and retains all cleanup
/// errors from the copy's join result without reporting the source twice.
fn discard_duplicate_delivery_failure(
    error: EngineError,
    receipt: &UndeliveredWorkerFailure,
) -> Option<EngineError> {
    match error {
        EngineError::UndeliveredWorkerFailure(copy)
            if copy.ticket == receipt.ticket && copy.completion == receipt.completion =>
        {
            None
        }
        EngineError::Cleanup { primary, cleanup } => {
            let primary = discard_duplicate_delivery_failure(*primary, receipt);
            let cleanup = cleanup
                .into_iter()
                .filter_map(|error| discard_duplicate_delivery_failure(error, receipt))
                .collect();
            finish_errors(primary, cleanup).err()
        }
        error => Some(error),
    }
}

struct RuntimePort<'a>(&'a mut dyn ManagedEvaluator);
impl contract::Evaluator<RulesState> for RuntimePort<'_> {
    fn submit(
        &mut self,
        request: Arc<contract::EvalRequest<RulesState>>,
    ) -> Result<(), contract::ContractError> {
        self.0.submit(request)
    }
    fn poll(&mut self) -> Option<contract::EvalResult> {
        self.0.poll()
    }
    fn cancel(&mut self, request: contract::RequestId) -> Result<(), contract::ContractError> {
        self.0.cancel(request)
    }
}
fn shutdown(
    runtime: &mut dyn ManagedEvaluator,
    limit: Duration,
) -> Result<(), contract::ContractError> {
    runtime.shutdown(Instant::now().checked_add(limit).ok_or_else(|| {
        failure(
            contract::ErrorCode::ResourceExhausted,
            "shutdown deadline overflow",
        )
    })?)
}
fn failed_completion(error: contract::ContractError) -> SearchCompletion {
    SearchCompletion::Failed {
        code: format!("{:?}/{:?}", error.stage, error.code),
        message: error.detail.into(),
    }
}
fn append_contract_result<S>(
    out: &mut SessionResult<S>,
    next: crate::contracts::ContractSessionOutcome<S>,
) {
    for error in next.errors {
        out.diagnostics.push(Diagnostic {
            code: "SharedContractError",
            message: error.to_string(),
        });
    }
    append_result(out, next.session);
}
fn append_result<S>(out: &mut SessionResult<S>, next: SessionResult<S>) {
    out.protocol.extend(next.protocol);
    out.diagnostics.extend(next.diagnostics);
    out.effects.extend(next.effects);
    out.accepted &= next.accepted;
}

// Preserve the exact rejected receipt on bounded retention failure without a
// new allocation on that failure path.
#[allow(clippy::result_large_err)]
fn consume_pump_diagnostic(
    event: &rz_search::contracts::ContractPumpEvent,
    receipts: &Mutex<Vec<(WorkerDiagnosticReceipt, u64)>>,
    sender: &SyncSender<Event>,
) -> Result<(), DiagnosticRetentionFailure> {
    use rz_search::contracts::ContractPumpEvent;
    let receipt = match event {
        ContractPumpEvent::Diagnostic(error) => WorkerDiagnosticReceipt::Boundary(*error),
        ContractPumpEvent::RejectedResult { error, result } => match result.as_ref() {
            contract::EvalResult::Failed(failed) => WorkerDiagnosticReceipt::RejectedFailure {
                rejection: *error,
                failure: Box::new(failed.clone()),
            },
            result => {
                let (kind, context) = match result {
                    contract::EvalResult::Completed(output) => (
                        RejectedCompletionKind::Completed,
                        contract::CompletionContext {
                            request: output.context,
                            execution: output.actual.execution,
                        },
                    ),
                    contract::EvalResult::Canceled(context) => {
                        (RejectedCompletionKind::Canceled, *context)
                    }
                    contract::EvalResult::Expired(context) => {
                        (RejectedCompletionKind::Expired, *context)
                    }
                    contract::EvalResult::Stale(context) => {
                        (RejectedCompletionKind::Stale, *context)
                    }
                    contract::EvalResult::Failed(_) => unreachable!("handled failed result above"),
                };
                WorkerDiagnosticReceipt::RejectedCompletion {
                    rejection: *error,
                    kind,
                    context: Box::new(context),
                }
            }
        },
        _ => return Ok(()),
    };
    let retained = retain_diagnostic(receipts, receipt);
    // The typed receipt stays owner-held even if the bounded queue is full.
    let _ = sender.try_send(Event::RejectedInput {
        code: "OwnerWake",
        message: String::new(),
    });
    retained
}

// The 32-class owner returns an overflow/poisoned receipt intact; allocating a
// replacement error must not become a prerequisite for preserving that receipt.
#[allow(clippy::result_large_err)]
fn retain_diagnostic(
    receipts: &Mutex<Vec<(WorkerDiagnosticReceipt, u64)>>,
    receipt: WorkerDiagnosticReceipt,
) -> Result<(), DiagnosticRetentionFailure> {
    let mut receipts = match receipts.lock() {
        Ok(receipts) => receipts,
        Err(_) => {
            return Err(DiagnosticRetentionFailure {
                error: failure(
                    contract::ErrorCode::BackendFailure,
                    "worker diagnostic receipt owner poisoned",
                ),
                receipt,
            });
        }
    };
    if let Some((_, count)) = receipts
        .iter_mut()
        .find(|(existing, _)| *existing == receipt)
    {
        let Some(next) = count.checked_add(1) else {
            return Err(DiagnosticRetentionFailure {
                error: failure(
                    contract::ErrorCode::ResourceExhausted,
                    "worker diagnostic occurrence count exhausted",
                ),
                receipt,
            });
        };
        *count = next;
    } else if receipts.len() < 32 {
        receipts.push((receipt, 1));
    } else {
        // Overflow remains an explicit completion failure containing the original
        // receipt; it neither overwrites nor silently drops the 32 owned classes.
        return Err(DiagnosticRetentionFailure {
            error: failure(
                contract::ErrorCode::ResourceExhausted,
                "worker diagnostic receipt class limit exhausted",
            ),
            receipt,
        });
    }
    Ok(())
}
/// Independent input, timer, search and physical evaluator owners have bounded
/// event/worker counts. CPU/mock success does not assert NN/GPU support.
pub fn serve<O: Write, D: Write>(
    events: Receiver<Event>,
    sender: SyncSender<Event>,
    protocol: &mut O,
    diagnostics: &mut D,
    process: EngineProcess,
    settings: EngineSettings,
) -> Result<(), EngineError> {
    let EngineProcess {
        factory,
        driver,
        owners,
        clock,
        identity,
    } = process;
    if settings.max_workers == 0
        || settings.shutdown_limit.is_zero()
        || settings.search.max_simulations == 0
        || (factory.is_some() && settings.search.max_simulations > 4096)
        || factory.is_some() == driver.is_some()
    {
        return Err(failure(
            contract::ErrorCode::InvalidInput,
            "finite workers/drain/work and exactly one search startup required",
        )
        .into());
    }
    let scope = if let Some(factory) = &factory {
        let profile = factory.profile();
        crate::contracts::SessionScope::Evaluation(contract::AcceptanceScope {
            game: contract::GameGeneration(1),
            root: contract::RootGeneration(0),
            model: profile.model.handle(),
            encoding: profile.model.encoding().handle,
            backend: profile.backend,
        })
    } else {
        crate::contracts::SessionScope::Search(contract::pals::SearchAuthority {
            epoch: clock.epoch(),
            game: contract::GameGeneration(1),
            root: contract::RootGeneration(0),
            implementation: driver
                .as_ref()
                .expect("validated driver startup")
                .implementation(),
        })
    };
    let session_owner = crate::contracts::ContractSessionOwner::new_scope_with_cancel_history(
        scope,
        clock.epoch(),
        clock.b_clock(),
        settings.search,
        settings.max_workers,
    )?;
    let goal_capability = driver
        .as_ref()
        .and_then(|driver| driver.search_goal_capability());
    let node_budget_semantics = driver.as_ref().map_or(
        crate::bridge::NodeBudgetSemantics::LegacySimulations,
        |driver| driver.node_budget_semantics(),
    );
    let owner = Arc::new(Mutex::new(Owner {
        session_owner,
        active: None,
        scope: Arc::new(Mutex::new(scope)),
        evaluation_scope: scope.evaluation().map(|scope| Arc::new(Mutex::new(scope))),
        factory,
        driver,
        ids: Arc::clone(&clock.ids),
        clock,
        settings,
        workers: Vec::new(),
        pending_failures: VecDeque::new(),
        pending_bestmove: None,
        ponder: None,
        held_ponder_bestmove: None,
        admitted_ponder_report: None,
        pending_original_go: None,
        pending_close: None,
        fence_failure: None,
        diagnostics: Arc::new(Mutex::new(Vec::new())),
    }));
    let mut session = Session::new(
        RulesUciPort::new(owners, settings.position),
        identity,
        vec![OptionSpec {
            name: "Ponder".into(),
            kind: OptionKind::Check { default: false },
        }],
        settings.parser,
    )
    .map_err(EngineError::Session)?;
    session
        .set_search_goal_capability(goal_capability)
        .map_err(EngineError::Session)?;
    session
        .set_node_budget_semantics(node_budget_semantics)
        .map_err(EngineError::Session)?;
    let timer_owner = Arc::clone(&owner);
    let timer_events = sender.clone();
    let timer_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let timer_flag = Arc::clone(&timer_stop);
    let timer = thread::spawn(move || {
        let mut delivered = None;
        while !timer_flag.load(Ordering::Acquire) {
            if let Ok(owner) = timer_owner.lock() {
                if owner.pending_bestmove.as_ref().is_some_and(|pending| {
                    Instant::now() >= pending.until || owner.workers.iter().any(Worker::is_finished)
                }) {
                    let _ = timer_events.try_send(Event::RejectedInput {
                        code: "OwnerWake",
                        message: String::new(),
                    });
                }
                if let Some(ticket) = owner.deadline_ticket(delivered.as_ref(), Instant::now())
                    && timer_events
                        .try_send(Event::Deadline(ticket.clone()))
                        .is_ok()
                {
                    delivered = Some(ticket);
                }
            }
            thread::sleep(Duration::from_millis(1));
        }
    });
    let handler_owner = Arc::clone(&owner);
    let dispatch_owner = Arc::clone(&owner);
    let result = serve_events_with_handler(
        &mut session,
        &events,
        protocol,
        diagnostics,
        move |session, event| {
            handler_owner
                .lock()
                .expect("UCI owner")
                .handle(session, event)
        },
        move |effect| {
            dispatch_owner
                .lock()
                .map_err(|_| failure(contract::ErrorCode::BackendFailure, "UCI owner poisoned"))?
                .dispatch(effect, &sender)
        },
    );
    timer_stop.store(true, Ordering::Release);
    drop(events);
    let primary = result
        .err()
        .map(|error| EngineError::Transport(Box::new(error)));
    let mut cleanup = Vec::new();
    if timer.join().is_err() {
        cleanup.push(EngineError::WorkerPanic);
    }
    let cleanup_deadline = Instant::now().checked_add(settings.shutdown_limit);
    if let Some(until) = cleanup_deadline {
        loop {
            let mut owner = match owner.lock() {
                Ok(owner) => owner,
                Err(_) => {
                    cleanup.push(
                        failure(contract::ErrorCode::BackendFailure, "UCI owner poisoned").into(),
                    );
                    break;
                }
            };
            owner.cancel();
            if let Err(error) = owner.reap_workers(true) {
                cleanup.push(error);
            }
            if owner.workers.is_empty() {
                break;
            }
            if Instant::now() >= until {
                cleanup.push(EngineError::DrainTimeout);
                break;
            }
            drop(owner);
            thread::sleep(Duration::from_millis(1));
        }
    } else {
        cleanup.push(
            failure(
                contract::ErrorCode::ResourceExhausted,
                "shutdown deadline overflow",
            )
            .into(),
        );
    }
    let mut joined_driver = None;
    match owner.lock() {
        Ok(mut owner) => {
            if owner.workers.is_empty() {
                joined_driver = owner.driver.clone();
            }
            if let Some(error) = owner.fence_failure.take() {
                cleanup.push(error);
            }
            // A timed-out/panicking drain does not discard failures that were
            // already published by workers which have not yet finished joining.
            cleanup.extend(owner.collect_unconsumed_failures());
            match owner.diagnostics.lock() {
                Ok(mut receipts) if !receipts.is_empty() => cleanup.push(
                    EngineError::UndeliveredDiagnostics(std::mem::take(&mut *receipts)),
                ),
                Err(_) => cleanup.push(
                    failure(
                        contract::ErrorCode::BackendFailure,
                        "worker diagnostic receipt owner poisoned",
                    )
                    .into(),
                ),
                _ => {}
            }
        }
        Err(_) => {
            cleanup.push(failure(contract::ErrorCode::BackendFailure, "UCI owner poisoned").into())
        }
    }
    // The search thread is joined before its persistent model/search owners
    // close. Keep the same cleanup deadline; expiry grants no fresh window.
    // No hook means no additional lifecycle observation for legacy drivers.
    if let (Some(driver), Some(until)) = (joined_driver, cleanup_deadline) {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            driver.finish_session(until)
        })) {
            Ok(Ok(Some(crate::search_driver::DriverPhysicalCompletion::Unknown))) => {
                cleanup.push(EngineError::SearchDriverClose(
                    crate::search_driver::SearchSessionFailure::physical_completion_unknown(),
                ));
            }
            Ok(Err(error)) => cleanup.push(EngineError::SearchDriverClose(error)),
            Err(_) => cleanup.push(EngineError::WorkerPanic),
            Ok(Ok(_)) => {}
        }
    }
    finish_errors(primary, cleanup)
}
fn finish_errors(
    primary: Option<EngineError>,
    mut cleanup: Vec<EngineError>,
) -> Result<(), EngineError> {
    let primary = match primary {
        Some(primary) => primary,
        None if cleanup.is_empty() => return Ok(()),
        None => cleanup.remove(0),
    };
    if cleanup.is_empty() {
        Err(primary)
    } else {
        Err(EngineError::Cleanup {
            primary: Box::new(primary),
            cleanup,
        })
    }
}
pub fn event_channel() -> (SyncSender<Event>, Receiver<Event>) {
    mpsc::sync_channel(128)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_clock_rejects_actual_instants_before_origin_even_for_tiny_deltas() {
        let clock = ProcessClock::new(contract::ProcessEpoch(8123));
        for delta in [Duration::from_nanos(1), Duration::from_millis(1)] {
            let before = clock.origin.checked_sub(delta).unwrap();
            assert!(before < clock.origin);
            assert_eq!(
                clock.tick_at(before).unwrap_err().code,
                contract::ErrorCode::InvalidInput
            );
            assert_eq!(
                clock.deadline(before).unwrap_err().code,
                contract::ErrorCode::InvalidInput
            );
        }
        assert_eq!(
            clock.tick_at(clock.origin).unwrap(),
            contract::MonotonicTick(0)
        );
    }

    struct CloseObservedDriver {
        state: crate::search_driver::DriverPhysicalCompletion,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl crate::search_driver::SearchSessionDriver for CloseObservedDriver {
        fn kind(&self) -> crate::search_driver::SearchKind {
            crate::search_driver::SearchKind::Cpu
        }
        fn implementation(&self) -> contract::Digest {
            contract::Digest([77; 32])
        }
        fn identity(&self) -> EngineIdentity {
            EngineIdentity {
                name: "persistent owner close fixture".into(),
                author: "test".into(),
            }
        }
        fn finish_session(
            &self,
            _: Instant,
        ) -> Result<
            Option<crate::search_driver::DriverPhysicalCompletion>,
            crate::search_driver::SearchSessionFailure,
        > {
            self.calls.fetch_add(1, Ordering::AcqRel);
            Ok(Some(self.state))
        }
        fn run(
            &self,
            _: &Position,
            _: &crate::search_driver::SearchSessionContext,
            _: &mut dyn FnMut(BoardMove),
        ) -> Result<
            crate::search_driver::SearchSessionReport,
            crate::search_driver::SearchSessionFailure,
        > {
            unreachable!("quit without go must not invoke a search")
        }
    }

    #[test]
    fn persistent_driver_close_runs_once_and_unknown_prevents_success() {
        use crate::search_driver::DriverPhysicalCompletion::{Confirmed, Unknown};
        for state in [Confirmed, Unknown] {
            let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let process = EngineProcess::with_search_driver(
                Arc::new(CloseObservedDriver {
                    state,
                    calls: Arc::clone(&calls),
                }),
                Arc::new(OwnerRegistry::default()),
                ProcessClock::new(contract::ProcessEpoch(78)),
            );
            let (sender, events) = event_channel();
            sender.send(Event::Line("quit".into())).unwrap();
            let mut protocol = Vec::new();
            let mut diagnostics = Vec::new();
            let result = serve(
                events,
                sender,
                &mut protocol,
                &mut diagnostics,
                process,
                EngineSettings::default(),
            );
            assert_eq!(calls.load(Ordering::Acquire), 1);
            if state == Confirmed {
                result.unwrap();
            } else {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("PhysicalCompletionUnknown")
                );
            }
            assert!(!String::from_utf8(protocol).unwrap().contains("bestmove "));
        }
    }

    struct UnknownPhysicalDriver;
    impl crate::search_driver::SearchSessionDriver for UnknownPhysicalDriver {
        fn kind(&self) -> crate::search_driver::SearchKind {
            crate::search_driver::SearchKind::Pals
        }
        fn implementation(&self) -> contract::Digest {
            contract::Digest([9; 32])
        }
        fn identity(&self) -> EngineIdentity {
            EngineIdentity {
                name: "unknown physical completion fixture".into(),
                author: "test".into(),
            }
        }
        fn run(
            &self,
            position: &Position,
            _: &crate::search_driver::SearchSessionContext,
            progress: &mut dyn FnMut(BoardMove),
        ) -> Result<
            crate::search_driver::SearchSessionReport,
            crate::search_driver::SearchSessionFailure,
        > {
            if let Some(movement) = position.legal_moves().first() {
                progress(*movement);
            }
            Err(crate::search_driver::SearchSessionFailure::physical_completion_unknown())
        }
    }

    #[test]
    fn unknown_native_driver_completion_closes_output_without_a_fallback_acknowledgment() {
        let process = EngineProcess::with_search_driver(
            Arc::new(UnknownPhysicalDriver),
            Arc::new(OwnerRegistry::default()),
            ProcessClock::new(contract::ProcessEpoch(19)),
        );
        let (sender, events) = event_channel();
        sender.send(Event::Line("uci".into())).unwrap();
        sender.send(Event::Line("go nodes 1".into())).unwrap();
        sender.send(Event::Line("stop".into())).unwrap();
        let handle = thread::spawn(move || {
            let mut protocol = Vec::new();
            let mut diagnostics = Vec::new();
            let result = serve(
                events,
                sender,
                &mut protocol,
                &mut diagnostics,
                process,
                EngineSettings::default(),
            );
            (result, protocol, diagnostics)
        });
        let until = Instant::now() + Duration::from_secs(5);
        while !handle.is_finished() {
            assert!(
                Instant::now() < until,
                "fatal physical state must close the UCI owner"
            );
            thread::sleep(Duration::from_millis(1));
        }
        let (result, protocol, diagnostics) = handle.join().unwrap();
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("PhysicalCompletionUnknown")
        );
        assert!(!String::from_utf8(protocol).unwrap().contains("bestmove "));
        assert!(
            String::from_utf8(diagnostics)
                .unwrap()
                .contains("PhysicalFenceFailure")
        );
    }

    struct OutputAckFixture(Arc<Mutex<Vec<bool>>>);
    impl crate::search_driver::SearchSessionDriver for OutputAckFixture {
        fn kind(&self) -> crate::search_driver::SearchKind {
            crate::search_driver::SearchKind::Cpu
        }
        fn implementation(&self) -> contract::Digest {
            contract::Digest([11; 32])
        }
        fn identity(&self) -> EngineIdentity {
            EngineIdentity {
                name: "output acknowledgment fixture".into(),
                author: "test".into(),
            }
        }
        fn run(
            &self,
            _: &Position,
            _: &crate::search_driver::SearchSessionContext,
            _: &mut dyn FnMut(BoardMove),
        ) -> Result<
            crate::search_driver::SearchSessionReport,
            crate::search_driver::SearchSessionFailure,
        > {
            panic!("fixture handles events directly")
        }
        fn accept_uci_output(
            &self,
            _: contract::pals::SearchAuthority,
            bestmove: &str,
            from_report: bool,
        ) -> Result<(), contract::ContractError> {
            BoardMove::from_uci(bestmove).expect("only the move token reaches the driver");
            self.0.lock().unwrap().push(from_report);
            Ok(())
        }
    }
    #[test]
    fn pipelined_close_preserves_stopped_move_and_late_work_receipt_until_join() {
        for close in [Event::Line("quit".into()), Event::EndOfInput] {
            let (mut owner, mut session) = fixture();
            let expected = format!(
                "bestmove {}",
                session.snapshot().rules_position().legal_moves()[0]
            );
            let start = owner.handle(&mut session, Event::Line("go nodes 1".into()));
            assert!(
                start
                    .effects
                    .iter()
                    .any(|e| matches!(e, Effect::Start { .. }))
            );
            let (release, waiting) = mpsc::channel();
            let receipts = Arc::clone(&owner.diagnostics);
            owner.workers.push(Worker {
                handle: thread::spawn(move || {
                    waiting
                        .recv_timeout(Duration::from_secs(2))
                        .expect("bounded fixture release after pipelined close");
                    retain_diagnostic(
                        &receipts,
                        WorkerDiagnosticReceipt::DriverWork {
                            kind: crate::search_driver::SearchKind::Pals,
                            work: crate::search_driver::DriverWork::Pals {
                                rounds: 0,
                                cpu_nodes: 0,
                                completed_tasks: 0,
                                consumed_role_outputs: 0,
                                retained_situations: 0,
                            },
                        },
                    )
                    .map_err(|error| {
                        EngineError::UndeliveredDiagnostics(vec![(error.receipt, 1)])
                    })?;
                    Ok(())
                }),
                failure: Arc::new(Mutex::new(None)),
            });
            let stopped = owner.handle(&mut session, Event::Line("stop".into()));
            assert!(stopped.protocol.is_empty());
            assert!(owner.pending_bestmove.is_some());
            assert_eq!(
                owner
                    .handle(&mut session, Event::Line("isready".into()))
                    .protocol,
                ["readyok"]
            );
            let scope = owner.session_owner.lifecycle_scope();
            let closing = owner.handle(&mut session, close);
            assert!(closing.protocol.is_empty());
            assert!(!session.is_closed());
            assert!(owner.pending_close.is_some());
            for command in ["go nodes 1", "position startpos moves e2e4", "quit"] {
                let ignored = owner.handle(&mut session, Event::Line(command.into()));
                assert!(ignored.protocol.is_empty());
                assert!(ignored.effects.is_empty());
                assert_eq!(owner.session_owner.lifecycle_scope(), scope);
            }
            release.send(()).unwrap();
            let until = Instant::now() + Duration::from_secs(2);
            while !owner.workers[0].is_finished() {
                assert!(Instant::now() < until, "bounded fixture worker join");
                thread::sleep(Duration::from_millis(1));
            }
            let ended = owner.handle(
                &mut session,
                Event::RejectedInput {
                    code: "OwnerWake",
                    message: String::new(),
                },
            );
            assert_eq!(ended.protocol, [expected]);
            assert!(session.is_closed());
            assert!(ended.effects.iter().any(|e| matches!(e, Effect::Shutdown)));
            assert!(ended.diagnostics.iter().any(|d| {
                d.code == "SearchDiagnostic" && d.message.contains("search_kind=Pals")
            }));
            assert!(owner.workers.is_empty());
            assert!(owner.diagnostics.lock().unwrap().is_empty());
            assert!(owner.pending_bestmove.is_none());
            assert!(owner.pending_close.is_none());
            assert!(owner.fence_failure.is_none());
        }
    }

    #[test]
    fn late_completed_event_cannot_acknowledge_the_guard_rejected_report() {
        let (mut owner, mut session) = fixture();
        let acknowledgments = Arc::new(Mutex::new(Vec::new()));
        let driver = Arc::new(OutputAckFixture(Arc::clone(&acknowledgments)));
        let scope = crate::contracts::SessionScope::Search(contract::pals::SearchAuthority {
            epoch: owner.clock.epoch(),
            game: contract::GameGeneration(1),
            root: contract::RootGeneration(0),
            implementation: contract::Digest([11; 32]),
        });
        owner.session_owner = crate::contracts::ContractSessionOwner::new_scope(
            scope,
            owner.clock.epoch(),
            owner.clock.b_clock(),
            owner.settings.search,
        )
        .unwrap();
        owner.scope = Arc::new(Mutex::new(scope));
        owner.evaluation_scope = None;
        owner.factory = None;
        owner.driver = Some(driver);
        owner.handle(
            &mut session,
            Event::Line("setoption name Ponder value true".into()),
        );
        let started = owner.handle(&mut session, Event::Line("go movetime 20".into()));
        let ticket = started
            .effects
            .iter()
            .find_map(|effect| match effect {
                Effect::Start { ticket, .. } => Some(ticket.clone()),
                _ => None,
            })
            .unwrap();
        let bestmove = session.snapshot().rules_position().legal_moves()[0].to_string();
        let deadline = owner.session_owner.control().unwrap().deadline;
        thread::sleep(
            deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
        );
        let handle = thread::spawn(|| Ok(()));
        while !handle.is_finished() {
            thread::yield_now();
        }
        owner.workers.push(Worker {
            handle,
            failure: Arc::new(Mutex::new(None)),
        });
        let output = owner.handle(
            &mut session,
            Event::Complete {
                ticket,
                completion: SearchCompletion::Completed {
                    bestmove: Some(bestmove),
                },
            },
        );
        assert!(
            output
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "SharedContractError")
        );
        assert!(
            output
                .protocol
                .iter()
                .any(|line| line.starts_with("bestmove "))
        );
        assert_eq!(*acknowledgments.lock().unwrap(), vec![false]);
    }

    #[derive(Clone, Copy)]
    enum FailureMode {
        Factory,
        Matching,
        ForeignFlood,
        NoEvaluation,
        PanicDrain,
    }
    struct FailureFactory {
        profile: EvaluatorProfile,
        mode: FailureMode,
        drained: Arc<AtomicU64>,
        cleanup: Option<contract::ContractError>,
    }
    struct FailureRuntime {
        mode: FailureMode,
        context: Option<contract::EvalContext>,
        callbacks: u64,
        drained: Arc<AtomicU64>,
        cleanup: Option<contract::ContractError>,
        drain_gate: Option<(SyncSender<()>, Receiver<()>)>,
    }
    fn injected_backend_failure() -> contract::ContractError {
        contract::ContractError::new(
            contract::ErrorCode::BackendFailure,
            contract::Stage::Backend,
            "worker physical failure before owner consumption",
        )
    }
    impl EvaluatorFactory for FailureFactory {
        fn profile(&self) -> EvaluatorProfile {
            self.profile.clone()
        }
        fn create(
            &self,
            _: ProcessClock,
            authority: SearchAuthority,
        ) -> Result<Box<dyn ManagedEvaluator>, contract::ContractError> {
            if matches!(self.mode, FailureMode::Factory) {
                return Err(injected_backend_failure());
            }
            if matches!(self.mode, FailureMode::NoEvaluation) {
                authority.cancel();
            }
            Ok(Box::new(FailureRuntime {
                mode: self.mode,
                context: None,
                callbacks: 0,
                drained: Arc::clone(&self.drained),
                cleanup: self.cleanup,
                drain_gate: None,
            }))
        }
        fn input_key(
            &self,
            _: &RulesState,
            _: &[contract::Move],
        ) -> Result<contract::EvalInputKey, contract::ContractError> {
            Ok(contract::EvalInputKey(contract::Digest([9; 32])))
        }
    }
    impl contract::Evaluator<RulesState> for FailureRuntime {
        fn submit(
            &mut self,
            request: Arc<contract::EvalRequest<RulesState>>,
        ) -> Result<(), contract::ContractError> {
            self.context = Some(request.context());
            Ok(())
        }
        fn poll(&mut self) -> Option<contract::EvalResult> {
            let mut context = self.context?;
            self.callbacks += 1;
            if matches!(self.mode, FailureMode::ForeignFlood) {
                context.request.sequence += 100 + self.callbacks;
            } else {
                self.context = None;
            }
            Some(contract::EvalResult::Failed(contract::EvalFailure {
                context: contract::CompletionContext {
                    request: context,
                    execution: Some(contract::ExecutionId::new(
                        context.request.epoch,
                        self.callbacks,
                    )),
                },
                error: injected_backend_failure(),
                recovery: contract::RecoveryOutcome::Failed,
            }))
        }
        fn cancel(&mut self, _: contract::RequestId) -> Result<(), contract::ContractError> {
            Ok(())
        }
    }
    impl ManagedEvaluator for FailureRuntime {
        fn shutdown(&mut self, _: Instant) -> Result<(), contract::ContractError> {
            if let Some((entered, release)) = self.drain_gate.take() {
                entered.send(()).unwrap();
                release.recv_timeout(Duration::from_secs(2)).unwrap();
            }
            self.drained.fetch_add(1, Ordering::AcqRel);
            if matches!(self.mode, FailureMode::PanicDrain) {
                panic!("injected drain panic after publication");
            }
            self.cleanup.map_or(Ok(()), Err)
        }
    }

    struct UnusedFactory(EvaluatorProfile);
    impl EvaluatorFactory for UnusedFactory {
        fn profile(&self) -> EvaluatorProfile {
            self.0.clone()
        }
        fn create(
            &self,
            _: ProcessClock,
            _: SearchAuthority,
        ) -> Result<Box<dyn ManagedEvaluator>, contract::ContractError> {
            panic!("these owner refusal tests must not create a runtime")
        }
        fn input_key(
            &self,
            _: &RulesState,
            _: &[contract::Move],
        ) -> Result<contract::EvalInputKey, contract::ContractError> {
            panic!("these owner refusal tests must not encode")
        }
    }
    fn fixture() -> (Owner, Session<RulesUciPort>) {
        let owners = Arc::new(OwnerRegistry::default());
        let encoding = contract::EncodingHandle {
            owner: owners.allocate().unwrap(),
            slot: 0,
            generation: contract::SlotGeneration(1),
            manifest: contract::Digest([1; 32]),
        };
        let model = contract::ModelHandle {
            owner: owners.allocate().unwrap(),
            slot: 0,
            generation: contract::SlotGeneration(1),
            manifest: contract::Digest([2; 32]),
        };
        let descriptor = contract::ModelDescriptor::try_new(
            model,
            contract::EncodingDescriptor {
                handle: encoding,
                history_length: 1,
                action_map: contract::Digest([3; 32]),
                history_policy: contract::Digest([4; 32]),
            },
            vec![contract::PrecisionProfile::Fp32],
            1,
            1,
        )
        .unwrap();
        let profile = EvaluatorProfile {
            model: Arc::new(descriptor),
            precision: contract::PrecisionProfile::Fp32,
            backend: contract::Digest([5; 32]),
            compute: contract::ComputeBudget {
                min_steps: 1,
                max_steps: 1,
                require_full: true,
            },
            bytes: contract::ByteBudget {
                host: 1,
                device: 0,
                pinned: 0,
            },
        };
        let clock = ProcessClock::new(contract::ProcessEpoch(7));
        let scope = contract::AcceptanceScope {
            game: contract::GameGeneration(1),
            root: contract::RootGeneration(0),
            model,
            encoding,
            backend: profile.backend,
        };
        let settings = EngineSettings {
            max_workers: 1,
            ..EngineSettings::default()
        };
        let owner = Owner {
            session_owner: crate::contracts::ContractSessionOwner::new(
                scope,
                clock.epoch(),
                clock.b_clock(),
                settings.search,
            )
            .unwrap(),
            active: None,
            scope: Arc::new(Mutex::new(crate::contracts::SessionScope::Evaluation(
                scope,
            ))),
            evaluation_scope: Some(Arc::new(Mutex::new(scope))),
            factory: Some(Arc::new(UnusedFactory(profile))),
            driver: None,
            clock,
            ids: Arc::new(rz_search::contracts::IdAllocator::new(
                contract::ProcessEpoch(7),
            )),
            settings,
            workers: Vec::new(),
            pending_failures: VecDeque::new(),
            pending_bestmove: None,
            ponder: None,
            held_ponder_bestmove: None,
            admitted_ponder_report: None,
            pending_original_go: None,
            pending_close: None,
            fence_failure: None,
            diagnostics: Arc::new(Mutex::new(Vec::new())),
        };
        let session = Session::new(
            RulesUciPort::new(owners, settings.position),
            EngineIdentity {
                name: "actual Rules refusal fixture".into(),
                author: "test".into(),
            },
            vec![OptionSpec {
                name: "Ponder".into(),
                kind: OptionKind::Check { default: false },
            }],
            settings.parser,
        )
        .unwrap();
        (owner, session)
    }

    fn fixture_with_cancel_history(capacity: usize) -> (Owner, Session<RulesUciPort>) {
        let (mut owner, session) = fixture();
        owner.session_owner =
            crate::contracts::ContractSessionOwner::new_scope_with_cancel_history(
                owner.session_owner.lifecycle_scope(),
                owner.clock.epoch(),
                owner.clock.b_clock(),
                owner.settings.search,
                capacity,
            )
            .unwrap();
        (owner, session)
    }

    #[test]
    fn driver_preadmission_rejection_preserves_the_active_ticket_and_control() {
        struct RejectGo;
        impl crate::search_driver::SearchSessionDriver for RejectGo {
            fn kind(&self) -> crate::search_driver::SearchKind {
                crate::search_driver::SearchKind::Rovent
            }
            fn implementation(&self) -> contract::Digest {
                contract::Digest([31; 32])
            }
            fn identity(&self) -> EngineIdentity {
                EngineIdentity {
                    name: "pure rejection fixture".into(),
                    author: "test".into(),
                }
            }
            fn validate_go(
                &self,
                _: &Position,
                _: &crate::parser::GoLimits,
            ) -> Result<(), contract::ContractError> {
                Err(contract::ContractError::new(
                    contract::ErrorCode::InvalidInput,
                    contract::Stage::Admission,
                    "exact workload differs",
                ))
            }
            fn run(
                &self,
                _: &Position,
                _: &crate::search_driver::SearchSessionContext,
                _: &mut dyn FnMut(BoardMove),
            ) -> Result<
                crate::search_driver::SearchSessionReport,
                crate::search_driver::SearchSessionFailure,
            > {
                panic!("pure rejected go cannot dispatch a worker")
            }
        }
        let (mut owner, mut session) = fixture();
        assert!(
            owner
                .handle(&mut session, Event::Line("go infinite".into()))
                .accepted
        );
        let ticket = session.active_ticket().unwrap();
        let scope = owner.session_owner.lifecycle_scope();
        let control = owner.session_owner.control().unwrap().clone();
        owner.driver = Some(Arc::new(RejectGo));
        let rejected = owner.handle(&mut session, Event::Line("go nodes 7".into()));
        assert!(!rejected.accepted);
        assert!(rejected.effects.is_empty() && rejected.protocol.is_empty());
        assert!(
            rejected
                .diagnostics
                .iter()
                .any(|d| d.code == "DriverGoRejected"
                    && d.message.contains("exact workload differs"))
        );
        assert_eq!(session.active_ticket(), Some(ticket));
        assert_eq!(owner.session_owner.lifecycle_scope(), scope);
        assert!(!control.cancellation.load(Ordering::Acquire));
    }

    fn assert_canceled_result_ignored(out: &SessionResult<RulesSearchPosition>) {
        assert!(!out.accepted);
        assert!(out.protocol.is_empty());
        assert!(out.effects.is_empty());
        assert_eq!(out.diagnostics.len(), 1);
        assert_eq!(out.diagnostics[0].code, "CanceledSearchResultIgnored");
        assert!(
            out.diagnostics[0]
                .message
                .contains("no result or work admission")
        );
    }

    #[test]
    fn service_history_distinguishes_owned_cancellation_from_foreign_and_failed_results() {
        let (mut owner, mut session) = fixture_with_cancel_history(1);
        owner.handle(&mut session, Event::Line("go infinite".into()));
        let ticket = session.active_ticket().unwrap();
        owner.handle(
            &mut session,
            Event::Progress {
                ticket: ticket.clone(),
                bestmove: "e2e4".into(),
            },
        );
        let stopped = owner.handle(&mut session, Event::Line("stop".into()));
        assert_eq!(stopped.protocol, ["bestmove e2e4"]);
        let scope = owner.session_owner.lifecycle_scope();
        for event in [
            Event::Progress {
                ticket: ticket.clone(),
                bestmove: "d2d4".into(),
            },
            Event::Complete {
                ticket: ticket.clone(),
                completion: SearchCompletion::Completed {
                    bestmove: Some("d2d4".into()),
                },
            },
        ] {
            assert_canceled_result_ignored(&owner.handle(&mut session, event));
            assert_eq!(owner.session_owner.lifecycle_scope(), scope);
            assert!(session.active_ticket().is_none());
            assert!(owner.diagnostics.lock().unwrap().is_empty());
        }
        let (mut foreign, mut other_session) = fixture();
        foreign.handle(&mut other_session, Event::Line("go infinite".into()));
        let foreign_result = owner.handle(
            &mut session,
            Event::Complete {
                ticket: other_session.active_ticket().unwrap(),
                completion: SearchCompletion::Completed {
                    bestmove: Some("e2e4".into()),
                },
            },
        );
        assert!(!foreign_result.accepted);
        assert!(foreign_result.protocol.is_empty());
        assert!(foreign_result.effects.is_empty());
        assert!(foreign_result.diagnostics.iter().any(|d| {
            d.code == "SharedContractError" && d.message.contains("inactive or foreign")
        }));
        assert!(
            !foreign_result
                .diagnostics
                .iter()
                .any(|d| d.code == "CanceledSearchResultIgnored")
        );
        let failed = owner.handle(
            &mut session,
            Event::Complete {
                ticket,
                completion: SearchCompletion::Failed {
                    code: "LateBackendFailure".into(),
                    message: "original late physical callback failure".into(),
                },
            },
        );
        assert!(!failed.accepted);
        assert!(failed.protocol.is_empty());
        assert!(failed.effects.is_empty());
        assert!(
            failed
                .diagnostics
                .iter()
                .any(|d| d.code == "SharedContractError")
        );
        assert!(failed.diagnostics.iter().any(|d| {
            d.code == "SearchFailed"
                && d.message == "LateBackendFailure: original late physical callback failure"
        }));
        assert!(
            !failed
                .diagnostics
                .iter()
                .any(|d| d.code == "CanceledSearchResultIgnored")
        );
    }

    #[test]
    fn canceled_ticket_history_eviction_and_natural_completion_remain_typed_stale() {
        let (mut owner, mut session) = fixture_with_cancel_history(1);
        let mut tickets = Vec::new();
        for _ in 0..2 {
            owner.handle(&mut session, Event::Line("go infinite".into()));
            tickets.push(session.active_ticket().unwrap());
            owner.handle(&mut session, Event::Line("stop".into()));
        }
        let evicted = owner.session_owner.handle_event(
            &mut session,
            Event::Progress {
                ticket: tickets[0].clone(),
                bestmove: "e2e4".into(),
            },
        );
        assert_eq!(evicted.errors.len(), 1);
        assert_eq!(evicted.errors[0].code, contract::ErrorCode::Stale);
        assert!(evicted.session.protocol.is_empty());
        let retained = owner.session_owner.handle_event(
            &mut session,
            Event::Progress {
                ticket: tickets[1].clone(),
                bestmove: "e2e4".into(),
            },
        );
        assert!(retained.errors.is_empty());
        assert_canceled_result_ignored(&retained.session);
        owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let natural = session.active_ticket().unwrap();
        let completed = owner.handle(
            &mut session,
            Event::Complete {
                ticket: natural.clone(),
                completion: SearchCompletion::Completed {
                    bestmove: Some("e2e4".into()),
                },
            },
        );
        assert_eq!(completed.protocol, ["bestmove e2e4"]);
        let duplicate = owner.session_owner.handle_event(
            &mut session,
            Event::Complete {
                ticket: natural,
                completion: SearchCompletion::Completed {
                    bestmove: Some("d2d4".into()),
                },
            },
        );
        assert_eq!(duplicate.errors.len(), 1);
        assert_eq!(duplicate.errors[0].code, contract::ErrorCode::Stale);
        assert!(duplicate.session.protocol.is_empty());
        assert!(duplicate.session.effects.is_empty());
    }

    #[test]
    fn only_actual_cancel_effects_record_reasons_and_closed_owners_never_reenter() {
        for (command, reason) in [
            ("stop", "Stop"),
            ("go infinite", "ReplacedSearch"),
            ("position startpos moves e2e4", "ReplacedPosition"),
            ("ucinewgame", "NewGame"),
            ("quit", "Quit"),
            ("EOF", "EndOfInput"),
        ] {
            let (mut owner, mut session) = fixture_with_cancel_history(1);
            owner.handle(&mut session, Event::Line("go infinite".into()));
            let ticket = session.active_ticket().unwrap();
            let scope = owner.session_owner.lifecycle_scope();
            let rejected = owner.handle(&mut session, Event::Line("stop extra".into()));
            assert!(!rejected.accepted);
            assert_eq!(session.active_ticket(), Some(ticket.clone()));
            assert_eq!(owner.session_owner.lifecycle_scope(), scope);
            let close = if command == "EOF" {
                Event::EndOfInput
            } else {
                Event::Line(command.into())
            };
            owner.handle(&mut session, close);
            let late = owner.session_owner.handle_event(
                &mut session,
                Event::Progress {
                    ticket,
                    bestmove: "d2d4".into(),
                },
            );
            assert!(late.errors.is_empty(), "{command}: {late:?}");
            assert_canceled_result_ignored(&late.session);
            assert!(late.session.diagnostics[0].message.contains(reason));
            if matches!(command, "quit" | "EOF") {
                let closed_scope = owner.session_owner.lifecycle_scope();
                let new_go = owner.handle(&mut session, Event::Line("go infinite".into()));
                assert!(!new_go.accepted);
                assert!(new_go.protocol.is_empty());
                assert!(new_go.effects.is_empty());
                assert!(session.active_ticket().is_none());
                assert_eq!(owner.session_owner.lifecycle_scope(), closed_scope);
            }
        }
    }

    #[test]
    fn actual_deadline_cancel_is_recorded_but_old_timer_production_is_closed() {
        let (mut owner, mut session) = fixture_with_cancel_history(1);
        owner.handle(&mut session, Event::Line("go movetime 1000".into()));
        let ticket = session.active_ticket().unwrap();
        let control = owner.session_owner.control().unwrap().clone();
        owner.active = Some(ActiveBinding {
            ticket: ticket.clone(),
            control: control.clone(),
            authority: None,
            driver_output: None,
            driver_output_until: None,
            driver_start: None,
            ponder_clock: None,
        });
        assert_eq!(
            owner.deadline_ticket(None, control.deadline),
            Some(ticket.clone())
        );
        assert!(
            owner
                .deadline_ticket(Some(&ticket), control.deadline)
                .is_none()
        );
        let expired = owner.session_owner.handle_event_at(
            &mut session,
            Event::Deadline(ticket.clone()),
            control.deadline + Duration::from_millis(1),
        );
        assert_eq!(expired.session.protocol.len(), 1);
        assert!(expired.session.effects.iter().any(|e| {
            matches!(e, Effect::Cancel { ticket: t, reason: crate::CancelReason::Deadline } if t == &ticket)
        }));
        assert!(owner.deadline_ticket(None, control.deadline).is_none());
        let late = owner.session_owner.handle_event(
            &mut session,
            Event::Complete {
                ticket,
                completion: SearchCompletion::Completed {
                    bestmove: Some("d2d4".into()),
                },
            },
        );
        assert!(late.errors.is_empty());
        assert_canceled_result_ignored(&late.session);
        assert!(late.session.diagnostics[0].message.contains("Deadline"));
        // A new active root still owns its own timer, independently of this
        // still-retained old physical binding.
        owner.handle(&mut session, Event::Line("go infinite".into()));
        assert!(owner.deadline_ticket(None, control.deadline).is_none());
        owner.handle(&mut session, Event::Line("stop".into()));
    }

    #[test]
    fn known_cancellation_never_masks_a_physical_unknown_publication() {
        let (mut owner, mut session) = fixture_with_cancel_history(1);
        owner.handle(&mut session, Event::Line("go infinite".into()));
        let ticket = session.active_ticket().unwrap();
        let (release, waiting) = mpsc::channel();
        let publication = Arc::new(Mutex::new(None));
        owner.workers.push(Worker {
            handle: thread::spawn(move || {
                waiting.recv_timeout(Duration::from_secs(2)).unwrap();
                Ok(())
            }),
            failure: Arc::clone(&publication),
        });
        let stopped = owner.handle(&mut session, Event::Line("stop".into()));
        assert!(stopped.protocol.is_empty());
        assert!(owner.pending_bestmove.is_some());
        let error = crate::search_driver::SearchSessionFailure::physical_completion_unknown();
        let completion = SearchCompletion::Failed {
            code: error.code.into(),
            message: error.detail.clone(),
        };
        *publication.lock().unwrap() = Some(Box::new(UndeliveredWorkerFailure {
            ticket: ticket.clone(),
            completion,
            source: WorkerFailureSource::Driver(error),
        }));
        let out = owner.handle(
            &mut session,
            Event::Complete {
                ticket,
                completion: SearchCompletion::Completed {
                    bestmove: Some("e2e4".into()),
                },
            },
        );
        assert!(out.protocol.is_empty());
        assert!(session.is_closed());
        assert!(owner.fence_failure.is_some());
        assert!(out.diagnostics.iter().any(|d| {
            d.code == "PhysicalFenceFailure" && d.message.contains("PhysicalCompletionUnknown")
        }));
        assert!(
            !out.diagnostics
                .iter()
                .any(|d| d.code == "CanceledSearchResultIgnored")
        );
        assert!(owner.pending_bestmove.is_none());
        release.send(()).unwrap();
        let until = Instant::now() + Duration::from_secs(2);
        while !owner.workers[0].is_finished() {
            assert!(Instant::now() < until);
            thread::yield_now();
        }
        owner.reap_workers(true).unwrap();
        assert!(owner.workers.is_empty());
    }

    #[test]
    fn canceled_failure_publication_preserves_original_cause_until_owner_consumption_and_join() {
        let (mut owner, mut session) = fixture_with_cancel_history(1);
        owner.handle(&mut session, Event::Line("go infinite".into()));
        let ticket = session.active_ticket().unwrap();
        owner.handle(
            &mut session,
            Event::Progress {
                ticket: ticket.clone(),
                bestmove: "e2e4".into(),
            },
        );
        let (sender, events) = mpsc::sync_channel(1);
        let (release, waiting) = mpsc::channel();
        let publication = Arc::new(Mutex::new(None));
        let worker_publication = Arc::clone(&publication);
        let shutdown_limit = owner.settings.shutdown_limit;
        owner.workers.push(Worker {
            handle: thread::spawn(move || {
                waiting.recv_timeout(Duration::from_secs(2)).unwrap();
                finish_worker(
                    &sender,
                    ticket,
                    WorkerCompletion::failed(WorkerFailureSource::Factory(
                        injected_backend_failure(),
                    )),
                    &worker_publication,
                    None,
                    shutdown_limit,
                )
            }),
            failure: Arc::clone(&publication),
        });
        let stopped = owner.handle(&mut session, Event::Line("stop".into()));
        assert!(stopped.protocol.is_empty());
        assert!(owner.pending_bestmove.is_some());
        release.send(()).unwrap();
        let event = events.recv_timeout(Duration::from_secs(2)).unwrap();
        let until = Instant::now() + Duration::from_secs(2);
        while !owner.workers[0].is_finished() {
            assert!(Instant::now() < until);
            thread::yield_now();
        }
        assert!(publication.lock().unwrap().is_some());
        owner.reap().unwrap();
        assert_eq!(
            owner.workers.len(),
            1,
            "unconsumed failed publication retains its slot"
        );
        let out = owner.handle(&mut session, event);
        assert_eq!(out.protocol, ["bestmove e2e4"]);
        assert!(out.diagnostics.iter().any(|d| {
            d.code == "SearchFailed" && d.message.contains(injected_backend_failure().detail)
        }));
        assert!(
            out.diagnostics
                .iter()
                .any(|d| d.code == "SharedContractError")
        );
        assert!(
            !out.diagnostics
                .iter()
                .any(|d| d.code == "CanceledSearchResultIgnored")
        );
        assert!(publication.lock().unwrap().is_none());
        assert!(owner.workers.is_empty());
        assert!(owner.fence_failure.is_none());
    }

    #[test]
    fn full_event_queue_cannot_lose_worker_refusal_or_checked_fallback() {
        let (mut owner, mut session) = fixture();
        let (release, waiting) = mpsc::channel();
        owner.workers.push(Worker {
            handle: thread::spawn(move || {
                waiting.recv().unwrap();
                Ok(())
            }),
            failure: Arc::new(Mutex::new(None)),
        });
        let out = owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let (sender, events) = mpsc::sync_channel(1);
        sender.send(Event::Line("isready".into())).unwrap();
        for effect in out.effects {
            owner.dispatch(effect, &sender).unwrap();
        }
        assert_eq!(owner.pending_failures.len(), 1);
        let out = owner.handle(&mut session, events.recv().unwrap());
        assert_eq!(
            out.protocol
                .iter()
                .filter(|line| line.starts_with("bestmove "))
                .count(),
            0
        );
        assert!(out.protocol.iter().any(|line| line == "readyok"));
        assert!(
            out.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("WorkerLimit"))
        );
        assert!(owner.pending_failures.is_empty());
        assert!(session.active_ticket().is_none());
        release.send(()).unwrap();
        let until = Instant::now() + Duration::from_secs(2);
        while !owner.workers[0].is_finished() {
            assert!(Instant::now() < until);
            thread::yield_now();
        }
        let out = owner.handle(
            &mut session,
            Event::RejectedInput {
                code: "OwnerWake",
                message: String::new(),
            },
        );
        assert_eq!(
            out.protocol
                .iter()
                .filter(|line| line.starts_with("bestmove "))
                .count(),
            1
        );
        assert!(owner.workers.is_empty());
    }

    #[test]
    fn stale_provider_failure_keeps_source_without_emitting_old_bestmove() {
        let (mut owner, mut session) = fixture();
        owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let old = session.active_ticket().unwrap();
        assert!(
            owner
                .handle(
                    &mut session,
                    Event::Line("position startpos moves e2e4".into())
                )
                .accepted
        );
        let out = owner.handle(
            &mut session,
            Event::Complete {
                ticket: old,
                completion: SearchCompletion::Failed {
                    code: "InjectedDeviceFailure".into(),
                    message: "physical callback after root replacement".into(),
                },
            },
        );
        assert!(out.protocol.is_empty());
        assert!(out.diagnostics.iter().any(|diagnostic| {
            diagnostic.message.contains("InjectedDeviceFailure")
                && diagnostic.message.contains("physical callback")
        }));
        assert_eq!(
            session.snapshot().state().snapshot().side_to_move(),
            contract::Color::Black
        );
    }

    #[test]
    fn transport_primary_and_source_diagnostics_survive_physical_drain_failure() {
        let source = Diagnostic {
            code: "SearchFailed",
            message: "original physical backend failure".into(),
        };
        let primary = EngineError::Transport(Box::new(ServeError {
            failure: ServeFailure::Io(std::io::Error::from(std::io::ErrorKind::BrokenPipe)),
            cleanup_failures: Vec::new(),
            diagnostics: vec![source.clone()],
        }));
        let error = finish_errors(Some(primary), vec![EngineError::DrainTimeout]).unwrap_err();
        let EngineError::Cleanup { primary, cleanup } = error else {
            panic!("compound cleanup required")
        };
        assert!(matches!(cleanup.as_slice(), [EngineError::DrainTimeout]));
        let EngineError::Transport(transport) = *primary else {
            panic!("transport primary required")
        };
        assert_eq!(transport.diagnostics, [source]);
        assert!(
            matches!(transport.failure, ServeFailure::Io(error) if error.kind() == std::io::ErrorKind::BrokenPipe)
        );
    }

    #[test]
    fn diagnostic_receipts_survive_full_wakeup_queue_and_preserve_repeated_causes() {
        let (mut owner, mut session) = fixture();
        let (sender, events) = mpsc::sync_channel(1);
        sender.send(Event::Line("isready".into())).unwrap();
        let cause = contract::ContractError::new(
            contract::ErrorCode::IdentityMismatch,
            contract::Stage::Output,
            "foreign response context",
        );
        retain_diagnostic(&owner.diagnostics, WorkerDiagnosticReceipt::Boundary(cause)).unwrap();
        retain_diagnostic(&owner.diagnostics, WorkerDiagnosticReceipt::Boundary(cause)).unwrap();
        assert!(
            sender
                .try_send(Event::RejectedInput {
                    code: "OwnerWake",
                    message: String::new()
                })
                .is_err()
        );
        let out = owner.handle(&mut session, events.recv().unwrap());
        assert_eq!(out.protocol, ["readyok"]);
        assert!(
            out.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "SearchDiagnostic"
                    && diagnostic.message.contains("Output/IdentityMismatch")
                    && diagnostic.message.contains("foreign response context")
                    && diagnostic.message.contains("occurrences=2"))
        );
        assert!(owner.diagnostics.lock().unwrap().is_empty());
    }

    fn foreign_failure_event(
        owner: &mut Owner,
        session: &mut Session<RulesUciPort>,
    ) -> rz_search::contracts::ContractPumpEvent {
        use rz_search::contracts::{ContractPumpEvent, ContractSearch, ContractSearchConfig};
        struct ForeignRuntime(Option<contract::EvalContext>);
        impl contract::Evaluator<RulesState> for ForeignRuntime {
            fn submit(
                &mut self,
                request: Arc<contract::EvalRequest<RulesState>>,
            ) -> Result<(), contract::ContractError> {
                assert!(self.0.replace(request.context()).is_none());
                Ok(())
            }
            fn poll(&mut self) -> Option<contract::EvalResult> {
                let mut context = self.0.take()?;
                context.request.sequence += 100;
                Some(contract::EvalResult::Failed(contract::EvalFailure {
                    context: contract::CompletionContext {
                        request: context,
                        execution: Some(contract::ExecutionId::new(context.request.epoch, 51)),
                    },
                    error: contract::ContractError::new(
                        contract::ErrorCode::BackendFailure,
                        contract::Stage::Backend,
                        "foreign physical callback failed",
                    ),
                    recovery: contract::RecoveryOutcome::Failed,
                }))
            }
            fn cancel(&mut self, _: contract::RequestId) -> Result<(), contract::ContractError> {
                Ok(())
            }
        }
        let start = owner.handle(session, Event::Line("go nodes 1".into()));
        let snapshot = start
            .effects
            .into_iter()
            .find_map(|effect| match effect {
                Effect::Start { snapshot, .. } => Some(snapshot),
                _ => None,
            })
            .unwrap();
        let profile = owner.factory.as_ref().unwrap().profile();
        let scope = owner.session_owner.scope();
        let config = ContractSearchConfig {
            scope,
            model: profile.model,
            precision: profile.precision,
            compute: profile.compute,
            bytes: profile.bytes,
            policy_tolerance: 1e-5,
            wdl_tolerance: 1e-5,
            cancellation: owner.session_owner.cancellation().unwrap().token(),
            deadlines: *owner.session_owner.deadlines().unwrap(),
            ids: Arc::clone(&owner.ids),
            max_simulations: 1,
            tree_limits: owner.settings.tree,
        };
        let mut search = ContractSearch::new(snapshot, config).unwrap();
        let mut runtime = ForeignRuntime(None);
        let pump = |search: &mut ContractSearch<RulesSearchPosition>,
                    runtime: &mut ForeignRuntime| {
            search.pump(
                runtime,
                &owner.clock,
                || scope,
                |_, _| Ok(contract::EvalInputKey(contract::Digest([9; 32]))),
            )
        };
        assert!(matches!(
            pump(&mut search, &mut runtime),
            ContractPumpEvent::Submitted { .. }
        ));
        let event = pump(&mut search, &mut runtime);
        assert!(
            matches!(&event, ContractPumpEvent::RejectedResult { result, .. } if matches!(result.as_ref(), contract::EvalResult::Failed(_)))
        );
        assert!(!search.is_finished());
        assert_eq!(search.outcome().counters.completed_visits, 0);
        event
    }

    #[test]
    fn actual_rules_search_rejection_keeps_original_failure_across_full_queue_and_new_root() {
        let (mut owner, mut session) = fixture();
        let event = foreign_failure_event(&mut owner, &mut session);
        // Replacement precedes delivery: a prior worker may only report causes.
        assert!(
            owner
                .handle(
                    &mut session,
                    Event::Line("position startpos moves e2e4".into())
                )
                .accepted
        );
        let (sender, events) = mpsc::sync_channel(1);
        sender.send(Event::Line("isready".into())).unwrap();
        consume_pump_diagnostic(&event, &owner.diagnostics, &sender).unwrap();
        consume_pump_diagnostic(&event, &owner.diagnostics, &sender).unwrap();
        let mut recovered = event.clone();
        let rz_search::contracts::ContractPumpEvent::RejectedResult { result, .. } = &mut recovered
        else {
            panic!("rejected result required")
        };
        let contract::EvalResult::Failed(failure) = result.as_mut() else {
            panic!("original failure required")
        };
        failure.recovery = contract::RecoveryOutcome::Completed;
        consume_pump_diagnostic(&recovered, &owner.diagnostics, &sender).unwrap();
        {
            let receipts = owner.diagnostics.lock().unwrap();
            assert_eq!(receipts.len(), 2);
            assert_eq!(receipts[0].1, 2);
            assert_eq!(receipts[1].1, 1);
            let WorkerDiagnosticReceipt::RejectedFailure { rejection, failure } = &receipts[0].0
            else {
                panic!("original failure required")
            };
            assert_eq!(rejection.code, contract::ErrorCode::IdentityMismatch);
            assert_eq!(failure.error.code, contract::ErrorCode::BackendFailure);
            assert_eq!(failure.error.detail, "foreign physical callback failed");
            assert_eq!(failure.recovery, contract::RecoveryOutcome::Failed);
            assert_eq!(failure.context.execution.unwrap().sequence, 51);
        }
        let out = owner.handle(&mut session, events.recv().unwrap());
        assert_eq!(out.protocol, ["readyok"]);
        assert!(out.diagnostics.iter().any(|diagnostic| {
            diagnostic
                .message
                .contains("foreign physical callback failed")
                && diagnostic.message.contains("recovery=Failed")
                && diagnostic.message.contains("occurrences=2")
        }));
        assert!(session.active_ticket().is_none());
        assert_eq!(
            session.snapshot().state().snapshot().side_to_move(),
            contract::Color::Black
        );
    }

    #[test]
    fn rejection_receipt_limit_retains_all_owned_classes_and_overflow_source() {
        let (mut owner, mut session) = fixture();
        let event = foreign_failure_event(&mut owner, &mut session);
        let rz_search::contracts::ContractPumpEvent::RejectedResult { error, result } = event
        else {
            panic!("rejected result required")
        };
        let contract::EvalResult::Failed(original) = *result else {
            panic!("original failure required")
        };
        for sequence in 1..=32 {
            let mut failure = original.clone();
            failure.context.request.request.sequence = sequence;
            retain_diagnostic(
                &owner.diagnostics,
                WorkerDiagnosticReceipt::RejectedFailure {
                    rejection: error,
                    failure: Box::new(failure),
                },
            )
            .unwrap();
        }
        let mut failure = original;
        failure.context.request.request.sequence = 33;
        failure.recovery = contract::RecoveryOutcome::Completed;
        let overflow = retain_diagnostic(
            &owner.diagnostics,
            WorkerDiagnosticReceipt::RejectedFailure {
                rejection: error,
                failure: Box::new(failure),
            },
        )
        .unwrap_err();
        assert_eq!(overflow.error.code, contract::ErrorCode::ResourceExhausted);
        let WorkerDiagnosticReceipt::RejectedFailure { failure, .. } = &overflow.receipt else {
            panic!("overflow cause required")
        };
        assert_eq!(failure.context.request.request.sequence, 33);
        assert_eq!(failure.recovery, contract::RecoveryOutcome::Completed);
        assert_eq!(failure.error.detail, "foreign physical callback failed");
        assert!(overflow.to_string().contains("recovery=Completed"));
        let receipts = owner.diagnostics.lock().unwrap();
        assert_eq!(receipts.len(), 32);
        assert!(receipts.iter().all(|(_, count)| *count == 1));
    }

    fn closed_receiver_worker(
        mode: FailureMode,
        invalid_constructor: bool,
        cleanup: Option<contract::ContractError>,
    ) -> (Result<(), EngineError>, u64, usize) {
        let (mut owner, mut session) = fixture();
        let drained = Arc::new(AtomicU64::new(0));
        owner.factory = Some(Arc::new(FailureFactory {
            profile: owner.factory.as_ref().unwrap().profile(),
            mode,
            drained: Arc::clone(&drained),
            cleanup,
        }));
        if invalid_constructor {
            owner.settings.tree.max_nodes = 0;
        }
        let start = owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let (sender, receiver) = mpsc::sync_channel(1);
        drop(receiver);
        for effect in start.effects {
            owner.dispatch(effect, &sender).unwrap();
        }
        assert_eq!(owner.workers.len(), 1);
        let until = Instant::now() + Duration::from_secs(5);
        while !owner.workers[0].is_finished() {
            assert!(Instant::now() < until, "worker must finitely terminate");
            thread::sleep(Duration::from_millis(1));
        }
        let result = owner.reap_workers(true);
        assert!(owner.workers.is_empty());
        let receipts = owner.diagnostics.lock().unwrap().len();
        (result, drained.load(Ordering::Acquire), receipts)
    }

    #[test]
    fn stop_and_deadline_keep_output_private_until_owned_physical_worker_finishes() {
        for stop in [true, false] {
            let (mut owner, mut session) = fixture();
            owner.handle(&mut session, Event::Line("go movetime 30".into()));
            let ticket = session.active_ticket().unwrap();
            let preferred = "e2e4";
            owner.handle(
                &mut session,
                Event::Progress {
                    ticket: ticket.clone(),
                    bestmove: preferred.into(),
                },
            );
            let (release, waiting) = mpsc::channel();
            owner.workers.push(Worker {
                handle: thread::spawn(move || {
                    let _ = waiting.recv();
                    Ok(())
                }),
                failure: Arc::new(Mutex::new(None)),
            });
            let event = if stop {
                Event::Line("stop".into())
            } else {
                let until = owner.session_owner.control().unwrap().deadline;
                thread::sleep(
                    until.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
                );
                Event::Deadline(ticket.clone())
            };
            let canceled = owner.handle(&mut session, event);
            assert!(
                !canceled
                    .protocol
                    .iter()
                    .any(|line| line.starts_with("bestmove ")),
                "stop/deadline exposed output while an owned physical worker was still blocked"
            );
            assert!(session.active_ticket().is_none());
            let ready = owner.handle(&mut session, Event::Line("isready".into()));
            assert_eq!(ready.protocol, ["readyok"]);
            let repeated = owner.handle(&mut session, Event::Line("stop".into()));
            assert!(repeated.protocol.is_empty());
            release.send(()).unwrap();
            let until = Instant::now() + Duration::from_secs(2);
            while !owner.workers[0].is_finished() {
                assert!(Instant::now() < until);
                thread::yield_now();
            }
            let finished = owner.handle(
                &mut session,
                Event::Complete {
                    ticket,
                    completion: SearchCompletion::Completed {
                        bestmove: Some("d2d4".into()),
                    },
                },
            );
            assert_eq!(finished.protocol, [format!("bestmove {preferred}")]);
            assert!(owner.workers.is_empty());
            assert!(
                owner
                    .handle(&mut session, Event::Line("stop".into()))
                    .protocol
                    .is_empty()
            );
        }
    }

    #[test]
    fn deferred_move_is_discarded_on_root_game_replacement_but_flushed_before_quit() {
        for replacement in ["position startpos moves e2e4", "ucinewgame", "quit"] {
            let (mut owner, mut session) = fixture();
            owner.handle(&mut session, Event::Line("go infinite".into()));
            let ticket = session.active_ticket().unwrap();
            let (release, waiting) = mpsc::channel();
            owner.workers.push(Worker {
                handle: thread::spawn(move || {
                    let _ = waiting.recv();
                    Ok(())
                }),
                failure: Arc::new(Mutex::new(None)),
            });
            owner.handle(&mut session, Event::Line("stop".into()));
            assert!(owner.pending_bestmove.is_some());
            let frozen = owner.pending_bestmove.as_ref().unwrap().line.clone();
            let next = owner.handle(&mut session, Event::Line(replacement.into()));
            assert!(next.accepted);
            assert!(next.protocol.is_empty());
            if replacement == "quit" {
                assert!(!session.is_closed());
                assert!(owner.pending_bestmove.is_some());
                assert!(owner.pending_close.is_some());
                assert!(!next.effects.iter().any(|e| matches!(e, Effect::Shutdown)));
            } else {
                assert!(owner.pending_bestmove.is_none());
                assert!(owner.pending_close.is_none());
            }
            release.send(()).unwrap();
            let until = Instant::now() + Duration::from_secs(2);
            while !owner.workers[0].is_finished() {
                assert!(Instant::now() < until);
                thread::yield_now();
            }
            let stale = owner.handle(
                &mut session,
                Event::Complete {
                    ticket,
                    completion: SearchCompletion::Completed {
                        bestmove: Some("e2e4".into()),
                    },
                },
            );
            if replacement == "quit" {
                assert_eq!(stale.protocol, [frozen]);
                assert!(session.is_closed());
                assert!(stale.effects.iter().any(|e| matches!(e, Effect::Shutdown)));
                assert!(owner.pending_bestmove.is_none());
                assert!(owner.pending_close.is_none());
                assert!(owner.fence_failure.is_none());
            } else {
                assert!(stale.protocol.is_empty());
                assert!(!session.is_closed());
            }
            owner.reap().unwrap();
            assert!(owner.workers.is_empty());
        }
    }

    #[test]
    fn failed_or_panicking_drain_keeps_original_failure_and_emits_no_acknowledgment() {
        for panic in [false, true] {
            let (mut owner, mut session) = fixture();
            owner.handle(&mut session, Event::Line("go infinite".into()));
            let (release, waiting) = mpsc::channel();
            let cause = failure(
                contract::ErrorCode::BackendFailure,
                "injected physical output fence failure",
            );
            owner.workers.push(Worker {
                handle: thread::spawn(move || {
                    let _ = waiting.recv();
                    if panic {
                        panic!("injected physical drain panic");
                    }
                    Err(EngineError::Contract(cause))
                }),
                failure: Arc::new(Mutex::new(None)),
            });
            owner.handle(&mut session, Event::Line("stop".into()));
            release.send(()).unwrap();
            let until = Instant::now() + Duration::from_secs(2);
            while !owner.workers[0].is_finished() {
                assert!(Instant::now() < until);
                thread::yield_now();
            }
            let closed = owner.handle(
                &mut session,
                Event::RejectedInput {
                    code: "OwnerWake",
                    message: String::new(),
                },
            );
            assert!(closed.protocol.is_empty());
            assert!(session.is_closed());
            assert!(owner.pending_bestmove.is_none());
            assert!(closed.effects.iter().any(|e| matches!(e, Effect::Shutdown)));
            match owner.fence_failure.take().unwrap() {
                EngineError::WorkerPanic if panic => {}
                EngineError::Contract(original) if !panic => assert_eq!(original, cause),
                other => panic!("physical cause was replaced: {other}"),
            }
        }
    }

    #[test]
    fn physical_fence_failure_reaches_serve_result_without_a_bestmove_acknowledgment() {
        let (owner, session) = fixture();
        let original = failure(
            contract::ErrorCode::BackendFailure,
            "injected final physical drain failure",
        );
        let factory = Arc::new(FailureFactory {
            profile: owner.factory.as_ref().unwrap().profile(),
            mode: FailureMode::Matching,
            drained: Arc::new(AtomicU64::new(0)),
            cleanup: Some(original),
        });
        let process =
            EngineProcess::new(factory, Arc::clone(&session.snapshot().owners), owner.clock);
        let (sender, receiver) = event_channel();
        sender.send(Event::Line("go nodes 1".into())).unwrap();
        let handle = thread::spawn(move || {
            let mut protocol = Vec::new();
            let mut diagnostics = Vec::new();
            let result = serve(
                receiver,
                sender,
                &mut protocol,
                &mut diagnostics,
                process,
                owner.settings,
            );
            (result, protocol, diagnostics)
        });
        let until = Instant::now() + Duration::from_secs(5);
        while !handle.is_finished() {
            assert!(
                Instant::now() < until,
                "bounded physical fence failure must close the event loop"
            );
            thread::sleep(Duration::from_millis(1));
        }
        let (result, protocol, diagnostics) = handle.join().unwrap();
        assert!(result.unwrap_err().to_string().contains(original.detail));
        assert!(!String::from_utf8(protocol).unwrap().contains("bestmove "));
        assert!(
            String::from_utf8(diagnostics)
                .unwrap()
                .contains("PhysicalFenceFailure")
        );
    }

    #[test]
    fn output_fence_timeout_is_finite_and_does_not_release_a_blocked_worker() {
        let (mut owner, mut session) = fixture();
        owner.settings.shutdown_limit = Duration::from_millis(1);
        owner.handle(&mut session, Event::Line("go infinite".into()));
        let (release, waiting) = mpsc::channel();
        owner.workers.push(Worker {
            handle: thread::spawn(move || {
                let _ = waiting.recv();
                Ok(())
            }),
            failure: Arc::new(Mutex::new(None)),
        });
        owner.handle(&mut session, Event::Line("stop".into()));
        thread::sleep(Duration::from_millis(3));
        let expired = owner.handle(
            &mut session,
            Event::RejectedInput {
                code: "OwnerWake",
                message: String::new(),
            },
        );
        assert!(expired.protocol.is_empty());
        assert!(session.is_closed());
        assert!(matches!(
            owner.fence_failure,
            Some(EngineError::DrainTimeout)
        ));
        assert_eq!(owner.workers.len(), 1);
        assert!(!owner.workers[0].is_finished());
        release.send(()).unwrap();
        let until = Instant::now() + Duration::from_secs(2);
        while !owner.workers[0].is_finished() {
            assert!(Instant::now() < until);
            thread::yield_now();
        }
        owner.reap().unwrap();
    }

    #[test]
    fn natural_worker_completion_is_not_published_while_physical_drain_is_blocked() {
        let (mut owner, mut session) = fixture();
        owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let ticket = session.active_ticket().unwrap();
        let (sender, events) = mpsc::sync_channel(1);
        let (entered, drain_started) = mpsc::sync_channel(1);
        let (release, drain_release) = mpsc::sync_channel(1);
        let drained = Arc::new(AtomicU64::new(0));
        let observed = Arc::clone(&drained);
        let handle = thread::spawn(move || {
            let mut runtime = FailureRuntime {
                mode: FailureMode::NoEvaluation,
                context: None,
                callbacks: 0,
                drained,
                cleanup: None,
                drain_gate: Some((entered, drain_release)),
            };
            finish_worker(
                &sender,
                ticket,
                WorkerCompletion::Completed {
                    bestmove: Some("e2e4".into()),
                },
                &Mutex::new(None),
                Some(&mut runtime),
                Duration::from_secs(2),
            )
        });
        drain_started.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(
            events.try_recv().is_err(),
            "completion escaped before physical drain"
        );
        release.send(()).unwrap();
        assert!(matches!(
            events.recv_timeout(Duration::from_secs(2)).unwrap(),
            Event::Complete { .. }
        ));
        handle.join().unwrap().unwrap();
        assert_eq!(observed.load(Ordering::Acquire), 1);
    }

    #[test]
    fn closed_receiver_preserves_factory_source_in_actual_worker_join() {
        let (result, drained, _) = closed_receiver_worker(FailureMode::Factory, false, None);
        let EngineError::UndeliveredWorkerFailure(failure) = result.unwrap_err() else {
            panic!("undelivered factory failure required")
        };
        assert!(
            matches!(&failure.completion, SearchCompletion::Failed { message, .. } if message == injected_backend_failure().detail)
        );
        let WorkerFailureSource::Factory(source) = failure.source else {
            panic!("typed factory source required")
        };
        assert_eq!(source, injected_backend_failure());
        assert_eq!(drained, 0, "failed factory created no runtime");
    }

    #[test]
    fn closed_receiver_search_constructor_failure_still_drains_and_keeps_cleanup() {
        let cleanup_cause = failure(
            contract::ErrorCode::BackendFailure,
            "physical drain failed after failed delivery",
        );
        let (result, drained, _) =
            closed_receiver_worker(FailureMode::Matching, true, Some(cleanup_cause));
        let EngineError::Cleanup { primary, cleanup } = result.unwrap_err() else {
            panic!("delivery and drain failures required")
        };
        assert!(
            matches!(cleanup.as_slice(), [EngineError::Contract(error)] if *error == cleanup_cause)
        );
        let EngineError::UndeliveredWorkerFailure(failure) = *primary else {
            panic!("undelivered constructor source required")
        };
        assert!(matches!(
            failure.source,
            WorkerFailureSource::SearchConstructor(_)
        ));
        assert!(matches!(
            failure.completion,
            SearchCompletion::Failed { .. }
        ));
        assert_eq!(drained, 1);
    }

    #[test]
    fn closed_receiver_matching_eval_failure_preserves_typed_context_and_recovery() {
        let (result, drained, _) = closed_receiver_worker(FailureMode::Matching, false, None);
        let EngineError::UndeliveredWorkerFailure(failure) = result.unwrap_err() else {
            panic!("undelivered search failure required")
        };
        let WorkerFailureSource::Search(source) = failure.source else {
            panic!("typed search source required")
        };
        let rz_search::contracts::ContractSearchFailure::Evaluation(evaluation) = *source else {
            panic!("original evaluator failure required")
        };
        assert_eq!(evaluation.error, injected_backend_failure());
        assert_eq!(evaluation.recovery, contract::RecoveryOutcome::Failed);
        assert_eq!(evaluation.context.execution.unwrap().sequence, 1);
        assert_eq!(
            evaluation.context.request.request.epoch,
            contract::ProcessEpoch(7)
        );
        assert_eq!(drained, 1);
    }

    #[test]
    fn closed_receiver_cap33_failure_keeps_original_receipt_and_drain_failure() {
        let cleanup_cause = failure(
            contract::ErrorCode::BackendFailure,
            "physical drain failed after failed delivery",
        );
        let (result, drained, receipts) =
            closed_receiver_worker(FailureMode::ForeignFlood, false, Some(cleanup_cause));
        let EngineError::Cleanup { primary, cleanup } = result.unwrap_err() else {
            panic!("overflow and drain failures required")
        };
        assert!(
            matches!(cleanup.as_slice(), [EngineError::Contract(error)] if *error == cleanup_cause)
        );
        let EngineError::UndeliveredWorkerFailure(failure) = *primary else {
            panic!("undelivered overflow source required")
        };
        let WorkerFailureSource::DiagnosticRetention { error, receipt } = failure.source else {
            panic!("typed overflow receipt required")
        };
        assert_eq!(error.code, contract::ErrorCode::ResourceExhausted);
        let WorkerDiagnosticReceipt::RejectedFailure { failure, .. } = *receipt else {
            panic!("original cap33 evaluator failure required")
        };
        assert_eq!(failure.error, injected_backend_failure());
        assert_eq!(failure.recovery, contract::RecoveryOutcome::Failed);
        assert_eq!(failure.context.execution.unwrap().sequence, 33);
        assert_eq!(
            failure.context.request.request.epoch,
            contract::ProcessEpoch(7)
        );
        assert_eq!(receipts, 32);
        assert_eq!(drained, 1);
    }

    #[test]
    fn closed_receiver_normal_cancel_completion_drains_without_delivery_error() {
        let (result, drained, _) = closed_receiver_worker(FailureMode::NoEvaluation, false, None);
        result.unwrap();
        assert_eq!(drained, 1);
    }

    #[test]
    fn queued_quit_before_successful_failed_send_keeps_unconsumed_typed_source() {
        let (mut owner, mut session) = fixture();
        owner.factory = Some(Arc::new(FailureFactory {
            profile: owner.factory.as_ref().unwrap().profile(),
            mode: FailureMode::Factory,
            drained: Arc::new(AtomicU64::new(0)),
            cleanup: None,
        }));
        let start = owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let (sender, receiver) = mpsc::sync_channel(2);
        sender.send(Event::Line("quit".into())).unwrap();
        for effect in start.effects {
            owner.dispatch(effect, &sender).unwrap();
        }
        let until = Instant::now() + Duration::from_secs(5);
        while !owner.workers[0].is_finished() {
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
        let failure_ticket = owner.workers[0]
            .failure
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .ticket
            .clone();
        // A completed worker whose failed publication is still queued continues
        // occupying its existing worker slot until actual owner consumption.
        owner.reap().unwrap();
        assert_eq!(owner.workers.len(), owner.settings.max_workers);
        let out = owner.handle(&mut session, receiver.recv().unwrap());
        assert!(out.protocol.is_empty());
        drop(receiver); // The successful second send was never consumed.
        let error = owner.reap_workers(true).unwrap_err();
        let EngineError::UndeliveredWorkerFailure(failure) = error else {
            panic!("unconsumed successful publication required")
        };
        assert_eq!(failure.ticket, failure_ticket);
        assert!(matches!(
            failure.completion,
            SearchCompletion::Failed { .. }
        ));
        let WorkerFailureSource::Factory(source) = failure.source else {
            panic!("typed queued factory source required")
        };
        assert_eq!(source, injected_backend_failure());
        assert!(owner.workers.is_empty());
    }

    #[test]
    fn consumed_failed_publication_acknowledges_slot_and_allows_next_root() {
        let (mut owner, mut session) = fixture();
        owner.factory = Some(Arc::new(FailureFactory {
            profile: owner.factory.as_ref().unwrap().profile(),
            mode: FailureMode::Factory,
            drained: Arc::new(AtomicU64::new(0)),
            cleanup: None,
        }));
        let start = owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let (sender, receiver) = mpsc::sync_channel(1);
        for effect in start.effects {
            owner.dispatch(effect, &sender).unwrap();
        }
        let until = Instant::now() + Duration::from_secs(5);
        while !owner.workers[0].is_finished() {
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
        let out = owner.handle(&mut session, receiver.recv().unwrap());
        assert_eq!(
            out.protocol
                .iter()
                .filter(|line| line.starts_with("bestmove "))
                .count(),
            1
        );
        assert!(out.diagnostics.iter().any(|diagnostic| {
            diagnostic
                .message
                .contains(injected_backend_failure().detail)
        }));
        // Publication acknowledgment and the physical output fence reap it.
        assert!(owner.workers.is_empty());
        let next = owner.handle(&mut session, Event::Line("go nodes 1".into()));
        assert!(next.accepted);
        assert!(
            next.effects
                .iter()
                .any(|effect| matches!(effect, Effect::Start { .. }))
        );
    }

    #[test]
    fn closed_receiver_panicking_drain_keeps_source_and_worker_panic() {
        let (result, drained, _) = closed_receiver_worker(FailureMode::PanicDrain, false, None);
        let EngineError::Cleanup { primary, cleanup } = result.unwrap_err() else {
            panic!("source and panic required")
        };
        assert!(matches!(cleanup.as_slice(), [EngineError::WorkerPanic]));
        let EngineError::UndeliveredWorkerFailure(failure) = *primary else {
            panic!("source must survive drain unwind")
        };
        let WorkerFailureSource::Search(source) = failure.source else {
            panic!("typed source required")
        };
        let rz_search::contracts::ContractSearchFailure::Evaluation(evaluation) = *source else {
            panic!("original evaluator failure required")
        };
        assert_eq!(evaluation.error, injected_backend_failure());
        assert_eq!(evaluation.recovery, contract::RecoveryOutcome::Failed);
        assert_eq!(drained, 1);
    }

    #[test]
    fn early_rejected_failure_after_diagnostic_poison_does_not_ack_original() {
        let (mut owner, mut session) = fixture();
        owner.factory = Some(Arc::new(FailureFactory {
            profile: owner.factory.as_ref().unwrap().profile(),
            mode: FailureMode::Factory,
            drained: Arc::new(AtomicU64::new(0)),
            cleanup: None,
        }));
        let start = owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let (sender, receiver) = mpsc::sync_channel(1);
        for effect in start.effects {
            owner.dispatch(effect, &sender).unwrap();
        }
        let until = Instant::now() + Duration::from_secs(5);
        while !owner.workers[0].is_finished() {
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
        let diagnostics = Arc::clone(&owner.diagnostics);
        assert!(
            thread::spawn(move || {
                let _guard = diagnostics.lock().unwrap();
                panic!("injected diagnostic owner poison");
            })
            .join()
            .is_err()
        );
        let out = owner.handle(&mut session, receiver.recv().unwrap());
        assert!(
            !out.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "SearchFailed")
        );
        assert!(owner.workers[0].failure.lock().unwrap().is_some());
        let EngineError::UndeliveredWorkerFailure(failure) = owner.reap_workers(true).unwrap_err()
        else {
            panic!("unconsumed source required")
        };
        assert!(
            matches!(failure.source, WorkerFailureSource::Factory(error) if error == injected_backend_failure())
        );
    }

    #[test]
    fn unfinished_drain_keeps_published_source_reclaimable_by_closing_owner() {
        struct BlockingDrain {
            entered: SyncSender<()>,
            release: Receiver<()>,
        }
        impl contract::Evaluator<RulesState> for BlockingDrain {
            fn submit(
                &mut self,
                _: Arc<contract::EvalRequest<RulesState>>,
            ) -> Result<(), contract::ContractError> {
                unreachable!("only shutdown is exercised")
            }
            fn poll(&mut self) -> Option<contract::EvalResult> {
                None
            }
            fn cancel(&mut self, _: contract::RequestId) -> Result<(), contract::ContractError> {
                Ok(())
            }
        }
        impl ManagedEvaluator for BlockingDrain {
            fn shutdown(&mut self, _: Instant) -> Result<(), contract::ContractError> {
                self.entered.send(()).unwrap();
                self.release.recv_timeout(Duration::from_secs(5)).unwrap();
                Ok(())
            }
        }
        struct ReleaseOnDrop(Option<SyncSender<()>>);
        impl Drop for ReleaseOnDrop {
            fn drop(&mut self) {
                if let Some(sender) = self.0.take() {
                    let _ = sender.try_send(());
                }
            }
        }
        let (mut owner, mut session) = fixture();
        owner.handle(&mut session, Event::Line("go nodes 1".into()));
        let ticket = session.active_ticket().unwrap();
        let expected_ticket = ticket.clone();
        let (sender, receiver) = mpsc::sync_channel(1);
        drop(receiver);
        let (entered, waiting) = mpsc::sync_channel(1);
        let (release, gate) = mpsc::sync_channel(1);
        let mut release = ReleaseOnDrop(Some(release));
        let publication = Arc::new(Mutex::new(None));
        let worker_publication = Arc::clone(&publication);
        let handle = thread::spawn(move || {
            let mut runtime = BlockingDrain {
                entered,
                release: gate,
            };
            finish_worker(
                &sender,
                ticket,
                WorkerCompletion::failed(WorkerFailureSource::Factory(injected_backend_failure())),
                &worker_publication,
                Some(&mut runtime),
                Duration::from_secs(5),
            )
        });
        owner.workers.push(Worker {
            handle,
            failure: publication,
        });
        waiting.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(!owner.workers[0].is_finished());
        owner.reap_workers(true).unwrap();
        assert_eq!(owner.workers.len(), 1);
        // The final owner collection also runs when its drain deadline expires.
        // A reclaimed failure does not mean the physical worker has completed.
        let mut recovered = owner.collect_unconsumed_failures();
        assert_eq!(recovered.len(), 1);
        let EngineError::UndeliveredWorkerFailure(failure) = recovered.pop().unwrap() else {
            panic!("registered source required before physical completion")
        };
        assert_eq!(failure.ticket, expected_ticket);
        assert!(
            matches!(failure.source, WorkerFailureSource::Factory(error) if error == injected_backend_failure())
        );
        assert!(!owner.workers[0].is_finished());
        release.0.take().unwrap().send(()).unwrap();
        let until = Instant::now() + Duration::from_secs(5);
        while !owner.workers[0].is_finished() {
            assert!(Instant::now() < until);
            thread::sleep(Duration::from_millis(1));
        }
        owner.reap_workers(true).unwrap();
    }

    #[test]
    fn returned_failure_before_thread_finish_keeps_original_owner_visible() {
        let (mut owner, mut session) = fixture();
        let event = foreign_failure_event(&mut owner, &mut session);
        let rz_search::contracts::ContractPumpEvent::RejectedResult { result, .. } = event else {
            panic!("foreign result fixture")
        };
        let contract::EvalResult::Failed(original) = *result else {
            panic!("original failure fixture")
        };
        let expected = original.context;
        let ticket = session.active_ticket().unwrap();
        let (sender, receiver) = mpsc::sync_channel(1);
        drop(receiver);
        let (entered, waiting) = mpsc::sync_channel(1);
        let (release, gate) = mpsc::sync_channel(1);
        let publication = Arc::new(Mutex::new(None));
        let worker_publication = Arc::clone(&publication);
        let handle = thread::spawn(move || {
            let result = finish_worker(
                &sender,
                ticket,
                WorkerCompletion::failed(WorkerFailureSource::Search(Box::new(
                    rz_search::contracts::ContractSearchFailure::Evaluation(Box::new(original)),
                ))),
                &worker_publication,
                None,
                Duration::from_secs(1),
            );
            // Model preemption after finish_worker returned and before the
            // thread publishes completion through JoinHandle::is_finished.
            entered.send(()).unwrap();
            gate.recv_timeout(Duration::from_secs(5)).unwrap();
            result
        });
        owner.workers.push(Worker {
            handle,
            failure: publication,
        });
        waiting.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(!owner.workers[0].is_finished());
        let mut recovered = owner.collect_unconsumed_failures();
        let EngineError::UndeliveredWorkerFailure(source) = recovered.pop().unwrap() else {
            panic!("original must remain visible after closure return")
        };
        let WorkerFailureSource::Search(source) = source.source else {
            panic!("typed original search failure")
        };
        let rz_search::contracts::ContractSearchFailure::Evaluation(source) = *source else {
            panic!("original evaluator failure")
        };
        assert_eq!(source.context, expected);
        assert_eq!(source.recovery, contract::RecoveryOutcome::Failed);
        release.send(()).unwrap();
        let result = owner.workers.pop().unwrap().handle.join().unwrap();
        let EngineError::UndeliveredWorkerFailure(copy) = result.unwrap_err() else {
            panic!("join must retain typed failure copy")
        };
        let WorkerFailureSource::Search(copy) = copy.source else {
            panic!("typed join search failure")
        };
        let rz_search::contracts::ContractSearchFailure::Evaluation(copy) = *copy else {
            panic!("join evaluator failure")
        };
        assert_eq!(copy.context, expected);
        assert_eq!(copy.recovery, contract::RecoveryOutcome::Failed);
    }

    #[test]
    fn pondering_is_explicit_and_keeps_the_normal_search_root_on_rejection() {
        let (mut owner, mut session) = fixture();
        let scope = owner.session_owner.scope();
        let rejected = owner.handle(
            &mut session,
            Event::Line("go ponder wtime 1000 btime 1000".into()),
        );
        assert!(!rejected.accepted && rejected.effects.is_empty());
        assert_eq!(owner.session_owner.scope().root, scope.root);
        assert!(session.active_ticket().is_none());
        owner.handle(
            &mut session,
            Event::Line("setoption name Ponder value true".into()),
        );
        let started = owner.handle(&mut session, Event::Line("go ponder nodes 1".into()));
        assert!(started.accepted);
        assert!(started.protocol.is_empty());
        assert_eq!(owner.ponder.unwrap().1, Some(1));
        assert!(owner.session_owner.control().unwrap().deadline > Instant::now());
    }

    #[test]
    fn ponder_normalization_preserves_root_restriction_and_registered_goals() {
        let (mut owner, mut session) = fixture();
        session
            .set_search_goal_capability(std::num::NonZeroU16::new(8))
            .unwrap();
        owner.handle(
            &mut session,
            Event::Line("setoption name Ponder value true".into()),
        );
        let started = owner.handle(
            &mut session,
            Event::Line(
                "go ponder wtime 1000 btime 2000 winc 1 binc 2 movestogo 8 nodes 1 depth 6 mate 3 searchmoves e2e4".into(),
            ),
        );
        assert!(started.accepted, "{:?}", started.diagnostics);
        let [Effect::Start { limits, .. }] = started.effects.as_slice() else {
            panic!(
                "expected one constrained ponder start: {:?}",
                started.effects
            );
        };
        assert_eq!(limits.searchmoves, Some(vec!["e2e4".into()]));
        assert_eq!(limits.depth_plies, std::num::NonZeroU16::new(6));
        assert_eq!(limits.mate_moves, std::num::NonZeroU16::new(3));
        assert_eq!(limits.nodes, Some(1));
        assert!(limits.infinite && !limits.ponder);
        assert_eq!(limits.white_time_ms, None);
        assert_eq!(limits.black_time_ms, None);
        assert_eq!(limits.white_increment_ms, None);
        assert_eq!(limits.black_increment_ms, None);
        assert_eq!(limits.moves_to_go, None);
        let (original_ticket, original_go) = owner.pending_original_go.as_ref().unwrap();
        let Effect::Start { ticket, .. } = &started.effects[0] else {
            unreachable!();
        };
        assert_eq!(original_ticket, ticket);
        assert!(original_go.ponder && !original_go.infinite);
        assert_eq!(original_go.white_time_ms, Some(1000));
        assert_eq!(original_go.black_time_ms, Some(2000));
        assert_eq!(original_go.white_increment_ms, Some(1));
        assert_eq!(original_go.black_increment_ms, Some(2));
        assert_eq!(original_go.moves_to_go, Some(8));
        assert_eq!(
            owner.session_owner.active_binding().unwrap().goal(),
            contract::rovent::SearchGoal {
                depth_plies: std::num::NonZeroU16::new(6),
                mate_moves: std::num::NonZeroU16::new(3),
            }
        );
        let stopped = owner.handle(&mut session, Event::Line("stop".into()));
        let [line] = stopped.protocol.as_slice() else {
            panic!("expected one constrained bestmove: {:?}", stopped.protocol);
        };
        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        let ["bestmove", "e2e4", "ponder", prediction] = fields.as_slice() else {
            panic!("enabled Ponder must retain the restricted root move: {line}");
        };
        // searchmoves restricts this root's move, not the opponent's reply.
        let mut child = Position::startpos();
        child
            .make_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        assert!(
            child
                .legal_moves()
                .contains(&BoardMove::from_uci(prediction).unwrap())
        );
    }

    type DriverOutputAcks = Arc<Mutex<Vec<(contract::pals::SearchAuthority, String, bool)>>>;

    struct DriverOutputProbe {
        inner: Arc<dyn crate::search_driver::SearchSessionDriver>,
        acknowledgments: DriverOutputAcks,
    }
    impl crate::search_driver::SearchSessionDriver for DriverOutputProbe {
        fn kind(&self) -> crate::search_driver::SearchKind {
            self.inner.kind()
        }
        fn implementation(&self) -> contract::Digest {
            self.inner.implementation()
        }
        fn identity(&self) -> EngineIdentity {
            self.inner.identity()
        }
        fn reset_game(&self) -> Result<(), contract::ContractError> {
            self.inner.reset_game()
        }
        fn run(
            &self,
            position: &Position,
            context: &crate::search_driver::SearchSessionContext,
            progress: &mut dyn FnMut(BoardMove),
        ) -> Result<
            crate::search_driver::SearchSessionReport,
            crate::search_driver::SearchSessionFailure,
        > {
            self.inner.run(position, context, progress)
        }
        fn accept_uci_output(
            &self,
            authority: contract::pals::SearchAuthority,
            bestmove: &str,
            from_report: bool,
        ) -> Result<(), contract::ContractError> {
            BoardMove::from_uci(bestmove)
                .expect("annotated output must retain one legal move token");
            self.inner
                .accept_uci_output(authority, bestmove, from_report)?;
            self.acknowledgments
                .lock()
                .unwrap()
                .push((authority, bestmove.into(), from_report));
            Ok(())
        }
    }

    pub(super) fn search_driver_output_fixture(
        pals: bool,
    ) -> (Owner, Session<RulesUciPort>, DriverOutputAcks) {
        let (mut owner, session) = fixture();
        let cpu = rz_search::cpu::CpuConfig {
            tt_entries: 0,
            max_depth: 1,
            ..rz_search::cpu::CpuConfig::default()
        };
        let inner: Arc<dyn crate::search_driver::SearchSessionDriver> = if pals {
            Arc::new(
                crate::search_driver::PalsSessionDriver::new(
                    rz_search::pals::engine::PalsConfig::default(),
                    rz_search::pals::engine::LegalOrderRoleMock,
                    cpu,
                    1,
                    8,
                    1,
                    EngineIdentity {
                        name: "PALS ponder CPU/mock fixture".into(),
                        author: "test".into(),
                    },
                )
                .unwrap(),
            )
        } else {
            Arc::new(crate::search_driver::CpuSessionDriver::new(cpu, 8).unwrap())
        };
        let scope = crate::contracts::SessionScope::Search(contract::pals::SearchAuthority {
            epoch: owner.clock.epoch(),
            game: contract::GameGeneration(1),
            root: contract::RootGeneration(0),
            implementation: inner.implementation(),
        });
        let acknowledgments = Arc::new(Mutex::new(Vec::new()));
        owner.session_owner = crate::contracts::ContractSessionOwner::new_scope(
            scope,
            owner.clock.epoch(),
            owner.clock.b_clock(),
            owner.settings.search,
        )
        .unwrap();
        owner.scope = Arc::new(Mutex::new(scope));
        owner.evaluation_scope = None;
        owner.factory = None;
        owner.driver = Some(Arc::new(DriverOutputProbe {
            inner,
            acknowledgments: Arc::clone(&acknowledgments),
        }));
        (owner, session, acknowledgments)
    }

    pub(super) fn completed_driver_search(
        owner: &mut Owner,
        session: &mut Session<RulesUciPort>,
        ponder: bool,
    ) -> SessionResult<RulesSearchPosition> {
        if ponder {
            assert!(
                owner
                    .handle(
                        session,
                        Event::Line("setoption name Ponder value true".into()),
                    )
                    .accepted
            );
        }
        let start = owner.handle(
            session,
            Event::Line(
                if ponder {
                    "go ponder nodes 1"
                } else {
                    "go nodes 1"
                }
                .into(),
            ),
        );
        assert!(start.accepted);
        let (sender, events) = event_channel();
        for effect in start.effects {
            owner.dispatch(effect, &sender).unwrap();
        }
        let until = Instant::now() + Duration::from_secs(5);
        let mut out = SessionResult::default();
        loop {
            let event = events
                .recv_timeout(until.saturating_duration_since(Instant::now()))
                .expect("finite actual CPU/mock driver completion");
            let terminal = matches!(&event, Event::Complete { .. });
            append_result(&mut out, owner.handle(session, event));
            if terminal {
                break;
            }
            assert!(Instant::now() < until);
        }
        while owner.workers.iter().any(|worker| !worker.is_finished()) {
            assert!(
                Instant::now() < until,
                "physical worker must finish finitely"
            );
            thread::yield_now();
        }
        if !ponder {
            append_result(
                &mut out,
                owner.handle(
                    session,
                    Event::RejectedInput {
                        code: "OwnerWake",
                        message: String::new(),
                    },
                ),
            );
        }
        out
    }

    #[test]
    fn own_cpu_and_pals_ponder_hit_stop_preserve_the_admitted_report_after_physical_join() {
        for pals in [false, true] {
            for release in ["ponderhit", "stop"] {
                let (mut owner, mut session, acknowledgments) = search_driver_output_fixture(pals);
                let complete = completed_driver_search(&mut owner, &mut session, true);
                assert!(complete.protocol.is_empty());
                assert!(owner.held_ponder_bestmove.is_none());
                let report = owner.admitted_ponder_report.as_ref().unwrap();
                assert_eq!(Some(&report.ticket), session.active_ticket().as_ref());
                assert_eq!(report.scope, owner.session_owner.lifecycle_scope());
                assert!(acknowledgments.lock().unwrap().is_empty());
                let released = owner.handle(&mut session, Event::Line(release.into()));
                assert_eq!(released.protocol.len(), 1);
                let bestmove = released.protocol[0]
                    .strip_prefix("bestmove ")
                    .unwrap()
                    .split_ascii_whitespace()
                    .next()
                    .unwrap();
                assert!(released.protocol[0].contains(" ponder "));
                let crate::contracts::SessionScope::Search(authority) =
                    owner.session_owner.lifecycle_scope()
                else {
                    panic!("actual CPU/PALS search authority");
                };
                assert_eq!(
                    *acknowledgments.lock().unwrap(),
                    vec![(authority, bestmove.into(), true)]
                );
                assert!(owner.workers.is_empty());
                assert!(owner.held_ponder_bestmove.is_none());
                assert!(owner.admitted_ponder_report.is_none());
                assert!(
                    owner
                        .handle(&mut session, Event::Line(release.into()))
                        .protocol
                        .is_empty()
                );
                assert_eq!(acknowledgments.lock().unwrap().len(), 1);
            }
            let (mut owner, mut session, acknowledgments) = search_driver_output_fixture(pals);
            let normal = completed_driver_search(&mut owner, &mut session, false);
            assert_eq!(normal.protocol.len(), 1);
            assert!(!normal.protocol[0].contains(" ponder "));
            assert!(owner.ponder.is_none() && owner.held_ponder_bestmove.is_none());
            assert!(owner.admitted_ponder_report.is_none());
            assert_eq!(acknowledgments.lock().unwrap().len(), 1);
            assert!(acknowledgments.lock().unwrap()[0].2);
        }
    }

    #[test]
    fn own_cpu_and_pals_root_newgame_revoke_held_ponder_report_authority() {
        for pals in [false, true] {
            for replacement in ["position startpos moves e2e4", "ucinewgame"] {
                let (mut owner, mut session, acknowledgments) = search_driver_output_fixture(pals);
                completed_driver_search(&mut owner, &mut session, true);
                let prior = owner.session_owner.lifecycle_scope();
                let replaced = owner.handle(&mut session, Event::Line(replacement.into()));
                assert!(replaced.accepted && replaced.protocol.is_empty());
                assert_ne!(owner.session_owner.lifecycle_scope(), prior);
                assert!(owner.ponder.is_none() && owner.held_ponder_bestmove.is_none());
                assert!(owner.admitted_ponder_report.is_none());
                let (sender, _events) = event_channel();
                for effect in replaced.effects {
                    owner.dispatch(effect, &sender).unwrap();
                }
                let stale = owner.handle(&mut session, Event::Line("ponderhit".into()));
                assert!(!stale.accepted && stale.protocol.is_empty());
                assert!(acknowledgments.lock().unwrap().is_empty());
                owner.reap_workers(true).unwrap();
                assert!(owner.workers.is_empty());
            }
        }
    }

    #[test]
    fn own_cpu_and_pals_physical_unknown_discards_held_ponder_without_output_ack() {
        for pals in [false, true] {
            let (mut owner, mut session, acknowledgments) = search_driver_output_fixture(pals);
            completed_driver_search(&mut owner, &mut session, true);
            let ticket = owner.active.as_ref().unwrap().ticket.clone();
            let error = crate::search_driver::SearchSessionFailure::physical_completion_unknown();
            *owner.workers[0].failure.lock().unwrap() = Some(Box::new(UndeliveredWorkerFailure {
                ticket,
                completion: SearchCompletion::Failed {
                    code: error.code.into(),
                    message: error.detail.clone(),
                },
                source: WorkerFailureSource::Driver(error),
            }));
            let stopped = owner.handle(&mut session, Event::Line("ponderhit".into()));
            assert!(stopped.protocol.is_empty());
            assert!(session.is_closed() && owner.fence_failure.is_some());
            assert!(owner.ponder.is_none() && owner.held_ponder_bestmove.is_none());
            assert!(owner.admitted_ponder_report.is_none());
            assert!(owner.pending_bestmove.is_none());
            assert!(acknowledgments.lock().unwrap().is_empty());
            assert!(
                stopped
                    .diagnostics
                    .iter()
                    .any(|d| d.code == "PhysicalFenceFailure")
            );
            owner.reap_workers(true).unwrap();
        }
    }

    #[test]
    fn own_cpu_and_pals_admitted_ponder_report_survives_the_passive_resource_deadline() {
        for pals in [false, true] {
            let (mut owner, mut session, acknowledgments) = search_driver_output_fixture(pals);
            completed_driver_search(&mut owner, &mut session, true);
            let ticket = session.active_ticket().unwrap();
            let deadline = owner.session_owner.control().unwrap().deadline;
            let token = owner.session_owner.cancellation().unwrap().token();
            let expired = owner.session_owner.handle_event_at(
                &mut session,
                Event::Deadline(ticket),
                deadline + Duration::from_millis(1),
            );
            assert!(expired.errors.is_empty());
            assert!(expired.session.protocol.is_empty());
            assert!(token.is_canceled());
            assert!(owner.admitted_ponder_report.is_some());
            let hit = owner.handle(&mut session, Event::Line("ponderhit".into()));
            assert_eq!(hit.protocol.len(), 1);
            assert_eq!(acknowledgments.lock().unwrap().len(), 1);
            assert!(acknowledgments.lock().unwrap()[0].2);
            assert!(owner.workers.is_empty());
            assert!(owner.admitted_ponder_report.is_none());
        }
    }

    #[test]
    fn own_cpu_and_pals_ponder_stop_before_late_completion_keeps_fallback_provenance() {
        for pals in [false, true] {
            let (mut owner, mut session, acknowledgments) = search_driver_output_fixture(pals);
            owner.handle(
                &mut session,
                Event::Line("setoption name Ponder value true".into()),
            );
            owner.handle(&mut session, Event::Line("go ponder nodes 1".into()));
            let ticket = session.active_ticket().unwrap();
            owner.active = Some(ActiveBinding {
                ticket: ticket.clone(),
                control: owner.session_owner.control().unwrap().clone(),
                authority: None,
                driver_output: None,
                driver_output_until: None,
                driver_start: None,
                ponder_clock: None,
            });
            owner.handle(
                &mut session,
                Event::Progress {
                    ticket: ticket.clone(),
                    bestmove: "e2e4".into(),
                },
            );
            let (release, waiting) = mpsc::channel();
            owner.workers.push(Worker {
                handle: thread::spawn(move || {
                    waiting.recv_timeout(Duration::from_secs(2)).unwrap();
                    Ok(())
                }),
                failure: Arc::new(Mutex::new(None)),
            });
            let stopped = owner.handle(&mut session, Event::Line("stop".into()));
            assert!(stopped.protocol.is_empty());
            assert!(!owner.pending_bestmove.as_ref().unwrap().from_driver_report);
            let late = owner.handle(
                &mut session,
                Event::Complete {
                    ticket,
                    completion: SearchCompletion::Completed {
                        bestmove: Some("e2e4".into()),
                    },
                },
            );
            assert!(!late.accepted && late.protocol.is_empty());
            assert!(owner.admitted_ponder_report.is_none());
            assert!(!owner.pending_bestmove.as_ref().unwrap().from_driver_report);
            release.send(()).unwrap();
            let until = Instant::now() + Duration::from_secs(2);
            while !owner.workers[0].is_finished() {
                assert!(Instant::now() < until);
                thread::yield_now();
            }
            let finished = owner.handle(
                &mut session,
                Event::RejectedInput {
                    code: "OwnerWake",
                    message: String::new(),
                },
            );
            assert!(finished.protocol[0].starts_with("bestmove e2e4 ponder "));
            assert_eq!(acknowledgments.lock().unwrap().len(), 1);
            assert!(!acknowledgments.lock().unwrap()[0].2);
            assert!(owner.workers.is_empty());
        }
    }

    #[test]
    fn completed_ponder_waits_for_hit_and_returns_the_candidate_exactly_once() {
        let (mut owner, mut session) = fixture();
        owner.handle(
            &mut session,
            Event::Line("setoption name Ponder value true".into()),
        );
        owner.handle(
            &mut session,
            Event::Line("go ponder wtime 1 btime 1".into()),
        );
        let ticket = session.active_ticket().unwrap();
        let token = owner.session_owner.cancellation().unwrap().token();
        let complete = owner.handle(
            &mut session,
            Event::Complete {
                ticket: ticket.clone(),
                completion: SearchCompletion::Completed {
                    bestmove: Some("e2e4".into()),
                },
            },
        );
        assert!(complete.protocol.is_empty());
        assert_eq!(
            owner
                .handle(&mut session, Event::Line("isready".into()))
                .protocol,
            ["readyok"]
        );
        let hit = owner.handle(&mut session, Event::Line("ponderhit".into()));
        assert!(hit.accepted);
        assert!(hit.protocol[0].starts_with("bestmove e2e4 ponder "));
        assert!(token.is_canceled());
        assert!(session.active_ticket().is_none());
        assert!(
            owner
                .handle(&mut session, Event::Line("ponderhit".into()))
                .protocol
                .is_empty()
        );
        assert!(
            owner
                .handle(
                    &mut session,
                    Event::Complete {
                        ticket,
                        completion: SearchCompletion::Completed {
                            bestmove: Some("d2d4".into())
                        }
                    }
                )
                .protocol
                .is_empty()
        );
    }

    #[test]
    fn failed_ponder_preserves_diagnostics_and_holds_fallback_until_stop() {
        let (mut owner, mut session) = fixture();
        owner.handle(
            &mut session,
            Event::Line("setoption name Ponder value true".into()),
        );
        owner.handle(&mut session, Event::Line("go ponder".into()));
        let ticket = session.active_ticket().unwrap();
        let failed = owner.handle(
            &mut session,
            Event::Complete {
                ticket,
                completion: SearchCompletion::Failed {
                    code: "InjectedPonderFailure".into(),
                    message: "original source".into(),
                },
            },
        );
        assert!(failed.protocol.is_empty());
        assert!(
            failed
                .diagnostics
                .iter()
                .any(|d| d.message.contains("original source"))
        );
        let stop = owner.handle(&mut session, Event::Line("stop".into()));
        assert_eq!(
            stop.protocol
                .iter()
                .filter(|s| s.starts_with("bestmove "))
                .count(),
            1
        );
        assert!(
            owner
                .handle(&mut session, Event::Line("stop".into()))
                .protocol
                .is_empty()
        );
    }

    #[test]
    fn accepted_position_replacement_suppresses_an_old_held_ponder_failure() {
        let (mut owner, mut session) = fixture();
        owner.handle(
            &mut session,
            Event::Line("setoption name Ponder value true".into()),
        );
        owner.handle(&mut session, Event::Line("go ponder".into()));
        let ticket = session.active_ticket().unwrap();
        owner.handle(
            &mut session,
            Event::Complete {
                ticket,
                completion: SearchCompletion::Failed {
                    code: "Injected".into(),
                    message: "retained".into(),
                },
            },
        );
        assert!(owner.held_ponder_bestmove.is_some());
        let bad = owner.handle(
            &mut session,
            Event::Line("position startpos moves e2e5".into()),
        );
        assert!(!bad.accepted && owner.held_ponder_bestmove.is_some());
        let good = owner.handle(
            &mut session,
            Event::Line("position startpos moves e2e4".into()),
        );
        assert!(good.accepted && good.protocol.is_empty());
        assert!(owner.ponder.is_none() && owner.held_ponder_bestmove.is_none());
        assert!(
            owner
                .handle(&mut session, Event::Line("ponderhit".into()))
                .protocol
                .is_empty()
        );
    }

    #[test]
    fn injected_identity_reaches_uci_and_newline_identity_is_rejected_before_output() {
        let run = |name: &str| {
            let (owner, session) = fixture();
            let process = EngineProcess::new(
                Arc::clone(owner.factory.as_ref().unwrap()),
                Arc::clone(&session.snapshot().owners),
                owner.clock.clone(),
            )
            .with_identity(EngineIdentity {
                name: name.into(),
                author: "RoveZero identity fixture".into(),
            });
            let (sender, events) = event_channel();
            sender.send(Event::Line("uci".into())).unwrap();
            sender.send(Event::Line("quit".into())).unwrap();
            let mut protocol = Vec::new();
            let mut diagnostics = Vec::new();
            let result = serve(
                events,
                sender,
                &mut protocol,
                &mut diagnostics,
                process,
                owner.settings,
            );
            (result, String::from_utf8(protocol).unwrap(), diagnostics)
        };

        let (result, protocol, diagnostics) = run("RoveZero native CPU identity fixture");
        result.unwrap();
        assert_eq!(
            protocol.lines().collect::<Vec<_>>(),
            [
                "id name RoveZero native CPU identity fixture",
                "id author RoveZero identity fixture",
                "option name Ponder type check default false",
                "uciok",
            ]
        );
        assert!(diagnostics.is_empty());

        let (result, protocol, diagnostics) = run("RoveZero native CPU\nid name injected");
        let Err(EngineError::Session(error)) = result else {
            panic!("Session must reject the injected newline identity")
        };
        assert!(error.to_string().contains("engine identity requires"));
        assert!(protocol.is_empty());
        assert!(diagnostics.is_empty());
    }
}
