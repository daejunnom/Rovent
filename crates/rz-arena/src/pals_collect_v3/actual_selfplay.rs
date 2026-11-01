//! Actual in-process native session selfplay. Each game owns two independently
//! fenced PCV/CPU_T session epochs; this does not attest an OS process restart.
//! A formal process-restart profile still requires an outer whole-game worker
//! whose mapped executable and private cwd satisfy both startup declarations.

use super::*;
use rz_contracts::ProcessEpoch;
use rz_contracts::pals::SearchAuthority;
use rz_search::pals::engine::PalsCompletion;
use rz_search::time::{TimeBudget, TimeBudgetConfig, TimeControl};
use rz_uci::pals_adaptive::{
    AdaptiveCollectionDispatchObserver, AdaptiveManagerError, AdaptiveSessionManager,
    AdaptiveSessionProcessReceipt,
};
use rz_uci::pals_adaptive_bootstrap::{
    ActualAdaptiveBackendV5, AdaptiveBootstrapError, AdaptiveCollectionHooksV5,
    PreparedAdaptiveProcessV5, prepare_collection_actual,
    prepare_collection_actual_with_cancellation,
};
use rz_uci::pals_adaptive_startup::PinnedPalsStartupV5;
use rz_uci::pals_native::startup_virtual::NativeVirtualReviewIntakeV5;
use rz_uci::search_driver::{
    SearchSessionContext, SearchSessionDriver, SearchSessionFailure, SearchSessionReport,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

static NEXT_ACTUAL_SESSION_EPOCH_V3: AtomicU64 = AtomicU64::new(1);

/// Reserve a nonzero, never reused range for real in-process native owners.
/// Selfplay and isolated-action factories share this allocator in one worker.
pub fn reserve_actual_session_epochs_v3(count: u64) -> Result<ProcessEpoch, ArenaError> {
    check(
        count > 0 && count <= 8,
        "actual session epoch range capacity",
    )?;
    NEXT_ACTUAL_SESSION_EPOCH_V3
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |next| {
            next.checked_add(count)
        })
        .map(ProcessEpoch)
        .map_err(|_| invalid("actual session epoch range exhausted"))
}

/// Executes only after an explicit collector calls `start_game`. Construction
/// does not load a model, start a CPU helper or change the process cwd.
pub struct ActualGenerationSelfplayFactoryV3 {
    pinned: [Arc<PinnedPalsStartupV5>; 2],
    time_config: TimeBudgetConfig,
    native_intakes: [Option<NativeVirtualReviewIntakeV5>; 2],
    native_intakes_required: bool,
    cancellation: Option<Arc<AtomicBool>>,
}
impl ActualGenerationSelfplayFactoryV3 {
    pub fn new(
        pinned: [Arc<PinnedPalsStartupV5>; 2],
        time_config: TimeBudgetConfig,
    ) -> Result<Self, ArenaError> {
        time_config
            .validate()
            .map_err(|e| invalid(format!("actual selfplay clock settings: {e}")))?;
        check(
            pinned.iter().all(|owner| {
                !owner.metadata_preflight
                    && owner.config.endpoint.mode == rz_experiments::PalsEngineModeV5::Collection
            }),
            "actual selfplay requires two collection startup pins",
        )?;
        Ok(Self {
            pinned,
            time_config,
            native_intakes: [None, None],
            native_intakes_required: false,
            cancellation: None,
        })
    }

    /// One whole-game child supplies two once-only readbacks from its actual
    /// loaded backends. Reusing this factory for a second game is rejected;
    /// a new OS child owns its own hooks and native sessions.
    pub fn with_native_intakes(mut self, intakes: [NativeVirtualReviewIntakeV5; 2]) -> Self {
        self.native_intakes = intakes.map(Some);
        self.native_intakes_required = true;
        self
    }

    /// Share the caller's original child lifetime. Startup and each actual
    /// root observe this same latched signal; no new authority is issued.
    pub fn with_cancellation(mut self, cancellation: Arc<AtomicBool>) -> Self {
        self.cancellation = Some(cancellation);
        self
    }

    fn prepare_actual(
        &self,
        index: usize,
        epoch: ProcessEpoch,
        hooks: AdaptiveCollectionHooksV5,
        until: Instant,
    ) -> Result<PreparedAdaptiveProcessV5, AdaptiveBootstrapError> {
        match &self.cancellation {
            Some(cancellation) => prepare_collection_actual_with_cancellation(
                Arc::clone(&self.pinned[index]),
                epoch,
                hooks,
                until,
                Arc::clone(cancellation),
            ),
            None => prepare_collection_actual(Arc::clone(&self.pinned[index]), epoch, hooks, until),
        }
    }
}

struct ActualRootExpectationV3 {
    authority: SearchAuthority,
    anchor: GenerationRulesAnchorV3,
}
#[derive(Default)]
struct ActualTurnSlotV3 {
    expected: Option<ActualRootExpectationV3>,
    admitted: bool,
    completed: Option<GenerationActualTurnV3>,
}

struct ActualDecisionWorkerV3 {
    owner: usize,
    context: SearchSessionContext,
    handle: std::thread::JoinHandle<ActualDecisionFinishedV3>,
}
struct ActualDecisionFinishedV3 {
    result: Result<SearchSessionReport, SearchSessionFailure>,
    finished: Instant,
}

fn manager_root_v3(
    manager: &AdaptiveSessionManager<ActualAdaptiveBackendV5>,
) -> Result<(SearchAuthority, GenerationRulesAnchorV3), AdaptiveManagerError> {
    let port = manager
        .driver()
        .port()
        .ok_or(AdaptiveManagerError::Invalid(
            "actual Search owner unavailable",
        ))?;
    let root = Position::from_snapshot(
        port.root_rules().rules().snapshot(),
        rz_position::PositionLimits::default(),
    )
    .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
    Ok((
        port.authority(),
        exact_rules_anchor(&root).map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?,
    ))
}

fn actual_hooks_v3(
    native_observer: NativeGenerationObserverV3,
    dispatch_observer: AdaptiveCollectionDispatchObserver,
    teacher_observer: GenerationTeacherObserverV3,
    slot: Arc<Mutex<ActualTurnSlotV3>>,
    native_intake: Option<NativeVirtualReviewIntakeV5>,
) -> AdaptiveCollectionHooksV5 {
    let admission_slot = Arc::clone(&slot);
    let teacher_source = teacher_observer.clone();
    AdaptiveCollectionHooksV5 {
        native_observer: Box::new(native_observer),
        native_intake,
        dispatch_observer,
        teacher_observer: teacher_observer.observer(),
        teacher_admission: Arc::new(move |admitted| {
            teacher_source
                .register_source(admitted)
                .map_err(|e| AdaptiveBootstrapError {
                    stage: "CollectionTeacherSource",
                    detail: e.to_string(),
                    cleanup_failures: None,
                })
        }),
        root_admission: Arc::new(move |manager| {
            let (authority, anchor) = manager_root_v3(manager)?;
            let mut slot = admission_slot
                .lock()
                .map_err(|_| AdaptiveManagerError::Invalid("actual turn slot poisoned"))?;
            let expected = slot.expected.as_ref().ok_or(AdaptiveManagerError::Invalid(
                "no actual decision authority",
            ))?;
            if slot.admitted
                || slot.completed.is_some()
                || expected.authority != authority
                || expected.anchor != anchor
            {
                return Err(AdaptiveManagerError::Invalid(
                    "actual decision admission root or authority differs",
                ));
            }
            slot.admitted = true;
            Ok(())
        }),
        root_closure: Arc::new(move |manager, completion: PalsCompletion| {
            let (authority, anchor) = manager_root_v3(manager)?;
            let mut slot = slot
                .lock()
                .map_err(|_| AdaptiveManagerError::Invalid("actual turn slot poisoned"))?;
            let expected = slot.expected.as_ref().ok_or(AdaptiveManagerError::Invalid(
                "no actual decision authority",
            ))?;
            if !slot.admitted
                || slot.completed.is_some()
                || expected.authority != authority
                || expected.anchor != anchor
            {
                return Err(AdaptiveManagerError::Invalid(
                    "actual decision closure root or authority differs",
                ));
            }
            let turn = GenerationActualTurnV3::from_actual_manager(manager, completion)
                .map_err(|e| AdaptiveManagerError::Semantic(e.to_string()))?;
            slot.completed = Some(turn);
            Ok(())
        }),
    }
}

pub struct ActualGenerationSelfplayGameV3 {
    processes: [PreparedAdaptiveProcessV5; 2],
    slots: [Arc<Mutex<ActualTurnSlotV3>>; 2],
    epochs: [ProcessEpoch; 2],
    names: [String; 2],
    engine_a_white: bool,
    initial: GenerationRulesAnchorV3,
    decided_once: bool,
    time_config: TimeBudgetConfig,
    drain: Duration,
    close_deadline: Option<Instant>,
    failed: Option<String>,
    active_worker: Option<ActualDecisionWorkerV3>,
    cancellation: Option<Arc<AtomicBool>>,
}

impl GenerationSelfplayFactoryV3 for ActualGenerationSelfplayFactoryV3 {
    type Game = ActualGenerationSelfplayGameV3;

    fn start_game(
        &mut self,
        assignment: &GenerationGameAssignmentV3,
        initial: &Position,
        native_observers: [NativeGenerationObserverV3; 2],
        dispatch_observers: [AdaptiveCollectionDispatchObserver; 2],
        teacher_observers: [GenerationTeacherObserverV3; 2],
        until: Instant,
    ) -> Result<Self::Game, ArenaError> {
        check(
            Instant::now() < until
                && !self
                    .cancellation
                    .as_ref()
                    .is_some_and(|signal| signal.load(Ordering::Acquire)),
            "actual selfplay startup deadline/cancellation",
        )?;
        check(
            !self.native_intakes_required || self.native_intakes.iter().all(Option::is_some),
            "whole-game native intake hooks were already consumed",
        )?;
        let [intake_a, intake_b] = std::mem::take(&mut self.native_intakes);
        let initial = exact_rules_anchor(initial)?;
        let first = reserve_actual_session_epochs_v3(2)?;
        let epochs = [first, ProcessEpoch(first.0 + 1)]; // range already checked
        let slots = [
            Arc::new(Mutex::new(ActualTurnSlotV3::default())),
            Arc::new(Mutex::new(ActualTurnSlotV3::default())),
        ];
        let [native_a, native_b] = native_observers;
        let [dispatch_a, dispatch_b] = dispatch_observers;
        let [teacher_a, teacher_b] = teacher_observers;
        let a = self
            .prepare_actual(
                0,
                epochs[0],
                actual_hooks_v3(
                    native_a,
                    dispatch_a,
                    teacher_a,
                    Arc::clone(&slots[0]),
                    intake_a,
                ),
                until,
            )
            .map_err(|e| invalid(format!("actual engine A startup: {e}")))?;
        let b = match self.prepare_actual(
            1,
            epochs[1],
            actual_hooks_v3(
                native_b,
                dispatch_b,
                teacher_b,
                Arc::clone(&slots[1]),
                intake_b,
            ),
            until,
        ) {
            Ok(b) => b,
            Err(error) => {
                // Cleanup uses the original startup bound; a failed attempt
                // never gains an extended deadline by dropping its first owner.
                let cleanup = a.finish_process(until);
                return Err(invalid(match cleanup {
                    Ok(_) => format!("actual engine B startup: {error}"),
                    Err(cleanup) => format!(
                        "actual engine B startup: {error}; engine A physical cleanup: {cleanup}; retained={:?}",
                        a.driver().retained_owner_status()
                    ),
                }));
            }
        };
        let names = [a.driver().identity().name, b.driver().identity().name];
        let drain = Duration::from_millis(
            self.pinned[0]
                .config
                .native_options
                .drain_ms
                .max(self.pinned[1].config.native_options.drain_ms),
        );
        Ok(ActualGenerationSelfplayGameV3 {
            processes: [a, b],
            slots,
            epochs,
            names,
            engine_a_white: assignment.engine_a_white,
            initial,
            decided_once: false,
            time_config: self.time_config,
            drain,
            close_deadline: None,
            failed: None,
            active_worker: None,
            cancellation: self.cancellation.clone(),
        })
    }
}

fn joined_failure_v3(
    primary: String,
    retirement: Result<(), AdaptiveBootstrapError>,
    poll: Result<bool, ArenaError>,
) -> String {
    let mut detail = primary;
    if let Err(error) = retirement {
        detail.push_str(&format!("; exact root retirement: {error}"));
    }
    if let Err(error) = poll {
        detail.push_str(&format!("; native capture drain: {error}"));
    }
    detail
}

fn append_worker_failure_v3(failure: &mut Option<String>, detail: String) {
    match failure {
        Some(failure) => {
            failure.push_str("; ");
            failure.push_str(&detail);
        }
        None => *failure = Some(detail),
    }
}

impl GenerationSelfplayGameV3 for ActualGenerationSelfplayGameV3 {
    fn engine_names(&self) -> [String; 2] {
        self.names.clone()
    }
    fn process_epochs(&self) -> [ProcessEpoch; 2] {
        self.epochs
    }
    fn decide(
        &mut self,
        position: &Position,
        remaining: Duration,
        increment: Duration,
        hard_deadline: Instant,
        collector: &mut GenerationGameCaptureV3,
    ) -> Result<GenerationActualTurnV3, ArenaError> {
        let started = Instant::now();
        check(
            self.close_deadline.is_none()
                && self.failed.is_none()
                && self.active_worker.is_none()
                && !self
                    .cancellation
                    .as_ref()
                    .is_some_and(|signal| signal.load(Ordering::Acquire)),
            "actual game is closed, failed or canceled",
        )?;
        let anchor = exact_rules_anchor(position)?;
        check(
            self.decided_once || anchor == self.initial,
            "first actual decision differs from the assigned opening",
        )?;
        collector.poll()?;
        let budget = TimeBudget::new(
            started,
            TimeControl::Clock {
                remaining,
                increment,
                moves_to_go: None,
            },
            self.time_config,
        )
        .map_err(|e| invalid(format!("actual UCI clock budget: {e}")))?;
        let result_deadline = budget.hard_deadline.min(hard_deadline);
        let shutdown_deadline = budget.output_deadline.min(hard_deadline);
        check(
            started < result_deadline && result_deadline <= shutdown_deadline,
            "actual UCI decision has no live bounded result/drain window",
        )?;
        let index = usize::from((position.side_to_move() == Color::White) != self.engine_a_white);
        let process = &self.processes[index];
        let mut context = process
            .context(result_deadline, shutdown_deadline)
            .map_err(|e| invalid(format!("actual decision context: {e}")))?;
        if let Some(cancellation) = &self.cancellation {
            context.control.cancellation = Arc::clone(cancellation);
        }
        context.control.soft_deadline = budget.soft_deadline.min(result_deadline);
        context.control.admission_deadline = budget.admission_deadline.min(result_deadline);
        // Preserve the root's original cleanup ceiling before any retirement
        // attempt. Only a fully returned, accepted turn releases this ceiling
        // so its later whole-game close can use a separately supplied bound.
        self.close_deadline = Some(shutdown_deadline);
        {
            let mut slot = match self.slots[index].lock() {
                Ok(slot) => slot,
                Err(_) => {
                    let retirement = process.retire_context(&context);
                    let detail = joined_failure_v3(
                        "actual turn slot poisoned".into(),
                        retirement,
                        collector.poll(),
                    );
                    self.failed = Some(detail.clone());
                    return Err(invalid(detail));
                }
            };
            if slot.expected.is_some() || slot.completed.is_some() || slot.admitted {
                let retirement = process.retire_context(&context);
                let detail = joined_failure_v3(
                    "previous actual turn slot is unconsumed".into(),
                    retirement,
                    collector.poll(),
                );
                self.failed = Some(detail.clone());
                return Err(invalid(detail));
            }
            slot.expected = Some(ActualRootExpectationV3 {
                authority: context.authority,
                anchor,
            });
        }
        self.decided_once = true;
        let driver = process.driver();
        let worker_context = context.clone();
        let worker_position = position.clone();
        let handle = match std::thread::Builder::new()
            .name(format!(
                "pals-collection-search-{}",
                context.authority.epoch.0
            ))
            .spawn(move || {
                let result = driver.run(&worker_position, &worker_context, &mut |_| {});
                ActualDecisionFinishedV3 {
                    result,
                    finished: Instant::now(),
                }
            }) {
            Ok(handle) => handle,
            Err(error) => {
                let detail = joined_failure_v3(
                    format!("actual search owner spawn: {error}"),
                    process.retire_context(&context),
                    collector.poll(),
                );
                self.failed = Some(detail.clone());
                return Err(invalid(detail));
            }
        };
        self.active_worker = Some(ActualDecisionWorkerV3 {
            owner: index,
            context: context.clone(),
            handle,
        });
        let mut capture_failure = None;
        loop {
            if self
                .cancellation
                .as_ref()
                .is_some_and(|signal| signal.load(Ordering::Acquire))
            {
                capture_failure.get_or_insert_with(|| "original whole-game child canceled".into());
                context.cancellation.cancel();
                context.control.cancel();
            }
            if let Err(error) = collector.poll() {
                capture_failure.get_or_insert_with(|| error.to_string());
                context.cancellation.cancel();
                context.control.cancel();
            }
            if self
                .active_worker
                .as_ref()
                .expect("owned actual search")
                .handle
                .is_finished()
            {
                break;
            }
            if Instant::now() >= shutdown_deadline {
                context.cancellation.cancel();
                context.control.cancel();
                let detail = joined_failure_v3(
                    format!(
                        "actual search owner not joined before original root deadline; capture={capture_failure:?}"
                    ),
                    process.retire_context(&context),
                    collector.poll(),
                );
                // Keep the live join handle and prepared owner together. A
                // logical cancellation never grants a completed turn.
                self.failed = Some(detail.clone());
                return Err(invalid(detail));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let worker = self.active_worker.take().expect("finished actual search");
        let finished = worker
            .handle
            .join()
            .unwrap_or_else(|_| ActualDecisionFinishedV3 {
                result: Err(SearchSessionFailure::physical_completion_unknown()),
                finished: Instant::now(),
            });
        let retirement = process.retire_context(&context);
        let poll = collector.poll();
        let mut slot = match self.slots[index].lock() {
            Ok(slot) => slot,
            Err(_) => {
                let detail = joined_failure_v3(
                    "actual turn slot poisoned after root fence".into(),
                    retirement,
                    poll,
                );
                self.failed = Some(detail.clone());
                return Err(invalid(detail));
            }
        };
        let actual_turn = slot.completed.take();
        slot.expected = None;
        slot.admitted = false;
        let mut primary = capture_failure;
        if finished.finished > shutdown_deadline {
            append_worker_failure_v3(
                &mut primary,
                format!(
                    "actual Search owner returned after original root deadline: authority={:?}; finished={:?}; shutdown_deadline={shutdown_deadline:?}",
                    context.authority, finished.finished,
                ),
            );
        }
        match finished.result {
            Err(error) => append_worker_failure_v3(
                &mut primary,
                format!(
                    "actual native decision: {error}; physical_completion={:?}",
                    error.physical_completion
                ),
            ),
            Ok(ref report) => match actual_turn.as_ref() {
                None => append_worker_failure_v3(
                    &mut primary,
                    "actual manager closure produced no collection turn".into(),
                ),
                Some(turn) if turn.best_move() != report.best_move => append_worker_failure_v3(
                    &mut primary,
                    "actual turn/report bestmove differs".into(),
                ),
                Some(_) => {}
            },
        }
        if let Some(primary) = primary {
            let detail = joined_failure_v3(primary, retirement, poll);
            self.failed = Some(detail.clone());
            return Err(invalid(detail));
        }
        if retirement.is_err() || poll.is_err() {
            let detail = joined_failure_v3(
                "actual decision closure or capture failed".into(),
                retirement,
                poll,
            );
            self.failed = Some(detail.clone());
            return Err(invalid(detail));
        }
        self.close_deadline = None;
        Ok(actual_turn.expect("actual typed turn checked above"))
    }
    fn close(
        &mut self,
        until: Instant,
        collector: &mut GenerationGameCaptureV3,
    ) -> Result<GenerationActualGameCloseV3, ArenaError> {
        let original = self.close_deadline.map_or(until, |old| old.min(until));
        self.close_deadline = Some(original);
        let mut worker_failure = self.failed.clone();
        if let Some(worker) = &self.active_worker {
            worker.context.cancellation.cancel();
            worker.context.control.cancel();
        }
        let worker_until = self.active_worker.as_ref().map_or(original, |worker| {
            original.min(worker.context.shutdown_deadline)
        });
        while self
            .active_worker
            .as_ref()
            .is_some_and(|worker| !worker.handle.is_finished())
            && Instant::now() < worker_until
        {
            if let Err(error) = collector.poll() {
                worker_failure.get_or_insert_with(|| error.to_string());
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        if self
            .active_worker
            .as_ref()
            .is_some_and(|worker| worker.handle.is_finished())
        {
            let worker = self.active_worker.take().expect("finished actual search");
            match worker.handle.join() {
                Ok(finished) => {
                    if finished.finished > worker.context.shutdown_deadline {
                        append_worker_failure_v3(
                            &mut worker_failure,
                            format!(
                                "retained actual Search returned after original root deadline: authority={:?}; finished={:?}; shutdown_deadline={:?}",
                                worker.context.authority,
                                finished.finished,
                                worker.context.shutdown_deadline,
                            ),
                        );
                    }
                    if let Err(error) = finished.result {
                        append_worker_failure_v3(
                            &mut worker_failure,
                            format!(
                                "retained actual Search result: {error}; physical_completion={:?}",
                                error.physical_completion,
                            ),
                        );
                    }
                }
                Err(_) => append_worker_failure_v3(
                    &mut worker_failure,
                    "actual search owner panicked; physical_completion=Unknown".into(),
                ),
            }
            if let Err(error) = self.processes[worker.owner].retire_context(&worker.context) {
                append_worker_failure_v3(
                    &mut worker_failure,
                    format!("actual root retirement: {error}"),
                );
            }
        }
        // Always attempt both physical owners, even when the first fails.
        let a = self.processes[0].finish_process(original);
        let b = self.processes[1].finish_process(original);
        let poll = collector.poll();
        // This owned scope is safe to inspect even if Search still holds its
        // driver lock. Retention diagnostics must never wait for that lock.
        let retained_scope = self.active_worker.as_ref().map(|worker| {
            (
                worker.owner,
                worker.context.authority,
                worker.context.shutdown_deadline,
            )
        });
        match (a, b, poll, worker_failure, self.active_worker.is_some()) {
            (Ok(a), Ok(b), Ok(_), None, false) => {
                GenerationActualGameCloseV3::from_actual_session_receipts(&a, &b)
            }
            (a, b, poll, worker, unjoined) => Err(invalid(format!(
                "actual game physical close: engine A={a:?}; engine B={b:?}; capture={poll:?}; search={worker:?}; search_unjoined={unjoined}; retained_scope={retained_scope:?}"
            ))),
        }
    }
}

impl Drop for ActualGenerationSelfplayGameV3 {
    fn drop(&mut self) {
        let until = self.close_deadline.unwrap_or_else(|| {
            Instant::now()
                .checked_add(self.drain)
                .unwrap_or_else(Instant::now)
        });
        self.close_deadline = Some(until);
        if let Some(worker) = self.active_worker.take() {
            worker.context.cancellation.cancel();
            worker.context.control.cancel();
            if worker.handle.is_finished() {
                let _ = worker.handle.join();
                let _ = self.processes[worker.owner].retire_context(&worker.context);
            } else {
                // Preserve the concrete failed Search owner and exact scope,
                // rather than silently detaching an unobserved JoinHandle.
                let _quarantined_worker: &'static mut ActualDecisionWorkerV3 =
                    Box::leak(Box::new(worker));
            }
        }
        // Prepared owners retain/quarantine unknown physical work. Drop neither
        // fabricates game closure nor gives repeated cleanup a fresh deadline.
        let _a: Result<AdaptiveSessionProcessReceipt, _> = self.processes[0].finish_process(until);
        let _b: Result<AdaptiveSessionProcessReceipt, _> = self.processes[1].finish_process(until);
    }
}
