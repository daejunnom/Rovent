//! Owned continuation of the CPU traversal. Every charged node has an explicit
//! entry stage; resuming starts at the saved stage, never at iteration entry.
//! The engine retains its exclusive TT/history owner while this task is paused.

use super::*;
use crate::cpu_value::CpuAccumulatorUndo;
use rz_position::{PieceChange, RuleMoveDelta, UndoToken};

const MAX_FRAMES: usize = MAX_DEPTH as usize + MAX_QUIESCENCE_PLY as usize + 1;
const MAX_EXCHANGE_FRAMES: usize = 32;

pub(super) struct PausedTask {
    root: PositionIdentity,
    search_conditions: String,
    maximum_bytes: usize,
    value_identity: CpuValueIdentity,
    root_moves: Option<Vec<BoardMove>>,
    legal: Vec<BoardMove>,
    work: Position,
    value: Option<CpuAccumulator>,
    value_retained_bytes: usize,
    frames: Vec<Frame>,
    returned: Option<NodeValue>,
    best: Vec<BoardMove>,
    score: i32,
    completed_depth: u16,
    target_depth: u16,
    iteration_depth: u16,
    window: (i32, i32),
    full_window_retry: bool,
    cumulative: CpuWork,
    reciprocal: Option<ReciprocalTask>,
}

struct ReciprocalTask {
    scope: CpuReciprocalScope,
    hard_deadline: Option<Instant>,
    issued: u16,
    next_node: u64,
    pending: Option<CpuReciprocalRequest>,
}

impl PausedTask {
    fn new(
        engine: &CpuEngine,
        position: &Position,
        root_moves: Option<&[BoardMove]>,
        legal: Vec<BoardMove>,
        limits: CpuLimits,
        resume: Option<&CpuResumeToken>,
        capture: Option<&crate::cpu_value::nnuec::v2::capture::NnuecActiveCapture>,
    ) -> Result<Self, CpuError> {
        let value = engine
            .evaluator
            .initialize(position)
            .map_err(CpuError::Value)?;
        let score = if let Some(token) = resume {
            token.score
        } else {
            let score = match capture {
                Some(capture) => engine
                    .evaluator
                    .score_with_capture(&value, position, capture),
                None => engine.evaluator.score(&value, position),
            }
            .map_err(CpuError::Value)?;
            if !(-CPU_FRONTIER_SCORE_LIMIT..=CPU_FRONTIER_SCORE_LIMIT).contains(&score) {
                return Err(CpuError::Value(
                    CpuValueError::ScoreOutsideFrontierNamespace,
                ));
            }
            score
        };
        let value_retained_bytes = value.retained_bytes();
        let best = resume.map_or_else(
            || legal.first().copied().into_iter().collect(),
            |token| token.pv.clone(),
        );
        let mut frames = Vec::new();
        frames
            .try_reserve_exact(MAX_FRAMES)
            .map_err(|_| CpuError::Allocation)?;
        let task = Self {
            root: position.position_identity(),
            search_conditions: engine.search_conditions(),
            maximum_bytes: engine.paused_stack_max_bytes,
            value_identity: (*engine.value_identity).clone(),
            root_moves: root_moves.map(<[BoardMove]>::to_vec),
            legal,
            work: position.clone(),
            value: Some(value),
            value_retained_bytes,
            frames,
            returned: None,
            best,
            score,
            completed_depth: resume.map_or(0, |token| token.completed_depth),
            target_depth: limits.max_depth,
            iteration_depth: 0,
            window: (-INFINITY, INFINITY),
            full_window_retry: false,
            cumulative: CpuWork::default(),
            reciprocal: None,
        };
        task.check_memory()?;
        Ok(task)
    }

    /// Conservative retained traversal charge. Shared Rules prefixes count in
    /// each live exchange owner, allocator metadata is reserved per allocation,
    /// and the opaque accumulator reserves its fixed exact192 backing as well.
    /// The configured TT/history and frozen evaluator already belong to the
    /// engine; no copy of either is made or charged as a second paused task.
    pub(super) fn retained_bytes(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>();
        for charge in [
            vector_bytes(&self.frames),
            vector_bytes(&self.legal),
            vector_bytes(&self.best),
            self.root_moves.as_ref().map_or(0, vector_bytes),
            self.work
                .snapshot()
                .retained_history_bytes()
                .unwrap_or(usize::MAX),
            self.search_conditions.capacity(),
            self.value
                .as_ref()
                .map_or(self.value_retained_bytes, CpuAccumulator::retained_bytes),
            4096, // bounded value namespace/control/Arc and allocator allowance
        ] {
            bytes = bytes.saturating_add(charge);
        }
        for frame in &self.frames {
            bytes = bytes.saturating_add(frame.retained_bytes());
        }
        if let Some(value) = &self.returned {
            bytes = bytes.saturating_add(vector_bytes(&value.pv));
        }
        if let Some(reciprocal) = &self.reciprocal {
            bytes = bytes.saturating_add(std::mem::size_of::<ReciprocalTask>());
            if let Some(pending) = &reciprocal.pending {
                bytes = bytes
                    .saturating_add(vector_bytes(&pending.candidates))
                    .saturating_add(vector_bytes(&pending.relative_prefix))
                    .saturating_add(
                        pending
                            .position
                            .snapshot()
                            .retained_history_bytes()
                            .unwrap_or(usize::MAX),
                    );
            }
        }
        bytes
    }

    fn check_memory(&self) -> Result<(), CpuError> {
        let retained_bytes = self.retained_bytes();
        if retained_bytes > self.maximum_bytes {
            Err(CpuError::PausedStackMemoryLimit {
                retained_bytes,
                maximum_bytes: self.maximum_bytes,
            })
        } else {
            Ok(())
        }
    }

    fn start_iteration(&mut self, full_window: bool) -> Result<(), CpuError> {
        if !full_window {
            self.iteration_depth = self.completed_depth + 1;
            self.full_window_retry = false;
            self.window = if self.completed_depth == 0 || self.score.abs() >= MATE_THRESHOLD {
                (-INFINITY, INFINITY)
            } else {
                (
                    self.score - ASPIRATION_WINDOW,
                    self.score + ASPIRATION_WINDOW,
                )
            };
        } else {
            self.full_window_retry = true;
        }
        let window = if full_window {
            (-INFINITY, INFINITY)
        } else {
            self.window
        };
        let mut frame = Frame::new(Kind::Root, self.iteration_depth, 0, window.0, window.1);
        frame.moves = self.legal.clone();
        frame.previous_best = self.best.first().copied();
        self.frames.push(frame);
        self.check_memory()
    }

    fn request_feedback(
        &mut self,
        frame: &mut Frame,
        control: &Control<'_>,
    ) -> Result<bool, CpuError> {
        let Some(reciprocal) = self.reciprocal.as_mut() else {
            return Ok(false);
        };
        let total =
            self.cumulative
                .nodes
                .checked_add(control.nodes)
                .ok_or(CpuError::InvalidLimits(
                    "reciprocal node observation overflow",
                ))?;
        if reciprocal.issued >= reciprocal.scope.max_questions
            || total >= reciprocal.scope.job.total_native_node_credit
            || total < reciprocal.next_node
            || control.cancellation.load(Ordering::Acquire)
            || control.cooperative.is_some_and(|slice| {
                let state = slice.mailbox.snapshot();
                state.is_cancelled() || state.terminal
            })
            || control
                .limits
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
            || frame.kind == Kind::Quiescence
            || frame.reciprocal_queried
            || !matches!(frame.stage, Stage::Ordering(_))
            || frame.moves.is_empty()
        {
            return Ok(false);
        }
        // `frame` is the current popped node. Every retained ancestor must be
        // waiting on its real active Rules child. The one-use UndoToken's
        // checked before/after chain attests the job-relative path without
        // replaying the game, a PV, or another speculative continuation.
        if self.frames.len() != usize::from(frame.ply) || self.frames.len() >= MAX_FRAMES {
            return Err(CpuError::ResumeMismatch(
                "reciprocal ancestor depth mismatch",
            ));
        }
        let mut relative_prefix = Vec::new();
        relative_prefix
            .try_reserve_exact(self.frames.len())
            .map_err(|_| CpuError::Allocation)?;
        let mut endpoint = self.root.clone();
        for ancestor in &self.frames {
            let Stage::Await(active) = &ancestor.stage else {
                return Err(CpuError::ResumeMismatch(
                    "reciprocal ancestor is not an active Rules child",
                ));
            };
            let delta = active
                .undo
                .as_ref()
                .ok_or(CpuError::ResumeMismatch(
                    "reciprocal ancestor lost Rules authority",
                ))?
                .delta();
            if active.mv != delta.mv || delta.before.position_identity() != endpoint {
                return Err(CpuError::ResumeMismatch(
                    "reciprocal ancestor Rules chain changed",
                ));
            }
            relative_prefix.push(delta.mv);
            endpoint = delta.after.position_identity();
        }
        if endpoint != self.work.position_identity() {
            return Err(CpuError::ResumeMismatch(
                "reciprocal prefix does not end at the saved node",
            ));
        }
        let ordering = std::mem::replace(&mut frame.stage, Stage::Finish);
        let Stage::Ordering(ordering) = ordering else {
            unreachable!("checked node ordering");
        };
        frame.stage = Stage::AwaitReciprocal(ordering);
        frame.reciprocal_queried = true;
        reciprocal.issued += 1;
        reciprocal.next_node = total.saturating_add(reciprocal.scope.node_interval);
        reciprocal.pending = Some(CpuReciprocalRequest {
            seal: Arc::new(()),
            scope: reciprocal.scope.clone(),
            position: self.work.clone(),
            relative_prefix,
            ply: frame.ply,
            remaining_depth: frame.depth,
            ordinal: reciprocal.issued,
            candidates: frame.moves.clone(),
        });
        self.check_memory()?;
        Ok(true)
    }
}

fn vector_bytes<T>(values: &Vec<T>) -> usize {
    values
        .capacity()
        .saturating_mul(std::mem::size_of::<T>())
        .saturating_add(32)
}

fn undo_bytes(undo: &UndoToken) -> usize {
    vector_bytes(&undo.delta().removals) + vector_bytes(&undo.delta().additions) + 32
}

pub(super) fn token_is_current(engine: &CpuEngine, token: &CpuResumeToken) -> bool {
    let Some(handle) = token.paused_stack.as_ref() else {
        return false;
    };
    let Some(task) = engine.paused.as_ref() else {
        return false;
    };
    let retained_bytes = task.retained_bytes();
    engine.resume_policy == CpuResumePolicy::PausedStack
        && Arc::ptr_eq(&handle.owner, &engine.stack_owner)
        && handle.serial == engine.stack_serial
        && task.root == token.root
        && task.root_moves == token.root_moves
        && task.completed_depth == token.completed_depth
        && task.best == token.pv
        && task.score == token.score
        && task.cumulative == handle.cumulative
        && retained_bytes == handle.retained_bytes
        && task.maximum_bytes == engine.paused_stack_max_bytes
        && retained_bytes <= task.maximum_bytes
        && task.search_conditions == token.search_conditions
        && token.search_conditions == engine.search_conditions()
        && task.value_identity == token.value_identity
        && token.value_identity == *engine.value_identity
        && task
            .value
            .as_ref()
            .is_some_and(|value| engine.evaluator.cache_context(value, &task.work).is_ok())
        && engine.evaluator.identity() == engine.value_identity.as_ref()
        && token.profile == engine.config.profile
        && token.quiescence_ply == engine.config.quiescence_ply
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    engine: &mut CpuEngine,
    position: &Position,
    root_moves: Option<&[BoardMove]>,
    limits: CpuLimits,
    cancel: &AtomicBool,
    resume: Option<&CpuResumeToken>,
    observer: &mut dyn FnMut(CpuIterationProgress),
) -> Result<CpuReport, CpuError> {
    run_inner(
        engine, position, root_moves, limits, cancel, resume, observer, None, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_controlled(
    engine: &mut CpuEngine,
    position: &Position,
    root_moves: Option<&[BoardMove]>,
    limits: CpuLimits,
    cancel: &AtomicBool,
    resume: Option<&CpuResumeToken>,
    observer: &mut dyn FnMut(CpuIterationProgress),
    slice: &CpuSliceControl,
) -> Result<CpuReport, CpuError> {
    run_inner(
        engine,
        position,
        root_moves,
        limits,
        cancel,
        resume,
        observer,
        Some(slice),
        None,
    )
}

pub(super) fn pending_reciprocal_request(engine: &CpuEngine) -> Option<CpuReciprocalRequest> {
    engine.paused.as_ref()?.reciprocal.as_ref()?.pending.clone()
}

pub(super) fn validate_reciprocal_resume(
    engine: &CpuEngine,
    token: Option<&CpuResumeToken>,
    scope: &CpuReciprocalScope,
    reply: Option<&CpuReciprocalReply>,
) -> Result<(), CpuError> {
    if let Some(token) = token {
        if !token_is_current(engine, token) {
            return Err(CpuError::ResumeMismatch(
                "reciprocal continuation is stale or foreign",
            ));
        }
        let task = engine.paused.as_ref().expect("current token retains task");
        let reciprocal = task.reciprocal.as_ref().ok_or(CpuError::ResumeMismatch(
            "ordinary paused task has no reciprocal scope",
        ))?;
        if reciprocal.scope != *scope {
            return Err(CpuError::ResumeMismatch(
                "reciprocal job, owner or frozen context changed",
            ));
        }
        match (&reciprocal.pending, reply) {
            (Some(request), Some(reply)) if request.same_request(&reply.request) => {}
            (None, None) => {}
            _ => {
                return Err(CpuError::ResumeMismatch(
                    "reciprocal reply is missing, duplicate or foreign",
                ));
            }
        }
    } else if reply.is_some() || engine.paused.is_some() {
        return Err(CpuError::ResumeMismatch(
            "fresh reciprocal job cannot replace a retained owner",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_reciprocal(
    engine: &mut CpuEngine,
    position: &Position,
    root_moves: Option<&[BoardMove]>,
    limits: CpuLimits,
    cancel: &AtomicBool,
    resume: Option<&CpuResumeToken>,
    observer: &mut dyn FnMut(CpuIterationProgress),
    slice: &CpuSliceControl,
    scope: &CpuReciprocalScope,
    reply: Option<&CpuReciprocalReply>,
) -> Result<CpuReport, CpuError> {
    run_inner(
        engine,
        position,
        root_moves,
        limits,
        cancel,
        resume,
        observer,
        Some(slice),
        Some((scope, reply)),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_inner(
    engine: &mut CpuEngine,
    position: &Position,
    root_moves: Option<&[BoardMove]>,
    limits: CpuLimits,
    cancel: &AtomicBool,
    resume: Option<&CpuResumeToken>,
    observer: &mut dyn FnMut(CpuIterationProgress),
    cooperative: Option<&CpuSliceControl>,
    reciprocal: Option<(&CpuReciprocalScope, Option<&CpuReciprocalReply>)>,
) -> Result<CpuReport, CpuError> {
    engine.paused_stack_limit_sealed = true;
    engine.last_attempt_work = Some(CpuWork::default());
    if resume.is_none() {
        engine.invalidate_paused_stack();
    }
    if let Err(error) = engine.validate_limits(position, limits) {
        if matches!(error, CpuError::InvalidConfig(_)) {
            engine.observe_paused_stack_rejection(resume);
        }
        if matches!(error, CpuError::InvalidConfig(_)) || cancel.load(Ordering::Acquire) {
            engine.invalidate_paused_stack();
        }
        return Err(error);
    }
    if engine.paused.as_ref().is_some_and(|task| {
        task.search_conditions != engine.search_conditions()
            || task.maximum_bytes != engine.paused_stack_max_bytes
            || task.value_identity != *engine.value_identity
            || task.value.is_none()
    }) {
        engine.invalidate_paused_stack();
        engine.observe_paused_stack_rejection(resume);
        return Err(CpuError::ResumeMismatch("retained CPU namespace changed"));
    }
    if let Some(error) = engine.paused.as_ref().and_then(|task| {
        task.value
            .as_ref()
            .and_then(|value| engine.evaluator.cache_context(value, &task.work).err())
    }) {
        // Keep the evaluator's exact typed cause. This preflight has not
        // accepted a new score or work, so it cannot manufacture CpuReport.
        engine.invalidate_paused_stack();
        engine.observe_paused_stack_rejection(resume);
        return Err(CpuError::Value(error));
    }
    let started = Instant::now();
    if let Some(token) = resume {
        if engine
            .paused
            .as_ref()
            .is_some_and(|task| task.reciprocal.is_some())
            && reciprocal.is_none()
        {
            if cancel.load(Ordering::Acquire) {
                engine.invalidate_paused_stack();
            }
            return Err(CpuError::ResumeMismatch(
                "reciprocal task requires its sealed response path",
            ));
        }
        if token.root != position.position_identity() {
            if cancel.load(Ordering::Acquire) {
                engine.invalidate_paused_stack();
            }
            engine.observe_paused_stack_rejection(resume);
            return Err(CpuError::ResumeMismatch(
                "Rules state or full known history changed",
            ));
        }
        if token.profile != engine.config.profile
            || token.quiescence_ply != engine.config.quiescence_ply
            || token.value_identity != *engine.value_identity
            || token.search_conditions != engine.search_conditions()
            || limits.max_depth < token.completed_depth
        {
            if cancel.load(Ordering::Acquire) {
                engine.invalidate_paused_stack();
            }
            engine.observe_paused_stack_rejection(resume);
            return Err(CpuError::ResumeMismatch(
                "CPU search conditions or value namespace changed",
            ));
        }
    }
    let mut task = if let Some(handle) = resume.and_then(|token| token.paused_stack.as_ref()) {
        if !Arc::ptr_eq(&handle.owner, &engine.stack_owner) || handle.serial != engine.stack_serial
        {
            if cancel.load(Ordering::Acquire) {
                engine.invalidate_paused_stack();
            }
            engine.observe_paused_stack_rejection(resume);
            return Err(CpuError::ResumeMismatch(
                "paused stack owner or one-use serial is stale",
            ));
        }
        let Some(retained) = engine.paused.as_ref() else {
            engine.observe_paused_stack_rejection(resume);
            return Err(CpuError::ResumeMismatch(
                "paused stack was consumed or invalidated",
            ));
        };
        let token = resume.expect("handle came from token");
        if let Some((scope, _)) = reciprocal {
            if retained.reciprocal.as_ref().is_some_and(|task| {
                task.scope != *scope
                    || task.hard_deadline != limits.deadline
                    || retained
                        .cumulative
                        .nodes
                        .checked_add(limits.max_nodes)
                        .is_none_or(|total| total > scope.job.total_native_node_credit)
            }) {
                return Err(CpuError::ResumeMismatch(
                    "reciprocal original deadline or scope changed",
                ));
            }
        }
        if retained.root != token.root
            || retained.root_moves != token.root_moves
            || retained.completed_depth != token.completed_depth
            || retained.best != token.pv
            || retained.score != token.score
            || retained.cumulative != handle.cumulative
            || retained.retained_bytes() != handle.retained_bytes
            || retained.target_depth != limits.max_depth
        {
            if cancel.load(Ordering::Acquire) {
                engine.invalidate_paused_stack();
            }
            engine.observe_paused_stack_rejection(resume);
            return Err(CpuError::ResumeMismatch(
                "paused task metadata, restrictions or target depth changed",
            ));
        }
        let task = engine.paused.take().expect("validated live task");
        engine.observe_paused_stack_resumed();
        task
    } else {
        // Starting a new traversal from a historical completed result also
        // invalidates any unrelated paused owner.
        engine.invalidate_paused_stack();
        let view = position.ordered_legal_moves();
        let status = position.play_status_from_view(&view)?;
        let mut legal = view.moves().to_vec();
        if let Some(requested) = root_moves {
            if requested.is_empty() || requested.len() > MAX_ROOT_MOVES {
                return Err(CpuError::InvalidRootMoves(
                    "root restriction must contain 1..=256 moves",
                ));
            }
            for (index, mv) in requested.iter().enumerate() {
                if !legal.contains(mv) || requested[..index].contains(mv) {
                    return Err(CpuError::InvalidRootMoves(
                        "root restriction contains illegal or duplicate moves",
                    ));
                }
            }
            legal.retain(|mv| requested.contains(mv));
        }
        if let PlayStatus::Terminal { reason, winner } = status {
            return Ok(CpuReport {
                best_move: None,
                pv: Vec::new(),
                score: terminal_score(position, winner, 0),
                completed_depth: 0,
                nodes: 0,
                quiescence_nodes: 0,
                tt_hits: 0,
                completion: CpuCompletion::Terminal(reason),
                score_scope: CpuScoreScope::RulesTerminal,
                profile: engine.config.profile,
                score_provenance: "rz-position/rules-terminal",
                value_identity: (*engine.value_identity).clone(),
                search_version: engine.search_identity(),
                root_restricted: root_moves.is_some(),
                elapsed: started.elapsed(),
                reused_completed_depth: 0,
                resume: None,
            });
        }
        PausedTask::new(
            engine,
            position,
            root_moves,
            legal,
            limits,
            resume,
            cooperative.and_then(|slice| slice.nnuec_capture.as_ref()),
        )?
    };
    if let Some((scope, reply)) = reciprocal {
        if task.reciprocal.is_none() {
            if limits.deadline.is_none() || limits.max_nodes > scope.job.total_native_node_credit {
                return Err(CpuError::InvalidLimits(
                    "reciprocal execution requires original bounded job",
                ));
            }
            task.reciprocal = Some(ReciprocalTask {
                scope: scope.clone(),
                hard_deadline: limits.deadline,
                issued: 0,
                next_node: scope.node_interval,
                pending: None,
            });
        }
        let live = !cancel.load(Ordering::Acquire)
            && !cooperative.is_some_and(|slice| slice.mailbox.snapshot().is_cancelled())
            && limits
                .deadline
                .is_none_or(|deadline| Instant::now() < deadline);
        if let Some(reply) = reply.filter(|_| live) {
            let pending = task
                .reciprocal
                .as_mut()
                .expect("scope installed")
                .pending
                .take()
                .expect("read-only admission checked pending reply");
            if !Arc::ptr_eq(&pending.seal, &reply.request.seal) {
                return Err(CpuError::ResumeMismatch(
                    "reciprocal request authority changed",
                ));
            }
            let frame = task.frames.last_mut().ok_or(CpuError::ResumeMismatch(
                "reciprocal response has no saved node",
            ))?;
            let old = std::mem::replace(&mut frame.stage, Stage::Finish);
            let Stage::AwaitReciprocal(ordering) = old else {
                return Err(CpuError::ResumeMismatch(
                    "reciprocal response has no awaiting stage",
                ));
            };
            frame.feedback_priority = reply.preferred.clone();
            frame.stage = Stage::Ordering(ordering);
        }
    }
    if let Err(error) = task.check_memory() {
        engine.invalidate_paused_stack();
        return Err(error);
    }
    let reused_completed_depth = task.completed_depth;
    let mut control = Control {
        limits,
        cancellation: cancel,
        cooperative,
        nodes: 0,
        quiescence_nodes: 0,
        tt_hits: 0,
        value: task.value.take().expect("paused task owns its accumulator"),
    };
    if resume.is_some_and(CpuResumeToken::is_paused_stack) {
        // Observe actual invocation counters before any new work. The saved
        // task.cumulative diagnostics never initialize these counters.
        engine.observe_paused_stack_initial_work(control.actual_work());
    }
    if cooperative.is_none() || task.completed_depth > 0 {
        observer(CpuIterationProgress {
            best_move: task.best.first().copied(),
            score: task.score,
            completed_depth: task.completed_depth,
            nodes: 0,
        });
    }
    let result = drive(engine, &mut task, &mut control, observer);
    let work = control.actual_work();
    task.value = Some(control.value);
    engine.last_attempt_work = Some(work);
    let cumulative =
        (|| {
            Ok(CpuWork {
                nodes: task.cumulative.nodes.checked_add(work.nodes).ok_or(
                    CpuError::InvalidLimits("cumulative paused CPU nodes exceed u64"),
                )?,
                quiescence_nodes: task
                    .cumulative
                    .quiescence_nodes
                    .checked_add(work.quiescence_nodes)
                    .ok_or(CpuError::InvalidLimits(
                        "cumulative paused CPU qnodes exceed u64",
                    ))?,
                tt_hits: task.cumulative.tt_hits.checked_add(work.tt_hits).ok_or(
                    CpuError::InvalidLimits("cumulative paused CPU TT hits exceed u64"),
                )?,
            })
        })();
    task.cumulative = match cumulative {
        Ok(cumulative) => cumulative,
        Err(error) => {
            engine.invalidate_paused_stack();
            return Err(error);
        }
    };
    let (completion, value_authority_stopped) = match result {
        Ok(()) => (CpuCompletion::DepthLimit, false),
        Err(Abort::Stop(reason)) => (reason, false),
        Err(Abort::ValueAuthorityStop(reason)) => (reason, true),
        Err(Abort::Error(error)) => {
            engine.invalidate_paused_stack();
            return Err(error);
        }
    };
    let can_pause = !value_authority_stopped
        && if cooperative.is_some() {
            matches!(
                completion,
                CpuCompletion::Yielded | CpuCompletion::AwaitReciprocal
            )
        } else {
            matches!(
                completion,
                CpuCompletion::NodeLimit | CpuCompletion::Deadline
            )
        };
    let mut token = None;
    if can_pause {
        if let Err(error) = task.check_memory() {
            engine.invalidate_paused_stack();
            return Err(error);
        }
        engine.stack_serial = engine
            .stack_serial
            .checked_add(1)
            .ok_or(CpuError::Unsupported(
                "paused stack one-use serial exhausted",
            ))?;
        token = Some(make_token(
            engine,
            &task,
            Some(PausedStackToken {
                owner: Arc::clone(&engine.stack_owner),
                serial: engine.stack_serial,
                cumulative: task.cumulative,
                retained_bytes: task.retained_bytes(),
            }),
        ));
    } else if completion == CpuCompletion::DepthLimit && task.completed_depth > 0 {
        token = Some(make_token(engine, &task, None));
    }
    let report = CpuReport {
        best_move: task.best.first().copied(),
        pv: task.best.clone(),
        score: task.score,
        completed_depth: task.completed_depth,
        nodes: work.nodes,
        quiescence_nodes: work.quiescence_nodes,
        tt_hits: work.tt_hits,
        completion,
        score_scope: if task.completed_depth == 0 {
            CpuScoreScope::FrontierOnly
        } else {
            CpuScoreScope::CompletedIteration
        },
        profile: engine.config.profile,
        score_provenance: engine.evaluator.provenance(),
        value_identity: (*engine.value_identity).clone(),
        search_version: engine.search_identity(),
        root_restricted: task.root_moves.is_some(),
        elapsed: started.elapsed(),
        reused_completed_depth,
        resume: token,
    };
    if can_pause {
        engine.paused = Some(task);
        engine.observe_paused_stack_created();
    } else {
        engine.invalidate_paused_stack();
    }
    Ok(report)
}

fn make_token(
    engine: &CpuEngine,
    task: &PausedTask,
    paused_stack: Option<PausedStackToken>,
) -> CpuResumeToken {
    CpuResumeToken {
        root: task.root.clone(),
        root_moves: task.root_moves.clone(),
        profile: engine.config.profile,
        quiescence_ply: engine.config.quiescence_ply,
        value_identity: (*engine.value_identity).clone(),
        search_conditions: engine.search_conditions(),
        completed_depth: task.completed_depth,
        score: task.score,
        pv: task.best.clone(),
        paused_stack,
    }
}

fn drive(
    engine: &mut CpuEngine,
    task: &mut PausedTask,
    control: &mut Control<'_>,
    observer: &mut dyn FnMut(CpuIterationProgress),
) -> Result<(), Abort> {
    loop {
        // A coherent completed iteration is the only publishable score. A
        // control request never exposes frame alpha or an unfinished child.
        if let Some(slice) = control.cooperative {
            if let Some(sequence) = slice
                .mailbox
                .snapshot()
                .pending_publish()
                .filter(|_| task.completed_depth > 0)
            {
                observer(CpuIterationProgress {
                    best_move: task.best.first().copied(),
                    score: task.score,
                    completed_depth: task.completed_depth,
                    nodes: control.nodes,
                });
                slice
                    .mailbox
                    .acknowledge_publish_progress(sequence)
                    .map_err(|_| {
                        Abort::Error(CpuError::InvalidLimits(
                            "CPU progress acknowledgement failed",
                        ))
                    })?;
            }
            if task.frames.is_empty()
                && task.returned.is_none()
                && task.completed_depth >= task.target_depth
            {
                let controls = slice.mailbox.snapshot();
                if control.cancellation.load(Ordering::Acquire)
                    || controls.is_cancelled()
                    || controls.terminal
                {
                    return Err(Abort::Stop(CpuCompletion::Canceled));
                }
                if control
                    .limits
                    .deadline
                    .is_some_and(|deadline| Instant::now() >= deadline)
                {
                    return Err(Abort::Stop(CpuCompletion::Deadline));
                }
                return Ok(());
            }
        }
        control.check_stop()?;
        if task.frames.is_empty() {
            if let Some(value) = task.returned.take() {
                if !task.full_window_retry
                    && (value.score <= task.window.0 || value.score >= task.window.1)
                {
                    task.start_iteration(true).map_err(Abort::Error)?;
                    continue;
                }
                if task.work.position_identity() != task.root {
                    return Err(Abort::Error(CpuError::ResumeMismatch(
                        "completed traversal did not restore Rules root",
                    )));
                }
                task.score = value.score;
                task.best = value.pv;
                task.completed_depth = task.iteration_depth;
                observer(CpuIterationProgress {
                    best_move: task.best.first().copied(),
                    score: task.score,
                    completed_depth: task.completed_depth,
                    nodes: control.nodes,
                });
                continue;
            }
            if task.completed_depth >= task.target_depth {
                return Ok(());
            }
            task.start_iteration(false).map_err(Abort::Error)?;
        }
        let mut frame = task.frames.pop().expect("iteration has a frame");
        let action = frame.advance(
            engine,
            &mut task.work,
            control,
            &mut task.returned,
            &task.legal,
        );
        task.value_retained_bytes = control.value.retained_bytes();
        match action {
            Ok(Action::Continue) => {
                let awaiting = task
                    .request_feedback(&mut frame, control)
                    .map_err(Abort::Error)?;
                task.frames.push(frame);
                if awaiting {
                    return Err(Abort::Stop(CpuCompletion::AwaitReciprocal));
                }
            }
            Ok(Action::Push(child)) => {
                task.frames.push(frame);
                if task.frames.len() >= MAX_FRAMES {
                    return Err(Abort::Error(CpuError::Unsupported(
                        "paused search frame bound exhausted",
                    )));
                }
                task.frames.push(child);
            }
            Ok(Action::Return(value)) => task.returned = Some(value),
            Err(error) => {
                task.frames.push(frame);
                return Err(error);
            }
        }
        task.check_memory().map_err(Abort::Error)?;
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Kind {
    Root,
    Negamax,
    Quiescence,
}

enum Stage {
    Enter,
    Ordering(OrderingState),
    AwaitReciprocal(OrderingState),
    Select,
    Await(ActiveMove),
    Finish,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ChildPhase {
    Scout,
    Full,
    RootTieConfirm,
    Quiescence,
}

struct ActiveMove {
    mv: BoardMove,
    undo: Option<UndoToken>,
    value_undo: Option<CpuAccumulatorUndo>,
    phase: ChildPhase,
    quiet: bool,
    side: usize,
}

struct Frame {
    kind: Kind,
    depth: u16,
    ply: u16,
    alpha: i32,
    beta: i32,
    alpha_original: i32,
    previous_best: Option<BoardMove>,
    moves: Vec<BoardMove>,
    cursor: usize,
    best: NodeValue,
    tt: Option<(u64, PositionIdentity, CpuValueCacheContext)>,
    stage: Stage,
    reciprocal_queried: bool,
    feedback_priority: Vec<BoardMove>,
}

// A transient result in the nonrecursive driver. Keeping the child inline
// avoids an allocation per searched edge; retained frames live in the one
// fallibly reserved and byte-charged task Vec.
#[allow(clippy::large_enum_variant)]
enum Action {
    Continue,
    Push(Frame),
    Return(NodeValue),
}

impl Frame {
    fn new(kind: Kind, depth: u16, ply: u16, alpha: i32, beta: i32) -> Self {
        Self {
            kind,
            depth,
            ply,
            alpha,
            beta,
            alpha_original: alpha,
            previous_best: None,
            moves: Vec::new(),
            cursor: 0,
            best: NodeValue {
                score: -INFINITY,
                pv: Vec::new(),
            },
            tt: None,
            stage: Stage::Enter,
            reciprocal_queried: false,
            feedback_priority: Vec::new(),
        }
    }

    fn child(engine: &CpuEngine, depth: u16, ply: u16, alpha: i32, beta: i32) -> Self {
        if depth == 0 {
            Self::new(
                Kind::Quiescence,
                engine.config.quiescence_ply,
                ply,
                alpha,
                beta,
            )
        } else {
            Self::new(Kind::Negamax, depth, ply, alpha, beta)
        }
    }

    fn retained_bytes(&self) -> usize {
        let own = vector_bytes(&self.moves)
            .saturating_add(vector_bytes(&self.best.pv))
            .saturating_add(vector_bytes(&self.feedback_priority));
        own.saturating_add(match &self.stage {
            Stage::Ordering(ordering) | Stage::AwaitReciprocal(ordering) => {
                ordering.retained_bytes()
            }
            Stage::Await(active) => active.undo.as_ref().map_or(0, undo_bytes).saturating_add(
                active
                    .value_undo
                    .as_ref()
                    .map_or(0, CpuAccumulatorUndo::retained_bytes),
            ),
            _ => 0,
        })
    }

    fn advance(
        &mut self,
        engine: &mut CpuEngine,
        position: &mut Position,
        control: &mut Control<'_>,
        returned: &mut Option<NodeValue>,
        root_legal: &[BoardMove],
    ) -> Result<Action, Abort> {
        match &mut self.stage {
            Stage::Enter => self.enter(engine, position, control),
            Stage::Ordering(ordering) => {
                if let Some(mut moves) =
                    ordering.advance(engine, position, &self.moves, self.previous_best, control)?
                {
                    // Stable ordering preserves the declared native policy for
                    // candidates omitted by P/C, without treating logits as a bound.
                    if !self.feedback_priority.is_empty() {
                        moves.sort_by_key(|movement| {
                            self.feedback_priority
                                .iter()
                                .position(|candidate| candidate == movement)
                                .unwrap_or(usize::MAX)
                        });
                    }
                    self.moves = moves;
                    self.stage = Stage::Select;
                }
                Ok(Action::Continue)
            }
            Stage::AwaitReciprocal(_) => Err(Abort::Error(CpuError::ResumeMismatch(
                "awaiting reciprocal node cannot execute without a sealed reply",
            ))),
            Stage::Select => {
                if self.cursor >= self.moves.len() {
                    self.stage = Stage::Finish;
                    return Ok(Action::Continue);
                }
                let mv = self.moves[self.cursor];
                let quiet = !is_tactical(position, mv);
                let side = position.side_to_move() as usize;
                let undo = position.make_move(mv)?;
                let value_undo = match engine
                    .evaluator
                    .apply_delta(&mut control.value, undo.delta())
                {
                    Ok(value) => value,
                    Err(error) => {
                        position.unmake(undo)?;
                        return Err(error.into());
                    }
                };
                let phase = if self.kind == Kind::Quiescence {
                    ChildPhase::Quiescence
                } else if self.cursor == 0 {
                    ChildPhase::Full
                } else {
                    ChildPhase::Scout
                };
                let child = if self.kind == Kind::Quiescence {
                    Frame::new(
                        Kind::Quiescence,
                        self.depth - 1,
                        self.ply + 1,
                        -self.beta,
                        -self.alpha,
                    )
                } else {
                    let (alpha, beta) = if phase == ChildPhase::Full {
                        (-self.beta, -self.alpha)
                    } else {
                        (-self.alpha - 1, -self.alpha)
                    };
                    Frame::child(engine, self.depth - 1, self.ply + 1, alpha, beta)
                };
                self.cursor += 1;
                self.stage = Stage::Await(ActiveMove {
                    mv,
                    undo: Some(undo),
                    value_undo: Some(value_undo),
                    phase,
                    quiet,
                    side,
                });
                Ok(Action::Push(child))
            }
            Stage::Await(active) => {
                let child = returned
                    .take()
                    .ok_or(Abort::Error(CpuError::ResumeMismatch(
                        "missing completed child frame",
                    )))?;
                let score = -child.score;
                if active.phase == ChildPhase::Scout && score > self.alpha && score < self.beta {
                    active.phase = ChildPhase::Full;
                    return Ok(Action::Push(Frame::child(
                        engine,
                        self.depth - 1,
                        self.ply + 1,
                        -self.beta,
                        -self.alpha,
                    )));
                }
                // The scout value at alpha can be an upper bound, not an
                // exact tie. Only re-search an earlier Rules root candidate;
                // keep the same move/undo and paused owner through that work.
                if engine.root_tie_policy == CpuRootTiePolicy::RulesOrderConfirmed
                    && self.kind == Kind::Root
                    && active.phase == ChildPhase::Scout
                    && score == self.best.score
                    && self.alpha == self.best.score
                    && self.alpha > -INFINITY
                    && root_move_precedes(root_legal, active.mv, self.best.pv.first().copied())
                {
                    active.phase = ChildPhase::RootTieConfirm;
                    return Ok(Action::Push(Frame::child(
                        engine,
                        self.depth - 1,
                        self.ply + 1,
                        -self.beta,
                        1 - self.alpha,
                    )));
                }
                let mv = active.mv;
                position.unmake(active.undo.take().expect("active move owns Rules undo"))?;
                engine.evaluator.restore(
                    &mut control.value,
                    active
                        .value_undo
                        .take()
                        .expect("active move owns value undo"),
                    position,
                )?;
                if score > self.best.score
                    || (active.phase == ChildPhase::RootTieConfirm
                        && score == self.best.score
                        && root_move_precedes(root_legal, mv, self.best.pv.first().copied()))
                {
                    self.best.score = score;
                    self.best.pv.clear();
                    self.best.pv.push(mv);
                    self.best.pv.extend(child.pv);
                }
                self.alpha = self.alpha.max(score);
                if self.alpha >= self.beta {
                    if self.kind == Kind::Negamax && active.quiet {
                        let history = &mut engine.history[active.side][mv.from.index() as usize]
                            [mv.to.index() as usize];
                        *history = history
                            .saturating_add(i32::from(self.depth).pow(2))
                            .min(20_000);
                    }
                    self.stage = Stage::Finish;
                } else {
                    self.stage = Stage::Select;
                }
                Ok(Action::Continue)
            }
            Stage::Finish => {
                control.check_stop()?;
                if let Some((hash, identity, value_context)) = self.tt.take() {
                    let bound = if self.best.score <= self.alpha_original {
                        Bound::Upper
                    } else if self.best.score >= self.beta {
                        Bound::Lower
                    } else {
                        Bound::Exact
                    };
                    engine.store(TtEntry {
                        hash,
                        identity,
                        profile: engine.config.profile,
                        value_identity: Arc::clone(&engine.value_identity),
                        value_context,
                        depth: self.depth,
                        score: to_tt_score(self.best.score, self.ply),
                        bound,
                        best_move: self.best.pv.first().copied(),
                    });
                }
                Ok(Action::Return(std::mem::replace(
                    &mut self.best,
                    NodeValue {
                        score: -INFINITY,
                        pv: Vec::new(),
                    },
                )))
            }
        }
    }

    fn enter(
        &mut self,
        engine: &mut CpuEngine,
        position: &Position,
        control: &mut Control<'_>,
    ) -> Result<Action, Abort> {
        control.visit(self.kind == Kind::Quiescence)?;
        if self.kind != Kind::Root {
            let view = position.ordered_legal_moves();
            if let PlayStatus::Terminal { winner, .. } = position.play_status_from_view(&view)? {
                return Ok(Action::Return(NodeValue {
                    score: terminal_score(position, winner, self.ply),
                    pv: Vec::new(),
                }));
            }
            self.moves = view.moves().to_vec();
            if self.kind == Kind::Negamax {
                let hash = tt_hash(position);
                let identity = position.position_identity();
                let value_context = engine.evaluator.cache_context(&control.value, position)?;
                if let Some(entry) = engine.probe(hash, &identity, &value_context) {
                    control.tt_hits += 1;
                    let score = from_tt_score(entry.score, self.ply);
                    if entry.depth == self.depth
                        && (entry.bound == Bound::Exact
                            || (entry.bound == Bound::Lower && score >= self.beta)
                            || (entry.bound == Bound::Upper && score <= self.alpha))
                    {
                        return Ok(Action::Return(NodeValue {
                            score,
                            pv: entry.best_move.into_iter().collect(),
                        }));
                    }
                    self.previous_best = entry.best_move;
                }
                self.tt = Some((hash, identity, value_context));
            } else {
                let in_check = position.in_check();
                if in_check && self.depth == 0 {
                    return Err(Abort::Stop(CpuCompletion::QuiescenceLimit));
                }
                if !in_check {
                    self.best.score = engine.frontier_score(control, position)?;
                    if self.best.score >= self.beta || self.depth == 0 {
                        return Ok(Action::Return(std::mem::replace(
                            &mut self.best,
                            NodeValue {
                                score: -INFINITY,
                                pv: Vec::new(),
                            },
                        )));
                    }
                    self.alpha = self.alpha.max(self.best.score);
                    self.moves.retain(|mv| is_tactical(position, *mv));
                }
            }
        }
        if self.moves.len() > MAX_ROOT_MOVES {
            return Err(Abort::Error(CpuError::Unsupported(
                "Rules move count exceeds bounded paused frame",
            )));
        }
        self.stage = Stage::Ordering(OrderingState::new(self.moves.len())?);
        Ok(Action::Continue)
    }
}

struct OrderingState {
    scored: Vec<(BoardMove, (u8, i32))>,
    cursor: usize,
    exchange_positions: u32,
    exchange: Option<Exchange>,
}

impl OrderingState {
    fn new(moves: usize) -> Result<Self, Abort> {
        let mut scored = Vec::new();
        scored
            .try_reserve_exact(moves)
            .map_err(|_| Abort::Error(CpuError::Allocation))?;
        Ok(Self {
            scored,
            cursor: 0,
            exchange_positions: 0,
            exchange: None,
        })
    }

    fn retained_bytes(&self) -> usize {
        vector_bytes(&self.scored)
            .saturating_add(self.exchange.as_ref().map_or(0, Exchange::retained_bytes))
    }

    fn advance(
        &mut self,
        engine: &CpuEngine,
        position: &Position,
        moves: &[BoardMove],
        best: Option<BoardMove>,
        control: &mut Control<'_>,
    ) -> Result<Option<Vec<BoardMove>>, Abort> {
        if engine.ordering == CpuOrderingPolicy::LegacyMvvLvaV1 {
            let mut ordered = moves.to_vec();
            ordered.sort_by_key(|mv| std::cmp::Reverse(engine.order_score(position, *mv, best)));
            return Ok(Some(ordered));
        }
        if self.cursor == moves.len() {
            control.check_stop()?;
            self.scored.sort_by_key(|(_, key)| std::cmp::Reverse(*key));
            return Ok(Some(self.scored.iter().map(|(mv, _)| *mv).collect()));
        }
        let mv = moves[self.cursor];
        let key = if best == Some(mv) {
            Some((3, 0))
        } else if !is_tactical(position, mv) {
            Some((1, engine.order_score(position, mv, None)))
        } else {
            if self.exchange.is_none() {
                visit_exchange(control, &mut self.exchange_positions)?;
                self.exchange = Some(Exchange::new(position, mv)?);
            }
            let score = self
                .exchange
                .as_mut()
                .expect("created exchange")
                .advance(control, &mut self.exchange_positions)?;
            if let Some(score) = score {
                self.exchange = None;
                Some((if score >= 0 { 2 } else { 0 }, score))
            } else {
                None
            }
        };
        if let Some(key) = key {
            self.scored.push((mv, key));
            self.cursor += 1;
        }
        Ok(None)
    }
}

fn visit_exchange(control: &mut Control<'_>, positions: &mut u32) -> Result<(), Abort> {
    control.check_stop()?;
    if *positions >= MAX_SEE_POSITIONS_PER_ORDER {
        return Err(Abort::Error(CpuError::Unsupported(
            "legal SEE ordering exchange position budget exhausted",
        )));
    }
    control.visit(false)?;
    *positions += 1;
    Ok(())
}

struct Exchange {
    position: Position,
    target: Square,
    initial_gain: i32,
    initial_undo: Option<UndoToken>,
    frames: Vec<ExchangeFrame>,
    returned: Option<i32>,
}

struct ExchangeFrame {
    plies: u16,
    entered: bool,
    moves: Vec<BoardMove>,
    cursor: usize,
    best: i32,
    active: Option<(UndoToken, i32)>,
}

impl ExchangeFrame {
    fn new(plies: u16) -> Self {
        Self {
            plies,
            entered: false,
            moves: Vec::new(),
            cursor: 0,
            best: 0,
            active: None,
        }
    }
}

impl Exchange {
    fn new(position: &Position, mv: BoardMove) -> Result<Self, Abort> {
        let mut position = position.clone();
        let mover = position.side_to_move();
        let undo = position.make_move(mv)?;
        let gain = exchange_gain(undo.delta(), mover);
        let mut frames = Vec::new();
        frames
            .try_reserve_exact(MAX_EXCHANGE_FRAMES)
            .map_err(|_| Abort::Error(CpuError::Allocation))?;
        frames.push(ExchangeFrame::new(1));
        Ok(Self {
            position,
            target: mv.to,
            initial_gain: gain,
            initial_undo: Some(undo),
            frames,
            returned: None,
        })
    }

    fn retained_bytes(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>();
        for charge in [
            vector_bytes(&self.frames),
            self.position
                .snapshot()
                .retained_history_bytes()
                .unwrap_or(usize::MAX),
            self.initial_undo.as_ref().map_or(0, undo_bytes),
        ] {
            bytes = bytes.saturating_add(charge);
        }
        for frame in &self.frames {
            bytes = bytes.saturating_add(vector_bytes(&frame.moves));
            if let Some((undo, _)) = &frame.active {
                bytes = bytes.saturating_add(undo_bytes(undo));
            }
        }
        bytes
    }

    fn advance(
        &mut self,
        control: &mut Control<'_>,
        positions: &mut u32,
    ) -> Result<Option<i32>, Abort> {
        if self.frames.is_empty() {
            let reply = self.returned.take().expect("exchange root completed");
            self.position.unmake(
                self.initial_undo
                    .take()
                    .expect("exchange owns initial undo"),
            )?;
            return Ok(Some(self.initial_gain - reply));
        }
        let frame = self.frames.last_mut().expect("live exchange frame");
        if !frame.entered {
            visit_exchange(control, positions)?;
            frame.moves = self
                .position
                .ordered_legal_moves()
                .moves()
                .iter()
                .filter(|mv| mv.to == self.target)
                .copied()
                .collect();
            frame.entered = true;
            return Ok(None);
        }
        if let Some((undo, gain)) = frame.active.take() {
            let reply = self.returned.take().expect("completed exchange child");
            self.position.unmake(undo)?;
            frame.best = frame.best.max(gain - reply);
            return Ok(None);
        }
        if frame.cursor == frame.moves.len() {
            self.returned = Some(frame.best);
            self.frames.pop();
            return Ok(None);
        }
        if frame.plies >= MAX_EXCHANGE_FRAMES as u16 {
            return Err(Abort::Error(CpuError::Unsupported(
                "legal SEE exchange ply bound exhausted",
            )));
        }
        let mv = frame.moves[frame.cursor];
        let mover = self.position.side_to_move();
        let undo = self.position.make_move(mv)?;
        if !undo.delta().kind.capture {
            self.position.unmake(undo)?;
            return Err(Abort::Error(CpuError::Unsupported(
                "legal SEE target recapture was not a capture",
            )));
        }
        let gain = exchange_gain(undo.delta(), mover);
        frame.cursor += 1;
        let plies = frame.plies + 1;
        frame.active = Some((undo, gain));
        self.frames.push(ExchangeFrame::new(plies));
        Ok(None)
    }
}

fn exchange_gain(delta: &RuleMoveDelta, mover: Color) -> i32 {
    let value = |change: &PieceChange| {
        let material = material(change.piece.kind);
        if change.piece.color == mover {
            material
        } else {
            -material
        }
    };
    delta.additions.iter().map(value).sum::<i32>() - delta.removals.iter().map(value).sum::<i32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_value::{CpuFloatCheckpoint, CpuFloatValue};
    use std::sync::atomic::AtomicU64;

    fn pause_at_root_tie(
        cpu: &mut CpuEngine,
        position: &Position,
        moves: &[BoardMove],
        cancel: &AtomicBool,
    ) -> (CpuReport, u64) {
        let mut report = cpu
            .analyze_root_moves(position, moves, limits(1), cancel)
            .unwrap();
        let mut charged = report.nodes;
        for _ in 0..10_000 {
            if cpu.paused.as_ref().is_some_and(|task| task.frames.iter().any(|frame| {
                matches!(&frame.stage, Stage::Await(active) if active.phase == ChildPhase::RootTieConfirm)
            })) {
                return (report, charged);
            }
            assert!(report.completed_depth < 2, "confirmation was not reached");
            let token = report.resume.take().expect("original paused owner");
            report = cpu.resume(position, &token, limits(1), cancel).unwrap();
            charged += report.nodes;
        }
        panic!("bounded work did not reach root tie confirmation");
    }

    #[test]
    fn root_tie_confirmation_can_pause_and_resume_the_same_owned_work() {
        let position = Position::startpos();
        let before = position.snapshot();
        let moves = [
            BoardMove::from_uci("b1c3").unwrap(),
            BoardMove::from_uci("g1f3").unwrap(),
        ];
        let cancel = AtomicBool::new(false);
        for earlier_is_worse in [false, true] {
            let mut whole = super::super::tests::root_tie_engine(
                CpuResumePolicy::PausedStack,
                earlier_is_worse,
                0,
            );
            let expected = whole
                .analyze_root_moves(&position, &moves, limits(30_000), &cancel)
                .unwrap();
            let mut sliced = super::super::tests::root_tie_engine(
                CpuResumePolicy::PausedStack,
                earlier_is_worse,
                0,
            );
            let (mut report, mut charged) =
                pause_at_root_tie(&mut sliced, &position, &moves, &cancel);
            for _ in 0..10_000 {
                if report.completed_depth == 2 {
                    break;
                }
                let token = report.resume.take().expect("original paused owner");
                report = sliced
                    .resume(&position, &token, limits(1), &cancel)
                    .unwrap();
                charged += report.nodes;
            }
            assert_eq!(
                (
                    report.completed_depth,
                    report.score,
                    report.best_move,
                    &report.pv
                ),
                (
                    expected.completed_depth,
                    expected.score,
                    expected.best_move,
                    &expected.pv
                )
            );
            assert_eq!(
                charged, expected.nodes,
                "resuming must not recharge consumed nodes"
            );
            assert!(position.matches_snapshot(&before));
        }
    }

    #[test]
    fn root_tie_confirmation_cancel_and_value_error_discard_only_the_owned_task() {
        let position = Position::startpos();
        let before = position.snapshot();
        let moves = [
            BoardMove::from_uci("b1c3").unwrap(),
            BoardMove::from_uci("g1f3").unwrap(),
        ];
        for cancel_instead in [false, true] {
            let cancel = AtomicBool::new(false);
            let (mut cpu, fail_score) = super::super::tests::root_tie_engine_with_failure();
            let (mut report, _) = pause_at_root_tie(&mut cpu, &position, &moves, &cancel);
            let token = report.resume.take().expect("original paused owner");
            if cancel_instead {
                cancel.store(true, Ordering::Release);
                let stopped = cpu
                    .resume(&position, &token, limits(30_000), &cancel)
                    .unwrap();
                assert_eq!(stopped.completion, CpuCompletion::Canceled);
                assert!(stopped.resume.is_none());
                cancel.store(false, Ordering::Release);
            } else {
                fail_score.store(true, Ordering::Release);
                assert!(matches!(
                    cpu.resume(&position, &token, limits(30_000), &cancel),
                    Err(CpuError::Value(CpuValueError::NonFiniteForward))
                ));
                fail_score.store(false, Ordering::Release);
            }
            assert!(!cpu.token_is_current(&token));
            assert!(cpu.paused.is_none());
            assert!(position.matches_snapshot(&before));
            // A new task must reconstruct the Rules/value owner; a dropped
            // in-flight confirmation must not leak its partial state.
            let fresh = cpu
                .analyze_root_moves(&position, &moves, limits(30_000), &cancel)
                .unwrap();
            assert_eq!(
                (fresh.completed_depth, fresh.score, fresh.best_move),
                (2, 0, Some(moves[0]))
            );
            assert!(position.matches_snapshot(&before));
        }
    }

    fn config(tt_entries: usize) -> CpuConfig {
        CpuConfig {
            tt_entries,
            max_depth: 4,
            quiescence_ply: 4,
            ..CpuConfig::default()
        }
    }

    fn limits(nodes: u64) -> CpuLimits {
        CpuLimits {
            max_depth: 2,
            max_nodes: nodes,
            deadline: None,
        }
    }

    fn cpu(ordering: CpuOrderingPolicy, tt_entries: usize) -> CpuEngine {
        CpuEngine::with_evaluator_ordering_and_resume_policy(
            config(tt_entries),
            Arc::new(BootstrapCpuValue::default()),
            ordering,
            CpuResumePolicy::PausedStack,
        )
        .unwrap()
    }

    fn reciprocal_fixture(
        position: &Position,
    ) -> (CpuReciprocalScope, Arc<CpuControlMailbox>, CpuLimits) {
        use rz_contracts::pals::SearchAuthority;
        use rz_contracts::pals_adaptive::{CpuJobEnvelope, CpuJobId, CpuWorkerId, ReviewViewSeal};
        use rz_contracts::rovent::{
            ChessPly, QuestionEnvelope, QuestionPurpose, ReciprocalQuestionId, SearchRecipient,
        };
        use rz_contracts::{
            ClockDomain, Deadline, Digest, GameGeneration, MonotonicTick, OwnerId, ProcessEpoch,
            RootGeneration,
        };
        let authority = SearchAuthority {
            epoch: ProcessEpoch(3),
            game: GameGeneration(4),
            root: RootGeneration(5),
            implementation: Digest([6; 32]),
        };
        let anchor = rz_position::contracts::ContractPosition::new(OwnerId(8), position.clone())
            .export()
            .unwrap()
            .snapshot()
            .identity();
        let deadline = Deadline {
            clock: ClockDomain(ProcessEpoch(3)),
            at: MonotonicTick(u64::MAX),
        };
        let parent = QuestionEnvelope {
            id: ReciprocalQuestionId {
                authority,
                sequence: 1,
            },
            parent: None,
            sender: SearchRecipient::Proposer,
            recipient: SearchRecipient::Researcher,
            purpose: QuestionPurpose::TacticalReview,
            anchor,
            anchor_root_ply: ChessPly(0),
            original_first_move: None,
            prefix_digest: Digest([1; 32]),
            line_digest: None,
            line_offset: ChessPly(0),
            question_digest: Digest([2; 32]),
            view: ReviewViewSeal {
                authority,
                view_sequence: 1,
                context_policy: Digest([3; 32]),
                view_digest: Digest([4; 32]),
                root_state: anchor,
                actual_tokens: 8,
                selected_records: 1,
                anchors: 1,
                length_bucket: 8,
            },
            evidence_revision: 1,
            root_budget: Digest([5; 32]),
            original_deadline: deadline,
        };
        let job = CpuJobEnvelope {
            id: CpuJobId {
                authority,
                sequence: 1,
            },
            state_identity: anchor,
            question: parent.question_digest,
            root_restriction: Digest([6; 32]),
            evaluation_namespace: Digest([7; 32]),
            search_conditions: Digest([8; 32]),
            total_native_node_credit: 100_000,
            hard_deadline: deadline,
        };
        let owner = CpuWorkerId(0);
        let mailbox = Arc::new(CpuControlMailbox::new(
            CpuMailboxScope { job: job.id, owner },
            rz_runtime::CompletionSignal::default(),
        ));
        (
            CpuReciprocalScope {
                job,
                owner,
                parent,
                recipient: SearchRecipient::Critic,
                max_questions: 1,
                node_interval: 1,
            },
            mailbox,
            CpuLimits {
                deadline: Some(Instant::now() + Duration::from_secs(60)),
                ..limits(100_000)
            },
        )
    }

    fn reciprocal_reply(
        request: &CpuReciprocalRequest,
        preferred: Vec<BoardMove>,
    ) -> CpuReciprocalReply {
        let anchor = rz_position::contracts::ContractPosition::new(
            rz_contracts::OwnerId(9),
            request.position().clone(),
        )
        .export()
        .unwrap();
        let mut question = request.scope().parent;
        question.parent = Some(question.id);
        question.id.sequence += u64::from(request.ordinal());
        question.sender = rz_contracts::rovent::SearchRecipient::Researcher;
        question.recipient = request.scope().recipient;
        question.anchor = anchor.snapshot().identity();
        question.anchor_root_ply.0 += request.ply();
        question.question_digest = rz_contracts::Digest([9; 32]);
        CpuReciprocalReply::new(request, question, &anchor, preferred).unwrap()
    }

    #[test]
    fn reciprocal_empty_reply_resumes_saved_node_without_replay_or_score_fabrication() {
        let position = Position::startpos();
        let (scope, mailbox, bounded) = reciprocal_fixture(&position);
        let mut baseline = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let expected = baseline
            .analyze(&position, bounded, &AtomicBool::new(false))
            .unwrap();
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let first = engine
            .analyze_controlled_reciprocal(
                &position,
                None,
                bounded,
                &AtomicBool::new(false),
                &CpuSliceControl::new(100_000, None, mailbox.clone()).unwrap(),
                None,
                &scope,
                None,
                |_| {},
            )
            .unwrap();
        assert_eq!(
            first.controlled.report.completion,
            CpuCompletion::AwaitReciprocal
        );
        assert_eq!(first.controlled.report.nodes, 1);
        assert_eq!(first.controlled.report.completed_depth, 0);
        assert_eq!(
            first.controlled.report.score_scope,
            CpuScoreScope::FrontierOnly
        );
        assert_eq!(first.controlled.yield_cause, None);
        let token = first.controlled.report.resume.unwrap();
        let request = first.request.unwrap();
        assert!(request.relative_prefix().is_empty());
        let reply = reciprocal_reply(&request, Vec::new());
        let remaining = CpuLimits {
            max_nodes: bounded.max_nodes - 1,
            ..bounded
        };
        let done = engine
            .analyze_controlled_reciprocal(
                &position,
                None,
                remaining,
                &AtomicBool::new(false),
                &CpuSliceControl::new(100_000, None, mailbox).unwrap(),
                Some(&token),
                &scope,
                Some(&reply),
                |_| {},
            )
            .unwrap();
        assert_eq!(done.controlled.report.completion, CpuCompletion::DepthLimit);
        assert!(done.request.is_none());
        assert_eq!(
            first.controlled.report.nodes + done.controlled.report.nodes,
            expected.nodes
        );
        assert_eq!(
            (
                done.controlled.report.best_move,
                done.controlled.report.score,
                done.controlled.report.pv
            ),
            (expected.best_move, expected.score, expected.pv)
        );
        assert!(!engine.token_is_current(&token));
        assert!(
            engine
                .validate_reciprocal_resume(Some(&token), &scope, Some(&reply))
                .is_err()
        );
    }

    #[test]
    fn reciprocal_response_is_consumed_in_actual_ordering_and_plain_resume_is_rejected() {
        let position = Position::startpos();
        let (scope, mailbox, bounded) = reciprocal_fixture(&position);
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let first = engine
            .analyze_controlled_reciprocal(
                &position,
                None,
                bounded,
                &AtomicBool::new(false),
                &CpuSliceControl::new(100_000, None, mailbox.clone()).unwrap(),
                None,
                &scope,
                None,
                |_| {},
            )
            .unwrap();
        let token = first.controlled.report.resume.unwrap();
        let request = first.request.unwrap();
        let preferred = *request.candidates().last().unwrap();
        let reply = reciprocal_reply(&request, vec![preferred]);
        assert!(
            engine
                .analyze_controlled(
                    &position,
                    None,
                    bounded,
                    &AtomicBool::new(false),
                    &CpuSliceControl::new(1, None, mailbox.clone()).unwrap(),
                    Some(&token),
                    |_| {}
                )
                .is_err()
        );
        assert!(engine.token_is_current(&token));
        let next = engine
            .analyze_controlled_reciprocal(
                &position,
                None,
                CpuLimits {
                    max_nodes: bounded.max_nodes - 1,
                    ..bounded
                },
                &AtomicBool::new(false),
                &CpuSliceControl::new(1, None, mailbox).unwrap(),
                Some(&token),
                &scope,
                Some(&reply),
                |_| {},
            )
            .unwrap();
        assert_eq!(next.controlled.report.completion, CpuCompletion::Yielded);
        assert_eq!(
            engine.paused.as_ref().unwrap().frames[0].moves[0],
            preferred
        );
        assert!(next.request.is_none());
        assert_eq!(next.controlled.report.completed_depth, 0);
        assert_eq!(next.controlled.report.nodes, 1);
    }

    #[test]
    fn reciprocal_wrong_owner_context_or_foreign_reply_preserves_live_token() {
        let position = Position::startpos();
        let (scope, mailbox, bounded) = reciprocal_fixture(&position);
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let first = engine
            .analyze_controlled_reciprocal(
                &position,
                None,
                bounded,
                &AtomicBool::new(false),
                &CpuSliceControl::new(100_000, None, mailbox.clone()).unwrap(),
                None,
                &scope,
                None,
                |_| {},
            )
            .unwrap();
        let token = first.controlled.report.resume.unwrap();
        let request = first.request.unwrap();
        let reply = reciprocal_reply(&request, Vec::new());
        assert!(
            engine
                .validate_reciprocal_resume(Some(&token), &scope, None)
                .is_err()
        );
        let mut changed = scope.clone();
        changed.owner.0 += 1;
        assert!(
            engine
                .validate_reciprocal_resume(Some(&token), &changed, Some(&reply))
                .is_err()
        );
        let mut changed = scope.clone();
        changed.parent.view.view_sequence += 1;
        assert!(
            engine
                .validate_reciprocal_resume(Some(&token), &changed, Some(&reply))
                .is_err()
        );
        let mut foreign = reply.clone();
        foreign.request.seal = Arc::new(());
        assert!(
            engine
                .validate_reciprocal_resume(Some(&token), &scope, Some(&foreign))
                .is_err()
        );
        let mut wrong_prefix = reply.clone();
        wrong_prefix
            .request
            .relative_prefix
            .push("e2e4".parse().unwrap());
        assert!(
            engine
                .validate_reciprocal_resume(Some(&token), &scope, Some(&wrong_prefix))
                .is_err()
        );
        assert!(engine.token_is_current(&token));
        let canceled = engine
            .analyze_controlled_reciprocal(
                &position,
                None,
                CpuLimits {
                    max_nodes: bounded.max_nodes - 1,
                    ..bounded
                },
                &AtomicBool::new(true),
                &CpuSliceControl::new(10, None, mailbox).unwrap(),
                Some(&token),
                &scope,
                Some(&reply),
                |_| {},
            )
            .unwrap();
        assert_eq!(
            canceled.controlled.report.completion,
            CpuCompletion::Canceled
        );
        assert_eq!(canceled.controlled.report.nodes, 0);
        assert!(canceled.request.is_none());
        assert!(!engine.token_is_current(&token));
    }

    #[test]
    fn reciprocal_nested_anchor_is_actual_rules_state_and_illegal_reply_is_rejected() {
        let position = Position::startpos();
        let (mut scope, mailbox, bounded) = reciprocal_fixture(&position);
        scope.max_questions = 3;
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let mut token = None;
        let mut reply = None;
        let mut spent = 0;
        let mut nested = None;
        for _ in 0..3 {
            let next = engine
                .analyze_controlled_reciprocal(
                    &position,
                    None,
                    CpuLimits {
                        max_nodes: bounded.max_nodes - spent,
                        ..bounded
                    },
                    &AtomicBool::new(false),
                    &CpuSliceControl::new(100_000, None, mailbox.clone()).unwrap(),
                    token.as_ref(),
                    &scope,
                    reply.as_ref(),
                    |_| {},
                )
                .unwrap();
            spent += next.controlled.report.nodes;
            token = next.controlled.report.resume;
            let request = next.request.unwrap();
            reply = Some(reciprocal_reply(&request, Vec::new()));
            nested = Some(request);
        }
        let request = nested.unwrap();
        assert_eq!(request.ply(), 1);
        assert_eq!(request.relative_prefix().len(), usize::from(request.ply()));
        let mut replayed = position.clone();
        for &movement in request.relative_prefix() {
            replayed.make_move(movement).unwrap();
        }
        assert!(
            replayed
                .snapshot()
                .same_state(&request.position().snapshot())
        );
        assert!(
            !request
                .position()
                .snapshot()
                .same_state(&position.snapshot())
        );
        let reply = reciprocal_reply(&request, Vec::new());
        let root_anchor = rz_position::contracts::ContractPosition::new(
            rz_contracts::OwnerId(9),
            position.clone(),
        )
        .export()
        .unwrap();
        assert!(
            CpuReciprocalReply::new(&request, reply.question(), &root_anchor, Vec::new()).is_err()
        );
        let exact_anchor = rz_position::contracts::ContractPosition::new(
            rz_contracts::OwnerId(9),
            request.position().clone(),
        )
        .export()
        .unwrap();
        let movement = request.candidates()[0];
        assert!(
            CpuReciprocalReply::new(
                &request,
                reply.question(),
                &exact_anchor,
                vec![movement, movement]
            )
            .is_err()
        );
        engine.clear();
        assert!(engine.pending_reciprocal_request().is_none());
    }

    #[test]
    fn reciprocal_does_not_issue_unresumable_question_after_last_job_node() {
        let position = Position::startpos();
        let (mut scope, mailbox, bounded) = reciprocal_fixture(&position);
        scope.job.total_native_node_credit = 1;
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let done = engine
            .analyze_controlled_reciprocal(
                &position,
                None,
                CpuLimits {
                    max_nodes: 1,
                    ..bounded
                },
                &AtomicBool::new(false),
                &CpuSliceControl::new(1, None, mailbox).unwrap(),
                None,
                &scope,
                None,
                |_| {},
            )
            .unwrap();
        assert_eq!(done.controlled.report.completion, CpuCompletion::NodeLimit);
        assert_eq!(done.controlled.report.nodes, 1);
        assert!(done.controlled.report.resume.is_none());
        assert!(done.request.is_none());
        assert!(engine.paused_stack_bytes().is_none());
    }

    #[test]
    fn reciprocal_prefix_excludes_job_history_and_resumes_without_recharging_that_path() {
        let mut position = Position::startpos();
        // Retain four actual history moves without introducing a tactical
        // checked qsearch frontier into this path-ownership regression.
        position
            .apply_uci_moves(&["g1f3", "g8f6", "f3g1", "f6g8"])
            .unwrap();
        let original = position.snapshot();
        let (mut scope, mailbox, bounded) = reciprocal_fixture(&position);
        scope.parent.anchor_root_ply = rz_contracts::rovent::ChessPly(4);
        scope.max_questions = 3;
        let mut baseline = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let expected = baseline
            .analyze(&position, bounded, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(
            expected.completion,
            CpuCompletion::DepthLimit,
            "the unchanged qsearch profile must complete this quiet baseline fixture"
        );
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let mut token = None;
        let mut reply = None;
        let mut spent = 0;
        let mut deepest_prefix = Vec::new();
        for _ in 0..4 {
            let next = engine
                .analyze_controlled_reciprocal(
                    &position,
                    None,
                    CpuLimits {
                        max_nodes: bounded.max_nodes - spent,
                        ..bounded
                    },
                    &AtomicBool::new(false),
                    &CpuSliceControl::new(100_000, None, mailbox.clone()).unwrap(),
                    token.as_ref(),
                    &scope,
                    reply.as_ref(),
                    |_| {},
                )
                .unwrap();
            spent += next.controlled.report.nodes;
            if let Some(request) = next.request {
                assert_eq!(request.relative_prefix().len(), usize::from(request.ply()));
                let mut replayed = position.clone();
                for &movement in request.relative_prefix() {
                    replayed.make_move(movement).unwrap();
                }
                assert!(
                    replayed
                        .snapshot()
                        .same_state(&request.position().snapshot())
                );
                assert_eq!(
                    request.position().known_history_len(),
                    position.known_history_len() + request.relative_prefix().len()
                );
                deepest_prefix = request.relative_prefix().to_vec();
                token = next.controlled.report.resume;
                reply = Some(reciprocal_reply(&request, Vec::new()));
            } else {
                assert_eq!(next.controlled.report.completion, expected.completion);
                assert_eq!(spent, expected.nodes);
                assert_eq!(
                    (
                        next.controlled.report.best_move,
                        next.controlled.report.score,
                        next.controlled.report.pv
                    ),
                    (expected.best_move, expected.score, expected.pv)
                );
                assert_eq!(deepest_prefix.len(), 1);
                assert!(position.snapshot().same_state(&original));
                return;
            }
        }
        panic!("bounded reciprocal queries failed to complete the original traversal");
    }

    #[test]
    fn configured_paused_frame_limit_is_validated_declared_and_enforced() {
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let original_conditions = engine.search_conditions();
        assert_eq!(engine.paused_stack_max_bytes(), CPU_PAUSED_STACK_MAX_BYTES);
        assert!(original_conditions.contains("paused-memory:max-1-task-8388608-bytes"));
        for invalid in [0, CPU_PAUSED_STACK_MAX_BYTES + 1] {
            assert!(matches!(
                engine.set_paused_stack_max_bytes(invalid),
                Err(CpuError::InvalidConfig(_))
            ));
            assert_eq!(engine.search_conditions(), original_conditions);
        }
        engine.set_paused_stack_max_bytes(1).unwrap();
        assert!(
            engine
                .search_conditions()
                .contains("paused-memory:max-1-task-1-bytes")
        );
        assert_eq!(
            engine.paused_stack_snapshot().unwrap().bytes_limit_current,
            1
        );
        assert!(matches!(
            engine.analyze(&Position::startpos(), limits(1), &AtomicBool::new(false)),
            Err(CpuError::PausedStackMemoryLimit {
                retained_bytes,
                maximum_bytes: 1,
            }) if retained_bytes > 1
        ));
        assert_eq!(engine.last_attempt_work(), Some(CpuWork::default()));
        let snapshot = engine.paused_stack_snapshot().unwrap();
        assert_eq!(snapshot.tokens_created, 0);
        assert_eq!(snapshot.bytes_current, 0);
        assert!(snapshot.complete);
    }

    #[test]
    fn configured_ceiling_survives_pause_resume_and_cannot_relabel_live_owner() {
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 64);
        let maximum_bytes = 1024 * 1024;
        engine.set_paused_stack_max_bytes(maximum_bytes).unwrap();
        let token = engine
            .analyze(&position, limits(1), &cancel)
            .unwrap()
            .resume
            .unwrap();
        assert!(token.is_paused_stack());
        assert!(engine.token_is_current(&token));
        let before = engine.paused_stack_snapshot().unwrap();
        assert_eq!(before.bytes_limit_current, maximum_bytes as u64);
        assert!(before.bytes_current <= before.bytes_limit_current);
        engine.set_paused_stack_max_bytes(maximum_bytes).unwrap();
        assert!(matches!(
            engine.set_paused_stack_max_bytes(2 * maximum_bytes),
            Err(CpuError::InvalidConfig(_))
        ));
        assert_eq!(engine.paused_stack_snapshot(), Some(before));
        assert!(engine.token_is_current(&token));
        assert_eq!(
            engine
                .resume(&position, &token, limits(1), &cancel)
                .unwrap()
                .nodes,
            1
        );
        assert_eq!(
            engine.paused_stack_snapshot().unwrap().bytes_limit_current,
            maximum_bytes as u64
        );
        let before_clear = engine.paused_stack_snapshot().unwrap();
        engine.clear();
        let after_clear = engine.paused_stack_snapshot().unwrap();
        assert_eq!(after_clear.owner_id, before_clear.owner_id);
        assert_eq!(after_clear.bytes_peak, before_clear.bytes_peak);
        assert_eq!(after_clear.bytes_limit_current, maximum_bytes as u64);
        assert_eq!(after_clear.bytes_current, 0);
        engine.set_paused_stack_max_bytes(maximum_bytes).unwrap();
        assert!(matches!(
            engine.set_paused_stack_max_bytes(2 * maximum_bytes),
            Err(CpuError::InvalidConfig(_))
        ));
    }

    #[test]
    fn task_memory_checks_use_the_captured_ceiling_at_safe_boundaries() {
        let engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let position = Position::startpos();
        let mut task = PausedTask::new(
            &engine,
            &position,
            None,
            position.ordered_legal_moves().moves().to_vec(),
            limits(1),
            None,
            None,
        )
        .unwrap();
        let retained = task.retained_bytes();
        task.maximum_bytes = retained;
        task.check_memory().unwrap();
        assert!(matches!(
            task.start_iteration(false),
            Err(CpuError::PausedStackMemoryLimit {
                retained_bytes,
                maximum_bytes,
            }) if maximum_bytes == retained && retained_bytes > retained
        ));
    }

    #[test]
    fn held_side_scope_gets_one_exact_ceiling_before_either_owner_starts() {
        let mut engine = CpuEngine::with_paused_side_scope(
            config(64),
            Arc::new(BootstrapCpuValue::default()),
            CpuOrderingPolicy::LegacyMvvLvaV1,
        )
        .unwrap();
        let maximum_bytes = 1024 * 1024;
        engine.set_paused_stack_max_bytes(maximum_bytes).unwrap();
        assert_eq!(engine.paused_stack_max_bytes(), maximum_bytes);
        assert_eq!(
            engine
                .paused_side_scope
                .as_ref()
                .unwrap()
                .paused_stack_max_bytes(),
            maximum_bytes
        );
        let cancel = AtomicBool::new(false);
        let position = Position::startpos();
        let token = engine
            .analyze(&position, limits(1), &cancel)
            .unwrap()
            .resume
            .unwrap();
        let side_before = engine.paused_side_scope_snapshot();
        assert!(matches!(
            engine.set_paused_stack_max_bytes(2 * maximum_bytes),
            Err(CpuError::InvalidConfig(_))
        ));
        assert_eq!(engine.paused_side_scope_snapshot(), side_before);
        assert!(engine.token_is_current(&token));
        assert_eq!(
            engine
                .paused_side_scope
                .as_ref()
                .unwrap()
                .paused_stack_max_bytes(),
            maximum_bytes
        );
    }

    fn add(total: &mut CpuWork, part: CpuWork) {
        total.nodes += part.nodes;
        total.quiescence_nodes += part.quiescence_nodes;
        total.tt_hits += part.tt_hits;
    }

    #[test]
    #[ignore = "opt-in API cost probe; fixed interrupted workload, not a strength result"]
    fn cost_same_interrupted_workload() {
        use std::fmt::Write as _;

        const FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        const TARGET_DEPTH: u16 = 2;
        const TT_ENTRIES: usize = 64;
        const NODE_LIMIT_SCRIPT: [u64; 16] = [64; 16];
        const WARMUPS: usize = 3;
        const MEASUREMENTS: usize = 5;
        const WALL_LIMIT: Duration = Duration::from_secs(45);

        struct Attempt {
            api: &'static str,
            span: Duration,
            work_delta: CpuWork,
            before: Option<CpuPausedStackSnapshot>,
            after: Option<CpuPausedStackSnapshot>,
            returned_token: bool,
            token_is_paused_stack: bool,
            token_bytes: Option<usize>,
            completion: CpuCompletion,
            completed_depth: u16,
        }

        struct Sample {
            policy: CpuResumePolicy,
            attempts: Vec<Attempt>,
            api_span: Duration,
            work: CpuWork,
            resumes: usize,
            final_report: Option<CpuReport>,
            not_exercised: Option<&'static str>,
        }

        fn interrupted(policy: CpuResumePolicy, deadline: Instant) -> Sample {
            // A fresh owner gives both policies the same empty TT and history.
            // Setup, token checks, snapshots and assertions are outside API spans.
            let position = Position::from_fen(FEN).unwrap();
            let initial_position = position.snapshot();
            let cancellation = AtomicBool::new(false);
            let mut engine = CpuEngine::with_evaluator_ordering_and_resume_policy(
                config(TT_ENTRIES),
                Arc::new(BootstrapCpuValue::default()),
                CpuOrderingPolicy::LegalSeeV1,
                policy,
            )
            .unwrap();
            let mut sample = Sample {
                policy,
                attempts: Vec::with_capacity(NODE_LIMIT_SCRIPT.len()),
                api_span: Duration::ZERO,
                work: CpuWork::default(),
                resumes: 0,
                final_report: None,
                not_exercised: Some("fixed_script_exhausted"),
            };
            let mut token: Option<CpuResumeToken> = None;
            for max_nodes in NODE_LIMIT_SCRIPT {
                if Instant::now() >= deadline {
                    sample.not_exercised = Some("global_wall_limit");
                    break;
                }
                let limits = CpuLimits {
                    max_depth: TARGET_DEPTH,
                    max_nodes,
                    deadline: Some(deadline),
                };
                let before = engine.paused_stack_snapshot();
                let (api, result, span) = if let Some(token) = token.as_ref() {
                    assert!(engine.token_is_current(token));
                    sample.resumes += 1;
                    let start = Instant::now();
                    let result = engine.resume(&position, token, limits, &cancellation);
                    let span = start.elapsed();
                    ("resume", result, span)
                } else {
                    let start = Instant::now();
                    let result = engine.analyze(&position, limits, &cancellation);
                    let span = start.elapsed();
                    ("analyze", result, span)
                };
                let mut report = result.expect("actual API failure is not not_exercised");
                // last_attempt_work is already the fresh delta for this call;
                // subtracting the preceding call would miscount replayed work.
                let work_delta = engine.last_attempt_work().expect("CpuEngine observes work");
                let after = engine.paused_stack_snapshot();
                assert_eq!(work_delta.nodes, report.nodes);
                assert_eq!(work_delta.quiescence_nodes, report.quiescence_nodes);
                assert_eq!(work_delta.tt_hits, report.tt_hits);
                assert!(work_delta.nodes <= max_nodes);
                assert!(initial_position.same_state(&position.snapshot()));
                match policy {
                    CpuResumePolicy::CompletedIteration => {
                        assert!(before.is_none() && after.is_none());
                    }
                    CpuResumePolicy::PausedStack => {
                        let before = before.unwrap();
                        let after = after.unwrap();
                        assert_eq!(before.owner_id, after.owner_id);
                        assert!(after.complete);
                        assert!(after.tokens_retained <= 1);
                        assert!(after.bytes_current <= CPU_PAUSED_STACK_MAX_BYTES as u64);
                        assert!(after.bytes_peak <= CPU_PAUSED_STACK_MAX_BYTES as u64);
                        assert_eq!(after.replayed_consumed_work, 0);
                    }
                }
                sample.api_span += span;
                add(&mut sample.work, work_delta);
                let returned_token = report.resume.is_some();
                let token_is_paused_stack = report
                    .resume
                    .as_ref()
                    .is_some_and(CpuResumeToken::is_paused_stack);
                let token_bytes = report
                    .resume
                    .as_ref()
                    .and_then(CpuResumeToken::paused_stack_bytes);
                sample.attempts.push(Attempt {
                    api,
                    span,
                    work_delta,
                    before,
                    after,
                    returned_token,
                    token_is_paused_stack,
                    token_bytes,
                    completion: report.completion,
                    completed_depth: report.completed_depth,
                });
                let completion = report.completion;
                let completed_depth = report.completed_depth;
                token = report.resume.take();
                sample.final_report = Some(report);
                if completion == CpuCompletion::DepthLimit {
                    sample.not_exercised = if completed_depth != TARGET_DEPTH {
                        Some("target_depth_not_reached")
                    } else if sample.resumes == 0 {
                        Some("no_node_limit_resume")
                    } else {
                        None
                    };
                    break;
                }
                if completion != CpuCompletion::NodeLimit {
                    sample.not_exercised = Some(if completion == CpuCompletion::Deadline {
                        "global_wall_limit"
                    } else {
                        "non_node_limit_completion"
                    });
                    break;
                }
                if token.is_none() {
                    sample.not_exercised = Some("resume_token_unavailable_at_fixed_budget");
                    break;
                }
            }
            // This disposes only this owner's frames after measurement. Snapshot
            // rows prove traversal accounting, not bridge closure/owner release.
            engine.discard_paused_stack();
            sample
        }

        let deadline = Instant::now() + WALL_LIMIT;
        let mut output = String::new();
        writeln!(
            output,
            "cpu_cost_manifest fen={FEN:?} ordering={} tt_entries={TT_ENTRIES} \
             config_max_depth=4 quiescence_ply=4 target_depth={TARGET_DEPTH} \
             node_limit_script={NODE_LIMIT_SCRIPT:?} warmups={WARMUPS} \
             measurements={MEASUREMENTS} wall_limit_seconds=45 \
             timer=API_entry_to_return setup_in_timer=false \
             comparison=same_interrupted_workload budget_adaptation=false \
             fresh_TT_and_history_per_owner=true paused_snapshot_None=unavailable \
             max_API_calls=256",
            CpuOrderingPolicy::LegalSeeV1.identity(),
        )
        .unwrap();
        let mut warmups_exercised = true;
        let mut measurements_exercised = 0;
        let mut measured_api_spans = [Duration::ZERO; 2];
        let mut measured_work = [CpuWork::default(); 2];
        for repetition in 0..WARMUPS + MEASUREMENTS {
            // Alternate order, but never transfer TT/history/tokens between owners.
            let policies = if repetition % 2 == 0 {
                [
                    CpuResumePolicy::CompletedIteration,
                    CpuResumePolicy::PausedStack,
                ]
            } else {
                [
                    CpuResumePolicy::PausedStack,
                    CpuResumePolicy::CompletedIteration,
                ]
            };
            let first = interrupted(policies[0], deadline);
            let second = interrupted(policies[1], deadline);
            let samples = if first.policy == CpuResumePolicy::CompletedIteration {
                [first, second]
            } else {
                [second, first]
            };
            let exercised = samples.iter().all(|sample| sample.not_exercised.is_none());
            if exercised {
                let baseline = samples[0].final_report.as_ref().unwrap();
                let paused = samples[1].final_report.as_ref().unwrap();
                assert_eq!(
                    (
                        baseline.score,
                        &baseline.pv,
                        baseline.best_move,
                        baseline.completed_depth
                    ),
                    (
                        paused.score,
                        &paused.pv,
                        paused.best_move,
                        paused.completed_depth
                    ),
                    "cost comparison requires the same completed workload result",
                );
            }
            let warmup = repetition < WARMUPS;
            if warmup {
                warmups_exercised &= exercised;
            } else if exercised {
                measurements_exercised += 1;
                for (index, sample) in samples.iter().enumerate() {
                    measured_api_spans[index] += sample.api_span;
                    add(&mut measured_work[index], sample.work);
                }
            }
            for sample in &samples {
                for (slice, attempt) in sample.attempts.iter().enumerate() {
                    writeln!(
                        output,
                        "cpu_cost_attempt repetition={repetition} warmup={warmup} \
                         policy={} slice={slice} node_limit={} api={} api_ns={} \
                         work_delta={:?} returned_token={} token_is_paused_stack={} \
                         token_bytes={:?} snapshot_before={:?} snapshot_after={:?} \
                         completion={:?} completed_depth={}",
                        sample.policy.identity(),
                        NODE_LIMIT_SCRIPT[slice],
                        attempt.api,
                        attempt.span.as_nanos(),
                        attempt.work_delta,
                        attempt.returned_token,
                        attempt.token_is_paused_stack,
                        attempt.token_bytes,
                        attempt.before,
                        attempt.after,
                        attempt.completion,
                        attempt.completed_depth,
                    )
                    .unwrap();
                }
                writeln!(
                    output,
                    "cpu_cost_sample repetition={repetition} warmup={warmup} policy={} \
                     status={} reason={:?} API_calls={} resumes={} api_ns={} work_delta_total={:?}",
                    sample.policy.identity(),
                    if sample.not_exercised.is_none() {
                        "exercised"
                    } else {
                        "not_exercised"
                    },
                    sample.not_exercised,
                    sample.attempts.len(),
                    sample.resumes,
                    sample.api_span.as_nanos(),
                    sample.work,
                )
                .unwrap();
            }
        }
        let comparison_ready = warmups_exercised && measurements_exercised == MEASUREMENTS;
        writeln!(
            output,
            "cpu_cost_summary status={} warmups_exercised={warmups_exercised} \
             measured_pairs_exercised={measurements_exercised}/{MEASUREMENTS} \
             completed_iteration_api_ns={} paused_stack_api_ns={} \
             completed_iteration_work={:?} paused_stack_work={:?} \
             comparison_ready={comparison_ready} timing_scope=API_only",
            if comparison_ready {
                "exercised"
            } else {
                "not_exercised"
            },
            measured_api_spans[0].as_nanos(),
            measured_api_spans[1].as_nanos(),
            measured_work[0],
            measured_work[1],
        )
        .unwrap();
        print!("{output}");
    }

    fn run_slices(
        engine: &mut CpuEngine,
        position: &Position,
        root_moves: Option<&[BoardMove]>,
        interval: u64,
    ) -> (CpuReport, CpuWork, bool) {
        let before = position.snapshot();
        let owner = engine.paused_stack_snapshot().unwrap().owner_id;
        let cancel = AtomicBool::new(false);
        let mut report = if let Some(moves) = root_moves {
            engine
                .analyze_root_moves(position, moves, limits(interval), &cancel)
                .unwrap()
        } else {
            engine.analyze(position, limits(interval), &cancel).unwrap()
        };
        let mut total = CpuWork::default();
        let mut saw_exchange = false;
        for _ in 0..100_000 {
            assert_eq!(engine.last_attempt_work().unwrap().nodes, report.nodes);
            assert!(report.nodes <= interval);
            add(&mut total, engine.last_attempt_work().unwrap());
            assert!(before.same_state(&position.snapshot()));
            if report.completion != CpuCompletion::NodeLimit {
                assert_eq!(report.completion, CpuCompletion::DepthLimit);
                assert!(engine.paused.is_none());
                let snapshot = engine.paused_stack_snapshot().unwrap();
                assert!(snapshot.complete);
                assert_eq!(snapshot.owner_id, owner);
                assert_eq!(snapshot.tokens_retained, 0);
                assert_eq!(snapshot.bytes_current, 0);
                assert_eq!(snapshot.replayed_consumed_work, 0);
                assert!(!snapshot.admission_closed && !snapshot.owner_released);
                return (report, total, saw_exchange);
            }
            let token = report
                .resume
                .take()
                .expect("even depth-zero partial frames resume");
            assert!(token.is_paused_stack());
            assert!(engine.token_is_current(&token));
            assert_eq!(token.cumulative_work(), Some(total));
            assert_eq!(token.paused_stack_bytes(), engine.paused_stack_bytes());
            assert!(token.paused_stack_bytes().unwrap() <= CPU_PAUSED_STACK_MAX_BYTES);
            let snapshot = engine.paused_stack_snapshot().unwrap();
            assert_eq!(snapshot, engine.paused_stack_snapshot().unwrap());
            assert!(snapshot.complete);
            assert_eq!(snapshot.owner_id, owner);
            assert_eq!(snapshot.tokens_retained, 1);
            assert_eq!(snapshot.tokens_peak, 1);
            assert_eq!(
                snapshot.bytes_current,
                token.paused_stack_bytes().unwrap() as u64
            );
            assert!(snapshot.bytes_peak >= snapshot.bytes_current);
            assert!(snapshot.bytes_peak <= CPU_PAUSED_STACK_MAX_BYTES as u64);
            assert_eq!(snapshot.replayed_consumed_work, 0);
            assert_eq!(
                snapshot.tokens_created,
                snapshot.tokens_resumed + snapshot.tokens_invalidated + 1
            );
            let task = engine.paused.as_ref().unwrap();
            let fresh = engine.evaluator.initialize(&task.work).unwrap();
            let live = task.value.as_ref().unwrap();
            assert_eq!(
                fresh.values().map(|side| side.map(f32::to_bits)),
                live.values().map(|side| side.map(f32::to_bits))
            );
            assert_eq!(
                engine.evaluator.score(&fresh, &task.work).unwrap(),
                engine.evaluator.score(live, &task.work).unwrap()
            );
            saw_exchange |= task.frames.iter().any(|frame| {
                matches!(&frame.stage,
                Stage::Ordering(order) if order.exchange.is_some())
            });
            report = engine
                .resume(position, &token, limits(interval), &cancel)
                .unwrap();
            let after = engine.paused_stack_snapshot().unwrap();
            assert_eq!(after.tokens_resumed, snapshot.tokens_resumed + 1);
            assert_eq!(after.tokens_invalidated, snapshot.tokens_invalidated);
            assert_eq!(
                after.event_sequence,
                snapshot.event_sequence + 1 + u64::from(after.tokens_retained)
            );
        }
        panic!("finite fixture did not complete within its declared call bound");
    }

    fn compare_tt_and_history(left: &CpuEngine, right: &CpuEngine) {
        assert_eq!(left.history, right.history);
        assert_eq!(left.tt.len(), right.tt.len());
        for (a, b) in left.tt.iter().zip(&right.tt) {
            match (a, b) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(
                        (a.hash, a.depth, a.score, a.bound, a.best_move),
                        (b.hash, b.depth, b.score, b.bound, b.best_move)
                    );
                    assert_eq!(a.identity, b.identity);
                    assert_eq!(a.value_identity, b.value_identity);
                }
                _ => panic!("TT completion changed across pause intervals"),
            }
        }
    }

    #[test]
    fn disposing_a_paused_owner_preserves_tt_history_and_observed_work() {
        let position = Position::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        let cancel = AtomicBool::new(false);
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 64);
        let completed = engine.analyze(&position, limits(100_000), &cancel).unwrap();
        assert_eq!(completed.completion, CpuCompletion::DepthLimit);
        assert!(engine.tt.iter().any(Option::is_some));
        assert!(
            engine
                .history
                .iter()
                .flatten()
                .flatten()
                .any(|value| *value != 0)
        );
        let partial = engine
            .analyze(
                &position,
                CpuLimits {
                    max_depth: 3,
                    ..limits(1)
                },
                &cancel,
            )
            .unwrap();
        let token = partial.resume.unwrap();
        let work = engine.last_attempt_work();
        let mut unchanged = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 64);
        unchanged.tt = engine.tt.clone();
        unchanged.history = engine.history.clone();
        assert!(engine.token_is_current(&token));
        assert!(engine.supports_paused_stack_discard());
        let before_discard = engine.paused_stack_snapshot().unwrap();
        assert!(engine.discard_paused_stack());
        let discarded = engine.paused_stack_snapshot().unwrap();
        assert!(discarded.complete);
        assert_eq!(
            discarded.tokens_invalidated,
            before_discard.tokens_invalidated + 1
        );
        assert_eq!(discarded.event_sequence, before_discard.event_sequence + 1);
        assert_eq!(discarded.tokens_retained, 0);
        assert_eq!(discarded.bytes_current, 0);
        assert_eq!(discarded.bytes_peak, before_discard.bytes_peak);
        assert!(!discarded.owner_released && !discarded.admission_closed);
        assert!(!engine.token_is_current(&token));
        assert!(engine.paused_stack_bytes().is_none());
        assert_eq!(engine.last_attempt_work(), work);
        compare_tt_and_history(&engine, &unchanged);
        assert!(!engine.discard_paused_stack());
        assert_eq!(discarded, engine.paused_stack_snapshot().unwrap());
        assert_eq!(engine.last_attempt_work(), work);
        compare_tt_and_history(&engine, &unchanged);
        assert!(matches!(
            engine.resume(&position, &token, limits(1), &cancel),
            Err(CpuError::ResumeMismatch(_))
        ));
        assert_eq!(engine.last_attempt_work().unwrap().nodes, 0);
        let rejected = engine.paused_stack_snapshot().unwrap();
        assert_eq!(
            rejected.stale_context_attempts,
            discarded.stale_context_attempts + 1
        );
        assert_eq!(
            rejected.stale_context_rejections,
            discarded.stale_context_rejections + 1
        );
        assert_eq!(rejected.event_sequence, discarded.event_sequence + 1);
    }

    #[test]
    fn paused_owner_lifetime_survives_new_game_and_counter_overflow_is_unknown() {
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let initial = engine.paused_stack_snapshot().unwrap();
        assert!(initial.complete && initial.owner_id != 0);
        assert_eq!(initial.event_sequence, 0);
        assert_eq!(initial.tokens_peak, 0);
        assert_eq!(initial.bytes_peak, 0);
        assert_eq!(initial, engine.paused_stack_snapshot().unwrap());
        let foreign = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        assert_ne!(
            initial.owner_id,
            foreign.paused_stack_snapshot().unwrap().owner_id
        );
        engine.analyze(&position, limits(1), &cancel).unwrap();
        let paused = engine.paused_stack_snapshot().unwrap();
        assert_eq!(paused.tokens_created, 1);
        engine.clear();
        let reset = engine.paused_stack_snapshot().unwrap();
        assert_eq!(reset.owner_id, initial.owner_id);
        assert_eq!(reset.tokens_created, 1);
        assert_eq!(reset.tokens_invalidated, 1);
        assert_eq!(reset.event_sequence, paused.event_sequence + 1);
        assert_eq!(reset.bytes_peak, paused.bytes_peak);
        assert!(reset.complete);
        assert!(engine.last_attempt_work().is_none());
        engine.clear();
        assert_eq!(reset, engine.paused_stack_snapshot().unwrap());

        // Exhaust the local sequence without mutating the process allocator or
        // another owner. Real create/discard continue, but release stays unknown.
        engine
            .paused_stack_telemetry
            .as_mut()
            .unwrap()
            .event_sequence = u64::MAX;
        let report = engine.analyze(&position, limits(1), &cancel).unwrap();
        assert_eq!(report.nodes, 1);
        let overflow = engine.paused_stack_snapshot().unwrap();
        assert!(!overflow.complete);
        assert_eq!(overflow.event_sequence, u64::MAX);
        assert_eq!(overflow.tokens_created, 2);
        assert_eq!(overflow.tokens_retained, 1);
        assert!(engine.discard_paused_stack());
        let released_frames = engine.paused_stack_snapshot().unwrap();
        assert_eq!(released_frames.owner_id, initial.owner_id);
        assert_eq!(released_frames.tokens_invalidated, 2);
        assert_eq!(released_frames.tokens_retained, 0);
        assert_eq!(released_frames.bytes_current, 0);
        assert!(!released_frames.complete);
        assert!(!released_frames.owner_released);
    }

    #[test]
    fn subdivisions_preserve_recursive_score_pv_depth_work_and_tt() {
        for ordering in [
            CpuOrderingPolicy::LegacyMvvLvaV1,
            CpuOrderingPolicy::LegalSeeV1,
        ] {
            for tt in [0, 64] {
                for fen in [
                    "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
                    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
                    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
                ] {
                    let position = Position::from_fen(fen).unwrap();
                    let mut recursive = CpuEngine::with_ordering(config(tt), ordering).unwrap();
                    let expected = recursive
                        .analyze(&position, limits(100_000), &AtomicBool::new(false))
                        .unwrap();
                    assert_eq!(expected.completion, CpuCompletion::DepthLimit);
                    for interval in [1, 3, 13] {
                        let mut sliced = cpu(ordering, tt);
                        let (actual, total, _) = run_slices(&mut sliced, &position, None, interval);
                        assert_eq!(
                            (actual.score, &actual.pv, actual.completed_depth),
                            (expected.score, &expected.pv, expected.completed_depth)
                        );
                        assert_eq!(total, recursive.last_attempt_work().unwrap());
                        compare_tt_and_history(&sliced, &recursive);
                    }
                }
            }
        }
    }

    #[test]
    fn saved_see_frames_and_root_restrictions_resume_without_recharging() {
        let position = Position::from_fen("3rk3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        let moves = [BoardMove::from_uci("e4d5").unwrap()];
        let mut full = cpu(CpuOrderingPolicy::LegalSeeV1, 16);
        let expected = full
            .analyze_root_moves(&position, &moves, limits(100_000), &AtomicBool::new(false))
            .unwrap();
        let mut sliced = cpu(CpuOrderingPolicy::LegalSeeV1, 16);
        let (actual, work, saw_exchange) = run_slices(&mut sliced, &position, Some(&moves), 1);
        assert!(
            saw_exchange,
            "fixture must actually pause in material-exchange frames"
        );
        assert!(actual.root_restricted);
        assert_eq!(
            (actual.score, actual.pv, actual.completed_depth),
            (expected.score, expected.pv, expected.completed_depth)
        );
        assert_eq!(work, full.last_attempt_work().unwrap());
        compare_tt_and_history(&sliced, &full);
    }

    #[test]
    fn pausing_does_not_repeat_evaluator_or_make_unmake_work_without_charging_it() {
        #[derive(Default)]
        struct Counts {
            initializations: AtomicU64,
            scores: AtomicU64,
            deltas: AtomicU64,
            restores: AtomicU64,
        }
        impl Counts {
            fn observed(&self) -> [u64; 4] {
                [
                    &self.initializations,
                    &self.scores,
                    &self.deltas,
                    &self.restores,
                ]
                .map(|count| count.load(Ordering::Relaxed))
            }
        }
        struct CountedValue {
            base: BootstrapCpuValue,
            counts: Arc<Counts>,
        }
        impl CpuValueEvaluator for CountedValue {
            fn identity(&self) -> &CpuValueIdentity {
                self.base.identity()
            }
            fn provenance(&self) -> &'static str {
                self.base.provenance()
            }
            fn initialize(&self, position: &Position) -> Result<CpuAccumulator, CpuValueError> {
                self.counts.initializations.fetch_add(1, Ordering::Relaxed);
                self.base.initialize(position)
            }
            fn score(
                &self,
                value: &CpuAccumulator,
                position: &Position,
            ) -> Result<i32, CpuValueError> {
                self.counts.scores.fetch_add(1, Ordering::Relaxed);
                self.base.score(value, position)
            }
            fn apply_delta(
                &self,
                value: &mut CpuAccumulator,
                delta: &RuleMoveDelta,
            ) -> Result<CpuAccumulatorUndo, CpuValueError> {
                self.counts.deltas.fetch_add(1, Ordering::Relaxed);
                self.base.apply_delta(value, delta)
            }
            fn restore(
                &self,
                value: &mut CpuAccumulator,
                undo: CpuAccumulatorUndo,
                position: &Position,
            ) -> Result<(), CpuValueError> {
                self.counts.restores.fetch_add(1, Ordering::Relaxed);
                self.base.restore(value, undo, position)
            }
        }
        let position = Position::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        let cancel = AtomicBool::new(false);
        let full_counts = Arc::new(Counts::default());
        let sliced_counts = Arc::new(Counts::default());
        let mut full = CpuEngine::with_evaluator(
            config(16),
            Arc::new(CountedValue {
                base: BootstrapCpuValue::default(),
                counts: Arc::clone(&full_counts),
            }),
        )
        .unwrap();
        let expected = full.analyze(&position, limits(100_000), &cancel).unwrap();
        let mut sliced: Box<dyn CpuSearcher> = Box::new(
            CpuEngine::with_evaluator_ordering_and_resume_policy(
                config(16),
                Arc::new(CountedValue {
                    base: BootstrapCpuValue::default(),
                    counts: Arc::clone(&sliced_counts),
                }),
                CpuOrderingPolicy::LegacyMvvLvaV1,
                CpuResumePolicy::PausedStack,
            )
            .unwrap(),
        );
        let mut report = sliced.analyze(&position, limits(1), &cancel).unwrap();
        let mut total = report.nodes;
        for _ in 0..100_000 {
            if report.completion == CpuCompletion::DepthLimit {
                break;
            }
            assert_eq!(report.completion, CpuCompletion::NodeLimit);
            let token = report.resume.unwrap();
            assert!(sliced.token_is_current(&token));
            report = sliced
                .resume(&position, &token, limits(1), &cancel)
                .unwrap();
            assert!(!sliced.token_is_current(&token));
            total += report.nodes;
        }
        assert_eq!(report.completion, CpuCompletion::DepthLimit);
        assert_eq!(
            (report.score, report.pv, report.completed_depth),
            (expected.score, expected.pv, expected.completed_depth)
        );
        assert_eq!(total, expected.nodes);
        assert_eq!(sliced_counts.observed(), full_counts.observed());
        let observed = sliced_counts.observed();
        assert_eq!(
            observed[0], 1,
            "resume must retain the original accumulator"
        );
        assert!(observed[2] > 0);
        assert_eq!(
            observed[2], observed[3],
            "every actual make must restore once"
        );
    }

    #[test]
    fn resumed_value_failure_consumes_one_token_and_preserves_new_work_without_partial_tt() {
        struct FaultingScore {
            base: BootstrapCpuValue,
            fail: Arc<AtomicBool>,
        }
        impl CpuValueEvaluator for FaultingScore {
            fn identity(&self) -> &CpuValueIdentity {
                self.base.identity()
            }
            fn provenance(&self) -> &'static str {
                self.base.provenance()
            }
            fn initialize(&self, p: &Position) -> Result<CpuAccumulator, CpuValueError> {
                self.base.initialize(p)
            }
            fn score(&self, a: &CpuAccumulator, p: &Position) -> Result<i32, CpuValueError> {
                if self.fail.load(Ordering::Acquire) {
                    return Err(CpuValueError::NonFiniteForward);
                }
                self.base.score(a, p)
            }
            fn apply_delta(
                &self,
                a: &mut CpuAccumulator,
                d: &RuleMoveDelta,
            ) -> Result<CpuAccumulatorUndo, CpuValueError> {
                self.base.apply_delta(a, d)
            }
            fn restore(
                &self,
                a: &mut CpuAccumulator,
                u: CpuAccumulatorUndo,
                p: &Position,
            ) -> Result<(), CpuValueError> {
                self.base.restore(a, u, p)
            }
        }
        let position = Position::startpos();
        let before = position.snapshot();
        let restriction = [BoardMove::from_uci("e2e4").unwrap()];
        let cancel = AtomicBool::new(false);
        let fail = Arc::new(AtomicBool::new(false));
        let mut engine = CpuEngine::with_evaluator_ordering_and_resume_policy(
            config(64),
            Arc::new(FaultingScore {
                base: BootstrapCpuValue::default(),
                fail: Arc::clone(&fail),
            }),
            CpuOrderingPolicy::LegacyMvvLvaV1,
            CpuResumePolicy::PausedStack,
        )
        .unwrap();
        let partial = engine
            .analyze_root_moves(&position, &restriction, limits(1), &cancel)
            .unwrap();
        assert_eq!(partial.nodes, 1);
        let token = partial.resume.unwrap();
        let paused = engine.paused_stack_snapshot().unwrap();
        assert!(engine.tt.iter().all(Option::is_none));
        fail.store(true, Ordering::Release);
        // A one-node resume must reach the injected scorer. Seeding Control
        // with the old charged node would stop before that new work and hide
        // this typed failure behind NodeLimit.
        assert!(matches!(
            engine.resume(&position, &token, limits(1), &cancel),
            Err(CpuError::Value(CpuValueError::NonFiniteForward))
        ));
        let failed_work = engine.last_attempt_work().unwrap();
        assert_eq!(failed_work.nodes, 1);
        assert_eq!(failed_work.quiescence_nodes, 1);
        assert_eq!(failed_work.tt_hits, 0);
        assert_eq!(token.cumulative_work().unwrap().nodes, 1);
        assert!(before.same_state(&position.snapshot()));
        assert!(engine.tt.iter().all(Option::is_none));
        assert!(!engine.token_is_current(&token));
        let failed_owner = engine.paused_stack_snapshot().unwrap();
        assert!(failed_owner.complete);
        assert_eq!(failed_owner.owner_id, paused.owner_id);
        assert_eq!(failed_owner.tokens_created, 1);
        assert_eq!(failed_owner.tokens_resumed, 1);
        assert_eq!(failed_owner.tokens_invalidated, 0);
        assert_eq!(failed_owner.tokens_retained, 0);
        assert_eq!(failed_owner.bytes_current, 0);
        assert_eq!(failed_owner.replayed_consumed_work, 0);
        assert!(matches!(
            engine.resume(&position, &token, limits(1), &cancel),
            Err(CpuError::ResumeMismatch(_))
        ));
        assert_eq!(engine.last_attempt_work(), Some(CpuWork::default()));
        let rejected = engine.paused_stack_snapshot().unwrap();
        assert_eq!(rejected.tokens_resumed, 1);
        assert_eq!(rejected.stale_context_attempts, 1);
        assert_eq!(rejected.stale_context_rejections, 1);

        fail.store(false, Ordering::Release);
        let recovered = engine
            .analyze_root_moves(&position, &restriction, limits(100_000), &cancel)
            .unwrap();
        let mut fresh = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 64);
        let expected = fresh
            .analyze_root_moves(&position, &restriction, limits(100_000), &cancel)
            .unwrap();
        assert_eq!(recovered.completion, CpuCompletion::DepthLimit);
        assert_eq!(
            (recovered.score, recovered.pv, recovered.completed_depth),
            (expected.score, expected.pv, expected.completed_depth)
        );
        assert_eq!(engine.last_attempt_work(), fresh.last_attempt_work());
        compare_tt_and_history(&engine, &fresh);
        assert!(before.same_state(&position.snapshot()));
        assert_eq!(
            engine.paused_stack_snapshot().unwrap().owner_id,
            paused.owner_id
        );
    }

    #[test]
    fn aspiration_failure_and_pvs_research_keep_the_same_work_after_one_node_slices() {
        let position = Position::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        let mut checkpoint = CpuFloatCheckpoint::zeros_untrained();
        checkpoint.output_bias = 0.075;
        let evaluator: Arc<dyn CpuValueEvaluator> =
            Arc::new(CpuFloatValue::new(checkpoint).unwrap());
        let mut recursive = CpuEngine::with_evaluator(config(16), Arc::clone(&evaluator)).unwrap();
        let expected = recursive
            .analyze(&position, limits(100_000), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(expected.completed_depth, 2);
        assert_eq!(expected.score, 75); // depth one is -75, outside its +/-40 aspiration.
        let mut sliced = CpuEngine::with_evaluator_ordering_and_resume_policy(
            config(16),
            evaluator,
            CpuOrderingPolicy::LegacyMvvLvaV1,
            CpuResumePolicy::PausedStack,
        )
        .unwrap();
        let (actual, total, _) = run_slices(&mut sliced, &position, None, 1);
        assert_eq!(
            (actual.score, actual.pv, actual.completed_depth),
            (expected.score, expected.pv, expected.completed_depth)
        );
        assert_eq!(total, recursive.last_attempt_work().unwrap());
        compare_tt_and_history(&sliced, &recursive);
    }

    #[test]
    fn exact_nonzero_float_accumulator_restores_special_moves_after_partial_frames() {
        let mut checkpoint = CpuFloatCheckpoint::zeros_untrained();
        for (index, value) in checkpoint.feature_weights.iter_mut().enumerate() {
            *value = ((index * 13 % 97) as f32 - 48.0) * 0.0001;
        }
        checkpoint.accumulator_bias.fill(0.4);
        checkpoint.hidden_weights.fill(0.001);
        checkpoint.hidden_bias.fill(0.03);
        checkpoint.output_weights.fill(0.02);
        let model: Arc<dyn CpuValueEvaluator> = Arc::new(CpuFloatValue::new(checkpoint).unwrap());
        for (fen, text) in [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
        ] {
            let position = Position::from_fen(fen).unwrap();
            let moves = [BoardMove::from_uci(text).unwrap()];
            let mut sliced = CpuEngine::with_evaluator_ordering_and_resume_policy(
                config(16),
                Arc::clone(&model),
                CpuOrderingPolicy::LegacyMvvLvaV1,
                CpuResumePolicy::PausedStack,
            )
            .unwrap();
            let mut full = CpuEngine::with_evaluator_and_ordering(
                config(16),
                Arc::clone(&model),
                CpuOrderingPolicy::LegacyMvvLvaV1,
            )
            .unwrap();
            let expected = full
                .analyze_root_moves(&position, &moves, limits(100_000), &AtomicBool::new(false))
                .unwrap();
            let (actual, work, _) = run_slices(&mut sliced, &position, Some(&moves), 3);
            assert_eq!(
                (actual.score, actual.pv, actual.completed_depth),
                (expected.score, expected.pv, expected.completed_depth)
            );
            assert_eq!(work, full.last_attempt_work().unwrap());
            // Inspect the fully restored root before the owning task is dropped.
            let legal = vec![moves[0]];
            let mut task = PausedTask::new(
                &sliced,
                &position,
                Some(&moves),
                legal,
                limits(100_000),
                None,
                None,
            )
            .unwrap();
            let fresh = model.initialize(&position).unwrap();
            let mut control = Control {
                limits: limits(100_000),
                cancellation: &AtomicBool::new(false),
                cooperative: None,
                nodes: 0,
                quiescence_nodes: 0,
                tt_hits: 0,
                value: task.value.take().unwrap(),
            };
            drive(&mut sliced, &mut task, &mut control, &mut |_| {}).unwrap();
            assert_eq!(task.work.position_identity(), position.position_identity());
            assert_eq!(
                control.value.values().map(|side| side.map(f32::to_bits)),
                fresh.values().map(|side| side.map(f32::to_bits))
            );
        }
    }

    #[test]
    fn one_use_foreign_malformed_and_stale_tokens_are_rejected_atomically() {
        let mut position = Position::startpos();
        position
            .apply_uci_moves(&["g1f3", "g8f6", "f3g1", "f6g8"])
            .unwrap();
        let restriction = [BoardMove::from_uci("e2e4").unwrap()];
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let token = engine
            .analyze_root_moves(&position, &restriction, limits(1), &AtomicBool::new(false))
            .unwrap()
            .resume
            .unwrap();
        assert!(token.is_paused_stack());
        assert!(engine.token_is_current(&token));
        let observed_owner = engine.paused_stack_snapshot().unwrap();
        let observed_work = engine.last_attempt_work();
        assert!(engine.token_is_current(&token));
        assert_eq!(engine.last_attempt_work(), observed_work);
        assert_eq!(observed_owner, engine.paused_stack_snapshot().unwrap());
        let mut malformed = token.clone();
        malformed.pv.clear();
        assert!(!engine.token_is_current(&malformed));
        assert!(matches!(
            engine.resume(&position, &malformed, limits(1), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        malformed = token.clone();
        malformed.root_moves = None;
        assert!(matches!(
            engine.resume(&position, &malformed, limits(1), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        malformed = token.clone();
        malformed.paused_stack.as_mut().unwrap().retained_bytes += 1;
        assert!(matches!(
            engine.resume(&position, &malformed, limits(1), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        let imported = Position::from_fen(&position.to_fen()).unwrap();
        assert!(matches!(
            engine.resume(&imported, &token, limits(1), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        assert!(matches!(
            engine.resume(
                &position,
                &token,
                CpuLimits {
                    max_depth: 3,
                    ..limits(1)
                },
                &AtomicBool::new(false)
            ),
            Err(CpuError::ResumeMismatch(_))
        ));
        let mut foreign = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        assert!(matches!(
            foreign.resume(&position, &token, limits(1), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        let rejected = engine.paused_stack_snapshot().unwrap();
        assert_eq!(rejected.tokens_created, 1);
        assert_eq!(rejected.tokens_retained, 1);
        assert_eq!(rejected.tokens_resumed, 0);
        assert_eq!(rejected.tokens_invalidated, 0);
        assert_eq!(rejected.stale_context_attempts, 5);
        assert_eq!(rejected.stale_context_rejections, 5);
        let foreign_owner = foreign.paused_stack_snapshot().unwrap();
        assert_ne!(foreign_owner.owner_id, rejected.owner_id);
        assert_eq!(foreign_owner.tokens_created, 0);
        assert_eq!(foreign_owner.stale_context_attempts, 1);
        assert_eq!(foreign_owner.stale_context_rejections, 1);
        assert_eq!(engine.last_attempt_work(), Some(CpuWork::default()));
        let next = engine
            .resume(&position, &token, limits(1), &AtomicBool::new(false))
            .unwrap()
            .resume
            .unwrap();
        assert!(!engine.token_is_current(&token));
        assert!(engine.token_is_current(&next));
        assert!(matches!(
            engine.resume(&position, &token, limits(1), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        engine.clear();
        assert!(!engine.token_is_current(&next));
        assert!(matches!(
            engine.resume(&position, &next, limits(1), &AtomicBool::new(false)),
            Err(CpuError::ResumeMismatch(_))
        ));
        let final_owner = engine.paused_stack_snapshot().unwrap();
        assert!(final_owner.complete);
        assert_eq!(final_owner.owner_id, observed_owner.owner_id);
        assert_eq!(final_owner.tokens_created, 2);
        assert_eq!(final_owner.tokens_resumed, 1);
        assert_eq!(final_owner.tokens_invalidated, 1);
        assert_eq!(final_owner.tokens_retained, 0);
        assert_eq!(final_owner.stale_context_attempts, 7);
        assert_eq!(final_owner.stale_context_rejections, 7);
        assert_eq!(final_owner.replayed_consumed_work, 0);
    }

    #[test]
    fn cancellation_new_analysis_and_namespace_mutation_invalidate_retained_work() {
        let position = Position::startpos();
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let cancel = AtomicBool::new(false);
        let token = engine
            .analyze(&position, limits(1), &cancel)
            .unwrap()
            .resume
            .unwrap();
        let canceled = engine
            .resume(&position, &token, limits(1), &AtomicBool::new(true))
            .unwrap();
        assert_eq!(canceled.completion, CpuCompletion::Canceled);
        assert_eq!(canceled.nodes, 0);
        assert!(canceled.resume.is_none() && engine.paused.is_none());
        assert!(!engine.token_is_current(&token));
        let canceled_owner = engine.paused_stack_snapshot().unwrap();
        assert!(canceled_owner.complete);
        assert_eq!(canceled_owner.tokens_created, 1);
        assert_eq!(canceled_owner.tokens_resumed, 1);
        assert_eq!(canceled_owner.tokens_invalidated, 0);
        assert_eq!(canceled_owner.tokens_retained, 0);
        assert_eq!(canceled_owner.bytes_current, 0);
        assert!(matches!(
            engine.resume(&position, &token, limits(1), &cancel),
            Err(CpuError::ResumeMismatch(_))
        ));
        let old = engine
            .analyze(&position, limits(1), &cancel)
            .unwrap()
            .resume
            .unwrap();
        let new = engine
            .analyze(&position, limits(1), &cancel)
            .unwrap()
            .resume
            .unwrap();
        assert!(!engine.token_is_current(&old));
        assert!(engine.token_is_current(&new));
        assert!(matches!(
            engine.resume(&position, &old, limits(1), &cancel),
            Err(CpuError::ResumeMismatch(_))
        ));
        Arc::make_mut(&mut engine.value_identity).semantics = "changed-bootstrap-namespace".into();
        assert!(!engine.token_is_current(&new));
        assert!(matches!(
            engine.resume(&position, &new, limits(1), &cancel),
            Err(CpuError::InvalidConfig(_))
        ));
        assert!(engine.paused.is_none());
        assert_eq!(engine.last_attempt_work(), Some(CpuWork::default()));
        let final_owner = engine.paused_stack_snapshot().unwrap();
        assert!(final_owner.complete);
        assert_eq!(final_owner.owner_id, canceled_owner.owner_id);
        assert_eq!(final_owner.tokens_created, 3);
        assert_eq!(final_owner.tokens_resumed, 1);
        assert_eq!(final_owner.tokens_invalidated, 2);
        assert_eq!(final_owner.tokens_retained, 0);
        assert_eq!(final_owner.stale_context_attempts, 3);
        assert_eq!(final_owner.stale_context_rejections, 3);
    }

    #[test]
    fn expired_deadline_preserves_next_stage_and_memory_refusal_is_explicit() {
        let position = Position::startpos();
        let mut engine = cpu(CpuOrderingPolicy::LegacyMvvLvaV1, 0);
        let cancel = AtomicBool::new(false);
        let first = engine
            .analyze(&position, limits(1), &cancel)
            .unwrap()
            .resume
            .unwrap();
        let expired = engine
            .resume(
                &position,
                &first,
                CpuLimits {
                    deadline: Some(Instant::now()),
                    ..limits(1)
                },
                &cancel,
            )
            .unwrap();
        assert_eq!(expired.completion, CpuCompletion::Deadline);
        assert_eq!(expired.nodes, 0);
        let next = expired.resume.unwrap();
        assert_eq!(next.cumulative_work(), first.cumulative_work());
        assert!(matches!(
            engine.resume(&position, &first, limits(1), &cancel),
            Err(CpuError::ResumeMismatch(_))
        ));
        assert_eq!(
            engine
                .resume(&position, &next, limits(1), &cancel)
                .unwrap()
                .nodes,
            1
        );
        let task = engine.paused.as_mut().unwrap();
        task.best
            .try_reserve_exact(CPU_PAUSED_STACK_MAX_BYTES / std::mem::size_of::<BoardMove>() + 1)
            .unwrap();
        assert!(matches!(
            task.check_memory(),
            Err(CpuError::PausedStackMemoryLimit { .. })
        ));
    }
}
