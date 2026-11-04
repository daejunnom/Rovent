//! Native CPU checkpoint ownership at admission and logical cancellation.
//! Historical task evidence does not certify that a backend frame still lives.

use super::*;
use crate::cpu::CpuResumePolicy;
use std::collections::BTreeSet;

impl<M: RoleModel> PalsEngine<M> {
    pub(super) fn prepare_cpu_interleave_root(
        &mut self,
        root: usize,
        limits: PalsLimits,
    ) -> Result<(), PalsError> {
        if self.cpu_interleave == CpuInterleavePolicy::ResumeAfterAcceptedRepairC
            && self.held_cpu_task.as_ref().is_some_and(|held| {
                held.active_root != self.semantic.nodes[root].state
                    || held.target_depth != limits.cpu_depth
            })
        {
            self.discard_cpu_stack_owner()?;
            self.synchronize_cpu_checkpoints()?;
            self.held_cpu_task = None;
        }
        Ok(())
    }

    /// Select the ordered Repair/C lane only for a live original CPU owner in
    /// this active Rules root. A root AnalyzePosition has an empty prefix and
    /// no first-move restriction; the accepted Repair supplies the queue scope.
    pub(super) fn held_cpu_repair_priority(
        &self,
        root: usize,
        original_first: BoardMove,
        limits: PalsLimits,
    ) -> Result<bool, PalsError> {
        if self.cpu_interleave != CpuInterleavePolicy::ResumeAfterAcceptedRepairC
            || self.post_repair_recheck != PostRepairRecheckPolicy::IterativeFrozenModelWdlV2
        {
            return Ok(false);
        }
        let Some(held) = self.held_cpu_task.as_ref() else {
            return Ok(false);
        };
        self.validate_checker_namespace()?;
        let root_node = self
            .semantic
            .nodes
            .get(root)
            .ok_or(StoreError::InvalidHandle("held Repair root"))?;
        if root_node.state != held.active_root
            || held.target_depth != limits.cpu_depth
            || held
                .prefix
                .first()
                .is_some_and(|first| *first != original_first)
        {
            return Ok(false);
        }
        if self.semantic.stores.root() != Some(root_node.situation) {
            return Err(StoreError::StaleConsumer.into());
        }
        let node = self
            .semantic
            .nodes
            .get(held.node)
            .ok_or(StoreError::InvalidHandle("held CPU node"))?;
        let Some(token) = node.resume.as_ref() else {
            return Err(StoreError::InvalidConditions("held CPU task has no token").into());
        };
        if !token.is_paused_stack() || !self.researcher.cpu.token_is_current(token) {
            return Err(StoreError::InvalidConditions("held CPU task owner is stale").into());
        }
        let task = self.semantic.stores.tasks.get(held.execution)?;
        if task.key.state != node.state
            || task.key.line.is_some()
            || task.key.question != TaskQuestion::AnalyzePosition
            || !task.key.root_moves.is_empty()
            || task.key.requested_depth != held.target_depth
            || task.key.value_identity.as_ref() != self.researcher.cpu_registered_value.as_ref()
            || task.key.cpu_condition.as_ref() != Some(&self.researcher.cpu_registered_condition)
            || task
                .key
                .checker_identity
                .as_ref()
                .is_some_and(|identity| identity != &self.researcher.checker_registered_identity)
            || !matches!(task.status, TaskStatus::Paused { checkpoint, .. }
                if checkpoint == held.node as u64)
        {
            return Err(StoreError::InvalidConditions("held CPU execution is not paused").into());
        }
        Ok(true)
    }

    /// Called only after a completed, actually accepted post-Repair C tail.
    /// The frozen model value still comes from its own Fresh namespace. This
    /// executes the original CPU state/task with the remaining shared budget.
    pub(super) fn resume_cpu_after_repair_c(
        &mut self,
        limits: PalsLimits,
        cancel: &AtomicBool,
        counters: &mut PalsCounters,
    ) -> Result<(), PalsError> {
        if self.cpu_interleave != CpuInterleavePolicy::ResumeAfterAcceptedRepairC
            || self.stopped(limits, cancel).is_some()
            || counters.cpu_nodes >= limits.max_cpu_nodes
        {
            return Ok(());
        }
        self.synchronize_cpu_checkpoints()?;
        let Some(held) = self.held_cpu_task.as_ref() else {
            return Ok(());
        };
        let root = self
            .semantic
            .stores
            .root()
            .ok_or(StoreError::StaleConsumer)?;
        if self.semantic.stores.situations.get(root)?.state != held.active_root {
            return Err(StoreError::StaleConsumer.into());
        }
        if held.target_depth != limits.cpu_depth {
            return Err(StoreError::InvalidConditions("held CPU target depth changed").into());
        }
        let node = held.node;
        let Some(token) = self.semantic.nodes[node].resume.as_ref() else {
            return Err(StoreError::InvalidConditions("held CPU task has no token").into());
        };
        if !token.is_paused_stack() || !self.researcher.cpu.token_is_current(token) {
            return Err(StoreError::InvalidConditions("held CPU task owner is stale").into());
        }
        let previous = held.execution;
        let task = self.semantic.stores.tasks.get(previous)?;
        if task.key.state != self.semantic.nodes[node].state
            || task.key.question != TaskQuestion::AnalyzePosition
            || !matches!(task.status, TaskStatus::Paused { checkpoint, .. }
                if checkpoint == node as u64)
        {
            return Err(StoreError::InvalidConditions("held CPU execution is not paused").into());
        }
        // TaskTable keys include the requested node chunk. A smaller remainder
        // is a different task, not permission to replace or rename this owner.
        let remaining = limits.max_cpu_nodes.saturating_sub(counters.cpu_nodes);
        if remaining.min(self.config.cpu_nodes_per_task) != task.key.node_budget {
            return Ok(());
        }
        // The retained path is authority from the same active Rules root, not
        // a record hash or a repaired/counter state's approximate identity.
        let root_node = self
            .semantic
            .nodes
            .iter()
            .find(|node| node.state == held.active_root)
            .ok_or(StoreError::StaleConsumer)?;
        let mut checked = root_node.position.clone();
        for &movement in &held.prefix {
            checked.make_move(movement)?;
        }
        if checked.position_identity() != self.semantic.nodes[node].position.position_identity() {
            return Err(StoreError::InvalidConditions("held CPU prefix/history mismatch").into());
        }
        // Move the retained path into this invocation rather than cloning a
        // third prefix buffer. verify_checked_cpu publishes its successor path
        // under the same combined 8MiB pause reservation.
        let held = self.held_cpu_task.take().expect("validated held CPU task");
        let requested_before = counters.cpu_tasks_requested;
        let result = self.verify_checked_cpu(node, &held.prefix, limits, cancel, counters);
        if self.held_cpu_task.is_none()
            && self.semantic.nodes[node]
                .resume
                .as_ref()
                .is_some_and(|token| {
                    token.is_paused_stack() && self.researcher.cpu.token_is_current(token)
                })
        {
            self.held_cpu_task = Some(held);
        }
        result?;
        if self.stopped(limits, cancel).is_none() && counters.cpu_tasks_requested > requested_before
        {
            let (_, execution) = self.semantic.nodes[node]
                .evidence
                .as_ref()
                .and_then(|evidence| evidence.provenance)
                .ok_or(StoreError::InvalidEvidence("CPU resume publication absent"))?;
            if self.semantic.stores.tasks.get(execution)?.resumed_from != Some(previous) {
                return Err(StoreError::InvalidEvidence("held CPU task was not resumed").into());
            }
        }
        if self.semantic.nodes[node]
            .resume
            .as_ref()
            .is_none_or(|token| !token.is_paused_stack())
        {
            self.held_cpu_task = None;
        }
        Ok(())
    }

    fn cpu_stack_discard_required(&self) -> Result<bool, PalsError> {
        let policy = self.researcher.cpu.owned_resume_policy();
        let supports_discard = self.researcher.cpu.supports_paused_stack_discard();
        let stack_token = self
            .semantic
            .nodes
            .iter()
            .filter_map(|node| node.resume.as_ref())
            .any(CpuResumeToken::is_paused_stack);
        let observed_stack = self
            .researcher
            .cpu
            .paused_stack_snapshot()
            .is_some_and(|snapshot| {
                snapshot.tokens_retained != 0 || snapshot.bytes_current != 0 || !snapshot.complete
            });
        // Disposal capability alone does not select or retain a stack: the
        // native CompletedIteration owner deliberately has no stack snapshot.
        // A custom capable owner with no disclosed policy remains unknown and
        // still needs the original disposal witness.
        let required = policy == Some(CpuResumePolicy::PausedStack)
            || stack_token
            || observed_stack
            || (supports_discard && policy.is_none());
        if required && !supports_discard {
            return Err(CheckerError::Unsupported("owned paused stack disposal").into());
        }
        Ok(required)
    }

    /// Before every own CPU admission, reconcile retained task metadata with
    /// the actual owner. A fresh analysis may have invalidated another Node's
    /// single live stack since the previous admission, even with archive off.
    /// This neither observes new work nor resets an attempt's measured work.
    pub(super) fn synchronize_cpu_checkpoints(&mut self) -> Result<(), PalsError> {
        if self.is_external() {
            return Ok(());
        }
        let discard_required = self.cpu_stack_discard_required()?;
        let mut current = BTreeSet::new();
        let mut stale = Vec::new();
        stale
            .try_reserve_exact(self.semantic.nodes.len())
            .map_err(|_| StoreError::Capacity("CPU checkpoint synchronization"))?;
        for (index, node) in self.semantic.nodes.iter().enumerate() {
            let Some(token) = node.resume.as_ref() else {
                continue;
            };
            let mut latest = None;
            if self.researcher.cpu.token_is_current(token) {
                for execution in self
                    .semantic
                    .stores
                    .tasks
                    .paused_executions_for_state(node.state)?
                {
                    let task = self.semantic.stores.tasks.get(execution)?;
                    if task.key.line.is_none()
                        && task.key.question == TaskQuestion::AnalyzePosition
                        && task.key.root_moves.is_empty()
                        && task.key.value_identity.as_ref()
                            == self.researcher.cpu_registered_value.as_ref()
                        && task.key.cpu_condition.as_ref()
                            == Some(&self.researcher.cpu_registered_condition)
                        && task.key.checker_identity.as_ref().is_none_or(|identity| {
                            identity == &self.researcher.checker_registered_identity
                        })
                        && matches!(task.status, TaskStatus::Paused { checkpoint, .. }
                            if checkpoint == index as u64)
                    {
                        latest =
                            Some(latest.map_or(execution, |old: ExecutionId| old.max(execution)));
                    }
                }
            }
            if let Some(execution) = latest {
                current.insert(execution);
            } else {
                stale.push(index);
            }
        }
        if current.is_empty() && discard_required {
            // Restricted discovery can leave frames without a resumable Node.
            // Such an orphan has no live task consumer to preserve.
            self.discard_cpu_stack_owner()?;
        }
        // Unknown or retained physical ownership above must not retire the
        // logical checkpoint. Store preflights before changing any statuses.
        self.semantic.stores.tasks.retire_paused_except(&current)?;
        for index in stale {
            self.semantic.nodes[index].resume = None;
            if self
                .held_cpu_task
                .as_ref()
                .is_some_and(|held| held.node == index)
            {
                self.held_cpu_task = None;
            }
        }
        if self.held_cpu_task.as_ref().is_some_and(|held| {
            self.semantic.nodes.get(held.node).is_none_or(|node| {
                node.resume.as_ref().is_none_or(|token| {
                    !token.is_paused_stack() || !self.researcher.cpu.token_is_current(token)
                })
            })
        }) {
            self.held_cpu_task = None;
        }
        Ok(())
    }

    /// Called at the logical canceled search boundary, including cancellation
    /// during progress or a model role after an earlier CPU pause. This never
    /// invokes checker new_game/shutdown or the model's physical lease owner.
    pub(super) fn discard_canceled_cpu_checkpoints(&mut self) -> Result<(), PalsError> {
        if self.is_external() {
            return Ok(());
        }
        if self.cpu_stack_discard_required()? {
            self.discard_cpu_stack_owner()?;
        }
        // Drop logical handles even if the finite Store retirement allocation
        // fails. The physical native frames have already been disposed above.
        for node in &mut self.semantic.nodes {
            node.resume = None;
        }
        self.held_cpu_task = None;
        self.semantic
            .stores
            .tasks
            .retire_paused_except(&BTreeSet::new())?;
        Ok(())
    }

    pub(super) fn discard_cpu_stack_owner(&mut self) -> Result<(), PalsError> {
        self.researcher.cpu.discard_paused_stack();
        // An advertised custom implementation must actually invalidate known
        // frame owners; a false return alone proves no physical completion.
        if self
            .semantic
            .nodes
            .iter()
            .filter_map(|node| node.resume.as_ref())
            .any(|token| token.is_paused_stack() && self.researcher.cpu.token_is_current(token))
        {
            return Err(CheckerError::Invalid("paused stack disposal retained its owner").into());
        }
        let snapshot =
            self.researcher
                .cpu
                .paused_stack_snapshot()
                .ok_or(CheckerError::Unsupported(
                    "paused stack disposal completion is unknown",
                ))?;
        if snapshot.tokens_retained != 0 || snapshot.bytes_current != 0 {
            return Err(CheckerError::Invalid("paused stack disposal retained its owner").into());
        }
        if !snapshot.complete {
            return Err(
                CheckerError::Invalid("paused stack disposal completion is unknown").into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{CpuCapabilities, CpuConfig, CpuWork};
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Mutex};

    struct Trace {
        tokens: Mutex<Vec<CpuResumeToken>>,
        discards: AtomicUsize,
        supports_discard: AtomicBool,
        honors_discard: AtomicBool,
        snapshot_mode: AtomicUsize,
    }

    impl Default for Trace {
        fn default() -> Self {
            Self {
                tokens: Mutex::new(Vec::new()),
                discards: AtomicUsize::new(0),
                supports_discard: AtomicBool::new(true),
                honors_discard: AtomicBool::new(true),
                snapshot_mode: AtomicUsize::new(0),
            }
        }
    }

    struct RecordingCpu {
        inner: CpuEngine,
        trace: Arc<Trace>,
    }

    impl RecordingCpu {
        fn record(&self, result: Result<CpuReport, CpuError>) -> Result<CpuReport, CpuError> {
            if let Ok(report) = &result
                && let Some(token) = &report.resume
            {
                self.trace.tokens.lock().unwrap().push(token.clone());
            }
            result
        }
    }

    impl CpuSearcher for RecordingCpu {
        fn config(&self) -> &CpuConfig {
            self.inner.config()
        }
        fn value_identity(&self) -> &CpuValueIdentity {
            self.inner.value_identity()
        }
        fn search_identity(&self) -> &'static str {
            self.inner.search_identity()
        }
        fn search_conditions(&self) -> String {
            self.inner.search_conditions()
        }
        fn capabilities(&self) -> CpuCapabilities {
            CpuSearcher::capabilities(&self.inner)
        }
        fn resume_policy(&self) -> CpuResumePolicy {
            self.inner.resume_policy()
        }
        fn token_is_current(&self, token: &CpuResumeToken) -> bool {
            self.inner.token_is_current(token)
        }
        fn paused_stack_bytes(&self) -> Option<usize> {
            self.inner.paused_stack_bytes()
        }
        fn paused_stack_snapshot(&self) -> Option<crate::cpu::CpuPausedStackSnapshot> {
            let mut snapshot = self.inner.paused_stack_snapshot()?;
            match self.trace.snapshot_mode.load(Ordering::Acquire) {
                1 => None,
                2 => {
                    snapshot.complete = false;
                    Some(snapshot)
                }
                _ => Some(snapshot),
            }
        }
        fn supports_paused_stack_discard(&self) -> bool {
            self.trace.supports_discard.load(Ordering::Acquire)
        }
        fn discard_paused_stack(&mut self) -> bool {
            self.trace.discards.fetch_add(1, Ordering::Relaxed);
            self.trace.honors_discard.load(Ordering::Acquire) && self.inner.discard_paused_stack()
        }
        fn last_attempt_work(&self) -> Option<CpuWork> {
            self.inner.last_attempt_work()
        }
        fn clear(&mut self) {
            self.inner.clear();
        }
        fn analyze(
            &mut self,
            position: &Position,
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<CpuReport, CpuError> {
            let result = self.inner.analyze(position, limits, cancel);
            self.record(result)
        }
        fn analyze_root_moves(
            &mut self,
            position: &Position,
            moves: &[BoardMove],
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<CpuReport, CpuError> {
            let result = self
                .inner
                .analyze_root_moves(position, moves, limits, cancel);
            self.record(result)
        }
        fn analyze_divergence(
            &mut self,
            position: &Position,
            prefix: &[BoardMove],
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<CpuReport, CpuError> {
            let result = self
                .inner
                .analyze_divergence(position, prefix, limits, cancel);
            self.record(result)
        }
        fn resume(
            &mut self,
            position: &Position,
            token: &CpuResumeToken,
            limits: CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<CpuReport, CpuError> {
            let result = self.inner.resume(position, token, limits, cancel);
            self.record(result)
        }
    }

    fn engine<M: RoleModel>(model: M) -> (PalsEngine<M>, Arc<Trace>) {
        engine_with_policy(model, CpuResumePolicy::PausedStack)
    }

    fn engine_with_policy<M: RoleModel>(
        model: M,
        policy: CpuResumePolicy,
    ) -> (PalsEngine<M>, Arc<Trace>) {
        let trace = Arc::new(Trace::default());
        let cpu = RecordingCpu {
            inner: CpuEngine::with_resume_policy(
                CpuConfig {
                    tt_entries: 64,
                    max_depth: 4,
                    quiescence_ply: 4,
                    ..CpuConfig::default()
                },
                policy,
            )
            .unwrap(),
            trace: trace.clone(),
        };
        // Exercise both Box<CpuSearcher> and OwnedCpuChecker forwarding.
        let engine = PalsEngine::new_with_boxed_cpu(
            PalsConfig {
                beam_width: 1,
                line_plies: 2,
                max_nodes: 257,
                max_records: 32,
                max_role_calls: 64,
                cpu_nodes_per_task: 1,
            },
            model,
            Box::new(cpu),
        )
        .unwrap();
        (engine, trace)
    }

    fn limits() -> PalsLimits {
        PalsLimits {
            deadline: Instant::now() + Duration::from_secs(20),
            max_rounds: 1,
            max_cpu_nodes: 32,
            cpu_depth: 2,
        }
    }

    #[test]
    fn observed_progress_preserves_default_choices_and_current_authority() {
        let config = PalsConfig {
            beam_width: 1,
            line_plies: 4,
            cpu_nodes_per_task: 8,
            max_role_calls: 64,
            ..PalsConfig::default()
        };
        let create = || {
            PalsEngine::new(
                config.clone(),
                LegalOrderRoleMock,
                CpuEngine::new(CpuConfig {
                    tt_entries: 64,
                    max_depth: 2,
                    ..CpuConfig::default()
                })
                .unwrap(),
            )
            .unwrap()
        };
        let mut legacy = create();
        let mut observed = create();
        let position = Position::startpos();
        let cancel = AtomicBool::new(false);
        let budget = limits();
        let mut legacy_choices = Vec::new();
        let a = legacy
            .search_with_progress(&position, budget, &cancel, |movement| {
                legacy_choices.push(movement)
            })
            .unwrap();
        let mut observed_choices = Vec::new();
        let b = observed
            .search_with_observed_progress(
                &position,
                budget,
                &cancel,
                |movement, stores, records| {
                    let root = stores.root().unwrap();
                    assert_eq!(
                        stores
                            .states
                            .get(stores.situations.get(root).unwrap().state)
                            .unwrap()
                            .position_identity(),
                        position.position_identity()
                    );
                    assert!(!records.is_empty());
                    observed_choices.push(movement);
                },
            )
            .unwrap();
        assert_eq!(legacy_choices, observed_choices);
        assert_eq!(
            (a.best_move, a.score, a.counters),
            (b.best_move, b.score, b.counters)
        );
        assert_eq!(legacy.search_identity(), observed.search_identity());
        assert_eq!(
            observed.cpu_interleave_policy(),
            CpuInterleavePolicy::ReplaceOnFresh
        );
        assert_eq!(observed.cpu_side_scope_snapshot(), None);
    }

    fn root<M: RoleModel>(engine: &mut PalsEngine<M>, position: &Position) -> usize {
        engine
            .semantic
            .stores
            .focus_actual_moves(position.snapshot())
            .unwrap();
        engine.intern(position.clone()).unwrap()
    }

    #[test]
    fn completed_iteration_without_stack_skips_disposal_witness_at_admission_and_cancel() {
        let (mut engine, trace) =
            engine_with_policy(LegalOrderRoleMock, CpuResumePolicy::CompletedIteration);
        assert!(engine.researcher.cpu.supports_paused_stack_discard());
        assert_eq!(engine.researcher.cpu.paused_stack_snapshot(), None);
        engine.synchronize_cpu_checkpoints().unwrap();
        let node = root(&mut engine, &Position::startpos());
        let cancel = AtomicBool::new(false);
        let limits = limits();
        let mut counters = PalsCounters::default();
        engine
            .verify_checked_cpu(node, &[], limits, &cancel, &mut counters)
            .unwrap();
        assert_eq!(counters.cpu_tasks_requested, 1);
        assert_eq!(counters.cpu_nodes, 1);
        let observation = engine.semantic.nodes[node]
            .evidence
            .as_ref()
            .unwrap()
            .provenance
            .unwrap()
            .0;
        let raw = engine
            .semantic
            .stores
            .observations
            .get(observation)
            .unwrap()
            .clone();
        let work = engine.researcher.cpu.last_attempt().unwrap().work;
        let before = counters;
        engine.synchronize_cpu_checkpoints().unwrap();
        engine.discard_canceled_cpu_checkpoints().unwrap();
        assert_eq!(trace.discards.load(Ordering::Acquire), 0);
        assert_eq!(engine.researcher.cpu.paused_stack_snapshot(), None);
        assert_eq!(engine.researcher.cpu.last_attempt().unwrap().work, work);
        assert_eq!(
            engine
                .semantic
                .stores
                .observations
                .get(observation)
                .unwrap(),
            &raw
        );
        assert_eq!(counters, before);
        engine.try_new_game(limits.deadline, &cancel).unwrap();
        engine.synchronize_cpu_checkpoints().unwrap();
        assert_eq!(trace.discards.load(Ordering::Acquire), 0);
        assert_eq!(engine.researcher.cpu.paused_stack_snapshot(), None);
    }

    #[test]
    fn archive_off_a_pause_b_fresh_a_same_key_starts_without_recharging_history() {
        let (mut engine, _) = engine(LegalOrderRoleMock);
        assert!(!engine.archive_enabled());
        let position = Position::startpos();
        let a = root(&mut engine, &position);
        let mut position_b = position.clone();
        position_b
            .make_move(BoardMove::from_uci("e2e4").unwrap())
            .unwrap();
        let b = engine.intern(position_b).unwrap();
        let cancel = AtomicBool::new(false);
        let limits = limits();
        let mut counters = PalsCounters::default();
        engine
            .verify_checked_cpu(a, &[], limits, &cancel, &mut counters)
            .unwrap();
        let token_a = engine.semantic.nodes[a].resume.clone().unwrap();
        let (old_observation, old_execution) = engine.semantic.nodes[a]
            .evidence
            .as_ref()
            .unwrap()
            .provenance
            .unwrap();
        let old_fact = engine
            .semantic
            .stores
            .observations
            .get(old_observation)
            .unwrap()
            .clone();
        let old_key = engine
            .semantic
            .stores
            .tasks
            .get(old_execution)
            .unwrap()
            .key
            .clone();
        assert_eq!(old_fact.budget, 1);
        engine
            .verify_checked_cpu(b, &[], limits, &cancel, &mut counters)
            .unwrap();
        assert!(!engine.researcher.cpu.token_is_current(&token_a));
        // No explicit synchronization here: actual CPU admission must perform it.
        engine
            .verify_checked_cpu(a, &[], limits, &cancel, &mut counters)
            .unwrap();
        let (_, new_execution) = engine.semantic.nodes[a]
            .evidence
            .as_ref()
            .unwrap()
            .provenance
            .unwrap();
        let new_task = engine.semantic.stores.tasks.get(new_execution).unwrap();
        assert_ne!(new_execution, old_execution);
        assert_eq!(new_task.key, old_key);
        assert!(new_task.resumed_from.is_none());
        assert!(
            engine
                .researcher
                .cpu
                .token_is_current(engine.semantic.nodes[a].resume.as_ref().unwrap())
        );
        assert!(
            matches!(engine.semantic.stores.tasks.get(old_execution).unwrap().status,
            TaskStatus::RetiredPaused { evidence: Some(id), .. } if id == old_observation)
        );
        assert_eq!(
            engine
                .semantic
                .stores
                .observations
                .get(old_observation)
                .unwrap(),
            &old_fact
        );
        assert_eq!(counters.cpu_tasks_requested, 3);
        assert_eq!(counters.cpu_nodes, 3);
        let before = counters;
        let work = engine.researcher.cpu.last_attempt().unwrap().work;
        engine.synchronize_cpu_checkpoints().unwrap();
        assert!(engine.semantic.nodes[b].resume.is_none());
        assert_eq!(counters, before);
        assert_eq!(engine.researcher.cpu.last_attempt().unwrap().work, work);
    }

    #[test]
    fn current_a_resumes_once_and_only_latest_paused_execution_remains_live() {
        let (mut engine, _) = engine(LegalOrderRoleMock);
        let node = root(&mut engine, &Position::startpos());
        let mut counters = PalsCounters::default();
        let cancel = AtomicBool::new(false);
        let limits = limits();
        engine
            .verify_checked_cpu(node, &[], limits, &cancel, &mut counters)
            .unwrap();
        let first_token = engine.semantic.nodes[node].resume.clone().unwrap();
        let (observation, previous) = engine.semantic.nodes[node]
            .evidence
            .as_ref()
            .unwrap()
            .provenance
            .unwrap();
        let fact = engine
            .semantic
            .stores
            .observations
            .get(observation)
            .unwrap()
            .clone();
        engine
            .verify_checked_cpu(node, &[], limits, &cancel, &mut counters)
            .unwrap();
        let (_, execution) = engine.semantic.nodes[node]
            .evidence
            .as_ref()
            .unwrap()
            .provenance
            .unwrap();
        let token = engine.semantic.nodes[node].resume.as_ref().unwrap();
        assert!(!engine.researcher.cpu.token_is_current(&first_token));
        assert!(engine.researcher.cpu.token_is_current(token));
        assert_eq!(token.cumulative_work().unwrap().nodes, 2);
        assert_eq!(
            engine
                .semantic
                .stores
                .tasks
                .get(execution)
                .unwrap()
                .resumed_from,
            Some(previous)
        );
        engine.synchronize_cpu_checkpoints().unwrap();
        assert!(matches!(
            engine.semantic.stores.tasks.get(previous).unwrap().status,
            TaskStatus::RetiredPaused { .. }
        ));
        assert!(matches!(
            engine.semantic.stores.tasks.get(execution).unwrap().status,
            TaskStatus::Paused { .. }
        ));
        assert_eq!(
            engine
                .semantic
                .stores
                .observations
                .get(observation)
                .unwrap(),
            &fact
        );
        assert_eq!(counters.cpu_nodes, 2);
    }

    fn assert_canceled_owner<M: RoleModel>(
        engine: &PalsEngine<M>,
        trace: &Trace,
        report: &PalsResult,
    ) {
        assert_eq!(report.completion, PalsCompletion::Canceled);
        assert!(report.counters.cpu_tasks_requested > 0);
        assert_eq!(
            report.counters.cpu_nodes,
            report.counters.cpu_tasks_requested
        );
        let tokens = trace.tokens.lock().unwrap();
        assert!(
            !tokens.is_empty(),
            "cancellation must follow an actual CPU pause"
        );
        assert!(
            tokens
                .iter()
                .all(|token| !engine.researcher.cpu.token_is_current(token))
        );
        assert!(
            engine
                .semantic
                .nodes
                .iter()
                .all(|node| node.resume.is_none())
        );
        let mut retired = 0;
        for state in engine.semantic.nodes.iter().map(|node| node.state) {
            assert!(
                engine
                    .semantic
                    .stores
                    .tasks
                    .paused_executions_for_state(state)
                    .unwrap()
                    .is_empty()
            );
        }
        for execution in 0..engine.semantic.stores.tasks.len() {
            if let TaskStatus::RetiredPaused {
                evidence: Some(observation),
                ..
            } = engine
                .semantic
                .stores
                .tasks
                .get(ExecutionId(execution))
                .unwrap()
                .status
            {
                retired += 1;
                let fact = engine
                    .semantic
                    .stores
                    .observations
                    .get(observation)
                    .unwrap();
                assert_eq!(fact.budget, 1);
                assert_eq!(fact.execution, Some(ExecutionId(execution)));
            }
        }
        assert!(retired > 0);
        assert_eq!(
            engine.researcher.cpu.last_attempt().unwrap().work.nodes,
            Some(1)
        );
    }

    #[test]
    fn cancel_from_progress_after_cpu_pause_disposes_frames_and_keeps_raw_facts() {
        let (mut engine, trace) = engine(LegalOrderRoleMock);
        let cancel = AtomicBool::new(false);
        let report = engine
            .search_with_progress(&Position::startpos(), limits(), &cancel, |_| {
                cancel.store(true, Ordering::Release);
            })
            .unwrap();
        assert!(
            cancel.load(Ordering::Acquire),
            "progress must have run after verification"
        );
        assert_canceled_owner(&engine, &trace, &report);
        let work = engine.researcher.cpu.last_attempt().unwrap().work;
        let observations = engine.semantic.stores.observations.len();
        engine.discard_canceled_cpu_checkpoints().unwrap();
        assert_eq!(engine.researcher.cpu.last_attempt().unwrap().work, work);
        assert_eq!(engine.semantic.stores.observations.len(), observations);
        cancel.store(false, Ordering::Release);
        assert!(
            engine
                .search(&Position::startpos(), limits(), &cancel)
                .is_ok()
        );
    }

    struct CancelOnReply;

    impl RoleModel for CancelOnReply {
        fn identity(&self) -> &str {
            LegalOrderRoleMock.identity()
        }
        fn value_identity(&self) -> Option<&ModelValueIdentity> {
            LegalOrderRoleMock.value_identity()
        }
        fn evaluate_value(&mut self, query: RoleQuery<'_>) -> Result<ModelValueOutput, RoleError> {
            LegalOrderRoleMock.evaluate_value(query)
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            LegalOrderRoleMock.propose(query)
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            let cancel = query.cancel;
            let output = LegalOrderRoleMock::evaluate(query)?;
            cancel.store(true, Ordering::Release);
            Ok(output)
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            LegalOrderRoleMock.repair(query)
        }
        fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            LegalOrderRoleMock.divergences(query)
        }
    }

    #[test]
    fn cancel_in_critic_after_cpu_discovery_disposes_orphan_frames() {
        let (mut engine, trace) = engine(CancelOnReply);
        let cancel = AtomicBool::new(false);
        let report = engine
            .search(&Position::startpos(), limits(), &cancel)
            .unwrap();
        assert!(report.counters.critic_calls > 0);
        assert_canceled_owner(&engine, &trace, &report);
    }

    #[test]
    fn custom_paused_owner_must_explicitly_support_and_honor_disposal() {
        let (mut engine, trace) = engine(LegalOrderRoleMock);
        let node = root(&mut engine, &Position::startpos());
        let mut counters = PalsCounters::default();
        engine
            .verify_checked_cpu(node, &[], limits(), &AtomicBool::new(false), &mut counters)
            .unwrap();
        let token = engine.semantic.nodes[node].resume.clone().unwrap();
        let tasks = engine.semantic.stores.tasks.len();
        let facts = engine.semantic.stores.observations.len();
        trace.supports_discard.store(false, Ordering::Release);
        assert!(matches!(
            engine.synchronize_cpu_checkpoints(),
            Err(PalsError::Checker(CheckerError::Unsupported(
                "owned paused stack disposal"
            )))
        ));
        assert!(matches!(
            engine.discard_canceled_cpu_checkpoints(),
            Err(PalsError::Checker(CheckerError::Unsupported(
                "owned paused stack disposal"
            )))
        ));
        assert!(engine.researcher.cpu.token_is_current(&token));
        trace.supports_discard.store(true, Ordering::Release);
        trace.honors_discard.store(false, Ordering::Release);
        assert!(matches!(
            engine.discard_canceled_cpu_checkpoints(),
            Err(PalsError::Checker(CheckerError::Invalid(
                "paused stack disposal retained its owner"
            )))
        ));
        assert!(engine.researcher.cpu.token_is_current(&token));
        assert_eq!(engine.semantic.stores.tasks.len(), tasks);
        assert_eq!(engine.semantic.stores.observations.len(), facts);
        trace.honors_discard.store(true, Ordering::Release);
        engine.discard_canceled_cpu_checkpoints().unwrap();
        assert!(!engine.researcher.cpu.token_is_current(&token));
        assert!(engine.semantic.nodes[node].resume.is_none());
        assert_eq!(counters.cpu_nodes, 1);
    }

    #[test]
    fn orphan_frames_and_unknown_disposal_do_not_retire_paused_task_or_raw_evidence() {
        let (mut engine, trace) = engine(LegalOrderRoleMock);
        let node = root(&mut engine, &Position::startpos());
        let mut counters = PalsCounters::default();
        engine
            .verify_checked_cpu(node, &[], limits(), &AtomicBool::new(false), &mut counters)
            .unwrap();
        let (observation, execution) = engine.semantic.nodes[node]
            .evidence
            .as_ref()
            .unwrap()
            .provenance
            .unwrap();
        let token = engine.semantic.nodes[node].resume.take().unwrap();
        let raw = engine
            .semantic
            .stores
            .observations
            .get(observation)
            .unwrap()
            .clone();
        assert!(engine.researcher.cpu.token_is_current(&token));
        // Actual owner survives even though no Node has a logical token.
        trace.honors_discard.store(false, Ordering::Release);
        assert!(matches!(
            engine.synchronize_cpu_checkpoints(),
            Err(PalsError::Checker(CheckerError::Invalid(
                "paused stack disposal retained its owner"
            )))
        ));
        assert!(matches!(
            engine.semantic.stores.tasks.get(execution).unwrap().status,
            TaskStatus::Paused { .. }
        ));
        assert!(
            engine
                .researcher
                .cpu
                .paused_stack_snapshot()
                .unwrap()
                .tokens_retained
                > 0
        );
        trace.honors_discard.store(true, Ordering::Release);
        trace.snapshot_mode.store(1, Ordering::Release);
        assert!(matches!(
            engine.synchronize_cpu_checkpoints(),
            Err(PalsError::Checker(CheckerError::Unsupported(
                "paused stack disposal completion is unknown"
            )))
        ));
        assert!(matches!(
            engine.semantic.stores.tasks.get(execution).unwrap().status,
            TaskStatus::Paused { .. }
        ));
        trace.snapshot_mode.store(2, Ordering::Release);
        assert!(matches!(
            engine.synchronize_cpu_checkpoints(),
            Err(PalsError::Checker(CheckerError::Invalid(
                "paused stack disposal completion is unknown"
            )))
        ));
        assert!(matches!(
            engine.semantic.stores.tasks.get(execution).unwrap().status,
            TaskStatus::Paused { .. }
        ));
        trace.snapshot_mode.store(0, Ordering::Release);
        engine.synchronize_cpu_checkpoints().unwrap();
        let actual = engine.researcher.cpu.paused_stack_snapshot().unwrap();
        assert!(actual.complete);
        assert_eq!((actual.tokens_retained, actual.bytes_current), (0, 0));
        assert!(
            matches!(engine.semantic.stores.tasks.get(execution).unwrap().status,
            TaskStatus::RetiredPaused { evidence: Some(id), .. } if id == observation)
        );
        assert_eq!(
            engine
                .semantic
                .stores
                .observations
                .get(observation)
                .unwrap(),
            &raw
        );
        assert_eq!(counters.cpu_nodes, 1);
    }
}
