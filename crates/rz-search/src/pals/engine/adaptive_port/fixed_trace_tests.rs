//! Synchronous callback versus fixed-port prepared-input trace checks.
//!
//! These are component checks, not a substitute for an actual manager-driven
//! full-search trace. The CPU check below uses a real private worker receipt;
//! no completed CPU or native neural receipt is fabricated for equivalence.
use super::*;
use crate::contract_time::InstantClock;
use crate::cpu::{CpuConfig, CpuResumePolicy};
use crate::cpu_worker::{CpuWorker, CpuWorkerEvent, cpu_root_restriction_digest};
use crate::pals::coordinator::FrameId;
use crate::pals::review_context::{ReviewIndexLimits, ReviewRecordMetadata, ReviewSelectionPolicy};
use rz_contracts::pals_adaptive::{CpuJobEnvelope, CpuJobId, CpuWorkerId, SliceGrant};
use rz_contracts::{ClockDomain, GameGeneration, MonotonicTick, ProcessEpoch, RootGeneration};

#[derive(Debug, PartialEq)]
struct InputTrace {
    context: RoleLogicalContext,
    focus_history: Vec<String>,
    purpose: RoleQueryPurpose,
    prefix: Vec<BoardMove>,
    proposal: Vec<BoardMove>,
    counter: Option<Vec<BoardMove>>,
    legal: Vec<BoardMove>,
    divergences: Vec<usize>,
    records: Vec<String>,
    relations: Vec<ReviewRecordMetadata>,
    anchor_histories: Vec<Vec<String>>,
    input: [u8; 32],
    record_scope: [u8; 32],
    context_policy: [u8; 32],
    tokens: u16,
    bucket: u16,
}
impl InputTrace {
    fn new(position: &Position, context: &RoleLogicalContext, view: &ReviewView) -> Self {
        let history = |snapshot: &PositionSnapshot| snapshot.known_history_fens();
        let query = view.query();
        Self {
            context: context.clone(),
            focus_history: history(&position.snapshot()),
            purpose: query.purpose,
            prefix: query.prefix.clone(),
            proposal: query.proposal.clone(),
            counter: query.counterexample.clone(),
            legal: query.legal.clone(),
            divergences: query.divergences.clone(),
            records: view
                .records()
                .iter()
                .map(|record| format!("{record:?}"))
                .collect(),
            relations: view.records_metadata().to_vec(),
            anchor_histories: view
                .anchors()
                .iter()
                .map(|anchor| history(&anchor.snapshot))
                .collect(),
            input: view.digest(),
            record_scope: view.record_scope_digest(),
            context_policy: view.policy_digest(),
            tokens: view.token_count(),
            bucket: view.length_bucket(),
        }
    }
    fn prepared(input: &FrameInput<ReviewView, AdaptivePortQuestion>) -> Self {
        assert_eq!(input.public_context.0, input.view.record_scope_digest());
        Self::new(
            &input.question.position,
            &input.question.context,
            &input.view,
        )
    }
}

#[derive(Default)]
struct RecordingMock {
    inner: LegalOrderRoleMock,
    pending: Option<(Arc<ReviewView>, RoleLogicalContext)>,
    traces: Vec<InputTrace>,
}
impl RecordingMock {
    fn observe(&mut self, query: &RoleQuery<'_>) -> Result<(), RoleError> {
        let (view, context) = self.pending.take().ok_or(RoleError::InvalidOutput)?;
        let prepared = view.query();
        if prepared.prefix != query.prefix
            || prepared.proposal != query.proposal
            || prepared.counterexample.as_deref() != query.counterexample
            || prepared.legal != query.legal
            || format!("{:?}", view.records()) != format!("{:?}", query.records)
            || context.public_revision != query.revision
        {
            return Err(RoleError::InvalidOutput);
        }
        self.traces
            .push(InputTrace::new(query.position, &context, &view));
        Ok(())
    }
}
impl RoleModel for RecordingMock {
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn supports_selected_review_view(&self) -> bool {
        self.inner.supports_selected_review_view()
    }
    fn value_identity(&self) -> Option<&ModelValueIdentity> {
        self.inner.value_identity()
    }
    fn bind_selected_review_view(
        &mut self,
        view: Arc<ReviewView>,
        context: &RoleLogicalContext,
    ) -> Result<(), RoleError> {
        self.inner
            .bind_selected_review_view(Arc::clone(&view), context)?;
        self.pending = Some((view, context.clone()));
        Ok(())
    }
    fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        self.observe(&query)?;
        self.inner.propose(query)
    }
    fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        self.observe(&query)?;
        self.inner.reply(query)
    }
    fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        self.observe(&query)?;
        self.inner.repair(query)
    }
    fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
        self.inner.divergences(query)
    }
    fn evaluate_value(&mut self, query: RoleQuery<'_>) -> Result<ModelValueOutput, RoleError> {
        self.observe(&query)?;
        self.inner.evaluate_value(query)
    }
}
type Port = AdaptiveEnginePort<RecordingMock>;

fn cpu_config() -> CpuConfig {
    CpuConfig {
        tt_entries: 16,
        max_depth: 8,
        ..Default::default()
    }
}
fn setup(position: Position, limits: PalsLimits) -> Port {
    let cpu = CpuEngine::with_resume_policy(cpu_config(), CpuResumePolicy::PausedStack).unwrap();
    let engine = PalsEngine::new(PalsConfig::default(), RecordingMock::default(), cpu).unwrap();
    setup_port(engine, position, limits)
}
fn setup_model_value(position: Position, limits: PalsLimits) -> Port {
    let cpu = CpuEngine::with_resume_policy(cpu_config(), CpuResumePolicy::PausedStack).unwrap();
    // Owned-raw startup deliberately does not register a model value namespace.
    // This Fresh fixture selects the WDL resolver so the real mock identity is
    // validated and frozen at construction, before either endpoint query.
    let engine = PalsEngine::new_with_boxed_checker_and_policies(
        PalsConfig::default(),
        RecordingMock::default(),
        Box::new(OwnedCpuChecker::new(Box::new(cpu)).unwrap()),
        ResolverPolicy::ModelWdlRestricted,
        PostRepairRecheckPolicy::Disabled,
    )
    .unwrap();
    assert_eq!(
        engine.semantic.model_registered_value.as_ref(),
        engine.semantic.model.value_identity()
    );
    assert!(engine.semantic.model_registered_value.is_some());
    setup_port(engine, position, limits)
}
fn setup_port(engine: PalsEngine<RecordingMock>, position: Position, limits: PalsLimits) -> Port {
    let engine = engine
        .with_selected_review_context(
            ReviewSelectionPolicy::default(),
            ReviewIndexLimits::default(),
        )
        .unwrap();
    let authority = SearchAuthority {
        epoch: ProcessEpoch(23),
        game: GameGeneration(0),
        root: RootGeneration(1),
        implementation: ContractDigest([7; 32]),
    };
    let deadline = Deadline {
        clock: ClockDomain(authority.epoch),
        at: MonotonicTick(engine.tick_at(limits.deadline)),
    };
    let next = Arc::new(std::sync::atomic::AtomicU64::new(1));
    let issuer: AdaptiveRulesOwnerIssuer =
        Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::Relaxed))));
    let mut port = Port::new(engine, position, authority, deadline, limits, issuer).unwrap();
    port.configure_root_fanout(AdaptiveRootFanoutPolicy::SequentialFixed)
        .unwrap();
    port
}
fn limits() -> PalsLimits {
    PalsLimits {
        deadline: Instant::now() + Duration::from_secs(20),
        max_rounds: 1,
        max_cpu_nodes: 2000,
        cpu_depth: 1,
    }
}
fn line(moves: &[&str]) -> Vec<BoardMove> {
    moves
        .iter()
        .map(|mv| BoardMove::from_uci(mv).unwrap())
        .collect()
}

/// Authored public observations make ordering/relations nontrivial. They do
/// not claim any physical CPU or NN completion.
fn publish_fixture(port: &mut Port, synchronous: bool) {
    for (kind, moves) in [
        (RecordKind::Proposal, line(&["e2e4", "e7e5"])),
        (RecordKind::Counterexample, line(&["e2e4", "c7c5"])),
        (RecordKind::Repair, line(&["e2e4", "e7e6"])),
    ] {
        port.replay_node(&moves).unwrap();
        if synchronous {
            port.engine
                .record_checked(
                    kind,
                    &moves,
                    None,
                    0,
                    None,
                    None,
                    port.limits,
                    &AtomicBool::new(false),
                )
                .unwrap();
        } else {
            port.record_hot(kind, &moves, None).unwrap();
        }
    }
}

#[test]
fn fixed_prepared_policy_queries_match_real_synchronous_callbacks() {
    // Private counter bookkeeping must not leak into Reply model inputs.
    for (stage, prefix, proposal, counter, call) in [
        (
            RefinementStage::ProposerRanking,
            vec![],
            vec![],
            None,
            Call::Propose,
        ),
        (
            RefinementStage::ProposalContinuation,
            line(&["e2e4"]),
            vec![],
            None,
            Call::Propose,
        ),
        (
            RefinementStage::CriticReply,
            line(&["e2e4"]),
            line(&["e2e4", "e7e5"]),
            Some(line(&["e2e4", "c7c5"])),
            Call::Reply,
        ),
        (
            RefinementStage::CounterContinuation,
            line(&["e2e4", "c7c5"]),
            line(&["e2e4", "e7e5"]),
            Some(line(&["e2e4", "c7c5"])),
            Call::Reply,
        ),
        (
            RefinementStage::Repair,
            line(&["e2e4", "c7c5"]),
            line(&["e2e4", "e7e5"]),
            Some(line(&["e2e4", "c7c5"])),
            Call::Repair,
        ),
    ] {
        let limits = limits();
        let mut baseline = setup(Position::startpos(), limits);
        let mut fixed = setup(Position::startpos(), limits);
        publish_fixture(&mut baseline, true);
        publish_fixture(&mut fixed, false);
        let old_node = baseline.replay_node(&prefix).unwrap();
        let new_node = fixed.replay_node(&prefix).unwrap();
        let model_counter = if matches!(
            stage,
            RefinementStage::CriticReply | RefinementStage::CounterContinuation
        ) {
            None
        } else {
            counter.as_deref()
        };
        let mut counters = PalsCounters::default();
        baseline
            .engine
            .ranked(
                old_node,
                call,
                &prefix,
                &proposal,
                model_counter,
                limits,
                &AtomicBool::new(false),
                &mut counters,
            )
            .unwrap();
        let input = fixed
            .input(
                new_node,
                stage,
                prefix.first().copied(),
                prefix,
                proposal,
                counter.clone(),
                None,
                None,
                false,
                None,
                None,
            )
            .unwrap();
        assert_eq!(
            input.question.counter, counter,
            "private branch continuation was lost"
        );
        assert_eq!(baseline.engine.semantic.model.traces.len(), 1);
        assert_eq!(
            baseline.engine.semantic.model.traces[0],
            InputTrace::prepared(&input),
            "stage {stage:?}"
        );
        assert_eq!(
            counters.completed_proposer_calls
                + counters.completed_critic_calls
                + counters.completed_repair_calls,
            1
        );
        assert_eq!(counters.consumed_role_outputs, 1);
        // A prepared fixed input has not executed or consumed a physical role.
        assert_eq!(fixed.counters.role_calls, 0);
    }
}

#[test]
fn fixed_fresh_endpoint_uses_endpoint_line_not_private_branch_metadata() {
    for stage in [
        RefinementStage::CounterValueFresh,
        RefinementStage::RepairValueFresh,
    ] {
        let limits = limits();
        let mut baseline = setup_model_value(Position::startpos(), limits);
        let mut fixed = setup_model_value(Position::startpos(), limits);
        publish_fixture(&mut baseline, true);
        publish_fixture(&mut fixed, false);
        let endpoint = line(&["e2e4", "c7c5"]);
        let old_node = baseline.replay_node(&endpoint).unwrap();
        let new_node = fixed.replay_node(&endpoint).unwrap();
        let mut counters = PalsCounters::default();
        baseline
            .engine
            .evaluate_model_value(
                old_node,
                &endpoint,
                limits,
                &AtomicBool::new(false),
                &mut counters,
            )
            .unwrap();
        let input = fixed
            .input(
                new_node,
                stage,
                endpoint.first().copied(),
                endpoint.clone(),
                line(&["e2e4", "e7e5"]),
                Some(endpoint.clone()),
                None,
                None,
                false,
                None,
                None,
            )
            .unwrap();
        assert_eq!(input.view.query().proposal, endpoint);
        assert!(input.view.query().counterexample.is_none());
        assert_eq!(baseline.engine.semantic.model.traces.len(), 1);
        assert_eq!(
            baseline.engine.semantic.model.traces[0],
            InputTrace::prepared(&input),
            "stage {stage:?}"
        );
        assert_eq!(
            (
                counters.completed_value_calls,
                counters.accepted_value_outputs
            ),
            (1, 1)
        );
        assert_eq!(fixed.counters.value_calls, 0);
    }
}

#[test]
fn actual_private_cpu_worker_preserves_verification_work_and_next_public_view() {
    let limits = limits();
    let mut baseline = setup(Position::startpos(), limits);
    let mut fixed = setup(Position::startpos(), limits);
    let prefix = line(&["e2e4"]);
    for port in [&mut baseline, &mut fixed] {
        port.replay_node(&prefix).unwrap();
    }
    baseline
        .engine
        .record_checked(
            RecordKind::Proposal,
            &prefix,
            None,
            0,
            None,
            None,
            limits,
            &AtomicBool::new(false),
        )
        .unwrap();
    fixed
        .record_hot(RecordKind::Proposal, &prefix, None)
        .unwrap();
    let old_node = baseline.replay_node(&prefix).unwrap();
    let new_node = fixed.replay_node(&prefix).unwrap();
    let mut old_work = PalsCounters::default();
    baseline
        .engine
        .verify_checked_cpu(
            old_node,
            &prefix,
            limits,
            &AtomicBool::new(false),
            &mut old_work,
        )
        .unwrap();

    let input = fixed
        .input(
            new_node,
            RefinementStage::CpuValidation,
            prefix.first().copied(),
            prefix.clone(),
            prefix.clone(),
            None,
            None,
            None,
            false,
            None,
            None,
        )
        .unwrap();
    let work = CoordinatorWork {
        frame: FrameId {
            authority: fixed.authority,
            sequence: 1,
        },
        original_first_move: rz_contracts::pals::Move16::encode(
            rz_contracts::Move::try_from(prefix[0]).unwrap(),
        ),
        original_deadline: fixed.deadline,
        budget: FrameBudget {
            original_cpu_credit: fixed.limits.max_cpu_nodes,
            role_submissions_remaining: fixed.engine.config.max_role_calls,
            original_store_byte_limit: 8 * 1024 * 1024,
        },
        input,
    };
    let worker = CpuWorker::spawn(
        CpuWorkerId(1),
        CpuEngine::with_resume_policy(cpu_config(), CpuResumePolicy::PausedStack).unwrap(),
    )
    .unwrap();
    let prepared = fixed.cpu_question(&work, &[worker.profile()]).unwrap();
    let job = CpuJobEnvelope {
        id: CpuJobId {
            authority: fixed.authority,
            sequence: 1,
        },
        state_identity: prepared.position.export().unwrap().snapshot().identity(),
        question: work.input.question_digest,
        root_restriction: cpu_root_restriction_digest(prepared.root_moves.as_deref()),
        evaluation_namespace: worker.profile().evaluation_namespace,
        search_conditions: worker.profile().search_conditions,
        total_native_node_credit: prepared.total_native_node_credit,
        hard_deadline: fixed.deadline,
    };
    let grant = SliceGrant {
        job: job.id,
        execution: rz_contracts::ExecutionId::new(fixed.authority.epoch, 1),
        owner: worker.profile().owner,
        node_credit: job.total_native_node_credit,
        soft_lease: Duration::from_secs(1),
        control_sequence: 1,
    };
    let clock = InstantClock::new(
        ClockDomain(fixed.authority.epoch),
        fixed.engine.semantic.clock_origin,
    );
    let physical_job = worker
        .prepare_job(
            job,
            prepared.position,
            prepared.root_moves,
            prepared.target_depth,
            &clock,
        )
        .unwrap();
    fixed.cpu_admitted(&work, &job, &grant).unwrap();
    worker.begin(physical_job, grant).unwrap();
    let receipt = loop {
        assert!(
            Instant::now() < limits.deadline,
            "real CPU worker did not finish"
        );
        match worker.try_event().unwrap() {
            Some(CpuWorkerEvent::Slice(report)) => break report,
            Some(_) => panic!("unexpected terminal event before CPU slice"),
            None => std::thread::park_timeout(Duration::from_millis(1)),
        }
    };
    // The finite depth-one workload completes inside the single full-credit
    // slice. A yield is a failed equivalence fixture, never synthetic success.
    assert!(!receipt.continuation_retained);
    let (observation, complete) = fixed.commit_cpu_report(&work, &receipt).unwrap();
    assert!(complete);
    let transition = fixed.after_cpu(&work, observation, complete).unwrap();
    let transition = fixed.materialize_step(transition).unwrap();
    assert!(matches!(
        transition.transition,
        FrameTransition::Complete {
            scope: FrameCompletionScope::NoDivergence
        }
    ));
    let shutdown = worker.shutdown().unwrap();
    assert!(shutdown.thread_joined);
    assert!(!shutdown.thread_panicked);
    assert!(
        shutdown.events.is_empty(),
        "CPU fixture left an unsettled receipt"
    );
    let a = baseline.engine.semantic.nodes[old_node]
        .evidence
        .as_ref()
        .unwrap();
    let b = fixed.engine.semantic.nodes[new_node]
        .evidence
        .as_ref()
        .unwrap();
    assert_eq!(
        (a.score, a.depth, a.scope, &a.value_identity),
        (b.score, b.depth, b.scope, &b.value_identity)
    );
    assert_eq!(
        (
            old_work.cpu_tasks_requested,
            old_work.cpu_tasks,
            old_work.cpu_nodes,
            old_work.cpu_quiescence_nodes,
            old_work.cpu_tt_hits,
            old_work.completed_cpu_tasks,
            old_work.consumed_cpu_tasks
        ),
        (
            fixed.counters.cpu_tasks_requested,
            fixed.counters.cpu_tasks,
            fixed.counters.cpu_nodes,
            fixed.counters.cpu_quiescence_nodes,
            fixed.counters.cpu_tt_hits,
            fixed.counters.completed_cpu_tasks,
            fixed.counters.consumed_cpu_tasks
        )
    );
    assert_eq!(
        baseline.engine.value(old_node, 0),
        fixed.engine.value(new_node, 0)
    );
    assert_eq!(
        baseline.engine.semantic.nodes[baseline.root].edges.len(),
        fixed.engine.semantic.nodes[fixed.root].edges.len()
    );
    assert_eq!(
        baseline
            .engine
            .semantic
            .records
            .iter()
            .map(|r| format!("{r:?}"))
            .collect::<Vec<_>>(),
        fixed
            .engine
            .semantic
            .records
            .iter()
            .map(|r| format!("{r:?}"))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        baseline.engine.semantic.revision,
        fixed.engine.semantic.revision
    );

    // Inspect a subsequent real synchronous Reply callback against a fixed
    // prepared input after the same CPU publication. This one-ply frame has
    // no divergence, so this is a model-input component check, not a claim
    // that its actual refinement continuation dispatched a CriticReply.
    baseline
        .engine
        .ranked(
            old_node,
            Call::Reply,
            &prefix,
            &prefix,
            None,
            limits,
            &AtomicBool::new(false),
            &mut old_work,
        )
        .unwrap();
    let next = fixed
        .input(
            new_node,
            RefinementStage::CriticReply,
            prefix.first().copied(),
            prefix.clone(),
            prefix.clone(),
            Some(prefix),
            None,
            None,
            false,
            None,
            None,
        )
        .unwrap();
    assert_eq!(
        baseline.engine.semantic.model.traces.last().unwrap(),
        &InputTrace::prepared(&next)
    );
}
