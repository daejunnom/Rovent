//! Actual CPU owner/native command-loop lifecycle on terminal Rules inputs.
//! Terminal lifecycle, opt-in authored root ranking and full-role goal profile;
//! all role outputs are deterministic mock work with no NN/GPU inference.
use super::*;
use crate::contracts::SessionScope;
use crate::pals_native::ticketed::{
    TicketExecutionEvidence, TicketPhysicalResult, TicketPreparation, TicketServiceLimits,
};
use rz_contracts::pals::{Move16, RepresentationKey, RoleOutput, RolePayload};
use rz_contracts::pals_adaptive::{CpuWorkerId, RoleTicketEnvelope};
use rz_contracts::{
    CancelToken, ExecutionId, GameGeneration, OwnerId, PrecisionProfile, RequestId, RootGeneration,
    Wdl,
};
use rz_runtime::contracts::ContractSystemClock;
use rz_search::cpu::{CpuConfig, CpuEngine, CpuOrderingPolicy, CpuResumePolicy};
use rz_search::cpu_value::BootstrapCpuValue;
use rz_search::cpu_worker::{
    CpuWorkerProfile, cpu_evaluation_namespace_digest, cpu_search_conditions_digest,
};
use rz_search::driver::SearchControl;
use rz_search::pals::coordinator::{PhysicalRoleCompletion, RoleFailure};
use rz_search::pals::engine::adaptive_port::AdaptiveModelNamespace;
use rz_search::pals::engine::{
    PalsConfig, PostRepairRecheckPolicy, ResolverPolicy, RoleQueryPurpose,
};
use rz_search::pals::review_context::{ReviewIndexLimits, ReviewSelectionPolicy};
use rz_search::pals::value::{MODEL_WDL_VALUE_SEMANTICS, ModelValueIdentity};
use sha2::Sha256;

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[test]
fn engine_storage_initial_game_zero_and_next_game_consume_the_actual_mirror_reset_once() {
    // Actual fixture CPU + its original metadata mirror; this does not create
    // an opaque product/storage admission or claim NN/OS acceptance.
    for game in [0, 1] {
        let (driver, facts, clock, actual_owner) = fixture();
        let original = context(&clock, 1, game);
        {
            let mut state = driver.state.lock().unwrap();
            let SessionState::Idle(owners) = &mut *state else {
                panic!("original idle owner")
            };
            let mut initial_bound = false;
            driver
                .prepare_initial_engine_archive_game(
                    owners,
                    &original,
                    game != 0,
                    &mut initial_bound,
                )
                .unwrap();
            assert_eq!(
                owners.cpu[0]
                    .engine
                    .paused_stack_snapshot()
                    .unwrap()
                    .owner_id,
                actual_owner
            );
            let engine = owners.engine.as_mut().unwrap();
            assert_eq!(engine.stores().generation(), 0);
            assert!(engine.stores().root().is_none());
            assert!(engine.last_adaptive_authority().is_none());
            // Same exact ready may be checked twice without another CPU arm.
            engine
                .prepare_unused_archive_game(
                    original.authority,
                    original.control.deadline,
                    &original.control.cancellation,
                )
                .unwrap();
            let mut duplicate = driver.checker_metadata.clone().into_mirror();
            assert!(
                rz_search::cpu_checker::CpuChecker::new_game(
                    &mut duplicate,
                    original.control.deadline,
                    &original.control.cancellation
                )
                .is_err(),
                "the actual mirror reset arm was consumed exactly once"
            );
            let root = engine
                .register_adaptive_root(&Position::startpos(), original.authority)
                .unwrap();
            assert_eq!(engine.last_adaptive_authority(), Some(original.authority));
            engine
                .abandon_unconsumed_adaptive_root(original.authority)
                .unwrap();
            drop(root);
        }
        assert_eq!(facts.role_runs.load(Ordering::Acquire), 0);
        assert!(
            driver
                .finish_process(original.shutdown_deadline)
                .unwrap()
                .permits_process_close()
        );
    }
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[test]
fn engine_storage_unused_selection_closes_after_an_actual_root_without_replacing_cpu() {
    let (driver, facts, clock, actual_owner) = fixture();
    {
        let state = driver.state.lock().unwrap();
        let SessionState::Idle(owners) = &*state else {
            panic!("original idle owner")
        };
        assert!(storage::unused_engine_storage_owner(owners));
        assert_eq!(
            owners.cpu[0]
                .engine
                .paused_stack_snapshot()
                .unwrap()
                .owner_id,
            actual_owner
        );
        assert!(
            driver.storage.lock().unwrap().is_none(),
            "ordinary PALS stays archive-off"
        );
    }
    let root = context(&clock, 1, 0);
    let terminal = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1").unwrap();
    driver.run(&terminal, &root, &mut |_| {}).unwrap();
    {
        let state = driver.state.lock().unwrap();
        let SessionState::Idle(owners) = &*state else {
            panic!("same fenced idle owner")
        };
        assert!(!storage::unused_engine_storage_owner(owners));
        assert_eq!(
            owners.cpu[0]
                .engine
                .paused_stack_snapshot()
                .unwrap()
                .owner_id,
            actual_owner
        );
        assert!(
            owners
                .engine
                .as_ref()
                .unwrap()
                .last_adaptive_authority()
                .is_some()
        );
    }
    assert!(facts.quiesces.load(Ordering::Acquire) > 0);
    assert!(
        driver
            .finish_process(root.shutdown_deadline)
            .unwrap()
            .permits_process_close()
    );
}

#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
#[test]
fn engine_storage_rejects_legacy_budget_without_issuing_a_replacement_original_go() {
    let legacy = RoventPcrRootAdmission {
        budget: Digest([19; 32]),
        original: None,
        original_v2: None,
    };
    for admission in [None, Some(&legacy)] {
        let rejected = storage::original_storage_admission(admission).unwrap_err();
        assert_eq!(rejected.code, "RoventEngineStorageAdmission");
        assert_eq!(legacy.budget, Digest([19; 32]));
        assert!(legacy.original.is_none());
    }
}

#[test]
fn failed_root_closure_keeps_first_cause_without_confirming_physical_completion() {
    let original = [
        AdaptiveManagerError::Semantic("original external response failure".into()),
        AdaptiveManagerError::Capacity("later cleanup failure"),
    ];
    let failure = failed_root_closure(original.first());
    assert_eq!(failure.code, "AdaptiveRootClosure");
    assert!(
        failure
            .detail
            .contains("original external response failure")
    );
    assert!(!failure.detail.contains("later cleanup failure"));
    assert_eq!(
        failure.physical_completion,
        crate::search_driver::DriverPhysicalCompletion::Unknown
    );
}

#[test]
fn failed_root_closure_bounds_original_unicode_cause() {
    let original = AdaptiveManagerError::Semantic("원인".repeat(1024));
    let failure = failed_root_closure(Some(&original));
    assert_eq!(failure.code, "AdaptiveRootClosure");
    assert!(failure.detail.len() <= 512 + " [cause truncated]".len());
    assert!(failure.detail.ends_with(" [cause truncated]"));
    assert!(failure.detail.contains("원인"));
    assert_eq!(
        failure.physical_completion,
        crate::search_driver::DriverPhysicalCompletion::Unknown
    );
}

#[test]
fn failed_root_closure_without_cause_keeps_original_unknown_failure() {
    let failure = failed_root_closure(None);
    let original = SearchSessionFailure::physical_completion_unknown();
    assert_eq!(failure.code, original.code);
    assert_eq!(failure.detail, original.detail);
    assert_eq!(failure.physical_completion, original.physical_completion);
}

#[derive(Default)]
struct OwnerFacts {
    factories: AtomicU64,
    quiesces: AtomicU64,
    finishes: AtomicU64,
    finish_ok: AtomicBool,
    root_ranking: AtomicBool,
    full_roles: AtomicBool,
    role_runs: AtomicU64,
    repair_role_runs: AtomicU64,
    capture_role_sources: AtomicBool,
    role_sources: Mutex<Vec<(RequestId, RoleQueryPurpose)>>,
}
struct TerminalOwner {
    clock: ContractSystemClock,
    facts: Arc<OwnerFacts>,
}
impl TicketNativeBackend<ReviewView, AdaptivePortQuestion> for TerminalOwner {
    type Prepared = RepresentationKey;
    type Output = RoleOutput;
    fn clock(&self) -> ContractSystemClock {
        self.clock.clone()
    }
    fn prepare(
        &mut self,
        question: &TicketPreparation<ReviewView, AdaptivePortQuestion>,
        _: &AtomicBool,
    ) -> Result<(Self::Prepared, RepresentationKey), RoleFailure> {
        if !self.facts.full_roles.load(Ordering::Acquire)
            && (!self.facts.root_ranking.load(Ordering::Acquire)
                || !question.view.query().prefix.is_empty())
        {
            return Err(RoleFailure::Backend(
                "terminal lifecycle fixture forbids non-root model preparation".into(),
            ));
        }
        let mut candidates = Sha256::new();
        for movement in &question.view.query().legal {
            candidates.update(
                Move16::encode(rz_contracts::Move::try_from(*movement).unwrap())
                    .bits()
                    .to_le_bytes(),
            );
        }
        let key = RepresentationKey {
            input: Digest(question.view.digest()),
            history: question.seal.root_state.semantic,
            candidates: Digest(candidates.finalize().into()),
            mask_positions: Digest([4; 32]),
            model: Digest([5; 32]),
            encoding: Digest([6; 32]),
            precision: PrecisionProfile::Fp32,
            frozen_epoch: 1,
            record_revision: question.query.context.public_revision,
            role: question.role,
        };
        Ok((key, key))
    }
    fn prepared_host_bytes(&self, _: &Self::Prepared) -> u64 {
        std::mem::size_of::<RepresentationKey>() as u64
    }
    fn run(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, AdaptivePortQuestion>,
        _: &Self::Prepared,
        _: &AtomicBool,
    ) -> TicketPhysicalResult<RoleOutput> {
        self.facts.role_runs.fetch_add(1, Ordering::AcqRel);
        if self.facts.capture_role_sources.load(Ordering::Acquire) {
            self.facts
                .role_sources
                .lock()
                .unwrap()
                .push((ticket.id, question.query.context.purpose));
        }
        if question.query.context.purpose == rz_search::pals::engine::RoleQueryPurpose::RepairPolicy
        {
            self.facts.repair_role_runs.fetch_add(1, Ordering::AcqRel);
        }
        TicketPhysicalResult {
            physical: PhysicalRoleCompletion::Confirmed,
            output: Ok(RoleOutput {
                id: ticket.id,
                authority: ticket.authority,
                key: ticket.representation,
                payload: {
                    let candidate_logits = (0..question.view.query().legal.len())
                        .map(|i| i as f32)
                        .collect();
                    let wdl = Wdl::try_new(0.25, 0.5, 0.25, 1e-4).unwrap();
                    match question.role {
                        rz_contracts::pals::Role::Proposer => RolePayload::Proposal {
                            candidate_logits,
                            wdl,
                        },
                        rz_contracts::pals::Role::Critic => RolePayload::Counterexample {
                            candidate_logits,
                            divergence_logits: vec![0.; question.view.query().divergences.len()],
                            wdl,
                        },
                        rz_contracts::pals::Role::Validator => {
                            panic!("V is not enabled in this P/C lifecycle fixture")
                        }
                    }
                },
                private_latent: vec![0.; 16 * 384],
            }),
        }
    }
    fn output_host_bytes(&self, output: &RoleOutput) -> u64 {
        let values = match &output.payload {
            RolePayload::Proposal {
                candidate_logits, ..
            } => candidate_logits.len(),
            RolePayload::Counterexample {
                candidate_logits,
                divergence_logits,
                ..
            } => candidate_logits.len() + divergence_logits.len(),
            RolePayload::TaskRanking { task_logits } => task_logits.len(),
        };
        ((values + output.private_latent.len()) * 4) as u64
    }
    fn acceptance_output_digest(&self, output: &Self::Output) -> Option<Digest> {
        rz_search::pals::coordinator::native_acceptance_output_digest(output).ok()
    }
    fn execution_evidence(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Option<TicketExecutionEvidence> {
        Some(TicketExecutionEvidence {
            execution: Some(ExecutionId {
                epoch: ticket.id.epoch,
                sequence: ticket.id.sequence,
            }),
            // Authored mock service accounting, never actual NN/GPU evidence.
            completed_nn_inputs: Some(1),
            measurement_complete: true,
            physical_fresh: Some(true),
            accounting: None,
        })
    }
    fn accept(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, AdaptivePortQuestion>,
        _: &Self::Prepared,
        output: &RoleOutput,
    ) -> Result<(), RoleFailure> {
        if output.id != ticket.id
            || output.authority != ticket.authority
            || output.key != ticket.representation
            || ticket.view.view_digest.0 != question.view.seal_digest()
        {
            return Err(RoleFailure::InvalidOutput);
        }
        Ok(())
    }
    fn discard(&mut self, _: &RoleTicketEnvelope) -> Result<(), RoleFailure> {
        Ok(())
    }
    fn quiesce(&mut self, until: Instant) -> bool {
        self.facts.quiesces.fetch_add(1, Ordering::AcqRel);
        Instant::now() < until
    }
    fn finish(&mut self, _: Instant) -> bool {
        self.facts.finishes.fetch_add(1, Ordering::AcqRel);
        self.facts.finish_ok.load(Ordering::Acquire)
    }
}
fn fixture() -> (
    AdaptiveSessionDriver<TerminalOwner>,
    Arc<OwnerFacts>,
    ProcessClock,
    u64,
) {
    fixture_with_startup_gate(None)
}
fn fixture_with_startup_gate(
    startup_gate: Option<std::sync::mpsc::Receiver<()>>,
) -> (
    AdaptiveSessionDriver<TerminalOwner>,
    Arc<OwnerFacts>,
    ProcessClock,
    u64,
) {
    fixture_with_role_budget(startup_gate, None)
}
fn fixture_with_role_budget(
    startup_gate: Option<std::sync::mpsc::Receiver<()>>,
    role_limit: Option<u64>,
) -> (
    AdaptiveSessionDriver<TerminalOwner>,
    Arc<OwnerFacts>,
    ProcessClock,
    u64,
) {
    fixture_with_rovent_mode(startup_gate, role_limit, None)
}
fn fixture_with_rovent_mode(
    startup_gate: Option<std::sync::mpsc::Receiver<()>>,
    role_limit: Option<u64>,
    rovent_mode: Option<rz_contracts::rovent::ExecutionMode>,
) -> (
    AdaptiveSessionDriver<TerminalOwner>,
    Arc<OwnerFacts>,
    ProcessClock,
    u64,
) {
    fixture_with_rovent_profile(startup_gate, role_limit, rovent_mode, 1, false)
}
fn fixture_with_rovent_profile(
    startup_gate: Option<std::sync::mpsc::Receiver<()>>,
    role_limit: Option<u64>,
    rovent_mode: Option<rz_contracts::rovent::ExecutionMode>,
    line_plies: usize,
    full_roles: bool,
) -> (
    AdaptiveSessionDriver<TerminalOwner>,
    Arc<OwnerFacts>,
    ProcessClock,
    u64,
) {
    fixture_with_rovent_profile_and_peer_sources(
        startup_gate,
        role_limit,
        rovent_mode,
        line_plies,
        full_roles,
        1,
        PalsConfig::default().max_records,
    )
}
fn fixture_with_rovent_profile_and_peer_sources(
    startup_gate: Option<std::sync::mpsc::Receiver<()>>,
    role_limit: Option<u64>,
    rovent_mode: Option<rz_contracts::rovent::ExecutionMode>,
    line_plies: usize,
    full_roles: bool,
    peer_sources: u16,
    max_records: usize,
) -> (
    AdaptiveSessionDriver<TerminalOwner>,
    Arc<OwnerFacts>,
    ProcessClock,
    u64,
) {
    fixture_with_rovent_profile_and_review_limits(
        startup_gate,
        role_limit,
        rovent_mode,
        line_plies,
        full_roles,
        peer_sources,
        (max_records, ReviewIndexLimits::default()),
    )
}
fn fixture_with_rovent_profile_and_review_limits(
    startup_gate: Option<std::sync::mpsc::Receiver<()>>,
    role_limit: Option<u64>,
    rovent_mode: Option<rz_contracts::rovent::ExecutionMode>,
    line_plies: usize,
    full_roles: bool,
    peer_sources: u16,
    record_limits: (usize, ReviewIndexLimits),
) -> (
    AdaptiveSessionDriver<TerminalOwner>,
    Arc<OwnerFacts>,
    ProcessClock,
    u64,
) {
    let (max_records, review_limits) = record_limits;
    let clock = ProcessClock::new(ProcessEpoch(31));
    let maximum_goal_plies = std::num::NonZeroU16::new(8).unwrap();
    let facts = Arc::new(OwnerFacts::default());
    facts.finish_ok.store(true, Ordering::Release);
    facts.full_roles.store(full_roles, Ordering::Release);
    facts
        .root_ranking
        .store(role_limit.is_some(), Ordering::Release);
    let factory_facts = Arc::clone(&facts);
    let native_clock = clock.runtime_clock();
    let native_factory = move || {
        factory_facts.factories.fetch_add(1, Ordering::AcqRel);
        if let Some(gate) = startup_gate {
            gate.recv_timeout(Duration::from_secs(3)).map_err(|_| {
                RoleFailure::Backend("terminal fixture startup gate expired".into())
            })?;
        }
        Ok(TerminalOwner {
            clock: native_clock,
            facts: factory_facts,
        })
    };
    let native = match rovent_mode {
        Some(_) => TicketedNativeService::spawn_with_capacity(
            native_factory,
            TicketServiceLimits::default(),
            AdaptiveExecutionLimits::default().role_tickets_max,
            AdaptiveExecutionLimits::default().worker_commands_max,
        ),
        None => TicketedNativeService::spawn(native_factory, TicketServiceLimits::default()),
    }
    .unwrap();
    let cpu = CpuEngine::with_evaluator_ordering_and_resume_policy(
        CpuConfig {
            tt_entries: 16,
            max_depth: 2,
            quiescence_ply: 2,
            ..Default::default()
        },
        Arc::new(BootstrapCpuValue::default()),
        CpuOrderingPolicy::LegacyMvvLvaV1,
        CpuResumePolicy::PausedStack,
    )
    .unwrap();
    let actual_owner = cpu.paused_stack_snapshot().unwrap().owner_id;
    let profile = CpuWorkerProfile {
        owner: CpuWorkerId(0),
        max_depth: cpu.config().max_depth,
        evaluation_namespace: cpu_evaluation_namespace_digest(&cpu).unwrap(),
        search_conditions: cpu_search_conditions_digest(&cpu),
    };
    let metadata = AdaptiveCheckerMetadata::capture_owned(&cpu).unwrap();
    let value = ModelValueIdentity {
        semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
        model: "terminal-owned-fixture".into(),
        encoding: "terminal-fixture-v3".into(),
        precision: "fp32".into(),
        model_epoch: [5; 32],
    };
    let namespace = AdaptiveModelNamespace::new(
        value.clone(),
        Digest([5; 32]),
        Digest([6; 32]),
        PrecisionProfile::Fp32,
        1,
    )
    .unwrap();
    let rovent_namespace = Arc::new(namespace.clone());
    let model =
        AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value.clone()), namespace)
            .unwrap();
    let registration = PalsCheckerRegistration {
        identity: metadata.identity().clone(),
        conditions: metadata.conditions().into(),
        capabilities: metadata.capabilities(),
        owned_descriptor: metadata.owned_descriptor(),
        cpu_resume_policy: metadata.resume_policy(),
        role_model: value.model.clone(),
        model_value: full_roles.then_some(value),
        resolver_version: if full_roles {
            rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION
        } else {
            rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION
        },
        resolver_semantics: if full_roles {
            rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS
        } else {
            rz_search::pals::engine::PALS_VALUE_RESOLVER_SEMANTICS
        },
    };
    let config = PalsConfig {
        beam_width: 1,
        line_plies,
        max_records,
        cpu_nodes_per_task: 8,
        max_role_calls: role_limit.unwrap_or_else(|| PalsConfig::default().max_role_calls),
        ..Default::default()
    };
    let resolver = if full_roles {
        ResolverPolicy::ModelWdlRestricted
    } else {
        ResolverPolicy::OwnRawRestricted
    };
    let recheck = PostRepairRecheckPolicy::IterativeFrozenModelWdlV2;
    let engine = if rovent_mode.is_some() {
        PalsEngine::new_with_boxed_checker_and_rovent_profile(
            rz_search::pals::engine::ResearcherSemanticProfile {
                config,
                maximum_root_prefix_plies: maximum_goal_plies.get(),
            },
            model,
            Box::new(metadata.clone().into_mirror()),
            resolver,
            recheck,
        )
    } else {
        PalsEngine::new_with_boxed_checker_and_policies(
            config,
            model,
            Box::new(metadata.clone().into_mirror()),
            resolver,
            recheck,
        )
    }
    .unwrap()
    .with_selected_review_context(ReviewSelectionPolicy::default(), review_limits)
    .unwrap();
    let next = Arc::new(AtomicU64::new(1));
    let issuer: AdaptiveRulesOwnerIssuer =
        Arc::new(move || Ok(OwnerId(next.fetch_add(1, Ordering::AcqRel))));
    let options = AdaptiveSessionOptions {
        implementation: Digest([4; 32]),
        identity: EngineIdentity {
            name: "PALS terminal lifecycle fixture".into(),
            author: "RoveZero contributors".into(),
        },
        max_rounds: 1,
        max_cpu_nodes: 64,
        cpu_depth: 1,
        token_budget: 512,
        policy: CoordinatorPolicy::Fixed,
        endpoints: CpuEndpointSelection::deployment(),
        controller: ControllerConfig::default(),
        device_digest: Digest([7; 32]),
        cpu_device_digest: Digest([8; 32]),
        external_policy: ExternalSlicePolicy::default(),
    };
    let sources = AdaptiveSessionSources {
        native,
        engine,
        cpu: vec![PersistentCpuOwner {
            profile,
            engine: cpu,
        }],
        external: None,
        checker_metadata: metadata,
    };
    let driver = match rovent_mode {
        Some(mode) => AdaptiveSessionDriver::new_rovent_pcr_with_resource_limits(
            options,
            AdaptiveExecutionLimits::default(),
            sources,
            clock.clone(),
            issuer,
            registration,
            RoventPcrSessionProfile {
                mode,
                maximum_goal_plies,
                input_lock: Digest([9; 32]),
                root_budget: Digest([10; 32]),
                selective_policy: Digest([11; 32]),
                model_namespace: rovent_namespace,
                frontier_limits: rz_search::pals::rovent_frontier::FrontierLimits {
                    maximum_goal_plies,
                    maximum_frames: 16,
                    maximum_paths: 64,
                    maximum_mate_nodes: 128,
                    maximum_native_receipts: 16,
                },
                reciprocal: crate::pals_adaptive::RoventReciprocalPolicy {
                    mode,
                    root_budget: Digest([10; 32]),
                    recipient: rz_contracts::rovent::SearchRecipient::Proposer,
                    max_questions_per_job: 1,
                    node_interval: 1,
                    first_question_sequence: 1,
                },
                peers: crate::pals_adaptive::RoventPeerFollowupPolicy {
                    pc_peer: true,
                    researcher: true,
                    max_sources_per_root: peer_sources,
                },
            },
        ),
        None => AdaptiveSessionDriver::new(
            options,
            sources,
            clock.clone(),
            issuer,
            registration,
            None,
            None,
        ),
    }
    .unwrap_or_else(|rejected| panic!("{:?}", rejected.failure));
    (driver, facts, clock, actual_owner)
}
fn context(clock: &ProcessClock, root: u64, game: u64) -> SearchSessionContext {
    let now = Instant::now();
    let authority = SearchAuthority {
        epoch: clock.epoch(),
        game: GameGeneration(game),
        root: RootGeneration(root),
        implementation: Digest([4; 32]),
    };
    SearchSessionContext {
        authority,
        root_moves: None,
        goal: rz_contracts::rovent::SearchGoal::default(),
        original_go: None,
        original_time_budget: None,
        original_start: None,
        effective_clock: None,
        requested_nodes: None,
        control: SearchControl::new(now + Duration::from_secs(5), 64),
        cancellation: CancelToken::new(),
        shutdown_deadline: now + Duration::from_secs(6),
        current: Arc::new(Mutex::new(SessionScope::Search(authority))),
    }
}

#[test]
fn rovent_initial_nodes_boundary_quiesces_without_a_role_ticket_or_physical_failure() {
    use rz_contracts::rovent::ExecutionMode;
    use rz_search::pals::rovent_frontier::{FrontierGoalStatus, FrontierStop};

    let position = Position::startpos();
    let allowed: BoardMove = "e2e4".parse().unwrap();
    // Zero admits no root. One pays for the root only. Two exactly pays for
    // this restricted root's one checked child. None may submit P ranking.
    for nodes in [0, 1, 2] {
        let (driver, facts, clock, _) =
            fixture_with_rovent_mode(None, None, Some(ExecutionMode::CpuPcr));
        let mut original = context(&clock, 1, 0);
        original.requested_nodes = Some(nodes);
        original.root_moves = Some(Arc::from([allowed]));
        let mut published = Vec::new();
        let report = driver
            .run(&position, &original, &mut |movement| {
                published.push(movement)
            })
            .unwrap();
        assert_eq!(report.best_move, Some(allowed));
        assert!(published.iter().all(|movement| *movement == allowed));
        assert_eq!(facts.role_runs.load(Ordering::Acquire), 0);
        let actual = driver.latest_rovent_root().unwrap().unwrap();
        assert_eq!(actual.authority, original.authority);
        let frontier = actual.frontier.as_ref().unwrap();
        assert_eq!(frontier.requested_nodes, Some(nodes));
        assert_eq!(
            frontier.goal,
            FrontierGoalStatus::Stopped(FrontierStop::WorkBudget)
        );
        assert_eq!(frontier.work.total_node_visits(), Some(nodes));
        assert_eq!(frontier.horizon.completed_depth, None);
        assert_eq!(actual.counters.submitted_role_tickets, 0);
        assert!(actual.physical.as_ref().unwrap().permits_root_reuse());
        assert!(matches!(
            report.work,
            DriverWork::RoventNative {
                total_node_visits: Some(actual_nodes),
                submitted_role_tickets: 0,
                completed_depth: None,
                ..
            } if actual_nodes == nodes
        ));
        assert!(driver.work_receipt().unwrap().is_none());
        assert!(driver.pals_checker_registration().is_none());
        assert_eq!(
            driver.finish_session(original.shutdown_deadline).unwrap(),
            Some(crate::search_driver::DriverPhysicalCompletion::Confirmed)
        );
        assert_eq!(facts.finishes.load(Ordering::Acquire), 1);
    }
}

#[test]
fn rovent_original_depth_one_and_two_close_actual_p_c_fresh_goal() {
    use rz_contracts::rovent::ExecutionMode;
    use rz_search::pals::rovent_frontier::FrontierGoalStatus;

    for (line_plies, depth) in [(4, 1), (4, 2), (2, 3), (4, 3)] {
        let (driver, facts, clock, _) = fixture_with_rovent_profile(
            None,
            Some(64),
            Some(ExecutionMode::CpuPcr),
            line_plies,
            true,
        );
        let position = Position::startpos();
        let allowed: BoardMove = "e2e4".parse().unwrap();
        let mut original = context(&clock, 1, 0);
        original.goal.depth_plies = std::num::NonZeroU16::new(depth);
        original.root_moves = Some(Arc::from([allowed]));
        let report = driver
            .run(&position, &original, &mut |_| {})
            .unwrap_or_else(|error| {
                panic!(
                    "original go depth {depth}, chunk {line_plies} failed: {error:?}; actual root: {:?}",
                    driver.latest_rovent_root(),
                )
            });
        let actual = driver.latest_rovent_root().unwrap().unwrap();
        let frontier = actual.frontier.as_ref().unwrap();
        assert_eq!(frontier.goal, FrontierGoalStatus::DepthSatisfied);
        assert_eq!(frontier.horizon.completed_depth.unwrap().0, depth);
        assert_eq!(frontier.horizon.maximum_examined_ply.0, depth);
        if depth == 2 {
            // The CPU-discovered counter and the accepted C counter select
            // different actual sibling lines under the same original frame.
            // Both must be declared and completed in addition to Candidate.
            assert_eq!(frontier.horizon.declared_obligations, 3);
            assert_eq!(frontier.horizon.completed_obligations, 3);
            assert_eq!(frontier.registered_frames, 3);
            assert_eq!(frontier.completed_frames, 3);
        }
        if line_plies == 4 && depth == 3 {
            // Each actual selected counter has a nonzero accepted P Repair.
            // The second Repair must own another duty under the same frame,
            // rather than closing the first Repair's completed source again.
            assert_eq!(facts.repair_role_runs.load(Ordering::Acquire), 2);
            assert_eq!(frontier.horizon.declared_obligations, 5);
            assert_eq!(frontier.horizon.completed_obligations, 5);
            assert_eq!(frontier.registered_frames, 5);
            assert_eq!(frontier.completed_frames, 5);
        } else {
            assert_eq!(facts.repair_role_runs.load(Ordering::Acquire), 0);
        }
        assert!(actual.counters.consumed_role_outputs >= 2);
        assert!(facts.role_runs.load(Ordering::Acquire) >= 2);
        assert!(actual.physical.as_ref().unwrap().permits_root_reuse());
        assert!(matches!(report.work, DriverWork::RoventNative {
            completed_depth: Some(completed), maximum_examined_ply, ..
        } if completed == depth && maximum_examined_ply == depth));
        assert_eq!(
            driver.finish_session(original.shutdown_deadline).unwrap(),
            Some(crate::search_driver::DriverPhysicalCompletion::Confirmed),
        );
    }
}

#[test]
fn rovent_selected_counter_keeps_actual_parent_budget_after_peer_credit_spend() {
    use crate::pals_adaptive::PeerFollowupDisposition;
    use rz_contracts::rovent::ExecutionMode;
    use rz_search::pals::rovent_frontier::FrontierGoalStatus;

    let (mut driver, facts, clock, _) = fixture_with_rovent_profile_and_peer_sources(
        None,
        Some(64),
        Some(ExecutionMode::CpuPcr),
        4,
        true,
        4,
        PalsConfig::default().max_records,
    );
    facts.capture_role_sources.store(true, Ordering::Release);
    let selected_source_peers = Arc::new(AtomicU64::new(0));
    let observed_peers = Arc::clone(&selected_source_peers);
    let observed_facts = Arc::clone(&facts);
    driver.collection_hooks.root_closure = Some(Arc::new(move |manager, _| {
        let sources = observed_facts.role_sources.lock().unwrap();
        let actual = manager
            .peer_followup_observations()
            .filter(|observation| {
                observation.disposition == PeerFollowupDisposition::Admitted
                    && sources.iter().any(|(request, purpose)| {
                        *request == observation.source
                            && matches!(
                                purpose,
                                RoleQueryPurpose::ReplyPolicy | RoleQueryPurpose::RepairPolicy
                            )
                    })
            })
            .count();
        observed_peers.store(actual as u64, Ordering::Release);
        Ok(())
    }));
    let position = Position::startpos();
    let first = BoardMove::from_uci("e2e4").unwrap();
    let mut original = context(&clock, 1, 0);
    original.goal.depth_plies = std::num::NonZeroU16::new(3);
    original.root_moves = Some(Arc::from([first]));
    let report = driver
        .run(&position, &original, &mut |_| {})
        .unwrap_or_else(|error| {
            panic!(
                "actual Counter/Repair peer credit root failed: {error:?}; actual root: {:?}",
                driver.latest_rovent_root(),
            )
        });
    assert!(
        selected_source_peers.load(Ordering::Acquire) > 0,
        "the actual selected Counter/Repair source must dispatch a real peer child",
    );
    let actual = driver.latest_rovent_root().unwrap().unwrap();
    let frontier = actual.frontier.as_ref().unwrap();
    assert_eq!(frontier.goal, FrontierGoalStatus::DepthSatisfied);
    assert_eq!(frontier.horizon.completed_depth.unwrap().0, 3);
    assert_eq!(frontier.horizon.maximum_examined_ply.0, 3);
    assert!(frontier.registered_frames >= 3);
    assert_eq!(frontier.registered_frames, frontier.completed_frames);
    assert!(actual.physical.as_ref().unwrap().permits_root_reuse());
    assert!(matches!(
        report.work,
        DriverWork::RoventNative {
            completed_depth: Some(3),
            maximum_examined_ply: 3,
            ..
        }
    ));
    assert_eq!(
        driver.finish_session(original.shutdown_deadline).unwrap(),
        Some(crate::search_driver::DriverPhysicalCompletion::Confirmed),
    );
}

#[test]
fn rovent_game_record_capacity_keeps_evidence_and_closes_each_root_without_depth() {
    assert_rovent_game_record_capacity(4, ReviewIndexLimits::default());
}

#[test]
fn rovent_same_bundle_capacity_keeps_evidence_and_closes_each_root_without_depth() {
    assert_rovent_game_record_capacity(
        PalsConfig::default().max_records,
        ReviewIndexLimits {
            records_per_bundle: 4,
            ..ReviewIndexLimits::default()
        },
    );
}

fn assert_rovent_game_record_capacity(max_records: usize, review_limits: ReviewIndexLimits) {
    use rz_contracts::rovent::ExecutionMode;
    use rz_search::pals::rovent_frontier::{FrontierGoalStatus, FrontierStop};

    // Use the existing real owner/manager/Rules/Fresh fixture. Each completed
    // depth-one root publishes one actual Proposal; do not prefill fake rows.
    let (driver, facts, clock, _) = fixture_with_rovent_profile_and_review_limits(
        None,
        Some(64),
        Some(ExecutionMode::CpuPcr),
        1,
        true,
        1,
        (max_records, review_limits),
    );
    let retained_records = || {
        let state = driver.state.lock().unwrap();
        let SessionState::Idle(owners) = &*state else {
            panic!("the original root did not return its actual idle owners");
        };
        owners
            .engine
            .as_ref()
            .expect("original semantic owner after confirmed root fence")
            .records()
            .iter()
            .map(|record| {
                // critical is a root-relative ReviewView projection. Keep the
                // complete original raw identity, relationships and evidence.
                (
                    record.revision,
                    record.parent_revision,
                    record.supersedes_revision,
                    record.origin_state,
                    record.kind,
                    record.line.clone(),
                    record.value,
                    record.completed_depth,
                    record.score_scope,
                    record.cpu_observation,
                    record.perspective,
                )
            })
            .collect::<Vec<_>>()
    };
    let position = Position::startpos();
    let first: BoardMove = "e2e4".parse().unwrap();
    let mut full_records = None;
    let mut final_until = None;
    for root in 1..=6 {
        // No ucinewgame: raw evidence is game-wide and must survive each root.
        let mut original = context(&clock, root, 0);
        original.goal.depth_plies = std::num::NonZeroU16::new(1);
        original.root_moves = Some(Arc::from([first]));
        let report = driver
            .run(&position, &original, &mut |_| {})
            .unwrap_or_else(|error| {
                panic!("actual record/bundle-bound root {root}: {error:?}; limits={max_records}/{review_limits:?}")
            });
        final_until = Some(original.shutdown_deadline);
        assert_eq!(report.best_move, Some(first));
        let actual = driver.latest_rovent_root().unwrap().unwrap();
        let frontier = actual.frontier.as_ref().unwrap();
        assert!(actual.physical.as_ref().unwrap().permits_root_reuse());
        assert!(!driver.closed.load(Ordering::Acquire));
        if root <= 4 {
            assert_eq!(frontier.goal, FrontierGoalStatus::DepthSatisfied);
            assert_eq!(frontier.horizon.completed_depth.unwrap().0, 1);
            assert_eq!(retained_records().len(), root as usize);
            if root == 4 {
                full_records = Some(retained_records());
            }
        } else {
            assert_eq!(
                frontier.goal,
                FrontierGoalStatus::Stopped(FrontierStop::Capacity),
            );
            assert_eq!(frontier.horizon.completed_depth, None);
            assert!(!frontier.horizon.target_satisfied);
            assert!(matches!(
                report.work,
                DriverWork::RoventNative {
                    completed_depth: None,
                    ..
                }
            ));
            assert_eq!(&retained_records(), full_records.as_ref().unwrap());
            // The original ranking is consumed before candidate publication.
            // Capacity rejects that publication, not the real consumption;
            // draining must neither erase it nor consume the output again.
            assert_eq!(actual.counters.submitted_role_tickets, 1);
            assert_eq!(actual.counters.physical_role_completions, 1);
            assert_eq!(actual.counters.consumed_role_outputs, 1);
            assert_eq!(actual.counters.unused_role_results, 0);
            let last = driver.completion_diagnostic().unwrap().last.unwrap();
            assert_eq!(last.semantic_completion, Some(PalsCompletion::Capacity));
            assert_eq!(last.driver_return, AdaptiveCompletionReturn::Ok);
            assert!(last.failure.is_none());
        }
    }
    assert!(facts.role_runs.load(Ordering::Acquire) > 0);
    assert_eq!(
        driver.finish_session(final_until.unwrap()).unwrap(),
        Some(crate::search_driver::DriverPhysicalCompletion::Confirmed),
    );
    assert_eq!(facts.finishes.load(Ordering::Acquire), 1);
}

#[test]
fn rovent_initial_rules_mate_closes_before_root_ranking() {
    use rz_contracts::rovent::ExecutionMode;
    use rz_search::pals::rovent_frontier::FrontierGoalStatus;

    let (driver, facts, clock, _) =
        fixture_with_rovent_mode(None, None, Some(ExecutionMode::CpuPcr));
    let position = Position::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1").unwrap();
    let allowed: BoardMove = "g6g7".parse().unwrap();
    assert!(position.legal_moves().contains(&allowed));
    let mut original = context(&clock, 1, 0);
    original.goal.mate_moves = std::num::NonZeroU16::new(1);
    original.root_moves = Some(Arc::from([allowed]));
    let report = driver.run(&position, &original, &mut |_| {}).unwrap();
    assert_eq!(report.best_move, Some(allowed));
    assert_eq!(facts.role_runs.load(Ordering::Acquire), 0);
    let actual = driver.latest_rovent_root().unwrap().unwrap();
    let frontier = actual.frontier.as_ref().unwrap();
    assert_eq!(frontier.goal, FrontierGoalStatus::MateSatisfied);
    assert_eq!(frontier.work.root_entries, 1);
    // The normal checked child and the two OR/AND mate lanes each perform
    // their own Rules replay. Those real visits remain separately charged.
    assert_eq!(frontier.work.checked_transitions, 3);
    assert_eq!(actual.counters.submitted_role_tickets, 0);
    assert!(actual.physical.as_ref().unwrap().permits_root_reuse());
    assert_eq!(
        driver.finish_session(original.shutdown_deadline).unwrap(),
        Some(crate::search_driver::DriverPhysicalCompletion::Confirmed)
    );
}

#[test]
fn actual_session_projects_final_role_limit_without_a_caller_supplied_reason() {
    let (driver, facts, clock, _) = fixture_with_role_budget(None, Some(1));
    let position = Position::startpos();
    let first = context(&clock, 1, 0);
    let report = driver.run(&position, &first, &mut |_| {}).unwrap();
    assert!(position.legal_moves().contains(&report.best_move.unwrap()));
    assert_eq!(facts.role_runs.load(Ordering::Acquire), 1);
    let actual = driver.completion_diagnostic().unwrap();
    let last = actual.last.unwrap();
    assert_eq!(last.authority, first.authority);
    assert_eq!(
        last.semantic_completion,
        Some(PalsCompletion::RoleCallLimit)
    );
    assert_eq!(last.driver_return, AdaptiveCompletionReturn::Ok);
    assert!(last.failure.is_none());
    assert_eq!(
        actual
            .successful_driver_return_reason_counts
            .unwrap()
            .role_call_limit,
        1
    );
    assert_eq!(actual.observed_driver_returns, Some(1));
    assert!(
        driver
            .latest_root_receipt()
            .unwrap()
            .unwrap()
            .permits_root_reuse()
    );
    assert!(
        driver
            .finish_process(first.shutdown_deadline)
            .unwrap()
            .permits_process_close()
    );
}

#[test]
fn initial_game_failure_guard_closes_admission_and_retains_owner() {
    let (driver, facts, clock, _) = fixture();
    let driver = Arc::new(driver);
    let mate = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1").unwrap();
    let context = context(&clock, 1, 0);
    let until = context.shutdown_deadline;
    let state = driver.state.lock().unwrap();
    // Inject only the failed-initial-binding close boundary. This verifies
    // sticky close and owner retention; begin alone does not prove that the
    // other run has already passed its pre-lock check and started waiting.
    let failed_initial = InitialGameFailureGuard {
        closed: &driver.closed,
        armed: true,
    };
    let waiting_driver = Arc::clone(&driver);
    let waiting = std::thread::spawn(move || {
        waiting_driver.run(&mate, &context, &mut |_| {
            panic!("closed waiting root must never publish a move")
        })
    });
    while driver.work_receipt().unwrap().unwrap().active_invocations != 1 {
        assert!(Instant::now() < until);
        std::thread::yield_now();
    }
    drop(failed_initial);
    assert!(driver.closed.load(Ordering::Acquire));
    drop(state);
    let error = waiting.join().unwrap().unwrap_err();
    assert_eq!(error.code, "PhysicalCompletionUnknown");
    assert!(driver.latest_root_receipt().unwrap().is_none());
    assert_eq!(facts.quiesces.load(Ordering::Acquire), 0);
    let closed = driver.finish_process(until).unwrap();
    assert!(closed.permits_process_close());
    assert_eq!(facts.finishes.load(Ordering::Acquire), 1);
}

#[test]
fn terminal_roots_and_new_game_preserve_actual_session_and_cpu_owner_then_finish_once() {
    let (driver, facts, clock, actual_owner) = fixture();
    let mate = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1").unwrap();
    assert!(mate.legal_moves().is_empty());
    for (root, game) in [(1, 0), (2, 0), (3, 1)] {
        let context = context(&clock, root, game);
        let report = driver
            .run(&mate, &context, &mut |_| {
                panic!("terminal Rules state has no bestmove")
            })
            .unwrap();
        assert_eq!(report.best_move, None);
        let completion = driver.completion_diagnostic().unwrap();
        let last = completion.last.as_ref().unwrap();
        assert_eq!(last.authority, context.authority);
        assert_eq!(last.semantic_completion, Some(PalsCompletion::Terminal));
        assert_eq!(last.driver_return, AdaptiveCompletionReturn::Ok);
        assert_eq!(completion.observed_driver_returns, Some(root));
        assert_eq!(
            completion
                .successful_driver_return_reason_counts
                .unwrap()
                .terminal,
            root
        );
        let receipt = driver.latest_root_receipt().unwrap().unwrap();
        assert_eq!(receipt.authority, context.authority);
        assert!(receipt.permits_root_reuse());
        assert!(!receipt.permits_archive_close());
        assert_eq!(facts.factories.load(Ordering::Acquire), 1);
        assert_eq!(facts.finishes.load(Ordering::Acquire), 0);
        let state = driver.state.lock().unwrap();
        let SessionState::Idle(owners) = &*state else {
            panic!("not reusable")
        };
        assert_eq!(owners.cpu.len(), 1);
        assert_eq!(
            owners.cpu[0]
                .engine
                .paused_stack_snapshot()
                .unwrap()
                .owner_id,
            actual_owner
        );
        drop(state);
        driver
            .accept_uci_output(context.authority, "0000", true)
            .unwrap();
        assert!(
            driver
                .accept_uci_output(context.authority, "0000", true)
                .is_err()
        );
    }
    let until = Instant::now() + Duration::from_secs(3);
    let closed = driver.finish_process(until).unwrap();
    assert!(closed.permits_process_close());
    assert_eq!(closed.epoch(), clock.epoch());
    assert_eq!(facts.factories.load(Ordering::Acquire), 1);
    assert_eq!(facts.quiesces.load(Ordering::Acquire), 3);
    assert_eq!(facts.finishes.load(Ordering::Acquire), 1);
    assert!(
        driver
            .finish_process(until)
            .unwrap()
            .permits_process_close()
    );
    assert_eq!(facts.finishes.load(Ordering::Acquire), 1);
}
#[test]
fn actual_native_ready_after_terminal_quiesce_keeps_revoked_admission_closed_and_owner_reusable() {
    let (release, wait) = std::sync::mpsc::sync_channel(1);
    let (mut driver, facts, clock, _) = fixture_with_startup_gate(Some(wait));
    let mate = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1").unwrap();
    let first = context(&clock, 1, 0);
    let until = first.shutdown_deadline;
    driver.collection_hooks.root_admission = Some(Arc::new(move |manager| {
        // The actual factory cannot publish Ready until the exact terminal
        // Rules root has revoked admission. No sleeps or scheduling assumption
        // decide which event wins this regression.
        manager.driver_mut().prepare_root_rules()?;
        assert!(manager.driver().root_is_terminal()?);
        manager.begin_root_quiesce(until)?;
        release.send(()).map_err(|_| {
            AdaptiveManagerError::Invalid("terminal fixture startup gate receiver lost")
        })?;
        Ok(())
    }));
    let report = driver
        .run(&mate, &first, &mut |_| {
            panic!("terminal Rules root has no bestmove")
        })
        .unwrap();
    assert_eq!(report.best_move, None);
    assert!(
        driver
            .latest_root_receipt()
            .unwrap()
            .unwrap()
            .permits_root_reuse()
    );
    assert_eq!(facts.factories.load(Ordering::Acquire), 1);
    assert_eq!(facts.finishes.load(Ordering::Acquire), 0);
    driver
        .accept_uci_output(first.authority, "0000", true)
        .unwrap();

    // A new root installs the persistent owner's actual verifier through normal
    // Running admission, without starting another native factory or NN call.
    driver.collection_hooks.root_admission = None;
    let second = context(&clock, 2, 0);
    assert_eq!(
        driver
            .run(&mate, &second, &mut |_| panic!(
                "terminal Rules root has no bestmove"
            ))
            .unwrap()
            .best_move,
        None
    );
    assert!(
        driver
            .latest_root_receipt()
            .unwrap()
            .unwrap()
            .permits_root_reuse()
    );
    assert_eq!(facts.factories.load(Ordering::Acquire), 1);
    assert!(
        driver
            .finish_process(second.shutdown_deadline)
            .unwrap()
            .permits_process_close()
    );
    assert_eq!(facts.finishes.load(Ordering::Acquire), 1);
}
#[test]
fn final_semantic_completion_survives_a_later_collection_failure() {
    let (mut driver, _, clock, _) = fixture();
    driver.closure_observer = Some(Arc::new(|_| {
        Err(AdaptiveManagerError::Invalid(
            "fixture after final semantic result",
        ))
    }));
    let mate = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1").unwrap();
    let first = context(&clock, 1, 0);
    let failure = driver
        .run(&mate, &first, &mut |_| panic!("terminal root"))
        .unwrap_err();
    assert_eq!(failure.code, "AdaptiveCollectionClosure");
    let actual = driver.completion_diagnostic().unwrap();
    assert_eq!(actual.semantic_observed_but_driver_failed, Some(1));
    assert_eq!(
        actual
            .successful_driver_return_reason_counts
            .unwrap()
            .terminal,
        0
    );
    let last = actual.last.unwrap();
    assert_eq!(last.semantic_completion, Some(PalsCompletion::Terminal));
    assert_eq!(last.driver_return, AdaptiveCompletionReturn::Err);
    let observed_failure = last.failure.unwrap();
    assert_eq!(observed_failure.code, failure.code);
    assert_eq!(
        observed_failure.physical_completion,
        failure.physical_completion
    );

    // A later root rejected before admission must not inherit Terminal.
    let second = context(&clock, 2, 0);
    second.cancellation.cancel();
    assert!(
        driver
            .run(&mate, &second, &mut |_| panic!("canceled root"))
            .is_err()
    );
    let actual = driver.completion_diagnostic().unwrap();
    assert_eq!(actual.last.unwrap().semantic_completion, None);
    assert_eq!(actual.unobserved_completion_returns, Some(1));
    assert!(
        driver
            .finish_process(first.shutdown_deadline)
            .unwrap()
            .permits_process_close()
    );
}

#[test]
fn terminal_port_reason_is_preserved_when_final_selector_and_work_observe_cancel() {
    let (mut driver, _, clock, _) = fixture();
    let mate = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1").unwrap();
    let first = context(&clock, 1, 0);
    let canceled = Arc::clone(&first.control.cancellation);
    driver.collection_hooks.root_admission = Some(Arc::new(move |manager| {
        manager.driver_mut().prepare_root_rules()?;
        assert!(manager.driver().root_is_terminal()?);
        canceled.store(true, Ordering::Release);
        Ok(())
    }));
    assert!(
        driver
            .run(&mate, &first, &mut |_| panic!("terminal root"))
            .is_ok()
    );
    let actual = driver.completion_diagnostic().unwrap();
    assert_eq!(
        actual.last.unwrap().semantic_completion,
        Some(PalsCompletion::Terminal)
    );
    let counts = actual.successful_driver_return_reason_counts.unwrap();
    assert_eq!(counts.terminal, 1);
    assert_eq!(counts.canceled, 0);
    assert_eq!(driver.work_receipt().unwrap().unwrap().canceled_returns, 1);
    assert!(
        driver
            .finish_process(first.shutdown_deadline)
            .unwrap()
            .permits_process_close()
    );
}

#[test]
fn poisoned_completion_diagnostic_does_not_change_search_or_physical_finish() {
    let (driver, _, clock, _) = fixture();
    std::thread::scope(|scope| {
        assert!(
            scope
                .spawn(|| {
                    let _guard = driver.completion.lock().unwrap();
                    panic!("fixture poisons only the optional diagnostic owner");
                })
                .join()
                .is_err()
        );
    });
    let mate = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1").unwrap();
    let first = context(&clock, 1, 0);
    assert!(
        driver
            .run(&mate, &first, &mut |_| panic!("terminal root"))
            .is_ok()
    );
    assert!(driver.completion_diagnostic().is_err());
    assert!(
        driver
            .finish_process(first.shutdown_deadline)
            .unwrap()
            .permits_process_close()
    );
}

#[test]
fn physical_finish_unknown_is_retained_and_never_a_process_receipt() {
    let (driver, facts, _, _) = fixture();
    facts.finish_ok.store(false, Ordering::Release);
    let failure = driver
        .finish_process(Instant::now() + Duration::from_secs(3))
        .unwrap_err();
    assert_eq!(
        failure.physical_completion,
        crate::search_driver::DriverPhysicalCompletion::Unknown
    );
    assert!(!matches!(
        &*driver.state.lock().unwrap(),
        SessionState::Closed(_)
    ));
    assert_eq!(facts.finishes.load(Ordering::Acquire), 1);
    assert_eq!(
        driver.retained_owner_status().unwrap().retained_cpu_engines,
        1
    );
}
