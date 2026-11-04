use super::*;
use crate::contracts::SessionScope;
use rz_contracts::rovent::SearchGoal;
use rz_contracts::{CancelToken, GameGeneration, ProcessEpoch, RootGeneration};
use rz_search::cpu::{CpuConfig, CpuEngine};
use rz_search::driver::SearchControl;
use rz_search::pals::engine::{
    DivergenceQuery, LegalOrderRoleMock, ModelValueIdentity, ModelValueOutput, RoleEvaluation,
    RoleQuery,
};
use rz_search::pals::rovent_engine::{PcRegistration, RoventConfig};
use std::sync::atomic::AtomicUsize;

fn config(mode: ExecutionMode) -> RoventConfig {
    RoventConfig {
        mode,
        maximum_goal_plies: NonZeroU16::new(16).unwrap(),
        candidate_width: 1,
        normal_line_plies: NonZeroU16::new(2).unwrap(),
        maximum_positions: 2048,
        maximum_role_calls: 128,
        r_depth: NonZeroU16::new(1).unwrap(),
        r_nodes_per_call: 32,
        selective_policy: Digest([29; 32]),
    }
}

fn settings() -> RoventSessionSettings {
    RoventSessionSettings {
        maximum_role_calls: 128,
        maximum_native_nodes: 512,
    }
}

fn engine<M: RoleModel + 'static>(mode: ExecutionMode, model: M) -> RoventEngine<M> {
    let registration = PcRegistration {
        compute: mode.pc_compute(),
        model_identity: model.identity().into(),
    };
    if mode.has_native_r() {
        RoventEngine::pcr(
            config(mode),
            model,
            registration,
            CpuEngine::new(CpuConfig {
                tt_entries: 0,
                max_depth: 1,
                quiescence_ply: 2,
                ..CpuConfig::default()
            })
            .unwrap(),
        )
        .unwrap()
    } else {
        RoventEngine::pc(config(mode), model, registration).unwrap()
    }
}

fn context<M: RoleModel + 'static>(driver: &RoventSessionDriver<M>) -> SearchSessionContext {
    let authority = SearchAuthority {
        epoch: ProcessEpoch(8101),
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: driver.implementation(),
    };
    let deadline = Instant::now() + Duration::from_secs(8);
    SearchSessionContext {
        authority,
        root_moves: Some(Arc::from([BoardMove::from_uci("e2e4").unwrap()])),
        goal: SearchGoal {
            depth_plies: NonZeroU16::new(3),
            mate_moves: None,
        },
        original_go: None,
        original_time_budget: None,
        original_start: None,
        effective_clock: None,
        requested_nodes: None,
        control: SearchControl::new(deadline, 4096),
        cancellation: CancelToken::new(),
        shutdown_deadline: deadline + Duration::from_secs(1),
        current: Arc::new(Mutex::new(SessionScope::Search(authority))),
    }
}

fn close<M: RoleModel + 'static>(driver: &RoventSessionDriver<M>) {
    assert_eq!(
        driver
            .finish_session(Instant::now() + Duration::from_secs(1))
            .unwrap(),
        Some(DriverPhysicalCompletion::Confirmed)
    );
}

#[test]
fn four_declared_modes_use_mock_roles_and_report_distinct_rovent_work() {
    for mode in [
        ExecutionMode::HybridPcr,
        ExecutionMode::HybridPc,
        ExecutionMode::CpuPcr,
        ExecutionMode::CpuPc,
    ] {
        // Accelerator here is a declaration for the mock, not GPU execution.
        let driver =
            RoventSessionDriver::new(engine(mode, LegalOrderRoleMock), settings()).unwrap();
        assert_eq!(driver.kind(), SearchKind::Rovent);
        assert_eq!(driver.mode(), mode);
        assert_eq!(driver.search_goal_capability(), NonZeroU16::new(16));
        assert!(driver.identity().name.starts_with("Rovent "));
        assert!(driver.work_receipt().unwrap().is_none());
        assert!(driver.pals_checker_registration().is_none());
        let context = context(&driver);
        let mut observed = Vec::new();
        let report = driver
            .run(&Position::startpos(), &context, &mut |movement| {
                observed.push(movement)
            })
            .unwrap();
        let allowed = BoardMove::from_uci("e2e4").unwrap();
        assert_eq!(report.best_move, Some(allowed));
        assert!(!observed.is_empty() && observed.iter().all(|movement| *movement == allowed));
        let DriverWork::Rovent {
            completed_depth,
            native_r_calls,
            native_nodes,
            external_teacher_calls,
            ..
        } = report.work
        else {
            panic!("new engine must not emit PALS/PUCT work")
        };
        assert_eq!(completed_depth, Some(3));
        assert_eq!(external_teacher_calls, 0);
        if mode.has_native_r() {
            assert!(native_r_calls > 0 && native_nodes > 0);
        } else {
            assert_eq!((native_r_calls, native_nodes), (0, 0));
        }
        let latest = driver.latest_result().unwrap().unwrap();
        assert_eq!(latest.authority, context.authority);
        assert!(latest.uci_output_eligible);
        assert_eq!(latest.result.completion, RoventCompletion::DepthGoal);
        close(&driver);
        let shutdown = driver.shutdown_receipt().unwrap().unwrap();
        assert!(shutdown.model_completion_known);
        assert_eq!(shutdown.researcher.is_some(), mode.has_native_r());
        assert!(
            driver
                .run(&Position::startpos(), &context, &mut |_| {})
                .is_err()
        );
    }
}

#[test]
fn original_node_limit_is_not_turned_into_role_calls_or_legacy_simulations() {
    let driver =
        RoventSessionDriver::new(engine(ExecutionMode::CpuPc, LegalOrderRoleMock), settings())
            .unwrap();
    let mut context = context(&driver);
    context.requested_nodes = Some(1);
    context.control.max_simulations = 4096;
    let report = driver
        .run(&Position::startpos(), &context, &mut |_| {})
        .unwrap();
    let DriverWork::Rovent {
        root_entries,
        total_node_visits,
        role_calls,
        native_nodes,
        ..
    } = report.work
    else {
        panic!("Rovent node domain")
    };
    assert_eq!(
        (root_entries, total_node_visits, role_calls, native_nodes),
        (1, 1, 0, 0)
    );
    let latest = driver.latest_result().unwrap().unwrap();
    assert_eq!(latest.result.completion, RoventCompletion::WorkBudget);
    assert_eq!(latest.result.horizon.completed_depth, None);
    close(&driver);
}

#[test]
fn canceled_or_foreign_context_does_not_start_the_engine() {
    let driver =
        RoventSessionDriver::new(engine(ExecutionMode::CpuPc, LegalOrderRoleMock), settings())
            .unwrap();
    let canceled = context(&driver);
    canceled.cancellation.cancel();
    let report = driver
        .run(&Position::startpos(), &canceled, &mut |_| {
            panic!("canceled progress")
        })
        .unwrap();
    assert_eq!(report.best_move, None);
    assert!(driver.latest_result().unwrap().is_none());
    let mut foreign = context(&driver);
    foreign.authority.implementation = Digest([255; 32]);
    let failure = driver
        .run(&Position::startpos(), &foreign, &mut |_| {})
        .unwrap_err();
    assert_eq!(failure.code, "RoventAuthority");
    assert_eq!(
        failure.physical_completion,
        DriverPhysicalCompletion::Confirmed
    );
    close(&driver);
}

struct ProbeModel {
    inner: LegalOrderRoleMock,
    calls: Arc<AtomicUsize>,
    preflights: Arc<AtomicUsize>,
    resets: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    unknown: bool,
    namespace: bool,
    replace_root: Option<Arc<Mutex<SessionScope>>>,
    reset_fail_once: Arc<AtomicBool>,
}

impl Drop for ProbeModel {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::AcqRel);
    }
}

impl RoleModel for ProbeModel {
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn value_identity(&self) -> Option<&ModelValueIdentity> {
        self.inner.value_identity()
    }
    fn requires_registered_representation_namespace(&self) -> bool {
        self.namespace
    }
    fn evaluate_value(&mut self, query: RoleQuery<'_>) -> Result<ModelValueOutput, RoleError> {
        self.inner.evaluate_value(query)
    }
    fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        if self.unknown {
            return Err(RoleError::PhysicalCompletionUnknown);
        }
        if let Some(scope) = self.replace_root.take() {
            let mut scope = scope.lock().unwrap();
            let SessionScope::Search(mut authority) = *scope else {
                panic!("search authority")
            };
            authority.root = RootGeneration(authority.root.0 + 1);
            *scope = SessionScope::Search(authority);
        }
        self.inner.propose(query)
    }
    fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        self.inner.reply(query)
    }
    fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
        self.inner.repair(query)
    }
    fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
        self.inner.divergences(query)
    }
    fn preflight_new_game(
        &mut self,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), RoleError> {
        self.preflights.fetch_add(1, Ordering::AcqRel);
        if self.reset_fail_once.swap(false, Ordering::AcqRel) {
            return Err(RoleError::Deadline);
        }
        self.inner.preflight_new_game(deadline, cancel)
    }
    fn new_game(&mut self) {
        self.resets.fetch_add(1, Ordering::AcqRel);
    }
}

fn probe() -> ProbeModel {
    ProbeModel {
        inner: LegalOrderRoleMock,
        calls: Arc::new(AtomicUsize::new(0)),
        preflights: Arc::new(AtomicUsize::new(0)),
        resets: Arc::new(AtomicUsize::new(0)),
        drops: Arc::new(AtomicUsize::new(0)),
        unknown: false,
        namespace: false,
        replace_root: None,
        reset_fail_once: Arc::new(AtomicBool::new(false)),
    }
}

#[test]
fn revoked_invocation_preserves_reset_intent_without_provider_preflight() {
    let model = probe();
    let preflights = Arc::clone(&model.preflights);
    let resets = Arc::clone(&model.resets);
    let driver = RoventSessionDriver::new(engine(ExecutionMode::CpuPc, model), settings()).unwrap();
    driver.reset_game().unwrap();
    let context = context(&driver);
    let mut replacement = context.authority;
    replacement.root = RootGeneration(replacement.root.0 + 1);
    *context.current.lock().unwrap() = SessionScope::Search(replacement);
    let report = driver
        .run(&Position::startpos(), &context, &mut |_| {
            panic!("revoked progress")
        })
        .unwrap();
    assert_eq!(report.best_move, None);
    assert_eq!(preflights.load(Ordering::Acquire), 0);
    assert_eq!(resets.load(Ordering::Acquire), 0);
    assert!(driver.reset_pending.load(Ordering::Acquire));
    *context.current.lock().unwrap() = SessionScope::Search(context.authority);
    driver
        .run(&Position::startpos(), &context, &mut |_| {})
        .unwrap();
    assert_eq!(preflights.load(Ordering::Acquire), 1);
    assert_eq!(resets.load(Ordering::Acquire), 1);
    close(&driver);
}

#[test]
fn reset_intent_runs_on_the_next_worker_and_failed_preflight_is_retried() {
    let model = probe();
    let resets = Arc::clone(&model.resets);
    model.reset_fail_once.store(true, Ordering::Release);
    let driver = RoventSessionDriver::new(engine(ExecutionMode::CpuPc, model), settings()).unwrap();
    driver.reset_game().unwrap();
    assert_eq!(resets.load(Ordering::Acquire), 0);
    let context = context(&driver);
    let failure = driver
        .run(&Position::startpos(), &context, &mut |_| {})
        .unwrap_err();
    assert_eq!(failure.code, "RoventNewGame");
    assert_eq!(resets.load(Ordering::Acquire), 0);
    driver
        .run(&Position::startpos(), &context, &mut |_| {})
        .unwrap();
    assert_eq!(resets.load(Ordering::Acquire), 1);
    close(&driver);
}

#[test]
fn root_replacement_suppresses_old_progress_and_retains_original_result_authority() {
    let replacement = Arc::new(Mutex::new(SessionScope::Search(SearchAuthority {
        epoch: ProcessEpoch(8101),
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest([0; 32]),
    })));
    let mut model = probe();
    model.replace_root = Some(Arc::clone(&replacement));
    let driver = RoventSessionDriver::new(engine(ExecutionMode::CpuPc, model), settings()).unwrap();
    let mut context = context(&driver);
    *replacement.lock().unwrap() = SessionScope::Search(context.authority);
    context.current = Arc::clone(&replacement);
    let report = driver
        .run(&Position::startpos(), &context, &mut |_| {
            panic!("stale progress")
        })
        .unwrap();
    assert_eq!(report.best_move, None);
    let latest = driver.latest_result().unwrap().unwrap();
    assert_eq!(latest.authority, context.authority);
    assert!(!latest.uci_output_eligible);
    assert_ne!(
        *replacement.lock().unwrap(),
        SessionScope::Search(context.authority)
    );
    close(&driver);
}

#[test]
fn native_namespace_is_rejected_without_plain_model_calls() {
    let mut model = probe();
    model.namespace = true;
    let calls = Arc::clone(&model.calls);
    let failure = match RoventSessionDriver::new(engine(ExecutionMode::CpuPc, model), settings()) {
        Err(failure) => failure,
        Ok(_) => panic!("native provider must use its actual ticket facade"),
    };
    assert_eq!(failure.code, "RoventNativeTicketFacadeUnsupported");
    assert_eq!(calls.load(Ordering::Acquire), 0);
}

#[test]
fn physical_unknown_is_sticky_and_retains_the_owner_instead_of_reusing_or_dropping_it() {
    let mut model = probe();
    model.unknown = true;
    let calls = Arc::clone(&model.calls);
    let drops = Arc::clone(&model.drops);
    let driver = RoventSessionDriver::new(engine(ExecutionMode::CpuPc, model), settings()).unwrap();
    let context = context(&driver);
    let failure = driver
        .run(&Position::startpos(), &context, &mut |_| {})
        .unwrap_err();
    assert_eq!(
        failure.physical_completion,
        DriverPhysicalCompletion::Unknown
    );
    let again = driver
        .run(&Position::startpos(), &context, &mut |_| {})
        .unwrap_err();
    assert_eq!(again.physical_completion, DriverPhysicalCompletion::Unknown);
    assert_eq!(calls.load(Ordering::Acquire), 1);
    assert!(driver.reset_game().is_err());
    assert!(
        driver
            .finish_session(Instant::now() + Duration::from_secs(1))
            .is_err()
    );
    assert!(driver.shutdown_receipt().unwrap().is_none());
    drop(driver);
    assert_eq!(drops.load(Ordering::Acquire), 0);
}
