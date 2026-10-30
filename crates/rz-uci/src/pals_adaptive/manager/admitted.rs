//! Infallible owner transfer after source-kind-specific admission. Native and
//! profiled entry points reuse the same Store/archive/ticket lifecycle fields.
use super::*;

pub(super) struct ManagerAssembly<Q, O> {
    pub cpu_clock: InstantClock,
    pub deadline: Deadline,
    pub native_ready: bool,
    pub registry: RoleTicketRegistry<ReviewView, Arc<O>>,
    pub native_verifier_registered: bool,
    pub machine: RefinementMachine<ReviewView, Q>,
    pub ready: ReadySelector<FrameId, Move16, Digest>,
    pub costs: CostEstimator,
    pub researcher: ResearcherResources,
    pub role_call_limit: Option<u64>,
    pub next_job: u64,
    pub next_execution: u64,
    pub next_control: u64,
    pub cleanup_boundary: Option<CleanupBoundary>,
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    pub(super) fn assemble(
        registration: AdaptiveManagerRegistration,
        clock: ContractSystemClock,
        limits: AdaptiveExecutionLimits,
        sources: AdaptiveManagerSources<Q, B, D>,
        prepared: ManagerAssembly<Q, B::Output>,
    ) -> Self {
        let ManagerAssembly {
            cpu_clock,
            deadline,
            native_ready,
            registry,
            native_verifier_registered,
            machine,
            ready,
            costs,
            researcher,
            role_call_limit,
            next_job,
            next_execution,
            next_control,
            cleanup_boundary,
        } = prepared;
        let AdaptiveManagerSources {
            native,
            own_workers,
            external,
            driver,
        } = sources;
        Self {
            registration,
            limits,
            clock,
            cpu_clock,
            deadline,
            native,
            registry,
            machine,
            driver,
            role_call_limit,
            collection_observer: None,
            collection_admission: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            collection_observer_v2: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            collection_v2_disabled: false,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            collection_v2_disabled_first_ticket: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            collection_v2_busy_ticket: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            collection_v2_first_failure: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            collection_v2_abandon_custody: Vec::with_capacity(limits.role_tickets_max),
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            collection_v2_disable_failed: false,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            collection_admission_v2: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            collection_v2_settlement: None,
            roles: Vec::with_capacity(limits.role_tickets_max),
            acknowledgements: VecDeque::with_capacity(limits.role_tickets_max),
            own: own_workers
                .into_iter()
                .map(|worker| OwnSlot {
                    worker: Some(worker),
                    binding: None,
                    joined: false,
                    panicked: false,
                    settled_execution: None,
                    recovered_engine: None,
                    ready_readback: None,
                    last_value_installation: None,
                    nnuec_retirement: None,
                })
                .collect(),
            external,
            foreign: None,
            researcher,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            profiled_binding: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            profiled_orphan: None,
            cpu_slice_diagnostic: OnceLock::new(),
            #[cfg(feature = "pals-adaptive-diagnostics")]
            continuation_diagnostic: None,
            ready,
            costs,
            advised_yield: None,
            observed_role_classes: Vec::with_capacity(64),
            observation_sequence: 0,
            next_job,
            next_execution,
            next_control,
            phase: ManagerPhase::Running,
            native_finished: None,
            native_joined: false,
            native_root_fence: None,
            close_mode: AdaptiveCloseMode::ProcessShutdown,
            native_root_quiesce_requested: false,
            external_finished: None,
            external_root_fence: None,
            external_startup_observed: false,
            counters: AdaptiveManagerCounters::default(),
            native_accounting: AdaptiveNativeAccountingSnapshot::default(),
            shutdown_until: None,
            cleanup_boundary,
            effective_clock: None,
            native_ready,
            native_verifier_registered,
            retained_native_permits: Vec::with_capacity(limits.role_tickets_max),
            retained_native_receipts: Vec::with_capacity(limits.role_tickets_max),
            failures: VecDeque::with_capacity(64),
            quarantine_rejections: Vec::with_capacity(16),
            retained_native_completions: Vec::with_capacity(limits.role_tickets_max),
            native_observation_failures: Vec::new(),
            quarantined_cpu_receipts: Vec::with_capacity(8),
            quarantined_foreign_receipts: Vec::with_capacity(1),
            ready_ids: Vec::with_capacity(limits.waiting_jobs_max),
            deferred: VecDeque::with_capacity(128),
            native_frontier_events: VecDeque::with_capacity(32),
            review_leases: RoventReviewLeaseObservation::new(limits.role_tickets_max),
            validator_waiting: Vec::with_capacity(limits.waiting_jobs_max),
            root_started: false,
            cpu_node_limit_reached: false,
            root_ranking_proof: None,
            exclusive_requested: false,
            registered_action: None,
            exclusive_scope: None,
            finished_at: None,
            reciprocal_policy: None,
            reciprocal_routing: None,
            next_reciprocal_question: 1,
            peer_policy: None,
            accepted_roles: Vec::with_capacity(MAX_ROLE_PENDING),
            peer_sources_admitted: 0,
            peer_observations: Vec::with_capacity(128),
            goal_prepared: None,
            goal_continuation_pending: false,
            nnuec: None,
            nnuec_pending: None,
            nnuec_jobs: Vec::with_capacity(CPU_JOB_CAPACITY),
            nnuec_observations: Vec::with_capacity(CPU_JOB_CAPACITY),
            nnuec_observations_omitted: 0,
            cpu_semantic: None,
            utility_route: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            utility_cohort: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            utility_intervention: None,
            utility_route_sequence: 0,
            retained_cpu_jobs: Vec::new(),
        }
    }
}
