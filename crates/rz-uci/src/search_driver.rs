//! Startup-selected search sessions below UCI. A session returns only after its
//! owned work has completed or been drained; publication remains with UCI.
//! CPU/PALS counters are never translated into PUCT simulations or NN receipts.

use crate::{EngineIdentity, contracts::SessionScope};
use rz_contracts::{CancelToken, ContractError, Digest, ErrorCode, Stage};
use rz_position::{BoardMove, Position};
use rz_search::cpu_checker::{
    CheckerAttempt, CheckerCapabilities, CheckerIdentity, CheckerShutdown, CpuChecker,
    ExternalUciIdentity, OwnedCheckerDescriptor, OwnedCpuChecker,
};
use rz_search::driver::SearchControl;
use rz_search::pals::engine::{ModelValueIdentity, PostRepairRecheckPolicy, ResolverPolicy};
use sha2::{Digest as _, Sha256};
use std::{
    fmt,
    sync::{
        Arc, Mutex, TryLockError,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
pub(crate) mod work;
use work::{AttemptObservation, ProcessWorkJournal};
pub use work::{
    CpuWorkTotals, PalsResolverIdentity, PalsSearchPolicyIdentity, PalsWorkTotals,
    ProcessSearchWorkReceipt,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "search-work-receipts", derive(serde::Serialize))]
#[cfg_attr(feature = "search-work-receipts", serde(rename_all = "snake_case"))]
pub enum SearchKind {
    Cpu,
    Pals,
    Rovent,
}

#[derive(Clone, Debug)]
pub struct SearchSessionContext {
    pub authority: rz_contracts::pals::SearchAuthority,
    /// The received go, before ponder normalization or Rules subset sealing.
    /// Legacy callers may omit it; native Rovent admission requires it.
    pub original_go: Option<Arc<crate::parser::GoLimits>>,
    /// Exact actual Start allocation. No later hit or root descriptor rebuild
    /// may replace this original clock evidence.
    pub original_time_budget: Option<rz_search::time::TimeBudget>,
    /// Private immutable custody issued only by the actual Session Start.
    /// Legacy consumers have no native Rovent work authority through `None`.
    pub(crate) original_start: Option<Arc<crate::engine::DriverStartSource>>,
    /// Read-only same-Start effective admission/stop clock. It never replaces
    /// the immutable deadlines of an already submitted role or CPU job.
    pub effective_clock: Option<crate::rovent_ponder::RoventPonderClockHandle>,
    /// Canonical root-only subset. Child legal/policy order remains Rules-owned.
    pub root_moves: Option<Arc<[BoardMove]>>,
    pub goal: rz_contracts::rovent::SearchGoal,
    /// Exact explicit UCI nodes limit, before legacy simulation defaults.
    pub requested_nodes: Option<u64>,
    pub control: SearchControl,
    pub cancellation: CancelToken,
    pub shutdown_deadline: Instant,
    pub(crate) current: Arc<Mutex<SessionScope>>,
}
/// Sticky marker in the original Collection game word, reserved by the
/// opt-in collection owner. Normal search never installs or reads it.
pub const ROVENT_COLLECTION_ABORT_BIT: u64 = 1u64 << 63;

impl SearchSessionContext {
    /// Bind the exact original Start to its optional Collection game word and
    /// root target flag. UCI stop touches only these shared atomic owners.
    #[cfg(feature = "pals-adaptive")]
    pub fn bind_original_collection_abort(
        &self,
        game_gate: Arc<std::sync::atomic::AtomicU64>,
        root_targets_revoked: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<(), ContractError> {
        self.original_start.as_ref()
            .ok_or_else(|| ContractError::new(ErrorCode::IdentityMismatch, Stage::Admission, "original UCI Start absent"))?
            .bind_collection_abort(self, game_gate, root_targets_revoked)
    }

    pub fn soft_expired(&self, now: Instant) -> bool {
        self.effective_clock
            .as_ref()
            .map_or(now >= self.control.soft_deadline, |clock| {
                clock.snapshot(now).soft_expired(now)
            })
    }
    /// Called at a task/commit boundary. An old root cannot mutate a new root.
    pub fn accepts(&self) -> Result<bool, ContractError> {
        let current = self.current.lock().map_err(|_| {
            ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Admission,
                "UCI search authority poisoned",
            )
        })?;
        Ok(*current == SessionScope::Search(self.authority)
            && !self.cancellation.is_canceled()
            && !self
                .control
                .cancellation
                .load(std::sync::atomic::Ordering::Acquire)
            && Instant::now() < self.control.deadline)
    }

    /// Admission while another owner lock is held must not wait for UCI.
    /// Contention remains an error, not a false stale/canceled observation.
    #[cfg(any(feature = "pals-adaptive", test))]
    pub(crate) fn try_accepts(&self) -> Result<bool, ContractError> {
        let current = self.current.try_lock().map_err(|cause| match cause {
            TryLockError::WouldBlock => ContractError::new(
                ErrorCode::ResourceExhausted,
                Stage::Admission,
                "UCI search authority is busy",
            ),
            TryLockError::Poisoned(_) => ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Admission,
                "UCI search authority poisoned",
            ),
        })?;
        Ok(*current == SessionScope::Search(self.authority)
            && !self.cancellation.is_canceled()
            && !self.control.cancellation.load(Ordering::Acquire)
            && Instant::now() < self.control.deadline)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DriverWork {
    Cpu {
        nodes: u64,
        completed_depth: u16,
    },
    Pals {
        rounds: u64,
        cpu_nodes: u64,
        completed_tasks: u64,
        consumed_role_outputs: u64,
        retained_situations: usize,
    },
    Rovent {
        root_entries: u64,
        total_node_visits: u64,
        role_calls: u64,
        accepted_policy: u64,
        accepted_fresh_values: u64,
        checked_transitions: u64,
        native_r_calls: u64,
        native_nodes: u64,
        native_qsearch_nodes: u64,
        native_pvs_entries: Option<u64>,
        native_see_nodes: Option<u64>,
        external_teacher_calls: u64,
        completed_depth: Option<u16>,
        maximum_examined_ply: u16,
    },
    /// Actual ticket-manager observations; head-specific acceptance is not
    /// inferred from total role counts. Unknown physical work stays unknown.
    RoventNative {
        root_entries: u64,
        total_node_visits: Option<u64>,
        checked_transitions: u64,
        native_nodes: Option<u64>,
        native_qsearch_nodes: Option<u64>,
        submitted_role_tickets: u64,
        consumed_role_outputs: u64,
        completed_nn_inputs: Option<u64>,
        consumed_nn_inputs: Option<u64>,
        completed_depth: Option<u16>,
        maximum_examined_ply: u16,
    },
}

/// Immutable startup declarations. These identify the selected helper, not
/// proof of loaded foreign weights, applied options or physical shutdown.
#[derive(Clone, Debug)]
pub struct PalsCheckerRegistration {
    pub identity: CheckerIdentity,
    pub conditions: String,
    pub capabilities: CheckerCapabilities,
    pub owned_descriptor: Option<OwnedCheckerDescriptor>,
    pub cpu_resume_policy: Option<rz_search::cpu::CpuResumePolicy>,
    pub role_model: String,
    pub model_value: Option<ModelValueIdentity>,
    pub resolver_version: &'static str,
    pub resolver_semantics: &'static str,
}
impl PalsCheckerRegistration {
    fn capture<M: rz_search::pals::engine::RoleModel>(
        checker: &dyn CpuChecker,
        model: &M,
    ) -> Result<Self, SearchSessionFailure> {
        let policy = if matches!(checker.identity(), CheckerIdentity::ExternalUci(_)) {
            ResolverPolicy::ModelWdlRestricted
        } else {
            ResolverPolicy::OwnRawRestricted
        };
        Self::capture_with_resolver(checker, model, policy)
    }
    pub(crate) fn capture_with_resolver<M: rz_search::pals::engine::RoleModel>(
        checker: &dyn CpuChecker,
        model: &M,
        policy: ResolverPolicy,
    ) -> Result<Self, SearchSessionFailure> {
        checker
            .validate_namespace()
            .map_err(|error| SearchSessionFailure::debug("PalsCheckerRegistration", &error))?;
        let external = matches!(checker.identity(), CheckerIdentity::ExternalUci(_));
        if external && policy == ResolverPolicy::OwnRawRestricted {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsResolverUnsupported",
                detail: "own raw resolver requires an owned checker; no foreign helper was started"
                    .into(),
            });
        }
        let (resolver_version, resolver_semantics) = if policy == ResolverPolicy::ModelWdlRestricted
        {
            (
                rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION,
                rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS,
            )
        } else {
            (
                rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION,
                rz_search::pals::engine::PALS_VALUE_RESOLVER_SEMANTICS,
            )
        };
        Ok(Self {
            identity: checker.identity().clone(),
            conditions: checker.conditions().to_owned(),
            capabilities: checker.capabilities(),
            owned_descriptor: checker.owned_descriptor(),
            cpu_resume_policy: checker.owned_resume_policy(),
            role_model: model.identity().to_owned(),
            model_value: if policy == ResolverPolicy::ModelWdlRestricted {
                model.value_identity().cloned()
            } else {
                None
            },
            resolver_version,
            resolver_semantics,
        })
    }
}

#[derive(Default)]
struct PalsCheckerLifecycle {
    active_cancel: Option<Arc<AtomicBool>>,
    // A bounded latest-search snapshot, replaced instead of appended per go.
    attempts: Vec<CheckerAttempt>,
    last_attempt: Option<CheckerAttempt>,
    shutdown: Option<CheckerShutdown>,
    startup_uci: Option<ExternalUciIdentity>,
    #[cfg(feature = "search-work-receipts")]
    startup_resources: Option<crate::pals_attestation::checker::resource::ReadyBoundaryResources>,
    #[cfg(feature = "search-work-receipts")]
    startup_resource_unavailable: Option<&'static str>,
}
struct PalsActiveLease<'a> {
    lifecycle: &'a Mutex<PalsCheckerLifecycle>,
    cancellation: Arc<AtomicBool>,
}
impl Drop for PalsActiveLease<'_> {
    fn drop(&mut self) {
        let Ok(mut state) = self.lifecycle.lock() else {
            return;
        };
        if state
            .active_cancel
            .as_ref()
            .is_some_and(|active| Arc::ptr_eq(active, &self.cancellation))
        {
            state.active_cancel = None;
        }
    }
}

/// Persistent P/C + startup-selected CPU checker session. The model is selected once at
/// startup; callers must explicitly identify a mock model as mock.
pub struct PalsSessionDriver<M: rz_search::pals::engine::RoleModel + 'static> {
    engine: Mutex<rz_search::pals::engine::PalsEngine<M>>,
    reset_requested: AtomicU64,
    reset_applied: AtomicU64,
    implementation: Digest,
    resolver_version: &'static str,
    identity: EngineIdentity,
    max_rounds: u64,
    max_cpu_nodes: u64,
    cpu_depth: u16,
    work: ProcessWorkJournal,
    registration: PalsCheckerRegistration,
    search_policy: Option<PalsSearchPolicyIdentity>,
    checker_closed: AtomicBool,
    checker_started: AtomicBool,
    checker_lifecycle: Mutex<PalsCheckerLifecycle>,
    #[cfg(feature = "search-work-receipts")]
    followup: Mutex<crate::pals_attestation::followup::FollowupJournal>,
    #[cfg(feature = "search-work-receipts")]
    followup_capture_complete: AtomicBool,
}
impl<M: rz_search::pals::engine::RoleModel + 'static> PalsSessionDriver<M> {
    pub fn new(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        cpu_config: rz_search::cpu::CpuConfig,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
    ) -> Result<Self, SearchSessionFailure> {
        if max_rounds == 0
            || max_rounds > 1_000_000
            || max_cpu_nodes == 0
            || cpu_depth == 0
            || cpu_depth > cpu_config.max_depth
        {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsConfiguration",
                detail: "finite PALS round/CPU limits are required".into(),
            });
        }
        let mut hash = Sha256::new();
        hash.update(b"rz-uci-pals-session/1\0");
        hash.update(rz_search::pals::engine::PALS_SEARCH_VERSION.as_bytes());
        hash.update(b"\0value-resolver\0");
        hash.update(rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION.as_bytes());
        hash.update([0]);
        hash.update(rz_search::pals::engine::PALS_VALUE_RESOLVER_SEMANTICS.as_bytes());
        hash.update(rz_search::cpu::CPU_SEARCH_VERSION.as_bytes());
        hash.update(rz_search::cpu::BOOTSTRAP_SCORE_VERSION.as_bytes());
        hash.update(model.identity().as_bytes());
        for value in [
            config.beam_width,
            config.line_plies,
            config.max_nodes,
            config.max_records,
        ] {
            hash.update((value as u64).to_le_bytes());
        }
        hash.update(config.max_role_calls.to_le_bytes());
        hash.update(config.cpu_nodes_per_task.to_le_bytes());
        hash.update(cpu_config.profile.identity().as_bytes());
        hash.update((cpu_config.tt_entries as u64).to_le_bytes());
        hash.update(cpu_config.max_depth.to_le_bytes());
        hash.update(cpu_config.quiescence_ply.to_le_bytes());
        hash.update(cpu_depth.to_le_bytes());
        hash.update(max_rounds.to_le_bytes());
        hash.update(max_cpu_nodes.to_le_bytes());
        let cpu = rz_search::cpu::CpuEngine::new(cpu_config)
            .map_err(|error| SearchSessionFailure::debug("PalsCpuConstructor", &error))?;
        let checker = OwnedCpuChecker::new(cpu)
            .map_err(|error| SearchSessionFailure::debug("PalsCheckerConstructor", &error))?;
        let registration = PalsCheckerRegistration::capture(&checker, &model)?;
        let engine = rz_search::pals::engine::PalsEngine::new_with_checker(config, model, checker)
            .map_err(|error| SearchSessionFailure::debug("PalsConstructor", &error))?;
        Ok(Self {
            engine: Mutex::new(engine),
            reset_requested: AtomicU64::new(0),
            reset_applied: AtomicU64::new(0),
            implementation: Digest(hash.finalize().into()),
            resolver_version: rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION,
            identity,
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            work: ProcessWorkJournal::new(SearchKind::Pals),
            registration,
            search_policy: None,
            checker_closed: AtomicBool::new(false),
            checker_started: AtomicBool::new(true),
            checker_lifecycle: Mutex::new(PalsCheckerLifecycle::default()),
            #[cfg(feature = "search-work-receipts")]
            followup: Mutex::new(Default::default()),
            #[cfg(feature = "search-work-receipts")]
            followup_capture_complete: AtomicBool::new(true),
        })
    }

    /// The omitted/Disabled selection delegates to the historical own v1
    /// constructor. Only the explicit lane uses the full checker namespace.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_refinement_policy(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        cpu_config: rz_search::cpu::CpuConfig,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
        policy: PostRepairRecheckPolicy,
    ) -> Result<Self, SearchSessionFailure> {
        if policy == PostRepairRecheckPolicy::Disabled {
            return Self::new(
                config,
                model,
                cpu_config,
                max_rounds,
                max_cpu_nodes,
                cpu_depth,
                identity,
            );
        }
        if max_rounds == 0
            || max_rounds > 1_000_000
            || max_cpu_nodes == 0
            || cpu_depth == 0
            || cpu_depth > cpu_config.max_depth
        {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsConfiguration",
                detail: "finite PALS round/CPU limits are required".into(),
            });
        }
        let cpu = rz_search::cpu::CpuEngine::new(cpu_config)
            .map_err(|error| SearchSessionFailure::debug("PalsCpuConstructor", &error))?;
        let checker = OwnedCpuChecker::new(cpu)
            .map_err(|error| SearchSessionFailure::debug("PalsCheckerConstructor", &error))?;
        Self::new_with_checker_and_refinement_policy(
            config,
            model,
            checker,
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            identity,
            policy,
        )
    }

    /// Explicit checker-selected path with a separate v2 source namespace.
    /// The legacy own constructor above keeps its v1 hash byte-for-byte.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_checker<C: CpuChecker + 'static>(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        checker: C,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
    ) -> Result<Self, SearchSessionFailure> {
        Self::new_with_boxed_checker(
            config,
            model,
            Box::new(checker),
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            identity,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_checker_and_refinement_policy<C: CpuChecker + 'static>(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        checker: C,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
        policy: PostRepairRecheckPolicy,
    ) -> Result<Self, SearchSessionFailure> {
        Self::new_with_boxed_checker_and_refinement_policy(
            config,
            model,
            Box::new(checker),
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            identity,
            policy,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_boxed_checker(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        checker: Box<dyn CpuChecker>,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
    ) -> Result<Self, SearchSessionFailure> {
        Self::new_with_boxed_checker_and_refinement_policy(
            config,
            model,
            checker,
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            identity,
            PostRepairRecheckPolicy::Disabled,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_boxed_checker_and_refinement_policy(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        checker: Box<dyn CpuChecker>,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
        policy: PostRepairRecheckPolicy,
    ) -> Result<Self, SearchSessionFailure> {
        let resolver = if matches!(checker.identity(), CheckerIdentity::ExternalUci(_)) {
            ResolverPolicy::ModelWdlRestricted
        } else {
            ResolverPolicy::OwnRawRestricted
        };
        Self::new_with_boxed_checker_and_selected_policies(
            config,
            model,
            checker,
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            identity,
            resolver,
            policy,
            false,
        )
    }

    /// Explicit independent resolver/refinement selection; legacy constructors
    /// above retain their historical checker-dependent defaults and hashes.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_boxed_checker_and_policies(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        checker: Box<dyn CpuChecker>,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
        resolver: ResolverPolicy,
        policy: PostRepairRecheckPolicy,
    ) -> Result<Self, SearchSessionFailure> {
        Self::new_with_boxed_checker_and_selected_policies(
            config,
            model,
            checker,
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            identity,
            resolver,
            policy,
            true,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_policies(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        cpu_config: rz_search::cpu::CpuConfig,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
        resolver: ResolverPolicy,
        policy: PostRepairRecheckPolicy,
        resume: rz_search::cpu::CpuResumePolicy,
    ) -> Result<Self, SearchSessionFailure> {
        let cpu = rz_search::cpu::CpuEngine::with_resume_policy(cpu_config, resume)
            .map_err(|error| SearchSessionFailure::debug("PalsCpuConstructor", &error))?;
        let checker = OwnedCpuChecker::new(cpu)
            .map_err(|error| SearchSessionFailure::debug("PalsCheckerConstructor", &error))?;
        Self::new_with_boxed_checker_and_policies(
            config,
            model,
            Box::new(checker),
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            identity,
            resolver,
            policy,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn new_with_boxed_checker_and_selected_policies(
        config: rz_search::pals::engine::PalsConfig,
        model: M,
        checker: Box<dyn CpuChecker>,
        max_rounds: u64,
        max_cpu_nodes: u64,
        cpu_depth: u16,
        identity: EngineIdentity,
        resolver: ResolverPolicy,
        policy: PostRepairRecheckPolicy,
        explicit: bool,
    ) -> Result<Self, SearchSessionFailure> {
        let registration =
            PalsCheckerRegistration::capture_with_resolver(checker.as_ref(), &model, resolver)?;
        if policy != PostRepairRecheckPolicy::Disabled
            && !policy.uses_frozen_model_wdl()
            && matches!(registration.identity, CheckerIdentity::ExternalUci(_))
        {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsRefinementUnsupported",
                detail: "post-Repair recheck v1 requires own CPU/Rules evidence; no foreign helper was started".into(),
            });
        }
        if max_rounds == 0
            || max_rounds > 1_000_000
            || max_cpu_nodes == 0
            || cpu_depth == 0
            || cpu_depth > registration.capabilities.max_depth
            || registration.role_model.is_empty()
            || registration.role_model.len() > 1024
            || registration.role_model.chars().any(char::is_control)
        {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsConfiguration",
                detail: "finite PALS limits and bounded model identity are required".into(),
            });
        }
        let legacy_implementation =
            checker_implementation(&registration, &config, max_rounds, max_cpu_nodes, cpu_depth);
        let engine = if explicit {
            rz_search::pals::engine::PalsEngine::new_with_boxed_checker_and_policies(
                config, model, checker, resolver, policy,
            )
        } else {
            rz_search::pals::engine::PalsEngine::new_with_boxed_checker_and_refinement_policy(
                config, model, checker, policy,
            )
        }
        .map_err(|error| SearchSessionFailure::debug("PalsConstructor", &error))?;
        let search_policy = selected_search_policy(&engine, policy)?;
        let legacy_implementation = if explicit {
            let mut hash = Sha256::new();
            hash.update(b"rz-uci-pals-followup-session/2\0");
            hash.update(legacy_implementation.0);
            hash.update(engine.search_identity());
            hash.update(resolver.version());
            Digest(hash.finalize().into())
        } else {
            legacy_implementation
        };
        let implementation = search_policy
            .as_ref()
            .map_or(legacy_implementation, |selected| {
                refinement_implementation(legacy_implementation, selected)
            });
        let resolver = PalsResolverIdentity::from_semantics(
            registration.resolver_version,
            registration.resolver_semantics,
        );
        let work = match &search_policy {
            Some(selected) => {
                ProcessWorkJournal::new_pals_with_search_policy(resolver, selected.clone())
                    .map_err(|error| {
                        SearchSessionFailure::debug("PalsSearchPolicyRegistration", &error)
                    })?
            }
            None => ProcessWorkJournal::new_pals(resolver),
        };
        Ok(Self {
            engine: Mutex::new(engine),
            reset_requested: AtomicU64::new(0),
            reset_applied: AtomicU64::new(0),
            implementation,
            resolver_version: registration.resolver_version,
            identity,
            max_rounds,
            max_cpu_nodes,
            cpu_depth,
            work,
            checker_started: AtomicBool::new(matches!(
                registration.identity,
                CheckerIdentity::Owned(_)
            )),
            registration,
            search_policy,
            checker_closed: AtomicBool::new(false),
            checker_lifecycle: Mutex::new(PalsCheckerLifecycle::default()),
            #[cfg(feature = "search-work-receipts")]
            followup: Mutex::new(Default::default()),
            #[cfg(feature = "search-work-receipts")]
            followup_capture_complete: AtomicBool::new(true),
        })
    }

    pub fn checker_registration(&self) -> &PalsCheckerRegistration {
        &self.registration
    }
    pub fn enable_archive(
        &self,
        config: rz_search::pals::store::ArchiveConfig,
    ) -> Result<(), SearchSessionFailure> {
        self.engine
            .lock()
            .map_err(|_| SearchSessionFailure::debug("PalsArchive", &"engine lock poisoned"))?
            .enable_archive(config)
            .map_err(|error| SearchSessionFailure::debug("PalsArchive", &error))
    }
    pub fn set_archive_runtime_limits(
        &self,
        limits: rz_search::pals::store::ArchiveRuntimeLimits,
    ) -> Result<(), SearchSessionFailure> {
        let mut engine = self.engine.lock().map_err(|_| {
            SearchSessionFailure::debug("PalsArchiveLimits", &"engine lock poisoned")
        })?;
        engine
            .set_archive_runtime_limits(limits)
            .map_err(|error| SearchSessionFailure::debug("PalsArchiveLimits", &error))?;
        if engine.archive_runtime_limits().copied() != Some(limits) {
            return Err(SearchSessionFailure::debug(
                "PalsArchiveLimits",
                &"actual limits differ from selected limits",
            ));
        }
        Ok(())
    }
    pub fn observe_archive_startup(
        &self,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), SearchSessionFailure> {
        self.engine
            .try_lock()
            .map_err(|_| {
                SearchSessionFailure::debug("PalsArchiveStartup", &"engine owner unavailable")
            })?
            .observe_archive_startup(deadline, cancel)
            .map_err(|error| SearchSessionFailure::debug("PalsArchiveStartup", &error))
    }
    #[cfg(feature = "search-work-receipts")]
    pub fn enable_followup_execution(
        &self,
        process_epoch: Option<u64>,
        cuda_warm_implementation: Option<[u8; 32]>,
    ) -> Result<(), SearchSessionFailure> {
        use crate::pals_attestation::followup as f;
        let engine = self
            .engine
            .lock()
            .map_err(|_| SearchSessionFailure::debug("PalsFollowup", &"engine lock poisoned"))?;
        let repair = engine.post_repair_recheck_policy().uses_frozen_model_wdl();
        let stack = self.registration.cpu_resume_policy
            == Some(rz_search::cpu::CpuResumePolicy::PausedStack);
        let archive = engine.archive_enabled();
        if !repair && !stack && !archive && cuda_warm_implementation.is_none() {
            return Ok(());
        }
        let method = |method: &str, digest: [u8; 32]| f::NativeOwnerMethodIdentityV4 {
            method: method.into(),
            implementation_sha256: f::hex(&digest),
        };
        let methods = f::NativeFollowupOwnerMethodsV4 {
            producer_domain: f::DOMAIN.into(),
            producer_schema_version: 1,
            producer_implementation_sha256: f::hex(&f::compiled_followup_producer_sha256()),
            trace_capacity: f::TRACE_CAPACITY as u32,
            trace_bytes_max: f::TRACE_BYTES_MAX as u64,
            repair: repair.then(|| {
                method(
                    "engine_actual_followup_search_v4",
                    rz_search::pals::engine::compiled_search_implementation_sha256(),
                )
            }),
            paused_stack: stack.then(|| {
                method(
                    rz_search::cpu::CPU_PAUSED_STACK_OWNER_ACCOUNTING_METHOD,
                    rz_search::cpu::compiled_paused_stack_owner_sha256(),
                )
            }),
            cold_archive: archive.then(|| {
                method(
                    "archive_actual_owner_events_v4",
                    rz_search::pals::store::archive_accounting_source_sha256(),
                )
            }),
            cuda_warm: cuda_warm_implementation
                .map(|sha| method("cuda_actual_backend_and_bank_events_v4", sha)),
        };
        let mut journal = self
            .followup
            .lock()
            .map_err(|_| SearchSessionFailure::debug("PalsFollowup", &"journal poisoned"))?;
        if !journal.enable(
            process_epoch,
            methods,
            engine.post_repair_recheck_policy()
                == PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
        ) {
            return Err(SearchSessionFailure::debug(
                "PalsFollowup",
                &"followup capture already selected",
            ));
        }
        journal.capture_initial_owner(&engine.followup_owner_snapshot());
        Ok(())
    }
    #[cfg(feature = "search-work-receipts")]
    pub fn followup_lifecycle(
        &self,
    ) -> Result<Option<crate::pals_attestation::followup::PalsFollowupLifecycleV4>, ContractError>
    {
        self.followup
            .try_lock()
            .map(|journal| {
                let mut lifecycle = journal.lifecycle.clone();
                if !self.followup_capture_complete.load(Ordering::Acquire)
                    && let Some(lifecycle) = lifecycle.as_mut()
                {
                    lifecycle.capture_complete = false;
                    lifecycle.owner_shutdown_complete = false;
                    lifecycle.capture_failure.get_or_insert(
                        crate::pals_attestation::followup::PalsCaptureFailureV4::CaptureIncomplete,
                    );
                }
                lifecycle
            })
            .map_err(|_| {
                ContractError::new(
                    ErrorCode::BackendFailure,
                    Stage::Output,
                    "followup journal metadata unavailable",
                )
            })
    }
    #[cfg(feature = "search-work-receipts")]
    pub fn capture_warm_followup(
        &self,
        value: Option<crate::pals_attestation::followup::NativeCudaWarmObservationV4>,
    ) -> Result<(), SearchSessionFailure> {
        self.followup
            .try_lock()
            .map_err(|_| {
                self.followup_capture_complete
                    .store(false, Ordering::Release);
                checker_owner_unknown("PalsFollowupMetadata")
            })?
            .capture_warm(value);
        Ok(())
    }
    #[cfg(feature = "search-work-receipts")]
    pub fn finish_followup_owners(
        &self,
        deadline: Instant,
        native_fenced: bool,
    ) -> Result<(), ContractError> {
        if !self
            .followup
            .try_lock()
            .map_err(|_| {
                ContractError::new(
                    ErrorCode::BackendFailure,
                    Stage::Output,
                    "followup journal unavailable",
                )
            })?
            .enabled()
        {
            return Ok(());
        }
        let checker_fenced = self
            .checker_lifecycle
            .try_lock()
            .ok()
            .and_then(|s| s.shutdown)
            .is_some_and(|s| s.cleanup_complete && !s.quarantined && !s.ownership_lost);
        if !checker_fenced || !native_fenced {
            return Err(ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Output,
                "actual followup owner fences are incomplete",
            ));
        }
        let mut engine = loop {
            match self.engine.try_lock() {
                Ok(engine) => break engine,
                Err(TryLockError::WouldBlock) if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(1))
                }
                _ => {
                    return Err(ContractError::new(
                        ErrorCode::BackendFailure,
                        Stage::Output,
                        "followup owner close unavailable on original deadline",
                    ));
                }
            }
        };
        let mut journal = self.followup.try_lock().map_err(|_| {
            ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Output,
                "followup journal unavailable",
            )
        })?;
        if !journal.enabled() {
            return Ok(());
        }
        let closed = engine.finish_followup_owners(deadline);
        journal.capture_owner(&engine.followup_owner_snapshot());
        closed.map(|_| ()).map_err(|_| {
            ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Output,
                "actual archive owner close failed",
            )
        })
    }
    #[cfg(feature = "search-work-receipts")]
    pub fn seal_followup_lifecycle(
        &self,
        native_fenced: bool,
        protocol_retired: bool,
        warm: Option<crate::pals_attestation::followup::NativeCudaWarmObservationV4>,
    ) -> Result<(), ContractError> {
        let work = self.work.snapshot()?;
        let checker_fenced = self
            .checker_lifecycle
            .try_lock()
            .ok()
            .and_then(|state| state.shutdown)
            .is_some_and(|shutdown| {
                shutdown.cleanup_complete && !shutdown.quarantined && !shutdown.ownership_lost
            });
        let mut journal = self.followup.try_lock().map_err(|_| {
            ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Output,
                "followup journal metadata unavailable",
            )
        })?;
        if !self.followup_capture_complete.load(Ordering::Acquire) {
            journal.incomplete(
                crate::pals_attestation::followup::PalsCaptureFailureV4::CaptureIncomplete,
            );
        }
        if protocol_retired && work.active_invocations == 0 {
            journal.output_retired();
        }
        journal.capture_warm(warm);
        journal.seal(checker_fenced, native_fenced);
        if journal
            .lifecycle
            .as_ref()
            .is_some_and(|l| !l.capture_complete || !l.owner_shutdown_complete)
        {
            return Err(ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Output,
                "actual followup lifecycle capture or owner fence is incomplete",
            ));
        }
        Ok(())
    }
    pub fn selected_policies(
        &self,
    ) -> Result<
        (
            ResolverPolicy,
            PostRepairRecheckPolicy,
            &'static str,
            Option<&'static str>,
        ),
        SearchSessionFailure,
    > {
        let engine = self
            .engine
            .lock()
            .map_err(|_| SearchSessionFailure::debug("PalsPolicies", &"engine lock poisoned"))?;
        Ok((
            engine.resolver_policy(),
            engine.post_repair_recheck_policy(),
            engine.search_identity(),
            engine.refinement_conditions(),
        ))
    }
    /// Available before any go, including when a later result is early/unknown.
    /// This does not attest that a recheck branch ran or refuted a repaired line.
    pub fn search_policy_registration(&self) -> Option<&PalsSearchPolicyIdentity> {
        self.search_policy.as_ref()
    }
    /// Startup success is independent of registration and UCI option claims.
    pub fn checker_started(&self) -> bool {
        self.checker_started.load(Ordering::Acquire)
    }

    /// Starts an unstarted foreign helper on the caller's clock. The shared
    /// signal also permits finish_checker to cancel an active handshake.
    pub fn start_checker(
        &self,
        deadline: Instant,
        cancel: Arc<AtomicBool>,
    ) -> Result<(), SearchSessionFailure> {
        if self.checker_closed.load(Ordering::Acquire) {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsCheckerClosed",
                detail: "checker startup is permanently closed".into(),
            });
        }
        let now = Instant::now();
        if cancel.load(Ordering::Acquire)
            || deadline <= now
            || deadline.duration_since(now) > Duration::from_secs(180)
        {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsCheckerStartupDeadline",
                detail: "checker startup requires an uncanceled finite deadline".into(),
            });
        }
        let mut engine = loop {
            if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PalsCheckerStartupLockDeadline",
                    detail: "no checker startup admitted before cancellation/deadline".into(),
                });
            }
            match self.engine.try_lock() {
                Ok(engine) => break engine,
                Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(1)),
                Err(TryLockError::Poisoned(_)) => {
                    return Err(checker_owner_unknown("PalsCheckerStartupOwner"));
                }
            }
        };
        let _active = self.activate_checker(Arc::clone(&cancel))?;
        if matches!(self.registration.identity, CheckerIdentity::ExternalUci(_))
            && self.checker_started()
        {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsCheckerAlreadyStarted",
                detail: "foreign helper startup is not repeated implicitly".into(),
            });
        }
        let startup = engine.start_checker(deadline, &cancel);
        self.capture_checker(&engine)?;
        startup.map_err(|error| pals_failure("PalsCheckerStartup", &error, &engine))?;
        #[cfg(feature = "search-work-receipts")]
        let startup_resources =
            if matches!(self.registration.identity, CheckerIdentity::ExternalUci(_)) {
                let process = engine
                    .checker_last_attempt()
                    .and_then(|attempt| attempt.external.as_ref())
                    .and_then(|external| external.process.process_identity);
                crate::pals_attestation::checker::resource::observe(process, deadline, &cancel)
                    .map_err(|error| {
                        SearchSessionFailure::debug("PalsCheckerStartupResource", &error)
                    })?
            } else {
                crate::pals_attestation::checker::resource::ResourceObservation {
                    snapshot: None,
                    unavailable: None,
                }
            };
        // An adapter returning after control expiry does not gain ready status.
        if cancel.load(Ordering::Acquire)
            || Instant::now() >= deadline
            || self.checker_closed.load(Ordering::Acquire)
        {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsCheckerStartupLate",
                detail: "checker startup returned after its admission controls".into(),
            });
        }
        let mut lifecycle = self
            .checker_lifecycle
            .lock()
            .map_err(|_| checker_owner_unknown("PalsCheckerStartupEvidence"))?;
        lifecycle.startup_uci = engine.checker_startup_uci();
        #[cfg(feature = "search-work-receipts")]
        if lifecycle.startup_resources.is_none() {
            lifecycle.startup_resources = startup_resources.snapshot;
            lifecycle.startup_resource_unavailable = startup_resources.unavailable;
        }
        self.checker_started.store(true, Ordering::Release);
        Ok(())
    }
    pub fn checker_startup_uci(&self) -> Result<Option<ExternalUciIdentity>, SearchSessionFailure> {
        self.checker_lifecycle
            .lock()
            .map(|state| state.startup_uci.clone())
            .map_err(|_| checker_owner_unknown("PalsCheckerStartupEvidence"))
    }
    #[cfg(feature = "search-work-receipts")]
    pub fn checker_startup_resources(
        &self,
    ) -> Result<
        Option<crate::pals_attestation::checker::resource::ReadyBoundaryResources>,
        SearchSessionFailure,
    > {
        self.checker_lifecycle
            .lock()
            .map(|state| state.startup_resources.clone())
            .map_err(|_| checker_owner_unknown("PalsCheckerStartupResource"))
    }
    #[cfg(feature = "search-work-receipts")]
    pub fn checker_startup_resource_unavailable(
        &self,
    ) -> Result<Option<&'static str>, SearchSessionFailure> {
        self.checker_lifecycle
            .lock()
            .map(|state| state.startup_resource_unavailable)
            .map_err(|_| checker_owner_unknown("PalsCheckerStartupResource"))
    }

    /// Latest-search physical attempts only. The latest lifecycle snapshot is
    /// available separately; callers must not sum it again as a new request.
    pub fn checker_attempts(&self) -> Result<Vec<CheckerAttempt>, SearchSessionFailure> {
        self.checker_lifecycle
            .lock()
            .map(|state| state.attempts.clone())
            .map_err(|_| checker_owner_unknown("PalsCheckerEvidence"))
    }
    pub fn checker_last_attempt(&self) -> Result<Option<CheckerAttempt>, SearchSessionFailure> {
        self.checker_lifecycle
            .lock()
            .map(|state| state.last_attempt.clone())
            .map_err(|_| checker_owner_unknown("PalsCheckerEvidence"))
    }
    pub fn checker_shutdown(&self) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        self.checker_lifecycle
            .lock()
            .map(|state| state.shutdown)
            .map_err(|_| checker_owner_unknown("PalsCheckerEvidence"))
    }

    fn activate_checker(
        &self,
        cancellation: Arc<AtomicBool>,
    ) -> Result<PalsActiveLease<'_>, SearchSessionFailure> {
        let mut state = self
            .checker_lifecycle
            .lock()
            .map_err(|_| checker_owner_unknown("PalsCheckerOwner"))?;
        if self.checker_closed.load(Ordering::Acquire) {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsCheckerClosed",
                detail: "startup-selected helper is closing or closed; no work admitted".into(),
            });
        }
        state.active_cancel = Some(Arc::clone(&cancellation));
        Ok(PalsActiveLease {
            lifecycle: &self.checker_lifecycle,
            cancellation,
        })
    }

    fn capture_checker(
        &self,
        engine: &rz_search::pals::engine::PalsEngine<M>,
    ) -> Result<(), SearchSessionFailure> {
        let mut state = self
            .checker_lifecycle
            .lock()
            .map_err(|_| checker_owner_unknown("PalsCheckerEvidence"))?;
        state.attempts = engine.checker_attempts().to_vec();
        state.last_attempt = engine.checker_last_attempt().cloned();
        Ok(())
    }

    /// Permanently closes admission, cancels active work and drains the checker
    /// under a finite deadline. This does not shut down the role-model owner.
    /// A successful own result claims no subprocess exit or pipe drain.
    pub fn finish_checker(
        &self,
        deadline: Instant,
    ) -> Result<CheckerShutdown, SearchSessionFailure> {
        self.checker_closed.store(true, Ordering::Release);
        {
            let state = self
                .checker_lifecycle
                .lock()
                .map_err(|_| checker_owner_unknown("PalsCheckerShutdown"))?;
            if let Some(cancel) = &state.active_cancel {
                cancel.store(true, Ordering::Release);
            }
            if let Some(shutdown) = state.shutdown.filter(|value| {
                value.cleanup_complete && !value.quarantined && !value.ownership_lost
            }) {
                return Ok(shutdown);
            }
        }
        let now = Instant::now();
        if deadline <= now || deadline.duration_since(now) > Duration::from_secs(180) {
            return Err(checker_owner_unknown("PalsCheckerShutdownDeadline"));
        }
        let mut engine = loop {
            if Instant::now() >= deadline {
                return Err(checker_owner_unknown("PalsCheckerShutdownLockDeadline"));
            }
            match self.engine.try_lock() {
                Ok(engine) => break engine,
                Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(1)),
                Err(TryLockError::Poisoned(_)) => {
                    return Err(checker_owner_unknown("PalsCheckerShutdownOwner"));
                }
            }
        };
        let result = engine.shutdown_checker(deadline);
        self.capture_checker(&engine)?;
        // Preserve the actual checker fence even if independent archive close
        // fails below. Neither error may erase an already returned owner fact.
        {
            let mut state = self
                .checker_lifecycle
                .lock()
                .map_err(|_| checker_owner_unknown("PalsCheckerEvidence"))?;
            state.shutdown = result.as_ref().ok().copied().or_else(|| {
                state
                    .last_attempt
                    .as_ref()
                    .and_then(|attempt| attempt.external.as_ref())
                    .map(|attempt| attempt.process)
            });
        }
        #[cfg(feature = "search-work-receipts")]
        {
            let mut journal = self
                .followup
                .lock()
                .map_err(|_| checker_owner_unknown("PalsFollowupOwnerEvidence"))?;
            if journal.enabled() {
                if !journal.requires_native_fence()
                    && result
                        .as_ref()
                        .is_ok_and(|s| s.cleanup_complete && !s.quarantined && !s.ownership_lost)
                {
                    let closed = engine.finish_followup_owners(deadline);
                    journal.capture_owner(&engine.followup_owner_snapshot());
                    if let Err(error) = closed {
                        journal.incomplete(
                            crate::pals_attestation::followup::PalsCaptureFailureV4::OwnerUnknown,
                        );
                        return Err(SearchSessionFailure::debug("PalsArchiveOwnerClose", &error));
                    }
                } else {
                    journal.capture_owner(&engine.followup_owner_snapshot());
                }
            }
        }
        match result {
            Ok(shutdown)
                if shutdown.cleanup_complete
                    && !shutdown.quarantined
                    && !shutdown.ownership_lost =>
            {
                Ok(shutdown)
            }
            Ok(_) => Err(checker_owner_unknown("PalsCheckerShutdownIncomplete")),
            Err(error) => {
                let mut failure = SearchSessionFailure::debug("PalsCheckerShutdown", &error);
                failure.physical_completion = DriverPhysicalCompletion::Unknown;
                Err(failure)
            }
        }
    }
}
impl<M: rz_search::pals::engine::RoleModel + 'static> SearchSessionDriver for PalsSessionDriver<M> {
    fn kind(&self) -> SearchKind {
        SearchKind::Pals
    }
    fn implementation(&self) -> Digest {
        self.implementation
    }
    fn identity(&self) -> EngineIdentity {
        self.identity.clone()
    }
    fn work_receipt(&self) -> Result<Option<ProcessSearchWorkReceipt>, ContractError> {
        self.work.snapshot().map(Some)
    }
    #[cfg(feature = "search-work-receipts")]
    fn followup_lifecycle(
        &self,
    ) -> Result<Option<crate::pals_attestation::followup::PalsFollowupLifecycleV4>, ContractError>
    {
        PalsSessionDriver::followup_lifecycle(self)
    }
    #[cfg(feature = "search-work-receipts")]
    fn seal_followup_lifecycle(&self, protocol_retired: bool) -> Result<(), ContractError> {
        PalsSessionDriver::seal_followup_lifecycle(self, true, protocol_retired, None)
    }
    fn accept_uci_output(
        &self,
        _authority: rz_contracts::pals::SearchAuthority,
        _bestmove: &str,
        _from_report: bool,
    ) -> Result<(), ContractError> {
        #[cfg(feature = "search-work-receipts")]
        if let Ok(mut journal) = self.followup.try_lock() {
            journal.output_retired();
        }
        Ok(())
    }
    fn pals_checker_registration(&self) -> Option<PalsCheckerRegistration> {
        Some(self.registration.clone())
    }
    fn checker_started(&self) -> Option<bool> {
        Some(PalsSessionDriver::checker_started(self))
    }
    fn checker_startup_uci(&self) -> Result<Option<ExternalUciIdentity>, SearchSessionFailure> {
        PalsSessionDriver::checker_startup_uci(self)
    }
    #[cfg(feature = "search-work-receipts")]
    fn checker_startup_resources(
        &self,
    ) -> Result<
        Option<crate::pals_attestation::checker::resource::ReadyBoundaryResources>,
        SearchSessionFailure,
    > {
        PalsSessionDriver::checker_startup_resources(self)
    }
    #[cfg(feature = "search-work-receipts")]
    fn checker_startup_resource_unavailable(
        &self,
    ) -> Result<Option<&'static str>, SearchSessionFailure> {
        PalsSessionDriver::checker_startup_resource_unavailable(self)
    }
    fn start_checker(
        &self,
        deadline: Instant,
        cancel: Arc<AtomicBool>,
    ) -> Result<Option<bool>, SearchSessionFailure> {
        PalsSessionDriver::start_checker(self, deadline, cancel).map(|()| Some(true))
    }
    fn checker_attempts(&self) -> Result<Option<Vec<CheckerAttempt>>, SearchSessionFailure> {
        PalsSessionDriver::checker_attempts(self).map(Some)
    }
    fn checker_last_attempt(&self) -> Result<Option<CheckerAttempt>, SearchSessionFailure> {
        PalsSessionDriver::checker_last_attempt(self)
    }
    fn finish_checker(
        &self,
        deadline: Instant,
    ) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        PalsSessionDriver::finish_checker(self, deadline).map(Some)
    }
    fn checker_shutdown(&self) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        PalsSessionDriver::checker_shutdown(self)
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        if self.checker_closed.load(Ordering::Acquire) {
            return Err(ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Admission,
                "PALS checker admission is permanently closed",
            ));
        }
        let requested = self
            .reset_requested
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| {
                ContractError::new(
                    ErrorCode::BackendFailure,
                    Stage::Admission,
                    "PALS reset request generation exhausted",
                )
            })?
            + 1;
        // A foreign reset needs real deadline/cancellation and may fail. Do not
        // send ucinewgame, clear its evidence or acknowledge success here.
        if matches!(self.registration.identity, CheckerIdentity::ExternalUci(_)) {
            return Ok(());
        }
        match self.engine.try_lock() {
            Ok(mut engine) => {
                if engine.archive_enabled() {
                    return Ok(());
                }
                engine.new_game();
                self.reset_applied.fetch_max(requested, Ordering::AcqRel);
                Ok(())
            }
            Err(TryLockError::WouldBlock) => Ok(()), // Closing old work owns the lock; reset before the next admitted search.
            Err(TryLockError::Poisoned(_)) => Err(ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Admission,
                "PALS session owner poisoned",
            )),
        }
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        self.work
            .begin()
            .map_err(|error| SearchSessionFailure::debug("PalsWorkReceipt", &error))?;
        let mut observation = AttemptObservation::NoWork;
        let result = (|| {
            if self.checker_closed.load(Ordering::Acquire) {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PalsCheckerClosed",
                    detail: "startup-selected helper is closing or closed; no work admitted".into(),
                });
            }
            if !self.checker_started() {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PalsCheckerNotStarted",
                    detail: "foreign helper must complete explicit startup before go".into(),
                });
            }
            if context.control.max_simulations > self.max_cpu_nodes {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PalsNodeLimit",
                    detail: "UCI node request exceeds registered CPU work bound".into(),
                });
            }
            // One owner retains game evidence. Waiting for a canceled previous root
            // is bounded by the new root deadline, not an unbounded Mutex::lock.
            let mut engine = loop {
                if !context.accepts().map_err(|error| SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PalsAuthority",
                    detail: error.to_string(),
                })? {
                    return Ok(SearchSessionReport {
                        mate_strategy: None,
                        best_move: None,
                        work: DriverWork::Pals {
                            rounds: 0,
                            cpu_nodes: 0,
                            completed_tasks: 0,
                            consumed_role_outputs: 0,
                            retained_situations: 0,
                        },
                    });
                }
                match self.engine.try_lock() {
                    Ok(engine) => break engine,
                    Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(1)),
                    Err(TryLockError::Poisoned(_)) => {
                        return Err(SearchSessionFailure {
                            physical_completion: DriverPhysicalCompletion::Confirmed,
                            code: "PalsOwner",
                            detail: "persistent PALS owner poisoned".into(),
                        });
                    }
                }
            };
            if !context.accepts().map_err(|error| SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsAuthority",
                detail: error.to_string(),
            })? {
                return Ok(SearchSessionReport {
                    mate_strategy: None,
                    best_move: None,
                    work: DriverWork::Pals {
                        rounds: 0,
                        cpu_nodes: 0,
                        completed_tasks: 0,
                        consumed_role_outputs: 0,
                        retained_situations: engine.retained_situations(),
                    },
                });
            }
            let selected_policy = self
                .search_policy
                .as_ref()
                .map(PalsSearchPolicyIdentity::selected_policy)
                .transpose()
                .map_err(|error| SearchSessionFailure::debug("PalsSearchPolicyIdentity", &error))?
                .unwrap_or(PostRepairRecheckPolicy::Disabled);
            if selected_search_policy(&engine, selected_policy)? != self.search_policy {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PalsSearchPolicyIdentity",
                    detail: "effective PALS lane differs from its immutable startup registration"
                        .into(),
                });
            }
            let _active = self.activate_checker(Arc::clone(&context.control.cancellation))?;
            let reset_requested = self.reset_requested.load(Ordering::Acquire);
            if reset_requested != self.reset_applied.load(Ordering::Acquire) {
                let reset = engine.try_new_game(
                    context
                        .control
                        .soft_deadline
                        .min(context.control.admission_deadline),
                    &context.control.cancellation,
                );
                self.capture_checker(&engine)?;
                #[cfg(feature = "search-work-receipts")]
                if reset.is_err() {
                    self.followup
                        .lock()
                        .map_err(|_| checker_owner_unknown("PalsFollowupResetEvidence"))?
                        .capture_owner(&engine.followup_owner_snapshot());
                }
                reset.map_err(|error| pals_failure("PalsReset", &error, &engine))?;
                // A failed reset stays unapplied. A newer concurrent request
                // remains pending even when this older reset succeeds.
                self.reset_applied
                    .fetch_max(reset_requested, Ordering::AcqRel);
            }
            observation = AttemptObservation::PalsStarted;
            let outcome = engine.search_with_progress(
                position,
                rz_search::pals::engine::PalsLimits {
                    // End normal PALS work at the soft target. The existing
                    // hard/output boundaries retain time for report validation,
                    // physical completion, owner notification and UCI output.
                    deadline: context
                        .control
                        .soft_deadline
                        .min(context.control.admission_deadline),
                    max_rounds: self.max_rounds,
                    max_cpu_nodes: context.control.max_simulations,
                    cpu_depth: self.cpu_depth,
                },
                &context.control.cancellation,
                &mut *progress,
            );
            #[cfg(feature = "search-work-receipts")]
            match self.followup.try_lock() {
                Ok(mut journal) if journal.enabled() => {
                    if let Some(snapshot) = engine.last_followup_execution_snapshot() {
                        journal.capture_search(snapshot);
                    } else {
                        journal.incomplete(
                            crate::pals_attestation::followup::PalsCaptureFailureV4::CaptureIncomplete,
                        );
                    }
                }
                Ok(_) => {}
                Err(_) => self
                    .followup_capture_complete
                    .store(false, Ordering::Release),
            }
            if let Some(counters) = engine.last_search_counters() {
                observation = AttemptObservation::Pals {
                    counters,
                    root_coverage_observed: counters.root_scope_observation_complete,
                };
            }
            self.capture_checker(&engine)?;
            let report = outcome.map_err(|error| pals_failure("PalsSearch", &error, &engine))?;
            if report.resolver_version != self.resolver_version {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "PalsResolverIdentity",
                    detail: "returned PALS resolver version differs from its startup registration"
                        .into(),
                });
            }
            if context.accepts().map_err(|error| SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "PalsAuthority",
                detail: error.to_string(),
            })? {
                if let Some(movement) = report.best_move {
                    progress(movement);
                }
            }
            Ok(SearchSessionReport {
                mate_strategy: None,
                best_move: report.best_move,
                work: DriverWork::Pals {
                    rounds: report.counters.rounds,
                    cpu_nodes: report.counters.cpu_nodes,
                    completed_tasks: report.counters.completed_cpu_tasks,
                    consumed_role_outputs: report.counters.consumed_role_outputs,
                    retained_situations: report.counters.retained_situations,
                },
            })
        })();
        retain_work_failure(&self.work, context, result, observation)
    }
}
fn checker_owner_unknown(code: &'static str) -> SearchSessionFailure {
    SearchSessionFailure {
        physical_completion: DriverPhysicalCompletion::Unknown,
        code,
        detail: "checker owner completion could not be confirmed within its finite boundary".into(),
    }
}

fn pals_failure<M: rz_search::pals::engine::RoleModel>(
    code: &'static str,
    error: &rz_search::pals::engine::PalsError,
    engine: &rz_search::pals::engine::PalsEngine<M>,
) -> SearchSessionFailure {
    let role_unknown = matches!(
        error,
        rz_search::pals::engine::PalsError::Role(
            rz_search::pals::engine::RoleError::PhysicalCompletionUnknown
        )
    );
    // An idle persistent process need not have exited or drained. Only actual
    // quarantine/ownership loss establishes unconfirmed physical completion.
    let checker_unknown = engine
        .checker_last_attempt()
        .and_then(|attempt| attempt.external.as_ref())
        .is_some_and(|attempt| attempt.process.quarantined || attempt.process.ownership_lost);
    let mut failure = if role_unknown {
        SearchSessionFailure::physical_completion_unknown()
    } else {
        SearchSessionFailure::debug(code, error)
    };
    if checker_unknown {
        failure.physical_completion = DriverPhysicalCompletion::Unknown;
    }
    if let Some(cleanup_error) = engine.last_cpu_checkpoint_cleanup_error() {
        // Keep the original failure and completion classification. The cleanup
        // is a distinct typed cause; each backend-provided detail stays bounded.
        let secondary = SearchSessionFailure::debug("CpuCheckpointCleanup", cleanup_error);
        failure.detail.push_str("; ");
        failure.detail.push_str(&secondary.to_string());
        if matches!(
            cleanup_error,
            rz_search::pals::engine::PalsError::Role(
                rz_search::pals::engine::RoleError::PhysicalCompletionUnknown
            )
        ) {
            failure.physical_completion = DriverPhysicalCompletion::Unknown;
        }
    }
    failure
}

/// Capture only actual immutable engine getters. A result's resolver/score does
/// not identify this lane and cannot establish that its recheck branch executed.
fn selected_search_policy<M: rz_search::pals::engine::RoleModel>(
    engine: &rz_search::pals::engine::PalsEngine<M>,
    requested: PostRepairRecheckPolicy,
) -> Result<Option<PalsSearchPolicyIdentity>, SearchSessionFailure> {
    let mismatch = || SearchSessionFailure {
        physical_completion: DriverPhysicalCompletion::Confirmed,
        code: "PalsSearchPolicyIdentity",
        detail: "actual PALS search/refinement getters differ from selected startup policy".into(),
    };
    if engine.post_repair_recheck_policy() != requested {
        return Err(mismatch());
    }
    if requested == PostRepairRecheckPolicy::Disabled {
        let expected = match engine.search_identity() {
            rz_search::pals::engine::PALS_SEARCH_VERSION => None,
            rz_search::pals::engine::PALS_FOLLOWUP_SEARCH_VERSION => {
                Some(rz_search::pals::engine::PALS_FOLLOWUP_CONDITIONS)
            }
            _ => return Err(mismatch()),
        };
        if engine.refinement_conditions() != expected {
            return Err(mismatch());
        }
        return Ok(None);
    }
    let conditions = engine.refinement_conditions().ok_or_else(mismatch)?;
    let identity = PalsSearchPolicyIdentity {
        version: requested
            .registration_version()
            .ok_or_else(mismatch)?
            .into(),
        policy: requested.registration_policy().ok_or_else(mismatch)?.into(),
        search_identity: engine.search_identity().into(),
        conditions_sha256: Sha256::digest(conditions.as_bytes()).into(),
    };
    identity
        .validate()
        .map_err(|error| SearchSessionFailure::debug("PalsSearchPolicyIdentity", &error))?;
    Ok(Some(identity))
}

fn refinement_implementation(base: Digest, policy: &PalsSearchPolicyIdentity) -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"rz-uci-pals-post-repair-recheck-session/1\0");
    hash.update(base.0);
    for value in [&policy.version, &policy.policy, &policy.search_identity] {
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value.as_bytes());
    }
    hash.update(policy.conditions_sha256);
    Digest(hash.finalize().into())
}

/// Canonical length-prefixed declaration hash. It binds every admitted checker
/// field without depending on Debug formatting, platform usize width or paths.
/// It is a source namespace, not independent evidence of foreign execution.
fn checker_implementation(
    registration: &PalsCheckerRegistration,
    config: &rz_search::pals::engine::PalsConfig,
    max_rounds: u64,
    max_cpu_nodes: u64,
    cpu_depth: u16,
) -> Digest {
    fn text(hash: &mut Sha256, value: &str) {
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value.as_bytes());
    }
    fn optional_text(hash: &mut Sha256, value: Option<&str>) {
        hash.update([u8::from(value.is_some())]);
        if let Some(value) = value {
            text(hash, value);
        }
    }
    fn sizes(hash: &mut Sha256, depth: u16, prefix: usize, roots: usize) {
        hash.update(depth.to_le_bytes());
        hash.update((prefix as u64).to_le_bytes());
        hash.update((roots as u64).to_le_bytes());
    }
    let mut hash = Sha256::new();
    hash.update(b"rz-uci-pals-checker-session/2\0");
    text(&mut hash, rz_search::pals::engine::PALS_SEARCH_VERSION);
    text(&mut hash, registration.resolver_version);
    text(&mut hash, registration.resolver_semantics);
    text(&mut hash, &registration.role_model);
    text(&mut hash, &registration.conditions);
    match &registration.identity {
        CheckerIdentity::Owned(value) => {
            hash.update([0]);
            text(&mut hash, &value.semantics);
            optional_text(&mut hash, value.weights_sha256.as_deref());
            match &value.training {
                rz_search::cpu_value::CpuTrainingState::Bootstrap => hash.update([0]),
                rz_search::cpu_value::CpuTrainingState::Untrained => hash.update([1]),
                rz_search::cpu_value::CpuTrainingState::Learned {
                    run_id,
                    steps,
                    dataset_sha256,
                } => {
                    hash.update([2]);
                    text(&mut hash, run_id);
                    hash.update(steps.to_le_bytes());
                    text(&mut hash, dataset_sha256);
                }
            }
        }
        CheckerIdentity::ExternalUci(value) => {
            hash.update([1]);
            for value in [
                &value.adapter_semantics,
                &value.binary_sha256,
                &value.launch_arguments_sha256,
                &value.declared_name,
                &value.declared_version,
                &value.declared_source,
                &value.declared_license,
            ] {
                text(&mut hash, value);
            }
            hash.update((value.options.len() as u64).to_le_bytes());
            for (name, value) in &value.options {
                text(&mut hash, name);
                text(&mut hash, value);
            }
            hash.update((value.assets.len() as u64).to_le_bytes());
            for asset in &value.assets {
                text(&mut hash, &asset.purpose);
                text(&mut hash, &asset.sha256);
            }
            optional_text(&mut hash, value.model_metadata.weights_sha256.as_deref());
            match value.model_metadata.training {
                rz_search::cpu_checker::ExternalTrainingKnowledge::Unknown => hash.update([0]),
            }
            optional_text(&mut hash, value.model_metadata.declared_rights.as_deref());
            optional_text(&mut hash, value.model_metadata.precision.as_deref());
        }
    }
    let caps = registration.capabilities;
    sizes(
        &mut hash,
        caps.max_depth,
        caps.max_prefix_plies,
        caps.max_root_moves,
    );
    hash.update([
        u8::from(caps.root_moves),
        u8::from(caps.divergence),
        u8::from(caps.resume),
        match caps.selective_search {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
    ]);
    hash.update([u8::from(registration.owned_descriptor.is_some())]);
    if let Some(owned) = &registration.owned_descriptor {
        text(&mut hash, owned.search_identity);
        text(&mut hash, &owned.search_conditions);
        text(&mut hash, owned.config.profile.identity());
        hash.update((owned.config.tt_entries as u64).to_le_bytes());
        hash.update(owned.config.max_depth.to_le_bytes());
        hash.update(owned.config.quiescence_ply.to_le_bytes());
        let caps = owned.capabilities;
        sizes(
            &mut hash,
            caps.max_depth,
            caps.max_prefix_plies,
            caps.max_root_moves,
        );
        hash.update([
            u8::from(caps.root_moves),
            u8::from(caps.divergence),
            u8::from(caps.completed_iteration_resume),
            u8::from(caps.selective_reductions),
        ]);
    }
    hash.update([u8::from(registration.model_value.is_some())]);
    if let Some(value) = &registration.model_value {
        for text_value in [
            &value.semantics,
            &value.model,
            &value.encoding,
            &value.precision,
        ] {
            text(&mut hash, text_value);
        }
        hash.update(value.model_epoch);
    }
    for value in [
        config.beam_width,
        config.line_plies,
        config.max_nodes,
        config.max_records,
    ] {
        hash.update((value as u64).to_le_bytes());
    }
    hash.update(config.max_role_calls.to_le_bytes());
    hash.update(config.cpu_nodes_per_task.to_le_bytes());
    hash.update(max_rounds.to_le_bytes());
    hash.update(max_cpu_nodes.to_le_bytes());
    hash.update(cpu_depth.to_le_bytes());
    Digest(hash.finalize().into())
}
#[derive(Clone, Debug)]
pub struct SearchSessionReport {
    /// The actual checked strategy belongs to this immutable final report.
    /// Numeric CP/WDL and a later global observation cannot create it.
    pub mate_strategy: Option<rz_search::pals::rovent_frontier::FrontierMateStrategy>,
    pub best_move: Option<BoardMove>,
    pub work: DriverWork,
}
#[derive(Clone, Debug)]
pub struct SearchSessionFailure {
    pub physical_completion: DriverPhysicalCompletion,
    pub code: &'static str,
    pub detail: String,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DriverPhysicalCompletion {
    Confirmed,
    Unknown,
}
impl SearchSessionFailure {
    pub fn physical_completion_unknown() -> Self {
        Self {
            physical_completion: DriverPhysicalCompletion::Unknown,
            code: "PhysicalCompletionUnknown",
            detail: "search driver retains quarantined/unconfirmed physical work".into(),
        }
    }
    /// A role backend may supply a dynamic cause. Preserve a finite receipt
    /// without formatting or retaining an unbounded external backend string.
    pub(crate) fn debug(code: &'static str, value: &dyn fmt::Debug) -> Self {
        struct Detail {
            text: String,
            truncated: bool,
        }
        impl fmt::Write for Detail {
            fn write_str(&mut self, value: &str) -> fmt::Result {
                let available = 512usize.saturating_sub(self.text.len());
                if value.len() <= available {
                    self.text.push_str(value);
                    return Ok(());
                }
                let mut end = available;
                while !value.is_char_boundary(end) {
                    end -= 1;
                }
                self.text.push_str(&value[..end]);
                self.truncated = true;
                Err(fmt::Error)
            }
        }
        let mut detail = Detail {
            text: String::with_capacity(512),
            truncated: false,
        };
        let _ = fmt::write(&mut detail, format_args!("{value:?}"));
        if detail.truncated {
            detail.text.push_str(" [cause truncated]");
        }
        Self {
            physical_completion: DriverPhysicalCompletion::Confirmed,
            code,
            detail: detail.text,
        }
    }
}
impl fmt::Display for SearchSessionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.detail)
    }
}
impl std::error::Error for SearchSessionFailure {}

/// Implementations own their queues, evaluator and physical leases. Returning
/// from `run` acknowledges bounded completion/drain, not just logical cancel.
/// Progress is advisory: UCI checks its exact root legal set and authority.
pub trait SearchSessionDriver: Send + Sync + 'static {
    /// Opt in only when the actual root owner consumes the effective clock.
    /// Legacy hit-to-stop behavior remains a separate compatibility path.
    fn supports_ponder_clock(&self) -> bool {
        false
    }
    /// The node unit is explicit. Legacy simulation limits are not reused as
    /// Rovent position-visit limits or converted into role/NN call counts.
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        crate::bridge::NodeBudgetSemantics::LegacySimulations
    }
    /// Pure preadmission, before replacing/canceling an existing UCI search.
    /// It must not create a root, clock, worker, native call or live capability.
    fn validate_go(
        &self,
        _position: &Position,
        _go: &crate::parser::GoLimits,
    ) -> Result<(), ContractError> {
        Ok(())
    }
    /// Advertise only a driver that consumes the root-relative depth and
    /// Rules mate goals. Legacy drivers keep admission disabled.
    fn search_goal_capability(&self) -> Option<std::num::NonZeroU16> {
        None
    }
    /// Close the complete search-session owner after its workers have joined.
    /// This is distinct from ending only an optional CPU checker. None is an
    /// unavailable/legacy interface and never an invented successful fence.
    fn finish_session(
        &self,
        _deadline: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        Ok(None)
    }
    #[cfg(feature = "search-work-receipts")]
    fn followup_lifecycle(
        &self,
    ) -> Result<Option<crate::pals_attestation::followup::PalsFollowupLifecycleV4>, ContractError>
    {
        Ok(None)
    }
    #[cfg(feature = "search-work-receipts")]
    fn seal_followup_lifecycle(&self, _protocol_retired: bool) -> Result<(), ContractError> {
        Ok(())
    }
    fn kind(&self) -> SearchKind;
    fn implementation(&self) -> Digest;
    fn identity(&self) -> EngineIdentity;
    /// None is unavailable observation, never an invented all-zero receipt.
    fn work_receipt(&self) -> Result<Option<ProcessSearchWorkReceipt>, ContractError> {
        Ok(None)
    }
    /// None means this driver has no PALS checker registration, not a foreign
    /// helper with an empty or implicitly selected identity.
    fn pals_checker_registration(&self) -> Option<PalsCheckerRegistration> {
        None
    }
    fn checker_started(&self) -> Option<bool> {
        None
    }
    fn checker_startup_uci(&self) -> Result<Option<ExternalUciIdentity>, SearchSessionFailure> {
        Ok(None)
    }
    #[cfg(feature = "search-work-receipts")]
    fn checker_startup_resources(
        &self,
    ) -> Result<
        Option<crate::pals_attestation::checker::resource::ReadyBoundaryResources>,
        SearchSessionFailure,
    > {
        Ok(None)
    }
    #[cfg(feature = "search-work-receipts")]
    fn checker_startup_resource_unavailable(
        &self,
    ) -> Result<Option<&'static str>, SearchSessionFailure> {
        Ok(None)
    }
    fn start_checker(
        &self,
        _deadline: Instant,
        _cancel: Arc<AtomicBool>,
    ) -> Result<Option<bool>, SearchSessionFailure> {
        Ok(None)
    }
    fn checker_attempts(&self) -> Result<Option<Vec<CheckerAttempt>>, SearchSessionFailure> {
        Ok(None)
    }
    fn checker_last_attempt(&self) -> Result<Option<CheckerAttempt>, SearchSessionFailure> {
        Ok(None)
    }
    /// Ends only the CPU-checker owner. The role-model owner has an independent
    /// fence and receipt. None means no checker shutdown interface is present.
    fn finish_checker(
        &self,
        _deadline: Instant,
    ) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        Ok(None)
    }
    fn checker_shutdown(&self) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        Ok(None)
    }
    fn accept_uci_output(
        &self,
        _authority: rz_contracts::pals::SearchAuthority,
        _bestmove: &str,
        _from_report: bool,
    ) -> Result<(), ContractError> {
        Ok(())
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        Ok(())
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure>;
}

/// Own Rust CPU_R. No ModelAdapter, ORT session, CUDA provider or EvalOutput is
/// constructed. One owner retains bounded scratch/TT and exact Rules history.
pub struct CpuSessionDriver {
    config: rz_search::cpu::CpuConfig,
    engine: Mutex<rz_search::cpu::CpuEngine>,
    reset_pending: AtomicBool,
    max_nodes: u64,
    work: ProcessWorkJournal,
}
impl CpuSessionDriver {
    pub fn new(
        config: rz_search::cpu::CpuConfig,
        max_nodes: u64,
    ) -> Result<Self, SearchSessionFailure> {
        if max_nodes == 0 {
            return Err(SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "CpuConfiguration",
                detail: "CPU node limit must be positive".into(),
            });
        }
        let engine = rz_search::cpu::CpuEngine::new(config.clone())
            .map_err(|error| SearchSessionFailure::debug("CpuConfiguration", &error))?;
        Ok(Self {
            config,
            engine: Mutex::new(engine),
            reset_pending: AtomicBool::new(false),
            max_nodes,
            work: ProcessWorkJournal::new(SearchKind::Cpu),
        })
    }
}
impl SearchSessionDriver for CpuSessionDriver {
    fn kind(&self) -> SearchKind {
        SearchKind::Cpu
    }
    fn implementation(&self) -> Digest {
        // Semantic search/profile configuration identity, not a weights hash.
        // Source and executable SHA-256 are registered separately by experiments.
        let mut hash = Sha256::new();
        hash.update(b"rz-uci-cpu-session/1\0");
        hash.update(rz_search::cpu::CPU_SEARCH_VERSION.as_bytes());
        hash.update(rz_search::cpu::BOOTSTRAP_SCORE_VERSION.as_bytes());
        hash.update(self.config.profile.identity().as_bytes());
        hash.update((self.config.tt_entries as u64).to_le_bytes());
        hash.update(self.config.max_depth.to_le_bytes());
        hash.update(self.config.quiescence_ply.to_le_bytes());
        hash.update(self.max_nodes.to_le_bytes());
        Digest(hash.finalize().into())
    }
    fn identity(&self) -> EngineIdentity {
        EngineIdentity {
            name: "RoveZero own Rust CPU_R".into(),
            author: "RoveZero contributors".into(),
        }
    }
    fn work_receipt(&self) -> Result<Option<ProcessSearchWorkReceipt>, ContractError> {
        self.work.snapshot().map(Some)
    }
    fn accept_uci_output(
        &self,
        authority: rz_contracts::pals::SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        self.work
            .accept_cpu_output(authority, bestmove, from_report)
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        self.reset_pending.store(true, Ordering::Release);
        match self.engine.try_lock() {
            Ok(mut engine) => {
                engine.clear();
                self.reset_pending.store(false, Ordering::Release);
                Ok(())
            }
            Err(TryLockError::WouldBlock) => Ok(()),
            Err(TryLockError::Poisoned(_)) => Err(ContractError::new(
                ErrorCode::BackendFailure,
                Stage::Admission,
                "CPU session owner poisoned",
            )),
        }
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        self.work
            .begin()
            .map_err(|error| SearchSessionFailure::debug("CpuWorkReceipt", &error))?;
        let mut observation = AttemptObservation::NoWork;
        let result = (|| {
            if context.goal.has_goal() {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "CpuUnsupportedSearchGoal",
                    detail: "legacy CPU session does not implement Rovent depth/mate goals".into(),
                });
            }
            if context.control.max_simulations > self.max_nodes {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "CpuNodeLimit",
                    detail: "UCI node request exceeds the registered CPU limit".into(),
                });
            }
            let mut engine = loop {
                if !context.accepts().map_err(|error| SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "CpuAuthority",
                    detail: error.to_string(),
                })? {
                    return Ok(SearchSessionReport {
                        mate_strategy: None,
                        best_move: None,
                        work: DriverWork::Cpu {
                            nodes: 0,
                            completed_depth: 0,
                        },
                    });
                }
                match self.engine.try_lock() {
                    Ok(engine) => break engine,
                    Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(1)),
                    Err(TryLockError::Poisoned(_)) => {
                        return Err(SearchSessionFailure {
                            physical_completion: DriverPhysicalCompletion::Confirmed,
                            code: "CpuOwner",
                            detail: "persistent CPU owner poisoned".into(),
                        });
                    }
                }
            };
            if !context.accepts().map_err(|error| SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "CpuAuthority",
                detail: error.to_string(),
            })? {
                return Ok(SearchSessionReport {
                    mate_strategy: None,
                    best_move: None,
                    work: DriverWork::Cpu {
                        nodes: 0,
                        completed_depth: 0,
                    },
                });
            }
            if self.reset_pending.swap(false, Ordering::AcqRel) {
                engine.clear();
            }
            let limits = rz_search::cpu::CpuLimits {
                max_depth: self.config.max_depth,
                max_nodes: context.control.max_simulations,
                deadline: Some(context.control.admission_deadline),
            };
            observation = AttemptObservation::CpuStarted;
            // The existing restricted API preserves child legality and TT
            // namespace. It has no per-iteration observer; only the final
            // validated report is published for this branch.
            let attempted = if let Some(root_moves) = &context.root_moves {
                engine.analyze_root_moves(
                    position,
                    root_moves,
                    limits,
                    &context.control.cancellation,
                )
            } else {
                engine.analyze_with_progress(
                    position,
                    limits,
                    &context.control.cancellation,
                    |iteration| {
                        if let Some(movement) = iteration.best_move {
                            progress(movement);
                        }
                    },
                )
            };
            if let Some(work) = engine.last_attempt_work() {
                observation = AttemptObservation::CpuFailed(work);
            }
            let report =
                attempted.map_err(|error| SearchSessionFailure::debug("CpuSearch", &error))?;
            observation = AttemptObservation::cpu(&report, limits.max_depth);
            if report.best_move.is_some_and(|movement| {
                context
                    .root_moves
                    .as_ref()
                    .is_some_and(|moves| !moves.contains(&movement))
            }) {
                return Err(SearchSessionFailure {
                    physical_completion: DriverPhysicalCompletion::Confirmed,
                    code: "CpuRootRestriction",
                    detail: "CPU result escaped the immutable root restriction".into(),
                });
            }
            if context.accepts().map_err(|error| SearchSessionFailure {
                physical_completion: DriverPhysicalCompletion::Confirmed,
                code: "CpuAuthority",
                detail: error.to_string(),
            })? {
                if let Some(movement) = report.best_move {
                    progress(movement);
                }
            }
            Ok(SearchSessionReport {
                mate_strategy: None,
                best_move: report.best_move,
                work: DriverWork::Cpu {
                    nodes: report.nodes,
                    completed_depth: report.completed_depth,
                },
            })
        })();
        retain_work_failure(&self.work, context, result, observation)
    }
}
fn retain_work_failure(
    journal: &ProcessWorkJournal,
    context: &SearchSessionContext,
    result: Result<SearchSessionReport, SearchSessionFailure>,
    observation: AttemptObservation,
) -> Result<SearchSessionReport, SearchSessionFailure> {
    if let Err(error) = journal.finish(context, &result, observation) {
        return match result {
            Err(mut primary) => {
                primary.detail.push_str("; work observation unavailable");
                Err(primary)
            }
            Ok(_) => Err(SearchSessionFailure::debug("WorkReceipt", &error)),
        };
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::{GameGeneration, ProcessEpoch, RootGeneration, pals::SearchAuthority};
    use rz_search::{
        cpu::CpuConfig,
        pals::engine::{
            DivergenceQuery, PalsConfig, RoleError, RoleEvaluation, RoleModel, RoleQuery,
        },
        time::{TimeBudget, TimeBudgetConfig, TimeControl},
    };

    struct DeadlineObserver {
        observed: Arc<Mutex<Vec<Instant>>>,
    }
    impl DeadlineObserver {
        fn observe(&self, deadline: Instant) -> RoleError {
            self.observed.lock().unwrap().push(deadline);
            // Stop at the first actual role query, without sleeps or CPU work.
            RoleError::Backend("test role deadline observed".into())
        }
    }
    impl RoleModel for DeadlineObserver {
        fn identity(&self) -> &str {
            "test-only-pals-role-deadline-observer"
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            Err(self.observe(query.deadline))
        }
        fn reply(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            Err(self.observe(query.deadline))
        }
        fn repair(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            Err(self.observe(query.deadline))
        }
        fn divergences(&mut self, query: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            Err(self.observe(query.deadline))
        }
    }

    #[test]
    fn pals_driver_role_query_uses_soft_target_without_extending_admission() {
        for admission_is_earlier in [false, true] {
            let observed = Arc::new(Mutex::new(Vec::new()));
            let driver = PalsSessionDriver::new(
                PalsConfig::default(),
                DeadlineObserver {
                    observed: Arc::clone(&observed),
                },
                CpuConfig {
                    tt_entries: 0,
                    ..CpuConfig::default()
                },
                1,
                1,
                1,
                EngineIdentity {
                    name: "PALS role deadline consumer test".into(),
                    author: "RoveZero contributors".into(),
                },
            )
            .unwrap();
            // Reproduce the CPU06 low-clock allocation while keeping every
            // timestamp in the future; the probe never waits for a deadline.
            let start = Instant::now() + Duration::from_secs(60);
            let budget = TimeBudget::new(
                start,
                TimeControl::Clock {
                    remaining: Duration::from_millis(1010),
                    increment: Duration::from_millis(1000),
                    moves_to_go: None,
                },
                TimeBudgetConfig::default(),
            )
            .unwrap();
            let mut control = SearchControl::from_budget(&budget, 1);
            let expected = if admission_is_earlier {
                // Shared contracts also permit admission to precede soft.
                control.soft_deadline = control.deadline;
                budget.admission_deadline
            } else {
                budget.soft_deadline
            };
            let authority = SearchAuthority {
                epoch: ProcessEpoch(1),
                game: GameGeneration(1),
                root: RootGeneration(1),
                implementation: driver.implementation(),
            };
            let context = SearchSessionContext {
                authority,
                root_moves: None,
                goal: rz_contracts::rovent::SearchGoal::default(),
                original_go: None,
                original_time_budget: None,
                original_start: None,
                effective_clock: None,
                requested_nodes: None,
                control,
                cancellation: CancelToken::new(),
                shutdown_deadline: budget.output_deadline,
                current: Arc::new(Mutex::new(SessionScope::Search(authority))),
            };
            let failure = driver
                .run(&Position::startpos(), &context, &mut |_| {})
                .unwrap_err();
            assert_eq!(failure.code, "PalsSearch");
            assert!(failure.detail.contains("test role deadline observed"));
            assert_eq!(*observed.lock().unwrap(), [expected]);
            assert_eq!(context.control.deadline, budget.hard_deadline);
            assert!(!context.cancellation.is_canceled());
        }
    }

    #[derive(Default)]
    struct CheckerProbeState {
        starts: AtomicU64,
        resets: AtomicU64,
        analyses: AtomicU64,
        shutdowns: AtomicU64,
        fail_start: AtomicBool,
        fail_reset: AtomicBool,
        fail_shutdown: AtomicBool,
        late_start: AtomicBool,
        quarantine: AtomicBool,
        block_analysis: AtomicBool,
        analysis_entered: AtomicBool,
        block_reset: AtomicBool,
        reset_entered: AtomicBool,
        reset_deadlines: Mutex<Vec<Instant>>,
    }
    /// In-process lifecycle fixture only; its synthetic shutdown states are
    /// not real UCI subprocess, pipe drain or engine-strength evidence.
    struct CheckerProbe {
        identity: CheckerIdentity,
        state: Arc<CheckerProbeState>,
        started: bool,
        attempt: Option<CheckerAttempt>,
    }
    impl CheckerProbe {
        fn new(state: Arc<CheckerProbeState>) -> Self {
            Self {
                identity: CheckerIdentity::ExternalUci(
                    rz_search::cpu_checker::ExternalCheckerIdentity {
                        adapter_semantics: "test-only-driver-checker/1".into(),
                        binary_sha256: "a".repeat(64),
                        launch_arguments_sha256: "b".repeat(64),
                        declared_name: "declared fixture name".into(),
                        declared_version: "fixture-v1".into(),
                        declared_source: "in-process unit fixture".into(),
                        declared_license: "MIT".into(),
                        options: std::collections::BTreeMap::new(),
                        assets: Vec::new(),
                        model_metadata: rz_search::cpu_checker::ExternalModelMetadata {
                            weights_sha256: None,
                            training: rz_search::cpu_checker::ExternalTrainingKnowledge::Unknown,
                            declared_rights: None,
                            precision: None,
                        },
                    },
                ),
                state,
                started: false,
                attempt: None,
            }
        }
        fn record(&mut self, request: u64, nodes: Option<u64>) {
            self.attempt = Some(CheckerAttempt {
                work: rz_search::cpu_checker::CheckerWork {
                    nodes,
                    qnodes: None,
                    tt_hits: None,
                },
                elapsed: Duration::ZERO,
                external: Some(rz_search::cpu_checker::ExternalAttemptEvidence {
                    request_id: request,
                    partial_report: None,
                    process: CheckerShutdown {
                        quarantined: self.state.quarantine.load(Ordering::Acquire),
                        ..CheckerShutdown::default()
                    },
                }),
            });
        }
    }
    impl CpuChecker for CheckerProbe {
        fn identity(&self) -> &CheckerIdentity {
            &self.identity
        }
        fn conditions(&self) -> &str {
            "test-only-bounded-checker;not-a-real-process"
        }
        fn capabilities(&self) -> CheckerCapabilities {
            CheckerCapabilities {
                max_depth: 64,
                max_prefix_plies: 4096,
                max_root_moves: 256,
                root_moves: true,
                divergence: true,
                resume: false,
                selective_search: None,
            }
        }
        fn start(
            &mut self,
            _: Instant,
            cancel: &AtomicBool,
        ) -> Result<(), rz_search::cpu_checker::CheckerError> {
            self.state.starts.fetch_add(1, Ordering::AcqRel);
            self.record(0, None);
            if self.state.fail_start.load(Ordering::Acquire) {
                return Err(rz_search::cpu_checker::CheckerError::External {
                    stage: "test-startup",
                    code: "preserved-startup-failure",
                });
            }
            self.started = true;
            if self.state.late_start.load(Ordering::Acquire) {
                cancel.store(true, Ordering::Release);
            }
            Ok(())
        }
        fn startup_uci(&self) -> Option<ExternalUciIdentity> {
            self.started.then(|| ExternalUciIdentity {
                name: "actual fixture handshake name".into(),
                author: Some("fixture author".into()),
            })
        }
        fn last_attempt(&self) -> Option<&CheckerAttempt> {
            self.attempt.as_ref()
        }
        fn reset_attempt(&mut self) {
            self.attempt = None;
        }
        fn analyze(
            &mut self,
            _: &Position,
            limits: rz_search::cpu::CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<rz_search::cpu_checker::CheckerReport, rz_search::cpu_checker::CheckerError>
        {
            let request = self.state.analyses.fetch_add(1, Ordering::AcqRel) + 1;
            self.state.analysis_entered.store(true, Ordering::Release);
            while self.state.block_analysis.load(Ordering::Acquire)
                && !cancel.load(Ordering::Acquire)
                && limits
                    .deadline
                    .is_some_and(|deadline| Instant::now() < deadline)
            {
                thread::yield_now();
            }
            self.record(request, Some(7));
            Err(rz_search::cpu_checker::CheckerError::External {
                stage: "test-analysis",
                code: "actual-observed-work-failure",
            })
        }
        fn analyze_root_moves(
            &mut self,
            position: &Position,
            _: &[BoardMove],
            limits: rz_search::cpu::CpuLimits,
            cancel: &AtomicBool,
        ) -> Result<rz_search::cpu_checker::CheckerReport, rz_search::cpu_checker::CheckerError>
        {
            self.analyze(position, limits, cancel)
        }
        fn new_game(
            &mut self,
            deadline: Instant,
            cancel: &AtomicBool,
        ) -> Result<(), rz_search::cpu_checker::CheckerError> {
            self.state.resets.fetch_add(1, Ordering::AcqRel);
            self.state.reset_deadlines.lock().unwrap().push(deadline);
            self.state.reset_entered.store(true, Ordering::Release);
            while self.state.block_reset.load(Ordering::Acquire)
                && !cancel.load(Ordering::Acquire)
                && Instant::now() < deadline
            {
                thread::yield_now();
            }
            if self.state.fail_reset.load(Ordering::Acquire) {
                return Err(rz_search::cpu_checker::CheckerError::External {
                    stage: "test-reset",
                    code: "fallible-reset-not-acknowledged",
                });
            }
            Ok(())
        }
        fn shutdown(
            &mut self,
            _: Instant,
        ) -> Result<CheckerShutdown, rz_search::cpu_checker::CheckerError> {
            self.state.shutdowns.fetch_add(1, Ordering::AcqRel);
            let shutdown = CheckerShutdown {
                cleanup_complete: !self.state.fail_shutdown.load(Ordering::Acquire),
                quarantined: self.state.fail_shutdown.load(Ordering::Acquire),
                ..CheckerShutdown::default()
            };
            if let Some(attempt) = self
                .attempt
                .as_mut()
                .and_then(|value| value.external.as_mut())
            {
                attempt.process = shutdown;
            }
            if shutdown.cleanup_complete {
                Ok(shutdown)
            } else {
                Err(rz_search::cpu_checker::CheckerError::External {
                    stage: "test-shutdown",
                    code: "physical-owner-unconfirmed",
                })
            }
        }
    }
    fn checker_driver(
        state: Arc<CheckerProbeState>,
    ) -> PalsSessionDriver<rz_search::pals::engine::LegalOrderRoleMock> {
        PalsSessionDriver::new_with_checker(
            PalsConfig {
                beam_width: 1,
                line_plies: 2,
                cpu_nodes_per_task: 8,
                ..PalsConfig::default()
            },
            rz_search::pals::engine::LegalOrderRoleMock,
            CheckerProbe::new(state),
            1,
            8,
            1,
            EngineIdentity {
                name: "PALS driver checker test".into(),
                author: "RoveZero contributors".into(),
            },
        )
        .unwrap()
    }
    fn checker_context(driver: &dyn SearchSessionDriver) -> SearchSessionContext {
        let authority = SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(1),
            root: RootGeneration(1),
            implementation: driver.implementation(),
        };
        let now = Instant::now();
        let mut control = SearchControl::new(now + Duration::from_secs(5), 8);
        control.soft_deadline = now + Duration::from_secs(4);
        SearchSessionContext {
            authority,
            root_moves: None,
            goal: rz_contracts::rovent::SearchGoal::default(),
            original_go: None,
            original_time_budget: None,
            original_start: None,
            effective_clock: None,
            requested_nodes: None,
            control,
            cancellation: CancelToken::new(),
            shutdown_deadline: now + Duration::from_secs(6),
            current: Arc::new(Mutex::new(SessionScope::Search(authority))),
        }
    }

    #[test]
    fn nonblocking_authority_observation_preserves_busy_and_live_checks() {
        let driver = CpuSessionDriver::new(CpuConfig::default(), 64).unwrap();
        let mut context = checker_context(&driver);
        assert!(context.try_accepts().unwrap());
        {
            // Holding this exact mutex on this thread proves the observer
            // cannot wait for it, rather than relying on a timing threshold.
            let _held = context.current.lock().unwrap();
            let failure = context.try_accepts().unwrap_err();
            assert_eq!(failure.code, ErrorCode::ResourceExhausted);
            assert_eq!(failure.stage, Stage::Admission);
        }
        assert!(context.try_accepts().unwrap());
        context.cancellation.cancel();
        assert!(!context.try_accepts().unwrap());
        context.cancellation = CancelToken::new();
        context.control.cancellation.store(true, Ordering::Release);
        assert!(!context.try_accepts().unwrap());
        context.control.cancellation.store(false, Ordering::Release);
        let original = *context.current.lock().unwrap();
        *context.current.lock().unwrap() = SessionScope::Search(SearchAuthority {
            root: RootGeneration(2),
            ..context.authority
        });
        assert!(!context.try_accepts().unwrap());
        *context.current.lock().unwrap() = original;
        context.control.deadline = Instant::now();
        assert!(!context.try_accepts().unwrap());
        let current = Arc::clone(&context.current);
        let poisoned = thread::spawn(move || {
            let _held = current.lock().unwrap();
            panic!("intentional authority poison");
        });
        assert!(poisoned.join().is_err());
        assert_eq!(
            context.try_accepts().unwrap_err().code,
            ErrorCode::BackendFailure
        );
    }

    #[test]
    fn own_cpu_session_consumes_the_root_restriction_for_search_and_progress() {
        let driver = CpuSessionDriver::new(
            CpuConfig {
                tt_entries: 0,
                max_depth: 1,
                ..CpuConfig::default()
            },
            64,
        )
        .unwrap();
        let mut context = checker_context(&driver);
        let allowed = BoardMove::from_uci("e2e4").unwrap();
        context.root_moves = Some(Arc::from([allowed]));
        let mut observed = Vec::new();
        let report = driver
            .run(&Position::startpos(), &context, &mut |movement| {
                observed.push(movement)
            })
            .unwrap();
        assert_eq!(report.best_move, Some(allowed));
        assert!(!observed.is_empty());
        assert!(observed.iter().all(|movement| *movement == allowed));
        let DriverWork::Cpu { nodes, .. } = report.work else {
            panic!("CPU work only")
        };
        assert!(nodes > 0 && nodes <= context.control.max_simulations);
    }

    #[test]
    fn own_cpu_session_preserves_special_root_moves_and_rejects_goals_without_capability() {
        for (fen, movement) in [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
        ] {
            let driver = CpuSessionDriver::new(
                CpuConfig {
                    tt_entries: 0,
                    max_depth: 1,
                    ..CpuConfig::default()
                },
                64,
            )
            .unwrap();
            let mut context = checker_context(&driver);
            let allowed = BoardMove::from_uci(movement).unwrap();
            let position = Position::from_fen(fen).unwrap();
            assert!(position.legal_moves().contains(&allowed));
            context.root_moves = Some(Arc::from([allowed]));
            let report = driver.run(&position, &context, &mut |_| {}).unwrap();
            assert_eq!(report.best_move, Some(allowed));
        }
        let driver = CpuSessionDriver::new(
            CpuConfig {
                tt_entries: 0,
                ..CpuConfig::default()
            },
            64,
        )
        .unwrap();
        let mut context = checker_context(&driver);
        context.goal.depth_plies = std::num::NonZeroU16::new(1);
        let failure = driver
            .run(&Position::startpos(), &context, &mut |_| {
                panic!("unsupported goal must not publish progress")
            })
            .unwrap_err();
        assert_eq!(failure.code, "CpuUnsupportedSearchGoal");
        assert_eq!(
            failure.physical_completion,
            DriverPhysicalCompletion::Confirmed
        );
    }

    fn refinement_config() -> PalsConfig {
        PalsConfig {
            beam_width: 1,
            line_plies: 2,
            cpu_nodes_per_task: 8,
            ..PalsConfig::default()
        }
    }
    fn refinement_cpu_config() -> CpuConfig {
        CpuConfig {
            tt_entries: 0,
            ..CpuConfig::default()
        }
    }
    fn refinement_name() -> EngineIdentity {
        EngineIdentity {
            name: "PALS immutable recheck lane fixture".into(),
            author: "RoveZero contributors".into(),
        }
    }
    fn refinement_driver<M: RoleModel + 'static>(model: M) -> PalsSessionDriver<M> {
        PalsSessionDriver::new_with_refinement_policy(
            refinement_config(),
            model,
            refinement_cpu_config(),
            1,
            8,
            1,
            refinement_name(),
            PostRepairRecheckPolicy::SameRepairedLineOnceV1,
        )
        .unwrap()
    }
    fn refinement_checker() -> OwnedCpuChecker<rz_search::cpu::CpuEngine> {
        OwnedCpuChecker::new(rz_search::cpu::CpuEngine::new(refinement_cpu_config()).unwrap())
            .unwrap()
    }

    struct CancelingRoleFailure(RoleError);
    impl RoleModel for CancelingRoleFailure {
        fn identity(&self) -> &str {
            "test-only-canceling-role-failure"
        }
        fn propose(&mut self, query: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            query.cancel.store(true, Ordering::Release);
            Err(self.0.clone())
        }
        fn reply(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            unreachable!("the first Proposer call fails")
        }
        fn repair(&mut self, _: RoleQuery<'_>) -> Result<RoleEvaluation, RoleError> {
            unreachable!("the first Proposer call fails")
        }
        fn divergences(&mut self, _: DivergenceQuery<'_>) -> Result<Vec<f32>, RoleError> {
            unreachable!("the first Proposer call fails")
        }
    }

    /// This in-process fixture declares a stack policy without its disposal
    /// capability. It runs no CPU analysis and represents no physical GPU work.
    struct MissingStackDisposal(OwnedCpuChecker<rz_search::cpu::CpuEngine>);
    impl CpuChecker for MissingStackDisposal {
        fn identity(&self) -> &CheckerIdentity {
            self.0.identity()
        }
        fn conditions(&self) -> &str {
            self.0.conditions()
        }
        fn capabilities(&self) -> CheckerCapabilities {
            self.0.capabilities()
        }
        fn owned_descriptor(&self) -> Option<OwnedCheckerDescriptor> {
            self.0.owned_descriptor()
        }
        fn owned_resume_policy(&self) -> Option<rz_search::cpu::CpuResumePolicy> {
            self.0.owned_resume_policy()
        }
        fn last_attempt(&self) -> Option<&CheckerAttempt> {
            self.0.last_attempt()
        }
        fn reset_attempt(&mut self) {
            self.0.reset_attempt();
        }
        fn analyze(
            &mut self,
            _: &Position,
            _: rz_search::cpu::CpuLimits,
            _: &AtomicBool,
        ) -> Result<rz_search::cpu_checker::CheckerReport, rz_search::cpu_checker::CheckerError>
        {
            unreachable!("the first Proposer call fails before CPU admission")
        }
        fn new_game(
            &mut self,
            deadline: Instant,
            cancel: &AtomicBool,
        ) -> Result<(), rz_search::cpu_checker::CheckerError> {
            self.0.new_game(deadline, cancel)
        }
        fn shutdown(
            &mut self,
            deadline: Instant,
        ) -> Result<CheckerShutdown, rz_search::cpu_checker::CheckerError> {
            self.0.shutdown(deadline)
        }
    }

    #[test]
    fn pals_failure_preserves_secondary_checkpoint_cleanup_and_primary_completion() {
        for primary in [
            RoleError::Backend("actual primary role failure".into()),
            RoleError::PhysicalCompletionUnknown,
        ] {
            let cpu = rz_search::cpu::CpuEngine::with_resume_policy(
                refinement_cpu_config(),
                rz_search::cpu::CpuResumePolicy::PausedStack,
            )
            .unwrap();
            let checker = MissingStackDisposal(OwnedCpuChecker::new(cpu).unwrap());
            let mut engine = rz_search::pals::engine::PalsEngine::new_with_boxed_checker(
                refinement_config(),
                CancelingRoleFailure(primary.clone()),
                Box::new(checker),
            )
            .unwrap();
            let cancel = AtomicBool::new(false);
            let error = engine
                .search(
                    &Position::startpos(),
                    rz_search::pals::engine::PalsLimits {
                        deadline: Instant::now() + Duration::from_secs(1),
                        max_rounds: 1,
                        max_cpu_nodes: 8,
                        cpu_depth: 1,
                    },
                    &cancel,
                )
                .unwrap_err();
            assert!(
                matches!(&error, rz_search::pals::engine::PalsError::Role(actual) if actual == &primary)
            );
            assert!(matches!(
                engine.last_cpu_checkpoint_cleanup_error(),
                Some(rz_search::pals::engine::PalsError::Checker(
                    rz_search::cpu_checker::CheckerError::Unsupported(
                        "owned paused stack disposal"
                    )
                ))
            ));
            let failure = pals_failure("PalsSearch", &error, &engine);
            assert!(
                failure
                    .detail
                    .contains("CpuCheckpointCleanup: Checker(Unsupported")
            );
            assert!(failure.detail.contains("owned paused stack disposal"));
            if primary == RoleError::PhysicalCompletionUnknown {
                assert_eq!(failure.code, "PhysicalCompletionUnknown");
                assert_eq!(
                    failure.physical_completion,
                    DriverPhysicalCompletion::Unknown
                );
            } else {
                assert_eq!(failure.code, "PalsSearch");
                assert_eq!(
                    failure.physical_completion,
                    DriverPhysicalCompletion::Confirmed
                );
                assert!(failure.detail.contains("actual primary role failure"));
            }
            assert!(failure.detail.len() <= 1100);
        }
    }

    #[test]
    fn followup_constructor_starts_owned_raw_and_model_wdl_frozen_lanes() {
        for policy in [
            PostRepairRecheckPolicy::Disabled,
            PostRepairRecheckPolicy::FrozenModelWdlV2,
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
        ] {
            for resolver in [
                ResolverPolicy::OwnRawRestricted,
                ResolverPolicy::ModelWdlRestricted,
            ] {
                let driver = PalsSessionDriver::new_with_policies(
                    refinement_config(),
                    rz_search::pals::engine::LegalOrderRoleMock,
                    refinement_cpu_config(),
                    1,
                    8,
                    1,
                    refinement_name(),
                    resolver,
                    policy,
                    rz_search::cpu::CpuResumePolicy::CompletedIteration,
                )
                .unwrap();
                driver
                    .start_checker(
                        Instant::now() + Duration::from_secs(1),
                        Arc::new(AtomicBool::new(false)),
                    )
                    .unwrap();
                assert!(driver.checker_started());
                let (actual_resolver, actual_policy, identity, conditions) =
                    driver.selected_policies().unwrap();
                assert_eq!((actual_resolver, actual_policy), (resolver, policy));
                if policy == PostRepairRecheckPolicy::Disabled {
                    assert_eq!(
                        identity,
                        rz_search::pals::engine::PALS_FOLLOWUP_SEARCH_VERSION
                    );
                    assert_eq!(
                        conditions,
                        Some(rz_search::pals::engine::PALS_FOLLOWUP_CONDITIONS)
                    );
                } else {
                    assert_eq!(identity, policy.search_identity());
                    assert_eq!(conditions, policy.conditions());
                }
                let receipt = driver.work_receipt().unwrap().unwrap();
                let registered = receipt.pals_resolver.unwrap();
                assert_eq!(registered.version, resolver.version());
                assert_eq!(registered.semantics_sha256, resolver.semantics_sha256());
                assert_eq!(
                    receipt.pals_search_policy.is_some(),
                    policy != PostRepairRecheckPolicy::Disabled
                );
                assert_eq!(receipt.go_invocations, 0);
                assert_eq!(receipt.pals.unwrap().role_calls, Some(0));
                assert_eq!(
                    driver
                        .finish_checker(Instant::now() + Duration::from_secs(1))
                        .unwrap(),
                    CheckerShutdown::owned_no_process()
                );
            }
        }
    }

    #[test]
    fn followup_constructor_starts_external_model_wdl_frozen_lanes_without_search_work() {
        for policy in [
            PostRepairRecheckPolicy::Disabled,
            PostRepairRecheckPolicy::FrozenModelWdlV2,
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
        ] {
            let state = Arc::new(CheckerProbeState::default());
            let driver = PalsSessionDriver::new_with_boxed_checker_and_policies(
                refinement_config(),
                rz_search::pals::engine::LegalOrderRoleMock,
                Box::new(CheckerProbe::new(Arc::clone(&state))),
                1,
                8,
                1,
                refinement_name(),
                ResolverPolicy::ModelWdlRestricted,
                policy,
            )
            .unwrap();
            assert!(!driver.checker_started());
            assert_eq!(state.starts.load(Ordering::Acquire), 0);
            driver
                .start_checker(
                    Instant::now() + Duration::from_secs(1),
                    Arc::new(AtomicBool::new(false)),
                )
                .unwrap();
            assert!(driver.checker_started());
            assert_eq!(state.starts.load(Ordering::Acquire), 1);
            assert_eq!(state.analyses.load(Ordering::Acquire), 0);
            let receipt = driver.work_receipt().unwrap().unwrap();
            let resolver = receipt.pals_resolver.unwrap();
            assert_eq!(
                resolver.version,
                rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION
            );
            assert_eq!(
                resolver.semantics_sha256,
                ResolverPolicy::ModelWdlRestricted.semantics_sha256()
            );
            assert_eq!(
                receipt.pals_search_policy.is_some(),
                policy != PostRepairRecheckPolicy::Disabled
            );
            assert_eq!(receipt.go_invocations, 0);
            assert_eq!(receipt.pals.unwrap().role_calls, Some(0));
            assert!(
                driver
                    .finish_checker(Instant::now() + Duration::from_secs(1))
                    .unwrap()
                    .cleanup_complete
            );
            assert_eq!(state.shutdowns.load(Ordering::Acquire), 1);
        }
    }

    #[test]
    fn followup_constructor_keeps_legacy_recheck_own_resolver_only() {
        for policy in [
            PostRepairRecheckPolicy::SameRepairedLineOnceV1,
            PostRepairRecheckPolicy::ActualOpponentContinuationV1,
        ] {
            let result = PalsSessionDriver::new_with_policies(
                refinement_config(),
                rz_search::pals::engine::LegalOrderRoleMock,
                refinement_cpu_config(),
                1,
                8,
                1,
                refinement_name(),
                ResolverPolicy::ModelWdlRestricted,
                policy,
                rz_search::cpu::CpuResumePolicy::CompletedIteration,
            );
            let Err(failure) = result else {
                panic!("legacy own-only recheck admitted a model resolver")
            };
            assert_eq!(failure.code, "PalsSearchPolicyRegistration");
        }
    }

    #[test]
    fn disabled_policy_keeps_v1_v2_hashes_and_opt_in_has_a_separate_actual_namespace() {
        use rz_search::pals::engine::LegalOrderRoleMock;
        let legacy = PalsSessionDriver::new(
            refinement_config(),
            LegalOrderRoleMock,
            refinement_cpu_config(),
            1,
            8,
            1,
            refinement_name(),
        )
        .unwrap();
        let disabled = PalsSessionDriver::new_with_refinement_policy(
            refinement_config(),
            LegalOrderRoleMock,
            refinement_cpu_config(),
            1,
            8,
            1,
            refinement_name(),
            PostRepairRecheckPolicy::Disabled,
        )
        .unwrap();
        assert_eq!(legacy.implementation(), disabled.implementation());
        assert_eq!(
            legacy.work_receipt().unwrap(),
            disabled.work_receipt().unwrap()
        );
        assert!(legacy.search_policy_registration().is_none());
        let explicit = PalsSessionDriver::new_with_checker(
            refinement_config(),
            LegalOrderRoleMock,
            refinement_checker(),
            1,
            8,
            1,
            refinement_name(),
        )
        .unwrap();
        let explicit_disabled = PalsSessionDriver::new_with_checker_and_refinement_policy(
            refinement_config(),
            LegalOrderRoleMock,
            refinement_checker(),
            1,
            8,
            1,
            refinement_name(),
            PostRepairRecheckPolicy::Disabled,
        )
        .unwrap();
        assert_eq!(
            explicit.implementation(),
            explicit_disabled.implementation()
        );
        assert_eq!(
            explicit.work_receipt().unwrap(),
            explicit_disabled.work_receipt().unwrap()
        );
        let selected = refinement_driver(LegalOrderRoleMock);
        assert_ne!(selected.implementation(), legacy.implementation());
        assert_ne!(selected.implementation(), explicit.implementation());
        let policy = selected.search_policy_registration().unwrap();
        policy.validate().unwrap();
        let engine = selected.engine.lock().unwrap();
        assert_eq!(policy.search_identity, engine.search_identity());
        assert_eq!(
            policy.conditions_sha256,
            <[u8; 32]>::from(Sha256::digest(
                engine.refinement_conditions().unwrap().as_bytes(),
            ))
        );
        assert_eq!(
            selected
                .work_receipt()
                .unwrap()
                .unwrap()
                .pals_search_policy
                .as_ref(),
            Some(policy)
        );
        #[cfg(feature = "search-work-receipts")]
        assert_eq!(
            serde_json::to_vec(&legacy.work_receipt().unwrap()).unwrap(),
            serde_json::to_vec(&disabled.work_receipt().unwrap()).unwrap(),
        );
        let changed_budget = PalsSessionDriver::new_with_refinement_policy(
            refinement_config(),
            LegalOrderRoleMock,
            refinement_cpu_config(),
            1,
            9,
            1,
            refinement_name(),
            PostRepairRecheckPolicy::SameRepairedLineOnceV1,
        )
        .unwrap();
        assert_ne!(selected.implementation(), changed_budget.implementation());
    }

    #[test]
    fn actual_c_continuation_driver_registers_its_own_lane_and_rejects_cross_lane_getters() {
        use rz_search::pals::engine::LegalOrderRoleMock;
        let driver = PalsSessionDriver::new_with_refinement_policy(
            refinement_config(),
            LegalOrderRoleMock,
            refinement_cpu_config(),
            1,
            8,
            1,
            refinement_name(),
            PostRepairRecheckPolicy::ActualOpponentContinuationV1,
        )
        .unwrap();
        let old = refinement_driver(LegalOrderRoleMock);
        assert_ne!(driver.implementation(), old.implementation());
        let policy = driver.search_policy_registration().unwrap();
        assert_eq!(
            policy.selected_policy().unwrap(),
            PostRepairRecheckPolicy::ActualOpponentContinuationV1
        );
        assert_ne!(policy, old.search_policy_registration().unwrap());
        let engine = driver.engine.lock().unwrap();
        assert!(
            selected_search_policy(&engine, PostRepairRecheckPolicy::SameRepairedLineOnceV1)
                .is_err()
        );
        assert_eq!(
            selected_search_policy(
                &engine,
                PostRepairRecheckPolicy::ActualOpponentContinuationV1
            )
            .unwrap()
            .as_ref(),
            Some(policy)
        );
        drop(engine);
        let context = checker_context(&driver);
        context.cancellation.cancel();
        let report = driver
            .run(&Position::startpos(), &context, &mut |_| {})
            .unwrap();
        assert!(report.best_move.is_none());
        let receipt = driver.work_receipt().unwrap().unwrap();
        assert_eq!(receipt.pals_search_policy.as_ref(), Some(policy));
        assert_eq!(receipt.pals.unwrap().consumed_role_outputs, Some(0));
        #[cfg(feature = "search-work-receipts")]
        {
            let raw = serde_json::to_string(policy).unwrap();
            assert_eq!(
                serde_json::from_str::<PalsSearchPolicyIdentity>(&raw).unwrap(),
                *policy
            );
            let mut mixed = policy.clone();
            mixed.conditions_sha256 = old.search_policy_registration().unwrap().conditions_sha256;
            assert!(mixed.validate().is_err());
            assert!(
                serde_json::from_str::<PalsSearchPolicyIdentity>(
                    &serde_json::to_string(&mixed).unwrap()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn opted_in_driver_early_cancel_preserves_selection_with_no_recheck_claim() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let driver = refinement_driver(DeadlineObserver {
            observed: Arc::clone(&observed),
        });
        let before = driver.work_receipt().unwrap().unwrap();
        let context = checker_context(&driver);
        context.cancellation.cancel();
        let report = driver
            .run(&Position::startpos(), &context, &mut |_| {})
            .unwrap();
        assert!(report.best_move.is_none());
        assert!(observed.lock().unwrap().is_empty());
        let after = driver.work_receipt().unwrap().unwrap();
        assert_eq!(after.pals_search_policy, before.pals_search_policy);
        assert_eq!(
            (
                after.go_invocations,
                after.successful_returns,
                after.canceled_returns
            ),
            (1, 1, 1)
        );
        assert_eq!(after.unobserved_work_invocations, 0);
        let totals = after.pals.unwrap();
        assert_eq!(totals.role_calls, Some(0));
        assert_eq!(totals.completed_cpu_tasks, Some(0));
        assert_eq!(totals.consumed_role_outputs, Some(0));
    }

    #[test]
    fn opted_in_driver_preserves_backend_failure_marker_and_original_deadline() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let driver = refinement_driver(DeadlineObserver {
            observed: Arc::clone(&observed),
        });
        let context = checker_context(&driver);
        let hard = context.control.deadline;
        let soft = context
            .control
            .soft_deadline
            .min(context.control.admission_deadline);
        let failure = driver
            .run(&Position::startpos(), &context, &mut |_| {})
            .unwrap_err();
        assert_eq!(failure.code, "PalsSearch");
        assert!(failure.detail.contains("test role deadline observed"));
        assert_eq!(*observed.lock().unwrap(), [soft]);
        assert_eq!(context.control.deadline, hard);
        let work = driver.work_receipt().unwrap().unwrap();
        assert_eq!(
            work.pals_search_policy.as_ref(),
            driver.search_policy_registration()
        );
        assert_eq!((work.successful_returns, work.failed_returns), (0, 1));
        assert_eq!(work.active_invocations, 0);
        assert_eq!(work.pals.unwrap().accepted_proposer_outputs, Some(0));
    }

    #[test]
    fn selected_external_checker_is_rejected_without_start_or_analysis() {
        let state = Arc::new(CheckerProbeState::default());
        let result = PalsSessionDriver::new_with_checker_and_refinement_policy(
            refinement_config(),
            rz_search::pals::engine::LegalOrderRoleMock,
            CheckerProbe::new(Arc::clone(&state)),
            1,
            8,
            1,
            refinement_name(),
            PostRepairRecheckPolicy::SameRepairedLineOnceV1,
        );
        let Err(failure) = result else {
            panic!("unsupported foreign lane admitted")
        };
        assert_eq!(failure.code, "PalsRefinementUnsupported");
        assert_eq!(state.starts.load(Ordering::Acquire), 0);
        assert_eq!(state.analyses.load(Ordering::Acquire), 0);
    }

    #[test]
    fn registration_drift_fails_before_any_role_work() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let mut driver = refinement_driver(DeadlineObserver {
            observed: Arc::clone(&observed),
        });
        driver.search_policy.as_mut().unwrap().conditions_sha256[0] ^= 1;
        let context = checker_context(&driver);
        let failure = driver
            .run(&Position::startpos(), &context, &mut |_| {})
            .unwrap_err();
        assert_eq!(failure.code, "PalsSearchPolicyIdentity");
        assert!(observed.lock().unwrap().is_empty());
        let receipt = driver.work_receipt().unwrap().unwrap();
        receipt
            .pals_search_policy
            .as_ref()
            .unwrap()
            .validate()
            .unwrap();
        assert_eq!(receipt.failed_returns, 1);
        assert_eq!(receipt.pals.unwrap().role_calls, Some(0));
    }

    fn start_probe(driver: &PalsSessionDriver<rz_search::pals::engine::LegalOrderRoleMock>) {
        driver
            .start_checker(
                Instant::now() + Duration::from_secs(2),
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
    }

    #[cfg(feature = "search-work-receipts")]
    #[test]
    fn historical_ready_resource_snapshot_is_preserved_after_helper_cleanup() {
        let state = Arc::new(CheckerProbeState::default());
        let driver = checker_driver(state);
        assert!(driver.checker_startup_resources().unwrap().is_none());
        assert!(
            driver
                .checker_startup_resource_unavailable()
                .unwrap()
                .is_none()
        );
        start_probe(&driver);
        // In-process probes have no real helper PID, hence actual observation
        // remains None. Inject a pure snapshot only to test historical storage.
        assert!(driver.checker_startup_resources().unwrap().is_none());
        assert_eq!(
            driver.checker_startup_resource_unavailable().unwrap(),
            Some("helper_process_identity_not_observed")
        );
        let fixture = crate::pals_attestation::checker::resource::fixture();
        {
            let mut lifecycle = driver.checker_lifecycle.lock().unwrap();
            lifecycle.startup_resources = Some(fixture.clone());
            lifecycle.startup_resource_unavailable = None;
        }
        let before = crate::pals_attestation::checker::PalsCheckerProcessReceipt::observe(
            &driver,
            &"a".repeat(64),
            &"b".repeat(64),
        )
        .unwrap();
        driver
            .finish_checker(Instant::now() + Duration::from_secs(1))
            .unwrap();
        assert_eq!(driver.checker_startup_resources().unwrap(), Some(fixture));
        let after = crate::pals_attestation::checker::PalsCheckerProcessReceipt::observe(
            &driver,
            &"a".repeat(64),
            &"b".repeat(64),
        )
        .unwrap();
        let before = serde_json::to_value(before).unwrap();
        let after = serde_json::to_value(after).unwrap();
        assert_eq!(
            before["startup_resource_observation"],
            after["startup_resource_observation"]
        );
        assert_eq!(
            after["startup_resource_observation"]["helper"]["proc_start_ticks_after"],
            456
        );
        assert_eq!(after["applied_option_values"], "unknown");
        assert!(after.get("startup_resource_unavailable").is_none());
    }
    fn wait_probe(flag: &AtomicBool) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !flag.load(Ordering::Acquire) && Instant::now() < deadline {
            thread::yield_now();
        }
        assert!(
            flag.load(Ordering::Acquire),
            "bounded fixture entry not observed"
        );
    }

    #[test]
    fn checker_registration_selects_model_resolver_and_full_v2_namespace() {
        let driver = checker_driver(Arc::new(CheckerProbeState::default()));
        let registration = driver.checker_registration();
        assert_eq!(
            registration.resolver_version,
            rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION
        );
        assert!(registration.model_value.is_some());
        assert!(registration.owned_descriptor.is_none());
        assert_eq!(
            driver
                .work_receipt()
                .unwrap()
                .unwrap()
                .pals_resolver
                .unwrap(),
            PalsResolverIdentity::from_semantics(
                registration.resolver_version,
                registration.resolver_semantics
            )
        );
        let config = PalsConfig {
            beam_width: 1,
            line_plies: 2,
            cpu_nodes_per_task: 8,
            ..PalsConfig::default()
        };
        let original = checker_implementation(registration, &config, 1, 8, 1);
        assert_eq!(original, driver.implementation());
        let changed_digest =
            |changed: &PalsCheckerRegistration| checker_implementation(changed, &config, 1, 8, 1);
        let mut changed = registration.clone();
        changed.conditions.push_str(";different-window");
        assert_ne!(original, changed_digest(&changed));
        changed = registration.clone();
        changed.capabilities.selective_search = Some(false);
        assert_ne!(original, changed_digest(&changed));
        changed = registration.clone();
        changed.model_value.as_mut().unwrap().model_epoch[0] ^= 1;
        assert_ne!(original, changed_digest(&changed));
        changed = registration.clone();
        changed.model_value.as_mut().unwrap().precision = "different-precision".into();
        assert_ne!(original, changed_digest(&changed));
        changed = registration.clone();
        let CheckerIdentity::ExternalUci(identity) = &mut changed.identity else {
            panic!()
        };
        identity.options.insert("Hash".into(), "256".into());
        assert_ne!(original, changed_digest(&changed));
        assert_ne!(
            original,
            checker_implementation(registration, &config, 1, 9, 1)
        );
    }

    #[test]
    fn explicit_start_preserves_failure_zero_request_and_actual_success_identity() {
        let state = Arc::new(CheckerProbeState::default());
        state.fail_start.store(true, Ordering::Release);
        let driver = checker_driver(Arc::clone(&state));
        assert!(!driver.checker_started());
        let failure = driver
            .run(
                &Position::startpos(),
                &checker_context(&driver),
                &mut |_| {},
            )
            .unwrap_err();
        assert_eq!(failure.code, "PalsCheckerNotStarted");
        assert_eq!(state.starts.load(Ordering::Acquire), 0);
        let failure = driver
            .start_checker(
                Instant::now() + Duration::from_secs(2),
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap_err();
        assert_eq!(failure.code, "PalsCheckerStartup");
        assert!(failure.detail.contains("preserved-startup-failure"));
        assert!(!driver.checker_started());
        assert_eq!(
            driver
                .checker_last_attempt()
                .unwrap()
                .unwrap()
                .external
                .unwrap()
                .request_id,
            0
        );
        assert!(driver.checker_attempts().unwrap().is_empty());
        assert!(driver.checker_startup_uci().unwrap().is_none());
        assert!(
            driver
                .finish_checker(Instant::now() + Duration::from_secs(2))
                .unwrap()
                .cleanup_complete
        );
        assert_eq!(state.shutdowns.load(Ordering::Acquire), 1);

        let driver = checker_driver(Arc::new(CheckerProbeState::default()));
        start_probe(&driver);
        assert!(driver.checker_started());
        assert_eq!(
            driver.checker_startup_uci().unwrap().unwrap().name,
            "actual fixture handshake name"
        );
        assert_ne!(
            driver.checker_startup_uci().unwrap().unwrap().name.as_str(),
            match &driver.checker_registration().identity {
                CheckerIdentity::ExternalUci(identity) => identity.declared_name.as_str(),
                _ => panic!(),
            }
        );
    }

    #[test]
    fn late_startup_does_not_gain_ready_or_admit_go() {
        let state = Arc::new(CheckerProbeState::default());
        state.late_start.store(true, Ordering::Release);
        let driver = checker_driver(state);
        let cancel = Arc::new(AtomicBool::new(false));
        let failure = driver
            .start_checker(Instant::now() + Duration::from_secs(2), Arc::clone(&cancel))
            .unwrap_err();
        assert_eq!(failure.code, "PalsCheckerStartupLate");
        assert!(cancel.load(Ordering::Acquire));
        assert!(!driver.checker_started());
        assert!(driver.checker_startup_uci().unwrap().is_none());
        assert!(driver.checker_last_attempt().unwrap().is_some());
        assert_eq!(
            driver
                .run(
                    &Position::startpos(),
                    &checker_context(&driver),
                    &mut |_| {}
                )
                .unwrap_err()
                .code,
            "PalsCheckerNotStarted"
        );
        driver
            .finish_checker(Instant::now() + Duration::from_secs(2))
            .unwrap();
    }

    #[test]
    fn foreign_reset_uses_actual_controls_and_failed_reset_remains_unapplied() {
        let state = Arc::new(CheckerProbeState::default());
        let driver = checker_driver(Arc::clone(&state));
        start_probe(&driver);
        state.fail_reset.store(true, Ordering::Release);
        driver.reset_game().unwrap();
        assert_eq!(state.resets.load(Ordering::Acquire), 0);
        let context = checker_context(&driver);
        let failure = driver
            .run(&Position::startpos(), &context, &mut |_| {})
            .unwrap_err();
        assert_eq!(failure.code, "PalsReset");
        assert!(failure.detail.contains("fallible-reset-not-acknowledged"));
        assert_eq!(
            *state.reset_deadlines.lock().unwrap(),
            [context
                .control
                .soft_deadline
                .min(context.control.admission_deadline)]
        );
        assert_eq!(driver.reset_applied.load(Ordering::Acquire), 0);
        assert_eq!(state.analyses.load(Ordering::Acquire), 0);
        state.fail_reset.store(false, Ordering::Release);
        let failure = driver
            .run(&Position::startpos(), &context, &mut |_| {})
            .unwrap_err();
        assert_eq!(failure.code, "PalsSearch");
        assert_eq!(driver.reset_applied.load(Ordering::Acquire), 1);
        assert_eq!(state.resets.load(Ordering::Acquire), 2);
        // Normal idle fixture retains no exit/drain event and is not quarantine.
        assert_eq!(
            failure.physical_completion,
            DriverPhysicalCompletion::Confirmed
        );
    }

    #[test]
    fn newer_reset_is_not_erased_by_older_success_and_generation_overflow_is_explicit() {
        let state = Arc::new(CheckerProbeState::default());
        state.block_reset.store(true, Ordering::Release);
        let driver = Arc::new(checker_driver(Arc::clone(&state)));
        start_probe(&driver);
        driver.reset_game().unwrap();
        let running = Arc::clone(&driver);
        let thread = thread::spawn(move || {
            running.run(
                &Position::startpos(),
                &checker_context(&*running),
                &mut |_| {},
            )
        });
        wait_probe(&state.reset_entered);
        driver.reset_game().unwrap();
        state.block_reset.store(false, Ordering::Release);
        assert_eq!(thread.join().unwrap().unwrap_err().code, "PalsSearch");
        assert_eq!(driver.reset_requested.load(Ordering::Acquire), 2);
        assert_eq!(driver.reset_applied.load(Ordering::Acquire), 1);
        assert_eq!(
            driver
                .run(
                    &Position::startpos(),
                    &checker_context(&*driver),
                    &mut |_| {}
                )
                .unwrap_err()
                .code,
            "PalsSearch"
        );
        assert_eq!(driver.reset_applied.load(Ordering::Acquire), 2);
        assert_eq!(state.resets.load(Ordering::Acquire), 2);
        driver.reset_requested.store(u64::MAX, Ordering::Release);
        assert!(
            driver
                .reset_game()
                .unwrap_err()
                .to_string()
                .contains("generation exhausted")
        );
        assert_eq!(state.resets.load(Ordering::Acquire), 2);
    }

    #[test]
    fn finite_finish_cancels_active_owner_preserves_work_and_closes_future_admission() {
        let state = Arc::new(CheckerProbeState::default());
        state.block_analysis.store(true, Ordering::Release);
        let driver = Arc::new(checker_driver(Arc::clone(&state)));
        start_probe(&driver);
        let context = checker_context(&*driver);
        let cancel = Arc::clone(&context.control.cancellation);
        let running = Arc::clone(&driver);
        let thread =
            thread::spawn(move || running.run(&Position::startpos(), &context, &mut |_| {}));
        wait_probe(&state.analysis_entered);
        let shutdown = driver
            .finish_checker(Instant::now() + Duration::from_secs(2))
            .unwrap();
        assert!(shutdown.cleanup_complete);
        assert!(cancel.load(Ordering::Acquire));
        assert_eq!(thread.join().unwrap().unwrap_err().code, "PalsSearch");
        assert_eq!(state.analyses.load(Ordering::Acquire), 1);
        let attempts = driver.checker_attempts().unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].work.nodes, Some(7));
        assert_eq!(
            driver.checker_last_attempt().unwrap().unwrap().work.nodes,
            Some(7)
        );
        assert_eq!(driver.checker_shutdown().unwrap(), Some(shutdown));
        assert_eq!(
            driver
                .finish_checker(Instant::now() + Duration::from_secs(2))
                .unwrap(),
            shutdown
        );
        assert_eq!(state.shutdowns.load(Ordering::Acquire), 1);
        assert_eq!(
            driver
                .run(
                    &Position::startpos(),
                    &checker_context(&*driver),
                    &mut |_| {}
                )
                .unwrap_err()
                .code,
            "PalsCheckerClosed"
        );
        assert!(driver.reset_game().is_err());
    }

    #[test]
    fn lock_timeout_and_quarantine_never_become_confirmed_shutdown() {
        let state = Arc::new(CheckerProbeState::default());
        let driver = checker_driver(Arc::clone(&state));
        start_probe(&driver);
        let owner = driver.engine.lock().unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        driver.checker_lifecycle.lock().unwrap().active_cancel = Some(Arc::clone(&cancel));
        let failure = driver
            .finish_checker(Instant::now() + Duration::from_millis(5))
            .unwrap_err();
        assert_eq!(failure.code, "PalsCheckerShutdownLockDeadline");
        assert_eq!(
            failure.physical_completion,
            DriverPhysicalCompletion::Unknown
        );
        assert!(cancel.load(Ordering::Acquire));
        assert!(driver.checker_shutdown().unwrap().is_none());
        assert_eq!(state.shutdowns.load(Ordering::Acquire), 0);
        drop(owner);
        driver
            .finish_checker(Instant::now() + Duration::from_secs(2))
            .unwrap();

        let state = Arc::new(CheckerProbeState::default());
        state.quarantine.store(true, Ordering::Release);
        state.fail_shutdown.store(true, Ordering::Release);
        let driver = checker_driver(state);
        start_probe(&driver);
        let failure = driver
            .run(
                &Position::startpos(),
                &checker_context(&driver),
                &mut |_| {},
            )
            .unwrap_err();
        assert_eq!(
            failure.physical_completion,
            DriverPhysicalCompletion::Unknown
        );
        let failure = driver
            .finish_checker(Instant::now() + Duration::from_secs(2))
            .unwrap_err();
        assert_eq!(
            failure.physical_completion,
            DriverPhysicalCompletion::Unknown
        );
        let actual = driver.checker_shutdown().unwrap().unwrap();
        assert!(actual.quarantined && !actual.cleanup_complete);
        assert!(!actual.exit_observed && !actual.stdout_drained && !actual.stderr_drained);
        assert_eq!(
            driver.checker_last_attempt().unwrap().unwrap().work.nodes,
            Some(7)
        );
    }

    #[test]
    fn old_own_driver_retains_raw_resolver_and_no_foreign_shutdown_claims() {
        let driver = PalsSessionDriver::new(
            PalsConfig::default(),
            rz_search::pals::engine::LegalOrderRoleMock,
            CpuConfig {
                tt_entries: 0,
                ..CpuConfig::default()
            },
            1,
            8,
            1,
            EngineIdentity {
                name: "own compatibility fixture".into(),
                author: "RoveZero contributors".into(),
            },
        )
        .unwrap();
        assert!(driver.checker_started());
        assert!(matches!(
            driver.checker_registration().identity,
            CheckerIdentity::Owned(_)
        ));
        assert!(driver.checker_registration().model_value.is_none());
        assert_eq!(
            driver.checker_registration().resolver_version,
            rz_search::pals::engine::PALS_VALUE_RESOLVER_VERSION
        );
        let shutdown = driver
            .finish_checker(Instant::now() + Duration::from_secs(2))
            .unwrap();
        assert_eq!(shutdown, CheckerShutdown::owned_no_process());
        let cpu = CpuSessionDriver::new(
            CpuConfig {
                tt_entries: 0,
                ..CpuConfig::default()
            },
            8,
        )
        .unwrap();
        assert!(cpu.pals_checker_registration().is_none());
        assert!(
            cpu.finish_checker(Instant::now() + Duration::from_secs(1))
                .unwrap()
                .is_none()
        );
    }
}
