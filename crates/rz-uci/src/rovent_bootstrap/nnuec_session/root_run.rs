//! Stack-bounded transport of one original NNUEC root.
//!
//! Every phase borrows the same heap custody. Large successful owners and
//! failed by-value transfers stay in explicit slots; an error unsizes this
//! same allocation into the session's existing retained state. This changes
//! neither the original admission nor any physical completion requirement.

use super::retirement::NnuecRootRetired;
use super::*;
use crate::pals_adaptive::AdaptiveSearchDriver as _;
use crate::pals_adaptive::{
    AdaptiveCheckerMetadata, AdaptiveNnuecRootParts, AdaptiveSequenceContinuation,
    NnuecPreparationRecord, UtilityRoutePendingClosure,
};
use rz_runtime::CompletionSignal;
use rz_search::pals::engine::PalsResult;
use rz_search::pals::engine::adaptive_port::AdaptivePreparingRootCustody;
use rz_search::pals::engine::preparing_root::ResearcherPreparingRoot;
use rz_search::pals::rovent_engine::RootMoveDomain;

type RootParts =
    AdaptiveNnuecRootParts<AdaptivePortQuestion, Backend, AdaptivePortDriver<AdaptivePortModel>>;

/// Original transfer metadata stays alive if a later port/retirement fails.
struct TransferReadback {
    continuation: AdaptiveSequenceContinuation,
    receipt: AdaptiveShutdownReceipt,
    _preparation_observations: Vec<NnuecPreparationRecord>,
    _preparation_observations_omitted: u64,
}

struct RootRunFailure {
    code: &'static str,
    detail: String,
}

/// Empty slots indicate a transfer to another slot or the same physical
/// actor/worker, never a release acknowledgement. Large phase layouts are
/// boxed independently so even constructing this custody has a small frame.
struct RootRunCustody {
    idle: Option<Box<Idle>>,
    native: Option<NativeOwner>,
    owner: Option<Box<SemanticOwner>>,
    actor: Option<PreparationActor>,
    root: Option<Box<ResearcherPreparingRoot<AdaptivePortModel>>>,
    initial: Option<InitialNnuecPreparing>,
    ready: Option<Box<InitialNnuecReady>>,
    port: Option<Box<AdaptiveEnginePort<AdaptivePortModel>>>,
    preparing_custody: Option<Box<AdaptivePreparingRootCustody>>,
    worker: Option<CpuWorker>,
    metadata: Option<AdaptiveCheckerMetadata>,
    pins: Option<NnuecRootRetirementPins>,
    ledger: Option<CpuCreditLedger>,
    root_budget: Option<Digest>,
    cleanup_deadline: Option<Instant>,
    root_moves: Option<Arc<RootMoveDomain>>,
    driver: Option<Box<AdaptivePortDriver<AdaptivePortModel>>>,
    manager: Option<Box<Manager>>,
    parts: Option<Box<RootParts>>,
    transfer: Option<Box<TransferReadback>>,
    retired_worker: Option<AdaptiveNnuecRetiredOwner>,
    utility_pending: Option<Box<UtilityRoutePendingClosure<AdaptivePortQuestion>>>,
    retirement: Option<NnuecRootRetirement>,
    retired: Option<Box<NnuecRootRetired>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    intervention_trial:
        Option<Arc<crate::pals_adaptive::UtilityInterventionTrialPhysicalRetirement>>,
    #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
    cohort_trial: Option<Arc<crate::pals_adaptive::UtilityTrialPhysicalRetirement>>,
    completed_utility: Option<Arc<crate::pals_adaptive::CompletedUtilityRouteAction>>,
    admission: NnuecRootAdmission,
    signal: Option<CompletionSignal>,
    next_request_sequence: u64,
    game: Option<u64>,
    collection_admitted: bool,
    /// Original collection owners remain retained even if bounded delivery
    /// fails or the teacher's physical retirement remains unknown.
    collection_handoff_v2: Option<Arc<Mutex<RoventNnuecCollectionHandoffV2>>>,
    receipt: Option<NnuecRootReceipt>,
    result: Option<Box<PalsResult>>,
    work: Option<DriverWork>,
    retirement_deadline: Option<Instant>,
    /// Original typed error/rejection or panic payload, including any owners
    /// returned by a failed consuming API. It is never a replacement owner.
    failure_payload: Option<Box<dyn Any + Send>>,
    /// Semantic failure is distinct from the physical retirement below. Keep
    /// actual typed observations until that retirement can return known Idle.
    semantic_failures: Option<Box<[AdaptiveManagerError]>>,
    /// First exact cooperative Collection error. Later drain/closure failures
    /// cannot replace its primary diagnostic or create a physical fence.
    collection_failure: Option<crate::pals_adaptive::AdaptiveCollectionError>,
    finished: Option<Box<RootReturn>>,
    retained_failure: Option<RootRunFailure>,
    retained_actor_error: Option<PreparationError>,
    retained_native_error: Option<SearchSessionFailure>,
    retained_storage_error: Option<SearchSessionFailure>,
}

impl RootRunCustody {
    #[inline(never)]
    fn new(idle: Box<Idle>, admission: NnuecRootAdmission) -> Box<Self> {
        Box::new(Self {
            idle: Some(idle),
            native: None,
            owner: None,
            actor: None,
            root: None,
            initial: None,
            ready: None,
            port: None,
            preparing_custody: None,
            worker: None,
            metadata: None,
            pins: None,
            ledger: None,
            root_budget: None,
            cleanup_deadline: None,
            root_moves: None,
            driver: None,
            manager: None,
            parts: None,
            transfer: None,
            retired_worker: None,
            utility_pending: None,
            retirement: None,
            retired: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            intervention_trial: None,
            #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
            cohort_trial: None,
            completed_utility: None,
            admission,
            signal: None,
            next_request_sequence: 0,
            game: None,
            collection_admitted: false,
            collection_handoff_v2: None,
            receipt: None,
            result: None,
            work: None,
            retirement_deadline: None,
            failure_payload: None,
            semantic_failures: None,
            collection_failure: None,
            finished: None,
            retained_failure: None,
            retained_actor_error: None,
            retained_native_error: None,
            retained_storage_error: None,
        })
    }

    fn fail(&self, code: &'static str, detail: impl Into<String>) -> RootRunFailure {
        let detail = detail.into();
        let detail = match &self.collection_failure {
            Some(original) => format!(
                "{}; subsequent {code}: {detail}",
                AdaptiveManagerError::Collection(original.clone())
            ),
            None => detail,
        };
        RootRunFailure { code, detail }
    }

    fn retain_failure<T: Any + Send>(
        &mut self,
        code: &'static str,
        detail: impl Into<String>,
        original: T,
    ) -> RootRunFailure {
        self.failure_payload = Some(Box::new(original));
        self.fail(code, detail)
    }

    fn retain_panic(
        &mut self,
        code: &'static str,
        detail: &'static str,
        payload: Box<dyn Any + Send>,
    ) -> RootRunFailure {
        self.failure_payload = Some(payload);
        self.fail(code, detail)
    }
}

// The primary semantic failure survives both independent native cleanup and
// optional storage cleanup. A missing/unavailable archive cannot skip the first.
fn drain_independent_before_storage<T, S>(
    retained: &mut T,
    original: SearchSessionFailure,
    independent: impl FnOnce(&mut T) -> Option<S>,
    storage: impl FnOnce(&mut T, S),
) -> Result<(), SearchSessionFailure> {
    if let Some(signal) = independent(retained) {
        storage(retained, signal);
    }
    Err(original)
}

#[cfg(test)]
mod retained_drain_tests {
    use super::*;
    #[derive(Default)]
    struct FakeRetained {
        events: Vec<&'static str>,
        actor_owned: bool,
        native_owned: bool,
        storage_error: bool,
    }
    #[test]
    fn archive_absence_or_failure_never_skips_independent_drain_or_replaces_failure() {
        for storage in ["absent", "locked", "present"] {
            let mut retained = FakeRetained {
                actor_owned: true,
                native_owned: true,
                ..Default::default()
            };
            let original = unknown("RootFailed", "original semantic failure");
            let result = drain_independent_before_storage(
                &mut retained,
                original,
                |r| {
                    r.events.push("actor_shutdown");
                    r.events.push("native_drain");
                    Some(())
                },
                |r, ()| {
                    r.events.push(storage);
                    r.storage_error = storage == "locked";
                },
            )
            .unwrap_err();
            assert_eq!(result.code, "RootFailed");
            assert_eq!(retained.events, ["actor_shutdown", "native_drain", storage]);
            assert!(retained.actor_owned && retained.native_owned);
            assert_eq!(retained.storage_error, storage == "locked");
        }
    }
}

impl RoventNnuecSessionDriver {
    /// Only pointer-sized phase custody crosses this orchestration frame.
    #[inline(never)]
    pub(super) fn run_admitted(
        &self,
        idle: Box<Idle>,
        position: &Position,
        context: &SearchSessionContext,
        admission: NnuecRootAdmission,
        progress: &mut dyn FnMut(BoardMove),
    ) -> RootReturn {
        #[cfg(feature = "pals-adaptive-diagnostics")]
        self.nnuec_diagnostics
            .begin(context.authority, admission.digest());
        let mut run = RootRunCustody::new(idle, admission);
        let outcome = (|| {
            if !self.prepare_admitted_root(&mut run, position, context)? {
                return Ok(());
            }
            if !self.adopt_ready_manager(&mut run, context)? {
                return Ok(());
            }
            self.execute_admitted_manager(&mut run, context, progress)?;
            self.retire_executed_root(&mut run, context)
        })();
        match outcome {
            Ok(()) => *run
                .finished
                .take()
                .expect("completed phase has its original return"),
            Err(failed) => {
                #[cfg(feature = "pals-adaptive-diagnostics")]
                {
                    self.nnuec_diagnostics.failure(failed.code);
                    if let Some(manager) = run.manager.as_ref() {
                        self.nnuec_diagnostics.preparations(
                            manager.nnuec_preparation_observations(),
                            manager.nnuec_preparation_observations_omitted(),
                        );
                    }
                }
                self.closed.store(true, Ordering::Release);
                let error = unknown(failed.code, &failed.detail);
                run.retained_failure = Some(failed);
                Err((State::Retained { _owners: run }, error))
            }
        }
    }

    /// Best-effort bounded drain for an already failed root. Even real joins
    /// cannot erase its original semantic failure or turn Retained into Closed.
    /// In particular first-go setup failures still own the original Idle/native
    /// custody here, and the archive worker must not be left unobserved.
    pub(super) fn drain_retained_storage(
        &self,
        retained: &mut (dyn Any + Send),
        until: Instant,
    ) -> Result<(), SearchSessionFailure> {
        let Some(run) = retained.downcast_mut::<RootRunCustody>() else {
            return Err(SearchSessionFailure::physical_completion_unknown());
        };
        let original = run.retained_failure.as_ref().map_or_else(
            SearchSessionFailure::physical_completion_unknown,
            |failed| unknown(failed.code, &failed.detail),
        );
        let until = until.min(run.admission.original_context().shutdown_deadline);
        drain_independent_before_storage(
            run,
            original,
            |run| {
                // Only an intact Idle or an unbound pre-root transfer has a
                // known native/actor pair. Later phases keep their owners.
                let pair = if let Some(idle) = run.idle.as_mut() {
                    Some((&mut idle.actor, &mut idle.native))
                } else if run.root.is_none()
                    && run.initial.is_none()
                    && run.ready.is_none()
                    && run.port.is_none()
                    && run.worker.is_none()
                    && run.manager.is_none()
                    && run.preparing_custody.is_none()
                    && run.retirement.is_none()
                {
                    run.native.as_mut().map(|native| (&mut run.actor, native))
                } else {
                    None
                };
                let signal = if let Some((actor_slot, native)) = pair {
                    let signal = native.service.signal();
                    if let Some(actor) = actor_slot.as_mut() {
                        actor.request_shutdown();
                        loop {
                            match actor.try_join() {
                                Ok(Some(())) => {
                                    actor_slot.take();
                                    break;
                                }
                                Ok(None) if Instant::now() < until => std::thread::yield_now(),
                                Ok(None) => break,
                                Err(error) => {
                                    run.retained_actor_error.get_or_insert(error);
                                    break;
                                }
                            }
                        }
                    }
                    if let Err(error) = native.finish(until) {
                        run.retained_native_error.get_or_insert(error);
                    }
                    Some(signal)
                } else {
                    run.signal.clone()
                };
                signal
            },
            |run, signal| match self.storage.try_lock() {
                Ok(mut slot) => {
                    if let Some(storage) = slot.as_mut() {
                        if let Err(error) = storage.abort_and_join(
                            run.retained_failure.as_ref().map_or_else(
                                SearchSessionFailure::physical_completion_unknown,
                                |failed| unknown(failed.code, &failed.detail),
                            ),
                            until,
                            &signal,
                        ) {
                            run.retained_storage_error.get_or_insert(error);
                        }
                    }
                }
                Err(_) => {
                    run.retained_storage_error.get_or_insert_with(|| {
                        unknown(
                            "RoventNnuecStorageDrain",
                            "original storage lock unavailable during retained cleanup",
                        )
                    });
                }
            },
        )
    }

    #[inline(never)]
    fn prepare_admitted_root(
        &self,
        run: &mut RootRunCustody,
        position: &Position,
        context: &SearchSessionContext,
    ) -> Result<bool, RootRunFailure> {
        let idle = run.idle.as_mut().expect("original idle custody");
        let reset = self.reset_requested.swap(false, Ordering::AcqRel)
            || idle.game != Some(context.authority.game.0);
        let mut storage = match self.storage.try_lock() {
            Ok(storage) => storage,
            Err(_) => {
                return Err(run.fail(
                    "RoventNnuecStorage",
                    "original storage slot is busy/poisoned",
                ));
            }
        };
        if reset && !storage.as_ref().is_some_and(|storage| storage.configured()) {
            if let Err(error) = idle
                .owner
                .as_mut()
                .expect("original semantic owner")
                .try_adaptive_new_game(
                    context.authority,
                    context.control.deadline,
                    &context.control.cancellation,
                )
            {
                return Err(run.retain_failure("RoventNnuecNewGame", error.to_string(), error));
            }
            idle.game = Some(context.authority.game.0);
        }
        if let Some(storage) = storage.as_mut() {
            let Some(admission) = run.admission.v1() else {
                return Err(run.fail(
                    "RoventNnuecStorageV2Unsupported",
                    "V2 storage has no original admission consumer",
                ));
            };
            let admission = Arc::clone(admission);
            let owner = idle.owner.take().expect("original semantic owner");
            let signal = idle.native.service.signal();
            match storage.prepare(
                StorageSemanticOwner::Pcr(owner),
                admission,
                reset,
                position,
                &signal,
            ) {
                Ok(StorageSemanticOwner::Pcr(owner)) => {
                    idle.owner = Some(owner);
                    idle.game = Some(context.authority.game.0);
                }
                Ok(owner) => {
                    storage.retain_wrong_return(owner);
                    return Err(
                        run.fail("RoventNnuecStorageReturn", "returned semantic kind differs")
                    );
                }
                Err(mut rejected) => {
                    if let Some(owner) = rejected.owner.take() {
                        match owner {
                            StorageSemanticOwner::Pcr(owner) => idle.owner = Some(owner),
                            owner => storage.retain_wrong_return(owner),
                        }
                    }
                    return Err(run.retain_failure(
                        "RoventNnuecStorage",
                        rejected.failure.detail.clone(),
                        rejected.failure,
                    ));
                }
            }
        }
        drop(storage);
        let deadline = match self.clock.deadline(context.control.deadline) {
            Ok(deadline) => deadline,
            Err(error) => {
                return Err(run.retain_failure("RoventNnuecClock", error.to_string(), error));
            }
        };
        let Idle {
            native,
            owner,
            actor,
            next_request_sequence,
            game,
            previous_root: _,
        } = *run.idle.take().expect("same original idle");
        let owner = owner.expect("original semantic owner returned from archive");
        run.native = Some(native);
        run.actor = actor;
        run.next_request_sequence = next_request_sequence;
        run.game = game;
        let root = match (*owner).bind_root(
            position.clone(),
            context.authority,
            deadline,
            context.control.deadline,
            Arc::clone(&context.control.cancellation),
        ) {
            Ok(root) => root,
            Err((owner, error)) => {
                run.owner = Some(Box::new(owner));
                return Err(run.retain_failure("RoventNnuecBindRoot", error.to_string(), error));
            }
        };
        run.root = Some(Box::new(root));
        let root = run.root.as_mut().expect("bound original root");
        if let Err(error) = root.attach_original_controls(
            ResearcherRootStartup {
                mode: self.snapshot.mode,
                root_budget: run.admission.root_budget_digest(),
                goal: context.goal,
                nodes: context.requested_nodes,
                selective_policy: self.selective_policy,
                frontier_limits: self.frontier_limits.clone(),
                clock: self.clock.b_clock(),
                limits: PalsLimits {
                    deadline: context.control.deadline,
                    max_rounds: self.max_rounds,
                    max_cpu_nodes: self.max_native_nodes,
                    cpu_depth: self.cpu_depth,
                },
                cleanup_deadline: context.shutdown_deadline,
            },
            Arc::clone(&self.issuer),
            context.root_moves.as_deref(),
        ) {
            return Err(run.retain_failure("RoventNnuecControls", error.to_string(), error));
        }
        if let Err(error) = root.prepare_original_root_rules() {
            let controlled = matches!(
                (
                    &error,
                    root.original_frontier_receipt()
                        .map(|receipt| receipt.snapshot.goal)
                ),
                (
                    ResearcherPreparingError::Frontier(FrontierError::NodeBudget),
                    Some(FrontierGoalStatus::Stopped(FrontierStop::WorkBudget))
                ) | (
                    ResearcherPreparingError::Frontier(FrontierError::Deadline),
                    Some(FrontierGoalStatus::Stopped(FrontierStop::Deadline))
                ) | (
                    ResearcherPreparingError::Frontier(FrontierError::Revoked),
                    Some(FrontierGoalStatus::Stopped(FrontierStop::Canceled))
                )
            );
            if !controlled {
                return Err(run.retain_failure("RoventNnuecRootRules", error.to_string(), error));
            }
        }
        let accepted = match context.accepts() {
            Ok(accepted) => accepted,
            Err(error) => {
                return Err(run.retain_failure(
                    "RoventNnuecInitialControl",
                    error.to_string(),
                    error,
                ));
            }
        };
        if !accepted || context.soft_expired(Instant::now()) {
            root.cancel();
        }
        let terminal = root.deadline_before_frontier()
            || root.original_frontier_receipt().is_some_and(|receipt| {
                !matches!(receipt.snapshot.goal, FrontierGoalStatus::Pending)
            });
        if terminal {
            let root = run.root.take().expect("same stopped root");
            let retired = match (*root).retire_before_materialization() {
                Ok(retired) => retired,
                Err((root, error)) => {
                    run.root = Some(Box::new(root));
                    return Err(run.retain_failure(
                        "RoventNnuecInitialRetire",
                        error.to_string(),
                        error,
                    ));
                }
            };
            run.finished = Some(Box::new(self.finish_preparing_retirement(
                run.native.take().expect("same native owner"),
                retired,
                run.actor.take(),
                run.next_request_sequence,
                run.game,
                context,
                run.admission.clone(),
                None,
            )));
            return Ok(false);
        }
        let signal = run
            .native
            .as_ref()
            .expect("same native owner")
            .service
            .signal();
        run.signal = Some(signal.clone());
        if run.actor.is_none() {
            run.actor = match PreparationActor::spawn_notified(
                Arc::clone(self.loaded.model()),
                signal.clone(),
            ) {
                Ok(actor) => Some(actor),
                Err(error) => {
                    return Err(run.retain_failure(
                        "RoventNnuecActor",
                        format!("{error:?}"),
                        error,
                    ));
                }
            };
        }
        run.initial = Some(InitialNnuecPreparing::new(
            *run.root.take().expect("same unmaterialized root"),
            run.actor.take().expect("actual actor just admitted"),
            CpuWorkerId(0),
            signal.clone(),
        ));
        loop {
            if run.initial.as_ref().expect("same preparation").phase()
                != InitialNnuecPhase::AdoptPort
            {
                let accepted = match context.accepts() {
                    Ok(accepted) => accepted,
                    Err(error) => {
                        return Err(run.retain_failure(
                            "RoventNnuecInitialControl",
                            error.to_string(),
                            error,
                        ));
                    }
                };
                if !accepted || context.soft_expired(Instant::now()) {
                    let cleanup = run
                        .initial
                        .take()
                        .expect("same stopped preparation")
                        .into_cleanup();
                    run.finished = Some(Box::new(self.cleanup_initial(
                        run.native.take().expect("same native owner"),
                        cleanup,
                        context,
                        run.admission.clone(),
                        run.next_request_sequence,
                        run.game,
                        "RoventNnuecInitialStopped",
                    )));
                    return Ok(false);
                }
            }
            let observed = signal.version();
            match run.initial.take().expect("same preparation").step() {
                InitialNnuecStep::Pending(pending) => run.initial = Some(pending),
                InitialNnuecStep::Ready(ready) => {
                    run.ready = Some(ready);
                    return Ok(true);
                }
                InitialNnuecStep::RetainedFailure(failed) => {
                    run.finished = Some(Box::new(self.cleanup_initial(
                        run.native.take().expect("same native owner"),
                        failed.into_cleanup(),
                        context,
                        run.admission.clone(),
                        run.next_request_sequence,
                        run.game,
                        "RoventNnuecInitialPreparation",
                    )));
                    return Ok(false);
                }
            }
            signal.wait_changed(
                observed,
                context
                    .shutdown_deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(1)),
            );
        }
    }

    #[inline(never)]
    fn adopt_ready_manager(
        &self,
        run: &mut RootRunCustody,
        context: &SearchSessionContext,
    ) -> Result<bool, RootRunFailure> {
        let InitialNnuecReady {
            port,
            custody,
            worker,
            actor,
            metadata,
        } = *run.ready.take().expect("actual initial Ready");
        run.port = Some(Box::new(port));
        run.preparing_custody = Some(Box::new(custody));
        run.worker = Some(worker);
        run.actor = Some(actor);
        run.metadata = Some(metadata);
        let metadata = run.metadata.as_ref().expect("actual metadata");
        if let Some(admission) = run.admission.v2() {
            let lock = admission.owner().v2().expect("original V2 owner");
            let rz_experiments::RoventRootTieDeclarationV2::Own {
                expected_search_identity,
                expected_worker_conditions_sha256,
                ..
            } = &lock.manifest.own_r_root_tie
            else {
                return Err(run.fail(
                    "RoventNnuecV2WorkerIdentity",
                    "actual own worker declaration is absent",
                ));
            };
            let Some(actual) = metadata.owned_descriptor() else {
                return Err(run.fail(
                    "RoventNnuecV2WorkerIdentity",
                    "actual own worker metadata is absent",
                ));
            };
            let mut hash = Sha256::new();
            hash.update(b"rz-cpu-search-conditions/1");
            hash.update(actual.search_conditions.as_bytes());
            let conditions = Digest(hash.finalize().into());
            let expected =
                rz_eval::asset::parse_sha256(expected_worker_conditions_sha256).map(Digest);
            if actual.search_identity != expected_search_identity.as_str()
                || expected.ok() != Some(conditions)
            {
                return Err(run.fail(
                    "RoventNnuecV2WorkerIdentity",
                    "actual own worker differs from original V2 lock",
                ));
            }
        }
        let metadata = run.metadata.as_ref().expect("actual metadata");
        let port = run.port.as_ref().expect("same ready port");
        let registration = PalsCheckerRegistration {
            identity: metadata.identity().clone(),
            conditions: metadata.conditions().to_owned(),
            capabilities: metadata.capabilities(),
            owned_descriptor: metadata.owned_descriptor(),
            cpu_resume_policy: metadata.resume_policy(),
            role_model: port.engine().model_identity().to_owned(),
            model_value: port.engine().model_value_identity().cloned(),
            resolver_version: rz_search::pals::value::MODEL_WDL_RESOLVER_VERSION,
            resolver_semantics: rz_search::pals::value::MODEL_WDL_RESOLVER_SEMANTICS,
        };
        match self.latest_checker.lock() {
            Ok(mut latest) => *latest = Some(registration),
            Err(_) => {
                return Err(run.fail(
                    "RoventNnuecMetadata",
                    "actual metadata observation mutex poisoned",
                ));
            }
        }
        run.root_moves = Some(Arc::clone(
            &run.preparing_custody
                .as_ref()
                .expect("original handoff")
                .root_moves,
        ));
        #[cfg(feature = "pals-adaptive-diagnostics")]
        self.nnuec_diagnostics.initial_ready(
            &run.preparing_custody
                .as_ref()
                .expect("original handoff")
                .ready,
        );
        let (ledger, root_budget, cleanup_deadline, pins) = NnuecRootRetirementPins::from_custody(
            *run.preparing_custody.take().expect("same original handoff"),
            run.metadata.take().expect("same actual metadata"),
        );
        run.ledger = Some(ledger);
        run.root_budget = Some(root_budget);
        run.cleanup_deadline = Some(cleanup_deadline);
        run.pins = Some(pins);
        let accepted = match context.accepts() {
            Ok(accepted) => accepted,
            Err(error) => {
                return Err(run.retain_failure(
                    "RoventNnuecAdoptedControl",
                    error.to_string(),
                    error,
                ));
            }
        };
        if !accepted || context.soft_expired(Instant::now()) {
            self.finish_adopted_initial(run, context);
            return Ok(false);
        }
        let driver = AdaptivePortDriver::new_with_coordinator_policy_retaining(
            *run.port.take().expect("same ready port"),
            context.control.deadline,
            Arc::new(run.signal.as_ref().expect("same native signal").clone()),
            ExternalSlicePolicy::default(),
            self.policy,
        );
        match driver {
            Ok(driver) => run.driver = Some(Box::new(driver)),
            Err((port, AdaptiveManagerError::Deadline)) => {
                run.port = Some(Box::new(port));
                self.finish_adopted_initial(run, context);
                return Ok(false);
            }
            Err((port, error)) => {
                run.port = Some(Box::new(port));
                return Err(run.retain_failure("RoventNnuecPort", error.to_string(), error));
            }
        }
        if let Some(observer) = &self.collection_hooks.external_teacher {
            if let Err(error) = run
                .driver
                .as_mut()
                .expect("actual driver")
                .set_external_teacher_observer(Arc::clone(observer))
            {
                return Err(run.retain_failure(
                    "RoventNnuecTeacherObserver",
                    error.to_string(),
                    error,
                ));
            }
        }
        let registration = AdaptiveManagerRegistration {
            authority: context.authority,
            hard_deadline: context.control.deadline,
            token_budget: self.configuration.fields().review.token_budget,
            policy: self.policy,
            first_request_sequence: run.next_request_sequence,
            native_cpu_node_limit: self.max_native_nodes,
            endpoints: nnuec_endpoints(
                self.collection_origin.is_some() || self.collection_origin_v2.is_some(),
            ),
            controller: self.controller,
            device_digest: self.snapshot.backend,
            cpu_device_digest: Digest(Sha256::digest(b"native-own-nnuec-v2-host-cpu").into()),
        };
        let sources = AdaptiveRegisteredLedgerSources {
            sources: AdaptiveManagerSources {
                native: run.native.take().expect("same native owner").service,
                own_workers: vec![run.worker.take().expect("one actual own worker")],
                external: None,
                driver: *run.driver.take().expect("same actual driver"),
            },
            ledger: run.ledger.take().expect("same original ledger"),
            root_budget,
            cleanup_deadline,
        };
        match Manager::new_retaining_with_registered_ledger(
            registration,
            self.clock.runtime_clock(),
            self.execution_limits,
            sources,
        ) {
            Ok(manager) => run.manager = Some(Box::new(manager)),
            Err(rejection) => {
                let detail = rejection.error.to_string();
                return Err(run.retain_failure("RoventNnuecManager", detail, rejection));
            }
        }
        Ok(true)
    }

    /// All physical slots move to the existing original cleanup helper once.
    #[inline(never)]
    fn finish_adopted_initial(&self, run: &mut RootRunCustody, context: &SearchSessionContext) {
        run.finished = Some(Box::new(self.cleanup_adopted_initial(
            run.native.take().expect("same native owner"),
            *run.port.take().expect("same adopted port"),
            run.worker.take().expect("same original worker"),
            run.actor.take().expect("same preparation actor"),
            run.pins.take().expect("same original pins"),
            run.ledger.take().expect("same original ledger"),
            run.root_budget.expect("same root budget"),
            run.cleanup_deadline.expect("same cleanup boundary"),
            run.root_moves.take().expect("same root move domain"),
            run.next_request_sequence,
            run.game,
            context,
            run.admission.clone(),
        )));
    }

    #[inline(never)]
    fn execute_admitted_manager(
        &self,
        run: &mut RootRunCustody,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<(), RootRunFailure> {
        let root_budget = run.root_budget.expect("same original root budget");
        let mut collection_admitted = false;
        let mut collection_failure = None;
        let mut v2_hook_started = false;
        let mut v2_revoked = false;
        let mut v2_admission_failure = None;
        let execution = {
            let manager = run.manager.as_mut().expect("actual admitted manager");
            #[cfg(feature = "pals-adaptive-diagnostics")]
            self.nnuec_diagnostics.install_cpu_slices(manager);
            let actor = run.actor.as_ref().expect("same preparation actor");
            let admission = &run.admission;
            catch_unwind(AssertUnwindSafe(|| -> Result<(), AdaptiveManagerError> {
                if manager.registered_cleanup_deadline() != Some(context.shutdown_deadline) {
                    return Err(AdaptiveManagerError::Invalid(
                        "NNUEC original cleanup boundary differs",
                    ));
                }
                if let Some(clock) = &context.effective_clock {
                    manager.register_effective_clock(clock.clone())?;
                }
                manager.enable_nnuec_preparation(Arc::clone(self.loaded.model()))?;
                let fields = self.configuration.fields();
                if let ExecutionConfiguration::OwnResearcher {
                    reciprocal, peers, ..
                } = fields.execution
                {
                    manager.configure_rovent_reciprocal(
                        reciprocal
                            .actual(self.snapshot.mode, root_budget)
                            .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))?,
                    )?;
                    manager.configure_rovent_peer_followups(
                        peers
                            .actual()
                            .map_err(|error| AdaptiveManagerError::Semantic(error.to_string()))?,
                    )?;
                }
                if context.accepts().unwrap_or(false) && !context.soft_expired(Instant::now()) {
                    if let Some(origin) = &self.collection_origin {
                        let original_admission =
                            admission.v1().ok_or(AdaptiveManagerError::Invalid(
                                "V2 collection has no original admission consumer",
                            ))?;
                        let frontier = manager
                            .driver()
                            .port()
                            .and_then(|port| port.rovent_frontier())
                            .ok_or(AdaptiveManagerError::Invalid(
                                "actual NNUEC collection frontier absent",
                            ))?;
                        let cap = RoventCollectionRootAdmission::from_attached_frontier(
                            origin,
                            frontier,
                            self.snapshot.input_lock,
                            root_budget,
                            self.selective_policy,
                            Arc::clone(original_admission),
                        )?;
                        manager.admit_collection_root(cap)?;
                        if let Some(observer) = &self.collection_hooks.root_admission {
                            observer(manager)?;
                        }
                        collection_admitted = true;
                    }
                    if let Some(origin) = &self.collection_origin_v2 {
                        let original_admission = admission.v2().ok_or(
                            AdaptiveManagerError::Invalid("V2 collection root owner is absent"),
                        )?;
                        let frontier = manager
                            .driver()
                            .port()
                            .and_then(|port| port.rovent_frontier())
                            .ok_or(AdaptiveManagerError::Invalid(
                                "actual V2 NNUEC collection frontier absent",
                            ))?;
                        let cap = RoventCollectionRootAdmissionV2::from_attached_frontier(
                            origin,
                            frontier,
                            Arc::clone(original_admission),
                            root_budget,
                            self.selective_policy,
                        )?;
                        let hooks = self.collection_hooks_v2.as_ref().ok_or(
                            AdaptiveManagerError::Invalid("V2 collection hook owner absent"),
                        )?;
                        // A hook may start a physical teacher before it fails.
                        // Mark the attempt first so failure requests revoke and
                        // retains the original hook along with all root owners.
                        v2_hook_started = true;
                        let expected_cap = cap.clone();
                        let mut port = RoventNnuecCollectionAdmissionPortV2 { manager: &mut *manager };
                        if let Err(rejected) = hooks.admit(&mut port, cap) {
                            let error = if rejected.admission.same_original_admission(&expected_cap)
                            {
                                rejected.error.clone()
                            } else {
                                AdaptiveManagerError::Invalid(
                                    "V2 rejected collection admission has a different root owner",
                                )
                            };
                            v2_admission_failure = Some(rejected);
                            return Err(error);
                        }
                        let installed = manager.collection_root_admission_v2().ok_or(
                            AdaptiveManagerError::Invalid(
                                "V2 collection observer was not installed",
                            ),
                        )?;
                        if !installed.same_original_admission(&expected_cap) {
                            return Err(AdaptiveManagerError::Invalid(
                                "V2 installed collection admission has a different root owner",
                            ));
                        }
                        collection_admitted = true;
                    }
                }
                // The same original gate precedes root input and every C pump.
                if context.accepts().unwrap_or(false)
                    && !context.soft_expired(Instant::now())
                    && !should_quiesce(manager)?
                {
                    let input = manager.driver_mut().root_input()?;
                    manager.prepare_root_ranking(input)?;
                }
                let mut published = None;
                manager.run_root_until_observed(
                    context.shutdown_deadline,
                    &context.control.cancellation,
                    &mut |manager| {
                        let accepts = context.accepts().unwrap_or(false);
                        if manager.phase() == ManagerPhase::Running
                            && (!accepts
                                || context.soft_expired(Instant::now())
                                || should_quiesce(manager)?)
                        {
                            manager.begin_root_quiesce(context.shutdown_deadline)?;
                        }
                        if v2_hook_started
                            && !v2_revoked
                            && (!accepts
                                || context.soft_expired(Instant::now())
                                || manager.phase() != ManagerPhase::Running)
                        {
                            if let Some(hooks) = &self.collection_hooks_v2 {
                                hooks.revoke();
                            }
                            v2_revoked = true;
                        }
                        if collection_admitted {
                            if let Some(pump) = self
                                .collection_hooks
                                .cooperative
                                .as_ref()
                                .filter(|_| collection_failure.is_none())
                            {
                                match pump() {
                                    Ok(()) => {}
                                    Err(AdaptiveManagerError::Collection(error)) => {
                                        // Disable only the failed caller pump.
                                        // The same actor/native cleanup below
                                        // must keep progressing while closing.
                                        collection_failure = Some(error.clone());
                                        manager.record_collection_caller_failure(
                                            error,
                                            context.shutdown_deadline,
                                        )?;
                                    }
                                    Err(error) => return Err(error),
                                }
                            }
                            if let Some(hooks) = self.collection_hooks_v2.as_ref()
                                .filter(|_| collection_failure.is_none()) {
                                match hooks.pump() {
                                    Ok(()) => {}
                                    Err(AdaptiveManagerError::Collection(error)) => {
                                        if !v2_revoked {
                                            hooks.revoke();
                                            v2_revoked = true;
                                        }
                                        if collection_failure.is_none() {
                                            collection_failure = Some(error.clone());
                                            manager.record_collection_caller_failure(
                                                error,
                                                context.shutdown_deadline,
                                            )?;
                                        }
                                    }
                                    Err(error) => return Err(error),
                                }
                            }
                        }
                        manager.pump_nnuec_preparation(actor)?;
                        if context.accepts().unwrap_or(false) {
                            if let Ok(result) = manager.driver().result(PalsCompletion::RoundLimit)
                            {
                                if let Some(movement) = result.best_move {
                                    if !admission.allowed_root_moves().contains(&movement) {
                                        return Err(AdaptiveManagerError::Invalid(
                                            "NNUEC progress escaped original root moves",
                                        ));
                                    }
                                    if published != Some(movement) {
                                        progress(movement);
                                        published = Some(movement);
                                    }
                                }
                            }
                        }
                        Ok(())
                    },
                )?;
                Ok(())
            }))
        };
        run.collection_admitted = collection_admitted;
        run.collection_failure = collection_failure;
        match execution {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                if v2_hook_started {
                    if let Some(hooks) = &self.collection_hooks_v2 {
                        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| hooks.revoke())) {
                            self.retain_v2_revoke_panic(payload);
                        }
                    }
                }
                if let Some(rejected) = v2_admission_failure {
                    return Err(run.retain_failure(
                        "RoventNnuecExecution",
                        error.to_string(),
                        rejected,
                    ));
                }
                return Err(run.retain_failure("RoventNnuecExecution", error.to_string(), error));
            }
            Err(payload) => {
                if v2_hook_started {
                    if let Some(hooks) = &self.collection_hooks_v2 {
                        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| hooks.revoke())) {
                            self.retain_v2_revoke_panic(payload);
                        }
                    }
                }
                return Err(run.retain_panic(
                    "RoventNnuecExecutionPanic",
                    "actual root execution panicked; original panic and physical owners retained",
                    payload,
                ));
            }
        }
        if v2_hook_started && !v2_revoked {
            if let Some(hooks) = &self.collection_hooks_v2 {
                if let Err(payload) = catch_unwind(AssertUnwindSafe(|| hooks.revoke())) {
                    return Err(run.retain_panic(
                        "RoventNnuecV2CollectionRevokePanic",
                        "V2 collection revoke panicked; original hook and owners retained",
                        payload,
                    ));
                }
            }
        }
        let physical = {
            let manager = run.manager.as_ref().expect("same manager after execution");
            match manager.shutdown_receipt() {
                Ok(receipt)
                    if receipt.permits_root_reuse()
                        && manager.phase() == ManagerPhase::Finished =>
                {
                    receipt
                }
                _ => return Err(run.fail("RoventNnuecFence", "actual root fence absent")),
            }
        };
        let manager = run.manager.as_ref().expect("same fenced manager");
        let mut semantic_failures: Vec<_> = manager.failures().cloned().collect();
        for original in [run.collection_failure.as_ref(), manager.first_v2_collection_failure()]
            .into_iter().flatten().rev() {
            // The manager's existing bounded history may have rolled over
            // while draining. Keep this exact first caller cause primary
            // without changing that history's capacity or error semantics.
            semantic_failures.retain(|error| {
                !matches!(error, AdaptiveManagerError::Collection(error) if error == original)
            });
            semantic_failures.insert(0, AdaptiveManagerError::Collection(original.clone()));
        }
        let semantic_failures = semantic_failures.into_boxed_slice();
        if !semantic_failures.is_empty() {
            run.semantic_failures = Some(semantic_failures);
        }
        let publication_capacity = match manager.driver().rovent_publication_capacity_closed() {
            Ok(closed) => closed,
            Err(error) => {
                return Err(run.retain_failure(
                    "RoventNnuecCapacityObservation",
                    error.to_string(),
                    error,
                ));
            }
        };
        let completion = if context.cancellation.is_canceled()
            || context.control.cancellation.load(Ordering::Acquire)
        {
            PalsCompletion::Canceled
        } else if Instant::now() >= context.control.deadline {
            PalsCompletion::Deadline
        } else if manager.driver().root_is_terminal().unwrap_or(false) {
            PalsCompletion::Terminal
        } else if publication_capacity {
            PalsCompletion::Capacity
        } else if manager.cpu_node_limit_reached() {
            PalsCompletion::CpuNodeLimit
        } else if manager.role_call_limit_reached() {
            PalsCompletion::RoleCallLimit
        } else {
            PalsCompletion::RoundLimit
        };
        run.result = match manager.driver().result(completion) {
            Ok(result) => Some(Box::new(result)),
            Err(error) => {
                return Err(run.retain_failure("RoventNnuecResult", error.to_string(), error));
            }
        };
        if run.collection_admitted {
            if let Some(observer) = &self.collection_hooks.root_closure {
                let completion = run
                    .result
                    .as_ref()
                    .expect("same original result")
                    .completion;
                let closure = catch_unwind(AssertUnwindSafe(|| {
                    observer(
                        run.manager.as_mut().expect("same original manager"),
                        completion,
                    )
                }));
                match closure {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => return Err(run.retain_failure("RoventNnuecCollectionClosure", error.to_string(), error)),
                    Err(payload) => return Err(run.retain_panic("RoventNnuecCollectionClosurePanic",
                        "root closure panicked; original pending utility and physical owners retained", payload)),
                }
            }
            if self.collection_hooks_v2.is_some() {
                // Obtain the bounded sender before removing the original
                // observer from the fenced manager. A panic leaves it there.
                let sender = match catch_unwind(AssertUnwindSafe(|| {
                    self.collection_hooks_v2.as_ref().expect("started hook").handoff_sender()
                })) {
                    Ok(sender) => sender,
                    Err(payload) => return Err(run.retain_panic(
                        "RoventNnuecV2HandoffSenderPanic",
                        "original observer and manager retained",
                        payload,
                    )),
                };
                let (admission, observer, settlement) = match run.manager.as_mut()
                    .expect("same fenced manager").take_collection_observer_v2()
                {
                    Ok(Some(original)) => original,
                    _ => return Err(run.fail(
                        "RoventNnuecV2HandoffUnavailable",
                        "original observer and manager retained",
                    )),
                };
                // Preserve collection diagnostics with the original handoff.
                // They make the corpus incomplete, but are not a native NN
                // or search failure after the actual native fence succeeded.
                let mut collection_failures = Vec::new();
                let mut other_failures = Vec::new();
                if let Some(failures) = run.semantic_failures.take() {
                    for failure in Vec::from(failures) {
                        match failure {
                            AdaptiveManagerError::Collection(error) => collection_failures.push(error),
                            other => other_failures.push(other),
                        }
                    }
                }
                if !other_failures.is_empty() {
                    run.semantic_failures = Some(other_failures.into_boxed_slice());
                }
                let handoff = Arc::new(Mutex::new(RoventNnuecCollectionHandoffV2 {
                    admission, observer, settlement, native_fence: physical.clone(),
                    collection_failures: collection_failures.into_boxed_slice(),
                    teacher_physical_ack: None,
                    teacher_aggregate_ack: None,
                }));
                run.collection_handoff_v2 = Some(Arc::clone(&handoff));
                // Keep an independent original reference: successful send
                // alone does not prove the receiver actually kept the data.
                loop {
                    match self.v2_handoff_custody.try_lock() {
                        Ok(mut slot) if slot.is_none() => { *slot = Some(Arc::clone(&handoff)); break; }
                        Ok(_) => return Err(run.fail("RoventNnuecV2HandoffCapacity", "prior root handoff is still retained")),
                        Err(TryLockError::WouldBlock) if Instant::now() < context.shutdown_deadline => std::thread::sleep(Duration::from_millis(1)),
                        Err(TryLockError::WouldBlock) => return Err(run.fail("RoventNnuecV2HandoffCustodyBusy", "session handoff custody remained busy until the shutdown deadline")),
                        Err(TryLockError::Poisoned(_)) => return Err(run.fail("RoventNnuecV2HandoffCustodyPoisoned", "session handoff custody is poisoned")),
                    }
                }
                if sender.try_send(Arc::clone(&handoff)).is_err() {
                    return Err(run.fail(
                        "RoventNnuecV2HandoffUndelivered",
                        "original handoff and manager retained",
                    ));
                }
                // The bounded channel now owns the very same handoff.  The
                // Arena worker retains it together with the original teacher
                // and collection data; UCI records only the native fence.
                // Its join and corpus publication are separate outer results.
            }
        }
        let frontier = match run
            .manager
            .as_mut()
            .expect("same manager")
            .driver_mut()
            .retire_rovent_frontier()
        {
            Ok(Some(receipt)) => receipt.snapshot,
            original => {
                return Err(run.retain_failure(
                    "RoventNnuecFrontierRetire",
                    "actual frontier retirement absent",
                    original,
                ));
            }
        };
        // Store the same Arc before the fallible latest/observer boundary.
        // A failed observer retains both this original receipt and its owners.
        run.receipt = Some(self.make_executed_receipt(
            context,
            &run.admission,
            &frontier,
            &physical,
            run.manager.as_ref().expect("same manager").counters(),
        ));
        let receipt = run.receipt.as_ref().expect("same original root receipt");
        if let Err(error) = self.observe_any_root_receipt(receipt) {
            return Err(run.retain_failure(
                "RoventNnuecRootObservation",
                error.detail.clone(),
                error,
            ));
        }
        run.work = match receipt.work() {
            Ok(work) => Some(work),
            Err(error) => {
                return Err(run.retain_failure("RoventNnuecWork", error.detail.clone(), error));
            }
        };
        if run
            .result
            .as_ref()
            .expect("same result")
            .best_move
            .is_some_and(|movement| !run.admission.allowed_root_moves().contains(&movement))
        {
            return Err(run.fail(
                "RoventNnuecRootRestriction",
                "actual result escaped original move domain",
            ));
        }
        run.utility_pending = match run
            .manager
            .as_mut()
            .expect("same manager")
            .take_utility_route_retirement()
        {
            Ok(pending) => pending.map(Box::new),
            Err(error) => {
                return Err(run.retain_failure(
                    "RoventNnuecUtilityRetirement",
                    error.to_string(),
                    error,
                ));
            }
        };
        let retirement_deadline = context.shutdown_deadline;
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        let retirement_deadline = run
            .utility_pending
            .as_ref()
            .and_then(|pending| pending.cohort_cleanup_deadline())
            .map_or(retirement_deadline, |cutoff| {
                retirement_deadline.min(cutoff)
            });
        run.retirement_deadline = Some(retirement_deadline);
        Ok(())
    }

    #[inline(never)]
    fn retire_executed_root(
        &self,
        run: &mut RootRunCustody,
        context: &SearchSessionContext,
    ) -> Result<(), RootRunFailure> {
        let manager = run.manager.take().expect("same fenced manager");
        #[cfg(feature = "pals-adaptive-diagnostics")]
        self.nnuec_diagnostics.preparations(
            manager.nnuec_preparation_observations(),
            manager.nnuec_preparation_observations_omitted(),
        );
        run.parts = match (*manager).into_nnuec_root_parts() {
            Ok(parts) => Some(Box::new(parts)),
            Err(rejection) => {
                let detail = rejection.error.to_string();
                return Err(run.retain_failure("RoventNnuecRootTransfer", detail, rejection));
            }
        };
        let parts = run.parts.as_ref().expect("same original root parts");
        #[cfg(feature = "pals-adaptive-diagnostics")]
        {
            if let Some(owner) = parts.cpu.first() {
                self.nnuec_diagnostics.joined(owner);
            }
        }
        if parts.cpu.len() != 1 || parts.external.is_some() {
            return Err(run.fail(
                "RoventNnuecCpuTransfer",
                "actual researcher owner set differs",
            ));
        }
        let RootParts {
            native,
            mut cpu,
            external: _,
            driver,
            continuation,
            receipt,
            preparation_observations,
            preparation_observations_omitted,
        } = *run.parts.take().expect("validated original owner set");
        run.native = Some(NativeOwner::new(native));
        run.retired_worker = cpu.pop();
        run.driver = Some(Box::new(driver));
        run.next_request_sequence = continuation.first_request_sequence();
        run.transfer = Some(Box::new(TransferReadback {
            continuation,
            receipt,
            _preparation_observations: preparation_observations,
            _preparation_observations_omitted: preparation_observations_omitted,
        }));
        let driver = run.driver.take().expect("same fenced driver");
        let port =
            match (*driver).into_port(&run.transfer.as_ref().expect("same root fence").receipt) {
                Ok(port) => port,
                Err((driver, error)) => {
                    run.driver = Some(Box::new(driver));
                    return Err(run.retain_failure(
                        "RoventNnuecSemanticFence",
                        error.to_string(),
                        error,
                    ));
                }
            };
        run.port = Some(Box::new(port));
        run.retirement = Some(NnuecRootRetirement::new(
            run.actor.take().expect("same preparation actor"),
            run.retired_worker
                .take()
                .expect("one actual joined researcher"),
            run.pins.take().expect("same original pins"),
            run.port.take().expect("same fenced port").into_engine(),
        ));
        if let Some(pending) = run.utility_pending.take() {
            if let Err((error, pending)) = run
                .retirement
                .as_mut()
                .expect("same retirement")
                .attach_utility_route_pending(*pending)
            {
                run.utility_pending = Some(Box::new(pending));
                return Err(run.retain_failure(
                    "RoventNnuecUtilityRetirement",
                    "original utility pending owner was rejected before same-actor retirement",
                    error,
                ));
            }
        }
        let retirement_deadline = run
            .retirement_deadline
            .expect("original retirement boundary");
        let signal = run.signal.as_ref().expect("same native signal").clone();
        loop {
            if Instant::now() >= retirement_deadline {
                return Err(run.fail(
                    "RoventNnuecRetirementDeadline",
                    "original cleanup deadline expired before same-actor release",
                ));
            }
            let observed = signal.version();
            let retirement = run.retirement.take().expect("same ongoing retirement");
            match retirement.step() {
                NnuecRootRetirementStep::Pending(pending) => run.retirement = Some(pending),
                NnuecRootRetirementStep::Ready(retired) => {
                    #[cfg(feature = "pals-adaptive-diagnostics")]
                    self.nnuec_diagnostics.retired(&retired);
                    run.retired = Some(retired);
                    break;
                }
                NnuecRootRetirementStep::RetainedFailure(failed) => {
                    let detail = format!(
                        "same-actor release failed in {:?}: {}; original owners retained",
                        failed.phase(),
                        failed.error()
                    );
                    return Err(run.retain_failure("RoventNnuecRetirement", detail, failed));
                }
            }
            signal.wait_changed(
                observed,
                retirement_deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(1)),
            );
        }
        #[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
        {
            let retired = run.retired.as_ref().expect("same acknowledged retirement");
            if usize::from(retired.utility_route.is_some())
                + usize::from(retired.utility_cohort.is_some())
                + usize::from(retired.utility_intervention.is_some())
                > 1
            {
                return Err(run.fail(
                    "RoventNnuecUtilityRetirementKind",
                    "one original retirement returned both root-local and cohort outputs",
                ));
            }
            run.intervention_trial = run
                .retired
                .as_mut()
                .expect("same retired owner")
                .utility_intervention
                .take();
            if let Some(trial) = &run.intervention_trial {
                if !run.collection_admitted
                    || trial.authority() != context.authority
                    || run
                        .receipt
                        .as_ref()
                        .expect("original receipt")
                        .v1()
                        .is_none_or(|receipt| {
                            !trial.matches_root_admission(&receipt.root_admission)
                        })
                {
                    return Err(run.fail(
                        "RoventNnuecInterventionTrialScope",
                        "/2 retirement differs from original Collection root",
                    ));
                }
                let Some(observer) = &self.collection_hooks.utility_intervention_trial_retired
                else {
                    return Err(run.fail(
                        "RoventNnuecInterventionTrialObserverMissing",
                        "actual /2 retirement has no original final consumer",
                    ));
                };
                match catch_unwind(AssertUnwindSafe(|| observer(trial))) {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        return Err(run.retain_failure(
                            "RoventNnuecInterventionTrialPublication",
                            error.to_string(),
                            error,
                        ));
                    }
                    Err(payload) => {
                        return Err(run.retain_panic(
                            "RoventNnuecInterventionTrialPublicationPanic",
                            "/2 consumer panicked; original cap and actual owners retained",
                            payload,
                        ));
                    }
                }
                run.intervention_trial = None;
            }
            run.cohort_trial = run
                .retired
                .as_mut()
                .expect("same retired owner")
                .utility_cohort
                .take();
            if let Some(trial) = &run.cohort_trial {
                if !run.collection_admitted
                    || trial.authority() != context.authority
                    || run
                        .receipt
                        .as_ref()
                        .expect("original receipt")
                        .v1()
                        .is_none_or(|receipt| {
                            !trial.matches_root_admission(&receipt.root_admission)
                        })
                {
                    return Err(run.fail(
                        "RoventNnuecUtilityTrialScope",
                        "cohort retirement differs from the original collection root admission",
                    ));
                }
                let Some(observer) = &self.collection_hooks.utility_trial_retired else {
                    return Err(run.fail(
                        "RoventNnuecUtilityTrialObserverMissing",
                        "actual cohort retirement has no final trial consumer",
                    ));
                };
                match catch_unwind(AssertUnwindSafe(|| observer(trial))) {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => return Err(run.retain_failure("RoventNnuecUtilityTrialPublication", error.to_string(), error)),
                    Err(payload) => return Err(run.retain_panic("RoventNnuecUtilityTrialPublicationPanic",
                        "cohort trial consumer panicked; original physical cap and actual owners retained", payload)),
                }
                run.cohort_trial = None;
            }
        }
        run.completed_utility = run
            .retired
            .as_mut()
            .expect("same retired owner")
            .utility_route
            .take()
            .map(Arc::new);
        if let Some(completed) = &run.completed_utility {
            if !run.collection_admitted
                || completed.id().authority() != context.authority
                || completed.semantic().root_budget() != run.admission.root_budget_digest()
            {
                return Err(run.fail(
                    "RoventNnuecUtilityScope",
                    "completed utility action differs from the original admitted collection root",
                ));
            }
            let Some(observer) = &self.collection_hooks.utility_route_completed else {
                return Err(run.fail(
                    "RoventNnuecUtilityObserverMissing",
                    "actual utility completion has no final collection consumer",
                ));
            };
            match catch_unwind(AssertUnwindSafe(|| observer(completed))) {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    return Err(run.retain_failure(
                        "RoventNnuecUtilityPublication",
                        error.to_string(),
                        error,
                    ));
                }
                Err(payload) => return Err(run.retain_panic(
                    "RoventNnuecUtilityPublicationPanic",
                    "final utility consumer panicked; original action and actual owners retained",
                    payload,
                )),
            }
            run.completed_utility = None;
        }
        let receipt = run.receipt.take().expect("same original observed receipt");
        let result = run.result.take().expect("same original result");
        let mate_strategy = receipt
            .frontier()
            .and_then(|frontier| frontier.mate_strategy.clone())
            .filter(|strategy| result.best_move == Some(strategy.best_move()));
        let retired = run.retired.take().expect("same fully acknowledged owner");
        // Continuation and physical metadata are retained until all callbacks
        // succeeded. The next sequence remains this original continuation's.
        let next_request_sequence = run
            .transfer
            .as_ref()
            .expect("same transfer readback")
            .continuation
            .first_request_sequence();
        let idle = Box::new(Idle {
            native: run.native.take().expect("same original native service"),
            owner: Some(Box::new(retired.owner)),
            actor: Some(retired.actor),
            next_request_sequence,
            game: run.game,
            previous_root: Some(receipt.clone()),
        });
        if let Some(original) = run.semantic_failures.take() {
            let mut failure = SearchSessionFailure::debug("RoventNnuecManagerFailure", &original);
            failure.detail.push_str(&format!(
                "; actual_frontier_goal={:?}",
                receipt.frontier().map(|frontier| frontier.goal),
            ));
            // Every original physical retirement above succeeded. Preserve
            // the same known Idle sources for session finish, rather than
            // relabeling a semantic rejection as unknown physical work.
            self.closed.store(true, Ordering::Release);
            run.finished = Some(Box::new(Err((State::Idle(idle), failure))));
            return Ok(());
        }
        run.finished = Some(Box::new(Ok((
            idle,
            SearchSessionReport {
                best_move: result.best_move,
                work: run.work.take().expect("same original work receipt"),
                mate_strategy,
            },
            receipt,
        ))));
        Ok(())
    }
}

fn nnuec_endpoints(has_collection_origin: bool) -> CpuEndpointSelection {
    let mut endpoints = CpuEndpointSelection::deployment();
    if has_collection_origin {
        // The product's original Collection origin grants this use. Its
        // external teacher is an independent Arena owner, not product CPU_T.
        endpoints.execution_use = rz_search::external_cpu::CpuExecutionUse::Collection;
    }
    endpoints
}

#[cfg(test)]
mod endpoint_tests {
    use super::*;
    use rz_search::external_cpu::{CpuEndpoint, CpuExecutionUse};

    #[test]
    fn collection_origin_changes_use_without_loading_v_or_product_teacher() {
        let deployment = nnuec_endpoints(false);
        assert_eq!(deployment, CpuEndpointSelection::deployment());
        let collection = nnuec_endpoints(true);
        assert_eq!(collection.execution_use, CpuExecutionUse::Collection);
        assert_eq!(collection.cpu_t, CpuEndpoint::Disabled);
        assert_eq!(collection.cpu_r, CpuEndpoint::Own);
        assert!(!collection.load_v);
    }
}
