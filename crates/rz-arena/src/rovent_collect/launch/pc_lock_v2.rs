//! P-C input-lock /2 launch custody. A returned run is an observation, not a
//! published corpus or a claim of native/teacher physical completion.
use super::*;
use crate::rovent_collect::pc_hooks_lock_v2::{
    RoventPcCollectionBridgeV2, RoventPcCollectionWorkerV2, PcWorkerExitKindV2, PcWorkerStartErrorV2,
};
use crate::rovent_collect::teacher::v2_session::RoventTeacherRosterEntryV2;
use rz_experiments::RoventInputLockV2;
use rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2;
use rz_uci::rovent_bootstrap::launch::{
    RoventPcCollectionLaunchFailureV2, run_pc_collection_v2_with_resource_admission,
};
use std::sync::atomic::{AtomicU8, Ordering};

const WORKER_NOT_STARTED: u8 = 0;
const WORKER_STARTING: u8 = 1;
const WORKER_STARTED: u8 = 2;
const WORKER_START_FAILED: u8 = 3;

/// Keeps the original resource, native failure and teacher worker together.
/// Neither the live readback nor the thread exit establishes publication.
pub struct RoventPcCollectionRunV2 {
    resource: Arc<Mutex<resources::CollectionResourceCustody>>,
    original_lock: Arc<Mutex<Option<Arc<RoventInputLockV2>>>>,
    owner: RoventCollectionOwnerLockV2,
    bridge: Arc<RoventPcCollectionBridgeV2>,
    worker: Arc<Mutex<RoventPcCollectionWorkerV2>>,
    worker_start: Arc<AtomicU8>,
    worker_exit: Option<PcWorkerExitKindV2>,
    worker_thread_panicked: bool,
    native_failure: Option<RoventPcCollectionLaunchFailureV2>,
    native_failure_taken: bool,
    resource_error: Option<String>,
}

impl RoventPcCollectionRunV2 {
    /// Collect only a thread that has already exited; never wait indefinitely
    /// or infer that a still-running worker has physically stopped.
    pub fn poll_worker(&mut self) -> bool {
        let start = self.worker_start.load(Ordering::Acquire);
        if matches!(start, WORKER_NOT_STARTED | WORKER_START_FAILED) { return true; }
        match self.bridge.poll_worker() {
            Ok(Some(kind)) => {
                self.worker_exit = Some(kind);
                if kind == PcWorkerExitKindV2::OuterPanic { self.worker_thread_panicked = true; }
            }
            Ok(None) => {}
            Err(error) => { self.resource_error.get_or_insert_with(|| format!("teacher worker custody: {error:?}")); }
        }
        self.worker_exit.is_some() || self.worker_thread_panicked
    }

    pub fn observation(&self) -> serde_json::Value {
        let resource = self.resource.lock().ok().map(|state| state.observation());
        let (owner_observation, owner) = match self.owner.try_status() {
            Ok(Some(status)) => ("available", Some(status)),
            Ok(None) => ("busy", None),
            Err(_) => ("poisoned", None),
        };
        let start = self.worker_start.load(Ordering::Acquire);
        let worker_exit = self.worker_exit.map(PcWorkerExitKindV2::label).unwrap_or_else(|| match start {
            WORKER_NOT_STARTED => "not_started", WORKER_STARTING => "start_unknown",
            WORKER_START_FAILED => "start_failed", _ => "pending",
        });
        let worker_observation = self.worker.try_lock().ok().map(|worker| {
            let failures = worker.failure_observation();
            json!({"primary_failure":failures.primary.map(|failure|failure.kind()),
                "additional_failures":failures.additional.iter().map(|failure|failure.kind()).collect::<Vec<_>>(),
                "omitted_failures":failures.omitted,
                "diagnostic_actors":worker.diagnostics().len()})
        });
        json!({"schema":"rz-rovent-pc-lock-v2-run-observation/1",
            "native_error":self.native_failure.as_ref().map(|error|error.error.to_string()),
            "native_failure_transferred":self.native_failure_taken,
            "resource_error":self.resource_error,"resource":resource,
            "teacher_worker_start":match start { WORKER_NOT_STARTED => "not_started", WORKER_STARTING => "unknown", WORKER_STARTED => "started", WORKER_START_FAILED => "failed_before_start", _ => "unknown" },
            "teacher_worker_returned":self.worker_exit.is_some(),
            "teacher_worker_exit":worker_exit,
            "teacher_worker_running_or_unknown": match start {
                WORKER_NOT_STARTED | WORKER_START_FAILED => false,
                _ => self.worker_thread_panicked || self.worker_exit != Some(PcWorkerExitKindV2::Retirement),
            },
            "teacher_worker_thread_panicked":self.worker_thread_panicked,
            "teacher_worker_observation":worker_observation,
            "collection_owner_observation":owner_observation,
            "collection_owner":owner.map(|status| json!({"admitted":status.admitted,"settled":status.settled,
                "pending":status.pending_count,"retired":status.pending_retired_count,
                "pending_disabled_markers":status.pending_disabled_markers,
                "collection_incomplete_due_to_backpressure":status.collection_disabled_first_ticket.is_some() || !status.prior_disabled_first_tickets.is_empty(),
                "first_omitted_ticket":status.collection_disabled_first_ticket.as_ref().map(ticket_json),
                "disable_reason":status.collection_disabled_reason.map(|reason|format!("{reason:?}")),
                "prior_disabled_first_tickets":status.prior_disabled_first_tickets.iter().map(ticket_json).collect::<Vec<_>>(),
                "prior_disabled_reasons":status.prior_disabled_reasons.iter().map(|reason|format!("{reason:?}")).collect::<Vec<_>>(),
                "incomplete_capture":status.incomplete_capture_count,
                "failed_captures":status.retained_failed_captures,
                "failed_staged_inputs":status.retained_failed_staged,
                "failure":status.failure.map(|error|format!("{error:?}")),
                "callback_lock_failure":status.callback_lock_failure})),
            "physical_completion_claimed":false,"corpus_published":false,"training_executed":false})
    }

    /// Move the original native failure to a caller that will retain/recover
    /// it. Taking it never changes the original run into a success.
    pub fn take_native_failure(&mut self) -> Option<RoventPcCollectionLaunchFailureV2> {
        let failure = self.native_failure.take();
        if failure.is_some() {
            self.native_failure_taken = true;
        }
        failure
    }

    /// Transfer the original outer worker panic only to a caller retaining
    /// its potentially resource-owning payload. This never clears the unknown
    /// physical state or turns the run into a successful completion.
    pub fn take_worker_panic_payload(&mut self) -> Option<Box<dyn std::any::Any + Send>> {
        self.bridge.take_outer_panic()
    }

    /// Transfer actual teacher diagnostic bytes only after the worker thread
    /// has exited. A busy or poisoned owner stays retained as unknown.
    pub fn take_teacher_diagnostics(
        &mut self,
    ) -> Result<
        BTreeMap<String, rz_search::external_cpu::collection_teacher::TeacherDiagnostics>,
        ArenaError,
    > {
        check(
            self.worker_exit.is_some() && !self.worker_thread_panicked,
            "Collection /2 teacher worker has not exited cleanly",
        )?;
        self.worker
            .try_lock()
            .map_err(|_| invalid("Collection /2 worker busy or poisoned"))
            .map(|mut worker| worker.take_diagnostics())
    }

    /// Read the actor retirement observations independently of diagnostic
    /// bytes. These values do not themselves authorize corpus publication.
    pub fn teacher_retirement_observations(
        &self,
    ) -> Result<
        BTreeMap<String, rz_uci::rovent_external_launch::RoventTeacherRetirementObservationV2>,
        ArenaError,
    > {
        self.worker
            .try_lock()
            .map_err(|_| invalid("Collection /2 worker busy or poisoned"))
            .map(|worker| worker.teacher_observations().clone())
    }

    /// Give the caller the same collection owner only after its original
    /// worker observed the same fenced handoff and teacher retirement. This
    /// authorizes inspection/partial assembly, not final publication.
    pub fn closed_owner(&self) -> Result<RoventCollectionOwnerLockV2, ArenaError> {
        check(
            self.worker_exit == Some(PcWorkerExitKindV2::Retirement) && self
                .worker
                .try_lock()
                .is_ok_and(|worker| !worker.has_failure()),
            "Collection /2 teacher/worker retirement is not observed",
        )?;
        let status = self
            .owner
            .try_status()?
            .ok_or_else(|| invalid("Collection /2 owner busy"))?;
        check(
            status.admitted
                && status.settled
                && status.pending_count == 0
                && status.pending_disabled_markers == Some(0)
                && status.incomplete_capture_count == 0
                && status.retained_failed_captures == 0
                && status.retained_failed_staged == 0
                && status.failure.is_none()
                && !status.callback_lock_failure,
            "Collection /2 native observer is not settled",
        )?;
        Ok(self.owner.clone())
    }
}

impl Drop for RoventPcCollectionRunV2 {
    fn drop(&mut self) {
        let owner_resolved = self
            .owner
            .try_status()
            .ok()
            .flatten()
            .is_some_and(|status| {
                status.admitted
                    && status.settled
                    && status.pending_count == 0
                    && status.pending_disabled_markers == Some(0)
                    && status.collection_disabled_first_ticket.is_none()
                    && status.prior_disabled_first_tickets.is_empty()
                    && status.pending_retired_count == 0
                    && status.incomplete_capture_count == 0
                    && status.retained_failed_captures == 0
                    && status.retained_failed_staged == 0
                    && status.failure.is_none()
                    && !status.callback_lock_failure
            });
        let physically_resolved = owner_resolved
            && self.native_failure.is_none()
            && !self.native_failure_taken
            && self.resource_error.is_none()
            && self
                .worker
                .try_lock()
                .is_ok_and(|worker| !worker.has_failure())
            && self.worker_exit == Some(PcWorkerExitKindV2::Retirement);
        if !physically_resolved {
            // A dropped observation must not release a still-running or
            // unverified owner. Keep the exact objects for process lifetime;
            // this is quarantine, not a physical-completion receipt.
            std::mem::forget(Arc::clone(&self.resource));
            std::mem::forget(Arc::clone(&self.original_lock));
            std::mem::forget(self.owner.clone());
            std::mem::forget(Arc::clone(&self.bridge));
            std::mem::forget(Arc::clone(&self.worker));
        }
    }
}

/// Load an explicitly pinned /2 manifest and start the actual P-C UCI path.
/// The plan's single-root limit is the bridge's current capability; it must
/// not be reported as multi-root Collection support.
pub fn run_rovent_pc_collection_lock_v2(
    arguments: Vec<String>,
    plan: RoventCollectionLockV2LaunchPlan,
    input_directory: &Dir,
    worker_deadline: Instant,
    cleanup_deadline: Instant,
) -> Result<RoventPcCollectionRunV2, ArenaError> {
    run_rovent_pc_collection_lock_v2_inner(
        arguments,
        plan,
        input_directory,
        worker_deadline,
        cleanup_deadline,
        |_| Ok(()),
    )
}

/// Test harness hook: receive the actual owner created by this launch. This
/// does not bypass pinned inputs, startup, Ready, or UCI/native admission.
#[cfg(feature = "test-fixtures")]
pub fn run_rovent_pc_collection_lock_v2_with_test_owner(
    arguments: Vec<String>,
    plan: RoventCollectionLockV2LaunchPlan,
    input_directory: &Dir,
    worker_deadline: Instant,
    cleanup_deadline: Instant,
    on_owner: impl FnOnce(&RoventCollectionOwnerLockV2) -> Result<(), ArenaError>,
) -> Result<RoventPcCollectionRunV2, ArenaError> {
    run_rovent_pc_collection_lock_v2_inner(
        arguments,
        plan,
        input_directory,
        worker_deadline,
        cleanup_deadline,
        on_owner,
    )
}

fn run_rovent_pc_collection_lock_v2_inner(
    arguments: Vec<String>,
    plan: RoventCollectionLockV2LaunchPlan,
    input_directory: &Dir,
    worker_deadline: Instant,
    cleanup_deadline: Instant,
    on_owner: impl FnOnce(&RoventCollectionOwnerLockV2) -> Result<(), ArenaError>,
) -> Result<RoventPcCollectionRunV2, ArenaError> {
    check(
        plan.maximum_roots == 1,
        "P/C Collection /2 bridge currently accepts one root",
    )?;
    check(
        Instant::now() < worker_deadline && worker_deadline <= cleanup_deadline,
        "P/C Collection /2 finite worker/cleanup deadline",
    )?;
    let bytes = read_pin(input_directory, &plan.manifest, 16 << 20)?;
    let manifest = Arc::new(RoventCorpusManifestLockV2::from_json(
        std::str::from_utf8(&bytes).map_err(|_| invalid("Collection /2 manifest UTF8"))?,
    )?);
    plan.validate_registration(&manifest)?;
    let (owner, observer) =
        RoventCollectionOwnerLockV2::new_shared(Arc::clone(&manifest), plan.metadata.clone())?;
    let roster = plan
        .teacher_slots
        .iter()
        .map(|entry| RoventTeacherRosterEntryV2 {
            actor_id: entry.actor_id.clone(),
            source_index: entry.source_index,
            collection_owner: entry.collection_owner,
        })
        .collect();
    let (bridge, worker) =
        RoventPcCollectionBridgeV2::new(owner.clone(), Arc::clone(&manifest), roster)?;
    let worker = Arc::new(Mutex::new(worker));
    let resource = Arc::new(Mutex::new(resources::CollectionResourceCustody::prepare(
        input_directory,
        &plan.resource_plan,
    )?));
    let original_lock = Arc::new(Mutex::new(None));
    let admission_resource = Arc::clone(&resource);
    let admission_lock = Arc::clone(&original_lock);
    let stop = Arc::new(AtomicBool::new(false));
    let worker_start = Arc::new(AtomicU8::new(WORKER_NOT_STARTED));
    let admission_bridge = Arc::clone(&bridge);
    let admission_worker = Arc::clone(&worker);
    let admission_owner = owner.clone();
    let admission_stop = Arc::clone(&stop);
    let admission_worker_start = Arc::clone(&worker_start);
    let hooks: Arc<dyn rz_uci::rovent_bootstrap::pc_session::RoventPcCollectionHooksV2> =
        bridge.clone();
    let native = run_pc_collection_v2_with_resource_admission(
        arguments,
        plan.teacher_sources,
        Box::new(observer),
        hooks,
        move |lock| {
            let mut original = admission_lock.lock().map_err(|_| RoventLaunchError {
                stage: "resource_admission",
                detail: "Collection /2 lock custody poisoned".into(),
            })?;
            *original = Some(Arc::clone(lock));
            drop(original);
            manifest.validate(lock).map_err(|error| RoventLaunchError {
                stage: "collection_manifest_v2",
                detail: error.to_string(),
            })?;
            admission_resource
                .lock()
                .map_err(|_| RoventLaunchError {
                    stage: "resource_admission",
                    detail: "Collection /2 resource custody poisoned".into(),
                })?
                .admit_v2(lock)
                .map_err(|error| RoventLaunchError {
                    stage: "resource_admission",
                    detail: error.to_string(),
                })?;
            // cgroup entry applies to the whole process and must happen
            // before test callbacks or the collection worker create threads.
            on_owner(&admission_owner).map_err(|error| RoventLaunchError {
                stage: "collection_owner_callback", detail: error.to_string(),
            })?;
            admission_worker_start.store(WORKER_STARTING, Ordering::Release);
            match admission_bridge.start_worker(&admission_worker, admission_stop, worker_deadline) {
                Ok(()) => admission_worker_start.store(WORKER_STARTED, Ordering::Release),
                Err(error) => {
                    let error = match error {
                        PcWorkerStartErrorV2::BeforeStart(error) => {
                            admission_worker_start.store(WORKER_START_FAILED, Ordering::Release);
                            error
                        }
                        PcWorkerStartErrorV2::Unknown(error) => error,
                    };
                    return Err(RoventLaunchError { stage: "collection_worker_start", detail: error.to_string() });
                }
            }
            Ok(())
        },
    );
    stop.store(true, Ordering::Release);
    let mut run = RoventPcCollectionRunV2 {
        resource,
        original_lock,
        owner,
        bridge,
        worker,
        worker_start,
        worker_exit: None,
        worker_thread_panicked: false,
        native_failure: native.err(),
        native_failure_taken: false,
        resource_error: None,
    };
    if let Ok(mut resource) = run.resource.lock() {
        if let Err(error) = resource.finish_live() {
            run.resource_error = Some(error.to_string());
        } else if run.native_failure.is_none() {
            let validated = run
                .original_lock
                .lock()
                .map_err(|_| invalid("Collection /2 original lock custody poisoned"))
                .and_then(|lock| {
                    lock.as_ref()
                        .ok_or_else(|| invalid("Collection /2 original lock missing"))
                        .and_then(|lock| resource.validate_success_v2(lock))
                });
            if let Err(error) = validated {
                run.resource_error = Some(error.to_string());
            }
        }
    } else {
        run.resource_error = Some("Collection /2 resource custody poisoned".into());
    }
    while !run.poll_worker() && Instant::now() < cleanup_deadline {
        std::thread::sleep(Duration::from_millis(2));
    }
    Ok(run)
}
