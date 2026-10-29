//! P-C input-lock /2 launch custody. A returned run is an observation, not a
//! published corpus or a claim of native/teacher physical completion.
use super::*;
use crate::rovent_collect::pc_hooks_lock_v2::{
    RoventPcCollectionBridgeV2, RoventPcCollectionWorkerExitV2, RoventPcCollectionWorkerV2,
};
use crate::rovent_collect::teacher::v2_session::RoventTeacherRosterEntryV2;
use rz_experiments::RoventInputLockV2;
use rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2;
use rz_uci::rovent_bootstrap::launch::{
    RoventPcCollectionLaunchFailureV2, run_pc_collection_v2_with_resource_admission,
};
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;

/// Keeps the original resource, native failure and teacher worker together.
/// Neither the live readback nor the thread exit establishes publication.
pub struct RoventPcCollectionRunV2 {
    resource: Arc<Mutex<resources::CollectionResourceCustody>>,
    original_lock: Arc<Mutex<Option<Arc<RoventInputLockV2>>>>,
    owner: RoventCollectionOwnerLockV2,
    bridge: Arc<RoventPcCollectionBridgeV2>,
    worker: Arc<Mutex<RoventPcCollectionWorkerV2>>,
    worker_thread: Option<JoinHandle<RoventPcCollectionWorkerExitV2>>,
    worker_exit: Option<RoventPcCollectionWorkerExitV2>,
    worker_thread_panicked: bool,
    native_failure: Option<RoventPcCollectionLaunchFailureV2>,
    resource_error: Option<String>,
}

impl RoventPcCollectionRunV2 {
    /// Collect only a thread that has already exited; never wait indefinitely
    /// or infer that a still-running worker has physically stopped.
    pub fn poll_worker(&mut self) -> bool {
        if self.worker_thread.as_ref().is_some_and(JoinHandle::is_finished) {
            let handle = self.worker_thread.take().expect("finished worker retained");
            match handle.join() {
                Ok(exit) => self.worker_exit = Some(exit),
                Err(payload) => {
                    // The worker normally catches its own panic and returns
                    // its owner; an outer panic cannot prove physical exit.
                    std::mem::forget(payload);
                    self.worker_thread_panicked = true;
                    self.resource_error.get_or_insert_with(|| "teacher worker outer panic; original worker retained, physical completion unknown".into());
                }
            }
        }
        self.worker_exit.is_some() || self.worker_thread_panicked
    }

    pub fn observation(&self) -> serde_json::Value {
        let resource = self.resource.lock().ok().map(|state| state.observation());
        let owner = self.owner.try_status().ok().flatten();
        json!({"schema":"rz-rovent-pc-lock-v2-run-observation/1",
            "native_error":self.native_failure.as_ref().map(|error|error.error.to_string()),
            "resource_error":self.resource_error,"resource":resource,
            "teacher_worker_returned":self.worker_exit.is_some(),
            "teacher_worker_running_or_unknown":self.worker_thread.is_some() || self.worker_thread_panicked
                || !matches!(self.worker_exit.as_ref(), Some(RoventPcCollectionWorkerExitV2::RetirementAndHandoffObserved)),
            "teacher_worker_thread_panicked":self.worker_thread_panicked,
            "collection_owner":owner.map(|status| json!({"admitted":status.admitted,"settled":status.settled,
                "pending":status.pending_count,"retired":status.pending_retired_count,
                "incomplete_capture":status.incomplete_capture_count,
                "failed_captures":status.retained_failed_captures,
                "failure":status.failure.map(|error|format!("{error:?}")),
                "callback_lock_failure":status.callback_lock_failure})),
            "physical_completion_claimed":false,"corpus_published":false,"training_executed":false})
    }
}

impl Drop for RoventPcCollectionRunV2 {
    fn drop(&mut self) {
        let owner_resolved = self.owner.try_status().ok().flatten().is_some_and(|status|
            status.admitted && status.settled && status.pending_count == 0
                && status.pending_retired_count == 0 && status.incomplete_capture_count == 0
                && status.retained_failed_captures == 0 && status.failure.is_none()
                && !status.callback_lock_failure);
        let physically_resolved = owner_resolved && self.native_failure.is_none()
            && self.resource_error.is_none()
            && self.worker.try_lock().is_ok_and(|worker| !worker.has_failure())
            && matches!(self.worker_exit.as_ref(), Some(RoventPcCollectionWorkerExitV2::RetirementAndHandoffObserved));
        if !physically_resolved {
            // A dropped observation must not release a still-running or
            // unverified owner. Keep the exact objects for process lifetime;
            // this is quarantine, not a physical-completion receipt.
            std::mem::forget(Arc::clone(&self.resource));
            std::mem::forget(Arc::clone(&self.original_lock));
            std::mem::forget(self.owner.clone());
            std::mem::forget(Arc::clone(&self.bridge));
            std::mem::forget(Arc::clone(&self.worker));
            if let Some(handle) = self.worker_thread.take() { std::mem::forget(handle); }
            if let Some(exit) = self.worker_exit.take() { std::mem::forget(exit); }
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
    check(plan.maximum_roots == 1, "P/C Collection /2 bridge currently accepts one root")?;
    check(Instant::now() < worker_deadline && worker_deadline <= cleanup_deadline,
        "P/C Collection /2 finite worker/cleanup deadline")?;
    let bytes = read_pin(input_directory, &plan.manifest, 16 << 20)?;
    let manifest = RoventCorpusManifestLockV2::from_json(
        std::str::from_utf8(&bytes).map_err(|_| invalid("Collection /2 manifest UTF8"))?,
    )?;
    plan.validate_registration(&manifest)?;
    let (owner, observer) = RoventCollectionOwnerLockV2::new(manifest.clone(), plan.metadata.clone())?;
    let roster = plan.teacher_slots.iter().map(|entry| RoventTeacherRosterEntryV2 {
        actor_id: entry.actor_id.clone(), source_index: entry.source_index,
        collection_owner: entry.collection_owner,
    }).collect();
    let (bridge, worker) = RoventPcCollectionBridgeV2::new(
        owner.clone(), Arc::new(manifest.registration.clone()), roster,
    )?;
    let worker = Arc::new(Mutex::new(worker));
    let resource = Arc::new(Mutex::new(resources::CollectionResourceCustody::prepare(
        input_directory, &plan.resource_plan,
    )?));
    let original_lock = Arc::new(Mutex::new(None));
    let admission_resource = Arc::clone(&resource);
    let admission_lock = Arc::clone(&original_lock);
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = Arc::clone(&stop);
    let thread_worker = Arc::clone(&worker);
    let worker_thread = std::thread::Builder::new().name("rovent-pc-collection-v2".into())
        .spawn(move || match thread_worker.lock() {
            Ok(mut owner) => owner.run_until(worker_stop, worker_deadline),
            Err(_) => panic!("teacher worker owner mutex poisoned before run"),
        })
        .map_err(io)?;
    let hooks: Arc<dyn rz_uci::rovent_bootstrap::pc_session::RoventPcCollectionHooksV2> = bridge.clone();
    let native = run_pc_collection_v2_with_resource_admission(
        arguments, plan.teacher_sources, Box::new(observer), hooks,
        move |lock| {
            let mut original = admission_lock.lock().map_err(|_| RoventLaunchError {
                stage: "resource_admission", detail: "Collection /2 lock custody poisoned".into(),
            })?;
            *original = Some(Arc::clone(lock));
            manifest.validate(lock).map_err(|error| RoventLaunchError {
                stage: "collection_manifest_v2", detail: error.to_string(),
            })?;
            admission_resource.lock().map_err(|_| RoventLaunchError {
                stage: "resource_admission", detail: "Collection /2 resource custody poisoned".into(),
            })?.admit_v2(lock).map_err(|error| RoventLaunchError {
                stage: "resource_admission", detail: error.to_string(),
            })
        },
    );
    stop.store(true, Ordering::Release);
    let mut run = RoventPcCollectionRunV2 {
        resource, original_lock, owner, bridge, worker, worker_thread: Some(worker_thread), worker_exit: None,
        worker_thread_panicked: false, native_failure: native.err(), resource_error: None,
    };
    if let Ok(mut resource) = run.resource.lock() {
        if let Err(error) = resource.finish_live() {
            run.resource_error = Some(error.to_string());
        } else if run.native_failure.is_none() {
            let validated = run.original_lock.lock().map_err(|_| invalid("Collection /2 original lock custody poisoned"))
                .and_then(|lock| lock.as_ref().ok_or_else(|| invalid("Collection /2 original lock missing"))
                    .and_then(|lock| resource.validate_success_v2(lock)));
            if let Err(error) = validated { run.resource_error = Some(error.to_string()); }
        }
    } else {
        run.resource_error = Some("Collection /2 resource custody poisoned".into());
    }
    while !run.poll_worker() && Instant::now() < cleanup_deadline {
        std::thread::sleep(Duration::from_millis(2));
    }
    Ok(run)
}
