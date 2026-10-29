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
    pump_consumer: Arc<super::own_publisher_v2::OwnPumpConsumerV2>,
    pump_thread: Option<std::thread::JoinHandle<Result<(), ArenaError>>>,
    pump_thread_failure: Option<String>,
    pump_thread_panic: Option<Box<dyn std::any::Any + Send>>,
    pump_thread_joined: bool,
    pump_published: bool,
    publication_manifest: Arc<RoventCorpusManifestLockV2>,
    publication_directory: std::path::PathBuf,
    publication_maximum_records: u64,
    publication_assembly: Option<RoventCorpusAssemblyLockV2>,
    publication_dir: Option<Dir>,
    publication_attempted: bool,
    publication: Option<rz_experiments::rovent_corpus_lock_v2::publication::RoventLockV2Publication>,
    late_rows: std::collections::VecDeque<RoventRetiredLockV2Record>,
    worker_start: Arc<AtomicU8>,
    worker_exit: Option<PcWorkerExitKindV2>,
    worker_thread_panicked: bool,
    native_failure: Option<RoventPcCollectionLaunchFailureV2>,
    native_failure_taken: bool,
    resource_error: Option<String>,
}

impl RoventPcCollectionRunV2 {
    /// Write serializable failure observations and original recovered teacher
    /// byte streams without consuming their owners. A partial write is left
    /// in place and a later attempt cannot silently replace it.
    pub fn write_failure_evidence(&self, publication_failure: Option<&str>) -> Result<bool, ArenaError> {
        use sha2::{Digest as _, Sha256};
        let mut raw = Vec::<(String, Vec<u8>)>::new();
        let observation = self.observation();
        let (worker_evidence, mut complete) = if self.worker_exit.is_some() && !self.worker_thread_panicked {
            match self.worker.try_lock() {
                Ok(worker) => {
                    let diagnostics = worker.diagnostics();
                    let bytes = diagnostics.values().try_fold(0usize, |total, value| {
                        total.checked_add(value.stdout().len())?.checked_add(value.stderr().len())
                    });
                    if !bytes.is_some_and(|n| n <= 4 * 1024 * 1024) {
                        (json!({"status":"diagnostics_over_limit"}), false)
                    } else {
                        let mut entries = Vec::new();
                        for (index, (actor, value)) in diagnostics.iter().enumerate() {
                            let stdout = format!("teacher-{index}-stdout.bin");
                            let stderr = format!("teacher-{index}-stderr.bin");
                            entries.push(json!({"actor":actor,
                                "stdout":{"path":&stdout,"bytes":value.stdout().len(),"sha256":format!("{:x}", Sha256::digest(value.stdout()))},
                                "stderr":{"path":&stderr,"bytes":value.stderr().len(),"sha256":format!("{:x}", Sha256::digest(value.stderr()))}}));
                            raw.push((stdout, value.stdout().to_vec()));
                            raw.push((stderr, value.stderr().to_vec()));
                        }
                        let failure = worker.failure_observation();
                        let retirement = format!("{:?}", worker.teacher_observations());
                        if retirement.len() > 256 * 1024 {
                            raw.clear();
                            (json!({"status":"retirement_observation_over_limit"}), false)
                        } else {
                            (json!({"status":"observed", "primary_failure":failure.primary.map(|f| json!({"kind":f.kind(),"detail":f.diagnostic()})),
                                "additional_failures":failure.additional.iter().map(|f| json!({"kind":f.kind(),"detail":f.diagnostic()})).collect::<Vec<_>>(),
                                "omitted_failures":failure.omitted, "teacher_retirement_observation_debug":retirement,
                                "teacher_diagnostics":entries}), failure.omitted == 0)
                        }
                    }
                }
                Err(std::sync::TryLockError::WouldBlock) => (json!({"status":"worker_busy"}), false),
                Err(std::sync::TryLockError::Poisoned(_)) => (json!({"status":"worker_poisoned"}), false),
            }
        } else { (json!({"status":"worker_not_joined_or_panicked"}), false) };
        // A serialized late record is diagnostic data only; it cannot
        // reconstruct the original in-process admission or producer owner.
        let mut late_entries = Vec::new();
        let mut late_bytes = 0usize;
        for (index, row) in self.late_rows.iter().enumerate() {
            let bytes = serde_json::to_vec(&row.record).map_err(|e| invalid(e.to_string()))?;
            let Some(next) = late_bytes.checked_add(bytes.len()) else { complete = false; break; };
            if next > 4 * 1024 * 1024 { complete = false; break; }
            late_bytes = next;
            let name = format!("late-row-{index}.json");
            late_entries.push(json!({"path":&name, "bytes":bytes.len(),
                "sha256":format!("{:x}", Sha256::digest(&bytes)), "authority":"not_reconstructable"}));
            raw.push((name, bytes));
        }
        let mut evidence = serde_json::to_vec(&json!({"schema":"rz-rovent-pc-failure-evidence/1",
            "raw_files_complete_only_when_receipt_exists":true,
            "publication_failure":publication_failure.map(|s| s.chars().take(4096).collect::<String>()),
            "run_observation":observation, "worker_evidence":worker_evidence,
            "late_rows_total":self.late_rows.len(), "late_rows_serialized":late_entries}))
            .map_err(|e| invalid(e.to_string()))?;
        if evidence.len() > 1024 * 1024 {
            raw.clear();
            complete = false;
            evidence = serde_json::to_vec(&json!({"schema":"rz-rovent-pc-failure-evidence/1",
                "status":"metadata_over_limit", "worker_exit":self.worker_exit.map(PcWorkerExitKindV2::label),
                "worker_thread_panicked":self.worker_thread_panicked,
                "physical_completion":"not_recorded",
                "publication_failure":publication_failure.map(|s| s.chars().take(4096).collect::<String>()),
                "native_failure_present":self.native_failure.is_some(), "resource_failure_present":self.resource_error.is_some(),
                "detail":"Original owner remains in custody; detailed metadata exceeded 1 MiB"}))
                .map_err(|e| invalid(e.to_string()))?;
        }
        let parent = self.publication_directory.parent().ok_or_else(|| invalid("P/C output parent"))?;
        let leaf = self.publication_directory.file_name().ok_or_else(|| invalid("P/C output leaf"))?;
        let root = Dir::open_ambient_dir(parent, cap_std::ambient_authority()).map_err(io)?;
        let directory = match root.open_dir_nofollow(leaf) {
            Ok(directory) => directory,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                root.create_dir(leaf).map_err(io)?;
                root.open_dir_nofollow(leaf).map_err(io)?
            }
            Err(error) => return Err(io(error)),
        };
        directory.create_dir("pc-failure-evidence").map_err(io)?;
        let directory = directory.open_dir_nofollow("pc-failure-evidence").map_err(io)?;
        let mut options = cap_std::fs::OpenOptions::new();
        options.write(true).create_new(true).follow(FollowSymlinks::No);
        // Preserve the planned metadata even if a later raw byte write fails;
        // only receipt.json records completion of all listed writes.
        let mut planned = directory.open_with("planned.json", &options).map_err(io)?;
        planned.write_all(&evidence).and_then(|_| planned.sync_all()).map_err(io)?;
        for (name, bytes) in raw {
            let mut file = directory.open_with(name, &options).map_err(io)?;
            file.write_all(&bytes).and_then(|_| file.sync_all()).map_err(io)?;
        }
        let mut file = directory.open_with("receipt.json", &options).map_err(io)?;
        file.write_all(&evidence).and_then(|_| file.sync_all()).map_err(io)?;
        Ok(complete)
    }
    /// Complete a single-root publication using only the original joined
    /// worker, settled handoff and received rows. Failures retain the owner,
    /// consumer and any assembly; they never imply physical completion.
    pub fn publish_single_root(&mut self, until: Instant) -> Result<bool, ArenaError> {
        use super::own_publisher_v2::{OwnOfferOutcomeV2, OwnPumpPollV2};
        if self.pump_published { return Ok(true); }
        check(!self.publication_attempted, "P/C previous publication attempt requires explicit recovery")?;
        check(self.pump_thread_joined && self.pump_thread_failure.is_none()
            && matches!(self.worker_exit, Some(PcWorkerExitKindV2::Aggregate | PcWorkerExitKindV2::Retirement))
            && self.bridge.returned_handoff_settled().map_err(|e| invalid(format!("P/C handoff: {e:?}")))?
            && self.closed_owner().is_ok(), "P/C original worker, native fence or teacher aggregate unavailable")?;
        // The original observer may retire rows just before its root fence.
        // Recover them after the original worker joined, outside the native
        // manager; never synthesize a teacher result for those late rows.
        loop {
            check(Instant::now() < until, "P/C late retired row recovery deadline reached")?;
            let Some(status) = self.owner.try_status()? else {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            };
            if status.pending_retired_count == 0 { break; }
            check(self.late_rows.len() < self.publication_maximum_records as usize,
                "P/C late retired row recovery capacity reached")?;
            self.late_rows.try_reserve(1).map_err(|_| invalid("P/C late row custody capacity"))?;
            match self.owner.try_take_settled_retired()? {
                Some(Some(row)) => self.late_rows.push_back(row),
                Some(None) => break,
                None => std::thread::sleep(Duration::from_millis(1)),
            }
        }
        match self.worker.try_lock() {
            Ok(worker) => {
                check(!worker.has_failure(), "P/C worker has retained failure")?;
                check(!worker.has_unpublished_pump_rows(), "P/C worker has unpublished pump rows")?;
            }
            Err(std::sync::TryLockError::WouldBlock) => return Err(invalid("P/C worker observation busy")),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("P/C worker observation poisoned")),
        }
        check(self.late_rows.iter().all(|row| row.submitted_input_seal().teacher_actor_id.is_none()
            || row.record.0.input.legal_moves.is_empty()),
            "P/C late retired rows require teacher evidence; originals retained")?;
        check(self.owner.try_status()?.is_some_and(|status| status.pending_retired_count == 0),
            "P/C owner still retains or queues retired original rows")?;
        check(matches!(self.bridge.try_has_deferred()?, Some(false))
            && self.owner.try_status()?.is_some_and(|status|
                status.collection_disabled_first_ticket.is_none() && status.prior_disabled_first_tickets.is_empty()),
            "P/C collection was deferred or capture was disabled")?;
        check(Instant::now() < until, "P/C publication deadline expired")?;
        let admission = self.bridge.installed_admission_for_recovery()
            .ok_or_else(|| invalid("P/C original publication admission unavailable"))?.clone();
        // Mark before filesystem mutation; a partial create/write must not be
        // silently retried as a fresh publication.
        self.publication_attempted = true;
        let parent = self.publication_directory.parent().ok_or_else(|| invalid("P/C output parent"))?;
        let leaf = self.publication_directory.file_name().ok_or_else(|| invalid("P/C output leaf"))?;
        let root = Dir::open_ambient_dir(parent, cap_std::ambient_authority()).map_err(io)?;
        root.create_dir(leaf).map_err(io)?;
        let directory = root.open_dir_nofollow(leaf).map_err(io)?;
        directory.create_dir("chunks").map_err(io)?;
        self.publication_dir = Some(directory);
        self.publication_assembly = Some(RoventCorpusAssemblyLockV2::new(
            &self.owner, self.publication_dir.as_ref().expect("original output owner"),
            &self.publication_manifest, &admission, self.publication_maximum_records)?);
        let mut assembly = self.publication_assembly.take().expect("original assembly");
        let result = (|| -> Result<bool, ArenaError> {
            loop {
                check(Instant::now() < until, "P/C publication deadline expired")?;
                let polled = self.pump_consumer.poll()?;
                match self.try_offer_consumer_item(&self.pump_consumer, &mut assembly)? {
                    OwnOfferOutcomeV2::Offered => {
                        let advanced = match self.try_advance_consumer_assembly(&mut assembly) {
                            Err(error) if assembly.chunk_flush_required()? => {
                                assembly.flush()?;
                                self.try_advance_consumer_assembly(&mut assembly).map_err(|retry| invalid(format!("P/C append before flush: {error}; retry: {retry}")))?
                            }
                            other => other?,
                        };
                        if !advanced {
                            return Err(invalid("P/C original commit gate remained busy"));
                        }
                    }
                    OwnOfferOutcomeV2::FailedRetained => return Err(invalid("P/C teacher row failed; original retained")),
                    OwnOfferOutcomeV2::Pending => match polled {
                        OwnPumpPollV2::Received | OwnPumpPollV2::Full => return Err(invalid("P/C pump item remained pending")),
                        OwnPumpPollV2::Empty | OwnPumpPollV2::Disconnected => break,
                    },
                }
            }
            while !self.late_rows.is_empty() {
                check(Instant::now() < until, "P/C late row publication deadline expired")?;
                let row = self.late_rows.pop_front().expect("late original row retained");
                if let Err(original) = assembly.offer(row) {
                    let (error, row) = *original;
                    self.late_rows.push_front(row);
                    return Err(error);
                }
                let advanced = match self.try_advance_consumer_assembly(&mut assembly) {
                    Err(error) if assembly.chunk_flush_required()? => {
                        assembly.flush()?;
                        self.try_advance_consumer_assembly(&mut assembly).map_err(|retry| invalid(format!("P/C append before flush: {error}; retry: {retry}")))?
                    }
                    other => other?,
                };
                check(advanced, "P/C late row original commit gate remained busy")?;
            }
            check(assembly.records() > 0 && matches!(self.pump_consumer.try_has_failures()?, Some(false)),
                "P/C publication has no valid rows or retains a teacher failure")?;
            check(self.owner.try_status()?.is_some_and(|status| status.pending_retired_count == 0)
                && self.worker.try_lock().is_ok_and(|worker| !worker.has_failure() && !worker.has_unpublished_pump_rows()),
                "P/C original rows remain outside the assembled corpus")?;
            assembly.flush()?;
            let directory = self.publication_dir.as_ref().expect("original output owner");
            let finished = assembly.finish()?;
            let publication = finished.publish_atomic(directory, until, &AtomicBool::new(false))?;
            self.publication = Some(publication);
            Ok(true)
        })();
        self.publication_assembly = Some(assembly);
        if matches!(&result, Ok(true)) { self.pump_published = true; }
        result
    }
    /// Offer an original pump item only under this root's real joined worker,
    /// handoff and commit gate. It is not a publication receipt.
    pub(super) fn try_offer_consumer_item(
        &self,
        consumer: &super::own_publisher_v2::OwnPumpConsumerV2,
        assembly: &mut RoventCorpusAssemblyLockV2,
    ) -> Result<super::own_publisher_v2::OwnOfferOutcomeV2, ArenaError> {
        use super::own_publisher_v2::OwnOfferOutcomeV2;
        if self.worker_exit.is_none() || self.worker_thread_panicked { return Ok(OwnOfferOutcomeV2::Pending); }
        let worker = match self.worker.try_lock() {
            Ok(worker) => worker,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnOfferOutcomeV2::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("P/C original collection worker poisoned")),
        };
        let Some(handoff) = worker.original_handoff() else { return Ok(OwnOfferOutcomeV2::Pending); };
        let mut handoff = match handoff.try_lock() {
            Ok(handoff) => handoff,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnOfferOutcomeV2::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("P/C original handoff poisoned")),
        };
        let admission = handoff.parts_mut().0.clone();
        let Some(gate) = self.bridge.try_begin_commit(&admission)? else { return Ok(OwnOfferOutcomeV2::Pending); };
        consumer.try_offer_front_pc(&self.owner, assembly, &gate, &mut handoff)
    }

    pub(super) fn try_advance_consumer_assembly(
        &self,
        assembly: &mut RoventCorpusAssemblyLockV2,
    ) -> Result<bool, ArenaError> {
        if self.worker_exit.is_none() || self.worker_thread_panicked { return Ok(false); }
        let worker = match self.worker.try_lock() {
            Ok(worker) => worker,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("P/C original collection worker poisoned")),
        };
        let Some(handoff) = worker.original_handoff() else { return Ok(false); };
        let mut handoff = match handoff.try_lock() {
            Ok(handoff) => handoff,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("P/C original handoff poisoned")),
        };
        let admission = handoff.parts_mut().0.clone();
        let Some(gate) = self.bridge.try_begin_commit(&admission)? else { return Ok(false); };
        assembly.try_advance_fenced_pc(&gate, &mut handoff)
    }
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
            let (unlabeled, prepared, failed, committed) = worker.pump_ready_counts();
            json!({"primary_failure":failures.primary.map(|failure|failure.kind()),
                "worker_failed":worker.has_failure(),
                "additional_failures":failures.additional.iter().map(|failure|failure.kind()).collect::<Vec<_>>(),
                "omitted_failures":failures.omitted,
                "pending_pump_rows":worker.pending_pump_rows(),
                "ready_unlabeled":unlabeled,"ready_prepared":prepared,"ready_failed":failed,"ready_committed":committed,
                "diagnostic_actors":worker.diagnostics().len()})
        });
        let deferred_observation = match self.bridge.try_deferred_observation() {
            Ok(Some(value)) => json!({"status":"available", "detail":value}),
            Ok(None) => json!({"status":"busy"}),
            Err(error) => json!({"status":"failed", "error":error.to_string()}),
        };
        json!({"schema":"rz-rovent-pc-lock-v2-run-observation/1",
            "native_error":self.native_failure.as_ref().map(|error|error.error.to_string()),
            "native_failure_transferred":self.native_failure_taken,
            "resource_error":self.resource_error,"resource":resource,
            "teacher_worker_start":match start { WORKER_NOT_STARTED => "not_started", WORKER_STARTING => "unknown", WORKER_STARTED => "started", WORKER_START_FAILED => "failed_before_start", _ => "unknown" },
            "teacher_worker_returned":self.worker_exit.is_some(),
            "teacher_worker_exit":worker_exit,
            "teacher_worker_running_or_unknown": match start {
                WORKER_NOT_STARTED | WORKER_START_FAILED => false,
                _ => self.worker_exit.is_none() && !self.worker_thread_panicked,
            },
            "teacher_worker_thread_panicked":self.worker_thread_panicked,
            "pump_consumer_joined":self.pump_thread_joined,
            "pump_consumer_failure":self.pump_thread_failure,
            "corpus_published":self.pump_published,
            "publisher_receipt":self.publication,
            "publication_attempted":self.publication_attempted,
            "late_retired_rows_retained":self.late_rows.len(),
            "teacher_worker_observation":worker_observation,
            "deferred_collection":deferred_observation,
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
            "physical_completion_claimed":false,"training_executed":false})
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

    pub fn take_pump_panic_payload(&mut self) -> Option<Box<dyn std::any::Any + Send>> {
        self.pump_thread_panic.take()
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
        let fenced = self.worker_exit == Some(PcWorkerExitKindV2::Retirement)
            || (self.worker_exit == Some(PcWorkerExitKindV2::Aggregate)
                && self.bridge.returned_handoff_settled().map_err(|error| invalid(format!("P/C returned handoff: {error:?}")))?);
        check(
            fenced && self.native_failure.is_none() && !self.native_failure_taken && self.resource_error.is_none() && self
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
            && matches!(self.bridge.try_has_deferred(), Ok(Some(false)))
            && self.native_failure.is_none()
            && !self.native_failure_taken
            && self.resource_error.is_none()
            && self
                .worker
                .try_lock()
                .is_ok_and(|worker| !worker.has_failure() && !worker.has_unpublished_pump_rows())
            && (self.worker_exit == Some(PcWorkerExitKindV2::Retirement)
                || (self.worker_exit == Some(PcWorkerExitKindV2::Aggregate)
                    && self.bridge.returned_handoff_settled().unwrap_or(false)));
        // Native completion does not release the producer while any original
        // row, unpublished assembly or consumer may still depend on it.
        let collection_custody_resolved = self.pump_published && self.late_rows.is_empty();
        if !physically_resolved || !collection_custody_resolved {
            // A dropped observation must not release a still-running or
            // unverified owner. Keep the exact objects for process lifetime;
            // this is quarantine, not a physical-completion receipt.
            std::mem::forget(Arc::clone(&self.resource));
            std::mem::forget(Arc::clone(&self.original_lock));
            std::mem::forget(self.owner.clone());
            std::mem::forget(Arc::clone(&self.bridge));
            std::mem::forget(Arc::clone(&self.worker));
        }
        if !self.pump_thread_joined || self.pump_thread_failure.is_some() || !self.pump_published {
            std::mem::forget(Arc::clone(&self.pump_consumer));
            if let Some(thread) = self.pump_thread.take() { std::mem::forget(thread); }
            if let Some(payload) = self.pump_thread_panic.take() { std::mem::forget(payload); }
            if let Some(assembly) = self.publication_assembly.take() { std::mem::forget(assembly); }
            if let Some(directory) = self.publication_dir.take() { std::mem::forget(directory); }
            if !self.late_rows.is_empty() { std::mem::forget(std::mem::take(&mut self.late_rows)); }
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
    let (bridge, mut worker) =
        RoventPcCollectionBridgeV2::new(owner.clone(), Arc::clone(&manifest), roster)?;
    let (pump_consumer, pump_sender) = super::own_publisher_v2::OwnPumpConsumerV2::new(plan.maximum_records as usize);
    worker.configure_row_pump(plan.teacher_constraint, Duration::from_nanos(plan.teacher_soft_quantum_ns))?;
    worker.configure_pump_sender(pump_sender)?;
    let worker = Arc::new(Mutex::new(worker));
    let resource = Arc::new(Mutex::new(resources::CollectionResourceCustody::prepare(
        input_directory,
        &plan.resource_plan,
    )?));
    let original_lock = Arc::new(Mutex::new(None));
    let admission_resource = Arc::clone(&resource);
    let admission_lock = Arc::clone(&original_lock);
    let stop = Arc::new(AtomicBool::new(false));
    let pump_stop = Arc::new(AtomicBool::new(false));
    let worker_start = Arc::new(AtomicU8::new(WORKER_NOT_STARTED));
    let admission_bridge = Arc::clone(&bridge);
    let admission_worker = Arc::clone(&worker);
    let admission_owner = owner.clone();
    let admission_stop = Arc::clone(&stop);
    let admission_worker_start = Arc::clone(&worker_start);
    let pump_thread = Arc::new(Mutex::new(None));
    let admission_pump_thread = Arc::clone(&pump_thread);
    let admission_pump_consumer = Arc::clone(&pump_consumer);
    let admission_pump_stop = Arc::clone(&pump_stop);
    let publication_manifest = Arc::clone(&manifest);
    let publication_directory = plan.output_directory.clone();
    let publication_maximum_records = plan.maximum_records;
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
            let mut pump_slot = admission_pump_thread.lock().map_err(|_| RoventLaunchError {
                stage: "collection_pump_start", detail: "pump thread custody poisoned".into(),
            })?;
            let thread = std::thread::Builder::new().name("rovent-pc-pump-consumer-v2".into())
                .spawn(move || admission_pump_consumer.run(&admission_pump_stop, worker_deadline))
                .map_err(|error| RoventLaunchError { stage: "collection_pump_start", detail: error.to_string() })?;
            *pump_slot = Some(thread);
            drop(pump_slot);
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
    let pump_thread = pump_thread.lock().map_err(|_| invalid("P/C pump thread custody poisoned"))?.take();
    let mut run = RoventPcCollectionRunV2 {
        resource,
        original_lock,
        owner,
        bridge,
        worker,
        pump_consumer,
        pump_thread,
        pump_thread_failure: None,
        pump_thread_panic: None,
        pump_thread_joined: false,
        pump_published: false,
        publication_manifest,
        publication_directory,
        publication_maximum_records,
        publication_assembly: None,
        publication_dir: None,
        publication_attempted: false,
        publication: None,
        late_rows: std::collections::VecDeque::new(),
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
    pump_stop.store(true, Ordering::Release);
    while run.pump_thread.as_ref().is_some_and(|thread| !thread.is_finished()) && Instant::now() < cleanup_deadline {
        std::thread::sleep(Duration::from_millis(2));
    }
    match run.pump_thread.take() {
        Some(thread) if thread.is_finished() => {
            run.pump_thread_joined = true;
            match thread.join() {
                Ok(Ok(())) => {},
                Ok(Err(error)) => run.pump_thread_failure = Some(error.to_string()),
                Err(payload) => {
                    run.pump_thread_panic = Some(payload);
                    run.pump_thread_failure = Some("P/C pump consumer panicked".into());
                }
            }
        }
        Some(thread) => run.pump_thread = Some(thread),
        None => run.pump_thread_joined = true,
    }
    Ok(run)
}
