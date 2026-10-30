//! P-C input-lock /2 launch custody. A returned run is an observation, not a
//! published corpus or a claim of native/teacher physical completion.
use super::*;
use crate::rovent_collect::pc_hooks_lock_v2::{
    RoventPcCollectionBridgeV2, RoventPcCollectionWorkerV2, PcWorkerExitKindV2, PcWorkerStartErrorV2,
};
use crate::rovent_collect::teacher::v2_session::RoventTeacherRosterEntryV2;
use crate::rovent_collect::pc_worker_supervisor_v2::{PcSecondarySlotV2, PcWorkerSupervisorV2};
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
    supervisor: Option<PcWorkerSupervisorV2>,
    secondary_slots: Vec<PcSecondarySlotV2>,
    supervisor_error: Option<ArenaError>,
    worker_start: Arc<AtomicU8>,
    worker_exit: Option<PcWorkerExitKindV2>,
    worker_thread_panicked: bool,
    native_failure: Option<RoventPcCollectionLaunchFailureV2>,
    native_failure_taken: bool,
    resource_error: Option<String>,
}

impl RoventPcCollectionRunV2 {
    fn failure_projection(failure: &crate::rovent_collect::collection_hooks_core_v2::WorkerFailure)
        -> (serde_json::Value, bool) {
        let (detail, original_chars, truncated) = failure.diagnostic_projection();
        (json!({"kind":failure.kind(), "detail":detail, "original_chars":original_chars,
            "detail_truncated":truncated}), truncated)
    }
    /// Drain only an already stopped sender into its original bounded
    /// consumer. Empty counts alone do not prove its receiver was drained.
    fn failure_consumer_observation(
        &self,
        consumer: &super::own_publisher_v2::OwnPumpConsumerV2,
        sender_stopped: bool,
        slot_index: usize,
        raw: &mut Vec<(String, Vec<u8>)>,
        record_bytes: &mut usize,
    ) -> (serde_json::Value, bool) {
        use super::own_publisher_v2::OwnPumpPollV2;
        use sha2::{Digest as _, Sha256};
        if !sender_stopped { return (json!({"status":"sender_not_joined"}), false); }
        let mut drained = false;
        let mut problem = None;
        for _ in 0..self.publication_maximum_records.saturating_add(17) {
            match consumer.poll() {
                Ok(OwnPumpPollV2::Received) => continue,
                Ok(OwnPumpPollV2::Empty | OwnPumpPollV2::Disconnected) => { drained = true; break; }
                Ok(OwnPumpPollV2::Full) => { problem = Some("consumer_capacity".to_owned()); break; }
                Err(error) => { problem = Some(error.to_string()); break; }
            }
        }
        let counts = match consumer.try_counts() {
            Ok(Some(counts)) => counts,
            Ok(None) => return (json!({"status":"busy", "receiver_drained":drained}), false),
            Err(error) => return (json!({"status":"error", "detail":error.to_string(), "receiver_drained":drained}), false),
        };
        let failures = match consumer.try_has_failures() {
            Ok(Some(value)) => value,
            Ok(None) => return (json!({"status":"busy", "receiver_drained":drained}), false),
            Err(error) => return (json!({"status":"error", "detail":error.to_string(), "receiver_drained":drained}), false),
        };
        let (projections, projected) = match consumer.try_snapshot_retained_rows(4 * 1024 * 1024 - *record_bytes) {
            Ok(Some(rows)) => {
                let mut entries = Vec::new();
                for (index, (stage, bytes)) in rows.into_iter().enumerate() {
                    *record_bytes += bytes.len();
                    let name = format!("root-{slot_index}-consumer-{index}.json");
                    entries.push(json!({"path":&name, "stage":stage, "bytes":bytes.len(),
                        "sha256":format!("{:x}", Sha256::digest(&bytes)), "authority":"not_reconstructable"}));
                    raw.push((name, bytes));
                }
                (json!({"status":"observed", "rows":entries}), true)
            },
            Ok(None) => (json!({"status":"busy"}), false),
            Err(error) => (json!({"status":"error", "detail":error.to_string()}), false),
        };
        let complete = drained && problem.is_none() && counts.0 == 0 && counts.1 == 0 && !failures && projected;
        (json!({"status":if complete { "drained_empty" } else { "retained_or_unknown" },
            "receiver_drained":drained, "items":counts.0, "failed_items":counts.1,
            "teacher_evidence":counts.2, "disconnected":counts.3,
            "failure_or_overflow":failures, "detail":problem, "record_projections":projections}), complete)
    }
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
                        let primary = failure.primary.map(Self::failure_projection);
                        let additional: Vec<_> = failure.additional.iter().map(Self::failure_projection).collect();
                        let truncated = primary.as_ref().is_some_and(|(_, truncated)| *truncated)
                            || additional.iter().any(|(_, truncated)| *truncated);
                        let retirement = format!("{:?}", worker.teacher_observations());
                        if retirement.len() > 256 * 1024 {
                            raw.clear();
                            (json!({"status":"retirement_observation_over_limit"}), false)
                        } else {
                            (json!({"status":"observed", "primary_failure":primary.map(|(value, _)| value),
                                "additional_failures":additional.into_iter().map(|(value, _)| value).collect::<Vec<_>>(),
                                "omitted_failures":failure.omitted, "teacher_retirement_observation_debug":retirement,
                                "pending_pump_rows":worker.pending_pump_rows(), "teacher_diagnostics":entries}),
                                failure.omitted == 0 && worker.pending_pump_rows() == 0 && !truncated)
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
        let mut assembly_evidence = json!({"status":"not_created"});
        if let Some(assembly) = &self.publication_assembly {
            let pending = assembly.pending_row().map(|row| &row.record);
            let sealed = assembly.sealed_row().map(|row| row.record());
            let mut records = Vec::new();
            for (name, record) in [("assembly-pending.json", pending), ("assembly-sealed.json", sealed)] {
                let Some(record) = record else { continue; };
                let bytes = serde_json::to_vec(record).map_err(|error| invalid(error.to_string()))?;
                if late_bytes.checked_add(bytes.len()).is_none_or(|next| next > 4 * 1024 * 1024) {
                    complete = false; continue;
                }
                late_bytes += bytes.len();
                records.push(json!({"path":name, "bytes":bytes.len(),
                    "sha256":format!("{:x}", Sha256::digest(&bytes)), "authority":"not_reconstructable"}));
                raw.push((name.to_owned(), bytes));
            }
            let chunk = assembly.pending_chunk().map(|packet| {
                let bytes = &packet.bytes;
                if late_bytes.checked_add(bytes.len()).is_some_and(|next| next <= 4 * 1024 * 1024) {
                    late_bytes += bytes.len();
                    raw.push(("assembly-pending-chunk.bin".to_owned(), bytes.clone()));
                    json!({"status":"projected","path":"assembly-pending-chunk.bin",
                        "bytes":bytes.len(), "sha256":format!("{:x}", Sha256::digest(bytes))})
                } else { complete = false; json!({"status":"over_limit","bytes":bytes.len()}) }
            });
            if !self.pump_published && (assembly.records() != 0 || pending.is_some() || sealed.is_some() || chunk.is_some()) {
                complete = false;
            }
            assembly_evidence = json!({"status":"retained", "records":assembly.records(),
                "pending_row":pending.is_some(), "sealed_row":sealed.is_some(), "record_projections":records,
                "pending_chunk":chunk, "chunk_write_attempted":assembly.chunk_write_attempted(),
                "writer_buffer":"not_exported"});
        }
        let (primary_consumer, primary_consumer_complete) = self.failure_consumer_observation(
            &self.pump_consumer, self.pump_thread_joined && self.worker_exit.is_some() && !self.worker_thread_panicked,
            0, &mut raw, &mut late_bytes,
        );
        complete &= primary_consumer_complete;
        match self.bridge.try_revoke_observations() {
            Ok(Some(value)) if value.get("omitted").and_then(serde_json::Value::as_u64) == Some(0) => {},
            _ => complete = false,
        }
        let mut secondary_evidence = Vec::new();
        let mut teacher_bytes = raw.iter().filter(|(name, _)| name.starts_with("teacher-"))
            .map(|(_, bytes)| bytes.len()).sum::<usize>();
        let mut bridges = Vec::with_capacity(self.secondary_slots.len() + 1);
        bridges.push(Arc::clone(&self.bridge));
        bridges.extend(self.secondary_slots.iter().map(|slot| Arc::clone(&slot.bridge)));
        for (slot_index, bridge) in bridges.iter().enumerate() {
            let admission = bridge.installed_admission_for_recovery().map(|root| root.raw_sha256().to_owned());
            if slot_index > 0 {
                let slot = &self.secondary_slots[slot_index - 1];
                if admission.is_none() && slot.worker_start == crate::rovent_collect::pc_worker_supervisor_v2::PcSecondaryWorkerStartV2::FailedBeforeStart
                    && slot.consumer_empty_before_worker_start {
                    let deferred = match bridge.try_deferred_observation() {
                        Ok(Some(value)) => json!({"status":"observed","detail":value}),
                        Ok(None) => { complete = false; json!({"status":"busy"}) },
                        Err(error) => { complete = false; json!({"status":"error","detail":error.to_string()}) },
                    };
                    secondary_evidence.push(json!({"slot":slot_index,"admission":null,"worker":{"status":"failed_before_start"},"separated":[],"deferred":deferred}));
                    continue;
                }
            }
            let (projections, separated_status) = match bridge.try_snapshot_separated_rows(4 * 1024 * 1024 - late_bytes) {
                Ok(Some(rows)) => (Some(rows), json!({"status":"observed"})),
                Ok(None) => { complete = false; (None, json!({"status":"busy"})) },
                Err(error) => { complete = false; (None, json!({"status":"error","detail":error.to_string()})) }
            };
            let mut separated = Vec::new();
            if let Some(projections) = projections {
                for (row_index, bytes) in projections.into_iter().enumerate() {
                    late_bytes += bytes.len();
                    let name = format!("root-{slot_index}-separated-{row_index}.json");
                    separated.push(json!({"path":&name,"bytes":bytes.len(),
                        "sha256":format!("{:x}", Sha256::digest(&bytes)), "authority":"not_reconstructable"}));
                    raw.push((name, bytes));
                }
            }
            if slot_index == 0 {
                secondary_evidence.push(json!({"slot":slot_index,"admission":admission,"separated":separated,"separated_status":separated_status}));
                continue;
            }
            let slot = &self.secondary_slots[slot_index - 1];
            let revokes = match bridge.try_revoke_observations() {
                Ok(Some(value)) => {
                    if value.get("omitted").and_then(serde_json::Value::as_u64) != Some(0) { complete = false; }
                    json!({"status":"observed","detail":value})
                },
                Ok(None) => { complete = false; json!({"status":"busy"}) },
                Err(error) => { complete = false; json!({"status":"error","detail":error.to_string()}) },
            };
            let (consumer, consumer_complete) = self.failure_consumer_observation(&slot.consumer,
                slot.worker_join_observed && slot.consumer_drained_after_join, slot_index, &mut raw, &mut late_bytes);
            complete &= consumer_complete;
            let deferred = match bridge.try_deferred_observation() {
                Ok(Some(value)) => json!({"status":"observed","detail":value}),
                Ok(None) => { complete = false; json!({"status":"busy"}) },
                Err(error) => { complete = false; json!({"status":"error","detail":error.to_string()}) },
            };
            if !slot.consumer_drained_after_join || !matches!(slot.worker_exit_kind,
                Some(PcWorkerExitKindV2::NoAdmission | PcWorkerExitKindV2::Retirement | PcWorkerExitKindV2::Aggregate)) {
                complete = false;
            }
            let detail = if !slot.worker_join_observed {
                complete = false;
                json!({"status":"worker_not_joined", "start":format!("{:?}", slot.worker_start)})
            } else {
                match slot.worker.try_lock() {
                    Ok(worker) => {
                        let failure = worker.failure_observation();
                        let primary = failure.primary.map(Self::failure_projection);
                        let additional: Vec<_> = failure.additional.iter().map(Self::failure_projection).collect();
                        if primary.as_ref().is_some_and(|(_, truncated)| *truncated)
                            || additional.iter().any(|(_, truncated)| *truncated) { complete = false; }
                        if failure.omitted != 0 || worker.pending_pump_rows() != 0 { complete = false; }
                        let mut diagnostics = Vec::new();
                        for (index, (actor, value)) in worker.diagnostics().iter().enumerate() {
                            let size = value.stdout().len().saturating_add(value.stderr().len());
                            if teacher_bytes.checked_add(size).is_none_or(|total| total > 4 * 1024 * 1024) {
                                complete = false; diagnostics.push(json!({"actor":actor,"status":"diagnostic_over_limit"})); continue;
                            }
                            teacher_bytes += size;
                            let stdout = format!("root-{slot_index}-teacher-{index}-stdout.bin");
                            let stderr = format!("root-{slot_index}-teacher-{index}-stderr.bin");
                            diagnostics.push(json!({"actor":actor,
                                "stdout":{"path":&stdout,"bytes":value.stdout().len(),"sha256":format!("{:x}",Sha256::digest(value.stdout()))},
                                "stderr":{"path":&stderr,"bytes":value.stderr().len(),"sha256":format!("{:x}",Sha256::digest(value.stderr()))}}));
                            raw.push((stdout, value.stdout().to_vec())); raw.push((stderr, value.stderr().to_vec()));
                        }
                        let retirement = format!("{:?}", worker.teacher_observations());
                        if retirement.len() > 256 * 1024 { complete = false; }
                        json!({"status":"observed", "exit":slot.worker_exit_kind.map(PcWorkerExitKindV2::label),
                            "consumer_drained_after_join":slot.consumer_drained_after_join,
                            "pending_pump_rows":worker.pending_pump_rows(),
                            "primary_failure":primary.map(|(value, _)| value),
                            "additional_failures":additional.into_iter().map(|(value, _)| value).collect::<Vec<_>>(),
                            "omitted_failures":failure.omitted,
                            "teacher_retirement_observation_debug":if retirement.len() <= 256 * 1024 { Some(retirement) } else { None },
                            "teacher_diagnostics":diagnostics})
                    }
                    Err(_) => { complete = false; json!({"status":"worker_busy_or_poisoned"}) }
                }
            };
            secondary_evidence.push(json!({"slot":slot_index,"admission":admission,"separated":separated,
                "separated_status":separated_status,"deferred":deferred,"revokes":revokes,"consumer":consumer,"worker":detail}));
        }
        if self.supervisor.is_some() { complete = false; }
        let publication_failure_chars = publication_failure.map(|s| s.chars().count());
        let publication_failure_truncated = publication_failure_chars.is_some_and(|count| count > 4096);
        if publication_failure_truncated { complete = false; }
        let mut evidence = serde_json::to_vec(&json!({"schema":"rz-rovent-pc-failure-evidence/1",
            "raw_files_complete_only_when_receipt_exists":true,
            "observation_complete":complete,
            "publication_failure":publication_failure.map(|s| s.chars().take(4096).collect::<String>()),
            "publication_failure_original_chars":publication_failure_chars,
            "publication_failure_truncated":publication_failure_truncated,
            "run_observation":observation, "worker_evidence":worker_evidence, "primary_consumer":primary_consumer,
            "assembly_evidence":assembly_evidence,
            "late_rows_total":self.late_rows.len(), "late_rows_serialized":late_entries,
            "secondary_slot_evidence":secondary_evidence, "supervisor_unresolved":self.supervisor.is_some()}))
            .map_err(|e| invalid(e.to_string()))?;
        if evidence.len() > 1024 * 1024 {
            raw.clear();
            complete = false;
            evidence = serde_json::to_vec(&json!({"schema":"rz-rovent-pc-failure-evidence/1",
                "observation_complete":false,
                "status":"metadata_over_limit", "worker_exit":self.worker_exit.map(PcWorkerExitKindV2::label),
                "worker_thread_panicked":self.worker_thread_panicked,
                "physical_completion":"not_recorded",
                "publication_failure":publication_failure.map(|s| s.chars().take(4096).collect::<String>()),
                "publication_failure_original_chars":publication_failure_chars,
                "publication_failure_truncated":publication_failure_truncated,
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
        check(self.supervisor.is_none() && self.supervisor_error.is_none(),
            "P/C game supervisor or secondary custody remains unresolved")?;
        if !self.secondary_slots.is_empty() {
            return self.publish_collected_game(until);
        }
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
                    OwnOfferOutcomeV2::Pending => {
                        let Some((items, failed, _, _)) = self.pump_consumer.try_counts()? else {
                            std::thread::sleep(Duration::from_millis(1)); continue;
                        };
                        check(failed == 0, "P/C consumer retains failed rows")?;
                        if items != 0 { std::thread::sleep(Duration::from_millis(1)); continue; }
                        match polled {
                            OwnPumpPollV2::Received | OwnPumpPollV2::Full => return Err(invalid("P/C pump item remained pending")),
                            OwnPumpPollV2::Empty | OwnPumpPollV2::Disconnected => break,
                        }
                    }
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
            check(self.owner.try_status()?.is_some_and(|status|
                status.game_reserved_slots == Some(0) && u64::try_from(status.game_retired_total) == Ok(assembly.records())),
                "P/C assembled row count does not match the original game owner")?;
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

    /// Process original per-root consumers and separated rows through one
    /// game writer. A joined supervisor and every root's own handoff are
    /// checked independently; pending rows remain in their original owners.
    fn publish_collected_game(&mut self, until: Instant) -> Result<bool, ArenaError> {
        use super::own_publisher_v2::{OwnOfferOutcomeV2, OwnPumpPollV2};
        use crate::rovent_collect::pc_hooks_lock_v2::PcRootSeparationV2;
        check(!self.publication_attempted && self.supervisor.is_none() && self.supervisor_error.is_none(),
            "P/C game supervisor or prior publication remains unresolved")?;
        check(self.pump_thread_joined && self.pump_thread_failure.is_none() && self.closed_owner().is_ok(),
            "P/C primary root worker or original handoff is unresolved")?;
        let mut roots = vec![(Arc::clone(&self.bridge), Arc::clone(&self.worker), Arc::clone(&self.pump_consumer))];
        roots.try_reserve(self.secondary_slots.len()).map_err(|_| invalid("P/C game publisher slot allocation"))?;
        for slot in &self.secondary_slots {
            check(matches!(slot.bridge.try_has_deferred()?, Some(false)),
                "P/C secondary slot retains deferred collection")?;
            if slot.bridge.installed_admission_for_recovery().is_none() {
                let proven_unstarted = (slot.worker_join_observed && slot.consumer_drained_after_join
                    && slot.worker_exit_kind == Some(PcWorkerExitKindV2::NoAdmission))
                    || (slot.worker_start == crate::rovent_collect::pc_worker_supervisor_v2::PcSecondaryWorkerStartV2::FailedBeforeStart
                        && slot.consumer_empty_before_worker_start);
                check(proven_unstarted, "P/C unused secondary slot start or installation is unknown")?;
                continue;
            }
            check(slot.worker_join_observed && slot.consumer_drained_after_join
                && matches!(slot.worker_exit_kind, Some(PcWorkerExitKindV2::Aggregate | PcWorkerExitKindV2::Retirement))
                && slot.bridge.returned_handoff_settled().map_err(|e| invalid(format!("P/C secondary handoff: {e:?}")))?,
                "P/C secondary original worker, consumer or handoff is unresolved")?;
            roots.push((Arc::clone(&slot.bridge), Arc::clone(&slot.worker), Arc::clone(&slot.consumer)));
        }
        for (bridge, worker, _) in &roots {
            check(matches!(bridge.try_has_deferred()?, Some(false)), "P/C game contains deferred collection")?;
            let worker = worker.try_lock().map_err(|_| invalid("P/C game worker busy or poisoned"))?;
            check(!worker.has_failure() && !worker.has_unpublished_pump_rows(),
                "P/C game worker retains failure or pending original rows")?;
        }
        check(self.owner.try_status()?.is_some_and(|status|
            status.collection_disabled_first_ticket.is_none() && status.prior_disabled_first_tickets.is_empty()
                && status.incomplete_capture_count == 0 && status.failure.is_none() && !status.callback_lock_failure),
            "P/C game owner retains disabled, incomplete or failed collection")?;
        let (last, _, _) = roots.last().expect("primary root retained");
        let last_admission = last.installed_admission_for_recovery()
            .ok_or_else(|| invalid("P/C last root admission missing"))?.clone();
        loop {
            check(Instant::now() < until, "P/C game root separation deadline reached")?;
            match last.try_separate_root_for_next(&last_admission, self.publication_maximum_records as usize)? {
                PcRootSeparationV2::Pending => std::thread::sleep(Duration::from_millis(1)),
                PcRootSeparationV2::Released { .. } => break,
            }
        }
        let first_admission = self.bridge.installed_admission_for_recovery()
            .ok_or_else(|| invalid("P/C first root admission missing"))?.clone();
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
            &self.publication_manifest, &first_admission, self.publication_maximum_records)?);
        let mut assembly = self.publication_assembly.take().expect("original assembly");
        let result = (|| -> Result<bool, ArenaError> {
            for (bridge, worker, consumer) in &roots {
                loop {
                    check(Instant::now() < until, "P/C game consumer deadline reached")?;
                    let polled = consumer.poll()?;
                    match self.try_offer_consumer_item_for(bridge, worker, consumer, &mut assembly)? {
                        OwnOfferOutcomeV2::Offered => self.advance_game_row(bridge, worker, &mut assembly)?,
                        OwnOfferOutcomeV2::FailedRetained => return Err(invalid("P/C game teacher row failed; original retained")),
                        OwnOfferOutcomeV2::Pending => {
                            let Some((items, failed, _, _)) = consumer.try_counts()? else {
                                std::thread::sleep(Duration::from_millis(1)); continue;
                            };
                            check(failed == 0, "P/C game consumer retains failed rows")?;
                            if items != 0 { std::thread::sleep(Duration::from_millis(1)); continue; }
                            match polled {
                                OwnPumpPollV2::Received | OwnPumpPollV2::Full => return Err(invalid("P/C game consumer item remains pending")),
                                OwnPumpPollV2::Empty | OwnPumpPollV2::Disconnected => break,
                            }
                        }
                    }
                }
                let admission = bridge.installed_admission_for_recovery()
                    .ok_or_else(|| invalid("P/C game original root admission missing"))?;
                loop {
                    check(Instant::now() < until, "P/C separated row deadline reached")?;
                    self.late_rows.try_reserve(1).map_err(|_| invalid("P/C separated row custody capacity"))?;
                    let row = match bridge.try_take_separated_row(admission)? {
                        None => { std::thread::sleep(Duration::from_millis(1)); continue; }
                        Some(None) => break,
                        Some(Some(row)) => row,
                    };
                    self.late_rows.push_back(row);
                    let row = self.late_rows.front().expect("original separated row retained");
                    check(row.submitted_input_seal().teacher_actor_id.is_none() || row.record.0.input.legal_moves.is_empty(),
                        "P/C separated row requires unobserved teacher evidence; original retained")?;
                    let row = self.late_rows.pop_front().expect("original separated row retained");
                    if let Err(original) = assembly.offer(row) {
                        let (error, row) = *original; self.late_rows.push_front(row); return Err(error);
                    }
                    self.advance_game_row(bridge, worker, &mut assembly)?;
                }
                check(matches!(consumer.try_has_failures()?, Some(false)), "P/C game consumer retains failure")?;
            }
            check(assembly.records() > 0 && self.owner.try_status()?.is_some_and(|status|
                status.pending_retired_count == 0 && status.game_reserved_slots == Some(0)
                    && u64::try_from(status.game_retired_total) == Ok(assembly.records())),
                "P/C game owner and assembled records differ")?;
            assembly.flush()?;
            let directory = self.publication_dir.as_ref().expect("original output owner");
            self.publication = Some(assembly.finish()?.publish_atomic(directory, until, &AtomicBool::new(false))?);
            Ok(true)
        })();
        self.publication_assembly = Some(assembly);
        if matches!(&result, Ok(true)) { self.pump_published = true; }
        result
    }

    fn advance_game_row(&self, bridge: &RoventPcCollectionBridgeV2,
        worker: &Arc<Mutex<RoventPcCollectionWorkerV2>>, assembly: &mut RoventCorpusAssemblyLockV2,
    ) -> Result<(), ArenaError> {
        let advanced = match self.try_advance_consumer_assembly_for(bridge, worker, assembly) {
            Err(error) if assembly.chunk_flush_required()? => {
                assembly.flush()?;
                self.try_advance_consumer_assembly_for(bridge, worker, assembly)
                    .map_err(|retry| invalid(format!("P/C game append before flush: {error}; retry: {retry}")))?
            }
            other => other?,
        };
        check(advanced, "P/C game original root commit gate remained busy")
    }
    /// Offer an original pump item only under this root's real joined worker,
    /// handoff and commit gate. It is not a publication receipt.
    pub(super) fn try_offer_consumer_item(
        &self,
        consumer: &super::own_publisher_v2::OwnPumpConsumerV2,
        assembly: &mut RoventCorpusAssemblyLockV2,
    ) -> Result<super::own_publisher_v2::OwnOfferOutcomeV2, ArenaError> {
        if self.worker_exit.is_none() || self.worker_thread_panicked {
            return Ok(super::own_publisher_v2::OwnOfferOutcomeV2::Pending);
        }
        self.try_offer_consumer_item_for(&self.bridge, &self.worker, consumer, assembly)
    }

    fn try_offer_consumer_item_for(
        &self,
        bridge: &RoventPcCollectionBridgeV2,
        worker: &Arc<Mutex<RoventPcCollectionWorkerV2>>,
        consumer: &super::own_publisher_v2::OwnPumpConsumerV2,
        assembly: &mut RoventCorpusAssemblyLockV2,
    ) -> Result<super::own_publisher_v2::OwnOfferOutcomeV2, ArenaError> {
        use super::own_publisher_v2::OwnOfferOutcomeV2;
        let worker = match worker.try_lock() {
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
        let Some(gate) = bridge.try_begin_commit(&admission)? else { return Ok(OwnOfferOutcomeV2::Pending); };
        consumer.try_offer_front_pc(&self.owner, assembly, &gate, &mut handoff)
    }

    pub(super) fn try_advance_consumer_assembly(
        &self,
        assembly: &mut RoventCorpusAssemblyLockV2,
    ) -> Result<bool, ArenaError> {
        if self.worker_exit.is_none() || self.worker_thread_panicked { return Ok(false); }
        self.try_advance_consumer_assembly_for(&self.bridge, &self.worker, assembly)
    }

    fn try_advance_consumer_assembly_for(
        &self,
        bridge: &RoventPcCollectionBridgeV2,
        worker: &Arc<Mutex<RoventPcCollectionWorkerV2>>,
        assembly: &mut RoventCorpusAssemblyLockV2,
    ) -> Result<bool, ArenaError> {
        let worker = match worker.try_lock() {
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
        let Some(gate) = bridge.try_begin_commit(&admission)? else { return Ok(false); };
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
            let stages = worker.pump_stage_counts().map(|v| json!({
                "queued":v.queued, "unlabeled":v.unlabeled, "transition_unknown":v.transition_unknown,
                "pending":v.pending, "completed":v.completed, "sealed":v.sealed,
                "prepared":v.prepared, "failed":v.failed,
            }));
            json!({"primary_failure":failures.primary.map(|failure|failure.kind()),
                "worker_failed":worker.has_failure(),
                "additional_failures":failures.additional.iter().map(|failure|failure.kind()).collect::<Vec<_>>(),
                "omitted_failures":failures.omitted,
                "pending_pump_rows":worker.pending_pump_rows(),
                "pump_stages":stages,
                "teacher_startup_observations_debug":format!("{:?}", worker.teacher_startups()),
                "ready_unlabeled":unlabeled,"ready_prepared":prepared,"ready_failed":failed,"ready_committed":committed,
                "diagnostic_actors":worker.diagnostics().len()})
        });
        let deferred_observation = match self.bridge.try_deferred_observation() {
            Ok(Some(value)) => json!({"status":"available", "detail":value}),
            Ok(None) => json!({"status":"busy"}),
            Err(error) => json!({"status":"failed", "error":error.to_string()}),
        };
        let revoke_observation = match self.bridge.try_revoke_observations() {
            Ok(Some(value)) => json!({"status":"available", "detail":value}),
            Ok(None) => json!({"status":"busy"}),
            Err(error) => json!({"status":"error", "detail":error.to_string()}),
        };
        let secondary_revokes: Vec<_> = self.secondary_slots.iter().map(|slot| {
            match slot.bridge.try_revoke_observations() {
                Ok(Some(value)) => json!({"status":"available", "detail":value}),
                Ok(None) => json!({"status":"busy"}),
                Err(error) => json!({"status":"error", "detail":error.to_string()}),
            }
        }).collect();
        let collection_owner = owner.map(|status| json!({"admitted":status.admitted,"settled":status.settled,
            "pending":status.pending_count,"retired":status.pending_retired_count,
            "game_record_limit":status.game_record_limit,"game_retired_total":status.game_retired_total,
            "game_reserved_slots":status.game_reserved_slots,
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
            "callback_lock_failure":status.callback_lock_failure}));
        let mut observation = json!({"schema":"rz-rovent-pc-lock-v2-run-observation/1",
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
            "publisher_receipt":self.publication});
        let remaining = json!({
            "publication_attempted":self.publication_attempted,
            "secondary_supervisor_present":self.supervisor.is_some(),
            "secondary_slots_recovered":self.secondary_slots.len(),
            "secondary_supervisor_error":self.supervisor_error.as_ref().map(|error| error.to_string()),
            "assembly_records":self.publication_assembly.as_ref().map(|assembly| assembly.records()),
            "published_records":self.publication.as_ref().map(|receipt| receipt.record_count),
            "late_retired_rows_retained":self.late_rows.len(),
            "teacher_worker_observation":worker_observation,
            "deferred_collection":deferred_observation,
            "revoke_observation":revoke_observation, "secondary_revoke_observations":secondary_revokes,
            "collection_owner_observation":owner_observation,
            "collection_owner":collection_owner,
            "physical_completion_claimed":false,"training_executed":false});
        let serde_json::Value::Object(remaining) = remaining else {
            unreachable!("remaining observation is an object");
        };
        observation.as_object_mut().expect("observation is an object").extend(remaining);
        observation
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
            if !self.secondary_slots.is_empty() { std::mem::forget(std::mem::take(&mut self.secondary_slots)); }
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
        RoventCollectionOwnerLockV2::new_shared_with_record_limit(
            Arc::clone(&manifest), plan.metadata.clone(), plan.maximum_records,
        )?;
    let roster_identity: Vec<_> = plan.teacher_slots.iter().map(|entry|
        (entry.actor_id.clone(), entry.source_index, entry.collection_owner)).collect();
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
    // Route by the original admission even for the current one-root
    // profile. Additional router slots remain unavailable until the
    // supervisor proves the preceding root has released shared custody.
    let router = Arc::new(crate::rovent_collect::pc_root_router_v2::RoventPcRootRouterV2::new(
        Arc::clone(&bridge), plan.maximum_roots as usize,
    )?);
    let hooks: Arc<dyn rz_uci::rovent_bootstrap::pc_session::RoventPcCollectionHooksV2> = router.clone();
    let supervisor_custody = Arc::new(Mutex::new(None));
    let admission_supervisor = Arc::clone(&supervisor_custody);
    let supervisor_router = Arc::clone(&router);
    let supervisor_owner = owner.clone();
    let supervisor_manifest = Arc::clone(&manifest);
    let supervisor_roots = plan.maximum_roots as usize;
    let supervisor_records = plan.maximum_records as usize;
    let supervisor_constraint = plan.teacher_constraint;
    let supervisor_quantum = Duration::from_nanos(plan.teacher_soft_quantum_ns);
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
            if supervisor_roots > 1 {
                let mut custody = admission_supervisor.lock().map_err(|_| RoventLaunchError {
                    stage: "collection_supervisor_start", detail: "supervisor custody poisoned".into(),
                })?;
                *custody = Some(PcWorkerSupervisorV2::start(
                    supervisor_router, supervisor_owner, supervisor_manifest, roster_identity,
                    supervisor_constraint, supervisor_quantum, supervisor_roots, supervisor_records,
                    worker_deadline, cleanup_deadline,
                ).map_err(|error| RoventLaunchError { stage: "collection_supervisor_start", detail: error.to_string() })?);
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
        supervisor: supervisor_custody.lock().map_err(|_| invalid("P/C supervisor custody poisoned"))?.take(),
        secondary_slots: Vec::new(),
        supervisor_error: None,
        worker_start,
        worker_exit: None,
        worker_thread_panicked: false,
        native_failure: native.err(),
        native_failure_taken: false,
        resource_error: None,
    };
    if let Some(supervisor) = &run.supervisor { supervisor.request_stop(); }
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
    if let Some(supervisor) = &mut run.supervisor {
        while !supervisor.poll_join() && Instant::now() < cleanup_deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
        if supervisor.poll_join() {
            match supervisor.take_joined_slots() {
                Ok(Some((slots, error))) => {
                    run.secondary_slots = slots;
                    run.supervisor_error = error;
                    run.supervisor = None;
                }
                Ok(None) => {}
                Err(error) => run.supervisor_error = Some(error),
            }
        }
    }
    Ok(run)
}
