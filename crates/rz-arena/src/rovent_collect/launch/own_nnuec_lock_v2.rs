//! Own NNUEC input-lock /2 launch custody. A returned run is an observation, not a
//! published corpus or a claim of native/teacher physical completion.
use super::*;
use crate::rovent_collect::own_hooks_lock_v2::{
    OwnRootSeparationV2, OwnRootWorkerV2, OwnWorkerExitKindV2, OwnWorkerStartErrorV2, RoventNnuecCollectionBridgeV2, RoventNnuecCollectionWorkerExitV2, RoventNnuecCollectionWorkerV2,
};
use crate::rovent_collect::own_worker_supervisor_v2::{OwnSecondarySlotV2, OwnWorkerSupervisorV2};
use crate::rovent_collect::teacher::v2_session::RoventTeacherRosterEntryV2;
use crate::rovent_collect::teacher::lock_v2_policy::CommitTeacherPolicyFailureV2;
use crate::rovent_collect::teacher::lock_v2_row_pump::PumpItemV2;
use rz_experiments::RoventInputLockV2;
use rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2;
use rz_uci::rovent_bootstrap::launch::{
    RoventOwnNnuecCollectionLaunchFailureV2, run_own_nnuec_collection_v2_with_resource_admission,
};
use rz_uci::rovent_bootstrap::nnuec_session::{
    RoventNnuecCollectionHandoffV2, RoventNnuecHandoffTransferErrorV2,
    RoventNnuecTransferredHandoffV2,
};
use rz_uci::rovent_bootstrap::nnuec_session::handoff_escrow_v2::{
    HandoffEscrowErrorV2, HandoffEscrowKeyV2, HandoffEscrowLeaseV2,
};
use std::sync::atomic::{AtomicU8, Ordering};
use std::thread::JoinHandle;
use super::own_publisher_v2::{OwnPumpConsumerV2, OwnOfferOutcomeV2, OwnPumpPollV2};

const WORKER_NOT_STARTED: u8 = 0;
const WORKER_STARTING: u8 = 1;
const WORKER_STARTED: u8 = 2;
const WORKER_START_FAILED: u8 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OwnPumpCommitProgressV2 {
    Pending,
    Empty,
    NotPrepared,
    AlreadyAttached,
    Attached,
}

/// Keeps the original resource, native failure and teacher worker together.
/// Neither the live readback nor the thread exit establishes publication.
pub struct RoventNnuecCollectionRunV2 {
    resource: Arc<Mutex<resources::CollectionResourceCustody>>,
    original_lock: Arc<Mutex<Option<Arc<RoventInputLockV2>>>>,
    owner: RoventCollectionOwnerLockV2,
    bridge: Arc<RoventNnuecCollectionBridgeV2>,
    worker: Arc<Mutex<RoventNnuecCollectionWorkerV2>>,
    worker_thread: Option<JoinHandle<RoventNnuecCollectionWorkerExitV2>>,
    worker_not_started: bool,
    worker_exit: Option<RoventNnuecCollectionWorkerExitV2>,
    bridge_worker_exit: Option<OwnWorkerExitKindV2>,
    worker_thread_panicked: bool,
    worker_custody_poisoned: bool,
    transferred_handoff: Option<RoventNnuecTransferredHandoffV2>,
    recovered_handoff: Option<HandoffEscrowLeaseV2>,
    escrow_recovery_error: Option<ArenaError>,
    handoff_transfer_error: Option<RoventNnuecHandoffTransferErrorV2>,
    commit_failure: Option<Box<CommitTeacherPolicyFailureV2>>,
    publisher: Arc<OwnPumpConsumerV2>,
    publisher_sender: std::sync::mpsc::SyncSender<PumpItemV2>,
    publisher_stop: Arc<AtomicBool>,
    publisher_thread: Option<JoinHandle<Result<(), ArenaError>>>,
    publisher_exit: Option<Result<(), ArenaError>>,
    publisher_unknown: bool,
    publisher_assembly: Option<RoventCorpusAssemblyLockV2>,
    publisher_directory: Option<Dir>,
    publisher_output_parent: Option<Dir>,
    publisher_manifest: Arc<RoventCorpusManifestLockV2>,
    publisher_output: PathBuf,
    publisher_max_records: u64,
    publisher_timeout: Duration,
    publisher_status: &'static str,
    publisher_error: Option<ArenaError>,
    publisher_panic: Option<Box<dyn std::any::Any + Send>>,
    publisher_receipt: Option<rz_experiments::rovent_corpus_lock_v2::publication::RoventLockV2Publication>,
    supervisor: Option<OwnWorkerSupervisorV2>,
    secondary_slots: Vec<OwnSecondarySlotV2>,
    supervisor_error: Option<ArenaError>,
    retained_rows: std::collections::VecDeque<RoventRetiredLockV2Record>,
    completed_workers: Vec<OwnRootWorkerV2>,
    native_failure: Option<RoventOwnNnuecCollectionLaunchFailureV2>,
    native_failure_taken: bool,
    pre_admission_failure_transferred: bool,
    resource_error: Option<String>,
}

impl RoventNnuecCollectionRunV2 {
    fn primary_join_observed(&self) -> bool {
        self.worker_exit.is_some() || self.bridge_worker_exit.is_some()
    }
    /// Assemble each actual root with its own worker, handoff and commit gate.
    /// A partial failure retains the row, worker and assembly in this run.
    fn finish_game_publisher(&mut self, until: Instant) -> Result<(), ArenaError> {
        use crate::rovent_collect::own_worker_supervisor_v2::OwnSecondaryWorkerStartV2;
        check(self.primary_join_observed() && !self.primary_panic_observed()
            && self.publisher_exit.as_ref().is_some_and(Result::is_ok)
            && self.publisher_thread.is_none() && !self.publisher_unknown,
            "Own game primary worker or publisher not joined")?;
        check(self.bridge.returned_handoff_settled().map_err(|error| invalid(format!("Own primary handoff: {error:?}")))?,
            "Own game primary handoff unresolved")?;
        let mut roots = vec![(Arc::clone(&self.bridge), Arc::clone(&self.worker), Arc::clone(&self.publisher), self.publisher_sender.clone())];
        roots.try_reserve(self.secondary_slots.len()).map_err(|_| invalid("Own game root allocation"))?;
        for slot in &self.secondary_slots {
            check(matches!(slot.bridge.try_has_deferred_observations()?, Some(false)),
                "Own game secondary deferred collection retained")?;
            check(slot.bridge.first_pending_event().is_none()
                && slot.bridge.sealed_pending_event().is_none()
                && slot.bridge.before_start_failure().is_none(),
                "Own game secondary pending or failed admission retained")?;
            let pending = slot.bridge.try_pending_stopped_observation()?
                .ok_or_else(|| invalid("Own secondary pending observation busy"))?;
            check(pending.first.is_none() && pending.summarized == 0 && pending.other == 0,
                "Own game secondary pending observations retained")?;
            if slot.bridge.installed_admission_for_recovery().is_none() {
                let unstarted = (slot.worker_join_observed && slot.consumer_drained_after_join
                    && slot.worker_exit_kind == Some(OwnWorkerExitKindV2::NoAdmission))
                    || (slot.worker_start == OwnSecondaryWorkerStartV2::FailedBeforeStart
                        && slot.consumer_empty_before_worker_start);
                check(unstarted, "Own secondary start or admission unresolved")?;
                continue;
            }
            check(slot.worker_join_observed && slot.consumer_drained_after_join
                && matches!(slot.worker_exit_kind, Some(OwnWorkerExitKindV2::Aggregate | OwnWorkerExitKindV2::Retirement))
                && slot.bridge.returned_handoff_settled().map_err(|error| invalid(format!("Own secondary handoff: {error:?}")))?,
                "Own game secondary worker, consumer or handoff unresolved")?;
            roots.push((Arc::clone(&slot.bridge), Arc::clone(&slot.worker), Arc::clone(&slot.consumer), slot.sender.clone()));
        }
        for (bridge, worker, _, _) in &roots {
            check(matches!(bridge.try_has_deferred_observations()?, Some(false)), "Own game deferred collection retained")?;
            let worker = worker.try_lock().map_err(|_| invalid("Own game worker busy or poisoned"))?;
            check(!worker.has_failure(), "Own game worker retains failure")?;
        }
        let (last, _, _, _) = roots.last().expect("primary root retained");
        let last_admission = last.installed_admission_for_recovery().ok_or_else(|| invalid("Own last root admission missing"))?.clone();
        loop {
            check(Instant::now() < until, "Own game root separation deadline")?;
            match last.try_separate_root_for_next(&last_admission, self.publisher_max_records as usize)? {
                OwnRootSeparationV2::Pending => std::thread::sleep(Duration::from_millis(1)),
                OwnRootSeparationV2::Released { .. } => break,
            }
        }
        check(self.native_failure.is_none() && !self.native_failure_taken && self.resource_error.is_none(),
            "Own native or resource failure retained")?;
        check(self.publisher_assembly.is_none() && self.publisher_directory.is_none(),
            "Own previous publication attempt retained")?;
        let first = self.bridge.installed_admission_for_recovery().ok_or_else(|| invalid("Own first root admission missing"))?.clone();
        let parent = self.publisher_output.parent().ok_or_else(|| invalid("Own output parent"))?;
        let leaf = self.publisher_output.file_name().ok_or_else(|| invalid("Own output leaf"))?;
        self.publisher_output_parent = Some(Dir::open_ambient_dir(parent, cap_std::ambient_authority()).map_err(io)?);
        let root = self.publisher_output_parent.as_ref().expect("parent retained");
        root.create_dir(leaf).map_err(io)?;
        self.publisher_directory = Some(root.open_dir_nofollow(leaf).map_err(io)?);
        let directory = self.publisher_directory.as_ref().expect("directory retained");
        directory.create_dir("chunks").map_err(io)?;
        self.publisher_assembly = Some(RoventCorpusAssemblyLockV2::new(
            &self.owner, directory, &self.publisher_manifest, &first, self.publisher_max_records,
        )?);
        let mut assembly = self.publisher_assembly.take().expect("assembly retained");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), ArenaError> {
            for (bridge, worker, consumer, sender) in &roots {
                loop {
                    check(Instant::now() < until, "Own game consumer deadline")?;
                    let sent = worker.try_lock().map_err(|_| invalid("Own game worker busy or poisoned"))?
                        .try_send_first_pump_item(sender)?;
                    let polled = consumer.poll()?;
                    match self.try_offer_consumer_item_for(bridge, worker, consumer, &mut assembly)? {
                        OwnOfferOutcomeV2::Offered => self.advance_game_row(bridge, worker, &mut assembly)?,
                        OwnOfferOutcomeV2::FailedRetained => return Err(invalid("Own game teacher row failed; original retained")),
                        OwnOfferOutcomeV2::Pending => {
                            let Some((items, failed, _, _)) = consumer.try_counts()? else {
                                std::thread::sleep(Duration::from_millis(1)); continue;
                            };
                            check(failed == 0, "Own consumer retains failed rows")?;
                            if items != 0 { std::thread::sleep(Duration::from_millis(1)); continue; }
                            match polled {
                                OwnPumpPollV2::Received | OwnPumpPollV2::Full => return Err(invalid("Own consumer item remains pending")),
                                OwnPumpPollV2::Empty | OwnPumpPollV2::Disconnected
                                    if matches!(sent, crate::rovent_collect::collection_hooks_core_v2::PumpSendOutcomeV2::Empty) => break,
                                OwnPumpPollV2::Empty | OwnPumpPollV2::Disconnected => continue,
                            }
                        }
                    }
                }
                check(worker.try_lock().is_ok_and(|original| !original.has_unpublished_pump_rows()),
                    "Own original worker retains unpublished rows")?;
                let admission = bridge.installed_admission_for_recovery().ok_or_else(|| invalid("Own root admission missing"))?;
                loop {
                    check(Instant::now() < until, "Own separated row deadline")?;
                    self.retained_rows.try_reserve(1).map_err(|_| invalid("Own separated row custody"))?;
                    let row = match bridge.try_take_separated_row(admission)? {
                        None => { std::thread::sleep(Duration::from_millis(1)); continue; }
                        Some(None) => break,
                        Some(Some(row)) => row,
                    };
                    self.retained_rows.push_back(row);
                    let row = self.retained_rows.front().expect("original separated row retained");
                    check(row.submitted_input_seal().teacher_actor_id.is_none() || row.record.0.input.legal_moves.is_empty(),
                        "Own separated row requires teacher evidence; original retained")?;
                    let row = self.retained_rows.pop_front().expect("original separated row retained");
                    if let Err(original) = assembly.offer(row) {
                        let (error, row) = *original; self.retained_rows.push_front(row); return Err(error);
                    }
                    self.advance_game_row(bridge, worker, &mut assembly)?;
                }
                check(matches!(consumer.try_has_failures()?, Some(false)), "Own consumer retains failure")?;
            }
            self.closed_owner()?;
            check(assembly.records() > 0 && self.owner.try_status()?.is_some_and(|status|
                status.pending_retired_count == 0 && status.game_reserved_slots == Some(0)
                    && u64::try_from(status.game_retired_total) == Ok(assembly.records())),
                "Own game owner and assembled records differ")?;
            self.completed_workers.try_reserve(roots.len() + self.secondary_slots.len()).map_err(|_| invalid("Own worker custody capacity"))?;
            for (bridge, _, _, _) in &roots {
                let admission = bridge.installed_admission_for_recovery().ok_or_else(|| invalid("Own original admission missing"))?;
                loop {
                    check(Instant::now() < until, "Own worker transfer deadline")?;
                    match bridge.try_take_separated_worker(admission)? {
                        Some(original) => { self.completed_workers.push(original); break; }
                        None => std::thread::sleep(Duration::from_millis(1)),
                    }
                }
            }
            for slot in &self.secondary_slots {
                if slot.bridge.installed_admission_for_recovery().is_some()
                    || slot.worker_start == OwnSecondaryWorkerStartV2::FailedBeforeStart { continue; }
                loop {
                    check(Instant::now() < until, "Own unadmitted worker transfer deadline")?;
                    match slot.bridge.try_take_unadmitted_worker()? {
                        Some(original) => { self.completed_workers.push(original); break; }
                        None => std::thread::sleep(Duration::from_millis(1)),
                    }
                }
            }
            assembly.flush()?;
            let directory = self.publisher_directory.as_ref().expect("directory retained");
            self.publisher_receipt = Some(assembly.finish()?.publish_atomic(directory, until, &AtomicBool::new(false))?);
            Ok(())
        }));
        self.publisher_assembly = Some(assembly);
        match result {
            Ok(result) => result,
            Err(payload) => {
                self.publisher_panic = Some(payload);
                Err(invalid("Own game publisher panicked; original custody retained"))
            }
        }
    }

    fn advance_game_row(&self, bridge: &RoventNnuecCollectionBridgeV2,
        worker: &Arc<Mutex<RoventNnuecCollectionWorkerV2>>, assembly: &mut RoventCorpusAssemblyLockV2,
    ) -> Result<(), ArenaError> {
        let advanced = match self.try_advance_assembly_for(bridge, worker, assembly) {
            Err(error) if assembly.chunk_flush_required()? => {
                assembly.flush()?;
                self.try_advance_assembly_for(bridge, worker, assembly)
                    .map_err(|retry| invalid(format!("Own append before flush: {error}; retry: {retry}")))?
            }
            other => other?,
        };
        check(advanced, "Own original root commit gate remained busy")
    }

    fn primary_retirement_observed(&self) -> bool {
        matches!(self.worker_exit.as_ref(), Some(RoventNnuecCollectionWorkerExitV2::RetirementAndHandoffObserved))
            || matches!(self.bridge_worker_exit, Some(OwnWorkerExitKindV2::Retirement))
    }

    fn primary_panic_observed(&self) -> bool {
        self.worker_thread_panicked
            || matches!(self.worker_exit.as_ref(), Some(RoventNnuecCollectionWorkerExitV2::Panicked(_)))
            || self.bridge_worker_exit == Some(OwnWorkerExitKindV2::Panicked)
    }
    /// Drain and assemble this run's original rows after both worker joins.
    /// A failure retains the consumer, assembly and output directory in run.
    fn finish_publisher(&mut self, until: Instant) -> Result<(), ArenaError> {
        check(self.supervisor.is_none() && self.supervisor_error.is_none(),
            "Own supervisor custody is unresolved")?;
        if !self.secondary_slots.is_empty() { return self.finish_game_publisher(until); }
        use crate::rovent_collect::collection_hooks_core_v2::PumpSendOutcomeV2;
        check(self.primary_join_observed() && self.publisher_exit.as_ref().is_some_and(Result::is_ok)
            && self.worker_thread.is_none() && self.publisher_thread.is_none()
            && !self.worker_thread_panicked && !self.publisher_unknown && !self.worker_custody_poisoned,
            "Own publisher worker joins not confirmed")?;
        let admission = {
            let worker = self.worker.try_lock().map_err(|_| invalid("Own publisher worker busy"))?;
            let handoff = worker.original_handoff().ok_or_else(|| invalid("Own original handoff missing"))?;
            let mut handoff = handoff.try_lock().map_err(|_| invalid("Own handoff busy"))?;
            let admission = handoff.parts_mut().0.clone();
            check(handoff.teacher_aggregate_ack().is_some(), "Own teacher aggregate not observed")?;
            admission
        };
        if self.publisher_assembly.is_none() {
            let parent = self.publisher_output.parent().ok_or_else(|| invalid("Own output parent"))?;
            let leaf = self.publisher_output.file_name().ok_or_else(|| invalid("Own output leaf"))?;
            self.publisher_output_parent = Some(Dir::open_ambient_dir(parent, cap_std::ambient_authority()).map_err(io)?);
            let original_parent = self.publisher_output_parent.as_ref().expect("parent retained");
            original_parent.create_dir(leaf).map_err(io)?;
            self.publisher_directory = Some(original_parent.open_dir_nofollow(leaf).map_err(io)?);
            let directory = self.publisher_directory.as_ref().expect("directory retained");
            directory.create_dir("chunks").map_err(io)?;
            self.publisher_assembly = Some(RoventCorpusAssemblyLockV2::new(
                &self.owner, directory, &self.publisher_manifest, &admission, self.publisher_max_records,
            )?);
        }
        let mut assembly = self.publisher_assembly.take().expect("publisher assembly installed");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), ArenaError> {
            loop {
                check(Instant::now() < until, "Own publisher deadline elapsed")?;
                let sent = self.worker.try_lock().map_err(|_| invalid("Own publisher worker busy"))?
                    .try_send_first_pump_item(&self.publisher_sender)?;
                let polled = self.publisher.poll()?;
                let counts = self.publisher.try_counts()?.ok_or_else(|| invalid("Own publisher custody busy"))?;
                if counts.0 == 0 {
                    if matches!(sent, PumpSendOutcomeV2::Empty) && matches!(polled, OwnPumpPollV2::Empty | OwnPumpPollV2::Disconnected) { break; }
                    if matches!(polled, OwnPumpPollV2::Full) { return Err(invalid("Own publisher custody full")); }
                    continue;
                }
                match self.try_offer_consumer_item(&self.publisher, &mut assembly)? {
                    OwnOfferOutcomeV2::Pending => return Err(invalid("Own publisher root gate busy")),
                    OwnOfferOutcomeV2::FailedRetained => continue,
                    OwnOfferOutcomeV2::Offered => {}
                }
                match self.try_advance_assembly(&mut assembly) {
                    Ok(true) => {},
                    Ok(false) => return Err(invalid("Own publisher root gate busy")),
                    Err(original) => {
                        if !assembly.chunk_flush_required()? { return Err(original); }
                        assembly.flush()?;
                        if !self.try_advance_assembly(&mut assembly)? { return Err(invalid("Own publisher root gate busy")); }
                    }
                }
            }
            check(!self.publisher.try_has_failures()?.ok_or_else(|| invalid("Own publisher custody busy"))?, "Own collection has retained failed rows")?;
            self.closed_owner()?;
            check(self.native_failure.is_none() && !self.native_failure_taken && self.resource_error.is_none(), "Own native/resource failure retained")?;
            assembly.flush()?;
            let directory = self.publisher_directory.as_ref().ok_or_else(|| invalid("Own output directory missing"))?;
            let cancelled = AtomicBool::new(false);
            self.publisher_receipt = Some(assembly.finish()?.publish_atomic(directory, until, &cancelled)?);
            Ok(())
        }));
        self.publisher_assembly = Some(assembly);
        match result {
            Ok(result) => result,
            Err(payload) => { self.publisher_panic = Some(payload); Err(invalid("Own publisher panicked; original custody retained")) }
        }
    }
    /// Move one original consumer item to assembly only under the exact
    /// original root gate and handoff. This is not a publication receipt.
    pub(super) fn try_offer_consumer_item(
        &self,
        consumer: &super::own_publisher_v2::OwnPumpConsumerV2,
        assembly: &mut RoventCorpusAssemblyLockV2,
    ) -> Result<super::own_publisher_v2::OwnOfferOutcomeV2, ArenaError> {
        use super::own_publisher_v2::OwnOfferOutcomeV2;
        if self.worker_thread.is_some() || !self.primary_join_observed()
            || self.worker_thread_panicked || self.worker_custody_poisoned
            || self.primary_panic_observed()
        { return Ok(OwnOfferOutcomeV2::Pending); }
        self.try_offer_consumer_item_for(&self.bridge, &self.worker, consumer, assembly)
    }

    /// The caller must first establish that this exact root's worker joined.
    /// The gate and handoff are taken from that same root, not the newest root.
    fn try_offer_consumer_item_for(
        &self,
        bridge: &RoventNnuecCollectionBridgeV2,
        worker: &Arc<Mutex<RoventNnuecCollectionWorkerV2>>,
        consumer: &OwnPumpConsumerV2,
        assembly: &mut RoventCorpusAssemblyLockV2,
    ) -> Result<OwnOfferOutcomeV2, ArenaError> {
        let worker = match worker.try_lock() {
            Ok(worker) => worker,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnOfferOutcomeV2::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("Collection /2 worker poisoned")),
        };
        let Some(handoff) = worker.original_handoff() else { return Ok(OwnOfferOutcomeV2::Pending); };
        let mut handoff = match handoff.try_lock() {
            Ok(handoff) => handoff,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnOfferOutcomeV2::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("Collection /2 handoff poisoned")),
        };
        let admission = handoff.parts_mut().0.clone();
        let Some(gate) = bridge.try_begin_commit(&admission)? else { return Ok(OwnOfferOutcomeV2::Pending); };
        consumer.try_offer_front(&self.owner, assembly, &gate, &mut handoff)
    }
    /// Advance a caller-owned assembly with this run's original root gate and
    /// handoff. The assembly keeps the original row on busy/error; success is
    /// only in-memory assembly, not corpus publication or teacher retirement.
    pub fn try_advance_assembly(&self, assembly: &mut RoventCorpusAssemblyLockV2) -> Result<bool, ArenaError> {
        if self.worker_thread.is_some() || !self.primary_join_observed()
            || self.worker_thread_panicked || self.worker_custody_poisoned
            || self.primary_panic_observed()
        { return Ok(false); }
        self.try_advance_assembly_for(&self.bridge, &self.worker, assembly)
    }

    fn try_advance_assembly_for(
        &self,
        bridge: &RoventNnuecCollectionBridgeV2,
        worker: &Arc<Mutex<RoventNnuecCollectionWorkerV2>>,
        assembly: &mut RoventCorpusAssemblyLockV2,
    ) -> Result<bool, ArenaError> {
        let worker = match worker.try_lock() {
            Ok(worker) => worker,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("Collection /2 worker poisoned")),
        };
        let Some(handoff) = worker.original_handoff() else { return Ok(false); };
        let mut handoff = match handoff.try_lock() {
            Ok(handoff) => handoff,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("Collection /2 handoff poisoned")),
        };
        let admission = handoff.parts_mut().0.clone();
        let Some(gate) = bridge.try_begin_commit(&admission)? else { return Ok(false); };
        assembly.try_advance_fenced_own(&gate, &mut handoff)
    }
    /// Recover only the original installed admission after the Arena worker
    /// has stopped. The lease preserves custody and proves no completion.
    pub fn try_recover_original_handoff(&mut self) -> Result<bool, ArenaError> {
        if self.recovered_handoff.is_some() { return Ok(true); }
        if self.worker_thread.is_some() || self.worker_custody_poisoned
            || (!self.primary_join_observed() && !self.worker_not_started && !self.worker_thread_panicked)
        {
            return Ok(false);
        }
        let Some(admission) = self.bridge.installed_admission_for_recovery() else { return Ok(false); };
        let lease = match HandoffEscrowKeyV2::try_recover_for_admission(admission) {
            Ok(Some(lease)) => lease,
            Ok(None) | Err(HandoffEscrowErrorV2::Busy | HandoffEscrowErrorV2::AlreadyLeased) => return Ok(false),
            Err(error) => return Err(invalid(format!("Collection /2 escrow recovery: {error:?}"))),
        };
        let delivery_observed = match lease.original().try_lock() {
            Ok(mut handoff) => {
                if !handoff.parts_mut().0.same_original_admission(admission) {
                    return Err(invalid("Collection /2 recovered handoff admission mismatch"));
                }
                handoff.handoff_delivery().is_some()
            }
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("Collection /2 recovered handoff poisoned")),
        };
        if !delivery_observed { return Ok(false); }
        self.recovered_handoff = Some(lease);
        Ok(true)
    }

    /// Transfer the original recovery error for a caller to retain. It is a
    /// collection observation, not a native or teacher physical failure.
    pub fn take_escrow_recovery_error(&mut self) -> Option<ArenaError> {
        self.escrow_recovery_error.take()
    }

    /// Transfer the original UCI failure and resource observation only for a
    /// proven no-thread, no-storage, pre-resource-admission failure. The
    /// caller must retain the original failure (including hook and observer)
    /// and the observation. This is not a successful run or a receipt.
    pub fn try_take_pre_admission_failure(
        &mut self,
    ) -> Result<Option<(RoventOwnNnuecCollectionLaunchFailureV2, serde_json::Value)>, ArenaError> {
        if self.pre_admission_failure_transferred
            || self.native_failure_taken
            || !self.worker_not_started
            || self.worker_custody_poisoned
            || self.worker_thread.is_some()
            || self.worker_thread_panicked
            || self.resource_error.is_some()
        {
            return Ok(None);
        }
        let Some(failure) = self.native_failure.as_ref() else { return Ok(None); };
        if failure.physical_completion_unknown()
            || failure.parsed_storage_selected() != Some(false)
            || !failure.settled_observer_available()
            || failure.observer_custody_retained()
        {
            return Ok(None);
        }
        let resource = match self.resource.try_lock() {
            Ok(resource) => resource,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("Collection /2 original resource custody poisoned")),
        };
        if !resource.no_admission_started() {
            return Ok(None);
        }
        let observation = resource.observation();
        drop(resource);
        let original = self.native_failure.take().expect("checked original failure");
        self.native_failure_taken = true;
        self.pre_admission_failure_transferred = true;
        Ok(Some((original, observation)))
    }
    /// Attempt one in-memory policy attachment after the original worker has
    /// joined. Pending/other items stay in the original worker. This operation
    /// does not publish rows or assert teacher/process physical completion.
    /// Boundary errors are transferred to the caller; the original row and
    /// prepared evidence remain in the worker. Policy-commit failures are
    /// instead retained by this run, including any panic payload.
    pub fn try_commit_pump_head(&mut self) -> Result<OwnPumpCommitProgressV2, ArenaError> {
        if self.primary_panic_observed() {
            return Err(invalid("original row worker panicked; exit payload remains retained"));
        }
        if self.worker_thread_panicked || self.worker_custody_poisoned {
            return Err(invalid("original row worker completion is unknown"));
        }
        if self.worker_thread.is_some() || !self.primary_join_observed() {
            return Ok(OwnPumpCommitProgressV2::Pending);
        }
        if self.commit_failure.is_some() {
            return Err(invalid("original row commit failure remains retained"));
        }
        let mut worker = match self.worker.try_lock() {
            Ok(worker) => worker,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPumpCommitProgressV2::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("original row worker poisoned")),
        };
        let Some(handoff) = worker.original_handoff() else { return Ok(OwnPumpCommitProgressV2::Pending); };
        let mut handoff = match handoff.try_lock() {
            Ok(handoff) => handoff,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnPumpCommitProgressV2::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("original row handoff poisoned")),
        };
        let admission = handoff.parts_mut().0.clone();
        let Some(gate) = self.bridge.try_begin_commit(&admission)? else { return Ok(OwnPumpCommitProgressV2::Pending); };
        let Some(item) = worker.first_pump_item_mut() else { return Ok(OwnPumpCommitProgressV2::Empty); };
        let PumpItemV2::Prepared { row, evidence, sealed } = item else {
            return Ok(OwnPumpCommitProgressV2::NotPrepared);
        };
        if sealed.is_some() { return Ok(OwnPumpCommitProgressV2::AlreadyAttached); }
        let result = self.owner.try_with_fenced_own_producer_mut(row, &gate, &mut handoff,
            |producer, row| producer.attach_prepared_teacher_policy_v2(row, evidence))?;
        match result {
            None => Ok(OwnPumpCommitProgressV2::Pending),
            Some(Ok(original_sealed)) => { *sealed = Some(original_sealed); Ok(OwnPumpCommitProgressV2::Attached) }
            Some(Err(original)) => {
                self.commit_failure = Some(original);
                Err(invalid("original row policy commit failed and is retained"))
            }
        }
    }
    /// Recover the session's original handoff only after joining this run's
    /// original bridge worker. This is custody, not publication or a teacher
    /// semantic-success receipt. The same run keeps the worker and rows.
    pub fn try_transfer_joined_handoff(
        &mut self,
    ) -> Result<bool, RoventNnuecHandoffTransferErrorV2> {
        if self.transferred_handoff.is_some() {
            return Ok(true);
        }
        if self.worker_thread.is_some() || !self.primary_join_observed() || self.primary_panic_observed() || self.worker_custody_poisoned {
            return Ok(false);
        }
        let handoff = {
            let worker = self.worker.try_lock().map_err(|error| match error {
                std::sync::TryLockError::WouldBlock => RoventNnuecHandoffTransferErrorV2::Busy,
                std::sync::TryLockError::Poisoned(_) => RoventNnuecHandoffTransferErrorV2::Poisoned,
            })?;
            let Some(handoff) = worker.original_handoff() else { return Ok(false); };
            handoff
        };
        let admission = {
            let mut original = handoff.try_lock().map_err(|error| match error {
                std::sync::TryLockError::WouldBlock => RoventNnuecHandoffTransferErrorV2::Busy,
                std::sync::TryLockError::Poisoned(_) => RoventNnuecHandoffTransferErrorV2::Poisoned,
            })?;
            original.parts_mut().0.clone()
        };
        let custody = RoventNnuecCollectionHandoffV2::try_transfer_session_custody(&handoff, &admission)?;
        self.transferred_handoff = Some(custody);
        self.handoff_transfer_error = None;
        Ok(true)
    }

    /// Collect only a thread that has already exited; never wait indefinitely
    /// or infer that a still-running worker has physically stopped.
    pub fn poll_worker(&mut self) -> bool {
        if self.worker_thread.is_none() && !self.worker_not_started
            && self.worker_exit.is_none() && self.bridge_worker_exit.is_none()
            && !self.worker_custody_poisoned
        {
            match self.bridge.poll_worker() {
                Ok(Some(kind)) => self.bridge_worker_exit = Some(kind),
                Ok(None) => {},
                Err(error) => { self.resource_error.get_or_insert_with(|| format!("teacher worker custody: {error:?}")); }
            }
        }
        if self
            .worker_thread
            .as_ref()
            .is_some_and(JoinHandle::is_finished)
        {
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
        if self.primary_join_observed() && self.transferred_handoff.is_none() {
            if let Err(error) = self.try_transfer_joined_handoff() {
                self.handoff_transfer_error = Some(error);
            }
        }
        if (self.primary_join_observed() || self.worker_thread_panicked) && self.transferred_handoff.is_none()
            && self.recovered_handoff.is_none() && self.escrow_recovery_error.is_none()
        {
            if let Err(error) = self.try_recover_original_handoff() {
                self.escrow_recovery_error = Some(error);
            }
        }
        if self.primary_join_observed() || self.worker_thread_panicked || self.worker_not_started {
            self.publisher_stop.store(true, Ordering::Release);
        }
        if self.publisher_thread.as_ref().is_some_and(JoinHandle::is_finished) {
            let handle = self.publisher_thread.take().expect("finished publisher retained");
            match handle.join() {
                Ok(result) => self.publisher_exit = Some(result),
                Err(payload) => { self.publisher_unknown = true; std::mem::forget(payload); }
            }
        }
        self.primary_join_observed() || self.worker_thread_panicked || self.worker_not_started
            || (self.worker_custody_poisoned && self.worker_thread.is_none())
    }

    pub fn observation(&self) -> serde_json::Value {
        let recovered_delivery = self.recovered_handoff.as_ref().map(|lease| {
            match lease.original().try_lock() {
                Ok(handoff) => match handoff.handoff_delivery() {
                    None => json!({"status":"pending"}),
                    Some(Ok(())) => json!({"status":"delivered"}),
                    Some(Err(error)) => json!({"status":"failed","error":format!("{error:?}")}),
                },
                Err(std::sync::TryLockError::WouldBlock) => json!({"status":"busy"}),
                Err(std::sync::TryLockError::Poisoned(_)) => json!({"status":"poisoned"}),
            }
        });
        let (pending_observation, pending_stopped) = match self.bridge.try_pending_stopped_observation() {
            Ok(Some(summary)) => ("available", Some(summary)),
            Ok(None) => ("busy", None),
            Err(_) => ("poisoned", None),
        };
        let (deferred_observation, deferred_roots) = match self.bridge.try_deferred_observations() {
            Ok(Some(roots)) => ("available", Some(roots)),
            Ok(None) => ("busy", None),
            Err(_) => ("poisoned", None),
        };
        let (resource_observation, resource) = match self.resource.try_lock() {
            Ok(state) => ("available", Some(state.observation())),
            Err(std::sync::TryLockError::WouldBlock) => ("busy", None),
            Err(std::sync::TryLockError::Poisoned(_)) => ("poisoned", None),
        };
        let (owner_observation, owner) = match self.owner.try_status() {
            Ok(Some(status)) => ("available", Some(status)),
            Ok(None) => ("busy", None),
            Err(_) => ("poisoned", None),
        };
        let worker_exit = match self.worker_exit.as_ref() {
            Some(RoventNnuecCollectionWorkerExitV2::NoAdmission) => "no_admission",
            Some(RoventNnuecCollectionWorkerExitV2::RetirementAndHandoffObserved) => {
                "retirement_and_handoff_observed"
            }
            Some(RoventNnuecCollectionWorkerExitV2::Deadline) => "deadline",
            Some(RoventNnuecCollectionWorkerExitV2::AdmissionCanceled) => "admission_canceled",
            Some(RoventNnuecCollectionWorkerExitV2::PhysicalAckDelivered) => {
                "physical_scope_ack_delivered"
            }
            Some(RoventNnuecCollectionWorkerExitV2::AggregateHandoffObserved) => {
                "physical_aggregate_handoff_observed"
            }
            Some(RoventNnuecCollectionWorkerExitV2::Panicked(_)) => "caught_panic",
            None if self.bridge_worker_exit == Some(OwnWorkerExitKindV2::NoAdmission) => "no_admission",
            None if self.bridge_worker_exit == Some(OwnWorkerExitKindV2::Retirement) => "retirement_and_handoff_observed",
            None if self.bridge_worker_exit == Some(OwnWorkerExitKindV2::Deadline) => "deadline",
            None if self.bridge_worker_exit == Some(OwnWorkerExitKindV2::AdmissionCanceled) => "admission_canceled",
            None if self.bridge_worker_exit == Some(OwnWorkerExitKindV2::PhysicalAck) => "physical_scope_ack_delivered",
            None if self.bridge_worker_exit == Some(OwnWorkerExitKindV2::Aggregate) => "physical_aggregate_handoff_observed",
            None if self.bridge_worker_exit == Some(OwnWorkerExitKindV2::Panicked) => "caught_panic",
            None if self.worker_not_started => "not_started",
            None if self.worker_custody_poisoned => "custody_poisoned",
            None if self.worker_thread_panicked => "outer_panic",
            None => "pending",
        };
        let worker_observation = self.worker.try_lock().ok().map(|worker| {
            let failures = worker.failure_observation();
            let (pump_unlabeled, pump_prepared, pump_committed, pump_failed) = worker.pump_ready_counts();
            let handoff = match worker.try_handoff_observation(|handoff| {
                let delivery = match handoff.handoff_delivery() {
                    None => json!({"status":"pending"}),
                    Some(Ok(())) => json!({"status":"delivered"}),
                    Some(Err(error)) => json!({"status":"failed","error":format!("{error:?}")}),
                };
                (handoff.collection_failures().iter().map(|error| format!("{error:?}")).collect::<Vec<_>>(), delivery)
            }) {
                Ok((failures,delivery)) => json!({"status":"available","incomplete":!failures.is_empty() || delivery["status"] != "delivered","failures":failures,"delivery":delivery}),
                Err(error) => json!({"status":format!("{error:?}"),"incomplete":"unknown","failures":null}),
            };
            json!({"primary_failure":failures.primary.map(|failure|failure.kind()),
                "additional_failures":failures.additional.iter().map(|failure|failure.kind()).collect::<Vec<_>>(),
                "omitted_failures":failures.omitted,
                "diagnostic_actors":worker.diagnostics().len(),"handoff_collection":handoff,
                "pump_unpublished_rows":worker.pending_pump_rows(),
                "pump_ready_unlabeled":pump_unlabeled,"pump_ready_prepared":pump_prepared,"pump_ready_committed":pump_committed,"pump_ready_failed":pump_failed})
        });
        let owner_value = owner.map(|status| {
            let mut value = json!({"admitted":status.admitted,"settled":status.settled,
                "pending":status.pending_count,"retired":status.pending_retired_count,
                "game_record_limit":status.game_record_limit,"retired_outbox_capacity":status.retired_outbox_capacity,"game_retired_total":status.game_retired_total,
                "game_reserved_slots":status.game_reserved_slots,
                "collection_incomplete_due_to_backpressure":status.collection_disabled_first_ticket.is_some() || !status.prior_disabled_first_tickets.is_empty(),
                "first_omitted_ticket":status.collection_disabled_first_ticket.as_ref().map(ticket_json),
                "disable_reason":status.collection_disabled_reason.map(|reason|format!("{reason:?}"))});
            let rest = json!({"prior_disabled_first_tickets":status.prior_disabled_first_tickets.iter().map(ticket_json).collect::<Vec<_>>(),
                "prior_disabled_reasons":status.prior_disabled_reasons.iter().map(|reason|format!("{reason:?}")).collect::<Vec<_>>(),
                "incomplete_capture":status.incomplete_capture_count,"failed_captures":status.retained_failed_captures,
                "failed_staged_inputs":status.retained_failed_staged,"pending_disabled_markers":status.pending_disabled_markers,
                "failure":status.failure.map(|error|format!("{error:?}")),"callback_lock_failure":status.callback_lock_failure});
            let serde_json::Value::Object(rest) = rest else { unreachable!("owner status is an object"); };
            value.as_object_mut().expect("owner status is an object").extend(rest);
            value
        });
        let mut observation = json!({"schema":"rz-rovent-own-nnuec-lock-v2-run-observation/1",
            "native_error":self.native_failure.as_ref().map(|error|error.error.to_string()),
            "native_failure_transferred":self.native_failure_taken,
            "resource_error":self.resource_error,"resource_observation":resource_observation,"resource":resource,
            "teacher_worker_returned":self.primary_join_observed(),
            "teacher_worker_exit":worker_exit,
            "teacher_worker_running_or_unknown":!self.worker_not_started && (self.worker_thread.is_some() || self.worker_thread_panicked || self.worker_custody_poisoned
                || !self.primary_retirement_observed()),
            "teacher_worker_thread_panicked":self.primary_panic_observed(),
            "teacher_worker_custody_poisoned":self.worker_custody_poisoned,
            "recovered_handoff_lease_retained":self.recovered_handoff.is_some(),
            "recovered_handoff_delivery":recovered_delivery,
            "escrow_recovery_error":self.escrow_recovery_error.as_ref().map(|error|error.to_string()),
            "escrow_recovery_state":if self.transferred_handoff.is_some() { "transferred" }
                else if self.recovered_handoff.is_some() { "recovered" }
                else if self.escrow_recovery_error.is_some() { "failed" }
                else if self.bridge.installed_admission_for_recovery().is_some() { "pending" }
                else { "no_installed_admission_observed" },
            "publisher_status":self.publisher_status,
            "publisher_joined":self.publisher_exit.is_some(),
            "publisher_worker_error":self.publisher_exit.as_ref().and_then(|result| result.as_ref().err()).map(|error| error.to_string()),
            "publisher_unknown":self.publisher_unknown,
            "publisher_error":self.publisher_error.as_ref().map(|error| error.to_string()),
            "publisher_receipt":self.publisher_receipt,
            "teacher_worker_observation":worker_observation,
            "collection_owner_observation":owner_observation,
            "collection_owner":owner_value
        });
        let remaining = json!({
            "deferred_collection_roots_observation":deferred_observation,
            "pending_stopped_observation":pending_observation,
            "before_start_failure":self.bridge.before_start_failure().map(|error|format!("{error:?}")),
            "sealed_pending":self.bridge.sealed_pending_event().map(|event|json!({
                "admission_sha256":event.admission.raw_sha256(),
                "cause":event.cause.as_ref().map(|cause|format!("{cause:?}")),
                "reason":format!("{:?}",event.reason)
            })),
            "first_pending_original":self.bridge.first_pending_event().map(|event|json!({
                "admission_sha256":event.admission.raw_sha256(),
                "cause":event.cause.as_ref().map(|cause|format!("{cause:?}")),
                "reason":format!("{:?}",event.reason)
            })),
            "pending_stopped":pending_stopped.map(|summary| json!({
                "first":summary.first.map(|event| json!({
                    "admission_sha256":event.admission.raw_sha256(),
                    "cause":event.cause.as_ref().map(|cause| format!("{cause:?}")),
                    "reason":format!("{:?}",event.reason),
                })),
                "summarized":summary.summarized,
                "reasons":summary.reasons.into_iter().map(|(cause, reason, count)| json!({
                    "cause":cause.map(|cause| format!("{cause:?}")),
                    "reason":format!("{reason:?}"), "count":count,
                })).collect::<Vec<_>>(),
                "other":summary.other,
            })),
            "deferred_collection_roots":deferred_roots.map(|(roots, summarized, first_summarized, reasons, other_reasons)| json!({
                "retained":roots.into_iter().map(|(admission, reason)| json!({"admission_sha256":admission,"reason":format!("{reason:?}")})).collect::<Vec<_>>(),
                "summarized_count":summarized,
                "first_summarized_admission_sha256":first_summarized,
                "summarized_by_reason":reasons.into_iter().map(|(reason,count)| json!({"stage":format!("{:?}",reason.stage),"code":reason.code,"detail":reason.detail,"count":count})).collect::<Vec<_>>(),
                "other_summarized_reasons":other_reasons,
            })),
            "uci_handoff_custody":if self.transferred_handoff.is_some() { "leased_to_run" } else if self.recovered_handoff.is_some() { "recovered_from_escrow" } else { "unknown" },
            "uci_handoff_registry_released":false,
            "uci_handoff_transfer_error":self.handoff_transfer_error.map(|error|format!("{error:?}")),
            "physical_completion_claimed":false,"corpus_published":self.publisher_status == "published","training_executed":false});
        let serde_json::Value::Object(remaining) = remaining else {
            unreachable!("remaining observation is an object");
        };
        observation.as_object_mut().expect("observation is an object").extend(remaining);
        // This bounded projection does not consume the original rows or prove
        // that an unknown sender has drained. Preserve busy/error explicitly.
        let publisher_failures = match self.publisher.try_failure_observations() {
            Ok(Some(value)) => json!({"status":"observed", "failures":value}),
            Ok(None) => json!({"status":"busy"}),
            Err(error) => json!({"status":"error", "detail":error.to_string()}),
        };
        observation.as_object_mut().expect("observation is an object")
            .insert("publisher_failure_observations".into(), publisher_failures);
        observation
    }

    /// Move the original native failure to a caller that will retain/recover
    /// it. Taking it never changes the original run into a success.
    pub fn take_native_failure(&mut self) -> Option<RoventOwnNnuecCollectionLaunchFailureV2> {
        let failure = self.native_failure.take();
        if failure.is_some() {
            self.native_failure_taken = true;
        }
        failure
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
            self.worker_thread.is_none() && self.primary_join_observed()
                && !self.worker_thread_panicked && !self.worker_custody_poisoned,
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

    /// The original handoff retains every Collection failure. Failure or a
    /// temporarily unavailable observation is never evidence of GPU failure.
    pub fn collection_failures(&self) -> Result<Vec<AdaptiveCollectionError>, ArenaError> {
        let worker = self
            .worker
            .try_lock()
            .map_err(|_| invalid("Collection /2 worker busy or poisoned"))?;
        worker
            .try_handoff_observation(|handoff| handoff.collection_failures().to_vec())
            .map_err(|error| invalid(format!("Collection /2 handoff observation: {error:?}")))
    }

    /// Give the caller the same collection owner only after its original
    /// worker observed the same fenced handoff and teacher retirement. This
    /// authorizes inspection/partial assembly, not final publication.
    pub fn closed_owner(&self) -> Result<RoventCollectionOwnerLockV2, ArenaError> {
        check(self.bridge.first_pending_event().is_none(), "Collection /2 original pending event retained")?;
        check(self.bridge.sealed_pending_event().is_none(), "Collection /2 sealed pending event retained")?;
        check(self.bridge.before_start_failure().is_none(), "Collection /2 before-start failure retained")?;
        check(self.bridge.try_has_deferred_observations()? == Some(false),
            "Collection /2 deferred observations retained or busy")?;
        let pending = self.bridge.try_pending_stopped_observation()?
            .ok_or_else(|| invalid("Collection /2 pending observation busy"))?;
        check(pending.first.is_none() && pending.summarized == 0 && pending.other == 0,
            "Collection /2 retained pending collection observations")?;
        let worker = self.worker.try_lock().map_err(|_| invalid("Collection /2 worker busy or poisoned"))?;
        let delivered = worker.try_handoff_observation(|handoff| matches!(handoff.handoff_delivery(), Some(Ok(()))))
            .map_err(|error| invalid(format!("Collection /2 handoff delivery observation: {error:?}")))?;
        drop(worker);
        check(delivered, "Collection /2 handoff delivery pending or failed")?;
        check(
            self.collection_failures()?.is_empty(),
            "Collection /2 has incomplete handoff observations",
        )?;
        check(
            self.primary_retirement_observed() && self
                .worker
                .try_lock()
                .is_ok_and(|worker| !worker.has_failure() && !worker.has_unpublished_pump_rows()),
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
                && status.pending_retired_count == 0
                && status.collection_disabled_first_ticket.is_none()
                && status.prior_disabled_first_tickets.is_empty()
                && status.incomplete_capture_count == 0
                && status.retained_failed_captures == 0
                && status.retained_failed_staged == 0
                && status.pending_disabled_markers == Some(0)
                && status.failure.is_none()
                && !status.callback_lock_failure,
            "Collection /2 native observer is not settled",
        )?;
        Ok(self.owner.clone())
    }
}

impl Drop for RoventNnuecCollectionRunV2 {
    fn drop(&mut self) {
        // Only the explicit ownership transfer above releases an unstarted
        // outer owner. The resource evidence directory is never deleted.
        if self.pre_admission_failure_transferred {
            return;
        }
        let owner_resolved = self
            .owner
            .try_status()
            .ok()
            .flatten()
            .is_some_and(|status| {
                status.admitted
                    && status.settled
                    && status.pending_count == 0
                    && status.collection_disabled_first_ticket.is_none()
                    && status.prior_disabled_first_tickets.is_empty()
                    && status.pending_retired_count == 0
                    && status.incomplete_capture_count == 0
                    && status.retained_failed_captures == 0
                    && status.retained_failed_staged == 0
                    && status.pending_disabled_markers == Some(0)
                    && status.failure.is_none()
                    && !status.callback_lock_failure
            });
        // This only decides whether the Arena-owned worker can be dropped.
        // UCI's independent retained handoff slot remains unknown.
        let arena_worker_resolved = owner_resolved
            && self.bridge.first_pending_event().is_none()
            && self.bridge.sealed_pending_event().is_none()
            && self.bridge.before_start_failure().is_none()
            && self.bridge.try_has_deferred_observations().is_ok_and(|observation| observation == Some(false))
            && self.bridge.try_pending_stopped_observation().is_ok_and(|observation|
                observation.is_some_and(|summary| summary.first.is_none() && summary.summarized == 0 && summary.other == 0))
            && self.native_failure.is_none()
            && !self.native_failure_taken
            && self.resource_error.is_none()
            && !self.worker_custody_poisoned
            && self.commit_failure.is_none()
            && self.escrow_recovery_error.is_none()
            && self.publisher_status == "published"
            && self.publisher_panic.is_none()
            && self
                .collection_failures()
                .is_ok_and(|failures| failures.is_empty())
            && self
                .worker
                .try_lock()
                .is_ok_and(|worker| !worker.has_failure() && !worker.has_unpublished_pump_rows())
            && self.primary_retirement_observed();
        if !arena_worker_resolved {
            // A dropped observation must not release a still-running or
            // unverified owner. Keep the exact objects for process lifetime;
            // this is quarantine, not a physical-completion receipt.
            std::mem::forget(Arc::clone(&self.resource));
            std::mem::forget(Arc::clone(&self.original_lock));
            std::mem::forget(self.owner.clone());
            std::mem::forget(Arc::clone(&self.bridge));
            std::mem::forget(Arc::clone(&self.worker));
            std::mem::forget(Arc::clone(&self.publisher));
            if let Some(supervisor) = self.supervisor.take() { std::mem::forget(supervisor); }
            if !self.secondary_slots.is_empty() { std::mem::forget(std::mem::take(&mut self.secondary_slots)); }
            if !self.completed_workers.is_empty() { std::mem::forget(std::mem::take(&mut self.completed_workers)); }
            if !self.retained_rows.is_empty() { std::mem::forget(std::mem::take(&mut self.retained_rows)); }
            if let Some(assembly) = self.publisher_assembly.take() { std::mem::forget(assembly); }
            if let Some(directory) = self.publisher_directory.take() { std::mem::forget(directory); }
            if let Some(parent) = self.publisher_output_parent.take() { std::mem::forget(parent); }
            if let Some(payload) = self.publisher_panic.take() { std::mem::forget(payload); }
            if let Some(handle) = self.publisher_thread.take() { std::mem::forget(handle); }
            if let Some(original) = self.commit_failure.take() {
                std::mem::forget(original);
            }
            if let Some(handle) = self.worker_thread.take() {
                std::mem::forget(handle);
            }
            if let Some(exit) = self.worker_exit.take() {
                std::mem::forget(exit);
            }
        }
    }
}

/// Load an explicitly pinned /2 manifest and start the actual Own NNUEC UCI path.
/// The plan's single-root limit is the bridge's current capability; it must
/// not be reported as multi-root Collection support.
pub fn run_rovent_own_nnuec_collection_lock_v2(
    arguments: Vec<String>,
    plan: RoventCollectionLockV2LaunchPlan,
    input_directory: &Dir,
    worker_deadline: Instant,
    cleanup_deadline: Instant,
) -> Result<RoventNnuecCollectionRunV2, ArenaError> {
    check((1..=64).contains(&plan.maximum_roots), "Own Collection /2 root limit")?;
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
        RoventNnuecCollectionBridgeV2::new(owner.clone(), Arc::clone(&manifest), roster)?;
    let (publisher, pump_sender) = OwnPumpConsumerV2::new(plan.maximum_records as usize);
    worker.configure_pump_sender(pump_sender.clone())?;
    worker.configure_row_pump(plan.teacher_constraint, Duration::from_nanos(plan.teacher_soft_quantum_ns))?;
    let publisher_stop = Arc::new(AtomicBool::new(false));
    let publisher_slot = Arc::new(Mutex::new(None));
    let admission_publisher_slot = Arc::clone(&publisher_slot);
    let thread_publisher = Arc::clone(&publisher);
    let thread_publisher_stop = Arc::clone(&publisher_stop);
    let publisher_manifest = Arc::clone(&manifest);
    let publisher_output = plan.output_directory.clone();
    let publisher_max_records = plan.maximum_records;
    let publisher_timeout = Duration::from_millis(plan.publication_timeout_ms);
    let worker = Arc::new(Mutex::new(worker));
    let resource = Arc::new(Mutex::new(resources::CollectionResourceCustody::prepare(
        input_directory,
        &plan.resource_plan,
    )?));
    let original_lock = Arc::new(Mutex::new(None));
    let admission_resource = Arc::clone(&resource);
    let admission_lock = Arc::clone(&original_lock);
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = Arc::clone(&stop);
    let thread_worker = Arc::clone(&worker);
    // The output/cgroup resource gate can migrate only this startup thread.
    // Do not spawn any worker until that original gate has succeeded.
    let worker_start = Arc::new(AtomicU8::new(WORKER_NOT_STARTED));
    let admission_worker_start = Arc::clone(&worker_start);
    let admission_bridge = Arc::clone(&bridge);
    let router = Arc::new(crate::rovent_collect::own_root_router_v2::RoventOwnRootRouterV2::new(
        Arc::clone(&bridge), plan.maximum_roots as usize,
    )?);
    let hooks: Arc<dyn rz_uci::rovent_bootstrap::nnuec_session::RoventNnuecCollectionHooksV2> = router.clone();
    let supervisor_custody = Arc::new(Mutex::new(None));
    let admission_supervisor = Arc::clone(&supervisor_custody);
    let supervisor_router = Arc::clone(&router);
    let supervisor_owner = owner.clone();
    let supervisor_manifest = Arc::clone(&manifest);
    let supervisor_roots = plan.maximum_roots as usize;
    let supervisor_records = plan.maximum_records as usize;
    let supervisor_constraint = plan.teacher_constraint;
    let supervisor_quantum = Duration::from_nanos(plan.teacher_soft_quantum_ns);
    let native = run_own_nnuec_collection_v2_with_resource_admission(
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
            manifest.validate(lock).map_err(|error| RoventLaunchError {
                stage: "collection_manifest_v2",
                detail: error.to_string(),
            })?;
            {
                let mut resource = admission_resource.lock().map_err(|_| RoventLaunchError {
                    stage: "resource_admission",
                    detail: "Collection /2 resource custody poisoned".into(),
                })?;
                resource.admit_v2(lock).map_err(|error| RoventLaunchError {
                    stage: "resource_admission",
                    detail: error.to_string(),
                })?;
            }
            let mut publisher_handle = admission_publisher_slot.lock().map_err(|_| RoventLaunchError {
                stage: "publisher_worker_start", detail: "publisher handle custody poisoned".into(),
            })?;
            let handle = std::thread::Builder::new().name("rovent-own-publisher-v2".into())
                .spawn(move || thread_publisher.run(&thread_publisher_stop, cleanup_deadline))
                .map_err(|error| RoventLaunchError { stage: "publisher_worker_start", detail: error.to_string() })?;
            *publisher_handle = Some(handle);
            drop(publisher_handle);
            admission_worker_start.store(WORKER_STARTING, Ordering::Release);
            match admission_bridge.start_worker(&thread_worker, worker_stop, worker_deadline) {
                Ok(()) => admission_worker_start.store(WORKER_STARTED, Ordering::Release),
                Err(error) => {
                    let error = match error {
                        OwnWorkerStartErrorV2::BeforeStart(error) => {
                            admission_worker_start.store(WORKER_START_FAILED, Ordering::Release);
                            error
                        }
                        OwnWorkerStartErrorV2::Unknown(error) => error,
                    };
                    return Err(RoventLaunchError { stage: "teacher_worker_start", detail: error.to_string() });
                }
            }
            if supervisor_roots > 1 {
                let mut custody = admission_supervisor.lock().map_err(|_| RoventLaunchError {
                    stage: "collection_supervisor_start", detail: "supervisor custody poisoned".into(),
                })?;
                *custody = Some(crate::rovent_collect::own_worker_supervisor_v2::OwnWorkerSupervisorV2::start(
                    supervisor_router, supervisor_owner, supervisor_manifest, roster_identity,
                    supervisor_constraint, supervisor_quantum, supervisor_roots, supervisor_records,
                    worker_deadline, cleanup_deadline,
                ).map_err(|error| RoventLaunchError { stage: "collection_supervisor_start", detail: error.to_string() })?);
            }
            Ok(())
        },
    );
    stop.store(true, Ordering::Release);
    let worker_state = worker_start.load(Ordering::Acquire);
    let worker_not_started = matches!(worker_state, WORKER_NOT_STARTED | WORKER_START_FAILED);
    let slot_poisoned = worker_state == WORKER_STARTING;
    let (publisher_thread, publisher_unknown) = match publisher_slot.lock() {
        Ok(mut slot) => (slot.take(), false),
        Err(poison) => (poison.into_inner().take(), true),
    };
    if worker_not_started { publisher_stop.store(true, Ordering::Release); }
    let mut run = RoventNnuecCollectionRunV2 {
        resource,
        original_lock,
        owner,
        bridge,
        worker,
        worker_thread: None,
        worker_not_started,
        worker_exit: None,
        bridge_worker_exit: None,
        supervisor: supervisor_custody.lock().map_err(|_| invalid("Own supervisor custody poisoned"))?.take(),
        secondary_slots: Vec::new(),
        supervisor_error: None,
        retained_rows: std::collections::VecDeque::new(),
        completed_workers: Vec::new(),
        worker_thread_panicked: false,
        worker_custody_poisoned: slot_poisoned,
        transferred_handoff: None,
        recovered_handoff: None,
        escrow_recovery_error: None,
        handoff_transfer_error: None,
        commit_failure: None,
        publisher,
        publisher_sender: pump_sender,
        publisher_stop,
        publisher_thread,
        publisher_exit: None,
        publisher_unknown,
        publisher_assembly: None,
        publisher_directory: None,
        publisher_output_parent: None,
        publisher_manifest,
        publisher_output,
        publisher_max_records,
        publisher_timeout,
        publisher_status: "pending",
        publisher_error: None,
        publisher_panic: None,
        publisher_receipt: None,
        native_failure: native.err(),
        native_failure_taken: false,
        pre_admission_failure_transferred: false,
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
    while run.publisher_thread.is_some() && Instant::now() < cleanup_deadline {
        run.poll_worker();
        if run.publisher_thread.is_some() { std::thread::sleep(Duration::from_millis(2)); }
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
    if run.publisher_exit.as_ref().is_some_and(Result::is_ok) && run.primary_join_observed() {
        let until = Instant::now().checked_add(run.publisher_timeout)
            .map_or(cleanup_deadline, |publication_until| publication_until.min(cleanup_deadline));
        match run.finish_publisher(until) {
            Ok(()) => run.publisher_status = "published",
            Err(error) => { run.publisher_status = "partial"; run.publisher_error = Some(error); }
        }
    } else {
        run.publisher_status = "unknown_or_not_started";
    }
    Ok(run)
}
