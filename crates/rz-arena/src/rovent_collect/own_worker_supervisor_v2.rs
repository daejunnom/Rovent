//! Resource-admitted, bounded preparation of Own Collection workers.
//!
//! This owner never executes from a native manager callback.  It keeps the
//! original secondary bridges, workers and consumer queues together; publishing
//! a prepared root is not corpus publication or physical completion.
use super::{invalid, ArenaError, RoventCollectionOwnerLockV2};
use super::launch::own_publisher_v2::{OwnPumpConsumerV2, OwnPumpPollV2};
use super::own_hooks_lock_v2::{OwnRootSeparationV2, OwnWorkerExitKindV2, OwnWorkerStartErrorV2, RoventNnuecCollectionBridgeV2, RoventNnuecCollectionWorkerV2};
use super::own_root_router_v2::{OwnPreparedSlotOutcomeV2, RoventOwnRootRouterV2};
use super::teacher::v2_session::RoventTeacherRosterEntryV2;
use super::teacher::lock_v2_row_pump::PumpItemV2;
use rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2;
use rz_search::external_cpu::ExternalGoConstraint;
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2;
use rz_uci::rovent_bootstrap::nnuec_session::RoventNnuecCollectionHooksV2;
use std::{sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}}, thread::{self, JoinHandle}, time::{Duration, Instant}};

/// These are the actual objects created for a secondary root.  A returned
/// entry is custody, not a receipt or proof that its rows were published.
pub(crate) struct OwnSecondarySlotV2 {
    pub(crate) previous: Arc<RoventNnuecCollectionBridgeV2>,
    pub(crate) previous_admission: RoventCollectionRootAdmissionV2,
    pub(crate) bridge: Arc<RoventNnuecCollectionBridgeV2>,
    pub(crate) worker: Arc<Mutex<RoventNnuecCollectionWorkerV2>>,
    pub(crate) consumer: Arc<OwnPumpConsumerV2>,
    /// The original sender is retained to drain rows left with this worker
    /// after it joins; a newly created channel cannot represent that custody.
    pub(crate) sender: std::sync::mpsc::SyncSender<PumpItemV2>,
    pub(crate) retained_previous_rows: usize,
    pub(crate) published_to_router: bool,
    pub(crate) worker_start: OwnSecondaryWorkerStartV2,
    pub(crate) worker_join_observed: bool,
    pub(crate) worker_exit_kind: Option<OwnWorkerExitKindV2>,
    pub(crate) consumer_drained_after_join: bool,
    pub(crate) consumer_empty_before_worker_start: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OwnSecondaryWorkerStartV2 { StartingUnknown, Started, FailedBeforeStart }

struct SupervisorState {
    slots: Vec<OwnSecondarySlotV2>,
    pending: Option<OwnSecondarySlotV2>,
    error: Option<ArenaError>,
}

/// Owns the original supervisor thread and its exact slot state.  Callers
/// must retain this object across native execution and inspect/join it before
/// transferring slot custody to the game publisher.
pub(crate) struct OwnWorkerSupervisorV2 {
    state: Arc<Mutex<SupervisorState>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    joined: bool,
    panic: Option<Box<dyn std::any::Any + Send>>,
    transferred: bool,
}

impl OwnWorkerSupervisorV2 {
    /// Called only after original resource admission has succeeded.  It does
    /// not extend any original deadline or create an external engine itself.
    pub(crate) fn start(
        router: Arc<RoventOwnRootRouterV2>,
        owner: RoventCollectionOwnerLockV2,
        manifest: Arc<RoventCorpusManifestLockV2>,
        roster: Vec<(String, usize, u32)>,
        constraint: ExternalGoConstraint,
        quantum: Duration,
        maximum_roots: usize,
        maximum_records: usize,
        worker_deadline: Instant,
        cleanup_deadline: Instant,
    ) -> Result<Self, ArenaError> {
        if !(2..=64).contains(&maximum_roots) || maximum_records == 0 || Instant::now() >= worker_deadline
            || worker_deadline > cleanup_deadline {
            return Err(invalid("Own worker supervisor bounds or deadline"));
        }
        let startup = router.original_startup().ok_or_else(|| invalid("Own original startup control unavailable"))?.clone();
        let mut slots = Vec::new();
        slots.try_reserve_exact(maximum_roots - 1).map_err(|_| invalid("Own supervisor slot capacity"))?;
        let state = Arc::new(Mutex::new(SupervisorState { slots, pending: None, error: None }));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_state = Arc::clone(&state);
        let thread_stop = Arc::clone(&stop);
        let thread = thread::Builder::new().name("rovent-own-root-supervisor-v2".into()).spawn(move || {
            while !thread_stop.load(Ordering::Acquire) && Instant::now() < worker_deadline {
                let mut guard = match thread_state.lock() {
                    Ok(guard) => guard,
                    Err(_) => break,
                };
                // The same supervisor drains each bounded receiver.  It does
                // not hold a manager or producer lock while doing so.
                let mut poll_error = None;
                for slot in guard.slots.iter().chain(guard.pending.iter()) {
                    match slot.consumer.poll() {
                        Ok(OwnPumpPollV2::Received | OwnPumpPollV2::Empty | OwnPumpPollV2::Disconnected) => {},
                        Ok(OwnPumpPollV2::Full) => { poll_error = Some(invalid("Own secondary consumer capacity reached")); break; }
                        Err(error) => { poll_error = Some(error); break; }
                    }
                }
                if let Some(error) = poll_error { guard.error.get_or_insert(error); }
                if guard.error.is_none() {
                    if let Err(error) = supervisor_step(&mut guard, &router, &owner, &manifest, &roster,
                        &startup, constraint, quantum, maximum_records, worker_deadline, &thread_stop) {
                        guard.error = Some(error);
                    }
                }
                drop(guard);
                thread::sleep(Duration::from_millis(2));
            }
            // Once the original stop/deadline is reached, stop the secondary
            // workers and drain only until the already registered cleanup
            // ceiling. Empty is final only after that same worker joined.
            thread_stop.store(true, Ordering::Release);
            while Instant::now() < cleanup_deadline {
                let mut guard = match thread_state.lock() {
                    Ok(guard) => guard,
                    Err(_) => break,
                };
                let mut failure = None;
                let mut all_done = true;
                let SupervisorState { slots, pending, .. } = &mut *guard;
                for slot in slots.iter_mut().chain(pending.iter_mut()) {
                    if slot.worker_start != OwnSecondaryWorkerStartV2::FailedBeforeStart {
                        match slot.bridge.poll_worker() {
                            Ok(Some(kind)) => { slot.worker_join_observed = true; slot.worker_exit_kind = Some(kind); }
                            Ok(None) => {},
                            Err(error) => { failure = Some(invalid(format!("Own secondary worker join: {error:?}"))); }
                        }
                    }
                    match slot.consumer.poll() {
                        Ok(OwnPumpPollV2::Received) => slot.consumer_drained_after_join = false,
                        Ok(OwnPumpPollV2::Empty | OwnPumpPollV2::Disconnected) if slot.worker_join_observed => {
                            slot.consumer_drained_after_join = true;
                        }
                        Ok(OwnPumpPollV2::Empty | OwnPumpPollV2::Disconnected)
                            if slot.worker_start == OwnSecondaryWorkerStartV2::FailedBeforeStart => {
                            slot.consumer_empty_before_worker_start = true;
                        }
                        Ok(OwnPumpPollV2::Empty | OwnPumpPollV2::Disconnected) => {},
                        Ok(OwnPumpPollV2::Full) => { failure = Some(invalid("Own secondary consumer capacity reached during drain")); }
                        Err(error) => { failure = Some(error); }
                    }
                    if !(slot.worker_join_observed && slot.consumer_drained_after_join)
                        && !(slot.worker_start == OwnSecondaryWorkerStartV2::FailedBeforeStart && slot.consumer_empty_before_worker_start) {
                        all_done = false;
                    }
                }
                if let Some(error) = failure { guard.error.get_or_insert(error); }
                drop(guard);
                if all_done { break; }
                thread::sleep(Duration::from_millis(2));
            }
        }).map_err(|error| ArenaError::Io(error.to_string()))?;
        Ok(Self { state, stop, thread: Some(thread), joined: false, panic: None, transferred: false })
    }

    pub(crate) fn request_stop(&self) { self.stop.store(true, Ordering::Release); }

    /// Collect only an already-finished supervisor.  A still-running thread is
    /// unknown and retains all of its original slot owners.
    pub(crate) fn poll_join(&mut self) -> bool {
        if self.joined { return true; }
        if !self.thread.as_ref().is_some_and(JoinHandle::is_finished) { return false; }
        match self.thread.take().expect("finished original supervisor").join() {
            Ok(()) => {},
            Err(payload) => self.panic = Some(payload),
        }
        self.joined = true;
        true
    }

    /// Transfer exact slots only once their worker joins and terminal custody
    /// are observed as well. A child that remains unknown stays in this
    /// supervisor; supervisor join alone never establishes child completion.
    pub(crate) fn take_joined_slots(&mut self) -> Result<Option<(Vec<OwnSecondarySlotV2>, Option<ArenaError>)>, ArenaError> {
        if !self.joined || self.panic.is_some() { return Ok(None); }
        if self.transferred { return Err(invalid("Own supervisor slots already transferred")); }
        let mut state = self.state.try_lock().map_err(|_| invalid("Own supervisor state busy or poisoned"))?;
        for slot in state.slots.iter().chain(state.pending.iter()) {
            if slot.worker_start == OwnSecondaryWorkerStartV2::FailedBeforeStart
                && slot.consumer_empty_before_worker_start { continue; }
            if !slot.worker_join_observed || !slot.consumer_drained_after_join { return Ok(None); }
            match slot.worker_exit_kind {
                Some(OwnWorkerExitKindV2::NoAdmission) => {},
                Some(OwnWorkerExitKindV2::Aggregate | OwnWorkerExitKindV2::Retirement) => {
                    if !slot.bridge.returned_handoff_settled()
                        .map_err(|error| invalid(format!("Own secondary handoff: {error:?}")))? {
                        return Ok(None);
                    }
                }
                // A thread join does not resolve a child/process panic,
                // timeout, cancellation or physical-only observation.
                _ => return Ok(None),
            }
        }
        if let Some(pending) = state.pending.take() { state.slots.push(pending); }
        let slots = std::mem::take(&mut state.slots);
        let error = state.error.take();
        self.transferred = true;
        Ok(Some((slots, error)))
    }
}

impl Drop for OwnWorkerSupervisorV2 {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if !self.joined || self.panic.is_some() || !self.transferred {
            // Retain the original thread/state if a caller drops unresolved
            // physical work or unpublished rows. This is not a completion.
            std::mem::forget(Arc::clone(&self.state));
            if let Some(thread) = self.thread.take() { std::mem::forget(thread); }
            if let Some(payload) = self.panic.take() { std::mem::forget(payload); }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn supervisor_step(
    state: &mut SupervisorState,
    router: &RoventOwnRootRouterV2,
    owner: &RoventCollectionOwnerLockV2,
    manifest: &Arc<RoventCorpusManifestLockV2>,
    roster: &[(String, usize, u32)],
    startup: &rz_uci::rovent_bootstrap::RoventStartupControl,
    constraint: ExternalGoConstraint,
    quantum: Duration,
    maximum_records: usize,
    deadline: Instant,
    stop: &Arc<AtomicBool>,
) -> Result<(), ArenaError> {
    if !router.next_slot_unprepared() { return Ok(()); }
    let Some((previous, admission)) = router.latest_installed() else { return Ok(()); };
    // A published slot is merely ready; wait until it is itself installed
    // before preparing another worker from the prior root.
    if state.slots.last().is_some_and(|slot| slot.published_to_router
        && Arc::ptr_eq(&slot.previous, &previous)
        && slot.previous_admission.same_original_admission(&admission)) {
        return Ok(());
    }
    if let Some(pending) = &state.pending {
        if !Arc::ptr_eq(&pending.previous, &previous) || !pending.previous_admission.same_original_admission(&admission) {
            return Err(invalid("Own prepared worker previous original root changed"));
        }
    } else {
        // Do not contend with the live root's observer for the shared owner.
        // Only inspect its budget after the original worker and handoff have
        // settled; the game gate still prevents a later root installation.
        if !previous.returned_handoff_settled()
            .map_err(|error| invalid(format!("Own previous handoff: {error:?}")))? {
            return Ok(());
        }
        let Some(status) = owner.try_status()? else { return Ok(()); };
        if status.game_record_limit != maximum_records {
            return Err(invalid("Own supervisor and original owner record limits differ"));
        }
        let Some(reserved) = status.game_reserved_slots else { return Err(invalid("Own game record budget unknown")); };
        if status.game_retired_total.checked_add(reserved).is_none_or(|used| used >= status.game_record_limit) {
            return Ok(());
        }
        if !matches!(previous.try_separate_root_for_next(&admission, maximum_records)?, OwnRootSeparationV2::Released { .. }) {
            return Ok(());
        }
        let entries = roster.iter().map(|(actor, index, collection_owner)| RoventTeacherRosterEntryV2 {
            actor_id: actor.clone(), source_index: *index, collection_owner: *collection_owner,
        }).collect();
        let (next, mut worker) = RoventNnuecCollectionBridgeV2::new(owner.clone(), Arc::clone(manifest), entries)?;
        next.register_startup_control(startup).map_err(|error| invalid(format!("Own original startup registration: {error:?}")))?;
        let (consumer, sender) = OwnPumpConsumerV2::new(maximum_records);
        worker.configure_row_pump(constraint, quantum)?;
        worker.configure_pump_sender(sender.clone())?;
        let worker = Arc::new(Mutex::new(worker));
        state.pending = Some(OwnSecondarySlotV2 { previous, previous_admission: admission,
            bridge: Arc::clone(&next), worker: Arc::clone(&worker), consumer, sender,
            retained_previous_rows: 0, published_to_router: false,
            worker_start: OwnSecondaryWorkerStartV2::StartingUnknown,
            worker_join_observed: false, worker_exit_kind: None,
            consumer_drained_after_join: false, consumer_empty_before_worker_start: false });
        match next.start_worker(&worker, Arc::clone(stop), deadline) {
            Ok(()) => state.pending.as_mut().expect("original secondary slot retained").worker_start = OwnSecondaryWorkerStartV2::Started,
            Err(OwnWorkerStartErrorV2::BeforeStart(error)) => {
                state.pending.as_mut().expect("original secondary slot retained").worker_start = OwnSecondaryWorkerStartV2::FailedBeforeStart;
                return Err(error);
            }
            Err(OwnWorkerStartErrorV2::Unknown(error)) => return Err(error),
        }
    }
    let pending = state.pending.as_ref().expect("prepared original secondary slot");
    match router.publish_next_prepared(&pending.previous, &pending.previous_admission, &pending.bridge, maximum_records)? {
        OwnPreparedSlotOutcomeV2::Pending => Ok(()),
        OwnPreparedSlotOutcomeV2::Published { retained_rows } => {
            let mut pending = state.pending.take().expect("prepared original secondary slot");
            pending.retained_previous_rows = retained_rows;
            pending.published_to_router = true;
            // Space was reserved for every possible secondary slot before the
            // supervisor thread was started.
            state.slots.push(pending);
            Ok(())
        }
    }
}
