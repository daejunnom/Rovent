//! Bounded, nonblocking P-C /2 handoff. The manager owns the observer during
//! callbacks; this worker never accesses its producer while that root is live.
//! Teacher process retirement and native handoff remain separate observations.
#![allow(dead_code)]
use super::teacher::v2_session::{
    RoventCollectionTeacherSessionV2, RoventTeacherRosterEntryV2, TeacherSessionRetirementSealV2,
    TeacherSessionStartErrorV2,
};
use super::{ArenaError, RoventCollectionOwnerLockV2, check};
use rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2;
use rz_uci::pals_adaptive::{AdaptiveManagerError, RoventCollectionRootAdmissionV2};
use rz_uci::rovent_bootstrap::RoventStartupControl;
use rz_uci::rovent_bootstrap::pc_session::{
    RoventPcCollectionAdmissionFailureV2, RoventPcCollectionAdmissionPortV2,
    RoventPcCollectionHandoffV2, RoventPcCollectionHooksV2,
};
use rz_uci::rovent_external_launch::{RoventTeacherRetirementTokenV2, TeacherRecoveryCustodianV2};
use std::collections::BTreeMap;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, AtomicU8, Ordering},
    mpsc,
};
use std::time::{Duration, Instant};

const RESERVING: u8 = 0;
const INSTALLED: u8 = 1;
const REJECTED: u8 = 2;
const ADMISSION_OPEN: u8 = 0;
const ADMISSION_RESERVED: u8 = 1;
const ADMISSION_CLOSED: u8 = 2;
const ADMISSION_INSTALLED: u8 = 3;

struct AdmissionSlot {
    admission: RoventCollectionRootAdmissionV2,
    state: AtomicU8,
}

struct AdmissionReservationGuard(Arc<AdmissionSlot>);
impl Drop for AdmissionReservationGuard {
    fn drop(&mut self) {
        let _ =
            self.0
                .state
                .compare_exchange(RESERVING, REJECTED, Ordering::AcqRel, Ordering::Acquire);
    }
}

/// One original P-C root at a time; the worker and native owner retain their
/// own copies until retirement. A channel transfer is not completion evidence.
pub(crate) struct RoventPcCollectionBridgeV2 {
    owner: RoventCollectionOwnerLockV2,
    admissions: mpsc::SyncSender<Arc<AdmissionSlot>>,
    handoffs: mpsc::SyncSender<Arc<Mutex<RoventPcCollectionHandoffV2>>>,
    admission_phase: Arc<AtomicU8>,
    revoked: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    startup: Arc<OnceLock<RoventStartupControl>>,
    startup_registration_started: AtomicBool,
}

impl RoventPcCollectionBridgeV2 {
    pub(crate) fn new(
        owner: RoventCollectionOwnerLockV2,
        manifest: Arc<RoventCorpusManifestV2>,
        roster: Vec<RoventTeacherRosterEntryV2>,
    ) -> Result<(Arc<Self>, RoventPcCollectionWorkerV2), ArenaError> {
        manifest.validate()?;
        check(
            owner.matches_registration(&manifest)?,
            "V2 teacher and original corpus registration differ",
        )?;
        check(
            !roster.is_empty() && roster.len() <= 16,
            "bounded V2 teacher roster",
        )?;
        let (admissions, admission_receiver) = mpsc::sync_channel(1);
        let (handoffs, handoff_receiver) = mpsc::sync_channel(1);
        let revoked = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(AtomicBool::new(false));
        let admission_phase = Arc::new(AtomicU8::new(ADMISSION_OPEN));
        let startup = Arc::new(OnceLock::new());
        let bridge = Arc::new(Self {
            owner,
            admissions,
            handoffs,
            admission_phase: Arc::clone(&admission_phase),
            revoked: Arc::clone(&revoked),
            failed: Arc::clone(&failed),
            startup: Arc::clone(&startup),
            startup_registration_started: AtomicBool::new(false),
        });
        let worker = RoventPcCollectionWorkerV2 {
            admission_receiver,
            handoff_receiver,
            revoked,
            failed,
            admission_phase,
            startup,
            manifest,
            roster,
            slot: None,
            handoff: None,
            session: None,
            retirement: None,
            custodian: TeacherRecoveryCustodianV2::new(),
            tokens: BTreeMap::new(),
            diagnostics: BTreeMap::new(),
            next_actor: 0,
            shutdown_requested: false,
            failure: None,
            additional_failures: Vec::new(),
            omitted_failures: 0,
        };
        Ok((bridge, worker))
    }
}

impl RoventPcCollectionHooksV2 for RoventPcCollectionBridgeV2 {
    fn register_startup_control(
        &self,
        control: &RoventStartupControl,
    ) -> Result<(), AdaptiveManagerError> {
        if self
            .startup_registration_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(AdaptiveManagerError::Invalid(
                "P/C V2 startup control registration repeated",
            ));
        }
        self.startup
            .set(control.clone())
            .map_err(|_| AdaptiveManagerError::Invalid("P/C V2 startup control already registered"))
    }
    fn admit(
        &self,
        port: &mut RoventPcCollectionAdmissionPortV2<'_>,
        admission: RoventCollectionRootAdmissionV2,
    ) -> Result<(), RoventPcCollectionAdmissionFailureV2> {
        let reject = |error, admission| RoventPcCollectionAdmissionFailureV2 {
            error,
            admission,
            owners: Box::new(()),
        };
        if self.startup.get().is_none()
            || self.revoked.load(Ordering::Acquire)
            || self.failed.load(Ordering::Acquire)
            || self
                .admission_phase
                .compare_exchange(
                    ADMISSION_OPEN,
                    ADMISSION_RESERVED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
        {
            return Err(reject(AdaptiveManagerError::AdmissionClosed, admission));
        }
        // Reserve the bounded outbox before installing the observer. The
        // consumer cannot use the root until the release-store of INSTALLED.
        let slot = Arc::new(AdmissionSlot {
            admission: admission.clone(),
            state: AtomicU8::new(RESERVING),
        });
        if self.admissions.try_send(Arc::clone(&slot)).is_err() {
            self.admission_phase
                .store(ADMISSION_CLOSED, Ordering::Release);
            return Err(reject(
                AdaptiveManagerError::Capacity("P/C V2 admission outbox"),
                admission,
            ));
        }
        // If installation panics, UCI retains its manager/panic evidence and
        // this guard prevents an unpublished reservation from spinning forever.
        let _reservation = AdmissionReservationGuard(Arc::clone(&slot));
        if self.admission_phase.load(Ordering::Acquire) != ADMISSION_RESERVED {
            return Err(reject(AdaptiveManagerError::Deadline, admission));
        }
        match port.install(admission, self.owner.observer()) {
            Ok(()) => {
                // The deadline may have closed admission while install ran.
                // The manager still owns the installed observer in that case.
                if self
                    .admission_phase
                    .compare_exchange(
                        ADMISSION_RESERVED,
                        ADMISSION_INSTALLED,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_err()
                {
                    return Err(reject(
                        AdaptiveManagerError::Deadline,
                        slot.admission.clone(),
                    ));
                }
                if slot
                    .state
                    .compare_exchange(RESERVING, INSTALLED, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    return Err(reject(
                        AdaptiveManagerError::Deadline,
                        slot.admission.clone(),
                    ));
                }
                Ok(())
            }
            Err(original) => {
                slot.state.store(REJECTED, Ordering::Release);
                Err(RoventPcCollectionAdmissionFailureV2 {
                    error: original.error.clone(),
                    admission: original.admission.clone(),
                    owners: Box::new(original),
                })
            }
        }
    }
    fn pump(&self) -> Result<(), AdaptiveManagerError> {
        // Teacher failure remains with the worker and invalidates collection,
        // but does not by itself fail an unrelated native search or infer a
        // physical error. Admission of another root is still closed.
        Ok(())
    }
    fn revoke(&self) {
        self.revoked.store(true, Ordering::Release);
        let _ = self.admission_phase.compare_exchange(
            ADMISSION_OPEN,
            ADMISSION_CLOSED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
    fn handoff_sender(&self) -> mpsc::SyncSender<Arc<Mutex<RoventPcCollectionHandoffV2>>> {
        self.handoffs.clone()
    }
}

pub(crate) enum WorkerFailure {
    Preparation(ArenaError),
    ActorStart(TeacherSessionStartErrorV2),
    Retirement(ArenaError),
    HandoffDisconnected,
    HandoffMismatch,
    RootClosedBeforeStartup,
    WorkerPanicked,
    RetirementSealRejected,
}

/// Worker methods must be called from a separate owner, outside manager/root
/// locks. The caller retains this value on every failure or unknown completion.
pub(crate) struct RoventPcCollectionWorkerV2 {
    admission_receiver: mpsc::Receiver<Arc<AdmissionSlot>>,
    handoff_receiver: mpsc::Receiver<Arc<Mutex<RoventPcCollectionHandoffV2>>>,
    revoked: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    admission_phase: Arc<AtomicU8>,
    startup: Arc<OnceLock<RoventStartupControl>>,
    manifest: Arc<RoventCorpusManifestV2>,
    roster: Vec<RoventTeacherRosterEntryV2>,
    slot: Option<Arc<AdmissionSlot>>,
    handoff: Option<Arc<Mutex<RoventPcCollectionHandoffV2>>>,
    session: Option<RoventCollectionTeacherSessionV2>,
    retirement: Option<TeacherSessionRetirementSealV2>,
    custodian: TeacherRecoveryCustodianV2,
    tokens: BTreeMap<String, RoventTeacherRetirementTokenV2>,
    diagnostics: BTreeMap<String, rz_search::external_cpu::collection_teacher::TeacherDiagnostics>,
    next_actor: usize,
    shutdown_requested: bool,
    failure: Option<WorkerFailure>,
    additional_failures: Vec<WorkerFailure>,
    omitted_failures: u64,
}

/// Outcome of the worker loop only. The returned worker still owns any
/// unresolved session, original errors, tokens and native handoff.
pub(crate) enum RoventPcCollectionWorkerExitV2 {
    NoAdmission,
    RetirementAndHandoffObserved,
    Deadline,
    Panicked(Vec<Box<dyn std::any::Any + Send>>),
}

impl RoventPcCollectionWorkerV2 {
    fn record_failure(&mut self, failure: WorkerFailure) {
        if self.failure.is_none() {
            self.failure = Some(failure);
        } else if self.additional_failures.len() < 32 {
            self.additional_failures.push(failure);
        } else {
            self.omitted_failures = self.omitted_failures.saturating_add(1);
        }
        self.failed.store(true, Ordering::Release);
    }

    /// Does bounded control work on the separate owner. Teacher source
    /// preparation and process startup may perform I/O: never call from the
    /// manager, a native callback, or while holding the collection mutex.
    pub(crate) fn poll(&mut self) {
        if self.slot.is_none() {
            match self.admission_receiver.try_recv() {
                Ok(slot) => self.slot = Some(slot),
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.record_failure(WorkerFailure::HandoffDisconnected);
                    return;
                }
            }
        }
        let slot = Arc::clone(self.slot.as_ref().expect("admission retained"));
        match slot.state.load(Ordering::Acquire) {
            RESERVING => return,
            REJECTED => return,
            INSTALLED => {}
            _ => {
                self.record_failure(WorkerFailure::HandoffMismatch);
                return;
            }
        }
        let root = slot.admission.root_admission();
        let Some(until) = root
            .runtime_clock()
            .origin()
            .checked_add(Duration::from_nanos(root.admission_deadline().at.0))
        else {
            self.record_failure(WorkerFailure::HandoffMismatch);
            return;
        };
        let Some(cleanup_until) = root
            .runtime_clock()
            .origin()
            .checked_add(Duration::from_nanos(root.cleanup_deadline().at.0))
        else {
            self.record_failure(WorkerFailure::HandoffMismatch);
            return;
        };
        if self.session.is_none()
            && self.retirement.is_none()
            && self.failure.is_none()
            && (self.revoked.load(Ordering::Acquire) || Instant::now() >= until)
        {
            self.record_failure(WorkerFailure::RootClosedBeforeStartup);
        }
        if self.session.is_none()
            && self.retirement.is_none()
            && self.failure.is_none()
            && !self.revoked.load(Ordering::Acquire)
        {
            let Some(startup) = self.startup.get() else {
                self.record_failure(WorkerFailure::HandoffMismatch);
                return;
            };
            match RoventCollectionTeacherSessionV2::prepare(
                &slot.admission,
                startup,
                &self.manifest,
                &self.roster,
            ) {
                Ok(session) => self.session = Some(session),
                Err(error) => self.record_failure(WorkerFailure::Preparation(error)),
            }
        }
        if let Some(session) = &mut self.session {
            if !self.revoked.load(Ordering::Acquire)
                && self.failure.is_none()
                && self.next_actor < self.roster.len()
                && Instant::now() < until
            {
                let actor = &self.roster[self.next_actor].actor_id;
                match session.start_actor(actor, until, cleanup_until, &mut self.custodian) {
                    Ok(_) => self.next_actor += 1,
                    Err(error) => self.record_failure(WorkerFailure::ActorStart(error)),
                }
            }
        }
        if self.session.is_some()
            && self.next_actor < self.roster.len()
            && self.failure.is_none()
            && Instant::now() >= until
        {
            self.record_failure(WorkerFailure::RootClosedBeforeStartup);
        }
        if (self.revoked.load(Ordering::Acquire) || self.failure.is_some())
            && !self.shutdown_requested
        {
            if let Some(session) = &mut self.session {
                session.request_shutdown(cleanup_until);
            }
            self.shutdown_requested = true;
        }
        if self.shutdown_requested {
            if let Some(session) = &mut self.session {
                for item in &self.roster {
                    if self.tokens.contains_key(&item.actor_id) {
                        continue;
                    }
                    if let Some(worker) = session.worker_mut(&item.actor_id) {
                        if let Some(original) = worker.take_diagnostics_after_join() {
                            self.diagnostics.insert(item.actor_id.clone(), original);
                        }
                    }
                    match session.try_retire_actor(&item.actor_id) {
                        Ok(Some(token)) => {
                            self.tokens.insert(item.actor_id.clone(), token);
                        }
                        Ok(None) => {}
                        Err(error) => {
                            if self.failure.is_none() {
                                self.failure = Some(WorkerFailure::Retirement(error));
                            } else if self.additional_failures.len() < 32 {
                                self.additional_failures
                                    .push(WorkerFailure::Retirement(error));
                            } else {
                                self.omitted_failures = self.omitted_failures.saturating_add(1);
                            }
                            self.failed.store(true, Ordering::Release);
                        }
                    }
                }
            }
            if !self.failed.load(Ordering::Acquire)
                && self.next_actor == self.roster.len()
                && self.tokens.len() == self.roster.len()
            {
                if let Some(session) = self.session.take() {
                    match session.try_seal_retirement(std::mem::take(&mut self.tokens)) {
                        Ok(seal) => self.retirement = Some(seal),
                        Err(rejected) => {
                            self.session = Some(rejected.session);
                            self.tokens = rejected.actors;
                            self.record_failure(WorkerFailure::RetirementSealRejected);
                        }
                    }
                }
            }
        }
        if self.handoff.is_none() {
            match self.handoff_receiver.try_recv() {
                Ok(handoff) => self.handoff = Some(handoff),
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.record_failure(WorkerFailure::HandoffDisconnected)
                }
            }
        }
    }

    /// Retains the actual handoff and the session seal; neither is a published
    /// corpus or proof that a teacher label was consumed.
    pub(crate) fn retirement_matches_handoff(&mut self) -> bool {
        if self.failed.load(Ordering::Acquire) {
            return false;
        }
        let (Some(slot), Some(handoff), Some(seal)) = (&self.slot, &self.handoff, &self.retirement)
        else {
            return false;
        };
        let Ok(mut original) = handoff.try_lock() else {
            return false;
        };
        let (admission, _, settlement, fence) = original.parts_mut();
        admission.same_original_admission(&slot.admission)
            && seal.matches_admission(admission)
            && settlement.is_ok()
            && fence.permits_semantic_closure()
    }

    /// Return the original root cleanup ceiling once admitted. The startup
    /// deadline does not bound waiting for a later UCI `go`.
    pub(crate) fn deadline(&self) -> Option<Instant> {
        self.slot.as_ref().and_then(|slot| {
            let root = slot.admission.root_admission();
            root.runtime_clock()
                .origin()
                .checked_add(Duration::from_nanos(root.cleanup_deadline().at.0))
        })
    }

    pub(crate) fn has_failure(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    pub(crate) fn failure(&self) -> Option<&WorkerFailure> {
        self.failure.as_ref()
    }

    pub(crate) fn additional_failures(&self) -> (&[WorkerFailure], u64) {
        (&self.additional_failures, self.omitted_failures)
    }

    pub(crate) fn diagnostics(
        &self,
    ) -> &BTreeMap<String, rz_search::external_cpu::collection_teacher::TeacherDiagnostics> {
        &self.diagnostics
    }

    /// Run with a separately retained original owner, typically held by an
    /// Arc<Mutex<_>> shared with the caller. Keep the JoinHandle if it is not
    /// finished; a join timeout is never physical-completion proof. A poisoned
    /// lock preserves its inner worker but is not a successful completion.
    /// `stop` requests shutdown but does not end the loop before the root's
    /// handoff and teacher retirement or the bounded cleanup deadline.
    pub(crate) fn run_until(
        &mut self,
        stop: Arc<AtomicBool>,
        until: Instant,
    ) -> RoventPcCollectionWorkerExitV2 {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loop {
                if stop.load(Ordering::Acquire) {
                    self.revoked.store(true, Ordering::Release);
                }
                self.poll();
                if self.retirement_matches_handoff() {
                    break RoventPcCollectionWorkerExitV2::RetirementAndHandoffObserved;
                }
                if stop.load(Ordering::Acquire)
                    && self.slot.is_none()
                    && self
                        .admission_phase
                        .compare_exchange(
                            ADMISSION_OPEN,
                            ADMISSION_CLOSED,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                {
                    break RoventPcCollectionWorkerExitV2::NoAdmission;
                }
                if stop.load(Ordering::Acquire)
                    && self.slot.is_none()
                    && self.admission_phase.load(Ordering::Acquire) == ADMISSION_CLOSED
                {
                    break RoventPcCollectionWorkerExitV2::NoAdmission;
                }
                let deadline = self.deadline().map_or(until, |root| root.min(until));
                if Instant::now() >= deadline {
                    self.revoked.store(true, Ordering::Release);
                    let _ = self.admission_phase.compare_exchange(
                        ADMISSION_OPEN,
                        ADMISSION_CLOSED,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                    // Invalidate an unfinished admission reservation before
                    // returning the original worker as deadline/unknown. A
                    // concurrent install checks the same atomic transition.
                    let _ = self.admission_phase.compare_exchange(
                        ADMISSION_RESERVED,
                        ADMISSION_CLOSED,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                    if self.slot.is_none() {
                        if let Ok(slot) = self.admission_receiver.try_recv() {
                            self.slot = Some(slot);
                        }
                    }
                    if let Some(slot) = &self.slot {
                        let _ = slot.state.compare_exchange(
                            RESERVING,
                            REJECTED,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        );
                    }
                    if !self.shutdown_requested {
                        if let Some(session) = &mut self.session {
                            session.request_shutdown(deadline);
                        }
                        self.shutdown_requested = true;
                    }
                    break RoventPcCollectionWorkerExitV2::Deadline;
                }
                let interval = if self.slot.is_none() { 20 } else { 2 };
                std::thread::sleep(
                    deadline
                        .saturating_duration_since(Instant::now())
                        .min(Duration::from_millis(interval)),
                );
            }
        }));
        let exit = match outcome {
            Ok(exit) => exit,
            Err(payload) => {
                self.revoked.store(true, Ordering::Release);
                let _ = self.admission_phase.compare_exchange(
                    ADMISSION_OPEN,
                    ADMISSION_CLOSED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                let deadline = self.deadline().map_or(until, |root| root.min(until));
                let shutdown = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if let Some(session) = &mut self.session {
                        session.request_shutdown(deadline);
                        self.shutdown_requested = true;
                    }
                }));
                self.record_failure(WorkerFailure::WorkerPanicked);
                let mut payloads = vec![payload];
                if let Err(second) = shutdown {
                    payloads.push(second);
                }
                RoventPcCollectionWorkerExitV2::Panicked(payloads)
            }
        };
        exit
    }
}
