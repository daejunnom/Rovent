//! Shared bounded, nonblocking single-root /2 handoff. The manager owns the observer during
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
use rz_uci::rovent_external_launch::{
    RoventTeacherRetirementObservationV2, RoventTeacherRetirementTokenV2,
    TeacherRecoveryCustodianV2,
};
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
const ADMISSION_CANCELED: u8 = 4;

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

/// One original root at a time; the worker and native owner retain their
/// own copies until retirement. A channel transfer is not completion evidence.
pub(crate) struct CollectionBridgeCoreV2<H> {
    owner: RoventCollectionOwnerLockV2,
    admissions: mpsc::SyncSender<Arc<AdmissionSlot>>,
    handoffs: mpsc::SyncSender<Arc<Mutex<H>>>,
    admission_phase: Arc<AtomicU8>,
    revoked: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    startup: Arc<OnceLock<RoventStartupControl>>,
    startup_registration_started: AtomicBool,
}

impl<H: Send + 'static> CollectionBridgeCoreV2<H> {
    pub(crate) fn new(
        owner: RoventCollectionOwnerLockV2,
        manifest: Arc<RoventCorpusManifestV2>,
        roster: Vec<RoventTeacherRosterEntryV2>,
        handoff_matches: HandoffMatches<H>,
    ) -> Result<(Arc<Self>, CollectionWorkerCoreV2<H>), ArenaError> {
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
        let worker = CollectionWorkerCoreV2 {
            admission_receiver,
            handoff_receiver,
            handoff_matches,
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
            teacher_observations: BTreeMap::new(),
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

pub(crate) type HandoffMatches<H> = fn(&mut H, &RoventCollectionRootAdmissionV2, &TeacherSessionRetirementSealV2) -> bool;

pub(crate) struct CoreAdmissionFailure {
    pub(crate) error: AdaptiveManagerError,
    pub(crate) admission: RoventCollectionRootAdmissionV2,
    pub(crate) owners: Box<dyn std::any::Any + Send>,
}

impl<H: Send + 'static> CollectionBridgeCoreV2<H> {
    pub(crate) fn register_startup_control(
        &self,
        control: &RoventStartupControl,
    ) -> Result<(), AdaptiveManagerError> {
        if self
            .startup_registration_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(AdaptiveManagerError::Invalid(
                "Collection V2 startup control registration repeated",
            ));
        }
        self.startup
            .set(control.clone())
            .map_err(|_| AdaptiveManagerError::Invalid("Collection V2 startup control already registered"))
    }
    pub(crate) fn admit(
        &self,
        admission: RoventCollectionRootAdmissionV2,
        install: impl FnOnce(RoventCollectionRootAdmissionV2, &RoventCollectionOwnerLockV2) -> Result<(), CoreAdmissionFailure>,
    ) -> Result<(), CoreAdmissionFailure> {
        let reject = |error, admission| CoreAdmissionFailure {
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
            let _ = self.admission_phase.compare_exchange(
                ADMISSION_RESERVED,
                ADMISSION_CLOSED,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            return Err(reject(
                AdaptiveManagerError::Capacity("Collection V2 admission outbox"),
                admission,
            ));
        }
        // If installation panics, UCI retains its manager/panic evidence and
        // this guard prevents an unpublished reservation from spinning forever.
        let _reservation = AdmissionReservationGuard(Arc::clone(&slot));
        if self.admission_phase.load(Ordering::Acquire) != ADMISSION_RESERVED {
            return Err(reject(AdaptiveManagerError::Deadline, admission));
        }
        match install(admission, &self.owner) {
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
                Err(CoreAdmissionFailure {
                    error: original.error,
                    admission: original.admission,
                    owners: original.owners,
                })
            }
        }
    }
    pub(crate) fn pump(&self) -> Result<(), AdaptiveManagerError> {
        // Teacher failure remains with the worker and invalidates collection,
        // but does not by itself fail an unrelated native search or infer a
        // physical error. Admission of another root is still closed.
        Ok(())
    }
    pub(crate) fn revoke(&self) {
        self.revoked.store(true, Ordering::Release);
        let _ = self.admission_phase.compare_exchange(
            ADMISSION_OPEN,
            ADMISSION_CLOSED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        let _ = self.admission_phase.compare_exchange(
            ADMISSION_RESERVED,
            ADMISSION_CANCELED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
    pub(crate) fn handoff_sender(&self) -> mpsc::SyncSender<Arc<Mutex<H>>> {
        self.handoffs.clone()
    }
}

pub(crate) enum WorkerFailure {
    Preparation(ArenaError),
    ActorStart(TeacherSessionStartErrorV2),
    Shutdown(ArenaError),
    TeacherWorkerJoin {
        actor_id: String,
        error: rz_search::cpu_checker::CheckerError,
    },
    TeacherRegistryPoisoned {
        actor_id: String,
    },
    Retirement(ArenaError),
    HandoffDisconnected,
    HandoffMismatch,
    RootClosedBeforeStartup,
    WorkerPanicked,
    RetirementSealRejected,
}

impl WorkerFailure {
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Preparation(_) => "preparation",
            Self::ActorStart(TeacherSessionStartErrorV2::Admission(_)) => "actor_admission",
            Self::ActorStart(TeacherSessionStartErrorV2::Binding(_)) => "actor_binding",
            Self::ActorStart(TeacherSessionStartErrorV2::BeforeSpawn(_)) => "actor_before_spawn",
            Self::ActorStart(TeacherSessionStartErrorV2::AfterSpawn { .. }) => "actor_after_spawn",
            Self::Shutdown(_) => "shutdown",
            Self::TeacherWorkerJoin { .. } => "teacher_worker_join",
            Self::TeacherRegistryPoisoned { .. } => "teacher_registry_poisoned",
            Self::Retirement(_) => "retirement",
            Self::HandoffDisconnected => "handoff_disconnected",
            Self::HandoffMismatch => "handoff_mismatch",
            Self::RootClosedBeforeStartup => "root_closed_before_startup",
            Self::WorkerPanicked => "worker_panicked",
            Self::RetirementSealRejected => "retirement_seal_rejected",
        }
    }
}

/// Borrowed bounded failure observations. The caller must use a nonblocking
/// lock on the worker before reading them; this value does not consume or
/// reclassify any original error and is not completion evidence.
pub(crate) struct WorkerFailureObservation<'a> {
    pub(crate) primary: Option<&'a WorkerFailure>,
    pub(crate) additional: &'a [WorkerFailure],
    pub(crate) omitted: u64,
}

/// Worker methods must be called from a separate owner, outside manager/root
/// locks. The caller retains this value on every failure or unknown completion.
pub(crate) struct CollectionWorkerCoreV2<H> {
    admission_receiver: mpsc::Receiver<Arc<AdmissionSlot>>,
    handoff_receiver: mpsc::Receiver<Arc<Mutex<H>>>,
    handoff_matches: HandoffMatches<H>,
    revoked: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    admission_phase: Arc<AtomicU8>,
    startup: Arc<OnceLock<RoventStartupControl>>,
    manifest: Arc<RoventCorpusManifestV2>,
    roster: Vec<RoventTeacherRosterEntryV2>,
    slot: Option<Arc<AdmissionSlot>>,
    handoff: Option<Arc<Mutex<H>>>,
    session: Option<RoventCollectionTeacherSessionV2>,
    retirement: Option<TeacherSessionRetirementSealV2>,
    custodian: TeacherRecoveryCustodianV2,
    tokens: BTreeMap<String, RoventTeacherRetirementTokenV2>,
    teacher_observations: BTreeMap<String, RoventTeacherRetirementObservationV2>,
    diagnostics: BTreeMap<String, rz_search::external_cpu::collection_teacher::TeacherDiagnostics>,
    next_actor: usize,
    shutdown_requested: bool,
    failure: Option<WorkerFailure>,
    additional_failures: Vec<WorkerFailure>,
    omitted_failures: u64,
}

/// Outcome of the worker loop only. The returned worker still owns any
/// unresolved session, original errors, tokens and native handoff.
pub(crate) enum CollectionWorkerExitV2 {
    NoAdmission,
    AdmissionCanceled,
    RetirementAndHandoffObserved,
    Deadline,
    Panicked(Vec<Box<dyn std::any::Any + Send>>),
}

impl<H: Send + 'static> CollectionWorkerCoreV2<H> {
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

    fn request_teacher_shutdown(&mut self, until: Instant) {
        // The session requests termination of every started actor even if
        // closing admission fails. Keep that failure and all original owners;
        // requesting shutdown is not physical completion.
        let failure = self.session.as_mut().and_then(|session| session.request_shutdown(until).err());
        self.shutdown_requested = true;
        if let Some(error) = failure {
            self.record_failure(WorkerFailure::Shutdown(error));
        }
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
            self.request_teacher_shutdown(cleanup_until);
        }
        if self.shutdown_requested {
            // Retired tokens keep the original physical evidence even when a
            // separate worker or registry error makes Collection unsuccessful.
            let mut retirement_failures = Vec::new();
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
                            let observation = token.observation();
                            // Preserve all independent fields, including a
                            // root observation error and diagnostics disposition,
                            // whether the original token later enters a seal or
                            // stays with this worker after an error.
                            self.teacher_observations.insert(item.actor_id.clone(), observation.clone());
                            if let Some(Err(error)) = &observation.worker_join {
                                retirement_failures.push(WorkerFailure::TeacherWorkerJoin {
                                    actor_id: item.actor_id.clone(),
                                    error: error.clone(),
                                });
                            }
                            if observation.registry_poisoned {
                                retirement_failures.push(WorkerFailure::TeacherRegistryPoisoned {
                                    actor_id: item.actor_id.clone(),
                                });
                            }
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
            for failure in retirement_failures {
                self.record_failure(failure);
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
        (self.handoff_matches)(&mut *original, &slot.admission, seal)
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

    pub(crate) fn failure_observation(&self) -> WorkerFailureObservation<'_> {
        WorkerFailureObservation {
            primary: self.failure.as_ref(),
            additional: &self.additional_failures,
            omitted: self.omitted_failures,
        }
    }

    /// Actual per-actor tokens remain available on failure. A confirmed
    /// physical retirement is not turned into Unknown merely because its
    /// worker or registry reported a separate error.
    pub(crate) fn teacher_retirements(&self) -> &BTreeMap<String, RoventTeacherRetirementTokenV2> {
        &self.tokens
    }

    /// Read-only observations are not a retirement token or authority.
    /// Unavailable diagnostics stay distinct from any bytes already recovered.
    pub(crate) fn teacher_observations(&self) -> &BTreeMap<String, RoventTeacherRetirementObservationV2> {
        &self.teacher_observations
    }

    pub(crate) fn diagnostics(
        &self,
    ) -> &BTreeMap<String, rz_search::external_cpu::collection_teacher::TeacherDiagnostics> {
        &self.diagnostics
    }

    /// Transfer only bytes already recovered from the original worker. The
    /// caller must separately confirm that the outer worker thread has exited.
    pub(crate) fn take_diagnostics(
        &mut self,
    ) -> BTreeMap<String, rz_search::external_cpu::collection_teacher::TeacherDiagnostics> {
        std::mem::take(&mut self.diagnostics)
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
    ) -> CollectionWorkerExitV2 {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loop {
                if stop.load(Ordering::Acquire) {
                    self.revoked.store(true, Ordering::Release);
                    let _ = self.admission_phase.compare_exchange(
                        ADMISSION_RESERVED,
                        ADMISSION_CANCELED,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                }
                self.poll();
                if self.retirement_matches_handoff() {
                    break CollectionWorkerExitV2::RetirementAndHandoffObserved;
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
                    break CollectionWorkerExitV2::NoAdmission;
                }
                if stop.load(Ordering::Acquire)
                    && self.slot.is_none()
                    && self.admission_phase.load(Ordering::Acquire) == ADMISSION_CLOSED
                {
                    break CollectionWorkerExitV2::NoAdmission;
                }
                if stop.load(Ordering::Acquire)
                    && self.slot.is_none()
                    && self.admission_phase.load(Ordering::Acquire) == ADMISSION_CANCELED
                {
                    break CollectionWorkerExitV2::AdmissionCanceled;
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
                        ADMISSION_CANCELED,
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
                        self.request_teacher_shutdown(deadline);
                    }
                    break CollectionWorkerExitV2::Deadline;
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
                // Invalidate an in-flight install before this worker exits;
                // otherwise an installer can publish into an abandoned queue.
                let _ = self.admission_phase.compare_exchange(
                    ADMISSION_RESERVED,
                    ADMISSION_CANCELED,
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
                let deadline = self.deadline().map_or(until, |root| root.min(until));
                let shutdown = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.request_teacher_shutdown(deadline);
                }));
                self.record_failure(WorkerFailure::WorkerPanicked);
                let mut payloads = vec![payload];
                if let Err(second) = shutdown {
                    payloads.push(second);
                }
                CollectionWorkerExitV2::Panicked(payloads)
            }
        };
        exit
    }
}
