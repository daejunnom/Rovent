//! Shared bounded, nonblocking single-root /2 handoff. The manager owns the observer during
//! callbacks; this worker never accesses its producer while that root is live.
//! Teacher process retirement and native handoff remain separate observations.
#![allow(dead_code)]
use super::teacher::v2_session::{
    RoventCollectionTeacherSessionV2, RoventTeacherRosterEntryV2, TeacherSessionPrepareOwnerV2,
    TeacherSessionRetirementSealV2, TeacherSessionStartErrorV2,
};
use super::teacher::lock_v2_row_pump::{PumpItemV2, RetiredRowPumpV2};
use super::{ArenaError, RoventCollectionOwnerLockV2, check, invalid};
use rz_search::external_cpu::ExternalGoConstraint;
use rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2;
use rz_uci::pals_adaptive::{
    AdaptiveManagerError, RoventCollectionRootAdmissionV2, RoventTeacherScopePhysicalAckV2,
    RoventTeacherScopeWorkerAckV2,
};
use rz_uci::rovent_bootstrap::RoventStartupControl;
use rz_uci::rovent_external_launch::{
    RoventTeacherRetiredActorV2, RoventTeacherRetirementObservationV2,
    RoventTeacherRetirementTokenV2, RoventTeacherScopeAggregateAckV2,
    RoventTeacherScopeAggregateFailureV2, RoventTeacherWorkerTerminalTokenV2,
    TeacherRecoveryCustodianV2,
};
use std::collections::{BTreeMap, VecDeque};
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
const ADMISSION_COMMITTING: u8 = 5;

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
    installed_published: AtomicBool,
    installed_admission: OnceLock<RoventCollectionRootAdmissionV2>,
}

/// Exclusive transition for committing rows of the original single root.
/// It excludes a new observer admission; it does not prove native completion,
/// teacher retirement, or publication on its own.
pub(crate) struct RoventCollectionCommitGateV2<'a, H> {
    core: &'a CollectionBridgeCoreV2<H>,
    admission: RoventCollectionRootAdmissionV2,
}

impl<H> RoventCollectionCommitGateV2<'_, H> {
    pub(crate) fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV2) -> bool {
        self.core.admission_phase.load(Ordering::Acquire) == ADMISSION_COMMITTING
            && self.admission.same_original_admission(admission)
    }

    pub(crate) fn owner(&self) -> &RoventCollectionOwnerLockV2 {
        &self.core.owner
    }
}

impl<H> Drop for RoventCollectionCommitGateV2<'_, H> {
    fn drop(&mut self) {
        // A changed phase is not replaced: an unexpected owner transition
        // must remain observable instead of being overwritten.
        let _ = self.core.admission_phase.compare_exchange(
            ADMISSION_COMMITTING, ADMISSION_INSTALLED, Ordering::AcqRel, Ordering::Acquire,
        );
    }
}

impl<H: Send + 'static> CollectionBridgeCoreV2<H> {
    /// Actual installed admission, retained even after the bridge closes.
    /// This is custody identity only, not completion or publication evidence.
    pub(crate) fn installed_admission_for_recovery(&self) -> Option<&RoventCollectionRootAdmissionV2> {
        if !self.installed_published.load(Ordering::Acquire) {
            return None;
        }
        self.installed_admission.get()
    }
    /// The single-root bridge can no longer install another observer only
    /// after its admission has irrevocably closed. A reserved installation
    /// remains unknown and must never be represented as a no-capture root.
    pub(crate) fn closed_before_next_admission(&self, next: &RoventCollectionRootAdmissionV2) -> bool {
        if !self.revoked.load(Ordering::Acquire) {
            return false;
        }
        if !matches!(self.admission_phase.load(Ordering::Acquire), ADMISSION_INSTALLED | ADMISSION_COMMITTING)
            || !self.installed_published.load(Ordering::Acquire) {
            return false;
        }
        let Some(previous) = self.installed_admission.get() else { return false; };
        let old = previous.root_admission().authority();
        let new = next.root_admission().authority();
        !previous.same_original_admission(next)
            && previous.same_original_origin(next)
            && old.epoch == new.epoch
            && old.implementation == new.implementation
            && (new.game > old.game || (new.game == old.game && new.root > old.root))
    }

    pub(crate) fn try_begin_commit(
        &self,
        admission: &RoventCollectionRootAdmissionV2,
    ) -> Result<Option<RoventCollectionCommitGateV2<'_, H>>, ArenaError> {
        if !self.revoked.load(Ordering::Acquire) || !self.installed_published.load(Ordering::Acquire)
            || !self.installed_admission.get().is_some_and(|original| original.same_original_admission(admission)) {
            return Err(ArenaError::Invalid("commit does not match the original closed collection root".into()));
        }
        match self.admission_phase.compare_exchange(ADMISSION_INSTALLED, ADMISSION_COMMITTING, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => Ok(Some(RoventCollectionCommitGateV2 { core: self, admission: admission.clone() })),
            Err(ADMISSION_COMMITTING) => Ok(None),
            Err(_) => Err(ArenaError::Invalid("original collection admission is not commit-ready".into())),
        }
    }

    pub(crate) fn try_install_no_capture(
        &self,
        capability: &rz_uci::pals_adaptive::RoventNoCaptureRootCapabilityV2,
    ) -> Result<bool, ArenaError> {
        self.owner.try_install_whole_root_no_capture(capability)
    }

    pub(crate) fn try_finish_no_capture(
        &self,
        capability: &rz_uci::pals_adaptive::RoventNoCaptureRootCapabilityV2,
        receipt: &rz_uci::pals_adaptive::AdaptiveShutdownReceipt,
    ) -> Result<bool, ArenaError> {
        self.owner.try_finish_whole_root_no_capture(capability, receipt)
    }

    pub(crate) fn try_finish_no_capture_role_only(
        &self,
        capability: &rz_uci::pals_adaptive::RoventNoCaptureRootCapabilityV2,
        receipt: &rz_uci::pals_adaptive::RoleOnlyShutdownReceipt,
    ) -> Result<bool, ArenaError> {
        self.owner.try_finish_whole_root_no_capture_role_only(capability, receipt)
    }

    pub(crate) fn new(
        owner: RoventCollectionOwnerLockV2,
        manifest: Arc<RoventCorpusManifestLockV2>,
        roster: Vec<RoventTeacherRosterEntryV2>,
        handoff_matches: HandoffMatches<H>,
        handoff_submit: HandoffSubmit<H>,
    ) -> Result<(Arc<Self>, CollectionWorkerCoreV2<H>), ArenaError> {
        // Full /2 validation uses the original live input lock once admitted.
        manifest.registration.validate()?;
        check(
            owner.matches_manifest(&manifest)?,
            "V2 teacher and original corpus manifest owners differ",
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
            installed_published: AtomicBool::new(false),
            installed_admission: OnceLock::new(),
        });
        let worker = CollectionWorkerCoreV2 {
            owner: bridge.owner.clone(),
            pump_config: None,
            pump: None,
            pump_ready: VecDeque::with_capacity(16),
            pump_sender: None,
            pump_sender_failed: false,
            pump_rejected: None,
            pump_failed: false,
            admission_receiver,
            handoff_receiver,
            handoff_matches,
            handoff_submit,
            handoff_recovery: None,
            handoff_identity: None,
            handoff_recovery_failed: false,
            handoff_disconnected: false,
            revoked,
            failed,
            admission_phase,
            startup,
            manifest,
            roster,
            slot: None,
            handoff: None,
            pending_handoff: None,
            foreign_handoff: None,
            physical_ack: None,
            worker_ack: None,
            aggregate_ack: None,
            aggregate_failure: None,
            aggregate_release_failed: false,
            physical_ack_delivered: false,
            physical_ack_error: None,
            scope_busy_observations: 0,
            scope_close_failure_recorded: false,
            session: None,
            prepare_owner: None,
            unclaimed_scope_outcome: None,
            retirement: None,
            custodian: TeacherRecoveryCustodianV2::new(),
            tokens: BTreeMap::new(),
            worker_tokens: BTreeMap::new(),
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

pub(crate) type HandoffMatches<H> =
    fn(&mut H, &RoventCollectionRootAdmissionV2, &TeacherSessionRetirementSealV2) -> bool;
/// On rejection the same opaque roster aggregate returns to its owner.
pub(crate) type HandoffSubmit<H> =
    fn(&mut H, RoventTeacherScopeAggregateAckV2) -> Result<(), RoventTeacherScopeAggregateAckV2>;
/// Custody recovery only. It grants no completion or publication evidence.
pub(crate) type HandoffRecovery<H> = fn(
    &RoventCollectionRootAdmissionV2,
) -> Result<Option<Arc<Mutex<H>>>, &'static str>;
pub(crate) type HandoffIdentity<H> = fn(&mut H, &RoventCollectionRootAdmissionV2) -> bool;

pub(crate) struct CoreAdmissionFailure {
    pub(crate) error: AdaptiveManagerError,
    pub(crate) admission: RoventCollectionRootAdmissionV2,
    pub(crate) owners: Box<dyn std::any::Any + Send>,
    /// True only before the original observer install callback was entered.
    pub(crate) before_start: bool,
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
        self.startup.set(control.clone()).map_err(|_| {
            AdaptiveManagerError::Invalid("Collection V2 startup control already registered")
        })
    }
    pub(crate) fn admit(
        &self,
        admission: RoventCollectionRootAdmissionV2,
        install: impl FnOnce(
            RoventCollectionRootAdmissionV2,
            &RoventCollectionOwnerLockV2,
        ) -> Result<(), CoreAdmissionFailure>,
    ) -> Result<(), CoreAdmissionFailure> {
        let reject = |error, admission| CoreAdmissionFailure {
            error,
            admission,
            owners: Box::new(()),
            before_start: true,
        };
        let reject_started = |error, admission| CoreAdmissionFailure {
            error, admission, owners: Box::new(()), before_start: false,
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
            // The phase/owner may belong to an earlier or concurrent attempt;
            // this guard alone does not prove this root never installed.
            return Err(reject_started(AdaptiveManagerError::AdmissionClosed, admission));
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
                    return Err(reject_started(
                        AdaptiveManagerError::Deadline,
                        slot.admission.clone(),
                    ));
                }
                if slot
                    .state
                    .compare_exchange(RESERVING, INSTALLED, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    return Err(reject_started(
                        AdaptiveManagerError::Deadline,
                        slot.admission.clone(),
                    ));
                }
                if self.installed_admission.set(slot.admission.clone()).is_err() {
                    return Err(reject_started(AdaptiveManagerError::Invalid("Collection V2 original admission already installed"), slot.admission.clone()));
                }
                self.installed_published.store(true, Ordering::Release);
                Ok(())
            }
            Err(original) => {
                slot.state.store(REJECTED, Ordering::Release);
                Err(CoreAdmissionFailure {
                    error: original.error,
                    admission: original.admission,
                    owners: original.owners,
                    before_start: false,
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
    Shutdown(rz_uci::pals_adaptive::AdaptiveCollectionError),
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
    HandoffRecovery(&'static str),
    RootClosedBeforeStartup,
    WorkerPanicked,
    RetirementSealRejected,
    AggregateRejected(&'static str),
    RowPump(ArenaError),
    RowPumpInputRejected,
}

impl WorkerFailure {
    /// Diagnostic projection only; original errors and recovery owners remain
    /// in the worker. Limit the emitted text without changing failure state.
    pub(crate) fn diagnostic(&self) -> String {
        let detail = match self {
            Self::Preparation(error) | Self::Retirement(error) | Self::RowPump(error) => format!("{error:?}"),
            Self::ActorStart(TeacherSessionStartErrorV2::Admission(error)) => format!("{error:?}"),
            Self::ActorStart(TeacherSessionStartErrorV2::Binding(error)) => format!("{error:?}"),
            Self::ActorStart(TeacherSessionStartErrorV2::BeforeSpawn(cause))
            | Self::ActorStart(TeacherSessionStartErrorV2::AfterSpawn { cause, .. }) => format!("{cause:?}"),
            Self::Shutdown(error) => format!("{error:?}"),
            Self::TeacherWorkerJoin { actor_id, error } => format!("actor={actor_id} error={error:?}"),
            Self::TeacherRegistryPoisoned { actor_id } => format!("actor={actor_id}"),
            Self::HandoffRecovery(detail) | Self::AggregateRejected(detail) => (*detail).to_owned(),
            _ => String::new(),
        };
        detail.chars().take(4096).collect()
    }
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
            Self::HandoffRecovery(_) => "handoff_recovery",
            Self::RootClosedBeforeStartup => "root_closed_before_startup",
            Self::WorkerPanicked => "worker_panicked",
            Self::RetirementSealRejected => "retirement_seal_rejected",
            Self::AggregateRejected(_) => "aggregate_rejected",
            Self::RowPump(_) => "row_pump",
            Self::RowPumpInputRejected => "row_pump_input_rejected",
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
    owner: RoventCollectionOwnerLockV2,
    pump_config: Option<(ExternalGoConstraint, Duration)>,
    pump: Option<RetiredRowPumpV2>,
    pump_ready: VecDeque<PumpItemV2>,
    pump_sender: Option<mpsc::SyncSender<PumpItemV2>>,
    pump_sender_failed: bool,
    pump_rejected: Option<Box<(ArenaError, super::RoventRetiredLockV2Record)>>,
    pump_failed: bool,
    admission_receiver: mpsc::Receiver<Arc<AdmissionSlot>>,
    handoff_receiver: mpsc::Receiver<Arc<Mutex<H>>>,
    handoff_matches: HandoffMatches<H>,
    handoff_submit: HandoffSubmit<H>,
    handoff_recovery: Option<HandoffRecovery<H>>,
    handoff_identity: Option<HandoffIdentity<H>>,
    handoff_recovery_failed: bool,
    handoff_disconnected: bool,
    revoked: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    admission_phase: Arc<AtomicU8>,
    startup: Arc<OnceLock<RoventStartupControl>>,
    manifest: Arc<RoventCorpusManifestLockV2>,
    roster: Vec<RoventTeacherRosterEntryV2>,
    slot: Option<Arc<AdmissionSlot>>,
    handoff: Option<Arc<Mutex<H>>>,
    // Never discard an unexamined or different-root original sent through
    // the channel. A single foreign handoff closes further channel reads.
    pending_handoff: Option<Arc<Mutex<H>>>,
    foreign_handoff: Option<Arc<Mutex<H>>>,
    physical_ack: Option<RoventTeacherScopePhysicalAckV2>,
    worker_ack: Option<RoventTeacherScopeWorkerAckV2>,
    aggregate_ack: Option<RoventTeacherScopeAggregateAckV2>,
    aggregate_failure: Option<Box<RoventTeacherScopeAggregateFailureV2>>,
    aggregate_release_failed: bool,
    physical_ack_delivered: bool,
    physical_ack_error: Option<String>,
    scope_busy_observations: u64,
    scope_close_failure_recorded: bool,
    session: Option<RoventCollectionTeacherSessionV2>,
    prepare_owner: Option<Box<TeacherSessionPrepareOwnerV2>>,
    unclaimed_scope_outcome:
        Option<rz_search::external_cpu::process_scope::CollectionScopePhysicalOutcome>,
    retirement: Option<TeacherSessionRetirementSealV2>,
    custodian: TeacherRecoveryCustodianV2,
    tokens: BTreeMap<String, RoventTeacherRetirementTokenV2>,
    // Original joined workers which do not yet have physical retirement.
    // Never drop or replace them on a no-child or unknown observation.
    worker_tokens: BTreeMap<String, RoventTeacherWorkerTerminalTokenV2>,
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
    /// Only the process scope proof was transferred to the same root. Actor
    /// worker join, Collection success and publication remain separate.
    PhysicalAckDelivered,
    /// The original aggregate was handed off. Semantic success, publication,
    /// native completion and this bridge thread's join remain separate.
    AggregateHandoffObserved,
    Deadline,
    Panicked(Vec<Box<dyn std::any::Any + Send>>),
}

/// A read-only observation of the original handoff is not a transfer or a
/// completion receipt. In particular, temporary lock contention is not a
/// collection failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HandoffObservationErrorV2 {
    Absent,
    Busy,
    Poisoned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PumpSendOutcomeV2 { Empty, Sent, Full }

impl<H: Send + 'static> CollectionWorkerCoreV2<H> {
    /// Install before the worker starts; P-C keeps its existing channel path.
    pub(crate) fn configure_handoff_recovery(&mut self, recovery: HandoffRecovery<H>, identity: HandoffIdentity<H>) {
        self.handoff_recovery = Some(recovery);
        self.handoff_identity = Some(identity);
    }
    /// Configure from the registered launch plan before this worker starts.
    /// This does not begin a teacher or consume any source row.
    pub(crate) fn configure_row_pump(
        &mut self,
        constraint: ExternalGoConstraint,
        soft_quantum: Duration,
    ) -> Result<(), ArenaError> {
        check(self.slot.is_none() && self.pump_config.is_none(), "row pump already configured or started")?;
        check(soft_quantum >= rz_search::external_cpu::EXTERNAL_MIN_QUANTUM, "teacher quantum below minimum")?;
        self.pump_config = Some((constraint, soft_quantum));
        Ok(())
    }

    /// Unpublished and rejected rows remain in this original worker custody.
    pub(crate) fn has_unpublished_pump_rows(&self) -> bool {
        self.pump.as_ref().is_some_and(|pump| pump.len() != 0)
            || !self.pump_ready.is_empty() || self.pump_rejected.is_some()
    }

    pub(crate) fn pending_pump_rows(&self) -> usize {
        self.pump.as_ref().map_or(0, RetiredRowPumpV2::len)
            + self.pump_ready.len() + usize::from(self.pump_rejected.is_some())
    }

    pub(crate) fn first_pump_item_mut(&mut self) -> Option<&mut PumpItemV2> {
        self.pump_ready.front_mut()
    }

    /// Install the original bounded consumer before the worker starts.
    /// The receiver must retain every item until publication or recovery.
    pub(crate) fn configure_pump_sender(&mut self, sender: mpsc::SyncSender<PumpItemV2>) -> Result<(), ArenaError> {
        check(self.slot.is_none() && self.pump_sender.is_none(), "pump sender must be registered before root admission")?;
        self.pump_sender = Some(sender);
        Ok(())
    }

    /// Transfer one exact row to a bounded publisher channel. A full or
    /// disconnected channel restores the original item at the front before
    /// returning. This does not itself establish publication or quiescence.
    pub(crate) fn try_send_first_pump_item(
        &mut self,
        sender: &mpsc::SyncSender<PumpItemV2>,
    ) -> Result<PumpSendOutcomeV2, ArenaError> {
        let Some(item) = self.pump_ready.front() else { return Ok(PumpSendOutcomeV2::Empty); };
        let row = match item {
            PumpItemV2::Unlabeled { row } | PumpItemV2::Prepared { row, .. }
            | PumpItemV2::Committed { row, .. } | PumpItemV2::Failed { row, .. } => row,
        };
        let admission = self.slot.as_ref().ok_or_else(|| invalid("P/C or Own pump root admission absent"))?;
        check(row.admission.same_original_admission(&admission.admission), "pump row belongs to a different original root")?;
        let item = self.pump_ready.pop_front().expect("checked original pump item");
        match sender.try_send(item) {
            Ok(()) => Ok(PumpSendOutcomeV2::Sent),
            Err(mpsc::TrySendError::Full(item)) => {
                self.pump_ready.push_front(item);
                Ok(PumpSendOutcomeV2::Full)
            }
            Err(mpsc::TrySendError::Disconnected(item)) => {
                self.pump_ready.push_front(item);
                Err(invalid("original pump publisher channel disconnected"))
            }
        }
    }

    pub(crate) fn pump_ready_counts(&self) -> (usize, usize, usize, usize) {
        let mut unlabeled = 0;
        let mut prepared = 0;
        let mut failed = 0;
        let mut committed = 0;
        for item in &self.pump_ready {
            match item {
                PumpItemV2::Unlabeled { .. } => unlabeled += 1,
                PumpItemV2::Prepared { sealed: Some(_), .. } => committed += 1,
                PumpItemV2::Prepared { sealed: None, .. } => prepared += 1,
                PumpItemV2::Committed { .. } => committed += 1,
                PumpItemV2::Failed { .. } => failed += 1,
            }
        }
        (unlabeled, prepared, committed, failed)
    }

    fn poll_row_pump(&mut self) {
        let Some((constraint, quantum)) = self.pump_config else { return; };
        if self.session.is_none() || self.pump_failed || (self.failure.is_some() && self.pump.is_none()) { return; }
        if self.pump.is_none() {
            let session = self.session.as_mut().expect("original teacher session");
            match RetiredRowPumpV2::new(session, 16, constraint, quantum) {
                Ok(pump) => self.pump = Some(pump),
                Err(error) => { self.pump_failed = true; self.record_failure(WorkerFailure::RowPump(error)); return; }
            }
        }
        if self.pump_rejected.is_some() { return; }
        let pump = self.pump.as_mut().expect("original session pump");
        // The output and input together remain bounded. Never remove a row
        // from its original outbox if this worker cannot retain it.
        if self.pump_ready.len() < 16 {
            if let Some(item) = pump.take_ready() { self.pump_ready.push_back(item); }
        }
        if self.pump_ready.len() + pump.len() < 16 {
            match self.owner.take_retired() {
                Ok(Some(row)) => {
                    if let Err(original) = pump.enqueue(row) {
                        self.pump_rejected = Some(original);
                        self.pump_failed = true;
                        self.record_failure(WorkerFailure::RowPumpInputRejected);
                        return;
                    }
                }
                Ok(None) => {}
                Err(error) => { self.pump_failed = true; self.record_failure(WorkerFailure::RowPump(error)); return; }
            }
        }
        let session = self.session.as_mut().expect("same original teacher session");
        if let Err(error) = pump.poll(session) {
            self.pump_failed = true;
            self.record_failure(WorkerFailure::RowPump(error));
        }
    }

    fn poll_pump_sender(&mut self) {
        if self.pump_sender_failed { return; }
        let Some(sender) = self.pump_sender.clone() else { return; };
        if let Err(error) = self.try_send_first_pump_item(&sender) {
            // The original item was restored before reporting the failure.
            self.pump_sender_failed = true;
            self.record_failure(WorkerFailure::RowPump(error));
        }
    }

    /// Clone the actual handoff received by this worker.  This is solely an
    /// ownership reference; it does not prove worker join, teacher closure,
    /// settlement or publication.  Callers must first join the original
    /// outer worker and retain its core.
    pub(crate) fn original_handoff(&self) -> Option<Arc<Mutex<H>>> {
        self.handoff.as_ref().map(Arc::clone)
    }

    pub(crate) fn try_handoff_observation<T>(
        &self,
        observe: impl FnOnce(&H) -> T,
    ) -> Result<T, HandoffObservationErrorV2> {
        let handoff = self
            .handoff
            .as_ref()
            .ok_or(HandoffObservationErrorV2::Absent)?;
        let handoff = handoff.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => HandoffObservationErrorV2::Busy,
            std::sync::TryLockError::Poisoned(_) => HandoffObservationErrorV2::Poisoned,
        })?;
        Ok(observe(&handoff))
    }

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
        let failure = self
            .session
            .as_mut()
            .and_then(|session| session.request_shutdown(until).err());
        self.shutdown_requested = true;
        if let Some(error) = failure {
            if error.code == "TeacherScopeBusy" {
                // Contention is retried by the physical observation path;
                // preserve its count without declaring a semantic failure.
                self.scope_busy_observations = self.scope_busy_observations.saturating_add(1);
            } else {
                self.scope_close_failure_recorded = true;
                self.record_failure(WorkerFailure::Shutdown(error));
            }
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
            REJECTED => {
                self.poll_handoff();
                self.try_transfer_physical_ack();
                return;
            }
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
                Err(rejected) => {
                    self.prepare_owner = rejected.owner;
                    self.record_failure(WorkerFailure::Preparation(rejected.error));
                }
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
        self.poll_row_pump();
        self.poll_pump_sender();
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
                    if let Some(original) = session.take_actor_diagnostics_after_join(&item.actor_id) {
                        self.diagnostics.insert(item.actor_id.clone(), original);
                    }
                    if self.worker_tokens.contains_key(&item.actor_id) {
                        continue;
                    }
                    match session.try_finish_worker_actor(&item.actor_id) {
                        Ok(Some(token)) => {
                            let observation = token.observation();
                            // Preserve all independent fields, including a
                            // root observation error and diagnostics disposition,
                            // whether the original token later enters a seal or
                            // stays with this worker after an error.
                            self.teacher_observations
                                .insert(item.actor_id.clone(), observation.clone());
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
                            // Keep original terminal and lifecycle intact for
                            // the same-scope worker/physical aggregate.
                            self.worker_tokens.insert(item.actor_id.clone(), token);
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
        self.poll_handoff();
        self.try_transfer_physical_ack();
    }

    fn poll_handoff(&mut self) {
        if self.handoff.is_none() && self.pending_handoff.is_none() && self.foreign_handoff.is_none() {
            match self.handoff_receiver.try_recv() {
                Ok(handoff) if self.handoff_identity.is_some() => self.pending_handoff = Some(handoff),
                Ok(handoff) => self.handoff = Some(handoff),
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) if self.handoff_recovery.is_none() => {
                    self.record_failure(WorkerFailure::HandoffDisconnected)
                }
                Err(mpsc::TryRecvError::Disconnected) => self.handoff_disconnected = true,
            }
        }
        if let (Some(identity), Some(slot), Some(pending)) =
            (self.handoff_identity, self.slot.as_ref(), self.pending_handoff.as_ref()) {
            let matched = match pending.try_lock() {
                Ok(mut handoff) => Some(Ok(identity(&mut *handoff, &slot.admission))),
                Err(std::sync::TryLockError::WouldBlock) => None,
                Err(std::sync::TryLockError::Poisoned(_)) => Some(Err(())),
            };
            match matched {
                Some(Ok(true)) => self.handoff = self.pending_handoff.take(),
                Some(Ok(false)) => {
                    self.foreign_handoff = self.pending_handoff.take();
                    self.record_failure(WorkerFailure::HandoffMismatch);
                }
                Some(Err(())) => {
                    self.foreign_handoff = self.pending_handoff.take();
                    self.record_failure(WorkerFailure::HandoffRecovery("received handoff poisoned"));
                }
                None => {}
            }
        }
        if self.handoff.is_none() && !self.handoff_recovery_failed
            && (self.foreign_handoff.is_some() || self.handoff_disconnected) {
            if let (Some(recover), Some(slot)) = (self.handoff_recovery, self.slot.as_ref()) {
                match recover(&slot.admission) {
                    Ok(Some(original)) => self.handoff = Some(original),
                    Ok(None) => {}
                    Err(error) => {
                        self.handoff_recovery_failed = true;
                        self.record_failure(WorkerFailure::HandoffRecovery(error));
                    }
                }
            }
        }
    }

    /// Get the physical-only proof outside the handoff lock. In particular,
    /// a Collection semantic error or observer settlement error must not
    /// turn a confirmed teacher-process outcome into physical unknown.
    fn try_transfer_physical_ack(&mut self) {
        if self.physical_ack_delivered || self.handoff.is_none() {
            return;
        }
        if self.aggregate_failure.is_some() || self.aggregate_release_failed {
            return;
        }
        if self.aggregate_ack.is_some() {
            self.transfer_aggregate_ack();
            return;
        }
        let Some(slot) = self.slot.as_ref().cloned() else {
            return;
        };
        let rejected = slot.state.load(Ordering::Acquire) == REJECTED;
        if !rejected && !self.revoked.load(Ordering::Acquire) && self.failure.is_none() {
            return;
        }

        if self.physical_ack.is_none() {
            // A previous nonblocking close may have met contention; retry
            // without interpreting it as a physical failure. Started actors
            // were already signalled by request_teacher_shutdown.
            let close_error = self
                .session
                .as_mut()
                .and_then(|session| session.close_admission().err());
            if let Some(error) = close_error {
                if error.code == "TeacherScopeBusy" {
                    self.scope_busy_observations = self.scope_busy_observations.saturating_add(1);
                } else if !self.scope_close_failure_recorded {
                    self.scope_close_failure_recorded = true;
                    self.record_failure(WorkerFailure::Shutdown(error.clone()));
                }
                self.physical_ack_error = Some(error.to_string());
                return;
            }
            let result = if let Some(session) = &self.session {
                session.try_physical_retirement()
            } else if let Some(seal) = &self.retirement {
                seal.try_physical_retirement()
            } else if let Some(owner) = self.prepare_owner.as_deref() {
                match owner {
                    TeacherSessionPrepareOwnerV2::Claimed(capability) => capability
                        .close()
                        .and_then(|()| capability.try_physical_retirement()),
                    TeacherSessionPrepareOwnerV2::Unclaimed(scope) => {
                        // This issuer was rejected before claim and cannot be
                        // submitted to a worker. Keep its actual closed roster
                        // outcome independently of the root's no-scope seal.
                        if self.unclaimed_scope_outcome.is_none() {
                            scope.close_admission();
                            match scope.try_physical_retirement() {
                                Ok(Some(outcome)) => self.unclaimed_scope_outcome = Some(outcome),
                                Ok(None) => return,
                                Err(error) => {
                                    self.physical_ack_error = Some(error.to_string());
                                    return;
                                }
                            }
                        }
                        slot.admission.seal_unclaimed_teacher_scope().map(Some)
                    }
                }
            } else {
                slot.admission.seal_unclaimed_teacher_scope().map(Some)
            };
            match result {
                Ok(Some(ack)) => {
                    self.physical_ack = Some(ack);
                    self.physical_ack_error = None;
                }
                Ok(None) => return,
                Err(error) => {
                    self.physical_ack_error = Some(error.to_string());
                    return;
                }
            }
        }
        // Keep the original physical ACK in this worker until the independent
        // worker roster has been observed and same-member no-child custody is
        // resolved. NoScope is itself a one-shot proof that no compliant
        // scoped worker could have started.
        let Some(physical) = self.physical_ack.as_ref() else {
            return;
        };
        if !physical.is_sealed_no_scope() && self.worker_ack.is_none() {
            let result = if let Some(session) = &self.session {
                session.try_worker_retirement()
            } else if let Some(seal) = &self.retirement {
                seal.try_worker_retirement()
            } else if let Some(TeacherSessionPrepareOwnerV2::Claimed(capability)) =
                self.prepare_owner.as_deref()
            {
                capability.try_worker_retirement()
            } else {
                self.physical_ack_error =
                    Some("scoped physical proof has no original worker scope owner".into());
                return;
            };
            match result {
                Ok(Some(ack)) if ack.matches_admission(&slot.admission) => {
                    self.worker_ack = Some(ack)
                }
                Ok(Some(ack)) => {
                    // Preserve a mismatched original capability for diagnosis;
                    // never convert it into authority for another root.
                    self.worker_ack = Some(ack);
                    self.physical_ack_error = Some("worker roster admission mismatch".into());
                    return;
                }
                Ok(None) => return,
                Err(error) => {
                    self.physical_ack_error = Some(error.to_string());
                    return;
                }
            }
        }
        if !physical.is_sealed_no_scope() {
            let Some(workers) = self.worker_ack.as_ref() else {
                return;
            };
            if !workers.matches_admission(&slot.admission) {
                return;
            }
            let joined = workers.outcome().members().iter().filter(|(_, state)|
                *state == rz_search::external_cpu::process_scope::CollectionMemberWorkerState::Joined
            ).count();
            // A joined thread may still have an undelivered terminal; defer
            // the consuming builder until every original token is available.
            if joined != self.worker_tokens.len() {
                return;
            }
        }
        let identities: Vec<_> = self
            .worker_tokens
            .iter()
            .map(|(id, token)| (token.registration(), id.clone()))
            .collect();
        let physical = self.physical_ack.take().expect("original proof retained");
        let workers = self.worker_ack.take();
        let tokens = std::mem::take(&mut self.worker_tokens)
            .into_values()
            .collect();
        match RoventTeacherScopeAggregateAckV2::try_build(physical, workers, tokens) {
            Ok(ack) => {
                self.observe_retired_actors(&identities, ack.retired());
                if let Some(session) = &mut self.session {
                    if let Err(error) = session.release_retired_permits(&ack) {
                        self.aggregate_ack = Some(ack);
                        self.aggregate_release_failed = true;
                        self.record_failure(WorkerFailure::Retirement(error));
                        return;
                    }
                }
                if let Some(seal) = &mut self.retirement {
                    if let Err(error) = seal.release_retired_permits(&ack) {
                        self.aggregate_ack = Some(ack);
                        self.aggregate_release_failed = true;
                        self.record_failure(WorkerFailure::Retirement(error));
                        return;
                    }
                }
                self.aggregate_ack = Some(ack);
            }
            Err(failure) => {
                let reason = failure.reason;
                self.observe_retired_actors(&identities, &failure.retired);
                self.aggregate_failure = Some(Box::new(failure));
                self.record_failure(WorkerFailure::AggregateRejected(reason));
                return;
            }
        }
        self.transfer_aggregate_ack();
    }

    fn observe_retired_actors(
        &mut self,
        identities: &[(
            rz_search::external_cpu::collection_teacher::TeacherWorkerRegistration,
            String,
        )],
        retired: &[RoventTeacherRetiredActorV2],
    ) {
        for actor in retired {
            let (owner, observation) = match actor {
                RoventTeacherRetiredActorV2::Physical(token) => {
                    (token.owner(), token.observation())
                }
                RoventTeacherRetiredActorV2::NoChild(token) => (token.owner(), token.observation()),
            };
            let registration = owner.worker().registration();
            if let Some((_, actor_id)) = identities.iter().find(|(r, _)| *r == registration) {
                let poison_new = observation.registry_poisoned
                    && !self
                        .teacher_observations
                        .get(actor_id)
                        .is_some_and(|old| old.registry_poisoned);
                self.teacher_observations
                    .insert(actor_id.clone(), observation.clone());
                if poison_new {
                    self.record_failure(WorkerFailure::TeacherRegistryPoisoned {
                        actor_id: actor_id.clone(),
                    });
                }
            } else {
                // Preserve the aggregate and flag the missing original actor
                // mapping rather than discarding the owner's observation.
                self.record_failure(WorkerFailure::HandoffMismatch);
            }
        }
    }

    fn transfer_aggregate_ack(&mut self) {
        let Some(handoff) = &self.handoff else {
            return;
        };
        let Ok(mut original) = handoff.try_lock() else {
            return;
        };
        let Some(ack) = self.aggregate_ack.take() else {
            return;
        };
        match (self.handoff_submit)(&mut *original, ack) {
            Ok(()) => {
                self.physical_ack_delivered = true;
                if self.unclaimed_scope_outcome.is_some()
                    && matches!(
                        self.prepare_owner.as_deref(),
                        Some(TeacherSessionPrepareOwnerV2::Unclaimed(_))
                    )
                {
                    self.prepare_owner = None;
                }
            }
            Err(ack) => {
                self.aggregate_ack = Some(ack);
                self.physical_ack_error = Some("same-root handoff rejected aggregate proof".into());
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

    /// The proof only describes the original external-process scope. It is
    /// independent of native settlement and of Collection semantic success.
    pub(crate) fn physical_ack_observation(&self) -> (bool, bool, Option<&str>) {
        (
            self.physical_ack_delivered,
            self.physical_ack.is_some()
                || self.aggregate_ack.is_some()
                || self.aggregate_failure.is_some(),
            self.physical_ack_error.as_deref(),
        )
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
    pub(crate) fn teacher_observations(
        &self,
    ) -> &BTreeMap<String, RoventTeacherRetirementObservationV2> {
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
                if self.retirement_matches_handoff() && self.physical_ack_delivered {
                    break CollectionWorkerExitV2::RetirementAndHandoffObserved;
                }
                if self.physical_ack_delivered
                    && (self.session.is_some() || self.retirement.is_some())
                {
                    break CollectionWorkerExitV2::AggregateHandoffObserved;
                }
                // A no-session or rejected-claim path can prove that the
                // process scope has no unresolved child. Keep its errors and
                // original owners in this worker; do not claim actor join or
                // Collection success from this physical-only transfer.
                if self.physical_ack_delivered
                    && self.session.is_none()
                    && self.retirement.is_none()
                {
                    break CollectionWorkerExitV2::PhysicalAckDelivered;
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
