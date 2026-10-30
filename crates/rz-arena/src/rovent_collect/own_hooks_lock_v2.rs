//! Typed adapter around the shared bounded single-root Collection /2 worker.
//! This layer does not combine P-C and Own manager or handoff authority.
use super::collection_hooks_core_v2::{
    CollectionBridgeCoreV2, CollectionWorkerCoreV2, CoreAdmissionFailure, RoventCollectionCommitGateV2,
};
use super::teacher::v2_session::{RoventTeacherRosterEntryV2, TeacherSessionRetirementSealV2};
use super::{ArenaError, RoventCollectionOwnerLockV2};
use rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2;
use rz_uci::pals_adaptive::{AdaptiveCollectionBoundaryV2, AdaptiveCollectionError,
    AdaptiveCollectionStage, AdaptiveManagerError, AdaptiveShutdownReceipt,
    RoventCollectionRootAdmissionV2, RoventNoCaptureRootCapabilityV2};
use rz_uci::rovent_bootstrap::RoventStartupControl;
use rz_uci::rovent_bootstrap::nnuec_session::{
    RoventNnuecCollectionAdmissionFailureV2, RoventNnuecCollectionAdmissionPortV2,
    RoventNnuecCollectionAdmissionOutcomeV2, RoventNnuecCollectionHandoffV2,
    RoventNnuecCollectionHooksV2, RoventNnuecPendingCollectionV2, RoventPendingEvidenceDispositionV2,
    RoventNnuecRootHookFailureV2,
};
use rz_uci::rovent_bootstrap::nnuec_session::handoff_escrow_v2::{
    HandoffEscrowErrorV2, HandoffEscrowKeyV2,
};
use std::sync::{Arc, Mutex, OnceLock, mpsc, atomic::AtomicBool};
use std::thread::JoinHandle;
use std::time::Instant;

pub(crate) use super::collection_hooks_core_v2::CollectionWorkerExitV2 as RoventNnuecCollectionWorkerExitV2;
pub(crate) type RoventNnuecCollectionWorkerV2 =
    CollectionWorkerCoreV2<RoventNnuecCollectionHandoffV2>;

pub(crate) struct RoventNnuecCollectionBridgeV2 {
    core: Arc<CollectionBridgeCoreV2<RoventNnuecCollectionHandoffV2>>,
    owner: RoventCollectionOwnerLockV2,
    manifest: Arc<RoventCorpusManifestLockV2>,
    roster_identity: Vec<(String, usize, u32)>,
    worker: Mutex<Option<OwnRootWorkerV2>>,
    deferred: Mutex<DeferredCollectionHistoryV2>,
    pending_stopped: Mutex<PendingCollectionSummaryV2>,
    original_attempt: OnceLock<RoventCollectionRootAdmissionV2>,
    before_start_failure: OnceLock<AdaptiveManagerError>,
    unstarted_sealed: OnceLock<Arc<RoventNnuecPendingCollectionV2>>,
    first_pending_event: OnceLock<Arc<RoventNnuecPendingCollectionV2>>,
}

pub(crate) struct OwnRootWorkerV2 {
    original: Arc<Mutex<RoventNnuecCollectionWorkerV2>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<RoventNnuecCollectionWorkerExitV2>>,
    exit: Option<RoventNnuecCollectionWorkerExitV2>,
    panic: Option<Box<dyn std::any::Any + Send>>,
    separated_rows: std::collections::VecDeque<super::RoventRetiredLockV2Record>,
    separation_failed: bool,
    separation_complete: bool,
}

pub(crate) enum OwnRootSeparationV2 { Pending, Released { retained_rows: usize } }
pub(crate) enum OwnWorkerStartErrorV2 { BeforeStart(ArenaError), Unknown(ArenaError) }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OwnWorkerExitKindV2 { NoAdmission, AdmissionCanceled, Retirement, PhysicalAck, Aggregate, Deadline, Panicked }

impl OwnRootWorkerV2 {
    pub(crate) fn original_worker(&self) -> &Arc<Mutex<RoventNnuecCollectionWorkerV2>> { &self.original }
    pub(crate) fn take_separated_row(&mut self) -> Option<super::RoventRetiredLockV2Record> { self.separated_rows.pop_front() }
    pub(crate) fn remaining_rows(&self) -> usize { self.separated_rows.len() }
}

impl Drop for RoventNnuecCollectionBridgeV2 {
    fn drop(&mut self) {
        // The outer supervisor must take separated custody explicitly. If it
        // drops the bridge before doing so, the original worker, join handle,
        // diagnostics and rows remain quarantined rather than detached/lost.
        let slot = match self.worker.get_mut() {
            Ok(slot) => slot,
            Err(poison) => poison.into_inner(),
        };
        if let Some(original) = slot.take() { std::mem::forget(original); }
    }
}

struct DeferredCollectionHistoryV2 {
    retained: Vec<(RoventCollectionRootAdmissionV2, AdaptiveCollectionError)>,
    summarized: u64,
    first_summarized: Option<String>,
    summarized_reasons: Vec<(AdaptiveCollectionError, u64)>,
    other_summarized_reasons: u64,
}

#[derive(Clone)]
pub(crate) struct PendingCollectionSummaryV2 {
    pub(crate) first: Option<Arc<RoventNnuecPendingCollectionV2>>,
    pub(crate) summarized: u64,
    pub(crate) reasons: Vec<(Option<AdaptiveCollectionError>, AdaptiveCollectionError, u64)>,
    pub(crate) other: u64,
}

impl PendingCollectionSummaryV2 {
    fn new() -> Self {
        Self { first: None, summarized: 0, reasons: Vec::with_capacity(16), other: 0 }
    }
}

fn handoff_matches(
    handoff: &mut RoventNnuecCollectionHandoffV2,
    expected: &RoventCollectionRootAdmissionV2,
    seal: &TeacherSessionRetirementSealV2,
) -> bool {
    let (admission, _, settlement, fence) = handoff.parts_mut();
    admission.same_original_admission(expected)
        && seal.matches_admission(admission)
        && settlement.is_ok()
        && fence.permits_semantic_closure()
}

fn handoff_submit(
    handoff: &mut RoventNnuecCollectionHandoffV2,
    ack: rz_uci::rovent_external_launch::RoventTeacherScopeAggregateAckV2,
) -> Result<(), rz_uci::rovent_external_launch::RoventTeacherScopeAggregateAckV2> {
    handoff.submit_teacher_aggregate_ack(ack)
}

fn handoff_identity(
    handoff: &mut RoventNnuecCollectionHandoffV2,
    expected: &RoventCollectionRootAdmissionV2,
) -> bool {
    handoff.parts_mut().0.same_original_admission(expected)
}

/// A completed channel delivery can also be observed through the same
/// original root's bounded escrow. This returns custody, not completion
/// evidence; a successfully queued duplicate is the identical original Arc.
fn recover_root_handoff(
    admission: &RoventCollectionRootAdmissionV2,
) -> Result<Option<Arc<Mutex<RoventNnuecCollectionHandoffV2>>>, &'static str> {
    let lease = match HandoffEscrowKeyV2::try_recover_for_admission(admission) {
        Ok(Some(lease)) => lease,
        Ok(None) | Err(HandoffEscrowErrorV2::Busy | HandoffEscrowErrorV2::AlreadyLeased) => return Ok(None),
        Err(HandoffEscrowErrorV2::DuplicateAdmission) => return Err("duplicate original admission in handoff escrow"),
        Err(HandoffEscrowErrorV2::Poisoned) => return Err("handoff escrow poisoned"),
        Err(_) => return Err("handoff escrow recovery failed"),
    };
    let original = Arc::clone(lease.original());
    let should_recover = match original.try_lock() {
        Ok(mut handoff) => {
            let same = handoff.parts_mut().0.same_original_admission(admission);
            if !same { return Err("recovered handoff admission differs from worker"); }
            handoff.handoff_delivery().is_some()
        }
        Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
        Err(std::sync::TryLockError::Poisoned(_)) => return Err("recovered handoff poisoned"),
    };
    // Dropping this lease leaves the original in escrow; the worker now
    // holds a second Arc and the outer owner later performs the actual
    // transfer using the normal same-admission custody path.
    if should_recover { Ok(Some(original)) } else { Ok(None) }
}

impl RoventNnuecCollectionBridgeV2 {
    /// The caller invokes this only after resource admission. It preserves the
    /// original worker and handle, including every later join/panic outcome.
    pub(crate) fn start_worker(&self, worker: &Arc<Mutex<RoventNnuecCollectionWorkerV2>>, stop: Arc<AtomicBool>, deadline: Instant)
        -> Result<(), OwnWorkerStartErrorV2> {
        {
            let original = worker.try_lock().map_err(|_| OwnWorkerStartErrorV2::Unknown(super::invalid("Own original worker busy or poisoned")))?;
            if !self.core.matches_original_worker(&original) {
                return Err(OwnWorkerStartErrorV2::BeforeStart(super::invalid("Own worker differs from original bridge")));
            }
        }
        let mut slot = self.worker.try_lock().map_err(|_| OwnWorkerStartErrorV2::Unknown(super::invalid("Own worker custody busy or poisoned")))?;
        if slot.is_some() { return Err(OwnWorkerStartErrorV2::Unknown(super::invalid("Own worker already registered"))); }
        let original = Arc::clone(worker);
        let thread_stop = Arc::clone(&stop);
        let handle = std::thread::Builder::new().name("rovent-own-root-worker-v2".into())
            .spawn(move || match original.lock() {
                Ok(mut worker) => worker.run_until(thread_stop, deadline),
                Err(_) => panic!("Own original worker poisoned"),
            }).map_err(|error| OwnWorkerStartErrorV2::BeforeStart(ArenaError::Io(error.to_string())))?;
        *slot = Some(OwnRootWorkerV2 { original: Arc::clone(worker), stop, handle: Some(handle), exit: None,
            panic: None, separated_rows: std::collections::VecDeque::new(), separation_failed: false, separation_complete: false });
        Ok(())
    }

    /// Signal the exact worker started by this bridge, then join only its
    /// original handle. This narrow path proves no collection admission ever
    /// reached that worker; all admitted/unknown cases remain unresolved.
    pub(crate) fn try_finish_physical_no_admission(&self) -> Result<bool, AdaptiveManagerError> {
        {
            let slot = match self.worker.try_lock() {
                Ok(slot) => slot,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
                Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("Own worker custody poisoned")),
            };
            let Some(worker) = slot.as_ref() else { return Ok(false); };
            worker.stop.store(true, std::sync::atomic::Ordering::Release);
        }
        Ok(matches!(self.poll_worker()?, Some(OwnWorkerExitKindV2::NoAdmission)))
    }

    /// Observe only the original worker's physical completion. Collection
    /// settlement and unpublished rows remain with their own owners and are
    /// not turned into success by this physical-only observation.
    pub(crate) fn try_finish_physical_installed(&self) -> Result<bool, AdaptiveManagerError> {
        {
            let slot = match self.worker.try_lock() {
                Ok(slot) => slot,
                Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
                Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("Own worker custody poisoned")),
            };
            let Some(worker) = slot.as_ref() else { return Ok(false); };
            worker.stop.store(true, std::sync::atomic::Ordering::Release);
        }
        if !matches!(self.poll_worker()?, Some(OwnWorkerExitKindV2::Retirement | OwnWorkerExitKindV2::Aggregate)) {
            return Ok(false);
        }
        let slot = match self.worker.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("Own worker custody poisoned")),
        };
        let Some(worker) = slot.as_ref() else { return Ok(false); };
        if worker.handle.is_some() || worker.panic.is_some() { return Ok(false); }
        let core = match worker.original.try_lock() {
            Ok(core) => core,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("Own original worker poisoned")),
        };
        let Some(handoff) = core.original_handoff() else { return Ok(false); };
        let mut handoff = match handoff.try_lock() {
            Ok(handoff) => handoff,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("Own original handoff poisoned")),
        };
        let Some(original) = self.core.installed_admission_for_recovery() else { return Ok(false); };
        if !handoff.teacher_aggregate_ack().is_some_and(|ack| ack.matches_admission(original)) {
            return Ok(false);
        }
        let (admission, _, _, fence) = handoff.parts_mut();
        Ok(original.same_original_admission(admission) && fence.permits_semantic_closure())
    }

    pub(crate) fn try_prepared_for_same_owner(&self, previous: &Self) -> Result<Option<bool>, ArenaError> {
        if !self.owner.same_original_owner(&previous.owner) || !Arc::ptr_eq(&self.manifest, &previous.manifest)
            || self.roster_identity != previous.roster_identity || !self.core.same_original_startup(&previous.core)
            || self.core.installed_admission_for_recovery().is_some() {
            return Err(super::invalid("Own prepared root owner, startup, roster or admission mismatch"));
        }
        let slot = match self.worker.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(super::invalid("Own prepared worker custody poisoned")),
        };
        Ok(Some(slot.as_ref().is_some_and(|worker| worker.handle.as_ref().is_some_and(|handle| !handle.is_finished())
            && worker.exit.is_none() && worker.panic.is_none() && worker.separated_rows.is_empty() && !worker.separation_failed)))
    }

    /// Poll the original handle without detaching it; retain its complete
    /// exit or panic payload for the outer owner and receipt.
    pub(crate) fn poll_worker(&self) -> Result<Option<OwnWorkerExitKindV2>, AdaptiveManagerError> {
        let mut slot = match self.worker.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("Own worker custody poisoned")),
        };
        let worker = slot.as_mut().ok_or(AdaptiveManagerError::Invalid("Own original worker missing"))?;
        if worker.handle.as_ref().is_some_and(JoinHandle::is_finished) {
            match worker.handle.take().expect("finished worker retained").join() {
                Ok(exit) => worker.exit = Some(exit),
                Err(payload) => worker.panic = Some(payload),
            }
        }
        if worker.panic.is_some() { return Ok(Some(OwnWorkerExitKindV2::Panicked)); }
        Ok(worker.exit.as_ref().map(|exit| match exit {
            RoventNnuecCollectionWorkerExitV2::NoAdmission => OwnWorkerExitKindV2::NoAdmission,
            RoventNnuecCollectionWorkerExitV2::AdmissionCanceled => OwnWorkerExitKindV2::AdmissionCanceled,
            RoventNnuecCollectionWorkerExitV2::RetirementAndHandoffObserved => OwnWorkerExitKindV2::Retirement,
            RoventNnuecCollectionWorkerExitV2::PhysicalAckDelivered => OwnWorkerExitKindV2::PhysicalAck,
            RoventNnuecCollectionWorkerExitV2::AggregateHandoffObserved => OwnWorkerExitKindV2::Aggregate,
            RoventNnuecCollectionWorkerExitV2::Deadline => OwnWorkerExitKindV2::Deadline,
            RoventNnuecCollectionWorkerExitV2::Panicked(_) => OwnWorkerExitKindV2::Panicked,
        }))
    }

    /// Observe only this joined worker's original handoff; neither this
    /// observation nor a physical acknowledgement publishes the root rows.
    pub(crate) fn returned_handoff_settled(&self) -> Result<bool, AdaptiveManagerError> {
        let slot = match self.worker.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("Own worker custody poisoned")),
        };
        let Some(worker) = slot.as_ref() else { return Ok(false); };
        if worker.handle.is_some() || worker.panic.is_some()
            || !matches!(worker.exit.as_ref(), Some(RoventNnuecCollectionWorkerExitV2::RetirementAndHandoffObserved | RoventNnuecCollectionWorkerExitV2::AggregateHandoffObserved)) {
            return Ok(false);
        }
        let core = match worker.original.try_lock() {
            Ok(core) => core,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("Own original worker poisoned")),
        };
        if core.has_failure() { return Ok(false); }
        let Some(handoff) = core.original_handoff() else { return Ok(false); };
        let mut handoff = match handoff.try_lock() {
            Ok(handoff) => handoff,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("Own original handoff poisoned")),
        };
        if handoff.teacher_aggregate_ack().is_none() { return Ok(false); }
        let (admission, _, settlement, fence) = handoff.parts_mut();
        Ok(self.core.installed_admission_for_recovery().is_some_and(|original| original.same_original_admission(admission))
            && settlement.is_ok() && fence.permits_semantic_closure())
    }

    /// Join only the original worker and separate at most one same-root late
    /// row. The shared game gate is released only after the actual handoff,
    /// native fence and teacher aggregate have all been observed.
    pub(crate) fn try_separate_root_for_next(&self, admission: &RoventCollectionRootAdmissionV2, retained_row_limit: usize)
        -> Result<OwnRootSeparationV2, ArenaError> {
        if !self.core.installed_admission_for_recovery().is_some_and(|original| original.same_original_admission(admission)) {
            return Err(super::invalid("Own separation original admission mismatch"));
        }
        let mut slot = match self.worker.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnRootSeparationV2::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(super::invalid("Own worker custody poisoned")),
        };
        let worker = slot.as_mut().ok_or_else(|| super::invalid("Own original worker missing"))?;
        if worker.separation_failed { return Err(super::invalid("Own separation failed; originals retained")); }
        if worker.handle.as_ref().is_some_and(JoinHandle::is_finished) {
            match worker.handle.take().expect("finished worker retained").join() {
                Ok(exit) => worker.exit = Some(exit),
                Err(payload) => worker.panic = Some(payload),
            }
        }
        if worker.handle.is_some() { return Ok(OwnRootSeparationV2::Pending); }
        if worker.panic.is_some() { return Err(super::invalid("Own worker panic retained")); }
        if !matches!(worker.exit.as_ref(), Some(RoventNnuecCollectionWorkerExitV2::RetirementAndHandoffObserved | RoventNnuecCollectionWorkerExitV2::AggregateHandoffObserved)) {
            return Err(super::invalid("Own teacher/worker completion unknown"));
        }
        let core = match worker.original.try_lock() {
            Ok(core) => core,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnRootSeparationV2::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(super::invalid("Own original worker poisoned")),
        };
        let Some(handoff) = core.original_handoff() else { return Ok(OwnRootSeparationV2::Pending); };
        let mut handoff = match handoff.try_lock() {
            Ok(handoff) => handoff,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnRootSeparationV2::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(super::invalid("Own original handoff poisoned")),
        };
        if handoff.teacher_aggregate_ack().is_none() { return Ok(OwnRootSeparationV2::Pending); }
        let (original, _, settlement, fence) = handoff.parts_mut();
        if !original.same_original_admission(admission) || settlement.is_err() || !fence.permits_semantic_closure() {
            return Err(super::invalid("Own original handoff or native fence mismatch"));
        }
        drop(handoff);
        drop(core);
        let Some(status) = self.owner.try_status()? else { return Ok(OwnRootSeparationV2::Pending); };
        if !status.admitted || !status.settled || status.pending_count != 0 { return Ok(OwnRootSeparationV2::Pending); }
        if status.pending_retired_count != 0 {
            if worker.separated_rows.len() >= retained_row_limit { return Err(super::invalid("Own separated row capacity")); }
            worker.separated_rows.try_reserve(1).map_err(|_| super::invalid("Own separated row allocation"))?;
            match self.owner.try_take_settled_retired()? {
                Some(Some(row)) => {
                    if !row.admission.same_original_admission(admission) {
                        worker.separated_rows.push_back(row);
                        worker.separation_failed = true;
                        return Err(super::invalid("Own separated row belongs to another root"));
                    }
                    worker.separated_rows.push_back(row);
                    return Ok(OwnRootSeparationV2::Pending);
                }
                Some(None) | None => return Ok(OwnRootSeparationV2::Pending),
            }
        }
        if !self.core.try_release_game_root(admission)? { return Ok(OwnRootSeparationV2::Pending); }
        worker.separation_complete = true;
        Ok(OwnRootSeparationV2::Released { retained_rows: worker.separated_rows.len() })
    }

    /// Transfer the original joined worker and any separated rows together.
    /// The caller must also retain that root's consumer and publication state.
    pub(crate) fn try_take_separated_worker(&self, admission: &RoventCollectionRootAdmissionV2)
        -> Result<Option<OwnRootWorkerV2>, ArenaError> {
        if !self.core.installed_admission_for_recovery().is_some_and(|original| original.same_original_admission(admission)) {
            return Err(super::invalid("Own worker transfer original admission mismatch"));
        }
        let mut slot = match self.worker.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(super::invalid("Own worker custody poisoned")),
        };
        if !slot.as_ref().is_some_and(|worker| worker.separation_complete) { return Ok(None); }
        Ok(slot.take())
    }

    /// Transfer only a joined worker whose original loop observed the closed
    /// admission without taking a slot. This is custody, not physical or
    /// publication success; its exit and any diagnostic state stay intact.
    pub(crate) fn try_take_unadmitted_worker(&self) -> Result<Option<OwnRootWorkerV2>, ArenaError> {
        let mut slot = match self.worker.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(super::invalid("Own worker custody poisoned")),
        };
        let Some(worker) = slot.as_ref() else { return Ok(None); };
        if worker.handle.is_some() || worker.panic.is_some()
            || !matches!(worker.exit.as_ref(), Some(RoventNnuecCollectionWorkerExitV2::NoAdmission))
            || self.core.installed_admission_for_recovery().is_some() {
            return Ok(None);
        }
        Ok(slot.take())
    }

    pub(crate) fn try_take_separated_row(&self, admission: &RoventCollectionRootAdmissionV2)
        -> Result<Option<Option<super::RoventRetiredLockV2Record>>, ArenaError> {
        if !self.core.installed_admission_for_recovery().is_some_and(|original| original.same_original_admission(admission)) {
            return Err(super::invalid("Own separated row original admission mismatch"));
        }
        let mut slot = match self.worker.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(super::invalid("Own worker custody poisoned")),
        };
        let worker = slot.as_mut().ok_or_else(|| super::invalid("Own original worker missing"))?;
        if worker.separation_failed { return Err(super::invalid("Own separation failed; original rows retained")); }
        Ok(Some(worker.separated_rows.pop_front()))
    }
    pub(crate) fn matches_original_attempt(&self, admission: &RoventCollectionRootAdmissionV2) -> bool {
        self.original_attempt.get().is_some_and(|original| original.same_original_admission(admission))
    }
    pub(crate) fn closed_before_next_admission(&self, admission: &RoventCollectionRootAdmissionV2) -> bool {
        self.core.closed_before_next_admission(admission)
    }
    pub(crate) fn try_seal_unstarted(&self, admission: &RoventCollectionRootAdmissionV2)
        -> Result<super::collection_hooks_core_v2::CoreUnstartedSealV2, ArenaError> {
        self.core.try_seal_unstarted(admission)
    }
    /// Install no-capture only after the original attempt has been atomically
    /// sealed before observer installation. Pending retains this request.
    fn defer_after_unstarted_seal(
        &self,
        port: &mut RoventNnuecCollectionAdmissionPortV2<'_>,
        original: &RoventCollectionRootAdmissionV2,
        admission: RoventCollectionRootAdmissionV2,
        reason: AdaptiveCollectionError,
    ) -> Result<RoventNnuecCollectionAdmissionOutcomeV2, RoventNnuecCollectionAdmissionFailureV2> {
        match self.core.try_seal_unstarted(original) {
            Ok(super::collection_hooks_core_v2::CoreUnstartedSealV2::Sealed) => self.defer_before_start(port, admission, reason),
            Ok(super::collection_hooks_core_v2::CoreUnstartedSealV2::Busy) => Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission }),
            Ok(super::collection_hooks_core_v2::CoreUnstartedSealV2::Unknown) => Err(RoventNnuecCollectionAdmissionFailureV2 {
                error: AdaptiveManagerError::Invalid("Collection original unstarted state unknown"), admission, owners: Box::new(()),
            }),
            Err(error) => Err(RoventNnuecCollectionAdmissionFailureV2 {
                error: AdaptiveManagerError::Collection(AdaptiveCollectionError::new(AdaptiveCollectionStage::Dispatch, "RoventUnstartedSeal", error.to_string())),
                admission, owners: Box::new(error),
            }),
        }
    }
    /// Original pre-install failure remains available even if no-capture
    /// registration stayed pending until the root deadline.
    pub(crate) fn before_start_failure(&self) -> Option<&AdaptiveManagerError> {
        self.before_start_failure.get()
    }
    pub(crate) fn sealed_pending_event(&self) -> Option<&Arc<RoventNnuecPendingCollectionV2>> {
        self.unstarted_sealed.get()
    }
    pub(crate) fn first_pending_event(&self) -> Option<&Arc<RoventNnuecPendingCollectionV2>> {
        self.first_pending_event.get()
    }
    fn verify_original_attempt(&self, admission: &RoventCollectionRootAdmissionV2) -> Result<(), RoventNnuecRootHookFailureV2> {
        if self.original_attempt.get().is_some_and(|original| original.same_original_admission(admission)) {
            Ok(())
        } else {
            Err(RoventNnuecRootHookFailureV2 { admission: admission.clone(), error: AdaptiveManagerError::Invalid("Collection V2 root hook admission mismatch") })
        }
    }
    pub(crate) fn installed_admission_for_recovery(&self) -> Option<&RoventCollectionRootAdmissionV2> {
        self.core.installed_admission_for_recovery()
    }
    pub(crate) fn try_begin_commit(&self, admission: &RoventCollectionRootAdmissionV2)
        -> Result<Option<RoventCollectionCommitGateV2<'_, RoventNnuecCollectionHandoffV2>>, ArenaError> {
        self.core.try_begin_commit(admission)
    }
    pub(crate) fn new(
        owner: RoventCollectionOwnerLockV2,
        manifest: Arc<RoventCorpusManifestLockV2>,
        roster: Vec<RoventTeacherRosterEntryV2>,
    ) -> Result<(Arc<Self>, RoventNnuecCollectionWorkerV2), ArenaError> {
        let roster_identity = roster.iter().map(|entry|
            (entry.actor_id.clone(), entry.source_index, entry.collection_owner)).collect();
        let (core, mut worker) =
            CollectionBridgeCoreV2::new(owner.clone(), Arc::clone(&manifest), roster, handoff_matches, handoff_submit)?;
        worker.configure_handoff_recovery(recover_root_handoff, handoff_identity);
        Ok((Arc::new(Self { core, owner, manifest, roster_identity, worker: Mutex::new(None), original_attempt: OnceLock::new(), before_start_failure: OnceLock::new(), unstarted_sealed: OnceLock::new(), first_pending_event: OnceLock::new(), pending_stopped: Mutex::new(PendingCollectionSummaryV2::new()), deferred: Mutex::new(DeferredCollectionHistoryV2 {
            retained: Vec::with_capacity(256), summarized: 0, first_summarized: None,
            summarized_reasons: Vec::with_capacity(16), other_summarized_reasons: 0,
        }) }), worker))
    }

    /// Observations retain the actual original root admission; they never
    /// confer collection or physical completion.
    pub(crate) fn try_deferred_observations(&self) -> Result<Option<(Vec<(String, AdaptiveCollectionError)>, u64, Option<String>, Vec<(AdaptiveCollectionError, u64)>, u64)>, ArenaError> {
        match self.deferred.try_lock() {
            Ok(roots) => Ok(Some((roots.retained.iter().map(|(admission, reason)| (admission.raw_sha256().to_owned(), reason.clone())).collect(), roots.summarized, roots.first_summarized.clone(), roots.summarized_reasons.clone(), roots.other_summarized_reasons))),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => Err(ArenaError::Invalid("deferred collection root custody poisoned".into())),
        }
    }

    pub(crate) fn try_has_deferred_observations(&self) -> Result<Option<bool>, ArenaError> {
        match self.deferred.try_lock() {
            Ok(roots) => Ok(Some(!roots.retained.is_empty() || roots.summarized != 0)),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => Err(ArenaError::Invalid("deferred collection root custody poisoned".into())),
        }
    }

    pub(crate) fn try_pending_stopped_observation(&self) -> Result<Option<PendingCollectionSummaryV2>, ArenaError> {
        match self.pending_stopped.try_lock() {
            Ok(summary) => Ok(Some(summary.clone())),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => Err(ArenaError::Invalid("pending collection summary poisoned".into())),
        }
    }

    pub(crate) fn defer_before_start(
        &self,
        port: &mut RoventNnuecCollectionAdmissionPortV2<'_>,
        admission: RoventCollectionRootAdmissionV2,
        reason: AdaptiveCollectionError,
    ) -> Result<RoventNnuecCollectionAdmissionOutcomeV2, RoventNnuecCollectionAdmissionFailureV2> {
        let mut deferred = match self.deferred.try_lock() {
            Ok(roots) => roots,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission }),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(RoventNnuecCollectionAdmissionFailureV2 {
                error: AdaptiveManagerError::Invalid("deferred collection root custody poisoned"), admission, owners: Box::new(()),
            }),
        };
        let capability = match port.no_capture_capability(&admission) {
            Ok(capability) => capability,
            Err(AdaptiveManagerError::AuxiliaryPending) => return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission }),
            Err(error) => return Err(RoventNnuecCollectionAdmissionFailureV2 { error, admission, owners: Box::new(()) }),
        };
        let evidence = if deferred.retained.len() < 256 {
            Some((admission.clone(), reason.clone()))
        } else { None };
        let first_summarized = if evidence.is_none() && deferred.first_summarized.is_none() {
            Some(admission.raw_sha256().to_owned())
        } else { None };
        let new_summary_reason = if evidence.is_none()
            && deferred.summarized_reasons.len() < 16
            && !deferred.summarized_reasons.iter().any(|(original, _)| *original == reason) {
            Some(reason.clone())
        } else { None };
        match self.core.try_install_no_capture(&capability) {
            Ok(true) => {
                if let Some(evidence) = evidence {
                    deferred.retained.push(evidence);
                } else {
                    deferred.summarized = deferred.summarized.saturating_add(1);
                    if let Some(first) = first_summarized { deferred.first_summarized = Some(first); }
                    if let Some((_, count)) = deferred.summarized_reasons.iter_mut().find(|(original, _)| *original == reason) {
                        *count = count.saturating_add(1);
                    } else if let Some(reason) = new_summary_reason {
                        deferred.summarized_reasons.push((reason, 1));
                    } else {
                        deferred.other_summarized_reasons = deferred.other_summarized_reasons.saturating_add(1);
                    }
                }
                Ok(RoventNnuecCollectionAdmissionOutcomeV2::DeferredNoCapture { capability, reason })
            }
            Ok(false) => Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission }),
            Err(error) => Err(RoventNnuecCollectionAdmissionFailureV2 {
                error: AdaptiveManagerError::Collection(AdaptiveCollectionError::new(
                    AdaptiveCollectionStage::Dispatch, "RoventNoCaptureMarker", error.to_string())),
                admission, owners: Box::new(error),
            }),
        }
    }
}

impl RoventNnuecCollectionHooksV2 for RoventNnuecCollectionBridgeV2 {
    fn try_record_pending_stopped(
        &self, event: &Arc<RoventNnuecPendingCollectionV2>,
    ) -> Result<RoventPendingEvidenceDispositionV2, AdaptiveManagerError> {
        if self.original_attempt.get().is_some_and(|original| original.same_original_admission(&event.admission)) {
            let _ = self.first_pending_event.set(Arc::clone(event));
            match self.core.try_seal_unstarted(&event.admission) {
                Ok(super::collection_hooks_core_v2::CoreUnstartedSealV2::Sealed) => {
                    let _ = self.unstarted_sealed.set(Arc::clone(event));
                }
                Ok(super::collection_hooks_core_v2::CoreUnstartedSealV2::Busy | super::collection_hooks_core_v2::CoreUnstartedSealV2::Unknown) => {},
                Err(error) => return Err(AdaptiveManagerError::Collection(AdaptiveCollectionError::new(
                    AdaptiveCollectionStage::Settled, "RoventUnstartedSeal", error.to_string()))),
            }
        }
        let prepared_cause = event.cause.clone();
        let prepared_reason = event.reason.clone();
        let mut pending = match self.pending_stopped.try_lock() {
            Ok(pending) => pending,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(RoventPendingEvidenceDispositionV2::Busy),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("pending collection summary poisoned")),
        };
        if let Some(first) = &pending.first {
            if Arc::ptr_eq(first, event) { return Ok(RoventPendingEvidenceDispositionV2::OriginalRetained); }
        } else {
            pending.first = Some(Arc::clone(event));
            return Ok(RoventPendingEvidenceDispositionV2::OriginalRetained);
        }
        pending.summarized = pending.summarized.saturating_add(1);
        if let Some((_, _, count)) = pending.reasons.iter_mut().find(|(cause, reason, _)| *cause == event.cause && *reason == event.reason) {
            *count = count.saturating_add(1);
        } else if pending.reasons.len() < 16 {
            pending.reasons.push((prepared_cause, prepared_reason, 1));
        } else {
            pending.other = pending.other.saturating_add(1);
        }
        Ok(RoventPendingEvidenceDispositionV2::Summarized)
    }
    fn register_startup_control(
        &self,
        control: &RoventStartupControl,
    ) -> Result<(), AdaptiveManagerError> {
        self.core.register_startup_control(control)
    }
    fn admit(
        &self,
        port: &mut RoventNnuecCollectionAdmissionPortV2<'_>,
        admission: RoventCollectionRootAdmissionV2,
    ) -> Result<RoventNnuecCollectionAdmissionOutcomeV2, RoventNnuecCollectionAdmissionFailureV2> {
        if let Some(first) = self.original_attempt.get().filter(|first| !first.same_original_admission(&admission)) {
            if self.unstarted_sealed.get().is_none() {
                if let Some(event) = self.first_pending_event.get().filter(|event| event.admission.same_original_admission(first)) {
                    match self.core.try_seal_unstarted(first) {
                        Ok(super::collection_hooks_core_v2::CoreUnstartedSealV2::Sealed) => { let _ = self.unstarted_sealed.set(Arc::clone(event)); },
                        Ok(super::collection_hooks_core_v2::CoreUnstartedSealV2::Busy) => return Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission }),
                        Ok(super::collection_hooks_core_v2::CoreUnstartedSealV2::Unknown) => {},
                        Err(error) => return Err(RoventNnuecCollectionAdmissionFailureV2 {
                            error: AdaptiveManagerError::Collection(AdaptiveCollectionError::new(AdaptiveCollectionStage::Settled, "RoventUnstartedSeal", error.to_string())),
                            admission, owners: Box::new(error),
                        }),
                    }
                }
            }
        }
        if let Some(event) = self.unstarted_sealed.get() {
            let first = &event.admission;
            let previous = first.root_admission().authority();
            let next = admission.root_admission().authority();
            let later = first.native_origin().same_original_origin(admission.native_origin())
                && previous.epoch == next.epoch && previous.implementation == next.implementation
                && (next.game > previous.game || (next.game == previous.game && next.root > previous.root));
            if later {
                return self.defer_before_start(port, admission, event.reason.clone());
            }
        }
        if let (Some(first), Some(error)) = (self.original_attempt.get(), self.before_start_failure.get()) {
            let previous = first.root_admission().authority();
            let next = admission.root_admission().authority();
            let same = first.same_original_admission(&admission);
            let later = first.native_origin().same_original_origin(admission.native_origin())
                && previous.epoch == next.epoch && previous.implementation == next.implementation
                && (next.game > previous.game || (next.game == previous.game && next.root > previous.root));
            if same || later {
                let reason = AdaptiveCollectionError::new(AdaptiveCollectionStage::Dispatch,
                    "RoventCollectionBeforeStart", format!("{error:?}"));
                return self.defer_after_unstarted_seal(port, first, admission, reason);
            }
        }
        // A previous single-root bridge can no longer start work for this
        // admission. Only this known pre-install state may be deferred; an
        // in-progress reservation is not evidence that no worker was started.
        if self.core.closed_before_next_admission(&admission) {
            let reason = AdaptiveCollectionError::new(AdaptiveCollectionStage::Dispatch,
                "RoventCollectionPreviousRootRetained", "prior single-root collector remains closed; this root has no collection");
            return self.defer_before_start(port, admission, reason);
        }
        let _ = self.original_attempt.set(admission.clone());
        if let Err(failure) = self.verify_original_attempt(&admission) {
            return Err(RoventNnuecCollectionAdmissionFailureV2 {
                error: failure.error, admission: failure.admission, owners: Box::new(()),
            });
        }
        match self.core
            .admit(admission, |admission, owner| {
                port.install(admission, owner.observer())
                    .map_err(|original| CoreAdmissionFailure {
                        error: original.error.clone(),
                        admission: original.admission.clone(),
                        owners: Box::new(original),
                        before_start: false,
                        game_gate_status: super::collection_hooks_core_v2::CoreGameGateStatus::NotApplicable,
                    })
            }) {
            Ok(()) => Ok(RoventNnuecCollectionAdmissionOutcomeV2::Installed),
            Err(failure) if failure.game_gate_status == super::collection_hooks_core_v2::CoreGameGateStatus::Busy => {
                if failure.before_start {
                    Ok(RoventNnuecCollectionAdmissionOutcomeV2::Pending { admission: failure.admission })
                } else {
                    Ok(RoventNnuecCollectionAdmissionOutcomeV2::PendingUncertain { admission: failure.admission })
                }
            }
            Err(failure) if failure.before_start => {
                let _ = self.before_start_failure.set(failure.error.clone());
                let error = self.before_start_failure.get().expect("before-start failure retained");
                let reason = AdaptiveCollectionError::new(AdaptiveCollectionStage::Dispatch,
                    "RoventCollectionBeforeStart", format!("{error:?}"));
                let original = failure.admission.clone();
                self.defer_after_unstarted_seal(port, &original, failure.admission, reason)
            }
            Err(failure) => Err(RoventNnuecCollectionAdmissionFailureV2 {
                error: failure.error, admission: failure.admission, owners: failure.owners,
            }),
        }
    }
    fn finish_deferred(
        &self,
        capability: &RoventNoCaptureRootCapabilityV2,
        fence: &AdaptiveShutdownReceipt,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveManagerError> {
        self.core.try_finish_no_capture(capability, fence).map(|finished| {
            if finished { AdaptiveCollectionBoundaryV2::Ready } else { AdaptiveCollectionBoundaryV2::Busy }
        }).map_err(|error| AdaptiveManagerError::Collection(AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Settled, "RoventNoCaptureFinish", error.to_string())))
    }
    fn defer_no_capture(
        &self,
        port: &mut RoventNnuecCollectionAdmissionPortV2<'_>,
        admission: RoventCollectionRootAdmissionV2,
        reason: AdaptiveCollectionError,
    ) -> Result<RoventNnuecCollectionAdmissionOutcomeV2, RoventNnuecCollectionAdmissionFailureV2> {
        // UCI invokes this only before starting any observer/teacher for the
        // same admission; the port independently verifies that precondition.
        self.defer_before_start(port, admission, reason)
    }
    fn pump(&self) -> Result<(), AdaptiveManagerError> {
        self.core.pump()
    }
    fn revoke(&self) {
        self.core.revoke();
    }
    fn handoff_sender(&self) -> mpsc::SyncSender<Arc<Mutex<RoventNnuecCollectionHandoffV2>>> {
        self.core.handoff_sender()
    }
    fn pump_root(&self, admission: &RoventCollectionRootAdmissionV2) -> Result<(), RoventNnuecRootHookFailureV2> {
        self.verify_original_attempt(admission)?;
        self.core.pump().map_err(|error| RoventNnuecRootHookFailureV2 { admission: admission.clone(), error })
    }
    fn revoke_root(&self, admission: &RoventCollectionRootAdmissionV2) -> Result<(), RoventNnuecRootHookFailureV2> {
        self.verify_original_attempt(admission)?;
        self.core.revoke();
        Ok(())
    }
    fn handoff_sender_root(&self, admission: &RoventCollectionRootAdmissionV2)
        -> Result<mpsc::SyncSender<Arc<Mutex<RoventNnuecCollectionHandoffV2>>>, RoventNnuecRootHookFailureV2> {
        self.verify_original_attempt(admission)?;
        Ok(self.core.handoff_sender())
    }
}
