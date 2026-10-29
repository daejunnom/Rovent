//! Typed adapter around the shared bounded single-root Collection /2 worker.
//! This layer does not combine P-C and Own manager or handoff authority.
use super::collection_hooks_core_v2::{
    CollectionBridgeCoreV2, CollectionWorkerCoreV2, CoreAdmissionFailure,
};
use super::teacher::v2_session::{RoventTeacherRosterEntryV2, TeacherSessionRetirementSealV2};
use super::{ArenaError, RoventCollectionOwnerLockV2};
use rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2;
use rz_uci::pals_adaptive::{AdaptiveCollectionError, AdaptiveCollectionStage, AdaptiveManagerError, RoventCollectionRootAdmissionV2};
use rz_uci::rovent_bootstrap::RoventStartupControl;
use rz_uci::rovent_bootstrap::pc_session::{
    RoventPcCollectionAdmissionFailureV2, RoventPcCollectionAdmissionPortV2,
    RoventPcCollectionHandoffV2, RoventPcCollectionHooksV2, RoventPcCollectionRootCustodyV2, RoventPcCustodyReturnV2,
    RoventPcCollectionAdmissionOutcomeV2,
    RoventPcDeferredCleanupV2,
};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

pub(crate) use super::collection_hooks_core_v2::CollectionWorkerExitV2 as RoventPcCollectionWorkerExitV2;
pub(crate) type RoventPcCollectionWorkerV2 = CollectionWorkerCoreV2<RoventPcCollectionHandoffV2>;

pub(crate) struct RoventPcCollectionBridgeV2 {
    core: Arc<CollectionBridgeCoreV2<RoventPcCollectionHandoffV2>>,
    worker: Mutex<Option<PcWorkerCustodianV2>>,
    deferred: Mutex<PcDeferredRootsV2>,
}

struct PcDeferredRootsV2 {
    originals: Vec<(RoventCollectionRootAdmissionV2, AdaptiveCollectionError)>,
    cleanup_failures: Vec<Arc<Mutex<RoventPcDeferredCleanupV2>>>,
    overflow: u64,
    first_overflow: Option<(RoventCollectionRootAdmissionV2, AdaptiveCollectionError)>,
}

struct PcWorkerCustodianV2 {
    original: Arc<Mutex<RoventPcCollectionWorkerV2>>,
    handle: Option<JoinHandle<RoventPcCollectionWorkerExitV2>>,
    exit: Option<RoventPcCollectionWorkerExitV2>,
    exit_kind: Option<PcWorkerExitKindV2>,
    panic: Option<Box<dyn std::any::Any + Send>>,
    transferred: bool,
    returned: Option<Arc<Mutex<RoventPcCollectionRootCustodyV2>>>,
}

/// The real joined worker core and its exit travel together. This object is
/// evidence custody only, never authority for collection success.
struct PcJoinedWorkerV2 {
    _worker: Arc<Mutex<RoventPcCollectionWorkerV2>>,
    _exit: RoventPcCollectionWorkerExitV2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PcWorkerExitKindV2 {
    NoAdmission, Retirement, Deadline, AdmissionCanceled, PhysicalAck, Aggregate, CaughtPanic, OuterPanic,
}
pub(crate) enum PcWorkerStartErrorV2 {
    BeforeStart(ArenaError),
    Unknown(ArenaError),
}
impl PcWorkerExitKindV2 {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::NoAdmission => "no_admission", Self::Retirement => "retirement_and_handoff_observed",
            Self::Deadline => "deadline", Self::AdmissionCanceled => "admission_canceled",
            Self::PhysicalAck => "physical_scope_ack_delivered", Self::Aggregate => "physical_aggregate_handoff_observed",
            Self::CaughtPanic => "caught_panic", Self::OuterPanic => "outer_panic",
        }
    }
}
fn exit_kind(exit: &RoventPcCollectionWorkerExitV2) -> PcWorkerExitKindV2 {
    match exit {
        RoventPcCollectionWorkerExitV2::NoAdmission => PcWorkerExitKindV2::NoAdmission,
        RoventPcCollectionWorkerExitV2::RetirementAndHandoffObserved => PcWorkerExitKindV2::Retirement,
        RoventPcCollectionWorkerExitV2::Deadline => PcWorkerExitKindV2::Deadline,
        RoventPcCollectionWorkerExitV2::AdmissionCanceled => PcWorkerExitKindV2::AdmissionCanceled,
        RoventPcCollectionWorkerExitV2::PhysicalAckDelivered => PcWorkerExitKindV2::PhysicalAck,
        RoventPcCollectionWorkerExitV2::AggregateHandoffObserved => PcWorkerExitKindV2::Aggregate,
        RoventPcCollectionWorkerExitV2::Panicked(_) => PcWorkerExitKindV2::CaughtPanic,
    }
}

impl PcWorkerCustodianV2 {
    fn poll_join(&mut self) {
        if self.handle.as_ref().is_some_and(JoinHandle::is_finished) {
            match self.handle.take().expect("finished original worker").join() {
                Ok(exit) => { self.exit_kind = Some(exit_kind(&exit)); self.exit = Some(exit); }
                Err(payload) => { self.exit_kind = Some(PcWorkerExitKindV2::OuterPanic); self.panic = Some(payload); }
            }
        }
    }
}

fn handoff_matches(
    handoff: &mut RoventPcCollectionHandoffV2,
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
    handoff: &mut RoventPcCollectionHandoffV2,
    ack: rz_uci::rovent_external_launch::RoventTeacherScopeAggregateAckV2,
) -> Result<(), rz_uci::rovent_external_launch::RoventTeacherScopeAggregateAckV2> {
    handoff.submit_teacher_aggregate_ack(ack)
}

impl RoventPcCollectionBridgeV2 {
    pub(crate) fn installed_admission_for_recovery(&self) -> Option<&RoventCollectionRootAdmissionV2> {
        self.core.installed_admission_for_recovery()
    }
    pub(crate) fn try_begin_commit(&self, admission: &RoventCollectionRootAdmissionV2)
        -> Result<Option<super::collection_hooks_core_v2::RoventCollectionCommitGateV2<'_, RoventPcCollectionHandoffV2>>, ArenaError> {
        self.core.try_begin_commit(admission)
    }
    pub(crate) fn new(
        owner: RoventCollectionOwnerLockV2,
        manifest: Arc<RoventCorpusManifestLockV2>,
        roster: Vec<RoventTeacherRosterEntryV2>,
    ) -> Result<(Arc<Self>, RoventPcCollectionWorkerV2), ArenaError> {
        let (core, worker) =
            CollectionBridgeCoreV2::new(owner, manifest, roster, handoff_matches, handoff_submit)?;
        Ok((Arc::new(Self { core, worker: Mutex::new(None), deferred: Mutex::new(PcDeferredRootsV2 {
            originals: Vec::with_capacity(256), cleanup_failures: Vec::new(), overflow: 0, first_overflow: None,
        }) }), worker))
    }

    fn defer_before_start(
        &self, port: &mut RoventPcCollectionAdmissionPortV2<'_>,
        admission: RoventCollectionRootAdmissionV2, reason: AdaptiveCollectionError,
    ) -> Result<RoventPcCollectionAdmissionOutcomeV2, RoventPcCollectionAdmissionFailureV2> {
        let mut deferred = match self.deferred.try_lock() {
            Ok(deferred) => deferred,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(RoventPcCollectionAdmissionOutcomeV2::Pending { admission }),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(RoventPcCollectionAdmissionFailureV2 {
                error: AdaptiveManagerError::Invalid("P/C deferred root custody poisoned"), admission, owners: Box::new(()),
            }),
        };
        let capability = match port.no_capture_capability(&admission) {
            Ok(capability) => capability,
            Err(AdaptiveManagerError::AuxiliaryPending) => return Ok(RoventPcCollectionAdmissionOutcomeV2::Pending { admission }),
            Err(error) => return Err(RoventPcCollectionAdmissionFailureV2 { error, admission, owners: Box::new(()) }),
        };
        // Prepare evidence before installing the native marker. Keep bounded
        // original root/causes; overflow remains explicitly summarized.
        let evidence = (admission.clone(), reason.clone());
        match self.core.try_install_no_capture(&capability) {
            Ok(true) => {
                if deferred.originals.len() < 256 { deferred.originals.push(evidence); }
                else { deferred.overflow = deferred.overflow.saturating_add(1); if deferred.first_overflow.is_none() { deferred.first_overflow = Some(evidence); } }
                Ok(RoventPcCollectionAdmissionOutcomeV2::DeferredNoCapture { capability, reason })
            }
            Ok(false) => Ok(RoventPcCollectionAdmissionOutcomeV2::Pending { admission }),
            Err(error) => Err(RoventPcCollectionAdmissionFailureV2 {
                error: AdaptiveManagerError::Collection(AdaptiveCollectionError::new(
                    AdaptiveCollectionStage::Dispatch, "RoventNoCaptureMarker", error.to_string())),
                admission, owners: Box::new(error),
            }),
        }
    }

    /// Copy bounded reporting metadata, retaining the original admissions and
    /// reasons in the bridge. Busy is unknown and never means zero omissions.
    pub(crate) fn try_deferred_observation(&self) -> Result<Option<serde_json::Value>, ArenaError> {
        let deferred = match self.deferred.try_lock() {
            Ok(deferred) => deferred,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(super::invalid("P/C deferred root custody poisoned")),
        };
        let summarize = |(admission, reason): &(RoventCollectionRootAdmissionV2, AdaptiveCollectionError)|
            serde_json::json!({"root_admission_sha256": admission.raw_sha256(), "stage": format!("{:?}", reason.stage),
                "code":reason.code, "detail":reason.detail});
        Ok(Some(serde_json::json!({"retained": deferred.originals.iter().map(summarize).collect::<Vec<_>>(),
            "overflow_count": deferred.overflow, "first_overflow": deferred.first_overflow.as_ref().map(summarize),
            "unresolved_cleanup_count": deferred.cleanup_failures.len()})))
    }

    pub(crate) fn try_has_deferred(&self) -> Result<Option<bool>, ArenaError> {
        match self.deferred.try_lock() {
            Ok(deferred) => Ok(Some(!deferred.originals.is_empty() || deferred.overflow != 0)),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => Err(super::invalid("P/C deferred root custody poisoned")),
        }
    }

    pub(crate) fn start_worker(&self, worker: &Arc<Mutex<RoventPcCollectionWorkerV2>>, stop: Arc<std::sync::atomic::AtomicBool>, deadline: std::time::Instant) -> Result<(), PcWorkerStartErrorV2> {
        let mut slot = self.worker.try_lock().map_err(|_| PcWorkerStartErrorV2::Unknown(super::invalid("PC V2 worker custodian busy or poisoned")))?;
        if slot.is_some() { return Err(PcWorkerStartErrorV2::Unknown(super::invalid("PC V2 worker already registered"))); }
        let original = Arc::clone(worker);
        let handle = std::thread::Builder::new().name("rovent-pc-collection-v2".into()).spawn(move || {
            let exit = match original.lock() {
                Ok(mut core) => core.run_until(stop, deadline),
                Err(_) => panic!("PC V2 original worker mutex poisoned"),
            };
            exit
        }).map_err(|error| PcWorkerStartErrorV2::BeforeStart(ArenaError::Io(error.to_string())))?;
        *slot = Some(PcWorkerCustodianV2 { original: Arc::clone(worker), handle: Some(handle), exit: None, exit_kind: None, panic: None, transferred: false, returned: None });
        Ok(())
    }

    pub(crate) fn poll_worker(&self) -> Result<Option<PcWorkerExitKindV2>, AdaptiveManagerError> {
        let mut slot = match self.worker.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("PC V2 worker custodian poisoned; original retained")),
        };
        let owner = slot.as_mut().ok_or(AdaptiveManagerError::Invalid("PC V2 worker unregistered"))?;
        owner.poll_join();
        Ok(owner.exit_kind)
    }

    pub(crate) fn take_outer_panic(&self) -> Option<Box<dyn std::any::Any + Send>> {
        self.worker.try_lock().ok()?.as_mut()?.panic.take()
    }

    /// Inspect only the same joined/returned root. This confirms the native
    /// fence and observer settlement but does not publish any retained row.
    pub(crate) fn returned_handoff_settled(&self) -> Result<bool, AdaptiveManagerError> {
        let slot = match self.worker.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("PC V2 worker custodian poisoned; original retained")),
        };
        let Some(owner) = slot.as_ref() else { return Ok(false); };
        let Some(custody) = &owner.returned else { return Ok(false); };
        let custody = match custody.try_lock() {
            Ok(custody) => custody,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("PC V2 original custody poisoned")),
        };
        let Some(joined) = custody.original_worker().downcast_ref::<PcJoinedWorkerV2>() else { return Ok(false); };
        if !Arc::ptr_eq(&joined._worker, &owner.original) { return Ok(false); }
        let core = match owner.original.try_lock() {
            Ok(core) => core,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("PC V2 original worker poisoned; original retained")),
        };
        if core.has_failure() || !core.original_handoff().is_some_and(|actual| Arc::ptr_eq(&actual, custody.handoff())) { return Ok(false); }
        let mut handoff = match custody.handoff().try_lock() {
            Ok(handoff) => handoff,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("PC V2 original handoff poisoned")),
        };
        if handoff.teacher_aggregate_ack().is_none() { return Ok(false); }
        let (_, _, settlement, fence) = handoff.parts_mut();
        Ok(settlement.is_ok() && fence.permits_semantic_closure())
    }
}

impl RoventPcCollectionHooksV2 for RoventPcCollectionBridgeV2 {
    fn register_startup_control(
        &self,
        control: &RoventStartupControl,
    ) -> Result<(), AdaptiveManagerError> {
        self.core.register_startup_control(control)
    }
    fn admit(
        &self,
        port: &mut RoventPcCollectionAdmissionPortV2<'_>,
        admission: RoventCollectionRootAdmissionV2,
    ) -> Result<RoventPcCollectionAdmissionOutcomeV2, RoventPcCollectionAdmissionFailureV2> {
        if self.core.closed_before_next_admission(&admission) {
            return self.defer_before_start(port, admission, AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Dispatch, "PcCollectionCapacity",
                "The previous root's bounded collection worker remains owned; this root is not captured",
            ));
        }
        self.core
            .admit(admission, |admission, owner| {
                port.install(admission, owner.observer())
                    .map_err(|original| CoreAdmissionFailure {
                        error: original.error.clone(),
                        admission: original.admission.clone(),
                        owners: Box::new(original),
                        before_start: false,
                    })
            })
            .map(|()| RoventPcCollectionAdmissionOutcomeV2::Installed)
            .map_err(|failure| RoventPcCollectionAdmissionFailureV2 {
                error: failure.error,
                admission: failure.admission,
                owners: failure.owners,
            })
    }
    fn finish_deferred(
        &self,
        capability: &rz_uci::pals_adaptive::RoventNoCaptureRootCapabilityV2,
        fence: &rz_uci::pals_adaptive::RoleOnlyShutdownReceipt,
    ) -> Result<rz_uci::pals_adaptive::AdaptiveCollectionBoundaryV2, AdaptiveManagerError> {
        self.core.try_finish_no_capture_role_only(capability, fence).map(|finished| {
            if finished { rz_uci::pals_adaptive::AdaptiveCollectionBoundaryV2::Ready }
            else { rz_uci::pals_adaptive::AdaptiveCollectionBoundaryV2::Busy }
        }).map_err(|error| AdaptiveManagerError::Collection(AdaptiveCollectionError::new(
            AdaptiveCollectionStage::Settled, "PcNoCaptureFinish", error.to_string())))
    }
    fn retain_deferred_cleanup(
        &self,
        cleanup: Arc<Mutex<RoventPcDeferredCleanupV2>>,
    ) -> Result<RoventPcCustodyReturnV2, AdaptiveManagerError> {
        let mut deferred = match self.deferred.try_lock() {
            Ok(value) => value,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(RoventPcCustodyReturnV2::Busy),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("P/C deferred root custody poisoned")),
        };
        if deferred.cleanup_failures.iter().any(|original| Arc::ptr_eq(original, &cleanup)) {
            return Ok(RoventPcCustodyReturnV2::Accepted);
        }
        let item = match cleanup.try_lock() {
            Ok(value) => value,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(RoventPcCustodyReturnV2::Busy),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("P/C deferred cleanup owner poisoned")),
        };
        let belongs = deferred.originals.iter().chain(deferred.first_overflow.iter()).any(|(admission, _)|
            item.capability().same_original_admission(admission));
        if !belongs || !item.capability().matches_role_only_shutdown_receipt(item.fence()) {
            return Err(AdaptiveManagerError::Invalid("P/C deferred cleanup original root mismatch"));
        }
        drop(item);
        if deferred.cleanup_failures.len() >= 256 || deferred.cleanup_failures.try_reserve(1).is_err() {
            return Err(AdaptiveManagerError::Invalid("P/C deferred cleanup custody capacity"));
        }
        deferred.cleanup_failures.push(cleanup);
        Ok(RoventPcCustodyReturnV2::Accepted)
    }
    fn pump(&self) -> Result<(), AdaptiveManagerError> {
        self.core.pump()
    }
    fn revoke(&self) {
        self.core.revoke();
    }
    fn handoff_sender(&self) -> mpsc::SyncSender<Arc<Mutex<RoventPcCollectionHandoffV2>>> {
        self.core.handoff_sender()
    }
    fn try_take_root_custody(&self, handoff: &Arc<Mutex<RoventPcCollectionHandoffV2>>) -> Result<Option<RoventPcCollectionRootCustodyV2>, AdaptiveManagerError> {
        let mut slot = match self.worker.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("PC V2 worker custodian poisoned; original retained")),
        };
        let owner = slot.as_mut().ok_or(AdaptiveManagerError::Invalid("PC V2 worker unregistered"))?;
        if owner.transferred { return Err(AdaptiveManagerError::Invalid("PC V2 worker custody already transferred")); }
        owner.poll_join();
        if owner.panic.is_some() { return Err(AdaptiveManagerError::Invalid("PC V2 worker panicked; original retained")); }
        let Some(exit) = owner.exit.as_ref() else { return Ok(None); };
        if !matches!(exit, RoventPcCollectionWorkerExitV2::AggregateHandoffObserved | RoventPcCollectionWorkerExitV2::RetirementAndHandoffObserved | RoventPcCollectionWorkerExitV2::PhysicalAckDelivered) {
            return Err(AdaptiveManagerError::Invalid("PC V2 worker exit unresolved; original retained"));
        }
        let core = match owner.original.try_lock() {
            Ok(core) => core,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("PC V2 original worker poisoned; original retained")),
        };
        if !core.original_handoff().is_some_and(|actual| Arc::ptr_eq(&actual, handoff)) {
            return Err(AdaptiveManagerError::Invalid("PC V2 root handoff mismatch; original retained"));
        }
        drop(core);
        let exit = owner.exit.take().expect("joined original exit");
        owner.transferred = true;
        Ok(Some(RoventPcCollectionRootCustodyV2::from_joined_worker(
            Arc::clone(handoff), Box::new(PcJoinedWorkerV2 { _worker: Arc::clone(&owner.original), _exit: exit }),
        )))
    }
    fn return_root_custody(&self, custody: Arc<Mutex<RoventPcCollectionRootCustodyV2>>) -> Result<RoventPcCustodyReturnV2, AdaptiveManagerError> {
        let mut slot = match self.worker.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(RoventPcCustodyReturnV2::Busy),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("PC V2 worker custodian poisoned; original retained")),
        };
        let owner = slot.as_mut().ok_or(AdaptiveManagerError::Invalid("PC V2 worker unregistered"))?;
        if let Some(original) = &owner.returned {
            if !Arc::ptr_eq(original, &custody) { return Err(AdaptiveManagerError::Invalid("PC V2 different custody already retained")); }
        }
        if !owner.transferred { return Err(AdaptiveManagerError::Invalid("PC V2 custody has not been transferred")); }
        let original = match custody.try_lock() {
            Ok(original) => original,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(RoventPcCustodyReturnV2::Busy),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("PC V2 original custody poisoned")),
        };
        let joined = original.original_worker().downcast_ref::<PcJoinedWorkerV2>()
            .ok_or(AdaptiveManagerError::Invalid("PC V2 original worker custody type mismatch"))?;
        if !Arc::ptr_eq(&joined._worker, &owner.original) {
            return Err(AdaptiveManagerError::Invalid("PC V2 original worker mismatch"));
        }
        let core = match owner.original.try_lock() {
            Ok(core) => core,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(RoventPcCustodyReturnV2::Busy),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(AdaptiveManagerError::Invalid("PC V2 original worker poisoned; original retained")),
        };
        if !core.original_handoff().is_some_and(|actual| Arc::ptr_eq(&actual, original.handoff())) {
            return Err(AdaptiveManagerError::Invalid("PC V2 original handoff mismatch"));
        }
        drop(core);
        drop(original);
        if owner.returned.is_none() { owner.returned = Some(custody); }
        Ok(RoventPcCustodyReturnV2::Accepted)
    }
}
