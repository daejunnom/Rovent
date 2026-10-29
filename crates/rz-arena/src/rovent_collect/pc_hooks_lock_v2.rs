//! Typed adapter around the shared bounded single-root Collection /2 worker.
//! This layer does not combine P-C and Own manager or handoff authority.
use super::collection_hooks_core_v2::{
    CollectionBridgeCoreV2, CollectionWorkerCoreV2, CoreAdmissionFailure,
};
use super::teacher::v2_session::{RoventTeacherRosterEntryV2, TeacherSessionRetirementSealV2};
use super::{ArenaError, RoventCollectionOwnerLockV2};
use rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2;
use rz_uci::pals_adaptive::{AdaptiveManagerError, RoventCollectionRootAdmissionV2};
use rz_uci::rovent_bootstrap::RoventStartupControl;
use rz_uci::rovent_bootstrap::pc_session::{
    RoventPcCollectionAdmissionFailureV2, RoventPcCollectionAdmissionPortV2,
    RoventPcCollectionHandoffV2, RoventPcCollectionHooksV2, RoventPcCollectionRootCustodyV2,
};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

pub(crate) use super::collection_hooks_core_v2::CollectionWorkerExitV2 as RoventPcCollectionWorkerExitV2;
pub(crate) type RoventPcCollectionWorkerV2 = CollectionWorkerCoreV2<RoventPcCollectionHandoffV2>;

pub(crate) struct RoventPcCollectionBridgeV2 {
    core: Arc<CollectionBridgeCoreV2<RoventPcCollectionHandoffV2>>,
    worker: Mutex<Option<PcWorkerCustodianV2>>,
}

struct PcWorkerCustodianV2 {
    original: Arc<Mutex<RoventPcCollectionWorkerV2>>,
    handle: Option<JoinHandle<RoventPcCollectionWorkerExitV2>>,
    exit: Option<RoventPcCollectionWorkerExitV2>,
    exit_kind: Option<PcWorkerExitKindV2>,
    panic: Option<Box<dyn std::any::Any + Send>>,
    transferred: bool,
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
    pub(crate) fn new(
        owner: RoventCollectionOwnerLockV2,
        manifest: Arc<RoventCorpusManifestLockV2>,
        roster: Vec<RoventTeacherRosterEntryV2>,
    ) -> Result<(Arc<Self>, RoventPcCollectionWorkerV2), ArenaError> {
        let (core, worker) =
            CollectionBridgeCoreV2::new(owner, manifest, roster, handoff_matches, handoff_submit)?;
        Ok((Arc::new(Self { core, worker: Mutex::new(None) }), worker))
    }

    pub(crate) fn start_worker(&self, worker: &Arc<Mutex<RoventPcCollectionWorkerV2>>, stop: Arc<std::sync::atomic::AtomicBool>, deadline: std::time::Instant) -> Result<(), ArenaError> {
        let mut slot = self.worker.try_lock().map_err(|_| super::invalid("PC V2 worker custodian busy or poisoned"))?;
        super::check(slot.is_none(), "PC V2 worker already registered")?;
        let original = Arc::clone(worker);
        let handle = std::thread::Builder::new().name("rovent-pc-collection-v2".into()).spawn(move || {
            let exit = match original.lock() {
                Ok(mut core) => core.run_until(stop, deadline),
                Err(_) => panic!("PC V2 original worker mutex poisoned"),
            };
            exit
        }).map_err(|error| ArenaError::Io(error.to_string()))?;
        *slot = Some(PcWorkerCustodianV2 { original: Arc::clone(worker), handle: Some(handle), exit: None, exit_kind: None, panic: None, transferred: false });
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
    ) -> Result<(), RoventPcCollectionAdmissionFailureV2> {
        self.core
            .admit(admission, |admission, owner| {
                port.install(admission, owner.observer())
                    .map_err(|original| CoreAdmissionFailure {
                        error: original.error.clone(),
                        admission: original.admission.clone(),
                        owners: Box::new(original),
                    })
            })
            .map_err(|failure| RoventPcCollectionAdmissionFailureV2 {
                error: failure.error,
                admission: failure.admission,
                owners: failure.owners,
            })
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
}
