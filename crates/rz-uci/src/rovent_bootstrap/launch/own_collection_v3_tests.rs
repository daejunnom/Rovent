//! CPU-only custody tests. The synthetic marker below is not a native fence.
//!
//! These tests exercise ownership gates without starting a worker or device.

use super::*;
use crate::pals_adaptive::{
    AdaptiveCollectionBoundaryV2, RoventBoardCollectionAdmissionFailureV3,
    RoventBoardCollectionAdmissionOutcomeV3, RoventBoardCollectionAdmissionPortV3,
    RoventBoardCollectionHandoffV3, RoventBoardCollectionRootCustodyV3, RoventBoardCustodyReturnV3,
    RoventBoardDeferredCleanupV3, RoventCollectionRootAdmissionV3, RoventNoCaptureRootCapabilityV3,
    RoventOwnNoCaptureFenceV3,
};
use crate::pals_native::ticketed::{
    NativeTicketQuestion, TicketExecutionEvidence, TicketPreparation,
    VirtualReviewPhysicalInvocationEvidence, VirtualReviewPreparedTicket,
    VirtualReviewTicketOutput,
};
use rz_contracts::pals::RoleOutput;
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_search::pals::coordinator::{PhysicalRoleCompletion, RoleFailure};
use rz_search::pals::review_context::ReviewView;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc::SyncSender;

type Backend = RoventNativeBackend<AdaptivePortQuestion>;

struct Observer;
impl VirtualReviewCollectionObserver for Observer {
    fn submitted(
        &mut self,
        _: &RoleTicketEnvelope,
        _: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        _: &VirtualReviewPreparedTicket,
    ) -> Result<(), RoleFailure> {
        Ok(())
    }
    fn completed(
        &mut self,
        _: &RoleTicketEnvelope,
        _: PhysicalRoleCompletion,
        _: Option<&RoleOutput>,
        _: Option<TicketExecutionEvidence>,
        _: Option<VirtualReviewPhysicalInvocationEvidence>,
    ) -> Result<(), RoleFailure> {
        Ok(())
    }
    fn retired(&mut self, _: &RoleTicketEnvelope, _: bool) -> Result<(), RoleFailure> {
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Finish {
    Ready,
    Error,
    Panic,
}

struct Hooks {
    drops: Arc<AtomicUsize>,
    finish: Finish,
}
impl Drop for Hooks {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
impl RoventBoardCollectionHooksV3<Backend> for Hooks {
    fn try_finish_launch(
        &self,
        _: &RoventOwnCollectionNativeFenceV3,
    ) -> Result<Option<Box<dyn Any + Send>>, AdaptiveManagerError> {
        match self.finish {
            Finish::Ready => Ok(Some(Box::new(()))),
            Finish::Error => Err(AdaptiveManagerError::Invalid("fixture failure")),
            Finish::Panic => panic!("fixture settlement panic"),
        }
    }
    fn register_startup_control(
        &self,
        _: &crate::rovent_bootstrap::RoventStartupControl,
    ) -> Result<(), AdaptiveManagerError> {
        unreachable!("no worker starts")
    }
    fn admit(
        &self,
        _: &mut RoventBoardCollectionAdmissionPortV3<'_, Backend>,
        _: RoventCollectionRootAdmissionV3,
    ) -> Result<RoventBoardCollectionAdmissionOutcomeV3, RoventBoardCollectionAdmissionFailureV3>
    {
        unreachable!("no root starts")
    }
    fn pump_root(&self, _: &RoventCollectionRootAdmissionV3) -> Result<(), AdaptiveManagerError> {
        unreachable!("no root starts")
    }
    fn revoke_root(&self, _: &RoventCollectionRootAdmissionV3) -> Result<(), AdaptiveManagerError> {
        unreachable!("no root starts")
    }
    fn finish_deferred(
        &self,
        _: &RoventNoCaptureRootCapabilityV3,
        _: &RoventOwnNoCaptureFenceV3,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveManagerError> {
        unreachable!("no root starts")
    }
    fn retain_deferred_cleanup(
        &self,
        _: Arc<Mutex<RoventBoardDeferredCleanupV3>>,
    ) -> Result<RoventBoardCustodyReturnV3, AdaptiveManagerError> {
        unreachable!("no root starts")
    }
    fn handoff_sender_root(
        &self,
        _: &RoventCollectionRootAdmissionV3,
    ) -> Result<
        SyncSender<Arc<Mutex<RoventBoardCollectionHandoffV3<VirtualReviewTicketOutput>>>>,
        AdaptiveManagerError,
    > {
        unreachable!("no root starts")
    }
    fn try_take_root_custody(
        &self,
        _: &Arc<Mutex<RoventBoardCollectionHandoffV3<VirtualReviewTicketOutput>>>,
    ) -> Result<
        Option<RoventBoardCollectionRootCustodyV3<VirtualReviewTicketOutput>>,
        AdaptiveManagerError,
    > {
        unreachable!("no root starts")
    }
    fn return_root_custody(
        &self,
        _: Arc<Mutex<RoventBoardCollectionRootCustodyV3<VirtualReviewTicketOutput>>>,
    ) -> Result<RoventBoardCustodyReturnV3, AdaptiveManagerError> {
        unreachable!("no root starts")
    }
}

fn owner(drops: &Arc<AtomicUsize>, finish: Finish) -> Arc<RoventOwnCollectionLaunchOwnerV3> {
    Arc::new(RoventOwnCollectionLaunchOwnerV3::new(
        Vec::new(),
        Vec::new(),
        Arc::new(Hooks {
            drops: Arc::clone(drops),
            finish,
        }),
        Box::new(Observer),
    ))
}

#[test]
fn unissued_reclaim_releases_the_original_hook() {
    let drops = Arc::new(AtomicUsize::new(0));
    let owner = owner(&drops, Finish::Ready);
    drop(owner.take_unissued_observer().expect("unissued observer"));
    assert!(owner.observer_proxy().is_err());
    drop(owner);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn issued_proxy_without_reclaim_preserves_hook() {
    let drops = Arc::new(AtomicUsize::new(0));
    let owner = owner(&drops, Finish::Ready);
    let proxy = owner.observer_proxy().expect("original proxy");
    drop(owner);
    drop(proxy);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
}

#[test]
fn failed_or_panicked_settlement_retains_original_hook() {
    for finish in [Finish::Error, Finish::Panic] {
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = owner(&drops, finish);
        let proxy = owner.observer_proxy().expect("original proxy");
        // Test-only synthetic marker exercises custody branches, not native completion.
        owner.mark_native_session_finished();
        assert!(owner.try_settle_launch().is_err());
        assert!(owner.try_settle_launch().is_err());
        drop(owner);
        drop(proxy);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn permits_require_the_same_owner_and_native_marker() {
    let drops = Arc::new(AtomicUsize::new(0));
    let first = owner(&drops, Finish::Ready);
    let second = owner(&drops, Finish::Ready);
    let permit = RoventOwnCollectionSettlementPermitV3 {
        owner: Arc::clone(&first),
        hook_custody: Some(Box::new(())),
    };
    assert!(matches!(
        second.try_reclaim_observer(&permit),
        Err(RoventOwnObserverReclaimErrorV3::WrongOwner)
    ));
    assert!(matches!(
        first.try_reclaim_observer(&permit),
        Err(RoventOwnObserverReclaimErrorV3::WrongOwner)
    ));
    assert!(first.native_session_fence().is_none());
}
