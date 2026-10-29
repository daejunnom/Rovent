//! CPU-only regression checks for original V2 Collection observer custody.

use std::sync::{Arc, Mutex, mpsc};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::collection::run_pc_collection_v2_with_resource_admission;
use super::collection_v2_observer::{V2ObserverCustody, V2ObserverReclaimError};
use crate::pals_adaptive::{AdaptiveManagerError, RoventCollectionRootAdmissionV2};
use crate::pals_native::ticketed::{
    NativeTicketQuestion, TicketExecutionEvidence, TicketPreparation,
    VirtualReviewCaptureFailure, VirtualReviewCollectionObserver,
    VirtualReviewPhysicalInvocationEvidence, VirtualReviewPreparedTicket,
};
use crate::rovent_bootstrap::pc_session::{
    RoventPcCollectionAdmissionFailureV2, RoventPcCollectionAdmissionPortV2,
    RoventPcCollectionHandoffV2, RoventPcCollectionHooksV2,
};
use rz_contracts::pals::{RepresentationKey, Role, RoleOutput, SearchAuthority};
use rz_contracts::pals_adaptive::{ReviewViewSeal, RoleTicketEnvelope};
use rz_contracts::{ClockDomain, Deadline, Digest, GameGeneration, MonotonicTick, OwnerId,
    PrecisionProfile, ProcessEpoch, RequestId, RootGeneration, StateIdentity, StateRevision};
use rz_search::pals::coordinator::{PhysicalRoleCompletion, RoleFailure};
use rz_search::pals::review_context::ReviewView;

struct Observer {
    classified_calls: Arc<AtomicUsize>,
}
impl VirtualReviewCollectionObserver for Observer {
    fn submitted(&mut self, _: &RoleTicketEnvelope, _: &TicketPreparation<ReviewView, NativeTicketQuestion>, _: &VirtualReviewPreparedTicket) -> Result<(), RoleFailure> {
        Err(RoleFailure::InvalidOutput)
    }
    fn completed(&mut self, _: &RoleTicketEnvelope, _: PhysicalRoleCompletion, _: Option<&RoleOutput>, _: Option<TicketExecutionEvidence>, _: Option<VirtualReviewPhysicalInvocationEvidence>) -> Result<(), RoleFailure> {
        Err(RoleFailure::InvalidOutput)
    }
    fn retired(&mut self, _: &RoleTicketEnvelope, _: bool) -> Result<(), RoleFailure> {
        Err(RoleFailure::InvalidOutput)
    }
    fn retired_classified(&mut self, _: &RoleTicketEnvelope, _: bool) -> Result<(), VirtualReviewCaptureFailure> {
        self.classified_calls.fetch_add(1, Ordering::SeqCst);
        Err(VirtualReviewCaptureFailure::Observation(RoleFailure::Canceled))
    }
}
struct Hooks {
    calls: Arc<AtomicUsize>,
    handoff: mpsc::SyncSender<Arc<Mutex<RoventPcCollectionHandoffV2>>>,
}
impl RoventPcCollectionHooksV2 for Hooks {
    fn register_startup_control(&self, _: &crate::rovent_bootstrap::RoventStartupControl) -> Result<(), AdaptiveManagerError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn admit(&self, _: &mut RoventPcCollectionAdmissionPortV2<'_>, _: RoventCollectionRootAdmissionV2) -> Result<(), RoventPcCollectionAdmissionFailureV2> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        unreachable!("invalid CLI cannot admit a root")
    }
    fn pump(&self) -> Result<(), AdaptiveManagerError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn revoke(&self) { self.calls.fetch_add(1, Ordering::SeqCst); }
    fn handoff_sender(&self) -> mpsc::SyncSender<Arc<Mutex<RoventPcCollectionHandoffV2>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.handoff.clone()
    }
}

fn ticket() -> RoleTicketEnvelope {
    let epoch = ProcessEpoch(17);
    let authority = SearchAuthority { epoch, game: GameGeneration(1), root: RootGeneration(2), implementation: Digest([1; 32]) };
    RoleTicketEnvelope {
        id: RequestId::new(epoch, 1), authority, question: Digest([2; 32]),
        view: ReviewViewSeal { authority, view_sequence: 1, context_policy: Digest([3; 32]), view_digest: Digest([4; 32]),
            root_state: StateIdentity { owner: OwnerId(3), revision: StateRevision(1), semantic: Digest([5; 32]) },
            actual_tokens: 0, selected_records: 0, anchors: 1, length_bucket: 8 },
        representation: RepresentationKey { input: Digest([6; 32]), history: Digest([7; 32]), candidates: Digest([8; 32]), mask_positions: Digest([9; 32]), model: Digest([10; 32]), encoding: Digest([11; 32]), precision: PrecisionProfile::Fp32, frozen_epoch: 1, record_revision: 1, role: Role::Proposer },
        role: Role::Proposer, original_deadline: Deadline { clock: ClockDomain(epoch), at: MonotonicTick(1) },
    }
}

#[test]
fn v2_collection_invalid_cli_returns_original_observer_and_hooks_without_native_start() {
    let observer: Box<dyn VirtualReviewCollectionObserver> = Box::new(Observer { classified_calls: Arc::new(AtomicUsize::new(0)) });
    let original: *const dyn VirtualReviewCollectionObserver = &*observer;
    let calls = Arc::new(AtomicUsize::new(0));
    let (handoff, _rx) = mpsc::sync_channel(1);
    let hooks: Arc<dyn RoventPcCollectionHooksV2> = Arc::new(Hooks { calls: Arc::clone(&calls), handoff });
    let resource_called = Arc::new(AtomicUsize::new(0));
    let resource_seen = Arc::clone(&resource_called);
    let mut rejected = match run_pc_collection_v2_with_resource_admission(
        vec!["--invalid".to_owned()], vec![], observer, Arc::clone(&hooks),
        move |_| { resource_seen.fetch_add(1, Ordering::SeqCst); Ok(()) },
    ) { Err(rejected) => rejected, Ok(()) => panic!("invalid CLI must be rejected") };
    assert_eq!(rejected.error.stage, "Arguments");
    assert!(Arc::ptr_eq(rejected.hooks(), &hooks));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(resource_called.load(Ordering::SeqCst), 0);
    let returned = rejected.take_settled_observer().expect("original observer returned");
    assert!(std::ptr::eq(original, &*returned));
    assert!(rejected.take_settled_observer().is_none());
}

#[test]
fn v2_collection_proxy_preserves_classified_result_and_original_identity() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observer: Box<dyn VirtualReviewCollectionObserver> = Box::new(Observer { classified_calls: Arc::clone(&calls) });
    let original: *const dyn VirtualReviewCollectionObserver = &*observer;
    let mut custody = V2ObserverCustody::new(observer);
    let mut proxy = custody.proxy().expect("proxy");
    assert_eq!(proxy.retired_classified(&ticket(), true), Err(VirtualReviewCaptureFailure::Observation(RoleFailure::Canceled)));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    drop(proxy);
    let returned = match custody.try_reclaim_after_verified_boundary() { Ok(value) => value, Err(_) => panic!("settled observer should be returned") };
    assert!(std::ptr::eq(original, &*returned));
    assert!(matches!(custody.try_reclaim_after_verified_boundary(), Err(V2ObserverReclaimError::AlreadyReclaimed)));
}

#[test]
fn v2_collection_custody_refuses_reclaim_while_proxy_exists() {
    let observer: Box<dyn VirtualReviewCollectionObserver> = Box::new(Observer { classified_calls: Arc::new(AtomicUsize::new(0)) });
    let mut custody = V2ObserverCustody::new(observer);
    let proxy = custody.proxy().expect("proxy");
    assert!(matches!(custody.try_reclaim_after_verified_boundary(), Err(V2ObserverReclaimError::ProxyStillOwned)));
    drop(proxy);
    assert!(custody.try_reclaim_after_verified_boundary().is_ok());
}
