//! P-C /3 root resource custody primitives.
//!
//! This ledger alone is not a collection hook, observer, physical fence, or
//! successful run. Root setup and actual worker integration remain separate.
use super::launch::PreparedPcCollectionInputsV3;
use super::speculative_route_v3::SpeculativeRootV3;
use super::teacher::v3_pump::TeacherPumpV3;
use super::teacher::v3_session::TeacherSessionStagingV3;
use super::native_capture::RoventNativeCaptureRegistration;
use super::observer_lock_v3::RoventCollectionOwnerLockV3;
use super::pc_worker_v3::{ClaimedUnstartedPollV3, PcConsumerRetirementV3, PcRootRecoveryV3, PcWorkerSupervisorV3, PcSupervisorJoinedProofV3, SupervisorStartCustodyV3};
use super::{ArenaError, invalid};
use rz_uci::pals_adaptive::{AdaptiveCollectionBoundaryV2, AdaptiveCollectionError, AdaptiveCollectionObserverBoxV3,
    AdaptiveManagerError, RoleOnlyCollectionV3InstallAttempt, RoventCollectionRootAdmissionV3, RoventNoCaptureRootCapabilityV3,
    RoventRoleOnlyNoCaptureFenceV3, RoventTeacherScopePhysicalAckV3};
use rz_uci::rovent_bootstrap::pc_collection_v3::{
    RoventPcCollectionAdmissionFailureV3, RoventPcCollectionAdmissionOutcomeV3,
    RoventPcCollectionAdmissionPortV3, RoventPcCollectionHandoffV3,
    RoventPcCollectionHooksV3, RoventPcCollectionRootCustodyV3,
    RoventPcCollectionWorkerJoinFailureV3, RoventPcCollectionWorkerV3,
    RoventPcCustodyReturnV3, RoventPcDeferredCleanupV3,
};
use rz_uci::rovent_bootstrap::launch::{RoventPcCollectionLaunchOwnerV3, RoventPcCollectionNativeFenceV3};
use rz_uci::rovent_bootstrap::RoventStartupControl;
use rz_uci::pals_native::ticketed::VirtualReviewTicketOutput;
use rz_uci::rovent_external_launch::RoventTeacherScopeAggregateAckV3;
use rz_search::external_cpu::process_scope::{CollectionMemberPhysicalState, CollectionMemberWorkerState};
use rz_search::pals::engine::pc_port::PcPortQuestion;
use std::sync::{Arc, Mutex, OnceLock, TryLockError, Weak, mpsc};

/// Quota is per original root. Unresolved roots retain their reservation.
struct RootQuotaV3 {
    admission: RoventCollectionRootAdmissionV3,
    worker_reserved: bool,
    teacher_reserved: bool,
    joined: Option<Arc<Mutex<RoventPcCollectionRootCustodyV3>>>,
    /// Holds the original aggregate and associated handoff alive after quota
    /// release; this does not imply that native or route retirement succeeded.
    retired_teacher_handoff: Option<Arc<Mutex<RoventPcCollectionHandoffV3>>>,
    unclaimed_teacher_ack: Option<RoventTeacherScopePhysicalAckV3>,
    teacher_seal_unknown: bool,
    teacher_seal_panic: Option<Box<dyn std::any::Any + Send>>,
    teacher_seal_error: Option<AdaptiveCollectionError>,
    unstarted_recovery: Option<Arc<Mutex<PcRootRecoveryV3>>>,
    claimed_unstarted_recovery: Option<Arc<Mutex<PcRootRecoveryV3>>>,
}

pub(crate) enum PcQuotaOutcomeV3 {
    Reserved,
    AlreadyReserved,
    AlreadyRetired,
    Busy,
    Exhausted,
}

pub(crate) enum PcQuotaEvidenceErrorV3 {
    Busy,
    Poisoned,
    WrongRoot,
    AlreadyRetired,
    MissingTeacherAggregate,
}

/// The caller serializes operations on this ledger. Its retained roots may
/// outlive active worker quota; this keeps historical original evidence.
pub(crate) struct PcRootQuotaLedgerV3 {
    inputs: Arc<PreparedPcCollectionInputsV3>,
    roots: Vec<RootQuotaV3>,
}

/// Recheck the original issuer/roster, rather than accepting an arbitrary
/// aggregate or a copied numeric worker count as no-worker evidence.
fn claimed_unstarted_matches(
    aggregate: &RoventTeacherScopeAggregateAckV3,
    admission: &RoventCollectionRootAdmissionV3,
) -> bool {
    if !aggregate.matches_admission(admission) || !aggregate.retired().is_empty() {
        return false;
    }
    let (Some(physical), Some(worker_ack)) =
        (aggregate.physical().scope_outcome(), aggregate.workers()) else { return false; };
    if !worker_ack.matches_admission(admission) { return false; }
    let workers = worker_ack.outcome();
    physical.same_issuer_as_worker(workers)
        && !physical.members().is_empty()
        && physical.members().len() == workers.members().len()
        && physical.members().iter().zip(workers.members()).all(|((pr, ps), (wr, ws))|
            pr == wr && *ps == CollectionMemberPhysicalState::NeverIssuedNoChild
                && *ws == CollectionMemberWorkerState::NoThread)
}

/// A route unlink proof retains the original ledger and pump rather than
/// copying their observations. Construction and every use recheck both.
pub(crate) struct RoventTeacherRouteRetirementV3 {
    ledger: Arc<Mutex<PcRootQuotaLedgerV3>>,
    route: SpeculativeRootV3,
    pump: Arc<Mutex<TeacherPumpV3>>,
    recovery: Arc<Mutex<PcRootRecoveryV3>>,
}

impl RoventTeacherRouteRetirementV3 {
    /// `None` is retryable and makes no assertion that any work is complete.
    fn try_prepare(
        ledger: Arc<Mutex<PcRootQuotaLedgerV3>>,
        route: SpeculativeRootV3,
        pump: Arc<Mutex<TeacherPumpV3>>,
        recovery: Arc<Mutex<PcRootRecoveryV3>>,
    ) -> Result<Option<Self>, ArenaError> {
        let proof = Self { ledger, route, pump, recovery };
        match proof.matches_original_route(&proof.route)? {
            Some(true) => Ok(Some(proof)),
            Some(false) | None => Ok(None),
        }
    }

    /// Called immediately before unlink. Busy is not an identity failure.
    /// Teacher aggregate, actual joined worker, native root fence, and the
    /// original route receiver's accounting are independent requirements.
    pub(crate) fn matches_original_route(
        &self,
        requested: &SpeculativeRootV3,
    ) -> Result<Option<bool>, ArenaError> {
        if !self.route.same_original_route(requested) { return Ok(Some(false)); }
        let ledger = match self.ledger.try_lock() {
            Ok(value) => value,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => return Err(invalid("V3 root ledger poisoned")),
        };
        let Some(root) = ledger.roots.iter().find(|root|
            self.route.matches_admission(&root.admission)) else { return Ok(Some(false)); };
        {
            let recovery = match self.recovery.try_lock() {
                Ok(value) => value,
                Err(TryLockError::WouldBlock) => return Ok(None),
                Err(TryLockError::Poisoned(_)) => return Err(invalid("V3 original consumer custody poisoned")),
            };
            if !recovery.matches_admission(&root.admission)
                || !recovery.matches_route_pump(&self.route, &self.pump) {
                return Ok(Some(false));
            }
            match recovery.consumer_route_retirement() {
                PcConsumerRetirementV3::Ready => {},
                PcConsumerRetirementV3::Pending | PcConsumerRetirementV3::Rejected
                | PcConsumerRetirementV3::Unknown => return Ok(Some(false)),
            }
        }
        if root.worker_reserved || root.teacher_reserved { return Ok(Some(false)); }
        let Some(joined_owner) = &root.joined else { return Ok(Some(false)); };
        let joined = match joined_owner.try_lock() {
            Ok(value) => value,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => return Err(invalid("V3 joined custody poisoned")),
        };
        let Some(teacher_handoff) = &root.retired_teacher_handoff else { return Ok(Some(false)); };
        if !Arc::ptr_eq(joined.handoff(), teacher_handoff) { return Ok(Some(false)); }
        {
            let mut handoff = match joined.handoff().try_lock() {
                Ok(value) => value,
                Err(TryLockError::WouldBlock) => return Ok(None),
                Err(TryLockError::Poisoned(_)) => return Err(invalid("V3 original handoff poisoned")),
            };
            let (admission, _, _settlement, fence) = handoff.parts_mut();
            if !admission.same_original_admission(&root.admission)
                || !fence.permits_semantic_closure()
            { return Ok(Some(false)); }
            if !handoff.teacher_aggregate_ack().is_some_and(|ack|
                ack.matches_admission(&root.admission)) { return Ok(Some(false)); }
        }
        let pump = match self.pump.try_lock() {
            Ok(value) => value,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => return Err(invalid("V3 original teacher pump poisoned")),
        };
        pump.try_retirement_ready(&self.route)
    }
}

impl PcRootQuotaLedgerV3 {
    pub(crate) fn new(inputs: Arc<PreparedPcCollectionInputsV3>) -> Self {
        Self { inputs, roots: Vec::new() }
    }

    pub(crate) fn inputs(&self) -> &Arc<PreparedPcCollectionInputsV3> { &self.inputs }

    /// Reserve before starting a root worker. A duplicate original root is
    /// never reissued as a new owner, even after its previous worker joined.
    pub(crate) fn reserve_worker(
        &mut self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> PcQuotaOutcomeV3 {
        if let Some(root) = self.roots.iter().find(|root| root.admission.same_original_admission(admission)) {
            return if root.worker_reserved { PcQuotaOutcomeV3::AlreadyReserved }
                else { PcQuotaOutcomeV3::AlreadyRetired };
        }
        let plan = self.inputs.plan();
        if self.roots.len() >= usize::from(plan.maximum_roots)
            || self.roots.iter().filter(|root| root.worker_reserved).count()
                >= usize::from(plan.maximum_root_workers)
        { return PcQuotaOutcomeV3::Exhausted; }
        if self.roots.try_reserve(1).is_err() { return PcQuotaOutcomeV3::Busy; }
        self.roots.push(RootQuotaV3 {
            admission: admission.clone(), worker_reserved: true, teacher_reserved: false,
            joined: None, retired_teacher_handoff: None, unclaimed_teacher_ack: None,
            teacher_seal_unknown: false, teacher_seal_panic: None, teacher_seal_error: None,
            unstarted_recovery: None, claimed_unstarted_recovery: None,
        });
        PcQuotaOutcomeV3::Reserved
    }

    /// This separate reservation must be held before any teacher actor starts
    /// and until the original root's physical aggregate is retained.
    pub(crate) fn reserve_teacher(
        &mut self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<PcQuotaOutcomeV3, PcQuotaEvidenceErrorV3> {
        let Some(index) = self.roots.iter().position(|root| root.admission.same_original_admission(admission))
        else { return Err(PcQuotaEvidenceErrorV3::WrongRoot); };
        if self.roots[index].teacher_reserved { return Ok(PcQuotaOutcomeV3::AlreadyReserved); }
        if !self.roots[index].worker_reserved || self.roots[index].retired_teacher_handoff.is_some()
            || self.roots[index].unclaimed_teacher_ack.is_some() || self.roots[index].teacher_seal_unknown {
            return Err(PcQuotaEvidenceErrorV3::AlreadyRetired);
        }
        if self.roots.iter().filter(|root| root.teacher_reserved).count()
            >= usize::from(self.inputs.plan().maximum_simultaneous_teacher_roots)
        { return Ok(PcQuotaOutcomeV3::Exhausted); }
        self.roots[index].teacher_reserved = true;
        Ok(PcQuotaOutcomeV3::Reserved)
    }

    /// A worker slot is returned only with the original V3 joined custody
    /// returned by the manager. The caller retains the same Arc on failure.
    pub(crate) fn retain_joined_worker(
        &mut self,
        admission: &RoventCollectionRootAdmissionV3,
        custody: Arc<Mutex<RoventPcCollectionRootCustodyV3>>,
    ) -> Result<(), PcQuotaEvidenceErrorV3> {
        let Some(root) = self.roots.iter_mut().find(|root| root.admission.same_original_admission(admission))
        else { return Err(PcQuotaEvidenceErrorV3::WrongRoot); };
        if !root.worker_reserved || root.joined.is_some() {
            return Err(PcQuotaEvidenceErrorV3::AlreadyRetired);
        }
        let handoff_owner = match custody.try_lock() {
            Ok(original) => Arc::clone(original.handoff()),
            Err(TryLockError::WouldBlock) => return Err(PcQuotaEvidenceErrorV3::Busy),
            Err(TryLockError::Poisoned(_)) => return Err(PcQuotaEvidenceErrorV3::Poisoned),
        };
        if root.retired_teacher_handoff.as_ref().is_some_and(|original|
            !Arc::ptr_eq(original, &handoff_owner)) {
            return Err(PcQuotaEvidenceErrorV3::WrongRoot);
        }
        {
            let mut handoff = match handoff_owner.try_lock() {
                Ok(handoff) => handoff,
                Err(TryLockError::WouldBlock) => return Err(PcQuotaEvidenceErrorV3::Busy),
                Err(TryLockError::Poisoned(_)) => return Err(PcQuotaEvidenceErrorV3::Poisoned),
            };
            if !handoff.parts_mut().0.same_original_admission(admission) {
                drop(handoff);
                return Err(PcQuotaEvidenceErrorV3::WrongRoot);
            }
        }
        root.joined = Some(custody);
        root.worker_reserved = false;
        Ok(())
    }

    /// A live process, logical cancel, worker join alone, or no GPU result is
    /// not physical teacher retirement. Keep the same original handoff Arc.
    pub(crate) fn retain_teacher_aggregate(
        &mut self,
        admission: &RoventCollectionRootAdmissionV3,
        handoff: Arc<Mutex<RoventPcCollectionHandoffV3>>,
    ) -> Result<(), PcQuotaEvidenceErrorV3> {
        let Some(root) = self.roots.iter_mut().find(|root| root.admission.same_original_admission(admission))
        else { return Err(PcQuotaEvidenceErrorV3::WrongRoot); };
        if !root.teacher_reserved || root.retired_teacher_handoff.is_some() {
            return Err(PcQuotaEvidenceErrorV3::AlreadyRetired);
        }
        if let Some(joined) = &root.joined {
            let joined = match joined.try_lock() {
                Ok(value) => value,
                Err(TryLockError::WouldBlock) => return Err(PcQuotaEvidenceErrorV3::Busy),
                Err(TryLockError::Poisoned(_)) => return Err(PcQuotaEvidenceErrorV3::Poisoned),
            };
            if !Arc::ptr_eq(joined.handoff(), &handoff) {
                return Err(PcQuotaEvidenceErrorV3::WrongRoot);
            }
        }
        {
            let mut original = match handoff.try_lock() {
                Ok(original) => original,
                Err(TryLockError::WouldBlock) => return Err(PcQuotaEvidenceErrorV3::Busy),
                Err(TryLockError::Poisoned(_)) => return Err(PcQuotaEvidenceErrorV3::Poisoned),
            };
            if !original.parts_mut().0.same_original_admission(admission) {
                return Err(PcQuotaEvidenceErrorV3::WrongRoot);
            }
            if !original.teacher_aggregate_ack().is_some_and(|ack| ack.matches_admission(admission)) {
                return Err(PcQuotaEvidenceErrorV3::MissingTeacherAggregate);
            }
        }
        root.retired_teacher_handoff = Some(handoff);
        root.teacher_reserved = false;
        Ok(())
    }
}

type PcHandoffV3 = Arc<Mutex<RoventPcCollectionHandoffV3>>;
type PcObserverV3 = AdaptiveCollectionObserverBoxV3<PcPortQuestion, VirtualReviewTicketOutput>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PcRootDispatchV3 {
    BeforeDispatch,
    DispatchingUnknown,
    Enqueued,
}

/// Original game abort receipt combined with this slot's pre-dispatch and
/// no-reservation state. Neither worker nor device completion is implied.
struct GuardAbortedNoDispatchV3 {
    abort: super::game_authority_v3::GameRootAbortReceiptV3<PcWorkerSupervisorV3>,
}

/// Retains the actual same-game, same-root terminal send outcome. This is
/// not native, teacher, output or GPU completion evidence.
struct DispatchRejectedNoWorkerV3 {
    receipt: super::game_authority_v3::GameDispatchReceiptV3<PcWorkerSupervisorV3>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum RootNotAcceptingQuotaV3 { Unreserved, Returned }

/// A same-root authority observation plus this slot's pre-dispatch custody.
/// The particular reason for nonacceptance remains unknown.
struct RootNotAcceptingNoDispatchV3 {
    game: Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>,
    admission: RoventCollectionRootAdmissionV3,
    quota: RootNotAcceptingQuotaV3,
}

struct PcRootSlotV3 {
    admission: RoventCollectionRootAdmissionV3,
    game_reservation: Option<super::game_authority_v3::GameRootReservationV3<PcWorkerSupervisorV3>>,
    game_abort: Option<super::game_authority_v3::GameRootAbortReceiptV3<PcWorkerSupervisorV3>>,
    guard_aborted: Option<GuardAbortedNoDispatchV3>,
    dispatch_claim: Option<super::game_authority_v3::GameDispatchClaimV3<PcWorkerSupervisorV3>>,
    dispatch_receipt: Option<super::game_authority_v3::GameDispatchReceiptV3<PcWorkerSupervisorV3>>,
    dispatch_rejected: Option<DispatchRejectedNoWorkerV3>,
    root_not_accepting: Option<RootNotAcceptingNoDispatchV3>,
    root_authority_failure: Option<rz_contracts::ContractError>,
    game_registered: bool,
    quota_registered: bool,
    dispatch: PcRootDispatchV3,
    requested: bool,
    installed: bool,
    unstarted_terminal: bool,
    claimed_unstarted_terminal: bool,
    recovery: Option<Arc<Mutex<PcRootRecoveryV3>>>,
    owner: Option<RoventCollectionOwnerLockV3>,
    route: Option<SpeculativeRootV3>,
    route_retirement: Option<RoventTeacherRouteRetirementV3>,
    stage: Option<Arc<Mutex<TeacherSessionStagingV3>>>,
    pump: Option<Arc<Mutex<TeacherPumpV3>>>,
    observer: Option<PcObserverV3>,
    worker: Option<RoventPcCollectionWorkerV3>,
    handoff: Option<PcHandoffV3>,
    joined: Option<RoventPcCollectionRootCustodyV3>,
    custody_taken: bool,
    returned: Option<Arc<Mutex<RoventPcCollectionRootCustodyV3>>>,
    join_failure: Option<RoventPcCollectionWorkerJoinFailureV3>,
    join_panic: Option<Box<dyn std::any::Any + Send>>,
}

struct PcBridgeStateV3 {
    roots: Vec<PcRootSlotV3>,
    deferred: Vec<Arc<Mutex<RoventPcDeferredCleanupV3>>>,
}

impl PcRootSlotV3 {
    /// Before a dispatch claim exists, observe this exact native root while
    /// holding bridge state then ledger. An authority lock conflict is
    /// Pending; a poisoned/other failure is retained for this root.
    fn try_seal_root_not_accepting(
        &mut self,
        game: &Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>,
        ledger: &mut PcRootQuotaLedgerV3,
    ) -> Result<Option<bool>, rz_contracts::ContractError> {
        if self.root_not_accepting.is_some() {
            return Ok(Some(self.root_not_accepting_valid(game, ledger)));
        }
        if let Some(error) = self.root_authority_failure { return Err(error); }
        if !self.local_no_worker_owners_absent()
            || self.dispatch_receipt.is_some() || self.dispatch_rejected.is_some() {
            return Ok(Some(false));
        }
        let quota = if self.quota_registered {
            let Some(index) = ledger.roots.iter().position(|root|
                root.admission.same_original_admission(&self.admission)) else { return Ok(Some(false)); };
            let root = &ledger.roots[index];
            if !root.worker_reserved || root.teacher_reserved || root.joined.is_some()
                || root.retired_teacher_handoff.is_some() || root.unclaimed_teacher_ack.is_some()
                || root.teacher_seal_unknown || root.teacher_seal_panic.is_some()
                || root.teacher_seal_error.is_some() || root.unstarted_recovery.is_some()
                || root.claimed_unstarted_recovery.is_some() { return Ok(Some(false)); }
            Some(index)
        } else {
            if ledger.roots.iter().any(|root|
                root.admission.same_original_admission(&self.admission)
                    || root.admission.root_admission().authority() == self.admission.root_admission().authority()) {
                return Ok(Some(false));
            }
            None
        };
        match self.admission.root_admission().try_accepts_original() {
            Ok(true) => return Ok(Some(false)),
            Ok(false) => {},
            Err(error) if error.code == rz_contracts::ErrorCode::ResourceExhausted => return Ok(None),
            Err(error) => { self.root_authority_failure = Some(error); return Err(error); },
        }
        let proof = RootNotAcceptingNoDispatchV3 {
            game: Arc::clone(game), admission: self.admission.clone(),
            quota: if quota.is_some() { RootNotAcceptingQuotaV3::Returned } else { RootNotAcceptingQuotaV3::Unreserved },
        };
        // After the real observation, preserve its original owner and return
        // only this root's unused reservation; no fallible operation intervenes.
        self.root_not_accepting = Some(proof);
        if let Some(index) = quota { ledger.roots[index].worker_reserved = false; }
        Ok(Some(true))
    }

    fn root_not_accepting_valid(
        &self,
        game: &Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>,
        ledger: &PcRootQuotaLedgerV3,
    ) -> bool {
        let Some(proof) = &self.root_not_accepting else { return false; };
        if !Arc::ptr_eq(&proof.game, game) || !proof.admission.same_original_admission(&self.admission)
            || !self.local_no_worker_owners_absent() || self.dispatch_receipt.is_some()
            || self.dispatch_rejected.is_some() || self.root_authority_failure.is_some() { return false; }
        match proof.quota {
            RootNotAcceptingQuotaV3::Unreserved => !self.quota_registered && !ledger.roots.iter().any(|root|
                root.admission.same_original_admission(&self.admission)
                    || root.admission.root_admission().authority() == self.admission.root_admission().authority()),
            RootNotAcceptingQuotaV3::Returned => self.quota_registered && ledger.roots.iter().find(|root|
                root.admission.same_original_admission(&self.admission)).is_some_and(|root|
                    !root.worker_reserved && !root.teacher_reserved && root.joined.is_none()
                    && root.retired_teacher_handoff.is_none() && root.unclaimed_teacher_ack.is_none()
                    && !root.teacher_seal_unknown && root.teacher_seal_panic.is_none()
                    && root.teacher_seal_error.is_none() && root.unstarted_recovery.is_none()
                    && root.claimed_unstarted_recovery.is_none()),
        }
    }

    /// The caller holds this exact bridge state and ledger, in that order.
    /// Only a completed terminal send observation may retire this root's
    /// otherwise unused worker reservation. Busy/unknown retains the receipt.
    fn seal_dispatch_rejected_no_worker(
        &mut self,
        game: &Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>,
        ledger: &mut PcRootQuotaLedgerV3,
    ) -> Result<bool, &'static str> {
        if self.dispatch_rejected.is_some() {
            return self.dispatch_rejected_no_worker_valid(game, ledger);
        }
        if self.root_not_accepting.is_some() || self.root_authority_failure.is_some() { return Ok(false); }
        let Some(receipt) = self.dispatch_receipt.as_ref() else { return Ok(false); };
        if !receipt.matches_game(game) || !receipt.matches_root(&self.admission) {
            return Err("P-C V3 original dispatch receipt differs");
        }
        if !matches!(receipt.outcome(),
            super::game_authority_v3::GameDispatchOutcomeV3::Aborted
            | super::game_authority_v3::GameDispatchOutcomeV3::AdmissionSealed
            | super::game_authority_v3::GameDispatchOutcomeV3::Disconnected) {
            return Err("P-C V3 dispatch is not terminal and unissued");
        }
        if !self.local_no_worker_owners_absent() || !self.quota_registered {
            return Ok(false);
        }
        let Some(root) = ledger.roots.iter_mut().find(|root|
            root.admission.same_original_admission(&self.admission)) else { return Ok(false); };
        if !root.worker_reserved || root.teacher_reserved || root.joined.is_some()
            || root.retired_teacher_handoff.is_some() || root.unclaimed_teacher_ack.is_some()
            || root.teacher_seal_unknown || root.teacher_seal_panic.is_some()
            || root.teacher_seal_error.is_some() || root.unstarted_recovery.is_some()
            || root.claimed_unstarted_recovery.is_some() {
            return Ok(false);
        }
        // No fallible call or external callback between retaining the exact
        // receipt and returning the same root's reservation.
        if let Some(receipt) = self.dispatch_receipt.take() {
            self.dispatch_rejected = Some(DispatchRejectedNoWorkerV3 { receipt });
            root.worker_reserved = false;
            return Ok(true);
        }
        Ok(false)
    }

    fn local_no_worker_owners_absent(&self) -> bool {
        self.game_registered && self.game_reservation.is_none() && self.game_abort.is_none() && self.guard_aborted.is_none()
            && self.dispatch_claim.is_none()
            && self.dispatch == PcRootDispatchV3::BeforeDispatch
            && !self.requested && !self.installed && !self.unstarted_terminal
            && !self.claimed_unstarted_terminal && self.recovery.is_none()
            && self.owner.is_none() && self.route.is_none() && self.route_retirement.is_none()
            && self.stage.is_none() && self.pump.is_none() && self.observer.is_none()
            && self.worker.is_none() && self.handoff.is_none() && self.joined.is_none()
            && !self.custody_taken && self.returned.is_none()
            && self.join_failure.is_none() && self.join_panic.is_none()
    }

    fn dispatch_rejected_no_worker_valid(
        &self,
        game: &Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>,
        ledger: &PcRootQuotaLedgerV3,
    ) -> Result<bool, &'static str> {
        let Some(proof) = &self.dispatch_rejected else { return Ok(false); };
        let receipt = &proof.receipt;
        if !receipt.matches_game(game) || !receipt.matches_root(&self.admission)
            || !matches!(receipt.outcome(),
                super::game_authority_v3::GameDispatchOutcomeV3::Aborted
                | super::game_authority_v3::GameDispatchOutcomeV3::AdmissionSealed
                | super::game_authority_v3::GameDispatchOutcomeV3::Disconnected) {
            return Err("P-C V3 original terminal dispatch proof differs");
        }
        if !self.local_no_worker_owners_absent() || !self.quota_registered || self.dispatch_receipt.is_some() {
            return Ok(false);
        }
        let Some(root) = ledger.roots.iter().find(|root|
            root.admission.same_original_admission(&self.admission)) else { return Ok(false); };
        Ok(!root.worker_reserved && !root.teacher_reserved && root.joined.is_none()
            && root.retired_teacher_handoff.is_none() && root.unclaimed_teacher_ack.is_none()
            && !root.teacher_seal_unknown && root.teacher_seal_panic.is_none()
            && root.teacher_seal_error.is_none() && root.unstarted_recovery.is_none()
            && root.claimed_unstarted_recovery.is_none())
    }

    /// Called only with this bridge's state and ledger held in that order.
    /// The receipt is moved only after every no-dispatch check succeeds.
    fn seal_guard_aborted_no_dispatch(
        &mut self,
        game: &Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>,
        ledger: &PcRootQuotaLedgerV3,
    ) -> Result<bool, &'static str> {
        if self.guard_aborted.is_some() {
            return self.guard_aborted_no_dispatch_valid(game, ledger);
        }
        let Some(abort) = self.game_abort.as_ref() else { return Ok(false); };
        if !abort.matches_game(game) || !abort.matches_original(&self.admission) {
            return Err("P-C V3 game abort receipt differs");
        }
        if !self.no_dispatch_owners_absent(ledger) || !self.game_registered {
            return Ok(false);
        }
        // No external callback or fallible operation between take and retain.
        if let Some(abort) = self.game_abort.take() {
            self.guard_aborted = Some(GuardAbortedNoDispatchV3 { abort });
            return Ok(true);
        }
        Ok(false)
    }

    fn no_dispatch_owners_absent(&self, ledger: &PcRootQuotaLedgerV3) -> bool {
        self.dispatch == PcRootDispatchV3::BeforeDispatch
            && self.dispatch_claim.is_none() && self.dispatch_receipt.is_none() && self.dispatch_rejected.is_none()
            && self.root_not_accepting.is_none() && self.root_authority_failure.is_none()
            && !self.requested && !self.installed && !self.quota_registered
            && !self.unstarted_terminal && !self.claimed_unstarted_terminal
            && self.recovery.is_none() && self.owner.is_none() && self.route.is_none()
            && self.route_retirement.is_none() && self.stage.is_none() && self.pump.is_none()
            && self.observer.is_none() && self.worker.is_none() && self.handoff.is_none()
            && self.joined.is_none() && !self.custody_taken && self.returned.is_none()
            && self.join_failure.is_none() && self.join_panic.is_none()
            && !ledger.roots.iter().any(|root|
                root.admission.same_original_admission(&self.admission)
                    || root.admission.root_admission().authority() == self.admission.root_admission().authority())
    }

    fn guard_aborted_no_dispatch_valid(
        &self,
        game: &Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>,
        ledger: &PcRootQuotaLedgerV3,
    ) -> Result<bool, &'static str> {
        let Some(proof) = &self.guard_aborted else { return Ok(false); };
        if !proof.abort.matches_game(game) || !proof.abort.matches_original(&self.admission) {
            return Err("P-C V3 no-dispatch original receipt differs");
        }
        Ok(self.game_registered && self.game_abort.is_none() && self.no_dispatch_owners_absent(ledger))
    }
}

/// Same-launch native admission fence. It does not prove teacher, output, or
/// game completion and cannot be constructed without the original bridge.
pub(super) struct PcGameAdmissionSealV3 {
    game: Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>,
    _native: RoventPcCollectionNativeFenceV3,
}

/// The bridge observed the same supervisor's actual non-panicking join.
/// Weak references avoid a bridge/supervisor ownership cycle. This is not
/// evidence of successful work, output publication or GPU completion.
struct PcGameJoinedSealV3 {
    game: Weak<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>,
    custody: Weak<Mutex<SupervisorStartCustodyV3>>,
    supervisor: Weak<Mutex<PcWorkerSupervisorV3>>,
    native: Weak<RoventPcCollectionLaunchOwnerV3>,
}

impl PcGameAdmissionSealV3 {
    pub(super) fn matches_game(
        &self,
        game: &Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>,
    ) -> bool {
        Arc::ptr_eq(&self.game, game)
    }
}

/// Manager callbacks only exchange bounded state; setup runs on the separate
/// supervisor. The same bridge, native owner and inputs must remain retained.
pub(crate) struct RoventPcCollectionBridgeV3 {
    self_weak: Weak<RoventPcCollectionBridgeV3>,
    inputs: Arc<PreparedPcCollectionInputsV3>,
    game: Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>,
    registration: RoventNativeCaptureRegistration,
    ledger: Arc<Mutex<PcRootQuotaLedgerV3>>,
    startup: OnceLock<RoventStartupControl>,
    native_owner: OnceLock<Weak<RoventPcCollectionLaunchOwnerV3>>,
    game_admission_seal: OnceLock<PcGameAdmissionSealV3>,
    game_joined_seal: OnceLock<PcGameJoinedSealV3>,
    input_lock_identity: OnceLock<rz_uci::rovent_bootstrap::launch::RoventPcCollectionInputLockIdentityV3>,
    supervisor: OnceLock<Weak<Mutex<PcWorkerSupervisorV3>>>,
    state: Mutex<PcBridgeStateV3>,
    requests: super::game_authority_v3::GameDispatchChannelV3<RoventCollectionRootAdmissionV3>,
    request_receiver: Mutex<mpsc::Receiver<RoventCollectionRootAdmissionV3>>,
}

/// Supervisor retains all members and the worker outside the callback until
/// this exact bundle is accepted. It must reserve teacher quota before spawn.
pub(crate) struct PcPreparedRootV3 {
    pub admission: RoventCollectionRootAdmissionV3,
    pub owner: RoventCollectionOwnerLockV3,
    pub route: SpeculativeRootV3,
    pub stage: Arc<Mutex<TeacherSessionStagingV3>>,
    pub pump: Arc<Mutex<TeacherPumpV3>>,
    pub worker: RoventPcCollectionWorkerV3,
    pub recovery: Arc<Mutex<PcRootRecoveryV3>>,
}

pub(crate) enum PcPreparedOutcomeV3 {
    Accepted,
    Busy(PcPreparedRootV3),
    Rejected(&'static str, PcPreparedRootV3),
}

pub(crate) enum PcRootAccountingV3 {
    Pending,
    TeacherPending,
    PhysicallyRetired,
}

impl RoventPcCollectionBridgeV3 {
    /// Bounded recovery observation for quota contention. A busy or damaged
    /// root does not prevent examination of other original slots. The caller
    /// should use this on an actual quota wait, not on every manager tick.
    pub(crate) fn try_recover_nonaccepting_roots(&self) -> Result<bool, &'static str> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(false),
            Err(_) => return Err("P-C V3 root custody poisoned"),
        };
        let mut ledger = match self.ledger.try_lock() {
            Ok(ledger) => ledger,
            Err(TryLockError::WouldBlock) => return Ok(false),
            Err(_) => return Err("P-C V3 quota ledger poisoned"),
        };
        let mut recovered = false;
        for slot in &mut state.roots {
            if slot.root_not_accepting.is_some() { continue; }
            // The exact per-root failure is already retained by the slot.
            if matches!(slot.try_seal_root_not_accepting(&self.game, &mut ledger), Ok(Some(true))) {
                recovered = true;
            }
        }
        Ok(recovered)
    }

    /// Construction starts no native or teacher work. Boundary validation is
    /// performed by the original supervisor before it prepares a root.
    pub(crate) fn new(
        inputs: Arc<PreparedPcCollectionInputsV3>,
        registration: RoventNativeCaptureRegistration,
    ) -> Result<Arc<Self>, &'static str> {
        let max = usize::from(inputs.plan().maximum_roots);
        if !(1..=64).contains(&max) { return Err("P-C V3 root capacity invalid"); }
        let mut roots = Vec::new();
        let mut deferred = Vec::new();
        roots.try_reserve_exact(max).map_err(|_| "P-C V3 root allocation")?;
        deferred.try_reserve_exact(max).map_err(|_| "P-C V3 cleanup allocation")?;
        let (requests, receiver) = mpsc::sync_channel(max);
        let game = super::game_authority_v3::GameAuthorityV3::new_candidate(Arc::clone(&inputs));
        let requests = super::game_authority_v3::GameDispatchChannelV3::new(requests);
        match game.try_bind_dispatch_channel(&requests) {
            super::game_authority_v3::GameStepV3::Ready => {},
            _ => return Err("P-C V3 original game channel binding failed"),
        }
        Ok(Arc::new_cyclic(|self_weak| Self {
            self_weak: self_weak.clone(),
            ledger: Arc::new(Mutex::new(PcRootQuotaLedgerV3::new(Arc::clone(&inputs)))),
            inputs, game, registration, startup: OnceLock::new(), native_owner: OnceLock::new(), game_admission_seal: OnceLock::new(), game_joined_seal: OnceLock::new(), input_lock_identity: OnceLock::new(), supervisor: OnceLock::new(),
            state: Mutex::new(PcBridgeStateV3 { roots, deferred }), requests,
            request_receiver: Mutex::new(receiver),
        }))
    }

    pub(crate) fn inputs(&self) -> &Arc<PreparedPcCollectionInputsV3> { &self.inputs }
    pub(super) fn game_authority(&self) -> &Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>> { &self.game }
    /// Record only this launch's actual joined driver. A panicked driver is
    /// kept separate from physical cleanup and cannot grant game closure.
    pub(super) fn try_bind_game_joined(
        &self,
        proof: &PcSupervisorJoinedProofV3,
        custody: &Arc<Mutex<SupervisorStartCustodyV3>>,
    ) -> Result<Option<bool>, &'static str> {
        let Some(native) = self.native_owner.get().and_then(Weak::upgrade) else {
            return Err("P-C V3 original native owner absent");
        };
        let Some(supervisor) = self.supervisor.get().and_then(Weak::upgrade) else {
            return Err("P-C V3 original supervisor absent");
        };
        if !proof.matches_original(custody, &supervisor, &native) {
            return Err("P-C V3 joined driver belongs to another launch");
        }
        if proof.panicked() { return Ok(Some(false)); }
        if self.game_admission_seal.get().is_none() { return Ok(None); }
        let candidate = PcGameJoinedSealV3 {
            game: Arc::downgrade(&self.game),
            custody: Arc::downgrade(custody),
            supervisor: Arc::downgrade(&supervisor),
            native: Arc::downgrade(&native),
        };
        if let Err(candidate) = self.game_joined_seal.set(candidate) {
            let Some(original) = self.game_joined_seal.get() else { return Ok(None); };
            if !Weak::ptr_eq(&original.game, &candidate.game)
                || !Weak::ptr_eq(&original.custody, &candidate.custody)
                || !Weak::ptr_eq(&original.supervisor, &candidate.supervisor)
                || !Weak::ptr_eq(&original.native, &candidate.native) {
                return Err("P-C V3 joined driver identity changed");
            }
        }
        Ok(Some(true))
    }
    /// Bind the lock identity issued by this launch's authenticated gate
    /// before the native driver or any root can be started.
    pub(super) fn bind_input_lock_identity(
        &self,
        identity: rz_uci::rovent_bootstrap::launch::RoventPcCollectionInputLockIdentityV3,
    ) -> Result<(), &'static str> {
        match self.input_lock_identity.set(identity) {
            Ok(()) => Ok(()),
            Err(candidate) if self.input_lock_identity.get().is_some_and(|old| old.same_original(&candidate)) => Ok(()),
            Err(_) => Err("P-C V3 original input lock identity changed"),
        }
    }
    /// Fence only new admission on this original native launch. Root drain,
    /// teacher custody and durable publication remain independent.
    pub(super) fn try_game_admission_seal(
        self: &Arc<Self>,
        fence: RoventPcCollectionNativeFenceV3,
    ) -> Result<PcGameAdmissionSealV3, (RoventPcCollectionNativeFenceV3, &'static str)> {
        let Some(owner) = self.native_owner.get().and_then(Weak::upgrade) else {
            return Err((fence, "P-C V3 original native owner absent"));
        };
        let hooks: Arc<dyn RoventPcCollectionHooksV3> = self.clone();
        if !fence.matches_original_owner(&owner) || !fence.matches_original_hooks(&hooks) {
            return Err((fence, "P-C V3 native fence belongs to another owner"));
        }
        Ok(PcGameAdmissionSealV3 { game: Arc::clone(&self.game), _native: fence })
    }
    pub(crate) fn startup(&self) -> Option<&RoventStartupControl> { self.startup.get() }
    pub(crate) fn capture_registration(&self) -> &RoventNativeCaptureRegistration { &self.registration }

    /// Bind the one original supervisor while constructing the launch bundle.
    /// This neither starts its driver nor certifies any physical completion.
    pub(crate) fn bind_supervisor(
        self: &Arc<Self>, supervisor: &Arc<Mutex<PcWorkerSupervisorV3>>,
    ) -> Result<(), &'static str> {
        {
            let owner = supervisor.try_lock().map_err(|_| "P-C V3 supervisor unavailable")?;
            if !owner.matches_original_bridge(self) { return Err("P-C V3 supervisor bridge differs"); }
        }
        match self.game.try_bind_supervisor(supervisor) {
            super::game_authority_v3::GameStepV3::Ready | super::game_authority_v3::GameStepV3::Existing => {},
            _ => return Err("P-C V3 game supervisor binding unavailable"),
        }
        if let Some(original) = self.supervisor.get() {
            return if original.upgrade().is_some_and(|old| Arc::ptr_eq(&old, supervisor)) {
                Ok(())
            } else { Err("P-C V3 original supervisor changed") };
        }
        self.supervisor.set(Arc::downgrade(supervisor)).map_err(|_| "P-C V3 supervisor binding raced")
    }

    /// The bundle supplies this once, before any native root admission.
    pub(crate) fn bind_native_owner(
        self: &Arc<Self>, owner: &Arc<RoventPcCollectionLaunchOwnerV3>,
    ) -> Result<(), &'static str> {
        let hooks: Arc<dyn RoventPcCollectionHooksV3> = self.clone();
        if !owner.matches_original_hooks(&hooks) { return Err("P-C V3 native hook identity differs"); }
        if let Some(original) = self.native_owner.get() {
            return if original.upgrade().is_some_and(|old| Arc::ptr_eq(&old, owner)) { Ok(()) }
                else { Err("P-C V3 native launch owner changed") };
        }
        self.native_owner.set(Arc::downgrade(owner)).map_err(|_| "P-C V3 owner binding raced")
    }

    pub(crate) fn try_next_root(&self) -> Result<Option<RoventCollectionRootAdmissionV3>, &'static str> {
        let receiver = self.request_receiver.try_lock().map_err(|_| "P-C V3 request receiver unavailable")?;
        match receiver.try_recv() {
            Ok(value) => Ok(Some(value)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err("P-C V3 request channel closed"),
        }
    }

    /// Must be called before any actor starts. The same root's quota persists
    /// through failed/unknown cleanup until its original physical aggregate.
    pub(crate) fn try_reserve_teacher(&self, admission: &RoventCollectionRootAdmissionV3)
        -> Result<PcQuotaOutcomeV3, PcQuotaEvidenceErrorV3> {
        let mut ledger = match self.ledger.try_lock() {
            Ok(ledger) => ledger,
            Err(TryLockError::WouldBlock) => return Err(PcQuotaEvidenceErrorV3::Busy),
            Err(TryLockError::Poisoned(_)) => return Err(PcQuotaEvidenceErrorV3::Poisoned),
        };
        ledger.reserve_teacher(admission)
    }

    /// Recover this terminal unstarted root's teacher and uncreated-worker
    /// quota. The original no-scope acknowledgement and sealed recovery stay
    /// in the ledger; this neither settles native work nor reissues the scope.
    pub(crate) fn try_release_unstarted_teacher(
        &self, admission: &RoventCollectionRootAdmissionV3,
        recovery: &Arc<Mutex<PcRootRecoveryV3>>,
    ) -> Result<Option<bool>, &'static str> {
        let mut original = match recovery.try_lock() {
            Ok(owner) => owner,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 recovery owner poisoned"),
        };
        if !original.matches_admission(admission) || !original.definitely_unstarted() {
            return Err("P-C V3 teacher is not proven unstarted");
        }
        {
            let mut state = match self.state.try_lock() {
                Ok(state) => state,
                Err(TryLockError::WouldBlock) => return Ok(None),
                Err(_) => return Err("P-C V3 roots poisoned"),
            };
            let slot = state.roots.iter_mut().find(|slot| slot.admission.same_original_admission(admission))
                .ok_or("P-C V3 original request absent")?;
            if !slot.recovery.as_ref().is_some_and(|owner| Arc::ptr_eq(owner, recovery)) {
                return Err("P-C V3 recovery owner changed");
            }
            if slot.installed || slot.worker.is_some() || slot.route.is_some() || slot.observer.is_some() {
                return Err("P-C V3 native installation or worker already exists");
            }
            if !original.seal_definitely_unstarted() {
                return Err("P-C V3 original failure cannot be sealed unstarted");
            }
            slot.unstarted_terminal = true;
        }
        let mut ledger = match self.ledger.try_lock() {
            Ok(ledger) => ledger,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 quota ledger poisoned"),
        };
        let root = ledger.roots.iter_mut().find(|root| root.admission.same_original_admission(admission))
            .ok_or("P-C V3 quota root absent")?;
        if let Some(ack) = &root.unclaimed_teacher_ack {
            return if !root.teacher_reserved && !root.worker_reserved && !root.teacher_seal_unknown
                && root.unstarted_recovery.as_ref().is_some_and(|owner| Arc::ptr_eq(owner, recovery))
                && ack.matches_admission(admission) && ack.is_sealed_no_scope() {
                Ok(Some(true))
            } else { Err("P-C V3 retained no-scope witness is unresolved") };
        }
        if !root.teacher_reserved || root.teacher_seal_unknown {
            return Err("P-C V3 teacher reservation or no-scope seal uncertain");
        }
        // Persist the unknown state before entering the one-shot issuer.
        root.teacher_seal_unknown = true;
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| admission.seal_unclaimed_teacher_scope())) {
            Ok(Ok(ack)) if ack.matches_admission(admission) && ack.is_sealed_no_scope() => {
                root.unclaimed_teacher_ack = Some(ack);
                root.teacher_seal_unknown = false;
                root.teacher_reserved = false;
                root.unstarted_recovery = Some(Arc::clone(recovery));
                root.worker_reserved = false;
                Ok(Some(true))
            }
            Ok(Ok(ack)) => {
                // A mismatched original witness remains owned, but is not
                // permission to return quota or retry the one-shot claim.
                root.unclaimed_teacher_ack = Some(ack);
                Err("P-C V3 no-scope witness differs")
            }
            Ok(Err(error)) => {
                root.teacher_seal_error = Some(error);
                Err("P-C V3 no-scope issuance unresolved")
            }
            Err(payload) => {
                root.teacher_seal_panic = Some(payload);
                Err("P-C V3 no-scope issuance panicked")
            }
        }
    }

    /// A post-claim rejection is a different domain from the no-scope path.
    /// Retain its original closed issuer and all no-child/no-thread proofs in
    /// recovery; unknown or partially issued workers never release quota.
    pub(crate) fn try_release_claimed_unstarted_teacher(
        &self, admission: &RoventCollectionRootAdmissionV3,
        recovery: &Arc<Mutex<PcRootRecoveryV3>>,
    ) -> Result<Option<bool>, &'static str> {
        let mut original = match recovery.try_lock() {
            Ok(owner) => owner,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 claimed recovery poisoned"),
        };
        if !original.matches_admission(admission) {
            return Err("P-C V3 claimed recovery root differs");
        }
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 roots poisoned"),
        };
        let slot = state.roots.iter_mut().find(|slot| slot.admission.same_original_admission(admission))
            .ok_or("P-C V3 claimed original request absent")?;
        if !slot.recovery.as_ref().is_some_and(|owner| Arc::ptr_eq(owner, recovery))
            || slot.installed || slot.worker.is_some() || slot.route.is_some() || slot.observer.is_some() {
            return Err("P-C V3 claimed recovery or unstarted owner differs");
        }
        match original.try_claimed_unstarted() {
            ClaimedUnstartedPollV3::Pending => return Ok(None),
            ClaimedUnstartedPollV3::Ineligible => return Err("P-C V3 claimed scope is not proven unstarted"),
            ClaimedUnstartedPollV3::Unknown => return Err("P-C V3 claimed scope physical state unknown"),
            ClaimedUnstartedPollV3::Confirmed => {},
        }
        if !original.claimed_unstarted_aggregate().is_some_and(|ack|
            claimed_unstarted_matches(ack, admission)) {
            return Err("P-C V3 claimed original aggregate differs");
        }
        if !original.seal_claimed_unstarted() {
            return Err("P-C V3 claimed recovery could not be sealed");
        }
        slot.claimed_unstarted_terminal = true;
        let mut ledger = match self.ledger.try_lock() {
            Ok(ledger) => ledger,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 quota ledger poisoned"),
        };
        let root = ledger.roots.iter_mut().find(|root| root.admission.same_original_admission(admission))
            .ok_or("P-C V3 claimed quota root absent")?;
        if let Some(old) = &root.claimed_unstarted_recovery {
            return if Arc::ptr_eq(old, recovery) && !root.teacher_reserved && !root.worker_reserved {
                Ok(Some(true))
            } else { Err("P-C V3 claimed quota recovery differs") };
        }
        if !root.teacher_reserved || !root.worker_reserved || root.teacher_seal_unknown
            || root.unclaimed_teacher_ack.is_some() || root.retired_teacher_handoff.is_some() {
            return Err("P-C V3 claimed quota ownership differs");
        }
        // The exact aggregate remains inside this same recovery Arc.
        root.claimed_unstarted_recovery = Some(Arc::clone(recovery));
        root.teacher_reserved = false;
        // The same sealed recovery also proves no root worker was ever
        // created and terminal admission forbids recreating it.
        root.worker_reserved = false;
        Ok(Some(true))
    }

    /// Publish the exact root recovery owner before route creation, scope
    /// claim, worker startup, or teacher actor spawn. The caller retains its
    /// Arc on Busy or error; this does not itself reserve teacher quota.
    pub(crate) fn try_publish_recovery(
        &self, admission: &RoventCollectionRootAdmissionV3,
        recovery: &Arc<Mutex<PcRootRecoveryV3>>,
    ) -> Result<bool, &'static str> {
        {
            let owner = match recovery.try_lock() {
                Ok(owner) => owner,
                Err(TryLockError::WouldBlock) => return Ok(false),
                Err(TryLockError::Poisoned(_)) => return Err("P-C V3 recovery owner poisoned"),
            };
            if !owner.matches_admission(admission) { return Err("P-C V3 recovery root differs"); }
        }
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Ok(false),
            Err(TryLockError::Poisoned(_)) => return Err("P-C V3 roots poisoned"),
        };
        let slot = state.roots.iter_mut().find(|slot| slot.admission.same_original_admission(admission))
            .ok_or("P-C V3 original request absent")?;
        if let Some(old) = &slot.recovery {
            return if Arc::ptr_eq(old, recovery) { Ok(true) } else { Err("P-C V3 recovery owner changed") };
        }
        slot.recovery = Some(Arc::clone(recovery));
        Ok(true)
    }

    /// Called by the supervisor after route installation, native owner
    /// preparation and physical actor startup outside the manager callback.
    pub(crate) fn try_publish_prepared(&self, original: PcPreparedRootV3) -> PcPreparedOutcomeV3 {
        if !original.worker.matches_admission(&original.admission)
            || !original.route.matches_admission(&original.admission)
            || !self.registration.owns_speculative_root_v3(&original.route)
            || !original.owner.capture_registration().same_original_owner(&self.registration)
        { return PcPreparedOutcomeV3::Rejected("P-C V3 prepared owner differs", original); }
        if self.inputs.validate_against(original.admission.native_origin().original_input_lock()).is_err() {
            return PcPreparedOutcomeV3::Rejected("P-C V3 original inputs differ", original);
        }
        {
            let recovery_owner = Arc::clone(&original.recovery);
            let recovery = match recovery_owner.try_lock() {
                Ok(value) => value,
                Err(TryLockError::WouldBlock) => return PcPreparedOutcomeV3::Busy(original),
                Err(TryLockError::Poisoned(_)) => return PcPreparedOutcomeV3::Rejected("P-C V3 recovery poisoned", original),
            };
            if !recovery.matches_prepared(&original.admission, &original.route, &original.stage,
                &original.pump, &original.owner, &original.worker) {
                drop(recovery);
                return PcPreparedOutcomeV3::Rejected("P-C V3 recovery members differ", original);
            }
        }
        {
            let ledger = match self.ledger.try_lock() {
                Ok(ledger) => ledger,
                Err(TryLockError::WouldBlock) => return PcPreparedOutcomeV3::Busy(original),
                Err(TryLockError::Poisoned(_)) => return PcPreparedOutcomeV3::Rejected("P-C V3 quota ledger poisoned", original),
            };
            if !ledger.roots.iter().any(|root| root.worker_reserved && root.teacher_reserved
                && root.admission.same_original_admission(&original.admission)) {
                drop(ledger);
                return PcPreparedOutcomeV3::Rejected("P-C V3 original worker/teacher reservation missing", original);
            }
        }
        let stage_owner = Arc::clone(&original.stage);
        let stage_session = match stage_owner.try_lock() {
            Ok(stage) => stage.session(),
            Err(TryLockError::WouldBlock) => return PcPreparedOutcomeV3::Busy(original),
            Err(TryLockError::Poisoned(_)) => return PcPreparedOutcomeV3::Rejected("P-C V3 teacher stage poisoned", original),
        };
        let Some(stage_session) = stage_session else { return PcPreparedOutcomeV3::Busy(original); };
        {
            let pump_owner = Arc::clone(&original.pump);
            let pump = match pump_owner.try_lock() {
                Ok(pump) => pump,
                Err(TryLockError::WouldBlock) => return PcPreparedOutcomeV3::Busy(original),
                Err(TryLockError::Poisoned(_)) => return PcPreparedOutcomeV3::Rejected("P-C V3 pump poisoned", original),
            };
            if !Arc::ptr_eq(pump.session(), &stage_session) {
                drop(pump); return PcPreparedOutcomeV3::Rejected("P-C V3 staged session differs", original);
            }
            let receiver_owner = Arc::clone(pump.receiver());
            drop(pump);
            let receiver = match receiver_owner.try_lock() {
                Ok(receiver) => receiver,
                Err(TryLockError::WouldBlock) => return PcPreparedOutcomeV3::Busy(original),
                Err(TryLockError::Poisoned(_)) => return PcPreparedOutcomeV3::Rejected("P-C V3 receiver poisoned", original),
            };
            if !receiver.matches_original_route(&original.route) {
                drop(receiver);
                return PcPreparedOutcomeV3::Rejected("P-C V3 original receiver differs", original);
            }
        }
        {
            let session = match stage_session.try_lock() {
                Ok(session) => session,
                Err(TryLockError::WouldBlock) => return PcPreparedOutcomeV3::Busy(original),
                Err(TryLockError::Poisoned(_)) => return PcPreparedOutcomeV3::Rejected("P-C V3 session poisoned", original),
            };
            if !session.original_admission().same_original_admission(&original.admission)
                || !Arc::ptr_eq(session.manifest_lock(), self.inputs.manifest())
                || !session.all_actor_owners_started() {
                drop(session); return PcPreparedOutcomeV3::Rejected("P-C V3 input or root session differs", original);
            }
        }
        match original.owner.try_attach_speculative_root(&original.admission, &original.route) {
            Ok(Some(())) => {}
            Ok(None) => return PcPreparedOutcomeV3::Busy(original),
            Err(_) => return PcPreparedOutcomeV3::Rejected("P-C V3 route pre-attach failed", original),
        }
        match original.owner.try_prepare_original_admission(&original.admission) {
            Ok(Some(())) => {}
            Ok(None) => return PcPreparedOutcomeV3::Busy(original),
            Err(_) => return PcPreparedOutcomeV3::Rejected("P-C V3 observer preparation failed", original),
        }
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return PcPreparedOutcomeV3::Busy(original),
            Err(TryLockError::Poisoned(_)) => return PcPreparedOutcomeV3::Rejected("P-C V3 roots poisoned", original),
        };
        let Some(slot) = state.roots.iter_mut().find(|slot| slot.admission.same_original_admission(&original.admission))
        else { return PcPreparedOutcomeV3::Rejected("P-C V3 original request absent", original); };
        if !slot.recovery.as_ref().is_some_and(|owner| Arc::ptr_eq(owner, &original.recovery)) {
            return PcPreparedOutcomeV3::Rejected("P-C V3 original recovery owner missing", original);
        }
        if slot.owner.is_some() || slot.worker.is_some() || slot.installed {
            return PcPreparedOutcomeV3::Rejected("P-C V3 root already prepared", original);
        }
        let PcPreparedRootV3 { owner, route, stage, pump, worker, .. } = original;
        slot.observer = Some(owner.observer::<PcPortQuestion, VirtualReviewTicketOutput>());
        slot.owner = Some(owner); slot.route = Some(route); slot.stage = Some(stage);
        slot.pump = Some(pump); slot.worker = Some(worker);
        PcPreparedOutcomeV3::Accepted
    }

    /// The supervisor, never the manager callback, joins the exact worker.
    pub(crate) fn try_join_root(&self, admission: &RoventCollectionRootAdmissionV3) -> Result<bool, &'static str> {
        let mut state = self.state.try_lock().map_err(|_| "P-C V3 roots unavailable")?;
        let slot = state.roots.iter_mut().find(|slot| slot.admission.same_original_admission(admission))
            .ok_or("P-C V3 root absent")?;
        if slot.returned.is_some() || slot.joined.is_some() { return Ok(true); }
        if slot.custody_taken { return Ok(false); }
        if slot.join_failure.is_some() || slot.join_panic.is_some() { return Err("P-C V3 worker join unresolved"); }
        let Some(handoff) = &slot.handoff else { return Ok(false); };
        let worker = slot.worker.as_mut().ok_or("P-C V3 worker absent")?;
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker.try_join(handoff))) {
            Ok(Ok(Some(custody))) => { slot.joined = Some(custody); Ok(true) }
            Ok(Ok(None)) => Ok(false),
            Ok(Err(error)) => { slot.join_failure = Some(error); Err("P-C V3 original join failed") }
            Err(payload) => { slot.join_panic = Some(payload); Err("P-C V3 original join panicked") }
        }
    }

    /// Separately account manager-returned joined custody and the original
    /// same-handoff teacher aggregate. Never releases the route or labels.
    pub(crate) fn try_account_root(&self, admission: &RoventCollectionRootAdmissionV3)
        -> Result<PcRootAccountingV3, &'static str> {
        let state = self.state.try_lock().map_err(|_| "P-C V3 roots unavailable")?;
        let slot = state.roots.iter().find(|slot| slot.admission.same_original_admission(admission))
            .ok_or("P-C V3 root absent")?;
        let Some(custody) = &slot.returned else { return Ok(PcRootAccountingV3::Pending); };
        let handoff = match custody.try_lock() {
            Ok(custody) => Arc::clone(custody.handoff()),
            Err(TryLockError::WouldBlock) => return Ok(PcRootAccountingV3::Pending),
            Err(_) => return Err("P-C V3 original custody poisoned"),
        };
        let mut ledger = match self.ledger.try_lock() {
            Ok(ledger) => ledger,
            Err(TryLockError::WouldBlock) => return Ok(PcRootAccountingV3::Pending),
            Err(_) => return Err("P-C V3 ledger poisoned"),
        };
        let Some(index) = ledger.roots.iter().position(|root| root.admission.same_original_admission(admission))
            else { return Err("P-C V3 quota root absent"); };
        if ledger.roots[index].joined.is_none() {
            match ledger.retain_joined_worker(admission, Arc::clone(custody)) {
                Ok(()) => {},
                Err(PcQuotaEvidenceErrorV3::Busy) => return Ok(PcRootAccountingV3::Pending),
                Err(_) => return Err("P-C V3 original joined custody differs"),
            }
        }
        if ledger.roots[index].teacher_reserved {
            match ledger.retain_teacher_aggregate(admission, handoff) {
                Ok(()) => {},
                Err(PcQuotaEvidenceErrorV3::Busy | PcQuotaEvidenceErrorV3::MissingTeacherAggregate) =>
                    return Ok(PcRootAccountingV3::TeacherPending),
                Err(_) => return Err("P-C V3 teacher aggregate differs"),
            }
        }
        if ledger.roots[index].retired_teacher_handoff.is_none() {
            return Ok(PcRootAccountingV3::TeacherPending);
        }
        Ok(PcRootAccountingV3::PhysicallyRetired)
    }

    /// Unlink only after independent joined/native/teacher/receiver evidence.
    /// Failed semantic settlement remains in the original handoff custody.
    pub(crate) fn try_retire_route(&self, admission: &RoventCollectionRootAdmissionV3)
        -> Result<bool, ArenaError> {
        let mut state = match self.state.try_lock() {
            Ok(value) => value,
            Err(TryLockError::WouldBlock) => return Ok(false),
            Err(_) => return Err(invalid("P-C V3 root custody poisoned")),
        };
        let slot = state.roots.iter_mut().find(|slot| slot.admission.same_original_admission(admission))
            .ok_or_else(|| invalid("P-C V3 route root absent"))?;
        if slot.route_retirement.is_some() { return Ok(true); }
        let Some(route) = slot.route.as_ref() else { return Ok(false); };
        let Some(pump) = slot.pump.as_ref() else { return Ok(false); };
        let Some(recovery) = slot.recovery.as_ref() else { return Ok(false); };
        let Some(proof) = RoventTeacherRouteRetirementV3::try_prepare(
            Arc::clone(&self.ledger), route.clone(), Arc::clone(pump), Arc::clone(recovery),
        )? else { return Ok(false); };
        match self.registration.try_unlink_finished_speculative_root_v3(route, &proof)? {
            Some(true) => { slot.route_retirement = Some(proof); Ok(true) }
            Some(false) | None => Ok(false),
        }
    }

    /// Keep consumer semantic rejection and unknown distinct from physical
    /// teacher/native evidence and from this route's separate unlink status.
    pub(crate) fn try_consumer_retirement(
        &self, admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<Option<PcConsumerRetirementV3>, ArenaError> {
        let state = match self.state.try_lock() {
            Ok(value) => value,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err(invalid("P-C V3 root custody poisoned")),
        };
        let slot = state.roots.iter().find(|slot| slot.admission.same_original_admission(admission))
            .ok_or_else(|| invalid("P-C V3 consumer root absent"))?;
        let Some(recovery) = &slot.recovery else { return Ok(None); };
        let recovery = match recovery.try_lock() {
            Ok(value) => value,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err(invalid("P-C V3 consumer recovery poisoned")),
        };
        if !recovery.matches_admission(admission) { return Err(invalid("P-C V3 consumer root differs")); }
        Ok(Some(recovery.consumer_route_retirement()))
    }

    /// Phase one of normal driver shutdown. The same supervisor stays alive
    /// while it drains root requests and retains each physical worker result.
    /// A semantic failure or undurable output is not physical unknown here.
    pub(crate) fn try_authorize_supervisor_stop(
        self: &Arc<Self>,
        fence: &RoventPcCollectionNativeFenceV3,
        supervisor: &Arc<Mutex<PcWorkerSupervisorV3>>,
    ) -> Result<Option<PcSupervisorStopAuthorizationV3>, &'static str> {
        let Some(native_owner) = self.native_owner.get().and_then(Weak::upgrade) else {
            return Err("P-C V3 original native owner absent");
        };
        let hooks: Arc<dyn RoventPcCollectionHooksV3> = self.clone();
        if !fence.matches_original_owner(&native_owner) || !fence.matches_original_hooks(&hooks) {
            return Err("P-C V3 native fence belongs to another owner");
        }
        if self.game_admission_seal.get().is_none() {
            let Some(original_fence) = native_owner.native_session_fence() else { return Ok(None); };
            let seal = self.try_game_admission_seal(original_fence)
                .map_err(|(_, error)| error)?;
            // Another poll may have installed the equivalent same-owner seal.
            let _ = self.game_admission_seal.set(seal);
        }
        let Some(seal) = self.game_admission_seal.get() else { return Ok(None); };
        match self.game.try_seal_pc(seal) {
            super::game_authority_v3::GameStepV3::Ready | super::game_authority_v3::GameStepV3::Existing => {},
            super::game_authority_v3::GameStepV3::Busy => return Ok(None),
            _ => return Err("P-C V3 game admission seal rejected"),
        }
        if !self.supervisor.get().and_then(Weak::upgrade)
            .is_some_and(|original| Arc::ptr_eq(&original, supervisor)) {
            return Err("P-C V3 original supervisor differs");
        }
        // Same order as the running supervisor: supervisor -> bridge -> ledger.
        let original = match supervisor.try_lock() {
            Ok(value) => value,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 original supervisor poisoned"),
        };
        if !original.matches_original_bridge(self) { return Err("P-C V3 supervisor bridge differs"); }
        if !original.pending_root_absent() { return Ok(None); }
        let mut state = match self.state.try_lock() {
            Ok(value) => value,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 root custody poisoned"),
        };
        let mut ledger = match self.ledger.try_lock() {
            Ok(value) => value,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err("P-C V3 quota ledger poisoned"),
        };
        for slot in &mut state.roots {
            match slot.try_seal_root_not_accepting(&self.game, &mut ledger) {
                Ok(None) => return Ok(None),
                Ok(Some(_)) => {},
                Err(error) => return Err(error.detail),
            }
        }
        if state.roots.iter().filter(|slot| slot.guard_aborted.is_none()
            && !slot.root_not_accepting.as_ref().is_some_and(|proof| proof.quota == RootNotAcceptingQuotaV3::Unreserved)).count() != ledger.roots.len() { return Ok(None); }
        for slot in &state.roots {
            if slot.root_not_accepting.is_some() {
                if !slot.root_not_accepting_valid(&self.game, &ledger) { return Ok(None); }
                continue;
            }
            if slot.guard_aborted.is_some() {
                if !slot.guard_aborted_no_dispatch_valid(&self.game, &ledger)? { return Ok(None); }
                continue;
            }
            if slot.dispatch_rejected.is_some() {
                if !slot.dispatch_rejected_no_worker_valid(&self.game, &ledger)? { return Ok(None); }
                continue;
            }
            let Some(recovery) = &slot.recovery else { return Ok(None); };
            if !slot.requested || !original.owns_recovery(recovery) { return Ok(None); }
            let Some(root) = ledger.roots.iter().find(|root|
                root.admission.same_original_admission(&slot.admission)) else { return Ok(None); };
            if root.worker_reserved || root.teacher_reserved || root.teacher_seal_unknown {
                return Ok(None);
            }
            if slot.unstarted_terminal {
                let (Some(owner), Some(ack)) = (&root.unstarted_recovery, &root.unclaimed_teacher_ack)
                    else { return Ok(None); };
                if !Arc::ptr_eq(owner, recovery) || !ack.matches_admission(&slot.admission)
                    || !ack.is_sealed_no_scope() { return Err("P-C V3 no-worker evidence differs"); }
                let recovery = match recovery.try_lock() {
                    Ok(value) => value,
                    Err(TryLockError::WouldBlock) => return Ok(None),
                    Err(_) => return Err("P-C V3 original recovery poisoned"),
                };
                if !recovery.is_sealed_unstarted() { return Ok(None); }
            } else if slot.claimed_unstarted_terminal {
                let Some(owner) = &root.claimed_unstarted_recovery else { return Ok(None); };
                if !Arc::ptr_eq(owner, recovery) { return Err("P-C V3 claimed recovery differs"); }
                let recovery = match recovery.try_lock() {
                    Ok(value) => value,
                    Err(TryLockError::WouldBlock) => return Ok(None),
                    Err(_) => return Err("P-C V3 original claimed recovery poisoned"),
                };
                if !recovery.is_sealed_claimed_unstarted()
                    || !recovery.claimed_unstarted_aggregate().is_some_and(|ack|
                        claimed_unstarted_matches(ack, &slot.admission)) { return Ok(None); }
            } else {
                let (Some(returned), Some(joined), Some(handoff)) =
                    (&slot.returned, &root.joined, &root.retired_teacher_handoff) else { return Ok(None); };
                if !Arc::ptr_eq(returned, joined) { return Err("P-C V3 joined owner differs"); }
                let custody = match joined.try_lock() {
                    Ok(value) => value,
                    Err(TryLockError::WouldBlock) => return Ok(None),
                    Err(_) => return Err("P-C V3 original joined owner poisoned"),
                };
                if !Arc::ptr_eq(custody.handoff(), handoff) { return Err("P-C V3 physical handoff differs"); }
                let handoff = match handoff.try_lock() {
                    Ok(value) => value,
                    Err(TryLockError::WouldBlock) => return Ok(None),
                    Err(_) => return Err("P-C V3 original handoff poisoned"),
                };
                if !handoff.teacher_aggregate_ack().is_some_and(|ack|
                    ack.matches_admission(&slot.admission)) { return Ok(None); }
            }
        }
        drop(ledger);
        drop(state);
        drop(original);
        Ok(Some(PcSupervisorStopAuthorizationV3 {
            _bridge: Arc::clone(self), supervisor: Arc::clone(supervisor), native_owner,
        }))
    }
}

/// Retains all original root proof owners after the native owner seals the
/// same installed hook. This is not a model/teacher/collection success claim.
struct PcV3VerifiedClosureRoot {
    admission: RoventCollectionRootAdmissionV3,
    observer: Option<RoventCollectionOwnerLockV3>,
    requires_observer: bool,
}

pub(super) struct PcV3LaunchClosure {
    _bridge: Arc<RoventPcCollectionBridgeV3>,
    _native_owner: Arc<RoventPcCollectionLaunchOwnerV3>,
    // Captured while the same bridge state lock still protects the verified
    // roots. This alone is not game closure or publication authority.
    verified_roots: Option<Vec<PcV3VerifiedClosureRoot>>,
}

/// Game-only evidence retained outside the bridge. Physical settlement,
/// pipeline finish and publication have their own original owners.
pub(super) struct PcGameClosureProofV3 {
    closure: Arc<PcV3LaunchClosure>,
    game: Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>,
    _custody: Arc<Mutex<SupervisorStartCustodyV3>>,
    _supervisor: Arc<Mutex<PcWorkerSupervisorV3>>,
    pipeline: super::policy_row_pipeline_v3::PolicyRowPipelineOwnerV3,
    observers: Vec<(RoventCollectionOwnerLockV3, RoventCollectionRootAdmissionV3, super::observer_lock_v3::ObserverRootDrainedV3)>,
}

impl PcGameClosureProofV3 {
    pub(super) fn matches_game(&self, game: &Arc<super::game_authority_v3::GameAuthorityV3<PcWorkerSupervisorV3>>) -> bool {
        Arc::ptr_eq(&self.game, game)
    }

    pub(super) fn matches_roots(&self, roots: &[RoventCollectionRootAdmissionV3]) -> bool {
        let Some(original) = self.closure.verified_roots.as_ref() else { return false; };
        original.len() == roots.len() && original.iter().all(|entry|
            roots.iter().any(|root| root.same_original_admission(&entry.admission)))
    }

    /// The stored observer tokens are observations, not a promise that the
    /// owners stayed drained. Reobserve the same roots before game closure.
    pub(super) fn try_revalidate_observers(&self) -> Result<Option<bool>, ArenaError> {
        let Some(roots) = self.closure.verified_roots.as_ref() else { return Ok(None); };
        {
            let state = match self.closure._bridge.state.try_lock() {
                Ok(state) => state,
                Err(TryLockError::WouldBlock) => return Ok(None),
                Err(_) => return Err(invalid("P-C V3 original root custody poisoned")),
            };
            if state.roots.len() != roots.len() { return Ok(Some(false)); }
            for root in roots {
                let Some(slot) = state.roots.iter().find(|slot|
                    slot.admission.same_original_admission(&root.admission)) else { return Ok(Some(false)); };
                let (Some(original_owner), Some(owner), Some(route), Some(retirement)) =
                    (&root.observer, &slot.owner, &slot.route, &slot.route_retirement) else { return Ok(None); };
                if !root.requires_observer || !original_owner.same_original_owner(owner) {
                    return Ok(Some(false));
                }
                match retirement.matches_original_route(route)? {
                    Some(true) => {},
                    None => return Ok(None),
                    Some(false) => return Ok(Some(false)),
                }
            }
        }
        for (owner, admission, original) in &self.observers {
            if !original.matches_original(owner, admission, &self.pipeline) {
                return Err(invalid("P-C V3 original observer drain differs"));
            }
            match owner.try_observe_root_drained(admission, &self.pipeline)? {
                super::observer_lock_v3::ObserverRootDrainAttemptV3::Ready(current)
                    if current.matches_original(owner, admission, &self.pipeline) => {},
                super::observer_lock_v3::ObserverRootDrainAttemptV3::Busy
                | super::observer_lock_v3::ObserverRootDrainAttemptV3::Pending => return Ok(None),
                _ => return Ok(Some(false)),
            }
        }
        Ok(Some(true))
    }
}

impl PcV3LaunchClosure {
    /// Build a separate game candidate only after the exact driver's actual
    /// join. Busy and missing optional game evidence do not undo physical
    /// settlement. The caller retains the settled outcome and original proof.
    pub(super) fn try_game_proof(
        self: &Arc<Self>,
        joined: &PcSupervisorJoinedProofV3,
        custody: &Arc<Mutex<SupervisorStartCustodyV3>>,
    ) -> Result<Option<PcGameClosureProofV3>, ArenaError> {
        let bridge = &self._bridge;
        let Some(seal) = bridge.game_joined_seal.get() else { return Ok(None); };
        let Some(game) = seal.game.upgrade() else { return Ok(None); };
        let Some(original_custody) = seal.custody.upgrade() else { return Ok(None); };
        let Some(supervisor) = seal.supervisor.upgrade() else { return Ok(None); };
        let Some(native) = seal.native.upgrade() else { return Ok(None); };
        if !Arc::ptr_eq(&game, &bridge.game) || !Arc::ptr_eq(&original_custody, custody)
            || !Arc::ptr_eq(&native, &self._native_owner)
            || joined.panicked() || !joined.matches_original(custody, &supervisor, &native) {
            return Err(invalid("P-C V3 joined game owner differs"));
        }
        match joined.try_driver_observation() {
            Ok(super::pc_worker_v3::PcDriverPollV3::Pending) => return Ok(None),
            Ok(super::pc_worker_v3::PcDriverPollV3::Ready(
                super::pc_worker_v3::PcSupervisorDriverObservationV3::Completed,
            )) => {},
            Ok(super::pc_worker_v3::PcDriverPollV3::Ready(failure)) => {
                return Err(invalid(format!("P-C V3 game driver did not complete: {failure:?}")));
            },
            Err(error) => return Err(invalid(format!("P-C V3 driver observation: {error:?}"))),
        }
        let pipeline = match supervisor.try_lock() {
            Ok(supervisor) => supervisor.game_pipeline(),
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err(invalid("P-C V3 joined supervisor poisoned")),
        };
        let Some(pipeline) = pipeline else { return Ok(None); };
        let Some(roots) = self.verified_roots.as_ref() else { return Ok(None); };
        let mut observers = Vec::new();
        if observers.try_reserve_exact(roots.len()).is_err() { return Ok(None); }
        for root in roots {
            // Physical no-worker/no-dispatch evidence does not classify a
            // missing collection root as eligible for game publication.
            // A separate same-root semantic outcome is required first.
            if !root.requires_observer { return Ok(None); }
            let Some(owner) = &root.observer else { return Ok(None); };
            let token = match owner.try_observe_root_drained(&root.admission, &pipeline)? {
                super::observer_lock_v3::ObserverRootDrainAttemptV3::Ready(token) => token,
                super::observer_lock_v3::ObserverRootDrainAttemptV3::Busy
                | super::observer_lock_v3::ObserverRootDrainAttemptV3::Pending => return Ok(None),
                super::observer_lock_v3::ObserverRootDrainAttemptV3::Failed => return Err(invalid("P-C V3 observer drain failed")),
            };
            if !token.matches_original(owner, &root.admission, &pipeline) {
                return Err(invalid("P-C V3 observer drain owner differs"));
            }
            observers.push((owner.clone(), root.admission.clone(), token));
        }
        Ok(Some(PcGameClosureProofV3 {
            closure: Arc::clone(self), game, _custody: original_custody,
            _supervisor: supervisor, pipeline, observers,
        }))
    }
}

/// Non-cloneable authorization for stopping the exact original supervisor
/// after native admission is closed and every original root's physical
/// ownership is retained. It does not attest output or semantic success.
pub(crate) struct PcSupervisorStopAuthorizationV3 {
    _bridge: Arc<RoventPcCollectionBridgeV3>,
    supervisor: Arc<Mutex<PcWorkerSupervisorV3>>,
    native_owner: Arc<RoventPcCollectionLaunchOwnerV3>,
}

impl PcSupervisorStopAuthorizationV3 {
    pub(crate) fn matches_original(
        &self, supervisor: &Arc<Mutex<PcWorkerSupervisorV3>>,
        native_owner: &Arc<RoventPcCollectionLaunchOwnerV3>,
    ) -> bool {
        Arc::ptr_eq(&self.supervisor, supervisor) && Arc::ptr_eq(&self.native_owner, native_owner)
    }
}

fn pc_state<'a>(state: &'a Mutex<PcBridgeStateV3>)
    -> Result<std::sync::MutexGuard<'a, PcBridgeStateV3>, AdaptiveManagerError> {
    match state.try_lock() {
        Ok(value) => Ok(value),
        Err(TryLockError::WouldBlock) => Err(AdaptiveManagerError::AuxiliaryPending),
        Err(TryLockError::Poisoned(_)) => Err(AdaptiveManagerError::Invalid("P-C V3 root custody poisoned")),
    }
}

impl RoventPcCollectionHooksV3 for RoventPcCollectionBridgeV3 {
    fn register_startup_control(&self, control: &RoventStartupControl) -> Result<(), AdaptiveManagerError> {
        if let Some(old) = self.startup.get() {
            return if old.until == control.until && old.cleanup_until == control.cleanup_until
                && Arc::ptr_eq(&old.cancellation, &control.cancellation) { Ok(()) }
                else { Err(AdaptiveManagerError::Invalid("P-C V3 startup owner changed")) };
        }
        self.startup.set(control.clone()).map_err(|_| AdaptiveManagerError::AuxiliaryPending)
    }

    fn admit(&self, port: &mut RoventPcCollectionAdmissionPortV3<'_>, admission: RoventCollectionRootAdmissionV3)
        -> Result<RoventPcCollectionAdmissionOutcomeV3, RoventPcCollectionAdmissionFailureV3> {
        let failure = |error, admission| RoventPcCollectionAdmissionFailureV3 { error, admission, owners: Box::new(()) };
        if self.native_owner.get().and_then(Weak::upgrade).is_none() {
            return Err(failure(AdaptiveManagerError::Invalid("P-C V3 native owner not bound"), admission));
        }
        let Some(input_identity) = self.input_lock_identity.get() else {
            return Err(failure(AdaptiveManagerError::Invalid("P-C V3 input lock identity not bound"), admission));
        };
        if !input_identity.matches_origin(admission.native_origin()) {
            return Err(failure(AdaptiveManagerError::Invalid("P-C V3 native origin input lock differs"), admission));
        }
        let mut state = match pc_state(&self.state) {
            Ok(value) => value,
            Err(AdaptiveManagerError::AuxiliaryPending) => return Ok(RoventPcCollectionAdmissionOutcomeV3::Pending { admission }),
            Err(error) => return Err(failure(error, admission)),
        };
        let index = match state.roots.iter().position(|slot| slot.admission.same_original_admission(&admission)) {
            Some(index) => index,
            None => {
                if state.roots.len() >= usize::from(self.inputs.plan().maximum_roots)
                    || state.roots.iter().any(|slot| slot.admission.root_admission().authority() == admission.root_admission().authority()) {
                    return Err(failure(AdaptiveManagerError::Capacity("P-C V3 root capacity or duplicate authority"), admission));
                }
                let reservation = match self.game.try_reserve_root(&admission) {
                    super::game_authority_v3::RootReservationV3::Reserved(reservation) => reservation,
                    super::game_authority_v3::RootReservationV3::Rejected(super::game_authority_v3::GameStepV3::Busy) => return Ok(RoventPcCollectionAdmissionOutcomeV3::Pending { admission }),
                    super::game_authority_v3::RootReservationV3::Rejected(super::game_authority_v3::GameStepV3::Closed | super::game_authority_v3::GameStepV3::Aborted) => return Err(failure(AdaptiveManagerError::AdmissionClosed, admission)),
                    super::game_authority_v3::RootReservationV3::Rejected(super::game_authority_v3::GameStepV3::Capacity) => return Err(failure(AdaptiveManagerError::Capacity("P-C V3 game root capacity"), admission)),
                    _ => return Err(failure(AdaptiveManagerError::Invalid("P-C V3 game reservation differs or is unresolved"), admission)),
                };
                state.roots.push(PcRootSlotV3 { admission: admission.clone(), game_reservation: Some(reservation), game_abort: None, guard_aborted: None, dispatch_claim: None, dispatch_receipt: None, dispatch_rejected: None, root_not_accepting: None, root_authority_failure: None, game_registered: false, quota_registered: false,
                    dispatch: PcRootDispatchV3::BeforeDispatch, requested: false, installed: false, unstarted_terminal: false, claimed_unstarted_terminal: false, recovery: None,
                    owner: None, route: None, route_retirement: None, stage: None, pump: None, observer: None, worker: None,
                    handoff: None, joined: None, custody_taken: false, returned: None, join_failure: None, join_panic: None });
                state.roots.len() - 1
            }
        };
        let slot = &mut state.roots[index];
        if slot.root_not_accepting.is_some() {
            return Err(failure(AdaptiveManagerError::AdmissionClosed, admission));
        }
        if slot.dispatch_rejected.is_some() {
            return Err(failure(AdaptiveManagerError::AdmissionClosed, admission));
        }
        if let Some(reservation) = slot.game_reservation.as_mut() {
            match reservation.commit_registered() {
                super::game_authority_v3::GameStepV3::Ready => {
                    slot.game_registered = true;
                    slot.game_reservation.take();
                },
                super::game_authority_v3::GameStepV3::Aborted => {
                    slot.game_registered = true;
                    if !reservation.transfer_abort_receipt(&mut slot.game_abort) {
                        return Err(failure(AdaptiveManagerError::Invalid("P-C V3 original abort receipt unavailable"), admission));
                    }
                    slot.game_reservation.take();
                },
                super::game_authority_v3::GameStepV3::Busy => return Ok(RoventPcCollectionAdmissionOutcomeV3::Pending { admission }),
                _ => return Err(failure(AdaptiveManagerError::Invalid("P-C V3 game registration unresolved"), admission)),
            }
        }
        if slot.game_abort.is_some() {
            let ledger = match self.ledger.try_lock() {
                Ok(value) => value,
                Err(TryLockError::WouldBlock) => return Ok(RoventPcCollectionAdmissionOutcomeV3::Pending { admission }),
                Err(_) => return Err(failure(AdaptiveManagerError::Invalid("P-C V3 ledger poisoned"), admission)),
            };
            match slot.seal_guard_aborted_no_dispatch(&self.game, &ledger) {
                Ok(true) => {},
                Ok(false) => return Ok(RoventPcCollectionAdmissionOutcomeV3::Pending { admission }),
                Err(_) => return Err(failure(AdaptiveManagerError::Invalid("P-C V3 game abort evidence differs"), admission)),
            }
        }
        if slot.guard_aborted.is_some() {
            return Err(failure(AdaptiveManagerError::AdmissionClosed, admission));
        }
        if !slot.quota_registered {
            let mut ledger = match self.ledger.try_lock() {
                Ok(value) => value,
                Err(TryLockError::WouldBlock) => return Ok(RoventPcCollectionAdmissionOutcomeV3::Pending { admission }),
                Err(_) => return Err(failure(AdaptiveManagerError::Invalid("P-C V3 ledger poisoned"), admission)),
            };
            match ledger.reserve_worker(&admission) {
                PcQuotaOutcomeV3::Reserved | PcQuotaOutcomeV3::AlreadyReserved => slot.quota_registered = true,
                PcQuotaOutcomeV3::Busy => return Ok(RoventPcCollectionAdmissionOutcomeV3::Pending { admission }),
                PcQuotaOutcomeV3::Exhausted => {
                    // Do not scan on every tick. Only quota contention can
                    // benefit from retiring other root-local no-dispatch work.
                    drop(ledger);
                    drop(state);
                    if let Err(error) = self.try_recover_nonaccepting_roots() {
                        return Err(failure(AdaptiveManagerError::Invalid(error), admission));
                    }
                    return Ok(RoventPcCollectionAdmissionOutcomeV3::Pending { admission });
                },
                _ => return Err(failure(AdaptiveManagerError::Invalid("P-C V3 duplicate root reservation"), admission)),
            }
        }
        if slot.unstarted_terminal || slot.claimed_unstarted_terminal {
            return Err(failure(AdaptiveManagerError::Invalid("P-C V3 original root failed before worker startup"), admission));
        }
        if !slot.requested {
            // Until the same game-owned dispatch claim is wired, do not send
            // work across an abort/admission-seal race.
            return Ok(RoventPcCollectionAdmissionOutcomeV3::Pending { admission });
        }
        if slot.installed { return Ok(RoventPcCollectionAdmissionOutcomeV3::Installed); }
        let Some(observer) = slot.observer.take() else {
            // Once the original request is sent, the supervisor may already
            // be creating a worker before it publishes the prepared bundle.
            return Ok(RoventPcCollectionAdmissionOutcomeV3::PendingUncertain { admission });
        };
        match port.try_install(admission, observer) {
            Ok(RoleOnlyCollectionV3InstallAttempt::Installed) => {
                slot.installed = true;
                Ok(RoventPcCollectionAdmissionOutcomeV3::Installed)
            }
            Ok(RoleOnlyCollectionV3InstallAttempt::Busy { admission, observer }) => {
                if !admission.same_original_admission(&slot.admission) {
                    return Err(RoventPcCollectionAdmissionFailureV3 {
                        error: AdaptiveManagerError::Invalid("P-C V3 busy admission owner differs"),
                        admission: admission.clone(),
                        owners: Box::new((admission, observer)),
                    });
                }
                // The supervisor may already own physical work for this
                // root. Keep this exact observer and retry the same admission.
                slot.observer = Some(observer);
                Ok(RoventPcCollectionAdmissionOutcomeV3::PendingUncertain { admission })
            }
            Err(original) => Err(RoventPcCollectionAdmissionFailureV3 {
                error: original.error.clone(), admission: original.admission.clone(), owners: Box::new(original),
            }),
        }
    }

    fn pump_root(&self, admission: &RoventCollectionRootAdmissionV3) -> Result<(), AdaptiveManagerError> {
        let state = pc_state(&self.state)?;
        if state.roots.iter().any(|slot| slot.installed && slot.admission.same_original_admission(admission)) { Ok(()) }
        else { Err(AdaptiveManagerError::Invalid("P-C V3 root missing")) }
    }

    fn revoke_root(&self, admission: &RoventCollectionRootAdmissionV3) -> Result<(), AdaptiveManagerError> {
        let state = pc_state(&self.state)?;
        let slot = state.roots.iter().find(|slot| slot.admission.same_original_admission(admission))
            .ok_or(AdaptiveManagerError::Invalid("P-C V3 revoke root missing"))?;
        slot.route.as_ref().ok_or(AdaptiveManagerError::AuxiliaryPending)?.cancel_all();
        Ok(())
    }

    fn finish_deferred(&self, capability: &RoventNoCaptureRootCapabilityV3, fence: &RoventRoleOnlyNoCaptureFenceV3)
        -> Result<AdaptiveCollectionBoundaryV2, AdaptiveManagerError> {
        match self.registration.try_finish_whole_root_no_capture_v3_role_only(capability, fence) {
            Ok(true) => Ok(AdaptiveCollectionBoundaryV2::Ready),
            Ok(false) => Ok(AdaptiveCollectionBoundaryV2::Busy),
            Err(error) => Err(AdaptiveManagerError::Semantic(error.to_string())),
        }
    }

    fn retain_deferred_cleanup(&self, cleanup: Arc<Mutex<RoventPcDeferredCleanupV3>>) -> Result<RoventPcCustodyReturnV3, AdaptiveManagerError> {
        let mut state = match pc_state(&self.state) {
            Ok(value) => value, Err(AdaptiveManagerError::AuxiliaryPending) => return Ok(RoventPcCustodyReturnV3::Busy), Err(error) => return Err(error),
        };
        if state.deferred.iter().any(|old| Arc::ptr_eq(old, &cleanup)) { return Ok(RoventPcCustodyReturnV3::Accepted); }
        if state.deferred.len() >= usize::from(self.inputs.plan().maximum_roots) { return Ok(RoventPcCustodyReturnV3::Busy); }
        state.deferred.push(cleanup); Ok(RoventPcCustodyReturnV3::Accepted)
    }

    fn handoff_sender_root(&self, admission: &RoventCollectionRootAdmissionV3) -> Result<mpsc::SyncSender<PcHandoffV3>, AdaptiveManagerError> {
        let state = pc_state(&self.state)?;
        let slot = state.roots.iter().find(|slot| slot.admission.same_original_admission(admission))
            .ok_or(AdaptiveManagerError::Invalid("P-C V3 handoff root absent"))?;
        Ok(slot.worker.as_ref().ok_or(AdaptiveManagerError::AuxiliaryPending)?.sender())
    }

    fn try_take_root_custody(&self, handoff: &PcHandoffV3) -> Result<Option<RoventPcCollectionRootCustodyV3>, AdaptiveManagerError> {
        let admission = match handoff.try_lock() {
            Ok(mut value) => value.parts_mut().0.clone(), Err(TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err(AdaptiveManagerError::Invalid("P-C V3 handoff poisoned")),
        };
        let mut state = match pc_state(&self.state) { Ok(value) => value, Err(AdaptiveManagerError::AuxiliaryPending) => return Ok(None), Err(error) => return Err(error) };
        let slot = state.roots.iter_mut().find(|slot| slot.admission.same_original_admission(&admission))
            .ok_or(AdaptiveManagerError::Invalid("P-C V3 handoff root absent"))?;
        if slot.handoff.as_ref().is_some_and(|old| !Arc::ptr_eq(old, handoff)) {
            return Err(AdaptiveManagerError::Invalid("P-C V3 handoff changed"));
        }
        if slot.handoff.is_none() { slot.handoff = Some(Arc::clone(handoff)); }
        if slot.join_failure.is_some() || slot.join_panic.is_some() { return Err(AdaptiveManagerError::Invalid("P-C V3 worker join unresolved")); }
        let custody = slot.joined.take();
        if custody.is_some() { slot.custody_taken = true; }
        Ok(custody)
    }

    fn return_root_custody(&self, custody: Arc<Mutex<RoventPcCollectionRootCustodyV3>>) -> Result<RoventPcCustodyReturnV3, AdaptiveManagerError> {
        let handoff = match custody.try_lock() {
            Ok(value) => Arc::clone(value.handoff()), Err(TryLockError::WouldBlock) => return Ok(RoventPcCustodyReturnV3::Busy),
            Err(_) => return Err(AdaptiveManagerError::Invalid("P-C V3 returned custody poisoned")),
        };
        let mut state = match pc_state(&self.state) { Ok(value) => value, Err(AdaptiveManagerError::AuxiliaryPending) => return Ok(RoventPcCustodyReturnV3::Busy), Err(error) => return Err(error) };
        let slot = state.roots.iter_mut().find(|slot| slot.handoff.as_ref().is_some_and(|old| Arc::ptr_eq(old, &handoff)))
            .ok_or(AdaptiveManagerError::Invalid("P-C V3 returned owner differs"))?;
        match &slot.returned {
            Some(old) if Arc::ptr_eq(old, &custody) => Ok(RoventPcCustodyReturnV3::Accepted),
            Some(_) => Err(AdaptiveManagerError::Invalid("P-C V3 returned custody changed")),
            None => { slot.returned = Some(custody); Ok(RoventPcCustodyReturnV3::Accepted) }
        }
    }

    fn try_finish_launch(&self, fence: &RoventPcCollectionNativeFenceV3)
        -> Result<Option<Box<dyn std::any::Any + Send>>, AdaptiveManagerError> {
        let Some(bridge) = self.self_weak.upgrade() else {
            return Err(AdaptiveManagerError::Invalid("P-C V3 original bridge absent"));
        };
        let hooks: Arc<dyn RoventPcCollectionHooksV3> = bridge.clone();
        let Some(owner) = self.native_owner.get().and_then(Weak::upgrade) else {
            return Err(AdaptiveManagerError::Invalid("P-C V3 native owner absent"));
        };
        if !fence.matches_original_owner(&owner) || !fence.matches_original_hooks(&hooks) {
            return Err(AdaptiveManagerError::Invalid("P-C V3 native fence belongs to another launch"));
        }
        let state = match pc_state(&self.state) {
            Ok(state) => state,
            Err(AdaptiveManagerError::AuxiliaryPending) => return Ok(None),
            Err(error) => return Err(error),
        };
        if !state.deferred.is_empty() { return Ok(None); }
        for slot in &state.roots {
            if slot.root_not_accepting.is_some() {
                let ledger = match self.ledger.try_lock() {
                    Ok(value) => value,
                    Err(TryLockError::WouldBlock) => return Ok(None),
                    Err(_) => return Err(AdaptiveManagerError::Invalid("P-C V3 ledger poisoned")),
                };
                if !slot.root_not_accepting_valid(&self.game, &ledger) { return Ok(None); }
                continue;
            }
            if slot.dispatch_rejected.is_some() {
                let ledger = match self.ledger.try_lock() {
                    Ok(value) => value,
                    Err(TryLockError::WouldBlock) => return Ok(None),
                    Err(_) => return Err(AdaptiveManagerError::Invalid("P-C V3 ledger poisoned")),
                };
                match slot.dispatch_rejected_no_worker_valid(&self.game, &ledger) {
                    Ok(true) => continue,
                    Ok(false) => return Ok(None),
                    Err(error) => return Err(AdaptiveManagerError::Invalid(error)),
                }
            }
            if slot.guard_aborted.is_some() {
                let ledger = match self.ledger.try_lock() {
                    Ok(value) => value,
                    Err(TryLockError::WouldBlock) => return Ok(None),
                    Err(_) => return Err(AdaptiveManagerError::Invalid("P-C V3 ledger poisoned")),
                };
                match slot.guard_aborted_no_dispatch_valid(&self.game, &ledger) {
                    Ok(true) => continue,
                    Ok(false) => return Ok(None),
                    Err(error) => return Err(AdaptiveManagerError::Invalid(error)),
                }
            }
            if slot.claimed_unstarted_terminal {
                if slot.installed || slot.worker.is_some() || slot.route.is_some()
                    || slot.observer.is_some() || slot.stage.is_some() || slot.pump.is_some() {
                    return Err(AdaptiveManagerError::Invalid("P-C V3 claimed unstarted root acquired owners"));
                }
                let ledger = match self.ledger.try_lock() {
                    Ok(value) => value,
                    Err(TryLockError::WouldBlock) => return Ok(None),
                    Err(_) => return Err(AdaptiveManagerError::Invalid("P-C V3 ledger poisoned")),
                };
                let Some(root) = ledger.roots.iter().find(|root|
                    root.admission.same_original_admission(&slot.admission)) else {
                    return Err(AdaptiveManagerError::Invalid("P-C V3 claimed root missing"));
                };
                let (Some(recovery), Some(original)) =
                    (&slot.recovery, &root.claimed_unstarted_recovery) else { return Ok(None); };
                if root.worker_reserved { return Ok(None); }
                if !Arc::ptr_eq(recovery, original) || root.teacher_reserved
                    || root.teacher_seal_unknown || root.unclaimed_teacher_ack.is_some() {
                    return Err(AdaptiveManagerError::Invalid("P-C V3 claimed unstarted proof differs"));
                }
                let recovery = match recovery.try_lock() {
                    Ok(value) => value,
                    Err(TryLockError::WouldBlock) => return Ok(None),
                    Err(_) => return Err(AdaptiveManagerError::Invalid("P-C V3 claimed recovery poisoned")),
                };
                if !recovery.is_sealed_claimed_unstarted()
                    || !recovery.claimed_unstarted_aggregate().is_some_and(|ack|
                        claimed_unstarted_matches(ack, &slot.admission)) {
                    return Ok(None);
                }
                continue;
            }
            if slot.unstarted_terminal {
                if slot.installed || slot.worker.is_some() || slot.route.is_some()
                    || slot.observer.is_some() || slot.stage.is_some() || slot.pump.is_some() {
                    return Err(AdaptiveManagerError::Invalid("P-C V3 unstarted root acquired owners"));
                }
                let ledger = match self.ledger.try_lock() {
                    Ok(value) => value,
                    Err(TryLockError::WouldBlock) => return Ok(None),
                    Err(_) => return Err(AdaptiveManagerError::Invalid("P-C V3 ledger poisoned")),
                };
                let Some(root) = ledger.roots.iter().find(|root|
                    root.admission.same_original_admission(&slot.admission)) else {
                    return Err(AdaptiveManagerError::Invalid("P-C V3 unstarted root missing"));
                };
                let (Some(recovery), Some(original), Some(ack)) =
                    (&slot.recovery, &root.unstarted_recovery, &root.unclaimed_teacher_ack)
                    else { return Ok(None); };
                if !Arc::ptr_eq(recovery, original) || root.worker_reserved || root.teacher_reserved
                    || root.teacher_seal_unknown || !ack.matches_admission(&slot.admission)
                    || !ack.is_sealed_no_scope() {
                    return Err(AdaptiveManagerError::Invalid("P-C V3 unstarted proof differs"));
                }
                let recovery = match recovery.try_lock() {
                    Ok(value) => value,
                    Err(TryLockError::WouldBlock) => return Ok(None),
                    Err(_) => return Err(AdaptiveManagerError::Invalid("P-C V3 unstarted recovery poisoned")),
                };
                if !recovery.is_sealed_unstarted() { return Ok(None); }
                continue;
            }
            let (Some(route), Some(proof), Some(_returned)) =
                (&slot.route, &slot.route_retirement, &slot.returned) else { return Ok(None); };
            match proof.matches_original_route(route) {
                Ok(Some(true)) => {},
                Ok(Some(false) | None) => return Ok(None),
                Err(error) => return Err(AdaptiveManagerError::Semantic(error.to_string())),
            }
        }
        let mut roots = Vec::new();
        let roots = if roots.try_reserve_exact(state.roots.len()).is_ok() {
            roots.extend(state.roots.iter().map(|slot| PcV3VerifiedClosureRoot {
                admission: slot.admission.clone(),
                observer: slot.owner.clone(),
                requires_observer: slot.route.is_some(),
            }));
            Some(roots)
        } else {
            // Optional game evidence must not block independently proven
            // physical cleanup. The game closure remains unavailable.
            None
        };
        drop(state);
        Ok(Some(Box::new(Arc::new(PcV3LaunchClosure {
            _bridge: bridge,
            _native_owner: owner,
            verified_roots: roots,
        }))))
    }
}
