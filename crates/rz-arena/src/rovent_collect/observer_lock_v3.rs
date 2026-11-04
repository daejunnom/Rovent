//! Single-original-root system-runtime collection observer.
//!
//! A channel and producer belong to this owner for their entire lifetime. In
//! particular, a second root cannot silently inherit an unfinished capture.
//! The caller must keep this owner alive through the native worker fence and
//! preserve it on errors. This is not a teacher or GPU completion receipt.
use super::capture_router_v3::{CapturePollV3, CaptureRouterV3, FailurePollV3};
use super::native_capture::{
    LockV3CancelAttempt, LockV3RegistrationAttempt, RoventFailedCapturedInput,
    RoventUnstartedRouteRollbackV3,
};
use super::policy_row_pipeline_v3::PolicyRowPipelineOwnerV3;
use super::producer_v3::{
    RoventCorpusProducerLockV3, RoventInputCaptureReservationV3, RoventRetiredLockV3Record,
};
use super::speculative_route_v3::{SpeculativeRootV3, TargetCapabilityTakeV3};
use super::*;
use crate::pals_collect_v3::{GenerationActionSupportV3, GenerationCaptureWorkScopeV3};
use rz_experiments::rovent_corpus_lock_v3::record::RoventCorpusManifestLockV3;
use rz_search::pals::coordinator::FrameInput;
use rz_uci::pals_adaptive::{
    AdaptiveCollectionBoundaryV2, AdaptiveCollectionDispatchV2, AdaptiveCollectionError,
    AdaptiveCollectionObserverBoxV3, AdaptiveCollectionObserverV3, AdaptiveCollectionStage,
    RoleOnlySearchConsumptionV2, RoventCollectionAcceptedWitnessV3,
    RoventCollectionConsumedWitnessV3, RoventCollectionDispatchWitnessV3,
    RoventCollectionNativeEventWitnessV3, RoventCollectionRootAdmissionV3,
    RoventRootRankingAcceptanceV3,
};
use std::{
    collections::VecDeque,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

const MAX_INFLIGHT: usize = 16;
const MAX_RETIRED: usize = 256;

pub(crate) enum UnstartedRouteDetachV3 {
    Detached,
    Busy(RoventUnstartedRouteRollbackV3),
    Rejected(ArenaError, RoventUnstartedRouteRollbackV3),
}

pub(super) enum RetiredTransferV3 {
    Transferred,
    Busy,
    NotFound,
    SlotOccupied,
}

pub(super) enum PipelinePreparationV3 {
    Ready,
    Busy,
    NotReady,
    Failed,
    Unknown,
}

pub(super) enum OriginalRouteMatchV3 {
    Match,
    Different,
    NotReady,
    Busy,
}

pub(super) enum PipelineCapabilityV3 {
    NotReady,
    Busy,
    Outcome(TargetCapabilityTakeV3),
}

struct Staged {
    ticket: RoleTicketEnvelope,
    reservation: RoventInputCaptureReservationV3,
}
struct Pending {
    ticket: RoleTicketEnvelope,
    sequence: u64,
    accepted: bool,
}
struct State {
    manifest: Arc<RoventCorpusManifestLockV3>,
    metadata: RoventCollectionMetadata,
    registration: RoventNativeCaptureRegistration,
    router: Arc<CaptureRouterV3>,
    policy_pipeline: Option<PolicyRowPipelineOwnerV3>,
    admission: Option<RoventCollectionRootAdmissionV3>,
    prepared_admission: Option<RoventCollectionRootAdmissionV3>,
    route_admission: Option<RoventCollectionRootAdmissionV3>,
    speculative_root: Option<SpeculativeRootV3>,
    producer: Option<RoventCorpusProducerLockV3>,
    position: Option<Position>,
    history: Vec<BoardMove>,
    staged: BTreeMap<RequestId, Staged>,
    pending: BTreeMap<RequestId, Pending>,
    handled: BTreeMap<RequestId, RoleTicketEnvelope>,
    abandoned: BTreeMap<RequestId, RoleTicketEnvelope>,
    // Both forms preserve the original owner if an admission check fails.
    rejected_captures: Vec<super::producer_v3::RoventCapturedInputV3>,
    failed_captures: Vec<RoventFailedCapturedInput>,
    failed_reservations: Vec<Staged>,
    retired: VecDeque<RoventRetiredLockV3Record>,
    settled: bool,
}

/// Clones share one original owner; constructing a second observer or root
/// does not grant a second admission. The owner must outlive the worker.
#[derive(Clone)]
pub(crate) struct RoventCollectionOwnerLockV3 {
    state: Arc<Mutex<State>>,
    registration: RoventNativeCaptureRegistration,
}
impl RoventCollectionOwnerLockV3 {
    /// Identity only: this grants no execution, publication or completion.
    pub(crate) fn same_original_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    /// Identity observation only. Busy does not imply a different route,
    /// installation, native completion or permission to issue teacher work.
    pub(super) fn try_matches_original_route(
        &self,
        route: &SpeculativeRootV3,
    ) -> Result<OriginalRouteMatchV3, ArenaError> {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OriginalRouteMatchV3::Busy),
            Err(_) => return Err(invalid("lock-v3 owner poisoned")),
        };
        Ok(match state.speculative_root.as_ref() {
            Some(original) if original.same_original_route(route) => OriginalRouteMatchV3::Match,
            Some(_) => OriginalRouteMatchV3::Different,
            None => OriginalRouteMatchV3::NotReady,
        })
    }

    /// Bind this root to the supervisor-retained game pipeline. The same
    /// game owner may be bound by later roots, but a root cannot switch slots.
    pub(super) fn try_bind_policy_pipeline(
        &self,
        pipeline: &PolicyRowPipelineOwnerV3,
    ) -> Result<Option<()>, ArenaError> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err(invalid("lock-v3 owner poisoned")),
        };
        if let Some(original) = &state.policy_pipeline {
            check(
                original.same_owner(pipeline),
                "lock-v3 policy pipeline owner changed",
            )?;
        } else {
            state.policy_pipeline = Some(pipeline.clone());
        }
        Ok(Some(()))
    }

    /// Move an exact retired row directly into the caller-retained shared
    /// pipeline slot. Both locks are nonblocking and no fallible operation
    /// occurs between removing the row and transferring or restoring it.
    /// This is custody only, not publication or completion authority.
    pub(super) fn try_transfer_retired_to_pipeline(
        &self,
        ticket: &RoleTicketEnvelope,
        pipeline: &PolicyRowPipelineOwnerV3,
    ) -> Result<RetiredTransferV3, ArenaError> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(RetiredTransferV3::Busy),
            Err(_) => return Err(invalid("lock-v3 owner poisoned")),
        };
        check(
            state
                .policy_pipeline
                .as_ref()
                .is_some_and(|original| original.same_owner(pipeline)),
            "lock-v3 policy pipeline differs from original supervisor",
        )?;
        let mut slot = match pipeline.try_lock().map_err(invalid)? {
            Some(slot) => slot,
            None => return Ok(RetiredTransferV3::Busy),
        };
        let Some(index) = state
            .retired
            .iter()
            .position(|row| row.ticket().id == ticket.id)
        else {
            return Ok(RetiredTransferV3::NotFound);
        };
        check(
            state.retired[index].ticket() == *ticket,
            "lock-v3 retired ticket differs",
        )?;
        let mut row = state.retired.remove(index);
        if slot.install_row(&mut row) {
            return Ok(RetiredTransferV3::Transferred);
        }
        state.retired.insert(
            index,
            row.expect("rejected transfer preserves original row"),
        );
        Ok(RetiredTransferV3::SlotOccupied)
    }

    /// The same nonblocking lock order as row transfer: original observer,
    /// then pipeline. Run on a worker: preparation can encode and seal data.
    /// Target publication gate and writer append are separate.
    pub(super) fn try_prepare_pipeline(
        &self,
        pipeline: &PolicyRowPipelineOwnerV3,
        ticket: &RoleTicketEnvelope,
    ) -> Result<PipelinePreparationV3, ArenaError> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(PipelinePreparationV3::Busy),
            Err(_) => return Err(invalid("lock-v3 owner poisoned")),
        };
        check(
            state
                .policy_pipeline
                .as_ref()
                .is_some_and(|original| original.same_owner(pipeline)),
            "lock-v3 policy pipeline differs from original supervisor",
        )?;
        let mut slot = match pipeline.try_lock().map_err(invalid)? {
            Some(slot) => slot,
            None => return Ok(PipelinePreparationV3::Busy),
        };
        let admission = state
            .admission
            .as_ref()
            .ok_or_else(|| invalid("lock-v3 root not admitted"))?
            .clone();
        if slot.is_uncertain() {
            Ok(PipelinePreparationV3::Unknown)
        } else if slot.prepare(state.producer()?, &admission, ticket) {
            Ok(PipelinePreparationV3::Ready)
        } else if slot.is_uncertain() {
            Ok(PipelinePreparationV3::Unknown)
        } else if slot.failure().is_some() {
            Ok(PipelinePreparationV3::Failed)
        } else {
            Ok(PipelinePreparationV3::NotReady)
        }
    }

    /// Transfer only the original route's ticket capability into the bound
    /// pipeline. Manager/native execution and writer append remain separate.
    pub(super) fn try_take_pipeline_capability(
        &self,
        pipeline: &PolicyRowPipelineOwnerV3,
        ticket: &RoleTicketEnvelope,
    ) -> Result<PipelineCapabilityV3, ArenaError> {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(PipelineCapabilityV3::Busy),
            Err(_) => return Err(invalid("lock-v3 owner poisoned")),
        };
        check(
            state
                .policy_pipeline
                .as_ref()
                .is_some_and(|original| original.same_owner(pipeline)),
            "lock-v3 policy pipeline differs from original supervisor",
        )?;
        let Some(route) = state.speculative_root.as_ref() else {
            return Ok(PipelineCapabilityV3::NotReady);
        };
        let mut slot = match pipeline.try_lock().map_err(invalid)? {
            Some(slot) => slot,
            None => return Ok(PipelineCapabilityV3::Busy),
        };
        Ok(match slot.try_take_target_capability(route, ticket)? {
            Some(outcome) => PipelineCapabilityV3::Outcome(outcome),
            None => PipelineCapabilityV3::NotReady,
        })
    }
    pub(crate) fn new_shared(
        manifest: Arc<RoventCorpusManifestLockV3>,
        metadata: RoventCollectionMetadata,
        router: Arc<CaptureRouterV3>,
    ) -> Result<Self, ArenaError> {
        check(
            generation_identifier(&metadata.game_id)
                && generation_identifier(&metadata.opening_id)
                && generation_identifier(&metadata.line_genealogy_id)
                && manifest
                    .registration
                    .opening_splits
                    .contains_key(&metadata.opening_id),
            "lock-v3 collection metadata/opening is not registered",
        )?;
        let registration = router.registration();
        Ok(Self {
            registration: registration.clone(),
            state: Arc::new(Mutex::new(State {
                manifest,
                metadata,
                registration,
                router,
                policy_pipeline: None,
                admission: None,
                prepared_admission: None,
                route_admission: None,
                speculative_root: None,
                producer: None,
                position: None,
                history: Vec::new(),
                staged: BTreeMap::new(),
                pending: BTreeMap::new(),
                handled: BTreeMap::new(),
                abandoned: BTreeMap::new(),
                rejected_captures: Vec::new(),
                failed_captures: Vec::new(),
                failed_reservations: Vec::new(),
                retired: VecDeque::new(),
                settled: false,
            })),
        })
    }
    pub(crate) fn capture_registration(&self) -> RoventNativeCaptureRegistration {
        self.registration.clone()
    }
    /// Prepare the original root on a worker before installing the observer.
    /// This can replay history and initialize the producer, so the manager
    /// must not call it as part of its synchronous install callback. A lock
    /// collision returns None without changing the original owner.
    pub(crate) fn try_prepare_original_admission(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<Option<()>, ArenaError> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err(invalid("lock-v3 owner poisoned")),
        };
        check(
            state.admission.is_none() && !state.settled,
            "lock-v3 owner already admitted",
        )?;
        if let Some(original) = &state.prepared_admission {
            check(
                original.same_original_admission(admission),
                "lock-v3 prepared admission changed",
            )?;
            return Ok(Some(()));
        }
        check(
            state
                .route_admission
                .as_ref()
                .is_none_or(|original| original.same_original_admission(admission))
                && state.producer.is_none()
                && state.position.is_none(),
            "lock-v3 preparation differs from original route",
        )?;
        let snapshot = admission.rules().rules().snapshot();
        let history = snapshot
            .uci_replay(16_384)
            .map_err(|e| invalid(e.to_string()))?
            .moves;
        let position = Position::from_snapshot(
            snapshot,
            rz_position::PositionLimits {
                max_history_positions: 16_385,
                ..rz_position::PositionLimits::default()
            },
        )
        .map_err(|e| invalid(e.to_string()))?;
        let producer =
            RoventCorpusProducerLockV3::new_shared(Arc::clone(&state.manifest), admission.clone())?;
        state.history = history;
        state.position = Some(position);
        state.producer = Some(producer);
        state.prepared_admission = Some(admission.clone());
        Ok(Some(()))
    }
    /// Must be installed before this root registers its first ticket.
    /// None denotes a retryable owner-lock collision, not route installation.
    pub(crate) fn try_attach_speculative_root(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
        route: &SpeculativeRootV3,
    ) -> Result<Option<()>, ArenaError> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(_) => return Err(invalid("lock-v3 owner poisoned")),
        };
        check(
            state
                .admission
                .as_ref()
                .is_none_or(|original| original.same_original_admission(admission))
                && state
                    .prepared_admission
                    .as_ref()
                    .is_none_or(|original| original.same_original_admission(admission))
                && state
                    .route_admission
                    .as_ref()
                    .is_none_or(|original| original.same_original_admission(admission))
                && state.registration.owns_speculative_root_v3(route)
                && route.matches_admission(admission)
                && state.staged.is_empty()
                && state.pending.is_empty()
                && state.handled.is_empty()
                && !state.settled,
            "lock-v3 speculative route admission or timing mismatch",
        )?;
        if let Some(existing) = &state.speculative_root {
            check(
                existing.same_original_route(route),
                "lock-v3 speculative route changed",
            )?;
        } else {
            state.speculative_root = Some(route.clone());
            state.route_admission = Some(admission.clone());
        }
        Ok(Some(()))
    }

    /// Detach only before the observer is admitted and only with the
    /// registration's original unstarted rollback witness. The caller must
    /// serialize install/rollback/detach on the same route owner; this witness
    /// does not grant native-worker or teacher completion authority.
    pub(crate) fn try_detach_unstarted_speculative_root(
        &self,
        witness: RoventUnstartedRouteRollbackV3,
    ) -> UnstartedRouteDetachV3 {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => {
                return UnstartedRouteDetachV3::Busy(witness);
            }
            Err(_) => {
                return UnstartedRouteDetachV3::Rejected(
                    invalid("lock-v3 owner poisoned"),
                    witness,
                );
            }
        };
        let valid = state.admission.is_none()
            && !state.settled
            && state.staged.is_empty()
            && state.pending.is_empty()
            && state.handled.is_empty()
            && state.route_admission.is_some()
            && state
                .speculative_root
                .as_ref()
                .is_some_and(|route| witness.matches_original_route(route));
        if !valid {
            return UnstartedRouteDetachV3::Rejected(
                invalid("lock-v3 route rollback does not match the unstarted owner"),
                witness,
            );
        }
        state.speculative_root = None;
        state.route_admission = None;
        UnstartedRouteDetachV3::Detached
    }
    pub(crate) fn observer<Q: 'static, O: ProposerRankingOutput + 'static>(
        &self,
    ) -> AdaptiveCollectionObserverBoxV3<Q, O> {
        Box::new(Observer {
            owner: self.clone(),
            marker: PhantomData,
        })
    }
    /// Seal a row using its original producer. On error the row stays owned.
    pub(crate) fn try_publish_next(
        &self,
    ) -> Result<Option<PublishedRoventLockV3Record>, ArenaError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid("lock-v3 owner poisoned"))?;
        check(state.producer.is_some(), "lock-v3 producer missing")?;
        let Some(row) = state.retired.pop_front() else {
            return Ok(None);
        };
        let result = state
            .producer
            .as_ref()
            .expect("checked producer")
            .publish(row);
        match result {
            Ok(row) => Ok(Some(row)),
            Err(failure) => {
                let (error, row) = *failure;
                state.retired.push_front(row);
                Err(error)
            }
        }
    }
}

struct Observer<Q, O> {
    owner: RoventCollectionOwnerLockV3,
    marker: PhantomData<fn(Q, O)>,
}
impl<Q, O> Observer<Q, O> {
    fn try_dispatch_apply(
        &self,
        f: impl FnOnce(&mut State) -> Result<AdaptiveCollectionDispatchV2, ArenaError>,
    ) -> Result<AdaptiveCollectionDispatchV2, AdaptiveCollectionError> {
        let mut state = match self.owner.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => {
                return Ok(AdaptiveCollectionDispatchV2::Busy);
            }
            Err(error) => {
                return Err(AdaptiveCollectionError::new(
                    AdaptiveCollectionStage::Dispatch,
                    "RoventLockV3Owner",
                    error.to_string(),
                ));
            }
        };
        f(&mut state).map_err(|error| {
            AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Dispatch,
                "RoventLockV3Collection",
                error.to_string(),
            )
        })
    }
    fn try_apply(
        &self,
        stage: AdaptiveCollectionStage,
        f: impl FnOnce(&mut State) -> Result<AdaptiveCollectionBoundaryV2, ArenaError>,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        let mut state = match self.owner.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => {
                return Ok(AdaptiveCollectionBoundaryV2::Busy);
            }
            Err(error) => {
                return Err(AdaptiveCollectionError::new(
                    stage,
                    "RoventLockV3Owner",
                    error.to_string(),
                ));
            }
        };
        f(&mut state).map_err(|error| {
            AdaptiveCollectionError::new(stage, "RoventLockV3Collection", error.to_string())
        })
    }

    fn apply<T>(
        &self,
        stage: AdaptiveCollectionStage,
        f: impl FnOnce(&mut State) -> Result<T, ArenaError>,
    ) -> Result<T, AdaptiveCollectionError> {
        let mut state = self.owner.state.try_lock().map_err(|error| {
            AdaptiveCollectionError::new(stage, "RoventLockV3Owner", error.to_string())
        })?;
        f(&mut state).map_err(|error| {
            AdaptiveCollectionError::new(stage, "RoventLockV3Collection", error.to_string())
        })
    }
}
impl State {
    fn producer(&mut self) -> Result<&mut RoventCorpusProducerLockV3, ArenaError> {
        self.producer
            .as_mut()
            .ok_or_else(|| invalid("lock-v3 root not admitted"))
    }

    /// Process the same fenced ticket list incrementally. Busy keeps the
    /// current ticket and its reservation untouched; earlier completed
    /// cleanup remains indexed by exact ticket so retry cannot duplicate it.
    fn try_settle_tickets(
        &mut self,
        admission: &RoventCollectionRootAdmissionV3,
        tickets: &[RoleTicketEnvelope],
    ) -> Result<AdaptiveCollectionBoundaryV2, ArenaError> {
        self.producer()?.validate_admission(admission)?;
        for ticket in tickets {
            if let Some(original) = self.handled.get(&ticket.id) {
                check(original == ticket, "lock-v3 settled ticket differs")?;
                continue;
            }
            if let Some(staged) = self.staged.get(&ticket.id) {
                check(staged.ticket == *ticket, "lock-v3 staged ticket differs")?;
            }
            if let Some(pending) = self.pending.get(&ticket.id) {
                check(pending.ticket == *ticket, "lock-v3 pending ticket differs")?;
            }
            let unstarted = match self
                .registration
                .cancel_unstarted_lock_v3(ticket, admission)?
            {
                LockV3CancelAttempt::Busy => return Ok(AdaptiveCollectionBoundaryV2::Busy),
                LockV3CancelAttempt::Missing => None,
                LockV3CancelAttempt::Canceled(reservation) => Some(reservation),
            };
            if let Some(staged) = self.staged.remove(&ticket.id) {
                self.producer()?
                    .discard_reserved_input(staged.reservation.sequence());
            }
            if let Some(reservation) = unstarted {
                self.producer()?
                    .discard_reserved_input(reservation.sequence());
            }
            if let Some(pending) = self.pending.remove(&ticket.id) {
                self.producer()?.discard_reserved_input(pending.sequence);
                self.producer()?.discard_unpublished(ticket.id);
            }
            self.handled.insert(ticket.id, *ticket);
        }
        Ok(AdaptiveCollectionBoundaryV2::Ready)
    }
    fn try_drain(&mut self) -> Result<AdaptiveCollectionBoundaryV2, ArenaError> {
        let admission = self
            .admission
            .as_ref()
            .ok_or_else(|| invalid("lock-v3 root missing"))?
            .clone();
        loop {
            let capture = match self.router.try_take(&admission)? {
                CapturePollV3::Ready(capture) => capture,
                CapturePollV3::Empty => return Ok(AdaptiveCollectionBoundaryV2::Ready),
                CapturePollV3::Busy => return Ok(AdaptiveCollectionBoundaryV2::Busy),
            };
            if !capture.admission().same_original_admission(&admission) {
                self.rejected_captures.push(capture);
                return Err(invalid("lock-v3 captured input belongs to another root"));
            }
            if let Err(failure) = self.producer()?.admit_captured_input(capture) {
                let (error, input) = *failure;
                self.rejected_captures.push(input);
                return Err(error);
            }
        }
    }
    fn drain(&mut self) -> Result<(), ArenaError> {
        match self.try_drain()? {
            AdaptiveCollectionBoundaryV2::Ready => Ok(()),
            AdaptiveCollectionBoundaryV2::Busy => Err(invalid("lock-v3 capture router busy")),
        }
    }
    fn try_retire(
        &mut self,
        witness: RoventCollectionConsumedWitnessV3<'_>,
    ) -> Result<AdaptiveCollectionBoundaryV2, ArenaError> {
        let ticket = *witness.accepted().ticket();
        let pending = self
            .pending
            .get(&ticket.id)
            .ok_or_else(|| invalid("lock-v3 consumed ticket not registered"))?;
        check(
            pending.ticket == ticket && pending.accepted && self.retired.len() < MAX_RETIRED,
            "lock-v3 consumption identity or outbox capacity",
        )?;
        if self.try_drain()? == AdaptiveCollectionBoundaryV2::Busy {
            return Ok(AdaptiveCollectionBoundaryV2::Busy);
        }
        let row = self.producer()?.retire_consumed(witness)?;
        self.pending.remove(&ticket.id);
        self.handled.insert(ticket.id, ticket);
        self.retired.push_back(row);
        Ok(AdaptiveCollectionBoundaryV2::Ready)
    }
    fn retire(&mut self, witness: RoventCollectionConsumedWitnessV3<'_>) -> Result<(), ArenaError> {
        match self.try_retire(witness)? {
            AdaptiveCollectionBoundaryV2::Ready => Ok(()),
            AdaptiveCollectionBoundaryV2::Busy => Err(invalid("lock-v3 capture router busy")),
        }
    }
    fn try_abandon(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<AdaptiveCollectionBoundaryV2, ArenaError> {
        let registered = if let Some(pending) = self.pending.get(&ticket.id) {
            check(
                pending.ticket == *ticket,
                "lock-v3 abandoned ticket differs",
            )?;
            true
        } else if let Some(original) = self.handled.get(&ticket.id) {
            check(
                original == ticket,
                "lock-v3 abandoned handled ticket differs",
            )?;
            true
        } else {
            if let Some(staged) = self.staged.get(&ticket.id) {
                check(
                    staged.ticket == *ticket,
                    "lock-v3 abandoned staged ticket differs",
                )?;
            }
            false
        };
        if registered {
            if let Some(route) = &self.speculative_root {
                let admission = self
                    .admission
                    .as_ref()
                    .ok_or_else(|| invalid("lock-v3 admission missing"))?;
                check(
                    route.matches_admission(admission),
                    "lock-v3 abandoned route admission mismatch",
                )?;
                if route.try_cancel_ticket(ticket)?.is_none() {
                    return Ok(AdaptiveCollectionBoundaryV2::Busy);
                }
            }
        }
        self.abandoned.insert(ticket.id, *ticket);
        Ok(AdaptiveCollectionBoundaryV2::Ready)
    }
}

impl<Q, O: ProposerRankingOutput> AdaptiveCollectionObserverV3<Q, O> for Observer<Q, O> {
    fn admitted(
        &mut self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Dispatch, |state| {
            check(
                state.admission.is_none()
                    && state.producer.is_some()
                    && state.position.is_some()
                    && state
                        .prepared_admission
                        .as_ref()
                        .is_some_and(|original| original.same_original_admission(admission)),
                "lock-v3 owner permits one original root only",
            )?;
            check(
                state
                    .route_admission
                    .as_ref()
                    .is_none_or(|original| original.same_original_admission(admission)),
                "lock-v3 admitted root differs from sealed speculative route",
            )?;
            state.admission = Some(admission.clone());
            Ok(())
        })
    }
    fn dispatch(
        &mut self,
        witness: RoventCollectionDispatchWitnessV3<'_>,
        input: &FrameInput<ReviewView, Q>,
    ) -> Result<AdaptiveCollectionDispatchV2, AdaptiveCollectionError> {
        self.try_dispatch_apply(|state| {
            let ticket = *witness.ticket();
            state.producer()?.validate_admission(witness.admission())?;
            check(
                !state.settled
                    && input.view_seal == ticket.view
                    && input.question_digest == ticket.question
                    && !state.pending.contains_key(&ticket.id)
                    && !state.handled.contains_key(&ticket.id),
                "lock-v3 dispatch identity",
            )?;
            if state.pending.len() + state.staged.len() >= MAX_INFLIGHT
                && !state.staged.contains_key(&ticket.id)
            {
                return Ok(AdaptiveCollectionDispatchV2::Busy);
            }
            // Reserve outbox capacity when admitting the ticket, not when a
            // successfully completed native result tries to retire.
            if !state.staged.contains_key(&ticket.id)
                && state.retired.len() + state.pending.len() + state.staged.len() >= MAX_RETIRED
            {
                return Ok(AdaptiveCollectionDispatchV2::Busy);
            }
            if !state.staged.contains_key(&ticket.id) {
                let context = GenerationCaptureContextV3 {
                    game_id: state.metadata.game_id.clone(),
                    opening_id: state.metadata.opening_id.clone(),
                    line_genealogy_id: state.metadata.line_genealogy_id.clone(),
                    model_epoch: state.manifest.registration.model_epoch,
                    context_epoch: state.manifest.registration.context_epoch,
                    capture_sequence: 0,
                    work_scope: GenerationCaptureWorkScopeV3::from_native_ticket(&ticket),
                    actual_position: state
                        .position
                        .as_ref()
                        .ok_or_else(|| invalid("lock-v3 Rules root missing"))?
                        .clone(),
                    actual_history: state.history.clone(),
                    action_support: GenerationActionSupportV3::non_validator(),
                };
                let lineage = state.metadata.lineage.clone();
                let Some(reservation) = state
                    .producer()?
                    .try_reserve_input(&ticket, context, lineage)?
                else {
                    return Ok(AdaptiveCollectionDispatchV2::Busy);
                };
                state.staged.insert(
                    ticket.id,
                    Staged {
                        ticket,
                        reservation,
                    },
                );
            }
            check(
                state
                    .staged
                    .get(&ticket.id)
                    .is_some_and(|staged| staged.ticket == ticket),
                "lock-v3 retry changed ticket",
            )?;
            let staged = state
                .staged
                .remove(&ticket.id)
                .expect("checked staged input");
            let sequence = staged.reservation.sequence();
            match state.registration.try_register_lock_v3(
                &ticket,
                staged.reservation,
                witness.admission(),
                state.speculative_root.as_ref(),
            ) {
                LockV3RegistrationAttempt::Ready => {
                    state.pending.insert(
                        ticket.id,
                        Pending {
                            ticket,
                            sequence,
                            accepted: false,
                        },
                    );
                    Ok(AdaptiveCollectionDispatchV2::Captured)
                }
                LockV3RegistrationAttempt::Busy(reservation) => {
                    state.staged.insert(
                        ticket.id,
                        Staged {
                            ticket,
                            reservation,
                        },
                    );
                    Ok(AdaptiveCollectionDispatchV2::Busy)
                }
                LockV3RegistrationAttempt::Failed(error, reservation) => {
                    state.failed_reservations.push(Staged {
                        ticket,
                        reservation,
                    });
                    Err(error)
                }
            }
        })
    }
    fn native_event(
        &mut self,
        event: RoventCollectionNativeEventWitnessV3<'_, O>,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::NativeEvent, |state| {
            state.producer()?.validate_admission(event.admission())?;
            // Record original failures even if the capture queue is unhealthy.
            let failed = match event.event() {
                TicketNativeEvent::Completed {
                    observation_failures,
                    ..
                }
                | TicketNativeEvent::Retired {
                    observation_failures,
                    ..
                } => !observation_failures.is_empty(),
                _ => false,
            };
            if !failed {
                state.drain()?;
            }
            state.producer()?.observe_native_event(event)
        })
    }
    fn accepted(
        &mut self,
        witness: RoventCollectionAcceptedWitnessV3<'_>,
        _: &FrameInput<ReviewView, Q>,
        _: &O,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Accepted, |state| {
            state.producer()?.validate_admission(witness.admission())?;
            let ticket = witness.accepted().ticket();
            let pending = state
                .pending
                .get_mut(&ticket.id)
                .ok_or_else(|| invalid("lock-v3 acceptance not registered"))?;
            check(
                pending.ticket == *ticket && !pending.accepted,
                "lock-v3 duplicate or changed acceptance",
            )?;
            pending.accepted = true;
            Ok(())
        })
    }
    fn consumed(
        &mut self,
        witness: RoventCollectionConsumedWitnessV3<'_>,
        _: RoleOnlySearchConsumptionV2,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Accepted, |state| {
            state.retire(witness)
        })
    }
    fn adaptive_consumed(
        &mut self,
        witness: RoventCollectionConsumedWitnessV3<'_>,
        _: bool,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Accepted, |state| {
            state.retire(witness)
        })
    }
    fn adaptive_root_ranking(
        &mut self,
        accepted: &RoventRootRankingAcceptanceV3,
        _: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Accepted, |state| {
            state.producer()?.validate_admission(accepted.admission())?;
            let ticket = accepted.ranking().source().ticket;
            let pending = state
                .pending
                .get(&ticket.id)
                .ok_or_else(|| invalid("lock-v3 ranking not registered"))?;
            check(
                pending.ticket == ticket && state.retired.len() < MAX_RETIRED,
                "lock-v3 ranking identity or capacity",
            )?;
            state.drain()?;
            let row = state
                .producer()?
                .retire_root_ranking(accepted, output.proposer_ranking_output())?;
            state.pending.remove(&ticket.id);
            state.handled.insert(ticket.id, ticket);
            state.retired.push_back(row);
            Ok(())
        })
    }
    fn abandoned(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Abandoned, |state| {
            match state.try_abandon(ticket)? {
                AdaptiveCollectionBoundaryV2::Ready => Ok(()),
                AdaptiveCollectionBoundaryV2::Busy => {
                    Err(invalid("lock-v3 speculative cancellation busy"))
                }
            }
        })
    }
    fn settled(&mut self, tickets: &[RoleTicketEnvelope]) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Settled, |state| {
            let admission = state
                .admission
                .as_ref()
                .ok_or_else(|| invalid("lock-v3 admission missing"))?
                .clone();
            // The manager calls settled only after its native fence. Preserve any
            // failed captured input and report the failure instead of success.
            if let Err(error) = state.drain() {
                // Preserve the original failed object even if draining itself
                // reports the original root-scoped capture failure. A busy
                // router retains it and the returned error remains observable.
                if let Ok(FailurePollV3::Ready(input)) = state.router.try_take_failure(&admission) {
                    state.failed_captures.push(input);
                    // Keep earlier queued originals after this root's fence;
                    // these inputs are evidence only and are not admitted or
                    // published after the original collection failure.
                    while let Ok(CapturePollV3::Ready(input)) =
                        state.router.try_take_for_cleanup(&admission)
                    {
                        state.rejected_captures.push(input);
                    }
                }
                return Err(error);
            }
            if state.try_settle_tickets(&admission, tickets)? == AdaptiveCollectionBoundaryV2::Busy
            {
                return Err(invalid("lock-v3 fenced ticket cleanup busy"));
            }
            check(
                state.staged.is_empty()
                    && state.pending.is_empty()
                    && state.failed_reservations.is_empty()
                    && state.failed_captures.is_empty()
                    && state.rejected_captures.is_empty()
                    && state.producer()?.incomplete_capture_count() == 0,
                "lock-v3 fenced root retains unresolved evidence",
            )?;
            state.settled = true;
            Ok(())
        })
    }

    fn try_native_event(
        &mut self,
        event: RoventCollectionNativeEventWitnessV3<'_, O>,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.try_apply(AdaptiveCollectionStage::NativeEvent, |state| {
            state.producer()?.validate_admission(event.admission())?;
            let failed = match event.event() {
                TicketNativeEvent::Completed {
                    observation_failures,
                    ..
                }
                | TicketNativeEvent::Retired {
                    observation_failures,
                    ..
                } => !observation_failures.is_empty(),
                _ => false,
            };
            // Never apply the original event until the captures needed by it
            // have been admitted. Earlier captures drained here stay admitted.
            if !failed && state.try_drain()? == AdaptiveCollectionBoundaryV2::Busy {
                return Ok(AdaptiveCollectionBoundaryV2::Busy);
            }
            state.producer()?.observe_native_event(event)?;
            Ok(AdaptiveCollectionBoundaryV2::Ready)
        })
    }

    fn try_accepted(
        &mut self,
        witness: RoventCollectionAcceptedWitnessV3<'_>,
        _: &FrameInput<ReviewView, Q>,
        _: &O,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.try_apply(AdaptiveCollectionStage::Accepted, |state| {
            state.producer()?.validate_admission(witness.admission())?;
            let ticket = witness.accepted().ticket();
            let pending = state
                .pending
                .get_mut(&ticket.id)
                .ok_or_else(|| invalid("lock-v3 acceptance not registered"))?;
            check(
                pending.ticket == *ticket && !pending.accepted,
                "lock-v3 duplicate or changed acceptance",
            )?;
            pending.accepted = true;
            Ok(AdaptiveCollectionBoundaryV2::Ready)
        })
    }

    fn try_consumed(
        &mut self,
        witness: RoventCollectionConsumedWitnessV3<'_>,
        _: RoleOnlySearchConsumptionV2,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.try_apply(AdaptiveCollectionStage::Accepted, |state| {
            state.try_retire(witness)
        })
    }

    fn try_adaptive_consumed(
        &mut self,
        witness: RoventCollectionConsumedWitnessV3<'_>,
        _: bool,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.try_apply(AdaptiveCollectionStage::Accepted, |state| {
            state.try_retire(witness)
        })
    }

    fn try_adaptive_root_ranking(
        &mut self,
        accepted: &RoventRootRankingAcceptanceV3,
        _: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.try_apply(AdaptiveCollectionStage::Accepted, |state| {
            state.producer()?.validate_admission(accepted.admission())?;
            let ticket = accepted.ranking().source().ticket;
            let pending = state
                .pending
                .get(&ticket.id)
                .ok_or_else(|| invalid("lock-v3 ranking not registered"))?;
            check(
                pending.ticket == ticket && state.retired.len() < MAX_RETIRED,
                "lock-v3 ranking identity or capacity",
            )?;
            if state.try_drain()? == AdaptiveCollectionBoundaryV2::Busy {
                return Ok(AdaptiveCollectionBoundaryV2::Busy);
            }
            let row = state
                .producer()?
                .retire_root_ranking(accepted, output.proposer_ranking_output())?;
            state.pending.remove(&ticket.id);
            state.handled.insert(ticket.id, ticket);
            state.retired.push_back(row);
            Ok(AdaptiveCollectionBoundaryV2::Ready)
        })
    }

    fn try_abandoned(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.try_apply(AdaptiveCollectionStage::Abandoned, |state| {
            state.try_abandon(ticket)
        })
    }

    fn try_settled(
        &mut self,
        tickets: &[RoleTicketEnvelope],
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        self.try_apply(AdaptiveCollectionStage::Settled, |state| {
            let admission = state
                .admission
                .as_ref()
                .ok_or_else(|| invalid("lock-v3 admission missing"))?
                .clone();
            match state.try_drain() {
                Ok(AdaptiveCollectionBoundaryV2::Busy) => {
                    return Ok(AdaptiveCollectionBoundaryV2::Busy);
                }
                Ok(AdaptiveCollectionBoundaryV2::Ready) => {}
                Err(error) => {
                    if let Ok(FailurePollV3::Ready(input)) =
                        state.router.try_take_failure(&admission)
                    {
                        state.failed_captures.push(input);
                        while let Ok(CapturePollV3::Ready(input)) =
                            state.router.try_take_for_cleanup(&admission)
                        {
                            state.rejected_captures.push(input);
                        }
                    }
                    return Err(error);
                }
            }
            // No caller settlement evidence has been applied before this point.
            if state.try_settle_tickets(&admission, tickets)? == AdaptiveCollectionBoundaryV2::Busy
            {
                return Ok(AdaptiveCollectionBoundaryV2::Busy);
            }
            check(
                state.staged.is_empty()
                    && state.pending.is_empty()
                    && state.failed_reservations.is_empty()
                    && state.failed_captures.is_empty()
                    && state.rejected_captures.is_empty()
                    && state.producer()?.incomplete_capture_count() == 0,
                "lock-v3 fenced root retains unresolved evidence",
            )?;
            state.settled = true;
            Ok(AdaptiveCollectionBoundaryV2::Ready)
        })
    }
}
