//! Single-original-root system-runtime collection observer.
//!
//! A channel and producer belong to this owner for their entire lifetime. In
//! particular, a second root cannot silently inherit an unfinished capture.
//! The caller must keep this owner alive through the native worker fence and
//! preserve it on errors. This is not a teacher or GPU completion receipt.
use super::capture_router_v3::{CapturePollV3, CaptureRouterV3, FailurePollV3};
use super::native_capture::{LockV3RegistrationAttempt, RoventFailedCapturedInput};
use super::producer_v3::{
    RoventCorpusProducerLockV3, RoventInputCaptureReservationV3, RoventRetiredLockV3Record,
};
use super::*;
use crate::pals_collect_v3::{GenerationActionSupportV3, GenerationCaptureWorkScopeV3};
use rz_experiments::rovent_corpus_lock_v3::record::RoventCorpusManifestLockV3;
use rz_search::pals::coordinator::FrameInput;
use rz_uci::pals_adaptive::{
    AdaptiveCollectionDispatchV2, AdaptiveCollectionError, AdaptiveCollectionObserverBoxV3,
    AdaptiveCollectionObserverV3, AdaptiveCollectionStage, RoleOnlySearchConsumptionV2,
    RoventCollectionAcceptedWitnessV3, RoventCollectionConsumedWitnessV3,
    RoventCollectionDispatchWitnessV3, RoventCollectionNativeEventWitnessV3,
    RoventCollectionRootAdmissionV3, RoventRootRankingAcceptanceV3,
};
use std::{
    collections::VecDeque,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

const MAX_INFLIGHT: usize = 16;
const MAX_RETIRED: usize = 256;

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
    admission: Option<RoventCollectionRootAdmissionV3>,
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
                admission: None,
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
    fn drain(&mut self) -> Result<(), ArenaError> {
        let admission = self
            .admission
            .as_ref()
            .ok_or_else(|| invalid("lock-v3 root missing"))?
            .clone();
        loop {
            let capture = match self.router.try_take(&admission)? {
                CapturePollV3::Ready(capture) => capture,
                CapturePollV3::Empty => return Ok(()),
                CapturePollV3::Busy => return Err(invalid("lock-v3 capture router busy")),
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
    fn retire(&mut self, witness: RoventCollectionConsumedWitnessV3<'_>) -> Result<(), ArenaError> {
        let ticket = *witness.accepted().ticket();
        let pending = self
            .pending
            .get(&ticket.id)
            .ok_or_else(|| invalid("lock-v3 consumed ticket not registered"))?;
        check(
            pending.ticket == ticket && pending.accepted && self.retired.len() < MAX_RETIRED,
            "lock-v3 consumption identity or outbox capacity",
        )?;
        self.drain()?;
        let row = self.producer()?.retire_consumed(witness)?;
        self.pending.remove(&ticket.id);
        self.handled.insert(ticket.id, ticket);
        self.retired.push_back(row);
        Ok(())
    }
}

impl<Q, O: ProposerRankingOutput> AdaptiveCollectionObserverV3<Q, O> for Observer<Q, O> {
    fn admitted(
        &mut self,
        admission: &RoventCollectionRootAdmissionV3,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Dispatch, |state| {
            check(
                state.admission.is_none() && state.producer.is_none(),
                "lock-v3 owner permits one original root only",
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
            let producer = RoventCorpusProducerLockV3::new_shared(
                Arc::clone(&state.manifest),
                admission.clone(),
            )?;
            state.history = history;
            state.position = Some(position);
            state.producer = Some(producer);
            state.admission = Some(admission.clone());
            Ok(())
        })
    }
    fn dispatch(
        &mut self,
        witness: RoventCollectionDispatchWitnessV3<'_>,
        input: &FrameInput<ReviewView, Q>,
    ) -> Result<AdaptiveCollectionDispatchV2, AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Dispatch, |state| {
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
            if let Some(pending) = state.pending.get(&ticket.id) {
                check(
                    pending.ticket == *ticket,
                    "lock-v3 abandoned ticket differs",
                )?;
            }
            state.abandoned.insert(ticket.id, *ticket);
            Ok(())
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
                }
                return Err(error);
            }
            for ticket in tickets {
                if let Some(original) = state.handled.get(&ticket.id) {
                    check(original == ticket, "lock-v3 settled ticket differs")?;
                    continue;
                }
                if let Some(staged) = state.staged.get(&ticket.id) {
                    check(staged.ticket == *ticket, "lock-v3 staged ticket differs")?;
                }
                if let Some(staged) = state.staged.remove(&ticket.id) {
                    state
                        .producer()?
                        .discard_reserved_input(staged.reservation.sequence());
                }
                if let Some(reservation) = state
                    .registration
                    .cancel_unstarted_lock_v3(ticket, &admission)?
                {
                    state
                        .producer()?
                        .discard_reserved_input(reservation.sequence());
                }
                if let Some(pending) = state.pending.get(&ticket.id) {
                    check(pending.ticket == *ticket, "lock-v3 pending ticket differs")?;
                }
                if let Some(pending) = state.pending.remove(&ticket.id) {
                    state.producer()?.discard_reserved_input(pending.sequence);
                    state.producer()?.discard_unpublished(ticket.id);
                }
                state.handled.insert(ticket.id, *ticket);
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
}
