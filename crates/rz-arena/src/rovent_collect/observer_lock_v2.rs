//! Observer for a single original input-lock /2 Collection root.
//! Input capture, native completion, acceptance and actual search consumption
//! remain separate. This object does not own or retire the teacher process.
use super::*;
use crate::pals_collect_v3::{GenerationActionSupportV3, GenerationCaptureWorkScopeV3};
use rz_search::pals::coordinator::FrameInput;
use rz_uci::pals_adaptive::{
    AdaptiveCollectionError, AdaptiveCollectionObserverV2, AdaptiveCollectionStage,
    RoleOnlySearchConsumptionV2, RoventCollectionAcceptedWitnessV2,
    RoventCollectionConsumedWitnessV2, RoventCollectionDispatchWitnessV2,
    RoventCollectionNativeEventWitnessV2, RoventCollectionRootAdmissionV2,
    RoventRootRankingAcceptanceV2,
};
use std::{collections::VecDeque, marker::PhantomData, sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}}};

const CAPACITY: usize = 16;
const RETAINED_ROOT_CAPACITY: usize = 256;

struct Pending {
    ticket: RoleTicketEnvelope,
    sequence: u64,
    accepted: bool,
}

struct State {
    manifest: rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2,
    producer: Option<RoventCorpusProducerLockV2>,
    producer_root: Option<String>,
    prior_producers: Vec<(String, RoventCorpusProducerLockV2)>,
    registration: RoventNativeCaptureRegistration,
    receiver: RoventCapturedInputReceiver,
    metadata: RoventCollectionMetadata,
    position: Option<Position>,
    history: Vec<BoardMove>,
    pending: BTreeMap<RequestId, Pending>,
    handled_tickets: BTreeMap<RequestId, RoleTicketEnvelope>,
    retired: VecDeque<RoventRetiredLockV2Record>,
    admitted: bool,
    settled: bool,
    failure: Option<AdaptiveCollectionError>,
    failed_captures: Vec<super::native_capture::RoventFailedCapturedInput>,
}

/// A clone refers to the same original producer and outbox; it does not issue
/// or reconstruct a live admission. A new root needs its own original owner.
#[derive(Clone)]
pub struct RoventCollectionOwnerLockV2 {
    state: Arc<Mutex<State>>,
    callback_lock_failure: Arc<AtomicBool>,
}

impl RoventCollectionOwnerLockV2 {
    pub fn new(
        manifest: rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2,
        metadata: RoventCollectionMetadata,
    ) -> Result<(Self, RoventNativeInputObserver), ArenaError> {
        check(generation_identifier(&metadata.game_id)
            && generation_identifier(&metadata.opening_id)
            && generation_identifier(&metadata.line_genealogy_id)
            && manifest.registration.opening_splits.contains_key(&metadata.opening_id),
            "lock-v2 collection metadata/opening is not registered")?;
        let (registration, observer, receiver) = rovent_native_capture_channel();
        Ok((Self { state: Arc::new(Mutex::new(State {
            manifest, producer: None, producer_root: None, prior_producers: Vec::new(), registration, receiver, metadata, position: None, history: Vec::new(),
            pending: BTreeMap::new(), handled_tickets: BTreeMap::new(), retired: VecDeque::new(), admitted: false,
            settled: false, failure: None, failed_captures: Vec::new(),
        })), callback_lock_failure: Arc::new(AtomicBool::new(false)) }, observer))
    }

    pub fn observer<Q: 'static, O: ProposerRankingOutput + 'static>(&self)
        -> Box<dyn AdaptiveCollectionObserverV2<Q, O>> {
        Box::new(Observer { owner: self.clone(), marker: PhantomData })
    }

    pub fn with_producer<T>(&self, action: impl FnOnce(&mut RoventCorpusProducerLockV2)
        -> Result<T, ArenaError>) -> Result<T, ArenaError> {
        let mut state = self.state.lock().map_err(|_| invalid("lock-v2 collection owner poisoned"))?;
        action(state.producer.as_mut().ok_or_else(|| invalid("lock-v2 root not yet admitted"))?)
    }

    /// Targets an already retired row's original producer; no input receipt or
    /// stored digest is promoted into a new live root capability.
    pub fn with_retired_producer<T>(&self, row: &RoventRetiredLockV2Record,
        action: impl FnOnce(&mut RoventCorpusProducerLockV2) -> Result<T, ArenaError>) -> Result<T, ArenaError> {
        let mut state = self.state.lock().map_err(|_| invalid("lock-v2 collection owner poisoned"))?;
        let root = row.admission.raw_sha256();
        if state.producer_root.as_deref() == Some(root) {
            return action(state.producer.as_mut().ok_or_else(|| invalid("lock-v2 active producer missing"))?);
        }
        let (_, producer) = state.prior_producers.iter_mut().find(|(id, _)| id == root)
            .ok_or_else(|| invalid("lock-v2 retired row's original producer missing"))?;
        action(producer)
    }

    pub fn take_retired(&self) -> Result<Option<RoventRetiredLockV2Record>, ArenaError> {
        Ok(self.state.lock().map_err(|_| invalid("lock-v2 collection owner poisoned"))?.retired.pop_front())
    }

    pub fn failure(&self) -> Result<Option<AdaptiveCollectionError>, ArenaError> {
        Ok(self.state.lock().map_err(|_| invalid("lock-v2 collection owner poisoned"))?.failure.clone())
    }

    /// A local collection-status observation, never a teacher retirement,
    /// physical fence, corpus publication or trainer admission certificate.
    pub fn status(&self) -> Result<RoventCollectionOwnerStatusLockV2, ArenaError> {
        let state = self.state.lock().map_err(|_| invalid("lock-v2 collection owner poisoned"))?;
        Ok(RoventCollectionOwnerStatusLockV2 {
            admitted: state.admitted, settled: state.settled,
            pending_count: state.pending.len(), pending_retired_count: state.retired.len(),
            incomplete_capture_count: state.producer.as_ref().map_or(0, |producer| producer.incomplete_capture_count()),
            retained_prior_roots: state.prior_producers.len(),
            retained_prior_root_limit: RETAINED_ROOT_CAPACITY,
            retained_failed_captures: state.failed_captures.len(),
            failure: state.failure.clone(),
            callback_lock_failure: self.callback_lock_failure.load(Ordering::Acquire),
        })
    }
}

#[derive(Clone, Debug)]
pub struct RoventCollectionOwnerStatusLockV2 {
    pub admitted: bool,
    pub settled: bool,
    pub pending_count: usize,
    pub pending_retired_count: usize,
    pub incomplete_capture_count: usize,
    /// Root producers are held for caller-owned retired rows. No automatic
    /// release is safe until all row and target ownership can be proven.
    pub retained_prior_roots: usize,
    pub retained_prior_root_limit: usize,
    pub retained_failed_captures: usize,
    pub failure: Option<AdaptiveCollectionError>,
    pub callback_lock_failure: bool,
}

struct Observer<Q, O> {
    owner: RoventCollectionOwnerLockV2,
    marker: PhantomData<fn(Q, O)>,
}

impl State {
    fn producer(&self) -> Result<&RoventCorpusProducerLockV2, ArenaError> {
        self.producer.as_ref().ok_or_else(|| invalid("lock-v2 root not yet admitted"))
    }
    fn producer_mut(&mut self) -> Result<&mut RoventCorpusProducerLockV2, ArenaError> {
        self.producer.as_mut().ok_or_else(|| invalid("lock-v2 root not yet admitted"))
    }
    fn drain(&mut self) -> Result<(), ArenaError> {
        let producer = self.producer.as_mut().ok_or_else(|| invalid("lock-v2 root not yet admitted"))?;
        match self.receiver.admit_available_lock_v2(producer) {
            Ok(_) => Ok(()),
            Err(failure) => {
                let (error, input) = *failure;
                if let Some(input) = input {
                    self.failed_captures.push(super::native_capture::RoventFailedCapturedInput::LockV2(input));
                }
                Err(error)
            }
        }
    }
    fn discard(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), ArenaError> {
        if let Some(handled) = self.handled_tickets.get(&ticket.id) {
            return check(handled == ticket, "lock-v2 handled ticket changed at settlement");
        }
        let pending = self.pending.get(&ticket.id).ok_or_else(|| invalid("lock-v2 ticket not registered"))?;
        check(pending.ticket == *ticket, "lock-v2 ticket differs from original dispatch")?;
        if let Some(sequence) = self.registration.cancel_sequence(ticket.id)? {
            self.producer_mut()?.discard_reserved_input(sequence);
        }
        if let Some(pending) = self.pending.remove(&ticket.id) {
            self.producer_mut()?.discard_reserved_input(pending.sequence);
        }
        self.producer_mut()?.discard_unpublished(ticket.id);
        self.handled_tickets.insert(ticket.id, *ticket);
        Ok(())
    }
}

impl<Q, O> Observer<Q, O> {
    fn apply<T>(&self, stage: AdaptiveCollectionStage, f: impl FnOnce(&mut State) -> Result<T, ArenaError>)
        -> Result<T, AdaptiveCollectionError> {
        let mut state = self.owner.state.try_lock().map_err(|_| {
            self.owner.callback_lock_failure.store(true, Ordering::Release);
            AdaptiveCollectionError::new(stage, "RoventCollectionUnavailable",
                "lock-v2 collection owner busy or poisoned")
        })?;
        match f(&mut state) {
            Ok(result) => Ok(result),
            Err(error) => {
                let error = AdaptiveCollectionError::new(stage, "RoventCollectionEvidence", error.to_string());
                state.failure.get_or_insert_with(|| error.clone());
                Err(error)
            }
        }
    }
}

impl<Q, O: ProposerRankingOutput> AdaptiveCollectionObserverV2<Q, O> for Observer<Q, O> {
    fn admitted(&mut self, admission: &RoventCollectionRootAdmissionV2) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Dispatch, |state| {
            check(!self.owner.callback_lock_failure.load(Ordering::Acquire)
                && (!state.admitted || state.settled) && state.pending.is_empty() && state.failure.is_none()
                && state.producer.as_ref().is_none_or(|producer| producer.incomplete_capture_count() == 0)
                && (state.producer.is_none() || state.prior_producers.len() < RETAINED_ROOT_CAPACITY),
                "lock-v2 prior root is not settled or producer retention is full")?;
            let snapshot = admission.rules().rules().snapshot();
            let history = snapshot.uci_replay(16_384).map_err(|error| invalid(error.to_string()))?.moves;
            let position = Position::from_snapshot(snapshot, rz_position::PositionLimits {
                max_history_positions: 16_385, ..rz_position::PositionLimits::default()
            }).map_err(|error| invalid(error.to_string()))?;
            let producer = RoventCorpusProducerLockV2::new(state.manifest.clone(), admission.clone())?;
            if state.producer.is_some() {
                let id = state.producer_root.as_ref().ok_or_else(|| invalid("lock-v2 previous producer identity missing"))?.clone();
                check(id != admission.raw_sha256() && !state.prior_producers.iter().any(|(old, _)| old == admission.raw_sha256()),
                    "lock-v2 new root duplicates a retained producer")?;
                let previous = state.producer.take().expect("checked previous producer");
                state.producer_root = None;
                state.prior_producers.push((id, previous));
            }
            state.producer = Some(producer);
            state.producer_root = Some(admission.raw_sha256().to_owned());
            state.handled_tickets.clear();
            state.position = Some(position);
            state.history = history;
            state.admitted = true;
            state.settled = false;
            Ok(())
        })
    }

    fn dispatch(&mut self, witness: RoventCollectionDispatchWitnessV2<'_>, input: &FrameInput<ReviewView, Q>)
        -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Dispatch, |state| {
            let ticket = witness.ticket();
            state.producer()?.validate_admission(witness.admission())?;
            check(state.admitted && !state.settled && state.failure.is_none()
                && state.retired.len() + state.pending.len() < CAPACITY
                && state.handled_tickets.len() + state.pending.len() < rz_experiments::rovent_corpus::MAX_CORPUS_ROWS
                && !state.pending.contains_key(&ticket.id) && !state.handled_tickets.contains_key(&ticket.id)
                && input.view_seal == ticket.view && input.question_digest == ticket.question,
                "lock-v2 collection dispatch scope/queue capacity")?;
            let registration = &state.manifest.registration;
            let context = GenerationCaptureContextV3 {
                game_id: state.metadata.game_id.clone(), opening_id: state.metadata.opening_id.clone(),
                line_genealogy_id: state.metadata.line_genealogy_id.clone(), model_epoch: registration.model_epoch,
                context_epoch: registration.context_epoch, capture_sequence: 0,
                work_scope: GenerationCaptureWorkScopeV3::from_native_ticket(ticket),
                actual_position: state.position.as_ref().ok_or_else(|| invalid("lock-v2 Rules root missing"))?.clone(), actual_history: state.history.clone(),
                action_support: GenerationActionSupportV3::non_validator(),
            };
            let lineage = state.metadata.lineage.clone();
            let reservation = state.producer_mut()?.reserve_input(ticket, context, lineage)?;
            let sequence = reservation.sequence();
            if let Err(error) = state.registration.register_lock_v2(ticket, reservation, witness.admission()) {
                state.producer_mut()?.discard_reserved_input(sequence);
                return Err(error);
            }
            state.pending.insert(ticket.id, Pending { ticket: *ticket, sequence, accepted: false });
            Ok(())
        })
    }

    fn native_event(&mut self, event: RoventCollectionNativeEventWitnessV2<'_, O>)
        -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::NativeEvent, |state| {
            check(!state.settled, "lock-v2 native event after settled owner fence")?;
            // Preserve the original terminal's typed observation failures even
            // if the capture queue is already unhealthy. No input is admitted
            // or published on this path; native completion remains independent.
            let failed_observation = match event.event() {
                TicketNativeEvent::Completed { observation_failures, .. }
                | TicketNativeEvent::Retired { observation_failures, .. } => !observation_failures.is_empty(),
                _ => false,
            };
            if failed_observation {
                return state.producer_mut()?.observe_native_event(event);
            }
            state.drain()?;
            let failed = match event.event() {
                TicketNativeEvent::Retired { accepted: false, result: Err(error), .. } => {
                    Some(format!("original native retirement failed: {error:?}"))
                }
                _ => None,
            };
            state.producer_mut()?.observe_native_event(event)?;
            if let Some(error) = failed { return Err(invalid(error)); }
            Ok(())
        })
    }

    fn accepted(&mut self, witness: RoventCollectionAcceptedWitnessV2<'_>, _: &FrameInput<ReviewView, Q>, _: &O)
        -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Accepted, |state| {
            state.producer()?.validate_admission(witness.admission())?;
            let ticket = witness.accepted().ticket();
            let pending = state.pending.get_mut(&ticket.id).ok_or_else(|| invalid("lock-v2 accepted ticket not registered"))?;
            check(pending.ticket == *ticket && !pending.accepted, "lock-v2 duplicate or changed acceptance")?;
            pending.accepted = true;
            Ok(())
        })
    }

    fn consumed(&mut self, witness: RoventCollectionConsumedWitnessV2<'_>, _: RoleOnlySearchConsumptionV2)
        -> Result<(), AdaptiveCollectionError> {
        self.retire_consumed(witness)
    }

    fn adaptive_consumed(&mut self, witness: RoventCollectionConsumedWitnessV2<'_>, _: bool)
        -> Result<(), AdaptiveCollectionError> {
        self.retire_consumed(witness)
    }

    fn adaptive_root_ranking(&mut self, accepted: &RoventRootRankingAcceptanceV2,
        _: &FrameInput<ReviewView, Q>, output: &O) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Accepted, |state| {
            state.producer()?.validate_admission(accepted.admission())?;
            let ticket = &accepted.ranking().source().ticket;
            let pending = state.pending.get(&ticket.id).ok_or_else(|| invalid("lock-v2 ranking ticket not registered"))?;
            check(pending.ticket == *ticket && state.retired.len() < CAPACITY,
                "lock-v2 ranking original ticket/outbox mismatch")?;
            state.drain()?;
            let row = state.producer_mut()?.retire_root_ranking(accepted, output.proposer_ranking_output())?;
            state.pending.remove(&ticket.id);
            state.handled_tickets.insert(ticket.id, *ticket);
            state.retired.push_back(row);
            Ok(())
        })
    }

    fn abandoned(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Abandoned, |state| { state.drain()?; state.discard(ticket) })
    }

    fn settled(&mut self, tickets: &[RoleTicketEnvelope]) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Settled, |state| {
            // The manager supplies this callback only after the native fence.
            // Keep failed captures as evidence without admitting or publishing them.
            let capture_failure = state.receiver.capture_failure()?;
            if capture_failure.is_some() {
                if let Some(input) = state.receiver.take_failed_input_for_cleanup()? {
                    state.failed_captures.push(input);
                }
                state.failed_captures.extend(state.receiver.take_queued_inputs_for_cleanup());
            } else {
                state.drain()?;
            }
            for ticket in tickets { state.discard(ticket)?; }
            check(state.pending.is_empty() && state.producer()?.incomplete_capture_count() == 0,
                "lock-v2 fenced root retains unresolved inputs")?;
            state.settled = true;
            if let Some(failure) = capture_failure {
                return Err(invalid(format!("native input capture failed before fenced cleanup: {failure:?}")));
            }
            Ok(())
        })
    }
}

impl<Q, O> Observer<Q, O> {
    fn retire_consumed(&self, witness: RoventCollectionConsumedWitnessV2<'_>) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Accepted, |state| {
            state.producer()?.validate_admission(witness.admission())?;
            let ticket = *witness.accepted().ticket();
            let pending = state.pending.get(&ticket.id).ok_or_else(|| invalid("lock-v2 consumed ticket not registered"))?;
            check(pending.ticket == ticket && pending.accepted && state.retired.len() < CAPACITY,
                "lock-v2 consumption lacks original acceptance or outbox capacity")?;
            state.drain()?;
            let row = state.producer_mut()?.retire_consumed(witness)?;
            state.pending.remove(&ticket.id);
            state.handled_tickets.insert(ticket.id, ticket);
            state.retired.push_back(row);
            Ok(())
        })
    }
}
