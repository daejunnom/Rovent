//! Observer for a single original input-lock /2 Collection root.
//! Input capture, native completion, acceptance and actual search consumption
//! remain separate. This object does not own or retire the teacher process.
use super::*;
use crate::pals_collect_v3::{GenerationActionSupportV3, GenerationCaptureWorkScopeV3};
use rz_search::pals::coordinator::FrameInput;
use rz_uci::pals_adaptive::{
    AdaptiveCollectionBoundaryV2, AdaptiveCollectionDispatchV2, AdaptiveCollectionError,
    AdaptiveCollectionObserverV2, AdaptiveCollectionStage, RoleOnlySearchConsumptionV2,
    RoventCollectionAcceptedWitnessV2, RoventCollectionConsumedWitnessV2,
    RoventCollectionDispatchWitnessV2, RoventCollectionNativeEventWitnessV2,
    RoventCollectionRootAdmissionV2, RoventRootRankingAcceptanceV2, RoventTicketValidationKindV2,
};
use std::{
    collections::VecDeque,
    marker::PhantomData,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
};

const CAPACITY: usize = 16;
// Bound retained rows independently of the game-wide record quota. Channel
// slots hold the original rows and do not allocate their payloads in advance.
const MAX_RETIRED_OUTBOX_CAPACITY: usize = 256;
const RETAINED_ROOT_CAPACITY: usize = 256;

struct Pending {
    ticket: RoleTicketEnvelope,
    sequence: u64,
    accepted: bool,
}
struct StagedInput {
    ticket: RoleTicketEnvelope,
    admission: RoventCollectionRootAdmissionV2,
    reservation: RoventInputCaptureReservationV2,
}

#[derive(Clone)]
struct DisabledRoot {
    admission: RoventCollectionRootAdmissionV2,
    first_ticket: RoleTicketEnvelope,
    reason: RoventCollectionDisableReasonV2,
    original_failure: Option<AdaptiveCollectionError>,
    prior_tickets: Vec<RoleTicketEnvelope>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventCollectionDisableReasonV2 {
    RetiredOutboxFull,
    TicketEvidenceLimit,
    GameRecordLimit,
    ObserverFailure,
}

/// Observer installation has not started. A blocked owner is distinct from
/// a transient lock collision and retains the original collection state.
pub(crate) enum OwnerInstallReadinessV2 {
    Ready,
    Blocked(AdaptiveCollectionError),
}

struct State {
    game_record_limit: usize,
    retired_outbox_capacity: usize,
    game_retired_total: usize,
    manifest: Arc<rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2>,
    producer: Option<RoventCorpusProducerLockV2>,
    producer_root: Option<String>,
    prior_producers: Vec<(String, RoventCorpusProducerLockV2)>,
    current_admission: Option<RoventCollectionRootAdmissionV2>,
    settled_roots: Vec<RoventCollectionRootAdmissionV2>,
    registration: RoventNativeCaptureRegistration,
    receiver: RoventCapturedInputReceiver,
    metadata: RoventCollectionMetadata,
    position: Option<Position>,
    history: Vec<BoardMove>,
    pending: BTreeMap<RequestId, Pending>,
    staged: BTreeMap<RequestId, StagedInput>,
    failed_staged: Vec<StagedInput>,
    disabled: Option<DisabledRoot>,
    prior_disabled: Vec<DisabledRoot>,
    disabled_prior: BTreeMap<RequestId, RoleTicketEnvelope>,
    handled_tickets: BTreeMap<RequestId, RoleTicketEnvelope>,
    retired: VecDeque<RoventRetiredLockV2Record>,
    retired_sender: mpsc::SyncSender<RoventRetiredLockV2Record>,
    retired_queued: Arc<AtomicUsize>,
    admitted: bool,
    settled: bool,
    failure: Option<AdaptiveCollectionError>,
    failed_captures: Vec<super::native_capture::RoventFailedCapturedInput>,
    #[cfg(feature = "test-fixtures")]
    test_completed: BTreeMap<RequestId, (RoleTicketEnvelope, u64, bool, bool)>,
    #[cfg(feature = "test-fixtures")]
    test_started: BTreeMap<RequestId, RoleTicketEnvelope>,
}

/// A clone refers to the same original producer and outbox; it does not issue
/// or reconstruct a live admission. A new root needs its own original owner.
#[derive(Clone)]
pub struct RoventCollectionOwnerLockV2 {
    state: Arc<Mutex<State>>,
    game_gate: Arc<AtomicU64>,
    registration: RoventNativeCaptureRegistration,
    callback_lock_failure: Arc<AtomicBool>,
    retired_receiver: Arc<Mutex<mpsc::Receiver<RoventRetiredLockV2Record>>>,
    retired_queued: Arc<AtomicUsize>,
}

impl RoventCollectionOwnerLockV2 {
    /// Called with the shared game install guard held, before `port.install`.
    /// The owner and capture registration use only nonblocking lock attempts.
    pub(crate) fn try_install_readiness(
        &self,
        admission: &RoventCollectionRootAdmissionV2,
    ) -> Result<Option<OwnerInstallReadinessV2>, ArenaError> {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("lock-v2 collection owner poisoned")),
        };
        let blocked = |detail: &'static str| OwnerInstallReadinessV2::Blocked(
            AdaptiveCollectionError::new(AdaptiveCollectionStage::Dispatch, "CollectionOwnerNotReady", detail));
        if let Some(error) = &state.failure {
            return Ok(Some(OwnerInstallReadinessV2::Blocked(error.clone())));
        }
        if self.callback_lock_failure.load(Ordering::Acquire) {
            return Ok(Some(blocked("original collection callback lock failure is retained")));
        }
        check(state.admitted == state.current_admission.is_some()
            && state.admitted == state.producer.is_some(),
            "lock-v2 original owner admission/producer state is inconsistent")?;
        if (state.admitted && !state.settled) || !state.pending.is_empty() || !state.staged.is_empty()
            || !state.disabled_prior.is_empty() || !state.retired.is_empty()
            || self.retired_queued.load(Ordering::Acquire) != 0 {
            return Ok(Some(blocked("previous root still has unsettled or undrained original work")));
        }
        if state.disabled.is_some() || !state.prior_disabled.is_empty()
            || !state.failed_staged.is_empty() || !state.failed_captures.is_empty()
            || state.producer.as_ref().is_some_and(|producer| producer.incomplete_capture_count() != 0) {
            return Ok(Some(blocked("previous root has retained disabled, failed or incomplete capture")));
        }
        if state.producer.is_some() && state.prior_producers.len() >= RETAINED_ROOT_CAPACITY {
            return Ok(Some(blocked("original producer retention capacity reached")));
        }
        if state.game_used_and_reserved()? >= state.game_record_limit {
            return Ok(Some(blocked("game record limit or retained reservations exhausted")));
        }
        match self.registration.try_disabled_root_count()? {
            Some(0) => {},
            Some(_) => return Ok(Some(blocked("original no-capture marker has not been settled"))),
            None => return Ok(None),
        }
        if let Some(previous) = &state.current_admission {
            let old = previous.root_admission().authority();
            let next = admission.root_admission().authority();
            check(!previous.same_original_admission(admission)
                && previous.native_origin().same_original_origin(admission.native_origin())
                && old.epoch == next.epoch && old.implementation == next.implementation
                && next.game == old.game && next.root > old.root,
                "lock-v2 next root does not follow the original collection owner")?;
        }
        Ok(Some(OwnerInstallReadinessV2::Ready))
    }
    pub(crate) fn same_original_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
            && Arc::ptr_eq(&self.game_gate, &other.game_gate)
            && self.registration.same_original_owner(&other.registration)
    }
    pub(crate) fn game_gate(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.game_gate)
    }
    pub fn new(
        manifest: rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2,
        metadata: RoventCollectionMetadata,
    ) -> Result<(Self, RoventNativeInputObserver), ArenaError> {
        Self::new_shared(Arc::new(manifest), metadata)
    }

    /// Install the same original full manifest owner used by the bridge and
    /// teacher session. Equal registration metadata alone is insufficient.
    pub fn new_shared(
        manifest: Arc<rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2>,
        metadata: RoventCollectionMetadata,
    ) -> Result<(Self, RoventNativeInputObserver), ArenaError> {
        Self::new_shared_with_record_limit(manifest, metadata, rz_experiments::rovent_corpus::MAX_CORPUS_ROWS as u64)
    }

    /// The bounded value is owned once by the game Collection owner and is
    /// shared by all of its root observers. Existing callers keep the legacy
    /// corpus maximum through `new_shared`.
    pub fn new_shared_with_record_limit(
        manifest: Arc<rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2>,
        metadata: RoventCollectionMetadata,
        maximum_records: u64,
    ) -> Result<(Self, RoventNativeInputObserver), ArenaError> {
        let game_record_limit = usize::try_from(maximum_records)
            .map_err(|_| invalid("lock-v2 game record limit exceeds platform capacity"))?;
        check(game_record_limit > 0 && game_record_limit <= rz_experiments::rovent_corpus::MAX_CORPUS_ROWS,
            "lock-v2 game record limit is invalid")?;
        check(
            generation_identifier(&metadata.game_id)
                && generation_identifier(&metadata.opening_id)
                && generation_identifier(&metadata.line_genealogy_id)
                && manifest
                    .registration
                    .opening_splits
                    .contains_key(&metadata.opening_id),
            "lock-v2 collection metadata/opening is not registered",
        )?;
        let (registration, observer, receiver) = rovent_native_capture_channel();
        let registration_for_owner = registration.clone();
        // This is a finite custody allowance, not a claim about host memory
        // usage. It cannot exceed the user's registered game record quota.
        let retired_outbox_capacity = game_record_limit.min(MAX_RETIRED_OUTBOX_CAPACITY);
        let (retired_sender, retired_receiver) = mpsc::sync_channel(retired_outbox_capacity);
        let retired_queued = Arc::new(AtomicUsize::new(0));
        Ok((
            Self {
                game_gate: Arc::new(AtomicU64::new(0)),
                state: Arc::new(Mutex::new(State {
                    game_record_limit,
                    retired_outbox_capacity,
                    game_retired_total: 0,
                    manifest,
                    producer: None,
                    producer_root: None,
                    prior_producers: Vec::new(),
                    current_admission: None,
                    settled_roots: Vec::new(),
                    registration,
                    receiver,
                    metadata,
                    position: None,
                    history: Vec::new(),
                    pending: BTreeMap::new(),
                    staged: BTreeMap::new(),
                    failed_staged: Vec::new(),
                    disabled: None,
                    prior_disabled: Vec::new(),
                    disabled_prior: BTreeMap::new(),
                    handled_tickets: BTreeMap::new(),
                    retired: VecDeque::new(),
                    retired_sender,
                    retired_queued: Arc::clone(&retired_queued),
                    admitted: false,
                    settled: false,
                    failure: None,
                    failed_captures: Vec::new(),
                    #[cfg(feature = "test-fixtures")]
                    test_completed: BTreeMap::new(),
                    #[cfg(feature = "test-fixtures")]
                    test_started: BTreeMap::new(),
                })),
                callback_lock_failure: Arc::new(AtomicBool::new(false)),
            registration: registration_for_owner,
                retired_receiver: Arc::new(Mutex::new(retired_receiver)),
                retired_queued,
            },
            observer,
        ))
    }

    pub fn observer<Q: 'static, O: ProposerRankingOutput + 'static>(
        &self,
    ) -> Box<dyn AdaptiveCollectionObserverV2<Q, O>> {
        Box::new(Observer {
            owner: self.clone(),
            marker: PhantomData,
        })
    }

    /// Installs no-capture only on the actual native observer's registration
    /// owner. It does not touch the failed producer or its retained inputs.
    pub(crate) fn try_install_whole_root_no_capture(
        &self,
        capability: &rz_uci::pals_adaptive::RoventNoCaptureRootCapabilityV2,
    ) -> Result<bool, ArenaError> {
        self.registration.try_install_whole_root_no_capture(capability)
    }

    /// Only the same original UCI root-run owner supplies the receipt here.
    pub(crate) fn try_finish_whole_root_no_capture(
        &self,
        capability: &rz_uci::pals_adaptive::RoventNoCaptureRootCapabilityV2,
        receipt: &rz_uci::pals_adaptive::AdaptiveShutdownReceipt,
    ) -> Result<bool, ArenaError> {
        self.registration.try_finish_whole_root_no_capture(capability, receipt)
    }

    pub(crate) fn try_finish_whole_root_no_capture_role_only(
        &self,
        capability: &rz_uci::pals_adaptive::RoventNoCaptureRootCapabilityV2,
        receipt: &rz_uci::pals_adaptive::RoleOnlyShutdownReceipt,
    ) -> Result<bool, ArenaError> {
        self.registration.try_finish_whole_root_no_capture_role_only(capability, receipt)
    }

    /// Exercise the real registration mutex without holding the manager
    /// state mutex. The test controls when the closure returns and releases it.
    #[cfg(feature = "test-fixtures")]
    pub fn test_with_capture_registration_held<T>(
        &self,
        action: impl FnOnce() -> T,
    ) -> Result<T, ArenaError> {
        let registration = self
            .state
            .lock()
            .map_err(|_| invalid("lock-v2 collection owner poisoned"))?
            .registration
            .clone();
        registration.test_with_registration_held(action)
    }

    /// Observes a staged input without creating a witness or advancing it.
    #[cfg(feature = "test-fixtures")]
    pub fn test_staged_sequence(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<Option<u64>, ArenaError> {
        let state = self
            .state
            .lock()
            .map_err(|_| invalid("lock-v2 collection owner poisoned"))?;
        match state.staged.get(&ticket.id) {
            Some(staged) if staged.ticket == *ticket => Ok(Some(staged.reservation.sequence())),
            Some(_) => Err(invalid("test ticket differs from original staged ticket")),
            None => Ok(None),
        }
    }

    #[cfg(feature = "test-fixtures")]
    pub fn test_staged_snapshot(&self) -> Result<Vec<(RoleTicketEnvelope, u64)>, ArenaError> {
        let state = self
            .state
            .lock()
            .map_err(|_| invalid("lock-v2 collection owner poisoned"))?;
        Ok(state
            .staged
            .values()
            .map(|input| (input.ticket, input.reservation.sequence()))
            .collect())
    }

    /// Read the actual native completion callback observed for this ticket.
    #[cfg(feature = "test-fixtures")]
    pub fn test_completion(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<Option<(u64, bool, bool)>, ArenaError> {
        let state = self
            .state
            .lock()
            .map_err(|_| invalid("lock-v2 collection owner poisoned"))?;
        match state.test_completed.get(&ticket.id) {
            Some((original, sequence, physical, successful)) if original == ticket => {
                Ok(Some((*sequence, *physical, *successful)))
            }
            Some(_) => Err(invalid("test completion ticket differs from original")),
            None => Ok(None),
        }
    }

    #[cfg(feature = "test-fixtures")]
    pub fn test_native_started(&self, ticket: &RoleTicketEnvelope) -> Result<bool, ArenaError> {
        let state = self
            .state
            .lock()
            .map_err(|_| invalid("lock-v2 collection owner poisoned"))?;
        match state.test_started.get(&ticket.id) {
            Some(original) if original == ticket => Ok(true),
            Some(_) => Err(invalid("test started ticket differs from original")),
            None => Ok(false),
        }
    }

    pub fn with_producer<T>(
        &self,
        action: impl FnOnce(&mut RoventCorpusProducerLockV2) -> Result<T, ArenaError>,
    ) -> Result<T, ArenaError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid("lock-v2 collection owner poisoned"))?;
        action(
            state
                .producer
                .as_mut()
                .ok_or_else(|| invalid("lock-v2 root not yet admitted"))?,
        )
    }

    /// Targets an already retired row's original producer; no input receipt or
    /// stored digest is promoted into a new live root capability.
    pub fn with_retired_producer<T>(
        &self,
        row: &RoventRetiredLockV2Record,
        action: impl FnOnce(&mut RoventCorpusProducerLockV2) -> Result<T, ArenaError>,
    ) -> Result<T, ArenaError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid("lock-v2 collection owner poisoned"))?;
        let root = row.admission.raw_sha256();
        if state.producer_root.as_deref() == Some(root) {
            let producer = state
                .producer
                .as_mut()
                .ok_or_else(|| invalid("lock-v2 active producer missing"))?;
            producer.validate_admission(&row.admission)?;
            return action(producer);
        }
        let (_, producer) = state
            .prior_producers
            .iter_mut()
            .find(|(id, _)| id == root)
            .ok_or_else(|| invalid("lock-v2 retired row's original producer missing"))?;
        producer.validate_admission(&row.admission)?;
        action(producer)
    }

    /// Mutate a row only through the producer that issued its original root.
    /// On error the caller still owns the same row and all its evidence.
    pub fn with_retired_producer_mut<T>(
        &self,
        row: &mut RoventRetiredLockV2Record,
        action: impl FnOnce(
            &mut RoventCorpusProducerLockV2,
            &mut RoventRetiredLockV2Record,
        ) -> Result<T, ArenaError>,
    ) -> Result<T, ArenaError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid("lock-v2 collection owner poisoned"))?;
        let root = row.admission.raw_sha256();
        if state.producer_root.as_deref() == Some(root) {
            let producer = state
                .producer
                .as_mut()
                .ok_or_else(|| invalid("lock-v2 active producer missing"))?;
            producer.validate_admission(&row.admission)?;
            return action(producer, row);
        }
        let (_, producer) = state
            .prior_producers
            .iter_mut()
            .find(|(id, _)| id == root)
            .ok_or_else(|| invalid("lock-v2 retired row's original producer missing"))?;
        producer.validate_admission(&row.admission)?;
        action(producer, row)
    }

    /// A short owner-only commit after the actual Own root handoff. The gate
    /// prevents a competing observer admission while this producer is held.
    /// Busy and all validation failures leave `row` and the caller's prepared
    /// evidence untouched. This does not publish the row or prove teacher
    /// retirement; those are separate owner obligations.
    pub(crate) fn try_with_fenced_own_producer_mut<T>(
        &self,
        row: &mut RoventRetiredLockV2Record,
        gate: &super::collection_hooks_core_v2::RoventCollectionCommitGateV2<
            '_, rz_uci::rovent_bootstrap::RoventNnuecCollectionHandoffV2,
        >,
        handoff: &mut rz_uci::rovent_bootstrap::RoventNnuecCollectionHandoffV2,
        action: impl FnOnce(&mut RoventCorpusProducerLockV2, &mut RoventRetiredLockV2Record) -> T,
    ) -> Result<Option<T>, ArenaError> {
        let (admission, _, settlement, fence) = handoff.parts_mut();
        check(self.same_original_owner(gate.owner()) && gate.matches_admission(admission)
            && row.admission.same_original_admission(admission)
            && settlement.is_ok() && fence.authority == admission.root_admission().authority()
            && fence.permits_semantic_closure(),
            "lock-v2 commit gate/original handoff/physical fence mismatch")?;
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("lock-v2 collection owner poisoned")),
        };
        check(gate.matches_admission(admission)
            && state.settled_roots.iter().any(|original| original.same_original_admission(admission))
            && (!state.admitted || state.settled) && state.pending.is_empty() && state.staged.is_empty(),
            "lock-v2 original root not settled or observer callbacks still active")?;
        let root = row.admission.raw_sha256();
        let producer = if state.producer_root.as_deref() == Some(root) {
            state.producer.as_mut().ok_or_else(|| invalid("lock-v2 active producer missing"))?
        } else {
            &mut state.prior_producers.iter_mut().find(|(id, _)| id == root)
                .ok_or_else(|| invalid("lock-v2 retired row original producer missing"))?.1
        };
        producer.validate_admission(admission)?;
        producer.validate_teacher_row(row)?;
        Ok(Some(action(producer, row)))
    }

    /// P-C counterpart: authority comes from its original RoleOnly handoff,
    /// and the same commit gate excludes a competing root admission.
    pub(crate) fn try_with_fenced_pc_producer_mut<T>(
        &self,
        row: &mut RoventRetiredLockV2Record,
        gate: &super::collection_hooks_core_v2::RoventCollectionCommitGateV2<
            '_, rz_uci::rovent_bootstrap::pc_session::RoventPcCollectionHandoffV2,
        >,
        handoff: &mut rz_uci::rovent_bootstrap::pc_session::RoventPcCollectionHandoffV2,
        action: impl FnOnce(&mut RoventCorpusProducerLockV2, &mut RoventRetiredLockV2Record) -> T,
    ) -> Result<Option<T>, ArenaError> {
        let (admission, _, settlement, fence) = handoff.parts_mut();
        check(self.same_original_owner(gate.owner()) && gate.matches_admission(admission)
            && row.admission.same_original_admission(admission)
            && settlement.is_ok() && fence.authority == admission.root_admission().authority()
            && fence.root_budget == admission.root_budget() && fence.permits_semantic_closure(),
            "lock-v2 P/C commit gate/original handoff/physical fence mismatch")?;
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("lock-v2 collection owner poisoned")),
        };
        check(gate.matches_admission(admission)
            && state.settled_roots.iter().any(|original| original.same_original_admission(admission))
            && (!state.admitted || state.settled) && state.pending.is_empty() && state.staged.is_empty(),
            "lock-v2 P/C original root not settled or observer callbacks still active")?;
        let root = row.admission.raw_sha256();
        let producer = if state.producer_root.as_deref() == Some(root) {
            state.producer.as_mut().ok_or_else(|| invalid("lock-v2 active producer missing"))?
        } else {
            &mut state.prior_producers.iter_mut().find(|(id, _)| id == root)
                .ok_or_else(|| invalid("lock-v2 retired row original producer missing"))?.1
        };
        producer.validate_admission(admission)?;
        producer.validate_teacher_row(row)?;
        Ok(Some(action(producer, row)))
    }

    pub fn take_retired(&self) -> Result<Option<RoventRetiredLockV2Record>, ArenaError> {
        let receiver = match self.retired_receiver.try_lock() {
            Ok(receiver) => receiver,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid("lock-v2 retired receiver poisoned"));
            }
        };
        if let Ok(row) = receiver.try_recv() {
            self.retired_queued.fetch_sub(1, Ordering::AcqRel);
            return Ok(Some(row));
        }
        Ok(None)
    }

    /// Distinguish contention from a genuinely empty, settled root outbox.
    /// The original row remains with its owner on Busy or any error.
    pub(crate) fn try_take_settled_retired(
        &self,
    ) -> Result<Option<Option<RoventRetiredLockV2Record>>, ArenaError> {
        let receiver = match self.retired_receiver.try_lock() {
            Ok(receiver) => receiver,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("lock-v2 retired receiver poisoned")),
        };
        if let Ok(row) = receiver.try_recv() {
            self.retired_queued.fetch_sub(1, Ordering::AcqRel);
            return Ok(Some(Some(row)));
        }
        if self.retired_queued.load(Ordering::Acquire) != 0 { return Ok(None); }
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("lock-v2 collection owner poisoned")),
        };
        check(state.settled, "lock-v2 retained recovery requires a settled root")?;
        Ok(Some(state.retired.pop_front()))
    }

    /// Exceptional send failures retain their original row in the manager.
    /// This recovery is only available after settlement; the caller must also
    /// exclude a concurrent next-root admission while it holds the owner.
    pub fn take_retained_after_fence(
        &self,
    ) -> Result<Option<RoventRetiredLockV2Record>, ArenaError> {
        if self.retired_queued.load(Ordering::Acquire) != 0 {
            return Err(invalid("lock-v2 queued retired rows must be drained first"));
        }
        match self.state.try_lock() {
            Ok(mut state) if state.settled => Ok(state.retired.pop_front()),
            Ok(_) => Err(invalid("lock-v2 retained recovery requires a settled root")),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                Err(invalid("lock-v2 collection owner poisoned"))
            }
        }
    }

    pub fn failure(&self) -> Result<Option<AdaptiveCollectionError>, ArenaError> {
        Ok(self
            .state
            .lock()
            .map_err(|_| invalid("lock-v2 collection owner poisoned"))?
            .failure
            .clone())
    }

    /// Checked before launch while the original owner is idle. A busy or
    /// poisoned observation does not create a collection failure or grant
    /// authority to substitute another manifest.
    pub(crate) fn matches_manifest(
        &self,
        manifest: &Arc<rz_experiments::rovent_corpus_lock_v2::record::RoventCorpusManifestLockV2>,
    ) -> Result<bool, ArenaError> {
        let state = self
            .state
            .try_lock()
            .map_err(|_| invalid("lock-v2 owner manifest unavailable"))?;
        Ok(Arc::ptr_eq(&state.manifest, manifest))
    }

    /// A local collection-status observation, never a teacher retirement,
    /// physical fence, corpus publication or trainer admission certificate.
    pub fn status(&self) -> Result<RoventCollectionOwnerStatusLockV2, ArenaError> {
        let state = self
            .state
            .lock()
            .map_err(|_| invalid("lock-v2 collection owner poisoned"))?;
        Ok(self.status_from(&state))
    }

    /// Contention is an unavailable snapshot, not an observation failure.
    /// Poison is an error and neither case creates physical-completion proof.
    pub fn try_status(&self) -> Result<Option<RoventCollectionOwnerStatusLockV2>, ArenaError> {
        match self.state.try_lock() {
            Ok(state) => Ok(Some(self.status_from(&state))),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                Err(invalid("lock-v2 collection owner poisoned"))
            }
        }
    }

    fn status_from(&self, state: &State) -> RoventCollectionOwnerStatusLockV2 {
        RoventCollectionOwnerStatusLockV2 {
            admitted: state.admitted,
            settled: state.settled,
            game_record_limit: state.game_record_limit,
            retired_outbox_capacity: state.retired_outbox_capacity,
            game_retired_total: state.game_retired_total,
            game_reserved_slots: state.game_reserved_slots().ok(),
            pending_count: state.pending.len() + state.staged.len(),
            pending_retired_count: state.retired.len()
                + self.retired_queued.load(Ordering::Acquire),
            pending_disabled_markers: self.registration.try_disabled_root_count().ok().flatten(),
            collection_disabled_first_ticket: state
                .disabled
                .as_ref()
                .map(|disabled| disabled.first_ticket),
            collection_disabled_reason: state.disabled.as_ref().map(|disabled| disabled.reason),
            collection_disabled_failure: state
                .disabled
                .as_ref()
                .and_then(|disabled| disabled.original_failure.clone()),
            collection_disabled_prior_tickets: state
                .disabled
                .as_ref()
                .map_or_else(Vec::new, |disabled| disabled.prior_tickets.clone()),
            prior_disabled_first_tickets: state
                .prior_disabled
                .iter()
                .map(|disabled| disabled.first_ticket)
                .collect(),
            prior_disabled_reasons: state
                .prior_disabled
                .iter()
                .map(|disabled| disabled.reason)
                .collect(),
            prior_disabled_failures: state
                .prior_disabled
                .iter()
                .map(|disabled| disabled.original_failure.clone())
                .collect(),
            prior_disabled_prior_tickets: state
                .prior_disabled
                .iter()
                .map(|disabled| disabled.prior_tickets.clone())
                .collect(),
            incomplete_capture_count: state
                .producer
                .as_ref()
                .map_or(0, |producer| producer.incomplete_capture_count()),
            retained_prior_roots: state.prior_producers.len(),
            retained_prior_root_limit: RETAINED_ROOT_CAPACITY,
            retained_failed_captures: state.failed_captures.len(),
            retained_failed_staged: state.failed_staged.len(),
            failure: state.failure.clone(),
            callback_lock_failure: self.callback_lock_failure.load(Ordering::Acquire),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RoventCollectionOwnerStatusLockV2 {
    pub admitted: bool,
    pub settled: bool,
    /// Game-owner cap and monotonic retired count; not a publication count.
    pub game_record_limit: usize,
    /// Maximum original pending/staged/retired rows, not allocated bytes.
    pub retired_outbox_capacity: usize,
    pub game_retired_total: usize,
    /// Unknown only on arithmetic overflow; never infer zero in that case.
    pub game_reserved_slots: Option<usize>,
    pub pending_count: usize,
    pub pending_retired_count: usize,
    /// Unknown on contention or poison; neither permits owner release.
    pub pending_disabled_markers: Option<usize>,
    /// A disabled root is incomplete, from this original ticket onward.
    pub collection_disabled_first_ticket: Option<RoleTicketEnvelope>,
    pub collection_disabled_reason: Option<RoventCollectionDisableReasonV2>,
    pub collection_disabled_failure: Option<AdaptiveCollectionError>,
    pub collection_disabled_prior_tickets: Vec<RoleTicketEnvelope>,
    pub prior_disabled_first_tickets: Vec<RoleTicketEnvelope>,
    pub prior_disabled_reasons: Vec<RoventCollectionDisableReasonV2>,
    pub prior_disabled_failures: Vec<Option<AdaptiveCollectionError>>,
    pub prior_disabled_prior_tickets: Vec<Vec<RoleTicketEnvelope>>,
    pub incomplete_capture_count: usize,
    /// Root producers are held for caller-owned retired rows. No automatic
    /// release is safe until all row and target ownership can be proven.
    pub retained_prior_roots: usize,
    pub retained_prior_root_limit: usize,
    pub retained_failed_captures: usize,
    pub retained_failed_staged: usize,
    pub failure: Option<AdaptiveCollectionError>,
    pub callback_lock_failure: bool,
}

struct Observer<Q, O> {
    owner: RoventCollectionOwnerLockV2,
    marker: PhantomData<fn(Q, O)>,
}

impl State {
    /// Pending and staged originals each reserve at most one future retired
    /// row. Failed staged inputs keep their reservation until explicit
    /// recovery; a row already retired is counted monotonically instead.
    fn game_reserved_slots(&self) -> Result<usize, ArenaError> {
        self.pending.len().checked_add(self.staged.len())
            .and_then(|value| value.checked_add(self.failed_staged.len()))
            .ok_or_else(|| invalid("lock-v2 game record reservation overflow"))
    }

    fn game_used_and_reserved(&self) -> Result<usize, ArenaError> {
        self.game_retired_total.checked_add(self.game_reserved_slots()?)
            .ok_or_else(|| invalid("lock-v2 game record total overflow"))
    }

    fn next_retired_record_count(&self) -> Result<usize, ArenaError> {
        let next = self.game_retired_total.checked_add(1)
            .ok_or_else(|| invalid("lock-v2 game retired record overflow"))?;
        check(next <= self.game_record_limit, "lock-v2 game retired record limit exceeded")?;
        Ok(next)
    }
    fn is_disabled_ticket(&self, ticket: &RoleTicketEnvelope) -> Result<bool, ArenaError> {
        if self.pending.contains_key(&ticket.id)
            || self.staged.contains_key(&ticket.id)
            || self.handled_tickets.contains_key(&ticket.id)
        {
            return Ok(false);
        }
        let Some(disabled) = &self.disabled else {
            return Ok(false);
        };
        if let Some(original) = self.disabled_prior.get(&ticket.id) {
            check(original == ticket, "disabled prior ticket identity changed")?;
            self.producer()?.validate_admission(&disabled.admission)?;
            return Ok(true);
        }
        check(
            ticket.id.epoch == disabled.first_ticket.id.epoch
                && ticket.id.sequence >= disabled.first_ticket.id.sequence
                && ticket.authority == disabled.first_ticket.authority
                && ticket.original_deadline == disabled.first_ticket.original_deadline
                && ticket.view.root_state == disabled.first_ticket.view.root_state,
            "missing collection ticket predates disabled boundary",
        )?;
        self.producer()?.validate_admission(&disabled.admission)?;
        Ok(true)
    }
    fn outstanding_rows(&self) -> usize {
        self.retired.len()
            + self.retired_queued.load(Ordering::Acquire)
            + self.pending.len()
            + self.staged.len()
    }
    fn retain_row(&mut self, row: RoventRetiredLockV2Record) {
        // Once an overflow exists, later rows must not overtake it.
        if !self.retired.is_empty() {
            self.retired.push_back(row);
            return;
        }
        self.retired_queued.fetch_add(1, Ordering::AcqRel);
        if let Err(error) = self.retired_sender.try_send(row) {
            self.retired_queued.fetch_sub(1, Ordering::AcqRel);
            // A full or disconnected channel never destroys original evidence.
            let row = match error {
                std::sync::mpsc::TrySendError::Full(row)
                | std::sync::mpsc::TrySendError::Disconnected(row) => row,
            };
            self.retired.push_back(row);
            self.failure.get_or_insert_with(|| {
                AdaptiveCollectionError::new(
                    AdaptiveCollectionStage::Accepted,
                    "RetiredOutboxUnavailable",
                    "original retired row retained after unexpected outbox failure",
                )
            });
        }
    }
    fn producer(&self) -> Result<&RoventCorpusProducerLockV2, ArenaError> {
        self.producer
            .as_ref()
            .ok_or_else(|| invalid("lock-v2 root not yet admitted"))
    }
    fn producer_mut(&mut self) -> Result<&mut RoventCorpusProducerLockV2, ArenaError> {
        self.producer
            .as_mut()
            .ok_or_else(|| invalid("lock-v2 root not yet admitted"))
    }
    fn drain(&mut self) -> Result<(), ArenaError> {
        let producer = self
            .producer
            .as_mut()
            .ok_or_else(|| invalid("lock-v2 root not yet admitted"))?;
        match self.receiver.admit_available_lock_v2(producer) {
            Ok(_) => Ok(()),
            Err(failure) => {
                let (error, input) = *failure;
                if let Some(input) = input {
                    self.failed_captures.push(
                        super::native_capture::RoventFailedCapturedInput::LockV2(input),
                    );
                }
                Err(error)
            }
        }
    }
    fn discard(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), ArenaError> {
        if let Some(staged) = self.staged.get(&ticket.id) {
            check(
                staged.ticket == *ticket,
                "lock-v2 staged ticket changed at settlement",
            )?;
            let staged = self
                .staged
                .remove(&ticket.id)
                .expect("checked staged ticket");
            self.producer_mut()?
                .discard_reserved_input(staged.reservation.sequence());
            self.handled_tickets.insert(ticket.id, *ticket);
            return Ok(());
        }
        if self.is_disabled_ticket(ticket)? {
            if self.disabled_prior.contains_key(&ticket.id) {
                self.registration.cancel_sequence(ticket.id)?;
            }
            return Ok(());
        }
        if let Some(handled) = self.handled_tickets.get(&ticket.id) {
            return check(
                handled == ticket,
                "lock-v2 handled ticket changed at settlement",
            );
        }
        let pending = self
            .pending
            .get(&ticket.id)
            .ok_or_else(|| invalid("lock-v2 ticket not registered"))?;
        check(
            pending.ticket == *ticket,
            "lock-v2 ticket differs from original dispatch",
        )?;
        if let Some(sequence) = self.registration.cancel_sequence(ticket.id)? {
            self.producer_mut()?.discard_reserved_input(sequence);
        }
        if let Some(pending) = self.pending.remove(&ticket.id) {
            self.producer_mut()?
                .discard_reserved_input(pending.sequence);
        }
        self.producer_mut()?.discard_unpublished(ticket.id);
        self.handled_tickets.insert(ticket.id, *ticket);
        Ok(())
    }
}

impl<Q, O> Observer<Q, O> {
    fn apply_dispatch(
        &self,
        f: impl FnOnce(&mut State) -> Result<AdaptiveCollectionDispatchV2, ArenaError>,
    ) -> Result<AdaptiveCollectionDispatchV2, AdaptiveCollectionError> {
        let mut state = match self.owner.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => {
                return Ok(AdaptiveCollectionDispatchV2::Busy);
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(AdaptiveCollectionError::new(
                    AdaptiveCollectionStage::Dispatch,
                    "RoventCollectionUnavailable",
                    "lock-v2 collection owner poisoned",
                ));
            }
        };
        match f(&mut state) {
            Ok(result) => Ok(result),
            Err(cause) => {
                let error = AdaptiveCollectionError::new(
                    AdaptiveCollectionStage::Dispatch,
                    "RoventCollectionEvidence",
                    cause.to_string(),
                );
                state.failure.get_or_insert_with(|| error.clone());
                Err(error)
            }
        }
    }
    fn apply<T>(
        &self,
        stage: AdaptiveCollectionStage,
        f: impl FnOnce(&mut State) -> Result<T, ArenaError>,
    ) -> Result<T, AdaptiveCollectionError> {
        let mut state = self.owner.state.try_lock().map_err(|_| {
            self.owner
                .callback_lock_failure
                .store(true, Ordering::Release);
            AdaptiveCollectionError::new(
                stage,
                "RoventCollectionUnavailable",
                "lock-v2 collection owner busy or poisoned",
            )
        })?;
        match f(&mut state) {
            Ok(result) => Ok(result),
            Err(error) => {
                let error = AdaptiveCollectionError::new(
                    stage,
                    "RoventCollectionEvidence",
                    error.to_string(),
                );
                state.failure.get_or_insert_with(|| error.clone());
                Err(error)
            }
        }
    }
}

impl<Q, O: ProposerRankingOutput> AdaptiveCollectionObserverV2<Q, O> for Observer<Q, O> {
    fn admitted(
        &mut self,
        admission: &RoventCollectionRootAdmissionV2,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Dispatch, |state| {
            check(
                !self.owner.callback_lock_failure.load(Ordering::Acquire)
                    && (!state.admitted || state.settled)
                    && state.pending.is_empty()
                    && state.staged.is_empty()
                    && state.disabled_prior.is_empty()
                    && state.failure.is_none()
                    && state
                        .producer
                        .as_ref()
                        .is_none_or(|producer| producer.incomplete_capture_count() == 0)
                    && (state.producer.is_none()
                        || state.prior_producers.len() < RETAINED_ROOT_CAPACITY),
                "lock-v2 prior root is not settled or producer retention is full",
            )?;
            let snapshot = admission.rules().rules().snapshot();
            let history = snapshot
                .uci_replay(16_384)
                .map_err(|error| invalid(error.to_string()))?
                .moves;
            let position = Position::from_snapshot(
                snapshot,
                rz_position::PositionLimits {
                    max_history_positions: 16_385,
                    ..rz_position::PositionLimits::default()
                },
            )
            .map_err(|error| invalid(error.to_string()))?;
            let producer = RoventCorpusProducerLockV2::new_shared(
                Arc::clone(&state.manifest),
                admission.clone(),
            )?;
            if state.producer.is_some() {
                let id = state
                    .producer_root
                    .as_ref()
                    .ok_or_else(|| invalid("lock-v2 previous producer identity missing"))?
                    .clone();
                check(
                    id != admission.raw_sha256()
                        && !state
                            .prior_producers
                            .iter()
                            .any(|(old, _)| old == admission.raw_sha256()),
                    "lock-v2 new root duplicates a retained producer",
                )?;
                let previous = state.producer.take().expect("checked previous producer");
                state.producer_root = None;
                state.prior_producers.push((id, previous));
            }
            state.producer = Some(producer);
            state.producer_root = Some(admission.raw_sha256().to_owned());
            state.current_admission = Some(admission.clone());
            state.handled_tickets.clear();
            if let Some(disabled) = state.disabled.take() {
                state.prior_disabled.push(disabled);
            }
            state.position = Some(position);
            state.history = history;
            state.admitted = true;
            state.settled = false;
            Ok(())
        })
    }

    fn dispatch(
        &mut self,
        witness: RoventCollectionDispatchWitnessV2<'_>,
        input: &FrameInput<ReviewView, Q>,
    ) -> Result<AdaptiveCollectionDispatchV2, AdaptiveCollectionError> {
        self.apply_dispatch(|state| {
            let ticket = witness.ticket();
            state.producer()?.validate_admission(witness.admission())?;
            check(
                state.admitted
                    && !state.settled
                    && state.failure.is_none()
                    && !state.pending.contains_key(&ticket.id)
                    && !state.handled_tickets.contains_key(&ticket.id)
                    && input.view_seal == ticket.view
                    && input.question_digest == ticket.question,
                "lock-v2 collection dispatch scope",
            )?;
            if let Some(staged) = state.staged.get(&ticket.id) {
                check(
                    staged.ticket == *ticket
                        && staged
                            .admission
                            .same_original_admission(witness.admission()),
                    "lock-v2 staged retry belongs to another ticket or root",
                )?;
            }
            if state.is_disabled_ticket(ticket)? {
                return Ok(AdaptiveCollectionDispatchV2::DisabledBackpressure);
            }
            let game_limit_reached = state.game_retired_total >= state.game_record_limit;
            if !state.staged.contains_key(&ticket.id)
                && !game_limit_reached
                && state.game_used_and_reserved()? >= state.game_record_limit {
                // Existing reservations can still retire or be cancelled;
                // a transient full reservation is not a permanent boundary.
                return Ok(AdaptiveCollectionDispatchV2::Busy);
            }
            if !state.staged.contains_key(&ticket.id)
                && (game_limit_reached || state.outstanding_rows() >= state.retired_outbox_capacity
                    || state.handled_tickets.len() + state.pending.len()
                        >= rz_experiments::rovent_corpus::MAX_CORPUS_ROWS)
            {
                // A different ticket's staged reservation must be registered
                // or canceled before disabling the root collection boundary.
                if !state.staged.is_empty() {
                    return Ok(AdaptiveCollectionDispatchV2::Busy);
                }
                let reason = if game_limit_reached {
                    RoventCollectionDisableReasonV2::GameRecordLimit
                } else if state.outstanding_rows() >= state.retired_outbox_capacity {
                    RoventCollectionDisableReasonV2::RetiredOutboxFull
                } else {
                    RoventCollectionDisableReasonV2::TicketEvidenceLimit
                };
                if !state
                    .registration
                    .try_disable_lock_v2_root(witness.admission(), ticket)?
                {
                    return Ok(AdaptiveCollectionDispatchV2::Busy);
                }
                state.disabled = Some(DisabledRoot {
                    admission: witness.admission().clone(),
                    first_ticket: *ticket,
                    reason,
                    original_failure: None,
                    prior_tickets: Vec::new(),
                });
                return Ok(AdaptiveCollectionDispatchV2::DisabledBackpressure);
            }
            if !state.staged.contains_key(&ticket.id) {
                let registration = &state.manifest.registration;
                let context = GenerationCaptureContextV3 {
                    game_id: state.metadata.game_id.clone(),
                    opening_id: state.metadata.opening_id.clone(),
                    line_genealogy_id: state.metadata.line_genealogy_id.clone(),
                    model_epoch: registration.model_epoch,
                    context_epoch: registration.context_epoch,
                    capture_sequence: 0,
                    work_scope: GenerationCaptureWorkScopeV3::from_native_ticket(ticket),
                    actual_position: state
                        .position
                        .as_ref()
                        .ok_or_else(|| invalid("lock-v2 Rules root missing"))?
                        .clone(),
                    actual_history: state.history.clone(),
                    action_support: GenerationActionSupportV3::non_validator(),
                };
                let lineage = state.metadata.lineage.clone();
                let Some(reservation) = state
                    .producer_mut()?
                    .try_reserve_input(ticket, context, lineage)?
                else {
                    return Ok(AdaptiveCollectionDispatchV2::Busy);
                };
                state.staged.insert(
                    ticket.id,
                    StagedInput {
                        ticket: *ticket,
                        admission: witness.admission().clone(),
                        reservation,
                    },
                );
            }
            let staged = state
                .staged
                .remove(&ticket.id)
                .expect("staged original input exists");
            let sequence = staged.reservation.sequence();
            match state.registration.try_register_lock_v2(
                ticket,
                staged.reservation,
                witness.admission(),
            ) {
                super::native_capture::LockV2RegistrationAttempt::Ready => {
                    state.pending.insert(
                        ticket.id,
                        Pending {
                            ticket: *ticket,
                            sequence,
                            accepted: false,
                        },
                    );
                    Ok(AdaptiveCollectionDispatchV2::Captured)
                }
                super::native_capture::LockV2RegistrationAttempt::Busy(reservation) => {
                    state.staged.insert(
                        ticket.id,
                        StagedInput {
                            reservation,
                            ..staged
                        },
                    );
                    Ok(AdaptiveCollectionDispatchV2::Busy)
                }
                super::native_capture::LockV2RegistrationAttempt::Failed(error, reservation) => {
                    // Registration has not committed. Keep the actual failed
                    // reservation separately, so disabling this root cannot
                    // reclassify or discard its original collection evidence.
                    state.failed_staged.push(StagedInput {
                        reservation,
                        ..staged
                    });
                    Err(error)
                }
            }
        })
    }

    fn disable_after_failure(
        &mut self,
        witness: RoventCollectionDispatchWitnessV2<'_>,
        original: &AdaptiveCollectionError,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        // This recovery intentionally does not use `apply`: the original
        // failure is already sticky, but its evidence must not prevent a
        // separate, verified decision to stop optional collection.
        let error = |detail: String| {
            AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Dispatch,
                "RoventDisableAfterFailure",
                detail,
            )
        };
        let mut state = match self.owner.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => {
                return Ok(AdaptiveCollectionBoundaryV2::Busy);
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(error("original collection owner poisoned".into()));
            }
        };
        let ticket = *witness.ticket();
        let admission = witness.admission();
        state
            .producer()
            .and_then(|producer| producer.validate_admission(admission))
            .map_err(|cause| error(cause.to_string()))?;
        if let Err(cause) = admission.try_validate_ticket(&ticket) {
            return if cause.kind == RoventTicketValidationKindV2::Busy {
                Ok(AdaptiveCollectionBoundaryV2::Busy)
            } else {
                Err(cause.error)
            };
        }
        if !state.admitted || state.settled {
            return Err(error("collection root is not active".into()));
        }
        if !state.staged.is_empty() {
            return Err(error(
                "a staged original input must be retired before disabling collection".into(),
            ));
        }
        if let Some(disabled) = state.disabled.as_mut() {
            if !disabled.admission.same_original_admission(admission)
                || ticket.id.epoch != disabled.first_ticket.id.epoch
                || ticket.id.sequence < disabled.first_ticket.id.sequence
            {
                return Err(error(
                    "existing disabled boundary differs from original root".into(),
                ));
            }
            if disabled.original_failure.is_none() {
                disabled.original_failure = Some(original.clone());
            }
            return Ok(AdaptiveCollectionBoundaryV2::Ready);
        }
        // A pre-existing registration may already own a submitted input. Do
        // not reclassify that ticket as the first omitted capture.
        if state.pending.contains_key(&ticket.id)
            || state.staged.contains_key(&ticket.id)
            || state.handled_tickets.contains_key(&ticket.id)
        {
            return Err(error(
                "failure boundary already owns a collection input".into(),
            ));
        }
        match state
            .registration
            .try_has_registration(ticket.id)
            .map_err(|cause| error(cause.to_string()))?
        {
            Some(true) => {
                return Err(error(
                    "failure boundary already owns a collection input".into(),
                ));
            }
            Some(false) => {}
            None => return Ok(AdaptiveCollectionBoundaryV2::Busy),
        }
        if !state
            .registration
            .try_disable_lock_v2_root(admission, &ticket)
            .map_err(|cause| error(cause.to_string()))?
        {
            return Ok(AdaptiveCollectionBoundaryV2::Busy);
        }
        state.disabled = Some(DisabledRoot {
            admission: admission.clone(),
            first_ticket: ticket,
            reason: RoventCollectionDisableReasonV2::ObserverFailure,
            original_failure: Some(original.clone()),
            prior_tickets: Vec::new(),
        });
        Ok(AdaptiveCollectionBoundaryV2::Ready)
    }

    fn allow_disabled_prior_ticket(
        &mut self,
        witness: RoventCollectionDispatchWitnessV2<'_>,
    ) -> Result<AdaptiveCollectionBoundaryV2, AdaptiveCollectionError> {
        let error = |detail: String| {
            AdaptiveCollectionError::new(
                AdaptiveCollectionStage::Dispatch,
                "RoventDisabledPriorTicket",
                detail,
            )
        };
        let mut state = match self.owner.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => {
                return Ok(AdaptiveCollectionBoundaryV2::Busy);
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(error("original collection owner poisoned".into()));
            }
        };
        let ticket = *witness.ticket();
        let admission = witness.admission();
        let disabled = state
            .disabled
            .as_ref()
            .ok_or_else(|| error("collection root is not disabled".into()))?;
        if !state.admitted
            || state.settled
            || !disabled.admission.same_original_admission(admission)
            || ticket.id.epoch != disabled.first_ticket.id.epoch
            || ticket.id.sequence >= disabled.first_ticket.id.sequence
        {
            return Err(error(
                "prior ticket does not match disabled root boundary".into(),
            ));
        }
        if let Err(cause) = admission.try_validate_ticket(&ticket) {
            return if cause.kind == RoventTicketValidationKindV2::Busy {
                Ok(AdaptiveCollectionBoundaryV2::Busy)
            } else {
                Err(cause.error)
            };
        }
        if state.pending.contains_key(&ticket.id)
            || state.staged.contains_key(&ticket.id)
            || state.handled_tickets.contains_key(&ticket.id)
            || state.disabled_prior.contains_key(&ticket.id)
            || state.disabled_prior.len() >= CAPACITY
        {
            return Err(error(
                "prior ticket already owned or evidence capacity exhausted".into(),
            ));
        }
        if !state
            .registration
            .try_register_disabled_prior(&ticket, admission)
            .map_err(|cause| error(cause.to_string()))?
        {
            return Ok(AdaptiveCollectionBoundaryV2::Busy);
        }
        state.disabled_prior.insert(ticket.id, ticket);
        state
            .disabled
            .as_mut()
            .expect("disabled root checked")
            .prior_tickets
            .push(ticket);
        Ok(AdaptiveCollectionBoundaryV2::Ready)
    }

    fn native_event(
        &mut self,
        event: RoventCollectionNativeEventWitnessV2<'_, O>,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::NativeEvent, |state| {
            check(
                !state.settled,
                "lock-v2 native event after settled owner fence",
            )?;
            #[cfg(feature = "test-fixtures")]
            if let TicketNativeEvent::Started(ticket) = event.event() {
                if state.test_started.len() < CAPACITY {
                    state.test_started.insert(ticket.id, *ticket);
                }
            }
            #[cfg(feature = "test-fixtures")]
            let completion_for_test = match event.event() {
                TicketNativeEvent::Completed { completion, .. } => state
                    .pending
                    .get(&completion.ticket.id)
                    .filter(|pending| pending.ticket == completion.ticket)
                    .map(|pending| {
                        (
                            completion.ticket,
                            pending.sequence,
                            completion.physical
                                == rz_search::pals::coordinator::PhysicalRoleCompletion::Confirmed,
                            completion.result.is_ok(),
                        )
                    }),
                _ => None,
            };
            // Preserve the original terminal's typed observation failures even
            // if the capture queue is already unhealthy. No input is admitted
            // or published on this path; native completion remains independent.
            let failed_observation = match event.event() {
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
            if failed_observation {
                return state.producer_mut()?.observe_native_event(event);
            }
            state.drain()?;
            let event_ticket = match event.event() {
                TicketNativeEvent::Completed { completion, .. } => Some(completion.ticket),
                TicketNativeEvent::Retired { ticket, .. } => Some(*ticket),
                _ => None,
            };
            if let Some(ticket) = event_ticket {
                if state.is_disabled_ticket(&ticket)? {
                    state.producer()?.validate_admission(event.admission())?;
                    if let TicketNativeEvent::Retired { result, .. } = event.event() {
                        if let Err(error) = result {
                            return Err(invalid(format!(
                                "original native retirement failed: {error:?}"
                            )));
                        }
                    }
                    return Ok(());
                }
            }
            let failed = match event.event() {
                TicketNativeEvent::Retired {
                    accepted: false,
                    result: Err(error),
                    ..
                } => Some(format!("original native retirement failed: {error:?}")),
                _ => None,
            };
            state.producer_mut()?.observe_native_event(event)?;
            #[cfg(feature = "test-fixtures")]
            if let Some((ticket, sequence, physical, successful)) = completion_for_test {
                if state.test_completed.len() < CAPACITY {
                    state
                        .test_completed
                        .insert(ticket.id, (ticket, sequence, physical, successful));
                }
            }
            if let Some(error) = failed {
                return Err(invalid(error));
            }
            Ok(())
        })
    }

    fn accepted(
        &mut self,
        witness: RoventCollectionAcceptedWitnessV2<'_>,
        _: &FrameInput<ReviewView, Q>,
        _: &O,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Accepted, |state| {
            state.producer()?.validate_admission(witness.admission())?;
            let ticket = witness.accepted().ticket();
            if state.is_disabled_ticket(ticket)? {
                return Ok(());
            }
            let pending = state
                .pending
                .get_mut(&ticket.id)
                .ok_or_else(|| invalid("lock-v2 accepted ticket not registered"))?;
            check(
                pending.ticket == *ticket && !pending.accepted,
                "lock-v2 duplicate or changed acceptance",
            )?;
            pending.accepted = true;
            Ok(())
        })
    }

    fn consumed(
        &mut self,
        witness: RoventCollectionConsumedWitnessV2<'_>,
        _: RoleOnlySearchConsumptionV2,
    ) -> Result<(), AdaptiveCollectionError> {
        self.retire_consumed(witness)
    }

    fn adaptive_consumed(
        &mut self,
        witness: RoventCollectionConsumedWitnessV2<'_>,
        _: bool,
    ) -> Result<(), AdaptiveCollectionError> {
        self.retire_consumed(witness)
    }

    fn adaptive_root_ranking(
        &mut self,
        accepted: &RoventRootRankingAcceptanceV2,
        _: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Accepted, |state| {
            state.producer()?.validate_admission(accepted.admission())?;
            let ticket = &accepted.ranking().source().ticket;
            if state.is_disabled_ticket(ticket)? {
                return Ok(());
            }
            let pending = state
                .pending
                .get(&ticket.id)
                .ok_or_else(|| invalid("lock-v2 ranking ticket not registered"))?;
            check(
                pending.ticket == *ticket,
                "lock-v2 ranking original ticket mismatch",
            )?;
            let next_retired = state.next_retired_record_count()?;
            state.drain()?;
            let row = state
                .producer_mut()?
                .retire_root_ranking(accepted, output.proposer_ranking_output())?;
            state.game_retired_total = next_retired;
            state.pending.remove(&ticket.id);
            state.handled_tickets.insert(ticket.id, *ticket);
            state.retain_row(row);
            Ok(())
        })
    }

    fn abandoned(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Abandoned, |state| {
            state.drain()?;
            state.discard(ticket)
        })
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
                state
                    .failed_captures
                    .extend(state.receiver.take_queued_inputs_for_cleanup());
            } else {
                state.drain()?;
            }
            for ticket in tickets {
                state.discard(ticket)?;
            }
            let staged_tickets: Vec<_> =
                state.staged.values().map(|staged| staged.ticket).collect();
            for ticket in staged_tickets {
                state.discard(&ticket)?;
            }
            // After the native fence, release only producer-side reservations;
            // the original failed reservation objects remain evidence.
            let failed_sequences: Vec<_> = state
                .failed_staged
                .iter()
                .filter(|staged| {
                    state.producer().is_ok_and(|producer| {
                        producer.validate_admission(&staged.admission).is_ok()
                    })
                })
                .map(|staged| staged.reservation.sequence())
                .collect();
            for sequence in failed_sequences {
                state.producer_mut()?.discard_reserved_input(sequence);
            }
            for id in state.disabled_prior.keys().copied().collect::<Vec<_>>() {
                state.registration.cancel_sequence(id)?;
            }
            state.disabled_prior.clear();
            check(
                state.pending.is_empty()
                    && state.staged.is_empty()
                    && state.producer()?.incomplete_capture_count() == 0,
                "lock-v2 fenced root retains unresolved inputs",
            )?;
            if let Some(disabled) = state.disabled.as_ref() {
                state.registration.clear_disabled_lock_v2_root(&disabled.admission)?;
            }
            state.settled = true;
            if let Some(failure) = capture_failure {
                return Err(invalid(format!(
                    "native input capture failed before fenced cleanup: {failure:?}"
                )));
            }
            let admission = state.current_admission.as_ref()
                .ok_or_else(|| invalid("lock-v2 settled root original admission missing"))?.clone();
            state.producer()?.validate_admission(&admission)?;
            check(state.settled_roots.len() <= RETAINED_ROOT_CAPACITY
                && !state.settled_roots.iter().any(|original| original.same_original_admission(&admission)),
                "lock-v2 settled root evidence duplicate/capacity")?;
            state.settled_roots.push(admission);
            Ok(())
        })
    }
}

impl<Q, O> Observer<Q, O> {
    fn retire_consumed(
        &self,
        witness: RoventCollectionConsumedWitnessV2<'_>,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Accepted, |state| {
            state.producer()?.validate_admission(witness.admission())?;
            let ticket = *witness.accepted().ticket();
            if state.is_disabled_ticket(&ticket)? {
                return Ok(());
            }
            let pending = state
                .pending
                .get(&ticket.id)
                .ok_or_else(|| invalid("lock-v2 consumed ticket not registered"))?;
            check(
                pending.ticket == ticket && pending.accepted,
                "lock-v2 consumption lacks original acceptance or outbox capacity",
            )?;
            let next_retired = state.next_retired_record_count()?;
            state.drain()?;
            let row = state.producer_mut()?.retire_consumed(witness)?;
            state.game_retired_total = next_retired;
            state.pending.remove(&ticket.id);
            state.handled_tickets.insert(ticket.id, ticket);
            state.retain_row(row);
            Ok(())
        })
    }
}
