//! 원 manager hook와 native 입력 owner 사이의 유한 수집 어댑터다.
//! 파일 I/O·신경망 인코딩·forward는 수행하지 않으며, 실제 퇴역 row만
//! caller 소유 outbox로 이동한다. 오류가 원 물리 owner의 처리를 대체하지 않는다.
use super::*;
use crate::pals_collect_v3::{GenerationActionSupportV3, GenerationCaptureWorkScopeV3};
use rz_search::pals::coordinator::FrameInput;
use rz_uci::pals_adaptive::{
    AdaptiveCollectionError, AdaptiveCollectionObserver, AdaptiveCollectionStage,
};
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

const CAPACITY: usize = 16;

// 하나의 원 manager hook을 공유하되 형식별 입력 예약과 퇴역 row는 끝까지
// 구분한다. legacy caller가 새 row를 받거나 schema를 바꾸어 게시하지 못한다.
enum CollectionProducerFamily {
    Legacy(Box<RoventCorpusProducer>),
    SubmittedV1(Box<RoventCorpusProducerV2>),
}
enum CollectionRetiredFamily {
    Legacy(RoventRetiredRecord),
    SubmittedV1(RoventRetiredRecordV2),
}
impl CollectionProducerFamily {
    fn epochs(&self) -> (u64, u64) {
        match self {
            Self::Legacy(p) => (p.manifest().model_epoch, p.manifest().context_epoch),
            Self::SubmittedV1(p) => (p.manifest().model_epoch, p.manifest().context_epoch),
        }
    }
    fn validate_admission(
        &self,
        admission: &RoventCollectionRootAdmission,
    ) -> Result<(), ArenaError> {
        match self {
            Self::Legacy(p) => p.validate_admission(admission),
            Self::SubmittedV1(p) => p.validate_admission(admission),
        }
    }
    fn incomplete_capture_count(&self) -> usize {
        match self {
            Self::Legacy(p) => p.incomplete_capture_count(),
            Self::SubmittedV1(p) => p.incomplete_capture_count(),
        }
    }
    fn discard_reserved_input(&mut self, sequence: u64) -> bool {
        match self {
            Self::Legacy(p) => p.discard_reserved_input(sequence),
            Self::SubmittedV1(p) => p.discard_reserved_input(sequence),
        }
    }
    fn discard_unpublished(&mut self, id: RequestId) -> bool {
        match self {
            Self::Legacy(p) => p.discard_unpublished(id),
            Self::SubmittedV1(p) => p.discard_unpublished(id),
        }
    }
    fn reserve_and_register(
        &mut self,
        registration: &RoventNativeCaptureRegistration,
        ticket: &RoleTicketEnvelope,
        context: GenerationCaptureContextV3,
        lineage: RoventLineage,
    ) -> Result<u64, ArenaError> {
        let (sequence, result) = match self {
            Self::Legacy(p) => {
                let reservation = p.reserve_input(context, lineage)?;
                let sequence = reservation.sequence();
                (sequence, registration.register(ticket, reservation))
            }
            Self::SubmittedV1(p) => {
                let reservation = p.reserve_input(context, lineage)?;
                let sequence = reservation.sequence();
                (
                    sequence,
                    registration.register_submitted_input_v1(ticket, reservation),
                )
            }
        };
        if let Err(error) = result {
            self.discard_reserved_input(sequence);
            return Err(error);
        }
        Ok(sequence)
    }
    fn observe_native_event<O: ProposerRankingOutput>(
        &mut self,
        event: &TicketNativeEvent<O>,
    ) -> Result<(), ArenaError> {
        match self {
            Self::Legacy(p) => p.observe_native_event(event),
            Self::SubmittedV1(p) => p.observe_native_event(event),
        }
    }
    fn retire_accepted(
        &mut self,
        admission: &RoventCollectionRootAdmission,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
    ) -> Result<CollectionRetiredFamily, ArenaError> {
        match self {
            Self::Legacy(p) => p
                .retire_accepted(admission, accepted, output)
                .map(CollectionRetiredFamily::Legacy),
            Self::SubmittedV1(p) => p
                .retire_accepted(admission, accepted, output)
                .map(CollectionRetiredFamily::SubmittedV1),
        }
    }
    fn retire_root_ranking(
        &mut self,
        accepted: &RoventRootRankingAcceptance,
        output: &RoleOutput,
    ) -> Result<CollectionRetiredFamily, ArenaError> {
        match self {
            Self::Legacy(p) => p
                .retire_root_ranking(accepted, output)
                .map(CollectionRetiredFamily::Legacy),
            Self::SubmittedV1(p) => p
                .retire_root_ranking(accepted, output)
                .map(CollectionRetiredFamily::SubmittedV1),
        }
    }
}

/// 실제 수집 실행에서 사전 등록한 식별자다. 이 값만으로 native·교사·학습
/// 권한을 발급하지 않으며, 원 root admission과 실제 ticket을 추가로 요구한다.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoventCollectionMetadata {
    pub game_id: String,
    pub opening_id: String,
    pub line_genealogy_id: String,
    pub lineage: RoventLineage,
}

struct CollectionState {
    producer: CollectionProducerFamily,
    registration: RoventNativeCaptureRegistration,
    receiver: RoventCapturedInputReceiver,
    metadata: RoventCollectionMetadata,
    admission: Option<RoventCollectionRootAdmission>,
    actual_position: Option<Position>,
    actual_history: Vec<BoardMove>,
    slots: BTreeMap<RequestId, u64>,
    retired: VecDeque<CollectionRetiredFamily>,
    failure: Option<AdaptiveCollectionError>,
    // Captures withheld after a collection error remain owned until this
    // observer is explicitly disposed; they never become publication proof.
    failed_captures: Vec<super::native_capture::RoventFailedCapturedInput>,
}

/// caller는 root 사이에도 이 원 owner를 보존한다. 공개 record를 읽어 이
/// owner의 private native witness나 원 퇴역 proof를 재구성할 수 없다.
#[derive(Clone)]
pub struct RoventCollectionOwner {
    state: Arc<Mutex<CollectionState>>,
}

/// 새 형식의 명시 제품 hook owner다. 기존 owner와 manager 처리 코드를 공유하며
/// 원 native observer의 /3 입력만 새 producer/writer에 전달한다.
#[derive(Clone)]
pub struct RoventCollectionOwnerV2 {
    original: RoventCollectionOwner,
}
impl RoventCollectionOwnerV2 {
    pub fn new(
        manifest: rz_experiments::rovent_corpus_v2::RoventCorpusManifestV2,
        metadata: RoventCollectionMetadata,
    ) -> Result<(Self, RoventNativeInputObserver), ArenaError> {
        check(
            generation_identifier(&metadata.game_id)
                && generation_identifier(&metadata.opening_id)
                && generation_identifier(&metadata.line_genealogy_id)
                && manifest.opening_splits.contains_key(&metadata.opening_id),
            "submitted-input collection metadata/opening not registered",
        )?;
        let producer = RoventCorpusProducerV2::new(manifest)?;
        let (registration, observer, receiver) = rovent_native_capture_channel();
        let original = RoventCollectionOwner {
            state: Arc::new(Mutex::new(CollectionState {
                producer: CollectionProducerFamily::SubmittedV1(Box::new(producer)),
                registration,
                receiver,
                metadata,
                admission: None,
                actual_position: None,
                actual_history: Vec::new(),
                slots: BTreeMap::new(),
                retired: VecDeque::new(),
                failure: None,
                failed_captures: Vec::new(),
            })),
        };
        Ok((Self { original }, observer))
    }
    pub fn observer<Q: 'static, O: ProposerRankingOutput + 'static>(
        &self,
    ) -> Box<dyn AdaptiveCollectionObserver<Q, O>> {
        self.original.observer()
    }
    pub fn with_producer<T>(
        &self,
        action: impl FnOnce(&mut RoventCorpusProducerV2) -> Result<T, ArenaError>,
    ) -> Result<T, ArenaError> {
        let mut state = self
            .original
            .state
            .lock()
            .map_err(|_| invalid("submitted-input collection owner poisoned"))?;
        match &mut state.producer {
            CollectionProducerFamily::SubmittedV1(p) => action(p),
            _ => Err(invalid("submitted-input borrow refuses legacy producer")),
        }
    }
    pub fn take_retired(&self) -> Result<Option<RoventRetiredRecordV2>, ArenaError> {
        let mut state = self
            .original
            .state
            .lock()
            .map_err(|_| invalid("submitted-input collection owner poisoned"))?;
        check(
            !matches!(
                state.retired.front(),
                Some(CollectionRetiredFamily::Legacy(_))
            ),
            "submitted-input read refuses legacy row",
        )?;
        Ok(state.retired.pop_front().map(|row| match row {
            CollectionRetiredFamily::SubmittedV1(row) => row,
            _ => unreachable!("checked submitted-input"),
        }))
    }
    pub fn failure(&self) -> Result<Option<AdaptiveCollectionError>, ArenaError> {
        self.original.failure()
    }
    /// 실제 observer.admitted가 보존한 같은 cap을 빌린다. 원 cap이 아직
    /// 없는 경우 metadata에서 admission을 만들지 않고 None을 반환한다.
    pub fn current_admission(&self) -> Result<Option<RoventCollectionRootAdmission>, ArenaError> {
        Ok(self
            .original
            .state
            .lock()
            .map_err(|_| invalid("submitted-input collection owner poisoned"))?
            .admission
            .clone())
    }
    /// 원 manager outbox의 남은 실제 row 수다. 이 값은 퇴역/수락이나 빈
    /// 성공 row를 만들지 않으며 caller의 종료 전 전수 회수 검사에만 쓴다.
    pub fn pending_retired_count(&self) -> Result<usize, ArenaError> {
        Ok(self
            .original
            .state
            .lock()
            .map_err(|_| invalid("submitted-input collection owner poisoned"))?
            .retired
            .len())
    }
}

impl RoventCollectionOwner {
    pub fn new(
        manifest: RoventCorpusManifest,
        metadata: RoventCollectionMetadata,
    ) -> Result<(Self, RoventNativeInputObserver), ArenaError> {
        check(
            generation_identifier(&metadata.game_id)
                && generation_identifier(&metadata.opening_id)
                && generation_identifier(&metadata.line_genealogy_id)
                && manifest.opening_splits.contains_key(&metadata.opening_id),
            "collection metadata/opening is not registered",
        )?;
        let producer = RoventCorpusProducer::new(manifest)?;
        let (registration, observer, receiver) = rovent_native_capture_channel();
        Ok((
            Self {
                state: Arc::new(Mutex::new(CollectionState {
                    producer: CollectionProducerFamily::Legacy(Box::new(producer)),
                    registration,
                    receiver,
                    metadata,
                    admission: None,
                    actual_position: None,
                    actual_history: Vec::new(),
                    slots: BTreeMap::new(),
                    retired: VecDeque::new(),
                    failure: None,
                    failed_captures: Vec::new(),
                })),
            },
            observer,
        ))
    }

    pub fn observer<Q: 'static, O: ProposerRankingOutput + 'static>(
        &self,
    ) -> Box<dyn AdaptiveCollectionObserver<Q, O>> {
        Box::new(RoventManagerCollectionObserver {
            owner: self.clone(),
            marker: PhantomData,
        })
    }

    /// teacher/outcome 부착·publication은 manager 밖의 원 caller가 수행한다.
    /// 이 borrow는 원 native·교사·Rules proof를 대신하지 않는다.
    pub fn with_producer<T>(
        &self,
        action: impl FnOnce(&mut RoventCorpusProducer) -> Result<T, ArenaError>,
    ) -> Result<T, ArenaError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid("collection owner poisoned"))?;
        match &mut state.producer {
            CollectionProducerFamily::Legacy(producer) => action(producer),
            _ => Err(invalid(
                "legacy producer borrow refuses submitted-input owner",
            )),
        }
    }

    pub fn take_retired(&self) -> Result<Option<RoventRetiredRecord>, ArenaError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid("collection owner poisoned"))?;
        check(
            !matches!(
                state.retired.front(),
                Some(CollectionRetiredFamily::SubmittedV1(_))
            ),
            "legacy retired read refuses submitted-input owner",
        )?;
        Ok(state.retired.pop_front().map(|row| match row {
            CollectionRetiredFamily::Legacy(row) => row,
            _ => unreachable!("checked legacy"),
        }))
    }

    pub fn failure(&self) -> Result<Option<AdaptiveCollectionError>, ArenaError> {
        Ok(self
            .state
            .lock()
            .map_err(|_| invalid("collection owner poisoned"))?
            .failure
            .clone())
    }
}

struct RoventManagerCollectionObserver<Q, O> {
    owner: RoventCollectionOwner,
    marker: PhantomData<fn(Q, O)>,
}

impl CollectionState {
    fn drain_captures(&mut self) -> Result<(), ArenaError> {
        match &mut self.producer {
            CollectionProducerFamily::Legacy(producer) => {
                self.receiver.admit_available(producer)?;
            }
            CollectionProducerFamily::SubmittedV1(producer) => {
                self.receiver.admit_available_submitted_input_v1(producer)?;
            }
        }
        Ok(())
    }
    fn cancel_collector_copy(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), ArenaError> {
        if let Some(sequence) = self.registration.cancel_sequence(ticket.id)? {
            self.producer.discard_reserved_input(sequence);
        }
        if let Some(sequence) = self.slots.remove(&ticket.id) {
            self.producer.discard_reserved_input(sequence);
        }
        self.producer.discard_unpublished(ticket.id);
        Ok(())
    }
    fn current(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<&RoventCollectionRootAdmission, ArenaError> {
        let admission = self
            .admission
            .as_ref()
            .ok_or_else(|| invalid("original root was not admitted"))?;
        admission
            .validate_ticket(ticket)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(admission)
    }
}

impl<Q, O> RoventManagerCollectionObserver<Q, O> {
    fn apply<T>(
        &self,
        stage: AdaptiveCollectionStage,
        action: impl FnOnce(&mut CollectionState) -> Result<T, ArenaError>,
    ) -> Result<T, AdaptiveCollectionError> {
        let mut state = self.owner.state.lock().map_err(|_| {
            AdaptiveCollectionError::new(
                stage,
                "RoventCollectionPoisoned",
                "original collection owner poisoned",
            )
        })?;
        match action(&mut state) {
            Ok(value) => Ok(value),
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

impl<Q, O: ProposerRankingOutput> AdaptiveCollectionObserver<Q, O>
    for RoventManagerCollectionObserver<Q, O>
{
    fn admitted(
        &mut self,
        admission: &RoventCollectionRootAdmission,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Dispatch, |state| {
            check(
                state.admission.is_none()
                    && state.slots.is_empty()
                    && state.producer.incomplete_capture_count() == 0
                    && state.failure.is_none(),
                "previous collection root is not settled",
            )?;
            state.producer.validate_admission(admission)?;
            let snapshot = admission.rules().rules().snapshot();
            let trace = snapshot
                .uci_replay(16_384)
                .map_err(|error| invalid(error.to_string()))?;
            let position = Position::from_snapshot(
                snapshot,
                rz_position::PositionLimits {
                    max_history_positions: 16_385,
                    ..rz_position::PositionLimits::default()
                },
            )
            .map_err(|error| invalid(error.to_string()))?;
            state.actual_position = Some(position);
            state.actual_history = trace.moves;
            state.admission = Some(admission.clone());
            Ok(())
        })
    }
    fn dispatch(
        &mut self,
        ticket: &RoleTicketEnvelope,
        input: &FrameInput<ReviewView, Q>,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Dispatch, |state| {
            state.current(ticket)?;
            check(
                state.failure.is_none()
                    && state.retired.len() + state.slots.len() < CAPACITY
                    && !state.slots.contains_key(&ticket.id)
                    && input.view_seal == ticket.view
                    && input.question_digest == ticket.question,
                "collection dispatch scope/queue capacity",
            )?;
            let (model_epoch, context_epoch) = state.producer.epochs();
            let context = GenerationCaptureContextV3 {
                game_id: state.metadata.game_id.clone(),
                opening_id: state.metadata.opening_id.clone(),
                line_genealogy_id: state.metadata.line_genealogy_id.clone(),
                model_epoch,
                context_epoch,
                capture_sequence: 0,
                work_scope: GenerationCaptureWorkScopeV3::from_native_ticket(ticket),
                actual_position: state
                    .actual_position
                    .as_ref()
                    .ok_or_else(|| invalid("original root Rules context missing"))?
                    .clone(),
                actual_history: state.actual_history.clone(),
                action_support: GenerationActionSupportV3::non_validator(),
            };
            let sequence = state.producer.reserve_and_register(
                &state.registration,
                ticket,
                context,
                state.metadata.lineage.clone(),
            )?;
            state.slots.insert(ticket.id, sequence);
            Ok(())
        })
    }
    fn native_event(
        &mut self,
        event: &TicketNativeEvent<O>,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::NativeEvent, |state| {
            state.drain_captures()?;
            state.producer.observe_native_event(event)?;
            if let TicketNativeEvent::Retired {
                accepted: false,
                result: Err(error),
                ..
            } = event
            {
                return Err(invalid(format!(
                    "original native retirement failed: {error:?}"
                )));
            }
            Ok(())
        })
    }
    fn accepted(
        &mut self,
        accepted: &AcceptedRoleStep,
        _: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Accepted, |state| {
            let admission = state.current(accepted.ticket())?.clone();
            check(
                state.retired.len() < CAPACITY,
                "retired collection outbox full",
            )?;
            let row = state.producer.retire_accepted(
                &admission,
                accepted,
                output.proposer_ranking_output(),
            )?;
            state.slots.remove(&accepted.ticket().id);
            state.retired.push_back(row);
            Ok(())
        })
    }
    fn root_ranking(
        &mut self,
        accepted: &RoventRootRankingAcceptance,
        _: &FrameInput<ReviewView, Q>,
        output: &O,
    ) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Accepted, |state| {
            let ticket = &accepted.ranking().source().ticket;
            check(
                state.current(ticket)?.raw_sha256() == accepted.admission().raw_sha256()
                    && state.retired.len() < CAPACITY,
                "initial P admission/outbox mismatch",
            )?;
            let row = state
                .producer
                .retire_root_ranking(accepted, output.proposer_ranking_output())?;
            state.slots.remove(&ticket.id);
            state.retired.push_back(row);
            Ok(())
        })
    }
    fn abandoned(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Abandoned, |state| {
            state.current(ticket)?;
            state.drain_captures()?;
            state.cancel_collector_copy(ticket)
        })
    }
    fn settled(&mut self, tickets: &[RoleTicketEnvelope]) -> Result<(), AdaptiveCollectionError> {
        self.apply(AdaptiveCollectionStage::Settled, |state| {
            // This callback is made only after the original native owner fence.
            // Failed/queued inputs are retained, never admitted or published.
            let capture_failure = state.receiver.capture_failure()?;
            if capture_failure.is_some() {
                if let Some(input) = state.receiver.take_failed_input_for_cleanup()? {
                    state.failed_captures.push(input);
                }
                state.failed_captures.extend(state.receiver.take_queued_inputs_for_cleanup());
            } else {
                state.drain_captures()?;
            }
            for ticket in tickets {
                state.current(ticket)?;
                state.cancel_collector_copy(ticket)?;
            }
            check(
                state.slots.is_empty() && state.producer.incomplete_capture_count() == 0,
                "fenced root retains unresolved collector inputs",
            )?;
            state.admission = None;
            state.actual_position = None;
            state.actual_history.clear();
            // 원 완료·퇴역 row와 최초 오류는 caller가 회수할 때까지 보존한다.
            if let Some(failure) = capture_failure {
                return Err(invalid(format!("native input capture failed before fenced cleanup: {failure:?}")));
            }
            Ok(())
        })
    }
}
