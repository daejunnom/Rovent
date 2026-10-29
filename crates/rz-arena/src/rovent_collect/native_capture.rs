//! 원 native owner의 submitted callback과 manager 수집기를 연결한다.
//! 큐에는 검증한 입력만 이동하며, 이 채널 자체는 NN 완료·수락 증거가 아니다.
use super::*;
use rz_search::pals::coordinator::RoleFailure;
use rz_uci::pals_native::ticketed::{
    NativeTicketQuestion, TicketExecutionEvidence, TicketPreparation, TicketOwnerDispatchContext,
    VirtualReviewCollectionObserver, VirtualReviewPhysicalInvocationEvidence, VirtualReviewCaptureFailureStage, VirtualReviewCaptureFailure,
    VirtualReviewPreparedTicket,
};
use std::cell::Cell;
use std::sync::{Arc, Mutex, mpsc};

enum RegisteredInputCapture {
    Legacy(RoventInputCaptureReservation),
    SubmittedV1(RoventInputCaptureReservationV2),
}
type Registrations = Arc<Mutex<BTreeMap<RequestId, RegisteredInputCapture>>>;
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoventNativeCaptureFailure {
    pub stage: VirtualReviewCaptureFailureStage,
    pub ticket: RoleTicketEnvelope,
    pub cause: RoleFailure,
}
pub(super) enum RoventFailedCapturedInput {
    Legacy(RoventCapturedInput),
    SubmittedV1(RoventCapturedInputV2),
}
#[derive(Default)]
struct CaptureFailureState {
    failure: Option<RoventNativeCaptureFailure>,
    input: Option<RoventFailedCapturedInput>,
}
type CaptureFailure = Arc<Mutex<CaptureFailureState>>;
pub struct RoventNativeCaptureRegistration {
    registered: Registrations,
}
pub struct RoventNativeInputObserver {
    registered: Registrations,
    failure: CaptureFailure,
    failed_input: Option<RoventFailedCapturedInput>,
    sender: mpsc::SyncSender<RoventCapturedInput>,
    submitted_sender: mpsc::SyncSender<RoventCapturedInputV2>,
}
pub struct RoventCapturedInputReceiver {
    receiver: mpsc::Receiver<RoventCapturedInput>,
    submitted_receiver: mpsc::Receiver<RoventCapturedInputV2>,
    disconnected: Cell<bool>,
    failure: CaptureFailure,
}
pub fn rovent_native_capture_channel() -> (
    RoventNativeCaptureRegistration,
    RoventNativeInputObserver,
    RoventCapturedInputReceiver,
) {
    let registered = Arc::new(Mutex::new(BTreeMap::new()));
    let failure = Arc::new(Mutex::new(CaptureFailureState::default()));
    let (sender, receiver) = mpsc::sync_channel(16);
    let (submitted_sender, submitted_receiver) = mpsc::sync_channel(16);
    (
        RoventNativeCaptureRegistration {
            registered: Arc::clone(&registered),
        },
        RoventNativeInputObserver {
            registered,
            failure: Arc::clone(&failure),
            failed_input: None,
            sender,
            submitted_sender,
        },
        RoventCapturedInputReceiver {
            receiver,
            submitted_receiver,
            disconnected: Cell::new(false),
            failure,
        },
    )
}
impl RoventNativeCaptureRegistration {
    /// 원 dispatch observer가 ticket 발급 후 execute 전에 호출한다.
    pub fn register(
        &self,
        ticket: &RoleTicketEnvelope,
        reservation: RoventInputCaptureReservation,
    ) -> Result<(), ArenaError> {
        let mut registered = self
            .registered
            .lock()
            .map_err(|_| invalid("Rovent capture registration poisoned"))?;
        check(
            registered.len() < 16 && !registered.contains_key(&ticket.id),
            "capture input registration duplicate/cap",
        )?;
        registered.insert(ticket.id, RegisteredInputCapture::Legacy(reservation));
        Ok(())
    }
    pub fn register_submitted_input_v1(
        &self,
        ticket: &RoleTicketEnvelope,
        reservation: RoventInputCaptureReservationV2,
    ) -> Result<(), ArenaError> {
        let mut registered = self
            .registered
            .lock()
            .map_err(|_| invalid("Rovent capture registration poisoned"))?;
        check(
            registered.len() < 16 && !registered.contains_key(&ticket.id),
            "capture input registration duplicate/cap",
        )?;
        registered.insert(ticket.id, RegisteredInputCapture::SubmittedV1(reservation));
        Ok(())
    }
    /// 아직 native owner가 읽지 않은 예약만 반환한다. 반환 sequence는
    /// producer.discard_reserved_input에 전달하고 물리 취소는 원 owner가 한다.
    pub fn cancel_unstarted(
        &self,
        id: RequestId,
    ) -> Result<Option<RoventInputCaptureReservation>, ArenaError> {
        let mut registered = self
            .registered
            .lock()
            .map_err(|_| invalid("Rovent capture registration poisoned"))?;
        check(
            !matches!(
                registered.get(&id),
                Some(RegisteredInputCapture::SubmittedV1(_))
            ),
            "legacy cancel refuses submitted-input reservation",
        )?;
        Ok(registered.remove(&id).map(|value| match value {
            RegisteredInputCapture::Legacy(value) => value,
            _ => unreachable!("checked legacy"),
        }))
    }
    pub(super) fn cancel_sequence(&self, id: RequestId) -> Result<Option<u64>, ArenaError> {
        Ok(self
            .registered
            .lock()
            .map_err(|_| invalid("Rovent capture registration poisoned"))?
            .remove(&id)
            .map(|value| match value {
                RegisteredInputCapture::Legacy(value) => value.sequence(),
                RegisteredInputCapture::SubmittedV1(value) => value.sequence(),
            }))
    }
}
impl RoventCapturedInputReceiver {
    /// The first capture error remains available even after the native owner
    /// continues. It is a collection failure, never NN failure evidence.
    pub fn capture_failure(&self) -> Result<Option<RoventNativeCaptureFailure>, ArenaError> {
        Ok(self.failure.lock().map_err(|_| invalid("capture failure state poisoned"))?.failure.clone())
    }
    /// Move the original unsent input to the collection owner for explicit
    /// cleanup. This is neither permission to publish nor a physical fence.
    pub(super) fn take_failed_input_for_cleanup(&self) -> Result<Option<RoventFailedCapturedInput>, ArenaError> {
        Ok(self.failure.lock().map_err(|_| invalid("capture failure state poisoned"))?.input.take())
    }
    /// Transfer queued original captures to the cleanup owner after its actual
    /// native fence. This path never validates or publishes these inputs.
    pub(super) fn take_queued_inputs_for_cleanup(&self) -> Vec<RoventFailedCapturedInput> {
        let mut inputs = Vec::new();
        while let Ok(input) = self.receiver.try_recv() {
            inputs.push(RoventFailedCapturedInput::Legacy(input));
        }
        while let Ok(input) = self.submitted_receiver.try_recv() {
            inputs.push(RoventFailedCapturedInput::SubmittedV1(input));
        }
        inputs
    }
    fn require_healthy_capture(&self) -> Result<(), ArenaError> {
        if let Some(failure) = self.capture_failure()? {
            return Err(invalid(format!("native input capture {:?} failed for {:?}: {:?}", failure.stage, failure.ticket.id, failure.cause)));
        }
        Ok(())
    }
    pub fn input_owner_disconnected(&self) -> bool {
        self.disconnected.get()
    }
    pub fn take(&self) -> Result<Option<RoventCapturedInput>, ArenaError> {
        self.require_healthy_capture()?;
        match self.receiver.try_recv() {
            Ok(input) => Ok(Some(input)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.disconnected.set(true);
                Ok(None)
            }
        }
    }
    /// 제출 입력은 같은 native owner가 Completion event를 발행하기 전에
    /// 전송한다. manager는 원 Completion을 처리하기 전에 이 큐를 비운다.
    pub fn admit_available(
        &self,
        producer: &mut RoventCorpusProducer,
    ) -> Result<usize, ArenaError> {
        let mut admitted = 0;
        while let Some(input) = self.take()? {
            producer.admit_captured_input(input)?;
            admitted += 1;
        }
        Ok(admitted)
    }
    pub fn admit_available_submitted_input_v1(
        &self,
        producer: &mut RoventCorpusProducerV2,
    ) -> Result<usize, ArenaError> {
        self.require_healthy_capture()?;
        let mut admitted = 0;
        loop {
            match self.submitted_receiver.try_recv() {
                Ok(input) => {
                    producer.admit_captured_input(input)?;
                    admitted += 1;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.disconnected.set(true);
                    break;
                }
            }
        }
        Ok(admitted)
    }
    /// Reuses the original submitted-input channel, but moves each captured
    /// input only into its original /2-lock producer. The queue itself does
    /// not grant native acceptance or physical-completion authority.
    pub(super) fn admit_available_lock_v2(
        &self,
        producer: &mut RoventCorpusProducerLockV2,
    ) -> Result<usize, ArenaError> {
        self.require_healthy_capture()?;
        let mut admitted = 0;
        loop {
            match self.submitted_receiver.try_recv() {
                Ok(input) => {
                    producer.admit_captured_input(input)?;
                    admitted += 1;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.disconnected.set(true);
                    break;
                }
            }
        }
        Ok(admitted)
    }
}
impl RoventNativeInputObserver {
    fn capture_submitted(
        &mut self,
        ticket: &RoleTicketEnvelope,
        preparation: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        prepared: &VirtualReviewPreparedTicket,
    ) -> Result<(), VirtualReviewCaptureFailure> {
        if self.failed_input.is_some() {
            return Err(VirtualReviewCaptureFailure::Required(RoleFailure::Backend(
                "original failed capture remains owned".into(),
            )));
        }
        if prepared.representation != ticket.representation {
            return Err(VirtualReviewCaptureFailure::Required(RoleFailure::Backend(
                "Rovent prepared input representation differs from original ticket".into(),
            )));
        }
        let reservation = self.registered.lock()
            .map_err(|_| RoleFailure::Backend("Rovent capture registration poisoned".into()))?
            .remove(&ticket.id)
            .ok_or_else(|| RoleFailure::Backend("Rovent native input lacks original manager reservation".into()))?;
        match reservation {
            RegisteredInputCapture::Legacy(reservation) => {
                let captured = reservation.capture(*ticket, &preparation.view, &prepared.input, &prepared.identity)
                    .map_err(|e| RoleFailure::Backend(e.to_string()))?;
                match self.sender.try_send(captured) {
                    Ok(()) => Ok(()),
                    Err(error) => {
                        let (cause, input) = send_error(error);
                        self.failed_input = Some(RoventFailedCapturedInput::Legacy(input));
                        Err(VirtualReviewCaptureFailure::Observation(cause))
                    }
                }
            }
            RegisteredInputCapture::SubmittedV1(reservation) => {
                let captured = reservation.capture(*ticket, &preparation.view, &prepared.input, &prepared.identity)
                    .map_err(|e| RoleFailure::Backend(e.to_string()))?;
                match self.submitted_sender.try_send(captured) {
                    Ok(()) => Ok(()),
                    Err(error) => {
                        let (cause, input) = send_error(error);
                        self.failed_input = Some(RoventFailedCapturedInput::SubmittedV1(input));
                        Err(VirtualReviewCaptureFailure::Observation(cause))
                    }
                }
            }
        }
    }
}
impl VirtualReviewCollectionObserver for RoventNativeInputObserver {
    fn preserve_observation_failure(
        &mut self,
        stage: VirtualReviewCaptureFailureStage,
        ticket: &RoleTicketEnvelope,
        failure: &RoleFailure,
    ) -> bool {
        // The native worker must never wait behind a collection consumer.
        // If the failure cannot be recorded, leave the existing fail-closed
        // behavior in place rather than silently losing the reason.
        let Ok(mut state) = self.failure.try_lock() else { return false };
        if state.failure.is_some() || state.input.is_some() || self.failed_input.is_none() { return false; }
        state.failure = Some(RoventNativeCaptureFailure { stage, ticket: *ticket, cause: failure.clone() });
        state.input = self.failed_input.take();
        true
    }
    fn submitted(
        &mut self,
        _: &RoleTicketEnvelope,
        _: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        _: &VirtualReviewPreparedTicket,
    ) -> Result<(), RoleFailure> {
        Err(RoleFailure::Unsupported("original ticket service owner required for capture"))
    }
    fn submitted_classified(
        &mut self,
        context: &TicketOwnerDispatchContext<'_, ReviewView, NativeTicketQuestion, VirtualReviewPreparedTicket>,
    ) -> Result<(), VirtualReviewCaptureFailure> {
        self.capture_submitted(context.ticket(), context.question(), context.prepared())
    }
    fn completed(
        &mut self,
        _: &RoleTicketEnvelope,
        _: PhysicalRoleCompletion,
        _: Option<&RoleOutput>,
        _: Option<TicketExecutionEvidence>,
        _: Option<VirtualReviewPhysicalInvocationEvidence>,
    ) -> Result<(), RoleFailure> {
        // 완료 권한은 manager가 받은 원 TicketOwnerRunWitness와 native event가
        // 발급한다. 이 callback의 공개 flag로 완료 권한을 재구성하지 않는다.
        Ok(())
    }
    fn retired(&mut self, _: &RoleTicketEnvelope, _: bool) -> Result<(), RoleFailure> {
        // 수락·퇴역은 원 registry의 AcceptedRoleStep을 따로 빌려 검증한다.
        Ok(())
    }
}

fn send_error<T>(error: mpsc::TrySendError<T>) -> (RoleFailure, T) {
    match error {
        mpsc::TrySendError::Full(input) => (RoleFailure::Backend("Rovent captured input channel full".into()), input),
        mpsc::TrySendError::Disconnected(input) => (RoleFailure::Backend("Rovent captured input owner disconnected".into()), input),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::pals::{RepresentationKey, SearchAuthority};
    use rz_contracts::pals_adaptive::ReviewViewSeal;
    use rz_contracts::{ClockDomain, Deadline, GameGeneration, MonotonicTick, OwnerId, PrecisionProfile, ProcessEpoch, RootGeneration, StateIdentity, StateRevision};

    #[test]
    fn capture_failure_retains_cause_and_blocks_future_capture_publication() {
        let (_, mut observer, receiver) = rovent_native_capture_channel();
        let authority = SearchAuthority { epoch: ProcessEpoch(1), game: GameGeneration(2), root: RootGeneration(3), implementation: Digest([1; 32]) };
        let ticket = RoleTicketEnvelope {
            id: RequestId::new(authority.epoch, 4), authority, question: Digest([2; 32]), role: Role::Proposer,
            view: ReviewViewSeal {
                authority, view_sequence: 5, context_policy: Digest([3; 32]), view_digest: Digest([4; 32]),
                root_state: StateIdentity { owner: OwnerId(1), revision: StateRevision(1), semantic: Digest([5; 32]) },
                actual_tokens: 0, selected_records: 0, anchors: 1, length_bucket: 8,
            },
            representation: RepresentationKey {
                input: Digest([6; 32]), history: Digest([7; 32]), candidates: Digest([8; 32]), mask_positions: Digest([9; 32]),
                model: Digest([10; 32]), encoding: Digest([11; 32]), precision: PrecisionProfile::Fp32,
                frozen_epoch: 1, record_revision: 0, role: Role::Proposer,
            },
            original_deadline: Deadline { clock: ClockDomain(authority.epoch), at: MonotonicTick(100) },
        };
        let cause = RoleFailure::Backend("capture queue full".into());
        // A report cannot authorize continuation without retaining its payload.
        assert!(!observer.preserve_observation_failure(VirtualReviewCaptureFailureStage::Submitted, &ticket, &cause));
        receiver.failure.lock().unwrap().failure = Some(RoventNativeCaptureFailure {
            stage: VirtualReviewCaptureFailureStage::Submitted, ticket, cause: cause.clone(),
        });
        let failure = receiver.capture_failure().unwrap().unwrap();
        assert_eq!(failure.ticket, ticket);
        assert_eq!(failure.cause, cause);
        assert_eq!(failure.stage, VirtualReviewCaptureFailureStage::Submitted);
        assert!(receiver.take().is_err());
        assert!(!observer.preserve_observation_failure(VirtualReviewCaptureFailureStage::Retired, &ticket, &RoleFailure::InvalidOutput));
        assert_eq!(receiver.capture_failure().unwrap(), Some(failure));
    }

    #[test]
    fn full_capture_queue_returns_the_original_owned_input() {
        let (cause, input) = send_error(mpsc::TrySendError::Full(37));
        assert_eq!(input, 37);
        assert!(matches!(cause, RoleFailure::Backend(_)));
    }
}
