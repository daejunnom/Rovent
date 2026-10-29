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
    LockV2 {
        ticket: RoleTicketEnvelope,
        reservation: RoventInputCaptureReservationV2,
        admission: rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
    },
    /// An already-issued ticket dispatched after the root was disabled, but
    /// whose sequence predates the disabled boundary. Exact identity remains
    /// in the bounded per-ticket registration until native dispatch consumes it.
    LockV2DisabledPrior {
        ticket: RoleTicketEnvelope,
        admission: rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
    },
}
pub(super) enum LockV2RegistrationAttempt {
    Ready,
    Busy(RoventInputCaptureReservationV2),
    Failed(ArenaError, RoventInputCaptureReservationV2),
}
/// Created only after the original service context and root admission match.
/// This is still input capture, not proof of execution or acceptance.
pub(super) struct RoventCapturedLockInputV2 {
    captured: RoventCapturedInputV2,
    admission: rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
}
impl RoventCapturedLockInputV2 {
    pub(super) fn captured(&self) -> &RoventCapturedInputV2 { &self.captured }
    pub(super) fn admission(&self) -> &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2 { &self.admission }
    pub(super) fn into_original(self) -> (RoventCapturedInputV2, rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2) {
        (self.captured, self.admission)
    }
}
type Registrations = Arc<Mutex<BTreeMap<RequestId, RegisteredInputCapture>>>;
type DisabledRoot = Arc<Mutex<Option<(rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2, RoleTicketEnvelope)>>>;
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoventNativeCaptureFailure {
    pub stage: VirtualReviewCaptureFailureStage,
    pub ticket: RoleTicketEnvelope,
    pub cause: RoleFailure,
}
pub(super) enum RoventFailedCapturedInput {
    Legacy(RoventCapturedInput),
    SubmittedV1(RoventCapturedInputV2),
    LockV2(RoventCapturedLockInputV2),
}
#[derive(Default)]
struct CaptureFailureState {
    failure: Option<RoventNativeCaptureFailure>,
    input: Option<RoventFailedCapturedInput>,
}
type CaptureFailure = Arc<Mutex<CaptureFailureState>>;
#[derive(Clone)]
pub struct RoventNativeCaptureRegistration {
    registered: Registrations,
    disabled_root: DisabledRoot,
}
pub struct RoventNativeInputObserver {
    registered: Registrations,
    disabled_root: DisabledRoot,
    failure: CaptureFailure,
    failed_input: Option<RoventFailedCapturedInput>,
    sender: mpsc::SyncSender<RoventCapturedInput>,
    submitted_sender: mpsc::SyncSender<RoventCapturedInputV2>,
    lock_v2_sender: mpsc::SyncSender<RoventCapturedLockInputV2>,
}
pub struct RoventCapturedInputReceiver {
    receiver: mpsc::Receiver<RoventCapturedInput>,
    submitted_receiver: mpsc::Receiver<RoventCapturedInputV2>,
    lock_v2_receiver: mpsc::Receiver<RoventCapturedLockInputV2>,
    disconnected: Cell<bool>,
    failure: CaptureFailure,
}
pub fn rovent_native_capture_channel() -> (
    RoventNativeCaptureRegistration,
    RoventNativeInputObserver,
    RoventCapturedInputReceiver,
) {
    let registered = Arc::new(Mutex::new(BTreeMap::new()));
    let disabled_root = Arc::new(Mutex::new(None));
    let failure = Arc::new(Mutex::new(CaptureFailureState::default()));
    let (sender, receiver) = mpsc::sync_channel(16);
    let (submitted_sender, submitted_receiver) = mpsc::sync_channel(16);
    let (lock_v2_sender, lock_v2_receiver) = mpsc::sync_channel(16);
    (
        RoventNativeCaptureRegistration {
            registered: Arc::clone(&registered),
            disabled_root: Arc::clone(&disabled_root),
        },
        RoventNativeInputObserver {
            registered,
            disabled_root,
            failure: Arc::clone(&failure),
            failed_input: None,
            sender,
            submitted_sender,
            lock_v2_sender,
        },
        RoventCapturedInputReceiver {
            receiver,
            submitted_receiver,
            lock_v2_receiver,
            disconnected: Cell::new(false),
            failure,
        },
    )
}
impl RoventNativeCaptureRegistration {
    /// Test-only contention injection. The closure is controlled by the test;
    /// no ticket, admission, capture or physical completion is manufactured.
    #[cfg(feature = "test-fixtures")]
    pub(super) fn test_with_registration_held<T>(&self, action: impl FnOnce() -> T) -> Result<T, ArenaError> {
        let _held = self.registered.lock()
            .map_err(|_| invalid("Rovent capture registration poisoned"))?;
        Ok(action())
    }
    pub(super) fn try_has_registration(&self, id: RequestId) -> Result<Option<bool>, ArenaError> {
        match self.registered.try_lock() {
            Ok(registered) => Ok(Some(registered.contains_key(&id))),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => Err(invalid("Rovent capture registration poisoned")),
        }
    }
    pub(super) fn try_register_disabled_prior(
        &self, ticket: &RoleTicketEnvelope,
        admission: &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
    ) -> Result<bool, ArenaError> {
        if let Err(error) = admission.try_validate_ticket(ticket) {
            return if error.kind == rz_uci::pals_adaptive::RoventTicketValidationKindV2::Busy {
                Ok(false)
            } else { Err(invalid(error.error.to_string())) };
        }
        let mut registered = match self.registered.try_lock() {
            Ok(registered) => registered,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("Rovent capture registration poisoned")),
        };
        check(registered.len() < 16 && !registered.contains_key(&ticket.id),
            "disabled prior input registration duplicate/cap")?;
        registered.insert(ticket.id, RegisteredInputCapture::LockV2DisabledPrior {
            ticket: *ticket, admission: admission.clone(),
        });
        Ok(true)
    }
    pub(super) fn try_disable_lock_v2_root(&self,
        admission: &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
        first_ticket: &RoleTicketEnvelope,
    ) -> Result<bool, ArenaError> {
        if let Err(error) = admission.try_validate_ticket(first_ticket) {
            return if error.kind == rz_uci::pals_adaptive::RoventTicketValidationKindV2::Busy {
                Ok(false)
            } else { Err(invalid(error.error.to_string())) };
        }
        let mut disabled = match self.disabled_root.try_lock() {
            Ok(disabled) => disabled,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("lock-v2 disabled root poisoned")),
        };
        check(disabled.is_none(), "lock-v2 capture root already disabled")?;
        *disabled = Some((admission.clone(), *first_ticket));
        Ok(true)
    }
    pub(super) fn clear_disabled_lock_v2_root(&self) -> Result<(), ArenaError> {
        *self.disabled_root.lock().map_err(|_| invalid("lock-v2 disabled root poisoned"))? = None;
        Ok(())
    }
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
    pub(super) fn try_register_lock_v2(
        &self,
        ticket: &RoleTicketEnvelope,
        reservation: RoventInputCaptureReservationV2,
        admission: &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
    ) -> LockV2RegistrationAttempt {
        if let Err(error) = admission.try_validate_ticket(ticket) {
            return if error.kind == rz_uci::pals_adaptive::RoventTicketValidationKindV2::Busy {
                LockV2RegistrationAttempt::Busy(reservation)
            } else { LockV2RegistrationAttempt::Failed(invalid(error.error.to_string()), reservation) };
        }
        let mut registered = match self.registered.try_lock() {
            Ok(registered) => registered,
            Err(std::sync::TryLockError::WouldBlock) => return LockV2RegistrationAttempt::Busy(reservation),
            Err(std::sync::TryLockError::Poisoned(_)) => return LockV2RegistrationAttempt::Failed(
                invalid("Rovent capture registration poisoned"), reservation),
        };
        if let Err(error) = check(registered.len() < 16 && !registered.contains_key(&ticket.id),
            "capture input registration duplicate/cap") {
            return LockV2RegistrationAttempt::Failed(error, reservation);
        }
        registered.insert(ticket.id, RegisteredInputCapture::LockV2 {
            ticket: *ticket, reservation, admission: admission.clone(),
        });
        LockV2RegistrationAttempt::Ready
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
                Some(RegisteredInputCapture::SubmittedV1(_) | RegisteredInputCapture::LockV2 { .. } | RegisteredInputCapture::LockV2DisabledPrior { .. })
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
            .and_then(|value| match value {
                RegisteredInputCapture::Legacy(value) => Some(value.sequence()),
                RegisteredInputCapture::SubmittedV1(value) => Some(value.sequence()),
                RegisteredInputCapture::LockV2 { reservation, .. } => Some(reservation.sequence()),
                RegisteredInputCapture::LockV2DisabledPrior { .. } => None,
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
        while let Ok(input) = self.lock_v2_receiver.try_recv() {
            inputs.push(RoventFailedCapturedInput::LockV2(input));
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
    /// Uses a distinct original submitted-input channel and moves each captured
    /// input only into its original /2-lock producer. The queue itself does
    /// not grant native acceptance or physical-completion authority.
    pub(super) fn admit_available_lock_v2(
        &self,
        producer: &mut RoventCorpusProducerLockV2,
    ) -> Result<usize, Box<(ArenaError, Option<RoventCapturedLockInputV2>)>> {
        self.require_healthy_capture().map_err(|error| Box::new((error, None)))?;
        let mut admitted = 0;
        loop {
            match self.lock_v2_receiver.try_recv() {
                Ok(input) => {
                    if let Err(failure) = producer.admit_captured_lock_input(input) {
                        let (error, input) = *failure;
                        return Err(Box::new((error, Some(input))));
                    }
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
        context: &TicketOwnerDispatchContext<'_, ReviewView, NativeTicketQuestion, VirtualReviewPreparedTicket>,
    ) -> Result<(), VirtualReviewCaptureFailure> {
        let ticket = context.ticket();
        let preparation = context.question();
        let prepared = context.prepared();
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
        let reservation = {
            let mut registered = self.registered.lock()
                .map_err(|_| RoleFailure::Backend("Rovent capture registration poisoned".into()))?;
            let Some(candidate) = registered.get(&ticket.id) else {
                let disabled = self.disabled_root.lock().map_err(|_| RoleFailure::Backend("lock-v2 disabled root poisoned".into()))?;
                let (admission, first_ticket) = disabled.as_ref().ok_or_else(|| RoleFailure::Backend("Rovent native input lacks original manager reservation".into()))?;
                if ticket.id.epoch != first_ticket.id.epoch || ticket.id.sequence < first_ticket.id.sequence {
                    return Err(VirtualReviewCaptureFailure::Required(RoleFailure::Backend(
                        "missing reservation predates disabled capture boundary".into())));
                }
                // This path discards optional collection input only. A late
                // native callback may outlive logical root authority; it must
                // still match the original immutable root and physical owner.
                admission.validate_native_discard_owner(context)
                    .map_err(|error| RoleFailure::Backend(error.to_string()))?;
                return Ok(());
            };
            if let RegisteredInputCapture::LockV2 { ticket: registered_ticket, admission, .. } = candidate {
                if registered_ticket != ticket {
                    return Err(VirtualReviewCaptureFailure::Required(RoleFailure::Backend(
                        "lock-v2 capture ticket differs from the original registration".into(),
                    )));
                }
                admission.validate_native_dispatch_owner(context)
                    .map_err(|error| RoleFailure::Backend(error.to_string()))?;
            }
            if let RegisteredInputCapture::LockV2DisabledPrior { ticket: registered_ticket, admission } = candidate {
                if registered_ticket != ticket {
                    return Err(VirtualReviewCaptureFailure::Required(RoleFailure::Backend(
                        "disabled prior ticket differs from original registration".into(),
                    )));
                }
                admission.validate_native_discard_owner(context)
                    .map_err(|error| RoleFailure::Backend(error.to_string()))?;
            }
            registered.remove(&ticket.id).expect("checked registration")
        };
        match reservation {
            RegisteredInputCapture::LockV2DisabledPrior { .. } => Ok(()),
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
            RegisteredInputCapture::LockV2 { reservation, admission, .. } => {
                let captured = reservation.capture(*ticket, &preparation.view, &prepared.input, &prepared.identity)
                    .map_err(|error| RoleFailure::Backend(error.to_string()))?;
                let input = RoventCapturedLockInputV2 { captured, admission };
                match self.lock_v2_sender.try_send(input) {
                    Ok(()) => Ok(()),
                    Err(error) => {
                        let (cause, input) = send_error(error);
                        self.failed_input = Some(RoventFailedCapturedInput::LockV2(input));
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
        self.capture_submitted(context)
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
