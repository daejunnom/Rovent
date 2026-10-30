//! 원 native owner의 submitted callback과 manager 수집기를 연결한다.
//! 큐에는 검증한 입력만 이동하며, 이 채널 자체는 NN 완료·수락 증거가 아니다.
use super::*;
use rz_search::pals::coordinator::RoleFailure;
use rz_uci::pals_native::ticketed::{
    NativeTicketQuestion, TicketExecutionEvidence, TicketOwnerDispatchContext, TicketPreparation,
    VirtualReviewCaptureFailure, VirtualReviewCaptureFailureStage, VirtualReviewCollectionObserver,
    VirtualReviewPhysicalInvocationEvidence, VirtualReviewPreparedTicket,
};
use std::cell::Cell;
use std::sync::{Arc, Mutex, mpsc, atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering}};
use super::producer_v2::{SubmittedTeacherQueryV2, SpeculativeTicketAuthorityV2};

/// A ticket's independent speculative handoff state. It never proves NN
/// dispatch, acceptance, physical completion, a teacher target or retirement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpeculativeHandoffV2 { Reserved, Sent, QueueFull, Disconnected, PreparationFailed, OriginalCaptureFailed, Canceled, PreparationFailureDetailUnknown, NotRequired }
impl SpeculativeHandoffV2 {
    fn decode(value: u8) -> Self {
        match value { 1 => Self::Sent, 2 => Self::QueueFull, 3 => Self::Disconnected,
            4 => Self::PreparationFailed, 5 => Self::OriginalCaptureFailed, 6 => Self::Canceled,
            7 => Self::PreparationFailureDetailUnknown, 8 => Self::NotRequired, _ => Self::Reserved }
    }
}
struct SpeculativeTicketV2 {
    ticket: RoleTicketEnvelope,
    state: AtomicU8,
    error: Mutex<Option<ArenaError>>,
    authority: SpeculativeTicketAuthorityV2,
}
struct SpeculativeRouteV2 {
    registration_owner: Registrations,
    installed: AtomicBool,
    admission: rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
    sender: mpsc::SyncSender<SubmittedTeacherQueryV2>,
    tickets: Mutex<std::collections::HashMap<RequestId, Arc<SpeculativeTicketV2>>>,
    maximum_tickets: usize,
    unattributed: AtomicU64,
    cancel_all: Arc<AtomicBool>,
    root_revoked: Arc<AtomicBool>,
    targets_revoked: Arc<AtomicBool>,
}
/// Retains each original ticket and its one callback outcome, including
/// Full/closed. The root worker owns this handle and the matching receiver.
#[derive(Clone)]
pub(crate) struct SpeculativeRootV2 { route: Arc<SpeculativeRouteV2> }
impl SpeculativeRootV2 {
    pub(crate) fn matches_admission(&self, admission: &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2) -> bool {
        self.route.admission.same_original_admission(admission)
    }
    pub(crate) fn same_original_route(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.route, &other.route)
    }
    pub(crate) fn matches_ticket_authority(&self, ticket: &RoleTicketEnvelope) -> bool {
        ticket.authority == self.route.admission.root_admission().authority()
    }
    pub(crate) fn unattributed_count(&self) -> u64 {
        self.route.unattributed.load(Ordering::Acquire)
    }
    /// Request cancellation, not completion. On contention or an unknown
    /// ticket, close only this root and report unattributed collection state.
    pub(crate) fn cancel_ticket(&self, ticket: &RoleTicketEnvelope) -> bool {
        // A delayed discard from a different root cannot cancel this root.
        if !self.matches_ticket_authority(ticket) { return false; }
        let found = match self.route.tickets.try_lock() {
            Ok(tickets) => tickets.get(&ticket.id).filter(|item| item.ticket == *ticket)
                .map(|item| { item.authority.cancel(); }),
            Err(_) => None,
        };
        if found.is_none() {
            // Publish the conservative target revocation first so closure
            // is never visible while the join guard still sees it as open.
            self.route.targets_revoked.store(true, Ordering::Release);
            self.route.cancel_all.store(true, Ordering::Release);
            let _ = self.route.unattributed.fetch_update(Ordering::AcqRel, Ordering::Acquire,
                |value| Some(value.saturating_add(1)));
        }
        true
    }
    pub(crate) fn cancel_all(&self) { self.route.cancel_all.store(true, Ordering::Release); }
    /// Shared original root flag for the session owner. Sharing this flag
    /// does not grant authority, prove completion or linearize target commit.
    pub(crate) fn target_revoke_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.route.targets_revoked)
    }
    /// Used only for actual/unknown root abort; target authority is revoked.
    pub(crate) fn revoke_all_targets(&self) {
        self.route.targets_revoked.store(true, Ordering::Release);
        self.route.cancel_all.store(true, Ordering::Release);
    }
    pub(crate) fn try_observations(&self) -> Result<Option<Vec<(RoleTicketEnvelope, SpeculativeHandoffV2, Option<String>)>>, ArenaError> {
        let tickets = match self.route.tickets.try_lock() {
            Ok(tickets) => tickets,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("speculative ticket custody poisoned")),
        };
        let mut observations = Vec::new();
        observations.try_reserve(tickets.len()).map_err(|_| invalid("speculative observations allocation"))?;
        for item in tickets.values() {
            let state = SpeculativeHandoffV2::decode(item.state.load(Ordering::Acquire));
            let error = if state == SpeculativeHandoffV2::PreparationFailed {
                match item.error.try_lock() {
                    Ok(error) => error.as_ref().map(ToString::to_string),
                    Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                    Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("speculative error custody poisoned")),
                }
            } else { None };
            observations.push((item.ticket, state, error));
        }
        observations.sort_unstable_by_key(|(ticket, _, _)| ticket.id);
        Ok(Some(observations))
    }
}


enum RegisteredInputCapture {
    Legacy(RoventInputCaptureReservation),
    SubmittedV1(RoventInputCaptureReservationV2),
    LockV2 {
        ticket: RoleTicketEnvelope,
        reservation: RoventInputCaptureReservationV2,
        admission: rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
        speculative: Option<(Arc<SpeculativeRouteV2>, Arc<SpeculativeTicketV2>)>,
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
    pub(super) fn captured(&self) -> &RoventCapturedInputV2 {
        &self.captured
    }
    pub(super) fn admission(&self) -> &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2 {
        &self.admission
    }
    pub(super) fn into_original(
        self,
    ) -> (
        RoventCapturedInputV2,
        rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
    ) {
        (self.captured, self.admission)
    }
}
type Registrations = Arc<Mutex<BTreeMap<RequestId, RegisteredInputCapture>>>;
enum DisabledCaptureRoot {
    AfterTicket {
        admission: rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
        first_ticket: RoleTicketEnvelope,
    },
    WholeRoot(rz_uci::pals_adaptive::RoventNoCaptureRootCapabilityV2),
}
impl DisabledCaptureRoot {
    fn authority(&self) -> rz_contracts::pals::SearchAuthority {
        match self {
            Self::AfterTicket { admission, .. } => admission.root_admission().authority(),
            Self::WholeRoot(capability) => capability.authority(),
        }
    }
}
const MAX_RETAINED_DISABLED_ROOTS: usize = 256;
type DisabledRoot = Arc<Mutex<Vec<DisabledCaptureRoot>>>;
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
    speculative_roots: Arc<Mutex<Vec<Arc<SpeculativeRouteV2>>>>,
    speculative_active: Arc<AtomicUsize>,
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
    let disabled_root = Arc::new(Mutex::new(Vec::new()));
    let speculative_roots = Arc::new(Mutex::new(Vec::new()));
    let failure = Arc::new(Mutex::new(CaptureFailureState::default()));
    let (sender, receiver) = mpsc::sync_channel(16);
    let (submitted_sender, submitted_receiver) = mpsc::sync_channel(16);
    let (lock_v2_sender, lock_v2_receiver) = mpsc::sync_channel(16);
    (
        RoventNativeCaptureRegistration {
            registered: Arc::clone(&registered),
            disabled_root: Arc::clone(&disabled_root),
            speculative_roots,
            speculative_active: Arc::new(AtomicUsize::new(0)),
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
    pub(super) fn same_original_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.registered, &other.registered)
            && Arc::ptr_eq(&self.disabled_root, &other.disabled_root)
            && Arc::ptr_eq(&self.speculative_roots, &other.speculative_roots)
            && Arc::ptr_eq(&self.speculative_active, &other.speculative_active)
    }
    pub(crate) fn owns_route(&self, route: &SpeculativeRootV2) -> bool {
        Arc::ptr_eq(&self.registered, &route.route.registration_owner)
            && route.route.installed.load(Ordering::Acquire)
    }
    /// Install a dedicated root channel before its first ticket registration.
    /// None means transient contention; existing routes are never replaced.
    pub(crate) fn try_install_speculative_root(
        &self, admission: &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
        channel_capacity: usize, maximum_tickets: usize,
        root_revoked: Arc<AtomicBool>,
    ) -> Result<Option<(SpeculativeRootV2, mpsc::Receiver<SubmittedTeacherQueryV2>)>, ArenaError> {
        check((1..=16).contains(&channel_capacity) && (1..=65_536).contains(&maximum_tickets),
            "speculative root channel/ticket capacity")?;
        // Same lock order as ticket registration. A late installation must
        // never omit already-registered tickets and masquerade as complete.
        let registered = match self.registered.try_lock() {
            Ok(registered) => registered,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("capture registrations poisoned")),
        };
        check(!registered.values().any(|entry| match entry {
            RegisteredInputCapture::LockV2 { admission: old, .. }
            | RegisteredInputCapture::LockV2DisabledPrior { admission: old, .. } => old.same_original_admission(admission),
            _ => false,
        }), "speculative root installed after ticket registration")?;
        let mut routes = match self.speculative_roots.try_lock() {
            Ok(routes) => routes,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("speculative root custody poisoned")),
        };
        check(routes.len() < MAX_RETAINED_DISABLED_ROOTS && !routes.iter().any(|route|
            route.admission.root_admission().authority() == admission.root_admission().authority()),
            "speculative root duplicate or retained capacity")?;
        routes.try_reserve(1).map_err(|_| invalid("speculative root allocation"))?;
        let (sender, receiver) = mpsc::sync_channel(channel_capacity);
        let route = Arc::new(SpeculativeRouteV2 { registration_owner: Arc::clone(&self.registered),
            installed: AtomicBool::new(true), admission: admission.clone(), sender,
            tickets: Mutex::new(std::collections::HashMap::new()), maximum_tickets,
            unattributed: AtomicU64::new(0), cancel_all: Arc::new(AtomicBool::new(false)), root_revoked,
            targets_revoked: Arc::new(AtomicBool::new(false)) });
        routes.push(Arc::clone(&route));
        self.speculative_active.fetch_add(1, Ordering::Release);
        Ok(Some((SpeculativeRootV2 { route }, receiver)))
    }

    /// Roll back only an unpublished admission slot whose `try_send` failed
    /// before observer installation. The caller must retain that original
    /// send failure; this method cannot infer process or physical completion.
    /// Once any ticket is registered this path must not remove the route.
    pub(crate) fn try_rollback_unstarted_speculative_root(
        &self, root: &SpeculativeRootV2,
    ) -> Result<Option<bool>, ArenaError> {
        let registered = match self.registered.try_lock() {
            Ok(registered) => registered,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("capture registrations poisoned")),
        };
        let mut routes = match self.speculative_roots.try_lock() {
            Ok(routes) => routes,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("speculative root custody poisoned")),
        };
        let tickets = match root.route.tickets.try_lock() {
            Ok(tickets) => tickets,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("speculative ticket custody poisoned")),
        };
        let admission = &root.route.admission;
        if !tickets.is_empty() || root.route.unattributed.load(Ordering::Acquire) != 0
            || registered.values().any(|entry| match entry {
            RegisteredInputCapture::LockV2 { admission: old, .. }
            | RegisteredInputCapture::LockV2DisabledPrior { admission: old, .. } => old.same_original_admission(admission),
            _ => false,
        }) { return Ok(Some(false)); }
        let Some(index) = routes.iter().position(|route| Arc::ptr_eq(route, &root.route)) else {
            return Ok(Some(false));
        };
        root.route.installed.store(false, Ordering::Release);
        routes.remove(index);
        self.speculative_active.fetch_sub(1, Ordering::AcqRel);
        Ok(Some(true))
    }
    /// Remove only the registration routing entry after the original PC
    /// worker's joined handoff/fence proof. The worker retains the same Arc
    /// and its observations; unlinking grants no collection success.
    #[cfg(target_os = "linux")]
    pub(super) fn try_unlink_finished_speculative_root(
        &self,
        root: &SpeculativeRootV2,
        proof: &super::pc_hooks_lock_v2::PcSpeculativeRouteFenceV2,
        received: usize,
    ) -> Result<Option<bool>, ArenaError> {
        check(proof.matches(&root.route.admission)
            && Arc::ptr_eq(&self.registered, &root.route.registration_owner),
            "speculative route fence or original registration differs")?;
        let registered = match self.registered.try_lock() {
            Ok(registered) => registered,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("capture registrations poisoned")),
        };
        let mut routes = match self.speculative_roots.try_lock() {
            Ok(routes) => routes,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("speculative route custody poisoned")),
        };
        let tickets = match root.route.tickets.try_lock() {
            Ok(tickets) => tickets,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("speculative ticket custody poisoned")),
        };
        if root.route.unattributed.load(Ordering::Acquire) != 0
            || registered.values().any(|entry| match entry {
                RegisteredInputCapture::LockV2 { admission, .. }
                | RegisteredInputCapture::LockV2DisabledPrior { admission, .. } => admission.same_original_admission(&root.route.admission),
                _ => false,
            }) { return Ok(Some(false)); }
        let mut sent = 0usize;
        for ticket in tickets.values() {
            match SpeculativeHandoffV2::decode(ticket.state.load(Ordering::Acquire)) {
                SpeculativeHandoffV2::Reserved | SpeculativeHandoffV2::PreparationFailureDetailUnknown => return Ok(Some(false)),
                SpeculativeHandoffV2::Sent => sent = sent.saturating_add(1),
                _ => {}
            }
        }
        if sent != received { return Ok(Some(false)); }
        let Some(index) = routes.iter().position(|candidate| Arc::ptr_eq(candidate, &root.route)) else {
            // The same root handle can be inspected after a successful unlink;
            // no different route or synthetic proof may stand in for it.
            return Ok(Some(false));
        };
        root.route.installed.store(false, Ordering::Release);
        routes.remove(index);
        self.speculative_active.fetch_sub(1, Ordering::AcqRel);
        Ok(Some(true))
    }
    pub(super) fn try_disabled_root_count(&self) -> Result<Option<usize>, ArenaError> {
        match self.disabled_root.try_lock() {
            Ok(markers) => Ok(Some(markers.len())),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => Err(invalid("lock-v2 disabled root poisoned")),
        }
    }
    /// Test-only contention injection. The closure is controlled by the test;
    /// no ticket, admission, capture or physical completion is manufactured.
    #[cfg(feature = "test-fixtures")]
    pub(super) fn test_with_registration_held<T>(
        &self,
        action: impl FnOnce() -> T,
    ) -> Result<T, ArenaError> {
        let _held = self
            .registered
            .lock()
            .map_err(|_| invalid("Rovent capture registration poisoned"))?;
        Ok(action())
    }
    pub(super) fn try_has_registration(&self, id: RequestId) -> Result<Option<bool>, ArenaError> {
        match self.registered.try_lock() {
            Ok(registered) => Ok(Some(registered.contains_key(&id))),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                Err(invalid("Rovent capture registration poisoned"))
            }
        }
    }
    pub(super) fn try_register_disabled_prior(
        &self,
        ticket: &RoleTicketEnvelope,
        admission: &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
    ) -> Result<bool, ArenaError> {
        if let Err(error) = admission.try_validate_ticket(ticket) {
            return if error.kind == rz_uci::pals_adaptive::RoventTicketValidationKindV2::Busy {
                Ok(false)
            } else {
                Err(invalid(error.error.to_string()))
            };
        }
        let mut registered = match self.registered.try_lock() {
            Ok(registered) => registered,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid("Rovent capture registration poisoned"));
            }
        };
        check(
            registered.len() < 16 && !registered.contains_key(&ticket.id),
            "disabled prior input registration duplicate/cap",
        )?;
        registered.insert(
            ticket.id,
            RegisteredInputCapture::LockV2DisabledPrior {
                ticket: *ticket,
                admission: admission.clone(),
            },
        );
        Ok(true)
    }
    pub(super) fn try_disable_lock_v2_root(
        &self,
        admission: &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
        first_ticket: &RoleTicketEnvelope,
    ) -> Result<bool, ArenaError> {
        if let Err(error) = admission.try_validate_ticket(first_ticket) {
            return if error.kind == rz_uci::pals_adaptive::RoventTicketValidationKindV2::Busy {
                Ok(false)
            } else {
                Err(invalid(error.error.to_string()))
            };
        }
        let mut disabled = match self.disabled_root.try_lock() {
            Ok(disabled) => disabled,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid("lock-v2 disabled root poisoned"));
            }
        };
        check(disabled.len() < MAX_RETAINED_DISABLED_ROOTS
            && !disabled.iter().any(|marker| marker.authority() == admission.root_admission().authority()),
            "lock-v2 root authority marker duplicate or retained marker capacity")?;
        disabled.push(DisabledCaptureRoot::AfterTicket {
            admission: admission.clone(), first_ticket: *first_ticket,
        });
        Ok(true)
    }
    /// Install an original UCI-issued root capability before any ticket. An
    /// existing marker is never replaced, even if its root has expired.
    pub(super) fn try_install_whole_root_no_capture(
        &self,
        capability: &rz_uci::pals_adaptive::RoventNoCaptureRootCapabilityV2,
    ) -> Result<bool, ArenaError> {
        let mut marker = match self.disabled_root.try_lock() {
            Ok(marker) => marker,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("lock-v2 disabled root poisoned")),
        };
        if marker.iter().any(|entry| matches!(entry,
            DisabledCaptureRoot::WholeRoot(original) if original.same_original_capability(capability))) {
            return Ok(true);
        }
        check(marker.len() < MAX_RETAINED_DISABLED_ROOTS
            && !marker.iter().any(|entry| entry.authority() == capability.authority()),
            "lock-v2 root authority marker duplicate or retained marker capacity")?;
        marker.push(DisabledCaptureRoot::WholeRoot(capability.clone()));
        Ok(true)
    }
    /// The hook calls this only with the original root-run shutdown receipt.
    /// This releases no native owner and does not claim collection success.
    pub(super) fn try_finish_whole_root_no_capture(
        &self,
        capability: &rz_uci::pals_adaptive::RoventNoCaptureRootCapabilityV2,
        receipt: &rz_uci::pals_adaptive::AdaptiveShutdownReceipt,
    ) -> Result<bool, ArenaError> {
        let mut marker = match self.disabled_root.try_lock() {
            Ok(marker) => marker,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("lock-v2 disabled root poisoned")),
        };
        let index = marker.iter().position(|entry| matches!(entry,
            DisabledCaptureRoot::WholeRoot(original)
                if original.same_original_capability(capability) && original.matches_shutdown_receipt(receipt)))
            .ok_or_else(|| invalid("whole-root no-capture marker, original capability or native fence mismatch"))?;
        marker.remove(index);
        Ok(true)
    }

    /// Remove the exact P-C no-capture marker only after its original
    /// RoleOnly root receipt proves native closure for the same capability.
    pub(super) fn try_finish_whole_root_no_capture_role_only(
        &self,
        capability: &rz_uci::pals_adaptive::RoventNoCaptureRootCapabilityV2,
        receipt: &rz_uci::pals_adaptive::RoleOnlyShutdownReceipt,
    ) -> Result<bool, ArenaError> {
        let mut marker = match self.disabled_root.try_lock() {
            Ok(marker) => marker,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(invalid("lock-v2 disabled root poisoned")),
        };
        let index = marker.iter().position(|entry| matches!(entry,
            DisabledCaptureRoot::WholeRoot(original)
                if original.same_original_capability(capability)
                    && original.matches_role_only_shutdown_receipt(receipt)))
            .ok_or_else(|| invalid("P/C no-capture marker, original capability or native fence mismatch"))?;
        marker.remove(index);
        Ok(true)
    }
    pub(super) fn clear_disabled_lock_v2_root(
        &self,
        admission: &rz_uci::pals_adaptive::RoventCollectionRootAdmissionV2,
    ) -> Result<(), ArenaError> {
        let mut marker = self.disabled_root.lock()
            .map_err(|_| invalid("lock-v2 disabled root poisoned"))?;
        let index = marker.iter().position(|entry| matches!(entry,
            DisabledCaptureRoot::AfterTicket { admission: original, .. } if original.same_original_admission(admission)))
            .ok_or_else(|| invalid("lock-v2 disabled marker is absent or belongs to another original root"))?;
        marker.remove(index);
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
        speculative_root: Option<&SpeculativeRootV2>,
    ) -> LockV2RegistrationAttempt {
        if let Err(error) = admission.try_validate_ticket(ticket) {
            return if error.kind == rz_uci::pals_adaptive::RoventTicketValidationKindV2::Busy {
                LockV2RegistrationAttempt::Busy(reservation)
            } else {
                LockV2RegistrationAttempt::Failed(invalid(error.error.to_string()), reservation)
            };
        }
        let mut registered = match self.registered.try_lock() {
            Ok(registered) => registered,
            Err(std::sync::TryLockError::WouldBlock) => {
                return LockV2RegistrationAttempt::Busy(reservation);
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return LockV2RegistrationAttempt::Failed(
                    invalid("Rovent capture registration poisoned"),
                    reservation,
                );
            }
        };
        if let Err(error) = check(
            registered.len() < 16 && !registered.contains_key(&ticket.id),
            "capture input registration duplicate/cap",
        ) {
            return LockV2RegistrationAttempt::Failed(error, reservation);
        }
        // The original registration is held while selecting this exact root.
        // Ticket outcome has its own preallocated atomic cell, so callback
        // handoff never locks or waits and cannot lose a Full/closed identity.
        // The original observer passes its exact root handle. Other active
        // roots and unrelated route-lock contention cannot taint this ticket.
        let route = speculative_root.and_then(|root| {
            if self.owns_route(root) && root.route.admission.same_original_admission(admission) {
                Some(Arc::clone(&root.route))
            } else {
                let _ = root.route.unattributed.fetch_update(Ordering::AcqRel, Ordering::Acquire,
                    |value| Some(value.saturating_add(1)));
                None
            }
        });
        let speculative = if let Some(route) = route {
            let item = match route.tickets.try_lock() {
                Ok(mut tickets) if tickets.len() < route.maximum_tickets
                    && !tickets.contains_key(&ticket.id) => {
                    if tickets.try_reserve(1).is_ok() {
                        let item = Arc::new(SpeculativeTicketV2 { ticket: *ticket, state: AtomicU8::new(0),
                            error: Mutex::new(None), authority: SpeculativeTicketAuthorityV2::new() });
                        tickets.insert(ticket.id, Arc::clone(&item));
                        Some(item)
                    } else { None }
                }
                _ => None,
            };
            match item {
                Some(item) => Some((route, item)),
                None => {
                    let _ = route.unattributed.fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| Some(value.saturating_add(1)));
                    None
                }
            }
        } else { None };
        registered.insert(
            ticket.id,
            RegisteredInputCapture::LockV2 {
                ticket: *ticket,
                reservation,
                admission: admission.clone(),
                speculative,
            },
        );
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
                Some(
                    RegisteredInputCapture::SubmittedV1(_)
                        | RegisteredInputCapture::LockV2 { .. }
                        | RegisteredInputCapture::LockV2DisabledPrior { .. }
                )
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
                RegisteredInputCapture::LockV2 { reservation, speculative, .. } => {
                    if let Some((_, ticket)) = speculative { ticket.state.store(6, Ordering::Release); }
                    Some(reservation.sequence())
                },
                RegisteredInputCapture::LockV2DisabledPrior { .. } => None,
            }))
    }
}
impl RoventCapturedInputReceiver {
    /// The first capture error remains available even after the native owner
    /// continues. It is a collection failure, never NN failure evidence.
    pub fn capture_failure(&self) -> Result<Option<RoventNativeCaptureFailure>, ArenaError> {
        Ok(self
            .failure
            .lock()
            .map_err(|_| invalid("capture failure state poisoned"))?
            .failure
            .clone())
    }
    /// Move the original unsent input to the collection owner for explicit
    /// cleanup. This is neither permission to publish nor a physical fence.
    pub(super) fn take_failed_input_for_cleanup(
        &self,
    ) -> Result<Option<RoventFailedCapturedInput>, ArenaError> {
        Ok(self
            .failure
            .lock()
            .map_err(|_| invalid("capture failure state poisoned"))?
            .input
            .take())
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
            return Err(invalid(format!(
                "native input capture {:?} failed for {:?}: {:?}",
                failure.stage, failure.ticket.id, failure.cause
            )));
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
        self.require_healthy_capture()
            .map_err(|error| Box::new((error, None)))?;
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
        context: &TicketOwnerDispatchContext<
            '_,
            ReviewView,
            NativeTicketQuestion,
            VirtualReviewPreparedTicket,
        >,
    ) -> Result<(), VirtualReviewCaptureFailure> {
        let ticket = context.ticket();
        let preparation = context.question();
        let prepared = context.prepared();
        if prepared.representation != ticket.representation {
            return Err(VirtualReviewCaptureFailure::Required(RoleFailure::Backend(
                "Rovent prepared input representation differs from original ticket".into(),
            )));
        }
        let reservation =
            {
                let mut registered = self.registered.lock().map_err(|_| {
                    RoleFailure::Backend("Rovent capture registration poisoned".into())
                })?;
                let Some(candidate) = registered.get(&ticket.id) else {
                    let disabled = self.disabled_root.lock().map_err(|_| {
                        RoleFailure::Backend("lock-v2 disabled root poisoned".into())
                    })?;
                    let mut matching = 0usize;
                    for marker in disabled.iter() {
                        let matches = match marker {
                            DisabledCaptureRoot::AfterTicket { admission, first_ticket } => {
                                ticket.id.epoch == first_ticket.id.epoch
                                    && ticket.id.sequence >= first_ticket.id.sequence
                                    && admission.validate_native_discard_owner(context).is_ok()
                            }
                            DisabledCaptureRoot::WholeRoot(capability) => capability.validate_native_discard_owner(context).is_ok(),
                        };
                        matching += usize::from(matches);
                    }
                    if matching != 1 {
                        return Err(VirtualReviewCaptureFailure::Required(RoleFailure::Backend(
                            "original native input has no unique same-root no-capture marker".into(),
                        )));
                    }
                    return Ok(());
                };
                if let RegisteredInputCapture::LockV2 {
                    ticket: registered_ticket,
                    admission,
                    ..
                } = candidate
                {
                    if registered_ticket != ticket {
                        return Err(VirtualReviewCaptureFailure::Required(RoleFailure::Backend(
                            "lock-v2 capture ticket differs from the original registration".into(),
                        )));
                    }
                    admission
                        .validate_native_dispatch_owner(context)
                        .map_err(|error| RoleFailure::Backend(error.to_string()))?;
                }
                if let RegisteredInputCapture::LockV2DisabledPrior {
                    ticket: registered_ticket,
                    admission,
                } = candidate
                {
                    if registered_ticket != ticket {
                        return Err(VirtualReviewCaptureFailure::Required(RoleFailure::Backend(
                            "disabled prior ticket differs from original registration".into(),
                        )));
                    }
                    admission
                        .validate_native_discard_owner(context)
                        .map_err(|error| RoleFailure::Backend(error.to_string()))?;
                }
                // A retained failed input prevents only another actual capture,
                // which could otherwise overwrite it. A verified no-capture
                // marker is independently safe and must still be consumed.
                if self.failed_input.is_some()
                    && !matches!(candidate, RegisteredInputCapture::LockV2DisabledPrior { .. })
                {
                    return Err(VirtualReviewCaptureFailure::Required(RoleFailure::Backend(
                        "original failed capture remains owned".into(),
                    )));
                }
                registered.remove(&ticket.id).expect("checked registration")
            };
        match reservation {
            RegisteredInputCapture::LockV2DisabledPrior { .. } => Ok(()),
            RegisteredInputCapture::Legacy(reservation) => {
                let captured = reservation
                    .capture(
                        *ticket,
                        &preparation.view,
                        &prepared.input,
                        &prepared.identity,
                    )
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
                let captured = reservation
                    .capture(
                        *ticket,
                        &preparation.view,
                        &prepared.input,
                        &prepared.identity,
                    )
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
            RegisteredInputCapture::LockV2 {
                reservation,
                admission,
                speculative,
                ..
            } => {
                let captured = reservation
                    .capture(
                        *ticket,
                        &preparation.view,
                        &prepared.input,
                        &prepared.identity,
                    )
                    .map_err(|error| {
                        if let Some((_, ticket)) = speculative.as_ref() {
                            ticket.state.store(5, Ordering::Release);
                        }
                        RoleFailure::Backend(error.to_string())
                    })?;
                let teacher_required = captured.speculative_teacher_required();
                let prepared_speculative = speculative.as_ref().filter(|_| teacher_required).map(|(route, ticket)| {
                    captured.speculative_teacher_query(admission.clone(), &preparation.view, &prepared.input,
                        ticket.authority.clone(), context.original_cancellation(), Arc::clone(&route.root_revoked),
                        Arc::clone(&route.cancel_all), Arc::clone(&route.targets_revoked))
                });
                let input = RoventCapturedLockInputV2 { captured, admission };
                match self.lock_v2_sender.try_send(input) {
                    Ok(()) => {
                        if let Some((route, ticket)) = speculative {
                            if !teacher_required {
                                ticket.state.store(8, Ordering::Release);
                            } else if let Some(result) = prepared_speculative {
                                match result {
                                    Ok(query) => {
                                        let outcome = match route.sender.try_send(query) {
                                            Ok(()) => 1,
                                            Err(mpsc::TrySendError::Full(_)) => 2,
                                            Err(mpsc::TrySendError::Disconnected(_)) => 3,
                                        };
                                        ticket.state.store(outcome, Ordering::Release);
                                    }
                                    Err(error) => {
                                        let recorded = match ticket.error.try_lock() {
                                            Ok(mut slot) => { *slot = Some(error); true }
                                            Err(_) => false,
                                        };
                                        ticket.state.store(if recorded { 4 } else { 7 }, Ordering::Release);
                                    }
                                }
                            }
                        }
                        Ok(())
                    }
                    Err(error) => {
                        if let Some((_, ticket)) = speculative { ticket.state.store(5, Ordering::Release); }
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
        let Ok(mut state) = self.failure.try_lock() else {
            return false;
        };
        if state.failure.is_some() || state.input.is_some() || self.failed_input.is_none() {
            return false;
        }
        state.failure = Some(RoventNativeCaptureFailure {
            stage,
            ticket: *ticket,
            cause: failure.clone(),
        });
        state.input = self.failed_input.take();
        true
    }
    fn submitted(
        &mut self,
        _: &RoleTicketEnvelope,
        _: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        _: &VirtualReviewPreparedTicket,
    ) -> Result<(), RoleFailure> {
        Err(RoleFailure::Unsupported(
            "original ticket service owner required for capture",
        ))
    }
    fn submitted_classified(
        &mut self,
        context: &TicketOwnerDispatchContext<
            '_,
            ReviewView,
            NativeTicketQuestion,
            VirtualReviewPreparedTicket,
        >,
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
        mpsc::TrySendError::Full(input) => (
            RoleFailure::Backend("Rovent captured input channel full".into()),
            input,
        ),
        mpsc::TrySendError::Disconnected(input) => (
            RoleFailure::Backend("Rovent captured input owner disconnected".into()),
            input,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::pals::{RepresentationKey, SearchAuthority};
    use rz_contracts::pals_adaptive::ReviewViewSeal;
    use rz_contracts::{
        ClockDomain, Deadline, GameGeneration, MonotonicTick, OwnerId, PrecisionProfile,
        ProcessEpoch, RootGeneration, StateIdentity, StateRevision,
    };

    #[test]
    fn capture_failure_retains_cause_and_blocks_future_capture_publication() {
        let (_, mut observer, receiver) = rovent_native_capture_channel();
        let authority = SearchAuthority {
            epoch: ProcessEpoch(1),
            game: GameGeneration(2),
            root: RootGeneration(3),
            implementation: Digest([1; 32]),
        };
        let ticket = RoleTicketEnvelope {
            id: RequestId::new(authority.epoch, 4),
            authority,
            question: Digest([2; 32]),
            role: Role::Proposer,
            view: ReviewViewSeal {
                authority,
                view_sequence: 5,
                context_policy: Digest([3; 32]),
                view_digest: Digest([4; 32]),
                root_state: StateIdentity {
                    owner: OwnerId(1),
                    revision: StateRevision(1),
                    semantic: Digest([5; 32]),
                },
                actual_tokens: 0,
                selected_records: 0,
                anchors: 1,
                length_bucket: 8,
            },
            representation: RepresentationKey {
                input: Digest([6; 32]),
                history: Digest([7; 32]),
                candidates: Digest([8; 32]),
                mask_positions: Digest([9; 32]),
                model: Digest([10; 32]),
                encoding: Digest([11; 32]),
                precision: PrecisionProfile::Fp32,
                frozen_epoch: 1,
                record_revision: 0,
                role: Role::Proposer,
            },
            original_deadline: Deadline {
                clock: ClockDomain(authority.epoch),
                at: MonotonicTick(100),
            },
        };
        let cause = RoleFailure::Backend("capture queue full".into());
        // A report cannot authorize continuation without retaining its payload.
        assert!(!observer.preserve_observation_failure(
            VirtualReviewCaptureFailureStage::Submitted,
            &ticket,
            &cause
        ));
        receiver.failure.lock().unwrap().failure = Some(RoventNativeCaptureFailure {
            stage: VirtualReviewCaptureFailureStage::Submitted,
            ticket,
            cause: cause.clone(),
        });
        let failure = receiver.capture_failure().unwrap().unwrap();
        assert_eq!(failure.ticket, ticket);
        assert_eq!(failure.cause, cause);
        assert_eq!(failure.stage, VirtualReviewCaptureFailureStage::Submitted);
        assert!(receiver.take().is_err());
        assert!(!observer.preserve_observation_failure(
            VirtualReviewCaptureFailureStage::Retired,
            &ticket,
            &RoleFailure::InvalidOutput
        ));
        assert_eq!(receiver.capture_failure().unwrap(), Some(failure));
    }

    #[test]
    fn full_capture_queue_returns_the_original_owned_input() {
        let (cause, input) = send_error(mpsc::TrySendError::Full(37));
        assert_eq!(input, 37);
        assert!(matches!(cause, RoleFailure::Backend(_)));
    }
}
