//! Root-scoped custody for optional V3 speculative teacher queries.
//! This queue is not NN acceptance, physical completion, or a target.
use super::producer_v3::{SpeculativeTicketAuthorityV3, SubmittedTeacherQueryV3};
use super::{ArenaError, RequestId, RoleTicketEnvelope, invalid};
use rz_uci::pals_adaptive::RoventCollectionRootAdmissionV3;
use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    mpsc,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpeculativeHandoffV3 {
    Reserved,
    Sent,
    QueueFull,
    Disconnected,
    PreparationFailed,
    OriginalCaptureFailed,
    Canceled,
    PreparationFailureDetailUnknown,
    NotRequired,
}
impl SpeculativeHandoffV3 {
    fn decode(value: u8) -> Self {
        match value {
            1 => Self::Sent,
            2 => Self::QueueFull,
            3 => Self::Disconnected,
            4 => Self::PreparationFailed,
            5 => Self::OriginalCaptureFailed,
            6 => Self::Canceled,
            7 => Self::PreparationFailureDetailUnknown,
            8 => Self::NotRequired,
            _ => Self::Reserved,
        }
    }
}

pub(super) struct SpeculativeTicketV3 {
    ticket: RoleTicketEnvelope,
    route_owner: Arc<()>,
    state: AtomicU8,
    error: Mutex<Option<ArenaError>>,
    authority: SpeculativeTicketAuthorityV3,
    target_gate: Arc<Mutex<()>>,
    target_capability: Mutex<Option<TargetPublicationV3>>,
}
impl SpeculativeTicketV3 {
    pub(super) fn ticket(&self) -> &RoleTicketEnvelope {
        &self.ticket
    }
    pub(super) fn authority(&self) -> SpeculativeTicketAuthorityV3 {
        self.authority.clone()
    }
    pub(super) fn capture_failed(&self) {
        self.state.store(5, Ordering::Release);
    }
    pub(super) fn try_cancel(&self) -> Result<Option<()>, ArenaError> {
        let _gate = match self.target_gate.try_lock() {
            Ok(gate) => gate,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid("lock-v3 target gate poisoned"));
            }
        };
        self.authority.cancel();
        let _ = self
            .state
            .compare_exchange(0, 6, Ordering::AcqRel, Ordering::Acquire);
        Ok(Some(()))
    }
    pub(super) fn try_take_target_capability(&self) -> Result<TargetCapabilityTakeV3, ArenaError> {
        let mut slot = match self.target_capability.try_lock() {
            Ok(slot) => slot,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(TargetCapabilityTakeV3::Busy),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid("lock-v3 target capability custody poisoned"));
            }
        };
        Ok(match slot.take() {
            Some(capability) => TargetCapabilityTakeV3::Ready(capability),
            None => TargetCapabilityTakeV3::AlreadyTaken,
        })
    }
    pub(super) fn not_required(&self) {
        self.state.store(8, Ordering::Release);
    }
    pub(super) fn preparation_failed(&self, error: ArenaError) {
        let recorded = match self.error.try_lock() {
            Ok(mut slot) => {
                *slot = Some(error);
                true
            }
            Err(_) => false,
        };
        self.state
            .store(if recorded { 4 } else { 7 }, Ordering::Release);
    }
}

pub(super) enum TargetCapabilityTakeV3 {
    Ready(TargetPublicationV3),
    Busy,
    AlreadyTaken,
}

/// Read-only root revocation view. No caller can mutate the shared flag
/// without going through the owning route's publication gate.
#[derive(Clone)]
pub(crate) struct TargetRevocationV3(Arc<AtomicBool>);
impl TargetRevocationV3 {
    pub(crate) fn is_revoked(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    pub(crate) fn try_observe_open(&self) -> bool {
        self.0
            .compare_exchange(false, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

/// Non-clone publication authority for one original root and exact ticket.
pub(crate) struct TargetPublicationV3 {
    admission: RoventCollectionRootAdmissionV3,
    ticket: RoleTicketEnvelope,
    gate: Arc<Mutex<()>>,
    revoked: TargetRevocationV3,
    authority: SpeculativeTicketAuthorityV3,
    state: AtomicU8,
}
impl TargetPublicationV3 {
    pub(crate) fn matches_revocation(&self, revocation: &TargetRevocationV3) -> bool {
        Arc::ptr_eq(&self.revoked.0, &revocation.0)
    }
    pub(crate) fn matches(
        &self,
        admission: &RoventCollectionRootAdmissionV3,
        ticket: &RoleTicketEnvelope,
    ) -> bool {
        self.admission.same_original_admission(admission) && self.ticket == *ticket
    }
    /// None is transient contention. A cancellation, prior commit, or
    /// uncertain previous append is an error and never grants a new attempt.
    pub(crate) fn try_begin_append(&self) -> Result<Option<TargetAppendGuardV3<'_>>, ArenaError> {
        let gate = match self.gate.try_lock() {
            Ok(gate) => gate,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid("lock-v3 target publication gate poisoned"));
            }
        };
        if self.revoked.is_revoked() || self.authority.canceled() {
            return Err(invalid("lock-v3 target revoked before append"));
        }
        if self.state.load(Ordering::Acquire) != 0 {
            return Err(invalid(
                "lock-v3 target append already committed or uncertain",
            ));
        }
        self.state.store(1, Ordering::Release);
        Ok(Some(TargetAppendGuardV3 {
            capability: self,
            _gate: gate,
            resolved: false,
        }))
    }
}

/// The gate remains held across the real in-memory append. Dropping after a
/// panic/unknown result makes subsequent attempts fail rather than guessing.
pub(crate) struct TargetAppendGuardV3<'a> {
    capability: &'a TargetPublicationV3,
    _gate: std::sync::MutexGuard<'a, ()>,
    resolved: bool,
}
impl TargetAppendGuardV3<'_> {
    pub(crate) fn committed(mut self) {
        self.capability.state.store(2, Ordering::Release);
        self.resolved = true;
    }
    pub(crate) fn known_not_written(mut self) {
        self.capability.state.store(0, Ordering::Release);
        self.resolved = true;
    }
}
impl Drop for TargetAppendGuardV3<'_> {
    fn drop(&mut self) {
        if !self.resolved {
            self.capability.state.store(3, Ordering::Release);
        }
    }
}

pub(super) enum SpeculativeTicketRegistrationV3 {
    Ready(Arc<SpeculativeTicketV3>),
    Busy,
    Failed(ArenaError),
}

pub(super) struct SpeculativeRouteV3 {
    pub(super) installed: AtomicBool,
    registration_owner: Arc<()>,
    route_owner: Arc<()>,
    admission: RoventCollectionRootAdmissionV3,
    sender: mpsc::SyncSender<Arc<SubmittedTeacherQueryV3>>,
    tickets: Mutex<HashMap<RequestId, Arc<SpeculativeTicketV3>>>,
    maximum_tickets: usize,
    unattributed: AtomicU64,
    cancel_all: Arc<AtomicBool>,
    root_revoked: Arc<AtomicBool>,
    targets_revoked: TargetRevocationV3,
    target_gate: Arc<Mutex<()>>,
}

/// The same Arc is retained by registration and the external teacher owner.
#[derive(Clone)]
pub(crate) struct SpeculativeRootV3 {
    pub(super) route: Arc<SpeculativeRouteV3>,
}

/// Owns the original queue and its route. Only actual successful receives
/// advance the private handoff count, and no caller can manufacture it.
pub(crate) struct SpeculativeReceiverV3 {
    route: Arc<SpeculativeRouteV3>,
    receiver: mpsc::Receiver<Arc<SubmittedTeacherQueryV3>>,
    received: usize,
    overflowed: bool,
    identity_failed: bool,
    pending_delivery: Option<Arc<SubmittedTeacherQueryV3>>,
}

/// A received capsule is returned intact on identity or counter failure.
pub(crate) enum SpeculativeReceiveErrorV3 {
    Disconnected,
    CounterOverflow(Arc<SubmittedTeacherQueryV3>),
    CounterExhausted,
    ForeignRoot(Arc<SubmittedTeacherQueryV3>),
    PendingDelivery,
}

impl SpeculativeReceiverV3 {
    pub(crate) fn matches_original_route(&self, root: &SpeculativeRootV3) -> bool {
        Arc::ptr_eq(&self.route, &root.route)
    }

    pub(crate) fn try_recv(
        &mut self,
    ) -> Result<Option<Arc<SubmittedTeacherQueryV3>>, SpeculativeReceiveErrorV3> {
        if self.pending_delivery.is_some() {
            return Err(SpeculativeReceiveErrorV3::PendingDelivery);
        }
        if self.overflowed {
            return Err(SpeculativeReceiveErrorV3::CounterExhausted);
        }
        let query = match self.receiver.try_recv() {
            Ok(query) => query,
            Err(mpsc::TryRecvError::Empty) => return Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(SpeculativeReceiveErrorV3::Disconnected);
            }
        };
        // Keep the original Arc before moving another reference to the pump.
        // A panic before custody acknowledgement cannot lose this capsule.
        self.pending_delivery = Some(Arc::clone(&query));
        let Some(next) = self.received.checked_add(1) else {
            self.overflowed = true;
            return Err(SpeculativeReceiveErrorV3::CounterOverflow(query));
        };
        self.received = next;
        if !self
            .route
            .admission
            .same_original_admission(query.admission())
        {
            self.identity_failed = true;
            return Err(SpeculativeReceiveErrorV3::ForeignRoot(query));
        }
        Ok(Some(query))
    }

    /// Inspect the original pending capsule after an interrupted handoff.
    pub(crate) fn pending_delivery(&self) -> Option<Arc<SubmittedTeacherQueryV3>> {
        self.pending_delivery.as_ref().map(Arc::clone)
    }

    /// The caller must first place this exact Arc in the original pump's
    /// external custody. Acknowledgement never authorizes physical reuse.
    pub(crate) fn acknowledge_delivery(
        &mut self,
        query: &Arc<SubmittedTeacherQueryV3>,
    ) -> Result<(), ArenaError> {
        if self.overflowed || self.identity_failed {
            return Err(invalid(
                "lock-v3 failed delivery must remain in original custody",
            ));
        }
        match self.pending_delivery.as_ref() {
            Some(original) if Arc::ptr_eq(original, query) => {
                self.pending_delivery = None;
                Ok(())
            }
            _ => Err(invalid(
                "lock-v3 delivery acknowledgement differs from original capsule",
            )),
        }
    }

    /// This only checks callback-to-receiver accounting for this route; it is
    /// not the external teacher worker's join or physical completion proof.
    pub(super) fn try_handoffs_accounted(
        &self,
        root: &SpeculativeRootV3,
    ) -> Result<Option<bool>, ArenaError> {
        if !self.matches_original_route(root) {
            return Err(invalid(
                "lock-v3 receiver belongs to another original route",
            ));
        }
        if self.overflowed || self.identity_failed || self.pending_delivery.is_some() {
            return Ok(Some(false));
        }
        root.try_handoffs_accounted_count(self.received)
    }
}
impl SpeculativeRootV3 {
    /// Call only after the original registration owner has locked and checked
    /// that this root has no existing route or registered ticket.
    pub(super) fn new(
        admission: RoventCollectionRootAdmissionV3,
        channel_capacity: usize,
        maximum_tickets: usize,
        root_revoked: Arc<AtomicBool>,
        registration_owner: Arc<()>,
    ) -> Result<(Self, SpeculativeReceiverV3), ArenaError> {
        if !(1..=16).contains(&channel_capacity) || !(1..=65_536).contains(&maximum_tickets) {
            return Err(invalid("lock-v3 speculative root channel/ticket capacity"));
        }
        let (sender, receiver) = mpsc::sync_channel(channel_capacity);
        let route = Arc::new(SpeculativeRouteV3 {
            installed: AtomicBool::new(true),
            registration_owner,
            route_owner: Arc::new(()),
            admission,
            sender,
            tickets: Mutex::new(HashMap::new()),
            maximum_tickets,
            unattributed: AtomicU64::new(0),
            cancel_all: Arc::new(AtomicBool::new(false)),
            root_revoked,
            targets_revoked: TargetRevocationV3(Arc::new(AtomicBool::new(false))),
            target_gate: Arc::new(Mutex::new(())),
        });
        Ok((
            Self {
                route: Arc::clone(&route),
            },
            SpeculativeReceiverV3 {
                route,
                receiver,
                received: 0,
                overflowed: false,
                identity_failed: false,
                pending_delivery: None,
            },
        ))
    }
    pub(crate) fn matches_admission(&self, admission: &RoventCollectionRootAdmissionV3) -> bool {
        self.route.admission.same_original_admission(admission)
    }
    pub(crate) fn same_original_route(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.route, &other.route)
    }
    pub(super) fn same_registration_owner(&self, owner: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.route.registration_owner, owner)
    }
    pub(crate) fn matches_ticket_authority(&self, ticket: &RoleTicketEnvelope) -> bool {
        ticket.authority == self.route.admission.root_admission().authority()
    }
    pub(super) fn admission(&self) -> &RoventCollectionRootAdmissionV3 {
        &self.route.admission
    }
    pub(super) fn root_revoked(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.route.root_revoked)
    }
    pub(super) fn root_canceled(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.route.cancel_all)
    }
    pub(super) fn target_revocation(&self) -> TargetRevocationV3 {
        self.route.targets_revoked.clone()
    }
    pub(crate) fn unattributed_count(&self) -> u64 {
        self.route.unattributed.load(Ordering::Acquire)
    }
    pub(super) fn note_unattributed(&self) {
        let _ = self
            .route
            .unattributed
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                Some(n.saturating_add(1))
            });
    }
    /// Registration must hold the original registration lock and verify the
    /// exact admission before adding this ticket to the route.
    pub(super) fn try_register_ticket(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> SpeculativeTicketRegistrationV3 {
        if !self.route.installed.load(Ordering::Acquire) || !self.matches_ticket_authority(ticket) {
            return SpeculativeTicketRegistrationV3::Failed(invalid(
                "lock-v3 speculative root or ticket mismatch",
            ));
        }
        let mut tickets = match self.route.tickets.try_lock() {
            Ok(tickets) => tickets,
            Err(std::sync::TryLockError::WouldBlock) => {
                return SpeculativeTicketRegistrationV3::Busy;
            }
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return SpeculativeTicketRegistrationV3::Failed(invalid(
                    "lock-v3 speculative ticket custody poisoned",
                ));
            }
        };
        if tickets.contains_key(&ticket.id) {
            return SpeculativeTicketRegistrationV3::Failed(invalid(
                "lock-v3 speculative ticket duplicate",
            ));
        }
        if tickets.len() >= self.route.maximum_tickets || tickets.try_reserve(1).is_err() {
            return SpeculativeTicketRegistrationV3::Failed(invalid(
                "lock-v3 speculative ticket capacity",
            ));
        }
        let authority = SpeculativeTicketAuthorityV3::new();
        let capability = TargetPublicationV3 {
            admission: self.route.admission.clone(),
            ticket: *ticket,
            gate: Arc::clone(&self.route.target_gate),
            revoked: self.route.targets_revoked.clone(),
            authority: authority.clone(),
            state: AtomicU8::new(0),
        };
        let item = Arc::new(SpeculativeTicketV3 {
            ticket: *ticket,
            route_owner: Arc::clone(&self.route.route_owner),
            state: AtomicU8::new(0),
            error: Mutex::new(None),
            authority,
            target_gate: Arc::clone(&self.route.target_gate),
            target_capability: Mutex::new(Some(capability)),
        });
        tickets.insert(ticket.id, Arc::clone(&item));
        SpeculativeTicketRegistrationV3::Ready(item)
    }
    /// Preserve the actual ticket's mailbox outcome; successful handoff is
    /// not proof that the teacher has issued or finished a physical query.
    pub(super) fn try_send(
        &self,
        ticket: &Arc<SpeculativeTicketV3>,
        query: Arc<SubmittedTeacherQueryV3>,
    ) -> Result<(), ArenaError> {
        if !Arc::ptr_eq(&ticket.route_owner, &self.route.route_owner) {
            self.note_unattributed();
            return Err(invalid("lock-v3 teacher ticket belongs to another route"));
        }
        if ticket.ticket != query.ticket() || !self.matches_admission(query.admission()) {
            ticket.preparation_failed(invalid(
                "lock-v3 teacher query differs from original ticket or admission",
            ));
            return Err(invalid("lock-v3 teacher query identity mismatch"));
        }
        match self.route.sender.try_send(query) {
            Ok(()) => {
                ticket.state.store(1, Ordering::Release);
                Ok(())
            }
            Err(mpsc::TrySendError::Full(_)) => {
                ticket.state.store(2, Ordering::Release);
                Ok(())
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                ticket.state.store(3, Ordering::Release);
                Ok(())
            }
        }
    }
    /// Typed callback path: contention is retryable with no state change;
    /// unknown or changed tickets are explicit errors, never another root's
    /// cancellation or a guessed terminal result.
    pub(crate) fn try_cancel_ticket(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<Option<()>, ArenaError> {
        if !self.matches_ticket_authority(ticket) {
            return Err(invalid("lock-v3 speculative ticket root mismatch"));
        }
        let tickets = match self.route.tickets.try_lock() {
            Ok(tickets) => tickets,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid("lock-v3 speculative ticket custody poisoned"));
            }
        };
        let item = tickets
            .get(&ticket.id)
            .filter(|item| item.ticket == *ticket)
            .ok_or_else(|| invalid("lock-v3 original speculative ticket missing or changed"))?;
        item.try_cancel()
    }

    /// Take the only publication capability from this exact root/ticket.
    /// The caller must retain it if later preparation or append fails.
    pub(super) fn try_take_target_capability(
        &self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<TargetCapabilityTakeV3, ArenaError> {
        if !self.matches_ticket_authority(ticket) {
            return Err(invalid("lock-v3 target ticket root mismatch"));
        }
        let tickets = match self.route.tickets.try_lock() {
            Ok(tickets) => tickets,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(TargetCapabilityTakeV3::Busy),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid("lock-v3 speculative ticket custody poisoned"));
            }
        };
        let item = tickets
            .get(&ticket.id)
            .filter(|item| item.ticket == *ticket)
            .ok_or_else(|| invalid("lock-v3 original target ticket missing or changed"))?;
        item.try_take_target_capability()
    }
    pub(crate) fn cancel_all(&self) {
        self.route.cancel_all.store(true, Ordering::Release);
    }
    pub(crate) fn try_revoke_all_targets(&self) -> Result<Option<()>, ArenaError> {
        let _gate = match self.route.target_gate.try_lock() {
            Ok(gate) => gate,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid("lock-v3 target revocation gate poisoned"));
            }
        };
        self.route.targets_revoked.0.store(true, Ordering::Release);
        self.route.cancel_all.store(true, Ordering::Release);
        Ok(Some(()))
    }
    /// Used only with the original registration owner lock for rollback.
    pub(super) fn try_has_registered_tickets(&self) -> Result<Option<bool>, ArenaError> {
        match self.route.tickets.try_lock() {
            Ok(tickets) => Ok(Some(!tickets.is_empty())),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                Err(invalid("lock-v3 speculative ticket custody poisoned"))
            }
        }
    }
    /// Counts only callback handoffs. The caller must separately verify the
    /// original root/worker fence, all registrations and the receiver owner.
    fn try_handoffs_accounted_count(&self, received: usize) -> Result<Option<bool>, ArenaError> {
        let tickets = match self.route.tickets.try_lock() {
            Ok(tickets) => tickets,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid("lock-v3 speculative ticket custody poisoned"));
            }
        };
        if self.unattributed_count() != 0 {
            return Ok(Some(false));
        }
        let mut sent = 0usize;
        for ticket in tickets.values() {
            match SpeculativeHandoffV3::decode(ticket.state.load(Ordering::Acquire)) {
                SpeculativeHandoffV3::Reserved
                | SpeculativeHandoffV3::PreparationFailureDetailUnknown => return Ok(Some(false)),
                SpeculativeHandoffV3::Sent => sent = sent.saturating_add(1),
                _ => {}
            }
        }
        Ok(Some(sent == received))
    }
    pub(crate) fn try_observations(
        &self,
    ) -> Result<Option<Vec<(RoleTicketEnvelope, SpeculativeHandoffV3, Option<String>)>>, ArenaError>
    {
        let tickets = match self.route.tickets.try_lock() {
            Ok(tickets) => tickets,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid("lock-v3 speculative ticket custody poisoned"));
            }
        };
        let mut observations = Vec::new();
        observations
            .try_reserve(tickets.len())
            .map_err(|_| invalid("lock-v3 speculative observations allocation"))?;
        for item in tickets.values() {
            let state = SpeculativeHandoffV3::decode(item.state.load(Ordering::Acquire));
            let error = if state == SpeculativeHandoffV3::PreparationFailed {
                match item.error.try_lock() {
                    Ok(error) => error.as_ref().map(ToString::to_string),
                    Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
                    Err(std::sync::TryLockError::Poisoned(_)) => {
                        return Err(invalid("lock-v3 speculative error custody poisoned"));
                    }
                }
            } else {
                None
            };
            observations.push((item.ticket, state, error));
        }
        observations.sort_unstable_by_key(|(ticket, _, _)| ticket.id);
        Ok(Some(observations))
    }
}
