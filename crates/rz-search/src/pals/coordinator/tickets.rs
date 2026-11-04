use super::ticket_observation::{
    RoleTicketCumulativeObservation, RoleTicketLifecycleEvent, RoleTicketPhaseOccupancy,
};
use super::{CoordinatorError, ranking::HeldProposerRanking, validator::HeldValidatorRanking};
use rz_contracts::pals::{RepresentationKey, Role, SearchAuthority};
use rz_contracts::pals_adaptive::{ReviewViewSeal, RoleTicketEnvelope};
use rz_contracts::{ClockDomain, Color, Deadline, Digest, MonotonicTick, RequestId};
use rz_runtime::adaptive::CoordinatorPolicy;
use std::collections::VecDeque;
use std::sync::Arc;

pub const MAX_ROLE_TICKETS: usize = 16;

/// Search owns the actual view and verifies this seal against its contents.
/// The registry holds the same Arc until acceptance or physical retirement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoleSubmission {
    pub question: Digest,
    pub view: ReviewViewSeal,
    pub representation: RepresentationKey,
    pub role: Role,
    pub original_deadline: Deadline,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TicketPhase {
    Prepared,
    Submitted,
    PhysicallyRunning,
    Completed,
    Delivered,
    Quarantined,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalRoleCompletion {
    Confirmed,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RoleFailure {
    Canceled,
    Expired,
    Backend(String),
    InvalidOutput,
    PhysicalUnknown,
    Unsupported(&'static str),
}

/// Workers must echo every submitted identity. Native shape/numerical and Rules
/// validation remains the consumer's responsibility before `accept`.
#[derive(Debug)]
pub struct RoleCompletion<O> {
    pub ticket: RoleTicketEnvelope,
    pub physical: PhysicalRoleCompletion,
    pub result: Result<O, RoleFailure>,
}

/// A rejected message is returned with its owned payload; rejection is never a
/// permission to release an input/output whose physical completion is unknown.
#[derive(Debug)]
pub struct RejectedRoleCompletion<O> {
    pub error: CoordinatorError,
    pub completion: RoleCompletion<O>,
}

#[derive(Debug)]
pub struct RoleDispatch<V> {
    pub ticket: RoleTicketEnvelope,
    pub view: Arc<V>,
}

#[derive(Debug)]
pub struct ReadyRole<V, O> {
    pub ticket: RoleTicketEnvelope,
    pub view: Arc<V>,
    /// Revoked/expired results still retire through the normal manager path.
    pub logically_eligible: bool,
    pub result: Result<O, RoleFailure>,
}

pub(super) struct HeldRole<V, O> {
    pub(super) envelope: RoleTicketEnvelope,
    pub(super) view: Arc<V>,
    pub(super) phase: TicketPhase,
    pub(super) revoked: bool,
    result: Option<Result<O, RoleFailure>>,
    pub(super) successful_payload: bool,
    pub(super) proposer_ranking: Option<HeldProposerRanking>,
    pub(super) validator_ranking: Option<HeldValidatorRanking>,
    pub(super) native_acceptance_issued: Option<super::native_acceptance::NativeAcceptanceSeal>,
    pub(super) native_acceptance_confirmed: bool,
    pub(super) reciprocal_output_digest: Option<Digest>,
}

/// Bounded independent questions, with one fixed authority for the whole root.
/// Later submissions neither replace older questions nor alter their views.
pub struct RoleTicketRegistry<V, O> {
    authority: SearchAuthority,
    pub(super) token_budget: u16,
    policy: CoordinatorPolicy,
    capacity: usize,
    next_sequence: u64,
    admission_open: bool,
    revocation_dispatched: bool,
    held: VecDeque<HeldRole<V, O>>,
    completion_order: VecDeque<RequestId>,
    cumulative: RoleTicketCumulativeObservation,
    pub(super) native_acceptance_verifier: Option<super::NativeAcceptanceVerifier>,
}

impl<V, O> RoleTicketRegistry<V, O> {
    pub fn new(
        authority: SearchAuthority,
        token_budget: u16,
        policy: CoordinatorPolicy,
        first_request_sequence: u64,
    ) -> Result<Self, CoordinatorError> {
        Self::new_with_capacity(
            authority,
            token_budget,
            policy,
            first_request_sequence,
            MAX_ROLE_TICKETS,
        )
    }
    /// Apply the registered ceiling to both admission and retained storage.
    /// The old constructor retains its sixteen-question default.
    pub fn new_with_capacity(
        authority: SearchAuthority,
        token_budget: u16,
        policy: CoordinatorPolicy,
        first_request_sequence: u64,
        capacity: usize,
    ) -> Result<Self, CoordinatorError> {
        if !(1..=MAX_ROLE_TICKETS).contains(&capacity) {
            return Err(CoordinatorError::Capacity(
                "registered role ticket capacity",
            ));
        }
        if ![128, 256, 512].contains(&token_budget) || first_request_sequence == 0 {
            return Err(CoordinatorError::Capacity("review token profile"));
        }
        Ok(Self {
            authority,
            token_budget,
            policy,
            capacity,
            next_sequence: first_request_sequence,
            admission_open: true,
            revocation_dispatched: false,
            held: VecDeque::with_capacity(capacity),
            completion_order: VecDeque::with_capacity(capacity),
            cumulative: RoleTicketCumulativeObservation::default(),
            native_acceptance_verifier: None,
        })
    }

    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn occupancy(&self) -> usize {
        self.held.len()
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Retired history is retained as cumulative owner facts. Current phases
    /// require only the bounded <=16 held-ticket scan, never deleted history.
    pub fn cumulative_observation(&self) -> RoleTicketCumulativeObservation {
        let mut observation = self.cumulative;
        observation.current = self.current_phase_occupancy();
        observation
    }
    /// Preserve this high-water value when the same process epoch opens a new
    /// root; a root generation change never resets process RequestId issuance.
    pub fn next_request_sequence(&self) -> u64 {
        self.next_sequence
    }
    pub fn admission_open(&self) -> bool {
        self.admission_open
    }
    pub fn ready_tickets(&self) -> impl Iterator<Item = &RoleTicketEnvelope> {
        self.held
            .iter()
            .filter(|held| held.phase == TicketPhase::Completed)
            .map(|held| &held.envelope)
    }
    pub fn physical_outstanding(&self) -> usize {
        self.held
            .iter()
            .filter(|held| {
                matches!(
                    held.phase,
                    TicketPhase::Submitted
                        | TicketPhase::PhysicallyRunning
                        | TicketPhase::Quarantined
                )
            })
            .count()
    }
    pub fn phase(&self, id: RequestId) -> Option<TicketPhase> {
        self.held
            .iter()
            .find(|held| held.envelope.id == id)
            .map(|held| held.phase)
    }
    pub fn ticket(&self, id: RequestId) -> Option<&RoleTicketEnvelope> {
        self.held
            .iter()
            .find(|held| held.envelope.id == id)
            .map(|held| &held.envelope)
    }

    pub fn admit(
        &mut self,
        submission: RoleSubmission,
        view: Arc<V>,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<RoleTicketEnvelope, CoordinatorError> {
        if !self.admission_open {
            return Err(CoordinatorError::AdmissionClosed);
        }
        if self.held.len() >= self.capacity
            || (self.policy == CoordinatorPolicy::Synchronous && !self.held.is_empty())
        {
            return Err(CoordinatorError::Capacity("pending role tickets"));
        }
        let next = self
            .next_sequence
            .checked_add(1)
            .ok_or(CoordinatorError::SequenceExhausted)?;
        let envelope = RoleTicketEnvelope {
            id: RequestId::new(self.authority.epoch, self.next_sequence),
            authority: self.authority,
            question: submission.question,
            view: submission.view,
            representation: submission.representation,
            role: submission.role,
            original_deadline: submission.original_deadline,
        };
        envelope.validate(clock, now, self.token_budget)?;
        self.held.push_back(HeldRole {
            envelope,
            view,
            phase: TicketPhase::Prepared,
            revoked: false,
            result: None,
            successful_payload: false,
            proposer_ranking: None,
            validator_ranking: None,
            native_acceptance_issued: None,
            native_acceptance_confirmed: false,
            reciprocal_output_digest: None,
        });
        self.next_sequence = next;
        self.cumulative
            .record(RoleTicketLifecycleEvent::Admitted, 1);
        self.refresh_observation();
        Ok(envelope)
    }

    /// Submission itself is distinct from a physical-start notification.
    pub fn take_dispatch(&mut self) -> Option<RoleDispatch<V>> {
        if !self.admission_open {
            return None;
        }
        let held = self
            .held
            .iter_mut()
            .find(|held| held.phase == TicketPhase::Prepared)?;
        held.phase = TicketPhase::Submitted;
        let dispatch = RoleDispatch {
            ticket: held.envelope,
            view: Arc::clone(&held.view),
        };
        self.cumulative
            .record(RoleTicketLifecycleEvent::ServiceDispatched, 1);
        self.refresh_observation();
        Some(dispatch)
    }

    pub fn physical_started(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), CoordinatorError> {
        let held = self.match_ticket_mut(ticket)?;
        if held.phase != TicketPhase::Submitted {
            return Err(CoordinatorError::InvalidTransition(
                "physical start requires submitted ticket",
            ));
        }
        held.phase = TicketPhase::PhysicallyRunning;
        self.cumulative
            .record(RoleTicketLifecycleEvent::PhysicalStartNotification, 1);
        self.refresh_observation();
        Ok(())
    }

    #[expect(
        clippy::result_large_err,
        reason = "return the original bounded ticket completion intact under failure/memory pressure without a new allocation"
    )]
    pub fn complete(
        &mut self,
        completion: RoleCompletion<O>,
    ) -> Result<(), RejectedRoleCompletion<O>> {
        let validation = self.match_ticket_mut(&completion.ticket).and_then(|held| {
            if !matches!(
                held.phase,
                TicketPhase::Submitted | TicketPhase::PhysicallyRunning
            ) {
                return Err(CoordinatorError::Duplicate);
            }
            Ok(())
        });
        if let Err(error) = validation {
            return Err(RejectedRoleCompletion { error, completion });
        }
        let newly_revoked = if completion.physical == PhysicalRoleCompletion::Unknown {
            self.held.iter().filter(|held| !held.revoked).count() as u64
        } else {
            0
        };
        let held = self
            .held
            .iter_mut()
            .find(|held| held.envelope.id == completion.ticket.id)
            .expect("validated held ticket");
        held.successful_payload = completion.result.is_ok();
        held.result = Some(completion.result);
        if completion.physical == PhysicalRoleCompletion::Unknown {
            held.phase = TicketPhase::Quarantined;
            held.revoked = true;
            // Quarantine closes admission. The original view/output stay owned.
            self.admission_open = false;
            for outstanding in &mut self.held {
                outstanding.revoked = true;
            }
            self.cumulative
                .record(RoleTicketLifecycleEvent::PhysicalUnknownEvent, 1);
            self.cumulative
                .record(RoleTicketLifecycleEvent::Revoked, newly_revoked);
        } else {
            held.phase = TicketPhase::Completed;
            self.completion_order.push_back(held.envelope.id);
            self.cumulative
                .record(RoleTicketLifecycleEvent::PhysicallyCompleted, 1);
        }
        self.refresh_observation();
        Ok(())
    }

    pub fn take_ready(
        &mut self,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Option<ReadyRole<V, O>> {
        let id = match self.policy {
            CoordinatorPolicy::Synchronous | CoordinatorPolicy::Fixed => {
                self.held.front()?.envelope.id
            }
            CoordinatorPolicy::Adaptive => *self.completion_order.front()?,
        };
        self.take_ready_id(id, clock, now).ok().flatten()
    }

    /// Adaptive callers may use a bounded deadline/dependency/aging selector.
    /// Fixed mode refuses an out-of-submission-order selection.
    pub fn take_ready_id(
        &mut self,
        id: RequestId,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Option<ReadyRole<V, O>>, CoordinatorError> {
        if self.policy != CoordinatorPolicy::Adaptive
            && self.held.front().is_some_and(|held| held.envelope.id != id)
        {
            return Err(CoordinatorError::InvalidTransition("fixed event order"));
        }
        let held = self
            .held
            .iter_mut()
            .find(|held| held.envelope.id == id)
            .ok_or(CoordinatorError::UnknownRequest)?;
        if held.phase != TicketPhase::Completed {
            return Ok(None);
        }
        let logically_eligible = !held.revoked
            && held
                .envelope
                .validate(clock, now, self.token_budget)
                .is_ok();
        held.phase = TicketPhase::Delivered;
        self.completion_order.retain(|completed| *completed != id);
        let ready = ReadyRole {
            ticket: held.envelope,
            view: Arc::clone(&held.view),
            logically_eligible,
            result: held.result.take().expect("completed ticket owns result"),
        };
        self.cumulative
            .record(RoleTicketLifecycleEvent::Delivered, 1);
        self.refresh_observation();
        Ok(Some(ready))
    }

    /// A successful return proves logical acceptance only. The caller already
    /// supplied a physical-complete result and must validate Rules/shape first.
    pub fn accept(
        &mut self,
        ticket: &RoleTicketEnvelope,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<(), CoordinatorError> {
        let held = self.match_ticket_mut(ticket)?;
        if held.phase != TicketPhase::Delivered || held.revoked || !held.successful_payload {
            return Err(CoordinatorError::InvalidTransition(
                "accept requires eligible delivered ticket",
            ));
        }
        ticket.validate(clock, now, self.token_budget)?;
        self.retire_accepted(ticket.id);
        Ok(())
    }

    /// Known completed rejected/error outputs are retired without consumption.
    pub fn discard_delivered(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), CoordinatorError> {
        let held = self.match_ticket_mut(ticket)?;
        if held.phase != TicketPhase::Delivered {
            return Err(CoordinatorError::InvalidTransition(
                "discard requires delivered result",
            ));
        }
        self.retire(ticket.id);
        Ok(())
    }

    /// Return IDs whose workers still need cancellation. Prepared work has not
    /// acquired any physical lease and can be retired immediately.
    pub fn revoke_all(&mut self) -> Vec<RequestId> {
        if self.revocation_dispatched {
            return Vec::new();
        }
        self.revocation_dispatched = true;
        self.admission_open = false;
        let newly_revoked = self.held.iter().filter(|held| !held.revoked).count() as u64;
        let before = self.held.len();
        let mut cancel = Vec::with_capacity(self.held.len());
        for held in &mut self.held {
            held.revoked = true;
            if matches!(
                held.phase,
                TicketPhase::Submitted | TicketPhase::PhysicallyRunning
            ) {
                cancel.push(held.envelope.id);
            }
        }
        self.held.retain(|held| held.phase != TicketPhase::Prepared);
        let released = (before - self.held.len()) as u64;
        self.cumulative
            .record(RoleTicketLifecycleEvent::Revoked, newly_revoked);
        self.cumulative.record(
            RoleTicketLifecycleEvent::TerminalBeforeServiceDispatch,
            released,
        );
        self.cumulative
            .record(RoleTicketLifecycleEvent::Released, released);
        self.refresh_observation();
        cancel
    }

    /// Only an actual worker fence may release a quarantined ticket. It cannot
    /// resurrect its logical authority or reuse its result as a fresh output.
    pub fn retire_after_physical_fence(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), CoordinatorError> {
        let held = self.match_ticket_mut(ticket)?;
        if held.phase != TicketPhase::Quarantined {
            return Err(CoordinatorError::InvalidTransition(
                "fence retirement requires quarantine",
            ));
        }
        self.retire(ticket.id);
        Ok(())
    }

    /// Owner cleanup hook after the actual native service has reported a known
    /// physical shutdown and its thread join has succeeded. This method does
    /// not produce or certify either fact; the single-writer manager must keep
    /// those original receipts. It never converts canceled work to consumption.
    /// Unknown-completion quarantine remains sticky and is not released here.
    pub fn retire_revoked_after_worker_join(
        &mut self,
    ) -> Result<Vec<RoleTicketEnvelope>, CoordinatorError> {
        if self.admission_open || !self.revocation_dispatched {
            return Err(CoordinatorError::InvalidTransition(
                "owner join retirement requires revoked admission",
            ));
        }
        if self
            .held
            .iter()
            .any(|held| !held.revoked || held.phase == TicketPhase::Quarantined)
        {
            return Err(CoordinatorError::InvalidTransition(
                "owner join cannot normalize unknown physical completion",
            ));
        }
        let retired = self.held.iter().map(|held| held.envelope).collect();
        let released = self.held.len() as u64;
        self.held.clear();
        self.completion_order.clear();
        self.cumulative
            .record(RoleTicketLifecycleEvent::Released, released);
        self.refresh_observation();
        Ok(retired)
    }

    pub(super) fn match_ticket_mut(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Result<&mut HeldRole<V, O>, CoordinatorError> {
        if ticket.authority != self.authority {
            return Err(CoordinatorError::IdentityMismatch("root authority"));
        }
        let held = self
            .held
            .iter_mut()
            .find(|held| held.envelope.id == ticket.id)
            .ok_or(CoordinatorError::UnknownRequest)?;
        if held.envelope != *ticket {
            return Err(CoordinatorError::IdentityMismatch(
                "submitted question/view/representation/deadline",
            ));
        }
        Ok(held)
    }
    pub(super) fn retire_accepted(&mut self, id: RequestId) {
        if self.retire_held(id) {
            self.cumulative
                .record(RoleTicketLifecycleEvent::LogicallyAccepted, 1);
        }
    }

    pub(super) fn retire(&mut self, id: RequestId) {
        self.retire_held(id);
    }

    fn retire_held(&mut self, id: RequestId) -> bool {
        let before = self.held.len();
        self.held.retain(|held| held.envelope.id != id);
        self.completion_order.retain(|completed| *completed != id);
        if self.held.len() == before {
            return false;
        }
        self.cumulative
            .record(RoleTicketLifecycleEvent::Released, 1);
        self.refresh_observation();
        true
    }

    fn current_phase_occupancy(&self) -> RoleTicketPhaseOccupancy {
        let mut current = RoleTicketPhaseOccupancy::default();
        for held in &self.held {
            current.observe_phase(held.phase);
        }
        current
    }

    fn refresh_observation(&mut self) {
        let current = self.current_phase_occupancy();
        self.cumulative.observe_occupancy(current);
    }
}

/// Endpoint inputs may differ, while every comparison uses the identical
/// frozen selected view, model, encoding, precision, epoch and perspective.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FreshPairSeal {
    /// VR owner computes exact anchor/record-order/line/relation/context-policy
    /// contents, excluding endpoint query/candidate/length-bucket features.
    pub public_context: Digest,
    pub authority: SearchAuthority,
    pub root_state: rz_contracts::StateIdentity,
    pub context_policy: Digest,
    pub selected_records: u16,
    pub anchors: u8,
    pub model: Digest,
    pub encoding: Digest,
    pub precision: rz_contracts::PrecisionProfile,
    pub frozen_epoch: u64,
    pub record_revision: u64,
    pub role: Role,
    pub perspective: Color,
}
impl FreshPairSeal {
    pub fn from_ticket(
        ticket: &RoleTicketEnvelope,
        public_context: Digest,
        perspective: Color,
    ) -> Self {
        Self {
            public_context,
            authority: ticket.authority,
            root_state: ticket.view.root_state,
            context_policy: ticket.view.context_policy,
            selected_records: ticket.view.selected_records,
            anchors: ticket.view.anchors,
            model: ticket.representation.model,
            encoding: ticket.representation.encoding,
            precision: ticket.representation.precision,
            frozen_epoch: ticket.representation.frozen_epoch,
            record_revision: ticket.representation.record_revision,
            role: ticket.role,
            perspective,
        }
    }
    pub fn validates_public_view(&self, view: &ReviewViewSeal, public_context: Digest) -> bool {
        self.public_context == public_context
            && self.authority == view.authority
            && self.root_state == view.root_state
            && self.context_policy == view.context_policy
            && self.selected_records == view.selected_records
            && self.anchors == view.anchors
    }
    pub fn validate_pair(&self, other: &Self) -> Result<(), CoordinatorError> {
        if self != other {
            return Err(CoordinatorError::IdentityMismatch(
                "Fresh pair frozen view/model/precision/perspective",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "ticket_tests.rs"]
mod tests;
