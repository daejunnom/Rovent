//! Opt-in copied scalar facts at the original Search semantic boundaries.
//! No output/model/view payload, callback or coordinator-transition authority.
use super::profiled_external_port::feedback_observation::ProfiledFeedbackRoleEdge;
use super::{BoardMove, LineId, ObservationId, PalsError, RoleError, StateId, StoreError};
use crate::pals::coordinator::{FrameId, RefinementStage};
use rz_contracts::pals::{Role, SearchAuthority};
use rz_contracts::{Deadline, Digest, RequestId};

pub const MAX_ROLE_CONSUMPTION_EVENTS: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveRoleConsumptionStage {
    SemanticCommit,
    ViewConsumption,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveRoleConsumptionOutcome {
    /// The original commit_role returned this exact step evidence. This does
    /// not assert view consumption or accepted coordinator continuation.
    SemanticResult,
    /// Original review_index.consume and first consumed_role_views insert
    /// both succeeded. A later materialization/transition can still fail.
    ConsumedView,
    DuplicateNoNewConsumption,
    Rejected(AdaptiveRoleConsumptionRejection),
}

/// Finite original error classes only. The complete original error is returned
/// unchanged; foreign ticket metadata is never promoted to a successful event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveRoleConsumptionRejection {
    Deadline,
    Canceled,
    ExpiredConsumer,
    StaleConsumer,
    DuplicateAttempt,
    InvalidOutput,
    PhysicalCompletionUnknown,
    RoleCallLimit,
    Capacity,
    OtherOriginalFailure,
}
impl AdaptiveRoleConsumptionRejection {
    pub(super) fn from_error(error: &PalsError) -> Self {
        match error {
            PalsError::Role(RoleError::Deadline) => Self::Deadline,
            PalsError::Role(RoleError::Canceled)
            | PalsError::Store(StoreError::CancelledConsumer) => Self::Canceled,
            PalsError::Store(StoreError::ExpiredConsumer) => Self::ExpiredConsumer,
            PalsError::Store(StoreError::StaleConsumer) => Self::StaleConsumer,
            PalsError::Store(StoreError::AlreadyConsumed)
            | PalsError::Store(StoreError::InvalidEvidence(
                "role semantic commit was already attempted",
            )) => Self::DuplicateAttempt,
            PalsError::Role(RoleError::InvalidOutput) => Self::InvalidOutput,
            PalsError::Role(RoleError::PhysicalCompletionUnknown) => {
                Self::PhysicalCompletionUnknown
            }
            PalsError::RoleCallLimit => Self::RoleCallLimit,
            PalsError::Capacity
            | PalsError::Store(StoreError::Capacity(_))
            | PalsError::Rovent(
                crate::pals::rovent_frontier::FrontierError::PublicRecordCapacity
                | crate::pals::rovent_frontier::FrontierError::PublicReviewIndexCapacity,
            ) => Self::Capacity,
            _ => Self::OtherOriginalFailure,
        }
    }
}

/// Copied by the original Search record producer only after publish_record
/// succeeded. Store-local IDs require the event's original authority; they are
/// not Rules StateIdentity/OwnerId or a neural execution identity.
/// Publication may precede a later semantic commit failure, so this fact alone
/// never asserts SemanticResult, view consumption or machine acceptance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptivePublishedRepairLine {
    pub line: LineId,
    pub observation: ObservationId,
    pub record_revision: u64,
    pub parent_revision: Option<u64>,
    pub supersedes_revision: Option<u64>,
    pub origin_state: StateId,
    pub committed_plies: usize,
    pub first_move: Option<BoardMove>,
}

/// Original successful semantic producer attached this copy to its own
/// continuation. It is diagnostic provenance, never a continuation permit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveRepairLineProvenance {
    pub producer_frame: FrameId,
    pub producer_request: RequestId,
    pub submitted_question: Digest,
    pub submitted_view: Digest,
    pub completed_evidence: Digest,
    pub original_deadline: Deadline,
    pub published_line: AdaptivePublishedRepairLine,
}
/// Copied from the consumer's ORIGINAL submitted opaque Question and selected
/// query. No lookup by equal moves, latest slot, digest or public revision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveRepairLineInput {
    pub refinement_stage: RefinementStage,
    pub provenance: AdaptiveRepairLineProvenance,
    pub model_proposal_sha256: Digest,
    pub model_proposal_plies: usize,
    pub repaired_plies: Option<usize>,
}

/// RequestId is the original role ticket's id. Every field is a copied scalar
/// identity/digest; neither the ReviewView nor RoleOutput is retained here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveRoleConsumptionEvent {
    pub authority: SearchAuthority,
    pub request: RequestId,
    pub role: Role,
    pub question: Digest,
    /// commit_role: exact submitted work question. ViewConsumption: original
    /// committed_role_questions entry, if present. Never substituted from ticket.
    pub submitted_question: Option<Digest>,
    pub ticket_view: Digest,
    pub submitted_view: Digest,
    /// Only commit_role receives the original coordinator work frame.
    pub frame: Option<FrameId>,
    pub stage: AdaptiveRoleConsumptionStage,
    pub outcome: AdaptiveRoleConsumptionOutcome,
    /// Copied from the successful original AdaptivePortStep, never a native
    /// receipt or an observer-generated digest. None on consume-only events.
    pub completed_evidence: Option<Digest>,
    /// Exact attempted_role_commits membership before this call. A true value
    /// does not assert that the earlier attempt succeeded or consumed a view.
    pub prior_semantic_attempt: Option<bool>,
    /// Exact original Repair record publication, if this invocation reached it.
    /// None for an unfinished Repair prefix, a pre-publication error, all other
    /// roles and consume-only events. Never copied from a previous invocation.
    pub published_repair_line: Option<AdaptivePublishedRepairLine>,
    /// Actual submitted input lineage; consume-only events have no Question.
    /// SemanticResult and the original first ConsumedView must be joined by
    /// request/question/view before claiming consumption of this input.
    pub repair_line_input: Option<AdaptiveRepairLineInput>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveRoleConsumptionUnavailable {
    ReservationFailed,
    EventCapacity,
}

/// Root-local fixed record cap. The default port contains None and allocates
/// no journal. Overflow preserves its prefix as incomplete/unavailable and
/// never changes the original search result or error.
#[derive(Debug)]
pub struct AdaptiveRoleConsumptionObservation {
    authority: SearchAuthority,
    events: Vec<AdaptiveRoleConsumptionEvent>,
    unavailable: Option<AdaptiveRoleConsumptionUnavailable>,
    // Same opaque journal, enabled once before the first observed operation.
    // These copied facts neither retain a view nor become a role/R capability.
    feedback_roles: Vec<ProfiledFeedbackRoleEdge>,
    feedback_unavailable: bool,
}
impl AdaptiveRoleConsumptionObservation {
    pub(super) fn new(authority: SearchAuthority) -> Self {
        let mut events = Vec::new();
        let unavailable = events
            .try_reserve_exact(MAX_ROLE_CONSUMPTION_EVENTS)
            .err()
            .map(|_| AdaptiveRoleConsumptionUnavailable::ReservationFailed);
        let mut feedback_roles = Vec::new();
        let feedback_unavailable = feedback_roles
            .try_reserve_exact(MAX_ROLE_CONSUMPTION_EVENTS)
            .is_err();
        Self {
            authority,
            events,
            unavailable,
            feedback_roles,
            feedback_unavailable,
        }
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn events(&self) -> &[AdaptiveRoleConsumptionEvent] {
        &self.events
    }
    pub fn unavailable(&self) -> Option<AdaptiveRoleConsumptionUnavailable> {
        self.unavailable
    }
    pub(super) fn feedback_roles(&self) -> Option<&[ProfiledFeedbackRoleEdge]> {
        (!self.feedback_unavailable && self.unavailable.is_none())
            .then_some(self.feedback_roles.as_slice())
    }
    pub(super) fn record_feedback_role(&mut self, edge: ProfiledFeedbackRoleEdge) {
        if self.feedback_unavailable || self.unavailable.is_some() {
            return;
        }
        if self.feedback_roles.len() == MAX_ROLE_CONSUMPTION_EVENTS {
            self.feedback_unavailable = true;
            return;
        }
        self.feedback_roles.push(edge);
    }
    pub(super) fn record(&mut self, event: AdaptiveRoleConsumptionEvent) {
        if self.unavailable.is_some() {
            return;
        }
        if self.events.len() >= MAX_ROLE_CONSUMPTION_EVENTS {
            self.unavailable = Some(AdaptiveRoleConsumptionUnavailable::EventCapacity);
            return;
        }
        // Reservation was made once at opt-in; this push never grows beyond it.
        self.events.push(event);
    }
}

#[cfg(test)]
mod capacity_tests {
    use super::*;
    use crate::pals::review_context::ReviewError;
    use crate::pals::rovent_frontier::FrontierError;

    #[test]
    fn original_frontier_capacity_causes_do_not_reclassify_generic_review_errors() {
        for cause in [
            FrontierError::PublicRecordCapacity,
            FrontierError::PublicReviewIndexCapacity,
        ] {
            assert_eq!(
                AdaptiveRoleConsumptionRejection::from_error(&PalsError::Rovent(cause)),
                AdaptiveRoleConsumptionRejection::Capacity,
            );
        }
        // Only the original producer joins the bundle refusal to its frontier.
        // A standalone ReviewError value is never that same-controller evidence.
        for cause in [
            ReviewError::BundleRecordCapacity,
            ReviewError::IndexCapacity,
            ReviewError::OwnerMismatch,
        ] {
            assert_eq!(
                AdaptiveRoleConsumptionRejection::from_error(&PalsError::Review(cause)),
                AdaptiveRoleConsumptionRejection::OtherOriginalFailure,
            );
        }
    }
}
