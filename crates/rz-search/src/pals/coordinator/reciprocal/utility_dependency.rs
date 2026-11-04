//! Read-only custody of an actually admitted reciprocal dependency.
//!
//! A public frame ID, input or transition cannot construct this proof. The
//! manager freezes it only while the original private wait edge and both
//! frames still exist. It records causality, not new work or credit authority.

use super::*;
use crate::pals::coordinator::{AcceptedRoleOrigin, AcceptedRoleOriginKind};
use std::sync::Weak;

/// The actual source parent and canonical child of one manager-owned wait.
/// Weak target references preserve pointer identity without extending any
/// selected ReviewView's Store pin lifetime. No parent view is retained.
pub struct ReciprocalDependencyProof<V, Q> {
    parent: FrameId,
    child: FrameId,
    origin: Option<AcceptedRoleOrigin>,
    question: QuestionEnvelope,
    alias_question: QuestionEnvelope,
    parent_question: Option<QuestionEnvelope>,
    first: Move16,
    deadline: Deadline,
    stage: RefinementStage,
    public_context: Digest,
    target_question: Weak<Q>,
    target_view: Weak<V>,
}

impl<V, Q> ReciprocalDependencyProof<V, Q> {
    pub fn origin(&self) -> Option<AcceptedRoleOrigin> {
        self.origin
    }

    /// A joined waiter receives the original child's canonical envelope,
    /// never an envelope relabelled as a new evaluation of the waiter's view.
    pub fn question(&self) -> QuestionEnvelope {
        self.question
    }

    /// Exact issuance recorded on this parent's private edge. For Joined this
    /// alias can differ from `question()` while sharing its canonical child.
    pub fn alias_question(&self) -> QuestionEnvelope {
        self.alias_question
    }

    pub fn parent(&self) -> FrameId {
        self.parent
    }

    pub fn child(&self) -> FrameId {
        self.child
    }

    /// The original parent's actual reciprocal scope, when it has one. Local
    /// accepted P/C sources remain LocalAccepted and need no fictitious peer.
    pub fn parent_question(&self) -> Option<QuestionEnvelope> {
        self.parent_question
    }

    /// Historical identity check only. This neither reopens the dependency
    /// nor accepts its result or authorizes a new role/CPU reservation.
    pub fn validates_child(&self, work: &CoordinatorWork<V, Q>) -> bool {
        work.frame == self.child
            && work.frame.authority == self.question.id.authority
            && work.original_first_move == self.first
            && work.original_deadline == self.deadline
            && work.input.stage == self.stage
            && work.input.question_digest == self.question.question_digest
            && work.input.view_seal == self.question.view
            && work.input.public_context == self.public_context
            && Weak::ptr_eq(&self.target_question, &Arc::downgrade(&work.input.question))
            && Weak::ptr_eq(&self.target_view, &Arc::downgrade(&work.input.view))
    }
}

impl<V, Q> RefinementMachine<V, Q> {
    /// Freeze only an actual private edge, its original accepted/paused source
    /// and the still-live canonical target. This method never mutates budgets,
    /// revokes a token, forwards a model or extends the original deadline.
    pub fn reciprocal_dependency_proof(
        &self,
        parent: FrameId,
        child: FrameId,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<ReciprocalDependencyProof<V, Q>, CoordinatorError> {
        self.check_control(clock, now)?;
        if parent == child
            || parent.authority != self.authority
            || child.authority != self.authority
        {
            return Err(CoordinatorError::IdentityMismatch(
                "reciprocal dependency proof root/frame",
            ));
        }
        let edge = self
            .dependencies
            .iter()
            .find(|edge| edge.waiter == parent && edge.dependency == child)
            .ok_or(CoordinatorError::MissingEvidence)?;
        let root = self
            .reciprocal
            .as_ref()
            .ok_or(CoordinatorError::MissingEvidence)?;
        let source = self.frame(parent).ok_or(CoordinatorError::UnknownRequest)?;
        let target = self.frame(child).ok_or(CoordinatorError::UnknownRequest)?;
        if source.original_deadline != self.hard_deadline
            || target.original_deadline != self.hard_deadline
            || source.original_first_move != target.original_first_move
            || source.input.view_seal.root_state != target.input.view_seal.root_state
            || source.budget.original_cpu_credit != target.budget.original_cpu_credit
            || source.budget.original_store_byte_limit != target.budget.original_store_byte_limit
            || !self
                .dependent_credit_owners
                .iter()
                .any(|(held, _)| *held == child)
        {
            return Err(CoordinatorError::IdentityMismatch(
                "reciprocal dependency proof original root limits",
            ));
        }
        source
            .input
            .view_seal
            .validate(self.authority, self.token_budget)?;
        target
            .input
            .view_seal
            .validate(self.authority, self.token_budget)?;
        // The first mapping is the real allocation's envelope. Later joined
        // aliases have distinct issuance IDs but point at this same input.
        let question = root
            .questions
            .iter()
            .find_map(|(question, held)| {
                (*held == child
                    && question.question_digest == target.input.question_digest
                    && question.view == target.input.view_seal)
                    .then_some(*question)
            })
            .ok_or(CoordinatorError::MissingEvidence)?;
        question.validate(root.mode, clock, now, self.token_budget)?;
        if question.id.authority != self.authority
            || question.root_budget != root.root_budget
            || question.original_deadline != target.original_deadline
            || question.original_first_move != Some(target.original_first_move)
            || recipient(target.input.stage) != Some(question.recipient)
        {
            return Err(CoordinatorError::IdentityMismatch(
                "reciprocal dependency proof canonical child",
            ));
        }
        let parent_question = self.reciprocal_question(parent);
        let source_recipient =
            recipient(source.input.stage).ok_or(CoordinatorError::MissingEvidence)?;
        if let Some(scope) = parent_question {
            scope.validate(root.mode, clock, now, self.token_budget)?;
            if scope.id.authority != self.authority
                || scope.root_budget != root.root_budget
                || scope.original_deadline != source.original_deadline
                || scope.original_first_move != Some(source.original_first_move)
                || scope.recipient != source_recipient
            {
                return Err(CoordinatorError::IdentityMismatch(
                    "reciprocal dependency proof original parent scope",
                ));
            }
        }
        // Select the exact alias ID stored by the actual admission, not some
        // other local source's same-sender/parent=None matching question.
        let alias = root
            .questions
            .iter()
            .find_map(|(alias, held)| {
                (*held == child && alias.id == edge.question).then_some(*alias)
            })
            .ok_or(CoordinatorError::MissingEvidence)?;
        alias.validate(root.mode, clock, now, self.token_budget)?;
        if !same_work(&alias, &question)
            || alias.sender != source_recipient
            || alias.parent != parent_question.map(|parent| parent.id)
        {
            return Err(CoordinatorError::IdentityMismatch(
                "reciprocal dependency proof original alias",
            ));
        }
        let origin = match edge.origin_seal {
            Some(seal) => {
                let origin = root
                    .accepted_origins
                    .get(&(parent, source.input.question_digest))
                    .ok_or(CoordinatorError::MissingEvidence)?;
                let ticket = origin.ticket();
                let execution = StepExecution::Role(ticket.id);
                let expected_peer = match origin.kind() {
                    AcceptedRoleOriginKind::LocalAccepted => None,
                    AcceptedRoleOriginKind::PeerQuestion(peer) => Some(peer),
                };
                if origin.seal_digest() != seal
                    || origin.frame() != parent
                    || ticket.authority != self.authority
                    || ticket.question != source.input.question_digest
                    || ticket.view != source.input.view_seal
                    || ticket.original_deadline != source.original_deadline
                    || source.input.stage.role() != Some(ticket.role)
                    || source.phase != FramePhase::Waiting(execution)
                    || source.last_execution != Some(execution)
                    || expected_peer != parent_question
                {
                    return Err(CoordinatorError::IdentityMismatch(
                        "reciprocal dependency proof accepted source origin",
                    ));
                }
                Some(*origin)
            }
            None => {
                // Native R asks only from its actual same-owner checkpoint.
                // This is not an AcceptedRoleOrigin, a second CPU grant, or a
                // substitute for the worker/ledger's settled Await receipt.
                let scope = parent_question.ok_or(CoordinatorError::MissingEvidence)?;
                let owner = source.cpu_owner.ok_or(CoordinatorError::MissingEvidence)?;
                if source_recipient != SearchRecipient::Researcher
                    || scope.recipient != SearchRecipient::Researcher
                    || alias.parent != Some(scope.id)
                    || source.phase != FramePhase::Suspended
                    || !matches!(source.last_execution,
                        Some(StepExecution::Cpu { job, .. }) if job == owner.0)
                {
                    return Err(CoordinatorError::IdentityMismatch(
                        "reciprocal dependency proof paused original R source",
                    ));
                }
                None
            }
        };
        Ok(ReciprocalDependencyProof {
            parent,
            child,
            origin,
            question,
            alias_question: alias,
            parent_question,
            first: target.original_first_move,
            deadline: target.original_deadline,
            stage: target.input.stage,
            public_context: target.input.public_context,
            target_question: Arc::downgrade(&target.input.question),
            target_view: Arc::downgrade(&target.input.view),
        })
    }
}
