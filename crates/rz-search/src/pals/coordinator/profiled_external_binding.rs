//! Actual profiled external dispatch binding, distinct from legacy CPU Jobs.
use super::*;
use crate::external_cpu::{ProfiledExternalDispatch, ProfiledExternalRootSource};
use crate::pals::review_context::ReviewView;
use rz_contracts::pals_adaptive::ReviewViewSeal;
use rz_contracts::rovent::{ExecutionMode, QuestionEnvelope};
use rz_contracts::{ClockDomain, Digest, MonotonicTick};

#[derive(Clone, Debug)]
pub struct CheckedProfiledExternalBinding {
    pub(super) frame: FrameId,
    pub(super) view: ReviewViewSeal,
    pub(super) public_context: Digest,
    pub(super) input_digest: Digest,
    pub(super) question: QuestionEnvelope,
    pub(super) execution: rz_contracts::ExecutionId,
}
impl CheckedProfiledExternalBinding {
    pub(crate) fn attest<Q>(
        work: &CoordinatorWork<ReviewView, Q>,
        source: &ProfiledExternalRootSource,
        dispatch: &ProfiledExternalDispatch,
        mode: ExecutionMode,
        clock: ClockDomain,
        _now: MonotonicTick,
    ) -> Result<Self, CoordinatorError> {
        let question = dispatch.question();
        // A Dispatch was already admitted by its actual worker. This maps
        // historical submission to its source; machine.bind still checks live
        // control before it can authorize a new semantic transition.
        question
            .view
            .validate(work.frame.authority, work.input.view.token_budget())?;
        super::ranking::validate_actual_view(&work.input.view, work.input.view_seal)?;
        if work.input.stage.role().is_some()
            || !mode.has_native_r()
            || source.authority() != work.frame.authority
            || question.original_deadline.clock != clock
            || source.clock().domain != clock
            || question.id.authority != work.frame.authority
            || question.question_digest != work.input.question_digest
            || question.view != work.input.view_seal
            || question.original_first_move != Some(work.original_first_move)
            || question.original_deadline != work.original_deadline
            || question.recipient != rz_contracts::rovent::SearchRecipient::Researcher
            || dispatch.scope().question != question.id
            || dispatch.scope().owner != source.owner()
            || dispatch.scope().execution.epoch != work.frame.authority.epoch
            || dispatch.scope().execution.sequence == 0
            || dispatch.process_identity() != source.process_identity()
            || Digest(work.input.view.record_scope_digest()) != work.input.public_context
        {
            return Err(CoordinatorError::IdentityMismatch(
                "profiled external actual original binding",
            ));
        }
        Ok(Self {
            frame: work.frame,
            view: work.input.view_seal,
            public_context: work.input.public_context,
            input_digest: Digest(work.input.view.digest()),
            question,
            execution: dispatch.scope().execution,
        })
    }
    pub fn question(&self) -> QuestionEnvelope {
        self.question
    }
    pub fn execution(&self) -> rz_contracts::ExecutionId {
        self.execution
    }
    pub fn frame(&self) -> FrameId {
        self.frame
    }
}
