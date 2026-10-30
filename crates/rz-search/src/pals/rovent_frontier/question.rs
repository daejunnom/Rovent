//! Crate-only bridge from the two actual port producers. There is no public
//! trait, raw constructor or serialized representation that can mint this
//! capability. Each smart admission first checks that producer's private
//! original question/view seal. Manager frame liveness is still independent.
use super::FrontierError;
use crate::pals::coordinator::{CoordinatorWork, FrameId, FrameInput, RefinementStage};
use crate::pals::engine::RoleLogicalContext;
use crate::pals::engine::adaptive_port::AdaptivePortQuestion;
use crate::pals::engine::pc_port::PcPortQuestion;
use crate::pals::review_context::ReviewView;
use crate::pals::store::PalsStores;
use rz_contracts::pals::Move16;
use rz_contracts::pals_adaptive::ReviewViewSeal;
use rz_contracts::{Deadline, Digest};
use rz_position::Position;
use std::sync::{Arc, Weak};

pub(super) enum QuestionOwner<'a> {
    Adaptive(&'a Arc<AdaptivePortQuestion>),
    Pc(&'a Arc<PcPortQuestion>),
}
pub(super) enum WeakQuestionOwner {
    Adaptive(Weak<AdaptivePortQuestion>),
    Pc(Weak<PcPortQuestion>),
}
impl WeakQuestionOwner {
    pub(super) fn matches(&self, source: &VerifiedFrontierQuestion<'_>) -> bool {
        match (self, &source.owner) {
            (Self::Adaptive(original), QuestionOwner::Adaptive(actual)) => {
                original.ptr_eq(&Arc::downgrade(actual))
            }
            (Self::Pc(original), QuestionOwner::Pc(actual)) => {
                original.ptr_eq(&Arc::downgrade(actual))
            }
            _ => false,
        }
    }
}

pub(crate) struct VerifiedFrontierQuestion<'a> {
    pub(super) frame: FrameId,
    pub(super) original_first_move: Move16,
    pub(super) original_deadline: Deadline,
    pub(super) stage: RefinementStage,
    pub(super) question_digest: Digest,
    pub(super) view_seal: ReviewViewSeal,
    pub(super) public_context: Digest,
    pub(super) view: &'a Arc<ReviewView>,
    pub(super) position: &'a Position,
    pub(super) context: &'a RoleLogicalContext,
    pub(super) pc_question: Option<&'a PcPortQuestion>,
    owner: QuestionOwner<'a>,
}

impl<'a> VerifiedFrontierQuestion<'a> {
    pub(crate) fn from_adaptive(
        work: &'a CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        stores: &PalsStores,
    ) -> Result<Self, FrontierError> {
        AdaptivePortQuestion::validate_frontier_source(work, stores)
            .map_err(|_| FrontierError::Invalid("actual adaptive port question/view"))?;
        Ok(Self {
            frame: work.frame,
            original_first_move: work.original_first_move,
            original_deadline: work.original_deadline,
            stage: work.input.stage,
            question_digest: work.input.question_digest,
            view_seal: work.input.view_seal,
            public_context: work.input.public_context,
            view: &work.input.view,
            position: &work.input.question.position,
            context: &work.input.question.context,
            pc_question: None,
            owner: QuestionOwner::Adaptive(&work.input.question),
        })
    }

    /// The role-only owner has no CPU frame budget or RefinementMachine.
    /// Borrow its actual immutable input directly, after its private producer
    /// validation, without manufacturing a CoordinatorWork or new allowance.
    pub(crate) fn from_pc_input(
        input: &'a FrameInput<ReviewView, PcPortQuestion>,
        stores: &PalsStores,
    ) -> Result<Self, FrontierError> {
        PcPortQuestion::validate_frontier_input(input, stores)
            .map_err(|_| FrontierError::Invalid("actual R-free port input/view"))?;
        let question = input.question.as_ref();
        let first = question.first_move().ok_or(FrontierError::Invalid(
            "root ranking is not a candidate responsibility",
        ))?;
        let first = rz_contracts::Move::try_from(first)?;
        Ok(Self {
            frame: question.frontier_frame(),
            original_first_move: Move16::encode(first),
            original_deadline: question.original_deadline(),
            stage: input.stage,
            question_digest: input.question_digest,
            view_seal: input.view_seal,
            public_context: input.public_context,
            view: &input.view,
            position: question.position(),
            context: question.context(),
            pc_question: Some(question),
            owner: QuestionOwner::Pc(&input.question),
        })
    }

    pub(super) fn weak_owner(&self) -> WeakQuestionOwner {
        match &self.owner {
            QuestionOwner::Adaptive(owner) => WeakQuestionOwner::Adaptive(Arc::downgrade(owner)),
            QuestionOwner::Pc(owner) => WeakQuestionOwner::Pc(Arc::downgrade(owner)),
        }
    }
}
