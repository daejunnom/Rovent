//! Actual cross-role P/C questions from linear native accepted heads.
//!
//! This module is intentionally registered only with the port field/call-site
//! unit. A source seal retains no old selected-view pin or model latent.
use super::*;
use crate::pals::coordinator::{
    AcceptedPeerSourceSeal, AcceptedRoleOriginKind, accepted_peer_semantic_digest,
};
use rz_contracts::rovent::{
    ChessPly, QuestionEnvelope, QuestionPurpose, ReciprocalQuestionId, SearchRecipient,
};

#[derive(Clone)]
pub(super) struct PcPeerSource {
    accepted: AcceptedPeerSourceSeal,
    question: Arc<PcQuestionSource>,
}

#[derive(Clone)]
pub(super) struct PcPeerQuestion {
    envelope: QuestionEnvelope,
    anchor: Arc<ContractState>,
    source: PcPeerSource,
}

impl PcPeerSource {
    pub(super) fn freeze(
        input: &FrameInput<ReviewView, PcPortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
    ) -> Result<Self, PalsError> {
        let query = &input.question;
        let origin = accepted
            .origin(query.frontier_frame(), query.reciprocal_question())
            .map_err(|_| RoleError::InvalidOutput)?;
        let seal = AcceptedPeerSourceSeal::freeze(origin, &input.view, output)
            .map_err(|_| RoleError::InvalidOutput)?;
        if origin.ticket().question != input.question_digest
            || origin.ticket().view != input.view_seal
            || origin.ticket().question != query.source.question
            || origin.ticket().view != query.source.view
            || origin.ticket().authority != query.source.authority
            || origin.ticket().original_deadline != query.source.original_deadline
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(Self {
            accepted: seal,
            question: Arc::clone(&query.source),
        })
    }
}

impl PcPeerQuestion {
    pub(super) fn envelope(&self) -> QuestionEnvelope {
        self.envelope
    }

    pub(super) fn validate(
        &self,
        query: &PcPortQuestion,
        view: &ReviewView,
        mode: ExecutionMode,
    ) -> Result<(), PalsError> {
        let source = &query.source;
        let envelope = self.envelope;
        let original = self.source.accepted.origin();
        let parent = match original.kind() {
            AcceptedRoleOriginKind::LocalAccepted => None,
            AcceptedRoleOriginKind::PeerQuestion(parent) => Some(parent.id),
        };
        let sender = recipient_for_role(original.ticket().role)?;
        let recipient = recipient_for_role(role_for_purpose(query.context.purpose)?)?;
        if mode.has_native_r()
            || sender == recipient
            || !Arc::ptr_eq(&source.owner, &self.source.question.owner)
            || source.authority != self.source.question.authority
            || source.original_deadline != self.source.question.original_deadline
            || source.root_budget.is_none()
            || source.root_budget != self.source.question.root_budget
            || envelope.id.authority != source.authority
            || envelope.id.sequence != source.sequence
            || envelope.parent != parent
            || envelope.sender != sender
            || envelope.recipient != recipient
            || envelope.question_digest != source.question
            || envelope.view != source.view
            || envelope.root_budget != source.root_budget.ok_or(RoleError::InvalidOutput)?
            || envelope.original_deadline != source.original_deadline
            || envelope.anchor != self.anchor.snapshot().identity()
            || !self
                .anchor
                .rules()
                .snapshot()
                .same_state(&query.position.snapshot())
            || envelope.anchor_root_ply.0 as usize != query.context.prefix.len()
            || envelope.original_first_move != encode_first(query.first)?
            || envelope.prefix_digest.0 != query.context.prefix_sha256
            || envelope.line_digest.is_some()
            || envelope.line_offset != ChessPly(0)
            || envelope.evidence_revision != query.context.public_revision
            || envelope.purpose != peer_purpose(query.context.purpose)?
            || query.goal_local.is_some()
            || view.query().purpose == RoleQueryPurpose::ValueFresh
        {
            return Err(RoleError::InvalidOutput.into());
        }
        if let Some(parent) = envelope.parent
            && parent.sequence >= envelope.id.sequence
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }
}

impl<M: RoleModel> PcSemanticPort<M> {
    /// The source is an actual consumed native head. This only declares the
    /// next exact peer question and grants no Rules, budget or acceptance.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_peer_question(
        &self,
        source: Option<&PcPeerSource>,
        purpose: RoleQueryPurpose,
        context: &RoleLogicalContext,
        position: &Position,
        view: &ReviewView,
        view_seal: ReviewViewSeal,
        sequence: u64,
        stage: RefinementStage,
        original_first: Option<BoardMove>,
    ) -> Result<Option<PcPeerQuestion>, PalsError> {
        let Some(source) = source else {
            return Ok(None);
        };
        if purpose == RoleQueryPurpose::ValueFresh {
            return Ok(None);
        }
        let sender = recipient_for_role(source.accepted.origin().ticket().role)?;
        let recipient = recipient_for_role(role_for_purpose(purpose)?)?;
        if sender == recipient {
            // A local continuation is not a fictitious P->P question.
            return Ok(None);
        }
        let budget = self
            .original_root_budget()
            .ok_or(RoleError::InvalidOutput)?;
        if !Arc::ptr_eq(&source.question.owner, &self.identity)
            || source.question.authority != self.authority
            || source.question.original_deadline != self.deadline
            || source.question.root_budget != Some(budget)
            || sequence == 0
            || sequence <= source.question.sequence
            || self.mode.has_native_r()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let semantic = accepted_peer_semantic_digest(
            self.peer_semantic_digest(context, view, stage, budget),
            &source.accepted,
        );
        let anchor = Arc::new(
            ContractPosition::new(
                (self.rules_owner_issuer)().map_err(|_| RoleError::InvalidOutput)?,
                position.clone(),
            )
            .export()
            .map_err(|_| RoleError::InvalidOutput)?,
        );
        let envelope = QuestionEnvelope {
            id: ReciprocalQuestionId {
                authority: self.authority,
                sequence,
            },
            parent: match source.accepted.origin().kind() {
                AcceptedRoleOriginKind::LocalAccepted => None,
                AcceptedRoleOriginKind::PeerQuestion(parent) => Some(parent.id),
            },
            sender,
            recipient,
            purpose: peer_purpose(purpose)?,
            anchor: anchor.snapshot().identity(),
            anchor_root_ply: ChessPly(
                u16::try_from(context.prefix.len()).map_err(|_| PalsError::Capacity)?,
            ),
            original_first_move: encode_first(original_first)?,
            prefix_digest: ContractDigest(context.prefix_sha256),
            line_digest: None,
            line_offset: ChessPly(0),
            question_digest: semantic,
            view: view_seal,
            evidence_revision: context.public_revision,
            root_budget: budget,
            original_deadline: self.deadline,
        };
        use crate::contract_time::ContractClock;
        let clock = self.original_frontier_clock()?;
        envelope
            .validate(
                self.mode,
                clock.domain(),
                clock.now().map_err(|_| RoleError::InvalidOutput)?,
                view.token_budget(),
            )
            .map_err(|_| RoleError::InvalidOutput)?;
        Ok(Some(PcPeerQuestion {
            envelope,
            anchor,
            source: source.clone(),
        }))
    }

    fn peer_semantic_digest(
        &self,
        context: &RoleLogicalContext,
        view: &ReviewView,
        stage: RefinementStage,
        budget: ContractDigest,
    ) -> ContractDigest {
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-exact-reciprocal-question/1\0");
        hash.update([stage as u8]);
        hash.update(view.digest());
        hash.update(view.policy_digest());
        hash.update(view.record_scope_digest());
        hash.update(context.focus_prefix_sha256());
        hash.update(self.rules.snapshot().identity().semantic.0);
        hash.update(budget.0);
        let model = self.owner.model_identity().as_bytes();
        hash.update((model.len() as u64).to_le_bytes());
        hash.update(model);
        if let Some(value) = self.owner.model_value_identity() {
            for text in [
                &value.semantics,
                &value.model,
                &value.encoding,
                &value.precision,
            ] {
                hash.update((text.len() as u64).to_le_bytes());
                hash.update(text.as_bytes());
            }
            hash.update(value.model_epoch);
        }
        ContractDigest(hash.finalize().into())
    }
}

fn peer_purpose(purpose: RoleQueryPurpose) -> Result<QuestionPurpose, PalsError> {
    match purpose {
        RoleQueryPurpose::ProposePolicy => Ok(QuestionPurpose::Propose),
        RoleQueryPurpose::ReplyPolicy => Ok(QuestionPurpose::Counter),
        RoleQueryPurpose::DivergencePolicy => Ok(QuestionPurpose::Counter),
        RoleQueryPurpose::RepairPolicy => Ok(QuestionPurpose::Repair),
        _ => Err(RoleError::InvalidOutput.into()),
    }
}

fn recipient_for_role(role: Role) -> Result<SearchRecipient, PalsError> {
    match role {
        Role::Proposer => Ok(SearchRecipient::Proposer),
        Role::Critic => Ok(SearchRecipient::Critic),
        Role::Validator => Err(RoleError::InvalidOutput.into()),
    }
}

fn encode_first(
    movement: Option<BoardMove>,
) -> Result<Option<rz_contracts::pals::Move16>, PalsError> {
    movement
        .map(|movement| {
            rz_contracts::Move::try_from(movement)
                .map(rz_contracts::pals::Move16::encode)
                .map_err(|_| RoleError::InvalidOutput.into())
        })
        .transpose()
}
