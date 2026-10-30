//! Actual imposed Rules / unfinished-horizon origin, carried to its endpoint.
//! Private checked successors preserve causal origin without forging peers,
//! model opinions or ordinary Candidate/Counter/Repair responsibilities.
use super::*;
use crate::pals::rovent_frontier::{
    CheckedFrontierLine, FrontierError, MateQuestionKind, MissingMateQuestion, UnmetHorizonQuestion,
};

enum GoalOriginKind {
    Mate(Arc<MissingMateQuestion>),
    Horizon(Arc<UnmetHorizonQuestion>),
}
struct PcGoalOrigin {
    kind: GoalOriginKind,
    owner: Arc<()>,
    authority: SearchAuthority,
    root_budget: ContractDigest,
    original_deadline: Deadline,
    prefix: Vec<BoardMove>,
    root_side: Color,
    digest: ContractDigest,
}

pub(super) struct PcGoalLocalRules {
    origin: Arc<PcGoalOrigin>,
    checked: Arc<CheckedFrontierLine>,
    node: usize,
    continuation_end: usize,
    digest: ContractDigest,
}

impl PcGoalLocalRules {
    pub(super) fn prepare<M: RoleModel>(
        port: &mut PcSemanticPort<M>,
        question: MissingMateQuestion,
    ) -> Result<Self, FrontierError> {
        let frontier = port
            .owner
            .semantic
            .rovent_frontier
            .as_ref()
            .ok_or(FrontierError::ForeignRoot)?;
        frontier.validate_missing_question(&question)?;
        let prefix = frontier.missing_question_prefix(&question)?;
        let chunk = frontier.missing_question_chunk_limit(
            &question,
            u16::try_from(port.owner.config.line_plies).map_err(|_| FrontierError::DepthHorizon)?,
        )?;
        let root_budget = frontier.registration().root_budget;
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pc-goal-local-rules-origin/1\0");
        hash.update(root_budget.0);
        hash.update(
            role_line_sha256(b"imposed-root-prefix", &prefix)
                .map_err(|_| FrontierError::ForeignRoot)?,
        );
        hash.update([match question.kind() {
            MateQuestionKind::AttackCandidate => 0,
            MateQuestionKind::RequiredDefense => 1,
        }]);
        hash.update([question.attacker() as u8]);
        hash.update(question.remaining().0.to_le_bytes());
        hash.update(question.original_deadline().at.0.to_le_bytes());
        let (clock, now) = port.frontier_time()?.ok_or(FrontierError::ForeignRoot)?;
        let checked = port
            .owner
            .semantic
            .rovent_frontier
            .as_deref_mut()
            .ok_or(FrontierError::ForeignRoot)?
            .observe_missing_child(&question, clock, now)?;
        Self::from_admitted(
            port,
            GoalOriginKind::Mate(Arc::new(question)),
            prefix,
            chunk,
            root_budget,
            ContractDigest(hash.finalize().into()),
            checked,
        )
    }

    pub(super) fn prepare_horizon<M: RoleModel>(
        port: &mut PcSemanticPort<M>,
        question: UnmetHorizonQuestion,
    ) -> Result<Self, FrontierError> {
        let frontier = port
            .owner
            .semantic
            .rovent_frontier
            .as_ref()
            .ok_or(FrontierError::ForeignRoot)?;
        frontier.validate_unmet_horizon_question(&question)?;
        let prefix = frontier.unmet_horizon_question_prefix(&question)?;
        let chunk = frontier.unmet_horizon_question_chunk_limit(
            &question,
            u16::try_from(port.owner.config.line_plies).map_err(|_| FrontierError::DepthHorizon)?,
        )?;
        if chunk == 0 {
            return Err(FrontierError::DepthHorizon);
        }
        let root_budget = question.root_budget();
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pc-goal-local-horizon-origin/1\0");
        hash.update(root_budget.0);
        hash.update(question.source_question().0);
        hash.update(question.frame().sequence.to_le_bytes());
        hash.update([question.kind() as u8]);
        hash.update(question.target().0.to_le_bytes());
        hash.update(
            role_line_sha256(b"unmet-root-prefix", &prefix)
                .map_err(|_| FrontierError::ForeignRoot)?,
        );
        let (clock, now) = port.frontier_time()?.ok_or(FrontierError::ForeignRoot)?;
        let checked = port
            .owner
            .semantic
            .rovent_frontier
            .as_deref_mut()
            .ok_or(FrontierError::ForeignRoot)?
            .observe_unmet_horizon_anchor(&question, &port.owner.semantic.stores, clock, now)?;
        Self::from_admitted(
            port,
            GoalOriginKind::Horizon(Arc::new(question)),
            prefix,
            chunk,
            root_budget,
            ContractDigest(hash.finalize().into()),
            checked,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_admitted<M: RoleModel>(
        port: &mut PcSemanticPort<M>,
        kind: GoalOriginKind,
        prefix: Vec<BoardMove>,
        chunk: u16,
        root_budget: ContractDigest,
        origin_digest: ContractDigest,
        checked: CheckedFrontierLine,
    ) -> Result<Self, FrontierError> {
        if prefix.is_empty()
            || prefix != checked.prefix()
            || prefix.len() > usize::from(port.owner.profile.maximum_root_prefix_plies)
        {
            return Err(FrontierError::DepthHorizon);
        }
        let continuation_end = prefix
            .len()
            .checked_add(usize::from(chunk))
            .filter(|end| *end <= usize::from(port.owner.profile.maximum_root_prefix_plies))
            .ok_or(FrontierError::DepthHorizon)?;
        let node = port
            .adopt_frontier_line(&checked)
            .map_err(|_| FrontierError::StoreUnavailable)?;
        let origin = Arc::new(PcGoalOrigin {
            kind,
            owner: Arc::clone(&port.identity),
            authority: port.authority,
            root_budget,
            original_deadline: port.deadline,
            prefix,
            root_side: port.owner.semantic.nodes[port.root].position.side_to_move(),
            digest: origin_digest,
        });
        Self::from_checked(origin, checked, node, continuation_end)
    }

    fn from_checked(
        origin: Arc<PcGoalOrigin>,
        checked: CheckedFrontierLine,
        node: usize,
        continuation_end: usize,
    ) -> Result<Self, FrontierError> {
        if !checked.prefix().starts_with(&origin.prefix)
            || checked.prefix().len() > continuation_end
        {
            return Err(FrontierError::ForeignHandle);
        }
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-pc-goal-local-checked-input/1\0");
        hash.update(origin.digest.0);
        hash.update(
            role_line_sha256(b"current-paid-prefix", checked.prefix())
                .map_err(|_| FrontierError::ForeignRoot)?,
        );
        hash.update((continuation_end as u64).to_le_bytes());
        Ok(Self {
            origin,
            checked: Arc::new(checked),
            node,
            continuation_end,
            digest: ContractDigest(hash.finalize().into()),
        })
    }

    /// Only the actual accepted policy and controller-paid single child may
    /// advance this source. No new Missing capability or original view is made.
    pub(super) fn successor(
        &self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        movement: BoardMove,
        checked: CheckedFrontierLine,
        node: usize,
    ) -> Result<Self, PalsError> {
        if input
            .question
            .goal_local
            .as_ref()
            .is_none_or(|held| !std::ptr::eq(held.as_ref(), self))
            || input.question.reciprocal_question().is_some()
            || !accepted.validates_output(output)
            || accepted.ticket().question != input.question_digest
            || accepted.ticket().view != input.view_seal
            || !input.view.query().legal.contains(&movement)
            || checked.prefix().len() != self.prefix().len() + 1
            || !checked.prefix().starts_with(self.prefix())
            || checked.prefix().last() != Some(&movement)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Self::from_checked(
            Arc::clone(&self.origin),
            checked,
            node,
            self.continuation_end,
        )
        .map_err(PalsError::from)
    }

    pub(super) fn root_budget(&self) -> ContractDigest {
        self.origin.root_budget
    }
    pub(super) fn node(&self) -> usize {
        self.node
    }
    pub(super) fn prefix(&self) -> &[BoardMove] {
        self.checked.prefix()
    }
    pub(super) fn digest(&self) -> ContractDigest {
        self.digest
    }
    pub(super) fn origin_digest(&self) -> ContractDigest {
        self.origin.digest
    }
    pub(super) fn continuation_end(&self) -> usize {
        self.continuation_end
    }
    pub(super) fn horizon(&self) -> Option<&UnmetHorizonQuestion> {
        match &self.origin.kind {
            GoalOriginKind::Horizon(question) => Some(question),
            _ => None,
        }
    }
    pub(super) fn frame(&self) -> Option<FrameId> {
        self.horizon().map(|q| q.frame())
    }
    pub(super) fn purpose(&self) -> RoleQueryPurpose {
        if self.continuation_end == self.prefix().len() {
            RoleQueryPurpose::ValueFresh
        } else if self.checked.endpoint_snapshot().side_to_move() == self.origin.root_side {
            RoleQueryPurpose::ProposePolicy
        } else {
            RoleQueryPurpose::ReplyPolicy
        }
    }

    pub(super) fn validates_input(
        &self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        owner: &Arc<()>,
        authority: SearchAuthority,
        root_budget: ContractDigest,
    ) -> Result<(), PalsError> {
        let original_deadline = match &self.origin.kind {
            GoalOriginKind::Mate(question) => question.original_deadline(),
            GoalOriginKind::Horizon(question) => question.original_deadline(),
        };
        if !Arc::ptr_eq(&self.origin.owner, owner)
            || self.origin.authority != authority
            || self.root_budget() != root_budget
            || original_deadline != self.origin.original_deadline
            || original_deadline != input.question.source.original_deadline
            || self.prefix() != input.question.context.prefix
            || !self
                .checked
                .endpoint_snapshot()
                .same_state(&input.question.position.snapshot())
            || input.question.node != self.node
            || input.question.context.purpose != self.purpose()
            || !input.view.query().proposal.is_empty()
            || input.view.query().counterexample.is_some()
            || !input.view.query().divergences.is_empty()
            || input.question.context.proposal_sha256 != role_line_sha256(b"proposal", &[])?
            || input.question.context.refutation_sha256.is_some()
            || !input.question.responsibilities.is_empty()
            || input.question.reciprocal_question().is_some()
            || self
                .frame()
                .is_some_and(|frame| frame != input.question.frontier_frame())
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(())
    }
}
