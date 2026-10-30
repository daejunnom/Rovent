//! A continuation authority for an existing, unfinished selective-depth duty.
//!
//! This module creates no frame, CPU job, model opinion, Rules transition or
//! completed depth. A real shallow Fresh acceptance is retained on its original
//! duty. Only that controller-owned progress can issue the next opaque question.
//! The ordinary acceptance and physical-lifetime contracts remain independent.
use super::*;
use crate::pals::engine::adaptive_port::GoalLocalInput;
use rz_contracts::RequestId;

/// An observation can be used without having reached the root's depth target.
/// Callers retain semantic consumption for all four dispositions. Only Closed
/// closes the original Candidate/Counter/Repair responsibility.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontierHorizonAcceptance {
    ProgressAccepted,
    AlreadyAcceptedProgressExact,
    Closed,
    AlreadyClosedExact,
}
impl FrontierHorizonAcceptance {
    pub fn closed(self) -> bool {
        matches!(self, Self::Closed | Self::AlreadyClosedExact)
    }
}

/// A copied key is a pending-set identity and grants no Rules/NN authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct UnmetHorizonQuestionKey {
    owner: u64,
    obligation: usize,
    progress_request: RequestId,
    progress_output: Digest,
}

/// Only this controller constructs an unmet-depth capability. It retains the
/// exact original obligation and an immutable accepted progress seal, without
/// keeping a chain of ReviewView Store leases alive.
pub struct UnmetHorizonQuestion {
    owner: u64,
    obligation: FrontierObligation,
    frame: FrameId,
    kind: FrontierFrameKind,
    first: BoardMove,
    source_question: Digest,
    source_path: FrontierPath,
    pc_source: bool,
    horizon: HorizonObligationId,
    anchor: FrontierPath,
    target: ChessPly,
    original_deadline: Deadline,
    root_budget: Digest,
    progress: HorizonProgress,
}
impl UnmetHorizonQuestion {
    pub fn key(&self) -> UnmetHorizonQuestionKey {
        UnmetHorizonQuestionKey {
            owner: self.owner,
            obligation: self.obligation.slot,
            progress_request: self.progress.ticket.id,
            progress_output: self.progress.output,
        }
    }
    pub fn kind(&self) -> FrontierFrameKind {
        self.kind
    }
    pub fn frame(&self) -> FrameId {
        self.frame
    }
    pub fn original_first_move(&self) -> BoardMove {
        self.first
    }
    pub fn source_question(&self) -> Digest {
        self.source_question
    }
    pub fn anchor(&self) -> FrontierPath {
        self.anchor
    }
    pub fn target(&self) -> ChessPly {
        self.target
    }
    pub fn original_deadline(&self) -> Deadline {
        self.original_deadline
    }
    pub fn root_budget(&self) -> Digest {
        self.root_budget
    }
}

struct HorizonProgress {
    scope: CompletionScope,
    ticket: RoleTicketEnvelope,
    output: Digest,
    physical: FreshPhysicalSeal,
}

/// Only concrete actual port producers can build this borrowed endpoint.
/// Its frame is the existing depth duty, not a manager admission for this
/// work=None role invocation. There is no public trait or raw constructor.
struct VerifiedHorizonEndpoint<'a> {
    source: &'a UnmetHorizonQuestion,
    scope: CompletionScope,
    position: &'a Position,
    context: &'a super::super::engine::RoleLogicalContext,
    view: &'a Arc<ReviewView>,
}
impl<'a> VerifiedHorizonEndpoint<'a> {
    fn from_adaptive(
        input: &'a GoalLocalInput,
        stores: &PalsStores,
        endpoint: FrontierPath,
    ) -> Result<Self, FrontierError> {
        let source = input.validated_horizon_source(stores)?;
        if source.pc_source {
            return Err(FrontierError::MissingAcceptance);
        }
        let actual = input.input();
        Ok(Self {
            source,
            scope: goal_scope(source, actual, endpoint)?,
            position: &actual.question.position,
            context: &actual.question.context,
            view: &actual.view,
        })
    }

    fn from_pc(
        input: &'a FrameInput<ReviewView, PcPortQuestion>,
        stores: &PalsStores,
        endpoint: FrontierPath,
    ) -> Result<Self, FrontierError> {
        let source = PcPortQuestion::validated_horizon_source(input, stores)
            .map_err(|_| FrontierError::MissingAcceptance)?;
        if !source.pc_source {
            return Err(FrontierError::MissingAcceptance);
        }
        let first = rz_contracts::pals::Move16::encode(rz_contracts::Move::try_from(source.first)?);
        Ok(Self {
            source,
            scope: CompletionScope {
                question: WeakQuestionOwner::Pc(Arc::downgrade(&input.question)),
                view: Arc::downgrade(&input.view),
                frame: source.frame,
                stage: input.stage,
                first,
                original_deadline: source.original_deadline,
                question_digest: input.question_digest,
                view_seal: input.view_seal,
                public_context: input.public_context,
                endpoint,
            },
            position: input.question.position(),
            context: input.question.context(),
            view: &input.view,
        })
    }
}
impl HorizonProgress {
    fn from_completion(completion: &FrontierCompletionSeal) -> Option<Self> {
        let FrontierCompletionSeal::Fresh {
            scope,
            ticket,
            output,
            physical,
        } = completion
        else {
            return None;
        };
        Some(Self {
            scope: clone_scope(scope),
            ticket: *ticket,
            output: *output,
            physical: *physical,
        })
    }
    fn matches_completion(&self, completion: &FrontierCompletionSeal) -> bool {
        match completion {
            FrontierCompletionSeal::Fresh {
                scope,
                ticket,
                output,
                physical,
            } => {
                same_scope(&self.scope, scope)
                    && self.ticket == *ticket
                    && self.output == *output
                    && self.physical == *physical
            }
            _ => false,
        }
    }
}

fn clone_scope(source: &CompletionScope) -> CompletionScope {
    CompletionScope {
        question: match &source.question {
            WeakQuestionOwner::Adaptive(owner) => WeakQuestionOwner::Adaptive(owner.clone()),
            WeakQuestionOwner::Pc(owner) => WeakQuestionOwner::Pc(owner.clone()),
        },
        view: source.view.clone(),
        frame: source.frame,
        stage: source.stage,
        first: source.first,
        original_deadline: source.original_deadline,
        question_digest: source.question_digest,
        view_seal: source.view_seal,
        public_context: source.public_context,
        endpoint: source.endpoint,
    }
}

fn same_scope(left: &CompletionScope, right: &CompletionScope) -> bool {
    let same_question = match (&left.question, &right.question) {
        (WeakQuestionOwner::Adaptive(left), WeakQuestionOwner::Adaptive(right)) => {
            left.ptr_eq(right)
        }
        (WeakQuestionOwner::Pc(left), WeakQuestionOwner::Pc(right)) => left.ptr_eq(right),
        _ => false,
    };
    same_question
        && left.view.ptr_eq(&right.view)
        && left.frame == right.frame
        && left.stage == right.stage
        && left.first == right.first
        && left.original_deadline == right.original_deadline
        && left.question_digest == right.question_digest
        && left.view_seal == right.view_seal
        && left.public_context == right.public_context
        && left.endpoint == right.endpoint
}

impl RoventFrontier {
    /// New depth-aware consumer. The legacy strict/once endpoint APIs keep
    /// their existing completion-only behavior and are not silently widened.
    #[allow(clippy::too_many_arguments)]
    pub fn accept_local_fresh_horizon_once(
        &mut self,
        obligation: FrontierObligation,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        stores: &PalsStores,
        endpoint: FrontierPath,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierHorizonAcceptance, FrontierError> {
        let question = VerifiedFrontierQuestion::from_adaptive(work, stores)?;
        self.accept_verified_fresh_horizon_once(
            obligation, &question, stores, endpoint, accepted, output, physical, clock, now,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accept_verified_fresh_horizon_once(
        &mut self,
        obligation: FrontierObligation,
        question: &VerifiedFrontierQuestion<'_>,
        stores: &PalsStores,
        endpoint: FrontierPath,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierHorizonAcceptance, FrontierError> {
        let started = Instant::now();
        if clock != self.registration.original_deadline.clock {
            return Err(FrontierError::ForeignRoot);
        }
        self.check_verified_work(question, stores)?;
        let at = self.obligation_slot(obligation, question.frame)?;
        let pc_source = self.check_completion_source(at, question)?;
        if !(if pc_source {
            any_fresh_stage(question.stage)
        } else {
            fresh_stage_matches(self.frames[at].kind, question.stage)
        }) {
            return Err(FrontierError::MissingAcceptance);
        }
        self.check_verified_endpoint(endpoint, question)?;
        let fresh_boundary =
            self.checked_fresh_boundary(question, endpoint, accepted, output, physical)?;
        let rules = Position::from_snapshot(self.path_snapshot(endpoint)?, Default::default())?;
        let terminal_boundary = HorizonBoundary::rules_terminal(&rules).ok();
        let rules_terminal = terminal_boundary.is_some();
        let boundary = terminal_boundary.unwrap_or(fresh_boundary);
        let completion = FrontierCompletionSeal::Fresh {
            scope: CompletionScope::new(question, endpoint),
            ticket: *accepted.ticket(),
            output: accepted.output_digest(),
            physical: physical.into(),
        };
        if let Some(disposition) = self.exact_horizon_retry(at, &completion)? {
            return Ok(disposition);
        }
        self.check_control(clock, now)?;
        if self.frames[at].finished {
            return Err(FrontierError::Duplicate);
        }
        self.check_horizon_progress_lineage(at, endpoint)?;
        self.check_control(
            clock,
            self.original_tick_after_elapsed(clock, now, started.elapsed())?,
        )?;
        if !rules_terminal && self.horizon_endpoint_is_shallow(at, endpoint)? {
            self.frames[at].completion = Some(completion);
            return Ok(FrontierHorizonAcceptance::ProgressAccepted);
        }
        self.finish_obligation(at, endpoint, boundary)?;
        self.frames[at].completion = Some(completion);
        Ok(FrontierHorizonAcceptance::Closed)
    }

    /// Bounded descriptors only. No node, NN input or completed depth is
    /// created by inspecting the original unfinished responsibilities.
    pub fn unmet_horizon_questions(
        &mut self,
        maximum: usize,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Vec<UnmetHorizonQuestion>, FrontierError> {
        self.check_control(clock, now)?;
        if maximum > self.limits.maximum_frames {
            return Err(FrontierError::Capacity("unmet-horizon result bound"));
        }
        let mut result = Vec::new();
        result
            .try_reserve(maximum)
            .map_err(|_| FrontierError::Capacity("unmet-horizon result"))?;
        let Some(target) = self
            .registration
            .goal
            .depth_plies
            .map(|depth| ChessPly(depth.get()))
        else {
            return Ok(result);
        };
        if maximum == 0 {
            return Ok(result);
        }
        for (slot, frame) in self.frames.iter().enumerate() {
            let Some(horizon) = frame.horizon else {
                continue;
            };
            if frame.finished {
                continue;
            }
            let Some(progress) = frame
                .completion
                .as_ref()
                .and_then(HorizonProgress::from_completion)
            else {
                // No accepted endpoint exists yet. Normal work retains its
                // responsibility; lack of evidence is not invented progress.
                continue;
            };
            if self.path(progress.scope.endpoint)?.anchor.ply() >= target {
                return Err(FrontierError::Invalid("unfinished reached horizon"));
            }
            self.check_horizon_progress_lineage(slot, progress.scope.endpoint)?;
            result.push(UnmetHorizonQuestion {
                owner: self.owner,
                obligation: FrontierObligation {
                    owner: self.owner,
                    slot,
                },
                frame: frame.frame,
                kind: frame.kind,
                first: frame.original_first,
                source_question: frame.question,
                source_path: frame.source_path,
                pc_source: frame.pc_source,
                horizon,
                anchor: progress.scope.endpoint,
                target,
                original_deadline: self.registration.original_deadline,
                root_budget: self.registration.root_budget,
                progress,
            });
            if result.len() == maximum {
                break;
            }
        }
        Ok(result)
    }

    /// Current-question admission is stricter than frozen-origin validation:
    /// after another actual endpoint advances this duty, its old descriptor
    /// cannot start another independent continuation.
    pub fn validate_unmet_horizon_question(
        &self,
        question: &UnmetHorizonQuestion,
    ) -> Result<(), FrontierError> {
        let at = self.validate_horizon_origin(question)?;
        if self.frames[at].finished
            || self.frames[at]
                .completion
                .as_ref()
                .is_none_or(|completion| !question.progress.matches_completion(completion))
        {
            return Err(FrontierError::ForeignHandle);
        }
        Ok(())
    }

    pub fn unmet_horizon_question_prefix(
        &self,
        question: &UnmetHorizonQuestion,
    ) -> Result<Vec<BoardMove>, FrontierError> {
        self.validate_unmet_horizon_question(question)?;
        self.path_prefix(question.anchor)
    }

    pub fn unmet_horizon_question_chunk_limit(
        &self,
        question: &UnmetHorizonQuestion,
        configured: u16,
    ) -> Result<u16, FrontierError> {
        self.validate_unmet_horizon_question(question)?;
        let ply = self.path(question.anchor)?.anchor.ply();
        Ok(configured.min(question.target.0.saturating_sub(ply.0)))
    }

    /// Reuse the exact already charged endpoint. Starting a new continuation
    /// does not replay its history or charge it as a new Rules transition.
    pub(crate) fn observe_unmet_horizon_anchor(
        &mut self,
        question: &UnmetHorizonQuestion,
        stores: &PalsStores,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<CheckedFrontierLine, FrontierError> {
        self.check_control(clock, now)?;
        self.validate_unmet_horizon_question(question)?;
        self.checked_line(question.anchor, stores)
    }

    /// A real work=None GoalLocal native ticket evaluates this exact endpoint.
    /// The frame below is the original responsibility lineage, never a newly
    /// admitted manager frame, peer envelope or role-acceptance declaration.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accept_goal_horizon_fresh(
        &mut self,
        input: &GoalLocalInput,
        stores: &PalsStores,
        endpoint: FrontierPath,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierHorizonAcceptance, FrontierError> {
        let endpoint = VerifiedHorizonEndpoint::from_adaptive(input, stores, endpoint)?;
        self.accept_goal_horizon_fresh_inner(
            endpoint, stores, accepted, output, physical, clock, now,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accept_pc_goal_horizon_fresh(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        stores: &PalsStores,
        endpoint: FrontierPath,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierHorizonAcceptance, FrontierError> {
        let endpoint = VerifiedHorizonEndpoint::from_pc(input, stores, endpoint)?;
        self.accept_goal_horizon_fresh_inner(
            endpoint, stores, accepted, output, physical, clock, now,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn accept_goal_horizon_fresh_inner(
        &mut self,
        question: VerifiedHorizonEndpoint<'_>,
        stores: &PalsStores,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierHorizonAcceptance, FrontierError> {
        let started = Instant::now();
        let origin = question.source;
        let endpoint = question.scope.endpoint;
        let at = self.validate_horizon_origin(origin)?;
        self.check_goal_horizon_input(&question, stores)?;
        let fresh_boundary =
            self.checked_goal_horizon_fresh(&question, accepted, output, physical)?;
        let rules = Position::from_snapshot(self.path_snapshot(endpoint)?, Default::default())?;
        let terminal_boundary = HorizonBoundary::rules_terminal(&rules).ok();
        let rules_terminal = terminal_boundary.is_some();
        let boundary = terminal_boundary.unwrap_or(fresh_boundary);
        let completion = FrontierCompletionSeal::Fresh {
            scope: question.scope,
            ticket: *accepted.ticket(),
            output: accepted.output_digest(),
            physical: physical.into(),
        };
        if clock != self.registration.original_deadline.clock {
            return Err(FrontierError::ForeignRoot);
        }
        if let Some(disposition) = self.exact_horizon_retry(at, &completion)? {
            return Ok(disposition);
        }
        self.check_control(clock, now)?;
        self.validate_unmet_horizon_question(origin)?;
        self.check_horizon_progress_lineage(at, endpoint)?;
        self.check_control(
            clock,
            self.original_tick_after_elapsed(clock, now, started.elapsed())?,
        )?;
        if !rules_terminal && self.horizon_endpoint_is_shallow(at, endpoint)? {
            self.frames[at].completion = Some(completion);
            return Ok(FrontierHorizonAcceptance::ProgressAccepted);
        }
        self.finish_obligation(at, endpoint, boundary)?;
        self.frames[at].completion = Some(completion);
        Ok(FrontierHorizonAcceptance::Closed)
    }

    pub(crate) fn accept_pc_goal_horizon_rules(
        &mut self,
        input: &FrameInput<ReviewView, PcPortQuestion>,
        stores: &PalsStores,
        endpoint: FrontierPath,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierHorizonAcceptance, FrontierError> {
        let question = VerifiedHorizonEndpoint::from_pc(input, stores, endpoint)?;
        self.check_goal_horizon_input(&question, stores)?;
        self.close_goal_horizon_rules_origin(question.source, stores, endpoint, clock, now)
    }

    /// The private origin can close an actual Rules terminal even when no
    /// descendant NN question was needed. A paid path alone is not a Fresh
    /// evaluation and never closes a nonterminal shallow duty.
    pub(crate) fn close_goal_horizon_rules_origin(
        &mut self,
        origin: &UnmetHorizonQuestion,
        stores: &PalsStores,
        endpoint: FrontierPath,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<FrontierHorizonAcceptance, FrontierError> {
        self.check_store(stores)?;
        let at = self.validate_horizon_origin(origin)?;
        let position = Position::from_snapshot(self.path_snapshot(endpoint)?, Default::default())?;
        let boundary = HorizonBoundary::rules_terminal(&position)?;
        self.check_descendant_path(origin.anchor, endpoint)?;
        if clock != self.registration.original_deadline.clock {
            return Err(FrontierError::ForeignRoot);
        }
        if self.frames[at].finished {
            return if self.horizon.obligation_kind(origin.horizon).is_some()
                && self.frames[at]
                    .completion
                    .as_ref()
                    .is_some_and(|completion| {
                        let mut expected = clone_scope(&origin.progress.scope);
                        expected.endpoint = endpoint;
                        matches!(completion, FrontierCompletionSeal::Rules { scope }
                        if same_scope(scope, &expected))
                    })
            {
                Ok(FrontierHorizonAcceptance::AlreadyClosedExact)
            } else {
                Err(FrontierError::MissingAcceptance)
            };
        }
        self.check_control(clock, now)?;
        self.validate_unmet_horizon_question(origin)?;
        self.check_horizon_progress_lineage(at, endpoint)?;
        // Keep the actual originally accepted source allocation, while the
        // exact new terminal path is supplied by this controller's Rules.
        let mut scope = clone_scope(&origin.progress.scope);
        scope.endpoint = endpoint;
        self.finish_obligation(at, endpoint, boundary)?;
        self.frames[at].completion = Some(FrontierCompletionSeal::Rules { scope });
        Ok(FrontierHorizonAcceptance::Closed)
    }

    fn check_goal_horizon_input(
        &mut self,
        question: &VerifiedHorizonEndpoint<'_>,
        stores: &PalsStores,
    ) -> Result<(), FrontierError> {
        self.check_store(stores)?;
        let origin = question.source;
        let endpoint = question.scope.endpoint;
        self.validate_horizon_origin(origin)?;
        if question.scope.frame.authority != self.registration.authority
            || origin.root_budget != self.registration.root_budget
            || question.scope.original_deadline != self.registration.original_deadline
            || question.scope.view_seal.authority != self.registration.authority
            || question.scope.view_seal.root_state
                != self.registration.root_rules.snapshot().identity()
            || question.view.root_state() != self.registration.root_state
            || question.view.generation() != self.registration.authority.root.0
            || !question.view.validates(stores)
            || question.scope.public_context.0 != question.view.record_scope_digest()
            || self.path_prefix(endpoint)? != question.context.prefix
            || !self
                .path_snapshot(endpoint)?
                .same_state(&question.position.snapshot())
        {
            return Err(FrontierError::ForeignRoot);
        }
        self.check_descendant_path(origin.anchor, endpoint)
    }

    fn checked_goal_horizon_fresh(
        &self,
        question: &VerifiedHorizonEndpoint<'_>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
    ) -> Result<HorizonBoundary, FrontierError> {
        let ticket = accepted.ticket();
        if ticket.authority != self.registration.authority
            || ticket.question != question.scope.question_digest
            || ticket.view != question.scope.view_seal
            || ticket.original_deadline != self.registration.original_deadline
            || ticket.role != Role::Proposer
            || ticket.representation.record_revision != question.context.public_revision
            || !accepted.validates_output(output)
            || question.context.purpose != RoleQueryPurpose::ValueFresh
            || physical.physical_fresh != Some(true)
            || !physical.measurement_complete
            || physical.completed_nn_inputs.is_none_or(|count| count == 0)
            || physical
                .execution
                .is_none_or(|execution| execution.epoch != self.registration.authority.epoch)
        {
            return Err(FrontierError::MissingAcceptance);
        }
        self.registration
            .model_namespace
            .verify(
                ticket,
                output,
                Some(self.registration.model_namespace.value_identity()),
            )
            .map_err(|_| FrontierError::MissingAcceptance)?;
        let RolePayload::Proposal { wdl, .. } = output.payload else {
            return Err(FrontierError::MissingAcceptance);
        };
        let position = Position::from_snapshot(
            self.path_snapshot(question.scope.endpoint)?,
            Default::default(),
        )?;
        let value = ModelValueOutput {
            identity: self.registration.model_namespace.value_identity().clone(),
            input_sha256: output.key.input.0,
            state: position.position_identity(),
            perspective: position.side_to_move(),
            wdl: wdl.probabilities(),
        };
        Ok(HorizonBoundary::fresh_value(
            &position,
            &value,
            self.registration.model_namespace.value_identity(),
        )?)
    }

    fn exact_horizon_retry(
        &self,
        at: usize,
        incoming: &FrontierCompletionSeal,
    ) -> Result<Option<FrontierHorizonAcceptance>, FrontierError> {
        let Some(previous) = self.frames[at].completion.as_ref() else {
            return Ok(None);
        };
        let Some(progress) = HorizonProgress::from_completion(incoming) else {
            return Err(FrontierError::MissingAcceptance);
        };
        if progress.matches_completion(previous) {
            return Ok(Some(if self.frames[at].finished {
                FrontierHorizonAcceptance::AlreadyClosedExact
            } else {
                FrontierHorizonAcceptance::AlreadyAcceptedProgressExact
            }));
        }
        if let FrontierCompletionSeal::Fresh { scope, .. } = previous {
            if scope.endpoint == progress.scope.endpoint {
                // Another ticket/view is actual additional work, not an exact
                // retry or deeper progress that can replace this source.
                return Err(FrontierError::MissingAcceptance);
            }
        }
        Ok(None)
    }

    fn validate_horizon_origin(
        &self,
        question: &UnmetHorizonQuestion,
    ) -> Result<usize, FrontierError> {
        if question.owner != self.owner
            || question.original_deadline != self.registration.original_deadline
            || question.root_budget != self.registration.root_budget
            || self
                .registration
                .goal
                .depth_plies
                .map(|depth| ChessPly(depth.get()))
                != Some(question.target)
        {
            return Err(FrontierError::ForeignHandle);
        }
        let at = self.obligation_slot(question.obligation, question.frame)?;
        let held = &self.frames[at];
        if held.kind != question.kind
            || held.original_first != question.first
            || held.question != question.source_question
            || held.source_path != question.source_path
            || held.pc_source != question.pc_source
            || held.horizon != Some(question.horizon)
            || question.anchor != question.progress.scope.endpoint
            || question.progress.ticket.authority != self.registration.authority
            || !self
                .registration
                .root_moves
                .allowed()
                .contains(&question.first)
            || self.path(question.anchor)?.anchor.ply() >= question.target
        {
            return Err(FrontierError::ForeignHandle);
        }
        self.check_descendant_path(question.source_path, question.anchor)?;
        Ok(at)
    }

    fn check_descendant_path(
        &self,
        ancestor: FrontierPath,
        endpoint: FrontierPath,
    ) -> Result<(), FrontierError> {
        self.path(ancestor)?;
        self.path(endpoint)?;
        let prefix = self.path_prefix(ancestor)?;
        if !self.path_prefix(endpoint)?.starts_with(&prefix) {
            return Err(FrontierError::Invalid(
                "horizon endpoint escaped original path",
            ));
        }
        Ok(())
    }

    fn check_horizon_progress_lineage(
        &self,
        at: usize,
        endpoint: FrontierPath,
    ) -> Result<(), FrontierError> {
        self.check_descendant_path(self.frames[at].source_path, endpoint)?;
        if self.path_prefix(endpoint)?.first().copied() != Some(self.frames[at].original_first) {
            return Err(FrontierError::Invalid(
                "horizon endpoint escaped original first",
            ));
        }
        if let Some(FrontierCompletionSeal::Fresh { scope, .. }) = &self.frames[at].completion {
            self.check_descendant_path(scope.endpoint, endpoint)?;
        }
        Ok(())
    }

    fn horizon_endpoint_is_shallow(
        &self,
        at: usize,
        endpoint: FrontierPath,
    ) -> Result<bool, FrontierError> {
        Ok(self.frames[at].horizon.is_some()
            && self
                .remaining_depth(endpoint)?
                .is_some_and(|remaining| remaining.0 != 0))
    }
}

fn goal_scope(
    origin: &UnmetHorizonQuestion,
    input: &FrameInput<ReviewView, AdaptivePortQuestion>,
    endpoint: FrontierPath,
) -> Result<CompletionScope, FrontierError> {
    let first = rz_contracts::pals::Move16::encode(rz_contracts::Move::try_from(origin.first)?);
    Ok(CompletionScope {
        question: WeakQuestionOwner::Adaptive(Arc::downgrade(&input.question)),
        view: Arc::downgrade(&input.view),
        frame: origin.frame,
        stage: input.stage,
        first,
        original_deadline: origin.original_deadline,
        question_digest: input.question_digest,
        view_seal: input.view_seal,
        public_context: input.public_context,
        endpoint,
    })
}
