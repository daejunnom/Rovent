//! Owner-private continuation plans. Semantic commits construct bounded owned
//! recipes, not selected ReviewViews. Materialization follows actual submitted
//! view consumption and preserves the original root/deadline and frozen pair.

use super::*;
use crate::pals::store::StoreReadOwner;

// Keep this bounded, owner-private blueprint inline. Boxing it would add an
// allocation per continuation around the immutable exact authority and Arc
// custodians retained until materialization; it would not narrow that lifetime.
#[allow(clippy::large_enum_variant)]
pub enum AdaptivePortTransition {
    Continue(AdaptiveContinuationBlueprint),
    Complete { scope: FrameCompletionScope },
    Unknown { reason: &'static str },
}

pub struct AdaptivePortMaterializedStep {
    pub evidence: ContractDigest,
    pub transition: FrameTransition<ReviewView, AdaptivePortQuestion>,
    pub fresh_pair: Option<FreshPairSeal>,
    pub rules_pair: Option<AdaptiveRulesPairProof>,
    pub queued_recheck: Option<super::branch_queue::AdaptiveQueuedRecheck>,
}

struct BlueprintSeal {
    authority: SearchAuthority,
    deadline: Deadline,
    hard_deadline: Instant,
    owner: StoreReadOwner,
    generation: u64,
    root: StateId,
    revision: u64,
    endpoint: (usize, StateId, SituationId, PositionSnapshot),
}
struct RoleCause {
    request: RequestId,
    question: ContractDigest,
    view: ContractDigest,
    frame: crate::pals::coordinator::FrameId,
}
struct InputRecipe {
    node: usize,
    stage: RefinementStage,
    first: Option<BoardMove>,
    prefix: Vec<BoardMove>,
    proposal: Vec<BoardMove>,
    counter: Option<Vec<BoardMove>>,
    repaired: Option<Vec<BoardMove>>,
    attack_ply: Option<usize>,
    recheck: bool,
    frozen: Option<Arc<ReviewView>>,
    paired_repaired: Option<ObservationId>,
    discovery_restricted: bool,
    counter_source: Option<ObservationId>,
    lane_selection: Option<Box<super::goal_hooks::AdaptiveLaneSelection>>,
}
enum BlueprintContent {
    Input(InputRecipe),
    /// This is an already selected, still pinned SECOND frozen endpoint. No
    /// replacement/latest view is selected after the first value is accepted.
    FrozenSecond(FrameInput<ReviewView, AdaptivePortQuestion>),
    FreshPair {
        first: Option<BoardMove>,
        attack_ply: Option<usize>,
        repaired_node: usize,
        repaired: Vec<BoardMove>,
        counter_node: usize,
        counter: Vec<BoardMove>,
        counter_snapshot: PositionSnapshot,
    },
}
/// Fields and constructors remain private; consumers cannot supply an owner,
/// fake frame, placeholder view or a boolean permission to bypass consumption.
pub struct AdaptiveContinuationBlueprint {
    seal: BlueprintSeal,
    cause: Option<RoleCause>,
    content: BlueprintContent,
    // This branch's original scalar provenance, never a global/latest record.
    diagnostic_repair_line: Option<AdaptiveRepairLineProvenance>,
}
impl AdaptiveContinuationBlueprint {
    pub(super) fn with_goal_lane_selection(
        mut self,
        source: Option<Box<super::goal_hooks::AdaptiveLaneSelection>>,
    ) -> Result<Self, PalsError> {
        let BlueprintContent::Input(recipe) = &mut self.content else {
            return Err(RoleError::InvalidOutput.into());
        };
        if recipe.lane_selection.is_some() {
            return Err(RoleError::InvalidOutput.into());
        }
        recipe.lane_selection = source;
        Ok(self)
    }
    pub(super) fn carry_diagnostic_repair_line(
        &mut self,
        source: Option<AdaptiveRepairLineProvenance>,
    ) {
        if source.is_none() {
            return;
        }
        // Input recipes with a new/unrepaired branch deliberately keep None.
        // FrozenSecond already contains the exact metadata captured on both
        // endpoints at their original shared-view materialization.
        match &self.content {
            BlueprintContent::Input(recipe) if recipe.repaired.is_some() => {
                self.diagnostic_repair_line = source;
            }
            BlueprintContent::FreshPair { .. } => self.diagnostic_repair_line = source,
            _ => {}
        }
    }
    pub(super) fn queued_recheck_identity(&self) -> Result<(BoardMove, Vec<BoardMove>), PalsError> {
        match &self.content {
            BlueprintContent::Input(recipe) => Ok((
                recipe.first.ok_or(RoleError::InvalidOutput)?,
                recipe
                    .repaired
                    .as_ref()
                    .ok_or(RoleError::InvalidOutput)?
                    .clone(),
            )),
            _ => Err(RoleError::InvalidOutput.into()),
        }
    }
    pub(super) fn bind_role_cause(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), PalsError> {
        if self.cause.is_some()
            || self.seal.authority != ticket.authority
            || work.frame.authority != ticket.authority
            || self.seal.deadline != work.original_deadline
            || ticket.question != work.input.question_digest
            || ticket.view != work.input.view_seal
        {
            return Err(RoleError::InvalidOutput.into());
        }
        self.cause = Some(RoleCause {
            request: ticket.id,
            question: ticket.question,
            view: ticket.view.view_digest,
            frame: work.frame,
        });
        Ok(())
    }
    pub(super) fn with_counter_source(mut self, source: Option<ObservationId>) -> Self {
        if let BlueprintContent::Input(recipe) = &mut self.content {
            recipe.counter_source = source;
        }
        self
    }
    pub(super) fn with_restricted_discovery(mut self) -> Self {
        if let BlueprintContent::Input(recipe) = &mut self.content {
            recipe.discovery_restricted = true;
        }
        self
    }
}
impl AdaptivePortTransition {
    pub(super) fn carry_diagnostic_repair_line(
        &mut self,
        source: Option<AdaptiveRepairLineProvenance>,
    ) {
        if let Self::Continue(blueprint) = self {
            blueprint.carry_diagnostic_repair_line(source);
        }
    }
    pub(super) fn bind_role_cause(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
    ) -> Result<(), PalsError> {
        if let Self::Continue(blueprint) = self {
            blueprint.bind_role_cause(work, ticket)?;
        }
        Ok(())
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// An unchanged full Repair prefix publishes an observation, never an
    /// accepted P continuation. Attach its actual record only after the normal
    /// producer returned that zero-step validation/value recipe.
    pub(super) fn bind_zero_step_repair_goal_transition(
        &self,
        original: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        transition: &mut AdaptivePortTransition,
    ) -> Result<(), PalsError> {
        let AdaptivePortTransition::Continue(blueprint) = transition else {
            return Ok(());
        };
        let BlueprintContent::Input(recipe) = &mut blueprint.content else {
            return Ok(());
        };
        if !matches!(
            recipe.stage,
            RefinementStage::RepairValidation | RefinementStage::RepairValueFresh
        ) || recipe.prefix.len() != self.continuation_limit()
            || recipe.repaired.as_ref() != Some(&recipe.prefix)
        {
            return Ok(());
        }
        if recipe.lane_selection.is_some() {
            return Err(RoleError::InvalidOutput.into());
        }
        recipe.lane_selection = self.prepare_zero_step_repair_goal_observation(
            original,
            RoleQuestion {
                purpose: RoleQueryPurpose::RepairPolicy,
                prefix: &recipe.prefix,
                proposal: &recipe.proposal,
                refutation: recipe.counter.as_deref(),
                divergences: &[],
            },
            recipe.attack_ply.ok_or(RoleError::InvalidOutput)?,
            recipe.counter_source,
        )?;
        Ok(())
    }
    fn blueprint_seal(&self, node: usize) -> Result<BlueprintSeal, PalsError> {
        self.check_live()?;
        let endpoint = self
            .engine
            .semantic
            .nodes
            .get(node)
            .ok_or(RoleError::InvalidOutput)?;
        if self
            .engine
            .semantic
            .stores
            .situations
            .get(endpoint.situation)?
            .state
            != endpoint.state
            || !self
                .engine
                .semantic
                .stores
                .states
                .get(endpoint.state)?
                .same_state(&endpoint.position.snapshot())
        {
            return Err(StoreError::StaleConsumer.into());
        }
        Ok(BlueprintSeal {
            authority: self.authority,
            deadline: self.deadline,
            hard_deadline: self.limits.deadline,
            owner: self.engine.semantic.stores.read_owner(),
            generation: self.engine.semantic.stores.generation(),
            root: self.engine.semantic.nodes[self.root].state,
            revision: self.engine.semantic.revision,
            endpoint: (
                node,
                endpoint.state,
                endpoint.situation,
                endpoint.position.snapshot(),
            ),
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn blueprint(
        &self,
        node: usize,
        stage: RefinementStage,
        first: Option<BoardMove>,
        prefix: Vec<BoardMove>,
        proposal: Vec<BoardMove>,
        counter: Option<Vec<BoardMove>>,
        repaired: Option<Vec<BoardMove>>,
        attack_ply: Option<usize>,
        recheck: bool,
        frozen: Option<Arc<ReviewView>>,
        paired_repaired: Option<ObservationId>,
    ) -> Result<AdaptiveContinuationBlueprint, PalsError> {
        let cap = self.engine.config.line_plies;
        if prefix.len() > cap
            || proposal.len() > cap
            || counter.as_ref().is_some_and(|line| line.len() > cap)
            || repaired.as_ref().is_some_and(|line| line.len() > cap)
        {
            return Err(ReviewError::ViewCapacity.into());
        }
        let seal = self.blueprint_seal(node)?;
        let mut replay = self.engine.semantic.nodes[self.root].position.clone();
        for movement in &prefix {
            replay.make_move(*movement)?;
        }
        if !replay.snapshot().same_state(&seal.endpoint.3)
            || first.is_some_and(|first| !prefix.is_empty() && prefix.first() != Some(&first))
        {
            return Err(ReviewError::AnchorMismatch.into());
        }
        Ok(AdaptiveContinuationBlueprint {
            seal,
            cause: None,
            diagnostic_repair_line: None,
            content: BlueprintContent::Input(InputRecipe {
                node,
                stage,
                first,
                prefix,
                proposal,
                counter,
                repaired,
                attack_ply,
                recheck,
                frozen,
                paired_repaired,
                discovery_restricted: false,
                counter_source: None,
                lane_selection: None,
            }),
        })
    }
    pub(super) fn blueprint_ready(
        &self,
        input: FrameInput<ReviewView, AdaptivePortQuestion>,
    ) -> Result<AdaptiveContinuationBlueprint, PalsError> {
        let seal = self.blueprint_seal(input.question.node)?;
        if input.stage != RefinementStage::FreshCounterEndpoint
            || input.view_seal != self.seal(&input.view)
            || !input.view.validates(&self.engine.semantic.stores)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(AdaptiveContinuationBlueprint {
            seal,
            cause: None,
            diagnostic_repair_line: input.question.diagnostic_repair_line,
            content: BlueprintContent::FrozenSecond(input),
        })
    }
    pub(super) fn blueprint_fresh_pair(
        &self,
        query: &AdaptivePortQuestion,
        repaired_node: usize,
        repaired: Vec<BoardMove>,
        counter_node: usize,
        counter: Vec<BoardMove>,
    ) -> Result<AdaptiveContinuationBlueprint, PalsError> {
        let seal = self.blueprint_seal(repaired_node)?;
        let counter_snapshot = self
            .engine
            .semantic
            .nodes
            .get(counter_node)
            .ok_or(RoleError::InvalidOutput)?
            .position
            .snapshot();
        if repaired.len() > self.engine.config.line_plies
            || counter.len() > self.engine.config.line_plies
        {
            return Err(ReviewError::ViewCapacity.into());
        }
        Ok(AdaptiveContinuationBlueprint {
            seal,
            cause: None,
            diagnostic_repair_line: query.diagnostic_repair_line,
            content: BlueprintContent::FreshPair {
                first: query.first,
                attack_ply: query.attack_ply,
                repaired_node,
                repaired,
                counter_node,
                counter,
                counter_snapshot,
            },
        })
    }
    pub(super) fn validate_blueprint(
        &self,
        blueprint: &AdaptiveContinuationBlueprint,
    ) -> Result<(), PalsError> {
        self.check_live()?;
        let seal = &blueprint.seal;
        let node = self
            .engine
            .semantic
            .nodes
            .get(seal.endpoint.0)
            .ok_or(RoleError::InvalidOutput)?;
        if seal.authority != self.authority
            || seal.deadline != self.deadline
            || seal.hard_deadline != self.limits.deadline
            || !seal.owner.matches(&self.engine.semantic.stores)
            || seal.generation != self.engine.semantic.stores.generation()
            || seal.root != self.engine.semantic.nodes[self.root].state
            || seal.revision > self.engine.semantic.revision
            || node.state != seal.endpoint.1
            || node.situation != seal.endpoint.2
            || !node.position.snapshot().same_state(&seal.endpoint.3)
        {
            return Err(StoreError::StaleConsumer.into());
        }
        if let Some(cause) = &blueprint.cause {
            if cause.frame.authority != self.authority
                || !self.accepted.contains(&cause.request)
                || self.consumed_role_views.get(&cause.request) != Some(&cause.view)
                || self.committed_role_questions.get(&cause.request) != Some(&cause.question)
            {
                return Err(StoreError::InvalidConditions(
                    "continuation precedes submitted view consumption",
                )
                .into());
            }
        }
        Ok(())
    }
    /// Takes each opaque recipe once. Selection begins only after all checked
    /// semantic commits and, for role results, the exact original view consume.
    pub fn materialize_step(
        &mut self,
        step: AdaptivePortStep,
    ) -> Result<AdaptivePortMaterializedStep, PalsError> {
        if step.consumption_gate.is_some() || step.feedback_gate.is_some() {
            // Historical consumption survives expiry for final receipts, but
            // it cannot authorize a new successful semantic transition.
            self.check_live()?;
        }
        if let Some((request, question, view)) = step.consumption_gate {
            if !self.accepted.contains(&request)
                || self.committed_role_questions.get(&request) != Some(&question)
                || self.consumed_role_views.get(&request) != Some(&view)
            {
                return Err(StoreError::InvalidConditions(
                    "step precedes submitted view consumption",
                )
                .into());
            }
        }
        if let Some((question, observation)) = step.feedback_gate {
            if !self.cpu_feedback_still_authorized(question, observation)
                && !self.external_feedback_still_authorized(question, observation)
                && !self.profiled_external_feedback_still_authorized(question, observation)
            {
                return Err(StoreError::StaleConsumer.into());
            }
        }
        let utility_cause = if self.utility_continuations.enabled() {
            match &step.transition {
                AdaptivePortTransition::Continue(blueprint) => {
                    self.validate_blueprint(blueprint)?;
                    if let Some(cause) = &blueprint.cause {
                        if step.consumption_gate
                            != Some((cause.request, cause.question, cause.view))
                        {
                            return Err(RoleError::InvalidOutput.into());
                        }
                        Some(utility_dependencies::UtilityContinuationCause::Role {
                            request: cause.request,
                            frame: cause.frame,
                            question: cause.question,
                            view: cause.view,
                        })
                    } else if let Some((question, observation)) = step.feedback_gate {
                        self.utility_cpu_continuation_cause(question, observation)?
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            None
        };
        if utility_cause.is_some() {
            self.utility_continuations.reserve()?;
        }
        let transition = match step.transition {
            AdaptivePortTransition::Complete { scope } => FrameTransition::Complete { scope },
            AdaptivePortTransition::Unknown { reason } => FrameTransition::Unknown { reason },
            AdaptivePortTransition::Continue(blueprint) => {
                self.validate_blueprint(&blueprint)?;
                let diagnostic_repair_line = blueprint.diagnostic_repair_line;
                let input = match blueprint.content {
                    BlueprintContent::Input(recipe) => {
                        let lane_selection = recipe.lane_selection;
                        let mut input = self.input(
                            recipe.node,
                            recipe.stage,
                            recipe.first,
                            recipe.prefix,
                            recipe.proposal,
                            recipe.counter,
                            recipe.repaired,
                            recipe.attack_ply,
                            recipe.recheck,
                            recipe.frozen,
                            recipe.paired_repaired,
                        )?;
                        let query = Arc::make_mut(&mut input.question);
                        query.counter_source = recipe.counter_source;
                        query.discovery_restricted = recipe.discovery_restricted;
                        query.diagnostic_repair_line = diagnostic_repair_line;
                        if let Some(selection) = lane_selection {
                            self.bind_goal_lane_selection(selection, &input)?;
                        }
                        input
                    }
                    BlueprintContent::FrozenSecond(input) => {
                        if input.view_seal != self.seal(&input.view)
                            || !input.view.validates(&self.engine.semantic.stores)
                        {
                            return Err(RoleError::InvalidOutput.into());
                        }
                        input
                    }
                    BlueprintContent::FreshPair {
                        first,
                        attack_ply,
                        repaired_node,
                        repaired,
                        counter_node,
                        counter,
                        counter_snapshot,
                    } => {
                        if !self
                            .engine
                            .semantic
                            .nodes
                            .get(counter_node)
                            .ok_or(RoleError::InvalidOutput)?
                            .position
                            .snapshot()
                            .same_state(&counter_snapshot)
                        {
                            return Err(ReviewError::AnchorMismatch.into());
                        }
                        let pair = self
                            .engine
                            .prepare_frozen_review_views(
                                repaired_node,
                                RoleQuestion {
                                    purpose: RoleQueryPurpose::ValueFresh,
                                    prefix: &repaired,
                                    proposal: &repaired,
                                    refutation: None,
                                    divergences: &[],
                                },
                                &self.engine.semantic.nodes[repaired_node]
                                    .position
                                    .legal_moves(),
                                counter_node,
                                RoleQuestion {
                                    purpose: RoleQueryPurpose::ValueFresh,
                                    prefix: &counter,
                                    proposal: &counter,
                                    refutation: None,
                                    divergences: &[],
                                },
                                &self.engine.semantic.nodes[counter_node]
                                    .position
                                    .legal_moves(),
                            )?
                            .ok_or(PalsError::InvalidConfig)?;
                        let mut first_input = self.input(
                            repaired_node,
                            RefinementStage::FreshRepairedEndpoint,
                            first,
                            repaired.clone(),
                            repaired.clone(),
                            None,
                            Some(repaired.clone()),
                            attack_ply,
                            false,
                            Some(pair.0),
                            None,
                        )?;
                        let mut second_input = self.input(
                            counter_node,
                            RefinementStage::FreshCounterEndpoint,
                            first,
                            counter.clone(),
                            counter,
                            None,
                            Some(repaired),
                            attack_ply,
                            false,
                            Some(pair.1),
                            None,
                        )?;
                        // Both newly allocated opaque Questions still have a
                        // unique Arc here. Default-off adds no mutation/clone.
                        if diagnostic_repair_line.is_some() {
                            Arc::make_mut(&mut first_input.question).diagnostic_repair_line =
                                diagnostic_repair_line;
                            Arc::make_mut(&mut second_input.question).diagnostic_repair_line =
                                diagnostic_repair_line;
                        }
                        // Full-baseline endpoints must retain the very same
                        // submission-time raw scope, not merely two equal
                        // selected views rebuilt around newer public rows.
                        match (
                            &first_input.question.legacy_full_records,
                            &second_input.question.legacy_full_records,
                        ) {
                            (Some(first), Some(second)) if Arc::ptr_eq(first, second) => {}
                            (None, None) => {}
                            _ => return Err(RoleError::InvalidOutput.into()),
                        }
                        if first_input.public_context != second_input.public_context
                            || self
                                .fresh_pending
                                .contains_key(&first_input.question_digest)
                        {
                            return Err(RoleError::InvalidOutput.into());
                        }
                        self.fresh_pending
                            .insert(first_input.question_digest, second_input);
                        first_input
                    }
                };
                FrameTransition::Continue(input)
            }
        };
        if let (Some(cause), FrameTransition::Continue(input)) = (utility_cause, &transition) {
            self.record_utility_continuation(cause, input)?;
        }
        let queued_recheck = self.seal_queued_recheck(step.queue_question, step.evidence)?;
        Ok(AdaptivePortMaterializedStep {
            evidence: step.evidence,
            transition,
            fresh_pair: step.fresh_pair,
            rules_pair: step.rules_pair,
            queued_recheck,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::port_tests::{output, setup, setup_with_resolver, work};
    use super::*;

    #[test]
    fn role_commit_builds_no_next_view_and_materialization_requires_original_consume() {
        let mut port = setup(Position::startpos(), 3);
        let submitted = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
        let (ticket, response) = output(&submitted, 1);
        let questions = port.next_question;
        let leases = port.engine.semantic.stores.review_pin_count().unwrap();
        let step = port
            .commit_role(&submitted, &ticket, &response, None)
            .unwrap();
        assert_eq!(port.next_question, questions);
        assert_eq!(
            port.engine.semantic.stores.review_pin_count().unwrap(),
            leases
        );
        assert!(matches!(
            step.transition,
            AdaptivePortTransition::Continue(_)
        ));
        assert!(matches!(
            port.materialize_step(step),
            Err(PalsError::Store(StoreError::InvalidConditions(
                "step precedes submitted view consumption"
            )))
        ));
        assert_eq!(port.next_question, questions);
    }

    #[test]
    fn actual_original_consumption_is_idempotent_then_selects_next_view_once() {
        let mut port = setup(Position::startpos(), 3);
        let submitted = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
        let (ticket, response) = output(&submitted, 1);
        let questions = port.next_question;
        let step = port
            .commit_role(&submitted, &ticket, &response, None)
            .unwrap();
        port.role_consumed(&ticket, &submitted.input.view).unwrap();
        port.role_consumed(&ticket, &submitted.input.view).unwrap();
        let materialized = port.materialize_step(step).unwrap();
        let FrameTransition::Continue(next) = materialized.transition else {
            panic!("continuation")
        };
        assert_eq!(port.next_question, questions + 1);
        assert_eq!(next.stage, RefinementStage::ProposalContinuation);
        assert_eq!(next.question.context.prefix.len(), 2);
        assert_ne!(next.view.seal_digest(), submitted.input.view.seal_digest());
    }

    #[test]
    fn terminal_step_shapes_cannot_bypass_the_original_consumption_gate() {
        for unknown in [false, true] {
            let mut port = setup(Position::startpos(), 3);
            let submitted = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
            let (ticket, response) = output(&submitted, 1);
            let mut step = port
                .commit_role(&submitted, &ticket, &response, None)
                .unwrap();
            // Authored shape variation isolates the all-transition gate. The
            // real role commit and its original seal remain unchanged.
            step.transition = if unknown {
                AdaptivePortTransition::Unknown {
                    reason: "authored terminal shape",
                }
            } else {
                AdaptivePortTransition::Complete {
                    scope: FrameCompletionScope::NoAlternative,
                }
            };
            assert!(matches!(
                port.materialize_step(step),
                Err(PalsError::Store(StoreError::InvalidConditions(
                    "step precedes submitted view consumption"
                )))
            ));
            assert!(!port.consumed_role_views.contains_key(&ticket.id));
        }
    }

    #[test]
    fn consumed_semantic_terminal_step_expires_without_erasing_historical_consumption() {
        let mut port = setup(Position::startpos(), 3);
        let submitted = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
        let (ticket, response) = output(&submitted, 1);
        let mut step = port
            .commit_role(&submitted, &ticket, &response, None)
            .unwrap();
        port.role_consumed(&ticket, &submitted.input.view).unwrap();
        step.transition = AdaptivePortTransition::Complete {
            scope: FrameCompletionScope::NoAlternative,
        };
        port.limits.deadline = Instant::now();
        assert!(
            port.role_consumption_recorded(&ticket, &submitted.input.view)
                .unwrap()
        );
        assert!(matches!(
            port.materialize_step(step),
            Err(PalsError::Role(RoleError::Deadline))
        ));
        assert!(
            port.role_consumption_recorded(&ticket, &submitted.input.view)
                .unwrap()
        );
        // A raw-only closing result remains reportable after expiry without
        // asserting a successful semantic completion or selecting a new view.
        let raw = AdaptivePortStep {
            evidence: ContractDigest([9; 32]),
            transition: AdaptivePortTransition::Unknown {
                reason: "raw-only expired feedback",
            },
            fresh_pair: None,
            rules_pair: None,
            consumption_gate: None,
            feedback_gate: None,
            queue_question: None,
        };
        assert!(matches!(
            port.materialize_step(raw).unwrap().transition,
            FrameTransition::Unknown { .. }
        ));
    }

    #[test]
    fn fresh_physical_admission_failure_does_not_consume_a_semantic_attempt() {
        let mut port =
            setup_with_resolver(Position::startpos(), 3, ResolverPolicy::ModelWdlRestricted);
        let submitted = work(&mut port, "e2e4", RefinementStage::ProposalValueFresh, 1);
        let (ticket, response) = output(&submitted, 1);
        let observations = port
            .engine
            .semantic
            .stores
            .observations
            .next_observation_sequence();
        let completed = AdaptivePortRoleEvidence {
            execution: Some(rz_contracts::ExecutionId::new(ticket.id.epoch, 1)),
            completed_nn_inputs: Some(2),
            measurement_complete: true,
            physical_fresh: Some(true),
        };
        for physical in [
            None,
            Some(AdaptivePortRoleEvidence {
                completed_nn_inputs: None,
                ..completed
            }),
            Some(AdaptivePortRoleEvidence {
                completed_nn_inputs: Some(0),
                ..completed
            }),
            Some(AdaptivePortRoleEvidence {
                measurement_complete: false,
                ..completed
            }),
            Some(AdaptivePortRoleEvidence {
                physical_fresh: None,
                ..completed
            }),
            Some(AdaptivePortRoleEvidence {
                physical_fresh: Some(false),
                ..completed
            }),
            Some(AdaptivePortRoleEvidence {
                execution: None,
                ..completed
            }),
        ] {
            assert!(matches!(
                port.commit_role(&submitted, &ticket, &response, physical),
                Err(PalsError::Role(RoleError::InvalidOutput))
            ));
            assert!(!port.attempted_role_commits.contains(&ticket.id));
            assert!(!port.accepted.contains(&ticket.id));
            assert_eq!(
                port.engine
                    .semantic
                    .stores
                    .observations
                    .next_observation_sequence(),
                observations
            );
        }
    }

    #[test]
    fn fresh_value_consumes_one_output_with_one_or_two_completed_graph_inputs() {
        // Authored completion metadata isolates the semantic admission boundary;
        // this is not a native NN, provider, or physical-completion attestation.
        for completed_nn_inputs in [1, 2] {
            let mut port =
                setup_with_resolver(Position::startpos(), 3, ResolverPolicy::ModelWdlRestricted);
            let submitted = work(&mut port, "e2e4", RefinementStage::ProposalValueFresh, 1);
            let (ticket, response) = output(&submitted, 1);
            let observations = port
                .engine
                .semantic
                .stores
                .observations
                .next_observation_sequence();
            let physical = Some(AdaptivePortRoleEvidence {
                execution: Some(rz_contracts::ExecutionId::new(ticket.id.epoch, 1)),
                completed_nn_inputs: Some(completed_nn_inputs),
                measurement_complete: true,
                physical_fresh: Some(true),
            });
            let step = port
                .commit_role(&submitted, &ticket, &response, physical)
                .unwrap_or_else(|error| {
                    panic!("{completed_nn_inputs} completed inputs: {error:?}")
                });
            let observation = port.engine.semantic.nodes[submitted.input.question.node]
                .model_observation
                .expect("one committed model value");
            let stored = port
                .engine
                .semantic
                .stores
                .observations
                .get(observation)
                .unwrap();
            assert!(matches!(stored.score, RawScore::ContextWdl { .. }));
            assert_eq!(stored.model_value_input, Some(response.key.input.0));
            assert!(port.attempted_role_commits.contains(&ticket.id));
            assert!(port.accepted.contains(&ticket.id));
            let committed = port
                .engine
                .semantic
                .stores
                .observations
                .next_observation_sequence();
            assert_eq!(committed, observations + 1);
            assert!(
                port.commit_role(&submitted, &ticket, &response, physical)
                    .is_err()
            );
            assert_eq!(
                port.engine
                    .semantic
                    .stores
                    .observations
                    .next_observation_sequence(),
                committed
            );
            port.role_consumed(&ticket, &submitted.input.view).unwrap();
            let consumed = port.counters.consumed_role_outputs;
            port.role_consumed(&ticket, &submitted.input.view).unwrap();
            assert_eq!(port.counters.consumed_role_outputs, consumed);
            port.materialize_step(step).unwrap();
        }
    }

    #[test]
    fn partial_semantic_failure_cannot_replay_the_same_response() {
        let mut port = setup(Position::startpos(), 2);
        let submitted = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
        let (ticket, response) = output(&submitted, 1);
        port.engine.config.max_records = 0;
        let nodes = port.engine.semantic.nodes.len();
        assert!(matches!(
            port.commit_role(&submitted, &ticket, &response, None),
            Err(PalsError::Store(StoreError::PinSaturated(_)))
        ));
        assert!(port.engine.semantic.nodes.len() > nodes);
        assert!(port.attempted_role_commits.contains(&ticket.id));
        assert!(!port.accepted.contains(&ticket.id));
        let nodes_after_failure = port.engine.semantic.nodes.len();
        assert!(matches!(
            port.commit_role(&submitted, &ticket, &response, None),
            Err(PalsError::Store(StoreError::InvalidEvidence(
                "role semantic commit was already attempted"
            )))
        ));
        assert_eq!(port.engine.semantic.nodes.len(), nodes_after_failure);
        assert_eq!(port.counters.consumed_role_outputs, 0);
    }

    #[test]
    fn later_pending_commit_does_not_stale_an_independently_consumed_blueprint() {
        let mut port = setup(Position::startpos(), 3);
        let first = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
        let second = work(&mut port, "d2d4", RefinementStage::ProposalContinuation, 2);
        let (a, output_a) = output(&first, 1);
        let (b, output_b) = output(&second, 2);
        let step_a = port.commit_role(&first, &a, &output_a, None).unwrap();
        port.role_consumed(&a, &first.input.view).unwrap();
        let step_b = port.commit_role(&second, &b, &output_b, None).unwrap();
        port.role_consumed(&b, &second.input.view).unwrap();
        assert!(matches!(
            port.materialize_step(step_a).unwrap().transition,
            FrameTransition::Continue(_)
        ));
        assert!(matches!(
            port.materialize_step(step_b).unwrap().transition,
            FrameTransition::Continue(_)
        ));
    }

    #[test]
    fn frozen_pair_views_are_created_only_at_materialization_and_share_public_scope() {
        let mut port = setup(Position::startpos(), 3);
        let submitted = work(&mut port, "e2e4", RefinementStage::RepairValueFresh, 1);
        let repaired: Vec<BoardMove> = ["e2e4", "e7e5"]
            .into_iter()
            .map(|m| m.parse().unwrap())
            .collect();
        let counter: Vec<BoardMove> = ["e2e4", "c7c5"]
            .into_iter()
            .map(|m| m.parse().unwrap())
            .collect();
        let repaired_node = port.replay_node(&repaired).unwrap();
        let counter_node = port.replay_node(&counter).unwrap();
        let questions = port.next_question;
        let leases = port.engine.semantic.stores.review_pin_count().unwrap();
        let blueprint = port
            .blueprint_fresh_pair(
                &submitted.input.question,
                repaired_node,
                repaired,
                counter_node,
                counter,
            )
            .unwrap();
        assert!(port.fresh_pending.is_empty());
        assert_eq!(port.next_question, questions);
        assert_eq!(
            port.engine.semantic.stores.review_pin_count().unwrap(),
            leases
        );
        let step = AdaptivePortStep {
            evidence: ContractDigest([1; 32]),
            transition: AdaptivePortTransition::Continue(blueprint),
            fresh_pair: None,
            rules_pair: None,
            consumption_gate: None,
            feedback_gate: None,
            queue_question: None,
        };
        let FrameTransition::Continue(first) = port.materialize_step(step).unwrap().transition
        else {
            panic!("Fresh pair first")
        };
        let second = port.fresh_pending.get(&first.question_digest).unwrap();
        assert_eq!(first.public_context, second.public_context);
        assert_ne!(first.view.digest(), second.view.digest());
        assert_eq!(port.next_question, questions + 2);
    }

    #[test]
    fn original_view_release_and_real_archive_reclaim_preserve_blueprint_endpoint() {
        use super::super::archive_worker::{
            ArchiveOperation, ArchiveOwner, ArchiveWork, ArchiveWorker,
        };
        use crate::pals::store::ArchiveConfig;
        use std::path::PathBuf;
        let mut port = setup(Position::startpos(), 3);
        let output_root = if cfg!(windows) {
            std::env::var_os("APPDATA").map(PathBuf::from)
        } else {
            std::env::var_os("RUNNER_TEMP").map(PathBuf::from)
        }
        .unwrap_or_else(std::env::temp_dir);
        let serial = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = output_root.join("RoveZero").join("tmp").join(format!(
            "blueprint-archive-fixture-{}-{serial}",
            std::process::id()
        ));
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_owned();
        port.engine
            .enable_archive(ArchiveConfig::new(root.clone(), repository))
            .unwrap();
        let position = port.engine.semantic.nodes[port.root].position.clone();
        port.engine
            .prepare_archive_root(&position, port.limits.deadline, &AtomicBool::new(false))
            .unwrap();
        let submitted = work(&mut port, "e2e4", RefinementStage::ProposalContinuation, 1);
        let (ticket, response) = output(&submitted, 1);
        let step = port
            .commit_role(&submitted, &ticket, &response, None)
            .unwrap();
        port.role_consumed(&ticket, &submitted.input.view).unwrap();
        drop(submitted);
        assert_eq!(port.engine.semantic.stores.review_pin_count().unwrap(), 0);
        let inactive = port
            .engine
            .semantic
            .stores
            .insert_situation(
                Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 100")
                    .unwrap()
                    .snapshot(),
            )
            .unwrap();
        let state = port
            .engine
            .semantic
            .stores
            .situations
            .get(inactive)
            .unwrap()
            .state;
        let authority = port.authority;
        let deadline = port.limits.deadline;
        let allowance = port.archive_remaining_io();
        let mut worker = ArchiveWorker::start().unwrap();
        let transfer = worker
            .submit(
                ArchiveOwner::Port(port),
                authority,
                ArchiveWork {
                    operation: ArchiveOperation::ReclaimHot {
                        incoming_pins: Default::default(),
                    },
                    deadline,
                    max_io_bytes: allowance,
                },
            )
            .unwrap_or_else(|_| panic!("archive owner transfer"));
        let completion = loop {
            if let Some(completed) = worker.try_receive(&transfer).unwrap() {
                break completed;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        };
        let (ArchiveOwner::Port(mut port), _) = completion
            .install(authority, transfer.nonce())
            .unwrap_or_else(|_| panic!("archive owner return"))
        else {
            panic!("port owner kind")
        };
        assert!(matches!(
            port.engine.semantic.stores.states.get(state),
            Err(StoreError::ColdRecord(_))
        ));
        assert!(matches!(
            port.materialize_step(step).unwrap().transition,
            FrameTransition::Continue(_)
        ));
        worker.finish().unwrap();
        drop(port);
        // Only this uniquely named, outside-repository fixture directory.
        if root.file_name().is_some_and(|name| {
            name.to_string_lossy()
                .starts_with("blueprint-archive-fixture-")
        }) && std::fs::symlink_metadata(&root)
            .is_ok_and(|metadata| !metadata.file_type().is_symlink())
        {
            std::fs::remove_dir_all(&root).unwrap();
        }
    }
}
