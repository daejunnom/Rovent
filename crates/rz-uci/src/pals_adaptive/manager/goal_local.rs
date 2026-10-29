//! Native P/C tickets for controller-issued local duties. No ordinary frame,
//! ranking proof, reciprocal parent or R stack is manufactured for this lane.
use super::*;

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    /// Called only after the explicit ordinary root-round selection seal. One
    /// turn asks the owner for at most one paid edge or one immutable input.
    pub(super) fn drive_goal_local(&mut self) -> Result<bool, AdaptiveManagerError> {
        if !self.can_admit_work() || self.utility_scope_open() {
            return Ok(false);
        }
        if self.driver.rovent_root_budget()?.is_none() {
            return Ok(false);
        }
        let now = self.check_running()?;
        if self.goal_prepared.is_none() {
            match self
                .driver
                .next_rovent_goal_input(self.clock.domain(), now)?
            {
                AdaptiveGoalLocalAdmission::RoleInput(original) => {
                    if original.authority() != self.registration.authority
                        || original.original_deadline() != self.deadline
                        || Some(original.root_budget()) != self.driver.rovent_root_budget()?
                        || original.input().stage.role().is_none()
                    {
                        // Preserve the original owner even on a bad echo.
                        self.goal_prepared = Some(original);
                        return Err(AdaptiveManagerError::Invalid("GoalLocal root registration"));
                    }
                    self.goal_prepared = Some(original);
                    self.goal_continuation_pending = true;
                }
                AdaptiveGoalLocalAdmission::RulesAdvanced { .. } => {
                    self.goal_continuation_pending = true;
                    return Ok(true);
                }
                AdaptiveGoalLocalAdmission::NoQuestion => {
                    if self.driver.rovent_goal_has_pending()? {
                        return Err(AdaptiveManagerError::Invalid(
                            "GoalLocal NoQuestion still retains an unowned duty",
                        ));
                    }
                    let changed = self.goal_continuation_pending;
                    self.goal_continuation_pending = false;
                    return Ok(changed);
                }
                AdaptiveGoalLocalAdmission::Blocked { reason } => {
                    self.goal_continuation_pending = true;
                    if let GoalLocalBlockReason::Frontier(FrontierError::NodeBudget) = reason {
                        return self.native_admission_result(Err(AdaptiveManagerError::Frontier(
                            FrontierError::NodeBudget,
                        )));
                    }
                    // A real reservation can settle; a copied snapshot cannot
                    // justify an unbounded retry after all owners are idle.
                    if matches!(
                        reason,
                        GoalLocalBlockReason::Frontier(FrontierError::NodesReserved)
                    ) && self.semantic_replay_can_wait(&AdaptiveManagerError::Frontier(
                        FrontierError::NodesReserved,
                    ))? {
                        return Ok(false);
                    }
                    if reason == GoalLocalBlockReason::ViewLeaseCapacity
                        && self.review_lease_return_can_progress()
                    {
                        return Ok(false);
                    }
                    return Err(AdaptiveManagerError::GoalLocalBlocked(reason));
                }
            }
        }
        if self.roles.len() >= self.role_pending_limit() || !self.role_call_admission_available() {
            return Ok(false);
        }
        let original = self
            .goal_prepared
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "GoalLocal prepared custody missing",
            ))?;
        let input = original.input().clone();
        let preparation = match self.prepare_role(None, input, RolePurpose::RoventGoalLocal) {
            Ok(preparation) => preparation,
            Err(AdaptiveManagerError::Native(TicketServiceError::Capacity)) => return Ok(false),
            Err(error) => return Err(error),
        };
        let original = self
            .goal_prepared
            .take()
            .ok_or(AdaptiveManagerError::Invalid(
                "GoalLocal original source disappeared",
            ))?;
        let binding = self
            .roles
            .iter_mut()
            .find(|binding| binding.preparation == preparation)
            .ok_or(AdaptiveManagerError::Invalid(
                "GoalLocal native preparation missing",
            ))?;
        binding.goal_local = Some(original);
        Ok(true)
    }

    pub(super) fn consume_goal_local_role(
        &mut self,
        index: usize,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        physical: Option<TicketExecutionEvidence>,
    ) -> Result<(), AdaptiveManagerError> {
        let binding = &self.roles[index];
        let original = binding
            .goal_local
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "GoalLocal retired source missing",
            ))?;
        if binding.work.is_some() || original.input().question_digest != ticket.question {
            return Err(AdaptiveManagerError::Invalid(
                "GoalLocal became an ordinary frame",
            ));
        }
        let proof = binding
            .reciprocal_acceptance
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "GoalLocal has no physically retired acceptance",
            ))?;
        let now = self.check_running()?;
        let step = self.driver.consume_rovent_goal_input(
            original,
            proof,
            output,
            physical,
            self.clock.domain(),
            now,
        )?;
        match step {
            GoalLocalStep::ContinuePrepared { key } if key != original.key() => {
                return Err(AdaptiveManagerError::Invalid(
                    "GoalLocal continuation changed origin",
                ));
            }
            GoalLocalStep::Leaf(leaf) if leaf.source_key() != original.key() => {
                return Err(AdaptiveManagerError::Invalid(
                    "GoalLocal leaf changed origin",
                ));
            }
            _ => {}
        }
        self.mark_role_consumed(index, false)?;
        if self.has_collection_failure() {
            return Ok(());
        }
        // A local leaf may be only shallow horizon progress. Only the owner's
        // next explicit NoQuestion can release the root continuation barrier.
        self.goal_continuation_pending = true;
        Ok(())
    }

    /// This runs after the real native fence but before archive closure. Failed
    /// discards retain their original opaque sources for an owner retry.
    pub(super) fn discard_goal_local_custody(&mut self) -> Result<(), AdaptiveManagerError> {
        if let Some(original) = &self.goal_prepared {
            self.driver.discard_rovent_goal_input(original)?;
            self.goal_prepared = None;
        }
        for binding in &mut self.roles {
            if let Some(original) = &binding.goal_local {
                self.driver.discard_rovent_goal_input(original)?;
                binding.goal_local = None;
            }
        }
        self.goal_continuation_pending = false;
        Ok(())
    }
}
