//! Actual Fresh head use is separate from a later legacy Store commit.
//! This module is registered with the corresponding private source-state
//! changes, never used as a caller-declared acceptance or a replacement NN.
use super::*;

pub(in crate::pals::engine::adaptive_port) struct PreparedFreshSourceUse {
    source: ConsumedRoleSource,
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Validate/capacity-check before the controller reads the Fresh head.
    /// The permit owns original immutable custody and carries no new credit.
    pub(in crate::pals::engine::adaptive_port) fn prepare_fresh_source_use(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        accepted: &AcceptedRoleStep,
        output: &RoleOutput,
        physical: AdaptivePortRoleEvidence,
    ) -> Result<Option<PreparedFreshSourceUse>, PalsError> {
        self.validate_role(work, accepted.ticket(), output)?;
        if !accepted.validates_output(output)
            || work.input.question.context.purpose != RoleQueryPurpose::ValueFresh
            || work.input.question.first.is_none()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        self.checked_role_value(work, accepted.ticket(), output, Some(physical))?;
        if self.reciprocal_source_consumed(work, accepted.ticket(), output) {
            return match self
                .reciprocal
                .consumed_sources
                .get(&accepted.ticket().id)
                .and_then(|source| source.failure.clone())
            {
                Some(error) => Err(error),
                None => Ok(None),
            };
        }
        if self.counters.role_calls >= self.engine.config.max_role_calls {
            return Err(PalsError::RoleCallLimit);
        }
        if self.reciprocal.consumed_sources.len() >= MAX_COORDINATOR_FRAMES {
            return Err(PalsError::Capacity);
        }
        if self
            .committed_role_questions
            .get(&accepted.ticket().id)
            .is_some_and(|question| *question != accepted.ticket().question)
            || self
                .reciprocal
                .consumed_sources
                .contains_key(&accepted.ticket().id)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        Ok(Some(PreparedFreshSourceUse {
            source: ConsumedRoleSource {
                frame: work.frame,
                original_first_move: work.original_first_move,
                ticket: *accepted.ticket(),
                output: accepted.output_digest(),
                question: Arc::clone(&work.input.question),
                view: Arc::clone(&work.input.view),
                failure: None,
            },
        }))
    }

    /// Invoke only after the original controller actually used this head.
    /// There is deliberately no new live/deadline admission here: use has
    /// already occurred and must remain visible even if bookkeeping fails.
    pub(in crate::pals::engine::adaptive_port) fn record_used_fresh_source(
        &mut self,
        permit: PreparedFreshSourceUse,
    ) -> Result<(), PalsError> {
        let source = permit.source;
        let ticket = source.ticket;
        let view = Arc::clone(&source.view);
        self.accepted.insert(ticket.id);
        self.committed_role_questions
            .insert(ticket.id, ticket.question);
        self.reciprocal.consumed_sources.insert(ticket.id, source);
        self.account_reciprocal_role(RoleQueryPurpose::ValueFresh);
        let consumed = self.role_consumed(&ticket, &view);
        if let Err(error) = &consumed {
            // The actual head use cannot be refunded or reclassified unused.
            // Keep protection/Store failure distinct from successful consume.
            self.reciprocal
                .consumed_sources
                .get_mut(&ticket.id)
                .expect("original Fresh source was just recorded")
                .failure = Some(error.clone());
        }
        consumed
    }

    /// Readback of actual head use, including a retained later failure. This
    /// does not claim that review_index protection or Store commit succeeded.
    pub(in crate::pals::engine::adaptive_port) fn reciprocal_head_use_recorded(
        &self,
        ticket: &RoleTicketEnvelope,
        view: &ReviewView,
    ) -> bool {
        self.reciprocal
            .consumed_sources
            .get(&ticket.id)
            .is_some_and(|source| {
                source.ticket == *ticket
                    && source.ticket.authority == self.authority
                    && source.ticket.view == self.seal(view)
                    && std::ptr::eq(source.view.as_ref(), view)
                    && self.accepted.contains(&ticket.id)
                    && self.committed_role_questions.get(&ticket.id) == Some(&ticket.question)
            })
    }

    pub(in crate::pals::engine::adaptive_port) fn reciprocal_source_failure(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Option<PalsError> {
        if !self.reciprocal_source_consumed(work, ticket, output) {
            return None;
        }
        self.reciprocal
            .consumed_sources
            .get(&ticket.id)
            .and_then(|source| source.failure.clone())
    }
}
