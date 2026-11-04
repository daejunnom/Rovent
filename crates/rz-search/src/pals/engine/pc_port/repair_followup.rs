//! Iteration admission reuses an actual completed C frontier, independently of
//! whether its conditional frozen-WDL comparison produced a known label.
use super::*;

pub(super) struct PreparedPcRepairFollowup {
    step: PcRepairStep,
    counter: PublishedPcLine,
    proposal: PublishedPcLine,
}

impl<M: RoleModel> PcSemanticPort<M> {
    pub(super) fn prepare_completed_repair_followup(
        &mut self,
        completed: u64,
    ) -> Result<Option<PreparedPcRepairFollowup>, PalsError> {
        let cycles = self
            .repair_cycles
            .as_ref()
            .ok_or(PalsError::InvalidConfig)?;
        let facts = cycles
            .active
            .get(&completed)
            .ok_or(RoleError::InvalidOutput)?;
        let recipe = facts.followup.as_ref().ok_or(RoleError::InvalidOutput)?;
        let count = *cycles
            .per_first
            .get(&recipe.origin.first)
            .ok_or(RoleError::InvalidOutput)?;
        if count >= MAX_PER_FIRST {
            return Ok(None);
        }
        if recipe.phase != PcRepairPhase::FreezePair
            || recipe.cycle != completed
            || recipe.frozen.is_some()
            || recipe.frozen_context.is_some()
            || facts.repair_endpoint.is_none()
            || facts.counter_endpoint.is_none()
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let repair = facts.repair.as_ref().ok_or(RoleError::InvalidOutput)?;
        let recheck = facts.recheck.as_ref().ok_or(RoleError::InvalidOutput)?;
        let published = facts.recheck_publication.ok_or(RoleError::InvalidOutput)?;
        let counter = recipe.counter.as_ref().ok_or(RoleError::InvalidOutput)?;
        let observation = self
            .owner
            .semantic
            .stores
            .observations
            .get(published.observation)?;
        if published.kind != RecordKind::Counterexample
            || published.line != recheck.line()
            || published.observation != recheck.observation()
            || published.revision != recheck.revision()
            || recipe.proposal != repair.moves()
            || counter.as_slice() != recheck.moves()
            || self.owner.semantic.stores.lines.moves(published.line)? != *counter
            || observation.kind != ObservationKind::Refutation
            || observation.line != Some(published.line)
            || counter.first() != Some(&recipe.origin.first)
        {
            return Err(RoleError::InvalidOutput.into());
        }
        let attack = counter
            .iter()
            .zip(&recipe.proposal)
            .position(|(current, previous)| current != previous)
            .ok_or(RoleError::InvalidOutput)?;
        let attack_node = self.repair_node(&counter[..attack])?;
        let root_side = self.owner.semantic.nodes[self.root].position.side_to_move();
        if attack == 0
            || self.owner.semantic.nodes[attack_node]
                .position
                .side_to_move()
                == root_side
            || !counter.starts_with(&recipe.proposal[..attack])
        {
            return Err(RoleError::InvalidOutput.into());
        }
        // After this exact C response there must still be a P move and a
        // subsequent C turn within the SAME original bounded line horizon.
        if attack
            .checked_add(2)
            .is_none_or(|last| last >= recipe.origin.end)
        {
            return Ok(None);
        }
        let id = cycles.next_cycle;
        id.checked_add(1).ok_or(PalsError::Capacity)?;
        if cycles.active.len() > MAX_CYCLES || self.ready.len() >= MAX_PC_PENDING {
            return Err(PalsError::Capacity);
        }
        let mut step = recipe.clone();
        step.cycle = id;
        step.parent_cycle = Some(completed);
        step.phase = PcRepairPhase::VisibleCounterReview;
        step.prefix.clear();
        step.attack = Some(attack);
        step.divergences = vec![attack];
        step.repaired = None;
        step.excluded = None;
        let proposal = PublishedPcLine {
            line: repair.line(),
            observation: repair.observation(),
            revision: repair.revision(),
            kind: repair.kind(),
        };
        // Reserve before retiring any old fact. Neither the comparison label
        // nor the latest public view supplies an alternative admission right.
        self.ready.try_reserve(1).map_err(|_| PalsError::Capacity)?;
        self.repair_cycles
            .as_mut()
            .ok_or(PalsError::InvalidConfig)?
            .active
            .try_reserve(1)
            .map_err(|_| PalsError::Capacity)?;
        Ok(Some(PreparedPcRepairFollowup {
            step,
            counter: published,
            proposal,
        }))
    }

    pub(super) fn enqueue_completed_repair_followup(
        &mut self,
        followup: PreparedPcRepairFollowup,
    ) -> Result<(), PalsError> {
        let PreparedPcRepairFollowup {
            step,
            counter,
            proposal,
        } = followup;
        let cycles = self
            .repair_cycles
            .as_mut()
            .ok_or(PalsError::InvalidConfig)?;
        let count = cycles
            .per_first
            .get_mut(&step.origin.first)
            .ok_or(RoleError::InvalidOutput)?;
        if *count >= MAX_PER_FIRST || cycles.active.len() >= MAX_CYCLES {
            return Err(PalsError::Capacity);
        }
        let id = step.cycle;
        let next = id.checked_add(1).ok_or(PalsError::Capacity)?;
        if cycles.next_cycle != id || cycles.active.contains_key(&id) {
            return Err(RoleError::InvalidOutput.into());
        }
        cycles.active.insert(
            id,
            CycleFacts {
                phase: step.phase,
                stopped_reason: None,
                progress: super::super::repair_report::PcRepairProgress::capture(&step),
                last_accepted: None,
                proposal: Some(proposal),
                counter: Some(counter),
                previous_repair: Some(proposal),
                target: None,
                attack_source: None,
                repair: None,
                recheck: None,
                recheck_publication: None,
                followup: None,
                repair_endpoint: None,
                counter_endpoint: None,
            },
        );
        cycles.next_cycle = next;
        *count += 1;
        // Root inspection reads the actual new counter and no fabricated
        // policy response. The following accepted C→P question owns its peer.
        self.push_repair_plan(self.root, step, None)
    }
}
