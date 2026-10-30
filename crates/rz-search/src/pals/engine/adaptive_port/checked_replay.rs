//! Adopt only Rules snapshots already charged by the original goal owner.
//! The caller retains its original role/CPU acceptance; this bridge cannot
//! issue a model result, a native CPU receipt or another search authority.
use super::*;
use crate::pals::rovent_frontier::{CheckedFrontierLine, FrontierError};

/// The existing stage's deterministic choice, computed once before semantic
/// mutation. Reserving its Rules path must not add an examined graph edge and
/// thereby change the alternative selector before the real commit consumes it.
#[derive(Clone, Copy)]
pub(super) enum PreparedRoleChoice {
    None,
    Continuation(BoardMove),
    Divergence(Option<usize>),
    CriticReply(Option<BoardMove>),
    CriticRecheck(Option<BoardMove>),
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    pub(super) fn prepare_role_choice(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        output: &RoleOutput,
    ) -> Result<PreparedRoleChoice, PalsError> {
        let query = &work.input.question;
        Ok(match query.stage {
            RefinementStage::ProposalContinuation
            | RefinementStage::CounterContinuation
            | RefinementStage::Repair => PreparedRoleChoice::Continuation(
                *role_commit::rank_moves(&work.input.view.query().legal, output)?
                    .first()
                    .ok_or(RoleError::InvalidOutput)?,
            ),
            RefinementStage::CriticDivergence => {
                let RolePayload::Counterexample {
                    divergence_logits, ..
                } = &output.payload
                else {
                    return Err(RoleError::InvalidOutput.into());
                };
                let sites = &work.input.view.query().divergences;
                let mut ranked: Vec<_> = (0..sites.len()).collect();
                ranked.sort_by(|a, b| {
                    divergence_logits[*b]
                        .total_cmp(&divergence_logits[*a])
                        .then(a.cmp(b))
                });
                PreparedRoleChoice::Divergence(if ranked.is_empty() {
                    None
                } else {
                    Some(
                        sites[ranked[usize::try_from(self.root_round)
                            .map_err(|_| PalsError::InvalidLimits)?
                            % ranked.len()]],
                    )
                })
            }
            RefinementStage::CriticReply => {
                let ply = query.attack_ply.ok_or(RoleError::InvalidOutput)?;
                let original = *query.proposal.get(ply).ok_or(RoleError::InvalidOutput)?;
                let cpu_response = query
                    .counter
                    .as_ref()
                    .and_then(|line| line.get(ply))
                    .copied();
                let ranked = role_commit::rank_moves(&work.input.view.query().legal, output)?;
                let examined = &self.engine.semantic.nodes[query.node].edges;
                PreparedRoleChoice::CriticReply(
                    ranked
                        .iter()
                        .copied()
                        .find(|mv| {
                            *mv != original
                                && Some(*mv) != cpu_response
                                && !examined.iter().any(|edge| edge.movement == *mv)
                        })
                        .or_else(|| {
                            ranked
                                .iter()
                                .copied()
                                .find(|mv| *mv != original && Some(*mv) != cpu_response)
                        }),
                )
            }
            RefinementStage::CriticRecheck => {
                let ranked = role_commit::rank_moves(&work.input.view.query().legal, output)?;
                PreparedRoleChoice::CriticRecheck(if query.recheck {
                    ranked.first().copied()
                } else {
                    let old = *query
                        .repaired
                        .as_ref()
                        .and_then(|line| line.get(query.context.prefix.len()))
                        .ok_or(RoleError::InvalidOutput)?;
                    let examined = &self.engine.semantic.nodes[query.node].edges;
                    ranked
                        .iter()
                        .copied()
                        .find(|mv| *mv != old && !examined.iter().any(|edge| edge.movement == *mv))
                        .or_else(|| ranked.iter().copied().find(|mv| *mv != old))
                })
            }
            _ => PreparedRoleChoice::None,
        })
    }

    pub(super) fn preflight_role_choice(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        choice: PreparedRoleChoice,
    ) -> Result<(), PalsError> {
        if self.rovent_frontier().is_none() {
            return Ok(());
        }
        let query = &work.input.question;
        let prefix = match choice {
            PreparedRoleChoice::Continuation(movement)
            | PreparedRoleChoice::CriticReply(Some(movement))
            | PreparedRoleChoice::CriticRecheck(Some(movement)) => {
                let mut prefix = query.context.prefix.clone();
                prefix.push(movement);
                prefix
            }
            PreparedRoleChoice::Divergence(Some(ply)) => query
                .proposal
                .get(..ply)
                .ok_or(RoleError::InvalidOutput)?
                .to_vec(),
            _ => return Ok(()),
        };
        self.checked_frontier_line(&prefix)?;
        Ok(())
    }

    pub(super) fn checked_frontier_line(
        &mut self,
        prefix: &[BoardMove],
    ) -> Result<CheckedFrontierLine, PalsError> {
        self.check_rovent_scan_live()?;
        self.with_rovent_frontier(|frontier, stores| {
            let (clock, now) = frontier.observed_original_tick()?;
            frontier.observe_rules_prefix(prefix, stores, clock, now)
        })?
        .ok_or_else(|| FrontierError::Invalid("original frontier absent").into())
    }

    pub(super) fn existing_checked_frontier_line(
        &self,
        prefix: &[BoardMove],
    ) -> Result<Option<CheckedFrontierLine>, PalsError> {
        self.rovent_frontier()
            .ok_or(FrontierError::Invalid("original frontier absent"))?
            .existing_rules_prefix(prefix, &self.engine.semantic.stores)
            .map_err(Into::into)
    }

    pub(super) fn replay_checked_frontier(
        &mut self,
        prefix: &[BoardMove],
    ) -> Result<usize, PalsError> {
        let checked = match self.existing_checked_frontier_line(prefix)? {
            Some(checked) => checked,
            None => self.checked_frontier_line(prefix)?,
        };
        self.adopt_checked_frontier_line(&checked)
    }

    /// The line was checked under the original quota before it reached this
    /// owner. A partial hot allocation failure retains that charged evidence;
    /// an exact retry adopts existing snapshots without replay or refund.
    pub(super) fn adopt_checked_frontier_line(
        &mut self,
        checked: &CheckedFrontierLine,
    ) -> Result<usize, PalsError> {
        if !self
            .rovent_frontier()
            .is_some_and(|frontier| checked.matches(frontier, &self.engine.semantic.stores))
            || !checked
                .root_snapshot()
                .same_state(&self.engine.semantic.nodes[self.root].position.snapshot())
        {
            return Err(FrontierError::ForeignRoot.into());
        }
        let mut node = self.root;
        for step in checked.steps() {
            if let Some(edge) = self.engine.semantic.nodes[node]
                .edges
                .iter()
                .find(|edge| edge.movement == step.movement())
            {
                if !self.engine.semantic.nodes[edge.child]
                    .position
                    .snapshot()
                    .same_state(step.snapshot())
                {
                    return Err(FrontierError::ForeignRoot.into());
                }
                node = edge.child;
                continue;
            }
            let position = Position::from_snapshot(step.snapshot(), Default::default())?;
            let child = Self::intern_hot(&mut self.engine, position)?;
            self.engine.semantic.nodes[node].edges.push(Edge {
                movement: step.movement(),
                child,
            });
            self.counters.examined_edges = self
                .counters
                .examined_edges
                .checked_add(1)
                .ok_or(PalsError::Capacity)?;
            node = child;
        }
        if !self.engine.semantic.nodes[node]
            .position
            .snapshot()
            .same_state(checked.endpoint_snapshot())
        {
            return Err(FrontierError::ForeignRoot.into());
        }
        Ok(node)
    }
}
