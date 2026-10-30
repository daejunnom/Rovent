//! Final selection from the original checked Rules strategy. The legacy
//! integer/WDL estimate is not a mate distance or a proof authority.
use super::*;
use crate::pals::rovent_frontier::FrontierError;

impl<M: RoleModel> AdaptiveEnginePort<M> {
    pub(super) fn apply_rovent_mate_strategy(
        &self,
        mut result: PalsResult,
    ) -> Result<PalsResult, PalsError> {
        let Some(frontier) = self.rovent_frontier() else {
            return Ok(result);
        };
        let Some(strategy) = frontier.mate_strategy() else {
            return Ok(result);
        };
        let registration = frontier.registration();
        let actual = &self.engine.semantic.nodes[self.root];
        if registration.authority != self.authority
            || registration.original_deadline != self.deadline
            || registration.root_state != actual.state
            || !Arc::ptr_eq(&registration.root_rules, &self.rules)
            || self.engine.semantic.game_generation != Some(self.authority.game.0)
            || self.engine.semantic.stores.generation() != self.authority.root.0
            || !strategy
                .validates_root(&actual.position.snapshot(), frontier.root_moves().allowed())
        {
            return Err(FrontierError::ForeignRoot.into());
        }
        let movement = strategy.best_move();
        if !frontier.root_moves().allowed().contains(&movement) {
            return Err(FrontierError::ForeignRoot.into());
        }
        // This read-only lookup consumes the original paid edge. It neither
        // replays make_move nor turns the numerical resolver into a proof.
        let child = frontier
            .existing_rules_prefix(&[movement], &self.engine.semantic.stores)?
            .ok_or(FrontierError::MissingAcceptance)?;
        if !child.matches(frontier, &self.engine.semantic.stores)
            || !child
                .endpoint_snapshot()
                .same_state(strategy.checked_child_snapshot())
        {
            return Err(FrontierError::ForeignRoot.into());
        }
        result.best_move = Some(movement);
        // The original resolver's root_values remain numerical observations.
        // The separate Frontier receipt owns Within/Exact and root scope;
        // UCI may emit a mate score only for this same strategy and move.
        result.score = None;
        result.resolved_value = PalsResolvedValue::Unknown;
        result.value_scope = PalsValueScope::Unknown;
        Ok(result)
    }
}
