//! Semantic game reset before an actual researcher exists.
//! The caller first proves its native/actor fence and releases old view custody.
//! No researcher, TT, worker reset or physical receipt is created by this file.
use super::*;

impl<M: RoleModel> PalsSemanticState<M> {
    pub(in crate::pals::engine) fn try_adaptive_semantic_new_game(
        &mut self,
        config: &PalsSemanticConfig,
        next: SearchAuthority,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        self.ensure_archive_admission_open()?;
        self.ensure_adaptive_logical_quiescence()?;
        self.validate_adaptive_namespace(next)?;
        let old = self.game_generation.ok_or(StoreError::StaleConsumer)?;
        if next.game.0 <= old || self.adaptive_root_registry.pending_nonce.is_some() {
            return Err(StoreError::StaleConsumer.into());
        }
        if self.stores.archive_enabled() || self.archive_is_configured() {
            return Err(StoreError::InvalidConditions(
                "R-free archive game reset requires its original archive lifecycle owner",
            )
            .into());
        }
        allocation_controls(Some((deadline, cancel)))?;
        self.model.preflight_new_game(deadline, cancel)?;
        let next_store = PalsStores::new(config.store_limits());
        allocation_controls(Some((deadline, cancel)))?;
        // Validation/preflight finishes before the old logical owner mutates.
        // The registry itself remains in place, including its monotonic nonce.
        self.model.new_game_with_generation(Some(next.game.0));
        self.nodes.clear();
        self.records.clear();
        self.reset_review_context();
        self.stores = next_store;
        self.game_generation = Some(next.game.0);
        self.adaptive_root_registry.new_game_ready = Some(next);
        self.revision = 0;
        self.consumer_id = 0;
        self.last_search_counters = None;
        self.last_recheck_observer_error = None;
        self.repair_queue.clear();
        self.repair_questions.clear();
        self.repair_counts.clear();
        self.new_game_transition_error = None;
        Ok(())
    }
}

impl<M: RoleModel> PcSemanticOwner<M> {
    pub fn try_adaptive_new_game(
        &mut self,
        next: SearchAuthority,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        let config = self.profile.sequence_config();
        self.semantic
            .try_adaptive_semantic_new_game(&config, next, deadline, cancel)
    }
}
