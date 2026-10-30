//! The pre-researcher owner resets semantic state only after original fences.
use super::*;

impl<M: RoleModel> ResearcherPreparingOwner<M> {
    pub fn last_adaptive_authority(&self) -> Option<SearchAuthority> {
        self.semantic.last_adaptive_authority()
    }
    pub fn source(&self) -> &DeferredNnuecRSource {
        &self.source
    }
    pub fn profile(&self) -> &ResearcherSemanticProfile {
        &self.profile
    }
    pub fn try_adaptive_new_game(
        &mut self,
        next: SearchAuthority,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        self.semantic.try_adaptive_semantic_new_game(
            &self.profile.sequence_config(),
            next,
            deadline,
            cancel,
        )
    }
}
