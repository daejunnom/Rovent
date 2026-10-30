//! Child of adaptive_root: only the opaque actual Ready handoff authorizes
//! consumption while its exact two original review leases remain alive.
use super::super::preparing_root::PreparingRegistrationSeal;
use super::*;

impl<M: RoleModel> PalsSemanticState<M> {
    pub(in crate::pals::engine) fn validate_unconsumed_preparing_registration(
        &self,
        registration: &AdaptiveRootRegistration,
        position: &Position,
        authority: SearchAuthority,
    ) -> Result<(), PalsError> {
        if !registration
            .registry
            .ptr_eq(&Arc::downgrade(&self.adaptive_root_registry.identity))
            || self.adaptive_root_registry.pending_nonce != Some(registration.nonce)
            || registration.authority != authority
            || !registration.owner.matches(&self.stores)
            || registration.generation != authority.root.0
            || self.stores.generation() != registration.generation
            || self.stores.root() != Some(registration.root)
            || self.game_generation != Some(authority.game.0)
            || !registration.snapshot.same_state(&position.snapshot())
            || !self
                .stores
                .states
                .get(self.stores.situations.get(registration.root)?.state)?
                .same_state(&registration.snapshot)
        {
            return Err(StoreError::StaleConsumer.into());
        }
        Ok(())
    }
    #[allow(clippy::result_large_err)]
    pub(in crate::pals::engine) fn consume_prepared_adaptive_root_registration(
        &mut self,
        registration: AdaptiveRootRegistration,
        position: &Position,
        authority: SearchAuthority,
        proof: &PreparingRegistrationSeal<'_>,
    ) -> Result<SituationId, (AdaptiveRootRegistration, PalsError)> {
        let checked = (|| -> Result<(), PalsError> {
            self.ensure_archive_admission_open()?;
            proof.validate(self)?;
            if !registration
                .registry
                .ptr_eq(&Arc::downgrade(&self.adaptive_root_registry.identity))
                || self.adaptive_root_registry.pending_nonce != Some(registration.nonce)
                || registration.authority != authority
                || !registration.owner.matches(&self.stores)
                || registration.generation != authority.root.0
                || self.stores.generation() != registration.generation
                || self.stores.root() != Some(registration.root)
                || self.game_generation != Some(authority.game.0)
                || !registration.snapshot.same_state(&position.snapshot())
                || !self
                    .stores
                    .states
                    .get(self.stores.situations.get(registration.root)?.state)?
                    .same_state(&registration.snapshot)
            {
                return Err(StoreError::StaleConsumer.into());
            }
            Ok(())
        })();
        match checked {
            Err(error) => Err((registration, error)),
            Ok(()) => {
                self.adaptive_root_registry.pending_nonce = None;
                Ok(registration.root)
            }
        }
    }
}
