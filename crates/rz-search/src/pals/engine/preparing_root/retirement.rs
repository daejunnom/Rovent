//! Move the same semantic owner back only after actual engine/cache release.
//!
//! The actual session first proves native cleanup and obtains port.into_engine.
//! This boundary neither joins a worker nor drops an engine/TT on the manager.
use super::*;
use crate::cpu_value::nnuec::v2::retirement::NnuecCacheReleaseAck;
use crate::cpu_worker::{
    CpuWorkerProfile, NnuecEngineReleaseAck, NnuecReadyRegistration, ValueInstallReceipt,
};

/// Copy only from the actual one-use initial Ready handoff. Keeping the Ready
/// itself here would retain its old Bound/view pin and prevent quiescence.
pub(in crate::pals::engine) struct DeferredResearcherInitialReadback {
    profile: CpuWorkerProfile,
    stack_owner: u64,
    binding: rz_contracts::Digest,
}
impl DeferredResearcherInitialReadback {
    pub(super) fn from_ready(ready: &NnuecReadyRegistration) -> Self {
        Self {
            profile: ready.profile(),
            stack_owner: ready.actual_stack_owner_id(),
            binding: ready.binding_digest(),
        }
    }
}

/// This selects an already attested readback. Neither variant issues release
/// authority: the private actual actor ack must match the original worker.
pub enum ReleasedResearcherReadback<'a> {
    Initial,
    Installed(&'a ValueInstallReceipt),
}

pub struct RetiredResearcherOwner<M: RoleModel> {
    pub owner: ResearcherPreparingOwner<M>,
    pub engine_release: NnuecEngineReleaseAck,
    pub cache_release: NnuecCacheReleaseAck,
}

pub struct ResearcherRetirementRejection<M: RoleModel> {
    pub engine: PalsEngine<M>,
    pub engine_release: NnuecEngineReleaseAck,
    pub cache_release: NnuecCacheReleaseAck,
    pub error: PalsError,
}

impl<M: RoleModel> PalsEngine<M> {
    /// Every rejecting path retains the original semantic owner and actual
    /// cleanup witnesses. A stale/unknown physical owner is never IdleReady.
    #[allow(clippy::result_large_err)]
    pub fn into_preparing_owner(
        self,
        engine_release: NnuecEngineReleaseAck,
        cache_release: NnuecCacheReleaseAck,
        final_readback: ReleasedResearcherReadback<'_>,
    ) -> Result<RetiredResearcherOwner<M>, ResearcherRetirementRejection<M>> {
        let checked =
            (|| -> Result<(), PalsError> {
                let source = self.deferred_researcher_source.as_ref().ok_or(
                    StoreError::InvalidConditions("no actual deferred researcher source"),
                )?;
                let initial = self.deferred_researcher_initial.as_ref().ok_or(
                    StoreError::InvalidConditions("no actual initial researcher readback"),
                )?;
                self.semantic.ensure_archive_admission_open()?;
                self.semantic.ensure_adaptive_logical_quiescence()?;
                if self.held_cpu_task.is_some()
                    || !engine_release.matches_source(source)
                    || !engine_release.matches_cache_release(&cache_release)
                    || engine_release.owner() != initial.profile.owner
                    || engine_release.model_identity() != source.model().identity()
                    || self.researcher.cpu_registered_value.as_ref()
                        != Some(engine_release.model_identity())
                {
                    return Err(StoreError::StaleConsumer.into());
                }
                let authority = self
                    .semantic
                    .last_adaptive_authority()
                    .ok_or(StoreError::StaleConsumer)?;
                let root = self
                    .semantic
                    .stores
                    .root()
                    .ok_or(StoreError::StaleConsumer)?;
                let root_state = self.semantic.stores.situations.get(root)?.state;
                if engine_release.provenance().root != root_state
                    || engine_release.provenance().generation != authority.root.0
                    || self.semantic.stores.generation() != authority.root.0
                    || self.semantic.game_generation != Some(authority.game.0)
                {
                    return Err(StoreError::StaleConsumer.into());
                }
                let (profile, stack_owner, binding) = match final_readback {
                    ReleasedResearcherReadback::Initial => {
                        (initial.profile, initial.stack_owner, initial.binding)
                    }
                    ReleasedResearcherReadback::Installed(receipt) => {
                        let readback = receipt.result().map_err(|_| {
                            StoreError::InvalidEvidence("last researcher installation failed")
                        })?;
                        (
                            readback.profile(),
                            readback.actual_stack_owner_id(),
                            receipt.binding_digest(),
                        )
                    }
                };
                if profile != engine_release.profile()
                    || profile.owner != initial.profile.owner
                    || profile.evaluation_namespace != binding
                    || stack_owner == 0
                    || stack_owner != engine_release.actual_stack_owner_id()
                    || binding != engine_release.binding_digest()
                {
                    return Err(StoreError::InvalidEvidence(
                        "released researcher readback changed",
                    )
                    .into());
                }
                if self.rovent_root_prefix_plies.is_none() {
                    return Err(PalsError::InvalidConfig);
                }
                Ok(())
            })();
        if let Err(error) = checked {
            return Err(ResearcherRetirementRejection {
                engine: self,
                engine_release,
                cache_release,
                error,
            });
        }
        let PalsEngine {
            config,
            resolver,
            post_repair_recheck,
            rovent_root_prefix_plies,
            mut deferred_researcher_source,
            semantic,
            ..
        } = self;
        Ok(RetiredResearcherOwner {
            owner: ResearcherPreparingOwner {
                semantic,
                profile: ResearcherSemanticProfile {
                    config,
                    maximum_root_prefix_plies: rovent_root_prefix_plies
                        .expect("actual optional researcher profile was validated"),
                },
                resolver,
                post_repair_recheck,
                source: deferred_researcher_source
                    .take()
                    .expect("actual Source declaration was validated"),
            },
            engine_release,
            cache_release,
        })
    }
}
