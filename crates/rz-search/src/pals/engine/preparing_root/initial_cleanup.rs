//! Cleanup-only return after an initial C request exists. Actual actor/worker
//! acknowledgements are distinct from live Ready and grant no search rights.
use super::*;
use crate::cpu_value::nnuec::v2::retirement::NnuecCacheReleaseAck;
use crate::cpu_value::nnuec::v2::review::{
    NnuecPreparationTerminalAck, Provenance, Request, RequestIdentity,
};
use crate::cpu_worker::{NnuecEngineReleaseAck, NnuecMaterializationReleaseAck};
use crate::pals::engine::adaptive_port::PreparingRootViewSeal;
use crate::pals::rovent_frontier::{FrontierGoalStatus, FrontierStop};

/// Each variant owns actual physical custody, not a caller success flag.
pub enum InitialResearcherRelease {
    Unsubmitted(Request),
    Preparation(NnuecPreparationTerminalAck),
    Materialization(NnuecMaterializationReleaseAck),
    Worker(NnuecEngineReleaseAck),
}
/// Only this root can issue the cleanup record after consuming its exact
/// original, non-Clone Request. It contains no strong View/Bound/model owner.
pub struct InitialUnsubmittedRequestReceipt {
    provenance: Provenance,
    _request: RequestIdentity,
}
impl InitialUnsubmittedRequestReceipt {
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}
pub enum InitialResearcherReleaseReceipt {
    Unsubmitted(InitialUnsubmittedRequestReceipt),
    Preparation(NnuecPreparationTerminalAck),
    Materialization(NnuecMaterializationReleaseAck),
    Worker(NnuecEngineReleaseAck),
}
/// An error retains the exact original release input until it was actually
/// retired. A post-release logical error keeps the resulting historical ack.
pub enum InitialPreparingReleaseCustody {
    Unreleased(InitialResearcherRelease),
    Released(InitialResearcherReleaseReceipt),
}
pub struct RetiredInitialPreparingRoot<M: RoleModel> {
    pub retired: RetiredPreparingRoot<M>,
    pub release: InitialResearcherReleaseReceipt,
    pub cache_release: NnuecCacheReleaseAck,
}
pub struct InitialPreparingRetirementRejection<M: RoleModel> {
    pub root: ResearcherPreparingRoot<M>,
    pub original_review_seal: Option<PreparingRootViewSeal>,
    pub release: InitialPreparingReleaseCustody,
    pub cache_release: NnuecCacheReleaseAck,
    pub error: ResearcherPreparingError,
}

impl<M: RoleModel> ResearcherPreparingRoot<M> {
    /// Unlike live preparation admission, this read-only borrowed authority
    /// tolerates the original cancel/deadline. It never selects a new View.
    fn cleanup_review_source(&self) -> Result<PreparingRootReviewSource<'_>, PalsError> {
        self.semantic.ensure_archive_admission_open()?;
        let view = self.initial_view.as_ref().ok_or(RoleError::Unavailable)?;
        let registration = self
            .registration
            .as_ref()
            .ok_or(StoreError::StaleConsumer)?;
        self.semantic.validate_unconsumed_preparing_registration(
            registration,
            &self.position,
            self.authority,
        )?;
        if self.semantic.game_generation != Some(self.authority.game.0)
            || self.semantic.stores.generation() != self.authority.root.0
            || self.retired_frontier.is_some()
            || self.semantic.rovent_frontier.is_none()
            || self.original.is_none()
            || self.semantic.stores.tasks.active_states().next().is_some()
            || self.semantic.nodes.iter().any(|node| node.resume.is_some())
            || self.semantic.stores.review_pin_count()? != 2
            || !self.semantic.nodes[self.root]
                .position
                .snapshot()
                .same_state(&self.position.snapshot())
            || !view.validates(&self.semantic.stores)
            || view.root_state() != self.root_state_id()
            || view.generation() != self.authority.root.0
            || !view.query().prefix.is_empty()
            || !view.query().proposal.is_empty()
            || view.query().counterexample.is_some()
        {
            return Err(PalsError::from(StoreError::StaleConsumer));
        }
        Ok(PreparingRootReviewSource {
            identity: &self.identity,
            stores: &self.semantic.stores,
            view,
            root: self.root_state_id(),
            authority: self.authority,
            original_deadline: self.original_deadline,
            hard_deadline: self.hard_deadline,
            position: &self.position,
            cancellation: &self.cancellation,
            source: &self.source,
        })
    }

    /// Same owner/allocator/root sequence returns only after the exact request
    /// is known absent or the same actor actually released C/TT/Bound/worker.
    /// Actor shutdown is deliberately not part of this reusable boundary.
    #[allow(clippy::result_large_err)]
    pub fn retire_after_initial_preparation(
        mut self,
        original_review_seal: PreparingRootViewSeal,
        release: InitialResearcherRelease,
        cache_release: NnuecCacheReleaseAck,
    ) -> Result<RetiredInitialPreparingRoot<M>, InitialPreparingRetirementRejection<M>> {
        let checked = (|| -> Result<(), ResearcherPreparingError> {
            let actual = self.cleanup_review_source()?;
            original_review_seal
                .validate_root_source(&actual)
                .map_err(|_| StoreError::StaleConsumer)
                .map_err(PalsError::from)?;
            let provenance = match &release {
                InitialResearcherRelease::Unsubmitted(request) => {
                    if !original_review_seal.matches_preparation_request(request) {
                        return Err(PalsError::from(StoreError::StaleConsumer).into());
                    }
                    request.provenance()
                }
                InitialResearcherRelease::Preparation(ack) => {
                    if !original_review_seal.matches_preparation_terminal(ack)
                        || !ack.matches_original(self.source.model(), actual.view())
                        || !ack.allows_c_only_cleanup()
                        || !ack.matches_cache_release(&cache_release)
                    {
                        return Err(PalsError::from(StoreError::StaleConsumer).into());
                    }
                    ack.provenance()
                }
                InitialResearcherRelease::Materialization(ack) => {
                    if !ack.matches_source(&self.source)
                        || !ack.matches_request(original_review_seal.preparation_request_identity())
                        || !ack.matches_cache_release(&cache_release)
                    {
                        return Err(PalsError::from(StoreError::StaleConsumer).into());
                    }
                    ack.provenance()
                }
                InitialResearcherRelease::Worker(ack) => {
                    if !ack.matches_source(&self.source)
                        || !ack.matches_request(original_review_seal.preparation_request_identity())
                        || !ack.matches_cache_release(&cache_release)
                        || ack.actual_stack_owner_id() == 0
                        || ack.profile().evaluation_namespace != ack.binding_digest()
                        || ack.profile().max_depth != self.source.cpu_plan().config().max_depth
                    {
                        return Err(PalsError::from(StoreError::StaleConsumer).into());
                    }
                    ack.provenance()
                }
            };
            if provenance.model != *self.source.model().identity()
                || provenance.root != self.root_state_id()
                || provenance.generation != self.authority.root.0
                || provenance.sequence != actual.view().sequence()
                || provenance.full_view_digest != actual.view().digest()
                || provenance.view_seal != actual.view().seal_digest()
                || provenance.record_scope_digest != actual.view().record_scope_digest()
                || provenance.policy_digest != actual.view().policy_digest()
            {
                return Err(PalsError::from(StoreError::StaleConsumer).into());
            }
            // The root and original seal are the only strong View holders
            // after release; an actually owned unsubmitted Request adds one.
            let expected =
                2 + usize::from(matches!(&release, InitialResearcherRelease::Unsubmitted(_)));
            if Arc::strong_count(actual.view()) != expected {
                return Err(PalsError::from(StoreError::InvalidConditions(
                    "initial preparation still has external View custody",
                ))
                .into());
            }
            let ledger = self
                .original
                .as_ref()
                .ok_or(PalsError::InvalidLimits)?
                .ledger
                .registration_observation()?;
            if ledger.active_jobs != 0
                || ledger.outstanding_executions != 0
                || ledger.reserved_native_nodes != 0
                || ledger.quarantined_jobs != 0
                || ledger.spent_native_nodes != 0
            {
                return Err(crate::pals::rovent_frontier::FrontierError::NativeUnknown.into());
            }
            let stopped = self.original_frontier_receipt().is_some_and(|receipt| {
                matches!(
                    receipt.snapshot.goal,
                    FrontierGoalStatus::Stopped(
                        FrontierStop::WorkBudget | FrontierStop::Deadline | FrontierStop::Canceled
                    ) | FrontierGoalStatus::DepthSatisfied
                        | FrontierGoalStatus::MateSatisfied
                )
            });
            if !stopped
                && !self.cancellation.load(Ordering::Acquire)
                && Instant::now() < self.hard_deadline
            {
                return Err(PalsError::InvalidLimits.into());
            }
            Ok(())
        })();
        if let Err(error) = checked {
            return Err(InitialPreparingRetirementRejection {
                root: self,
                original_review_seal: Some(original_review_seal),
                release: InitialPreparingReleaseCustody::Unreleased(release),
                cache_release,
                error,
            });
        }
        // Publish the original stopping condition before touching its leases.
        if self.cancellation.load(Ordering::Acquire) {
            self.cancel();
        } else if Instant::now() >= self.hard_deadline {
            let result = self
                .semantic
                .rovent_frontier
                .as_deref_mut()
                .ok_or(crate::pals::rovent_frontier::FrontierError::ForeignRoot)
                .and_then(|frontier| {
                    frontier.observe_original_hard_deadline_expiry(
                        self.authority,
                        self.original_deadline,
                    )
                });
            if result != Err(crate::pals::rovent_frontier::FrontierError::Deadline) {
                return Err(InitialPreparingRetirementRejection {
                    root: self,
                    original_review_seal: Some(original_review_seal),
                    release: InitialPreparingReleaseCustody::Unreleased(release),
                    cache_release,
                    error: crate::pals::rovent_frontier::FrontierError::Invalid(
                        "original expiry did not close preparing root",
                    )
                    .into(),
                });
            }
        }
        let release = match release {
            InitialResearcherRelease::Unsubmitted(request) => {
                let receipt = InitialUnsubmittedRequestReceipt {
                    provenance: request.provenance().clone(),
                    _request: request.identity(),
                };
                drop(request);
                InitialResearcherReleaseReceipt::Unsubmitted(receipt)
            }
            InitialResearcherRelease::Preparation(ack) => {
                InitialResearcherReleaseReceipt::Preparation(ack)
            }
            InitialResearcherRelease::Materialization(ack) => {
                InitialResearcherReleaseReceipt::Materialization(ack)
            }
            InitialResearcherRelease::Worker(ack) => InitialResearcherReleaseReceipt::Worker(ack),
        };
        drop(original_review_seal);
        drop(self.initial_view.take());
        match self.retire_before_materialization() {
            Ok(retired) => Ok(RetiredInitialPreparingRoot {
                retired,
                release,
                cache_release,
            }),
            Err((root, error)) => Err(InitialPreparingRetirementRejection {
                root,
                original_review_seal: None,
                release: InitialPreparingReleaseCustody::Released(release),
                cache_release,
                error,
            }),
        }
    }
}

#[cfg(test)]
#[path = "initial_cleanup_tests.rs"]
mod tests;
