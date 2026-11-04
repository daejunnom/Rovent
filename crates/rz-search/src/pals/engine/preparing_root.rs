//! Same-Store preparation before the first real native researcher exists.
//!
//! This is neither PalsEngine nor NoResearcher. Its semantic graph, pending
//! logical root registration and actual first view move into the registered
//! researcher assembly only after the preparation actor and worker Ready.
use super::archive_lifecycle::allocation_controls;
use super::*;
use crate::cpu_worker::DeferredNnuecRSource;
use rz_contracts::Deadline;
use rz_contracts::pals::SearchAuthority;

mod controls;
mod handoff;
mod initial_cleanup;
mod lifecycle;
mod retirement;
pub use controls::{ResearcherPreparingError, ResearcherRootStartup, RetiredPreparingRoot};
pub(super) use handoff::PreparingRegistrationSeal;
pub use handoff::{PreparingRootFinishRejection, PreparingRootHandoff};
pub use initial_cleanup::{
    InitialPreparingReleaseCustody, InitialPreparingRetirementRejection, InitialResearcherRelease,
    InitialResearcherReleaseReceipt, InitialUnsubmittedRequestReceipt, RetiredInitialPreparingRoot,
};
pub(super) use retirement::DeferredResearcherInitialReadback;
pub use retirement::{
    ReleasedResearcherReadback, ResearcherRetirementRejection, RetiredResearcherOwner,
};

/// Root-independent semantic owner while no actual researcher exists.
/// Early bounded roots return this same owner, not a newly allocated Store.
pub struct ResearcherPreparingOwner<M: RoleModel> {
    pub(super) semantic: PalsSemanticState<M>,
    profile: ResearcherSemanticProfile,
    resolver: ResolverPolicy,
    post_repair_recheck: PostRepairRecheckPolicy,
    source: DeferredNnuecRSource,
}

pub struct ResearcherPreparingRoot<M: RoleModel> {
    pub(super) semantic: PalsSemanticState<M>,
    pub(super) profile: ResearcherSemanticProfile,
    pub(super) resolver: ResolverPolicy,
    pub(super) post_repair_recheck: PostRepairRecheckPolicy,
    root: usize,
    authority: SearchAuthority,
    original_deadline: Deadline,
    hard_deadline: Instant,
    cancellation: Arc<AtomicBool>,
    identity: Arc<()>,
    source: DeferredNnuecRSource,
    position: Position,
    registration: Option<adaptive_root::AdaptiveRootRegistration>,
    initial_view: Option<Arc<ReviewView>>,
    original: Option<controls::OriginalRootControls>,
    retired_frontier: Option<adaptive_port::AdaptiveFrontierReceipt>,
}

/// Only the actual preparing owner can construct this borrowed authority.
/// Its private fields cannot be rebuilt from a copied digest or raw metadata.
pub(crate) struct PreparingRootReviewSource<'a> {
    identity: &'a Arc<()>,
    stores: &'a PalsStores,
    view: &'a Arc<ReviewView>,
    root: StateId,
    authority: SearchAuthority,
    original_deadline: Deadline,
    hard_deadline: Instant,
    position: &'a Position,
    cancellation: &'a Arc<AtomicBool>,
    source: &'a DeferredNnuecRSource,
}
impl PreparingRootReviewSource<'_> {
    pub(crate) fn identity(&self) -> &Arc<()> {
        self.identity
    }
    pub(crate) fn stores(&self) -> &PalsStores {
        self.stores
    }
    pub(crate) fn view(&self) -> &Arc<ReviewView> {
        self.view
    }
    pub(crate) fn root_state(&self) -> StateId {
        self.root
    }
    pub(crate) fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub(crate) fn original_deadline(&self) -> Deadline {
        self.original_deadline
    }
    pub(crate) fn hard_deadline(&self) -> Instant {
        self.hard_deadline
    }
    pub(crate) fn position(&self) -> &Position {
        self.position
    }
    pub(crate) fn cancellation(&self) -> &Arc<AtomicBool> {
        self.cancellation
    }
    pub(crate) fn source(&self) -> &DeferredNnuecRSource {
        self.source
    }
}

impl<M: RoleModel> ResearcherPreparingOwner<M> {
    /// Allocates only common semantic structures. It creates no CpuEngine,
    /// checker, worker, TT or native job reservation.
    #[allow(clippy::too_many_arguments)]
    pub fn new_selected(
        profile: ResearcherSemanticProfile,
        model: M,
        resolver: ResolverPolicy,
        post_repair_recheck: PostRepairRecheckPolicy,
        review_policy: ReviewSelectionPolicy,
        review_limits: ReviewIndexLimits,
        source: DeferredNnuecRSource,
    ) -> Result<Self, PalsError> {
        profile.validate()?;
        let value = model
            .value_identity()
            .ok_or(RoleError::Unavailable)?
            .clone();
        value.validate()?;
        if value.model != model.identity() || value.semantics != MODEL_WDL_VALUE_SEMANTICS {
            return Err(RoleError::InvalidOutput.into());
        }
        model
            .registered_representation_namespace()
            .ok_or(RoleError::Unavailable)?
            .verify_identity(model.identity(), Some(&value))?;
        let mut semantic =
            PalsSemanticState::new_researcher_admitted(&profile, model, Some(value))?;
        semantic.select_review_context(review_policy, review_limits)?;
        Ok(Self {
            semantic,
            profile,
            resolver,
            post_repair_recheck,
            source,
        })
    }

    pub fn stores(&self) -> &PalsStores {
        &self.semantic.stores
    }
    pub fn last_authority(&self) -> Option<SearchAuthority> {
        self.semantic.last_adaptive_authority()
    }

    /// Move-only logical root registration. First C/TT/worker construction is
    /// still deferred. A failed registration returns this exact owner.
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    pub fn bind_root(
        mut self,
        position: Position,
        authority: SearchAuthority,
        original_deadline: Deadline,
        hard_deadline: Instant,
        cancellation: Arc<AtomicBool>,
    ) -> Result<ResearcherPreparingRoot<M>, (Self, PalsError)> {
        let prepared = (|| -> Result<_, PalsError> {
            if original_deadline.clock.0 != authority.epoch {
                return Err(PalsError::InvalidLimits);
            }
            allocation_controls(Some((hard_deadline, cancellation.as_ref())))?;
            if self.semantic.last_adaptive_authority().is_none()
                && self.semantic.stores.generation() == 0
                && self.semantic.stores.root().is_none()
            {
                // First logical game only; this is not an existing worker reset.
                self.semantic.game_generation = Some(authority.game.0);
            }
            let registration = self.semantic.register_adaptive_root(&position, authority)?;
            let root = match self
                .semantic
                .intern_hot(&self.profile.sequence_config(), position.clone())
            {
                Ok(root) => root,
                Err(error) => {
                    // No view/actor/worker exists yet. Keep the committed root
                    // history while revoking its unpublished registration.
                    self.semantic.abandon_unconsumed_adaptive_root(authority)?;
                    return Err(error);
                }
            };
            self.semantic
                .refresh_projection(self.semantic.nodes[root].state);
            Ok((root, registration))
        })();
        let (root, registration) = match prepared {
            Ok(value) => value,
            Err(error) => return Err((self, error)),
        };
        let Self {
            semantic,
            profile,
            resolver,
            post_repair_recheck,
            source,
        } = self;
        Ok(ResearcherPreparingRoot {
            semantic,
            profile,
            resolver,
            post_repair_recheck,
            root,
            authority,
            original_deadline,
            hard_deadline,
            cancellation,
            identity: Arc::new(()),
            source,
            position,
            registration: Some(registration),
            initial_view: None,
            original: None,
            retired_frontier: None,
        })
    }
}

impl<M: RoleModel> ResearcherPreparingRoot<M> {
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn stores(&self) -> &PalsStores {
        &self.semantic.stores
    }
    pub fn root_state_id(&self) -> StateId {
        self.semantic.nodes[self.root].state
    }
    pub fn source(&self) -> &DeferredNnuecRSource {
        &self.source
    }

    /// Mint once from the actual common selected-view producer. No fake role
    /// ticket, AdaptivePortQuestion, ContractPosition owner or peer is needed.
    pub(super) fn initial_review_source(
        &mut self,
    ) -> Result<PreparingRootReviewSource<'_>, PalsError> {
        self.validate_preparing_source()?;
        if self.initial_view.is_none() {
            use crate::pals::rovent_frontier::FrontierGoalStatus;
            if self.semantic.nodes[self.root].terminal.is_some()
                || self
                    .semantic
                    .rovent_frontier
                    .as_deref()
                    .is_some_and(|frontier| {
                        matches!(
                            frontier.snapshot().goal,
                            FrontierGoalStatus::DepthSatisfied | FrontierGoalStatus::MateSatisfied
                        )
                    })
            {
                return Err(StoreError::InvalidConditions(
                    "preparing root is already terminal or goal satisfied",
                )
                .into());
            }
            let original = self.original.as_ref().ok_or(PalsError::InvalidLimits)?;
            let clock = original.startup.clock;
            use crate::contract_time::ContractClock;
            self.semantic
                .rovent_frontier
                .as_deref_mut()
                .ok_or(PalsError::InvalidLimits)?
                .admit_role_input(
                    clock.domain(),
                    clock.now().map_err(|_| PalsError::InvalidLimits)?,
                )
                .map_err(|_| PalsError::InvalidLimits)?;
            if self.semantic.stores.review_pin_count()? != 1 {
                return Err(StoreError::InvalidConditions(
                    "preparing root has a foreign review lease",
                )
                .into());
            }
            let query = RoleQuestion {
                purpose: RoleQueryPurpose::ProposePolicy,
                prefix: &[],
                proposal: &[],
                refutation: None,
                divergences: &[],
            };
            let legal = self.position.legal_moves();
            self.initial_view = self.semantic.prepare_review_view(
                &self.profile.sequence_config(),
                self.root,
                query,
                &legal,
            )?;
        }
        self.prepared_review_source()
    }

    /// Completion validation reads exactly the old input. It cannot allocate a
    /// new selected view or relabel an old Bound as the current context.
    pub(super) fn prepared_review_source(
        &self,
    ) -> Result<PreparingRootReviewSource<'_>, PalsError> {
        self.validate_preparing_source()?;
        let view = self.initial_view.as_ref().ok_or(RoleError::Unavailable)?;
        if self.semantic.stores.review_pin_count()? != 2
            || !view.validates(&self.semantic.stores)
            || view.root_state() != self.root_state_id()
            || view.generation() != self.authority.root.0
            || !view.query().prefix.is_empty()
            || !view.query().proposal.is_empty()
            || view.query().counterexample.is_some()
        {
            return Err(RoleError::InvalidOutput.into());
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

    fn validate_preparing_source(&self) -> Result<(), PalsError> {
        allocation_controls(Some((self.hard_deadline, self.cancellation.as_ref())))?;
        if self.registration.is_none()
            || self.semantic.game_generation != Some(self.authority.game.0)
            || self.semantic.stores.generation() != self.authority.root.0
            || self.retired_frontier.is_some()
            || self.semantic.rovent_frontier.is_none()
            || !self.original.as_ref().is_some_and(|original| {
                original.root_rules_scanned && !original.deadline_before_frontier
            })
            || self.semantic.stores.tasks.active_states().next().is_some()
            || self.semantic.nodes.iter().any(|node| node.resume.is_some())
            || !self.semantic.nodes[self.root]
                .position
                .snapshot()
                .same_state(&self.position.snapshot())
        {
            return Err(StoreError::StaleConsumer.into());
        }
        self.semantic.ensure_archive_admission_open()?;
        Ok(())
    }
}

#[cfg(test)]
pub(in crate::pals::engine) mod tests;
