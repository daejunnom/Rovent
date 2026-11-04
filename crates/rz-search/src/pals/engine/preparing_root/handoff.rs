//! One-use same-Store handoff after the real preparation actor and R Ready.
use super::*;
use crate::cpu_worker::{CpuCreditLedger, NnuecReadyRegistration};
use crate::pals::engine::adaptive_port::{AdaptiveRulesOwnerIssuer, PreparingRootViewSeal};
use crate::pals::rovent_engine::RootMoveDomain;
use crate::pals::rovent_frontier::FrontierAttachment;
use crate::pals::store::StoreReadOwner;
use rz_position::contracts::ContractState;

/// Private constructor: neither raw metadata nor an owner number can mint it.
pub(in crate::pals::engine) struct PreparingRegistrationSeal<'a> {
    identity: &'a Arc<()>,
    store_owner: StoreReadOwner,
    root: usize,
    authority: SearchAuthority,
    original_deadline: Deadline,
    hard_deadline: Instant,
    position: &'a Position,
    view: &'a Arc<ReviewView>,
    cancellation: &'a Arc<AtomicBool>,
    source: &'a DeferredNnuecRSource,
    original: &'a controls::OriginalRootControls,
    review_seal: &'a PreparingRootViewSeal,
    ready: &'a NnuecReadyRegistration,
}
impl PreparingRegistrationSeal<'_> {
    /// The ordinary quiescence guard remains strict. This proof authorizes
    /// exactly the original Frontier lease and its original selected C lease.
    pub(in crate::pals::engine) fn validate<M: RoleModel>(
        &self,
        semantic: &PalsSemanticState<M>,
    ) -> Result<(), PalsError> {
        allocation_controls(Some((self.hard_deadline, self.cancellation)))?;
        let frontier = semantic
            .rovent_frontier
            .as_deref()
            .ok_or(StoreError::StaleConsumer)?;
        let source = PreparingRootReviewSource {
            identity: self.identity,
            stores: &semantic.stores,
            view: self.view,
            root: semantic.nodes[self.root].state,
            authority: self.authority,
            original_deadline: self.original_deadline,
            hard_deadline: self.hard_deadline,
            position: self.position,
            cancellation: self.cancellation,
            source: self.source,
        };
        if !self.store_owner.matches(&semantic.stores)
            || semantic.stores.review_pin_count()? != 2
            || semantic.stores.tasks.active_states().next().is_some()
            || semantic.nodes.iter().any(|node| node.resume.is_some())
            || !semantic.nodes[self.root]
                .position
                .snapshot()
                .same_state(&self.position.snapshot())
            || !self.ready.matches_source(self.source)
            || !self.ready.validates(
                &semantic.stores,
                source.root_state(),
                self.authority.root.0,
                self.view.seal_digest(),
            )
            || self.ready.profile().evaluation_namespace != self.ready.binding_digest()
        {
            return Err(StoreError::StaleConsumer.into());
        }
        self.review_seal
            .validate_root_source(&source)
            .map_err(|_| StoreError::InvalidEvidence("initial C original source changed"))?;
        frontier
            .validate_attachment(
                FrontierAttachment {
                    authority: self.authority,
                    mode: self.original.startup.mode,
                    original_deadline: self.original_deadline,
                    root_budget: self.original.startup.root_budget,
                    root_rules: &self.original.rules,
                    root_moves: &self.original.domain,
                    root_state: source.root_state(),
                    model_namespace: semantic
                        .model
                        .registered_representation_namespace()
                        .ok_or(RoleError::Unavailable)?,
                    native_ledger: Some(&self.original.ledger.identity()),
                },
                &semantic.stores,
            )
            .map_err(|_| StoreError::StaleConsumer)?;
        let observed = self
            .original
            .ledger
            .registration_observation()
            .map_err(|_| StoreError::InvalidConditions("original native ledger unavailable"))?;
        if observed.authority != self.authority
            || observed.active_jobs != 0
            || observed.outstanding_executions != 0
            || observed.reserved_native_nodes != 0
            || observed.quarantined_jobs != 0
            || observed.spent_native_nodes != 0
        {
            return Err(
                StoreError::InvalidConditions("researcher work preceded initial Ready").into(),
            );
        }
        Ok(())
    }
}

pub struct PreparingRootHandoff<M: RoleModel> {
    parts: PreparingRootHandoffParts<M>,
}
pub(in crate::pals::engine) struct PreparingRootHandoffParts<M: RoleModel> {
    pub(in crate::pals::engine) engine: PalsEngine<M>,
    pub(in crate::pals::engine) root: usize,
    pub(in crate::pals::engine) rules: Arc<ContractState>,
    pub(in crate::pals::engine) root_moves: Arc<RootMoveDomain>,
    pub(in crate::pals::engine) authority: SearchAuthority,
    pub(in crate::pals::engine) deadline: Deadline,
    pub(in crate::pals::engine) limits: PalsLimits,
    pub(in crate::pals::engine) started: Instant,
    pub(in crate::pals::engine) root_rules_scanned: bool,
    pub(in crate::pals::engine) rules_owner_issuer: AdaptiveRulesOwnerIssuer,
    pub(in crate::pals::engine) ledger: CpuCreditLedger,
    pub(in crate::pals::engine) root_budget: rz_contracts::Digest,
    pub(in crate::pals::engine) cleanup_deadline: Instant,
    pub(in crate::pals::engine) original_view: Arc<ReviewView>,
    pub(in crate::pals::engine) original_review_seal: PreparingRootViewSeal,
    pub(in crate::pals::engine) ready: NnuecReadyRegistration,
    pub(in crate::pals::engine) source: DeferredNnuecRSource,
    pub(in crate::pals::engine) cancellation: Arc<AtomicBool>,
}
impl<M: RoleModel> PreparingRootHandoff<M> {
    pub(in crate::pals::engine) fn into_parts(self) -> PreparingRootHandoffParts<M> {
        self.parts
    }
}

pub struct PreparingRootFinishRejection<M: RoleModel> {
    pub root: ResearcherPreparingRoot<M>,
    pub original_review_seal: PreparingRootViewSeal,
    pub ready: NnuecReadyRegistration,
    pub checker: Box<dyn CpuChecker>,
    pub error: PalsError,
}

impl<M: RoleModel> ResearcherPreparingRoot<M> {
    /// Reject before mutating logical registration or moving any of the
    /// caller's worker/source custody. A successful handoff resets no Store.
    #[allow(clippy::result_large_err)]
    pub fn finish_with_ready(
        mut self,
        original_review_seal: PreparingRootViewSeal,
        ready: NnuecReadyRegistration,
        checker: Box<dyn CpuChecker>,
    ) -> Result<PreparingRootHandoff<M>, PreparingRootFinishRejection<M>> {
        let admitted = (|| -> Result<_, PalsError> {
            let actual = self.prepared_review_source()?;
            original_review_seal
                .validate_root_source(&actual)
                .map_err(|_| StoreError::StaleConsumer)?;
            if !ready.matches_source(&self.source)
                || !ready.validates(
                    actual.stores(),
                    actual.root_state(),
                    self.authority.root.0,
                    actual.view().seal_digest(),
                )
                || checker.identity() != &CheckerIdentity::Owned(ready.value_identity().clone())
                || checker.registered_worker_stack_owner_id() != Some(ready.actual_stack_owner_id())
                || ready.profile().evaluation_namespace != ready.binding_digest()
            {
                return Err(StoreError::StaleConsumer.into());
            }
            let descriptor = checker
                .owned_descriptor()
                .ok_or(CheckerError::Invalid("initial actual R descriptor absent"))?;
            let plan = ready.cpu_plan().config();
            let mut conditions = Sha256::new();
            conditions.update(b"rz-cpu-search-conditions/1");
            conditions.update(checker.conditions().as_bytes());
            let condition_digest = rz_contracts::Digest(conditions.finalize().into());
            if descriptor.config.profile != plan.profile
                || descriptor.config.max_depth != plan.max_depth
                || descriptor.config.quiescence_ply != plan.quiescence_ply
                || descriptor.config.tt_entries != plan.tt_entries
                || ready.profile().max_depth != plan.max_depth
                || ready.profile().search_conditions != condition_digest
            {
                return Err(
                    StoreError::InvalidConditions("initial actual R plan/Ready differ").into(),
                );
            }
            let admission = PalsEngine::admit_researcher(
                &self.profile.config,
                &self.semantic.model,
                checker.as_ref(),
                self.resolver,
                self.post_repair_recheck,
            )?;
            if admission
                .model_registered_value
                .as_ref()
                .is_some_and(|expected| {
                    self.semantic.model_registered_value.as_ref() != Some(expected)
                })
            {
                return Err(RoleError::InvalidOutput.into());
            }
            Ok(admission)
        })();
        let admission = match admitted {
            Ok(admission) => admission,
            Err(error) => {
                return Err(PreparingRootFinishRejection {
                    root: self,
                    original_review_seal,
                    ready,
                    checker,
                    error,
                });
            }
        };
        let registration = self
            .registration
            .take()
            .expect("prepared source requires original root registration");
        let proof = PreparingRegistrationSeal {
            identity: &self.identity,
            store_owner: self.semantic.stores.read_owner(),
            root: self.root,
            authority: self.authority,
            original_deadline: self.original_deadline,
            hard_deadline: self.hard_deadline,
            position: &self.position,
            view: self
                .initial_view
                .as_ref()
                .expect("prepared source requires original view"),
            cancellation: &self.cancellation,
            source: &self.source,
            original: self
                .original
                .as_ref()
                .expect("prepared source requires original controls"),
            review_seal: &original_review_seal,
            ready: &ready,
        };
        if let Err((registration, error)) =
            self.semantic.consume_prepared_adaptive_root_registration(
                registration,
                &self.position,
                self.authority,
                &proof,
            )
        {
            self.registration = Some(registration);
            return Err(PreparingRootFinishRejection {
                root: self,
                original_review_seal,
                ready,
                checker,
                error,
            });
        }
        let original = self
            .original
            .take()
            .expect("original controls checked before registration commit");
        let view = self
            .initial_view
            .take()
            .expect("actual initial C view checked");
        let mut engine = PalsEngine::from_admitted_researcher(
            self.profile.config,
            checker,
            self.resolver,
            self.post_repair_recheck,
            Some(self.profile.maximum_root_prefix_plies),
            admission,
            self.semantic,
        );
        engine.deferred_researcher_source = Some(self.source.clone());
        engine.deferred_researcher_initial =
            Some(DeferredResearcherInitialReadback::from_ready(&ready));
        Ok(PreparingRootHandoff {
            parts: PreparingRootHandoffParts {
                engine,
                root: self.root,
                rules: original.rules,
                root_moves: original.domain,
                authority: self.authority,
                deadline: self.original_deadline,
                limits: original.startup.limits,
                started: original.startup.clock.origin(),
                root_rules_scanned: original.root_rules_scanned,
                rules_owner_issuer: original.rules_owner_issuer,
                ledger: original.ledger,
                root_budget: original.startup.root_budget,
                cleanup_deadline: original.startup.cleanup_deadline,
                original_view: view,
                original_review_seal,
                ready,
                source: self.source,
                cancellation: self.cancellation,
            },
        })
    }
}
