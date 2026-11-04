//! Original root controls exist before C preparation or a native R owner.
use super::*;
use crate::contract_time::{ContractClock, InstantClock};
use crate::cpu_worker::{CpuCreditLedger, CpuWorkerError};
use crate::pals::engine::adaptive_port::{AdaptiveFrontierReceipt, AdaptiveRulesOwnerIssuer};
use crate::pals::rovent_engine::RootMoveDomain;
use crate::pals::rovent_frontier::{
    FrontierAttachment, FrontierError, FrontierGoalStatus, FrontierLimits, FrontierRegistration,
    FrontierStop, RoventFrontier,
};
use rz_contracts::rovent::{ExecutionMode, SearchGoal};
use rz_position::contracts::{ContractPosition, ContractState};

pub struct ResearcherRootStartup {
    pub mode: ExecutionMode,
    pub root_budget: rz_contracts::Digest,
    pub goal: SearchGoal,
    pub nodes: Option<u64>,
    pub selective_policy: rz_contracts::Digest,
    pub frontier_limits: FrontierLimits,
    pub clock: InstantClock,
    pub limits: PalsLimits,
    pub cleanup_deadline: Instant,
}

#[derive(Debug)]
pub enum ResearcherPreparingError {
    Semantic(PalsError),
    Frontier(FrontierError),
    Ledger(CpuWorkerError),
}
impl std::fmt::Display for ResearcherPreparingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "researcher root preparation: {self:?}")
    }
}
impl std::error::Error for ResearcherPreparingError {}
impl From<PalsError> for ResearcherPreparingError {
    fn from(error: PalsError) -> Self {
        Self::Semantic(error)
    }
}
impl From<FrontierError> for ResearcherPreparingError {
    fn from(error: FrontierError) -> Self {
        Self::Frontier(error)
    }
}
impl From<CpuWorkerError> for ResearcherPreparingError {
    fn from(error: CpuWorkerError) -> Self {
        Self::Ledger(error)
    }
}

pub(super) struct OriginalRootControls {
    pub(super) startup: ResearcherRootStartup,
    pub(super) rules: Arc<ContractState>,
    pub(super) domain: Arc<RootMoveDomain>,
    pub(super) rules_owner_issuer: AdaptiveRulesOwnerIssuer,
    pub(super) ledger: CpuCreditLedger,
    pub(super) root_rules_scanned: bool,
    pub(super) deadline_before_frontier: bool,
}

/// No Ready/worker is represented here. The receipt distinguishes an actual
/// attached controller from expiry before that controller could be created.
pub struct RetiredPreparingRoot<M: RoleModel> {
    pub owner: ResearcherPreparingOwner<M>,
    pub ledger: CpuCreditLedger,
    pub root_budget: rz_contracts::Digest,
    pub frontier: Option<AdaptiveFrontierReceipt>,
    pub deadline_before_frontier: bool,
    pub root_rules: Arc<ContractState>,
    pub root_moves: Arc<RootMoveDomain>,
    pub cleanup_deadline: Instant,
}

impl<M: RoleModel> ResearcherPreparingRoot<M> {
    /// Create the original ledger/Rules/domain only once. After those owners
    /// exist, any rejected Frontier admission retains them in this same root.
    pub fn attach_original_controls(
        &mut self,
        startup: ResearcherRootStartup,
        rules_owner_issuer: AdaptiveRulesOwnerIssuer,
        searchmoves: Option<&[BoardMove]>,
    ) -> Result<(), ResearcherPreparingError> {
        if self.original.is_some()
            || self.initial_view.is_some()
            || self.semantic.rovent_frontier.is_some()
            || !startup.mode.has_native_r()
            || startup.clock.domain() != self.original_deadline.clock
            || startup
                .clock
                .instant_at(self.original_deadline.at)
                .map_err(FrontierError::from)?
                != self.hard_deadline
            || startup.limits.deadline != self.hard_deadline
            || startup.limits.max_cpu_nodes == 0
            || startup.limits.max_rounds == 0
            || startup.limits.cpu_depth == 0
            || startup.limits.cpu_depth > self.source.cpu_plan().config().max_depth
            || startup.frontier_limits.maximum_goal_plies.get()
                > self.profile.maximum_root_prefix_plies
            || startup.cleanup_deadline < self.hard_deadline
        {
            return Err(FrontierError::Invalid("original preparing root controls").into());
        }
        startup
            .goal
            .validate(startup.frontier_limits.maximum_goal_plies)
            .map_err(FrontierError::from)?;
        let namespace = self
            .semantic
            .model
            .registered_representation_namespace()
            .ok_or(RoleError::Unavailable)
            .map_err(PalsError::from)?;
        namespace
            .verify_identity(
                self.semantic.model.identity(),
                self.semantic.model.value_identity(),
            )
            .map_err(PalsError::from)?;
        let rules = Arc::new(
            ContractPosition::new(
                rules_owner_issuer().map_err(|_| PalsError::from(RoleError::InvalidOutput))?,
                self.position.clone(),
            )
            .export()
            .map_err(|_| PalsError::from(RoleError::InvalidOutput))?,
        );
        let domain = Arc::new(
            RootMoveDomain::new(&rules, searchmoves)
                .map_err(|_| PalsError::from(RoleError::InvalidOutput))?,
        );
        let ledger =
            CpuCreditLedger::new(self.authority, startup.limits.max_cpu_nodes, startup.clock)?;
        self.original = Some(OriginalRootControls {
            startup,
            rules,
            domain,
            rules_owner_issuer,
            ledger,
            root_rules_scanned: false,
            deadline_before_frontier: false,
        });
        let original = self
            .original
            .as_mut()
            .expect("original owners just installed");
        let now = original.startup.clock.now().map_err(FrontierError::from)?;
        if Instant::now() >= self.hard_deadline
            || self
                .original_deadline
                .accepts(original.startup.clock.domain(), now)
                .is_err()
        {
            original.deadline_before_frontier = true;
            return Ok(());
        }
        let namespace = Arc::new(
            self.semantic
                .model
                .registered_representation_namespace()
                .ok_or(RoleError::Unavailable)
                .map_err(PalsError::from)?
                .clone(),
        );
        let registration = FrontierRegistration {
            authority: self.authority,
            mode: original.startup.mode,
            original_deadline: self.original_deadline,
            root_budget: original.startup.root_budget,
            goal: original.startup.goal,
            nodes: original.startup.nodes,
            selective_policy: original.startup.selective_policy,
            root_rules: Arc::clone(&original.rules),
            root_moves: Arc::clone(&original.domain),
            root_state: self.semantic.nodes[self.root].state,
            model_namespace: namespace,
            native_ledger: Some(original.ledger.identity()),
        };
        let frontier = RoventFrontier::new(
            registration,
            original.startup.frontier_limits.clone(),
            &self.semantic.stores,
            original.startup.clock.domain(),
            now,
        )?;
        frontier.validate_attachment(
            FrontierAttachment {
                authority: self.authority,
                mode: original.startup.mode,
                original_deadline: self.original_deadline,
                root_budget: original.startup.root_budget,
                root_rules: &original.rules,
                root_moves: &original.domain,
                root_state: self.semantic.nodes[self.root].state,
                model_namespace: self
                    .semantic
                    .model
                    .registered_representation_namespace()
                    .ok_or(RoleError::Unavailable)
                    .map_err(PalsError::from)?,
                native_ledger: Some(&original.ledger.identity()),
            },
            &self.semantic.stores,
        )?;
        self.semantic.rovent_frontier = Some(Box::new(frontier));
        Ok(())
    }

    pub fn original_frontier_receipt(&self) -> Option<AdaptiveFrontierReceipt> {
        self.semantic
            .rovent_frontier
            .as_deref()
            .map(|frontier| AdaptiveFrontierReceipt {
                root_budget: frontier.registration().root_budget,
                snapshot: frontier.snapshot(),
            })
            .or_else(|| {
                self.retired_frontier
                    .as_ref()
                    .map(|receipt| AdaptiveFrontierReceipt {
                        root_budget: receipt.root_budget,
                        snapshot: receipt.snapshot.clone(),
                    })
            })
    }
    pub fn deadline_before_frontier(&self) -> bool {
        self.original
            .as_ref()
            .is_some_and(|original| original.deadline_before_frontier)
    }
    pub fn original_ledger(&self) -> Option<&CpuCreditLedger> {
        self.original.as_ref().map(|original| &original.ledger)
    }
    pub fn original_root_rules(&self) -> Option<&Arc<ContractState>> {
        self.original.as_ref().map(|original| &original.rules)
    }
    pub fn original_root_moves(&self) -> Option<&Arc<RootMoveDomain>> {
        self.original.as_ref().map(|original| &original.domain)
    }

    /// Revoke the original request authority. This retains every view/Bound
    /// and says nothing about actor or worker physical completion.
    pub fn cancel(&mut self) {
        self.cancellation.store(true, Ordering::Release);
        if let Some(frontier) = &mut self.semantic.rovent_frontier {
            frontier.cancel();
        }
    }

    /// Original paid inspection, before any C view, TT or native worker exists.
    pub fn prepare_original_root_rules(&mut self) -> Result<(), ResearcherPreparingError> {
        if self.initial_view.is_some() {
            return Err(PalsError::InvalidLimits.into());
        }
        if self.cancellation.load(Ordering::Acquire) {
            self.cancel();
            return Err(FrontierError::Revoked.into());
        }
        let original = self.original.as_ref().ok_or(PalsError::InvalidLimits)?;
        if original.deadline_before_frontier {
            return Ok(());
        }
        if original.root_rules_scanned {
            return Ok(());
        }
        let clock = original.startup.clock;
        let moves = original.domain.allowed().to_vec();
        self.semantic
            .rovent_frontier
            .as_deref_mut()
            .ok_or(FrontierError::ForeignRoot)?
            .admit_role_input(clock.domain(), clock.now().map_err(FrontierError::from)?)?;
        for movement in moves {
            if self.cancellation.load(Ordering::Acquire) {
                self.cancel();
                return Err(FrontierError::Revoked.into());
            }
            let frontier = self
                .semantic
                .rovent_frontier
                .as_deref_mut()
                .ok_or(FrontierError::ForeignRoot)?;
            let path = frontier.observe_root_child(
                movement,
                clock.domain(),
                clock.now().map_err(FrontierError::from)?,
            )?;
            let checked = frontier.checked_line(path, &self.semantic.stores)?;
            let endpoint = Position::from_snapshot(checked.endpoint_snapshot(), Default::default())
                .map_err(PalsError::from)?;
            if matches!(
                endpoint
                    .classify_position()
                    .map_err(PalsError::from)?
                    .play_status,
                PlayStatus::Terminal { .. }
            ) {
                self.semantic.adopt_checked_frontier_line(
                    &self.profile.sequence_config(),
                    self.root,
                    &checked,
                )?;
            }
        }
        self.original.as_mut().unwrap().root_rules_scanned = true;
        Ok(())
    }

    /// Only a pre-materialization root can return to its exact original idle
    /// owner. Once a view/request might exist the session retains cleanup duty.
    #[allow(clippy::result_large_err)]
    pub fn retire_before_materialization(
        mut self,
    ) -> Result<RetiredPreparingRoot<M>, (Self, ResearcherPreparingError)> {
        let checked = (|| -> Result<_, ResearcherPreparingError> {
            let original = self.original.as_ref().ok_or(PalsError::InvalidLimits)?;
            if self.initial_view.is_some()
                || self.registration.is_none()
                || self
                    .semantic
                    .stores
                    .review_pin_count()
                    .map_err(PalsError::from)?
                    != usize::from(self.semantic.rovent_frontier.is_some())
                || self.semantic.stores.tasks.active_states().next().is_some()
                || self.semantic.nodes.iter().any(|node| node.resume.is_some())
            {
                return Err(PalsError::InvalidLimits.into());
            }
            let ledger = original.ledger.registration_observation()?;
            if ledger.active_jobs != 0
                || ledger.outstanding_executions != 0
                || ledger.reserved_native_nodes != 0
                || ledger.quarantined_jobs != 0
            {
                return Err(FrontierError::NativeUnknown.into());
            }
            let receipt = self.original_frontier_receipt();
            if !original.deadline_before_frontier
                && !receipt.as_ref().is_some_and(|receipt| {
                    matches!(
                        receipt.snapshot.goal,
                        FrontierGoalStatus::Stopped(
                            FrontierStop::WorkBudget
                                | FrontierStop::Deadline
                                | FrontierStop::Canceled
                        ) | FrontierGoalStatus::DepthSatisfied
                            | FrontierGoalStatus::MateSatisfied
                    )
                })
            {
                return Err(PalsError::InvalidLimits.into());
            }
            Ok(receipt)
        })();
        let receipt = match checked {
            Ok(receipt) => receipt,
            Err(error) => return Err((self, error)),
        };
        // All actual work is known absent and the only remaining lease is
        // this controller's own lease. Preserve its terminal receipt before
        // releasing that lease for the ordinary nonce-retirement guard.
        self.retired_frontier = receipt.as_ref().map(|receipt| AdaptiveFrontierReceipt {
            root_budget: receipt.root_budget,
            snapshot: receipt.snapshot.clone(),
        });
        drop(self.semantic.rovent_frontier.take());
        if let Err(error) = self
            .semantic
            .abandon_unconsumed_adaptive_root(self.authority)
        {
            return Err((self, error.into()));
        }
        self.registration.take();
        let original = self
            .original
            .take()
            .expect("retirement checked its original owners");
        Ok(RetiredPreparingRoot {
            owner: ResearcherPreparingOwner {
                semantic: self.semantic,
                profile: self.profile,
                resolver: self.resolver,
                post_repair_recheck: self.post_repair_recheck,
                source: self.source,
            },
            ledger: original.ledger,
            root_budget: original.startup.root_budget,
            frontier: receipt,
            deadline_before_frontier: original.deadline_before_frontier,
            root_rules: original.rules,
            root_moves: original.domain,
            cleanup_deadline: original.startup.cleanup_deadline,
        })
    }
}
