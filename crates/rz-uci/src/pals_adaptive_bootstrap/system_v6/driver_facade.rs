//! Guarded V6 UCI facade. This file is not an executable entry point.
//!
//! The same facade must be retained by the supervisor and handed to UCI.
//! No concrete driver handle leaves this owner. A receipt covers only the
//! producer scopes represented by the original driver, not a GPU-wide fence.

use crate::{
    EngineIdentity,
    bridge::NodeBudgetSemantics,
    pals_adaptive::{AdaptiveSessionProcessReceipt, PalsV6IdleFinish},
    pals_adaptive_bootstrap::ActualDriver,
    parser::GoLimits,
    search_driver::{
        DriverPhysicalCompletion, PalsCheckerRegistration, ProcessSearchWorkReceipt, SearchKind,
        SearchSessionContext, SearchSessionDriver, SearchSessionFailure, SearchSessionReport,
    },
};
use rz_contracts::{ContractError, Digest, pals::SearchAuthority};
use rz_position::{BoardMove, Position};
use rz_search::cpu_checker::{CheckerAttempt, CheckerShutdown, ExternalUciIdentity};
use std::{
    any::Any,
    num::NonZeroU16,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex, atomic::AtomicBool},
    time::Instant,
};

enum FinishState {
    Pending,
    Closed(AdaptiveSessionProcessReceipt),
    Incomplete(AdaptiveSessionProcessReceipt),
    Failed(SearchSessionFailure),
    Panicked(Box<dyn Any + Send>),
}

/// An immutable observation, never authority to destroy the original owner.
#[derive(Clone, Debug)]
pub(super) enum V6FacadeFinishObservation {
    Pending,
    Closed(AdaptiveSessionProcessReceipt),
    Incomplete(AdaptiveSessionProcessReceipt),
    Failed(SearchSessionFailure),
    Panicked,
    BusyOrPoisoned,
}

/// Retain this object and its process while physical completion is unknown.
/// Dropping the facade or its enclosing process is not a completion fence.
#[must_use]
pub(super) struct V6DriverFacade {
    driver: ActualDriver,
    finish: Mutex<FinishState>,
}

impl V6DriverFacade {
    /// Move in the actual driver: the facade never exposes a concrete handle.
    pub(super) fn new(driver: ActualDriver) -> Self {
        Self {
            driver,
            finish: Mutex::new(FinishState::Pending),
        }
    }

    pub(super) fn finish_observation(&self) -> V6FacadeFinishObservation {
        match self.finish.try_lock() {
            Ok(state) => match &*state {
                FinishState::Pending => V6FacadeFinishObservation::Pending,
                FinishState::Closed(receipt) => V6FacadeFinishObservation::Closed(receipt.clone()),
                FinishState::Incomplete(receipt) => {
                    V6FacadeFinishObservation::Incomplete(receipt.clone())
                }
                FinishState::Failed(error) => V6FacadeFinishObservation::Failed(error.clone()),
                FinishState::Panicked(_) => V6FacadeFinishObservation::Panicked,
            },
            Err(_) => V6FacadeFinishObservation::BusyOrPoisoned,
        }
    }

    /// Exactly one actual finish attempt. The caller must first establish that
    /// the UCI worker was joined, or that startup failed before any worker
    /// could exist. A service error by itself does not establish either fact.
    fn finish_once(
        &self,
        until: Instant,
    ) -> Result<AdaptiveSessionProcessReceipt, SearchSessionFailure> {
        let mut state = self
            .finish
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        match &*state {
            FinishState::Closed(receipt) => return Ok(receipt.clone()),
            FinishState::Failed(error) => return Err(error.clone()),
            FinishState::Incomplete(_) | FinishState::Panicked(_) => {
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
            FinishState::Pending => {}
        }
        // No attempt has begun if the original deadline is already expired.
        if Instant::now() >= until {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            self.driver.finish_pals_system_v6_idle_borrowed(until)
        }));
        match outcome {
            Ok(PalsV6IdleFinish::Complete(receipt)) if receipt.permits_process_close() => {
                *state = FinishState::Closed(receipt.clone());
                Ok(receipt)
            }
            Ok(PalsV6IdleFinish::Complete(receipt)) => {
                *state = FinishState::Incomplete(receipt);
                Err(SearchSessionFailure::physical_completion_unknown())
            }
            Ok(PalsV6IdleFinish::Unavailable(error)) => {
                *state = FinishState::Failed(error.clone());
                Err(error)
            }
            Ok(PalsV6IdleFinish::Panicked(payload)) | Err(payload) => {
                *state = FinishState::Panicked(payload);
                Err(SearchSessionFailure::physical_completion_unknown())
            }
        }
    }

    /// The lifecycle may use this only before it has started the UCI
    /// protocol or created any UCI worker. The same one-shot state is used.
    pub(super) fn finish_before_service(
        &self,
        until: Instant,
    ) -> Result<AdaptiveSessionProcessReceipt, SearchSessionFailure> {
        self.finish_once(until)
    }
}

impl SearchSessionDriver for V6DriverFacade {
    fn supports_ponder_clock(&self) -> bool {
        self.driver.supports_ponder_clock()
    }
    fn node_budget_semantics(&self) -> NodeBudgetSemantics {
        self.driver.node_budget_semantics()
    }
    fn validate_go(&self, position: &Position, go: &GoLimits) -> Result<(), ContractError> {
        self.driver.validate_go(position, go)
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        self.driver.search_goal_capability()
    }
    fn finish_session(
        &self,
        deadline: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        self.finish_once(deadline)
            .map(|_| Some(DriverPhysicalCompletion::Confirmed))
    }
    #[cfg(feature = "search-work-receipts")]
    fn followup_lifecycle(
        &self,
    ) -> Result<Option<crate::pals_attestation::followup::PalsFollowupLifecycleV4>, ContractError>
    {
        self.driver.followup_lifecycle()
    }
    #[cfg(feature = "search-work-receipts")]
    fn seal_followup_lifecycle(&self, protocol_retired: bool) -> Result<(), ContractError> {
        self.driver.seal_followup_lifecycle(protocol_retired)
    }
    fn kind(&self) -> SearchKind {
        self.driver.kind()
    }
    fn implementation(&self) -> Digest {
        self.driver.implementation()
    }
    fn identity(&self) -> EngineIdentity {
        self.driver.identity()
    }
    fn work_receipt(&self) -> Result<Option<ProcessSearchWorkReceipt>, ContractError> {
        self.driver.work_receipt()
    }
    fn pals_checker_registration(&self) -> Option<PalsCheckerRegistration> {
        self.driver.pals_checker_registration()
    }
    fn checker_started(&self) -> Option<bool> {
        self.driver.checker_started()
    }
    fn checker_startup_uci(&self) -> Result<Option<ExternalUciIdentity>, SearchSessionFailure> {
        self.driver.checker_startup_uci()
    }
    #[cfg(feature = "search-work-receipts")]
    fn checker_startup_resources(
        &self,
    ) -> Result<
        Option<crate::pals_attestation::checker::resource::ReadyBoundaryResources>,
        SearchSessionFailure,
    > {
        self.driver.checker_startup_resources()
    }
    #[cfg(feature = "search-work-receipts")]
    fn checker_startup_resource_unavailable(
        &self,
    ) -> Result<Option<&'static str>, SearchSessionFailure> {
        self.driver.checker_startup_resource_unavailable()
    }
    fn start_checker(
        &self,
        deadline: Instant,
        cancel: Arc<AtomicBool>,
    ) -> Result<Option<bool>, SearchSessionFailure> {
        let state = self
            .finish
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        if !matches!(&*state, FinishState::Pending) {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        // Retain the original panic while dropping only this facade's guard
        // normally; a separately poisoned driver remains unknown.
        let result = catch_unwind(AssertUnwindSafe(|| self.driver.start_checker(deadline, cancel)));
        drop(state);
        match result {
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
    fn checker_attempts(&self) -> Result<Option<Vec<CheckerAttempt>>, SearchSessionFailure> {
        self.driver.checker_attempts()
    }
    fn checker_last_attempt(&self) -> Result<Option<CheckerAttempt>, SearchSessionFailure> {
        self.driver.checker_last_attempt()
    }
    fn finish_checker(
        &self,
        _deadline: Instant,
    ) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        // The engine's `finish_session` call follows its own worker-drain
        // boundary. An arbitrary checker query is not that boundary and must
        // never start physical shutdown on its own.
        self.checker_shutdown()
    }
    fn checker_shutdown(&self) -> Result<Option<CheckerShutdown>, SearchSessionFailure> {
        let state = self
            .finish
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        match &*state {
            FinishState::Pending => self.driver.checker_shutdown(),
            FinishState::Closed(receipt) => Ok(receipt.external_shutdown()),
            FinishState::Failed(error) => Err(error.clone()),
            FinishState::Incomplete(_) | FinishState::Panicked(_) => {
                Err(SearchSessionFailure::physical_completion_unknown())
            }
        }
    }
    fn accept_uci_output(
        &self,
        authority: SearchAuthority,
        bestmove: &str,
        from_report: bool,
    ) -> Result<(), ContractError> {
        self.driver
            .accept_uci_output(authority, bestmove, from_report)
    }
    fn reset_game(&self) -> Result<(), ContractError> {
        self.driver.reset_game()
    }
    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        // Keep active driver execution and one-shot closure mutually exclusive.
        // This is not a substitute for the UCI owner's worker-join evidence:
        // an OS worker can exist before it reaches this method.
        let state = self
            .finish
            .try_lock()
            .map_err(|_| SearchSessionFailure::physical_completion_unknown())?;
        if !matches!(*state, FinishState::Pending) {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        let result = self.driver.run(position, context, progress);
        drop(state);
        result
    }
}
