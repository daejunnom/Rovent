//! UCI owner for the explicit synchronous Rovent compatibility engine.
//!
//! This does not turn PALS work receipts into Rovent receipts or route a native
//! selected-view provider through plain RoleModel calls. Native ticket support
//! and measured CPU/GPU admission remain separate from CPU/mock assembly.

use crate::EngineIdentity;
use crate::search_driver::{
    DriverPhysicalCompletion, DriverWork, SearchKind, SearchSessionContext, SearchSessionDriver,
    SearchSessionFailure, SearchSessionReport,
};
use rz_contracts::rovent::ExecutionMode;
use rz_contracts::{ContractError, Digest, ErrorCode, Stage, pals::SearchAuthority};
use rz_position::{BoardMove, Position};
use rz_search::pals::engine::{RoleError, RoleModel};
use rz_search::pals::rovent_engine::{
    RoventCompletion, RoventEngine, RoventError, RoventLimits, RoventResult, RoventShutdown,
};
use sha2::{Digest as _, Sha256};
use std::num::NonZeroU16;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct RoventSessionSettings {
    /// Actual role calls, never PUCT simulations or completed visits.
    pub maximum_role_calls: u64,
    /// Actual own R nodes; PC has no researcher and may register zero.
    pub maximum_native_nodes: u64,
}

/// One bounded latest observation. `result` belongs to this exact invocation,
/// even when root replacement makes it ineligible for UCI publication.
#[derive(Clone, Debug)]
pub struct RoventInvocationObservation {
    pub authority: SearchAuthority,
    pub result: RoventResult,
    pub uci_output_eligible: bool,
}

pub struct RoventSessionDriver<M: RoleModel + 'static> {
    engine: Mutex<Option<RoventEngine<M>>>,
    settings: RoventSessionSettings,
    implementation: Digest,
    mode: ExecutionMode,
    goal_capability: NonZeroU16,
    reset_pending: AtomicBool,
    closing: AtomicBool,
    closed: AtomicBool,
    quarantined: AtomicBool,
    latest: Mutex<Option<Arc<RoventInvocationObservation>>>,
    shutdown: Mutex<Option<RoventShutdown>>,
}

impl<M: RoleModel + 'static> RoventSessionDriver<M> {
    pub fn new(
        engine: RoventEngine<M>,
        settings: RoventSessionSettings,
    ) -> Result<Self, SearchSessionFailure> {
        if settings.maximum_role_calls == 0
            || settings.maximum_role_calls > 1_000_000
            || engine.mode().has_native_r() && settings.maximum_native_nodes == 0
            || engine.model_identity().len() > 256
        {
            return Err(reject(
                "RoventConfiguration",
                "invalid finite Rovent session settings",
            ));
        }
        if engine.requires_admitted_ticket_facade() {
            return Err(reject(
                "RoventNativeTicketFacadeUnsupported",
                "native selected-view providers require the actual admitted ticket facade",
            ));
        }
        let engine_identity = engine
            .implementation_identity()
            .map_err(|error| SearchSessionFailure::debug("RoventIdentity", &error))?;
        let mut digest = Sha256::new();
        digest.update(b"rovent-uci-session/0.1;nodes=rovent-position-visits/1\0");
        digest.update(engine_identity.0);
        digest.update(settings.maximum_role_calls.to_le_bytes());
        digest.update(settings.maximum_native_nodes.to_le_bytes());
        let implementation = Digest(digest.finalize().into());
        let mode = engine.mode();
        let goal_capability = engine.goal_capability();
        Ok(Self {
            engine: Mutex::new(Some(engine)),
            settings,
            implementation,
            mode,
            goal_capability,
            reset_pending: AtomicBool::new(false),
            closing: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            quarantined: AtomicBool::new(false),
            latest: Mutex::new(None),
            shutdown: Mutex::new(None),
        })
    }

    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    pub fn latest_result(&self) -> Result<Option<Arc<RoventInvocationObservation>>, ContractError> {
        self.latest
            .lock()
            .map(|latest| latest.clone())
            .map_err(|_| fault("Rovent result observation poisoned"))
    }

    pub fn shutdown_receipt(&self) -> Result<Option<RoventShutdown>, ContractError> {
        self.shutdown
            .lock()
            .map(|shutdown| shutdown.clone())
            .map_err(|_| fault("Rovent shutdown observation poisoned"))
    }

    fn unavailable(&self) -> Option<SearchSessionFailure> {
        if self.quarantined.load(Ordering::Acquire) {
            Some(SearchSessionFailure::physical_completion_unknown())
        } else if self.closing.load(Ordering::Acquire) || self.closed.load(Ordering::Acquire) {
            Some(reject(
                "RoventSessionClosed",
                "Rovent session admission is closed",
            ))
        } else {
            None
        }
    }

    fn map_error(&self, code: &'static str, error: &RoventError) -> SearchSessionFailure {
        if matches!(
            error,
            RoventError::Role(RoleError::PhysicalCompletionUnknown)
        ) {
            self.quarantined.store(true, Ordering::Release);
            SearchSessionFailure::physical_completion_unknown()
        } else {
            SearchSessionFailure::debug(code, error)
        }
    }

    pub fn shutdown(
        &self,
        deadline: Instant,
    ) -> Result<DriverPhysicalCompletion, SearchSessionFailure> {
        if self.quarantined.load(Ordering::Acquire) {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        if self.closed.load(Ordering::Acquire) {
            return Ok(DriverPhysicalCompletion::Confirmed);
        }
        self.closing.store(true, Ordering::Release);
        let mut owner = loop {
            if Instant::now() >= deadline {
                self.quarantined.store(true, Ordering::Release);
                return Err(SearchSessionFailure::physical_completion_unknown());
            }
            match self.engine.try_lock() {
                Ok(owner) => break owner,
                Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(1)),
                Err(TryLockError::Poisoned(_)) => {
                    self.quarantined.store(true, Ordering::Release);
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
            }
        };
        if self.quarantined.load(Ordering::Acquire) {
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        if self.closed.load(Ordering::Acquire) {
            return Ok(DriverPhysicalCompletion::Confirmed);
        }
        let engine = owner
            .as_mut()
            .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
        let actual = match engine.shutdown(deadline) {
            Ok(actual) => actual,
            Err(error) => {
                self.quarantined.store(true, Ordering::Release);
                let mut failure = SearchSessionFailure::debug("RoventShutdown", &error);
                failure.physical_completion = DriverPhysicalCompletion::Unknown;
                return Err(failure);
            }
        };
        let confirmed = actual.model_completion_known
            && actual.researcher.as_ref().is_none_or(|researcher| {
                researcher.cleanup_complete && !researcher.quarantined && !researcher.ownership_lost
            });
        *self.shutdown.lock().map_err(|_| {
            self.quarantined.store(true, Ordering::Release);
            SearchSessionFailure::physical_completion_unknown()
        })? = Some(actual);
        if !confirmed || Instant::now() >= deadline {
            self.quarantined.store(true, Ordering::Release);
            return Err(SearchSessionFailure::physical_completion_unknown());
        }
        self.closed.store(true, Ordering::Release);
        Ok(DriverPhysicalCompletion::Confirmed)
    }
}

impl<M: RoleModel + 'static> SearchSessionDriver for RoventSessionDriver<M> {
    fn node_budget_semantics(&self) -> crate::bridge::NodeBudgetSemantics {
        crate::bridge::NodeBudgetSemantics::RoventPositionVisits
    }
    fn search_goal_capability(&self) -> Option<NonZeroU16> {
        Some(self.goal_capability)
    }

    fn kind(&self) -> SearchKind {
        SearchKind::Rovent
    }

    fn implementation(&self) -> Digest {
        self.implementation
    }

    fn identity(&self) -> EngineIdentity {
        EngineIdentity {
            name: match self.mode {
                ExecutionMode::HybridPcr => "Rovent Hybrid P-C-R",
                ExecutionMode::HybridPc => "Rovent Hybrid P-C",
                ExecutionMode::CpuPcr => "Rovent CPU P-C-R",
                ExecutionMode::CpuPc => "Rovent CPU P-C",
            }
            .into(),
            author: "Rovent contributors".into(),
        }
    }

    fn reset_game(&self) -> Result<(), ContractError> {
        if self.quarantined.load(Ordering::Acquire)
            || self.closing.load(Ordering::Acquire)
            || self.closed.load(Ordering::Acquire)
        {
            return Err(fault(
                "Rovent owner cannot reset after closing or quarantine",
            ));
        }
        // The UCI manager records intent only. Model preflight and reset happen
        // on the next search worker under that invocation's original budget.
        self.reset_pending.store(true, Ordering::Release);
        *self
            .latest
            .lock()
            .map_err(|_| fault("Rovent result observation poisoned"))? = None;
        Ok(())
    }

    fn finish_session(
        &self,
        deadline: Instant,
    ) -> Result<Option<DriverPhysicalCompletion>, SearchSessionFailure> {
        self.shutdown(deadline).map(Some)
    }

    fn run(
        &self,
        position: &Position,
        context: &SearchSessionContext,
        progress: &mut dyn FnMut(BoardMove),
    ) -> Result<SearchSessionReport, SearchSessionFailure> {
        if let Some(failure) = self.unavailable() {
            return Err(failure);
        }
        if context.authority.implementation != self.implementation {
            return Err(reject(
                "RoventAuthority",
                "Rovent context belongs to a different implementation",
            ));
        }
        let mut owner = loop {
            if let Some(failure) = self.unavailable() {
                return Err(failure);
            }
            if !context
                .accepts()
                .map_err(|error| SearchSessionFailure::debug("RoventAuthority", &error))?
            {
                return Ok(SearchSessionReport {
                    mate_strategy: None,
                    best_move: None,
                    work: empty_work(),
                });
            }
            match self.engine.try_lock() {
                Ok(owner) => break owner,
                Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(1)),
                Err(TryLockError::Poisoned(_)) => {
                    self.quarantined.store(true, Ordering::Release);
                    return Err(SearchSessionFailure::physical_completion_unknown());
                }
            }
        };
        if let Some(failure) = self.unavailable() {
            return Err(failure);
        }
        // Scope can be revoked while the worker waits for the owner. Do not
        // consume a deferred reset or run provider preflight for that request.
        if !context
            .accepts()
            .map_err(|error| SearchSessionFailure::debug("RoventAuthority", &error))?
        {
            return Ok(SearchSessionReport {
                mate_strategy: None,
                best_move: None,
                work: empty_work(),
            });
        }
        let engine = owner
            .as_mut()
            .ok_or_else(SearchSessionFailure::physical_completion_unknown)?;
        let deadline = context
            .control
            .admission_deadline
            .min(context.control.deadline);
        if self.reset_pending.swap(false, Ordering::AcqRel) {
            if let Err(error) = engine.new_game(deadline, &context.control.cancellation) {
                self.reset_pending.store(true, Ordering::Release);
                return Err(self.map_error("RoventNewGame", &error));
            }
        }
        if !context
            .accepts()
            .map_err(|error| SearchSessionFailure::debug("RoventAuthority", &error))?
        {
            return Ok(SearchSessionReport {
                mate_strategy: None,
                best_move: None,
                work: empty_work(),
            });
        }
        let limits = RoventLimits {
            goal: context.goal,
            maximum_node_visits: context.requested_nodes,
            deadline,
            maximum_role_calls: self.settings.maximum_role_calls,
            maximum_native_nodes: self.settings.maximum_native_nodes,
        };
        let mut progress_invalid = false;
        let result = engine
            .search_with_progress(
                position,
                context.root_moves.as_deref(),
                limits,
                &context.control.cancellation,
                |snapshot| {
                    if context
                        .root_moves
                        .as_ref()
                        .is_some_and(|moves| !moves.contains(&snapshot.best_move))
                    {
                        progress_invalid = true;
                        context.control.cancel();
                    } else if context.accepts().unwrap_or(false)
                        && !self.closing.load(Ordering::Acquire)
                    {
                        progress(snapshot.best_move);
                    } else {
                        context.control.cancel();
                    }
                },
            )
            .map_err(|error| self.map_error("RoventSearch", &error))?;
        if progress_invalid
            || result.best_move.is_some_and(|movement| {
                !position.legal_moves().contains(&movement)
                    || context
                        .root_moves
                        .as_ref()
                        .is_some_and(|moves| !moves.contains(&movement))
            })
        {
            return Err(reject(
                "RoventRootRestriction",
                "Rovent progress or result escaped its original root domain",
            ));
        }
        let eligible = context
            .accepts()
            .map_err(|error| SearchSessionFailure::debug("RoventAuthority", &error))?
            && !self.closing.load(Ordering::Acquire)
            && !matches!(
                result.completion,
                RoventCompletion::Canceled | RoventCompletion::Deadline
            );
        let work = work(&result)?;
        let best_move = eligible.then_some(result.best_move).flatten();
        let mate_strategy = result
            .mate_strategy
            .clone()
            .filter(|strategy| best_move == Some(strategy.best_move()));
        *self
            .latest
            .lock()
            .map_err(|_| reject("RoventObservation", "Rovent result observation poisoned"))? =
            Some(Arc::new(RoventInvocationObservation {
                authority: context.authority,
                result,
                uci_output_eligible: eligible,
            }));
        Ok(SearchSessionReport {
            best_move,
            work,
            mate_strategy,
        })
    }
}

impl<M: RoleModel + 'static> Drop for RoventSessionDriver<M> {
    fn drop(&mut self) {
        if self.closed.load(Ordering::Acquire) && !self.quarantined.load(Ordering::Acquire) {
            return;
        }
        // Unclosed/unconfirmed owners are quarantine, not reusable cache. The
        // process-close hook must confirm physical lifetime before normal Drop.
        let owner = self
            .engine
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(retained) = owner.take() {
            let _quarantined: &'static mut RoventEngine<M> = Box::leak(Box::new(retained));
        }
    }
}

fn work(result: &RoventResult) -> Result<DriverWork, SearchSessionFailure> {
    let total_node_visits = result.work.total_node_visits().ok_or_else(|| {
        reject(
            "RoventWorkOverflow",
            "Rovent node observations overflow the registered node domain",
        )
    })?;
    Ok(DriverWork::Rovent {
        root_entries: result.work.root_entries,
        total_node_visits,
        role_calls: result.work.role_calls,
        accepted_policy: result.work.accepted_policy,
        accepted_fresh_values: result.work.accepted_fresh_values,
        checked_transitions: result.work.checked_transitions,
        native_r_calls: result.work.native_r_calls,
        native_nodes: result.work.native_nodes,
        native_qsearch_nodes: result.work.native_qsearch_nodes,
        native_pvs_entries: result.work.native_pvs_entries,
        native_see_nodes: result.work.native_see_nodes,
        external_teacher_calls: result.work.external_teacher_calls,
        completed_depth: result.horizon.completed_depth.map(|depth| depth.0),
        maximum_examined_ply: result.horizon.maximum_examined_ply.0,
    })
}

fn empty_work() -> DriverWork {
    DriverWork::Rovent {
        root_entries: 0,
        total_node_visits: 0,
        role_calls: 0,
        accepted_policy: 0,
        accepted_fresh_values: 0,
        checked_transitions: 0,
        native_r_calls: 0,
        native_nodes: 0,
        native_qsearch_nodes: 0,
        native_pvs_entries: Some(0),
        native_see_nodes: Some(0),
        external_teacher_calls: 0,
        completed_depth: None,
        maximum_examined_ply: 0,
    }
}

fn reject(code: &'static str, detail: &'static str) -> SearchSessionFailure {
    SearchSessionFailure {
        physical_completion: DriverPhysicalCompletion::Confirmed,
        code,
        detail: detail.into(),
    }
}

fn fault(detail: &'static str) -> ContractError {
    ContractError::new(ErrorCode::BackendFailure, Stage::Admission, detail)
}

#[cfg(test)]
mod tests;
