//! One actual Start's immutable report publication, independent of driver-wide
//! latest snapshots. This is not a physical-completion capability: the owning
//! engine must separately confirm its worker join/fence before publication.

use crate::SearchTicket;
use crate::contracts::SessionScope;
use crate::search_driver::SearchSessionReport;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::{ContractError, ErrorCode, Stage};
use rz_position::{BoardMove, Position, PositionSnapshot};
use rz_search::pals::rovent_frontier::FrontierMateStrategy;
use std::sync::{Arc, Mutex};

/// This envelope is constructed once from the actual Start ticket and root.
/// It never allocates a replacement Rules owner or reads a global latest root.
pub(super) struct DriverOutputEnvelope {
    ticket: SearchTicket,
    authority: SearchAuthority,
    root: PositionSnapshot,
    allowed: Arc<[BoardMove]>,
    report: Mutex<Option<Arc<SearchSessionReport>>>,
}

impl DriverOutputEnvelope {
    pub(super) fn scope(&self) -> SessionScope {
        SessionScope::Search(self.authority)
    }
    pub(super) fn new(
        ticket: SearchTicket,
        authority: SearchAuthority,
        position: &Position,
        root_moves: Option<&[BoardMove]>,
    ) -> Result<Self, ContractError> {
        let legal = position.ordered_legal_moves();
        if let Some(requested) = root_moves {
            if requested.is_empty()
                || requested.len() > legal.moves().len()
                || requested.iter().enumerate().any(|(at, movement)| {
                    !legal.moves().contains(movement) || requested[..at].contains(movement)
                })
            {
                return Err(admission(
                    "driver output has an invalid original root subset",
                ));
            }
        }
        let mut allowed = Vec::new();
        allowed
            .try_reserve_exact(legal.moves().len())
            .map_err(|_| {
                ContractError::new(
                    ErrorCode::ResourceExhausted,
                    Stage::Admission,
                    "driver output root-domain allocation failed",
                )
            })?;
        for movement in legal.moves().iter().copied() {
            if root_moves.is_none_or(|requested| requested.contains(&movement)) {
                allowed.push(movement);
            }
        }
        // Session has already sealed Rules order. Reject a substituted order;
        // do not silently turn another proof's move domain into this root's.
        if root_moves.is_some_and(|requested| requested != allowed.as_slice()) {
            return Err(admission(
                "driver output root subset is not in original Rules order",
            ));
        }
        Ok(Self {
            ticket,
            authority,
            root: legal.snapshot().clone(),
            allowed: allowed.into(),
            report: Mutex::new(None),
        })
    }

    /// Records the actual report once. Rejected input never replaces an older
    /// report or occupies an empty slot. Physical completion remains the engine
    /// owner's responsibility, not a caller-provided boolean on this object.
    pub(super) fn record_retaining(
        &self,
        report: SearchSessionReport,
    ) -> Result<Arc<SearchSessionReport>, (Arc<SearchSessionReport>, ContractError)> {
        let report = Arc::new(report);
        let mut slot = self
            .report
            .lock()
            .map_err(|_| (Arc::clone(&report), poisoned()))?;
        if slot.is_some() {
            return Err((report, stale("driver output report was already recorded")));
        }
        self.validate_report(&report)
            .map_err(|error| (Arc::clone(&report), error))?;
        *slot = Some(Arc::clone(&report));
        Ok(report)
    }

    /// Fixture convenience only. Production must retain a rejected report and
    /// its actual work through `record_retaining`.
    #[cfg(test)]
    pub(super) fn record(
        &self,
        report: SearchSessionReport,
    ) -> Result<Arc<SearchSessionReport>, ContractError> {
        self.record_retaining(report).map_err(|(_, error)| error)
    }

    pub(super) fn report_for(
        &self,
        ticket: &SearchTicket,
        scope: SessionScope,
    ) -> Result<Option<Arc<SearchSessionReport>>, ContractError> {
        self.validate_scope(scope)?;
        if *ticket != self.ticket {
            return Err(stale("driver output belongs to another Start ticket"));
        }
        self.report
            .lock()
            .map(|slot| slot.clone())
            .map_err(|_| poisoned())
    }

    /// Only the same recorded report may provide a mate for its own bestmove.
    /// A fallback output has no report-origin proof, even if the caller still
    /// holds an Arc to an older report. `from_report` grants no physical proof.
    pub(super) fn mate_for(
        &self,
        report: &Arc<SearchSessionReport>,
        scope: SessionScope,
        bestmove: &str,
        from_report: bool,
    ) -> Result<Option<FrontierMateStrategy>, ContractError> {
        if !from_report {
            return Ok(None);
        }
        self.validate_scope(scope)?;
        let slot = self.report.lock().map_err(|_| poisoned())?;
        if !slot.as_ref().is_some_and(|held| Arc::ptr_eq(held, report)) {
            return Err(stale(
                "driver output proof is not from this Start's recorded report",
            ));
        }
        self.validate_report(report)?;
        let Some(movement) = report.best_move else {
            return Ok(None);
        };
        if bestmove != movement.to_string() {
            return Err(output_mismatch(
                "driver output bestmove differs from the recorded report",
            ));
        }
        Ok(report.mate_strategy.clone())
    }

    fn validate_scope(&self, scope: SessionScope) -> Result<(), ContractError> {
        if scope != SessionScope::Search(self.authority) {
            return Err(stale("driver output search authority is no longer current"));
        }
        Ok(())
    }

    fn validate_report(&self, report: &SearchSessionReport) -> Result<(), ContractError> {
        if report
            .best_move
            .is_some_and(|movement| !self.allowed.contains(&movement))
        {
            return Err(ContractError::new(
                ErrorCode::InvalidInput,
                Stage::Output,
                "driver output bestmove escaped the original legal root domain",
            ));
        }
        if let Some(strategy) = &report.mate_strategy {
            if report.best_move != Some(strategy.best_move())
                || !strategy.validates_root(&self.root, &self.allowed)
            {
                return Err(output_mismatch(
                    "driver output mate strategy differs from its actual root, domain or bestmove",
                ));
            }
        }
        Ok(())
    }
}

fn admission(detail: &'static str) -> ContractError {
    ContractError::new(ErrorCode::InvalidInput, Stage::Admission, detail)
}
fn output_mismatch(detail: &'static str) -> ContractError {
    ContractError::new(ErrorCode::IdentityMismatch, Stage::Output, detail)
}
fn stale(detail: &'static str) -> ContractError {
    ContractError::new(ErrorCode::Stale, Stage::Output, detail)
}
fn poisoned() -> ContractError {
    ContractError::new(
        ErrorCode::BackendFailure,
        Stage::Output,
        "driver output report slot is poisoned",
    )
}

#[cfg(test)]
#[path = "driver_output/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "driver_output/ponder_tests.rs"]
mod ponder_tests;

#[cfg(all(test, feature = "pals-adaptive"))]
#[path = "driver_output/rovent_ponder_tests.rs"]
mod rovent_ponder_tests;
