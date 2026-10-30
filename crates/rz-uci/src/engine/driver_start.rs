//! The actual UCI Start's immutable input custody. This is deliberately not a
//! codec: metadata, a copied hash or a new clock cannot reconstruct this owner.

use super::{ProcessClock, SearchTicket, failure};
use crate::bridge::SearchBinding;
use crate::contracts::SessionScope;
use crate::parser::GoLimits;
use crate::search_driver::SearchSessionContext;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::{CancelToken, ContractError, ErrorCode};
use rz_position::{BoardMove, Position, PositionSnapshot};
use rz_runtime::contracts::{ContractClock, ContractSystemClock};
use rz_search::{driver::SearchControl, time::TimeBudget};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

#[cfg(test)]
pub(crate) mod fixture;

#[cfg(all(test, feature = "pals-adaptive"))]
mod collection_abort_tests;

struct CollectionAbortHook {
    game_gate: Arc<std::sync::atomic::AtomicU64>,
    root_targets_revoked: Arc<std::sync::atomic::AtomicBool>,
}
impl CollectionAbortHook {
    fn abort(&self) {
        self.game_gate.fetch_or(crate::search_driver::ROVENT_COLLECTION_ABORT_BIT, std::sync::atomic::Ordering::AcqRel);
        self.root_targets_revoked.store(true, std::sync::atomic::Ordering::Release);
    }
}
impl std::fmt::Debug for CollectionAbortHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CollectionAbortHook(original Start)")
    }
}

#[derive(Debug)]
pub(crate) struct DriverStartSource {
    teacher_scope_claimed: std::sync::atomic::AtomicBool,
    // 0 = unbound, 1 = bound, 2 = canceled, 3 = optional owner binding failed.
    collection_abort_state: std::sync::atomic::AtomicU8,
    collection_abort_hook: std::sync::OnceLock<CollectionAbortHook>,
    collection_owner_binding_error: std::sync::OnceLock<ContractError>,
    ticket: SearchTicket,
    authority: SearchAuthority,
    root: PositionSnapshot,
    original_go: Arc<GoLimits>,
    budget: TimeBudget,
    root_moves: Option<Arc<[BoardMove]>>,
    goal: rz_contracts::rovent::SearchGoal,
    requested_nodes: Option<u64>,
    control: SearchControl,
    cancellation: CancelToken,
    current: Arc<Mutex<SessionScope>>,
    runtime_clock: ContractSystemClock,
    cleanup: Instant,
    effective_clock: Option<crate::rovent_ponder::RoventPonderClockHandle>,
}
impl DriverStartSource {
    /// Only the engine's actual Effect::Start branch can call this constructor.
    pub(super) fn from_start(
        ticket: &SearchTicket,
        root: &Position,
        binding: &SearchBinding,
        context: &SearchSessionContext,
        clock: &ProcessClock,
    ) -> Result<Arc<Self>, ContractError> {
        let original_go = context
            .original_go
            .as_ref()
            .ok_or_else(|| invalid("actual Start original go is absent"))?;
        let actual = binding.control();
        if binding.ticket() != ticket
            || context.original_time_budget != Some(*binding.budget())
            || context.control.deadline != actual.deadline
            || context.control.soft_deadline != actual.soft_deadline
            || context.control.admission_deadline != actual.admission_deadline
            || context.control.max_simulations > actual.max_simulations
            || !Arc::ptr_eq(&context.control.cancellation, &actual.cancellation)
            || context.authority.epoch != clock.epoch()
        {
            return Err(invalid(
                "actual Start input differs from its original binding",
            ));
        }
        Ok(Arc::new(Self {
            teacher_scope_claimed: std::sync::atomic::AtomicBool::new(false),
            collection_abort_state: std::sync::atomic::AtomicU8::new(0),
            collection_abort_hook: std::sync::OnceLock::new(),
            collection_owner_binding_error: std::sync::OnceLock::new(),
            ticket: ticket.clone(),
            authority: context.authority,
            root: root.snapshot(),
            original_go: Arc::clone(original_go),
            budget: *binding.budget(),
            root_moves: context.root_moves.clone(),
            goal: context.goal,
            requested_nodes: context.requested_nodes,
            control: context.control.clone(),
            cancellation: context.cancellation.clone(),
            current: Arc::clone(&context.current),
            runtime_clock: clock.runtime_clock(),
            cleanup: context.shutdown_deadline,
            effective_clock: context.effective_clock.clone(),
        }))
    }

    /// Bind only to this actual Start. The second cancellation check closes
    /// the race between the owner's cancellation and a late optional install.
    #[cfg(feature = "pals-adaptive")]
    pub(crate) fn bind_collection_abort(
        &self,
        context: &SearchSessionContext,
        game_gate: Arc<std::sync::atomic::AtomicU64>,
        root_targets_revoked: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<(), ContractError> {
        if self.authority != context.authority
            || !Arc::ptr_eq(&self.control.cancellation, &context.control.cancellation)
            || !self.cancellation.shares_cancellation_with(&context.cancellation)
            || !Arc::ptr_eq(&self.current, &context.current)
        {
            return Err(invalid("collection abort hook differs from original Start"));
        }
        self.collection_abort_hook.set(CollectionAbortHook { game_gate, root_targets_revoked })
            .map_err(|_| invalid("collection abort hook already installed"))?;
        // One atomic word makes registration and actual owner cancellation
        // mutually ordered: if cancel wins, this installer performs the abort;
        // if install wins, the owner sees the already-initialized hook.
        let aborted = match self.collection_abort_state.compare_exchange(
            0, 1, std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Acquire,
        ) {
            Ok(_) => false,
            Err(2) => true,
            Err(3) => {
                self.collection_abort_hook.get().expect("hook just installed").abort();
                return Err(self.collection_owner_binding_error.get().copied()
                    .unwrap_or_else(|| invalid("original Start owner binding failure unavailable")));
            }
            Err(_) => return Err(invalid("collection abort registration state unknown")),
        };
        if aborted || self.control.cancellation.load(std::sync::atomic::Ordering::Acquire)
            || self.cancellation.is_canceled()
        {
            self.collection_abort_hook.get().expect("hook just installed").abort();
            return Err(self.collection_owner_binding_error.get().copied()
                .unwrap_or_else(|| invalid("original Start already canceled before Collection binding")));
        }
        Ok(())
    }

    /// A failure of the optional Collection association must not fail an
    /// otherwise valid search. A later Collection binding is still rejected.
    pub(crate) fn mark_collection_owner_binding_failed(&self, error: ContractError) {
        let _ = self.collection_owner_binding_error.set(error);
        self.collection_abort_state.store(3, std::sync::atomic::Ordering::Release);
    }

    /// Called by the actual UCI owner before its regular cancel. The hook is
    /// optional and must perform only nonblocking atomic state transitions.
    pub(crate) fn abort_collection(&self) {
        let previous = self.collection_abort_state.swap(2, std::sync::atomic::Ordering::AcqRel);
        if previous == 1 {
            if let Some(hook) = self.collection_abort_hook.get() { hook.abort(); }
        }
    }

    /// All root-admission materializations of this same actual UCI Start
    /// share one teacher scope claim; later Starts own different cells.
    pub(crate) fn claim_teacher_scope_once(&self) -> bool {
        self.teacher_scope_claimed
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_ok()
    }

    /// Validation does not require a live deadline: an original stopped Start
    /// still owns metadata, while the later gate alone may issue live work.
    pub(crate) fn verify(
        &self,
        root: &Position,
        context: &SearchSessionContext,
        clock: &ProcessClock,
    ) -> Result<(), ContractError> {
        let current_clock = clock.runtime_clock();
        if self.authority != context.authority
            || !self.root.same_state(&root.snapshot())
            || !context
                .original_go
                .as_ref()
                .is_some_and(|go| Arc::ptr_eq(go, &self.original_go))
            || context.original_time_budget != Some(self.budget)
            || context.goal != self.goal
            || context.requested_nodes != self.requested_nodes
            || !same_moves(&self.root_moves, &context.root_moves)
            || context.control.deadline != self.control.deadline
            || context.control.soft_deadline != self.control.soft_deadline
            || context.control.admission_deadline != self.control.admission_deadline
            || context.control.max_simulations != self.control.max_simulations
            || !Arc::ptr_eq(&self.control.cancellation, &context.control.cancellation)
            || !self
                .cancellation
                .shares_cancellation_with(&context.cancellation)
            || !Arc::ptr_eq(&self.current, &context.current)
            || self.cleanup != context.shutdown_deadline
            || !same_effective_clock(&self.effective_clock, &context.effective_clock)
            || current_clock.domain() != self.runtime_clock.domain()
            || current_clock.origin() != self.runtime_clock.origin()
        {
            return Err(invalid(
                "original Start source/control/Rules instance differs",
            ));
        }
        Ok(())
    }
    pub(crate) fn ticket(&self) -> &SearchTicket {
        &self.ticket
    }
    pub(super) fn original_go(&self) -> &Arc<GoLimits> {
        &self.original_go
    }
}
fn same_effective_clock(
    a: &Option<crate::rovent_ponder::RoventPonderClockHandle>,
    b: &Option<crate::rovent_ponder::RoventPonderClockHandle>,
) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.shares_clock_with(b),
        _ => false,
    }
}
fn same_moves(a: &Option<Arc<[BoardMove]>>, b: &Option<Arc<[BoardMove]>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}
fn invalid(detail: &'static str) -> ContractError {
    failure(ErrorCode::IdentityMismatch, detail)
}
