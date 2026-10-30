//! Same-ticket ponder-hit clock custody. Submitted work keeps its original
//! request deadlines, visit budget, credit and physical cleanup boundary.
//!
//! Only the non-cloneable owner may hit or close this clock. Worker handles
//! observe value snapshots; they cannot renew a deadline or issue a Start.

use crate::SearchTicket;
use crate::bridge::{NodeBudgetSemantics, PreparedSearchBinding};
use crate::contracts::SessionScope;
use crate::engine::ProcessClock;
use rz_contracts::pals::SearchAuthority;
use rz_contracts::{ContractError, Deadline, ErrorCode, Stage};
use rz_runtime::CompletionSignal;
use rz_runtime::contracts::{ContractClock, ContractSystemClock};
use rz_search::time::TimeBudget;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A hit can release a held result even when no live work remains. It never
/// authorizes a replacement worker or revives an expired original hard wall.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HitDisposition {
    Live,
    NoLiveWork,
}

/// A validated hit is recorded once. The value remains unchanged on close.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoventPonderHitSnapshot {
    hit_at: Instant,
    budget: TimeBudget,
    disposition: HitDisposition,
}
impl RoventPonderHitSnapshot {
    pub fn hit_at(&self) -> Instant {
        self.hit_at
    }
    pub fn budget(&self) -> &TimeBudget {
        &self.budget
    }
    pub fn disposition(&self) -> HitDisposition {
        self.disposition
    }
}

#[derive(Clone, Copy, Debug)]
struct ClockState {
    hit: Option<RoventPonderHitSnapshot>,
    closed: bool,
    // Observation may only advance. A stale event timestamp must never reopen
    // an admission/result boundary already observed by another worker handle.
    observed_at: Instant,
}

#[derive(Debug)]
struct SharedClock {
    ticket: SearchTicket,
    authority: SearchAuthority,
    process_clock: ProcessClock,
    runtime_clock: ContractSystemClock,
    original: TimeBudget,
    original_hard: Deadline,
    cleanup: Instant,
    state: Mutex<ClockState>,
    signal: CompletionSignal,
}

/// This mutable authority is deliberately not Clone. Dropping it closes all
/// existing read handles and wakes a waiter without touching physical buffers.
#[derive(Debug)]
pub struct RoventPonderClockOwner {
    shared: Arc<SharedClock>,
}

/// Clones retain exactly the same clock owner rather than copying live state.
#[derive(Clone, Debug)]
pub struct RoventPonderClockHandle {
    shared: Arc<SharedClock>,
}

/// An immutable observation. Time predicates preserve TimeBudget's exclusive
/// boundaries and additionally reject closed or NoLiveWork states.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoventPonderClockSnapshot {
    budget: TimeBudget,
    hit: Option<RoventPonderHitSnapshot>,
    closed: bool,
    valid_observation: bool,
    observed_at: Instant,
}
impl RoventPonderClockSnapshot {
    pub fn budget(&self) -> &TimeBudget {
        &self.budget
    }
    pub fn hit(&self) -> Option<&RoventPonderHitSnapshot> {
        self.hit.as_ref()
    }
    pub fn has_hit(&self) -> bool {
        self.hit.is_some()
    }
    pub fn is_closed(&self) -> bool {
        self.closed
    }
    pub fn can_admit(&self, now: Instant) -> bool {
        self.live() && self.budget.can_admit(now.max(self.observed_at))
    }
    pub fn can_accept(&self, now: Instant) -> bool {
        self.live() && self.budget.can_accept(now.max(self.observed_at))
    }
    pub fn soft_expired(&self, now: Instant) -> bool {
        !self.live() || self.budget.soft_expired(now.max(self.observed_at))
    }
    pub fn hard_expired(&self, now: Instant) -> bool {
        !self.live() || self.budget.hard_expired(now.max(self.observed_at))
    }
    fn live(&self) -> bool {
        !self.closed
            && self.valid_observation
            && self
                .hit
                .is_none_or(|hit| hit.disposition == HitDisposition::Live)
    }
}

impl RoventPonderClockOwner {
    pub fn new(
        ticket: SearchTicket,
        authority: SearchAuthority,
        original: TimeBudget,
        original_cleanup: Instant,
        clock: &ProcessClock,
    ) -> Result<(Self, RoventPonderClockHandle), ContractError> {
        if authority.epoch != clock.epoch() {
            return Err(error(
                ErrorCode::IdentityMismatch,
                "ponder authority and original process clock differ",
            ));
        }
        validate_budget(clock, &original)?;
        clock.deadline(original_cleanup)?;
        if original_cleanup < original.output_deadline {
            return Err(error(
                ErrorCode::InvalidInput,
                "ponder cleanup precedes the original output deadline",
            ));
        }
        let shared = Arc::new(SharedClock {
            ticket,
            authority,
            process_clock: clock.clone(),
            runtime_clock: clock.runtime_clock(),
            original,
            original_hard: clock.deadline(original.hard_deadline)?,
            cleanup: original_cleanup,
            state: Mutex::new(ClockState {
                hit: None,
                closed: false,
                observed_at: original.start,
            }),
            signal: CompletionSignal::default(),
        });
        Ok((
            Self {
                shared: Arc::clone(&shared),
            },
            RoventPonderClockHandle { shared },
        ))
    }

    /// The caller prepares the original go's normal time allocation at hit_at.
    /// Only its value is consumed: the original binding/control is never edited.
    pub fn hit(
        &mut self,
        ticket: &SearchTicket,
        scope: SessionScope,
        hit_at: Instant,
        normal: &PreparedSearchBinding,
        now: Instant,
    ) -> Result<HitDisposition, ContractError> {
        let shared = &self.shared;
        if ticket != &shared.ticket || scope != SessionScope::Search(shared.authority) {
            return Err(error(
                ErrorCode::IdentityMismatch,
                "ponder hit does not own the original ticket and search scope",
            ));
        }
        let mut state = shared
            .state
            .lock()
            .map_err(|_| error(ErrorCode::BackendFailure, "ponder clock state is poisoned"))?;
        if state.closed {
            return Err(error(ErrorCode::Canceled, "ponder clock owner is closed"));
        }
        if state.hit.is_some() {
            return Err(error(ErrorCode::Stale, "ponder hit was already consumed"));
        }
        shared.process_clock.tick_at(hit_at)?;
        shared.process_clock.tick_at(now)?;
        if hit_at < shared.original.start || now < hit_at {
            return Err(error(
                ErrorCode::InvalidInput,
                "ponder hit precedes its Start or follows its observation",
            ));
        }
        let budget = normal.budget();
        if budget.start != hit_at
            || normal.node_budget_semantics() != NodeBudgetSemantics::RoventPositionVisits
            || normal.control().deadline != budget.hard_deadline
            || normal.control().soft_deadline != budget.soft_deadline
            || normal.control().admission_deadline != budget.admission_deadline
        {
            return Err(error(
                ErrorCode::IdentityMismatch,
                "ponder normal binding differs from its hit time or visit domain",
            ));
        }
        validate_budget(&shared.process_clock, budget)?;
        if normal.control().cancellation.load(Ordering::Acquire) {
            return Err(error(
                ErrorCode::Canceled,
                "ponder normal binding was canceled before hit acceptance",
            ));
        }
        let effective = capped_budget(shared.original, *budget)?;
        // Preparation may consume the remaining wall time. An earlier event
        // timestamp cannot reopen live work at the actual acceptance point.
        let now = now.max(Instant::now()).max(state.observed_at);
        shared.process_clock.tick_at(now)?;
        let disposition = if now >= shared.original.hard_deadline || now >= effective.hard_deadline
        {
            HitDisposition::NoLiveWork
        } else {
            HitDisposition::Live
        };
        state.hit = Some(RoventPonderHitSnapshot {
            hit_at,
            budget: effective,
            disposition,
        });
        state.observed_at = now;
        drop(state);
        shared.signal.notify();
        Ok(disposition)
    }

    /// Closing is sticky and does not reclaim a submitted job's resources.
    pub fn close(&self) {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !state.closed {
            state.closed = true;
            drop(state);
            self.shared.signal.notify();
        }
    }
}
impl Drop for RoventPonderClockOwner {
    fn drop(&mut self) {
        self.close();
    }
}

impl RoventPonderClockHandle {
    pub fn snapshot(&self, now: Instant) -> RoventPonderClockSnapshot {
        match self.shared.state.lock() {
            Ok(mut state) => {
                let valid_observation = self.shared.process_clock.tick_at(now).is_ok()
                    && now >= self.shared.original.start;
                if valid_observation {
                    state.observed_at = state.observed_at.max(now);
                }
                RoventPonderClockSnapshot {
                    budget: state.hit.map_or(self.shared.original, |hit| hit.budget),
                    hit: state.hit,
                    closed: state.closed,
                    valid_observation,
                    observed_at: state.observed_at,
                }
            }
            // A poisoned mutable authority cannot reopen worker admission.
            Err(poisoned) => {
                let state = poisoned.into_inner();
                RoventPonderClockSnapshot {
                    budget: state.hit.map_or(self.shared.original, |hit| hit.budget),
                    hit: state.hit,
                    closed: true,
                    valid_observation: false,
                    observed_at: state.observed_at,
                }
            }
        }
    }

    /// Match the actual registration, including origin; an equal epoch with a
    /// newly sampled Instant is a different clock and cannot consume this handle.
    pub fn matches_registration(
        &self,
        authority: SearchAuthority,
        clock: &ContractSystemClock,
        original_hard: Deadline,
        cleanup: Instant,
    ) -> bool {
        self.shared.authority == authority
            && self.shared.runtime_clock.domain() == clock.domain()
            && self.shared.runtime_clock.origin() == clock.origin()
            && self.shared.original_hard == original_hard
            && self.shared.cleanup == cleanup
    }
    pub fn shares_clock_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.shared, &other.shared)
    }
    /// Read version BEFORE taking the snapshot and checking work/deadlines.
    /// A change between that check and wait_changed cannot lose its wakeup.
    pub fn version(&self) -> u64 {
        self.shared.signal.version()
    }
    pub fn wait_changed(&self, observed: u64, timeout: Duration) -> bool {
        self.shared.signal.wait_changed(observed, timeout)
    }
    pub fn signal(&self) -> CompletionSignal {
        self.shared.signal.clone()
    }
}

fn validate_budget(clock: &ProcessClock, budget: &TimeBudget) -> Result<(), ContractError> {
    for instant in [
        budget.start,
        budget.soft_deadline,
        budget.admission_deadline,
        budget.hard_deadline,
        budget.output_deadline,
    ] {
        clock.deadline(instant)?;
    }
    if budget.start > budget.soft_deadline
        || budget.soft_deadline > budget.admission_deadline
        || budget.admission_deadline > budget.hard_deadline
        || budget.hard_deadline > budget.output_deadline
        || budget.start.checked_add(budget.allocation) != Some(budget.output_deadline)
    {
        return Err(error(
            ErrorCode::InvalidInput,
            "ponder time allocation boundaries are incoherent",
        ));
    }
    Ok(())
}

fn capped_budget(original: TimeBudget, normal: TimeBudget) -> Result<TimeBudget, ContractError> {
    let output_deadline = normal.output_deadline.min(original.output_deadline);
    // Keep the original Start as the value's accounting origin. A hit after a
    // closed boundary can then retain expired absolute walls without subtraction
    // underflow, and no elapsed pondering time disappears from the allocation.
    let allocation = output_deadline
        .checked_duration_since(original.start)
        .ok_or_else(|| error(ErrorCode::InvalidInput, "ponder output precedes its Start"))?;
    Ok(TimeBudget {
        start: original.start,
        soft_deadline: normal.soft_deadline.min(original.soft_deadline),
        admission_deadline: normal.admission_deadline.min(original.admission_deadline),
        hard_deadline: normal.hard_deadline.min(original.hard_deadline),
        output_deadline,
        allocation,
    })
}

fn error(code: ErrorCode, detail: &'static str) -> ContractError {
    ContractError::new(code, Stage::Admission, detail)
}

#[cfg(test)]
mod tests;
