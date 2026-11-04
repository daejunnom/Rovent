//! Shared production state transitions for a V3 target publication gate.
//!
//! These transitions own no admission, ticket, row or teacher authority.
//! The parent capability keeps those originals and passes the same root
//! gate and exact ticket authority into this state machine.
use super::{TargetBeginRejectionV3, TargetRevocationV3};
use crate::rovent_collect::producer_v3::SpeculativeTicketAuthorityV3;
use crate::rovent_collect::{ArenaError, invalid};
use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicBool, AtomicU8, Ordering},
};

const AVAILABLE: u8 = 0;
const IN_PROGRESS: u8 = 1;
const COMMITTED: u8 = 2;
const UNKNOWN: u8 = 3;

pub(super) struct TargetGateV3 {
    gate: Arc<Mutex<()>>,
    revoked: TargetRevocationV3,
    authority: SpeculativeTicketAuthorityV3,
    state: AtomicU8,
}

impl TargetGateV3 {
    pub(super) fn new(
        gate: Arc<Mutex<()>>,
        revoked: TargetRevocationV3,
        authority: SpeculativeTicketAuthorityV3,
    ) -> Self {
        Self {
            gate,
            revoked,
            authority,
            state: AtomicU8::new(AVAILABLE),
        }
    }

    pub(super) fn matches_revocation(&self, other: &TargetRevocationV3) -> bool {
        self.revoked.same_owner(other)
    }

    pub(super) fn try_begin(
        &self,
    ) -> Result<Option<TargetAppendGuardV3<'_>>, TargetBeginRejectionV3> {
        let gate = match self.gate.try_lock() {
            Ok(gate) => gate,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(TargetBeginRejectionV3::Poisoned);
            }
        };
        match self.state.load(Ordering::Acquire) {
            AVAILABLE => {}
            COMMITTED => return Err(TargetBeginRejectionV3::AlreadyCommitted),
            _ => return Err(TargetBeginRejectionV3::Uncertain),
        }
        if self.revoked.is_revoked() || self.authority.canceled() {
            return Err(TargetBeginRejectionV3::Revoked);
        }
        self.state.store(IN_PROGRESS, Ordering::Release);
        Ok(Some(TargetAppendGuardV3 {
            target: self,
            _gate: gate,
            resolved: false,
        }))
    }
}

/// Revoke and cancel use the exact same gate that guards the append. A Busy
/// return leaves the original authority and root flags unchanged.
pub(super) fn try_cancel_ticket(
    gate: &Mutex<()>,
    authority: &SpeculativeTicketAuthorityV3,
    after_cancel: impl FnOnce(),
) -> Result<Option<()>, ArenaError> {
    let _gate = match gate.try_lock() {
        Ok(gate) => gate,
        Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
        Err(std::sync::TryLockError::Poisoned(_)) => {
            return Err(invalid("lock-v3 target gate poisoned"));
        }
    };
    authority.cancel();
    after_cancel();
    Ok(Some(()))
}

pub(super) fn try_revoke_root(
    gate: &Mutex<()>,
    revoked: &TargetRevocationV3,
    cancel_all: &AtomicBool,
) -> Result<Option<()>, ArenaError> {
    let _gate = match gate.try_lock() {
        Ok(gate) => gate,
        Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
        Err(std::sync::TryLockError::Poisoned(_)) => {
            return Err(invalid("lock-v3 target revocation gate poisoned"));
        }
    };
    revoked.revoke();
    cancel_all.store(true, Ordering::Release);
    Ok(Some(()))
}

/// Dropping an unresolved guard quarantines the ticket as unknown. It is not
/// evidence that a writer or physical worker completed.
pub(crate) struct TargetAppendGuardV3<'a> {
    target: &'a TargetGateV3,
    _gate: MutexGuard<'a, ()>,
    resolved: bool,
}
impl TargetAppendGuardV3<'_> {
    pub(crate) fn committed(mut self) {
        self.target.state.store(COMMITTED, Ordering::Release);
        self.resolved = true;
    }
    pub(crate) fn known_not_written(mut self) {
        self.target.state.store(AVAILABLE, Ordering::Release);
        self.resolved = true;
    }
}
impl Drop for TargetAppendGuardV3<'_> {
    fn drop(&mut self) {
        if !self.resolved {
            self.target.state.store(UNKNOWN, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> (
        TargetGateV3,
        Arc<Mutex<()>>,
        TargetRevocationV3,
        SpeculativeTicketAuthorityV3,
    ) {
        let lock = Arc::new(Mutex::new(()));
        let revoked = TargetRevocationV3::new();
        let authority = SpeculativeTicketAuthorityV3::new();
        let target = TargetGateV3::new(Arc::clone(&lock), revoked.clone(), authority.clone());
        (target, lock, revoked, authority)
    }

    #[test]
    fn cancellation_before_append_rejects_and_other_owner_is_distinct() {
        let (target, lock, revoked, authority) = gate();
        let other = TargetRevocationV3::new();
        assert!(target.matches_revocation(&revoked));
        assert!(!target.matches_revocation(&other));
        assert!(matches!(
            try_cancel_ticket(&lock, &authority, || {}),
            Ok(Some(()))
        ));
        assert!(matches!(
            target.try_begin(),
            Err(TargetBeginRejectionV3::Revoked)
        ));
    }

    #[test]
    fn append_holds_gate_until_commit_and_cancel_can_retry() {
        let (target, lock, _revoked, authority) = gate();
        let guard = target.try_begin().unwrap().expect("available target");
        assert!(matches!(
            try_cancel_ticket(&lock, &authority, || {}),
            Ok(None)
        ));
        assert!(!authority.canceled());
        guard.committed();
        assert!(matches!(
            try_cancel_ticket(&lock, &authority, || {}),
            Ok(Some(()))
        ));
        assert!(matches!(
            target.try_begin(),
            Err(TargetBeginRejectionV3::AlreadyCommitted)
        ));
    }

    #[test]
    fn busy_and_known_not_written_are_retryable() {
        let (target, lock, _revoked, _authority) = gate();
        let held = lock.lock().unwrap();
        assert!(matches!(target.try_begin(), Ok(None)));
        drop(held);
        target
            .try_begin()
            .unwrap()
            .expect("retry after Busy")
            .known_not_written();
        target
            .try_begin()
            .unwrap()
            .expect("known unwritten retry")
            .committed();
        assert!(matches!(
            target.try_begin(),
            Err(TargetBeginRejectionV3::AlreadyCommitted)
        ));
    }

    #[test]
    fn root_revoke_waits_for_same_gate_and_then_blocks_append() {
        let (target, lock, revoked, _authority) = gate();
        let canceled = AtomicBool::new(false);
        let held = lock.lock().unwrap();
        assert!(matches!(
            try_revoke_root(&lock, &revoked, &canceled),
            Ok(None)
        ));
        assert!(!revoked.is_revoked());
        assert!(!canceled.load(Ordering::Acquire));
        drop(held);
        assert!(matches!(
            try_revoke_root(&lock, &revoked, &canceled),
            Ok(Some(()))
        ));
        assert!(canceled.load(Ordering::Acquire));
        assert!(matches!(
            target.try_begin(),
            Err(TargetBeginRejectionV3::Revoked)
        ));
    }

    #[test]
    fn panic_with_live_append_guard_remains_unknown() {
        let (target, _lock, _revoked, _authority) = gate();
        // The real writer catches its append panic while the gate guard is
        // still held, then drops the unresolved guard after the catch.
        let guard = target.try_begin().unwrap().expect("available target");
        let outcome = std::panic::catch_unwind(|| panic!("test interrupted append"));
        assert!(outcome.is_err());
        drop(guard);
        assert!(matches!(
            target.try_begin(),
            Err(TargetBeginRejectionV3::Uncertain)
        ));
    }

    #[test]
    fn poisoned_gate_is_distinct_from_busy() {
        let (target, lock, _revoked, authority) = gate();
        let other = Arc::clone(&lock);
        let thread = std::thread::spawn(move || {
            let _held = other.lock().unwrap();
            panic!("test poisoned gate");
        });
        assert!(thread.join().is_err());
        assert!(matches!(
            target.try_begin(),
            Err(TargetBeginRejectionV3::Poisoned)
        ));
        assert!(try_cancel_ticket(&lock, &authority, || {}).is_err());
        assert!(!authority.canceled());
    }
}
