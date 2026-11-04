//! The original finite window for a user-initiated V6 arena execution.
//!
//! A prepared snapshot or `NativeLaunchOwner` is not the start of this
//! window. The eventual execution entry point must create this owner before
//! preparation and retain it through the original supervisor's cleanup.
//! This type alone does not admit a process, authorize an IPC peer, or prove
//! native physical completion.

use super::LockedPalsSystemArenaPairV6;
use crate::ArenaError;
use rz_experiments::NativeEngineRole;
use std::time::{Duration, Instant};

/// One non-cloneable owner for one explicitly started V6 arena execution.
///
/// Its original start and finite deadlines are never reconstructed from a
/// snapshot, subprocess, serialized timestamp, or a later retry. A fresh
/// execution requires a fresh call to `begin` and its own execution record.
pub struct PalsV6ExecutionStart {
    pair_sha256: String,
    started: Instant,
    run_until: Instant,
    cleanup_until: Instant,
    startup_ms: [u64; 2],
    handshake_ms: [u64; 2],
}

impl PalsV6ExecutionStart {
    /// Call at the explicit execution boundary, before snapshot preparation.
    /// Validation and all later preparation consume the original wall window.
    pub fn begin(pair: &LockedPalsSystemArenaPairV6) -> Result<Self, ArenaError> {
        let started = Instant::now();
        pair.input().validate()?;
        let endpoints = &pair.input().endpoints;
        let window = &endpoints[0].startup_lock.config.window;
        let run_until = started
            .checked_add(Duration::from_millis(window.wall_time_max_ms))
            .ok_or_else(|| invalid("PALS V6 wall deadline overflow"))?;
        let cleanup_until = run_until
            .checked_add(Duration::from_millis(window.cleanup_max_ms))
            .ok_or_else(|| invalid("PALS V6 cleanup deadline overflow"))?;
        if Instant::now() >= run_until {
            return Err(invalid(
                "PALS V6 execution deadline expired during validation",
            ));
        }
        Ok(Self {
            pair_sha256: pair.sha256().to_owned(),
            started,
            run_until,
            cleanup_until,
            startup_ms: endpoints
                .each_ref()
                .map(|endpoint| endpoint.startup_lock.config.native_options.startup_ms),
            handshake_ms: endpoints
                .each_ref()
                .map(|endpoint| endpoint.handshake_max_ms),
        })
    }

    pub fn pair_sha256(&self) -> &str {
        &self.pair_sha256
    }

    pub fn started(&self) -> Instant {
        self.started
    }

    pub fn run_until(&self) -> Instant {
        self.run_until
    }

    pub fn cleanup_until(&self) -> Instant {
        self.cleanup_until
    }

    pub fn require_running(&self) -> Result<(), ArenaError> {
        if Instant::now() >= self.run_until {
            return Err(invalid("PALS V6 execution deadline expired"));
        }
        Ok(())
    }

    /// Child-specific budgets begin when the original supervisor actually
    /// starts that child. They can never extend the original arena window.
    /// The passed instant must be captured by that supervisor, not the child.
    pub fn child_startup_until(
        &self,
        role: NativeEngineRole,
        child_started: Instant,
    ) -> Result<Instant, ArenaError> {
        self.child_deadline(child_started, self.startup_ms[index(role)])
    }

    pub fn child_handshake_until(
        &self,
        role: NativeEngineRole,
        child_started: Instant,
    ) -> Result<Instant, ArenaError> {
        self.child_deadline(child_started, self.handshake_ms[index(role)])
    }

    fn child_deadline(
        &self,
        child_started: Instant,
        budget_ms: u64,
    ) -> Result<Instant, ArenaError> {
        let now = Instant::now();
        if child_started < self.started || child_started > now {
            return Err(invalid(
                "PALS V6 child start is outside the original execution",
            ));
        }
        let until = child_started
            .checked_add(Duration::from_millis(budget_ms))
            .ok_or_else(|| invalid("PALS V6 child deadline overflow"))?
            .min(self.run_until);
        if now >= until {
            return Err(invalid("PALS V6 child deadline expired"));
        }
        Ok(until)
    }
}

fn index(role: NativeEngineRole) -> usize {
    match role {
        NativeEngineRole::Baseline => 0,
        NativeEngineRole::Candidate => 1,
    }
}

fn invalid(message: &str) -> ArenaError {
    ArenaError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_budget_is_role_specific_and_never_extends_arena_window() {
        let started = Instant::now();
        let owner = PalsV6ExecutionStart {
            pair_sha256: String::new(),
            started,
            run_until: started + Duration::from_secs(600),
            cleanup_until: started + Duration::from_secs(610),
            startup_ms: [60_000, 900_000],
            handshake_ms: [120_000, 900_000],
        };
        assert_eq!(
            owner
                .child_startup_until(NativeEngineRole::Baseline, started)
                .unwrap(),
            started + Duration::from_secs(60)
        );
        assert_eq!(
            owner
                .child_startup_until(NativeEngineRole::Candidate, started)
                .unwrap(),
            owner.run_until
        );
        assert_eq!(
            owner
                .child_handshake_until(NativeEngineRole::Baseline, started)
                .unwrap(),
            started + Duration::from_secs(120)
        );
    }

    #[test]
    fn rejects_non_original_and_expired_child_start() {
        let now = Instant::now();
        let started = now.checked_sub(Duration::from_secs(5)).unwrap();
        let owner = PalsV6ExecutionStart {
            pair_sha256: String::new(),
            started,
            run_until: now + Duration::from_secs(10),
            cleanup_until: now + Duration::from_secs(11),
            startup_ms: [1_000; 2],
            handshake_ms: [1_000; 2],
        };
        assert!(
            owner
                .child_startup_until(NativeEngineRole::Baseline, started)
                .is_err()
        );
        assert!(
            owner
                .child_startup_until(NativeEngineRole::Baseline, started - Duration::from_secs(1))
                .is_err()
        );
        assert!(
            owner
                .child_startup_until(NativeEngineRole::Baseline, now + Duration::from_secs(1))
                .is_err()
        );
    }
}
