//! Original-owner deadline grants for the separate PALS proxy.
//!
//! This module only computes bounded wire messages. The caller owns the IPC
//! listener, authenticates its peer and executable, and prevents replay. A
//! grant is neither process ownership nor evidence of physical completion.

use rz_contracts::pals::deadline_grant_v1::DeadlinePhase;
use rz_contracts::pals::deadline_grant_v1::{DeadlineGrant, DeadlineRequest, DeadlineScope};
use std::time::{Duration, Instant};

/// Kept by the original arena owner for the entire run, not reconstructed
/// from the wire or from a later proxy's startup time.
pub(crate) struct OwnerWindow {
    started: Instant,
    startup_until: Instant,
    run_until: Instant,
    cleanup_until: Instant,
}

impl OwnerWindow {
    pub(crate) fn new(
        started: Instant,
        startup_until: Instant,
        run_until: Instant,
        cleanup_until: Instant,
    ) -> Result<Self, &'static str> {
        if started > Instant::now()
            || startup_until <= started
            || run_until < startup_until
            || cleanup_until < run_until
        {
            return Err("deadline_window_invalid");
        }
        Ok(Self {
            started,
            startup_until,
            run_until,
            cleanup_until,
        })
    }

    /// Compare the same original monotonic bounds; this is not a process or
    /// endpoint capability and does not authorize an IPC peer.
    pub(crate) fn matches_bounds(
        &self,
        started: Instant,
        startup_until: Instant,
        run_until: Instant,
        cleanup_until: Instant,
    ) -> bool {
        self.started == started
            && self.startup_until == startup_until
            && self.run_until == run_until
            && self.cleanup_until == cleanup_until
    }

    pub(crate) fn phase_deadline(&self, phase: DeadlinePhase) -> Instant {
        match phase {
            DeadlinePhase::Startup => self.startup_until,
            DeadlinePhase::Run => self.run_until,
            DeadlinePhase::Cleanup => self.cleanup_until,
        }
    }

    /// `admitted_scope` must have been bound by the owner to the authenticated
    /// peer and its current game/generation. This method cannot establish that
    /// relationship; replay prevention is also the owner's responsibility.
    pub(crate) fn grant(
        &self,
        admitted_scope: &DeadlineScope,
        request_bytes: &[u8],
    ) -> Result<Vec<u8>, &'static str> {
        admitted_scope.validate()?;
        let request = DeadlineRequest::decode(request_bytes)?;
        if request.scope != *admitted_scope {
            return Err("deadline_request_scope_mismatch");
        }
        let now = Instant::now();
        if now < self.started {
            return Err("deadline_clock_invalid");
        }
        let until = match admitted_scope.phase {
            DeadlinePhase::Startup => self.startup_until,
            DeadlinePhase::Run => self.run_until,
            DeadlinePhase::Cleanup => self.cleanup_until,
        };
        let remaining = until
            .checked_duration_since(now)
            .ok_or("deadline_expired")?;
        let remaining_ns = u64::try_from(remaining.as_nanos()).map_err(|_| "deadline_overflow")?;
        if remaining_ns == 0 {
            return Err("deadline_expired");
        }
        Ok(DeadlineGrant {
            scope: admitted_scope.clone(),
            remaining_ns,
        }
        .encode()?
        .to_vec())
    }
}

/// The proxy creates this immediately before issuing the IPC request. The
/// returned Instant is conservative even when transport takes time, provided
/// the caller authenticates the same live owner and preserves message order.
pub(crate) struct PendingGrant {
    requested_at: Instant,
    scope: DeadlineScope,
}

impl PendingGrant {
    pub(crate) fn begin(scope: DeadlineScope) -> Result<(Self, Vec<u8>), &'static str> {
        scope.validate()?;
        let bytes = DeadlineRequest {
            scope: scope.clone(),
        }
        .encode()?
        .to_vec();
        Ok((
            Self {
                requested_at: Instant::now(),
                scope,
            },
            bytes,
        ))
    }

    pub(crate) fn accept(self, grant_bytes: &[u8]) -> Result<Instant, &'static str> {
        let grant = DeadlineGrant::decode(grant_bytes)?;
        if grant.scope != self.scope {
            return Err("deadline_grant_scope_mismatch");
        }
        let deadline = self
            .requested_at
            .checked_add(Duration::from_nanos(grant.remaining_ns))
            .ok_or("deadline_overflow")?;
        if Instant::now() >= deadline {
            return Err("deadline_expired");
        }
        Ok(deadline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::pals::deadline_grant_v1::{DeadlineRole, REQUEST_BYTES};

    fn scope() -> DeadlineScope {
        DeadlineScope {
            owner_id: [1; 16],
            owner_epoch: 1,
            role: DeadlineRole::Baseline,
            game: 1,
            generation: 1,
            request_id: 1,
            lock_sha256: [10; 32],
            phase: DeadlinePhase::Run,
        }
    }

    #[test]
    fn granted_deadline_never_extends_original() {
        let started = Instant::now();
        let run_until = started + Duration::from_secs(20);
        let owner = OwnerWindow::new(
            started,
            started + Duration::from_secs(5),
            run_until,
            run_until + Duration::from_secs(5),
        )
        .unwrap();
        let (pending, request) = PendingGrant::begin(scope()).unwrap();
        let response = owner.grant(&scope(), &request).unwrap();
        assert!(pending.accept(&response).unwrap() <= run_until);
    }

    #[test]
    fn grant_cannot_cross_role_generation_or_phase() {
        let started = Instant::now();
        let until = started + Duration::from_secs(20);
        let owner = OwnerWindow::new(started, until, until, until).unwrap();
        let (_, request) = PendingGrant::begin(scope()).unwrap();
        let mut other = scope();
        other.role = DeadlineRole::Candidate;
        assert_eq!(
            owner.grant(&other, &request),
            Err("deadline_request_scope_mismatch")
        );
        other = scope();
        other.generation += 1;
        assert_eq!(
            owner.grant(&other, &request),
            Err("deadline_request_scope_mismatch")
        );
        other = scope();
        other.phase = DeadlinePhase::Cleanup;
        assert_eq!(
            owner.grant(&other, &request),
            Err("deadline_request_scope_mismatch")
        );
    }

    #[test]
    fn expired_or_oversized_messages_are_rejected() {
        let now = Instant::now();
        let started = now.checked_sub(Duration::from_secs(2)).unwrap();
        let until = now.checked_sub(Duration::from_secs(1)).unwrap();
        let owner = OwnerWindow::new(started, until, until, until).unwrap();
        let (_, request) = PendingGrant::begin(scope()).unwrap();
        assert_eq!(owner.grant(&scope(), &request), Err("deadline_expired"));
        assert_eq!(
            owner.grant(&scope(), &vec![0; REQUEST_BYTES + 1]),
            Err("deadline_wire_length_invalid")
        );
    }
}
