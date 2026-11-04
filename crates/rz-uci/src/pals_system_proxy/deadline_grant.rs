//! Conservative proxy-side conversion of an original owner's deadline grant.
//!
//! This module does not authenticate a peer or perform IPC. The caller must
//! retain the original connection and validate the live arena owner before
//! accepting a response; an ID carried by a message is not that proof.

use crate::pals_adaptive_startup::{PalsSystemSourceObservationV6, PinnedPalsSystemStartupV6};
use rz_contracts::pals::deadline_grant_v1::{
    DeadlineGrant, DeadlinePhase, DeadlineRequest, DeadlineScope, REQUEST_BYTES,
};
use std::{
    io,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

/// Captured immediately before issuing a request to the original owner.
/// Neither this timestamp nor the derived deadline is the owner's start time.
pub(super) struct PendingDeadlineGrant {
    requested_at: Instant,
    scope: DeadlineScope,
}

impl PendingDeadlineGrant {
    pub(super) fn begin(scope: DeadlineScope) -> io::Result<(Self, [u8; REQUEST_BYTES])> {
        let request = DeadlineRequest {
            scope: scope.clone(),
        }
        .encode()
        .map_err(io::Error::other)?;
        Ok((
            Self {
                requested_at: Instant::now(),
                scope,
            },
            request,
        ))
    }

    /// Local reference used only for conservative bound checks; this is not
    /// the remote arena owner's execution start or a source of timing proof.
    pub(super) fn bound_reference(&self) -> Instant {
        self.requested_at
    }

    /// Given an authenticated, in-order response, adding its owner-measured
    /// remaining time to the pre-request timestamp cannot extend that owner's
    /// original monotonic deadline. An expired response authorizes nothing.
    pub(super) fn accept(self, bytes: &[u8]) -> Result<ReceivedDeadlineGrant, &'static str> {
        let grant = DeadlineGrant::decode(bytes)?;
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
        Ok(ReceivedDeadlineGrant {
            deadline,
            requested_at: self.requested_at,
            scope: self.scope,
        })
    }
}

/// A converted response may bound source pinning but is not native admission.
/// The original peer must still be authenticated by the retained channel.
pub(super) struct ReceivedDeadlineGrant {
    deadline: Instant,
    requested_at: Instant,
    scope: DeadlineScope,
}

/// Assemble only grants for the same owner, game and startup lock. The caller
/// must still prove the retained IPC peer belongs to that original owner.
pub(super) fn assemble_window(
    source: &PinnedPalsSystemStartupV6,
    startup: ReceivedDeadlineGrant,
    run: ReceivedDeadlineGrant,
    cleanup: ReceivedDeadlineGrant,
    cancelled: Arc<AtomicBool>,
    cleanup_cancelled: Arc<AtomicBool>,
) -> io::Result<(
    super::cli_v6::OriginalProxyWindowV6,
    [PalsSystemSourceObservationV6; 3],
)> {
    let same_scope = |left: &DeadlineScope, right: &DeadlineScope| {
        left.owner_id == right.owner_id
            && left.owner_epoch == right.owner_epoch
            && left.role == right.role
            && left.game == right.game
            && left.generation == right.generation
            && left.lock_sha256 == right.lock_sha256
    };
    if startup.scope.phase != DeadlinePhase::Startup
        || run.scope.phase != DeadlinePhase::Run
        || cleanup.scope.phase != DeadlinePhase::Cleanup
        || !same_scope(&startup.scope, &run.scope)
        || !same_scope(&run.scope, &cleanup.scope)
        || startup.scope.request_id == run.scope.request_id
        || startup.scope.request_id == cleanup.scope.request_id
        || run.scope.request_id == cleanup.scope.request_id
    {
        return Err(io::Error::other(
            "deadline grants belong to different execution scopes",
        ));
    }
    let bound_reference = startup
        .requested_at
        .min(run.requested_at)
        .min(cleanup.requested_at);
    let (startup_until, first_observation) = startup.bind_source(source)?;
    let (run_until, second_observation) = run.bind_source(source)?;
    let (cleanup_until, third_observation) = cleanup.bind_source(source)?;
    // Independent round trips can conservatively shorten cleanup more than
    // run; shorten run too rather than rejecting an otherwise valid window.
    let run_until = run_until.min(cleanup_until);
    let declared_cleanup_end = run_until
        .checked_add(Duration::from_millis(
            source.lock().config.window.cleanup_max_ms,
        ))
        .ok_or_else(|| io::Error::other("deadline cleanup bound overflow"))?;
    let startup_until = startup_until.min(run_until);
    let cleanup_until = cleanup_until.min(declared_cleanup_end);
    if Instant::now() >= startup_until {
        return Err(io::Error::other(
            "deadline grant window expired or inconsistent",
        ));
    }
    Ok((
        super::cli_v6::OriginalProxyWindowV6 {
            bound_reference,
            startup_until,
            run_until,
            cleanup_until,
            cancelled,
            cleanup_cancelled,
        },
        [first_observation, second_observation, third_observation],
    ))
}

impl ReceivedDeadlineGrant {
    pub(super) fn original_scope(&self) -> &DeadlineScope {
        &self.scope
    }

    pub(super) fn requested_at(&self) -> Instant {
        self.requested_at
    }

    pub(super) fn pinning_deadline(&self) -> Instant {
        self.deadline
    }

    /// Observe/verify source only after the very same startup is pinned.
    /// This binds scope to that lock; it does not prove peer or GPU ownership.
    pub(super) fn bind_source(
        self,
        source: &PinnedPalsSystemStartupV6,
    ) -> io::Result<(Instant, PalsSystemSourceObservationV6)> {
        let lock = source.lock();
        let lock_hash: String = self
            .scope
            .lock_sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        if lock.canonical_sha256 != lock_hash
            || lock.config.process_generation != self.scope.generation
        {
            return Err(io::Error::other(
                "deadline grant differs from pinned startup scope",
            ));
        }
        if Instant::now() >= self.deadline {
            return Err(io::Error::other(
                "deadline grant expired before source binding",
            ));
        }
        let observation = source.observe_deadline_wire_profile()?;
        Ok((self.deadline, observation))
    }
}
