//! Conservative proxy-side conversion of an original owner's deadline grant.
//!
//! This module does not authenticate a peer or perform IPC. The caller must
//! retain the original connection and validate the live arena owner before
//! accepting a response; an ID carried by a message is not that proof.

use crate::pals_adaptive_startup::{PalsSystemSourceObservationV6, PinnedPalsSystemStartupV6};
use rz_contracts::pals::deadline_grant_v1::{
    DeadlineGrant, DeadlineRequest, DeadlineScope, REQUEST_BYTES,
};
use std::{
    io,
    time::{Duration, Instant},
};

/// Captured immediately before issuing a request to the original owner.
/// Neither this timestamp nor the derived deadline is the owner's start time.
pub(super) struct PendingDeadlineGrant {
    requested_at: Instant,
    scope: DeadlineScope,
    source_observation: PalsSystemSourceObservationV6,
}

impl PendingDeadlineGrant {
    pub(super) fn begin(
        source: &PinnedPalsSystemStartupV6,
        scope: DeadlineScope,
    ) -> io::Result<(Self, [u8; REQUEST_BYTES])> {
        // This applies only when the new deadline-wire profile is selected.
        // Observe mode records uncertainty; strict mode rejects a mismatch.
        let source_observation = source.observe_deadline_wire_profile()?;
        let request = DeadlineRequest {
            scope: scope.clone(),
        }
        .encode()
        .map_err(io::Error::other)?;
        Ok((
            Self {
                requested_at: Instant::now(),
                scope,
                source_observation,
            },
            request,
        ))
    }

    /// Given an authenticated, in-order response, adding its owner-measured
    /// remaining time to the pre-request timestamp cannot extend that owner's
    /// original monotonic deadline. An expired response authorizes nothing.
    pub(super) fn accept(
        self,
        bytes: &[u8],
    ) -> Result<
        (Instant, PalsSystemSourceObservationV6),
        (&'static str, PalsSystemSourceObservationV6),
    > {
        let observation = self.source_observation;
        let grant = DeadlineGrant::decode(bytes).map_err(|error| (error, observation))?;
        if grant.scope != self.scope {
            return Err(("deadline_grant_scope_mismatch", observation));
        }
        let deadline = self
            .requested_at
            .checked_add(Duration::from_nanos(grant.remaining_ns))
            .ok_or(("deadline_overflow", observation))?;
        if Instant::now() >= deadline {
            return Err(("deadline_expired", observation));
        }
        Ok((deadline, observation))
    }
}
