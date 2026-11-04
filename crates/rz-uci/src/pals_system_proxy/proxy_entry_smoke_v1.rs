//! Separate functional-smoke startup cap from a caller-supplied proxy-local
//! entry observation. This is not the exact OS child-spawn policy or proof of
//! execution, peer authorization, model readiness, or physical completion.

use super::{arena_grant_channel::RetainedOriginalArenaGrant, cli_v6::OriginalProxyWindowV6};
use crate::pals_adaptive_startup::{PalsSystemSourceObservationV6, PinnedPalsSystemStartupV6};
use rz_contracts::pals::deadline_grant_v1::{DeadlinePhase, DeadlineRole, DeadlineScope};
use serde::Serialize;
use std::{
    io,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

pub const PROXY_ENTRY_FUNCTIONAL_SMOKE_V1_DOMAIN: &str = "rz-pals-proxy-entry-functional-smoke/1";

/// Caller must supply the Instant captured at the real proxy entry. Constructing
/// this object alone does not establish when an OS process was spawned.
pub struct ProxyEntryFunctionalSmokeV1 {
    entry: Instant,
}

pub(super) struct OriginalRunCleanupGrants {
    pub run: RetainedOriginalArenaGrant,
    pub cleanup: RetainedOriginalArenaGrant,
}

/// Holds both exact original grant channels beside their bounded local window.
pub(super) struct ProxyEntrySmokeBundleV1 {
    pub inputs: OriginalRunCleanupGrants,
    pub window: OriginalProxyWindowV6,
    pub observations: [PalsSystemSourceObservationV6; 2],
}

/// Relative, process-local observations only; no Instant is serialized.
#[derive(Serialize)]
pub(super) struct ProxyEntrySmokeReceiptV1 {
    domain: &'static str,
    entry_observation: &'static str,
    os_spawn_to_entry: &'static str,
    evidence_scope: &'static str,
    owner_id: String,
    owner_epoch: u64,
    run_request_id: u64,
    cleanup_request_id: u64,
    source_scope: &'static str,
    run_source_observation: &'static str,
    cleanup_source_observation: &'static str,
    role: &'static str,
    game: u32,
    generation: u64,
    lock_sha256: String,
    entry_to_binding_ns: u64,
    local_startup_cap_ms: u64,
    startup_remaining_at_binding_ns: u64,
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub(super) fn source_observation(value: PalsSystemSourceObservationV6) -> &'static str {
    match value {
        PalsSystemSourceObservationV6::Verified => "verified",
        PalsSystemSourceObservationV6::Missing => "unknown",
        PalsSystemSourceObservationV6::Mismatch => "mismatch",
    }
}

fn same_scope(left: &DeadlineScope, right: &DeadlineScope) -> bool {
    left.owner_id == right.owner_id
        && left.owner_epoch == right.owner_epoch
        && left.role == right.role
        && left.game == right.game
        && left.generation == right.generation
        && left.lock_sha256 == right.lock_sha256
        && left.request_id != right.request_id
}

pub(super) fn local_startup_cap(
    entry: Instant,
    cap_ms: u64,
    run_until: Instant,
    cleanup_until: Instant,
    now: Instant,
) -> io::Result<Instant> {
    if entry > now || cap_ms == 0 {
        return Err(invalid("proxy entry or local startup cap invalid"));
    }
    let cap = entry
        .checked_add(Duration::from_millis(cap_ms))
        .ok_or_else(|| invalid("proxy startup cap overflow"))?;
    let until = cap.min(run_until).min(cleanup_until);
    if now >= until {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "proxy local startup cap expired",
        ));
    }
    Ok(until)
}

impl ProxyEntryFunctionalSmokeV1 {
    /// Intended to receive an Instant captured as the actual proxy main's
    /// first statement. This API itself cannot prove the capture location.
    pub fn from_local_entry(entry: Instant) -> Self {
        Self { entry }
    }

    pub(super) fn local_entry(&self) -> Instant {
        self.entry
    }

    /// Only the distinct functional-smoke route uses two original grants. The
    /// exact-spawn route keeps its existing startup grant and remains separate.
    /// Failure returns this entry and both original stream owners unchanged.
    pub(super) fn bind_original(
        self,
        inputs: OriginalRunCleanupGrants,
        source: &PinnedPalsSystemStartupV6,
        cancelled: &Arc<AtomicBool>,
        cleanup_cancelled: &Arc<AtomicBool>,
    ) -> Result<
        (ProxyEntrySmokeBundleV1, ProxyEntrySmokeReceiptV1),
        (io::Error, Self, OriginalRunCleanupGrants),
    > {
        let result = (|| {
            let run_scope = inputs.run.grant.original_scope();
            let cleanup_scope = inputs.cleanup.grant.original_scope();
            if run_scope.phase != DeadlinePhase::Run
                || cleanup_scope.phase != DeadlinePhase::Cleanup
                || !same_scope(run_scope, cleanup_scope)
                || inputs.run.peer != inputs.cleanup.peer
                || self.entry > inputs.run.grant.requested_at()
                || self.entry > inputs.cleanup.grant.requested_at()
            {
                return Err(invalid("proxy smoke original grant scope or entry differs"));
            }
            inputs.run.verify_original_peer()?;
            inputs.cleanup.verify_original_peer()?;
            let (run_deadline, run_observation) = inputs.run.grant.bind_source(source)?;
            let (cleanup_deadline, cleanup_observation) =
                inputs.cleanup.grant.bind_source(source)?;
            let run_until = run_deadline.min(cleanup_deadline);
            let cleanup_cap = run_until
                .checked_add(Duration::from_millis(
                    source.lock().config.window.cleanup_max_ms,
                ))
                .ok_or_else(|| invalid("proxy smoke cleanup cap overflow"))?;
            let cleanup_until = cleanup_deadline.min(cleanup_cap);
            let now = Instant::now();
            let startup_ms = source.lock().config.native_options.startup_ms;
            let startup_until =
                local_startup_cap(self.entry, startup_ms, run_until, cleanup_until, now)?;
            inputs.run.verify_original_peer()?;
            inputs.cleanup.verify_original_peer()?;
            let now = Instant::now();
            if now >= startup_until {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "proxy smoke startup deadline expired",
                ));
            }
            let elapsed = u64::try_from(now.duration_since(self.entry).as_nanos())
                .map_err(|_| invalid("proxy entry elapsed overflow"))?;
            let remaining = u64::try_from(startup_until.duration_since(now).as_nanos())
                .map_err(|_| invalid("proxy startup remaining overflow"))?;
            let role = match run_scope.role {
                DeadlineRole::Baseline => "baseline",
                DeadlineRole::Candidate => "candidate",
            };
            let lock_sha256 = run_scope
                .lock_sha256
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            let receipt = ProxyEntrySmokeReceiptV1 {
                domain: PROXY_ENTRY_FUNCTIONAL_SMOKE_V1_DOMAIN,
                entry_observation: "caller_supplied_proxy_local_instant",
                os_spawn_to_entry: "unknown",
                evidence_scope: "functional_smoke_only",
                owner_id: run_scope
                    .owner_id
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect(),
                owner_epoch: run_scope.owner_epoch,
                run_request_id: run_scope.request_id,
                cleanup_request_id: cleanup_scope.request_id,
                source_scope: "pinned_startup_deadline_wire_profile",
                run_source_observation: source_observation(run_observation),
                cleanup_source_observation: source_observation(cleanup_observation),
                role,
                game: run_scope.game,
                generation: run_scope.generation,
                lock_sha256,
                entry_to_binding_ns: elapsed,
                local_startup_cap_ms: startup_ms,
                startup_remaining_at_binding_ns: remaining,
            };
            let window = OriginalProxyWindowV6 {
                bound_reference: inputs
                    .run
                    .grant
                    .requested_at()
                    .min(inputs.cleanup.grant.requested_at()),
                startup_until,
                run_until,
                cleanup_until,
                cancelled: Arc::clone(cancelled),
                cleanup_cancelled: Arc::clone(cleanup_cancelled),
            };
            Ok((window, [run_observation, cleanup_observation], receipt))
        })();
        match result {
            Ok((window, observations, receipt)) => Ok((
                ProxyEntrySmokeBundleV1 {
                    inputs,
                    window,
                    observations,
                },
                receipt,
            )),
            Err(error) => Err((error, self, inputs)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_startup_never_extends_arena_run_or_cleanup() {
        let entry = Instant::now();
        let run = entry + Duration::from_secs(4);
        let cleanup = entry + Duration::from_secs(5);
        assert_eq!(
            local_startup_cap(entry, 10_000, run, cleanup, entry).unwrap(),
            run
        );
        assert_eq!(
            local_startup_cap(entry, 10_000, cleanup, run, entry).unwrap(),
            run
        );
        assert_eq!(
            local_startup_cap(entry, 1_000, run, cleanup, entry).unwrap(),
            entry + Duration::from_secs(1)
        );
        assert!(
            local_startup_cap(entry, 1_000, run, cleanup, entry + Duration::from_secs(1)).is_err()
        );
        assert!(local_startup_cap(entry, 0, run, cleanup, entry).is_err());
    }
    #[test]
    fn scope_identity_and_request_ids_remain_distinct() {
        let mut run = DeadlineScope {
            owner_id: [1; 16],
            owner_epoch: 2,
            role: DeadlineRole::Baseline,
            game: 1,
            generation: 3,
            request_id: 4,
            lock_sha256: [5; 32],
            phase: DeadlinePhase::Run,
        };
        let mut cleanup = run.clone();
        cleanup.phase = DeadlinePhase::Cleanup;
        cleanup.request_id = 5;
        assert!(same_scope(&run, &cleanup));
        cleanup.game += 1;
        assert!(!same_scope(&run, &cleanup));
        cleanup.game = run.game;
        cleanup.request_id = run.request_id;
        assert!(!same_scope(&run, &cleanup));
        run.owner_epoch += 1;
        assert!(!same_scope(&run, &cleanup));
    }
}
