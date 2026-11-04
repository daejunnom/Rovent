//! Separate NoGame functional-smoke window over the original direct-parent
//! Run and Cleanup grants. This is not game, Elo, performance or exact-spawn
//! evidence and does not establish role listener provenance or GPU completion.

use super::{
    cli_v6::OriginalProxyWindowV6,
    proxy_entry_smoke_channel_v1::OriginalDirectSmokeGrant,
    proxy_entry_smoke_v1::{ProxyEntryFunctionalSmokeV1, local_startup_cap, source_observation},
};
use crate::pals_adaptive_startup::PinnedPalsSystemStartupV6;
use rz_contracts::pals::{
    deadline_grant_v1::DeadlineRole,
    proxy_entry_smoke_grant_v1::{
        PROXY_ENTRY_SMOKE_POLICY_DOMAIN, ProxyEntrySmokePhase, ProxyEntrySmokeScope,
    },
};
use serde::Serialize;
use std::{
    io,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Exact received Run/Cleanup grants and their original direct-parent streams.
pub(super) struct OriginalNoGameSmokeGrants {
    pub run: OriginalDirectSmokeGrant,
    pub cleanup: OriginalDirectSmokeGrant,
}

pub(super) struct ProxyEntryNoGameSmokeBundleV1 {
    pub inputs: OriginalNoGameSmokeGrants,
    pub window: OriginalProxyWindowV6,
}

/// Relative process-local observations. This receipt contains no game number,
/// absolute Instant or claim about OS spawn-to-entry or physical completion.
#[derive(Serialize)]
pub(super) struct ProxyEntryNoGameSmokeReceiptV1 {
    policy_domain: &'static str,
    evidence_scope: &'static str,
    entry_observation: &'static str,
    os_spawn_to_entry: &'static str,
    owner_id: String,
    owner_epoch: u64,
    run_request_id: u64,
    cleanup_request_id: u64,
    role: &'static str,
    generation: u64,
    lock_sha256: String,
    source_scope: &'static str,
    source_observation: &'static str,
    entry_to_binding_ns: u64,
    local_startup_cap_ms: u64,
    startup_remaining_at_binding_ns: u64,
}

fn same_no_game_scope(run: &ProxyEntrySmokeScope, cleanup: &ProxyEntrySmokeScope) -> bool {
    run.phase == ProxyEntrySmokePhase::Run
        && cleanup.phase == ProxyEntrySmokePhase::Cleanup
        && run.policy == cleanup.policy
        && run.owner_id == cleanup.owner_id
        && run.owner_epoch == cleanup.owner_epoch
        && run.role == cleanup.role
        && run.generation == cleanup.generation
        && run.lock_sha256 == cleanup.lock_sha256
        && run.request_id != cleanup.request_id
}

impl ProxyEntryFunctionalSmokeV1 {
    /// This strictly separate NoGame path holds original direct-parent streams
    /// on success and returns them with this exact entry owner on all failures.
    /// It does not authorize a role listener or create a native session.
    pub(super) fn bind_no_game_original(
        self,
        inputs: OriginalNoGameSmokeGrants,
        source: &PinnedPalsSystemStartupV6,
        cancelled: &Arc<AtomicBool>,
        cleanup_cancelled: &Arc<AtomicBool>,
    ) -> Result<
        (
            ProxyEntryNoGameSmokeBundleV1,
            ProxyEntryNoGameSmokeReceiptV1,
        ),
        (io::Error, Self, OriginalNoGameSmokeGrants),
    > {
        let result = (|| {
            let entry = self.local_entry();
            let run_scope = inputs.run.scope();
            let cleanup_scope = inputs.cleanup.scope();
            if !same_no_game_scope(run_scope, cleanup_scope)
                || !inputs.run.same_original_peer_and_challenge(&inputs.cleanup)
                || entry > inputs.run.requested_at()
                || entry > inputs.cleanup.requested_at()
            {
                return Err(invalid("NoGame smoke grant scope, peer or entry differs"));
            }
            let lock_sha256: String = run_scope
                .lock_sha256
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            if lock_sha256 != source.lock().canonical_sha256
                || run_scope.generation != source.lock().config.process_generation
            {
                return Err(invalid(
                    "NoGame smoke grant differs from original startup lock",
                ));
            }
            inputs.run.verify_original_peer()?;
            inputs.cleanup.verify_original_peer()?;
            let run_until = inputs.run.deadline().min(inputs.cleanup.deadline());
            let cleanup_cap = run_until
                .checked_add(Duration::from_millis(
                    source.lock().config.window.cleanup_max_ms,
                ))
                .ok_or_else(|| invalid("NoGame smoke cleanup cap overflow"))?;
            let cleanup_until = inputs.cleanup.deadline().min(cleanup_cap);
            let cap_ms = source.lock().config.native_options.startup_ms;
            let startup_until =
                local_startup_cap(entry, cap_ms, run_until, cleanup_until, Instant::now())?;
            inputs.run.verify_original_peer()?;
            inputs.cleanup.verify_original_peer()?;
            let now = Instant::now();
            if now >= startup_until {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "NoGame smoke startup deadline expired",
                ));
            }
            let elapsed = u64::try_from(now.duration_since(entry).as_nanos())
                .map_err(|_| invalid("NoGame smoke elapsed overflow"))?;
            let remaining = u64::try_from(startup_until.duration_since(now).as_nanos())
                .map_err(|_| invalid("NoGame smoke startup remaining overflow"))?;
            let role = match run_scope.role {
                DeadlineRole::Baseline => "baseline",
                DeadlineRole::Candidate => "candidate",
            };
            let receipt = ProxyEntryNoGameSmokeReceiptV1 {
                policy_domain: PROXY_ENTRY_SMOKE_POLICY_DOMAIN,
                evidence_scope: "no_game_functional_smoke_only",
                entry_observation: "caller_supplied_proxy_local_instant",
                os_spawn_to_entry: "unknown",
                owner_id: run_scope
                    .owner_id
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect(),
                owner_epoch: run_scope.owner_epoch,
                run_request_id: run_scope.request_id,
                cleanup_request_id: cleanup_scope.request_id,
                role,
                generation: run_scope.generation,
                lock_sha256,
                source_scope: "pinned_system_v6_source_observation_only",
                source_observation: source_observation(source.system_v6_source()),
                entry_to_binding_ns: elapsed,
                local_startup_cap_ms: cap_ms,
                startup_remaining_at_binding_ns: remaining,
            };
            let window = OriginalProxyWindowV6 {
                bound_reference: inputs.run.requested_at().min(inputs.cleanup.requested_at()),
                startup_until,
                run_until,
                cleanup_until,
                cancelled: Arc::clone(cancelled),
                cleanup_cancelled: Arc::clone(cleanup_cancelled),
            };
            Ok((window, receipt))
        })();
        match result {
            Ok((window, receipt)) => {
                Ok((ProxyEntryNoGameSmokeBundleV1 { inputs, window }, receipt))
            }
            Err(error) => Err((error, self, inputs)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::pals::proxy_entry_smoke_grant_v1::ProxyEntrySmokePolicy;
    fn run() -> ProxyEntrySmokeScope {
        ProxyEntrySmokeScope {
            policy: ProxyEntrySmokePolicy::ProxyEntryFunctionalSmokeV1,
            owner_id: [1; 16],
            owner_epoch: 2,
            role: DeadlineRole::Baseline,
            generation: 3,
            request_id: 4,
            lock_sha256: [5; 32],
            phase: ProxyEntrySmokePhase::Run,
        }
    }
    #[test]
    fn two_phases_share_exact_nogame_owner_role_and_lock() {
        let run = run();
        let mut cleanup = run.clone();
        cleanup.phase = ProxyEntrySmokePhase::Cleanup;
        cleanup.request_id += 1;
        assert!(same_no_game_scope(&run, &cleanup));
        cleanup.role = DeadlineRole::Candidate;
        assert!(!same_no_game_scope(&run, &cleanup));
        cleanup.role = run.role;
        cleanup.generation += 1;
        assert!(!same_no_game_scope(&run, &cleanup));
        cleanup.generation = run.generation;
        cleanup.request_id = run.request_id;
        assert!(!same_no_game_scope(&run, &cleanup));
    }
}
