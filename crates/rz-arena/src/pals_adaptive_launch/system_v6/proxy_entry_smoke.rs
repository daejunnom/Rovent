//! Separate proxy-entry functional smoke declaration and observation.
//!
//! This never alters the V6 pair/lock/receipt, reconstructs a process clock,
//! admits a peer or establishes any GPU/physical completion. A proxy entry
//! claim is not the operating system's process-spawn observation.

use super::{
    LockedPalsSystemArenaPairV6, PalsV6ExecutionStart, endpoint_snapshot::PalsV6EndpointSnapshot,
};
use crate::{ArenaError, NativeLaunchOwner};
use rz_experiments::{NativeEngineRole, NativeRuntimeSourceV6};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::sync::Arc;

const SPEC_DOMAIN: &str = "rz-pals-proxy-entry-functional-smoke-spec/1";
const LOCK_DOMAIN: &str = "rz-pals-proxy-entry-functional-smoke-lock/1";
const RECEIPT_DOMAIN: &str = "rz-pals-proxy-entry-functional-smoke-receipt/1";
const SMOKE_SOURCE_DOMAIN: &str = "rz-pals-proxy-entry-functional-smoke-source/1";
pub(crate) const SMOKE_RECEIPT_FILE: &str = "pals-proxy-entry-functional-smoke.v1.json";

/// A separate diagnostic declaration. SHA values identify registered inputs,
/// not currently executing images or a model/provider result.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct ProxyEntryFunctionalSmokeSpecV1 {
    domain: &'static str,
    pair_sha256: String,
    proxy_sidecar_sha256: String,
    proxy_binary_sha256: String,
    child_binary_sha256: String,
    startup_lock_sha256: String,
    system_runtime_lock_sha256: String,
    declared_runtime_kind: &'static str,
    declared_role: &'static str,
    process_generation: u64,
    owner_id_hex: String,
    owner_epoch: u64,
    startup_budget_ms: u64,
    arena_run_budget_ms: u64,
    arena_cleanup_budget_ms: u64,
    startup_reference: &'static str,
    grant_policy: &'static str,
    os_spawn_to_entry: &'static str,
    eligible_for_paired_or_performance_acceptance: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProxyEntryFunctionalSmokeLockV1 {
    domain: &'static str,
    spec: ProxyEntryFunctionalSmokeSpecV1,
    spec_sha256: String,
}

/// The claim is retained as a self-report only. Its elapsed duration uses the
/// reporting proxy's local monotonic clock and is not compared with an arena
/// `Instant`.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum ProxyEntryObservationV1 {
    Unknown,
    ProxySelfReported {
        entry_to_binding_ns: u64,
        clock_scope: &'static str,
    },
}

/// Individual fields remain unknown until their own consumer provides actual
/// evidence; entry, ACK, process and physical completion are independent.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProxyEntryFunctionalSmokeReceiptV1 {
    domain: &'static str,
    smoke_spec_sha256: String,
    entry: ProxyEntryObservationV1,
    os_spawn_observation: &'static str,
    os_spawn_to_entry: &'static str,
    role_game_authority: &'static str,
    grant_policy_binding: &'static str,
    cancel_ack: &'static str,
    process_completion: &'static str,
    physical_completion: &'static str,
    running_proxy_source: &'static str,
    selected_compiled_source_domain: &'static str,
    selected_compiled_source_labels: Vec<&'static str>,
    selected_compiled_source_sha256: String,
    eligible_for_paired_or_performance_acceptance: bool,
}

/// Same live arena execution allocation, retained separately from the lock's
/// serializable declaration. No `Instant` crosses the process boundary.
pub(crate) struct ProxyEntryFunctionalSmokeOwnerV1 {
    execution: Arc<PalsV6ExecutionStart>,
    lock: ProxyEntryFunctionalSmokeLockV1,
    receipt: ProxyEntryFunctionalSmokeReceiptV1,
}

fn invalid(detail: &'static str) -> ArenaError {
    ArenaError::Invalid(detail.into())
}

fn spec_digest(spec: &ProxyEntryFunctionalSmokeSpecV1) -> Result<String, ArenaError> {
    let bytes = serde_json::to_vec(spec).map_err(|_| invalid("smoke spec serialization failed"))?;
    let mut digest = Sha256::new();
    digest.update(LOCK_DOMAIN.as_bytes());
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}

/// A framed source-byte observation, never a gate or proof of a built ELF.
fn smoke_sources() -> &'static [(&'static str, &'static [u8])] {
    &[
        ("arena_smoke", include_bytes!("proxy_entry_smoke.rs")),
        (
            "proxy_smoke",
            include_bytes!("../../../../rz-uci/src/pals_system_proxy/proxy_entry_smoke_v1.rs"),
        ),
        (
            "proxy_module",
            include_bytes!("../../../../rz-uci/src/pals_system_proxy.rs"),
        ),
        (
            "proxy_binary_entry",
            include_bytes!("../../../../rz-uci/src/bin/pals-adaptive.rs"),
        ),
        (
            "proxy_bootstrap",
            include_bytes!("../../../../rz-uci/src/pals_adaptive_bootstrap.rs"),
        ),
    ]
}

fn compiled_smoke_source_digest() -> String {
    let mut hash = Sha256::new();
    hash.update(SMOKE_SOURCE_DOMAIN.as_bytes());
    for (label, bytes) in smoke_sources() {
        hash.update((label.len() as u64).to_le_bytes());
        hash.update(label.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    format!("{:x}", hash.finalize())
}

impl ProxyEntryFunctionalSmokeReceiptV1 {
    fn new(smoke_spec_sha256: String) -> Self {
        Self {
            domain: RECEIPT_DOMAIN,
            smoke_spec_sha256,
            entry: ProxyEntryObservationV1::Unknown,
            os_spawn_observation: "unknown",
            os_spawn_to_entry: "unknown",
            role_game_authority: "unknown",
            grant_policy_binding: "unknown",
            cancel_ack: "unknown",
            process_completion: "unknown",
            physical_completion: "unknown",
            running_proxy_source: "unknown",
            selected_compiled_source_domain: SMOKE_SOURCE_DOMAIN,
            selected_compiled_source_labels: smoke_sources()
                .iter()
                .map(|(label, _)| *label)
                .collect(),
            selected_compiled_source_sha256: compiled_smoke_source_digest(),
            eligible_for_paired_or_performance_acceptance: false,
        }
    }

    fn record_entry_claim(&mut self, entry_to_binding_ns: u64) -> Result<(), ArenaError> {
        if !matches!(self.entry, ProxyEntryObservationV1::Unknown) {
            return Err(invalid("smoke entry claim invalid or already recorded"));
        }
        self.entry = ProxyEntryObservationV1::ProxySelfReported {
            entry_to_binding_ns,
            clock_scope: "proxy_local_monotonic_elapsed_entry_to_binding",
        };
        Ok(())
    }
}

impl ProxyEntryFunctionalSmokeOwnerV1 {
    pub(crate) fn bind_original(
        owner: &NativeLaunchOwner<LockedPalsSystemArenaPairV6>,
        execution: &Arc<PalsV6ExecutionStart>,
        role: NativeEngineRole,
        owner_id: [u8; 16],
        owner_epoch: u64,
    ) -> Result<Self, ArenaError> {
        let pair = owner.spec();
        execution.require_running()?;
        if pair.sha256() != execution.pair_sha256()
            || !execution.matches_deadline_identity(owner_id, owner_epoch)
        {
            return Err(invalid(
                "smoke requires the registered original arena execution",
            ));
        }
        let sidecar = pair
            .proxy_inputs()
            .ok_or_else(|| invalid("smoke proxy sidecar missing"))?;
        sidecar.validate_for_pair(pair)?;
        PalsV6EndpointSnapshot::from_original(owner, role)?;
        let sidecar_sha256 = pair
            .proxy_inputs_sha256()
            .ok_or_else(|| invalid("smoke registered proxy sidecar digest missing"))?;
        let endpoint = pair.endpoint(role);
        let window = &endpoint.startup_lock.config.window;
        let run_duration = execution
            .run_until()
            .checked_duration_since(execution.started())
            .ok_or_else(|| invalid("smoke original run window invalid"))?;
        let cleanup_duration = execution
            .cleanup_until()
            .checked_duration_since(execution.run_until())
            .ok_or_else(|| invalid("smoke original cleanup window invalid"))?;
        let run_ms = u64::try_from(run_duration.as_millis())
            .map_err(|_| invalid("smoke run window too large"))?;
        let cleanup_ms = u64::try_from(cleanup_duration.as_millis())
            .map_err(|_| invalid("smoke cleanup window too large"))?;
        if run_ms != window.wall_time_max_ms || cleanup_ms != window.cleanup_max_ms {
            return Err(invalid(
                "smoke declaration would change the original arena window",
            ));
        }
        let spec = ProxyEntryFunctionalSmokeSpecV1 {
            domain: SPEC_DOMAIN,
            pair_sha256: pair.sha256().to_owned(),
            proxy_sidecar_sha256: sidecar_sha256.to_owned(),
            proxy_binary_sha256: sidecar.proxy_binary.sha256.clone(),
            child_binary_sha256: endpoint.startup_lock.config.endpoint.binary.sha256.clone(),
            startup_lock_sha256: endpoint.startup_lock.canonical_sha256.clone(),
            system_runtime_lock_sha256: endpoint
                .startup_lock
                .config
                .system_runtime_lock
                .canonical_sha256
                .clone(),
            declared_runtime_kind: match &endpoint.startup_lock.config.system_runtime_lock.runtime {
                NativeRuntimeSourceV6::Cpu => "cpu",
                NativeRuntimeSourceV6::SystemCuda(_) => "system_cuda",
            },
            declared_role: match role {
                NativeEngineRole::Baseline => "baseline",
                NativeEngineRole::Candidate => "candidate",
            },
            process_generation: endpoint.startup_lock.config.process_generation,
            owner_id_hex: owner_id.iter().map(|b| format!("{b:02x}")).collect(),
            owner_epoch,
            startup_budget_ms: endpoint.startup_lock.config.native_options.startup_ms,
            arena_run_budget_ms: run_ms,
            arena_cleanup_budget_ms: cleanup_ms,
            startup_reference: "proxy_entry_self_observation",
            grant_policy: "original_arena_run_and_cleanup_only",
            os_spawn_to_entry: "unknown",
            eligible_for_paired_or_performance_acceptance: false,
        };
        let spec_sha256 = spec_digest(&spec)?;
        let receipt = ProxyEntryFunctionalSmokeReceiptV1::new(spec_sha256.clone());
        Ok(Self {
            execution: Arc::clone(execution),
            lock: ProxyEntryFunctionalSmokeLockV1 {
                domain: LOCK_DOMAIN,
                spec,
                spec_sha256,
            },
            receipt,
        })
    }

    pub(crate) fn lock(&self) -> &ProxyEntryFunctionalSmokeLockV1 {
        &self.lock
    }
    pub(crate) fn receipt(&self) -> &ProxyEntryFunctionalSmokeReceiptV1 {
        &self.receipt
    }
    pub(crate) fn execution(&self) -> &Arc<PalsV6ExecutionStart> {
        &self.execution
    }

    /// This records an unverified claim, never authoritative process evidence.
    /// Keep the first claim and reject later replacements, even exact copies.
    pub(crate) fn record_entry_claim(
        &mut self,
        entry_to_binding_ns: u64,
    ) -> Result<(), ArenaError> {
        self.receipt.record_entry_claim(entry_to_binding_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_lock_digest_covers_its_own_policy_and_inputs() {
        let fixture = format!("{:x}", Sha256::digest(b"smoke input fixture"));
        let spec = ProxyEntryFunctionalSmokeSpecV1 {
            domain: SPEC_DOMAIN,
            pair_sha256: fixture.clone(),
            proxy_sidecar_sha256: fixture.clone(),
            proxy_binary_sha256: fixture.clone(),
            child_binary_sha256: fixture.clone(),
            startup_lock_sha256: fixture.clone(),
            system_runtime_lock_sha256: fixture,
            declared_runtime_kind: "cpu",
            declared_role: "baseline",
            process_generation: 1,
            owner_id_hex: "01".repeat(16),
            owner_epoch: 1,
            startup_budget_ms: 1000,
            arena_run_budget_ms: 5000,
            arena_cleanup_budget_ms: 1000,
            startup_reference: "proxy_entry_self_observation",
            grant_policy: "original_arena_run_and_cleanup_only",
            os_spawn_to_entry: "unknown",
            eligible_for_paired_or_performance_acceptance: false,
        };
        let original = spec_digest(&spec).unwrap();
        assert_eq!(original, spec_digest(&spec).unwrap());
        let mut changed = spec.clone();
        changed.startup_budget_ms += 1;
        assert_ne!(original, spec_digest(&changed).unwrap());
        changed = spec;
        changed.proxy_binary_sha256 = format!("{:x}", Sha256::digest(b"other binary"));
        assert_ne!(original, spec_digest(&changed).unwrap());
    }

    #[test]
    fn smoke_receipt_keeps_entry_separate_from_unobserved_lifecycle() {
        let spec_id = format!("{:x}", Sha256::digest(b"separate smoke fixture"));
        let mut receipt = ProxyEntryFunctionalSmokeReceiptV1::new(spec_id);
        assert_eq!(receipt.entry, ProxyEntryObservationV1::Unknown);
        receipt.record_entry_claim(10).unwrap();
        assert!(receipt.record_entry_claim(10).is_err());
        assert!(matches!(
            receipt.entry,
            ProxyEntryObservationV1::ProxySelfReported { .. }
        ));
        assert_eq!(receipt.os_spawn_to_entry, "unknown");
        assert_eq!(receipt.role_game_authority, "unknown");
        assert_eq!(receipt.grant_policy_binding, "unknown");
        assert_eq!(receipt.cancel_ack, "unknown");
        assert_eq!(receipt.process_completion, "unknown");
        assert_eq!(receipt.physical_completion, "unknown");
        assert!(!receipt.eligible_for_paired_or_performance_acceptance);
    }
}
