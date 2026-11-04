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
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest as _, Sha256};
use std::{os::unix::fs::MetadataExt, sync::Arc};

const SPEC_DOMAIN: &str = "rz-pals-proxy-entry-functional-smoke-spec/1";
const LOCK_DOMAIN: &str = "rz-pals-proxy-entry-functional-smoke-lock/1";
const RECEIPT_DOMAIN: &str = "rz-pals-proxy-entry-functional-smoke-receipt/1";
const SMOKE_SOURCE_DOMAIN: &str = "rz-pals-proxy-entry-functional-smoke-source/1";
pub(crate) const SMOKE_RECEIPT_FILE: &str = "pals-proxy-entry-functional-smoke.v1.json";

/// A separate diagnostic declaration. SHA values identify registered inputs,
/// not currently executing images or a model/provider result.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProxyEntryFunctionalSmokeSpecV1 {
    domain: String,
    pair_sha256: String,
    proxy_sidecar_sha256: String,
    proxy_binary_sha256: String,
    child_binary_sha256: String,
    startup_lock_sha256: String,
    system_runtime_lock_sha256: String,
    declared_runtime_kind: String,
    declared_role: String,
    process_generation: u64,
    owner_id_hex: String,
    owner_epoch: u64,
    startup_budget_ms: u64,
    arena_run_budget_ms: u64,
    arena_cleanup_budget_ms: u64,
    startup_reference: String,
    grant_policy: String,
    os_spawn_to_entry: String,
    eligible_for_paired_or_performance_acceptance: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProxyEntryFunctionalSmokeLockV1 {
    domain: String,
    spec: ProxyEntryFunctionalSmokeSpecV1,
    spec_sha256: String,
}

/// The claim is retained as a self-report only. Its elapsed duration uses the
/// reporting proxy's local monotonic clock and is not compared with an arena
/// `Instant`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ProxyEntryObservationV1 {
    Unknown,
    ProxySelfReported {
        entry_to_binding_ns: u64,
        clock_scope: String,
    },
}

/// Individual fields remain unknown until their own consumer provides actual
/// evidence; entry, ACK, process and physical completion are independent.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProxyEntryFunctionalSmokeReceiptV1 {
    domain: String,
    smoke_spec_sha256: String,
    entry: ProxyEntryObservationV1,
    os_spawn_observation: String,
    os_spawn_to_entry: String,
    role_game_authority: String,
    grant_policy_binding: String,
    cancel_ack: String,
    process_completion: String,
    physical_completion: String,
    running_proxy_source: String,
    selected_compiled_source_domain: String,
    selected_compiled_source_labels: Vec<String>,
    #[serde(default)]
    selected_compiled_source_sha256: Option<String>,
    eligible_for_paired_or_performance_acceptance: bool,
}

/// Same live arena execution allocation, retained separately from the lock's
/// serializable declaration. No `Instant` crosses the process boundary.
pub(crate) struct ProxyEntryFunctionalSmokeOwnerV1 {
    execution: Arc<PalsV6ExecutionStart>,
    identity: Arc<()>,
    cwd_device: u64,
    cwd_inode: u64,
    lock: ProxyEntryFunctionalSmokeLockV1,
    receipt: ProxyEntryFunctionalSmokeReceiptV1,
}

fn invalid(detail: &'static str) -> ArenaError {
    ArenaError::Invalid(detail.into())
}

const MAX_SMOKE_RECORD_BYTES: usize = 64 * 1024;

fn parse_record<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, ArenaError> {
    if bytes.len() > MAX_SMOKE_RECORD_BYTES {
        return Err(invalid("smoke record exceeds its byte limit"));
    }
    // Decode directly into deny_unknown_fields structs, not Value: serde's
    // typed decoder rejects duplicate struct fields instead of overwriting.
    serde_json::from_slice(bytes).map_err(|_| invalid("smoke record JSON is invalid"))
}

fn lowercase_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl ProxyEntryFunctionalSmokeSpecV1 {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, ArenaError> {
        let spec: Self = parse_record(bytes)?;
        spec.verify()?;
        Ok(spec)
    }

    fn verify(&self) -> Result<(), ArenaError> {
        let hashes = [
            &self.pair_sha256,
            &self.proxy_sidecar_sha256,
            &self.proxy_binary_sha256,
            &self.child_binary_sha256,
            &self.startup_lock_sha256,
            &self.system_runtime_lock_sha256,
        ];
        if self.domain != SPEC_DOMAIN
            || hashes.iter().any(|hash| !lowercase_hex(hash, 64))
            || !lowercase_hex(&self.owner_id_hex, 32)
            || self.owner_id_hex.bytes().all(|b| b == b'0')
            || self.owner_epoch == 0
            || self.process_generation == 0
            || self.startup_budget_ms == 0
            || self.startup_budget_ms > self.arena_run_budget_ms
            || self.arena_cleanup_budget_ms == 0
            || !matches!(self.declared_runtime_kind.as_str(), "cpu" | "system_cuda")
            || !matches!(self.declared_role.as_str(), "baseline" | "candidate")
            || self.startup_reference != "proxy_entry_self_observation"
            || self.grant_policy != "original_arena_run_and_cleanup_only"
            || self.os_spawn_to_entry != "unknown"
            || self.eligible_for_paired_or_performance_acceptance
        {
            return Err(invalid("smoke spec contract invalid"));
        }
        Ok(())
    }
}

impl ProxyEntryFunctionalSmokeLockV1 {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, ArenaError> {
        let lock: Self = parse_record(bytes)?;
        lock.verify()?;
        Ok(lock)
    }

    fn verify(&self) -> Result<(), ArenaError> {
        self.spec.verify()?;
        if self.domain != LOCK_DOMAIN || self.spec_sha256 != spec_digest(&self.spec)? {
            return Err(invalid("smoke lock domain or spec digest differs"));
        }
        Ok(())
    }
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
        ("arena_no_game_plan", include_bytes!("no_game_plan.rs")),
        (
            "arena_endpoint_snapshot",
            include_bytes!("endpoint_snapshot.rs"),
        ),
        (
            "arena_smoke_preparation",
            include_bytes!("smoke_preparation.rs"),
        ),
        ("arena_smoke_sockets", include_bytes!("no_game_sockets.rs")),
        (
            "arena_smoke_deadline",
            include_bytes!("smoke_deadline_control.rs"),
        ),
        (
            "arena_direct_smoke_launch",
            include_bytes!("direct_smoke_launch.rs"),
        ),
        ("arena_process", include_bytes!("../../process.rs")),
        (
            "smoke_grant_wire",
            include_bytes!("../../../../rz-contracts/src/pals/proxy_entry_smoke_grant_v1.rs"),
        ),
        (
            "proxy_smoke",
            include_bytes!("../../../../rz-uci/src/pals_system_proxy/proxy_entry_smoke_v1.rs"),
        ),
        (
            "proxy_no_game_smoke",
            include_bytes!(
                "../../../../rz-uci/src/pals_system_proxy/proxy_entry_no_game_smoke_v1.rs"
            ),
        ),
        (
            "proxy_smoke_grant_channel",
            include_bytes!(
                "../../../../rz-uci/src/pals_system_proxy/proxy_entry_smoke_channel_v1.rs"
            ),
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
    /// Parsing establishes only a well-formed receipt for this lock. It does
    /// not authenticate the live owner, source, peer or reported outcome.
    pub(crate) fn parse(
        bytes: &[u8],
        lock: &ProxyEntryFunctionalSmokeLockV1,
    ) -> Result<Self, ArenaError> {
        let receipt: Self = parse_record(bytes)?;
        receipt.verify(lock)?;
        Ok(receipt)
    }

    fn verify(&self, lock: &ProxyEntryFunctionalSmokeLockV1) -> Result<(), ArenaError> {
        lock.verify()?;
        if self.domain != RECEIPT_DOMAIN
            || self.smoke_spec_sha256 != lock.spec_sha256
            || self.eligible_for_paired_or_performance_acceptance
        {
            return Err(invalid("smoke receipt domain or lock differs"));
        }
        // Observation text, including source or physical status, is untrusted
        // data. The parser preserves it and confers no completion authority;
        // real lifecycle evidence is consumed by a separate owner-bound path.
        Ok(())
    }

    fn new(smoke_spec_sha256: String) -> Self {
        Self {
            domain: RECEIPT_DOMAIN.into(),
            smoke_spec_sha256,
            entry: ProxyEntryObservationV1::Unknown,
            os_spawn_observation: "unknown".into(),
            os_spawn_to_entry: "unknown".into(),
            role_game_authority: "unknown".into(),
            grant_policy_binding: "unknown".into(),
            cancel_ack: "unknown".into(),
            process_completion: "unknown".into(),
            physical_completion: "unknown".into(),
            running_proxy_source: "unknown".into(),
            selected_compiled_source_domain: SMOKE_SOURCE_DOMAIN.into(),
            selected_compiled_source_labels: smoke_sources()
                .iter()
                .map(|(label, _)| (*label).to_owned())
                .collect(),
            selected_compiled_source_sha256: Some(compiled_smoke_source_digest()),
            eligible_for_paired_or_performance_acceptance: false,
        }
    }

    fn record_entry_claim(&mut self, entry_to_binding_ns: u64) -> Result<(), ArenaError> {
        if !matches!(self.entry, ProxyEntryObservationV1::Unknown) {
            return Err(invalid("smoke entry claim invalid or already recorded"));
        }
        self.entry = ProxyEntryObservationV1::ProxySelfReported {
            entry_to_binding_ns,
            clock_scope: "proxy_local_monotonic_elapsed_entry_to_binding".into(),
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
        let cwd = owner
            .snapshot
            .cwd
            .metadata()
            .map_err(|_| invalid("smoke original cwd metadata unavailable"))?;
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
            domain: SPEC_DOMAIN.into(),
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
            declared_runtime_kind:
                match &endpoint.startup_lock.config.system_runtime_lock.runtime {
                    NativeRuntimeSourceV6::Cpu => "cpu",
                    NativeRuntimeSourceV6::SystemCuda(_) => "system_cuda",
                }
                .into(),
            declared_role: match role {
                NativeEngineRole::Baseline => "baseline",
                NativeEngineRole::Candidate => "candidate",
            }
            .into(),
            process_generation: endpoint.startup_lock.config.process_generation,
            owner_id_hex: owner_id.iter().map(|b| format!("{b:02x}")).collect(),
            owner_epoch,
            startup_budget_ms: endpoint.startup_lock.config.native_options.startup_ms,
            arena_run_budget_ms: run_ms,
            arena_cleanup_budget_ms: cleanup_ms,
            startup_reference: "proxy_entry_self_observation".into(),
            grant_policy: "original_arena_run_and_cleanup_only".into(),
            os_spawn_to_entry: "unknown".into(),
            eligible_for_paired_or_performance_acceptance: false,
        };
        spec.verify()?;
        let spec_sha256 = spec_digest(&spec)?;
        let receipt = ProxyEntryFunctionalSmokeReceiptV1::new(spec_sha256.clone());
        Ok(Self {
            execution: Arc::clone(execution),
            identity: Arc::new(()),
            cwd_device: cwd.dev(),
            cwd_inode: cwd.ino(),
            lock: ProxyEntryFunctionalSmokeLockV1 {
                domain: LOCK_DOMAIN.into(),
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

    pub(crate) fn identity(&self) -> &Arc<()> {
        &self.identity
    }

    /// Reconnect only to the same original execution allocation and cwd.
    /// This checks the declaration and snapshot, not a child/game identity.
    pub(crate) fn verify_original_binding(
        &self,
        owner: &NativeLaunchOwner<LockedPalsSystemArenaPairV6>,
        execution: &Arc<PalsV6ExecutionStart>,
        role: NativeEngineRole,
        owner_id: [u8; 16],
        owner_epoch: u64,
    ) -> Result<(), ArenaError> {
        self.lock.verify()?;
        let pair = owner.spec();
        let endpoint = pair.endpoint(role);
        let expected_role = match role {
            NativeEngineRole::Baseline => "baseline",
            NativeEngineRole::Candidate => "candidate",
        };
        let id: String = owner_id.iter().map(|b| format!("{b:02x}")).collect();
        let cwd = owner
            .snapshot
            .cwd
            .metadata()
            .map_err(|_| invalid("smoke original cwd metadata unavailable"))?;
        if !Arc::ptr_eq(&self.execution, execution)
            || !execution.matches_deadline_identity(owner_id, owner_epoch)
            || (cwd.dev(), cwd.ino()) != (self.cwd_device, self.cwd_inode)
            || pair.sha256() != self.lock.spec.pair_sha256
            || pair.proxy_inputs_sha256() != Some(self.lock.spec.proxy_sidecar_sha256.as_str())
            || self.lock.spec.declared_role != expected_role
            || self.lock.spec.owner_id_hex != id
            || self.lock.spec.owner_epoch != owner_epoch
            || self.lock.spec.process_generation != endpoint.startup_lock.config.process_generation
            || self.lock.spec.startup_lock_sha256 != endpoint.startup_lock.canonical_sha256
        {
            return Err(invalid("smoke original owner or endpoint differs"));
        }
        PalsV6EndpointSnapshot::from_original(owner, role)?;
        Ok(())
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

    fn fixture_spec() -> ProxyEntryFunctionalSmokeSpecV1 {
        let fixture = format!("{:x}", Sha256::digest(b"smoke input fixture"));
        ProxyEntryFunctionalSmokeSpecV1 {
            domain: SPEC_DOMAIN.into(),
            pair_sha256: fixture.clone(),
            proxy_sidecar_sha256: fixture.clone(),
            proxy_binary_sha256: fixture.clone(),
            child_binary_sha256: fixture.clone(),
            startup_lock_sha256: fixture.clone(),
            system_runtime_lock_sha256: fixture,
            declared_runtime_kind: "cpu".into(),
            declared_role: "baseline".into(),
            process_generation: 1,
            owner_id_hex: "01".repeat(16),
            owner_epoch: 1,
            startup_budget_ms: 1000,
            arena_run_budget_ms: 5000,
            arena_cleanup_budget_ms: 1000,
            startup_reference: "proxy_entry_self_observation".into(),
            grant_policy: "original_arena_run_and_cleanup_only".into(),
            os_spawn_to_entry: "unknown".into(),
            eligible_for_paired_or_performance_acceptance: false,
        }
    }

    #[test]
    fn smoke_lock_digest_covers_its_own_policy_and_inputs() {
        let spec = fixture_spec();
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
    fn smoke_parser_roundtrips_and_preserves_unknown_observations() {
        let spec = fixture_spec();
        let lock = ProxyEntryFunctionalSmokeLockV1 {
            domain: LOCK_DOMAIN.into(),
            spec_sha256: spec_digest(&spec).unwrap(),
            spec,
        };
        let spec_bytes = serde_json::to_vec(&lock.spec).unwrap();
        let lock_bytes = serde_json::to_vec(&lock).unwrap();
        let parsed_spec = ProxyEntryFunctionalSmokeSpecV1::parse(&spec_bytes).unwrap();
        let parsed_lock = ProxyEntryFunctionalSmokeLockV1::parse(&lock_bytes).unwrap();
        assert_eq!(parsed_spec, lock.spec);
        assert_eq!(parsed_lock.spec_sha256, lock.spec_sha256);
        let mut receipt = ProxyEntryFunctionalSmokeReceiptV1::new(lock.spec_sha256.clone());
        receipt.selected_compiled_source_domain = "unknown".into();
        receipt.selected_compiled_source_labels.clear();
        receipt.selected_compiled_source_sha256 = None;
        let bytes = serde_json::to_vec(&receipt).unwrap();
        let parsed = ProxyEntryFunctionalSmokeReceiptV1::parse(&bytes, &parsed_lock).unwrap();
        assert_eq!(parsed.entry, ProxyEntryObservationV1::Unknown);
        assert_eq!(parsed.physical_completion, "unknown");
        assert!(parsed.selected_compiled_source_sha256.is_none());
        receipt.record_entry_claim(123).unwrap();
        let bytes = serde_json::to_vec(&receipt).unwrap();
        let parsed = ProxyEntryFunctionalSmokeReceiptV1::parse(&bytes, &parsed_lock).unwrap();
        assert!(matches!(
            parsed.entry,
            ProxyEntryObservationV1::ProxySelfReported { .. }
        ));
        assert_eq!(parsed.role_game_authority, "unknown");
    }

    #[test]
    fn smoke_parser_rejects_duplicates_unknown_fields_and_tampering() {
        let spec = fixture_spec();
        let lock = ProxyEntryFunctionalSmokeLockV1 {
            domain: LOCK_DOMAIN.into(),
            spec_sha256: spec_digest(&spec).unwrap(),
            spec,
        };
        let mut changed = lock.clone();
        changed.spec.startup_budget_ms += 1;
        assert!(
            ProxyEntryFunctionalSmokeLockV1::parse(&serde_json::to_vec(&changed).unwrap()).is_err()
        );
        let raw = serde_json::to_string(&lock).unwrap();
        let duplicate = raw.replacen(
            "\"domain\":",
            &format!(
                "\"domain\":{},\"domain\":",
                serde_json::to_string(LOCK_DOMAIN).unwrap()
            ),
            1,
        );
        assert!(ProxyEntryFunctionalSmokeLockV1::parse(duplicate.as_bytes()).is_err());
        let unknown = format!("{{\"extra_field\":true,{}", &raw[1..]);
        assert!(ProxyEntryFunctionalSmokeLockV1::parse(unknown.as_bytes()).is_err());
        let mut receipt = ProxyEntryFunctionalSmokeReceiptV1::new(lock.spec_sha256.clone());
        let receipt_raw = serde_json::to_string(&receipt).unwrap();
        let duplicate_entry = receipt_raw.replacen(
            "\"status\":\"unknown\"",
            "\"status\":\"unknown\",\"status\":\"unknown\"",
            1,
        );
        assert_ne!(receipt_raw, duplicate_entry);
        assert!(
            ProxyEntryFunctionalSmokeReceiptV1::parse(duplicate_entry.as_bytes(), &lock).is_err()
        );
        receipt.physical_completion = "confirmed".into();
        let parsed = ProxyEntryFunctionalSmokeReceiptV1::parse(
            &serde_json::to_vec(&receipt).unwrap(),
            &lock,
        )
        .unwrap();
        // This is a preserved untrusted claim, never a verified physical fence.
        assert_eq!(parsed.physical_completion, "confirmed");
        assert!(!parsed.eligible_for_paired_or_performance_acceptance);
        receipt.physical_completion = "unknown".into();
        receipt.smoke_spec_sha256 = format!("{:x}", Sha256::digest(b"another smoke spec"));
        assert!(
            ProxyEntryFunctionalSmokeReceiptV1::parse(
                &serde_json::to_vec(&receipt).unwrap(),
                &lock
            )
            .is_err()
        );
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
